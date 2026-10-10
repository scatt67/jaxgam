"""Finite-precision completion checks for pinned streamed NB theta."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

import jaxgam.execution.reml as reml_execution
from jaxgam.execution.reml import StreamREMLControl

_ACCEPTED = np.array([-0.6931471805599453, 0.7061978415958204])
_STATIONARY = np.array([-0.6931471805599453, 0.7061978383114913])
_WORSE_STATIONARY = np.array([-0.6931471805599453, 0.706197836])
_NONSTATIONARY = np.array([-0.6931471805599453, 0.7061978415932526])
_SCORE = 388.38788334485787
_BOUNDS = [(_ACCEPTED[0], _ACCEPTED[0]), (None, None)]
_CONTROL = StreamREMLControl(gtol=1e-9, ftol=0.0)


def _run_trace(
    monkeypatch,
    *,
    stationary_score_ulps: int = 1,
    stationary_gradient: float = -3e-13,
    stationary_lineage: str = "source",
    stationary_params: np.ndarray = _STATIONARY,
    inner_converged: bool = True,
    later_worse_stationary: bool = False,
):
    """Model the Linux score plateau with one immutable trial per evaluation."""
    calls = Counter()

    def evaluate(params, _warm_start):
        params = np.array(params, copy=True)
        stationary = np.array_equal(params, stationary_params)
        calls["stationary" if stationary else "other"] += 1
        worse_stationary = np.array_equal(params, _WORSE_STATIONARY)
        score_ulps = (
            4 if worse_stationary else stationary_score_ulps if stationary else 0
        )
        score = _SCORE + score_ulps * np.spacing(_SCORE)
        gradient = (
            0.0 if worse_stationary else stationary_gradient if stationary else 8.13e-8
        )
        lineage = stationary_lineage if stationary else "source"
        return SimpleNamespace(
            params=params,
            score=score,
            gradient=np.array([-2.22, gradient]),
            source_scans=3,
            batches_scanned=9,
            source_fingerprint=lineage,
            basis_fingerprint="basis",
            family_name="Negative Binomial",
            link_name="identity",
            fit_result=SimpleNamespace(
                state=SimpleNamespace(
                    converged=inner_converged,
                    line_search_failed=False,
                    stationarity=1e-14,
                )
            ),
        )

    source = SimpleNamespace(fingerprint=lambda: "source")
    objective = reml_execution._AcceptedParameterizedTrialObjective(
        SimpleNamespace(source=source),
        _ACCEPTED,
        _CONTROL,
        evaluate,
        objective_name="NB streamed REML",
        source_name="NB streamed REML",
        trial_name="NB streamed",
    )

    def plateau(observe, _params, **_kwargs):
        observe(_ACCEPTED)
        observe(stationary_params)
        if later_worse_stationary:
            observe(_WORSE_STATIONARY)
        observe(_NONSTATIONARY)
        return SimpleNamespace(
            x=_ACCEPTED.copy(),
            success=True,
            status=0,
            nit=4,
            message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
        )

    monkeypatch.setattr(reml_execution, "minimize", plateau)
    return reml_execution._run_parameterized_stream_reml(
        objective,
        _ACCEPTED,
        _BOUNDS,
        _CONTROL,
        roundoff_stationary_completion=True,
        completion_inner_tolerance=1e-11,
    ), calls


def test_pinned_theta_roundoff_completion_reevaluates_stationary_trial(
    monkeypatch,
) -> None:
    result, calls = _run_trace(monkeypatch)
    assert result.converged
    np.testing.assert_array_equal(result.trial.params, _STATIONARY)
    assert result.projected_gradient_inf < _CONTROL.gtol
    assert result.status == 0
    assert "RELATIVE REDUCTION" in result.message
    assert "one score ULP" in result.message
    assert calls["stationary"] == 1
    assert result.n_evaluations == 3
    assert result.cumulative_source_scans == 9
    assert result.cumulative_batches_scanned == 27


@pytest.mark.parametrize(
    ("score_ulps", "gradient"),
    [(4, -3e-13), (-4, -3e-13), (1, 8.13e-8)],
    ids=["genuinely-worse", "materially-lower", "ulp-tie-with-large-gradient"],
)
def test_pinned_theta_roundoff_completion_rejects_unqualified_trial(
    monkeypatch, score_ulps: int, gradient: float
) -> None:
    result, calls = _run_trace(
        monkeypatch,
        stationary_score_ulps=score_ulps,
        stationary_gradient=gradient,
    )
    assert not result.converged
    np.testing.assert_array_equal(result.trial.params, _ACCEPTED)
    assert result.projected_gradient_inf > _CONTROL.gtol
    assert "exceeds gtol" in result.message
    assert calls["stationary"] == 1


def test_pinned_theta_roundoff_completion_rejects_changed_lineage(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="changed source/family lineage"):
        _run_trace(monkeypatch, stationary_lineage="different")


def test_pinned_theta_roundoff_completion_rejects_changed_pin(monkeypatch) -> None:
    moved_pin = _STATIONARY.copy()
    moved_pin[0] += 1e-10
    with pytest.raises(RuntimeError, match="violated a parameter bound"):
        _run_trace(monkeypatch, stationary_params=moved_pin)


def test_pinned_theta_roundoff_completion_requires_inner_convergence(
    monkeypatch,
) -> None:
    result, _ = _run_trace(monkeypatch, inner_converged=False)
    assert not result.converged
    np.testing.assert_array_equal(result.trial.params, _ACCEPTED)
    assert "exceeds gtol" in result.message


def test_pinned_theta_roundoff_completion_keeps_useful_tie_over_worse_zero_gradient(
    monkeypatch,
) -> None:
    result, calls = _run_trace(monkeypatch, later_worse_stationary=True)
    assert result.converged
    np.testing.assert_array_equal(result.trial.params, _STATIONARY)
    assert calls["stationary"] == 1
    assert result.n_evaluations == 4
