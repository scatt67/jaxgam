"""Finite-precision completion checks for pinned streamed NB theta."""

from collections import Counter
from types import SimpleNamespace
from weakref import ref

import numpy as np
import pytest

import jaxgam.execution.nb_reml as nb_reml_execution
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
    local_comparison=None,
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
        local_score_change=local_comparison,
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


def _local_change(change, error=1e-17, raw=None):
    if raw is None:
        raw = 2 * np.spacing(_SCORE)

    def compare(_objective, _accepted, _candidate, _inner_tolerance):
        return reml_execution._LocalScoreChange(
            change=change,
            error_estimate=error,
            raw_change=raw,
            roundoff_bound=4 * np.spacing(_SCORE),
            linearity_error=0.0,
        )

    return compare


def test_pinned_theta_local_comparison_keeps_raw_source_score(monkeypatch):
    result, _ = _run_trace(
        monkeypatch,
        stationary_score_ulps=2,
        local_comparison=_local_change(-1e-16),
    )
    assert result.converged
    assert float(result.trial.score) == _SCORE + 2 * np.spacing(_SCORE)
    assert result.accepted_score_history[-1] == float(result.trial.score)
    assert "empirical local-gradient comparison" in result.message
    assert "raw source score change" in result.message
    assert result.projected_gradient_inf <= _CONTROL.gtol


def test_pinned_theta_local_comparison_does_not_bypass_invalid_inner_or_bounds(
    monkeypatch,
):
    calls = Counter()

    def compare(*_args):
        calls["comparison"] += 1
        return _local_change(-1e-16)(*_args)

    result, _ = _run_trace(
        monkeypatch,
        stationary_score_ulps=2,
        inner_converged=False,
        local_comparison=compare,
    )
    assert not result.converged
    assert calls["comparison"] == 0
    moved_pin = _STATIONARY.copy()
    moved_pin[0] += 1e-10
    with pytest.raises(RuntimeError, match="violated a parameter bound"):
        _run_trace(
            monkeypatch,
            stationary_score_ulps=2,
            stationary_params=moved_pin,
            local_comparison=compare,
        )
    assert calls["comparison"] == 0


@pytest.mark.parametrize(
    ("change", "error", "raw"),
    [
        (1e-16, 1e-17, 2 * np.spacing(_SCORE)),
        (-1e-18, 1e-17, 2 * np.spacing(_SCORE)),
        (-1e-16, 1e-17, 10 * np.spacing(_SCORE)),
        (-np.inf, 1e-17, 2 * np.spacing(_SCORE)),
        (-1e-16, -1e-17, 2 * np.spacing(_SCORE)),
        (-1e-16, 1e-17, np.nan),
    ],
    ids=[
        "ascending",
        "error-dominated",
        "inconsistent-raw",
        "nonfinite-change",
        "negative-error",
        "nonfinite-raw",
    ],
)
def test_pinned_theta_local_comparison_rejects_unsafe_change(
    monkeypatch, change, error, raw
):
    result, _ = _run_trace(
        monkeypatch,
        stationary_score_ulps=2,
        local_comparison=_local_change(change, error, raw),
    )
    assert not result.converged
    np.testing.assert_array_equal(result.trial.params, _ACCEPTED)
    assert "stationary completion rejected" in result.message


def _local_probe_case(
    gradient_at,
    *,
    invalid_inner=False,
    changed_source=False,
    source_residual_at=None,
    observed_residual_at=None,
    residual_value=1e-8,
):
    """Scalar source replay with observable live trial/state/factor ownership."""
    live = {name: [] for name in ("trial", "state", "factor")}
    base = np.array([-np.log(2.0), 0.7])
    end = base.copy()
    end[-1] -= 3e-9

    class Trial:
        pass

    class State:
        pass

    class Factor:
        def logdet_hessian(self):
            return 5.0

    def trial(params, gradient, *, source="source", role="probe"):
        value = Trial()
        value.params = np.array(params, copy=True)
        value.score = 400.0 + (2 * np.spacing(400.0) if source == "source" else 0)
        value.gradient = np.array([0.0, gradient])
        value.source_scans = 4
        value.batches_scanned = 44
        value.source_fingerprint = source
        value.basis_fingerprint = "basis"
        value.family_name = "nb"
        value.link_name = "sqrt"
        value.source_factor_residual = (
            residual_value if source_residual_at == role else 1e-14
        )
        value.observed_factor_residual = (
            residual_value if observed_residual_at == role else 1e-14
        )
        state = State()
        factor = Factor()
        state.converged = not invalid_inner
        state.line_search_failed = False
        state.stationarity = 1e-14
        state.saturated_loglik = -280.0
        state.coefficient_factor = factor
        value.fit_result = SimpleNamespace(
            score_penalized_deviance=190.0,
            state=state,
        )
        live["trial"].append(ref(value))
        live["state"].append(ref(state))
        live["factor"].append(ref(factor))
        return value

    accepted = trial(base, gradient_at(0.0), role="accepted")
    accepted.score = 400.0
    candidate = trial(end, gradient_at(1.0), role="candidate")

    def evaluate(params, _warm_start):
        for ownership in live.values():
            assert sum(item() is not None for item in ownership) == 2
        fraction = (params[-1] - base[-1]) / (end[-1] - base[-1])
        return trial(
            params,
            gradient_at(fraction),
            source="different" if changed_source else "source",
        )

    objective = SimpleNamespace(
        _stream=SimpleNamespace(source=SimpleNamespace(fingerprint=lambda: "source")),
        _evaluate=evaluate,
        n_evaluations=0,
        cumulative_source_scans=0,
        cumulative_batches_scanned=0,
    )
    return objective, accepted, candidate, live


def test_nb_local_score_comparison_streams_three_probes_with_bounded_retention():
    objective, accepted, candidate, live = _local_probe_case(
        lambda fraction: 8e-8 * (1.0 - fraction)
    )
    comparison = nb_reml_execution._nb_local_stationary_score_change(
        objective, accepted, candidate, 1e-11
    )
    assert comparison is not None
    assert comparison.change < 0
    assert comparison.change + comparison.error_estimate < 0
    assert abs(comparison.change) + comparison.error_estimate < np.spacing(400.0)
    assert comparison.raw_change == 2 * np.spacing(400.0)
    assert objective.n_evaluations == 3
    assert objective.cumulative_source_scans == 12
    assert objective.cumulative_batches_scanned == 132
    for ownership in live.values():
        assert sum(item() is not None for item in ownership) == 2


@pytest.mark.parametrize(
    ("source_at", "observed_at", "residual", "expected_evaluations"),
    [
        ("accepted", None, 2e-11, 0),
        (None, "candidate", 2e-11, 0),
        ("probe", None, 2e-11, 1),
        (None, "probe", 2e-11, 1),
        ("probe", None, np.nan, 1),
    ],
    ids=[
        "source-endpoint",
        "observed-endpoint",
        "source-probe",
        "observed-probe",
        "nonfinite-probe",
    ],
)
def test_nb_local_score_comparison_rejects_unresolved_factor_residuals(
    source_at, observed_at, residual, expected_evaluations
):
    objective, accepted, candidate, _ = _local_probe_case(
        lambda fraction: 8e-8 * (1.0 - fraction),
        source_residual_at=source_at,
        observed_residual_at=observed_at,
        residual_value=residual,
    )
    assert (
        nb_reml_execution._nb_local_stationary_score_change(
            objective, accepted, candidate, 1e-11
        )
        is None
    )
    assert objective.n_evaluations == expected_evaluations
    assert objective.cumulative_source_scans == 4 * expected_evaluations


@pytest.mark.parametrize(
    ("gradient_at", "invalid_inner", "changed_source", "error"),
    [
        (lambda f: 8e-8 * (1 - f) + 1e-8 * np.sin(2 * np.pi * f), False, False, None),
        (lambda f: 8e-8 * (1 - f), True, False, None),
        (
            lambda f: np.nan if np.isclose(f, 0.25) else 8e-8 * (1 - f),
            False,
            False,
            None,
        ),
        (lambda f: 8e-8 * (1 - f), False, True, RuntimeError),
    ],
    ids=["nonlinear-path", "invalid-inner", "nonfinite-gradient", "changed-source"],
)
def test_nb_local_score_comparison_rejects_invalid_path(
    gradient_at, invalid_inner, changed_source, error
):
    objective, accepted, candidate, _ = _local_probe_case(
        gradient_at, invalid_inner=invalid_inner, changed_source=changed_source
    )
    if error is not None:
        with pytest.raises(error, match="source/family lineage"):
            nb_reml_execution._nb_local_stationary_score_change(
                objective, accepted, candidate, 1e-11
            )
    else:
        assert (
            nb_reml_execution._nb_local_stationary_score_change(
                objective, accepted, candidate, 1e-11
            )
            is None
        )
