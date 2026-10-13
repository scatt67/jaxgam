"""Finite-precision completion checks for pinned streamed NB theta."""

import re
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from weakref import ref

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

import jaxgam.execution.nb_reml as nb_reml_execution
import jaxgam.execution.reml as reml_execution
from jaxgam import GAM, FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.reml import StreamREMLControl
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.negative_binomial import NegativeBinomial
from tests.helpers import _AssertCollector, nb_optimizer_case_data, r_available
from tests.r_bridge import RBridge
from tests.test_execution.test_nb_reml import _LINKS
from tests.test_execution.test_nb_reml_optimizer import (
    _OPTIMIZER_MAXIMUM_BYTES,
    _OPTIMIZER_PIRLS_CONTROL,
    _optimizer_case,
    _pinned_selected_reference,
)
from tests.tolerances import STRICT

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


# ---------------------------------------------------------------------------
# Pinned log-theta secant Newton completion after an L-BFGS-B roundoff stop
# ---------------------------------------------------------------------------

# A smooth, strictly convex non-quadratic fixed-rho profile in t = log theta:
# f(t) = C + K (expm1(d) - d), g(t) = K expm1(d), f''(t) = K exp(d), d = t - t*.
_T_STAR = 0.7061978383114954
_K = 24.78
_PROFILE_SCORE = 388.0
_PINNED_RHO = np.array([-0.6931471805599453])
_INNER_TOLERANCE = 1e-11
_PUBLIC_CONTROL = StreamREMLControl(gtol=1e-9, ftol=0.0)


class _ProfileTrial(SimpleNamespace):
    """Weak-referenceable immutable-by-convention synthetic exact trial."""


def _profile_trial(params, *, score=None, gradient=None, **overrides):
    params = np.array(params, dtype=np.float64, copy=True)
    d = float(params[-1]) - _T_STAR
    exact_score = _PROFILE_SCORE + _K * (np.expm1(d) - d)
    exact_gradient = _K * np.expm1(d)
    state = SimpleNamespace(
        converged=overrides.pop("converged", True),
        line_search_failed=False,
        stationarity=1e-14,
        saturated_loglik=-280.0,
        coefficient_factor=SimpleNamespace(logdet_hessian=lambda: 19.0),
    )
    trial = _ProfileTrial(
        params=params,
        score=np.float64(exact_score if score is None else score),
        gradient=np.array([0.3, exact_gradient if gradient is None else gradient]),
        source_scans=2,
        batches_scanned=5,
        source_fingerprint="source",
        basis_fingerprint="basis",
        family_name="nb",
        link_name="IdentityLink",
        source_factor_residual=1e-15,
        observed_factor_residual=1e-15,
        workspace=SimpleNamespace(required_bytes=256),
        fit_result=SimpleNamespace(
            log_theta=params[-1:].copy(),
            score_penalized_deviance=190.0,
            state=state,
        ),
    )
    for name, value in overrides.items():
        setattr(trial, name, value)
    return trial


class _ThetaProfile:
    """Exact profile evaluator recording requests and live trial ownership."""

    def __init__(self, scripted=None):
        self.requests = []
        self.scripted = dict(scripted or {})
        self.live = []
        self.peak_live = 0

    def __call__(self, _stream, _family, params, **_kwargs):
        alive = sum(item() is not None for item in self.live)
        self.peak_live = max(self.peak_live, alive)
        params = np.asarray(params, dtype=np.float64)
        self.requests.append(params.copy())
        override = self.scripted.get(len(self.requests), {})
        trial = (
            override(params)
            if callable(override)
            else _profile_trial(params, **override)
        )
        self.live.append(ref(trial))
        return trial


def _theta_params(d):
    return np.r_[_PINNED_RHO, _T_STAR + d]


def _scripted_lbfgsb(accepted_offsets, *, message, status, success=None, probes=()):
    """Accept exact iterates through the callback, probe, then stop."""

    def minimize_call(fun, x0, **kwargs):
        assert kwargs["bounds"][0] == (_PINNED_RHO[0], _PINNED_RHO[0])
        fun(np.asarray(x0))
        x = np.asarray(x0)
        for offset in accepted_offsets:
            x = _theta_params(offset)
            fun(x)
            kwargs["callback"](OptimizeResult(x=x.copy(), fun=0.0))
        for offset in probes:
            fun(_theta_params(offset))
        return SimpleNamespace(
            x=x.copy(),
            success=(status == 0) if success is None else success,
            status=status,
            message=message,
            nit=len(accepted_offsets),
        )

    return minimize_call


def _theta_stream():
    fitting = SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1))
    return SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
        prepared=SimpleNamespace(fitting=fitting, n_coef=2),
    )


def _optimize_profile(
    monkeypatch,
    profile,
    *,
    start=0.3,
    accepted=(0.02, 3e-4),
    message="ABNORMAL: ",
    status=2,
    probes=(3e-4 - 1e-12,),
):
    monkeypatch.setattr(nb_reml_execution, "evaluate_nb_stream_reml", profile)
    monkeypatch.setattr(
        reml_execution,
        "minimize",
        _scripted_lbfgsb(accepted, message=message, status=status, probes=probes),
    )
    return nb_reml_execution.optimize_nb_stream_reml(
        _theta_stream(),
        NegativeBinomial(theta=2.7, fixed=False, link="identity"),
        _theta_params(start),
        maximum_bytes=1_000_000,
        pin_lambda=True,
        pirls_control=StreamPIRLSControl(solver_policy="qr", tol=_INNER_TOLERANCE),
        control=_PUBLIC_CONTROL,
    )


@pytest.mark.parametrize(
    ("message", "status"),
    [
        ("ABNORMAL: ", 2),
        ("ABNORMAL_TERMINATION_IN_LNSRCH", 2),
        ("CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH", 0),
    ],
    ids=["abnormal", "legacy-abnormal", "relative-reduction"],
)
def test_theta_secant_completion_finishes_roundoff_stops(monkeypatch, message, status):
    profile = _ThetaProfile()
    result = _optimize_profile(monkeypatch, profile, message=message, status=status)

    assert result.converged
    assert result.status == status
    assert result.theta_completion_steps >= 2
    assert result.theta_completion_stop in {
        "resolved",
        "no gradient reduction",
        "quadratic model error",
    }
    assert message in result.message
    assert "secant Newton completion" in result.message
    assert "projected gradient" in result.message
    # Rho is bitwise pinned in every exact trial, including completion steps.
    for request in profile.requests:
        np.testing.assert_array_equal(request[:-1], _PINNED_RHO)
    t = float(result.trial.params[-1])
    assert abs(t - _T_STAR) <= 4 * np.spacing(_T_STAR)
    assert result.projected_gradient_inf <= _PUBLIC_CONTROL.gtol
    # The returned score and gradient belong to the returned theta.
    expected = _profile_trial(result.trial.params)
    assert float(result.trial.score) == float(expected.score)
    np.testing.assert_array_equal(result.trial.gradient, expected.gradient)
    np.testing.assert_array_equal(result.trial.fit_result.log_theta, [t])
    # Accepted (and at most one other) trial at every new evaluation.
    assert profile.peak_live <= 2
    assert result.n_evaluations == len(profile.requests)
    assert result.cumulative_source_scans == 2 * len(profile.requests)
    assert result.cumulative_batches_scanned == 5 * len(profile.requests)
    assert result.n_accepted == 2 + result.theta_completion_steps
    history = np.asarray(result.accepted_score_history)
    assert history[-1] == float(result.trial.score)
    assert np.all(np.diff(history) <= 0.0)


def test_theta_secant_completion_steps_from_previous_accepted_secant(monkeypatch):
    profile = _ThetaProfile()
    result = _optimize_profile(monkeypatch, profile, probes=())
    completion_requests = profile.requests[3:]
    first = float(completion_requests[0][-1])
    previous, accepted = _T_STAR + 0.02, _T_STAR + 3e-4
    secant = _K * (np.expm1(3e-4) - np.expm1(0.02)) / (accepted - previous)
    np.testing.assert_allclose(
        first - accepted,
        -_K * np.expm1(3e-4) / secant,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert result.converged


def test_theta_secant_completion_caps_step_like_newton(monkeypatch):
    # A resolved but nearly flat secant asks for a step beyond maxNstep = 5.
    profile = _ThetaProfile({1: {"gradient": 1.0}, 2: {"gradient": 1.0 + 1e-10}})
    _optimize_profile(monkeypatch, profile, start=0.0, accepted=(1e-3,), probes=())
    assert float(profile.requests[2][-1]) == _T_STAR + 1e-3 - 5.0


@pytest.mark.parametrize(
    ("message", "status", "stop"),
    [
        ("STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT", 1, "limit or error"),
        ("ERROR: STPMAX < STPMIN", 2, "limit or error"),
        ("CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL", 0, "stage converged"),
    ],
    ids=["iteration-limit", "error", "converged"],
)
def test_theta_secant_completion_only_follows_roundoff_stops(
    monkeypatch, message, status, stop
):
    profile = _ThetaProfile()
    accepted = (0.02, 3e-4) if status else (0.02, 3e-4, 1e-12)
    result = _optimize_profile(
        monkeypatch, profile, message=message, status=status, accepted=accepted
    )
    assert result.theta_completion_steps == 0
    assert stop in result.theta_completion_stop
    assert result.theta_completion_stop.startswith("not run")
    assert len(profile.requests) == 1 + len(accepted) + 1
    assert result.converged == (status == 0)
    assert "not run" in result.message


@pytest.mark.parametrize(
    ("accepted", "stop"),
    [
        ((), "no resolved secant"),
        # A gradient change below twice the inner tolerance is not a secant.
        ((1e-3 + 1e-14,), "no resolved secant"),
    ],
    ids=["initial-point", "unresolved-pair"],
)
def test_theta_secant_completion_requires_resolved_secant(monkeypatch, accepted, stop):
    profile = _ThetaProfile()
    result = _optimize_profile(
        monkeypatch, profile, start=1e-3 if accepted else 0.3, accepted=accepted
    )
    assert not result.converged
    assert result.theta_completion_stop == stop
    assert result.theta_completion_steps == 0
    assert "no positive resolved log-theta curvature" in result.message
    assert len(profile.requests) == 1 + len(accepted) + 1


def test_theta_secant_completion_reports_nonpositive_secant(monkeypatch):
    # The accepted pair's gradient falls as log theta rises: concave secant.
    profile = _ThetaProfile({1: {"gradient": 0.5}, 2: {"gradient": 0.4}})
    result = _optimize_profile(
        monkeypatch, profile, start=0.01, accepted=(0.02,), probes=()
    )
    assert not result.converged
    assert result.theta_completion_stop == "nonpositive curvature"
    assert result.message.endswith("no positive resolved log-theta curvature.")
    assert len(profile.requests) == 2


def _rejected_completion_candidate(reason):
    accepted_gradient = _K * np.expm1(3e-4)
    if reason == "no gradient reduction":
        return {"gradient": -2.0 * accepted_gradient}
    if reason == "measurable score increase":
        return {"score": _PROFILE_SCORE + 1e-3}
    if reason == "nonfinite trial":
        return {"score": np.nan}
    if reason == "nonpositive trial curvature":
        return {"gradient": 1.01 * accepted_gradient}
    # With a positive secant and a smaller gradient the trapezoid change of a
    # -g/c step is always negative; an overshoot to r = g_new/g_old <= -0.8
    # fails newton()'s quadratic-model test instead.
    if reason == "quadratic model error":
        return {"gradient": -0.9 * accepted_gradient}
    raise AssertionError(reason)


@pytest.mark.parametrize(
    "reason",
    [
        "no gradient reduction",
        "measurable score increase",
        "nonfinite trial",
        "nonpositive trial curvature",
        "quadratic model error",
    ],
)
def test_theta_secant_completion_rejects_candidate_and_keeps_accepted(
    monkeypatch, reason
):
    profile = _ThetaProfile({4: _rejected_completion_candidate(reason)})
    result = _optimize_profile(monkeypatch, profile, probes=())
    assert not result.converged
    assert result.theta_completion_stop == reason
    assert result.theta_completion_steps == 0
    np.testing.assert_array_equal(result.trial.params, _theta_params(3e-4))
    assert result.n_accepted == 2
    assert result.n_evaluations == 4
    # A failed public fit explains itself: L-BFGS-B task, completion outcome,
    # final proposed and last accepted log-theta steps, projected gradient.
    step = float(profile.requests[3][-1] - result.trial.params[-1])
    for part in (
        "L-BFGS-B stage: ABNORMAL: ",
        f"secant Newton completion: {reason}, 0 accepted step(s)",
        f"final proposed step {step:+.6e} (rejected)",
        f"last accepted log-theta step {3e-4 - 0.02:+.6e}",
        f"projected gradient {_K * np.expm1(3e-4):.6e} (gtol 1.000e-09)",
        "not converged: projected gradient",
    ):
        assert part in result.message, (part, result.message)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (lambda params: _profile_trial(np.r_[0.0, params[-1]]), "changed supplied rho"),
        (lambda params: _profile_trial(params, link_name="LogLink"), "lineage"),
        (
            lambda params: _profile_trial(params, source_fingerprint="other"),
            "lineage",
        ),
    ],
)
def test_theta_secant_completion_refuses_changed_rho_or_lineage(
    monkeypatch, override, match
):
    profile = _ThetaProfile({4: override})
    with pytest.raises(RuntimeError, match=match):
        _optimize_profile(monkeypatch, profile, probes=())


def test_theta_secant_completion_refuses_accepted_state_with_moved_rho(monkeypatch):
    profile = _ThetaProfile()
    monkeypatch.setattr(nb_reml_execution, "evaluate_nb_stream_reml", profile)

    def moved_pin(fun, x0, **kwargs):
        moved = np.r_[_PINNED_RHO + 1e-12, _T_STAR + 1e-3]
        fun(np.asarray(x0))
        fun(moved)
        kwargs["callback"](OptimizeResult(x=moved.copy(), fun=0.0))
        return SimpleNamespace(
            x=moved, success=False, status=2, message="ABNORMAL: ", nit=1
        )

    monkeypatch.setattr(reml_execution, "minimize", moved_pin)
    with pytest.raises(RuntimeError, match="changed supplied rho"):
        nb_reml_execution.optimize_nb_stream_reml(
            _theta_stream(),
            NegativeBinomial(theta=2.7, fixed=False, link="identity"),
            _theta_params(0.3),
            maximum_bytes=1_000_000,
            pin_lambda=True,
            pirls_control=StreamPIRLSControl(solver_policy="qr", tol=_INNER_TOLERANCE),
            control=_PUBLIC_CONTROL,
        )
    assert len(profile.requests) == 2


def test_theta_secant_completion_refuses_invalid_accepted_trial(monkeypatch):
    profile = _ThetaProfile({3: {"converged": False}})
    result = _optimize_profile(monkeypatch, profile, probes=())
    assert not result.converged
    assert result.theta_completion_stop == "not run: accepted trial is invalid"
    assert len(profile.requests) == 3


def test_resolved_theta_secant_rejects_unusable_pairs():
    secant = nb_reml_execution._resolved_theta_secant
    old = _profile_trial(_theta_params(1e-3))
    new = _profile_trial(_theta_params(2e-3))
    expected = (new.gradient[-1] - old.gradient[-1]) / (new.params[-1] - old.params[-1])
    assert secant(old, new, _INNER_TOLERANCE) == expected
    moved = _profile_trial(np.r_[0.0, new.params[-1]])
    assert secant(old, moved, _INNER_TOLERANCE) is None
    assert secant(old, _profile_trial(np.r_[0.0, 1.0, 2.0]), 1e-11) is None
    assert secant(old, old, _INNER_TOLERANCE) is None
    assert secant(old, new, 1.0) is None
    nonfinite = _profile_trial(new.params, gradient=np.inf)
    assert secant(old, nonfinite, _INNER_TOLERANCE) is None


def test_nb_objective_rejects_invalid_secant_gradient_error(monkeypatch):
    monkeypatch.setattr(nb_reml_execution, "evaluate_nb_stream_reml", _ThetaProfile())
    for value in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="secant_gradient_error"):
            nb_reml_execution._AcceptedNBTrialObjective(
                _theta_stream(),
                NegativeBinomial(theta=2.7, fixed=False),
                _theta_params(0.1),
                StreamPIRLSControl(solver_policy="qr"),
                _PUBLIC_CONTROL,
                1_000_000,
                None,
                None,
                secant_gradient_error=value,
            )


# ---------------------------------------------------------------------------
# Real generated source: forced roundoff stops on the public free-theta model
# ---------------------------------------------------------------------------


_ROUNDOFF_STOPS = {
    "abnormal": ("ABNORMAL: ", 2),
    "relative-reduction": ("CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH", 0),
}


def _plateau_then_abnormal(record, stop="abnormal", plateau_gtol=1e-6):
    """Run real L-BFGS-B into the score plateau, then stop as native AMD64 did.

    The native run accepted an iterate with |g| ~ 8e-8 whose remaining score
    decrease (~g^2/2c ~ 1e-16) is far below one score ULP; its line search
    then failed every sufficient-decrease test and ended ABNORMAL. A
    relative-reduction stop leaves the same accepted plateau trial.
    """
    message, status = _ROUNDOFF_STOPS[stop]

    def minimize_call(fun, x0, **kwargs):
        options = {**kwargs["options"], "gtol": plateau_gtol}
        plateau = scipy_minimize(fun, x0, **{**kwargs, "options": options})
        assert plateau.success
        record["plateau_gradient"] = float(np.asarray(plateau.jac)[-1])
        return SimpleNamespace(
            x=np.asarray(plateau.x),
            success=status == 0,
            status=status,
            message=message,
            nit=int(plateau.nit),
        )

    return minimize_call


def _optimize_generated(link, monkeypatch=None, record=None):
    stream, family, initial = _optimizer_case(link, estimated=True)
    if monkeypatch is not None:
        monkeypatch.setattr(reml_execution, "minimize", _plateau_then_abnormal(record))
    result = nb_reml_execution.optimize_nb_stream_reml(
        stream,
        family,
        initial,
        maximum_bytes=_OPTIMIZER_MAXIMUM_BYTES,
        pin_lambda=True,
        pirls_control=_OPTIMIZER_PIRLS_CONTROL,
        control=_PUBLIC_CONTROL,
    )
    return stream, initial, result


@pytest.mark.parametrize("link", _LINKS)
def test_generated_abnormal_stop_completes_to_unforced_stationary_theta(
    monkeypatch, link
):
    _, _, reference = _optimize_generated(link)
    record = {}
    _, initial, forced = _optimize_generated(link, monkeypatch, record)

    # The completion starts from a valid plateau trial that misses gtol.
    assert abs(record["plateau_gradient"]) > _PUBLIC_CONTROL.gtol
    assert reference.converged
    assert forced.status == 2
    assert forced.message.startswith("L-BFGS-B stage: ABNORMAL")
    assert forced.converged, forced.message
    assert forced.theta_completion_steps >= 1
    assert forced.projected_gradient_inf <= _PUBLIC_CONTROL.gtol
    np.testing.assert_array_equal(np.asarray(forced.trial.params)[:-1], initial[:-1])
    np.testing.assert_array_equal(
        forced.trial.fit_result.log_theta, np.asarray(forced.trial.params)[-1:]
    )
    checks = _AssertCollector()
    for name, actual, expected in (
        ("log theta", forced.trial.params, reference.trial.params),
        ("score", forced.trial.score, reference.trial.score),
        (
            "coefficients",
            forced.trial.fit_result.state.coefficients,
            reference.trial.fit_result.state.coefficients,
        ),
        (
            "deviance",
            forced.trial.fit_result.state.deviance,
            reference.trial.fit_result.state.deviance,
        ),
    ):
        checks.check(
            name,
            lambda actual=actual, expected=expected: np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    checks.raise_if_any(f"forced ABNORMAL NB/{link} completion")


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", _LINKS)
def test_generated_abnormal_completion_is_pinned_gam_fit4_stationary(monkeypatch, link):
    """Pinned R's exact source state at the completed theta is stationary."""
    stream, _, forced = _optimize_generated(link, monkeypatch, {})
    assert forced.converged, forced.message
    expected = _pinned_selected_reference(
        stream,
        link,
        np.asarray(forced.trial.params),
        estimated=True,
        epsilon=1e-12,
        max_iter=400,
    )
    state = forced.trial.fit_result.state
    checks = _AssertCollector()
    for name, actual, reference in (
        ("score", forced.trial.score, expected["score"]),
        ("gradient", forced.trial.gradient, expected["gradient"]),
        ("coefficients", state.coefficients, expected["beta"]),
        ("deviance", state.deviance, expected["deviance"]),
    ):
        checks.check(
            name,
            lambda actual=actual, reference=reference: np.testing.assert_allclose(
                actual, reference, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    checks.raise_if_any(f"forced ABNORMAL NB/{link} pinned gam.fit4 state")
    # R's own exact log-theta derivative certifies the completed theta.
    assert abs(float(expected["gradient"][-1])) <= _PUBLIC_CONTROL.gtol


# Stationary exact trials read this many ULPs high. It exceeds the few-ULP
# warm-trial score scatter, so L-BFGS-B can never accept them, and stays
# inside the documented float64 score envelope (about 55 ULPs here).
_STATIONARY_SCORE_ULPS = 40


@pytest.mark.parametrize("link", _LINKS)
def test_public_free_theta_completes_when_stationary_scores_read_high(
    monkeypatch, link
):
    """Score roundoff on the plateau cannot leave the public fit unconverged."""
    data, weight, offset = nb_optimizer_case_data()

    def fit():
        model = GAM(
            'y~s(x,bs="cr",k=8)',
            family=NegativeBinomial(theta=2.7, fixed=False, link=link),
            sp=[0.5],
            control=FitControl(
                execution="stream",
                linear_solver="qr",
                batch_rows=17,
                uncertainty="fisher",
            ),
        )
        source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
        return model.fit(source, result="prediction")

    reference = fit()
    exact = nb_reml_execution.evaluate_nb_stream_reml
    forced_trials = []

    def high_stationary_score(*args, **kwargs):
        trial = exact(*args, **kwargs)
        if abs(float(np.asarray(trial.gradient)[-1])) < _PUBLIC_CONTROL.gtol:
            score = float(np.asarray(trial.score))
            forced_trials.append(score)
            trial = replace(
                trial,
                score=jnp.asarray(score + _STATIONARY_SCORE_ULPS * np.spacing(score)),
            )
        return trial

    monkeypatch.setattr(
        nb_reml_execution, "evaluate_nb_stream_reml", high_stationary_score
    )
    forced = fit()
    info = forced.convergence_info
    assert reference.converged, reference.convergence_info
    assert forced_trials
    assert forced.converged, info
    # L-BFGS-B never accepts a stationary trial here; a completion must.
    assert "NORM OF PROJECTED GRADIENT" not in info
    assert "retained exact stationary candidate" in info or re.search(
        r"completion: [a-z ]+, [1-9]\d* accepted step", info
    ), info
    checks = _AssertCollector()
    for name, actual, expected in (
        ("theta", forced.theta, reference.theta),
        ("coefficients", forced.coefficients, reference.coefficients),
        ("score", forced.score, reference.score),
        ("deviance", forced.deviance, reference.deviance),
    ):
        checks.check(
            name,
            lambda actual=actual, expected=expected: np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    checks.raise_if_any(f"public NB/{link} forced plateau completion")


def _optimize_no_smooth(link, monkeypatch=None, record=None, stop="abnormal"):
    """Estimated theta without smoothing: log theta is the only coordinate."""
    stream, family, initial = _optimizer_case(link, estimated=True, smooth=False)
    assert stream.prepared.fitting.penalty_structure.n_penalties == 0
    assert initial.shape == (1,)
    if monkeypatch is not None:
        # Without a smooth the first-order stage converges faster; a looser
        # scripted stage still leaves a plateau trial above gtol.
        monkeypatch.setattr(
            reml_execution,
            "minimize",
            _plateau_then_abnormal(record, stop, plateau_gtol=1e-3),
        )
    return nb_reml_execution.optimize_nb_stream_reml(
        stream,
        family,
        initial,
        maximum_bytes=_OPTIMIZER_MAXIMUM_BYTES,
        pin_lambda=False,
        pirls_control=_OPTIMIZER_PIRLS_CONTROL,
        control=_PUBLIC_CONTROL,
    )


def _assert_forced_completion(forced, record, stop):
    message, status = _ROUNDOFF_STOPS[stop]
    # The completion starts from a valid plateau trial that misses gtol.
    assert abs(record["plateau_gradient"]) > _PUBLIC_CONTROL.gtol
    assert forced.status == status
    assert forced.message.startswith(f"L-BFGS-B stage: {message}")
    assert forced.converged, forced.message
    assert forced.theta_completion_steps >= 1
    assert forced.projected_gradient_inf <= _PUBLIC_CONTROL.gtol
    np.testing.assert_array_equal(
        forced.trial.fit_result.log_theta, np.asarray(forced.trial.params)
    )


@pytest.mark.parametrize("stop", sorted(_ROUNDOFF_STOPS))
@pytest.mark.parametrize("link", _LINKS)
def test_no_smooth_estimated_theta_roundoff_stop_completes_to_unforced_endpoint(
    monkeypatch, link, stop
):
    reference = _optimize_no_smooth(link)
    record = {}
    forced = _optimize_no_smooth(link, monkeypatch, record, stop)
    assert reference.converged, reference.message
    _assert_forced_completion(forced, record, stop)
    # The theta-only route charges the retained stationary trial and the two
    # recorded secant scalars, exactly as the pinned-rho route does.
    retained, history, lbfgs = reml_execution._parameterized_optimizer_workspace_bytes(
        2, 1, _PUBLIC_CONTROL, _OPTIMIZER_PIRLS_CONTROL
    )
    assert forced.outer_workspace_bytes == (
        2 * retained + 2 * history + lbfgs + nb_reml_execution._THETA_SECANT_BYTES
    )
    checks = _AssertCollector()
    for name, actual, expected in (
        ("log theta", forced.trial.params, reference.trial.params),
        ("score", forced.trial.score, reference.trial.score),
        (
            "coefficients",
            forced.trial.fit_result.state.coefficients,
            reference.trial.fit_result.state.coefficients,
        ),
        (
            "deviance",
            forced.trial.fit_result.state.deviance,
            reference.trial.fit_result.state.deviance,
        ),
    ):
        checks.check(
            name,
            lambda actual=actual, expected=expected: np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    checks.raise_if_any(f"no-smooth NB/{link} forced {stop} completion")


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", _LINKS)
def test_no_smooth_estimated_theta_abnormal_completion_matches_pinned_r(
    monkeypatch, link
):
    """Pinned mgcv's own theta-only newton() fit agrees at STRICT."""
    record = {}
    forced = _optimize_no_smooth(link, monkeypatch, record)
    _assert_forced_completion(forced, record, "abnormal")
    data, weight, offset = nb_optimizer_case_data()
    expected = RBridge(mode="rpy2").public_fixed_sp_fit(
        data, weight, offset, "nb", link, theta=-2.7, sp=None
    )
    state = forced.trial.fit_result.state
    checks = _AssertCollector()
    for name, actual, reference in (
        ("theta", np.exp(np.asarray(forced.trial.params)[-1]), expected["theta"]),
        ("coefficients", state.coefficients, expected["coefficients"]),
        ("score", forced.trial.score, expected["score"]),
        ("deviance", state.deviance, expected["deviance"]),
    ):
        checks.check(
            name,
            lambda actual=actual, reference=reference: np.testing.assert_allclose(
                actual, reference, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    checks.raise_if_any(f"no-smooth NB/{link} forced ABNORMAL pinned R parity")
