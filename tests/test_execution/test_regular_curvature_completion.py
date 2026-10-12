"""Free-parameter curvature Newton completion after L-BFGS-B roundoff stops.

Near a free (log sp, log scale) optimum the remaining REML decrease is below
one ULP of the O(n) score. L-BFGS-B's raw-score line search then stops by
relative reduction or ``ABNORMAL`` at an accepted trial whose raw score is
typically a lucky-low roundoff draw, before ``gtol``. These scripted exact
quadratic objectives force that stop and check that the completion follows
pinned ``gam.fit3.r::newton`` on derivative-integrated changes, with no raw
score increase beyond the float64 envelope of the two scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

import jaxgam.execution.reml as reml_execution
from jaxgam.execution.reml import StreamREMLControl, optimize_regular_stream_reml
from tests.tolerances import STRICT

# Unforced pinned-Linux Gamma/inverse_squared exact score and the magnitude of
# its O(n) terms (each trial's float64 envelope is 5.127e-13 there).
_OPTIMUM_SCORE = -92.16465418694843
_ULP = float(np.spacing(abs(_OPTIMUM_SCORE)))
_MAGNITUDE = 288.7
_EPS = float(np.finfo(np.float64).eps)
_CONTROL = StreamREMLControl(ftol=0.0)
_RELATIVE = "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
_ABNORMAL = "ABNORMAL: "
# Finite-difference curvature of the Linux B=13 stop (rho, log phi).
_GAMMA_HESSIAN = np.array([[1.8665, -1.8756], [-1.8756, 46.477]])
_GAMMA_STOP_GRADIENT = np.array([-4.337156e-7, 1.404458e-6])


def _envelope(_trial) -> float:
    return 8.0 * _EPS * _MAGNITUDE


@dataclass
class _Quadratic:
    """Exact quadratic REML profile with scripted roundoff and failures."""

    hessian: np.ndarray
    target: np.ndarray
    score_override: dict[bytes, float] = field(default_factory=dict)
    gradient_override: dict[int, np.ndarray] = field(default_factory=dict)
    invalid: set[int] = field(default_factory=set)
    lineage_override: dict[int, str] = field(default_factory=dict)
    params_override: dict[int, np.ndarray] = field(default_factory=dict)
    evaluated: list[tuple[np.ndarray, np.ndarray | None]] = field(default_factory=list)
    on_evaluate: object = None

    def __call__(self, params, warm_start):
        index = len(self.evaluated)
        x = np.array(params, dtype=np.float64, copy=True)
        self.evaluated.append(
            (
                x.copy(),
                None
                if warm_start is None
                else np.asarray(warm_start.params, dtype=np.float64).copy(),
            )
        )
        if self.on_evaluate is not None:
            self.on_evaluate(index)
        difference = x - self.target
        gradient = self.gradient_override.get(index, self.hessian @ difference)
        score = self.score_override.get(
            x.tobytes(),
            _OPTIMUM_SCORE + 0.5 * float(difference @ self.hessian @ difference),
        )
        return SimpleNamespace(
            params=self.params_override.get(index, x),
            score=np.float64(score),
            gradient=np.asarray(gradient, dtype=np.float64),
            fit_result=SimpleNamespace(
                state=SimpleNamespace(
                    converged=True,
                    line_search_failed=False,
                    stationarity=1.0 if index in self.invalid else 0.0,
                )
            ),
            source_scans=2,
            batches_scanned=4,
            source_fingerprint="source",
            basis_fingerprint="basis",
            family_name=self.lineage_override.get(index, "Gamma"),
            link_name="InverseSquaredLink",
        )


def _run(
    monkeypatch,
    problem: _Quadratic,
    start: np.ndarray,
    stop: np.ndarray,
    *,
    abnormal: bool = False,
    bounds=None,
    control: StreamREMLControl = _CONTROL,
    max_step: float | None = None,
    message: str | None = None,
    status: int | None = None,
    fingerprint=None,
):
    """Script one accepted L-BFGS-B stop, then run the common driver."""
    source = SimpleNamespace(
        fingerprint=(lambda: "source") if fingerprint is None else fingerprint
    )
    objective = reml_execution._AcceptedParameterizedTrialObjective(
        SimpleNamespace(source=source),
        start,
        control,
        problem,
        objective_name="scripted",
        source_name="scripted",
        trial_name="scripted",
    )

    def scripted(fun, _x0, *, callback, **_kwargs):
        point = np.asarray(stop, dtype=np.float64)
        fun(point)
        callback(SimpleNamespace(x=point))
        default_message = _ABNORMAL if abnormal else _RELATIVE
        return SimpleNamespace(
            x=point.copy(),
            success=not abnormal if message is None else status == 0,
            status=(2 if abnormal else 0) if status is None else status,
            message=default_message if message is None else message,
            nit=1,
        )

    monkeypatch.setattr(reml_execution, "minimize", scripted)
    if bounds is None:
        bounds = [(-40.0, 40.0)] * (start.size - 1) + [(None, None)]
        if start.size == 1:
            bounds = [(-40.0, 40.0)]
    return reml_execution._run_parameterized_stream_reml(
        objective,
        start,
        bounds,
        control,
        curvature_stationary_completion=True,
        completion_inner_tolerance=1e-11,
        completion_score_roundoff=_envelope,
        max_step=max_step,
    )


def _case(m: int):
    """Return curvature, optimum and an above-gtol stop for m free parameters."""
    if m == 1:
        hessian = np.array([[1.7876]])
        target = np.array([-4.2975453706053])
        gradient = np.array([1.404458e-6])
    elif m == 2:
        hessian = _GAMMA_HESSIAN
        target = np.array([-4.2975453706053, -3.6649752613096])
        gradient = _GAMMA_STOP_GRADIENT
    else:
        hessian = np.array(
            [[1.8665, -1.8756, 0.3], [-1.8756, 46.477, -0.5], [0.3, -0.5, 4.0]]
        )
        target = np.array([-4.2975453706053, 2.1, -3.6649752613096])
        gradient = np.array([-4.337156e-7, 9.0e-7, 1.404458e-6])
    stop = target + np.linalg.solve(hessian, gradient)
    start = target + 0.5
    return hessian, target, stop, start


@pytest.mark.parametrize("abnormal", [False, True], ids=["relative", "abnormal"])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_completion_converges_from_lucky_low_roundoff_stop(monkeypatch, m, abnormal):
    """The exact Newton trial reads 26 ULPs above a lucky-low accepted score."""
    hessian, target, stop, start = _case(m)
    problem = _Quadratic(
        hessian,
        target,
        score_override={stop.tobytes(): _OPTIMUM_SCORE - 26 * _ULP},
    )
    result = _run(monkeypatch, problem, start, stop, abnormal=abnormal)
    assert result.converged
    assert result.status == (2 if abnormal else 0)
    assert result.completion_stop == "converged"
    assert result.completion_steps == 1
    assert result.n_accepted == 2
    assert result.n_iter == 2
    assert result.n_evaluations == 2 + 2 * m + 1
    assert result.cumulative_source_scans == 2 * result.n_evaluations
    assert result.cumulative_batches_scanned == 4 * result.n_evaluations
    np.testing.assert_allclose(
        np.asarray(result.trial.params), target, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert result.projected_gradient_inf <= _CONTROL.gtol
    # The returned raw score rose by more than one ULP: the former two-
    # parameter correction rejected exactly this trial. It stays inside the
    # float64 envelope of the two scores, and it is the trial's own score.
    rise = result.accepted_score_history[-1] - result.accepted_score_history[-2]
    assert _ULP < rise <= 2.0 * _envelope(None)
    assert float(result.trial.score) == result.accepted_score_history[-1]
    assert (
        "free-parameter curvature Newton completion: 1 accepted step(s), "
        "stopped by converged" in result.message
    )
    assert "exceeds gtol" not in result.message
    for _params, warm_start in problem.evaluated[2:]:
        np.testing.assert_array_equal(warm_start, stop)


@pytest.mark.parametrize(
    ("override", "reason", "evaluations"),
    [
        ("measurable_increase", "measurable score increase", 8),
        ("model_error", "quadratic model error", 8),
        ("negative_secant", "nonpositive secant curvature", 8),
        ("larger_gradient", "no gradient reduction", 8),
        ("invalid_trial", "invalid trial", 8),
    ],
)
def test_completion_rejects_every_halving_and_keeps_accepted(
    monkeypatch, override, reason, evaluations
):
    hessian, target, stop, start = _case(1)
    gradient = hessian @ (stop - target)
    problem = _Quadratic(hessian, target)
    trial_indices = range(4, 8)
    if override == "measurable_increase":
        # A 120-ULP rise exceeds the 1.03e-12 envelope of the two scores.
        problem.score_override[stop.tobytes()] = _OPTIMUM_SCORE - 120 * _ULP
    for index in trial_indices:
        if override == "model_error":
            problem.gradient_override[index] = -0.9 * gradient
        elif override == "negative_secant":
            problem.gradient_override[index] = 2.0 * gradient
        elif override == "larger_gradient":
            problem.gradient_override[index] = -1.5 * gradient
        elif override == "invalid_trial":
            problem.invalid.add(index)
    result = _run(monkeypatch, problem, start, stop)
    assert not result.converged
    assert result.completion_stop == f"rejected: {reason}"
    assert result.completion_steps == 0
    assert result.n_evaluations == evaluations
    np.testing.assert_array_equal(np.asarray(result.trial.params), stop)
    assert result.projected_gradient_inf > _CONTROL.gtol
    assert "(rejected)" in result.message
    assert "exceeds gtol" in result.message
    # Halved trials are fractions of the same Newton step from the stop.
    steps = [params - stop for params, _ in problem.evaluated[4:]]
    for halving, step in enumerate(steps):
        np.testing.assert_allclose(
            step, steps[0] / 2.0**halving, rtol=STRICT.rtol, atol=STRICT.atol
        )


def test_completion_requires_derivative_integrated_descent(monkeypatch):
    """An exact trial gradient inconsistent with descent is never accepted."""
    hessian = np.diag([1.0, 1e4])
    target = np.array([0.3, -3.0])
    stop = target + np.array([1e-8, 2e-10])
    problem = _Quadratic(hessian, target)
    for index in range(6, 10):
        problem.gradient_override[index] = np.array([-1e-7, 0.0])
    result = _run(monkeypatch, problem, target + 0.5, stop)
    assert not result.converged
    assert result.completion_stop == "rejected: no derivative-integrated descent"
    assert result.n_evaluations == 2 + 4 + 4


@pytest.mark.parametrize(
    ("scenario", "stop_reason", "evaluations"),
    [
        ("invalid_probe", "invalid curvature probe", 3),
        ("indefinite", "indefinite curvature", 4),
        ("probe_bound", "curvature probe outside bounds", 2),
        ("step_bound", "step leaves parameter bounds", 4),
        ("unrepresentable", "step not representable", 4),
    ],
)
def test_completion_reports_unusable_curvature_or_step(
    monkeypatch, scenario, stop_reason, evaluations
):
    hessian, target, stop, start = _case(1)
    problem = _Quadratic(hessian, target)
    if scenario == "invalid_probe":
        problem.invalid.add(2)
    elif scenario == "indefinite":
        problem.hessian = -hessian
        stop = target - (stop - target)
    elif scenario == "probe_bound":
        target = np.array([40.0 - 1e-3])
        problem.target = target
        stop = target + (stop - _case(1)[1])
        start = target - 0.5
    elif scenario == "step_bound":
        target = np.array([41.0])
        problem.target = target
        stop = np.array([39.0])
        start = np.array([38.0])
    else:
        problem.hessian = np.array([[1e20]])
        target = np.array([1.0])
        problem.target = target
        stop = target.copy()
        start = target + 0.5
        problem.gradient_override[1] = np.array([2e-6])
    result = _run(monkeypatch, problem, start, stop)
    assert not result.converged
    assert result.completion_stop == stop_reason
    assert result.completion_steps == 0
    assert result.n_evaluations == evaluations
    np.testing.assert_array_equal(np.asarray(result.trial.params), stop)
    assert "exceeds gtol" in result.message


@pytest.mark.parametrize(
    ("max_step", "converged", "stop_reason", "positions"),
    [
        (None, True, "converged", [5.0, 8.0]),
        (2.0, False, "step limit", [2.0, 4.0, 6.0]),
    ],
)
def test_completion_caps_newton_steps_like_newton(
    monkeypatch, max_step, converged, stop_reason, positions
):
    """Large corrections follow newton()'s ``maxNstep * Nstep/ms`` scaling."""
    problem = _Quadratic(np.array([[2.0]]), np.array([8.0]))
    result = _run(
        monkeypatch,
        problem,
        np.array([-1.0]),
        np.array([0.0]),
        max_step=max_step,
    )
    assert result.converged is converged
    assert result.completion_stop == stop_reason
    assert result.completion_steps == len(positions)
    assert result.n_evaluations == 2 + 3 * len(positions)
    np.testing.assert_allclose(
        np.asarray(result.trial.params),
        [positions[-1]],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    # Each iteration evaluates two curvature probes, then its accepted step.
    accepted = [problem.evaluated[4 + 3 * k][0][0] for k in range(len(positions))]
    np.testing.assert_allclose(accepted, positions, rtol=STRICT.rtol, atol=STRICT.atol)
    # The exact quadratic decreases by -55 and -9 (or -28, -20 and -12).
    assert np.all(np.diff(result.accepted_score_history[1:]) < 0.0)
    for params, warm_start in problem.evaluated[2:]:
        assert np.max(np.abs(params - warm_start)) <= (max_step or 5.0)


@pytest.mark.parametrize(
    ("control", "stop_reason"),
    [
        (StreamREMLControl(ftol=0.0, maxfun=10), "evaluation limit"),
        (StreamREMLControl(ftol=0.0, max_iter=2), "iteration limit"),
    ],
)
def test_completion_stops_at_shared_limits_after_an_accepted_step(
    monkeypatch, control, stop_reason
):
    problem = _Quadratic(np.array([[2.0]]), np.array([8.0]))
    result = _run(
        monkeypatch,
        problem,
        np.array([-1.0]),
        np.array([0.0]),
        control=control,
    )
    assert not result.converged
    assert result.completion_stop == stop_reason
    assert result.completion_steps == 1
    assert result.n_evaluations == 5
    assert result.n_iter == 2
    np.testing.assert_array_equal(np.asarray(result.trial.params), [5.0])


@pytest.mark.parametrize("indefinite", [False, True])
def test_abnormal_stop_at_stationary_trial_needs_positive_curvature(
    monkeypatch, indefinite
):
    """ABNORMAL below gtol converges only with a positive definite model."""
    hessian, target, _stop, start = _case(2)
    stop = target + np.linalg.solve(hessian, 0.4 * _GAMMA_STOP_GRADIENT)
    problem = _Quadratic(hessian, target)
    if indefinite:
        problem.hessian = np.array([[-1.8665, -1.8756], [-1.8756, 46.477]])
        problem.gradient_override[1] = 0.4 * _GAMMA_STOP_GRADIENT
    result = _run(monkeypatch, problem, start, stop, abnormal=True)
    assert result.converged is not indefinite
    assert result.status == 2
    assert result.completion_steps == 0
    assert result.n_evaluations == 2 + 4
    np.testing.assert_array_equal(np.asarray(result.trial.params), stop)
    assert result.projected_gradient_inf <= _CONTROL.gtol
    if indefinite:
        assert result.completion_stop == "indefinite curvature"
        assert result.message.endswith("no positive definite curvature model.")
    else:
        assert result.completion_stop == "stationary"
        assert not result.message.endswith(".")


@pytest.mark.parametrize(
    ("scenario", "stop_reason"),
    [
        ("limit", None),
        ("error", None),
        ("invalid_accepted", "not run: accepted trial is invalid"),
        ("too_many", "not run: free parameter count outside the completion range"),
    ],
)
def test_completion_is_not_run_outside_roundoff_stops(
    monkeypatch, scenario, stop_reason
):
    m = 5 if scenario == "too_many" else 1
    hessian = np.eye(m) * 1.7876
    target = np.full(m, -4.2975453706053)
    stop = target + 1e-6
    problem = _Quadratic(hessian, target)
    kwargs = {}
    if scenario == "limit":
        kwargs = {"message": "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT", "status": 1}
    elif scenario == "error":
        kwargs = {"message": "line search failed", "status": 2}
    elif scenario == "invalid_accepted":
        problem.invalid.add(1)
    result = _run(
        monkeypatch,
        problem,
        target + 0.5,
        stop,
        bounds=[(-40.0, 40.0)] * m,
        **kwargs,
    )
    assert not result.converged
    assert result.completion_stop == stop_reason
    assert result.n_evaluations == 2
    np.testing.assert_array_equal(np.asarray(result.trial.params), stop)
    if stop_reason is None:
        assert "curvature Newton completion" not in result.message
    else:
        assert stop_reason in result.message


def test_completion_moves_only_free_coordinates(monkeypatch):
    hessian, target, _stop, _start = _case(2)
    pinned = target[0] + 0.25
    start = np.array([pinned, target[1] + 0.5])
    stop = np.array([pinned, target[1] + 3e-8])
    problem = _Quadratic(hessian, target)
    result = _run(
        monkeypatch,
        problem,
        start,
        stop,
        bounds=[(pinned, pinned), (None, None)],
    )
    # Only log phi is free; the pinned rho never moves.
    assert result.completion_stop == "converged"
    assert result.n_evaluations == 2 + 2 + 1
    for params, _ in problem.evaluated:
        assert params[0] == pinned
    assert result.converged
    assert result.projected_gradient_inf <= _CONTROL.gtol


@pytest.mark.parametrize("failure", ["lineage", "params", "source"])
def test_completion_refuses_changed_lineage_params_or_source(monkeypatch, failure):
    hessian, target, stop, start = _case(1)
    problem = _Quadratic(hessian, target)
    fingerprint = ["source"]
    if failure == "lineage":
        problem.lineage_override[2] = "Gaussian"
        match = "lineage"
    elif failure == "params":
        problem.params_override[2] = stop.copy()
        match = "other parameters"
    else:

        def mutate(index):
            if index == 4:
                fingerprint[0] = "changed"

        problem.on_evaluate = mutate
        match = "RowSource changed"
    with pytest.raises(RuntimeError, match=match):
        _run(monkeypatch, problem, start, stop, fingerprint=lambda: fingerprint[0])


def test_curvature_completion_requires_score_envelope() -> None:
    objective = SimpleNamespace(accepted=SimpleNamespace(params=np.zeros(1)))
    with pytest.raises(ValueError, match="score roundoff envelope"):
        reml_execution._run_parameterized_stream_reml(
            objective,
            np.zeros(1),
            [(None, None)],
            _CONTROL,
            curvature_stationary_completion=True,
            completion_inner_tolerance=1e-11,
        )


def test_newton_rejection_reports_nonfinite_trial_and_invalid_envelope():
    accepted = SimpleNamespace(
        params=np.array([0.0]), score=1.0, gradient=np.array([-1.0])
    )
    trial = SimpleNamespace(
        params=np.array([1.0]), score=np.nan, gradient=np.array([0.0])
    )
    arguments = (np.array([1.0]), np.eye(1), np.array([0]), [(None, None)])
    assert (
        reml_execution._curvature_newton_rejection(
            accepted, trial, *arguments, _envelope
        )
        == "nonfinite trial"
    )
    trial.score = 0.5
    assert (
        reml_execution._curvature_newton_rejection(
            accepted, trial, *arguments, lambda _trial: np.nan
        )
        == "invalid score envelope"
    )


def _regular_stub(params, *, score, gradient, log_phi_curvature=None):
    """Minimal regular trial carrying every field the regular envelope reads."""
    return SimpleNamespace(
        params=np.asarray(params, dtype=np.float64).copy(),
        score=np.float64(score),
        gradient=np.asarray(gradient, dtype=np.float64),
        log_phi_curvature=log_phi_curvature,
        source_scans=2,
        batches_scanned=4,
        source_fingerprint="source",
        basis_fingerprint="basis",
        family_name="Poisson",
        link_name="LogLink",
        workspace=SimpleNamespace(required_bytes=128),
        fit_result=SimpleNamespace(
            source_score=SimpleNamespace(penalized_deviance=90.0),
            state=SimpleNamespace(
                converged=True,
                line_search_failed=False,
                stationarity=0.0,
                score_scale=np.float64(1.0),
                saturated_loglik=np.float64(-150.0),
                coefficient_factor=SimpleNamespace(
                    logdet_hessian=lambda: np.float64(12.0)
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("n_penalties", "scale_known", "pin_lambda", "enabled"),
    [
        (1, True, False, True),
        (1, False, False, True),
        (3, False, False, True),
        (4, True, False, True),
        (4, False, False, False),
        (1, False, True, False),
    ],
)
def test_regular_optimizer_enables_completion_for_small_free_counts(
    monkeypatch, n_penalties, scale_known, pin_lambda, enabled
):
    """Free smoothing with at most four free coordinates owns the completion."""
    n_params = n_penalties + int(not scale_known)
    target = np.linspace(-1.0, 1.0, n_params)
    stop = target + 6e-7
    score_ulps = {stop.tobytes(): -26}

    def evaluate(_stream, _family, params, **_kwargs):
        params = np.asarray(params, dtype=np.float64)
        difference = params - target
        return _regular_stub(
            params,
            score=_OPTIMUM_SCORE
            + float(difference @ difference)
            + score_ulps.get(params.tobytes(), 0) * _ULP,
            gradient=2.0 * difference,
            # Exact second derivative of the quadratic in log scale.
            log_phi_curvature=None if scale_known else 2.0,
        )

    def scripted(fun, _x0, *, callback, **_kwargs):
        fun(stop)
        callback(SimpleNamespace(x=stop))
        return SimpleNamespace(
            x=stop.copy(), success=False, status=2, message=_ABNORMAL, nit=1
        )

    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", evaluate)
    monkeypatch.setattr(reml_execution, "minimize", scripted)
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
        prepared=SimpleNamespace(
            fitting=SimpleNamespace(
                penalty_structure=SimpleNamespace(n_penalties=n_penalties)
            ),
            n_coef=4,
        ),
    )
    control = StreamREMLControl(ftol=0.0, history_size=4)
    result = optimize_regular_stream_reml(
        stream,
        SimpleNamespace(scale_known=scale_known),
        stop if not pin_lambda else np.r_[stop[:n_penalties], target[-1] + 0.5],
        maximum_bytes=1_000_000,
        pin_lambda=pin_lambda,
        control=control,
    )
    parts = reml_execution._regular_optimizer_workspace_bytes(
        4, n_params, control, reml_execution.StreamPIRLSControl(solver_policy="qr")
    )
    curvature_bytes = 8 * (n_params * n_params + 24 * n_params) if enabled else 0
    assert result.outer_workspace_bytes == sum(parts) + curvature_bytes
    if pin_lambda:
        # With every rho pinned the log scale is the only free coordinate:
        # the pinned exact-curvature log-scale completion takes precedence
        # (it admits this ABNORMAL stop) and the free completion never runs.
        assert result.curvature_completion_stop is None
        assert result.curvature_completion_steps == 0
        assert result.scale_completion_steps == 1
        assert result.scale_completion_stop in {"resolved", "no gradient reduction"}
        assert result.converged
        np.testing.assert_array_equal(
            np.asarray(result.trial.params)[:-1], stop[:n_penalties]
        )
        np.testing.assert_allclose(
            np.asarray(result.trial.params)[-1],
            target[-1],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        return
    assert result.converged is enabled
    if enabled:
        assert result.curvature_completion_stop == "converged"
        assert result.curvature_completion_steps == 1
        np.testing.assert_allclose(
            np.asarray(result.trial.params),
            target,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    else:
        assert result.curvature_completion_stop is None
        assert result.curvature_completion_steps == 0
        np.testing.assert_array_equal(np.asarray(result.trial.params), stop)
