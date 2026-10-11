"""Pinned-rho free-log-scale Newton completion for regular streamed REML."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgam.execution.reml as reml_execution
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.reml import StreamREMLControl, optimize_regular_stream_reml
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Gamma, Gaussian
from jaxgam.formula.design_provider import StreamDesign
from tests.test_execution.test_regular_penalized_matrix import (
    _rank_one_preparation,
    _source_reference,
)
from tests.test_execution.test_regular_starts import _case
from tests.tolerances import STRICT

# A Gaussian-shaped fixed-rho profile f(t) = A exp(-t) + B t + C in t = log phi.
_A = 0.02
_B = 40.0
_C = -150.0
_STATIONARY = float(np.log(_A / _B))
_PINNED = np.array([40.01])


def _trial(params, *, score=None, gradient=None, curvature=None, **lineage):
    params = np.asarray(params, dtype=np.float64)
    t = float(params[-1])
    phi = float(np.exp(t))
    exact_score = _A * np.exp(-t) + _B * t + _C
    exact_gradient = -_A * np.exp(-t) + _B
    exact_curvature = _A * np.exp(-t)
    names = {
        "source_fingerprint": "source",
        "basis_fingerprint": "basis",
        "family_name": "profile",
        "link_name": "ProfileLink",
    }
    names.update(lineage)
    return SimpleNamespace(
        params=params.copy(),
        score=jnp.asarray(exact_score if score is None else score),
        gradient=jnp.asarray(
            np.r_[0.7, exact_gradient if gradient is None else gradient]
        ),
        log_phi_curvature=exact_curvature if curvature is None else curvature,
        source_scans=2,
        batches_scanned=5,
        fit_result=SimpleNamespace(
            state=SimpleNamespace(
                score_scale=jnp.asarray(phi),
                saturated_loglik=jnp.asarray(-200.0),
                coefficient_factor=SimpleNamespace(
                    logdet_hessian=lambda: jnp.asarray(12.0)
                ),
            ),
            source_score=SimpleNamespace(penalized_deviance=2.0 * _A),
        ),
        workspace=SimpleNamespace(required_bytes=256),
        **names,
    )


class _Profile:
    """Exact profile evaluator recording every requested parameter vector."""

    def __init__(self, scripted=None):
        self.requests = []
        self.objective = None
        self.scripted = dict(scripted or {})

    def __call__(self, _stream, _family, params, **kwargs):
        params = np.asarray(params, dtype=np.float64)
        if self.objective is not None:
            # A replacement is built only after the previous candidate is
            # released: accepted plus one candidate is the peak retention.
            assert self.objective.candidate is None
            assert kwargs["warm_start"] is self.objective.accepted
        self.requests.append(params.copy())
        overrides = self.scripted.get(len(self.requests), {})
        if callable(overrides):
            return overrides(params)
        return _trial(params, **overrides)


def _objective(monkeypatch, start, profile):
    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", profile)
    objective = reml_execution._AcceptedRegularTrialObjective(
        SimpleNamespace(source=SimpleNamespace(fingerprint=lambda: "source")),
        SimpleNamespace(),
        np.r_[_PINNED, start],
        StreamPIRLSControl(solver_policy="qr"),
        StreamREMLControl(history_size=16),
        1_000_000,
        None,
        None,
    )
    profile.objective = objective
    return objective


def _optimizer_stream():
    fitting = SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1))
    return SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
        prepared=SimpleNamespace(fitting=fitting, n_coef=2),
    )


def _lbfgs_returns_initial(success=True, status=0):
    def minimize_call(objective, params, **kwargs):
        objective(params)
        assert kwargs["bounds"][0] == (40.01, 40.01)
        return SimpleNamespace(
            x=np.asarray(params),
            success=success,
            message="CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
            status=status,
            nit=3,
        )

    return minimize_call


def test_completion_reaches_profile_stationary_scale_with_exact_rho(monkeypatch):
    profile = _Profile()
    start = _STATIONARY + 0.3
    objective = _objective(monkeypatch, start, profile)
    completion = reml_execution._complete_pinned_log_scale(objective, _PINNED)

    assert completion.stop in {"resolved", "no gradient reduction"}
    assert completion.steps >= 3
    assert objective.n_accepted == completion.steps
    assert objective.n_evaluations == len(profile.requests)
    assert objective.cumulative_source_scans == 2 * len(profile.requests)
    assert objective.cumulative_batches_scanned == 5 * len(profile.requests)
    assert objective.candidate is None
    for request in profile.requests:
        np.testing.assert_array_equal(request[:-1], _PINNED)
    final = objective.accepted
    t = float(final.params[-1])
    assert abs(t - _STATIONARY) <= 4 * np.spacing(abs(_STATIONARY))
    # The retained score, gradient and curvature belong to the returned t.
    expected = _trial(final.params)
    assert float(final.score) == float(expected.score)
    np.testing.assert_array_equal(final.gradient, expected.gradient)
    assert final.log_phi_curvature == expected.log_phi_curvature
    history = np.asarray(objective.accepted_score_history)
    assert len(history) == completion.steps + 1
    assert np.all(np.diff(history) <= 0.0)


def test_completion_caps_newton_step_like_newton(monkeypatch):
    # Far above the optimum the exact Newton step exceeds maxNstep = 5.
    start = _STATIONARY + 9.0
    profile = _Profile()
    objective = _objective(monkeypatch, start, profile)
    reml_execution._complete_pinned_log_scale(objective, _PINNED, max_steps=1)
    assert profile.requests[1][-1] == start - 5.0


def test_completion_step_limit_is_reported(monkeypatch):
    profile = _Profile()
    objective = _objective(monkeypatch, _STATIONARY + 0.3, profile)
    completion = reml_execution._complete_pinned_log_scale(
        objective, _PINNED, max_steps=1
    )
    assert completion.stop == "step limit"
    assert completion.steps == 1
    assert objective.n_accepted == 1


def test_completion_stops_without_evaluation_at_exact_stationarity(monkeypatch):
    profile = _Profile({1: {"gradient": 0.0}})
    objective = _objective(monkeypatch, _STATIONARY, profile)
    completion = reml_execution._complete_pinned_log_scale(objective, _PINNED)
    assert completion == reml_execution._ScaleNewtonCompletion(0, "resolved")
    assert len(profile.requests) == 1


@pytest.mark.parametrize("curvature", [0.0, -1.0, np.nan, None])
def test_completion_requires_positive_accepted_curvature(monkeypatch, curvature):
    profile = _Profile({1: {"curvature": curvature}})
    if curvature is None:
        profile.scripted[1] = lambda params: SimpleNamespace(
            **{**vars(_trial(params)), "log_phi_curvature": None}
        )
    objective = _objective(monkeypatch, _STATIONARY + 0.1, profile)
    completion = reml_execution._complete_pinned_log_scale(objective, _PINNED)
    assert completion == reml_execution._ScaleNewtonCompletion(
        0, "nonpositive curvature"
    )
    assert len(profile.requests) == 1


def test_completion_reports_nonfinite_accepted_gradient(monkeypatch):
    profile = _Profile({1: {"gradient": np.nan}})
    objective = _objective(monkeypatch, _STATIONARY + 0.1, profile)
    completion = reml_execution._complete_pinned_log_scale(objective, _PINNED)
    assert completion == reml_execution._ScaleNewtonCompletion(0, "nonfinite gradient")
    assert len(profile.requests) == 1


def test_completion_refuses_accepted_state_with_other_rho(monkeypatch):
    objective = _objective(monkeypatch, _STATIONARY + 0.1, _Profile())
    with pytest.raises(RuntimeError, match="changed supplied rho"):
        reml_execution._complete_pinned_log_scale(objective, np.array([40.0]))
    with pytest.raises(RuntimeError, match="No exact candidate"):
        objective._promote_candidate()


def _rejecting_candidate(reason):
    """Return candidate overrides for one deterministic rejection reason."""
    if reason == "no gradient reduction":
        return {"gradient": -2.0 * (-_A * np.exp(-(_STATIONARY + 0.1)) + _B)}
    if reason == "nonpositive trial curvature":
        return {"curvature": -1e-3}
    if reason == "nonfinite trial":
        return {"score": np.nan}
    if reason == "measurable score increase":
        return {"score": 1e3}
    raise AssertionError(reason)


@pytest.mark.parametrize(
    "reason",
    [
        "no gradient reduction",
        "nonpositive trial curvature",
        "nonfinite trial",
        "measurable score increase",
    ],
)
def test_completion_rejects_candidate_and_keeps_accepted(monkeypatch, reason):
    profile = _Profile({2: _rejecting_candidate(reason)})
    objective = _objective(monkeypatch, _STATIONARY + 0.1, profile)
    accepted = objective.accepted
    completion = reml_execution._complete_pinned_log_scale(objective, _PINNED)
    assert completion == reml_execution._ScaleNewtonCompletion(0, reason)
    assert objective.accepted is accepted
    assert objective.candidate is None
    assert objective.n_accepted == 0
    assert objective.n_evaluations == 2
    assert objective.cumulative_source_scans == 4
    assert objective.accepted_score_history == (float(accepted.score),)


def _scripted_derivatives(gradient_old, curvature_old, gradient_new, curvature_new):
    start = 0.0
    step = -gradient_old / curvature_old

    def candidate(params):
        assert float(params[-1]) == start + step
        return _trial(
            params,
            score=-1.0,
            gradient=gradient_new,
            curvature=curvature_new,
        )

    return start, {
        1: {"score": -1.0, "gradient": gradient_old, "curvature": curvature_old},
        2: candidate,
    }


@pytest.mark.parametrize(
    ("derivatives", "reason"),
    [
        # Hermite change -0.05 - 99/12 against a -0.5 quadratic prediction.
        ((-1.0, 1.0, 0.9, 100.0), "quadratic model error"),
        # Endpoint curvature collapse makes the integrated change positive.
        ((-1.0, 1.0, 0.9, 1e-6), "no derivative-integrated descent"),
    ],
)
def test_completion_applies_newton_descent_and_quadratic_model_tests(
    monkeypatch, derivatives, reason
):
    start, scripted = _scripted_derivatives(*derivatives)
    profile = _Profile(scripted)
    objective = _objective(monkeypatch, start, profile)
    completion = reml_execution._complete_pinned_log_scale(objective, _PINNED)
    assert completion == reml_execution._ScaleNewtonCompletion(0, reason)
    assert objective.n_accepted == 0
    assert objective.candidate is None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (lambda params: _trial(np.r_[40.0, params[-1]]), "changed supplied rho"),
        (lambda params: _trial(params, source_fingerprint="other"), "lineage"),
        (lambda params: _trial(params, link_name="OtherLink"), "lineage"),
    ],
)
def test_completion_refuses_candidate_with_changed_rho_or_lineage(
    monkeypatch, override, message
):
    profile = _Profile({2: override})
    objective = _objective(monkeypatch, _STATIONARY + 0.1, profile)
    with pytest.raises(RuntimeError, match=message):
        reml_execution._complete_pinned_log_scale(objective, _PINNED)


def test_optimizer_applies_completion_only_to_pinned_unknown_scale(monkeypatch):
    profile = _Profile()
    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", profile)
    monkeypatch.setattr(reml_execution, "minimize", _lbfgs_returns_initial())
    start = _STATIONARY + 1e-8
    result = optimize_regular_stream_reml(
        _optimizer_stream(),
        SimpleNamespace(scale_known=False),
        np.r_[_PINNED, start],
        maximum_bytes=1_000_000,
        pin_lambda=True,
        control=StreamREMLControl(history_size=8),
    )
    assert result.converged
    assert result.status == 0
    assert result.n_iter == 3
    assert result.scale_completion_steps >= 1
    assert result.scale_completion_stop in {"resolved", "no gradient reduction"}
    assert "pinned log-scale Newton completion" in result.message
    np.testing.assert_array_equal(np.asarray(result.trial.params)[:-1], _PINNED)
    assert abs(float(result.trial.params[-1]) - _STATIONARY) <= 4 * np.spacing(
        abs(_STATIONARY)
    )
    assert result.projected_gradient_inf == abs(float(result.trial.gradient[-1]))
    assert result.n_evaluations == len(profile.requests)
    assert result.n_accepted == result.scale_completion_steps
    assert result.cumulative_source_scans == 2 * len(profile.requests)
    assert result.accepted_score_history[-1] == float(result.trial.score)
    assert result.required_bytes < 1_000_000

    # Joint estimation keeps its existing first-order result untouched.
    joint_profile = _Profile()
    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", joint_profile)

    def joint_minimize(objective, params, **_kwargs):
        objective(params)
        return SimpleNamespace(
            x=np.asarray(params), success=True, message="ok", status=0, nit=1
        )

    monkeypatch.setattr(reml_execution, "minimize", joint_minimize)
    joint = optimize_regular_stream_reml(
        _optimizer_stream(),
        SimpleNamespace(scale_known=False),
        np.r_[1.0, start],
        maximum_bytes=1_000_000,
    )
    assert joint.scale_completion_stop is None
    assert joint.scale_completion_steps == 0
    assert len(joint_profile.requests) == 1
    assert "completion" not in joint.message


def test_optimizer_does_not_complete_a_failed_first_order_stage(monkeypatch):
    profile = _Profile()
    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", profile)
    monkeypatch.setattr(
        reml_execution, "minimize", _lbfgs_returns_initial(success=False, status=1)
    )
    result = optimize_regular_stream_reml(
        _optimizer_stream(),
        SimpleNamespace(scale_known=False),
        np.r_[_PINNED, _STATIONARY + 1e-3],
        maximum_bytes=1_000_000,
        pin_lambda=True,
    )
    assert not result.converged
    assert result.scale_completion_stop == "not run: L-BFGS-B failed"
    assert result.scale_completion_steps == 0
    assert len(profile.requests) == 1
    assert "completion" not in result.message


def test_optimizer_reports_nonpositive_scale_curvature_as_unconverged(monkeypatch):
    profile = _Profile({1: {"gradient": 1e-9, "curvature": -2.0}})
    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", profile)
    monkeypatch.setattr(reml_execution, "minimize", _lbfgs_returns_initial())
    result = optimize_regular_stream_reml(
        _optimizer_stream(),
        SimpleNamespace(scale_known=False),
        np.r_[_PINNED, _STATIONARY],
        maximum_bytes=1_000_000,
        pin_lambda=True,
    )
    assert not result.converged
    assert result.projected_gradient_inf <= StreamREMLControl().gtol
    assert result.scale_completion_stop == "nonpositive curvature"
    assert result.message.endswith("log-scale curvature is not positive.")
    assert len(profile.requests) == 1


@pytest.mark.parametrize("candidate_rejected", [False, True])
def test_optimizer_restates_gtol_on_the_completed_state(
    monkeypatch, candidate_rejected
):
    """A successful ftol-style stop is judged on the final accepted trial."""
    start = _STATIONARY + 0.01
    scripted = (
        {2: {"gradient": 2.0 * (-_A * np.exp(-start) + _B)}}
        if candidate_rejected
        else {}
    )
    profile = _Profile(scripted)
    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", profile)
    monkeypatch.setattr(reml_execution, "minimize", _lbfgs_returns_initial())
    result = optimize_regular_stream_reml(
        _optimizer_stream(),
        SimpleNamespace(scale_known=False),
        np.r_[_PINNED, start],
        maximum_bytes=1_000_000,
        pin_lambda=True,
    )
    gtol = StreamREMLControl().gtol
    if candidate_rejected:
        assert not result.converged
        assert result.scale_completion_stop == "no gradient reduction"
        assert result.projected_gradient_inf > gtol
        assert result.message.endswith(f"projected gradient exceeds gtol {gtol:.3e}.")
        assert float(result.trial.params[-1]) == start
    else:
        assert result.converged
        assert result.scale_completion_steps >= 2
        assert result.projected_gradient_inf <= gtol


def _pinned_rank_one(family_class, link):
    family, data, weight, offset, start = _case(family_class, link)
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared, rho = _rank_one_preparation(source, family)
    return family, data, weight, offset, start, source, prepared, rho


@pytest.mark.parametrize("link", ["log", "logit", "cloglog"])
def test_pinned_gaussian_scale_is_the_closed_form_stationary_scale(link):
    """For Gaussian at fixed rho, exact REML stationarity is Dp / (n - Mp)."""
    family, _data, weight, _offset, start, source, prepared, rho = _pinned_rank_one(
        Gaussian, link
    )
    result = optimize_regular_stream_reml(
        StreamDesign(prepared, source),
        family,
        np.r_[rho, 0.0],
        maximum_bytes=64_000_000,
        pin_lambda=True,
        pirls_control=StreamPIRLSControl(batch_rows=17, solver_policy="qr", tol=1e-11),
        initial_coefficients=start,
    )
    trial = result.trial
    assert result.converged
    assert result.scale_completion_steps >= 1
    assert result.scale_completion_stop in {"resolved", "no gradient reduction"}
    np.testing.assert_array_equal(np.asarray(trial.params)[:-1], rho)
    informative = int(np.count_nonzero(weight > 0))
    mp = prepared.fitting.total_penalty_null_dim
    np.testing.assert_allclose(
        float(np.asarray(trial.fit_result.state.score_scale)),
        trial.fit_result.source_score.penalized_deviance / (informative - mp),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize(
    ("family_class", "link"),
    [(Gaussian, "inverse"), (Gamma, "inverse"), (Gamma, "logit"), (Gamma, "cloglog")],
)
def test_pinned_completed_scale_is_pinned_gam_fit3_stationary(family_class, link):
    """Pinned gam.fit3's own Newton correction at JAX's scale is below STRICT."""
    family, data, weight, offset, start, source, prepared, rho = _pinned_rank_one(
        family_class, link
    )
    result = optimize_regular_stream_reml(
        StreamDesign(prepared, source),
        family,
        np.r_[rho, 0.0],
        maximum_bytes=64_000_000,
        pin_lambda=True,
        pirls_control=StreamPIRLSControl(batch_rows=17, solver_policy="qr", tol=1e-11),
        initial_coefficients=start,
    )
    trial = result.trial
    assert result.converged
    np.testing.assert_array_equal(np.asarray(trial.params)[:-1], rho)
    phi = float(np.asarray(trial.fit_result.state.score_scale))
    X = np.column_stack((np.ones(len(data)), data.x))
    oracle = _source_reference(
        family,
        link,
        X,
        data.y,
        weight,
        offset,
        start,
        derivatives=True,
        rho=rho,
        score_phi=phi,
    )
    r_gradient = float(oracle["gradient"][-1])
    assert abs(r_gradient) / trial.log_phi_curvature <= STRICT.rtol
