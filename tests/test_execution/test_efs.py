"""Dense known-scale EFS execution adapter tests."""

from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pandas as pd
import pytest

from jaxgam.execution import efs as execution_efs
from jaxgam.execution.efs import (
    EFSControl,
    _fit_state,
    dense_efs_known_scale,
    dense_efs_unknown_scale,
    efs_initial_log_lambda,
    efs_initial_log_scale,
)
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.efs import EFSRawUpdate, prepare_efs_statistics
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests.fixtures.efs_weighted_additive_cr_repro import FORMULA, make_data
from tests.helpers import _AssertCollector, r_available
from tests.r_bridge import RBridge, RBridgeError
from tests.tolerances import MODERATE, STRICT


def _build(
    formula: str,
    data: pd.DataFrame,
    family,
    *,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
):
    from jaxgam.fitting.data import FittingData

    setup = ModelSetup.build(
        parse_formula(formula), data, weights=weights, offset=offset
    )
    return setup, FittingData.from_setup(setup, family)


def _oracle_data(family_name: str, *, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, 80)
    z = rng.uniform(-1.0, 1.0, len(x))
    eta = 0.15 + 0.45 * np.sin(2.5 * x) + 0.2 * x * z
    if family_name == "poisson":
        y = rng.poisson(np.exp(eta))
    else:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _fixed_nb_data(*, seed: int = 812, n: int = 84) -> pd.DataFrame:
    """Deterministic positive-weight NB/log fixture for the pinned oracle."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    offset = 0.12 * z
    theta = 2.7
    eta = offset + 0.1 + 0.5 * np.sin(2.6 * x) - 0.18 * z
    mu = np.exp(eta)
    y = rng.negative_binomial(theta, theta / (theta + mu))
    return pd.DataFrame(
        {"y": y, "x": x, "z": z, "w": 0.5 + rng.random(n), "off": offset}
    )


@pytest.mark.parametrize("family", [Poisson(), Binomial()])
def test_dense_known_scale_efs_runs_from_one_time_shift(family) -> None:
    rng = np.random.default_rng(320)
    x = np.linspace(-1.0, 1.0, 48)
    eta = 0.2 + 0.5 * np.sin(3.0 * x)
    if family.family_name == "poisson":
        y = rng.poisson(np.exp(eta))
    else:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    _, fd = _build("y ~ s(x, bs='cr', k=6)", pd.DataFrame({"x": x, "y": y}), family)
    result = dense_efs_known_scale(fd, control=EFSControl(outer_limit=2))
    assert result.n_iter == 2
    assert result.convergence_info == "iteration limit reached"
    assert result.scale == 1.0
    assert result.score_history
    assert np.all(np.isfinite(np.asarray(result.smoothing_params)))
    assert result.update_residual is not None


def test_fixed_nb_efs_requires_extended_initial_sp_and_returns_fixed_theta() -> None:
    data = _fixed_nb_data(n=48)
    family = NegativeBinomial(theta=2.7, fixed=True)
    setup, fd = _build(
        "y ~ s(x, bs='cr', k=6)",
        data,
        family,
        weights=data["w"].to_numpy(),
        offset=data["off"].to_numpy(),
    )
    with pytest.raises(ValueError, match="efs_initial_log_lambda"):
        dense_efs_known_scale(fd, control=EFSControl(outer_limit=1))
    result = dense_efs_known_scale(
        fd,
        initial_log_lambda=efs_initial_log_lambda(setup, family),
        control=EFSControl(outer_limit=1),
    )
    assert result.scale == 1.0
    assert result.theta == pytest.approx(2.7)
    assert result.smoothing_params.shape == (fd.n_penalties,)
    assert np.all(np.isfinite(np.asarray(result.pirls_result.XtWX)))
    assert np.all(np.isfinite(np.asarray(result.pirls_result.XtWX_fisher)))


def test_known_scale_efs_rejects_nonlog_nb() -> None:
    data = _fixed_nb_data(n=40)
    family = NegativeBinomial(theta=2.7, fixed=True, link="identity")
    setup, fd = _build("y ~ s(x, bs='cr', k=5)", data, family)
    with pytest.raises(NotImplementedError, match="NB/log"):
        dense_efs_known_scale(
            fd,
            initial_log_lambda=efs_initial_log_lambda(setup, family),
            control=EFSControl(outer_limit=1),
        )


def test_estimated_nb_efs_requires_explicit_staged_start_contract() -> None:
    data = _fixed_nb_data(n=40)
    family = NegativeBinomial(theta=2.7)
    setup, fd = _build("y ~ s(x, bs='cr', k=5)", data, family)
    initial = efs_initial_log_lambda(setup, family)
    with pytest.raises(ValueError, match="explicit beta_init"):
        dense_efs_known_scale(
            fd, initial_log_lambda=initial, control=EFSControl(outer_limit=1)
        )
    with pytest.raises(ValueError, match="beta_old_init"):
        dense_efs_known_scale(
            fd,
            initial_log_lambda=initial,
            beta_init=jax.numpy.zeros((fd.n_coef,)),
            control=EFSControl(outer_limit=1),
        )


def test_estimated_nb_efs_keeps_theta_explicit_and_family_unchanged() -> None:
    data = _fixed_nb_data(n=48)
    family = NegativeBinomial(theta=2.7)
    setup, fd = _build(
        "y ~ s(x, bs='cr', k=6)",
        data,
        family,
        weights=data["w"].to_numpy(),
        offset=data["off"].to_numpy(),
    )
    beta0 = jax.numpy.zeros((fd.n_coef,), dtype=fd.X.dtype)
    theta0 = jax.numpy.asarray(family.get_theta(transformed=False))
    result = dense_efs_known_scale(
        fd,
        initial_log_lambda=efs_initial_log_lambda(setup, family),
        initial_log_theta=theta0,
        beta_init=beta0,
        beta_old_init=beta0,
        control=EFSControl(outer_limit=4),
    )
    assert result.theta is not None
    assert np.isfinite(result.theta)
    assert result.scale == 1.0
    np.testing.assert_array_equal(
        family.get_theta(transformed=False), np.asarray(theta0)
    )
    assert np.all(np.isfinite(np.asarray(result.pirls_result.XtWX_fisher)))


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_pinned_nb_efs_trace_pins_explicit_theta_and_old_state_starts() -> None:
    """The private EFS trace records selected theta, not mutable family residue."""
    data = _fixed_nb_data(n=48)
    family = NegativeBinomial(theta=2.7)
    formula = "y ~ s(x, bs='cr', k=6)"
    setup, fd = _build(
        formula,
        data,
        family,
        weights=data["w"].to_numpy(),
        offset=data["off"].to_numpy(),
    )
    beta0 = jax.numpy.zeros((fd.n_coef,), dtype=fd.X.dtype)
    theta0 = jax.numpy.asarray(family.get_theta(transformed=False))
    rho = efs_initial_log_lambda(setup, family)
    r_trace = RBridge(mode="subprocess").efs_diagnostics(
        formula,
        data,
        "nb",
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
        scale=1.0,
        initial_log_theta=float(np.asarray(theta0[0])),
        initial_beta=np.asarray(beta0),
        beta_old_init=np.asarray(beta0),
        controls={"efs_tol": 1e-12},
    )
    j_initial = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        rho + 2.5,
        beta0,
        EFSControl(),
        log_theta_start=theta0,
        beta_old_init=beta0,
    )
    np.testing.assert_allclose(
        np.asarray(j_initial.pirls_result.mu),
        r_trace["fitted_values"][0],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    assert j_initial.log_theta is not None
    np.testing.assert_allclose(
        np.asarray(j_initial.log_theta[0]),
        r_trace["theta_trace"][0, 1],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    starts = r_trace["starts"]
    theta_trace = r_trace["theta_trace"]
    np.testing.assert_allclose(starts[0], beta0, rtol=STRICT.rtol, atol=STRICT.atol)
    np.testing.assert_allclose(
        theta_trace[0, 0], theta0[0], rtol=STRICT.rtol, atol=STRICT.atol
    )
    # Away from an extension/contraction, the pinned private trace advances
    # both warm components from the immediately selected fit.
    np.testing.assert_allclose(starts[1:], r_trace["coefficients"][:-1])
    np.testing.assert_allclose(theta_trace[1:, 0], theta_trace[:-1, 1])


def test_estimated_nb_controller_keeps_old_beta_and_theta_for_all_trials(
    monkeypatch,
) -> None:
    """Candidate, extension, and contraction cannot leak a trial theta."""
    data = _fixed_nb_data(n=48)
    family = NegativeBinomial(theta=2.7)
    setup, fd = _build(
        "y ~ s(x, bs='cr', k=6)",
        data,
        family,
        weights=data["w"].to_numpy(),
        offset=data["off"].to_numpy(),
    )
    beta0 = jax.numpy.zeros((fd.n_coef,), dtype=fd.X.dtype)
    theta0 = jax.numpy.asarray(family.get_theta(transformed=False))
    rho = efs_initial_log_lambda(setup, family)
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        rho + 2.5,
        beta0,
        EFSControl(),
        log_theta_start=theta0,
        beta_old_init=beta0,
    )
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    scores = iter([10.0, 9.0, 8.0, 11.0, 10.5])
    output_thetas = iter([0.1, 0.2, 0.3, 0.4, 0.5])

    def scripted(_fd, _plan, rho, beta, _control, **kwargs):
        incoming_theta = kwargs["log_theta_start"]
        calls.append((np.asarray(beta), np.asarray(incoming_theta)))
        number = len(calls)
        return replace(
            base,
            log_lambda=rho,
            pirls_result=replace(
                base.pirls_result,
                coefficients=jax.numpy.full_like(beta, number),
                deviance=jax.numpy.asarray(20.0 + number),
            ),
            score=jax.numpy.asarray(next(scores)),
            log_theta=jax.numpy.asarray([next(output_thetas)]),
            theta_status=jax.numpy.asarray(0),
            valid=True,
            inner_converged=True,
        )

    monkeypatch.setattr(execution_efs, "_fit_state", scripted)
    monkeypatch.setattr(
        execution_efs,
        "efs_raw_update",
        lambda rho, _stats, phi, multiplier, _cap: EFSRawUpdate(
            jax.numpy.ones_like(rho),
            jax.numpy.exp(jax.numpy.full_like(rho, 0.01)),
            rho + 0.01 * multiplier,
            phi == 1,
        ),
    )
    result = dense_efs_known_scale(
        fd,
        initial_log_lambda=rho,
        initial_log_theta=theta0,
        beta_init=beta0,
        beta_old_init=beta0,
        control=EFSControl(outer_limit=2),
    )
    assert len(calls) == 5
    np.testing.assert_array_equal(calls[1][0], np.full(fd.n_coef, 1.0))
    np.testing.assert_array_equal(calls[2][0], np.full(fd.n_coef, 1.0))
    np.testing.assert_array_equal(calls[3][0], np.full(fd.n_coef, 3.0))
    np.testing.assert_array_equal(calls[4][0], np.full(fd.n_coef, 3.0))
    np.testing.assert_allclose(
        [call[1][0] for call in calls], [theta0[0], 0.1, 0.1, 0.3, 0.3]
    )
    assert result.theta == pytest.approx(np.exp(0.5))
    assert result.scale == 1.0


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_estimated_nb_efs_rejects_the_pinned_r_start_reset_path() -> None:
    """Do not pretend a discarded R warm start is a valid EFS5c start."""
    data = _fixed_nb_data(n=48)
    family = NegativeBinomial(theta=2.7)
    formula = "y ~ s(x, bs='cr', k=6)"
    setup, fd = _build(
        formula,
        data,
        family,
        weights=data["w"].to_numpy(),
        offset=data["off"].to_numpy(),
    )
    beta_old = jax.numpy.zeros((fd.n_coef,), dtype=fd.X.dtype)
    beta_bad = jax.numpy.full((fd.n_coef,), 30.0, dtype=fd.X.dtype)
    theta0 = jax.numpy.asarray(family.get_theta(transformed=False))
    rho = efs_initial_log_lambda(setup, family)
    r_trace = RBridge(mode="subprocess").efs_diagnostics(
        formula,
        data,
        "nb",
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
        scale=1.0,
        initial_log_theta=float(np.asarray(theta0[0])),
        initial_beta=np.asarray(beta_bad),
        beta_old_init=np.asarray(beta_old),
    )
    assert not r_trace["start_retained"][0]
    with pytest.raises(ValueError, match="would reset to mustart"):
        dense_efs_known_scale(
            fd,
            initial_log_lambda=rho,
            initial_log_theta=theta0,
            beta_init=beta_bad,
            beta_old_init=beta_old,
            control=EFSControl(outer_limit=1),
        )


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_estimated_nb_efs_controller_matches_pinned_matched_start_trace() -> None:
    """Internal EFS5d parity is only claimed for the explicit admitted start."""
    data = _fixed_nb_data(n=48)
    family = NegativeBinomial(theta=2.7)
    formula = "y ~ s(x, bs='cr', k=6)"
    setup, fd = _build(
        formula,
        data,
        family,
        weights=data["w"].to_numpy(),
        offset=data["off"].to_numpy(),
    )
    beta0 = jax.numpy.zeros((fd.n_coef,), dtype=fd.X.dtype)
    theta0 = jax.numpy.asarray(family.get_theta(transformed=False))
    rho = efs_initial_log_lambda(setup, family)
    r_trace = RBridge(mode="subprocess").efs_diagnostics(
        formula,
        data,
        "nb",
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
        scale=1.0,
        initial_log_theta=float(np.asarray(theta0[0])),
        initial_beta=np.asarray(beta0),
        beta_old_init=np.asarray(beta0),
    )
    j_fit = dense_efs_known_scale(
        fd,
        initial_log_lambda=rho,
        initial_log_theta=theta0,
        beta_init=beta0,
        beta_old_init=beta0,
    )
    selected_sp = r_trace["selected_packed_sp"]
    assert selected_sp.shape == (fd.n_penalties + 1,)
    collector = _AssertCollector()
    collector.check(
        "coefficients",
        lambda: np.testing.assert_allclose(
            penalty_ops.transform_coefficients(
                fd.penalty_structure, j_fit.pirls_result.coefficients
            ),
            r_trace["selected_coefficients"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "fitted values",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.mu,
            r_trace["selected_fitted_values"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "deviance",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.deviance,
            r_trace["selected_deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "criterion",
        lambda: np.testing.assert_allclose(
            j_fit.score,
            r_trace["final_score"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "selected theta",
        lambda: np.testing.assert_allclose(
            j_fit.theta,
            selected_sp[0],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "smoothing parameters",
        lambda: np.testing.assert_allclose(
            j_fit.smoothing_params,
            selected_sp[1:],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any("estimated-NB EFS matched-start parity")


def test_efs_bridge_fixed_nb_theta_requires_positive_nb_family() -> None:
    bridge = RBridge(mode="subprocess")
    assert bridge._get_efs_subprocess_family("nb", 2.7) == "nb(theta=2.7)"
    with pytest.raises(ValueError, match="finite and positive"):
        bridge._get_efs_subprocess_family("nb", 0.0)
    with pytest.raises(ValueError, match="only for family='nb'"):
        bridge._get_efs_subprocess_family("poisson", 2.7)


def test_fixed_nb_initial_sp_uses_global_fisher_fallback_across_batches() -> None:
    """A zero after row 8192 switches all of R's initial.spg to Fisher."""
    n = 8193
    x = np.linspace(-1.0, 1.0, n)
    y = np.ones(n)
    y[-1] = 0.0  # NB's mustart=1/6 gives a negative observed start weight.
    weights = 0.6 + 0.2 * (x + 1.0)
    data = pd.DataFrame({"y": y, "x": x})
    family = NegativeBinomial(theta=2.7, fixed=True)
    setup, _ = _build("y ~ s(x, bs='cr', k=5)", data, family, weights=weights)
    mu = np.asarray(family.initialize(setup.y, setup.weights), dtype=np.float64)
    eta = np.asarray(family.link.link(mu), dtype=np.float64)
    mu_eta = np.asarray(family.link.mu_eta(eta), dtype=np.float64)
    fisher = setup.weights * mu_eta**2 / np.asarray(family.variance(mu))
    expected = FittingData._initial_sp_from_crossproduct_diag(
        setup.X,
        setup.penalties,
        np.sum(fisher[:, None] * setup.X**2, axis=0),
    )
    np.testing.assert_allclose(
        efs_initial_log_lambda(setup, family),
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_fixed_nb_instances_reuse_dynamic_theta_pirls_compilation() -> None:
    """Equivalent fixed-NB instances preserve the extended PIRLS cache."""
    data = _fixed_nb_data(n=47)
    formula = "y ~ s(x, bs='cr', k=5)"
    first_family = NegativeBinomial(theta=2.7, fixed=True)
    second_family = NegativeBinomial(theta=2.7, fixed=True)
    _, first = _build(formula, data, first_family)
    _, second = _build(formula, data, second_family)
    penalty = jax.numpy.eye(first.n_coef) * 0.2
    beta = jax.numpy.zeros(first.n_coef)
    first_result = pirls_loop(
        first.X,
        first.y,
        beta,
        penalty,
        first_family,
        first.wt,
        first.offset,
        extended_observed=True,
    )
    cache_after_first = pirls_loop._cache_size()
    second_result = pirls_loop(
        second.X,
        second.y,
        beta,
        penalty,
        second_family,
        second.wt,
        second.offset,
        extended_observed=True,
    )
    assert pirls_loop._cache_size() == cache_after_first
    np.testing.assert_allclose(first_result.coefficients, second_result.coefficients)
    assert first_family.get_theta(transformed=True)[0] == pytest.approx(2.7)
    assert second_family.get_theta(transformed=True)[0] == pytest.approx(2.7)


def test_dense_efs_rejects_out_of_scope_family_before_any_newton_dispatch() -> None:
    x = np.linspace(0.0, 1.0, 20)
    _, fd = _build(
        "y ~ s(x, bs='cr', k=5)",
        pd.DataFrame({"x": x, "y": x}),
        Gaussian(),
    )
    with pytest.raises(NotImplementedError, match="Poisson/log"):
        dense_efs_known_scale(fd)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outer_limit": True},
        {"pirls_max_iter": 1.5},
        {"history_limit": 0},
        {"score_tolerance": np.nan},
        {"pirls_tolerance": np.inf},
    ],
)
def test_efs_control_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError, match="EFS"):
        EFSControl(**kwargs)


def test_production_controller_extension_and_contraction_keep_old_warm_start(
    monkeypatch,
) -> None:
    """The actual host controller, not the EFS1 oracle, owns trial state."""
    data = _oracle_data("poisson")
    _, fd = _build("y ~ s(x, bs='cr', k=6)", data, Poisson())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
    )
    scores = iter([10.0, 9.0, 8.0, 11.0, 10.5])
    starts: list[np.ndarray] = []
    states = []

    def scripted(_fd, _plan, rho, beta, _control):
        starts.append(np.asarray(beta))
        fit = replace(
            base.pirls_result,
            coefficients=jax.numpy.full_like(beta, len(starts)),
            deviance=jax.numpy.asarray(20.0 + len(starts)),
        )
        states.append(fit)
        return execution_efs.EFSFitState(
            rho,
            fit,
            jax.numpy.asarray(next(scores)),
            base.edf,
            base.statistics,
            None,
            True,
            True,
        )

    def update(rho, statistics, phi, multiplier, cap):
        del statistics, phi
        ratio = jax.numpy.exp(jax.numpy.ones_like(rho) * 0.01)
        trial = jax.numpy.minimum(rho + 0.01 * multiplier, cap)
        return EFSRawUpdate(
            jax.numpy.ones_like(rho), ratio, trial, jax.numpy.array(True)
        )

    monkeypatch.setattr(execution_efs, "_fit_state", scripted)
    monkeypatch.setattr(execution_efs, "efs_raw_update", update)
    result = execution_efs.dense_efs_known_scale(fd, control=EFSControl(outer_limit=2))
    assert result.convergence_info == "iteration limit reached"
    assert result.multiplier == 1.0
    # Every candidate/extension/contraction starts from the accepted beta;
    # no rejected trial can become a warm start.
    assert len(starts) == 5
    for index in (1, 2):
        np.testing.assert_array_equal(starts[index], states[0].coefficients)
    for index in (3, 4):
        np.testing.assert_array_equal(starts[index], states[2].coefficients)
    assert result.score_history == (8.0, 10.5)  # finite worsening at the floor
    np.testing.assert_array_equal(
        result.pirls_result.coefficients, states[4].coefficients
    )


def test_production_controller_preserves_failure_at_iteration_boundary(
    monkeypatch,
) -> None:
    data = _oracle_data("poisson")
    _, fd = _build("y ~ s(x, bs='cr', k=6)", data, Poisson())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
    )

    calls = 0

    def invalid_trial(_fd, _plan, rho, _beta, _control):
        nonlocal calls
        calls += 1
        if calls == 1:
            return base
        return execution_efs.EFSFitState(
            rho,
            base.pirls_result,
            base.score,
            base.edf,
            base.statistics,
            None,
            False,
            False,
        )

    monkeypatch.setattr(execution_efs, "_fit_state", invalid_trial)
    result = execution_efs.dense_efs_known_scale(fd, control=EFSControl(outer_limit=1))
    assert result.convergence_info == "inner_failure"
    assert calls == 2
    assert result.n_iter == 1
    assert not result.converged


def test_production_controller_losing_extension_keeps_multiplier(monkeypatch) -> None:
    data = _oracle_data("poisson")
    _, fd = _build("y ~ s(x, bs='cr', k=6)", data, Poisson())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
    )
    scores = iter([10.0, 9.0, 9.1])
    calls = []

    def scripted(_fd, _plan, rho, _beta, _control):
        calls.append(np.asarray(rho))
        return execution_efs.EFSFitState(
            rho,
            base.pirls_result,
            jax.numpy.asarray(next(scores)),
            base.edf,
            base.statistics,
            None,
            True,
            True,
        )

    monkeypatch.setattr(execution_efs, "_fit_state", scripted)
    monkeypatch.setattr(
        execution_efs,
        "efs_raw_update",
        lambda rho, _stats, _phi, multiplier, _cap: EFSRawUpdate(
            jax.numpy.ones_like(rho),
            jax.numpy.exp(jax.numpy.full_like(rho, 0.01)),
            rho + 0.01 * multiplier,
            jax.numpy.array(True),
        ),
    )
    result = execution_efs.dense_efs_known_scale(fd, control=EFSControl(outer_limit=1))
    assert len(calls) == 3
    np.testing.assert_array_equal(result.log_lambda, calls[1])
    assert result.multiplier == 1.0
    assert result.convergence_info == "iteration limit reached"


def test_production_controller_reports_invalid_raw_update(monkeypatch) -> None:
    data = _oracle_data("poisson")
    _, fd = _build("y ~ s(x, bs='cr', k=6)", data, Poisson())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
    )
    monkeypatch.setattr(execution_efs, "_fit_state", lambda *_: base)
    monkeypatch.setattr(
        execution_efs,
        "efs_raw_update",
        lambda rho, *_: EFSRawUpdate(
            jax.numpy.ones_like(rho),
            jax.numpy.ones_like(rho),
            rho,
            jax.numpy.array(False),
        ),
    )
    result = execution_efs.dense_efs_known_scale(fd, control=EFSControl(outer_limit=1))
    assert result.convergence_info == "invalid_update"


@pytest.mark.parametrize(
    ("mode", "iterations", "reason"),
    [
        ("score", 4, "score_window"),
        ("deviance", 2, "deviance_change"),
        ("limit", 200, "iteration limit reached"),
    ],
)
def test_production_controller_named_stops_and_iteration_200(
    monkeypatch,
    mode,
    iterations,
    reason,
) -> None:
    _, fd = _build("y ~ s(x,bs='cr',k=6)", _oracle_data("poisson"), Poisson())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
    )
    calls = 0

    def scripted(_fd, _plan, rho, _beta, _control):
        nonlocal calls
        calls += 1
        deviance = 10.0 if mode == "deviance" else float(calls)
        # Worsening at multiplier=1 forces exactly one accepted trial per
        # iteration; a .01 score increment stays inside the score window.
        return replace(
            base,
            log_lambda=rho,
            score=jax.numpy.asarray(10.0 + 0.01 * calls),
            pirls_result=replace(
                base.pirls_result, deviance=jax.numpy.asarray(deviance)
            ),
        )

    displacement = 0.01 if mode == "score" else 0.1
    monkeypatch.setattr(execution_efs, "_fit_state", scripted)
    monkeypatch.setattr(
        execution_efs,
        "efs_raw_update",
        lambda rho, _stats, _phi, multiplier, _cap: EFSRawUpdate(
            jax.numpy.ones_like(rho),
            jax.numpy.exp(jax.numpy.full_like(rho, displacement)),
            rho + displacement * multiplier,
            jax.numpy.array(True),
        ),
    )
    result = dense_efs_known_scale(fd, control=EFSControl(history_limit=3))
    assert result.n_iter == iterations
    assert calls == iterations + 1
    assert result.convergence_info == reason
    assert result.converged == (mode != "limit")
    assert len(result.score_history) == min(iterations, 3)


def test_invalid_initial_fisher_factor_is_not_an_inner_failure(monkeypatch) -> None:
    _, fd = _build("y ~ s(x,bs='cr',k=6)", _oracle_data("poisson"), Poisson())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
    )
    bad_fit = replace(
        base.pirls_result,
        L_fisher=jax.numpy.full_like(base.pirls_result.L_fisher, jax.numpy.nan),
    )
    monkeypatch.setattr(execution_efs, "pirls_loop", lambda *_args, **_kwargs: bad_fit)
    result = dense_efs_known_scale(fd)
    assert result.convergence_info == "invalid_initial"
    assert not result.converged


@pytest.mark.parametrize("weight", [0.0, 1e-11, 1e11, np.nan])
def test_efs_preflight_excludes_unvalidated_prior_weight_edges(weight) -> None:
    _, fd = _build("y ~ s(x,bs='cr',k=6)", _oracle_data("poisson"), Poisson())
    with pytest.raises(ValueError, match="clipping bounds"):
        dense_efs_known_scale(replace(fd, wt=fd.wt.at[0].set(weight)))


def test_existing_efs_statistics_kernel_compiles_and_executes() -> None:
    # This regression belongs near the execution adapter because every outer
    # proposal consumes its JIT statistics kernel.
    from jaxgam.fitting.efs import EFSStatistics, efs_raw_update

    stats = EFSStatistics(
        jax.numpy.array([1.0]),
        jax.numpy.array([0.25]),
        jax.numpy.array([1.0]),
        jax.numpy.array(True),
        jax.numpy.array(True),
    )
    result = jax.jit(efs_raw_update)(
        jax.numpy.array([0.0]),
        stats,
        jax.numpy.array(1.0),
        jax.numpy.array(1.0),
        jax.numpy.array(15.0),
    )
    assert bool(result.finite_positive)


def test_unknown_scale_initialization_uses_unweighted_mean_and_original_n() -> None:
    """``get.null.coef`` is not the public weighted null-deviance path."""
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 2.0, 9.0]})
    weights = np.array([1.0, 3.0, 2.0])
    setup = ModelSetup.build(parse_formula("y ~ s(x, k=3)"), data, weights=weights)
    observed = float(np.exp(efs_initial_log_scale(setup, Gaussian())))
    mean = float(np.mean(data["y"]))
    expected = float(np.sum(weights * (data["y"] - mean) ** 2) / len(data) / 10.0)
    np.testing.assert_allclose(observed, expected, rtol=STRICT.rtol, atol=STRICT.atol)

    constant = data.assign(y=2.0)
    constant_setup = ModelSetup.build(
        parse_formula("y ~ s(x, k=3)"), constant, weights=weights
    )
    with pytest.raises(ValueError, match=r"null\.scale / 10"):
        efs_initial_log_scale(constant_setup, Gaussian())


def test_unknown_scale_extension_uses_old_accepted_phi(monkeypatch) -> None:
    """A losing extension cannot leak its Fletcher scale into the next fit."""
    data = _oracle_data("poisson").assign(y=lambda frame: frame["x"] ** 2)
    _, fd = _build("y ~ s(x, bs='cr', k=6)", data, Gaussian())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
        jax.numpy.array(0.05),
    )
    calls: list[float] = []
    ratio_phi: list[float] = []
    outcomes = iter([(10.0, 0.2), (9.0, 0.3), (9.1, 99.0), (8.0, 0.4), (8.1, 98.0)])

    def scripted(_fd, _plan, rho, _beta, _control, score_phi=None):
        calls.append(float(np.asarray(score_phi)))
        score, update_phi = next(outcomes)
        return replace(
            base,
            log_lambda=rho,
            score=jax.numpy.asarray(score),
            score_phi=jax.numpy.asarray(score_phi),
            update_phi=jax.numpy.asarray(update_phi),
            reported_phi=jax.numpy.asarray(update_phi),
            carried_phi=jax.numpy.asarray(update_phi),
            valid=True,
            inner_converged=True,
        )

    def raw(rho, _statistics, phi, multiplier, cap):
        ratio_phi.append(float(np.asarray(phi)))
        ratio = jax.numpy.exp(jax.numpy.full_like(rho, 0.01))
        return EFSRawUpdate(
            jax.numpy.ones_like(rho),
            ratio,
            jax.numpy.minimum(rho + 0.01 * multiplier, cap),
            jax.numpy.array(True),
        )

    monkeypatch.setattr(execution_efs, "_fit_state", scripted)
    monkeypatch.setattr(execution_efs, "efs_raw_update", raw)
    result = dense_efs_unknown_scale(
        fd,
        initial_log_scale=jax.numpy.log(jax.numpy.array(0.05)),
        control=EFSControl(outer_limit=2),
    )
    assert result.convergence_info == "iteration limit reached"
    # Initial, candidate, losing extension, then next candidate/extension.
    np.testing.assert_allclose(calls, [0.05, 0.2, 0.2, 0.3, 0.3])
    np.testing.assert_allclose(ratio_phi, [0.2, 0.3])


def test_unknown_scale_contraction_reuses_old_phi_and_rejects_invalid_input(
    monkeypatch,
) -> None:
    """Contraction starts from accepted nuisance state, never a trial's scale."""
    data = _oracle_data("poisson").assign(y=lambda frame: frame["x"] ** 2)
    _, fd = _build("y ~ s(x, bs='cr', k=6)", data, Gaussian())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
        jax.numpy.array(0.05),
    )
    calls: list[float] = []
    ratio_phi: list[float] = []
    # First extension wins (multiplier becomes 2). Pinned R carries the first
    # candidate's phi=.3 into the next score, but the next EFS ratio uses the
    # selected extension's phi=.4. Its contraction starts from that carried
    # candidate phi, rather than the rejected trial's .5.
    outcomes = iter([(10.0, 0.2), (9.0, 0.3), (8.0, 0.4), (11.0, 0.5), (12.0, 77.0)])

    def scripted(_fd, _plan, rho, _beta, _control, score_phi=None):
        calls.append(float(np.asarray(score_phi)))
        score, update_phi = next(outcomes)
        return replace(
            base,
            log_lambda=rho,
            score=jax.numpy.asarray(score),
            score_phi=jax.numpy.asarray(score_phi),
            update_phi=jax.numpy.asarray(update_phi),
            reported_phi=jax.numpy.asarray(update_phi),
            carried_phi=jax.numpy.asarray(update_phi),
            valid=True,
            inner_converged=True,
        )

    def raw(rho, _statistics, phi, multiplier, cap):
        ratio_phi.append(float(np.asarray(phi)))
        ratio = jax.numpy.exp(jax.numpy.full_like(rho, 0.01))
        return EFSRawUpdate(
            jax.numpy.ones_like(rho),
            ratio,
            jax.numpy.minimum(rho + 0.01 * multiplier, cap),
            jax.numpy.array(True),
        )

    monkeypatch.setattr(execution_efs, "_fit_state", scripted)
    monkeypatch.setattr(execution_efs, "efs_raw_update", raw)
    result = dense_efs_unknown_scale(
        fd,
        initial_log_scale=jax.numpy.log(jax.numpy.array(0.05)),
        control=EFSControl(outer_limit=2),
    )
    assert result.multiplier == 1.0
    np.testing.assert_allclose(calls, [0.05, 0.2, 0.2, 0.3, 0.3])
    np.testing.assert_allclose(ratio_phi, [0.2, 0.4])
    with pytest.raises(ValueError, match="initial unknown scale"):
        dense_efs_unknown_scale(fd, initial_log_scale=jax.numpy.array(np.nan))


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_pinned_unknown_scale_winning_extension_has_split_phi_timing() -> None:
    """Record the pinned efsudr candidate/extension full-lsp quirk directly."""
    rng = np.random.default_rng(3)
    x = np.linspace(-1.0, 1.0, 50)
    data = pd.DataFrame(
        {"x": x, "y": 0.2 + 0.5 * np.sin(2.0 * x) + rng.normal(0.0, 0.15, len(x))}
    )
    trace = RBridge(mode="subprocess").efs_diagnostics(
        "y ~ s(x, bs='cr', k=6)",
        data,
        "gaussian",
        controls={"efs_tol": 1e-20},
    )["statistics"]
    # Calls 13/14 are the first-candidate/extension pair. efsudr accepts 14,
    # but call 15 carries call 13's update scale (not call 14's) into scoring.
    candidate = trace.query("call == 13").iloc[0]
    extension = trace.query("call == 14").iloc[0]
    next_fit = trace.query("call == 15").iloc[0]
    np.testing.assert_allclose(extension["score_phi"], candidate["score_phi"])
    assert extension["update_phi"] != candidate["update_phi"]
    np.testing.assert_allclose(next_fit["score_phi"], candidate["update_phi"])


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_pinned_efs_extension_loss_returns_selected_fit_not_last_trial() -> None:
    """A rejected extension leaves the mutable family/trial trace stale."""
    rng = np.random.default_rng(0)
    x = np.linspace(-1.0, 1.0, 50)
    data = pd.DataFrame(
        {
            "x": x,
            "y": 0.2 + 0.5 * np.sin(2.0 * x) + rng.normal(0.0, 0.15, len(x)),
        }
    )
    diagnostic = RBridge(mode="subprocess").efs_diagnostics(
        "y ~ s(x, bs='cr', k=6)",
        data,
        "gaussian",
        controls={"efs_tol": 1e-20},
    )
    assert "extension_lost" in diagnostic["branches"]
    last_call = diagnostic["statistics"]["call"].max()
    last_deviance = (
        diagnostic["statistics"]
        .loc[diagnostic["statistics"]["call"] == last_call, "deviance"]
        .iloc[0]
    )
    # The selected fit is the efsudr return value. The last private gam.fit3
    # trace belongs to a rejected extension in this deterministic fixture.
    assert (
        np.max(
            np.abs(diagnostic["selected_coefficients"] - diagnostic["coefficients"][-1])
        )
        > 1e-8
    )
    assert (
        np.max(
            np.abs(
                diagnostic["selected_fitted_values"] - diagnostic["fitted_values"][-1]
            )
        )
        > 1e-8
    )
    assert abs(diagnostic["selected_deviance"] - last_deviance) > 1e-8


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_gamma_inverse_pinned_default_requires_valid_null_coef() -> None:
    """Pin mgcv 1.9-3's EFS null-coefficient omission and supported override."""
    rng = np.random.default_rng(0)
    x = np.linspace(-1.0, 1.0, 50)
    data = pd.DataFrame(
        {"x": x, "y": rng.gamma(50.0, (2.0 + 0.2 * np.sin(3.0 * x)) / 50.0)}
    )
    family = Gamma()
    formula = "y ~ s(x, bs='cr', k=5)"
    setup, _ = _build(formula, data, family)
    rho = efs_initial_log_lambda(setup, family)
    phi = float(np.exp(efs_initial_log_scale(setup, family)))
    bridge = RBridge(mode="subprocess")
    with pytest.raises(RBridgeError, match=r"pdev - old\.pdev"):
        bridge.fit_efs(
            formula,
            data,
            "gamma",
            initial_smoothing=np.exp(np.asarray(rho)),
            initial_scale=phi,
        )
    fit = bridge.fit_efs(
        formula,
        data,
        "gamma",
        initial_smoothing=np.exp(np.asarray(rho)),
        initial_scale=phi,
        null_coef=True,
    )
    assert np.isfinite(fit["reml_score"])
    assert fit["scale"] > 0


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_unknown_scale_gaussian_efs_keeps_score_phi_separate_from_fletcher() -> None:
    """Pinned EFS trial scale timing with real weights and offsets."""
    rng = np.random.default_rng(911)
    x = np.linspace(-1.0, 1.0, 70)
    z = rng.uniform(-1.0, 1.0, len(x))
    offset = 0.15 * z
    weights = 0.5 + rng.uniform(size=len(x))
    y = (
        offset
        + 0.3
        + 0.4 * np.sin(2.4 * x)
        + 0.15 * z
        + rng.normal(scale=0.12, size=len(x))
    )
    data = pd.DataFrame({"x": x, "z": z, "y": y, "w": weights, "off": offset})
    formula = "y ~ s(x, bs='cr', k=6) + s(z, bs='cr', k=5)"
    setup, fd = _build(formula, data, Gaussian(), weights=weights, offset=offset)
    rho = efs_initial_log_lambda(setup, Gaussian())
    log_phi = efs_initial_log_scale(setup, Gaussian())
    bridge = RBridge(mode="subprocess")
    diagnostic = bridge.efs_diagnostics(
        formula,
        data,
        "gaussian",
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
        initial_scale=float(np.exp(log_phi)),
    )
    r_fit = bridge.fit_efs(
        formula,
        data,
        "gaussian",
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
        initial_scale=float(np.exp(log_phi)),
    )
    j_fit = dense_efs_unknown_scale(
        fd, initial_log_lambda=rho, initial_log_scale=log_phi
    )
    first = diagnostic["statistics"].query("call == 1").sort_values("parameter")
    collector = _AssertCollector()
    collector.check(
        "initial score phi",
        lambda: np.testing.assert_allclose(
            np.exp(log_phi),
            first["score_phi"].iloc[0],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "initial update phi",
        lambda: np.testing.assert_allclose(
            _fit_state(
                fd,
                prepare_efs_statistics(fd),
                rho + 2.5,
                fd.beta_init,
                EFSControl(),
                np.exp(log_phi),
            ).update_phi,
            first["update_phi"].iloc[0],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "fitted values",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.mu,
            r_fit["fitted_values"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "deviance",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.deviance,
            r_fit["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "criterion at incoming phi",
        lambda: np.testing.assert_allclose(
            j_fit.score, r_fit["reml_score"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        "reported Fletcher scale",
        lambda: np.testing.assert_allclose(
            j_fit.scale, r_fit["scale"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.raise_if_any("unknown-scale Gaussian EFS parity")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    ("family", "family_name", "null_coef"),
    [(Gamma(), "gamma", True), (Gamma("log"), "gamma_log", False)],
)
def test_unknown_scale_gamma_efs_matches_pinned_r(
    family, family_name: str, null_coef: bool
) -> None:
    """Gamma/inverse needs R's valid null coefficient; Gamma/log needs neither."""
    rng = np.random.default_rng(4)
    x = np.linspace(-1.0, 1.0, 60)
    eta = 0.5 + 0.2 * np.sin(3.0 * x)
    data = pd.DataFrame({"x": x, "y": rng.gamma(15.0, np.exp(eta) / 15.0)})
    formula = "y ~ s(x, bs='cr', k=6)"
    setup, fd = _build(formula, data, family)
    rho = efs_initial_log_lambda(setup, family)
    j_fit = dense_efs_unknown_scale(
        fd,
        initial_log_lambda=rho,
        initial_log_scale=efs_initial_log_scale(setup, family),
    )
    r_fit = RBridge(mode="subprocess").fit_efs(
        formula,
        data,
        family_name,
        initial_smoothing=np.exp(np.asarray(rho)),
        initial_scale=float(np.exp(efs_initial_log_scale(setup, family))),
        null_coef=null_coef,
    )
    collector = _AssertCollector()
    collector.check(
        "smoothing parameters",
        lambda: np.testing.assert_allclose(
            j_fit.smoothing_params,
            r_fit["smoothing_params"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "criterion",
        lambda: np.testing.assert_allclose(
            j_fit.score, r_fit["reml_score"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        "reported scale",
        lambda: np.testing.assert_allclose(
            j_fit.scale, r_fit["scale"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.raise_if_any(f"unknown-scale {family_name} EFS parity")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    ("family", "family_name"), [(Poisson(), "poisson"), (Binomial(), "binomial")]
)
def test_known_scale_efs_matches_pinned_r_from_matched_initial_state(
    family, family_name: str
) -> None:
    """Real optimizer parity, not a default-Newton substitute."""
    formula = "y ~ s(x, bs='cr', k=6)"
    data = _oracle_data(family_name)
    setup, fd = _build(formula, data, family)
    rho = efs_initial_log_lambda(setup, family)
    oracle = RBridge(mode="subprocess")
    r_fit = oracle.fit_efs(
        formula, data, family_name, initial_smoothing=np.exp(np.asarray(rho))
    )
    j_fit = dense_efs_known_scale(fd, initial_log_lambda=rho)
    collector = _AssertCollector()
    collector.check(
        "fitted values",
        lambda: np.testing.assert_allclose(
            np.asarray(j_fit.pirls_result.mu),
            r_fit["fitted_values"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "deviance",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.deviance,
            r_fit["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "criterion",
        lambda: np.testing.assert_allclose(
            j_fit.score, r_fit["reml_score"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        "smoothing parameters",
        lambda: np.testing.assert_allclose(
            j_fit.smoothing_params,
            r_fit["smoothing_params"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any("known-scale pinned EFS parity")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize("term", ["te(x,z,k=5)", "ti(x,z,k=5)"])
def test_coupled_efs_statistics_and_fit_match_pinned_r(term: str) -> None:
    """Coupled penalty trace/update state and final fit share one R run."""
    formula = f"y ~ {term}"
    data = _oracle_data("poisson", seed=44)
    family = Poisson()
    setup, fd = _build(formula, data, family)
    rho = efs_initial_log_lambda(setup, family)
    bridge = RBridge(mode="subprocess")
    diagnostic = bridge.efs_diagnostics(
        formula, data, "poisson", initial_smoothing=np.exp(np.asarray(rho))
    )
    state = _fit_state(
        fd, prepare_efs_statistics(fd), rho + 2.5, fd.beta_init, EFSControl()
    )
    first = diagnostic["statistics"].query("call == 1").sort_values("parameter")
    r_fit = bridge.fit_efs(
        formula, data, "poisson", initial_smoothing=np.exp(np.asarray(rho))
    )
    j_fit = dense_efs_known_scale(fd, initial_log_lambda=rho)
    collector = _AssertCollector()
    collector.check(
        "initial d",
        lambda: np.testing.assert_allclose(
            state.statistics.determinant_derivative,
            first["ldetS1"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "initial t",
        lambda: np.testing.assert_allclose(
            state.statistics.fisher_trace,
            first["trVS"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "initial q",
        lambda: np.testing.assert_allclose(
            state.statistics.quadratic,
            first["bSb"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "final deviance",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.deviance,
            r_fit["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "final criterion",
        lambda: np.testing.assert_allclose(
            j_fit.score, r_fit["reml_score"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.raise_if_any(f"pinned coupled EFS parity ({term})")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    ("family", "family_name"), [(Poisson(), "poisson"), (Binomial(), "binomial")]
)
def test_weighted_offset_additive_cr_matches_pinned_r(family, family_name: str) -> None:
    """Ordinary two-smooth case; setup must receive the same w/off as R."""
    data = make_data()
    if family_name == "binomial":
        probability = 1.0 / (1.0 + np.exp(-0.3 * data["x"]))
        data = data.assign(y=np.random.default_rng(93).binomial(1, probability))
    setup, fd = _build(
        FORMULA,
        data,
        family,
        weights=data["w"].to_numpy(),
        offset=data["off"].to_numpy(),
    )
    rho = efs_initial_log_lambda(setup, family)
    bridge = RBridge(mode="subprocess")
    r_diag = bridge.efs_diagnostics(
        FORMULA,
        data,
        family_name,
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
    )
    initial = _fit_state(
        fd, prepare_efs_statistics(fd), rho + 2.5, fd.beta_init, EFSControl()
    )
    first = r_diag["statistics"].query("call == 1").sort_values("parameter")
    r_fit = bridge.fit_efs(
        FORMULA,
        data,
        family_name,
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
    )
    j_fit = dense_efs_known_scale(fd, initial_log_lambda=rho)
    collector = _AssertCollector()
    collector.check(
        "initial lsp",
        lambda: np.testing.assert_allclose(
            rho + 2.5, first["log_smoothing"], rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "initial deviance",
        lambda: np.testing.assert_allclose(
            initial.pirls_result.deviance,
            first["deviance"].iloc[0],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "initial d",
        lambda: np.testing.assert_allclose(
            initial.statistics.determinant_derivative,
            first["ldetS1"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "initial t",
        lambda: np.testing.assert_allclose(
            initial.statistics.fisher_trace,
            first["trVS"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "initial q",
        lambda: np.testing.assert_allclose(
            initial.statistics.quadratic,
            first["bSb"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "final fitted",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.mu,
            r_fit["fitted_values"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "final deviance",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.deviance,
            r_fit["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "final score",
        lambda: np.testing.assert_allclose(
            j_fit.score, r_fit["reml_score"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.raise_if_any(f"weighted-offset additive CR EFS parity ({family_name})")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    "formula",
    [FORMULA, "y ~ te(x, z, k=5)"],
)
def test_fixed_theta_nb_log_efs_matches_pinned_r_with_real_weights_and_offsets(
    formula: str,
) -> None:
    """Fixed NB EFS preserves R's observed-score/Fisher-EDF split."""
    theta = 2.7
    data = _fixed_nb_data()
    family = NegativeBinomial(theta=theta, fixed=True)
    weights = data["w"].to_numpy()
    offset = data["off"].to_numpy()
    setup, fd = _build(formula, data, family, weights=weights, offset=offset)
    rho = efs_initial_log_lambda(setup, family)
    bridge = RBridge(mode="subprocess")
    diagnostic = bridge.efs_diagnostics(
        formula,
        data,
        "nb",
        theta=theta,
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
    )
    initial = _fit_state(
        fd, prepare_efs_statistics(fd), rho + 2.5, fd.beta_init, EFSControl()
    )
    assert (
        float(
            np.max(
                np.abs(
                    np.asarray(initial.pirls_result.XtWX)
                    - np.asarray(initial.pirls_result.XtWX_fisher)
                )
            )
        )
        > 1e-8
    )
    first = diagnostic["statistics"].query("call == 1").sort_values("parameter")
    r_fit = bridge.fit_efs(
        formula,
        data,
        "nb",
        theta=theta,
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
    )
    j_fit = dense_efs_known_scale(fd, initial_log_lambda=rho)
    collector = _AssertCollector()
    collector.check(
        "initial lsp",
        lambda: np.testing.assert_allclose(
            rho + 2.5, first["log_smoothing"], rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "initial deviance",
        lambda: np.testing.assert_allclose(
            initial.pirls_result.deviance,
            first["deviance"].iloc[0],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "initial determinant derivative",
        lambda: np.testing.assert_allclose(
            initial.statistics.determinant_derivative,
            first["ldetS1"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "initial Fisher trace",
        lambda: np.testing.assert_allclose(
            initial.statistics.fisher_trace,
            first["trVS"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "initial quadratic",
        lambda: np.testing.assert_allclose(
            initial.statistics.quadratic,
            first["bSb"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "fixed theta from R",
        lambda: np.testing.assert_allclose(
            r_fit["theta"], theta, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "fixed theta result",
        lambda: np.testing.assert_allclose(
            j_fit.theta, theta, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "only smoothing parameters",
        lambda: np.testing.assert_equal(
            j_fit.smoothing_params.shape, (fd.n_penalties,)
        ),
    )
    collector.check(
        "fitted values",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.mu,
            r_fit["fitted_values"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "deviance",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.deviance,
            r_fit["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "criterion",
        lambda: np.testing.assert_allclose(
            j_fit.score, r_fit["reml_score"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        "smoothing parameters",
        lambda: np.testing.assert_allclose(
            j_fit.smoothing_params,
            r_fit["smoothing_params"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any(f"fixed-theta NB EFS parity ({formula})")
