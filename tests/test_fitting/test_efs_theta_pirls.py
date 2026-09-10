"""EFS-only in-loop conditional-theta PIRLS tests.

These tests intentionally exercise the internal NB/log route directly. EFS
outer-controller wiring belongs to a later change: this module establishes the
immutable beta/theta state and the pinned ``gam.fit4`` ordering first.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting import pirls as pirls_module
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.efs_theta import (
    _efs_nb_deviance,
    _efs_nb_log_deviance,
    conditional_theta_nll,
)
from jaxgam.fitting.pirls import (
    _EFS_STATUS_BETA_STEP_FAILED,
    _EFS_STATUS_INVALID_INPUT,
    _EFS_STATUS_INVALID_WORKING_FACTORS,
    _EFS_STATUS_ITERATION_LIMIT,
    _EFS_STATUS_NONFINITE_RECOVERY_FAILED,
    _EFS_STATUS_THETA_FAILED,
    _efs_beta_step_with_recovery,
    _efs_nb_observed_working_factors,
    _efs_observed_working_factors,
    _efs_observed_working_factors_from_derivatives,
    _EFSBetaRecoveryResult,
    efs_theta_pirls_loop,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector, check_that, r_available
from tests.r_bridge import RBridge
from tests.tolerances import MODERATE, STRICT


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_nonlog_nb_near_boundary_observed_factors_match_pinned_r_strict() -> None:
    """Raw valid eta keeps source derivatives below the family reporting floor."""
    y = np.array([0.0, 0.0, 1.0, 5.0, 50.0, 4000.0])
    mu = np.array([1e-12, 0.05, 0.9, 3.0, 1e3, 1e4])
    weights = np.array([1.0, 0.7, 1.3, 2.0, 0.5, 1.1])
    offset = np.array([0.013, 0.01, -0.02, 0.03, -0.04, 0.05])
    bridge = RBridge(mode="subprocess")
    collector = _AssertCollector()
    for link in ("identity", "sqrt"):
        family = NegativeBinomial(theta=2.7, link=link)
        raw_eta = jnp.asarray(mu if link == "identity" else np.sqrt(mu))

        @jax.jit
        def factors(eta, log_theta, family=family):
            return _efs_nb_observed_working_factors(
                eta,
                log_theta,
                jnp.asarray(y),
                jnp.asarray(weights),
                jnp.asarray(offset),
                family,
            )

        for theta in (0.1, 2.7, 1e6):
            actual = factors(raw_eta, jnp.asarray([np.log(theta)]))
            reference = bridge.efs_nb_working_factors(
                link, y, mu, weights, theta, offset
            )
            collector.check(
                f"{link} theta={theta:g} weight",
                lambda a=actual.weight, e=reference["weight"]: (
                    np.testing.assert_allclose(a, e, rtol=STRICT.rtol, atol=STRICT.atol)
                ),
            )
            collector.check(
                f"{link} theta={theta:g} weighted response",
                lambda a=actual.weighted_response, e=reference["weighted_response"]: (
                    np.testing.assert_allclose(a, e, rtol=STRICT.rtol, atol=STRICT.atol)
                ),
            )
            collector.check(
                f"{link} theta={theta:g} finite factors",
                lambda a=actual: check_that(
                    bool(a.valid),
                    "near-boundary observed W/Wz must remain usable",
                ),
            )
    collector.raise_if_any("nonlog NB near-boundary observed factors")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_nonlog_nb_finite_tail_deviance_derivatives_match_pinned_r_strict() -> None:
    """Large valid means retain finite source deviance and eta derivatives."""
    y = np.array([0.25, 0.0, 0.0])
    mu = np.array([1e8, 1e16, 1e20])
    weights = np.array([1.0, 0.7, 1.3])
    theta = 2.7
    bridge = RBridge(mode="subprocess")
    collector = _AssertCollector()
    for link in ("identity", "sqrt"):
        family = NegativeBinomial(theta=theta, link=link)
        eta = jnp.asarray(mu if link == "identity" else np.sqrt(mu))
        log_theta = jnp.asarray([np.log(theta)])

        def deviance(current_eta, log_theta=log_theta, family=family):
            return _efs_nb_deviance(
                current_eta,
                log_theta,
                jnp.asarray(y),
                jnp.asarray(weights),
                family,
            )

        actual_value = jax.jit(deviance)(eta)
        actual_deta = jax.jit(jax.grad(deviance))(eta)
        actual_deta2 = jnp.diag(jax.jit(jax.hessian(deviance))(eta))
        reference = bridge.efs_nb_deviance_derivatives(link, y, mu, weights, theta)
        collector.check(
            f"{link} deviance",
            lambda a=actual_value, e=reference["deviance"]: np.testing.assert_allclose(
                a, np.sum(e), rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        collector.check(
            f"{link} first derivative",
            lambda a=actual_deta, e=reference["deta"]: np.testing.assert_allclose(
                a, e, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        collector.check(
            f"{link} second derivative",
            lambda a=actual_deta2, e=reference["deta2"]: np.testing.assert_allclose(
                a, e, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        collector.check(
            f"{link} finite",
            lambda values=(actual_value, actual_deta, actual_deta2): check_that(
                all(np.all(np.isfinite(np.asarray(value))) for value in values),
                "valid NB tail values and derivatives must remain finite",
            ),
        )
    collector.raise_if_any("nonlog NB finite tail deviance derivatives")


def _fixture(*, fractional_below_one: bool = False):
    rng = np.random.default_rng(5053)
    n = 42
    x = np.linspace(-1.0, 1.0, n)
    offset = 0.08 * np.cos(2.1 * x)
    wt = 0.5 + rng.random(n)
    mu = np.exp(offset + 0.25 + 0.4 * np.sin(2.2 * x))
    y = rng.negative_binomial(2.2, 2.2 / (2.2 + mu)).astype(np.float64)
    if fractional_below_one:
        y[0] = 0.25
    data = pd.DataFrame({"y": y, "x": x})
    setup = ModelSetup.build(
        parse_formula("y ~ s(x, bs='cr', k=6)"), data, weights=wt, offset=offset
    )
    fd = FittingData.from_setup(setup, NegativeBinomial(theta=0.8))
    assert fd.count_prefix_plan is not None
    S_lambda = jnp.eye(fd.X.shape[1]) * 0.15
    return fd, S_lambda


def _fit(fd: FittingData, S_lambda: jax.Array, **kwargs):
    plan = fd.count_prefix_plan
    assert plan is not None
    return efs_theta_pirls_loop(
        fd.X,
        fd.y,
        fd.beta_init,
        S_lambda,
        fd.family,
        fd.wt,
        fd.offset,
        jnp.asarray([np.log(0.8)]),
        plan.indices,
        max_y=fd.max_y,
        integer_counts=plan.integer_counts,
        **kwargs,
    )


def _call(fd: FittingData, S_lambda: jax.Array, **kwargs):
    plan = fd.count_prefix_plan
    assert plan is not None
    defaults = {
        "X": fd.X,
        "y": fd.y,
        "beta_init": fd.beta_init,
        "S_lambda": S_lambda,
        "family": fd.family,
        "wt": fd.wt,
        "offset": fd.offset,
        "log_theta_init": jnp.asarray([np.log(0.8)]),
        "count_indices": plan.indices,
        "max_y": fd.max_y,
        "integer_counts": plan.integer_counts,
    }
    defaults.update(kwargs)
    return efs_theta_pirls_loop(**defaults)


def test_efs_theta_pirls_jit_rebuilds_final_quantities_at_returned_theta():
    fd, S_lambda = _fixture()
    result = _fit(fd, S_lambda)
    pr = result.pirls_result

    assert bool(np.asarray(pr.converged))
    assert int(np.asarray(result.status)) == 0
    assert float(np.asarray(pr.scale)) == 1.0
    assert np.all(np.isfinite(np.asarray(pr.coefficients)))
    assert np.all(np.isfinite(np.asarray(pr.XtWX)))
    assert np.all(np.isfinite(np.asarray(pr.XtWX_fisher)))

    deviance = fd.family.deviance_fn(fd.y, fd.wt)(pr.eta, result.log_theta)
    expected_pdev = deviance + pr.coefficients @ S_lambda @ pr.coefficients
    np.testing.assert_allclose(
        pr.deviance, deviance, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        pr.penalized_deviance, expected_pdev, rtol=STRICT.rtol, atol=STRICT.atol
    )

    W_fisher = fd.family.working_weights_fn(fd.wt)(pr.eta, result.log_theta)
    expected_fisher = (W_fisher[:, None] * fd.X).T @ fd.X
    np.testing.assert_allclose(
        pr.XtWX_fisher, expected_fisher, rtol=STRICT.rtol, atol=STRICT.atol
    )

    # The wrapped route is itself JIT-safe and theta remains a dynamic input.
    compiled = jax.jit(
        efs_theta_pirls_loop,
        static_argnames=(
            "family",
            "max_y",
            "integer_counts",
            "max_iter",
            "tol",
            "initial_start_retained",
        ),
    )
    plan = fd.count_prefix_plan
    assert plan is not None
    compiled_result = compiled(
        fd.X,
        fd.y,
        fd.beta_init,
        S_lambda,
        fd.family,
        fd.wt,
        fd.offset,
        jnp.asarray([np.log(1.7)]),
        plan.indices,
        max_y=fd.max_y,
        integer_counts=plan.integer_counts,
    )
    assert np.all(np.isfinite(np.asarray(compiled_result.log_theta)))
    # A default/reset EFS start supplies link(mustart), which need not equal
    # the projected coefficient predictor. It remains dynamic JIT input.
    mustart = fd.family.initialize(fd.y, fd.wt)
    mustart_eta = fd.family.link.link(mustart)
    reset_result = compiled(
        fd.X,
        fd.y,
        fd.beta_init,
        S_lambda,
        fd.family,
        fd.wt,
        fd.offset,
        jnp.asarray([np.log(1.7)]),
        plan.indices,
        initial_eta=mustart_eta,
        initial_start_retained=False,
        max_y=fd.max_y,
        integer_counts=plan.integer_counts,
    )
    assert np.all(np.isfinite(np.asarray(reset_result.log_theta)))


def test_efs_theta_pirls_rejects_fractional_response_before_convergence():
    fd, S_lambda = _fixture(fractional_below_one=True)
    result = _fit(fd, S_lambda)

    assert not bool(np.asarray(result.pirls_result.converged))
    assert int(np.asarray(result.status)) == _EFS_STATUS_THETA_FAILED
    assert int(np.asarray(result.theta_status)) == 8


def test_efs_theta_first_step_uses_explicit_old_baseline_and_origin():
    fd, S_lambda = _fixture()
    beta_old = jnp.zeros_like(fd.beta_init)
    result = _fit(fd, S_lambda, beta_old_init=beta_old, max_iter=1)
    eta_old = fd.X @ beta_old + fd.offset
    initial_pdev = (
        fd.family.deviance_fn(fd.y, fd.wt)(eta_old, jnp.asarray([np.log(0.8)]))
        + beta_old @ S_lambda @ beta_old
    )

    # Unlike ordinary PIRLS, EFS does not accept its first proposal merely
    # because it is finite: gam.fit4 compares it to old.pdev/null.coef.
    threshold = 10.0 * (0.1 + abs(float(initial_pdev))) * np.sqrt(np.finfo(float).eps)
    assert (
        float(np.asarray(result.stopping_penalized_deviance))
        <= float(initial_pdev) + threshold
    )


def test_efs_theta_input_validation_is_static_before_jit():
    fd, S_lambda = _fixture()
    plan = fd.count_prefix_plan
    assert plan is not None
    with pytest.raises(ValueError, match="max_iter"):
        efs_theta_pirls_loop(
            fd.X,
            fd.y,
            fd.beta_init,
            S_lambda,
            fd.family,
            fd.wt,
            fd.offset,
            jnp.asarray([0.0]),
            plan.indices,
            max_y=fd.max_y,
            integer_counts=plan.integer_counts,
            max_iter=True,
        )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("beta_init", jnp.zeros(1), "beta_init"),
        ("S_lambda", jnp.eye(1), "S_lambda"),
        ("count_indices", jnp.zeros(42, dtype=jnp.float64), "count_indices"),
        ("X", jnp.zeros((0, 2)), "nonempty"),
    ],
)
def test_efs_theta_pirls_rejects_static_shape_and_prefix_dtype(name, value, message):
    fd, S_lambda = _fixture()
    call = (
        (lambda: _call(fd, value))
        if name == "S_lambda"
        else lambda: _call(fd, S_lambda, **{name: value})
    )
    with pytest.raises(ValueError, match=message):
        call()


def test_efs_theta_pirls_reports_dynamic_malformed_inputs_without_convergence():
    fd, S_lambda = _fixture()
    bad_weights = _call(fd, S_lambda, wt=-fd.wt)
    bad_prefix = _call(fd, S_lambda, count_indices=jnp.full(fd.y.shape, -1))
    for result in (bad_weights, bad_prefix):
        assert not bool(np.asarray(result.pirls_result.converged))
        assert int(np.asarray(result.status)) == _EFS_STATUS_INVALID_INPUT


def test_efs_theta_pirls_reports_invalid_factor_and_iteration_limit():
    fd, S_lambda = _fixture()
    invalid_factor = _call(fd, jnp.full_like(S_lambda, jnp.nan))
    limited = _call(fd, S_lambda, max_iter=1)
    assert not bool(np.asarray(invalid_factor.pirls_result.converged))
    assert int(np.asarray(invalid_factor.status)) == _EFS_STATUS_INVALID_WORKING_FACTORS
    assert not bool(np.asarray(limited.pirls_result.converged))
    assert int(np.asarray(limited.status)) == _EFS_STATUS_ITERATION_LIMIT


def test_efs_theta_pirls_distinguishes_unrecoverable_beta_step(monkeypatch):
    """A valid NB/log old baseline is eventually reached by binary halving.

    The beta-step failure status is therefore exercised with an isolated pure
    step double, rather than fabricating invalid NB data and mislabelling that
    as an ordinary divergence.
    """
    fd, S_lambda = _fixture()

    def rejected_step(**kwargs):
        beta = kwargs["beta"]
        return _EFSBetaRecoveryResult(
            beta=beta,
            mu=kwargs["mu"],
            eta=kwargs["eta"],
            penalized_deviance=kwargs["baseline"],
            accepted=jnp.array(False),
            factors_valid=jnp.array(True),
            solver_valid=jnp.array(True),
            failure_status=jnp.array(0, dtype=jnp.int32),
        )

    monkeypatch.setattr(pirls_module, "_efs_beta_step_with_recovery", rejected_step)
    pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    try:
        result = _call(fd, S_lambda)
    finally:
        pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    assert not bool(np.asarray(result.pirls_result.converged))
    assert int(np.asarray(result.status)) == _EFS_STATUS_BETA_STEP_FAILED


def test_efs_theta_pirls_names_nonfinite_recovery_exhaustion(monkeypatch):
    """A failed EFS recovery is distinct from WLS/factor failure."""
    fd, S_lambda = _fixture()

    def exhausted_nonfinite_recovery(**kwargs):
        beta = kwargs["beta"]
        return _EFSBetaRecoveryResult(
            beta=beta,
            mu=kwargs["mu"],
            eta=kwargs["eta"],
            penalized_deviance=kwargs["baseline"],
            accepted=jnp.array(False),
            factors_valid=jnp.array(True),
            solver_valid=jnp.array(True),
            failure_status=jnp.array(
                _EFS_STATUS_NONFINITE_RECOVERY_FAILED, dtype=jnp.int32
            ),
        )

    monkeypatch.setattr(
        pirls_module, "_efs_beta_step_with_recovery", exhausted_nonfinite_recovery
    )
    pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    try:
        result = _call(
            fd,
            S_lambda,
            beta_old_init=jnp.zeros_like(fd.beta_init),
        )
    finally:
        pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    assert not bool(np.asarray(result.pirls_result.converged))
    assert int(np.asarray(result.status)) == _EFS_STATUS_NONFINITE_RECOVERY_FAILED


def test_efs_theta_pirls_final_factor_guard_overrides_stale_convergence(monkeypatch):
    """Final curvature failure cannot be reported as a selected-theta fit."""
    fd, S_lambda = _fixture()

    def nonfinite_factor(XtWX, _S):
        return jnp.full_like(XtWX, jnp.nan), jnp.array(jnp.nan)

    monkeypatch.setattr(pirls_module, "penalized_cholesky", nonfinite_factor)
    pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    try:
        result = _call(fd, S_lambda)
    finally:
        pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    assert not bool(np.asarray(result.pirls_result.converged))
    assert int(np.asarray(result.status)) == _EFS_STATUS_INVALID_WORKING_FACTORS


def _fake_recovery_step(
    *,
    beta: float,
    eta: float,
    beta_old: float,
    eta_old: float,
    null_beta: float,
    null_eta: float,
    retained: bool,
    iteration: int,
    baseline: float,
    max_recovery_halvings: int,
    dev_fn,
):
    """Exercise recovery anchors without depending on a full WLS fixture."""
    X = jnp.zeros((1, 1))
    S_lambda = jnp.ones((1, 1))
    offset = jnp.asarray([10.0])
    family = NegativeBinomial(theta=0.8)

    def compute_working_factors(_mu, _eta):
        return _efs_observed_working_factors(
            jnp.ones((1,)), jnp.zeros((1,)), jnp.zeros((1,))
        )

    def form_wls(_factors):
        # With X=0, this deliberately supplies the raw WLS right side.  The
        # penalty makes the proposal beta=10 while eta remains offset=10.
        return jnp.zeros((1, 1)), jnp.asarray([10.0])

    return _efs_beta_step_with_recovery(
        X=X,
        S_lambda=S_lambda,
        offset=offset,
        family=family,
        beta=jnp.asarray([beta]),
        eta=jnp.asarray([eta]),
        mu=family.link.inverse(jnp.asarray([eta])),
        beta_old=jnp.asarray([beta_old]),
        eta_old=jnp.asarray([eta_old]),
        null_beta=jnp.asarray([null_beta]),
        null_eta=jnp.asarray([null_eta]),
        initial_start_retained=jnp.asarray(retained),
        iteration=jnp.asarray(iteration, dtype=jnp.int32),
        baseline=jnp.asarray(baseline),
        compute_working_factors=compute_working_factors,
        form_wls=form_wls,
        compute_dev=lambda _mu, current_eta: dev_fn(current_eta),
        max_recovery_halvings=max_recovery_halvings,
    )


def test_efs_beta_recovery_uses_retained_and_mustart_eta_anchors_under_jit():
    """First validity recovery distinguishes retained and absent starts."""

    def dev_fn(current_eta):
        return jnp.where(current_eta[0] > 7.0, jnp.inf, current_eta[0] ** 2)

    retained = jax.jit(
        lambda: _fake_recovery_step(
            beta=5.0,
            eta=5.0,
            beta_old=0.0,
            eta_old=0.0,
            null_beta=0.0,
            null_eta=0.0,
            retained=True,
            iteration=0,
            baseline=100.0,
            max_recovery_halvings=2,
            dev_fn=dev_fn,
        )
    )()
    assert bool(retained.accepted)
    np.testing.assert_allclose(retained.beta, [6.25])
    np.testing.assert_allclose(retained.eta, [6.25])

    # An absent coefficient start has a per-row mustart eta (5) but R's
    # recovery origin remains null eta (2 here), not that mustart eta.
    mustart = jax.jit(
        lambda: _fake_recovery_step(
            beta=0.0,
            eta=5.0,
            beta_old=0.0,
            eta_old=2.0,
            null_beta=0.0,
            null_eta=2.0,
            retained=False,
            iteration=0,
            baseline=100.0,
            max_recovery_halvings=1,
            dev_fn=dev_fn,
        )
    )()
    assert bool(mustart.accepted)
    np.testing.assert_allclose(mustart.beta, [5.0])
    np.testing.assert_allclose(mustart.eta, [6.0])


def test_efs_beta_recovery_uses_maxit_budget_and_null_first_divergence_anchor():
    """Validity loops use EFS maxit; first divergence uses null separately."""

    def nonfinite_dev(current_eta):
        return jnp.where(current_eta[0] > 7.0, jnp.inf, current_eta[0] ** 2)

    exhausted = jax.jit(
        lambda: _fake_recovery_step(
            beta=5.0,
            eta=5.0,
            beta_old=0.0,
            eta_old=0.0,
            null_beta=0.0,
            null_eta=0.0,
            retained=True,
            iteration=0,
            baseline=100.0,
            max_recovery_halvings=1,
            dev_fn=nonfinite_dev,
        )
    )()
    assert not bool(exhausted.accepted)
    assert (
        int(np.asarray(exhausted.failure_status))
        == _EFS_STATUS_NONFINITE_RECOVERY_FAILED
    )

    def finite_dev(_current_eta):
        return jnp.asarray(0.0)

    divergent = jax.jit(
        lambda: _fake_recovery_step(
            beta=5.0,
            eta=5.0,
            beta_old=5.0,
            eta_old=5.0,
            null_beta=0.0,
            null_eta=0.0,
            retained=True,
            iteration=0,
            baseline=10.0,
            max_recovery_halvings=1,
            dev_fn=finite_dev,
        )
    )()
    assert bool(divergent.accepted)
    np.testing.assert_allclose(divergent.beta, [2.5])
    np.testing.assert_allclose(divergent.eta, [2.5])


def test_efs_beta_recovery_rejects_nonfinite_post_divergence_state():
    """NaN after a divergence midpoint cannot pass a false comparison."""

    def dev_fn(current_eta):
        return jnp.where(current_eta[0] < 10.0, jnp.nan, jnp.asarray(0.0))

    result = jax.jit(
        lambda: _fake_recovery_step(
            beta=5.0,
            eta=5.0,
            beta_old=5.0,
            eta_old=5.0,
            null_beta=0.0,
            null_eta=0.0,
            retained=True,
            iteration=0,
            baseline=10.0,
            max_recovery_halvings=1,
            dev_fn=dev_fn,
        )
    )()
    assert not bool(result.accepted)
    assert (
        int(np.asarray(result.failure_status)) == _EFS_STATUS_NONFINITE_RECOVERY_FAILED
    )


def test_efs_observed_finite_wz_fallback_is_jitted_and_fails_closed():
    """Use finite ``wz`` directly, preserving zero rows and invalid rejection."""
    X = jnp.eye(2)
    S_lambda = jnp.eye(2)
    eta = jnp.zeros(2)
    offset = jnp.zeros(2)
    beta = jnp.zeros(2)
    family = NegativeBinomial(theta=0.8)

    def run(d1, d2):
        factors = _efs_observed_working_factors_from_derivatives(eta, offset, d1, d2)

        def compute_working_factors(_mu, _eta):
            return factors

        def form_wls(current_factors):
            weight = jnp.clip(current_factors.weight, -1e10, 1e10)
            rhs = jnp.where(
                current_factors.use_weighted_response,
                current_factors.weighted_response,
                weight * current_factors.response,
            )
            return (weight[:, None] * X).T @ X, X.T @ rhs

        return _efs_beta_step_with_recovery(
            X=X,
            S_lambda=S_lambda,
            offset=offset,
            family=family,
            beta=beta,
            eta=eta,
            mu=family.link.inverse(eta),
            beta_old=beta,
            eta_old=eta,
            null_beta=beta,
            null_eta=eta,
            initial_start_retained=jnp.asarray(False),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            baseline=jnp.asarray(0.0),
            compute_working_factors=compute_working_factors,
            form_wls=form_wls,
            compute_dev=lambda _mu, current_eta: -current_eta @ current_eta,
            max_recovery_halvings=1,
        )

    direct_wz = jax.jit(run)(
        jnp.asarray([0.25, 0.0]),
        jnp.asarray([0.0, 0.0]),
    )
    zero_prior_weight = jax.jit(run)(jnp.zeros(2), jnp.zeros(2))
    invalid_wz = jax.jit(run)(jnp.asarray([jnp.nan, 0.0]), jnp.zeros(2))
    clipped_fallback = jax.jit(_efs_observed_working_factors)(
        jnp.asarray([2e10]), jnp.asarray([jnp.nan]), jnp.asarray([1.0])
    )

    assert bool(direct_wz.factors_valid)
    assert bool(direct_wz.accepted)
    np.testing.assert_allclose(direct_wz.beta, [-0.125, 0.0])
    assert (
        int(np.asarray(invalid_wz.failure_status))
        == _EFS_STATUS_INVALID_WORKING_FACTORS
    )
    assert bool(zero_prior_weight.factors_valid)
    assert bool(zero_prior_weight.solver_valid)
    assert bool(zero_prior_weight.accepted)
    np.testing.assert_allclose(zero_prior_weight.beta, beta)
    assert not bool(clipped_fallback.valid)
    assert not bool(clipped_fallback.use_weighted_response)


def _r_efs_inner_oracle(
    X: np.ndarray,
    y: np.ndarray,
    wt: np.ndarray,
    offset: np.ndarray,
    penalty: float,
    log_theta: float,
    *,
    start: np.ndarray | None = None,
    null_coef: np.ndarray | None = None,
    maxit: int = 100,
) -> dict[str, np.ndarray | float]:
    """Trace pinned ``gam.fit4(scoreType='EFS')`` without namespace edits."""
    if start is None:
        start = np.zeros(X.shape[1])
    if null_coef is None:
        null_coef = np.zeros(X.shape[1])
    if start.shape != (X.shape[1],) or null_coef.shape != (X.shape[1],):
        raise ValueError("R EFS inner oracle starts must align with X")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_path = root / "data.csv"
        output_path = root / "output.csv"
        pre_path = root / "pre.csv"
        post_path = root / "post.csv"
        theta_path = root / "theta.csv"
        beta_path = root / "beta.csv"
        raw_beta_path = root / "raw_beta.csv"
        raw_eta_path = root / "raw_eta.csv"
        raw_deviance_path = root / "raw_deviance.csv"
        halving_path = root / "halvings.csv"
        initial_factors_path = root / "initial_factors.csv"
        use_wy_path = root / "use_wy.csv"
        theta_state_path = root / "theta_state.csv"
        np.savetxt(
            data_path,
            np.column_stack((X, y, wt, offset)),
            delimiter=",",
            header=",".join([*(f"x{i}" for i in range(X.shape[1])), "y", "wt", "off"]),
            comments="",
        )
        x_columns = ", ".join(f"d[, {i + 1}]" for i in range(X.shape[1]))
        pre_anchor = 'if (scoreType == "EFS" && family$n.theta > 0) {'
        pre_replacement = (
            "trace$pre <- c(trace$pre, pdev); "
            "trace$theta_in <- c(trace$theta_in, theta[1]); "
            "trace$beta <- rbind(trace$beta, as.numeric(start)[seq_len(ncol(x))]); "
            'if (scoreType == "EFS" && family$n.theta > 0) {'
        )
        script = f"""
library(mgcv)
if (as.character(getRversion()) != "4.5.2" ||
    packageVersion("mgcv") != package_version("1.9.3")) {{
  stop("EFS inner oracle requires R 4.5.2 and mgcv 1.9-3")
}}
d <- as.matrix(read.csv({str(data_path)!r}, check.names=FALSE))
x <- cbind({x_columns})
y <- d[, {X.shape[1] + 1}]
wt <- d[, {X.shape[1] + 2}]
off <- d[, {X.shape[1] + 3}]
trace_env <- new.env(parent=asNamespace("mgcv"))
trace_env$trace <- new.env(parent=emptyenv())
trace <- trace_env$trace
trace$pre <- numeric(); trace$post <- numeric(); trace$theta_in <- numeric()
trace$theta_out <- numeric(); trace$beta <- matrix(numeric(), 0, ncol(x))
trace$raw_beta <- matrix(numeric(), 0, ncol(x))
trace$raw_eta <- matrix(numeric(), 0, length(y))
trace$raw_deviance <- numeric(); trace$nonfinite_halvings <- integer()
trace$domain_halvings <- integer(); trace$divergence_halvings <- integer()
trace$start_retained <- logical()
trace$initial_w <- numeric()
trace$initial_wz <- numeric()
trace$initial_z <- numeric()
trace$use_wy <- logical()
source_lines <- capture.output(mgcv:::gam.fit4)
source_lines <- source_lines[!grepl("^<", source_lines)]
source_lines[1] <- sub("^function", "gam.fit4.trace <- function", source_lines[1])
pre_anchor <- {pre_anchor!r}
pre_replacement <- {pre_replacement!r}
theta_anchor <- "family$putTheta(theta)"
replace_once <- function(old, new) {{
  if (sum(grepl(old, source_lines, fixed=TRUE)) != 1) stop(paste("anchor changed", old))
  source_lines <<- sub(old, new, source_lines, fixed=TRUE)
}}
if (sum(grepl(pre_anchor, source_lines, fixed=TRUE)) != 2)
  stop("pre anchor changed")
source_lines <- sub(pre_anchor, pre_replacement, source_lines, fixed=TRUE)
if (sum(grepl(theta_anchor, source_lines, fixed=TRUE)) != 2)
  stop("theta anchor changed")
source_lines <- gsub(
  theta_anchor,
  "family$putTheta(theta); trace$theta_out <- c(trace$theta_out, theta[1])",
  source_lines, fixed=TRUE
)
raw_anchor <- "        if (!is.finite(dev)) {{"
if (sum(source_lines == raw_anchor) != 1L) stop("raw proposal anchor changed")
source_lines[source_lines == raw_anchor] <- paste0(
  "        trace$raw_beta <- rbind(trace$raw_beta, start)\\n",
  "        trace$raw_eta <- rbind(trace$raw_eta, eta)\\n",
  "        trace$raw_deviance <- c(trace$raw_deviance, dev)\\n",
  raw_anchor
)
halve_anchor <- "                start <- (start + coefold)/2"
halve_positions <- which(source_lines == halve_anchor)
if (length(halve_positions) != 3L) stop("halving anchors changed")
for (entry in list(
  c(1L, "nonfinite_halvings"), c(2L, "domain_halvings"),
  c(3L, "divergence_halvings")
)) {{
  index <- halve_positions[as.integer(entry[[1]])]
  source_lines[index] <- paste0(
    "                trace$", entry[[2]], " <- c(trace$", entry[[2]], ", iter)\\n",
    halve_anchor
  )
}}
retained_anchor <- "    coefold <- null.coef"
if (sum(source_lines == retained_anchor) != 1L) stop("retained-start anchor changed")
source_lines[source_lines == retained_anchor] <- paste0(
  "    trace$start_retained <- c(trace$start_retained, !is.null(start))\\n",
  retained_anchor
)
factor_anchor <- "    good <- is.finite(z) & is.finite(w)"
factor_positions <- which(source_lines == factor_anchor)
if (length(factor_positions) < 1L) stop("factor anchor changed")
source_lines[factor_positions[1]] <- paste0(
  "   trace$initial_w <- w; trace$initial_wz <- wz; trace$initial_z <- z\\n",
  factor_anchor
)
use_wy_positions <- grep("use.wy <- TRUE", source_lines, fixed=TRUE)
if (length(use_wy_positions) != 2L) stop("use.wy anchor changed")
source_lines[use_wy_positions] <- paste0(
  source_lines[use_wy_positions], "; trace$use_wy <- c(trace$use_wy, TRUE)"
)
replace_once(
  'old.pdev <- pdev <- dev + penalty',
  'old.pdev <- pdev <- dev + penalty; trace$post <- c(trace$post, pdev)'
)
eval(parse(text=source_lines), envir=trace_env)
family <- nb(theta=-exp({float(log_theta)!r}))
family <- mgcv:::fix.family.ls(mgcv:::fix.family.var(mgcv:::fix.family.link(family)))
fit <- trace_env$gam.fit4.trace(
  x=x, y=y, sp=c({float(log_theta)!r}, log({float(penalty)!r})), Eb=diag(ncol(x)),
  UrS=list(diag(ncol(x))), weights=wt, offset=off, U1=diag(ncol(x)), Mp=0,
  family=family, control=gam.control(epsilon=1e-7, maxit={maxit}), deriv=0,
  scoreType="EFS", scale=1,
  start=c({", ".join(repr(float(value)) for value in start)}),
  null.coef=c({", ".join(repr(float(value)) for value in null_coef)})
)
write.csv(data.frame(coefficient=as.numeric(fit$coefficients)),
          {str(output_path)!r}, row.names=FALSE)
write.csv(data.frame(value=trace$pre), {str(pre_path)!r}, row.names=FALSE)
write.csv(data.frame(value=trace$post), {str(post_path)!r}, row.names=FALSE)
write.csv(data.frame(value=trace$theta_out),
          {str(theta_path)!r}, row.names=FALSE)
write.csv(as.data.frame(trace$beta), {str(beta_path)!r}, row.names=FALSE)
write.csv(as.data.frame(trace$raw_beta), {str(raw_beta_path)!r}, row.names=FALSE)
write.csv(as.data.frame(trace$raw_eta), {str(raw_eta_path)!r}, row.names=FALSE)
write.csv(data.frame(value=trace$raw_deviance),
          {str(raw_deviance_path)!r}, row.names=FALSE)
write.csv(data.frame(
  retained=trace$start_retained[1],
  nonfinite=length(trace$nonfinite_halvings),
  domain=length(trace$domain_halvings),
  divergence=length(trace$divergence_halvings),
  first_nonfinite=sum(trace$nonfinite_halvings == 1L),
  first_domain=sum(trace$domain_halvings == 1L),
  first_divergence=sum(trace$divergence_halvings == 1L)
), {str(halving_path)!r}, row.names=FALSE)
write.csv(data.frame(w=trace$initial_w, wz=trace$initial_wz, z=trace$initial_z),
          {str(initial_factors_path)!r}, row.names=FALSE)
write.csv(data.frame(value=trace$use_wy), {str(use_wy_path)!r}, row.names=FALSE)
theta.final <- tail(trace$theta_out, 1L)
mu.final <- as.numeric(fit$fitted.values)
dev.final <- sum(family$dev.resids(y, mu.final, wt, theta.final))
Dd.final <- family$Dd(y, mu.final, theta.final, wt=wt, level=2)
ls.final <- family$ls(y, w=wt, theta=theta.final, scale=1)
nll.final <- dev.final / 2 - ls.final$ls
gradient.final <- sum(as.matrix(Dd.final$Dth)) / 2 - ls.final$lsth1[1]
write.csv(data.frame(
  log_theta=theta.final,
  nll=nll.final,
  gradient=gradient.final,
  threshold=1e-7 * (abs(nll.final) + 1),
  deviance=dev.final,
  eta_min=min(fit$linear.predictors),
  eta_max=max(fit$linear.predictors)
), {str(theta_state_path)!r}, row.names=FALSE)
"""
        script_path = root / "oracle.R"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True, timeout=30
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        return {
            "coefficients": np.loadtxt(output_path, delimiter=",", skiprows=1),
            "pre": np.loadtxt(pre_path, delimiter=",", skiprows=1, ndmin=1),
            "post": np.loadtxt(post_path, delimiter=",", skiprows=1, ndmin=1),
            "theta": np.loadtxt(theta_path, delimiter=",", skiprows=1, ndmin=1),
            "beta": np.loadtxt(beta_path, delimiter=",", skiprows=1, ndmin=2),
            "raw_beta": pd.read_csv(raw_beta_path).to_numpy(dtype=np.float64),
            "raw_eta": pd.read_csv(raw_eta_path).to_numpy(dtype=np.float64),
            "raw_deviance": pd.read_csv(raw_deviance_path)["value"].to_numpy(
                dtype=np.float64
            ),
            "halvings": pd.read_csv(halving_path).iloc[0].to_dict(),
            "initial_factors": pd.read_csv(initial_factors_path).to_numpy(
                dtype=np.float64
            ),
            "use_wy": pd.read_csv(use_wy_path)["value"].to_numpy(dtype=bool),
            "theta_state": pd.read_csv(theta_state_path).iloc[0].to_dict(),
        }


def _r_efs_nonsaturated_inner_oracle(
    X: np.ndarray,
    y: np.ndarray,
    wt: np.ndarray,
    offset: np.ndarray,
    penalty_diagonal: np.ndarray,
    log_theta: float,
    *,
    mp: int,
) -> dict[str, np.ndarray | float]:
    """Return one pinned-R final inner state for a diagonal penalty profile."""
    _, p = X.shape
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_path = root / "data.csv"
        penalty_path = root / "penalty.csv"
        output_path = root / "output.csv"
        np.savetxt(
            data_path,
            np.column_stack((X, y, wt, offset)),
            delimiter=",",
            header=",".join([*(f"x{i}" for i in range(p)), "y", "wt", "off"]),
            comments="",
        )
        np.savetxt(penalty_path, penalty_diagonal, delimiter=",")
        x_columns = ", ".join(f"d[, {i + 1}]" for i in range(p))
        script = f"""
suppressPackageStartupMessages(library(mgcv))
if (as.character(getRversion()) != "4.5.2" ||
    packageVersion("mgcv") != package_version("1.9.3")) {{
  stop("EFS nonsaturated inner oracle requires R 4.5.2 and mgcv 1.9-3")
}}
d <- as.matrix(read.csv({str(data_path)!r}, check.names=FALSE))
x <- cbind({x_columns})
y <- d[, {p + 1}]
w <- d[, {p + 2}]
off <- d[, {p + 3}]
pd <- as.numeric(read.csv({str(penalty_path)!r}, header=FALSE)[, 1])
root <- diag(sqrt(pd))
keep <- which(pd > 0)
U1 <- diag({p})[, c(keep, which(pd == 0)), drop=FALSE]
compact_root <- diag(sqrt(pd[keep]), nrow=length(keep))
f <- nb(theta=-exp({float(log_theta)!r}))
f <- mgcv:::fix.family.ls(mgcv:::fix.family.var(mgcv:::fix.family.link(f)))
fit <- mgcv:::gam.fit4(
  x=x, y=y, sp=c({float(log_theta)!r}, 0), Eb=root,
  UrS=list(compact_root), weights=w, offset=off, U1=U1, Mp={mp},
  family=f, control=gam.control(epsilon=1e-7, maxit=100), deriv=0,
  scoreType="EFS", scale=1, start=rep(0, {p}), null.coef=rep(0, {p})
)
log_theta_out <- f$getTheta(FALSE)
eta <- drop(x %*% fit$coefficients) + off
mu <- exp(eta)
deviance <- sum(f$dev.resids(y, mu, w, log_theta_out))
write.csv(as.data.frame(t(c(
  fit$coefficients, log_theta_out, deviance, fit$deviance
))), {str(output_path)!r}, row.names=FALSE)
"""
        script_path = root / "oracle.R"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True, timeout=30
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        output = pd.read_csv(output_path).iloc[0].to_numpy(dtype=np.float64)
    return {
        "coefficients": output[:p],
        "log_theta": float(output[p]),
        "deviance": float(output[p + 1]),
        "fit_deviance": float(output[p + 2]),
    }


def _nonsaturated_inner_profile(
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Construct the three independent, well-conditioned EFS inner profiles."""
    profiles = {
        "weighted_ridge": (93, 3, 0.65, 0.08, True),
        "zero_weight_unpenalized_intercept": (107, 5, 1.3, 8.0, False),
        "multiple_penalty_directions": (81, 4, 3.1, 0.8, False),
    }
    rng = np.random.default_rng(909506)
    for name, (n, p, generating_theta, starting_theta, ridge) in profiles.items():
        x = np.linspace(-1.5, 1.5, n)
        X = np.column_stack([np.ones(n), x, np.sin(2 * x), np.cos(3 * x), x * x])[:, :p]
        offset = 0.17 * np.cos(x)
        wt = rng.uniform(0.35, 1.8, n)
        if name == "zero_weight_unpenalized_intercept":
            wt[::9] = 0.0
        mu = np.exp(offset + 0.25 + 0.5 * np.sin(2 * x))
        y = rng.negative_binomial(
            generating_theta, generating_theta / (generating_theta + mu)
        ).astype(np.float64)
        penalty_diagonal = np.linspace(0.3, 1.2, p)
        if not ridge:
            penalty_diagonal[0] = 0.0
        if name == label:
            return (
                X,
                y,
                wt,
                offset,
                penalty_diagonal,
                float(np.log(starting_theta)),
                0 if ridge else 1,
            )
    raise ValueError(f"unknown nonsaturated inner profile {label!r}")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    "label",
    [
        "weighted_ridge",
        "zero_weight_unpenalized_intercept",
        "multiple_penalty_directions",
    ],
)
def test_efs_theta_pirls_nonsaturated_inner_contract_matches_pinned_r(label: str):
    """Keep one pinned final-state contract for each non-saturated inner profile."""
    X, y, wt, offset, penalty_diagonal, log_theta, mp = _nonsaturated_inner_profile(
        label
    )
    oracle = _r_efs_nonsaturated_inner_oracle(
        X, y, wt, offset, penalty_diagonal, log_theta, mp=mp
    )
    penalty = np.diag(penalty_diagonal)
    result = efs_theta_pirls_loop(
        jnp.asarray(X),
        jnp.asarray(y),
        jnp.zeros(X.shape[1]),
        jnp.asarray(penalty),
        NegativeBinomial(theta=float(np.exp(log_theta))),
        jnp.asarray(wt),
        jnp.asarray(offset),
        jnp.asarray([log_theta]),
        jnp.asarray(y, dtype=jnp.int64),
        beta_old_init=jnp.zeros(X.shape[1]),
        max_y=int(np.max(y)),
        integer_counts=True,
        max_iter=100,
        tol=1e-7,
    )
    pr = result.pirls_result
    theta = float(np.exp(np.asarray(result.log_theta)[0]))
    mu = np.asarray(pr.mu)
    observed_weight = wt * mu * theta * (y + theta) / (mu + theta) ** 2
    fisher_weight = wt * mu * theta / (mu + theta)
    score_equations = 2 * X.T @ (wt * theta * (mu - y) / (mu + theta)) + 2 * (
        penalty @ np.asarray(pr.coefficients)
    )
    collector = _AssertCollector()
    collector.check(
        "converged",
        lambda: check_that(
            bool(np.asarray(pr.converged)), f"status={int(np.asarray(result.status))}"
        ),
    )
    collector.check(
        "coefficients",
        lambda: np.testing.assert_allclose(
            pr.coefficients,
            oracle["coefficients"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "log theta",
        lambda: np.testing.assert_allclose(
            result.log_theta[0],
            oracle["log_theta"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "deviance",
        lambda: np.testing.assert_allclose(
            pr.deviance,
            oracle["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "observed curvature",
        lambda: np.testing.assert_allclose(
            pr.XtWX,
            X.T @ (observed_weight[:, None] * X),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "Fisher curvature",
        lambda: np.testing.assert_allclose(
            pr.XtWX_fisher,
            X.T @ (np.clip(fisher_weight, 1e-10, 1e10)[:, None] * X),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "score equations",
        lambda: check_that(
            np.max(np.abs(score_equations))
            <= 1e-7 * (abs(float(np.asarray(result.stopping_penalized_deviance))) + 1),
            f"max residual={np.max(np.abs(score_equations))}",
        ),
    )
    collector.raise_if_any(f"nonsaturated EFS inner contract ({label})")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_efs_theta_pirls_matches_pinned_fixed_penalty_inner_ordering():
    """Compare the dedicated loop to real pinned ``gam.fit4`` EFS internals."""
    rng = np.random.default_rng(5061)
    n = 38
    x = np.linspace(-1.0, 1.0, n)
    X = np.column_stack((np.ones(n), x))
    offset = 0.07 * np.cos(2.0 * x)
    wt = 0.6 + rng.random(n)
    mu = np.exp(offset + 0.2 + 0.35 * x)
    y = rng.negative_binomial(2.1, 2.1 / (2.1 + mu)).astype(np.float64)
    penalty = 0.18
    log_theta = np.log(0.8)
    oracle = _r_efs_inner_oracle(X, y, wt, offset, penalty, log_theta)

    assert oracle["beta"].shape[0] == oracle["pre"].size
    # The pinned EFS branch reaches the first theta condition twice per
    # accepted beta update (before/after its score bookkeeping). The paired
    # rows must be identical; theta changes only on the first row of each pair.
    theta_rows = np.arange(0, oracle["pre"].size, 2)
    paired_rows = theta_rows[:-1] + 1
    np.testing.assert_allclose(
        oracle["pre"][theta_rows[:-1]], oracle["pre"][paired_rows]
    )
    np.testing.assert_allclose(
        oracle["beta"][theta_rows[:-1]], oracle["beta"][paired_rows]
    )
    assert oracle["theta"].size == theta_rows.size + 1

    def fit_prefix(iteration: int):
        return efs_theta_pirls_loop(
            jnp.asarray(X),
            jnp.asarray(y),
            jnp.zeros(X.shape[1]),
            jnp.eye(X.shape[1]) * penalty,
            NegativeBinomial(theta=np.exp(log_theta)),
            jnp.asarray(wt),
            jnp.asarray(offset),
            jnp.asarray([log_theta]),
            jnp.asarray(y, dtype=jnp.int64),
            beta_old_init=jnp.zeros(X.shape[1]),
            max_y=int(np.max(y)),
            integer_counts=True,
            max_iter=iteration,
        )

    prefix_results = [
        fit_prefix(iteration) for iteration in range(1, theta_rows.size + 1)
    ]
    for index, result in enumerate(prefix_results):
        np.testing.assert_allclose(
            result.pirls_result.coefficients,
            oracle["beta"][theta_rows[index]],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        np.testing.assert_allclose(
            result.log_theta[0],
            oracle["theta"][index + 1],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        # ``stopping_penalized_deviance`` intentionally retains the pre-theta
        # value used by R's convergence predicate, separate from returned pdev.
        np.testing.assert_allclose(
            result.stopping_penalized_deviance,
            oracle["pre"][theta_rows[index]],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
    for index, value in enumerate(oracle["post"]):
        np.testing.assert_allclose(
            prefix_results[index].post_theta_penalized_deviance,
            value,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
    final = prefix_results[-1]
    assert bool(np.asarray(final.pirls_result.converged))
    np.testing.assert_allclose(
        final.pirls_result.coefficients,
        oracle["coefficients"],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_efs_positive_weight_finite_wz_fallback_matches_pinned_first_beta():
    """Exercise the actual NB/log cancellation that sends ``gam.fit4`` to wy."""
    X = jnp.eye(2)
    y = jnp.asarray([1.0, 2.0])
    wt = jnp.ones(2)
    offset = jnp.zeros(2)
    retained_start = jnp.asarray([37.0, 0.0])
    null_beta = jnp.asarray([50.0, 0.0])
    log_theta = jnp.asarray([np.log(0.8)])
    penalty = jnp.eye(2) * 0.2
    oracle = _r_efs_inner_oracle(
        np.asarray(X),
        np.asarray(y),
        np.asarray(wt),
        np.asarray(offset),
        penalty=0.2,
        log_theta=float(log_theta[0]),
        start=np.asarray(retained_start),
        null_coef=np.asarray(null_beta),
        maxit=1,
    )
    zero_weight_oracle = _r_efs_inner_oracle(
        np.asarray(X),
        np.asarray(y),
        np.asarray([0.0, 1.0]),
        np.asarray(offset),
        penalty=0.2,
        log_theta=float(log_theta[0]),
        start=np.zeros(2),
        null_coef=np.zeros(2),
        maxit=1,
    )
    family = NegativeBinomial(theta=0.8)

    def run_prefix():
        def dev_fn(current_eta):
            return _efs_nb_log_deviance(current_eta, log_theta, y, wt)

        def compute_working_factors(_mu, current_eta):
            return _efs_nb_observed_working_factors(
                current_eta, log_theta, y, wt, offset
            )

        def form_wls(factors):
            weight = jnp.clip(factors.weight, -1e10, 1e10)
            rhs = jnp.where(
                factors.use_weighted_response,
                factors.weighted_response,
                weight * factors.response,
            )
            return (weight[:, None] * X).T @ X, X.T @ rhs

        return _efs_beta_step_with_recovery(
            X=X,
            S_lambda=penalty,
            offset=offset,
            family=family,
            beta=retained_start,
            eta=retained_start,
            mu=family.link.inverse(retained_start),
            beta_old=null_beta,
            eta_old=null_beta,
            null_beta=null_beta,
            null_eta=null_beta,
            initial_start_retained=jnp.asarray(True),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            baseline=dev_fn(null_beta),
            compute_working_factors=compute_working_factors,
            form_wls=form_wls,
            compute_dev=lambda _mu, current_eta: dev_fn(current_eta),
            max_recovery_halvings=200,
        )

    factors = jax.jit(_efs_nb_observed_working_factors)(
        retained_start, log_theta, y, wt, offset
    )
    zero_weight_factors = jax.jit(_efs_nb_observed_working_factors)(
        jnp.zeros(2), log_theta, y, jnp.asarray([0.0, 1.0]), offset
    )
    prefix = jax.jit(run_prefix)()
    loop = efs_theta_pirls_loop(
        X,
        y,
        retained_start,
        penalty,
        family,
        wt,
        offset,
        log_theta,
        y.astype(jnp.int64),
        beta_old_init=null_beta,
        initial_eta=retained_start,
        initial_start_retained=True,
        max_y=2,
        integer_counts=True,
        max_iter=1,
    )
    collector = _AssertCollector()
    collector.check(
        "source finite-wz fallback",
        lambda: check_that(
            bool(np.any(oracle["use_wy"])),
            f"source did not set use.wy: {oracle['use_wy']}",
        ),
    )
    collector.check(
        "source positive-weight cancellation",
        lambda: np.testing.assert_allclose(
            oracle["initial_factors"][0], [0.0, -0.8, -np.inf]
        ),
    )
    collector.check(
        "JAX finite-wz factors",
        lambda: check_that(
            bool(factors.valid) and bool(factors.use_weighted_response),
            f"unexpected factors: {factors}",
        ),
    )
    collector.check(
        "source zero-prior-weight fallback",
        lambda: check_that(
            bool(np.any(zero_weight_oracle["use_wy"])),
            f"source did not set use.wy: {zero_weight_oracle['use_wy']}",
        ),
    )
    collector.check(
        "zero prior weight stays finite direct RHS",
        lambda: check_that(
            bool(zero_weight_factors.valid)
            and bool(zero_weight_factors.use_weighted_response)
            and float(zero_weight_factors.weighted_response[0]) == 0.0
            and np.isclose(zero_weight_oracle["initial_factors"][0, 0], 0.0)
            and np.isclose(zero_weight_oracle["initial_factors"][0, 1], 0.0),
            f"unexpected zero-weight factors: {zero_weight_factors}",
        ),
    )
    collector.check(
        "first beta from source wy branch",
        lambda: np.testing.assert_allclose(
            prefix.beta, oracle["beta"][0], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        "compiled EFS loop retains direct-wz result",
        lambda: check_that(
            int(np.asarray(loop.status)) != _EFS_STATUS_INVALID_WORKING_FACTORS
            and bool(np.all(np.isfinite(np.asarray(loop.pirls_result.coefficients)))),
            f"unexpected loop status {loop.status}",
        ),
    )
    collector.raise_if_any("positive-weight finite-wz fallback")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_efs_retained_start_recovery_matches_pinned_source_once():
    """One pinned I2 trace covers source counters, prefix, and full loop."""
    X = jnp.eye(2)
    y = jnp.asarray([1.0, 100.0])
    wt = jnp.ones(2)
    offset = jnp.zeros(2)
    retained_start = jnp.asarray([-20.0, np.log(100.0)])
    log_theta = jnp.asarray([np.log(0.8)])
    penalty = jnp.eye(2) * 1e-20
    oracle = _r_efs_inner_oracle(
        np.asarray(X),
        np.asarray(y),
        np.asarray(wt),
        np.asarray(offset),
        penalty=1e-20,
        log_theta=float(log_theta[0]),
        start=np.asarray(retained_start),
        null_coef=np.zeros(2),
        maxit=200,
    )

    family = NegativeBinomial(theta=0.8)

    def direct_prefix():
        def dev_fn(current_eta):
            return _efs_nb_log_deviance(current_eta, log_theta, y, wt)

        def compute_working_factors(_mu, current_eta):
            return _efs_nb_observed_working_factors(
                current_eta, log_theta, y, wt, offset
            )

        def form_wls(factors):
            bounded_weight = jnp.clip(factors.weight, -1e10, 1e10)
            rhs = jnp.where(
                factors.use_weighted_response,
                factors.weighted_response,
                bounded_weight * factors.response,
            )
            return (bounded_weight[:, None] * X).T @ X, X.T @ rhs

        return _efs_beta_step_with_recovery(
            X=X,
            S_lambda=penalty,
            offset=offset,
            family=family,
            beta=retained_start,
            eta=retained_start,
            mu=family.link.inverse(retained_start),
            beta_old=jnp.zeros(2),
            eta_old=jnp.zeros(2),
            null_beta=jnp.zeros(2),
            null_eta=jnp.zeros(2),
            initial_start_retained=jnp.asarray(True),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            baseline=dev_fn(jnp.zeros(2)),
            compute_working_factors=compute_working_factors,
            form_wls=form_wls,
            compute_dev=lambda _mu, current_eta: dev_fn(current_eta),
            max_recovery_halvings=200,
        )

    prefix = jax.jit(direct_prefix)()
    limited = efs_theta_pirls_loop(
        X,
        y,
        retained_start,
        penalty,
        family,
        wt,
        offset,
        log_theta,
        y.astype(jnp.int64),
        beta_old_init=jnp.zeros(2),
        initial_eta=retained_start,
        initial_start_retained=True,
        max_y=100,
        integer_counts=True,
        max_iter=1,
    )
    full = efs_theta_pirls_loop(
        X,
        y,
        retained_start,
        penalty,
        family,
        wt,
        offset,
        log_theta,
        y.astype(jnp.int64),
        beta_old_init=jnp.zeros(2),
        initial_eta=retained_start,
        initial_start_retained=True,
        max_y=100,
        integer_counts=True,
        max_iter=200,
    )
    full_eta = np.asarray(full.pirls_result.eta)
    full_nll, full_gradient = jax.value_and_grad(
        lambda current_theta: conditional_theta_nll(
            current_theta,
            full.pirls_result.eta,
            y,
            wt,
            y.astype(jnp.int64),
            family,
            max_y=100,
            integer_counts=True,
        )
    )(full.log_theta)
    full_threshold = 1e-7 * (abs(float(full_nll)) + 1.0)

    halvings = oracle["halvings"]
    collector = _AssertCollector()
    collector.check(
        "retained-start provenance",
        lambda: check_that(bool(halvings["retained"]), "pinned start was reset"),
    )
    collector.check(
        "raw WLS beta",
        lambda: np.testing.assert_allclose(
            oracle["raw_beta"][0], [215628949.025003, np.log(100.0)]
        ),
    )
    collector.check(
        "raw WLS eta",
        lambda: np.testing.assert_allclose(oracle["raw_eta"][0], oracle["raw_beta"][0]),
    )
    collector.check(
        "raw deviance nonfinite",
        lambda: check_that(
            not np.isfinite(oracle["raw_deviance"][0]), "raw deviance was finite"
        ),
    )
    collector.check(
        "source recovery counters",
        lambda: check_that(
            halvings["nonfinite"] == 19
            and halvings["domain"] == 0
            and halvings["first_nonfinite"] == 19
            and halvings["first_domain"] == 0
            and halvings["first_divergence"] == 7
            and halvings["divergence"] >= halvings["first_divergence"],
            f"unexpected counters: {halvings}",
        ),
    )
    collector.check(
        "200-step direct beta prefix",
        lambda: np.testing.assert_allclose(
            prefix.beta, oracle["beta"][0], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        "one-step recovery boundary",
        lambda: check_that(
            int(np.asarray(limited.status)) == _EFS_STATUS_NONFINITE_RECOVERY_FAILED,
            f"unexpected limited status {limited.status}",
        ),
    )
    collector.check(
        "full-loop convergence",
        lambda: check_that(
            bool(np.asarray(full.pirls_result.converged)),
            "full recovery did not converge",
        ),
    )
    collector.check(
        "full-loop final beta",
        lambda: np.testing.assert_allclose(
            full.pirls_result.coefficients,
            oracle["coefficients"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "full-loop returned theta is finite",
        lambda: check_that(
            bool(np.isfinite(np.asarray(full.log_theta[0]))),
            f"nonfinite returned theta {full.log_theta[0]}",
        ),
    )
    # At saturated mu=y this theta coordinate has no finite interior optimum.
    # Its terminal coordinate depends on floating-point reductions, so prove
    # the actual pinned-source and JAX stopping states instead of fixing an
    # architecture-specific theta snapshot.
    theta_state = oracle["theta_state"]
    collector.check(
        "saturated source theta-limit state",
        lambda: check_that(
            np.isfinite(theta_state["nll"])
            and np.isfinite(theta_state["gradient"])
            and theta_state["log_theta"] > 15.0
            and abs(theta_state["gradient"]) <= theta_state["threshold"],
            f"source theta state did not satisfy its stopping rule: {theta_state}",
        ),
    )
    collector.check(
        "saturated JAX theta-limit state",
        lambda: check_that(
            np.isfinite(float(full_nll))
            and np.isfinite(float(full_gradient[0]))
            and float(np.asarray(full.log_theta[0])) > 15.0
            and abs(float(full_gradient[0])) <= full_threshold,
            "JAX theta state did not satisfy its stopping rule: "
            f"theta={full.log_theta[0]}, nll={full_nll}, "
            f"gradient={full_gradient[0]}, threshold={full_threshold}",
        ),
    )
    collector.check(
        "saturated source/JAX conditional score",
        lambda: np.testing.assert_allclose(
            full_nll,
            theta_state["nll"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "saturated source/JAX predictor",
        lambda: np.testing.assert_allclose(
            full_eta,
            oracle["coefficients"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "saturated source/JAX deviance",
        lambda: np.testing.assert_allclose(
            full.pirls_result.deviance,
            theta_state["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any("retained-start recovery parity")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    ("label", "first", "second", "theta", "weights", "offset"),
    [
        ("retained20", -20.0, 100.0, 0.8, [1.0, 1.0], [0.0, 0.0]),
        (
            "retained24_weighted_offset",
            -24.0,
            300.0,
            1.3,
            [0.7, 1.6],
            [0.1, -0.2],
        ),
    ],
)
def test_efs_production_observed_prefix_matches_pinned_recovery(
    label: str,
    first: float,
    second: float,
    theta: float,
    weights: list[float],
    offset: list[float],
):
    """Production observed W/z keeps valid small-curvature recovery exact."""
    X = jnp.eye(2)
    y = jnp.asarray([1.0, second])
    wt = jnp.asarray(weights)
    offset_array = jnp.asarray(offset)
    eta = jnp.asarray([first, np.log(second)])
    beta = eta - offset_array
    log_theta = jnp.asarray([np.log(theta)])
    penalty = jnp.eye(2) * 1e-20
    family = NegativeBinomial(theta=theta)
    oracle = _r_efs_inner_oracle(
        np.asarray(X),
        np.asarray(y),
        np.asarray(wt),
        np.asarray(offset_array),
        penalty=1e-20,
        log_theta=float(log_theta[0]),
        start=np.asarray(beta),
        null_coef=np.zeros(2),
        maxit=200,
    )

    def run_prefix():
        def dev_fn(current_eta):
            return _efs_nb_log_deviance(current_eta, log_theta, y, wt)

        def compute_working_factors(_mu, current_eta):
            return _efs_nb_observed_working_factors(
                current_eta, log_theta, y, wt, offset_array
            )

        def form_wls(factors):
            bounded_weight = jnp.clip(factors.weight, -1e10, 1e10)
            rhs = jnp.where(
                factors.use_weighted_response,
                factors.weighted_response,
                bounded_weight * factors.response,
            )
            return (bounded_weight[:, None] * X).T @ X, X.T @ rhs

        return _efs_beta_step_with_recovery(
            X=X,
            S_lambda=penalty,
            offset=offset_array,
            family=family,
            beta=beta,
            eta=eta,
            mu=family.link.inverse(eta),
            beta_old=jnp.zeros(2),
            eta_old=offset_array,
            null_beta=jnp.zeros(2),
            null_eta=offset_array,
            initial_start_retained=jnp.asarray(True),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            baseline=dev_fn(offset_array),
            compute_working_factors=compute_working_factors,
            form_wls=form_wls,
            compute_dev=lambda _mu, current_eta: dev_fn(current_eta),
            max_recovery_halvings=200,
        )

    result = jax.jit(run_prefix)()
    collector = _AssertCollector()
    collector.check(
        f"{label} accepted",
        lambda: check_that(bool(result.accepted), f"status={result.failure_status}"),
    )
    collector.check(
        f"{label} beta",
        lambda: np.testing.assert_allclose(
            result.beta, oracle["beta"][0], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        f"{label} eta",
        lambda: np.testing.assert_allclose(
            result.eta,
            oracle["beta"][0] + np.asarray(offset_array),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        f"{label} penalized deviance",
        lambda: np.testing.assert_allclose(
            result.penalized_deviance,
            oracle["pre"][0],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any(f"{label} production EFS recovery")
