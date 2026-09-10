"""Pinned ``gam.fit3`` checks for the regular-family EFS inner loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.fitting.efs_regular_pirls import (
    EFS_REGULAR_STATUS_CONVERGED,
    _efs_regular_pirls_loop_jit,
    efs_regular_pirls_loop,
)
from jaxgam.fitting.reml import estimate_edf, fletcher_scale
from jaxgam.links.links import LogLink
from tests.helpers import r_available
from tests.r_bridge import RBridge
from tests.tolerances import STRICT


def _gaussian_log_recovery_fit(penalty: jax.Array, *, tolerance: float):
    X = jnp.eye(2, dtype=jnp.float64)
    y = jnp.array([1.0, 2.0], dtype=jnp.float64)
    start = jnp.array([-20.0, np.log(2.0)], dtype=jnp.float64)
    null = jnp.array([0.0, np.log(2.0)], dtype=jnp.float64)
    return efs_regular_pirls_loop(
        X,
        y,
        start,
        X @ start,
        null,
        X @ null,
        penalty,
        Gaussian(LogLink()),
        jnp.ones(2, dtype=jnp.float64),
        jnp.zeros(2, dtype=jnp.float64),
        jnp.array(1.0, dtype=jnp.float64),
        start_present=True,
        max_iter=100,
        tol=tolerance,
    )


def test_first_regular_newton_step_matches_signed_working_system() -> None:
    X = jnp.array(
        [[1.0, -0.8], [1.0, -0.2], [1.0, 0.4], [1.0, 0.9]],
        dtype=jnp.float64,
    )
    y = jnp.array([0.7, 1.1, 1.9, 2.8], dtype=jnp.float64)
    beta = jnp.array([0.2, 0.15], dtype=jnp.float64)
    eta = X @ beta
    mu = jnp.exp(eta)
    weight = mu * (2.0 * mu - y)
    response = eta + (y - mu) / (2.0 * mu - y)
    penalty = jnp.diag(jnp.array([0.0, 0.35], dtype=jnp.float64))
    expected = np.linalg.solve(
        np.asarray((weight[:, None] * X).T @ X + penalty),
        np.asarray(X.T @ (weight * response)),
    )

    result = efs_regular_pirls_loop(
        X,
        y,
        beta,
        eta,
        jnp.asarray(expected),
        X @ jnp.asarray(expected),
        penalty,
        Gaussian(LogLink()),
        jnp.ones(4, dtype=jnp.float64),
        None,
        jnp.array(1.0, dtype=jnp.float64),
        start_present=True,
        max_iter=1,
        tol=1e-7,
    )

    np.testing.assert_allclose(
        result.pre_gdi1_coefficients,
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert int(result.pirls_result.n_iter) == 1


def test_retained_start_recovery_keeps_pre_and_post_gdi1_provenance() -> None:
    result = _gaussian_log_recovery_fit(jnp.zeros((2, 2)), tolerance=1e-7)

    # Pinned R 4.5.2 / mgcv 1.9-3, gam.fit3 with X=I and the same retained
    # start.  The main loop first retries with Fisher scoring and then recovers
    # toward the retained/null anchor before C_gdi1 performs its final solve.
    np.testing.assert_allclose(
        result.pre_gdi1_coefficients,
        [1.10313789e-4, np.log(2.0)],
        rtol=STRICT.rtol,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.gdi1_coefficients,
        [1.82507900e-8, np.log(2.0)],
        rtol=STRICT.rtol,
        atol=1e-15,
    )
    assert bool(result.gdi1_candidate_valid)
    assert bool(result.gdi1_coefficients_selected)
    assert int(result.status) == EFS_REGULAR_STATUS_CONVERGED
    np.testing.assert_array_equal(
        result.pirls_result.coefficients, result.gdi1_coefficients
    )
    np.testing.assert_allclose(
        result.pre_gdi1_deviance,
        1.2170474531475107e-8,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_final_gdi1_score_penalty_is_distinct_from_stopping_penalty() -> None:
    penalty = 0.5 * jnp.ones((2, 2), dtype=jnp.float64)
    result = _gaussian_log_recovery_fit(penalty, tolerance=1e-2)

    # Pinned R 4.5.2 / mgcv 1.9-3 with UrS=sqrt(2), lambda=.5 and the same
    # input.  A deliberately loose inner epsilon makes the final gdi1 polish
    # appreciable, so a pre/post provenance mix-up cannot pass this fixture.
    np.testing.assert_allclose(
        result.pre_gdi1_coefficients,
        [-0.27459290492714816, 0.64664114928523397],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.gdi1_coefficients,
        [-0.27698022098910069, 0.64378931204395928],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.pre_gdi1_penalized_deviance,
        0.13512649750661931,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.gdi1_penalty,
        0.067274454640245798,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert not np.isclose(
        float(result.pirls_result.penalized_deviance),
        float(result.pre_gdi1_penalized_deviance),
        rtol=1e-5,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.pirls_result.penalized_deviance,
        0.13319100408189812,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_final_gdi1_provenance_matches_live_pinned_r() -> None:
    X = np.eye(2, dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)
    start = np.array([-20.0, np.log(2.0)], dtype=np.float64)
    null = np.array([0.0, np.log(2.0)], dtype=np.float64)
    penalty = 0.5 * np.ones((2, 2), dtype=np.float64)
    result = _gaussian_log_recovery_fit(jnp.asarray(penalty), tolerance=1e-2)
    oracle = RBridge(mode="subprocess").efs_regular_gdi1_diagnostics(
        X,
        y,
        penalty,
        start,
        null,
        tolerance=1e-2,
    )

    np.testing.assert_allclose(
        result.pre_gdi1_coefficients,
        oracle["pre_beta"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.gdi1_coefficients,
        oracle["gdi_beta"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.pirls_result.coefficients,
        oracle["selected_beta"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.pre_gdi1_deviance,
        oracle["pre_deviance"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.pre_gdi1_penalized_deviance,
        oracle["pre_pdev"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.gdi1_penalty,
        oracle["gdi_penalty"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.pirls_result.XtWX,
        oracle["XtWX"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.pirls_result.XtWX_fisher,
        oracle["XtWX_fisher"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.pirls_result.working_weights,
        oracle["observed_weight"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.pirls_result.eta,
        oracle["selected_eta"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.pirls_result.mu,
        oracle["selected_mu"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    edf = estimate_edf(result.pirls_result.XtWX_fisher, result.pirls_result.L_fisher)
    reported_scale = fletcher_scale(
        jnp.asarray(y),
        result.pirls_result.mu,
        jnp.ones_like(jnp.asarray(y)),
        Gaussian(LogLink()),
        edf,
    )
    np.testing.assert_allclose(
        reported_scale,
        oracle["reported_scale"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert oracle["source_commit"] == "fb7e8e718377513e78ba6c6bf7e60757fc6a32a9"


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_invalid_gdi1_candidate_returns_pre_gdi1_feasible_state() -> None:
    X = jnp.eye(2, dtype=jnp.float64)
    y = jnp.array([0.01, 100.0], dtype=jnp.float64)
    start = y + 0.1
    penalty = 0.1 * jnp.ones((2, 2), dtype=jnp.float64)
    result = efs_regular_pirls_loop(
        X,
        y,
        start,
        start,
        start,
        start,
        penalty,
        Poisson("identity"),
        jnp.ones(2, dtype=jnp.float64),
        None,
        jnp.array(1.0, dtype=jnp.float64),
        start_present=True,
        max_iter=100,
        tol=0.1,
    )
    oracle = RBridge(mode="subprocess").efs_regular_gdi1_diagnostics(
        np.eye(2),
        np.asarray(y),
        np.asarray(penalty),
        np.asarray(start),
        np.asarray(start),
        family="poisson",
        link="identity",
        tolerance=0.1,
    )

    assert not bool(result.gdi1_candidate_valid)
    assert not bool(result.gdi1_coefficients_selected)
    assert result.gdi1_coefficients[0] < 0.0
    np.testing.assert_allclose(
        result.pre_gdi1_coefficients,
        oracle["pre_beta"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.gdi1_coefficients,
        oracle["gdi_beta"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.pirls_result.coefficients,
        oracle["selected_beta"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_array_equal(
        result.pirls_result.coefficients, result.pre_gdi1_coefficients
    )
    np.testing.assert_allclose(
        result.gdi1_penalty,
        oracle["gdi_penalty"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    selected_penalty = (
        result.pirls_result.coefficients @ penalty @ result.pirls_result.coefficients
    )
    assert not np.isclose(
        float(result.gdi1_penalty),
        float(selected_penalty),
        rtol=1e-5,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.pirls_result.penalized_deviance,
        result.pre_gdi1_deviance + result.gdi1_penalty,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_regular_efs_inner_loop_compiles_as_one_jax_program() -> None:
    X = jnp.eye(2, dtype=jnp.float64)
    y = jnp.array([1.0, 2.0], dtype=jnp.float64)
    beta = jnp.array([0.0, np.log(2.0)], dtype=jnp.float64)
    family = Gaussian(LogLink())
    executable = _efs_regular_pirls_loop_jit.lower(
        X,
        y,
        beta,
        beta,
        beta,
        beta,
        jnp.zeros((2, 2), dtype=jnp.float64),
        family,
        jnp.ones(2, dtype=jnp.float64),
        jnp.zeros(2, dtype=jnp.float64),
        jnp.array(1.0, dtype=jnp.float64),
        start_present=True,
        max_iter=2,
        tol=1e-7,
    ).compile()
    result = executable(
        X,
        y,
        beta,
        beta,
        beta,
        beta,
        jnp.zeros((2, 2), dtype=jnp.float64),
        jnp.ones(2, dtype=jnp.float64),
        jnp.zeros(2, dtype=jnp.float64),
        jnp.array(1.0, dtype=jnp.float64),
    )
    assert np.all(np.isfinite(np.asarray(result.pirls_result.coefficients)))
