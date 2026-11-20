"""gam.reparam coordinates of the dense fit (gam.fit3.r 144-179, 812-813)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM
from jaxgam.families.standard import Gamma, Gaussian
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.newton import NewtonOptimizer, pirls_from_stable
from jaxgam.fitting.stable_log_det import stable_block_roots, stable_system
from tests.helpers import _AssertCollector, _setup_fd
from tests.tolerances import MODERATE, STRICT

_TE = "y ~ te(x0, x1, bs='cr', k=6)"
_TI = "y ~ ti(x0, x1, bs='cr', k=6)"


def _close(actual, expected, tolerance) -> None:
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=tolerance.rtol,
        atol=tolerance.atol,
    )


@pytest.fixture(scope="module")
def f4_data() -> pd.DataFrame:
    """Acceptance finding F4: a te whose REML log sp spread ~20."""
    rng = np.random.default_rng(4)
    n = 1000
    data = pd.DataFrame({"x0": rng.uniform(0, 1, n), "x1": rng.uniform(0, 1, n)})
    data["y"] = np.sin(4 * np.pi * data.x0) + rng.normal(0, 0.02, n)
    return data


def _system(fd, log_lambda):
    blocks = stable_block_roots(
        fd.multi_block_sp_indices, fd.multi_block_proj_S, fd.penalty_structure
    )
    return stable_system(
        blocks,
        fd.penalty_structure,
        fd.penalty_roots,
        np.asarray(log_lambda, dtype=np.float64),
        deriv=2,
    )


def test_rotation_is_orthogonal_and_jit_compiles() -> None:
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotation = penalty_ops.JaxCoefficientRotation((1,), (5,), (jnp.asarray(Q),))
    beta = jnp.asarray(rng.normal(size=6))
    M = jnp.asarray(rng.normal(size=(6, 6)))
    X = jnp.asarray(rng.normal(size=(3, 6)))

    @jax.jit
    def roundtrip(rotation, beta, M, X):
        stable = rotation.to_stable(beta)
        return (
            rotation.from_stable(stable),
            rotation.from_stable_matrix(rotation.to_stable_matrix(M)),
            rotation.design(X) @ stable,
        )

    beta_back, M_back, eta = roundtrip(rotation, beta, M, X)
    _close(beta_back, beta, STRICT)
    _close(M_back, M, STRICT)
    # X b = (X T)(T' b): the linear predictor is coordinate free.
    _close(eta, X @ beta, STRICT)
    np.testing.assert_array_equal(
        np.asarray(beta_back)[[0, 5]], np.asarray(beta)[[0, 5]]
    )


@pytest.mark.parametrize("formula", [_TE, _TI])
def test_stable_system_places_get_stables_penalties(f4_data, formula) -> None:
    """T is orthogonal; penalties and roots are rS_j rS_j' and [rS_j', 0]."""
    fd = _setup_fd(formula, f4_data, Gaussian())
    collector = _AssertCollector()
    for log_lambda in ([-6.02, 3.98], [-6.02, 33.98], [20.0, -3.0]):
        system = _system(fd, log_lambda)
        assert len(system.rotation.transforms) == 1
        start, stop, T = system.host_rotation[0]
        size = stop - start
        block = next(b for b in system.penalty_structure.blocks if b.start == start)
        original = next(b for b in fd.penalty_structure.blocks if b.start == start)
        roots = {
            sp: np.asarray(root)
            for s, sp, root in zip(
                system.penalty_roots.starts,
                system.penalty_roots.sp_indices,
                system.penalty_roots.roots,
                strict=True,
            )
            if s == start
        }
        rank = fd.multi_block_ranks[0]

        def orthogonal(T=T, size=size) -> None:
            _close(T.T @ T, np.eye(size), STRICT)

        def exact_null(block=block, rank=rank) -> None:
            for penalty in block.penalties:
                values = np.asarray(penalty.values)
                assert not np.any(values[rank:, :])
                assert not np.any(values[:, rank:])

        def roots_square_to_penalties(block=block, roots=roots) -> None:
            for sp, penalty in zip(block.sp_indices, block.penalties, strict=True):
                _close(roots[sp].T @ roots[sp], np.asarray(penalty.values), STRICT)

        def same_penalties(block=block, original=original, T=T) -> None:
            # T' S_j T, up to the rank truncation of the range roots.
            for rotated, penalty in zip(
                block.penalties, original.penalties, strict=True
            ):
                S = np.asarray(penalty.values)
                scale = np.linalg.norm(S)
                _close(
                    np.asarray(rotated.values) / scale, T.T @ S @ T / scale, MODERATE
                )

        label = f"log sp {log_lambda}"
        collector.check(f"{label} orthogonal", orthogonal)
        collector.check(f"{label} exact null", exact_null)
        collector.check(f"{label} roots", roots_square_to_penalties)
        collector.check(f"{label} penalties", same_penalties)
    collector.raise_if_any(formula)


def test_rotated_fit_and_score_compiles_and_maps_back(f4_data) -> None:
    """The jitted fused fit runs in stable coordinates and returns fitting ones."""
    fd = _setup_fd(_TE, f4_data, Gaussian())
    optimizer = NewtonOptimizer(fd, pin_lambda=True)
    params = jnp.asarray([-6.02, 3.98, np.log(0.0185)])
    pirls, score = optimizer._fit_and_score(params, fd.beta_init)
    system = optimizer._stable_system(params, deriv=0)
    assert system is not None
    assert bool(pirls.converged)
    assert np.isfinite(float(score))
    # gam.fit3's gradient test, recorded by the loop, passed.
    assert float(pirls.inner_gradient_ratio) <= 1.0
    stable_beta = system.rotation.to_stable(pirls.coefficients)
    # The factor is that of the stable-coordinate Hessian.
    H = system.rotation.to_stable_matrix(pirls.XtWX) + penalty_ops.materialize(
        system.penalty_structure, params[:2]
    )
    L = np.asarray(pirls.L)
    _close(L @ L.T / np.linalg.norm(H), np.asarray(H) / np.linalg.norm(H), STRICT)
    _close(
        fd.X @ pirls.coefficients, system.rotation.design(fd.X) @ stable_beta, STRICT
    )
    grad, hess = optimizer._diff_grad_hess(params, pirls.coefficients)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.all(np.isfinite(np.asarray(hess)))

    mapped = pirls_from_stable(pirls, system.rotation)
    assert mapped.fisher_edf is not None


def test_well_conditioned_fixed_sp_fit_is_unchanged_by_rotation(f4_data) -> None:
    """At a small spread the stable and fitting coordinates agree to rounding."""
    fd = _setup_fd(_TE, f4_data, Gaussian())
    optimizer = NewtonOptimizer(fd, pin_lambda=True)
    params = jnp.asarray([-1.0, 1.0, np.log(0.0185)])
    rotated, rotated_score = optimizer._fit_and_score(params, fd.beta_init)
    kwargs = dict(optimizer._jit_kwargs)
    kwargs["multi_block_proj_S"] = optimizer._stable_system(
        params, deriv=0
    ).log_det_terms
    from jaxgam.fitting.newton import _jit_fit_and_score

    plain_score, plain = _jit_fit_and_score(params, fd.beta_init, **kwargs)
    _close(rotated.coefficients, plain.coefficients, STRICT)
    _close(rotated.mu, plain.mu, STRICT)
    _close(rotated_score, plain_score, STRICT)


# ---------------------------------------------------------------------------
# Pinned R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", ["te", "te_by"])
def test_gamma_inner_loop_matches_gam_fit3_iterations(cell, r_bridge) -> None:
    """The fresh-start inner fit converges in pinned gam.fit3's count +/- 1.

    At ``SOURCE_EPSILON`` the gradient test sits at a rounding floor in fixed
    coordinates (ratio 0.83-0.96, above 1 on x86_64; te_by fails outright at
    R's sp). In gam.reparam's coordinates it clears the threshold by more
    than an order of magnitude, as R's does.
    """
    from tests import test_validation_matrix as vm

    config = vm.SMOOTH_CONFIGS[cell]
    data = vm._get_data(config, "gamma")
    estimated = r_bridge.source_gam_fit3_inner_trace(config.r_formula, data, "gamma")
    sp = estimated.smoothing_params
    fixed = r_bridge.source_gam_fit3_inner_trace(config.r_formula, data, "gamma", sp=sp)
    first = fixed.calls[0]
    assert first.converged
    assert first.final_ratio <= 1.0

    fd = _setup_fd(config.py_formula, data, Gamma())
    optimizer = NewtonOptimizer(fd, pin_lambda=True)
    log_lambda = jnp.log(jnp.asarray(sp))
    result = optimizer._eager_inner_fit(log_lambda, fd.S_lambda(log_lambda), None)
    assert bool(result.converged), "inner fit did not converge"
    assert abs(int(result.n_iter) - first.n_iter) <= 1, (
        f"JaxGAM {int(result.n_iter)} iterations, gam.fit3 {first.n_iter}"
    )
    assert float(result.inner_gradient_ratio) <= 1.0


@pytest.mark.parametrize("gap", [10.0, 20.0, 30.0, 40.0])
@pytest.mark.parametrize("formula", [_TE, _TI])
def test_spread_sweep_matches_gam_fit3(f4_data, formula, gap, r_bridge) -> None:
    """Score, gradient and Hessian at fixed spread sp, against gam.fit3's own.

    Measured on this host: score within 1e-14 relative and REML2 within
    5.3e-12 absolute (STRICT) at every gap. REML1 is a near-stationary
    cancellation residual of O(1) terms, so its rounding floor is absolute:
    measured 5.3e-12 / 1.3e-12 / 8.2e-13 / 6.0e-13 (te) and <= 7.6e-13 (ti)
    at gaps 10 / 20 / 30 / 40, which leaves STRICT's relative bound only on
    te's 0.0135 and 0.0025 entries at gaps 10 and 20 (3.9e-10, 5.4e-10
    relative): it is compared one class out, at MODERATE.
    """
    log_sp = np.array([-6.02, -6.02 + gap])
    sp = np.exp(log_sp)
    fixed = r_bridge.fit_fixed_sp(formula, f4_data, "gaussian", sp)
    model = GAM(formula, family="gaussian", sp=sp).fit(f4_data)
    log_phi = float(np.log(model.scale))
    reference = r_bridge.source_gam_fit3_reml_derivatives(
        formula, f4_data, "gaussian", log_sp, log_scale=log_phi
    )
    fd = _setup_fd(formula, f4_data, Gaussian())
    optimizer = NewtonOptimizer(fd, pin_lambda=True)
    params = jnp.asarray(np.r_[log_sp, log_phi])
    pirls, score = optimizer._fit_and_score(params, fd.beta_init)
    gradient, hessian = optimizer._diff_grad_hess(params, pirls.coefficients)

    collector = _AssertCollector()
    collector.check("public score", lambda: _close(model.score, fixed["score"], STRICT))
    collector.check(
        "fitted values",
        lambda: _close(model.fitted_values, fixed["fitted_values"], STRICT),
    )
    collector.check("REML", lambda: _close(score, reference.score, STRICT))
    collector.check("REML1", lambda: _close(gradient, reference.gradient, MODERATE))
    collector.check("REML2", lambda: _close(hessian, reference.hessian, STRICT))
    collector.raise_if_any(f"{formula} gap {gap:g}")


def test_f4_converges_on_the_flat_ridge(f4_data, r_bridge) -> None:
    """F4 by the flat-ridge rule (R's second log sp lies on a flat ridge).

    JaxGAM converges; its criterion at its own endpoint is no higher than R's
    at R's endpoint, each evaluated by both sides; fitted values, deviance,
    EDF and score agree with pinned gam at MODERATE (measured: fitted 4.5e-6,
    deviance 1.4e-6 rel, EDF 9.0e-5 rel, score 3.4e-7 rel, all
    endpoint-dependent). The selected sp are not compared.
    """
    reference = r_bridge.fit_gam(_TE, f4_data, "gaussian", "REML")
    model = GAM(_TE, family="gaussian").fit(f4_data)
    assert model.converged, model.convergence_info
    own = np.asarray(model.smoothing_params)
    theirs = np.asarray(reference["smoothing_params"])
    j_at_j = GAM(_TE, family="gaussian", sp=own).fit(f4_data).score
    j_at_r = GAM(_TE, family="gaussian", sp=theirs).fit(f4_data).score
    r_at_j = r_bridge.fit_fixed_sp(_TE, f4_data, "gaussian", own)["score"]
    r_at_r = r_bridge.fit_fixed_sp(_TE, f4_data, "gaussian", theirs)["score"]

    collector = _AssertCollector()
    collector.check("JaxGAM's criterion", lambda: _assert_le(j_at_j, j_at_r))
    collector.check("R's criterion", lambda: _assert_le(r_at_j, r_at_r))
    collector.check("endpoint scores agree", lambda: _close(j_at_j, r_at_j, STRICT))
    collector.check(
        "fitted",
        lambda: _close(model.fitted_values, reference["fitted_values"], MODERATE),
    )
    collector.check(
        "deviance", lambda: _close(model.deviance, reference["deviance"], MODERATE)
    )
    collector.check(
        "edf", lambda: _close(model.edf_total, reference["edf_total"], MODERATE)
    )
    collector.check(
        "score", lambda: _close(model.score, reference["reml_score"], MODERATE)
    )
    collector.raise_if_any("F4")


def _assert_le(value: float, bound: float) -> None:
    assert value <= bound, f"{value!r} > {bound!r}"
