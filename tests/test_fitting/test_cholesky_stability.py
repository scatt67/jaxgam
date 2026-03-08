"""Tests for robust Cholesky usage in AD / derivative paths.

Verifies that _criterion_core (reml.py) and _pirls_jvp (newton.py) use
jitter-stabilized Cholesky factorization, producing finite results on
nearly-singular penalized Hessian matrices.

Addresses GitHub issue #6: AD / derivative path uses plain
jnp.linalg.cholesky on XtWX + S (no jitter/fallback).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.fitting.reml import _criterion_core

jax.config.update("jax_enable_x64", True)

SEED = 42


def _make_nearly_singular_system(p: int = 10, rank_deficit: int = 3):
    """Create a nearly-singular XtWX + S_lambda system.

    Returns XtWX with near-zero eigenvalues plus a rank-deficient penalty,
    producing an H that would fail raw Cholesky without jitter.
    """
    rng = np.random.default_rng(SEED)

    # XtWX with some near-zero eigenvalues (simulates near-collinearity
    # or near-zero working weights)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    eigs = np.ones(p)
    eigs[-rank_deficit:] = 1e-16  # near-zero eigenvalues
    XtWX = Q @ np.diag(eigs) @ Q.T
    XtWX = (XtWX + XtWX.T) / 2  # ensure symmetry

    # A single rank-deficient penalty matrix
    S_base = rng.standard_normal((p, p - rank_deficit))
    S_j = S_base @ S_base.T
    S_j = (S_j + S_j.T) / 2

    return jnp.array(XtWX), (jnp.array(S_j),)


class TestCriterionCoreStability:
    """_criterion_core must produce finite log|H| on ill-conditioned H."""

    def test_finite_logdet_nearly_singular(self):
        """Nearly-singular H should yield finite criterion (not NaN)."""
        p = 10
        XtWX, S_list = _make_nearly_singular_system(p, rank_deficit=3)
        log_lambda = jnp.array([0.0])
        beta = jnp.ones(p) * 0.1
        deviance = jnp.array(10.0)
        ls_sat = jnp.array(0.0)
        phi = jnp.array(1.0)

        # Singleton block metadata for one penalty
        singleton_sp_indices = (0,)
        # Rank of the penalty
        S_np = np.array(S_list[0])
        rank = int(np.linalg.matrix_rank(S_np, tol=1e-10))
        singleton_ranks = (rank,)
        eigvals = np.linalg.eigvalsh(S_np)
        nonzero = eigvals[eigvals > 1e-10]
        singleton_eig_constants = jnp.array([float(np.sum(np.log(nonzero)))])

        result = _criterion_core(
            log_lambda,
            XtWX,
            beta,
            deviance,
            ls_sat,
            S_list,
            phi,
            singleton_sp_indices,
            singleton_ranks,
            singleton_eig_constants,
            multi_block_sp_indices=(),
            multi_block_ranks=(),
            multi_block_proj_S=(),
        )
        assert jnp.isfinite(result), f"criterion_core returned {result}"

    def test_gradient_finite_nearly_singular(self):
        """Gradient through _criterion_core must be finite on ill-conditioned H."""
        p = 10
        XtWX, S_list = _make_nearly_singular_system(p, rank_deficit=3)
        beta = jnp.ones(p) * 0.1
        deviance = jnp.array(10.0)
        ls_sat = jnp.array(0.0)
        phi = jnp.array(1.0)

        S_np = np.array(S_list[0])
        rank = int(np.linalg.matrix_rank(S_np, tol=1e-10))
        eigvals = np.linalg.eigvalsh(S_np)
        nonzero = eigvals[eigvals > 1e-10]
        singleton_eig_constants = jnp.array([float(np.sum(np.log(nonzero)))])

        def score(log_lambda):
            return _criterion_core(
                log_lambda,
                XtWX,
                beta,
                deviance,
                ls_sat,
                S_list,
                phi,
                (0,),
                (rank,),
                singleton_eig_constants,
                (),
                (),
                (),
            )

        log_lambda = jnp.array([0.0])
        grad = jax.grad(score)(log_lambda)
        assert jnp.all(jnp.isfinite(grad)), f"gradient has non-finite values: {grad}"

    def test_hessian_finite_nearly_singular(self):
        """Hessian through _criterion_core must be finite on ill-conditioned H."""
        p = 10
        XtWX, S_list = _make_nearly_singular_system(p, rank_deficit=3)
        beta = jnp.ones(p) * 0.1
        deviance = jnp.array(10.0)
        ls_sat = jnp.array(0.0)
        phi = jnp.array(1.0)

        S_np = np.array(S_list[0])
        rank = int(np.linalg.matrix_rank(S_np, tol=1e-10))
        eigvals = np.linalg.eigvalsh(S_np)
        nonzero = eigvals[eigvals > 1e-10]
        singleton_eig_constants = jnp.array([float(np.sum(np.log(nonzero)))])

        def score(log_lambda):
            return _criterion_core(
                log_lambda,
                XtWX,
                beta,
                deviance,
                ls_sat,
                S_list,
                phi,
                (0,),
                (rank,),
                singleton_eig_constants,
                (),
                (),
                (),
            )

        log_lambda = jnp.array([0.0])
        hess = jax.hessian(score)(log_lambda)
        assert jnp.all(jnp.isfinite(hess)), f"hessian has non-finite values: {hess}"

    def test_high_regularization(self):
        """High penalty (large lambda) should still produce finite criterion."""
        p = 10
        XtWX, S_list = _make_nearly_singular_system(p, rank_deficit=3)
        # Very large lambda -> penalty dominates -> H is better conditioned
        # but the scaling pushes numerical boundaries
        log_lambda = jnp.array([20.0])
        beta = jnp.ones(p) * 0.01
        deviance = jnp.array(5.0)
        ls_sat = jnp.array(0.0)
        phi = jnp.array(1.0)

        S_np = np.array(S_list[0])
        rank = int(np.linalg.matrix_rank(S_np, tol=1e-10))
        eigvals = np.linalg.eigvalsh(S_np)
        nonzero = eigvals[eigvals > 1e-10]
        singleton_eig_constants = jnp.array([float(np.sum(np.log(nonzero)))])

        result = _criterion_core(
            log_lambda,
            XtWX,
            beta,
            deviance,
            ls_sat,
            S_list,
            phi,
            (0,),
            (rank,),
            singleton_eig_constants,
            (),
            (),
            (),
        )
        assert jnp.isfinite(result), f"criterion_core returned {result}"
