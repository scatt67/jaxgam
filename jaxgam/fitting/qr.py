"""Pure JIT QR kernels: pivot-aware triangular actions and the penalized solve."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla


@jax.jit
def qr_root_inverse(R: jax.Array, qtz: jax.Array, pivots: jax.Array) -> jax.Array:
    """Return P R^-1 qtz for A P = Q R; qtz is already QR ordered."""
    x = jsla.solve_triangular(R, qtz, lower=False)
    return jnp.zeros_like(x).at[pivots].set(x)


@jax.jit
def qr_root_transpose_inverse(
    R: jax.Array, rhs: jax.Array, pivots: jax.Array
) -> jax.Array:
    """Return R^-T P^T rhs."""
    return jsla.solve_triangular(R, rhs[pivots], lower=False, trans="T")


@jax.jit
def qr_hessian_inverse(R: jax.Array, rhs: jax.Array, pivots: jax.Array) -> jax.Array:
    """Return P R^-1 R^-T P^T rhs."""
    return qr_root_inverse(R, qr_root_transpose_inverse(R, rhs, pivots), pivots)


@jax.jit
def penalized_qr_solve(
    weighted_X: jax.Array, weighted_z: jax.Array, penalty_root: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Solve the positive-weight penalized least squares problem by QR.

    Port of ``pls_fit1``'s arithmetic (gdi.c) for a structurally full-rank
    system: QR of ``sqrt(W) X`` gives ``R1`` and ``f = Q1' sqrt(W) z``; a
    second QR of ``[R1; E]`` with right side ``[f; 0]`` gives ``R2`` and the
    coefficients ``R2^-1 g``.  No normal equations are formed, so the solve
    error scales with ``cond([sqrt(W) X; E])`` rather than its square, and no
    jitter is ever added.

    Returns the coefficients and ``L``, the transpose of ``R2`` with row signs
    fixed so its diagonal is positive: ``L L' = X'WX + E'E`` exactly as a
    Cholesky factor, for EDF, covariance and IFT consumers.
    """
    q1, r1 = jnp.linalg.qr(weighted_X)
    f = q1.T @ weighted_z
    q2, r2 = jnp.linalg.qr(jnp.concatenate([r1, penalty_root], axis=0))
    g = q2.T @ jnp.concatenate([f, jnp.zeros(penalty_root.shape[0], dtype=f.dtype)])
    beta = jsla.solve_triangular(r2, g, lower=False)
    sign = jnp.where(jnp.diag(r2) < 0.0, -1.0, 1.0)
    return beta, (sign[:, None] * r2).T
