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


@jax.jit
def penalized_qr_hat(
    weighted_X: jax.Array, penalty_root: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """EDF of a nonnegative-weight penalized system from its QR, as gdi does.

    With ``sqrt(W) X = Q1 R1`` and ``[R1; E] = Q2 R2``, the leading rows
    ``T`` of ``Q2`` satisfy ``R1 = T R2``, so the influence (hat) matrix of
    the coefficients is ``F = (X'WX + E'E)^-1 X'WX = R2^-1 T' R1`` and
    ``tr(F) = |T|_F^2 = |R1 R2^-1|_F^2`` (gdi1's ``P K' sqrt(w) X``, gdi.c
    2763-2811).  No normal-equation product is formed.

    Returns ``(F, edf)`` with ``edf = |T|_F^2``.
    """
    _, r1 = jnp.linalg.qr(weighted_X)
    q2, r2 = jnp.linalg.qr(jnp.concatenate([r1, penalty_root], axis=0))
    top = q2[: r1.shape[0]]
    F = jsla.solve_triangular(r2, top.T @ r1, lower=False)
    return F, jnp.sum(top * top)


def _qr_r2(weighted_X: jax.Array, penalty_root: jax.Array) -> jax.Array:
    _, r1 = jnp.linalg.qr(weighted_X)
    return jnp.linalg.qr(jnp.concatenate([r1, penalty_root], axis=0), mode="r")


@jax.custom_jvp
def qr_hessian_inverse_dense(
    H: jax.Array, weighted_X: jax.Array, penalty_root: jax.Array
) -> jax.Array:
    """``H^-1`` from the QR factor ``R2`` of ``[sqrt(W) X; E]`` (``H = R2'R2``).

    ``H`` carries the derivative information (its tangent is ``dX'WX + dS``);
    the factor inputs only supply the primal and are treated as constants.
    """
    del H
    r2 = _qr_r2(weighted_X, penalty_root)
    inverse_root = jsla.solve_triangular(
        r2, jnp.eye(r2.shape[0], dtype=r2.dtype), lower=False
    )
    return inverse_root @ inverse_root.T


@qr_hessian_inverse_dense.defjvp
def _qr_hessian_inverse_dense_jvp(primals, tangents):
    H, weighted_X, penalty_root = primals
    dH = tangents[0]
    H_inv = qr_hessian_inverse_dense(H, weighted_X, penalty_root)
    return H_inv, -H_inv @ dH @ H_inv


@jax.custom_jvp
def qr_log_det(
    H: jax.Array, weighted_X: jax.Array, penalty_root: jax.Array
) -> jax.Array:
    """``log|H|`` as ``2 sum log|diag R2|`` from pls_fit1/gdiPK's QR.

    The value comes from the QR of ``[sqrt(W) X; E]`` (gdiPK's ldetXWXS); the
    derivative is ``tr(H^-1 dH)`` with ``H^-1`` from the same factor, so the
    existing custom JVP of the inner fit (which supplies ``dH`` through
    ``dX'WX`` and ``dS``) is unchanged.  ``H`` must equal ``R2'R2``.
    """
    del H
    r2 = _qr_r2(weighted_X, penalty_root)
    return 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(r2))))


@qr_log_det.defjvp
def _qr_log_det_jvp(primals, tangents):
    H, weighted_X, penalty_root = primals
    dH = tangents[0]
    value = qr_log_det(H, weighted_X, penalty_root)
    H_inv = qr_hessian_inverse_dense(H, weighted_X, penalty_root)
    return value, jnp.sum(H_inv * dH)
