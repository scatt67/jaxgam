"""JAX utilities: array dispatch and linear algebra primitives.

Provides:
- ``array_module`` / ``is_jax_array``: runtime NumPy/JAX dispatch for
  backend-agnostic code (links, families).
- ``cho_factor``: Cholesky with scale-relative jitter (design.md §4.8).
- ``penalized_cholesky`` / ``penalized_solve``: penalized Hessian solves
  for PIRLS.
- ``numerical_rank``: rank estimation via pivoted QR.

All linalg functions are JIT-compiled, float64, and pure. Other JAX linalg
functions (cho_solve, slogdet, solve_triangular) should be imported directly
from ``jax.scipy.linalg`` / ``jax.numpy.linalg``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Array dispatch
# ---------------------------------------------------------------------------


def is_jax_array(x: object) -> bool:
    """Check if x is a JAX array."""
    return type(x).__module__.startswith(("jax", "jaxlib"))


def array_module(x: object):
    """Return ``jax.numpy`` if *x* is a JAX array, else ``numpy``."""
    if is_jax_array(x):
        return jnp
    return np


# ---------------------------------------------------------------------------
# Linear algebra primitives
# ---------------------------------------------------------------------------


@jax.jit
def cho_factor(
    H: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Cholesky factorization with two-level scale-relative jitter.

    Follows the stabilization strategy from design.md §4.8 (lines 3190–3212):
    always applies a small jitter ``eps_small * trace(H)/p``, and falls back
    to a larger jitter ``eps_large * trace(H)/p`` if the first attempt
    produces NaN (indicating non-positive-definiteness under JIT).

    Uses ``jax.lax.cond`` for the fallback, which is gradient-safe
    (only differentiates the executed branch).

    Parameters
    ----------
    H : jax.Array, shape (p, p)
        Symmetric positive-definite (or nearly PD) matrix.

    Returns
    -------
    L : jax.Array, shape (p, p)
        Lower-triangular Cholesky factor. Pass ``(L, True)`` to
        ``jax.scipy.linalg.cho_solve``.
    jitter_applied : jax.Array
        Scalar jitter level that was actually used (for diagnostics).
    """
    p = H.shape[0]
    I_p = jnp.eye(p)
    trace_H = jnp.trace(H)

    eps_small = jnp.maximum(1e-12 * trace_H / p, 1e-10)
    eps_large = jnp.maximum(1e-6 * trace_H / p, 1e-10)

    L_small = jnp.linalg.cholesky(H + eps_small * I_p)
    has_nan = jnp.any(jnp.isnan(L_small))

    def _use_large_jitter(_: None) -> tuple[jax.Array, jax.Array]:
        L_large = jnp.linalg.cholesky(H + eps_large * I_p)
        return L_large, eps_large

    def _use_small_jitter(_: None) -> tuple[jax.Array, jax.Array]:
        return L_small, eps_small

    L, jitter_applied = jax.lax.cond(
        has_nan, _use_large_jitter, _use_small_jitter, None
    )

    return L, jitter_applied


@jax.jit
def penalized_cholesky(
    XtWX: jax.Array, S_lambda: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Cholesky factor of the penalized Hessian ``XtWX + S_lambda``.

    Parameters
    ----------
    XtWX : jax.Array, shape (p, p)
        Weighted cross-product matrix.
    S_lambda : jax.Array, shape (p, p)
        Combined weighted penalty matrix.

    Returns
    -------
    L : jax.Array, shape (p, p)
        Lower-triangular Cholesky factor.
    jitter_applied : jax.Array
        Jitter level used (for diagnostics).
    """
    H = XtWX + S_lambda
    return cho_factor(H)


@jax.jit
def penalized_solve(
    XtWX: jax.Array, S_lambda: jax.Array, rhs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve ``(XtWX + S_lambda) @ beta = rhs`` via penalized Cholesky.

    This is the core PIRLS solve: ``beta = (XtWX + S_lambda)^{-1} rhs``.

    Parameters
    ----------
    XtWX : jax.Array, shape (p, p)
        Weighted cross-product matrix.
    S_lambda : jax.Array, shape (p, p)
        Combined weighted penalty matrix.
    rhs : jax.Array, shape (p,) or (p, k)
        Right-hand side (typically ``X^T W z``).

    Returns
    -------
    beta : jax.Array, shape (p,) or (p, k)
        Solution coefficients.
    L : jax.Array, shape (p, p)
        Lower-triangular Cholesky factor, retained for downstream use
        (e.g. EDF computation, log-determinant).
    jitter_applied : jax.Array
        Jitter level used.
    """
    L, jitter_applied = penalized_cholesky(XtWX, S_lambda)
    beta = jsla.cho_solve((L, True), rhs)
    return beta, L, jitter_applied


@jax.jit(static_argnames=("tol",))
def numerical_rank(A: jax.Array, tol: float | None = None) -> jax.Array:
    """Estimate the numerical rank of a matrix via pivoted QR.

    Parameters
    ----------
    A : jax.Array, shape (m, n)
    tol : float, optional
        Threshold for diagonal elements of R. Defaults to
        ``eps * max(m, n) * max(|diag(R)|)``.

    Returns
    -------
    rank : jax.Array
        Scalar integer estimate of rank.
    """
    _, R, _ = jsla.qr(A, pivoting=True)
    m, n = A.shape
    diag_R = jnp.abs(jnp.diag(R[: min(m, n), : min(m, n)]))
    if tol is None:
        tol = jnp.finfo(A.dtype).eps * max(m, n) * jnp.max(diag_R)
    return jnp.sum(diag_R > tol)
