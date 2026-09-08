"""Pure JIT pivot-aware triangular actions for CPU QR factors."""

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
