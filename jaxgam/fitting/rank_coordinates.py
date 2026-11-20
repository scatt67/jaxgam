"""JAX kernels for reduced Gaussian QR rank-coordinate solves.

CPU global-coordinate construction is in :mod:`jaxgam.formula.rank_coordinates`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla


def reduced_augmented_qr(
    data_factor: jax.Array,
    data_target: jax.Array,
    penalty_roots: tuple[jax.Array, ...],
    log_lambda: jax.Array,
    root_sp_indices: tuple[int, ...],
    coordinate_basis: jax.Array,
    public_basis: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve and reconstruct one frozen rank-aware augmented QR system.

    ``coordinate_basis`` must be the exact map used when determining the
    retained columns.  ``public_basis`` must be the same map composed with the
    public-coefficient transform.  The returned determinant is the selected
    coordinate minor used by mgcv, rather than a pseudo-determinant.
    """
    if len(penalty_roots) != len(root_sp_indices):
        raise ValueError("penalty roots and smoothing indices must agree")
    roots = tuple(
        jnp.exp(log_lambda[index] / 2.0) * root @ coordinate_basis
        for root, index in zip(penalty_roots, root_sp_indices, strict=True)
    )
    reduced_data = data_factor @ coordinate_basis
    augmented = jnp.vstack((reduced_data, *roots))
    rhs = jnp.concatenate(
        (
            data_target,
            jnp.zeros((sum(root.shape[0] for root in roots),), dtype=data_target.dtype),
        )
    )
    q, r = jnp.linalg.qr(augmented, mode="reduced")
    reduced_coefficients = jsla.solve_triangular(r, q.T @ rhs, lower=False)
    inverse_r = jsla.solve_triangular(
        r, jnp.eye(r.shape[0], dtype=r.dtype), lower=False
    )
    reduced_covariance = inverse_r @ inverse_r.T
    return (
        public_basis @ reduced_coefficients,
        public_basis @ reduced_covariance @ public_basis.T,
        r,
        2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(r)))),
    )


jit_reduced_augmented_qr = jax.jit(
    reduced_augmented_qr, static_argnames="root_sp_indices"
)
