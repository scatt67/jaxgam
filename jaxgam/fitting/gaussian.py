"""Frozen Gaussian QR REML kernel.

This module is deliberately a Phase-2-only building block.  Its caller
prepares a weighted-data QR compression and the reviewed rank coordinates on
the host.  A score evaluation consumes only those fixed-size arrays; it never
receives a row source or constructs a penalty eigensystem.

R references: ``bam.r::qr_update`` / ``bam.fit`` and
``fast-REML.r::Sl.fit``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla


@dataclass(frozen=True)
class FrozenGaussianQR:
    """Sufficient weighted-data QR information for one frozen Gaussian fit.

    ``R`` and ``qtz`` are the compact ``R`` and ``Q.T @ z`` from
    the weighted, offset-adjusted data system.  ``residual_tail`` is retained
    from that QR reduction rather than reconstructed from normal equations.
    The row-count fields are provenance/likelihood metadata: the Gaussian
    saturated likelihood uses positive-weight rows, while ``n_obs`` retains
    the original count for a later scale adapter.
    """

    R: jax.Array
    qtz: jax.Array
    residual_tail: jax.Array
    n_obs: jax.Array
    n_positive_weights: jax.Array
    sum_log_positive_weights: jax.Array


_FROZEN_QR_FIELDS = tuple(field.name for field in fields(FrozenGaussianQR))
jax.tree_util.register_pytree_node(
    FrozenGaussianQR,
    lambda value: ([getattr(value, name) for name in _FROZEN_QR_FIELDS], None),
    lambda _, children: FrozenGaussianQR(
        **dict(zip(_FROZEN_QR_FIELDS, children, strict=True))
    ),
)


@dataclass(frozen=True)
class GaussianQRScore:
    """One internally consistent fixed-rho Gaussian QR evaluation."""

    score: jax.Array
    coefficients: jax.Array
    reduced_coefficients: jax.Array
    augmented_factor: jax.Array
    penalty_factor: jax.Array
    residual_sum_squares: jax.Array
    penalty_quadratic: jax.Array
    penalized_residual_sum_squares: jax.Array
    augmented_residual_sum_squares: jax.Array
    log_det_hessian: jax.Array
    log_det_penalty: jax.Array
    scale: jax.Array
    n_obs: jax.Array
    n_positive_weights: jax.Array


_SCORE_FIELDS = tuple(field.name for field in fields(GaussianQRScore))
jax.tree_util.register_pytree_node(
    GaussianQRScore,
    lambda value: ([getattr(value, name) for name in _SCORE_FIELDS], None),
    lambda _, children: GaussianQRScore(
        **dict(zip(_SCORE_FIELDS, children, strict=True))
    ),
)


def _validate_layout(
    compression: FrozenGaussianQR,
    penalty_roots: tuple[jax.Array, ...],
    coordinate_basis: jax.Array,
    public_basis: jax.Array,
    penalty_range_basis: jax.Array,
    root_sp_indices: tuple[int, ...],
    n_smoothing_parameters: int,
    penalty_nullity: int,
) -> None:
    """Reject incompatible frozen layouts before tracing their QR kernels."""
    if len(penalty_roots) != len(root_sp_indices):
        raise ValueError("penalty roots and smoothing indices must agree")
    if (
        not isinstance(n_smoothing_parameters, int)
        or isinstance(n_smoothing_parameters, bool)
        or n_smoothing_parameters < 0
    ):
        raise ValueError("declared smoothing-parameter count must be non-negative")
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= n_smoothing_parameters
        for index in root_sp_indices
    ):
        raise ValueError("penalty smoothing indices are outside the declared range")
    if penalty_nullity < 0:
        raise ValueError("penalty nullity must be non-negative")
    if compression.R.ndim != 2 or compression.qtz.ndim != 1:
        raise ValueError(
            "frozen Gaussian QR factor and target must be matrices/vectors"
        )
    if compression.R.shape[0] != compression.qtz.shape[0]:
        raise ValueError("frozen Gaussian QR factor and target have incompatible rows")
    if coordinate_basis.ndim != 2 or public_basis.ndim != 2:
        raise ValueError("frozen coefficient maps must be matrices")
    n_coef = compression.R.shape[1]
    if coordinate_basis.shape[0] != n_coef:
        raise ValueError("coordinate basis does not match frozen QR coefficients")
    if public_basis.shape[1] != coordinate_basis.shape[1]:
        raise ValueError("public basis does not match reduced coordinates")
    if (
        compression.R.shape[0] + sum(root.shape[0] for root in penalty_roots)
        < coordinate_basis.shape[1]
    ):
        raise ValueError(
            "frozen QR and penalty rows cannot identify reduced coordinates"
        )
    if penalty_range_basis.ndim != 2 or penalty_range_basis.shape[0] != n_coef:
        raise ValueError("penalty range basis does not match frozen QR coefficients")
    penalty_rank = penalty_range_basis.shape[1]
    if penalty_roots and penalty_rank == 0:
        raise ValueError("non-empty penalty roots require a penalty range")
    if not penalty_roots and (n_smoothing_parameters or penalty_rank):
        raise ValueError("zero-penalty QR requires no smoothing parameters or range")
    if penalty_roots and sum(root.shape[0] for root in penalty_roots) < penalty_rank:
        raise ValueError("penalty roots cannot span the supplied penalty range")
    for root in penalty_roots:
        if root.ndim != 2 or root.shape[1] != n_coef:
            raise ValueError("compact penalty root has incompatible coefficients")


def _scaled_roots(
    log_lambda: jax.Array,
    penalty_roots: tuple[jax.Array, ...],
    root_sp_indices: tuple[int, ...],
) -> tuple[jax.Array, ...]:
    """Apply one smoothing parameter per compact root without padding."""
    return tuple(
        jnp.exp(log_lambda[index] / 2.0) * root
        for root, index in zip(penalty_roots, root_sp_indices, strict=True)
    )


def gaussian_qr_reml_score(
    params: jax.Array,
    compression: FrozenGaussianQR,
    penalty_roots: tuple[jax.Array, ...],
    coordinate_basis: jax.Array,
    public_basis: jax.Array,
    penalty_range_basis: jax.Array,
    fixed_scale: jax.Array,
    *,
    root_sp_indices: tuple[int, ...],
    n_smoothing_parameters: int,
    penalty_nullity: int,
    scale_known: bool,
) -> GaussianQRScore:
    """Evaluate frozen-coordinate Gaussian REML at ``params``.

    The first ``n_smoothing_parameters`` entries are log smoothing
    parameters.  Several compact roots may share one index.  Each supplied
    root must have exactly its declared, non-null penalty rows: callers must
    remove numerical null-space rows during frozen-layout preparation rather
    than passing a square factor with near-zero singular directions.  When
    ``scale_known`` is false, the final entry is ``log(phi)``.  The
    penalty range basis must be an orthonormal, caller-reviewed basis for the
    combined penalty range in the declared fitting coordinates.  This permits
    a stacked-root QR determinant with every range direction retained, even
    for overlapping penalties.

    ``coordinate_basis`` is independent from that range basis: it selects the
    frozen identifiable coefficient coordinates, and ``public_basis`` maps
    those coefficients to the caller's public coordinates.  The function does
    not infer or validate any aliasing/rank mapping.
    """
    _validate_layout(
        compression,
        penalty_roots,
        coordinate_basis,
        public_basis,
        penalty_range_basis,
        root_sp_indices,
        n_smoothing_parameters,
        penalty_nullity,
    )
    n_lambda = n_smoothing_parameters
    expected_params = n_lambda if scale_known else n_lambda + 1
    if params.ndim != 1 or params.shape[0] != expected_params:
        raise ValueError("Gaussian QR REML parameters have incompatible length")

    log_lambda = params[:n_lambda]
    scale = fixed_scale if scale_known else jnp.exp(params[n_lambda])
    roots = _scaled_roots(log_lambda, penalty_roots, root_sp_indices)

    # QR(E_range) computes log|S_lambda|_+ without forming a Gram matrix.
    # The range basis is frozen by the caller, so all its columns are active.
    if roots:
        penalty_range_rows = jnp.vstack(
            tuple(root @ penalty_range_basis for root in roots)
        )
        _, penalty_factor = jnp.linalg.qr(penalty_range_rows, mode="reduced")
        log_det_penalty = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(penalty_factor))))
    else:
        penalty_factor = jnp.empty((0, 0), dtype=compression.R.dtype)
        log_det_penalty = jnp.array(0.0, dtype=compression.R.dtype)

    data_rows = compression.R @ coordinate_basis
    penalty_rows = tuple(root @ coordinate_basis for root in roots)
    augmented_rows = jnp.vstack((data_rows, *penalty_rows))
    augmented_target = jnp.concatenate(
        (
            compression.qtz,
            jnp.zeros(
                (sum(root.shape[0] for root in roots),),
                dtype=compression.qtz.dtype,
            ),
        )
    )
    # Reduced QR is differentiable in JAX.  The data tail is preserved in the
    # compression and the residual itself is evaluated below, avoiding the
    # unstable normal-equation subtraction ``z'z - beta'X'z``.
    augmented_q, augmented_r = jnp.linalg.qr(augmented_rows, mode="reduced")
    n_reduced = coordinate_basis.shape[1]
    augmented_factor = augmented_r[:n_reduced, :]
    augmented_qtz = augmented_q.T @ augmented_target
    reduced_coefficients = jsla.solve_triangular(
        augmented_factor, augmented_qtz[:n_reduced], lower=False
    )

    # Keep the data and penalty pieces explicit.  In particular, this avoids
    # treating an augmented residual tail as a data RSS and adding beta'Sbeta
    # a second time.  The equality with the augmented QR tail is an algebraic
    # check on the frozen layout, rather than an alternate likelihood.
    data_residual = compression.qtz - data_rows @ reduced_coefficients
    residual_rss = compression.residual_tail + data_residual @ data_residual
    penalty_quadratic = (
        sum(
            (root @ reduced_coefficients) @ (root @ reduced_coefficients)
            for root in penalty_rows
        )
        if penalty_rows
        else jnp.array(0.0, dtype=compression.R.dtype)
    )
    penalized_rss = residual_rss + penalty_quadratic
    augmented_residual = augmented_target - augmented_rows @ reduced_coefficients
    augmented_rss = compression.residual_tail + augmented_residual @ augmented_residual
    log_det_hessian = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(augmented_factor))))
    ls_sat = (
        -compression.n_positive_weights * jnp.log(2.0 * jnp.pi * scale) / 2.0
        + compression.sum_log_positive_weights / 2.0
    )
    score = (
        penalized_rss / (2.0 * scale)
        - ls_sat
        + log_det_hessian / 2.0
        - log_det_penalty / 2.0
        - penalty_nullity * jnp.log(2.0 * jnp.pi * scale) / 2.0
    )
    return GaussianQRScore(
        score=score,
        coefficients=public_basis @ reduced_coefficients,
        reduced_coefficients=reduced_coefficients,
        augmented_factor=augmented_factor,
        penalty_factor=penalty_factor,
        residual_sum_squares=residual_rss,
        penalty_quadratic=penalty_quadratic,
        penalized_residual_sum_squares=penalized_rss,
        augmented_residual_sum_squares=augmented_rss,
        log_det_hessian=log_det_hessian,
        log_det_penalty=log_det_penalty,
        scale=scale,
        n_obs=compression.n_obs,
        n_positive_weights=compression.n_positive_weights,
    )


def gaussian_qr_reml_value(
    params: jax.Array,
    compression: FrozenGaussianQR,
    penalty_roots: tuple[jax.Array, ...],
    coordinate_basis: jax.Array,
    public_basis: jax.Array,
    penalty_range_basis: jax.Array,
    fixed_scale: jax.Array,
    *,
    root_sp_indices: tuple[int, ...],
    n_smoothing_parameters: int,
    penalty_nullity: int,
    scale_known: bool,
) -> jax.Array:
    """Return only the frozen QR REML scalar for autodiff callers."""
    return gaussian_qr_reml_score(
        params,
        compression,
        penalty_roots,
        coordinate_basis,
        public_basis,
        penalty_range_basis,
        fixed_scale,
        root_sp_indices=root_sp_indices,
        n_smoothing_parameters=n_smoothing_parameters,
        penalty_nullity=penalty_nullity,
        scale_known=scale_known,
    ).score


_GAUSSIAN_QR_STATIC = (
    "root_sp_indices",
    "n_smoothing_parameters",
    "penalty_nullity",
    "scale_known",
)
jit_gaussian_qr_reml_score = jax.jit(
    gaussian_qr_reml_score, static_argnames=_GAUSSIAN_QR_STATIC
)
jit_gaussian_qr_reml_value_and_grad = jax.jit(
    jax.value_and_grad(gaussian_qr_reml_value), static_argnames=_GAUSSIAN_QR_STATIC
)
jit_gaussian_qr_reml_hessian = jax.jit(
    jax.hessian(gaussian_qr_reml_value), static_argnames=_GAUSSIAN_QR_STATIC
)
