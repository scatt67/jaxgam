"""One Gaussian working-model REML proposal from frozen statistics.

This ports ``Sl.fitChol``'s proposal policy, not its surrounding performance
iteration or the nested exact-objective Newton controller. No training rows
are retained or scanned. The caller owns accepted/trial states and damping.
The initial component requires a positive-definite penalized coefficient
system; rank-dropping and penalty-link matrices remain separate extensions.

Pivoted Cholesky factors are kernel *inputs*: ``working_model_factor_inputs``
assembles the preconditioned matrices that pinned R passes to
``chol(pivot=TRUE)``, the host factors them with the pinned DPSTRF provider,
and ``gaussian_working_model_proposal`` consumes the factors. Factors have no
derivatives; every derivative is an analytic trace or implicit derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from jaxgam.fitting import penalty_ops
from jaxgam.formula.working_penalties import (
    build_working_penalty_blocks,
    source_root_product,
)
from jaxgam.penalties.structure import PenaltyStructure

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WorkingModelCriterionMetadata:
    """Row-free structural penalty metadata and Gaussian normalization."""

    penalties: penalty_ops.JaxPenaltyStructure
    n_obs: int
    null_dimension: int
    singleton_sp_indices: tuple[int, ...]
    singleton_ranks: tuple[int, ...]
    singleton_eig_constants: jax.Array
    multi_block_sp_indices: tuple[tuple[int, ...], ...]
    multi_block_ranks: tuple[int, ...]
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...]
    multi_block_roots: tuple[tuple[jax.Array, ...], ...]

    @classmethod
    def from_structure(
        cls,
        structure: PenaltyStructure,
        n_obs: int,
        device: jax.Device | None = None,
        *,
        determinant_structure: PenaltyStructure | None = None,
    ) -> WorkingModelCriterionMetadata:
        """Transfer shared setup-owned range metadata at the host boundary."""
        if isinstance(n_obs, bool) or not isinstance(n_obs, int) or n_obs < 1:
            raise ValueError("n_obs must be a positive integer.")
        source_blocks = build_working_penalty_blocks(structure)
        determinant_blocks = (
            source_blocks
            if determinant_structure is None
            else build_working_penalty_blocks(
                determinant_structure, source_eigensolver=True
            )
        )
        if len(source_blocks) != len(determinant_blocks) or any(
            a.sp_indices != b.sp_indices or a.rank != b.rank
            for a, b in zip(source_blocks, determinant_blocks, strict=True)
        ):
            raise ValueError("Determinant penalty blocks differ in rank or order.")
        null_dimension = structure.n_coef - sum(block.rank for block in source_blocks)
        if n_obs <= null_dimension:
            raise ValueError("n_obs must exceed the penalty null dimension.")
        blocks = []
        singleton_indices = []
        singleton_ranks = []
        multi_indices = []
        multi_ranks = []
        multi_projected = []
        multi_roots = []
        for block, determinant_block in zip(
            source_blocks, determinant_blocks, strict=True
        ):
            size = block.stop - block.start
            blocks.append(
                penalty_ops.JaxPenaltyBlock(
                    block.start,
                    block.stop,
                    block.sp_indices,
                    tuple(
                        penalty_ops.JaxLocalPenalty(
                            "dense", jax.device_put(S, device), size
                        )
                        for S in block.penalties
                    ),
                    penalty_ops.JaxTransform(
                        "dense", jax.device_put(block.transform, device), size
                    ),
                )
            )
            if len(block.penalties) == 1:
                singleton_indices.append(block.sp_indices[0])
                singleton_ranks.append(block.rank)
            else:
                multi_indices.append(block.sp_indices)
                multi_ranks.append(block.rank)
                multi_roots.append(
                    tuple(
                        jax.device_put(root, device) for root in determinant_block.roots
                    )
                )
                multi_projected.append(
                    tuple(
                        jax.device_put(source_root_product(root), device)
                        for root in determinant_block.roots
                    )
                )
        return cls(
            penalty_ops.JaxPenaltyStructure(structure.n_coef, tuple(blocks)),
            n_obs,
            null_dimension,
            tuple(singleton_indices),
            tuple(singleton_ranks),
            jax.device_put(np.zeros(len(singleton_indices)), device),
            tuple(multi_indices),
            tuple(multi_ranks),
            tuple(multi_projected),
            tuple(multi_roots),
        )

    def tree_flatten(self):
        return (
            self.penalties,
            self.singleton_eig_constants,
            self.multi_block_proj_S,
            self.multi_block_roots,
        ), (
            self.n_obs,
            self.null_dimension,
            self.singleton_sp_indices,
            self.singleton_ranks,
            self.multi_block_sp_indices,
            self.multi_block_ranks,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            n_obs,
            null_dimension,
            singleton_indices,
            singleton_ranks,
            multi_indices,
            multi_ranks,
        ) = aux
        penalties, singleton_constants, multi_penalties, roots = children
        return cls(
            penalties,
            n_obs,
            null_dimension,
            singleton_indices,
            singleton_ranks,
            singleton_constants,
            multi_indices,
            multi_ranks,
            multi_penalties,
            roots,
        )


class WorkingModelFactor(NamedTuple):
    """One R-style pivoted factor of a diagonally preconditioned matrix.

    ``upper``/``pivots``/``rank`` are ``chol(pivot=TRUE)``'s upper storage,
    zero-based pivots and rank for ``t(A/d)/d``; ``diagonal`` is ``d``.
    """

    upper: jax.Array
    pivots: jax.Array
    rank: jax.Array
    diagonal: jax.Array


class WorkingModelFactorInputs(NamedTuple):
    """Preconditioned matrices R factors in Sl.fitChol and ldetSblock."""

    hessian: jax.Array
    hessian_diagonal: jax.Array
    penalties: tuple[jax.Array, ...]
    penalty_diagonals: tuple[jax.Array, ...]


class WorkingModelFactors(NamedTuple):
    """Host factors of the penalized Hessian and each coupled penalty block."""

    hessian: WorkingModelFactor
    penalties: tuple[WorkingModelFactor, ...]


class GaussianWorkingModelProposal(NamedTuple):
    """Trial coefficients and one source-policy smoothing/scale proposal."""

    coefficients: jax.Array
    criterion: jax.Array
    gradient: jax.Array
    hessian: jax.Array
    step: jax.Array
    factor_upper: jax.Array
    factor_pivots: jax.Array
    factor_rank: jax.Array
    factor_diagonal: jax.Array
    logdet_hessian: jax.Array
    logdet_penalty: jax.Array


def _sl_newton_step(
    gradient: jax.Array,
    hessian: jax.Array,
    tolerance: jax.Array,
) -> jax.Array:
    """Port ``Sl.fitChol`` active-coordinate and absolute-eigenvalue policy."""
    if gradient.shape[0] == 0:
        return jnp.zeros_like(gradient)
    active = (jnp.abs(gradient) > tolerance) | (jnp.abs(jnp.diag(hessian)) > tolerance)

    def propose(_):
        # Static shapes preserve the active subproblem by adding zero inactive
        # directions; those directions have zero gradient and zero final step.
        restricted = jnp.where(active[:, None] & active[None, :], hessian, 0.0)
        values, vectors = jnp.linalg.eigh(restricted, UPLO="L", symmetrize_input=False)
        values = jnp.abs(values)
        floor = jnp.max(values) * jnp.sqrt(jnp.finfo(hessian.dtype).eps)
        values = jnp.maximum(values, floor)
        masked_gradient = jnp.where(active, gradient, 0.0)
        step = -(vectors @ ((vectors.T @ masked_gradient) / values))
        step = jnp.where(active, step, 0.0)
        maximum = jnp.max(jnp.abs(step))
        return step * jnp.where(maximum > 4.0, 4.0 / maximum, 1.0)

    return jax.lax.cond(
        jnp.any(active), propose, lambda _: jnp.zeros_like(gradient), None
    )


def _initial_working_gram(gram, penalties):
    """Sl.initial.repara's initial row/column block transformation of X'WX."""
    transformed = gram
    for block in penalties.blocks:
        C = block.transform.values
        transformed = transformed.at[block.start : block.stop, :].set(
            C.T @ transformed[block.start : block.stop, :]
        )
        transformed = transformed.at[:, block.start : block.stop].set(
            transformed[:, block.start : block.stop] @ C
        )
    return transformed


def _initial_working_rhs(rhs, penalties):
    """Sl.initial.repara's initial block transformation of X'Wz."""
    adjusted = rhs
    for block in penalties.blocks:
        adjusted = adjusted.at[block.start : block.stop].set(
            block.transform.values.T @ adjusted[block.start : block.stop]
        )
    return adjusted


def _original_working_coefficients(beta, penalties):
    result = beta
    for block in penalties.blocks:
        result = result.at[block.start : block.stop].set(
            block.transform.values @ beta[block.start : block.stop]
        )
    return result


def _source_cholesky_solve(
    upper: jax.Array,
    pivots: jax.Array,
    diagonal: jax.Array,
    rhs: jax.Array,
    rank: jax.Array | None = None,
) -> jax.Array:
    """Sl.fitChol's scaled pivoted solve, with static-size rank padding."""
    if rank is None:
        rank = len(upper)
    active = jnp.arange(len(upper)) < rank
    upper = jnp.where(
        active[:, None] & active[None, :],
        upper,
        jnp.diag((~active).astype(upper.dtype)),
    )
    scale = diagonal.reshape((-1,) + (1,) * (rhs.ndim - 1))
    active_rows = active.reshape((-1,) + (1,) * (rhs.ndim - 1))
    pivoted_rhs = jnp.where(active_rows, (rhs / scale)[pivots], 0.0)
    first = jsl.solve_triangular(upper.T, pivoted_rhs, lower=True)
    solved = jsl.solve_triangular(upper, first, lower=False)
    return jnp.zeros_like(rhs).at[pivots].set(solved / scale[pivots])


def _source_ift_chol(
    gram: jax.Array,
    beta: jax.Array,
    penalties: penalty_ops.JaxPenaltyStructure,
    rho: jax.Array,
    upper: jax.Array,
    pivots: jax.Array,
    diagonal: jax.Array,
    rank: jax.Array | None = None,
    smoothing: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sl.iftChol's linear-penalty RSS plus quadratic derivatives.

    Only D/db/S.db/G.db have a p by smoothing-parameter axis; local
    coefficient penalties never acquire a smoothing-parameter stack of
    global square matrices. The expanded source contractions deliberately
    preserve its operation order rather than differentiating yy-beta'f.
    """
    count = penalties.n_penalties
    D = jnp.zeros((penalties.n_coef, count), dtype=beta.dtype)
    for block in penalties.blocks:
        local_beta = beta[block.start : block.stop]
        for sp, component in zip(block.sp_indices, block.penalties, strict=True):
            column = penalty_ops._smoothing(
                rho, smoothing, sp
            ) * penalty_ops._apply_local(component, local_beta)
            D = D.at[block.start : block.stop, sp].add(column)
    db = -_source_cholesky_solve(upper, pivots, diagonal, D, rank)
    first = jnp.sum(beta[:, None] * D, axis=0)
    Sdb = jax.vmap(
        lambda vector: penalty_ops.apply(penalties, vector, rho, smoothing=smoothing),
        in_axes=1,
        out_axes=1,
    )(db)
    quadratic_second = jnp.diag(first) + 2.0 * (db.T @ (D + Sdb) + D.T @ db)
    residual_second = 2.0 * (db.T @ (gram @ db))
    return first, quadratic_second + residual_second, db


def _source_hessian_determinant_derivatives(
    inverse: jax.Array,
    penalties: penalty_ops.JaxPenaltyStructure,
    rho: jax.Array,
    smoothing: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """d.detXXS trace contractions on sequential local coefficient blocks.

    Each SPP owns only local rows of the inverse. Pair products select the
    other term's columns; there is no global square array per sp pair.
    """
    first = jnp.zeros_like(rho)
    second = jnp.zeros((rho.size, rho.size), dtype=inverse.dtype)
    for left in penalties.blocks:
        left_rows = inverse[left.start : left.stop, :]
        for left_sp, left_component in zip(
            left.sp_indices, left.penalties, strict=True
        ):
            left_S = penalty_ops._smoothing(
                rho, smoothing, left_sp
            ) * penalty_ops._dense_local(left_component)
            left_product = left_S @ left_rows
            first = first.at[left_sp].add(
                jnp.trace(left_product[:, left.start : left.stop])
            )
            for right in penalties.blocks:
                right_rows = inverse[right.start : right.stop, left.start : left.stop]
                for right_sp, right_component in zip(
                    right.sp_indices, right.penalties, strict=True
                ):
                    right_S = penalty_ops._smoothing(
                        rho, smoothing, right_sp
                    ) * penalty_ops._dense_local(right_component)
                    right_product = right_S @ right_rows
                    value = -jnp.sum(
                        left_product[:, right.start : right.stop].T * right_product
                    )
                    second = second.at[left_sp, right_sp].add(value)
    second = second + jnp.diag(first)
    return first, second


def _source_penalty_determinant(rho, metadata, factors, smoothing=None):
    """ldetSblock's root/triangular trace derivatives from supplied factors.

    ``factors`` holds one host ``chol(pivot=TRUE)`` factor per coupled block,
    of the matrix ``working_model_factor_inputs`` assembles from the same
    ``smoothing`` (by default ``exp(rho)``).
    """
    value = jnp.asarray(0.0, dtype=rho.dtype)
    first = jnp.zeros_like(rho)
    second = jnp.zeros((rho.size, rho.size), dtype=rho.dtype)
    for sp, rank in zip(
        metadata.singleton_sp_indices, metadata.singleton_ranks, strict=True
    ):
        value = value + rank * rho[sp]
        first = first.at[sp].add(rank)
    for indices, roots, factor in zip(
        metadata.multi_block_sp_indices,
        metadata.multi_block_roots,
        factors,
        strict=True,
    ):
        lam = (
            jnp.exp(rho[jnp.asarray(indices)])
            if smoothing is None
            else smoothing[jnp.asarray(indices)]
        )
        diagonal = factor.diagonal
        rank = len(diagonal)
        valid = factor.rank == rank
        block_value = 2.0 * jnp.sum(
            jnp.log(jnp.diag(factor.upper)) + jnp.log(diagonal[factor.pivots])
        )
        value = value + jnp.where(valid, block_value, jnp.nan)
        products = []
        for index, root in enumerate(roots):
            solved = jsl.solve_triangular(
                factor.upper.T,
                root[factor.pivots, :] / diagonal[factor.pivots, None],
                lower=True,
            )
            derivative = jnp.sum(solved**2) * lam[index]
            first = first.at[indices[index]].add(derivative)
            products.append(_source_tcrossprod(solved))
            for previous in range(index + 1):
                entry = (
                    -jnp.sum(products[index] * products[previous])
                    * lam[index]
                    * lam[previous]
                )
                second = second.at[indices[index], indices[previous]].add(entry)
                if previous != index:
                    second = second.at[indices[previous], indices[index]].add(entry)
            second = second.at[indices[index], indices[index]].add(derivative)
    return value, first, second


def _source_rounded(value):
    """Keep separate scalar operations at the reference BLAS boundaries."""
    return jnp.nextafter(value, value)


def _source_tcrossprod(matrix):
    """pcrossprod(trans=TRUE)'s ordered reference DSYRK inner products."""

    def add(column, total):
        values = matrix[:, column]
        # The pinned reference DSYRK contracts TEMP += A(l,i)*A(l,j).
        # Keep that fused contraction, then round before the next column.
        return _source_rounded(total + values[:, None] * values[None, :])

    return jax.lax.fori_loop(
        0,
        matrix.shape[1],
        add,
        jnp.zeros((matrix.shape[0], matrix.shape[0]), dtype=matrix.dtype),
    )


def _source_add_penalties(penalties, gram, rho, smoothing=None):
    """Sl.addS: add each rounded ``lambda_j S_j`` to the matrix in turn.

    mgcv_madi updates ``A[ind, ind]`` component by component; summing the
    components first rounds differently once one smoothing parameter is
    very large.
    """
    result = gram
    for block in penalties.blocks:
        for sp, component in zip(block.sp_indices, block.penalties, strict=True):
            local = result[block.start : block.stop, block.start : block.stop]
            product = _source_rounded(
                penalty_ops._smoothing(rho, smoothing, sp)
                * penalty_ops._dense_local(component)
            )
            result = result.at[block.start : block.stop, block.start : block.stop].set(
                _source_rounded(local + product)
            )
    return result


def _source_penalty_factor_input(root_products, lam):
    """ldetSblock's ``S`` and its diagonal preconditioning, before ``chol``."""
    combined = _source_rounded(root_products[0] * lam[0])
    for index in range(1, len(root_products)):
        combined = _source_rounded(
            combined + _source_rounded(root_products[index] * lam[index])
        )
    diagonal = jnp.sqrt(jnp.where(jnp.diag(combined) > 0.0, jnp.diag(combined), 1.0))
    scaled = _source_rounded(
        _source_rounded(combined / diagonal[:, None]).T / diagonal[:, None]
    )
    return scaled, diagonal


@jax.jit
def working_model_factor_inputs(
    gram: jax.Array,
    metadata: WorkingModelCriterionMetadata,
    smoothing: jax.Array,
) -> WorkingModelFactorInputs:
    """Matrices pinned R factors at one trial, before ``chol(pivot=TRUE)``.

    ``gram`` is in public coordinates; ``smoothing`` is the trial's
    ``exp(log sp)``, which the proposal must receive unchanged.
    """
    _check_smoothing(smoothing, metadata)
    if gram.shape != (metadata.penalties.n_coef, metadata.penalties.n_coef):
        raise ValueError("gram must match the coefficient dimension.")
    H = _source_add_penalties(
        metadata.penalties,
        _initial_working_gram(gram, metadata.penalties),
        None,
        smoothing,
    )
    diagonal = jnp.where(jnp.diag(H) > 0.0, jnp.sqrt(jnp.diag(H)), 1.0)
    scaled = _source_rounded(
        _source_rounded(H / diagonal[:, None]).T / diagonal[:, None]
    )
    penalties = [
        _source_penalty_factor_input(products, smoothing[jnp.asarray(indices)])
        for indices, products in zip(
            metadata.multi_block_sp_indices, metadata.multi_block_proj_S, strict=True
        )
    ]
    return WorkingModelFactorInputs(
        scaled,
        diagonal,
        tuple(matrix for matrix, _ in penalties),
        tuple(values for _, values in penalties),
    )


def _check_smoothing(smoothing, metadata):
    if jnp.shape(smoothing) != (metadata.penalties.n_penalties,):
        raise ValueError("smoothing must contain one value per log(sp).")


def _source_gaussian_state(
    parameters,
    gram,
    rhs,
    yy,
    metadata,
    log_scale,
    scale_is_fixed,
    factors,
    smoothing,
):
    rho = parameters if scale_is_fixed else parameters[:-1]
    log_phi = log_scale if scale_is_fixed else parameters[-1]
    phi = jnp.exp(log_phi)
    factor = factors.hessian
    diagonal = factor.diagonal
    beta = _source_cholesky_solve(
        factor.upper, factor.pivots, diagonal, rhs, factor.rank
    )
    inverse = _source_cholesky_solve(
        factor.upper, factor.pivots, diagonal, jnp.eye(len(diagonal)), factor.rank
    )
    used = jnp.arange(len(diagonal)) < factor.rank
    upper_diagonal = jnp.where(used, jnp.diag(factor.upper), 1.0)
    scales = jnp.where(used, diagonal[factor.pivots], 1.0)
    H_value = 2.0 * jnp.sum(jnp.log(upper_diagonal) + jnp.log(scales))
    H_first, H_second = _source_hessian_determinant_derivatives(
        inverse, metadata.penalties, rho, smoothing
    )
    S_value, S_first, S_second = _source_penalty_determinant(
        rho, metadata, factors.penalties, smoothing
    )
    penalized = yy - jnp.vdot(beta, rhs).real
    first, second, _ = _source_ift_chol(
        gram,
        beta,
        metadata.penalties,
        rho,
        factor.upper,
        factor.pivots,
        diagonal,
        factor.rank,
        smoothing,
    )
    gradient = (H_first - S_first + first / phi) / 2.0
    hessian = (H_second - S_second + second / phi) / 2.0
    normalization = (metadata.n_obs - metadata.null_dimension) * jnp.log(
        2.0 * jnp.pi * phi
    )
    value = (penalized / phi - S_value + H_value + normalization) / 2.0
    if not scale_is_fixed:
        cross = -first / (2.0 * phi)
        scale_gradient = (
            -penalized / phi + metadata.n_obs - metadata.null_dimension
        ) / 2.0
        gradient = jnp.r_[gradient, scale_gradient]
        hessian = jnp.block(
            [
                [hessian, cross[:, None]],
                [cross[None, :], (penalized / (2.0 * phi)).reshape(1, 1)],
            ]
        )
    valid = factor.rank >= 0
    return (
        jnp.where(valid, value, jnp.nan),
        jnp.where(valid, beta, jnp.nan),
        jnp.where(valid, gradient, jnp.nan),
        jnp.where(valid, hessian, jnp.nan),
        factor,
        diagonal,
        H_value,
        S_value,
    )


@partial(jax.jit, static_argnames=("scale_is_fixed", "debug"))
def gaussian_working_model_proposal(
    parameters: jax.Array,
    gram: jax.Array,
    rhs: jax.Array,
    response_square_sum: jax.Array,
    metadata: WorkingModelCriterionMetadata,
    factors: WorkingModelFactors,
    *,
    smoothing: jax.Array,
    scale_is_fixed: bool = False,
    log_scale: jax.Array = 0.0,
    tolerance: jax.Array = 0.0,
    debug: bool = False,
) -> GaussianWorkingModelProposal:
    """Evaluate one row-free Gaussian trial at log smoothing/scale parameters.

    Unknown-scale parameters append log(phi) to log(sp); fixed-scale mode
    takes log(sp) and the separate log_scale argument. ``tolerance`` owns the
    source active-coordinate filter, and must be finite and nonnegative.
    The returned step caps its maximum absolute coordinate at four. This
    function does not accept the proposal, perform a line search, or decide
    convergence. With no free parameters its gradient/Hessian/step are empty.
    Response square sums have the same cancellation boundary as pinned
    ``Sl.fitChol``; stable final selected-residual evaluation belongs to the
    surrounding execution controller. ``smoothing`` is the trial's
    ``exp(log sp)`` and ``factors`` the host factors of the matrices
    ``working_model_factor_inputs`` built from that same ``smoothing``.
    """
    _check_smoothing(smoothing, metadata)
    expected_parameters = metadata.penalties.n_penalties + (not scale_is_fixed)
    if parameters.shape != (expected_parameters,):
        raise ValueError("parameters must contain log(sp) and optional log(scale).")
    n_coef = metadata.penalties.n_coef
    if gram.shape != (n_coef, n_coef) or rhs.shape != (n_coef,):
        raise ValueError("gram and rhs must match the coefficient dimension.")
    if jnp.shape(response_square_sum) != ():
        raise ValueError("response_square_sum must be scalar.")
    if len(factors.penalties) != len(metadata.multi_block_sp_indices):
        raise ValueError("factors must hold one factor per coupled penalty block.")
    gram = _initial_working_gram(gram, metadata.penalties)
    rhs = _initial_working_rhs(rhs, metadata.penalties)

    value, beta, gradient, hessian, factor, diagonal, H_value, S_value = (
        _source_gaussian_state(
            parameters,
            gram,
            rhs,
            response_square_sum,
            metadata,
            log_scale,
            scale_is_fixed,
            factors,
            smoothing,
        )
    )
    step = _sl_newton_step(gradient, hessian, tolerance)
    if debug:
        jax.debug.print(
            "Gaussian working model criterion={value}, step={step}",
            value=value,
            step=step,
        )
    return GaussianWorkingModelProposal(
        _original_working_coefficients(beta, metadata.penalties),
        value,
        gradient,
        hessian,
        step,
        factor.upper,
        factor.pivots,
        factor.rank,
        diagonal,
        H_value,
        S_value,
    )
