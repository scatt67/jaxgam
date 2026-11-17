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

# R's ``sum`` (``rsum``) accumulates in C ``long double`` (LDOUBLE) in element
# order and rounds once. NumPy's ``longdouble`` is the same C type on this
# platform: double on macOS arm64 (52 stored mantissa bits), the x87 80-bit
# format on x86_64 Linux (63) and IEEE quad on aarch64 Linux (112).
_LONG_DOUBLE_MANTISSA_BITS = int(np.finfo(np.longdouble).nmant)
_DOUBLE_MANTISSA_BITS = int(np.finfo(np.float64).nmant)
_X87_MANTISSA_BITS = 63


def _two_sum(left: jax.Array, right: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Knuth's TwoSum: ``left + right == total + error`` exactly."""
    total = left + right
    virtual = total - left
    error = (left - (total - virtual)) + (right - virtual)
    return total, error


def _sequential_double_sum(values: jax.Array) -> jax.Array:
    """``long double`` is ``double``: R's element-order double sum."""
    zero = jnp.zeros((), dtype=values.dtype)
    return jax.lax.fori_loop(
        0, values.size, lambda index, total: total + values[index], zero
    )


def _double_double_sum(values: jax.Array) -> jax.Array:
    """Element-order TwoSum accumulator (about 106 bits), rounded once.

    Serves an IEEE quad ``long double`` (113 bits): the two accumulators
    round to the same double unless the exact sum lies within about 1e-32
    relative of a rounding boundary.
    """
    zero = jnp.zeros((), dtype=values.dtype)

    def accumulate(index, carry):
        total, compensation = carry
        total, error = _two_sum(total, values[index])
        return total, compensation + error

    total, compensation = jax.lax.fori_loop(0, values.size, accumulate, (zero, zero))
    return total + compensation


def _x87_add(state: tuple[jax.Array, jax.Array], value: jax.Array):
    """One x87 ``fadd`` at 64-bit precision on an exact double-double state.

    ``state = (high, low)`` holds an x87 extended value exactly, with
    ``|low| <= ulp(high)/2``. The exact sum with ``value`` is formed as
    ``high + low + tail`` and rounded to 64 significant bits, ties to even,
    as the x87 unit rounds every addition. Finite, normal-range inputs only.
    """
    high, low = state
    total, first = _two_sum(high, value)
    middle, second = _two_sum(first, low)
    head, rest = _two_sum(total, middle)
    rest, tail = _two_sum(rest, second)
    head, rest = _two_sum(head, rest)
    # Exact sum: head + rest + tail, |rest| <= ulp(head)/2, |tail| tiny.
    bits = jax.lax.bitcast_convert_type(head, jnp.int64)
    biased = jnp.right_shift(bits, 52) & 0x7FF
    power_of_two = (bits & ((1 << 52) - 1)) == 0
    below = rest + tail
    lower = power_of_two & (below != 0.0) & ((below < 0.0) != (head < 0.0))
    exponent = biased - 1023 - lower.astype(jnp.int64)
    quantum = jax.lax.bitcast_convert_type(
        jnp.left_shift(exponent - _X87_MANTISSA_BITS + 1023, 52), jnp.float64
    )
    scaled = rest / quantum
    small = tail / quantum
    nearest = jnp.round(scaled)
    fraction = scaled - nearest
    upper = fraction - 0.5
    lower_edge = fraction + 0.5
    odd = jnp.fmod(nearest, 2.0) != 0.0
    up = (upper > -small) | ((upper == -small) & odd)
    down = (lower_edge < -small) | ((lower_edge == -small) & odd)
    count = nearest + up.astype(head.dtype) - down.astype(head.dtype)
    rounded = _two_sum(head, count * quantum)
    exact_zero = head == 0.0
    return (
        jnp.where(exact_zero, tail, rounded[0]),
        jnp.where(exact_zero, 0.0, rounded[1]),
    )


def _x87_sum(values: jax.Array) -> jax.Array:
    """R's ``sum`` on x86_64: an x87 80-bit accumulator, rounded once."""
    zero = jnp.zeros((), dtype=values.dtype)
    high, low = jax.lax.fori_loop(
        0,
        values.size,
        lambda index, state: _x87_add(state, values[index]),
        (zero, zero),
    )
    return high + low


def source_sum(values: jax.Array) -> jax.Array:
    """Port R's ``sum`` of a double vector: element order, LDOUBLE, one rounding.

    The accumulator follows this platform's C ``long double``, the type R's
    LDOUBLE is: a sequential double sum where it is ``double``, an exact x87
    64-bit-precision emulation where it is the 80-bit format, and a TwoSum
    accumulator beyond quad precision otherwise. ``Sl.fitChol``'s
    ``rss.bSb = yy - sum(beta*f)`` cancels its two terms to the
    scale-gradient size, so XLA's own reduction order is not source
    arithmetic there.
    """
    values = jnp.ravel(values)
    if not values.size:
        return jnp.zeros((), dtype=values.dtype)
    if _LONG_DOUBLE_MANTISSA_BITS == _DOUBLE_MANTISSA_BITS:
        return _sequential_double_sum(values)
    if _LONG_DOUBLE_MANTISSA_BITS == _X87_MANTISSA_BITS:
        return _x87_sum(values)
    return _double_double_sum(values)


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
    # ``penalties`` stacked by block shape; kernels loop over these stacks.
    # Built from ``penalties`` when not supplied.
    stacked: penalty_ops.StackedPenaltyStructure | None = None

    def __post_init__(self):
        if self.stacked is None:
            object.__setattr__(
                self, "stacked", penalty_ops.stack_penalty_structure(self.penalties)
            )

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
        penalties = penalty_ops.JaxPenaltyStructure(structure.n_coef, tuple(blocks))
        return cls(
            penalties,
            n_obs,
            null_dimension,
            tuple(singleton_indices),
            tuple(singleton_ranks),
            jax.device_put(np.zeros(len(singleton_indices)), device),
            tuple(multi_indices),
            tuple(multi_ranks),
            tuple(multi_projected),
            tuple(multi_roots),
            jax.device_put(penalty_ops.stack_penalty_structure(penalties), device),
        )

    def tree_flatten(self):
        return (
            self.penalties,
            self.singleton_eig_constants,
            self.multi_block_proj_S,
            self.multi_block_roots,
            self.stacked,
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
        penalties, singleton_constants, multi_penalties, roots, stacked = children
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
            stacked,
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


def _transform_block_rows(matrix, penalties):
    """Replace each block's row slab by ``C.T @ rows``; blocks are disjoint."""
    width = matrix.shape[1]
    for group in penalties.groups:

        def body(result, item, size=group.size):
            C, start = item
            rows = jax.lax.dynamic_slice(result, (start, 0), (size, width))
            return jax.lax.dynamic_update_slice(result, C.T @ rows, (start, 0)), None

        matrix, _ = jax.lax.scan(
            body, matrix, (group.transforms, group.member_starts())
        )
    return matrix


def _transform_block_columns(matrix, penalties):
    """Replace each block's column slab by ``columns @ C``; blocks are disjoint."""
    height = matrix.shape[0]
    for group in penalties.groups:

        def body(result, item, size=group.size):
            C, start = item
            columns = jax.lax.dynamic_slice(result, (0, start), (height, size))
            return (
                jax.lax.dynamic_update_slice(result, columns @ C, (0, start)),
                None,
            )

        matrix, _ = jax.lax.scan(
            body, matrix, (group.transforms, group.member_starts())
        )
    return matrix


def _initial_working_gram(gram, penalties):
    """Sl.initial.repara's initial row/column block transformation of X'WX.

    The source transforms block by block, rows then columns, so the block
    pair (I, J) is formed as ``(C_I' G_IJ) C_J`` when I comes first (or
    I = J) and as ``C_I' (G_IJ C_J)`` when J comes first. Both orders are
    formed here with one stacked pass per group and the source's order is
    selected per pair; each slab product is the source's own product.
    """
    penalties = penalty_ops.as_stacked(penalties)
    rows_first = _transform_block_columns(
        _transform_block_rows(gram, penalties), penalties
    )
    columns_first = _transform_block_rows(
        _transform_block_columns(gram, penalties), penalties
    )
    owner = jnp.asarray(penalties.block_owner())
    later_row = (owner[:, None] > owner[None, :]) & (owner[None, :] >= 0)
    return jnp.where(later_row, columns_first, rows_first)


def _initial_working_rhs(rhs, penalties):
    """Sl.initial.repara's initial block transformation of X'Wz.

    Blocks are disjoint and each reads only its own slab, so one traced
    loop per block shape replaces the per-block loop.
    """
    penalties = penalty_ops.as_stacked(penalties)
    adjusted = rhs
    for group in penalties.groups:

        def body(adjusted, item, size=group.size):
            C, start = item
            local = jax.lax.dynamic_slice(rhs, (start,), (size,))
            return jax.lax.dynamic_update_slice(adjusted, C.T @ local, (start,)), None

        adjusted, _ = jax.lax.scan(
            body, adjusted, (group.transforms, group.member_starts())
        )
    return adjusted


def _original_working_coefficients(beta, penalties):
    """Map working coefficients back to public coordinates, block by block.

    This linear loop stays unrolled over per-block transforms: XLA rounds a
    small matrix-vector product differently inside a traced loop body, and
    the back-transformed coefficients feed bgam.fitd's halving decisions,
    whose reviewed tie-replay counts were measured on this arithmetic.
    """
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
    Its loops are linear in the smooth count and stay unrolled, for the
    rounding reason given in :func:`_original_working_coefficients`.
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
    penalties: penalty_ops.JaxPenaltyStructure | penalty_ops.StackedPenaltyStructure,
    rho: jax.Array,
    smoothing: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """d.detXXS trace contractions on sequential local coefficient blocks.

    Each SPP owns only local rows of the inverse. Pair products select the
    other term's columns; there is no global square array per sp pair.
    Terms are stacked by block shape: one ``lax.scan`` per shape group over
    left terms nests one per group over right terms, so the graph does not
    grow with the number of smooths. Every pair is formed by the same
    operations as before and writes its own Hessian entry once.
    """
    penalties = penalty_ops.as_stacked(penalties)
    width = inverse.shape[1]
    first = jnp.zeros_like(rho)
    second = jnp.zeros((rho.size, rho.size), dtype=inverse.dtype)
    groups = [group for group in penalties.groups if group.count]
    for left in groups:

        def left_term(carry, item, left_size=left.size):
            first, second = carry
            left_component, left_start, left_sp = item
            left_rows = jax.lax.dynamic_slice(
                inverse, (left_start, 0), (left_size, width)
            )
            left_S = penalty_ops._smoothing(rho, smoothing, left_sp) * left_component
            left_product = left_S @ left_rows
            first = first.at[left_sp].add(
                jnp.trace(
                    jax.lax.dynamic_slice(
                        left_product, (0, left_start), (left_size, left_size)
                    )
                )
            )
            for right in groups:

                def right_term(second, item, right_size=right.size):
                    right_component, right_start, right_sp = item
                    right_rows = jax.lax.dynamic_slice(
                        inverse, (right_start, left_start), (right_size, left_size)
                    )
                    right_S = (
                        penalty_ops._smoothing(rho, smoothing, right_sp)
                        * right_component
                    )
                    right_product = right_S @ right_rows
                    value = -jnp.sum(
                        jax.lax.dynamic_slice(
                            left_product, (0, right_start), (left_size, right_size)
                        ).T
                        * right_product
                    )
                    return second.at[left_sp, right_sp].add(value), None

                second, _ = jax.lax.scan(right_term, second, right.terms())
            return (first, second), None

        (first, second), _ = jax.lax.scan(left_term, (first, second), left.terms())
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
    very large. Blocks are disjoint; one traced loop runs per block shape.
    """
    penalties = penalty_ops.as_stacked(penalties)
    result = gram
    for group in penalties.groups:

        def block(result, item, size=group.size, count=group.count):
            components, start, sp_indices = item
            local = jax.lax.dynamic_slice(result, (start, start), (size, size))
            for component in range(count):
                product = _source_rounded(
                    penalty_ops._smoothing(rho, smoothing, sp_indices[component])
                    * components[component]
                )
                local = _source_rounded(local + product)
            return jax.lax.dynamic_update_slice(result, local, (start, start)), None

        result, _ = jax.lax.scan(
            block,
            result,
            (group.penalties, group.member_starts(), group.member_sp_indices()),
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
        metadata.stacked,
        _initial_working_gram(gram, metadata.stacked),
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
        inverse, metadata.stacked, rho, smoothing
    )
    S_value, S_first, S_second = _source_penalty_determinant(
        rho, metadata, factors.penalties, smoothing
    )
    # Sl.fitChol: rss.bSb <- yy - sum(beta*f).
    penalized = yy - source_sum(beta * rhs)
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


@partial(jax.jit, static_argnames=("scale_is_fixed", "fixed_smoothing", "debug"))
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
    fixed_smoothing: bool = False,
    log_scale: jax.Array = 0.0,
    tolerance: jax.Array = 0.0,
    debug: bool = False,
) -> GaussianWorkingModelProposal:
    """Evaluate one row-free ``Sl.fitChol`` trial at log smoothing/scale.

    The proposal is the same for every family's working model: an additive
    Gaussian model passes its frozen ``X'X``/``X'y``, and a regular family
    passes ``bgam.fitd``'s current Fisher ``X'WX``/``X'Wz``/``sum(w z^2)``.
    Unknown-scale parameters append log(phi) to log(sp); fixed-scale mode
    takes log(sp) and the separate log_scale argument. ``tolerance`` owns the
    source active-coordinate filter, and must be finite and nonnegative.
    The returned step caps its maximum absolute coordinate at four. This
    function does not accept the proposal, perform a line search, or decide
    convergence. With no free parameters its gradient/Hessian/step are empty.
    ``fixed_smoothing`` ports ``Sl.fitChol``'s zero-column ``L`` (every
    smoothing parameter supplied): log(sp) still enters through
    ``parameters``, but the gradient, Hessian and step keep only the trailing
    log(phi) coordinate, or are empty when the scale is fixed too.
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
    gram = _initial_working_gram(gram, metadata.stacked)
    rhs = _initial_working_rhs(rhs, metadata.stacked)

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
    if fixed_smoothing:
        # t(L) %*% reml1 and t(L) %*% reml2 %*% L with every column of L fixed.
        free = slice(metadata.penalties.n_penalties, None)
        gradient = gradient[free]
        hessian = hessian[free, free]
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
