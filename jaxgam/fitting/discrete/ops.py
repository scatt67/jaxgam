"""JIT-compatible exact indexed design actions.

Family code owns working weights and responses.  These kernels accept those
already-computed vectors and perform only the linear algebra described by
mgcv's ``Xbd``, ``XWyd``, ``XWXd`` and ``diagXVXd`` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.formula.discrete.design import (
    FrozenDiscreteDesign,
    LookupBlock,
    TensorLookupBlock,
)
from jaxgam.formula.kernel_groups import (
    BlockKey,
    LookupKey,
    PairReductionPlan,
    TensorKey,
    discrete_pair_schedule,
    pair_reduction_plan,
    shape_groups,
)

__all__ = [
    "DeviceDiscreteDesign",
    "DeviceLookupBlock",
    "DeviceTensorLookupBlock",
    "PairReductionPlan",
    "discrete_diag_xvxd",
    "discrete_row_xwxd",
    "discrete_row_xwxd_bytes",
    "discrete_row_xwxd_cost",
    "discrete_xb_transpose",
    "discrete_xbd",
    "discrete_xwxd",
    "discrete_xwxd_cost",
    "discrete_xwyd",
    "lookup_pair_histogram_crossproduct",
    "pair_reduction_plan",
    "to_device_discrete_design",
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DeviceLookupBlock:
    """Device table for one ordinary exact lookup block."""

    table: jax.Array
    col_start: int

    def tree_flatten(self):
        return (self.table,), self.col_start

    @classmethod
    def tree_unflatten(cls, col_start, children):
        return cls(children[0], col_start)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DeviceTensorLookupBlock:
    """Device marginal tables plus frozen tensor constraint metadata."""

    tables: tuple[jax.Array, ...]
    centering: jax.Array | None
    keep: jax.Array
    col_start: int
    multiplier_table: jax.Array | None = None
    level_codes: jax.Array | None = None
    n_levels: int = 1

    def tree_flatten(self):
        children = (
            (*self.tables, self.keep)
            if self.centering is None
            else (*self.tables, self.centering, self.keep)
        )
        if self.multiplier_table is not None:
            children = (*children, self.multiplier_table)
        if self.level_codes is not None:
            children = (*children, self.level_codes)
        return children, (
            len(self.tables),
            self.centering is not None,
            self.col_start,
            self.multiplier_table is not None,
            self.level_codes is not None,
            self.n_levels,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_tables, has_centering, col_start, has_multiplier, has_levels, n_levels = aux
        tables = tuple(children[:n_tables])
        if has_centering:
            end = n_tables + 2
            return cls(
                tables,
                children[n_tables],
                children[n_tables + 1],
                col_start,
                children[end] if has_multiplier else None,
                children[end + has_multiplier] if has_levels else None,
                n_levels,
            )
        end = n_tables + 1
        return cls(
            tables,
            None,
            children[n_tables],
            col_start,
            children[end] if has_multiplier else None,
            children[end + has_multiplier] if has_levels else None,
            n_levels,
        )


DeviceBlock = DeviceLookupBlock | DeviceTensorLookupBlock


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DeviceDiscreteDesign:
    """Compact device tables; selectors remain host-batched inputs."""

    blocks: tuple[DeviceBlock, ...]
    n_coef: int
    pair_policy: str
    max_pair_histogram_bytes: int
    batch_rows: int

    def tree_flatten(self):
        return self.blocks, (
            self.n_coef,
            self.pair_policy,
            self.max_pair_histogram_bytes,
            self.batch_rows,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_coef, pair_policy, max_pair_histogram_bytes, batch_rows = aux
        return cls(
            tuple(children), n_coef, pair_policy, max_pair_histogram_bytes, batch_rows
        )


def to_device_discrete_design(
    design: FrozenDiscreteDesign, device: jax.Device | None = None
) -> DeviceDiscreteDesign:
    """Transfer compact tables and frozen transforms, never n-row selectors."""
    blocks: list[DeviceBlock] = []
    for block in design.blocks:
        if isinstance(block, LookupBlock):
            blocks.append(
                DeviceLookupBlock(
                    jax.device_put(jnp.asarray(block.table), device), block.col_start
                )
            )
        else:
            assert isinstance(block, TensorLookupBlock)
            blocks.append(
                DeviceTensorLookupBlock(
                    tuple(
                        jax.device_put(jnp.asarray(table), device)
                        for table in block.margin_tables
                    ),
                    None
                    if block.centering is None
                    else jax.device_put(jnp.asarray(block.centering), device),
                    jax.device_put(jnp.asarray(block.keep), device),
                    block.col_start,
                    None
                    if block.multiplier_table is None
                    else jax.device_put(jnp.asarray(block.multiplier_table), device),
                    None
                    if block.level_codes is None
                    else jax.device_put(jnp.asarray(block.level_codes), device),
                    block.n_levels,
                )
            )
    return DeviceDiscreteDesign(
        tuple(blocks),
        design.n_coef,
        design.control.pair_policy,
        design.control.max_pair_histogram_bytes,
        design.control.batch_rows,
    )


def _validate_device_batch(
    design: DeviceDiscreteDesign, selectors: tuple[tuple[jax.Array, ...], ...]
) -> int:
    """Reject host dispatches larger than the frozen descriptor batch policy."""
    if len(selectors) != len(design.blocks) or not selectors or not selectors[0]:
        raise ValueError("selector block count does not match device design")
    size = selectors[0][0].shape[0]
    if size > design.batch_rows:
        raise ValueError(
            "selector batch exceeds the descriptor's configured batch_rows"
        )
    for block_selectors in selectors:
        if any(selector.shape != (size,) for selector in block_selectors):
            raise ValueError("all selector columns must have the same batch length")
    return size


def _tensor_rows(
    block: DeviceTensorLookupBlock, selectors: tuple[jax.Array, ...]
) -> jax.Array:
    rows = block.tables[0][selectors[0]]
    margin_selectors = selectors[: len(block.tables)]
    rows = block.tables[0][margin_selectors[0]]
    for table, selector in zip(block.tables[1:], margin_selectors[1:], strict=True):
        right = table[selector]
        rows = (rows[:, :, None] * right[:, None, :]).reshape(rows.shape[0], -1)
    if block.level_codes is not None:
        level_selector = selectors[-1]
        codes = block.level_codes[level_selector]
        raw_width = rows.shape[1]
        expanded = jnp.zeros(
            (rows.shape[0], raw_width * block.n_levels), dtype=rows.dtype
        )
        valid = codes >= 0
        safe_codes = jnp.maximum(codes, 0)
        columns = safe_codes[:, None] * raw_width + jnp.arange(raw_width)[None, :]
        expanded = expanded.at[jnp.arange(rows.shape[0])[:, None], columns].set(
            rows * valid[:, None]
        )
        rows = expanded
    if block.centering is not None:
        rows = rows @ block.centering
    rows = rows[:, block.keep]
    if block.multiplier_table is not None:
        rows = rows * block.multiplier_table[selectors[-1]][:, None]
    return rows


def _tensor_raw_coefficients(
    block: DeviceTensorLookupBlock, coefficients: jax.Array
) -> jax.Array:
    """Expand only one coefficient vector through frozen tensor transforms."""
    raw_width = (
        int(np.prod([table.shape[1] for table in block.tables])) * block.n_levels
    )
    if block.centering is None:
        return (
            jnp.zeros(raw_width, dtype=coefficients.dtype)
            .at[block.keep]
            .set(coefficients)
        )
    return block.centering[:, block.keep] @ coefficients


def _tensor_matvec(
    block: DeviceTensorLookupBlock,
    selectors: tuple[jax.Array, ...],
    coefficients: jax.Array,
) -> jax.Array:
    """Evaluate a tensor action without materializing batch-by-raw-tensor rows."""
    margin_selectors = selectors[: len(block.tables)]
    core = _tensor_raw_coefficients(block, coefficients).reshape(
        tuple(table.shape[1] for table in block.tables)
    )
    value = jnp.einsum("bi,i...->b...", block.tables[0][margin_selectors[0]], core)
    for table, selector in zip(block.tables[1:], margin_selectors[1:], strict=True):
        value = jnp.einsum("bi,bi...->b...", table[selector], value)
    if block.multiplier_table is not None:
        value = value * block.multiplier_table[selectors[-1]]
    return value


def _tensor_raw_adjoint(
    block: DeviceTensorLookupBlock,
    selectors: tuple[jax.Array, ...],
    values: jax.Array,
) -> jax.Array:
    """Accumulate one raw tensor coefficient array, never B-by-raw rows."""
    shape = tuple(table.shape[1] for table in block.tables)
    margin_selectors = selectors[: len(block.tables)]
    if block.multiplier_table is not None:
        values = values * block.multiplier_table[selectors[-1]]

    def body(index: int, total: jax.Array) -> jax.Array:
        row = block.tables[0][margin_selectors[0][index]]
        for table, selector in zip(block.tables[1:], margin_selectors[1:], strict=True):
            row = jnp.multiply.outer(row, table[selector[index]])
        return total + values[index] * row

    raw = jax.lax.fori_loop(
        0, values.shape[0], body, jnp.zeros(shape, dtype=values.dtype)
    ).reshape(-1)
    if block.centering is None:
        return raw[block.keep]
    return block.centering[:, block.keep].T @ raw


def _tensor_row_value(
    block: DeviceTensorLookupBlock,
    margin_selectors: tuple[jax.Array, ...],
    row: int,
    core: jax.Array,
) -> jax.Array:
    """Contract one active tensor coefficient core without batch expansion."""
    value = jnp.einsum("i,i...->...", block.tables[0][margin_selectors[0][row]], core)
    for table, selector in zip(block.tables[1:], margin_selectors[1:], strict=True):
        value = jnp.einsum("i,i...->...", table[selector[row]], value)
    return value


def _factor_tensor_matvec(
    block: DeviceTensorLookupBlock,
    selectors: tuple[jax.Array, ...],
    coefficients: jax.Array,
) -> jax.Array:
    """Use one selected level core per row, never B-by-level-by-raw rows."""
    assert block.level_codes is not None
    margin_selectors = selectors[: len(block.tables)]
    codes = block.level_codes[selectors[-1]]
    raw_width = int(np.prod([table.shape[1] for table in block.tables]))
    raw = _tensor_raw_coefficients(block, coefficients).reshape(
        block.n_levels, raw_width
    )

    def body(row: int, out: jax.Array) -> jax.Array:
        code = codes[row]
        return out.at[row].set(
            jax.lax.cond(
                code >= 0,
                lambda active: _tensor_row_value(
                    block,
                    margin_selectors,
                    row,
                    raw[active].reshape(
                        tuple(table.shape[1] for table in block.tables)
                    ),
                ),
                lambda _unused: jnp.asarray(0.0, dtype=coefficients.dtype),
                code,
            )
        )

    return jax.lax.fori_loop(
        0, codes.shape[0], body, jnp.zeros(codes.shape[0], coefficients.dtype)
    )


def _factor_tensor_adjoint(
    block: DeviceTensorLookupBlock,
    selectors: tuple[jax.Array, ...],
    values: jax.Array,
) -> jax.Array:
    """Accumulate bounded level cores before applying frozen transform transpose."""
    assert block.level_codes is not None
    margin_selectors = selectors[: len(block.tables)]
    codes = block.level_codes[selectors[-1]]
    shape = tuple(table.shape[1] for table in block.tables)
    raw_width = int(np.prod(shape))

    def body(row: int, total: jax.Array) -> jax.Array:
        vector = block.tables[0][margin_selectors[0][row]]
        for table, selector in zip(block.tables[1:], margin_selectors[1:], strict=True):
            vector = jnp.multiply.outer(vector, table[selector[row]])
        code = codes[row]
        return jax.lax.cond(
            code >= 0,
            lambda state: state.at[code].add(values[row] * vector.reshape(raw_width)),
            lambda state: state,
            total,
        )

    raw = jax.lax.fori_loop(
        0, values.shape[0], body, jnp.zeros((block.n_levels, raw_width), values.dtype)
    ).reshape(-1)
    if block.centering is None:
        return raw[block.keep]
    return block.centering[:, block.keep].T @ raw


def _block_rows(block: DeviceBlock, selectors: tuple[jax.Array, ...]) -> jax.Array:
    if isinstance(block, DeviceLookupBlock):
        if len(selectors) != 1:
            raise ValueError("ordinary lookup blocks need one selector")
        return block.table[selectors[0]]
    expected = (
        len(block.tables)
        + (block.multiplier_table is not None)
        + (block.level_codes is not None)
    )
    if len(selectors) != expected:
        raise ValueError("tensor selector count does not match tensor margins")
    return _tensor_rows(block, selectors)


def _block_stop(block: DeviceBlock) -> int:
    if isinstance(block, DeviceLookupBlock):
        return block.col_start + block.table.shape[1]
    return block.col_start + block.keep.shape[0]


def _final_table(block: DeviceBlock) -> jax.Array:
    return block.table if isinstance(block, DeviceLookupBlock) else block.tables[-1]


def _final_selector(block: DeviceBlock, selectors: tuple[jax.Array, ...]) -> jax.Array:
    if isinstance(block, DeviceLookupBlock):
        return selectors[0]
    return selectors[len(block.tables) - 1]


def _prefix_count(block: DeviceBlock) -> int:
    if isinstance(block, DeviceLookupBlock):
        return 1
    count = int(np.prod([table.shape[1] for table in block.tables[:-1]]))
    return count * block.n_levels


def _prefix_values(
    block: DeviceBlock,
    selectors: tuple[jax.Array, ...],
    prefix_number: int | jax.Array,
    dtype: jnp.dtype,
) -> jax.Array:
    """Evaluate one all-but-final tensor column, including by metadata."""
    n_rows = selectors[0].shape[0]
    if isinstance(block, DeviceLookupBlock):
        return jnp.ones(n_rows, dtype=dtype)
    widths = tuple(table.shape[1] for table in block.tables[:-1])
    base_count = int(np.prod(widths))
    level = prefix_number // base_count
    remainder = prefix_number % base_count
    columns: list[int | jax.Array] = [0] * len(widths)
    for margin_number in range(len(widths) - 1, -1, -1):
        columns[margin_number] = remainder % widths[margin_number]
        remainder = remainder // widths[margin_number]
    result = jnp.ones(n_rows, dtype=dtype)
    for table, selector, column in zip(
        block.tables[:-1], selectors[: len(widths)], columns, strict=True
    ):
        result = result * table[selector, column]
    if block.multiplier_table is not None:
        result = result * block.multiplier_table[selectors[len(block.tables)]]
    if block.level_codes is not None:
        codes = block.level_codes[selectors[-1]]
        result = result * (codes == level).astype(dtype)
    return result


def _subblock_map(block: DeviceBlock, prefix_number: int | jax.Array) -> jax.Array:
    """Map one raw final-margin block into retained coefficient coordinates."""
    if isinstance(block, DeviceLookupBlock):
        return jnp.eye(block.table.shape[1], dtype=block.table.dtype)
    final_width = block.tables[-1].shape[1]
    start = prefix_number * final_width
    if block.centering is not None:
        rows = jax.lax.dynamic_slice(
            block.centering,
            (start, 0),
            (final_width, block.centering.shape[1]),
        )
        return rows[:, block.keep]
    raw_positions = start + jnp.arange(final_width, dtype=block.keep.dtype)
    return (raw_positions[:, None] == block.keep[None, :]).astype(
        block.tables[-1].dtype
    )


def _final_margin_crossproduct(
    design: DeviceDiscreteDesign,
    left: DeviceBlock,
    right: DeviceBlock,
    left_selectors: tuple[jax.Array, ...],
    right_selectors: tuple[jax.Array, ...],
    weights: jax.Array,
) -> jax.Array:
    """Reduce one final-margin pair by bounded histogram or direct rows."""
    left_table = _final_table(left)
    right_table = _final_table(right)
    left_selector = _final_selector(left, left_selectors)
    right_selector = _final_selector(right, right_selectors)
    plan = pair_reduction_plan(
        left_table.shape[0],
        right_table.shape[0],
        design.max_pair_histogram_bytes,
        n_rows=weights.shape[0],
        k_left=left_table.shape[1],
        k_right=right_table.shape[1],
    )
    if design.pair_policy == "histogram_or_direct" and plan.method == "histogram":
        return lookup_pair_histogram_crossproduct(
            left_table,
            right_table,
            left_selector,
            right_selector,
            weights,
        )
    left_rows = left_table[left_selector]
    right_rows = right_table[right_selector]
    return left_rows.T @ (weights[:, None] * right_rows)


def _subblock_pair_crossproduct(
    design: DeviceDiscreteDesign,
    left: DeviceBlock,
    right: DeviceBlock,
    left_selectors: tuple[jax.Array, ...],
    right_selectors: tuple[jax.Array, ...],
    weights: jax.Array,
) -> jax.Array:
    """Port XWXijs final-margin reuse without full tensor row blocks."""
    initial = jnp.zeros(
        (_block_stop(left) - left.col_start, _block_stop(right) - right.col_start),
        dtype=weights.dtype,
    )
    right_count = _prefix_count(right)

    def body(pair_number: int, part: jax.Array) -> jax.Array:
        left_prefix = pair_number // right_count
        right_prefix = pair_number % right_count
        left_values = _prefix_values(left, left_selectors, left_prefix, weights.dtype)
        left_map = _subblock_map(left, left_prefix)
        right_values = _prefix_values(
            right, right_selectors, right_prefix, weights.dtype
        )
        right_map = _subblock_map(right, right_prefix)
        final_part = _final_margin_crossproduct(
            design,
            left,
            right,
            left_selectors,
            right_selectors,
            weights * left_values * right_values,
        )
        return part + left_map.T @ final_part @ right_map

    part = jax.lax.fori_loop(0, _prefix_count(left) * right_count, body, initial)
    if left is right:
        part = 0.5 * (part + part.T)
    return part


@jax.jit
def discrete_xbd(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    beta: jax.Array,
) -> jax.Array:
    """Bounded indexed ``Xbd`` action."""
    n_rows = _validate_device_batch(design, selectors)
    if beta.shape != (design.n_coef,):
        raise ValueError("discrete Xbd inputs have incompatible shapes")
    result = jnp.zeros(n_rows, dtype=beta.dtype)
    for block, block_selectors in zip(design.blocks, selectors, strict=True):
        coefficients = beta[block.col_start : _block_stop(block)]
        if isinstance(block, DeviceLookupBlock):
            result = result + (block.table @ coefficients)[block_selectors[0]]
        elif block.level_codes is not None:
            result = result + _factor_tensor_matvec(
                block, block_selectors, coefficients
            )
        else:
            result = result + _tensor_matvec(block, block_selectors, coefficients)
    return result


@jax.jit
def discrete_xb_transpose(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    values: jax.Array,
) -> jax.Array:
    """Bounded adjoint action ``X.T @ values``."""
    n_rows = _validate_device_batch(design, selectors)
    if values.shape != (n_rows,):
        raise ValueError("adjoint values do not match selector batch length")
    result = jnp.zeros(design.n_coef, dtype=values.dtype)
    for block, block_selectors in zip(design.blocks, selectors, strict=True):
        if isinstance(block, DeviceLookupBlock):
            selector = block_selectors[0]
            reduced = (
                jnp.zeros(block.table.shape[0], dtype=values.dtype)
                .at[selector]
                .add(values)
            )
            part = block.table.T @ reduced
        else:
            part = (
                _factor_tensor_adjoint(block, block_selectors, values)
                if block.level_codes is not None
                else _tensor_raw_adjoint(block, block_selectors, values)
            )
        result = result.at[block.col_start : _block_stop(block)].set(part)
    return result


@jax.jit
def discrete_xwyd(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    weights: jax.Array,
    response: jax.Array,
) -> jax.Array:
    """Bounded indexed ``X.T @ (w * y)`` action."""
    return discrete_xb_transpose(design, selectors, weights * response)


def _block_key(number: int, block: DeviceBlock, selectors) -> BlockKey:
    if isinstance(block, DeviceLookupBlock):
        rows, columns = block.table.shape
        return LookupKey(rows, columns, str(selectors[0].dtype))
    return TensorKey(number)


def _lookup_stacks(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    keys: tuple[BlockKey, ...],
) -> tuple[dict[BlockKey, tuple[jax.Array, jax.Array]], dict[int, int]]:
    """Stack equal-shape lookup tables and selectors; map block to stack row.

    One concatenation per shape group, not one operation per block, keeps the
    stacking itself out of the graph's growth with the smooth count.
    """
    stacks = {}
    positions = {}
    for key, members in shape_groups(keys):
        if isinstance(key, TensorKey):
            continue
        tables = jnp.concatenate(
            [design.blocks[int(member)].table for member in members]
        ).reshape(len(members), key.rows, key.columns)
        chosen = jnp.concatenate(
            [selectors[int(member)][0] for member in members]
        ).reshape(len(members), -1)
        stacks[key] = (tables, chosen)
        positions.update({int(member): index for index, member in enumerate(members)})
    return stacks, positions


def _pair_part(
    kind: str,
    design: DeviceDiscreteDesign,
    left: DeviceBlock,
    right: DeviceBlock,
    left_selectors: tuple[jax.Array, ...],
    right_selectors: tuple[jax.Array, ...],
    weights: jax.Array,
) -> jax.Array:
    """One block pair's ``X_l' W X_r`` by the reduction the schedule chose."""
    if kind == "lookup_diagonal":
        reduced = (
            jnp.zeros(left.table.shape[0], dtype=weights.dtype)
            .at[left_selectors[0]]
            .add(weights)
        )
        return left.table.T @ (reduced[:, None] * left.table)
    if kind == "tensor":
        return _subblock_pair_crossproduct(
            design, left, right, left_selectors, right_selectors, weights
        )
    if kind == "lookup_histogram":
        return lookup_pair_histogram_crossproduct(
            left.table, right.table, left_selectors[0], right_selectors[0], weights
        )
    left_rows = _block_rows(left, left_selectors)
    right_rows = _block_rows(right, right_selectors)
    return left_rows.T @ (weights[:, None] * right_rows)


@jax.jit
def discrete_xwxd(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    weights: jax.Array,
) -> jax.Array:
    """Bounded direct indexed ``X.T @ diag(w) @ X`` cross-product.

    Block pairs are grouped on the host by reduction and block shapes
    (:func:`~jaxgam.formula.kernel_groups.discrete_pair_schedule`). Each
    group runs one ``lax.scan`` over its pairs, each pair reduced exactly as
    a lone pair is and written to its own disjoint Gram blocks, so the graph
    grows with the number of distinct block shapes, not with the number of
    smooth pairs.
    """
    n_rows = _validate_device_batch(design, selectors)
    if weights.shape != (n_rows,):
        raise ValueError("weights do not match selector batch length")
    keys = tuple(
        _block_key(number, block, block_selectors)
        for number, (block, block_selectors) in enumerate(
            zip(design.blocks, selectors, strict=True)
        )
    )
    stacks, positions = _lookup_stacks(design, selectors, keys)
    schedule = discrete_pair_schedule(
        keys,
        pair_policy=design.pair_policy,
        max_pair_histogram_bytes=design.max_pair_histogram_bytes,
        n_rows=n_rows,
    )
    starts = np.asarray([block.col_start for block in design.blocks], dtype=np.int64)

    def side(key: BlockKey, index: jax.Array):
        if isinstance(key, TensorKey):
            return design.blocks[key.block], selectors[key.block]
        tables, chosen = stacks[key]
        return DeviceLookupBlock(tables[index], 0), (chosen[index],)

    def stack_rows(key: BlockKey, numbers: np.ndarray) -> np.ndarray:
        if isinstance(key, TensorKey):
            return np.zeros(len(numbers), dtype=np.int64)
        return np.asarray([positions[int(number)] for number in numbers])

    result = jnp.zeros((design.n_coef, design.n_coef), dtype=weights.dtype)
    for group in schedule:
        # A block with itself: a lookup's own histogram, or a tensor group
        # whose two keys name the same tensor block.
        diagonal = group.kind == "lookup_diagonal" or (
            group.kind == "tensor" and group.left_key == group.right_key
        )

        def pair(result, item, group=group, diagonal=diagonal):
            left_index, right_index, left_start, right_start = item
            left, left_selectors = side(group.left_key, left_index)
            right, right_selectors = (
                (left, left_selectors)
                if diagonal
                else side(group.right_key, right_index)
            )
            part = _pair_part(
                group.kind,
                design,
                left,
                right,
                left_selectors,
                right_selectors,
                weights,
            )
            result = jax.lax.dynamic_update_slice(
                result, part, (left_start, right_start)
            )
            if not diagonal:
                result = jax.lax.dynamic_update_slice(
                    result, part.T, (right_start, left_start)
                )
            return result, None

        result, _ = jax.lax.scan(
            pair,
            result,
            (
                jnp.asarray(stack_rows(group.left_key, group.left)),
                jnp.asarray(stack_rows(group.right_key, group.right)),
                jnp.asarray(starts[group.left]),
                jnp.asarray(starts[group.right]),
            ),
        )
    return result


# discrete_row_xwxd forms the upper panels of an equal split of the columns.
_ROW_XWXD_PANELS = 4


# Flop-equivalent cost of touching one row in a pair reduction (a gather or a
# histogram scatter): measured on the 100-smooth section 8 design, about 15 ns
# per row and pair against about 6 ps per matrix-product flop. Set below that
# ratio so row chunks are chosen only with a clear margin.
_ROW_TOUCH_FLOPS = 1000


def _block_shape(block: DeviceBlock) -> tuple[int, int] | None:
    """``(q, k)`` of a lookup block's table; None for a tensor block."""
    return tuple(block.table.shape) if isinstance(block, DeviceLookupBlock) else None


def discrete_xwxd_cost(design: DeviceDiscreteDesign, n_rows: int) -> float:
    """Flop-equivalent estimate of :func:`discrete_xwxd` on ``n_rows`` rows.

    Per pair, as the kernel reduces it: a lookup block with itself costs a
    weight histogram and ``T' diag(h) T``; two lookup blocks a pair histogram
    and ``T_l' H T_r`` or direct rows (:func:`pair_reduction_plan`); a pair
    with a tensor block is charged as direct rows of its full width.
    """
    total = 0.0
    blocks = design.blocks
    for left_number, left in enumerate(blocks):
        for right in blocks[left_number:]:
            left_shape, right_shape = _block_shape(left), _block_shape(right)
            touch = n_rows * _ROW_TOUCH_FLOPS
            if left_shape is None or right_shape is None:
                widths = (
                    _block_stop(left) - left.col_start,
                    _block_stop(right) - right.col_start,
                )
                total += touch + 2.0 * n_rows * widths[0] * widths[1]
                continue
            (q_left, k_left), (q_right, k_right) = left_shape, right_shape
            if left is right:
                total += touch + 2.0 * q_left * k_left * k_left
            elif (
                design.pair_policy == "histogram_or_direct"
                and pair_reduction_plan(
                    q_left,
                    q_right,
                    design.max_pair_histogram_bytes,
                    n_rows=n_rows,
                    k_left=k_left,
                    k_right=k_right,
                ).method
                == "histogram"
            ):
                total += (
                    touch
                    + q_left * q_right * (2.0 * k_left + 8.0)
                    + 2.0 * k_left * q_right * k_right
                )
            else:
                total += touch + 2.0 * n_rows * k_left * k_right
    return total


def discrete_row_xwxd_cost(n_coef: int, n_rows: int) -> float:
    """Flop-equivalent estimate of :func:`discrete_row_xwxd` on ``n_rows`` rows."""
    panels = _ROW_XWXD_PANELS
    return 2.0 * n_rows * n_coef * n_coef * (panels + 1) / (2 * panels) + (
        n_rows * n_coef
    )


def discrete_row_xwxd_bytes(n_coef: int, chunk_rows: int) -> int:
    """Device scratch of :func:`discrete_row_xwxd` beyond its p x p result.

    One chunk of materialized rows, its weighted copy and the gathered
    lookup rows before placement, plus the carried and updated sums.
    """
    float_bytes = np.dtype(np.float64).itemsize
    return float_bytes * (3 * chunk_rows * n_coef + 2 * n_coef * n_coef)


@partial(jax.jit, static_argnames=("chunk_rows",))
def discrete_row_xwxd(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    weights: jax.Array,
    *,
    chunk_rows: int,
) -> jax.Array:
    """``X.T @ diag(w) @ X`` from materialized row chunks of the compact design.

    Each chunk of ``chunk_rows`` rows is expanded to its dense ``rows x p``
    model matrix (lookup tables gathered per equal-shape group, tensor blocks
    by their row product) and reduced with one matrix product, so the work is
    ``n p^2`` in matrix products rather than one gather or histogram per pair
    of blocks. It equals :func:`discrete_xwxd` up to rounding; the upper
    triangle is mirrored so the result is exactly symmetric, as that kernel's
    is. Rows past the batch are padded with weight zero.
    """
    n_rows = _validate_device_batch(design, selectors)
    if weights.shape != (n_rows,):
        raise ValueError("weights do not match selector batch length")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    chunk_rows = min(chunk_rows, n_rows)
    n_chunks = -(-n_rows // chunk_rows)
    padded = n_chunks * chunk_rows

    def pad(values: jax.Array) -> jax.Array:
        return jnp.pad(values, (0, padded - n_rows)).reshape(n_chunks, chunk_rows)

    keys = tuple(
        _block_key(number, block, block_selectors)
        for number, (block, block_selectors) in enumerate(
            zip(design.blocks, selectors, strict=True)
        )
    )
    lookup_groups = []
    for key, members in shape_groups(keys):
        if isinstance(key, TensorKey):
            continue
        members = [int(member) for member in members]
        tables = jnp.stack([design.blocks[member].table for member in members])
        chosen = jnp.stack([pad(selectors[member][0]) for member in members], axis=1)
        columns = np.concatenate(
            [
                design.blocks[member].col_start + np.arange(key.columns)
                for member in members
            ]
        )
        lookup_groups.append((tables, chosen, columns))
    tensors = [
        (
            design.blocks[key.block],
            tuple(pad(selector) for selector in selectors[key.block]),
        )
        for key in keys
        if isinstance(key, TensorKey)
    ]
    member_index = [jnp.arange(len(tables))[:, None] for tables, _, _ in lookup_groups]

    def chunk(total, item):
        chunk_weights, lookup_selectors, tensor_selectors = item
        X = jnp.zeros((chunk_rows, design.n_coef), dtype=weights.dtype)
        for (tables, _, columns), index, chosen in zip(
            lookup_groups, member_index, lookup_selectors, strict=True
        ):
            rows = tables[index, chosen]
            X = X.at[:, columns].set(
                jnp.swapaxes(rows, 0, 1).reshape(chunk_rows, len(columns))
            )
        for (block, _), block_selectors in zip(tensors, tensor_selectors, strict=True):
            X = X.at[:, block.col_start : _block_stop(block)].set(
                _block_rows(block, block_selectors)
            )
        weighted = chunk_weights[:, None] * X
        return tuple(
            part
            + X[:, edges[i] : edges[i + 1]].T @ weighted[:, edges[j] : edges[j + 1]]
            for part, (i, j) in zip(total, panels, strict=True)
        ), None

    # Only the upper panels of the symmetric product are formed.
    edges = np.linspace(0, design.n_coef, _ROW_XWXD_PANELS + 1).astype(int)
    panels = [
        (i, j) for i in range(_ROW_XWXD_PANELS) for j in range(i, _ROW_XWXD_PANELS)
    ]
    total, _ = jax.lax.scan(
        chunk,
        tuple(
            jnp.zeros(
                (edges[i + 1] - edges[i], edges[j + 1] - edges[j]), dtype=weights.dtype
            )
            for i, j in panels
        ),
        (
            pad(weights),
            tuple(chosen for _, chosen, _ in lookup_groups),
            tuple(block_selectors for _, block_selectors in tensors),
        ),
    )
    upper = dict(zip(panels, total, strict=True))
    total = jnp.block(
        [
            [
                upper[(i, j)] if i <= j else upper[(j, i)].T
                for j in range(_ROW_XWXD_PANELS)
            ]
            for i in range(_ROW_XWXD_PANELS)
        ]
    )
    return jnp.triu(total) + jnp.triu(total, 1).T


@jax.jit
def discrete_diag_xvxd(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    covariance: jax.Array,
) -> jax.Array:
    """Bounded ``diag(X V X.T)`` action."""
    if covariance.shape != (design.n_coef, design.n_coef):
        raise ValueError("covariance does not match discrete coefficient width")
    n_rows = _validate_device_batch(design, selectors)
    result = jnp.zeros(n_rows, dtype=covariance.dtype)
    for left, left_selectors in zip(design.blocks, selectors, strict=True):
        left_rows = _block_rows(left, left_selectors)
        left_slice = slice(left.col_start, _block_stop(left))
        for right, right_selectors in zip(design.blocks, selectors, strict=True):
            right_rows = _block_rows(right, right_selectors)
            right_slice = slice(right.col_start, _block_stop(right))
            result = result + jnp.einsum(
                "bi,ij,bj->b",
                left_rows,
                covariance[left_slice, right_slice],
                right_rows,
            )
    return result


@jax.jit
def lookup_pair_histogram_crossproduct(
    left_table: jax.Array,
    right_table: jax.Array,
    left_selector: jax.Array,
    right_selector: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Bounded C_ij histogram reference for one ordinary pair of tables.

    Callers must choose this only after :func:`pair_reduction_plan` accepts
    its q_left-by-q_right storage.  The ordinary full cross-product kernel
    remains direct when the pair table would be sparse or too large.
    """
    histogram = jnp.zeros(
        (left_table.shape[0], right_table.shape[0]), dtype=weights.dtype
    )
    histogram = histogram.at[left_selector, right_selector].add(weights)
    return left_table.T @ histogram @ right_table
