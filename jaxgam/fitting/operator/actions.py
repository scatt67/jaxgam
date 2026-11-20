"""Pure JIT batch actions for bounded designs with indexed random effects.

These implement Xbd/XWyd's gather/reduce contract in public coordinates.
The host owns source iteration; no kernel constructs whole-source Xv, dense
random-effect rows or a coefficient cross-product matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.fitting import penalty_ops
from jaxgam.formula.operator.prepare import OperatorBatch


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DeviceIndexedRows:
    """One level ID/amplitude per row and static coefficient placement."""

    indices: jax.Array
    amplitudes: jax.Array
    start: int
    size: int

    def tree_flatten(self):
        return (self.indices, self.amplitudes), (self.start, self.size)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DeviceOperatorBatch:
    """Modest dense rows and symbolic indexed term actions."""

    modest_values: jax.Array
    modest_columns: jax.Array
    indexed_rows: tuple[DeviceIndexedRows, ...]
    n_coef: int

    def tree_flatten(self):
        return (
            self.modest_values,
            self.modest_columns,
            *self.indexed_rows,
        ), self.n_coef

    @classmethod
    def tree_unflatten(cls, n_coef, children):
        return cls(children[0], children[1], tuple(children[2:]), n_coef)


def to_device_operator_batch(
    batch: OperatorBatch,
    *,
    memory_budget_bytes: int,
    device: jax.Device | None = None,
) -> DeviceOperatorBatch:
    """Validate and charge host plus device row ownership before transfer.

    The orchestrator separately charges retained setup, Krylov vectors and
    executable scratch. This budget is its remaining row-transfer allowance.
    """
    if (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, int)
        or memory_budget_bytes < 1
    ):
        raise ValueError("memory_budget_bytes must be a positive integer.")
    if batch.modest_values.ndim != 2 or batch.modest_values.dtype != np.float64:
        raise ValueError("Modest operator rows must be a float64 matrix.")
    n, width = batch.modest_values.shape
    if width != len(batch.modest_columns):
        raise ValueError("Modest operator placement must match its row width.")
    required = 2 * (8 * n * width + 8 * width + 16 * n * len(batch.indexed_rows))
    if required > memory_budget_bytes:
        raise MemoryError(f"Operator row transfer needs {required} bytes.")
    if (
        batch.n_coef < 0
        or any(c < 0 or c >= batch.n_coef for c in batch.modest_columns)
        or len(set(batch.modest_columns)) != width
    ):
        raise ValueError("Operator coefficient placement is invalid.")
    if not np.all(np.isfinite(batch.modest_values)):
        raise ValueError("Modest operator values must be finite.")
    rows = []
    occupied = set(batch.modest_columns)
    # Intervals are validated without a p-sized column mask or Cartesian grid.
    intervals = []
    for ids, amplitudes, start, size in batch.indexed_rows:
        if (
            ids.shape != (n,)
            or ids.dtype.kind not in "iu"
            or amplitudes.shape != (n,)
            or amplitudes.dtype != np.float64
            or not np.all(np.isfinite(amplitudes))
            or np.any(ids < 0)
            or np.any(ids >= size)
            or size < 1
            or start < 0
            or start + size > batch.n_coef
            or any(start <= c < start + size for c in occupied)
        ):
            raise ValueError("Indexed operator rows or placement are invalid.")
        if any(start < right and left < start + size for left, right in intervals):
            raise ValueError("Indexed operator intervals must not overlap.")
        intervals.append((start, start + size))
        rows.append(
            DeviceIndexedRows(
                jax.device_put(ids, device),
                jax.device_put(amplitudes, device),
                start,
                size,
            )
        )
    return DeviceOperatorBatch(
        jax.device_put(batch.modest_values, device),
        jax.device_put(np.asarray(batch.modest_columns, dtype=np.int64), device),
        tuple(rows),
        batch.n_coef,
    )


@jax.jit
def matvec_batch(batch: DeviceOperatorBatch, vector: jax.Array) -> jax.Array:
    """Evaluate Xv in one batch, using gathers for random effects."""
    if vector.shape != (batch.n_coef,):
        raise ValueError("Operator vector must match the coefficient dimension.")
    result = batch.modest_values @ vector[batch.modest_columns]
    for rows in batch.indexed_rows:
        result = result + rows.amplitudes * vector[rows.start + rows.indices]
    return result


@jax.jit
def rmatvec_batch(batch: DeviceOperatorBatch, vector: jax.Array) -> jax.Array:
    """Evaluate X'u with indexed reductions and no dense random-effect rows."""
    if vector.shape != (batch.modest_values.shape[0],):
        raise ValueError("Adjoint vector must match the batch row count.")
    result = jnp.zeros(batch.n_coef, dtype=jnp.float64)
    result = result.at[batch.modest_columns].add(batch.modest_values.T @ vector)
    for rows in batch.indexed_rows:
        result = result.at[rows.start + rows.indices].add(rows.amplitudes * vector)
    return result


@jax.jit
def normal_matvec_batch(
    batch: DeviceOperatorBatch, vector: jax.Array, weight: jax.Array
) -> jax.Array:
    """Fuse X'W Xv while beta/theta/weights are frozen by the caller."""
    if weight.shape != (batch.modest_values.shape[0],):
        raise ValueError("Normal operator weights must match the batch row count.")
    return rmatvec_batch(batch, weight * matvec_batch(batch, vector))


def _accumulate_adjoint(
    accumulator: jax.Array, batch: DeviceOperatorBatch, values: jax.Array
) -> jax.Array:
    if accumulator.shape != (batch.n_coef,):
        raise ValueError("Accumulator must match the coefficient dimension.")
    if values.shape != (batch.modest_values.shape[0],):
        raise ValueError("Adjoint vector must match the batch row count.")
    result = accumulator.at[batch.modest_columns].add(batch.modest_values.T @ values)
    for rows in batch.indexed_rows:
        result = result.at[rows.start + rows.indices].add(rows.amplitudes * values)
    return result


@partial(jax.jit, donate_argnums=(0,))
def accumulate_rmatvec_batch(
    accumulator: jax.Array, batch: DeviceOperatorBatch, values: jax.Array
) -> jax.Array:
    """Accumulate an RHS/gradient in donated storage without another p-vector."""
    return _accumulate_adjoint(accumulator, batch, values)


@partial(jax.jit, donate_argnums=(0,))
def accumulate_normal_matvec_batch(
    accumulator: jax.Array,
    batch: DeviceOperatorBatch,
    vector: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Accumulate X'WXv in donated storage across host source batches.

    Only the original action allocates its p-vector accumulator. Subsequent
    batches update that same storage rather than zeroing another p-vector
    or transferring a coefficient vector through the host for each batch.
    """
    if weight.shape != (batch.modest_values.shape[0],):
        raise ValueError("Normal operator weights must match the batch row count.")
    return _accumulate_adjoint(accumulator, batch, weight * matvec_batch(batch, vector))


@jax.jit
def penalty_matvec(
    structure: penalty_ops.JaxPenaltyStructure, vector: jax.Array, rho: jax.Array
) -> jax.Array:
    """Use Sl.Sb's local actions without global penalty materialization."""
    if vector.shape != (structure.n_coef,) or rho.shape != (structure.n_penalties,):
        raise ValueError("Penalty vectors must match the frozen model dimensions.")
    return penalty_ops.apply(structure, vector, rho)
