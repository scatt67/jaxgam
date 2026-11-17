"""Budgeted inverse preconditioners for a frozen SPD operator system.

Large coefficient blocks retain a diagonal only. Small term factors and
coarse directions use an independently capped modest coordinate set; coarse
vectors are never expanded into a p-by-r array. These are new algorithms,
not mgcv's direct Cholesky solver or a change to the observed system.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.scipy.linalg as jsla
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SmallBlockInverse:
    columns: jax.Array
    lower: jax.Array

    def tree_flatten(self):
        return (self.columns, self.lower), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CoarseInverse:
    columns: jax.Array
    directions: jax.Array
    lower: jax.Array

    def tree_flatten(self):
        return (self.columns, self.directions, self.lower), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class InversePreconditioner:
    diagonal: jax.Array
    blocks: tuple[SmallBlockInverse, ...] = ()
    coarse: CoarseInverse | None = None

    def tree_flatten(self):
        children = (self.diagonal, *self.blocks)
        if self.coarse is not None:
            children = (*children, self.coarse)
        return children, (len(self.blocks), self.coarse is not None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        count, has_coarse = aux
        return cls(
            children[0],
            tuple(children[1 : count + 1]),
            children[-1] if has_coarse else None,
        )


@dataclass(frozen=True)
class PreparedPreconditioner:
    """Owned inverse action and conservative additional construction storage."""

    inverse: InversePreconditioner
    retained_device_bytes: int
    construction_peak_bound_bytes: int
    # The optional coarse correction's share, reported separately because
    # the plan accounts a permitted small coarse preconditioner on its own.
    coarse_retained_device_bytes: int = 0
    coarse_construction_bound_bytes: int = 0


def _owned_device(array: np.ndarray, device: jax.Device | None) -> jax.Array:
    copy = array.copy()
    copy.setflags(write=False)
    return jax.device_put(copy, device)


def _validate_columns(columns: np.ndarray, p: int) -> None:
    if (
        columns.ndim != 1
        or columns.dtype.kind not in "iu"
        or np.any(columns < 0)
        or np.any(columns >= p)
        or len(set(columns.tolist())) != len(columns)
    ):
        raise ValueError("Preconditioner columns must be distinct valid integers.")


def _factor(matrix: np.ndarray) -> np.ndarray:
    if matrix.dtype != np.float64 or not np.all(np.isfinite(matrix)):
        raise ValueError("Preconditioner matrices must be finite float64.")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError("Preconditioner matrices must be symmetric.")
    try:
        return np.linalg.cholesky((matrix + matrix.T) * 0.5)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "Preconditioner matrix is not positive definite."
        ) from error


def prepare_preconditioner(
    diagonal: np.ndarray,
    *,
    blocks: tuple[tuple[np.ndarray, np.ndarray], ...] = (),
    coarse: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    max_modest_coefficients: int = 64,
    memory_budget_bytes: int,
    device: jax.Device | None = None,
) -> PreparedPreconditioner:
    """Build frozen diagonal/block-Jacobi plus additive coarse inverse.

    Inputs are caller owned. The bound includes validation masks, private host
    copies, device ownership and one capped dense factorization at a time.
    The host separately charges input statistics and executable workspace.
    """
    for name, value in (
        ("max_modest_coefficients", max_modest_coefficients),
        ("memory_budget_bytes", memory_budget_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if (
        not isinstance(diagonal, np.ndarray)
        or diagonal.ndim != 1
        or diagonal.dtype != np.float64
    ):
        raise ValueError("Preconditioner diagonal must be a float64 vector.")
    p = len(diagonal)
    required = 17 * p
    modest_width = 0
    for columns, matrix in blocks:
        q = len(columns)
        if matrix.shape != (q, q) or q == 0:
            raise ValueError("Small preconditioner block shape is invalid.")
        modest_width += q
        required += 48 * q**2 + 24 * q
    if modest_width > max_modest_coefficients:
        raise MemoryError("Small preconditioner blocks exceed the hard modest cap.")
    coarse_required = 0
    if coarse is not None:
        columns, directions, matrix = coarse
        q = len(columns)
        if directions.ndim != 2:
            raise ValueError("Coarse directions must be a modest-coordinate matrix.")
        r = directions.shape[1]
        if (
            directions.shape[0] != q
            or matrix.shape != (r, r)
            or r == 0
            or r > q
            or q > max_modest_coefficients
        ):
            raise MemoryError("Coarse preconditioner dimensions exceed their hard cap.")
        coarse_required = 16 * q * r + 48 * r**2 + 24 * q
        required += coarse_required
    if required > memory_budget_bytes:
        raise MemoryError(f"Preconditioner construction requires {required} bytes.")
    if p == 0 or not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise ValueError("Preconditioner diagonal must be positive and finite.")
    occupied = set()
    inverse_blocks = []
    for columns, matrix in blocks:
        _validate_columns(columns, p)
        if occupied.intersection(columns.tolist()):
            raise ValueError("Small preconditioner blocks must not overlap.")
        occupied.update(columns.tolist())
        lower = _factor(matrix)
        inverse_blocks.append(
            SmallBlockInverse(
                _owned_device(columns, device), _owned_device(lower, device)
            )
        )
        del lower
    inverse_coarse = None
    if coarse is not None:
        columns, directions, matrix = coarse
        _validate_columns(columns, p)
        if directions.dtype != np.float64 or not np.all(np.isfinite(directions)):
            raise ValueError("Coarse directions must be finite float64.")
        lower = _factor(matrix)
        inverse_coarse = CoarseInverse(
            _owned_device(columns, device),
            _owned_device(directions, device),
            _owned_device(lower, device),
        )
    inverse = InversePreconditioner(
        _owned_device(diagonal, device), tuple(inverse_blocks), inverse_coarse
    )
    retained = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(inverse))
    coarse_retained = (
        0
        if inverse_coarse is None
        else sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(inverse_coarse))
    )
    return PreparedPreconditioner(
        inverse, retained, required, coarse_retained, coarse_required
    )


@jax.jit
def apply_preconditioner(
    inverse: InversePreconditioner, vector: jax.Array
) -> jax.Array:
    """Apply a fixed SPD inverse action with only vectors and modest matrices."""
    if vector.shape != inverse.diagonal.shape:
        raise ValueError("Preconditioner vector must match the coefficient dimension.")
    result = vector / inverse.diagonal
    for block in inverse.blocks:
        value = jsla.cho_solve((block.lower, True), vector[block.columns])
        result = result.at[block.columns].set(value)
    if inverse.coarse is not None:
        coarse = inverse.coarse
        rhs = coarse.directions.T @ vector[coarse.columns]
        solution = jsla.cho_solve((coarse.lower, True), rhs)
        result = result.at[coarse.columns].add(coarse.directions @ solution)
    return result
