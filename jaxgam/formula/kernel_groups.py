"""Host grouping of equal-shape items for stacked JIT kernels (NumPy only).

A JIT kernel that loops in Python over smooths, or over pairs of smooths,
unrolls one copy of the loop body per item into its graph: the graph grows
with the number of smooths (or its square) and XLA's compile time faster
still. Kernels instead stack the items that share a shape and drive each
stack with one traced loop. This module forms those groups and pair
schedules on the host from static shapes, so the number of traced loops
depends on the number of distinct shapes, not on the number of smooths.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def shape_groups(
    keys: Sequence[Hashable],
) -> tuple[tuple[Hashable, np.ndarray], ...]:
    """Group item positions by key.

    Groups follow the first appearance of their key and list member positions
    in ascending order, so a kernel that walks each group's members in turn
    visits equal-key items in their original relative order.
    """
    members: dict[Hashable, list[int]] = {}
    for position, key in enumerate(keys):
        members.setdefault(key, []).append(position)
    return tuple(
        (key, np.asarray(positions, dtype=np.int64))
        for key, positions in members.items()
    )


@dataclass(frozen=True)
class PairReductionPlan:
    """A bounded choice for one lookup-block cross-product."""

    method: str
    required_bytes: int
    budget_bytes: int


def pair_reduction_plan(
    q_left: int,
    q_right: int,
    max_bytes: int,
    *,
    n_rows: int | None = None,
    k_left: int = 1,
    k_right: int = 1,
) -> PairReductionPlan:
    """Choose mgcv-style pair accumulation only when bounded and worthwhile."""
    required = q_left * q_right * np.dtype(np.float64).itemsize
    direct_bytes = (
        None
        if n_rows is None
        else n_rows * (k_left + k_right) * np.dtype(np.float64).itemsize
    )
    return PairReductionPlan(
        "histogram"
        if required <= max_bytes and (direct_bytes is None or required <= direct_bytes)
        else "direct",
        required,
        max_bytes,
    )


@dataclass(frozen=True)
class LookupKey:
    """Static shape of an ordinary lookup block: table rows/columns, selector type."""

    rows: int
    columns: int
    selector_dtype: str


@dataclass(frozen=True)
class TensorKey:
    """A tensor lookup block; each is its own group (identified by position)."""

    block: int


BlockKey = LookupKey | TensorKey


@dataclass(frozen=True)
class DiscretePairGroup:
    """Upper-triangle block pairs sharing one reduction and both block shapes.

    ``left``/``right`` are block positions with ``left <= right``, in the
    row-major pair order of the unrolled ``XWXd`` loop.
    """

    kind: str  # lookup_diagonal | lookup_histogram | lookup_direct | tensor
    left_key: BlockKey
    right_key: BlockKey
    left: np.ndarray
    right: np.ndarray


@lru_cache(maxsize=64)
def discrete_pair_schedule(
    keys: tuple[BlockKey, ...],
    *,
    pair_policy: str,
    max_pair_histogram_bytes: int,
    n_rows: int,
) -> tuple[DiscretePairGroup, ...]:
    """Group ``XWXd``'s block pairs by reduction kind and block shapes.

    Each pair's reduction is the one the per-pair loop chose: a weight
    histogram for a lookup block with itself, the bounded pair histogram or
    direct rows for two lookup blocks (:func:`pair_reduction_plan`), and the
    final-margin subblock reduction when either block is a tensor. Results
    are cached per static structure and their arrays are read-only.
    """
    pairs: list[tuple[int, int]] = []
    pair_keys: list[tuple[str, BlockKey, BlockKey]] = []
    for left, left_key in enumerate(keys):
        for right in range(left, len(keys)):
            right_key = keys[right]
            if isinstance(left_key, TensorKey) or isinstance(right_key, TensorKey):
                kind = "tensor"
            elif left == right:
                kind = "lookup_diagonal"
            elif (
                pair_policy == "histogram_or_direct"
                and pair_reduction_plan(
                    left_key.rows,
                    right_key.rows,
                    max_pair_histogram_bytes,
                    n_rows=n_rows,
                    k_left=left_key.columns,
                    k_right=right_key.columns,
                ).method
                == "histogram"
            ):
                kind = "lookup_histogram"
            else:
                kind = "lookup_direct"
            pairs.append((left, right))
            pair_keys.append((kind, left_key, right_key))
    positions = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    groups = []
    for (kind, left_key, right_key), members in shape_groups(pair_keys):
        left, right = positions[members].T.copy()
        left.setflags(write=False)
        right.setflags(write=False)
        groups.append(DiscretePairGroup(kind, left_key, right_key, left, right))
    return tuple(groups)
