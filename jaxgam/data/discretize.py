"""Exact, bounded CPU indexing for discrete design operators.

This module changes only how repeated observed covariate values are stored.
It deliberately does not round continuous variables or construct a model
basis: those decisions belong to the already-frozen formula setup.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from jaxgam.data.source import RowSource


def _readonly(value: npt.ArrayLike, dtype: npt.DTypeLike | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DiscretizationBudget:
    """Explicit storage limits for exact discrete descriptors.

    ``max_unique_rows`` prevents the discovery dictionary from growing beyond
    the approved representation.  Formula construction supplies a tighter
    cap based on the prospective basis-table width before it evaluates one.
    """

    max_table_bytes: int = 64 * 1024 * 1024
    max_selector_bytes: int = 256 * 1024 * 1024
    max_batch_bytes: int = 64 * 1024 * 1024
    max_unique_rows: int = 1_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_table_bytes",
            "max_selector_bytes",
            "max_batch_bytes",
            "max_unique_rows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ExactIndexUnavailable:
    """A bounded exact representation could not be constructed."""

    reason: str
    required_bytes: int
    budget_bytes: int
    required_rows: int | None = None
    row_cap: int | None = None


@dataclass(frozen=True)
class ExactIndexTable:
    """Unique observed covariate rows and their zero-based row selector."""

    names: tuple[str, ...]
    values: tuple[npt.NDArray[Any], ...]
    selector: npt.NDArray[np.int32]
    n_rows: int
    storage: Literal["memory", "memmap"]
    construction_peak_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.values):
            raise ValueError(
                "exact index table needs one non-empty value column per name"
            )
        selector = np.asarray(self.selector)
        if selector.dtype != np.int32 or selector.ndim != 1:
            raise TypeError(
                "exact index selector must be a one-dimensional int32 array"
            )
        if selector.shape[0] != self.n_rows:
            raise ValueError("exact index selector has incompatible row count")
        q = len(self.values[0])
        if q == 0 or any(len(value) != q for value in self.values):
            raise ValueError("exact index values must share a positive row count")
        if np.any(selector < 0) or np.any(selector >= q):
            raise ValueError("exact index selector is out of bounds")
        object.__setattr__(
            self, "values", tuple(_readonly(value) for value in self.values)
        )
        # The builder owns this selector until construction succeeds.  Freezing
        # it in place avoids a second n-row copy solely for dataclass ownership.
        selector.setflags(write=False)
        object.__setattr__(self, "selector", selector)
        if self.construction_peak_bytes < 0:
            raise ValueError("construction peak bytes cannot be negative")

    @property
    def n_unique(self) -> int:
        return len(self.values[0])

    @property
    def selector_bytes(self) -> int:
        return int(self.selector.nbytes)

    @property
    def value_bytes(self) -> int:
        """Conservative resident size of retained unique-value columns."""
        total = 0
        for value in self.values:
            total += int(value.nbytes)
            if value.dtype == object:
                total += sum(sys.getsizeof(item) for item in value)
        return total

    def batch_selector(self, row_positions: npt.ArrayLike) -> npt.NDArray[np.int32]:
        positions = np.asarray(row_positions)
        if positions.ndim != 1 or positions.dtype.kind not in "iu":
            raise TypeError("row positions must be a one-dimensional integer array")
        if np.any(positions < 0) or np.any(positions >= self.n_rows):
            raise ValueError("row positions are out of bounds for exact index table")
        return np.asarray(self.selector[positions], dtype=np.int32)


def _key(values: tuple[Any, ...]) -> tuple[tuple[str, Any], ...]:
    """Make NaN equality explicit while retaining exact finite values."""
    result: list[tuple[str, Any]] = []
    for value in values:
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            result.append(("nan", 0))
        else:
            result.append((type(value).__qualname__, value))
    return tuple(result)


def _deep_size(value: object, seen: set[int] | None = None) -> int:
    """Conservatively count nested Python discovery state once per object.

    Exact indexing uses tuples nested inside a dictionary.  Shallow container
    sizes omit the tuple entries and NumPy/Python scalar payloads that dominate
    highly-cardinal inputs, so construction budgeting needs a small recursive
    counter.  This is used only for each newly discovered key, not to rescan
    the whole dictionary.
    """
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    elif isinstance(value, (tuple, list, set, frozenset)):
        total += sum(_deep_size(item, seen) for item in value)
    return total


def _scalar_array_dtype(value: Any) -> np.dtype[Any]:
    """Choose storage that preserves the exact source scalar and its type."""
    if isinstance(value, np.generic):
        return value.dtype
    # Values drawn from object/categorical source columns are Python scalars.
    # Retaining them as object references avoids NumPy's list coercion of a
    # mixed factor such as ("a", 12, 3.5) to strings.
    return np.dtype(object)


def _column_dtype_and_bytes(
    unique: list[tuple[Any, ...]], column_number: int
) -> tuple[np.dtype[Any], int]:
    """Bound a unique column's final storage before ``np.asarray`` runs."""
    dtype: np.dtype[Any] | None = None
    for row in unique:
        scalar_dtype = _scalar_array_dtype(row[column_number])
        if dtype is None:
            dtype = scalar_dtype
        elif dtype != scalar_dtype:
            # A RowSource with changing batch dtypes must preserve each scalar
            # type rather than promote values after exact keys were recorded.
            dtype = np.dtype(object)
    assert dtype is not None
    array_bytes = len(unique) * dtype.itemsize
    if dtype == np.dtype(object):
        array_bytes += sum(sys.getsizeof(row[column_number]) for row in unique)
    return dtype, array_bytes


def exact_index_columns(
    source: RowSource,
    names: tuple[str, ...],
    *,
    budget: DiscretizationBudget,
    max_unique_rows: int | None = None,
    selector_memmap: str | Path | None = None,
    batch_rows: int = 65_536,
    max_construction_bytes: int | None = None,
) -> ExactIndexTable | ExactIndexUnavailable:
    """Index exact joint observed values without continuous discretization.

    The selector allocation is checked from the advertised source length
    first.  Unique discovery is capped before a prospective lookup table can
    become unbounded.  A memmap path is caller-owned: this routine neither
    creates a temporary hidden source nor retains selectors on a device.
    """
    if not names or len(set(names)) != len(names):
        raise ValueError("exact indexing requires distinct non-empty column names")
    n_rows = source.n_rows
    if (
        isinstance(batch_rows, bool)
        or not isinstance(batch_rows, int)
        or batch_rows <= 0
    ):
        raise ValueError("batch_rows must be a positive integer")
    position_workspace_bytes = 2 * batch_rows * np.dtype(np.intp).itemsize
    if position_workspace_bytes > budget.max_batch_bytes:
        return ExactIndexUnavailable(
            "batch position-validation workspace exceeds budget",
            position_workspace_bytes,
            budget.max_batch_bytes,
        )
    selector_bytes = n_rows * np.dtype(np.int32).itemsize
    validation_bytes = n_rows * np.dtype(bool).itemsize
    resident_selector_bytes = validation_bytes + (
        selector_bytes if selector_memmap is None else 0
    )
    if resident_selector_bytes > budget.max_selector_bytes:
        return ExactIndexUnavailable(
            "row selector/validation state exceeds in-memory selector budget",
            resident_selector_bytes,
            budget.max_selector_bytes,
        )
    requested_cap = (
        budget.max_unique_rows if max_unique_rows is None else max_unique_rows
    )
    if (
        isinstance(requested_cap, bool)
        or not isinstance(requested_cap, int)
        or requested_cap <= 0
    ):
        raise ValueError("max_unique_rows must be positive")
    cap = min(budget.max_unique_rows, requested_cap)
    if n_rows > np.iinfo(np.int32).max:
        return ExactIndexUnavailable(
            "row selector exceeds int32 addressable range",
            selector_bytes,
            budget.max_selector_bytes,
        )
    construction_limit = (
        budget.max_table_bytes
        if max_construction_bytes is None
        else max_construction_bytes
    )
    if construction_limit <= 0:
        return ExactIndexUnavailable(
            "no remaining exact-index construction budget",
            1,
            construction_limit,
        )
    created_path: Path | None = None
    if selector_memmap is None:
        selector: npt.NDArray[np.int32] = np.empty(n_rows, dtype=np.int32)
        storage: Literal["memory", "memmap"] = "memory"
    else:
        path = Path(selector_memmap)
        if not path.parent.is_dir():
            raise FileNotFoundError(
                f"selector memmap parent does not exist: {path.parent}"
            )
        try:
            # Claim the caller-provided filename before open_memmap's w+ mode
            # writes its header.  We only remove this sentinel on a failed
            # construction that created it ourselves.
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
        except FileExistsError as error:
            raise FileExistsError(
                f"selector memmap path already exists: {path}"
            ) from error
        created_path = path
        try:
            selector = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.int32, shape=(n_rows,)
            )
        except BaseException:
            if created_path.exists():
                created_path.unlink()
            raise
        storage = "memmap"

    index: dict[tuple[tuple[str, Any], ...], int] = {}
    unique: list[tuple[Any, ...]] = []
    scan_offset = 0
    construction_bytes = sys.getsizeof(index) + sys.getsizeof(unique)
    construction_peak_bytes = construction_bytes
    try:
        start_fingerprint = source.fingerprint()
    except BaseException:
        if created_path is not None and created_path.exists():
            created_path.unlink()
        raise
    try:
        for batch in source.scan(batch_rows):
            positions = np.asarray(batch.row_positions)
            if (
                positions.ndim != 1
                or len(positions) != len(batch.valid)
                or len(positions) > batch_rows
                or positions.dtype.kind not in "iu"
            ):
                raise RuntimeError("RowSource returned invalid row positions")
            if not np.all(batch.valid):
                raise NotImplementedError(
                    "exact discrete indexing does not support padded rows"
                )
            if np.any(positions < 0):
                raise RuntimeError("RowSource row positions must be non-negative")
            scan_stop = scan_offset + len(positions)
            if scan_stop > n_rows:
                raise RuntimeError("RowSource yielded more than its advertised rows")
            columns = []
            for name in names:
                if name not in batch.columns:
                    raise ValueError(f"Discretized column '{name}' is not in RowSource")
                column = np.asarray(batch.columns[name])
                if column.shape != positions.shape:
                    raise ValueError(
                        f"Discretized column '{name}' has incompatible batch length"
                    )
                columns.append(column)
            for local in range(len(positions)):
                values = tuple(column[local] for column in columns)
                key = _key(values)
                item = index.get(key)
                if item is None:
                    old_container_bytes = sys.getsizeof(index) + sys.getsizeof(unique)
                    # Count the nested key tuples, their tags and the value tuple.
                    # The shared scalar objects are counted once by ``seen``.
                    payload_bytes = _deep_size((key, values))
                    # CPython dictionaries and lists may allocate replacement
                    # pointer tables before releasing the old ones.  Bound that
                    # simultaneous resize before insertion: a dict replacement
                    # is charged at three additional current tables and a list
                    # replacement at two, plus fixed allocator slack.  This is
                    # deliberately above CPython's current growth factors.
                    prospective_container_bytes = (
                        4 * sys.getsizeof(index) + 3 * sys.getsizeof(unique) + 512
                    )
                    container_growth_bound = (
                        prospective_container_bytes - old_container_bytes
                    )
                    item_bytes = (
                        payload_bytes
                        + sys.getsizeof(len(unique))
                        + container_growth_bound
                    )
                    construction_peak_bytes = max(
                        construction_peak_bytes, construction_bytes + item_bytes
                    )
                    if len(unique) >= cap:
                        if created_path is not None and created_path.exists():
                            created_path.unlink()
                        return ExactIndexUnavailable(
                            "unique observed rows exceed exact row cap",
                            construction_bytes + item_bytes,
                            construction_limit,
                            len(unique) + 1,
                            cap,
                        )
                    if construction_bytes + item_bytes > construction_limit:
                        if created_path is not None and created_path.exists():
                            created_path.unlink()
                        return ExactIndexUnavailable(
                            "unique rows exceed exact construction byte budget",
                            construction_bytes + item_bytes,
                            construction_limit,
                        )
                    item = len(unique)
                    index[key] = item
                    unique.append(values)
                    container_growth = max(
                        0,
                        sys.getsizeof(index)
                        + sys.getsizeof(unique)
                        - old_container_bytes,
                    )
                    # The prospective resize allowance was a peak, not retained
                    # storage.  Record only actual container growth afterwards.
                    retained_item_bytes = payload_bytes + sys.getsizeof(item)
                    construction_bytes += retained_item_bytes + container_growth
                    construction_peak_bytes = max(
                        construction_peak_bytes, construction_bytes
                    )
                    if construction_bytes > construction_limit:
                        if created_path is not None and created_path.exists():
                            created_path.unlink()
                        return ExactIndexUnavailable(
                            "unique rows exceed exact construction byte budget",
                            construction_bytes,
                            construction_limit,
                        )
                selector[scan_offset + local] = item
            scan_offset = scan_stop
        if scan_offset != n_rows:
            raise RuntimeError("RowSource yielded fewer than its advertised rows")
        # ``row_positions`` identify the original source rows and may be
        # reordered or repeated.  Descriptor selectors use scan-local ordinals
        # so their shape is exactly the source's advertised selected row count.
        if source.fingerprint() != start_fingerprint:
            raise RuntimeError("RowSource changed during exact discrete indexing")
        if storage == "memmap":
            selector.flush()
        columns_list: list[np.ndarray] = []
        retained_column_bytes = 0
        for j in range(len(names)):
            dtype, column_bytes = _column_dtype_and_bytes(unique, j)
            # A Python list stores pointers to the already-retained discovery
            # scalars.  Charge its header, pointer array, NumPy destination and
            # the descriptor's later owned copy before creating either array.
            values_list_bytes = (
                sys.getsizeof([]) + (len(unique) + 1) * np.dtype(np.intp).itemsize + 64
            )
            materialization_peak = (
                construction_bytes
                + retained_column_bytes
                + values_list_bytes
                + 2 * column_bytes
            )
            construction_peak_bytes = max(construction_peak_bytes, materialization_peak)
            if construction_peak_bytes > construction_limit:
                if created_path is not None and created_path.exists():
                    created_path.unlink()
                return ExactIndexUnavailable(
                    "unique-value materialization exceeds construction byte budget",
                    construction_peak_bytes,
                    construction_limit,
                )
            values_list = [row[j] for row in unique]
            column = np.asarray(values_list, dtype=dtype)
            actual_column_bytes = int(column.nbytes)
            if column.dtype == object:
                actual_column_bytes += sum(sys.getsizeof(item) for item in column)
            if actual_column_bytes != column_bytes:
                raise RuntimeError(
                    "prospective unique-column storage bound was incorrect"
                )
            # ExactIndexTable takes an owned read-only copy.  At that point the
            # discovery graph, temporary list, source column and copy coexist.
            actual_materialization_peak = (
                construction_bytes
                + retained_column_bytes
                + sys.getsizeof(values_list)
                + int(column.nbytes)
                + column_bytes
            )
            construction_peak_bytes = max(
                construction_peak_bytes, actual_materialization_peak
            )
            if construction_peak_bytes > construction_limit:
                if created_path is not None and created_path.exists():
                    created_path.unlink()
                return ExactIndexUnavailable(
                    "unique-value materialization exceeds construction byte budget",
                    construction_peak_bytes,
                    construction_limit,
                )
            columns_list.append(column)
            retained_column_bytes += column_bytes
        columns = tuple(columns_list)
        ownership_peak = construction_bytes + 2 * retained_column_bytes
        construction_peak_bytes = max(construction_peak_bytes, ownership_peak)
        if construction_peak_bytes > construction_limit:
            if created_path is not None and created_path.exists():
                created_path.unlink()
            return ExactIndexUnavailable(
                "unique-value ownership copy exceeds construction byte budget",
                construction_peak_bytes,
                construction_limit,
            )
        return ExactIndexTable(
            names,
            columns,
            selector,
            n_rows,
            storage,
            construction_peak_bytes,
        )
    except BaseException:
        if created_path is not None and created_path.exists():
            created_path.unlink()
        raise
