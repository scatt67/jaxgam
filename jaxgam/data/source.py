"""Restartable, bounded-row sources for prepared-model setup."""

from __future__ import annotations

import hashlib
import numbers
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

Column = npt.NDArray[Any] | pd.Series
_CHUNK_ROWS = 65_536


@dataclass(frozen=True)
class RowBatch:
    """One positional source batch; ``valid`` reserves future padding."""

    columns: Mapping[str, Column]
    y: npt.NDArray[np.floating] | None
    weight: npt.NDArray[np.floating]
    offset: npt.NDArray[np.floating]
    valid: npt.NDArray[np.bool_]
    row_positions: npt.NDArray[np.intp]

    def __post_init__(self) -> None:
        n_rows = len(self.row_positions)
        for name, value in self.columns.items():
            if len(value) != n_rows:
                raise ValueError(
                    f"Column '{name}' has {len(value)} rows but batch has {n_rows}."
                )
        for name, value in (
            ("weight", self.weight),
            ("offset", self.offset),
            ("valid", self.valid),
        ):
            if np.asarray(value).shape != (n_rows,):
                raise ValueError(f"{name} must have shape ({n_rows},).")
        if self.y is not None and np.asarray(self.y).shape != (n_rows,):
            raise ValueError(f"y must have shape ({n_rows},).")


@runtime_checkable
class RowSource(Protocol):
    """A source that can replay batches in stable positional order."""

    @property
    def n_rows(self) -> int: ...

    def scan(self, batch_rows: int) -> Iterator[RowBatch]: ...

    def fingerprint(self) -> str: ...


def _array(value: npt.ArrayLike | pd.Series, name: str) -> npt.NDArray[Any]:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return result


def _length(value: npt.ArrayLike | pd.Series, name: str) -> int:
    """Validate one-dimensional shape without decoding categorical Series."""
    if isinstance(value, pd.Series):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        return len(value)
    return len(_array(value, name))


def _slice(value: Column, positions: npt.NDArray[np.intp]) -> Column:
    if isinstance(value, pd.Series):
        return value.iloc[positions].reset_index(drop=True)
    return _array(value, "source column")[positions]


def _schema_token(name: str, value: Column) -> bytes:
    if isinstance(value, pd.Series) and isinstance(value.dtype, pd.CategoricalDtype):
        dtype = value.dtype
        return repr(
            (name, "categorical", tuple(map(repr, dtype.categories)), dtype.ordered)
        ).encode()
    array = _array(value, name)
    return repr((name, str(array.dtype), tuple(array.shape))).encode()


def _hash_value(digest: Any, value: Column | npt.NDArray[Any]) -> None:
    """Hash mutable input in bounded chunks, including categorical codes."""
    for start in range(0, len(value), _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, len(value))
        if isinstance(value, pd.Series):
            piece = value.iloc[start:stop]
            if isinstance(piece.dtype, pd.CategoricalDtype):
                digest.update(
                    np.ascontiguousarray(piece.cat.codes.to_numpy(np.int64)).view(
                        np.uint8
                    )
                )
                continue
            values = piece.to_numpy(copy=False)
        else:
            values = _array(value, "source value")[start:stop]
        if values.dtype == object:
            digest.update(repr(tuple(map(repr, values.tolist()))).encode())
        else:
            digest.update(np.ascontiguousarray(values).view(np.uint8))


def _validated_vector(
    value: npt.ArrayLike | None, n_rows: int, name: str, *, non_negative: bool = False
) -> npt.NDArray[np.floating] | None:
    if value is None:
        return None
    result = _array(value, name)
    if len(result) != n_rows:
        raise ValueError(f"{name} has {len(result)} rows but source has {n_rows}.")
    for start in range(0, n_rows, _CHUNK_ROWS):
        piece = np.asarray(result[start : start + _CHUNK_ROWS], dtype=float)
        if not np.all(np.isfinite(piece)):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf).")
        if non_negative and np.any(piece < 0):
            raise ValueError(f"{name} must be non-negative.")
    return result


class ArrayRowSource:
    """Replayable source backed by aligned arrays, Series, or memmaps.

    Missing weights/offsets remain scalar defaults and unfiltered positions
    remain implicit, so construction adds no observation-sized arrays.
    """

    def __init__(
        self,
        columns: Mapping[str, Column],
        *,
        y: npt.ArrayLike | None = None,
        weights: npt.ArrayLike | None = None,
        offset: npt.ArrayLike | None = None,
        row_selection: npt.ArrayLike | None = None,
        version: str | None = None,
    ) -> None:
        copied = dict(columns)
        lengths = {_length(value, name) for name, value in copied.items()}
        if len(lengths) > 1:
            raise ValueError("All source columns must have the same length.")
        if not copied and y is None:
            raise ValueError("RowSource requires a response or at least one column.")
        n_rows = next(iter(lengths), _length(y, "y") if y is not None else 0)
        self._y = _validated_vector(y, n_rows, "y")
        if self._y is not None and lengths and len(self._y) != n_rows:
            raise ValueError("y and source columns must have the same length.")
        self._source_n_rows = n_rows
        self._columns: Mapping[str, Column] = MappingProxyType(copied)
        self._weights = _validated_vector(weights, n_rows, "weights", non_negative=True)
        self._offset = _validated_vector(offset, n_rows, "offset")
        self._positions = self._validate_selection(row_selection)
        if self._weights is not None and self._selected_weight_sum() <= 0:
            raise ValueError(
                "weights sum to zero: at least one selected row must have "
                "positive weight."
            )
        self._version = version

    def _validate_selection(
        self, selection: npt.ArrayLike | None
    ) -> npt.NDArray[np.intp] | None:
        if selection is None:
            return None
        raw = _array(selection, "row_selection")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("row_selection must contain integer source positions.")
        positions = raw.astype(np.intp, copy=False)
        if len(positions) == 0:
            raise ValueError("row_selection cannot be empty.")
        if np.any(positions < 0) or np.any(positions >= self._source_n_rows):
            raise ValueError("row_selection contains a position outside the source.")
        return positions

    def _selected_weight_sum(self) -> float:
        assert self._weights is not None
        total = 0.0
        if self._positions is None:
            for start in range(0, self._source_n_rows, _CHUNK_ROWS):
                total += float(np.sum(self._weights[start : start + _CHUNK_ROWS]))
        else:
            for start in range(0, len(self._positions), _CHUNK_ROWS):
                total += float(
                    np.sum(self._weights[self._positions[start : start + _CHUNK_ROWS]])
                )
        return total

    @property
    def n_rows(self) -> int:
        return self._source_n_rows if self._positions is None else len(self._positions)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self._columns)

    @property
    def has_explicit_offset(self) -> bool:
        """Whether scan offsets came from caller data rather than defaults."""
        return self._offset is not None

    def scan(self, batch_rows: int) -> Iterator[RowBatch]:
        """Yield fresh batches with only batch-sized output allocations."""
        if (
            not isinstance(batch_rows, numbers.Integral)
            or isinstance(batch_rows, bool)
            or batch_rows <= 0
        ):
            raise ValueError("batch_rows must be a positive integer.")
        batch_rows = int(batch_rows)
        for start in range(0, self.n_rows, batch_rows):
            stop = min(start + batch_rows, self.n_rows)
            positions = (
                np.arange(start, stop, dtype=np.intp)
                if self._positions is None
                else self._positions[start:stop]
            )
            columns = {
                name: _slice(value, positions) for name, value in self._columns.items()
            }
            y = None if self._y is None else np.asarray(self._y[positions], dtype=float)
            weight = (
                np.ones(len(positions), dtype=float)
                if self._weights is None
                else np.asarray(self._weights[positions], dtype=float)
            )
            offset = (
                np.zeros(len(positions), dtype=float)
                if self._offset is None
                else np.asarray(self._offset[positions], dtype=float)
            )
            yield RowBatch(
                MappingProxyType(columns),
                y,
                weight,
                offset,
                np.ones(len(positions), bool),
                positions.copy(),
            )

    def fingerprint(self) -> str:
        """Content/schema/selection fingerprint, recomputed to reveal mutation."""
        digest = hashlib.sha256(b"jaxgam-row-source-v2\0")
        for name, value in self._columns.items():
            digest.update(_schema_token(name, value))
            _hash_value(digest, value)
        for name, value in (
            ("y", self._y),
            ("weights", self._weights),
            ("offset", self._offset),
            ("row_selection", self._positions),
        ):
            digest.update(name.encode())
            if value is not None:
                _hash_value(digest, value)
        digest.update((self._version or "content-addressed").encode())
        return digest.hexdigest()


class DataFrameRowSource(ArrayRowSource):
    """DataFrame source preserving categorical and ordered dtypes."""

    def __init__(
        self, data: pd.DataFrame, *, response: str | None = None, **kwargs: Any
    ) -> None:
        if response is not None and response not in data:
            raise ValueError(f"Response variable '{response}' not found in source.")
        super().__init__(
            {name: data[name] for name in data if name != response},
            y=None if response is None else data[response],
            **kwargs,
        )


class MemmapRowSource(ArrayRowSource):
    """Replayable source loaded lazily from one-dimensional ``.npy`` memmaps."""

    @classmethod
    def from_npy_files(
        cls,
        columns: Mapping[str, str | Path],
        *,
        y: str | Path | None = None,
        weights: str | Path | None = None,
        offset: str | Path | None = None,
        **kwargs: Any,
    ) -> MemmapRowSource:
        return cls(
            {name: np.load(path, mmap_mode="r") for name, path in columns.items()},
            y=None if y is None else np.load(y, mmap_mode="r"),
            weights=None if weights is None else np.load(weights, mmap_mode="r"),
            offset=None if offset is None else np.load(offset, mmap_mode="r"),
            **kwargs,
        )
