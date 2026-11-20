"""Whole-request ``predict.bamd`` compression with a bounded pre-pass.

Pinned mgcv 1.9-3 ``predict.bamd`` (``R/bam.r``) calls
``discrete.mf(object$dinfo$gp, newdata, object$dinfo$pmf.names)`` with its
default ``m=NULL`` before it evaluates any fitted smooth metadata.  For each
jointly discretized group (a smooth, a tensor margin, a by variable or a
parametric covariate, grouped exactly as during fitting) ``compress.df``
keeps the request's own unique covariate rows when there are at most
``mm*mf`` of them, and otherwise rounds every metric variable of the group to
an evenly spaced grid over the *request's* range.  ``mm`` is the default grid
resolution (1000, 100 or 25 points per metric variable for one, two or more
jointly discretized variables) raised to the number of metric variables and
``mf`` is the product of the distinct factor values present in the request.

Predictions depend only on each row's compressed covariates, never on mgcv's
row shuffle or padding.  This module therefore freezes, per group, only the
rounding decision, its per-variable range grid and, for an unrounded joint
group, the rare rows that ``uniquecombs`` merges with an earlier row whose
15-significant-digit token is identical (mgcv keeps the first occurrence).
The decision needs the whole request, so it is made by a bounded pre-pass
that retains at most ``min(mm*mf, max_unique_rows) + 1`` distinct row keys
per group and no row selectors.  Afterwards any batch of the same request is
compressed elementwise, which makes predictions invariant to batch
boundaries.  No joint grid across a group's variables is ever formed.

The policy changes only metric discretization, never factor semantics.
Factor-only groups (random effects, factor ``by`` variables, parametric
factors) can never be rounded, so their values pass through unchanged to the
frozen evaluators, which apply exactly the default-path semantics: a missing
or unseen random-effect level contributes zero (100% weight on the prior, an
intended divergence from mgcv's ``NA``), a missing or unseen ``by`` level
gives ``NaN`` with a warning, and a parametric factor gives ``NaN`` for a
missing value and rejects unseen levels.
In a group mixing factor and metric variables, rows whose factor value is
missing or unseen in the fitted levels are excluded from counting, from the
request range and from rounding; only a multi-variable random effect forms
such a group, and its contribution for those rows is zero.

This is Phase-1/3 NumPy code; it never imports JAX.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from jaxgam.data.continuous_discretize import (
    _points_per_metric,
    _r_number_text,
    range_grid,
    snap_to_range_grid,
)
from jaxgam.data.discretize import DiscretizationBudget
from jaxgam.data.source import RowSource

Column = npt.NDArray[Any] | pd.Series
_IN_MEMORY_CHUNK_ROWS = 65_536
# Conservative resident bytes for one retained distinct-row key: a Python
# float, or a formatted ``uniquecombs`` token plus its first-occurrence row
# tuple, together with its hash-table slot share.
_SCALAR_KEY_BYTES = 128
_TOKEN_KEY_BYTES = 256
_TOKEN_COLUMN_BYTES = 96


@dataclass(frozen=True)
class RequestGroupSpec:
    """One fitted discretization group, without any training values.

    ``factor_levels`` are the levels observed during fitting (mgcv's
    model-frame levels after ``drop.unused.levels``).  Only a group mixing
    factor and metric variables needs them, to recognize the rows it excludes.
    """

    names: tuple[str, ...]
    factor_names: tuple[str, ...] = ()
    factor_levels: tuple[tuple[Any, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("request group needs distinct variable names")
        if not set(self.factor_names) <= set(self.names):
            raise ValueError("request factor names must belong to the group")
        if self.factor_levels and len(self.factor_levels) != len(self.factor_names):
            raise ValueError("request factor levels do not match factor names")
        if any(not levels for levels in self.factor_levels):
            raise ValueError("request factor columns need their fitted levels")
        if self.factor_names and self.metric_names and not self.factor_levels:
            raise ValueError("a mixed factor/metric request group needs fitted levels")

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.names if name not in self.factor_names)

    @property
    def points_per_metric(self) -> int:
        """``compress.df`` grid points per metric variable for ``m=NULL``."""
        return _points_per_metric(len(self.names), None)


@dataclass(frozen=True)
class FrozenRequestGroup:
    """The frozen ``compress.df`` decision for one group of one request."""

    names: tuple[str, ...]
    metric_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    points_per_metric: int
    unique_threshold: int | None
    n_unique: int | None
    rounded: bool
    metric_ranges: tuple[tuple[float, float], ...]
    # Exact row -> first-occurrence row with the same ``uniquecombs`` token,
    # in ``names`` order; only for unrounded joint groups, usually empty.
    merged_rows: tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...] = ()
    # Fitted levels of a mixed group's factor columns, to exclude rows.
    factor_levels: tuple[tuple[Any, ...], ...] = ()

    def __post_init__(self) -> None:
        if len(self.metric_ranges) != len(self.metric_names):
            raise ValueError("frozen request ranges do not match metric variables")
        if self.rounded and self.unique_threshold is None:
            raise ValueError("a rounded request group needs its unique threshold")
        if self.merged_rows and (self.rounded or len(self.names) < 2):
            raise ValueError("only unrounded joint groups merge token-equal rows")
        mixed = bool(self.factor_names) and bool(self.metric_names)
        if len(self.factor_levels) != (len(self.factor_names) if mixed else 0):
            raise ValueError("only mixed groups carry their fitted factor levels")

    def included(self, columns: Mapping[str, Column]) -> npt.NDArray[np.bool_] | None:
        """Rows whose factor values are all fitted levels (mixed groups only)."""
        if not self.factor_levels:
            return None
        mask: npt.NDArray[np.bool_] | None = None
        for name, levels in zip(self.factor_names, self.factor_levels, strict=True):
            codes = _factor_codes(_factor_values(columns, name), _level_codes(levels))
            mask = codes > 0 if mask is None else mask & (codes > 0)
        return mask

    def merge(self, columns: dict[str, Column]) -> None:
        """Replace token-equal rows by their first occurrence, in place.

        Excluded rows (missing or unseen factor values) were never observed,
        so no merge entry matches them.
        """
        if not self.merged_rows:
            return
        mapping = dict(self.merged_rows)
        metric = {name: _metric_values(columns, name) for name in self.metric_names}
        values = [
            metric[name] if name in metric else _factor_values(columns, name)
            for name in self.names
        ]
        replaced = {name: metric[name].copy() for name in self.metric_names}
        for row in range(len(values[0])):
            representative = mapping.get(
                tuple(_row_value(column[row]) for column in values)
            )
            if representative is None:
                continue
            for number, name in enumerate(self.names):
                if name in replaced:
                    replaced[name][row] = representative[number]
        columns.update(replaced)

    def grid(self, name: str) -> npt.NDArray[np.float64]:
        """The request range grid of one metric variable."""
        try:
            low, high = self.metric_ranges[self.metric_names.index(name)]
        except ValueError as error:
            raise KeyError(
                f"'{name}' is not a metric variable of this group"
            ) from error
        return range_grid(low, high, self.points_per_metric)


@dataclass(frozen=True)
class FrozenRequestGrid:
    """Request-level compression frozen before any prediction batch."""

    groups: tuple[FrozenRequestGroup, ...]
    n_rows: int
    construction_peak_bytes: int
    source_fingerprint: str | None = None

    @property
    def rounded_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for group in self.groups
            if group.rounded
            for name in group.metric_names
        )

    def apply(self, data: Mapping[str, Column] | pd.DataFrame) -> dict[str, Column]:
        """Compress one batch of the frozen request by an elementwise snap.

        Columns of unrounded groups keep the request's unique rows (after
        ``uniquecombs``' first-occurrence merge of token-equal joint rows).
        A value outside a rounded group's frozen request range proves the
        batch is not part of this request.
        """
        columns = _as_columns(data)
        for group in self.groups:
            if not group.rounded:
                group.merge(columns)
                continue
            included = group.included(columns)
            for name, (low, high) in zip(
                group.metric_names, group.metric_ranges, strict=True
            ):
                values = _metric_values(columns, name)
                rounded = values if included is None else values[included]
                if rounded.size and (np.min(rounded) < low or np.max(rounded) > high):
                    raise ValueError(
                        f"Variable '{name}' lies outside the frozen prediction "
                        "request range; request compression must be frozen over "
                        "the batch's own request."
                    )
                snapped = snap_to_range_grid(rounded, group.grid(name))
                if included is None:
                    columns[name] = snapped
                else:
                    # Excluded rows keep their values: their term is zero.
                    values = values.copy()
                    values[included] = snapped
                    columns[name] = values
        return columns


def _row_value(value: Any) -> Any:
    """Hashable exact key component: Python float for numbers, else raw."""
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _as_columns(data: Mapping[str, Column] | pd.DataFrame) -> dict[str, Column]:
    if isinstance(data, pd.DataFrame):
        return {name: data[name].reset_index(drop=True) for name in data.columns}
    return dict(data)


def _column(columns: Mapping[str, Column], name: str) -> Column:
    try:
        return columns[name]
    except KeyError as error:
        raise ValueError(
            f"Variable '{name}' is required for prediction request compression "
            "but is absent from newdata."
        ) from error


def _metric_values(columns: Mapping[str, Column], name: str) -> npt.NDArray[np.float64]:
    raw = _column(columns, name)
    if isinstance(raw, pd.Series) and isinstance(raw.dtype, pd.CategoricalDtype):
        raise TypeError(f"Metric variable '{name}' must be numeric, not categorical.")
    array = np.asarray(raw)
    if array.ndim != 1:
        raise ValueError(f"Metric variable '{name}' must be one-dimensional.")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"Metric variable '{name}' must be real numeric data.")
    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Metric variable '{name}' contains non-finite values; request "
            "compression cannot form its range grid."
        )
    return values


def _factor_values(columns: Mapping[str, Column], name: str) -> npt.NDArray[Any]:
    raw = _column(columns, name)
    values = np.asarray(raw, dtype=object)
    if values.ndim != 1:
        raise ValueError(f"Factor variable '{name}' must be one-dimensional.")
    return values


def _level_codes(levels: tuple[Any, ...]) -> dict[Any, int]:
    """Raw-value -> one-based code map in fitted order, as ``as.numeric``.

    NumPy and Python scalars of equal value share keys.
    """
    return {level: code + 1 for code, level in enumerate(levels)}


def _factor_codes(
    values: npt.NDArray[Any], known: dict[Any, int]
) -> npt.NDArray[np.intp]:
    """One-based fitted codes, with 0 for missing or unseen values."""
    missing = np.asarray(pd.isna(values), dtype=bool)
    return np.asarray(
        [
            0 if absent else known.get(value, 0)
            for value, absent in zip(values, missing, strict=True)
        ],
        dtype=np.intp,
    )


class _GroupCounter:
    """Bounded distinct-row discovery for one request group."""

    def __init__(self, spec: RequestGroupSpec, budget: DiscretizationBudget) -> None:
        self.spec = spec
        self.budget = budget
        self.metric_names = spec.metric_names
        self.lows = dict.fromkeys(self.metric_names, np.inf)
        self.highs = dict.fromkeys(self.metric_names, -np.inf)
        self.factor_codes: dict[str, set[int]] = {
            name: set() for name in spec.factor_names
        }
        # Only mixed groups carry (and need) their fitted levels.
        self.levels = dict(zip(spec.factor_names, spec.factor_levels, strict=False))
        self.level_codes = {
            name: _level_codes(levels) for name, levels in self.levels.items()
        }
        self.keys: set[Any] | None = None
        # Joint groups: uniquecombs token -> first-occurrence exact row, and
        # exact rows that mgcv merges into an earlier token-equal row.
        self.first: dict[str, tuple[Any, ...]] | None = None
        self.merged: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        self.cap: int | None = None
        self.threshold: int | None = None
        self.exceeded = False
        self.peak_bytes = 0
        self.points = spec.points_per_metric

    @property
    def counts_rows(self) -> bool:
        return bool(self.metric_names)

    @property
    def mixed(self) -> bool:
        return bool(self.metric_names) and bool(self.spec.factor_names)

    @property
    def joint(self) -> bool:
        return len(self.spec.names) > 1

    def _key_bytes(self) -> int:
        if not self.joint:
            return _SCALAR_KEY_BYTES
        return _TOKEN_KEY_BYTES + _TOKEN_COLUMN_BYTES * len(self.spec.names)

    def start_counting(self) -> None:
        """Fix the unique threshold and reserve the bounded key state."""
        factor_product = math.prod(len(codes) for codes in self.factor_codes.values())
        self.threshold = factor_product * self.points ** len(self.metric_names)
        self.cap = min(self.threshold, self.budget.max_unique_rows)
        required = (self.cap + 1) * self._key_bytes() + sys.getsizeof({})
        if required > self.budget.max_table_bytes:
            raise MemoryError(
                "Prediction request compression for group "
                f"{self.spec.names} needs {required} bytes of distinct-row "
                f"state, exceeding max_table_bytes={self.budget.max_table_bytes}."
            )
        self.peak_bytes = max(self.peak_bytes, required)
        if self.joint:
            self.first = {}
        else:
            self.keys = set()

    def _encoded_factors(
        self, columns: Mapping[str, Column]
    ) -> tuple[dict[str, npt.NDArray[np.intp]], npt.NDArray[np.bool_] | None]:
        """Mixed groups: fitted codes per factor and the included-row mask."""
        codes = {
            name: _factor_codes(_factor_values(columns, name), self.level_codes[name])
            for name in self.spec.factor_names
        }
        mask = None
        for values in codes.values():
            mask = values > 0 if mask is None else mask & (values > 0)
        return codes, mask

    def observe_ranges(self, columns: Mapping[str, Column]) -> None:
        """Track request ranges of the rows this group compresses.

        Factor-only groups pass through untouched.  A mixed group excludes
        rows whose factor value is missing or unseen in the fitted levels.
        """
        if not self.counts_rows:
            return
        included = None
        if self.mixed:
            codes, included = self._encoded_factors(columns)
            for name, values in codes.items():
                self.factor_codes[name].update(
                    int(code) for code in np.unique(values[values > 0])
                )
        for name in self.metric_names:
            values = _metric_values(columns, name)
            if included is not None:
                values = values[included]
            if values.size:
                self.lows[name] = min(self.lows[name], float(np.min(values)))
                self.highs[name] = max(self.highs[name], float(np.max(values)))

    def observe_keys(self, columns: Mapping[str, Column]) -> None:
        if self.exceeded:
            return
        if not self.joint:
            assert self.keys is not None
            assert self.cap is not None
            for value in np.unique(_metric_values(columns, self.spec.names[0])):
                self.keys.add(float(value))
                if len(self.keys) > self.cap:
                    self._exceed()
                    return
            return
        self._observe_tokens(columns)

    def _exceed(self) -> None:
        self.exceeded = True
        self.keys = None
        self.first = None
        self.merged = {}

    def _observe_tokens(self, columns: Mapping[str, Column]) -> None:
        """``uniquecombs`` tokens, visiting distinct rows in request order."""
        assert self.first is not None
        assert self.cap is not None
        codes, included = self._encoded_factors(columns) if self.mixed else ({}, None)
        encoded = []
        factor_columns = set()
        for number, name in enumerate(self.spec.names):
            if name in codes:
                factor_columns.add(number)
                encoded.append(np.asarray(codes[name], dtype=np.float64))
            else:
                encoded.append(_metric_values(columns, name))
        stacked = np.column_stack(encoded)
        if included is not None:
            stacked = stacked[included]
        if not len(stacked):
            return
        rows, first_index = np.unique(stacked, axis=0, return_index=True)
        for row in rows[np.argsort(first_index, kind="stable")]:
            token = "*".join(
                str(int(value)) if number in factor_columns else _r_number_text(value)
                for number, value in enumerate(row)
            )
            exact = tuple(
                self.levels[name][int(value) - 1]
                if number in factor_columns
                else float(value)
                for number, (name, value) in enumerate(
                    zip(self.spec.names, row, strict=True)
                )
            )
            representative = self.first.get(token)
            if representative is None:
                self.first[token] = exact
                if len(self.first) > self.cap:
                    self._exceed()
                    return
            elif representative != exact:
                self.merged[exact] = representative
                if len(self.merged) > self.budget.max_unique_rows:
                    raise MemoryError(
                        "Prediction request compression for group "
                        f"{self.spec.names} found more than max_unique_rows="
                        f"{self.budget.max_unique_rows} token-merged rows."
                    )
                stored = (len(self.first) + len(self.merged)) * self._key_bytes()
                if stored > self.budget.max_table_bytes:
                    raise MemoryError(
                        "Prediction request compression for group "
                        f"{self.spec.names} needs {stored} bytes of token "
                        "state, exceeding max_table_bytes="
                        f"{self.budget.max_table_bytes}."
                    )
                self.peak_bytes = max(self.peak_bytes, stored)

    def _distinct_count(self) -> int:
        if self.joint:
            return len(self.first or ())
        return len(self.keys or ())

    def freeze(self) -> FrozenRequestGroup:
        ranges = tuple(
            (self.lows[name], self.highs[name]) for name in self.metric_names
        )
        if not self.counts_rows:
            # Factor-only groups can never exceed their own distinct-level
            # product, so compress.df keeps them exact.
            return FrozenRequestGroup(
                self.spec.names,
                (),
                self.spec.factor_names,
                self.points,
                None,
                None,
                False,
                (),
            )
        assert self.threshold is not None
        assert self.cap is not None
        if self.exceeded and self.cap < self.threshold:
            raise MemoryError(
                "Prediction request compression for group "
                f"{self.spec.names} found more than max_unique_rows="
                f"{self.budget.max_unique_rows} distinct rows before reaching "
                f"its mgcv rounding threshold {self.threshold}."
            )
        rounded = self.exceeded
        if rounded:
            constant = tuple(
                name
                for name, (low, high) in zip(self.metric_names, ranges, strict=True)
                if low == high
            )
            if constant:
                raise ValueError(
                    "pinned compress.df produces an undefined range grid for "
                    f"constant metric variable(s) {constant} when request "
                    "rounding is required"
                )
            if self.joint:
                self._require_distinct_grid_tokens(ranges)
        return FrozenRequestGroup(
            self.spec.names,
            self.metric_names,
            self.spec.factor_names,
            self.points,
            self.threshold,
            None if rounded else self._distinct_count(),
            rounded,
            ranges,
            () if rounded else tuple(self.merged.items()),
            self.spec.factor_levels if self.mixed else (),
        )

    def _require_distinct_grid_tokens(
        self, ranges: tuple[tuple[float, float], ...]
    ) -> None:
        """Refuse grids whose rounded rows ``uniquecombs`` would merge.

        After rounding, mgcv tokenizes the grid values again.  Distinct grid
        rows can share a token only when one variable's grid points coincide
        to 15 significant digits, which this policy does not emulate.
        """
        for name, (low, high) in zip(self.metric_names, ranges, strict=True):
            grid = range_grid(low, high, self.points)
            if len({_r_number_text(value) for value in grid}) != len(grid):
                raise NotImplementedError(
                    f"The request range grid of '{name}' has points equal to 15 "
                    "significant digits; mgcv_predict_bamd does not emulate "
                    "uniquecombs merging of rounded joint rows."
                )


@dataclass(frozen=True)
class RequestCompressionSpec:
    """Fitted groups and budget for ``predict.bamd`` request compression."""

    groups: tuple[RequestGroupSpec, ...]
    budget: DiscretizationBudget

    def __post_init__(self) -> None:
        if not isinstance(self.budget, DiscretizationBudget):
            raise TypeError("request compression budget must be a DiscretizationBudget")
        names = [name for group in self.groups for name in group.names]
        if len(names) != len(set(names)):
            raise ValueError("request groups must not share variables")

    def _counters(self, batch_rows: int) -> list[_GroupCounter]:
        widest = max((len(group.names) for group in self.groups), default=0)
        # Per batch: float copies, the stacked distinct-row matrix and its
        # sorted copy, plus the unique output, for the widest group.
        workspace = batch_rows * max(widest, 1) * 4 * np.dtype(np.float64).itemsize
        if workspace > self.budget.max_batch_bytes:
            raise MemoryError(
                f"Prediction request compression batch workspace {workspace} "
                f"bytes exceeds max_batch_bytes={self.budget.max_batch_bytes}."
            )
        return [_GroupCounter(group, self.budget) for group in self.groups]

    def _freeze(
        self,
        scan: Any,
        n_rows: int,
        counters: list[_GroupCounter],
        fingerprint: str | None,
    ) -> FrozenRequestGrid:
        # Pass 1: request ranges, validated factor values, and the distinct
        # rows of factor-free metric groups (their threshold is known).
        for counter in counters:
            if counter.counts_rows and not counter.mixed:
                counter.start_counting()
        seen = 0
        for columns in scan():
            for counter in counters:
                counter.observe_ranges(columns)
                if counter.counts_rows and not counter.mixed:
                    counter.observe_keys(columns)
            seen += _batch_length(columns)
        if seen != n_rows:
            raise RuntimeError("prediction request yielded an inconsistent row count")
        # Pass 2 only when a mixed group needs its request factor product.
        mixed = [counter for counter in counters if counter.mixed]
        if mixed:
            for counter in mixed:
                counter.start_counting()
            for columns in scan():
                for counter in mixed:
                    counter.observe_keys(columns)
        groups = tuple(counter.freeze() for counter in counters)
        peak = sum(counter.peak_bytes for counter in counters)
        return FrozenRequestGrid(groups, n_rows, peak, fingerprint)

    def freeze(self, data: Mapping[str, Column] | pd.DataFrame) -> FrozenRequestGrid:
        """Freeze compression over one in-memory prediction request."""
        columns = _as_columns(data)
        if not columns:
            raise ValueError("newdata is empty — no variables found.")
        needed = tuple(name for group in self.groups for name in group.names)
        n_rows = _batch_length({name: _column(columns, name) for name in needed})
        if n_rows == 0:
            raise ValueError("prediction request has no rows")
        chunk = min(n_rows, _IN_MEMORY_CHUNK_ROWS)
        counters = self._counters(chunk)

        def scan() -> Iterator[dict[str, Column]]:
            for start in range(0, n_rows, chunk):
                stop = min(start + chunk, n_rows)
                yield {
                    name: _slice(_column(columns, name), start, stop) for name in needed
                }

        return self._freeze(scan, n_rows, counters, None)

    def freeze_source(self, source: RowSource, batch_rows: int) -> FrozenRequestGrid:
        """Freeze compression over a replayable request by bounded scans."""
        if source.n_rows <= 0:
            raise ValueError("prediction request has no rows")
        counters = self._counters(batch_rows)
        fingerprint = source.fingerprint()

        def scan() -> Iterator[Mapping[str, Column]]:
            for batch in source.scan(batch_rows):
                if not np.all(batch.valid):
                    raise NotImplementedError(
                        "predict_iter does not support padded batches."
                    )
                yield batch.columns

        frozen = self._freeze(scan, source.n_rows, counters, fingerprint)
        if source.fingerprint() != fingerprint:
            raise RuntimeError(
                "RowSource changed during prediction request compression"
            )
        return frozen


def _batch_length(columns: Mapping[str, Column]) -> int:
    lengths = {len(value) for value in columns.values()}
    if len(lengths) > 1:
        raise ValueError("prediction request columns must share one length")
    return next(iter(lengths), 0)


def _slice(value: Column, start: int, stop: int) -> Column:
    if isinstance(value, pd.Series):
        return value.iloc[start:stop]
    return np.asarray(value)[start:stop]
