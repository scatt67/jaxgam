"""Bounded mgcv-style range-grid compression for continuous covariates.

This module implements only the representation policy of
``mgcv:::compress.df``.  Basis setup and working-model smoothing are separate
PR11 stages: callers must use the returned compact frame to construct and then
freeze the fitted basis rather than reuse an ordinary dense setup implicitly.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from jaxgam.data.discretize import (
    DiscretizationBudget,
    ExactIndexTable,
    ExactIndexUnavailable,
    _column_dtype_and_bytes,
    _deep_size,
    _key,
)
from jaxgam.data.source import RowBatch, RowSource, declared_row_vectors


@dataclass(frozen=True)
class RangeGridIndex:
    """Compact representatives and selectors from the range-grid policy."""

    index: ExactIndexTable
    factor_names: tuple[str, ...]
    factor_levels: tuple[tuple[Any, ...], ...]
    ordered_factors: frozenset[str]
    metric_names: tuple[str, ...]
    requested_m: int | None
    points_per_metric: int
    metric_ranges: tuple[tuple[float, float], ...]
    rounded: bool
    shuffle_seed: int
    rng_policy: Literal["r_4_5_mt_rejection"] = "r_4_5_mt_rejection"
    construction_peak_bytes: int = 0
    # Observed distinct values per factor column: ``compress.df``'s ``mf``.
    # Declared categorical levels can include unobserved categories, so the
    # grid threshold cannot be recovered from ``factor_levels``.
    factor_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if set(self.factor_names).intersection(self.metric_names):
            raise ValueError("factor and metric grid names must be disjoint")
        if self.factor_counts and len(self.factor_counts) != len(self.factor_names):
            raise ValueError("factor counts do not match factor columns")
        all_names = (*self.factor_names, *self.metric_names)
        if not all_names:
            raise ValueError("range-grid index requires at least one column")
        if {*all_names} != set(self.index.names):
            raise ValueError("range-grid metadata does not cover its indexed columns")
        if len(self.metric_names) != len(self.metric_ranges):
            raise ValueError("metric ranges do not match metric columns")
        if len(self.factor_names) != len(self.factor_levels):
            raise ValueError("factor levels do not match factor columns")
        if not self.ordered_factors <= set(self.factor_names):
            raise ValueError("ordered factor metadata references a non-factor column")
        if self.points_per_metric < 2:
            raise ValueError("range grid needs at least two points per metric")
        if self.construction_peak_bytes < self.index.construction_peak_bytes:
            raise ValueError("range-grid peak cannot be below exact-index peak")

    @property
    def names(self) -> tuple[str, ...]:
        return self.index.names

    @property
    def values(self) -> tuple[npt.NDArray[Any], ...]:
        return self.index.values

    @property
    def selector(self) -> npt.NDArray[np.int32]:
        return self.index.selector

    @property
    def n_rows(self) -> int:
        return self.index.n_rows

    @property
    def n_unique(self) -> int:
        return self.index.n_unique

    @property
    def unique_threshold(self) -> int:
        """``compress.df``'s ``mm*mf``: rounding happens above this count."""
        counts = self.factor_counts or tuple(
            len(levels) for levels in self.factor_levels
        )
        return math.prod(counts) * self.points_per_metric ** len(self.metric_names)


@dataclass(frozen=True)
class _ColumnAnalysis:
    factor_counts: tuple[int, ...]
    factor_levels: tuple[tuple[Any, ...], ...]
    ordered_factors: frozenset[str]
    metric_ranges: tuple[tuple[float, float], ...]
    peak_bytes: int
    source_fingerprint: str


def _points_per_metric(n_columns: int, m: int | None) -> int:
    if m is None:
        return 1000 if n_columns == 1 else 100 if n_columns == 2 else 25
    if isinstance(m, bool) or not isinstance(m, int) or m < 2:
        raise ValueError("m must be an integer of at least 2")
    if n_columns == 1:
        return m
    # R's round uses IEC 60559 ties-to-even, as does np.rint.
    return int(np.rint(math.pow(m, 1.0 / n_columns))) + 1


def _validate_batch_positions(
    batch: RowBatch, batch_rows: int, scan_offset: int, n_rows: int
) -> int:
    positions = np.asarray(batch.row_positions)
    if (
        positions.ndim != 1
        or positions.dtype.kind not in "iu"
        or len(positions) != len(batch.valid)
        or len(positions) > batch_rows
    ):
        raise RuntimeError("RowSource returned invalid row positions")
    if np.any(positions < 0):
        raise RuntimeError("RowSource row positions must be non-negative")
    if not np.all(batch.valid):
        raise NotImplementedError(
            "continuous discrete indexing does not support padded source rows"
        )
    scan_stop = scan_offset + len(positions)
    if scan_stop > n_rows:
        raise RuntimeError("RowSource yielded more than its advertised rows")
    return scan_stop


def _non_numeric_metric_message(name: str, column: Any) -> str:
    """Name the column and dtype, and how a factor is declared on this route.

    The column is not assumed to be a factor: it may equally be numeric data
    stored with a non-numeric dtype.
    """
    dtype = getattr(column, "dtype", None)
    return (
        f"Metric column '{name}' must be real numeric data, got dtype {dtype}. "
        "Variables not listed in factor_names are discretized as metric. If "
        f"'{name}' is a factor (for example the grouping variable of "
        f"s({name}, bs='re')), declare it with "
        f"ContinuousDiscretizationControl(factor_names=('{name}',)); otherwise "
        "supply it as real numbers."
    )


def _analyze_columns(
    source: RowSource,
    names: tuple[str, ...],
    factor_names: tuple[str, ...],
    metric_names: tuple[str, ...],
    budget: DiscretizationBudget,
    batch_rows: int,
) -> _ColumnAnalysis | ExactIndexUnavailable:
    """Find factor cardinalities and metric ranges without retaining rows."""
    factor_maps: dict[str, dict[tuple[tuple[str, Any], ...], Any]] = {
        name: {} for name in factor_names
    }
    declared_levels: dict[str, tuple[Any, ...] | None] = dict.fromkeys(factor_names)
    ordered_factors: set[str] = set()
    lows = dict.fromkeys(metric_names, np.inf)
    highs = dict.fromkeys(metric_names, -np.inf)
    factor_bytes = sum(sys.getsizeof(value) for value in factor_maps.values())
    factor_bytes += sys.getsizeof(factor_maps)
    range_metadata_bytes = _deep_size((tuple(lows.items()), tuple(highs.items())))
    peak_bytes = factor_bytes + range_metadata_bytes
    start_fingerprint = source.fingerprint()
    scan_offset = 0
    metric_workspace = batch_rows * max(1, len(metric_names)) * 2 * 8
    if metric_workspace > budget.max_batch_bytes:
        return ExactIndexUnavailable(
            "range-analysis batch workspace exceeds budget",
            metric_workspace,
            budget.max_batch_bytes,
        )
    for batch in source.scan(batch_rows):
        scan_stop = _validate_batch_positions(
            batch, batch_rows, scan_offset, source.n_rows
        )
        for name in names:
            if name not in batch.columns:
                raise ValueError(f"Compressed column '{name}' is not in RowSource")
            source_column = batch.columns[name]
            column = np.asarray(source_column)
            if column.shape != (len(batch.row_positions),):
                raise ValueError(
                    f"Compressed column '{name}' has incompatible batch length"
                )
            if name in factor_maps:
                if isinstance(
                    getattr(source_column, "dtype", None), pd.CategoricalDtype
                ):
                    previous = declared_levels[name]
                    if previous is None:
                        categories = source_column.cat.categories
                        level_bytes = (
                            64
                            + (len(categories) + 1) * np.dtype(np.intp).itemsize
                            + sum(sys.getsizeof(level) for level in categories)
                        )
                        candidate_peak = (
                            factor_bytes + range_metadata_bytes + level_bytes + 512
                        )
                        peak_bytes = max(peak_bytes, candidate_peak)
                        if peak_bytes > budget.max_table_bytes:
                            return ExactIndexUnavailable(
                                "factor-level metadata exceeds construction "
                                "byte budget",
                                peak_bytes,
                                budget.max_table_bytes,
                            )
                        # The scalar type of np.asarray(column), which the
                        # level keys must share (np.int64 for integer levels).
                        levels = tuple(np.asarray(categories))
                        declared_levels[name] = levels
                        factor_bytes += level_bytes
                        if source_column.cat.ordered:
                            ordered_factors.add(name)
                    else:
                        categories = np.asarray(source_column.cat.categories)
                        if len(previous) != len(categories) or any(
                            _key((old,)) != _key((new,))
                            for old, new in zip(previous, categories, strict=True)
                        ):
                            raise RuntimeError(
                                f"Factor column '{name}' changed levels during scan"
                            )
                        if source_column.cat.ordered != (name in ordered_factors):
                            raise RuntimeError(
                                f"Factor column '{name}' changed ordering during scan"
                            )
                mapping = factor_maps[name]
                for value in column:
                    key = _key((value,))
                    if key in mapping:
                        continue
                    item_bytes = _deep_size(key) + 4 * sys.getsizeof(mapping) + 512
                    peak_bytes = max(
                        peak_bytes,
                        factor_bytes + range_metadata_bytes + item_bytes,
                    )
                    if peak_bytes > budget.max_table_bytes:
                        return ExactIndexUnavailable(
                            "factor-level analysis exceeds construction byte budget",
                            peak_bytes,
                            budget.max_table_bytes,
                        )
                    old_size = sys.getsizeof(mapping)
                    mapping[key] = value
                    factor_bytes += _deep_size(key) + max(
                        0, sys.getsizeof(mapping) - old_size
                    )
                continue
            # A categorical column is a factor by declaration, never metric on
            # its codes, whatever the dtype of its categories.
            if column.dtype.kind not in "biuf" or isinstance(
                getattr(source_column, "dtype", None), pd.CategoricalDtype
            ):
                raise TypeError(_non_numeric_metric_message(name, source_column))
            numeric = np.asarray(column, dtype=np.float64)
            if not np.all(np.isfinite(numeric)):
                raise ValueError(f"Metric column '{name}' contains non-finite values")
            if len(numeric):
                lows[name] = min(lows[name], float(np.min(numeric)))
                highs[name] = max(highs[name], float(np.max(numeric)))
        scan_offset = scan_stop
    if scan_offset != source.n_rows:
        raise RuntimeError("RowSource yielded fewer than its advertised rows")
    if source.fingerprint() != start_fingerprint:
        raise RuntimeError("RowSource changed during range-grid analysis")
    factor_levels: list[tuple[Any, ...]] = []
    for name in factor_names:
        levels = declared_levels[name]
        if levels is None:
            observed = tuple(factor_maps[name].values())
            try:
                levels = tuple(sorted(observed))
            except TypeError as error:
                raise TypeError(
                    f"Mixed factor column '{name}' requires declared categorical levels"
                ) from error
            factor_level_peak = (
                factor_bytes
                + range_metadata_bytes
                + sys.getsizeof(levels)
                + len(levels) * np.dtype(np.intp).itemsize
            )
            peak_bytes = max(peak_bytes, factor_level_peak)
            if peak_bytes > budget.max_table_bytes:
                return ExactIndexUnavailable(
                    "factor-level metadata exceeds construction byte budget",
                    peak_bytes,
                    budget.max_table_bytes,
                )
        factor_levels.append(levels)
    return _ColumnAnalysis(
        tuple(len(factor_maps[name]) for name in factor_names),
        tuple(factor_levels),
        frozenset(ordered_factors),
        tuple((lows[name], highs[name]) for name in metric_names),
        peak_bytes,
        start_fingerprint,
    )


def range_grid(low: float, high: float, points: int) -> npt.NDArray[np.float64]:
    """Pinned ``seq(low, high, length=points)`` used by ``compress.df``."""
    return np.linspace(low, high, points, dtype=np.float64)


def snap_to_range_grid(
    values: npt.ArrayLike, grid: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Round metric values to ``compress.df``'s evenly spaced grid.

    Ports ``kx <- round((x - xl[1]) / dx) + 1; x <- xu[kx]`` with
    ``dx <- xu[2] - xu[1]``; R's ``round`` and ``np.rint`` both round half
    to even.  A degenerate one-value grid maps every value to that value.
    """
    numeric = np.asarray(values, dtype=np.float64)
    if grid[0] == grid[-1]:
        return np.full(numeric.shape, grid[0], dtype=np.float64)
    spacing = grid[1] - grid[0]
    positions = np.rint((numeric - grid[0]) / spacing).astype(np.intp)
    np.clip(positions, 0, len(grid) - 1, out=positions)
    return grid[positions]


class _RangeGridSource:
    """Replay a source with metric columns replaced by frozen range-grid values.

    It stands for ``source``, so it forwards the source's weight and offset
    declarations.
    """

    def __init__(
        self,
        source: RowSource,
        metric_names: tuple[str, ...],
        grids: tuple[np.ndarray, ...],
    ) -> None:
        self.source = source
        self.metric_names = metric_names
        self.grids = grids
        weights, offset = declared_row_vectors(source)
        self.has_explicit_weights = weights
        self.has_explicit_offset = offset

    @property
    def n_rows(self) -> int:
        return self.source.n_rows

    def fingerprint(self) -> str:
        digest = hashlib.sha256(b"jaxgam-range-grid-source-v1\0")
        digest.update(self.source.fingerprint().encode())
        for name, grid in zip(self.metric_names, self.grids, strict=True):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(grid).view(np.uint8))
        return digest.hexdigest()

    def scan(self, batch_rows: int):
        for batch in self.source.scan(batch_rows):
            columns = dict(batch.columns)
            for name, grid in zip(self.metric_names, self.grids, strict=True):
                columns[name] = snap_to_range_grid(columns[name], grid)
            yield RowBatch(
                MappingProxyType(columns),
                batch.y,
                batch.weight,
                batch.offset,
                batch.valid,
                batch.row_positions,
            )


def _cleanup_memmap(path: str | Path | None) -> None:
    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            candidate.unlink()


class _RRandom:
    """R 4.5's default Mersenne Twister plus rejection index sampler."""

    def __init__(self, seed: int) -> None:
        state = seed & 0xFFFFFFFF
        for _ in range(50):
            state = (69069 * state + 1) & 0xFFFFFFFF
        words = np.empty(625, dtype=np.uint32)
        for index in range(625):
            state = (69069 * state + 1) & 0xFFFFFFFF
            words[index] = state
        # R discards the first initialized word by replacing i_seed[0] with
        # the MT position 624.  NumPy's legacy MT state exposes the same 624
        # words and position, while its high-level float generator does not.
        self._generator = np.random.RandomState()
        self._generator.set_state(("MT19937", words[1:], 624, 0, 0.0))

    def _word(self) -> int:
        return int(self._generator.randint(0, 2**32, dtype=np.uint32))

    def index(self, upper: int) -> int:
        if upper <= 0:
            return 0
        bits = (upper - 1).bit_length()
        while True:
            value = 0
            for _ in range(0, bits + 1, 16):
                value = 65536 * value + (self._word() >> 16)
            value &= (1 << bits) - 1
            if value < upper:
                return value


def _r_sample_without_replacement(
    size: int, seed: int, generator: _RRandom | None = None
) -> np.ndarray:
    """Port ``sample.int(size, size)`` for R 4.5's default RNG kinds."""
    if generator is None:
        generator = _RRandom(seed)
    pool = np.arange(size, dtype=np.int32)
    result = np.empty(size, dtype=np.int32)
    remaining = size
    for output in range(size):
        chosen = generator.index(remaining)
        result[output] = pool[chosen]
        remaining -= 1
        pool[chosen] = pool[remaining]
    return result


def _r_number_text(value: Any) -> str:
    """Format a finite scalar as R's numeric ``paste0`` representation."""
    if isinstance(value, (bool, np.bool_, int, np.integer)):
        return str(int(value))
    numeric = float(value)
    if np.isnan(numeric):
        return "NaN"
    if np.isposinf(numeric):
        return "Inf"
    if np.isneginf(numeric):
        return "-Inf"
    if numeric == 0.0:
        return "0"

    # StringFromReal temporarily sets R_print.digits to DBL_DIG (15), then
    # calls scientific() and formatReal() to choose the representation.  The
    # significant-digit calculation selects fixed versus exponential output;
    # it does *not* round the value passed to the final ``%f`` conversion.
    # That distinction preserves all integer digits of exactly represented
    # large doubles (for example 8956883870379507).
    digits = 15
    magnitude = abs(numeric)
    k_pos = int(np.floor(np.log10(magnitude))) - digits + 1
    has_extended_precision = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
    if has_extended_precision:
        magnitude_precise: float | np.longdouble = np.longdouble(magnitude)
        if abs(k_pos) <= 27:
            scale = np.longdouble(f"1e{abs(k_pos)}")
            if k_pos > 0:
                magnitude_precise /= scale
            elif k_pos < 0:
                magnitude_precise *= scale
        else:
            magnitude_precise /= np.longdouble(10.0) ** np.longdouble(k_pos)
    else:
        # R's non-long-double branch rescales subnormal values before the
        # power-of-ten division.  Directly evaluating 10**k_pos underflows on
        # platforms where longdouble is the same width as double (macOS).
        magnitude_precise = magnitude
        if abs(k_pos) <= 22:
            scale = float(f"1e{abs(k_pos)}")
            if k_pos > 0:
                magnitude_precise /= scale
            elif k_pos < 0:
                magnitude_precise *= scale
        elif k_pos <= sys.float_info.min_10_exp:
            magnitude_precise = (magnitude_precise * 1e303) / math.pow(
                10.0, k_pos + 303
            )
        else:
            magnitude_precise /= math.pow(10.0, k_pos)
    digit_floor = np.longdouble("1e14") if has_extended_precision else 1e14
    if magnitude_precise < digit_floor:
        magnitude_precise *= np.longdouble(10.0)
        k_pos -= 1

    alpha = float(np.rint(magnitude_precise))
    n_significant = digits
    for _ in range(digits):
        alpha /= 10.0
        if alpha == np.floor(alpha):
            n_significant -= 1
        else:
            break
    if n_significant == 0:
        n_significant = 1
        k_pos += 1
    k_power = k_pos + digits - 1

    right_cut = min(max(digits - k_power, 0), 27)
    fuzz = 0.5 / float(np.longdouble(f"1e{right_cut}"))
    rounding_widens = (
        0 < k_power <= 27 and magnitude < float(np.longdouble(f"1e{k_power}")) - fuzz
    )

    negative = int(numeric < 0.0)
    left = k_power + 1 - int(rounding_widens)
    signed_left = negative + (1 if left <= 0 else left)
    right = n_significant - left
    fixed_digits = max(right, 0)
    if left < 0:
        signed_left = 1 + negative
    fixed_width = signed_left + fixed_digits + int(fixed_digits != 0)

    exponent_digits_minus_one = 2 if left > 100 or left <= -99 else 1
    scientific_digits = n_significant - 1
    scientific_width = (
        negative
        + int(scientific_digits > 0)
        + scientific_digits
        + 4
        + exponent_digits_minus_one
    )
    if fixed_width <= scientific_width:
        return (
            format(numeric, f".{fixed_digits}f").rstrip("0").rstrip(".")
            if fixed_digits
            else format(numeric, ".0f")
        )

    encoded = format(numeric, f"#{'.' + str(scientific_digits)}e")
    mantissa, exponent = encoded.split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_value = int(exponent)
    exponent_width = exponent_digits_minus_one + 1
    return f"{mantissa}e{exponent_value:+0{exponent_width + 1}d}"


def _factor_code_maps(
    analysis: _ColumnAnalysis, factor_names: tuple[str, ...]
) -> dict[str, dict[tuple[tuple[str, Any], ...], int]]:
    return {
        name: {_key((level,)): code + 1 for code, level in enumerate(levels)}
        for name, levels in zip(factor_names, analysis.factor_levels, strict=True)
    }


def _factor_code_storage_bound(analysis: _ColumnAnalysis) -> int:
    total = 0
    for levels in analysis.factor_levels:
        total += 4 * (64 + len(levels) * 16) + 512
        total += sum(
            _deep_size((_key((level,)), code))
            for code, level in enumerate(levels, start=1)
        )
    return total


def _r_row_token(
    names: tuple[str, ...],
    values: tuple[Any, ...],
    factor_codes: dict[str, dict[tuple[tuple[str, Any], ...], int]],
) -> str:
    components = []
    for name, value in zip(names, values, strict=True):
        if name in factor_codes:
            try:
                components.append(str(factor_codes[name][_key((value,))]))
            except KeyError as error:
                raise RuntimeError(
                    f"Observed factor value for '{name}' is absent from its levels"
                ) from error
        else:
            components.append(_r_number_text(value))
    return "*".join(components)


def _uniquecombs_index(
    source: RowSource,
    names: tuple[str, ...],
    analysis: _ColumnAnalysis,
    factor_names: tuple[str, ...],
    *,
    budget: DiscretizationBudget,
    max_unique_rows: int,
    selector_memmap: str | Path | None,
    batch_rows: int,
) -> ExactIndexTable | ExactIndexUnavailable:
    """Bounded index using mgcv ``uniquecombs`` row equivalence.

    One-column inputs use exact scalar equality.  Multi-column inputs use the
    same formatted numeric/factor-code token that mgcv uses for duplicate
    detection, so its intentional 15-digit numeric collisions are preserved.
    """
    n_rows = source.n_rows
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
    token_workspace = batch_rows * (128 + 64 * len(names)) if len(names) > 1 else 0
    position_workspace = 2 * batch_rows * np.dtype(np.intp).itemsize
    batch_workspace = token_workspace + position_workspace
    if batch_workspace > budget.max_batch_bytes:
        return ExactIndexUnavailable(
            "uniquecombs batch workspace exceeds budget",
            batch_workspace,
            budget.max_batch_bytes,
        )
    factor_code_bytes = _factor_code_storage_bound(analysis)
    retained_level_bytes = sum(_deep_size(levels) for levels in analysis.factor_levels)
    initial_construction_bytes = (
        sys.getsizeof({}) + sys.getsizeof([]) + factor_code_bytes + retained_level_bytes
    )
    if initial_construction_bytes > budget.max_table_bytes:
        return ExactIndexUnavailable(
            "uniquecombs factor metadata exceeds construction byte budget",
            initial_construction_bytes,
            budget.max_table_bytes,
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
            created_path.unlink(missing_ok=True)
            raise
        storage = "memmap"
    factor_codes = _factor_code_maps(analysis, factor_names)
    index: dict[Any, int] = {}
    unique: list[tuple[Any, ...]] = []
    scan_offset = 0
    construction_bytes = initial_construction_bytes
    construction_peak = construction_bytes
    start_fingerprint = source.fingerprint()
    try:
        for batch in source.scan(batch_rows):
            scan_stop = _validate_batch_positions(
                batch, batch_rows, scan_offset, n_rows
            )
            columns = []
            for name in names:
                if name not in batch.columns:
                    raise ValueError(f"Compressed column '{name}' is not in RowSource")
                column = np.asarray(batch.columns[name])
                if column.shape != (len(batch.row_positions),):
                    raise ValueError(
                        f"Compressed column '{name}' has incompatible batch length"
                    )
                columns.append(column)
            for local in range(len(batch.row_positions)):
                values = tuple(column[local] for column in columns)
                key: Any = (
                    _key(values)
                    if len(names) == 1
                    else _r_row_token(names, values, factor_codes)
                )
                item = index.get(key)
                if item is None:
                    old_containers = sys.getsizeof(index) + sys.getsizeof(unique)
                    payload_bytes = _deep_size((key, values))
                    growth_bound = (
                        4 * sys.getsizeof(index)
                        + 3 * sys.getsizeof(unique)
                        + 512
                        - old_containers
                    )
                    item_bytes = (
                        payload_bytes + sys.getsizeof(len(unique)) + growth_bound
                    )
                    construction_peak = max(
                        construction_peak, construction_bytes + item_bytes
                    )
                    if len(unique) >= max_unique_rows:
                        _cleanup_memmap(created_path)
                        return ExactIndexUnavailable(
                            "uniquecombs rows exceed row cap",
                            construction_bytes + item_bytes,
                            budget.max_table_bytes,
                            len(unique) + 1,
                            max_unique_rows,
                        )
                    if construction_bytes + item_bytes > budget.max_table_bytes:
                        _cleanup_memmap(created_path)
                        return ExactIndexUnavailable(
                            "uniquecombs rows exceed construction byte budget",
                            construction_bytes + item_bytes,
                            budget.max_table_bytes,
                        )
                    item = len(unique)
                    index[key] = item
                    unique.append(values)
                    container_growth = max(
                        0,
                        sys.getsizeof(index) + sys.getsizeof(unique) - old_containers,
                    )
                    construction_bytes += (
                        payload_bytes + sys.getsizeof(item) + container_growth
                    )
                selector[scan_offset + local] = item
            scan_offset = scan_stop
        if scan_offset != n_rows:
            raise RuntimeError("RowSource yielded fewer than its advertised rows")
        if source.fingerprint() != start_fingerprint:
            raise RuntimeError("RowSource changed during uniquecombs indexing")
        if storage == "memmap":
            selector.flush()
        columns_list: list[np.ndarray] = []
        retained_column_bytes = 0
        for column_number in range(len(names)):
            dtype, column_bytes = _column_dtype_and_bytes(unique, column_number)
            list_bytes = (
                sys.getsizeof([]) + (len(unique) + 1) * np.dtype(np.intp).itemsize + 64
            )
            materialization_peak = (
                construction_bytes
                + retained_column_bytes
                + list_bytes
                + 2 * column_bytes
            )
            construction_peak = max(construction_peak, materialization_peak)
            if construction_peak > budget.max_table_bytes:
                _cleanup_memmap(created_path)
                return ExactIndexUnavailable(
                    "uniquecombs materialization exceeds construction byte budget",
                    construction_peak,
                    budget.max_table_bytes,
                )
            values_list = [row[column_number] for row in unique]
            column = np.asarray(values_list, dtype=dtype)
            columns_list.append(column)
            retained_column_bytes += column_bytes
        ownership_peak = construction_bytes + 2 * retained_column_bytes
        construction_peak = max(construction_peak, ownership_peak)
        if construction_peak > budget.max_table_bytes:
            _cleanup_memmap(created_path)
            return ExactIndexUnavailable(
                "uniquecombs ownership copy exceeds construction byte budget",
                construction_peak,
                budget.max_table_bytes,
            )
        return ExactIndexTable(
            names,
            tuple(columns_list),
            selector,
            n_rows,
            storage,
            construction_peak,
        )
    except BaseException:
        _cleanup_memmap(created_path)
        raise


def _uniquecombs_order(
    index: ExactIndexTable,
    analysis: _ColumnAnalysis,
    factor_names: tuple[str, ...],
    budget: DiscretizationBudget,
    prior_peak_bytes: int,
) -> tuple[np.ndarray, int] | ExactIndexUnavailable:
    """Reproduce ``uniquecombs(..., ordered=TRUE)`` compact row ordering."""
    q = index.n_unique
    n_columns = len(index.names)
    token_bound = q * (128 + 64 * n_columns)
    factor_code_bound = _factor_code_storage_bound(analysis)
    order_arrays = q * 3 * np.dtype(np.int32).itemsize
    peak = max(
        prior_peak_bytes,
        index.value_bytes + factor_code_bound + token_bound + order_arrays,
    )
    if peak > budget.max_table_bytes:
        return ExactIndexUnavailable(
            "unique-row ordering exceeds construction byte budget",
            peak,
            budget.max_table_bytes,
        )
    factor_codes = _factor_code_maps(analysis, factor_names)
    if n_columns == 1:
        name = index.names[0]
        column = index.values[0]
        if name in factor_codes:
            order = sorted(
                range(q), key=lambda row: factor_codes[name][_key((column[row],))]
            )
        else:
            order = sorted(range(q), key=lambda row: float(column[row]))
        return np.asarray(order, dtype=np.int32), peak
    tokens: list[str] = []
    for row in range(q):
        values = tuple(column[row] for column in index.values)
        tokens.append(_r_row_token(index.names, values, factor_codes))
    order = np.asarray(
        sorted(range(q), key=lambda row: tokens[row].encode()), dtype=np.int32
    )
    return order, peak


def _shuffle_index(
    index: ExactIndexTable,
    *,
    unique_order: np.ndarray,
    seed: int,
    generator: _RRandom | None,
    budget: DiscretizationBudget,
    batch_rows: int,
    prior_peak_bytes: int,
    retained_grid_bytes: int,
    selector_memmap: str | Path | None,
) -> ExactIndexTable | ExactIndexUnavailable:
    """Shuffle representatives and remap one owned selector in bounded chunks."""
    q = index.n_unique
    if q == index.n_rows:
        return index
    transient_index_bytes = q * 4 * np.dtype(np.int32).itemsize
    remap_bytes = 2 * min(batch_rows, index.n_rows) * np.dtype(np.int32).itemsize
    peak = max(
        prior_peak_bytes,
        retained_grid_bytes
        + index.value_bytes * 3
        + transient_index_bytes
        + remap_bytes,
    )
    if peak > budget.max_table_bytes:
        _cleanup_memmap(selector_memmap)
        return ExactIndexUnavailable(
            "representative shuffle exceeds construction byte budget",
            peak,
            budget.max_table_bytes,
        )
    destinations = _r_sample_without_replacement(q, seed, generator)
    ordered_position = np.empty(q, dtype=np.int32)
    ordered_position[unique_order] = np.arange(q, dtype=np.int32)
    read_order = np.empty(q, dtype=np.int32)
    read_order[destinations] = unique_order
    selector = index.selector
    selector.setflags(write=True)
    for start in range(0, index.n_rows, batch_rows):
        stop = min(start + batch_rows, index.n_rows)
        selector[start:stop] = destinations[ordered_position[selector[start:stop]]]
    selector.setflags(write=False)
    if isinstance(selector, np.memmap):
        selector.flush()
    values = tuple(value[read_order] for value in index.values)
    return ExactIndexTable(
        index.names,
        values,
        selector,
        index.n_rows,
        index.storage,
        peak,
    )


def range_grid_index_columns(
    source: RowSource,
    names: tuple[str, ...],
    *,
    factor_names: tuple[str, ...] = (),
    m: int | None = None,
    budget: DiscretizationBudget | None = None,
    selector_memmap: str | Path | None = None,
    batch_rows: int = 65_536,
    shuffle_seed: int = 8547,
    _random: _RRandom | None = None,
) -> RangeGridIndex | ExactIndexUnavailable:
    """Apply pinned ``compress.df``'s exact-first range-grid policy.

    Representative ordering follows ``uniquecombs(..., ordered=TRUE)`` and the
    permutation uses a local port of R 4.5's default MT/rejection sampler.  It
    is portable, deterministic and does not alter process-global RNG state.
    """
    if budget is None:
        budget = DiscretizationBudget()
    if source.n_rows <= 0:
        raise ValueError("range-grid indexing requires at least one source row")
    if (
        isinstance(batch_rows, bool)
        or not isinstance(batch_rows, int)
        or batch_rows <= 0
    ):
        raise ValueError("batch_rows must be a positive integer")
    if not names or len(names) != len(set(names)):
        raise ValueError("range-grid indexing requires distinct column names")
    if len(factor_names) != len(set(factor_names)) or not set(factor_names) <= set(
        names
    ):
        raise ValueError("factor_names must be distinct members of names")
    if (
        isinstance(shuffle_seed, bool)
        or not isinstance(shuffle_seed, int)
        or not -(2**31) <= shuffle_seed < 2**31
    ):
        raise ValueError("shuffle_seed must be a signed 32-bit integer")
    points = _points_per_metric(len(names), m)
    factor_set = set(factor_names)
    metric_names = tuple(name for name in names if name not in factor_set)
    analysis = _analyze_columns(
        source, names, factor_names, metric_names, budget, batch_rows
    )
    if isinstance(analysis, ExactIndexUnavailable):
        return analysis
    if source.fingerprint() != analysis.source_fingerprint:
        raise RuntimeError("RowSource changed between range-grid scans")
    factor_product = math.prod(analysis.factor_counts)
    threshold = factor_product * points ** len(metric_names)
    decision_cap = min(threshold, budget.max_unique_rows)
    exact = _uniquecombs_index(
        source,
        names,
        analysis,
        factor_names,
        budget=budget,
        max_unique_rows=decision_cap,
        selector_memmap=selector_memmap,
        batch_rows=batch_rows,
    )
    rounded = False
    grid_bytes = 0
    if isinstance(exact, ExactIndexUnavailable):
        hit_decision_boundary = (
            exact.required_rows is not None
            and exact.row_cap == threshold
            and threshold <= budget.max_unique_rows
        )
        if not hit_decision_boundary:
            return exact
        rounded = True
        constant_metrics = tuple(
            name
            for name, (low, high) in zip(
                metric_names, analysis.metric_ranges, strict=True
            )
            if low == high
        )
        if constant_metrics:
            raise ValueError(
                "pinned compress.df produces an undefined range grid for "
                f"constant metric column(s) {constant_metrics} when joint "
                "rounding is required"
            )
        grids = tuple(
            range_grid(low, high, points) for low, high in analysis.metric_ranges
        )
        grid_bytes = sum(int(grid.nbytes) for grid in grids)
        # Each transformed metric remains in the emitted column mapping.  The
        # active metric may additionally need a float conversion, normalized
        # coordinates, rounded coordinates and integer selectors before its
        # representative array is gathered.  Exact indexing concurrently
        # validates scan-local positions with two integer arrays.
        rounded_workspace = batch_rows * (
            len(metric_names) * np.dtype(np.float64).itemsize
            + 4 * np.dtype(np.float64).itemsize
            + 3 * np.dtype(np.intp).itemsize
        )
        if rounded_workspace > budget.max_batch_bytes:
            return ExactIndexUnavailable(
                "range-grid indexing batch workspace exceeds budget",
                rounded_workspace,
                budget.max_batch_bytes,
            )
        remaining = budget.max_table_bytes - grid_bytes
        if remaining <= 0:
            return ExactIndexUnavailable(
                "range-grid representatives exceed construction byte budget",
                grid_bytes,
                budget.max_table_bytes,
            )
        rounded_budget = DiscretizationBudget(
            max_table_bytes=remaining,
            max_selector_bytes=budget.max_selector_bytes,
            max_batch_bytes=budget.max_batch_bytes,
            max_unique_rows=budget.max_unique_rows,
        )
        exact = _uniquecombs_index(
            _RangeGridSource(source, metric_names, grids),
            names,
            analysis,
            factor_names,
            budget=rounded_budget,
            max_unique_rows=budget.max_unique_rows,
            selector_memmap=selector_memmap,
            batch_rows=batch_rows,
        )
        if isinstance(exact, ExactIndexUnavailable):
            return ExactIndexUnavailable(
                exact.reason,
                exact.required_bytes + grid_bytes,
                budget.max_table_bytes,
                exact.required_rows,
                exact.row_cap,
            )
    if source.fingerprint() != analysis.source_fingerprint:
        _cleanup_memmap(selector_memmap)
        raise RuntimeError("RowSource changed between range-grid scans")
    construction_peak = max(
        analysis.peak_bytes,
        exact.construction_peak_bytes + grid_bytes,
    )
    if exact.n_unique == exact.n_rows:
        unique_order = np.arange(exact.n_unique, dtype=np.int32)
    else:
        ordered = _uniquecombs_order(
            exact,
            analysis,
            factor_names,
            budget,
            construction_peak,
        )
        if isinstance(ordered, ExactIndexUnavailable):
            _cleanup_memmap(selector_memmap)
            return ordered
        unique_order, ordering_peak = ordered
        construction_peak = max(construction_peak, ordering_peak)
    shuffled = _shuffle_index(
        exact,
        unique_order=unique_order,
        seed=shuffle_seed,
        generator=_random,
        budget=budget,
        batch_rows=batch_rows,
        prior_peak_bytes=construction_peak,
        retained_grid_bytes=grid_bytes,
        selector_memmap=selector_memmap,
    )
    if isinstance(shuffled, ExactIndexUnavailable):
        return shuffled
    construction_peak = max(construction_peak, shuffled.construction_peak_bytes)
    return RangeGridIndex(
        shuffled,
        factor_names,
        analysis.factor_levels,
        analysis.ordered_factors,
        metric_names,
        m,
        points,
        analysis.metric_ranges,
        rounded,
        shuffle_seed,
        construction_peak_bytes=construction_peak,
        factor_counts=analysis.factor_counts,
    )
