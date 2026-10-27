"""Shared range-grid groups and pinned ``discrete.mf`` setup frames.

This Phase-1 module owns term/margin discovery and padding only.  It does not
construct smooth bases; the compact frame is the input to the separate PR11
compact setup stage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from jaxgam.data.continuous_discretize import (
    RangeGridIndex,
    _RRandom,
    range_grid_index_columns,
)
from jaxgam.data.discretize import (
    DiscretizationBudget,
    ExactIndexUnavailable,
    _deep_size,
    _key,
)
from jaxgam.data.source import RowBatch, RowSource
from jaxgam.formula.terms import FormulaSpec, SmoothSpec


@dataclass(frozen=True)
class CompactGroup:
    """One shared smooth margin, by-variable, or parametric grid."""

    names: tuple[str, ...]
    factor_names: tuple[str, ...]
    index: RangeGridIndex
    role: Literal["by", "tensor_margin", "smooth", "parametric"]
    m: int | None

    def __post_init__(self) -> None:
        if self.names != self.index.names:
            raise ValueError("compact group names do not match its exact index")
        if self.factor_names != self.index.factor_names:
            raise ValueError("compact group factor metadata does not match its index")
        if self.m is not None and self.m < 2:
            raise ValueError("compact group m must be at least 2")


@dataclass(frozen=True)
class CompactFrameUnavailable:
    """A combined compact setup frame could not satisfy its budget."""

    reason: str
    required_bytes: int
    budget_bytes: int
    budget_scope: Literal["table", "selector", "batch"]
    group_names: tuple[str, ...] | None = None
    required_rows: int | None = None
    row_cap: int | None = None


@dataclass(frozen=True)
class CompactBatchSelectors:
    """One bounded gather per unique compact group for a source batch."""

    group_selectors: tuple[npt.NDArray[np.int32], ...]
    name_group_indices: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.group_selectors:
            raise ValueError("compact batch requires at least one group selector")
        size = len(self.group_selectors[0])
        for selector in self.group_selectors:
            if selector.dtype != np.int32 or selector.shape != (size,):
                raise TypeError("compact batch selectors must be aligned int32 vectors")
            selector.setflags(write=False)

    def for_name(self, name: str) -> npt.NDArray[np.int32]:
        """Reuse the already-gathered selector for a compact variable."""
        if name == "(Intercept)":
            return np.broadcast_to(np.int32(0), (len(self.group_selectors[0]),))
        try:
            group = dict(self.name_group_indices)[name]
        except KeyError as error:
            raise KeyError(f"No compact batch selector for '{name}'") from error
        return self.group_selectors[group]


@dataclass(frozen=True)
class CompactSetupFrame:
    """Padded compact frame and shared full-row selector metadata."""

    frame: pd.DataFrame
    groups: tuple[CompactGroup, ...]
    name_group_indices: tuple[tuple[str, int], ...]
    selector_paths: tuple[Path, ...]
    source_fingerprint: str
    construction_peak_bytes: int
    retained_table_bytes: int
    retained_group_bytes: int
    selector_peak_bytes: int
    retained_selector_bytes: int
    max_batch_bytes: int

    def __post_init__(self) -> None:
        mapping = dict(self.name_group_indices)
        if len(mapping) != len(self.name_group_indices):
            raise ValueError("compact frame contains duplicate variable metadata")
        if not self.groups:
            raise ValueError("compact frame requires at least one group")
        if self.frame.empty:
            raise ValueError("compact setup frame cannot be empty")
        if any(index < 0 or index >= len(self.groups) for index in mapping.values()):
            raise ValueError("compact variable references an invalid group")
        expected_names = {name for group in self.groups for name in group.names}
        if set(mapping) != expected_names or expected_names != set(self.frame):
            raise ValueError("compact variable metadata does not cover its frame")
        if any(
            group.index.n_rows != self.groups[0].index.n_rows for group in self.groups
        ):
            raise ValueError("compact groups do not share an observation count")
        if len(self.frame) != max(group.index.n_unique for group in self.groups):
            raise ValueError("compact frame length does not match its largest group")
        if self.construction_peak_bytes < self.retained_table_bytes:
            raise ValueError("compact construction peak is below retained storage")
        if self.retained_table_bytes < self.retained_group_bytes:
            raise ValueError("compact retained total is below retained group storage")
        if self.selector_peak_bytes < self.retained_selector_bytes:
            raise ValueError("compact selector peak is below retained storage")
        if self.max_batch_bytes <= 0:
            raise ValueError("compact batch budget must be positive")

    @property
    def n_rows(self) -> int:
        return self.groups[0].index.n_rows

    @property
    def max_compact_rows(self) -> int:
        return len(self.frame)

    @property
    def n_unique(self) -> dict[str, int]:
        result = {
            name: self.groups[group].index.n_unique
            for name, group in self.name_group_indices
        }
        result["(Intercept)"] = 1
        return result

    @property
    def index_spans(self) -> dict[str, tuple[int, int]]:
        result = {name: (group, group + 1) for name, group in self.name_group_indices}
        result["(Intercept)"] = (len(self.groups), len(self.groups) + 1)
        return result

    def selector_for(self, name: str) -> npt.NDArray[np.int32]:
        """Return the shared selector for a named compact variable."""
        if name == "(Intercept)":
            result = np.broadcast_to(np.int32(0), (self.n_rows,))
            result.setflags(write=False)
            return result
        try:
            group = dict(self.name_group_indices)[name]
        except KeyError as error:
            raise KeyError(f"No compact selector for '{name}'") from error
        return self.groups[group].index.selector

    def batch_selectors(
        self, row_ordinals: npt.NDArray[np.integer[Any]]
    ) -> CompactBatchSelectors:
        """Gather unique group indices for scan-local batch ordinals once."""
        if not isinstance(row_ordinals, np.ndarray):
            raise TypeError("row ordinals must be an integer NumPy array")
        if row_ordinals.ndim != 1 or row_ordinals.dtype.kind not in "iu":
            raise TypeError("row ordinals must be a one-dimensional integer array")
        required_bytes = (
            len(row_ordinals)
            * (
                len(self.groups) * np.dtype(np.int32).itemsize
                + np.dtype(np.intp).itemsize
                + np.dtype(bool).itemsize
            )
            + 256
            + 128 * len(self.groups)
        )
        if required_bytes > self.max_batch_bytes:
            raise MemoryError(
                "compact batch selector gathering exceeds max_batch_bytes: "
                f"required {required_bytes}, allowed {self.max_batch_bytes}"
            )
        if np.any(row_ordinals < 0) or np.any(row_ordinals >= self.n_rows):
            raise ValueError("row ordinals are outside the compact source")
        selectors = tuple(
            np.asarray(group.index.selector[row_ordinals], dtype=np.int32)
            for group in self.groups
        )
        return CompactBatchSelectors(selectors, self.name_group_indices)

    def cleanup(self) -> None:
        """Remove successfully-created caller-requested selector memmaps."""
        for path in self.selector_paths:
            path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _GroupRequest:
    names: tuple[str, ...]
    role: Literal["by", "tensor_margin", "smooth", "parametric"]
    m: int | None


class _FormulaRowSource:
    """Expose a RowSource response as its formula-named column."""

    def __init__(self, source: RowSource, response: str) -> None:
        self.source = source
        self.response = response

    @property
    def n_rows(self) -> int:
        return self.source.n_rows

    def fingerprint(self) -> str:
        digest = hashlib.sha256(b"jaxgam-formula-row-source-v1\0")
        digest.update(self.source.fingerprint().encode())
        digest.update(self.response.encode())
        return digest.hexdigest()

    def scan(self, batch_rows: int):
        for batch in self.source.scan(batch_rows):
            columns = dict(batch.columns)
            if self.response not in columns:
                if batch.y is None:
                    raise ValueError(
                        f"Response '{self.response}' is absent from RowSource"
                    )
                columns[self.response] = batch.y
            yield RowBatch(
                MappingProxyType(columns),
                batch.y,
                batch.weight,
                batch.offset,
                batch.valid,
                batch.row_positions,
            )


def _smooth_m_values(
    m: int | tuple[int, ...] | None, n_smooths: int
) -> tuple[tuple[int | None, ...], int | None]:
    if isinstance(m, tuple):
        if len(m) != n_smooths or not m:
            raise ValueError("m tuple must have one entry per smooth term")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in m
        ):
            raise ValueError("all m values must be integers of at least 2")
        return tuple(m), max(m)
    if m is not None and (isinstance(m, bool) or not isinstance(m, int) or m < 2):
        raise ValueError("m must be an integer of at least 2")
    return (m,) * n_smooths, m


def _smooth_requests(spec: SmoothSpec, m: int | None) -> list[_GroupRequest]:
    result = []
    if spec.by is not None:
        result.append(_GroupRequest((spec.by,), "by", m))
    if spec.smooth_type in ("te", "ti"):
        result.extend(
            _GroupRequest((name,), "tensor_margin", m) for name in spec.variables
        )
    else:
        result.append(_GroupRequest(tuple(spec.variables), "smooth", m))
    return result


def _discover_requests(
    formula: FormulaSpec, m: int | tuple[int, ...] | None
) -> tuple[tuple[_GroupRequest, ...], tuple[tuple[str, int], ...]]:
    smooth_m, parametric_m = _smooth_m_values(m, len(formula.smooth_terms))
    requests: list[_GroupRequest] = []
    name_groups: dict[str, int] = {}

    def add_smooth(request: _GroupRequest) -> None:
        if not request.names or len(request.names) != len(set(request.names)):
            raise ValueError("compact smooth groups require distinct variable names")
        overlaps = {name_groups[name] for name in request.names if name in name_groups}
        if overlaps:
            if len(overlaps) != 1:
                raise ValueError("continuous discretization has incompatible nesting")
            existing_index = next(iter(overlaps))
            existing = requests[existing_index]
            if len(existing.names) != len(request.names) or set(existing.names) != set(
                request.names
            ):
                raise ValueError("continuous discretization has incompatible nesting")
            return
        group_index = len(requests)
        requests.append(request)
        name_groups.update(dict.fromkeys(request.names, group_index))

    for spec, term_m in zip(formula.smooth_terms, smooth_m, strict=True):
        for request in _smooth_requests(spec, term_m):
            add_smooth(request)

    for name in (formula.response, *(term.name for term in formula.parametric_terms)):
        if name in name_groups:
            continue
        group_index = len(requests)
        requests.append(_GroupRequest((name,), "parametric", parametric_m))
        name_groups[name] = group_index
    return tuple(requests), tuple(name_groups.items())


def _metadata_bytes(index: RangeGridIndex) -> int:
    return (
        index.index.value_bytes
        + _deep_size(index.factor_levels)
        + _deep_size(index.metric_ranges)
        + 1024
    )


def _cleanup(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _unavailable_scope(
    unavailable: ExactIndexUnavailable,
    group_budget: DiscretizationBudget,
    retained_table_bytes: int,
    retained_selector_bytes: int,
) -> tuple[int, int, Literal["table", "selector", "batch"]]:
    reason = unavailable.reason.lower()
    if "selector" in reason or "int32 addressable" in reason:
        return (
            retained_selector_bytes + unavailable.required_bytes,
            retained_selector_bytes + group_budget.max_selector_bytes,
            "selector",
        )
    if "batch" in reason or "workspace" in reason:
        return unavailable.required_bytes, group_budget.max_batch_bytes, "batch"
    return (
        retained_table_bytes + unavailable.required_bytes,
        retained_table_bytes + group_budget.max_table_bytes,
        "table",
    )


def _padded_column(
    values: np.ndarray,
    size: int,
    random: _RRandom,
    *,
    factor_levels: tuple[Any, ...] | None,
    ordered: bool,
) -> Any:
    q = len(values)
    if factor_levels is None:
        output = np.empty(size, dtype=values.dtype)
        output[:q] = values
        for row in range(q, size):
            if (
                q == 1
                and np.issubdtype(values.dtype, np.number)
                and not np.issubdtype(values.dtype, np.bool_)
                and np.isfinite(values[0])
                and values[0] >= 1
            ):
                population = int(values[0])
                if float(values[0]) > 4.5e15:
                    raise ValueError("R-compatible scalar sample exceeds 4.5e15")
                output[row] = random.index(population) + 1
            else:
                output[row] = values[random.index(q)]
        return output
    codes_by_value = {_key((level,)): code for code, level in enumerate(factor_levels)}
    codes = np.empty(size, dtype=np.int32)
    for row, value in enumerate(values):
        codes[row] = codes_by_value[_key((value,))]
    for row in range(q, size):
        codes[row] = codes[random.index(q)]
    return pd.Series(
        pd.Categorical.from_codes(codes, categories=factor_levels, ordered=ordered)
    )


def build_compact_setup_frame(
    formula: FormulaSpec,
    source: RowSource,
    *,
    factor_names: tuple[str, ...] = (),
    m: int | tuple[int, ...] | None = None,
    budget: DiscretizationBudget | None = None,
    selector_memmap_dir: str | Path | None = None,
    batch_rows: int = 65_536,
    seed: int = 8547,
) -> CompactSetupFrame | CompactFrameUnavailable:
    """Build shared groups and the padded input for compact basis setup."""
    if budget is None:
        budget = DiscretizationBudget()
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not -(2**31) <= seed < 2**31
    ):
        raise ValueError("seed must be a signed 32-bit integer")
    requests, name_groups = _discover_requests(formula, m)
    all_names = {name for request in requests for name in request.names}
    if (
        len(factor_names) != len(set(factor_names))
        or not set(factor_names) <= all_names
    ):
        raise ValueError("factor_names must be distinct referenced variables")
    directory = None if selector_memmap_dir is None else Path(selector_memmap_dir)
    if directory is not None and not directory.is_dir():
        raise FileNotFoundError(
            f"selector memmap directory does not exist: {directory}"
        )
    wrapped = _FormulaRowSource(source, formula.response)
    source_fingerprint = wrapped.fingerprint()
    random = _RRandom(seed)
    groups: list[CompactGroup] = []
    paths: list[Path] = []
    retained_table_bytes = 0
    retained_selector_bytes = 0
    construction_peak = 0
    selector_peak = 0
    try:
        for group_number, request in enumerate(requests):
            remaining_table = budget.max_table_bytes - retained_table_bytes
            remaining_selector = budget.max_selector_bytes - retained_selector_bytes
            if remaining_table <= 0:
                _cleanup(paths)
                return CompactFrameUnavailable(
                    "retained compact groups exhaust table construction budget",
                    retained_table_bytes + 1,
                    budget.max_table_bytes,
                    "table",
                    request.names,
                )
            if remaining_selector <= 0:
                _cleanup(paths)
                return CompactFrameUnavailable(
                    "retained compact groups exhaust selector construction budget",
                    retained_selector_bytes + 1,
                    budget.max_selector_bytes,
                    "selector",
                    request.names,
                )
            group_budget = DiscretizationBudget(
                max_table_bytes=remaining_table,
                max_selector_bytes=remaining_selector,
                max_batch_bytes=budget.max_batch_bytes,
                max_unique_rows=budget.max_unique_rows,
            )
            path = (
                None
                if directory is None
                else directory / f"compact-selector-{group_number}.dat"
            )
            if path is not None and path.exists():
                raise FileExistsError(f"selector memmap already exists: {path}")
            selector_peak = max(
                selector_peak,
                retained_selector_bytes
                + wrapped.n_rows * np.dtype(bool).itemsize
                + (
                    0
                    if path is not None
                    else wrapped.n_rows * np.dtype(np.int32).itemsize
                ),
            )
            result = range_grid_index_columns(
                wrapped,
                request.names,
                factor_names=tuple(
                    name for name in request.names if name in factor_names
                ),
                m=request.m,
                budget=group_budget,
                selector_memmap=path,
                batch_rows=batch_rows,
                shuffle_seed=seed,
                _random=random,
            )
            if isinstance(result, ExactIndexUnavailable):
                required, allowed, scope = _unavailable_scope(
                    result,
                    group_budget,
                    retained_table_bytes,
                    retained_selector_bytes,
                )
                _cleanup(paths)
                return CompactFrameUnavailable(
                    result.reason,
                    required,
                    allowed,
                    scope,
                    request.names,
                    result.required_rows,
                    result.row_cap,
                )
            if path is not None:
                paths.append(path)
            group_bytes = _metadata_bytes(result)
            construction_peak = max(
                construction_peak,
                retained_table_bytes + result.construction_peak_bytes,
            )
            retained_table_bytes += group_bytes
            if result.index.storage == "memory":
                retained_selector_bytes += result.index.selector_bytes
            if retained_table_bytes > budget.max_table_bytes:
                _cleanup(paths)
                return CompactFrameUnavailable(
                    "retained compact group metadata exceeds construction budget",
                    retained_table_bytes,
                    budget.max_table_bytes,
                    "table",
                    request.names,
                )
            groups.append(
                CompactGroup(
                    request.names,
                    result.factor_names,
                    result,
                    request.role,
                    request.m,
                )
            )
        if wrapped.fingerprint() != source_fingerprint:
            raise RuntimeError("RowSource changed between compact group scans")
        max_rows = max(group.index.n_unique for group in groups)
        setup_storage = 0
        for group in groups:
            factor_levels = dict(
                zip(
                    group.index.factor_names,
                    group.index.factor_levels,
                    strict=True,
                )
            )
            for name, values in zip(group.names, group.index.values, strict=True):
                if name in factor_levels:
                    setup_storage += max_rows * np.dtype(np.int32).itemsize
                    setup_storage += 2 * _deep_size(factor_levels[name])
                else:
                    setup_storage += max_rows * values.dtype.itemsize
                setup_storage += 512
        # During DataFrame construction, the padded columns and their frame
        # manager copies may coexist.  Keep a separate conservative retained
        # bound for the completed frame and charge the source padded columns
        # once more as live construction workspace.
        frame_retained_bound = 2 * setup_storage + 8192
        padding_peak = retained_table_bytes + setup_storage + frame_retained_bound
        construction_peak = max(construction_peak, padding_peak)
        if construction_peak > budget.max_table_bytes:
            _cleanup(paths)
            return CompactFrameUnavailable(
                "padded compact setup frame exceeds construction budget",
                construction_peak,
                budget.max_table_bytes,
                "table",
            )
        padded: dict[str, Any] = {}
        for group in groups:
            levels = dict(
                zip(
                    group.index.factor_names,
                    group.index.factor_levels,
                    strict=True,
                )
            )
            for name, values in zip(group.names, group.index.values, strict=True):
                padded[name] = _padded_column(
                    values,
                    max_rows,
                    random,
                    factor_levels=levels.get(name),
                    ordered=name in group.index.ordered_factors,
                )
        frame_names = tuple(
            dict.fromkeys(
                (
                    formula.response,
                    *(term.name for term in formula.parametric_terms),
                    *(
                        name
                        for spec in formula.smooth_terms
                        for name in (
                            *spec.variables,
                            *((spec.by,) if spec.by is not None else ()),
                        )
                    ),
                )
            )
        )
        frame = pd.DataFrame({name: padded[name] for name in frame_names}, copy=False)
        frame_retained_bytes = int(
            frame.memory_usage(index=True, deep=True).sum()
        ) + _deep_size(frame_names)
        if frame_retained_bytes > frame_retained_bound:
            _cleanup(paths)
            return CompactFrameUnavailable(
                "padded compact frame storage exceeds prospective bound",
                retained_table_bytes + frame_retained_bytes,
                budget.max_table_bytes,
                "table",
            )
        retained_group_bytes = retained_table_bytes
        retained_table_bytes += frame_retained_bytes
        construction_peak = max(construction_peak, retained_table_bytes)
        return CompactSetupFrame(
            frame=frame,
            groups=tuple(groups),
            name_group_indices=name_groups,
            selector_paths=tuple(paths),
            source_fingerprint=source_fingerprint,
            construction_peak_bytes=construction_peak,
            retained_table_bytes=retained_table_bytes,
            retained_group_bytes=retained_group_bytes,
            selector_peak_bytes=selector_peak,
            retained_selector_bytes=retained_selector_bytes,
            max_batch_bytes=budget.max_batch_bytes,
        )
    except BaseException:
        _cleanup(paths)
        raise
