"""Phase-1 construction of exact indexed designs from frozen predictors.

The descriptor deliberately replays a ``PredictSpec`` rather than calling a
smooth constructor on unique rows.  Re-running setup on repeated-value data
would change centering and normalization, which is not an exact
representation of the fitted design.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from jaxgam.data.discretize import (
    DiscretizationBudget,
    ExactIndexTable,
    ExactIndexUnavailable,
    exact_index_columns,
)
from jaxgam.data.source import RowSource
from jaxgam.formula import predict_matrix
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.predict_matrix import PredictSpec, build_predict_spec
from jaxgam.formula.prepare import PreparedModel
from jaxgam.smooths.by_variable import FactorBySmooth, NumericBySmooth
from jaxgam.smooths.constraints import TermBlock
from jaxgam.smooths.gaussian_process import GaussianProcessSmooth
from jaxgam.smooths.tensor import TensorInteractionSmooth, TensorProductSmooth
from jaxgam.smooths.tprs import TPRSSmooth
from jaxgam.smooths.utils import DISTANCE_BATCH_ROWS, row_tensor


def _readonly(value: npt.ArrayLike, dtype: npt.DTypeLike | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DiscreteDesignControl:
    """Bounded construction and bounded device-batch policy."""

    budget: DiscretizationBudget = field(default_factory=DiscretizationBudget)
    batch_rows: int = 8192
    selector_memmap_directory: Path | None = None
    pair_policy: Literal["histogram_or_direct", "direct_only"] = "histogram_or_direct"
    max_pair_histogram_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if isinstance(self.batch_rows, bool) or self.batch_rows <= 0:
            raise ValueError("batch_rows must be a positive integer")
        if self.pair_policy not in {"histogram_or_direct", "direct_only"}:
            raise ValueError(
                "pair_policy must be 'histogram_or_direct' or 'direct_only'"
            )
        if self.max_pair_histogram_bytes <= 0:
            raise ValueError("max_pair_histogram_bytes must be positive")


@dataclass(frozen=True)
class DiscreteDesignLineage:
    """The source and frozen basis identity a descriptor is allowed to replay."""

    source_fingerprint: str
    basis_fingerprint: str
    origin: Literal["prepared", "dense_setup", "predict_spec"]


@dataclass(frozen=True)
class DiscreteDesignUnavailable:
    """Typed bounded fallback; the caller may select the ordinary route."""

    reason: str
    required_bytes: int
    budget_bytes: int
    lineage: DiscreteDesignLineage


@dataclass(frozen=True)
class DiscreteMemoryPlan:
    """Conservative byte bounds for retained state and one action batch."""

    host_retained_bytes: int
    device_retained_bytes: int
    selector_retained_bytes: int
    selector_resident_bytes: int
    construction_peak_bytes: int
    host_batch_bytes: int
    device_action_bytes: int
    host_peak_bytes: int
    device_peak_bytes: int


@dataclass(frozen=True)
class LookupBlock:
    """One constrained non-tensor term table and its exact selector."""

    col_start: int
    table: npt.NDArray[np.float64]
    index: ExactIndexTable

    def __post_init__(self) -> None:
        table = np.asarray(self.table, dtype=np.float64)
        if table.ndim != 2 or table.shape[0] != self.index.n_unique:
            raise ValueError("lookup table must have one row per exact index value")
        if self.col_start < 0:
            raise ValueError("lookup block has a negative coefficient offset")
        object.__setattr__(self, "table", _readonly(table, np.float64))

    @property
    def n_coef(self) -> int:
        return self.table.shape[1]


@dataclass(frozen=True)
class TensorLookupBlock:
    """Separate marginal tables for an exact tensor-product term."""

    col_start: int
    margin_tables: tuple[npt.NDArray[np.float64], ...]
    margin_indices: tuple[ExactIndexTable, ...]
    centering: npt.NDArray[np.float64] | None
    keep: npt.NDArray[np.int32]
    multiplier_index: ExactIndexTable | None = None
    multiplier_table: npt.NDArray[np.float64] | None = None
    level_index: ExactIndexTable | None = None
    level_codes: npt.NDArray[np.int32] | None = None
    n_levels: int = 1

    def __post_init__(self) -> None:
        if len(self.margin_tables) < 2 or len(self.margin_tables) != len(
            self.margin_indices
        ):
            raise ValueError("tensor lookup needs matching separate marginal tables")
        tables = tuple(
            np.asarray(table, dtype=np.float64) for table in self.margin_tables
        )
        if any(table.ndim != 2 for table in tables):
            raise ValueError("tensor marginal tables must be matrices")
        if any(
            table.shape[0] != index.n_unique
            for table, index in zip(tables, self.margin_indices, strict=True)
        ):
            raise ValueError("tensor marginal table/index shapes are incompatible")
        raw_dim = int(np.prod([table.shape[1] for table in tables]))
        centering = (
            None
            if self.centering is None
            else np.asarray(self.centering, dtype=np.float64)
        )
        expanded_raw_dim = raw_dim * self.n_levels
        if self.n_levels <= 0:
            raise ValueError("tensor factor-by needs positive frozen level count")
        if centering is not None and centering.shape[0] != expanded_raw_dim:
            raise ValueError(
                "tensor centering transform has incompatible raw dimension"
            )
        transformed_dim = expanded_raw_dim if centering is None else centering.shape[1]
        keep = np.asarray(self.keep)
        if (
            keep.ndim != 1
            or keep.dtype.kind not in "iu"
            or np.any(keep < 0)
            or np.any(keep >= transformed_dim)
        ):
            raise ValueError("tensor retained columns are invalid")
        object.__setattr__(
            self,
            "margin_tables",
            tuple(_readonly(table, np.float64) for table in tables),
        )
        object.__setattr__(
            self,
            "centering",
            None if centering is None else _readonly(centering, np.float64),
        )
        object.__setattr__(self, "keep", _readonly(keep, np.int32))
        if self.multiplier_index is not None:
            if self.multiplier_table is None:
                raise ValueError("numeric tensor by needs a multiplier table")
            multiplier = np.asarray(self.multiplier_table, dtype=np.float64)
            if multiplier.shape != (self.multiplier_index.n_unique,):
                raise ValueError("tensor multiplier table has incompatible index")
            object.__setattr__(self, "multiplier_table", _readonly(multiplier))
        elif self.multiplier_table is not None:
            raise ValueError("tensor multiplier table needs an exact index")
        if self.level_index is not None:
            codes = np.asarray(self.level_codes)
            if (
                codes.shape != (self.level_index.n_unique,)
                or codes.dtype.kind not in "iu"
            ):
                raise ValueError("tensor factor-by level codes have incompatible index")
            if np.any(codes < -1) or np.any(codes >= self.n_levels):
                raise ValueError("tensor factor-by level code is out of range")
            object.__setattr__(self, "level_codes", _readonly(codes, np.int32))
        elif self.level_codes is not None or self.n_levels != 1:
            raise ValueError("tensor factor-by codes need an exact factor index")

    @property
    def raw_n_coef(self) -> int:
        return int(np.prod([table.shape[1] for table in self.margin_tables]))

    @property
    def n_coef(self) -> int:
        return len(self.keep)


DiscreteBlock = LookupBlock | TensorLookupBlock


@dataclass(frozen=True)
class FrozenDiscreteDesign:
    """CPU descriptor containing tables and selectors, never an n-by-p design."""

    blocks: tuple[DiscreteBlock, ...]
    n_rows: int
    n_coef: int
    lineage: DiscreteDesignLineage
    control: DiscreteDesignControl
    memory_plan: DiscreteMemoryPlan | None = None

    def _validate_batch_positions(self, row_positions: npt.ArrayLike) -> np.ndarray:
        positions = np.asarray(row_positions)
        if positions.ndim != 1 or positions.dtype.kind not in "iu":
            raise TypeError(
                "batch row positions must be a one-dimensional integer array"
            )
        if len(positions) > self.control.batch_rows:
            raise ValueError(
                "batch row count exceeds the descriptor's configured batch_rows"
            )
        return positions

    def validate_source(self, source: RowSource) -> None:
        if source.fingerprint() != self.lineage.source_fingerprint:
            raise RuntimeError(
                "RowSource changed after discrete preparation; prepare again."
            )

    def batch_indices(
        self, row_positions: npt.ArrayLike
    ) -> tuple[tuple[np.ndarray, ...], ...]:
        """Fetch only one host batch of selectors for a later JIT dispatch."""
        positions = self._validate_batch_positions(row_positions)
        gathered: dict[int, np.ndarray] = {}

        def selector(index: ExactIndexTable) -> np.ndarray:
            identity = id(index)
            result = gathered.get(identity)
            if result is None:
                result = index.batch_selector(positions)
                gathered[identity] = result
            return result

        result: list[tuple[np.ndarray, ...]] = []
        for block in self.blocks:
            if isinstance(block, LookupBlock):
                result.append((selector(block.index),))
                continue
            block_indices = list(block.margin_indices)
            if block.multiplier_index is not None:
                block_indices.append(block.multiplier_index)
            if block.level_index is not None:
                block_indices.append(block.level_index)
            result.append(tuple(selector(index) for index in block_indices))
        return tuple(result)

    def evaluate_batch(self, row_positions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """CPU oracle for tests and descriptor validation; bounded by batch rows."""
        positions = self._validate_batch_positions(row_positions)
        selectors = self.batch_indices(positions)
        pieces: list[np.ndarray] = []
        for block, block_selectors in zip(self.blocks, selectors, strict=True):
            if isinstance(block, LookupBlock):
                pieces.append(block.table[block_selectors[0]])
                continue
            rows = [
                table[selector]
                for table, selector in zip(
                    block.margin_tables,
                    block_selectors[: len(block.margin_indices)],
                    strict=True,
                )
            ]
            matrix = rows[0]
            for row in rows[1:]:
                matrix = row_tensor(matrix, row)
            if block.level_index is None and block.centering is not None:
                matrix = matrix @ block.centering
            if block.level_index is None:
                matrix = matrix[:, block.keep]
            if block.multiplier_index is not None:
                assert block.multiplier_table is not None
                multiplier_number = len(block.margin_indices)
                matrix = (
                    matrix
                    * block.multiplier_table[block_selectors[multiplier_number]][
                        :, None
                    ]
                )
            if block.level_index is not None:
                assert block.level_codes is not None
                base = matrix
                raw = np.zeros((len(positions), block.n_levels * base.shape[1]))
                codes = block.level_codes[block_selectors[-1]]
                for level in range(block.n_levels):
                    raw[
                        codes == level,
                        level * base.shape[1] : (level + 1) * base.shape[1],
                    ] = base[codes == level]
                matrix = raw if block.centering is None else raw @ block.centering
                matrix = matrix[:, block.keep]
            pieces.append(matrix)
        result = np.column_stack(pieces) if pieces else np.empty((len(positions), 0))
        if result.shape[1] != self.n_coef:
            raise RuntimeError(
                "discrete descriptor produced an incompatible coefficient width"
            )
        return result


def _hash_frozen(value: object, digest: hashlib._Hash, seen: set[int]) -> None:
    """Hash prediction metadata without lossy object/array repr output."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        digest.update(type(value).__qualname__.encode())
        digest.update(repr(value).encode())
        return
    if isinstance(value, np.ndarray):
        digest.update(b"array\0")
        digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode())
        if value.dtype.hasobject:
            digest.update(b"object-values\0")
            digest.update(str(value.size).encode())
            for item in value.flat:
                _hash_frozen(item, digest, seen)
                digest.update(b"item-end\0")
            return
        digest.update(np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, np.generic):
        digest.update(b"numpy-scalar\0")
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
        return
    if isinstance(value, (tuple, list)):
        digest.update(type(value).__qualname__.encode())
        digest.update(str(len(value)).encode())
        for item in value:
            _hash_frozen(item, digest, seen)
            digest.update(b"item-end\0")
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        digest.update(str(len(value)).encode())
        for key in sorted(value, key=str):
            _hash_frozen(key, digest, seen)
            digest.update(b"key-end\0")
            _hash_frozen(value[key], digest, seen)
            digest.update(b"value-end\0")
        return
    if isinstance(value, (set, frozenset)):
        digest.update(type(value).__qualname__.encode())
        for item in sorted(value, key=repr):
            _hash_frozen(item, digest, seen)
        return
    if isinstance(value, Path):
        digest.update(b"path\0")
        digest.update(str(value).encode())
        return
    identity = id(value)
    if identity in seen:
        digest.update(b"cycle\0")
        return
    seen.add(identity)
    digest.update(f"{type(value).__module__}.{type(value).__qualname__}".encode())
    if hasattr(value, "__dict__"):
        members = vars(value).items()
    else:
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        members = (
            (name, getattr(value, name)) for name in slots if hasattr(value, name)
        )
        if not slots:
            raise TypeError(
                f"unsupported non-deterministic frozen metadata: {type(value)!r}"
            )
    for name, item in sorted(members):
        # PredictSpec promises to discard these training caches.  Excluding
        # them makes the hash test exactly its frozen prediction contract.
        if name in {"_X", "_S", "_penalties", "_E_knot"}:
            continue
        digest.update(name.encode())
        digest.update(b"member-name-end\0")
        _hash_frozen(item, digest, seen)
        digest.update(b"member-value-end\0")


def _basis_token(spec: PredictSpec) -> str:
    """Content hash of the full frozen prediction graph and transforms."""
    digest = hashlib.sha256(b"jaxgam-frozen-predict-spec-v1\0")
    _hash_frozen(spec, digest, set())
    return digest.hexdigest()


def _selector_path(
    control: DiscreteDesignControl, label: str, sequence: int
) -> Path | None:
    if control.selector_memmap_directory is None:
        return None
    return control.selector_memmap_directory / f"jaxgam-discrete-{sequence}-{label}.npy"


class _Ledger:
    def __init__(self, control: DiscreteDesignControl) -> None:
        self.control = control
        self.table_bytes = 0
        self.selector_bytes = 0
        self.selector_storage_bytes = 0
        self.construction_peak_bytes = 0
        self.sequence = 0
        self.owned_memmaps: list[Path] = []
        self.indices: dict[tuple[int, tuple[str, ...]], ExactIndexTable] = {}

    def cleanup(self) -> None:
        """Remove only selector files claimed by this failed construction."""
        for path in self.owned_memmaps:
            if path.exists():
                path.unlink()
        self.owned_memmaps.clear()

    def indexed(
        self,
        source: RowSource,
        names: tuple[str, ...],
        estimated_width: int,
        label: str,
    ) -> ExactIndexTable | DiscreteDesignUnavailable:
        cache_key = (id(source), names)
        cached = self.indices.get(cache_key)
        if cached is not None:
            return cached
        selector_bytes = source.n_rows * np.dtype(np.int32).itemsize
        memmap = _selector_path(self.control, label, self.sequence)
        self.sequence += 1
        validation_bytes = source.n_rows * np.dtype(bool).itemsize
        if (
            memmap is None
            and self.selector_bytes + selector_bytes + validation_bytes
            > self.control.budget.max_selector_bytes
        ):
            return DiscreteDesignUnavailable(
                "combined in-memory selectors exceed budget",
                self.selector_bytes + selector_bytes + validation_bytes,
                self.control.budget.max_selector_bytes,
                _PLACEHOLDER_LINEAGE,
            )
        remaining = self.control.budget.max_table_bytes - self.table_bytes
        cap = remaining // max(1, estimated_width * np.dtype(np.float64).itemsize)
        if cap <= 0:
            return DiscreteDesignUnavailable(
                "lookup tables exceed budget before unique-value allocation",
                self.table_bytes + max(1, estimated_width) * 8,
                self.control.budget.max_table_bytes,
                _PLACEHOLDER_LINEAGE,
            )
        indexed = exact_index_columns(
            source,
            names,
            budget=self.control.budget,
            max_unique_rows=int(cap),
            selector_memmap=memmap,
            batch_rows=self.control.batch_rows,
            max_construction_bytes=remaining,
        )
        if isinstance(indexed, ExactIndexUnavailable):
            return DiscreteDesignUnavailable(
                indexed.reason,
                self.table_bytes + indexed.required_bytes,
                self.control.budget.max_table_bytes,
                _PLACEHOLDER_LINEAGE,
            )
        if memmap is not None:
            self.owned_memmaps.append(memmap)
        persistent_values = indexed.value_bytes
        self.construction_peak_bytes = max(
            self.construction_peak_bytes,
            self.table_bytes + indexed.construction_peak_bytes,
        )
        if self.table_bytes + persistent_values > self.control.budget.max_table_bytes:
            return DiscreteDesignUnavailable(
                "retained unique values exceed lookup-table budget",
                self.table_bytes + persistent_values,
                self.control.budget.max_table_bytes,
                _PLACEHOLDER_LINEAGE,
            )
        self.table_bytes += persistent_values
        if memmap is None:
            self.selector_bytes += indexed.selector_bytes
        self.selector_storage_bytes += indexed.selector_bytes
        self.indices[cache_key] = indexed
        return indexed

    def add_table(
        self,
        n_bytes: int,
        lineage: DiscreteDesignLineage,
        *,
        live_temporary_bytes: int | None = None,
        reason: str = "evaluated lookup table exceeds budget",
    ) -> DiscreteDesignUnavailable | None:
        if n_bytes < 0:
            raise ValueError("retained table bytes cannot be negative")
        # The descriptor constructor takes an owned read-only copy.  Unless a
        # caller supplies a larger raw/transform peak, the source array and its
        # retained copy coexist here.
        temporary = n_bytes if live_temporary_bytes is None else live_temporary_bytes
        required = self.table_bytes + temporary + n_bytes
        self.construction_peak_bytes = max(self.construction_peak_bytes, required)
        if required > self.control.budget.max_table_bytes:
            return DiscreteDesignUnavailable(
                reason,
                required,
                self.control.budget.max_table_bytes,
                lineage,
            )
        self.table_bytes += n_bytes
        return None

    def add_metadata(
        self, n_bytes: int, lineage: DiscreteDesignLineage
    ) -> DiscreteDesignUnavailable | None:
        return self.add_table(
            n_bytes,
            lineage,
            reason="retained discrete transform metadata exceeds budget",
        )

    def check_temporary(
        self,
        n_bytes: int,
        lineage: DiscreteDesignLineage,
        reason: str,
    ) -> DiscreteDesignUnavailable | None:
        if n_bytes < 0:
            raise ValueError("temporary bytes cannot be negative")
        required = self.table_bytes + n_bytes
        self.construction_peak_bytes = max(self.construction_peak_bytes, required)
        if required > self.control.budget.max_table_bytes:
            return DiscreteDesignUnavailable(
                reason,
                required,
                self.control.budget.max_table_bytes,
                lineage,
            )
        return None


_PLACEHOLDER_LINEAGE = DiscreteDesignLineage("", "", "predict_spec")


def _unavailable(
    value: DiscreteDesignUnavailable, lineage: DiscreteDesignLineage
) -> DiscreteDesignUnavailable:
    return DiscreteDesignUnavailable(
        value.reason, value.required_bytes, value.budget_bytes, lineage
    )


def _term_names(term: TermBlock) -> tuple[str, ...]:
    smooth = term.smooth
    if smooth is None:
        return ()
    names = list(smooth.spec.variables)
    if smooth.spec.by is not None:
        names.append(smooth.spec.by)
    return tuple(dict.fromkeys(names))


def _index_values(index: ExactIndexTable) -> dict[str, npt.NDArray]:
    return dict(zip(index.names, index.values, strict=True))


def _prospective_matrix_bytes(
    n_rows: int,
    raw_width: int,
    transformed_widths: tuple[int, ...],
    *,
    evaluator_bytes: int = 0,
) -> int:
    """Bound predictor evaluation, sequential transforms and ownership copy."""
    float_bytes = np.dtype(np.float64).itemsize
    raw_bytes = n_rows * raw_width * float_bytes
    # Smooth/parametric predictors commonly hold an output plus basis/work
    # arrays of comparable width.  Eight arrays is a conservative, static
    # evaluator allowance for the currently supported dense predictors.
    peak = max(8 * raw_bytes, evaluator_bytes)
    previous = raw_width
    for width in transformed_widths:
        peak = max(peak, n_rows * (previous + width) * float_bytes)
        previous = width
    # The descriptor constructor owns a read-only copy while the evaluated
    # source table is still live.
    peak = max(peak, 2 * n_rows * previous * float_bytes)
    return peak


def _smooth_evaluator_bytes(smooth: object, n_rows: int, raw_width: int) -> int:
    """Bound frozen smooth prediction scratch from evaluator metadata."""
    float_bytes = np.dtype(np.float64).itemsize
    bool_bytes = np.dtype(bool).itemsize
    if isinstance(smooth, (NumericBySmooth, FactorBySmooth)):
        base = smooth.base_smooth
        base_peak = _smooth_evaluator_bytes(base, n_rows, base.n_coefs)
        # The wrapper retains the base prediction while allocating its numeric
        # product or factor-expanded coefficient block and masks/codes.
        wrapper_bytes = n_rows * raw_width * float_bytes + 2 * n_rows * bool_bytes
        return max(8 * n_rows * raw_width * float_bytes, base_peak + wrapper_bytes)
    if isinstance(smooth, TPRSSmooth):
        knots = np.asarray(smooth._Xu)
        projection = np.asarray(smooth._UZ)
        dimension = knots.shape[1]
        n_knots = knots.shape[0]
        null_width = projection.shape[0] - n_knots
        batch = min(n_rows, DISTANCE_BATCH_ROWS)
        persistent = n_rows * (2 * dimension + raw_width) * float_bytes
        # cdist, semi-kernel powers/log/masks, polynomial columns, concatenated
        # projection input and matrix-product output can overlap within a chunk.
        distance_work = batch * n_knots * (10 * float_bytes + bool_bytes)
        polynomial_work = batch * max(1, 4 * null_width) * float_bytes
        projection_work = batch * (n_knots + null_width + raw_width) * float_bytes
        return persistent + distance_work + polynomial_work + projection_work
    if isinstance(smooth, GaussianProcessSmooth):
        knots = np.asarray(smooth._knt)
        projection = np.asarray(smooth._UZ)
        dimension = knots.shape[1]
        n_knots = knots.shape[0]
        rank = projection.shape[1]
        null_width = raw_width - rank
        batch = min(n_rows, DISTANCE_BATCH_ROWS)
        persistent = n_rows * (2 * dimension + raw_width) * float_bytes
        # The most demanding registered kernels retain scaled distance,
        # exponential and polynomial temporaries alongside the kernel result.
        kernel_work = batch * n_knots * 10 * float_bytes
        projection_work = batch * (n_knots + rank + max(1, null_width)) * float_bytes
        return persistent + kernel_work + projection_work
    return 8 * n_rows * raw_width * float_bytes


def _constant_index(
    source: RowSource, ledger: _Ledger, lineage: DiscreteDesignLineage
) -> ExactIndexTable | DiscreteDesignUnavailable:
    """Return the one-row exact selector used by an intercept-only block."""
    selector_bytes = source.n_rows * np.dtype(np.int32).itemsize
    if (
        ledger.selector_bytes + selector_bytes
        > ledger.control.budget.max_selector_bytes
    ):
        return DiscreteDesignUnavailable(
            "combined in-memory selectors exceed budget",
            ledger.selector_bytes + selector_bytes,
            ledger.control.budget.max_selector_bytes,
            lineage,
        )
    ledger.selector_bytes += selector_bytes
    ledger.selector_storage_bytes += selector_bytes
    failed = ledger.add_metadata(np.dtype(np.int8).itemsize, lineage)
    if failed is not None:
        return failed
    return ExactIndexTable(
        ("__intercept__",),
        (np.array((0,), dtype=np.int8),),
        np.zeros(source.n_rows, dtype=np.int32),
        source.n_rows,
        "memory",
    )


def _make_tensor_block(
    term: TermBlock,
    source: RowSource,
    ledger: _Ledger,
    lineage: DiscreteDesignLineage,
    numeric_by: NumericBySmooth | None = None,
    factor_by: FactorBySmooth | None = None,
) -> TensorLookupBlock | DiscreteDesignUnavailable:
    smooth = (
        term.smooth
        if numeric_by is None and factor_by is None
        else (numeric_by or factor_by).base_smooth
    )
    assert isinstance(smooth, TensorProductSmooth)
    margin_widths: list[int] = []
    for margin_number, marginal in enumerate(smooth._marginals):
        width = marginal.n_coefs
        if isinstance(smooth, TensorInteractionSmooth):
            width = smooth._Z_list[margin_number].shape[1]
        xp = smooth._XP_list[margin_number]
        if xp is not None:
            width = xp.shape[1]
        margin_widths.append(width)
    raw_width = int(np.prod(margin_widths))
    transformed_width = (
        raw_width * factor_by.n_levels
        if factor_by is not None and term.Z_centering is None
        else smooth.n_coefs
        if term.Z_centering is None
        else term.Z_centering.shape[1]
    )
    keep = (
        np.arange(transformed_width, dtype=np.int32)
        if not term.del_index
        else np.asarray(
            [i for i in range(transformed_width) if i not in term.del_index],
            dtype=np.int32,
        )
    )
    # Reserve retained frozen transforms before any marginal evaluator runs.
    for metadata in (term.Z_centering, keep):
        if metadata is None:
            continue
        failed = ledger.add_metadata(int(metadata.nbytes), lineage)
        if failed is not None:
            return failed
    tables: list[np.ndarray] = []
    indices: list[ExactIndexTable] = []
    for margin_number, (variable, marginal) in enumerate(
        zip(smooth.spec.variables, smooth._marginals, strict=True)
    ):
        width = marginal.n_coefs
        indexed = ledger.indexed(
            source, (variable,), width, f"tensor-{term.label}-{margin_number}"
        )
        if isinstance(indexed, DiscreteDesignUnavailable):
            return _unavailable(indexed, lineage)
        transform_widths: list[int] = []
        if isinstance(smooth, TensorInteractionSmooth):
            transform_widths.append(smooth._Z_list[margin_number].shape[1])
        xp = smooth._XP_list[margin_number]
        if xp is not None:
            transform_widths.append(xp.shape[1])
        prospective = _prospective_matrix_bytes(
            indexed.n_unique,
            marginal.n_coefs,
            tuple(transform_widths),
            evaluator_bytes=_smooth_evaluator_bytes(
                marginal, indexed.n_unique, marginal.n_coefs
            ),
        )
        failed = ledger.check_temporary(
            prospective,
            lineage,
            "tensor marginal predictor workspace exceeds budget",
        )
        if failed is not None:
            return failed
        table = marginal.predict_matrix({variable: indexed.values[0]})
        if isinstance(smooth, TensorInteractionSmooth):
            table = table @ smooth._Z_list[margin_number]
        if xp is not None:
            table = table @ xp
        failed = ledger.add_table(
            int(table.nbytes),
            lineage,
            live_temporary_bytes=int(table.nbytes),
        )
        if failed is not None:
            return failed
        tables.append(table)
        indices.append(indexed)
    multiplier_index = None
    multiplier_table = None
    level_index = None
    level_codes = None
    if numeric_by is not None:
        multiplier_index = ledger.indexed(
            source, (numeric_by.by_variable,), 1, f"tensor-{term.label}-by"
        )
        if isinstance(multiplier_index, DiscreteDesignUnavailable):
            return _unavailable(multiplier_index, lineage)
        multiplier_table = np.asarray(multiplier_index.values[0], dtype=np.float64)
        failed = ledger.add_metadata(int(multiplier_table.nbytes), lineage)
        if failed is not None:
            return failed
    if factor_by is not None:
        level_index = ledger.indexed(
            source, (factor_by.by_variable,), 1, f"tensor-{term.label}-level"
        )
        if isinstance(level_index, DiscreteDesignUnavailable):
            return _unavailable(level_index, lineage)
        mapping = {value: number for number, value in enumerate(factor_by.levels)}
        unknown = [
            value
            for value in level_index.values[0]
            if value not in factor_by.all_levels
        ]
        if unknown:
            raise ValueError(
                "factor tensor replay contains levels absent from frozen "
                "prediction state"
            )
        level_codes = np.asarray(
            [mapping.get(value, -1) for value in level_index.values[0]], dtype=np.int32
        )
        failed = ledger.add_metadata(int(level_codes.nbytes), lineage)
        if failed is not None:
            return failed
    # TensorLookupBlock takes owned copies after all source tables and frozen
    # transform arrays have been assembled.  Count those still-live sources on
    # top of the retained copies already reserved above.
    source_copy_bytes = sum(int(table.nbytes) for table in tables) + int(keep.nbytes)
    if term.Z_centering is not None:
        source_copy_bytes += int(term.Z_centering.nbytes)
    if multiplier_table is not None:
        source_copy_bytes += int(multiplier_table.nbytes)
    if level_codes is not None:
        source_copy_bytes += int(level_codes.nbytes)
    failed = ledger.check_temporary(
        source_copy_bytes,
        lineage,
        "tensor table/transform ownership copy exceeds budget",
    )
    if failed is not None:
        return failed
    return TensorLookupBlock(
        term.col_start,
        tuple(tables),
        tuple(indices),
        term.Z_centering,
        keep,
        multiplier_index,
        multiplier_table,
        level_index,
        level_codes,
        1 if factor_by is None else factor_by.n_levels,
    )


def _block_device_bytes(block: DiscreteBlock) -> int:
    if isinstance(block, LookupBlock):
        return int(block.table.nbytes)
    total = sum(int(table.nbytes) for table in block.margin_tables)
    total += int(block.keep.nbytes)
    if block.centering is not None:
        total += int(block.centering.nbytes)
    if block.multiplier_table is not None:
        total += int(block.multiplier_table.nbytes)
    if block.level_codes is not None:
        total += int(block.level_codes.nbytes)
    return total


def _block_indices(block: DiscreteBlock) -> tuple[ExactIndexTable, ...]:
    if isinstance(block, LookupBlock):
        return (block.index,)
    return (
        *block.margin_indices,
        *((block.multiplier_index,) if block.multiplier_index is not None else ()),
        *((block.level_index,) if block.level_index is not None else ()),
    )


def _block_materialized_width(block: DiscreteBlock) -> int:
    if isinstance(block, LookupBlock):
        return block.n_coef
    return max(block.raw_n_coef * block.n_levels, block.n_coef)


def _block_transform_width(block: DiscreteBlock) -> int:
    if isinstance(block, LookupBlock):
        return block.n_coef
    if block.centering is None:
        return block.n_coef
    return int(block.centering.shape[1])


def _final_table_shape(block: DiscreteBlock) -> tuple[int, int]:
    table = block.table if isinstance(block, LookupBlock) else block.margin_tables[-1]
    return int(table.shape[0]), int(table.shape[1])


def _memory_plan(
    blocks: tuple[DiscreteBlock, ...],
    n_coef: int,
    ledger: _Ledger,
    control: DiscreteDesignControl,
) -> DiscreteMemoryPlan:
    """Compute static host/device bounds for one configured action batch.

    The bound is deliberately conservative.  It includes caller-visible
    inputs/outputs and the largest live kernel temporary, while pair and tensor
    subblocks are sequential rather than charged as if all pairs coexist.
    """
    float_bytes = np.dtype(np.float64).itemsize
    int_bytes = np.dtype(np.int32).itemsize
    pointer_bytes = np.dtype(np.intp).itemsize
    batch_rows = control.batch_rows
    selector_reference_count = sum(len(_block_indices(block)) for block in blocks)
    unique_selector_count = len(
        {id(index) for block in blocks for index in _block_indices(block)}
    )
    # Host dispatch owns row positions and one gathered int32 array per compact
    # selector.  Include tuple/reference metadata for the nested block layout.
    host_batch_bytes = (
        batch_rows * pointer_bytes
        + batch_rows * unique_selector_count * int_bytes
        + (len(blocks) + selector_reference_count) * 4 * pointer_bytes
    )
    # Device dispatch may materialize each repeated PyTree leaf separately, so
    # retain the conservative per-reference charge for action planning.
    selector_batch_bytes = batch_rows * selector_reference_count * int_bytes
    p_vector_bytes = n_coef * float_bytes
    p_matrix_bytes = n_coef * n_coef * float_bytes
    batch_vector_bytes = batch_rows * float_bytes
    widest_materialized = max(
        (_block_materialized_width(block) for block in blocks), default=0
    )
    largest_raw_core = max(
        (
            block.raw_n_coef * block.n_levels * float_bytes
            for block in blocks
            if isinstance(block, TensorLookupBlock)
        ),
        default=0,
    )

    pair_peak = 0
    for left_number, left in enumerate(blocks):
        q_left, k_left = _final_table_shape(left)
        for right in blocks[left_number:]:
            q_right, k_right = _final_table_shape(right)
            histogram_bytes = q_left * q_right * float_bytes
            direct_selection_bytes = batch_rows * (k_left + k_right) * float_bytes
            direct_bytes = (
                batch_rows * (k_left + 2 * k_right) * float_bytes
                + k_left * k_right * float_bytes
            )
            use_histogram = (
                control.pair_policy == "histogram_or_direct"
                and histogram_bytes <= control.max_pair_histogram_bytes
                and histogram_bytes <= direct_selection_bytes
            )
            histogram_work_bytes = (
                2 * histogram_bytes
                + k_left * q_right * float_bytes
                + k_left * k_right * float_bytes
            )
            reduction_bytes = histogram_work_bytes if use_histogram else direct_bytes
            # Tensor prefixes and final-margin maps are produced one pair at a
            # time.  Ordinary maps are identities of their small final table.
            map_bytes = (
                k_left * (_block_transform_width(left) + left.n_coef)
                + k_right * (_block_transform_width(right) + right.n_coef)
            ) * float_bytes
            part_bytes = left.n_coef * right.n_coef * float_bytes
            contraction_bytes = left.n_coef * k_right * float_bytes
            pair_peak = max(
                pair_peak,
                reduction_bytes
                + 5 * batch_vector_bytes
                + batch_rows * int_bytes
                + map_bytes
                + contraction_bytes
                + 3 * part_bytes,
            )

    xbd_bytes = (
        selector_batch_bytes
        + p_vector_bytes
        + batch_vector_bytes
        + largest_raw_core
        + batch_rows * widest_materialized * float_bytes
    )
    adjoint_bytes = xbd_bytes + p_vector_bytes
    xwxd_bytes = (
        selector_batch_bytes
        + batch_vector_bytes
        + 2 * p_matrix_bytes
        + largest_raw_core
        + pair_peak
    )
    diag_bytes = (
        selector_batch_bytes
        + p_matrix_bytes
        + batch_vector_bytes
        + 2 * batch_rows * widest_materialized * float_bytes
        + batch_rows * max(1, widest_materialized) * float_bytes
    )
    device_retained_bytes = sum(_block_device_bytes(block) for block in blocks)
    device_action_bytes = max(xbd_bytes, adjoint_bytes, xwxd_bytes, diag_bytes)
    return DiscreteMemoryPlan(
        host_retained_bytes=ledger.table_bytes,
        device_retained_bytes=device_retained_bytes,
        selector_retained_bytes=ledger.selector_storage_bytes,
        selector_resident_bytes=ledger.selector_bytes,
        construction_peak_bytes=ledger.construction_peak_bytes,
        host_batch_bytes=host_batch_bytes,
        device_action_bytes=device_action_bytes,
        host_peak_bytes=ledger.table_bytes + ledger.selector_bytes + host_batch_bytes,
        device_peak_bytes=device_retained_bytes + device_action_bytes,
    )


def build_discrete_design_from_predict_spec(
    spec: PredictSpec,
    source: RowSource,
    *,
    lineage: DiscreteDesignLineage | None = None,
    control: DiscreteDesignControl | None = None,
) -> FrozenDiscreteDesign | DiscreteDesignUnavailable:
    """Build an exact descriptor from an already-frozen prediction graph."""
    if control is None:
        control = DiscreteDesignControl()
    actual_lineage = lineage or DiscreteDesignLineage(
        source.fingerprint(), _basis_token(spec), "predict_spec"
    )
    if source.fingerprint() != actual_lineage.source_fingerprint:
        raise RuntimeError(
            "RowSource changed before discrete preparation; prepare again."
        )
    ledger = _Ledger(control)

    def unavailable(value: DiscreteDesignUnavailable) -> DiscreteDesignUnavailable:
        ledger.cleanup()
        return _unavailable(value, actual_lineage)

    blocks: list[DiscreteBlock] = []
    for term in spec.coef_map.terms:
        if term.term_type == "parametric":
            if term.n_coefs == 0:
                continue
            names = tuple(param.name for param in spec.parametric_terms)
            try:
                if not names:
                    indexed = _constant_index(source, ledger, actual_lineage)
                else:
                    indexed = ledger.indexed(source, names, term.n_coefs, "parametric")
            except BaseException:
                ledger.cleanup()
                raise
            if isinstance(indexed, DiscreteDesignUnavailable):
                return unavailable(indexed)
            raw_width = max(term.n_coefs_raw, term.n_coefs)
            transformed_widths = (
                (term.n_coefs,)
                if raw_width != term.n_coefs or spec.dropped_param_names
                else ()
            )
            failed = ledger.check_temporary(
                _prospective_matrix_bytes(
                    indexed.n_unique, raw_width, transformed_widths
                ),
                actual_lineage,
                "parametric predictor workspace exceeds budget",
            )
            if failed is not None:
                return unavailable(failed)
            try:
                if names:
                    raw, _ = predict_matrix._build_parametric_matrix(
                        spec.parametric_terms,
                        _index_values(indexed),
                        spec.has_intercept,
                        indexed.n_unique,
                        factor_info=spec.factor_info,
                        ordered_factors=spec.ordered_factors,
                    )
                else:
                    raw = np.ones((1, 1), dtype=np.float64)
            except BaseException:
                ledger.cleanup()
                raise
            if spec.dropped_param_names:
                raw = raw[:, list(spec.parametric_keep_cols)]
            failed = ledger.add_table(int(raw.nbytes), actual_lineage)
            if failed is not None:
                ledger.cleanup()
                return failed
            blocks.append(LookupBlock(term.col_start, raw, indexed))
            continue
        if term.smooth is None:
            ledger.cleanup()
            return DiscreteDesignUnavailable(
                "term has no frozen smooth evaluator", 0, 0, actual_lineage
            )
        tensor_base = (
            term.smooth.base_smooth
            if isinstance(term.smooth, (NumericBySmooth, FactorBySmooth))
            else term.smooth
        )
        if isinstance(tensor_base, TensorProductSmooth):
            try:
                tensor = _make_tensor_block(
                    term,
                    source,
                    ledger,
                    actual_lineage,
                    term.smooth if isinstance(term.smooth, NumericBySmooth) else None,
                    term.smooth if isinstance(term.smooth, FactorBySmooth) else None,
                )
            except BaseException:
                ledger.cleanup()
                raise
            if isinstance(tensor, DiscreteDesignUnavailable):
                ledger.cleanup()
                return tensor
            blocks.append(tensor)
            continue
        names = _term_names(term)
        try:
            indexed = ledger.indexed(source, names, term.n_coefs, term.label)
        except BaseException:
            ledger.cleanup()
            raise
        if isinstance(indexed, DiscreteDesignUnavailable):
            return unavailable(indexed)
        transform_widths: list[int] = []
        if term.Z_centering is not None:
            transform_widths.append(term.Z_centering.shape[1])
        if term.del_index:
            transform_widths.append(term.n_coefs)
        failed = ledger.check_temporary(
            _prospective_matrix_bytes(
                indexed.n_unique,
                term.n_coefs_raw,
                tuple(transform_widths),
                evaluator_bytes=_smooth_evaluator_bytes(
                    term.smooth, indexed.n_unique, term.n_coefs_raw
                ),
            ),
            actual_lineage,
            "smooth predictor workspace exceeds budget",
        )
        if failed is not None:
            return unavailable(failed)
        try:
            raw = term.smooth.predict_matrix(_index_values(indexed))
            table = spec.coef_map.transform_X(raw, term)
        except BaseException:
            ledger.cleanup()
            raise
        failed = ledger.add_table(
            int(table.nbytes),
            actual_lineage,
            live_temporary_bytes=int(raw.nbytes + table.nbytes),
        )
        if failed is not None:
            ledger.cleanup()
            return failed
        blocks.append(LookupBlock(term.col_start, table, indexed))
    block_tuple = tuple(blocks)
    memory_plan = _memory_plan(block_tuple, spec.total_coefs, ledger, control)
    batch_bytes = max(memory_plan.host_batch_bytes, memory_plan.device_action_bytes)
    if batch_bytes > control.budget.max_batch_bytes:
        ledger.cleanup()
        return DiscreteDesignUnavailable(
            "indexed operator batch workspace exceeds budget",
            batch_bytes,
            control.budget.max_batch_bytes,
            actual_lineage,
        )
    result = FrozenDiscreteDesign(
        block_tuple,
        source.n_rows,
        spec.total_coefs,
        actual_lineage,
        control,
        memory_plan,
    )
    if source.fingerprint() != actual_lineage.source_fingerprint:
        ledger.cleanup()
        raise RuntimeError(
            "RowSource changed during discrete preparation; prepare again."
        )
    return result


def build_discrete_design_from_prepared(
    prepared: PreparedModel,
    source: RowSource,
    *,
    control: DiscreteDesignControl | None = None,
) -> FrozenDiscreteDesign | DiscreteDesignUnavailable:
    """Adapt only PR5's already-supported prepared setup to a descriptor."""
    return build_discrete_design_from_predict_spec(
        prepared.predict_spec,
        source,
        lineage=DiscreteDesignLineage(
            prepared.source_fingerprint, prepared.basis_fingerprint, "prepared"
        ),
        control=control,
    )


def build_discrete_design_from_setup(
    setup: ModelSetup,
    source: RowSource,
    *,
    control: DiscreteDesignControl | None = None,
) -> FrozenDiscreteDesign | DiscreteDesignUnavailable:
    """Adapt a dense setup separately; it does not expand streamed eligibility."""
    if source.n_rows != setup.n_obs:
        raise ValueError("dense setup and RowSource have different observation counts")
    spec = build_predict_spec(setup)
    basis = hashlib.sha256(setup.X.tobytes()).hexdigest()
    return build_discrete_design_from_predict_spec(
        spec,
        source,
        lineage=DiscreteDesignLineage(source.fingerprint(), basis, "dense_setup"),
        control=control,
    )
