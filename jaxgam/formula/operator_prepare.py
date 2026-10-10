"""Bounded CPU preparation for cubic and indexed random-effect operators.

The modest submodel uses the existing exact preparation/constraint metadata.
Random effects never pass through their dense smooth constructor. Interaction
dimensions are symbolic; only one level ID and amplitude per batch row exist.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from jaxgam.data.source import RowBatch, RowSource
from jaxgam.formula.prepare import PreparedModel, prepare_model
from jaxgam.formula.terms import FormulaSpec
from jaxgam.penalties.structure import (
    IdentityPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
from jaxgam.smooths.utils import is_factor


@dataclass(frozen=True)
class OperatorPreparationControl:
    """Hard bounds independent of the global coefficient dimension."""

    batch_rows: int = 4096
    memory_budget_bytes: int = 128 * 1024 * 1024
    max_modest_coefficients: int = 64

    def __post_init__(self) -> None:
        for name in ("batch_rows", "memory_budget_bytes", "max_modest_coefficients"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class FactorEncoding:
    """Frozen levels with a shared immutable lookup and pickle ownership."""

    name: str
    levels: tuple[Any, ...]
    lookup: Mapping[Any, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "lookup", MappingProxyType({v: i for i, v in enumerate(self.levels)})
        )

    def __reduce__(self):
        return FactorEncoding, (self.name, self.levels)


@dataclass(frozen=True)
class IndexedRandomEffect:
    """One symbolic re interaction in original formula coefficient order."""

    term_id: int
    variables: tuple[str, ...]
    factors: tuple[FactorEncoding, ...]
    numeric_variables: tuple[str, ...]
    col_start: int
    n_coef: int
    sp_index: int
    penalty_scale: float

    def evaluate(self, columns: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return IDs/amplitudes; unknown levels have zero contribution."""
        n = len(columns[self.variables[0]])
        indices = np.zeros(n, dtype=np.int64)
        amplitude = np.ones(n)
        stride = 1
        for factor in self.factors:
            values = columns[factor.name]
            if values.ndim != 1 or len(values) != n:
                raise ValueError("Random-effect columns must have equal lengths.")
            codes = np.fromiter(
                (factor.lookup.get(_scalar(v), -1) for v in values),
                dtype=np.int64,
                count=n,
            )
            amplitude[codes < 0] = 0.0
            indices += np.maximum(codes, 0) * stride
            stride *= len(factor.levels)
        for name in self.numeric_variables:
            values = np.asarray(columns[name], dtype=float)
            if values.shape != (n,):
                raise ValueError("Random-effect columns must have equal lengths.")
            amplitude *= values
        # Predict.matrix.random.effect zeros non-finite interaction entries.
        amplitude[~np.isfinite(amplitude)] = 0.0
        return indices, amplitude


@dataclass(frozen=True)
class OperatorBatch:
    """Host row geometry, containing no expanded random-effect rows."""

    modest_values: np.ndarray
    modest_columns: tuple[int, ...]
    indexed_rows: tuple[tuple[np.ndarray, np.ndarray, int, int], ...]
    n_coef: int


@dataclass(frozen=True)
class PreparedOperatorModel:
    """Exact modest prediction metadata and symbolic random-effect blocks."""

    modest: PreparedModel
    modest_columns: tuple[int, ...]
    random_effects: tuple[IndexedRandomEffect, ...]
    penalties: PenaltyStructure
    source_fingerprint: str
    source_replay_digest: str
    n_obs: int
    metadata_bound_bytes: int
    preparation_peak_bound_bytes: int
    completed_preparation_scans: int
    started_preparation_scans: int
    preparation_yielded_batches: int
    batch_rows: int
    memory_budget_bytes: int
    modest_raw_width: int

    def evaluator_workspace_bound(self, n_rows: int) -> int:
        """Smooth-native scratch, including retained radial knot dimensions."""
        from jaxgam.formula.operator_workspace import (
            predict_spec_evaluator_workspace_bound,
        )

        return predict_spec_evaluator_workspace_bound(self.modest.predict_spec, n_rows)

    @property
    def n_coef(self) -> int:
        return self.penalties.n_coef

    def batch_workspace_bound(self, n_rows: int) -> int:
        """Conservative host evaluation workspace before predictors allocate."""
        variables = max((len(t.variables) for t in self.random_effects), default=0)
        generic = 8 * n_rows * (
            8 * self.modest_raw_width
            + 2 * len(self.random_effects)
            + 16
            + 2 * variables
        ) + 8 * len(self.modest_columns)
        return generic + self.evaluator_workspace_bound(n_rows)

    def evaluate_batch(
        self, batch: RowBatch, *, available_workspace_bytes: int | None = None
    ) -> OperatorBatch:
        """Evaluate bounded public-coordinate actions without dense RE rows."""
        if len(batch.row_positions) > self.batch_rows:
            raise MemoryError("Operator batch exceeds its frozen row bound.")
        available = (
            self.memory_budget_bytes - self.metadata_bound_bytes
            if available_workspace_bytes is None
            else available_workspace_bytes
        )
        if self.batch_workspace_bound(len(batch.row_positions)) > available:
            raise MemoryError("Operator predictor workspace exceeds its budget.")
        modest = self.modest.evaluate_batch(batch)
        rows = tuple(
            (*term.evaluate(batch.columns), term.col_start, term.n_coef)
            for term in self.random_effects
        )
        return OperatorBatch(modest, self.modest_columns, rows, self.n_coef)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _level_bound(value: Any) -> int:
    # Includes discovery/lookup resize overlap, level tuples, sorting pointers,
    # Python scalar/key/value ownership. Four bytes per character bounds UCS4.
    return 512 + (4 * len(value) if isinstance(value, (str, bytes)) else 64)


def _check_budget(required: int, control: OperatorPreparationControl) -> None:
    if required > control.memory_budget_bytes:
        raise MemoryError(
            f"Operator preparation needs at most {required} bytes; "
            f"budget is {control.memory_budget_bytes}."
        )


# A retained first-row hex digest plus its old/new digest and hashlib wrapper
# during verification. Payload/encoding bytes use the existing row ledger.
_REPLAY_HEADER_BYTES = 512


class _CheckedSource:
    """Bound existing setup scans and verify complete replay/selection order."""

    def __init__(
        self,
        source: RowSource,
        control: OperatorPreparationControl,
        reserved_bytes: int,
    ):
        self.source = source
        self.batch_rows = control.batch_rows
        self.control = control
        self.reserved_bytes = reserved_bytes
        self.peak_bound_bytes = reserved_bytes
        self.digest: str | None = None
        self.first_row_digest: str | None = None
        self.completed_scans = 0
        self.started_scans = 0
        self.yielded_batches = 0
        self.schemas: dict[str, Any] = {}
        value = source.n_rows
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or value < 1
        ):
            raise ValueError("Operator RowSource count must be a positive integer.")
        self._n_rows = int(value)

    @property
    def n_rows(self) -> int:
        value = self.source.n_rows
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or value != self._n_rows
        ):
            raise RuntimeError("RowSource changed its operator row count metadata.")
        return self._n_rows

    def fingerprint(self) -> str:
        return self.source.fingerprint()

    def scan(self, batch_rows: int):
        self.started_scans += 1
        digest = hashlib.sha256()
        first_row_digest = None
        count = 0
        before = self.fingerprint()
        requested_rows = min(batch_rows, self.batch_rows)
        for batch in self.source.scan(requested_rows):
            self.yielded_batches += 1
            positions = np.asarray(batch.row_positions)
            n = len(positions)
            if (
                positions.ndim != 1
                or positions.dtype.kind not in "iu"
                or np.any(positions < 0)
                or np.any(positions > np.iinfo(np.int64).max)
            ):
                raise ValueError("Source positions must be nonnegative integers.")
            if n > requested_rows or count + n > self.n_rows:
                raise RuntimeError("RowSource yielded an invalid operator row count.")
            if batch.y is None or not np.all(batch.valid):
                raise ValueError("Operator preparation needs valid response rows.")
            if (
                not np.all(np.isfinite(batch.y))
                or not np.all(np.isfinite(batch.weight))
                or not np.all(np.isfinite(batch.offset))
                or np.any(batch.weight < 0)
            ):
                raise ValueError("Operator responses/weights/offsets are invalid.")
            fixed_row_bound = 512 * (len(batch.columns) + 4) + sum(
                16 * len(name) for name in batch.columns
            )
            string_width = sum(
                col.dtype.itemsize
                for col in batch.columns.values()
                if getattr(col.dtype, "kind", None) in ("U", "S")
            )
            # NumPy fixed-string scalar extraction itself owns storage. Check
            # that bound before indexing, then check variable object strings
            # before serializing a row. Source-owned input/fingerprint work
            # remains outside this operator's storage ledger.
            _check_budget(
                self.reserved_bytes
                + fixed_row_bound
                + 4 * string_width
                + _REPLAY_HEADER_BYTES,
                self.control,
            )
            self.peak_bound_bytes = max(
                self.peak_bound_bytes,
                self.reserved_bytes
                + fixed_row_bound
                + 4 * string_width
                + _REPLAY_HEADER_BYTES,
            )
            # Keep source-owned columns as references. Converting categorical
            # Series to NumPy here would retain B object pointers per column,
            # including columns unused by the model, before the row bound was
            # enforced. Scalar streaming retains the complete replay contract
            # while its workspace depends on column count rather than B*C.
            columns = tuple(
                (name, batch.columns[name]) for name in sorted(batch.columns)
            )
            if self.schemas and set(self.schemas) != set(batch.columns):
                raise RuntimeError("RowSource changed its operator schema.")
            for name, col in batch.columns.items():
                dtype = col.dtype
                previous = self.schemas.setdefault(name, dtype)
                same = previous == dtype
                if isinstance(previous, pd.CategoricalDtype) and isinstance(
                    dtype, pd.CategoricalDtype
                ):
                    same = same and previous.categories.equals(dtype.categories)
                if not same:
                    raise RuntimeError("RowSource changed its operator schema.")
            # Empty source batches carry schema but no selected observations.
            # Verify their metadata and skip them before any basis/mean work.
            if n == 0:
                del columns
                continue
            for i in range(n):
                payload = (
                    int(positions[i]),
                    tuple(
                        (
                            name,
                            _scalar(
                                values.iloc[i]
                                if isinstance(values, pd.Series)
                                else values[i]
                            ),
                        )
                        for name, values in columns
                    ),
                    float(batch.y[i]),
                    float(batch.weight[i]),
                    float(batch.offset[i]),
                )
                row_bound = fixed_row_bound + sum(
                    16 * len(value)
                    for _, value in payload[1]
                    if isinstance(value, (str, bytes))
                )
                _check_budget(
                    self.reserved_bytes + row_bound + _REPLAY_HEADER_BYTES, self.control
                )
                self.peak_bound_bytes = max(
                    self.peak_bound_bytes,
                    self.reserved_bytes + row_bound + _REPLAY_HEADER_BYTES,
                )
                encoded = pickle.dumps(payload, protocol=5)
                if count == 0 and i == 0:
                    first_row_digest = hashlib.sha256(encoded).hexdigest()
                    if (
                        self.first_row_digest is not None
                        and first_row_digest != self.first_row_digest
                    ):
                        raise RuntimeError("RowSource changed its operator replay.")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
            count += n
            yield batch
        if count != self.n_rows:
            raise RuntimeError("RowSource yielded fewer operator rows than declared.")
        if self.fingerprint() != before:
            raise RuntimeError("RowSource changed during operator preparation.")
        current = digest.hexdigest()
        if self.digest is not None and current != self.digest:
            raise RuntimeError("RowSource changed its operator replay.")
        self.digest = current
        self.first_row_digest = first_row_digest
        self.completed_scans += 1


_DEFAULT_CONTROL = OperatorPreparationControl()


def prepare_operator_model(
    formula_spec: FormulaSpec,
    source: RowSource,
    *,
    control: OperatorPreparationControl = _DEFAULT_CONTROL,
) -> PreparedOperatorModel:
    """Prepare exact cubic/numeric plus indexed re geometry under hard bounds.

    This initial adapter explicitly rejects other basis/constraint contracts.
    It does not use global fitting preparation, dense training coefficient-map
    construction, penalty materialization or Cartesian-product level arrays.
    Caller-owned source backing storage and native allocator RSS are excluded
    from the conservative array/metadata estimate.
    """
    if not isinstance(control, OperatorPreparationControl):
        raise TypeError("control must be an OperatorPreparationControl.")
    if (
        not isinstance(source, RowSource)
        or isinstance(source.n_rows, bool)
        or not isinstance(source.n_rows, numbers.Integral)
        or source.n_rows < 1
    ):
        raise ValueError("Operator setup requires a nonempty restartable RowSource.")
    terms = formula_spec.smooth_terms
    if any(
        (t.by is not None and t.bs == "re")
        or t.smooth_type not in {"s", "te", "ti"}
        or (t.smooth_type != "s" and t.bs == "re")
        or t.bs not in {"re", "cr", "cs", "cc", "tp", "ts", "gp"}
        for t in terms
    ):
        raise NotImplementedError(
            "Operator preparation supports numeric terms, cr/cs/cc, tp/ts/gp and re."
        )
    if any(t.extra_args for t in terms if t.bs not in {"tp", "ts", "gp"}):
        raise NotImplementedError(
            "Operator custom smooth/precision/id options are unsupported."
        )
    modest_spec = replace(formula_spec, smooth_terms=[t for t in terms if t.bs != "re"])
    raw_width = int(formula_spec.has_intercept) + len(formula_spec.parametric_terms)
    from jaxgam.formula.operator_modest import _width

    raw_width += sum(_width(t) for t in modest_spec.smooth_terms)
    if raw_width > control.max_modest_coefficients:
        raise MemoryError(
            f"Modest operator width {raw_width} exceeds hard cap "
            f"{control.max_modest_coefficients}."
        )
    # Conservative simultaneous modest QR/basis/constraint/eigen ownership and
    # bounded row work. Its dimension is capped independently of global p.
    modest_bound = 8 * (24 * raw_width**2 + control.batch_rows * (8 * raw_width + 32))
    symbolic_bound = 1024 * (len(terms) + len(formula_spec.parametric_terms) + 1)
    symbolic_bound += sum(4 * len(v) + 512 for t in terms for v in t.variables)
    _check_budget(modest_bound + symbolic_bound, control)
    checked = _CheckedSource(source, control, modest_bound + symbolic_bound)
    fingerprint = checked.fingerprint()
    re_variables = tuple(
        dict.fromkeys(v for t in terms if t.bs == "re" for v in t.variables)
    )
    kinds: dict[str, bool] = {}
    discovered: dict[str, dict[Any, None]] = {}
    declared: dict[str, tuple[Any, ...] | None] = {}
    metadata_bound = symbolic_bound
    magnitudes = {i: 0.0 for i, t in enumerate(terms) if t.bs == "re"}
    for batch in checked.scan(control.batch_rows):
        for name in re_variables:
            if name not in batch.columns:
                raise ValueError(f"Random-effect variable '{name}' is missing.")
            col = batch.columns[name]
            factor = is_factor(col)
            if name in kinds and kinds[name] != factor:
                raise RuntimeError("Random-effect column changed factor type.")
            if name not in kinds:
                kinds[name] = factor
                if factor:
                    discovered[name] = {}
                    declared[name] = None
                    if isinstance(col, pd.Series) and isinstance(
                        col.dtype, pd.CategoricalDtype
                    ):
                        categories = col.cat.categories
                        cost = sum(_level_bound(_scalar(v)) for v in categories)
                        _check_budget(modest_bound + metadata_bound + cost, control)
                        metadata_bound += cost
                        checked.reserved_bytes = modest_bound + metadata_bound
                        declared[name] = tuple(_scalar(v) for v in categories)
                        discovered[name] = dict.fromkeys(declared[name])
            if factor:
                if declared[name] is not None and (
                    not isinstance(col, pd.Series)
                    or not isinstance(col.dtype, pd.CategoricalDtype)
                    or tuple(col.cat.categories) != declared[name]
                ):
                    raise RuntimeError(
                        "Random-effect categories changed during replay."
                    )
                for value in col:
                    value = _scalar(value)
                    if pd.isna(value):
                        raise ValueError(
                            "Training random-effect levels must be nonmissing."
                        )
                    if value not in discovered[name]:
                        if declared[name] is not None:
                            raise RuntimeError(
                                "Training value is outside declared levels."
                            )
                        cost = _level_bound(value)
                        _check_budget(modest_bound + metadata_bound + cost, control)
                        metadata_bound += cost
                        checked.reserved_bytes = modest_bound + metadata_bound
                        discovered[name][value] = None
            elif not np.all(np.isfinite(np.asarray(col, dtype=float))):
                raise ValueError("Numeric random-effect covariates must be finite.")
        for i in magnitudes:
            amplitude = np.ones(len(batch.row_positions))
            for name in terms[i].variables:
                if not kinds[name]:
                    amplitude *= np.asarray(batch.columns[name], dtype=float)
            if not np.all(np.isfinite(amplitude)):
                raise ValueError("Random-effect interaction amplitude overflowed.")
            magnitudes[i] = max(magnitudes[i], float(np.max(np.abs(amplitude))))
        del batch
    encodings = {
        name: FactorEncoding(
            name,
            declared[name]
            if declared[name] is not None
            else tuple(sorted(discovered[name])),
        )
        for name in discovered
    }
    del discovered
    if any(
        t.by is not None or t.smooth_type != "s" or t.bs in {"tp", "ts", "gp"}
        for t in modest_spec.smooth_terms
    ):
        from jaxgam.formula.operator_modest import prepare_modest_cubic_by

        modest, raw_width, modest_retained, modest_construction = (
            prepare_modest_cubic_by(modest_spec, checked, control)
        )
    else:
        modest = prepare_model(modest_spec, checked, allow_cyclic=True)
        modest_retained = 8 * 8 * raw_width**2
        modest_construction = modest_bound + metadata_bound
    modest_terms = modest.predict_spec.coef_map.terms
    parameter_width = sum(
        t.n_coefs for t in modest_terms if t.term_type == "parametric"
    )
    columns = list(range(parameter_width))
    cursor = parameter_width
    modest_smooth = iter(t for t in modest_terms if t.term_type == "smooth")
    sp_placements = {}
    sp_cursor = 0
    random_effects = []
    blocks = []
    for i, term in enumerate(terms):
        if term.bs == "re":
            factors = tuple(encodings[v] for v in term.variables if kinds[v])
            numeric = tuple(v for v in term.variables if not kinds[v])
            width = math.prod(len(f.levels) for f in factors)
            if width == 0 or cursor + width > np.iinfo(np.int64).max:
                raise ValueError("Random-effect symbolic dimension is invalid.")
            magnitude = magnitudes[i]
            if magnitude > math.sqrt(np.finfo(float).max):
                raise ValueError("Random-effect penalty normalization overflowed.")
            scale = magnitude**2 if magnitude > 0 else 1.0
            if not np.isfinite(scale):
                raise ValueError("Random-effect penalty normalization overflowed.")
            random_effects.append(
                IndexedRandomEffect(
                    i,
                    tuple(term.variables),
                    factors,
                    numeric,
                    cursor,
                    width,
                    sp_cursor,
                    scale,
                )
            )
            blocks.append(
                PenaltyBlock(
                    cursor,
                    cursor + width,
                    (sp_cursor,),
                    (IdentityPenalty(width, scale),),
                    IdentityTransform(width),
                    (width,),
                )
            )
            cursor += width
            sp_cursor += 1
        else:
            modest_term = next(modest_smooth)
            for index in modest_term.penalty_indices:
                sp_placements[index] = sp_cursor
                sp_cursor += 1
            columns.extend(range(cursor, cursor + modest_term.n_coefs))
            cursor += modest_term.n_coefs
    if modest.penalties is not None:
        for block in modest.penalties.blocks:
            start = columns[block.start]
            blocks.append(
                replace(
                    block,
                    start=start,
                    stop=start + block.size,
                    sp_indices=tuple(
                        sp_placements[index] for index in block.sp_indices
                    ),
                )
            )
    if checked.fingerprint() != fingerprint:
        raise RuntimeError("RowSource changed after operator preparation.")
    assert checked.digest is not None
    return PreparedOperatorModel(
        modest,
        tuple(columns),
        tuple(random_effects),
        PenaltyStructure(cursor, tuple(sorted(blocks, key=lambda b: b.start))),
        fingerprint,
        checked.digest,
        source.n_rows,
        metadata_bound + modest_retained,
        max(modest_construction, checked.peak_bound_bytes),
        checked.completed_scans,
        checked.started_scans,
        checked.yielded_batches,
        control.batch_rows,
        control.memory_budget_bytes,
        raw_width,
    )
