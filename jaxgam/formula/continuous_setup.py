"""Bounded basis setup on an mgcv-compatible compact data frame.

Continuous discretization deliberately separates two operations.  The data
layer chooses the compact representative rows and full-row selectors; this
module runs ordinary Phase-1 basis setup only on that compact frame and keeps
the resulting prediction metadata.  It never constructs a training-row design
matrix.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from jaxgam.data.continuous_frame import (
    CompactFrameUnavailable,
    CompactSetupFrame,
    build_compact_setup_frame,
)
from jaxgam.data.discretize import DiscretizationBudget
from jaxgam.data.source import RowSource
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.predict_matrix import PredictSpec, build_predict_spec
from jaxgam.formula.terms import FormulaSpec, SmoothSpec
from jaxgam.penalties.structure import PenaltyStructure
from jaxgam.smooths.tprs import default_penalty_order, null_space_dimension
from jaxgam.smooths.utils import get_factor_levels, is_factor, is_ordered_factor


@dataclass(frozen=True)
class ContinuousSetupUnavailable:
    """Compact basis construction cannot satisfy its combined memory budget."""

    reason: str
    required_bytes: int
    budget_bytes: int
    budget_scope: Literal["table", "selector", "batch"]
    stage: Literal["frame", "basis"]


@dataclass(frozen=True)
class ContinuousDiscreteSetup:
    """Frozen compact basis state and the selectors that produced it."""

    compact: CompactSetupFrame
    predict_spec: PredictSpec
    penalties: PenaltyStructure | None
    source_fingerprint: str
    basis_fingerprint: str
    n_obs: int
    response: str
    prospective_basis_bytes: int
    retained_basis_bytes: int
    construction_peak_bytes: int
    retained_table_bytes: int
    retained_selector_bytes: int

    def __post_init__(self) -> None:
        if self.n_obs != self.compact.n_rows:
            raise ValueError("compact basis observation count is inconsistent")
        if self.retained_table_bytes < (
            self.compact.retained_table_bytes + self.retained_basis_bytes
        ):
            raise ValueError("compact basis retained-byte ledger is incomplete")
        if self.construction_peak_bytes < self.retained_table_bytes:
            raise ValueError(
                "compact basis construction peak is below retained storage"
            )

    @property
    def n_coef(self) -> int:
        return self.predict_spec.total_coefs

    def cleanup(self) -> None:
        """Release selector files owned by the compact frame."""
        self.compact.cleanup()


def _factor_metadata(frame: pd.DataFrame, name: str) -> tuple[int, bool] | None:
    column = frame[name]
    if not is_factor(column):
        return None
    return len(get_factor_levels(column)), is_ordered_factor(column)


def _parametric_width(formula: FormulaSpec, frame: pd.DataFrame) -> int:
    width = int(formula.has_intercept)
    seen_factor = False
    for term in formula.parametric_terms:
        factor = _factor_metadata(frame, term.name)
        if factor is None:
            width += 1
            continue
        levels, _ordered = factor
        drop_reference = formula.has_intercept or seen_factor
        width += levels - int(drop_reference)
        seen_factor = True
    return width


def _tprs_width(k: int, dimensions: int, order: int | None = None) -> int:
    order = default_penalty_order(dimensions) if order is None else order
    null_width = null_space_dimension(dimensions, order)
    if k == -1:
        return null_width + (8, 27, 100)[min(dimensions, 3) - 1]
    return max(k, null_width + 1)


def _base_width(spec: SmoothSpec, frame: pd.DataFrame) -> int:
    dimensions = len(spec.variables)
    if spec.bs == "re":
        width = 1
        for name in spec.variables:
            factor = _factor_metadata(frame, name)
            width *= factor[0] if factor is not None else 1
        return width
    if spec.smooth_type in {"te", "ti"}:
        marginal = 5 if spec.k == -1 else max(spec.k, 1)
        if spec.bs in {"tp", "ts"}:
            raw_m = spec.extra_args.get("m")
            orders = (
                list(raw_m)
                if isinstance(raw_m, (list, tuple))
                else [raw_m] * dimensions
            )
            widths = [
                _tprs_width(marginal, 1, None if order is None else max(order, 0))
                for order in orders
            ]
            return int(np.prod(widths))
        return marginal**dimensions
    if spec.bs in {"cr", "cs", "cc"}:
        minimum = 4 if spec.bs == "cc" else 3
        return 10 if spec.k == -1 else max(spec.k, minimum)
    if spec.bs in {"tp", "ts"}:
        return _tprs_width(
            spec.k,
            dimensions,
            spec.extra_args.get("m"),
        )
    if spec.bs == "gp":
        if dimensions > 3 and spec.k == -1:
            raise ValueError("GP dimensions above 3 require an explicit k")
        return (
            dimensions + 1 + (10, 30, 100)[dimensions - 1]
            if spec.k == -1
            else max(spec.k, dimensions + 2)
        )
    # The registry will produce the precise unsupported-basis error.  Charging
    # an explicit k here still prevents an unbounded call before that error.
    return 10 if spec.k == -1 else max(spec.k, 1)


def _smooth_width(spec: SmoothSpec, frame: pd.DataFrame) -> int:
    width = _base_width(spec, frame)
    if spec.by is None:
        return width
    factor = _factor_metadata(frame, spec.by)
    if factor is None:
        return width
    levels, ordered = factor
    return width * (levels - int(ordered))


def _prospective_basis_bytes(formula: FormulaSpec, frame: pd.DataFrame) -> int:
    """Conservatively bound arrays live inside ``ModelSetup.build``.

    The estimate charges compact design copies, raw and constrained blocks,
    local penalties, global dependence/QR work, tensor marginal work, and
    ownership copies made while freezing ``PredictSpec``.  It intentionally
    uses raw widths: constraints can reduce storage but cannot make a prior
    allocation safe.
    """
    q = len(frame)
    parametric = _parametric_width(formula, frame)
    smooth_widths = [_smooth_width(spec, frame) for spec in formula.smooth_terms]
    total = parametric + sum(smooth_widths)
    # A factor-by constructor temporarily builds expanded square penalties;
    # tensor setup also owns one dense local penalty per margin.
    local_squares = 0
    margin_work = 0
    for spec, width in zip(formula.smooth_terms, smooth_widths, strict=True):
        penalties = len(spec.variables) if spec.smooth_type in {"te", "ti"} else 1
        if spec.by is not None:
            factor = _factor_metadata(frame, spec.by)
            if factor is not None:
                penalties *= factor[0] - int(factor[1])
        local_squares += penalties * width * width
        base = _base_width(spec, frame)
        margin_work += q * base + base * base
        if spec.bs in {"tp", "ts", "gp"}:
            knots = min(q, int(spec.extra_args.get("max_knots", 2000)))
            margin_work += q * knots + knots * knots
    float_items = (
        8 * q * max(total, 1)  # block, constrained, assembled and ownership copies
        + 8 * local_squares  # raw/constrained/frozen local penalty copies
        + 6 * max(total, 1) ** 2  # QR, dependence and centering work
        + 4 * margin_work  # marginal/kernel construction and transforms
    )
    # NumPy/SciPy temporaries are float64.  Keep an explicit metadata allowance
    # so a tiny budget rejects before entering the setup implementation.
    return 8 * float_items + 65_536 + 4096 * len(formula.smooth_terms)


def _resident_size(value: object, seen: set[int] | None = None) -> int:
    """Count retained Python and NumPy state once per shared object."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return sys.getsizeof(value) + int(value.nbytes)
    total = sys.getsizeof(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        total += sum(
            _resident_size(getattr(value, field.name), seen)
            for field in dataclasses.fields(value)
        )
    elif isinstance(value, dict):
        total += sum(
            _resident_size(key, seen) + _resident_size(item, seen)
            for key, item in value.items()
        )
    elif isinstance(value, (tuple, list, set, frozenset)):
        total += sum(_resident_size(item, seen) for item in value)
    elif hasattr(value, "__dict__") and value.__class__.__module__.startswith("jaxgam"):
        total += _resident_size(vars(value), seen)
    return total


def _basis_token(spec: PredictSpec) -> str:
    """Hash the complete prediction-only graph without importing Phase 2."""
    payload = pickle.dumps(spec, protocol=5)
    return hashlib.sha256(b"jaxgam-continuous-basis-v1\0" + payload).hexdigest()


def build_continuous_discrete_setup(
    formula: FormulaSpec,
    source: RowSource,
    *,
    factor_names: tuple[str, ...] = (),
    m: int | tuple[int, ...] | None = None,
    budget: DiscretizationBudget | None = None,
    selector_memmap_dir: str | Path | None = None,
    batch_rows: int = 65_536,
    seed: int = 8547,
) -> ContinuousDiscreteSetup | ContinuousSetupUnavailable:
    """Compress rows, construct the compact basis once, and freeze its state."""
    if budget is None:
        budget = DiscretizationBudget()
    source_fingerprint = source.fingerprint()
    compact = build_compact_setup_frame(
        formula,
        source,
        factor_names=factor_names,
        m=m,
        budget=budget,
        selector_memmap_dir=selector_memmap_dir,
        batch_rows=batch_rows,
        seed=seed,
    )
    if isinstance(compact, CompactFrameUnavailable):
        return ContinuousSetupUnavailable(
            compact.reason,
            compact.required_bytes,
            compact.budget_bytes,
            compact.budget_scope,
            "frame",
        )
    try:
        prospective = _prospective_basis_bytes(formula, compact.frame)
        required_peak = compact.retained_table_bytes + prospective
        if required_peak > budget.max_table_bytes:
            compact.cleanup()
            return ContinuousSetupUnavailable(
                "compact basis setup exceeds combined construction budget",
                required_peak,
                budget.max_table_bytes,
                "table",
                "basis",
            )
        # Pinned ``smoothCon(apply.by=FALSE)`` still applies each by variable
        # while deriving constraints/gam.side, then restores its by-free X0 for
        # the discrete tables.  ModelSetup derives the same applied-by
        # constraints here; PredictSpec retains the underlying base smooth so
        # the descriptor stage can store its by-free table and multiplier as
        # separate indexed margins.  The compact training X is discarded.
        setup = ModelSetup.build(formula, compact.frame)
        predict_spec = build_predict_spec(setup)
        penalties = setup.penalties
        retained_basis = _resident_size((predict_spec, penalties))
        retained_table = compact.retained_table_bytes + retained_basis
        construction_peak = max(
            compact.construction_peak_bytes,
            compact.retained_table_bytes + prospective,
            retained_table,
        )
        if retained_table > budget.max_table_bytes:
            compact.cleanup()
            return ContinuousSetupUnavailable(
                "frozen compact basis metadata exceeds construction budget",
                retained_table,
                budget.max_table_bytes,
                "table",
                "basis",
            )
        if source.fingerprint() != source_fingerprint:
            raise RuntimeError("RowSource changed during compact basis setup")
        return ContinuousDiscreteSetup(
            compact=compact,
            predict_spec=predict_spec,
            penalties=penalties,
            source_fingerprint=source_fingerprint,
            basis_fingerprint=_basis_token(predict_spec),
            n_obs=source.n_rows,
            response=formula.response,
            prospective_basis_bytes=prospective,
            retained_basis_bytes=retained_basis,
            construction_peak_bytes=construction_peak,
            retained_table_bytes=retained_table,
            retained_selector_bytes=compact.retained_selector_bytes,
        )
    except BaseException:
        compact.cleanup()
        raise
