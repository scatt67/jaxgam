"""Bounded source initial.spg start for a pure indexed identity penalty."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jaxgam.data.source import RowSource
from jaxgam.execution.operator_fit import _checked_fit_source, _evaluate_geometry
from jaxgam.families.base import ExponentialFamily
from jaxgam.formula.operator_prepare import PreparedOperatorModel
from jaxgam.penalties.structure import IdentityPenalty


@dataclass(frozen=True)
class OperatorEFSStart:
    """Unshifted source start and measured reduction work."""

    log_lambda: np.ndarray
    source_scans: int
    source_batches: int
    peak_bound_bytes: int

    def __post_init__(self) -> None:
        rho = np.array(self.log_lambda, dtype=np.float64, copy=True)
        if rho.shape != (1,) or not np.all(np.isfinite(rho)):
            raise ValueError("Operator EFS start must be one finite smoothing value.")
        rho.setflags(write=False)
        object.__setattr__(self, "log_lambda", rho)


def prepare_indexed_operator_efs_start(
    model: PreparedOperatorModel,
    source: RowSource,
    family: ExponentialFamily,
    *,
    memory_budget_bytes: int,
) -> OperatorEFSStart:
    """Replay initial.spg working diagonals without a global design matrix.

    This supports one full-rank indexed random effect with unit amplitudes,
    weights and no offset. Other model shapes need their own start provider.
    The balancing loop reproduces the shared initial.sp diagonal policy for
    a scaled identity penalty without expanding that penalty to p by p.
    """
    if not isinstance(model, PreparedOperatorModel):
        raise TypeError("Operator EFS start requires a prepared model.")
    if source.fingerprint() != model.source_fingerprint:
        raise RuntimeError("Operator EFS start source changed from preparation.")
    p = model.n_coef
    if (
        model.modest.n_coef
        or len(model.random_effects) != 1
        or len(model.penalties.blocks) != 1
        or model.penalties.n_penalties != 1
    ):
        raise NotImplementedError("Operator EFS source start needs one indexed effect.")
    term = model.random_effects[0]
    block = model.penalties.blocks[0]
    if (
        term.col_start != 0
        or term.n_coef != p
        or term.numeric_variables
        or block.start != 0
        or block.stop != p
        or block.sp_indices != (0,)
        or len(block.local_penalties) != 1
        or not isinstance(block.local_penalties[0], IdentityPenalty)
    ):
        raise NotImplementedError("Operator EFS source start needs identity penalty.")
    scale = block.local_penalties[0].scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Operator EFS start penalty scale must be finite positive.")
    if memory_budget_bytes <= 0:
        raise ValueError("Operator EFS start needs a positive memory budget.")
    reserved = model.metadata_bound_bytes + 64 * p + 8192
    if reserved + model.batch_workspace_bound(model.batch_rows) > memory_budget_bytes:
        raise MemoryError("Operator EFS start exceeds its memory budget.")
    diagonal = np.zeros(p, dtype=np.float64)
    checked = _checked_fit_source(model, source, reserved, memory_budget_bytes)
    batches = 0
    for batch in checked.scan(model.batch_rows):
        if (
            batch.y is None
            or not np.all(batch.weight == 1.0)
            or not np.all(batch.offset == 0.0)
        ):
            raise NotImplementedError(
                "Operator EFS source start needs unit, unoffset rows."
            )
        rows = _evaluate_geometry(model, batch)
        ids, amplitude, start, width = rows.indexed_rows[0]
        if start != 0 or width != p or not np.all(amplitude == 1.0):
            raise NotImplementedError("Operator EFS start needs unit indexed rows.")
        initial = family.initial_working_state_cpu(
            batch.y, batch.weight, np.ones(len(batch.y), dtype=bool)
        )
        if not initial.input_ok or not initial.domain_ok:
            raise ValueError("Operator EFS initial.spg family state is invalid.")
        derivative = np.asarray(family.link.mu_eta(initial.eta), dtype=np.float64)
        variance = np.asarray(family.variance(initial.mustart), dtype=np.float64)
        working = derivative * derivative / variance
        if not np.all(np.isfinite(working) & (working >= 0)):
            raise ValueError("Operator EFS initial.spg weights are invalid.")
        np.add.at(diagonal, ids, working)
        batches += 1
    if not np.all(np.isfinite(diagonal)) or not np.any(diagonal > 0):
        raise ValueError("Operator EFS start has no informative design diagonal.")
    initial = float(np.mean(diagonal) / scale)
    occupied = diagonal[diagonal > 0]
    weighted_penalty = initial * scale
    for _ in range(200):
        if np.mean(occupied / (occupied + weighted_penalty)) <= 0.4:
            break
        initial *= 10
        weighted_penalty *= 10
    for _ in range(200):
        if np.mean(occupied / (occupied + weighted_penalty)) >= 0.4:
            break
        initial /= 10
        weighted_penalty /= 10
    if not math.isfinite(initial) or initial <= 0:
        raise FloatingPointError("Operator EFS initial smoothing is invalid.")
    return OperatorEFSStart(
        np.log(np.array([initial], dtype=np.float64)),
        checked.completed_scans,
        batches,
        max(
            reserved + model.batch_workspace_bound(model.batch_rows),
            checked.peak_bound_bytes,
        ),
    )
