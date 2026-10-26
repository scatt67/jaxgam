"""Replay bounded family score reductions at one frozen operator fit state.

The returned score terms describe the supplied beta/theta/rho/score_phi.
An EFS accepted/trial adapter must separately preserve gam.fit4's pre-gdi1
score provenance and incoming-versus-reported scale timing.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import jax
import numpy as np

from jaxgam.data.source import RowSource
from jaxgam.execution.operator_efs_trace import _local_action
from jaxgam.execution.operator_fit import (
    FrozenOperatorWorkingState,
    _checked_fit_source,
    _evaluate_geometry,
)
from jaxgam.execution.operator_score import OperatorScoreTerms
from jaxgam.fitting.family_execution import batch_saturated_loglikelihood


@dataclass(frozen=True)
class OperatorScoreReductions:
    """Checked direct source terms and the actual replay work they required."""

    terms: OperatorScoreTerms
    raw_deviance: float
    penalty_quadratic: float
    source_scans: int
    source_batches: int
    peak_bound_bytes: int
    unit_weights: bool = False
    zero_offset: bool = False
    maximum_response: float = float("inf")
    maximum_absolute_response: float = float("inf")
    # Largest l1 norm of one public design row (modest values plus indexed
    # amplitudes); first-order score sensitivity to coefficient error uses it.
    maximum_row_l1_norm: float = float("inf")


class MatchedOperatorScoreReducer:
    """Compose frozen family rows, source replay and local penalty blocks."""

    def __init__(
        self,
        state: FrozenOperatorWorkingState,
        source: RowSource,
        log_lambda: np.ndarray,
        *,
        score_phi: float,
        memory_budget_bytes: int,
        max_y: int = 0,
    ) -> None:
        if not isinstance(state, FrozenOperatorWorkingState) or state.beta is None:
            raise TypeError("Score reductions require a frozen coefficient state.")
        model = state.model
        if (
            not isinstance(log_lambda, np.ndarray)
            or log_lambda.dtype != np.float64
            or log_lambda.shape != (model.penalties.n_penalties,)
            or not np.all(np.isfinite(log_lambda))
        ):
            raise ValueError("Score rho must be finite float64 in penalty order.")
        with np.errstate(over="ignore", under="ignore"):
            lambda_ = np.exp(log_lambda)
        if not np.all(np.isfinite(lambda_)) or np.any(lambda_ <= 0):
            raise ValueError("Score rho must give finite positive penalties.")
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, numbers.Integral)
            or memory_budget_bytes < 1
        ):
            raise ValueError("Score memory budget must be a positive integer.")
        if (
            isinstance(max_y, bool)
            or not isinstance(max_y, numbers.Integral)
            or max_y < 0
        ):
            raise ValueError("max_y must be a nonnegative integer.")
        if not np.isfinite(score_phi) or score_phi <= 0:
            raise ValueError("Score phi must be finite and positive.")
        if state.family.scale_known and score_phi != 1.0:
            raise ValueError("Known-scale family score_phi must be one.")
        if source.fingerprint() != model.source_fingerprint:
            raise RuntimeError("Score source changed from prepared operator model.")
        self.state = state
        self.source = source
        self.log_lambda = log_lambda.copy()
        self.log_lambda.setflags(write=False)
        self.score_phi = float(score_phi)
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.max_y = int(max_y)

    def _penalty_quadratic(self) -> float:
        state = self.state
        beta = state.beta
        assert beta is not None
        total = 0.0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for block in state.model.penalties.blocks:
                local_beta = beta[block.start : block.stop]
                for index, penalty in zip(
                    block.sp_indices, block.local_penalties, strict=True
                ):
                    quadratic = float(local_beta @ _local_action(penalty, local_beta))
                    total += np.exp(self.log_lambda[index]) * quadratic
        if not np.isfinite(total) or total < 0:
            raise FloatingPointError("Local score penalty quadratic is invalid.")
        return float(total)

    def reduce(self) -> OperatorScoreReductions:
        """One complete replay evaluates direct deviance and family$ls."""
        state = self.state
        state.validate()
        model = state.model
        B = model.batch_rows
        p = model.n_coef
        # Reserve the model's entire declared ownership, the frozen working
        # state and transfer budget, padded response/weight/valid leaves, and
        # small scalar/digest containers. Source-owned input remains excluded.
        reserved = (
            model.memory_budget_bytes
            + state.retained_bound_bytes
            + state.row_memory_budget_bytes
            + 8 * (20 * B + 5 * p + 64)
            + 8192
        )
        if reserved > self.memory_budget_bytes:
            raise MemoryError("Matched score replay exceeds its memory budget.")
        checked = _checked_fit_source(
            model, self.source, reserved, self.memory_budget_bytes
        )
        deviance = 0.0
        saturated = 0.0
        unit_weights = True
        zero_offset = True
        maximum_response = 0.0
        maximum_absolute_response = 0.0
        maximum_row_l1_norm = 0.0
        device = jax.devices("cpu")[0]
        padding = state.context.padding
        for batch in checked.scan(B):
            unit_weights = unit_weights and bool(np.all(batch.weight == 1.0))
            zero_offset = zero_offset and bool(np.all(batch.offset == 0.0))
            if batch.y is not None:
                maximum_response = max(maximum_response, float(np.max(batch.y)))
                maximum_absolute_response = max(
                    maximum_absolute_response, float(np.max(np.abs(batch.y)))
                )
            rows = _evaluate_geometry(model, batch)
            row_l1 = np.sum(np.abs(rows.modest_values), axis=1)
            for _, amplitude, _, _ in rows.indexed_rows:
                row_l1 = row_l1 + np.abs(amplitude)
            if len(row_l1):
                maximum_row_l1_norm = max(maximum_row_l1_norm, float(np.max(row_l1)))
            _, quantities = state.evaluate(batch, rows)
            if not bool(quantities.input_ok) or not bool(quantities.domain_ok):
                raise ValueError("Matched score row domain is invalid.")
            direct = float(quantities.deviance)
            if not np.isfinite(direct):
                raise FloatingPointError("Matched score deviance is nonfinite.")
            n = len(batch.row_positions)
            y = np.full(B, padding.response, dtype=np.float64)
            weight = np.zeros(B, dtype=np.float64)
            valid = np.zeros(B, dtype=bool)
            y[:n] = state.family.execution_initial_response_cpu(batch.y, batch.weight)
            weight[:n] = batch.weight
            valid[:n] = True
            value, admissible = batch_saturated_loglikelihood(
                jax.device_put(y, device),
                jax.device_put(weight, device),
                jax.device_put(valid, device),
                jax.device_put(self.score_phi, device),
                state.parameters,
                state.family,
                state.context,
                max_y=self.max_y,
            )
            if not bool(admissible):
                raise ValueError("Matched score saturated likelihood is invalid.")
            deviance += direct
            saturated += float(value)
            del batch, rows, quantities, y, weight, valid, value
        if checked.completed_scans != 1 or not np.all(
            np.isfinite([deviance, saturated])
        ):
            raise FloatingPointError("Matched score reduction did not complete.")
        penalty = self._penalty_quadratic()
        return OperatorScoreReductions(
            OperatorScoreTerms(deviance + penalty, saturated, self.score_phi, p),
            deviance,
            penalty,
            checked.completed_scans,
            checked.yielded_batches,
            checked.peak_bound_bytes,
            unit_weights,
            zero_offset,
            maximum_response,
            maximum_absolute_response,
            maximum_row_l1_norm,
        )
