"""Bounded family-owned rows and reductions for fixed-sp operator fitting."""

from __future__ import annotations

import copy
import math
import numbers
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.control import OperatorControl
from jaxgam.data.source import RowBatch, RowSource
from jaxgam.execution.null_coefficient import project_null_coefficients
from jaxgam.execution.operator import (
    FrozenScanSystem,
    PCGControl,
    ScanDiagnostics,
    _CountedSource,
    _freeze_host_arrays,
    _padded_geometry,
)
from jaxgam.execution.qr import qr_update
from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
)
from jaxgam.fitting.operator_family import OperatorFamilyRows, operator_family_rows
from jaxgam.fitting.operators import (
    DeviceIndexedRows,
    DeviceOperatorBatch,
    accumulate_rmatvec_batch,
    matvec_batch,
    to_device_operator_batch,
)
from jaxgam.formula.operator_prepare import (
    OperatorBatch,
    OperatorPreparationControl,
    PreparedOperatorModel,
    _CheckedSource,
)
from jaxgam.smooths.source_arithmetic import libm_exp, libm_log


@dataclass(frozen=True)
class FrozenOperatorWorkingState:
    """Own beta/theta while a source-replayed Newton system is solved.

    The model and family arguments are private snapshots owned by the fit
    orchestrator. beta=None selects the literal first family mustart/eta.
    Its retention and row workspace are charged before constructing it.
    """

    model: PreparedOperatorModel
    family: ExponentialFamily
    beta: np.ndarray | None
    row_memory_budget_bytes: int
    state_memory_budget_bytes: int
    summary: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or value < 1
            for value in (self.row_memory_budget_bytes, self.state_memory_budget_bytes)
        ):
            raise ValueError("Working-state budgets must be positive integers.")
        context = FamilyExecutionContext.from_family(self.family)
        snapshot = self.family.execution_parameter_snapshot()
        if snapshot.theta_mode == "estimated":
            raise NotImplementedError("Operator fitting requires fixed theta.")
        beta = self.beta
        bound = 17 * self.model.n_coef if beta is not None else 0
        bound += 16 * len(snapshot.log_theta) + 512
        if self.summary is not None:
            bound += 512 + sum(128 + 4 * len(key) for key in self.summary)
        if bound > self.state_memory_budget_bytes:
            raise MemoryError("Frozen working-state ownership exceeds its budget.")
        theta_values = np.asarray(snapshot.log_theta, dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            theta = libm_exp(theta_values)
        if not np.all(np.isfinite(theta_values)) or np.any(
            ~np.isfinite(theta) | (theta <= 0)
        ):
            raise ValueError("Frozen operator theta must be finite and positive.")
        object.__setattr__(self, "retained_bound_bytes", bound)
        object.__setattr__(
            self, "summary", None if self.summary is None else dict(self.summary)
        )
        object.__setattr__(
            self,
            "summary_snapshot",
            None if self.summary is None else tuple(self.summary.items()),
        )
        if beta is not None:
            if (
                not isinstance(beta, np.ndarray)
                or beta.dtype != np.float64
                or beta.shape != (self.model.n_coef,)
            ):
                raise ValueError("Working beta must match the float64 operator model.")
            if not np.all(np.isfinite(beta)):
                raise ValueError("Working beta must be finite.")
            beta = beta.copy()
            beta.setflags(write=False)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(
            self,
            "parameters",
            jax.device_put(
                FamilyExecutionParameters.from_snapshot(snapshot), jax.devices("cpu")[0]
            ),
        )
        object.__setattr__(
            self,
            "beta_device",
            None if beta is None else jax.device_put(beta, jax.devices("cpu")[0]),
        )

    def validate(self) -> None:
        if FamilyExecutionContext.from_family(self.family) != self.context:
            raise RuntimeError("Frozen operator family/link configuration changed.")
        if self.family.execution_parameter_snapshot() != self.snapshot:
            raise RuntimeError("Frozen operator theta state changed.")
        if (
            self.summary is not None
            and tuple(self.summary.items()) != self.summary_snapshot
        ):
            raise RuntimeError("Frozen operator global family summary changed.")

    def evaluate(
        self, batch: RowBatch, rows: OperatorBatch
    ) -> tuple[DeviceOperatorBatch, OperatorFamilyRows]:
        """One owned geometry transfer and bounded actual-family row evaluation."""
        self.validate()
        B = self.model.batch_rows
        n = len(batch.row_positions)
        if n > B:
            raise MemoryError("Working batch exceeds the frozen row bound.")
        device = jax.devices("cpu")[0]
        geometry = to_device_operator_batch(
            _padded_geometry(rows, B),
            memory_budget_bytes=self.row_memory_budget_bytes,
            device=device,
        )
        padding = self.context.padding
        y = np.full(B, padding.response)
        weights = np.zeros(B)
        offset = np.zeros(B)
        valid = np.zeros(B, dtype=bool)
        y[:n], weights[:n], offset[:n], valid[:n] = (
            batch.y,
            batch.weight,
            batch.offset,
            True,
        )
        if self.beta is None:
            initial = self.family.initial_working_state_cpu(
                y, weights, valid, summary=self.summary
            )
            if not initial.input_ok or not initial.domain_ok:
                raise ValueError("Family mustart has invalid real inputs/domain.")
            eta = jax.device_put(initial.eta, device)
        else:
            eta = matvec_batch(geometry, self.beta_device) + jax.device_put(
                offset, device
            )
        quantities = operator_family_rows(
            eta,
            jax.device_put(y, device),
            jax.device_put(weights, device),
            jax.device_put(offset, device),
            jax.device_put(valid, device),
            self.parameters,
            self.family,
            self.context,
        )
        return geometry, quantities

    def curvature(self, batch: RowBatch, rows: OperatorBatch) -> np.ndarray:
        _, quantities = self.evaluate(batch, rows)
        if not bool(quantities.input_ok):
            raise ValueError("Family working rows have invalid real inputs.")
        if not bool(quantities.domain_ok):
            raise ValueError("Family working rows lie outside the predictor domain.")
        if not bool(quantities.finite_system):
            raise FloatingPointError("Family observed working system is nonfinite.")
        return np.asarray(quantities.observed_weight)[: len(batch.row_positions)]

    def fisher_curvature(self, batch: RowBatch, rows: OperatorBatch) -> np.ndarray:
        """Expected curvature at this frozen beta/theta, without row retention."""
        _, quantities = self.evaluate(batch, rows)
        if not bool(quantities.input_ok) or not bool(quantities.domain_ok):
            raise ValueError("Family Fisher rows have invalid real inputs/domain.")
        n = len(batch.row_positions)
        mu_eta = np.asarray(self.family.link.mu_eta(quantities.eta))[:n]
        variance = np.asarray(self.family.variance(quantities.mu))[:n]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            weight = np.where(
                batch.weight > 0,
                batch.weight * mu_eta**2 / variance,
                0.0,
            )
        if not np.all(np.isfinite(weight)) or np.any(weight < 0):
            raise FloatingPointError("Frozen family Fisher curvature is invalid.")
        return weight


def _evaluate_geometry(model: PreparedOperatorModel, batch: RowBatch) -> OperatorBatch:
    spec = model.modest.predict_spec
    variables = tuple(
        dict.fromkeys(
            [
                *spec.required_covariates,
                *(
                    variable
                    for term in model.random_effects
                    for variable in term.variables
                ),
            ]
        )
    )
    subset = replace(batch, columns={name: batch.columns[name] for name in variables})
    if not variables and model.modest.n_coef == 0:
        return OperatorBatch(
            np.empty((len(batch.row_positions), 0)), (), (), model.n_coef
        )
    return model.evaluate_batch(subset)


def _checked_fit_source(
    model: PreparedOperatorModel, source: RowSource, reserved: int, budget: int
) -> _CheckedSource:
    if reserved > budget:
        raise MemoryError("Combined operator fit workspace exceeds its budget.")
    checked = _CheckedSource(
        source,
        OperatorPreparationControl(
            model.batch_rows, budget, max(1, model.modest_raw_width)
        ),
        reserved,
    )
    checked.digest = model.source_replay_digest
    return checked


def _reduce_rhs(
    state: FrozenOperatorWorkingState,
    source: RowSource,
    *,
    reserved: int,
    budget: int,
    gradient: bool = False,
) -> np.ndarray:
    """One complete checked source pass accumulates the literal source Wz."""
    checked = _checked_fit_source(state.model, source, reserved, budget)
    accumulator = jnp.zeros(state.model.n_coef, dtype=jnp.float64)
    for batch in checked.scan(state.model.batch_rows):
        rows = _evaluate_geometry(state.model, batch)
        geometry, quantities = state.evaluate(batch, rows)
        if not bool(quantities.finite_system):
            raise FloatingPointError("Operator RHS has invalid/nonfinite family rows.")
        accumulator = accumulate_rmatvec_batch(
            accumulator,
            geometry,
            quantities.half_gradient if gradient else quantities.weighted_response,
        )
        accumulator.block_until_ready()
        del batch, rows, geometry, quantities
    return np.asarray(accumulator).copy()


def _reduce_objective(
    state: FrozenOperatorWorkingState,
    source: RowSource,
    system: FrozenScanSystem,
    *,
    reserved: int,
    budget: int,
) -> tuple[float, float]:
    """Stable direct row deviance; invalid trial domains return infinity."""
    if state.beta is None:
        raise ValueError("A trial objective requires coefficients.")
    checked = _checked_fit_source(state.model, source, reserved, budget)
    deviance = 0.0
    correction = 0.0
    domain_ok = True
    for batch in checked.scan(state.model.batch_rows):
        rows = _evaluate_geometry(state.model, batch)
        geometry, quantities = state.evaluate(batch, rows)
        domain_ok = domain_ok and bool(quantities.domain_ok)
        value = float(quantities.deviance)
        if not np.isfinite(value):
            domain_ok = False
        # Neumaier/Kahan-style scalar merge avoids moment cancellation.
        increment = value - correction
        total = deviance + increment
        correction = (total - deviance) - increment
        deviance = total
        del batch, rows, geometry, quantities
    if not domain_ok:
        return np.inf, np.inf
    action = system._penalty(system._penalties, state.beta_device, system._rho)
    penalty = float(jnp.vdot(state.beta_device, action))
    del action
    return deviance, deviance + penalty


def _latest_independent_groups(rows: np.ndarray) -> list[int]:
    """Return the latest natural-order group rows spanning a small null space."""
    target = rows.shape[1]
    pivot_scale = np.max(np.linalg.norm(rows, axis=1), initial=0.0)
    basis: list[np.ndarray] = []
    selected: list[int] = []
    for group in range(len(rows) - 1, -1, -1):
        residual = rows[group].copy()
        for direction in basis:
            residual -= np.dot(residual, direction) * direction
        size = np.linalg.norm(residual)
        if size > 1e-10 * pivot_scale:
            basis.append(residual / size)
            selected.append(group)
            if len(selected) == target:
                break
    if len(selected) != target:
        raise np.linalg.LinAlgError(
            "Mixed source group null-space rank is inconsistent."
        )
    return selected


def _multiple_indexed_null_coefficients(
    model: PreparedOperatorModel,
    source: RowSource,
    diagonal: np.ndarray,
    rhs: np.ndarray,
    *,
    reserved: int,
    budget: int,
) -> np.ndarray:
    """Source QR coordinates for a certified full-rank indexed X.

    The normalized unpenalized Gram has identity diagonal blocks. For two
    terms, the cross-block spectral norm is bounded by the geometric mean
    of absolute row/column sums. With more terms, absolute off-diagonal row
    sums provide a Gershgorin lower bound. Neither path retains an observation,
    pair table or global coefficient square. Full rank makes R's natural QR
    coordinates unique.
    """
    terms = model.random_effects
    p = model.n_coef
    occupied = diagonal > 0
    if not np.any(occupied):
        return np.zeros(p)
    inverse_sqrt = np.zeros(p)
    inverse_sqrt[occupied] = 1.0 / np.sqrt(diagonal[occupied])
    row_sum = np.zeros(p)
    checked = _checked_fit_source(model, source, reserved, budget)
    for batch in checked.scan(model.batch_rows):
        indexed = [
            (term.col_start + ids, amplitude)
            for term in terms
            for ids, amplitude in (term.evaluate(batch.columns),)
        ]
        for first_position, (first_columns, first_amplitude) in enumerate(indexed):
            for second_columns, second_amplitude in indexed[first_position + 1 :]:
                cross = (
                    np.abs(first_amplitude * second_amplitude)
                    * inverse_sqrt[first_columns]
                    * inverse_sqrt[second_columns]
                )
                np.add.at(row_sum, first_columns, cross)
                np.add.at(row_sum, second_columns, cross)
        del batch, indexed
    if len(terms) == 2:
        first, second = terms
        first_max = np.max(row_sum[first.col_start : first.col_start + first.n_coef])
        second_max = np.max(
            row_sum[second.col_start : second.col_start + second.n_coef]
        )
        cross_bound = float(np.sqrt(first_max * second_max))
    else:
        cross_bound = float(np.max(row_sum[occupied]))
    if not np.isfinite(cross_bound) or cross_bound >= 1.0 - 1e-12:
        raise NotImplementedError(
            "Multiple indexed source QR needs a stronger full-rank certificate."
        )

    def normalized_action(vector: np.ndarray) -> np.ndarray:
        coefficients = inverse_sqrt * vector
        output = np.zeros(p)
        action_source = _checked_fit_source(model, source, reserved, budget)
        for batch in action_source.scan(model.batch_rows):
            indexed = [
                (term.col_start + ids, amplitude)
                for term in terms
                for ids, amplitude in (term.evaluate(batch.columns),)
            ]
            fitted = sum(
                (amplitude * coefficients[columns] for columns, amplitude in indexed),
                np.zeros(len(batch.row_positions)),
            )
            for columns, amplitude in indexed:
                np.add.at(output, columns, amplitude * fitted)
            del batch, indexed, fitted
        return inverse_sqrt * output

    normalized_rhs = inverse_sqrt * rhs
    solution = np.zeros(p)
    residual = normalized_rhs.copy()
    direction = residual.copy()
    residual_square = float(residual @ residual)
    tolerance = max(1e-14, 1e-13 * np.linalg.norm(normalized_rhs))
    max_iterations = min(1000, max(64, 4 * int(np.count_nonzero(occupied))))
    for _ in range(max_iterations):
        if np.sqrt(residual_square) <= tolerance:
            break
        action = normalized_action(direction)
        curvature = float(direction @ action)
        if not np.isfinite(curvature) or curvature <= 0:
            raise np.linalg.LinAlgError(
                "Certified crossed numeric null action lost positive curvature."
            )
        step = residual_square / curvature
        solution += step * direction
        residual -= step * action
        next_square = float(residual @ residual)
        if not np.isfinite(next_square):
            raise FloatingPointError("Crossed numeric null solve is nonfinite.")
        direction = residual + (next_square / residual_square) * direction
        residual_square = next_square
    coefficients = inverse_sqrt * solution
    # A fresh replay checks the actual unpenalized system rather than the
    # Krylov recurrence. The spectral lower bound turns its residual into a
    # coefficient-error bound in R's unique natural coordinates.
    normalized_true_residual = normalized_rhs - normalized_action(solution)
    true_residual = np.zeros(p)
    true_residual[occupied] = (
        normalized_true_residual[occupied] / inverse_sqrt[occupied]
    )
    lower = float(np.min(diagonal[occupied]) * (1.0 - cross_bound))
    error_bound = float(np.linalg.norm(true_residual) / lower)
    if not np.isfinite(error_bound) or error_bound > (
        1e-12 + 1e-10 * np.max(np.abs(coefficients), initial=0.0)
    ):
        raise np.linalg.LinAlgError(
            "Multiple indexed source QR solve did not certify its coordinates."
        )
    return coefficients


def _mixed_full_rank_null_coefficients(
    model: PreparedOperatorModel,
    source: RowSource,
    diagonal: np.ndarray,
    rhs: np.ndarray,
    modest_gram: np.ndarray,
    modest_rhs: np.ndarray,
    coupling: np.ndarray,
    *,
    reserved: int,
    budget: int,
) -> np.ndarray:
    """Bounded natural QR coordinates for a certified full-rank mixed X.

    On occupied indexed columns, their normalized Gram is bounded below by
    ``(1-t)D``. A K-square Schur certificate then proves the full unpenalized
    Gram is SPD. Matrix-free CG and a fresh true residual certify the unique
    coordinates, irrespective of whether indexed or modest columns came first.
    """
    p = model.n_coef
    modest_columns = np.asarray(model.modest_columns, dtype=np.int64)
    occupied = diagonal > 0
    inverse_sqrt = np.zeros(p)
    inverse_sqrt[occupied] = 1.0 / np.sqrt(diagonal[occupied])
    row_sum = np.zeros(p)
    checked = _checked_fit_source(model, source, reserved, budget)
    for batch in checked.scan(model.batch_rows):
        indexed = [
            (term.col_start + ids, amplitude)
            for term in model.random_effects
            for ids, amplitude in (term.evaluate(batch.columns),)
        ]
        for first_position, (first_columns, first_amplitude) in enumerate(indexed):
            for second_columns, second_amplitude in indexed[first_position + 1 :]:
                cross = (
                    np.abs(first_amplitude * second_amplitude)
                    * inverse_sqrt[first_columns]
                    * inverse_sqrt[second_columns]
                )
                np.add.at(row_sum, first_columns, cross)
                np.add.at(row_sum, second_columns, cross)
        del batch, indexed
    if len(model.random_effects) == 2:
        first, second = model.random_effects
        first_max = np.max(row_sum[first.col_start : first.col_start + first.n_coef])
        second_max = np.max(
            row_sum[second.col_start : second.col_start + second.n_coef]
        )
        cross_bound = float(np.sqrt(first_max * second_max))
    else:
        cross_bound = float(np.max(row_sum[occupied], initial=0.0))
    if not np.isfinite(cross_bound) or cross_bound >= 1.0 - 1e-12:
        raise NotImplementedError(
            "Mixed source QR needs a stronger indexed full-rank certificate."
        )
    indexed_lower = (1.0 - cross_bound) * diagonal[occupied]
    schur = modest_gram - coupling[occupied].T @ (
        coupling[occupied] / indexed_lower[:, None]
    )
    try:
        modest_factor = np.linalg.cholesky(modest_gram)
        np.linalg.cholesky((schur + schur.T) * 0.5)
    except np.linalg.LinAlgError as error:
        raise NotImplementedError(
            "Mixed source QR needs a stronger full-rank Schur certificate."
        ) from error
    schur_lower = float(np.linalg.eigvalsh((schur + schur.T) * 0.5)[0])
    indexed_min = float(np.min(indexed_lower))
    normalized_coupling_norm = float(
        np.linalg.norm(coupling[occupied] / indexed_lower[:, None])
    )
    eigen_lower = min(schur_lower, indexed_min) / (1.0 + normalized_coupling_norm) ** 2
    if not np.isfinite(eigen_lower) or eigen_lower <= 0:
        raise np.linalg.LinAlgError("Mixed source QR lower eigenvalue is invalid.")

    def from_coefficients(vector: np.ndarray) -> np.ndarray:
        coefficients = np.zeros(p)
        coefficients[modest_columns] = np.linalg.solve(
            modest_factor.T, vector[modest_columns]
        )
        coefficients[occupied] = inverse_sqrt[occupied] * vector[occupied]
        return coefficients

    def normalized_action(vector: np.ndarray) -> np.ndarray:
        coefficients = from_coefficients(vector)
        output = np.zeros(p)
        action_source = _checked_fit_source(model, source, reserved, budget)
        for batch in action_source.scan(model.batch_rows):
            rows = _evaluate_geometry(model, batch)
            fitted = rows.modest_values @ coefficients[modest_columns]
            for ids, amplitude, start, _ in rows.indexed_rows:
                fitted += amplitude * coefficients[start + ids]
            output[modest_columns] += rows.modest_values.T @ fitted
            for ids, amplitude, start, count in rows.indexed_rows:
                np.add.at(output[start : start + count], ids, amplitude * fitted)
            del batch, rows, fitted
        normalized = np.zeros(p)
        normalized[modest_columns] = np.linalg.solve(
            modest_factor, output[modest_columns]
        )
        normalized[occupied] = inverse_sqrt[occupied] * output[occupied]
        return normalized

    full_rhs = rhs.copy()
    full_rhs[modest_columns] = modest_rhs
    normalized_rhs = np.zeros(p)
    normalized_rhs[modest_columns] = np.linalg.solve(
        modest_factor, full_rhs[modest_columns]
    )
    normalized_rhs[occupied] = inverse_sqrt[occupied] * full_rhs[occupied]
    solution = np.zeros(p)
    residual = normalized_rhs.copy()
    direction = residual.copy()
    residual_square = float(residual @ residual)
    tolerance = max(1e-14, 1e-13 * np.linalg.norm(normalized_rhs))
    max_iterations = min(
        1000, max(64, 4 * (int(np.count_nonzero(occupied)) + len(modest_columns)))
    )
    for _ in range(max_iterations):
        if np.sqrt(residual_square) <= tolerance:
            break
        action = normalized_action(direction)
        curvature = float(direction @ action)
        if not np.isfinite(curvature) or curvature <= 0:
            raise np.linalg.LinAlgError(
                "Certified mixed null action lost positive curvature."
            )
        step = residual_square / curvature
        solution += step * direction
        residual -= step * action
        next_square = float(residual @ residual)
        if not np.isfinite(next_square):
            raise FloatingPointError("Mixed source null solve is nonfinite.")
        direction = residual + (next_square / residual_square) * direction
        residual_square = next_square
    coefficients = from_coefficients(solution)
    normalized_true_residual = normalized_rhs - normalized_action(solution)
    true_residual = np.zeros(p)
    true_residual[modest_columns] = (
        modest_factor @ normalized_true_residual[modest_columns]
    )
    true_residual[occupied] = (
        normalized_true_residual[occupied] / inverse_sqrt[occupied]
    )
    error_bound = float(np.linalg.norm(true_residual) / eigen_lower)
    if not np.isfinite(error_bound) or error_bound > (
        1e-12 + 1e-10 * np.max(np.abs(coefficients), initial=0.0)
    ):
        raise np.linalg.LinAlgError(
            "Mixed source QR solve did not certify its coordinates."
        )
    return coefficients


def _initial_null_beta(
    model: PreparedOperatorModel,
    source: RowSource,
    family: ExponentialFamily,
    *,
    reserved: int,
    budget: int,
) -> tuple[np.ndarray, dict[str, float], float, bool]:
    """Source constant null coefficients for provable bounded layouts.

    R get.null.coef QR-projects link(mean(initialized y)). An intercept
    represents this exactly. A leading pure-factor RE has disjoint indicator
    columns, so each observed level has that coefficient and unused levels
    and later indexed terms have zero, as in QR. Bounded modest-only layouts
    use the pinned natural-column QR projection. Full-rank modest variation
    within the first factor makes earlier modest coefficients zero; a small
    within-group null space identifies natural-column rank dependencies.
    """
    intercept = model.modest.predict_spec.has_intercept
    first_indexed_column = (
        model.random_effects[0].col_start if model.random_effects else None
    )
    leading_modest_columns = (
        tuple(
            column for column in model.modest_columns if column < first_indexed_column
        )
        if first_indexed_column is not None
        else model.modest_columns
    )
    leading_modest_local = tuple(
        position
        for position, column in enumerate(model.modest_columns)
        if column in leading_modest_columns
    )
    modest_precedes_first_indexed = bool(model.modest_columns) and (
        len(leading_modest_columns) == len(model.modest_columns)
    )
    first_indexed_precedes_modest = (
        bool(model.modest_columns)
        and all(column > first_indexed_column for column in model.modest_columns)
        if first_indexed_column is not None
        else False
    )
    pure_indexed = (
        not intercept
        and bool(model.random_effects)
        and not model.random_effects[0].numeric_variables
        and (model.modest.n_coef == 0 or first_indexed_precedes_modest)
    )
    numeric_indexed = (
        not intercept
        and model.modest.n_coef == 0
        and len(model.random_effects) == 1
        and bool(model.random_effects[0].numeric_variables)
    )
    multiple_indexed = (
        not intercept
        and model.modest.n_coef == 0
        and len(model.random_effects) >= 2
        and bool(model.random_effects[0].numeric_variables)
    )
    mixed_indexed = (
        not intercept
        and model.modest.n_coef > 0
        and bool(model.random_effects)
        and not model.random_effects[0].numeric_variables
        and bool(leading_modest_columns)
    )
    mixed_numeric = (
        not intercept
        and model.modest.n_coef > 0
        and len(model.random_effects) == 1
        and bool(model.random_effects[0].numeric_variables)
        and modest_precedes_first_indexed
    )
    mixed_full_rank = (
        not intercept
        and model.modest.n_coef > 0
        and bool(model.random_effects)
        and bool(model.random_effects[0].numeric_variables)
        and not mixed_numeric
    )
    general_mixed = (
        not intercept
        and model.modest.n_coef > 0
        and bool(model.random_effects)
        and not pure_indexed
        and bool(leading_modest_columns)
        and (mixed_indexed or mixed_numeric)
    )
    modest_only = not intercept and not model.random_effects
    if not any(
        (
            intercept,
            pure_indexed,
            numeric_indexed,
            multiple_indexed,
            mixed_indexed,
            mixed_numeric,
            mixed_full_rank,
            general_mixed,
            modest_only,
        )
    ):
        raise NotImplementedError(
            "General no-intercept operator null projection is not yet available."
        )
    full_rank_modest_width = len(model.modest_columns)
    mixed_reserve = (
        16 * model.n_coef * full_rank_modest_width
        + 64 * model.n_coef
        + 256 * full_rank_modest_width**2
        if mixed_full_rank
        else 0
    )
    checked = _checked_fit_source(model, source, reserved + mixed_reserve, budget)
    first_indexed = model.random_effects[0] if model.random_effects else None
    active = np.zeros(first_indexed.n_coef, dtype=bool) if pure_indexed else None
    numeric_sum = np.zeros(first_indexed.n_coef) if numeric_indexed else None
    numeric_square = np.zeros(first_indexed.n_coef) if numeric_indexed else None
    indexed_diagonal = np.zeros(model.n_coef) if multiple_indexed else None
    indexed_rhs = np.zeros(model.n_coef) if multiple_indexed else None
    full_rank_diagonal = np.zeros(model.n_coef) if mixed_full_rank else None
    full_rank_rhs = np.zeros(model.n_coef) if mixed_full_rank else None
    full_rank_coupling = (
        np.zeros((model.n_coef, full_rank_modest_width)) if mixed_full_rank else None
    )
    full_rank_modest_gram = (
        np.zeros((full_rank_modest_width, full_rank_modest_width))
        if mixed_full_rank
        else None
    )
    full_rank_modest_rhs = np.zeros(full_rank_modest_width) if mixed_full_rank else None
    mixed_numeric_sum = np.zeros(first_indexed.n_coef) if mixed_numeric else None
    mixed_numeric_square = np.zeros(first_indexed.n_coef) if mixed_numeric else None
    K = len(leading_modest_columns)
    group_count = (
        np.zeros(first_indexed.n_coef, dtype=np.int64) if mixed_indexed else None
    )
    group_sum = (
        np.zeros((first_indexed.n_coef, K)) if mixed_indexed or mixed_numeric else None
    )
    modest_gram = np.zeros((K, K)) if mixed_indexed or mixed_numeric else None
    modest_rhs = np.zeros(K) if mixed_numeric else None
    modest_null_qr = None
    response_scale = 0.0
    scaled_response_sum = np.longdouble(0.0)
    count = 0
    summary = None
    offset_was_nonzero = False
    for batch in checked.scan(model.batch_rows):
        y = family.execution_initial_response_cpu(batch.y, batch.weight)
        n = len(y)
        if n == 0:
            del batch, y
            continue
        offset_was_nonzero = offset_was_nonzero or not np.allclose(batch.offset, 0.0)
        if not family.response_support.check(y):
            raise ValueError("Family null initialization has invalid real inputs.")
        leaf = family.execution_summary_from_batch(
            np.asarray(batch.y), np.asarray(batch.weight), np.ones(n, dtype=bool)
        )
        summary = (
            leaf if summary is None else family.merge_execution_summaries(summary, leaf)
        )
        batch_scale = float(np.max(np.abs(y), initial=0.0))
        if batch_scale > response_scale:
            if response_scale:
                scaled_response_sum *= np.longdouble(response_scale / batch_scale)
            response_scale = batch_scale
        if response_scale:
            scaled_response_sum += np.sum(
                np.asarray(y, dtype=np.longdouble) / np.longdouble(response_scale),
                dtype=np.longdouble,
            )
        count += n
        if pure_indexed:
            indices, amplitudes = model.random_effects[0].evaluate(batch.columns)
            if not np.all(amplitudes == 1):
                raise RuntimeError(
                    "Pure indexed null layout is not an indicator design."
                )
            active[indices] = True
            del indices, amplitudes
        if numeric_indexed:
            indices, amplitudes = first_indexed.evaluate(batch.columns)
            np.add.at(numeric_sum, indices, amplitudes)
            np.add.at(numeric_square, indices, amplitudes**2)
            del indices, amplitudes
        if multiple_indexed:
            for term in model.random_effects:
                indices, amplitudes = term.evaluate(batch.columns)
                np.add.at(
                    indexed_diagonal[term.col_start : term.col_start + term.n_coef],
                    indices,
                    amplitudes**2,
                )
                np.add.at(
                    indexed_rhs[term.col_start : term.col_start + term.n_coef],
                    indices,
                    amplitudes,
                )
                del indices, amplitudes
        if mixed_full_rank:
            rows = _evaluate_geometry(model, batch)
            X = rows.modest_values
            full_rank_modest_gram += X.T @ X
            full_rank_modest_rhs += np.sum(X, axis=0)
            for ids, amplitudes, start, block_width in rows.indexed_rows:
                np.add.at(
                    full_rank_diagonal[start : start + block_width],
                    ids,
                    amplitudes**2,
                )
                np.add.at(full_rank_rhs[start : start + block_width], ids, amplitudes)
                np.add.at(
                    full_rank_coupling[start : start + block_width],
                    ids,
                    amplitudes[:, None] * X,
                )
            del rows, X
        if general_mixed:
            rows = _evaluate_geometry(model, batch)
            X = rows.modest_values[:, leading_modest_local]
            modest_null_qr = qr_update(modest_null_qr, X, np.ones(n), n_coef=K)
            if mixed_indexed or mixed_numeric:
                indices, amplitudes, _, _ = rows.indexed_rows[0]
                if mixed_indexed:
                    if not np.all(amplitudes == 1):
                        raise RuntimeError(
                            "Mixed indexed null layout is not an indicator design."
                        )
                    np.add.at(group_count, indices, 1)
                    np.add.at(group_sum, indices, X)
                else:
                    np.add.at(mixed_numeric_sum, indices, amplitudes)
                    np.add.at(mixed_numeric_square, indices, amplitudes**2)
                    np.add.at(group_sum, indices, amplitudes[:, None] * X)
                    modest_rhs += np.sum(X, axis=0)
                modest_gram += X.T @ X
                del indices, amplitudes
            del rows, X
        del batch, y, leaf
    if summary is None or not family.execution_summary_input_ok(summary):
        raise ValueError("Family null initialization has invalid real inputs.")
    finalized = family.finalize_execution_summary(tuple(float(x) for x in summary))
    mean = float(
        np.longdouble(response_scale) * (scaled_response_sum / np.longdouble(count))
    )
    scalar = float(np.asarray(family.link.initial_link_cpu(np.asarray(mean))))
    if not np.isfinite(scalar):
        raise ValueError("Finite source constant-null initialization is unavailable.")
    beta = np.zeros(model.n_coef)
    modest_constant = None
    modest_rank = K
    if general_mixed:
        modest_constant = project_null_coefficients(modest_null_qr, 1.0)
        modest_rank = modest_constant.rank
        residual = (
            modest_null_qr.qtz
            - modest_null_qr.R @ modest_constant.coefficients[modest_null_qr.pivots]
        )
        residual_square = modest_null_qr.residual_tail + float(residual @ residual)
        if residual_square > 1e-20 * count:
            modest_constant = None
    if intercept:
        beta[model.modest_columns[0]] = scalar
    elif modest_constant is not None:
        beta[np.asarray(leading_modest_columns)] = scalar * modest_constant.coefficients
    elif pure_indexed:
        beta[first_indexed.col_start + np.flatnonzero(active)] = scalar
    elif numeric_indexed:
        occupied = numeric_square > 0
        beta[first_indexed.col_start + np.flatnonzero(occupied)] = (
            scalar * numeric_sum[occupied] / numeric_square[occupied]
        )
    elif multiple_indexed:
        if scalar != 0.0:
            beta = scalar * _multiple_indexed_null_coefficients(
                model,
                source,
                indexed_diagonal,
                indexed_rhs,
                reserved=reserved,
                budget=budget,
            )
    elif mixed_full_rank:
        if scalar != 0.0:
            beta = scalar * _mixed_full_rank_null_coefficients(
                model,
                source,
                full_rank_diagonal,
                full_rank_rhs,
                full_rank_modest_gram,
                full_rank_modest_rhs,
                full_rank_coupling,
                reserved=reserved + mixed_reserve,
                budget=budget,
            )
    elif mixed_indexed:
        occupied = group_count > 0
        grouped = group_sum[occupied]
        within = modest_gram - grouped.T @ (grouped / group_count[occupied, None])
        eigenvalues, eigenvectors = np.linalg.eigh((within + within.T) * 0.5)
        scale = np.max(np.abs(eigenvalues), initial=0.0)
        rank_cutoff = 1e-10 * max(scale, np.finfo(float).tiny)
        if np.any(eigenvalues < -rank_cutoff):
            raise np.linalg.LinAlgError("Mixed source within-group Gram is invalid.")
        null = eigenvectors[:, eigenvalues <= rank_cutoff]
        if null.shape[1]:
            # R's natural-column QR retains the modest columns first. Among
            # the first factor's indicators it drops the latest independent
            # group rows in the modest within-group null space. Only those
            # small coordinates need a source-order rank choice.
            if modest_rank < K:
                raise NotImplementedError(
                    "Rank-deficient modest source QR is not yet available."
                )
            try:
                np.linalg.cholesky((modest_gram + modest_gram.T) * 0.5)
            except np.linalg.LinAlgError as error:
                raise NotImplementedError(
                    "Rank-deficient modest source QR is not yet available."
                ) from error
            means = grouped / group_count[occupied, None]
            group_null = means @ null
            dropped = _latest_independent_groups(group_null)
            small = group_null[dropped]
            modest_beta = null @ np.linalg.solve(small, np.full(len(dropped), scalar))
            beta[np.asarray(leading_modest_columns)] = modest_beta
            group_beta = scalar - means @ modest_beta
            group_beta[dropped] = 0.0
            beta[first_indexed.col_start + np.flatnonzero(occupied)] = group_beta
        else:
            beta[first_indexed.col_start + np.flatnonzero(occupied)] = scalar
    elif mixed_numeric:
        occupied = mixed_numeric_square > 0
        grouped = group_sum[occupied]
        square = mixed_numeric_square[occupied]
        within = modest_gram - grouped.T @ (grouped / square[:, None])
        eigenvalues, eigenvectors = np.linalg.eigh((within + within.T) * 0.5)
        scale = np.max(np.abs(eigenvalues), initial=0.0)
        gram_scale = np.max(np.abs(modest_gram), initial=0.0)
        rank_cutoff = max(1e-10 * scale, 1e-14 * gram_scale)
        if np.any(eigenvalues < -rank_cutoff):
            raise np.linalg.LinAlgError("Mixed numeric within-group Gram is invalid.")
        retained = eigenvalues > rank_cutoff
        null = eigenvectors[:, ~retained]
        rhs = modest_rhs - grouped.T @ (mixed_numeric_sum[occupied] / square)
        if null.shape[1]:
            if modest_rank < K:
                raise NotImplementedError(
                    "Rank-deficient modest source QR is not yet available."
                )
            if np.linalg.norm(null.T @ rhs) > 1e-9 * (1.0 + np.linalg.norm(rhs)):
                raise np.linalg.LinAlgError(
                    "Mixed numeric source null RHS is outside its range."
                )
            base = (
                eigenvectors[:, retained]
                @ ((eigenvectors[:, retained].T @ rhs) / eigenvalues[retained])
                * scalar
            )
            group_null = grouped @ null
            dropped = _latest_independent_groups(group_null)
            target = scalar * mixed_numeric_sum[occupied][dropped]
            target -= grouped[dropped] @ base
            modest_beta = base + null @ np.linalg.solve(group_null[dropped], target)
        else:
            modest_beta = scalar * np.linalg.solve(within, rhs)
        beta[np.asarray(model.modest_columns)] = modest_beta
        group_beta = (
            scalar * mixed_numeric_sum[occupied] - grouped @ modest_beta
        ) / square
        if null.shape[1]:
            group_beta[dropped] = 0.0
        modest_residual = (
            scalar * modest_rhs - modest_gram @ modest_beta - grouped.T @ group_beta
        )
        group_residual = (
            scalar * mixed_numeric_sum[occupied]
            - grouped @ modest_beta
            - square * group_beta
        )
        residual_bound = max(
            np.max(np.abs(modest_residual), initial=0.0),
            np.max(np.abs(group_residual), initial=0.0),
        )
        rhs_scale = max(
            1.0,
            np.max(np.abs(scalar * modest_rhs), initial=0.0),
            np.max(np.abs(scalar * mixed_numeric_sum[occupied]), initial=0.0),
        )
        if not np.isfinite(residual_bound) or residual_bound > 1e-10 * rhs_scale:
            raise np.linalg.LinAlgError(
                "Mixed numeric source QR coordinates fail normal stationarity."
            )
        beta[first_indexed.col_start + np.flatnonzero(occupied)] = group_beta
    elif general_mixed:
        raise NotImplementedError(
            "General mixed no-intercept source QR is not yet available."
        )
    else:
        state = None
        for batch in checked.scan(model.batch_rows):
            if len(batch.row_positions):
                rows = _evaluate_geometry(model, batch)
                state = qr_update(
                    state,
                    rows.modest_values,
                    np.ones(len(batch.row_positions)),
                )
        beta[np.asarray(model.modest_columns)] = project_null_coefficients(
            state, scalar
        ).coefficients
    beta.setflags(write=False)
    return beta, finalized, mean, offset_was_nonzero


def _reduce_null_scale(
    model: PreparedOperatorModel,
    source: RowSource,
    family: ExponentialFamily,
    mean: float,
    *,
    reserved: int,
    budget: int,
) -> float:
    """Source get.null.coef scale: deviance at initialized mean divided by n."""
    snapshot = family.execution_parameter_snapshot()
    # nb()'s getTheta(TRUE) is the C library's exp of the stored log theta.
    theta = libm_exp(np.asarray(snapshot.log_theta, dtype=np.float64))
    checked = _checked_fit_source(model, source, reserved, budget)
    deviance = 0.0
    correction = 0.0
    for batch in checked.scan(model.batch_rows):
        y = family.execution_initial_response_cpu(batch.y, batch.weight)
        contributions = family.deviance_contributions_for_parameters(
            y,
            np.full(len(y), mean),
            batch.weight,
            theta,
        )
        increment = float(np.sum(contributions)) - correction
        total = deviance + increment
        correction = (total - deviance) - increment
        deviance = total
        del y, contributions, batch
    scale = deviance / model.n_obs
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Source null scale is invalid.")
    return scale


@dataclass(frozen=True)
class OperatorSourceDiagnostics:
    """Immutable scalar snapshot of actual source operations."""

    started_scans: int
    completed_scans: int
    yielded_batches: int
    yielded_rows: int
    fingerprint_checks: int
    normal_actions: int


@dataclass(frozen=True)
class OperatorFitDiagnostics:
    """Actual source passes and bounded memory/solver convergence evidence."""

    source: OperatorSourceDiagnostics
    preparation_scans: int
    initialization_scans: int
    rhs_scans: int
    objective_scans: int
    certificate_scans: int
    certificate_actions: int
    krylov_actions: int
    krylov_iterations: int
    completed_solves: int
    peak_bound_bytes: int
    family_executable_temporary_bytes: int
    forward_executable_temporary_bytes: int
    adjoint_executable_temporary_bytes: int
    accepted_objectives: tuple[float, ...]
    inner_status: int | None
    final_gradient_norm: float
    initial_null_scale: float | None


@dataclass(frozen=True)
class OperatorFitResult:
    """Private row-free fit handoff; public retention is built separately."""

    coefficients: np.ndarray
    log_lambda: np.ndarray
    model: PreparedOperatorModel
    family: ExponentialFamily
    deviance: float
    converged: bool
    n_iter: int
    convergence_info: str
    diagnostics: OperatorFitDiagnostics
    offset_was_nonzero: bool

    def __post_init__(self) -> None:
        beta = self.coefficients.copy()
        beta.setflags(write=False)
        object.__setattr__(self, "coefficients", beta)
        rho = self.log_lambda.copy()
        rho.setflags(write=False)
        object.__setattr__(self, "log_lambda", rho)


#: Residual refinements of a strictly additive (single-step) operator solve.
_STRICTLY_ADDITIVE_REFINEMENTS = 3


def _remaining_memory(budget: int, reserved: int) -> int:
    if reserved >= budget:
        raise MemoryError(
            "Combined accepted/trial operator storage exceeds its budget."
        )
    return budget - reserved


def _operator_family_row_bound(model: PreparedOperatorModel) -> int:
    """Shared working-family transfer/replay allowance for one operator batch."""
    B = model.batch_rows
    return (
        model.batch_workspace_bound(B)
        + 64 * B * (len(model.modest_columns) + 2 * len(model.random_effects))
        + 256 * B
    )


def _family_ownership_bound(family: ExponentialFamily) -> int:
    """Bound a private family snapshot before deepcopy owns any array."""
    seen = set()

    def visit(value) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if len(seen) > 2048:
            raise MemoryError("Family snapshot exceeds the bounded metadata domain.")
        if isinstance(value, np.ndarray):
            return 512 + value.nbytes
        if isinstance(value, (str, bytes)):
            return 256 + 4 * len(value)
        if isinstance(value, dict):
            return 512 + sum(64 + visit(k) + visit(v) for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return 512 + sum(32 + visit(v) for v in value)
        if hasattr(value, "__dict__"):
            return 512 + visit(vars(value))
        return 256

    return visit(family)


_DEFAULT_OPERATOR_CONTROL = OperatorControl()


def fit_operator_fixed_sp(
    model: PreparedOperatorModel,
    source: RowSource,
    family: ExponentialFamily,
    sp: np.ndarray,
    *,
    control: OperatorControl = _DEFAULT_OPERATOR_CONTROL,
    memory_budget_bytes: int,
    beta_start: np.ndarray | None = None,
) -> OperatorFitResult:
    """Host Newton fitting with actual observed rows and certified CPU PCG.

    This private component handles explicit smoothing parameters only. It
    retains neither observations nor a global square coefficient matrix.
    Signed observed curvature is retained when the bounded provider certifies
    total-system SPD; it is never replaced by Fisher curvature. Unsupported
    no-intercept coordinate layouts remain explicit source-projection limits.
    """
    if not isinstance(control, OperatorControl):
        raise ValueError("control must be an OperatorControl.")
    if (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, numbers.Integral)
        or memory_budget_bytes < 1
    ):
        raise ValueError("Operator memory budget must be a positive integer.")
    if not isinstance(source, RowSource) or source.n_rows != model.n_obs:
        raise ValueError("Operator fit source must match preparation.")
    if (
        not isinstance(sp, np.ndarray)
        or sp.dtype != np.float64
        or sp.shape != (model.penalties.n_penalties,)
        or np.any(~np.isfinite(sp) | (sp <= 0))
    ):
        raise ValueError("Fixed sp must be a matching positive float64 vector.")
    p, B = model.n_coef, model.batch_rows
    if beta_start is not None and (
        not isinstance(beta_start, np.ndarray)
        or beta_start.dtype != np.float64
        or beta_start.shape != (p,)
        or not np.all(np.isfinite(beta_start))
    ):
        raise ValueError(
            "Operator beta start must be a finite float64 coefficient vector."
        )
    if model.modest_raw_width > control.max_modest_coefficients:
        raise MemoryError("Prepared modest block exceeds the independent fit cap.")
    family_bound = _family_ownership_bound(family)
    context = FamilyExecutionContext.from_family(family)
    snapshot = family.execution_parameter_snapshot()
    if snapshot.theta_mode == "estimated":
        raise NotImplementedError("Operator fitting requires fixed theta.")
    # 80p covers simultaneously retained null/accepted/trial/RHS/gradient,
    # the PCG solution copy, and frozen host/device beta ownership. PCG's
    # separately preflighted 112p bound is added by FrozenScanSystem.solve.
    # Row allowance covers the additional family geometry transfer, padded
    # inputs, eta, AD row outputs and host curvature copy while normal-action
    # geometry remains live. Native compiler/source backing are excluded.
    row_bound = _operator_family_row_bound(model)
    fit_bound = (
        model.metadata_bound_bytes
        + family_bound
        + 80 * p
        + (
            128 * p
            if not model.modest.predict_spec.has_intercept
            and not model.modest.n_coef
            and len(model.random_effects) >= 2
            and model.random_effects[0].numeric_variables
            else 0
        )
        + (
            256 * p * p
            if not model.modest.predict_spec.has_intercept and not model.random_effects
            else 0
        )
        + (
            48 * p * len(model.modest_columns) + 320 * len(model.modest_columns) ** 2
            if not model.modest.predict_spec.has_intercept
            and bool(model.random_effects)
            and model.modest.n_coef > 0
            else 0
        )
        + (
            32 * p
            if not model.modest.predict_spec.has_intercept
            and not model.modest.n_coef
            and len(model.random_effects) == 1
            and model.random_effects[0].numeric_variables
            else 0
        )
        + 16 * len(sp)
        + 128 * control.max_iterations
        + 8192
    )
    _remaining_memory(memory_budget_bytes, fit_bound + row_bound)
    family = copy.deepcopy(family)
    _freeze_host_arrays(family, set())
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(snapshot)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    row = jax.ShapeDtypeStruct((B,), np.float64, sharding=sharding)
    boolean = jax.ShapeDtypeStruct((B,), np.bool_, sharding=sharding)
    executable = operator_family_rows.lower(
        row, row, row, row, boolean, parameters, family, context
    ).compile()
    family_scratch = executable.memory_analysis().temp_size_in_bytes
    geometry = DeviceOperatorBatch(
        jax.ShapeDtypeStruct((B, len(model.modest_columns)), np.float64),
        jax.ShapeDtypeStruct((len(model.modest_columns),), np.int64),
        tuple(
            DeviceIndexedRows(row, row, term.col_start, term.n_coef)
            for term in model.random_effects
        ),
        p,
    )
    # Level IDs are integer metadata, not coefficient-valued inputs.
    geometry = replace(
        geometry,
        indexed_rows=tuple(
            replace(term, indices=jax.ShapeDtypeStruct((B,), np.int64))
            for term in geometry.indexed_rows
        ),
    )
    vector = jax.ShapeDtypeStruct((p,), np.float64)
    forward_scratch = (
        matvec_batch.lower(geometry, vector)
        .compile()
        .memory_analysis()
        .temp_size_in_bytes
    )
    adjoint_scratch = (
        accumulate_rmatvec_batch.lower(vector, geometry, row)
        .compile()
        .memory_analysis()
        .temp_size_in_bytes
    )
    scratch = max(family_scratch, forward_scratch, adjoint_scratch)
    extra = fit_bound + row_bound + scratch
    system_budget = _remaining_memory(memory_budget_bytes, extra)
    diagnostics = ScanDiagnostics()
    counted = _CountedSource(source, diagnostics)
    before_initialization_scans = diagnostics.completed_scans
    beta, summary, mean, offset_was_nonzero = _initial_null_beta(
        model,
        counted,
        family,
        reserved=extra,
        budget=memory_budget_bytes,
    )
    if beta_start is not None:
        beta = beta_start.copy()
    initialization_scans = diagnostics.completed_scans - before_initialization_scans
    null_scale = 1.0
    if not family.scale_known:
        null_scale = _reduce_null_scale(
            model,
            counted,
            family,
            mean,
            reserved=extra,
            budget=memory_budget_bytes,
        )
        initialization_scans += 1
    # Supplied sp enter R's fit as lsp0 = log(sp) (mgcv.r gam.setup), C log.
    rho = libm_log(sp)

    def state(values):
        return FrozenOperatorWorkingState(
            model,
            family,
            values,
            row_bound,
            17 * p + 16384,
            summary,
        )

    old_objective = None
    deviance = np.inf
    objectives = []
    rhs_scans = objective_scans = certificate_scans = 0
    certificate_actions = 0
    actions = iterations = solves = 0
    peak = extra
    status = None
    converged = False
    gradient_norm = math.inf
    info = "Maximum operator iterations reached."
    for iteration in range(1, control.max_iterations + 1):
        working = state(None if iteration == 1 else beta)
        before_certificate_scans = diagnostics.completed_scans
        system = FrozenScanSystem(
            model,
            counted,
            rho,
            curvature_from_rows=working.curvature,
            memory_budget_bytes=system_budget,
            use_coarse=control.use_coarse,
        )
        certificate_scans += diagnostics.completed_scans - before_certificate_scans
        certificate_actions += system.certification_actions
        diagnostics.normal_actions += system.certification_actions
        peak = max(peak, extra + system.construction_peak_bound_bytes)
        reserve = extra + system.retained_bound_bytes
        _remaining_memory(memory_budget_bytes, reserve)
        if old_objective is None:
            null_state = state(beta)
            deviance, old_objective = _reduce_objective(
                null_state,
                counted,
                system,
                reserved=reserve,
                budget=memory_budget_bytes,
            )
            objective_scans += 1
            del null_state
            if not np.isfinite(old_objective):
                raise ValueError(
                    "Source constant-null coefficients have invalid domain."
                )
            objectives.append(old_objective)
        rhs = _reduce_rhs(
            working,
            counted,
            reserved=reserve,
            budget=memory_budget_bytes,
        )
        rhs_scans += 1
        solved = system.solve(
            rhs,
            control=PCGControl(
                rtol=control.cg_rtol,
                atol=control.cg_atol,
                max_iterations=control.cg_max_iterations,
            ),
        )
        actions += solved.matvec_count
        diagnostics.normal_actions += solved.matvec_count
        iterations += solved.iterations
        status = solved.status
        peak = max(
            peak,
            extra + system.action_peak_bound_bytes + solved.additional_peak_bound_bytes,
        )
        if not solved.converged:
            info = f"Operator PCG failed with status {status}."
            del solved, rhs, system, working
            break
        solves += 1
        candidate = solved.solution.copy()
        del solved, rhs, working
        if context.capabilities.single_step_pirls:
            # A strictly additive system is solved once, so this solution is
            # final. Refine it against its true residual, reduced row by row
            # at the frozen state as -(X'(half gradient) + S beta); the
            # normal-form action b - H beta loses that residual to cancellation.
            for _ in range(_STRICTLY_ADDITIVE_REFINEMENTS):
                residual_state = state(candidate)
                residual = -_reduce_rhs(
                    residual_state,
                    counted,
                    reserved=reserve,
                    budget=memory_budget_bytes,
                    gradient=True,
                )
                residual -= np.asarray(
                    system._penalty(
                        system._penalties, residual_state.beta_device, system._rho
                    )
                )
                rhs_scans += 1
                del residual_state
                correction = system.solve(
                    residual,
                    control=PCGControl(
                        rtol=control.cg_rtol,
                        atol=control.cg_atol,
                        max_iterations=control.cg_max_iterations,
                    ),
                )
                actions += correction.matvec_count
                diagnostics.normal_actions += correction.matvec_count
                iterations += correction.iterations
                peak = max(
                    peak,
                    extra
                    + system.action_peak_bound_bytes
                    + correction.additional_peak_bound_bytes,
                )
                step = correction.solution
                del correction, residual
                if not np.all(np.isfinite(step)):
                    break
                candidate = candidate + step
                if np.max(np.abs(step), initial=0.0) <= np.finfo(float).eps * np.max(
                    np.abs(candidate), initial=0.0
                ):
                    break
        trial = state(candidate)
        divergence = 10 * (0.1 + abs(old_objective)) * np.sqrt(np.finfo(float).eps)
        for halving in range(101):
            deviance, objective = _reduce_objective(
                trial,
                counted,
                system,
                reserved=reserve,
                budget=memory_budget_bytes,
            )
            objective_scans += 1
            if np.isfinite(objective) and objective <= old_objective + divergence:
                break
            if halving == 100:
                raise RuntimeError(
                    "Operator step halving could not recover a valid step."
                )
            del trial
            candidate = (candidate + beta) * 0.5
            trial = state(candidate)
        relative_change = abs(objective - old_objective) / (0.1 + abs(objective))
        gradient = _reduce_rhs(
            trial,
            counted,
            reserved=reserve,
            budget=memory_budget_bytes,
            gradient=True,
        )
        rhs_scans += 1
        gradient += np.asarray(
            system._penalty(system._penalties, trial.beta_device, system._rho)
        )
        gradient_norm = 2 * np.max(np.abs(gradient), initial=0.0)
        if not np.isfinite(gradient_norm):
            raise FloatingPointError("Accepted operator gradient is nonfinite.")
        del gradient, system
        beta = candidate
        old_objective = objective
        objectives.append(objective)
        del trial, candidate
        # A strictly additive working system (gam.fit3.r:227-228,445) accepts
        # its first valid coefficient solve; see ``single_step_pirls``.
        if context.capabilities.single_step_pirls or (
            relative_change < control.tolerance
            and gradient_norm <= control.tolerance * (abs(objective) + null_scale)
        ):
            converged = True
            info = "Operator observed Newton fit converged."
            break
    result_diagnostics = OperatorFitDiagnostics(
        OperatorSourceDiagnostics(**vars(diagnostics)),
        model.completed_preparation_scans,
        initialization_scans,
        rhs_scans,
        objective_scans,
        certificate_scans,
        certificate_actions,
        actions,
        iterations,
        solves,
        peak,
        family_scratch,
        forward_scratch,
        adjoint_scratch,
        tuple(objectives),
        status,
        float(gradient_norm),
        None if family.scale_known else float(null_scale),
    )
    return OperatorFitResult(
        beta,
        rho,
        model,
        family,
        deviance,
        converged,
        iteration,
        info,
        result_diagnostics,
        offset_was_nonzero,
    )
