"""Internal operator EFS policy with explicit score-calibration boundaries.

The policy follows pinned ``efsudr`` for known scale. It owns accepted/trial
states, never substitutes Monte Carlo standard error for a score bound, and
refines common probes/depth before declaring a comparison indeterminate.
"""

from __future__ import annotations

import math
import numbers
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from jaxgam.control import EFSControl, OperatorControl
from jaxgam.data.source import RowSource
from jaxgam.execution.operator_efs_provider import (
    MatchedOperatorEFSComponents,
    MatchedOperatorEFSProvider,
    operator_efs_fit_fingerprint,
)
from jaxgam.execution.operator_fit import OperatorFitResult, fit_operator_fixed_sp
from jaxgam.execution.operator_logdet import OperatorLogDetControl
from jaxgam.execution.operator_score import compare_operator_scores
from jaxgam.families.base import ExponentialFamily
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.fitting.iterative import CONVERGED
from jaxgam.formula.operator_prepare import PreparedOperatorModel
from jaxgam.links.links import IdentityLink
from jaxgam.penalties.structure import IdentityPenalty


@dataclass(frozen=True)
class OperatorEFSState:
    """One frozen fit and all statistics at its exact rho/beta/theta state."""

    fit: OperatorFitResult
    components: MatchedOperatorEFSComponents
    score_error_bound: float | None
    score_bound_method: str = "unavailable"
    score_phi: float = 1.0
    update_phi: float = 1.0
    reported_phi: float = 1.0
    carried_phi: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.fit, OperatorFitResult) or not self.fit.converged:
            raise ValueError("Operator EFS requires a converged coefficient fit.")
        if not isinstance(self.components, MatchedOperatorEFSComponents):
            raise TypeError("Matched operator EFS components are required.")
        if self.components.trace_method != "exact":
            raise ValueError("Operator EFS update requires exact Fisher traces.")
        expected_fingerprint = operator_efs_fit_fingerprint(
            self.fit.model,
            self.fit.family,
            self.fit.coefficients,
            self.fit.log_lambda,
        )
        if self.components.fit_fingerprint != expected_fingerprint:
            raise ValueError("Operator EFS components do not match their fit state.")
        if self.components.penalty.derivative is None:
            raise ValueError("Operator EFS penalty derivative is missing.")
        m = self.fit.model.penalties.n_penalties
        if any(
            values.shape != (m,)
            for values in (
                self.fit.log_lambda,
                self.components.penalty.derivative,
                self.components.fisher.traces,
                self.components.fisher.quadratics,
            )
        ):
            raise ValueError("Operator EFS coordinates and penalties disagree.")
        if self.score_error_bound is not None and (
            not math.isfinite(self.score_error_bound) or self.score_error_bound < 0
        ):
            raise ValueError("Score error bound must be finite and nonnegative.")
        if self.score_error_bound is None and self.score_bound_method != "unavailable":
            raise ValueError("Uncalibrated score cannot name a bound method.")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.score_phi,
                self.update_phi,
                self.reported_phi,
                self.carried_phi,
            )
        ):
            raise ValueError("Operator EFS scales must be finite and positive.")
        if self.components.score_rows.terms.score_phi != self.score_phi:
            raise ValueError("Operator EFS score phi does not match row reductions.")
        if self.fit.family.scale_known and any(
            value != 1.0
            for value in (
                self.score_phi,
                self.update_phi,
                self.reported_phi,
                self.carried_phi,
            )
        ):
            raise ValueError("Known-scale operator EFS phi must remain one.")


@dataclass(frozen=True)
class OperatorEFSCalibratedDomain:
    """Model shapes over which an empirical score envelope was measured.

    Admission also needs the regime's family, row, control and conditioning
    checks. The measured distributions behind these limits are summarized in
    ``docs/scale_jaxgam/efs6_operator_calibration.md``.
    """

    minimum_levels: int
    maximum_levels: int
    minimum_rows_per_level: int
    maximum_rows_per_level: int
    minimum_batch_rows: int
    maximum_batch_rows: int

    def admits(self, levels: int, n_obs: int, batch_rows: int) -> bool:
        return (
            self.minimum_levels <= levels <= self.maximum_levels
            and self.minimum_rows_per_level * levels
            <= n_obs
            <= self.maximum_rows_per_level * levels
            and self.minimum_batch_rows <= batch_rows <= self.maximum_batch_rows
        )

    def describe(self) -> str:
        return (
            f"{self.minimum_levels}-{self.maximum_levels} indexed levels, "
            f"{self.minimum_rows_per_level}-{self.maximum_rows_per_level} rows per "
            f"level and batch_rows {self.minimum_batch_rows}-"
            f"{self.maximum_batch_rows}"
        )


# One measured domain is shared by the indexed Poisson, indexed Gaussian and
# modest-plus-indexed Poisson envelopes (calibration v2).
CALIBRATED_OPERATOR_EFS_DOMAIN = OperatorEFSCalibratedDomain(
    minimum_levels=257,
    maximum_levels=4096,
    minimum_rows_per_level=1,
    maximum_rows_per_level=16,
    minimum_batch_rows=64,
    maximum_batch_rows=65536,
)
CALIBRATED_OPERATOR_EFS_MAX_MODEST = 4


# Row deviance/saturated terms, penalty quadratics and indexed diagonal sums
# each have one sign, so batched recursive summation gives a first-order
# rounding bound of about (n + p) eps M. Calibration v2 measured at most
# 6.7e-4 (n + p) eps M over every calibration state; v1 used 512.
_ARITHMETIC_ROUNDING_FACTOR = 32


@dataclass(frozen=True)
class OperatorScoreEnvelope:
    """Separately reported terms of one empirical operator EFS score envelope.

    ``arithmetic`` covers float64 row, diagonal and Schur reductions,
    ``coefficient`` the final coefficient residual and ``determinant`` the
    structured log-determinant rounding. The envelope is only a validated
    bound when its regime's admission checks accept the state.
    """

    arithmetic: float
    coefficient: float
    determinant: float = 0.0

    @property
    def total(self) -> float:
        return self.arithmetic + self.coefficient + self.determinant


def _score_magnitude(
    components: MatchedOperatorEFSComponents, deviance_scale: float = 1.0
) -> float:
    terms = components.score_rows.terms
    return math.fsum(
        (
            1.0,
            abs(terms.penalized_deviance / deviance_scale),
            abs(terms.saturated_loglik),
            abs(components.penalty.value),
            abs(components.score.value),
        )
    )


def _identity_penalty_floor(fit: OperatorFitResult) -> float | None:
    """Strong-convexity floor of one scaled identity penalty, if that is all."""
    blocks = fit.model.penalties.blocks
    if len(blocks) != 1:
        return None
    block = blocks[0]
    if (
        block.sp_indices != (0,)
        or len(block.local_penalties) != 1
        or not isinstance(block.local_penalties[0], IdentityPenalty)
    ):
        return None
    floor = math.exp(float(fit.log_lambda[0])) * block.local_penalties[0].scale
    return floor if math.isfinite(floor) and floor > 0 else None


def _indexed_poisson_envelope(
    fit: OperatorFitResult, components: MatchedOperatorEFSComponents
) -> OperatorScoreEnvelope | None:
    """Unadmitted pure-indexed Poisson envelope terms for one matched state."""
    score = components.score
    gradient = fit.diagnostics.final_gradient_norm
    floor = _identity_penalty_floor(fit)
    if (
        score.strategy != "structured_diagonal"
        or score.determinant_rounding_bound is None
        or floor is None
        or not math.isfinite(gradient)
    ):
        return None
    model = fit.model
    arithmetic = (
        _ARITHMETIC_ROUNDING_FACTOR
        * np.finfo(np.float64).eps
        * (model.n_obs + model.n_coef)
        * _score_magnitude(components)
    )
    coefficient = model.n_coef * (gradient + gradient * gradient) / floor
    return OperatorScoreEnvelope(
        float(arithmetic), coefficient, score.determinant_rounding_bound
    )


def _indexed_gaussian_envelope(
    fit: OperatorFitResult, components: MatchedOperatorEFSComponents
) -> OperatorScoreEnvelope | None:
    """Unadmitted pure-indexed Gaussian envelope terms at incoming phi.

    For Gaussian identity log|H| does not depend on beta, and H >= floor I
    everywhere, so a coefficient residual changes the score by at most
    p g^2 / (8 floor phi); no first-order term is needed.
    """
    score = components.score
    phi = components.score_rows.terms.score_phi
    gradient = fit.diagnostics.final_gradient_norm
    floor = _identity_penalty_floor(fit)
    if (
        score.strategy != "structured_diagonal"
        or score.determinant_rounding_bound is None
        or floor is None
        or not math.isfinite(gradient)
    ):
        return None
    model = fit.model
    arithmetic = (
        _ARITHMETIC_ROUNDING_FACTOR
        * np.finfo(np.float64).eps
        * (model.n_obs + model.n_coef)
        * _score_magnitude(components, phi)
    )
    coefficient = model.n_coef * gradient * gradient / (floor * phi)
    return OperatorScoreEnvelope(
        float(arithmetic), coefficient, score.determinant_rounding_bound
    )


def _arrowhead_poisson_envelope(
    fit: OperatorFitResult, components: MatchedOperatorEFSComponents
) -> OperatorScoreEnvelope | None:
    """Unadmitted modest-plus-indexed Poisson envelope terms.

    For a canonical family d log|H| / d beta_j = sum_i a_i x_ij, where the
    weighted leverages a_i lie in [0, 1] and sum to the EDF. Its l1 norm is
    therefore at most EDF times the largest row l1 norm. With
    ||beta - beta_hat||_2 <= sqrt(p) g / (2 sigma), the first-order
    determinant change is a quarter of the first coefficient term below.
    The penalized deviance excess is at most p g^2 / (8 sigma). Both rely
    on the empirical spectral estimate sigma, not a certified eigenvalue.
    """
    score = components.score
    gradient = fit.diagnostics.final_gradient_norm
    lower = score.spectral_lower_estimate
    row_l1 = components.score_rows.maximum_row_l1_norm
    if (
        score.strategy != "structured_arrowhead"
        or lower is None
        or not math.isfinite(gradient)
        or not math.isfinite(row_l1)
        or not math.isfinite(components.edf_total)
        or components.edf_total <= 0
    ):
        return None
    model = fit.model
    p = model.n_coef
    k = len(model.modest_columns)
    arithmetic = (
        _ARITHMETIC_ROUNDING_FACTOR
        * np.finfo(np.float64).eps
        * (model.n_obs + p + k * k)
        * _score_magnitude(components)
    )
    first_order = components.edf_total * row_l1 * math.sqrt(p) * gradient
    coefficient = (first_order + p * gradient * gradient) / lower
    return OperatorScoreEnvelope(float(arithmetic), coefficient)


def operator_efs_score_envelope(
    fit: OperatorFitResult, components: MatchedOperatorEFSComponents
) -> OperatorScoreEnvelope | None:
    """Return the regime formula's terms without applying its admission.

    Calibration uses this to measure the formula outside its released domain.
    Production decisions must use the admitted bound functions instead.
    """
    if components.score.strategy == "structured_arrowhead":
        return _arrowhead_poisson_envelope(fit, components)
    if fit.family.scale_known:
        return _indexed_poisson_envelope(fit, components)
    return _indexed_gaussian_envelope(fit, components)


def _admitted_total(envelope: OperatorScoreEnvelope | None) -> float | None:
    if envelope is None:
        return None
    bound = envelope.total
    return float(bound) if math.isfinite(bound) else None


def _indexed_poisson_score_bound(
    fit: OperatorFitResult,
    components: MatchedOperatorEFSComponents,
    control: OperatorControl,
) -> float | None:
    """Empirical score envelope for bounded, well-conditioned pure indexed fits.

    The arithmetic allowance is a conservative float64 row/diagonal envelope
    calibrated against independent dense matched states and full EFS paths.
    The residual allowance uses the observed final gradient and the identity
    penalty's strong-convexity floor. This is not a general stochastic bound.
    """
    model = fit.model
    score = components.score
    if not (
        CALIBRATED_OPERATOR_EFS_DOMAIN.admits(
            model.n_coef, model.n_obs, model.batch_rows
        )
        and isinstance(fit.family, Poisson)
        and fit.family.is_canonical
        and not fit.offset_was_nonzero
        and score.strategy == "structured_diagonal"
        and components.fisher.diagnostics.strategy == "structured_diagonal"
        and fit.diagnostics.inner_status == CONVERGED
        and control.tolerance <= 1e-7
        and control.cg_rtol <= 1e-10
        and control.cg_atol <= 1e-12
        and len(model.penalties.blocks) == 1
        and components.score_rows.unit_weights
        and components.score_rows.zero_offset
        and components.score_rows.maximum_response <= 32
    ):
        return None
    block = model.penalties.blocks[0]
    if (
        block.sp_indices != (0,)
        or len(block.local_penalties) != 1
        or not isinstance(block.local_penalties[0], IdentityPenalty)
        or score.diagonal_condition_ratio is None
        or score.diagonal_condition_ratio > 1e4
        or score.minimum_diagonal is None
        or score.minimum_diagonal < 1e-6
        or score.determinant_rounding_bound is None
        or np.max(np.abs(fit.coefficients)) > 20
    ):
        return None
    gradient = fit.diagnostics.final_gradient_norm
    if not math.isfinite(gradient) or gradient > 1e-5:
        return None
    penalty_floor = _identity_penalty_floor(fit)
    if penalty_floor is None or penalty_floor < 1e-6:
        return None
    return _admitted_total(_indexed_poisson_envelope(fit, components))


def _indexed_gaussian_score_bound(
    fit: OperatorFitResult,
    components: MatchedOperatorEFSComponents,
    control: OperatorControl,
) -> float | None:
    """Conditional exact-diagonal Gaussian score envelope at incoming phi.

    This deliberately admits only the measured single-index, bounded-row
    regime. A separately measured floating arithmetic allowance covers the
    likelihood, penalty and diagonal reductions. A residual allowance covers
    coefficient error under the positive identity-penalty floor.
    """
    model = fit.model
    score = components.score
    rows = components.score_rows
    phi = rows.terms.score_phi
    if not (
        CALIBRATED_OPERATOR_EFS_DOMAIN.admits(
            model.n_coef, model.n_obs, model.batch_rows
        )
        and isinstance(fit.family, Gaussian)
        and isinstance(fit.family.link, IdentityLink)
        and rows.unit_weights
        and rows.zero_offset
        and rows.maximum_absolute_response <= 10
        and 1e-8 <= phi <= 1e4
        and score.strategy == "structured_diagonal"
        and components.fisher.diagnostics.strategy == "structured_diagonal"
        and fit.diagnostics.inner_status == CONVERGED
        and control.tolerance <= 1e-7
        and control.cg_rtol <= 1e-10
        and control.cg_atol <= 1e-12
        and len(model.penalties.blocks) == 1
    ):
        return None
    block = model.penalties.blocks[0]
    if (
        block.sp_indices != (0,)
        or len(block.local_penalties) != 1
        or not isinstance(block.local_penalties[0], IdentityPenalty)
        or score.diagonal_condition_ratio is None
        or score.diagonal_condition_ratio > 1e4
        or score.minimum_diagonal is None
        or score.minimum_diagonal < 1e-6
        or score.determinant_rounding_bound is None
        or np.max(np.abs(fit.coefficients)) > 20
    ):
        return None
    gradient = fit.diagnostics.final_gradient_norm
    if not math.isfinite(gradient) or gradient > 1e-5:
        return None
    penalty_floor = _identity_penalty_floor(fit)
    if penalty_floor is None or penalty_floor < 1e-6:
        return None
    return _admitted_total(_indexed_gaussian_envelope(fit, components))


def _arrowhead_poisson_score_bound(
    fit: OperatorFitResult,
    components: MatchedOperatorEFSComponents,
    control: OperatorControl,
) -> float | None:
    """Empirical score envelope for a conditioned modest-indexed Poisson fit.

    The structured determinant and Fisher trace have no probe or Krylov error.
    The arithmetic allowance covers finite row/Schur/reduction effects in the
    measured bounded regime. The local gradient allowance uses the Schur
    empirically buffered spectral estimate and is conditional on the fitted
    convex neighborhood. This is not a deterministic eigenvalue certificate.
    Neither allowance admits a general coupled or stochastic operator model.
    """
    model = fit.model
    score = components.score
    rows = components.score_rows
    k = len(model.modest_columns)
    indexed = model.random_effects[0].n_coef if model.random_effects else 0
    if not (
        CALIBRATED_OPERATOR_EFS_DOMAIN.admits(indexed, model.n_obs, model.batch_rows)
        and 1 <= k <= CALIBRATED_OPERATOR_EFS_MAX_MODEST
        and len(model.random_effects) == 1
        and len(model.penalties.blocks) == 1
        and isinstance(fit.family, Poisson)
        and fit.family.is_canonical
        and not fit.offset_was_nonzero
        and rows.unit_weights
        and rows.zero_offset
        and rows.maximum_response <= 32
        and score.strategy == "structured_arrowhead"
        and components.fisher.diagnostics.strategy == "structured_arrowhead"
        and fit.diagnostics.inner_status == CONVERGED
        and control.tolerance <= 1e-7
        and control.cg_rtol <= 1e-10
        and control.cg_atol <= 1e-12
    ):
        return None
    block = model.penalties.blocks[0]
    effect = model.random_effects[0]
    if (
        block.start != effect.col_start
        or block.size != effect.n_coef
        or block.sp_indices != (0,)
        or len(block.local_penalties) != 1
        or not isinstance(block.local_penalties[0], IdentityPenalty)
        or score.spectral_lower_estimate is None
        or score.spectral_lower_estimate < 1e-3
        or score.spectral_condition_estimate is None
        or score.spectral_condition_estimate > 1e5
        or score.minimum_diagonal is None
        or score.minimum_diagonal < 1e-6
        or np.max(np.abs(fit.coefficients)) > 20
    ):
        return None
    gradient = fit.diagnostics.final_gradient_norm
    if not math.isfinite(gradient) or gradient > 1e-5:
        return None
    return _admitted_total(_arrowhead_poisson_envelope(fit, components))


def admitted_operator_efs_score_bound(
    fit: OperatorFitResult,
    components: MatchedOperatorEFSComponents,
    control: OperatorControl,
) -> tuple[float | None, str]:
    """Return the released bound and its method label, or ``unavailable``."""
    if fit.family.scale_known:
        bound = _indexed_poisson_score_bound(fit, components, control)
        if bound is not None:
            return bound, "indexed_poisson_empirical_v2"
        bound = _arrowhead_poisson_score_bound(fit, components, control)
        if bound is not None:
            return bound, "arrowhead_poisson_empirical_v2"
        return None, "unavailable"
    bound = _indexed_gaussian_score_bound(fit, components, control)
    if bound is not None:
        return bound, "indexed_gaussian_empirical_v2"
    return None, "unavailable"


class OperatorEFSFitProvider(Protocol):
    """Refit and matched evaluation supplied by an operator backend."""

    def fit(self, rho: np.ndarray, beta_start: np.ndarray | None) -> OperatorFitResult:
        """Return a converged coefficient fit at rho."""

    def evaluate(
        self,
        fit: OperatorFitResult,
        control: OperatorLogDetControl,
        *,
        score_phi: float = 1.0,
    ) -> OperatorEFSState:
        """Return matched statistics and a calibrated absolute score bound."""

    def reserve(
        self,
        held: tuple[OperatorEFSState, ...],
        *,
        operation: str,
        pending_fit: OperatorFitResult | None = None,
        controller_bytes: int = 0,
    ) -> None:
        """Reject work beyond shared state and controller ownership."""


class SourceOperatorEFSFitProvider:
    """Compose bounded fixed-sp refits with frozen known-scale components.

    The optional calibrator must independently bound total score error,
    including probe sampling, Lanczos truncation and solve effects. Until a
    validated calibrator exists, comparisons stay explicitly indeterminate.
    """

    def __init__(
        self,
        model: PreparedOperatorModel,
        source: RowSource,
        family: ExponentialFamily,
        *,
        fit_control: OperatorControl,
        fit_budget_bytes: int,
        components_budget_bytes: int,
        aggregate_budget_bytes: int,
        max_y: int = 0,
        score_error_bound: Callable[
            [OperatorFitResult, MatchedOperatorEFSComponents], float | None
        ]
        | None = None,
    ) -> None:
        if not family.scale_known and not (
            isinstance(family, Gaussian) and isinstance(family.link, IdentityLink)
        ):
            raise NotImplementedError(
                "Unknown-scale operator EFS currently supports Gaussian identity."
            )
        if isinstance(family, NegativeBinomial) and family.n_theta > 0:
            raise NotImplementedError(
                "Estimated-theta operator EFS requires in-PIRLS theta updates."
            )
        if model.penalties.n_penalties == 0:
            raise ValueError("Operator EFS requires at least one penalty.")
        if (
            fit_budget_bytes < 1
            or components_budget_bytes < 1
            or aggregate_budget_bytes < max(fit_budget_bytes, components_budget_bytes)
        ):
            raise ValueError("Operator EFS aggregate budget is insufficient.")
        if (
            isinstance(max_y, bool)
            or not isinstance(max_y, numbers.Integral)
            or not 0 <= max_y <= 32
        ):
            raise ValueError("Operator EFS max_y must be an integer in [0, 32].")
        if isinstance(family, NegativeBinomial) and max_y == 0:
            raise ValueError("Fixed-theta NB operator EFS requires a positive max_y.")
        self.model = model
        self.source = source
        self.family = family
        self.fit_control = fit_control
        self.fit_budget_bytes = fit_budget_bytes
        self.components_budget_bytes = components_budget_bytes
        self.aggregate_budget_bytes = aggregate_budget_bytes
        self.max_y = int(max_y)
        self.aggregate_peak_bound_bytes = 0
        self.completed_source_scans = 0
        self.completed_source_batches = 0
        self.score_error_bound = score_error_bound

    @staticmethod
    def _fit_retained_bound(fit: OperatorFitResult) -> int:
        """Coefficient/family snapshot ownership, excluding shared model metadata."""
        from jaxgam.execution.operator_fit import _family_ownership_bound

        return (
            2 * fit.coefficients.nbytes
            + 2 * fit.log_lambda.nbytes
            + _family_ownership_bound(fit.family)
            + 8192
        )

    @classmethod
    def _state_retained_bound(cls, state: OperatorEFSState) -> int:
        components = state.components
        arrays = [
            components.score.per_probe,
            components.penalty.block_values,
            components.penalty.derivative,
            components.fisher.traces,
            components.fisher.quadratics,
        ]
        per_probe = getattr(components.fisher, "per_probe", None)
        if per_probe is not None:
            arrays.append(per_probe)
            arrays.append(components.fisher.monte_carlo_standard_error)
        arrays.append(getattr(components.fisher, "trace_rounding_bound", None))
        arrays.append(getattr(components.fisher, "quadratic_rounding_bound", None))
        return (
            cls._fit_retained_bound(state.fit)
            + sum(2 * array.nbytes for array in arrays if array is not None)
            + 8192
        )

    def reserve(
        self,
        held: tuple[OperatorEFSState, ...],
        *,
        operation: str,
        pending_fit: OperatorFitResult | None = None,
        controller_bytes: int = 0,
    ) -> None:
        """Bound shared accepted/trial storage plus the next complete workspace."""
        if controller_bytes < 0:
            raise ValueError("Controller ownership must be nonnegative.")
        if operation == "fit" and pending_fit is None:
            active = self.fit_budget_bytes
        elif operation == "evaluate" and pending_fit is not None:
            active = self.components_budget_bytes + self._fit_retained_bound(
                pending_fit
            )
        else:
            raise ValueError("Invalid operator EFS reservation request.")
        unique = {id(state): state for state in held}
        required = (
            active
            + controller_bytes
            + sum(self._state_retained_bound(state) for state in unique.values())
        )
        if required > self.aggregate_budget_bytes:
            raise MemoryError(
                f"Operator EFS requires {required} combined bytes; "
                f"aggregate budget is {self.aggregate_budget_bytes}."
            )
        self.aggregate_peak_bound_bytes = max(self.aggregate_peak_bound_bytes, required)

    def fit(self, rho: np.ndarray, beta_start: np.ndarray | None) -> OperatorFitResult:
        with np.errstate(over="ignore", invalid="ignore"):
            sp = np.exp(rho)
        fitted = fit_operator_fixed_sp(
            self.model,
            self.source,
            self.family,
            np.asarray(sp, dtype=np.float64),
            control=self.fit_control,
            memory_budget_bytes=self.fit_budget_bytes,
            beta_start=beta_start,
        )
        self.completed_source_scans += fitted.diagnostics.source.completed_scans
        self.completed_source_batches += fitted.diagnostics.source.yielded_batches
        return fitted

    def evaluate(
        self,
        fit: OperatorFitResult,
        control: OperatorLogDetControl,
        *,
        score_phi: float = 1.0,
    ) -> OperatorEFSState:
        if not math.isfinite(score_phi) or score_phi <= 0:
            raise ValueError("Operator EFS incoming score phi must be finite positive.")
        components = MatchedOperatorEFSProvider.from_fit(
            fit,
            self.source,
            score_phi=score_phi,
            memory_budget_bytes=self.components_budget_bytes,
            logdet_control=control,
            max_y=self.max_y,
        ).evaluate()
        self.completed_source_scans += components.source_scans
        self.completed_source_batches += components.source_batches
        if self.score_error_bound is None:
            bound, method = admitted_operator_efs_score_bound(
                fit, components, self.fit_control
            )
        else:
            bound = self.score_error_bound(fit, components)
            method = "external" if bound is not None else "unavailable"
        update_phi = 1.0
        if not fit.family.scale_known:
            denominator = fit.model.n_obs - components.edf_total
            if not math.isfinite(denominator) or denominator <= 0:
                raise FloatingPointError(
                    "Operator EFS scale degrees of freedom invalid."
                )
            update_phi = fit.deviance / denominator
            if not math.isfinite(update_phi) or update_phi <= 0:
                raise FloatingPointError("Operator EFS Fletcher scale is invalid.")
        return OperatorEFSState(
            fit,
            components,
            bound,
            method,
            score_phi,
            update_phi,
            update_phi,
            update_phi,
        )


@dataclass(frozen=True)
class OperatorEFSControllerResult:
    """Accepted state and truthful controller decision/accuracy diagnostics."""

    accepted: OperatorEFSState
    converged: bool
    stop_reason: str
    outer_iterations: int
    attempted_fits: int
    accuracy_refinements: int
    indeterminate_comparisons: int
    multiplier: float
    score_history: tuple[float, ...]
    score_bound_history: tuple[float | None, ...]
    score_phi_history: tuple[float, ...] = ()
    update_phi_history: tuple[float, ...] = ()


def _raw_ratio(state: OperatorEFSState) -> np.ndarray:
    """Pinned efsudr d/lambda minus Fisher t, with its clamp ordering."""
    rho = state.fit.log_lambda
    d = state.components.penalty.derivative
    t = state.components.fisher.traces
    q = state.components.fisher.quadratics
    assert d is not None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        numerator = np.maximum(0.0, np.exp(-rho) * d - t)
        ratio = numerator / np.maximum(0.0, q) * state.update_phi
        ratio = np.where((numerator == 0) & (q == 0), 1.0, ratio)
        ratio = np.where(np.isfinite(ratio), ratio, 1e6)
    if not np.all(np.isfinite(ratio) & (ratio > 0)):
        raise FloatingPointError("Operator EFS raw update is not finite positive.")
    return ratio


def _trial_rho(
    old: OperatorEFSState, ratio: np.ndarray, multiplier: float, cap: float
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        rho = np.minimum(old.fit.log_lambda + np.log(ratio) * multiplier, cap)
    if not np.all(np.isfinite(rho)):
        raise FloatingPointError("Operator EFS trial rho is nonfinite.")
    return np.asarray(rho, dtype=np.float64)


def _validate_fit_rho(fit: OperatorFitResult, rho: np.ndarray) -> None:
    # The source provider accepts positive sp and returns log(sp), so the
    # exp/log boundary can change the last few bits of a trial rho.
    rounding = 8 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(rho))
    if fit.log_lambda.shape != rho.shape or not np.all(
        np.abs(fit.log_lambda - rho) <= rounding
    ):
        raise ValueError("Operator EFS provider fit has mismatched rho.")


def _score_window_certified(
    scores: tuple[float, ...], bounds: tuple[float | None, ...], tolerance: float
) -> bool:
    """Require every adjacent score change to remain small within its bounds."""
    if len(scores) != 4 or len(bounds) != 4 or any(bound is None for bound in bounds):
        return False
    return (
        max(
            abs(scores[index + 1] - scores[index])
            + float(bounds[index])
            + float(bounds[index + 1])
            for index in range(3)
        )
        < tolerance
    )


def _run_operator_efs(
    provider: OperatorEFSFitProvider,
    initial_log_lambda: np.ndarray,
    *,
    control: EFSControl,
    initial_accuracy: OperatorLogDetControl,
    maximum_probes: int = 256,
    maximum_depth: int = 128,
    estimate_scale: bool,
    initial_score_phi: float | None,
) -> OperatorEFSControllerResult:
    """Run shared EFS policy, retaining accepted state on uncertain comparisons.

    The provider must supply a *calibrated absolute* score error bound at each
    requested accuracy; ``None`` means unavailable. Refinement reuses the same
    seed and fit, and grows probes/depth only within explicit caps. The caller
    owns validation of the bound's numerical/probability meaning.
    """
    if not isinstance(control, EFSControl) or not isinstance(
        initial_accuracy, OperatorLogDetControl
    ):
        raise TypeError("Validated EFS and logdet controls are required.")
    if (
        initial_log_lambda.ndim != 1
        or initial_log_lambda.size == 0
        or initial_log_lambda.dtype != np.float64
        or not np.all(np.isfinite(initial_log_lambda))
    ):
        raise ValueError("Initial EFS rho must be a finite float64 vector.")
    if initial_score_phi is not None and (
        not math.isfinite(initial_score_phi) or initial_score_phi <= 0
    ):
        raise ValueError("Initial EFS score phi must be finite positive.")
    if not estimate_scale and initial_score_phi not in (None, 1.0):
        raise ValueError("Known-scale EFS initial phi must be one.")
    if (
        not initial_accuracy.probes <= maximum_probes <= 256
        or not initial_accuracy.lanczos_depth <= maximum_depth <= 128
    ):
        raise ValueError("Accuracy caps must contain initial settings.")
    history_slots = min(control.outer_limit, max(4, control.history_limit))
    # Score/bound and score/update-phi deques, returned tuple copies,
    # accepted history metadata, and live rho/ratio/extension vectors.
    controller_bytes = 8192 + 384 * history_slots + 128 * len(initial_log_lambda)

    def evaluate_state(
        fit: OperatorFitResult, accuracy: OperatorLogDetControl, score_phi: float
    ) -> OperatorEFSState:
        if estimate_scale:
            return provider.evaluate(fit, accuracy, score_phi=score_phi)
        return provider.evaluate(fit, accuracy)

    def reserve(
        held: tuple[OperatorEFSState, ...],
        *,
        operation: str,
        pending_fit: OperatorFitResult | None = None,
    ) -> None:
        provider.reserve(
            held,
            operation=operation,
            pending_fit=pending_fit,
            controller_bytes=controller_bytes,
        )

    reserve((), operation="fit")
    shifted = np.asarray(initial_log_lambda + 2.5, dtype=np.float64)
    first_fit = provider.fit(shifted, None)
    _validate_fit_rho(first_fit, shifted)
    del shifted
    if not first_fit.converged:
        raise RuntimeError("Initial operator EFS coefficient fit did not converge.")
    if estimate_scale and initial_score_phi is None:
        null_scale = first_fit.diagnostics.initial_null_scale
        if null_scale is None or not math.isfinite(null_scale) or null_scale <= 0:
            raise ValueError("Initial operator EFS null scale must be finite positive.")
        initial_score_phi = null_scale / 10.0
    reserve((), operation="evaluate", pending_fit=first_fit)
    accepted = evaluate_state(first_fit, initial_accuracy, initial_score_phi or 1.0)
    del first_fit
    fits = 1
    refinements = indeterminate = 0
    multiplier = 1.0
    accuracy = initial_accuracy
    history: deque[float] = deque(maxlen=max(4, control.history_limit))
    bound_history: deque[float | None] = deque(maxlen=max(4, control.history_limit))
    score_phi_history: deque[float] = deque(maxlen=max(4, control.history_limit))
    update_phi_history: deque[float] = deque(maxlen=max(4, control.history_limit))
    old_deviance: float | None = None
    stop = "iteration_limit"

    def compare(
        old: OperatorEFSState,
        trial: OperatorEFSState,
        *,
        ambient: tuple[OperatorEFSState, ...] = (),
    ) -> tuple[OperatorEFSState, OperatorEFSState, OperatorLogDetControl, str, int]:
        current = accuracy
        count = 0
        # Caller bindings remain live until this function returns, even after
        # local refinement replaces old/trial. Charge both generations.
        original_old, original_trial = old, trial
        while True:
            result = compare_operator_scores(
                old.components.score,
                trial.components.score,
                accepted_error_bound=old.score_error_bound,
                trial_error_bound=trial.score_error_bound,
            )
            if result.status != "indeterminate":
                return old, trial, current, result.status, count
            if trial.components.score.strategy != "stochastic_lanczos":
                # Structured scores have no probes or Krylov depth to refine.
                return old, trial, current, "indeterminate", count
            probes = min(maximum_probes, 2 * current.probes)
            depth = min(maximum_depth, 2 * current.lanczos_depth)
            if probes == current.probes and depth == current.lanczos_depth:
                return old, trial, current, "indeterminate", count
            current = replace(current, probes=probes, lanczos_depth=depth)
            reserve(
                (
                    accepted,
                    *ambient,
                    original_old,
                    original_trial,
                    old,
                    trial,
                ),
                operation="evaluate",
                pending_fit=old.fit,
            )
            refined_old = evaluate_state(old.fit, current, old.score_phi)
            if estimate_scale:
                # Accuracy changes only the score/trace evidence. In efsudr a
                # winning extension carries the preceding candidate scale,
                # which differs from its own Fletcher update.
                refined_old = replace(refined_old, carried_phi=old.carried_phi)
            reserve(
                (
                    accepted,
                    *ambient,
                    original_old,
                    original_trial,
                    old,
                    trial,
                    refined_old,
                ),
                operation="evaluate",
                pending_fit=trial.fit,
            )
            old = refined_old
            refined_trial = evaluate_state(trial.fit, current, trial.score_phi)
            trial = (
                replace(refined_trial, carried_phi=trial.carried_phi)
                if estimate_scale
                else refined_trial
            )
            count += 1

    for iteration in range(1, control.outer_limit + 1):
        try:
            ratio = _raw_ratio(accepted)
            rho = _trial_rho(accepted, ratio, multiplier, control.log_lambda_max)
        except FloatingPointError:
            stop = "invalid_update"
            break
        maximum_step = float(np.max(np.abs(rho - accepted.fit.log_lambda)))
        old = accepted
        reserve((old,), operation="fit")
        trial_fit = provider.fit(rho, old.fit.coefficients)
        fits += 1
        _validate_fit_rho(trial_fit, rho)
        if not trial_fit.converged:
            stop = "inner_failure"
            break
        reserve((old,), operation="evaluate", pending_fit=trial_fit)
        candidate = evaluate_state(trial_fit, accuracy, old.carried_phi)
        del trial_fit
        old, candidate, accuracy, decision, count = compare(old, candidate)
        refinements += count
        if decision == "indeterminate":
            indeterminate += 1
            stop = "indeterminate_score"
            break
        if decision in {"improvement", "equal"}:
            if maximum_step < 0.05:
                extended_rho = _trial_rho(
                    old, ratio, multiplier * 2, control.log_lambda_max
                )
                reserve((accepted, old, candidate), operation="fit")
                extension_fit = provider.fit(extended_rho, old.fit.coefficients)
                fits += 1
                _validate_fit_rho(extension_fit, extended_rho)
                if not extension_fit.converged:
                    accepted = candidate
                else:
                    reserve(
                        (accepted, old, candidate),
                        operation="evaluate",
                        pending_fit=extension_fit,
                    )
                    extension = evaluate_state(extension_fit, accuracy, old.carried_phi)
                    candidate, extension, accuracy, extended_decision, count = compare(
                        candidate, extension, ambient=(old,)
                    )
                    refinements += count
                    if extended_decision == "improvement":
                        accepted = (
                            replace(extension, carried_phi=candidate.update_phi)
                            if estimate_scale
                            else extension
                        )
                        multiplier *= 2
                    else:
                        accepted = candidate
                        indeterminate += extended_decision == "indeterminate"
                    del extension
                del extension_fit
            else:
                accepted = candidate
        else:
            while decision == "worsening" and multiplier > 1:
                multiplier /= 2
                rho = _trial_rho(old, ratio, multiplier, control.log_lambda_max)
                reserve((accepted, old, candidate), operation="fit")
                trial_fit = provider.fit(rho, old.fit.coefficients)
                fits += 1
                _validate_fit_rho(trial_fit, rho)
                if not trial_fit.converged:
                    stop = "inner_failure"
                    break
                reserve(
                    (accepted, old, candidate),
                    operation="evaluate",
                    pending_fit=trial_fit,
                )
                candidate = evaluate_state(trial_fit, accuracy, old.carried_phi)
                del trial_fit
                old, candidate, accuracy, decision, count = compare(old, candidate)
                refinements += count
                if decision == "indeterminate":
                    indeterminate += 1
                    if multiplier > 1:
                        stop = "indeterminate_score"
                        break
                    # efsudr leaves its contraction loop once mult <= 1
                    # whatever the score, so this undecided comparison
                    # cannot change the accepted minimum-multiplier step.
            if stop in {"indeterminate_score", "inner_failure"}:
                break
            # Pinned efsudr allows a worsening minimum-multiplier step.
            accepted = candidate
            multiplier = max(multiplier, 1.0)
        del candidate, old
        history.append(accepted.components.score.value)
        bound_history.append(accepted.score_error_bound)
        score_phi_history.append(accepted.score_phi)
        update_phi_history.append(accepted.update_phi)
        window_scores = tuple(history)[-4:]
        window_bounds = tuple(bound_history)[-4:]
        score_window_certified = _score_window_certified(
            window_scores, window_bounds, control.score_tolerance
        )
        if iteration > 3 and maximum_step < 0.05 and score_window_certified:
            stop = "score_window"
            break
        deviance = accepted.fit.deviance
        if old_deviance is not None and abs(old_deviance - deviance) < (
            100 * control.pirls_tolerance * abs(deviance)
        ):
            stop = "deviance_change"
            break
        old_deviance = deviance
    else:
        iteration = control.outer_limit
    if iteration == control.outer_limit and stop in {"score_window", "deviance_change"}:
        stop = "iteration_limit"
    return OperatorEFSControllerResult(
        accepted,
        stop in {"score_window", "deviance_change"},
        stop,
        iteration,
        fits,
        refinements,
        indeterminate,
        multiplier,
        tuple(history)[-control.history_limit :],
        tuple(bound_history)[-control.history_limit :],
        tuple(score_phi_history)[-control.history_limit :],
        tuple(update_phi_history)[-control.history_limit :],
    )


def run_operator_efs_known_scale(
    provider: OperatorEFSFitProvider,
    initial_log_lambda: np.ndarray,
    *,
    control: EFSControl,
    initial_accuracy: OperatorLogDetControl,
    maximum_probes: int = 256,
    maximum_depth: int = 128,
) -> OperatorEFSControllerResult:
    """Run calibrated known-scale EFS through the shared controller."""
    return _run_operator_efs(
        provider,
        initial_log_lambda,
        control=control,
        initial_accuracy=initial_accuracy,
        maximum_probes=maximum_probes,
        maximum_depth=maximum_depth,
        estimate_scale=False,
        initial_score_phi=None,
    )


def run_operator_efs_gaussian_scale(
    provider: SourceOperatorEFSFitProvider,
    initial_log_lambda: np.ndarray,
    *,
    control: EFSControl,
    initial_accuracy: OperatorLogDetControl,
    initial_score_phi: float | None = None,
    maximum_probes: int = 256,
    maximum_depth: int = 128,
) -> OperatorEFSControllerResult:
    """Carry source Gaussian identity Fletcher scale through EFS trials."""
    if not isinstance(provider.family, Gaussian) or not isinstance(
        provider.family.link, IdentityLink
    ):
        raise TypeError("Gaussian scale EFS needs a Gaussian identity provider.")
    return _run_operator_efs(
        provider,
        initial_log_lambda,
        control=control,
        initial_accuracy=initial_accuracy,
        maximum_probes=maximum_probes,
        maximum_depth=maximum_depth,
        estimate_scale=True,
        initial_score_phi=initial_score_phi,
    )
