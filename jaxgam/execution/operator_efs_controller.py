"""Internal operator EFS policy with explicit score-calibration boundaries.

The policy follows pinned ``efsudr`` for known scale. It owns accepted/trial
states, never substitutes Monte Carlo standard error for a score bound, and
refines common probes/depth before declaring a comparison indeterminate.
"""

from __future__ import annotations

import math
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
from jaxgam.families.standard import Poisson
from jaxgam.fitting.iterative import CONVERGED
from jaxgam.formula.operator_prepare import PreparedOperatorModel
from jaxgam.penalties.structure import IdentityPenalty


@dataclass(frozen=True)
class OperatorEFSState:
    """One frozen fit and all statistics at its exact rho/beta/theta state."""

    fit: OperatorFitResult
    components: MatchedOperatorEFSComponents
    score_error_bound: float | None
    score_bound_method: str = "unavailable"

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
    p, n = model.n_coef, model.n_obs
    if not (
        256 < p <= 512
        and 2 * p <= n <= 3 * p
        and model.batch_rows == 64
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
    penalty_floor = math.exp(float(fit.log_lambda[0])) * block.local_penalties[0].scale
    if not math.isfinite(penalty_floor) or penalty_floor < 1e-6:
        return None
    terms = components.score_rows.terms
    magnitude = math.fsum(
        (
            1.0,
            abs(terms.penalized_deviance),
            abs(terms.saturated_loglik),
            abs(components.penalty.value),
            abs(score.value),
        )
    )
    arithmetic = 512 * np.finfo(np.float64).eps * (n + p) * magnitude
    solve = p * (gradient + gradient * gradient) / penalty_floor
    bound = arithmetic + solve + score.determinant_rounding_bound
    return float(bound) if math.isfinite(bound) else None


class OperatorEFSFitProvider(Protocol):
    """Refit and matched evaluation supplied by an operator backend."""

    def fit(self, rho: np.ndarray, beta_start: np.ndarray | None) -> OperatorFitResult:
        """Return a converged coefficient fit at rho."""

    def evaluate(
        self, fit: OperatorFitResult, control: OperatorLogDetControl
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
        score_error_bound: Callable[
            [OperatorFitResult, MatchedOperatorEFSComponents], float | None
        ]
        | None = None,
    ) -> None:
        if not family.scale_known:
            raise NotImplementedError("Known-scale operator EFS requires fixed scale.")
        if model.penalties.n_penalties == 0:
            raise ValueError("Operator EFS requires at least one penalty.")
        if (
            fit_budget_bytes < 1
            or components_budget_bytes < 1
            or aggregate_budget_bytes < max(fit_budget_bytes, components_budget_bytes)
        ):
            raise ValueError("Operator EFS aggregate budget is insufficient.")
        self.model = model
        self.source = source
        self.family = family
        self.fit_control = fit_control
        self.fit_budget_bytes = fit_budget_bytes
        self.components_budget_bytes = components_budget_bytes
        self.aggregate_budget_bytes = aggregate_budget_bytes
        self.aggregate_peak_bound_bytes = 0
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
        return fit_operator_fixed_sp(
            self.model,
            self.source,
            self.family,
            np.asarray(sp, dtype=np.float64),
            control=self.fit_control,
            memory_budget_bytes=self.fit_budget_bytes,
            beta_start=beta_start,
        )

    def evaluate(
        self, fit: OperatorFitResult, control: OperatorLogDetControl
    ) -> OperatorEFSState:
        components = MatchedOperatorEFSProvider.from_fit(
            fit,
            self.source,
            score_phi=1.0,
            memory_budget_bytes=self.components_budget_bytes,
            logdet_control=control,
        ).evaluate()
        if self.score_error_bound is None:
            bound = _indexed_poisson_score_bound(fit, components, self.fit_control)
            method = (
                "indexed_poisson_empirical_v1" if bound is not None else "unavailable"
            )
        else:
            bound = self.score_error_bound(fit, components)
            method = "external" if bound is not None else "unavailable"
        return OperatorEFSState(fit, components, bound, method)


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


def _raw_ratio(state: OperatorEFSState) -> np.ndarray:
    """Pinned efsudr d/lambda minus Fisher t, with its clamp ordering."""
    rho = state.fit.log_lambda
    d = state.components.penalty.derivative
    t = state.components.fisher.traces
    q = state.components.fisher.quadratics
    assert d is not None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        numerator = np.maximum(0.0, np.exp(-rho) * d - t)
        ratio = numerator / np.maximum(0.0, q)
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
    if not np.array_equal(fit.log_lambda, rho):
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


def run_operator_efs_known_scale(
    provider: OperatorEFSFitProvider,
    initial_log_lambda: np.ndarray,
    *,
    control: EFSControl,
    initial_accuracy: OperatorLogDetControl,
    maximum_probes: int = 256,
    maximum_depth: int = 128,
) -> OperatorEFSControllerResult:
    """Run known-scale EFS, retaining accepted state on uncertain comparisons.

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
    if (
        not initial_accuracy.probes <= maximum_probes <= 256
        or not initial_accuracy.lanczos_depth <= maximum_depth <= 128
    ):
        raise ValueError("Accuracy caps must contain initial settings.")
    history_slots = min(control.outer_limit, max(4, control.history_limit))
    # Two deques of Python score/bound objects, their returned tuple copies,
    # accepted history metadata, and live rho/ratio/extension vectors.
    controller_bytes = 8192 + 256 * history_slots + 128 * len(initial_log_lambda)

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
    reserve((), operation="evaluate", pending_fit=first_fit)
    accepted = provider.evaluate(first_fit, initial_accuracy)
    del first_fit
    fits = 1
    refinements = indeterminate = 0
    multiplier = 1.0
    accuracy = initial_accuracy
    history: deque[float] = deque(maxlen=max(4, control.history_limit))
    bound_history: deque[float | None] = deque(maxlen=max(4, control.history_limit))
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
            refined_old = provider.evaluate(old.fit, current)
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
            trial = provider.evaluate(trial.fit, current)
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
        candidate = provider.evaluate(trial_fit, accuracy)
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
                    extension = provider.evaluate(extension_fit, accuracy)
                    candidate, extension, accuracy, extended_decision, count = compare(
                        candidate, extension, ambient=(old,)
                    )
                    refinements += count
                    if extended_decision == "improvement":
                        accepted = extension
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
                candidate = provider.evaluate(trial_fit, accuracy)
                del trial_fit
                old, candidate, accuracy, decision, count = compare(old, candidate)
                refinements += count
                if decision == "indeterminate":
                    indeterminate += 1
                    stop = "indeterminate_score"
                    break
            if stop in {"indeterminate_score", "inner_failure"}:
                break
            # Pinned efsudr allows a worsening minimum-multiplier step.
            accepted = candidate
            multiplier = max(multiplier, 1.0)
        del candidate, old
        history.append(accepted.components.score.value)
        bound_history.append(accepted.score_error_bound)
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
    )
