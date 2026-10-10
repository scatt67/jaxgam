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
)
from jaxgam.execution.operator_fit import OperatorFitResult, fit_operator_fixed_sp
from jaxgam.execution.operator_logdet import OperatorLogDetControl
from jaxgam.execution.operator_score import compare_operator_scores
from jaxgam.families.base import ExponentialFamily
from jaxgam.formula.operator_prepare import PreparedOperatorModel


@dataclass(frozen=True)
class OperatorEFSState:
    """One frozen fit and all statistics at its exact rho/beta/theta state."""

    fit: OperatorFitResult
    components: MatchedOperatorEFSComponents
    score_error_bound: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.fit, OperatorFitResult) or not self.fit.converged:
            raise ValueError("Operator EFS requires a converged coefficient fit.")
        if not isinstance(self.components, MatchedOperatorEFSComponents):
            raise TypeError("Matched operator EFS components are required.")
        if self.components.trace_method != "exact":
            raise ValueError("Operator EFS update requires exact Fisher traces.")
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


class OperatorEFSFitProvider(Protocol):
    """Refit and matched evaluation supplied by an operator backend."""

    def fit(self, rho: np.ndarray, beta_start: np.ndarray | None) -> OperatorFitResult:
        """Return a converged coefficient fit at rho."""

    def evaluate(
        self, fit: OperatorFitResult, control: OperatorLogDetControl
    ) -> OperatorEFSState:
        """Return matched statistics and a calibrated absolute score bound."""


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
        score_error_bound: Callable[
            [OperatorFitResult, MatchedOperatorEFSComponents], float | None
        ]
        | None = None,
    ) -> None:
        if not family.scale_known:
            raise NotImplementedError("Known-scale operator EFS requires fixed scale.")
        self.model = model
        self.source = source
        self.family = family
        self.fit_control = fit_control
        self.fit_budget_bytes = fit_budget_bytes
        self.components_budget_bytes = components_budget_bytes
        self.score_error_bound = score_error_bound

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
        bound = (
            None
            if self.score_error_bound is None
            else self.score_error_bound(fit, components)
        )
        return OperatorEFSState(fit, components, bound)


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
        or initial_log_lambda.dtype != np.float64
        or not np.all(np.isfinite(initial_log_lambda))
    ):
        raise ValueError("Initial EFS rho must be a finite float64 vector.")
    if (
        not initial_accuracy.probes <= maximum_probes <= 256
        or not initial_accuracy.lanczos_depth <= maximum_depth <= 128
    ):
        raise ValueError("Accuracy caps must contain initial settings.")
    shifted = np.asarray(initial_log_lambda + 2.5, dtype=np.float64)
    first_fit = provider.fit(shifted, None)
    if not first_fit.converged:
        raise RuntimeError("Initial operator EFS coefficient fit did not converge.")
    accepted = provider.evaluate(first_fit, initial_accuracy)
    if not np.array_equal(accepted.fit.log_lambda, shifted):
        raise ValueError("Initial provider fit has mismatched rho.")
    fits = 1
    refinements = indeterminate = 0
    multiplier = 1.0
    accuracy = initial_accuracy
    history: deque[float] = deque(maxlen=max(4, control.history_limit))
    old_deviance: float | None = None
    stop = "iteration_limit"

    def compare(
        old: OperatorEFSState, trial: OperatorEFSState
    ) -> tuple[OperatorEFSState, OperatorEFSState, OperatorLogDetControl, str, int]:
        current = accuracy
        count = 0
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
            old = provider.evaluate(old.fit, current)
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
        trial_fit = provider.fit(rho, old.fit.coefficients)
        fits += 1
        if not trial_fit.converged:
            stop = "inner_failure"
            break
        candidate = provider.evaluate(trial_fit, accuracy)
        if not np.array_equal(candidate.fit.log_lambda, rho):
            raise ValueError("Trial provider fit has mismatched rho.")
        old, candidate, accuracy, decision, count = compare(old, candidate)
        refinements += count
        if decision == "indeterminate":
            indeterminate += 1
            stop = "indeterminate_score"
            break
        if decision == "improvement":
            if maximum_step < 0.05:
                extended_rho = _trial_rho(
                    old, ratio, multiplier * 2, control.log_lambda_max
                )
                extension_fit = provider.fit(extended_rho, old.fit.coefficients)
                fits += 1
                if not extension_fit.converged:
                    accepted = candidate
                else:
                    extension = provider.evaluate(extension_fit, accuracy)
                    candidate, extension, accuracy, extended_decision, count = compare(
                        candidate, extension
                    )
                    refinements += count
                    if extended_decision == "improvement":
                        accepted = extension
                        multiplier *= 2
                    else:
                        accepted = candidate
                        indeterminate += extended_decision == "indeterminate"
            else:
                accepted = candidate
        else:
            while decision == "worsening" and multiplier > 1:
                multiplier /= 2
                rho = _trial_rho(old, ratio, multiplier, control.log_lambda_max)
                trial_fit = provider.fit(rho, old.fit.coefficients)
                fits += 1
                if not trial_fit.converged:
                    stop = "inner_failure"
                    break
                candidate = provider.evaluate(trial_fit, accuracy)
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
        history.append(accepted.components.score.value)
        if (
            iteration > 3
            and maximum_step < 0.05
            and max(abs(np.diff(tuple(history)[-4:]))) < control.score_tolerance
        ):
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
    )
