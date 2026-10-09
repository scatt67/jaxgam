"""Matched-state operator score assembly and conservative trial comparisons.

This component accepts source/family reductions from a fit provider. It cannot
certify their provenance or supply missing scale, theta, EDF, or row reductions.
Its uncertainty accounting does not turn Monte Carlo error into a deterministic
bound; an external calibration must supply one before a trial is decidable.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Literal

import numpy as np

from jaxgam.execution.operator_logdet import OperatorLogDetEstimate
from jaxgam.execution.operator_penalty_determinant import OperatorPenaltyLogDet
from jaxgam.fitting.efs_operator_score import operator_reml_score


@dataclass(frozen=True)
class OperatorScoreTerms:
    """Exact scalar reductions from one matched coefficient/theta/rho state."""

    penalized_deviance: float
    saturated_loglik: float
    score_phi: float
    n_coef: int

    def __post_init__(self) -> None:
        for name in ("penalized_deviance", "saturated_loglik", "score_phi"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and real.")
        if self.score_phi <= 0:
            raise ValueError("score_phi must be positive.")
        if (
            isinstance(self.n_coef, bool)
            or not isinstance(self.n_coef, numbers.Integral)
            or self.n_coef < 1
        ):
            raise ValueError("n_coef must be a positive integer.")


@dataclass(frozen=True)
class OperatorScoreEstimate:
    """Approximate REML score with distinct sampling and depth diagnostics."""

    value: float
    per_probe: np.ndarray
    monte_carlo_standard_error: float
    maximum_depth_difference: float
    source_scans: int
    seed: int
    lanczos_depth: int

    def __post_init__(self) -> None:
        values = np.array(self.per_probe, dtype=np.float64, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "per_probe", values)


def assemble_operator_score(
    terms: OperatorScoreTerms,
    observed: OperatorLogDetEstimate,
    penalty: OperatorPenaltyLogDet,
) -> OperatorScoreEstimate:
    """Combine exact matched terms with an SPD observed logdet estimate."""
    if not isinstance(terms, OperatorScoreTerms):
        raise TypeError("terms must be OperatorScoreTerms.")
    if not isinstance(observed, OperatorLogDetEstimate):
        raise TypeError("observed must be OperatorLogDetEstimate.")
    if not isinstance(penalty, OperatorPenaltyLogDet):
        raise TypeError("penalty must be OperatorPenaltyLogDet.")
    if penalty.rank + penalty.null_dimension != terms.n_coef:
        raise ValueError("Penalty range and score coordinates disagree.")
    if (
        not math.isfinite(observed.value)
        or not math.isfinite(penalty.value)
        or observed.per_probe.ndim != 1
        or len(observed.per_probe) != observed.diagnostics.probes
        or not np.all(np.isfinite(observed.per_probe))
    ):
        raise ValueError("Score determinant components must be finite.")
    value = float(
        operator_reml_score(
            terms.penalized_deviance,
            terms.saturated_loglik,
            terms.score_phi,
            observed.value,
            penalty.value,
            penalty.null_dimension,
        )
    )
    per_probe = np.asarray(observed.per_probe, dtype=np.float64) / 2.0 + (
        value - observed.value / 2.0
    )
    if not math.isfinite(value) or not np.all(np.isfinite(per_probe)):
        raise FloatingPointError("Matched operator score is nonfinite.")
    return OperatorScoreEstimate(
        value=value,
        per_probe=per_probe,
        monte_carlo_standard_error=(
            observed.diagnostics.monte_carlo_standard_error / 2.0
        ),
        maximum_depth_difference=(observed.diagnostics.maximum_depth_difference / 2.0),
        source_scans=observed.diagnostics.source_scans,
        seed=observed.diagnostics.seed,
        lanczos_depth=observed.diagnostics.lanczos_depth,
    )


@dataclass(frozen=True)
class OperatorScoreComparison:
    """Trial-minus-accepted estimate and explicitly limited decision status."""

    difference: float
    paired_standard_error: float
    status: Literal["improvement", "worsening", "indeterminate"]


def compare_operator_scores(
    accepted: OperatorScoreEstimate,
    trial: OperatorScoreEstimate,
    *,
    accepted_error_bound: float | None = None,
    trial_error_bound: float | None = None,
) -> OperatorScoreComparison:
    """Use common probes; decide only with external absolute error bounds.

    Error bounds must include both Monte Carlo and Lanczos/solve effects. The
    paired standard error is diagnostic and never substitutes for those bounds.
    """
    if not isinstance(accepted, OperatorScoreEstimate) or not isinstance(
        trial, OperatorScoreEstimate
    ):
        raise TypeError("Both scores must be operator score estimates.")
    if (
        accepted.seed != trial.seed
        or accepted.lanczos_depth != trial.lanczos_depth
        or accepted.per_probe.shape != trial.per_probe.shape
    ):
        raise ValueError("Paired score comparisons require identical probes.")
    paired = trial.per_probe - accepted.per_probe
    difference = trial.value - accepted.value
    standard_error = (
        float(np.std(paired, ddof=1) / math.sqrt(len(paired)))
        if len(paired) > 1
        else math.inf
    )
    for bound in (accepted_error_bound, trial_error_bound):
        if bound is not None and (
            isinstance(bound, bool)
            or not isinstance(bound, numbers.Real)
            or not math.isfinite(bound)
            or bound < 0
        ):
            raise ValueError("Calibrated score bounds must be finite nonnegative.")
    status: Literal["improvement", "worsening", "indeterminate"] = "indeterminate"
    if accepted_error_bound is not None and trial_error_bound is not None:
        radius = accepted_error_bound + trial_error_bound
        if difference + radius < 0:
            status = "improvement"
        elif difference - radius > 0:
            status = "worsening"
    return OperatorScoreComparison(difference, standard_error, status)
