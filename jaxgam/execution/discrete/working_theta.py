"""Host port of pinned ``efam.r::estimate.theta`` for one theta parameter.

Pinned mgcv 1.9-3 ``bgam.fitd`` re-estimates an extended family's theta once
per outer iteration from iteration two, after coefficient step control and
before forming the working model (``bam.r:613-621``)::

    theta <- estimate.theta(theta, family, y, mu, scale=scale1, wt=G$w,
                            tol=1e-7)

``estimate.theta`` is a safeguarded Newton iteration on the conditional
negative log-likelihood ``nll(theta) = sum(dev.resids)/(2 scale) - ls`` at
fixed ``mu``. This module ports it for one log theta with a known scale
(``scale >= 0``, the NB case): every ``nlogl`` evaluation is supplied by the
caller as a complete bounded source scan, so no row is retained here.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# efam.r::estimate.theta controls.
_MAX_NEWTON_ITERATIONS = 100
_MAX_HALVINGS = 25
_MAX_STEP = 4.0
_CURVATURE_FLOOR = 1e-5
_HALVING_TOLERANCE = np.finfo(float).eps ** 0.75


@dataclass(frozen=True)
class ThetaObjective:
    """``nlogl`` at one theta: ``nll`` and, at ``deriv=2``, ``g`` and ``H``."""

    nll: float
    gradient: float
    hessian: float


@dataclass(frozen=True)
class ThetaUpdate:
    """One ``estimate.theta`` call: start, result and its Newton path.

    ``evaluations`` counts ``nlogl`` scans; ``step_failed`` is R's
    ``step.failed`` (it warns and keeps the last accepted theta).
    """

    log_theta_start: float
    log_theta: float
    newton_iterations: int
    evaluations: int
    step_failed: bool
    objective: ThetaObjective


def source_estimate_theta(
    log_theta: float,
    nlogl: Callable[[float, bool], ThetaObjective],
    *,
    tolerance: float = 1e-7,
) -> ThetaUpdate:
    """Port ``estimate.theta(theta, family, y, mu, scale >= 0, wt, tol)``.

    ``nlogl(theta, derivatives)`` returns the conditional objective at
    ``theta`` (and its gradient and Hessian when ``derivatives``). The loop
    is R's: an absolute-value Hessian when it is not positive (with the
    ``1e-5`` relative floor), a step capped at four, step halving while the
    objective rises by more than ``eps^0.75`` relative (at most 25 halvings
    or until ``theta + step == theta``), and convergence when
    ``|g| <= tol (|nll| + 1)``.
    """
    start = float(log_theta)
    theta = start
    evaluations = 1
    current = nlogl(theta, True)
    _require_finite(current, derivatives=True)
    active = abs(current.gradient) > tolerance * (abs(current.nll) + 1.0)
    step_failed = False
    iterations = 0
    while active and iterations < _MAX_NEWTON_ITERATIONS:
        iterations += 1
        # eigen() of the 1x1 Hessian: its value, made positive if needed.
        curvature = current.hessian
        if not curvature > 0.0:
            curvature = abs(curvature)
            floor = curvature * _CURVATURE_FLOOR
            if curvature < floor:
                curvature = floor
        if curvature == 0.0:  # R divides to an infinite step.
            raise FloatingPointError("Non-finite theta Newton step.")
        step = -(current.gradient / curvature)
        if not np.isfinite(step):
            raise FloatingPointError("Non-finite theta Newton step.")
        size = abs(step)
        if size > _MAX_STEP:
            step = step * _MAX_STEP / size
        trial = nlogl(theta + step, True)
        evaluations += 1
        halvings = 0
        while trial.nll - current.nll > _HALVING_TOLERANCE * abs(current.nll):
            step = step / 2.0
            halvings += 1
            if theta == theta + step or halvings > _MAX_HALVINGS:
                step_failed = True
                break
            trial = nlogl(theta + step, False)
            evaluations += 1
        if step_failed:
            break
        theta = theta + step
        if halvings:
            current = nlogl(theta, True)
            evaluations += 1
        else:
            current = trial
        _require_finite(current, derivatives=True)
        active = abs(current.gradient) > tolerance * (abs(current.nll) + 1.0)
    if step_failed:
        warnings.warn("step failure in theta estimation", RuntimeWarning, stacklevel=2)
    return ThetaUpdate(start, theta, iterations, evaluations, step_failed, current)


def _require_finite(objective: ThetaObjective, *, derivatives: bool) -> None:
    values = (objective.nll, objective.gradient, objective.hessian)
    if not np.all(np.isfinite(values if derivatives else values[:1])):
        raise FloatingPointError("Non-finite estimate.theta objective.")
