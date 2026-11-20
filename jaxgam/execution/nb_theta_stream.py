"""Replayed global conditional theta Newton inside streamed NB EFS PIRLS."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import numpy as np

from jaxgam.execution.stream import StreamPIRLSControl, _source_batches
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.efs_theta import _require_estimated_nb, _validate_static_inputs
from jaxgam.fitting.family_execution import FamilyExecutionLineage
from jaxgam.fitting.nb_theta_stream import (
    nb_conditional_theta_batch,
    nb_conditional_theta_step,
)
from jaxgam.formula.design_provider import StreamDesign

_batch = partial(jax.jit, static_argnames=("family", "max_y", "integer_counts"))(
    nb_conditional_theta_batch
)
_step = jax.jit(nb_conditional_theta_step)


@dataclass(frozen=True)
class NBThetaStreamControl:
    """The source estimate.theta scalar limits; no public parameter API."""

    tolerance: float = 1e-7
    max_iter: int = 100
    max_step: float = 4.0
    max_halvings: int = 25

    def __post_init__(self) -> None:
        empty = np.empty(0)
        _validate_static_inputs(
            np.zeros(1),
            empty,
            empty,
            empty,
            empty,
            max_y=0,
            integer_counts=True,
            tolerance=self.tolerance,
            max_iter=self.max_iter,
            max_step=self.max_step,
            max_halvings=self.max_halvings,
        )


@dataclass(frozen=True)
class NBStreamThetaResult:
    """Bounded scalar theta trajectory; no batch predictor or row retention."""

    log_theta: tuple[float, ...]
    nll: float
    gradient: float
    hessian: float
    n_iter: int
    status: int
    nll_history: tuple[float, ...]
    theta_history: tuple[float, ...]
    source_scans: int
    batches_scanned: int
    halvings: int
    stabilized_curvatures: int

    @property
    def converged(self) -> bool:
        return self.status == 0


def conditional_theta_stream(
    stream: StreamDesign,
    family: NegativeBinomial,
    lineage: FamilyExecutionLineage,
    beta: np.ndarray,
    log_theta: np.ndarray,
    pirls_control: StreamPIRLSControl,
    *,
    max_y: int,
    integer_counts: bool,
    control: NBThetaStreamControl | None = None,
    device: jax.Device | None = None,
) -> NBStreamThetaResult:
    """Reconverge conditional theta at one fixed coefficient proposal.

    Each objective and derivative evaluation is a counted bounded-row replay.
    Caller preflights live coefficient, batch, count-prefix and history buffers
    before dispatch; this engine never retains X/y/eta or changes the family.
    A failed conditional solve retains its last valid theta with truthful status.
    """
    _require_estimated_nb(family)
    if control is not None and not isinstance(control, NBThetaStreamControl):
        raise TypeError("conditional theta control must be NBThetaStreamControl")
    control = NBThetaStreamControl() if control is None else control
    beta = np.array(beta, dtype=float, copy=True)
    theta = np.array(log_theta, dtype=float, copy=True)
    if beta.shape != (stream.prepared.n_coef,) or not np.all(np.isfinite(beta)):
        raise ValueError("conditional theta requires finite matching coefficients")
    empty = np.empty(0)
    _validate_static_inputs(
        theta, empty, empty, empty, empty, max_y=max_y, integer_counts=integer_counts
    )
    scans = batches = halvings = stabilized = 0

    def evaluate(trial_theta):
        nonlocal scans, batches
        nll = gradient = hessian = 0.0
        admissible = True
        fractional = False
        for batch, y, weight, offset, valid in _source_batches(
            stream, family, lineage, pirls_control.batch_rows
        ):
            if len(y) > min(pirls_control.batch_rows, stream.prepared.n_obs):
                raise ValueError(
                    "conditional theta source exceeds prospective batch cap"
                )
            X = (
                stream.prepared.evaluate_fitting_batch(batch)
                if len(y)
                else np.empty((0, len(beta)))
            )
            if X.shape != (len(y), len(beta)) or not np.all(np.isfinite(X)):
                raise ValueError(
                    "conditional theta requires finite matching batch design"
                )
            lineage.validate(stream.prepared, family)
            value = _batch(
                *[
                    jax.device_put(a, device)
                    for a in (trial_theta, X @ beta + offset, y, weight, valid)
                ],
                family,
                max_y=max_y,
                integer_counts=integer_counts,
            )
            nll += float(np.asarray(value.nll))
            gradient += float(np.asarray(value.gradient))
            hessian += float(np.asarray(value.hessian))
            admissible = admissible and bool(np.asarray(value.admissible))
            fractional = fractional or bool(np.any(valid & (y > 0) & (y < 1)))
            batches += 1
        scans += 1
        return nll, gradient, hessian, admissible, fractional

    nll, gradient, hessian, admissible, fractional = evaluate(theta)
    history, theta_history = [nll], [float(theta[0])]
    status = (
        8
        if fractional
        else (
            0
            if admissible
            else (
                2
                if np.isfinite(nll)
                and np.isfinite(gradient)
                and not np.isfinite(hessian)
                else 1
            )
        )
    )
    iteration = 0
    active = abs(gradient) > control.tolerance * (abs(nll) + 1.0)
    for _iteration in range(control.max_iter):
        if status or not active:
            break
        iteration += 1
        step, usable = _step(gradient, hessian, control.max_step)
        step = float(np.asarray(step))
        if not bool(np.asarray(usable)):
            status = 2 if not np.isfinite(hessian) else (3 if hessian == 0 else 4)
            break
        stabilized += int(hessian <= 0)
        candidate = evaluate(theta + step)
        line_halvings = 0
        while candidate[0] - nll > np.finfo(float).eps ** 0.75 * abs(nll):
            step /= 2
            line_halvings += 1
            halvings += 1
            if np.all(theta == theta + step) or line_halvings > control.max_halvings:
                status = 5
                break
            candidate = evaluate(theta + step)
        if status or np.isnan(candidate[0]):
            status = 5
            break
        if not candidate[3]:
            status = 7
            break
        theta = theta + step
        nll, gradient, hessian, _, _ = candidate
        history.append(nll)
        theta_history.append(float(theta[0]))
        active = abs(gradient) > control.tolerance * (abs(nll) + 1.0)
    if not status and active:
        status = 6
    return NBStreamThetaResult(
        tuple(theta),
        nll,
        gradient,
        hessian,
        iteration,
        status,
        tuple(history),
        tuple(theta_history),
        scans,
        batches,
        halvings,
        stabilized,
    )
