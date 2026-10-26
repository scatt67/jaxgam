"""Host performance iteration of pinned ``bgam.fitd`` for regular families.

Pinned mgcv 1.9-3 ``bam(discrete=TRUE)`` fits a non-additive regular
exponential family (``bam.r::bgam.fitd`` lines 562-760) by *performance
iteration*: every outer iteration forms the Fisher working model at the
current coefficients and takes one ``Sl.fitChol`` proposal on it. This module
ports that controller exactly, including the parts the additive Gaussian
controller does not need:

* iteration one starts from ``eta = linkfun(mustart)`` of the family
  initializer, not from coefficients;
* from iteration three, a coefficient step that raises the penalized
  deviance ``dev + sum(rSb^2)`` (or makes it non-finite) is halved towards
  the previous accepted coefficients, at most 29 times (``kk < 30``);
* the working statistics are re-formed at every accepted coefficient vector,
  and the deviance/scale convergence test precedes each proposal.

Rows are reached only through a :class:`WorkingRows` provider that scans the
source; nothing here retains an ``n``-row array. A coefficient trial's scan
also forms the working statistics speculatively, so an accepted trial costs
one scan.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from jaxgam.execution.working_model import (
    AcceptedWorkingModelFactor,
    AcceptedWorkingModelProposal,
    accept_working_model_proposal,
)
from jaxgam.fitting.working_model import GaussianWorkingModelProposal

# bgam.fitd's ``kk < 30``: at most 29 coefficient step halvings.
_MAX_COEFFICIENT_TRIALS = 30


def source_sum(values: np.ndarray, carry: np.longdouble | float = 0.0) -> np.longdouble:
    """Continue R's ``sum`` (``rsum``) over ``values`` from ``carry``.

    R accumulates a double vector in element order in C ``long double``
    (``LDOUBLE``) and rounds once at the end. ``bgam.fitd``'s coefficient
    step control compares two such sums of ``dev.resids`` that can differ by
    rounding only near convergence, so their order and accumulator width
    decide the halving. NumPy's ``longdouble`` is the same C type, and
    ``add.accumulate`` is a strict left-to-right loop.
    """
    terms = np.asarray(values, dtype=np.longdouble).ravel()
    if not len(terms):
        return np.longdouble(carry)
    return np.add.accumulate(np.r_[np.longdouble(carry), terms])[-1]


@dataclass(frozen=True)
class WorkingStatistics:
    """One ``qrx``: ``X'WX`` (``qrx$R``), ``X'Wz`` (``qrx$f``), ``y.norm2``."""

    gram: np.ndarray
    rhs: np.ndarray
    response_square_sum: float


@dataclass(frozen=True)
class WorkingEvaluation:
    """Full-data deviance and, when requested, working statistics."""

    deviance: float
    statistics: WorkingStatistics | None


class WorkingRows(Protocol):
    """Row access for one fit: every call is a complete source scan."""

    def initial(self) -> WorkingEvaluation:
        """Deviance and statistics at ``eta = linkfun(mustart)``."""
        ...

    def evaluate(
        self, coefficients: np.ndarray, *, statistics: bool = True
    ) -> WorkingEvaluation:
        """Deviance (and statistics) at ``eta = X beta + offset``."""
        ...

    def response_variance(self) -> float:
        """Unweighted ``var(G$y)`` of the initialized response (after initial)."""
        ...


class WorkingNuisance(Protocol):
    """An extended family's per-iteration nuisance update (``estimate.theta``).

    ``update(coefficients)`` re-estimates the nuisance at the step-controlled
    coefficients, changes what the rows evaluate from then on, and returns a
    record kept in the iteration trace.
    """

    def update(self, coefficients: np.ndarray) -> object:
        """Re-estimate at ``coefficients``; the rows now use the new value."""
        ...


@dataclass(frozen=True)
class RegularWorkingModelState:
    """``bgam.fitd``'s carried state at the top of an outer iteration.

    ``completed`` counts finished iterations. ``parameters``/``step`` are
    the free ``lsp0``/``Nstep`` (free log smoothing parameters, then log
    scale when estimated). ``coefficients`` is ``coef`` from the last
    proposal; ``accepted_coefficients``/``accepted_deviance`` are ``coef0``
    and the unpenalized deviance at it (``dev0``); ``penalized_deviance`` is
    the next ``devold`` and ``criterion`` is ``crit``.
    """

    completed: int
    parameters: np.ndarray
    step: np.ndarray
    coefficients: np.ndarray
    accepted_coefficients: np.ndarray | None
    accepted_deviance: float
    penalized_deviance: float
    criterion: float

    @classmethod
    def initial(cls, n_coef: int) -> RegularWorkingModelState:
        """Before iteration one: ``Nstep = 0`` and no coefficients yet."""
        return cls(
            0, np.zeros(0), np.zeros(0), np.zeros(n_coef), None, np.nan, 0.0, 0.0
        )


@dataclass(frozen=True)
class RegularWorkingModelIteration:
    """One outer iteration, in public coefficient/sp order.

    ``step_control_margins`` holds, for every coefficient trial of the step
    control, ``(dev + bSb - (dev0 + bSb0)) / |dev0 + bSb0|``: a positive
    (or non-finite) margin is what makes ``bgam.fitd`` halve. A converging
    iteration has ``converged=True``, makes no proposal and keeps the
    previous parameters, criterion and step.
    """

    iteration: int
    coefficient_halvings: int
    deviance: float
    penalized_deviance: float
    parameters: np.ndarray
    proposed_step: np.ndarray
    criterion: float
    trial_evaluations: int
    step_control_margins: tuple[float, ...] = ()
    converged: bool = False
    nuisance: object | None = None


@dataclass(frozen=True)
class RegularWorkingModelAdvance:
    """One outer iteration: the step-controlled top and, unless converged,
    the accepted proposal and next carried state."""

    state: RegularWorkingModelState
    converged: bool
    coefficients: np.ndarray
    deviance: float
    penalized_deviance: float
    statistics: WorkingStatistics
    coefficient_halvings: int
    proposal: AcceptedWorkingModelProposal | None
    iteration: RegularWorkingModelIteration


@dataclass(frozen=True)
class RegularWorkingModelFit:
    """Row-free final ``bgam.fitd`` state for a regular family.

    ``coefficients``/``deviance`` are the final coefficients and the
    unpenalized deviance at them; ``penalized_deviance`` is the ``dev`` that
    enters the final criterion. ``factor``/``parameters`` belong to the last
    accepted ``Sl.fitChol`` proposal and ``gram`` is ``qrx$R`` from the last
    iteration top, exactly the pairing ``bgam.fitd`` uses for EDF and ``Vp``.
    """

    coefficients: np.ndarray
    deviance: float
    penalized_deviance: float
    parameters: np.ndarray
    log_scale: float
    converged: bool
    n_iter: int
    proposal_evaluations: int
    coefficient_halvings: int
    trace: tuple[RegularWorkingModelIteration, ...]
    factor: AcceptedWorkingModelFactor
    gram: np.ndarray


def _finite_statistics(evaluation: WorkingEvaluation) -> WorkingStatistics:
    statistics = evaluation.statistics
    if statistics is None:
        raise RuntimeError("Working-model rows returned no statistics.")
    if (
        not np.all(np.isfinite(statistics.gram))
        or not np.all(np.isfinite(statistics.rhs))
        or not np.isfinite(statistics.response_square_sum)
    ):
        raise FloatingPointError("Non-finite working-model statistics.")
    return statistics


def advance_regular_working_model(
    state: RegularWorkingModelState,
    rows: WorkingRows,
    propose: Callable[
        [np.ndarray, float, WorkingStatistics], GaussianWorkingModelProposal
    ],
    penalty: Callable[[np.ndarray, np.ndarray], float],
    *,
    scale_estimated: bool,
    epsilon: float,
    start: Callable[[WorkingStatistics, float], np.ndarray],
    nuisance: WorkingNuisance | None = None,
    _oracle_replay_halvings: int | None = None,
) -> RegularWorkingModelAdvance:
    """Port one non-additive ``bgam.fitd`` outer iteration from ``state``.

    ``propose(trial, tolerance, statistics)`` evaluates the compiled
    ``Sl.fitChol`` proposal on the given statistics; ``penalty(beta, trial)``
    is ``sum(Sl.rSb(Sl, lsp, beta)^2)`` at the log smoothing parameters of the
    free trial vector; ``start(statistics, variance)`` returns iteration
    one's free ``lsp0`` (``initial.sp``/``in.out`` and, when estimated,
    ``log(var(G$y)*.05)`` from ``variance``, which is ``nan`` for a known
    scale).

    ``nuisance`` (an extended family with estimated theta) follows
    ``bgam.fitd``'s order: ``dev0`` is re-evaluated with the current theta,
    the step control and the penalized deviance use that theta, and from
    iteration two the theta is re-estimated at the step-controlled
    coefficients before the working model is formed with it.

    ``_oracle_replay_halvings`` is a test/oracle hook, never set by the
    public API: it replays a known source step-control outcome (``kk - 1``)
    instead of deciding it, while still recording every trial's margin, so
    a matched-state oracle can follow a source trajectory through a
    decision whose two penalized deviances are equal to rounding.
    """
    iteration = state.completed + 1
    if _oracle_replay_halvings is not None and (
        iteration < 3 or not 0 <= _oracle_replay_halvings < _MAX_COEFFICIENT_TRIALS
    ):
        raise ValueError(
            "_oracle_replay_halvings needs bgam.fitd's step control (iteration > 2) "
            f"and at most {_MAX_COEFFICIENT_TRIALS - 1} halvings."
        )
    margins: list[float] = []
    if iteration == 1:
        evaluation = rows.initial()
        deviance = evaluation.deviance
        coefficients = state.coefficients
        penalized = deviance
        halvings = 0
    else:
        coefficients = state.coefficients
        bound = state.parameters
        bsb = penalty(coefficients, bound)
        previous = None
        if iteration > 2:
            assert state.accepted_coefficients is not None
            # dev0 <- sum(dev.resids(G$y, mu0, G$w, theta)): the current theta.
            accepted_deviance = (
                state.accepted_deviance
                if nuisance is None
                else rows.evaluate(
                    state.accepted_coefficients, statistics=False
                ).deviance
            )
            previous = accepted_deviance + penalty(state.accepted_coefficients, bound)
        halvings = 0
        while True:
            evaluation = rows.evaluate(coefficients, statistics=nuisance is None)
            deviance = evaluation.deviance
            if previous is not None:
                margins.append(
                    float((deviance + bsb - previous) / abs(previous))
                    if np.isfinite(deviance)
                    else np.inf
                )
            halve = (
                halvings < _oracle_replay_halvings
                if _oracle_replay_halvings is not None
                else previous is not None
                and (not np.isfinite(deviance) or previous < deviance + bsb)
                and halvings + 1 < _MAX_COEFFICIENT_TRIALS
            )
            if halve:
                # coef0*istepr + coef/stepr with stepr = 2.
                coefficients = state.accepted_coefficients * 0.5 + coefficients / 2.0
                bsb = penalty(coefficients, bound)
                halvings += 1
                continue
            break
        penalized = deviance + bsb
    update = None
    if nuisance is not None and iteration > 1:
        # estimate.theta at mu, then dDeta with the new theta. bgam.fitd runs
        # both before its stop("Non-finite deviance"), so a non-finite mu
        # fails inside the theta update first.
        update = nuisance.update(coefficients)
    if not np.isfinite(penalized):
        raise FloatingPointError(
            f"Non-finite working-model deviance at iteration {iteration}."
        )
    if update is not None:
        evaluation = rows.evaluate(coefficients)
    statistics = _finite_statistics(evaluation)
    # log.phi is the last accepted trial's log scale (log(1) when known).
    if (
        iteration > 2
        and abs(penalized - state.penalized_deviance) / (0.1 + abs(penalized)) < epsilon
        and (
            not scale_estimated
            or abs(state.step[-1]) < epsilon * (abs(state.parameters[-1]) + 1.0)
        )
    ):
        return RegularWorkingModelAdvance(
            state,
            True,
            coefficients,
            deviance,
            penalized,
            statistics,
            halvings,
            None,
            RegularWorkingModelIteration(
                iteration,
                halvings,
                deviance,
                penalized,
                state.parameters.copy(),
                state.step.copy(),
                state.criterion,
                0,
                tuple(margins),
                True,
                update,
            ),
        )
    if iteration == 1:
        variance = rows.response_variance() if scale_estimated else np.nan
        parameters = np.array(start(statistics, variance), dtype=np.float64)
        if parameters.ndim != 1 or not np.all(np.isfinite(parameters)):
            raise ValueError("Working-model start parameters must be finite.")
        step = np.zeros_like(parameters)
        criterion = deviance  # crit <- dev before the first proposal
    else:
        parameters, step, criterion = state.parameters, state.step, state.criterion
    accepted = accept_working_model_proposal(
        parameters,
        step,
        criterion,
        penalized,
        lambda trial, tolerance: propose(trial, tolerance, statistics),
    )
    accepted_log_scale = float(accepted.parameters[-1]) if scale_estimated else 0.0
    # crit <- (dev/(exp(log.phi)*gamma) - ldetS + ldetXXS)/2, gamma = 1.
    next_criterion = (
        penalized / float(np.exp(accepted_log_scale))
        - accepted.factor.logdet_penalty
        + accepted.factor.logdet_hessian
    ) / 2.0
    record = RegularWorkingModelIteration(
        iteration,
        halvings,
        deviance,
        penalized,
        accepted.parameters.copy(),
        accepted.proposed_step.copy(),
        next_criterion,
        accepted.evaluations,
        tuple(margins),
        False,
        update,
    )
    return RegularWorkingModelAdvance(
        RegularWorkingModelState(
            iteration,
            accepted.parameters,
            accepted.proposed_step,
            accepted.coefficients,
            None if iteration == 1 else coefficients,
            np.nan if iteration == 1 else deviance,
            penalized,
            next_criterion,
        ),
        False,
        coefficients,
        deviance,
        penalized,
        statistics,
        halvings,
        accepted,
        record,
    )


def fit_regular_working_model(
    rows: WorkingRows,
    propose: Callable[
        [np.ndarray, float, WorkingStatistics], GaussianWorkingModelProposal
    ],
    penalty: Callable[[np.ndarray, np.ndarray], float],
    n_coef: int,
    *,
    scale_estimated: bool,
    epsilon: float,
    max_iterations: int,
    start: Callable[[WorkingStatistics, float], np.ndarray],
    nuisance: WorkingNuisance | None = None,
    _oracle_replay_halvings: Mapping[int, int] | None = None,
) -> RegularWorkingModelFit:
    """Run ``bgam.fitd``'s outer loop to convergence or ``maxit``.

    With an estimated ``nuisance`` the converging iteration has already
    re-estimated it, so the final deviance is re-evaluated with that value,
    as ``bam``'s residuals are.

    ``_oracle_replay_halvings`` (test/oracle hook) maps outer iterations to
    replayed step-control outcomes (see :func:`advance_regular_working_model`).

    At convergence the final coefficients are the step-controlled ones of
    the converging iteration, paired with the previous proposal's factor and
    smoothing parameters. At ``maxit`` they are the last proposal's
    coefficients, whose deviance needs one more deviance-only scan.
    """
    state = RegularWorkingModelState.initial(n_coef)
    trace: list[RegularWorkingModelIteration] = []
    evaluations = 0
    halvings = 0
    last: RegularWorkingModelAdvance | None = None
    accepted: AcceptedWorkingModelProposal | None = None
    converged = False
    while state.completed < max_iterations:
        advance = advance_regular_working_model(
            state,
            rows,
            propose,
            penalty,
            scale_estimated=scale_estimated,
            epsilon=epsilon,
            start=start,
            nuisance=nuisance,
            _oracle_replay_halvings=None
            if _oracle_replay_halvings is None
            else _oracle_replay_halvings.get(state.completed + 1),
        )
        halvings += advance.coefficient_halvings
        last = advance
        trace.append(advance.iteration)
        if advance.converged:
            converged = True
            break
        assert advance.proposal is not None
        accepted = advance.proposal
        evaluations += accepted.evaluations
        state = advance.state
    assert last is not None
    assert accepted is not None
    if converged:
        coefficients = last.coefficients
        deviance = (
            last.deviance
            if nuisance is None
            else rows.evaluate(coefficients, statistics=False).deviance
        )
        if not np.isfinite(deviance):
            raise FloatingPointError("Non-finite final working-model deviance.")
        n_iter = state.completed + 1
    else:
        coefficients = state.coefficients
        deviance = rows.evaluate(coefficients, statistics=False).deviance
        if not np.isfinite(deviance):
            raise FloatingPointError("Non-finite final working-model deviance.")
        n_iter = state.completed
    coefficients = np.array(coefficients, dtype=np.float64)
    coefficients.setflags(write=False)
    return RegularWorkingModelFit(
        coefficients,
        float(deviance),
        float(last.penalized_deviance),
        accepted.parameters.copy(),
        float(accepted.parameters[-1]) if scale_estimated else 0.0,
        converged,
        n_iter,
        evaluations,
        halvings,
        tuple(trace),
        accepted.factor,
        last.statistics.gram,
    )
