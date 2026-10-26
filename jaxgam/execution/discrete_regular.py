"""Regular-family continuous-discrete fitting by pinned ``bgam.fitd``.

Pinned mgcv 1.9-3 ``bam(discrete=TRUE)`` fits a non-additive regular
exponential family (Poisson, Binomial, Gamma) by *performance iteration*
(``bam.r::bgam.fitd`` lines 562-760): every outer iteration forms the
working model of the current coefficients and takes one ``Sl.fitChol``
proposal on it. These are the discrete method's semantics for every link
and they differ from JaxGAM's dense ``gam.fit3`` route:

* working weights and pseudodata are IRLS/Fisher, ``w = G$w*mu.eta^2/
  variance(mu)`` and ``z = (eta-offset) + (y-mu)/mu.eta`` (a zero
  ``mu.eta`` is replaced by 0.1), with no observed-curvature split, weight
  clipping or ``good``-row filtering;
* coefficient step control halves towards the previous coefficients while
  the raw penalized ``sum(dev.resids) + b'Sb`` rises, with no ``validmu``/
  ``valideta`` check between iterations, and convergence is a penalized
  deviance change (plus the log-scale step when the scale is estimated);
* the scale is fixed at one for Poisson/Binomial and is a working-model
  parameter (``log.phi``) for Gamma, started at ``log(var(G$y)*.05)``.

``sp`` fixes every smoothing parameter (pinned ``bam(sp=)``: ``Sl.fitChol``
then updates only an estimated scale); ``sp=None`` with
``smoothing_strategy='working_model_reml'`` also selects them from each
iteration's working model. Rows are only reached by complete bounded source
scans; nothing here retains an ``n``-row array.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from jaxgam.control import FitControl
from jaxgam.data.source import RowBatch, RowSource
from jaxgam.execution.discrete import (
    CenteredMoments,
    DiscreteFit,
    DiscreteFitMemoryPlan,
    _check_fit_budget,
    _fixed_sp,
    _penalty_rank,
    _pinned_criterion_null_dimension,
    _prepared_discrete,
    _PreparedDiscrete,
)
from jaxgam.execution.regular_working_model import (
    RegularWorkingModelFit,
    WorkingEvaluation,
    WorkingRows,
    WorkingStatistics,
    fit_regular_working_model,
    source_sum,
)
from jaxgam.execution.working_model import (
    CompiledWorkingModelProposal,
    compile_working_model_proposal,
    source_penalty_quadratic,
)
from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting.discrete_ops import to_device_discrete_design
from jaxgam.fitting.discrete_working import (
    DiscreteWorkingBatch,
    discrete_initial_working_batch,
    discrete_working_batch,
)
from jaxgam.fitting.working_model import GaussianWorkingModelProposal
from jaxgam.formula.discrete_design import DiscreteMemoryPlan, FrozenDiscreteDesign
from jaxgam.formula.fitting_prepare import initial_log_sp_from_diagonal
from jaxgam.formula.terms import FormulaSpec
from jaxgam.inference.predictor import PivotedQRFisherFactor
from jaxgam.penalties.structure import DenseLocalPenalty, PenaltyStructure


def _regular_solver_plan(
    setup_retained_bytes: int,
    descriptor: DiscreteMemoryPlan,
    n_coef: int,
    penalties: PenaltyStructure,
    control: FitControl,
) -> DiscreteFitMemoryPlan:
    """Bound the performance-iteration row and statistic workspace.

    The coefficient-space proposal workspace is measured separately when it
    is compiled. Here: per batch, host response/weight/offset/positions plus
    the initial ``mustart``/``eta``/``mu`` arrays and the normalized response,
    the device family vectors (``eta``, ``mu``, ``mu.eta``, ``w``, ``z``,
    deviance rows and products) alongside the indexed action peak, and the
    host deviance-row reduction; and four host statistic sets (trial,
    accepted, final ``qrx$R`` and the reduction accumulator), each ``X'WX``
    plus ``X'Wz``.
    """
    float_size = np.dtype(np.float64).itemsize
    # Plus the host deviance rows and their two long-double accumulation
    # copies (R's row-order sum).
    row_workspace = control.batch_rows * (
        25 * float_size
        + 2 * np.dtype(np.longdouble).itemsize
        + 2 * np.dtype(np.bool_).itemsize
    )
    local_materializations = sum(
        block.size * block.size
        for block in penalties.blocks
        for component in block.local_penalties
        if not isinstance(component, DenseLocalPenalty)
    )
    matrix_count = 4 + (2 if control.uncertainty == "covariance" else 1)
    solver = (
        float_size
        * (
            matrix_count * n_coef * n_coef
            + local_materializations
            + 4 * (n_coef * n_coef + n_coef)
        )
        + row_workspace
    )
    host = descriptor.host_retained_bytes + descriptor.selector_resident_bytes
    peak = setup_retained_bytes + host + descriptor.device_peak_bytes + solver
    return DiscreteFitMemoryPlan(
        setup_retained_bytes,
        host,
        descriptor.device_peak_bytes,
        solver,
        peak,
        control.memory_budget_bytes,
    )


class _DiscreteWorkingRows:
    """Source scans that evaluate ``bgam.fitd``'s working model in batches.

    Every call replays the whole source. The response is the family's
    initialized ``G$y`` (Binomial zero-weight rows are set to zero, as in
    ``binomial()$initialize``); per-batch device arrays are released before
    the next batch is read.
    """

    def __init__(
        self,
        source: RowSource,
        descriptor: FrozenDiscreteDesign,
        device_descriptor: object,
        family: ExponentialFamily,
        batch_rows: int,
        fingerprint: str,
    ) -> None:
        self.source = source
        self.descriptor = descriptor
        self.device_descriptor = device_descriptor
        self.family = family
        self.batch_rows = batch_rows
        self.fingerprint = fingerprint
        self.scans = 0
        self.batches = 0
        self.offset_nonzero = False
        self.moments = CenteredMoments()

    def _scan(self) -> Iterator[tuple[RowBatch, tuple[tuple[np.ndarray, ...], ...]]]:
        cursor = 0
        for batch in self.source.scan(self.batch_rows):
            size = len(batch.row_positions)
            if size == 0 or cursor + size > self.source.n_rows:
                raise RuntimeError("RowSource yielded an invalid fitting row count.")
            if batch.y is None:
                raise ValueError("Discrete fitting requires a RowSource response.")
            if not np.all(np.asarray(batch.valid)):
                raise NotImplementedError(
                    "Discrete fitting does not yet support padded invalid rows."
                )
            weight = np.asarray(batch.weight, dtype=np.float64)
            if (
                not np.all(np.isfinite(np.asarray(batch.y, dtype=np.float64)))
                or not np.all(np.isfinite(weight))
                or not np.all(np.isfinite(np.asarray(batch.offset, dtype=np.float64)))
                or np.any(weight < 0.0)
            ):
                raise ValueError(
                    "Discrete fitting inputs must be finite with nonnegative weights."
                )
            rows = np.arange(cursor, cursor + size, dtype=np.intp)
            yield batch, self.descriptor.batch_indices(rows)
            cursor += size
            self.batches += 1
        if cursor != self.source.n_rows:
            raise RuntimeError("RowSource yielded fewer fitting rows than declared.")
        if self.source.fingerprint() != self.fingerprint:
            raise RuntimeError("RowSource changed during discrete fitting.")
        self.scans += 1

    def _reduce(
        self, batches: Iterator[DiscreteWorkingBatch], statistics: bool
    ) -> WorkingEvaluation:
        n_coef = self.descriptor.n_coef
        deviance = np.longdouble(0.0)
        gram = np.zeros((n_coef, n_coef)) if statistics else None
        rhs = np.zeros(n_coef) if statistics else None
        response_square_sum = 0.0
        for result in batches:
            # R's sum(dev.resids): row order, long double, one final rounding.
            deviance = source_sum(np.asarray(result.deviance_rows), deviance)
            if statistics:
                gram += np.asarray(result.gram)
                rhs += np.asarray(result.rhs)
                response_square_sum += float(result.response_square_sum)
            del result
        return WorkingEvaluation(
            float(deviance),
            WorkingStatistics(gram, rhs, response_square_sum) if statistics else None,
        )

    def initial(self) -> WorkingEvaluation:
        family = self.family

        def batches() -> Iterator[DiscreteWorkingBatch]:
            for batch, selectors in self._scan():
                y = np.asarray(batch.y, dtype=np.float64)
                weight = np.asarray(batch.weight, dtype=np.float64)
                offset = np.asarray(batch.offset, dtype=np.float64)
                state = family.initial_working_state_cpu(
                    y, weight, np.asarray(batch.valid, dtype=bool)
                )
                if not state.input_ok:
                    raise ValueError(
                        f"{family.family_name} initialization rejects the response: "
                        f"values must lie in {family.response_support}."
                    )
                if not state.domain_ok:
                    raise FloatingPointError(
                        "cannot find valid starting values: linkfun(mustart) is "
                        "outside the family domain."
                    )
                response = family.execution_initial_response_cpu(y, weight)
                self.moments.add(response)
                self.offset_nonzero |= bool(np.any(offset != 0.0))
                yield discrete_initial_working_batch(
                    self.device_descriptor,
                    selectors,
                    np.asarray(state.eta),
                    response,
                    weight,
                    offset,
                    family=family,
                )
                del batch, selectors, y, weight, offset, state, response

        return self._reduce(batches(), True)

    def evaluate(
        self, coefficients: np.ndarray, *, statistics: bool = True
    ) -> WorkingEvaluation:
        family = self.family
        beta = np.asarray(coefficients, dtype=np.float64)

        def batches() -> Iterator[DiscreteWorkingBatch]:
            for batch, selectors in self._scan():
                weight = np.asarray(batch.weight, dtype=np.float64)
                yield discrete_working_batch(
                    self.device_descriptor,
                    selectors,
                    beta,
                    family.execution_initial_response_cpu(
                        np.asarray(batch.y, dtype=np.float64), weight
                    ),
                    weight,
                    np.asarray(batch.offset, dtype=np.float64),
                    family=family,
                    statistics=statistics,
                )
                del batch, selectors, weight

        return self._reduce(batches(), statistics)

    def response_variance(self) -> float:
        return self.moments.variance()


@dataclass(frozen=True)
class RegularWorkingComponents:
    """Everything one regular-family performance iteration consumes.

    ``rows``/``propose``/``penalty``/``start`` are the arguments of
    :func:`fit_regular_working_model` (and of a matched-state replay of
    :func:`advance_regular_working_model`); ``compiled`` maps free trial
    parameters to the full log smoothing vector. :meth:`run` executes the
    controller and :meth:`result` assembles its row-free fit state.
    """

    problem: _PreparedDiscrete
    memory: DiscreteFitMemoryPlan
    compiled: CompiledWorkingModelProposal
    rows: _DiscreteWorkingRows
    propose: Callable[..., GaussianWorkingModelProposal]
    penalty: Callable[[np.ndarray, np.ndarray], float]
    start: Callable[[WorkingStatistics, float], np.ndarray]
    scale_estimated: bool
    smoothing: np.ndarray | None
    family: ExponentialFamily
    control: FitControl
    n_obs: int

    def run(
        self,
        rows: WorkingRows | None = None,
        *,
        _oracle_replay_halvings: Mapping[int, int] | None = None,
    ) -> RegularWorkingModelFit:
        """Run ``bgam.fitd``'s outer loop on ``rows`` (the source by default).

        Another :class:`WorkingRows` provider (an explicit design, say) runs
        the identical controller and proposals; ``_oracle_replay_halvings``
        is the test/oracle hook of :func:`fit_regular_working_model`.
        """
        return fit_regular_working_model(
            self.rows if rows is None else rows,
            self.propose,
            self.penalty,
            self.problem.descriptor.n_coef,
            scale_estimated=self.scale_estimated,
            epsilon=self.control.working_model_epsilon,
            max_iterations=self.control.working_model_max_iterations,
            start=self.start,
            _oracle_replay_halvings=_oracle_replay_halvings,
        )

    def result(self, fit: RegularWorkingModelFit) -> DiscreteFit:
        """Assemble pinned ``bgam.fitd``'s final fields from ``fit``.

        ``Vp = PP*scale`` and ``edf = diag(PP %*% qrx$R)`` pair the last
        accepted proposal's ``PP`` with the last iteration top's ``X'WX``;
        the score is the final ``crit`` with the reset null dimension ``Mp``.
        """
        problem, control = self.problem, self.control
        setup = problem.setup
        n_coef = problem.descriptor.n_coef
        working_model = self.smoothing is None
        log_smoothing = self.compiled.log_smoothing(fit.parameters)
        smoothing = np.exp(log_smoothing) if working_model else self.smoothing
        scale = float(np.exp(fit.log_scale))
        factor = fit.factor
        kept = factor.pivots[: factor.rank]
        qr_factor = PivotedQRFisherFactor(
            R=factor.upper[: factor.rank, : factor.rank]
            * factor.diagonal[kept][None, :],
            pivots=np.arange(factor.rank, dtype=np.intp),
            keep=kept,
            original_n_coef=n_coef,
        )
        inverse = qr_factor.hessian_inverse(np.eye(n_coef))
        transforms = []
        for block in self.compiled.metadata.penalties.blocks:
            values = np.asarray(block.transform.values).copy()
            values.setflags(write=False)
            transforms.append((block.start, block.stop, "dense", values))
            inverse[block.start : block.stop, :] = (
                values @ inverse[block.start : block.stop, :]
            )
            inverse[:, block.start : block.stop] = (
                inverse[:, block.start : block.stop] @ values.T
            )
        edf = float(np.trace(inverse @ fit.gram))
        null_dimension = n_coef - _penalty_rank(problem.penalties)
        criterion_null = _pinned_criterion_null_dimension(
            setup.predict_spec, null_dimension
        )
        score = float(
            (
                fit.penalized_deviance / scale
                - factor.logdet_penalty
                + factor.logdet_hessian
                + (self.n_obs - criterion_null) * np.log(2.0 * np.pi * scale)
            )
            / 2.0
        )
        if not np.isfinite(score) or not np.isfinite(edf):
            raise FloatingPointError("Unusable working-model score or EDF.")
        covariance = scale * inverse if control.uncertainty == "covariance" else None
        retain_factor = control.uncertainty == "fisher"
        return DiscreteFit(
            np.array(fit.coefficients),
            fit.deviance,
            score,
            scale,
            edf,
            np.asarray(smoothing, dtype=np.float64),
            None,
            covariance,
            setup.predict_spec,
            self.n_obs,
            self.rows.offset_nonzero,
            self.rows.scans,
            self.rows.batches,
            fit.n_iter,
            self.memory,
            "working_model_reml" if working_model else "fixed",
            fit.converged,
            fit.proposal_evaluations,
            fit.trace,
            qr_factor if retain_factor else None,
            tuple(transforms) if retain_factor else (),
            problem.discretization,
            problem.request_compression,
            performance_iteration=True,
            coefficient_halvings=fit.coefficient_halvings,
        )


@contextmanager
def regular_working_components(
    spec: FormulaSpec,
    source: RowSource,
    family: ExponentialFamily,
    sp: npt.ArrayLike | None,
    *,
    control: FitControl,
    device: object | None = None,
) -> Iterator[RegularWorkingComponents]:
    """Prepare, budget and compile one regular-family discrete fit.

    ``sp`` fixes every smoothing parameter (``Sl.fitChol`` then updates only
    an estimated scale); ``sp=None`` selects them from each working model.
    """
    working_model = sp is None
    if working_model and control.smoothing_strategy != "working_model_reml":
        raise NotImplementedError("Discrete estimated sp requires working_model_reml.")
    if not family.execution_capabilities().regular_performance_iteration:
        raise NotImplementedError(
            f"{family!r} lacks the regular_performance_iteration capability."
        )
    with _prepared_discrete(spec, source, control) as problem:
        setup = problem.setup
        descriptor = problem.descriptor
        penalties = problem.penalties
        smoothing = fixed_log_smoothing = None
        if not working_model:
            smoothing, fixed_log_smoothing = _fixed_sp(sp, penalties.n_penalties)
        scale_estimated = not family.scale_known
        memory = _regular_solver_plan(
            setup.retained_table_bytes,
            descriptor.memory_plan,
            descriptor.n_coef,
            penalties,
            control,
        )
        _check_fit_budget(memory)
        compiled = compile_working_model_proposal(
            penalties,
            source.n_rows,
            determinant_penalties=setup.source_determinant_penalties,
            control=control,
            available_workspace_bytes=memory.budget_bytes - memory.fit_peak_bytes,
            device=device,
            scale_is_fixed=not scale_estimated,
            fixed_log_smoothing=fixed_log_smoothing,
        )
        extra = compiled.memory_plan.additional_peak_bytes
        memory = replace(
            memory,
            solver_workspace_bytes=memory.solver_workspace_bytes + extra,
            fit_peak_bytes=memory.fit_peak_bytes + extra,
        )
        rows = _DiscreteWorkingRows(
            source,
            descriptor,
            to_device_discrete_design(descriptor, device),
            family,
            control.batch_rows,
            problem.fingerprint,
        )
        transferred: list[object] = [None, None]

        def propose(
            trial: np.ndarray, tolerance: float, statistics: WorkingStatistics
        ) -> GaussianWorkingModelProposal:
            # One device copy per iteration's statistics, reused by its trials.
            if transferred[0] is not statistics:
                transferred[0] = statistics
                transferred[1] = compiled.transfer(
                    statistics.gram, statistics.rhs, statistics.response_square_sum
                )
            return compiled(trial, tolerance, transferred[1])

        def penalty(beta: np.ndarray, trial: np.ndarray) -> float:
            return source_penalty_quadratic(
                np.asarray(beta), compiled.log_smoothing(trial), compiled.metadata
            )

        def start(statistics: WorkingStatistics, variance: float) -> np.ndarray:
            parameters: list[float] = []
            if working_model:
                # initial.sp on the untransformed iteration-one X'WX.
                parameters.extend(
                    initial_log_sp_from_diagonal(np.diag(statistics.gram), penalties)
                )
            if scale_estimated:
                if not np.isfinite(variance) or variance <= 0.0:
                    raise FloatingPointError(
                        "Scale initialization log(var(y)*0.05) is not finite."
                    )
                parameters.append(float(np.log(variance * 0.05)))
            return np.asarray(parameters, dtype=np.float64)

        try:
            yield RegularWorkingComponents(
                problem,
                memory,
                compiled,
                rows,
                propose,
                penalty,
                start,
                scale_estimated,
                smoothing,
                family,
                control,
                source.n_rows,
            )
        finally:
            transferred.clear()  # Release the last device statistics.


def fit_discrete_regular(
    spec: FormulaSpec,
    source: RowSource,
    family: ExponentialFamily,
    sp: npt.ArrayLike | None,
    *,
    control: FitControl,
    device: object | None = None,
    _oracle_replay_halvings: Mapping[int, int] | None = None,
) -> DiscreteFit:
    """Fit a regular exponential family by ``bgam.fitd`` performance iteration.

    ``sp`` fixes every smoothing parameter (pinned ``bam(sp=)``); ``sp=None``
    with ``smoothing_strategy='working_model_reml'`` also selects them from
    each iteration's working model.

    ``_oracle_replay_halvings`` (outer iteration to coefficient step
    halvings) is a test/oracle hook for matched-state R oracles: it replays a
    source step-control outcome at a rounding-level tie instead of deciding
    it. ``GAM``, ``FitControl`` and every result leave it unset, so public
    fits always decide every step.
    """
    with regular_working_components(
        spec, source, family, sp, control=control, device=device
    ) as parts:
        fit = parts.run(_oracle_replay_halvings=_oracle_replay_halvings)
        if not fit.converged:
            warnings.warn(
                f"{family.family_name} discrete performance iteration did not "
                "converge within its iteration limit.",
                RuntimeWarning,
                stacklevel=2,
            )
        return parts.result(fit)
