"""Host trial-state control for pinned additive-Gaussian performance REML.

Numerical proposals are JIT compiled in fitting/working_model.py. This
controller ports bgam.fitd's finite trial and convergence sequence over one
frozen coefficient-space working model; it never scans training rows.

Each trial crosses one explicit host boundary: the kernel assembles the
preconditioned matrices R passes to ``chol(pivot=TRUE)``, the host factors
them with the pinned scipy-openblas32 DPSTRF that pinned R's OpenBLAS 0.3.26
LAPACK runs, and the proposal kernel consumes those factors. ``exp(log sp)``
is evaluated once per trial with the C library ``exp`` R uses.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np

from jaxgam.control import FitControl
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.working_model import (
    GaussianWorkingModelProposal,
    WorkingModelCriterionMetadata,
    WorkingModelFactor,
    WorkingModelFactorInputs,
    WorkingModelFactors,
    gaussian_working_model_proposal,
    working_model_factor_inputs,
)
from jaxgam.formula.fitting_prepare import initial_log_sp_from_diagonal
from jaxgam.penalties.structure import PenaltyStructure
from jaxgam.smooths.source_lapack import (
    source_pivoted_cholesky,
    source_pivoted_cholesky_workspace_bound,
)


@dataclass(frozen=True)
class WorkingModelMemoryPlan:
    """Conservative live array storage, including measured compiled scratch."""

    metadata_host_peak_bytes: int
    metadata_device_peak_bytes: int
    statistic_device_bytes: int
    state_and_trace_bytes: int
    compiled_temporary_bytes: int
    compiled_output_bytes: int
    compiled_alias_bytes: int
    factor_workspace_bytes: int
    additional_peak_bytes: int


@dataclass(frozen=True)
class WorkingModelIteration:
    """One accepted trial and its proposed next step, in public sp order."""

    iteration: int
    parameters: np.ndarray
    proposed_step: np.ndarray
    previous_penalized_deviance: float
    penalized_deviance: float
    control_criterion: float
    trial_evaluations: int


@dataclass(frozen=True)
class GaussianWorkingModelFit:
    """Row-free accepted Gaussian smoothing/scale state."""

    coefficients: np.ndarray
    log_smoothing: np.ndarray
    scale: float
    converged: bool
    n_iter: int
    proposal_evaluations: int
    trace: tuple[WorkingModelIteration, ...]
    memory_plan: WorkingModelMemoryPlan
    factor_upper: np.ndarray
    factor_pivots: np.ndarray
    factor_rank: int
    factor_diagonal: np.ndarray
    logdet_hessian: float
    logdet_penalty: float
    coefficient_transforms: tuple[tuple[int, int, str, np.ndarray], ...]
    penalty_quadratic: float


@dataclass(frozen=True)
class WorkingModelState:
    """bgam.fitd's carried smoothing state at the top of an outer iteration.

    ``completed`` counts finished outer iterations (source ``iter - 1``).
    ``parameters`` and ``step`` are ``lsp0`` and ``Nstep`` (log smoothing
    parameters, then log scale). ``coefficients`` are the accepted ``coef``
    in the coordinates of the frozen statistics. ``deviance`` is the
    penalized deviance evaluated at the top of the last completed iteration
    (the next ``devold``); ``criterion`` is ``crit``, which owns the
    active-coordinate tolerance of the next proposal.
    """

    completed: int
    parameters: np.ndarray
    step: np.ndarray
    coefficients: np.ndarray
    deviance: float
    criterion: float

    @classmethod
    def initial(cls, parameters: np.ndarray, n_coef: int) -> WorkingModelState:
        """Source iteration-one state: zero step, coefficients and criterion."""
        start = np.array(parameters, dtype=np.float64)
        if start.ndim != 1 or not len(start) or not np.all(np.isfinite(start)):
            raise ValueError("Working-model start parameters must be finite.")
        return cls(0, start, np.zeros_like(start), np.zeros(n_coef), 0.0, 0.0)


@dataclass(frozen=True)
class AcceptedWorkingModelFactor:
    """Read-only host factor and log determinants of one accepted trial."""

    upper: np.ndarray
    pivots: np.ndarray
    rank: int
    diagonal: np.ndarray
    logdet_hessian: float
    logdet_penalty: float


@dataclass(frozen=True)
class WorkingModelAdvance:
    """One bgam.fitd outer iteration: convergence or one accepted proposal.

    ``deviance`` is the penalized deviance evaluated at the top of the
    iteration. A converged advance carries the unchanged state, no trace
    entry and no factor; otherwise ``state`` is the next carried state.
    """

    state: WorkingModelState
    converged: bool
    deviance: float
    iteration: WorkingModelIteration | None
    proposal_evaluations: int
    factor: AcceptedWorkingModelFactor | None


def _source_penalty_quadratic(
    coefficients: np.ndarray,
    log_smoothing: np.ndarray,
    metadata: WorkingModelCriterionMetadata,
) -> float:
    """Sl.rSb's ``b'Sb`` in the initial Sl.setup coefficient coordinates.

    The local working penalties are identities or projected range-space
    penalties there, so a very large smoothing parameter cannot cancel
    against null-space coefficients as an original-coordinate ``b'Sb`` can.
    """
    total = 0.0
    for block in metadata.penalties.blocks:
        local = np.linalg.solve(
            np.asarray(block.transform.values),
            coefficients[block.start : block.stop],
        )
        for sp, penalty in zip(block.sp_indices, block.penalties, strict=True):
            values = np.asarray(penalty_ops._dense_local(penalty))
            total += float(np.exp(log_smoothing[sp])) * float(local @ values @ local)
    return total


def source_exponentials(log_smoothing: np.ndarray) -> np.ndarray:
    """``exp(log sp)`` from the C library ``exp`` that pinned R calls."""
    return np.array([math.exp(float(value)) for value in log_smoothing])


def _host_factor(matrix: jax.Array, diagonal: jax.Array) -> WorkingModelFactor:
    host = np.asarray(matrix)
    scales = np.asarray(diagonal)
    if not np.all(np.isfinite(host)) or not np.all(np.isfinite(scales)):
        raise FloatingPointError("Non-finite Gaussian working-model proposal.")
    factor = source_pivoted_cholesky(host)
    del host
    return WorkingModelFactor(
        factor.upper, factor.pivots, np.int32(factor.rank), scales.copy()
    )


def factor_working_model(inputs: WorkingModelFactorInputs) -> WorkingModelFactors:
    """Pinned DPSTRF factors of one trial's Hessian and coupled penalties."""
    return WorkingModelFactors(
        _host_factor(inputs.hessian, inputs.hessian_diagonal),
        tuple(
            _host_factor(matrix, diagonal)
            for matrix, diagonal in zip(
                inputs.penalties, inputs.penalty_diagonals, strict=True
            )
        ),
    )


def _factor_sizes(metadata: WorkingModelCriterionMetadata) -> tuple[int, ...]:
    return (
        metadata.penalties.n_coef,
        *(len(products[0]) for products in metadata.multi_block_proj_S),
    )


def _factor_shapes(metadata: WorkingModelCriterionMetadata) -> WorkingModelFactors:
    def shape(n: int) -> WorkingModelFactor:
        return WorkingModelFactor(
            jax.ShapeDtypeStruct((n, n), np.float64),
            jax.ShapeDtypeStruct((n,), np.int32),
            jax.ShapeDtypeStruct((), np.int32),
            jax.ShapeDtypeStruct((n,), np.float64),
        )

    hessian, *penalties = _factor_sizes(metadata)
    return WorkingModelFactors(shape(hessian), tuple(map(shape, penalties)))


def evaluate_gaussian_working_model_proposal(
    parameters: np.ndarray,
    gram: np.ndarray,
    rhs: np.ndarray,
    response_square_sum: float,
    metadata: WorkingModelCriterionMetadata,
    *,
    scale_is_fixed: bool = False,
    log_scale: float = 0.0,
    tolerance: float = 0.0,
    smoothing: np.ndarray | None = None,
    debug: bool = False,
) -> GaussianWorkingModelProposal:
    """Evaluate one trial: assemble, factor on the host, then propose.

    ``smoothing`` defaults to :func:`source_exponentials` of the trial's
    log smoothing parameters; a matched-state caller may supply R's own.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    if smoothing is None:
        smoothing = source_exponentials(
            parameters if scale_is_fixed else parameters[:-1]
        )
    smoothing = np.asarray(smoothing, dtype=np.float64)
    factors = factor_working_model(
        working_model_factor_inputs(gram, metadata, smoothing)
    )
    return gaussian_working_model_proposal(
        parameters,
        gram,
        rhs,
        response_square_sum,
        metadata,
        factors,
        smoothing=smoothing,
        scale_is_fixed=scale_is_fixed,
        log_scale=log_scale,
        tolerance=tolerance,
        debug=debug,
    )


def _prospective_memory(
    structure: PenaltyStructure,
    control: FitControl,
    determinant_structure: PenaltyStructure | None = None,
) -> tuple[int, int, int, int]:
    """Bound shared metadata construction before predictors/device transfers."""
    host = 0
    device = 0
    components = 0
    for block in structure.blocks:
        count = len(block.local_penalties)
        components += count
        # Source Sl setup retains C, each padded local component, roots and
        # a root-product copy for the intermediate criterion. q bounds rank.
        # Host scratch covers symmetric eigensolver and one DPSTRF root at
        # a time; original caller-owned penalty storage is in shared setup.
        host += 8 * (2 * count + 7) * block.size**2 + 48 * block.size
        device += 8 * (count + 1) * block.size**2
        if count > 1:
            device += 8 * 2 * count * block.size**2
    if determinant_structure is not None:
        for block in determinant_structure.blocks:
            if len(block.local_penalties) > 1:
                # Source determinant blocks coexist with the fitting blocks
                # while their eigenspaces and roots are constructed.
                host += 8 * (2 * len(block.local_penalties) + 7) * block.size**2
    device += 8 * components
    n_parameters = structure.n_penalties + 1
    p = structure.n_coef
    statistics = 8 * (p**2 + p + 1 + n_parameters + 2)
    # Coefficient/gradient/step host ownership plus accepted/trial vectors,
    # and two parameter vectors/five scalars per accepted iteration. The
    # bounded trace stores accepted iterations only, not every halving trial.
    state = 8 * (2 * p + 8 * n_parameters)
    # Accepted factor and the next trial's owned factor can coexist. Each
    # owns one p-square upper array plus pivots and diagonal scaling.
    # Finite validation of the owned upper factor also creates one p-square
    # boolean mask. Rejected trial arrays are released before the next call.
    state += 17 * p**2 + 32 * p + 32
    state += 9 * n_parameters**2  # Owned Hessian plus finite-validation bitmap.
    state += 8 * control.working_model_max_iterations * (2 * n_parameters + 5)
    return host, device, statistics, state


def advance_gaussian_working_model(
    state: WorkingModelState,
    propose: Callable[[np.ndarray, float], GaussianWorkingModelProposal],
    rhs: np.ndarray,
    response_square_sum: float,
    n_obs: int,
    null_dimension: int,
    epsilon: float,
) -> WorkingModelAdvance:
    """Port one bgam.fitd additive-Gaussian outer iteration from ``state``.

    ``propose(trial, tolerance)`` evaluates the compiled Sl.fitChol proposal
    on the frozen statistics whose right-hand side is ``rhs``. The source
    convergence test precedes any proposal; a trial is accepted at once on a
    zero carried step, otherwise after directional-gradient halving.
    """
    iteration = state.completed + 1
    previous_deviance = state.deviance
    deviance = (
        state.deviance
        if iteration == 1
        else response_square_sum - float(state.coefficients @ rhs)
    )
    if not np.isfinite(deviance):
        raise FloatingPointError("Non-finite working-model penalized deviance.")
    if (
        iteration > 2
        and abs(deviance - previous_deviance) / (0.1 + abs(deviance)) < epsilon
        and abs(state.step[-1]) < epsilon * (abs(state.parameters[-1]) + 1.0)
    ):
        return WorkingModelAdvance(state, True, deviance, None, 0, None)
    next_step = state.step.copy()
    trial_evaluations = 0
    while True:
        trial = state.parameters + next_step
        tolerance = abs(state.criterion) * np.sqrt(np.finfo(float).eps)
        proposal = propose(trial, tolerance)
        criterion = float(proposal.criterion)
        trial_beta = np.asarray(proposal.coefficients).copy()
        gradient = np.asarray(proposal.gradient).copy()
        hessian = np.asarray(proposal.hessian).copy()
        proposed_step = np.asarray(proposal.step).copy()
        trial_upper = np.asarray(proposal.factor_upper).copy()
        trial_pivots = np.asarray(proposal.factor_pivots).copy()
        trial_rank = int(proposal.factor_rank)
        trial_diagonal = np.asarray(proposal.factor_diagonal).copy()
        trial_logdet_hessian = float(proposal.logdet_hessian)
        trial_logdet_penalty = float(proposal.logdet_penalty)
        for factor_array in (trial_upper, trial_pivots, trial_diagonal):
            factor_array.setflags(write=False)
        del proposal  # Do not retain old device outputs during the next call.
        trial_evaluations += 1
        if (
            not np.isfinite(criterion)
            or not np.all(np.isfinite(trial_beta))
            or not np.all(np.isfinite(gradient))
            or not np.all(np.isfinite(hessian))
            or not np.all(np.isfinite(proposed_step))
            or not np.all(np.isfinite(trial_upper))
            or not np.all(np.isfinite(trial_diagonal))
            or not np.isfinite(trial_logdet_hessian)
            or not np.isfinite(trial_logdet_penalty)
            or not 0 <= trial_rank <= len(trial_beta)
        ):
            raise FloatingPointError("Non-finite Gaussian working-model proposal.")
        del hessian
        # Preserve max(Nstep)==0 exactly, including source state timing.
        if np.max(next_step) == 0.0 or gradient @ next_step <= deviance * 1e-7:
            break
        next_step *= 0.5
        del trial_beta, gradient, proposed_step
        del trial_upper, trial_pivots, trial_diagonal
    phi = float(np.exp(trial[-1]))
    penalized = response_square_sum - float(trial_beta @ rhs)
    normalization = (n_obs - null_dimension) * np.log(2 * np.pi * phi) / 2
    # bgam.fitd's control criterion uses deviance BEFORE this proposal.
    criterion_control = criterion + (deviance - penalized) / (2 * phi) - normalization
    record = WorkingModelIteration(
        iteration,
        trial.copy(),
        proposed_step.copy(),
        deviance,
        penalized,
        criterion_control,
        trial_evaluations,
    )
    return WorkingModelAdvance(
        WorkingModelState(
            iteration,
            trial,
            proposed_step,
            trial_beta,
            deviance,
            criterion_control,
        ),
        False,
        deviance,
        record,
        trial_evaluations,
        AcceptedWorkingModelFactor(
            trial_upper,
            trial_pivots,
            trial_rank,
            trial_diagonal,
            trial_logdet_hessian,
            trial_logdet_penalty,
        ),
    )


def fit_gaussian_working_model_statistics(
    gram: np.ndarray,
    rhs: np.ndarray,
    response_square_sum: float,
    response_variance: float,
    penalties: PenaltyStructure,
    n_obs: int,
    *,
    determinant_penalties: PenaltyStructure | None = None,
    control: FitControl,
    available_workspace_bytes: int,
    device: jax.Device | None = None,
    initial_parameters: np.ndarray | None = None,
) -> GaussianWorkingModelFit:
    """Port bgam.fitd's Gaussian finite smoothing/scale sequence.

    The available budget is the remainder after retained setup, descriptor
    action peak and the shared Gaussian host/row solver ledger. Compilation
    precedes numerical execution so measured executable scratch is checked
    before proposals allocate it. Native compiler/allocator RSS is outside
    this array-storage estimate. Input backing datasets are caller owned.
    ``initial_parameters`` ports bgam.fitd's ``in.out`` start (log smoothing
    parameters, then log scale); by default the source initial.sp and
    ``var(y)*.05`` start is used.
    """
    p = penalties.n_coef
    if gram.shape != (p, p) or rhs.shape != (p,):
        raise ValueError("Working-model statistics must match coefficient dimension.")
    if (
        not np.all(np.isfinite(gram))
        or not np.all(np.isfinite(rhs))
        or not np.isfinite(response_square_sum)
        or not np.isfinite(response_variance)
        or response_variance <= 0.0
    ):
        raise FloatingPointError(
            "Working-model statistics and variance must be finite."
        )
    if initial_parameters is None:
        initial_parameters = np.r_[
            initial_log_sp_from_diagonal(np.diag(gram), penalties),
            np.log(response_variance * 0.05),
        ]
    state = WorkingModelState.initial(initial_parameters, p)
    if state.parameters.shape != (penalties.n_penalties + 1,):
        raise ValueError("Working-model start needs log(sp) and log(scale).")
    prospective = _prospective_memory(penalties, control, determinant_penalties)
    preliminary = sum(prospective)
    if preliminary > available_workspace_bytes:
        raise MemoryError(
            f"Working-model arguments require {preliminary} additional bytes; "
            f"available workspace is {available_workspace_bytes}."
        )
    metadata = WorkingModelCriterionMetadata.from_structure(
        penalties,
        n_obs,
        device,
        determinant_structure=determinant_penalties,
    )
    gram_device = jax.device_put(gram, device)
    rhs_device = jax.device_put(rhs, device)
    yy_device = jax.device_put(np.asarray(response_square_sum), device)
    parameter_shape = jax.ShapeDtypeStruct(state.parameters.shape, np.float64)
    smoothing_shape = jax.ShapeDtypeStruct(state.parameters[:-1].shape, np.float64)
    tolerance_shape = jax.ShapeDtypeStruct((), np.float64)
    factor_executable = working_model_factor_inputs.lower(
        gram_device, metadata, smoothing_shape
    ).compile()
    executable = gaussian_working_model_proposal.lower(
        parameter_shape,
        gram_device,
        rhs_device,
        yy_device,
        metadata,
        _factor_shapes(metadata),
        smoothing=smoothing_shape,
        tolerance=tolerance_shape,
    ).compile()
    analyses = (factor_executable.memory_analysis(), executable.memory_analysis())
    if any(analysis is None for analysis in analyses):
        raise NotImplementedError(
            "Working-model backend cannot report compiled memory."
        )
    temporary = sum(analysis.temp_size_in_bytes for analysis in analyses)
    output = sum(analysis.output_size_in_bytes for analysis in analyses)
    alias = sum(analysis.alias_size_in_bytes for analysis in analyses)
    # Per factored matrix: its host copy, the DPSTRF workspace (which owns the
    # returned factor) and the factor's device copy for the proposal.
    factor_workspace = sum(
        8 * n * n + source_pivoted_cholesky_workspace_bound(n) + 8 * n * n + 12 * n + 4
        for n in _factor_sizes(metadata)
    )
    required = preliminary + temporary + output - alias + factor_workspace
    memory = WorkingModelMemoryPlan(
        *prospective, temporary, output, alias, factor_workspace, required
    )
    if required > available_workspace_bytes:
        raise MemoryError(
            f"Working-model compiled workspace requires {required} additional bytes; "
            f"available workspace is {available_workspace_bytes}."
        )

    def propose(trial: np.ndarray, tolerance: float) -> GaussianWorkingModelProposal:
        smoothing = source_exponentials(trial[:-1])
        inputs = factor_executable(gram_device, metadata, smoothing)
        factors = jax.device_put(factor_working_model(inputs), device)
        del inputs
        return executable(
            trial,
            gram_device,
            rhs_device,
            yy_device,
            metadata,
            factors,
            smoothing=smoothing,
            tolerance=np.asarray(tolerance),
        )

    converged = False
    evaluations = 0
    factor = None
    trace: list[WorkingModelIteration] = []
    while state.completed < control.working_model_max_iterations:
        advance = advance_gaussian_working_model(
            state,
            propose,
            rhs,
            response_square_sum,
            n_obs,
            metadata.null_dimension,
            control.working_model_epsilon,
        )
        evaluations += advance.proposal_evaluations
        if advance.converged:
            converged = True
            break
        state = advance.state
        factor = advance.factor
        trace.append(advance.iteration)
        del advance
    transforms = []
    for block in metadata.penalties.blocks:
        values = np.asarray(block.transform.values).copy()
        values.setflags(write=False)
        transforms.append((block.start, block.stop, "dense", values))
    beta = state.coefficients.copy()
    beta.setflags(write=False)
    log_smoothing = state.parameters[:-1].copy()
    log_smoothing.setflags(write=False)
    return GaussianWorkingModelFit(
        beta,
        log_smoothing,
        float(np.exp(state.parameters[-1])),
        converged,
        state.completed + converged,
        evaluations,
        tuple(trace),
        memory,
        factor.upper,
        factor.pivots,
        factor.rank,
        factor.diagonal,
        factor.logdet_hessian,
        factor.logdet_penalty,
        tuple(transforms),
        _source_penalty_quadratic(state.coefficients, log_smoothing, metadata),
    )
