"""Bounded continuous-discrete fitting: shared setup and additive Gaussian.

Every discrete fit shares :func:`_prepared_discrete` (compact setup, fitted
grid record and indexed descriptor) and returns a row-free
:class:`DiscreteFit`. Additive Gaussian models use a direct fixed-sp solve or
the frozen-statistic working-model controller here; Poisson, Binomial and
Gamma use pinned ``bgam.fitd``'s Fisher performance iteration in
:mod:`jaxgam.execution.discrete_regular`.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

import jax
import numpy as np
import numpy.typing as npt
import scipy.linalg as sla

from jaxgam.control import ContinuousDiscretizationControl, FitControl
from jaxgam.data.continuous_request import RequestCompressionSpec
from jaxgam.data.source import RowSource
from jaxgam.execution.regular_working_model import RegularWorkingModelIteration
from jaxgam.execution.working_model import (
    WorkingModelIteration,
    fit_gaussian_working_model_statistics,
    source_exp,
    source_log,
)
from jaxgam.fitting.discrete_ops import (
    DeviceDiscreteDesign,
    discrete_row_xwxd,
    discrete_row_xwxd_bytes,
    discrete_row_xwxd_cost,
    discrete_xbd,
    discrete_xwxd,
    discrete_xwxd_cost,
    discrete_xwyd,
    to_device_discrete_design,
)
from jaxgam.fitting.working_model import working_model_hessian_inverse
from jaxgam.formula.continuous_design import build_continuous_discrete_design
from jaxgam.formula.continuous_metadata import (
    ContinuousDiscretizationMetadata,
    build_discretization_metadata,
    build_request_compression,
)
from jaxgam.formula.continuous_setup import (
    ContinuousDiscreteSetup,
    ContinuousSetupUnavailable,
    build_continuous_discrete_setup,
)
from jaxgam.formula.discrete_design import (
    DiscreteDesignControl,
    DiscreteDesignUnavailable,
    DiscreteMemoryPlan,
    FrozenDiscreteDesign,
)
from jaxgam.formula.fitting_prepare import total_penalty_spaces
from jaxgam.formula.predict_matrix import PredictSpec
from jaxgam.formula.terms import FormulaSpec
from jaxgam.inference.predictor import PivotedQRFisherFactor
from jaxgam.penalties.structure import DenseLocalPenalty, PenaltyStructure
from jaxgam.smooths.by_variable import FactorBySmooth


@dataclass(frozen=True)
class DiscreteFitMemoryPlan:
    """Conservative combined retained/action/solver byte bound.

    ``fit_peak_bytes`` is the bound the discrete route checks against
    ``memory_budget_bytes``: the retained Phase-1 compact setup tables plus
    the fit workspace. :attr:`fit_workspace_bytes` excludes those retained
    setup tables, as the streamed routes' ``known_workspace_bytes`` excludes
    their prepared setup.
    """

    setup_retained_bytes: int
    descriptor_host_retained_bytes: int
    descriptor_device_peak_bytes: int
    solver_workspace_bytes: int
    fit_peak_bytes: int
    budget_bytes: int

    @property
    def fit_workspace_bytes(self) -> int:
        """Known live fit workspace: indexed design, device actions, solver."""
        return (
            self.descriptor_host_retained_bytes
            + self.descriptor_device_peak_bytes
            + self.solver_workspace_bytes
        )


@dataclass(frozen=True)
class DiscreteFit:
    """Row-free discrete-route fit state with its actual smoothing strategy.

    ``performance_iteration`` marks a regular family fitted by ``bgam.fitd``'s
    Fisher performance iteration (fixed or working-model smoothing), whose
    ``scale_iterations`` are its outer iterations and whose
    ``coefficient_halvings`` count the coefficient step halvings;
    ``log_theta`` is an extended family's final (fixed or estimated) theta.
    """

    coefficients: npt.NDArray[np.float64]
    deviance: float
    score: float
    scale: float
    edf_total: float
    smoothing_params: npt.NDArray[np.float64]
    fisher_factor: npt.NDArray[np.float64] | None
    covariance: npt.NDArray[np.float64] | None
    predict_spec: PredictSpec
    n_obs: int
    offset_was_nonzero: bool
    fitting_source_scans: int
    fitting_batches_scanned: int
    scale_iterations: int
    memory_plan: DiscreteFitMemoryPlan
    smoothing_strategy: str = "fixed"
    converged: bool = True
    proposal_evaluations: int = 0
    working_model_trace: tuple[
        WorkingModelIteration | RegularWorkingModelIteration, ...
    ] = ()
    fisher_qr_factor: PivotedQRFisherFactor | None = None
    fisher_transforms: tuple[tuple[int, int, str, np.ndarray], ...] = ()
    discretization: ContinuousDiscretizationMetadata | None = None
    request_compression: RequestCompressionSpec | None = None
    performance_iteration: bool = False
    coefficient_halvings: int = 0
    log_theta: float | None = None


def _reject_unsupported_formula(spec: FormulaSpec) -> None:
    """Refuse terms whose pinned discrete representation JaxGAM lacks.

    Pinned ``bam(discrete=TRUE)`` converts a multi-variable ``re`` smooth
    (``tensor.possible``) into a tensor product discretized one variable per
    margin, whereas the compact setup groups its variables jointly.
    """
    for term in spec.smooth_terms:
        if term.bs == "re" and len(term.variables) > 1:
            raise NotImplementedError(
                f"s({','.join(term.variables)}, bs='re') is not supported by "
                "execution='discrete': pinned mgcv bam discretizes each "
                "variable of a multi-variable random effect as a separate "
                "tensor margin, which the compact setup does not reproduce."
            )


def _fixed_sp(sp: npt.ArrayLike, n_penalties: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.atleast_1d(np.asarray(sp, dtype=np.float64))
    if values.ndim != 1 or len(values) != n_penalties:
        raise ValueError(
            f"sp has {values.size} elements but model has {n_penalties} penalty terms."
        )
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        bad = values[(values <= 0.0) | ~np.isfinite(values)]
        raise ValueError(
            "sp must contain strictly positive, finite values; "
            f"got invalid value(s) {bad.tolist()}."
        )
    return values, source_log(values)


def _solver_plan(
    setup_retained_bytes: int,
    descriptor: DiscreteMemoryPlan,
    n_coef: int,
    penalties: PenaltyStructure,
    control: FitControl,
) -> DiscreteFitMemoryPlan:
    # Bound the known arrays by their largest coefficient-space shape:
    # five retained solver matrices (G/S/H/L/H^-1G), one set of structural
    # range bases, two unscaled block/eigenvector arrays, six simultaneous
    # projection/root arrays (projection intermediate, projected component,
    # symmetric LAPACK input, LAPACK factor, triangular copy, unpivoted root),
    # the combined range penalty and one product/ownership copy. Roots are
    # combined sequentially, so this count does not multiply by penalties.
    matrix_count = 5 + 1 + 2 + 6 + 1 + 1
    # Covariance inversion, scaling and predictor ownership can coexist.
    matrix_count += 3 if control.uncertainty == "covariance" else 0
    # DenseLocalPenalty.dense() is a view; compact diagonal/identity penalties
    # need explicit local matrices while the block metadata is evaluated.
    local_materializations = sum(
        block.size * block.size
        for block in penalties.blocks
        for component in block.local_penalties
        if not isinstance(component, DenseLocalPenalty)
    )
    # At residual reduction peak: three input float64 vectors, source scan
    # positions plus RowBatch positions, scan ordinals, fitted/residual and
    # two weighted-product temporaries. The validity vector is boolean.
    # Descriptor selector staging is charged in its action peak separately.
    row_workspace = control.batch_rows * (
        10 * np.dtype(np.float64).itemsize + np.dtype(np.bool_).itemsize
    )
    solver = (
        np.dtype(np.float64).itemsize
        * (matrix_count * n_coef * n_coef + local_materializations + 8 * n_coef)
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


def _penalty_rank(structure: PenaltyStructure) -> int:
    return sum(basis.shape[1] for _, basis in total_penalty_spaces(structure)[1])


def _pinned_criterion_null_dimension(
    spec: PredictSpec, total_null_dimension: int
) -> int:
    """Match pinned mgcv 1.9-3 ``bgam.fitd``'s final ``Mp`` reset."""
    parametric = next(
        (
            term.n_coefs
            for term in spec.coef_map.terms
            if term.term_type == "parametric"
        ),
        0,
    )
    smooths = [
        term.smooth
        for term in spec.coef_map.terms
        if term.term_type == "smooth" and term.smooth is not None
    ]
    expanded_smooth_count = sum(
        smooth.n_levels if isinstance(smooth, FactorBySmooth) else 1
        for smooth in smooths
    )
    return total_null_dimension if expanded_smooth_count > 1 else parametric


def _log_pdet(structure: PenaltyStructure, log_smoothing: np.ndarray) -> float:
    """Evaluate ``log|S|+`` on setup-owned structural range spaces.

    The range of each penalty block is fixed by its unweighted component
    penalties.  Re-estimating rank from the *scaled* combined penalty can drop
    a valid direction when smoothing parameters differ greatly, changing the
    REML normalization.  This is the NumPy counterpart of ``Sl.setup`` plus
    ``ldetS`` in pinned mgcv.
    """
    log_determinant = 0.0
    structural_ranges = {
        id(block): basis for block, basis in total_penalty_spaces(structure)[1]
    }
    for block in structure.blocks:
        penalties = block.dense_penalties()
        range_basis = structural_ranges[id(block)]
        rank = range_basis.shape[1]
        if rank == 0:
            continue
        if len(penalties) == 1:
            eigenvalues = np.linalg.eigvalsh(penalties[0])[-rank:]
            if np.any(eigenvalues <= 0.0):
                raise FloatingPointError("Penalty range determinant is not positive.")
            log_determinant += float(
                rank * log_smoothing[block.sp_indices[0]]
                + np.sum(source_log(eigenvalues))
            )
            continue

        # ``Sl.setup`` projects an overlapping multi-penalty block through
        # the range of the *unscaled* component sum, then calls ``mroot``
        # with the combined block rank before ``ldetSblock`` combines roots.
        # Preserve that source policy, including its floating-point roots.
        # Accumulate in source order without stacking all component matrices.
        total = penalties[0].copy()
        for component in penalties[1:]:
            total += component
        _, vectors = sla.eigh(0.5 * (total + total.T), driver="evr", check_finite=False)
        # R's eigen() returns descending eigenvalues/vectors.
        source_range = vectors[:, -rank:][:, ::-1]
        block_log_smoothing = np.asarray(
            [log_smoothing[index] for index in block.sp_indices]
        )
        maximum = float(np.max(block_log_smoothing))
        combined = np.zeros((rank, rank))
        for value, component in zip(block_log_smoothing, penalties, strict=True):
            root = _pivoted_cholesky_root(
                source_range.T @ component @ source_range, rank
            )
            combined += source_exp(float(value - maximum)) * (root @ root.T)
        log_determinant += rank * maximum + _positive_logdet(combined, rank)
    return log_determinant


def _pivoted_cholesky_root(matrix: np.ndarray, requested_rank: int) -> np.ndarray:
    """Port the pivoted-Cholesky ``mroot`` layout used by pinned mgcv."""
    factor, pivots, rank, info = sla.lapack.dpstrf(
        0.5 * (matrix + matrix.T), lower=0, tol=0.0
    )
    if info < 0:
        raise FloatingPointError("Penalty root factorization failed.")
    factor = np.triu(factor)
    if rank < len(factor):
        factor[rank:, rank:] = 0.0
    inverse_pivots = np.argsort(pivots - 1)
    return factor[:requested_rank, :][:, inverse_pivots].T


def _positive_logdet(matrix: np.ndarray, structural_rank: int) -> float:
    """Match ``ldetSblock`` diagonal scaling and pivoted Cholesky."""
    diagonal = np.diag(matrix).copy()
    diagonal[diagonal <= 0.0] = 1.0
    scaling = np.sqrt(diagonal)
    scaled = (matrix / scaling[:, None]) / scaling[None, :]
    factor, pivots, rank, info = sla.lapack.dpstrf(0.5 * (scaled + scaled.T), lower=0)
    if info < 0 or rank != structural_rank:
        raise FloatingPointError(
            "Penalty determinant factorization disagrees with structural rank."
        )
    selected = pivots[:rank] - 1
    return float(
        2.0 * np.sum(source_log(np.diag(factor)[:rank]) + source_log(scaling[selected]))
    )


def _reml_score(
    *,
    deviance: float,
    beta: np.ndarray,
    penalty: np.ndarray,
    penalty_structure: PenaltyStructure,
    log_smoothing: np.ndarray,
    factor: np.ndarray,
    score_scale: float,
    n_obs: int,
    null_dimension: int,
) -> float:
    log_det_hessian = 2.0 * float(np.sum(source_log(np.diag(factor))))
    penalized = deviance + float(beta @ penalty @ beta)
    return float(
        penalized / (2.0 * score_scale)
        + log_det_hessian / 2.0
        - _log_pdet(penalty_structure, log_smoothing) / 2.0
        + (n_obs - null_dimension) * source_log(2.0 * np.pi * score_scale) / 2.0
    )


def _pinned_fixed_scale(
    penalized_deviance: float,
    n_obs: int,
    null_dimension: int,
    y_centered_square_sum: float,
) -> tuple[float, int]:
    """Port pinned ``bgam.fitd``'s damped log-scale Newton sequence."""
    denominator = n_obs - null_dimension
    if denominator <= 0 or penalized_deviance <= 0.0:
        raise FloatingPointError("Invalid fixed-sp REML score scale.")
    if n_obs < 2:
        raise FloatingPointError("Gaussian scale initialization needs two rows.")
    variance = y_centered_square_sum / (n_obs - 1)
    if not np.isfinite(variance) or variance <= 0.0:
        raise FloatingPointError("Gaussian scale initialization is not positive.")
    log_scale = float(source_log(variance * 0.05))
    step = 0.0
    deviance = penalized_deviance
    deviance_old = np.inf
    for iteration in range(1, 201):
        if (
            iteration > 2
            and abs(deviance - deviance_old) / (0.1 + abs(deviance)) < 1e-7
            and abs(step) < 1e-7 * (abs(log_scale) + 1.0)
        ):
            return float(source_exp(log_scale)), iteration - 1
        deviance_old = deviance
        previous_step = step
        while True:
            trial = log_scale + previous_step
            scale = float(source_exp(trial))
            gradient = (-penalized_deviance / scale + denominator) / 2.0
            proposed = 1.0 - denominator * scale / penalized_deviance
            if abs(proposed) > 4.0:
                proposed = float(np.copysign(4.0, proposed))
            if previous_step != 0.0 and gradient * previous_step > deviance * 1e-7:
                previous_step /= 2.0
                continue
            log_scale = trial
            step = proposed
            break
    raise RuntimeError("Pinned fixed-sp Gaussian scale iteration did not converge.")


class CenteredMoments:
    """Unweighted ``var(y)`` from batch moments merged around one origin.

    Centering on the first value keeps a large response location from
    cancelling a small variance, as raw ``y'y`` moments would.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.centered_square_sum = 0.0
        self.origin: float | None = None

    def add(self, values: np.ndarray) -> None:
        if self.origin is None:
            self.origin = float(values[0])
        centered = values - self.origin
        batch_mean = float(np.mean(centered))
        batch_square_sum = float(np.sum((centered - batch_mean) ** 2))
        combined = self.count + len(values)
        delta = batch_mean - self.mean
        self.centered_square_sum += (
            batch_square_sum + delta * delta * self.count * len(values) / combined
        )
        self.mean += delta * len(values) / combined
        self.count = combined

    def variance(self) -> float:
        if self.count < 2:
            raise FloatingPointError("Scale initialization needs two rows.")
        return self.centered_square_sum / (self.count - 1)


@dataclass(frozen=True)
class _PreparedDiscrete:
    """Frozen compact setup, fitted grid record and indexed descriptor."""

    setup: ContinuousDiscreteSetup
    discretization: ContinuousDiscretizationMetadata
    request_compression: RequestCompressionSpec | None
    descriptor: FrozenDiscreteDesign
    penalties: PenaltyStructure
    fingerprint: str


@contextmanager
def _prepared_discrete(
    spec: FormulaSpec, source: RowSource, control: FitControl
) -> Iterator[_PreparedDiscrete]:
    """Build the bounded compact problem shared by every discrete fit.

    The result owns only frozen prediction metadata. Compact selectors,
    including any memmap files, are fitting-time resources released on exit.
    """
    policy = control.discretization
    if control.execution != "discrete" or not isinstance(
        policy, ContinuousDiscretizationControl
    ):
        raise ValueError("discrete fitting requires its explicit policy")
    _reject_unsupported_formula(spec)
    fingerprint = source.fingerprint()
    setup = build_continuous_discrete_setup(
        spec,
        source,
        factor_names=policy.factor_names,
        m=policy.m,
        budget=policy.budget,
        selector_memmap_dir=policy.selector_memmap_directory,
        batch_rows=control.batch_rows,
        seed=policy.seed,
    )
    if isinstance(setup, ContinuousSetupUnavailable):
        raise MemoryError(
            f"{setup.reason}: requires {setup.required_bytes} bytes, "
            f"budget is {setup.budget_bytes}."
        )
    try:
        # Bounded fitted record and prediction grouping, copied before the
        # compact selectors are released; neither retains any row data.
        discretization = build_discretization_metadata(
            setup,
            prediction_policy=policy.prediction_policy,
            max_retained_representatives=policy.max_retained_representatives,
            shuffle_seed=policy.seed,
        )
        request_compression = (
            build_request_compression(setup, policy.budget)
            if discretization.prediction_policy == "mgcv_predict_bamd"
            else None
        )
        descriptor = build_continuous_discrete_design(
            setup,
            source,
            control=DiscreteDesignControl(
                budget=policy.budget,
                batch_rows=control.batch_rows,
                selector_memmap_directory=policy.selector_memmap_directory,
                pair_policy=policy.pair_policy,
                max_pair_histogram_bytes=policy.max_pair_histogram_bytes,
            ),
        )
        if isinstance(descriptor, DiscreteDesignUnavailable):
            raise MemoryError(
                f"{descriptor.reason}: requires {descriptor.required_bytes} bytes, "
                f"budget is {descriptor.budget_bytes}."
            )
        assert descriptor.memory_plan is not None
        yield _PreparedDiscrete(
            setup,
            discretization,
            request_compression,
            descriptor,
            setup.penalties or PenaltyStructure(descriptor.n_coef, ()),
            fingerprint,
        )
    finally:
        setup.cleanup()


_ROW_XWXD_CHUNKS = (2048, 1024, 512, 256, 128, 64)


def _row_xwxd_chunk(design: DeviceDiscreteDesign, available_bytes: int) -> int | None:
    """Rows per chunk for :func:`discrete_row_xwxd`, or None for pair reductions.

    Materialized row chunks are used when their estimated cost undercuts the
    pair reductions' (:func:`discrete_xwxd_cost`): many smooths against few
    coefficients. A few wide blocks, such as a random effect with many levels,
    keep the pair reductions. The largest chunk whose scratch fits the
    remaining budget is chosen.
    """
    n_coef = design.n_coef
    if discrete_row_xwxd_cost(n_coef, design.batch_rows) >= discrete_xwxd_cost(
        design, design.batch_rows
    ):
        return None
    for chunk_rows in _ROW_XWXD_CHUNKS:
        if discrete_row_xwxd_bytes(n_coef, chunk_rows) <= available_bytes:
            return chunk_rows
    return None


def _check_fit_budget(memory: DiscreteFitMemoryPlan) -> None:
    if memory.fit_peak_bytes > memory.budget_bytes:
        raise MemoryError(
            "Discrete fit workspace requires "
            f"{memory.fit_peak_bytes} bytes, exceeding "
            f"FitControl.memory_budget_bytes={memory.budget_bytes}."
        )


def fit_fixed_sp_discrete_gaussian(
    spec: FormulaSpec,
    source: RowSource,
    sp: npt.ArrayLike,
    *,
    control: FitControl,
    device: object | None = None,
) -> DiscreteFit:
    """Fit an explicit fixed-sp Gaussian model on the compact design."""
    return fit_discrete_gaussian(spec, source, sp, control=control, device=device)


def fit_discrete_gaussian(
    spec: FormulaSpec,
    source: RowSource,
    sp: npt.ArrayLike | None,
    *,
    control: FitControl,
    device: object | None = None,
) -> DiscreteFit:
    """Fit one Gaussian identity model using bounded indexed actions.

    Two fitting scans are intentional. The first forms sufficient statistics;
    the second accumulates selected-row residuals directly, avoiding unstable
    cancellation for nearly exact fits.
    """
    working_model = sp is None
    if working_model and control.smoothing_strategy != "working_model_reml":
        raise NotImplementedError("Discrete estimated sp requires working_model_reml.")
    with _prepared_discrete(spec, source, control) as problem:
        setup = problem.setup
        descriptor = problem.descriptor
        penalties = problem.penalties
        fingerprint = problem.fingerprint
        memory = _solver_plan(
            setup.retained_table_bytes,
            descriptor.memory_plan,
            descriptor.n_coef,
            penalties,
            control,
        )
        _check_fit_budget(memory)
        if not working_model:
            smoothing, log_smoothing = _fixed_sp(sp, penalties.n_penalties)
            penalty = penalties.materialize(log_smoothing)
        device_descriptor = to_device_discrete_design(descriptor, device)
        # The row kernel replaces the pair kernel during the statistics scan
        # only, and its scratch fits the budget left after the fit plan, so
        # the scan's peak stays within the budget and later stages are unchanged.
        chunk_rows = _row_xwxd_chunk(
            device_descriptor, memory.budget_bytes - memory.fit_peak_bytes
        )
        gram = np.zeros((descriptor.n_coef, descriptor.n_coef))
        rhs = np.zeros(descriptor.n_coef)
        cursor = 0
        batches = 0
        moments = CenteredMoments()
        response_square_sum = 0.0
        offset_nonzero = False
        for batch in source.scan(control.batch_rows):
            size = len(batch.row_positions)
            if size == 0 or cursor + size > source.n_rows:
                raise RuntimeError("RowSource yielded an invalid fitting row count.")
            if batch.y is None:
                raise ValueError("Discrete fitting requires a RowSource response.")
            valid = np.asarray(batch.valid)
            y = np.asarray(batch.y, dtype=np.float64)
            weights = np.asarray(batch.weight, dtype=np.float64)
            offset = np.asarray(batch.offset, dtype=np.float64)
            if not np.all(valid):
                raise NotImplementedError(
                    "Discrete fitting does not yet support padded invalid rows."
                )
            if (
                not np.all(np.isfinite(y))
                or not np.all(np.isfinite(weights))
                or not np.all(np.isfinite(offset))
                or np.any(weights < 0.0)
            ):
                raise ValueError(
                    "Discrete fitting inputs must be finite with nonnegative weights."
                )
            rows = np.arange(cursor, cursor + size, dtype=np.intp)
            selectors = descriptor.batch_indices(rows)
            gram += np.asarray(
                discrete_xwxd(device_descriptor, selectors, weights)
                if chunk_rows is None
                else discrete_row_xwxd(
                    device_descriptor, selectors, weights, chunk_rows=chunk_rows
                )
            )
            rhs += np.asarray(
                discrete_xwyd(device_descriptor, selectors, weights, y - offset)
            )
            if working_model:
                response_square_sum += float(np.sum(weights * (y - offset) ** 2))
            # R initializes log(phi) from unweighted var(y).
            moments.add(y)
            offset_nonzero |= bool(np.any(offset != 0.0))
            cursor += size
            batches += 1
            # A finished fitting batch must not retain its response or source
            # columns while the next batch (or residual scan) is constructed.
            del batch, valid, y, weights, offset, rows, selectors
        if cursor != source.n_rows:
            raise RuntimeError("RowSource yielded fewer fitting rows than declared.")
        if source.fingerprint() != fingerprint:
            raise RuntimeError("RowSource changed during discrete fitting.")
        selected = None
        if working_model:
            if source.n_rows < 2:
                raise FloatingPointError(
                    "Gaussian scale initialization needs two rows."
                )
            selected = fit_gaussian_working_model_statistics(
                gram,
                rhs,
                response_square_sum,
                moments.centered_square_sum / (source.n_rows - 1),
                penalties,
                source.n_rows,
                determinant_penalties=setup.source_determinant_penalties,
                control=control,
                available_workspace_bytes=memory.budget_bytes - memory.fit_peak_bytes,
                device=device,
            )
            extra = selected.memory_plan.additional_peak_bytes
            memory = replace(
                memory,
                solver_workspace_bytes=memory.solver_workspace_bytes + extra,
                fit_peak_bytes=memory.fit_peak_bytes + extra,
            )
            log_smoothing = selected.log_smoothing
            smoothing = source_exp(log_smoothing)
            penalty = penalties.materialize(log_smoothing)
            if not selected.converged:
                warnings.warn(
                    "Gaussian working-model REML did not converge "
                    "within its iteration limit.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        qr_factor = None
        transforms = ()
        if selected is not None:
            selected_pivots = selected.factor_pivots[: selected.factor_rank]
            qr_factor = PivotedQRFisherFactor(
                R=selected.factor_upper[: selected.factor_rank, : selected.factor_rank]
                * selected.factor_diagonal[selected_pivots][None, :],
                pivots=np.arange(selected.factor_rank, dtype=np.intp),
                keep=selected_pivots,
                original_n_coef=descriptor.n_coef,
            )
            inverse = np.array(
                working_model_hessian_inverse(
                    *jax.device_put(
                        (
                            selected.factor_upper,
                            selected.factor_pivots,
                            selected.factor_diagonal,
                            np.int32(selected.factor_rank),
                        ),
                        device,
                    )
                )
            )
            transforms = selected.coefficient_transforms
            for start, stop, _, values in transforms:
                inverse[start:stop, :] = values @ inverse[start:stop, :]
                inverse[:, start:stop] = inverse[:, start:stop] @ values.T
            factor = None
        else:
            hessian = 0.5 * (gram + gram.T) + penalty
            try:
                factor = np.linalg.cholesky(hessian)
            except np.linalg.LinAlgError as error:
                raise np.linalg.LinAlgError(
                    "Discrete fixed-sp Hessian is not positive definite; "
                    "rank-deficient QR fitting is not implemented for this route."
                ) from error
        beta = (
            selected.coefficients
            if selected is not None
            else sla.cho_solve((factor, True), rhs)
        )

        # Direct residual scan is stable for exact and nearly exact fits.
        deviance = 0.0
        residual_cursor = 0
        residual_batches = 0
        for batch in source.scan(control.batch_rows):
            size = len(batch.row_positions)
            if size == 0 or residual_cursor + size > source.n_rows:
                raise RuntimeError("RowSource yielded an invalid residual row count.")
            if batch.y is None or not np.all(np.asarray(batch.valid)):
                raise RuntimeError(
                    "RowSource response/validity changed during fitting."
                )
            rows = np.arange(residual_cursor, residual_cursor + size, dtype=np.intp)
            selectors = descriptor.batch_indices(rows)
            fitted = np.asarray(discrete_xbd(device_descriptor, selectors, beta))
            residual = np.asarray(batch.y, dtype=np.float64) - (
                fitted + np.asarray(batch.offset, dtype=np.float64)
            )
            deviance += float(np.sum(np.asarray(batch.weight) * residual * residual))
            residual_cursor += size
            residual_batches += 1
            del batch, rows, selectors, fitted, residual
        if residual_cursor != source.n_rows:
            raise RuntimeError("RowSource yielded fewer residual rows than declared.")
        if source.fingerprint() != fingerprint:
            raise RuntimeError("RowSource changed during discrete residual evaluation.")

        # tr(H^-1 X'WX) needs only the elementwise product, not the p^3 matrix
        # product; the fixed-sp route still solves against X'WX.
        edf = float(
            np.sum(inverse * gram.T)
            if selected is not None
            else np.trace(sla.cho_solve((factor, True), gram))
        )
        rank = _penalty_rank(penalties)
        null_dimension = descriptor.n_coef - rank
        if selected is None:
            penalized = deviance + float(beta @ penalty @ beta)
            score_scale, scale_iterations = _pinned_fixed_scale(
                penalized,
                source.n_rows,
                null_dimension,
                moments.centered_square_sum,
            )
        else:
            # bgam.fitd's criterion penalty lives in Sl.setup coordinates;
            # an original-coordinate b'Sb cancels when one sp is very large.
            penalized = deviance + selected.penalty_quadratic
            score_scale, scale_iterations = selected.scale, selected.n_iter
        score = (
            _reml_score(
                deviance=deviance,
                beta=beta,
                penalty=penalty,
                penalty_structure=penalties,
                log_smoothing=log_smoothing,
                factor=factor,
                score_scale=score_scale,
                n_obs=source.n_rows,
                null_dimension=_pinned_criterion_null_dimension(
                    setup.predict_spec, null_dimension
                ),
            )
            if selected is None
            else float(
                penalized / (2.0 * score_scale)
                + (selected.logdet_hessian - selected.logdet_penalty) / 2.0
                + (
                    source.n_rows
                    - _pinned_criterion_null_dimension(
                        setup.predict_spec, null_dimension
                    )
                )
                * source_log(2.0 * np.pi * score_scale)
                / 2.0
            )
        )
        if not np.isfinite(score) or not np.isfinite(edf):
            raise FloatingPointError("Unusable Gaussian working-model score or EDF.")
        covariance = None
        fisher = None
        if control.uncertainty == "fisher":
            fisher = factor
        elif control.uncertainty == "covariance":
            covariance = score_scale * (
                inverse
                if selected is not None
                else sla.cho_solve((factor, True), np.eye(descriptor.n_coef))
            )
        if control.uncertainty != "fisher":
            qr_factor = None
            transforms = ()
        return DiscreteFit(
            np.asarray(beta),
            deviance,
            score,
            score_scale,
            edf,
            smoothing,
            fisher,
            covariance,
            setup.predict_spec,
            source.n_rows,
            offset_nonzero,
            2,
            batches + residual_batches,
            scale_iterations,
            memory,
            "working_model_reml" if working_model else "fixed",
            selected.converged if selected is not None else True,
            selected.proposal_evaluations if selected is not None else 0,
            selected.trace if selected is not None else (),
            qr_factor,
            transforms,
            problem.discretization,
            problem.request_compression,
        )
