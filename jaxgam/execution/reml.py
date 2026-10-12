"""Host replay driver for exact known-scale streamed REML gradients."""

from __future__ import annotations

import numbers
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from jaxgam.families.base import ExponentialFamily
from jaxgam.families.standard import Binomial, Poisson
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.fitting.family_execution import (
    FamilyExecutionLineage,
    FamilyExecutionParameters,
)
from jaxgam.fitting.state import StreamFitState
from jaxgam.fitting.stream_reml import (
    batch_is_adjoint_interior,
    batch_statistics_beta_vjp,
    regular_batch_statistics_vjp,
    regular_score_cotangents,
    reml_score_cotangents,
    solve_observed_adjoint,
)
from jaxgam.formula.design_provider import StreamDesign

from .regular_stream import (
    RegularStreamResult,
    RegularStreamWorkspace,
    fit_regular_streamed_pirls,
    preflight_regular_stream_workspace,
)
from .stream import StreamPIRLSControl, _fitting_batches, fit_streamed_pirls


@dataclass(frozen=True)
class StreamREMLTrial:
    """Immutable exact-rho result of one fully reconverged stream trial."""

    rho: jax.Array
    score: jax.Array
    gradient: jax.Array
    fit_state: StreamFitState
    adjoint_jitter: jax.Array
    source_scans: int
    batches_scanned: int
    source_fingerprint: str
    basis_fingerprint: str
    family_name: str
    link_name: str
    derivative_elapsed_s: float = 0.0
    final_refit_solves: int = 0


@dataclass(frozen=True)
class RegularREMLWorkspace:
    """Regular controller ledger plus live exact-gradient arrays."""

    controller: RegularStreamWorkspace
    adjoint_bytes: int

    @property
    def required_bytes(self) -> int:
        return self.controller.required_bytes + self.adjoint_bytes


@dataclass(frozen=True)
class RegularStreamREMLTrial:
    """One source-timed regular score and exact first derivative.

    ``log_phi_curvature`` is the exact second derivative of the score in
    free log scale at fixed rho (gam.fit3's ``d2lr.d2lphi``), or ``None``
    for known-scale families.
    """

    params: jax.Array
    score: jax.Array
    gradient: jax.Array
    fit_result: RegularStreamResult
    workspace: RegularREMLWorkspace
    source_factor_residual: float
    observed_factor_residual: float
    alpha_roundoff_recovered: bool
    source_scans: int
    batches_scanned: int
    source_fingerprint: str
    basis_fingerprint: str
    family_name: str
    link_name: str
    log_phi_curvature: float | None = None
    derivative_elapsed_s: float = 0.0
    final_refit_solves: int = 0


@dataclass(frozen=True)
class StreamREMLControl:
    """Bounded host controls for the internal streamed REML optimizer.

    ``maxcor`` bounds L-BFGS-B's correction history.  The driver retains at
    most two coefficient-space trials (the current accepted and candidate
    trial) and a bounded sequence of accepted scalar scores; it deliberately
    does not retain an outer Hessian or an array of trial ``rho`` vectors.
    """

    max_iter: int = 100
    maxfun: int = 500
    maxcor: int = 10
    maxls: int = 20
    gtol: float = 1e-6
    ftol: float = 1e-12
    history_size: int = 32

    def __post_init__(self) -> None:
        for name, value, allow_zero in (
            ("max_iter", self.max_iter, False),
            ("maxfun", self.maxfun, False),
            ("maxcor", self.maxcor, False),
            ("maxls", self.maxls, False),
            ("history_size", self.history_size, True),
        ):
            if (
                not isinstance(value, numbers.Integral)
                or isinstance(value, bool)
                or value < 0
                or (not allow_zero and value == 0)
            ):
                raise ValueError(f"{name} must be an integer in its valid range.")
        for name, value, allow_zero in (
            ("gtol", self.gtol, False),
            ("ftol", self.ftol, True),
        ):
            if (
                not isinstance(value, numbers.Real)
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value < 0
                or (not allow_zero and value == 0)
            ):
                raise ValueError(f"{name} must be a finite value in its valid range.")


@dataclass(frozen=True)
class StreamREMLOptimization:
    """Diagnostic outcome of bounded streamed L-BFGS-B optimization."""

    trial: StreamREMLTrial
    converged: bool
    message: str
    status: int
    n_iter: int
    n_evaluations: int
    n_accepted: int
    cumulative_source_scans: int
    cumulative_batches_scanned: int
    projected_gradient_inf: float
    accepted_score_history: tuple[float, ...]
    cumulative_inner_iterations: int = 0
    cumulative_final_refit_solves: int = 0
    cumulative_derivative_s: float = 0.0


@dataclass(frozen=True)
class RegularStreamREMLOptimization:
    """Diagnostic outcome of joint regular streamed L-BFGS-B optimization.

    ``status`` is the L-BFGS-B status and ``n_iter`` counts its iterations
    plus any accepted free-parameter completion steps; ``converged`` is
    restated on the returned trial. When no smoothing parameter is free
    (every one pinned, or none exists) and the scale is unknown,
    ``scale_completion_steps`` counts the accepted exact-curvature log-scale
    Newton steps that follow it (not included in ``n_iter``) and
    ``scale_completion_stop`` records why that completion ended. With free
    smoothing parameters,
    ``curvature_completion_steps``/``curvature_completion_stop`` describe the
    free-parameter Newton completion after an L-BFGS-B roundoff stop
    (``None`` when that completion was not admitted).
    """

    trial: RegularStreamREMLTrial
    converged: bool
    message: str
    status: int
    n_iter: int
    n_evaluations: int
    n_accepted: int
    cumulative_source_scans: int
    cumulative_batches_scanned: int
    projected_gradient_inf: float
    accepted_score_history: tuple[float, ...]
    outer_workspace_bytes: int
    scale_completion_steps: int = 0
    scale_completion_stop: str | None = None
    cumulative_inner_iterations: int = 0
    cumulative_final_refit_solves: int = 0
    cumulative_derivative_s: float = 0.0
    curvature_completion_steps: int = 0
    curvature_completion_stop: str | None = None

    @property
    def required_bytes(self) -> int:
        """Known trial plus retained optimizer workspace at the peak."""
        return self.trial.workspace.required_bytes + self.outer_workspace_bytes


class _ParameterizedTrial(Protocol):
    """Structural contract shared by exact regular and NB trial states."""

    params: jax.Array
    score: jax.Array
    gradient: jax.Array
    source_scans: int
    batches_scanned: int
    source_fingerprint: str
    derivative_elapsed_s: float
    final_refit_solves: int


_TrialT = TypeVar("_TrialT", bound=_ParameterizedTrial)


@dataclass(frozen=True)
class _ParameterizedOptimization(Generic[_TrialT]):
    """Common bounded L-BFGS-B outcome before family-specific wrapping."""

    trial: _TrialT
    converged: bool
    message: str
    status: int
    n_iter: int
    n_evaluations: int
    n_accepted: int
    cumulative_source_scans: int
    cumulative_batches_scanned: int
    projected_gradient_inf: float
    accepted_score_history: tuple[float, ...]
    cumulative_inner_iterations: int = 0
    cumulative_final_refit_solves: int = 0
    cumulative_derivative_s: float = 0.0
    completion_steps: int = 0
    completion_stop: str | None = None


@dataclass(frozen=True)
class _LocalScoreChange:
    """Independent small-step comparison; the trial's raw score is retained."""

    change: float
    error_estimate: float
    raw_change: float
    roundoff_bound: float
    linearity_error: float


def _preflight(
    stream: StreamDesign,
    family: ExponentialFamily,
    rho: np.ndarray | jax.Array,
    warm_start: StreamREMLTrial | None,
    device: jax.Device | None,
) -> tuple[jax.Array, PreparedFittingMetadata, FamilyExecutionLineage]:
    """Reject derivative regimes not covered by the initial exact proof."""
    if not isinstance(family, (Poisson, Binomial)) or not family.is_canonical:
        raise NotImplementedError(
            "Streamed REML adjoints initially support canonical Poisson log and "
            "Binomial logit only; unknown dispersion, noncanonical links, and "
            "extended families are not implemented."
        )
    if family.n_theta or not family.scale_known:
        raise NotImplementedError(
            "Streamed REML adjoints initially require known dispersion and no theta."
        )
    fitting = stream.prepared.fitting
    if fitting is None:
        raise ValueError("Streamed REML adjoints require fitting preparation.")
    if (
        fitting.family_name != family.family_name
        or fitting.link_name != type(family.link).__qualname__
    ):
        raise ValueError(
            "Prepared fitting metadata belongs to a different family or link."
        )
    metadata = PreparedFittingMetadata.from_prepared(stream.prepared, family, device)
    if metadata.rank_deficit:
        raise np.linalg.LinAlgError(
            "Streamed REML adjoints require a fixed full-rank identifiable subspace."
        )
    if stream.source.fingerprint() != stream.prepared.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again.")
    lineage = FamilyExecutionLineage.from_prepared(stream.prepared, family)
    rho_array = jnp.asarray(rho, dtype=jnp.float64)
    if rho_array.shape != (metadata.n_penalties,):
        raise ValueError(
            f"rho must have shape ({metadata.n_penalties},), got {rho_array.shape}."
        )
    if not np.all(np.isfinite(np.asarray(rho_array))):
        raise ValueError("rho must contain only finite log smoothing parameters.")
    if warm_start is not None and (
        not warm_start.fit_state.converged
        or warm_start.rho.shape != rho_array.shape
        or not np.all(np.isfinite(np.asarray(warm_start.fit_state.coefficients)))
        or warm_start.source_fingerprint != stream.prepared.source_fingerprint
        or warm_start.basis_fingerprint != stream.prepared.basis_fingerprint
        or warm_start.family_name != family.family_name
        or warm_start.link_name != type(family.link).__qualname__
    ):
        raise ValueError("Warm-start trial is not a finite compatible stream state.")
    return rho_array, metadata, lineage


def _preflight_regular_reml(
    stream: StreamDesign,
    family: ExponentialFamily,
    params: np.ndarray | jax.Array,
    control: StreamPIRLSControl,
    maximum_bytes: int,
    warm_start: RegularStreamREMLTrial | None,
    device: jax.Device | None,
) -> tuple[
    jax.Array,
    jax.Array,
    PreparedFittingMetadata,
    FamilyExecutionLineage,
    RegularREMLWorkspace,
]:
    """Validate one regular trial before scanning or transferring rows."""
    fitting = stream.prepared.fitting
    if fitting is None:
        raise ValueError("Regular streamed REML requires fitting preparation.")
    capabilities = family.execution_capabilities()
    if capabilities.dynamic_theta or family.n_theta:
        raise NotImplementedError(
            "Regular streamed REML does not optimize dynamic theta parameters."
        )
    n_lambda = fitting.penalty_structure.n_penalties
    n_params = n_lambda + int(not family.scale_known)

    # Reject the complete known controller and adjoint ledger while all
    # descriptors are still CPU-owned.  In particular, do not construct
    # PreparedFittingMetadata or convert trial parameters to JAX until an
    # undersized budget has failed prospectively.
    controller = preflight_regular_stream_workspace(stream, control, maximum_bytes)
    p = stream.prepared.n_coef
    B = controller.batch_rows
    # During the derivative scan, one fitting batch and its compiled copy can
    # coexist with observed/source inverse actions, cotangents and residuals.
    # Four scalar slots cover the batch/accumulated log-scale curvature and
    # its forward tangent. This charge is prospective and excludes opaque
    # XLA/BLAS scratch under the same documented numerical-workspace contract
    # as the controller.
    adjoint_bytes = 8 * (2 * B * p + 4 * p * p + 18 * p + 8 * n_params + 16 * B + 4)
    workspace = RegularREMLWorkspace(controller, adjoint_bytes)
    if workspace.required_bytes > maximum_bytes:
        raise MemoryError(
            "Regular streamed REML needs "
            f"{workspace.required_bytes} known workspace bytes, exceeding "
            f"maximum_bytes={maximum_bytes}."
        )

    if stream.source.fingerprint() != stream.prepared.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again.")
    if fitting.unpenalized_rank_deficit:
        raise np.linalg.LinAlgError(
            "Regular streamed REML requires a fixed full-rank identifiable subspace."
        )
    params_host = np.asarray(params, dtype=np.float64)
    if params_host.shape != (n_params,):
        raise ValueError(
            f"params must have shape ({n_params},), got {params_host.shape}."
        )
    if not np.all(np.isfinite(params_host)):
        raise ValueError("params must contain only finite values.")
    log_phi_host = 0.0 if family.scale_known else float(params_host[n_lambda])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        phi_host = float(np.exp(log_phi_host))
    if not np.isfinite(phi_host) or phi_host <= 0.0:
        raise ValueError("log_phi must define a finite positive score scale.")
    if warm_start is not None and (
        not warm_start.fit_result.state.converged
        or warm_start.params.shape != params_host.shape
        or warm_start.source_fingerprint != stream.prepared.source_fingerprint
        or warm_start.basis_fingerprint != stream.prepared.basis_fingerprint
        or warm_start.family_name != family.family_name
        or warm_start.link_name != type(family.link).__qualname__
    ):
        raise ValueError("Warm-start trial is not a compatible converged state.")

    lineage = FamilyExecutionLineage.from_prepared(stream.prepared, family)
    metadata = PreparedFittingMetadata.from_prepared(stream.prepared, family, device)
    rho = jnp.asarray(params_host[:n_lambda], dtype=jnp.float64)
    log_phi = jnp.asarray(log_phi_host, dtype=jnp.float64)
    return rho, log_phi, metadata, lineage, workspace


def _factor_inverse_residual(
    hessian: jax.Array,
    inverse: jax.Array,
) -> float:
    identity = jnp.eye(hessian.shape[0], dtype=hessian.dtype)
    product = hessian @ inverse
    numerator = jnp.max(jnp.abs(product - identity))
    denominator = 1.0 + jnp.max(jnp.abs(product))
    return float(np.asarray(numerator / denominator))


def _synchronize_fitted_state(
    state: StreamFitState, *additional_factors: object
) -> None:
    """Finish fit-owned device transfers before timing the derivative stage.

    ``StreamFitState`` itself is not a JAX pytree. Its array fields and both
    registered coefficient-factor pytrees must be walked explicitly; regular
    source scoring also owns a distinct final-refit factor.
    """
    pending = [
        value
        for entry in fields(state)
        if isinstance(value := getattr(state, entry.name), jax.Array)
    ]
    for factor in (
        state.coefficient_factor,
        state.fisher_coefficient_factor,
        *additional_factors,
    ):
        pending.extend(
            leaf
            for leaf in jax.tree_util.tree_leaves(factor)
            if isinstance(leaf, jax.Array)
        )
    jax.block_until_ready(tuple(pending))


def evaluate_regular_stream_reml(
    stream: StreamDesign,
    family: ExponentialFamily,
    params: np.ndarray | jax.Array,
    *,
    maximum_bytes: int,
    control: StreamPIRLSControl | None = None,
    warm_start: RegularStreamREMLTrial | None = None,
    initial_coefficients: np.ndarray | None = None,
    device: jax.Device | None = None,
) -> RegularStreamREMLTrial:
    """Reconverge one regular source trial and return its exact gradient.

    The score follows ``gam.fit3`` timing: raw deviance precedes the final
    source solve, while its penalty uses that solve's candidate coefficients.
    The derivative scan evaluates information statistics at the pre-refit
    coefficients and solves the IFT with the separately retained source
    coefficient factor.
    """
    control = (
        StreamPIRLSControl(solver_policy="qr", tol=1e-9) if control is None else control
    )
    if control.solver_policy != "qr":
        raise ValueError("Regular streamed REML requires solver_policy='qr'.")
    rho, log_phi, metadata, lineage, workspace = _preflight_regular_reml(
        stream, family, params, control, maximum_bytes, warm_start, device
    )
    rho_device = jax.device_put(rho, device)
    phi = float(np.asarray(jnp.exp(log_phi)))
    initial = (
        initial_coefficients
        if warm_start is None
        else np.asarray(warm_start.fit_result.state.coefficients)
    )
    result = fit_regular_streamed_pirls(
        stream,
        family,
        rho_device,
        maximum_bytes=maximum_bytes,
        score_scale=phi,
        control=control,
        initial_coefficients=initial,
        device=device,
    )
    _synchronize_fitted_state(result.state, result.source_coefficient_factor)
    derivative_started = time.perf_counter()
    state = result.state
    if (
        not state.converged
        or state.line_search_failed
        or not result.final_refit_accepted
        or not result.source_score.candidate_valid
        or not np.isfinite(state.stationarity)
        or state.stationarity >= control.tol
    ):
        raise RuntimeError(
            "Regular streamed REML requires a converged, valid final source state."
        )
    if not np.array_equal(np.asarray(state.log_lambda), np.asarray(rho_device)):
        raise RuntimeError("Regular streamed state does not match the requested rho.")
    if result.source_score.score_phi != phi:
        raise RuntimeError("Regular streamed source score does not match log_phi.")

    p = stream.prepared.n_coef
    if (
        state.coefficient_factor.rank != p
        or result.source_coefficient_factor.rank != p
        or not bool(np.asarray(state.coefficient_factor.score_admissible))
        or not bool(np.asarray(result.source_coefficient_factor.score_admissible))
    ):
        raise np.linalg.LinAlgError(
            "Regular streamed REML requires full-rank positive score/source factors."
        )
    identity = jax.device_put(jnp.eye(p, dtype=jnp.float64), device)
    observed_inverse = state.coefficient_factor.hessian_inverse(identity)
    source_inverse = result.source_coefficient_factor.hessian_inverse(identity)
    observed_hessian = penalty_ops.add_to_dense(
        metadata.penalty_structure, state.xtwx, rho_device
    )
    observed_residual = _factor_inverse_residual(observed_hessian, observed_inverse)
    source_residual = _factor_inverse_residual(observed_hessian, source_inverse)
    if (
        not np.isfinite(observed_residual)
        or not np.isfinite(source_residual)
        or observed_residual >= 1e-7
        or source_residual >= 1e-7
    ):
        raise np.linalg.LinAlgError(
            "Regular streamed score/source factors do not match the observed "
            "coefficient-stationarity Jacobian."
        )

    source_score = result.source_score
    core = regular_score_cotangents(
        rho_device,
        jax.device_put(log_phi, device),
        observed_inverse,
        jax.device_put(result.source_solve_coefficients, device),
        jax.device_put(source_score.raw_deviance, device),
        state.saturated_loglik,
        state.coefficient_factor.logdet_hessian(),
        metadata.penalty_structure,
        metadata.total_penalty_null_dim,
        metadata.singleton_sp_indices,
        metadata.singleton_ranks,
        metadata.singleton_eig_constants,
        metadata.multi_block_sp_indices,
        metadata.multi_block_ranks,
        metadata.multi_block_proj_S,
    )
    required = tuple(core)
    if not all(np.all(np.isfinite(np.asarray(value))) for value in required):
        raise FloatingPointError("Regular source score or cotangents are non-finite.")

    beta_bar = core.source_beta
    log_phi_bar = core.log_phi
    log_phi_curvature = core.log_phi_curvature
    informative_count = 0
    vjp_batches = 0
    alpha_roundoff_recovered = False
    parameters = jax.device_put(
        FamilyExecutionParameters.from_snapshot(lineage.parameters), device
    )
    information_beta = jax.device_put(result.information_coefficients, device)
    for X, y, weight, offset, valid in _fitting_batches(
        stream, family, lineage, control.batch_rows
    ):
        lineage.validate(stream.prepared, family)
        batch = regular_batch_statistics_vjp(
            information_beta,
            jax.device_put(log_phi, device),
            jax.device_put(X, device),
            jax.device_put(y, device),
            jax.device_put(weight, device),
            jax.device_put(offset, device),
            jax.device_put(valid, device),
            core.observed,
            core.deviance,
            core.saturated,
            parameters,
            family,
            lineage.context,
        )
        if not bool(np.asarray(batch.structural_admissible)):
            raise NotImplementedError(
                "Regular streamed derivative statistics are structurally inadmissible."
            )
        if not bool(np.asarray(batch.admissible)):
            if not bool(np.asarray(batch.alpha_unresolved)):
                raise NotImplementedError(
                    "Regular streamed derivative boundary is unresolved."
                )
            alpha_roundoff_recovered = True
        beta_bar = beta_bar + batch.beta
        log_phi_bar = log_phi_bar + batch.log_phi
        log_phi_curvature = log_phi_curvature + batch.log_phi_curvature
        informative_count += int(np.asarray(batch.informative_count))
        jax.block_until_ready(beta_bar)
        vjp_batches += 1
    if informative_count <= 0:
        raise ValueError(
            "Regular streamed derivative requires informative source rows."
        )
    if not np.all(np.isfinite(np.asarray(beta_bar))):
        raise FloatingPointError("Regular coefficient cotangent is non-finite.")

    adjoint = result.source_coefficient_factor.hessian_inverse(beta_bar)
    adjoint_product = observed_hessian @ adjoint
    adjoint_residual = float(
        np.asarray(
            jnp.max(jnp.abs(adjoint_product - beta_bar))
            / (1.0 + jnp.max(jnp.abs(adjoint_product)) + jnp.max(jnp.abs(beta_bar)))
        )
    )
    if not np.isfinite(adjoint_residual) or adjoint_residual >= 1e-7:
        raise np.linalg.LinAlgError(
            "Regular source adjoint has an unacceptable stationarity residual."
        )
    rho_gradient = core.rho - penalty_ops.parameter_vjp(
        metadata.penalty_structure,
        jax.device_put(result.source_solve_coefficients, device),
        adjoint,
        rho_device,
    )
    gradient = (
        rho_gradient
        if family.scale_known
        else jnp.concatenate((rho_gradient, jnp.atleast_1d(log_phi_bar)))
    )
    jax.block_until_ready(gradient)
    if not np.all(np.isfinite(np.asarray(gradient))):
        raise FloatingPointError("Regular streamed REML gradient is non-finite.")
    lineage.validate(stream.prepared, family)
    if stream.source.fingerprint() != lineage.source_fingerprint:
        raise RuntimeError("RowSource changed during regular REML evaluation.")
    jax.block_until_ready((core.score, gradient, log_phi_curvature))
    derivative_elapsed_s = time.perf_counter() - derivative_started
    return RegularStreamREMLTrial(
        params=jax.device_put(jnp.asarray(params, dtype=jnp.float64), device),
        score=core.score,
        gradient=gradient,
        fit_result=result,
        workspace=workspace,
        source_factor_residual=max(source_residual, adjoint_residual),
        observed_factor_residual=observed_residual,
        alpha_roundoff_recovered=alpha_roundoff_recovered,
        source_scans=state.source_scans + 1,
        batches_scanned=state.batches_scanned + vjp_batches,
        source_fingerprint=stream.prepared.source_fingerprint,
        basis_fingerprint=stream.prepared.basis_fingerprint,
        family_name=family.family_name,
        link_name=type(family.link).__qualname__,
        log_phi_curvature=(
            None if family.scale_known else float(np.asarray(log_phi_curvature))
        ),
        derivative_elapsed_s=derivative_elapsed_s,
        final_refit_solves=1,
    )


def _check_interior(
    stream: StreamDesign,
    beta: jax.Array,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    batch_rows: int,
    device: jax.Device | None,
) -> tuple[int, int]:
    """Replay rows to reject active working-weight clipping before a VJP."""
    batches = 0
    for X, _y, weight, offset, _valid in _fitting_batches(
        stream, family, lineage, batch_rows
    ):
        lineage.validate(stream.prepared, family)
        interior = batch_is_adjoint_interior(
            beta,
            jax.device_put(X, device),
            jax.device_put(weight, device),
            jax.device_put(offset, device),
            family,
        )
        if not bool(np.asarray(interior)):
            raise NotImplementedError(
                "Streamed REML adjoints reject active working-weight clipping or "
                "a family-domain boundary in the initial derivative scope."
            )
        jax.block_until_ready(interior)
        batches += 1
    return 1, batches


def evaluate_stream_reml(
    stream: StreamDesign,
    family: ExponentialFamily,
    rho: np.ndarray | jax.Array,
    *,
    control: StreamPIRLSControl | None = None,
    warm_start: StreamREMLTrial | None = None,
    device: jax.Device | None = None,
) -> StreamREMLTrial:
    """Reconverge one exact streamed REML trial and obtain its adjoint gradient.

    ``warm_start`` supplies only the previous accepted coefficient vector.  It
    is never used as a score cache: every exact ``rho`` runs the inner PIRLS
    convergence and all subsequent reductions against the replayable source.
    """
    control = (
        StreamPIRLSControl(tol=1e-9)
        if control is None
        else replace(control, tol=min(control.tol, 1e-9))
    )
    rho_array, metadata, lineage = _preflight(stream, family, rho, warm_start, device)
    rho_device = jax.device_put(rho_array, device)
    beta_init = None if warm_start is None else warm_start.fit_state.coefficients
    state = fit_streamed_pirls(
        stream,
        family,
        rho_device,
        control=control,
        beta_init=beta_init,
        device=device,
    )
    _synchronize_fitted_state(state)
    derivative_started = time.perf_counter()
    if (
        not state.converged
        or state.line_search_failed
        or not np.isfinite(state.stationarity)
        or state.stationarity >= control.tol
    ):
        raise RuntimeError(
            "Streamed REML trial inner PIRLS did not converge to the required "
            "stationarity tolerance; no gradient was returned."
        )
    if not np.array_equal(np.asarray(state.log_lambda), np.asarray(rho_device)):
        raise RuntimeError(
            "Streamed REML trial state does not match the requested rho."
        )

    interior_scans, interior_batches = _check_interior(
        stream, state.coefficients, family, lineage, control.batch_rows, device
    )
    score, (rho_bar, xtwx_bar, beta_bar, deviance_bar) = reml_score_cotangents(
        rho_device,
        state.xtwx,
        state.coefficients,
        state.deviance,
        state.saturated_loglik,
        metadata.penalty_structure,
        metadata.total_penalty_null_dim,
        metadata.singleton_sp_indices,
        metadata.singleton_ranks,
        metadata.singleton_eig_constants,
        metadata.multi_block_sp_indices,
        metadata.multi_block_ranks,
        metadata.multi_block_proj_S,
        metadata.rank_deficit,
    )
    if not all(
        np.all(np.isfinite(np.asarray(value)))
        for value in (score, rho_bar, xtwx_bar, beta_bar, deviance_bar)
    ):
        raise FloatingPointError(
            "Streamed REML core score or cotangents are non-finite."
        )
    vjp_batches = 0
    for X, y, weight, offset, _valid in _fitting_batches(
        stream, family, lineage, control.batch_rows
    ):
        lineage.validate(stream.prepared, family)
        beta_bar = beta_bar + batch_statistics_beta_vjp(
            state.coefficients,
            jax.device_put(X, device),
            jax.device_put(y, device),
            jax.device_put(weight, device),
            jax.device_put(offset, device),
            xtwx_bar,
            deviance_bar,
            family,
        )
        jax.block_until_ready(beta_bar)
        vjp_batches += 1
    if not np.all(np.isfinite(np.asarray(beta_bar))):
        raise FloatingPointError("Streamed REML batch VJP accumulation is non-finite.")
    adjoint, jitter, adjoint_residual = solve_observed_adjoint(
        state.xtwx, beta_bar, rho_device, metadata.penalty_structure
    )
    if (
        not np.isfinite(np.asarray(adjoint_residual))
        or float(np.asarray(adjoint_residual)) >= 1e-7
    ):
        raise np.linalg.LinAlgError(
            "Streamed REML observed-adjoint solve has an unacceptable scaled residual."
        )
    gradient = rho_bar - penalty_ops.parameter_vjp(
        metadata.penalty_structure, state.coefficients, adjoint, rho_device
    )
    jax.block_until_ready(gradient)
    if not all(
        np.all(np.isfinite(np.asarray(value)))
        for value in (score, beta_bar, adjoint, gradient, jitter, adjoint_residual)
    ):
        raise FloatingPointError(
            "Streamed REML adjoint produced a non-finite score, cotangent, or solve."
        )
    lineage.validate(stream.prepared, family)
    if stream.source.fingerprint() != lineage.source_fingerprint:
        raise RuntimeError("RowSource changed during streamed REML evaluation.")
    jax.block_until_ready((score, gradient))
    derivative_elapsed_s = time.perf_counter() - derivative_started
    return StreamREMLTrial(
        rho=rho_device,
        score=score,
        gradient=gradient,
        fit_state=state,
        adjoint_jitter=jitter,
        source_scans=state.source_scans + interior_scans + 1,
        batches_scanned=state.batches_scanned + interior_batches + vjp_batches,
        source_fingerprint=stream.prepared.source_fingerprint,
        basis_fingerprint=stream.prepared.basis_fingerprint,
        family_name=family.family_name,
        link_name=type(family.link).__qualname__,
        derivative_elapsed_s=derivative_elapsed_s,
    )


def _same_rho(left: np.ndarray, right: jax.Array | np.ndarray) -> bool:
    """Compare host smoothing vectors exactly at an optimizer boundary."""
    return np.array_equal(left, np.asarray(right, dtype=np.float64))


def _trial_inner_iterations(trial: object) -> int:
    """Count completed PIRLS loop passes, excluding the final score refit."""
    state = getattr(trial, "fit_state", None)
    if state is None:
        fit_result = getattr(trial, "fit_result", None)
        state = getattr(fit_result, "state", None)
    return int(getattr(state, "n_iter", 0))


def _projected_gradient(
    rho: np.ndarray,
    gradient: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Return the minimization projected gradient for box-constrained rho."""
    projected = np.array(gradient, dtype=np.float64, copy=True)
    at_lower = rho <= lower
    at_upper = rho >= upper
    projected[at_lower] = np.minimum(projected[at_lower], 0.0)
    projected[at_upper] = np.maximum(projected[at_upper], 0.0)
    return projected


class _AcceptedTrialObjective:
    """SciPy objective retaining at most accepted and candidate trial states."""

    def __init__(
        self,
        stream: StreamDesign,
        family: ExponentialFamily,
        rho_initial: np.ndarray,
        pirls_control: StreamPIRLSControl | None,
        reml_control: StreamREMLControl,
        device: jax.Device | None,
    ) -> None:
        self._stream = stream
        self._family = family
        self._pirls_control = pirls_control
        self._device = device
        self.accepted = evaluate_stream_reml(
            stream,
            family,
            rho_initial,
            control=pirls_control,
            device=device,
        )
        self.candidate: StreamREMLTrial | None = None
        self.n_evaluations = 1
        self.n_accepted = 0
        self.cumulative_source_scans = self.accepted.source_scans
        self.cumulative_batches_scanned = self.accepted.batches_scanned
        self.cumulative_inner_iterations = _trial_inner_iterations(self.accepted)
        self.cumulative_final_refit_solves = getattr(
            self.accepted, "final_refit_solves", 0
        )
        self.cumulative_derivative_s = getattr(
            self.accepted, "derivative_elapsed_s", 0.0
        )
        self._history: deque[float] = deque(maxlen=reml_control.history_size)
        self._history.append(float(np.asarray(self.accepted.score)))

    @property
    def accepted_score_history(self) -> tuple[float, ...]:
        return tuple(self._history)

    def _check_source_unchanged(self) -> None:
        """Keep cached exact trials invalid when their replayable source mutates."""
        if self._stream.source.fingerprint() != self.accepted.source_fingerprint:
            raise RuntimeError("RowSource changed during streamed REML optimization.")

    def __call__(self, rho: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate a trial from the last callback-accepted coefficient state."""
        self._check_source_unchanged()
        rho_host = np.asarray(rho, dtype=np.float64)
        if _same_rho(rho_host, self.accepted.rho):
            trial = self.accepted
        elif self.candidate is not None and _same_rho(rho_host, self.candidate.rho):
            trial = self.candidate
        else:
            # Drop an older rejected candidate before constructing its
            # replacement: the persistent and peak trial count stays two.
            self.candidate = None
            trial = evaluate_stream_reml(
                self._stream,
                self._family,
                rho_host,
                control=self._pirls_control,
                warm_start=self.accepted,
                device=self._device,
            )
            self.candidate = trial
            self.n_evaluations += 1
            self.cumulative_source_scans += trial.source_scans
            self.cumulative_batches_scanned += trial.batches_scanned
            self.cumulative_inner_iterations += _trial_inner_iterations(trial)
            self.cumulative_final_refit_solves += getattr(
                trial, "final_refit_solves", 0
            )
            self.cumulative_derivative_s += getattr(trial, "derivative_elapsed_s", 0.0)
        score = float(np.asarray(trial.score))
        gradient = np.asarray(trial.gradient, dtype=np.float64)
        if not np.isfinite(score) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                "Streamed REML optimizer received a non-finite trial."
            )
        return score, gradient

    def callback(self, intermediate_result: OptimizeResult) -> None:
        """Promote exactly the L-BFGS-B accepted point after ``NEW_X``."""
        self._check_source_unchanged()
        rho = np.asarray(intermediate_result.x, dtype=np.float64)
        if _same_rho(rho, self.accepted.rho):
            return
        if self.candidate is None or not _same_rho(rho, self.candidate.rho):
            raise RuntimeError(
                "L-BFGS-B accepted a point without the matching exact streamed "
                "trial; refusing to warm-start from an unverified state."
            )
        self.accepted = self.candidate
        self.candidate = None
        self.n_accepted += 1
        self._history.append(float(np.asarray(self.accepted.score)))


def optimize_stream_reml(
    stream: StreamDesign,
    family: ExponentialFamily,
    initial_rho: np.ndarray | jax.Array,
    *,
    pirls_control: StreamPIRLSControl | None = None,
    control: StreamREMLControl | None = None,
    device: jax.Device | None = None,
) -> StreamREMLOptimization:
    """Optimize initial-scope streamed REML with bounded L-BFGS-B state.

    This is an internal execution driver, not a public ``GAM.fit`` route.  It
    only invokes :func:`evaluate_stream_reml`, so every candidate is an exact,
    fully reconverged coefficient fit.  Candidate trials never become warm
    starts until SciPy reports them through its accepted-iteration callback.
    """
    reml_control = StreamREMLControl() if control is None else control
    fitting = stream.prepared.fitting
    if fitting is None:
        raise ValueError("Streamed REML optimization requires fitting preparation.")
    n_penalties = fitting.penalty_structure.n_penalties
    if n_penalties == 0:
        raise NotImplementedError(
            "Streamed REML L-BFGS-B requires at least one smoothing parameter."
        )
    rho_initial = np.asarray(initial_rho, dtype=np.float64)
    if rho_initial.shape != (n_penalties,):
        raise ValueError(
            f"initial_rho must have shape ({n_penalties},), got {rho_initial.shape}."
        )
    if not np.all(np.isfinite(rho_initial)):
        raise ValueError(
            "initial_rho must contain only finite log smoothing parameters."
        )

    lower, upper = -40.0, 40.0
    rho_initial = np.clip(rho_initial, lower, upper)
    base_pirls_control = (
        StreamPIRLSControl() if pirls_control is None else pirls_control
    )
    # mgcv's outer Newton tightens the inner solve to conv.tol / 100 before
    # differentiating.  Every L-BFGS-B trial needs the same relationship.
    pirls_control = replace(
        base_pirls_control,
        tol=min(base_pirls_control.tol, 1e-9, reml_control.gtol / 100.0),
    )
    objective = _AcceptedTrialObjective(
        stream,
        family,
        rho_initial,
        pirls_control,
        reml_control,
        device,
    )
    result = minimize(
        objective,
        rho_initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(lower, upper)] * n_penalties,
        callback=objective.callback,
        options={
            "maxiter": reml_control.max_iter,
            "maxfun": reml_control.maxfun,
            "maxcor": reml_control.maxcor,
            "maxls": reml_control.maxls,
            "gtol": reml_control.gtol,
            "ftol": reml_control.ftol,
        },
    )
    result_rho = np.asarray(result.x, dtype=np.float64)
    if not _same_rho(result_rho, objective.accepted.rho):
        raise RuntimeError(
            "L-BFGS-B returned a point without a matching accepted streamed "
            "trial; no optimization result is available."
        )
    objective._check_source_unchanged()
    gradient = np.asarray(objective.accepted.gradient, dtype=np.float64)
    projected_gradient_inf = float(
        np.max(
            np.abs(_projected_gradient(result_rho, gradient, lower=lower, upper=upper))
        )
    )
    converged = bool(result.success) and projected_gradient_inf <= reml_control.gtol
    message = str(result.message)
    if bool(result.success) and not converged:
        message = (
            f"{message}; projected gradient {projected_gradient_inf:.3e} exceeds "
            f"gtol {reml_control.gtol:.3e}."
        )
    return StreamREMLOptimization(
        trial=objective.accepted,
        converged=converged,
        message=message,
        status=int(result.status),
        n_iter=int(result.nit),
        n_evaluations=objective.n_evaluations,
        n_accepted=objective.n_accepted,
        cumulative_source_scans=objective.cumulative_source_scans,
        cumulative_batches_scanned=objective.cumulative_batches_scanned,
        projected_gradient_inf=projected_gradient_inf,
        accepted_score_history=objective.accepted_score_history,
        cumulative_inner_iterations=objective.cumulative_inner_iterations,
        cumulative_final_refit_solves=objective.cumulative_final_refit_solves,
        cumulative_derivative_s=objective.cumulative_derivative_s,
    )


class _AcceptedParameterizedTrialObjective(Generic[_TrialT]):
    """Common exact objective retaining only accepted and candidate trials."""

    def __init__(
        self,
        stream: StreamDesign,
        params_initial: np.ndarray,
        reml_control: StreamREMLControl,
        evaluator: Callable[[np.ndarray, _TrialT | None], _TrialT],
        *,
        objective_name: str,
        source_name: str,
        trial_name: str,
    ) -> None:
        self._stream = stream
        self._evaluate = evaluator
        self._objective_name = objective_name
        self._source_name = source_name
        self._trial_name = trial_name
        self.accepted = evaluator(params_initial, None)
        self.candidate: _TrialT | None = None
        self.n_evaluations = 1
        self.n_accepted = 0
        self.cumulative_source_scans = self.accepted.source_scans
        self.cumulative_batches_scanned = self.accepted.batches_scanned
        self.cumulative_inner_iterations = _trial_inner_iterations(self.accepted)
        self.cumulative_final_refit_solves = getattr(
            self.accepted, "final_refit_solves", 0
        )
        self.cumulative_derivative_s = getattr(
            self.accepted, "derivative_elapsed_s", 0.0
        )
        self._history: deque[float] = deque(maxlen=reml_control.history_size)
        self._history.append(float(np.asarray(self.accepted.score)))

    @property
    def accepted_score_history(self) -> tuple[float, ...]:
        return tuple(self._history)

    def _check_source_unchanged(self) -> None:
        if self._stream.source.fingerprint() != self.accepted.source_fingerprint:
            raise RuntimeError(
                f"RowSource changed during {self._source_name} optimization."
            )

    def _evaluate_candidate(self, params: np.ndarray) -> _TrialT:
        """Replace any candidate with one new exact trial and count its scans."""
        # Drop an older rejected candidate before constructing its
        # replacement: the persistent and peak trial count stays two.
        self.candidate = None
        trial = self._evaluate(params, self.accepted)
        self.candidate = trial
        self.n_evaluations += 1
        self.cumulative_source_scans += trial.source_scans
        self.cumulative_batches_scanned += trial.batches_scanned
        self.cumulative_inner_iterations += _trial_inner_iterations(trial)
        self.cumulative_final_refit_solves += getattr(trial, "final_refit_solves", 0)
        self.cumulative_derivative_s += getattr(trial, "derivative_elapsed_s", 0.0)
        return trial

    def _promote_candidate(self) -> None:
        """Make the current exact candidate the accepted warm-start state."""
        if self.candidate is None:
            raise RuntimeError("No exact candidate trial is available to accept.")
        self.accepted = self.candidate
        self.candidate = None
        self.n_accepted += 1
        self._history.append(float(np.asarray(self.accepted.score)))

    def __call__(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate from the last callback-accepted coefficient state."""
        self._check_source_unchanged()
        params_host = np.asarray(params, dtype=np.float64)
        if _same_rho(params_host, self.accepted.params):
            trial = self.accepted
        elif self.candidate is not None and _same_rho(
            params_host, self.candidate.params
        ):
            trial = self.candidate
        else:
            trial = self._evaluate_candidate(params_host)
        score = float(np.asarray(trial.score))
        gradient = np.asarray(trial.gradient, dtype=np.float64)
        if not np.isfinite(score) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                f"{self._objective_name} optimizer received a non-finite trial."
            )
        return score, gradient

    def callback(self, intermediate_result: OptimizeResult) -> None:
        """Promote only the exact trial accepted by L-BFGS-B."""
        self._check_source_unchanged()
        params = np.asarray(intermediate_result.x, dtype=np.float64)
        if _same_rho(params, self.accepted.params):
            return
        if self.candidate is None or not _same_rho(params, self.candidate.params):
            raise RuntimeError(
                "L-BFGS-B accepted a point without the matching exact "
                f"{self._trial_name} trial; refusing an unverified warm start."
            )
        self._promote_candidate()


class _AcceptedRegularTrialObjective(
    _AcceptedParameterizedTrialObjective[RegularStreamREMLTrial]
):
    """Regular adapter for the common accepted/candidate trial driver."""

    def __init__(
        self,
        stream: StreamDesign,
        family: ExponentialFamily,
        params_initial: np.ndarray,
        pirls_control: StreamPIRLSControl,
        reml_control: StreamREMLControl,
        maximum_bytes: int,
        initial_coefficients: np.ndarray | None,
        device: jax.Device | None,
    ) -> None:
        def evaluate(
            params: np.ndarray,
            warm_start: RegularStreamREMLTrial | None,
        ) -> RegularStreamREMLTrial:
            return evaluate_regular_stream_reml(
                stream,
                family,
                params,
                maximum_bytes=self._maximum_bytes,
                control=pirls_control,
                warm_start=warm_start,
                initial_coefficients=(
                    initial_coefficients if warm_start is None else None
                ),
                device=device,
            )

        self._maximum_bytes = maximum_bytes
        super().__init__(
            stream,
            params_initial,
            reml_control,
            evaluate,
            objective_name="Regular streamed REML",
            source_name="regular streamed REML",
            trial_name="regular streamed",
        )


def _parameterized_optimizer_workspace_bytes(
    p: int,
    n_params: int,
    control: StreamREMLControl,
    pirls_control: StreamPIRLSControl,
) -> tuple[int, int, int]:
    """Return retained exact-trial and bounded L-BFGS-B workspace bytes."""
    # The retained result owns observed, Fisher and source factors, their
    # compact corrections, two information matrices, coefficient roles,
    # parameters/gradient and scalar source diagnostics.  Twelve p-square and
    # forty-eight p-vector slots are a prospective upper bound measured by an
    # owning test against the recursively enumerated immutable result arrays.
    retained_trial_bytes = 8 * (12 * p * p + 48 * p)
    pointer_bytes = np.dtype(np.intp).itemsize
    # The accepted fit's penalized-deviance tuple coexists with the active
    # candidate, whose own history is already in the controller ledger.
    retained_history_bytes = (pirls_control.max_iter + 1) * (
        sys.getsizeof(0.0) + pointer_bytes
    ) + sys.getsizeof(())
    # L-BFGS-B's documented double workspace is
    # (2m+5)q + 11m^2 + 8m, with 3q integer slots. Charge those integers as
    # eight-byte entries. The accepted-score deque and returned tuple can
    # coexist, so charge their float objects, references and containers too.
    # Four pointer slots per entry conservatively cover both references plus
    # deque block links and rounded spare capacity at large history sizes.
    m = control.maxcor
    lbfgs_workspace_bytes = (
        8 * ((2 * m + 8) * n_params + 11 * m * m + 8 * m)
        + control.history_size * (sys.getsizeof(0.0) + 4 * pointer_bytes)
        + sys.getsizeof(deque())
        + sys.getsizeof(())
    )
    return retained_trial_bytes, retained_history_bytes, lbfgs_workspace_bytes


def _regular_optimizer_workspace_bytes(
    p: int,
    n_params: int,
    control: StreamREMLControl,
    pirls_control: StreamPIRLSControl,
) -> tuple[int, int, int]:
    """Compatibility name for the shared parameterized optimizer ledger."""
    return _parameterized_optimizer_workspace_bytes(
        p,
        n_params,
        control,
        pirls_control,
    )


def _projected_gradient_for_bounds(
    params: np.ndarray,
    gradient: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
) -> np.ndarray:
    """Apply minimization projected-gradient rules to general box bounds."""
    projected = np.array(gradient, dtype=np.float64, copy=True)
    for index, (value, bound) in enumerate(zip(params, bounds, strict=True)):
        lower, upper = bound
        if lower is not None and upper is not None and lower == upper:
            projected[index] = 0.0
        elif lower is not None and value <= lower:
            projected[index] = min(projected[index], 0.0)
        elif upper is not None and value >= upper:
            projected[index] = max(projected[index], 0.0)
    return projected


def _validate_max_step(max_step: float | None) -> None:
    if max_step is not None and (
        isinstance(max_step, bool)
        or not isinstance(max_step, numbers.Real)
        or not np.isfinite(max_step)
        or max_step <= 0.0
    ):
        raise ValueError("max_step must be finite and positive when supplied.")


# Free-parameter curvature completion after an L-BFGS-B roundoff stop. Exact
# trials carry exact gradients but no outer Hessian, so each Newton iteration
# spends 2m central-difference gradient probes on curvature and at most
# 1 + _CURVATURE_COMPLETION_MAX_HALVINGS trials on its step. From the plateau
# where L-BFGS-B stops one step normally reaches gtol; the iteration bound
# only guarantees termination. The free-parameter cap bounds the probes.
_CURVATURE_COMPLETION_MAX_FREE = 4
_CURVATURE_COMPLETION_MAX_STEPS = 3
_CURVATURE_COMPLETION_MAX_HALVINGS = 3
_CURVATURE_COMPLETION_PROBE_WIDTH = 1e-4


@dataclass(frozen=True)
class _CurvatureCompletion:
    """Outcome of the free-parameter finite-difference Newton completion.

    ``curvature_positive`` states that a positive definite curvature model
    belongs to the returned trial: either the central-difference Hessian of
    exact gradients evaluated at it, or the one at the origin of the accepted
    step that produced it together with a positive exact-gradient secant
    along that step. ``eigenvalues`` are those of the last curvature model
    formed; ``last_step_inf`` is the max-norm of the last proposed step.
    """

    steps: int
    stop: str
    curvature_positive: bool = False
    eigenvalues: tuple[float, ...] = ()
    last_step_inf: float | None = None
    last_step_accepted: bool = False


def _lbfgsb_roundoff_stop(result: OptimizeResult) -> bool:
    """Whether L-BFGS-B stopped on raw-score roundoff rather than a limit.

    A relative-reduction stop means the accepted score did not decrease by
    ``ftol`` (with ``ftol=0``, not at all). ``ABNORMAL`` means ``maxls``
    line-search trials failed the Moré-Thuente sufficient-decrease test and
    SciPy restored the last accepted iterate. Near the optimum both compare
    raw-score roundoff. Iteration/evaluation limits and errors are excluded.
    """
    return (
        bool(result.success) and "RELATIVE REDUCTION OF F" in str(result.message)
    ) or _abnormal_line_search_stop(result)


def _free_projected_norm(
    params: np.ndarray,
    gradient: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
) -> float:
    """Max-norm of the bound-projected gradient."""
    projected = _projected_gradient_for_bounds(params, gradient, bounds)
    return float(np.max(np.abs(projected)))


def _curvature_newton_rejection(
    accepted: _TrialT,
    trial: _TrialT,
    step: np.ndarray,
    curvature: np.ndarray,
    free: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    score_roundoff: Callable[[_TrialT], float],
) -> str | None:
    """Return why one free-parameter Newton trial is not acceptable.

    Pinned ``gam.fit3.r::newton`` accepts a step when the score decreases and
    the quadratic-model error ``|pred - change|/max(|pred|, |change|)`` is
    below ``qerror.thresh = 0.8`` (newton() also adds ``score.scale*conv.tol``
    to that denominator; omitting it is stricter). Near the optimum the raw
    score change is far below the roundoff of the O(n) score: a Gamma trial
    reconverged from a distant warm start read 2.6e-12 above nearby trials
    whose exact changes were below 6.1e-14, five times the first-order
    envelope of one score. These tests
    therefore use the trapezoid integral of the exact endpoint gradients
    along the step (exact for the local quadratic, error O(|s|^3)), with the
    single-coordinate completion's acceptance tests in the same order: a
    positive exact secant ``s'(g1 - g0)``, a smaller exact projected
    gradient, derivative-integrated descent, the quadratic-model error, and
    no raw score increase beyond the float64 envelope of both scores.
    """
    gradient_old = np.asarray(accepted.gradient, dtype=np.float64)
    gradient_new = np.asarray(trial.gradient, dtype=np.float64)
    score_old = float(np.asarray(accepted.score))
    score_new = float(np.asarray(trial.score))
    if not (
        np.isfinite(score_old)
        and np.isfinite(score_new)
        and np.all(np.isfinite(gradient_old))
        and np.all(np.isfinite(gradient_new))
    ):
        return "nonfinite trial"
    old_free = gradient_old[free]
    new_free = gradient_new[free]
    if not float(step @ (new_free - old_free)) > 0.0:
        return "nonpositive secant curvature"
    old_norm = _free_projected_norm(
        np.asarray(accepted.params, dtype=np.float64), gradient_old, bounds
    )
    new_norm = _free_projected_norm(
        np.asarray(trial.params, dtype=np.float64), gradient_new, bounds
    )
    if not new_norm < old_norm:
        return "no gradient reduction"
    predicted = float(old_free @ step + 0.5 * (step @ curvature @ step))
    change = float(0.5 * ((old_free + new_free) @ step))
    return _newton_model_rejection(
        predicted,
        change,
        score_old,
        score_new,
        float(score_roundoff(accepted)) + float(score_roundoff(trial)),
    )


def _complete_free_curvature_newton(
    objective: _AcceptedParameterizedTrialObjective[_TrialT],
    bounds: list[tuple[float | None, float | None]],
    control: StreamREMLControl,
    *,
    iterations: int,
    max_step: float | None,
    trial_valid: Callable[[_TrialT], bool],
    score_roundoff: Callable[[_TrialT], float],
) -> tuple[_CurvatureCompletion, int]:
    """Continue pinned ``newton()`` on a few free parameters after L-BFGS-B.

    For free (log sp, log scale) or (log sp, log theta) parameters pinned
    mgcv's ``gam.outer`` runs ``gam.fit3.r::newton``: from the exact Hessian
    it takes ``-H^{-1} g``, scaled so its largest component is at most
    ``maxNstep = 5``, accepts it on score decrease with quadratic-model error
    below 0.8, step-halves otherwise, and declares convergence when the new
    gradient is small and the Hessian that produced the step was positive
    definite (``converged <- !indef``). Near the optimum the remaining REML
    decrease is below one ULP of the O(n) score, so L-BFGS-B's raw-score line
    search stops by relative reduction or ``ABNORMAL`` before ``gtol``.

    The exact trial has no outer Hessian. Each iteration therefore forms a
    central-difference Hessian of exact gradients from ``2m`` fully
    reconverged probes at ``1e-4 * max(1, |x_j|)`` (capped by ``max_step``),
    symmetrizes it and requires newton()'s positive definiteness (no
    eigenvalue below ``max(d) * eps^0.7``). The Newton step, capped at
    ``min(maxNstep, max_step)``, is accepted or halved at most three times by
    :func:`_curvature_newton_rejection`. Every accepted state is a fully
    reconverged exact source trial promoted through the objective; at most
    the accepted and one candidate trial are held, plus host m-vectors and
    the m-by-m model. The completion stops when the accepted trial meets
    ``gtol`` (``converged`` after a step, ``stationary`` before one), at the
    first unusable curvature model, after an unaccepted halving sequence, or
    at the step, iteration or evaluation limits. Fixed coordinates
    (``lower == upper``) are never moved.
    """
    free = np.flatnonzero(
        [
            not (lower is not None and upper is not None and lower == upper)
            for lower, upper in bounds
        ]
    )
    m = int(free.size)
    step_cap = (
        _SCALE_NEWTON_MAX_STEP
        if max_step is None
        else min(_SCALE_NEWTON_MAX_STEP, float(max_step))
    )
    per_iteration = 2 * m + 1 + _CURVATURE_COMPLETION_MAX_HALVINGS
    lineage = ("source_fingerprint", "basis_fingerprint", "family_name", "link_name")
    steps = 0
    curvature_positive = False
    eigenvalues: tuple[float, ...] = ()
    last_step_inf: float | None = None
    last_step_accepted = False

    def outcome(stop: str) -> tuple[_CurvatureCompletion, int]:
        return (
            _CurvatureCompletion(
                steps,
                stop,
                curvature_positive,
                eigenvalues,
                last_step_inf,
                last_step_accepted,
            ),
            iterations,
        )

    def evaluate(params: np.ndarray) -> _TrialT:
        reference = objective.accepted
        trial = objective._evaluate_candidate(params)
        if not _same_rho(params, trial.params):
            raise RuntimeError(
                "Curvature completion received a trial for other parameters."
            )
        if any(
            getattr(trial, name, None) != getattr(reference, name, None)
            for name in lineage
        ):
            raise RuntimeError("Curvature completion changed source/family lineage.")
        return trial

    # Release SciPy's last rejected line-search trial: the completion holds
    # the accepted and at most one candidate trial.
    objective.candidate = None
    for _ in range(_CURVATURE_COMPLETION_MAX_STEPS):
        objective._check_source_unchanged()
        accepted = objective.accepted
        center = np.asarray(accepted.params, dtype=np.float64)
        gradient = np.asarray(accepted.gradient, dtype=np.float64)
        if iterations >= control.max_iter:
            return outcome("iteration limit")
        if objective.n_evaluations + per_iteration > control.maxfun:
            return outcome("evaluation limit")
        widths = np.array(
            [
                _CURVATURE_COMPLETION_PROBE_WIDTH * max(1.0, abs(center[index]))
                for index in free
            ]
        )
        if max_step is not None:
            widths = np.minimum(widths, max_step)
        if any(
            (bounds[index][0] is not None and center[index] - width < bounds[index][0])
            or (
                bounds[index][1] is not None
                and center[index] + width > bounds[index][1]
            )
            for index, width in zip(free, widths, strict=True)
        ):
            return outcome("curvature probe outside bounds")
        curvature = np.empty((m, m), dtype=np.float64)
        for column, (index, width) in enumerate(zip(free, widths, strict=True)):
            probes = []
            for sign in (1.0, -1.0):
                probe_params = center.copy()
                probe_params[index] = center[index] + sign * width
                probe = evaluate(probe_params)
                valid = trial_valid(probe)
                probes.append(np.asarray(probe.gradient, dtype=np.float64)[free])
                objective.candidate = None
                del probe
                if not valid:
                    return outcome("invalid curvature probe")
            curvature[:, column] = (probes[0] - probes[1]) / (2.0 * width)
        curvature = 0.5 * (curvature + curvature.T)
        values = (
            np.linalg.eigvalsh(curvature)
            if np.all(np.isfinite(curvature))
            else np.full(m, np.nan)
        )
        eigenvalues = tuple(float(value) for value in values)
        # newton(): pdef is FALSE when an eigenvalue is negative or below
        # max(d) * .Machine$double.eps^.7.
        curvature_positive = bool(
            np.all(np.isfinite(values))
            and values[-1] > 0.0
            and values[0] >= values[-1] * np.finfo(np.float64).eps ** 0.7
        )
        if not curvature_positive:
            return outcome("indefinite curvature")
        if _free_projected_norm(center, gradient, bounds) <= control.gtol:
            return outcome("stationary")
        newton = -np.linalg.solve(curvature, gradient[free])
        largest = float(np.max(np.abs(newton)))
        if largest > step_cap:
            # newton(): Nstep <- mns * Nstep/ms.
            newton = step_cap * (newton / largest)
        reason: str | None = None
        for halving in range(_CURVATURE_COMPLETION_MAX_HALVINGS + 1):
            trial_params = center.copy()
            trial_params[free] = center[free] + newton / 2.0**halving
            if np.array_equal(trial_params, center):
                return outcome("step not representable")
            if any(
                (lower is not None and value < lower)
                or (upper is not None and value > upper)
                for value, (lower, upper) in zip(trial_params, bounds, strict=True)
            ):
                return outcome("step leaves parameter bounds")
            # The realized float64 step defines the model and the integral.
            step = trial_params[free] - center[free]
            last_step_inf = float(np.max(np.abs(step)))
            last_step_accepted = False
            trial = evaluate(trial_params)
            reason = (
                "invalid trial"
                if not trial_valid(trial)
                else _curvature_newton_rejection(
                    accepted, trial, step, curvature, free, bounds, score_roundoff
                )
            )
            del trial
            if reason is None:
                break
            objective.candidate = None
        if reason is not None:
            return outcome(f"rejected: {reason}")
        objective._promote_candidate()
        steps += 1
        iterations += 1
        last_step_accepted = True
        # The model that produced this step was positive definite and the
        # exact secant along it is positive (newton()'s convergence rule).
        curvature_positive = True
        promoted = objective.accepted
        if (
            _free_projected_norm(
                np.asarray(promoted.params, dtype=np.float64),
                np.asarray(promoted.gradient, dtype=np.float64),
                bounds,
            )
            <= control.gtol
        ):
            return outcome("converged")
    return outcome("step limit")


def _run_parameterized_stream_reml(
    objective: _AcceptedParameterizedTrialObjective[_TrialT],
    params_initial: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    control: StreamREMLControl,
    *,
    roundoff_stationary_completion: bool = False,
    curvature_stationary_completion: bool = False,
    completion_inner_tolerance: float | None = None,
    completion_score_roundoff: Callable[[_TrialT], float] | None = None,
    local_score_change: Callable[
        [_AcceptedParameterizedTrialObjective[_TrialT], _TrialT, _TrialT, float],
        _LocalScoreChange | None,
    ]
    | None = None,
    max_step: float | None = None,
) -> _ParameterizedOptimization[_TrialT]:
    """Run the common exact accepted/candidate L-BFGS-B protocol.

    A pinned, one-dimensional nuisance parameter can reach a point whose
    projected gradient is resolved but whose score differs by only one float64
    ULP. The optional completion policy retains its already reconverged trial
    across later line-search probes; callers charge that additional ownership.
    With free parameters, ``curvature_stationary_completion`` continues
    pinned newton() after an L-BFGS-B roundoff stop (relative reduction or
    ``ABNORMAL``) from a valid accepted trial whose projected gradient still
    exceeds ``gtol`` (:func:`_complete_free_curvature_newton`).
    ``completion_score_roundoff`` is the float64 envelope of one trial score.
    Convergence is then restated on the returned trial: projected gradient
    at or below ``gtol`` with a positive definite curvature model.
    """
    if not _same_rho(np.asarray(params_initial), objective.accepted.params):
        raise ValueError("Initial parameters do not match the exact accepted trial.")
    if all(
        lower is not None and upper is not None and lower == upper
        for lower, upper in bounds
    ):
        objective._check_source_unchanged()
        return _ParameterizedOptimization(
            trial=objective.accepted,
            converged=True,
            message="All parameter coordinates are fixed; returned exact evaluation.",
            status=0,
            n_iter=0,
            n_evaluations=objective.n_evaluations,
            n_accepted=objective.n_accepted,
            cumulative_source_scans=objective.cumulative_source_scans,
            cumulative_batches_scanned=objective.cumulative_batches_scanned,
            projected_gradient_inf=0.0,
            accepted_score_history=objective.accepted_score_history,
            cumulative_inner_iterations=objective.cumulative_inner_iterations,
            cumulative_final_refit_solves=objective.cumulative_final_refit_solves,
            cumulative_derivative_s=objective.cumulative_derivative_s,
        )
    if roundoff_stationary_completion and (
        completion_inner_tolerance is None or completion_inner_tolerance <= 0.0
    ):
        raise ValueError("Stationary completion requires an inner tolerance.")
    if local_score_change is not None and not roundoff_stationary_completion:
        raise ValueError("Local score comparison requires stationary completion.")
    if curvature_stationary_completion and (
        completion_inner_tolerance is None or completion_inner_tolerance <= 0.0
    ):
        raise ValueError("Curvature completion requires an inner tolerance.")
    if curvature_stationary_completion and completion_score_roundoff is None:
        raise ValueError("Curvature completion requires a score roundoff envelope.")
    _validate_max_step(max_step)
    stationary_trial: _TrialT | None = None
    stationary_norm = np.inf
    stationary_raw_tie = False

    def inner_valid(trial: _TrialT) -> bool:
        inner = trial.fit_result.state
        return bool(
            inner.converged
            and not inner.line_search_failed
            and np.isfinite(inner.stationarity)
            and inner.stationarity < completion_inner_tolerance
        )

    def observe(params: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal stationary_trial, stationary_norm, stationary_raw_tie
        score, gradient = objective(params)
        projected = _projected_gradient_for_bounds(
            np.asarray(params, dtype=np.float64), gradient, bounds
        )
        norm = float(np.max(np.abs(projected)))
        accepted_score = float(np.asarray(objective.accepted.score))
        score_ulp = float(np.spacing(abs(accepted_score)))
        raw_tie = abs(score - accepted_score) <= score_ulp
        prefer = (
            stationary_trial is None
            or (raw_tie and not stationary_raw_tie)
            or (raw_tie == stationary_raw_tie and norm < stationary_norm)
        )
        if (
            norm <= control.gtol
            and prefer
            and objective.candidate is not None
            and _same_rho(np.asarray(params), objective.candidate.params)
            and (raw_tie or local_score_change is not None)
            and inner_valid(objective.candidate)
        ):
            stationary_trial = objective.candidate
            stationary_norm = norm
            stationary_raw_tie = raw_tie
        return score, gradient

    total_iterations = 0
    windows = 0
    while True:
        center = np.asarray(objective.accepted.params, dtype=np.float64)
        if max_step is None:
            local_bounds = bounds
        else:
            # Pinned fast-REML.r:fast.REML.fit caps each log-parameter
            # proposal at maxNstep=5 and halves unsuccessful trials. A local
            # L-BFGS-B window enforces the same step scale without evaluating an
            # ill-conditioned distant derivative. Boundary optima reopen a
            # fresh window from the exact accepted trial.
            local_bounds = [
                (
                    max(lower, value - max_step)
                    if lower is not None
                    else value - max_step,
                    min(upper, value + max_step)
                    if upper is not None
                    else value + max_step,
                )
                if not (lower is not None and upper is not None and lower == upper)
                else (lower, upper)
                for value, (lower, upper) in zip(center, bounds, strict=True)
            ]
        result = minimize(
            observe if roundoff_stationary_completion else objective,
            center,
            method="L-BFGS-B",
            jac=True,
            bounds=local_bounds,
            callback=objective.callback,
            options={
                "maxiter": max(1, control.max_iter - total_iterations),
                "maxfun": max(1, control.maxfun - objective.n_evaluations + 1),
                "maxcor": control.maxcor,
                "maxls": control.maxls,
                "gtol": control.gtol,
                "ftol": control.ftol,
            },
        )
        windows += 1
        total_iterations += int(result.nit)
        result_params = np.asarray(result.x, dtype=np.float64)
        if not _same_rho(result_params, objective.accepted.params):
            raise RuntimeError(
                "L-BFGS-B returned a point without a matching accepted exact "
                "streamed trial; no optimization result is available."
            )
        gradient = np.asarray(objective.accepted.gradient, dtype=np.float64)
        projected = _projected_gradient_for_bounds(result_params, gradient, bounds)
        projected_gradient_inf = float(np.max(np.abs(projected)))
        local_projected = _projected_gradient_for_bounds(
            result_params, gradient, local_bounds
        )
        local_gradient_inf = float(np.max(np.abs(local_projected)))
        # A source-sized window is a step cap, not a parameter constraint.
        # Continue from an accepted artificial edge whenever descent points
        # beyond it, even if another free coordinate has not reached gtol.
        # A true global bound must never be treated as a reason to reopen.
        artificial_edge_descent = any(
            (
                local_lower is not None
                and (global_lower is None or local_lower > global_lower)
                and value <= local_lower
                and derivative > control.gtol
            )
            or (
                local_upper is not None
                and (global_upper is None or local_upper < global_upper)
                and value >= local_upper
                and derivative < -control.gtol
            )
            for value, derivative, (local_lower, local_upper), (
                global_lower,
                global_upper,
            ) in zip(result_params, gradient, local_bounds, bounds, strict=True)
        )
        if not (
            max_step is not None
            and bool(result.success)
            and (local_gradient_inf <= control.gtol or artificial_edge_descent)
            and projected_gradient_inf > control.gtol
            and not _same_rho(result_params, center)
            and total_iterations < control.max_iter
            and objective.n_evaluations < control.maxfun
        ):
            break
    result_params = np.asarray(result.x, dtype=np.float64)
    if not _same_rho(result_params, objective.accepted.params):
        raise RuntimeError(
            "L-BFGS-B returned a point without a matching accepted exact "
            "streamed trial; no optimization result is available."
        )
    objective._check_source_unchanged()
    gradient = np.asarray(objective.accepted.gradient, dtype=np.float64)
    projected = _projected_gradient_for_bounds(result_params, gradient, bounds)
    projected_gradient_inf = float(np.max(np.abs(projected)))
    converged = bool(result.success) and projected_gradient_inf <= control.gtol
    message = str(result.message)
    if max_step is not None:
        message = f"{message}; {windows} source step window(s)."
    if (
        roundoff_stationary_completion
        and bool(result.success)
        and not converged
        and "RELATIVE REDUCTION OF F" in message
        and stationary_trial is not None
        and not _same_rho(stationary_trial.params, objective.accepted.params)
    ):
        accepted = objective.accepted
        accepted_score = float(np.asarray(accepted.score))
        candidate = stationary_trial
        stationary_params = np.asarray(candidate.params, dtype=np.float64)
        candidate_score = float(np.asarray(candidate.score))
        candidate_gradient = np.asarray(candidate.gradient, dtype=np.float64)
        source_fields = (
            "source_fingerprint",
            "basis_fingerprint",
            "family_name",
            "link_name",
        )
        if any(
            getattr(candidate, field) != getattr(accepted, field)
            for field in source_fields
        ):
            raise RuntimeError("Stationary completion changed source/family lineage.")
        if candidate.source_fingerprint != objective._stream.source.fingerprint():
            raise RuntimeError("Stationary completion source changed after evaluation.")
        for value, (lower, upper) in zip(stationary_params, bounds, strict=True):
            if (lower is not None and value < lower) or (
                upper is not None and value > upper
            ):
                raise RuntimeError("Stationary completion violated a parameter bound.")
        candidate_projected = _projected_gradient_for_bounds(
            stationary_params, candidate_gradient, bounds
        )
        candidate_norm = float(np.max(np.abs(candidate_projected)))
        candidate_inner_valid = inner_valid(candidate)
        score_ulp = float(np.spacing(abs(accepted_score)))
        comparison = None
        if (
            local_score_change is not None
            and candidate_norm <= control.gtol
            and candidate_inner_valid
            and abs(candidate_score - accepted_score) > score_ulp
        ):
            # The callback can replay one trial at a time.  Release SciPy's
            # last rejected candidate first; accepted and stationary states
            # remain owned by this scope, so peak retention stays at three.
            objective.candidate = None
            comparison = local_score_change(
                objective, accepted, candidate, completion_inner_tolerance
            )
        comparison_ok = bool(
            comparison is not None
            and np.isfinite(comparison.change)
            and np.isfinite(comparison.error_estimate)
            and np.isfinite(comparison.raw_change)
            and np.isfinite(comparison.roundoff_bound)
            and np.isfinite(comparison.linearity_error)
            and comparison.error_estimate >= 0.0
            and comparison.roundoff_bound >= 0.0
            and comparison.change + comparison.error_estimate < 0.0
            and abs(comparison.change) + comparison.error_estimate <= score_ulp
            and abs(comparison.raw_change - comparison.change)
            <= comparison.roundoff_bound
        )
        if (
            np.isfinite(candidate_score)
            and (abs(candidate_score - accepted_score) <= score_ulp or comparison_ok)
            and candidate_norm <= control.gtol
            and candidate_inner_valid
        ):
            # Promote through the objective so adapters observe every
            # accepted state; this releases the candidate slot as before.
            objective.candidate = candidate
            objective._promote_candidate()
            projected_gradient_inf = candidate_norm
            converged = True
            if comparison_ok:
                message = (
                    f"{message}; retained exact stationary candidate by empirical "
                    f"local-gradient comparison {comparison.change:.3e} ± "
                    f"{comparison.error_estimate:.3e} under the one-ULP "
                    f"comparison gate; raw source score change "
                    f"{comparison.raw_change:.3e} remains reported."
                )
            else:
                message = (
                    f"{message}; retained exact stationary candidate within "
                    "one score ULP after the line-search roundoff stop."
                )
        else:
            comparison_detail = (
                ""
                if comparison is None
                else (
                    f", local change {comparison.change:.3e} ± "
                    f"{comparison.error_estimate:.3e}, raw bound "
                    f"{comparison.roundoff_bound:.3e}, linearity error "
                    f"{comparison.linearity_error:.3e}"
                )
            )
            message = (
                f"{message}; stationary completion rejected: score change "
                f"{candidate_score - accepted_score:.3e} (limit {score_ulp:.3e}), "
                f"projected gradient {candidate_norm:.3e}, "
                f"inner valid {candidate_inner_valid}{comparison_detail}."
            )
    completion: _CurvatureCompletion | None = None
    if (
        curvature_stationary_completion
        and not converged
        and _lbfgsb_roundoff_stop(result)
    ):
        free_count = sum(
            not (lower is not None and upper is not None and lower == upper)
            for lower, upper in bounds
        )
        accepted = objective.accepted
        if not 1 <= free_count <= _CURVATURE_COMPLETION_MAX_FREE:
            completion = _CurvatureCompletion(
                0, "not run: free parameter count outside the completion range"
            )
        elif total_iterations >= control.max_iter:
            completion = _CurvatureCompletion(0, "not run: iteration limit")
        elif not _valid_exact_trial(accepted, completion_inner_tolerance):
            completion = _CurvatureCompletion(0, "not run: accepted trial is invalid")
        else:
            completion, total_iterations = _complete_free_curvature_newton(
                objective,
                bounds,
                control,
                iterations=total_iterations,
                max_step=max_step,
                trial_valid=lambda trial: _valid_exact_trial(
                    trial, completion_inner_tolerance
                ),
                score_roundoff=completion_score_roundoff,
            )
        objective._check_source_unchanged()
        projected_gradient_inf = _free_projected_norm(
            np.asarray(objective.accepted.params, dtype=np.float64),
            np.asarray(objective.accepted.gradient, dtype=np.float64),
            bounds,
        )
        converged = bool(
            projected_gradient_inf <= control.gtol and completion.curvature_positive
        )
        if completion.last_step_inf is None:
            proposed = "none"
        else:
            proposed = (
                f"{completion.last_step_inf:.3e} "
                f"({'accepted' if completion.last_step_accepted else 'rejected'})"
            )
        eigenvalues = ", ".join(f"{value:.6g}" for value in completion.eigenvalues)
        message = (
            f"{message}; free-parameter curvature Newton completion: "
            f"{completion.steps} accepted step(s), stopped by {completion.stop}; "
            f"last proposed step max-norm {proposed}; finite-difference "
            f"curvature eigenvalues [{eigenvalues}]"
        )
        if not converged and projected_gradient_inf <= control.gtol:
            message = f"{message}; no positive definite curvature model."
    if (
        (bool(result.success) or completion is not None)
        and not converged
        and projected_gradient_inf > control.gtol
    ):
        message = (
            f"{message}; projected gradient {projected_gradient_inf:.3e} exceeds "
            f"gtol {control.gtol:.3e}."
        )
    return _ParameterizedOptimization(
        trial=objective.accepted,
        converged=converged,
        message=message,
        status=int(result.status),
        n_iter=total_iterations,
        n_evaluations=objective.n_evaluations,
        n_accepted=objective.n_accepted,
        cumulative_source_scans=objective.cumulative_source_scans,
        cumulative_batches_scanned=objective.cumulative_batches_scanned,
        projected_gradient_inf=projected_gradient_inf,
        accepted_score_history=objective.accepted_score_history,
        cumulative_inner_iterations=objective.cumulative_inner_iterations,
        cumulative_final_refit_solves=objective.cumulative_final_refit_solves,
        cumulative_derivative_s=objective.cumulative_derivative_s,
        completion_steps=0 if completion is None else completion.steps,
        completion_stop=None if completion is None else completion.stop,
    )


# Pinned mgcv newton() (gam.fit3.r) caps each log-parameter step at
# gam.control()$newton$maxNstep = 5 and rejects a step whose quadratic-model
# error reaches qerror.thresh = 0.8.
_SCALE_NEWTON_MAX_STEP = 5.0
_SCALE_NEWTON_QERROR = 0.8
# Exact Newton steps square the relative scale error. From any point that
# passed the first-order gtol stage the float64 floor is reached in two or
# three steps; this bound only guarantees termination.
_SCALE_NEWTON_MAX_STEPS = 8


@dataclass(frozen=True)
class _ScaleNewtonCompletion:
    """Outcome of a pinned single-free-coordinate Newton completion.

    Equality compares the accepted step count and stop reason only.
    ``last_step`` is the final proposed coordinate step (``None`` when no
    step was proposed) and ``last_step_accepted`` whether it was accepted.
    """

    steps: int
    stop: str
    last_step: float | None = field(default=None, compare=False)
    last_step_accepted: bool = field(default=False, compare=False)


def _regular_score_roundoff(trial: RegularStreamREMLTrial) -> float:
    """First-order float64 envelope for one regular source score value.

    The score subtracts the O(n) penalized-deviance, saturated-likelihood and
    log-determinant terms. A raw score difference of this size cannot
    distinguish two trials; the envelope only rejects a measurable increase.
    """
    state = trial.fit_result.state
    phi = float(np.asarray(state.score_scale))
    magnitude = (
        abs(float(np.asarray(trial.score)))
        + abs(trial.fit_result.source_score.penalized_deviance) / (2.0 * phi)
        + abs(float(np.asarray(state.saturated_loglik)))
        + abs(float(np.asarray(state.coefficient_factor.logdet_hessian())))
    )
    return 8.0 * float(np.finfo(np.float64).eps) * magnitude


def _newton_model_rejection(
    predicted: float,
    change: float,
    score_old: float,
    score_new: float,
    envelope: float,
) -> str | None:
    """Apply pinned ``newton()``'s step acceptance to a measured change.

    ``gam.fit3.r::newton`` accepts a trial when ``score.change < 0`` and the
    quadratic-model error ``|pred.change - score.change| /
    max(|pred.change|, |score.change|)`` is below ``qerror.thresh = 0.8``
    (newton() also adds ``score.scale*conv.tol`` to that denominator;
    omitting it is stricter). Near the optimum the raw change is far below
    the float64 resolution of the O(n) score, so every completion passes the
    change measured from exact endpoint derivatives; the raw score may not
    rise beyond ``envelope``, the float64 envelope of both scores. Both the
    pinned single-coordinate and the free-parameter completion use this one
    test after their own curvature and gradient gates.
    """
    if not change < 0.0:
        return "no derivative-integrated descent"
    qerror = abs(predicted - change) / max(abs(predicted), abs(change))
    if not qerror < _SCALE_NEWTON_QERROR:
        return "quadratic model error"
    if not (np.isfinite(envelope) and envelope >= 0.0):
        return "invalid score envelope"
    if not score_new - score_old <= envelope:
        return "measurable score increase"
    return None


def _pinned_newton_rejection(
    accepted: _TrialT,
    trial: _TrialT,
    step: float,
    pinned: np.ndarray,
    *,
    label: str,
    curvature_old: float | None,
    curvature_new: float | None,
    integrated_change: Callable[[float, float, float, float, float], float],
    score_roundoff: Callable[[_TrialT], float],
) -> str | None:
    """Return why a pinned single-coordinate Newton trial is not acceptable.

    ``integrated_change(step, g_old, g_new, c_old, c_new)`` measures the score
    change from exact endpoint gradients (and curvatures), because near the
    optimum the raw change is far below the float64 resolution of the O(n)
    score. newton()'s descent and quadratic-model tests are applied to that
    measured change; the raw score may not rise beyond ``score_roundoff``.
    """
    trial_params = np.asarray(trial.params, dtype=np.float64)
    if not np.array_equal(trial_params[:-1], pinned):
        raise RuntimeError(f"Pinned {label} completion changed supplied rho.")
    for name in (
        "source_fingerprint",
        "basis_fingerprint",
        "family_name",
        "link_name",
    ):
        if getattr(trial, name) != getattr(accepted, name):
            raise RuntimeError(f"Pinned {label} completion changed lineage.")
    score_old = float(np.asarray(accepted.score))
    score_new = float(np.asarray(trial.score))
    gradient_old = float(np.asarray(accepted.gradient)[-1])
    gradient_new = float(np.asarray(trial.gradient)[-1])
    if (
        curvature_old is None
        or curvature_new is None
        or not np.all(
            np.isfinite(
                (
                    score_old,
                    score_new,
                    gradient_old,
                    gradient_new,
                    curvature_old,
                    curvature_new,
                )
            )
        )
    ):
        return "nonfinite trial"
    if curvature_new <= 0.0:
        return "nonpositive trial curvature"
    if abs(gradient_new) >= abs(gradient_old):
        return "no gradient reduction"
    predicted = gradient_old * step + 0.5 * curvature_old * step * step
    change = integrated_change(
        step, gradient_old, gradient_new, curvature_old, curvature_new
    )
    return _newton_model_rejection(
        predicted,
        change,
        score_old,
        score_new,
        score_roundoff(accepted) + score_roundoff(trial),
    )


def _hermite_change(
    step: float,
    gradient_old: float,
    gradient_new: float,
    curvature_old: float,
    curvature_new: float,
) -> float:
    """Cubic Hermite integral of exact endpoint gradients and curvatures.

    Its truncation error is O(step^5).
    """
    return 0.5 * step * (gradient_old + gradient_new) + (
        step * step * (curvature_old - curvature_new) / 12.0
    )


def _scale_newton_rejection(
    accepted: RegularStreamREMLTrial,
    trial: RegularStreamREMLTrial,
    step: float,
    pinned: np.ndarray,
) -> str | None:
    """Return why a pinned log-scale Newton trial is not acceptable."""
    return _pinned_newton_rejection(
        accepted,
        trial,
        step,
        pinned,
        label="log-scale",
        curvature_old=accepted.log_phi_curvature,
        curvature_new=trial.log_phi_curvature,
        integrated_change=_hermite_change,
        score_roundoff=_regular_score_roundoff,
    )


def _complete_pinned_coordinate(
    objective: _AcceptedParameterizedTrialObjective[_TrialT],
    pinned: np.ndarray,
    *,
    label: str,
    max_steps: int,
    curvature: Callable[[_TrialT], float | str | None],
    rejection: Callable[[_TrialT, _TrialT, float, float], str | None],
) -> _ScaleNewtonCompletion:
    """Continue pinned ``newton()`` on the only free (final) coordinate.

    ``curvature(accepted)`` returns the positive curvature newton() would use
    at the accepted trial, or a stop reason when none is usable. Each step is
    newton()'s ``-g/curvature``, capped at ``maxNstep``; its fully
    reconverged exact source trial is accepted only when ``rejection``
    returns ``None``. The completion stops at the first rejection, when the
    next step is no longer representable, or at ``max_steps``. At most the
    accepted and one candidate trial are retained. Rho is never optimized:
    every candidate differs from the accepted parameters only in the final
    coordinate.
    """
    steps = 0
    last_step: float | None = None
    for _ in range(max_steps):
        objective._check_source_unchanged()
        accepted = objective.accepted
        params = np.asarray(accepted.params, dtype=np.float64)
        if not np.array_equal(params[:-1], pinned):
            raise RuntimeError(f"Pinned {label} completion changed supplied rho.")
        gradient = float(np.asarray(accepted.gradient)[-1])
        value = curvature(accepted)
        if isinstance(value, str):
            return _ScaleNewtonCompletion(steps, value, last_step, steps > 0)
        if value is None or not np.isfinite(value) or value <= 0.0:
            return _ScaleNewtonCompletion(
                steps, "nonpositive curvature", last_step, steps > 0
            )
        if not np.isfinite(gradient):
            return _ScaleNewtonCompletion(
                steps, "nonfinite gradient", last_step, steps > 0
            )
        step = float(
            np.clip(-gradient / value, -_SCALE_NEWTON_MAX_STEP, _SCALE_NEWTON_MAX_STEP)
        )
        trial_params = params.copy()
        trial_params[-1] = params[-1] + step
        if trial_params[-1] == params[-1]:
            return _ScaleNewtonCompletion(steps, "resolved", last_step, steps > 0)
        last_step = step
        trial = objective._evaluate_candidate(trial_params)
        reason = rejection(accepted, trial, step, value)
        if reason is not None:
            objective.candidate = None
            return _ScaleNewtonCompletion(steps, reason, last_step, False)
        objective._promote_candidate()
        steps += 1
    return _ScaleNewtonCompletion(steps, "step limit", last_step, steps > 0)


def _complete_pinned_log_scale(
    objective: _AcceptedRegularTrialObjective,
    pinned: np.ndarray,
    *,
    max_steps: int = _SCALE_NEWTON_MAX_STEPS,
) -> _ScaleNewtonCompletion:
    """Continue pinned ``newton()`` on the only free coordinate, log scale.

    With every smoothing parameter supplied, pinned mgcv's ``gam.outer``
    calls ``newton()`` on ``log(scale)`` alone. gam.fit3 supplies the exact
    curvature ``d2lr.d2lphi = Dp/(2 phi) - ls[3] phi^2 - ls[2] phi``: for fixed
    rho the coefficients and log determinants do not depend on the scale, so
    it is the full second derivative. ``newton()`` takes full exact Newton
    steps, whose relative scale error squares at every step, and tests
    convergence only after a step; with a tight ``conv.tol`` it therefore
    returns the stationary scale to roundoff. The first-order L-BFGS-B stage
    stops at ``|g| <= gtol``, leaving a relative scale error ``|g|/curvature``.

    This completion continues newton()'s exact-curvature step from the
    accepted L-BFGS-B trial. Each candidate is a fully reconverged exact
    source trial; it is accepted only when its curvature is positive, the
    gradient magnitude decreases, and newton()'s descent and quadratic-model
    tests hold for the derivative-integrated score change. The iteration
    stops when the next correction is no longer representable in float64 log
    scale or a trial no longer reduces the exact gradient, i.e. at the
    gradient's evaluation floor. Rho is never optimized: every candidate
    differs from the accepted parameters only in the final coordinate.
    """
    return _complete_pinned_coordinate(
        objective,
        pinned,
        label="log-scale",
        max_steps=max_steps,
        curvature=lambda accepted: accepted.log_phi_curvature,
        rejection=lambda accepted, trial, step, _curvature: _scale_newton_rejection(
            accepted, trial, step, pinned
        ),
    )


def _abnormal_line_search_stop(
    optimized: _ParameterizedOptimization[_TrialT] | OptimizeResult,
) -> bool:
    """Whether L-BFGS-B ended ABNORMAL after failed line searches.

    SciPy reports ``ABNORMAL: `` (``ABNORMAL_TERMINATION_IN_LNSRCH`` before
    its C translation) when ``maxls`` evaluations fail the Moré-Thuente tests
    with no correction pairs left, and restores the last accepted iterate.
    Near an optimum these tests compare raw-score roundoff, so this stop does
    not mean the accepted trial is unusable. It reads SciPy's result and the
    restated stage result alike; every completion uses this one test.
    """
    return int(optimized.status) != 0 and str(optimized.message).startswith("ABNORMAL")


def _valid_exact_trial(trial: _TrialT, inner_tolerance: float) -> bool:
    """Require the converged inner state and finite score every trial owns."""
    inner = trial.fit_result.state
    return bool(
        inner.converged
        and not inner.line_search_failed
        and np.isfinite(inner.stationarity)
        and inner.stationarity < inner_tolerance
        and np.isfinite(float(np.asarray(trial.score)))
        and np.all(np.isfinite(np.asarray(trial.gradient)))
    )


def _complete_pinned_regular_scale(
    objective: _AcceptedRegularTrialObjective,
    optimized: _ParameterizedOptimization[RegularStreamREMLTrial],
    pinned: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    control: StreamREMLControl,
    *,
    inner_tolerance: float,
) -> tuple[_ParameterizedOptimization[RegularStreamREMLTrial], _ScaleNewtonCompletion]:
    """Apply the log-scale completion and restate the final accepted status.

    It runs after every status-0 L-BFGS-B stage, converged or not, because
    the first-order ``gtol`` leaves a relative scale error ``|g|/curvature``
    that pinned newton() would remove. It also runs after an ABNORMAL
    line-search stop from a valid accepted exact trial. It never runs after
    iteration/evaluation limits or other errors.
    """
    abnormal = _abnormal_line_search_stop(optimized)
    if optimized.status != 0 and not abnormal:
        return optimized, _ScaleNewtonCompletion(0, "not run: L-BFGS-B failed")
    if abnormal and not _valid_exact_trial(objective.accepted, inner_tolerance):
        return optimized, _ScaleNewtonCompletion(
            0, "not run: accepted trial is invalid"
        )
    # Only the log scale is free; continue pinned newton()'s exact step from
    # the L-BFGS-B result. The completion holds at most the accepted and one
    # candidate trial, as the L-BFGS-B stage does.
    completion = _complete_pinned_log_scale(objective, pinned)
    objective._check_source_unchanged()
    trial = objective.accepted
    projected_gradient_inf = float(
        np.max(
            np.abs(
                _projected_gradient_for_bounds(
                    np.asarray(trial.params, dtype=np.float64),
                    np.asarray(trial.gradient, dtype=np.float64),
                    bounds,
                )
            )
        )
    )
    curvature = trial.log_phi_curvature
    curvature_positive = bool(
        curvature is not None and np.isfinite(curvature) and curvature > 0.0
    )
    # newton() never reports convergence at an indefinite point.
    converged = projected_gradient_inf <= control.gtol and curvature_positive
    message = (
        f"{optimized.message}; pinned log-scale Newton completion: "
        f"{completion.steps} exact-curvature step(s), stopped by "
        f"{completion.stop}; projected gradient {projected_gradient_inf:.3e}, "
        f"log-scale curvature "
        f"{float('nan') if curvature is None else curvature:.10g}"
    )
    if not curvature_positive:
        message = f"{message}; log-scale curvature is not positive."
    elif not converged:
        message = f"{message}; projected gradient exceeds gtol {control.gtol:.3e}."
    return (
        replace(
            optimized,
            trial=trial,
            converged=converged,
            message=message,
            n_evaluations=objective.n_evaluations,
            n_accepted=objective.n_accepted,
            cumulative_source_scans=objective.cumulative_source_scans,
            cumulative_batches_scanned=objective.cumulative_batches_scanned,
            projected_gradient_inf=projected_gradient_inf,
            accepted_score_history=objective.accepted_score_history,
            cumulative_inner_iterations=objective.cumulative_inner_iterations,
            cumulative_final_refit_solves=objective.cumulative_final_refit_solves,
            cumulative_derivative_s=objective.cumulative_derivative_s,
        ),
        completion,
    )


def optimize_regular_stream_reml(
    stream: StreamDesign,
    family: ExponentialFamily,
    initial_params: np.ndarray | jax.Array,
    *,
    maximum_bytes: int,
    pin_lambda: bool = False,
    pirls_control: StreamPIRLSControl | None = None,
    control: StreamREMLControl | None = None,
    initial_coefficients: np.ndarray | None = None,
    max_step: float | None = None,
    device: jax.Device | None = None,
) -> RegularStreamREMLOptimization:
    """Optimize the exact regular source objective with bounded L-BFGS-B.

    Unknown-scale families jointly optimize ``[rho, log_phi]``. With
    ``pin_lambda=True`` the supplied rho is held exactly, including outside
    the estimated-parameter bounds, while phi remains free. Every accepted
    point owns a fully reconverged source trial.
    """
    reml_control = StreamREMLControl() if control is None else control
    _validate_max_step(max_step)
    if (
        not isinstance(maximum_bytes, numbers.Integral)
        or isinstance(maximum_bytes, bool)
        or maximum_bytes <= 0
    ):
        raise ValueError("maximum_bytes must be a positive integer")
    maximum_bytes = int(maximum_bytes)
    fitting = stream.prepared.fitting
    if fitting is None:
        raise ValueError("Regular streamed REML requires fitting preparation.")
    n_lambda = fitting.penalty_structure.n_penalties
    if not isinstance(pin_lambda, bool):
        raise ValueError("pin_lambda must be bool")
    if n_lambda == 0 and family.scale_known:
        raise NotImplementedError(
            "Regular streamed REML needs at least one free parameter."
        )
    n_params = n_lambda + int(not family.scale_known)
    params_initial = np.asarray(initial_params, dtype=np.float64)
    if params_initial.shape != (n_params,):
        raise ValueError(
            f"initial_params must have shape ({n_params},), got {params_initial.shape}."
        )
    if not np.all(np.isfinite(params_initial)):
        raise ValueError("initial_params must contain only finite values.")

    lower, upper = -40.0, 40.0
    params_initial = params_initial.copy()
    if not pin_lambda:
        params_initial[:n_lambda] = np.clip(params_initial[:n_lambda], lower, upper)
    base_pirls_control = (
        StreamPIRLSControl(solver_policy="qr")
        if pirls_control is None
        else pirls_control
    )
    if base_pirls_control.solver_policy != "qr":
        raise ValueError("Regular streamed REML requires solver_policy='qr'.")
    pirls_control = replace(
        base_pirls_control,
        tol=min(base_pirls_control.tol, 1e-9, reml_control.gtol / 100.0),
    )

    p = stream.prepared.n_coef
    # One retained accepted result coexists with the active candidate and
    # SciPy's bounded correction/history work. Opaque native scratch follows
    # the existing workspace convention.
    retained_trial_bytes, retained_history_bytes, lbfgs_workspace_bytes = (
        _parameterized_optimizer_workspace_bytes(
            p, n_params, reml_control, pirls_control
        )
    )
    # With free smoothing parameters and at most four free coordinates, an
    # L-BFGS-B roundoff stop continues newton() through the free-parameter
    # curvature completion. Beside the accepted and candidate trials it holds
    # the m-by-m curvature model, its eigenvalue/solve copies (2m^2) and about
    # fourteen m-vectors of parameters, gradients, probes and steps; for
    # m <= 4, 3m^2 + 14m <= m^2 + 24m.
    curvature_completion = (
        not pin_lambda and n_lambda >= 1 and n_params <= _CURVATURE_COMPLETION_MAX_FREE
    )
    curvature_completion_bytes = (
        8 * (n_params * n_params + 24 * n_params) if curvature_completion else 0
    )
    outer_workspace_bytes = (
        retained_trial_bytes
        + retained_history_bytes
        + lbfgs_workspace_bytes
        + curvature_completion_bytes
    )
    if outer_workspace_bytes >= maximum_bytes:
        raise MemoryError(
            "Regular streamed REML optimizer retention needs "
            f"{outer_workspace_bytes} bytes before a trial, exceeding "
            f"maximum_bytes={maximum_bytes}."
        )
    trial_budget = maximum_bytes - outer_workspace_bytes
    objective = _AcceptedRegularTrialObjective(
        stream,
        family,
        params_initial,
        pirls_control,
        reml_control,
        trial_budget,
        initial_coefficients,
        device,
    )
    bounds = (
        [(float(value), float(value)) for value in params_initial[:n_lambda]]
        if pin_lambda
        else [(lower, upper)] * n_lambda
    ) + [(None, None)] * (n_params - n_lambda)
    step_options = {} if max_step is None else {"max_step": max_step}
    optimized = _run_parameterized_stream_reml(
        objective,
        params_initial,
        bounds,
        reml_control,
        curvature_stationary_completion=curvature_completion,
        completion_inner_tolerance=(
            pirls_control.tol if curvature_completion else None
        ),
        completion_score_roundoff=_regular_score_roundoff,
        **step_options,
    )
    completion = None
    if not family.scale_known and (pin_lambda or n_lambda == 0):
        optimized, completion = _complete_pinned_regular_scale(
            objective,
            optimized,
            params_initial[:n_lambda],
            bounds,
            reml_control,
            inner_tolerance=pirls_control.tol,
        )
    return RegularStreamREMLOptimization(
        trial=optimized.trial,
        converged=optimized.converged,
        message=optimized.message,
        status=optimized.status,
        n_iter=optimized.n_iter,
        n_evaluations=optimized.n_evaluations,
        n_accepted=optimized.n_accepted,
        cumulative_source_scans=optimized.cumulative_source_scans,
        cumulative_batches_scanned=optimized.cumulative_batches_scanned,
        projected_gradient_inf=optimized.projected_gradient_inf,
        accepted_score_history=optimized.accepted_score_history,
        outer_workspace_bytes=outer_workspace_bytes,
        scale_completion_steps=0 if completion is None else completion.steps,
        scale_completion_stop=None if completion is None else completion.stop,
        cumulative_inner_iterations=optimized.cumulative_inner_iterations,
        cumulative_final_refit_solves=optimized.cumulative_final_refit_solves,
        cumulative_derivative_s=optimized.cumulative_derivative_s,
        curvature_completion_steps=optimized.completion_steps,
        curvature_completion_stop=optimized.completion_stop,
    )
