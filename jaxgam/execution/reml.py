"""Host replay driver for exact known-scale streamed REML gradients."""

from __future__ import annotations

import numbers
from collections import deque
from dataclasses import dataclass, replace

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
    """One source-timed regular score and exact first derivative."""

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
    # This charge is prospective and excludes opaque XLA/BLAS scratch under
    # the same documented numerical-workspace contract as the controller.
    adjoint_bytes = 8 * (2 * B * p + 4 * p * p + 18 * p + 8 * n_params + 16 * B)
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
    )


def _same_rho(left: np.ndarray, right: jax.Array | np.ndarray) -> bool:
    """Compare host smoothing vectors exactly at an optimizer boundary."""
    return np.array_equal(left, np.asarray(right, dtype=np.float64))


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
    )
