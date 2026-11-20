"""Bounded, lineage-checked regular and NB accepted/trial EFS fits.

These adapters consume explicit controller requests. Streamed initial-sp/scale
preparation and public dispatch compose around the shared provider protocol;
the adapters do not materialize training rows or change the reviewed outer
or coefficient-controller policies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.control import EFSControl
from jaxgam.execution.efs import EFSFitState
from jaxgam.execution.efs_provider import EFSControllerContext, EFSFitRequest
from jaxgam.execution.nb_stream import NBStreamResult, fit_nb_streamed_pirls
from jaxgam.execution.regular_stream import (
    fit_regular_streamed_pirls,
    preflight_regular_stream_workspace,
)
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.base import ExponentialFamily
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.fitting.efs import EFSStatisticsPlan, prepare_efs_statistics
from jaxgam.fitting.efs_factor import efs_factor_statistics
from jaxgam.fitting.family_execution import (
    FamilyExecutionLineage,
    FamilyExecutionParameters,
)
from jaxgam.fitting.reml import reml_criterion_from_penalized_deviance
from jaxgam.formula.design_provider import StreamDesign

_statistics = jax.jit(efs_factor_statistics)
_score = partial(jax.jit, static_argnums=(6, 7, 8, 10, 11, 13))(
    reml_criterion_from_penalized_deviance
)


def _vector(value: jax.Array, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64 or array.shape != shape:
        raise ValueError(f"{name} must be float64 with shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _check_nb_attribution(
    result: NBStreamResult,
    request: EFSFitRequest,
    log_theta: np.ndarray,
    *,
    estimated_theta: bool,
) -> None:
    """Reject stale result parameters before they can yield valid=True.

    Source stopping deviance intentionally belongs to the theta entering the
    last PIRLS iteration, whereas factors/likelihood use its outgoing theta.
    The bounded transition history makes that distinction verifiable.
    """
    if not np.array_equal(result.state.log_lambda, request.log_lambda):
        raise RuntimeError("NB provider returned a misattributed rho")
    if float(np.asarray(result.state.score_scale)) != 1.0:
        raise RuntimeError("NB provider returned a misattributed score_phi")
    final = np.asarray(result.log_theta)
    generating = np.asarray(result.source_deviance_log_theta)
    if estimated_theta:
        history = np.asarray(result.theta_history)
        if (
            history.ndim != 1
            or not 1 <= history.size <= result.state.n_iter + 1
            or not np.array_equal(history[:1], log_theta)
            or not np.array_equal(history[-1:], final)
        ):
            raise RuntimeError("NB provider returned a misattributed theta transition")
        if result.state.converged:
            generating_valid = (
                history.size == result.state.n_iter + 1
                and np.array_equal(history[-2:-1], generating)
            )
        else:
            # A coefficient safeguard can stop before conditional theta; a
            # continuing preceding iteration already refreshed its objective.
            generating_valid = any(
                np.array_equal(history[index : index + 1], generating)
                for index in range(max(0, history.size - 2), history.size)
            )
        if not generating_valid:
            raise RuntimeError("NB provider returned a misattributed generating theta")
    elif not (
        np.array_equal(final, log_theta) and np.array_equal(generating, log_theta)
    ):
        raise RuntimeError("NB provider returned a misattributed fixed theta")


def _provider_memory_bounds(stream: StreamDesign) -> tuple[int, int, int, int]:
    """Known descriptor-only bounds, before copies/transfers/eigendecompositions.

    Full local squares bound compressed structure values, projected matrices
    and roots without determining their numerical ranks. The metadata builder
    also constructs temporary dense penalties and unused transferred local
    matrices; charge those plus its largest-block explicit scratch separately.
    Native eigensolver scratch remains an explicitly excluded measurement.
    """
    prepared = stream.prepared
    if prepared.fitting is None:
        raise ValueError("EFS provider requires fitting preparation")
    p, m = prepared.n_coef, prepared.fitting.penalty_structure.n_penalties
    persistent_entries = 2 * m
    expanded_entries = largest_scratch = 0
    for block in prepared.fitting.penalty_structure.blocks:
        k, b = block.size, len(block.local_penalties)
        square = max(k * k, 1)
        persistent_entries += (2 * b + 1 + (b if b > 1 else 0)) * square
        expanded_entries += b * square
        largest_scratch = max(largest_scratch, (6 * b + 12) * square + 32 * k)
    persistent = 8 * persistent_entries
    preparation = 8 * (2 * expanded_entries + largest_scratch)
    # Per prior fit: rho+StreamFitState rho (2m), d/t/q (3m), possible
    # numerator/ratio/trial (3m). Another 8m covers live outer proposals.
    # Round up vector/index/scalar leaves; no conditional histories survive
    # in an EFSFitState after the NB provider drops its local NBStreamResult.
    prior_and_outer = 8 * (24 * p * p + 192 * p + 32 * m + 192)
    trace = 16 * p * min(32, p)
    return persistent, preparation, prior_and_outer, trace


def _provider_parts(
    stream: StreamDesign,
    family: ExponentialFamily,
    maximum_bytes: int,
    batch_rows: int,
    control: EFSControl | None,
    device: jax.Device | None,
) -> tuple[object, ...]:
    """Prepare shared bounded metadata without fitting or scanning rows."""
    if (
        not isinstance(maximum_bytes, Integral)
        or isinstance(maximum_bytes, bool)
        or maximum_bytes <= 0
    ):
        raise ValueError("maximum_bytes must be a positive integer")
    control = EFSControl() if control is None else control
    if not isinstance(control, EFSControl):
        raise TypeError("control must be EFSControl")
    coefficient_control = StreamPIRLSControl(
        batch_rows=batch_rows,
        max_iter=control.pirls_max_iter,
        tol=control.pirls_tolerance,
        solver_policy="qr",
    )
    lineage = FamilyExecutionLineage.from_prepared(stream.prepared, family)
    if stream.prepared.fitting.unpenalized_rank_deficit:
        raise np.linalg.LinAlgError("EFS provider requires identifiable null space")
    persistent_upper, preparation_bytes, prior_fit_bytes, trace_bytes = (
        _provider_memory_bounds(stream)
    )
    # Both score/phi deques and their returned tuple copies may overlap.
    # Charge Python scalar/container overhead separately from numeric leaves.
    history_entries = min(control.outer_limit, max(4, control.history_limit))
    outer_history_bytes = 128 * (history_entries + 4) + 1024
    provider_bytes = (
        persistent_upper + prior_fit_bytes + trace_bytes + outer_history_bytes
    )
    if max(persistent_upper + preparation_bytes, provider_bytes) >= maximum_bytes:
        raise MemoryError("EFS provider known buffers exceed maximum_bytes")
    coefficient_ledger = preflight_regular_stream_workspace(
        stream, coefficient_control, int(maximum_bytes) - provider_bytes
    )
    if lineage.parameters.theta_mode == "estimated":
        theta_known_bytes = 8 * 64 * coefficient_ledger.batch_rows + 128 * (
            100 + coefficient_control.max_iter + 2
        )
        if (
            coefficient_ledger.required_bytes + provider_bytes + theta_known_bytes
            > maximum_bytes
        ):
            raise MemoryError(
                "EFS provider conditional-theta workspace exceeds maximum_bytes"
            )
    frozen_family = copy.deepcopy(family)
    fitting = PreparedFittingMetadata.from_prepared(
        stream.prepared, frozen_family, device
    )
    plan = prepare_efs_statistics(fitting)
    metadata_arrays = jax.tree_util.tree_leaves(
        (
            fitting.penalty_structure,
            fitting.log_lambda_init,
            fitting.singleton_eig_constants,
            fitting.multi_block_proj_S,
            plan,
        )
    )
    # Projected penalty matrices are shared by plan/metadata. Count those
    # exact objects once, and charge the newly transferred metadata too.
    persistent_bytes = sum(
        int(leaf.nbytes)
        for leaf in {id(value): value for value in metadata_arrays}.values()
    )
    if persistent_bytes > persistent_upper:
        raise MemoryError("EFS provider actual metadata exceeds prospective bound")
    context = EFSControllerContext(
        estimated_theta=lineage.parameters.theta_mode == "estimated",
        fixed_theta=(
            float(np.exp(lineage.parameters.log_theta[0]))
            if lineage.parameters.theta_mode == "fixed"
            else None
        ),
        reference_profile="mgcv-1.9-3-efsudr-streamed",
        trace_method="exact-streamed-fisher",
        source_fingerprint=lineage.source_fingerprint,
        basis_fingerprint=lineage.basis_fingerprint,
    )
    return (
        stream,
        frozen_family,
        fitting,
        plan,
        lineage,
        context,
        coefficient_control,
        int(maximum_bytes),
        provider_bytes,
        persistent_upper,
        persistent_bytes,
        preparation_bytes,
        outer_history_bytes,
        device,
    )


@dataclass(frozen=True)
class NBStreamEFSProvider:
    """A replayable fit closure with immutable lineage and bounded workspace.

    Three earlier compact EFS fits may remain alive during an extension fit.
    Reserve 24p²+192p float64 entries for their known numeric leaves, one blocked trace
    RHS/action pair, and this adapter's retained local penalty roots. These
    buffers are charged in addition to the coefficient controller ledger.
    Source/prepared storage and opaque native/XLA scratch are excluded.
    """

    stream: StreamDesign
    family: NegativeBinomial
    fitting: PreparedFittingMetadata
    plan: EFSStatisticsPlan
    lineage: FamilyExecutionLineage
    context: EFSControllerContext
    control: StreamPIRLSControl
    maximum_bytes: int
    provider_bytes: int
    persistent_upper_bytes: int
    persistent_bytes: int
    preparation_workspace_bytes: int
    outer_history_bytes: int
    device: jax.Device | None = None

    @classmethod
    def create(
        cls,
        stream: StreamDesign,
        family: NegativeBinomial,
        *,
        maximum_bytes: int,
        batch_rows: int = 8192,
        control: EFSControl | None = None,
        device: jax.Device | None = None,
    ) -> NBStreamEFSProvider:
        if not isinstance(family, NegativeBinomial):
            raise TypeError("NB EFS provider requires NegativeBinomial")
        return cls(
            *_provider_parts(stream, family, maximum_bytes, batch_rows, control, device)
        )

    def __call__(self, request: EFSFitRequest) -> EFSFitState:
        self.lineage.validate(self.stream.prepared, self.family)
        if self.stream.source.fingerprint() != self.lineage.source_fingerprint:
            raise RuntimeError("RowSource changed before EFS provider dispatch")
        if (
            self.fitting.source_fingerprint != self.context.source_fingerprint
            or self.fitting.basis_fingerprint != self.context.basis_fingerprint
        ):
            raise RuntimeError("EFS provider fitting metadata has stale lineage")
        rho = _vector(request.log_lambda, (self.fitting.n_penalties,), "rho")
        beta = _vector(request.beta_start, (self.fitting.n_coef,), "beta_start")
        old = None
        if request.beta_old_init is not None:
            old = _vector(request.beta_old_init, beta.shape, "beta_old_init")
        if request.score_phi is not None:
            phi = _vector(request.score_phi, (), "score_phi")
            if float(phi) != 1.0:
                raise ValueError("NB provider score_phi must be one")
        estimated = self.context.estimated_theta
        if estimated and (request.log_theta_start is None or old is None):
            raise ValueError("estimated NB provider requires theta and old beta anchor")
        theta = (
            np.asarray(self.lineage.parameters.log_theta, dtype=np.float64)
            if request.log_theta_start is None
            else _vector(request.log_theta_start, (1,), "log_theta_start")
        )
        if not estimated and not np.array_equal(
            theta, self.lineage.parameters.log_theta
        ):
            raise ValueError("fixed NB provider requires the frozen family theta")
        # Requests are normally immutable JAX arrays. Preserve that contract
        # even for an internal caller supplying mutable NumPy arrays.
        request = replace(
            request,
            log_lambda=jax.device_put(np.array(rho, copy=True), self.device),
        )
        result = fit_nb_streamed_pirls(
            self.stream,
            self.family,
            rho,
            maximum_bytes=self.maximum_bytes - self.provider_bytes,
            parameters=FamilyExecutionParameters(jax.device_put(theta, self.device)),
            control=self.control,
            device=self.device,
            beta_start=beta,
            beta_old_init=old,
            start_is_absent=request.start_is_absent,
            estimate_theta=estimated,
        )
        self.lineage.validate(self.stream.prepared, self.family)
        if self.stream.source.fingerprint() != self.lineage.source_fingerprint:
            raise RuntimeError("RowSource changed during EFS provider fit")
        _check_nb_attribution(result, request, theta, estimated_theta=estimated)
        statistics = _statistics(
            self.plan,
            result.state.coefficients,
            result.state.fisher_coefficient_factor,
            request.log_lambda,
        )
        valid = bool(
            result.state.converged
            and result.theta_status == 0
            and np.isfinite(result.reml_score)
            and np.all(np.isfinite(result.state.coefficients))
            and np.asarray(statistics.input_valid)
        )
        one = jnp.asarray(1.0, dtype=jnp.float64)
        return EFSFitState(
            log_lambda=request.log_lambda,
            pirls_result=result.state,
            score=jnp.asarray(result.reml_score),
            edf=result.state.edf,
            statistics=statistics,
            raw_update=None,
            valid=valid,
            inner_converged=result.state.converged,
            score_phi=one,
            update_phi=one,
            reported_phi=one,
            carried_phi=one,
            log_theta=jnp.asarray(result.log_theta) if estimated else None,
            theta_status=jnp.asarray(result.theta_status) if estimated else None,
            theta_n_iter=jnp.asarray(result.theta_n_iter) if estimated else None,
            stopping_penalized_deviance=jnp.asarray(result.stopping_penalized_deviance)
            if estimated
            else None,
            positive_curvature_retry_count=jnp.asarray(
                result.positive_observed_recoveries
            ),
            source_scans=result.state.source_scans,
            batches_scanned=result.state.batches_scanned,
        )


@dataclass(frozen=True)
class RegularStreamEFSProvider(NBStreamEFSProvider):
    """Source regular fits with separate incoming and reported phi.

    The inherited storage contract contains only replayable rows, local
    penalty metadata and small immutable context. Regular and extended NB
    retain their distinct already-reviewed coefficient recovery policies.
    """

    family: ExponentialFamily
    source_null_coefficients: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        stream: StreamDesign,
        family: ExponentialFamily,
        *,
        maximum_bytes: int,
        batch_rows: int = 8192,
        control: EFSControl | None = None,
        device: jax.Device | None = None,
    ) -> RegularStreamEFSProvider:
        if not isinstance(family, ExponentialFamily):
            raise TypeError("regular EFS provider requires ExponentialFamily")
        if family.execution_parameter_snapshot().theta_mode != "none":
            raise ValueError("regular provider requires no nuisance theta")
        return cls(
            *_provider_parts(stream, family, maximum_bytes, batch_rows, control, device)
        )

    def __call__(self, request: EFSFitRequest) -> EFSFitState:
        self.lineage.validate(self.stream.prepared, self.family)
        if self.stream.source.fingerprint() != self.lineage.source_fingerprint:
            raise RuntimeError("RowSource changed before EFS provider dispatch")
        if (
            self.fitting.source_fingerprint != self.context.source_fingerprint
            or self.fitting.basis_fingerprint != self.context.basis_fingerprint
        ):
            raise RuntimeError("EFS provider fitting metadata has stale lineage")
        rho = _vector(request.log_lambda, (self.fitting.n_penalties,), "rho")
        beta = _vector(request.beta_start, (self.fitting.n_coef,), "beta_start")
        if request.log_theta_start is not None or request.beta_old_init is not None:
            raise ValueError("regular provider does not accept nuisance-theta anchors")
        if request.score_phi is None:
            if not self.family.scale_known:
                raise ValueError("unknown-scale provider requires explicit score_phi")
            phi = 1.0
        else:
            phi = float(_vector(request.score_phi, (), "score_phi"))
        if phi <= 0.0 or (self.family.scale_known and phi != 1.0):
            raise ValueError("positive score_phi must match the family scale mode")
        present = request.regular_start_present
        if present is not None and not isinstance(present, bool):
            raise ValueError("regular_start_present must be bool")
        rho_device = jax.device_put(np.array(rho, copy=True), self.device)
        result = fit_regular_streamed_pirls(
            self.stream,
            self.family,
            rho,
            maximum_bytes=self.maximum_bytes - self.provider_bytes,
            score_scale=phi,
            control=self.control,
            device=self.device,
            initial_coefficients=None if present is False else beta,
            null_coefficients=self.source_null_coefficients,
        )
        self.lineage.validate(self.stream.prepared, self.family)
        if self.stream.source.fingerprint() != self.lineage.source_fingerprint:
            raise RuntimeError("RowSource changed during EFS provider fit")
        if not np.array_equal(result.state.log_lambda, rho):
            raise RuntimeError("regular provider returned a misattributed rho")
        source = result.source_score
        if (
            source.score_phi != phi
            or float(np.asarray(result.state.score_scale)) != phi
        ):
            raise RuntimeError("regular provider returned a misattributed score_phi")
        if source.reported_phi != float(np.asarray(result.state.scale)):
            raise RuntimeError("regular provider returned a misattributed reported_phi")
        statistics = _statistics(
            self.plan,
            result.state.coefficients,
            result.state.fisher_coefficient_factor,
            rho_device,
        )
        score = _score(
            rho_device,
            result.state.xtwx,
            jnp.asarray(source.penalized_deviance),
            result.state.saturated_loglik,
            self.fitting.penalty_structure,
            jnp.asarray(phi),
            self.fitting.total_penalty_null_dim,
            self.fitting.singleton_sp_indices,
            self.fitting.singleton_ranks,
            self.fitting.singleton_eig_constants,
            self.fitting.multi_block_sp_indices,
            self.fitting.multi_block_ranks,
            self.fitting.multi_block_proj_S,
            self.fitting.rank_deficit,
            log_det_hessian=result.state.coefficient_factor.logdet_hessian(),
        )
        valid = bool(
            result.state.converged
            and np.isfinite(score)
            and np.isfinite(source.penalized_deviance)
            and np.isfinite(source.reported_phi)
            and source.reported_phi > 0.0
            and np.asarray(statistics.input_valid)
        )
        return EFSFitState(
            log_lambda=rho_device,
            pirls_result=result.state,
            score=score,
            edf=result.state.edf,
            statistics=statistics,
            raw_update=None,
            valid=valid,
            inner_converged=result.state.converged,
            score_phi=jnp.asarray(phi),
            update_phi=jnp.asarray(source.reported_phi),
            reported_phi=jnp.asarray(source.reported_phi),
            carried_phi=jnp.asarray(source.reported_phi),
            pre_gdi1_deviance=jnp.asarray(source.raw_deviance),
            pre_gdi1_penalized_deviance=jnp.asarray(source.stopping_penalized_deviance),
            gdi1_penalty=jnp.asarray(source.solve_penalty),
            gdi1_candidate_valid=jnp.asarray(source.candidate_valid),
            positive_curvature_retry_count=jnp.asarray(result.fisher_recoveries),
            source_scans=result.state.source_scans,
            batches_scanned=result.state.batches_scanned,
        )
