"""Host replay driver for exact known-scale streamed REML gradients."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.base import ExponentialFamily
from jaxgam.families.standard import Binomial, Poisson
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.fitting.state import StreamFitState
from jaxgam.fitting.stream_reml import (
    batch_is_adjoint_interior,
    batch_statistics_beta_vjp,
    reml_score_cotangents,
    solve_observed_adjoint,
)
from jaxgam.formula.design_provider import StreamDesign

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


def _preflight(
    stream: StreamDesign,
    family: ExponentialFamily,
    rho: np.ndarray | jax.Array,
    warm_start: StreamREMLTrial | None,
    device: jax.Device | None,
) -> tuple[jax.Array, PreparedFittingMetadata]:
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
    return rho_array, metadata


def _check_interior(
    stream: StreamDesign,
    beta: jax.Array,
    family: ExponentialFamily,
    batch_rows: int,
    device: jax.Device | None,
) -> tuple[int, int]:
    """Replay rows to reject active working-weight clipping before a VJP."""
    batches = 0
    for X, _y, weight, offset, _valid in _fitting_batches(stream, batch_rows):
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
    rho_array, metadata = _preflight(stream, family, rho, warm_start, device)
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
        stream, state.coefficients, family, control.batch_rows, device
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
    for X, y, weight, offset, _valid in _fitting_batches(stream, control.batch_rows):
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
