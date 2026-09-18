"""Row-free composition of source startup, fit providers and the EFS loop."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.control import EFSControl
from jaxgam.execution.efs import (
    EFSResult,
    _run_efs_known_scale,
    _run_efs_unknown_scale,
)
from jaxgam.execution.efs_stream_provider import (
    NBStreamEFSProvider,
    RegularStreamEFSProvider,
    _provider_memory_bounds,
)
from jaxgam.execution.efs_stream_start import (
    EFSStreamStartup,
    _startup_workspace_bytes,
    prepare_stream_efs_start,
)
from jaxgam.families.base import ExponentialFamily
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.formula.design_provider import StreamDesign


@dataclass(frozen=True)
class StreamEFSExecution:
    """One consumed fit and explicit initial/accepted family provenance.

    Metadata retains its initial family snapshot. ``family`` is an isolated
    reporting copy with selected theta synchronized only after fitting.
    Neither object owns training rows or the replayable source.
    """

    fit: EFSResult
    metadata: PreparedFittingMetadata
    family: ExponentialFamily
    startup: EFSStreamStartup
    retained_startup_bytes: int


def fit_streamed_efs(
    stream: StreamDesign,
    family: ExponentialFamily,
    *,
    maximum_bytes: int,
    batch_rows: int = 8192,
    control: EFSControl | None = None,
    device: jax.Device | None = None,
) -> StreamEFSExecution:
    """Run EFS with conditional theta inside streamed coefficient PIRLS.

    This internal fitting-boundary function does not change public dispatch
    or the ordinary/fixed-sp routes. Source preparation is caller-owned;
    startup and all actually executed provider trials have separate counts.
    """
    if (
        not isinstance(maximum_bytes, Integral)
        or isinstance(maximum_bytes, bool)
        or maximum_bytes <= 0
    ):
        raise ValueError("maximum_bytes must be a positive integer")
    control = EFSControl() if control is None else control
    if not isinstance(control, EFSControl):
        raise TypeError("control must be EFSControl")
    prepared = stream.prepared
    if prepared.fitting is None:
        raise ValueError("Streamed EFS requires fitting preparation")
    if prepared.penalties is None or prepared.penalties.n_penalties == 0:
        raise ValueError("Parametric bypass precedes streamed EFS dispatch")
    if (
        not isinstance(batch_rows, Integral)
        or isinstance(batch_rows, bool)
        or batch_rows <= 0
    ):
        raise ValueError("batch_rows must be a positive integer")
    # Immutable startup CPU arrays and device first-request arrays remain
    # alive across fitting. They are additional to provider trial ownership.
    retained_startup = 8 * (
        2 * prepared.n_coef + 3 * prepared.penalties.n_penalties + 32
    )
    if retained_startup >= maximum_bytes:
        raise MemoryError("Streamed EFS startup retention exceeds maximum_bytes")
    # Reject the combined startup/provider lifetime before metadata transfer
    # or penalty-root construction, using only already-prepared descriptors.
    persistent, _preparation, prior_fits, trace = _provider_memory_bounds(stream)
    history_entries = min(control.outer_limit, max(4, control.history_limit))
    history = 128 * (history_entries + 4) + 1024
    startup_workspace = _startup_workspace_bytes(
        stream, min(int(batch_rows), prepared.n_obs)
    )
    if (
        retained_startup + persistent + prior_fits + trace + history + startup_workspace
        > maximum_bytes
    ):
        raise MemoryError(
            "Streamed EFS startup/provider workspace exceeds maximum_bytes"
        )
    cls = (
        NBStreamEFSProvider
        if isinstance(family, NegativeBinomial)
        else RegularStreamEFSProvider
    )
    provider = cls.create(
        stream,
        family,
        maximum_bytes=int(maximum_bytes) - retained_startup,
        batch_rows=batch_rows,
        control=control,
        device=device,
    )
    startup = prepare_stream_efs_start(provider)
    # Include the selected device for loop-created constants as well as
    # first-request arrays; CPU reductions keep their existing phase seam.
    selected_device = jax.devices()[0] if device is None else device
    with jax.default_device(selected_device):
        initial = startup.initial_request(
            estimated_theta=provider.context.estimated_theta
        )
        fit_provider = provider
        if isinstance(provider.family, NegativeBinomial):
            # mgcv.r's EFS dispatch does not forward G$null.coef. gam.fit4's
            # default recovery anchor is zero on every first/refit call;
            # get.null.coef's projection remains separate startup provenance.
            source_null = jnp.zeros_like(initial.beta_start)
            initial = replace(
                initial, beta_start=source_null, beta_old_init=source_null
            )

            def fit_provider(request):
                return provider(replace(request, beta_old_init=source_null))

        run = (
            _run_efs_known_scale
            if provider.family.scale_known
            else _run_efs_unknown_scale
        )
        fit = run(fit_provider, provider.context, initial, control)
    reporting_family = copy.deepcopy(provider.family)
    if provider.context.estimated_theta and fit.theta is not None:
        reporting_family.put_theta(np.log(np.asarray([fit.theta])))
    diagnostics = fit.optimizer_diagnostics
    if diagnostics is not None:
        fit = replace(
            fit,
            optimizer_diagnostics=replace(
                diagnostics,
                startup_source_scans=startup.source_scans,
                startup_batches_scanned=startup.batches_scanned,
            ),
        )
    return StreamEFSExecution(
        fit, provider.fitting, reporting_family, startup, retained_startup
    )
