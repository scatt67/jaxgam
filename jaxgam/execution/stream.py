"""Host-side replayable-source controller for fixed-smoothing PIRLS."""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.base import ExponentialFamily
from jaxgam.families.standard import Binomial, Gaussian, Poisson
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import _to_jax_structure
from jaxgam.fitting.reml import estimate_edf
from jaxgam.fitting.state import StreamFitState
from jaxgam.fitting.stream_kernels import (
    accepts_trial,
    accumulate_working_statistics,
    coefficient_stationarity,
    empty_statistics,
    saturated_loglik_reduction,
    solve_penalized_system,
    trial_deviance,
)
from jaxgam.formula.design_provider import StreamDesign


@dataclass(frozen=True)
class StreamPIRLSControl:
    """Internal fixed-sp stream controls pending public ``FitControl`` wiring."""

    batch_rows: int = 8192
    max_iter: int = 100
    tol: float = 1e-7
    max_halvings: int = 25

    def __post_init__(self) -> None:
        for name, value, allow_zero in (
            ("batch_rows", self.batch_rows, False),
            ("max_iter", self.max_iter, False),
            ("max_halvings", self.max_halvings, True),
        ):
            if (
                not isinstance(value, numbers.Integral)
                or isinstance(value, bool)
                or value < 0
                or (not allow_zero and value == 0)
            ):
                raise ValueError(f"{name} must be an integer in its valid range.")
        if not isinstance(self.tol, numbers.Real) or isinstance(self.tol, bool):
            raise ValueError("Stream PIRLS tolerance must be a finite positive real.")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("Stream PIRLS controls must be positive (halvings >= 0).")


def _preflight(
    stream: StreamDesign, family: ExponentialFamily, control: StreamPIRLSControl
) -> None:
    prepared = stream.prepared
    if prepared.fitting is None:
        raise ValueError("Streamed PIRLS requires prepare_model(..., family=family).")
    if prepared.fitting.unpenalized_rank_deficit:
        raise np.linalg.LinAlgError(
            "Streamed PIRLS requires full rank in unpenalized directions; "
            "use the dense compatibility path or remove aliased columns."
        )
    if not family.is_canonical or family.n_theta:
        raise NotImplementedError(
            "Streamed fixed-sp PIRLS currently supports canonical Gaussian "
            "identity, Poisson log, and Binomial logit families only."
        )
    supported = isinstance(family, (Gaussian, Poisson, Binomial))
    if not supported:
        raise NotImplementedError(
            "Streamed fixed-sp PIRLS currently supports Gaussian, Poisson, "
            "and Binomial only."
        )
    if stream.source.fingerprint() != prepared.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again.")
    # Exercise the source's batch validation before launches, including a
    # useful early failure for unsupported zero/negative batch sizes.
    if control.batch_rows <= 0:  # defensive: dataclass validation above
        raise ValueError("batch_rows must be positive.")


def _source_batches(stream: StreamDesign, batch_rows: int):
    """Yield validated real-row vectors and reject unsupported padded designs."""
    prepared = stream.prepared
    if stream.source.fingerprint() != prepared.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again.")
    valid_rows = 0
    for batch in stream.source.scan(batch_rows):
        if batch.y is None:
            raise ValueError("Streamed PIRLS requires response values in every batch.")
        valid = np.asarray(batch.valid, dtype=bool)
        if not np.all(valid):
            raise NotImplementedError(
                "The initial streamed host path rejects padded RowSource batches; "
                "padding is supported by kernels but not prepared design evaluation."
            )
        y = np.asarray(batch.y, dtype=np.float64)
        weight = np.asarray(batch.weight, dtype=np.float64)
        offset = np.asarray(batch.offset, dtype=np.float64)
        if not all(np.all(np.isfinite(value)) for value in (y, weight, offset)):
            raise ValueError("Streamed fitting batches must contain finite vectors.")
        if np.any(weight < 0):
            raise ValueError("Streamed fitting prior weights must be non-negative.")
        valid_rows += int(np.sum(valid))
        yield batch, y, weight, offset, valid
    if valid_rows != prepared.n_obs:
        raise RuntimeError(
            "RowSource scan changed its valid row count after preparation; "
            "prepare again."
        )
    if stream.source.fingerprint() != prepared.source_fingerprint:
        raise RuntimeError(
            "RowSource changed during streamed fitting scan; prepare again."
        )


def _fitting_batches(stream: StreamDesign, batch_rows: int):
    """Yield bounded fitting-coordinate arrays; no source rows are retained."""
    for batch, y, weight, offset, valid in _source_batches(stream, batch_rows):
        X = stream.prepared.evaluate_fitting_batch(batch)
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Streamed fitting design batch contains non-finite values."
            )
        yield (
            np.asarray(X, dtype=np.float64),
            y,
            weight,
            offset,
            valid,
        )


def _working_scan(
    stream: StreamDesign,
    beta: jax.Array,
    family: ExponentialFamily,
    control: StreamPIRLSControl,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], int]:
    statistics = empty_statistics(stream.prepared.n_coef)
    batches = 0
    for X, y, weight, offset, valid in _fitting_batches(stream, control.batch_rows):
        statistics = accumulate_working_statistics(
            statistics,
            jax.device_put(X),
            jax.device_put(y),
            jax.device_put(weight),
            jax.device_put(offset),
            jax.device_put(valid),
            beta,
            family,
        )
        # Bound outstanding transfer/dispatch buffers to one batch.  This is
        # deliberately conservative until two-batch prefetch is instrumented.
        jax.block_until_ready(statistics[0])
        batches += 1
    return statistics, batches


def _trial_scan(
    stream: StreamDesign,
    beta: jax.Array,
    family: ExponentialFamily,
    control: StreamPIRLSControl,
) -> tuple[jax.Array, jax.Array, int]:
    deviance = jnp.array(0.0, dtype=jnp.float64)
    domain_ok = jnp.array(True)
    batches = 0
    for X, y, weight, offset, valid in _fitting_batches(stream, control.batch_rows):
        batch_deviance, batch_domain = trial_deviance(
            jax.device_put(X),
            jax.device_put(y),
            jax.device_put(weight),
            jax.device_put(offset),
            jax.device_put(valid),
            beta,
            family,
        )
        deviance = deviance + batch_deviance
        domain_ok = domain_ok & batch_domain
        jax.block_until_ready(deviance)
        batches += 1
    return deviance, domain_ok, batches


def _saturated_loglik_scan(
    stream: StreamDesign,
    family: ExponentialFamily,
    scale: jax.Array,
    control: StreamPIRLSControl,
) -> tuple[jax.Array, int]:
    """Collect only the likelihood scalar required by later result modes."""
    saturated = jnp.array(0.0, dtype=jnp.float64)
    batches = 0
    for _batch, y, weight, _offset, valid in _source_batches(
        stream, control.batch_rows
    ):
        saturated = saturated + saturated_loglik_reduction(
            jax.device_put(y),
            jax.device_put(weight),
            jax.device_put(valid),
            scale,
            family,
        )
        jax.block_until_ready(saturated)
        batches += 1
    return saturated, batches


def fit_streamed_pirls(
    stream: StreamDesign,
    family: ExponentialFamily,
    log_lambda: np.ndarray | jax.Array,
    *,
    control: StreamPIRLSControl | None = None,
    beta_init: np.ndarray | jax.Array | None = None,
) -> StreamFitState:
    """Fit canonical fixed-sp GAM coefficients by bounded replayable scans.

    This is an internal execution path until PR6's public ``FitControl`` is
    available.  It performs no `ModelSetup.build` call and does not return
    row-aligned outputs.
    """
    control = StreamPIRLSControl() if control is None else control
    _preflight(stream, family, control)
    prepared = stream.prepared
    assert prepared.fitting is not None
    structure = _to_jax_structure(prepared.fitting.penalty_structure, None)
    rho = jnp.asarray(log_lambda, dtype=jnp.float64)
    if rho.shape != prepared.fitting.log_lambda_init.shape:
        raise ValueError(
            f"Expected {len(prepared.fitting.log_lambda_init)} log smoothing "
            f"parameters, got shape {rho.shape}."
        )
    if not np.all(np.isfinite(np.asarray(rho))):
        raise ValueError("log_lambda must contain only finite values.")
    beta_source = prepared.fitting.beta_init if beta_init is None else beta_init
    beta = jnp.asarray(beta_source, dtype=jnp.float64)
    if beta.shape != (prepared.n_coef,):
        raise ValueError(f"beta_init must have shape ({prepared.n_coef},).")
    if not np.all(np.isfinite(np.asarray(beta))):
        raise ValueError("beta_init must contain only finite values.")

    scans = 0
    batches_scanned = 0
    converged = False
    line_search_failed = False
    backtracks = 0
    final_statistics: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None
    final_H: jax.Array | None = None
    final_factor: jax.Array | None = None
    stationarity = np.inf

    for iteration in range(control.max_iter):
        statistics, batch_count = _working_scan(stream, beta, family, control)
        scans += 1
        batches_scanned += batch_count
        G, b, deviance, domain_ok = statistics
        if not bool(np.asarray(domain_ok)):
            raise ValueError(
                "Current streamed PIRLS coefficients leave the family domain."
            )
        current_penalized = deviance + penalty_ops.quadratic(structure, beta, rho)
        proposal, factor, H = solve_penalized_system(G, b, structure, rho)
        if not bool(np.all(np.isfinite(np.asarray(factor)))):
            raise np.linalg.LinAlgError(
                "Streamed PIRLS normal equations are not SPD; refusing to add "
                "jitter that could hide a rank or conditioning failure."
            )

        accepted = False
        candidate = proposal
        candidate_deviance = jnp.array(jnp.inf, dtype=jnp.float64)
        for halving in range(control.max_halvings + 1):
            if halving:
                candidate = beta + 0.5**halving * (proposal - beta)
            candidate_deviance, candidate_domain, trial_batches = _trial_scan(
                stream, candidate, family, control
            )
            scans += 1
            batches_scanned += trial_batches
            if bool(
                np.asarray(
                    accepts_trial(
                        candidate_deviance,
                        current_penalized,
                        candidate,
                        structure,
                        rho,
                        candidate_domain,
                        iteration == 0,
                    )
                )
            ):
                accepted = True
                backtracks += halving
                break
        if not accepted:
            line_search_failed = True
            final_statistics = statistics
            final_H = H
            final_factor = factor
            break

        accepted_penalized = candidate_deviance + penalty_ops.quadratic(
            structure, candidate, rho
        )
        coefficient_change = float(
            np.max(np.abs(np.asarray(candidate - beta)))
            / (0.1 + np.max(np.abs(np.asarray(candidate))))
        )
        deviance_change = float(
            np.abs(np.asarray(accepted_penalized - current_penalized))
            / (0.1 + abs(float(np.asarray(accepted_penalized))))
        )
        beta = candidate
        # Re-scan next iteration at accepted beta so both the reported factor
        # and stationarity residual use its working model.
        if (
            iteration >= 3
            and coefficient_change < control.tol
            and deviance_change < control.tol
        ):
            final_statistics, batch_count = _working_scan(stream, beta, family, control)
            scans += 1
            batches_scanned += batch_count
            final_H_beta, final_factor_beta, final_H = solve_penalized_system(
                final_statistics[0], final_statistics[1], structure, rho
            )
            del final_H_beta
            stationarity = float(
                np.asarray(coefficient_stationarity(final_H, final_statistics[1], beta))
            )
            final_factor = final_factor_beta
            converged = stationarity < control.tol
            if converged:
                break
            # The accepted coefficients may change in a subsequent iteration;
            # do not accidentally report this prior working model at the
            # iteration limit.
            final_statistics = None
            final_H = None
            final_factor = None

    if final_statistics is None:
        final_statistics, batch_count = _working_scan(stream, beta, family, control)
        scans += 1
        batches_scanned += batch_count
        _unused, final_factor, final_H = solve_penalized_system(
            final_statistics[0], final_statistics[1], structure, rho
        )
        del _unused
    assert final_H is not None
    assert final_factor is not None
    G, b, final_deviance, final_domain = final_statistics
    final_penalized = final_deviance + penalty_ops.quadratic(structure, beta, rho)
    stationarity = float(np.asarray(coefficient_stationarity(final_H, b, beta)))
    edf = estimate_edf(G, final_factor)
    if not family.scale_known:
        denominator = float(prepared.n_obs - np.asarray(edf))
        if not np.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError(
                "Invalid Fisher-EDF scale denominator in streamed fit."
            )
        scale = final_deviance / denominator
    else:
        scale = jnp.array(1.0, dtype=jnp.float64)
    valid_final = (
        bool(np.all(np.isfinite(np.asarray(beta))))
        and bool(np.all(np.isfinite(np.asarray(final_factor))))
        and bool(np.isfinite(np.asarray(final_deviance)))
        and bool(np.asarray(final_domain))
        and bool(np.isfinite(np.asarray(scale)) and np.asarray(scale) > 0)
    )
    if not valid_final:
        raise FloatingPointError(
            "Streamed PIRLS final state is non-finite or outside domain."
        )
    converged = converged and not line_search_failed and stationarity < control.tol
    saturated_loglik, batch_count = _saturated_loglik_scan(
        stream, family, scale, control
    )
    scans += 1
    batches_scanned += batch_count
    return StreamFitState(
        coefficients=beta,
        deviance=final_deviance,
        penalized_deviance=final_penalized,
        scale=scale,
        saturated_loglik=saturated_loglik,
        edf=edf,
        xtwx=G,
        xtwx_fisher=G,
        factor=final_factor,
        fisher_factor=final_factor,
        n_iter=iteration + 1,
        converged=converged,
        line_search_failed=line_search_failed,
        backtracks=backtracks,
        stationarity=stationarity,
        source_scans=scans,
        batches_scanned=batches_scanned,
    )
