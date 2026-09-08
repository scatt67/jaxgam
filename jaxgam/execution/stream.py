"""Host-side replayable-source controller for fixed-smoothing PIRLS."""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.execution.qr import PositiveQRState, qr_update, solve_augmented_qr
from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import _to_jax_structure
from jaxgam.fitting.family_execution import (
    FamilyExecutionLineage,
    FamilyExecutionParameters,
    batch_execution_summary,
    finalize_execution_summary,
    merge_execution_summaries,
)
from jaxgam.fitting.reml import estimate_edf
from jaxgam.fitting.state import (
    CholeskyCoefficientFactor,
    PivotedQRCoefficientFactor,
    StreamFitState,
)
from jaxgam.fitting.stream_kernels import (
    accepts_trial,
    accumulate_working_statistics,
    coefficient_stationarity,
    coefficient_stationarity_from_parts,
    empty_statistics,
    positive_qr_working_rows,
    saturated_loglik_reduction,
    solve_penalized_system,
    trial_deviance,
)
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.fitting_prepare import qr_penalty_roots


@dataclass(frozen=True)
class StreamPIRLSControl:
    """Internal fixed-sp stream controls pending public ``FitControl`` wiring."""

    batch_rows: int = 8192
    max_iter: int = 100
    tol: float = 1e-7
    max_halvings: int = 25
    solver_policy: str = "cholesky"

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
        if self.solver_policy not in ("cholesky", "qr"):
            raise ValueError("solver_policy must be 'cholesky' or 'qr'.")


def _preflight(
    stream: StreamDesign, family: ExponentialFamily, control: StreamPIRLSControl
) -> FamilyExecutionLineage:
    prepared = stream.prepared
    if prepared.fitting is None:
        raise ValueError("Streamed PIRLS requires prepare_model(..., family=family).")
    if prepared.fitting.unpenalized_rank_deficit:
        raise np.linalg.LinAlgError(
            "Streamed PIRLS requires full rank in unpenalized directions; "
            "use the dense compatibility path or remove aliased columns."
        )
    lineage = FamilyExecutionLineage.from_prepared(prepared, family)
    capabilities = lineage.context.capabilities
    if (
        not capabilities.row_separable
        or not capabilities.fisher_working_system
        or not capabilities.direct_deviance
        or not capabilities.saturated_loglikelihood
    ):
        raise NotImplementedError(
            "Streamed PIRLS requires row-separable Fisher working, direct "
            "deviance, and saturated-likelihood family capabilities."
        )
    if (
        capabilities.coefficient_system != "fisher"
        or not capabilities.fisher_equals_observed_for_score
    ):
        raise NotImplementedError(
            "Streamed fixed-sp scoring currently requires a family that "
            "explicitly declares Fisher and observed information equivalent."
        )
    policy = lineage.context.reduction_policy
    if (
        policy.reported_scale not in {"known_one", "gaussian_fisher_edf_deviance"}
        or policy.score_scale == "unsupported"
    ):
        raise NotImplementedError(
            "This family's reported-scale/score reduction policy is not "
            "released for streamed fixed-sp PIRLS."
        )
    if stream.source.fingerprint() != prepared.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again.")
    # Exercise the source's batch validation before launches, including a
    # useful early failure for unsupported zero/negative batch sizes.
    if control.batch_rows <= 0:  # defensive: dataclass validation above
        raise ValueError("batch_rows must be positive.")
    lineage.validate(prepared, family)
    return lineage


def _source_batches(
    stream: StreamDesign,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    batch_rows: int,
):
    """Yield validated real-row vectors and reject unsupported padded designs."""
    prepared = stream.prepared
    lineage.validate(prepared, family)
    if stream.source.fingerprint() != lineage.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again.")
    valid_rows = 0
    for batch in stream.source.scan(batch_rows):
        lineage.validate(stream.prepared, family)
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
    lineage.validate(stream.prepared, family)
    if stream.source.fingerprint() != lineage.source_fingerprint:
        raise RuntimeError(
            "RowSource changed during streamed fitting scan; prepare again."
        )
    if valid_rows != stream.prepared.n_obs:
        raise RuntimeError(
            "RowSource scan changed its valid row count after preparation; "
            "prepare again."
        )


def _fitting_batches(
    stream: StreamDesign,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    batch_rows: int,
):
    """Yield bounded fitting-coordinate arrays; no source rows are retained."""
    for batch, y, weight, offset, valid in _source_batches(
        stream, family, lineage, batch_rows
    ):
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
    lineage: FamilyExecutionLineage,
    parameters: FamilyExecutionParameters,
    control: StreamPIRLSControl,
    device: jax.Device | None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], int]:
    statistics = jax.device_put(empty_statistics(stream.prepared.n_coef), device)
    batches = 0
    for X, y, weight, offset, valid in _fitting_batches(
        stream, family, lineage, control.batch_rows
    ):
        lineage.validate(stream.prepared, family)
        statistics = accumulate_working_statistics(
            statistics,
            jax.device_put(X, device),
            jax.device_put(y, device),
            jax.device_put(weight, device),
            jax.device_put(offset, device),
            jax.device_put(valid, device),
            beta,
            parameters,
            family,
            lineage.context,
        )
        # Bound outstanding transfer/dispatch buffers to one batch.  This is
        # deliberately conservative until two-batch prefetch is instrumented.
        jax.block_until_ready(statistics[0])
        batches += 1
    return statistics, batches


def _qr_working_scan(
    stream: StreamDesign,
    beta: jax.Array,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    parameters: FamilyExecutionParameters,
    control: StreamPIRLSControl,
    device: jax.Device | None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], PositiveQRState, int]:
    """Scan positive Fisher QR rows; G/b are reporting statistics, not a solve."""
    statistics = jax.device_put(empty_statistics(stream.prepared.n_coef), device)
    qr_state: PositiveQRState | None = None
    batches = 0
    for X, y, weight, offset, valid in _fitting_batches(
        stream, family, lineage, control.batch_rows
    ):
        weighted_X, weighted_z, G, b, deviance, domain_ok = positive_qr_working_rows(
            jax.device_put(X, device),
            jax.device_put(y, device),
            jax.device_put(weight, device),
            jax.device_put(offset, device),
            jax.device_put(valid, device),
            beta,
            parameters,
            family,
            lineage.context,
        )
        qr_state = qr_update(
            qr_state,
            np.asarray(weighted_X),
            np.asarray(weighted_z),
            n_coef=stream.prepared.n_coef,
        )
        old_G, old_b, old_deviance, old_domain = statistics
        statistics = (
            old_G + G,
            old_b + b,
            old_deviance + deviance,
            old_domain & domain_ok,
        )
        jax.block_until_ready(statistics[0])
        batches += 1
    if qr_state is None:
        raise ValueError("Streamed QR requires at least one source batch.")
    return statistics, qr_state, batches


def _qr_roots(
    stream: StreamDesign, rho: jax.Array
) -> tuple[tuple[tuple[slice, np.ndarray], ...], tuple[tuple[slice, np.ndarray], ...]]:
    """Build invariant local root layout once for one fixed-sp streamed fit."""
    assert stream.prepared.fitting is not None
    layout = qr_penalty_roots(stream.prepared.fitting.penalty_structure)
    rho_host = np.asarray(rho)
    balanced = tuple((slice(root.start, root.stop), root.root) for root in layout)
    actual_roots = []
    for root in layout:
        value = np.array(root.root * np.exp(0.5 * rho_host[root.sp_index]), copy=True)
        value.setflags(write=False)
        actual_roots.append((slice(root.start, root.stop), value))
    actual = tuple(actual_roots)
    return actual, balanced


def _qr_factor(
    qr_state: PositiveQRState,
    actual_roots: tuple[tuple[slice, np.ndarray], ...],
    balanced_roots: tuple[tuple[slice, np.ndarray], ...],
    device: jax.Device | None,
) -> tuple[jax.Array, PivotedQRCoefficientFactor]:
    """Attach actual rho-scaled roots while retaining unscaled rank metadata."""
    solved = solve_augmented_qr(qr_state, actual_roots, balanced_roots=balanced_roots)
    factor = PivotedQRCoefficientFactor(
        jax.device_put(solved.R, device),
        jax.device_put(solved.pivots, device),
        jax.device_put(solved.keep, device),
        solved.original_n_coef,
    )
    return jax.device_put(solved.coefficients, device), factor


def _qr_data_edf(
    factor: PivotedQRCoefficientFactor,
    data_state: PositiveQRState,
    device: jax.Device | None,
) -> jax.Array:
    """Return Fisher EDF from the data QR root without forming ``G H^-1``.

    ``data_state.R`` is pivoted.  Restoring its columns gives the data root
    in fitting coordinates, and ``B^-T R_data.T`` has squared Frobenius norm
    equal to ``trace(G H^-1)`` without the normal-equation product.
    """
    data_R = np.empty_like(data_state.R)
    data_R[:, data_state.pivots] = data_state.R
    rows = jax.device_put(jnp.asarray(data_R.T), device)
    transformed = factor.root_transpose_inverse(rows)
    return jnp.sum(transformed * transformed)


def _working_factor_scan(
    stream: StreamDesign,
    beta: jax.Array,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    parameters: FamilyExecutionParameters,
    control: StreamPIRLSControl,
    structure: penalty_ops.JaxPenaltyStructure,
    rho: jax.Array,
    qr_roots: tuple[
        tuple[tuple[slice, np.ndarray], ...], tuple[tuple[slice, np.ndarray], ...]
    ]
    | None,
    device: jax.Device | None,
):
    """Return one working scan and its explicitly selected coefficient solver."""
    if control.solver_policy == "qr":
        statistics, qr_state, batches = _qr_working_scan(
            stream, beta, family, lineage, parameters, control, device
        )
        assert qr_roots is not None
        proposal, factor = _qr_factor(qr_state, *qr_roots, device)
        return statistics, proposal, factor, None, batches, qr_state
    statistics, batches = _working_scan(
        stream, beta, family, lineage, parameters, control, device
    )
    proposal, lower, H = solve_penalized_system(
        statistics[0], statistics[1], structure, rho
    )
    return (
        statistics,
        proposal,
        CholeskyCoefficientFactor(lower, stream.prepared.n_coef),
        H,
        batches,
        None,
    )


def _trial_scan(
    stream: StreamDesign,
    beta: jax.Array,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    parameters: FamilyExecutionParameters,
    control: StreamPIRLSControl,
    device: jax.Device | None,
) -> tuple[jax.Array, jax.Array, int]:
    deviance = jnp.array(0.0, dtype=jnp.float64)
    domain_ok = jnp.array(True)
    batches = 0
    for X, y, weight, offset, valid in _fitting_batches(
        stream, family, lineage, control.batch_rows
    ):
        lineage.validate(stream.prepared, family)
        batch_deviance, batch_domain = trial_deviance(
            jax.device_put(X, device),
            jax.device_put(y, device),
            jax.device_put(weight, device),
            jax.device_put(offset, device),
            jax.device_put(valid, device),
            beta,
            parameters,
            family,
            lineage.context,
        )
        deviance = deviance + batch_deviance
        domain_ok = domain_ok & batch_domain
        jax.block_until_ready(deviance)
        batches += 1
    return deviance, domain_ok, batches


def _saturated_loglik_scan(
    stream: StreamDesign,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    parameters: FamilyExecutionParameters,
    scale: jax.Array,
    control: StreamPIRLSControl,
    device: jax.Device | None,
) -> tuple[jax.Array, int]:
    """Collect only the likelihood scalar required by later result modes."""
    saturated = jnp.array(0.0, dtype=jnp.float64)
    batches = 0
    domain_ok = jnp.array(True)
    for _batch, y, weight, _offset, valid in _source_batches(
        stream, family, lineage, control.batch_rows
    ):
        lineage.validate(stream.prepared, family)
        batch_value, batch_domain = saturated_loglik_reduction(
            jax.device_put(y, device),
            jax.device_put(weight, device),
            jax.device_put(valid, device),
            scale,
            parameters,
            family,
            lineage.context,
        )
        saturated = saturated + batch_value
        domain_ok = domain_ok & batch_domain
        jax.block_until_ready(saturated)
        batches += 1
    if not bool(np.asarray(domain_ok)):
        raise ValueError("Saturated likelihood scan left the family domain.")
    return saturated, batches


def _execution_summary_scan(
    stream: StreamDesign,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    control: StreamPIRLSControl,
    device: jax.Device | None,
) -> tuple[dict[str, float], int]:
    """Reduce family-owned global metadata without retaining source rows."""
    summary: tuple[jax.Array, ...] | None = None
    batches = 0
    for _batch, _y, weight, _offset, valid in _source_batches(
        stream, family, lineage, control.batch_rows
    ):
        lineage.validate(stream.prepared, family)
        batch_summary = batch_execution_summary(
            jax.device_put(_y, device),
            jax.device_put(weight, device),
            jax.device_put(valid, device),
            family,
            lineage.context,
        )
        summary = (
            batch_summary
            if summary is None
            else merge_execution_summaries(
                summary, batch_summary, family, lineage.context
            )
        )
        jax.block_until_ready(summary)
        batches += 1
    if summary is None:
        raise ValueError("Streamed fitting requires at least one source batch.")
    return finalize_execution_summary(summary, family), batches


def fit_streamed_pirls(
    stream: StreamDesign,
    family: ExponentialFamily,
    log_lambda: np.ndarray | jax.Array,
    *,
    control: StreamPIRLSControl | None = None,
    beta_init: np.ndarray | jax.Array | None = None,
    device: jax.Device | None = None,
) -> StreamFitState:
    """Fit canonical fixed-sp GAM coefficients by bounded replayable scans.

    This is an internal execution path until PR6's public ``FitControl`` is
    available.  It performs no `ModelSetup.build` call and does not return
    row-aligned outputs.
    """
    control = StreamPIRLSControl() if control is None else control
    lineage = _preflight(stream, family, control)
    prepared = stream.prepared
    assert prepared.fitting is not None
    structure = _to_jax_structure(prepared.fitting.penalty_structure, device)
    rho = jax.device_put(jnp.asarray(log_lambda, dtype=jnp.float64), device)
    if rho.shape != prepared.fitting.log_lambda_init.shape:
        raise ValueError(
            f"Expected {len(prepared.fitting.log_lambda_init)} log smoothing "
            f"parameters, got shape {rho.shape}."
        )
    if not np.all(np.isfinite(np.asarray(rho))):
        raise ValueError("log_lambda must contain only finite values.")
    # Root construction is CPU-only and invariant over PIRLS working scans.
    # ``actual`` has the fixed smoothing multipliers; ``balanced`` preserves
    # scale-independent structural-rank metadata for augmented QR.
    qr_roots = _qr_roots(stream, rho) if control.solver_policy == "qr" else None
    beta_source = prepared.fitting.beta_init if beta_init is None else beta_init
    beta = jax.device_put(jnp.asarray(beta_source, dtype=jnp.float64), device)
    if beta.shape != (prepared.n_coef,):
        raise ValueError(f"beta_init must have shape ({prepared.n_coef},).")
    if not np.all(np.isfinite(np.asarray(beta))):
        raise ValueError("beta_init must contain only finite values.")
    parameters = jax.device_put(
        FamilyExecutionParameters.from_snapshot(lineage.parameters), device
    )

    scans = 0
    batches_scanned = 0
    converged = False
    line_search_failed = False
    backtracks = 0
    final_statistics: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None
    final_H: jax.Array | None = None
    final_factor: CholeskyCoefficientFactor | PivotedQRCoefficientFactor | None = None
    final_qr_state: PositiveQRState | None = None
    stationarity = np.inf

    for iteration in range(control.max_iter):
        statistics, proposal, factor, H, batch_count, _qr_data_state = (
            _working_factor_scan(
                stream,
                beta,
                family,
                lineage,
                parameters,
                control,
                structure,
                rho,
                qr_roots,
                device,
            )
        )
        scans += 1
        batches_scanned += batch_count
        G, b, deviance, domain_ok = statistics
        if not bool(np.asarray(domain_ok)):
            raise ValueError(
                "Current streamed PIRLS coefficients leave the family domain."
            )
        current_penalized = deviance + penalty_ops.quadratic(structure, beta, rho)
        if not bool(np.all(np.isfinite(np.asarray(proposal)))):
            message = (
                "Streamed PIRLS normal equations are not SPD; refusing to add "
                "jitter that could hide a rank or conditioning failure."
                if control.solver_policy == "cholesky"
                else "Streamed PIRLS QR coefficient solve is non-finite."
            )
            raise np.linalg.LinAlgError(message)
        if isinstance(factor, CholeskyCoefficientFactor) and not bool(
            np.all(np.isfinite(np.asarray(factor.lower)))
        ):
            raise np.linalg.LinAlgError(
                "Streamed PIRLS Cholesky lower factor is non-finite; refusing "
                "to add jitter that could hide a rank or conditioning failure."
            )

        accepted = False
        candidate = proposal
        candidate_deviance = jnp.array(jnp.inf, dtype=jnp.float64)
        for halving in range(control.max_halvings + 1):
            if halving:
                candidate = beta + 0.5**halving * (proposal - beta)
            candidate_deviance, candidate_domain, trial_batches = _trial_scan(
                stream, candidate, family, lineage, parameters, control, device
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
            final_qr_state = _qr_data_state
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
            (
                final_statistics,
                _unused,
                final_factor_beta,
                final_H,
                batch_count,
                final_qr_state,
            ) = _working_factor_scan(
                stream,
                beta,
                family,
                lineage,
                parameters,
                control,
                structure,
                rho,
                qr_roots,
                device,
            )
            scans += 1
            batches_scanned += batch_count
            stationarity = float(
                np.asarray(
                    coefficient_stationarity(final_H, final_statistics[1], beta)
                    if control.solver_policy == "cholesky"
                    else coefficient_stationarity_from_parts(
                        final_statistics[0], final_statistics[1], structure, rho, beta
                    )
                )
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
        (
            final_statistics,
            _unused,
            final_factor,
            final_H,
            batch_count,
            final_qr_state,
        ) = _working_factor_scan(
            stream,
            beta,
            family,
            lineage,
            parameters,
            control,
            structure,
            rho,
            qr_roots,
            device,
        )
        scans += 1
        batches_scanned += batch_count
    if control.solver_policy == "cholesky":
        assert final_H is not None
    else:
        assert final_qr_state is not None
    assert final_factor is not None
    factor_valid = (
        bool(np.all(np.isfinite(np.asarray(final_factor.lower))))
        if isinstance(final_factor, CholeskyCoefficientFactor)
        else bool(np.all(np.isfinite(np.asarray(final_factor.R))))
        and bool(np.isfinite(np.asarray(final_factor.logdet_hessian())))
    )
    if not factor_valid:
        raise FloatingPointError(
            "Streamed PIRLS final state is non-finite or outside domain."
        )
    G, b, final_deviance, final_domain = final_statistics
    final_penalized = final_deviance + penalty_ops.quadratic(structure, beta, rho)
    stationarity = float(
        np.asarray(
            coefficient_stationarity(final_H, b, beta)
            if control.solver_policy == "cholesky"
            else coefficient_stationarity_from_parts(G, b, structure, rho, beta)
        )
    )
    edf = (
        estimate_edf(G, final_factor.lower)
        if isinstance(final_factor, CholeskyCoefficientFactor)
        else _qr_data_edf(final_factor, final_qr_state, device)
    )
    summary, batch_count = _execution_summary_scan(
        stream, family, lineage, control, device
    )
    scans += 1
    batches_scanned += batch_count
    if int(summary["n_valid_rows"]) != prepared.n_obs:
        raise RuntimeError(
            "Family execution summary changed its valid row count after preparation."
        )
    policy = lineage.context.reduction_policy
    if policy.reported_scale == "gaussian_fisher_edf_deviance":
        denominator = float(prepared.n_obs - np.asarray(edf))
        if not np.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError(
                "Invalid Fisher-EDF scale denominator in streamed fit."
            )
        scale = final_deviance / denominator
    elif policy.reported_scale == "known_one":
        scale = jnp.array(1.0, dtype=jnp.float64)
    else:  # preflight makes this unreachable; keep an execution fail-closed.
        raise NotImplementedError("Unsupported streamed reported-scale policy.")
    if policy.score_scale == "reported_scale" or structure.n_penalties == 0:
        score_scale = scale
    elif policy.score_scale == "gaussian_fixed_sp":
        score_denominator = int(summary["n_positive_weight"]) - (
            prepared.n_coef - prepared.fitting.total_penalty_rank
        )
        if score_denominator <= 0:
            raise FloatingPointError(
                "Invalid positive-weight REML score denominator in streamed fit."
            )
        score_scale = final_penalized / score_denominator
        if not bool(np.isfinite(np.asarray(score_scale))) or not bool(
            np.asarray(score_scale) > 0
        ):
            raise FloatingPointError(
                "Invalid fixed-sp REML score scale in streamed fit."
            )
    else:  # preflight makes this unreachable; keep an execution fail-closed.
        raise NotImplementedError("Unsupported streamed score-scale policy.")
    valid_final = (
        bool(np.all(np.isfinite(np.asarray(beta))))
        and factor_valid
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
        stream, family, lineage, parameters, score_scale, control, device
    )
    scans += 1
    batches_scanned += batch_count
    return StreamFitState(
        coefficients=beta,
        log_lambda=rho,
        deviance=final_deviance,
        penalized_deviance=final_penalized,
        scale=scale,
        score_scale=score_scale,
        saturated_loglik=saturated_loglik,
        edf=edf,
        xtwx=G,
        xtwx_fisher=G,
        coefficient_factor=final_factor,
        fisher_coefficient_factor=final_factor,
        n_iter=iteration + 1,
        converged=converged,
        line_search_failed=line_search_failed,
        backtracks=backtracks,
        stationarity=stationarity,
        source_scans=scans,
        batches_scanned=batches_scanned,
    )
