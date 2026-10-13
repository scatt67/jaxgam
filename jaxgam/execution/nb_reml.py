"""Exact fixed-state streamed REML score and gradient for Negative Binomial.

Each evaluation reconverges coefficients at one explicit parameter vector.
Estimated-theta families use ``[rho, log_theta]``; fixed-theta families use
``rho`` and retain theta solely as immutable family execution metadata.  The
host never invokes EFS conditional-theta updates.
"""

from __future__ import annotations

import numbers
import sys
import time
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.execution.nb_stream import (
    NBStreamResult,
    _nb_count_prefix_workspace_bytes,
    fit_nb_streamed_pirls,
)
from jaxgam.execution.regular_stream import (
    RegularStreamWorkspace,
    preflight_regular_stream_workspace,
)
from jaxgam.execution.reml import (
    _CURVATURE_COMPLETION_MAX_FREE,
    StreamREMLControl,
    _abnormal_line_search_stop,
    _AcceptedParameterizedTrialObjective,
    _complete_pinned_coordinate,
    _LocalScoreChange,
    _parameterized_optimizer_workspace_bytes,
    _ParameterizedOptimization,
    _pinned_newton_rejection,
    _projected_gradient_for_bounds,
    _run_parameterized_stream_reml,
    _ScaleNewtonCompletion,
    _synchronize_fitted_state,
    _trial_inner_iterations,
    _valid_exact_trial,
    _validate_max_step,
)
from jaxgam.execution.stream import (
    StreamPIRLSControl,
    _fitting_batches,
    _source_batches,
)
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.fitting.family_execution import (
    FamilyExecutionLineage,
    FamilyExecutionParameters,
)
from jaxgam.fitting.nb_stream_reml import nb_batch_statistics_vjp
from jaxgam.fitting.stream_reml import regular_score_cotangents
from jaxgam.formula.design_provider import StreamDesign


@dataclass(frozen=True)
class NBREMLWorkspace:
    """NB coefficient-controller ledger plus one derivative replay."""

    controller: RegularStreamWorkspace
    adjoint_bytes: int

    @property
    def required_bytes(self) -> int:
        return self.controller.required_bytes + self.adjoint_bytes


@dataclass(frozen=True)
class NBStreamREMLTrial:
    """One exact streamed NB score/gradient at immutable parameters."""

    params: jax.Array
    score: jax.Array
    gradient: jax.Array
    fit_result: NBStreamResult
    workspace: NBREMLWorkspace
    source_factor_residual: float
    observed_factor_residual: float
    theta_free: bool
    source_scans: int
    batches_scanned: int
    source_fingerprint: str
    basis_fingerprint: str
    family_name: str
    link_name: str
    derivative_elapsed_s: float = 0.0
    final_refit_solves: int = 0


@dataclass(frozen=True)
class NBStreamREMLOptimization:
    """Diagnostic outcome of exact joint NB streamed optimization.

    ``status`` is the L-BFGS-B status. When log theta is the only free
    coordinate (every smoothing parameter pinned, or none exists),
    ``theta_completion_steps`` counts the accepted exact secant-Newton
    log-theta steps that follow L-BFGS-B and ``theta_completion_stop``
    records why that completion ended or was not run; ``converged`` is
    restated on the returned trial.
    """

    trial: NBStreamREMLTrial
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
    theta_completion_steps: int = 0
    theta_completion_stop: str | None = None
    cumulative_inner_iterations: int = 0
    cumulative_final_refit_solves: int = 0
    cumulative_derivative_s: float = 0.0

    @property
    def required_bytes(self) -> int:
        """Known trial plus retained optimizer workspace at the peak."""
        return self.trial.workspace.required_bytes + self.outer_workspace_bytes


@dataclass(frozen=True)
class _NBCountPlan:
    """Bounded source summary needed before differentiated prefix allocation."""

    integer_counts: bool
    max_count: float
    source_scans: int
    batches_scanned: int


def _factor_inverse_residual(hessian: jax.Array, inverse: jax.Array) -> float:
    identity = jnp.eye(hessian.shape[0], dtype=hessian.dtype)
    product = hessian @ inverse
    numerator = jnp.max(jnp.abs(product - identity))
    denominator = 1.0 + jnp.max(jnp.abs(product))
    return float(np.asarray(numerator / denominator))


def _nb_count_plan(
    stream: StreamDesign,
    family: NegativeBinomial,
    lineage: FamilyExecutionLineage,
    batch_rows: int,
) -> _NBCountPlan:
    """Scan scalar count metadata without constructing a prefix table."""
    summary = None
    batches = 0
    rows = 0
    for _batch, y, weight, _offset, valid in _source_batches(
        stream, family, lineage, batch_rows
    ):
        item = family.execution_summary_from_batch(y, weight, valid)
        summary = (
            item if summary is None else family.merge_execution_summaries(summary, item)
        )
        rows += len(y)
        batches += 1
    if (
        summary is None
        or rows != stream.prepared.n_obs
        or not family.execution_summary_input_ok(summary)
    ):
        raise ValueError("NB source count summary is invalid")
    metadata = family.finalize_execution_summary(summary)
    return _NBCountPlan(
        integer_counts=bool(metadata["integer_counts"]),
        max_count=float(metadata["max_count"]),
        source_scans=1,
        batches_scanned=batches,
    )


def _preflight_nb_reml(
    stream: StreamDesign,
    family: NegativeBinomial,
    params: np.ndarray | jax.Array,
    control: StreamPIRLSControl,
    maximum_bytes: int,
    warm_start: NBStreamREMLTrial | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    bool,
    NBREMLWorkspace,
    _NBCountPlan,
    FamilyExecutionLineage,
]:
    """Reject incompatible parameters and memory before allocating fit state."""
    if not isinstance(family, NegativeBinomial):
        raise TypeError("NB streamed REML requires NegativeBinomial")
    if family.n_theta not in (0, 1):
        raise NotImplementedError("NB streamed REML supports one or fixed theta")
    fitting = stream.prepared.fitting
    if fitting is None:
        raise ValueError("NB streamed REML requires fitting preparation")
    n_lambda = fitting.penalty_structure.n_penalties
    theta_free = family.n_theta == 1
    n_params = n_lambda + int(theta_free)

    controller = preflight_regular_stream_workspace(stream, control, maximum_bytes)
    p = stream.prepared.n_coef
    B = controller.batch_rows
    # One input/design batch and compiled copy coexist with observed-system,
    # deviance and saturated-likelihood cotangents; coefficient derivatives,
    # theta stationarity and factor/inverse actions remain O(p^2 + Bp).
    # The p-vector term explicitly includes the retained source-solve
    # coefficient copy, which overlaps the reported beta and adjoint state.
    adjoint_bytes = 8 * (2 * B * p + 4 * p * p + 23 * p + 12 * n_params + 24 * B + 32)
    workspace = NBREMLWorkspace(controller, adjoint_bytes)
    if workspace.required_bytes > maximum_bytes:
        raise MemoryError(
            "NB streamed REML needs "
            f"{workspace.required_bytes} known workspace bytes, exceeding "
            f"maximum_bytes={maximum_bytes}."
        )

    params_host = np.asarray(params, dtype=np.float64)
    if params_host.shape != (n_params,):
        raise ValueError(
            f"params must have shape ({n_params},), got {params_host.shape}."
        )
    if not np.all(np.isfinite(params_host)):
        raise ValueError("params must contain only finite values")
    rho = np.array(params_host[:n_lambda], copy=True)
    log_theta = (
        np.array(params_host[n_lambda:], copy=True)
        if theta_free
        else np.asarray(family.get_theta(transformed=False), dtype=np.float64)
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        theta = np.exp(log_theta)
    if log_theta.shape != (1,) or not np.all(np.isfinite(theta)) or np.any(theta <= 0):
        raise ValueError("log_theta must define one finite positive theta")
    if stream.source.fingerprint() != stream.prepared.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again")
    if warm_start is not None and (
        not warm_start.fit_result.state.converged
        or warm_start.params.shape != params_host.shape
        or warm_start.theta_free != theta_free
        or warm_start.source_fingerprint != stream.prepared.source_fingerprint
        or warm_start.basis_fingerprint != stream.prepared.basis_fingerprint
        or warm_start.family_name != family.family_name
        or warm_start.link_name != type(family.link).__qualname__
    ):
        raise ValueError("Warm-start trial is not a compatible converged NB state")

    lineage = FamilyExecutionLineage.from_prepared(stream.prepared, family)
    count_plan = _nb_count_plan(stream, family, lineage, controller.batch_rows)
    # The fixed/trial controller's compatibility score kernel retains an
    # integer branch per batch even for a globally fractional source. The
    # planned derivative replay uses the global count mode explicitly.
    forward_prefix_bytes = _nb_count_prefix_workspace_bytes(
        count_plan.max_count,
        True,
        live_tables=4,
    )
    derivative_prefix_bytes = _nb_count_prefix_workspace_bytes(
        count_plan.max_count,
        count_plan.integer_counts,
        live_tables=4,
    )
    controller = replace(
        controller,
        device_visible_bytes=(controller.device_visible_bytes + forward_prefix_bytes),
    )
    workspace = NBREMLWorkspace(
        controller,
        adjoint_bytes + max(0, derivative_prefix_bytes - forward_prefix_bytes),
    )
    if workspace.required_bytes > maximum_bytes:
        raise MemoryError(
            "NB streamed REML differentiated count-prefix workspace needs "
            f"{workspace.required_bytes} known bytes, exceeding "
            f"maximum_bytes={maximum_bytes}."
        )
    return rho, log_theta, theta_free, workspace, count_plan, lineage


def evaluate_nb_stream_reml(
    stream: StreamDesign,
    family: NegativeBinomial,
    params: np.ndarray | jax.Array,
    *,
    maximum_bytes: int,
    control: StreamPIRLSControl | None = None,
    warm_start: NBStreamREMLTrial | None = None,
    initial_coefficients: np.ndarray | None = None,
    device: jax.Device | None = None,
) -> NBStreamREMLTrial:
    """Reconverge one NB trial and return its exact first derivative.

    The score follows ``gam.fit4``/``gdi2`` provenance: the raw deviance and
    observed information are evaluated at the reported coefficient state,
    while the penalty belongs to the final source solve candidate.  Theta is
    always an explicit immutable batch parameter.  This function does not run
    the EFS conditional-theta controller.
    """
    control = (
        StreamPIRLSControl(solver_policy="qr", tol=1e-9) if control is None else control
    )
    if control.solver_policy != "qr":
        raise ValueError("NB streamed REML requires solver_policy='qr'")
    rho, log_theta, theta_free, workspace, count_plan, lineage = _preflight_nb_reml(
        stream, family, params, control, maximum_bytes, warm_start
    )
    parameters = FamilyExecutionParameters(np.array(log_theta, copy=True))
    initial = (
        initial_coefficients
        if warm_start is None
        else np.asarray(warm_start.fit_result.state.coefficients)
    )
    result = fit_nb_streamed_pirls(
        stream,
        family,
        rho,
        maximum_bytes=maximum_bytes,
        parameters=parameters,
        control=control,
        device=device,
        beta_start=initial,
        start_is_absent=initial is None,
        estimate_theta=False,
    )
    _synchronize_fitted_state(result.state)
    derivative_started = time.perf_counter()
    state = result.state
    if (
        not state.converged
        or state.line_search_failed
        or not np.isfinite(state.stationarity)
        or state.stationarity >= control.tol
    ):
        raise RuntimeError(
            "NB streamed REML requires a converged valid coefficient state"
        )
    if not np.array_equal(np.asarray(state.log_lambda), rho):
        raise RuntimeError("NB streamed state does not match requested rho")
    if not np.array_equal(np.asarray(result.log_theta), log_theta):
        raise RuntimeError("NB streamed state does not match requested log_theta")
    if not np.array_equal(np.asarray(result.source_deviance_log_theta), log_theta):
        raise RuntimeError("NB source score has stale theta attribution")
    if (
        result.integer_counts != count_plan.integer_counts
        or result.max_count != count_plan.max_count
    ):
        raise RuntimeError("NB fit count metadata does not match derivative preflight")

    metadata = PreparedFittingMetadata.from_prepared(stream.prepared, family, device)
    p = stream.prepared.n_coef
    factor = state.coefficient_factor
    if (
        factor.rank != p
        or not bool(np.asarray(factor.score_admissible))
        or result.source_solve_coefficients.shape != (p,)
    ):
        raise np.linalg.LinAlgError(
            "NB streamed REML requires a full-rank positive observed factor"
        )
    rho_device = jax.device_put(jnp.asarray(rho, dtype=jnp.float64), device)
    identity = jax.device_put(jnp.eye(p, dtype=jnp.float64), device)
    observed_inverse = factor.hessian_inverse(identity)
    observed_hessian = penalty_ops.add_to_dense(
        metadata.penalty_structure, state.xtwx, rho_device
    )
    observed_residual = _factor_inverse_residual(observed_hessian, observed_inverse)
    if not np.isfinite(observed_residual) or observed_residual >= 1e-7:
        raise np.linalg.LinAlgError(
            "NB observed factor does not match the coefficient Jacobian"
        )

    source_beta = jax.device_put(result.source_solve_coefficients, device)
    core = regular_score_cotangents(
        rho_device,
        jax.device_put(jnp.array(0.0, dtype=jnp.float64), device),
        observed_inverse,
        source_beta,
        jax.device_put(result.source_deviance, device),
        state.saturated_loglik,
        factor.logdet_hessian(),
        metadata.penalty_structure,
        metadata.total_penalty_null_dim,
        metadata.singleton_sp_indices,
        metadata.singleton_ranks,
        metadata.singleton_eig_constants,
        metadata.multi_block_sp_indices,
        metadata.multi_block_ranks,
        metadata.multi_block_proj_S,
    )
    if not all(np.all(np.isfinite(np.asarray(value))) for value in tuple(core)):
        raise FloatingPointError("NB source score or cotangents are non-finite")
    score_residual = abs(float(np.asarray(core.score)) - result.reml_score) / (
        1.0 + abs(result.reml_score)
    )
    if not np.isfinite(score_residual) or score_residual >= 1e-10:
        raise RuntimeError("NB retained factor does not reproduce the source score")

    beta_bar = core.source_beta
    log_theta_bar = jnp.zeros((1,), dtype=jnp.float64)
    stationarity_log_theta = jnp.zeros((p,), dtype=jnp.float64)
    informative_count = source_good_count = batches = 0
    explicit_parameters = jax.device_put(parameters, device)
    information_beta = jax.device_put(state.coefficients, device)
    max_y = max(0, int(np.ceil(result.max_count)))
    for X, y, weight, offset, valid in _fitting_batches(
        stream, family, lineage, control.batch_rows
    ):
        lineage.validate(stream.prepared, family)
        count_indices = (
            np.rint(y).astype(np.int64)
            if result.integer_counts
            else np.zeros(len(y), dtype=np.int64)
        )
        batch = nb_batch_statistics_vjp(
            information_beta,
            jax.device_put(X, device),
            jax.device_put(y, device),
            jax.device_put(weight, device),
            jax.device_put(offset, device),
            jax.device_put(valid, device),
            jax.device_put(count_indices, device),
            core.observed,
            core.deviance,
            core.saturated,
            explicit_parameters,
            family,
            lineage.context,
            max_y=max_y,
            integer_counts=result.integer_counts,
        )
        if not bool(np.asarray(batch.admissible)):
            raise NotImplementedError("NB streamed derivative batch is inadmissible")
        beta_bar = beta_bar + batch.beta
        log_theta_bar = log_theta_bar + batch.log_theta
        stationarity_log_theta = stationarity_log_theta + batch.stationarity_log_theta
        source_good_count += int(np.asarray(batch.source_good_count))
        informative_count += int(np.asarray(batch.informative_count))
        jax.block_until_ready(beta_bar)
        batches += 1
    if source_good_count <= 0 or informative_count <= 0:
        raise ValueError("NB streamed derivative requires informative source rows")
    if not all(
        np.all(np.isfinite(np.asarray(value)))
        for value in (beta_bar, log_theta_bar, stationarity_log_theta)
    ):
        raise FloatingPointError("NB streamed derivative reduction is non-finite")

    adjoint = factor.hessian_inverse(beta_bar)
    adjoint_product = observed_hessian @ adjoint
    source_residual = float(
        np.asarray(
            jnp.max(jnp.abs(adjoint_product - beta_bar))
            / (1.0 + jnp.max(jnp.abs(adjoint_product)) + jnp.max(jnp.abs(beta_bar)))
        )
    )
    if not np.isfinite(source_residual) or source_residual >= 1e-7:
        raise np.linalg.LinAlgError("NB source adjoint has an unacceptable residual")
    rho_gradient = core.rho - penalty_ops.parameter_vjp(
        metadata.penalty_structure,
        source_beta,
        adjoint,
        rho_device,
    )
    theta_gradient = log_theta_bar - jnp.atleast_1d(
        jnp.vdot(adjoint, stationarity_log_theta)
    )
    gradient = (
        jnp.concatenate((rho_gradient, theta_gradient)) if theta_free else rho_gradient
    )
    jax.block_until_ready(gradient)
    if not np.all(np.isfinite(np.asarray(gradient))):
        raise FloatingPointError("NB streamed REML gradient is non-finite")
    lineage.validate(stream.prepared, family)
    if stream.source.fingerprint() != lineage.source_fingerprint:
        raise RuntimeError("RowSource changed during NB REML evaluation")
    jax.block_until_ready((core.score, gradient))
    derivative_elapsed_s = time.perf_counter() - derivative_started
    return NBStreamREMLTrial(
        params=jax.device_put(jnp.asarray(params, dtype=jnp.float64), device),
        score=core.score,
        gradient=gradient,
        fit_result=result,
        workspace=workspace,
        source_factor_residual=source_residual,
        observed_factor_residual=observed_residual,
        theta_free=theta_free,
        source_scans=state.source_scans + count_plan.source_scans + 1,
        batches_scanned=(state.batches_scanned + count_plan.batches_scanned + batches),
        source_fingerprint=stream.prepared.source_fingerprint,
        basis_fingerprint=stream.prepared.basis_fingerprint,
        family_name=family.family_name,
        link_name=type(family.link).__qualname__,
        derivative_elapsed_s=derivative_elapsed_s,
        final_refit_solves=1,
    )


def _resolved_theta_secant(
    old: NBStreamREMLTrial,
    new: NBStreamREMLTrial,
    gradient_error: float,
) -> float | None:
    """Return the log-theta secant curvature between two exact pinned trials.

    Both gradients are exact source derivatives of fully reconverged trials.
    Assuming each is evaluated within ``gradient_error`` (the inner stopping
    tolerance), the secant is resolved only when the gradient change exceeds
    their combined error; otherwise ``None`` is returned.
    """
    old_params = np.asarray(old.params, dtype=np.float64)
    new_params = np.asarray(new.params, dtype=np.float64)
    if old_params.shape != new_params.shape or not np.array_equal(
        old_params[:-1], new_params[:-1]
    ):
        return None
    step = float(new_params[-1] - old_params[-1])
    change = float(np.asarray(new.gradient)[-1]) - float(np.asarray(old.gradient)[-1])
    if (
        not np.isfinite(step)
        or not np.isfinite(change)
        or step == 0.0
        or not abs(change) > 2.0 * gradient_error
    ):
        return None
    secant = change / step
    return float(secant) if np.isfinite(secant) else None


class _AcceptedNBTrialObjective(
    _AcceptedParameterizedTrialObjective[NBStreamREMLTrial]
):
    """NB adapter for the common accepted/candidate trial driver.

    With ``secant_gradient_error`` set (log theta the only free coordinate),
    every promotion also records two host scalars: the accepted log-theta
    step and the most recent *resolved* secant curvature between consecutive
    accepted exact trials. No earlier trial is retained.
    """

    def __init__(
        self,
        stream: StreamDesign,
        family: NegativeBinomial,
        params_initial: np.ndarray,
        pirls_control: StreamPIRLSControl,
        reml_control: StreamREMLControl,
        maximum_bytes: int,
        initial_coefficients: np.ndarray | None,
        device: jax.Device | None,
        *,
        secant_gradient_error: float | None = None,
    ) -> None:
        if secant_gradient_error is not None and not (
            np.isfinite(secant_gradient_error) and secant_gradient_error > 0.0
        ):
            raise ValueError("secant_gradient_error must be finite and positive")
        self._secant_gradient_error = secant_gradient_error
        self.theta_secant: float | None = None
        self.last_theta_step: float | None = None

        def evaluate(
            params: np.ndarray,
            warm_start: NBStreamREMLTrial | None,
        ) -> NBStreamREMLTrial:
            return evaluate_nb_stream_reml(
                stream,
                family,
                params,
                maximum_bytes=maximum_bytes,
                control=pirls_control,
                warm_start=warm_start,
                initial_coefficients=(
                    initial_coefficients if warm_start is None else None
                ),
                device=device,
            )

        super().__init__(
            stream,
            params_initial,
            reml_control,
            evaluate,
            objective_name="NB streamed REML",
            source_name="NB streamed REML",
            trial_name="NB streamed",
        )

    def _promote_candidate(self) -> None:
        """Promote the candidate and record its pinned log-theta secant."""
        previous = self.accepted
        super()._promote_candidate()
        if self._secant_gradient_error is None:
            return
        self.last_theta_step = float(
            np.asarray(self.accepted.params)[-1] - np.asarray(previous.params)[-1]
        )
        secant = _resolved_theta_secant(
            previous, self.accepted, self._secant_gradient_error
        )
        if secant is not None:
            self.theta_secant = secant


def _nb_score_magnitude(trial: NBStreamREMLTrial) -> float:
    """Sum of the O(n) terms whose float64 difference forms one NB score."""
    return (
        abs(float(np.asarray(trial.score)))
        + abs(trial.fit_result.score_penalized_deviance)
        + abs(float(np.asarray(trial.fit_result.state.saturated_loglik)))
        + abs(
            float(
                np.asarray(trial.fit_result.state.coefficient_factor.logdet_hessian())
            )
        )
    )


def _nb_score_roundoff(trial: NBStreamREMLTrial) -> float:
    """First-order float64 envelope for one NB source score value."""
    return 8 * float(np.finfo(np.float64).eps) * _nb_score_magnitude(trial)


def _nb_local_stationary_score_change(
    objective: _AcceptedNBTrialObjective,
    accepted: NBStreamREMLTrial,
    candidate: NBStreamREMLTrial,
    inner_tolerance: float,
) -> _LocalScoreChange | None:
    """Estimate a representability-scale score change from local gradients.

    This is an empirical *comparison* for a pinned, scalar-theta plateau,
    not a replacement for either source score.  It assumes the exact REML
    gradient is locally smooth and its evaluation error is no larger than
    the inner stopping tolerance. Every endpoint/probe must therefore have
    inner stationarity and source/observed factor residuals at or below that
    tolerance. Quarter-point probes check that the observed gradient is
    monotone and nearly linear over the very small interval.
    Composite-vs-single Simpson disagreement plus the assumed
    derivative error is reported separately; it is not a universal bound.
    """
    base = np.asarray(accepted.params, dtype=np.float64)
    end = np.asarray(candidate.params, dtype=np.float64)
    if (
        base.ndim != 1
        or base.shape != end.shape
        or base.size < 1
        or not np.array_equal(base[:-1], end[:-1])
    ):
        return None
    step = float(end[-1] - base[-1])
    if (
        not np.isfinite(step)
        or step == 0.0
        or not np.isfinite(inner_tolerance)
        or inner_tolerance <= 0.0
        or abs(step) > np.sqrt(np.finfo(np.float64).eps) * (1.0 + abs(float(base[-1])))
    ):
        return None
    source = objective._stream.source.fingerprint()
    lineage_fields = (
        "source_fingerprint",
        "basis_fingerprint",
        "family_name",
        "link_name",
    )
    if accepted.source_fingerprint != source or any(
        getattr(accepted, field) != getattr(candidate, field)
        for field in lineage_fields
    ):
        raise RuntimeError("NB local score comparison changed source/family lineage")

    def source_valid(trial: NBStreamREMLTrial) -> bool:
        """Require every source/observed solve to support the error assumption."""
        inner = trial.fit_result.state
        residuals = (
            inner.stationarity,
            trial.source_factor_residual,
            trial.observed_factor_residual,
        )
        return bool(
            inner.converged
            and not inner.line_search_failed
            and np.all(np.isfinite(residuals))
            and min(residuals) >= 0.0
            and max(residuals) <= inner_tolerance
            and np.isfinite(float(np.asarray(trial.score)))
        )

    if not source_valid(accepted) or not source_valid(candidate):
        return None
    endpoints = [
        float(np.asarray(accepted.gradient)[-1]),
        float(np.asarray(candidate.gradient)[-1]),
    ]
    if not np.all(np.isfinite(endpoints)):
        return None
    gradients = [endpoints[0]]
    for fraction in (0.25, 0.5, 0.75):
        probe_params = base.copy()
        probe_params[-1] = base[-1] + fraction * step
        if not min(base[-1], end[-1]) < probe_params[-1] < max(base[-1], end[-1]):
            return None
        probe = objective._evaluate(probe_params, accepted)
        objective.n_evaluations += 1
        objective.cumulative_source_scans += probe.source_scans
        objective.cumulative_batches_scanned += probe.batches_scanned
        objective.cumulative_inner_iterations += _trial_inner_iterations(probe)
        objective.cumulative_final_refit_solves += getattr(
            probe, "final_refit_solves", 0
        )
        objective.cumulative_derivative_s += getattr(probe, "derivative_elapsed_s", 0.0)
        if (
            not np.array_equal(np.asarray(probe.params), probe_params)
            or any(
                getattr(probe, field) != getattr(accepted, field)
                for field in lineage_fields
            )
            or probe.source_fingerprint != objective._stream.source.fingerprint()
        ):
            raise RuntimeError("NB local score probe changed source/family lineage")
        gradient = float(np.asarray(probe.gradient)[-1])
        if not source_valid(probe) or not np.isfinite(gradient):
            return None
        gradients.append(gradient)
        del probe
    gradients.append(endpoints[1])
    slope = gradients[-1] - gradients[0]
    differences = np.diff(gradients)
    if np.any(differences * np.sign(slope) < -inner_tolerance):
        return None
    linearity_error = max(
        abs(gradients[index] - (gradients[0] + slope * index / 4.0))
        for index in range(1, 4)
    )
    if linearity_error > 0.01 * max(abs(slope), inner_tolerance):
        return None
    single = step * (gradients[0] + 4 * gradients[2] + gradients[4]) / 6.0
    composite = (
        step
        * (
            gradients[0]
            + 4 * gradients[1]
            + 2 * gradients[2]
            + 4 * gradients[3]
            + gradients[4]
        )
        / 12.0
    )
    error = abs(composite - single) / 15.0 + 2 * abs(step) * inner_tolerance
    raw_change = float(np.asarray(candidate.score)) - float(np.asarray(accepted.score))
    # The absolute criterion forms several O(deviance), O(saturated loglik),
    # and determinant terms before subtraction.  This first-order float64
    # envelope only rejects an inconsistent raw difference; the gradient
    # path, not this envelope, decides local descent.
    magnitudes = [_nb_score_magnitude(trial) for trial in (accepted, candidate)]
    roundoff_bound = 8 * np.finfo(np.float64).eps * sum(magnitudes)
    if not np.all(np.isfinite((composite, error, raw_change, roundoff_bound))):
        return None
    return _LocalScoreChange(
        composite, error, raw_change, roundoff_bound, linearity_error
    )


# Secant steps converge superlinearly; from an L-BFGS-B stop the float64
# gradient floor is normally reached in one to three steps. This bound only
# guarantees termination.
_THETA_SECANT_MAX_STEPS = 12
# The accepted log-theta step and the latest resolved secant curvature are the
# only extra state the pinned free-theta completion records during L-BFGS-B.
_THETA_SECANT_BYTES = 2 * sys.getsizeof(0.0)


def _trapezoid_change(
    step: float,
    gradient_old: float,
    gradient_new: float,
    _curvature_old: float,
    _curvature_new: float,
) -> float:
    """Integral of the secant model through two exact endpoint gradients.

    With the trial secant ``(g_new - g_old)/step`` this is the exact change of
    that quadratic; its truncation error is O(step^3 g'').
    """
    return 0.5 * step * (gradient_old + gradient_new)


def _nb_theta_secant_rejection(
    accepted: NBStreamREMLTrial,
    trial: NBStreamREMLTrial,
    step: float,
    curvature: float,
    pinned: np.ndarray,
) -> str | None:
    """Apply newton()'s acceptance tests with secant curvature to one step."""
    gradient_change = float(np.asarray(trial.gradient)[-1]) - float(
        np.asarray(accepted.gradient)[-1]
    )
    return _pinned_newton_rejection(
        accepted,
        trial,
        step,
        pinned,
        label="log-theta",
        curvature_old=curvature,
        curvature_new=gradient_change / step,
        integrated_change=_trapezoid_change,
        score_roundoff=_nb_score_roundoff,
    )


def _complete_pinned_nb_theta(
    objective: _AcceptedNBTrialObjective,
    optimized: _ParameterizedOptimization[NBStreamREMLTrial],
    pinned: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    control: StreamREMLControl,
    inner_tolerance: float,
) -> tuple[_ParameterizedOptimization[NBStreamREMLTrial], _ScaleNewtonCompletion]:
    """Continue pinned ``newton()`` on log theta after the L-BFGS-B stage.

    With every smoothing parameter supplied, or none in the model, pinned
    mgcv's ``gam.outer`` calls ``newton()`` on the NB log theta alone (``G$L``
    keeps only the theta column; supplied log sps enter through ``lsp0``).
    newton() steps by ``-g/|H|`` from exact first and second derivatives,
    caps the step at ``maxNstep = 5``, accepts on score decrease with
    quadratic-model error below 0.8 and step-halves otherwise.

    The exact NB trial provides the exact log-theta gradient but no second
    derivative, so curvature is the latest resolved secant between
    consecutive accepted exact trials (the curvature L-BFGS-B itself uses in
    one dimension). Near the optimum the remaining score decrease is far
    below one ULP of the O(n) score, so L-BFGS-B's Moré-Thuente line search
    can stop on raw-score roundoff: by relative reduction, or ABNORMAL after
    ``maxls`` failed sufficient-decrease tests. After either stop, from a
    valid accepted trial, each secant step's fully reconverged trial is
    accepted only with a positive trial secant, a smaller exact gradient,
    newton()'s descent and quadratic-model tests on the trapezoid integral
    of the exact gradients, and no measurable raw score increase. Rho stays
    bitwise pinned; at most the accepted and one candidate trial are held.
    Convergence is restated on the returned trial: projected gradient at or
    below ``gtol`` and a positive resolved secant curvature.
    """
    lbfgsb_message = optimized.message
    admissible = optimized.status == 0 or _abnormal_line_search_stop(optimized)
    accepted = objective.accepted
    if optimized.converged:
        completion = _ScaleNewtonCompletion(0, "not run: L-BFGS-B stage converged")
    elif not admissible:
        completion = _ScaleNewtonCompletion(
            0, "not run: L-BFGS-B stopped by an iteration/evaluation limit or error"
        )
    elif not np.array_equal(np.asarray(accepted.params, dtype=np.float64)[:-1], pinned):
        raise RuntimeError("Pinned log-theta completion changed supplied rho.")
    elif not _valid_exact_trial(accepted, inner_tolerance):
        completion = _ScaleNewtonCompletion(0, "not run: accepted trial is invalid")
    else:
        # Release SciPy's last rejected line-search trial: the completion
        # holds the accepted and at most one candidate trial.
        objective.candidate = None
        completion = _complete_pinned_coordinate(
            objective,
            pinned,
            label="log-theta",
            max_steps=_THETA_SECANT_MAX_STEPS,
            curvature=lambda _accepted: (
                "no resolved secant"
                if objective.theta_secant is None
                else objective.theta_secant
            ),
            rejection=lambda old, trial, step, curvature: _nb_theta_secant_rejection(
                old, trial, step, curvature, pinned
            ),
        )
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
    curvature = objective.theta_secant
    curvature_positive = bool(
        curvature is not None and np.isfinite(curvature) and curvature > 0.0
    )
    ran = not completion.stop.startswith("not run")
    converged = (
        projected_gradient_inf <= control.gtol and curvature_positive
        if ran
        else optimized.converged
    )
    if completion.last_step is None:
        proposed = "none"
    else:
        outcome = "accepted" if completion.last_step_accepted else "rejected"
        proposed = f"{completion.last_step:+.6e} ({outcome})"
    last_step = objective.last_theta_step
    message = (
        f"L-BFGS-B stage: {lbfgsb_message}; pinned log-theta secant Newton "
        f"completion: {completion.stop}, {completion.steps} accepted step(s), "
        f"final proposed step {proposed}; last accepted log-theta step "
        f"{'none' if last_step is None else f'{last_step:+.6e}'}; "
        f"projected gradient {projected_gradient_inf:.6e} (gtol {control.gtol:.3e}); "
        f"secant curvature "
        f"{'unresolved' if curvature is None else f'{curvature:.10g}'}"
    )
    if ran and not curvature_positive:
        message = f"{message}; no positive resolved log-theta curvature."
    elif not converged:
        message = (
            f"{message}; not converged: projected gradient "
            f"{projected_gradient_inf:.3e} vs gtol {control.gtol:.3e}."
        )
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


def optimize_nb_stream_reml(
    stream: StreamDesign,
    family: NegativeBinomial,
    initial_params: np.ndarray | jax.Array,
    *,
    maximum_bytes: int,
    pin_lambda: bool = False,
    pirls_control: StreamPIRLSControl | None = None,
    control: StreamREMLControl | None = None,
    initial_coefficients: np.ndarray | None = None,
    max_step: float | None = None,
    device: jax.Device | None = None,
) -> NBStreamREMLOptimization:
    """Optimize exact NB REML jointly over smoothing and free theta.

    Dynamic-theta families use ``[rho, log_theta]``. Fixed-theta families
    optimize rho only. With ``pin_lambda=True``, each supplied rho is held at
    its exact value, including values outside the estimated-rho box, while
    log-theta remains free and unbounded. Whenever log theta is the only free
    coordinate (every rho pinned, or no smoothing parameter at all), the
    roundoff stationary comparison and the log-theta secant Newton completion
    follow the L-BFGS-B stage; this pinned route takes precedence. Otherwise
    the free-parameter curvature completion follows an L-BFGS-B roundoff stop
    whenever a smoothing parameter is free: for one free rho with free log
    theta, and, with fixed theta, for one to four free rho (as the regular
    known-scale route does). At most one completion runs; with fixed theta
    and every rho pinned, or none, no coordinate is free and none runs.
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
    if not isinstance(pin_lambda, bool):
        raise ValueError("pin_lambda must be bool")
    if not isinstance(family, NegativeBinomial) or family.n_theta not in (0, 1):
        raise TypeError("NB streamed REML optimization requires NegativeBinomial")
    fitting = stream.prepared.fitting
    if fitting is None:
        raise ValueError("NB streamed REML requires fitting preparation")
    n_lambda = fitting.penalty_structure.n_penalties
    n_params = n_lambda + family.n_theta
    params_initial = np.asarray(initial_params, dtype=np.float64)
    if params_initial.shape != (n_params,):
        raise ValueError(
            f"initial_params must have shape ({n_params},), got {params_initial.shape}."
        )
    if not np.all(np.isfinite(params_initial)):
        raise ValueError("initial_params must contain only finite values")

    lower, upper = -40.0, 40.0
    params_initial = params_initial.copy()
    if pin_lambda:
        bounds: list[tuple[float | None, float | None]] = [
            (float(value), float(value)) for value in params_initial[:n_lambda]
        ]
    else:
        params_initial[:n_lambda] = np.clip(params_initial[:n_lambda], lower, upper)
        bounds = [(lower, upper)] * n_lambda
    bounds.extend([(None, None)] * family.n_theta)

    base_pirls_control = (
        StreamPIRLSControl(solver_policy="qr")
        if pirls_control is None
        else pirls_control
    )
    if base_pirls_control.solver_policy != "qr":
        raise ValueError("NB streamed REML requires solver_policy='qr'")
    pirls_control = replace(
        base_pirls_control,
        tol=min(base_pirls_control.tol, 1e-9, reml_control.gtol / 100.0),
    )

    retained_trial_bytes, retained_history_bytes, lbfgs_workspace_bytes = (
        _parameterized_optimizer_workspace_bytes(
            stream.prepared.n_coef,
            n_params,
            reml_control,
            pirls_control,
        )
    )
    # SciPy can discard a fully stationary free-theta trial when the score
    # rises by one float64 ULP. A model with no penalties has the same single
    # free theta coordinate as a pinned-smoothing model, matching the regular
    # single-free-log-scale rule. Retaining that trial across subsequent probes
    # adds one p²-owned result and its bounded penalized-deviance history
    # alongside accepted and active candidate states.
    theta_only = family.n_theta == 1 and (pin_lambda or n_lambda == 0)
    stationary_trial_bytes = (
        retained_trial_bytes + retained_history_bytes if theta_only else 0
    )
    # The log-theta completion records two host scalars per promotion; its
    # own trials reuse the accepted/candidate slots after L-BFGS-B returns.
    theta_secant_bytes = _THETA_SECANT_BYTES if theta_only else 0
    # The pinned log-theta route takes precedence; the free-parameter
    # completion needs a free smoothing parameter, so the two never both
    # follow one L-BFGS-B stage. Pinned newton() keeps iterating on free log
    # smoothing parameters whatever the theta mode: with free log theta the
    # completion covers the (rho, log theta) pair; with fixed theta, as on the
    # regular known-scale route, one to four free log smoothing parameters.
    curvature_completion = (
        not theta_only
        and not pin_lambda
        and (
            (family.n_theta == 1 and n_lambda == 1)
            or (family.n_theta == 0 and 1 <= n_lambda <= _CURVATURE_COMPLETION_MAX_FREE)
        )
    )
    # The completion holds the m-by-m finite-gradient curvature model, its
    # eigenvalue/solve copies and about fourteen m-vectors of parameters,
    # gradients, probes and steps while accepted and candidate trials
    # coexist; for m <= 4, 3m^2 + 14m <= m^2 + 24m.
    curvature_completion_bytes = (
        8 * (n_params * n_params + 24 * n_params) if curvature_completion else 0
    )
    outer_workspace_bytes = (
        retained_trial_bytes
        + stationary_trial_bytes
        + theta_secant_bytes
        + retained_history_bytes
        + lbfgs_workspace_bytes
        + curvature_completion_bytes
    )
    if outer_workspace_bytes >= maximum_bytes:
        raise MemoryError(
            "NB streamed REML optimizer retention needs "
            f"{outer_workspace_bytes} bytes before a trial, exceeding "
            f"maximum_bytes={maximum_bytes}."
        )
    trial_budget = maximum_bytes - outer_workspace_bytes
    objective = _AcceptedNBTrialObjective(
        stream,
        family,
        params_initial,
        pirls_control,
        reml_control,
        trial_budget,
        initial_coefficients,
        device,
        secant_gradient_error=pirls_control.tol if theta_only else None,
    )
    optimized = _run_parameterized_stream_reml(
        objective,
        params_initial,
        bounds,
        reml_control,
        roundoff_stationary_completion=theta_only,
        curvature_stationary_completion=curvature_completion,
        completion_inner_tolerance=(
            pirls_control.tol if theta_only or curvature_completion else None
        ),
        completion_score_roundoff=_nb_score_roundoff,
        local_score_change=(_nb_local_stationary_score_change if theta_only else None),
        max_step=max_step,
    )
    completion = None
    if theta_only:
        optimized, completion = _complete_pinned_nb_theta(
            objective,
            optimized,
            params_initial[:n_lambda],
            bounds,
            reml_control,
            pirls_control.tol,
        )
    trial = optimized.trial
    if family.n_theta and not np.array_equal(
        np.asarray(trial.fit_result.log_theta),
        np.asarray(trial.params[n_lambda:]),
    ):
        raise RuntimeError("NB optimizer returned a state from stale theta")
    return NBStreamREMLOptimization(
        trial=trial,
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
        theta_completion_steps=0 if completion is None else completion.steps,
        theta_completion_stop=None if completion is None else completion.stop,
        cumulative_inner_iterations=optimized.cumulative_inner_iterations,
        cumulative_final_refit_solves=optimized.cumulative_final_refit_solves,
        cumulative_derivative_s=optimized.cumulative_derivative_s,
    )
