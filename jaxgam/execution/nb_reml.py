"""Exact fixed-state streamed REML score and gradient for Negative Binomial.

Each evaluation reconverges coefficients at one explicit parameter vector.
Estimated-theta families use ``[rho, log_theta]``; fixed-theta families use
``rho`` and retain theta solely as immutable family execution metadata.  The
host never invokes EFS conditional-theta updates.
"""

from __future__ import annotations

import numbers
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
    StreamREMLControl,
    _AcceptedParameterizedTrialObjective,
    _parameterized_optimizer_workspace_bytes,
    _run_parameterized_stream_reml,
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


@dataclass(frozen=True)
class NBStreamREMLOptimization:
    """Diagnostic outcome of exact joint NB streamed optimization."""

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
    )


class _AcceptedNBTrialObjective(
    _AcceptedParameterizedTrialObjective[NBStreamREMLTrial]
):
    """NB adapter for the common accepted/candidate trial driver."""

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
    ) -> None:
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
    device: jax.Device | None = None,
) -> NBStreamREMLOptimization:
    """Optimize exact NB REML jointly over smoothing and free theta.

    Dynamic-theta families use ``[rho, log_theta]``. Fixed-theta families
    optimize rho only. With ``pin_lambda=True``, each supplied rho is held at
    its exact value, including values outside the estimated-rho box, while
    log-theta remains free and unbounded.
    """
    reml_control = StreamREMLControl() if control is None else control
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
    outer_workspace_bytes = (
        retained_trial_bytes + retained_history_bytes + lbfgs_workspace_bytes
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
    )
    optimized = _run_parameterized_stream_reml(
        objective,
        params_initial,
        bounds,
        reml_control,
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
    )
