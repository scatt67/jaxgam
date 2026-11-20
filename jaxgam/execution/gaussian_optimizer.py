"""Host integration for frozen Gaussian QR statistics and Newton fitting.

The score provider owns only coefficient-sized device arrays during outer
optimization.  Training rows are retained solely by the finalizer needed to
construct the existing :class:`~jaxgam.fitting.pirls.PIRLSResult` contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from jaxgam.families.standard import Gaussian
from jaxgam.fitting.data import FittingData, PreparedFittingMetadata
from jaxgam.fitting.gaussian import (
    GaussianQRScore,
    jit_gaussian_qr_reml_hessian,
    jit_gaussian_qr_reml_score,
    jit_gaussian_qr_reml_value_and_grad,
)
from jaxgam.fitting.newton import NewtonResult
from jaxgam.fitting.pirls import PIRLSResult
from jaxgam.fitting.state import CholeskyCoefficientFactor, StreamFitState
from jaxgam.fitting.stream_kernels import coefficient_stationarity
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.prepare import PreparedModel
from jaxgam.formula.rank_coordinates import IdentifiableCoordinateMap
from jaxgam.links.links import IdentityLink

from .gaussian import (
    GAUSSIAN_LBFGS_MAXCOR,
    GAUSSIAN_LBFGS_PENALTY_THRESHOLD,
    FrozenGaussianQRPreparation,
    GaussianQRIneligible,
    prepare_frozen_gaussian_qr,
    prepare_streamed_frozen_gaussian_qr,
)
from .reml import (
    StreamREMLControl,
    _hermite_change,
    _lbfgsb_roundoff_stop,
    _newton_model_rejection,
    _projected_gradient,
    function_reduction_exit,
    mgcv_relative_convergence,
    mgcv_relative_message,
)


@dataclass(frozen=True)
class GaussianQROuterState:
    """One coefficient-space trial with Newton's required scalar metadata."""

    score_state: GaussianQRScore
    coefficients: jax.Array
    fisher_factor: jax.Array
    xtwx: jax.Array
    edf: jax.Array
    scale: jax.Array


def _dense_transform(fd: FittingData) -> np.ndarray:
    """Return the block-local map from fitting to public coefficients."""
    result = np.eye(fd.n_coef)
    for block in fd.penalty_structure.blocks:
        transform = block.transform
        if transform.kind == "dense":
            values = np.asarray(transform.values, dtype=float)
        elif transform.kind == "diagonal":
            values = np.diag(np.asarray(transform.values, dtype=float))
        elif transform.kind == "identity":
            values = np.eye(transform.size)
        else:  # pragma: no cover - JaxTransform validates this upstream.
            raise ValueError("unsupported Gaussian QR fitting transform")
        result[block.start : block.stop, block.start : block.stop] = values
    return result


class DenseGaussianQROuterProvider:
    """Four-operation host seam consumed by :class:`NewtonOptimizer`.

    ``initial_parameters``, ``evaluate``, and ``differentiate`` touch only the
    frozen QR/penalty statistics.  ``finalize`` is the single post-optimization
    training-row evaluation needed by the legacy result surface.
    """

    def __init__(
        self,
        fitting: FittingData,
        preparation: FrozenGaussianQRPreparation,
    ) -> None:
        identity = np.eye(fitting.n_coef)
        if preparation.coordinate_basis.shape != identity.shape or not np.allclose(
            preparation.coordinate_basis, identity, rtol=0.0, atol=0.0
        ):
            raise GaussianQRIneligible(
                "The initial Gaussian QR optimizer requires full identity "
                "coordinate_basis."
            )
        if preparation.public_basis.shape != identity.shape or not np.allclose(
            preparation.public_basis, identity, rtol=0.0, atol=0.0
        ):
            raise GaussianQRIneligible(
                "The initial Gaussian QR optimizer requires full identity public_basis."
            )
        self._fitting = fitting
        self._preparation = preparation
        device = fitting.X.device
        self._compression = jax.device_put(preparation.compression, device)
        self._penalty_roots = tuple(
            jax.device_put(root, device) for root in preparation.penalty_roots
        )
        self._coordinate_basis = jax.device_put(preparation.coordinate_basis, device)
        self._public_basis = jax.device_put(preparation.public_basis, device)
        self._penalty_range_basis = jax.device_put(
            preparation.penalty_range_basis, device
        )
        public_from_fitting = _dense_transform(fitting)
        self._public_from_fitting = jax.device_put(public_from_fitting, device)
        self._fitting_from_public = jax.device_put(
            np.linalg.solve(public_from_fitting, np.eye(fitting.n_coef)), device
        )
        self._fixed_scale = jax.device_put(jnp.array(1.0, dtype=jnp.float64), device)
        data_factor = self._compression.R @ self._public_from_fitting
        self._data_rhs = data_factor.T @ self._compression.qtz
        self._kwargs = {
            "root_sp_indices": preparation.root_sp_indices,
            "n_smoothing_parameters": preparation.n_smoothing_parameters,
            "penalty_nullity": preparation.penalty_nullity,
            "scale_known": fitting.family.scale_known,
        }
        self._finalize_calls = 0

    @property
    def finalize_calls(self) -> int:
        """Number of final row materializations, exposed for execution gates."""
        return self._finalize_calls

    def initial_parameters(self, log_lambda: jax.Array) -> jax.Array:
        """Initialize unknown Gaussian scale from the frozen weighted QR."""
        log_lambda = jnp.asarray(log_lambda, dtype=jnp.float64)
        if self._fitting.family.scale_known:
            return log_lambda
        unit_scale = jnp.concatenate((log_lambda, jnp.zeros(1, log_lambda.dtype)))
        state, _ = self.evaluate(unit_scale)
        tiny = jnp.finfo(log_lambda.dtype).tiny
        log_scale = jnp.log(jnp.maximum(state.scale, tiny))
        return jnp.concatenate((log_lambda, log_scale[None]))

    def evaluate(self, params: jax.Array) -> tuple[GaussianQROuterState, jax.Array]:
        """Evaluate a trial without accepting any observation-sized input."""
        score_state = jit_gaussian_qr_reml_score(
            params,
            self._compression,
            self._penalty_roots,
            self._coordinate_basis,
            self._public_basis,
            self._penalty_range_basis,
            self._fixed_scale,
            **self._kwargs,
        )
        coefficients = self._fitting_from_public @ score_state.coefficients

        # Transform the stable public-coordinate QR factor to local-D fitting
        # coordinates with another QR.  This preserves H=L L' without forming
        # H or introducing the dense solver's normal-equation jitter.
        _, upper = jnp.linalg.qr(
            score_state.augmented_factor @ self._public_from_fitting,
            mode="reduced",
        )
        fisher_factor = upper.T
        data_factor = self._compression.R @ self._public_from_fitting
        xtwx = data_factor.T @ data_factor
        transformed_data = jax.scipy.linalg.solve_triangular(
            upper.T, data_factor.T, lower=True
        )
        edf = jnp.sum(transformed_data * transformed_data)
        denominator = jnp.maximum(
            self._compression.n_obs - edf, jnp.finfo(params.dtype).eps
        )
        # Gaussian has V(mu)=1 and dV/dmu=0, so Fletcher equals weighted
        # RSS/(n-edf).  The original n is retained by the compression.
        reported_scale = score_state.residual_sum_squares / denominator
        state = GaussianQROuterState(
            score_state=score_state,
            coefficients=coefficients,
            fisher_factor=fisher_factor,
            xtwx=xtwx,
            edf=edf,
            scale=reported_scale,
        )
        return state, score_state.score

    def differentiate(
        self,
        params: jax.Array,
        state: GaussianQROuterState,
        *,
        second_order: bool = True,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Return exact score derivatives from the same frozen statistics."""
        del state
        _, gradient = jit_gaussian_qr_reml_value_and_grad(
            params,
            self._compression,
            self._penalty_roots,
            self._coordinate_basis,
            self._public_basis,
            self._penalty_range_basis,
            self._fixed_scale,
            **self._kwargs,
        )
        if not second_order:
            return gradient, None
        hessian = jit_gaussian_qr_reml_hessian(
            params,
            self._compression,
            self._penalty_roots,
            self._coordinate_basis,
            self._public_basis,
            self._penalty_range_basis,
            self._fixed_scale,
            **self._kwargs,
        )
        return gradient, (hessian + hessian.T) / 2.0

    def finalize(self, params: jax.Array, state: GaussianQROuterState) -> PIRLSResult:
        """Materialize the legacy final fit exactly once after optimization."""
        del params
        self._finalize_calls += 1
        fd = self._fitting
        offset = (
            jnp.zeros(fd.n_obs, dtype=fd.X.dtype) if fd.offset is None else fd.offset
        )
        eta = fd.X @ state.coefficients + offset
        mu = eta
        working_weights = jnp.clip(fd.wt, 1e-10, 1e10)
        return PIRLSResult(
            coefficients=state.coefficients,
            mu=mu,
            eta=eta,
            deviance=state.score_state.residual_sum_squares,
            penalized_deviance=state.score_state.penalized_residual_sum_squares,
            n_iter=jnp.array(1, dtype=jnp.int32),
            converged=jnp.array(True),
            scale=state.scale,
            XtWX=state.xtwx,
            L=state.fisher_factor,
            working_weights=working_weights,
            XtWX_fisher=state.xtwx,
            L_fisher=state.fisher_factor,
        )


def _prepared_dense_transform(prepared: PreparedModel) -> np.ndarray:
    """Return the prepared block-local map from fitting to public coefficients."""
    fitting = prepared.fitting
    if fitting is None:
        raise ValueError("Gaussian QR RowSource provider requires fitting metadata")
    result = np.eye(prepared.n_coef)
    for block in fitting.penalty_structure.blocks:
        result[block.start : block.stop, block.start : block.stop] = (
            block.transform.dense()
        )
    return result


class StreamGaussianQROuterProvider(DenseGaussianQROuterProvider):
    """Row-free Gaussian outer provider built by one bounded source replay."""

    def __init__(
        self,
        stream: StreamDesign,
        fitting: object,
        preparation: FrozenGaussianQRPreparation,
    ) -> None:
        identity = np.eye(stream.prepared.n_coef)
        if preparation.coordinate_basis.shape != identity.shape or not np.array_equal(
            preparation.coordinate_basis, identity
        ):
            raise GaussianQRIneligible(
                "The initial Gaussian QR optimizer requires full identity "
                "coordinate_basis."
            )
        if preparation.public_basis.shape != identity.shape or not np.array_equal(
            preparation.public_basis, identity
        ):
            raise GaussianQRIneligible(
                "The initial Gaussian QR optimizer requires full identity public_basis."
            )
        self._fitting = fitting
        self._preparation = preparation
        device = fitting.log_lambda_init.device
        self._compression = jax.device_put(preparation.compression, device)
        self._penalty_roots = tuple(
            jax.device_put(root, device) for root in preparation.penalty_roots
        )
        self._coordinate_basis = jax.device_put(preparation.coordinate_basis, device)
        self._public_basis = jax.device_put(preparation.public_basis, device)
        self._penalty_range_basis = jax.device_put(
            preparation.penalty_range_basis, device
        )
        public_from_fitting = _prepared_dense_transform(stream.prepared)
        self._public_from_fitting = jax.device_put(public_from_fitting, device)
        self._fitting_from_public = jax.device_put(
            np.linalg.solve(public_from_fitting, identity), device
        )
        self._fixed_scale = jax.device_put(jnp.array(1.0, dtype=jnp.float64), device)
        data_factor = self._compression.R @ self._public_from_fitting
        self._data_rhs = data_factor.T @ self._compression.qtz
        self._kwargs = {
            "root_sp_indices": preparation.root_sp_indices,
            "n_smoothing_parameters": preparation.n_smoothing_parameters,
            "penalty_nullity": preparation.penalty_nullity,
            "scale_known": fitting.family.scale_known,
        }
        self._finalize_calls = 0
        self._final_score_state: GaussianQRScore | None = None

    @property
    def source_scans(self) -> int:
        """Source replays owned by QR fitting, excluding earlier setup scans."""
        return self._preparation.source_scans

    @property
    def batches_scanned(self) -> int:
        """Batches consumed by the single QR preparation replay."""
        return self._preparation.batches_scanned

    def finalize(self, params: jax.Array, state: GaussianQROuterState) -> PIRLSResult:
        """Return row-free fitting state; prediction evaluates future batches."""
        del params
        self._finalize_calls += 1
        self._final_score_state = state.score_state
        empty = jnp.empty(0, dtype=state.coefficients.dtype)
        return PIRLSResult(
            coefficients=state.coefficients,
            mu=empty,
            eta=empty,
            deviance=state.score_state.residual_sum_squares,
            penalized_deviance=state.score_state.penalized_residual_sum_squares,
            n_iter=jnp.array(1, dtype=jnp.int32),
            converged=jnp.array(True),
            scale=state.scale,
            XtWX=state.xtwx,
            L=state.fisher_factor,
            working_weights=empty,
            XtWX_fisher=state.xtwx,
            L_fisher=state.fisher_factor,
        )

    def to_stream_state(self, result: object) -> StreamFitState:
        """Adapt the completed Newton result to compact public result state."""
        score_state = self._final_score_state
        if score_state is None:
            raise RuntimeError("Gaussian QR optimizer must finalize before adaptation")
        pirls = result.pirls_result
        factor = CholeskyCoefficientFactor(pirls.L, self._fitting.n_coef)
        hessian = pirls.L @ pirls.L.T
        stationarity = float(
            np.asarray(
                coefficient_stationarity(hessian, self._data_rhs, pirls.coefficients)
            )
        )
        phi = score_state.scale
        saturated_loglik = (
            -self._compression.n_positive_weights * jnp.log(2.0 * jnp.pi * phi) / 2.0
            + self._compression.sum_log_positive_weights / 2.0
        )
        return StreamFitState(
            coefficients=pirls.coefficients,
            log_lambda=result.log_lambda,
            deviance=pirls.deviance,
            penalized_deviance=pirls.penalized_deviance,
            scale=result.scale,
            score_scale=phi,
            saturated_loglik=saturated_loglik,
            edf=result.edf,
            xtwx=pirls.XtWX,
            xtwx_fisher=pirls.XtWX_fisher,
            coefficient_factor=factor,
            fisher_coefficient_factor=factor,
            n_iter=result.n_iter,
            converged=result.converged,
            # These fields describe the inner coefficient solve in the
            # StreamFitState contract. Gaussian QR solves coefficients directly;
            # outer Newton convergence remains in result.convergence_info.
            line_search_failed=False,
            backtracks=0,
            stationarity=stationarity,
            source_scans=self.source_scans,
            batches_scanned=self.batches_scanned,
        )


@dataclass(frozen=True)
class _GaussianLBFGSTrial:
    """One exact frozen-statistics value/gradient evaluation."""

    params: jax.Array
    state: GaussianQROuterState
    score: jax.Array
    gradient: jax.Array


class _GaussianLBFGSObjective:
    """SciPy boundary retaining only the most recent coefficient-space trial."""

    def __init__(
        self,
        provider: DenseGaussianQROuterProvider,
        fixed_log_lambda: jax.Array | None,
    ) -> None:
        self._provider = provider
        self._fixed_log_lambda = fixed_log_lambda
        self.trial: _GaussianLBFGSTrial | None = None

    def _full_params(self, free_params: np.ndarray) -> jax.Array:
        free = jnp.asarray(free_params, dtype=jnp.float64)
        if self._fixed_log_lambda is None:
            return free
        return jnp.concatenate((self._fixed_log_lambda, free))

    def __call__(self, free_params: np.ndarray) -> tuple[float, np.ndarray]:
        params = self._full_params(free_params)
        if self.trial is None or not np.array_equal(
            np.asarray(params), np.asarray(self.trial.params)
        ):
            state, score = self._provider.evaluate(params)
            gradient, hessian = self._provider.differentiate(
                params, state, second_order=False
            )
            if hessian is not None:
                raise RuntimeError(
                    "Gaussian L-BFGS received an unexpected outer Hessian."
                )
            if not all(
                np.all(np.isfinite(np.asarray(value)))
                for value in (score, gradient, state.coefficients, state.scale)
            ):
                raise FloatingPointError(
                    "Gaussian L-BFGS received a non-finite frozen-statistics trial."
                )
            self.trial = _GaussianLBFGSTrial(params, state, score, gradient)
        gradient = np.asarray(self.trial.gradient, dtype=np.float64)
        if self._fixed_log_lambda is not None:
            gradient = gradient[len(self._fixed_log_lambda) :]
        return float(np.asarray(self.trial.score)), gradient


# The closed-form log-scale step is exact for fixed smoothing parameters, so
# the second step only confirms that the stationary scale is unchanged.
_GAUSSIAN_SCALE_COMPLETION_MAX_STEPS = 2


@dataclass(frozen=True)
class _GaussianScaleCompletion:
    """Outcome of the closed-form log-scale completion after a roundoff stop."""

    steps: int
    stop: str


def _gaussian_scale_profile(
    state: GaussianQROuterState, penalty_nullity: int
) -> tuple[float, float]:
    """Return the stationary log scale and the exact log-scale curvature.

    At fixed smoothing parameters the Gaussian QR score is
    ``Dp/(2 phi) + (n+ - Mp)/2 log(2 pi phi) + const``, with ``Dp`` (the
    penalized RSS) independent of ``phi``.  Its log-scale gradient is
    ``(n+ - Mp)/2 - Dp/(2 phi)`` and its exact curvature is ``Dp/(2 phi)``
    (pinned ``gam.fit3.r`` ``dlr.dlphi``/``d2lr.d2lphi`` with the Gaussian
    ``ls``), so the stationary scale is ``Dp/(n+ - Mp)`` in closed form.
    """
    score_state = state.score_state
    scale_df = float(np.asarray(score_state.n_positive_weights)) - penalty_nullity
    penalized_rss = float(np.asarray(score_state.penalized_residual_sum_squares))
    scale = float(np.asarray(score_state.scale))
    if not np.isfinite(scale_df) or scale_df <= 0.0:
        raise FloatingPointError(
            "Gaussian scale profile has invalid degrees of freedom."
        )
    stationary = penalized_rss / scale_df
    if not np.isfinite(stationary) or stationary <= 0.0:
        raise FloatingPointError("Gaussian scale profile is not positive and finite.")
    return float(np.log(stationary)), penalized_rss / (2.0 * scale)


def _gaussian_score_roundoff(trial: _GaussianLBFGSTrial, penalty_nullity: int) -> float:
    """First-order float64 envelope of one Gaussian QR score value."""
    score_state = trial.state.score_state
    scale = float(np.asarray(score_state.scale))
    n_positive = float(np.asarray(score_state.n_positive_weights))
    magnitude = (
        abs(float(np.asarray(trial.score)))
        + float(np.asarray(score_state.penalized_residual_sum_squares)) / (2.0 * scale)
        + abs((n_positive - penalty_nullity) * np.log(2.0 * np.pi * scale)) / 2.0
        + abs(float(np.asarray(score_state.log_det_hessian))) / 2.0
        + abs(float(np.asarray(score_state.log_det_penalty))) / 2.0
    )
    return 8.0 * float(np.finfo(np.float64).eps) * magnitude


def _free_projected(
    free_params: np.ndarray,
    free_gradient: np.ndarray,
    n_bounded: int,
) -> np.ndarray:
    """Projected free gradient: smoothing coordinates bounded on [-40, 40]."""
    projected = np.asarray(free_gradient, dtype=np.float64).copy()
    if n_bounded:
        projected[:n_bounded] = _projected_gradient(
            free_params[:n_bounded],
            projected[:n_bounded],
            lower=-40.0,
            upper=40.0,
        )
    return projected


def _complete_gaussian_log_scale(
    objective: _GaussianLBFGSObjective,
    free_params: np.ndarray,
    *,
    n_bounded: int,
    penalty_nullity: int,
) -> tuple[_GaussianScaleCompletion, np.ndarray]:
    """Complete the log-scale coordinate after an L-BFGS-B roundoff stop.

    Only the final (log-scale) coordinate moves; smoothing parameters are
    never changed.  Each step goes to the closed-form stationary scale of
    :func:`_gaussian_scale_profile`.  A candidate is accepted on the exact
    gradient, not the raw score: its exact curvature must be positive, both
    the log-scale gradient and the full projected gradient must decrease, and
    pinned ``newton()``'s descent and quadratic-model tests must hold for the
    change integrated from exact endpoint gradients and curvatures, with no
    raw-score increase beyond the float64 envelope of both scores.  A
    rejected candidate is discarded and the accepted trial restored.  The
    completion holds the accepted and one candidate trial only.
    """
    steps = 0
    accepted = objective.trial
    if accepted is None:  # pragma: no cover - the caller evaluated the stop.
        raise RuntimeError("Gaussian scale completion has no accepted trial.")
    params = np.asarray(free_params, dtype=np.float64).copy()
    for _ in range(_GAUSSIAN_SCALE_COMPLETION_MAX_STEPS):
        target, curvature_old = _gaussian_scale_profile(accepted.state, penalty_nullity)
        step = target - params[-1]
        candidate_params = params.copy()
        candidate_params[-1] = target
        if candidate_params[-1] == params[-1]:
            return _GaussianScaleCompletion(steps, "resolved"), params
        _, old_free = objective(params)
        _, new_free = objective(candidate_params)
        candidate = objective.trial
        if candidate is None:  # pragma: no cover - the objective just ran.
            raise RuntimeError("Gaussian scale completion produced no trial.")
        reason = _gaussian_scale_rejection(
            accepted,
            candidate,
            params,
            candidate_params,
            old_free,
            new_free,
            step,
            curvature_old,
            n_bounded=n_bounded,
            penalty_nullity=penalty_nullity,
        )
        if reason is not None:
            objective.trial = accepted
            return _GaussianScaleCompletion(steps, reason), params
        accepted, params = candidate, candidate_params
        steps += 1
    # Dp does not depend on phi, so the second iteration recomputes the same
    # stationary scale and stops "resolved"; this bound only guarantees exit.
    return _GaussianScaleCompletion(steps, "step limit"), params  # pragma: no cover


def _gaussian_scale_rejection(
    accepted: _GaussianLBFGSTrial,
    candidate: _GaussianLBFGSTrial,
    params: np.ndarray,
    candidate_params: np.ndarray,
    old_free: np.ndarray,
    new_free: np.ndarray,
    step: float,
    curvature_old: float,
    *,
    n_bounded: int,
    penalty_nullity: int,
) -> str | None:
    """Return why a closed-form log-scale candidate is not acceptable."""
    score_old = float(np.asarray(accepted.score))
    score_new = float(np.asarray(candidate.score))
    gradient_old = float(old_free[-1])
    gradient_new = float(new_free[-1])
    # The profile raises unless Dp > 0, so the exact curvature Dp/(2 phi) is
    # positive whenever it is returned.
    _, curvature_new = _gaussian_scale_profile(candidate.state, penalty_nullity)
    if not np.all(
        np.isfinite((score_old, score_new, gradient_old, gradient_new, curvature_new))
    ):
        return "nonfinite trial"
    if not abs(gradient_new) < abs(gradient_old):
        return "no gradient reduction"
    old_norm = np.max(np.abs(_free_projected(params, old_free, n_bounded)))
    new_norm = np.max(np.abs(_free_projected(candidate_params, new_free, n_bounded)))
    if not new_norm < old_norm:
        return "no projected gradient reduction"
    predicted = gradient_old * step + 0.5 * curvature_old * step * step
    change = _hermite_change(
        step, gradient_old, gradient_new, curvature_old, curvature_new
    )
    return _newton_model_rejection(
        predicted,
        change,
        score_old,
        score_new,
        _gaussian_score_roundoff(accepted, penalty_nullity)
        + _gaussian_score_roundoff(candidate, penalty_nullity),
    )


def _coordinate_name(index: int, n_free: int, n_bounded: int, pinned: bool) -> str:
    if index == n_free - 1 and (pinned or index >= n_bounded):
        return "log scale"
    return f"log smoothing parameter {index}"


def use_gaussian_qr_lbfgs(n_penalties: int) -> bool:
    """Choose limited-memory optimization above the reviewed m threshold."""
    return n_penalties > GAUSSIAN_LBFGS_PENALTY_THRESHOLD


def optimize_gaussian_qr_lbfgs(
    fitting: FittingData | PreparedFittingMetadata,
    provider: DenseGaussianQROuterProvider,
    *,
    log_lambda_init: jax.Array | None = None,
    pin_lambda: bool = False,
    control: StreamREMLControl | None = None,
) -> NewtonResult:
    """Optimize frozen Gaussian statistics without a full outer Hessian.

    L-BFGS-B retains at most ``maxcor`` correction vectors. Only smoothing
    parameters are bounded; the Gaussian log scale remains unbounded. Fixed
    smoothing parameters are removed from SciPy's free vector entirely.
    """
    if fitting.n_penalties == 0:
        raise ValueError("Gaussian QR L-BFGS requires at least one penalty.")
    reml_control = (
        StreamREMLControl(
            gtol=1e-10,
            ftol=0.0,
            maxcor=GAUSSIAN_LBFGS_MAXCOR,
        )
        if control is None
        else control
    )
    log_lambda = (
        fitting.log_lambda_init
        if log_lambda_init is None
        else jnp.asarray(log_lambda_init, dtype=jnp.float64)
    )
    if log_lambda.shape != (fitting.n_penalties,):
        raise ValueError("log_lambda_init must match the number of Gaussian penalties.")
    if not np.all(np.isfinite(np.asarray(log_lambda))):
        raise ValueError("log_lambda_init must contain only finite values.")
    if not pin_lambda:
        log_lambda = jnp.clip(log_lambda, -40.0, 40.0)
    params_initial = provider.initial_parameters(log_lambda)
    if params_initial.shape[0] < fitting.n_penalties:
        raise ValueError("Gaussian provider returned an incomplete parameter vector.")

    fixed_log_lambda = log_lambda if pin_lambda else None
    free_initial = np.asarray(
        params_initial[fitting.n_penalties :] if pin_lambda else params_initial,
        dtype=np.float64,
    )
    objective = _GaussianLBFGSObjective(provider, fixed_log_lambda)
    if pin_lambda:
        bounds = [(None, None)] * len(free_initial)
    else:
        bounds = [(-40.0, 40.0)] * fitting.n_penalties + [(None, None)] * (
            len(free_initial) - fitting.n_penalties
        )

    if len(free_initial):
        result = minimize(
            objective,
            free_initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": reml_control.max_iter,
                "maxfun": reml_control.maxfun,
                "maxcor": reml_control.maxcor,
                "maxls": reml_control.maxls,
                "gtol": reml_control.gtol,
                "ftol": reml_control.ftol,
            },
        )
        terminal_params = np.asarray(result.x, dtype=np.float64)
        score, free_gradient = objective(terminal_params)
        if int(result.status) == 2 and not fitting.family.scale_known:
            # At a flat optimum L-BFGS-B can exhaust its line search before
            # moving the Gaussian log scale by even one part in a billion.
            # Its conditional minimum is exact: phi = penalized RSS / (n-Mp).
            # Re-evaluate the same frozen statistics there, then apply the
            # ordinary full-coordinate gradient convergence check below.
            terminal_trial = objective.trial
            if terminal_trial is None:  # pragma: no cover - objective just ran.
                raise RuntimeError("Gaussian L-BFGS produced no terminal trial.")
            score_state = terminal_trial.state.score_state
            scale_df = (
                float(np.asarray(score_state.n_positive_weights))
                - provider._kwargs["penalty_nullity"]
            )
            penalized_rss = float(
                np.asarray(score_state.penalized_residual_sum_squares)
            )
            if not np.isfinite(scale_df) or scale_df <= 0.0:
                raise FloatingPointError(
                    "Gaussian scale profile has invalid degrees of freedom."
                )
            optimal_scale = penalized_rss / scale_df
            if not np.isfinite(optimal_scale) or optimal_scale <= 0.0:
                raise FloatingPointError(
                    "Gaussian scale profile is not positive and finite."
                )
            terminal_params = terminal_params.copy()
            terminal_params[-1] = np.log(optimal_scale)
            previous_score = score
            score, free_gradient = objective(terminal_params)
            roundoff = 16.0 * np.finfo(np.float64).eps * (1.0 + abs(previous_score))
            if score > previous_score + roundoff:
                raise FloatingPointError(
                    "Gaussian scale profile increased the criterion."
                )
        n_bounded = 0 if pin_lambda else fitting.n_penalties
        projected = _free_projected(terminal_params, free_gradient, n_bounded)
        projected_gradient_inf = float(np.max(np.abs(projected)))
        terminal_trial = objective.trial
        if terminal_trial is None:  # pragma: no cover - objective just ran.
            raise RuntimeError("Gaussian L-BFGS produced no terminal trial.")
        score_scale = 1.0 + abs(float(np.asarray(terminal_trial.score)))
        threshold = reml_control.gtol * score_scale
        gradient_converged = projected_gradient_inf <= threshold
        completion: _GaussianScaleCompletion | None = None
        roundoff_stop = _lbfgsb_roundoff_stop(result)
        if (
            not gradient_converged
            and roundoff_stop
            and not fitting.family.scale_known
            and np.all(np.abs(projected[:-1]) <= threshold)
        ):
            # Only the log scale exceeds gtol after a raw-score roundoff stop:
            # complete it on exact gradients, then recheck every coordinate.
            completion, terminal_params = _complete_gaussian_log_scale(
                objective,
                terminal_params,
                n_bounded=n_bounded,
                penalty_nullity=provider._kwargs["penalty_nullity"],
            )
            score, free_gradient = objective(terminal_params)
            projected = _free_projected(terminal_params, free_gradient, n_bounded)
            projected_gradient_inf = float(np.max(np.abs(projected)))
            terminal_trial = objective.trial
            if terminal_trial is None:  # pragma: no cover - objective just ran.
                raise RuntimeError("Gaussian L-BFGS produced no terminal trial.")
            score_scale = 1.0 + abs(float(np.asarray(terminal_trial.score)))
            threshold = reml_control.gtol * score_scale
            gradient_converged = projected_gradient_inf <= threshold
        # SciPy can report an abnormal line-search termination after reaching
        # a flat Gaussian optimum. Match NewtonOptimizer's established rule:
        # a failed step at a gradient-converged point is full convergence.
        converged = gradient_converged and (
            bool(result.success) or int(result.status) == 2
        )
        n_iter = int(result.nit)
        relative_test = None
        if not converged and function_reduction_exit(
            int(result.status), str(result.message)
        ):
            # No further decrease is representable: classify the exit by
            # pinned newton()'s relative test, after the log-scale
            # completion above. The exit bounds the last score change by
            # ftol * max(|f|, 1).
            terminal_score = float(np.asarray(terminal_trial.score))
            relative_test = mgcv_relative_convergence(
                terminal_score,
                projected_gradient_inf,
                scale_estimate=(
                    None
                    if fitting.family.scale_known
                    else float(np.asarray(terminal_trial.state.scale))
                ),
                score_change=reml_control.ftol * max(abs(terminal_score), 1.0),
            )
            converged = relative_test.met
        if converged and relative_test is not None:
            convergence_info = mgcv_relative_message(relative_test)
            if completion is not None:
                convergence_info += (
                    f"; log-scale completion: {completion.stop}, "
                    f"{completion.steps} accepted step(s)"
                )
        elif converged:
            convergence_info = "full convergence"
        elif int(result.status) == 1:
            convergence_info = "iteration limit"
        elif roundoff_stop:
            worst = int(np.argmax(np.abs(projected)))
            name = _coordinate_name(worst, len(projected), n_bounded, pin_lambda)
            convergence_info = (
                f"step failed: {name} projected gradient "
                f"{abs(projected[worst]):.3e} > {threshold:.3e} after an "
                f"L-BFGS-B roundoff stop"
            )
            if completion is not None:
                convergence_info += (
                    f"; log-scale completion: {completion.stop}, "
                    f"{completion.steps} accepted step(s)"
                )
        else:
            convergence_info = "step failed"
    else:
        objective(free_initial)
        converged = True
        n_iter = 0
        convergence_info = "fixed sp"

    trial = objective.trial
    if trial is None:  # pragma: no cover - every branch evaluates one trial.
        raise RuntimeError("Gaussian L-BFGS produced no exact trial.")
    gradient = trial.gradient[: fitting.n_penalties]
    if pin_lambda:
        gradient = jnp.zeros_like(gradient)
    pirls_result = provider.finalize(trial.params, trial.state)
    return NewtonResult(
        log_lambda=trial.params[: fitting.n_penalties],
        smoothing_params=jnp.exp(trial.params[: fitting.n_penalties]),
        converged=converged,
        n_iter=n_iter,
        score=trial.score,
        gradient=gradient,
        edf=trial.state.edf,
        scale=trial.state.scale,
        pirls_result=pirls_result,
        convergence_info=convergence_info,
        theta=None,
    )


def build_dense_gaussian_qr_provider(
    fitting: FittingData,
) -> DenseGaussianQROuterProvider:
    """Build the initial full-coordinate Gaussian QR route or fail explicitly."""
    if type(fitting.family) is not Gaussian or not isinstance(
        fitting.family.link, IdentityLink
    ):
        raise GaussianQRIneligible(
            "Gaussian QR requires the exact Gaussian family with identity link"
        )
    if fitting.n_penalties == 0:
        raise GaussianQRIneligible(
            "Gaussian QR smoothing optimization requires at least one penalty"
        )
    if fitting.rank_deficit:
        raise GaussianQRIneligible(
            "Gaussian QR currently requires a full identifiable coefficient map"
        )
    arrays = (fitting.X, fitting.y, fitting.wt)
    if fitting.offset is not None:
        arrays += (fitting.offset,)
    if not all(np.all(np.isfinite(np.asarray(value))) for value in arrays):
        raise GaussianQRIneligible("Gaussian QR requires finite dense fitting data")
    weight = np.asarray(fitting.wt)
    if np.any(weight < 1e-10) or np.any(weight > 1e10):
        raise GaussianQRIneligible(
            "Gaussian QR currently requires unclipped finite positive weights"
        )

    indices = np.arange(fitting.n_coef, dtype=np.int64)
    coordinate_map = IdentifiableCoordinateMap(
        basis=np.eye(fitting.n_coef),
        fitting_to_public=np.eye(fitting.n_coef),
        keep=indices,
        drop=np.empty(0, dtype=np.int64),
    )
    preparation = prepare_frozen_gaussian_qr(fitting, coordinate_map)
    return DenseGaussianQROuterProvider(fitting, preparation)


def build_stream_gaussian_qr_provider(
    stream: StreamDesign,
    fitting: object,
    *,
    batch_rows: int,
) -> StreamGaussianQROuterProvider:
    """Build a one-replay RowSource Gaussian provider or fail explicitly."""
    if type(fitting.family) is not Gaussian or not isinstance(
        fitting.family.link, IdentityLink
    ):
        raise GaussianQRIneligible(
            "Gaussian QR requires the exact Gaussian family with identity link"
        )
    if fitting.n_penalties == 0:
        raise GaussianQRIneligible(
            "Gaussian QR smoothing optimization requires at least one penalty"
        )
    if fitting.rank_deficit:
        raise GaussianQRIneligible(
            "Gaussian QR currently requires a full identifiable coefficient map"
        )
    preparation = prepare_streamed_frozen_gaussian_qr(stream, batch_rows)
    return StreamGaussianQROuterProvider(stream, fitting, preparation)
