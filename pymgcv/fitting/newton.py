"""Newton optimizer for smoothing parameter selection.

Minimizes the REML/ML criterion over ``log_lambda`` to find optimal
smoothing parameters. This is the outer loop in the GAM fitting
pipeline: Newton over ``log_lambda`` (outer) wrapping PIRLS over
``beta`` (inner). At each Newton step, PIRLS must re-converge at
the proposed ``log_lambda``.

The algorithm follows R's ``fast.REML.fit()`` (fast-REML.r lines
1740-1875): eigenvalue-safe Newton direction with step-halving
line search.

Design doc reference: Section 8.2 (Outer Newton with Damped Hessian)
R source reference: fast-REML.r lines 1740-1875
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import jax
import jax.numpy as jnp
import numpy as np

from pymgcv.fitting.data import FittingData
from pymgcv.fitting.initialization import initialize_beta
from pymgcv.fitting.pirls import PIRLSResult, pirls_loop
from pymgcv.fitting.reml import (
    JointMLCriterion,
    JointREMLCriterion,
    MLCriterion,
    REMLCriterion,
    _CriterionBase,
    _JointCriterionBase,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewtonResult:
    """Result of Newton smoothing parameter optimization.

    Not a JAX pytree -- this is a Python-level post-estimation container.
    Carries everything ``gam()`` (Task 2.6) needs.

    Attributes
    ----------
    log_lambda : jax.Array, shape (m,)
        Optimized log smoothing parameters.
    smoothing_params : jax.Array, shape (m,)
        ``exp(log_lambda)`` -- smoothing parameters on original scale.
    converged : bool
        Whether convergence criteria were met.
    n_iter : int
        Number of Newton iterations taken.
    score : jax.Array, scalar
        Final REML/ML criterion value.
    gradient : jax.Array, shape (m,)
        Final gradient of criterion w.r.t. log_lambda.
    edf : jax.Array, scalar
        Final effective degrees of freedom.
    scale : jax.Array, scalar
        Final estimated dispersion parameter.
    pirls_result : PIRLSResult
        Final PIRLS output (coefficients, mu, etc.).
    convergence_info : str
        ``"full convergence"`` | ``"step failed"`` | ``"iteration limit"``
    """

    log_lambda: jax.Array
    smoothing_params: jax.Array
    converged: bool
    n_iter: int
    score: jax.Array
    gradient: jax.Array
    edf: jax.Array
    scale: jax.Array
    pirls_result: PIRLSResult
    convergence_info: str


# ---------------------------------------------------------------------------
# Safe Newton step (pure function, no state)
# ---------------------------------------------------------------------------


@jax.jit(static_argnames=("max_step",))
def _safe_newton_step(
    gradient: jax.Array,
    hessian: jax.Array,
    max_step: float = 5.0,
) -> jax.Array:
    """Newton step with eigenvalue safety and norm capping.

    Follows R's ``fast.REML.fit``:

    1. Eigendecompose Hessian: ``H = V D V^T``
    2. Flip negative eigenvalues to positive: ``D = |D|``
    3. Floor small eigenvalues: ``D = max(D, max(|D|) * sqrt(eps))``
    4. Compute step: ``step = -V @ ((V^T @ g) / D_safe)``
    5. Cap step norm to ``max_step``

    Parameters
    ----------
    gradient : jax.Array, shape (m,)
        Gradient of criterion w.r.t. log_lambda.
    hessian : jax.Array, shape (m, m)
        Hessian of criterion w.r.t. log_lambda.
    max_step : float
        Maximum allowed step norm in log-lambda space.

    Returns
    -------
    jax.Array, shape (m,)
        Safe Newton direction.
    """
    eigs, V = jnp.linalg.eigh(hessian)

    # Flip negative eigenvalues
    eigs_safe = jnp.abs(eigs)

    # Floor small eigenvalues
    eps = jnp.finfo(jnp.float64).eps
    floor = jnp.max(eigs_safe) * jnp.sqrt(eps)
    eigs_safe = jnp.maximum(eigs_safe, floor)

    # Newton step: -H^{-1} g = -V diag(1/D) V^T g
    step = -V @ ((V.T @ gradient) / eigs_safe)

    # Cap step norm
    step_norm = jnp.sqrt(jnp.sum(step**2))
    step = jnp.where(step_norm > max_step, step * max_step / step_norm, step)

    return step


# ---------------------------------------------------------------------------
# Step-halving outcome
# ---------------------------------------------------------------------------


class _StepOutcome(Enum):
    ACCEPTED = auto()
    STUCK = auto()
    FAILED = auto()


# ---------------------------------------------------------------------------
# Newton optimizer
# ---------------------------------------------------------------------------


class NewtonOptimizer:
    """Newton optimizer for smoothing parameter selection.

    Holds the fixed problem data (``FittingData``, method, tolerances)
    and provides methods for the individual algorithmic steps.

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    method : str
        ``"REML"`` (default) or ``"ML"``.
    max_iter : int
        Maximum Newton iterations.
    tol : float, optional
        Convergence tolerance. Defaults to ``sqrt(machine epsilon) ~ 1.49e-8``.
    max_step : float
        Maximum step norm in log-lambda space.
    """

    def __init__(
        self,
        fd: FittingData,
        method: str = "REML",
        max_iter: int = 200,
        tol: float | None = None,
        max_step: float = 5.0,
    ) -> None:
        self._fd = fd
        self._method = method
        self._max_iter = max_iter
        self._tol = tol if tol is not None else float(jnp.finfo(jnp.float64).eps ** 0.5)
        self._max_step = max_step
        # Joint optimization: append log_phi to params when scale is unknown
        # (matches R's mgcv.r line 2033: lsp <- c(lsp, log.scale))
        self._joint_scale = not fd.family.scale_known and fd.n_penalties > 0

        if method not in ("REML", "ML"):
            raise ValueError(f"Unknown method: {method!r}. Use 'REML' or 'ML'.")

    def _initial_beta(self) -> jax.Array:
        """Compute initial beta via Phase 1 initialization (NumPy -> JAX)."""
        fd = self._fd
        offset_np = np.asarray(fd.offset) if fd.offset is not None else None
        return initialize_beta(
            np.asarray(fd.X),
            np.asarray(fd.y),
            np.asarray(fd.wt),
            fd.family,
            offset_np,
        )

    def _make_criterion(
        self, pirls_result: PIRLSResult
    ) -> _CriterionBase | _JointCriterionBase:
        """Create the appropriate criterion object from converged PIRLS."""
        if self._joint_scale:
            if self._method == "REML":
                return JointREMLCriterion(self._fd, pirls_result)
            return JointMLCriterion(self._fd, pirls_result)
        if self._method == "REML":
            return REMLCriterion(self._fd, pirls_result)
        return MLCriterion(self._fd, pirls_result)

    def _fit_and_score(
        self,
        params: jax.Array,
        beta_init: jax.Array,
    ) -> tuple[PIRLSResult, _CriterionBase | _JointCriterionBase, jax.Array]:
        """Run PIRLS at current params, build criterion, return score.

        For joint optimization, ``params = [log_lambda..., log_phi]``.
        PIRLS only uses the ``log_lambda`` portion (via ``S_lambda``);
        ``log_phi`` enters only through the criterion scoring.
        """
        fd = self._fd
        if self._joint_scale:
            log_lambda = params[: fd.n_penalties]
        else:
            log_lambda = params
        S = fd.S_lambda(log_lambda)
        pirls_result = pirls_loop(fd.X, fd.y, beta_init, S, fd.family, fd.wt, fd.offset)
        criterion = self._make_criterion(pirls_result)
        score = criterion.score(params)
        return pirls_result, criterion, score

    def _step_halve(
        self,
        log_lambda: jax.Array,
        step: jax.Array,
        score: float,
        beta_warm: jax.Array,
        reml_scale: float,
    ) -> tuple[jax.Array, PIRLSResult, _CriterionBase, jax.Array, _StepOutcome]:
        """Trial PIRLS fit at ``log_lambda + step``, halving until score decreases.

        At each halving, PIRLS re-converges at the new smoothing parameters.

        Returns
        -------
        log_lambda_new, pirls_new, crit_new, score_new, outcome
        """
        tol = self._tol
        log_lambda_new = log_lambda + step
        pirls_new, crit_new, score_new = self._fit_and_score(log_lambda_new, beta_warm)

        k = 0
        not_moved = 0
        while float(score_new) > score:
            if float(score_new) - score < tol * reml_scale:
                not_moved += 1
            else:
                not_moved = 0

            if k == 25 or not_moved > 3:
                break

            if bool(jnp.allclose(log_lambda, log_lambda + step)):
                break

            step = step / 2
            k += 1
            log_lambda_new = log_lambda + step
            pirls_new, crit_new, score_new = self._fit_and_score(
                log_lambda_new, beta_warm
            )

        if float(score_new) <= score:
            outcome = _StepOutcome.ACCEPTED
        elif float(score_new) - score >= tol * reml_scale:
            outcome = _StepOutcome.FAILED
        else:
            outcome = _StepOutcome.STUCK

        return log_lambda_new, pirls_new, crit_new, score_new, outcome

    def _check_convergence(
        self,
        criterion: _CriterionBase | _JointCriterionBase,
        params: jax.Array,
        score: float,
        score_old: float,
        deviance: jax.Array,
    ) -> tuple[jax.Array, jax.Array, float, bool]:
        """Compute gradient/Hessian and check convergence at accepted point.

        Returns (grad, hess, reml_scale, converged).
        """
        fd = self._fd
        tol = self._tol
        grad = criterion.gradient(params)
        hess = criterion.hessian(params)
        reml_scale = abs(score) + float(deviance) / fd.n_obs

        grad_converged = float(jnp.max(jnp.abs(grad))) < reml_scale * tol
        reml_converged = abs(score - score_old) < reml_scale * tol

        return grad, hess, reml_scale, grad_converged and reml_converged

    def _build_result(
        self,
        params: jax.Array,
        score: jax.Array,
        gradient: jax.Array,
        criterion: _CriterionBase | _JointCriterionBase,
        pirls_result: PIRLSResult,
        converged: bool,
        step_failed: bool,
        n_iter: int,
    ) -> NewtonResult:
        """Assemble the final NewtonResult."""
        if converged:
            info = "full convergence"
        elif step_failed:
            info = "step failed"
        else:
            info = "iteration limit"

        if self._joint_scale:
            # Extract log_lambda from joint params; keep Fletcher scale
            # for post-estimation (Vp) matching R's model$scale.
            # The REML-optimized phi is internal to the criterion only.
            log_lambda = params[: self._fd.n_penalties]
            grad_lambda = gradient[: self._fd.n_penalties]
        else:
            log_lambda = params
            grad_lambda = gradient

        return NewtonResult(
            log_lambda=log_lambda,
            smoothing_params=jnp.exp(log_lambda),
            converged=converged,
            n_iter=n_iter,
            score=score,
            gradient=grad_lambda,
            edf=criterion.edf,
            scale=criterion.scale,
            pirls_result=pirls_result,
            convergence_info=info,
        )

    def run(self) -> NewtonResult:
        """Run the Newton optimization loop.

        Returns
        -------
        NewtonResult
            Optimization result with converged smoothing parameters,
            final PIRLS fit, and diagnostics.
        """
        fd = self._fd

        # -- Purely parametric shortcut --
        if fd.n_penalties == 0:
            log_lambda = jnp.zeros(0)
            pirls_result, criterion, score = self._fit_and_score(
                log_lambda, self._initial_beta()
            )
            return self._build_result(
                log_lambda,
                score,
                jnp.zeros(0),
                criterion,
                pirls_result,
                converged=True,
                step_failed=False,
                n_iter=0,
            )

        # -- Build initial params vector --
        log_lambda = fd.log_lambda_init.copy()
        if self._joint_scale:
            # Run initial PIRLS to get Fletcher scale for log_phi init
            S_init = fd.S_lambda(log_lambda)
            pirls_init = pirls_loop(
                fd.X, fd.y, self._initial_beta(), S_init, fd.family, fd.wt, fd.offset
            )
            from pymgcv.fitting.reml import estimate_edf, fletcher_scale

            edf_init = estimate_edf(pirls_init.XtWX, pirls_init.L)
            phi_init = fletcher_scale(fd.y, pirls_init.mu, fd.wt, fd.family, edf_init)
            log_phi_init = jnp.log(phi_init)
            params = jnp.concatenate([log_lambda, log_phi_init[None]])
        else:
            params = log_lambda

        # -- Initial fit and score --
        pirls_result, criterion, score = self._fit_and_score(
            params, self._initial_beta()
        )
        grad, hess, reml_scale, _ = self._check_convergence(
            criterion,
            params,
            float(score),
            float(score),
            pirls_result.deviance,
        )

        converged = False
        step_failed = False
        n_iter = 0
        consecutive_stuck = 0

        for outer_iter in range(self._max_iter):
            step = _safe_newton_step(grad, hess, self._max_step)

            params_new, pirls_new, crit_new, score_new, outcome = self._step_halve(
                params,
                step,
                float(score),
                pirls_result.coefficients,
                reml_scale,
            )

            n_iter = outer_iter + 1

            if outcome is _StepOutcome.FAILED:
                step_failed = True
                break

            if outcome is _StepOutcome.STUCK:
                consecutive_stuck += 1
                if consecutive_stuck >= 3:
                    converged = True
                    break
                continue

            consecutive_stuck = 0
            score_old = score
            params = params_new
            pirls_result = pirls_new
            criterion = crit_new
            score = score_new

            grad, hess, reml_scale, converged = self._check_convergence(
                criterion,
                params,
                float(score),
                float(score_old),
                pirls_result.deviance,
            )
            if converged:
                break

        return self._build_result(
            params,
            score,
            grad,
            criterion,
            pirls_result,
            converged,
            step_failed,
            n_iter,
        )


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------


def newton_optimize(
    fd: FittingData,
    method: str = "REML",
    log_lambda_init: jax.Array | None = None,
    max_iter: int = 200,
    tol: float | None = None,
    max_step: float = 5.0,
) -> NewtonResult:
    """Minimize REML/ML criterion over log smoothing parameters.

    Outer Newton loop wrapping PIRLS inner loop. At each iteration:

    1. Compute eigenvalue-safe Newton step from gradient/Hessian
    2. Trial PIRLS fit at proposed log_lambda
    3. Step-halving until criterion decreases
    4. Accept and check convergence

    Follows R's ``fast.REML.fit()`` (fast-REML.r lines 1740-1875).

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    method : str
        ``"REML"`` (default) or ``"ML"``.
    log_lambda_init : jax.Array, shape (m,), optional
        Starting log smoothing parameters. Defaults to ``fd.log_lambda_init``.
    max_iter : int
        Maximum Newton iterations.
    tol : float, optional
        Convergence tolerance. Defaults to ``sqrt(machine epsilon) ~ 1.49e-8``.
    max_step : float
        Maximum step norm in log-lambda space.

    Returns
    -------
    NewtonResult
        Optimization result with converged smoothing parameters,
        final PIRLS fit, and diagnostics.
    """
    if log_lambda_init is not None:
        fd = FittingData(
            X=fd.X,
            y=fd.y,
            wt=fd.wt,
            offset=fd.offset,
            S_list=fd.S_list,
            log_lambda_init=jnp.asarray(log_lambda_init, dtype=jnp.float64),
            family=fd.family,
            n_obs=fd.n_obs,
            n_coef=fd.n_coef,
            penalty_ranks=fd.penalty_ranks,
            penalty_null_dims=fd.penalty_null_dims,
        )
    optimizer = NewtonOptimizer(
        fd, method=method, max_iter=max_iter, tol=tol, max_step=max_step
    )
    return optimizer.run()
