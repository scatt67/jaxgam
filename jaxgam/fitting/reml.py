"""REML and ML criterion functions for smoothing parameter selection.

Given fixed smoothing parameters lambda, PIRLS finds optimal coefficients beta*(lambda).
The REML/ML criteria score how good a particular lambda is -- the Newton optimizer
(Task 2.5) minimizes this criterion to find optimal lambda.

The key insight (Laplace approximation + PIRLS stationarity): at converged beta*,
the derivative of the penalized deviance through beta is zero. So ``jax.grad``
on the criterion w.r.t. log(lambda) gives the correct gradient when beta*, XtWX, mu
are treated as constants (not differentiated through). This matches R's
analytical IFT approach.

The criterion functions in this module do NOT differentiate through PIRLS --
they receive converged PIRLS outputs as fixed inputs. Only ``log_lambda``
flows through the AD trace (through S_lambda construction and log-det
computations). The end-to-end differentiable score (which does differentiate
through PIRLS via ``custom_jvp``) is in ``newton.py``.

Design doc reference: Section 4.3, 4.4
R source reference: gam.fit3.r lines 612-640 (general Laplace REML)

R's exact formula (with gamma=1)::

    REML = Dp/(2phi) - ls_sat + log|H|/2 - log|S+|/2 - Mp/2*log(2*pi*phi)
    ML   = Dp/(2phi) - ls_sat + log|H|/2 - log|S+|/2

Where:
    - Dp = dev + beta^T S_lambda beta  (penalized deviance)
    - ls_sat = family$ls(y, wt, n, phi)  (saturated log-likelihood)
    - H = XtWX + S_lambda  (penalized Hessian)
    - |S+| = pseudo-determinant of S_lambda (product of non-zero eigenvalues)
    - Mp = total penalty null space dimension

For unknown-scale families (Gaussian, Gamma), R jointly optimizes
``log(phi)`` alongside ``log(lambda)`` in the Newton loop (mgcv.r line
2033: ``lsp <- c(lsp, log.scale)``). The ``Joint*Criterion`` classes
implement this by extending the parameter vector with ``log_phi`` and
computing ``ls_sat(phi)`` inside the differentiable trace.

Scale estimation uses Fletcher (2012) correction by default:
    phi_pearson = sum(w_i (y_i - mu_i)^2 / V(mu_i)) / (n - trA)
    s_bar = max(-0.9, mean(V'(mu) * (y - mu) / V(mu)))
    phi_fletcher = phi_pearson / (1 + s_bar)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from jaxgam.jax_utils import block_log_det_S, build_S_lambda

if TYPE_CHECKING:
    from jaxgam.families.base import ExponentialFamily
    from jaxgam.fitting.data import FittingData
    from jaxgam.fitting.pirls import PIRLSResult


# ---------------------------------------------------------------------------
# REMLResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class REMLResult:
    """Result of REML/ML evaluation at given smoothing parameters.

    Attributes
    ----------
    score : jax.Array, scalar
        Criterion value (lower is better).
    edf : jax.Array, scalar
        Effective degrees of freedom (trace of hat matrix).
    scale : jax.Array, scalar
        Estimated dispersion parameter.
    """

    score: jax.Array
    edf: jax.Array
    scale: jax.Array


# Register as JAX pytree so REMLResult can be returned from jax.jit
_REML_FIELDS = [f.name for f in fields(REMLResult)]

jax.tree_util.register_pytree_node(
    REMLResult,
    lambda r: ([getattr(r, f) for f in _REML_FIELDS], None),
    lambda _, children: REMLResult(**dict(zip(_REML_FIELDS, children, strict=True))),
)


# ---------------------------------------------------------------------------
# Shared criterion core
# ---------------------------------------------------------------------------


def _criterion_core(
    log_lambda: jax.Array,
    XtWX: jax.Array,
    beta: jax.Array,
    deviance: jax.Array,
    ls_sat: jax.Array,
    S_list: tuple[jax.Array, ...],
    phi: jax.Array,
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    singleton_eig_constants: jax.Array,
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
) -> jax.Array:
    """Shared REML/ML computation. Pure JAX, differentiable.

    Computes: Dp/(2*phi) - ls_sat + log|H|/2 - log|S+|/2

    Uses block-structured log|S+| for accurate gradients:
    singletons use the exact ``rank * rho + const`` formula,
    multi-penalty blocks use scaled slogdet.

    Parameters
    ----------
    log_lambda : jax.Array, shape (m,)
        Log smoothing parameters.
    XtWX : jax.Array, shape (p, p)
        Weighted cross-product from PIRLS.
    beta : jax.Array, shape (p,)
        Converged coefficients.
    deviance : jax.Array, scalar
        Unpenalized deviance from PIRLS.
    ls_sat : jax.Array, scalar
        Saturated log-likelihood.
    S_list : tuple[jax.Array, ...]
        Per-penalty (p, p) matrices.
    phi : jax.Array, scalar
        Dispersion parameter.
    singleton_sp_indices, singleton_ranks : tuple[int, ...]
        Static block metadata for singleton penalties.
    singleton_eig_constants : jax.Array
        Precomputed eigenvalue constants for singletons.
    multi_block_sp_indices : tuple[tuple[int, ...], ...]
        Static log_lambda indices for multi-penalty blocks.
    multi_block_ranks : tuple[int, ...]
        Static ranks for multi-penalty blocks.
    multi_block_proj_S : tuple[tuple[jax.Array, ...], ...]
        Range-space-projected penalties for multi-penalty blocks.

    Returns
    -------
    jax.Array, scalar
        Core criterion value (before REML correction term).
    """
    p = beta.shape[0]
    S_lambda = build_S_lambda(log_lambda, S_list, p)

    penalty = beta @ S_lambda @ beta
    Dp = deviance + penalty

    H = XtWX + S_lambda
    # Diagonal preconditioning: scale H to unit diagonal before logdet.
    # This dramatically improves conditioning for binomial (where W
    # varies from ~0 to 0.25) and prevents AD noise when jax.hessian
    # differentiates through the factorization.
    d = jnp.sqrt(jnp.maximum(jnp.diag(H), jnp.finfo(H.dtype).tiny))
    d_inv = 1.0 / d
    H_scaled = H * (d_inv[:, None] * d_inv[None, :])
    L = jnp.linalg.cholesky(H_scaled)
    log_det_H = 2.0 * jnp.sum(jnp.log(jnp.diag(L))) + 2.0 * jnp.sum(jnp.log(d))

    log_det_S = block_log_det_S(
        log_lambda,
        singleton_sp_indices,
        singleton_ranks,
        singleton_eig_constants,
        multi_block_sp_indices,
        multi_block_ranks,
        multi_block_proj_S,
    )

    return Dp / (2.0 * phi) - ls_sat + log_det_H / 2.0 - log_det_S / 2.0


# ---------------------------------------------------------------------------
# Scale estimation
# ---------------------------------------------------------------------------


@jax.jit(static_argnames=("family",))
def pearson_rss(
    y: jax.Array,
    mu: jax.Array,
    wt: jax.Array,
    family: ExponentialFamily,
) -> jax.Array:
    """Pearson chi-square: sum(wt * (y - mu)^2 / V(mu)).

    Parameters
    ----------
    y : jax.Array, shape (n,)
        Response values.
    mu : jax.Array, shape (n,)
        Fitted mean values from PIRLS.
    wt : jax.Array, shape (n,)
        Prior weights.
    family : ExponentialFamily
        Family with variance function.

    Returns
    -------
    jax.Array, scalar
        Pearson chi-square statistic.
    """
    V = family.variance(mu)
    return jnp.sum(wt * (y - mu) ** 2 / V)


@jax.jit(static_argnames=("family",))
def fletcher_scale(
    y: jax.Array,
    mu: jax.Array,
    wt: jax.Array,
    family: ExponentialFamily,
    edf: jax.Array,
) -> jax.Array:
    """Fletcher (2012) bias-corrected Pearson scale estimate.

    R reference: gam.fit3.r lines 596-604, default ``scale.est="fletcher"``.

    The correction adjusts the Pearson estimator for the bias introduced
    by the variance function curvature::

        phi_pearson = sum(w_i (y_i - mu_i)^2 / V(mu_i)) / (n - trA)
        s_bar = max(-0.9, mean(V'(mu) * (y - mu) / V(mu)))
        phi_fletcher = phi_pearson / (1 + s_bar)

    Parameters
    ----------
    y : jax.Array, shape (n,)
        Response values.
    mu : jax.Array, shape (n,)
        Fitted mean values.
    wt : jax.Array, shape (n,)
        Prior weights.
    family : ExponentialFamily
        Family with ``variance()`` and ``dvar()`` methods.
    edf : jax.Array, scalar
        Effective degrees of freedom (trace of hat matrix, trA).

    Returns
    -------
    jax.Array, scalar
        Fletcher-corrected scale estimate.
    """
    n = y.shape[0]
    tiny = jnp.finfo(y.dtype).tiny
    eps = jnp.finfo(y.dtype).eps
    pearson = pearson_rss(y, mu, wt, family)
    denom = jnp.maximum(n - edf, eps)
    phi_pearson = pearson / denom

    V = family.variance(mu)
    dV = family.dvar(mu)
    s_bar = jnp.maximum(-0.9, jnp.mean(dV * (y - mu) / V))
    result = phi_pearson / (1.0 + s_bar)
    return jnp.maximum(result, tiny)


@jax.jit
def estimate_edf(
    XtWX: jax.Array,
    L: jax.Array,
) -> jax.Array:
    """Effective degrees of freedom: trace(H^{-1} @ XtWX).

    Parameters
    ----------
    XtWX : jax.Array, shape (p, p)
        Weighted cross-product matrix from PIRLS.
    L : jax.Array, shape (p, p)
        Lower Cholesky factor of H = XtWX + S_lambda from PIRLS.

    Returns
    -------
    jax.Array, scalar
        Effective degrees of freedom (trA).
    """
    H_inv_XtWX = jsla.cho_solve((L, True), XtWX)
    return jnp.trace(H_inv_XtWX)


def estimate_scale(
    y: jax.Array,
    mu: jax.Array,
    wt: jax.Array,
    family: ExponentialFamily,
    edf: jax.Array,
) -> jax.Array:
    """Estimate the dispersion parameter.

    For known-scale families (Binomial, Poisson): returns 1.0.
    For unknown-scale families (Gaussian, Gamma): uses Fletcher (2012).

    Parameters
    ----------
    y : jax.Array, shape (n,)
        Response values.
    mu : jax.Array, shape (n,)
        Fitted mean values.
    wt : jax.Array, shape (n,)
        Prior weights.
    family : ExponentialFamily
        Family with variance and dvar methods.
    edf : jax.Array, scalar
        Effective degrees of freedom (trA).

    Returns
    -------
    jax.Array, scalar
        Estimated dispersion parameter.
    """
    return jnp.where(
        family.scale_known,
        jnp.array(1.0),
        fletcher_scale(y, mu, wt, family, edf),
    )


# ---------------------------------------------------------------------------
# REML / ML criteria
# ---------------------------------------------------------------------------


def reml_criterion(
    log_lambda: jax.Array,
    XtWX: jax.Array,
    beta: jax.Array,
    deviance: jax.Array,
    ls_sat: jax.Array,
    S_list: tuple[jax.Array, ...],
    phi: jax.Array,
    Mp: int,
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    singleton_eig_constants: jax.Array,
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
) -> jax.Array:
    """REML criterion matching R's Laplace REML (gam.fit3.r line 616).

    Pure JAX, differentiable w.r.t. log_lambda via ``jax.grad(argnums=0)``.
    All other arguments are treated as constants (from converged PIRLS).

    Formula::

        REML = Dp/(2*phi) - ls_sat + log|H|/2 - log|S+|/2 - Mp/2*log(2*pi*phi)

    Parameters
    ----------
    log_lambda : jax.Array, shape (m,)
        Log smoothing parameters.
    XtWX, beta, deviance, ls_sat, S_list, phi, Mp
        See ``_criterion_core``.
    singleton_sp_indices, singleton_ranks, singleton_eig_constants,
    multi_block_sp_indices, multi_block_ranks, multi_block_proj_S
        Block-structured log|S+| metadata. See ``block_log_det_S``.

    Returns
    -------
    jax.Array, scalar
        REML score (lower is better).
    """
    core = _criterion_core(
        log_lambda,
        XtWX,
        beta,
        deviance,
        ls_sat,
        S_list,
        phi,
        singleton_sp_indices,
        singleton_ranks,
        singleton_eig_constants,
        multi_block_sp_indices,
        multi_block_ranks,
        multi_block_proj_S,
    )
    return core - Mp / 2.0 * jnp.log(2.0 * jnp.pi * phi)


def ml_criterion(
    log_lambda: jax.Array,
    XtWX: jax.Array,
    beta: jax.Array,
    deviance: jax.Array,
    ls_sat: jax.Array,
    S_list: tuple[jax.Array, ...],
    phi: jax.Array,
    Mp: int,  # noqa: ARG001
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    singleton_eig_constants: jax.Array,
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
) -> jax.Array:
    """ML criterion. Same as REML but without the -Mp/2*log(2*pi*phi) correction.

    Formula::

        ML = Dp/(2*phi) - ls_sat + log|H|/2 - log|S+|/2

    Parameters
    ----------
    log_lambda : jax.Array, shape (m,)
        Log smoothing parameters.
    XtWX, beta, deviance, ls_sat, S_list, phi, Mp
        See ``_criterion_core``.
    singleton_sp_indices, singleton_ranks, singleton_eig_constants,
    multi_block_sp_indices, multi_block_ranks, multi_block_proj_S
        Block-structured log|S+| metadata. See ``block_log_det_S``.

    Returns
    -------
    jax.Array, scalar
        ML score (lower is better).
    """
    return _criterion_core(
        log_lambda,
        XtWX,
        beta,
        deviance,
        ls_sat,
        S_list,
        phi,
        singleton_sp_indices,
        singleton_ranks,
        singleton_eig_constants,
        multi_block_sp_indices,
        multi_block_ranks,
        multi_block_proj_S,
    )


# ---------------------------------------------------------------------------
# Joint criteria (log_lambda + log_phi co-optimized)
# ---------------------------------------------------------------------------


def reml_criterion_joint(
    params: jax.Array,
    XtWX: jax.Array,
    beta: jax.Array,
    deviance: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    S_list: tuple[jax.Array, ...],
    Mp: int,
    n_lambda: int,
    family: ExponentialFamily,
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    singleton_eig_constants: jax.Array,
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
) -> jax.Array:
    """REML criterion with joint ``(log_lambda, log_phi)`` optimization.

    Matches R's approach: ``log(phi)`` is appended to the smoothing
    parameter vector and jointly optimized via Newton (mgcv.r line 2033,
    gam.fit3.r lines 628-637).

    ``ls_sat`` is computed inside the differentiable trace so that
    ``jax.grad`` correctly accounts for its dependence on ``phi``.

    Parameters
    ----------
    params : jax.Array, shape (n_lambda + 1,)
        ``[log_lambda_1, ..., log_lambda_m, log_phi]``.
    XtWX, beta, deviance, y, wt, S_list, Mp, n_lambda, family
        See ``_criterion_core`` / ``reml_criterion``.
    singleton_sp_indices, singleton_ranks, singleton_eig_constants,
    multi_block_sp_indices, multi_block_ranks, multi_block_proj_S
        Block-structured log|S+| metadata. See ``block_log_det_S``.

    Returns
    -------
    jax.Array, scalar
        REML score (lower is better).
    """
    log_lambda = params[:n_lambda]
    phi = jnp.exp(params[n_lambda])
    ls_sat = family.saturated_loglik(y, wt, phi)
    core = _criterion_core(
        log_lambda,
        XtWX,
        beta,
        deviance,
        ls_sat,
        S_list,
        phi,
        singleton_sp_indices,
        singleton_ranks,
        singleton_eig_constants,
        multi_block_sp_indices,
        multi_block_ranks,
        multi_block_proj_S,
    )
    return core - Mp / 2.0 * jnp.log(2.0 * jnp.pi * phi)


def ml_criterion_joint(
    params: jax.Array,
    XtWX: jax.Array,
    beta: jax.Array,
    deviance: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    S_list: tuple[jax.Array, ...],
    Mp: int,  # noqa: ARG001
    n_lambda: int,
    family: ExponentialFamily,
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    singleton_eig_constants: jax.Array,
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
) -> jax.Array:
    """ML criterion with joint ``(log_lambda, log_phi)`` optimization.

    Parameters
    ----------
    params : jax.Array, shape (n_lambda + 1,)
        ``[log_lambda_1, ..., log_lambda_m, log_phi]``.
    XtWX, beta, deviance, y, wt, S_list, Mp, n_lambda, family
        See ``reml_criterion_joint``.
    singleton_sp_indices, singleton_ranks, singleton_eig_constants,
    multi_block_sp_indices, multi_block_ranks, multi_block_proj_S
        Block-structured log|S+| metadata. See ``block_log_det_S``.

    Returns
    -------
    jax.Array, scalar
        ML score (lower is better).
    """
    log_lambda = params[:n_lambda]
    phi = jnp.exp(params[n_lambda])
    ls_sat = family.saturated_loglik(y, wt, phi)
    return _criterion_core(
        log_lambda,
        XtWX,
        beta,
        deviance,
        ls_sat,
        S_list,
        phi,
        singleton_sp_indices,
        singleton_ranks,
        singleton_eig_constants,
        multi_block_sp_indices,
        multi_block_ranks,
        multi_block_proj_S,
    )


# ---------------------------------------------------------------------------
# Pre-compiled JIT'd transforms for the Newton hot loop
# ---------------------------------------------------------------------------

_BLOCK_STATIC = (
    "Mp",
    "singleton_sp_indices",
    "singleton_ranks",
    "multi_block_sp_indices",
    "multi_block_ranks",
)

_jit_reml_score = jax.jit(reml_criterion, static_argnames=_BLOCK_STATIC)
_jit_reml_grad = jax.jit(jax.grad(reml_criterion), static_argnames=_BLOCK_STATIC)
_jit_reml_hess = jax.jit(jax.hessian(reml_criterion), static_argnames=_BLOCK_STATIC)

_jit_ml_score = jax.jit(ml_criterion, static_argnames=_BLOCK_STATIC)
_jit_ml_grad = jax.jit(jax.grad(ml_criterion), static_argnames=_BLOCK_STATIC)
_jit_ml_hess = jax.jit(jax.hessian(ml_criterion), static_argnames=_BLOCK_STATIC)

_JOINT_STATIC = (*_BLOCK_STATIC, "n_lambda", "family")
_jit_reml_joint_score = jax.jit(reml_criterion_joint, static_argnames=_JOINT_STATIC)
_jit_reml_joint_grad = jax.jit(
    jax.grad(reml_criterion_joint), static_argnames=_JOINT_STATIC
)
_jit_reml_joint_hess = jax.jit(
    jax.hessian(reml_criterion_joint),
    static_argnames=_JOINT_STATIC,
)

_jit_ml_joint_score = jax.jit(ml_criterion_joint, static_argnames=_JOINT_STATIC)
_jit_ml_joint_grad = jax.jit(
    jax.grad(ml_criterion_joint),
    static_argnames=_JOINT_STATIC,
)
_jit_ml_joint_hess = jax.jit(
    jax.hessian(ml_criterion_joint),
    static_argnames=_JOINT_STATIC,
)


# Fused gradient + Hessian: compute both in a single XLA program,
# halving the Python↔XLA round trips in _check_convergence().
def _make_grad_hess(
    criterion_fn: Callable[..., jax.Array],
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Create a fused function returning (gradient, hessian) for a criterion.

    Wraps ``jax.grad`` and ``jax.hessian`` of the given criterion into
    a single callable, so that when JIT-compiled the gradient and Hessian
    are computed in one XLA program. This halves the Python↔XLA round
    trips compared to calling grad and hessian separately.

    Parameters
    ----------
    criterion_fn : Callable[..., jax.Array]
        A criterion function (e.g. ``reml_criterion``) that takes
        ``log_lambda`` as its first argument and returns a scalar score.

    Returns
    -------
    Callable[..., tuple[jax.Array, jax.Array]]
        A function with the same signature as ``criterion_fn`` that
        returns ``(gradient, hessian)`` where gradient has shape ``(m,)``
        and hessian has shape ``(m, m)``.
    """
    _grad = jax.grad(criterion_fn)
    _hess = jax.hessian(criterion_fn)

    def _grad_hess(*args: Any, **kwargs: Any) -> tuple[jax.Array, jax.Array]:
        return _grad(*args, **kwargs), _hess(*args, **kwargs)

    return _grad_hess


_jit_reml_grad_hess = jax.jit(
    _make_grad_hess(reml_criterion),
    static_argnames=_BLOCK_STATIC,
)
_jit_ml_grad_hess = jax.jit(
    _make_grad_hess(ml_criterion),
    static_argnames=_BLOCK_STATIC,
)
_jit_reml_joint_grad_hess = jax.jit(
    _make_grad_hess(reml_criterion_joint),
    static_argnames=_JOINT_STATIC,
)
_jit_ml_joint_grad_hess = jax.jit(
    _make_grad_hess(ml_criterion_joint),
    static_argnames=_JOINT_STATIC,
)


# ---------------------------------------------------------------------------
# Criterion classes
# ---------------------------------------------------------------------------


class _CriterionBase(ABC):
    """Base class for REML/ML criterion evaluation.

    Precomputes and caches constants from converged PIRLS at construction.
    Subclasses implement ``score()``, ``gradient()``, ``hessian()``,
    and ``grad_hess()`` using the pre-compiled module-level JIT'd
    transforms.

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    pirls_result : PIRLSResult
        Converged PIRLS output providing coefficients, XtWX, deviance,
        and working weights.
    """

    def __init__(self, fd: FittingData, pirls_result: PIRLSResult) -> None:
        self.edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
        self.scale = estimate_scale(fd.y, pirls_result.mu, fd.wt, fd.family, self.edf)
        self._deviance = pirls_result.deviance
        self._ls_sat = fd.family.saturated_loglik(fd.y, fd.wt, self.scale)
        self._XtWX = pirls_result.XtWX
        self._beta = pirls_result.coefficients
        self._S_list = fd.S_list
        self._Mp = fd.total_penalty_null_dim
        # Block-structured log|S+| metadata
        self._singleton_sp_indices = fd.singleton_sp_indices
        self._singleton_ranks = fd.singleton_ranks
        self._singleton_eig_constants = fd.singleton_eig_constants
        self._multi_block_sp_indices = fd.multi_block_sp_indices
        self._multi_block_ranks = fd.multi_block_ranks
        self._multi_block_proj_S = fd.multi_block_proj_S

    def _kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for the JIT'd criterion function."""
        return {
            "XtWX": self._XtWX,
            "beta": self._beta,
            "deviance": self._deviance,
            "ls_sat": self._ls_sat,
            "S_list": self._S_list,
            "phi": self.scale,
            "Mp": self._Mp,
            "singleton_sp_indices": self._singleton_sp_indices,
            "singleton_ranks": self._singleton_ranks,
            "singleton_eig_constants": self._singleton_eig_constants,
            "multi_block_sp_indices": self._multi_block_sp_indices,
            "multi_block_ranks": self._multi_block_ranks,
            "multi_block_proj_S": self._multi_block_proj_S,
        }

    @abstractmethod
    def score(self, log_lambda: jax.Array) -> jax.Array: ...

    @abstractmethod
    def gradient(self, log_lambda: jax.Array) -> jax.Array: ...

    @abstractmethod
    def hessian(self, log_lambda: jax.Array) -> jax.Array: ...

    @abstractmethod
    def grad_hess(self, log_lambda: jax.Array) -> tuple[jax.Array, jax.Array]: ...

    def evaluate(self, log_lambda: jax.Array) -> REMLResult:
        """Full evaluation returning ``REMLResult``."""
        return REMLResult(score=self.score(log_lambda), edf=self.edf, scale=self.scale)


class REMLCriterion(_CriterionBase):
    """REML criterion for smoothing parameter optimization.

    Precomputes and caches constants from converged PIRLS at construction.
    Provides ``score()``, ``gradient()``, ``hessian()``, and
    ``grad_hess()`` for the Newton optimizer, and ``evaluate()`` for
    full diagnostic output.

    Uses pre-compiled module-level ``_jit_reml_*`` transforms so that
    the JIT cache is reused across Newton iterations (arrays flow as
    dynamic inputs, not baked-in closure constants).

    Usage by the Newton optimizer::

        obj = REMLCriterion(fd, pirls_result)
        score = obj.score(log_lambda)
        grad, hess = obj.grad_hess(log_lambda)  # fused, 1 XLA dispatch
        result = obj.evaluate(log_lambda)  # -> REMLResult

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    pirls_result : PIRLSResult
        Converged PIRLS output.
    """

    def score(self, log_lambda: jax.Array) -> jax.Array:
        """REML score at given log_lambda."""
        return _jit_reml_score(log_lambda, **self._kwargs())

    def gradient(self, log_lambda: jax.Array) -> jax.Array:
        """REML gradient via pre-compiled ``jax.grad``."""
        return _jit_reml_grad(log_lambda, **self._kwargs())

    def hessian(self, log_lambda: jax.Array) -> jax.Array:
        """REML Hessian via pre-compiled ``jax.hessian``."""
        return _jit_reml_hess(log_lambda, **self._kwargs())

    def grad_hess(self, log_lambda: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Fused gradient + Hessian in a single XLA dispatch."""
        return _jit_reml_grad_hess(log_lambda, **self._kwargs())


class MLCriterion(_CriterionBase):
    """ML criterion for smoothing parameter optimization.

    Same interface as ``REMLCriterion`` but uses the ML formula
    (no ``-Mp/2*log(2*pi*phi)`` correction).

    Uses pre-compiled module-level ``_jit_ml_*`` transforms.

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    pirls_result : PIRLSResult
        Converged PIRLS output.
    """

    def score(self, log_lambda: jax.Array) -> jax.Array:
        """ML score at given log_lambda."""
        return _jit_ml_score(log_lambda, **self._kwargs())

    def gradient(self, log_lambda: jax.Array) -> jax.Array:
        """ML gradient via pre-compiled ``jax.grad``."""
        return _jit_ml_grad(log_lambda, **self._kwargs())

    def hessian(self, log_lambda: jax.Array) -> jax.Array:
        """ML Hessian via pre-compiled ``jax.hessian``."""
        return _jit_ml_hess(log_lambda, **self._kwargs())

    def grad_hess(self, log_lambda: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Fused gradient + Hessian in a single XLA dispatch."""
        return _jit_ml_grad_hess(log_lambda, **self._kwargs())


# ---------------------------------------------------------------------------
# Joint criterion classes (log_lambda + log_phi co-optimized)
# ---------------------------------------------------------------------------


class _JointCriterionBase(ABC):
    """Base class for joint ``(log_lambda, log_phi)`` criterion.

    Used when ``family.scale_known`` is False (Gaussian, Gamma).
    The parameter vector is ``params = [log_lambda..., log_phi]``,
    and ``ls_sat`` is recomputed inside the differentiable trace
    so ``jax.grad`` accounts for its dependence on ``phi``.

    EDF is computed once at construction (it depends on S_lambda
    and XtWX, not on phi). The Fletcher scale is still computed
    for use as the initial ``log_phi`` estimate.

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    pirls_result : PIRLSResult
        Converged PIRLS output.
    """

    def __init__(self, fd: FittingData, pirls_result: PIRLSResult) -> None:
        self.edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
        # Fletcher scale used only as initial log_phi estimate
        self.scale = estimate_scale(fd.y, pirls_result.mu, fd.wt, fd.family, self.edf)
        self._deviance = pirls_result.deviance
        self._XtWX = pirls_result.XtWX
        self._beta = pirls_result.coefficients
        self._S_list = fd.S_list
        self._y = fd.y
        self._wt = fd.wt
        # Retained as attribute because joint criteria need to call
        # family.saturated_loglik(y, wt, phi) inside the JIT trace.
        self._family = fd.family
        self._n_lambda = fd.n_penalties
        self._Mp = fd.total_penalty_null_dim
        # Block-structured log|S+| metadata
        self._singleton_sp_indices = fd.singleton_sp_indices
        self._singleton_ranks = fd.singleton_ranks
        self._singleton_eig_constants = fd.singleton_eig_constants
        self._multi_block_sp_indices = fd.multi_block_sp_indices
        self._multi_block_ranks = fd.multi_block_ranks
        self._multi_block_proj_S = fd.multi_block_proj_S

    def _kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for the JIT'd joint criterion function."""
        return {
            "XtWX": self._XtWX,
            "beta": self._beta,
            "deviance": self._deviance,
            "y": self._y,
            "wt": self._wt,
            "S_list": self._S_list,
            "Mp": self._Mp,
            "n_lambda": self._n_lambda,
            "family": self._family,
            "singleton_sp_indices": self._singleton_sp_indices,
            "singleton_ranks": self._singleton_ranks,
            "singleton_eig_constants": self._singleton_eig_constants,
            "multi_block_sp_indices": self._multi_block_sp_indices,
            "multi_block_ranks": self._multi_block_ranks,
            "multi_block_proj_S": self._multi_block_proj_S,
        }

    @abstractmethod
    def score(self, params: jax.Array) -> jax.Array: ...

    @abstractmethod
    def gradient(self, params: jax.Array) -> jax.Array: ...

    @abstractmethod
    def hessian(self, params: jax.Array) -> jax.Array: ...

    @abstractmethod
    def grad_hess(self, params: jax.Array) -> tuple[jax.Array, jax.Array]: ...


class JointREMLCriterion(_JointCriterionBase):
    """REML criterion with joint ``(log_lambda, log_phi)`` optimization.

    Matches R's approach where ``log(phi)`` is appended to the smoothing
    parameter vector and jointly optimized via Newton.

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    pirls_result : PIRLSResult
        Converged PIRLS output.
    """

    def score(self, params: jax.Array) -> jax.Array:
        """REML score at given params = [log_lambda, log_phi]."""
        return _jit_reml_joint_score(params, **self._kwargs())

    def gradient(self, params: jax.Array) -> jax.Array:
        """REML gradient w.r.t. params via ``jax.grad``."""
        return _jit_reml_joint_grad(params, **self._kwargs())

    def hessian(self, params: jax.Array) -> jax.Array:
        """REML Hessian w.r.t. params via ``jax.hessian``."""
        return _jit_reml_joint_hess(params, **self._kwargs())

    def grad_hess(self, params: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Fused gradient + Hessian in a single XLA dispatch."""
        return _jit_reml_joint_grad_hess(params, **self._kwargs())


class JointMLCriterion(_JointCriterionBase):
    """ML criterion with joint ``(log_lambda, log_phi)`` optimization.

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    pirls_result : PIRLSResult
        Converged PIRLS output.
    """

    def score(self, params: jax.Array) -> jax.Array:
        """ML score at given params = [log_lambda, log_phi]."""
        return _jit_ml_joint_score(params, **self._kwargs())

    def gradient(self, params: jax.Array) -> jax.Array:
        """ML gradient w.r.t. params via ``jax.grad``."""
        return _jit_ml_joint_grad(params, **self._kwargs())

    def hessian(self, params: jax.Array) -> jax.Array:
        """ML Hessian w.r.t. params via ``jax.hessian``."""
        return _jit_ml_joint_hess(params, **self._kwargs())

    def grad_hess(self, params: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Fused gradient + Hessian in a single XLA dispatch."""
        return _jit_ml_joint_grad_hess(params, **self._kwargs())
