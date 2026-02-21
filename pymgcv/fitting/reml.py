"""REML and ML criterion functions for smoothing parameter selection.

Given fixed smoothing parameters lambda, PIRLS finds optimal coefficients beta*(lambda).
The REML/ML criteria score how good a particular lambda is -- the Newton optimizer
(Task 2.5) minimizes this criterion to find optimal lambda.

The key insight (Laplace approximation + PIRLS stationarity): at converged beta*,
the derivative of the penalized deviance through beta is zero. So ``jax.grad``
on the criterion w.r.t. log(lambda) gives the correct gradient when beta*, XtWX, mu
are treated as constants (not differentiated through). This matches R's
analytical IFT approach.

We do NOT differentiate through PIRLS -- we run PIRLS, extract results, pass
them as fixed inputs. Only ``log_lambda`` flows through the AD trace (through
S_lambda construction and log-det computations).

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

Scale estimation uses Fletcher (2012) correction by default:
    phi_pearson = sum(w_i (y_i - mu_i)^2 / V(mu_i)) / (n - trA)
    s_bar = max(-0.9, mean(V'(mu) * (y - mu) / V(mu)))
    phi_fletcher = phi_pearson / (1 + s_bar)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from pymgcv.jax_utils import build_S_lambda, log_pseudo_det

if TYPE_CHECKING:
    from pymgcv.families.base import ExponentialFamily
    from pymgcv.fitting.data import FittingData
    from pymgcv.fitting.pirls import PIRLSResult

jax.config.update("jax_enable_x64", True)


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
    lambda _, children: REMLResult(**dict(zip(_REML_FIELDS, children))),
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
) -> jax.Array:
    """Shared REML/ML computation. Pure JAX, differentiable.

    Computes: Dp/(2*phi) - ls_sat + log|H|/2 - log|S+|/2

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
    sign, log_det_H = jnp.linalg.slogdet(H)
    log_det_H = jnp.where(sign > 0, log_det_H, 1e10)

    log_det_S = log_pseudo_det(S_lambda)

    return Dp / (2.0 * phi) - ls_sat + log_det_H / 2.0 - log_det_S / 2.0


# ---------------------------------------------------------------------------
# Scale estimation
# ---------------------------------------------------------------------------


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
    pearson = pearson_rss(y, mu, wt, family)
    phi_pearson = pearson / (n - edf)

    V = family.variance(mu)
    dV = family.dvar(mu)
    s_bar = jnp.maximum(-0.9, jnp.mean(dV * (y - mu) / V))
    return phi_pearson / (1.0 + s_bar)


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
    XtWX : jax.Array, shape (p, p)
        Weighted cross-product from PIRLS (PIRLSResult.XtWX).
    beta : jax.Array, shape (p,)
        Converged coefficients (PIRLSResult.coefficients).
    deviance : jax.Array, scalar
        Unpenalized deviance from PIRLS (PIRLSResult.deviance).
    ls_sat : jax.Array, scalar
        Saturated log-likelihood (family.saturated_loglik).
    S_list : tuple[jax.Array, ...]
        Per-penalty (p, p) matrices (FittingData.S_list).
    phi : jax.Array, scalar
        Dispersion parameter.
    Mp : int (static)
        Penalty null space dimension = sum(penalty_null_dims).

    Returns
    -------
    jax.Array, scalar
        REML score (lower is better).
    """
    core = _criterion_core(log_lambda, XtWX, beta, deviance, ls_sat, S_list, phi)
    return core - Mp / 2.0 * jnp.log(2.0 * jnp.pi * phi)


def ml_criterion(
    log_lambda: jax.Array,
    XtWX: jax.Array,
    beta: jax.Array,
    deviance: jax.Array,
    ls_sat: jax.Array,
    S_list: tuple[jax.Array, ...],
    phi: jax.Array,
) -> jax.Array:
    """ML criterion. Same as REML but without the -Mp/2*log(2*pi*phi) correction.

    Formula::

        ML = Dp/(2*phi) - ls_sat + log|H|/2 - log|S+|/2

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

    Returns
    -------
    jax.Array, scalar
        ML score (lower is better).
    """
    return _criterion_core(log_lambda, XtWX, beta, deviance, ls_sat, S_list, phi)


# ---------------------------------------------------------------------------
# Criterion classes
# ---------------------------------------------------------------------------


class _CriterionBase(ABC):
    """Base class for REML/ML criterion evaluation.

    Precomputes and caches constants from converged PIRLS.
    Subclasses implement ``score()`` with the specific criterion function.
    """

    def __init__(self, fd: FittingData, pirls_result: PIRLSResult) -> None:
        self.edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
        self.scale = estimate_scale(fd.y, pirls_result.mu, fd.wt, fd.family, self.edf)
        self._deviance = pirls_result.deviance
        self._ls_sat = fd.family.saturated_loglik(fd.y, fd.wt, self.scale)
        self._XtWX = pirls_result.XtWX
        self._beta = pirls_result.coefficients
        self._S_list = fd.S_list

    @abstractmethod
    def score(self, log_lambda: jax.Array) -> jax.Array: ...

    def gradient(self, log_lambda: jax.Array) -> jax.Array:
        """Gradient via ``jax.grad``."""
        return jax.grad(self.score)(log_lambda)

    def hessian(self, log_lambda: jax.Array) -> jax.Array:
        """Hessian via ``jax.hessian``."""
        return jax.hessian(self.score)(log_lambda)

    def evaluate(self, log_lambda: jax.Array) -> REMLResult:
        """Full evaluation returning ``REMLResult``."""
        return REMLResult(score=self.score(log_lambda), edf=self.edf, scale=self.scale)


class REMLCriterion(_CriterionBase):
    """REML criterion for smoothing parameter optimization.

    Precomputes and caches constants from converged PIRLS at construction.
    Provides ``score()``, ``gradient()``, ``hessian()`` for the Newton
    optimizer, and ``evaluate()`` for full diagnostic output.

    Usage by the Newton optimizer (Task 2.5)::

        obj = REMLCriterion(fd, pirls_result)
        score = obj.score(log_lambda)
        grad = obj.gradient(log_lambda)
        hess = obj.hessian(log_lambda)
        result = obj.evaluate(log_lambda)  # -> REMLResult

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    pirls_result : PIRLSResult
        Converged PIRLS output.
    """

    def __init__(self, fd: FittingData, pirls_result: PIRLSResult) -> None:
        super().__init__(fd, pirls_result)
        self._Mp = fd.total_penalty_null_dim

    def score(self, log_lambda: jax.Array) -> jax.Array:
        """REML score at given log_lambda. Differentiable via ``jax.grad``."""
        return reml_criterion(
            log_lambda,
            self._XtWX,
            self._beta,
            self._deviance,
            self._ls_sat,
            self._S_list,
            self.scale,
            self._Mp,
        )


class MLCriterion(_CriterionBase):
    """ML criterion for smoothing parameter optimization.

    Same interface as ``REMLCriterion`` but uses the ML formula
    (no ``-Mp/2*log(2*pi*phi)`` correction).

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    pirls_result : PIRLSResult
        Converged PIRLS output.
    """

    def score(self, log_lambda: jax.Array) -> jax.Array:
        """ML score at given log_lambda. Differentiable via ``jax.grad``."""
        return ml_criterion(
            log_lambda,
            self._XtWX,
            self._beta,
            self._deviance,
            self._ls_sat,
            self._S_list,
            self.scale,
        )
