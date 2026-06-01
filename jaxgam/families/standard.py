"""Standard exponential families: Gaussian, Binomial, Poisson, Gamma.

Each family provides the variance function V(mu), deviance residuals,
AIC contribution, initialization, and validity checks needed by the
PIRLS fitting algorithm.

PIRLS-path methods (variance, deviance_resids) are backend-agnostic:
they accept both NumPy and JAX arrays via ``array_module()`` dispatch.

Design doc reference: Section 6.2
R source reference: R/family.R (stats package family definitions)
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from scipy.special import gammaln

from jaxgam.families.base import (
    NON_NEGATIVE,
    POSITIVE,
    REAL,
    UNIT_INTERVAL,
    ExponentialFamily,
)
from jaxgam.jax_utils import array_module
from jaxgam.links.links import IdentityLink, InverseLink, Link, LogitLink, LogLink

# Numerical stability constants for clamping near boundaries.
_MU_EPS = 1e-10
_LOG_EPS = 1e-30


class Gaussian(ExponentialFamily):
    """Gaussian (normal) family with V(mu) = 1.

    Parameters
    ----------
    link : str or Link or None
        Link function. Default is identity.
    """

    family_name: str = "gaussian"
    scale_known: bool = False
    response_support = REAL
    canonical_link_cls = IdentityLink

    @property
    def default_link(self) -> Link:
        return IdentityLink()

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(mu) = 1 for all mu."""
        xp = array_module(mu)
        return xp.ones_like(mu, dtype=float)

    def dvar(self, mu: np.ndarray) -> np.ndarray:
        """V'(mu) = 0 for Gaussian.  Phase 2 only (JAX)."""
        return jnp.zeros_like(mu, dtype=float)

    def saturated_loglik(
        self,
        y: np.ndarray,  # noqa: ARG002
        wt: np.ndarray,
        scale: float,
        *,
        max_y: int = 0,  # noqa: ARG002
    ) -> float:
        """Saturated log-likelihood for Gaussian.  Phase 2 only (JAX).

        R: -nobs*log(2*pi*scale)/2 + sum(log(w[w>0]))/2
        """
        nobs = jnp.sum(wt > 0)
        log_wt = jnp.where(wt > 0, jnp.log(jnp.maximum(wt, _LOG_EPS)), 0.0)
        return -nobs * jnp.log(2.0 * jnp.pi * scale) / 2.0 + jnp.sum(log_wt) / 2.0

    def deviance_resids(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Deviance residuals: sign(y - mu) * sqrt(wt * (y - mu)^2).

        The unit deviance for Gaussian is (y - mu)^2.
        """
        xp = array_module(y)
        d = wt * (y - mu) ** 2
        return xp.sign(y - mu) * xp.sqrt(d)

    def aic(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: float,  # noqa: ARG002
    ) -> float:
        """AIC contribution for Gaussian family.  Phase 3 only (NumPy).

        Matches R's ``gaussian()$aic``::

            nobs * (log(2*pi*dev/nobs) + 1) + 2 - sum(log(wt))

        where ``nobs = len(y)`` and ``dev = sum(wt*(y-mu)^2)``. The dispersion
        is the deviance-based MLE ``dev/nobs`` (not the passed scale), and the
        ``-sum(log wt)`` term accounts for non-unit prior weights.
        """
        nobs = len(y)
        dev = float(np.sum(wt * (y - mu) ** 2))
        sum_log_wt = float(np.sum(np.log(wt[wt > 0])))
        return float(nobs * (np.log(2 * np.pi * dev / nobs) + 1.0) + 2.0 - sum_log_wt)

    def _initialize_impl(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Initialize mu = y for Gaussian."""
        return y.copy()

    def valid_mu(self, mu: np.ndarray) -> np.ndarray:
        """All finite mu are valid for Gaussian."""
        xp = array_module(mu)
        return xp.isfinite(mu)

    def valid_eta(self, eta: np.ndarray) -> np.ndarray:
        """All finite eta are valid for Gaussian."""
        xp = array_module(eta)
        return xp.isfinite(eta)


class Binomial(ExponentialFamily):
    """Binomial family with V(mu) = mu * (1 - mu).

    Parameters
    ----------
    link : str or Link or None
        Link function. Default is logit.
    """

    family_name: str = "binomial"
    scale_known: bool = True
    response_support = UNIT_INTERVAL
    canonical_link_cls = LogitLink

    @property
    def default_link(self) -> Link:
        return LogitLink()

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(mu) = mu * (1 - mu)."""
        return mu * (1.0 - mu)

    def dvar(self, mu: np.ndarray) -> np.ndarray:
        """V'(mu) = 1 - 2*mu for Binomial."""
        return 1.0 - 2.0 * mu

    def saturated_loglik(
        self,
        y: np.ndarray,
        wt: np.ndarray,
        scale: float,  # noqa: ARG002
        *,
        max_y: int = 0,  # noqa: ARG002
    ) -> float:
        """Saturated log-likelihood for Binomial.  Phase 2 only (JAX).

        R: ``-binomial()$aic(y, n, y, w, 0) / 2`` which, at saturation mu=y,
        equals ``sum(wt * [y*log(y) + (1-y)*log(1-y)]) + sum(lchoose(m, m*y))``.
        The binomial-coefficient term ``lchoose`` is zero for Bernoulli (wt=1)
        but a large nonzero constant for grouped/trial-count binomial (wt>1);
        omitting it makes the reported REML score wrong by that constant.
        """
        y_safe = jnp.clip(y, _MU_EPS, 1.0 - _MU_EPS)
        interior = (y > 0) & (y < 1)
        ll = jnp.where(
            interior,
            y * jnp.log(y_safe) + (1.0 - y) * jnp.log(1.0 - y_safe),
            0.0,
        )
        # Binomial-coefficient term (R binomial()$aic via fix.family.ls):
        # m = trial count = prior weight wt; k = successes = round(m*y);
        # lchoose(m, k) = lgamma(m+1) - lgamma(k+1) - lgamma(m-k+1).
        # For Bernoulli (wt=1) this is identically 0, preserving that case.
        m = jnp.round(wt)
        k = jnp.round(wt * y)
        lchoose = jsp.gammaln(m + 1.0) - jsp.gammaln(k + 1.0) - jsp.gammaln(m - k + 1.0)
        lchoose = jnp.where(wt > 0, lchoose, 0.0)
        return jnp.sum(wt * ll) + jnp.sum(lchoose)

    def deviance_resids(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Deviance residuals for Binomial.

        Unit deviance: 2 * [y * log(y/mu) + (1-y) * log((1-y)/(1-mu))]
        with edge-case handling for y=0 and y=1.

        Matches R's binomial()$dev.resids.
        """
        xp = array_module(y)
        mu_safe = xp.clip(mu, _MU_EPS, 1.0 - _MU_EPS)

        y_pos = xp.where(y > 0, y, 1.0)
        y1_pos = xp.where(y < 1, 1.0 - y, 1.0)
        term1 = y * xp.log(y_pos / mu_safe)
        term2 = (1.0 - y) * xp.log(y1_pos / (1.0 - mu_safe))

        d = 2.0 * wt * (term1 + term2)
        d = xp.maximum(d, 0.0)
        return xp.sign(y - mu_safe) * xp.sqrt(d)

    def aic(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: float,  # noqa: ARG002
    ) -> float:
        """AIC contribution for Binomial family.  Phase 3 only (NumPy).

        Matches R's ``binomial()$aic``:
        ``-2 * sum((wt/m) * dbinom(round(m*y), round(m), mu, log=TRUE))``
        with single-column trial count ``m = wt``. Expanding the binomial pmf
        gives the ``lchoose(m, m*y)`` term (zero for Bernoulli, wt=1).
        """
        mu_safe = np.clip(mu, _MU_EPS, 1.0 - _MU_EPS)
        ll = wt * (y * np.log(mu_safe) + (1.0 - y) * np.log(1.0 - mu_safe))
        m = np.round(wt)
        k = np.round(wt * y)
        lchoose = gammaln(m + 1.0) - gammaln(k + 1.0) - gammaln(m - k + 1.0)
        lchoose = np.where(wt > 0, lchoose, 0.0)
        return float(-2.0 * (np.sum(ll) + np.sum(lchoose)))

    def _initialize_impl(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:
        """Initialize mu for Binomial: ``(wt*y + 0.5) / (wt + 1)``.

        Matches R's ``binomial()$initialize`` (prior-weight aware). For unit
        weights this reduces to ``(y + 0.5)/2``, mapping y in {0, 1} to mu in
        (0.25, 0.75), safely away from the boundary.
        """
        return (wt * y + 0.5) / (wt + 1.0)

    def valid_mu(self, mu: np.ndarray) -> np.ndarray:
        """Valid mu for Binomial: 0 < mu < 1."""
        return (mu > 0) & (mu < 1)

    def valid_eta(self, eta: np.ndarray) -> np.ndarray:
        """All finite eta are valid for Binomial."""
        xp = array_module(eta)
        return xp.isfinite(eta)


class Poisson(ExponentialFamily):
    """Poisson family with V(mu) = mu.

    Parameters
    ----------
    link : str or Link or None
        Link function. Default is log.
    """

    family_name: str = "poisson"
    scale_known: bool = True
    response_support = NON_NEGATIVE
    canonical_link_cls = LogLink

    @property
    def default_link(self) -> Link:
        return LogLink()

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(mu) = mu."""
        xp = array_module(mu)
        return xp.asarray(mu, dtype=float)

    def dvar(self, mu: np.ndarray) -> np.ndarray:
        """V'(mu) = 1 for Poisson.  Phase 2 only (JAX)."""
        return jnp.ones_like(mu, dtype=float)

    def saturated_loglik(
        self,
        y: np.ndarray,
        wt: np.ndarray,
        scale: float,  # noqa: ARG002
        *,
        max_y: int = 0,  # noqa: ARG002
    ) -> float:
        """Saturated log-likelihood for Poisson.  Phase 2 only (JAX).

        R: sum(dpois(y, y, log=TRUE) * w)
        = sum(w * [y*log(y) - y - lgamma(y+1)]) for y > 0, else 0.
        """
        y_safe = jnp.where(y > 0, y, 1.0)
        term = jnp.where(
            y > 0,
            y * jnp.log(y_safe) - y - jsp.gammaln(y + 1.0),
            0.0,
        )
        return jnp.sum(wt * term)

    def deviance_resids(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Deviance residuals for Poisson.

        Unit deviance: 2 * [y * log(y/mu) - (y - mu)]
        with y=0 handled as a special case (term = 0).

        Matches R's poisson()$dev.resids.
        """
        xp = array_module(y)
        mu_safe = xp.maximum(mu, _MU_EPS)

        y_pos = xp.where(y > 0, y, 1.0)
        term1 = y * xp.log(y_pos / mu_safe)
        d = 2.0 * wt * (term1 - (y - mu_safe))
        d = xp.maximum(d, 0.0)
        return xp.sign(y - mu_safe) * xp.sqrt(d)

    def aic(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: float,  # noqa: ARG002
    ) -> float:
        """AIC contribution for Poisson family.  Phase 3 only (NumPy).

        Matches R: -2 * sum(wt * dpois(y, mu, log=TRUE))
        = -2 * sum(wt * (y*log(mu) - mu - lgamma(y+1)))
        """
        mu_safe = np.maximum(mu, _MU_EPS)
        ll = wt * (y * np.log(mu_safe) - mu_safe - gammaln(y + 1.0))
        return float(-2.0 * np.sum(ll))

    def _initialize_impl(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Initialize mu for Poisson: ``mu = y + 0.1`` for ALL observations.

        Matches R's ``poisson()$initialize`` (``mustart <- y + 0.1``), which
        bumps every observation, not only zeros, avoiding log(0) in the first
        evaluation of the working quantities.
        """
        return y + 0.1

    def valid_mu(self, mu: np.ndarray) -> np.ndarray:
        """Valid mu for Poisson: mu > 0."""
        return mu > 0

    def valid_eta(self, eta: np.ndarray) -> np.ndarray:
        """All finite eta are valid for Poisson."""
        xp = array_module(eta)
        return xp.isfinite(eta)


class Gamma(ExponentialFamily):
    """Gamma family with V(mu) = mu^2.

    Parameters
    ----------
    link : str or Link or None
        Link function. Default is inverse (1/mu).
    """

    family_name: str = "Gamma"
    scale_known: bool = False
    response_support = POSITIVE
    canonical_link_cls = InverseLink

    @property
    def default_link(self) -> Link:
        return InverseLink()

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(mu) = mu^2."""
        return mu**2

    def dvar(self, mu: np.ndarray) -> np.ndarray:
        """V'(mu) = 2*mu for Gamma.  Phase 2 only (JAX)."""
        return 2.0 * mu

    def saturated_loglik(
        self,
        y: np.ndarray,
        wt: np.ndarray,
        scale: float,
        *,
        max_y: int = 0,  # noqa: ARG002
    ) -> float:
        """Saturated log-likelihood for Gamma.  Phase 2 only (JAX).

        R's fix.family.ls (gam.fit3.r line 2519):
            scale_i = scale / w_i  (per-observation scale)
            k_i = -lgamma(1/scale_i) - log(scale_i)/scale_i - 1/scale_i
            ls = sum(k_i - log(y_i))
        """
        # Per-observation scale: phi_i = scale / wt_i
        wt_safe = jnp.maximum(wt, _LOG_EPS)
        inv_phi = wt_safe / scale  # 1 / phi_i = wt_i / scale
        phi = scale / wt_safe

        k = -jsp.gammaln(inv_phi) - jnp.log(phi) * inv_phi - inv_phi
        y_safe = jnp.maximum(y, _LOG_EPS)
        return jnp.sum(jnp.where(wt > 0, k - jnp.log(y_safe), 0.0))

    def deviance_resids(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Deviance residuals for Gamma.

        Unit deviance: 2 * [-log(y/mu) + (y - mu)/mu]
        = -2 * [log(y/mu) - (y - mu)/mu]

        Matches R's Gamma()$dev.resids.
        """
        xp = array_module(y)
        mu_safe = xp.maximum(mu, _MU_EPS)
        y_safe = xp.maximum(y, _MU_EPS)

        d = 2.0 * wt * (-xp.log(y_safe / mu_safe) + (y - mu_safe) / mu_safe)
        d = xp.maximum(d, 0.0)
        return xp.sign(y - mu_safe) * xp.sqrt(d)

    def aic(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: float,  # noqa: ARG002
    ) -> float:
        """AIC contribution for Gamma family.  Phase 3 only (NumPy).

        Matches R's ``Gamma()$aic``::

            disp = dev / sum(wt)
            -2 * sum(wt * dgamma(y, 1/disp, scale=mu*disp, log=TRUE)) + 2

        where ``dev = 2*sum(wt*(-log(y/mu) + (y-mu)/mu))``. R uses the
        deviance-based dispersion ``dev/sum(wt)`` (not the passed scale); the
        ``+2`` accounts for the estimated dispersion parameter.
        """
        y_safe = np.maximum(y, _MU_EPS)
        mu_safe = np.maximum(mu, _MU_EPS)

        unit_dev = -np.log(y_safe / mu_safe) + (y - mu_safe) / mu_safe
        dev = 2.0 * float(np.sum(wt * unit_dev))
        disp = dev / float(np.sum(wt))
        shape = 1.0 / disp

        ll = wt * (
            (shape - 1.0) * np.log(y_safe)
            - y_safe / (mu_safe * disp)
            - shape * np.log(mu_safe * disp)
            - gammaln(shape)
        )
        return float(-2.0 * np.sum(ll) + 2.0)

    def _initialize_impl(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Initialize mu for Gamma: mu = y, clipped to positive values.

        Follows R's Gamma()$initialize which ensures mu > 0.
        """
        return np.maximum(y, np.finfo(float).eps)

    def valid_mu(self, mu: np.ndarray) -> np.ndarray:
        """Valid mu for Gamma: mu > 0."""
        return mu > 0

    def valid_eta(self, eta: np.ndarray) -> np.ndarray:
        """Valid eta for Gamma.

        For the inverse link (Gamma's default) R's ``valideta`` is
        ``is.finite(eta) & all(eta != 0)`` — eta==0 maps to mu=inf. Other
        links only require finiteness. The ``isinstance`` check is on the
        static link object (resolved at trace time), so this stays JAX-safe.
        """
        xp = array_module(eta)
        finite = xp.isfinite(eta)
        if isinstance(self.link, InverseLink):
            return finite & (eta != 0)
        return finite
