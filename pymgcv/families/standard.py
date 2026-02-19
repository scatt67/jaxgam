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

import numpy as np

from pymgcv.families.base import ExponentialFamily
from pymgcv.jax_utils import array_module
from pymgcv.links.links import IdentityLink, InverseLink, Link, LogitLink, LogLink


class Gaussian(ExponentialFamily):
    """Gaussian (normal) family with V(mu) = 1.

    Parameters
    ----------
    link : str or Link or None
        Link function. Default is identity.
    """

    family_name: str = "gaussian"  # type: ignore[assignment]
    scale_known: bool = False

    @property
    def default_link(self) -> Link:
        return IdentityLink()

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(mu) = 1 for all mu."""
        xp = array_module(mu)
        return xp.ones_like(mu, dtype=float)

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
        scale: float,
    ) -> float:
        """AIC contribution for Gaussian family.

        Matches R's gaussian()$aic which returns:
            sum(wt) * (log(2*pi*scale) + 1) + 2
        where the +2 accounts for the estimated scale parameter.
        """
        n = np.sum(wt)
        return float(n * (np.log(2 * np.pi * scale) + 1) + 2)

    def initialize(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:
        """Initialize mu = y for Gaussian."""
        return np.asarray(y, dtype=float).copy()

    def valid_mu(self, mu: np.ndarray) -> np.ndarray:
        """All finite mu are valid for Gaussian."""
        return np.isfinite(mu)

    def valid_eta(self, eta: np.ndarray) -> np.ndarray:
        """All finite eta are valid for Gaussian with identity link."""
        return np.isfinite(eta)


class Binomial(ExponentialFamily):
    """Binomial family with V(mu) = mu * (1 - mu).

    Parameters
    ----------
    link : str or Link or None
        Link function. Default is logit.
    """

    family_name: str = "binomial"  # type: ignore[assignment]
    scale_known: bool = True

    @property
    def default_link(self) -> Link:
        return LogitLink()

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(mu) = mu * (1 - mu)."""
        return mu * (1.0 - mu)

    def deviance_resids(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Deviance residuals for Binomial.

        Unit deviance: 2 * [y * log(y/mu) + (1-y) * log((1-y)/(1-mu))]
        with edge-case handling for y=0 and y=1.

        Matches R's binomial()$dev.resids.
        """
        xp = array_module(y)
        mu_safe = xp.clip(mu, 1e-10, 1.0 - 1e-10)

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
        scale: float,
    ) -> float:
        """AIC contribution for Binomial family.

        Matches R: -2 * sum(wt * dbinom(y, 1, mu, log=TRUE))
        For Bernoulli trials: -2 * sum(wt * [y*log(mu) + (1-y)*log(1-mu)])
        """
        mu_safe = np.clip(mu, 1e-10, 1.0 - 1e-10)
        ll = wt * (y * np.log(mu_safe) + (1.0 - y) * np.log(1.0 - mu_safe))
        return float(-2.0 * np.sum(ll))

    def initialize(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:
        """Initialize mu = (y + 0.5) / 2 for Binomial.

        This R convention maps y in {0, 1} to mu in (0.25, 0.75),
        ensuring the starting mu is safely away from the boundary.
        """
        return (np.asarray(y, dtype=float) + 0.5) / 2.0

    def valid_mu(self, mu: np.ndarray) -> np.ndarray:
        """Valid mu for Binomial: 0 < mu < 1."""
        return (mu > 0) & (mu < 1)

    def valid_eta(self, eta: np.ndarray) -> np.ndarray:
        """All finite eta are valid for Binomial."""
        return np.isfinite(eta)


class Poisson(ExponentialFamily):
    """Poisson family with V(mu) = mu.

    Parameters
    ----------
    link : str or Link or None
        Link function. Default is log.
    """

    family_name: str = "poisson"  # type: ignore[assignment]
    scale_known: bool = True

    @property
    def default_link(self) -> Link:
        return LogLink()

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(mu) = mu."""
        xp = array_module(mu)
        return xp.asarray(mu, dtype=float)

    def deviance_resids(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Deviance residuals for Poisson.

        Unit deviance: 2 * [y * log(y/mu) - (y - mu)]
        with y=0 handled as a special case (term = 0).

        Matches R's poisson()$dev.resids.
        """
        xp = array_module(y)
        mu_safe = xp.maximum(mu, 1e-10)

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
        scale: float,
    ) -> float:
        """AIC contribution for Poisson family.

        Matches R: -2 * sum(wt * dpois(y, mu, log=TRUE))
        = -2 * sum(wt * (y*log(mu) - mu - lgamma(y+1)))
        """
        from scipy.special import gammaln

        mu_safe = np.maximum(mu, 1e-10)
        ll = wt * (y * np.log(mu_safe) - mu_safe - gammaln(y + 1.0))
        return float(-2.0 * np.sum(ll))

    def initialize(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:
        """Initialize mu for Poisson: mu = y + 0.1 where y == 0.

        Following R's convention, this avoids log(0) in the first
        evaluation of the working quantities.
        """
        y = np.asarray(y, dtype=float)
        return np.where(y == 0, y + 0.1, y)

    def valid_mu(self, mu: np.ndarray) -> np.ndarray:
        """Valid mu for Poisson: mu > 0."""
        return mu > 0

    def valid_eta(self, eta: np.ndarray) -> np.ndarray:
        """All finite eta are valid for Poisson."""
        return np.isfinite(eta)


class Gamma(ExponentialFamily):
    """Gamma family with V(mu) = mu^2.

    Parameters
    ----------
    link : str or Link or None
        Link function. Default is inverse (1/mu).
    """

    family_name: str = "Gamma"  # type: ignore[assignment]
    scale_known: bool = False

    @property
    def default_link(self) -> Link:
        return InverseLink()

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(mu) = mu^2."""
        return mu**2

    def deviance_resids(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Deviance residuals for Gamma.

        Unit deviance: 2 * [-log(y/mu) + (y - mu)/mu]
        = -2 * [log(y/mu) - (y - mu)/mu]

        Matches R's Gamma()$dev.resids.
        """
        xp = array_module(y)
        mu_safe = xp.maximum(mu, 1e-10)
        y_safe = xp.maximum(y, 1e-10)

        d = 2.0 * wt * (-xp.log(y_safe / mu_safe) + (y - mu_safe) / mu_safe)
        d = xp.maximum(d, 0.0)
        return xp.sign(y - mu_safe) * xp.sqrt(d)

    def aic(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: float,
    ) -> float:
        """AIC contribution for Gamma family.

        Matches R's Gamma()$aic:
            -2 * sum(wt * dgamma(y, shape=1/scale, scale=mu*scale, log=TRUE)) + 2
        The +2 accounts for the estimated scale parameter.
        """
        from scipy.special import gammaln

        disp = scale
        shape = 1.0 / disp
        y_safe = np.maximum(y, 1e-10)
        mu_safe = np.maximum(mu, 1e-10)

        ll = wt * (
            (shape - 1.0) * np.log(y_safe)
            - y_safe / (mu_safe * disp)
            - shape * np.log(mu_safe * disp)
            - gammaln(shape)
        )
        return float(-2.0 * np.sum(ll) + 2.0)

    def initialize(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:
        """Initialize mu for Gamma: mu = y, clipped to positive values.

        Follows R's Gamma()$initialize which ensures mu > 0.
        """
        y = np.asarray(y, dtype=float)
        return np.maximum(y, np.finfo(float).eps)

    def valid_mu(self, mu: np.ndarray) -> np.ndarray:
        """Valid mu for Gamma: mu > 0."""
        return mu > 0

    def valid_eta(self, eta: np.ndarray) -> np.ndarray:
        """All finite eta are valid for Gamma."""
        return np.isfinite(eta)
