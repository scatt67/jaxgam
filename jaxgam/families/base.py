"""ExponentialFamily base class for GLM/GAM distributions.

Provides the interface contract that all standard exponential family
distributions must implement. The PIRLS algorithm relies on these
methods for working weights, working response, and deviance computation.

Design doc reference: Section 6.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from jaxgam.jax_utils import array_module
from jaxgam.links.links import Link


class ExponentialFamily(ABC):
    """Base class for exponential family distributions.

    Standard families provide:
    - variance(mu): V(mu) -- the variance function
    - deviance_resids(y, mu, wt): per-observation deviance residuals
    - dev_resids(y, mu, wt): scalar total deviance
    - aic(y, mu, wt, scale): AIC contribution
    - initialize(y, wt): starting mu values
    - valid_mu(mu): boolean array of valid mu values
    - valid_eta(eta): boolean array of valid eta values

    The PIRLS algorithm uses:
        working weights: W = 1 / (V(mu) * g'(mu)^2)
        working response: z = eta + (y - mu) * g'(mu)

    Parameters
    ----------
    link : str or Link or None
        Link function specification. If a string, looked up via
        ``Link.from_name()``. If ``None``, uses the family's default link.
    """

    # Subclasses should override these class attributes as needed.
    # n_theta: number of extra distribution parameters (for extended families).
    n_theta: int = 0
    scale_known: bool = False

    def __init__(self, link: str | Link | None = None) -> None:
        if link is None:
            self.link: Link = self.default_link
        elif isinstance(link, str):
            self.link = Link.from_name(link)
        elif isinstance(link, Link):
            self.link = link
        else:
            raise TypeError(
                f"link must be a string, Link instance, or None; got {type(link)!r}"
            )

    @property
    @abstractmethod
    def family_name(self) -> str:
        """Short name for the family (e.g. 'gaussian', 'binomial')."""
        ...

    @property
    @abstractmethod
    def default_link(self) -> Link:
        """Return the default Link instance for this family."""
        ...

    @abstractmethod
    def variance(self, mu: np.ndarray) -> np.ndarray:
        """Variance function V(mu).

        Parameters
        ----------
        mu : np.ndarray
            Mean parameter values.

        Returns
        -------
        np.ndarray
            Variance at each mu value.
        """
        ...

    @abstractmethod
    def dvar(self, mu: np.ndarray) -> np.ndarray:
        """Derivative of the variance function V'(mu).

        Used by the Fletcher (2012) scale estimator.

        Parameters
        ----------
        mu : np.ndarray
            Mean parameter values.

        Returns
        -------
        np.ndarray
            V'(mu) at each mu value.
        """
        ...

    @abstractmethod
    def saturated_loglik(self, y: np.ndarray, wt: np.ndarray, scale: float) -> float:
        """Saturated log-likelihood: log L(y; y, scale, wt).

        The log-likelihood of the saturated model (mu = y). Used in the
        REML/ML criterion (R's ``family$ls``).

        Backend-agnostic: accepts both NumPy and JAX arrays.

        Parameters
        ----------
        y : np.ndarray
            Response values.
        wt : np.ndarray
            Prior weights.
        scale : float
            Dispersion parameter.

        Returns
        -------
        float
            Scalar saturated log-likelihood.
        """
        ...

    @abstractmethod
    def deviance_resids(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Per-observation deviance residuals.

        The deviance residual for observation i is:
            sign(y_i - mu_i) * sqrt(wt_i * d_i)

        where d_i is the unit deviance component.

        Parameters
        ----------
        y : np.ndarray
            Response values.
        mu : np.ndarray
            Fitted mean values.
        wt : np.ndarray
            Prior weights.

        Returns
        -------
        np.ndarray
            Signed deviance residuals, one per observation.
        """
        ...

    def dev_resids(self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray) -> float:
        """Total deviance: sum of weighted unit deviance components.

        This is the sum of the *squared* deviance residuals (i.e. the
        sum of the raw unit deviance contributions before taking square
        roots and applying signs). Equivalently, it is
        ``sum(wt * unit_deviance(y, mu))``.

        Parameters
        ----------
        y : np.ndarray
            Response values.
        mu : np.ndarray
            Fitted mean values.
        wt : np.ndarray
            Prior weights.

        Returns
        -------
        float
            Scalar total deviance.
        """
        dr = self.deviance_resids(y, mu, wt)
        xp = array_module(dr)
        return xp.sum(dr**2)

    @abstractmethod
    def aic(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: float,
    ) -> float:
        """AIC contribution from this family.

        Returns -2 * log_likelihood + 2 * k, where k is the number of
        parameters in the family (typically 0 for known-scale families,
        1 for families with estimated scale).

        Parameters
        ----------
        y : np.ndarray
            Response values.
        mu : np.ndarray
            Fitted mean values.
        wt : np.ndarray
            Prior weights.
        scale : float
            Estimated or known scale parameter.

        Returns
        -------
        float
            AIC contribution (scalar).
        """
        ...

    @abstractmethod
    def initialize(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:
        """Compute starting mu values from the response.

        Called before the first PIRLS iteration to obtain a reasonable
        starting point.

        Parameters
        ----------
        y : np.ndarray
            Response values.
        wt : np.ndarray
            Prior weights.

        Returns
        -------
        np.ndarray
            Starting mu values (same shape as y).
        """
        ...

    @abstractmethod
    def valid_mu(self, mu: np.ndarray) -> np.ndarray:
        """Boolean mask of valid mu values for this family.

        Parameters
        ----------
        mu : np.ndarray
            Mean parameter values.

        Returns
        -------
        np.ndarray
            Boolean array; True where mu is in the valid range.
        """
        ...

    @abstractmethod
    def valid_eta(self, eta: np.ndarray) -> np.ndarray:
        """Boolean mask of valid eta (linear predictor) values.

        Parameters
        ----------
        eta : np.ndarray
            Linear predictor values.

        Returns
        -------
        np.ndarray
            Boolean array; True where eta is in the valid range.
        """
        ...

    def working_weights(self, mu: np.ndarray, wt: np.ndarray) -> np.ndarray:
        """PIRLS working weights: W = wt / (V(mu) * g'(mu)^2).

        Parameters
        ----------
        mu : np.ndarray
            Current mean estimates.
        wt : np.ndarray
            Prior weights.

        Returns
        -------
        np.ndarray
            Working weight for each observation.
        """
        g_prime = self.link.derivative(mu)
        return wt / (self.variance(mu) * g_prime**2)

    def working_response(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        eta: np.ndarray,
    ) -> np.ndarray:
        """PIRLS working response: z = eta + (y - mu) * g'(mu).

        Parameters
        ----------
        y : np.ndarray
            Response values.
        mu : np.ndarray
            Current mean estimates.
        eta : np.ndarray
            Current linear predictor values.

        Returns
        -------
        np.ndarray
            Working response for each observation.
        """
        g_prime = self.link.derivative(mu)
        return eta + (y - mu) * g_prime

    def scale_estimate(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        n: int,
        p: int,
    ) -> float:
        """Estimate the dispersion/scale parameter phi.

        For known-scale families (Binomial, Poisson) this returns 1.0.
        Otherwise returns deviance / (n - p).

        Parameters
        ----------
        y : np.ndarray
            Response values.
        mu : np.ndarray
            Fitted mean values.
        wt : np.ndarray
            Prior weights.
        n : int
            Number of observations.
        p : int
            Number of model parameters.

        Returns
        -------
        float
            Estimated scale parameter.
        """
        if self.scale_known:
            return 1.0
        return self.dev_resids(y, mu, wt) / (n - p)

    def log_likelihood(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        scale: float,
        wt: np.ndarray,
    ) -> float:
        """Log-likelihood. Default: computed from deviance for EDMs.

        Parameters
        ----------
        y : np.ndarray
            Response values.
        mu : np.ndarray
            Fitted mean values.
        scale : float
            Scale parameter.
        wt : np.ndarray
            Prior weights.

        Returns
        -------
        float
            Log-likelihood value.
        """
        return -0.5 * self.dev_resids(y, mu, wt) / scale

    def __repr__(self) -> str:
        return f"{type(self).__name__}(link={type(self.link).__name__})"
