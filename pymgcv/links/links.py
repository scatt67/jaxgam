"""Link function implementations for GLM/GAM families.

Each link function provides the mapping g(μ) = η between the mean μ
and the linear predictor η, along with inverse and derivatives needed
by PIRLS.

All methods are backend-agnostic: they accept both NumPy and JAX arrays,
dispatching to the appropriate backend at runtime via ``array_module()``.

Design doc reference: §6.4
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm

from pymgcv.jax_utils import array_module, is_jax_array


class Link(ABC):
    """Abstract link function: g(μ) = η, g⁻¹(η) = μ.

    Subclasses must implement ``link``, ``inverse``, and ``derivative``.
    The convenience methods ``linkinv`` (alias for ``inverse``) and
    ``mu_eta`` (derivative of inverse link) are provided on the base class
    but may be overridden for numerical stability.
    """

    @abstractmethod
    def link(self, mu: np.ndarray) -> np.ndarray:
        """Apply the link function: η = g(μ).

        Parameters
        ----------
        mu : np.ndarray
            Mean parameter values.

        Returns
        -------
        np.ndarray
            Linear predictor values.
        """

    @abstractmethod
    def inverse(self, eta: np.ndarray) -> np.ndarray:
        """Apply the inverse link: μ = g⁻¹(η).

        Parameters
        ----------
        eta : np.ndarray
            Linear predictor values.

        Returns
        -------
        np.ndarray
            Mean parameter values.
        """

    @abstractmethod
    def derivative(self, mu: np.ndarray) -> np.ndarray:
        """Link derivative: dη/dμ = g'(μ).

        Parameters
        ----------
        mu : np.ndarray
            Mean parameter values.

        Returns
        -------
        np.ndarray
            Derivative of link function evaluated at mu.
        """

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """Alias for ``inverse``: μ = g⁻¹(η)."""
        return self.inverse(eta)

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        """Derivative of the inverse link: dμ/dη = 1/g'(g⁻¹(η)).

        This default implementation computes via the chain rule.
        Subclasses may override for numerical stability.

        Parameters
        ----------
        eta : np.ndarray
            Linear predictor values.

        Returns
        -------
        np.ndarray
            Derivative of inverse link evaluated at eta.
        """
        mu = self.inverse(eta)
        return 1.0 / self.derivative(mu)

    @staticmethod
    def from_name(name: str) -> Link:
        """Look up a link function by name.

        Parameters
        ----------
        name : str
            Link name (e.g. "logit", "log", "identity").

        Returns
        -------
        Link
            An instance of the corresponding link class.

        Raises
        ------
        KeyError
            If the name is not in the registry.
        """
        return _LINK_REGISTRY[name]()


class IdentityLink(Link):
    """Identity link: g(μ) = μ."""

    def link(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        return xp.asarray(mu, dtype=float)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        xp = array_module(eta)
        return xp.asarray(eta, dtype=float)

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        return xp.ones_like(mu, dtype=float)

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        xp = array_module(eta)
        return xp.ones_like(eta, dtype=float)


class LogLink(Link):
    """Log link: g(μ) = log(μ)."""

    def link(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        return xp.log(xp.maximum(mu, 1e-10))

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        xp = array_module(eta)
        return xp.exp(eta)

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        return 1.0 / xp.maximum(mu, 1e-10)

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        xp = array_module(eta)
        return xp.exp(eta)


class LogitLink(Link):
    """Logit link: g(μ) = log(μ/(1-μ))."""

    def link(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        mu = xp.clip(mu, 1e-10, 1 - 1e-10)
        return xp.log(mu / (1 - mu))

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        xp = array_module(eta)
        return 1.0 / (1.0 + xp.exp(-eta))

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        mu = xp.clip(mu, 1e-10, 1 - 1e-10)
        return 1.0 / (mu * (1 - mu))

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        xp = array_module(eta)
        p = 1.0 / (1.0 + xp.exp(-eta))
        return p * (1.0 - p)


class InverseLink(Link):
    """Inverse (reciprocal) link: g(μ) = 1/μ."""

    def link(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        return 1.0 / xp.maximum(mu, 1e-10)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return 1.0 / eta

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        return -1.0 / xp.maximum(mu, 1e-10) ** 2


class ProbitLink(Link):
    """Probit link: g(μ) = Φ⁻¹(μ)."""

    def link(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        mu_clipped = xp.clip(mu, 1e-10, 1 - 1e-10)
        if is_jax_array(mu):
            from jax.scipy.special import ndtri as jndtri

            return jndtri(mu_clipped)
        return ndtri(mu_clipped)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        if is_jax_array(eta):
            from jax.scipy.special import ndtr as jndtr

            return jndtr(eta)
        return ndtr(eta)

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        if is_jax_array(mu):
            from jax.scipy.stats import norm as jnorm

            return 1.0 / jnorm.pdf(self.link(mu))
        return 1.0 / norm.pdf(self.link(mu))

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        if is_jax_array(eta):
            from jax.scipy.stats import norm as jnorm

            return jnorm.pdf(eta)
        return norm.pdf(eta)


class CloglogLink(Link):
    """Complementary log-log link: g(μ) = log(-log(1-μ))."""

    def link(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        mu = xp.clip(mu, 1e-10, 1 - 1e-10)
        return xp.log(-xp.log(1 - mu))

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        xp = array_module(eta)
        return 1.0 - xp.exp(-xp.exp(eta))

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        mu = xp.clip(mu, 1e-10, 1 - 1e-10)
        return 1.0 / ((1 - mu) * (-xp.log(1 - mu)))

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        xp = array_module(eta)
        return xp.exp(eta - xp.exp(eta))


class SqrtLink(Link):
    """Square root link: g(μ) = √μ."""

    def link(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        return xp.sqrt(xp.maximum(mu, 1e-10))

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return eta**2

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        return 0.5 / xp.sqrt(xp.maximum(mu, 1e-10))

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        return 2.0 * eta


class InverseSquaredLink(Link):
    """Inverse squared link: g(μ) = 1/μ².

    Default link for Inverse Gaussian family.
    """

    def link(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        return 1.0 / xp.maximum(mu, 1e-10) ** 2

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        xp = array_module(eta)
        return 1.0 / xp.sqrt(xp.maximum(eta, 1e-10))

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        xp = array_module(mu)
        return -2.0 / xp.maximum(mu, 1e-10) ** 3


_LINK_REGISTRY: dict[str, type[Link]] = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "inverse": InverseLink,
    "probit": ProbitLink,
    "cloglog": CloglogLink,
    "sqrt": SqrtLink,
    "inverse_squared": InverseSquaredLink,
}
