"""Link function implementations for GLM/GAM families.

Each link function provides the mapping g(μ) = η between the mean μ
and the linear predictor η, along with inverse and derivatives needed
by PIRLS.

All methods are backend-agnostic: they accept both NumPy and JAX arrays,
dispatching to the appropriate backend at runtime via ``array_module()``.
Methods use defensive clamping (``xp.maximum``, ``xp.clip``) instead of
raising exceptions on out-of-domain inputs, matching R's ``make.link``
behavior and preserving JAX JIT compatibility.

Design doc reference: §7
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm

from pymgcv.jax_utils import array_module, is_jax_array

if TYPE_CHECKING:
    import jax

    Array = np.ndarray | jax.Array

# Guard against log(0), 1/0, and similar singularities in link functions.
# Chosen as a practical balance between numerical safety and precision;
# smaller than sqrt(eps) ≈ 1.5e-8 but large enough to avoid underflow.
_EPS = 1e-10


class Link(ABC):
    """Abstract link function: g(μ) = η, g⁻¹(η) = μ.

    Subclasses must implement ``link``, ``inverse``, and ``derivative``.
    The convenience methods ``linkinv`` (alias for ``inverse``) and
    ``mu_eta`` (derivative of inverse link) are provided on the base class
    but may be overridden for numerical stability.
    """

    @abstractmethod
    def link(self, mu: Array) -> Array:
        """Apply the link function: η = g(μ).

        Parameters
        ----------
        mu : Array
            Mean parameter values (NumPy or JAX array).

        Returns
        -------
        Array
            Linear predictor values.
        """

    @abstractmethod
    def inverse(self, eta: Array) -> Array:
        """Apply the inverse link: μ = g⁻¹(η).

        Parameters
        ----------
        eta : Array
            Linear predictor values (NumPy or JAX array).

        Returns
        -------
        Array
            Mean parameter values.
        """

    @abstractmethod
    def derivative(self, mu: Array) -> Array:
        """Link derivative: dη/dμ = g'(μ).

        Parameters
        ----------
        mu : Array
            Mean parameter values (NumPy or JAX array).

        Returns
        -------
        Array
            Derivative of link function evaluated at mu.
        """

    def linkinv(self, eta: Array) -> Array:
        """Alias for ``inverse``: μ = g⁻¹(η)."""
        return self.inverse(eta)

    def mu_eta(self, eta: Array) -> Array:
        """Derivative of the inverse link: dμ/dη = 1/g'(g⁻¹(η)).

        This default implementation computes via the chain rule.
        Subclasses may override for numerical stability.

        Parameters
        ----------
        eta : Array
            Linear predictor values (NumPy or JAX array).

        Returns
        -------
        Array
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
        ValueError
            If the name is not in the registry.
        """
        try:
            return _LINK_REGISTRY[name]()
        except KeyError:
            valid = ", ".join(sorted(_LINK_REGISTRY))
            raise ValueError(
                f"Unknown link function {name!r}. Valid options: {valid}"
            ) from None


class IdentityLink(Link):
    """Identity link: g(μ) = μ."""

    def link(self, mu: Array) -> Array:
        xp = array_module(mu)
        return xp.asarray(mu, dtype=float)

    def inverse(self, eta: Array) -> Array:
        xp = array_module(eta)
        return xp.asarray(eta, dtype=float)

    def derivative(self, mu: Array) -> Array:
        xp = array_module(mu)
        return xp.ones_like(mu, dtype=float)

    def mu_eta(self, eta: Array) -> Array:
        xp = array_module(eta)
        return xp.ones_like(eta, dtype=float)


class LogLink(Link):
    """Log link: g(μ) = log(μ)."""

    def link(self, mu: Array) -> Array:
        xp = array_module(mu)
        return xp.log(xp.maximum(mu, _EPS))

    def inverse(self, eta: Array) -> Array:
        xp = array_module(eta)
        return xp.exp(eta)

    def derivative(self, mu: Array) -> Array:
        xp = array_module(mu)
        return 1.0 / xp.maximum(mu, _EPS)

    def mu_eta(self, eta: Array) -> Array:
        xp = array_module(eta)
        return xp.exp(eta)


class LogitLink(Link):
    """Logit link: g(μ) = log(μ/(1-μ))."""

    def link(self, mu: Array) -> Array:
        xp = array_module(mu)
        mu_clipped = xp.clip(mu, _EPS, 1 - _EPS)
        return xp.log(mu_clipped / (1 - mu_clipped))

    def inverse(self, eta: Array) -> Array:
        xp = array_module(eta)
        return 1.0 / (1.0 + xp.exp(-eta))

    def derivative(self, mu: Array) -> Array:
        xp = array_module(mu)
        mu_clipped = xp.clip(mu, _EPS, 1 - _EPS)
        return 1.0 / (mu_clipped * (1 - mu_clipped))

    def mu_eta(self, eta: Array) -> Array:
        xp = array_module(eta)
        p = 1.0 / (1.0 + xp.exp(-eta))
        return p * (1.0 - p)


class InverseLink(Link):
    """Inverse (reciprocal) link: g(μ) = 1/μ."""

    def link(self, mu: Array) -> Array:
        xp = array_module(mu)
        return 1.0 / xp.maximum(mu, _EPS)

    def inverse(self, eta: Array) -> Array:
        return 1.0 / eta

    def derivative(self, mu: Array) -> Array:
        xp = array_module(mu)
        return -1.0 / xp.maximum(mu, _EPS) ** 2

    def mu_eta(self, eta: Array) -> Array:
        """dμ/dη = -1/η² (since μ = 1/η)."""
        return -1.0 / eta**2


class ProbitLink(Link):
    """Probit link: g(μ) = Φ⁻¹(μ)."""

    def link(self, mu: Array) -> Array:
        xp = array_module(mu)
        mu_clipped = xp.clip(mu, _EPS, 1 - _EPS)
        if is_jax_array(mu):
            from jax.scipy.special import ndtri as jndtri

            return jndtri(mu_clipped)
        return ndtri(mu_clipped)

    def inverse(self, eta: Array) -> Array:
        if is_jax_array(eta):
            from jax.scipy.special import ndtr as jndtr

            return jndtr(eta)
        return ndtr(eta)

    def derivative(self, mu: Array) -> Array:
        if is_jax_array(mu):
            from jax.scipy.stats import norm as jnorm

            return 1.0 / jnorm.pdf(self.link(mu))
        return 1.0 / norm.pdf(self.link(mu))

    def mu_eta(self, eta: Array) -> Array:
        if is_jax_array(eta):
            from jax.scipy.stats import norm as jnorm

            return jnorm.pdf(eta)
        return norm.pdf(eta)


class CloglogLink(Link):
    """Complementary log-log link: g(μ) = log(-log(1-μ))."""

    def link(self, mu: Array) -> Array:
        xp = array_module(mu)
        mu_clipped = xp.clip(mu, _EPS, 1 - _EPS)
        return xp.log(-xp.log(1 - mu_clipped))

    def inverse(self, eta: Array) -> Array:
        xp = array_module(eta)
        return 1.0 - xp.exp(-xp.exp(eta))

    def derivative(self, mu: Array) -> Array:
        xp = array_module(mu)
        mu_clipped = xp.clip(mu, _EPS, 1 - _EPS)
        return 1.0 / ((1 - mu_clipped) * (-xp.log(1 - mu_clipped)))

    def mu_eta(self, eta: Array) -> Array:
        xp = array_module(eta)
        return xp.exp(eta - xp.exp(eta))


class SqrtLink(Link):
    """Square root link: g(μ) = √μ."""

    def link(self, mu: Array) -> Array:
        xp = array_module(mu)
        return xp.sqrt(xp.maximum(mu, _EPS))

    def inverse(self, eta: Array) -> Array:
        return eta**2

    def derivative(self, mu: Array) -> Array:
        xp = array_module(mu)
        return 0.5 / xp.sqrt(xp.maximum(mu, _EPS))

    def mu_eta(self, eta: Array) -> Array:
        return 2.0 * eta


class InverseSquaredLink(Link):
    """Inverse squared link: g(μ) = 1/μ².

    Default link for Inverse Gaussian family.
    """

    def link(self, mu: Array) -> Array:
        xp = array_module(mu)
        return 1.0 / xp.maximum(mu, _EPS) ** 2

    def inverse(self, eta: Array) -> Array:
        xp = array_module(eta)
        return 1.0 / xp.sqrt(xp.maximum(eta, _EPS))

    def derivative(self, mu: Array) -> Array:
        xp = array_module(mu)
        return -2.0 / xp.maximum(mu, _EPS) ** 3

    def mu_eta(self, eta: Array) -> Array:
        """dμ/dη = -1/(2η^{3/2}) (since μ = 1/√η)."""
        xp = array_module(eta)
        return -0.5 / xp.maximum(eta, _EPS) ** 1.5


# Maps canonical link names to their implementing classes.
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
