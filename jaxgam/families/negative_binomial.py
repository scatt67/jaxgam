"""Negative Binomial extended family: V(mu) = mu + mu^2/theta.

The NB distribution models count data with overdispersion. It generalizes
Poisson by adding a dispersion parameter theta (size/shape):

- Mean: mu
- Variance: mu + mu^2/theta
- As theta -> infinity, NB -> Poisson

Theta is stored internally as a log-scale array of shape ``(n_theta,)``
where ``n_theta = 1`` when theta is estimated, ``0`` when fixed.

Design doc reference: Section 7.2-7.4
R source reference: efam.r lines 161-310 (nb() extended family)
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from scipy.special import gammaln

from jaxgam.families.base import NON_NEGATIVE
from jaxgam.families.extended import ExtendedFamily
from jaxgam.jax_utils import array_module
from jaxgam.links.links import Link, LogLink

_MU_EPS = 1e-10


class NegativeBinomial(ExtendedFamily):
    """Negative Binomial family with V(mu) = mu + mu^2/theta.

    Parameters
    ----------
    theta : float or None
        Dispersion parameter. Interpretation follows R's ``nb()`` convention:

        - ``None`` or ``0``: estimate theta (``n_theta = 1``), start at
          ``theta = 1`` (i.e. ``log_theta = 0``).
        - Positive: fix theta at this value (``n_theta = 0``).
        - Negative: estimate theta (``n_theta = 1``), start at ``-theta``.

    link : str or Link or None
        Link function. Default is log. Supported: ``"log"``, ``"identity"``,
        ``"sqrt"``.

    Examples
    --------
    >>> fam = NegativeBinomial()          # estimate theta, start at 1
    >>> fam = NegativeBinomial(theta=2)   # fix theta = 2
    >>> fam = NegativeBinomial(theta=-3)  # estimate theta, start at 3
    """

    family_name: str = "nb"  # type: ignore[assignment]
    scale_known: bool = True  # phi = 1 for NB
    response_support = NON_NEGATIVE

    def __init__(
        self,
        theta: float | None = None,
        link: str | Link | None = None,
    ) -> None:
        super().__init__(link)
        # R's nb() convention: NULL/0 -> estimate from 1,
        # >0 -> fixed, <0 -> estimate from -theta
        if theta is None or theta == 0:
            self._log_theta = np.array([0.0])  # log(1) = 0
            self.n_theta: int = 1
        elif theta > 0:
            self._log_theta = np.array([np.log(theta)])
            self.n_theta = 0
        else:
            self._log_theta = np.array([np.log(-theta)])
            self.n_theta = 1

    @property
    def default_link(self) -> Link:
        return LogLink()

    @property
    def alpha(self) -> float:
        """Overdispersion parameter alpha = 1/theta.

        V(mu) = mu + alpha * mu^2. Convenience for users who prefer
        the econometrics parameterization.
        """
        return 1.0 / float(np.exp(self._log_theta[0]))

    # ------------------------------------------------------------------
    # ExtendedFamily interface: theta management
    # ------------------------------------------------------------------

    def get_theta(self, transformed: bool = False) -> np.ndarray:
        """Extra parameter vector, shape ``(n_theta,)`` = ``(1,)`` for NB.

        Parameters
        ----------
        transformed : bool
            If True, return ``exp(log_theta)`` (natural scale).
        """
        if transformed:
            return np.exp(self._log_theta)
        return self._log_theta.copy()

    def put_theta(self, log_theta: np.ndarray) -> None:
        """Set log(theta) vector. Called by Newton after each accepted step."""
        self._log_theta = np.asarray(log_theta, dtype=np.float64).reshape(
            self._log_theta.shape
        )

    # ------------------------------------------------------------------
    # Standard family methods (read theta from self._log_theta)
    # ------------------------------------------------------------------

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(mu) = mu + mu^2/theta."""
        xp = array_module(mu)
        theta = xp.exp(self._log_theta[0])
        return mu + mu**2 / theta

    def dvar(self, mu: np.ndarray) -> np.ndarray:
        """V'(mu) = 1 + 2*mu/theta.  Phase 2 only (JAX)."""
        theta = jnp.exp(self._log_theta[0])
        return 1.0 + 2.0 * mu / theta

    def saturated_loglik(
        self,
        y: np.ndarray,
        wt: np.ndarray,
        scale: float,  # noqa: ARG002
    ) -> float:
        """Saturated log-likelihood (R's family$ls).  Phase 2 only (JAX).

        R: efam.r lines 248-275 (forward pass only, no theta derivs).
        """
        theta = jnp.exp(self._log_theta[0])
        ylogy = jnp.where(y > 0, y * jnp.log(y), 0.0)
        term = (
            (y + theta) * jnp.log(y + theta)
            - ylogy
            + jsp.gammaln(y + 1.0)
            - theta * jnp.log(theta)
            + jsp.gammaln(theta)
            - jsp.gammaln(theta + y)
        )
        return -jnp.sum(term * wt)

    def deviance_resids(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Per-observation deviance residuals.

        R: efam.r lines 199-205.

        Unit deviance:
            2 * wt * [y * log(max(1,y)/mu) - (y+theta) * log((y+theta)/(mu+theta))]
        """
        xp = array_module(y)
        theta = xp.exp(self._log_theta[0])
        mu_safe = xp.maximum(mu, _MU_EPS)
        y_safe = xp.where(y > 0, y, 1.0)  # max(1, y) for the log
        d = (
            2.0
            * wt
            * (
                y * xp.log(y_safe / mu_safe)
                - (y + theta) * xp.log((y + theta) / (mu_safe + theta))
            )
        )
        d = xp.maximum(d, 0.0)
        return xp.sign(y - mu_safe) * xp.sqrt(d)

    def aic(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        scale: float,  # noqa: ARG002
    ) -> float:
        """AIC contribution.  Phase 3 only (NumPy).

        R: efam.r lines 239-246.
        """
        theta = float(np.exp(self._log_theta[0]))
        mu_safe = np.maximum(mu, _MU_EPS)
        term = (
            (y + theta) * np.log(mu_safe + theta)
            - y * np.log(mu_safe)
            + gammaln(y + 1.0)
            - theta * np.log(theta)
            + gammaln(theta)
            - gammaln(theta + y)
        )
        return float(2.0 * np.sum(term * wt))

    def _initialize_impl(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Initialize mu for NB: mustart = y + (y == 0) / 6.

        R: efam.r line 280. Domain validation handled by base class
        via ``response_support = NON_NEGATIVE``.
        """
        return np.where(y == 0, y + 1.0 / 6.0, y)

    def valid_mu(self, mu: np.ndarray) -> np.ndarray:
        """Valid mu for NB: mu > 0."""
        return mu > 0

    def valid_eta(self, eta: np.ndarray) -> np.ndarray:
        """All finite eta are valid for NB."""
        return np.isfinite(eta)

    # ------------------------------------------------------------------
    # Pure-function factories (explicit theta for AD in custom_jvp)
    # ------------------------------------------------------------------

    def saturated_loglik_theta(
        self,
        y: np.ndarray,
        wt: np.ndarray,
        scale: float,  # noqa: ARG002
        log_theta: np.ndarray,
    ):
        """Saturated log-likelihood with explicit theta for AD trace.

        ``log_theta`` has shape ``(n_theta,)`` = ``(1,)`` for NB.
        Called inside ``_diff_score`` where ``log_theta`` is a traced
        JAX array.
        """
        theta = jnp.exp(log_theta[0])
        ylogy = jnp.where(y > 0, y * jnp.log(y), 0.0)
        term = (
            (y + theta) * jnp.log(y + theta)
            - ylogy
            + jsp.gammaln(y + 1.0)
            - theta * jnp.log(theta)
            + jsp.gammaln(theta)
            - jsp.gammaln(theta + y)
        )
        return -jnp.sum(term * wt)

    def deviance_fn(self, y: np.ndarray, wt: np.ndarray):
        """Return pure JAX function ``D(eta, log_theta_vec) -> scalar``.

        ``log_theta_vec`` has shape ``(n_theta,)`` = ``(1,)`` for NB.

        Used by the custom_jvp for IFT theta terms and joint JVPs.
        Captures ``(y, wt, link)`` in closure; theta is an explicit arg.
        """
        link_inv = self.link.inverse

        def _dev(eta, log_theta):
            theta = jnp.exp(log_theta[0])
            mu = link_inv(eta)
            mu_safe = jnp.maximum(mu, _MU_EPS)
            y_safe = jnp.where(y > 0, y, 1.0)
            return jnp.sum(
                2.0
                * wt
                * (
                    y * jnp.log(y_safe / mu_safe)
                    - (y + theta) * jnp.log((y + theta) / (mu_safe + theta))
                )
            )

        return _dev

    def working_weights_fn(self, wt: np.ndarray):
        """Return pure JAX function ``W(eta, log_theta_vec) -> (n,) array``.

        ``log_theta_vec`` has shape ``(n_theta,)`` = ``(1,)`` for NB.

        Used by the custom_jvp for joint dW JVPs. Captures
        ``(wt, link)`` in closure; theta is an explicit arg.
        """
        link_inv = self.link.inverse
        link_deriv = self.link.derivative

        def _ww(eta, log_theta):
            theta = jnp.exp(log_theta[0])
            mu = link_inv(eta)
            V = mu + mu**2 / theta
            g_prime = link_deriv(mu)
            return wt / (V * g_prime**2)

        return _ww

    def __repr__(self) -> str:
        theta_val = float(np.exp(self._log_theta[0]))
        fixed = "fixed" if self.n_theta == 0 else "estimated"
        return (
            f"NegativeBinomial(theta={theta_val:.4g}, {fixed}, "
            f"link={type(self.link).__name__})"
        )
