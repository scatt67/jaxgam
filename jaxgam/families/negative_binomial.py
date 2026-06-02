"""Negative Binomial extended family: V(mu) = mu + mu^2/theta.

The NB distribution models count data with overdispersion. It generalizes
Poisson by adding a dispersion parameter theta (size/shape):

- Mean: mu
- Variance: mu + mu^2/theta
- As theta -> infinity, NB -> Poisson

Theta is stored internally as a log-scale array of shape ``(n_theta,)``
where ``n_theta = 1`` when theta is estimated, ``0`` when fixed.

Numerical stability: all expressions involving ``theta`` are rewritten
to avoid catastrophic cancellation at large theta (near-Poisson limit).
Key rewrites (going beyond R's mgcv ``nb()`` which uses raw arithmetic):

- ``log((y+θ)/(mu+θ))`` → ``log1p((y-mu)/(mu+θ))``
- ``(y+θ)*log(y+θ) - θ*log(θ)`` → ``y*log(y+θ) + θ*log1p(y/θ)``
- ``lgamma(θ) - lgamma(θ+y)`` → ``_lgamma_diff`` with recurrence-based
  custom_jvp for stable first and second derivatives via AD

These are exact algebraic identities (not approximations) and are
unconditionally at least as accurate as the original formulas.

Design doc reference: Section 7.2-7.4
R source reference: efam.r lines 161-310 (nb() extended family)
"""

from __future__ import annotations

import functools

import jax
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
    theta : float
        Dispersion parameter (must be positive). When ``fixed=False``
        (default), this is the starting value for estimation. When
        ``fixed=True``, theta is held constant during fitting.
        Default is ``1.0``.
    fixed : bool
        If ``False`` (default), theta is estimated during fitting
        (``n_theta = 1``). If ``True``, theta is held constant
        (``n_theta = 0``).
    link : str or Link or None
        Link function. Default is log. Supported: ``"log"``, ``"identity"``,
        ``"sqrt"``.

    Examples
    --------
    >>> fam = NegativeBinomial()                    # estimate theta, start at 1
    >>> fam = NegativeBinomial(theta=3)             # estimate theta, start at 3
    >>> fam = NegativeBinomial(theta=2, fixed=True) # fix theta = 2
    """

    family_name: str = "nb"
    scale_known: bool = True  # phi = 1 for NB
    response_support = NON_NEGATIVE

    def __init__(
        self,
        theta: float = 1.0,
        *,
        fixed: bool = False,
        link: str | Link | None = None,
    ) -> None:
        super().__init__(link)
        if theta <= 0:
            raise ValueError(
                f"theta must be positive, got {theta}. "
                "Pass the desired value directly (e.g. theta=3) and use "
                "fixed=True to hold it constant during fitting."
            )
        self._log_theta = np.array([np.log(theta)])
        self.n_theta: int = 0 if fixed else 1

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
        *,
        max_y: int = 0,
    ) -> float:
        """Saturated log-likelihood (R's family$ls).  Phase 2 only (JAX).

        R: efam.r lines 248-275 (forward pass only, no theta derivs).

        Rewritten for numerical stability at large theta:
        - ``(y+θ)*log(y+θ) - θ*log(θ)`` → ``y*log(y+θ) + θ*log1p(y/θ)``
        - ``lgamma(θ) - lgamma(θ+y)`` → ``_lgamma_diff`` with recurrence-based
          custom_jvp for stable first and second derivatives via AD
        """
        theta = jnp.exp(self._log_theta[0])
        return _saturated_loglik_jax(y, wt, theta, max_y)

    def deviance_resids(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Per-observation deviance residuals.

        R: efam.r lines 199-205.

        Unit deviance:
            2 * wt * [y * log(max(1,y)/mu) - (y+theta) * log1p((y-mu)/(mu+theta))]

        The ``log1p`` rewrite avoids catastrophic cancellation when
        theta is large and ``(y+theta)/(mu+theta) ≈ 1``.
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
                - (y + theta) * xp.log1p((y - mu_safe) / (mu_safe + theta))
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

        Rewritten for numerical stability at large theta.
        """
        theta = float(np.exp(self._log_theta[0]))
        mu_safe = np.maximum(mu, _MU_EPS)
        # ``y*log(mu+theta) + theta*log1p(mu/theta)`` already equals R's
        # ``(y+Theta)*log(mu+Theta) - Theta*log(Theta)`` (efam.r:239-246), so
        # the ``-theta*log(theta)`` term must NOT be subtracted a second time.
        term = (
            y * np.log(mu_safe + theta)
            + theta * np.log1p(mu_safe / theta)
            - y * np.log(mu_safe)
            + gammaln(y + 1.0)
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
        xp = array_module(eta)
        return xp.isfinite(eta)

    # ------------------------------------------------------------------
    # Pure-function factories (explicit theta for AD in custom_jvp)
    # ------------------------------------------------------------------

    def saturated_loglik_theta(
        self,
        y: np.ndarray,
        wt: np.ndarray,
        scale: float,  # noqa: ARG002
        log_theta: np.ndarray,
        *,
        max_y: int = 0,
    ):
        """Saturated log-likelihood with explicit theta for AD trace.

        ``log_theta`` has shape ``(n_theta,)`` = ``(1,)`` for NB.
        Called inside ``_diff_score`` where ``log_theta`` is a traced
        JAX array.

        Parameters
        ----------
        max_y : int
            Maximum count in ``y``. Controls the ``lax.scan`` loop bound
            in ``_lgamma_diff``. Must be a compile-time constant.
        """
        theta = jnp.exp(log_theta[0])
        return _saturated_loglik_jax(y, wt, theta, max_y)

    def deviance_fn(self, y: np.ndarray, wt: np.ndarray):
        """Return pure JAX function ``D(eta, log_theta_vec) -> scalar``.

        ``log_theta_vec`` has shape ``(n_theta,)`` = ``(1,)`` for NB.

        Used by the custom_jvp for IFT theta terms and joint JVPs,
        and by ``pirls_loop`` for penalized deviance when ``log_theta``
        is passed as a dynamic argument.

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
                    - (y + theta) * jnp.log1p((y - mu_safe) / (mu_safe + theta))
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

    def _static_cache_key(self) -> tuple:
        # When theta is estimated (n_theta == 1), it flows through PIRLS
        # as a dynamic JAX argument and does not affect the trace, so the
        # base cache key is sufficient. When theta is fixed (n_theta == 0),
        # ``variance``/``dev_resids`` read ``self._log_theta`` directly
        # inside the JIT trace, so theta becomes baked into the compiled
        # executable and must be part of the cache key.
        key = super()._static_cache_key()
        if self.n_theta == 0:
            key = (*key, float(self._log_theta[0]))
        return key


# ------------------------------------------------------------------
# Stable lgamma difference with custom_jvp for AD
# ------------------------------------------------------------------


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _lgamma_diff(theta, y, max_y):  # noqa: ARG001
    """``lgamma(theta) - lgamma(theta + y)`` with stable AD derivatives.

    Forward pass uses standard lgamma subtraction (accurate for the value).
    The JVP uses the digamma recurrence ``-sum_{k=0}^{y-1} 1/(theta+k)``
    which avoids the catastrophic cancellation in ``digamma(theta) -
    digamma(theta+y)`` when theta is large.

    Second derivatives (Hessian) get the trigamma recurrence
    ``sum 1/(theta+k)^2`` for free by differentiating through the JVP.

    Parameters
    ----------
    theta : jax.Array, scalar
        Dispersion parameter (positive).
    y : jax.Array, shape (n,)
        Non-negative integer counts (as float64).
    max_y : int
        Maximum value in y. Must be a compile-time static integer
        (controls the ``lax.scan`` length).
    """
    return jsp.gammaln(theta) - jsp.gammaln(theta + y)


@_lgamma_diff.defjvp
def _lgamma_diff_jvp(max_y, primals, tangents):
    theta, y = primals
    dtheta, dy = tangents

    primal_out = _lgamma_diff(theta, y, max_y)

    # Recurrence: d/d(theta)[lgamma(theta) - lgamma(theta+y)]
    #           = digamma(theta) - digamma(theta+y)
    #           = -sum_{k=0}^{y-1} 1/(theta+k)
    # This avoids subtracting two large digamma values at large theta.
    if max_y == 0:
        tangent_out = jnp.zeros_like(y) * dtheta
    else:

        def _scan_body(acc, k):
            mask = k < y
            acc = acc + jnp.where(mask, 1.0 / (theta + k), 0.0)
            return acc, None

        neg_psi_sum, _ = jax.lax.scan(
            _scan_body, jnp.zeros_like(y), jnp.arange(max_y, dtype=y.dtype)
        )
        # d(primal)/d(theta) = -neg_psi_sum, chain with dtheta.
        # The dy branch uses standard -digamma(theta+y).  This is fine
        # because y is integer count data and never differentiated in
        # practice — the recurrence is only needed for the theta
        # direction where digamma(theta) - digamma(theta+y) cancels
        # at large theta.
        tangent_out = -neg_psi_sum * dtheta + (-jsp.digamma(theta + y)) * dy

    return primal_out, tangent_out


def _saturated_loglik_jax(y, wt, theta, max_y):
    """Numerically stable saturated log-likelihood (JAX).

    Rewrites for stability at large theta (near-Poisson limit):
    - ``(y+θ)*log(y+θ) - θ*log(θ)`` → ``y*log(y+θ) + θ*log1p(y/θ)``
    - ``lgamma(θ) - lgamma(θ+y)`` → ``_lgamma_diff`` with recurrence-based
      JVP for stable first and second derivatives via AD.

    Parameters
    ----------
    max_y : int
        Maximum value in y (compile-time constant for ``lax.scan``).
    """
    ylogy = jnp.where(y > 0, y * jnp.log(y), 0.0)
    y_safe = jnp.where(y > 0, y, 1.0)
    lgamma_diff = jnp.where(
        y > 0,
        _lgamma_diff(theta, y_safe, max_y),
        0.0,
    )
    # Note: ``theta * log1p(y/theta)`` is optimized for the large-theta
    # regime (near-Poisson limit) where the original
    # ``(y+theta)*log(y+theta) - theta*log(theta)`` suffers cancellation.
    # For very small theta (high overdispersion), ``y/theta`` can be large
    # but ``log1p`` handles that correctly.
    term = (
        y * jnp.log(y_safe + theta)
        + theta * jnp.log1p(y / theta)
        - ylogy
        + jsp.gammaln(y + 1.0)
        + lgamma_diff
    )
    return -jnp.sum(term * wt)
