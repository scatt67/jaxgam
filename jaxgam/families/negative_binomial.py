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

from jaxgam.families.base import (
    NON_NEGATIVE,
    FamilyExecutionCapabilities,
    FamilyParameterSnapshot,
)
from jaxgam.families.extended import ExtendedFamily
from jaxgam.jax_utils import array_module
from jaxgam.links.links import Link, LogLink

_MU_EPS = 1e-10

# A table is one float64 value per possible count plus its zero prefix.  This
# deliberately bounds the extra workspace owned by the fast derivative path;
# larger tables use the mgcv-style polygamma difference where it is well
# conditioned.  The recurrence remains available to callers that need a
# different large-count policy.
# XLA's differentiated prefix kernel needs about 3.14x the raw table in the
# current CPU backend (26,284,424 bytes for a 1,048,576-entry table). Keep a
# 4x safety margin and budget the compiled derivative, not merely the table.
_PREFIX_WORKSPACE_BYTES = 8 << 20
_PREFIX_DIFFERENTIATED_WORKSPACE_MULTIPLIER = 4
_VECTOR_DIFF_MAX_THETA_TO_COUNT = 1e6
_FRACTIONAL_ASYMPTOTIC_THETA = 1e6
_FRACTIONAL_ASYMPTOTIC_THETA_TO_RESPONSE = 1e4


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

    def execution_capabilities(self) -> FamilyExecutionCapabilities:
        """Describe NB's explicit-theta, observed-information primitives."""
        return FamilyExecutionCapabilities(
            row_separable=True,
            fisher_working_system=True,
            observed_information=True,
            direct_deviance=True,
            differentiable_deviance=True,
            saturated_loglikelihood=True,
            dynamic_theta=self.n_theta > 0,
            dynamic_phi=False,
            coefficient_system="observed",
            fisher_equals_observed_for_score=False,
        )

    def execution_parameter_snapshot(self) -> FamilyParameterSnapshot:
        """Freeze theta mode/value so launches cannot observe later mutation."""
        return FamilyParameterSnapshot(
            theta_mode="estimated" if self.n_theta > 0 else "fixed",
            log_theta=tuple(float(value) for value in self._log_theta),
            phi_mode="known",
        )

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
        count_indices=None,
        integer_counts: bool | None = None,
    ) -> float:
        """Saturated log-likelihood (R's family$ls).  Phase 2 only (JAX).

        R: efam.r lines 248-275 (forward pass only, no theta derivs).

        Rewritten for numerical stability at large theta:
        - ``(y+θ)*log(y+θ) - θ*log(θ)`` → ``y*log(y+θ) + θ*log1p(y/θ)``
        - ``lgamma(θ) - lgamma(θ+y)`` → ``_lgamma_diff`` with recurrence-based
          custom_jvp for stable first and second derivatives via AD
        """
        theta = jnp.exp(self._log_theta[0])
        return _saturated_loglik_jax(y, wt, theta, max_y, count_indices, integer_counts)

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

    def deviance_contributions(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Direct NB deviance at the stored theta for non-JIT callers."""
        return self.deviance_contributions_for_parameters(
            y, mu, wt, jnp.asarray(self._log_theta)
        )

    def deviance_derivative_contributions(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Smooth NB deviance at stored theta for observed-information AD."""
        return self.deviance_derivative_contributions_for_parameters(
            y, mu, wt, jnp.asarray(self._log_theta)
        )

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
        count_indices=None,
        integer_counts: bool | None = None,
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
        return _saturated_loglik_jax(y, wt, theta, max_y, count_indices, integer_counts)

    def saturated_loglikelihood_for_parameters(
        self,
        y: np.ndarray,
        wt: np.ndarray,
        scale: float,
        log_theta: np.ndarray,
        *,
        max_y: int = 0,
    ):
        """Evaluate saturated likelihood from explicit, immutable theta data."""
        return self.saturated_loglik_theta(y, wt, scale, log_theta, max_y=max_y)

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

    def working_weights_for_parameters(
        self,
        mu: np.ndarray,  # noqa: ARG002 - eta is the stable factory input
        eta: np.ndarray,
        wt: np.ndarray,
        log_theta: np.ndarray,
    ) -> np.ndarray:
        """Use explicit theta; never read mutable ``_log_theta`` in a kernel."""
        return self.working_weights_fn(wt)(eta, log_theta)

    def deviance_contributions_for_parameters(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        log_theta: np.ndarray,
    ) -> np.ndarray:
        """Direct NB deviance at explicit theta without residual square roots."""
        xp = array_module(y)
        theta = xp.exp(log_theta[0])
        mu_safe = xp.maximum(mu, _MU_EPS)
        y_safe = xp.where(y > 0, y, 1.0)
        contribution = (
            2.0
            * wt
            * (
                y * xp.log(y_safe / mu_safe)
                - (y + theta) * xp.log1p((y - mu_safe) / (mu_safe + theta))
            )
        )
        return xp.maximum(contribution, 0.0)

    def deviance_derivative_contributions_for_parameters(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        log_theta: np.ndarray,
    ) -> np.ndarray:
        """Interior-domain NB deviance with no reporting clamp kink."""
        xp = array_module(y)
        theta = xp.exp(log_theta[0])
        mu_safe = xp.maximum(mu, _MU_EPS)
        y_safe = xp.where(y > 0, y, 1.0)
        return (
            2.0
            * wt
            * (
                y * xp.log(y_safe / mu_safe)
                - (y + theta) * xp.log1p((y - mu_safe) / (mu_safe + theta))
            )
        )

    def execution_summary_from_batch(
        self, y: np.ndarray, wt: np.ndarray, valid: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Append bounded count-prefix planning metadata to the base summary."""
        xp = array_module(y)
        base = super().execution_summary_from_batch(y, wt, valid)
        real = valid & xp.isfinite(y)
        safe_y = xp.where(real, y, 0.0)
        return (*base, xp.max(safe_y), xp.all(~valid | (y == xp.floor(y))))

    def merge_execution_summaries(
        self, left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        """Merge additive base leaves plus max/all NB count metadata."""
        if len(left) != 6 or len(right) != 6:
            raise ValueError("NB execution summaries require six leaves.")
        xp = array_module(left[4])
        return (
            left[0] + right[0],
            left[1] + right[1],
            left[2] + right[2],
            xp.logical_and(left[3], right[3]),
            xp.maximum(left[4], right[4]),
            xp.logical_and(left[5], right[5]),
        )

    def finalize_execution_summary(
        self, summary: tuple[float, ...]
    ) -> dict[str, float]:
        """Expose count-prefix metadata without retaining observations."""
        if len(summary) != 6:
            raise ValueError("NB execution summaries require six leaves.")
        result = super().finalize_execution_summary(summary[:4])
        result.update(
            max_count=float(summary[4]),
            integer_counts=bool(summary[5]),
        )
        return result

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


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4))
def _lgamma_diff_planned(theta, y, count_indices, capacity, integer_counts):  # noqa: ARG001
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
    count_indices : jax.Array
        Precomputed non-negative integer count indices.  They are metadata,
        not a differentiable representation of ``y``.
    capacity : int
        Static table capacity.  ``capacity >= max(count_indices)``.
    integer_counts : bool
        Static fast-path guard. Fractional responses retain the gamma
        derivative semantics instead of being rounded for a recurrence.
    """
    return jsp.gammaln(theta) - jsp.gammaln(theta + y)


@_lgamma_diff_planned.defjvp
def _lgamma_diff_jvp(capacity, integer_counts, primals, tangents):
    theta, y, count_indices = primals
    dtheta, dy, _ = tangents

    primal_out = _lgamma_diff_planned(theta, y, count_indices, capacity, integer_counts)

    # Recurrence: d/d(theta)[lgamma(theta) - lgamma(theta+y)]
    #           = digamma(theta) - digamma(theta+y)
    #           = -sum_{k=0}^{y-1} 1/(theta+k)
    # This avoids subtracting two large digamma values at large theta.
    indices = jnp.asarray(count_indices, dtype=jnp.int64)
    valid_indices = (indices >= 0) & (indices <= capacity)
    if not integer_counts:
        # This is mgcv nb()$ls' digamma derivative. It is also the exact
        # meaning for fractional NB responses, which are accepted today.
        raw_difference = jsp.digamma(theta) - jsp.digamma(theta + y)
        # Avoid subtracting nearly equal digammas for fractional responses,
        # where the integer recurrence is inapplicable. The first three terms
        # of psi(theta)-psi(theta+y) are enough once theta dwarfs y; AD of
        # this expression supplies the matching second theta derivative.
        asymptotic = (
            -y / theta
            + y * (y - 1.0) / (2.0 * theta**2)
            - y * (2.0 * y**2 - 3.0 * y + 1.0) / (6.0 * theta**3)
        )
        dtheta_value = jnp.where(
            theta
            >= jnp.maximum(
                _FRACTIONAL_ASYMPTOTIC_THETA * (1.0 - 1e-12),
                _FRACTIONAL_ASYMPTOTIC_THETA_TO_RESPONSE * jnp.abs(y),
            ),
            asymptotic,
            raw_difference,
        )
    elif capacity <= 0:
        dtheta_value = jnp.zeros_like(y)
    elif (capacity + 1) * np.dtype(
        np.float64
    ).itemsize * _PREFIX_DIFFERENTIATED_WORKSPACE_MULTIPLIER <= _PREFIX_WORKSPACE_BYTES:
        reciprocal = 1.0 / (theta + jnp.arange(capacity, dtype=y.dtype))
        prefix = jnp.concatenate((jnp.zeros(1, dtype=y.dtype), jnp.cumsum(reciprocal)))
        # FittingData validates this metadata once. Direct callers receive a
        # non-finite derivative for an undersized plan instead of truncation.
        # ``mode='fill'`` makes malformed direct-helper metadata visible. JAX
        # otherwise clamps out-of-range gathers, which would silently turn an
        # undersized plan into the wrong derivative.
        dtheta_value = jnp.where(
            valid_indices,
            -jnp.take(prefix, indices, mode="fill", fill_value=jnp.nan),
            jnp.nan,
        )
    else:
        # The table would exceed the budget. This branch is compiled without
        # a count-length table. The vectorized formula is used only before
        # subtraction loses material precision; otherwise retain the stable
        # recurrence (O(n * capacity), but bounded auxiliary workspace).
        def _stable_recurrence(_: None):
            def body(k, acc):
                return acc + jnp.where(
                    k < indices,
                    1.0 / (theta + k),
                    0.0,
                )

            return -jax.lax.fori_loop(0, capacity, body, jnp.zeros_like(y))

        def _vectorized(_: None):
            return jsp.digamma(theta) - jsp.digamma(theta + y)

        if indices.shape[0] == 0:
            smallest_count = jnp.array(1.0, dtype=y.dtype)
        else:
            positive_counts = jnp.where(
                indices > 0,
                jnp.asarray(indices, dtype=y.dtype),
                jnp.inf,
            )
            smallest_count = jnp.minimum(jnp.min(positive_counts), 1.0)
        dtheta_value = jax.lax.cond(
            theta <= _VECTOR_DIFF_MAX_THETA_TO_COUNT * smallest_count,
            _vectorized,
            _stable_recurrence,
            operand=None,
        )

    # Metadata must be valid even on the recurrence/vectorized/empty paths;
    # never turn an undersized plan into a finite but truncated derivative.
    dtheta_value = jnp.where(valid_indices, dtheta_value, jnp.nan)
    tangent_out = dtheta_value * dtheta - jsp.digamma(theta + y) * dy

    return primal_out, tangent_out


def _lgamma_diff(theta, y, max_y):
    """Compatibility wrapper for the historical internal three-argument API.

    Fitting passes precomputed indices to ``_lgamma_diff_planned``. Direct
    callers retain the old signature and receive the same integer fast path.
    """
    capacity = max(1, max_y)

    def integer_path(_: None):
        return _lgamma_diff_planned(
            theta, y, jnp.asarray(y, dtype=jnp.int64), capacity, True
        )

    def fractional_path(_: None):
        return _lgamma_diff_planned(
            theta, y, jnp.zeros(y.shape, dtype=jnp.int64), 0, False
        )

    return jax.lax.cond(jnp.all(y == jnp.floor(y)), integer_path, fractional_path, None)


def _saturated_loglik_jax(y, wt, theta, max_y, count_indices=None, integer_counts=None):
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
    if count_indices is None or integer_counts is None:
        lgamma_value = _lgamma_diff(theta, y_safe, max_y)
    else:
        capacity = max(1, max_y) if integer_counts else 0
        lgamma_value = _lgamma_diff_planned(
            theta, y_safe, count_indices, capacity, integer_counts
        )
    lgamma_diff = jnp.where(y > 0, lgamma_value, 0.0)
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
