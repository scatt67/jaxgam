"""ExponentialFamily base class for GLM/GAM distributions.

Provides the interface contract that all standard exponential family
distributions must implement. The PIRLS algorithm relies on these
methods for working weights, working response, and deviance computation.

Design doc reference: Section 6.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from jaxgam.jax_utils import array_module
from jaxgam.links.links import Link

# ---------------------------------------------------------------------------
# Response domain constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseSupport:
    """Domain constraint for the response variable.

    Declares the valid range for response values. The base class
    ``ExponentialFamily.initialize()`` validates ``y`` against this
    before dispatching to the family-specific ``_initialize_impl()``.

    Inspired by ``torch.distributions.constraints`` but minimal:
    just bounds checking, no transform registry.

    Parameters
    ----------
    lower : float
        Lower bound (default ``-inf``).
    upper : float
        Upper bound (default ``inf``).
    lower_inclusive : bool
        Whether the lower bound is inclusive (default True).
    upper_inclusive : bool
        Whether the upper bound is inclusive (default True).
    """

    lower: float = -np.inf
    upper: float = np.inf
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def check(self, y: np.ndarray) -> bool:
        """Return True if all values of ``y`` are in the support."""
        if self.lower_inclusive:
            low_ok = np.all(y >= self.lower)
        else:
            low_ok = np.all(y > self.lower)
        if self.upper_inclusive:
            high_ok = np.all(y <= self.upper)
        else:
            high_ok = np.all(y < self.upper)
        return bool(low_ok and high_ok)

    def __str__(self) -> str:
        lb = "[" if self.lower_inclusive else "("
        ub = "]" if self.upper_inclusive else ")"
        return f"{lb}{self.lower}, {self.upper}{ub}"


# Pre-built singletons matching common distribution supports.
REAL = ResponseSupport()
NON_NEGATIVE = ResponseSupport(lower=0)
POSITIVE = ResponseSupport(lower=0, lower_inclusive=False)
UNIT_INTERVAL = ResponseSupport(lower=0, upper=1)


# ---------------------------------------------------------------------------
# Execution-contract descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyExecutionCapabilities:
    """Static mathematical capabilities exposed to fitting backends.

    This is deliberately a family-owned descriptor rather than a registry of
    family names.  Backends may choose a *release policy* on top of these
    capabilities, but construction or registration alone is never evidence
    that a particular execution route has been validated.
    """

    row_separable: bool
    fisher_working_system: bool
    observed_information: bool
    direct_deviance: bool
    differentiable_deviance: bool
    saturated_loglikelihood: bool
    dynamic_theta: bool
    dynamic_phi: bool
    coefficient_system: Literal["fisher", "observed"] = "fisher"
    fisher_equals_observed_for_score: bool = False
    regular_fletcher_scale: bool = False


@dataclass(frozen=True)
class StreamReductionPolicy:
    """Family-owned policy for the bounded streamed fixed-sp reductions.

    A family being constructible, or merely having an unknown scale, is not
    enough to select a reported-scale or REML-score formula.  These names are
    intentionally narrow: a stream controller must reject ``unsupported``
    instead of borrowing Gaussian's formula for Gamma or a future family.
    """

    reported_scale: Literal[
        "known_one",
        "gaussian_fisher_edf_deviance",
        "regular_fletcher",
        "unsupported",
    ]
    score_scale: Literal["reported_scale", "gaussian_fixed_sp", "unsupported"]


@dataclass(frozen=True)
class FamilyPadding:
    """Finite values safe to evaluate for an invalid padded row.

    ``eta`` is separate from ``offset`` because an inverse link can be
    undefined at the usual zero-filled linear predictor.  Kernels must select
    this eta before calling ``link.inverse``.
    """

    response: float
    eta: float


@dataclass(frozen=True)
class FamilyParameterSnapshot:
    """Host snapshot of dynamic family parameters and their modes.

    The values are immutable Python scalars so a caller can validate that a
    mutable family was not changed between preparation and a device launch.
    JIT kernels receive a separate array pytree; they never read mutable
    family parameter storage.
    """

    theta_mode: Literal["none", "fixed", "estimated"]
    log_theta: tuple[float, ...]
    phi_mode: Literal["known", "estimated"]


def _freeze_execution_value(value: object) -> object:
    """Return a deterministic immutable snapshot for lineage comparison.

    This intentionally favours a clear host-side failure for unusual mutable
    configuration over claiming that object identity is a stable numerical
    contract.  Array payloads are small family/link configuration, never
    training data.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes, type)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_execution_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_execution_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if hasattr(value, "__dict__"):
        return (
            type(value),
            _freeze_execution_value(vars(value)),
        )
    raise TypeError(
        "Family/link execution configuration must be immutable or provide "
        "a serializable __dict__; got "
        f"{type(value)!r}."
    )


# ---------------------------------------------------------------------------
# ExponentialFamily base class
# ---------------------------------------------------------------------------


class ExponentialFamily(ABC):
    """Base class for exponential family distributions.

    Standard families provide:
    - variance(mu): V(mu) -- the variance function
    - deviance_resids(y, mu, wt): per-observation deviance residuals
    - dev_resids(y, mu, wt): scalar total deviance
    - aic(y, mu, wt, scale): AIC contribution
    - initialize(y, wt): starting mu values (validates response domain)
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
    response_support: ResponseSupport = REAL

    #: The link class for which this family's canonical link equals the
    #: identity Fisher==Newton. When the fitted link is canonical, Fisher
    #: (expected) and Newton (observed) information coincide; for non-canonical
    #: links mgcv uses observed information in the REML log|H| (gam.fit3.r:118).
    #: Subclasses set this to their canonical link class (None = unknown).
    canonical_link_cls: type[Link] | None = None

    @property
    def is_canonical(self) -> bool:
        """True if the fitted link is this family's canonical link.

        Static (depends only on the family/link objects, not array values), so
        it is safe to branch on inside JIT-traced code.
        """
        cls = type(self).canonical_link_cls
        return cls is not None and isinstance(self.link, cls)

    def execution_capabilities(self) -> FamilyExecutionCapabilities:
        """Return the static capabilities needed by bounded fit backends.

        Standard exponential families use a positive Fisher working system
        and expose observed information through their direct deviance.  A
        family that needs a different coefficient system or parameter state
        overrides this descriptor on its existing class; drivers do not
        branch on ``family_name``.
        """
        return FamilyExecutionCapabilities(
            row_separable=True,
            fisher_working_system=True,
            observed_information=True,
            direct_deviance=type(self).deviance_contributions
            is not ExponentialFamily.deviance_contributions,
            differentiable_deviance=type(self).deviance_derivative_contributions
            is not ExponentialFamily.deviance_derivative_contributions,
            saturated_loglikelihood=True,
            dynamic_theta=False,
            dynamic_phi=not self.scale_known,
            fisher_equals_observed_for_score=self.is_canonical,
        )

    def stream_reduction_policy(self) -> StreamReductionPolicy:
        """Return this family's explicitly supported streamed scale policy.

        Known-scale families use the mathematical constant one.  Unknown
        scale is deliberately unsupported by the base class: Fletcher and
        observed-information families require family-specific policy, not an
        inference from ``scale_known``.
        """
        if self.scale_known:
            return StreamReductionPolicy("known_one", "reported_scale")
        return StreamReductionPolicy("unsupported", "unsupported")

    def execution_padding(self) -> FamilyPadding:
        """Return a finite response/eta pair for masked padded rows.

        The mean used to obtain the eta is deliberately interior to the
        response support.  This makes the default safe for the built-in
        inverse and inverse-squared links as well as log links.  Custom
        families with a narrower link domain can override it explicitly.
        """
        if self.response_support == UNIT_INTERVAL:
            response, mean = 0.5, 0.5
        elif self.response_support == POSITIVE:
            response, mean = 1.0, 1.0
        elif self.response_support == NON_NEGATIVE:
            response, mean = 0.0, 1.0
        else:
            response, mean = 0.0, 1.0
        eta = float(np.asarray(self.link.link(np.asarray(mean))).reshape(()))
        mu = np.asarray(self.link.inverse(np.asarray(eta)))
        if (
            not np.isfinite(eta)
            or not bool(np.all(self.valid_mu(mu)))
            or not bool(np.all(self.valid_eta(np.asarray(eta))))
        ):
            raise ValueError(
                f"{type(self).__name__} must override execution_padding() with "
                "finite valid response and eta values."
            )
        return FamilyPadding(response=response, eta=eta)

    def execution_parameter_snapshot(self) -> FamilyParameterSnapshot:
        """Return the immutable host-side parameter snapshot for a launch."""
        return FamilyParameterSnapshot(
            theta_mode="none",
            log_theta=(),
            phi_mode="known" if self.scale_known else "estimated",
        )

    def execution_static_config(self) -> tuple[object, ...]:
        """Freeze family/link configuration that affects mathematical traces.

        Unlike ``_static_cache_key()``, this deliberately snapshots public and
        private instance attributes so a mutable custom link/family cannot be
        silently reused after preparation.  Dynamic distribution parameters
        belong to ``execution_parameter_snapshot()`` instead.
        """
        excluded = self.execution_dynamic_config_attributes()
        attributes = tuple(
            (name, _freeze_execution_value(value))
            for name, value in sorted(vars(self).items())
            if name not in excluded
        )
        return (
            type(self),
            self.family_name,
            self._static_cache_key(),
            type(self.link),
            _freeze_execution_value(vars(self.link)),
            attributes,
            self.execution_capabilities(),
            self.execution_padding(),
            self.stream_reduction_policy(),
        )

    def execution_dynamic_config_attributes(self) -> frozenset[str]:
        """Names owned by explicit dynamic parameter state, not static config.

        Extended families override this instead of relying on a base-class
        spelling such as ``_log_theta``.  This keeps the contract usable for a
        future family with different mutable parameter storage.
        """
        return frozenset()

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

    #: Short name for the family (e.g. 'gaussian', 'binomial').
    #: Subclasses must set as a class variable.
    family_name: str

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
    def saturated_loglik(
        self, y: np.ndarray, wt: np.ndarray, scale: float, *, max_y: int = 0
    ) -> float:
        """Saturated log-likelihood: log L(y; y, scale, wt).

        The log-likelihood of the saturated model (mu = y). Used in the
        REML criterion (R's ``family$ls``).

        Backend-agnostic: accepts both NumPy and JAX arrays.

        Parameters
        ----------
        y : np.ndarray
            Response values.
        wt : np.ndarray
            Prior weights.
        scale : float
            Dispersion parameter.
        max_y : int
            Maximum count in ``y``. Only used by extended families
            (e.g. NB) where ``lax.scan`` needs a compile-time loop bound.
            Standard families ignore this parameter.

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

    def deviance_contributions(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Return direct per-row deviance contributions without ``sqrt``.

        Streamed reductions and derivative providers must use this primitive
        rather than square ``deviance_resids``: differentiating through a
        square root at an exact fit creates an avoidable AD singularity.
        Built-in families override it with their existing unit-deviance
        arithmetic.  A custom family that omits it is constructible for dense
        compatibility code but explicitly lacks the direct-deviance execution
        capability.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement direct deviance "
            "contributions for bounded execution."
        )

    def deviance_derivative_contributions(
        self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
    ) -> np.ndarray:
        """Return the smooth direct-deviance primitive used for AD.

        Reported deviance may clamp tiny negative roundoff to zero.  That
        clamp is nondifferentiable at an exact fit and must never sit on an
        observed-information AD path.  Families therefore provide this
        separate, algebraically equivalent interior-domain expression.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement differentiable "
            "deviance contributions for bounded execution."
        )

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

    def initialize(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:
        """Validate response domain, then compute starting mu values.

        Called before the first PIRLS iteration. Checks that ``y`` is
        in the family's ``response_support`` before dispatching to the
        family-specific ``_initialize_impl()``.

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

        Raises
        ------
        ValueError
            If any ``y`` values are outside the family's response support.
        """
        y_arr = np.asarray(y, dtype=float)
        if not self.response_support.check(y_arr):
            raise ValueError(
                f"{self.family_name} family requires response values in "
                f"{self.response_support}, got range "
                f"[{np.min(y_arr):.4g}, {np.max(y_arr):.4g}]"
            )
        return self._initialize_impl(y_arr, wt)

    @abstractmethod
    def _initialize_impl(self, y: np.ndarray, wt: np.ndarray) -> np.ndarray:
        """Family-specific initialization (called after validation).

        Parameters
        ----------
        y : np.ndarray
            Response values (already validated and cast to float64).
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

    def working_weights_for_parameters(
        self,
        mu: np.ndarray,
        eta: np.ndarray,  # noqa: ARG002 - standard families need mu only
        wt: np.ndarray,
        log_theta: np.ndarray,
    ) -> np.ndarray:
        """Pure working-weight hook with explicit dynamic theta input.

        Standard families reject nonempty theta rather than reading mutable
        state inside a JIT kernel.  Extended families override this method and
        use the supplied array directly.
        """
        if log_theta.shape[0] != 0:
            raise ValueError(
                f"{type(self).__name__} must override "
                "working_weights_for_parameters() for theta-dependent fits."
            )
        return self.working_weights(mu, wt)

    def deviance_contributions_for_parameters(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        log_theta: np.ndarray,
    ) -> np.ndarray:
        """Pure direct-deviance hook with explicit dynamic theta input."""
        if log_theta.shape[0] != 0:
            raise ValueError(
                f"{type(self).__name__} must override "
                "deviance_contributions_for_parameters() for theta-dependent fits."
            )
        return self.deviance_contributions(y, mu, wt)

    def saturated_loglikelihood_for_parameters(
        self,
        y: np.ndarray,
        wt: np.ndarray,
        scale: float,
        log_theta: np.ndarray,
        *,
        max_y: int = 0,
    ) -> float:
        """Pure saturated-likelihood hook with explicit dynamic theta input."""
        if log_theta.shape[0] != 0:
            raise ValueError(
                f"{type(self).__name__} must override "
                "saturated_loglikelihood_for_parameters() for theta-dependent "
                "fits."
            )
        return self.saturated_loglik(y, wt, scale, max_y=max_y)

    def deviance_derivative_contributions_for_parameters(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        log_theta: np.ndarray,
    ) -> np.ndarray:
        """Explicit-parameter smooth-deviance hook used only by AD paths."""
        if log_theta.shape[0] != 0:
            raise ValueError(
                f"{type(self).__name__} must override "
                "deviance_derivative_contributions_for_parameters() for "
                "theta-dependent fits."
            )
        return self.deviance_derivative_contributions(y, mu, wt)

    def execution_summary_from_batch(
        self, y: np.ndarray, wt: np.ndarray, valid: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Return bounded mergeable summary leaves for one batch.

        The tuple is intentionally family-owned and extensible.  The base
        fields are valid-row count, positive-weight count, and the sum of
        positive-weight logs.  Families with additional global metadata (for
        example NB count-prefix planning) append leaves and override merge and
        finalization together.
        """
        xp = array_module(y)
        finite_input = xp.isfinite(y) & xp.isfinite(wt)
        weight_ok = wt >= 0.0
        real = valid
        safe_weight = xp.where(finite_input, wt, 0.0)
        positive = real & (safe_weight > 0)
        log_weight = xp.where(positive, xp.maximum(safe_weight, 1e-300), 1.0)
        return (
            xp.sum(real),
            xp.sum(positive),
            xp.sum(xp.log(log_weight)),
            xp.all(~valid | (finite_input & weight_ok)),
        )

    def merge_execution_summaries(
        self, left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        """Merge two summaries produced by ``execution_summary_from_batch``."""
        if len(left) != len(right):
            raise ValueError("Family execution summaries have incompatible shapes.")
        if len(left) < 1:
            raise ValueError("Family execution summary must include input validity.")
        xp = array_module(left[-1])
        return (
            *tuple(a + b for a, b in zip(left[:-1], right[:-1], strict=True)),
            xp.logical_and(left[-1], right[-1]),
        )

    def finalize_execution_summary(
        self, summary: tuple[float, ...]
    ) -> dict[str, float]:
        """Name base summary leaves after a bounded host reduction."""
        if len(summary) != 4:
            raise ValueError("Base family summary requires exactly four leaves.")
        return {
            "n_valid_rows": float(summary[0]),
            "n_positive_weight": float(summary[1]),
            "sum_log_positive_weight": float(summary[2]),
            "input_ok": bool(summary[3]),
        }

    def execution_summary_input_ok(self, summary: object) -> bool:
        """Return whether a finalized summary includes only valid real rows.

        Families with vector or mapping-shaped summary pytrees override this
        alongside their merge/finalize methods.  Keeping this family-owned
        avoids imposing a hidden fixed leaf layout on future reductions.
        """
        if not isinstance(summary, tuple) or len(summary) < 4:
            raise ValueError(
                f"{type(self).__name__} must implement "
                "execution_summary_input_ok() for its summary pytree."
            )
        return bool(summary[3])

    def regular_fletcher_statistics_from_batch(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return bounded Pearson/Fletcher sufficient statistics.

        This is the family-owned arithmetic used by the regular-family
        reported-scale reducer.  ``valid`` excludes padded rows only; it does
        not exclude real zero-weight observations, because mgcv's Fletcher
        correction and ``n.true`` are both unweighted (``gam.fit3.r``
        lines 596--604).  Families whose scale estimator differs must expose
        a different reduction policy rather than reuse this formula.
        """
        xp = array_module(y)
        valid = xp.asarray(valid, dtype=bool)
        variance = self.variance(mu)
        residual = y - mu
        pearson = xp.sum(xp.where(valid, wt * residual**2 / variance, 0.0))
        correction = xp.sum(xp.where(valid, self.dvar(mu) * residual / variance, 0.0))
        return pearson, correction, xp.sum(valid)

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

    def _static_cache_key(self) -> tuple:
        """Tuple of structural properties that affect a JIT trace.

        Two family instances that produce identical compiled code must
        return equal tuples. Used by ``__hash__`` / ``__eq__`` so that
        ``family`` can be a JAX ``static_argnames`` argument without
        per-instance recompilation (e.g. after ``copy.deepcopy``).

        Excludes any state that flows through as a dynamic JAX argument
        (e.g. estimated theta on extended families). The link contributes
        via ``link._cache_key()``, which is type-based for stateless
        built-ins and identity-based for unknown/parameterized subclasses.
        Subclasses extend this when additional state is baked into the trace.
        """
        return (
            type(self),
            self.link._cache_key(),
            self.n_theta,
            self.scale_known,
        )

    def __hash__(self) -> int:
        return hash(self._static_cache_key())

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return self._static_cache_key() == other._static_cache_key()
