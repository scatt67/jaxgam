"""ExtendedFamily base class for distributions with extra parameters.

Extended families (Negative Binomial, Tweedie, Beta, SHASH, etc.) have
distributional parameters beyond the standard exponential family that
must be estimated alongside smoothing parameters via Newton.

The theta interface uses arrays of shape ``(n_theta,)`` throughout, so
families with multiple extra parameters (e.g. ``scat`` with df + scale)
work without interface changes.

Design doc reference: Section 7.1
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from jaxgam.families.base import ExponentialFamily


class ExtendedFamily(ExponentialFamily):
    """Base class for families with extra parameters estimated via Newton.

    Subclasses must implement:

    - ``get_theta`` / ``put_theta``: mutable theta state for PIRLS runtime
    - ``deviance_fn``: pure-function factory ``D(eta, log_theta_vec) -> scalar``
      for the custom_jvp on PIRLS
    - ``working_weights_fn``: pure-function factory
      ``W(eta, log_theta_vec) -> (n,)`` for the custom_jvp on PIRLS
    - ``saturated_loglik_theta``: explicit-theta saturated log-likelihood
      for the REML criterion AD trace

    The fitting code branches on ``family.n_theta > 0`` (compile-time check
    via static ``family`` arg) to select the extended custom_jvp path.

    R source reference: efam.r (extended family objects)
    """

    @abstractmethod
    def get_theta(self, transformed: bool = False) -> np.ndarray:
        """Extra parameter vector, shape ``(n_theta,)``.

        Returns log-scale by default. When ``transformed=True``, returns
        natural scale (e.g. ``exp(log_theta)`` for positive parameters).

        Parameters
        ----------
        transformed : bool
            If True, return parameters on the natural (not log) scale.

        Returns
        -------
        np.ndarray, shape (n_theta,)
            Parameter vector.
        """
        ...

    @abstractmethod
    def put_theta(self, log_theta: np.ndarray) -> None:
        """Set extra parameter vector (log-scale), shape ``(n_theta,)``.

        Called by the Newton optimizer after each accepted step.

        Parameters
        ----------
        log_theta : np.ndarray, shape (n_theta,)
            New parameter values on log scale.
        """
        ...

    @abstractmethod
    def deviance_fn(self, y: np.ndarray, wt: np.ndarray):
        """Return pure JAX function ``D(eta, log_theta_vec) -> scalar``.

        ``log_theta_vec`` has shape ``(n_theta,)``.

        Used by the custom_jvp for IFT theta terms and joint JVPs.
        Must capture ``(y, wt, link)`` in closure; theta is an explicit
        argument for AD tracing.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
            Response values.
        wt : np.ndarray, shape (n,)
            Prior weights.

        Returns
        -------
        Callable[[jax.Array, jax.Array], jax.Array]
            Pure function ``(eta, log_theta_vec) -> scalar_deviance``.
        """
        ...

    @abstractmethod
    def working_weights_fn(self, wt: np.ndarray):
        """Return pure JAX function ``W(eta, log_theta_vec) -> (n,) array``.

        ``log_theta_vec`` has shape ``(n_theta,)``.

        Used by the custom_jvp for joint dW JVPs. Must capture
        ``(wt, link)`` in closure; theta is an explicit argument.

        Parameters
        ----------
        wt : np.ndarray, shape (n,)
            Prior weights.

        Returns
        -------
        Callable[[jax.Array, jax.Array], jax.Array]
            Pure function ``(eta, log_theta_vec) -> working_weights``.
        """
        ...

    @abstractmethod
    def saturated_loglik_theta(
        self,
        y: np.ndarray,
        wt: np.ndarray,
        scale: float,
        log_theta: np.ndarray,
    ):
        """Saturated log-likelihood with explicit theta for AD trace.

        ``log_theta`` has shape ``(n_theta,)``.

        Called inside ``_diff_score`` where ``log_theta`` is a traced
        JAX array. ``jax.grad`` differentiates through this w.r.t.
        ``log_theta``.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
            Response values.
        wt : np.ndarray, shape (n,)
            Prior weights.
        scale : float
            Dispersion parameter.
        log_theta : np.ndarray, shape (n_theta,)
            Extra parameters on log scale (traced JAX value).

        Returns
        -------
        jax.Array, scalar
            Saturated log-likelihood.
        """
        ...
