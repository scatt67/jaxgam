"""Starting value computation for PIRLS.

Provides ``initialize_beta`` which computes initial coefficient estimates
from the family's ``initialize(y, wt)`` → link → least-squares projection.

Design doc reference: Section 7.2 (initialization step)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.base import ExponentialFamily


def initialize_beta_cpu(
    X: np.ndarray,
    y: np.ndarray,
    wt: np.ndarray,
    family: ExponentialFamily,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """Compute starting coefficients for PIRLS.

    Steps:
    1. ``mu_init = family.initialize(y, wt)`` — family-specific start
    2. ``eta_init = link(mu_init)``
    3. ``beta_init = lstsq(X, eta_init - offset)``
    4. If the resulting fitted ``mu``/``eta`` leave the family's valid domain,
       fall back to the null model (constant ``eta = link(weighted mean(y))``).

    The fallback matters for non-canonical-positivity links such as the
    inverse-link Gamma: the per-observation target ``link(y) = 1/y`` is all
    positive, but its least-squares projection onto ``col(X)`` can have
    negative entries, giving ``mu = 1/eta < 0`` at the very first iteration.
    PIRLS step-halving then has no valid point to retreat toward and the fit
    diverges.  A constant target lies in ``col(X)`` whenever the model has an
    intercept and is always in the valid domain — mirroring R's use of
    ``null.coef`` as the valid step-halving anchor in ``gam.fit3``.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Model matrix.
    y : np.ndarray, shape (n,)
        Response values.
    wt : np.ndarray, shape (n,)
        Prior weights.
    family : ExponentialFamily
        Family with link function attached.
    offset : np.ndarray, shape (n,), optional
        Offset term. Defaults to zero.

    Returns
    -------
    np.ndarray, shape (p,)
        Initial coefficient vector in CPU fitting coordinates.
    """
    if offset is None:
        offset = np.zeros(len(y))

    mu_init = family.initialize(y, wt)
    eta_target = np.asarray(family.link.link(mu_init), dtype=np.float64)
    beta_init, _, _, _ = np.linalg.lstsq(X, eta_target - offset, rcond=None)

    # Validity guard: if the projected start leaves the family's domain,
    # fall back to the null model (constant eta), which is always valid.
    eta_fitted = X @ beta_init + offset
    mu_fitted = np.asarray(family.link.inverse(eta_fitted))
    valid = bool(np.all(family.valid_mu(mu_fitted))) and bool(
        np.all(family.valid_eta(eta_fitted))
    )
    if not valid:
        mu_bar = np.sum(wt * y) / np.sum(wt)
        eta_const = np.full_like(
            eta_target, float(family.link.link(np.asarray(mu_bar)))
        )
        beta_init, _, _, _ = np.linalg.lstsq(X, eta_const - offset, rcond=None)

    return beta_init


def initialize_beta(
    X: np.ndarray,
    y: np.ndarray,
    wt: np.ndarray,
    family: ExponentialFamily,
    offset: np.ndarray | None = None,
) -> jax.Array:
    """Compute a CPU start and transfer it to JAX for standalone callers."""
    return jnp.asarray(initialize_beta_cpu(X, y, wt, family, offset))


def r_null_coefficients(
    X: np.ndarray, y: np.ndarray, family: ExponentialFamily
) -> np.ndarray:
    """Pinned ``get.null.coef`` (mgcv.r): the divergence/validity anchor.

    ``qr.coef(qr(X), linkfun(mean(y)))`` with the *unweighted* response mean
    and no offset adjustment; it is distinct from the PIRLS start.
    """
    response_mean = float(np.mean(np.asarray(y, dtype=np.float64)))
    eta_mean = float(
        np.asarray(family.link.initial_link_cpu(np.asarray([response_mean])))[0]
    )
    if not np.isfinite(eta_mean):
        raise ValueError("The null coefficient anchor requires a finite link mean.")
    X = np.asarray(X, dtype=np.float64)
    return np.linalg.lstsq(X, np.full(X.shape[0], eta_mean), rcond=None)[0]
