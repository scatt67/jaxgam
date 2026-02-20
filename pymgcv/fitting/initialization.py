"""Starting value computation for PIRLS.

Provides ``initialize_beta`` which computes initial coefficient estimates
from the family's ``initialize(y, wt)`` → link → least-squares projection.

Design doc reference: Section 7.2 (initialization step)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from pymgcv.families.base import ExponentialFamily


def initialize_beta(
    X: np.ndarray,
    y: np.ndarray,
    wt: np.ndarray,
    family: ExponentialFamily,
    offset: np.ndarray | None = None,
) -> jnp.ndarray:
    """Compute starting coefficients for PIRLS.

    Steps:
    1. ``mu_init = family.initialize(y, wt)`` — family-specific start
    2. ``eta_init = link(mu_init)``
    3. ``beta_init = lstsq(X, eta_init - offset)``

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
    jax.Array, shape (p,)
        Initial coefficient vector as a JAX array.
    """
    if offset is None:
        offset = np.zeros(len(y))

    mu_init = family.initialize(y, wt)
    eta_init = np.asarray(family.link.link(mu_init), dtype=float)
    beta_init, _, _, _ = np.linalg.lstsq(X, eta_init - offset, rcond=None)
    return jnp.asarray(beta_init)
