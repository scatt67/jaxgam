"""Pure JIT batch reductions used by host-streamed PIRLS.

There is intentionally no file iteration in this module.  Each kernel sees a
single bounded fitting-coordinate design matrix and returns only p-sized or
p-by-p statistics.  Padding is made finite before *any* family calculation,
then masked after the dense PIRLS working-weight clipping convention.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.pirls import (
    _W_MAX,
    _W_MIN,
    accepted_penalized_step,
    canonical_working_quantities,
)


def empty_statistics(n_coef: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Create device-resident ``(G, b, deviance, domain_ok)`` accumulators."""
    return (
        jnp.zeros((n_coef, n_coef), dtype=jnp.float64),
        jnp.zeros(n_coef, dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
        jnp.array(True),
    )


def _padding_response(family: ExponentialFamily) -> float:
    """A finite response in each initially-supported family domain."""
    if family.family_name == "binomial":
        return 0.5
    # Gaussian and Poisson both accept zero.  Unsupported families are
    # rejected by the host preflight before this static branch is traced.
    return 0.0


@jax.jit(static_argnames=("family",))
def accumulate_working_statistics(
    statistics: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    X: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    valid: jax.Array,
    beta: jax.Array,
    family: ExponentialFamily,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Add one canonical-family PIRLS reduction to coefficient statistics."""
    G, b, deviance, domain_ok = statistics
    valid = jnp.asarray(valid, dtype=bool)
    # Invalid tail entries may contain NaN sentinels.  Replace *all* their
    # inputs first: `where(valid, 0 * nan, 0)` would otherwise still produce
    # a NaN in family/link arithmetic on some backends.
    X_safe = jnp.where(valid[:, None], X, 0.0)
    y_safe = jnp.where(valid, y, _padding_response(family))
    weight_safe = jnp.where(valid, prior_weight, 0.0)
    offset_safe = jnp.where(valid, offset, 0.0)
    eta = X_safe @ beta + offset_safe
    mu = family.link.inverse(eta)
    working_weight, z = canonical_working_quantities(
        family, y_safe, mu, eta, weight_safe, offset_safe
    )
    # This order deliberately mirrors dense PIRLS.  Real zero prior weights
    # retain its floor-clipping behavior; padding is zeroed only afterwards.
    working_weight = jnp.clip(working_weight, _W_MIN, _W_MAX)
    working_weight = jnp.where(valid, working_weight, 0.0)
    z = jnp.where(valid, z, 0.0)
    sqrt_weight = jnp.sqrt(working_weight)
    weighted_X = sqrt_weight[:, None] * X_safe
    residuals = family.deviance_resids(y_safe, mu, weight_safe)
    batch_deviance = jnp.sum(jnp.where(valid, residuals**2, 0.0))
    batch_domain = jnp.all(
        jnp.logical_or(~valid, family.valid_mu(mu) & family.valid_eta(eta))
    )
    return (
        G + weighted_X.T @ weighted_X,
        b + weighted_X.T @ (sqrt_weight * z),
        deviance + batch_deviance,
        domain_ok & batch_domain,
    )


@jax.jit(static_argnames=("family",))
def trial_deviance(
    X: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    valid: jax.Array,
    beta: jax.Array,
    family: ExponentialFamily,
) -> tuple[jax.Array, jax.Array]:
    """Return one masked deviance/domain reduction for a trial coefficient."""
    valid = jnp.asarray(valid, dtype=bool)
    X_safe = jnp.where(valid[:, None], X, 0.0)
    y_safe = jnp.where(valid, y, _padding_response(family))
    weight_safe = jnp.where(valid, prior_weight, 0.0)
    offset_safe = jnp.where(valid, offset, 0.0)
    eta = X_safe @ beta + offset_safe
    mu = family.link.inverse(eta)
    residuals = family.deviance_resids(y_safe, mu, weight_safe)
    return (
        jnp.sum(jnp.where(valid, residuals**2, 0.0)),
        jnp.all(jnp.logical_or(~valid, family.valid_mu(mu) & family.valid_eta(eta))),
    )


@jax.jit(static_argnames=("family",))
def saturated_loglik_reduction(
    y: jax.Array,
    prior_weight: jax.Array,
    valid: jax.Array,
    scale: jax.Array,
    family: ExponentialFamily,
) -> jax.Array:
    """Return a finite, masked saturated-log-likelihood batch scalar."""
    valid = jnp.asarray(valid, dtype=bool)
    y_safe = jnp.where(valid, y, _padding_response(family))
    weight_safe = jnp.where(valid, prior_weight, 0.0)
    return family.saturated_loglik(y_safe, weight_safe, scale)


@jax.jit
def solve_penalized_system(
    G: jax.Array,
    b: jax.Array,
    structure: penalty_ops.JaxPenaltyStructure,
    log_lambda: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve an SPD canonical working system without rank-masking jitter.

    Cholesky NaNs are returned to the host, which reports a rank/conditioning
    failure rather than silently perturbing the model with unconditional
    jitter.
    """
    H = penalty_ops.add_to_dense(structure, G, log_lambda)
    H = 0.5 * (H + H.T)
    factor = jnp.linalg.cholesky(H)
    beta = jsla.cho_solve((factor, True), b)
    return beta, factor, H


@jax.jit
def coefficient_stationarity(H: jax.Array, b: jax.Array, beta: jax.Array) -> jax.Array:
    """Scale-invariant infinity-norm normal-equation residual."""
    residual = H @ beta - b
    scale = 1.0 + jnp.max(jnp.abs(b)) + jnp.max(jnp.abs(H @ beta))
    return jnp.max(jnp.abs(residual)) / scale


def accepts_trial(
    trial_deviance_value: jax.Array,
    current_penalized_deviance: jax.Array,
    beta_trial: jax.Array,
    structure: penalty_ops.JaxPenaltyStructure,
    log_lambda: jax.Array,
    domain_ok: jax.Array,
    first_iteration: bool,
) -> jax.Array:
    """Shared dense-PIRLS acceptance rule with penalty added once globally."""
    trial_penalized = trial_deviance_value + penalty_ops.quadratic(
        structure, beta_trial, log_lambda
    )
    return accepted_penalized_step(
        trial_penalized, current_penalized_deviance, domain_ok, first_iteration
    )
