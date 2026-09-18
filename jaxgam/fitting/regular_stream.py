"""Shared JIT trial reductions for regular source-style streamed PIRLS."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
    sanitize_batch_inputs,
)


@partial(jax.jit, static_argnames=("family", "context"))
def regular_trial_deviance(
    X: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    valid: jax.Array,
    beta: jax.Array,
    parameters: FamilyExecutionParameters,
    family: ExponentialFamily,
    context: FamilyExecutionContext,
) -> tuple[jax.Array, jax.Array]:
    """Normalize source responses before validating a global trial.

    Zero-prior Binomial response normalization is part of initialization,
    and must persist during trial and post-fit reductions. The legacy
    canonical trial kernel retains its own compatibility arithmetic.
    """
    normalized_y = family.execution_initial_response(y, prior_weight)
    y_safe, weight_safe, _offset, _eta, mu, domain = sanitize_batch_inputs(
        X, normalized_y, prior_weight, offset, valid, beta, family, context
    )
    contributions = family.deviance_contributions_for_parameters(
        y_safe, mu, weight_safe, parameters.log_theta
    )
    admissible = (
        jnp.all(~valid | domain)
        & jnp.all(jnp.isfinite(contributions))
        & jnp.all(jnp.isfinite(parameters.log_theta))
        & jnp.all(jnp.isfinite(beta))
    )
    return jnp.sum(jnp.where(valid, contributions, 0.0)), admissible
