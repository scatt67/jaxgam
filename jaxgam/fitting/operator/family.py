"""Family-owned eta-level observed systems for bounded operator fitting."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
    _require_capabilities,
    _response_valid,
    _validate_context,
)


class OperatorFamilyRows(NamedTuple):
    """Actual row Newton system and stable direct-deviance reductions."""

    eta: jax.Array
    mu: jax.Array
    half_gradient: jax.Array
    observed_weight: jax.Array
    weighted_response: jax.Array
    deviance: jax.Array
    input_ok: jax.Array
    domain_ok: jax.Array
    finite_system: jax.Array


@partial(jax.jit, static_argnames=("family", "context"))
def operator_family_rows(
    eta: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    valid: jax.Array,
    parameters: FamilyExecutionParameters,
    family: ExponentialFamily,
    context: FamilyExecutionContext,
) -> OperatorFamilyRows:
    """Evaluate half Deta/Deta2 without an expanded design matrix.

    Pinned gam.fit4 forms Wz = W*(eta-offset)-Deta/2. This avoids division
    by zero curvature and retains the actual observed system. The host
    caller proves SPD and owns/validates the family snapshot before every
    dispatch; no Fisher fallback or curvature clipping occurs here.
    """
    _validate_context(family, context)
    _require_capabilities(
        context,
        "row_separable",
        "observed_information",
        "direct_deviance",
        "differentiable_deviance",
    )
    if eta.ndim != 1 or any(
        value.shape != eta.shape for value in (y, prior_weight, offset, valid)
    ):
        raise ValueError(
            "Operator family rows require matching one-dimensional arrays."
        )
    valid = jnp.asarray(valid, dtype=bool)
    normalized_y = family.execution_initial_response(y, prior_weight)
    input_rows = (
        _response_valid(family, normalized_y)
        & jnp.isfinite(eta)
        & jnp.isfinite(prior_weight)
        & (prior_weight >= 0)
        & jnp.isfinite(offset)
    )
    real = valid & input_rows
    padding = context.padding
    eta_candidate = jnp.where(real, eta, padding.eta)
    mu_candidate = family.link.inverse(eta_candidate)
    domain_rows = (
        real
        & family.valid_eta(eta_candidate)
        & family.valid_mu(mu_candidate)
        & jnp.isfinite(mu_candidate)
    )
    informative = domain_rows & (prior_weight > 0)
    # Zero-weight and unused rows need finite interior arguments before AD.
    eta_work = jnp.where(informative, eta_candidate, padding.eta)
    y_work = jnp.where(informative, normalized_y, padding.response)
    weight_work = jnp.where(informative, prior_weight, 0.0)
    offset_work = jnp.where(informative, offset, 0.0)
    theta_ok = jnp.all(jnp.isfinite(parameters.log_theta))
    theta = jnp.where(jnp.isfinite(parameters.log_theta), parameters.log_theta, 0.0)

    def smooth_deviance(values):
        means = family.link.inverse(values)
        return jnp.sum(
            family.deviance_derivative_contributions_for_parameters(
                y_work, means, weight_work, theta
            )
        )

    derivative = jax.grad(smooth_deviance)
    first, second = jax.jvp(derivative, (eta_work,), (jnp.ones_like(eta_work),))
    half_gradient = jnp.where(informative, 0.5 * first, 0.0)
    observed = jnp.where(informative, 0.5 * second, 0.0)
    weighted_response = observed * (eta_work - offset_work) - half_gradient
    mu_work = family.link.inverse(eta_work)
    direct = family.deviance_contributions_for_parameters(
        y_work, mu_work, weight_work, theta
    )
    deviance = jnp.sum(jnp.where(informative, direct, 0.0))
    input_ok = jnp.all(~valid | input_rows) & theta_ok
    domain_ok = input_ok & jnp.all(~valid | domain_rows)
    finite_system = (
        domain_ok
        & jnp.all(jnp.isfinite(half_gradient))
        & jnp.all(jnp.isfinite(observed))
        & jnp.all(jnp.isfinite(weighted_response))
        & jnp.isfinite(deviance)
    )
    return OperatorFamilyRows(
        eta_candidate,
        mu_candidate,
        half_gradient,
        observed,
        weighted_response,
        deviance,
        input_ok,
        domain_ok,
        finite_system,
    )
