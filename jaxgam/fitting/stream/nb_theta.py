"""Pure batch conditional NB theta reductions for streamed EFS PIRLS."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.efs_theta import (
    _efs_nb_log_deviance,
    _require_estimated_nb,
    _validate_static_inputs,
)
from jaxgam.links.links import LogLink, SqrtLink

jax.config.update("jax_enable_x64", True)


class NBConditionalThetaBatch(NamedTuple):
    """Additive scalar objective and derivatives with explicit trial theta."""

    nll: jax.Array
    gradient: jax.Array
    hessian: jax.Array
    admissible: jax.Array


def nb_conditional_theta_batch(
    log_theta: jax.Array,
    eta: jax.Array,
    y: jax.Array,
    weight: jax.Array,
    valid: jax.Array,
    family: NegativeBinomial,
    *,
    max_y: int,
    integer_counts: bool,
    debug: bool = False,
) -> NBConditionalThetaBatch:
    """Reduce dev/2-ls and its log-theta derivatives at fixed batch eta.

    Padding contributes zero without retaining predictors or count indices.
    The count-prefix capacity is global immutable metadata, independent of the
    trial theta. The inherited bounded-prefix/recurrence policy is unchanged.
    """
    _require_estimated_nb(family)
    indices = (
        jnp.where(valid, jnp.maximum(y, 0), 0).astype(jnp.int64)
        if integer_counts
        else jnp.zeros_like(y, dtype=jnp.int64)
    )
    _validate_static_inputs(
        log_theta,
        eta,
        y,
        weight,
        indices,
        max_y=max_y,
        integer_counts=integer_counts,
    )
    if valid.shape != y.shape or valid.dtype != jnp.bool_:
        raise ValueError("conditional theta valid mask must match batch rows")
    if any(a.dtype != jnp.float64 for a in (log_theta, eta, y, weight)):
        raise TypeError("conditional theta batch inputs require float64")
    safe_eta = jnp.where(valid, eta, 0.0 if isinstance(family.link, LogLink) else 1.0)
    safe_y = jnp.where(valid, y, 0.0)
    safe_weight = jnp.where(valid, weight, 0.0)

    # Conditional theta holds mu fixed. Express that same mean in log space
    # for every link, using the accepted stable eta-space NB deviance. This
    # avoids cancellation in nonlog theta derivatives without clipping valid
    # means or changing observed beta derivatives / ordinary family defaults.
    log_mu = (
        safe_eta
        if isinstance(family.link, LogLink)
        else jnp.log(
            jnp.square(safe_eta) if isinstance(family.link, SqrtLink) else safe_eta
        )
    )

    def objective(theta):
        deviance = _efs_nb_log_deviance(log_mu, theta, safe_y, safe_weight)
        saturated = family.saturated_loglik_theta(
            safe_y,
            safe_weight,
            1.0,
            theta,
            max_y=max_y,
            count_indices=indices,
            integer_counts=integer_counts,
        )
        return deviance / 2.0 - saturated

    nll, gradient = jax.value_and_grad(objective)(log_theta)
    hessian = jax.hessian(objective)(log_theta)
    admissible = (
        jnp.all(jnp.isfinite(log_theta))
        & jnp.all(jnp.isfinite(jnp.exp(log_theta)) & (jnp.exp(log_theta) > 0))
        & jnp.all(~valid | (jnp.isfinite(y) & (y >= 0)))
        & jnp.all(~valid | (jnp.isfinite(weight) & (weight >= 0)))
        & ~jnp.any(valid & (y > 0) & (y < 1))
        & jnp.all(~valid | (y <= max_y))
        & (jnp.all(~valid | (y == jnp.floor(y))) if integer_counts else True)
        & jnp.isfinite(nll)
        & jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(hessian))
    )
    result = NBConditionalThetaBatch(nll, gradient[0], hessian[0, 0], admissible)

    def report(_):
        jax.debug.print(
            "NB theta batch nll={n} g={g} H={h} valid={v}",
            n=result.nll,
            g=result.gradient,
            h=result.hessian,
            v=result.admissible,
        )

    jax.lax.cond(jnp.asarray(debug), report, lambda _: None, operand=None)
    return result


def nb_conditional_theta_step(
    gradient: jax.Array, hessian: jax.Array, max_step: float = 4.0
) -> tuple[jax.Array, jax.Array]:
    """Source scalar absolute-curvature repair and bounded Newton proposal."""
    curvature = jnp.where(
        hessian > 0, hessian, jnp.maximum(jnp.abs(hessian), jnp.abs(hessian) * 1e-5)
    )
    usable = jnp.isfinite(curvature) & (curvature > 0)
    raw = -gradient / jnp.where(usable, curvature, 1.0)
    return jnp.clip(raw, -max_step, max_step), usable & jnp.isfinite(raw)
