"""Pure batch derivatives for streamed Negative Binomial REML.

This module owns row-local differentiation only.  A host supplies the frozen
coefficient state, score cotangents and explicit trial ``log_theta``; source
iteration, coefficient factors and optimizer decisions remain outside the
kernel.  The implementation follows mgcv 1.9-3 ``nb()$Dd``, ``nb()$ls`` and
``gam.fit4``'s final observed-information system.

The coefficient stationarity convention is
``g = X.T @ (0.5 * Deta) + S @ beta``.  It is exactly half of the dense
custom-JVP convention ``X.T @ Deta + 2 * S @ beta``.  Consequently both the
observed factor and ``partial_log_theta g`` use the half-deviance convention;
the future host must solve with that same factor before applying
``-adjoint.T @ partial_log_theta_g``.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
)
from jaxgam.fitting.nb_stream_kernels import nb_working_batch
from jaxgam.links.links import IdentityLink, LogLink, SqrtLink


class NBBatchScoreVJP(NamedTuple):
    """One batch's exact score and stationarity derivatives.

    ``beta`` and ``log_theta`` are cotangents of the explicit score
    statistics. ``stationarity_log_theta`` is the fitting-coordinate vector
    ``partial g / partial log_theta`` and is intentionally not contracted
    with a host adjoint here.  All arrays have size bounded by the batch or
    coefficient dimension; no source rows or reverse tape escape the call.
    """

    beta: jax.Array
    log_theta: jax.Array
    stationarity_log_theta: jax.Array
    admissible: jax.Array
    source_good_count: jax.Array
    informative_count: jax.Array


def _link_name(family: NegativeBinomial) -> str:
    if isinstance(family.link, LogLink):
        return "log"
    if isinstance(family.link, IdentityLink):
        return "identity"
    if isinstance(family.link, SqrtLink):
        return "sqrt"
    raise NotImplementedError("streamed NB adjoints support log, identity, and sqrt")


def _validate_shapes(
    beta: jax.Array,
    X: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    valid: jax.Array,
    count_indices: jax.Array,
    observed_cotangent: jax.Array,
    parameters: FamilyExecutionParameters,
) -> None:
    if X.ndim != 2 or beta.shape != (X.shape[1],):
        raise ValueError("NB adjoint beta/X shapes are inconsistent")
    if any(value.shape != (X.shape[0],) for value in (y, prior_weight, offset, valid)):
        raise ValueError("NB adjoint batch vectors must match the X row count")
    if count_indices.shape != (X.shape[0],):
        raise ValueError("NB count indices must match the X row count")
    if observed_cotangent.shape != (X.shape[1], X.shape[1]):
        raise ValueError("NB observed cotangent must be a square coefficient matrix")
    if parameters.log_theta.shape != (1,):
        raise ValueError("NB log_theta must have shape (1,)")


@partial(
    jax.jit,
    static_argnames=("family", "context", "max_y", "integer_counts"),
)
def nb_batch_statistics_vjp(
    beta: jax.Array,
    X: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    valid: jax.Array,
    count_indices: jax.Array,
    observed_cotangent: jax.Array,
    deviance_cotangent: jax.Array,
    saturated_cotangent: jax.Array,
    parameters: FamilyExecutionParameters,
    family: NegativeBinomial,
    context: FamilyExecutionContext,
    *,
    max_y: int,
    integer_counts: bool,
) -> NBBatchScoreVJP:
    """Differentiate final NB score statistics at explicit dynamic theta.

    The observed system is the source ``0.5 * Deta2`` system used by
    ``gam.fit4``/``gdi2``.  Saturated-likelihood differentiation uses the
    accepted bounded count-prefix primitive and therefore needs the global
    count plan supplied explicitly.  ``phi`` is mathematically fixed at one
    for this family and is not represented by a placeholder derivative. A
    fixed-theta host reuses the beta cotangent and omits the returned theta
    coordinate from its parameter vector.
    """
    if not isinstance(family, NegativeBinomial):
        raise TypeError("NB batch adjoints require NegativeBinomial")
    if context.theta_mode not in ("estimated", "fixed"):
        raise NotImplementedError("NB batch adjoints require explicit NB theta")
    if context.phi_mode != "known":
        raise ValueError("NB batch adjoints require known phi=1")
    if context != FamilyExecutionContext.from_family(family):
        raise RuntimeError("NB family execution context is stale")
    _validate_shapes(
        beta,
        X,
        y,
        prior_weight,
        offset,
        valid,
        count_indices,
        observed_cotangent,
        parameters,
    )
    link = _link_name(family)
    valid = jnp.asarray(valid, dtype=bool)
    finite_X = jnp.all(jnp.isfinite(X), axis=1)
    X_safe = jnp.where((valid & finite_X)[:, None], X, 0.0)
    real_ok = (
        valid
        & finite_X
        & jnp.isfinite(y)
        & (y >= 0.0)
        & jnp.isfinite(prior_weight)
        & (prior_weight >= 0.0)
        & jnp.isfinite(offset)
    )
    y_safe = jnp.where(real_ok, y, 0.0)
    weight_safe = jnp.where(real_ok, prior_weight, 0.0)
    offset_safe = jnp.where(real_ok, offset, 0.0)
    count_safe = jnp.where(real_ok, count_indices, 0)

    def statistics(
        beta_value: jax.Array, log_theta_value: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        trial_parameters = FamilyExecutionParameters(log_theta_value)
        working = nb_working_batch(
            X_safe @ beta_value + offset_safe,
            y_safe,
            weight_safe,
            offset_safe,
            valid,
            trial_parameters,
            link=link,
        )
        good = working.direct_rows
        observed_weight = jnp.where(good, working.observed_weight, 0.0)
        observed = (X_safe.T * observed_weight) @ X_safe
        saturated = family.saturated_loglik_theta(
            y_safe,
            weight_safe,
            1.0,
            log_theta_value,
            max_y=max_y,
            count_indices=count_safe,
            integer_counts=integer_counts,
        )
        return observed, working.deviance, saturated

    _, pullback = jax.vjp(statistics, beta, parameters.log_theta)
    beta_bar, log_theta_bar = pullback(
        (observed_cotangent, deviance_cotangent, saturated_cotangent)
    )

    def stationarity(log_theta_value: jax.Array) -> jax.Array:
        working = nb_working_batch(
            X_safe @ beta + offset_safe,
            y_safe,
            weight_safe,
            offset_safe,
            valid,
            FamilyExecutionParameters(log_theta_value),
            link=link,
        )
        # derivative_rhs = -0.5 * Deta.  gam.fit4/gdi2 drops rows outside
        # the final finite observed/direct-response set.
        half_gradient = jnp.where(working.direct_rows, -working.derivative_rhs, 0.0)
        return X_safe.T @ half_gradient

    stationarity_log_theta = jax.jacfwd(stationarity)(parameters.log_theta)[:, 0]
    current = nb_working_batch(
        X_safe @ beta + offset_safe,
        y_safe,
        weight_safe,
        offset_safe,
        valid,
        parameters,
        link=link,
    )
    _, _, saturated_value = statistics(beta, parameters.log_theta)
    source_good_count = jnp.sum(current.direct_rows, dtype=jnp.int64)
    informative_count = jnp.sum(
        current.direct_rows & (weight_safe > 0.0), dtype=jnp.int64
    )
    admissible = (
        jnp.all(~valid | real_ok)
        & current.domain_ok
        & (source_good_count > 0)
        & jnp.all(jnp.isfinite(beta))
        & jnp.all(jnp.isfinite(parameters.log_theta))
        & jnp.all(jnp.isfinite(beta_bar))
        & jnp.all(jnp.isfinite(log_theta_bar))
        & jnp.all(jnp.isfinite(stationarity_log_theta))
        & jnp.isfinite(saturated_value)
    )
    return NBBatchScoreVJP(
        beta_bar,
        log_theta_bar,
        stationarity_log_theta,
        admissible,
        source_good_count,
        informative_count,
    )
