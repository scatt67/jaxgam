"""Shared, row-separable family primitives for bounded fitting backends.

This module is intentionally a small Phase-2 consumer of methods declared on
``ExponentialFamily``.  It does not select a family by name and it does not
route public fits.  Dense PIRLS retains its existing operation order; the
streamed and future provider paths can opt into these primitives after their
own release-specific preflight.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.base import (
    ExponentialFamily,
    FamilyExecutionCapabilities,
    FamilyPadding,
    FamilyParameterSnapshot,
    StreamReductionPolicy,
)
from jaxgam.fitting.pirls import _W_MAX, _W_MIN, canonical_working_quantities

if TYPE_CHECKING:
    from jaxgam.formula.prepare import PreparedModel


@dataclass(frozen=True)
class FamilyExecutionContext:
    """Frozen static family/link configuration carried in a JIT cache key."""

    static_config: tuple[object, ...]
    capabilities: FamilyExecutionCapabilities
    padding: FamilyPadding
    reduction_policy: StreamReductionPolicy
    theta_mode: str
    phi_mode: str

    @classmethod
    def from_family(cls, family: ExponentialFamily) -> FamilyExecutionContext:
        snapshot = family.execution_parameter_snapshot()
        return cls(
            static_config=family.execution_static_config(),
            capabilities=family.execution_capabilities(),
            padding=family.execution_padding(),
            reduction_policy=family.stream_reduction_policy(),
            theta_mode=snapshot.theta_mode,
            phi_mode=snapshot.phi_mode,
        )


@dataclass(frozen=True)
class FamilyExecutionLineage:
    """Prepared/source lineage plus immutable family configuration snapshots.

    Call :meth:`validate` immediately before every cached device dispatch.
    JAX can legitimately reuse a previously compiled executable without
    retracing Python static arguments, so trace-time context validation alone
    cannot observe a mutation made after that executable entered the cache.
    """

    source_fingerprint: str
    basis_fingerprint: str
    context: FamilyExecutionContext
    parameters: FamilyParameterSnapshot

    @classmethod
    def from_prepared(
        cls, prepared: PreparedModel, family: ExponentialFamily
    ) -> FamilyExecutionLineage:
        fitting = prepared.fitting
        if fitting is None:
            raise ValueError("Family execution requires fitting preparation.")
        if (
            fitting.family_execution_static_config is None
            or fitting.family_parameter_snapshot is None
        ):
            raise NotImplementedError(
                "Prepared family lacks the CPU execution-contract snapshots "
                "required by streamed fitting."
            )
        if fitting.family_execution_static_config != family.execution_static_config():
            raise RuntimeError(
                "Family or link static configuration changed after fitting "
                "preparation; prepare again."
            )
        if fitting.family_parameter_snapshot != family.execution_parameter_snapshot():
            raise RuntimeError(
                "Family parameter state changed after fitting preparation; "
                "prepare again."
            )
        return cls(
            source_fingerprint=prepared.source_fingerprint,
            basis_fingerprint=prepared.basis_fingerprint,
            context=FamilyExecutionContext.from_family(family),
            parameters=family.execution_parameter_snapshot(),
        )

    def validate(self, prepared: PreparedModel, family: ExponentialFamily) -> None:
        """Fail closed when source, basis, family/link, or state was mutated."""
        if prepared.source_fingerprint != self.source_fingerprint:
            raise RuntimeError("RowSource changed after family execution preparation.")
        if prepared.basis_fingerprint != self.basis_fingerprint:
            raise RuntimeError(
                "Prepared basis changed after family execution preparation."
            )
        fitting = prepared.fitting
        if fitting is None:
            raise RuntimeError(
                "Prepared fitting metadata disappeared after preparation."
            )
        if (
            fitting.family_name != family.family_name
            or fitting.link_name != type(family.link).__qualname__
        ):
            raise RuntimeError(
                "Prepared family/link metadata changed after preparation."
            )
        if FamilyExecutionContext.from_family(family) != self.context:
            raise RuntimeError(
                "Family or link static configuration changed after preparation."
            )
        if family.execution_parameter_snapshot() != self.parameters:
            raise RuntimeError(
                "Family parameter state changed after preparation; construct a "
                "new execution lineage."
            )


@dataclass(frozen=True)
class FamilyExecutionParameters:
    """Explicit dynamic theta passed to JIT primitives as an array leaf.

    Scale remains a distinct fit-state argument to the likelihood reduction in
    this bounded first stage.  PR8.1 will introduce a lineage-bound trial
    scale snapshot rather than carry an unused ``log_phi`` placeholder here.
    """

    log_theta: jax.Array

    @classmethod
    def from_snapshot(
        cls, snapshot: FamilyParameterSnapshot
    ) -> FamilyExecutionParameters:
        return cls(log_theta=jnp.asarray(snapshot.log_theta, dtype=jnp.float64))


_PARAMETER_FIELDS = [field.name for field in fields(FamilyExecutionParameters)]
jax.tree_util.register_pytree_node(
    FamilyExecutionParameters,
    lambda state: ([getattr(state, name) for name in _PARAMETER_FIELDS], None),
    lambda _, children: FamilyExecutionParameters(
        **dict(zip(_PARAMETER_FIELDS, children, strict=True))
    ),
)


@dataclass(frozen=True)
class BatchWorkingQuantities:
    """Raw information plus legacy-clipped solver quantities for one batch."""

    eta: jax.Array
    mu: jax.Array
    working_weight: jax.Array
    fisher_weight: jax.Array
    observed_weight: jax.Array
    working_response: jax.Array
    deviance: jax.Array
    domain_ok: jax.Array


_BATCH_FIELDS = [field.name for field in fields(BatchWorkingQuantities)]
jax.tree_util.register_pytree_node(
    BatchWorkingQuantities,
    lambda result: ([getattr(result, name) for name in _BATCH_FIELDS], None),
    lambda _, children: BatchWorkingQuantities(
        **dict(zip(_BATCH_FIELDS, children, strict=True))
    ),
)


def _response_valid(family: ExponentialFamily, y: jax.Array) -> jax.Array:
    """JIT counterpart to the static response-support descriptor."""
    support = family.response_support
    lower = y >= support.lower if support.lower_inclusive else y > support.lower
    upper = y <= support.upper if support.upper_inclusive else y < support.upper
    return jnp.isfinite(y) & lower & upper


def _validate_context(
    family: ExponentialFamily, context: FamilyExecutionContext
) -> None:
    """Run at trace time; makes frozen config part of the JIT contract."""
    if FamilyExecutionContext.from_family(family) != context:
        raise RuntimeError(
            "Family execution context is stale; rebuild it after mutating a "
            "family or link."
        )


def _require_capabilities(context: FamilyExecutionContext, *capabilities: str) -> None:
    """Reject missing mathematical primitives before tracing AD arithmetic."""
    missing = [name for name in capabilities if not getattr(context.capabilities, name)]
    if missing:
        raise NotImplementedError(
            "Family execution context does not provide required capabilities: "
            + ", ".join(missing)
        )


def sanitize_batch_inputs(
    X: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    valid: jax.Array,
    beta: jax.Array,
    family: ExponentialFamily,
    context: FamilyExecutionContext,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Make every unused tail finite before link/family arithmetic."""
    valid = jnp.asarray(valid, dtype=bool)
    finite_X = jnp.all(jnp.isfinite(X), axis=1)
    finite_vectors = jnp.isfinite(y) & jnp.isfinite(prior_weight) & jnp.isfinite(offset)
    weight_ok = prior_weight >= 0.0
    response_ok = _response_valid(family, y)
    real_ok = valid & finite_X & finite_vectors & weight_ok & response_ok
    padding = context.padding
    X_safe = jnp.where(real_ok[:, None], X, 0.0)
    y_safe = jnp.where(real_ok, y, padding.response)
    weight_safe = jnp.where(real_ok, prior_weight, 0.0)
    offset_safe = jnp.where(real_ok, offset, 0.0)
    eta_raw = X_safe @ beta + offset_safe
    eta_ok = real_ok & jnp.isfinite(eta_raw) & family.valid_eta(eta_raw)
    eta_safe = jnp.where(eta_ok, eta_raw, padding.eta)
    mu = family.link.inverse(eta_safe)
    domain_per_row = real_ok & eta_ok & family.valid_mu(mu)
    return y_safe, weight_safe, offset_safe, eta_safe, mu, domain_per_row


@partial(jax.jit, static_argnames=("family", "context"))
def batch_working_quantities(
    X: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    valid: jax.Array,
    beta: jax.Array,
    parameters: FamilyExecutionParameters,
    family: ExponentialFamily,
    context: FamilyExecutionContext,
) -> BatchWorkingQuantities:
    """Compute generic masked family quantities without family-name dispatch.

    Raw observed information is returned without clipping.  The positive
    Fisher weight is clipped only for the legacy coefficient-system consumer;
    a future signed-observed solver must choose its own policy explicitly.
    """
    _validate_context(family, context)
    _require_capabilities(
        context,
        "row_separable",
        "fisher_working_system",
        "observed_information",
        "direct_deviance",
        "differentiable_deviance",
    )
    y_safe, weight_safe, offset_safe, eta, mu, domain_per_row = sanitize_batch_inputs(
        X, y, prior_weight, offset, valid, beta, family, context
    )
    theta_finite = jnp.all(jnp.isfinite(parameters.log_theta))
    theta_safe = jnp.where(
        jnp.isfinite(parameters.log_theta), parameters.log_theta, 0.0
    )
    fisher_raw, z = canonical_working_quantities(
        family,
        y_safe,
        mu,
        eta,
        weight_safe,
        offset_safe,
        log_theta=theta_safe,
    )

    def _deviance_at_eta(eta_value: jax.Array) -> jax.Array:
        mu_value = family.link.inverse(eta_value)
        return jnp.sum(
            family.deviance_derivative_contributions_for_parameters(
                y_safe, mu_value, weight_safe, theta_safe
            )
        )

    grad = jax.grad(_deviance_at_eta)
    _, second = jax.jvp(grad, (eta,), (jnp.ones_like(eta),))
    observed_raw = 0.5 * second
    valid = jnp.asarray(valid, dtype=bool)
    working_weight = jnp.where(valid, jnp.clip(fisher_raw, _W_MIN, _W_MAX), 0.0)
    working_response = jnp.where(valid, z, 0.0)
    direct = family.deviance_contributions_for_parameters(
        y_safe, mu, weight_safe, theta_safe
    )
    domain_ok = (
        jnp.all(jnp.logical_or(~valid, domain_per_row))
        & theta_finite
        & jnp.all(jnp.isfinite(fisher_raw))
        & jnp.all(jnp.isfinite(observed_raw))
        & jnp.all(jnp.isfinite(z))
        & jnp.all(jnp.isfinite(direct))
    )
    return BatchWorkingQuantities(
        eta=eta,
        mu=mu,
        working_weight=working_weight,
        fisher_weight=fisher_raw,
        observed_weight=observed_raw,
        working_response=working_response,
        deviance=jnp.sum(jnp.where(valid, direct, 0.0)),
        domain_ok=domain_ok,
    )


@partial(jax.jit, static_argnames=("family", "context", "max_y"))
def batch_saturated_loglikelihood(
    y: jax.Array,
    prior_weight: jax.Array,
    valid: jax.Array,
    scale: jax.Array,
    parameters: FamilyExecutionParameters,
    family: ExponentialFamily,
    context: FamilyExecutionContext,
    *,
    max_y: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Return a masked direct saturated likelihood and an input/domain flag."""
    _validate_context(family, context)
    _require_capabilities(context, "row_separable", "saturated_loglikelihood")
    valid = jnp.asarray(valid, dtype=bool)
    response_ok = _response_valid(family, y)
    finite = jnp.isfinite(y) & jnp.isfinite(prior_weight)
    real_ok = valid & response_ok & finite & (prior_weight >= 0.0)
    y_safe = jnp.where(real_ok, y, context.padding.response)
    weight_safe = jnp.where(real_ok, prior_weight, 0.0)
    theta_finite = jnp.all(jnp.isfinite(parameters.log_theta))
    theta_safe = jnp.where(
        jnp.isfinite(parameters.log_theta), parameters.log_theta, 0.0
    )
    value = family.saturated_loglikelihood_for_parameters(
        y_safe, weight_safe, scale, theta_safe, max_y=max_y
    )
    return (
        value,
        jnp.all(~valid | real_ok)
        & theta_finite
        & jnp.isfinite(scale)
        & (scale > 0.0)
        & jnp.isfinite(value),
    )


@partial(jax.jit, static_argnames=("family", "context"))
def batch_execution_summary(
    y: jax.Array,
    prior_weight: jax.Array,
    valid: jax.Array,
    family: ExponentialFamily,
    context: FamilyExecutionContext,
) -> tuple[jax.Array, ...]:
    """Build family-owned bounded metadata leaves for one source batch."""
    _validate_context(family, context)
    _require_capabilities(context, "row_separable")
    return family.execution_summary_from_batch(y, prior_weight, valid)


@partial(jax.jit, static_argnames=("family", "context"))
def merge_execution_summaries(
    left: tuple[jax.Array, ...],
    right: tuple[jax.Array, ...],
    family: ExponentialFamily,
    context: FamilyExecutionContext,
) -> tuple[jax.Array, ...]:
    """Merge same-family bounded summary pytrees without row retention."""
    _validate_context(family, context)
    _require_capabilities(context, "row_separable")
    return family.merge_execution_summaries(left, right)


def finalize_execution_summary(
    summary: object, family: ExponentialFamily
) -> dict[str, float]:
    """Finalize a global host reduction exactly once and reject bad real rows."""
    host_summary = jax.tree.map(np.asarray, summary)
    result = family.finalize_execution_summary(host_summary)
    if not family.execution_summary_input_ok(host_summary):
        raise ValueError("Family execution summary contains non-finite real rows.")
    return result
