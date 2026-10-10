"""Pure coefficient-space kernels for exact streamed REML derivatives.

The functions here deliberately operate on one already-materialized batch or
on coefficient statistics.  Source iteration lives in ``execution.reml`` so
no reverse-mode trace can retain a replayable source's rows.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
    batch_initial_working_quantities,
    batch_saturated_loglikelihood,
)
from jaxgam.fitting.pirls import _W_MAX, _W_MIN
from jaxgam.fitting.reml import reml_criterion
from jaxgam.jax_utils import cho_factor


class RegularBatchScoreVJP(NamedTuple):
    """One batch's cotangents and explicit derivative eligibility.

    A neutral batch is admissible; the host must separately require globally
    informative data. ``admissible`` is the strict per-batch result, while
    ``structural_admissible`` lets a source-aware host distinguish the sole
    unresolved-alpha condition and apply its global recovery certificate.
    ``alpha_exact_zero`` is diagnostic only: pinned R substitutes epsilon for
    that row and a mixed-success global system can remain full rank. No source
    iterator or coefficient-fit tape is retained.
    """

    beta: jax.Array
    log_phi: jax.Array
    admissible: jax.Array
    informative_count: jax.Array
    structural_admissible: jax.Array
    alpha_unresolved: jax.Array
    alpha_exact_zero: jax.Array


class RegularScoreCotangents(NamedTuple):
    """Scalar source score and explicit coefficient/statistic cotangents."""

    score: jax.Array
    rho: jax.Array
    observed: jax.Array
    source_beta: jax.Array
    deviance: jax.Array
    saturated: jax.Array
    log_phi: jax.Array


@partial(jax.jit, static_argnames=("family", "context"))
def regular_batch_statistics_vjp(
    beta: jax.Array,
    log_phi: jax.Array,
    X: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    valid: jax.Array,
    observed_cotangent: jax.Array,
    deviance_cotangent: jax.Array,
    saturated_cotangent: jax.Array,
    parameters: FamilyExecutionParameters,
    family: ExponentialFamily,
    context: FamilyExecutionContext,
) -> RegularBatchScoreVJP:
    """Differentiate regular score statistics through the family contract.

    The score's observed system uses the same good-row mask and canonical
    Fisher dispatch as the regular controller. Deviance uses the smooth
    derivative primitive rather than the reporting clamp. Actual saturated
    likelihood supplies the scale derivative. This function does not decide
    coefficient recovery. A generic caller must require ``admissible``; a
    source-aware host may inspect ``structural_admissible`` and
    ``alpha_unresolved`` only as inputs to its separate global convergence,
    factor, stationarity, information and lineage certificate.
    """
    if context.capabilities.dynamic_theta:
        raise NotImplementedError("Regular adjoints require a static family parameter.")
    valid = jnp.asarray(valid, dtype=bool)
    finite_X = jnp.all(jnp.isfinite(X), axis=1)
    X_safe = jnp.where((valid & finite_X)[:, None], X, 0.0)
    normalized_y = family.execution_initial_response(y, prior_weight)

    def statistics(beta_value: jax.Array) -> tuple[jax.Array, jax.Array]:
        eta = X_safe @ beta_value + offset
        working = batch_initial_working_quantities(
            normalized_y,
            prior_weight,
            offset,
            valid,
            eta,
            parameters,
            family,
            context,
        )
        score_weight = (
            working.fisher_weight
            if context.capabilities.fisher_equals_observed_for_score
            else working.observed_weight
        )
        observed = (X_safe.T * score_weight) @ X_safe
        # Undefined responses in padding/zero-prior rows must stay outside
        # both the direct primitive and its derivative graph.
        informative = working.informative_mask
        y_safe = jnp.where(informative, normalized_y, context.padding.response)
        weight_safe = jnp.where(informative, prior_weight, 0.0)
        eta_safe = jnp.where(informative, eta, context.padding.eta)
        mu_safe = family.link.inverse(eta_safe)
        contributions = family.deviance_derivative_contributions_for_parameters(
            y_safe, mu_safe, weight_safe, parameters.log_theta
        )
        return observed, jnp.sum(jnp.where(informative, contributions, 0.0))

    _, pullback = jax.vjp(statistics, beta)
    beta_bar = pullback((observed_cotangent, deviance_cotangent))[0]

    def saturated(log_phi_value: jax.Array) -> jax.Array:
        value, _ = batch_saturated_loglikelihood(
            normalized_y,
            prior_weight,
            valid,
            jnp.exp(log_phi_value),
            parameters,
            family,
            context,
        )
        return value

    saturated_value, saturated_derivative = jax.value_and_grad(saturated)(log_phi)
    working = batch_initial_working_quantities(
        normalized_y,
        prior_weight,
        offset,
        valid,
        X_safe @ beta + offset,
        parameters,
        family,
        context,
    )
    _, likelihood_ok = batch_saturated_loglikelihood(
        normalized_y,
        prior_weight,
        valid,
        jnp.exp(log_phi),
        parameters,
        family,
        context,
    )
    log_phi_bar = saturated_cotangent * saturated_derivative
    score_information_ok = (
        working.fisher_system_ok
        if context.capabilities.fisher_equals_observed_for_score
        else working.observed_information_ok
    )
    coefficient_system_ok = (
        working.fisher_system_ok
        if context.capabilities.coefficient_system == "fisher"
        else working.newton_system_ok
    )
    alpha_unresolved = jnp.any(working.alpha_resolution_unresolved)
    alpha_exact_zero = jnp.any(
        working.informative_mask & (working.newton_alpha_raw == 0.0)
    )
    structural_admissible = (
        jnp.all(jnp.isfinite(beta))
        & jnp.all(~valid | finite_X)
        & working.domain_ok
        & working.working_inputs_ok
        & coefficient_system_ok
        & score_information_ok
        & likelihood_ok
        & jnp.all(jnp.isfinite(beta_bar))
        & jnp.isfinite(saturated_value)
        & jnp.isfinite(log_phi_bar)
    )
    admissible = structural_admissible & ~alpha_unresolved
    return RegularBatchScoreVJP(
        beta_bar,
        log_phi_bar,
        admissible,
        working.informative_count,
        structural_admissible,
        alpha_unresolved,
        alpha_exact_zero,
    )


@partial(
    jax.jit,
    static_argnames=(
        "Mp",
        "singleton_sp_indices",
        "singleton_ranks",
        "multi_block_sp_indices",
        "multi_block_ranks",
    ),
)
def regular_score_cotangents(
    rho: jax.Array,
    log_phi: jax.Array,
    observed_inverse: jax.Array,
    source_beta: jax.Array,
    raw_deviance: jax.Array,
    saturated_loglik: jax.Array,
    log_det_hessian: jax.Array,
    penalty_structure: penalty_ops.JaxPenaltyStructure,
    Mp: int,
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    singleton_eig_constants: jax.Array,
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
) -> RegularScoreCotangents:
    """Differentiate the source-timed regular REML score at a frozen fit.

    ``raw_deviance`` is the pre-``gdi1`` deviance and ``source_beta`` is the
    candidate returned by the final coefficient solve.  The supplied
    determinant and inverse belong to the observed score factor.  Keeping
    those roles explicit avoids rebuilding a different normal-equation
    factor while retaining the exact local penalty derivative actions.
    """
    phi = jnp.exp(log_phi)
    penalty = penalty_ops.quadratic(penalty_structure, source_beta, rho)
    penalized_deviance = raw_deviance + penalty
    log_det_penalty = penalty_ops.log_pdet(
        rho,
        singleton_sp_indices,
        singleton_ranks,
        singleton_eig_constants,
        multi_block_sp_indices,
        multi_block_ranks,
        multi_block_proj_S,
    )
    score = (
        penalized_deviance / (2.0 * phi)
        - saturated_loglik
        + log_det_hessian / 2.0
        - log_det_penalty / 2.0
        - Mp / 2.0 * jnp.log(2.0 * jnp.pi * phi)
    )

    def explicit_rho(rho_value: jax.Array) -> jax.Array:
        # At the frozen state d log|H| = tr(H^-1 dS).  The observed inverse
        # comes from the retained signed QR factor, not a second factorization.
        penalty_value = penalty_ops.quadratic(penalty_structure, source_beta, rho_value)
        dense_penalty = penalty_ops.materialize(penalty_structure, rho_value)
        hessian_logdet = 0.5 * jnp.vdot(observed_inverse, dense_penalty.T).real
        penalty_logdet = penalty_ops.log_pdet(
            rho_value,
            singleton_sp_indices,
            singleton_ranks,
            singleton_eig_constants,
            multi_block_sp_indices,
            multi_block_ranks,
            multi_block_proj_S,
        )
        return penalty_value / (2.0 * phi) + hessian_logdet - penalty_logdet / 2.0

    rho_bar = jax.grad(explicit_rho)(rho)
    observed_bar = 0.5 * observed_inverse.T
    source_beta_bar = penalty_ops.apply(penalty_structure, source_beta, rho) / phi
    deviance_bar = 1.0 / (2.0 * phi)
    saturated_bar = jnp.array(-1.0, dtype=source_beta.dtype)
    log_phi_bar = -penalized_deviance / (2.0 * phi) - Mp / 2.0
    return RegularScoreCotangents(
        score,
        rho_bar,
        observed_bar,
        source_beta_bar,
        deviance_bar,
        saturated_bar,
        log_phi_bar,
    )


@partial(
    jax.jit,
    static_argnames=(
        "Mp",
        "singleton_sp_indices",
        "singleton_ranks",
        "multi_block_sp_indices",
        "multi_block_ranks",
        "rank_deficit",
    ),
)
def reml_score_cotangents(
    rho: jax.Array,
    xtwx: jax.Array,
    beta: jax.Array,
    deviance: jax.Array,
    saturated_loglik: jax.Array,
    penalty_structure: penalty_ops.JaxPenaltyStructure,
    Mp: int,
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    singleton_eig_constants: jax.Array,
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
    rank_deficit: int = 0,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Evaluate known-scale REML and cotangents of its explicit inputs.

    The returned cotangents are for ``(rho, XtWX, beta, deviance)``.  The
    saturated likelihood is constant for the initial Poisson/Binomial scope.
    Direct penalty and log-pseudo-determinant terms are therefore included in
    the rho and beta cotangents before the host performs a source VJP scan.
    """

    def score(
        rho_: jax.Array,
        xtwx_: jax.Array,
        beta_: jax.Array,
        deviance_: jax.Array,
    ) -> jax.Array:
        return reml_criterion(
            rho_,
            xtwx_,
            beta_,
            deviance_,
            saturated_loglik,
            penalty_structure,
            jnp.array(1.0, dtype=beta.dtype),
            Mp,
            singleton_sp_indices,
            singleton_ranks,
            singleton_eig_constants,
            multi_block_sp_indices,
            multi_block_ranks,
            multi_block_proj_S,
            rank_deficit,
        )

    return jax.value_and_grad(score, argnums=(0, 1, 2, 3))(rho, xtwx, beta, deviance)


@partial(jax.jit, static_argnames=("family",))
def batch_statistics_beta_vjp(
    beta: jax.Array,
    X: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    xtwx_cotangent: jax.Array,
    deviance_cotangent: jax.Array,
    family: ExponentialFamily,
) -> jax.Array:
    """VJP of one batch's observed statistics with respect to coefficients."""

    def xtwx_statistic(beta_: jax.Array) -> jax.Array:
        eta = X @ beta_ + offset
        mu = family.link.inverse(eta)
        working_weight = family.working_weights(mu, prior_weight)
        return (X.T * working_weight) @ X

    _, pullback = jax.vjp(xtwx_statistic, beta)
    eta = X @ beta + offset
    mu = family.link.inverse(eta)
    # Canonical exponential-family identity dD/deta = 2 * wt * (mu - y).
    # Keep it local to this initially Poisson/Binomial-only derivative path:
    # the dense reference's wider family treatment remains unchanged.
    deviance_vjp = 2.0 * X.T @ (prior_weight * (mu - y))
    return pullback(xtwx_cotangent)[0] + deviance_cotangent * deviance_vjp


@partial(jax.jit, static_argnames=("family",))
def batch_is_adjoint_interior(
    beta: jax.Array,
    X: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    family: ExponentialFamily,
) -> jax.Array:
    """Check the initial derivative scope avoids clipping/domain boundaries."""
    eta = X @ beta + offset
    mu = family.link.inverse(eta)
    working_weight = family.working_weights(mu, prior_weight)
    return jnp.all(
        family.valid_eta(eta)
        & family.valid_mu(mu)
        & jnp.isfinite(working_weight)
        & (working_weight > _W_MIN)
        & (working_weight < _W_MAX)
    )


@jax.jit
def solve_observed_adjoint(
    xtwx: jax.Array,
    beta_cotangent: jax.Array,
    rho: jax.Array,
    penalty_structure: penalty_ops.JaxPenaltyStructure,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve the dense custom-JVP IFT convention for one adjoint vector.

    This intentionally uses ``cho_factor`` rather than reusing the REML
    core's diagonally-scaled log-determinant factor: they are different
    numerical operations in the existing dense derivative implementation.
    The host has already gated rank and clipping boundaries before this call.
    """
    H = penalty_ops.add_to_dense(penalty_structure, xtwx, rho)
    factor, jitter = cho_factor(H)
    adjoint = jsla.cho_solve((factor, True), beta_cotangent)
    residual = H @ adjoint - beta_cotangent
    scale = 1.0 + jnp.max(jnp.abs(beta_cotangent)) + jnp.max(jnp.abs(H @ adjoint))
    return adjoint, jitter, jnp.max(jnp.abs(residual)) / scale
