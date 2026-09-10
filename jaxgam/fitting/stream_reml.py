"""Pure coefficient-space kernels for exact streamed REML derivatives.

The functions here deliberately operate on one already-materialized batch or
on coefficient statistics.  Source iteration lives in ``execution.reml`` so
no reverse-mode trace can retain a replayable source's rows.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.pirls import _W_MAX, _W_MIN
from jaxgam.fitting.reml import reml_criterion
from jaxgam.jax_utils import cho_factor


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
