"""Scalar matched-state REML arithmetic for bounded operator EFS scores."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def operator_reml_score(
    penalized_deviance: jax.Array,
    saturated_loglik: jax.Array,
    score_phi: jax.Array,
    observed_logdet: jax.Array,
    penalty_logdet: jax.Array,
    penalty_null_dimension: jax.Array,
) -> jax.Array:
    """Match the repository's gamma-one REML score at one frozen fit state.

    Inputs must refer to the same coefficients, rho, theta and incoming score
    scale. In particular the Fisher determinant is not the observed determinant.
    Domain and provenance validation belong to the host adapter.
    """
    return (
        penalized_deviance / (2.0 * score_phi)
        - saturated_loglik
        + (observed_logdet - penalty_logdet) / 2.0
        - penalty_null_dimension * jnp.log(2.0 * jnp.pi * score_phi) / 2.0
    )
