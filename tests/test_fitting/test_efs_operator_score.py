"""Matched-state scalar score agrees with pinned gam.fit4 REML arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.fitting.operator.efs_score import operator_reml_score
from tests.tolerances import STRICT


def test_operator_reml_score_jit_matches_pinned_scalar_formula():
    # gam.fit4.r: REML=((dev+P)/(2*scale)-ls)/gamma
    #                 +(oo$ldet-rp$det)/2-Mp*log(2*pi*scale)/2, gamma=1.
    deviance, penalty, phi = 12.25, 3.75, 1.7
    saturated, logdet_h, logdet_s, null_dimension = -18.4, 4.6, 2.1, 3
    jit_score = jax.jit(operator_reml_score)
    actual = jit_score(
        jnp.asarray(deviance + penalty),
        jnp.asarray(saturated),
        jnp.asarray(phi),
        jnp.asarray(logdet_h),
        jnp.asarray(logdet_s),
        jnp.asarray(null_dimension),
    )
    expected = (
        (deviance + penalty) / (2 * phi)
        - saturated
        + (logdet_h - logdet_s) / 2
        - null_dimension * np.log(2 * np.pi * phi) / 2
    )
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)


def test_score_inputs_keep_incoming_scale_distinct_from_reported_scale():
    score_at_incoming = operator_reml_score(16.0, -18.4, 1.7, 4.6, 2.1, 3)
    score_at_reported = operator_reml_score(16.0, -18.4, 1.8, 4.6, 2.1, 3)
    assert float(score_at_incoming) != float(score_at_reported)
