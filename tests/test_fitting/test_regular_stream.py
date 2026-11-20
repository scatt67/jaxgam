"""JIT regular trial normalization and invalid-candidate reductions."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.standard import Binomial
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
)
from jaxgam.fitting.regular_stream import regular_trial_deviance
from tests.tolerances import STRICT


def test_regular_trial_jit_normalizes_zero_prior_responses():
    family = Binomial()
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    X = jnp.ones((3, 1))
    y = jnp.array([1.0, 0.0, 999.0])
    weight = jnp.array([1.0, 1.0, 0.0])
    kernel = jax.jit(
        lambda beta: regular_trial_deviance(
            X,
            y,
            weight,
            jnp.zeros(3),
            jnp.ones(3, dtype=bool),
            beta,
            parameters,
            family,
            context,
        )
    )
    deviance, domain = kernel(jnp.zeros(1))
    np.testing.assert_allclose(
        deviance, 4.0 * np.log(2.0), rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert bool(domain)
    _, invalid = kernel(jnp.array([np.inf]))
    assert not bool(invalid)
    # Finite coefficients and design inputs can still overflow X beta.
    # Candidate validity must reject that trial before convergence scoring.
    _, overflowed = jax.jit(
        regular_trial_deviance, static_argnames=("family", "context")
    )(
        2.0 * X,
        y,
        weight,
        jnp.zeros(3),
        jnp.ones(3, dtype=bool),
        jnp.array([1e308]),
        parameters,
        family,
        context,
    )
    assert not bool(overflowed)
