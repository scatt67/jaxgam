"""Pinned single-response likelihood and valid-domain Binomial curvature."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.families.standard import Binomial
from tests.r_bridge import RBridge
from tests.tolerances import STRICT


@pytest.mark.usefixtures("r_bridge")
def test_binomial_fractional_trial_weights_follow_pinned_dbinom():
    weights = np.r_[np.geomspace(0.05, 100.0, 81), 0.8, 0.0]
    y = np.resize(
        np.array([0.0, 0.6, 1.0, np.nextafter(1.0, 0.0), 1.0 - 1e-12]), len(weights)
    )
    y[-1] = 0.0
    mu = np.linspace(0.13, 0.87, len(y))
    reference = RBridge(mode="rpy2").binomial_likelihood_aic(y, weights, mu)
    family = Binomial()
    for disabled in (False, True):
        with jax.disable_jit(disabled):
            likelihood = jax.jit(lambda yy, ww: family.saturated_loglik(yy, ww, 1.0))(
                jnp.asarray(y), jnp.asarray(weights)
            )
        np.testing.assert_allclose(
            likelihood, reference[0], rtol=STRICT.rtol, atol=STRICT.atol
        )
    np.testing.assert_allclose(
        family.aic(y, mu, weights, 1.0),
        reference[1],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.usefixtures("r_bridge")
def test_binomial_valid_boundary_means_retain_source_deviance_and_curvature():
    mu = np.array([1e-12, 1e-11, 0.2, 1.0 - 1e-12, np.nextafter(1.0, 0.0)])
    y = np.array([0.0, 0.1, 0.7, 1.0 - 1e-12, 1.0])
    weight = np.array([0.3, 0.8, 1.2, 2.0, 0.7])
    reference = RBridge(mode="rpy2").binomial_deviance_curvature(y, weight, mu)
    family = Binomial()
    np.testing.assert_allclose(
        family.deviance_contributions(y, mu, weight),
        reference[:, 0],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        family.deviance_resids(y, mu, weight) ** 2,
        reference[:, 0],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )

    def direct(mean):
        return jnp.sum(
            family.deviance_derivative_contributions(
                jnp.asarray(y), mean, jnp.asarray(weight)
            )
        )

    kernel = jax.jit(lambda mean: 0.5 * jax.hessian(direct)(mean))
    for disabled in (False, True):
        with jax.disable_jit(disabled):
            hessian = kernel(jnp.asarray(mu))
        np.testing.assert_allclose(
            hessian, np.diag(reference[:, 1]), rtol=STRICT.rtol, atol=STRICT.atol
        )
    assert np.all(reference[:, 1] > 0)


@pytest.mark.usefixtures("r_bridge")
def test_binomial_cpu_aic_preserves_pinned_tail_endpoint_and_zero_trial_terms():
    y = np.array([0.0, 1.0, 0.6, 1.0, 0.0, 1.0, 777.0, -777.0, 1.0, 0.0])
    weight = np.array([0.3, 0.8, 2.3, 15.9, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    mu = np.array(
        [
            1e-12,
            1.0 - 1e-12,
            0.4,
            np.nextafter(1.0, 0.0),
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ]
    )
    reference = RBridge(mode="rpy2").binomial_aic_rows(y, weight, mu)
    family = Binomial()
    actual = np.array(
        [
            family.aic(y[i : i + 1], mu[i : i + 1], weight[i : i + 1], 1.0)
            for i in range(len(y))
        ]
    )
    np.testing.assert_allclose(actual, reference, rtol=STRICT.rtol, atol=STRICT.atol)
    assert np.all(np.isfinite(actual[:-2]))
    assert np.all(np.isposinf(actual[-2:]))
    assert np.all(actual[4:8] == 0.0)
