"""Raw stats Gaussian AIC diagnostics stay distinct from score likelihood."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.families.standard import Gaussian
from tests.helpers import _AssertCollector
from tests.r_bridge import RBridge
from tests.tolerances import STRICT


def _cases():
    y = np.array([0.2, 1.0, -0.5, 2.0])
    mu = np.array([0.0, 1.1, -0.4, 1.7])
    return [
        (y, mu, np.ones(4)),
        (y, mu, np.array([0.2, 0.7, 1.5, 4.0])),
        (y, mu, np.array([0.0, 1.0, 2.0, 0.0])),
        (y, mu, np.zeros(4)),
        (y, y, np.ones(4)),
        (y, y, np.array([0.0, 1.0, 2.0, 0.0])),
        (np.empty(0), np.empty(0), np.empty(0)),
        # Preserve the discovered finite-vs-Inf public-method discrepancy.
        (np.array([0.0, 1.0]), np.array([0.1, 0.9]), np.array([0.0, 1.0])),
    ]


@pytest.mark.usefixtures("r_bridge")
def test_gaussian_aic_cpu_matches_raw_pinned_weight_boundaries():
    cases = _cases()
    bridge = RBridge()
    checks = _AssertCollector()
    for i, (y, mu, w) in enumerate(cases):
        oracle = bridge.source_gaussian_aic(y, mu, w)
        for scale in (0.2, 1.0, 7.0):
            actual = Gaussian().aic(y, mu, w, scale)
            checks.check(
                f"case{i}/scale{scale}",
                lambda a=actual, r=oracle: np.testing.assert_allclose(
                    a, r, rtol=STRICT.rtol, atol=STRICT.atol, equal_nan=True
                ),
            )
    checks.raise_if_any("Pinned raw Gaussian AIC diagnostics")
    assert np.isposinf(Gaussian().aic(*cases[-1], scale=1.0))


@pytest.mark.usefixtures("r_bridge")
def test_gaussian_zero_prior_saturated_likelihood_remains_finite_cpu_jit():
    y = np.array([0.2, 1.0, -0.5, 2.0])
    w = np.array([0.2, 0.0, 0.7, 1.5])
    phi = 0.7
    oracle = RBridge().source_gaussian_saturated_likelihood(y, w, phi)
    family = Gaussian()

    def likelihood(scale):
        return family.saturated_loglik(jnp.asarray(y), jnp.asarray(w), scale)

    def derivatives(scale):
        return jnp.array(
            [
                likelihood(scale),
                jax.grad(likelihood)(scale),
                jax.grad(jax.grad(likelihood))(scale),
            ]
        )

    for eager in (False, True):
        with jax.disable_jit(eager):
            actual = jax.jit(derivatives)(phi)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, oracle, rtol=STRICT.rtol, atol=STRICT.atol)
