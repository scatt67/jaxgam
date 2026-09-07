"""Numerical gates for the bounded NB lgamma-difference derivative kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import digamma, polygamma

from jaxgam.families.negative_binomial import _lgamma_diff_planned
from jaxgam.fitting.data import CountPrefixPlan


def _planned(y: jax.Array, capacity: int | None = None):
    capacity = int(jnp.max(y)) if capacity is None else capacity
    return lambda theta: jnp.sum(
        _lgamma_diff_planned(theta, y, y.astype(jnp.int64), capacity, True)
    )


def test_prefix_theta_gradient_and_hessian_match_recurrence() -> None:
    y = jnp.array([0.0, 1.0, 1.0, 3.0, 9.0])
    theta = jnp.array(0.125)
    fn = _planned(y)
    expected_grad = -sum(np.sum(1.0 / (float(theta) + np.arange(int(v)))) for v in y)
    expected_hess = sum(
        np.sum(1.0 / (float(theta) + np.arange(int(v))) ** 2) for v in y
    )
    np.testing.assert_allclose(jax.grad(fn)(theta), expected_grad, rtol=2e-13)
    np.testing.assert_allclose(jax.hessian(fn)(theta), expected_hess, rtol=2e-13)


def test_log_theta_derivatives_and_y_jvp() -> None:
    y = jnp.array([1.0, 4.0, 7.0])
    log_theta = jnp.array(np.log(2.5))

    def fn(lt):
        return _planned(y)(jnp.exp(lt))

    theta = float(jnp.exp(log_theta))
    g_theta = float(jax.grad(_planned(y))(jnp.array(theta)))
    h_theta = float(jax.hessian(_planned(y))(jnp.array(theta)))
    np.testing.assert_allclose(jax.grad(fn)(log_theta), theta * g_theta, rtol=2e-13)
    np.testing.assert_allclose(
        jax.hessian(fn)(log_theta), theta * g_theta + theta**2 * h_theta, rtol=2e-13
    )

    def y_fn(yy):
        return _lgamma_diff_planned(jnp.array(theta), yy, yy.astype(jnp.int64), 7, True)

    value, tangent = jax.jvp(y_fn, (y,), (jnp.ones_like(y),))
    np.testing.assert_allclose(tangent, -digamma(theta + np.asarray(y)), rtol=2e-13)
    assert np.all(np.isfinite(value))


def test_fractional_responses_keep_gamma_derivative() -> None:
    y = jnp.array([0.25, 1.5, 3.75])
    theta = jnp.array(4.0)

    def fn(t):
        return jnp.sum(
            _lgamma_diff_planned(t, y, jnp.zeros(3, dtype=jnp.int64), 0, False)
        )

    np.testing.assert_allclose(
        jax.grad(fn)(theta),
        len(y) * digamma(4.0) - np.sum(digamma(4.0 + np.asarray(y))),
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        jax.hessian(fn)(theta),
        len(y) * polygamma(1, 4.0) - np.sum(polygamma(1, 4.0 + np.asarray(y))),
        rtol=2e-13,
    )


def test_fractional_large_theta_uses_defined_gamma_path() -> None:
    y = jnp.array([0.5, 1.5])
    theta = jnp.array(1e12)

    def fn(t):
        return jnp.sum(
            _lgamma_diff_planned(t, y, jnp.zeros(2, dtype=jnp.int64), 0, False)
        )

    # This asserts the documented gamma-function derivative semantics. The
    # integer stable recurrence is deliberately not applied to fractions.
    y_np = np.asarray(y)
    expected = np.sum(-y_np / 1e12 + y_np * (y_np - 1.0) / (2.0 * 1e24))
    np.testing.assert_allclose(jax.grad(fn)(theta), expected, rtol=2e-13, atol=1e-24)
    expected_hessian = np.sum(y_np / 1e24 - y_np * (y_np - 1.0) / 1e36)
    np.testing.assert_allclose(
        jax.hessian(fn)(theta), expected_hessian, rtol=2e-13, atol=1e-30
    )


def test_empty_zero_and_plan_sentinel_are_safe() -> None:
    plan = CountPrefixPlan(jnp.zeros(0, dtype=jnp.int64), 0, 1, True)
    out = _lgamma_diff_planned(
        jnp.array(2.0), jnp.zeros(0), plan.indices, plan.capacity, True
    )
    assert out.shape == (0,)
    zeros = _lgamma_diff_planned(
        jnp.array(2.0), jnp.zeros(3), jnp.zeros(3, dtype=jnp.int64), 1, True
    )
    np.testing.assert_allclose(zeros, 0.0)


def test_undersized_metadata_is_not_silently_clipped() -> None:
    got = _lgamma_diff_planned(
        jnp.array(2.0), jnp.array([4.0]), jnp.array([4], dtype=jnp.int64), 3, True
    )
    assert np.isnan(
        np.asarray(
            jax.grad(
                lambda t: jnp.sum(
                    _lgamma_diff_planned(
                        t, jnp.array([4.0]), jnp.array([4], dtype=jnp.int64), 3, True
                    )
                )
            )(jnp.array(2.0))
        )
    )
    assert np.all(np.isfinite(got))
