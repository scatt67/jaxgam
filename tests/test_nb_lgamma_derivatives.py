"""Numerical gates for the bounded NB lgamma-difference derivative kernel."""

from __future__ import annotations

import subprocess
from decimal import Decimal, getcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, polygamma

from jaxgam.families import negative_binomial as nb
from jaxgam.families.negative_binomial import (
    NegativeBinomial,
    _lgamma_diff,
    _lgamma_diff_planned,
    _saturated_loglik_jax,
)
from jaxgam.families.standard import Poisson
from jaxgam.fitting.data import CountPrefixPlan
from tests.helpers import _make_nb_data, _setup_fd
from tests.r_bridge import RBridge
from tests.tolerances import MODERATE, STRICT


def _planned(y: jax.Array, capacity: int | None = None):
    capacity = int(jnp.max(y)) if capacity is None else capacity
    return lambda theta: jnp.sum(
        _lgamma_diff_planned(theta, y, y.astype(jnp.int64), capacity, True)
    )


def _decimal_integer_reference(theta: float, y: np.ndarray) -> tuple[Decimal, Decimal]:
    """Independent recurrence oracle for first/second theta derivatives."""
    getcontext().prec = 80
    theta_decimal = Decimal(str(theta))
    first = Decimal(0)
    second = Decimal(0)
    for count in np.asarray(y, dtype=np.int64):
        for k in range(int(count)):
            reciprocal = Decimal(1) / (theta_decimal + Decimal(k))
            first -= reciprocal
            second += reciprocal * reciprocal
    return first, second


_BERNOULLI_EVEN = (
    Decimal(1) / 6,
    Decimal(-1) / 30,
    Decimal(1) / 42,
    Decimal(-1) / 30,
    Decimal(5) / 66,
    Decimal(-691) / 2730,
    Decimal(7) / 6,
)


def _decimal_digamma(x: Decimal) -> Decimal:
    """Euler--Maclaurin oracle, accurate far beyond float64 at x >= 1e6."""
    value = x.ln() - Decimal(1) / (2 * x)
    for order, bernoulli in enumerate(_BERNOULLI_EVEN, start=1):
        value -= bernoulli / (Decimal(2 * order) * x ** (2 * order))
    return value


def _decimal_trigamma(x: Decimal) -> Decimal:
    value = Decimal(1) / x + Decimal(1) / (2 * x * x)
    for order, bernoulli in enumerate(_BERNOULLI_EVEN, start=1):
        value += bernoulli / x ** (2 * order + 1)
    return value


def _decimal_fractional_reference(theta: float, y: np.ndarray) -> tuple[float, float]:
    """Independent decimal digamma/trigamma differences before conversion."""
    theta_decimal = Decimal(str(theta))
    first = Decimal(0)
    second = Decimal(0)
    for count in np.asarray(y):
        count_decimal = Decimal(str(float(count)))
        first += _decimal_digamma(theta_decimal) - _decimal_digamma(
            theta_decimal + count_decimal
        )
        second += _decimal_trigamma(theta_decimal) - _decimal_trigamma(
            theta_decimal + count_decimal
        )
    return float(first), float(second)


def test_prefix_theta_gradient_and_hessian_match_recurrence() -> None:
    y = jnp.array([0.0, 1.0, 1.0, 3.0, 9.0])
    theta = jnp.array(0.125)
    fn = _planned(y)
    expected_grad = -sum(np.sum(1.0 / (float(theta) + np.arange(int(v)))) for v in y)
    expected_hess = sum(
        np.sum(1.0 / (float(theta) + np.arange(int(v))) ** 2) for v in y
    )
    np.testing.assert_allclose(jax.grad(fn)(theta), expected_grad, rtol=STRICT.rtol)
    np.testing.assert_allclose(jax.hessian(fn)(theta), expected_hess, rtol=STRICT.rtol)


def test_log_theta_derivatives_and_y_jvp() -> None:
    y = jnp.array([1.0, 4.0, 7.0])
    log_theta = jnp.array(np.log(2.5))

    def fn(lt):
        return _planned(y)(jnp.exp(lt))

    theta = float(jnp.exp(log_theta))
    g_theta = float(jax.grad(_planned(y))(jnp.array(theta)))
    h_theta = float(jax.hessian(_planned(y))(jnp.array(theta)))
    np.testing.assert_allclose(
        jax.grad(fn)(log_theta), theta * g_theta, rtol=STRICT.rtol
    )
    np.testing.assert_allclose(
        jax.hessian(fn)(log_theta),
        theta * g_theta + theta**2 * h_theta,
        rtol=STRICT.rtol,
    )

    def y_fn(yy):
        return _lgamma_diff_planned(jnp.array(theta), yy, yy.astype(jnp.int64), 7, True)

    value, tangent = jax.jvp(y_fn, (y,), (jnp.ones_like(y),))
    np.testing.assert_allclose(
        tangent, -digamma(theta + np.asarray(y)), rtol=STRICT.rtol
    )
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
        rtol=STRICT.rtol,
    )
    np.testing.assert_allclose(
        jax.hessian(fn)(theta),
        len(y) * polygamma(1, 4.0) - np.sum(polygamma(1, 4.0 + np.asarray(y))),
        rtol=STRICT.rtol,
    )


def test_huge_theta_matches_decimal_recurrence_in_theta_and_log_theta() -> None:
    """Large-theta integer path is checked without gamma/digamma subtraction."""
    y = jnp.array([1000.0, 3000.0, 7000.0])
    theta = 1e12
    expected_first_decimal, expected_second_decimal = _decimal_integer_reference(
        theta, np.asarray(y)
    )
    fn = _planned(y)
    got_first = theta * float(jax.grad(fn)(jnp.array(theta)))
    got_second = theta**2 * float(jax.hessian(fn)(jnp.array(theta)))
    theta_decimal = Decimal(str(theta))
    expected_first = float(theta_decimal * expected_first_decimal)
    expected_second = float(theta_decimal**2 * expected_second_decimal)
    np.testing.assert_allclose(
        got_first, expected_first, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        got_second, expected_second, rtol=STRICT.rtol, atol=STRICT.atol
    )

    def log_fn(lt):
        return fn(jnp.exp(lt))

    log_first_reference = float(theta_decimal * expected_first_decimal)
    log_second_reference = float(
        theta_decimal * expected_first_decimal
        + theta_decimal**2 * expected_second_decimal
    )

    np.testing.assert_allclose(
        jax.grad(log_fn)(jnp.log(theta)),
        log_first_reference,
        rtol=MODERATE.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        jax.hessian(log_fn)(jnp.log(theta)),
        log_second_reference,
        rtol=MODERATE.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize("theta", [1e6, 1_000_001.0, 1e12])
def test_fractional_large_theta_matches_decimal_oracle_across_transition(theta) -> None:
    y = jnp.array([0.5, 1.5])
    theta_jax = jnp.array(theta)

    def fn(t):
        return jnp.sum(
            _lgamma_diff_planned(t, y, jnp.zeros(2, dtype=jnp.int64), 0, False)
        )

    expected_first, expected_second = _decimal_fractional_reference(
        theta, np.asarray(y)
    )
    theta_scaled_first = theta * float(jax.grad(fn)(theta_jax))
    theta_scaled_second = theta**2 * float(jax.hessian(fn)(theta_jax))
    np.testing.assert_allclose(
        theta_scaled_first,
        theta * expected_first,
        rtol=MODERATE.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        theta_scaled_second,
        theta**2 * expected_second,
        rtol=MODERATE.rtol,
        atol=STRICT.atol,
    )

    log_theta = jnp.log(theta_jax)
    log_first = float(jax.grad(lambda lt: fn(jnp.exp(lt)))(log_theta))
    log_second = float(jax.hessian(lambda lt: fn(jnp.exp(lt)))(log_theta))
    np.testing.assert_allclose(
        log_first,
        theta * expected_first,
        rtol=MODERATE.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        log_second,
        theta * expected_first + theta**2 * expected_second,
        rtol=MODERATE.rtol,
        atol=STRICT.atol,
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


def test_undersized_metadata_is_not_silently_clipped_in_every_branch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(nb, "_PREFIX_WORKSPACE_BYTES", 0)
    for capacity in (0, 3, 100):
        y = jnp.array([float(capacity + 1)])
        indices = jnp.array([capacity + 1], dtype=jnp.int64)

        def fn(theta, y=y, indices=indices, capacity=capacity):
            return jnp.sum(_lgamma_diff_planned(theta, y, indices, capacity, True))

        assert np.isnan(np.asarray(jax.grad(fn)(jnp.array(1e12))))


def test_unplanned_fractional_inputs_do_not_cast_to_count_indices() -> None:
    y = jnp.array([0.25, 1.5, 3.75])
    theta = jnp.array(4.0)
    got = jax.grad(lambda t: jnp.sum(_lgamma_diff(t, y, 4)))(theta)
    expected = jax.grad(
        lambda t: jnp.sum(
            _lgamma_diff_planned(t, y, jnp.zeros(3, dtype=jnp.int64), 0, False)
        )
    )(theta)
    np.testing.assert_allclose(got, expected, rtol=STRICT.rtol, atol=STRICT.atol)


def test_public_family_default_dispatches_fractional_responses() -> None:
    family = NegativeBinomial(theta=4.0)
    y = jnp.array([0.25, 1.5, 3.75])
    weight = jnp.ones_like(y)
    log_theta = jnp.array([np.log(4.0)])
    got = jax.grad(
        lambda lt: family.saturated_loglik_theta(y, weight, 1.0, lt, max_y=4)
    )(log_theta)
    expected = jax.grad(
        lambda lt: _saturated_loglik_jax(
            y,
            weight,
            jnp.exp(lt[0]),
            0,
            jnp.zeros(y.shape, dtype=jnp.int64),
            False,
        )
    )(log_theta)
    np.testing.assert_allclose(got, expected, rtol=STRICT.rtol, atol=STRICT.atol)


def test_compiled_plan_uses_current_count_metadata() -> None:
    @jax.jit
    def gradient(theta, y, indices):
        return jax.grad(
            lambda t: jnp.sum(_lgamma_diff_planned(t, y, indices, 5, True))
        )(theta)

    theta = jnp.array(2.0)
    first = gradient(theta, jnp.array([1.0, 1.0]), jnp.array([1, 1], dtype=jnp.int64))
    second = gradient(theta, jnp.array([5.0, 5.0]), jnp.array([5, 5], dtype=jnp.int64))
    assert float(first) != float(second)


def test_fitting_data_prepares_integer_indices_once_for_nb_only() -> None:
    data = _make_nb_data(n=24, true_theta=2.0)
    nb_data = _setup_fd("y ~ s(x, k=6, bs='cr')", data, NegativeBinomial())
    assert nb_data.count_prefix_plan is not None
    assert nb_data.count_prefix_plan.indices.dtype == jnp.int64
    assert nb_data.count_prefix_plan.indices.shape == nb_data.y.shape

    poisson_data = _setup_fd("y ~ s(x, k=6, bs='cr')", data, Poisson())
    assert poisson_data.count_prefix_plan is None
    assert poisson_data.max_y == 0


def test_compiled_memory_respects_prefix_boundary_and_empty_fallback() -> None:
    max_fast_capacity = (
        nb._PREFIX_WORKSPACE_BYTES
        // (
            np.dtype(np.float64).itemsize
            * nb._PREFIX_DIFFERENTIATED_WORKSPACE_MULTIPLIER
        )
        - 1
    )

    def compile_hessian(y, indices, capacity, log_theta):
        def hessian(log_theta, y, indices):
            return jax.hessian(
                lambda theta: jnp.sum(
                    _lgamma_diff_planned(jnp.exp(theta), y, indices, capacity, True)
                )
            )(log_theta)

        return (
            jax.jit(hessian)
            .lower(jnp.array(log_theta), y, indices)
            .compile()
            .memory_analysis()
        )

    fast_memory = compile_hessian(
        jnp.array([0.0, 1.0, float(max_fast_capacity)]),
        jnp.array([0, 1, max_fast_capacity], dtype=jnp.int64),
        max_fast_capacity,
        np.log(2.0),
    )
    assert fast_memory.temp_size_in_bytes <= nb._PREFIX_WORKSPACE_BYTES

    fallback_capacity = max_fast_capacity + 1
    fallback_memory = compile_hessian(
        jnp.array([0.0, 1.0, float(fallback_capacity)]),
        jnp.array([0, 1, fallback_capacity], dtype=jnp.int64),
        fallback_capacity,
        np.log(1e12),
    )
    assert fallback_memory.temp_size_in_bytes <= 4096
    empty_output = _lgamma_diff_planned(
        jnp.array(2.0),
        jnp.zeros(0),
        jnp.zeros(0, dtype=jnp.int64),
        fallback_capacity,
        True,
    )
    assert empty_output.shape == (0,)
    fast_bytes = (max_fast_capacity + 1) * np.dtype(np.float64).itemsize
    assert (
        fast_bytes * nb._PREFIX_DIFFERENTIATED_WORKSPACE_MULTIPLIER
        <= nb._PREFIX_WORKSPACE_BYTES
    )
    assert (
        (max_fast_capacity + 2)
        * np.dtype(np.float64).itemsize
        * nb._PREFIX_DIFFERENTIATED_WORKSPACE_MULTIPLIER
        > nb._PREFIX_WORKSPACE_BYTES
    )


@pytest.mark.skipif(
    not RBridge.available() or not RBridge.check_versions()[0],
    reason="pinned R 4.5.2 + mgcv 1.9-3 unavailable",
)
def test_saturated_likelihood_derivatives_match_pinned_r_nb_ls() -> None:
    """Compare weighted log-theta ls derivatives to mgcv 1.9-3 directly."""
    r_code = """
    library(mgcv)
    y <- c(0, 1, 4, 8)
    w <- c(1, 2, 1, 1)
    z <- nb()$ls(y, w, log(2), 1)
    cat(z$ls, z$lsth1, z$lsth2, sep=' ')
    """
    completed = subprocess.run(
        ["Rscript", "-e", r_code], check=True, capture_output=True, text=True
    )
    r_value, r_grad, r_hess = map(float, completed.stdout.split()[-3:])
    family = NegativeBinomial(theta=2.0)
    y = jnp.array([0.0, 1.0, 4.0, 8.0])
    weight = jnp.array([1.0, 2.0, 1.0, 1.0])

    def fn(log_theta):
        return family.saturated_loglik_theta(y, weight, 1.0, log_theta, max_y=8)

    log_theta = jnp.array([np.log(2.0)])
    np.testing.assert_allclose(
        fn(log_theta), r_value, rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        jax.grad(fn)(log_theta)[0], r_grad, rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        jax.hessian(fn)(log_theta)[0, 0],
        r_hess,
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
