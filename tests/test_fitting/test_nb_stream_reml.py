"""Dynamic-theta batch derivatives for exact streamed NB REML."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import root

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
)
from jaxgam.fitting.nb_stream_reml import nb_batch_statistics_vjp
from tests.helpers import r_available
from tests.r_bridge import RBridge
from tests.tolerances import STRICT

_LINKS = ("log", "identity", "sqrt")


def _inputs(link: str, theta: float = 2.7):
    X = np.array(
        [
            [1.0, -0.8, 0.3],
            [1.0, -0.4, -0.2],
            [1.0, 0.0, 0.5],
            [1.0, 0.35, -0.6],
            [1.0, 0.8, 0.1],
            [1.0, 1.1, 0.7],
        ]
    )
    beta = np.array([1.1, 0.16, -0.09])
    y = np.array([0.0, 1.0, 2.0, 4.0, 7.0, 3.0])
    weight = np.array([0.7, 1.1, 0.9, 1.25, 0.8, 1.05])
    offset = np.array([0.04, -0.03, 0.02, -0.01, 0.03, -0.02])
    observed_bar = np.array(
        [[0.21, -0.04, 0.03], [0.07, 0.18, -0.02], [-0.01, 0.05, 0.14]]
    )
    family = NegativeBinomial(theta=theta, fixed=False, link=link)
    return (
        family,
        jnp.asarray(beta),
        jnp.asarray(X),
        jnp.asarray(y),
        jnp.asarray(weight),
        jnp.asarray(offset),
        jnp.ones(len(y), dtype=bool),
        jnp.asarray(y, dtype=jnp.int64),
        jnp.asarray(observed_bar),
        jnp.array(0.63),
        jnp.array(-0.82),
        FamilyExecutionParameters(jnp.asarray([np.log(theta)])),
    )


def _run(link: str, theta: float = 2.7):
    family, *arguments = _inputs(link, theta)
    return nb_batch_statistics_vjp(
        *arguments,
        family,
        FamilyExecutionContext.from_family(family),
        max_y=7,
        integer_counts=True,
    )


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", _LINKS)
def test_nb_batch_derivatives_match_pinned_source_contractions(link):
    """Dd/ls contractions and half-gradient theta stationarity match R."""
    family, beta, X, y, weight, offset, valid, _, observed_bar, dbar, lsbar, _ = (
        _inputs(link)
    )
    result = _run(link)
    oracle = RBridge(mode="rpy2").nb_batch_derivative_contractions(
        link,
        2.7,
        np.asarray(beta),
        np.asarray(X),
        np.asarray(y),
        np.asarray(weight),
        np.asarray(offset),
        np.asarray(observed_bar),
        float(dbar),
        float(lsbar),
    )
    np.testing.assert_allclose(
        result.beta, oracle["beta"], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.log_theta[0],
        oracle["log.theta"][0],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.stationarity_log_theta,
        oracle["stationarity"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert result.source_good_count == oracle["good"]
    assert result.informative_count == len(y)
    assert result.admissible
    assert valid.all()
    assert family.n_theta == 1


@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("theta", [0.1, 2.7, 1e6])
def test_nb_batch_derivatives_match_independent_dense_ad(link, theta):
    """Explicit theta changes all three derivative paths without stale state."""
    family, beta, X, y, weight, offset, valid, indices, C, dbar, lsbar, params = (
        _inputs(link, theta)
    )
    context = FamilyExecutionContext.from_family(family)

    def dense_statistics(beta_value, log_theta_value):
        eta = X @ beta_value + offset
        dev_fn = family.deviance_fn(y, weight)
        grad_eta = jax.grad(dev_fn, argnums=0)
        _, d2 = jax.jvp(
            lambda value: grad_eta(value, log_theta_value),
            (eta,),
            (jnp.ones_like(eta),),
        )
        observed = (X.T * (0.5 * d2)) @ X
        saturated = family.saturated_loglik_theta(
            y,
            weight,
            1.0,
            log_theta_value,
            max_y=7,
            count_indices=indices,
            integer_counts=True,
        )
        return observed, dev_fn(eta, log_theta_value), saturated

    _, pullback = jax.vjp(dense_statistics, beta, params.log_theta)
    expected_beta, expected_theta = pullback((C, dbar, lsbar))
    grad_eta = jax.grad(family.deviance_fn(y, weight), argnums=0)
    expected_stationarity = jax.jacfwd(
        lambda lt: 0.5 * X.T @ grad_eta(X @ beta + offset, lt)
    )(params.log_theta)[:, 0]
    result = nb_batch_statistics_vjp(
        beta,
        X,
        y,
        weight,
        offset,
        valid,
        indices,
        C,
        dbar,
        lsbar,
        params,
        family,
        context,
        max_y=7,
        integer_counts=True,
    )
    np.testing.assert_allclose(
        result.beta, expected_beta, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.log_theta, expected_theta, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.stationarity_log_theta,
        expected_stationarity,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    with jax.disable_jit():
        eager = nb_batch_statistics_vjp(
            beta,
            X,
            y,
            weight,
            offset,
            valid,
            indices,
            C,
            dbar,
            lsbar,
            params,
            family,
            context,
            max_y=7,
            integer_counts=True,
        )
    for actual, wanted in zip(result, eager, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=STRICT.rtol, atol=STRICT.atol)


@pytest.mark.parametrize("link", _LINKS)
def test_nb_batch_derivatives_reduce_across_padded_batches(link):
    """Batch additivity keeps padded rows neutral and returns only O(p) state."""
    family, beta, X, y, weight, offset, valid, indices, C, dbar, lsbar, params = (
        _inputs(link)
    )
    context = FamilyExecutionContext.from_family(family)
    run = partial(
        nb_batch_statistics_vjp,
        observed_cotangent=C,
        deviance_cotangent=dbar,
        saturated_cotangent=lsbar,
        parameters=params,
        family=family,
        context=context,
        max_y=7,
        integer_counts=True,
    )
    full = run(beta, X, y, weight, offset, valid, indices)
    totals = [np.zeros_like(np.asarray(full.beta)), np.zeros(1), np.zeros_like(beta)]
    good = informative = 0
    for start in (0, 2, 4):
        rows = slice(start, start + 2)
        batch = run(
            beta,
            jnp.vstack((X[rows], jnp.full((1, X.shape[1]), jnp.nan))),
            jnp.concatenate((y[rows], jnp.array([jnp.nan]))),
            jnp.concatenate((weight[rows], jnp.array([jnp.nan]))),
            jnp.concatenate((offset[rows], jnp.array([jnp.inf]))),
            jnp.array([True, True, False]),
            jnp.concatenate((indices[rows], jnp.array([-1]))),
        )
        totals[0] += np.asarray(batch.beta)
        totals[1] += np.asarray(batch.log_theta)
        totals[2] += np.asarray(batch.stationarity_log_theta)
        good += int(batch.source_good_count)
        informative += int(batch.informative_count)
        assert batch.admissible
    for actual, expected in zip(totals, full[:3], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    assert good == full.source_good_count
    assert informative == full.informative_count
    assert all(np.size(leaf) <= max(beta.size, 1) for leaf in jax.tree.leaves(full))


def test_nb_batch_derivatives_fail_closed_on_state_and_metadata():
    family, beta, X, y, weight, offset, valid, indices, C, dbar, lsbar, params = (
        _inputs("log")
    )
    context = FamilyExecutionContext.from_family(family)
    arguments = (beta, X, y, weight, offset, valid, indices, C, dbar, lsbar, params)
    bad = nb_batch_statistics_vjp(
        beta,
        X,
        y.at[0].set(-1),
        weight,
        offset,
        valid,
        indices,
        C,
        dbar,
        lsbar,
        params,
        family,
        context,
        max_y=7,
        integer_counts=True,
    )
    assert not bad.admissible
    with pytest.raises(ValueError, match="count indices"):
        nb_batch_statistics_vjp(
            beta,
            X,
            y,
            weight,
            offset,
            valid,
            indices[:-1],
            C,
            dbar,
            lsbar,
            params,
            family,
            context,
            max_y=7,
            integer_counts=True,
        )
    with pytest.raises(RuntimeError, match="stale"):
        nb_batch_statistics_vjp(
            *arguments,
            family,
            FamilyExecutionContext.from_family(NegativeBinomial(link="identity")),
            max_y=7,
            integer_counts=True,
        )


@pytest.mark.parametrize("link", _LINKS)
def test_fixed_theta_reuses_same_batch_derivatives_and_omits_host_coordinate(link):
    """Family mode does not alter statistics; only host parameter scope does."""
    estimated, *arguments = _inputs(link)
    estimated_result = nb_batch_statistics_vjp(
        *arguments,
        estimated,
        FamilyExecutionContext.from_family(estimated),
        max_y=7,
        integer_counts=True,
    )
    fixed = NegativeBinomial(theta=2.7, fixed=True, link=link)
    fixed_result = nb_batch_statistics_vjp(
        *arguments,
        fixed,
        FamilyExecutionContext.from_family(fixed),
        max_y=7,
        integer_counts=True,
    )
    for actual, expected in zip(fixed_result, estimated_result, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert fixed.n_theta == 0
    assert fixed_result.admissible


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", _LINKS)
def test_fractional_response_derivatives_match_pinned_source(link):
    """Fractional NB uses source pmax deviance and gamma theta derivatives."""
    X = jnp.asarray([[1.0, -0.4], [1.0, -0.1], [1.0, 0.2], [1.0, 0.5]])
    beta = jnp.asarray([1.2, 0.1])
    y = jnp.asarray([0.25, 0.5, 1.5, 3.75])
    weight = jnp.asarray([0.8, 1.1, 0.9, 1.2])
    offset = jnp.asarray([0.02, -0.01, 0.03, -0.02])
    C = jnp.asarray([[0.2, -0.04], [0.07, 0.13]])
    dbar, lsbar = jnp.array(0.6), jnp.array(-0.7)
    family = NegativeBinomial(theta=2.7, fixed=False, link=link)
    result = nb_batch_statistics_vjp(
        beta,
        X,
        y,
        weight,
        offset,
        jnp.ones(4, dtype=bool),
        jnp.zeros(4, dtype=jnp.int64),
        C,
        dbar,
        lsbar,
        FamilyExecutionParameters(jnp.asarray([np.log(2.7)])),
        family,
        FamilyExecutionContext.from_family(family),
        max_y=0,
        integer_counts=False,
    )
    oracle = RBridge(mode="rpy2").nb_batch_derivative_contractions(
        link,
        2.7,
        np.asarray(beta),
        np.asarray(X),
        np.asarray(y),
        np.asarray(weight),
        np.asarray(offset),
        np.asarray(C),
        float(dbar),
        float(lsbar),
    )
    for actual, name in (
        (result.beta, "beta"),
        (result.log_theta, "log.theta"),
        (result.stationarity_log_theta, "stationarity"),
    ):
        np.testing.assert_allclose(
            actual,
            oracle[name],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    assert result.admissible


@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("theta", [0.1, 1e6])
def test_count_tail_derivatives_remain_finite_and_explicit(link, theta):
    """A count tail exercises the bounded prefix derivative in the host API."""
    family, beta, X, _, weight, offset, valid, _, C, dbar, lsbar, _ = _inputs(
        link, theta
    )
    y = jnp.asarray([0.0, 1.0, 7.0, 101.0, 1000.0, 3.0])
    result = nb_batch_statistics_vjp(
        beta,
        X,
        y,
        weight,
        offset,
        valid,
        y.astype(jnp.int64),
        C,
        dbar,
        lsbar,
        FamilyExecutionParameters(jnp.asarray([np.log(theta)])),
        family,
        FamilyExecutionContext.from_family(family),
        max_y=1000,
        integer_counts=True,
    )
    assert result.admissible
    assert all(np.all(np.isfinite(leaf)) for leaf in result[:3])


@pytest.mark.parametrize("link", _LINKS)
def test_nb_batch_partial_theta_matches_independent_refit_finite_difference(link):
    """The half-gradient adjoint reproduces a five-point theta refit FD."""
    X = jnp.asarray(np.column_stack((np.ones(10), np.linspace(-0.5, 0.5, 10))))
    y = jnp.asarray([0.0, 1, 1, 1, 2, 2, 2, 3, 3, 4])
    weight = jnp.asarray(np.linspace(0.8, 1.2, 10))
    offset = jnp.asarray(np.linspace(-0.03, 0.03, 10))
    penalty = jnp.diag(jnp.array([0.03, 0.4]))
    indices = y.astype(jnp.int64)
    family = NegativeBinomial(theta=2.7, fixed=False, link=link)
    context = FamilyExecutionContext.from_family(family)
    deviance_fn = family.deviance_fn(y, weight)

    def state(beta_value, log_theta_value):
        theta_vector = jnp.array([log_theta_value])
        eta = X @ beta_value + offset
        deviance_gradient = jax.grad(deviance_fn, argnums=0)(eta, theta_vector)
        _, deviance_second = jax.jvp(
            lambda value: jax.grad(deviance_fn, argnums=0)(value, theta_vector),
            (eta,),
            (jnp.ones_like(eta),),
        )
        hessian = (X.T * (0.5 * deviance_second)) @ X + penalty
        stationarity = 0.5 * X.T @ deviance_gradient + penalty @ beta_value
        saturated = family.saturated_loglik_theta(
            y,
            weight,
            1.0,
            theta_vector,
            max_y=4,
            count_indices=indices,
            integer_counts=True,
        )
        score = (
            0.5 * deviance_fn(eta, theta_vector)
            + 0.5 * beta_value @ penalty @ beta_value
            - saturated
            + 0.5 * jnp.linalg.slogdet(hessian)[1]
            - 0.5 * jnp.linalg.slogdet(penalty)[1]
        )
        return stationarity, hessian, score

    initial = {
        "log": np.array([0.4, 0.1]),
        "identity": np.array([1.5, 0.1]),
        "sqrt": np.array([1.2, 0.1]),
    }[link]

    def refit(log_theta_value: float, start: np.ndarray):
        solved = root(
            lambda value: np.asarray(state(jnp.asarray(value), log_theta_value)[0]),
            start,
            jac=lambda value: np.asarray(state(jnp.asarray(value), log_theta_value)[1]),
            tol=1e-12,
        )
        assert solved.success
        residual = np.max(
            np.abs(np.asarray(state(jnp.asarray(solved.x), log_theta_value)[0]))
        )
        assert residual < 1e-12
        return solved.x

    log_theta = float(np.log(2.7))
    beta = refit(log_theta, initial)
    _, hessian, _ = state(jnp.asarray(beta), log_theta)
    result = nb_batch_statistics_vjp(
        jnp.asarray(beta),
        X,
        y,
        weight,
        offset,
        jnp.ones(len(y), dtype=bool),
        indices,
        0.5 * jnp.linalg.inv(hessian).T,
        jnp.array(0.5),
        jnp.array(-1.0),
        FamilyExecutionParameters(jnp.array([log_theta])),
        family,
        context,
        max_y=4,
        integer_counts=True,
    )
    # The score's direct beta cotangent also contains d(beta'S beta/2).
    score_beta_cotangent = result.beta + penalty @ jnp.asarray(beta)
    adjoint = jnp.linalg.solve(hessian.T, score_beta_cotangent)
    assembled = result.log_theta[0] - jnp.vdot(adjoint, result.stationarity_log_theta)

    step = 0.003
    scores = []
    for multiplier in (2.0, 1.0, -1.0, -2.0):
        trial_theta = log_theta + multiplier * step
        trial_beta = refit(trial_theta, beta)
        scores.append(float(state(jnp.asarray(trial_beta), trial_theta)[2]))
    finite_difference = (-scores[0] + 8.0 * scores[1] - 8.0 * scores[2] + scores[3]) / (
        12.0 * step
    )
    np.testing.assert_allclose(
        assembled,
        finite_difference,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_nb_batch_theta_parameter_isolation_under_one_compiled_trace():
    """Trial theta is an array leaf; family state is never read by the kernel."""
    family, beta, X, y, weight, offset, valid, indices, C, dbar, lsbar, _ = _inputs(
        "identity"
    )
    context = FamilyExecutionContext.from_family(family)
    run = partial(
        nb_batch_statistics_vjp,
        beta,
        X,
        y,
        weight,
        offset,
        valid,
        indices,
        C,
        dbar,
        lsbar,
        family=family,
        context=context,
        max_y=7,
        integer_counts=True,
    )
    low = run(FamilyExecutionParameters(jnp.log(jnp.array([0.1]))))
    high = run(FamilyExecutionParameters(jnp.log(jnp.array([1e6]))))
    repeated = run(FamilyExecutionParameters(jnp.log(jnp.array([0.1]))))
    np.testing.assert_array_equal(low.beta, repeated.beta)
    np.testing.assert_array_equal(low.log_theta, repeated.log_theta)
    assert not np.allclose(low.log_theta, high.log_theta)


def test_nb_batch_compiled_workspace_is_bounded_by_batch_and_coefficient_size():
    """The executable returns O(p) state and stays within its batch ledger."""
    n, p = 128, 17
    rng = np.random.default_rng(93017)
    X = jnp.asarray(rng.normal(scale=0.05, size=(n, p))).at[:, 0].set(1.0)
    beta = jnp.zeros(p).at[0].set(0.5)
    y = jnp.asarray(np.arange(n) % 8, dtype=jnp.float64)
    weight = jnp.asarray(np.linspace(0.8, 1.2, n))
    family = NegativeBinomial(theta=2.7, fixed=False, link="log")
    arguments = (
        beta,
        X,
        y,
        weight,
        jnp.zeros(n),
        jnp.ones(n, dtype=bool),
        y.astype(jnp.int64),
        jnp.eye(p) * 0.01,
        jnp.array(0.5),
        jnp.array(-1.0),
        FamilyExecutionParameters(jnp.asarray([np.log(2.7)])),
        family,
        FamilyExecutionContext.from_family(family),
    )
    executable = nb_batch_statistics_vjp.lower(
        *arguments, max_y=7, integer_counts=True
    ).compile()
    result = executable(*arguments[:-2])
    assert result.beta.shape == (p,)
    assert result.log_theta.shape == (1,)
    assert result.stationarity_log_theta.shape == (p,)
    assert all(leaf.size <= p for leaf in jax.tree.leaves(result))
    memory = executable.memory_analysis()
    # Ledger: at most eight batch-by-coefficient float64 work arrays plus
    # sixteen coefficient-square arrays. No n-by-p tape escapes the call.
    prospective_bound = 8 * n * p * 8 + 16 * p * p * 8
    assert memory.temp_size_in_bytes <= prospective_bound
