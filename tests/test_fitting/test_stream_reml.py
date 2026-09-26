"""Family-contract batch cotangents for regular exact streamed REML."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
)
from jaxgam.fitting.stream_reml import regular_batch_statistics_vjp
from tests.helpers import _AssertCollector
from tests.test_execution.test_regular_starts import _LINKS, _case
from tests.tolerances import STRICT


@jax.custom_jvp
def _finite_first_derivative(value):
    return value


@_finite_first_derivative.defjvp
def _nonfinite_second_derivative(primals, tangents):
    (value,) = primals
    return value, jnp.full_like(value, jnp.nan) * tangents[0]


class _ObservedDiagnosticGaussian(Gaussian):
    """Finite first derivatives and Fisher score; poisoned unused AD diagnostic."""

    family_name = "observed_diagnostic_gaussian"

    def deviance_derivative_contributions(self, y, mu, wt):
        @jax.custom_jvp
        def primitive(value):
            return wt * (value - y) ** 2

        @primitive.defjvp
        def first_derivative(primals, tangents):
            (value,) = primals
            return (
                primitive(value),
                2 * wt * _finite_first_derivative(value - y) * tangents[0],
            )

        return primitive(mu)


@pytest.mark.parametrize("family_class", [Gaussian, Gamma, Poisson, Binomial])
@pytest.mark.parametrize("link", _LINKS)
def test_regular_batch_vjp_matches_independent_family_derivative(family_class, link):
    """Each cell proves same-state beta/scale cotangents in the smooth interior."""
    family, data, weight, offset, beta = _case(family_class, link)
    y = data.y.to_numpy().copy()
    if family_class is Binomial and link == "log":
        # A new interior derivative proof, separate from the preserved
        # binary-success cancellation fixture and its coefficient-fit gates.
        y = 0.9 * y + 0.03
    X = jnp.asarray(np.column_stack([np.ones(len(y)), data.x]))
    beta = jnp.asarray(beta)
    y = jnp.asarray(family.execution_initial_response(y, weight))
    weight = jnp.asarray(weight)
    offset = jnp.asarray(offset)
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters(jnp.empty(0))
    observed_bar = jnp.array([[0.3, -0.1], [0.2, 0.4]])
    deviance_bar = jnp.array(0.7)
    saturated_bar = jnp.array(-0.6)
    log_phi = jnp.log(0.8)

    def deviance_eta(eta):
        return jnp.sum(
            family.deviance_derivative_contributions(
                y, family.link.inverse(eta), weight
            )
        )

    def independent_score(beta_value):
        eta = X @ beta_value + offset
        mu = family.link.inverse(eta)
        if context.capabilities.fisher_equals_observed_for_score:
            observed_weight = family.working_weights(mu, weight)
        else:
            observed_weight = (
                0.5 * jax.jvp(jax.grad(deviance_eta), (eta,), (jnp.ones_like(eta),))[1]
            )
        observed = (X.T * observed_weight) @ X
        return jnp.sum(observed_bar * observed) + deviance_bar * deviance_eta(eta)

    expected_beta = jax.grad(independent_score)(beta)
    expected_phi = saturated_bar * jax.grad(
        lambda value: family.saturated_loglik(y, weight, jnp.exp(value))
    )(log_phi)
    arguments = (
        beta,
        log_phi,
        X,
        y,
        weight,
        offset,
        jnp.ones(len(y), dtype=bool),
        observed_bar,
        deviance_bar,
        saturated_bar,
        parameters,
        family,
        context,
    )
    compiled = regular_batch_statistics_vjp(*arguments)
    with jax.disable_jit():
        eager = regular_batch_statistics_vjp(*arguments)
    for result in (compiled, eager):
        assert bool(result.admissible)
        assert int(result.informative_count) == len(y) - 1
        np.testing.assert_allclose(
            result.beta, expected_beta, rtol=STRICT.rtol, atol=STRICT.atol
        )
        np.testing.assert_allclose(
            result.log_phi, expected_phi, rtol=STRICT.rtol, atol=STRICT.atol
        )


def test_regular_batch_vjp_padding_and_invalid_real_rows():
    """Unused nonfinite tails are neutral; identical real rows are rejected."""
    family = Gamma("log")
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters(jnp.empty(0))

    def call(valid):
        return regular_batch_statistics_vjp(
            jnp.array([0.1, 0.2]),
            jnp.log(0.7),
            jnp.array([[1.0, 0.2], [jnp.nan, jnp.nan]]),
            jnp.array([1.1, jnp.nan]),
            jnp.array([1.0, jnp.nan]),
            jnp.array([0.0, jnp.nan]),
            jnp.asarray(valid),
            jnp.eye(2),
            jnp.array(1.0),
            jnp.array(-1.0),
            parameters,
            family,
            context,
        )

    padded = call([True, False])
    assert bool(padded.admissible)
    assert np.all(np.isfinite(padded.beta))
    assert int(padded.informative_count) == 1
    assert not bool(call([True, True]).admissible)
    neutral = call([False, False])
    assert bool(neutral.admissible)
    assert int(neutral.informative_count) == 0
    np.testing.assert_array_equal(neutral.beta, np.zeros(2))
    assert float(neutral.log_phi) == 0.0


def test_regular_batch_vjp_preserves_unresolved_binomial_log_boundary():
    """A finite beta cotangent alone must not admit an unresolved source system."""
    family = Binomial("log")
    context = FamilyExecutionContext.from_family(family)
    result = regular_batch_statistics_vjp(
        jnp.array([-0.65]),
        jnp.array(0.0),
        jnp.ones((1, 1)),
        jnp.ones(1),
        jnp.ones(1),
        jnp.zeros(1),
        jnp.ones(1, dtype=bool),
        jnp.ones((1, 1)),
        jnp.array(0.5),
        jnp.array(-1.0),
        FamilyExecutionParameters(jnp.empty(0)),
        family,
        context,
    )
    assert not bool(result.admissible)
    assert bool(result.structural_admissible)
    assert bool(result.alpha_unresolved)


def test_canonical_score_vjp_ignores_unused_observed_diagnostic():
    """Admission follows the Fisher score and required finite first derivatives."""
    family = _ObservedDiagnosticGaussian()
    context = FamilyExecutionContext.from_family(family)
    result = regular_batch_statistics_vjp(
        jnp.array([0.5]),
        jnp.log(0.7),
        jnp.ones((1, 1)),
        jnp.array([0.3]),
        jnp.ones(1),
        jnp.zeros(1),
        jnp.ones(1, dtype=bool),
        jnp.ones((1, 1)),
        jnp.array(0.5),
        jnp.array(-1.0),
        FamilyExecutionParameters(jnp.empty(0)),
        family,
        context,
    )
    assert bool(result.admissible)
    np.testing.assert_allclose(result.beta, [0.2], rtol=STRICT.rtol, atol=STRICT.atol)
    np.testing.assert_allclose(result.log_phi, 0.5, rtol=STRICT.rtol, atol=STRICT.atol)


def test_neutral_batch_cannot_admit_a_nonfinite_coefficient_state():
    family = Gaussian()
    context = FamilyExecutionContext.from_family(family)
    result = regular_batch_statistics_vjp(
        jnp.array([jnp.nan]),
        jnp.array(0.0),
        jnp.zeros((1, 1)),
        jnp.zeros(1),
        jnp.zeros(1),
        jnp.zeros(1),
        jnp.zeros(1, dtype=bool),
        jnp.zeros((1, 1)),
        jnp.array(0.0),
        jnp.array(0.0),
        FamilyExecutionParameters(jnp.empty(0)),
        family,
        context,
    )
    assert not bool(result.admissible)


def test_regular_batch_vjp_matches_pinned_gdi_derivative_ratios(r_bridge):
    """Actual fix.family derivatives reproduce gdi1's first weight/deviance VJP."""
    assert r_bridge.check_versions()[0]
    collector = _AssertCollector()
    for family_class in (Gaussian, Gamma, Poisson, Binomial):
        for link in _LINKS:
            family, data, weight, offset, beta = _case(family_class, link)
            y = data.y.to_numpy().copy()
            if family_class is Binomial and link == "log":
                y = 0.9 * y + 0.03
            y = np.asarray(family.execution_initial_response(y, weight))
            X = np.column_stack([np.ones(len(y)), data.x])
            C = np.array([[0.3, -0.1], [0.2, 0.4]])
            context = FamilyExecutionContext.from_family(family)
            expected = r_bridge.source_regular_gdi_ratio(
                X,
                y,
                weight,
                offset,
                beta,
                C,
                family_name=family.family_name,
                link=link,
                fisher_equals_observed=context.capabilities.fisher_equals_observed_for_score,
            )
            result = regular_batch_statistics_vjp(
                jnp.asarray(beta),
                jnp.log(0.8),
                jnp.asarray(X),
                jnp.asarray(y),
                jnp.asarray(weight),
                jnp.asarray(offset),
                jnp.ones(len(y), dtype=bool),
                jnp.asarray(C),
                jnp.array(0.7),
                jnp.array(-0.6),
                FamilyExecutionParameters(jnp.empty(0)),
                family,
                context,
            )
            collector.check(
                f"{family.family_name}/{link} beta",
                lambda result=result, expected=expected: np.testing.assert_allclose(
                    result.beta,
                    expected[:2],
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
            collector.check(
                f"{family.family_name}/{link} log_phi",
                lambda result=result, expected=expected: np.testing.assert_allclose(
                    result.log_phi,
                    expected[2],
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
    collector.raise_if_any("Pinned gdi1 regular batch derivatives")
