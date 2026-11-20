"""Eta-level observed row contracts for operator fitting."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
)
from jaxgam.fitting.operator.family import operator_family_rows
from jaxgam.links.links import IdentityLink, InverseLink, LogLink, SqrtLink
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


def _rows(family, eta, y, weights=None, offset=None, valid=None, parameters=None):
    eta = jnp.asarray(eta, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    n = eta.shape[0]
    context = FamilyExecutionContext.from_family(family)
    if parameters is None:
        parameters = FamilyExecutionParameters.from_snapshot(
            family.execution_parameter_snapshot()
        )
    return operator_family_rows(
        eta,
        y,
        jnp.ones(n) if weights is None else jnp.asarray(weights),
        jnp.zeros(n) if offset is None else jnp.asarray(offset),
        jnp.ones(n, dtype=bool) if valid is None else jnp.asarray(valid),
        parameters,
        family,
        context,
    )


@pytest.mark.parametrize(
    "family",
    [
        Gaussian(),
        Poisson(),
        Binomial(),
        Gamma(),
        Gamma(link=LogLink()),
        NegativeBinomial(theta=2.0, fixed=True),
    ],
)
def test_observed_rows_match_analytic_half_deviance_derivatives_and_wz(family):
    eta = np.array([0.4, 0.7, 1.1])
    prior = np.array([0.6, 0.0, 1.4])
    offset = np.array([-0.1, 0.2, 0.3])
    if isinstance(family, Binomial):
        y = np.array([0.0, 0.2, 1.0])
    else:
        y = np.array([1.0, 2.0, 4.0])
    actual = _rows(family, eta, y, prior, offset)
    mu = np.asarray(family.link.inverse(eta))
    if isinstance(family, Gaussian):
        gradient = prior * (mu - y)
        curvature = prior
    elif isinstance(family, Poisson):
        gradient = prior * (mu - y)
        curvature = prior * mu
    elif isinstance(family, Binomial):
        gradient = prior * (mu - y)
        curvature = prior * mu * (1 - mu)
    elif isinstance(family, Gamma):
        if isinstance(family.link, LogLink):
            gradient = prior * (1 - y / mu)
            curvature = prior * y / mu
        else:
            gradient = prior * (y - mu)
            curvature = prior * mu**2
    else:
        gradient = prior * 2 * (mu - y) / (mu + 2)
        curvature = prior * 2 * mu * (y + 2) / (mu + 2) ** 2
    collector = _AssertCollector()
    for name, value, expected in (
        ("half Deta", actual.half_gradient, gradient),
        ("half Deta2", actual.observed_weight, curvature),
        ("Wz", actual.weighted_response, curvature * (eta - offset) - gradient),
        (
            "direct deviance",
            actual.deviance,
            np.sum(family.deviance_contributions(y, mu, prior)),
        ),
    ):
        collector.check(
            name,
            lambda value=value, expected=expected: np.testing.assert_allclose(
                value, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    assert actual.input_ok
    assert actual.domain_ok
    assert actual.finite_system
    collector.raise_if_any("eta-level observed family system")


def test_actual_negative_nb_identity_curvature_is_not_fisher_replaced():
    family = NegativeBinomial(theta=2.0, fixed=True, link=IdentityLink())
    actual = _rows(family, [3.0, 4.0], [0.0, 2.0])
    expected = np.array([-2 / 25, 2 / 16 - 4 / 36])
    np.testing.assert_allclose(
        actual.observed_weight, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert actual.observed_weight[0] < 0
    assert actual.finite_system


def test_poisoned_unused_padding_and_zero_weight_rows_are_finite_before_ad():
    family = NegativeBinomial(theta=2.0, fixed=True)
    actual = _rows(
        family,
        [0.7, 0.4, np.nan],
        [2.0, 0.0, np.nan],
        [1.0, 0.0, np.nan],
        [0.1, 0.0, np.nan],
        [True, True, False],
    )
    assert actual.finite_system
    for array in (
        actual.half_gradient,
        actual.observed_weight,
        actual.weighted_response,
    ):
        assert np.all(np.isfinite(array))
        np.testing.assert_array_equal(np.asarray(array)[1:], 0)
    assert np.isfinite(actual.deviance)


@pytest.mark.parametrize(
    ("field", "value"),
    [("y", -1.0), ("eta", np.inf), ("weights", -1.0), ("offset", np.nan)],
)
def test_invalid_real_input_fails_without_poisoning_operational_arrays(field, value):
    values = {"eta": [0.5], "y": [2.0], "weights": [1.0], "offset": [0.0]}
    values[field] = [value]
    actual = _rows(Poisson(), **values)
    assert not actual.input_ok
    assert not actual.domain_ok
    assert not actual.finite_system
    assert np.all(np.isfinite(actual.observed_weight))


def test_invalid_domain_theta_capability_context_and_shapes_fail_explicitly():
    family = NegativeBinomial(theta=2.0, fixed=True, link=IdentityLink())
    actual = _rows(family, [-0.5], [2.0])
    assert actual.input_ok
    assert not actual.domain_ok
    assert not actual.finite_system
    parameters = FamilyExecutionParameters(log_theta=jnp.array([jnp.nan]))
    actual = _rows(family, [1.0], [2.0], parameters=parameters)
    assert not actual.input_ok
    assert np.all(np.isfinite(actual.observed_weight))
    context = FamilyExecutionContext.from_family(family)
    missing = replace(
        context, capabilities=replace(context.capabilities, observed_information=False)
    )
    with pytest.raises(RuntimeError, match="stale"):
        operator_family_rows(
            jnp.ones(1),
            jnp.ones(1),
            jnp.ones(1),
            jnp.zeros(1),
            jnp.ones(1, dtype=bool),
            parameters,
            family,
            missing,
        )

    class Unsupported(Gaussian):
        def execution_capabilities(self):
            return replace(super().execution_capabilities(), observed_information=False)

    with pytest.raises(NotImplementedError, match="observed_information"):
        _rows(Unsupported(), [1.0], [2.0])
    with pytest.raises(ValueError, match="matching one-dimensional"):
        _rows(Gaussian(), [1.0, 2.0], [1.0])


def test_eta_kernel_compiles_with_only_bounded_row_arguments():
    family = Gaussian()
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    B = 4096
    compiled = operator_family_rows.lower(
        jnp.ones(B),
        jnp.ones(B),
        jnp.ones(B),
        jnp.zeros(B),
        jnp.ones(B, dtype=bool),
        parameters,
        family,
        context,
    ).compile()
    measured = compiled.memory_analysis()
    assert (
        measured.argument_size_in_bytes
        + measured.output_size_in_bytes
        + measured.temp_size_in_bytes
        < 256 * B
    )
    actual = compiled(
        jnp.ones(B),
        jnp.ones(B),
        jnp.ones(B),
        jnp.zeros(B),
        jnp.ones(B, dtype=bool),
        parameters,
    )
    assert actual.finite_system
    assert actual.deviance == 0


@pytest.mark.parametrize(
    ("link_name", "link"),
    [("identity", IdentityLink()), ("log", LogLink()), ("sqrt", SqrtLink())],
)
def test_fixed_theta_observed_rows_match_pinned_ddeta_and_wz(r_bridge, link_name, link):
    family = NegativeBinomial(theta=2.7, fixed=True, link=link)
    mu = np.array([0.7, 1.2, 2.0, 4.1])
    y = np.array([0.0, 2.0, 1.0, 5.0])
    weights = np.array([0.6, 0.0, 1.1, 1.4])
    offset = np.array([0.1, -0.2, 0.05, 0.3])
    eta = np.asarray(link.link(mu))
    actual = _rows(family, eta, y, weights, offset)
    reference = r_bridge.efs_nb_working_factors(link_name, y, mu, weights, 2.7, offset)
    collector = _AssertCollector()
    for name, value in (
        ("weight", actual.observed_weight),
        ("weighted_response", actual.weighted_response),
    ):
        collector.check(
            name,
            lambda name=name, value=value: np.testing.assert_allclose(
                value, reference[name], rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    assert actual.finite_system
    collector.raise_if_any("pinned NB observed row system")


@pytest.mark.parametrize(
    ("family", "response", "constant"),
    [
        (Gaussian(), 1.5, True),
        (Gaussian(link=LogLink()), 1.5, False),
        (Gaussian(link=InverseLink()), 1.5, False),
        (Poisson(), 2.0, False),
        (Binomial(), 0.5, False),
        (Gamma(), 1.5, False),
        (Gamma(link=LogLink()), 1.5, False),
        (NegativeBinomial(theta=2.0, fixed=True), 2.0, False),
    ],
)
def test_single_step_capability_matches_actual_row_curvature(
    family, response, constant
):
    """One-step convergence is certified only where curvature ignores beta."""
    capability = family.execution_capabilities()
    assert capability.single_step_pirls is constant
    first = float(_rows(family, [0.4], [response]).observed_weight[0])
    second = float(_rows(family, [0.9], [response]).observed_weight[0])
    if constant:
        np.testing.assert_allclose(first, second, rtol=STRICT.rtol, atol=STRICT.atol)
    else:
        assert not np.isclose(first, second, rtol=STRICT.rtol, atol=STRICT.atol)
    if isinstance(family, (Poisson, Binomial)):
        assert capability.fisher_equals_observed_for_score
        assert not capability.single_step_pirls
