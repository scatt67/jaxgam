"""JIT bgam.fitd working-statistic kernels on the compact discrete design."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.discrete_ops import to_device_discrete_design
from jaxgam.fitting.discrete_working import (
    discrete_initial_working_batch,
    discrete_working_batch,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.discrete_design import (
    DiscreteDesignUnavailable,
    build_discrete_design_from_setup,
)
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector, check_that
from tests.tolerances import STRICT

_FAMILIES = [
    Poisson(),
    Poisson(link="sqrt"),
    Poisson(link="identity"),
    Binomial(),
    Binomial(link="probit"),
    Binomial(link="cloglog"),
    Binomial(link="log"),
    Gamma(),
    Gamma(link="log"),
    Gamma(link="identity"),
]


def _design():
    n = 45
    x = np.resize(np.array((0.1, 0.3, 0.6, 0.9)), n)
    z = np.resize(np.array((0.2, 0.5, 0.8)), n)
    group = pd.Categorical(np.resize(np.array(("a", "b", "c")), n))
    data = pd.DataFrame({"x": x, "z": z, "group": group, "y": np.sin(x) + z})
    setup = ModelSetup.build(
        parse_formula("y ~ group + s(x, bs='cr', k=4) + te(x, z, bs='cr', k=3)"), data
    )
    descriptor = build_discrete_design_from_setup(
        setup, DataFrameRowSource(data, response="y")
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    return setup.X, descriptor


def _selectors(descriptor, positions):
    return tuple(
        tuple(jnp.asarray(value) for value in block)
        for block in descriptor.batch_indices(positions)
    )


def _inputs(family, n):
    """A response in the family domain and a predictor inside its link domain."""
    position = np.arange(n, dtype=float)
    weight = np.linspace(0.3, 1.7, n)
    weight[::11] = 0.0
    offset = np.linspace(-0.05, 0.08, n)
    if isinstance(family, Binomial):
        mean = 0.2 + 0.6 * (0.5 + 0.4 * np.sin(position))
        response = np.round(4 * mean) / 4
    elif isinstance(family, Poisson):
        mean = 1.5 + np.cos(position)
        response = np.floor(mean + 0.7 * np.sin(3 * position) + 0.5)
    else:
        mean = 1.2 + 0.5 * np.cos(position)
        response = mean * (1.0 + 0.3 * np.sin(2 * position))
    return mean, np.maximum(response, 0.0), weight, offset


def _source_working(family, eta, y, weight, offset):
    """bgam.fitd lines 637-645 in NumPy, including mu.eta's 0.1 replacement."""
    mu = np.asarray(family.link.inverse(eta))
    mu_eta = np.asarray(family.link.mu_eta(eta))
    mu_eta = np.where(mu_eta != 0.0, mu_eta, 0.1)
    z = (eta - offset) + (y - mu) / mu_eta
    w = (weight * mu_eta**2) / np.asarray(family.variance(mu))
    deviance = float(np.sum(family.deviance_contributions(y, mu, weight)))
    return deviance, w, z


@pytest.mark.parametrize("family", _FAMILIES, ids=repr)
def test_working_batches_compile_and_match_explicit_design(family) -> None:
    """Initial and coefficient batches equal explicit X'WX/X'Wz/sum(w z^2)."""
    X, descriptor = _design()
    device = to_device_discrete_design(descriptor)
    n, p = X.shape
    mean, y, weight, offset = _inputs(family, n)
    eta_start = np.asarray(family.link.link(mean))
    beta = np.linalg.lstsq(X, eta_start - offset, rcond=None)[0]
    totals = {"initial": [0.0, np.zeros((p, p)), np.zeros(p), 0.0]}
    totals["coefficients"] = [0.0, np.zeros((p, p)), np.zeros(p), 0.0]
    deviance_only = 0.0
    for start in range(0, n, 7):
        rows = np.arange(start, min(n, start + 7))
        selectors = _selectors(descriptor, rows)
        arguments = (y[rows], weight[rows], offset[rows])
        initial = discrete_initial_working_batch(
            device, selectors, eta_start[rows], *arguments, family=family
        )
        coefficient = discrete_working_batch(
            device, selectors, jnp.asarray(beta), *arguments, family=family
        )
        deviance_only += float(
            np.sum(
                discrete_working_batch(
                    device,
                    selectors,
                    jnp.asarray(beta),
                    *arguments,
                    family=family,
                    statistics=False,
                ).deviance_rows
            )
        )
        for name, result in (("initial", initial), ("coefficients", coefficient)):
            total = totals[name]
            total[0] += float(np.sum(result.deviance_rows))
            total[1] += np.asarray(result.gram)
            total[2] += np.asarray(result.rhs)
            total[3] += float(result.response_square_sum)
    collector = _AssertCollector()
    collector.check(
        "jit compiled",
        lambda: check_that(
            discrete_working_batch._cache_size() > 0
            and discrete_initial_working_batch._cache_size() > 0,
            "working kernels did not compile",
        ),
    )
    for name, eta in (("initial", eta_start), ("coefficients", X @ beta + offset)):
        deviance, w, z = _source_working(family, eta, y, weight, offset)
        expected = (deviance, X.T @ (w[:, None] * X), X.T @ (w * z), np.sum(w * z**2))
        for label, observed, target in zip(
            ("deviance", "gram", "rhs", "y.norm2"), totals[name], expected, strict=True
        ):
            collector.check(
                f"{name} {label}",
                lambda observed=observed, target=target: np.testing.assert_allclose(
                    observed, target, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
    collector.check(
        "deviance-only batches",
        lambda: np.testing.assert_allclose(
            deviance_only,
            totals["coefficients"][0],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any(repr(family))


def _domain_cases():
    """Per family: link, a predictor with rows outside the family domain, y."""
    eta = np.array([0.4, 0.9, -0.3, 1.2, 0.0, -0.6, 0.7])
    return [
        # identity: negative means at a zero and at a positive count.
        (
            "poisson",
            "identity",
            Poisson(link="identity"),
            eta,
            np.array([1.0, 0, 0, 2, 0, 3, 1]),
        ),
        # identity/inverse: mu < 0, and eta = 0 (mu = Inf) for the inverse.
        (
            "gamma",
            "identity",
            Gamma(link="identity"),
            eta,
            np.array([1.5, 0.4, 0.8, 1.1, 0.6, 0.9, 2.0]),
        ),
        (
            "gamma",
            "inverse",
            Gamma(),
            eta,
            np.array([1.5, 0.4, 0.8, 1.1, 0.6, 0.9, 2.0]),
        ),
        # log link: exp(eta) > 1 is outside (0, 1); y at 0, 1 and inside.
        (
            "binomial",
            "log",
            Binomial(link="log"),
            eta,
            np.array([0.0, 1.0, 0.25, 0.5, 1.0, 0.0, 0.75]),
        ),
    ]


def test_deviance_rows_follow_pinned_dev_resids_outside_the_domain(r_bridge) -> None:
    """bgam.fitd sums raw dev.resids with no validmu/valideta mask.

    Each kernel row equals pinned R's ``family$dev.resids`` at the same mean,
    including R's ``NaN`` (a log of a negative ratio, ``Inf/Inf``) and its
    finite values outside the domain (``2*mu*wt`` for a negative Poisson mean
    at a zero count).
    """
    _, descriptor = _design()
    device = to_device_discrete_design(descriptor)
    rows = np.arange(7)
    selectors = _selectors(descriptor, rows)
    weight = np.linspace(0.5, 1.5, 7)
    offset = np.zeros(7)
    collector = _AssertCollector()
    for name, link, family, eta, y in _domain_cases():
        result = discrete_initial_working_batch(
            device, selectors, eta, y, weight, offset, family=family
        )
        mu = np.asarray(family.link.inverse(eta))
        expected = r_bridge.family_dev_resids(name, y, mu, weight, link=link)
        observed = np.asarray(result.deviance_rows)
        collector.check(
            f"{name}/{link} rows",
            lambda observed=observed, expected=expected: np.testing.assert_allclose(
                observed, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        collector.check(
            f"{name}/{link} exercises the domain boundary",
            lambda family=family, mu=mu, eta=eta: check_that(
                not np.all(np.asarray(family.valid_mu(mu)))
                or not np.all(np.asarray(family.valid_eta(eta))),
                "case must leave the family domain",
            ),
        )
    collector.raise_if_any("pinned dev.resids outside the domain")


def test_out_of_domain_rows_keep_r_arithmetic() -> None:
    """Without R: NaN only where pinned dev.resids is undefined.

    A negative Poisson identity mean at a zero count is finite (``2*mu*wt``),
    a negative mean at a positive count is ``NaN``; for Gamma a negative mean
    and a zero (identity) or infinite (inverse, ``eta = 0``) mean are
    ``NaN``.
    """
    _, descriptor = _design()
    device = to_device_discrete_design(descriptor)
    rows = np.arange(7)
    selectors = _selectors(descriptor, rows)
    weight = np.linspace(0.5, 1.5, 7)
    offset = np.zeros(7)
    expected_nan = {
        "poisson/identity": [False, False, False, False, False, True, False],
        "gamma/identity": [False, False, True, False, True, True, False],
        "gamma/inverse": [False, False, True, False, True, True, False],
    }
    for name, link, family, eta, y in _domain_cases()[:3]:
        result = discrete_initial_working_batch(
            device, selectors, eta, y, weight, offset, family=family
        )
        rows_nan = np.isnan(np.asarray(result.deviance_rows))
        np.testing.assert_array_equal(rows_nan, expected_nan[f"{name}/{link}"])
    poisson = discrete_initial_working_batch(
        device,
        selectors,
        _domain_cases()[0][3],
        _domain_cases()[0][4],
        weight,
        offset,
        family=Poisson(link="identity"),
    )
    np.testing.assert_allclose(
        np.asarray(poisson.deviance_rows)[2], 2 * -0.3 * weight[2], rtol=STRICT.rtol
    )


def test_zero_mu_eta_uses_bgam_fitd_replacement() -> None:
    """sqrt link at eta=0 has mu.eta=0; w and z use 0.1, as bgam.fitd does.

    A unit variance keeps the replaced row finite, so the replacement itself
    (rather than a zero variance) determines the row's weight.
    """
    X, descriptor = _design()
    device = to_device_discrete_design(descriptor)
    rows = np.arange(7)
    selectors = _selectors(descriptor, rows)
    family = Gaussian(link="sqrt")
    eta = np.linspace(0.0, 1.2, 7)
    y = np.array([0.0, 1, 2, 0, 1, 3, 1])
    weight = np.linspace(0.5, 1.5, 7)
    offset = np.zeros(7)
    result = discrete_initial_working_batch(
        device, selectors, eta, y, weight, offset, family=family
    )
    _, w, z = _source_working(family, eta, y, weight, offset)
    np.testing.assert_allclose(w[0], weight[0] * 0.01, rtol=STRICT.rtol)
    np.testing.assert_allclose(z[0], (y[0] - 0.0) / 0.1, rtol=STRICT.rtol)
    x_rows = X[rows]
    for observed, target in (
        (result.rhs, x_rows.T @ (w * z)),
        (result.gram, x_rows.T @ (w[:, None] * x_rows)),
    ):
        np.testing.assert_allclose(
            np.asarray(observed), target, rtol=STRICT.rtol, atol=STRICT.atol
        )


def test_deviance_only_batches_are_empty_and_shapes_are_checked() -> None:
    X, descriptor = _design()
    device = to_device_discrete_design(descriptor)
    rows = np.arange(5)
    selectors = _selectors(descriptor, rows)
    beta = jnp.zeros(X.shape[1])
    result = discrete_working_batch(
        device,
        selectors,
        beta,
        np.ones(5),
        np.ones(5),
        np.zeros(5),
        family=Gaussian(link="log"),
        statistics=False,
    )
    assert result.gram.shape == (0, 0)
    assert result.rhs.shape == (0,)
    with pytest.raises(ValueError, match="share one shape"):
        discrete_working_batch(
            device,
            selectors,
            beta,
            np.ones(5),
            np.ones(4),
            np.zeros(5),
            family=Poisson(),
        )


def test_working_batch_runs_under_an_outer_jit() -> None:
    """The kernels are pure and compose inside another compiled function."""
    X, descriptor = _design()
    device = to_device_discrete_design(descriptor)
    rows = np.arange(7)
    selectors = _selectors(descriptor, rows)
    family = Binomial()

    @jax.jit
    def total(beta):
        result = discrete_working_batch(
            device,
            selectors,
            beta,
            jnp.full(7, 0.5),
            jnp.ones(7),
            jnp.zeros(7),
            family=family,
        )
        return jnp.sum(result.deviance_rows) + jnp.trace(result.gram)

    value = total(jnp.zeros(X.shape[1]))
    eta = np.zeros(7)
    deviance, w, _ = _source_working(family, eta, np.full(7, 0.5), np.ones(7), eta)
    np.testing.assert_allclose(
        float(value),
        deviance + np.trace(X[rows].T @ (w[:, None] * X[rows])),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
