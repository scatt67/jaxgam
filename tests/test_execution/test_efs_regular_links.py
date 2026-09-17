"""Pinned EFS parity gates for advertised regular-family links."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from jaxgam.execution.efs import (
    dense_efs_known_scale,
    dense_efs_unknown_scale,
    efs_initial_log_lambda,
    efs_initial_log_scale,
)
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import FittingData
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector, check_that, r_available
from tests.r_bridge import RBridge, RBridgeError
from tests.tolerances import STRICT


def _advertised_link_cases() -> dict[str, tuple[object, np.ndarray]]:
    """Return deterministic responses inside each advertised link's domain."""
    rng = np.random.default_rng(7712)
    x = np.linspace(-1.0, 1.0, 120)
    wave = np.sin(5.0 * x) + 0.3 * np.cos(8.0 * x)
    cases: dict[str, tuple[object, np.ndarray]] = {}

    mu = 1.0 / (1.0 + 0.22 * wave)
    cases["gaussian_inverse"] = (
        Gaussian("inverse"),
        np.maximum(mu + rng.normal(0.0, 0.06, len(x)), 0.05),
    )
    mu = 0.9 + 0.25 * wave
    cases["gamma_identity"] = (Gamma("identity"), rng.gamma(12.0, mu / 12.0))
    mu = 1.6 + 0.75 * wave
    cases["poisson_identity"] = (Poisson("identity"), rng.poisson(mu))
    mu = (1.25 + 0.3 * wave) ** 2
    cases["poisson_sqrt"] = (Poisson("sqrt"), rng.poisson(mu))
    eta = 0.1 + wave
    cases["binomial_probit"] = (Binomial("probit"), rng.binomial(1, ndtr(eta)))
    probability = 1.0 - np.exp(-np.exp(-0.2 + 0.7 * wave))
    cases["binomial_cloglog"] = (
        Binomial("cloglog"),
        rng.binomial(1, probability),
    )
    probability = np.exp(-1.25 + 0.35 * wave)
    cases["binomial_log"] = (Binomial("log"), rng.binomial(1, probability))

    # Generate this after the established seven-case sequence so adding its
    # regression gate cannot perturb the already reviewed fixtures above.
    mu = np.exp(0.05 + 0.18 * wave)
    cases["gaussian_log"] = (
        Gaussian("log"),
        np.maximum(mu + rng.normal(0.0, 0.05, len(x)), 0.05),
    )
    return cases


def _constructor_extension_cases() -> dict[str, tuple[object, np.ndarray]]:
    """Return valid-domain fixtures for R-accepted nonadvertised links."""
    rng = np.random.default_rng(91827)
    x = np.linspace(-1.0, 1.0, 120)
    wave = np.sin(4.1 * x) + 0.25 * np.cos(7.2 * x)
    eta_by_link = {
        "logit": -0.3 + 0.22 * wave,
        "probit": -0.2 + 0.18 * wave,
        "cloglog": -0.6 + 0.18 * wave,
        "identity": 0.55 + 0.1 * wave,
        "inverse": 2.0 + 0.18 * wave,
        "sqrt": 0.68 + 0.06 * wave,
        "inverse_squared": 2.0 + 0.18 * wave,
    }
    cases: dict[str, tuple[object, np.ndarray]] = {}

    for link in ("logit", "probit", "cloglog", "sqrt", "inverse_squared"):
        family = Gaussian(link)
        mean = np.asarray(family.link.inverse(eta_by_link[link]))
        response = mean + rng.normal(0.0, 0.025, len(x))
        if link in {"logit", "probit", "cloglog"}:
            response = np.clip(response, 0.02, 0.95)
        elif link in {"sqrt", "inverse_squared"}:
            response = np.clip(response, 0.03, None)
        cases[f"gaussian_{link}"] = (family, response)

    for link in ("identity", "inverse", "sqrt", "inverse_squared"):
        family = Binomial(link)
        mean = np.asarray(family.link.inverse(eta_by_link[link]))
        cases[f"binomial_{link}"] = (family, rng.binomial(1, mean))

    # Advance through the five Poisson constructor fixtures used by the
    # companion compatibility-boundary probe. This keeps the Gamma fixtures
    # stable if those Poisson cells later gain supported finite-score inputs.
    for link in ("logit", "inverse", "probit", "cloglog", "inverse_squared"):
        family = Poisson(link)
        mean = np.asarray(family.link.inverse(eta_by_link[link]))
        if link in {"logit", "probit", "cloglog"}:
            np.clip(mean + rng.normal(0.0, 0.025, len(x)), 0.02, 0.82)
        else:
            np.clip(mean + rng.normal(0.0, 0.04, len(x)), 0.02, None)

    for link in ("logit", "probit", "cloglog", "sqrt", "inverse_squared"):
        family = Gamma(link)
        mean = np.asarray(family.link.inverse(eta_by_link[link]))
        response = mean * np.exp(rng.normal(0.0, 0.06, len(x)))
        upper = 0.95 if link in {"logit", "probit", "cloglog"} else None
        cases[f"gamma_{link}"] = (family, np.clip(response, 0.03, upper))
    return cases


def _assert_regular_link_parity(
    case_name: str, family: object, response: np.ndarray
) -> None:
    """Compare one selected dense EFS fit with its pinned mgcv fit."""
    x = np.linspace(-1.0, 1.0, len(response))
    data = pd.DataFrame({"x": x, "y": response})
    formula = "y ~ s(x, bs='cr', k=7)"
    setup = ModelSetup.build(parse_formula(formula), data)
    fitting_data = FittingData.from_setup(setup, family)
    rho = efs_initial_log_lambda(setup, family)
    oracle_arguments: dict[str, object] = {
        "initial_smoothing": np.exp(np.asarray(rho)),
        "null_coef": True,
    }
    if family.scale_known:
        python_fit = dense_efs_known_scale(fitting_data, initial_log_lambda=rho)
        oracle_arguments["scale"] = 1.0
    else:
        log_scale = efs_initial_log_scale(setup, family)
        python_fit = dense_efs_unknown_scale(
            fitting_data,
            initial_log_lambda=rho,
            initial_log_scale=log_scale,
        )
        oracle_arguments["initial_scale"] = float(np.exp(log_scale))
    r_fit = RBridge(mode="subprocess").fit_efs(
        formula,
        data,
        case_name,
        **oracle_arguments,
    )

    collector = _AssertCollector()
    collector.check(
        "converged",
        lambda: check_that(python_fit.converged, python_fit.convergence_info),
    )
    collector.check(
        "outer iterations",
        lambda: np.testing.assert_equal(python_fit.n_iter, r_fit["outer_iterations"]),
    )
    collector.check(
        "valid selected mean",
        lambda: check_that(
            bool(np.all(np.asarray(family.valid_mu(python_fit.pirls_result.mu)))),
            "selected mean is outside the family domain",
        ),
    )
    collector.check(
        "fitted values",
        lambda: np.testing.assert_allclose(
            python_fit.pirls_result.mu,
            r_fit["fitted_values"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "deviance",
        lambda: np.testing.assert_allclose(
            python_fit.pirls_result.deviance,
            r_fit["deviance"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "criterion",
        lambda: np.testing.assert_allclose(
            python_fit.score,
            r_fit["reml_score"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "smoothing parameters",
        lambda: np.testing.assert_allclose(
            python_fit.smoothing_params,
            r_fit["smoothing_params"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "reported scale",
        lambda: np.testing.assert_allclose(
            python_fit.scale,
            r_fit["scale"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any(f"{case_name} EFS parity")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    "case_name",
    [
        "gaussian_log",
        "gaussian_inverse",
        "gamma_identity",
        "poisson_identity",
        "poisson_sqrt",
        "binomial_probit",
        "binomial_cloglog",
        "binomial_log",
    ],
)
def test_advertised_regular_link_efs_matches_pinned_r_strict(case_name: str) -> None:
    """Selected EFS fit agrees with pinned mgcv for every added advertised link."""
    family, response = _advertised_link_cases()[case_name]
    _assert_regular_link_parity(case_name, family, response)


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize("case_name", sorted(_constructor_extension_cases()))
def test_constructor_extension_regular_link_efs_matches_pinned_r_strict(
    case_name: str,
) -> None:
    """R-accepted regular link extensions have explicit valid-domain parity."""
    family, response = _constructor_extension_cases()[case_name]
    _assert_regular_link_parity(case_name, family, response)


def _integer_poisson_explicit_start_case(
    link: str,
) -> tuple[pd.DataFrame, Poisson, ModelSetup, FittingData, jnp.ndarray]:
    """Build a count response plus a coefficient-representable valid start."""
    rng = np.random.default_rng(77119)
    x = np.linspace(-1.0, 1.0, 140)
    wave = np.sin(3.4 * x) + 0.2 * np.cos(6.1 * x)
    response = rng.binomial(1, 0.38 + 0.09 * wave).astype(float)
    data = pd.DataFrame({"x": x, "y": response})
    family = Poisson(link)
    setup = ModelSetup.build(
        parse_formula("y ~ s(x, bs='cr', k=7)"),
        data,
    )
    fitting_data = FittingData.from_setup(setup, family)
    target_mean = 0.42 + 0.06 * wave
    target_eta = np.asarray(family.link.link(target_mean))
    beta = jnp.asarray(
        np.linalg.lstsq(np.asarray(fitting_data.X), target_eta, rcond=None)[0]
    )
    assert np.all(
        np.asarray(family.valid_mu(family.link.inverse(fitting_data.X @ beta)))
    )
    return data, family, setup, fitting_data, beta


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    "link", ["logit", "inverse", "probit", "cloglog", "inverse_squared"]
)
def test_nonadvertised_poisson_efs_valid_explicit_start_matches_pinned_r_strict(
    link: str,
) -> None:
    """Integer-count Poisson extensions have finite parity from a valid start."""
    data, family, setup, fitting_data, beta = _integer_poisson_explicit_start_case(link)
    formula = "y ~ s(x, bs='cr', k=7)"
    rho = efs_initial_log_lambda(setup, family)
    python_fit = dense_efs_known_scale(
        fitting_data,
        initial_log_lambda=rho,
        beta_init=beta,
    )
    public_beta = np.asarray(
        penalty_ops.transform_coefficients(fitting_data.penalty_structure, beta)
    )
    r_trace = RBridge(mode="subprocess").efs_diagnostics(
        formula,
        data,
        f"poisson_{link}",
        initial_smoothing=np.exp(np.asarray(rho)),
        scale=1.0,
        initial_beta=public_beta,
        null_coef=True,
    )

    collector = _AssertCollector()
    collector.check(
        "converged",
        lambda: check_that(python_fit.converged, python_fit.convergence_info),
    )
    collector.check(
        "finite criterion",
        lambda: check_that(
            bool(np.isfinite(np.asarray(python_fit.score))),
            "selected criterion is nonfinite",
        ),
    )
    collector.check(
        "fitted values",
        lambda: np.testing.assert_allclose(
            python_fit.pirls_result.mu,
            r_trace["selected_fitted_values"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "deviance",
        lambda: np.testing.assert_allclose(
            python_fit.pirls_result.deviance,
            r_trace["selected_deviance"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "criterion",
        lambda: np.testing.assert_allclose(
            python_fit.score,
            r_trace["final_score"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "smoothing parameters",
        lambda: np.testing.assert_allclose(
            python_fit.smoothing_params,
            r_trace["selected_packed_sp"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any(f"Poisson/{link} explicit-start EFS parity")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_bounded_poisson_efs_default_start_failure_matches_pinned_r(link: str) -> None:
    """Keep the invalid default mustart separate from explicit-start support."""
    data, family, setup, fitting_data, _ = _integer_poisson_explicit_start_case(link)
    rho = efs_initial_log_lambda(setup, family)
    python_fit = dense_efs_known_scale(fitting_data, initial_log_lambda=rho)
    assert not python_fit.converged
    assert python_fit.convergence_info == "inner_failure"
    with pytest.raises(RBridgeError, match="Pinned EFS Rscript failed"):
        RBridge(mode="subprocess").fit_efs(
            "y ~ s(x, bs='cr', k=7)",
            data,
            f"poisson_{link}",
            initial_smoothing=np.exp(np.asarray(rho)),
            scale=1.0,
            null_coef=True,
        )
