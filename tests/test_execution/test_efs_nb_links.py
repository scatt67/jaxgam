"""Pinned dense-EFS gates for constructor-supported NB links."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.execution.efs import dense_efs_known_scale, efs_initial_log_lambda
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import FittingData
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector, check_that, r_available
from tests.r_bridge import RBridge
from tests.tolerances import MODERATE, STRICT

FORMULA = "y ~ s(x, bs='cr', k=7) + s(z, bs='cr', k=5)"


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(812)
    n = 120
    x = np.linspace(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    theta = 2.7
    mu = 3.0 + 0.8 * np.sin(2.6 * x) - 0.25 * z
    y = rng.negative_binomial(theta, theta / (theta + mu))
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "w": 0.5 + rng.random(n),
            "off": 0.04 * z,
        }
    )


def _build(
    data: pd.DataFrame, family: NegativeBinomial
) -> tuple[ModelSetup, FittingData]:
    setup = ModelSetup.build(
        parse_formula(FORMULA),
        data,
        weights=data["w"].to_numpy(),
        offset=data["off"].to_numpy(),
    )
    return setup, FittingData.from_setup(setup, family)


def _estimated_starts(
    data: pd.DataFrame, family: NegativeBinomial, fd: FittingData
) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = data["x"].to_numpy()
    offset = data["off"].to_numpy()
    target_mu = 3.0 + 0.4 * np.sin(2.0 * x)
    target_eta = np.asarray(family.link.link(target_mu))
    beta = np.linalg.lstsq(np.asarray(fd.X), target_eta - offset, rcond=None)[0]
    null_mu = np.array([float(np.mean(data["y"]))])
    null_eta = float(np.asarray(family.link.link(null_mu))[0])
    beta_old = np.linalg.lstsq(
        np.asarray(fd.X), np.full(len(data), null_eta) - offset, rcond=None
    )[0]
    return jnp.asarray(beta), jnp.asarray(beta_old)


@pytest.mark.parametrize("link", ["identity", "sqrt"])
def test_fixed_nb_noncanonical_efs_rejects_fractional_response_below_one(
    link: str,
) -> None:
    """Keep the fixed-theta controller inside its score-parity response domain."""
    data = _data()
    data["y"] = data["y"].astype(float)
    data.loc[data.index[0], "y"] = 0.25
    family = NegativeBinomial(theta=2.7, fixed=True, link=link)
    setup, fd = _build(data, family)
    rho = efs_initial_log_lambda(setup, family)
    with pytest.raises(ValueError, match="fractional responses below one"):
        dense_efs_known_scale(fd, initial_log_lambda=rho)


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize("fixed", [True, False], ids=["fixed", "estimated"])
@pytest.mark.parametrize("link", ["identity", "sqrt"])
def test_nb_noncanonical_efs_matched_start_default_controller_profile_matches_pinned_r(
    fixed: bool, link: str
) -> None:
    """Matched-start fits use default controls and the reviewed MODERATE gate.

    The conditional-theta layer remains STRICT in
    ``test_conditional_theta_nonlog_links_match_pinned_r_strict``. Four
    distinct source/trajectory correction passes are recorded in
    ``/private/tmp/jaxgam-efs51-nb-links-tight-tolerance-pass4-20260910.json``.
    Default controller stopping leaves selected means within 1.78e-7 and smoothing
    parameters within 2.08e-5 of pinned R; no LOOSE tolerance is used.
    """
    theta = 2.7
    data = _data()
    family = NegativeBinomial(theta=theta, fixed=fixed, link=link)
    setup, fd = _build(data, family)
    rho = efs_initial_log_lambda(setup, family)
    bridge = RBridge(mode="subprocess")
    family_key = f"nb_{link}"

    if fixed:
        j_fit = dense_efs_known_scale(fd, initial_log_lambda=rho)
        r_fit = bridge.fit_efs(
            FORMULA,
            data,
            family_key,
            weights="w",
            offset="off",
            initial_smoothing=np.exp(np.asarray(rho)),
            scale=1.0,
            theta=theta,
            null_coef=True,
        )
        r_theta = float(r_fit["theta"])
        r_coefficients = r_fit["coefficients"]
        r_mu = r_fit["fitted_values"]
        r_deviance = r_fit["deviance"]
        r_score = r_fit["reml_score"]
        r_smoothing = r_fit["smoothing_params"]
        r_outer_iterations = int(r_fit["outer_iterations"])
    else:
        beta, beta_old = _estimated_starts(data, family, fd)
        j_fit = dense_efs_known_scale(
            fd,
            initial_log_lambda=rho,
            initial_log_theta=jnp.asarray([np.log(theta)]),
            beta_init=beta,
            beta_old_init=beta_old,
        )
        diagnostic = bridge.efs_diagnostics(
            FORMULA,
            data,
            family_key,
            weights="w",
            offset="off",
            initial_smoothing=np.exp(np.asarray(rho)),
            scale=1.0,
            initial_log_theta=np.log(theta),
            initial_beta=np.asarray(
                penalty_ops.transform_coefficients(fd.penalty_structure, beta)
            ),
            beta_old_init=np.asarray(
                penalty_ops.transform_coefficients(fd.penalty_structure, beta_old)
            ),
        )
        r_theta = float(diagnostic["selected_packed_sp"][0])
        r_coefficients = diagnostic["selected_coefficients"]
        r_mu = diagnostic["selected_fitted_values"]
        r_deviance = diagnostic["selected_deviance"]
        r_score = diagnostic["final_score"]
        r_smoothing = diagnostic["selected_packed_sp"][1:]
        r_outer_iterations = len(set(diagnostic["statistics"]["call"]))

    collector = _AssertCollector()
    collector.check(
        "coefficients",
        lambda: np.testing.assert_allclose(
            penalty_ops.transform_coefficients(
                fd.penalty_structure, j_fit.pirls_result.coefficients
            ),
            r_coefficients,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    for label, actual, expected in (
        ("fitted values", j_fit.pirls_result.mu, r_mu),
        ("deviance", j_fit.pirls_result.deviance, r_deviance),
        ("criterion", j_fit.score, r_score),
        ("smoothing parameters", j_fit.smoothing_params, r_smoothing),
        ("theta", j_fit.theta, r_theta),
    ):
        collector.check(
            label,
            lambda actual=actual, expected=expected: np.testing.assert_allclose(
                actual,
                expected,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
            ),
        )
    if fixed:
        collector.check(
            "fixed theta",
            lambda: np.testing.assert_allclose(
                j_fit.theta, theta, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        collector.check(
            "outer iterations",
            lambda: np.testing.assert_equal(j_fit.n_iter, r_outer_iterations),
        )
    else:
        collector.check(
            "default stopping branch proximity",
            lambda: check_that(
                abs(j_fit.n_iter - r_outer_iterations) <= 1,
                f"outer iteration difference exceeds one: {j_fit.n_iter} vs "
                f"{r_outer_iterations}",
            ),
        )
    collector.check(
        "finite converged state",
        lambda: check_that(
            j_fit.converged
            and j_fit.theta is not None
            and j_fit.theta > 0.0
            and np.all(np.isfinite(np.asarray(j_fit.pirls_result.mu)))
            and float(j_fit.pirls_result.deviance) >= 0.0,
            f"invalid selected state: {j_fit.convergence_info}",
        ),
    )
    collector.raise_if_any(f"NB/{link} {'fixed' if fixed else 'estimated'} EFS")
