"""Exact free-log-scale curvature carried by regular streamed REML trials."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.reml import evaluate_regular_stream_reml
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Gamma, Gaussian, Poisson
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from tests.helpers import _AssertCollector
from tests.r_bridge import RBridge
from tests.test_execution.test_regular_penalized_matrix import _rank_one_preparation
from tests.test_execution.test_regular_starts import _case
from tests.tolerances import STRICT


def _rank_one_trial(family_class, link, log_phi):
    family, data, weight, offset, start = _case(family_class, link)
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared, rho = _rank_one_preparation(source, family)
    params = rho if family.scale_known else np.r_[rho, log_phi]
    trial = evaluate_regular_stream_reml(
        StreamDesign(prepared, source),
        family,
        params,
        maximum_bytes=64_000_000,
        control=StreamPIRLSControl(
            batch_rows=17, tol=1e-9, max_iter=200, solver_policy="qr"
        ),
        initial_coefficients=start,
    )
    return family, data, weight, trial


@pytest.mark.parametrize("family_class", [Gaussian, Gamma])
@pytest.mark.parametrize("link", ["log", "inverse", "logit"])
@pytest.mark.parametrize("log_phi", [np.log(0.7), np.log(0.004)])
def test_regular_trial_log_scale_curvature_is_exact_fixed_rho_second_derivative(
    family_class, link, log_phi
):
    """With rho fixed only Dp/(2 phi) and the saturated likelihood vary."""
    family, data, weight, trial = _rank_one_trial(family_class, link, log_phi)
    y = jnp.asarray(family.execution_initial_response(data.y.to_numpy(), weight))

    def saturated(value):
        return family.saturated_loglik(y, jnp.asarray(weight), jnp.exp(value))

    saturated_curvature = float(jax.jacfwd(jax.grad(saturated))(jnp.asarray(log_phi)))
    phi = float(np.asarray(trial.fit_result.state.score_scale))
    np.testing.assert_allclose(phi, np.exp(log_phi), rtol=STRICT.rtol, atol=0.0)
    explicit = trial.fit_result.source_score.penalized_deviance / (2.0 * phi)
    np.testing.assert_allclose(
        trial.log_phi_curvature,
        explicit - saturated_curvature,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    if family_class is Gaussian:
        # ls is linear in log phi, so curvature is exactly Dp / (2 phi).
        assert saturated_curvature == 0.0


def test_known_scale_trial_has_no_log_scale_curvature():
    _family, _data, _weight, trial = _rank_one_trial(Poisson, "log", 0.0)
    assert trial.log_phi_curvature is None
    assert np.asarray(trial.gradient).shape == (1,)


@pytest.mark.usefixtures("r_bridge")
def test_smooth_trial_derivatives_match_pinned_gam_fit3_at_r_scale():
    """At R's returned scale, JAX reproduces pinned gam.fit3's 1-D derivatives.

    The R side is a matched-state ``gam.fit3`` evaluation at R's returned
    ``reml.scale`` on the public fit's own model object, so it does not depend
    on newton()'s outer status or history. The curvature is gam.fit3's
    ``d2lr.d2lphi``. Against 40-digit references, R's ``fix.family.ls`` Gamma
    ``ls[2]*phi`` term (digamma cancellation) is off by up to 1.22e-12 while
    JAX's Stirling form is exact; that moves the implied stationary phi by
    ~2.5e-14 relative, whereas a misattributed rejected-trial gradient
    (~1.8e-7) would move it by ~4.5e-9. Gamma gradients are therefore
    compared in phi units, ``phi_R * exp(-g / h)``, at STRICT.
    """
    bridge = RBridge(mode="rpy2")
    checks = _AssertCollector()
    for family_class, link in (
        (Gaussian, "log"),
        (Gaussian, "logit"),
        (Gaussian, "inverse"),
        (Gamma, "logit"),
        (Gamma, "cloglog"),
        (Gamma, "inverse_squared"),
    ):
        family, data, weight, offset, _ = _case(family_class, link)
        expected = bridge.public_fixed_sp_scale_derivatives(
            data, weight, offset, family.family_name.lower(), link, sp=0.5
        )
        assert expected["inner_converged"]
        np.testing.assert_allclose(
            expected["reml_scale"],
            expected["returned_scale"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
        stream = StreamDesign(
            prepare_model(parse_formula('y~s(x,bs="cr",k=8)'), source, family=family),
            source,
        )
        params = np.r_[np.log(0.5), np.log(expected["returned_scale"])]
        trial = evaluate_regular_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=64_000_000,
            control=StreamPIRLSControl(batch_rows=17, solver_policy="qr", tol=1e-11),
        )
        np.testing.assert_array_equal(np.asarray(trial.params), params)
        name = f"{family.family_name}/{link}"
        checks.check(
            f"{name} log-scale curvature",
            lambda trial=trial, expected=expected: np.testing.assert_allclose(
                trial.log_phi_curvature,
                expected["hessian"],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        if family_class is Gamma:
            scale = expected["returned_scale"]

            def implied_scale(gradient, curvature, scale=scale):
                return scale * np.exp(-gradient / curvature)

            checks.check(
                f"{name} log-scale gradient in phi units",
                lambda trial=trial, expected=expected, implied=implied_scale: (
                    np.testing.assert_allclose(
                        implied(
                            float(np.asarray(trial.gradient)[-1]),
                            trial.log_phi_curvature,
                        ),
                        implied(expected["gradient"], expected["hessian"]),
                        rtol=STRICT.rtol,
                        atol=STRICT.atol,
                    )
                ),
            )
        else:
            checks.check(
                f"{name} log-scale gradient",
                lambda trial=trial, expected=expected: np.testing.assert_allclose(
                    np.asarray(trial.gradient)[-1],
                    expected["gradient"],
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
    checks.raise_if_any("Pinned gam.fit3 log-scale derivatives at R scale")
