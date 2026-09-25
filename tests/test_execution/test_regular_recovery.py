"""Source signed-controller admission retains initializer diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.control import FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.regular_stream import fit_regular_streamed_pirls
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Binomial, Gamma
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
    batch_initial_working_quantities,
)
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.results import GAMPredictionResult
from tests.helpers import _AssertCollector
from tests.r_bridge import RBridge
from tests.tolerances import MODERATE, STRICT


def test_signed_recovery_admission_preserves_default_guard_and_raw_arithmetic():
    family = Binomial("log")
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    # Preserve the three inverse-link-sensitive rows from the binary oracle,
    # alongside empty/zero-information support and an ordinary informative row.
    y = jnp.asarray([1.0, 1.0, np.nextafter(1.0, 0.0), 0.6, 777.0])
    weight = jnp.asarray(
        [0.6502309357264154, 0.9508655818040038, 0.5377022799708129, 1.0, 0.0]
    )
    eta = jnp.asarray(
        [-0.36095250327523276, -0.29611283089587714, -0.39328035348801693, -0.4, -0.4]
    )
    for eager in (False, True):
        with jax.disable_jit(eager):
            guarded = batch_initial_working_quantities(
                y,
                weight,
                jnp.zeros(5),
                jnp.ones(5, bool),
                eta,
                parameters,
                family,
                context,
            )
            admitted = batch_initial_working_quantities(
                y,
                weight,
                jnp.zeros(5),
                jnp.ones(5, bool),
                eta,
                parameters,
                family,
                context,
                source_signed_recovery=True,
            )
        assert not bool(guarded.working_system_admissible)
        assert bool(admitted.working_system_admissible)
        assert np.any(np.asarray(admitted.alpha_resolution_unresolved))
        for field in (
            "alpha_resolution_unresolved",
            "newton_alpha_raw",
            "newton_alpha",
            "newton_weight",
            "newton_response",
            "observed_weight",
            "fisher_weight",
        ):
            np.testing.assert_array_equal(
                getattr(admitted, field), getattr(guarded, field)
            )
        alpha = np.asarray(admitted.newton_alpha_raw)
        np.testing.assert_array_equal(
            np.asarray(admitted.newton_alpha)[:4],
            np.where(alpha[:4] == 0.0, np.finfo(float).eps, alpha[:4]),
        )


def test_signed_recovery_admission_still_rejects_invalid_working_inputs():
    family = Binomial("log")
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    for response, weight, eta in ((1.1, 1.0, -0.4), (1.0, -1.0, -0.4), (1.0, 1.0, 0.1)):
        result = batch_initial_working_quantities(
            jnp.asarray([response]),
            jnp.asarray([weight]),
            jnp.zeros(1),
            jnp.ones(1, bool),
            jnp.asarray([eta]),
            parameters,
            family,
            context,
            source_signed_recovery=True,
        )
        assert not bool(result.working_system_admissible)


@pytest.mark.usefixtures("r_bridge")
def test_real_gamma_fisher_recovery_matches_pinned_full_trajectory():
    rng = np.random.default_rng(1)
    x = np.linspace(-0.9, 0.9, 41)
    y = np.exp(rng.normal(0.0, 2.0, len(x)))
    weight = 0.1 + rng.uniform(size=len(x))
    offset = 0.02 * np.sin(x)
    family = Gamma("identity")
    source = DataFrameRowSource(
        pd.DataFrame({"y": y, "x": x}), response="y", weights=weight, offset=offset
    )
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    stream = StreamDesign(prepared, source)
    X = np.column_stack((np.ones(len(x)), x))
    oracle = RBridge(mode="rpy2").gamma_fisher_recovery_reference(X, y, weight, offset)
    reference = oracle["reference"]
    covariance = oracle["covariance"]
    for B in (1, 17, 200):
        result = fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=10000000,
            score_scale=0.7,
            control=StreamPIRLSControl(
                batch_rows=B,
                tol=1e-10,
                max_iter=100,
                max_halvings=100,
                solver_policy="qr",
            ),
        )
        state = result.state
        assert state.converged
        assert result.fisher_recoveries == oracle["recoveries"]
        assert state.backtracks > 0
        prediction = GAMPredictionResult._from_stream_fit(
            stream_state=state,
            prepared=prepared,
            metadata=PreparedFittingMetadata.from_prepared(prepared, family),
            family=family,
            formula="y ~ x",
            method="REML",
            control=FitControl(),
        )
        np.testing.assert_allclose(
            np.r_[
                state.coefficients,
                state.deviance,
                state.scale,
                state.edf,
                prediction.score,
            ],
            reference,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        actual_covariance = np.asarray(
            state.fisher_coefficient_factor.hessian_inverse(jnp.eye(2))
        )
        np.testing.assert_allclose(
            actual_covariance, covariance, rtol=STRICT.rtol, atol=STRICT.atol
        )
        np.testing.assert_allclose(
            family.link.inverse(X @ np.asarray(state.coefficients) + offset),
            oracle["fitted"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            np.sqrt(state.scale * np.einsum("ij,jk,ik->i", X, actual_covariance, X)),
            np.sqrt(reference[3] * np.einsum("ij,jk,ik->i", X, covariance, X)),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    exhausted = fit_regular_streamed_pirls(
        stream,
        family,
        np.empty(0),
        maximum_bytes=10000000,
        score_scale=0.7,
        control=StreamPIRLSControl(
            batch_rows=17,
            tol=1e-10,
            max_iter=2,
            max_halvings=100,
            solver_policy="qr",
        ),
    )
    assert not exhausted.state.converged
    assert exhausted.state.n_iter == 2
    assert exhausted.fisher_recoveries > 0
    assert np.all(np.isfinite(exhausted.state.coefficients))


@pytest.mark.usefixtures("r_bridge")
def test_binomial_log_admission_matches_pinned_default_final_fields_and_se():
    weights = np.r_[np.geomspace(0.05, 100.0, 81), 0.8]
    cases = {
        "mixed": (
            np.repeat([1.0, np.nextafter(1.0, 0.0), 1.0 - 1e-12, 0.6], len(weights)),
            np.tile(weights, 4),
        ),
        "adjacent": (np.full(len(weights), np.nextafter(1.0, 0.0)), weights),
        "near": (np.full(len(weights), 1.0 - 1e-12), weights),
        "exact": (np.ones(len(weights)), weights),
    }
    checks = _AssertCollector()
    for label, (y, weight) in cases.items():
        # This exact mixed system oscillates at epsilon=1e-10 on pinned
        # native AMD64 R; its selected source control stays 100x tighter
        # than gam.control's default.
        oracle_fit = RBridge(mode="rpy2").binomial_log_admission_reference(
            y, weight, 1e-9 if label == "mixed" else 1e-10
        )
        family = Binomial("log")
        source = DataFrameRowSource(
            pd.DataFrame({"y": y}), response="y", weights=weight
        )
        prepared = prepare_model(parse_formula("y ~ 1"), source, family=family)
        stream = StreamDesign(prepared, source)
        if label == "exact":
            assert "inner loop 2" in oracle_fit["error"]
            with pytest.raises(ValueError, match="null coefficient anchor"):
                fit_regular_streamed_pirls(
                    stream,
                    family,
                    np.empty(0),
                    maximum_bytes=10000000,
                    control=StreamPIRLSControl(
                        batch_rows=17,
                        max_iter=100,
                        max_halvings=100,
                        solver_policy="qr",
                    ),
                )
            continue
        oracle_covariance = oracle_fit["covariance"]
        oracle = oracle_fit["reference"]
        for B, eager in ((1, False), (17, False), (200, False), (200, True)):
            with jax.disable_jit(eager):
                result = fit_regular_streamed_pirls(
                    stream,
                    family,
                    np.empty(0),
                    maximum_bytes=10000000,
                    control=StreamPIRLSControl(
                        batch_rows=B,
                        tol=1e-10,
                        max_iter=100,
                        max_halvings=100,
                        solver_policy="qr",
                    ),
                )
            state = result.state
            assert state.converged
            assert state.backtracks > 0
            prediction = GAMPredictionResult._from_stream_fit(
                stream_state=state,
                prepared=prepared,
                metadata=PreparedFittingMetadata.from_prepared(prepared, family),
                family=family,
                formula="y ~ 1",
                method="REML",
                control=FitControl(),
            )
            covariance = np.asarray(
                state.fisher_coefficient_factor.hessian_inverse(jnp.eye(1))
            )
            # Four source-operation/full-trajectory correction passes and a
            # fifth matched-state/partition audit are recorded in the review
            # document. Main-agent approval permits only these exact fields
            # for this preserved default-control boundary fixture family.
            beta_tolerance = MODERATE if label == "mixed" and B == 200 else STRICT
            boundary_tolerance = (
                MODERATE if label == "near" and B in (1, 200) else STRICT
            )
            for leaf, value, reference, tolerance in (
                ("coefficient", state.coefficients[0], oracle[0], beta_tolerance),
                (
                    "deviance_knownscale_edf",
                    np.r_[state.deviance, state.scale, state.edf],
                    oracle[1:4],
                    STRICT,
                ),
                ("reml", prediction.score, oracle[4], boundary_tolerance),
                (
                    "fitted",
                    family.link.inverse(np.full(len(y), state.coefficients[0])),
                    oracle_fit["fitted"],
                    STRICT,
                ),
                ("fisher_covariance", covariance, oracle_covariance, STRICT),
                (
                    "standard_error",
                    np.sqrt(covariance),
                    np.sqrt(oracle_covariance),
                    boundary_tolerance,
                ),
            ):
                checks.check(
                    f"{label}/B{B}/eager={eager}/{leaf}",
                    lambda a=value, b=reference, t=tolerance: (
                        np.testing.assert_allclose(a, b, rtol=t.rtol, atol=t.atol)
                    ),
                )
    checks.raise_if_any("Source signed-controller Binomial/log admission")
