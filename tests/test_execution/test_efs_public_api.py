"""Public dense-EFS dispatch, precedence, and Phase-3 result gates."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jaxgam import (
    GAM,
    EFSControl,
    EFSOptimizerDiagnostics,
    FitControl,
    GAMInferenceResult,
    GAMPredictionResult,
    GAMResults,
)
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs import _EFSDiagnosticsAccumulator
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.registry import get_family
from tests.helpers import _AssertCollector, check_that, r_available
from tests.r_bridge import RBridge, RBridgeError
from tests.tolerances import STRICT


def _poisson_data(*, seed: int = 882, n: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    mu = np.exp(0.2 + 0.45 * np.sin(2.5 * x))
    return pd.DataFrame({"x": x, "y": rng.poisson(mu)})


def _nb_data(*, seed: int = 883, n: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    theta = 0.8
    mu = np.exp(0.3 + 0.65 * np.sin(2.4 * x))
    return pd.DataFrame(
        {"x": x, "y": rng.negative_binomial(theta, theta / (theta + mu))}
    )


def test_compact_efs_diagnostics_count_failure_and_stabilization_events() -> None:
    accumulator = _EFSDiagnosticsAccumulator()
    statistics = SimpleNamespace(
        determinant_derivative=np.array([0.5, 1.0, 1.0]),
        fisher_trace=np.array([1.0, 0.0, 0.0]),
        quadratic=np.array([0.0, 0.0, 1.0]),
    )
    old = SimpleNamespace(
        log_lambda=np.zeros(3),
        statistics=statistics,
        valid=False,
        gdi1_candidate_valid=np.array(False),
        pirls_result=SimpleNamespace(n_iter=np.array(3)),
        theta_n_iter=np.array(2),
        positive_curvature_retry_count=np.array(0),
    )
    accepted = SimpleNamespace(log_lambda=np.array([0.2, -0.1, 0.0]))
    raw = SimpleNamespace(ratio=np.array([1.0, 1e6, 1.0]))
    accumulator.observe_fit(old)
    accumulator.observe_update(old, np.array(1.0), 1.0, 5.0, raw)
    accumulator.observe_accepted(old, accepted)
    diagnostics = accumulator.finish(
        stop_reason="invalid_trial",
        outer_iterations=1,
        history=(3.0, 2.0),
        score_phi_history=(1.0,),
        multiplier=1.0,
        update_residual=np.array([0.2, -0.1, 0.0]),
    )
    assert diagnostics.inner_iterations == 3
    assert diagnostics.theta_iterations == 2
    assert diagnostics.numerator_clamp_count == 1
    assert diagnostics.ratio_replacement_count == 1
    assert diagnostics.log_lambda_cap_count == 1
    assert diagnostics.invalid_fit_seen
    assert diagnostics.stabilized_solve_seen
    assert diagnostics.max_proposed_movement == np.log(1e6)
    assert diagnostics.max_accepted_movement == 0.2
    assert diagnostics.final_update_residual == (0.2, -0.1, 0.0)


def test_explicit_efs_materializes_equal_picklable_result_modes(monkeypatch) -> None:
    from jaxgam.execution import efs as execution_efs

    data = _poisson_data()
    formula = "y ~ s(x, bs='cr', k=6)"
    control = FitControl(efs=EFSControl(history_limit=8))
    original = execution_efs.dense_efs_known_scale
    calls: list[EFSControl] = []

    def recording_dispatch(*args, **kwargs):
        calls.append(kwargs["control"])
        return original(*args, **kwargs)

    monkeypatch.setattr(execution_efs, "dense_efs_known_scale", recording_dispatch)
    full = GAM(formula, family="poisson", optimizer="efs", control=control).fit(data)
    inference = GAM(formula, family="poisson", optimizer="efs", control=control).fit(
        data, result="inference"
    )
    prediction = GAM(formula, family="poisson", optimizer="efs", control=control).fit(
        data, result="prediction"
    )

    collector = _AssertCollector()
    for label, result, result_type in (
        ("full", full, GAMResults),
        ("inference", inference, GAMInferenceResult),
        ("prediction", prediction, GAMPredictionResult),
    ):
        collector.check(
            f"{label} type",
            lambda r=result, t=result_type, name=label: check_that(
                isinstance(r, t), f"{name} result has type {type(r)!r}"
            ),
        )
        collector.check(
            f"{label} strategy",
            lambda r=result: np.testing.assert_equal(r.lambda_strategy, "efs_reml"),
        )
        restored = pickle.loads(pickle.dumps(result))
        collector.check(
            f"{label} pickle prediction",
            lambda a=restored, e=result: np.testing.assert_array_equal(
                a.predict(data), e.predict(data)
            ),
        )
        collector.check(
            f"{label} pickle diagnostics",
            lambda a=restored, e=result: np.testing.assert_equal(
                a.optimizer_diagnostics, e.optimizer_diagnostics
            ),
        )
    collector.check(
        "full/inference coefficients",
        lambda: np.testing.assert_array_equal(
            full.coefficients, inference.coefficients
        ),
    )
    collector.check(
        "full/prediction coefficients",
        lambda: np.testing.assert_array_equal(
            full.coefficients, prediction.coefficients
        ),
    )
    collector.check(
        "full/inference prediction",
        lambda: np.testing.assert_array_equal(
            full.predict(data), inference.predict(data)
        ),
    )
    collector.check(
        "full/prediction prediction",
        lambda: np.testing.assert_array_equal(
            full.predict(data), prediction.predict(data)
        ),
    )
    collector.check(
        "composed control identity",
        lambda: np.testing.assert_equal(calls, [control.efs, control.efs, control.efs]),
    )
    diagnostics = full.optimizer_diagnostics
    collector.check(
        "immutable compact diagnostics",
        lambda: check_that(
            isinstance(diagnostics, EFSOptimizerDiagnostics)
            and diagnostics == inference.optimizer_diagnostics
            and diagnostics == prediction.optimizer_diagnostics,
            "all result modes must retain the same EFS diagnostics",
        ),
    )
    assert diagnostics is not None
    collector.check(
        "bounded accepted histories",
        lambda: check_that(
            len(diagnostics.accepted_score_history) <= control.efs.history_limit
            and len(diagnostics.accepted_score_phi_history)
            <= control.efs.history_limit,
            "accepted EFS histories exceeded EFSControl.history_limit",
        ),
    )
    collector.check(
        "controller profile",
        lambda: check_that(
            diagnostics.reference_profile == "mgcv-1.9-3-efsudr-dense"
            and diagnostics.trace_method == "exact-dense-fisher"
            and diagnostics.step_policy == "efsudr-extension-contraction"
            and diagnostics.stop_reason
            in {"score_window", "deviance_change", "iteration_limit"},
            f"unexpected controller profile: {diagnostics!r}",
        ),
    )
    collector.check(
        "honest iteration and movement diagnostics",
        lambda: check_that(
            diagnostics.outer_iterations == full.n_iter
            and diagnostics.inner_iterations > 0
            and diagnostics.theta_iterations == 0
            and diagnostics.max_proposed_movement
            >= diagnostics.max_accepted_movement
            >= 0.0
            and all(np.isfinite(diagnostics.final_update_residual)),
            f"invalid compact diagnostics: {diagnostics!r}",
        ),
    )
    collector.check(
        "event counter domains",
        lambda: check_that(
            min(
                diagnostics.numerator_clamp_count,
                diagnostics.ratio_replacement_count,
                diagnostics.log_lambda_cap_count,
            )
            >= 0,
            f"negative EFS event count: {diagnostics!r}",
        ),
    )
    collector.raise_if_any("public EFS result modes")


@pytest.mark.parametrize("result_mode", ["full", "inference", "prediction"])
def test_estimated_nb_efs_snapshots_selected_theta_without_mutating_caller(
    result_mode: str,
) -> None:
    data = _nb_data()
    family = NegativeBinomial(theta=0.8)
    incoming = family.get_theta(transformed=False).copy()
    result = GAM("y ~ s(x, bs='cr', k=6)", family=family, optimizer="efs").fit(
        data, result=result_mode
    )
    np.testing.assert_array_equal(family.get_theta(transformed=False), incoming)
    assert result.family is not family
    np.testing.assert_allclose(
        result.family.get_theta(transformed=True)[0],
        result.theta,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert result.optimizer_diagnostics is not None
    assert result.optimizer_diagnostics.theta_iterations > 0


def test_public_efs_dispatches_unknown_scale_with_composed_control(monkeypatch) -> None:
    from jaxgam.execution import efs as execution_efs

    rng = np.random.default_rng(885)
    x = np.linspace(-1.0, 1.0, 72)
    data = pd.DataFrame(
        {"x": x, "y": 1.2 + 0.5 * np.sin(2.2 * x) + rng.normal(0, 0.04, len(x))}
    )
    control = FitControl(efs=EFSControl(history_limit=6))
    original = execution_efs.dense_efs_unknown_scale
    calls: list[EFSControl] = []

    def recording_dispatch(*args, **kwargs):
        calls.append(kwargs["control"])
        return original(*args, **kwargs)

    monkeypatch.setattr(execution_efs, "dense_efs_unknown_scale", recording_dispatch)
    result = GAM("y ~ s(x, bs='cr', k=6)", optimizer="efs", control=control).fit(data)
    assert result.lambda_strategy == "efs_reml"
    assert result.scale > 0.0
    assert calls == [control.efs]
    assert result.optimizer_diagnostics is not None
    assert (
        len(result.optimizer_diagnostics.accepted_score_phi_history)
        <= control.efs.history_limit
    )
    assert result.optimizer_diagnostics.theta_iterations == 0


def test_fixed_sp_and_zero_penalty_take_precedence_over_efs(monkeypatch) -> None:
    from jaxgam.execution import efs as execution_efs

    def unexpected_efs(*_args, **_kwargs):
        raise AssertionError("EFS controller must be bypassed")

    monkeypatch.setattr(execution_efs, "dense_efs_known_scale", unexpected_efs)
    data = _poisson_data(n=48)
    fixed = GAM(
        "y ~ s(x, bs='cr', k=5)",
        family="poisson",
        optimizer="efs",
        sp=[0.3],
    ).fit(data)
    parametric = GAM("y ~ x", family="poisson", optimizer="efs").fit(data)
    assert fixed.lambda_strategy == "fixed"
    assert parametric.lambda_strategy == "newton_reml"
    assert fixed.optimizer_diagnostics is None
    assert parametric.optimizer_diagnostics is None


def test_parametric_nb_efs_bypass_matches_ordinary_newton_family_snapshot() -> None:
    data = _nb_data(n=72)
    family = NegativeBinomial(theta=0.8)
    incoming = family.get_theta(transformed=False).copy()
    result = GAM("y ~ x", family=family, optimizer="efs").fit(data)
    ordinary = GAM("y ~ x", family=NegativeBinomial(theta=0.8), optimizer="newton").fit(
        data
    )
    assert result.lambda_strategy == "newton_reml"
    assert result.optimizer_diagnostics is None
    np.testing.assert_array_equal(family.get_theta(transformed=False), incoming)
    assert result.family is not family
    assert result.theta is None
    np.testing.assert_array_equal(result.coefficients, ordinary.coefficients)
    np.testing.assert_array_equal(result.fitted_values, ordinary.fitted_values)
    assert result.score == ordinary.score
    assert ordinary.theta is None
    np.testing.assert_array_equal(
        result.family.get_theta(transformed=False),
        ordinary.family.get_theta(transformed=False),
    )
    np.testing.assert_array_equal(result.family.get_theta(transformed=False), incoming)


def test_omitted_and_explicit_newton_remain_identical() -> None:
    data = _poisson_data(n=56)
    formula = "y ~ s(x, bs='cr', k=5)"
    implicit = GAM(formula, family="poisson").fit(data)
    explicit = GAM(formula, family="poisson", optimizer="newton").fit(data)
    np.testing.assert_array_equal(implicit.coefficients, explicit.coefficients)
    np.testing.assert_array_equal(implicit.fitted_values, explicit.fitted_values)
    np.testing.assert_array_equal(implicit.smoothing_params, explicit.smoothing_params)
    assert implicit.score == explicit.score
    assert implicit.lambda_strategy == explicit.lambda_strategy == "newton_reml"
    assert implicit.optimizer_diagnostics is None
    assert explicit.optimizer_diagnostics is None


def test_string_nb_efs_does_not_mutate_registry_singleton() -> None:
    data = _nb_data(n=72)
    registered = get_family("nb")
    incoming = registered.get_theta(transformed=False).copy()
    result = GAM("y ~ s(x, bs='cr', k=5)", family="nb", optimizer="efs").fit(data)
    np.testing.assert_array_equal(registered.get_theta(transformed=False), incoming)
    assert result.family is not registered
    np.testing.assert_allclose(
        result.family.get_theta(transformed=True)[0],
        result.theta,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize("link", ["identity", "sqrt"])
def test_fixed_nb_nonlog_public_efs_is_supported(link: str) -> None:
    data = _nb_data(n=72)
    result = GAM(
        "y ~ s(x, bs='cr', k=5)",
        family=NegativeBinomial(theta=0.8, fixed=True, link=link),
        optimizer="efs",
    ).fit(data)
    assert result.converged
    assert result.lambda_strategy == "efs_reml"
    assert result.theta == 0.8
    assert result.optimizer_diagnostics is not None


@pytest.mark.parametrize("link", ["identity", "sqrt"])
def test_estimated_nb_nonlog_public_efs_rejects_pinned_default_start_boundary(
    link: str,
) -> None:
    with pytest.raises(ValueError, match="pinned mgcv rejects the same input"):
        GAM(
            "y ~ s(x, bs='cr', k=5)",
            family=NegativeBinomial(theta=0.8, link=link),
            optimizer="efs",
        ).fit(_nb_data(n=72))


def test_nb_identity_public_diagnostics_report_positive_curvature_retry() -> None:
    data = _nb_data(seed=883, n=96)
    result = GAM(
        "y ~ s(x, bs='cr', k=5)",
        family=NegativeBinomial(theta=0.8, link="identity"),
        optimizer="efs",
    ).fit(data, offset=np.ones(len(data)))

    assert result.converged
    assert result.optimizer_diagnostics is not None
    assert result.optimizer_diagnostics.stabilized_solve_seen


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize("link", ["identity", "sqrt"])
def test_estimated_nb_nonlog_zero_offset_boundary_matches_pinned_r_failure(
    link: str,
) -> None:
    data = _nb_data(n=72)
    with pytest.raises(RBridgeError, match=r"inner loop|missing value"):
        RBridge(mode="subprocess").fit_efs("y ~ s(x, bs='cr', k=5)", data, f"nb_{link}")


def test_public_efs_rejects_unsupported_links_and_optimizers() -> None:
    data = _nb_data(n=48)
    with pytest.raises(NotImplementedError, match="known-scale family/link"):
        GAM(
            "y ~ s(x, bs='cr', k=5)",
            family=NegativeBinomial(theta=0.8, link="logit"),
            optimizer="efs",
        ).fit(data)
    with pytest.raises(NotImplementedError, match="optimizer='bfgs'"):
        GAM("y ~ s(x)", optimizer="bfgs")
    with pytest.raises(ValueError, match="optimizer must"):
        GAM("y ~ s(x)", optimizer=True)


def test_estimated_efs_rejects_rowsource_explicitly_but_fixed_sp_precedes() -> None:
    data = _poisson_data(n=48)
    source = DataFrameRowSource(data, response="y")
    control = FitControl(execution="stream")
    with pytest.raises(NotImplementedError, match=r"estimated smoothing.*RowSource"):
        GAM(
            "y ~ s(x, bs='cr', k=5)",
            family="poisson",
            optimizer="efs",
            control=control,
        ).fit(source, result="prediction")
    result = GAM(
        "y ~ s(x, bs='cr', k=5)",
        family="poisson",
        optimizer="efs",
        sp=[0.4],
        control=control,
    ).fit(source, result="prediction")
    assert result.lambda_strategy == "fixed"
    assert result.optimizer_diagnostics is None
