"""Complete calibrated single-index EFS fits against dense and pinned mgcv."""

import math
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln, xlogy

from jaxgam.control import EFSControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs import dense_efs_known_scale, dense_efs_unknown_scale
from jaxgam.execution.operator_efs_controller import (
    CALIBRATED_OPERATOR_EFS_DOMAIN,
    OperatorEFSCalibratedDomain,
    SourceOperatorEFSFitProvider,
    _indexed_gaussian_score_bound,
    _indexed_poisson_score_bound,
    admitted_operator_efs_score_bound,
    operator_efs_score_envelope,
    run_operator_efs_gaussian_scale,
    run_operator_efs_known_scale,
)
from jaxgam.execution.operator_logdet import OperatorLogDetControl
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector
from tests.operator_efs_calibration import (
    CalibrationCase,
    DecimalReference,
    generate_frame,
    score_errors,
)
from tests.tolerances import STRICT

_TIGHT_OPERATOR_CONTROL = OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15)


@pytest.fixture(
    scope="module",
    params=[
        (300, 37, 2),
        (300, 49, 2),
        (400, 4024, 2),
        (257, 211, 3),
        (512, 712, 3, "unbalanced"),
    ],
)
def calibrated_single_index_fit(request):
    return calibrated_single_index_case(*request.param)


def calibrated_single_index_case(
    p: int, seed: int, multiplicity: int, mode: str = "balanced"
):
    """Build a reproducible complete fit and audit every accepted/trial score."""
    n = multiplicity * p
    rng = np.random.default_rng(seed)
    if mode == "balanced":
        group = np.arange(n) % p
        mean = np.exp(0.2 + 0.3 * np.sin(group / 31.0))
    elif mode == "unbalanced":
        probabilities = 1.0 / (1.0 + np.arange(p)) ** 1.1
        probabilities /= np.sum(probabilities)
        group = np.concatenate(
            (np.arange(p), rng.choice(p, size=n - p, p=probabilities))
        )
        mean = np.exp(1.5 + 0.7 * np.sin(group / 31.0))
    else:
        raise ValueError(f"Unknown calibration case mode: {mode}")
    response = rng.poisson(mean)
    if mode == "unbalanced":
        response[0] = 31
    frame = pd.DataFrame(
        {
            "y": response,
            "g": pd.Categorical(
                [f"g{index}" for index in group],
                categories=[f"g{index}" for index in range(p)],
            ),
        }
    )
    formula = "y~0+s(g,bs='re')"
    spec = parse_formula(formula)
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(
            batch_rows=64, memory_budget_bytes=4 * 1024 * 1024
        ),
    )
    dense_setup = ModelSetup.build(spec, frame)
    dense_data = FittingData.from_setup(dense_setup, Poisson())
    observed_errors = []
    observed_bounds = []

    class _AuditedProvider(SourceOperatorEFSFitProvider):
        def evaluate(self, fit, control):
            state = super().evaluate(fit, control)
            X = dense_setup.X
            beta = fit.coefficients
            eta = X @ beta
            mu = np.exp(eta)
            y = frame["y"].to_numpy(dtype=np.float64)
            deviance = 2 * np.sum(xlogy(y, y / mu) - (y - mu))
            saturated = np.sum(xlogy(y, y) - y - gammaln(y + 1))
            penalty = dense_setup.penalties.materialize(fit.log_lambda)
            H = X.T @ (mu[:, None] * X) + penalty
            logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(H))))
            penalty_logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(penalty))))
            exact = (
                (deviance + beta @ penalty @ beta) / 2
                - saturated
                + (logdet - penalty_logdet) / 2
            )
            observed_errors.append(abs(state.components.score.value - exact))
            observed_bounds.append(state.score_error_bound)
            return state

    provider = _AuditedProvider(
        model,
        source,
        Poisson(),
        fit_control=_TIGHT_OPERATOR_CONTROL,
        fit_budget_bytes=16 * 1024 * 1024,
        components_budget_bytes=32 * 1024 * 1024,
        aggregate_budget_bytes=128 * 1024 * 1024,
    )
    initial = np.log(np.array([0.8]))
    operator = run_operator_efs_known_scale(
        provider,
        initial,
        control=EFSControl(outer_limit=20, pirls_tolerance=1e-11),
        initial_accuracy=OperatorLogDetControl(probes=4, lanczos_depth=4, seed=seed),
    )
    dense = dense_efs_known_scale(
        dense_data,
        initial_log_lambda=initial,
        control=EFSControl(outer_limit=20, pirls_tolerance=1e-11),
    )
    return formula, frame, operator, dense, observed_errors, observed_bounds


def test_calibrated_operator_complete_fit_matches_dense_efs(
    calibrated_single_index_fit,
):
    _, frame, operator, dense, errors, bounds = calibrated_single_index_fit
    selected = operator.accepted
    assert operator.converged
    assert dense.converged
    assert bool(np.asarray(dense.pirls_result.converged))
    assert selected.fit.converged
    assert selected.score_bound_method == "indexed_poisson_empirical_v2"
    assert operator.stop_reason == "score_window"
    assert operator.outer_iterations == dense.n_iter
    assert operator.indeterminate_comparisons == 0
    assert operator.attempted_fits == len(errors)
    assert all(bound is not None for bound in bounds)
    assert all(error <= bound for error, bound in zip(errors, bounds, strict=True))
    assert selected.fit.model.n_coef > 256
    collector = _AssertCollector()
    collector.check(
        "dense score",
        lambda: np.testing.assert_allclose(
            selected.components.score.value,
            dense.score,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "dense smoothing",
        lambda: np.testing.assert_allclose(
            np.exp(selected.fit.log_lambda),
            dense.smoothing_params,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "dense deviance",
        lambda: np.testing.assert_allclose(
            selected.fit.deviance,
            dense.pirls_result.deviance,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any(f"calibrated diagonal EFS p={selected.fit.model.n_coef}")
    assert 2 * selected.fit.model.n_coef <= len(frame) <= 3 * selected.fit.model.n_coef


@pytest.mark.parametrize(
    "unsupported_rows",
    [
        {"unit_weights": False},
        {"zero_offset": False},
        {"maximum_response": 33.0},
    ],
)
def test_score_envelope_refuses_unvalidated_row_domain(
    calibrated_single_index_fit, unsupported_rows
):
    _, _, operator, _, _, _ = calibrated_single_index_fit
    selected = operator.accepted
    rows = replace(selected.components.score_rows, **unsupported_rows)
    components = replace(selected.components, score_rows=rows)
    assert (
        _indexed_poisson_score_bound(selected.fit, components, _TIGHT_OPERATOR_CONTROL)
        is None
    )


@pytest.fixture(
    scope="module",
    params=[
        (300, 907, 2),
        (300, 908, 2),
        (400, 909, 2),
        (512, 910, 3, "unbalanced"),
    ],
)
def calibrated_gaussian_scale_fit(request):
    return calibrated_gaussian_scale_case(*request.param)


def calibrated_gaussian_scale_case(
    p: int, seed: int, multiplicity: int, mode: str = "balanced"
):
    """Audit source scores at every trial and retain both complete fits."""
    n = p * multiplicity
    rng = np.random.default_rng(seed)
    if mode == "balanced":
        group = np.arange(n) % p
        signal = 1.2 + 0.4 * np.sin(group / 31.0)
    elif mode == "unbalanced":
        probabilities = 1.0 / (1.0 + np.arange(p)) ** 1.1
        probabilities /= np.sum(probabilities)
        group = np.concatenate(
            (np.arange(p), rng.choice(p, size=n - p, p=probabilities))
        )
        signal = 1.5 + 0.7 * np.sin(group / 31.0)
    else:
        raise ValueError(f"Unknown Gaussian calibration mode: {mode}")
    response = signal + rng.normal(0.0, 0.2, n)
    frame = pd.DataFrame(
        {
            "y": response,
            "g": pd.Categorical(
                [f"g{index}" for index in group],
                categories=[f"g{index}" for index in range(p)],
            ),
        }
    )
    formula = "y~0+s(g,bs='re')"
    spec = parse_formula(formula)
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(
            batch_rows=64, memory_budget_bytes=4 * 1024 * 1024
        ),
    )
    dense_setup = ModelSetup.build(spec, frame)
    family = Gaussian()
    errors = []
    bounds = []

    class _AuditedProvider(SourceOperatorEFSFitProvider):
        def evaluate(self, fit, control, *, score_phi=1.0):
            state = super().evaluate(fit, control, score_phi=score_phi)
            X = dense_setup.X
            beta = fit.coefficients
            y = frame.y.to_numpy(dtype=float)
            residual = y - X @ beta
            penalty = dense_setup.penalties.materialize(fit.log_lambda)
            H = X.T @ X + penalty
            logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(H))))
            penalty_logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(penalty))))
            saturated = float(
                np.asarray(family.saturated_loglik(y, np.ones(n), score_phi))
            )
            exact = (residual @ residual + beta @ penalty @ beta) / (2 * score_phi)
            exact += -saturated + (logdet - penalty_logdet) / 2
            errors.append(abs(state.components.score.value - exact))
            bounds.append(state.score_error_bound)
            return state

    provider = _AuditedProvider(
        model,
        source,
        family,
        fit_control=_TIGHT_OPERATOR_CONTROL,
        fit_budget_bytes=16 * 1024 * 1024,
        components_budget_bytes=32 * 1024 * 1024,
        aggregate_budget_bytes=128 * 1024 * 1024,
    )
    initial = np.log(np.array([0.8]))
    control = EFSControl(outer_limit=20, pirls_tolerance=1e-7)
    operator = run_operator_efs_gaussian_scale(
        provider,
        initial,
        control=control,
        initial_accuracy=OperatorLogDetControl(probes=4, lanczos_depth=4, seed=seed),
    )
    dense = dense_efs_unknown_scale(
        FittingData.from_setup(dense_setup, family),
        initial_log_lambda=initial,
        control=control,
    )
    return formula, frame, operator, dense, errors, bounds


def test_calibrated_gaussian_scale_complete_fit_matches_dense_efs(
    calibrated_gaussian_scale_fit,
):
    _, frame, operator, dense, errors, bounds = calibrated_gaussian_scale_fit
    assert operator.converged, (
        operator.stop_reason,
        operator.outer_iterations,
        operator.indeterminate_comparisons,
        errors,
        bounds,
    )
    assert dense.converged
    assert bool(np.asarray(dense.pirls_result.converged))
    assert operator.accepted.score_bound_method == "indexed_gaussian_empirical_v2"
    assert operator.outer_iterations == dense.n_iter
    assert operator.indeterminate_comparisons == 0
    assert operator.attempted_fits == len(errors)
    assert operator.stop_reason in {"score_window", "deviance_change"}
    assert all(bound is not None for bound in bounds)
    assert all(error <= bound for error, bound in zip(errors, bounds, strict=True))
    assert len(frame) >= 2 * operator.accepted.fit.model.n_coef
    np.testing.assert_allclose(
        operator.accepted.components.score.value,
        dense.score,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize(
    "unsupported_rows",
    [
        {"unit_weights": False},
        {"zero_offset": False},
        {"maximum_absolute_response": 11.0},
    ],
)
def test_gaussian_scale_envelope_refuses_unvalidated_rows(
    calibrated_gaussian_scale_fit, unsupported_rows
):
    _, _, operator, _, _, _ = calibrated_gaussian_scale_fit
    selected = operator.accepted
    rows = replace(selected.components.score_rows, **unsupported_rows)
    components = replace(selected.components, score_rows=rows)
    assert (
        _indexed_gaussian_score_bound(selected.fit, components, _TIGHT_OPERATOR_CONTROL)
        is None
    )


_DOMAIN = CALIBRATED_OPERATOR_EFS_DOMAIN
_METHODS = {
    "indexed_poisson": "indexed_poisson_empirical_v2",
    "indexed_gaussian": "indexed_gaussian_empirical_v2",
    "arrowhead_poisson": "arrowhead_poisson_empirical_v2",
}


def test_calibrated_domain_is_the_documented_v2_record():
    # These limits were measured with
    # scripts/benchmark_operator_efs_calibration.py; re-measure before
    # changing them.
    assert (
        OperatorEFSCalibratedDomain(
            minimum_levels=257,
            maximum_levels=4096,
            minimum_rows_per_level=1,
            maximum_rows_per_level=16,
            minimum_batch_rows=64,
            maximum_batch_rows=65536,
        )
        == _DOMAIN
    )


def test_released_bound_is_the_admitted_envelope_total(calibrated_single_index_fit):
    _, _, operator, _, _, _ = calibrated_single_index_fit
    selected = operator.accepted
    bound, method = admitted_operator_efs_score_bound(
        selected.fit, selected.components, _TIGHT_OPERATOR_CONTROL
    )
    envelope = operator_efs_score_envelope(selected.fit, selected.components)
    assert (bound, method) == (selected.score_error_bound, selected.score_bound_method)
    assert envelope is not None
    assert bound == envelope.total


def _matched_state(case: CalibrationCase, sp: float, score_phi: float = 1.0):
    """One operator state for a generated case and its decimal reference."""
    frame = generate_frame(case)
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula(case.formula),
        source,
        control=OperatorPreparationControl(
            batch_rows=case.batch_rows,
            memory_budget_bytes=64 << 20,
        ),
    )
    family = Gaussian() if case.regime == "indexed_gaussian" else Poisson()
    provider = SourceOperatorEFSFitProvider(
        model,
        source,
        family,
        fit_control=_TIGHT_OPERATOR_CONTROL,
        fit_budget_bytes=128 << 20,
        components_budget_bytes=256 << 20,
        aggregate_budget_bytes=512 << 20,
    )
    fit = provider.fit(np.log(np.array([sp])), None)
    assert fit.converged
    accuracy = OperatorLogDetControl(probes=4, lanczos_depth=4, seed=case.seed)
    state = (
        provider.evaluate(fit, accuracy)
        if family.scale_known
        else provider.evaluate(fit, accuracy, score_phi=score_phi)
    )
    reference = DecimalReference(
        case,
        frame,
        model.penalties.blocks[0].local_penalties[0].scale,
        len(model.modest_columns),
    )
    return state, reference


@pytest.mark.parametrize(
    ("regime", "levels", "rows_per_level", "batch_rows", "seed"),
    [
        (regime, _DOMAIN.maximum_levels, rows, batch, seed)
        for regime, seed in (
            ("indexed_poisson", 5101),
            ("indexed_gaussian", 5102),
            ("arrowhead_poisson", 5103),
        )
        for rows, batch in (
            (_DOMAIN.minimum_rows_per_level, _DOMAIN.minimum_batch_rows),
            (_DOMAIN.maximum_rows_per_level, _DOMAIN.maximum_batch_rows),
        )
    ],
)
def test_boundary_matched_state_error_stays_inside_admitted_bound(
    regime, levels, rows_per_level, batch_rows, seed
):
    case = CalibrationCase(
        regime,
        levels,
        float(rows_per_level),
        seed,
        "balanced",
        batch_rows,
        role="boundary",
    )
    phi = 0.04
    state, reference = _matched_state(case, sp=3.0, score_phi=phi)
    assert state.score_bound_method == _METHODS[regime]
    assert state.score_error_bound is not None
    rho = float(state.fit.log_lambda[0])
    own, total, _ = score_errors(
        reference,
        state.fit.coefficients,
        rho,
        state.components.score.value,
        state.score_phi,
    )
    assert own <= state.score_error_bound
    assert total <= state.score_error_bound
    # Statistics at the operator's own coefficients isolate source arithmetic;
    # the coefficient residual is the fit tolerance's, not the envelope's.
    at_fit = reference.evaluate(
        [Decimal(float(value)) for value in state.fit.coefficients],
        rho,
        state.score_phi,
    )
    checks = _AssertCollector()
    for name, actual, expected in (
        ("Fisher trace", state.components.fisher.traces[0], at_fit.trace),
        ("quadratic", state.components.fisher.quadratics[0], at_fit.quadratic),
        ("EDF", state.components.edf_total, at_fit.edf),
        ("deviance", state.fit.deviance, at_fit.deviance),
    ):
        checks.check(
            name,
            lambda actual=actual, expected=expected: np.testing.assert_allclose(
                actual, float(expected), rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    checks.raise_if_any(f"boundary {regime} levels={levels} rows={rows_per_level}")


@pytest.mark.parametrize(
    ("regime", "levels", "rows_per_level", "batch_rows", "mode"),
    [
        (
            "indexed_poisson",
            _DOMAIN.maximum_levels + 1,
            float(_DOMAIN.minimum_rows_per_level),
            _DOMAIN.maximum_batch_rows,
            "balanced",
        ),
        (
            "arrowhead_poisson",
            300,
            float(_DOMAIN.maximum_rows_per_level + 1),
            _DOMAIN.minimum_batch_rows,
            "balanced",
        ),
        (
            "indexed_gaussian",
            300,
            float(_DOMAIN.minimum_rows_per_level),
            _DOMAIN.minimum_batch_rows // 2,
            "balanced",
        ),
        # Inside the shape domain: Zipf rows put the diagonal condition ratio
        # (about 1.2e4) beyond the unchanged 1e4 conditioning gate.
        (
            "indexed_poisson",
            _DOMAIN.maximum_levels,
            float(_DOMAIN.maximum_rows_per_level),
            _DOMAIN.maximum_batch_rows,
            "unbalanced",
        ),
    ],
)
def test_outside_calibrated_domain_keeps_formula_but_refuses_bound(
    regime, levels, rows_per_level, batch_rows, mode
):
    case = CalibrationCase(
        regime,
        levels,
        rows_per_level,
        5201 if mode == "balanced" else 5101,
        mode,
        batch_rows,
        role="boundary",
    )
    state, _ = _matched_state(case, sp=3.0, score_phi=0.04)
    envelope = operator_efs_score_envelope(state.fit, state.components)
    assert envelope is not None
    assert state.score_error_bound is None
    assert state.score_bound_method == "unavailable"
    if mode == "unbalanced":
        assert _DOMAIN.admits(levels, state.fit.model.n_obs, batch_rows)
        assert state.components.score.diagonal_condition_ratio > 1e4


def _complete_case(case: CalibrationCase):
    """Complete operator and dense EFS with every operator score audited."""
    frame = generate_frame(case)
    source = DataFrameRowSource(frame, response="y")
    spec = parse_formula(case.formula)
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(
            batch_rows=case.batch_rows,
            memory_budget_bytes=64 << 20,
        ),
    )
    gaussian = case.regime == "indexed_gaussian"
    family = Gaussian() if gaussian else Poisson()
    reference = DecimalReference(
        case,
        frame,
        model.penalties.blocks[0].local_penalties[0].scale,
        len(model.modest_columns),
    )
    errors = []
    bounds = []

    class _AuditedProvider(SourceOperatorEFSFitProvider):
        def evaluate(self, fit, control, *, score_phi=1.0):
            state = super().evaluate(fit, control, score_phi=score_phi)
            _, total, _ = score_errors(
                reference,
                fit.coefficients,
                float(fit.log_lambda[0]),
                state.components.score.value,
                state.score_phi,
            )
            errors.append(total)
            bounds.append(state.score_error_bound)
            return state

    provider = _AuditedProvider(
        model,
        source,
        family,
        fit_control=_TIGHT_OPERATOR_CONTROL,
        fit_budget_bytes=128 << 20,
        components_budget_bytes=256 << 20,
        aggregate_budget_bytes=512 << 20,
    )
    control = EFSControl(outer_limit=50, pirls_tolerance=1e-7 if gaussian else 1e-11)
    initial = np.log(np.array([0.8]))
    accuracy = OperatorLogDetControl(probes=4, lanczos_depth=4, seed=case.seed)
    run = run_operator_efs_gaussian_scale if gaussian else run_operator_efs_known_scale
    operator = run(provider, initial, control=control, initial_accuracy=accuracy)
    dense_data = FittingData.from_setup(ModelSetup.build(spec, frame), family)
    dense_run = dense_efs_unknown_scale if gaussian else dense_efs_known_scale
    dense = dense_run(dense_data, initial_log_lambda=initial, control=control)
    return operator, dense, errors, bounds


@pytest.fixture(
    scope="module",
    params=[
        ("indexed_poisson", 1100, 5301, "balanced"),
        ("indexed_gaussian", 1100, 5302, "balanced"),
        ("arrowhead_poisson", 1100, 5303, "unbalanced"),
    ],
    ids=lambda param: param[0],
)
def beyond_p512_complete_fit(request):
    regime, levels, seed, mode = request.param
    rows = min(4, _DOMAIN.maximum_rows_per_level)
    case = CalibrationCase(
        regime,
        levels,
        float(rows),
        seed,
        mode,
        _DOMAIN.maximum_batch_rows,
        role="held_out",
    )
    return case, *_complete_case(case)


def test_beyond_p512_complete_operator_efs_matches_dense_efs(beyond_p512_complete_fit):
    case, operator, dense, errors, bounds = beyond_p512_complete_fit
    selected = operator.accepted
    assert operator.converged, (operator.stop_reason, errors, bounds)
    assert dense.converged
    assert bool(np.asarray(dense.pirls_result.converged))
    assert selected.score_bound_method == _METHODS[case.regime]
    assert operator.outer_iterations == dense.n_iter
    assert operator.indeterminate_comparisons == 0
    assert operator.attempted_fits == len(errors)
    assert all(bound is not None for bound in bounds)
    assert all(error <= bound for error, bound in zip(errors, bounds, strict=True))
    fields = [
        ("score", selected.components.score.value, dense.score),
        ("smoothing", np.exp(selected.fit.log_lambda), dense.smoothing_params),
        ("deviance", selected.fit.deviance, dense.pirls_result.deviance),
        ("coefficients", selected.fit.coefficients, dense.pirls_result.coefficients),
        ("EDF", selected.components.edf_total, np.sum(np.asarray(dense.edf))),
    ]
    if case.regime == "indexed_gaussian":
        fields.append(("scale", selected.reported_phi, dense.scale))
    checks = _AssertCollector()
    for name, actual, expected in fields:
        checks.check(
            name,
            lambda actual=actual, expected=expected: np.testing.assert_allclose(
                actual, np.asarray(expected), rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    checks.raise_if_any(f"beyond-p512 {case.regime} levels={case.levels}")


@pytest.mark.parametrize(
    "regime", ["indexed_poisson", "indexed_gaussian", "arrowhead_poisson"]
)
def test_envelope_requires_structured_finite_terms(regime):
    case = CalibrationCase(regime, 40, 2.0, 5501, role="boundary")
    state, _ = _matched_state(case, sp=3.0, score_phi=0.04)
    fit, components = state.fit, state.components
    assert operator_efs_score_envelope(fit, components) is not None
    assert state.score_error_bound is None
    unstructured = replace(
        components, score=replace(components.score, strategy="stochastic_lanczos")
    )
    assert operator_efs_score_envelope(fit, unstructured) is None
    stalled = replace(
        fit, diagnostics=replace(fit.diagnostics, final_gradient_norm=math.inf)
    )
    assert operator_efs_score_envelope(stalled, components) is None
    unbounded = replace(
        components,
        score_rows=replace(components.score_rows, maximum_row_l1_norm=math.inf),
    )
    assert (operator_efs_score_envelope(fit, unbounded) is None) == (
        regime == "arrowhead_poisson"
    )
