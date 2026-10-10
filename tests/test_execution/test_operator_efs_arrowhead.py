"""Matched Poisson arrowhead score and Fisher statistics at large p."""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln, xlogy

from jaxgam.control import EFSControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs import _fit_state, dense_efs_known_scale
from jaxgam.execution.operator_efs_controller import (
    SourceOperatorEFSFitProvider,
    _arrowhead_poisson_score_bound,
    _raw_ratio,
    _trial_rho,
    run_operator_efs_known_scale,
)
from jaxgam.execution.operator_efs_provider import MatchedOperatorEFSProvider
from jaxgam.execution.operator_fit import fit_operator_fixed_sp
from jaxgam.execution.operator_logdet import OperatorLogDetControl
from jaxgam.families.standard import Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.efs import efs_raw_update, prepare_efs_statistics
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


@pytest.mark.parametrize(
    ("levels", "seed", "unbalanced", "sp"),
    [(300, 31031, False, 0.8), (300, 31032, False, 9.75), (512, 31033, True, 1.1)],
)
def test_matched_arrowhead_score_and_trace_match_independent_dense_state(
    levels, seed, unbalanced, sp
):
    n = 2 * levels if not unbalanced else 3 * levels
    rng = np.random.default_rng(seed)
    if unbalanced:
        probabilities = 1.0 / (1.0 + np.arange(levels)) ** 1.1
        probabilities /= np.sum(probabilities)
        group = np.concatenate(
            (np.arange(levels), rng.choice(levels, n - levels, p=probabilities))
        )
    else:
        group = np.arange(n) % levels
    x = np.linspace(-0.9, 1.1, n)
    mean = np.exp(0.2 + 0.5 * x + 0.3 * np.sin(group / 31.0))
    y = rng.poisson(mean).astype(float)
    frame = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "g": pd.Categorical(
                [f"g{index}" for index in group],
                categories=[f"g{index}" for index in range(levels)],
            ),
        }
    )
    spec = parse_formula("y~x+s(g,bs='re')")
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(
            batch_rows=64, memory_budget_bytes=4 * 1024 * 1024
        ),
    )
    fitted = fit_operator_fixed_sp(
        model,
        source,
        Poisson(),
        np.array([sp]),
        control=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
        memory_budget_bytes=16 * 1024 * 1024,
    )
    assert fitted.converged
    matched = MatchedOperatorEFSProvider.from_fit(
        fitted,
        source,
        score_phi=1.0,
        memory_budget_bytes=32 * 1024 * 1024,
    ).evaluate()
    dense_setup = ModelSetup.build(spec, frame)
    X = dense_setup.X
    beta = fitted.coefficients
    eta = X @ beta
    mu = np.exp(eta)
    penalty = dense_setup.penalties.materialize(fitted.log_lambda)
    hessian = X.T @ (mu[:, None] * X) + penalty
    inverse = np.linalg.inv(hessian)
    deviance = 2 * np.sum(xlogy(y, y / mu) - (y - mu))
    saturated = np.sum(xlogy(y, y) - y - gammaln(y + 1))
    logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(hessian))))
    re_block = model.penalties.blocks[-1]
    re_scale = re_block.local_penalties[0].scale
    penalty_logdet = levels * math.log(sp * re_scale)
    expected_score = (
        (deviance + beta @ penalty @ beta) / 2
        - saturated
        + (logdet - penalty_logdet) / 2
        - math.log(2 * math.pi)
    )
    expected_trace = re_scale * np.trace(inverse[re_block.start :, re_block.start :])
    np.testing.assert_allclose(
        matched.score.value, expected_score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        matched.fisher.traces, [expected_trace], rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert matched.score.strategy == "structured_arrowhead"
    assert matched.score.spectral_lower_estimate is not None
    assert matched.score.spectral_condition_estimate is not None
    assert matched.fisher.diagnostics.strategy == "structured_arrowhead"
    assert matched.source_scans == 5


def calibrated_coupled_frame(
    levels: int, seed: int, multiplicity: int, mode: str = "balanced"
) -> pd.DataFrame:
    """Reproducible bounded Poisson rows for coupled EFS validation."""
    n = levels * multiplicity
    rng = np.random.default_rng(seed)
    if mode == "balanced":
        group = np.arange(n) % levels
    elif mode == "unbalanced":
        probabilities = 1.0 / (1.0 + np.arange(levels)) ** 1.1
        probabilities /= np.sum(probabilities)
        group = np.concatenate(
            (np.arange(levels), rng.choice(levels, n - levels, p=probabilities))
        )
    else:
        raise ValueError("Unknown coupled EFS calibration mode.")
    x = np.linspace(-1, 1, n)
    mean = np.exp(0.2 + 0.5 * x + 0.3 * np.sin(group / 31.0))
    response = rng.poisson(mean)
    if mode == "unbalanced":
        response[0] = 31
    return pd.DataFrame(
        {
            "y": response,
            "x": x,
            "g": pd.Categorical(
                [f"g{index}" for index in group],
                categories=[f"g{index}" for index in range(levels)],
            ),
        }
    )


def test_unbalanced_arrowhead_initial_shift_matches_dense_efs_state():
    frame = calibrated_coupled_frame(512, 32033, 3, "unbalanced")
    spec = parse_formula("y~x+s(g,bs='re')")
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(batch_rows=64, memory_budget_bytes=4 << 20),
    )
    dense_setup = ModelSetup.build(spec, frame)
    dense_data = FittingData.from_setup(dense_setup, Poisson())
    control = EFSControl(outer_limit=20, pirls_tolerance=1e-11)
    rho = np.log(np.array([0.8])) + 2.5
    dense = _fit_state(
        dense_data,
        prepare_efs_statistics(dense_data),
        rho,
        np.zeros(dense_data.n_coef),
        control,
        regular_start_present=False,
    )
    provider = SourceOperatorEFSFitProvider(
        model,
        source,
        Poisson(),
        fit_control=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
        fit_budget_bytes=16 << 20,
        components_budget_bytes=32 << 20,
        aggregate_budget_bytes=128 << 20,
    )
    operator = provider.evaluate(
        provider.fit(rho, None), OperatorLogDetControl(probes=4, lanczos_depth=4)
    )
    assert dense.valid
    assert operator.score_error_bound is not None
    checks = _AssertCollector()
    for name, actual, expected in (
        ("coefficients", operator.fit.coefficients, dense.pirls_result.coefficients),
        ("deviance", operator.fit.deviance, dense.pirls_result.deviance),
        ("score", operator.components.score.value, dense.score),
        (
            "determinant derivative",
            operator.components.penalty.derivative,
            dense.statistics.determinant_derivative,
        ),
        (
            "Fisher trace",
            operator.components.fisher.traces,
            dense.statistics.fisher_trace,
        ),
        (
            "quadratic",
            operator.components.fisher.quadratics,
            dense.statistics.quadratic,
        ),
    ):
        checks.check(
            name,
            lambda actual=actual, expected=expected: np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    checks.raise_if_any("unbalanced arrowhead first shifted state")
    dense_update = efs_raw_update(
        rho,
        dense.statistics,
        np.array(1.0),
        np.array(1.0),
        np.array(control.log_lambda_max),
    )
    operator_trial_rho = _trial_rho(
        operator, _raw_ratio(operator), 1.0, control.log_lambda_max
    )
    np.testing.assert_allclose(
        operator_trial_rho,
        dense_update.log_smoothing_trial,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    dense_trial = _fit_state(
        dense_data,
        prepare_efs_statistics(dense_data),
        dense_update.log_smoothing_trial,
        dense.pirls_result.coefficients,
        control,
    )
    operator_trial = provider.evaluate(
        provider.fit(operator_trial_rho, operator.fit.coefficients),
        OperatorLogDetControl(probes=4, lanczos_depth=4),
    )
    assert dense_trial.valid
    np.testing.assert_allclose(
        operator_trial.components.score.value,
        dense_trial.score,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def calibrated_coupled_case(
    levels: int, seed: int, multiplicity: int, mode: str = "balanced"
):
    """Run complete modest-indexed EFS and audit every matched dense score."""
    frame = calibrated_coupled_frame(levels, seed, multiplicity, mode)
    formula = "y~x+s(g,bs='re')"
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
    errors = []
    bounds = []

    class _AuditedProvider(SourceOperatorEFSFitProvider):
        def evaluate(self, fit, control):
            state = super().evaluate(fit, control)
            X = dense_setup.X
            beta = fit.coefficients
            mu = np.exp(X @ beta)
            y = frame.y.to_numpy(dtype=np.float64)
            deviance = 2 * np.sum(xlogy(y, y / mu) - (y - mu))
            saturated = np.sum(xlogy(y, y) - y - gammaln(y + 1))
            penalty = dense_setup.penalties.materialize(fit.log_lambda)
            hessian = X.T @ (mu[:, None] * X) + penalty
            logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(hessian))))
            re_block = model.penalties.blocks[0]
            penalty_scale = re_block.local_penalties[0].scale
            penalty_logdet = levels * math.log(
                math.exp(float(fit.log_lambda[0])) * penalty_scale
            )
            exact = (
                (deviance + beta @ penalty @ beta) / 2
                - saturated
                + (logdet - penalty_logdet) / 2
                - math.log(2 * math.pi)
            )
            errors.append(abs(state.components.score.value - exact))
            bounds.append(state.score_error_bound)
            return state

    provider = _AuditedProvider(
        model,
        source,
        Poisson(),
        fit_control=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
        fit_budget_bytes=16 * 1024 * 1024,
        components_budget_bytes=32 * 1024 * 1024,
        aggregate_budget_bytes=128 * 1024 * 1024,
    )
    initial = np.log(np.array([0.8]))
    control = EFSControl(outer_limit=20, pirls_tolerance=1e-11)
    operator = run_operator_efs_known_scale(
        provider,
        initial,
        control=control,
        initial_accuracy=OperatorLogDetControl(probes=4, lanczos_depth=4, seed=seed),
    )
    dense = dense_efs_known_scale(
        dense_data, initial_log_lambda=initial, control=control
    )
    return formula, frame, operator, dense, errors, bounds


def test_arrowhead_calibration_refuses_invalid_condition_and_weights():
    frame = calibrated_coupled_frame(300, 32031, 2)
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~x+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=64, memory_budget_bytes=4 << 20),
    )
    control = OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15)
    provider = SourceOperatorEFSFitProvider(
        model,
        source,
        Poisson(),
        fit_control=control,
        fit_budget_bytes=16 << 20,
        components_budget_bytes=32 << 20,
        aggregate_budget_bytes=128 << 20,
    )
    state = provider.evaluate(
        provider.fit(np.log(np.array([0.8])), None),
        OperatorLogDetControl(probes=4, lanczos_depth=4),
    )
    assert state.score_error_bound is not None
    score = replace(state.components.score, spectral_condition_estimate=1e6)
    assert (
        _arrowhead_poisson_score_bound(
            state.fit, replace(state.components, score=score), control
        )
        is None
    )
    rows = replace(state.components.score_rows, unit_weights=False)
    assert (
        _arrowhead_poisson_score_bound(
            state.fit, replace(state.components, score_rows=rows), control
        )
        is None
    )
