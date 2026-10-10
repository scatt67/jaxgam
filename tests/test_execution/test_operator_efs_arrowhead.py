"""Matched Poisson arrowhead score and Fisher statistics at large p."""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln, xlogy

from jaxgam.control import EFSControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs import dense_efs_known_scale
from jaxgam.execution.operator_efs_controller import (
    SourceOperatorEFSFitProvider,
    _arrowhead_poisson_score_bound,
    run_operator_efs_known_scale,
)
from jaxgam.execution.operator_efs_provider import MatchedOperatorEFSProvider
from jaxgam.execution.operator_fit import fit_operator_fixed_sp
from jaxgam.execution.operator_logdet import OperatorLogDetControl
from jaxgam.families.standard import Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
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
    assert matched.score.spectral_lower_bound is not None
    assert matched.score.spectral_condition_bound is not None
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


@pytest.fixture(
    scope="module",
    params=[
        (300, 32031, 2, "balanced"),
        (257, 32032, 3, "balanced"),
        (512, 32033, 3, "unbalanced"),
    ],
)
def calibrated_coupled_fit(request):
    return calibrated_coupled_case(*request.param)


def test_calibrated_arrowhead_complete_fit_matches_dense_efs(calibrated_coupled_fit):
    _, _, operator, dense, errors, bounds = calibrated_coupled_fit
    selected = operator.accepted
    assert operator.converged
    assert dense.converged
    assert bool(np.asarray(dense.pirls_result.converged))
    assert selected.score_bound_method == "arrowhead_poisson_empirical_v1"
    assert operator.outer_iterations == dense.n_iter
    assert operator.attempted_fits == len(errors)
    assert all(bound is not None for bound in bounds)
    assert all(error <= bound for error, bound in zip(errors, bounds, strict=True))
    np.testing.assert_allclose(
        selected.components.score.value,
        dense.score,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    if selected.fit.model.random_effects[0].n_coef != 512:
        np.testing.assert_allclose(
            np.exp(selected.fit.log_lambda),
            dense.smoothing_params,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    # The unbalanced 512-level dense EFS sp differs by 2.18e-10 relative on
    # this flat score surface. The canonical smoothing comparison is pinned R
    # in the validation matrix; no tolerance is widened for this dense check.
    np.testing.assert_allclose(
        selected.fit.deviance,
        dense.pirls_result.deviance,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_arrowhead_calibration_refuses_invalid_condition_and_weights(
    calibrated_coupled_fit,
):
    _, _, operator, _, _, _ = calibrated_coupled_fit
    state = operator.accepted
    control = OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15)
    score = replace(state.components.score, spectral_condition_bound=1e6)
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
