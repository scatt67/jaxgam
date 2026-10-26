"""Complete calibrated single-index EFS fits against dense and pinned mgcv."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln, xlogy

from jaxgam.control import EFSControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs import dense_efs_known_scale, dense_efs_unknown_scale
from jaxgam.execution.operator_efs_controller import (
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
    assert selected.score_bound_method == "indexed_poisson_empirical_v1"
    assert operator.stop_reason == "score_window"
    assert operator.outer_iterations == dense.n_iter
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
    assert operator.accepted.score_bound_method == "indexed_gaussian_empirical_v1"
    assert operator.outer_iterations == dense.n_iter
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
