"""Independent checks of the operator EFS decimal calibration reference."""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln, xlogy

from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests import operator_efs_calibration as harness
from tests.helpers import _AssertCollector
from tests.test_execution.test_operator_efs_arrowhead import calibrated_coupled_frame
from tests.tolerances import STRICT


def _dense_state(case, frame, beta, rho, phi):
    """Independent float64 dense REML pieces for a small generated model."""
    setup = ModelSetup.build(parse_formula(case.formula), frame)
    X = setup.X
    y = frame["y"].to_numpy(dtype=np.float64)
    penalty = setup.penalties.materialize(np.array([rho]))
    eta = X @ beta
    k = X.shape[1] - case.levels
    if case.regime == "indexed_gaussian":
        weights = np.ones_like(y)
        residual = y - eta
        deviance = residual @ residual
        saturated = -len(y) * math.log(2 * math.pi * phi) / 2
    else:
        weights = np.exp(eta)
        deviance = 2 * np.sum(xlogy(y, y / weights) - (y - weights))
        saturated = np.sum(xlogy(y, y) - y - gammaln(y + 1))
    hessian = X.T @ (weights[:, None] * X) + penalty
    block = penalty[k:, k:]
    log_det = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(hessian))))
    penalty_log_det = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(block))))
    score = (
        (deviance + beta @ penalty @ beta) / (2 * phi)
        - saturated
        + (log_det - penalty_log_det) / 2
        - k * math.log(2 * math.pi) / 2
    )
    inverse = np.linalg.inv(hessian)
    scale = block[0, 0] / math.exp(rho)
    trace = scale * np.trace(inverse[k:, k:])
    return {
        "score": score,
        "deviance": deviance,
        "trace": trace,
        "quadratic": scale * beta[k:] @ beta[k:],
        "edf": X.shape[1] - math.exp(rho) * trace,
        "scale": scale,
        "k": k,
        "X": X,
        "y": y,
        "penalty": penalty,
    }


_MODEST_COUNT = {"none": 0, "intercept": 1, "x": 2}
_REFERENCE_CASES = [
    ("indexed_poisson", None),
    ("indexed_gaussian", None),
    ("arrowhead_poisson", "x"),
    ("arrowhead_poisson", "intercept"),
]


@pytest.mark.parametrize(("regime", "modest"), _REFERENCE_CASES)
def test_decimal_reference_matches_dense_float64_state(regime, modest):
    case = harness.CalibrationCase(regime, 12, 3.0, 5, "unbalanced", modest=modest)
    frame = harness.generate_frame(case)
    rng = np.random.default_rng(17)
    rho, phi = 0.3, 0.7 if regime == "indexed_gaussian" else 1.0
    n_coef = case.levels + _MODEST_COUNT[case.modest]
    beta = 0.2 * rng.standard_normal(n_coef)
    dense = _dense_state(case, frame, beta, rho, phi)
    reference = harness.DecimalReference(case, frame, dense["scale"], dense["k"])
    state = reference.evaluate([Decimal(float(value)) for value in beta], rho, phi)
    checks = _AssertCollector()
    for name in ("score", "deviance", "trace", "quadratic", "edf"):
        checks.check(
            name,
            lambda name=name: np.testing.assert_allclose(
                float(getattr(state, name)),
                dense[name],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
    checks.raise_if_any(f"decimal reference {regime}")


@pytest.mark.parametrize(("regime", "modest"), _REFERENCE_CASES)
def test_decimal_optimum_solves_dense_penalized_likelihood(regime, modest):
    case = harness.CalibrationCase(regime, 10, 2.0, 9, modest=modest)
    frame = harness.generate_frame(case)
    rho = -0.4
    n_coef = case.levels + _MODEST_COUNT[case.modest]
    dense = _dense_state(case, frame, np.zeros(n_coef), rho, 1.0)
    reference = harness.DecimalReference(case, frame, dense["scale"], dense["k"])
    optimum = np.array(
        [float(value) for value in reference.optimum(np.zeros(n_coef), rho)]
    )
    X, y, penalty = dense["X"], dense["y"], dense["penalty"]
    beta = np.zeros(n_coef)
    for _ in range(100):
        mean = X @ beta if regime == "indexed_gaussian" else np.exp(X @ beta)
        weight = np.ones_like(y) if regime == "indexed_gaussian" else mean
        gradient = X.T @ (mean - y) + penalty @ beta
        step = np.linalg.solve(X.T @ (weight[:, None] * X) + penalty, gradient)
        beta = beta - step
        if np.max(np.abs(step)) == 0:
            break
    np.testing.assert_allclose(optimum, beta, rtol=STRICT.rtol, atol=STRICT.atol)


def test_generator_reproduces_released_coupled_calibration_rows():
    for levels, seed, multiplicity, mode in (
        (300, 32031, 2, "balanced"),
        (512, 32033, 3, "unbalanced"),
    ):
        case = harness.CalibrationCase(
            "arrowhead_poisson", levels, float(multiplicity), seed, mode
        )
        pd.testing.assert_frame_equal(
            harness.generate_frame(case),
            calibrated_coupled_frame(levels, seed, multiplicity, mode),
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"regime": "unknown"},
        {"mode": "skewed"},
        {"multiplicity": 0.5},
        {"role": "tuning"},
        {"levels": 1},
        {"modest": "x"},
    ],
)
def test_calibration_case_rejects_invalid_requests(arguments):
    values = {"regime": "indexed_poisson", "levels": 8, "multiplicity": 2.0, "seed": 1}
    values.update(arguments)
    with pytest.raises(ValueError, match=r"Unknown|invalid|at least one row"):
        harness.CalibrationCase(**values)
