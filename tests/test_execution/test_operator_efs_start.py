"""Source initial.spg reduction for the indexed operator EFS regime."""

import numpy as np
import pandas as pd
import pytest

from jaxgam.api import GAM
from jaxgam.control import FitControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs import efs_initial_log_lambda
from jaxgam.execution.operator_efs_start import prepare_indexed_operator_efs_start
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from tests.tolerances import STRICT


@pytest.mark.parametrize("family", [Gaussian(), Poisson()])
def test_indexed_source_initial_spg_matches_dense_at_p_above_coordinate_cap(family):
    p, n = 300, 600
    rng = np.random.default_rng(1409)
    group = np.arange(n) % p
    response = (
        1.2 + 0.4 * np.sin(group / 31) + rng.normal(0, 0.2, n)
        if isinstance(family, Gaussian)
        else rng.poisson(np.exp(0.2 + 0.3 * np.sin(group / 31)))
    )
    frame = pd.DataFrame(
        {
            "y": response,
            "g": pd.Categorical(
                [f"g{index}" for index in group],
                categories=[f"g{index}" for index in range(p)],
            ),
        }
    )
    spec = parse_formula("y~0+s(g,bs='re')")
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(batch_rows=64, memory_budget_bytes=4 << 20),
    )
    start = prepare_indexed_operator_efs_start(
        model, source, family, memory_budget_bytes=4 << 20
    )
    expected = efs_initial_log_lambda(ModelSetup.build(spec, frame), family)
    np.testing.assert_allclose(
        start.log_lambda, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert start.source_scans == 1
    assert start.source_batches == 10
    assert start.peak_bound_bytes < 4 << 20


def test_indexed_start_preflights_before_source_replay():
    p = 257
    frame = pd.DataFrame(
        {
            "y": np.ones(2 * p),
            "g": pd.Categorical(np.arange(2 * p) % p),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~0+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=64, memory_budget_bytes=4 << 20),
    )
    with pytest.raises(MemoryError, match="start exceeds"):
        prepare_indexed_operator_efs_start(
            model, source, Poisson(), memory_budget_bytes=8192
        )


def test_public_indexed_operator_efs_retains_score_and_prediction():
    p, n = 300, 600
    rng = np.random.default_rng(37)
    group = np.arange(n) % p
    response = rng.poisson(np.exp(0.2 + 0.3 * np.sin(group / 31)))
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
    fitted = GAM(
        formula,
        family="poisson",
        optimizer="efs",
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=64,
            memory_budget_bytes=128 << 20,
            operator=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
        ),
    ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
    assert fitted.converged
    assert fitted.lambda_strategy == "efs_reml"
    assert fitted.execution_route == "operator"
    assert fitted.score is not None
    assert fitted.score_error_bound is not None
    assert fitted.score_scale == fitted.scale == 1.0
    assert fitted.edf_total is not None
    assert fitted.source_scans is not None
    assert fitted.source_scans > 1
    assert fitted.batches_scanned is not None
    assert fitted.batches_scanned > 10
    assert fitted.known_workspace_bytes < 128 << 20
    np.testing.assert_allclose(
        fitted.predict(frame.iloc[:10]),
        np.exp(
            ModelSetup.build(parse_formula(formula), frame).X[:10] @ fitted.coefficients
        ),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize(("family", "batch_rows"), [("gaussian", 64), ("poisson", 32)])
def test_public_operator_efs_rejects_unreleased_regime_before_source_replay(
    family, batch_rows
):
    frame = pd.DataFrame({"y": [1.0], "g": pd.Categorical(["g0"])})

    class _UnreadableSource(DataFrameRowSource):
        def scan(self, _batch_rows):
            pytest.fail("Unsupported operator EFS scanned the source")

        def fingerprint(self):
            pytest.fail("Unsupported operator EFS fingerprinted the source")

    source = _UnreadableSource(frame, response="y")
    model = GAM(
        "y~0+s(g,bs='re')",
        family=family,
        optimizer="efs",
        control=FitControl(
            execution="operator", linear_solver="pcg", batch_rows=batch_rows
        ),
    )
    with pytest.raises(NotImplementedError, match=r"Calibrated|supports calibrated"):
        model.fit(source, result="prediction")
