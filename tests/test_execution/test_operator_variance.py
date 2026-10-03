"""Selected Fisher uncertainty against pinned mgcv and direct small systems."""

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator import FrozenScanSystem, PCGControl
from jaxgam.execution.operator_variance import (
    OperatorVarianceSolveError,
    SelectedOperatorVariance,
)
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.inference.operator_geometry import OperatorPredictionGeometry
from jaxgam.penalties.structure import PenaltyStructure
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_selected_fisher_variance_matches_pinned_r_and_dense_covariance(
    r_bridge, family
):
    row = np.arange(47)
    x = np.linspace(0.1, 1.4, len(row))
    group = pd.Categorical(np.resize(["a", "b", "c", "d"], len(row)))
    response = (
        1.3 + np.sin(2.0 * x) + 0.1 * np.cos(row)
        if family == "gaussian"
        else np.resize([1, 3, 2, 4, 2, 5], len(row))
    )
    frame = pd.DataFrame({"y": response, "x": x, "g": group})
    formula = "y~s(x,bs='cr',k=6)+s(g,bs='re')"
    reference = r_bridge.fit_fixed_sp(
        formula,
        frame,
        family,
        np.array([0.7, 1.1]),
        with_uncertainty=True,
    )
    selected = frame.iloc[[0, 2, 13, 32, 46]].copy()
    collector = _AssertCollector()
    for B in (1, 11):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            parse_formula(formula),
            source,
            control=OperatorPreparationControl(batch_rows=B),
        )
        curvature = (
            None
            if family == "gaussian"
            else lambda batch, mu=reference["fitted_values"]: mu[batch.row_positions]
        )
        system = FrozenScanSystem(
            model,
            source,
            np.log(np.array([0.7, 1.1])),
            curvature=curvature,
            memory_budget_bytes=8 * 1024 * 1024,
        )
        provider = SelectedOperatorVariance(
            system,
            scale=reference["scale"],
            memory_budget_bytes=16 * 1024 * 1024,
        )
        result = provider.variance(selected)
        design = provider.geometry.build_predict_matrix(selected)
        direct = np.einsum("ij,jk,ik->i", design, reference["Vp"], design)
        collector.check(
            f"{family}/B{B}/R covariance",
            lambda result=result, direct=direct: np.testing.assert_allclose(
                result.variance, direct, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        collector.check(
            f"{family}/B{B}/R predict.gam SE",
            lambda result=result: np.testing.assert_allclose(
                np.sqrt(result.variance),
                reference["link_se"][[0, 2, 13, 32, 46]],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        assert result.diagnostics.completed_rows == len(selected)
        assert result.diagnostics.normal_actions == result.diagnostics.source_scans
        assert not result.variance.flags.writeable
    collector.raise_if_any("Selected operator Fisher uncertainty")


def test_selected_large_symbolic_variance_has_no_coefficient_square(monkeypatch):
    n = 97
    row = np.arange(n)
    g = row % 3
    h = row % 4
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.1 * np.sin(row),
            "g": pd.Categorical(
                [f"g{i}" for i in g], categories=[f"g{i}" for i in range(1024)]
            ),
            "h": pd.Categorical(
                [f"h{i}" for i in h], categories=[f"h{i}" for i in range(1024)]
            ),
        }
    )
    formula = "y~s(g,bs='re')+s(h,bs='re')"
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula(formula),
        source,
        control=OperatorPreparationControl(batch_rows=13),
    )
    assert model.n_coef == 2049
    system = FrozenScanSystem(
        model,
        source,
        np.log(np.array([0.8, 1.1])),
        memory_budget_bytes=8 * 1024 * 1024,
    )
    selected = frame.iloc[[0, 11, 47]].copy()
    provider = SelectedOperatorVariance(
        system,
        scale=1.0,
        memory_budget_bytes=16 * 1024 * 1024,
    )
    full_design = provider.geometry.build_predict_matrix(selected)
    active = np.zeros((n, 8))
    active[:, 0] = 1.0
    active[row, 1 + g] = 1.0
    active[row, 4 + h] = 1.0
    penalty = np.diag(
        [0.0]
        + [0.8 * model.random_effects[0].penalty_scale] * 3
        + [1.1 * model.random_effects[1].penalty_scale] * 4
    )
    small_covariance = np.linalg.inv(active.T @ active + penalty)
    selected_active = full_design[:, np.r_[0:4, 1025:1029]]
    expected = np.einsum(
        "ij,jk,ik->i", selected_active, small_covariance, selected_active
    )
    assert 8 * model.n_coef**2 > 16 * 1024 * 1024
    monkeypatch.setattr(
        PenaltyStructure,
        "materialize",
        lambda *_a, **_k: pytest.fail("Global penalty materialized"),
    )
    monkeypatch.setattr(
        OperatorPredictionGeometry,
        "build_predict_matrix",
        lambda *_a: pytest.fail("Expanded prediction matrix materialized"),
    )
    result = provider.variance(selected)
    np.testing.assert_allclose(
        result.variance, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert result.diagnostics.peak_bound_bytes < 16 * 1024 * 1024
    assert result.diagnostics.normal_actions == result.diagnostics.source_scans


def test_selected_variance_preflights_rows_and_budget_before_any_solve(monkeypatch):
    row = np.arange(29)
    frame = pd.DataFrame(
        {"y": 1.0 + row / 29, "g": pd.Categorical(np.resize(["a", "b"], 29))}
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    system = FrozenScanSystem(
        model, source, np.array([0.0]), memory_budget_bytes=1024 * 1024
    )
    monkeypatch.setattr(
        system, "solve", lambda *_a, **_k: pytest.fail("Fisher solve started")
    )
    capped = SelectedOperatorVariance(
        system, scale=1, memory_budget_bytes=2 * 1024 * 1024, max_selected_rows=2
    )
    with pytest.raises(ValueError, match="row cap"):
        capped.variance(frame.iloc[:3])
    underbudgeted = SelectedOperatorVariance(
        system, scale=1, memory_budget_bytes=system.memory_budget_bytes + 1
    )
    with pytest.raises(MemoryError, match="requires"):
        underbudgeted.variance(frame.iloc[:1])


def test_selected_variance_preserves_failed_pcg_status():
    row = np.arange(39)
    frame = pd.DataFrame(
        {
            "y": 1 + 0.1 * row,
            "x": np.linspace(0, 1, len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~x+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    system = FrozenScanSystem(
        model, source, np.array([0.0]), memory_budget_bytes=1024 * 1024
    )
    provider = SelectedOperatorVariance(
        system,
        scale=1,
        memory_budget_bytes=2 * 1024 * 1024,
        control=PCGControl(rtol=1e-14, max_iterations=1),
    )
    with pytest.raises(OperatorVarianceSolveError) as failure:
        provider.variance(frame.iloc[[0]])
    assert failure.value.row == 0
    assert failure.value.iterations == 1
    assert system.diagnostics.normal_actions == system.diagnostics.completed_scans - 1
