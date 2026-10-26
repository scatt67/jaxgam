"""Knot-aware workspace contracts for compact operator evaluation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAMPredictor
from jaxgam.data.source import DataFrameRowSource
from jaxgam.families.standard import Gaussian
from jaxgam.formula import operator_radial
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    _CheckedSource,
    prepare_operator_model,
)
from jaxgam.formula.operator_workspace import smooth_evaluator_workspace_bound
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.terms import SmoothSpec
from jaxgam.inference.operator_geometry import OperatorPredictionGeometry
from jaxgam.smooths.gaussian_process import GaussianProcessSmooth
from jaxgam.smooths.tprs import TPRSSmooth
from tests.tolerances import STRICT


def _frame(n: int = 61) -> pd.DataFrame:
    x = np.linspace(-1.3, 1.7, n)
    z = np.sin(np.linspace(-0.8, 2.2, n)) + np.linspace(0, 0.1, n)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "fac": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "y": np.cos(1.7 * x) + 0.2 * z,
        }
    )


def test_plain_workspace_and_row_validation() -> None:
    smooth = SimpleNamespace(n_coefs=4)
    assert smooth_evaluator_workspace_bound(smooth, 3) == 8 * 3 * (3 * 4 + 8)
    with pytest.raises(ValueError, match="nonnegative"):
        smooth_evaluator_workspace_bound(smooth, -1)


def test_unfrozen_radial_workspace_rejects_missing_knots() -> None:
    smooth = TPRSSmooth(SmoothSpec(["x"], bs="tp", k=6))
    with pytest.raises(RuntimeError, match="knot table"):
        smooth_evaluator_workspace_bound(smooth, 3)


@pytest.mark.parametrize(
    "expression",
    [
        "s(x,bs='tp',k=6)",
        "s(x,bs='ts',k=6)",
        "s(x,bs='gp',k=6)",
        "s(x,bs='gp',kernel='spherical',k=6)",
        "s(x,bs='gp',kernel='power_exponential',k=6)",
        "s(x,bs='gp',kernel='matern_5_2',k=6)",
        "s(x,bs='gp',kernel='matern_7_2',k=6)",
    ],
)
def test_radial_workspace_is_knot_aware_and_rejects_before_evaluation(
    expression: str, monkeypatch
) -> None:
    frame = _frame(251)
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~" + expression),
        source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    smooth = model.modest.predict_spec.coef_map.terms[-1].smooth
    assert smooth is not None
    knots = smooth._Xu if isinstance(smooth, TPRSSmooth) else smooth._knt
    assert knots is not None
    assert len(knots) == len(frame)
    assert model.evaluator_workspace_bound(7) >= 10 * 8 * 7 * len(knots)
    advertised = model.batch_workspace_bound(7)
    old_width_only_bound = advertised - model.evaluator_workspace_bound(7)
    assert old_width_only_bound == 4080
    expected = model.evaluate_batch(next(source.scan(7))).modest_values
    calls = 0
    original = type(smooth).predict_matrix

    def observed(self, data):
        nonlocal calls
        calls += 1
        return original(self, data)

    monkeypatch.setattr(type(smooth), "predict_matrix", observed)
    batch = next(source.scan(7))
    with pytest.raises(MemoryError, match="workspace"):
        model.evaluate_batch(batch, available_workspace_bytes=old_width_only_bound)
    with pytest.raises(MemoryError, match="workspace"):
        model.evaluate_batch(batch, available_workspace_bytes=advertised - 1)
    assert calls == 0
    actual = model.evaluate_batch(batch, available_workspace_bytes=advertised)
    np.testing.assert_allclose(
        actual.modest_values, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert calls == 1

    geometry = OperatorPredictionGeometry(
        model.modest.predict_spec,
        model.modest_columns,
        model.random_effects,
        model.n_coef,
        7,
        model.modest_raw_width,
        model.metadata_bound_bytes,
    )
    heldout = frame.iloc[:7]
    prediction_bound = geometry.workspace_bound(len(heldout))
    predictor = GAMPredictor(
        np.zeros(model.n_coef),
        None,
        Gaussian(),
        "y~" + expression,
        False,
        geometry.modest,
        _operator_geometry=geometry,
        _memory_budget_bytes=prediction_bound - 1,
        _output_budget_bytes=1024 * 1024,
    )
    with pytest.raises(MemoryError, match="workspace"):
        predictor.predict(heldout)
    assert calls == 1
    exact_predictor = GAMPredictor(
        np.zeros(model.n_coef),
        None,
        Gaussian(),
        "y~" + expression,
        False,
        geometry.modest,
        _operator_geometry=geometry,
        _memory_budget_bytes=prediction_bound,
        _output_budget_bytes=1024 * 1024,
    )
    np.testing.assert_array_equal(exact_predictor.predict(heldout), 0.0)
    assert calls == 2


def test_setup_stats_reject_radial_scratch_before_predictor(monkeypatch) -> None:
    frame = _frame(73)
    fitted_source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(x,bs='tp',k=6)"),
        fitted_source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    smooth = model.modest.predict_spec.coef_map.terms[-1].smooth
    assert isinstance(smooth, TPRSSmooth)
    B = 7
    required = smooth_evaluator_workspace_bound(smooth, B) + 8 * (
        2 * smooth.n_coefs + B * (2 * smooth.n_coefs + 2)
    )
    control = OperatorPreparationControl(batch_rows=B, memory_budget_bytes=required - 1)
    checked = _CheckedSource(
        DataFrameRowSource(frame, response="y"), control, reserved_bytes=0
    )
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        pytest.fail("Radial predictor ran before prospective setup rejection.")

    monkeypatch.setattr(TPRSSmooth, "predict_matrix", forbidden)
    with pytest.raises(MemoryError):
        operator_radial._stream_design_stats(smooth, checked, control, rms=True)
    assert calls == 0


@pytest.mark.parametrize(
    "expression",
    [
        "s(x,bs='tp',k=6,by=z)",
        "s(x,bs='gp',k=6,by=fac)",
        "te(x,z,bs='tp',k=4)",
        "te(x,z,bs='gp',k=4)",
    ],
)
def test_by_tensor_radial_workspace_rejects_before_any_margin(
    expression: str, monkeypatch
) -> None:
    frame = _frame()
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~" + expression),
        source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    advertised = model.batch_workspace_bound(7)
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        pytest.fail("A radial margin ran before the model workspace preflight.")

    monkeypatch.setattr(TPRSSmooth, "predict_matrix", forbidden)
    monkeypatch.setattr(GaussianProcessSmooth, "predict_matrix", forbidden)
    with pytest.raises(MemoryError, match="workspace"):
        model.evaluate_batch(
            next(source.scan(7)), available_workspace_bytes=advertised - 1
        )
    assert calls == 0
