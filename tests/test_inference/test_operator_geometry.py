"""Shared public prediction using bounded operator geometry."""

import pickle

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, FitControl, GAMPredictionResult, OperatorControl
from jaxgam.data.source import ArrayRowSource, DataFrameRowSource
from jaxgam.families.standard import Gaussian
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator.prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.inference.operator_geometry import OperatorPredictionGeometry
from jaxgam.inference.predictor import GAMPredictor
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


@pytest.fixture
def operator_prediction():
    x = np.linspace(-1, 1, 43)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": np.sin(x),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(x))),
            "h": pd.Categorical(np.resize(["u", "u", "v"], len(x))),
        }
    )
    offset = 0.01 * x
    formula = "y~s(x,bs='cr',k=6)+s(g,h,bs='re')"
    source = DataFrameRowSource(frame, response="y", offset=offset)
    fitted = GAM(
        formula,
        sp=[0.3, 0.8],
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=7,
            memory_budget_bytes=8 * 1024 * 1024,
        ),
    ).fit(source, result="prediction")
    return fitted, frame, offset, ModelSetup.build(parse_formula(formula), frame)


def test_public_operator_result_shared_prediction_offsets_pickle_and_lineage(
    operator_prediction,
):
    fitted, frame, offset, dense = operator_prediction
    assert isinstance(fitted, GAMPredictionResult)
    assert fitted.converged
    assert fitted.score is None
    assert fitted.scale is None
    assert fitted.edf_total is None
    assert fitted.execution_route == "operator"
    assert fitted.lambda_strategy == "fixed"
    assert fitted.operator_diagnostics is not None
    predictor = fitted.to_predictor()
    assert predictor._fisher_factor is None
    assert predictor.Vp is None
    assert predictor._operator_geometry is not None
    collector = _AssertCollector()
    expected = dense.X @ fitted.coefficients + offset
    collector.check(
        "matrix",
        lambda: np.testing.assert_allclose(
            fitted.predict_matrix(frame), dense.X, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    for owner in (
        fitted,
        predictor,
        pickle.loads(pickle.dumps(fitted)),
        pickle.loads(pickle.dumps(predictor)),
    ):
        assert not owner.coefficients.flags.writeable
        collector.check(
            "response",
            lambda owner=owner: np.testing.assert_allclose(
                owner.predict(frame, offset=offset),
                expected,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        with pytest.warns(UserWarning, match="external offset"):
            result = owner.predict(frame, pred_type="link")
        collector.check(
            "missing external offset",
            lambda result=result: np.testing.assert_allclose(
                result,
                dense.X @ fitted.coefficients,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        with pytest.raises(ValueError, match="offset"):
            owner.predict(frame, offset=np.zeros(2))
        with pytest.raises(ValueError, match="finite"):
            owner.predict(frame, offset=np.full(len(frame), np.nan))
        with pytest.raises(ValueError, match="pred_type"):
            owner.predict(frame, pred_type="invalid")
        # The operator route keeps no Fisher factor, and refuses the
        # FitControl.uncertainty settings that would: the error must point at
        # what does exist, not at those settings.
        with pytest.raises(RuntimeError, match="operator route") as refused:
            owner.predict(frame, se_fit=True)
        assert "bulk_link_variance" in str(refused.value)
        assert "FitControl(uncertainty" not in str(refused.value)
        selection = np.array([17, 2, 2, 0, 35, 1])
        source = ArrayRowSource(
            dict(frame.items()), offset=offset, row_selection=selection
        )
        for B in (1, 4, 9):
            batches = list(owner.predict_iter(source, B))
            assert np.array_equal(
                np.concatenate([positions for positions, _ in batches]), selection
            )
            collector.check(
                f"B{B}/selected rows",
                lambda batches=batches, selection=selection: np.testing.assert_allclose(
                    np.concatenate([values for _, values in batches]),
                    expected[selection],
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
    with pytest.raises(RuntimeError, match="No covariance"):
        fitted.materialize_covariance(8 * 1024 * 1024)
    collector.raise_if_any("Public operator prediction")


@pytest.mark.parametrize(
    ("formula", "sp"),
    [
        ("y~0+x+s(g,x,bs='re')", [0.8]),
        ("y~0+s(g,u,bs='re')+s(h,v,bs='re')", [0.8, 1.1]),
        ("y~0+x+s(g,bs='re')+s(u,bs='cr',k=6)", [0.8, 1.1]),
    ],
)
def test_no_intercept_operator_public_prediction_roundtrip(formula, sp):
    n = 41
    row = np.arange(n)
    frame = pd.DataFrame(
        {
            "y": np.linspace(1.0, 2.0, n),
            "x": np.linspace(0.2, 1.8, n),
            "u": np.sin(0.33 * row),
            "v": np.cos(0.71 * row),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "h": pd.Categorical(np.resize(["u", "v", "w", "t"], n)),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    result = GAM(
        formula,
        sp=sp,
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=7,
            memory_budget_bytes=8 * 1024 * 1024,
        ),
    ).fit(source, result="prediction")
    dense = ModelSetup.build(parse_formula(formula), frame)
    assert result.converged
    np.testing.assert_allclose(
        result.predict_matrix(frame), dense.X, rtol=STRICT.rtol, atol=STRICT.atol
    )
    expected = dense.X @ result.coefficients
    np.testing.assert_allclose(
        result.predict(frame), expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    streamed = list(result.predict_iter(source, batch_rows=11))
    np.testing.assert_allclose(
        np.concatenate([values for _, values in streamed]),
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_symbolic_large_prediction_budget_avoids_expanded_rows_and_zeros_unknown_levels(
    monkeypatch,
):
    levels = [f"L{i}" for i in range(256)]
    n = 31
    frame = pd.DataFrame(
        {
            "y": np.linspace(1, 2, n),
            "g": pd.Categorical(np.resize(levels[:4], n), categories=levels),
            "h": pd.Categorical(np.resize(levels[:3], n), categories=levels),
        }
    )
    model = prepare_operator_model(
        parse_formula("y~s(g,h,bs='re')"),
        DataFrameRowSource(frame, response="y"),
        control=OperatorPreparationControl(batch_rows=7),
    )
    geometry = OperatorPredictionGeometry(
        model.modest.predict_spec,
        model.modest_columns,
        model.random_effects,
        model.n_coef,
        7,
        model.modest_raw_width,
        model.metadata_bound_bytes,
    )
    beta = np.arange(model.n_coef, dtype=float) * 0.001
    predictor = GAMPredictor(
        beta,
        None,
        Gaussian(),
        "y~s(g,h,bs='re')",
        False,
        geometry.modest,
        _operator_geometry=geometry,
        _memory_budget_bytes=1024 * 1024,
        _output_budget_bytes=32 * 1024 * 1024,
    )
    result = predictor.predict(frame)
    ids, amplitude = model.random_effects[0].evaluate(dict(frame.items()))
    np.testing.assert_allclose(
        result, beta[0] + amplitude * beta[1 + ids], rtol=STRICT.rtol, atol=STRICT.atol
    )
    beta[:] = -1
    assert predictor.coefficients[0] == 0
    novel = pd.DataFrame({"g": ["unknown", "L1"], "h": ["L1", "unknown"]})
    assert np.array_equal(predictor.predict(novel), [0.0, 0.0])

    def forbidden(*_args):
        pytest.fail("Dense matrix evaluator ran before prospective rejection.")

    monkeypatch.setattr(OperatorPredictionGeometry, "build_predict_matrix", forbidden)
    with pytest.raises(MemoryError, match="workspace"):
        predictor.predict_matrix(frame)


@pytest.mark.parametrize(
    "case",
    [
        "float_positions",
        "negative_positions",
        "too_many",
        "too_few",
        "changed_fingerprint",
        "float_metadata",
    ],
)
def test_shared_prediction_rejects_invalid_source_lineage(operator_prediction, case):
    from dataclasses import replace

    fitted, frame, offset, _ = operator_prediction
    base = DataFrameRowSource(frame, offset=offset)

    class InvalidSource:
        finished = False

        @property
        def n_rows(self):
            return (
                float(base.n_rows)
                if case == "float_metadata" and self.finished
                else base.n_rows
            )

        def fingerprint(self):
            return (
                "changed"
                if case == "changed_fingerprint" and self.finished
                else base.fingerprint()
            )

        def scan(self, batch_rows):
            batches = base.scan(batch_rows)
            for index, batch in enumerate(batches):
                if case == "too_few" and index == 1:
                    break
                positions = batch.row_positions
                if case == "float_positions":
                    positions = positions.astype(float)
                elif case == "negative_positions":
                    positions = -positions - 1
                yield replace(batch, row_positions=positions)
                if case == "too_many":
                    yield batch
            self.finished = True

    with pytest.raises((ValueError, RuntimeError), match=r"positions|row|changed"):
        list(fitted.predict_iter(InvalidSource(), 7))


@pytest.mark.parametrize(("formula", "sp"), [("y~1", []), ("y~0+s(g,bs='re')", [1.0])])
def test_constant_and_disjoint_public_geometry_preserves_predictions(formula, sp):
    frame = pd.DataFrame(
        {
            "y": np.resize([1.0, 2.0, 3.0], 31),
            "g": pd.Categorical(
                np.resize(["a", "b"], 31), categories=["a", "b", "unused"]
            ),
        }
    )
    fitted = GAM(
        formula,
        sp=sp,
        control=FitControl(execution="operator", linear_solver="pcg", batch_rows=7),
    ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
    matrix = fitted.predict_matrix(frame)
    assert fitted.converged
    np.testing.assert_allclose(
        fitted.predict(frame),
        matrix @ fitted.coefficients,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize(
    "case",
    [
        "full",
        "inference",
        "fisher",
        "covariance",
        "automatic",
        "solver",
        "gpu",
        "estimated_theta",
    ],
)
def test_unavailable_operator_requests_reject_before_source_scan(case):
    from jaxgam.families.negative_binomial import NegativeBinomial

    class UntouchedSource:
        n_rows = 3

        def scan(self, _batch_rows):
            pytest.fail("Preparation ran for an unsupported request.")

        def fingerprint(self):
            pytest.fail("Source fingerprint ran for an unsupported request.")

    control = FitControl(
        execution="operator",
        linear_solver="pcg",
        uncertainty=case if case in {"fisher", "covariance"} else "none",
        operator=OperatorControl(),
    )
    if case == "solver":
        control = FitControl(execution="operator")
    model = GAM(
        "y~s(g,bs='re')",
        sp=None if case == "automatic" else [1.0],
        family=NegativeBinomial() if case == "estimated_theta" else "gaussian",
        control=control,
        device="gpu" if case == "gpu" else None,
    )
    with pytest.raises((NotImplementedError, ValueError)):
        model.fit(
            UntouchedSource(),
            result=case if case in {"full", "inference"} else "prediction",
        )
