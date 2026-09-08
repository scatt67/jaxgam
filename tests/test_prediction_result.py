"""Gates for the compact streamed-prediction result mode."""

import pickle

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, FitControl, GAMPredictionResult, GAMPredictor
from jaxgam.data.source import ArrayRowSource
from tests.helpers import _AssertCollector, _inference_r_cases, r_available
from tests.tolerances import MODERATE


def _data() -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, 41)
    return pd.DataFrame(
        {"x": x, "g": pd.Categorical(np.where(x > 0.5, "b", "a")), "y": np.sin(x)}
    )


def test_prediction_mode_is_compact_and_matches_dense_point_predictions(
    monkeypatch,
) -> None:
    data = _data()
    full = GAM("y ~ x + s(x, bs='cr', k=6)", sp=[0.2]).fit(data)
    import jaxgam.results as results

    monkeypatch.setattr(
        results,
        "_compute_per_smooth_edf",
        lambda *_: (_ for _ in ()).throw(AssertionError("EDF allocated")),
    )
    compact = GAM("y ~ x + s(x, bs='cr', k=6)", sp=[0.2]).fit(data, result="prediction")
    assert isinstance(compact, GAMPredictionResult)
    assert not hasattr(compact, "edf")
    assert not hasattr(compact, "Vp")
    np.testing.assert_allclose(compact.predict(data[["x"]]), full.predict(data[["x"]]))
    with pytest.raises(RuntimeError, match="uncertainty is unavailable"):
        compact.predict(data[["x"]], se_fit=True)


def test_fisher_standard_errors_and_batched_positions_match_full() -> None:
    data = _data()
    formula = "y ~ x + s(x, bs='cr', k=6)"
    full = GAM(formula, sp=[0.2]).fit(data)
    compact = GAM(
        formula, sp=[0.2], control=FitControl(uncertainty="fisher", batch_rows=7)
    ).fit(data, result="prediction")
    expected = full.predict(data[["x"]], pred_type="response", se_fit=True)
    actual = compact.predict(data[["x"]], pred_type="response", se_fit=True)
    np.testing.assert_allclose(actual, expected, rtol=MODERATE.rtol, atol=MODERATE.atol)
    source = ArrayRowSource({"x": data.x.to_numpy()})
    batches = list(compact.predict_iter(source))
    np.testing.assert_array_equal(
        np.concatenate([p for p, _ in batches]), np.arange(len(data))
    )
    np.testing.assert_allclose(np.concatenate([v for _, v in batches]), expected[0])


def test_control_rejects_unimplemented_routes_and_budgets_covariance() -> None:
    with pytest.raises(NotImplementedError, match="gaussian_compression"):
        FitControl(gaussian_compression=True)
    with pytest.raises(NotImplementedError, match="execution"):
        FitControl(execution="stream")  # type: ignore[arg-type]
    data = _data()
    result = GAM(
        "y ~ s(x, bs='cr', k=6)", sp=[0.2], control=FitControl(uncertainty="fisher")
    ).fit(data, result="prediction")
    with pytest.raises(MemoryError, match="Vp materialization"):
        result.materialize_covariance(1)
    assert result.materialize_covariance(1_000_000).shape == (6, 6)


def test_empty_iterator_validates_arguments_and_predict_budget() -> None:
    data = _data()
    result = GAM(
        "y ~ s(x, bs='cr', k=6)",
        sp=[0.2],
        control=FitControl(output_budget_bytes=8),
    ).fit(data, result="prediction")
    empty = ArrayRowSource({"x": np.empty(0)})
    with pytest.raises(ValueError, match="batch_rows"):
        result.predict_iter(empty, 0)
    with pytest.raises(ValueError, match="pred_type"):
        result.predict_iter(empty, 2, pred_type="terms")
    with pytest.raises(MemoryError, match="output budget"):
        result.predict(data[["x"]])


def test_factor_provider_owns_inputs_and_validates_offsets_and_pickle() -> None:
    data = _data()
    result = GAM(
        "y ~ s(x, bs='cr', k=6)", sp=[0.2], control=FitControl(uncertainty="fisher")
    ).fit(data, result="prediction")
    factor = result._predictor._fisher_factor
    assert factor is not None
    assert not factor.flags.writeable
    caller_factor = factor.copy()
    caller_transforms = tuple(
        (start, stop, kind, values.copy())
        for start, stop, kind, values in result._predictor._fisher_transforms
    )
    owned = GAMPredictor(
        coefficients=result.coefficients,
        Vp=None,
        family=result.family,
        formula=result.formula,
        offset_was_nonzero=False,
        _predict_spec=result._predictor._predict_spec,
        _fisher_factor=caller_factor,
        _fisher_transforms=caller_transforms,
    )
    assert caller_factor.flags.writeable
    assert all(values.flags.writeable for _, _, _, values in caller_transforms)
    assert not owned._fisher_factor.flags.writeable
    restored = pickle.loads(pickle.dumps(result))
    assert not restored.smoothing_params.flags.writeable
    assert not restored._predictor._fisher_factor.flags.writeable
    for offset in (1.0, np.ones((len(data), 1)), np.full(len(data), np.nan)):
        with pytest.raises(ValueError, match="offset"):
            result.predict(data[["x"]], offset=offset)
    with pytest.raises(ValueError, match="pred_type"):
        result.predict(data[["x"]], pred_type="terms")


def test_workspace_budget_offset_warning_and_reordered_source() -> None:
    data = _data()
    p = 6
    point_budget = len(data) * p * np.dtype(float).itemsize
    result = GAM(
        "y ~ s(x, bs='cr', k=6)",
        sp=[0.2],
        control=FitControl(
            uncertainty="fisher",
            memory_budget_bytes=point_budget,
            output_budget_bytes=100_000,
        ),
    ).fit(data, offset=np.linspace(0.1, 0.2, len(data)), result="prediction")
    result.predict(data[["x"]])
    with pytest.raises(MemoryError, match="workspace"):
        result.predict(data[["x"]], se_fit=True)
    source = ArrayRowSource(
        {"x": data.x.to_numpy()}, row_selection=np.array([7, 2, 40, 1])
    )
    with pytest.warns(UserWarning, match="external offset") as caught:
        batches = list(result.predict_iter(source, 2))
    assert len(caught) == 1
    np.testing.assert_array_equal(
        np.concatenate([positions for positions, _ in batches]), np.array([7, 2, 40, 1])
    )


def test_point_only_result_retains_no_rows_or_dense_covariance() -> None:
    data = _data()
    result = GAM("y ~ s(x, bs='cr', k=6)", sp=[0.2]).fit(data, result="prediction")
    assert not hasattr(result, "setup")
    assert result._predictor.Vp is None
    assert result._predictor._fisher_factor is None
    assert all(
        array.shape != (len(data),)
        for array in result.__dict__.values()
        if isinstance(array, np.ndarray)
    )


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
def test_prediction_result_fisher_predictions_and_se_match_mgcv(r_bridge) -> None:
    """Compact factor uncertainty retains direct pinned-mgcv parity."""
    if r_bridge.mode != "rpy2":
        pytest.skip("Direct R prediction parity requires rpy2")
    collector = _AssertCollector()
    for name, case in _inference_r_cases().items():
        py_formula, r_formula, data, newdata, py_family, r_family = case
        result = GAM(
            py_formula,
            family=py_family,
            control=FitControl(uncertainty="fisher", batch_rows=11),
        ).fit(data, result="prediction")
        actual, actual_se = result.predict(newdata, se_fit=True)
        expected = r_bridge.predict_gam(
            r_formula, data, newdata, family=r_family, pred_type="response", se_fit=True
        )
        collector.check(
            f"{name}: prediction",
            lambda a=actual, e=expected["predictions"]: np.testing.assert_allclose(
                a, e, rtol=MODERATE.rtol, atol=MODERATE.atol
            ),
        )
        collector.check(
            f"{name}: standard error",
            lambda a=actual_se, e=expected["se"]: np.testing.assert_allclose(
                a, e, rtol=MODERATE.rtol, atol=MODERATE.atol
            ),
        )
    collector.raise_if_any("prediction-only direct R parity")
