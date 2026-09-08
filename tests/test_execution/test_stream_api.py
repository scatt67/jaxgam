"""Public fixed-sp RowSource routing for streamed PIRLS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, FitControl, GAMPredictionResult
from jaxgam.data.source import DataFrameRowSource
from tests.helpers import _AssertCollector, r_available
from tests.tolerances import MODERATE, STRICT


def _data(family: str, *, zero_weights: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(918)
    x = np.linspace(-0.8, 1.1, 67)
    offset = 0.12 * np.sin(2.0 * x)
    weight = 0.2 + rng.random(len(x))
    if zero_weights:
        weight[::11] = 0.0
    eta = 0.2 + 0.7 * x + offset
    if family == "gaussian":
        y = eta + rng.normal(scale=0.18, size=len(x))
    elif family == "poisson":
        y = rng.poisson(np.exp(eta))
    else:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    return pd.DataFrame({"x": x, "y": y, "weight": weight, "offset": offset})


@pytest.mark.parametrize("family", ["gaussian", "poisson", "binomial"])
@pytest.mark.parametrize("batch_rows", [7, 23])
def test_public_stream_fixed_sp_matches_dense_with_source_weights_and_offsets(
    family: str, batch_rows: int
) -> None:
    data = _data(family)
    source = DataFrameRowSource(
        data,
        response="y",
        weights=data.weight.to_numpy(),
        offset=data.offset.to_numpy(),
    )
    formula = 'y ~ s(x, bs="cr", k=6)'
    streamed = GAM(
        formula,
        family=family,
        sp=[0.3],
        control=FitControl(execution="stream", batch_rows=batch_rows),
    ).fit(source, result="prediction")
    dense = GAM(formula, family=family, sp=[0.3]).fit(
        data,
        weights=data.weight.to_numpy(),
        offset=data.offset.to_numpy(),
        result="prediction",
    )

    collector = _AssertCollector()
    collector.check("prediction result", lambda: assert_is_prediction(streamed))
    collector.check(
        "coefficients",
        lambda: np.testing.assert_allclose(
            streamed.coefficients,
            dense.coefficients,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "score",
        lambda: np.testing.assert_allclose(
            streamed.score, dense.score, rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        "offset prediction",
        lambda: np.testing.assert_allclose(
            streamed.predict(data[["x"]], offset=data.offset.to_numpy()),
            dense.predict(data[["x"]], offset=data.offset.to_numpy()),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any("public streamed fixed-sp dense parity")


def assert_is_prediction(result: object) -> None:
    """Keep the collector's public-route checks compact and named."""
    assert isinstance(result, GAMPredictionResult)
    assert result.execution_path == "jax"
    assert result.execution_route == "stream"
    assert result.execution_fallback_reason is None
    assert result.lambda_strategy == "fixed"


def test_public_stream_fisher_and_point_retention() -> None:
    data = _data("gaussian")
    source = DataFrameRowSource(data, response="y", offset=data.offset.to_numpy())
    formula = 'y ~ s(x, bs="cr", k=6)'
    fisher = GAM(
        formula,
        sp=[0.2],
        control=FitControl(execution="stream", uncertainty="fisher", batch_rows=9),
    ).fit(source, result="prediction")
    dense = GAM(formula, sp=[0.2], control=FitControl(uncertainty="fisher")).fit(
        data, offset=data.offset.to_numpy(), result="prediction"
    )
    actual = fisher.predict(data[["x"]], offset=data.offset.to_numpy(), se_fit=True)
    expected = dense.predict(data[["x"]], offset=data.offset.to_numpy(), se_fit=True)
    np.testing.assert_allclose(
        actual[0], expected[0], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        actual[1], expected[1], rtol=MODERATE.rtol, atol=MODERATE.atol
    )

    point = GAM(formula, sp=[0.2], control=FitControl(execution="stream")).fit(
        source, result="prediction"
    )
    assert not hasattr(point, "setup")
    assert point._predictor.Vp is None
    assert point._predictor._fisher_factor is None
    assert all(
        value.shape != (len(data),)
        for value in point.__dict__.values()
        if isinstance(value, np.ndarray)
    )


def test_public_stream_accepts_real_zero_and_extreme_prior_weights() -> None:
    data = _data("poisson", zero_weights=True)
    data.loc[data.index[1], "weight"] = 1e-14
    data.loc[data.index[2], "weight"] = 1e8
    result = GAM(
        'y ~ s(x, bs="cr", k=6)',
        family="poisson",
        sp=[0.2],
        control=FitControl(execution="stream", batch_rows=8),
    ).fit(
        DataFrameRowSource(data, response="y", weights=data.weight.to_numpy()),
        result="prediction",
    )
    assert result.converged
    assert np.isfinite(result.deviance)


def test_public_stream_covariance_budget_and_route_guards() -> None:
    data = _data("gaussian")
    source = DataFrameRowSource(data, response="y")
    formula = 'y ~ s(x, bs="cr", k=6)'
    with pytest.raises(MemoryError, match="Known streamed PIRLS workspace"):
        GAM(
            formula,
            sp=[0.2],
            control=FitControl(
                execution="stream", uncertainty="covariance", memory_budget_bytes=1
            ),
        ).fit(source, result="prediction")
    covariance = GAM(
        formula,
        sp=[0.2],
        control=FitControl(
            execution="stream", uncertainty="covariance", memory_budget_bytes=20_000_000
        ),
    ).fit(source, result="prediction")
    assert covariance._predictor.Vp is not None

    with pytest.raises(TypeError, match="RowSource"):
        GAM(formula, sp=[0.2]).fit(source)
    with pytest.raises(TypeError, match="requires a replayable RowSource"):
        GAM(formula, sp=[0.2], control=FitControl(execution="stream")).fit(data)
    with pytest.raises(NotImplementedError, match="result='prediction'"):
        GAM(formula, sp=[0.2], control=FitControl(execution="stream")).fit(source)
    with pytest.raises(NotImplementedError, match="requires explicit fixed sp"):
        GAM(formula, control=FitControl(execution="stream")).fit(
            source, result="prediction"
        )
    with pytest.raises(ValueError, match="owned by the RowSource"):
        GAM(formula, sp=[0.2], control=FitControl(execution="stream")).fit(
            source, weights=np.ones(len(data)), result="prediction"
        )
    with pytest.raises(NotImplementedError, match=r"Gaussian.*Poisson.*Binomial"):
        GAM(formula, family="nb", sp=[0.2], control=FitControl(execution="stream")).fit(
            source, result="prediction"
        )


def test_public_stream_accepts_empty_fixed_sp_for_unpenalized_model() -> None:
    data = _data("gaussian")
    source = DataFrameRowSource(data, response="y")
    result = GAM(
        "y ~ x",
        sp=[],
        control=FitControl(execution="stream", batch_rows=8),
        device="cpu",
    ).fit(source, result="prediction")
    dense = GAM("y ~ x", sp=[]).fit(data, result="prediction")
    assert result.smoothing_params.shape == (0,)
    assert result.converged
    np.testing.assert_allclose(
        result.score, dense.score, rtol=STRICT.rtol, atol=STRICT.atol
    )


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
def test_public_stream_fixed_sp_matches_pinned_r_predictions_and_scale(
    r_bridge,
) -> None:
    """The public RowSource route keeps the fixed-sp R contract end-to-end."""
    from tests.r_bridge import RBridge

    data = _data("gaussian")[["x", "y"]]
    formula = 'y ~ s(x, bs="cr", k=6)'
    reference = RBridge().fit_gam(formula, data, family="gaussian")
    result = GAM(
        formula,
        sp=reference["smoothing_params"],
        control=FitControl(execution="stream", uncertainty="fisher", batch_rows=7),
    ).fit(DataFrameRowSource(data, response="y"), result="prediction")
    prediction, se = result.predict(data[["x"]], se_fit=True)
    expected = r_bridge.predict_gam(
        formula, data, data[["x"]], family="gaussian", pred_type="response", se_fit=True
    )
    collector = _AssertCollector()
    collector.check(
        "coefficients",
        lambda: np.testing.assert_allclose(
            result.coefficients,
            reference["coefficients"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "prediction",
        lambda: np.testing.assert_allclose(
            prediction,
            expected["predictions"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "standard error",
        lambda: np.testing.assert_allclose(
            se, expected["se"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        "reported scale",
        lambda: np.testing.assert_allclose(
            result.scale, reference["scale"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.raise_if_any("public streamed fixed-sp R parity")
