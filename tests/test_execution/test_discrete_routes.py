"""Route selection between the published routes and explicit discrete fitting.

``execution='discrete'`` is reached only by an explicit request carrying a
discretization policy. Every other fit dispatches exactly as before and never
enters the discrete route. A discrete fit owns its solves: it rejects
``linear_solver='qr'`` and reports an unmet ``gaussian_compression`` request
in ``execution_fallback_reason`` while its numbers stay bitwise identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import jaxgam.execution.discrete as discrete_execution
from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data.source import DataFrameRowSource
from tests.helpers import _AssertCollector

_FORMULA = "y ~ z + s(x, bs='cr', k=6)"
_DISCRETE_ENTRY_POINTS = ("fit_fixed_sp_discrete_gaussian",)
_COMPARED_FIELDS = (
    "coefficients",
    "deviance",
    "score",
    "scale",
    "theta",
    "smoothing_params",
    "edf_total",
    "converged",
    "n_iter",
    "convergence_info",
    "lambda_strategy",
    "execution_path",
    "execution_route",
    "smoothing_strategy",
    "discretization_strategy",
    "prediction_policy",
    "fitting_source_scans",
    "fitting_batches_scanned",
    "fit_peak_bytes",
    "fit_memory_budget_bytes",
    "n",
)


def _frame(n_rows: int = 75) -> pd.DataFrame:
    position = np.arange(n_rows, dtype=float)
    x = (position + 0.3 * np.sin(position)) / n_rows
    z = np.cos(position * 0.31)
    y = 0.7 - 0.4 * z + np.sin(5.0 * x) + 0.05 * np.cos(3.1 * position)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _discrete_control(**overrides) -> FitControl:
    return FitControl(
        execution="discrete",
        batch_rows=11,
        discretization=ContinuousDiscretizationControl(m=12),
        **overrides,
    )


def test_fit_control_pairs_discrete_execution_with_its_policy() -> None:
    policy = ContinuousDiscretizationControl(m=12)
    assert FitControl().discretization is None
    assert FitControl(execution="discrete", discretization=policy).execution == (
        "discrete"
    )
    with pytest.raises(ValueError, match="explicit discretization policy"):
        FitControl(execution="discrete")
    for execution in ("dense", "stream"):
        with pytest.raises(ValueError, match="requires execution='discrete'"):
            FitControl(execution=execution, discretization=policy)
    with pytest.raises(TypeError, match="ContinuousDiscretizationControl"):
        FitControl(execution="discrete", discretization=object())
    with pytest.raises(ValueError, match="'dense', 'stream', or 'discrete'"):
        FitControl(execution="operator")


@pytest.fixture
def forbid_discrete_route(monkeypatch):
    """Fail any fit that reaches the discrete route or its entry points."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a non-discrete fit entered the discrete route")

    monkeypatch.setattr(GAM, "_fit_discrete", forbidden)
    for name in _DISCRETE_ENTRY_POINTS:
        monkeypatch.setattr(discrete_execution, name, forbidden)


@pytest.mark.parametrize(
    ("data_kind", "family", "sp", "control", "route"),
    [
        ("frame", "gaussian", [0.65], FitControl(), "dense"),
        ("frame", "gaussian", None, FitControl(), "dense"),
        (
            "frame",
            "gaussian",
            [0.65],
            FitControl(gaussian_compression=True),
            "gaussian_qr",
        ),
        ("source", "gaussian", [0.65], FitControl(execution="stream"), "stream"),
        (
            "source",
            "poisson",
            [0.65],
            FitControl(execution="stream", linear_solver="qr"),
            "stream_qr",
        ),
        (
            "source",
            "gaussian",
            None,
            FitControl(execution="stream", linear_solver="qr"),
            "stream_exact_reml_qr",
        ),
        (
            "source",
            "gaussian",
            [0.65],
            FitControl(execution="stream", gaussian_compression=True),
            "stream_gaussian_qr",
        ),
    ],
    ids=[
        "dense-fixed",
        "dense-estimated",
        "dense-gaussian-qr",
        "stream-fixed",
        "stream-qr-poisson",
        "stream-exact-reml",
        "stream-gaussian-qr",
    ],
)
@pytest.mark.usefixtures("forbid_discrete_route")
def test_published_routes_never_enter_the_discrete_route(
    data_kind, family, sp, control, route
) -> None:
    frame = _frame()
    if family == "poisson":
        frame = frame.assign(y=np.round(np.exp(frame["y"])))
    data = frame if data_kind == "frame" else DataFrameRowSource(frame, response="y")
    result = GAM(_FORMULA, family=family, sp=sp, control=control).fit(
        data, result="prediction"
    )
    assert control.discretization is None
    assert result.smoothing_strategy is None
    assert result.discretization_strategy is None
    assert result.prediction_policy is None
    assert result.fitting_source_scans is None
    assert result.execution_route == route


@pytest.mark.usefixtures("forbid_discrete_route")
def test_dense_route_still_rejects_a_row_source_naming_both_routes() -> None:
    source = DataFrameRowSource(_frame(), response="y")
    with pytest.raises(TypeError, match="'stream' or 'discrete'"):
        GAM(_FORMULA, sp=[0.65]).fit(source, result="prediction")


def test_discrete_route_rejects_the_streamed_qr_solver() -> None:
    source = DataFrameRowSource(_frame(), response="y")
    with pytest.raises(NotImplementedError, match="linear_solver='qr'"):
        GAM(_FORMULA, sp=[0.65], control=_discrete_control(linear_solver="qr")).fit(
            source, result="prediction"
        )


def test_discrete_gaussian_qr_request_keeps_the_discrete_fit_bitwise() -> None:
    frame = _frame()
    results = [
        GAM(
            _FORMULA,
            sp=[0.65],
            control=_discrete_control(gaussian_compression=request),
        ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
        for request in (False, True)
    ]
    expected, actual = results
    assert expected.execution_fallback_reason is None
    assert actual.execution_fallback_reason == (
        "Gaussian QR compression requires execution 'dense' or 'stream'."
    )
    collector = _AssertCollector()
    for name in _COMPARED_FIELDS:
        collector.check(
            name,
            lambda name=name: np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name)
            ),
        )
    collector.check(
        "prediction",
        lambda: np.testing.assert_array_equal(
            actual.predict(frame[["x", "z"]]), expected.predict(frame[["x", "z"]])
        ),
    )
    collector.raise_if_any("discrete fit with an unmet Gaussian QR request")
