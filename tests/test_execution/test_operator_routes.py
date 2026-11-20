"""Operator execution is opt-in and leaves every existing route unchanged.

``FitControl.operator`` only configures ``execution='operator'``. A dense or
streamed fit must take its existing route with bitwise-identical numbers
whatever ``OperatorControl`` it carries, and the operator-only
``linear_solver='pcg'`` must never be accepted by another route.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, FitControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.families.standard import Gaussian, Poisson
from tests.helpers import _AssertCollector

_FORMULA = 'y ~ s(x, bs="cr", k=5)'

_COMPARED_FIELDS = (
    "coefficients",
    "deviance",
    "score",
    "scale",
    "score_scale",
    "theta",
    "smoothing_params",
    "edf_total",
    "converged",
    "n_iter",
    "convergence_info",
    "lambda_strategy",
    "execution_route",
    "execution_fallback_reason",
    "source_scans",
    "batches_scanned",
    "known_workspace_bytes",
)

# A valid OperatorControl differing from the default in every field.
_NON_DEFAULT_OPERATOR = OperatorControl(
    max_modest_coefficients=7,
    max_iterations=3,
    tolerance=1e-3,
    cg_max_iterations=5,
    cg_rtol=1e-4,
    cg_atol=1e-5,
    use_coarse=True,
)


def _frame(n: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(73519)
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    signal = 0.3 + 0.5 * np.sin(3.0 * x)
    return pd.DataFrame(
        {
            "x": x,
            "y": signal + rng.normal(scale=0.2, size=n),
            "count": rng.poisson(np.exp(signal)).astype(float),
        }
    )


# route label -> (family factory, response, sp, execution, solver, compression)
_ROUTES = {
    "dense-estimated": (Gaussian, "y", None, "dense", "cholesky", False),
    "dense-fixed-poisson": (Poisson, "count", [0.8], "dense", "cholesky", False),
    "dense-gaussian-qr": (Gaussian, "y", None, "dense", "cholesky", True),
    "stream-fixed": (Gaussian, "y", [0.8], "stream", "cholesky", False),
    "stream-fixed-qr-poisson": (Poisson, "count", [0.8], "stream", "qr", False),
    "stream-estimated-qr": (Gaussian, "y", None, "stream", "qr", False),
    "stream-gaussian-qr": (Gaussian, "y", None, "stream", "cholesky", True),
}

_EXPECTED_ROUTE = {
    "dense-estimated": "dense",
    "dense-fixed-poisson": "dense",
    "dense-gaussian-qr": "gaussian_qr",
    "stream-fixed": "stream",
    "stream-fixed-qr-poisson": "stream_qr",
    "stream-estimated-qr": "stream_exact_reml_qr",
    "stream-gaussian-qr": "stream_gaussian_qr",
}


def _fit(label: str, operator: OperatorControl):
    family, response, sp, execution, solver, compression = _ROUTES[label]
    data = _frame()
    frame = data[["x", response]].rename(columns={response: "y"})
    model = GAM(
        _FORMULA,
        family=family(),
        sp=sp,
        control=FitControl(
            execution=execution,
            linear_solver=solver,
            batch_rows=23,
            gaussian_compression=compression,
            operator=operator,
        ),
    )
    source = DataFrameRowSource(frame, response="y") if execution == "stream" else frame
    return model.fit(source, result="prediction"), frame


@pytest.mark.parametrize("label", list(_ROUTES))
def test_operator_control_leaves_existing_routes_bitwise_unchanged(label) -> None:
    ordinary, frame = _fit(label, OperatorControl())
    configured, _ = _fit(label, _NON_DEFAULT_OPERATOR)
    assert ordinary.execution_route == _EXPECTED_ROUTE[label]
    assert ordinary.operator_diagnostics is None
    collector = _AssertCollector()
    for name in (*_COMPARED_FIELDS, "operator_diagnostics"):
        collector.check(
            name,
            lambda name=name: np.testing.assert_array_equal(
                getattr(configured, name), getattr(ordinary, name)
            ),
        )
    collector.check(
        "prediction",
        lambda: np.testing.assert_array_equal(
            configured.predict(frame[["x"]]), ordinary.predict(frame[["x"]])
        ),
    )
    collector.raise_if_any(label)


def test_operator_solver_is_rejected_by_dense_and_stream_routes() -> None:
    frame = _frame()[["x", "y"]]
    control = FitControl(linear_solver="pcg")
    with pytest.raises(NotImplementedError, match="linear_solver='pcg'"):
        GAM(_FORMULA, sp=[0.8], control=control).fit(frame)
    for gaussian_compression in (False, True):
        stream = FitControl(
            execution="stream",
            linear_solver="pcg",
            gaussian_compression=gaussian_compression,
        )
        for sp in ([0.8], None):
            with pytest.raises(ValueError, match="requires execution='operator'"):
                GAM(_FORMULA, sp=sp, control=stream).fit(
                    DataFrameRowSource(frame, response="y"), result="prediction"
                )


@pytest.mark.parametrize("linear_solver", ["cholesky", "qr"])
def test_operator_execution_requires_its_own_solver(linear_solver) -> None:
    frame = _frame()[["x", "y"]]
    control = FitControl(execution="operator", linear_solver=linear_solver)
    with pytest.raises(ValueError, match="requires linear_solver='pcg'"):
        GAM(_FORMULA, sp=[0.8], control=control).fit(
            DataFrameRowSource(frame, response="y"), result="prediction"
        )
