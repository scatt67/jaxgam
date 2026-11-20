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


def _small_operator_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(20261120)
    x = rng.uniform(0.0, 1.0, n)
    group = rng.integers(0, 20, n)
    return pd.DataFrame(
        {
            "x": x,
            "g": pd.Categorical(group, categories=range(20)),
            "y": rng.poisson(np.exp(0.2 + np.sin(6.0 * x))).astype(float),
        }
    )


@pytest.mark.parametrize(
    "formula",
    ["y ~ s(x) + s(g, bs='re')", "y ~ s(x, bs='gp') + s(g, bs='re')"],
)
def test_operator_fit_plans_for_the_rows_the_source_has(formula) -> None:
    """A small source fits at the default ``batch_rows`` and default budget.

    The operator kernels work on batches padded to the scan size, so a fit
    that planned for the default 65,536 rows exceeded the default budget on a
    300-row source with a thin plate or Gaussian process smooth. The fit now
    scans, pads and budgets for at most the rows the source holds, which is
    the same single batch, so asking for more rows changes nothing.
    """
    frame = _small_operator_frame()
    source = DataFrameRowSource(frame, response="y")

    def fit(batch_rows: int):
        control = FitControl(
            execution="operator", linear_solver="pcg", batch_rows=batch_rows
        )
        return GAM(formula, family="poisson", sp=[1.0, 1.0], control=control).fit(
            source, result="prediction"
        )

    default = fit(FitControl().batch_rows)
    exact = fit(len(frame))
    collector = _AssertCollector()
    collector.check("converged", lambda: check_converged(default, exact))
    collector.check(
        "same coefficients as a scan size equal to the row count",
        lambda: np.testing.assert_array_equal(default.coefficients, exact.coefficients),
    )
    collector.check(
        "same preparation bound as a scan size equal to the row count",
        lambda: check_equal(
            default.operator_diagnostics.preparation_peak_bound_bytes,
            exact.operator_diagnostics.preparation_peak_bound_bytes,
        ),
    )
    # Prediction keeps the batch size the caller asked for: a request larger
    # than the training data is served in one call.
    larger = pd.concat([frame] * 3, ignore_index=True)
    collector.check(
        "prediction is not limited to the training row count",
        lambda: np.testing.assert_array_equal(
            default.predict(larger)[: len(frame)], default.predict(frame)
        ),
    )
    collector.raise_if_any(formula)


def check_converged(*fits) -> None:
    assert all(fit.converged for fit in fits)


def check_equal(left, right) -> None:
    assert left == right


class _CountSource:
    def __init__(self, n_rows) -> None:
        self.n_rows = n_rows


@pytest.mark.parametrize(
    ("n_rows", "expected"),
    [
        (300, 300),
        (np.int64(300), 300),
        (4096, 4096),
        (10**9, 4096),
        # Not a row count: the preparation reports it, so nothing is clamped.
        (0, 4096),
        (True, 4096),
        (300.0, 4096),
    ],
)
def test_operator_fit_batch_rows_is_the_smaller_of_request_and_source(
    n_rows, expected
) -> None:
    from jaxgam.api import _operator_fit_batch_rows

    rows = _operator_fit_batch_rows(FitControl(batch_rows=4096), _CountSource(n_rows))
    assert rows == expected
    assert type(rows) is int
