"""Structural guards for the scaling plan's forbidden allocation shapes.

Plan section 8: exact direct paths must not allocate an n-by-n Q, an n-by-p X
in streaming, m-by-p-by-p penalty arrays or n-by-r-by-d distance differences.
Each test forbids or records the call that would create one shape on the
public routes it names; peak-memory sampling alone cannot prove an absence.
The m-by-p-by-p guard is not here: at test sizes a host allocation trace
cannot separate a penalty stack from ordinary working memory (see the
acceptance-cases document).
"""

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from jaxgam import GAM, FitControl
from jaxgam.control import ContinuousDiscretizationControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.formula.design import ModelSetup
from jaxgam.smooths import gaussian_process as gp_module
from jaxgam.smooths import tprs as tprs_module

_PARAMETRIC_FORMULA = "y ~ x2 + s(x0, bs='cr', k=8) + s(x1, bs='cr', k=8)"
_SHARED_FORMULA = "y ~ x0 + s(x0, bs='cr', k=6) + te(x0, x1, bs='cr', k=4)"


def _frame(n: int, seed: int = 20261117) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0, x1, x2 = (rng.uniform(0.0, 1.0, n) for _ in range(3))
    groups = rng.integers(0, 40, n)
    y = (
        np.sin(2 * np.pi * x0)
        + 0.5 * np.cos(np.pi * x1)
        + 0.3 * x2
        + 0.2 * np.cos(groups)
        + rng.normal(0.0, 0.3, n)
    )
    return pd.DataFrame(
        {
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "g": pd.Categorical.from_codes(groups, [f"g{i}" for i in range(40)]),
            "y": y,
        }
    )


def _discrete(**kwargs: object) -> FitControl:
    return FitControl(
        execution="discrete",
        batch_rows=256,
        smoothing_strategy="working_model_reml",
        discretization=ContinuousDiscretizationControl(**kwargs),
    )


# route -> (formula, fixed sp or None, control or None for the dense route)
_SOURCE_ROUTES = {
    "stream-exact": (
        _PARAMETRIC_FORMULA,
        None,
        FitControl(execution="stream", linear_solver="qr", batch_rows=256),
    ),
    "stream-gaussian-qr": (
        _PARAMETRIC_FORMULA,
        None,
        FitControl(execution="stream", gaussian_compression=True, batch_rows=256),
    ),
    "discrete-wm": (_PARAMETRIC_FORMULA, None, _discrete()),
    "operator": (
        "y ~ x2 + s(x0, bs='cr', k=8) + s(g, bs='re')",
        [1.0, 1.0],
        FitControl(execution="operator", linear_solver="pcg", batch_rows=256),
    ),
}


def _fit(frame: pd.DataFrame, formula: str, sp, control: FitControl | None):
    if control is None:
        return GAM(formula, sp=sp).fit(frame, result="prediction")
    return GAM(formula, sp=sp, control=control).fit(
        DataFrameRowSource(frame, response="y"), result="prediction"
    )


def _rows(data) -> int:
    return len(next(iter(data.values()))) if isinstance(data, dict) else len(data)


@pytest.mark.parametrize("route", sorted(_SOURCE_ROUTES))
def test_source_routes_never_build_the_training_design(monkeypatch, route) -> None:
    """No n-by-p X: the dense design builder never sees the training rows.

    Covers exact streamed REML, frozen Gaussian QR, the discrete working
    model and the operator route; fixed-sp streaming is guarded in
    test_stream_fixed_public_family.py. The discrete route builds its design
    on the compact frame (bam(discrete=TRUE)), whose rows are bounded by the
    discretization grid: about 1000 here at n = 5000 and at n = 20000.
    """
    n = 5000
    original = ModelSetup.build
    built: list[int] = []

    def guarded(_cls, formula, data, *args, **kwargs):
        rows = _rows(data)
        built.append(rows)
        if rows >= n // 2:
            raise AssertionError(f"{route} built a {rows}-row training design")
        return original(formula, data, *args, **kwargs)

    monkeypatch.setattr(ModelSetup, "build", classmethod(guarded))
    formula, sp, control = _SOURCE_ROUTES[route]
    result = _fit(_frame(n), formula, sp, control)

    assert np.all(np.isfinite(result.coefficients))
    assert result.source_scans > 0
    if route == "discrete-wm":
        assert built
        assert max(built) <= 1000
    else:
        assert built == []


_FULL_Q_ROWS = 1000


@pytest.mark.parametrize(
    ("route", "formula", "sp", "control"),
    [
        ("dense", _SHARED_FORMULA, None, None),
        ("discrete-wm", _SHARED_FORMULA, None, _discrete()),
        *[
            (route, formula, sp, control)
            for route, (formula, sp, control) in sorted(_SOURCE_ROUTES.items())
            if route != "discrete-wm"
        ],
    ],
)
def test_routes_never_request_an_n_by_n_q(
    monkeypatch, route, formula, sp, control
) -> None:
    """No n-by-n Q: no QR call asks for the full Q of a tall matrix.

    ``scipy.linalg.qr`` defaults to ``mode='full'``; NumPy's ``complete`` mode
    is the same request. Raw, economic and R-only modes are allowed. The
    shared-covariate formula drives gam.side dependence detection on the
    dense and discrete routes.
    """
    calls: list[str] = []
    scipy_qr, numpy_qr = scipy.linalg.qr, np.linalg.qr

    def guarded_scipy(a, *args, mode: str = "full", **kwargs):
        rows = np.shape(a)[0]
        calls.append(mode)
        if mode == "full" and rows >= _FULL_Q_ROWS:
            raise AssertionError(f"{route} requested a {rows} x {rows} Q")
        return scipy_qr(a, *args, mode=mode, **kwargs)

    def guarded_numpy(a, mode: str = "reduced"):
        rows = np.shape(a)[-2]
        calls.append(mode)
        if mode == "complete" and rows >= _FULL_Q_ROWS:
            raise AssertionError(f"{route} requested a {rows} x {rows} Q")
        return numpy_qr(a, mode=mode)

    monkeypatch.setattr(scipy.linalg, "qr", guarded_scipy)
    monkeypatch.setattr(np.linalg, "qr", guarded_numpy)
    result = _fit(_frame(3 * _FULL_Q_ROWS), formula, sp, control)

    assert np.all(np.isfinite(result.coefficients))
    if route == "dense":
        # The guard is live: dense dependence detection runs packed raw QR.
        assert "raw" in calls


@pytest.mark.parametrize(
    ("route", "formula", "control"),
    [
        ("dense-tp", "y ~ s(x0, x1, k=12, bs='tp')", None),
        ("dense-gp", "y ~ s(x0, k=12, bs='gp')", None),
        ("discrete-tp", "y ~ s(x0, x1, k=12, bs='tp')", "discrete"),
        ("operator-tp", "y ~ s(x0, x1, k=12, bs='tp')", "operator"),
        ("operator-gp", "y ~ s(x0, k=12, bs='gp')", "operator"),
    ],
)
def test_training_kernel_distances_are_row_batched(
    monkeypatch, route, formula, control
) -> None:
    """No n-by-r-by-d differences: kernel rows are evaluated in bounded batches.

    With 2500 rows the thin-plate route evaluates design rows against its
    2000 subsampled knots (mgcv max.knots), and the Gaussian process its
    kernel chunks; every data-row distance batch stays at the patched row
    limit. The one knot-by-knot kernel (gpE on the knot table, as in mgcv) is
    r x r x d with r capped at max.knots, not n x r x d.
    """
    del route  # labels the parametrized case
    limit = 64
    monkeypatch.setattr(tprs_module, "DISTANCE_BATCH_ROWS", limit)
    monkeypatch.setattr(gp_module, "DISTANCE_BATCH_ROWS", limit)
    batches: list[int] = []
    tps_rows, gp_distances = (
        tprs_module._source_tps_rows,
        gp_module._compute_distance_matrix,
    )

    def recorded_tps(rows, knots, m):
        batches.append(len(rows))
        return tps_rows(rows, knots, m)

    knot_tables: list[int] = []

    def recorded_gp(rows, knots):
        if rows.shape == knots.shape and np.array_equal(rows, knots):
            knot_tables.append(len(rows))
        else:
            batches.append(len(rows))
        return gp_distances(rows, knots)

    monkeypatch.setattr(tprs_module, "_source_tps_rows", recorded_tps)
    monkeypatch.setattr(gp_module, "_compute_distance_matrix", recorded_gp)
    controls = {
        None: None,
        "discrete": _discrete(budget=_large_discrete_budget()),
        "operator": FitControl(
            execution="operator", linear_solver="pcg", batch_rows=256
        ),
    }
    result = _fit(
        _frame(2500),
        formula,
        [1.0] if control == "operator" else None,
        controls[control],
    )

    assert np.all(np.isfinite(result.coefficients))
    assert batches
    assert max(batches) <= limit
    assert all(rows <= 2000 for rows in knot_tables)


def _large_discrete_budget():
    from jaxgam.data.discretize import DiscretizationBudget

    return DiscretizationBudget(max_table_bytes=1 << 30, max_batch_bytes=1 << 30)
