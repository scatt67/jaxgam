"""Production fixed-sp Gaussian fitting on continuous-discrete designs."""

from __future__ import annotations

import gc
import math
import weakref

import numpy as np
import pandas as pd
import pytest

import jaxgam.execution.discrete as discrete_execution
import jaxgam.execution.discrete_regular as discrete_regular_execution
from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.families.standard import Gaussian
from jaxgam.formula.continuous_design import build_continuous_discrete_design
from jaxgam.formula.continuous_setup import (
    ContinuousDiscreteSetup,
    build_continuous_discrete_setup,
)
from jaxgam.formula.discrete_design import (
    DiscreteDesignControl,
    DiscreteDesignUnavailable,
)
from jaxgam.formula.parser import parse_formula
from tests.tolerances import STRICT


class _CountingSource(DataFrameRowSource):
    def __init__(self, frame: pd.DataFrame):
        super().__init__(frame, response="y")
        self.scans = 0

    def scan(self, batch_rows):
        self.scans += 1
        yield from super().scan(batch_rows)


def _observed_two_scan_fit(monkeypatch, *, retain: bool):
    """Fit while checking the first fitting scan's batch arrays are released.

    At the start of the second fitting scan every weak reference to the first
    scan's batch arrays must be dead.  The cyclic garbage collector is run
    explicitly first, so the check does not depend on when it last ran: it
    fails only if a live reference still holds an array.  ``retain`` keeps
    one array alive on purpose, to show the check fails in that case.
    """

    class ObservedSource(DataFrameRowSource):
        fitting_started = False
        fitting_scans = 0

        def __init__(self, frame):
            super().__init__(frame, response="y")
            self.first_scan_arrays = []
            self.retained = []

        def scan(self, batch_rows):
            if not self.fitting_started:
                yield from super().scan(batch_rows)
                return
            self.fitting_scans += 1
            if self.fitting_scans == 2:
                assert self.first_scan_arrays
                gc.collect()
                live = sum(
                    reference() is not None for reference in self.first_scan_arrays
                )
                assert live == 0, f"{live} first-scan batch arrays still referenced"
            for batch in super().scan(batch_rows):
                if self.fitting_scans == 1:
                    self.first_scan_arrays.extend(
                        weakref.ref(value)
                        for value in (
                            batch.y,
                            batch.weight,
                            batch.offset,
                            batch.valid,
                            batch.row_positions,
                        )
                    )
                    if retain and not self.retained:
                        self.retained.append(batch.y)
                yield batch

    source = ObservedSource(_frame())
    transfer = discrete_execution.to_device_discrete_design

    def mark_fitting(*args, **kwargs):
        source.fitting_started = True
        return transfer(*args, **kwargs)

    monkeypatch.setattr(discrete_execution, "to_device_discrete_design", mark_fitting)
    result = GAM("y ~ z + s(x,bs='cr',k=6)", sp=[0.65], control=_control(8)).fit(
        source, result="prediction"
    )
    return source, result


def test_completed_fitting_scan_releases_batch_arrays_before_residual_scan(
    monkeypatch,
) -> None:
    source, result = _observed_two_scan_fit(monkeypatch, retain=False)
    assert source.fitting_scans == result.source_scans == 2


def test_release_check_fails_when_a_first_scan_array_is_retained(monkeypatch) -> None:
    with pytest.raises(AssertionError, match="1 first-scan batch arrays still"):
        _observed_two_scan_fit(monkeypatch, retain=True)


def _frame(n_rows: int = 75) -> pd.DataFrame:
    position = np.arange(n_rows, dtype=float)
    x = (position + 0.3 * np.sin(position)) / n_rows
    z = np.cos(position * 0.31)
    y = 0.7 - 0.4 * z + np.sin(5.0 * x)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _control(
    batch_rows: int,
    *,
    m: int = 12,
    memory_budget_bytes: int = 512 * 1024 * 1024,
    selector_memmap_directory=None,
    uncertainty: str = "none",
) -> FitControl:
    return FitControl(
        execution="discrete",
        batch_rows=batch_rows,
        memory_budget_bytes=memory_budget_bytes,
        uncertainty=uncertainty,
        discretization=ContinuousDiscretizationControl(
            m=m, selector_memmap_directory=selector_memmap_directory
        ),
    )


def _snapped_frame(setup: ContinuousDiscreteSetup) -> pd.DataFrame:
    columns = {}
    for group in setup.compact.groups:
        selector = group.index.selector
        for name, values in zip(group.names, group.index.values, strict=True):
            columns[name] = values[selector]
    return pd.DataFrame(columns)


def test_public_fixed_sp_fit_matches_identical_explicit_discretized_design() -> None:
    frame = _frame()
    formula = "y ~ z + s(x, bs='cr', k=6)"
    weights = np.linspace(0.6, 1.4, len(frame))
    offset = np.linspace(-0.1, 0.2, len(frame))
    source = DataFrameRowSource(frame, response="y", weights=weights, offset=offset)
    setup = build_continuous_discrete_setup(
        parse_formula(formula), source, m=12, batch_rows=7
    )
    assert isinstance(setup, ContinuousDiscreteSetup)
    descriptor = build_continuous_discrete_design(
        setup, source, control=DiscreteDesignControl(batch_rows=7)
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    X = setup.predict_spec.build_predict_matrix(_snapped_frame(setup))
    assert setup.penalties is not None
    penalty = setup.penalties.materialize(np.log(np.array([0.65])))
    expected = np.linalg.solve(
        X.T @ (weights[:, None] * X) + penalty,
        X.T @ (weights * (frame.y.to_numpy() - offset)),
    )
    expected_residual = frame.y.to_numpy() - (X @ expected + offset)
    setup.cleanup()

    result = GAM(formula, sp=[0.65], control=_control(7, m=12), device="cpu").fit(
        DataFrameRowSource(frame, response="y", weights=weights, offset=offset),
        result="prediction",
    )
    np.testing.assert_allclose(
        result.coefficients, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.deviance,
        weights @ (expected_residual * expected_residual),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert result.execution_route == "discrete"
    assert result.smoothing_strategy == "fixed"
    assert result.discretization_strategy == "mgcv_range_grid"
    assert result.prediction_policy == "continuous_frozen_basis"
    assert result.source_scans == 2
    assert result.batches_scanned == 2 * math.ceil(len(frame) / 7)
    assert result.fit_peak_bytes is not None
    assert result.fit_memory_budget_bytes is not None
    assert result.fit_peak_bytes <= result.fit_memory_budget_bytes
    assert result.to_predictor().offset_was_nonzero


@pytest.mark.parametrize(
    ("family", "module", "plan_name"),
    [
        ("gaussian", discrete_execution, "_solver_plan"),
        ("poisson", discrete_regular_execution, "_regular_solver_plan"),
    ],
)
def test_result_reports_the_published_fit_accounting(
    monkeypatch, family, module, plan_name
) -> None:
    """Fit scans exclude setup scans; the workspace excludes retained setup."""
    frame = _frame()
    if family == "poisson":
        frame = frame.assign(y=np.round(np.exp(frame["y"])))
    source = _CountingSource(frame)
    plans = []
    setup_scans = []
    planner = getattr(module, plan_name)
    transfer = module.to_device_discrete_design

    def recording_plan(*args, **kwargs):
        plans.append(planner(*args, **kwargs))
        return plans[-1]

    def recording_transfer(*args, **kwargs):
        setup_scans.append(source.scans)
        return transfer(*args, **kwargs)

    monkeypatch.setattr(module, plan_name, recording_plan)
    monkeypatch.setattr(module, "to_device_discrete_design", recording_transfer)
    result = GAM(
        "y ~ z + s(x, bs='cr', k=6)", family=family, sp=[0.65], control=_control(7)
    ).fit(source, result="prediction")
    (plan,) = plans
    (scans_before_fit,) = setup_scans
    assert scans_before_fit > 0
    assert result.source_scans == source.scans - scans_before_fit
    assert result.source_scans >= 2
    assert result.batches_scanned == result.source_scans * math.ceil(len(frame) / 7)
    planned_workspace = (
        plan.descriptor_host_retained_bytes
        + plan.descriptor_device_peak_bytes
        + plan.solver_workspace_bytes
    )
    # The performance iteration adds its compiled proposal workspace to the
    # solver workspace after this first plan; the Gaussian route does not.
    if family == "gaussian":
        assert result.known_workspace_bytes == planned_workspace
    else:
        assert result.known_workspace_bytes > planned_workspace
    assert plan.setup_retained_bytes > 0
    assert result.fit_peak_bytes == (
        plan.setup_retained_bytes + result.known_workspace_bytes
    )
    assert result.fit_memory_budget_bytes == 512 * 1024 * 1024
    assert 0 < result.known_workspace_bytes < result.fit_peak_bytes
    assert result.fit_peak_bytes <= result.fit_memory_budget_bytes


def test_fit_is_batch_invariant_and_cleans_owned_selector_files(tmp_path) -> None:
    frame = _frame(61)
    formula = "y ~ z + s(x, bs='cr', k=6)"
    results = []
    for batch_rows in (1, 4, 13, 61):
        directory = tmp_path / str(batch_rows)
        directory.mkdir()
        result = GAM(
            formula,
            sp=[0.4],
            control=_control(batch_rows, m=10, selector_memmap_directory=directory),
        ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
        results.append(result)
        assert not list(directory.iterdir())
    for result in results[1:]:
        np.testing.assert_allclose(
            result.coefficients,
            results[0].coefficients,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            [result.deviance, result.scale, result.edf_total, result.score],
            [
                results[0].deviance,
                results[0].scale,
                results[0].edf_total,
                results[0].score,
            ],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


def test_nearly_exact_fit_uses_direct_residual_scan() -> None:
    x = np.linspace(-1.0, 1.0, 80)
    y = 1.25 - 2.5 * x + 1e-10 * np.sin(np.arange(len(x)))
    frame = pd.DataFrame({"y": y, "x": x})
    result = GAM("y ~ x", sp=[], control=_control(9, m=100)).fit(
        DataFrameRowSource(frame, response="y"), result="prediction"
    )
    X = np.column_stack((np.ones(len(x)), x))
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    stable = float(np.sum((y - X @ beta) ** 2))
    cancelled = float(y @ y - 2 * beta @ (X.T @ y) + beta @ (X.T @ X) @ beta)
    np.testing.assert_allclose(result.deviance, stable, rtol=1e-6, atol=1e-28)
    assert result.deviance > 0.0
    assert abs(cancelled - stable) > stable


def test_fit_workspace_rejects_before_device_transfer(monkeypatch) -> None:
    called = False

    def spy(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("device transfer must not run")

    monkeypatch.setattr(discrete_execution, "to_device_discrete_design", spy)
    with pytest.raises(MemoryError, match="Discrete fit workspace requires"):
        GAM(
            "y ~ z + s(x, bs='cr', k=6)",
            sp=[0.5],
            control=_control(8, memory_budget_bytes=1),
        ).fit(DataFrameRowSource(_frame(), response="y"), result="prediction")
    assert not called


def test_public_route_rejects_unimplemented_domains() -> None:
    source = DataFrameRowSource(_frame(), response="y")
    with pytest.raises(NotImplementedError, match="requires explicit fixed sp"):
        GAM("y ~ s(x)", control=_control(8)).fit(source, result="prediction")
    # Poisson/Binomial/Gamma and NB now run bgam.fitd's performance iteration
    # (tests/test_execution/test_discrete_regular.py, test_discrete_nb.py); a
    # noncanonical Gaussian remains unimplemented.
    with pytest.raises(NotImplementedError, match="supports Gaussian identity"):
        GAM("y ~ s(x)", family=Gaussian(link="log"), sp=[1.0], control=_control(8)).fit(
            source, result="prediction"
        )
    with pytest.raises(NotImplementedError, match="result='prediction' only"):
        GAM("y ~ s(x)", sp=[1.0], control=_control(8)).fit(source, result="full")


class _NoScanSource(DataFrameRowSource):
    def scan(self, _batch_rows):
        raise AssertionError("unsupported terms must reject before scanning")


def test_public_route_rejects_linked_smoothing_before_source_scan() -> None:
    source = _NoScanSource(_frame(), response="y")
    with pytest.raises(
        NotImplementedError, match=r"Linked smoothing parameters \(id=\)"
    ):
        GAM(
            "y ~ s(x, id=1, bs='cr', k=6) + s(z, id=1, bs='cr', k=6)",
            sp=[1.0, 1.0],
            control=_control(8),
        ).fit(source, result="prediction")


def test_public_route_rejects_multi_variable_random_effects() -> None:
    frame = _frame()
    frame["g"] = pd.Categorical(np.resize(np.array(("a", "b", "c")), len(frame)))
    control = FitControl(
        execution="discrete",
        batch_rows=8,
        discretization=ContinuousDiscretizationControl(m=12, factor_names=("g",)),
    )
    with pytest.raises(NotImplementedError, match="multi-variable random effect"):
        GAM("y ~ s(x, g, bs='re')", sp=[1.0], control=control).fit(
            _NoScanSource(frame, response="y"), result="prediction"
        )


def test_fixed_sp_fit_matches_pinned_bam_on_valid_layout(r_bridge) -> None:
    position = np.arange(72, dtype=float)
    x = (position + 0.2 * np.sin(position * 0.7)) / 72
    z = np.cos(position * 0.23)
    y = 0.4 - 0.3 * z + np.sin(4.5 * x) + 0.05 * np.cos(position)
    frame = pd.DataFrame({"y": y, "x": x, "z": z})
    formula = "y ~ z + s(x, bs='cr', k=6)"
    smoothing = np.array([0.55])
    expected = r_bridge.fit_bam_discrete_fixed_sp(formula, frame, smoothing, m=30)
    result = GAM(formula, sp=smoothing, control=_control(11, m=30)).fit(
        DataFrameRowSource(frame, response="y"), result="prediction"
    )
    np.testing.assert_allclose(
        result.coefficients,
        expected.coefficients,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        [result.deviance, result.scale, result.edf_total, result.score],
        [expected.deviance, expected.scale, expected.edf_total, expected.score],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_factor_tensor_fixed_sp_fit_matches_pinned_bam(r_bridge) -> None:
    n_rows = 72
    position = np.arange(1, n_rows + 1, dtype=float)
    x = (position - 0.5) / n_rows + 0.03 * np.sin(position)
    w = np.cos(position * 0.71) + position / n_rows
    g = pd.Series(
        pd.Categorical(
            np.resize(np.array(("low", "mid", "high")), n_rows),
            categories=("low", "mid", "high"),
            ordered=True,
        )
    )
    y = 0.2 + np.sin(2.5 * x) + np.where(g == "mid", 0.3 * w, -0.2 * w)
    frame = pd.DataFrame({"y": y, "x": x, "w": w, "g": g})
    formula = "y ~ te(x, w, bs='cr', k=3, by=g)"
    smoothing = np.array([0.35, 0.7, 0.45, 0.8])
    expected = r_bridge.fit_bam_discrete_fixed_sp(
        formula, frame, smoothing, factor_names=("g",), m=30
    )
    control = FitControl(
        execution="discrete",
        batch_rows=9,
        discretization=ContinuousDiscretizationControl(m=30, factor_names=("g",)),
    )
    result = GAM(formula, sp=smoothing, control=control).fit(
        DataFrameRowSource(frame, response="y"), result="prediction"
    )
    np.testing.assert_allclose(
        result.coefficients,
        expected.coefficients,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        [result.deviance, result.scale, result.edf_total, result.score],
        [expected.deviance, expected.scale, expected.edf_total, expected.score],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def _grouped_frame(grouping) -> pd.DataFrame:
    frame = _frame(60)
    frame["g"] = grouping
    return frame


_STRING_GROUPS = np.resize(np.array(["a", "b", "c"]), 60)


@pytest.mark.parametrize(
    "grouping",
    [
        _STRING_GROUPS.astype(object),
        _STRING_GROUPS,
        pd.Categorical(_STRING_GROUPS),
    ],
    ids=["object", "numpy-str", "categorical"],
)
def test_undeclared_string_grouping_names_column_and_factor_control(grouping):
    """A random-effect grouping column must be declared in factor_names."""
    source = DataFrameRowSource(_grouped_frame(grouping), response="y")
    with pytest.raises(TypeError) as raised:
        GAM("y ~ s(x) + s(g, bs='re')", sp=[1.0, 1.0], control=_control(8)).fit(
            source, result="prediction"
        )
    message = str(raised.value)
    assert message.startswith("Metric column 'g' must be real numeric data, got dtype")
    assert "If 'g' is a factor" in message
    assert "ContinuousDiscretizationControl(factor_names=('g',))" in message
    # Declaring it fits the factor random effect (one coefficient per level).
    declared = FitControl(
        execution="discrete",
        batch_rows=8,
        discretization=ContinuousDiscretizationControl(m=12, factor_names=("g",)),
    )
    result = GAM("y ~ s(x) + s(g, bs='re')", sp=[1.0, 1.0], control=declared).fit(
        source, result="prediction"
    )
    assert len(result.coefficients) == 1 + 9 + 3


def test_integer_coded_grouping_is_a_numeric_covariate_unless_declared():
    """Numeric codes are not guessed to be a factor, as mgcv treats a numeric
    ``s(g, bs='re')`` covariate: one coefficient, no error. Declared, they
    are a factor with one coefficient per level."""
    source = DataFrameRowSource(
        _grouped_frame(np.resize(np.array([1, 2, 3]), 60)), response="y"
    )
    numeric = GAM("y ~ s(x) + s(g, bs='re')", sp=[1.0, 1.0], control=_control(8)).fit(
        source, result="prediction"
    )
    assert len(numeric.coefficients) == 1 + 9 + 1
    declared = FitControl(
        execution="discrete",
        batch_rows=8,
        discretization=ContinuousDiscretizationControl(m=12, factor_names=("g",)),
    )
    factor = GAM("y ~ s(x) + s(g, bs='re')", sp=[1.0, 1.0], control=declared).fit(
        source, result="prediction"
    )
    assert len(factor.coefficients) == 1 + 9 + 3


@pytest.mark.parametrize(
    "values",
    [
        np.array([f"{value:.3f}" for value in np.linspace(0, 1, 60)]),
        pd.date_range("2026-01-01", periods=60).to_numpy(),
    ],
    ids=["numeric-strings", "datetime"],
)
def test_non_numeric_metric_data_reports_the_metric_column(values):
    """A smooth's metric covariate with non-numeric data is reported as such."""
    frame = _frame(60)
    frame["x"] = values
    dtype = frame["x"].dtype
    source = DataFrameRowSource(frame, response="y")
    with pytest.raises(TypeError) as raised:
        GAM("y ~ s(x) + s(z)", sp=[1.0, 1.0], control=_control(8)).fit(
            source, result="prediction"
        )
    message = str(raised.value)
    assert message.startswith(
        f"Metric column 'x' must be real numeric data, got dtype {dtype}."
    )
    assert "otherwise supply it as real numbers" in message


def test_non_finite_metric_data_keeps_its_own_error():
    frame = _frame(60)
    frame.loc[3, "x"] = np.nan
    source = DataFrameRowSource(frame, response="y")
    with pytest.raises(ValueError, match="Metric column 'x' contains non-finite"):
        GAM("y ~ s(x) + s(z)", sp=[1.0, 1.0], control=_control(8)).fit(
            source, result="prediction"
        )
