"""Source initial.spg reduction for the indexed operator EFS regime."""

import numpy as np
import pandas as pd
import pytest

from jaxgam.api import GAM
from jaxgam.control import EFSControl, FitControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs import efs_initial_log_lambda
from jaxgam.execution.operator_efs_start import prepare_indexed_operator_efs_start
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from tests.tolerances import STRICT


@pytest.mark.parametrize("family", [Gaussian(), Poisson()])
@pytest.mark.parametrize("coupled", [False, True])
def test_indexed_source_initial_spg_matches_dense_at_p_above_coordinate_cap(
    family, coupled
):
    p, n = 300, 600
    rng = np.random.default_rng(1409)
    group = np.arange(n) % p
    response = (
        1.2 + 0.4 * np.sin(group / 31) + rng.normal(0, 0.2, n)
        if isinstance(family, Gaussian)
        else rng.poisson(np.exp(0.2 + 0.3 * np.sin(group / 31)))
    )
    frame = pd.DataFrame(
        {
            "y": response,
            "g": pd.Categorical(
                [f"g{index}" for index in group],
                categories=[f"g{index}" for index in range(p)],
            ),
            "x": np.linspace(-1, 1, n),
        }
    )
    spec = parse_formula("y~x+s(g,bs='re')" if coupled else "y~0+s(g,bs='re')")
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(batch_rows=64, memory_budget_bytes=4 << 20),
    )
    start = prepare_indexed_operator_efs_start(
        model, source, family, memory_budget_bytes=4 << 20
    )
    expected = efs_initial_log_lambda(ModelSetup.build(spec, frame), family)
    np.testing.assert_allclose(
        start.log_lambda, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert start.source_scans == 1
    assert start.source_batches == 10
    assert start.peak_bound_bytes < 4 << 20


def test_indexed_start_preflights_before_source_replay():
    p = 257
    frame = pd.DataFrame(
        {
            "y": np.ones(2 * p),
            "g": pd.Categorical(np.arange(2 * p) % p),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~0+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=64, memory_budget_bytes=4 << 20),
    )
    with pytest.raises(MemoryError, match="start exceeds"):
        prepare_indexed_operator_efs_start(
            model, source, Poisson(), memory_budget_bytes=8192
        )


def test_indexed_start_rejects_changed_source_and_nonunit_rows():
    frame = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 1.0], "g": pd.Categorical(["a", "b", "a", "b"])}
    )
    spec = parse_formula("y~0+s(g,bs='re')")
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(batch_rows=2, memory_budget_bytes=1 << 20),
    )
    changed = DataFrameRowSource(frame.assign(y=[2.0, 2.0, 3.0, 1.0]), response="y")
    with pytest.raises(RuntimeError, match="source changed"):
        prepare_indexed_operator_efs_start(
            model, changed, Poisson(), memory_budget_bytes=1 << 20
        )

    weighted = DataFrameRowSource(frame, response="y", weights=[1.0, 2.0, 1.0, 1.0])
    weighted_model = prepare_operator_model(
        spec,
        weighted,
        control=OperatorPreparationControl(batch_rows=2, memory_budget_bytes=1 << 20),
    )
    with pytest.raises(NotImplementedError, match="unit, unoffset rows"):
        prepare_indexed_operator_efs_start(
            weighted_model, weighted, Poisson(), memory_budget_bytes=1 << 20
        )


def test_indexed_start_rejects_multiple_effects_before_replay():
    frame = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 1.0],
            "g": pd.Categorical(["a", "b", "a", "b"]),
            "h": pd.Categorical(["u", "u", "v", "v"]),
        }
    )
    spec = parse_formula("y~0+s(g,bs='re')+s(h,bs='re')")
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(batch_rows=2, memory_budget_bytes=1 << 20),
    )
    with pytest.raises(NotImplementedError, match="one indexed effect"):
        prepare_indexed_operator_efs_start(
            model, source, Poisson(), memory_budget_bytes=1 << 20
        )


def test_public_indexed_operator_efs_retains_score_and_prediction():
    p, n = 300, 600
    rng = np.random.default_rng(37)
    group = np.arange(n) % p
    response = rng.poisson(np.exp(0.2 + 0.3 * np.sin(group / 31)))
    frame = pd.DataFrame(
        {
            "y": response,
            "g": pd.Categorical(
                [f"g{index}" for index in group],
                categories=[f"g{index}" for index in range(p)],
            ),
        }
    )
    formula = "y~0+s(g,bs='re')"
    fitted = GAM(
        formula,
        family="poisson",
        optimizer="efs",
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=64,
            memory_budget_bytes=128 << 20,
            operator=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
        ),
    ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
    assert fitted.converged
    assert fitted.lambda_strategy == "efs_reml"
    assert fitted.execution_route == "operator"
    assert fitted.score is not None
    assert fitted.score_error_bound is not None
    assert fitted.score_scale == fitted.scale == 1.0
    assert fitted.edf_total is not None
    assert fitted.source_scans is not None
    assert fitted.source_scans > 1
    assert fitted.batches_scanned is not None
    assert fitted.batches_scanned > 10
    assert fitted.known_workspace_bytes < 128 << 20
    np.testing.assert_allclose(
        fitted.predict(frame.iloc[:10]),
        np.exp(
            ModelSetup.build(parse_formula(formula), frame).X[:10] @ fitted.coefficients
        ),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_public_operator_efs_reports_actual_partitioned_source_batches():
    p, n = 257, 514
    group = np.arange(n) % p
    rng = np.random.default_rng(7734)
    response = rng.poisson(np.exp(0.2 + 0.2 * np.sin(group / 31)))
    frame = pd.DataFrame(
        {
            "y": response,
            "g": pd.Categorical(
                [f"g{index}" for index in group],
                categories=[f"g{index}" for index in range(p)],
            ),
        }
    )

    class _PartitionedSource(DataFrameRowSource):
        def __init__(self, data):
            super().__init__(data, response="y")
            self.actual_scans = 0
            self.actual_batches = 0

        def scan(self, batch_rows):
            self.actual_scans += 1
            for batch in super().scan(min(batch_rows, 32)):
                self.actual_batches += 1
                yield batch

    source = _PartitionedSource(frame)
    fitted = GAM(
        "y~0+s(g,bs='re')",
        family="poisson",
        optimizer="efs",
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=64,
            memory_budget_bytes=128 << 20,
            operator=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
        ),
    ).fit(source, result="prediction")
    assert fitted.converged
    assert fitted.source_scans == source.actual_scans
    assert fitted.batches_scanned == source.actual_batches
    assert fitted.batches_scanned > fitted.source_scans * ((n + 63) // 64)


def test_public_coupled_operator_efs_retains_score_and_prediction():
    from tests.test_execution.test_operator_efs_arrowhead import (
        calibrated_coupled_frame,
    )

    frame = calibrated_coupled_frame(300, 32031, 2)
    formula = "y~x+s(g,bs='re')"
    fitted = GAM(
        formula,
        family="poisson",
        optimizer="efs",
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=64,
            memory_budget_bytes=128 << 20,
            efs=EFSControl(outer_limit=20, pirls_tolerance=1e-11),
            operator=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
        ),
    ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
    assert fitted.converged
    assert fitted.lambda_strategy == "efs_reml"
    assert fitted.execution_route == "operator"
    assert fitted.score is not None
    assert fitted.score_error_bound is not None
    assert fitted.score_scale == fitted.scale == 1.0
    assert fitted.edf_total is not None
    assert fitted.source_scans is not None
    assert fitted.source_scans > 1
    assert fitted.batches_scanned is not None
    assert fitted.batches_scanned > 10
    assert fitted.known_workspace_bytes < 128 << 20
    expected = np.exp(
        ModelSetup.build(parse_formula(formula), frame).X[:10] @ fitted.coefficients
    )
    np.testing.assert_allclose(
        fitted.predict(frame.iloc[:10]),
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize(("family", "batch_rows"), [("gaussian", 64), ("poisson", 32)])
def test_public_operator_efs_rejects_unreleased_regime_before_source_replay(
    family, batch_rows
):
    frame = pd.DataFrame({"y": [1.0], "g": pd.Categorical(["g0"])})

    class _UnreadableSource(DataFrameRowSource):
        def scan(self, _batch_rows):
            pytest.fail("Unsupported operator EFS scanned the source")

        def fingerprint(self):
            pytest.fail("Unsupported operator EFS fingerprinted the source")

    source = _UnreadableSource(frame, response="y")
    model = GAM(
        "y~0+s(g,bs='re')",
        family=family,
        optimizer="efs",
        control=FitControl(
            execution="operator", linear_solver="pcg", batch_rows=batch_rows
        ),
    )
    with pytest.raises(NotImplementedError, match=r"Calibrated|supports calibrated"):
        model.fit(source, result="prediction")


def _indexed_poisson_frame(levels: int, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = np.arange(levels * rows) % levels
    return pd.DataFrame(
        {
            "y": rng.poisson(np.exp(0.2 + 0.3 * np.sin(group / 31))),
            "g": pd.Categorical.from_codes(
                group, categories=[f"g{index}" for index in range(levels)]
            ),
        }
    )


def test_public_operator_efs_labels_admitted_bound_beyond_p512():
    from jaxgam.execution.operator_efs_controller import (
        CALIBRATED_OPERATOR_EFS_DOMAIN as domain,
    )

    frame = _indexed_poisson_frame(1100, domain.minimum_rows_per_level, 5401)
    fitted = GAM(
        "y~0+s(g,bs='re')",
        family="poisson",
        optimizer="efs",
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=domain.maximum_batch_rows,
            operator=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
        ),
    ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
    assert fitted.converged
    assert fitted.score_bound_method == "indexed_poisson_empirical_v2"
    assert fitted.score_error_bound is not None
    assert fitted.execution_route == "operator"
    assert fitted.lambda_strategy == "efs_reml"


def test_public_operator_efs_rejects_levels_beyond_calibration_before_efs():
    from jaxgam.execution.operator_efs_controller import (
        CALIBRATED_OPERATOR_EFS_DOMAIN as domain,
    )

    frame = _indexed_poisson_frame(
        domain.maximum_levels + 1, domain.minimum_rows_per_level, 5402
    )

    class _CountingSource(DataFrameRowSource):
        def __init__(self, data):
            super().__init__(data, response="y")
            self.scans = 0

        def scan(self, batch_rows):
            self.scans += 1
            yield from super().scan(batch_rows)

    source = _CountingSource(frame)
    model = GAM(
        "y~0+s(g,bs='re')",
        family="poisson",
        optimizer="efs",
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=domain.maximum_batch_rows,
        ),
    )
    with pytest.raises(NotImplementedError, match="Calibrated operator EFS requires"):
        model.fit(source, result="prediction")
    preparation = _CountingSource(frame)
    prepared = prepare_operator_model(
        parse_formula("y~0+s(g,bs='re')"),
        preparation,
        control=OperatorPreparationControl(
            domain.maximum_batch_rows, (512 << 20) // 8, 64
        ),
    )
    assert source.scans == preparation.scans == prepared.started_preparation_scans


def test_public_operator_efs_rejects_batch_rows_above_calibration_before_replay():
    from jaxgam.execution.operator_efs_controller import (
        CALIBRATED_OPERATOR_EFS_DOMAIN as domain,
    )

    frame = _indexed_poisson_frame(300, 1, 5403)

    class _UnreadableSource(DataFrameRowSource):
        def scan(self, _batch_rows):
            pytest.fail("Unsupported operator EFS scanned the source")

        def fingerprint(self):
            pytest.fail("Unsupported operator EFS fingerprinted the source")

    model = GAM(
        "y~0+s(g,bs='re')",
        family="poisson",
        optimizer="efs",
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=domain.maximum_batch_rows + 1,
        ),
    )
    with pytest.raises(NotImplementedError, match="Calibrated operator EFS requires"):
        model.fit(_UnreadableSource(frame, response="y"), result="prediction")


@pytest.mark.parametrize("edge", ["levels_below", "rows_above", "modest_above"])
def test_public_operator_efs_rejects_shapes_outside_calibration_before_efs(edge):
    """Every admission limit rejects after preparation, before any EFS refit."""
    from jaxgam.execution.operator_efs_controller import (
        CALIBRATED_OPERATOR_EFS_DOMAIN as domain,
    )
    from jaxgam.execution.operator_efs_controller import (
        CALIBRATED_OPERATOR_EFS_MAX_MODEST,
    )

    formula = "y~0+s(g,bs='re')"
    if edge == "levels_below":
        frame = _indexed_poisson_frame(domain.minimum_levels - 1, 1, 5404)
    elif edge == "rows_above":
        frame = _indexed_poisson_frame(
            domain.minimum_levels, domain.maximum_rows_per_level + 1, 5405
        )
    else:
        frame = _indexed_poisson_frame(domain.minimum_levels + 43, 2, 5406)
        rng = np.random.default_rng(5407)
        names = [f"x{index}" for index in range(CALIBRATED_OPERATOR_EFS_MAX_MODEST)]
        for name in names:
            frame[name] = rng.uniform(-1.0, 1.0, len(frame))
        # The intercept plus one column per name exceeds the measured limit.
        formula = "y~" + "+".join(names) + "+s(g,bs='re')"

    class _CountingSource(DataFrameRowSource):
        def __init__(self, data):
            super().__init__(data, response="y")
            self.scans = 0

        def scan(self, batch_rows):
            self.scans += 1
            yield from super().scan(batch_rows)

    source = _CountingSource(frame)
    model = GAM(
        formula,
        family="poisson",
        optimizer="efs",
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=domain.minimum_batch_rows,
        ),
    )
    with pytest.raises(NotImplementedError, match="Calibrated operator EFS requires"):
        model.fit(source, result="prediction")
    preparation = _CountingSource(frame)
    prepared = prepare_operator_model(
        parse_formula(formula),
        preparation,
        control=OperatorPreparationControl(
            domain.minimum_batch_rows, (512 << 20) // 8, 64
        ),
    )
    assert source.scans == preparation.scans == prepared.started_preparation_scans


def test_public_operator_efs_without_admitted_bound_is_not_converged(monkeypatch):
    """An unbounded score comparison stops truthfully on the public result."""
    import jaxgam.execution.operator_efs_controller as controller

    monkeypatch.setattr(
        controller,
        "admitted_operator_efs_score_bound",
        lambda *_args: (None, "unavailable"),
    )
    frame = _indexed_poisson_frame(300, 2, 5408)
    fitted = GAM(
        "y~0+s(g,bs='re')",
        family="poisson",
        optimizer="efs",
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=64,
            operator=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
        ),
    ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
    assert not fitted.converged
    assert fitted.convergence_info == "indeterminate_score"
    assert fitted.score_error_bound is None
    assert fitted.score_bound_method == "unavailable"
    assert fitted.lambda_strategy == "efs_reml"
    assert np.isfinite(fitted.score)
    # The last accepted state is the shifted start: no trial was accepted.
    np.testing.assert_allclose(
        np.log(fitted.smoothing_params),
        efs_initial_log_lambda(
            ModelSetup.build(parse_formula("y~0+s(g,bs='re')"), frame), Poisson()
        )
        + 2.5,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
