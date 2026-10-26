"""Pinned range-grid compression and bounded replay contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.continuous_discretize import (
    RangeGridIndex,
    _r_number_text,
    range_grid_index_columns,
)
from jaxgam.data.discretize import DiscretizationBudget, ExactIndexUnavailable
from jaxgam.data.source import ArrayRowSource, RowBatch
from tests.helpers import r_available
from tests.r_bridge import RBridge


def _result(
    columns,
    names,
    *,
    factor_names=(),
    m=None,
    batch_rows=3,
    budget=None,
    selector_memmap=None,
    shuffle_seed=8547,
) -> RangeGridIndex:
    result = range_grid_index_columns(
        ArrayRowSource(columns),
        names,
        factor_names=factor_names,
        m=m,
        batch_rows=batch_rows,
        budget=budget,
        selector_memmap=selector_memmap,
        shuffle_seed=shuffle_seed,
    )
    assert isinstance(result, RangeGridIndex)
    return result


def test_exact_first_preserves_values_and_identity_when_all_rows_unique() -> None:
    x = np.array((0.125, 0.25, 0.5, 0.75))
    result = _result({"x": x}, ("x",), m=4, batch_rows=2)
    assert not result.rounded
    assert result.points_per_metric == 4
    np.testing.assert_array_equal(result.selector, np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(result.values[0], x)


def test_exact_first_does_not_round_repeated_rows_below_grid_cap() -> None:
    x = np.array((0.125, 0.25, 0.125, 0.25, 0.125))
    result = _result({"x": x}, ("x",), m=3, batch_rows=2)
    assert not result.rounded
    np.testing.assert_array_equal(result.values[0][result.selector], x)


@pytest.mark.parametrize(
    ("n_columns", "m", "expected"),
    [
        (1, None, 1000),
        (2, None, 100),
        (3, None, 25),
        (1, 16, 16),
        (2, 16, 5),
        (3, 27, 4),
    ],
)
def test_grid_dimension_and_m_conversion(
    n_columns: int, m: int | None, expected: int
) -> None:
    columns = {f"x{j}": np.zeros(2) for j in range(n_columns)}
    result = _result(columns, tuple(columns), m=m)
    assert result.points_per_metric == expected
    assert result.metric_names == tuple(columns)


def test_ties_use_nearest_even_grid_indices() -> None:
    x = np.array((0.0, 0.25, 0.75, 1.0))
    result = _result({"x": x}, ("x",), m=3, batch_rows=2)
    assert result.rounded
    np.testing.assert_array_equal(
        result.values[0][result.selector], np.array((0.0, 0.0, 1.0, 1.0))
    )


def test_factor_values_and_scalar_types_survive_metric_rounding() -> None:
    factor = np.resize(np.array(("a", 12, 3.5), dtype=object), 12)
    metric = np.linspace(-1.0, 1.0, 12)
    result = _result(
        {
            "group": pd.Series(
                pd.Categorical(factor, categories=("a", 12, 3.5), ordered=True)
            ),
            "x": metric,
        },
        ("group", "x"),
        factor_names=("group",),
        m=4,
        batch_rows=4,
    )
    assert result.rounded
    assert result.factor_names == ("group",)
    assert result.factor_levels == (("a", 12, 3.5),)
    assert result.ordered_factors == frozenset(("group",))
    assert result.metric_names == ("x",)
    reconstructed = result.values[0][result.selector]
    np.testing.assert_array_equal(reconstructed, factor)
    assert {type(value) for value in result.values[0]} == {str, int, float}


def test_shuffle_is_repeatable_recorded_and_does_not_touch_global_rng() -> None:
    source = {"x": np.resize(np.array((0.0, 0.5, 1.0)), 12)}
    before = np.random.get_state()  # noqa: NPY002 - assert untouched global state
    left = _result(source, ("x",), m=3, shuffle_seed=23)
    after = np.random.get_state()  # noqa: NPY002 - assert untouched global state
    right = _result(source, ("x",), m=3, shuffle_seed=23)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(left.selector, right.selector)
    np.testing.assert_array_equal(left.values[0], right.values[0])
    assert left.rng_policy == "r_4_5_mt_rejection"
    assert left.shuffle_seed == 23


def test_seed_8547_exact_unique_order_and_selectors_match_r45() -> None:
    encounter_order = np.array((9.0, 0.0, 7.0, 2.0, 5.0, 4.0, 3.0, 6.0, 1.0, 8.0))
    x = np.tile(encounter_order, 4)
    result = _result({"x": x}, ("x",), m=10, batch_rows=3)
    np.testing.assert_array_equal(
        result.values[0], np.array((0.0, 2.0, 4.0, 6.0, 8.0, 5.0, 7.0, 3.0, 1.0, 9.0))
    )
    np.testing.assert_array_equal(
        result.selector[:10], np.array((9, 0, 6, 1, 5, 2, 7, 3, 8, 4))
    )


def test_selected_reordered_repeated_rows_replay_across_batch_boundaries() -> None:
    values = np.linspace(0.0, 1.0, 48)
    selection = np.array((7, 2, 40, 1, 7, 47, 2))
    source = ArrayRowSource({"x": values}, row_selection=selection)
    left = range_grid_index_columns(source, ("x",), m=3, batch_rows=2)
    right = range_grid_index_columns(source, ("x",), m=3, batch_rows=4)
    assert isinstance(left, RangeGridIndex)
    assert isinstance(right, RangeGridIndex)
    np.testing.assert_array_equal(
        left.values[0][left.selector], right.values[0][right.selector]
    )
    np.testing.assert_array_equal(left.selector, right.selector)
    assert left.n_rows == len(selection)


def test_position_metadata_and_advertised_row_count_are_checked() -> None:
    class PositionSource:
        def __init__(self, n_rows, positions):
            self.n_rows = n_rows
            self.positions = np.asarray(positions)

        def fingerprint(self) -> str:
            return "stable"

        def scan(self, _batch_rows: int):
            size = len(self.positions)
            yield RowBatch(
                MappingProxyType({"x": np.arange(size, dtype=float)}),
                None,
                np.ones(size),
                np.zeros(size),
                np.ones(size, dtype=bool),
                self.positions,
            )

    cases = (
        (PositionSource(1, [0.0]), "invalid row positions"),
        (PositionSource(1, [-1]), "non-negative"),
        (PositionSource(1, [4, 4]), "more than"),
        (PositionSource(2, [4]), "fewer than"),
    )
    for source, match in cases:
        with pytest.raises(RuntimeError, match=match):
            range_grid_index_columns(source, ("x",), m=3, batch_rows=2)


def test_changed_source_between_analysis_and_exact_replay_is_rejected(
    tmp_path: Path,
) -> None:
    class ChangedReplaySource:
        n_rows = 4

        def __init__(self) -> None:
            self.fingerprint_calls = 0
            self.scan_calls = 0

        def fingerprint(self) -> str:
            self.fingerprint_calls += 1
            return "analysis" if self.fingerprint_calls <= 3 else "replay"

        def scan(self, _batch_rows: int):
            self.scan_calls += 1
            values = np.arange(4.0) + 10 * (self.scan_calls - 1)
            yield RowBatch(
                MappingProxyType({"x": values}),
                None,
                np.ones(4),
                np.zeros(4),
                np.ones(4, dtype=bool),
                np.arange(4),
            )

    path = tmp_path / "changed.npy"
    with pytest.raises(RuntimeError, match="changed between range-grid scans"):
        range_grid_index_columns(
            ChangedReplaySource(),  # type: ignore[arg-type]
            ("x",),
            m=4,
            batch_rows=4,
            selector_memmap=path,
        )
    assert not path.exists()


def test_shared_budgets_reject_growth_and_clean_failed_memmap(tmp_path: Path) -> None:
    factor = np.array(tuple(f"level-{j}-" + "x" * 100 for j in range(8)), dtype=object)
    unavailable = range_grid_index_columns(
        ArrayRowSource({"group": factor}),
        ("group",),
        factor_names=("group",),
        budget=DiscretizationBudget(max_table_bytes=512),
        batch_rows=2,
    )
    assert isinstance(unavailable, ExactIndexUnavailable)
    assert "factor-level analysis" in unavailable.reason

    x = np.linspace(0.0, 1.0, 20)
    success = _result({"x": x}, ("x",), m=3, batch_rows=3)
    path = tmp_path / "failed.npy"
    failed = range_grid_index_columns(
        ArrayRowSource({"x": x}),
        ("x",),
        m=3,
        batch_rows=3,
        selector_memmap=path,
        budget=DiscretizationBudget(
            max_table_bytes=success.construction_peak_bytes - 1
        ),
    )
    assert isinstance(failed, ExactIndexUnavailable)
    assert not path.exists()


def test_rounding_workspace_is_rejected_before_transform_scan(monkeypatch) -> None:
    transformed_scans: list[int] = []

    def unexpected_scan(_self, batch_rows):
        transformed_scans.append(batch_rows)
        raise AssertionError("rounded source must not be evaluated over budget")

    monkeypatch.setattr(
        "jaxgam.data.continuous_discretize._RangeGridSource.scan", unexpected_scan
    )
    result = range_grid_index_columns(
        ArrayRowSource({"x": np.linspace(0.0, 1.0, 8)}),
        ("x",),
        m=3,
        batch_rows=4,
        budget=DiscretizationBudget(max_batch_bytes=128),
    )
    assert isinstance(result, ExactIndexUnavailable)
    assert "range-grid indexing batch workspace" in result.reason
    assert transformed_scans == []


def test_range_grid_module_imports_without_jax() -> None:
    root = Path(__file__).resolve().parents[2] / "jaxgam"
    script = f"""
import sys
import types
from pathlib import Path
root = Path({str(root)!r})
for name, path in (("jaxgam", root), ("jaxgam.data", root / "data")):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
from jaxgam.data.continuous_discretize import range_grid_index_columns
assert callable(range_grid_index_columns)
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_tie_rounding_reconstruction_matches_pinned_compress_df(r_bridge) -> None:
    frame = pd.DataFrame({"x": (0.0, 0.25, 0.75, 1.0)})
    python = _result({"x": frame["x"].to_numpy()}, ("x",), m=3, batch_rows=2)
    reference = r_bridge.compress_dataframe(frame, m=3)
    np.testing.assert_array_equal(python.selector, reference.selector)
    np.testing.assert_allclose(
        python.values[0], reference.values["x"], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        python.values[0][python.selector],
        reference.reconstruct()["x"],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_factor_metric_reconstruction_matches_pinned_compress_df(r_bridge) -> None:
    frame = pd.DataFrame(
        {
            "group": np.resize(np.array(("a", "b"), dtype=object), 12),
            "x": np.linspace(-2.0, 2.0, 12),
        }
    )
    python = _result(
        {name: frame[name] for name in frame},
        ("group", "x"),
        factor_names=("group",),
        m=9,
        batch_rows=4,
    )
    reference = r_bridge.compress_dataframe(frame, factor_names=("group",), m=9)
    np.testing.assert_array_equal(python.selector, reference.selector)
    np.testing.assert_array_equal(python.values[0], reference.values["group"])
    np.testing.assert_allclose(
        python.values[1], reference.values["x"], rtol=0.0, atol=1e-15
    )
    reference_rows = reference.reconstruct()
    reconstructed = pd.DataFrame(
        {
            name: value[python.selector]
            for name, value in zip(python.names, python.values, strict=True)
        }
    )
    np.testing.assert_array_equal(reconstructed["group"], reference_rows["group"])
    np.testing.assert_allclose(
        reconstructed["x"], reference_rows["x"], rtol=0.0, atol=1e-15
    )


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_joint_exact_uniquecombs_order_and_selector_match_pinned_r(r_bridge) -> None:
    rows = np.array(
        (
            (10.0, 2.0),
            (2.0, 10.0),
            (1.0, 20.0),
            (10.0, 1.0),
            (2.0, 2.0),
            (1.0, 2.0),
        )
    )
    rows = np.vstack((rows, rows[[3, 0, 5, 1]]))
    frame = pd.DataFrame(rows, columns=("x", "z"))
    python = _result(
        {name: frame[name] for name in frame},
        ("x", "z"),
        m=100,
        batch_rows=3,
    )
    reference = r_bridge.compress_dataframe(frame, m=100)
    assert not python.rounded
    np.testing.assert_array_equal(python.selector, reference.selector)
    np.testing.assert_allclose(
        np.column_stack(python.values),
        reference.values[["x", "z"]],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_constant_metric_in_joint_rounding_rejects_pinned_invalid_grid(
    r_bridge,
) -> None:
    frame = pd.DataFrame({"x": np.ones(30), "z": np.linspace(-1.0, 1.0, 30)})
    reference = r_bridge.compress_dataframe(frame, m=4)
    assert reference.values["x"].isna().all()
    with pytest.raises(ValueError, match=r"undefined range grid.*constant metric"):
        range_grid_index_columns(
            ArrayRowSource({name: frame[name] for name in frame}),
            ("x", "z"),
            m=4,
            batch_rows=5,
        )


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_joint_numeric_token_collision_matches_uniquecombs(r_bridge) -> None:
    frame = pd.DataFrame(
        {
            "x": np.array((1.0, np.nextafter(1.0, 2.0), 2.0, 2.0)),
            "z": np.zeros(4),
        }
    )
    python = _result(
        {name: frame[name] for name in frame},
        ("x", "z"),
        m=100,
        batch_rows=2,
    )
    reference = r_bridge.compress_dataframe(frame, m=100)
    assert python.n_unique == 2
    np.testing.assert_array_equal(python.selector, reference.selector)
    np.testing.assert_allclose(
        np.column_stack(python.values),
        reference.values[["x", "z"]],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (8956883870379507.0, "8956883870379507"),
        (8956883870379508.0, "8956883870379508"),
        (1.4010301603817152e18, "1401030160381715200"),
        (np.nextafter(0.0, 1.0), "4.94065645841247e-324"),
        (-1e-320, "-9.99988867182683e-321"),
        (1e-4, "1e-04"),
        (1e-5, "1e-05"),
    ],
)
def test_numeric_tokens_match_r_string_from_real(value: float, expected: str) -> None:
    assert _r_number_text(value) == expected


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_numeric_tokens_match_pinned_r_across_binary_scales() -> None:
    rng = np.random.default_rng(90813)
    values = np.r_[
        rng.normal(size=2000) * 10.0 ** rng.integers(-290, 291, 2000),
        [
            0.0,
            -0.0,
            np.nextafter(0.0, 1.0),
            -np.nextafter(0.0, 1.0),
            np.finfo(float).tiny,
            np.nextafter(np.finfo(float).tiny, 0.0),
            1e-320,
            1.0,
            np.nextafter(1.0, 2.0),
            1e-4,
            1e-5,
            1e5,
            1e6,
        ],
    ]
    reference = RBridge().source_numeric_tokens(values)
    observed = np.array([_r_number_text(value) for value in values])
    np.testing.assert_array_equal(observed, reference)


def test_large_fixed_tokens_remain_distinct_in_joint_compaction() -> None:
    x = np.array((8956883870379507.0, 8956883870379508.0, 1.0, 1.0))
    result = _result({"x": x, "z": np.zeros(4)}, ("x", "z"), m=100, batch_rows=2)
    assert result.n_unique == 3
    np.testing.assert_array_equal(
        result.values[0], np.array((1.0, 8956883870379508.0, 8956883870379507.0))
    )
    np.testing.assert_array_equal(result.selector, np.array((2, 1, 0, 0)))


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_token_deduplication_precedes_grid_threshold(r_bridge) -> None:
    frame = pd.DataFrame(
        {
            "x": np.array((1.0, np.nextafter(1.0, 2.0), 2.0, 3.0, 4.0)),
            "z": np.zeros(5),
        }
    )
    python = _result(
        {name: frame[name] for name in frame},
        ("x", "z"),
        m=2,
        batch_rows=2,
    )
    reference = r_bridge.compress_dataframe(frame, m=2)
    assert not python.rounded
    assert python.n_unique == 4
    np.testing.assert_array_equal(python.selector, reference.selector)
    np.testing.assert_allclose(
        np.column_stack(python.values),
        reference.values[["x", "z"]],
        rtol=0.0,
        atol=0.0,
    )
