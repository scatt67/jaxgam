"""Exact CPU indexing contracts for discrete operators."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np

from jaxgam.data.discretize import (
    DiscretizationBudget,
    ExactIndexUnavailable,
    exact_index_columns,
)
from jaxgam.data.source import ArrayRowSource, RowBatch


def test_exact_index_preserves_repeats_order_and_readonly_ownership() -> None:
    source = ArrayRowSource(
        {
            "x": np.array((2.0, 1.0, 2.0, 3.0, 1.0)),
            "g": np.array(("b", "a", "b", "a", "a"), dtype=object),
        }
    )
    table = exact_index_columns(source, ("x", "g"), budget=DiscretizationBudget())
    assert not isinstance(table, ExactIndexUnavailable)
    assert table.n_unique == 3
    reconstructed = np.column_stack(
        (table.values[0][table.selector], table.values[1][table.selector])
    )
    np.testing.assert_array_equal(
        reconstructed[:, 0].astype(float), source._columns["x"]
    )
    np.testing.assert_array_equal(reconstructed[:, 1], source._columns["g"])
    with np.testing.assert_raises(ValueError):
        table.values[0][0] = 99.0
    with np.testing.assert_raises(ValueError):
        table.selector[0] = 1


def test_exact_index_rejects_selector_and_unique_growth_before_table_allocation() -> (
    None
):
    source = ArrayRowSource({"x": np.arange(8.0)})
    selector_limited = exact_index_columns(
        source,
        ("x",),
        budget=DiscretizationBudget(max_selector_bytes=4, max_unique_rows=10),
    )
    assert isinstance(selector_limited, ExactIndexUnavailable)
    assert "selector" in selector_limited.reason
    unique_limited = exact_index_columns(
        source,
        ("x",),
        budget=DiscretizationBudget(max_selector_bytes=128, max_unique_rows=3),
    )
    assert isinstance(unique_limited, ExactIndexUnavailable)
    assert "unique" in unique_limited.reason


def test_exact_index_uses_caller_owned_memmap_for_selectors(tmp_path: Path) -> None:
    source = ArrayRowSource({"x": np.resize(np.array((0.0, 1.0)), 11)})
    path = tmp_path / "selectors.npy"
    table = exact_index_columns(
        source,
        ("x",),
        budget=DiscretizationBudget(max_selector_bytes=12),
        selector_memmap=path,
    )
    assert not isinstance(table, ExactIndexUnavailable)
    assert table.storage == "memmap"
    assert path.exists()
    np.testing.assert_array_equal(
        table.batch_selector(np.array((0, 3, 10))),
        np.array((0, 1, 0), dtype=np.int32),
    )
    original = b"caller-owned"
    collision = tmp_path / "collision.npy"
    collision.write_bytes(original)
    with np.testing.assert_raises(FileExistsError):
        exact_index_columns(
            source,
            ("x",),
            budget=DiscretizationBudget(),
            selector_memmap=collision,
        )
    assert collision.read_bytes() == original


def test_exact_index_uses_scan_ordinals_for_reordered_repeated_selection() -> None:
    values = np.arange(48.0)
    selection = np.array((7, 2, 40, 1, 7))
    source = ArrayRowSource({"x": values}, row_selection=selection)
    left = exact_index_columns(
        source, ("x",), budget=DiscretizationBudget(), batch_rows=2
    )
    right = exact_index_columns(
        source, ("x",), budget=DiscretizationBudget(), batch_rows=3
    )
    assert not isinstance(left, ExactIndexUnavailable)
    assert not isinstance(right, ExactIndexUnavailable)
    np.testing.assert_array_equal(left.values[0][left.selector], values[selection])
    np.testing.assert_array_equal(right.values[0][right.selector], values[selection])
    np.testing.assert_array_equal(left.selector, right.selector)


def test_exact_index_validates_position_metadata_and_advertised_row_count() -> None:
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
        with np.testing.assert_raises_regex(RuntimeError, match):
            exact_index_columns(
                source,  # type: ignore[arg-type]
                ("x",),
                budget=DiscretizationBudget(),
                batch_rows=2,
            )


def test_exact_index_rejects_source_lineage_change_during_scan() -> None:
    class ChangedSource:
        n_rows = 1
        fingerprint_calls = 0

        def fingerprint(self) -> str:
            self.fingerprint_calls += 1
            return str(self.fingerprint_calls)

        def scan(self, _batch_rows: int):
            yield RowBatch(
                MappingProxyType({"x": np.array((1.0,))}),
                None,
                np.ones(1),
                np.zeros(1),
                np.ones(1, dtype=bool),
                np.array((40,)),
            )

    with np.testing.assert_raises_regex(RuntimeError, "changed"):
        exact_index_columns(
            ChangedSource(),  # type: ignore[arg-type]
            ("x",),
            budget=DiscretizationBudget(),
            batch_rows=1,
        )


def test_exact_index_checks_override_and_row_cap_separately() -> None:
    source = ArrayRowSource({"x": np.arange(3.0)})
    with np.testing.assert_raises(ValueError):
        exact_index_columns(
            source,
            ("x",),
            budget=DiscretizationBudget(),
            max_unique_rows=0,
        )
    unavailable = exact_index_columns(
        source,
        ("x",),
        budget=DiscretizationBudget(max_unique_rows=4),
        max_unique_rows=2,
    )
    assert isinstance(unavailable, ExactIndexUnavailable)
    assert unavailable.required_rows == 3
    assert unavailable.row_cap == 2
    batch_limited = exact_index_columns(
        source,
        ("x",),
        budget=DiscretizationBudget(max_batch_bytes=8),
        batch_rows=2,
    )
    assert isinstance(batch_limited, ExactIndexUnavailable)
    assert "batch" in batch_limited.reason


def test_discretize_module_imports_without_jax() -> None:
    """The true Phase-1 module must not obtain JAX through package imports."""
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
from jaxgam.data.discretize import exact_index_columns
assert callable(exact_index_columns)
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_exact_index_budget_counts_nested_keys_and_value_materialization() -> None:
    nested_limited = exact_index_columns(
        ArrayRowSource(
            {
                "x": np.array((1.0, 2.0)),
                "g": np.array(("a", "b"), dtype=object),
            }
        ),
        ("x", "g"),
        budget=DiscretizationBudget(max_table_bytes=256),
    )
    assert isinstance(nested_limited, ExactIndexUnavailable)
    assert "construction" in nested_limited.reason

    # Repeated rows keep discovery small, but converting the unique object
    # values to an owned read-only column has a separate, charged live peak.
    payload = "x" * 2048
    materialization_limited = exact_index_columns(
        ArrayRowSource({"g": np.array((payload,) * 4, dtype=object)}),
        ("g",),
        budget=DiscretizationBudget(max_table_bytes=5000),
    )
    assert isinstance(materialization_limited, ExactIndexUnavailable)
    assert "materialization" in materialization_limited.reason


def test_unique_string_storage_is_rejected_before_column_allocation(
    monkeypatch,
) -> None:
    payload = "a" * 5000
    source = ArrayRowSource({"group": np.array((payload,) * 7, dtype=object)})
    original_asarray = np.asarray
    unique_column_allocations: list[int] = []

    def traced_asarray(value, *args, **kwargs):
        result = original_asarray(value, *args, **kwargs)
        if isinstance(value, list) and len(value) == 1 and value[0] == payload:
            unique_column_allocations.append(result.nbytes)
        return result

    monkeypatch.setattr(np, "asarray", traced_asarray)
    result = exact_index_columns(
        source,
        ("group",),
        batch_rows=7,
        budget=DiscretizationBudget(max_table_bytes=12_000),
    )
    assert isinstance(result, ExactIndexUnavailable)
    assert "materialization" in result.reason
    assert unique_column_allocations == []


def test_prospective_unique_dtype_preserves_mixed_object_scalar_types() -> None:
    values = np.array(("a", 12, 3.5), dtype=object)
    table = exact_index_columns(
        ArrayRowSource({"mixed": values}),
        ("mixed",),
        budget=DiscretizationBudget(),
    )
    assert not isinstance(table, ExactIndexUnavailable)
    assert table.values[0].dtype == np.dtype(object)
    assert tuple(type(value) for value in table.values[0]) == (str, int, float)
    np.testing.assert_array_equal(table.values[0], values)


def test_memmap_selector_validation_respects_single_array_budget(
    tmp_path: Path, monkeypatch
) -> None:
    n = 101
    source = ArrayRowSource({"x": np.ones(n)})
    original_any = np.any
    simultaneous: list[int] = []

    def traced_any(value, *args, **kwargs):
        frame = inspect.currentframe()
        assert frame is not None
        ancestor = frame.f_back
        if (
            ancestor is not None
            and ancestor.f_code.co_name == "__post_init__"
            and getattr(value, "shape", None) == (n,)
        ):
            while ancestor is not None:
                covered = ancestor.f_locals.get("covered")
                if isinstance(covered, np.ndarray):
                    simultaneous.append(value.nbytes + covered.nbytes)
                    break
                ancestor = ancestor.f_back
        return original_any(value, *args, **kwargs)

    monkeypatch.setattr(np, "any", traced_any)
    table = exact_index_columns(
        source,
        ("x",),
        batch_rows=7,
        budget=DiscretizationBudget(
            max_table_bytes=4096,
            max_selector_bytes=110,
        ),
        selector_memmap=tmp_path / "selector.npy",
    )
    assert not isinstance(table, ExactIndexUnavailable)
    assert simultaneous == []
