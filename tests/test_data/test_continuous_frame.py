"""Pinned shared-group and compact setup-frame contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import jaxgam.data.continuous_frame as continuous_frame
from jaxgam.data.continuous_frame import (
    CompactFrameUnavailable,
    CompactSetupFrame,
    build_compact_setup_frame,
)
from jaxgam.data.discretize import DiscretizationBudget
from jaxgam.data.source import ArrayRowSource
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.terms import SmoothSpec
from tests.helpers import r_available
from tests.tolerances import STRICT


def _source(frame: pd.DataFrame, *, selection=None) -> ArrayRowSource:
    return ArrayRowSource(
        {name: frame[name] for name in frame}, row_selection=selection
    )


def _build(
    formula: str,
    frame: pd.DataFrame,
    *,
    factor_names=(),
    selection=None,
    m=4,
    batch_rows=3,
    budget=None,
    selector_memmap_dir=None,
    seed=8547,
) -> CompactSetupFrame:
    result = build_compact_setup_frame(
        parse_formula(formula),
        _source(frame, selection=selection),
        factor_names=factor_names,
        m=m,
        batch_rows=batch_rows,
        budget=budget,
        selector_memmap_dir=selector_memmap_dir,
        seed=seed,
    )
    assert isinstance(result, CompactSetupFrame)
    return result


def _oracle_frame() -> pd.DataFrame:
    n_rows = 12
    return pd.DataFrame(
        {
            "y": np.arange(1, n_rows + 1, dtype=float),
            "x": np.resize(np.array((0.0, 0.5, 1.0)), n_rows),
            "z": np.linspace(-1.0, 1.0, n_rows),
            "w": np.resize(np.array((1.0, 2.0)), n_rows),
            "g": pd.Series(
                pd.Categorical(
                    np.resize(np.array(("a", "b")), n_rows),
                    categories=("b", "a", "unused"),
                    ordered=True,
                )
            ),
        }
    )


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_shared_groups_frame_and_metadata_match_pinned_discrete_mf(r_bridge) -> None:
    frame = _oracle_frame()
    formula = 'y ~ s(x, bs="cr", k=3) + te(z, w, k=3) + g'
    observed = _build(formula, frame, factor_names=("g",))
    reference = r_bridge.discrete_model_frame(formula, frame, factor_names=("g",), m=4)

    assert observed.n_unique == reference.n_unique
    assert observed.index_spans == reference.index_spans
    assert tuple(observed.frame) == tuple(reference.frame)
    for name in observed.frame:
        if name == "g":
            np.testing.assert_array_equal(observed.frame[name], reference.frame[name])
        else:
            np.testing.assert_allclose(
                observed.frame[name],
                reference.frame[name],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            )
    for group_index, group in enumerate(observed.groups):
        np.testing.assert_array_equal(
            group.index.selector, reference.row_indices[:, group_index]
        )
    np.testing.assert_array_equal(
        observed.selector_for("(Intercept)"), reference.row_indices[:, -1]
    )
    assert observed.frame["g"].dtype.ordered
    assert tuple(observed.frame["g"].dtype.categories) == ("b", "a", "unused")
    frame_bytes = int(observed.frame.memory_usage(index=True, deep=True).sum())
    assert observed.retained_table_bytes >= observed.retained_group_bytes + frame_bytes
    assert observed.construction_peak_bytes >= observed.retained_table_bytes


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_scalar_numeric_padding_matches_r_sample_semantics(r_bridge) -> None:
    for constant in (3.0, 3.5):
        frame = pd.DataFrame(
            {
                "y": np.arange(1.0, 21.0),
                "x": np.full(20, constant),
                "z": np.linspace(-1.0, 1.0, 20),
            }
        )
        formula = "y ~ s(x) + z"
        observed = _build(formula, frame, m=6, batch_rows=3)
        reference = r_bridge.discrete_model_frame(formula, frame, m=6)
        np.testing.assert_allclose(
            observed.frame,
            reference.frame,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        for group_index, group in enumerate(observed.groups):
            np.testing.assert_array_equal(
                group.index.selector, reference.row_indices[:, group_index]
            )


def test_identical_joint_requests_share_index_selector_and_source_work(
    monkeypatch,
) -> None:
    frame = _oracle_frame()

    class CountingSource:
        def __init__(self, delegate):
            self.delegate = delegate
            self.scan_calls = 0

        @property
        def n_rows(self):
            return self.delegate.n_rows

        # It stands for the delegate, so it forwards what the delegate supplies.
        @property
        def has_explicit_weights(self):
            return self.delegate.has_explicit_weights

        @property
        def has_explicit_offset(self):
            return self.delegate.has_explicit_offset

        def fingerprint(self):
            return self.delegate.fingerprint()

        def scan(self, batch_rows):
            self.scan_calls += 1
            yield from self.delegate.scan(batch_rows)

    repeated_source = CountingSource(_source(frame))
    calls = []
    original = continuous_frame.range_grid_index_columns

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(continuous_frame, "range_grid_index_columns", counted)
    repeated = build_compact_setup_frame(
        parse_formula("y ~ s(x, z) + s(z, x) + x"),
        repeated_source,
        m=100,
        batch_rows=3,
    )
    assert isinstance(repeated, CompactSetupFrame)
    assert calls == [("x", "z"), ("y",)]
    assert repeated_source.scan_calls > len(calls)
    assert len(repeated.groups) == 2  # joint smooth group plus response
    assert repeated.selector_for("x") is repeated.selector_for("z")
    ordinals = np.array((0, 3, 11), dtype=np.intp)
    gathered = repeated.batch_selectors(ordinals)
    assert gathered.for_name("x") is gathered.for_name("z")
    np.testing.assert_array_equal(
        gathered.for_name("x"), repeated.selector_for("x")[ordinals]
    )
    assert not gathered.for_name("x").flags.writeable


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_by_group_order_padding_and_m_vector_match_pinned_r(r_bridge) -> None:
    frame = _oracle_frame()
    formula = "y ~ te(z, w, by=g) + s(x, by=g) + g"
    result = _build(
        formula,
        frame,
        factor_names=("g",),
        m=(4, 3),
    )
    assert [(group.names, group.role) for group in result.groups] == [
        (("g",), "by"),
        (("z",), "tensor_margin"),
        (("w",), "tensor_margin"),
        (("x",), "smooth"),
        (("y",), "parametric"),
    ]
    assert result.selector_for("g") is result.groups[0].index.selector
    assert [group.m for group in result.groups] == [4, 4, 4, 3, 4]
    reference = r_bridge.discrete_model_frame(
        formula, frame, factor_names=("g",), m=(4, 3)
    )
    assert result.n_unique == reference.n_unique
    assert result.index_spans == reference.index_spans
    for group_index, group in enumerate(result.groups):
        np.testing.assert_array_equal(
            group.index.selector, reference.row_indices[:, group_index]
        )
    for name in result.frame:
        if name == "g":
            np.testing.assert_array_equal(result.frame[name], reference.frame[name])
        else:
            np.testing.assert_allclose(
                result.frame[name],
                reference.frame[name],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            )


def test_partial_nesting_rejects_before_source_scan() -> None:
    frame = _oracle_frame()

    class NoScanSource:
        n_rows = len(frame)

        def fingerprint(self):
            return "stable"

        def scan(self, _batch_rows):
            raise AssertionError("incompatible nesting must fail before source scan")

    with pytest.raises(ValueError, match="incompatible nesting"):
        build_compact_setup_frame(
            parse_formula("y ~ s(x, z) + s(x)"), NoScanSource(), m=4
        )


def test_linked_smoothing_rejects_before_source_scan() -> None:
    class NoScanSource:
        n_rows = 12

        def fingerprint(self):
            return "stable"

        def scan(self, _batch_rows):
            raise AssertionError("id= must fail before source scan")

    # The shared formula/spec guard refuses id= (mgcv links smoothing
    # parameters through it, which JaxGAM does not implement), so neither the
    # compact frame nor any source scan is reached.
    with pytest.raises(NotImplementedError, match=r"Linked smoothing parameters"):
        build_compact_setup_frame(
            parse_formula("y ~ s(x, id=1) + te(z, w, id=1) + s(g, bs='re')"),
            NoScanSource(),
            m=4,
        )
    with pytest.raises(NotImplementedError, match=r"Linked smoothing parameters"):
        SmoothSpec(variables=["x"], extra_args={"id": 1})


@pytest.mark.skipif(not r_available(), reason="R with pinned mgcv not available")
def test_selected_reordered_repeated_rows_match_r_across_batches(r_bridge) -> None:
    frame = _oracle_frame()
    selection = np.array((7, 2, 10, 1, 7, 11, 2))
    selected = frame.iloc[selection].reset_index(drop=True)
    formula = "y ~ s(x) + te(z, w) + g"
    observed = _build(
        formula,
        frame,
        factor_names=("g",),
        selection=selection,
        m=3,
        batch_rows=2,
    )
    reference = r_bridge.discrete_model_frame(
        formula, selected, factor_names=("g",), m=3
    )
    assert observed.n_rows == len(selection)
    assert observed.n_unique == reference.n_unique
    assert observed.index_spans == reference.index_spans
    for group_index, group in enumerate(observed.groups):
        np.testing.assert_array_equal(
            group.index.selector, reference.row_indices[:, group_index]
        )
        for name, values in zip(group.names, group.index.values, strict=True):
            reconstructed = values[group.index.selector]
            if name == "g":
                np.testing.assert_array_equal(
                    values,
                    reference.frame[name].iloc[: group.index.n_unique],
                )
                np.testing.assert_array_equal(reconstructed, selected[name])
            else:
                reference_values = reference.frame[name].iloc[: group.index.n_unique]
                np.testing.assert_allclose(
                    values,
                    reference_values,
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                )
                if not group.index.rounded:
                    np.testing.assert_array_equal(reconstructed, selected[name])
    for start in range(0, observed.n_rows, 2):
        stop = min(start + 2, observed.n_rows)
        ordinals = np.arange(start, stop, dtype=np.intp)
        gathered = observed.batch_selectors(ordinals)
        for name, group_index in observed.name_group_indices:
            np.testing.assert_array_equal(
                gathered.for_name(name),
                observed.groups[group_index].index.selector[start:stop],
            )


def test_memmapped_groups_are_shared_and_cleanup_removes_paths(tmp_path) -> None:
    frame = _oracle_frame()
    result = _build(
        "y ~ s(x, z) + g",
        frame,
        factor_names=("g",),
        m=100,
        selector_memmap_dir=tmp_path,
    )
    assert len(result.selector_paths) == len(result.groups)
    assert all(path.exists() for path in result.selector_paths)
    assert all(group.index.index.storage == "memmap" for group in result.groups)
    assert result.selector_for("x") is result.selector_for("z")
    result.cleanup()
    assert not any(path.exists() for path in result.selector_paths)


def test_group_shuffling_and_padding_leave_numpy_rng_untouched() -> None:
    before = np.random.get_state()  # noqa: NPY002 - verify no global RNG mutation
    _build(
        "y ~ s(x) + te(z, w) + g",
        _oracle_frame(),
        factor_names=("g",),
        m=3,
    )
    after = np.random.get_state()  # noqa: NPY002 - verify no global RNG mutation
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_failed_later_group_cleans_earlier_memmap(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "y": np.arange(20.0),
            "x": np.resize(np.array((0.0, 0.5, 1.0)), 20),
            "z": np.resize(np.array((0.0, 1.0)), 20),
        }
    )
    result = build_compact_setup_frame(
        parse_formula("y ~ s(x) + z"),
        _source(frame),
        m=4,
        batch_rows=4,
        selector_memmap_dir=tmp_path,
        budget=DiscretizationBudget(
            max_table_bytes=3000,
            max_selector_bytes=100_000,
            max_batch_bytes=100_000,
        ),
    )
    assert isinstance(result, CompactFrameUnavailable)
    assert result.group_names == ("y",)
    assert result.budget_scope == "table"
    assert list(tmp_path.iterdir()) == []


def test_combined_selector_budget_includes_retained_groups() -> None:
    frame = _oracle_frame()
    result = build_compact_setup_frame(
        parse_formula("y ~ s(x) + g"),
        _source(frame),
        factor_names=("g",),
        m=4,
        batch_rows=3,
        budget=DiscretizationBudget(
            max_table_bytes=100_000,
            max_selector_bytes=100,
            max_batch_bytes=100_000,
        ),
    )
    assert isinstance(result, CompactFrameUnavailable)
    assert result.group_names == ("y",)
    assert result.budget_scope == "selector"
    assert result.required_bytes == 108
    assert result.budget_bytes == 100


def test_row_cap_metadata_is_preserved() -> None:
    frame = _oracle_frame()
    result = build_compact_setup_frame(
        parse_formula("y ~ s(x)"),
        _source(frame),
        m=100,
        batch_rows=3,
        budget=DiscretizationBudget(max_unique_rows=1),
    )
    assert isinstance(result, CompactFrameUnavailable)
    assert result.group_names == ("x",)
    assert result.required_rows == 2
    assert result.row_cap == 1


def test_batch_gather_rejects_before_exceeding_recorded_workspace() -> None:
    result = _build(
        "y ~ s(x) + s(z) + g",
        _oracle_frame(),
        factor_names=("g",),
        m=100,
        batch_rows=3,
        budget=DiscretizationBudget(max_batch_bytes=900),
    )
    ordinals = np.arange(result.n_rows, dtype=np.intp)
    with pytest.raises(MemoryError, match="required 1068, allowed 900"):
        result.batch_selectors(ordinals)


@pytest.mark.parametrize("seed", [True, 2**31, -(2**31) - 1])
def test_seed_requires_signed_32_bit_integer(seed) -> None:
    with pytest.raises(ValueError, match="signed 32-bit"):
        _build("y ~ s(x)", _oracle_frame(), seed=seed)


def test_existing_memmap_is_rejected_without_removal(tmp_path) -> None:
    existing = tmp_path / "compact-selector-0.dat"
    existing.write_bytes(b"caller-owned")
    with pytest.raises(FileExistsError, match="already exists"):
        _build(
            "y ~ s(x)",
            _oracle_frame(),
            selector_memmap_dir=tmp_path,
        )
    assert existing.read_bytes() == b"caller-owned"


def test_compact_frame_module_imports_without_jax() -> None:
    root = Path(__file__).resolve().parents[2] / "jaxgam"
    script = f"""
import sys
import types
from pathlib import Path
root = Path({str(root)!r})
for name, path in (
    ("jaxgam", root),
    ("jaxgam.data", root / "data"),
    ("jaxgam.formula", root / "formula"),
):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
from jaxgam.data.continuous_frame import build_compact_setup_frame
assert callable(build_compact_setup_frame)
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
