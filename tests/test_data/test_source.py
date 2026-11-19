"""Tests for source replay, validation, and categorical preservation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import jaxgam.data.source as source_module
from jaxgam.data.source import (
    ArrayRowSource,
    DataFrameRowSource,
    LazyColumns,
    MemmapRowSource,
    RowBatch,
    overlay_columns,
    single_fit_fingerprints,
    stage_fingerprint,
)


def test_array_source_replays_positional_batches() -> None:
    source = ArrayRowSource(
        {"x": np.arange(7.0)},
        y=np.arange(7.0) + 1,
        weights=np.arange(1.0, 8.0),
        offset=np.arange(7.0) / 10,
        row_selection=np.array([6, 2, 4, 0]),
        version="fixture-v1",
    )
    first = list(source.scan(3))
    second = list(source.scan(3))

    assert [batch.row_positions.tolist() for batch in first] == [[6, 2, 4], [0]]
    assert [batch.row_positions.tolist() for batch in second] == [[6, 2, 4], [0]]
    np.testing.assert_array_equal(first[0].columns["x"], [6.0, 2.0, 4.0])
    np.testing.assert_array_equal(first[0].y, [7.0, 3.0, 5.0])
    np.testing.assert_array_equal(first[0].weight, [7.0, 3.0, 5.0])
    np.testing.assert_array_equal(first[0].offset, [0.6, 0.2, 0.4])
    assert first[0].valid.all()
    assert source.fingerprint() == source.fingerprint()


def test_dataframe_source_preserves_ordered_integer_categorical() -> None:
    category = pd.Categorical([2, 1, 2], categories=[1, 2, 3], ordered=True)
    frame = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0], "group": category, "x": [0.0, 1.0, 2.0]}, index=[9, 5, 1]
    )
    batch = next(DataFrameRowSource(frame, response="y").scan(2))

    group = batch.columns["group"]
    assert isinstance(group, pd.Series)
    assert pd.api.types.is_categorical_dtype(group)
    assert group.cat.ordered
    assert group.cat.categories.tolist() == [1, 2, 3]
    assert batch.row_positions.tolist() == [0, 1]
    assert group.index.tolist() == [0, 1]


def test_source_validation_and_batch_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        ArrayRowSource({"x": np.ones(2), "z": np.ones(3)})
    with pytest.raises(ValueError, match="non-negative"):
        ArrayRowSource({"x": np.ones(2)}, weights=[1.0, -1.0])
    with pytest.raises(ValueError, match="positive"):
        ArrayRowSource({"x": np.ones(2)}, weights=[0.0, 0.0])
    with pytest.raises(ValueError, match="positive"):
        list(ArrayRowSource({"x": np.ones(2)}).scan(0))
    with pytest.raises(ValueError, match="integer"):
        ArrayRowSource({"x": np.ones(2)}, row_selection=[0.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        ArrayRowSource({"x": np.ones(2)}, weights=[1.0, 0.0], row_selection=[1])
    with pytest.raises(ValueError, match="Column 'x'"):
        RowBatch(
            {"x": np.ones(1)},
            None,
            np.ones(2),
            np.zeros(2),
            np.ones(2, bool),
            np.arange(2),
        )


def test_fingerprint_covers_content_schema_selection_and_version() -> None:
    base = ArrayRowSource({"x": np.array([1.0, 2.0])}, y=[1.0, 2.0], version="a")
    changed_value = ArrayRowSource(
        {"x": np.array([1.0, 3.0])}, y=[1.0, 2.0], version="a"
    )
    changed_selection = ArrayRowSource(
        {"x": np.array([1.0, 2.0])}, y=[1.0, 2.0], row_selection=[1, 0], version="a"
    )
    changed_version = ArrayRowSource(
        {"x": np.array([1.0, 2.0])}, y=[1.0, 2.0], version="b"
    )
    assert (
        len(
            {
                base.fingerprint(),
                changed_value.fingerprint(),
                changed_selection.fingerprint(),
                changed_version.fingerprint(),
            }
        )
        == 4
    )


def test_memmap_source_replays_npy_columns(tmp_path) -> None:
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.arange(5.0))
    np.save(y_path, np.arange(5.0) + 10)
    source = MemmapRowSource.from_npy_files({"x": x_path}, y=y_path)
    assert isinstance(source._columns["x"], np.memmap)
    batches = list(source.scan(4))
    np.testing.assert_array_equal(batches[1].columns["x"], [4.0])
    np.testing.assert_array_equal(batches[1].y, [14.0])


def test_default_vectors_and_positions_are_implicit_and_mutation_is_detected() -> None:
    values = np.arange(8.0)
    source = ArrayRowSource({"x": values}, y=values)
    assert source._weights is None
    assert source._offset is None
    assert source._positions is None
    before = source.fingerprint()
    values[0] = -1.0
    assert source.fingerprint() != before
    batch = next(source.scan(3))
    np.testing.assert_array_equal(batch.row_positions, [0, 1, 2])
    np.testing.assert_array_equal(batch.weight, np.ones(3))
    np.testing.assert_array_equal(batch.offset, np.zeros(3))


def test_y_only_dataframe_source_supports_intercept_models() -> None:
    source = DataFrameRowSource(pd.DataFrame({"y": [2.0, 3.0]}), response="y")
    batch = next(source.scan(4))
    assert dict(batch.columns) == {}
    np.testing.assert_array_equal(batch.y, [2.0, 3.0])


def test_array_sources_declare_supplied_weights_and_offset(tmp_path) -> None:
    columns = {"x": np.arange(4.0)}
    plain = ArrayRowSource(columns, y=np.ones(4))
    assert (plain.has_explicit_weights, plain.has_explicit_offset) == (False, False)
    # Explicit default values are still supplied by the caller.
    supplied = ArrayRowSource(
        columns, y=np.ones(4), weights=np.ones(4), offset=np.zeros(4)
    )
    assert (supplied.has_explicit_weights, supplied.has_explicit_offset) == (
        True,
        True,
    )
    selected = ArrayRowSource(
        columns, y=np.ones(4), weights=np.ones(4), row_selection=[0, 2]
    )
    assert (selected.has_explicit_weights, selected.has_explicit_offset) == (
        True,
        False,
    )
    frame = DataFrameRowSource(
        pd.DataFrame({"x": np.arange(4.0), "y": np.ones(4)}),
        response="y",
        offset=np.zeros(4),
    )
    assert (frame.has_explicit_weights, frame.has_explicit_offset) == (False, True)
    np.save(tmp_path / "x.npy", np.arange(4.0))
    np.save(tmp_path / "w.npy", np.ones(4))
    memmap = MemmapRowSource.from_npy_files(
        {"x": tmp_path / "x.npy"}, weights=tmp_path / "w.npy"
    )
    assert (memmap.has_explicit_weights, memmap.has_explicit_offset) == (True, False)


def _count_calls(monkeypatch: pytest.MonkeyPatch, owner: object, name: str) -> list:
    """Record each call of ``owner.name`` while leaving it working."""
    calls: list = []
    original = getattr(owner, name)

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counted)
    return calls


def test_array_source_slices_a_column_only_when_it_is_read(monkeypatch) -> None:
    source = ArrayRowSource(
        {"a": np.arange(5.0), "b": np.arange(5.0) * 2, "c": np.arange(5.0) * 3},
        y=np.arange(5.0),
    )
    slices = _count_calls(monkeypatch, source_module, "_slice")

    batches = list(source.scan(2))

    assert slices == []
    columns = batches[0].columns
    assert isinstance(columns, LazyColumns)
    assert list(columns) == ["a", "b", "c"]
    assert len(columns) == 3
    assert "b" in columns
    assert "z" not in columns
    np.testing.assert_array_equal(columns["b"], [0.0, 2.0])
    assert columns["b"] is columns["b"]
    assert len(slices) == 1
    with pytest.raises(KeyError):
        columns["z"]
    with pytest.raises(TypeError):
        columns["a"] = np.zeros(2)  # type: ignore[index]
    # Reading every column still yields what an eager scan did.
    assert {name: value.tolist() for name, value in batches[2].columns.items()} == {
        "a": [4.0],
        "b": [8.0],
        "c": [12.0],
    }


def test_lazy_columns_are_checked_against_the_batch_length_when_read() -> None:
    columns = LazyColumns(
        ["good", "bad"], {"good": np.zeros(2), "bad": np.zeros(3)}.get
    )
    batch = RowBatch(
        columns,
        None,
        np.ones(2),
        np.zeros(2),
        np.ones(2, bool),
        np.arange(2, dtype=np.intp),
    )

    assert len(batch.columns["good"]) == 2
    with pytest.raises(ValueError, match="Column 'bad' has 3 rows but batch has 2"):
        batch.columns["bad"]


def test_overlay_columns_adds_and_replaces_without_reading_the_rest() -> None:
    read: list[str] = []

    def load(name: str) -> np.ndarray:
        read.append(name)
        return np.full(2, float(len(name)))

    base = LazyColumns(["x", "yy"], load)
    produced: list[str] = []

    def replacement() -> np.ndarray:
        produced.append("x")
        return base["x"] + 10.0

    columns = overlay_columns(
        base, {"x": replacement, "extra": lambda: np.array([7.0, 8.0])}
    )

    assert list(columns) == ["x", "yy", "extra"]
    assert read == []
    assert produced == []
    np.testing.assert_array_equal(columns["x"], [11.0, 11.0])
    np.testing.assert_array_equal(columns["extra"], [7.0, 8.0])
    assert read == ["x"]
    assert produced == ["x"]


def test_single_fit_fingerprints_hash_once_then_verify_on_exit(monkeypatch) -> None:
    values = np.arange(6.0)
    source = ArrayRowSource({"x": values}, y=np.arange(6.0))
    hashes = _count_calls(monkeypatch, ArrayRowSource, "_content_fingerprint")

    outside = source.fingerprint()
    assert source.fingerprint() == outside
    assert len(hashes) == 2

    with single_fit_fingerprints():
        inside = [source.fingerprint() for _ in range(5)]
        with single_fit_fingerprints():
            inside.append(source.fingerprint())
        assert len(hashes) == 3
    assert set(inside) == {outside}
    assert len(hashes) == 4

    def mutate_inside_a_fit() -> None:
        with single_fit_fingerprints():
            before = source.fingerprint()
            values[0] = 99.0
            assert source.fingerprint() == before

    with pytest.raises(RuntimeError, match="RowSource changed during the fit"):
        mutate_inside_a_fit()
    assert source.fingerprint() != outside


def test_single_fit_fingerprints_do_not_mask_an_error_from_the_fit() -> None:
    values = np.arange(4.0)
    source = ArrayRowSource({"x": values})

    def fail_inside_a_fit() -> None:
        with single_fit_fingerprints():
            source.fingerprint()
            values[0] = 5.0
            raise KeyError("fit failed")

    with pytest.raises(KeyError, match="fit failed"):
        fail_inside_a_fit()


def test_stage_fingerprint_recomputes_unless_an_outer_scope_will_verify(
    monkeypatch,
) -> None:
    values = np.arange(4.0)
    source = ArrayRowSource({"x": values})
    hashes = _count_calls(monkeypatch, ArrayRowSource, "_content_fingerprint")

    with single_fit_fingerprints(verify_on_exit=False):
        start = source.fingerprint()
        assert source.fingerprint() == start
        assert len(hashes) == 1
        values[0] = 7.0
        assert source.fingerprint() == start
        assert stage_fingerprint(source) != start
        assert len(hashes) == 2
    # A stage scope leaves the final check to the stage.
    assert len(hashes) == 2

    def stage_inside_a_fit() -> None:
        with single_fit_fingerprints():
            begin = source.fingerprint()
            with single_fit_fingerprints(verify_on_exit=False):
                values[0] = 8.0
                assert stage_fingerprint(source) == begin

    with pytest.raises(RuntimeError, match="RowSource changed during the fit"):
        stage_inside_a_fit()
    assert stage_fingerprint(source) == source.fingerprint()
