"""Tests for source replay, validation, and categorical preservation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import (
    ArrayRowSource,
    DataFrameRowSource,
    MemmapRowSource,
    RowBatch,
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
