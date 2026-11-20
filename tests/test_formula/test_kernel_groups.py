"""Host shape groups and the XWXd pair schedule."""

from __future__ import annotations

import numpy as np
import pytest

from jaxgam.formula.kernel_groups import (
    LookupKey,
    TensorKey,
    discrete_pair_schedule,
    pair_reduction_plan,
    shape_groups,
)


def test_shape_groups_follow_first_appearance_with_ordered_members() -> None:
    groups = shape_groups([(4, 1), (6, 1), (4, 1), (4, 2), (6, 1)])
    assert [key for key, _ in groups] == [(4, 1), (6, 1), (4, 2)]
    assert [members.tolist() for _, members in groups] == [[0, 2], [1, 4], [3]]
    assert shape_groups([]) == ()


def _schedule(keys, **kwargs):
    arguments = {
        "pair_policy": "histogram_or_direct",
        "max_pair_histogram_bytes": 32 * 1024 * 1024,
        "n_rows": 10_000,
    }
    arguments.update(kwargs)
    return discrete_pair_schedule(tuple(keys), **arguments)


def _pairs(schedule):
    return {
        (group.kind, int(left), int(right))
        for group in schedule
        for left, right in zip(group.left, group.right, strict=True)
    }


def test_pair_schedule_reproduces_each_pair_reduction() -> None:
    small = LookupKey(50, 4, "int32")
    wide = LookupKey(60, 5, "int32")
    tensor = TensorKey(2)
    keys = (small, wide, tensor, small)
    schedule = _schedule(keys)
    expected = {
        ("lookup_diagonal", 0, 0),
        ("lookup_histogram", 0, 1),
        ("tensor", 0, 2),
        ("lookup_histogram", 0, 3),
        ("lookup_diagonal", 1, 1),
        ("tensor", 1, 2),
        ("lookup_histogram", 1, 3),
        ("tensor", 2, 2),
        ("tensor", 2, 3),
        ("lookup_diagonal", 3, 3),
    }
    assert _pairs(schedule) == expected
    # Every upper-triangle pair once, in row-major order within each group.
    for group in schedule:
        order = list(zip(group.left.tolist(), group.right.tolist(), strict=True))
        assert order == sorted(order)
        assert all(left <= right for left, right in order)
    # Equal shapes share a group: both small-small pairs, both diagonals.
    diagonal = [group for group in schedule if group.kind == "lookup_diagonal"]
    assert [group.left.tolist() for group in diagonal] == [[0, 3], [1]]
    with pytest.raises(ValueError, match="read-only"):
        schedule[0].left[0] = 5


def test_pair_schedule_follows_policy_and_histogram_budget() -> None:
    keys = (LookupKey(50, 4, "int32"), LookupKey(60, 5, "int32"))
    direct_only = _schedule(keys, pair_policy="direct_only")
    assert ("lookup_direct", 0, 1) in _pairs(direct_only)
    tight = _schedule(keys, max_pair_histogram_bytes=50 * 60 * 8 - 1)
    assert ("lookup_direct", 0, 1) in _pairs(tight)
    # Few rows: direct rows are cheaper than the pair histogram.
    few_rows = _schedule(keys, n_rows=10)
    assert ("lookup_direct", 0, 1) in _pairs(few_rows)
    plan = pair_reduction_plan(50, 60, 1 << 20, n_rows=10, k_left=4, k_right=5)
    assert plan.method == "direct"
    # The selector type is part of a lookup's shape.
    mixed = _schedule((LookupKey(50, 4, "int32"), LookupKey(50, 4, "int16")))
    assert len([group for group in mixed if group.kind == "lookup_diagonal"]) == 2


def test_pair_schedule_handles_one_block_and_is_cached() -> None:
    keys = (LookupKey(5, 1, "int32"),)
    first = _schedule(keys)
    assert _pairs(first) == {("lookup_diagonal", 0, 0)}
    assert _schedule(keys) is first
    assert np.asarray(first[0].left).dtype == np.int64
