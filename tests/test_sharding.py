"""The CI shards partition the suite, and a bad shard request is refused."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.conftest import pytest_collection_modifyitems
from tests.sharding import SHARD_ENV, Shard, parse_shard, shard_of

_NODEIDS = [
    f"tests/test_{module}.py::test_case[{family}-{index}]"
    for module in ("api", "fitting/test_pirls", "execution/test_stream")
    for family in ("gaussian", "poisson", "nb")
    for index in range(40)
]


@pytest.mark.parametrize("count", [1, 2, 8, 13])
def test_every_test_belongs_to_exactly_one_shard(count: int) -> None:
    shards = [Shard(index, count) for index in range(1, count + 1)]
    for nodeid in _NODEIDS:
        owners = [shard.index for shard in shards if shard.owns(nodeid)]
        assert owners == [shard_of(nodeid, count)]


def test_assignment_is_a_fixed_function_of_the_node_id() -> None:
    # CRC-32 values written out, so a change of hash (for example to the
    # per-process salted built-in) fails here rather than silently running a
    # test in two CI shards or in none.
    assert shard_of("tests/test_api.py::test_fit", 8) == 0x4949B095 % 8 + 1 == 6
    assert (
        shard_of("tests/test_api.py::test_fit[poisson]", 8) == 0x2DBCF0C3 % 8 + 1 == 4
    )


def test_shards_are_of_comparable_size() -> None:
    sizes = Counter(shard_of(nodeid, 8) for nodeid in _NODEIDS)
    assert set(sizes) == set(range(1, 9))
    assert max(sizes.values()) < 2 * min(sizes.values())


@pytest.mark.parametrize("spec", [None, "", "  "])
def test_no_request_selects_the_whole_suite(spec: str | None) -> None:
    assert parse_shard(spec) is None


def test_request_is_parsed() -> None:
    assert parse_shard("3/8") == Shard(3, 8)
    assert parse_shard(" 1/1 ") == Shard(1, 1)


@pytest.mark.parametrize(
    "spec", ["3", "0/8", "9/8", "3/0", "a/8", "3/b", "-1/8", "3/8/2"]
)
def test_malformed_request_is_refused(spec: str) -> None:
    with pytest.raises(ValueError, match=SHARD_ENV):
        parse_shard(spec)


def test_shard_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        shard_of("tests/test_api.py::test_fit", 0)


class _Item:
    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid


class _Config:
    """The part of pytest's config the collection hook uses."""

    def __init__(self) -> None:
        self.deselected: list[_Item] = []
        self.hook = self

    def pytest_deselected(self, items: list[_Item]) -> None:
        self.deselected.extend(items)


def _collected(monkeypatch: pytest.MonkeyPatch, spec: str | None):
    """(kept, deselected) node ids after the collection hook under ``spec``."""
    if spec is None:
        monkeypatch.delenv(SHARD_ENV, raising=False)
    else:
        monkeypatch.setenv(SHARD_ENV, spec)
    config, items = _Config(), [_Item(nodeid) for nodeid in _NODEIDS]
    pytest_collection_modifyitems(config, items)
    return [i.nodeid for i in items], [i.nodeid for i in config.deselected]


def test_collection_hook_keeps_the_whole_suite_without_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept, deselected = _collected(monkeypatch, None)
    assert kept == _NODEIDS
    assert deselected == []


def test_collection_hook_runs_each_test_in_exactly_one_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards = [_collected(monkeypatch, f"{index}/3") for index in (1, 2, 3)]
    for kept, deselected in shards:
        # Collection order is kept, and what is dropped is reported to pytest.
        assert kept == [n for n in _NODEIDS if n in set(kept)]
        assert sorted(kept + deselected) == sorted(_NODEIDS)
    assert sorted(n for kept, _ in shards for n in kept) == sorted(_NODEIDS)


def test_collection_hook_refuses_a_malformed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SHARD_ENV, "9/8")
    with pytest.raises(ValueError, match=SHARD_ENV):
        pytest_collection_modifyitems(_Config(), [_Item(_NODEIDS[0])])
