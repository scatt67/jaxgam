"""Deterministic split of the test suite into shards for parallel CI runners.

``JAXGAM_TEST_SHARD=i/N`` keeps the tests of shard ``i`` (1-based) of ``N``.
A test's shard depends only on its node id, so the ``N`` shards of one commit
partition the suite: every test runs in exactly one of them, whatever order
or machine collects it. Unset, the whole suite runs.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

SHARD_ENV = "JAXGAM_TEST_SHARD"


@dataclass(frozen=True)
class Shard:
    """Shard ``index`` (1-based) of ``count``."""

    index: int
    count: int

    def owns(self, nodeid: str) -> bool:
        """Whether the test with this node id runs in this shard."""
        return shard_of(nodeid, self.count) == self.index


def shard_of(nodeid: str, count: int) -> int:
    """The 1-based shard, of ``count``, that runs the test ``nodeid``.

    CRC-32 of the node id: stable across processes, platforms and Python
    versions, unlike the salted built-in ``hash``.
    """
    if count < 1:
        raise ValueError("shard count must be at least 1")
    return zlib.crc32(nodeid.encode("utf-8")) % count + 1


def parse_shard(spec: str | None) -> Shard | None:
    """Parse ``"i/N"``; ``None`` or an empty string selects the whole suite."""
    if spec is None or not spec.strip():
        return None
    index_text, separator, count_text = spec.strip().partition("/")
    if not separator or not index_text.isdigit() or not count_text.isdigit():
        raise ValueError(f"{SHARD_ENV} must look like '3/8', got {spec!r}")
    index, count = int(index_text), int(count_text)
    if count < 1 or not 1 <= index <= count:
        raise ValueError(f"{SHARD_ENV} needs 1 <= index <= count, got {spec!r}")
    return Shard(index, count)
