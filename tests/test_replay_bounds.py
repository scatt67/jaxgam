"""Per-platform reviewed replay bounds and the CPU-flag platform class."""

from __future__ import annotations

import platform
import sys

import pytest

from tests import helpers
from tests.helpers import check_replay_bound, cpu_platform_class


@pytest.mark.parametrize(
    ("machine", "system", "flags", "expected"),
    [
        ("arm64", "darwin", None, "macos-arm64"),
        ("aarch64", "linux", None, "linux-aarch64"),
        ("x86_64", "linux", frozenset({"sse2", "avx2", "avx512f"}), "x86_64-avx512"),
        ("x86_64", "linux", frozenset({"sse2", "avx2"}), "x86_64"),
        ("AMD64", "darwin", frozenset({"avx512f"}), "x86_64-avx512"),
        ("x86_64", "linux", None, "x86_64-unreadable-flags"),
        ("riscv64", "linux", None, "linux-riscv64"),
    ],
)
def test_platform_class_comes_from_cpu_flags(
    monkeypatch: pytest.MonkeyPatch, machine, system, flags, expected
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(helpers, "_x86_cpu_flags", lambda: flags)
    assert cpu_platform_class() == expected


def test_x86_flag_reader_parses_proc_cpuinfo(monkeypatch, tmp_path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nflags\t\t: fpu sse2 avx2 avx512f\n\n")
    real_open = open
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *args, **kwargs: real_open(
            cpuinfo if path == "/proc/cpuinfo" else path, *args, **kwargs
        ),
    )
    assert helpers._x86_cpu_flags() == frozenset({"fpu", "sse2", "avx2", "avx512f"})


def test_replay_bound_uses_the_running_platform_entry(monkeypatch) -> None:
    monkeypatch.setattr(helpers, "cpu_platform_class", lambda: "x86_64-avx512")
    monkeypatch.setattr(helpers, "REPLAY_COUNT_RECORDS", [])
    table = {"case": {"macos-arm64": (2, 2), "x86_64-avx512": (7, 7)}}
    check_replay_bound("T", table, "case", 7)
    check_replay_bound("T", table, "unlisted", 0)
    with pytest.raises(AssertionError, match=r"8 replayed decisions .* bound 7"):
        check_replay_bound("T", table, "case", 8)
    with pytest.raises(AssertionError, match=r"1 replayed decisions .* bound 0"):
        check_replay_bound("T", table, "unlisted", 1)
    assert [record[3:] for record in helpers.REPLAY_COUNT_RECORDS] == [
        (7, 7),
        (0, 0),
        (8, 7),
        (1, 0),
    ]


def test_listed_case_without_a_platform_entry_fails_with_its_count(
    monkeypatch,
) -> None:
    monkeypatch.setattr(helpers, "cpu_platform_class", lambda: "x86_64")
    monkeypatch.setattr(helpers, "REPLAY_COUNT_RECORDS", [])
    table = {"case": {"macos-arm64": (2, 2), "linux-aarch64": (2, 1)}}
    with pytest.raises(
        AssertionError, match=r"no reviewed replay bound .* x86_64; measured 0"
    ):
        check_replay_bound("T", table, "case", 0)
    assert helpers.REPLAY_COUNT_RECORDS == [("T", "'case'", "x86_64", 0, None)]


def test_other_classes_marker_bounds_unlisted_classes_at_zero(monkeypatch) -> None:
    table = {"case": {"x86_64": (1, 1), helpers.OTHER_CLASSES_NEED_NONE: (0, 0)}}
    monkeypatch.setattr(helpers, "REPLAY_COUNT_RECORDS", [])
    monkeypatch.setattr(helpers, "cpu_platform_class", lambda: "x86_64")
    check_replay_bound("T", table, "case", 1)
    monkeypatch.setattr(helpers, "cpu_platform_class", lambda: "linux-aarch64")
    check_replay_bound("T", table, "case", 0)
    with pytest.raises(AssertionError, match=r"1 replayed decisions .* bound 0"):
        check_replay_bound("T", table, "case", 1)
    with pytest.raises(ValueError, match="must be 0"):
        check_replay_bound(
            "T", {"case": {helpers.OTHER_CLASSES_NEED_NONE: (2, 2)}}, "case", 0
        )
