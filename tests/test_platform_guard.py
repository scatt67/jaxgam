"""Tests for the Apple Accelerate warning in ``jaxgam.platform_guard``."""

from __future__ import annotations

import sys
import types
import warnings

import numpy
import pytest
import scipy

from jaxgam import platform_guard
from jaxgam.smooths import source_lapack


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    system: str = "darwin",
    machine: str = "arm64",
    accelerate: bool = True,
    backend: str = "cpu",
) -> None:
    monkeypatch.setattr(platform_guard, "_warned", False)
    monkeypatch.setattr(platform_guard, "sys", types.SimpleNamespace(platform=system))
    monkeypatch.setattr(
        platform_guard, "platform", types.SimpleNamespace(machine=lambda: machine)
    )
    monkeypatch.setattr(
        platform_guard, "accelerate_lapack_in_process", lambda: accelerate
    )
    monkeypatch.setattr(
        platform_guard, "jax", types.SimpleNamespace(default_backend=lambda: backend)
    )
    monkeypatch.delenv(platform_guard.ACCELERATE_OPT_OUT, raising=False)


def _issued(calls: int = 1) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(calls):
            platform_guard.warn_if_accelerate_lapack()
    return [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_warns_once_on_apple_silicon_with_accelerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)
    caught = _issued(calls=3)
    assert len(caught) == 1
    message = str(caught[0].message)
    for phrase in (
        "Apple Accelerate",
        "corrupt memory",
        "wrong with no error",
        "about 100 smooths",
        "50 smooths or fewer",
        "Linux is unaffected",
        "--python-platform aarch64-apple-darwin",
        "--platform macosx_12_0_arm64",
        f"numpy=={numpy.__version__} scipy=={scipy.__version__}",
        "JAXGAM_ALLOW_ACCELERATE=1",
    ):
        assert phrase in message, phrase
    assert "1.7" not in message


@pytest.mark.parametrize(
    ("system", "machine", "accelerate", "backend"),
    [
        ("linux", "x86_64", True, "cpu"),
        ("linux", "aarch64", True, "cpu"),
        ("darwin", "x86_64", True, "cpu"),
        ("darwin", "arm64", False, "cpu"),
        ("darwin", "arm64", True, "METAL"),
    ],
    ids=["linux-x86", "linux-arm", "intel-mac", "openblas-builds", "non-cpu"],
)
def test_no_warning_outside_the_affected_configuration(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    accelerate: bool,
    backend: str,
) -> None:
    _configure(
        monkeypatch,
        system=system,
        machine=machine,
        accelerate=accelerate,
        backend=backend,
    )
    assert _issued() == []
    assert not platform_guard.accelerate_warning_applies()


def test_opt_out_silences_the_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch)
    monkeypatch.setenv(platform_guard.ACCELERATE_OPT_OUT, "1")
    assert _issued() == []
    monkeypatch.setenv(platform_guard.ACCELERATE_OPT_OUT, "0")
    assert len(_issued()) == 1


def test_backend_is_not_queried_off_apple_silicon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, system="linux", machine="x86_64")

    def fail() -> str:
        raise AssertionError("backend initialized off Apple silicon")

    monkeypatch.setattr(
        platform_guard, "jax", types.SimpleNamespace(default_backend=fail)
    )
    assert not platform_guard.accelerate_warning_applies()


@pytest.mark.parametrize(
    ("numpy_name", "scipy_name", "expected"),
    [
        ("accelerate", "Accelerate", True),
        ("scipy-openblas", "Accelerate", True),
        ("accelerate", "scipy-openblas", True),
        ("scipy-openblas", "scipy-openblas", False),
        ("openblas", "openblas", False),
    ],
)
def test_detector_reads_both_numpy_and_scipy_builds(
    monkeypatch: pytest.MonkeyPatch, numpy_name: str, scipy_name: str, expected: bool
) -> None:
    monkeypatch.setattr(
        source_lapack.numpy_config,
        "CONFIG",
        {"Build Dependencies": {"lapack": {"name": numpy_name}}},
    )
    monkeypatch.setattr(
        source_lapack.scipy_config,
        "CONFIG",
        {"Build Dependencies": {"lapack": {"name": scipy_name}}},
    )
    assert source_lapack.accelerate_lapack_in_process() is expected
    assert source_lapack.scipy_uses_accelerate_lapack() is (
        scipy_name.lower() == "accelerate"
    )
    assert source_lapack.numpy_uses_accelerate_lapack() is (
        numpy_name.lower() == "accelerate"
    )


def test_detector_tolerates_missing_build_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_lapack.numpy_config, "CONFIG", {})
    monkeypatch.setattr(
        source_lapack.scipy_config, "CONFIG", {"Build Dependencies": {}}
    )
    assert not source_lapack.accelerate_lapack_in_process()


def test_real_detector_matches_the_installed_builds() -> None:
    """The unpatched detector agrees with the builds' own reports.

    On Linux the wheels are built on OpenBLAS, so it must be false there; on
    a Mac it is false exactly when both NumPy and SciPy are OpenBLAS builds.
    """
    names = {
        module.__name__: module.show_config(mode="dicts")["Build Dependencies"][
            "lapack"
        ]["name"].lower()
        for module in (numpy, scipy)
    }
    expected = "accelerate" in names.values()
    assert source_lapack.accelerate_lapack_in_process() is expected
    if sys.platform.startswith("linux"):
        assert not source_lapack.accelerate_lapack_in_process()
