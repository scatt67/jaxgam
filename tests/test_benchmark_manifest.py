"""The per-run benchmark environment manifest (scripts/benchmark_manifest.py)."""

import json
import subprocess
from pathlib import Path

import pytest

from scripts import benchmark_manifest

_CPUINFO = """processor\t: 0
model name\t: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
flags\t\t: fpu sse2 avx2 avx512f
"""
_ARM_CPUINFO = """processor\t: 0
Features\t: fp asimd crc32
CPU part\t: 0xd0c
"""
_MEMINFO = "MemTotal:       16384000 kB\nMemFree: 1 kB\n"


def _fake_darwin(command: list[str]) -> str:
    outputs = {
        ("sysctl", "hw.optional"): (
            "hw.optional.arm.FEAT_CRC32: 1\nhw.optional.arm.FEAT_SME: 0\n"
            "hw.optional.floatingpoint: 1"
        ),
        ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M4",
        ("sysctl", "-n", "hw.ncpu"): "10",
        ("sysctl", "-n", "hw.memsize"): "17179869184",
        ("pmset", "-g", "assertions"): (
            "Assertion status system-wide:\n   PreventSystemSleep             1\n"
            "   PreventUserIdleSystemSleep     0\n"
            "   pid 1(caffeinate): PreventSystemSleep named: x"
        ),
    }
    return outputs[tuple(command)]


def test_darwin_cpu_records_model_flags_cores_and_ram() -> None:
    cpu = benchmark_manifest.cpu_description("Darwin", run=_fake_darwin)

    assert cpu == {
        "model": "Apple M4",
        "flags": ["arm.FEAT_CRC32", "floatingpoint"],
        "logical_cpus": 10,
        "ram_bytes": 17179869184,
    }


@pytest.mark.parametrize(
    ("cpuinfo", "model", "flags"),
    [
        (_CPUINFO, "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz", "avx512f"),
        (_ARM_CPUINFO, "0xd0c", "asimd"),
    ],
)
def test_linux_cpu_reads_proc_files(cpuinfo: str, model: str, flags: str) -> None:
    files = {"/proc/cpuinfo": cpuinfo, "/proc/meminfo": _MEMINFO}
    cpu = benchmark_manifest.cpu_description("Linux", read_text=files.__getitem__)

    assert cpu["model"] == model
    assert flags in cpu["flags"]
    assert cpu["ram_bytes"] == 16384000 * 1024


def test_unknown_system_and_failing_probes_are_recorded_not_raised() -> None:
    assert "unavailable" in benchmark_manifest.cpu_description("Plan9")

    def broken():
        raise OSError("no sysctl")

    assert benchmark_manifest._probe(broken) == {"unavailable": "OSError: no sysctl"}


def test_host_load_records_load_and_macos_sleep_assertions() -> None:
    darwin = benchmark_manifest.host_load("Darwin", run=_fake_darwin)
    linux = benchmark_manifest.host_load("Linux")

    assert len(darwin["load_average_1_5_15"]) == 3
    assert darwin["power_assertions"] == {
        "PreventSystemSleep": "1",
        "PreventUserIdleSystemSleep": "0",
    }
    assert "power_assertions" not in linux
    assert linux["time_utc"].endswith("Z")


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


def test_start_and_end_record_one_run_and_detect_a_changed_tree(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "--quiet")
    (checkout / "a.txt").write_text("a\n")
    _git(checkout, "add", "a.txt")
    _git(
        checkout,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "a",
    )
    output = tmp_path / "run" / "manifest.json"

    benchmark_manifest.main(
        ["--phase", "start", "--output", str(output), "--checkout", str(checkout)]
    )
    started = json.loads(output.read_text())
    assert started["schema"] == "jaxgam-benchmark-manifest-v1"
    assert started["checkout_start"]["clean"] is True
    assert started["numeric"]["jax"]["dtype"] == "float64"
    assert Path(started["numeric"]["jax"]["jaxgam_package"]) == (
        Path(benchmark_manifest.__file__).resolve().parents[1] / "jaxgam"
    )
    assert "numpy" in {name.lower() for name in started["software"]["packages"]}
    assert "OPENBLAS_NUM_THREADS" in started["thread_environment"]
    assert "load_average_1_5_15" in started["host_start"]

    with pytest.raises(SystemExit):
        benchmark_manifest.main(["--phase", "start", "--output", str(output)])

    (checkout / "a.txt").write_text("changed\n")
    benchmark_manifest.main(
        ["--phase", "end", "--output", str(output), "--checkout", str(checkout)]
    )
    finished = json.loads(output.read_text())
    assert finished["checkout_end"]["clean"] is False
    assert finished["checkout_end"]["changed_paths"] == 1
    assert finished["same_checkout"] is False
    assert "load_average_1_5_15" in finished["host_end"]


def test_end_requires_a_started_manifest(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("{}")
    with pytest.raises(ValueError, match="--phase start"):
        benchmark_manifest.main(["--phase", "end", "--output", str(output)])
