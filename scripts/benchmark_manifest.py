"""Write one environment manifest per benchmark run directory.

A measurement session runs ``--phase start`` before its first case and
``--phase end`` after its last; both write into the same JSON file, so the
run directory records the host, the software, the numeric configuration and
the host load around every record it holds::

    make benchmark-manifest ARGS="--phase start --output $RUN/manifest.json"
    ... cases ...
    make benchmark-manifest ARGS="--phase end --output $RUN/manifest.json"

Every probe is best effort: a probe that fails is recorded as unavailable
with its reason instead of stopping the run.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "XLA_FLAGS",
    "JAX_NUM_THREADS",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_ENABLE_X64",
)
_SCHEMA = "jaxgam-benchmark-manifest-v1"
# This checkout's package goes first, so the dtype probe imports the jaxgam
# the run measures rather than another editable install.
_CHECKOUT = Path(__file__).resolve().parents[1]
if str(_CHECKOUT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT))


def _run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(
        command, cwd=cwd, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _probe(function: Callable[[], Any]) -> Any:
    """Run one probe; a failure is recorded, never raised."""
    try:
        return function()
    except Exception as error:  # every probe is best effort
        return {"unavailable": f"{type(error).__name__}: {error}"}


def _proc_field(text: str, name: str) -> str | None:
    """First ``name : value`` field of a /proc-style text block."""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == name:
            return value.strip()
    return None


def cpu_description(
    system: str | None = None,
    read_text: Callable[[str], str] = lambda path: Path(path).read_text(),
    run: Callable[[list[str]], str] = _run,
) -> dict[str, Any]:
    """CPU model, feature flags, core count and RAM for macOS or Linux."""
    system = system or platform.system()
    if system == "Darwin":
        flags_text = run(["sysctl", "hw.optional"])
        flags = sorted(
            line.split(":")[0].removeprefix("hw.optional.")
            for line in flags_text.splitlines()
            if line.rstrip().endswith(": 1")
        )
        return {
            "model": run(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "flags": flags,
            "logical_cpus": int(run(["sysctl", "-n", "hw.ncpu"])),
            "ram_bytes": int(run(["sysctl", "-n", "hw.memsize"])),
        }
    if system == "Linux":
        cpuinfo = read_text("/proc/cpuinfo")
        meminfo = read_text("/proc/meminfo")
        flags = _proc_field(cpuinfo, "flags") or _proc_field(cpuinfo, "Features")
        mem_kib = _proc_field(meminfo, "MemTotal")
        return {
            "model": _proc_field(cpuinfo, "model name")
            or _proc_field(cpuinfo, "CPU part"),
            "flags": sorted((flags or "").split()),
            "logical_cpus": os.cpu_count(),
            "ram_bytes": None if mem_kib is None else int(mem_kib.split()[0]) * 1024,
        }
    return {"unavailable": f"no CPU probe for {system}"}


def host_load(
    system: str | None = None, run: Callable[[list[str]], str] = _run
) -> dict[str, Any]:
    """Load averages and, on macOS, the active sleep-prevention assertions."""
    system = system or platform.system()
    one, five, fifteen = os.getloadavg()
    record: dict[str, Any] = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "load_average_1_5_15": [one, five, fifteen],
    }
    if system == "Darwin":
        # "   PreventSystemSleep             1" in the system-wide summary.
        wanted = ("PreventSystemSleep", "PreventUserIdleSystemSleep")
        assertions: dict[str, str | None] = dict.fromkeys(wanted)
        for line in run(["pmset", "-g", "assertions"]).splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in assertions:
                assertions[parts[0]] = parts[1]
        record["power_assertions"] = assertions
    return record


def checkout_state(root: Path, run: Callable[..., str] = _run) -> dict[str, Any]:
    """The checkout's commit and whether its tree is clean."""
    status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"], root)
    return {
        "path": str(root.resolve()),
        "commit": run(["git", "rev-parse", "HEAD"], root),
        "clean": status == "",
        "changed_paths": len(status.splitlines()),
    }


def numeric_configuration() -> dict[str, Any]:
    """dtype, devices and the BLAS/LAPACK providers in use."""
    import numpy as np
    import scipy

    record: dict[str, Any] = {
        "numpy_blas": np.show_config(mode="dicts")["Build Dependencies"],
        "scipy_blas": scipy.show_config(mode="dicts")["Build Dependencies"],
    }
    record["source_lapack_provider"] = _probe(_openblas_provider)
    record["jax"] = _probe(_jax_configuration)
    return record


def _openblas_provider() -> dict[str, str]:
    """The pinned provider as jaxgam loads it: by path, never as a module.

    Importing ``scipy_openblas32`` would run its eager global loader and leave
    it in ``sys.modules``, which the product rule forbids; the product's own
    loader reads the version from package metadata and the configuration
    from the library it opens with ``RTLD_LOCAL``.
    """
    from jaxgam.smooths import source_lapack

    provider = source_lapack._source_lapack().module
    return {
        "package": provider.package,
        "version": provider.__version__,
        "config": provider.get_openblas_config(),
        "library": str(provider.library_path),
        "loaded_by": "jaxgam.smooths.source_lapack (by path; module not imported)",
    }


def _jax_configuration() -> dict[str, Any]:
    import jax

    import jaxgam  # enables float64 as every fit does

    return {
        "jaxgam_package": str(Path(jaxgam.__file__).resolve().parent),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "dtype": "float64" if jax.config.jax_enable_x64 else "float32",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }


def software() -> dict[str, Any]:
    """Interpreter and every installed distribution with its version."""
    distributions = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": dict(sorted(distributions.items(), key=lambda kv: kv[0].lower())),
    }


def start_manifest(root: Path) -> dict[str, Any]:
    return {
        "schema": _SCHEMA,
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": _probe(cpu_description),
        "software": _probe(software),
        "numeric": _probe(numeric_configuration),
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_VARIABLES
        },
        "checkout_start": _probe(lambda: checkout_state(root)),
        "host_start": _probe(host_load),
    }


def end_manifest(existing: dict[str, Any], root: Path) -> dict[str, Any]:
    if existing.get("schema") != _SCHEMA:
        raise ValueError(
            "The output is not a benchmark manifest started by --phase start."
        )
    finished = dict(existing)
    finished["checkout_end"] = _probe(lambda: checkout_state(root))
    finished["host_end"] = _probe(host_load)
    start, end = finished.get("checkout_start"), finished["checkout_end"]
    finished["same_checkout"] = (
        isinstance(start, dict)
        and isinstance(end, dict)
        and start.get("commit") == end.get("commit")
        and bool(start.get("clean"))
        and bool(end.get("clean"))
    )
    return finished


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("start", "end"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkout", type=Path, default=Path(__file__).resolve().parents[1]
    )
    options = parser.parse_args(argv)
    if options.phase == "start":
        if options.output.exists():
            parser.error(f"{options.output} exists; a run has one manifest.")
        manifest = start_manifest(options.checkout)
    else:
        manifest = end_manifest(
            json.loads(options.output.read_text()), options.checkout
        )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(options.output)


if __name__ == "__main__":
    main()
