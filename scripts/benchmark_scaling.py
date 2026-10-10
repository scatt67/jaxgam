"""Emit isolated, reproducible scaling-evidence records as JSON.

This harness deliberately benchmarks one estimator at a time. It does not
compare dense REML, streamed fixed-sp PIRLS, QR, EFS, or ``mgcv::bam`` as if
they estimated the same quantity. A cold run is always a fresh subprocess
with a task-owned JAX compilation cache; no user or shared cache is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_MAX_COMPARISON_COEFFICIENTS = 2_048
_MAX_CHILD_DIAGNOSTIC_CHARS = 4_096


def _tolerance_record(tolerance: Any) -> dict[str, float]:
    return {"rtol": tolerance.rtol, "atol": tolerance.atol}


def _check_reference_imports(root: Path, *, r_parity: bool) -> None:
    """Require typed test references from the selected checkout, not uv's link."""
    names = ["tolerances.py"]
    if r_parity:
        names.append("r_bridge.py")
    for filename in names:
        module = importlib.import_module(f"tests.{filename.removesuffix('.py')}")
        actual = Path(module.__file__).resolve()
        expected = (root / "tests" / filename).resolve()
        if actual != expected:
            raise RuntimeError(
                f"Imported {module.__name__} from {actual}, "
                f"not selected checkout {expected}."
            )


@dataclass(frozen=True)
class ScalingCase:
    """One bounded, same-estimand benchmark configuration."""

    n_rows: int
    basis_dimension: int
    execution: str = "dense"
    batch_rows: int = 1024
    repeats: int = 1
    seed: int = 42
    result_mode: str = "full"
    smooth_count: int = 1

    def __post_init__(self) -> None:
        if self.n_rows <= 0:
            raise ValueError("n_rows must be positive.")
        if self.basis_dimension < 3:
            raise ValueError("basis_dimension must be at least 3.")
        if self.batch_rows <= 0:
            raise ValueError("batch_rows must be positive.")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive.")
        if self.smooth_count <= 0:
            raise ValueError("smooth_count must be positive.")
        if self.execution not in {
            "dense",
            "stream",
            "stream-qr",
            "stream-exact",
            "efs",
        }:
            raise ValueError(
                "execution must be 'dense', 'stream', 'stream-qr', "
                "'stream-exact', or 'efs'."
            )
        if self.result_mode not in {"full", "inference", "prediction"}:
            raise ValueError(
                "result_mode must be 'full', 'inference', or 'prediction'."
            )
        if (
            self.execution in {"stream", "stream-qr", "stream-exact", "efs"}
            and self.result_mode != "prediction"
        ):
            raise ValueError(
                f"execution={self.execution!r} supports result_mode='prediction' only."
            )


class ChildRunError(RuntimeError):
    """A failed isolated child with bounded, visible diagnostic streams."""

    def __init__(
        self, child_mode: str, root: Path, returncode: int, stdout: str, stderr: str
    ) -> None:
        self.child_mode = child_mode
        self.root = root
        self.returncode = returncode
        self.stdout = _bounded_child_diagnostic(stdout)
        self.stderr = _bounded_child_diagnostic(stderr)
        message = f"{child_mode} child failed for {root} (exit {returncode})"
        if self.stderr:
            message += f"\nchild stderr:\n{self.stderr}"
        if self.stdout:
            message += f"\nchild stdout:\n{self.stdout}"
        super().__init__(message)


def _bounded_child_diagnostic(stream: str) -> str:
    """Retain the error-bearing tail without flooding the parent CLI log."""
    if len(stream) <= _MAX_CHILD_DIAGNOSTIC_CHARS:
        return stream
    omitted = len(stream) - _MAX_CHILD_DIAGNOSTIC_CHARS
    tail = stream[-_MAX_CHILD_DIAGNOSTIC_CHARS:]
    return f"[{omitted} earlier characters omitted]\n{tail}"


@dataclass(frozen=True)
class _FitPlan:
    """One route callable and its optional harness-owned source counter."""

    run: Callable[[], Any]
    formula: str
    source: _CountingArrayRowSource | None


def isolated_cache_dir() -> tempfile.TemporaryDirectory[str]:
    """Create a task-owned cache directory that is removed only on exit."""
    return tempfile.TemporaryDirectory(prefix="jaxgam-benchmark-jax-")


def source_identity(root: Path) -> dict[str, str]:
    """Return the exact checkout and a hash of all uncommitted tracked edits."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bytearray(
        subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    )
    dirty.extend(
        subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
        )
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=root, text=True
    ).splitlines()
    for relative_path in sorted(untracked):
        path = root / relative_path
        dirty.extend(relative_path.encode())
        dirty.extend(b"\0")
        dirty.extend(path.read_bytes())
    return {"commit": commit, "dirty_sha256": hashlib.sha256(dirty).hexdigest()}


def loaded_source_identity(module: Any, root: Path) -> dict[str, str]:
    """Verify the imported package comes from ``root`` and hash its Python source."""
    package_dir = Path(module.__file__).resolve().parent
    try:
        package_dir.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"Imported jaxgam from {package_dir}, not requested checkout {root}."
        ) from error

    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return {
        "loaded_package": str(package_dir),
        "loaded_source_sha256": digest.hexdigest(),
    }


def harness_identity() -> dict[str, str]:
    """Identify the script executed by every isolated child process."""
    path = Path(__file__).resolve()
    return {
        "harness_path": str(path),
        "harness_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def peak_rss_bytes() -> int:
    """Return the process high-water RSS, normalized to bytes."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def current_rss_bytes() -> int | None:
    """Return current resident memory when the host exposes it, else ``None``."""
    if sys.platform.startswith("linux"):
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    if platform.system() == "Darwin":
        try:
            output = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
            )
        except OSError:
            return None
        return int(output.strip()) * 1024
    return None


def _directory_bytes(path: Path | None) -> int | None:
    """Return regular-file bytes for a task-owned cache, if it still exists."""
    if path is None or not path.is_dir():
        return None
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _smooth_variables(case: ScalingCase) -> tuple[str, ...]:
    """Return stable additive term names, preserving the one-smooth fixture."""
    if case.smooth_count == 1:
        return ("x",)
    return tuple(f"x{index}" for index in range(case.smooth_count))


def _formula(case: ScalingCase) -> str:
    """Build one ordered additive-CR formula for every route and child."""
    terms = " + ".join(
        f"s({variable}, k={case.basis_dimension}, bs='cr')"
        for variable in _smooth_variables(case)
    )
    return f"y ~ {terms}"


def _smoothing_parameter_order(case: ScalingCase) -> list[str]:
    """Name the fixed-sp vector order derived from the formula term order."""
    return [f"s({variable})" for variable in _smooth_variables(case)]


def _fixed_smoothing_parameters(case: ScalingCase) -> list[float]:
    return [1.0] * case.smooth_count


def _make_data(case: ScalingCase) -> pd.DataFrame:
    rng = np.random.default_rng(case.seed)
    variables = _smooth_variables(case)
    columns = {variable: rng.uniform(0.0, 1.0, case.n_rows) for variable in variables}
    if case.smooth_count == 1:
        # Preserve the original one-smooth generator byte-for-byte so the
        # c1a3134 dense adapter remains a valid paired baseline.
        signal = np.sin(2.0 * np.pi * columns["x"])
    else:
        signal = sum(
            0.4 * np.sin(2.0 * np.pi * (index + 1) * columns[variable])
            for index, variable in enumerate(variables)
        )
    if case.execution == "efs":
        poisson_signal = 0.4 * signal if case.smooth_count == 1 else signal
        y = rng.poisson(np.exp(0.2 + poisson_signal))
    else:
        y = signal + rng.normal(0.0, 0.3, case.n_rows)
    return pd.DataFrame({**columns, "y": y})


def _input_provenance(data: pd.DataFrame, case: ScalingCase) -> dict[str, Any]:
    digest = hashlib.sha256()
    for name in data.columns:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(_array_sha256(data[name].to_numpy()).encode())
    return {
        "generator": (
            ("poisson_sine" if case.smooth_count == 1 else "poisson_additive_sine")
            if case.execution == "efs"
            else (
                "gaussian_sine" if case.smooth_count == 1 else "gaussian_additive_sine"
            )
        ),
        "seed": case.seed,
        "smooth_variables": list(_smooth_variables(case)),
        "frame_sha256": digest.hexdigest(),
    }


def _array_sha256(values: Any) -> str:
    """Hash a numeric observable with explicit shape and dtype framing."""
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_penalty_contract(n_coef: int, penalties: Any) -> dict[str, Any]:
    """Hash penalties in constrained public coordinates, one matrix at a time.

    Legacy setup stores global matrices while current setup stores local blocks.
    Expanding one local matrix into a temporary ``p x p`` public-coordinate
    matrix makes their mathematical contracts comparable without retaining an
    ``m x p x p`` stack. Values are rounded to 12 decimal places before
    hashing to ignore harmless representation-level roundoff.
    """
    digest = hashlib.sha256()
    ranks: list[int] = []
    digest.update(str((n_coef,)).encode())
    if penalties is None:
        return {"penalty_public_coordinate_sha256": digest.hexdigest(), "ranks": []}

    if hasattr(penalties, "blocks"):
        entries = sorted(
            (
                sp_index,
                block.start,
                block.stop,
                penalty,
                rank,
            )
            for block in penalties.blocks
            for sp_index, penalty, rank in zip(
                block.sp_indices, block.local_penalties, block.ranks, strict=True
            )
        )
        for sp_index, start, stop, penalty, rank in entries:
            matrix = np.zeros((n_coef, n_coef), dtype=np.float64)
            matrix[start:stop, start:stop] = penalty.dense()
            digest.update(str((sp_index, rank)).encode())
            digest.update(_array_sha256(np.round(matrix, decimals=12)).encode())
            ranks.append(int(rank))
    else:
        for sp_index, penalty in enumerate(penalties.penalties):
            matrix = np.asarray(penalty.S, dtype=np.float64)
            digest.update(str((sp_index, penalty.rank)).encode())
            digest.update(_array_sha256(np.round(matrix, decimals=12)).encode())
            ranks.append(int(penalty.rank))
    return {
        "penalty_public_coordinate_sha256": digest.hexdigest(),
        "ranks": ranks,
    }


def _model_contract(
    basis_fingerprint: str, n_coef: int, penalties: Any, rank_deficit: int
) -> dict[str, Any]:
    """Record a representation-independent frozen public model contract."""
    return {
        "basis_fingerprint": basis_fingerprint,
        "coordinates": "constrained_public_coefficients",
        "penalty": _canonical_penalty_contract(n_coef, penalties),
        "unpenalized_rank_deficit": rank_deficit,
    }


def _numeric_comparison(left: Any, right: Any) -> dict[str, Any]:
    """Return explicit strict/moderate parity evidence for bounded observables."""
    from tests.tolerances import MODERATE, STRICT

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    same_shape = left_array.shape == right_array.shape
    finite = bool(np.all(np.isfinite(left_array)) and np.all(np.isfinite(right_array)))
    if not same_shape or not finite:
        return {
            "available": False,
            "same_shape": same_shape,
            "finite": finite,
            "max_abs_error": None,
            "strict_allclose": False,
            "moderate_allclose": False,
        }
    max_abs_error = float(np.max(np.abs(left_array - right_array)))
    return {
        "available": True,
        "same_shape": True,
        "finite": True,
        "max_abs_error": max_abs_error,
        "strict_allclose": bool(
            np.allclose(left_array, right_array, rtol=STRICT.rtol, atol=STRICT.atol)
        ),
        "moderate_allclose": bool(
            np.allclose(left_array, right_array, rtol=MODERATE.rtol, atol=MODERATE.atol)
        ),
    }


def _timed(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def _timed_fit(fit: Callable[[], Any]) -> tuple[Any, float]:
    """Time a fit through device completion, never merely launch latency."""
    import jax

    started = time.perf_counter()
    result = fit()
    jax.block_until_ready(result.coefficients)
    return result, time.perf_counter() - started


def _fit_validity(result: Any) -> dict[str, Any]:
    """Check a completed timed fit before using its latency as evidence."""
    values = {
        "coefficients": result.coefficients,
        "smoothing_params": result.smoothing_params,
        "score": result.score,
        "deviance": result.deviance,
        "scale": result.scale,
        "edf_total": result.edf_total,
    }
    if getattr(result, "score_scale", None) is not None:
        values["score_scale"] = result.score_scale
    finite_fields = {
        name: bool(np.all(np.isfinite(np.asarray(value))))
        for name, value in values.items()
    }
    converged = bool(result.converged)
    return {
        "valid": converged and all(finite_fields.values()),
        "converged": converged,
        "finite_fields": finite_fields,
        "status": str(result.convergence_info),
    }


def _require_valid_timed_fit(result: Any, *, label: str) -> dict[str, Any]:
    validity = _fit_validity(result)
    if not validity["valid"]:
        raise RuntimeError(f"{label} returned an invalid fit: {validity}")
    return validity


def _partial_public_dataclass_array_inventory(value: Any) -> dict[str, int]:
    """Count direct dataclass arrays; this is not a retained-object graph size."""
    seen: set[int] = set()
    arrays = 0
    total_bytes = 0

    def visit(item: Any) -> None:
        nonlocal arrays, total_bytes
        identifier = id(item)
        if identifier in seen:
            return
        is_array = isinstance(item, np.ndarray)
        is_jax_array = all(
            hasattr(item, attribute) for attribute in ("shape", "dtype", "nbytes")
        )
        if is_array or is_jax_array:
            seen.add(identifier)
            arrays += 1
            total_bytes += int(item.nbytes)
            return
        if is_dataclass(item):
            seen.add(identifier)
            for item_field in fields(item):
                visit(getattr(item, item_field.name))
        elif isinstance(item, dict):
            seen.add(identifier)
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            seen.add(identifier)
            for child in item:
                visit(child)

    visit(value)
    return {"array_count": arrays, "array_bytes": total_bytes}


def _build_fit(case: ScalingCase, data: pd.DataFrame) -> _FitPlan:
    """Build one bounded route without silently changing its estimator."""
    from jaxgam import GAM

    formula = _formula(case)
    fixed_sp = _fixed_smoothing_parameters(case)
    if case.execution == "dense":

        def fit() -> Any:
            return GAM(formula, family="gaussian", sp=fixed_sp).fit(
                data, result=case.result_mode
            )

        return _FitPlan(fit, formula, None)

    if case.execution == "efs":

        def fit() -> Any:
            return GAM(formula, family="poisson", optimizer="efs").fit(
                data, result="prediction"
            )

        return _FitPlan(fit, formula, None)

    from jaxgam.control import FitControl
    from jaxgam.data.source import ArrayRowSource

    source = _CountingArrayRowSource(
        ArrayRowSource(
            {name: data[name].to_numpy() for name in _smooth_variables(case)},
            y=data.y.to_numpy(),
        )
    )
    control = FitControl(
        execution="stream", batch_rows=case.batch_rows, uncertainty="fisher"
    )

    if case.execution == "stream":

        def fit() -> Any:
            return GAM(formula, family="gaussian", sp=fixed_sp, control=control).fit(
                source, result="prediction"
            )

        return _FitPlan(fit, formula, source)

    if case.execution == "stream-exact":
        exact_control = FitControl(
            execution="stream",
            linear_solver="qr",
            batch_rows=case.batch_rows,
            uncertainty="fisher",
        )

        def fit() -> Any:
            return GAM(formula, family="gaussian", control=exact_control).fit(
                source, result="prediction"
            )

        return _FitPlan(fit, formula, source)

    def fit() -> Any:
        qr_control = FitControl(
            execution="stream",
            linear_solver="qr",
            batch_rows=case.batch_rows,
            uncertainty="fisher",
        )
        return GAM(formula, family="gaussian", sp=fixed_sp, control=qr_control).fit(
            source, result="prediction"
        )

    return _FitPlan(fit, formula, source)


def _route_contract(case: ScalingCase) -> dict[str, Any]:
    """Describe what the requested adapter actually measures."""
    contracts = {
        "dense": {
            "adapter": "public",
            "strategy": "fixed-sp dense PIRLS",
            "result_mode": case.result_mode,
        },
        "stream": {
            "adapter": "public",
            "strategy": "fixed-sp streamed Cholesky PIRLS",
            "result_mode": "prediction",
        },
        "stream-qr": {
            "adapter": "public",
            "strategy": "fixed-sp streamed pivoted-QR PIRLS",
            "result_mode": "prediction",
        },
        "stream-exact": {
            "adapter": "public",
            "strategy": "exact streamed Gaussian REML with estimated smoothing",
            "result_mode": "prediction",
        },
        "efs": {
            "adapter": "public",
            "strategy": "dense Poisson/log EFS with estimated smoothing",
            "result_mode": "prediction",
        },
    }
    return {
        "requested_execution": case.execution,
        "smoothing_parameter_order": _smoothing_parameter_order(case),
        "fixed_smoothing_parameter_values": (
            _fixed_smoothing_parameters(case)
            if case.execution not in {"efs", "stream-exact"}
            else None
        ),
        **contracts[case.execution],
    }


def _thread_configuration() -> dict[str, str | None]:
    """Record requested thread settings without inferring runtime enforcement."""
    return {
        name.lower(): os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "XLA_FLAGS",
            "JAX_NUM_THREADS",
        )
    }


def _blas_metadata() -> dict[str, str]:
    """Return the available NumPy build BLAS description without benchmarking it."""
    config = getattr(np.__config__, "CONFIG", {})
    dependencies = config.get("Build Dependencies", {})
    blas = dependencies.get("blas", {})
    lapack = dependencies.get("lapack", {})
    return {
        "blas_name": str(blas.get("name", "unknown")),
        "blas_found": str(blas.get("found", "unknown")),
        "lapack_name": str(lapack.get("name", "unknown")),
        "lapack_found": str(lapack.get("found", "unknown")),
    }


def _runtime_metadata(jax: Any, jaxgam: Any) -> dict[str, Any]:
    """Record versions and host configuration that can alter wall-clock evidence."""
    import jaxlib
    import scipy

    return {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jaxgam": jaxgam.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "requested_thread_environment": _thread_configuration(),
        "requested_thread_environment_status": (
            "environment settings observed at process start; they do not prove "
            "actual runtime thread counts"
        ),
        "blas": _blas_metadata(),
        "dtype": "float64",
        "cold_cache": "task-owned temporary directory",
    }


def _device_peak_memory(jax: Any) -> tuple[int | None, str]:
    """Read backend allocator peak when JAX exposes it for this process."""
    peaks = []
    for device in jax.devices():
        try:
            stats = device.memory_stats()
        except (AttributeError, NotImplementedError):
            stats = None
        if stats is not None and stats.get("peak_bytes_in_use") is not None:
            peaks.append(int(stats["peak_bytes_in_use"]))
    if not peaks:
        return None, "JAX backend did not expose peak_bytes_in_use"
    return max(peaks), "backend allocator peak since worker process start"


class _CountingArrayRowSource:
    """Count started scans and numeric bytes exposed by yielded row batches."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self.scan_calls = 0
        self.batch_bytes = 0
        self.rows_yielded = 0

    @property
    def n_rows(self) -> int:
        return self._source.n_rows

    def fingerprint(self) -> str:
        return self._source.fingerprint()

    def scan(self, batch_rows: int) -> Any:
        self.scan_calls += 1
        for batch in self._source.scan(batch_rows):
            values = (
                *batch.columns.values(),
                batch.y,
                batch.weight,
                batch.offset,
                batch.valid,
                batch.row_positions,
            )
            self.batch_bytes += sum(
                np.asarray(value).nbytes for value in values if value is not None
            )
            self.rows_yielded += len(batch.row_positions)
            yield batch


def _source_scan_count(source: _CountingArrayRowSource | None) -> int | None:
    return None if source is None else source.scan_calls


def _source_scan_delta(
    source: _CountingArrayRowSource | None, before: int | None
) -> int | None:
    if source is None or before is None:
        return None
    return source.scan_calls - before


def _source_batch_bytes(source: _CountingArrayRowSource | None) -> int | None:
    return None if source is None else source.batch_bytes


def _source_batch_bytes_delta(
    source: _CountingArrayRowSource | None, before: int | None
) -> int | None:
    if source is None or before is None:
        return None
    return source.batch_bytes - before


def run_public_smoke(root: Path) -> dict[str, Any]:
    """Exercise public dense and stream routes; this intentionally has no timings."""
    import jaxgam
    from jaxgam import GAM
    from jaxgam.control import FitControl
    from jaxgam.data.source import ArrayRowSource

    loaded_runtime = loaded_source_identity(jaxgam, root)
    x = np.linspace(0.05, 0.95, 19)
    data = pd.DataFrame({"x": x, "y": np.sin(2.0 * np.pi * x)})
    source = _CountingArrayRowSource(
        ArrayRowSource({"x": data.x.to_numpy()}, y=data.y.to_numpy())
    )
    formula = "y ~ s(x, k=5, bs='cr')"
    dense = GAM(formula, family="gaussian", sp=[0.2]).fit(data, result="prediction")
    streamed = GAM(
        formula,
        family="gaussian",
        sp=[0.2],
        control=FitControl(execution="stream", batch_rows=7),
    ).fit(source, result="prediction")
    prediction_difference = np.asarray(streamed.predict(data)) - np.asarray(
        dense.predict(data)
    )
    return {
        "dense_execution_route": dense.execution_route,
        "stream_execution_route": streamed.execution_route,
        "source_scan_calls": source.scan_calls,
        "source_scans_publicly_available": hasattr(streamed, "source_scans"),
        "source_scans_public": getattr(streamed, "source_scans", None),
        "source_scans_public_scope": (
            "public execution fit scans; proxy count also includes preparation"
        ),
        "coefficient_max_abs_difference": float(
            np.max(np.abs(streamed.coefficients - dense.coefficients))
        ),
        "prediction_max_abs_difference": float(np.max(np.abs(prediction_difference))),
        "score_abs_difference": float(abs(streamed.score - dense.score)),
        "runtime": {**source_identity(root), **loaded_runtime},
    }


def run_setup(case: ScalingCase, root: Path) -> dict[str, Any]:
    """Measure Phase 1 separately in an isolated process for one route."""
    import jaxgam
    from jaxgam.families.registry import get_family
    from jaxgam.formula.parser import parse_formula

    loaded_runtime = loaded_source_identity(jaxgam, root)
    data = _make_data(case)
    formula = _formula(case)
    family_name = "poisson" if case.execution == "efs" else "gaussian"
    rss_high_water_before_setup = peak_rss_bytes()
    rss_current_before_setup = current_rss_bytes()
    started = time.perf_counter()
    spec = parse_formula(formula)
    if case.execution in {"dense", "efs"}:
        from jaxgam.fitting.data import FittingData
        from jaxgam.formula.design import ModelSetup

        setup = ModelSetup.build(spec, data)
        setup_s = time.perf_counter() - started
        rss_high_water_after_setup = peak_rss_bytes()
        rss_current_after_setup = current_rss_bytes()
        basis_fingerprint = _array_sha256(setup.X)
        n_coef = int(setup.X.shape[1])
        n_penalties = 0 if setup.penalties is None else setup.penalties.n_penalties
        fingerprint_kind = "dense_model_matrix_sha256"
        fitting_data = FittingData.from_setup(setup, get_family(family_name))
        public_penalties = setup.penalties
        rank_deficit = fitting_data.rank_deficit
    else:
        from jaxgam.data.source import ArrayRowSource
        from jaxgam.formula.prepare import prepare_model

        source = _CountingArrayRowSource(
            ArrayRowSource(
                {name: data[name].to_numpy() for name in _smooth_variables(case)},
                y=data.y.to_numpy(),
            )
        )
        source_scans_before = source.scan_calls
        source_bytes_before = source.batch_bytes
        prepared = prepare_model(spec, source, family=get_family(family_name))
        source_scans_after = source.scan_calls
        source_bytes_after = source.batch_bytes
        setup_s = time.perf_counter() - started
        rss_high_water_after_setup = peak_rss_bytes()
        rss_current_after_setup = current_rss_bytes()
        basis_fingerprint = prepared.basis_fingerprint
        n_coef = prepared.n_coef
        assert prepared.fitting is not None
        n_penalties = prepared.fitting.penalty_structure.n_penalties
        fingerprint_kind = "stream_prepared_basis_fingerprint"
        public_penalties = prepared.penalties
        rank_deficit = prepared.fitting.unpenalized_rank_deficit
    if case.execution in {"dense", "efs"}:
        source_scans_before = None
        source_scans_after = None
        source_bytes_before = None
        source_bytes_after = None
    frozen_contract = _model_contract(
        basis_fingerprint, n_coef, public_penalties, rank_deficit
    )
    return {
        "formula": formula,
        "route": _route_contract(case),
        "input": _input_provenance(data, case),
        "timings": {"setup_s": setup_s},
        "model": {
            "n_rows": case.n_rows,
            "n_coef": n_coef,
            "n_penalties": n_penalties,
            "basis_fingerprint": basis_fingerprint,
            "basis_fingerprint_kind": fingerprint_kind,
            "frozen_contract": frozen_contract,
        },
        "memory": {
            "rss_high_water_before_setup_bytes": rss_high_water_before_setup,
            "rss_current_before_setup_bytes": rss_current_before_setup,
            "rss_high_water_after_setup_bytes": rss_high_water_after_setup,
            "rss_current_after_setup_bytes": rss_current_after_setup,
            "rss_scope": (
                "fresh-process high-water/current RSS sampled immediately after "
                "parse_formula and ModelSetup.build or prepare_model; includes "
                "imports and generated input, excludes later FittingData, "
                "penalty fingerprint and record assembly"
            ),
            "input_bytes": int(data.memory_usage(index=True, deep=True).sum()),
        },
        "source_scans": {
            "available": source_scans_after is not None,
            "setup_delta": (
                None
                if source_scans_after is None or source_scans_before is None
                else source_scans_after - source_scans_before
            ),
            "setup_batch_bytes": (
                None
                if source_bytes_after is None or source_bytes_before is None
                else source_bytes_after - source_bytes_before
            ),
            "batch_bytes_scope": (
                "sum of numeric array nbytes exposed by yielded RowBatch fields; "
                "not physical disk reads or unique backing storage"
            ),
            "status": (
                "measured RowSource.scan invocations during streamed preparation"
                " (started scan iterations)"
                if source_scans_after is not None
                else "not applicable to a dense setup route"
            ),
        },
        "runtime": {
            **source_identity(root),
            **loaded_runtime,
            **harness_identity(),
        },
    }


def run_case(case: ScalingCase, root: Path) -> dict[str, Any]:
    """Run one fresh-process cold fit and repeated warm fits.

    Setup is measured separately by ``run_setup``. This function reports only
    end-to-end fit calls, each synchronized through coefficient completion.
    The two stages run in separate clean processes, so their timings are not
    addends for a single fit.
    """
    import jax

    import jaxgam

    loaded_runtime = loaded_source_identity(jaxgam, root)
    data = _make_data(case)
    fit_plan = _build_fit(case, data)
    fit = fit_plan.run
    formula = fit_plan.formula
    route = _route_contract(case)
    rss_high_water_before = peak_rss_bytes()
    rss_current_before = current_rss_bytes()

    cold_scan_before = _source_scan_count(fit_plan.source)
    cold_bytes_before = _source_batch_bytes(fit_plan.source)
    cold_result, cold_fit_s = _timed_fit(fit)
    cold_validity = _require_valid_timed_fit(cold_result, label="cold fit")
    cold_scan_delta = _source_scan_delta(fit_plan.source, cold_scan_before)
    cold_bytes_delta = _source_batch_bytes_delta(fit_plan.source, cold_bytes_before)
    rss_high_water_after_cold = peak_rss_bytes()
    rss_current_after_cold = current_rss_bytes()

    started = time.perf_counter()
    coefficients = np.asarray(cold_result.coefficients)
    result_materialization_s = time.perf_counter() - started
    started = time.perf_counter()
    prediction_scan_before = _source_scan_count(fit_plan.source)
    prediction_bytes_before = _source_batch_bytes(fit_plan.source)
    prediction = cold_result.predict(data.iloc[:128])
    prediction_s = time.perf_counter() - started
    prediction_scan_delta = _source_scan_delta(fit_plan.source, prediction_scan_before)
    prediction_bytes_delta = _source_batch_bytes_delta(
        fit_plan.source, prediction_bytes_before
    )
    prediction_se_s: float | None = None
    prediction_se_status = "not requested for the EFS prediction-only route"
    if case.execution in {"dense", "stream", "stream-qr", "stream-exact"}:
        started = time.perf_counter()
        _, standard_error = cold_result.predict(data.iloc[:128], se_fit=True)
        prediction_se_s = time.perf_counter() - started
        if not np.all(np.isfinite(standard_error)):
            raise FloatingPointError(
                "benchmark prediction SE contains nonfinite values"
            )
        prediction_se_status = "measured for 128 requested rows"

    public_iterations = int(cold_result.n_iter)
    efs_diagnostics = getattr(cold_result, "optimizer_diagnostics", None)
    if case.execution == "stream-exact":
        iteration_semantics = "outer exact-REML optimizer iterations"
        inner_iterations = None
        outer_iterations = public_iterations
    elif case.execution == "efs":
        iteration_semantics = "outer EFS optimizer iterations"
        inner_iterations = (
            None if efs_diagnostics is None else efs_diagnostics.inner_iterations
        )
        outer_iterations = public_iterations
    else:
        iteration_semantics = (
            "public fixed-sp fit iteration count; PIRLS sub-iterations are "
            "not exposed separately"
        )
        inner_iterations = None
        outer_iterations = None
    result_metadata = {
        "cold_fit_validity": cold_validity,
        "converged": bool(cold_result.converged),
        "status": str(cold_result.convergence_info),
        "n_iter": public_iterations,
        "n_iter_semantics": iteration_semantics,
        "inner_iterations": inner_iterations,
        "outer_iterations": outer_iterations,
        "n_coef": int(coefficients.size),
        "n_penalties": int(np.asarray(cold_result.smoothing_params).size),
        "score": float(cold_result.score),
        "deviance": float(cold_result.deviance),
        "scale": float(cold_result.scale),
        "score_scale": (
            None
            if getattr(cold_result, "score_scale", None) is None
            else float(cold_result.score_scale)
        ),
        "edf_total": float(cold_result.edf_total),
        "smoothing_params": np.asarray(cold_result.smoothing_params).tolist(),
        "known_workspace_bytes": getattr(cold_result, "known_workspace_bytes", None),
        "exact_optimizer_diagnostics": (
            None
            if getattr(cold_result, "exact_optimizer_diagnostics", None) is None
            else {
                "status": cold_result.exact_optimizer_diagnostics.status,
                "n_evaluations": cold_result.exact_optimizer_diagnostics.n_evaluations,
                "n_accepted": cold_result.exact_optimizer_diagnostics.n_accepted,
                "projected_gradient_inf": (
                    cold_result.exact_optimizer_diagnostics.projected_gradient_inf
                ),
                "startup_source_scans": (
                    cold_result.exact_optimizer_diagnostics.startup_source_scans
                ),
            }
        ),
        "coefficient_sha256": _array_sha256(coefficients),
        "prediction_sha256": _array_sha256(prediction),
        "comparison_vectors": {
            "coefficient_values": (
                coefficients.tolist()
                if coefficients.size <= _MAX_COMPARISON_COEFFICIENTS
                else None
            ),
            "coefficient_values_status": (
                "recorded"
                if coefficients.size <= _MAX_COMPARISON_COEFFICIENTS
                else "omitted_above_bounded_comparison_limit"
            ),
            "prediction_values": np.asarray(prediction).reshape(-1)[:128].tolist(),
            "prediction_values_status": "first 128 requested-row predictions",
        },
        "partial_public_dataclass_array_inventory": (
            _partial_public_dataclass_array_inventory(cold_result)
        ),
        "partial_inventory_status": (
            "direct public dataclass fields only; arrays in non-dataclass "
            "predictor/smooth objects and ndarray view backing storage are excluded"
        ),
        "source_scans": {
            "available": fit_plan.source is not None,
            "counter": (
                "started RowSource.scan iterations on the original training source"
            ),
            "cold_fit_delta": cold_scan_delta,
            "cold_fit_batch_bytes": cold_bytes_delta,
            "cold_fit_scope": (
                "end-to-end fit call; includes preparation performed inside the "
                "public route"
            ),
            "warm_fit_deltas": [],
            "warm_fit_batch_bytes": [],
            "training_source_scan_delta_during_prediction": prediction_scan_delta,
            "training_source_batch_bytes_during_prediction": prediction_bytes_delta,
            "batch_bytes_scope": (
                "sum of numeric array nbytes exposed by yielded RowBatch fields; "
                "not physical disk reads or unique backing storage"
            ),
            "prediction_source_scope": (
                "predict received a DataFrame; zero means it did not rescan the "
                "original training RowSource, not that prediction used no source"
            ),
            "status": (
                "fit deltas measure started training-source scan iterations; "
                "prediction is measured separately"
                if fit_plan.source is not None
                else "not applicable to a dense fit route"
            ),
        },
        "execution_route": getattr(cold_result, "execution_route", None),
        "execution_route_available": hasattr(cold_result, "execution_route"),
        "route": route,
    }
    # Do not retain a cold result while collecting warm-fit RSS evidence.
    del cold_result

    warm_times: list[float] = []
    warm_validity: list[dict[str, Any]] = []
    warm_scan_deltas: list[int | None] = []
    warm_batch_bytes: list[int | None] = []
    for repeat in range(case.repeats):
        warm_scan_before = _source_scan_count(fit_plan.source)
        warm_bytes_before = _source_batch_bytes(fit_plan.source)
        warm_result, elapsed = _timed_fit(fit)
        warm_validity.append(
            _require_valid_timed_fit(warm_result, label=f"warm fit {repeat + 1}")
        )
        warm_times.append(elapsed)
        warm_scan_deltas.append(_source_scan_delta(fit_plan.source, warm_scan_before))
        warm_batch_bytes.append(
            _source_batch_bytes_delta(fit_plan.source, warm_bytes_before)
        )
        del warm_result
    result_metadata["source_scans"]["warm_fit_deltas"] = warm_scan_deltas
    result_metadata["source_scans"]["warm_fit_batch_bytes"] = warm_batch_bytes
    result_metadata["warm_fit_validity"] = warm_validity

    rss_high_water_after_warm = peak_rss_bytes()
    rss_current_after_warm = current_rss_bytes()
    cache_setting = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    cache_path = Path(cache_setting) if cache_setting is not None else None
    device_peak_bytes, device_peak_status = _device_peak_memory(jax)
    return {
        "case": asdict(case),
        "formula": formula,
        "route": route,
        "estimator": route["strategy"],
        "result_mode": route["result_mode"],
        "input": _input_provenance(data, case),
        "basis_comparison": {
            "status": "not_checked",
            "reason": "single-checkout record; use a paired comparison before claim",
        },
        "timings": {
            "setup_s": None,
            "setup_timing_status": (
                "separately measured in a distinct clean setup child process; "
                "not addable to this fit timing"
            ),
            "cold_compile_and_fit_s": cold_fit_s,
            "warm_fit_median_s": float(np.median(warm_times)),
            "coefficient_access_s": result_materialization_s,
            "result_finish_s": None,
            "result_finish_status": "public fit returns an already finished result",
            "prediction_s": prediction_s,
            "prediction_se_s": prediction_se_s,
            "prediction_se_status": prediction_se_status,
            "derivative_s": None,
            "derivative_status": (
                "public fit result does not expose separate derivative time"
            ),
        },
        "memory": {
            "rss_high_water_before_bytes": rss_high_water_before,
            "rss_high_water_after_cold_bytes": rss_high_water_after_cold,
            "rss_high_water_after_warm_bytes": rss_high_water_after_warm,
            "rss_current_before_bytes": rss_current_before,
            "rss_current_after_cold_bytes": rss_current_after_cold,
            "rss_current_after_warm_bytes": rss_current_after_warm,
            "input_bytes": int(data.memory_usage(index=True, deep=True).sum()),
            "temporary_compilation_cache_bytes": _directory_bytes(cache_path),
            "temporary_compilation_cache_status": (
                "measured before the task-owned cache directory is removed"
            ),
            "device_peak_memory_bytes": device_peak_bytes,
            "device_peak_memory_status": device_peak_status,
            "compiler_memory_bytes": None,
            "compiler_memory_status": "not captured by the public fit-result adapter",
        },
        "result": result_metadata,
        "runtime": {
            **source_identity(root),
            **loaded_runtime,
            **harness_identity(),
            **_runtime_metadata(jax, jaxgam),
        },
    }


def run_isolated(case: ScalingCase, root: Path) -> dict[str, Any]:
    """Run separately measured setup plus cold/warm fitting in child processes."""
    record = {
        "setup": _run_child(case, root, "setup"),
        "fit": _run_child(case, root, "fit"),
    }
    record["within_run_consistency"] = _within_run_consistency(record)
    return record


def _run_child(case: ScalingCase, root: Path, child_mode: str) -> dict[str, Any]:
    """Execute one child operation with a never-shared cold cache."""
    with isolated_cache_dir() as cache:
        environment = os.environ | {
            "JAX_COMPILATION_CACHE_DIR": cache,
            "PYTHONPATH": str(root),
        }
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                child_mode,
                "--case",
                json.dumps(asdict(case)),
                "--checkout",
                str(root.resolve()),
            ],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode:
        raise ChildRunError(
            child_mode, root, completed.returncode, completed.stdout, completed.stderr
        )
    return json.loads(completed.stdout)


def _within_run_consistency(record: dict[str, Any]) -> dict[str, Any]:
    """Verify separately measured setup/fit children used the same contract."""
    setup = record["setup"]
    fit = record["fit"]
    runtime_keys = (
        "commit",
        "dirty_sha256",
        "loaded_source_sha256",
        "harness_sha256",
    )
    checks = {
        "same_formula": setup["formula"] == fit["formula"],
        "same_input": setup["input"] == fit["input"],
        "same_route": setup["route"] == fit["route"],
        "same_source": all(
            key in setup["runtime"]
            and key in fit["runtime"]
            and setup["runtime"][key] == fit["runtime"][key]
            for key in runtime_keys
        ),
        "same_coefficient_count": (setup["model"]["n_coef"] == fit["result"]["n_coef"]),
        "same_penalty_count": (
            setup["model"]["n_penalties"] == fit["result"]["n_penalties"]
        ),
    }
    return {"valid": all(checks.values()), "checks": checks}


def compare_checkouts(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Report parity only when baseline/current use the identical route contract."""
    from tests.tolerances import MODERATE, STRICT

    baseline_setup = baseline["setup"]
    current_setup = current["setup"]
    baseline_fit = baseline["fit"]
    current_fit = current["fit"]
    baseline_consistency = baseline.get("within_run_consistency", {})
    current_consistency = current.get("within_run_consistency", {})
    route_fields = ("requested_execution", "adapter", "strategy", "result_mode")
    same_route_contract = all(
        baseline_setup["route"].get(field) == current_setup["route"].get(field)
        for field in route_fields
    )
    compatible = (
        baseline_consistency.get("valid") is True
        and current_consistency.get("valid") is True
        and same_route_contract
        and baseline_setup["formula"] == current_setup["formula"]
        and baseline_setup["input"] == current_setup["input"]
        and baseline_setup["model"]["basis_fingerprint_kind"]
        == current_setup["model"]["basis_fingerprint_kind"]
        and baseline_setup["model"]["basis_fingerprint"]
        == current_setup["model"]["basis_fingerprint"]
        and baseline_setup["model"]["n_coef"] == current_setup["model"]["n_coef"]
        and baseline_setup["model"]["n_penalties"]
        == current_setup["model"]["n_penalties"]
        and baseline_setup["model"]["frozen_contract"]
        == current_setup["model"]["frozen_contract"]
        and baseline_setup["runtime"].get("harness_sha256") is not None
        and baseline_setup["runtime"].get("harness_sha256")
        == current_setup["runtime"].get("harness_sha256")
    )
    if not compatible:
        return {
            "status": "incompatible",
            "same_route_contract": same_route_contract,
            "baseline_within_run_consistent": baseline_consistency.get("valid"),
            "current_within_run_consistent": current_consistency.get("valid"),
            "same_formula": baseline_setup["formula"] == current_setup["formula"],
            "same_input": baseline_setup["input"] == current_setup["input"],
            "reason": (
                "Baseline/current differ in route, result semantics, input, or "
                "frozen-basis provenance; numerical parity is unavailable."
            ),
        }
    baseline_vectors = baseline_fit["result"]["comparison_vectors"]
    current_vectors = current_fit["result"]["comparison_vectors"]
    return {
        "status": "comparable",
        "same_route_contract": True,
        "baseline_within_run_consistent": True,
        "current_within_run_consistent": True,
        "same_formula": baseline_setup["formula"] == current_setup["formula"],
        "same_basis_fingerprint_kind": (
            baseline_setup["model"]["basis_fingerprint_kind"]
            == current_setup["model"]["basis_fingerprint_kind"]
        ),
        "same_basis_fingerprint": (
            baseline_setup["model"]["basis_fingerprint"]
            == current_setup["model"]["basis_fingerprint"]
        ),
        "same_model_dimensions": (
            baseline_setup["model"]["n_coef"] == current_setup["model"]["n_coef"]
            and baseline_setup["model"]["n_penalties"]
            == current_setup["model"]["n_penalties"]
        ),
        "same_penalty_transform_rank_fingerprint": (
            baseline_setup["model"]["frozen_contract"]
            == current_setup["model"]["frozen_contract"]
        ),
        "same_harness": (
            baseline_setup["runtime"].get("harness_sha256")
            == current_setup["runtime"].get("harness_sha256")
        ),
        "same_coefficients": (
            baseline_fit["result"]["coefficient_sha256"]
            == current_fit["result"]["coefficient_sha256"]
        ),
        "same_prediction": (
            baseline_fit["result"]["prediction_sha256"]
            == current_fit["result"]["prediction_sha256"]
        ),
        "same_score": bool(
            np.isclose(
                baseline_fit["result"]["score"],
                current_fit["result"]["score"],
                rtol=0.0,
                atol=0.0,
            )
        ),
        "numerical_parity": {
            "tolerances": {
                "strict": _tolerance_record(STRICT),
                "moderate": _tolerance_record(MODERATE),
            },
            "coefficients": _numeric_comparison(
                baseline_vectors["coefficient_values"],
                current_vectors["coefficient_values"],
            ),
            "prediction_first_128": _numeric_comparison(
                baseline_vectors["prediction_values"],
                current_vectors["prediction_values"],
            ),
            "score": _numeric_comparison(
                baseline_fit["result"]["score"], current_fit["result"]["score"]
            ),
        },
    }


def compare_pinned_r_exact(case: ScalingCase, record: dict[str, Any]) -> dict[str, Any]:
    """Compare a bounded public exact-REML case with typed pinned RBridge."""
    from tests.tolerances import MODERATE, STRICT

    if case.execution != "stream-exact":
        raise ValueError("Pinned-R parity requires execution='stream-exact'.")
    if case.n_rows > 10_000 or case.smooth_count * case.basis_dimension > 256:
        raise ValueError("Pinned-R parity is bounded to n<=10000 and m*k<=256.")

    from tests.r_bridge import RBridge, RBridgeError

    ok, reason = RBridge.check_versions()
    if not ok:
        raise RBridgeError(f"Pinned R/mgcv benchmark reference unavailable: {reason}")
    data = _make_data(case)
    formula = _formula(case)
    started = time.perf_counter()
    reference = RBridge(mode="rpy2").fit_gam(formula, data, "gaussian", "REML")
    elapsed = time.perf_counter() - started
    actual = record["fit"]["result"]
    vectors = actual["comparison_vectors"]
    fields = {
        "coefficients": _numeric_comparison(
            vectors["coefficient_values"], reference["coefficients"]
        ),
        "prediction_first_128": _numeric_comparison(
            vectors["prediction_values"], reference["fitted_values"][:128]
        ),
        "smoothing": _numeric_comparison(
            actual["smoothing_params"], reference["smoothing_params"]
        ),
        "score": _numeric_comparison(actual["score"], reference["reml_score"]),
        "deviance": _numeric_comparison(actual["deviance"], reference["deviance"]),
        "scale": _numeric_comparison(actual["scale"], reference["scale"]),
        "score_scale": _numeric_comparison(
            actual["score_scale"], reference["reml_scale"]
        ),
        "edf_total": _numeric_comparison(actual["edf_total"], reference["edf_total"]),
    }
    within_run_consistent = bool(
        record.get("within_run_consistency", {}).get("valid", False)
    )
    reference_converged = bool(
        reference["inner_converged"] and reference["outer_converged"]
    )
    valid_pair = (
        bool(actual["converged"]) and within_run_consistent and reference_converged
    )
    return {
        "reference": "R 4.5.2 / mgcv 1.9-3 via RBridge(mode='rpy2')",
        "formula": formula,
        "family": "gaussian",
        "method": "REML",
        "r_fit_and_extraction_s": elapsed,
        "basis_status": (
            "same cubic-regression formula; direct prepared-basis equality is "
            "not established by this fit-level comparison"
        ),
        "tolerances": {
            "strict": _tolerance_record(STRICT),
            "moderate": _tolerance_record(MODERATE),
        },
        "fields": fields,
        "jaxgam_converged": bool(actual["converged"]),
        "within_run_consistent": within_run_consistent,
        "r_inner_converged": bool(reference["inner_converged"]),
        "r_outer_converged": bool(reference["outer_converged"]),
        "r_outer_status": reference["outer_status"],
        "strict_complete": valid_pair
        and all(field["strict_allclose"] for field in fields.values()),
        "moderate_numerical_match": valid_pair
        and all(field["moderate_allclose"] for field in fields.values()),
        "moderate_acceptance_status": (
            "diagnostic_only; requires four recorded source/numerical correction "
            "passes before MODERATE can be an acceptance gate"
        ),
    }


def compare_memory_growth(
    smaller: dict[str, Any], larger: dict[str, Any]
) -> dict[str, Any]:
    """Compare isolated streamed runs at fixed batch and coefficient shape."""
    small_fit, large_fit = smaller["fit"], larger["fit"]
    small_case, large_case = small_fit["case"], large_fit["case"]
    same_shape_controls = all(
        small_case[field] == large_case[field]
        for field in ("execution", "basis_dimension", "smooth_count", "batch_rows")
    )
    if (
        not same_shape_controls
        or small_case["execution"] not in {"stream", "stream-qr", "stream-exact"}
        or small_case["n_rows"] >= large_case["n_rows"]
        or not smaller["within_run_consistency"]["valid"]
        or not larger["within_run_consistency"]["valid"]
        or small_fit["result"]["n_coef"] != large_fit["result"]["n_coef"]
    ):
        raise ValueError("Memory pair requires valid streamed runs at fixed B/p.")

    def observation(record: dict[str, Any]) -> dict[str, Any]:
        fit = record["fit"]
        memory = fit["memory"]
        before = memory["rss_current_before_bytes"]
        after = memory["rss_high_water_after_cold_bytes"]
        return {
            "n_rows": fit["case"]["n_rows"],
            "input_bytes": memory["input_bytes"],
            "runtime_and_input_baseline_rss_bytes": before,
            "cold_peak_rss_bytes": after,
            "fit_rss_above_baseline_bytes": (
                None if before is None or after is None else max(0, after - before)
            ),
            "known_workspace_bytes": fit["result"]["known_workspace_bytes"],
            "cold_fit_scan_calls": fit["result"]["source_scans"]["cold_fit_delta"],
            "cold_fit_batch_bytes": fit["result"]["source_scans"][
                "cold_fit_batch_bytes"
            ],
        }

    small = observation(smaller)
    large = observation(larger)
    return {
        "smaller": small,
        "larger": large,
        "known_workspace_plateau": (
            small["known_workspace_bytes"] is not None
            and small["known_workspace_bytes"] == large["known_workspace_bytes"]
        ),
        "rss_scope": (
            "each isolated worker's cold peak minus its measured current RSS "
            "after input generation and runtime import; sampling/allocator noise "
            "remains and no single-pair RSS pass threshold is inferred"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--execution",
        choices=("dense", "stream", "stream-qr", "stream-exact", "efs"),
        default="dense",
    )
    parser.add_argument("--batch-rows", type=int, default=1_024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--smooth-count",
        type=int,
        default=1,
        help="number of ordered additive cubic-regression smooth terms",
    )
    parser.add_argument(
        "--result-mode", choices=("full", "inference", "prediction"), default="full"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--child", choices=("setup", "fit"), help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--baseline-checkout", type=Path)
    parser.add_argument("--current-checkout", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--r-parity",
        action="store_true",
        help="compare a bounded public stream-exact case with pinned RBridge",
    )
    parser.add_argument(
        "--memory-pair-n",
        nargs=2,
        type=int,
        metavar=("SMALL", "LARGE"),
        help="run two isolated streamed cases with fixed batch/p",
    )
    args = parser.parse_args()
    root = args.checkout or Path(__file__).resolve().parents[1]
    current_root = args.current_checkout or root
    # ``uv run python scripts/...`` starts with scripts/ on sys.path. Select
    # the requested checkout before any JaxGAM import, even when the active
    # uv environment was installed editable from another worktree.
    sys.path.insert(0, str(current_root.resolve()))
    _check_reference_imports(current_root, r_parity=args.r_parity)

    if args.smoke:
        print(json.dumps(run_public_smoke(current_root), sort_keys=True))
        return

    if args.child is not None:
        if args.case is None:
            raise ValueError("Child mode requires a serialized case.")
        child_case = ScalingCase(**json.loads(args.case))
        record = (
            run_setup(child_case, current_root)
            if args.child == "setup"
            else run_case(child_case, current_root)
        )
        print(json.dumps(record, sort_keys=True))
        return

    case = ScalingCase(
        n_rows=args.n,
        basis_dimension=args.k,
        execution=args.execution,
        batch_rows=args.batch_rows,
        repeats=args.repeats,
        result_mode=args.result_mode,
        smooth_count=args.smooth_count,
    )
    if args.memory_pair_n is not None:
        if args.r_parity or args.baseline_checkout is not None:
            raise ValueError(
                "Memory pairs cannot combine with R parity or checkout parity."
            )
        small_n, large_n = args.memory_pair_n
        if small_n >= large_n:
            raise ValueError("Memory pair needs SMALL < LARGE.")
        small = run_isolated(replace(case, n_rows=small_n), current_root)
        large = run_isolated(replace(case, n_rows=large_n), current_root)
        payload = json.dumps(
            {
                "memory_pair": {
                    "smaller": small,
                    "larger": large,
                    "comparison": compare_memory_growth(small, large),
                }
            },
            sort_keys=True,
        )
        if args.output is not None:
            args.output.write_text(f"{payload}\n")
        print(payload)
        return
    current = run_isolated(case, current_root)
    payload_value: dict[str, Any] = {"current": current}
    if args.r_parity:
        payload_value["pinned_r_parity"] = compare_pinned_r_exact(case, current)
    if args.baseline_checkout is not None:
        try:
            baseline = run_isolated(case, args.baseline_checkout)
        except (OSError, ChildRunError, json.JSONDecodeError) as error:
            payload_value["baseline"] = {
                "available": False,
                "checkout": str(args.baseline_checkout),
                "reason": str(error),
            }
            if isinstance(error, ChildRunError):
                payload_value["baseline"]["failed_child"] = error.child_mode
                payload_value["baseline"]["child_stdout"] = error.stdout
                payload_value["baseline"]["child_stderr"] = error.stderr
            payload_value["parity"] = {
                "status": "unavailable",
                "reason": (
                    "The requested baseline cannot run this exact route; no "
                    "fallback route or cross-strategy comparison was used."
                ),
            }
        else:
            payload_value["baseline"] = baseline
            payload_value["parity"] = compare_checkouts(baseline, current)
    payload = json.dumps(payload_value, sort_keys=True)
    if args.output is not None:
        args.output.write_text(f"{payload}\n")
    print(payload)


if __name__ == "__main__":
    main()
