"""Reproducible dense EFS benchmark against Newton and pinned mgcv.

The driver saves one deterministic input before launching isolated workers.
Python Newton and EFS workers each get their own temporary JAX compilation
cache.  Pinned R fits the same CSV/model with ``optimizer="efs"`` and the same
EFS initial smoothing parameters and controls.

Run the bounded representative case with ``make benchmark-efs`` or the small
end-to-end gate with ``make benchmark-efs-smoke``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PINNED_R_VERSION = "4.5.2"
_PINNED_MGCV_VERSION = "1.9.3"
_PINNED_MGCV_COMMIT = "fb7e8e718377513e78ba6c6bf7e60757fc6a32a9"
_TOLERANCES = {"STRICT": (1e-10, 1e-12), "MODERATE": (1e-4, 1e-6)}
_APPROVED_SMOKE_SHA256 = (
    "8a49df584d593df11ed6ffc44e66ec72ab37f2352ab37b353e1c42c4232a26b9"
)
_DEFAULT_EFS_CONTROL = {
    "outer_limit": 200,
    "log_lambda_max": 15.0,
    "score_tolerance": 0.1,
    "pirls_tolerance": 1e-7,
    "pirls_max_iter": 200,
    "history_limit": 200,
}
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_input(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["data_path"])
    if _sha256(path) != config["data_sha256"]:
        raise ValueError("saved benchmark input hash changed")
    return pd.read_csv(path, float_precision="round_trip")


def _fit_validity(result: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if result.get("converged") is not True:
        reasons.append("fit did not converge")
    reasons.extend(
        f"invalid {field}"
        for field in ("covariance_finite", "linear_predictor_finite")
        if result.get(field, True) is not True
    )
    for field in (
        "coefficients",
        "fitted_values",
        "smoothing_params",
        "edf",
        "deviance",
        "score",
        "scale",
    ):
        values = np.asarray(result[field], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            reasons.append(f"nonfinite {field}")
    if not np.all(np.asarray(result["smoothing_params"]) > 0):
        reasons.append("nonpositive smoothing parameter")
    if not result["scale"] > 0:
        reasons.append("nonpositive scale")
    if not result["deviance"] >= 0:
        reasons.append("negative deviance")
    return {"passed": not reasons, "reasons": reasons}


def _source_commit() -> str | None:
    explicit = os.environ.get("JAXGAM_BENCHMARK_SOURCE_COMMIT")
    if explicit:
        return explicit
    if shutil.which("git") is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return None


def _model_formula(n_smooths: int, basis_dimension: int) -> str:
    terms = " + ".join(
        f"s(x{index}, k={basis_dimension}, bs='cr')"
        for index in range(1, n_smooths + 1)
    )
    return f"y ~ {terms}"


def _make_data(n: int, n_smooths: int, seed: int) -> pd.DataFrame:
    """Reuse the general benchmark response convention for a many-m model."""
    # Imported lazily so worker cache variables are installed before JAXGAM is
    # imported by benchmark_vs_r at module scope.
    from scripts.benchmark_vs_r import _make_response

    rng = np.random.default_rng(seed)
    values: dict[str, np.ndarray] = {}
    eta = np.full(n, 0.35, dtype=np.float64)
    normalizer = np.sqrt(float(n_smooths))
    for index in range(1, n_smooths + 1):
        x = rng.uniform(0.0, 1.0, n)
        values[f"x{index}"] = x
        eta += (
            0.7 * np.sin(2.0 * np.pi * x * (1.0 + index % 3))
            + 0.2 * np.cos(2.0 * np.pi * x)
        ) / normalizer
    values["y"] = _make_response(eta, "poisson", rng)
    return pd.DataFrame(values)


def _thread_environment(threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    for name in _THREAD_VARIABLES:
        environment[name] = str(threads)
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return environment


def _hardware_metadata() -> dict[str, Any]:
    cpu_model = None
    total_memory_bytes = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                total_memory_bytes = int(line.split()[1]) * 1024
                break
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_model": cpu_model,
        "cpu_model_unavailable_reason": (
            None if cpu_model else "container /proc/cpuinfo exposes no model name"
        ),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory_bytes,
        "container_image": os.environ.get("JAXGAM_BENCHMARK_IMAGE_ID"),
        "container_image_unavailable_reason": (
            None
            if os.environ.get("JAXGAM_BENCHMARK_IMAGE_ID")
            else "set JAXGAM_BENCHMARK_IMAGE_ID for a containerized run"
        ),
    }


def _rss_bytes(pid: int) -> int | None:
    """Read summed Linux process-tree RSS, or report unsupported platforms."""
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return None
    total = 0
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                total += int(line.split()[1]) * 1024
                break
        children = Path(f"/proc/{pid}/task/{pid}/children")
        if children.exists():
            for child in children.read_text().split():
                child_rss = _rss_bytes(int(child))
                if child_rss is not None:
                    total += child_rss
    except (FileNotFoundError, ProcessLookupError):
        return None
    return total


def _run_worker(
    command: list[str], environment: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    peak_rss: int | None = None
    # Workers return observation-sized fitted vectors. Files let them finish
    # while the parent samples RSS, even when stdout or stderr exceeds the
    # capacity of an unread pipe.
    with (
        tempfile.TemporaryFile(mode="w+") as output,
        tempfile.TemporaryFile(mode="w+") as errors,
    ):
        process = subprocess.Popen(
            command, stdout=output, stderr=errors, text=True, env=environment
        )
        while process.poll() is None:
            current = _rss_bytes(process.pid)
            if current is not None:
                peak_rss = current if peak_rss is None else max(peak_rss, current)
            time.sleep(0.01)
        output.seek(0)
        errors.seek(0)
        stdout, stderr = output.read(), errors.read()
    if process.returncode != 0:
        raise RuntimeError(
            f"benchmark worker failed ({process.returncode}): {' '.join(command)}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    payload = json.loads(stdout)
    memory = {
        "peak_process_tree_rss_bytes": peak_rss,
        "peak_process_tree_rss_method": (
            "10ms Linux /proc VmRSS polling" if peak_rss is not None else None
        ),
        "peak_process_tree_rss_unavailable_reason": (
            None if peak_rss is not None else "Linux /proc VmRSS is unavailable"
        ),
    }
    return payload, memory


def _sync_result(result: Any) -> None:
    import jax

    for value in (
        result.coefficients,
        result.fitted_values,
        result.smoothing_params,
        result.Vp,
        result.score,
    ):
        jax.block_until_ready(value)


def _result_payload(result: Any) -> dict[str, Any]:
    diagnostics = result.optimizer_diagnostics
    efs_diagnostics = None
    if diagnostics is not None:
        efs_diagnostics = {
            "reference_profile": diagnostics.reference_profile,
            "trace_method": diagnostics.trace_method,
            "step_policy": diagnostics.step_policy,
            "stop_reason": diagnostics.stop_reason,
            "outer_iterations": diagnostics.outer_iterations,
            "inner_iterations": diagnostics.inner_iterations,
            "theta_iterations": diagnostics.theta_iterations,
            "accepted_score_count": len(diagnostics.accepted_score_history),
            "multiplier": diagnostics.multiplier,
            "max_proposed_movement": diagnostics.max_proposed_movement,
            "max_accepted_movement": diagnostics.max_accepted_movement,
            "numerator_clamp_count": diagnostics.numerator_clamp_count,
            "ratio_replacement_count": diagnostics.ratio_replacement_count,
            "log_lambda_cap_count": diagnostics.log_lambda_cap_count,
            "invalid_fit_seen": diagnostics.invalid_fit_seen,
            "stabilized_solve_seen": diagnostics.stabilized_solve_seen,
        }
    return {
        "coefficients": np.asarray(result.coefficients).tolist(),
        "fitted_values": np.asarray(result.fitted_values).tolist(),
        "smoothing_params": np.asarray(result.smoothing_params).tolist(),
        "edf": np.asarray(result.edf).tolist(),
        "deviance": float(result.deviance),
        "score": float(result.score),
        "scale": float(result.scale),
        "outer_iterations": int(result.n_iter),
        "converged": bool(result.converged),
        "convergence_info": result.convergence_info,
        "covariance_finite": bool(np.all(np.isfinite(result.Vp))),
        "linear_predictor_finite": bool(np.all(np.isfinite(result.linear_predictor))),
        "efs_diagnostics": efs_diagnostics,
    }


def _device_memory() -> dict[str, Any]:
    import jax

    records = []
    for device in jax.devices():
        stats = device.memory_stats()
        if stats is not None:
            records.append({"device": str(device), "stats": dict(stats)})
    return {
        "device_memory": records or None,
        "device_memory_unavailable_reason": (
            None if records else "JAX backend did not expose device memory_stats"
        ),
    }


def _python_worker(config_path: Path, optimizer: str) -> dict[str, Any]:
    import inspect

    import jax

    from jaxgam import GAM, EFSControl, FitControl
    from jaxgam.fitting.newton import _DEFAULT_CONV_TOL, newton_optimize
    from jaxgam.fitting.pirls import pirls_loop

    config = json.loads(config_path.read_text())
    data = _load_input(config)
    efs = EFSControl(**config["efs_control"])
    fit_control = FitControl(efs=efs)
    newton_defaults = inspect.signature(newton_optimize).parameters
    newton_control = {
        "policy": "unmodified public Poisson Newton defaults at recorded source",
        "outer_limit": newton_defaults["max_iter"].default,
        "convergence_tolerance": _DEFAULT_CONV_TOL,
        "max_step": newton_defaults["max_step"].default,
        "log_lambda_max": newton_defaults["lsp_max"].default,
        "pirls_tolerance": min(_DEFAULT_CONV_TOL / 100.0, 1e-8),
        "pirls_max_iter": inspect.signature(pirls_loop).parameters["max_iter"].default,
    }

    def fit_once():
        model = GAM(
            config["formula"],
            family="poisson",
            method="REML",
            optimizer=optimizer,
            control=fit_control,
        )
        fit = model.fit(data)
        _sync_result(fit)
        return fit

    start = time.perf_counter()
    result = fit_once()
    cold = time.perf_counter() - start
    validity = [_fit_validity(_result_payload(result))]
    warm_times = []
    for _ in range(config["warm_repeats"]):
        start = time.perf_counter()
        result = fit_once()
        warm_times.append(time.perf_counter() - start)
        validity.append(_fit_validity(_result_payload(result)))

    return {
        "implementation": f"jaxgam_{optimizer}",
        "controls": {
            "fit_control": asdict(fit_control),
            "optimizer": asdict(efs) if optimizer == "efs" else newton_control,
        },
        "timing": {
            "cold_complete_fit_seconds": cold,
            "cold_includes": "setup, compilation, fitting, and Phase-3 materialization",
            "cold_compile_seconds": None,
            "cold_compile_seconds_unavailable_reason": (
                "the public complete-fit API does not expose compilation separately "
                "from first execution"
            ),
            "warm_complete_fit_seconds": warm_times,
            "warm_median_seconds": median(warm_times),
            "synchronized": True,
        },
        "result": _result_payload(result),
        "measured_fit_validity": validity,
        "input_sha256": _sha256(Path(config["data_path"])),
        "input_parser": "pandas float_precision=round_trip",
        "runtime": {
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "jaxgam_version": importlib.metadata.version("jaxgam"),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "compilation_cache_policy": "isolated temporary directory per optimizer",
            "thread_environment": {
                name: os.environ.get(name) for name in _THREAD_VARIABLES
            },
        },
        "cost_counts": {
            "coefficient_factorizations": None,
            "coefficient_factorizations_unavailable_reason": (
                "public fit diagnostics do not count initial, trial, final, "
                "and retry factorizations"
            ),
            "rhs_solves": None,
            "rhs_solves_unavailable_reason": (
                "dense trace/root solve counts are not exposed by the public "
                "fit diagnostics"
            ),
            "score_evaluations": None,
            "score_evaluations_unavailable_reason": (
                "bounded accepted-score history is not a count of all trial evaluations"
            ),
            "dense_source_scans": None,
            "dense_source_scans_unavailable_reason": (
                "not applicable to the in-memory dense FittingData route"
            ),
            "penalty_derivative_seconds": None,
            "penalty_derivative_seconds_unavailable_reason": (
                "public EFS diagnostics do not expose kernel stage timings"
            ),
            "score_evaluation_seconds": None,
            "score_evaluation_seconds_unavailable_reason": (
                "public optimizer results do not expose score stage timings"
            ),
        },
        **_device_memory(),
    }


def _r_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _r_write_csv(expression: str, path: Path) -> str:
    return (
        f"write.csv(data.frame(value={expression}), {_r_quote(str(path))}, "
        "row.names=FALSE)"
    )


def _r_write_lines(expression: str, path: Path) -> str:
    return f"writeLines({expression}, {_r_quote(str(path))})"


def _r_worker(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    _load_input(config)
    with tempfile.TemporaryDirectory(prefix="jaxgam-efs-r-") as directory:
        root = Path(directory)
        output = root / "output"
        output.mkdir()
        initial = ", ".join(repr(float(value)) for value in config["efs_initial_sp"])
        control = config["efs_control"]
        script = root / "benchmark.R"
        r_version = _r_quote(_PINNED_R_VERSION)
        mgcv_version = _r_quote(_PINNED_MGCV_VERSION)
        repeats = config["warm_repeats"]
        script.write_text(
            "\n".join(
                [
                    "library(mgcv)",
                    f"stopifnot(as.character(getRversion()) == {r_version})",
                    (
                        "stopifnot(as.character(packageVersion('mgcv')) == "
                        f"{mgcv_version})"
                    ),
                    f"data <- read.csv({_r_quote(config['data_path'])})",
                    f"formula <- as.formula({_r_quote(config['formula'])})",
                    f"initial.sp <- c({initial})",
                    (
                        "control <- gam.control("
                        f"epsilon={control['pirls_tolerance']!r}, "
                        f"maxit={control['pirls_max_iter']}, "
                        f"nthreads={config['threads']}, "
                        f"efs.lspmax={control['log_lambda_max']!r}, "
                        f"efs.tol={control['score_tolerance']!r})"
                    ),
                    (
                        "fit.once <- function() gam(formula, data=data, "
                        "family=poisson(), "
                        "method='REML', optimizer='efs', scale=1, control=control, "
                        "in.out=list(sp=initial.sp, scale=1))"
                    ),
                    "cold <- system.time(model <- fit.once())[['elapsed']]",
                    (
                        "fit.valid <- function(model) isTRUE(model$converged) && "
                        "identical(model$outer.info$conv, 'full convergence') && "
                        "all(is.finite(c(coef(model), fitted(model), model$sp, "
                        "model$edf, model$Vp, model$linear.predictors, "
                        "deviance(model), model$gcv.ubre, model$scale)))"
                    ),
                    "valid <- fit.valid(model)",
                    f"warm <- numeric({repeats})",
                    (
                        f"for (i in seq_len({repeats})) {{ warm[i] <- "
                        "system.time(model <- fit.once())[['elapsed']]; "
                        "valid <- c(valid, fit.valid(model)) }"
                    ),
                    "summary.model <- summary(model)",
                    _r_write_csv(
                        "as.numeric(coef(model))", output / "coefficients.csv"
                    ),
                    _r_write_csv("as.numeric(fitted(model))", output / "fitted.csv"),
                    _r_write_csv("as.numeric(model$sp)", output / "sp.csv"),
                    _r_write_csv("as.numeric(summary.model$edf)", output / "edf.csv"),
                    _r_write_csv("warm", output / "warm.csv"),
                    _r_write_csv("as.integer(valid)", output / "valid.csv"),
                    _r_write_lines(
                        "as.character(isTRUE(model$converged))",
                        output / "inner_converged.txt",
                    ),
                    _r_write_lines("format(cold, digits=17)", output / "cold.txt"),
                    _r_write_lines(
                        "format(deviance(model), digits=17)", output / "deviance.txt"
                    ),
                    _r_write_lines(
                        "format(model$gcv.ubre, digits=17)", output / "score.txt"
                    ),
                    _r_write_lines(
                        "format(model$scale, digits=17)", output / "scale.txt"
                    ),
                    _r_write_lines(
                        "as.character(model$outer.info$iter)",
                        output / "iterations.txt",
                    ),
                    _r_write_lines("model$outer.info$conv", output / "convergence.txt"),
                    _r_write_lines(
                        "capture.output(sessionInfo())", output / "session.txt"
                    ),
                ]
            )
            + "\n"
        )
        completed = subprocess.run(
            ["Rscript", str(script)], capture_output=True, text=True, timeout=600
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"pinned R benchmark failed ({completed.returncode}):\n"
                f"{completed.stderr}"
            )

        def vector(name: str) -> list[float]:
            return (
                pd.read_csv(output / name, float_precision="round_trip")["value"]
                .astype(float)
                .tolist()
            )

        def scalar(name: str) -> float:
            return float((output / name).read_text().strip())

        warm = vector("warm.csv")
        return {
            "implementation": "pinned_r_mgcv_efs",
            "input_sha256": _sha256(Path(config["data_path"])),
            "input_parser": "R read.csv on identical saved decimal CSV",
            "measured_fit_validity": [
                {
                    "passed": bool(value),
                    "reasons": []
                    if value
                    else ["R fit did not converge or contains nonfinite state"],
                }
                for value in vector("valid.csv")
            ],
            "timing": {
                "cold_complete_fit_seconds": scalar("cold.txt"),
                "cold_includes": "first R complete fit; R has no JIT compilation",
                "cold_compile_seconds": None,
                "cold_compile_seconds_unavailable_reason": (
                    "not applicable: pinned R EFS does not JIT compile"
                ),
                "warm_complete_fit_seconds": warm,
                "warm_median_seconds": median(warm),
                "synchronized": True,
            },
            "result": {
                "coefficients": vector("coefficients.csv"),
                "fitted_values": vector("fitted.csv"),
                "smoothing_params": vector("sp.csv"),
                "edf": vector("edf.csv"),
                "deviance": scalar("deviance.txt"),
                "score": scalar("score.txt"),
                "scale": scalar("scale.txt"),
                "outer_iterations": int(scalar("iterations.txt")),
                "converged": (output / "convergence.txt").read_text().strip()
                == "full convergence"
                and (output / "inner_converged.txt").read_text().strip() == "TRUE",
                "convergence_info": (output / "convergence.txt").read_text().strip(),
            },
            "runtime": {
                "r_version": _PINNED_R_VERSION,
                "mgcv_version": _PINNED_MGCV_VERSION,
                "mgcv_source_commit": _PINNED_MGCV_COMMIT,
                "session_info": (output / "session.txt").read_text().splitlines(),
                "thread_environment": {
                    name: os.environ.get(name) for name in _THREAD_VARIABLES
                },
            },
            "cost_counts": {
                "coefficient_factorizations": None,
                "coefficient_factorizations_unavailable_reason": (
                    "pinned mgcv outer.info does not expose gam.fit3 "
                    "factorization count"
                ),
                "rhs_solves": None,
                "rhs_solves_unavailable_reason": (
                    "pinned mgcv result does not expose covariance-root solve count"
                ),
                "score_evaluations": None,
                "score_evaluations_unavailable_reason": (
                    "outer.info score history excludes unretained trial-fit details"
                ),
                "dense_source_scans": None,
                "dense_source_scans_unavailable_reason": (
                    "not applicable to mgcv's in-memory dense model matrix"
                ),
                "penalty_derivative_seconds": None,
                "penalty_derivative_seconds_unavailable_reason": (
                    "pinned mgcv result does not expose efsudr stage timings"
                ),
                "score_evaluation_seconds": None,
                "score_evaluation_seconds_unavailable_reason": (
                    "pinned mgcv result does not expose score stage timings"
                ),
            },
            "device_memory": None,
            "device_memory_unavailable_reason": (
                "R exposes no accelerator memory metric"
            ),
        }


def _comparison(
    actual: Any, expected: Any, tolerance: str = "STRICT"
) -> dict[str, Any]:
    rtol, atol = _TOLERANCES[tolerance]
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if (
        left.shape != right.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        return {
            "passed": False,
            "actual_shape": list(left.shape),
            "expected_shape": list(right.shape),
            "max_absolute_error": None,
            "max_relative_error": None,
        }
    absolute = np.abs(left - right)
    denominator = np.maximum(np.abs(right), atol)
    return {
        "passed": bool(np.allclose(left, right, rtol=rtol, atol=atol)),
        "actual_shape": list(left.shape),
        "expected_shape": list(right.shape),
        "max_absolute_error": float(np.max(absolute, initial=0.0)),
        "max_relative_error": float(np.max(absolute / denominator, initial=0.0)),
    }


def _numerical_comparisons(
    efs: dict[str, Any], reference: dict[str, Any], tolerance: str = "STRICT"
) -> dict[str, Any]:
    fields = (
        "coefficients",
        "fitted_values",
        "smoothing_params",
        "edf",
        "deviance",
        "score",
        "scale",
    )
    comparisons = {
        name: _comparison(efs["result"][name], reference["result"][name], tolerance)
        for name in fields
    }
    comparisons["outer_iterations"] = {
        "passed": efs["result"]["outer_iterations"]
        == reference["result"]["outer_iterations"],
        "jaxgam": efs["result"]["outer_iterations"],
        "pinned_r": reference["result"]["outer_iterations"],
    }
    validity = {
        "jaxgam": _fit_validity(efs["result"]),
        "pinned_r": _fit_validity(reference["result"]),
    }
    rtol, atol = _TOLERANCES[tolerance]
    return {
        "tolerance": {"name": tolerance, "rtol": rtol, "atol": atol},
        "fit_validity": validity,
        "fields": comparisons,
        "passed": all(value["passed"] for value in comparisons.values())
        and all(value["passed"] for value in validity.values()),
    }


def _prepare_config(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    n = 128 if args.smoke else args.rows
    n_smooths = 3 if args.smoke else args.smooths
    basis_dimension = 5 if args.smoke else args.basis_dimension
    data = _make_data(n, n_smooths, args.seed)
    data_path = (output / "input.csv").resolve()
    data.to_csv(data_path, index=False, float_format="%.17g", lineterminator="\n")
    data = pd.read_csv(data_path, float_precision="round_trip")
    formula = _model_formula(n_smooths, basis_dimension)

    from jaxgam.execution.efs import efs_initial_log_lambda
    from jaxgam.families.registry import get_family
    from jaxgam.fitting.data import FittingData
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    family = get_family("poisson")
    setup = ModelSetup.build(parse_formula(formula), data)
    efs_sp = np.exp(np.asarray(efs_initial_log_lambda(setup, family)))
    fitting = FittingData.from_setup(setup, family)
    newton_sp = np.exp(np.asarray(fitting.log_lambda_init))
    config = {
        "data_path": str(data_path),
        "data_sha256": _sha256(data_path),
        "initialization_input_parser": "pandas float_precision=round_trip on saved CSV",
        "formula": formula,
        "family": "poisson(log)",
        "seed": args.seed,
        "n": n,
        "p": int(setup.X.shape[1]),
        "m": int(setup.penalties.n_penalties),
        "rank_x": int(np.linalg.matrix_rank(setup.X)),
        "basis": {"type": "cr", "dimension_per_smooth": basis_dimension},
        "warm_repeats": args.warm_repeats,
        "threads": args.threads,
        "efs_control": _DEFAULT_EFS_CONTROL.copy(),
        "initial_smoothing_policy": {
            "jaxgam_newton": {
                "policy": "public FittingData prior-weight crossproduct initializer",
                "natural_scale": newton_sp.tolist(),
            },
            "jaxgam_efs_and_pinned_r": {
                "policy": "public efs_initial_log_lambda; identical R in.out sp",
                "natural_scale": efs_sp.tolist(),
                "scale": 1.0,
            },
        },
        "efs_initial_sp": efs_sp.tolist(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def _comparison_selection(config: dict[str, Any], tolerance: str) -> dict[str, Any]:
    """Restrict reviewed MODERATE acceptance to the saved default smoke case."""
    if tolerance == "STRICT":
        return {"name": tolerance, "reviewed_scope": None}
    approved = {
        "data_sha256": _APPROVED_SMOKE_SHA256,
        "formula": _model_formula(3, 5),
        "family": "poisson(log)",
        "seed": 20260910,
        "n": 128,
        "p": 13,
        "m": 3,
        "efs_control": _DEFAULT_EFS_CONTROL,
    }
    if tolerance != "MODERATE" or any(
        config.get(name) != value for name, value in approved.items()
    ):
        raise ValueError(
            "MODERATE comparison is approved only for the exact saved smoke "
            "input, model, and default controls; other cases require STRICT"
        )
    return {
        "name": tolerance,
        "reviewed_scope": approved,
        "review_record": ("docs/scale_jaxgam/efs_benchmark_validation_2026-09-17.md"),
    }


def _driver(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jaxgam-efs-driver-cache-") as cache:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache
        config = _prepare_config(args, output)
    selection = _comparison_selection(config, args.comparison_tolerance)

    environment = _thread_environment(args.threads)
    methods: dict[str, Any] = {}
    for optimizer in ("newton", "efs"):
        with tempfile.TemporaryDirectory(
            prefix=f"jaxgam-efs-{optimizer}-cache-"
        ) as cache:
            worker_environment = environment.copy()
            worker_environment["JAX_COMPILATION_CACHE_DIR"] = cache
            payload, memory = _run_worker(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    optimizer,
                    "--config",
                    str(output / "config.json"),
                ],
                worker_environment,
            )
        payload["memory"] = memory
        methods[optimizer] = payload

    reference = None
    parity = None
    if not args.python_only:
        payload, memory = _run_worker(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "r",
                "--config",
                str(output / "config.json"),
            ],
            environment,
        )
        payload["memory"] = memory
        reference = payload
        parity = {
            name: _numerical_comparisons(methods["efs"], reference, name)
            for name in _TOLERANCES
        }

    source_commit = _source_commit()
    report = {
        "schema_version": 2,
        "scope": (
            "bounded dense many-penalty observation; no general speedup or "
            "large-p claim"
        ),
        "source": {
            "git_commit": source_commit,
            "git_commit_unavailable_reason": (
                None
                if source_commit is not None
                else "set JAXGAM_BENCHMARK_SOURCE_COMMIT in images without git"
            ),
            "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "host": {
            **_hardware_metadata(),
            "thread_environment": {
                name: environment.get(name) for name in _THREAD_VARIABLES
            },
        },
        "input": config,
        "methods": methods,
        "newton_vs_efs": {
            "purpose": (
                "different-optimizer residuals; not an EFS implementation parity gate"
            ),
            "fields": {
                name: _comparison(
                    methods["newton"]["result"][name], methods["efs"]["result"][name]
                )
                for name in (
                    "coefficients",
                    "fitted_values",
                    "smoothing_params",
                    "edf",
                    "deviance",
                    "score",
                    "scale",
                )
            },
        },
        "pinned_r_efs": reference,
        "efs_vs_pinned_r": parity,
        "selected_comparison_tolerance": selection,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report_path)
    valid = all(
        _fit_validity(method["result"])["passed"]
        and all(item["passed"] for item in method["measured_fit_validity"])
        for method in methods.values()
    )
    if reference is not None:
        valid &= all(item["passed"] for item in reference["measured_fit_validity"])
    if not valid or (
        parity is not None and not parity[args.comparison_tolerance]["passed"]
    ):
        print(
            "benchmark fit validity or selected "
            f"{args.comparison_tolerance} EFS/pinned-R comparison failed",
            file=sys.stderr,
        )
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/jaxgam-efs-benchmark")
    )
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--smooths", type=int, default=10)
    parser.add_argument("--basis-dimension", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument(
        "--comparison-tolerance",
        choices=tuple(_TOLERANCES),
        default="STRICT",
        help="MODERATE is reviewed only for the exact default saved smoke case",
    )
    parser.add_argument(
        "--worker", choices=("newton", "efs", "r"), help=argparse.SUPPRESS
    )
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker is not None:
        if args.config is None:
            raise ValueError("--config is required for a benchmark worker")
        if args.worker == "r":
            payload = _r_worker(args.config)
        else:
            payload = _python_worker(args.config, args.worker)
        print(json.dumps(payload))
        return 0
    for name in ("rows", "smooths", "basis_dimension", "warm_repeats", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if shutil.which("Rscript") is None and not args.python_only:
        raise RuntimeError("Rscript is required unless --python-only is selected")
    return _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
