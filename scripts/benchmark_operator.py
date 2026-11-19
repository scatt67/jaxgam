"""Measure fixed-sp operator fits with indexed factors and actual source passes.

Run one size per process, for example::

    make benchmark-operator OUTPUT=/tmp/jaxgam-operator-benchmark LEVELS=10000
    make benchmark-operator OUTPUT=/tmp/jaxgam-operator-benchmark LEVELS=100000

The report separates the caller-owned input from the operator's prospective
workspace bound. It makes no dense-fit speedup claim at high cardinality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np
import pandas as pd

import jaxgam
from jaxgam import GAM
from jaxgam.control import FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests.tolerances import STRICT


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024 * value)


def _frame(levels: int, rows_per_level: int) -> pd.DataFrame:
    n = levels * rows_per_level
    row = np.arange(n)
    groups = np.tile(np.arange(levels), rows_per_level)
    labels = [f"g{index}" for index in range(levels)]
    return pd.DataFrame(
        {
            "y": 1.0 + 0.1 * np.sin(row * 0.13) + 0.02 * np.cos(groups * 0.07),
            "g": pd.Categorical.from_codes(groups, categories=labels),
        }
    )


def _git_revision() -> dict[str, str | bool | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if revision.returncode:
        return {
            "commit": os.environ.get("JAXGAM_SOURCE_COMMIT"),
            "dirty": None,
            "provenance_status": "caller-supplied-no-git"
            if os.environ.get("JAXGAM_SOURCE_COMMIT")
            else "unavailable-no-git",
        }
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "provenance_status": "git-worktree",
    }


def _fit(frame: pd.DataFrame, batch_rows: int, budget: int) -> tuple[object, float]:
    source = DataFrameRowSource(frame, response="y")
    control = FitControl(
        execution="operator",
        linear_solver="pcg",
        batch_rows=batch_rows,
        memory_budget_bytes=budget,
    )
    start = time.perf_counter()
    fitted = GAM("y~s(g,bs='re')", sp=[1.0], control=control).fit(
        source, result="prediction"
    )
    seconds = time.perf_counter() - start
    if not fitted.converged:
        raise RuntimeError(
            f"Operator benchmark fit did not converge: {fitted.convergence_info}"
        )
    if fitted.operator_diagnostics.peak_bound_bytes > budget:
        raise AssertionError("Operator peak bound exceeded its declared budget.")
    return fitted, seconds


def _direct_small_check(frame: pd.DataFrame, coefficients: np.ndarray) -> float:
    dense = ModelSetup.build(parse_formula("y~s(g,bs='re')"), frame)
    H = dense.X.T @ dense.X + dense.penalties.materialize(np.zeros(1))
    expected = np.linalg.solve(H, dense.X.T @ frame.y.to_numpy())
    np.testing.assert_allclose(
        coefficients, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    return float(np.max(np.abs(coefficients - expected)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=10_000)
    parser.add_argument("--rows-per-level", type=int, default=2)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--budget-mib", type=int, default=512)
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/jaxgam-operator-benchmark")
    )
    options = parser.parse_args()
    if (
        min(
            options.levels,
            options.rows_per_level,
            options.batch_rows,
            options.budget_mib,
        )
        < 1
    ):
        parser.error("all sizes and the budget must be positive")
    if options.warm_repeats < 0:
        parser.error("warm-repeats must be nonnegative")
    budget = options.budget_mib * 1024 * 1024
    before = _peak_rss_bytes()
    frame = _frame(options.levels, options.rows_per_level)
    input_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    after_input = _peak_rss_bytes()
    cold, cold_seconds = _fit(frame, options.batch_rows, budget)
    after_cold_fit = _peak_rss_bytes()
    warm = []
    for _ in range(options.warm_repeats):
        fitted, seconds = _fit(frame, options.batch_rows, budget)
        np.testing.assert_array_equal(fitted.coefficients, cold.coefficients)
        warm.append(seconds)
    after_warm_fits = _peak_rss_bytes()
    diagnostics = asdict(cold.operator_diagnostics)
    if diagnostics["source"]["completed_scans"] != sum(
        diagnostics[name]
        for name in (
            "initialization_scans",
            "rhs_scans",
            "objective_scans",
            "certificate_scans",
            "krylov_actions",
        )
    ):
        raise AssertionError("Reported fit source passes do not reconcile.")
    prediction = cold.predict(frame.iloc[: min(32, len(frame))])
    after_prediction = _peak_rss_bytes()
    report = {
        "schema": "jaxgam-pr12-operator-benchmark-v1",
        "runtime": {
            **_git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "loaded_package": str(Path(jaxgam.__file__).resolve()),
            "loaded_package_sha256": hashlib.sha256(
                Path(jaxgam.__file__).read_bytes()
            ).hexdigest(),
            "image_digest": os.environ.get("JAXGAM_IMAGE_DIGEST"),
            "thread_env": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")
            },
        },
        "case": {
            "formula": "y~s(g,bs='re')",
            "n_rows": len(frame),
            "n_levels": options.levels,
            "n_coef": len(cold.coefficients),
            "rows_per_level": options.rows_per_level,
            "batch_rows": options.batch_rows,
            "memory_budget_bytes": budget,
            "global_square_bytes_if_materialized": 8 * len(cold.coefficients) ** 2,
        },
        "memory": {
            "caller_input_bytes": input_bytes,
            "rss_high_water_before_bytes": before,
            "rss_high_water_after_input_bytes": after_input,
            "rss_high_water_after_cold_fit_bytes": after_cold_fit,
            "rss_high_water_after_warm_fits_bytes": after_warm_fits,
            "rss_high_water_after_prediction_bytes": after_prediction,
            "operator_peak_bound_bytes": diagnostics["peak_bound_bytes"],
        },
        "timing": {
            "cold_fit_seconds": cold_seconds,
            "warm_fit_seconds": warm,
        },
        "result": {
            "converged": cold.converged,
            "iterations": cold.n_iter,
            "deviance": cold.deviance,
            "score": cold.score,
            "coefficient_sha256": hashlib.sha256(
                cold.coefficients.tobytes()
            ).hexdigest(),
            "prediction_first_32_sha256": hashlib.sha256(
                np.asarray(prediction).tobytes()
            ).hexdigest(),
            "max_direct_coefficient_error": (
                _direct_small_check(frame, cold.coefficients)
                if options.levels <= 32
                else None
            ),
        },
        "diagnostics": diagnostics,
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    output = options.output_dir / f"operator-factor-{options.levels}-levels.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "report": str(output),
                "timing": report["timing"],
                "diagnostics": diagnostics,
            }
        )
    )


if __name__ == "__main__":
    main()
