"""Paired PCG behavior for a smooth confounded with indexed factor levels.

Run ``make benchmark-operator-preconditioner``. The comparison uses the same
frozen source, row curvature, penalty and right-hand side for both policies.
It reports actual actions and verified residuals, not a promised speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np
import pandas as pd

import jaxgam
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator import FrozenScanSystem, PCGControl
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from tests.tolerances import STRICT


def _case(levels: int, log_re_sp: float, budget: int) -> dict:
    n = levels * 4
    groups = np.tile(np.arange(levels), 4)
    x = (groups + 0.01 * np.sin(np.arange(n))) / levels
    frame = pd.DataFrame(
        {
            "y": 1 + np.sin(2 * np.pi * x),
            "x": x,
            "g": pd.Categorical([f"g{index}" for index in groups]),
        }
    )
    formula = "y~s(x,bs='cr',k=6)+s(g,bs='re')"
    rho = np.array([-5.0, log_re_sp])
    source = DataFrameRowSource(frame, response="y")
    spec = parse_formula(formula)
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=31)
    )
    dense = ModelSetup.build(spec, frame)
    H = dense.X.T @ dense.X + dense.penalties.materialize(rho)
    rhs = 1.0 + 0.1 * np.sin(np.arange(model.n_coef))
    expected = np.linalg.solve(H, rhs)
    outcomes = {}
    for use_coarse in (False, True):
        system = FrozenScanSystem(
            model,
            source,
            rho,
            memory_budget_bytes=budget,
            use_coarse=use_coarse,
        )
        solved = system.solve(rhs, control=PCGControl(rtol=1e-11))
        if not solved.converged:
            raise RuntimeError("Paired preconditioner solve did not converge.")
        # PCG stops on the residual norm relative to ||rhs||, so the check
        # against the independently assembled dense system is the same norm,
        # at STRICT: an elementwise comparison is not implied by that stop
        # (one coefficient of 1,006 was 4.9e-10 off at 1,000 levels while the
        # norm was 1.5e-11 of ||rhs||).
        dense_relative_residual = float(
            np.linalg.norm(H @ solved.solution - rhs) / np.linalg.norm(rhs)
        )
        if dense_relative_residual > STRICT.rtol:
            raise AssertionError(
                "Paired solve leaves a dense relative residual of "
                f"{dense_relative_residual:.3e}, above STRICT {STRICT.rtol:.0e}."
            )
        if system.diagnostics.completed_scans != solved.matvec_count + 1:
            raise AssertionError("Paired solve source passes did not reconcile.")
        outcomes[str(use_coarse).lower()] = {
            "iterations": solved.iterations,
            "matvec_count": solved.matvec_count,
            "relative_true_residual": solved.relative_residual,
            "dense_relative_residual": dense_relative_residual,
            "source_scans": system.diagnostics.completed_scans,
            "peak_bound_bytes": max(
                system.construction_peak_bound_bytes,
                system.action_peak_bound_bytes + solved.additional_peak_bound_bytes,
            ),
            "construction_peak_bound_bytes": system.construction_peak_bound_bytes,
            "solve_additional_peak_bound_bytes": solved.additional_peak_bound_bytes,
            "max_direct_coefficient_error": float(
                np.max(np.abs(solved.solution - expected))
            ),
        }
    return {
        "levels": levels,
        "n_rows": n,
        "n_coef": model.n_coef,
        "formula": formula,
        "log_sp": rho.tolist(),
        "dense_min_eigenvalue": float(np.linalg.eigvalsh(H)[0]),
        "dense_condition_number": float(np.linalg.cond(H)),
        "batch_rows": model.batch_rows,
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[20, 80])
    parser.add_argument("--log-re-sp", type=float, default=-5.0)
    parser.add_argument("--budget-mib", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/jaxgam-operator-benchmark")
    )
    options = parser.parse_args()
    if min(*options.levels, options.budget_mib) < 1:
        parser.error("levels and budget must be positive")
    if not np.isfinite(options.log_re_sp):
        parser.error("log-re-sp must be finite")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "schema": "jaxgam-pr12-operator-preconditioner-v1",
        "runtime": {
            "commit": revision.stdout.strip()
            if revision.returncode == 0
            else os.environ.get("JAXGAM_SOURCE_COMMIT"),
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "provenance_status": "git-worktree"
            if revision.returncode == 0
            else "caller-supplied-no-git"
            if os.environ.get("JAXGAM_SOURCE_COMMIT")
            else "unavailable-no-git",
            "image_digest": os.environ.get("JAXGAM_IMAGE_DIGEST"),
            "loaded_package": str(Path(jaxgam.__file__).resolve()),
            "loaded_package_sha256": hashlib.sha256(
                Path(jaxgam.__file__).read_bytes()
            ).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "thread_env": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")
            },
        },
        "cases": [
            _case(levels, options.log_re_sp, options.budget_mib * 1024 * 1024)
            for levels in options.levels
        ],
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    output = options.output_dir / "operator-coarse-preconditioner.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(output), "cases": report["cases"]}))


if __name__ == "__main__":
    main()
