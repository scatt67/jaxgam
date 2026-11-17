"""Measure operator bulk Fisher variance and its selected-row calibration.

One operator fit is built once; then, for every seed, the public
``bulk_link_variance`` builds a probe sketch and calibrates the selected rows
against exact Fisher solves. The call's own raise limit is set out of reach so
every seed is recorded; whether the public default rule
(``BULK_CALIBRATION_MAX_STANDARD_ERRORS`` Monte Carlo standard errors) would
have raised is recorded per seed instead.

Per seed and calibration row the record holds the exact variance, the
estimate, the Monte Carlo standard error, the current standardized error
``|estimate - exact| / se`` and the ratio ``estimate / exact``. With Gaussian
probes and exact solves, ``probes * ratio`` is chi-square with ``probes``
degrees of freedom, so the ratio can be checked against an exact reference
while the standardized error reproduces the current rule. Probe solves,
source scans and PCG residuals come from the provider's diagnostics::

    make benchmark-bulk-variance ARGS="--model re --levels 10000 --seeds 1000 \\
        --output $RUN/bulk.json"

The process uses a task-owned JAX compilation cache. No speedup is claimed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_OUT_OF_REACH = 1e300


@contextlib.contextmanager
def task_owned_cache() -> Iterator[str]:
    """Point JAX at a new empty cache for this run, restoring the caller's."""
    previous = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="jaxgam-bulk-cache-") as cache:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache
        try:
            yield cache
        finally:
            if previous is None:
                os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
            else:
                os.environ["JAX_COMPILATION_CACHE_DIR"] = previous


def test_model_frame(family: str) -> tuple[pd.DataFrame, str, list[float], list[int]]:
    """The 47-row cr + re model of the bulk-variance tests, with its rows."""
    row = np.arange(47)
    x = np.linspace(0.1, 1.4, len(row))
    y = (
        1.3 + np.sin(2 * x) + 0.1 * np.cos(row)
        if family == "gaussian"
        else np.resize([1, 3, 2, 4, 2, 5], len(row)).astype(float)
    )
    frame = pd.DataFrame(
        {"y": y, "x": x, "g": pd.Categorical(np.resize(["a", "b", "c", "d"], 47))}
    )
    return frame, "y~s(x,bs='cr',k=6)+s(g,bs='re')", [0.7, 1.1], [0, 2, 13, 32, 46]


def re_model_frame(
    family: str, levels: int, rows_per_level: int, calibration_rows: int, seed: int
) -> tuple[pd.DataFrame, str, list[float], list[int]]:
    """A cubic smooth plus a ``levels``-level random effect, every level seen."""
    rng = np.random.default_rng(seed)
    n = levels * rows_per_level
    codes = np.tile(np.arange(levels), rows_per_level)
    x = rng.uniform(0.0, 1.0, n)
    signal = np.sin(2 * np.pi * x) + rng.normal(0.0, 0.5, levels)[codes]
    y = (
        signal + rng.normal(0.0, 0.3, n)
        if family == "gaussian"
        else rng.poisson(np.exp(0.2 + 0.5 * signal)).astype(float)
    )
    frame = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "g": pd.Categorical.from_codes(
                codes, categories=[f"g{level}" for level in range(levels)]
            ),
        }
    )
    selected = np.linspace(0, n - 1, min(calibration_rows, n)).astype(int)
    return frame, "y~s(x,bs='cr',k=6)+s(g,bs='re')", [1.0, 1.0], selected.tolist()


def clopper_pearson(successes: int, trials: int, level: float = 0.95) -> list[float]:
    """Exact two-sided binomial interval for a rate."""
    from scipy.stats import beta

    alpha = 1.0 - level
    lower = (
        0.0
        if successes == 0
        else beta.ppf(alpha / 2, successes, trials - successes + 1)
    )
    upper = (
        1.0
        if successes == trials
        else beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return [float(lower), float(upper)]


def summarize(seeds: list[dict[str, Any]], probes: int, limit: float) -> dict[str, Any]:
    """Old-rule exceedance rate and the exact chi-square check of the ratios."""
    from scipy.stats import chi2, kstest

    maxima = np.asarray([seed["maximum_standardized_error"] for seed in seeds])
    ratios = np.asarray([seed["ratio"] for seed in seeds])  # seeds x rows
    exceed = int(np.sum(maxima > limit))
    scaled = probes * ratios

    def ks(values: np.ndarray) -> dict[str, float]:
        result = kstest(values, chi2(probes).cdf)
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}

    return {
        "seeds": len(seeds),
        "default_limit_standard_errors": limit,
        "default_rule_exceedances": exceed,
        "default_rule_rate": exceed / len(seeds),
        "default_rule_rate_clopper_pearson_95": clopper_pearson(exceed, len(seeds)),
        "maximum_standardized_error_quantiles": {
            str(q): float(np.quantile(maxima, q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "ratio_reference": (
            "probes * estimate / exact ~ chi-square(probes) for Gaussian probes "
            "and exact solves; mean 1, variance 2 / probes"
        ),
        "ratio_mean": float(ratios.mean()),
        "ratio_variance": float(ratios.var(ddof=1)) if ratios.size > 1 else None,
        "ratio_reference_variance": 2.0 / probes,
        "ratio_ks_per_row": [ks(scaled[:, row]) for row in range(scaled.shape[1])],
        "ratio_ks_pooled_rows": ks(scaled.reshape(-1)),
        "ratio_ks_pooled_status": (
            "rows of one seed share probes and are correlated; the per-row tests "
            "use independent seeds"
        ),
    }


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "clean": run("status", "--porcelain") == "",
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"unavailable": str(error)}


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024 * value)


def run(options: argparse.Namespace, cache: str) -> dict[str, Any]:
    """Fit once, then build and calibrate one bulk sketch per seed."""
    import jax

    from jaxgam import GAM, FitControl
    from jaxgam.data.source import DataFrameRowSource
    from jaxgam.results import BULK_CALIBRATION_MAX_STANDARD_ERRORS

    if options.model == "test":
        frame, formula, sp, rows = test_model_frame(options.family)
    else:
        frame, formula, sp, rows = re_model_frame(
            options.family,
            options.levels,
            options.rows_per_level,
            options.calibration_rows,
            options.data_seed,
        )
    budget = options.memory_budget_mib << 20
    source = DataFrameRowSource(frame, response="y")
    started = time.perf_counter()
    fitted = GAM(
        formula,
        family=options.family,
        sp=sp,
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=options.batch_rows,
            memory_budget_bytes=budget,
        ),
    ).fit(source, result="prediction")
    fit_s = time.perf_counter() - started
    if not fitted.converged:
        raise RuntimeError(f"Operator fit did not converge: {fitted.convergence_info}")
    selected = frame.iloc[rows].copy()
    scale = None if fitted.scale is not None else options.scale
    seeds: list[dict[str, Any]] = []
    bulk_pass: dict[str, Any] | None = None
    for seed in range(options.first_seed, options.first_seed + options.seeds):
        started = time.perf_counter()
        provider, calibration = fitted.bulk_link_variance(
            source,
            selected,
            probes=options.probes,
            seed=seed,
            memory_budget_bytes=budget,
            scale=scale,
            max_calibration_standard_errors=_OUT_OF_REACH,
        )
        elapsed = time.perf_counter() - started
        exact = np.asarray(calibration.exact)
        estimated = np.asarray(calibration.estimated)
        error = np.asarray(calibration.monte_carlo_se)
        standardized = np.abs(estimated - exact) / error
        if bulk_pass is None:
            started = time.perf_counter()
            batches = list(
                provider.estimate_iter(DataFrameRowSource(frame, response="y"))
            )
            bulk_pass = {
                "rows": int(sum(len(batch.positions) for batch in batches)),
                "estimate_iter_s": time.perf_counter() - started,
                "prediction_source_scans": provider.diagnostics.prediction_source_scans,
                "finite": bool(
                    all(np.all(np.isfinite(batch.variance)) for batch in batches)
                ),
            }
        maximum = float(calibration.maximum_standardized_error)
        seeds.append(
            {
                "seed": seed,
                "build_and_calibrate_s": elapsed,
                "exact": exact.tolist(),
                "estimated": estimated.tolist(),
                "monte_carlo_se": error.tolist(),
                "standardized_error": standardized.tolist(),
                "ratio": (estimated / exact).tolist(),
                "maximum_standardized_error": maximum,
                "default_rule_would_raise": maximum
                > BULK_CALIBRATION_MAX_STANDARD_ERRORS,
                "diagnostics": asdict(provider.diagnostics),
            }
        )
    return {
        "schema": "jaxgam-bulk-variance-benchmark-v1",
        "case": {
            "model": options.model,
            "formula": formula,
            "family": options.family,
            "n_rows": len(frame),
            "n_coef": len(fitted.coefficients),
            "levels": None if options.model == "test" else options.levels,
            "rows_per_level": None
            if options.model == "test"
            else options.rows_per_level,
            "smoothing_parameters": sp,
            "calibration_rows": rows,
            "probes": options.probes,
            "first_seed": options.first_seed,
            "batch_rows": options.batch_rows,
            "memory_budget_bytes": budget,
            "fisher_scale": fitted.scale if scale is None else scale,
            "fisher_scale_source": "fitted" if scale is None else "supplied (--scale)",
        },
        "fit": {
            "seconds": fit_s,
            "converged": True,
            "status": str(fitted.convergence_info),
        },
        "bulk_pass": bulk_pass,
        "seeds": seeds,
        "summary": summarize(
            seeds, options.probes, BULK_CALIBRATION_MAX_STANDARD_ERRORS
        ),
        "runtime": {
            **_git_state(Path(__file__).resolve().parents[1]),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "compilation_cache": cache,
            "compilation_cache_status": (
                "task-owned; effective when this script is the process entry point"
            ),
            "thread_env": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")
            },
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=("test", "re"), default="test")
    result.add_argument("--family", choices=("gaussian", "poisson"), default="gaussian")
    result.add_argument("--levels", type=int, default=1000)
    result.add_argument("--rows-per-level", type=int, default=2)
    result.add_argument("--calibration-rows", type=int, default=5)
    result.add_argument("--data-seed", type=int, default=42)
    result.add_argument("--probes", type=int, default=48)
    result.add_argument("--seeds", type=int, default=100)
    result.add_argument("--first-seed", type=int, default=20000)
    result.add_argument("--batch-rows", type=int, default=4096)
    result.add_argument("--memory-budget-mib", type=int, default=256)
    result.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Fisher scale for unknown-scale families (the ratio is scale-free)",
    )
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> None:
    options = parser().parse_args(argv)
    if min(options.levels, options.rows_per_level, options.calibration_rows) < 1:
        raise SystemExit("levels, rows per level and calibration rows must be positive")
    if options.probes < 2 or options.seeds < 1:
        raise SystemExit("at least two probes and one seed are required")
    with task_owned_cache() as cache:
        record = run(options, cache)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(options.output), "summary": record["summary"]}))


if __name__ == "__main__":
    main()
