"""Benchmark pymgcv vs R's mgcv wall-clock fitting speed.

Usage::

    uv run python scripts/benchmark_vs_r.py

Reports two Python timings per configuration:
- **cold**: first fit at each (smooth, family, n) — includes JIT compilation
  for that array shape. Reflects the one-shot experience.
- **warm**: median of subsequent fits after JIT cache is hot for that shape.
  Reflects the repeated-fit experience (iteration, CV, bootstrap).

R timing is measured with system.time() inside R to avoid rpy2 overhead.
"""

from __future__ import annotations

import csv
import os
import pathlib
import shutil
import time
from statistics import median

import numpy as np
import pandas as pd

from pymgcv.api import GAM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_SIZES = [500, 2000, 10000, 100000, 500000]
FAMILIES = ["gaussian", "poisson", "binomial", "gamma"]
N_WARM_REPEATS = 5

SMOOTH_CONFIGS: dict[str, dict] = {
    "cr": {
        "py_formula": "y ~ s(x, k=10, bs='cr')",
        "r_formula": "y ~ s(x, k=10, bs='cr')",
        "data_type": "single",
    },
    "two": {
        "py_formula": "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')",
        "r_formula": "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')",
        "data_type": "two_smooth",
    },
    "te": {
        "py_formula": "y ~ te(x1, x2, k=5)",
        "r_formula": "y ~ te(x1, x2, k=c(5,5))",
        "data_type": "two_smooth",
    },
    "cr_by": {
        "py_formula": "y ~ s(x, by=fac, k=10, bs='cr') + fac",
        "r_formula": "y ~ s(x, by=fac, k=10, bs='cr') + fac",
        "data_type": "factor_by",
    },
}

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_response(
    eta: np.ndarray, family: str, rng: np.random.Generator
) -> np.ndarray:
    """Generate response variable from linear predictor and family."""
    if family == "gaussian":
        return eta + rng.normal(0, 0.3, len(eta))
    elif family == "poisson":
        return rng.poisson(np.exp(eta)).astype(float)
    elif family == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        return rng.binomial(1, prob, len(eta)).astype(float)
    elif family == "gamma":
        mu = np.exp(eta)
        return rng.gamma(5.0, scale=mu / 5.0, size=len(eta))
    else:
        raise ValueError(f"Unknown family: {family}")


def make_single_data(n: int, family: str, seed: int = 42) -> pd.DataFrame:
    """Single predictor: y ~ sin(2*pi*x)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x)
    if family == "poisson":
        eta = eta + 0.5
    elif family == "gamma":
        eta = 0.5 * eta + 1.0
    elif family == "binomial":
        eta = 2 * eta
    y = _make_response(eta, family, rng)
    return pd.DataFrame({"x": x, "y": y})


def make_two_smooth_data(n: int, family: str, seed: int = 42) -> pd.DataFrame:
    """Two predictors: y ~ sin(2*pi*x1) + 0.5*x2."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x1) + 0.5 * x2
    if family == "poisson":
        eta = eta + 0.5
    elif family == "gamma":
        eta = 0.5 * eta + 1.0
    elif family == "binomial":
        eta = 2 * eta
    y = _make_response(eta, family, rng)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def make_factor_by_data(n: int, family: str, seed: int = 42) -> pd.DataFrame:
    """Factor-by: y ~ level-specific signal."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    levels = ["a", "b", "c"]
    fac = rng.choice(levels, n)

    level_funcs = {
        "a": lambda x_: np.sin(2 * np.pi * x_),
        "b": lambda x_: np.cos(2 * np.pi * x_),
        "c": lambda x_: 0.5 * x_,
    }
    eta = np.zeros(n)
    for level, func in level_funcs.items():
        mask = fac == level
        eta[mask] = func(x[mask])

    if family == "poisson":
        eta = eta + 0.5
    elif family == "gamma":
        eta = 0.5 * eta + 1.0
    elif family == "binomial":
        eta = 2 * eta
    y = _make_response(eta, family, rng)
    return pd.DataFrame({"x": x, "y": y, "fac": pd.Categorical(fac)})


DATA_MAKERS = {
    "single": make_single_data,
    "two_smooth": make_two_smooth_data,
    "factor_by": make_factor_by_data,
}

# ---------------------------------------------------------------------------
# R timing via rpy2
# ---------------------------------------------------------------------------


def check_r_available() -> bool:
    """Check if rpy2 and mgcv are available."""
    try:
        import rpy2.robjects  # noqa: F401

        return True
    except Exception:
        return False


def setup_r():
    """Initialize rpy2 and load mgcv."""
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    importr("mgcv")
    return ro


def time_r_fit(
    ro,
    r_formula: str,
    data: pd.DataFrame,
    family: str,
    nthreads: int = 1,
) -> tuple[float, int]:
    """Time R's gam() using system.time() and return (ms, n_iter)."""
    from rpy2.robjects import numpy2ri, pandas2ri

    family_map = {
        "gaussian": "gaussian()",
        "binomial": "binomial()",
        "poisson": "poisson()",
        "gamma": "Gamma()",
    }

    with ro.conversion.localconverter(
        ro.default_converter + pandas2ri.converter + numpy2ri.converter
    ):
        ro.globalenv["bench_data"] = ro.conversion.py2rpy(data)

    r_family = family_map[family]
    ctrl = f", control=list(nthreads={nthreads})" if nthreads > 1 else ""
    r_code = f"""
    tm <- system.time({{
        mod <- gam({r_formula}, data=bench_data, family={r_family}, method="REML"{ctrl})
    }})
    elapsed_ms <- tm["elapsed"] * 1000

    # Extract Newton iteration count
    n_iter <- tryCatch({{
        oi <- mod$outer.info
        if (!is.null(oi$iter)) {{
            as.integer(oi$iter[1])
        }} else if (!is.null(oi$grad)) {{
            length(oi$grad)
        }} else {{
            -1L
        }}
    }}, error=function(e) -1L)

    c(elapsed_ms, n_iter)
    """
    result = np.array(ro.r(r_code))
    elapsed_ms = float(result[0])
    n_iter = int(result[1])
    return elapsed_ms, n_iter


# ---------------------------------------------------------------------------
# Python timing
# ---------------------------------------------------------------------------


def time_py_fit(
    py_formula: str,
    data: pd.DataFrame,
    family: str,
) -> tuple[float, int]:
    """Time pymgcv GAM fitting and return (ms, n_iter)."""
    model = GAM(py_formula, family=family, method="REML")
    t0 = time.perf_counter()
    model.fit(data)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    n_iter = getattr(model, "n_iter_", -1)
    return elapsed_ms, n_iter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    has_r = check_r_available()
    if not has_r:
        print("WARNING: rpy2/R not available. Running Python-only benchmarks.")
        print()
    else:
        ro = setup_r()

    # Pre-generate all datasets
    print("Generating datasets...")
    datasets: dict[tuple[str, str, int], pd.DataFrame] = {}
    for smooth_key, cfg in SMOOTH_CONFIGS.items():
        data_type = cfg["data_type"]
        maker = DATA_MAKERS[data_type]
        for family in FAMILIES:
            for n in DATA_SIZES:
                datasets[(smooth_key, family, n)] = maker(n, family)
    print(f"  {len(datasets)} datasets ready.\n")

    # ---------------------------------------------------------------------------
    # True cold pass: first-ever fit at largest n, BEFORE any JIT warmup.
    # This measures the worst case: full trace compilation + XLA compilation
    # + actual fitting, all in one shot.
    # ---------------------------------------------------------------------------
    # Clear persistent compilation cache so XLA must recompile from scratch.
    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(pathlib.Path.home() / ".cache" / "pymgcv" / "jax"),
    )
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared JAX compilation cache: {cache_dir}")

    true_cold_n = DATA_SIZES[-1]
    print(f"True cold benchmark (no prior JIT, n={true_cold_n})...")
    true_cold_results: dict[tuple[str, str], tuple[float, int]] = {}
    total_tc = len(SMOOTH_CONFIGS) * len(FAMILIES)
    done_tc = 0
    for smooth_key, cfg in SMOOTH_CONFIGS.items():
        for family in FAMILIES:
            done_tc += 1
            data = datasets[(smooth_key, family, true_cold_n)]
            try:
                ms, it = time_py_fit(cfg["py_formula"], data, family)
                true_cold_results[(smooth_key, family)] = (ms, it)
                print(f"  [{done_tc}/{total_tc}] {smooth_key}/{family}: {ms:.0f}ms")
            except Exception as e:
                true_cold_results[(smooth_key, family)] = (float("nan"), -1)
                print(f"  [{done_tc}/{total_tc}] {smooth_key}/{family}: FAILED ({e})")
    print()

    # Minimal JIT warmup: one tiny fit per (smooth, family) to compile
    # the function traces. Shape-specific recompilation for each n is
    # measured as part of the "cold" benchmark.
    print("JIT warmup (minimal — compile traces only)...")
    for smooth_key, cfg in SMOOTH_CONFIGS.items():
        for family in FAMILIES:
            data = datasets[(smooth_key, family, DATA_SIZES[0])]
            try:
                GAM(cfg["py_formula"], family=family).fit(data)
                print(f"  {smooth_key}/{family}: OK")
            except Exception as e:
                print(f"  {smooth_key}/{family}: FAILED ({e})")
    print()

    # ---------------------------------------------------------------------------
    # Cold pass: first fit at each (smooth, family, n)
    # For n > smallest, this includes JIT recompilation for the new shape.
    # ---------------------------------------------------------------------------
    print("Cold benchmark (includes JIT shape recompilation)...")
    cold_results: dict[tuple[str, str, int], tuple[float, int]] = {}
    total = len(SMOOTH_CONFIGS) * len(FAMILIES) * len(DATA_SIZES)
    done = 0
    for smooth_key, cfg in SMOOTH_CONFIGS.items():
        for family in FAMILIES:
            for n in DATA_SIZES:
                done += 1
                data = datasets[(smooth_key, family, n)]
                try:
                    ms, it = time_py_fit(cfg["py_formula"], data, family)
                    cold_results[(smooth_key, family, n)] = (ms, it)
                    print(f"  [{done}/{total}] {smooth_key}/{family}/n={n}: {ms:.0f}ms")
                except Exception as e:
                    cold_results[(smooth_key, family, n)] = (float("nan"), -1)
                    print(
                        f"  [{done}/{total}] {smooth_key}/{family}/n={n}: FAILED ({e})"
                    )
    print()

    # ---------------------------------------------------------------------------
    # Warm pass: all shapes now compiled. Time median of N_WARM_REPEATS fits.
    # ---------------------------------------------------------------------------
    print(f"Warm benchmark (JIT cache hot, median of {N_WARM_REPEATS})...")
    warm_results: dict[tuple[str, str, int], tuple[float, int]] = {}
    done = 0
    for smooth_key, cfg in SMOOTH_CONFIGS.items():
        for family in FAMILIES:
            for n in DATA_SIZES:
                done += 1
                data = datasets[(smooth_key, family, n)]
                n_reps = 1 if n >= 100000 else N_WARM_REPEATS
                times = []
                iters = []
                try:
                    for _ in range(n_reps):
                        ms, it = time_py_fit(cfg["py_formula"], data, family)
                        times.append(ms)
                        iters.append(it)
                    med = median(times)
                    warm_results[(smooth_key, family, n)] = (med, iters[0])
                    print(
                        f"  [{done}/{total}] {smooth_key}/{family}/n={n}: {med:.1f}ms"
                    )
                except Exception as e:
                    warm_results[(smooth_key, family, n)] = (float("nan"), -1)
                    print(
                        f"  [{done}/{total}] {smooth_key}/{family}/n={n}: FAILED ({e})"
                    )
    print()

    # ---------------------------------------------------------------------------
    # R timing (no JIT, so just median of repeats)
    # ---------------------------------------------------------------------------
    r_results: dict[tuple[str, str, int], tuple[float, int]] = {}
    if has_r:
        print("R benchmark (median of 3, single-threaded)...")
        done = 0
        for smooth_key, cfg in SMOOTH_CONFIGS.items():
            for family in FAMILIES:
                for n in DATA_SIZES:
                    done += 1
                    data = datasets[(smooth_key, family, n)]
                    n_reps = 1 if n >= 10000 else 3
                    try:
                        r_times = []
                        r_iters = []
                        for _ in range(n_reps):
                            ms, it = time_r_fit(
                                ro,
                                cfg["r_formula"],
                                data,
                                family,
                            )
                            r_times.append(ms)
                            r_iters.append(it)
                        r_results[(smooth_key, family, n)] = (
                            median(r_times),
                            r_iters[0],
                        )
                        print(
                            f"  [{done}/{total}] {smooth_key}/{family}/n={n}: "
                            f"{median(r_times):.0f}ms"
                        )
                    except Exception as e:
                        r_results[(smooth_key, family, n)] = (float("nan"), -1)
                        print(
                            f"  [{done}/{total}] {smooth_key}/{family}/n={n}: "
                            f"FAILED ({e})"
                        )
        print()

    # ---------------------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------------------
    def _ratio_str(py_ms: float, r_ms: float) -> str:
        if np.isnan(py_ms) or np.isnan(r_ms) or py_ms <= 0:
            return "N/A"
        ratio = r_ms / py_ms
        return f"{ratio:.1f}x" if ratio >= 1.0 else f"{ratio:.2f}x"

    # ---------------------------------------------------------------------------
    # True cold summary (n=100k only)
    # ---------------------------------------------------------------------------
    print("=" * 110)
    print(f"TRUE COLD vs R (n={true_cold_n}, no prior JIT)")
    print("=" * 110)
    print()
    tc_header = (
        f"| {'smooth':<6} | {'family':<8} "
        f"| {'true_cold':>10} | {'warm':>8} | {'r_ms':>7} "
        f"| {'tcold/R':>8} | {'warm/R':>7} "
        f"| {'jit_cost':>8} |"
    )
    tc_sep = (
        f"|{'-' * 8}|{'-' * 10}"
        f"|{'-' * 12}|{'-' * 10}|{'-' * 9}"
        f"|{'-' * 10}|{'-' * 9}"
        f"|{'-' * 10}|"
    )
    print(tc_header)
    print(tc_sep)
    for smooth_key in SMOOTH_CONFIGS:
        for family in FAMILIES:
            tc_key = (smooth_key, family)
            tc_ms, _ = true_cold_results.get(tc_key, (float("nan"), -1))
            w_key = (smooth_key, family, true_cold_n)
            w_ms, _ = warm_results.get(w_key, (float("nan"), -1))
            r_key = (smooth_key, family, true_cold_n)
            r_ms_val, _ = r_results.get(r_key, (float("nan"), -1))
            both_valid = not (np.isnan(tc_ms) or np.isnan(w_ms))
            jit_cost = tc_ms - w_ms if both_valid else float("nan")
            tc_ratio = _ratio_str(tc_ms, r_ms_val)
            w_ratio = _ratio_str(w_ms, r_ms_val)

            def _fmt(v: float) -> str:
                if np.isnan(v):
                    return "N/A"
                return f"{v:.0f}"

            print(
                f"| {smooth_key:<6} | {family:<8} "
                f"| {_fmt(tc_ms):>10} | {_fmt(w_ms):>8} | {_fmt(r_ms_val):>7} "
                f"| {tc_ratio:>8} | {w_ratio:>7} "
                f"| {_fmt(jit_cost):>8} |"
            )
    print()

    # ---------------------------------------------------------------------------
    # Full results table (all sizes)
    # ---------------------------------------------------------------------------
    print("=" * 110)
    print("FULL RESULTS (all sizes)")
    print("=" * 110)
    print()
    header = (
        f"| {'smooth':<6} | {'family':<8} | {'n':>6} "
        f"| {'py_cold':>8} | {'py_warm':>8} | {'r_ms':>7} "
        f"| {'cold/R':>7} | {'warm/R':>7} "
        f"| {'py_iter':>7} | {'r_iter':>6} |"
    )
    sep = (
        f"|{'-' * 8}|{'-' * 10}|{'-' * 8}"
        f"|{'-' * 10}|{'-' * 10}|{'-' * 9}"
        f"|{'-' * 9}|{'-' * 9}"
        f"|{'-' * 9}|{'-' * 8}|"
    )
    print(header)
    print(sep)

    rows = []
    for smooth_key in SMOOTH_CONFIGS:
        for family in FAMILIES:
            for n in DATA_SIZES:
                key = (smooth_key, family, n)
                cold_ms, cold_it = cold_results.get(key, (float("nan"), -1))
                warm_ms, warm_it = warm_results.get(key, (float("nan"), -1))
                r_ms, r_it = r_results.get(key, (float("nan"), -1))

                cold_ratio = _ratio_str(cold_ms, r_ms)
                warm_ratio = _ratio_str(warm_ms, r_ms)

                py_iter = warm_it if warm_it >= 0 else cold_it

                def _fmt(v: float) -> str:
                    if np.isnan(v):
                        return "FAIL"
                    return f"{v:.1f}"

                print(
                    f"| {smooth_key:<6} | {family:<8} | {n:>6} "
                    f"| {_fmt(cold_ms):>8} | {_fmt(warm_ms):>8} | {_fmt(r_ms):>7} "
                    f"| {cold_ratio:>7} | {warm_ratio:>7} "
                    f"| {py_iter:>7} | {r_it:>6} |"
                )

                rows.append(
                    {
                        "smooth": smooth_key,
                        "family": family,
                        "n": n,
                        "py_cold_ms": round(cold_ms, 1),
                        "py_warm_ms": round(warm_ms, 1),
                        "r_ms": round(r_ms, 1),
                        "cold_vs_r": cold_ratio,
                        "warm_vs_r": warm_ratio,
                        "py_iter": py_iter,
                        "r_iter": r_it,
                    }
                )

    # Save CSV
    csv_path = "scripts/benchmark_results.csv"
    # Add true cold column for rows at true_cold_n
    for row in rows:
        tc_key = (row["smooth"], row["family"])
        if row["n"] == true_cold_n and tc_key in true_cold_results:
            row["py_true_cold_ms"] = round(true_cold_results[tc_key][0], 1)
        else:
            row["py_true_cold_ms"] = ""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "smooth",
                "family",
                "n",
                "py_true_cold_ms",
                "py_cold_ms",
                "py_warm_ms",
                "r_ms",
                "cold_vs_r",
                "warm_vs_r",
                "py_iter",
                "r_iter",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
