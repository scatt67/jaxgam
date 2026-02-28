"""Benchmark pymgcv vs R mgcv with large basis dimensions (p=100,200,500).

Usage::

    uv run python scripts/benchmark_large_p.py

At large p, R's O(np²) BLAS operations become significant and its OpenMP
parallelism via ``control=list(nthreads=N)`` should matter.  This script
compares pymgcv (JAX, single-device) against R with 1 thread and R with
all available threads.

Produces ``scripts/large_p_results.png``.
"""

from __future__ import annotations

import functools
import os
import pathlib
import shutil
import time
from statistics import geometric_mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymgcv.api import GAM

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASIS_DIMS = [100, 200, 500]
N_SIZES = [100_000]
FAMILIES = ["gaussian", "poisson"]

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_response(eta, family, rng):
    if family == "gaussian":
        return eta + rng.normal(0, 0.3, len(eta))
    elif family == "poisson":
        return rng.poisson(np.exp(eta)).astype(float)
    else:
        raise ValueError(family)


def make_data(n, family, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x)
    if family == "poisson":
        eta = eta + 0.5
    y = _make_response(eta, family, rng)
    return pd.DataFrame({"x": x, "y": y})


# ---------------------------------------------------------------------------
# R timing
# ---------------------------------------------------------------------------


def setup_r():
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    importr("mgcv")
    return ro


def time_r_fit(ro, k, data, family, nthreads=1):
    from rpy2.robjects import numpy2ri, pandas2ri

    family_map = {"gaussian": "gaussian()", "poisson": "poisson()"}
    with ro.conversion.localconverter(
        ro.default_converter + pandas2ri.converter + numpy2ri.converter
    ):
        ro.globalenv["bench_data"] = ro.conversion.py2rpy(data)

    r_family = family_map[family]
    ctrl = f", control=list(nthreads={nthreads})" if nthreads > 1 else ""
    r_code = f"""
    tm <- system.time({{
        mod <- gam(y ~ s(x, k={k}, bs='cr'), data=bench_data,
                   family={r_family}, method="REML"{ctrl})
    }})
    elapsed_ms <- tm["elapsed"] * 1000
    n_iter <- tryCatch({{
        oi <- mod$outer.info
        if (!is.null(oi$iter)) as.integer(oi$iter[1])
        else if (!is.null(oi$grad)) length(oi$grad)
        else -1L
    }}, error=function(e) -1L)
    c(elapsed_ms, n_iter)
    """
    result = np.array(ro.r(r_code))
    return float(result[0]), int(result[1])


# ---------------------------------------------------------------------------
# Python timing
# ---------------------------------------------------------------------------


def time_py_fit(k, data, family):
    formula = f"y ~ s(x, k={k}, bs='cr')"
    model = GAM(formula, family=family, method="REML")
    t0 = time.perf_counter()
    model.fit(data)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    n_iter = getattr(model, "n_iter_", -1)
    return elapsed_ms, n_iter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        import rpy2.robjects  # noqa: F401

        has_r = True
    except Exception:
        has_r = False

    if not has_r:
        print("ERROR: rpy2/R required.")
        return

    ro = setup_r()
    r_ncores = int(np.array(ro.r("parallel::detectCores()"))[0])
    print(f"Detected {r_ncores} CPU cores for R nthreads.\n")

    # Clear JAX persistent compilation cache
    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(pathlib.Path.home() / ".cache" / "pymgcv" / "jax"),
    )
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared JAX compilation cache: {cache_dir}")

    # Pre-generate datasets
    print("Generating datasets...")
    datasets: dict[tuple[int, str], pd.DataFrame] = {}
    for n in N_SIZES:
        for family in FAMILIES:
            datasets[(n, family)] = make_data(n, family)
    print(f"  {len(datasets)} datasets ready.\n")

    # Storage: results[k][family][n] = {py_cold, py_warm, r_1t, r_mt, ...}
    results: dict[int, dict[str, dict[int, dict]]] = {
        k: {f: {} for f in FAMILIES} for k in BASIS_DIMS
    }

    total = len(BASIS_DIMS) * len(FAMILIES) * len(N_SIZES)

    # --- Python true cold ---
    print("Python true-cold benchmark (no prior JIT)...")
    done = 0
    for k in BASIS_DIMS:
        for family in FAMILIES:
            for n in N_SIZES:
                done += 1
                data = datasets[(n, family)]
                try:
                    ms, it = time_py_fit(k, data, family)
                    results[k][family][n] = {"py_cold": ms, "py_iter": it}
                    print(
                        f"  [{done}/{total}] k={k}/{family}"
                        f"/n={n:>7,}: {ms:>8.0f}ms ({it} iter)"
                    )
                except Exception as e:
                    results[k][family][n] = {
                        "py_cold": float("nan"),
                        "py_iter": -1,
                    }
                    print(f"  [{done}/{total}] k={k}/{family}/n={n:>7,}: FAILED ({e})")
    print()

    # --- R single-threaded ---
    print("R benchmark (nthreads=1)...")
    done = 0
    for k in BASIS_DIMS:
        for family in FAMILIES:
            for n in N_SIZES:
                done += 1
                data = datasets[(n, family)]
                try:
                    ms, it = time_r_fit(ro, k, data, family, nthreads=1)
                    results[k][family][n]["r_1t"] = ms
                    results[k][family][n]["r_iter"] = it
                    print(
                        f"  [{done}/{total}] k={k}/{family}"
                        f"/n={n:>7,}: {ms:>8.0f}ms ({it} iter)"
                    )
                except Exception as e:
                    results[k][family][n]["r_1t"] = float("nan")
                    print(f"  [{done}/{total}] k={k}/{family}/n={n:>7,}: FAILED ({e})")
    print()

    # --- R multi-threaded ---
    print(f"R benchmark (nthreads={r_ncores})...")
    done = 0
    for k in BASIS_DIMS:
        for family in FAMILIES:
            for n in N_SIZES:
                done += 1
                data = datasets[(n, family)]
                try:
                    ms, it = time_r_fit(
                        ro,
                        k,
                        data,
                        family,
                        nthreads=r_ncores,
                    )
                    results[k][family][n]["r_mt"] = ms
                    print(
                        f"  [{done}/{total}] k={k}/{family}"
                        f"/n={n:>7,}: {ms:>8.0f}ms ({it} iter)"
                    )
                except Exception as e:
                    results[k][family][n]["r_mt"] = float("nan")
                    print(f"  [{done}/{total}] k={k}/{family}/n={n:>7,}: FAILED ({e})")
    print()

    # --- Results table ---
    def _ratio(a, b):
        if np.isnan(a) or np.isnan(b) or a <= 0:
            return "N/A"
        r = b / a
        return f"{r:.1f}x" if r >= 1.0 else f"{r:.2f}x"

    def _fmt(v):
        if np.isnan(v):
            return "N/A"
        return f"{v:.0f}"

    print("=" * 110)
    print(f"LARGE-P BENCHMARK  |  R threads: 1 vs {r_ncores}  |  pymgcv: true cold")
    print("=" * 110)
    header = (
        f"| {'k':>4} | {'family':<8} | {'n':>7} "
        f"| {'py_cold':>8} "
        f"| {'R(1t)':>8} | {'R(Mt)':>8} "
        f"| {'R Mt/1t':>7} "
        f"| {'cold/1t':>7} | {'cold/Mt':>7} "
        f"| {'py_it':>5} | {'r_it':>5} |"
    )
    sep = (
        f"|{'-' * 6}|{'-' * 10}|{'-' * 9}"
        f"|{'-' * 10}"
        f"|{'-' * 10}|{'-' * 10}"
        f"|{'-' * 9}"
        f"|{'-' * 9}|{'-' * 9}"
        f"|{'-' * 7}|{'-' * 7}|"
    )
    print(header)
    print(sep)

    for k in BASIS_DIMS:
        for family in FAMILIES:
            for n in N_SIZES:
                d = results[k][family].get(n, {})
                pc = d.get("py_cold", float("nan"))
                r1 = d.get("r_1t", float("nan"))
                rm = d.get("r_mt", float("nan"))
                pi = d.get("py_iter", -1)
                ri = d.get("r_iter", -1)

                r_thread_speedup = _ratio(rm, r1)
                cold_vs_r1 = _ratio(pc, r1)
                cold_vs_rm = _ratio(pc, rm)

                print(
                    f"| {k:>4} | {family:<8} | {n:>7,} "
                    f"| {_fmt(pc):>8} "
                    f"| {_fmt(r1):>8} | {_fmt(rm):>8} "
                    f"| {r_thread_speedup:>7} "
                    f"| {cold_vs_r1:>7} | {cold_vs_rm:>7} "
                    f"| {pi:>5} | {ri:>5} |"
                )
    print()

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute geometric mean across families for each k
    bar_data = {}  # k -> (py_cold, r_1t, r_mt)
    for k in BASIS_DIMS:
        pc_vals, r1_vals, rm_vals = [], [], []
        for family in FAMILIES:
            for n in N_SIZES:
                d = results[k][family].get(n, {})
                pc = d.get("py_cold", float("nan"))
                r1 = d.get("r_1t", float("nan"))
                rm = d.get("r_mt", float("nan"))
                if not any(np.isnan(v) for v in [pc, r1, rm]):
                    pc_vals.append(pc)
                    r1_vals.append(r1)
                    rm_vals.append(rm)
        if pc_vals:
            bar_data[k] = (
                geometric_mean(pc_vals),
                geometric_mean(r1_vals),
                geometric_mean(rm_vals),
            )

    x_labels = [f"k={k}" for k in bar_data]
    x = np.arange(len(x_labels))
    width = 0.25

    py_vals = [bar_data[k][0] for k in bar_data]
    r1_vals = [bar_data[k][1] for k in bar_data]
    rm_vals = [bar_data[k][2] for k in bar_data]

    bars_py = ax.bar(
        x - width,
        py_vals,
        width,
        label="pymgcv (true cold)",
        color="#1f77b4",
    )
    bars_r1 = ax.bar(
        x,
        r1_vals,
        width,
        label="R (1 thread)",
        color="#d62728",
        alpha=0.7,
    )
    bars_rm = ax.bar(
        x + width,
        rm_vals,
        width,
        label=f"R ({r_ncores} threads)",
        color="#d62728",
    )

    # Add time labels on bars
    for bars in [bars_py, bars_r1, bars_rm]:
        for bar in bars:
            h = bar.get_height()
            label = f"{h / 1000:.1f}s" if h >= 1000 else f"{h:.0f}ms"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_yscale("log")
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_title(
        f"pymgcv (true cold) vs R mgcv at n={N_SIZES[0]:,}\n"
        "(geometric mean across gaussian & poisson)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    out_path = "scripts/large_p_results.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
