"""Benchmark jaxgam vs R mgcv with large basis dimensions (p=100,200,500).

Usage::

    uv run python scripts/benchmark_large_p.py

At large p, R's O(np^2) BLAS operations become significant and its OpenMP
parallelism via ``control=list(nthreads=N)`` should matter.  This script
compares jaxgam (JAX, single-device) against R gam(REML) and bam(fREML).

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

from jaxgam.api import GAM

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASIS_DIMS = [100, 200, 500]
N_SIZES = [100_000]
FAMILIES = ["gaussian", "poisson"]
BAM_NTHREADS = 8

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


def time_r_fit(ro, k, data, family, method="gam", nthreads=1):
    """Time an R gam() or bam() fit.

    Parameters
    ----------
    method : str
        "gam" for gam(method="REML"), "bam" for bam(method="fREML").
    """
    from rpy2.robjects import numpy2ri, pandas2ri

    family_map = {"gaussian": "gaussian()", "poisson": "poisson()"}
    with ro.conversion.localconverter(
        ro.default_converter + pandas2ri.converter + numpy2ri.converter
    ):
        ro.globalenv["bench_data"] = ro.conversion.py2rpy(data)

    r_family = family_map[family]
    ctrl = f", control=list(nthreads={nthreads})" if nthreads > 1 else ""

    if method == "bam":
        r_code = f"""
        tm <- system.time({{
            mod <- bam(y ~ s(x, k={k}, bs='cr'), data=bench_data,
                       family={r_family}, method="fREML",
                       discrete=FALSE{ctrl})
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
    else:
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

    # Clear JAX persistent compilation cache
    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(pathlib.Path.home() / ".cache" / "jaxgam" / "jax"),
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

    # Storage: results[k][family][n] = {py_cold, r_gam, r_bam, ...}
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

    # --- R gam(REML) single-threaded ---
    print("R benchmark: gam(REML, nthreads=1)...")
    done = 0
    for k in BASIS_DIMS:
        for family in FAMILIES:
            for n in N_SIZES:
                done += 1
                data = datasets[(n, family)]
                try:
                    ms, it = time_r_fit(ro, k, data, family, method="gam")
                    results[k][family][n]["r_gam"] = ms
                    results[k][family][n]["r_gam_iter"] = it
                    print(
                        f"  [{done}/{total}] k={k}/{family}"
                        f"/n={n:>7,}: {ms:>8.0f}ms ({it} iter)"
                    )
                except Exception as e:
                    results[k][family][n]["r_gam"] = float("nan")
                    print(f"  [{done}/{total}] k={k}/{family}/n={n:>7,}: FAILED ({e})")
    print()

    # --- R bam(fREML) with threads ---
    print(f"R benchmark: bam(fREML, nthreads={BAM_NTHREADS})...")
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
                        method="bam",
                        nthreads=BAM_NTHREADS,
                    )
                    results[k][family][n]["r_bam"] = ms
                    results[k][family][n]["r_bam_iter"] = it
                    print(
                        f"  [{done}/{total}] k={k}/{family}"
                        f"/n={n:>7,}: {ms:>8.0f}ms ({it} iter)"
                    )
                except Exception as e:
                    results[k][family][n]["r_bam"] = float("nan")
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

    print("=" * 120)
    print(
        f"LARGE-P BENCHMARK  |  R: gam(REML) vs bam(fREML, {BAM_NTHREADS}t)"
        "  |  jaxgam: true cold"
    )
    print("=" * 120)
    header = (
        f"| {'k':>4} | {'family':<8} | {'n':>7} "
        f"| {'py_cold':>8} "
        f"| {'R gam':>8} | {'R bam':>8} "
        f"| {'py/gam':>7} | {'py/bam':>7} "
        f"| {'py_it':>5} | {'gam_it':>6} | {'bam_it':>6} |"
    )
    sep = (
        f"|{'-' * 6}|{'-' * 10}|{'-' * 9}"
        f"|{'-' * 10}"
        f"|{'-' * 10}|{'-' * 10}"
        f"|{'-' * 9}|{'-' * 9}"
        f"|{'-' * 7}|{'-' * 8}|{'-' * 8}|"
    )
    print(header)
    print(sep)

    for k in BASIS_DIMS:
        for family in FAMILIES:
            for n in N_SIZES:
                d = results[k][family].get(n, {})
                pc = d.get("py_cold", float("nan"))
                rg = d.get("r_gam", float("nan"))
                rb = d.get("r_bam", float("nan"))
                pi = d.get("py_iter", -1)
                gi = d.get("r_gam_iter", -1)
                bi = d.get("r_bam_iter", -1)

                cold_vs_gam = _ratio(pc, rg)
                cold_vs_bam = _ratio(pc, rb)

                print(
                    f"| {k:>4} | {family:<8} | {n:>7,} "
                    f"| {_fmt(pc):>8} "
                    f"| {_fmt(rg):>8} | {_fmt(rb):>8} "
                    f"| {cold_vs_gam:>7} | {cold_vs_bam:>7} "
                    f"| {pi:>5} | {gi:>6} | {bi:>6} |"
                )
    print()

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute geometric mean across families for each k
    bar_data = {}  # k -> (py_cold, r_gam, r_bam)
    for k in BASIS_DIMS:
        pc_vals, rg_vals, rb_vals = [], [], []
        for family in FAMILIES:
            for n in N_SIZES:
                d = results[k][family].get(n, {})
                pc = d.get("py_cold", float("nan"))
                rg = d.get("r_gam", float("nan"))
                rb = d.get("r_bam", float("nan"))
                if not any(np.isnan(v) for v in [pc, rg, rb]):
                    pc_vals.append(pc)
                    rg_vals.append(rg)
                    rb_vals.append(rb)
        if pc_vals:
            bar_data[k] = (
                geometric_mean(pc_vals),
                geometric_mean(rg_vals),
                geometric_mean(rb_vals),
            )

    x_labels = [f"k={k}" for k in bar_data]
    x = np.arange(len(x_labels))
    width = 0.25

    py_vals = [bar_data[k][0] for k in bar_data]
    rg_vals = [bar_data[k][1] for k in bar_data]
    rb_vals = [bar_data[k][2] for k in bar_data]

    bars_py = ax.bar(
        x - width,
        py_vals,
        width,
        label="jaxgam (true cold)",
        color="#1f77b4",
    )
    bars_rg = ax.bar(
        x,
        rg_vals,
        width,
        label="R gam(REML)",
        color="#d62728",
        alpha=0.7,
    )
    bars_rb = ax.bar(
        x + width,
        rb_vals,
        width,
        label=f"R bam(fREML, {BAM_NTHREADS}t)",
        color="#d62728",
    )

    # Add time labels on bars
    for bars in [bars_py, bars_rg, bars_rb]:
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
        f"jaxgam (true cold) vs R mgcv at n={N_SIZES[0]:,}\n"
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
