"""Plot true-cold jaxgam/R speedup ratio vs dataset size.

Usage::

    uv run python scripts/plot_speedup_vs_n.py

Clears the JAX persistent compilation cache, then runs one fit per
(smooth, family, n) combination — measuring the absolute worst-case
first-ever-fit time including full JIT tracing + XLA compilation.

Produces ``scripts/speedup_vs_n.png``.
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

# Unbuffered print so background runs show progress
print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SIZES = [500, 5_000, 10_000, 20_000, 50_000, 100_000, 500_000]

FAMILIES = ["gaussian", "poisson", "binomial", "gamma"]

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
# Data generators (same as benchmark_vs_r.py)
# ---------------------------------------------------------------------------


def _make_response(
    eta: np.ndarray, family: str, rng: np.random.Generator
) -> np.ndarray:
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
# R timing
# ---------------------------------------------------------------------------


def setup_r():
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    importr("mgcv")
    return ro


def time_r_fit(ro, r_formula, data, family, nthreads=1):
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

    ctrl = f", control=list(nthreads={nthreads})" if nthreads > 1 else ""
    r_code = f"""
    tm <- system.time({{
        mod <- gam({r_formula}, data=bench_data,
                    family={family_map[family]}, method="REML"{ctrl})
    }})
    tm["elapsed"] * 1000
    """
    return float(np.array(ro.r(r_code))[0])


# ---------------------------------------------------------------------------
# Python timing
# ---------------------------------------------------------------------------


def time_py_fit(py_formula, data, family):
    model = GAM(py_formula, family=family, method="REML")
    t0 = time.perf_counter()
    model.fit(data)
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Check R
    try:
        import rpy2.robjects  # noqa: F401

        has_r = True
    except Exception:
        has_r = False

    if not has_r:
        print("ERROR: rpy2/R required for speedup plot.")
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

    # Storage: results[smooth][family][n] = {"py": ms, "r": ms}
    results: dict[str, dict[str, dict[int, dict]]] = {
        s: {f: {} for f in FAMILIES} for s in SMOOTH_CONFIGS
    }

    total = len(SMOOTH_CONFIGS) * len(FAMILIES) * len(N_SIZES)
    done = 0

    # --- Python true cold (run first, before any warmup) ---
    print("Python true-cold benchmark...")
    for n in N_SIZES:
        for smooth_key, cfg in SMOOTH_CONFIGS.items():
            data_type = cfg["data_type"]
            maker = DATA_MAKERS[data_type]
            for family in FAMILIES:
                done += 1
                data = maker(n, family)
                try:
                    ms = time_py_fit(cfg["py_formula"], data, family)
                    results[smooth_key][family][n] = {"py": ms}
                    print(
                        f"  [{done}/{total}] {smooth_key}/{family}"
                        f"/n={n:>9,}: {ms:>9.0f}ms"
                    )
                except Exception as e:
                    results[smooth_key][family][n] = {"py": float("nan")}
                    print(
                        f"  [{done}/{total}] {smooth_key}/{family}"
                        f"/n={n:>9,}: FAILED ({e})"
                    )
    print()

    # --- R benchmark (single-threaded) ---
    print("R benchmark (single-threaded)...")
    done = 0
    for n in N_SIZES:
        for smooth_key, cfg in SMOOTH_CONFIGS.items():
            data_type = cfg["data_type"]
            maker = DATA_MAKERS[data_type]
            for family in FAMILIES:
                done += 1
                data = maker(n, family)
                try:
                    ms = time_r_fit(
                        ro,
                        cfg["r_formula"],
                        data,
                        family,
                    )
                    results[smooth_key][family][n]["r"] = ms
                    print(
                        f"  [{done}/{total}] {smooth_key}/{family}"
                        f"/n={n:>9,}: {ms:>9.0f}ms"
                    )
                except Exception as e:
                    results[smooth_key][family][n]["r"] = float("nan")
                    print(
                        f"  [{done}/{total}] {smooth_key}/{family}"
                        f"/n={n:>9,}: FAILED ({e})"
                    )
    print()

    # --- Print summary table ---
    print("=" * 80)
    print("TRUE COLD SPEEDUP (R_ms / py_ms)")
    print("=" * 80)
    header = f"| {'smooth':<6} | {'family':<8} |"
    for n in N_SIZES:
        header += f" {n:>9,} |"
    print(header)
    print("|" + "-" * 8 + "|" + "-" * 10 + "|" + ("-" * 12 + "|") * len(N_SIZES))

    for smooth_key in SMOOTH_CONFIGS:
        for family in FAMILIES:
            row = f"| {smooth_key:<6} | {family:<8} |"
            for n in N_SIZES:
                d = results[smooth_key][family].get(n, {})
                py = d.get("py", float("nan"))
                r = d.get("r", float("nan"))
                if np.isnan(py) or np.isnan(r) or py <= 0:
                    row += f" {'N/A':>9} |"
                else:
                    ratio = r / py
                    row += f" {ratio:>8.1f}x |"
            print(row)
    print()

    # --- Build plot ---
    smooth_labels = {
        "cr": "Single smooth (cr)",
        "two": "Two smooths",
        "te": "Tensor product",
        "cr_by": "Factor-by",
    }
    colors = {
        "cr": "#1f77b4",
        "two": "#ff7f0e",
        "te": "#2ca02c",
        "cr_by": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for smooth_key in SMOOTH_CONFIGS:
        gmeans = []
        mins = []
        maxs = []
        valid_ns = []

        for n in N_SIZES:
            ratios = []
            for family in FAMILIES:
                d = results[smooth_key][family].get(n, {})
                py = d.get("py", float("nan"))
                r = d.get("r", float("nan"))
                if not (np.isnan(py) or np.isnan(r) or py <= 0):
                    ratios.append(r / py)

            if ratios:
                gmeans.append(geometric_mean(ratios))
                mins.append(min(ratios))
                maxs.append(max(ratios))
                valid_ns.append(n)

        if valid_ns:
            color = colors[smooth_key]
            ax.plot(
                valid_ns,
                gmeans,
                "o-",
                color=color,
                label=smooth_labels[smooth_key],
                linewidth=2,
                markersize=5,
            )
            ax.fill_between(
                valid_ns,
                mins,
                maxs,
                alpha=0.15,
                color=color,
            )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Dataset size (n)", fontsize=12)
    ax.set_ylabel("Speedup (R / jaxgam)", fontsize=12)
    ax.set_title(
        "True cold-start jaxgam vs R mgcv\n(no JIT cache; R single-threaded)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Label y-axis reference points
    ax.text(
        N_SIZES[0] * 0.7,
        1.0,
        "parity",
        fontsize=9,
        color="gray",
        va="bottom",
    )

    fig.tight_layout()
    out_path = "scripts/speedup_vs_n.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
