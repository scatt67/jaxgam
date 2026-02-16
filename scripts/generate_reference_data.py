"""Generate reference data from R's mgcv for testing.

Usage:
  uv run python scripts/generate_reference_data.py          # skip existing
  uv run python scripts/generate_reference_data.py --force   # regenerate all

Produces NPZ files in tests/reference_data/ for the 32-cell validation surface:
  3 smooth types (tp, cr, cc) x 4 families (gaussian, binomial, poisson, gamma)
  + tensor products (te, ti) x 4 families
  + factor-by smooths (tp, cr, cc) x 4 families

Each NPZ file contains:
  coefficients, fitted_values, smoothing_params, edf, Vp  (arrays)
  deviance, scale, reml_score  (scalars as 0-d arrays)
  basis_0, basis_1, ...  (per-smooth basis matrices)
  pen_0_0, pen_0_1, ...  (per-smooth penalty matrices)

Shared covariate data is saved as data_{family}.npz files.

Requires R with mgcv installed (accessed via RBridge).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure pymgcv is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymgcv.compat.r_bridge import RBridge

OUTPUT_DIR = Path("tests/reference_data")

FAMILIES = ["gaussian", "binomial", "poisson", "gamma"]
SMOOTH_TYPES = ["tp", "cr", "cc"]
TENSOR_TYPES = ["te", "ti"]


def generate_data(bridge: RBridge) -> dict[str, pd.DataFrame]:
    """Generate synthetic data matching the R script (set.seed(42)).

    We use R to generate the data so the random seed matches exactly.
    """
    script = """\
set.seed(42)
n <- 200
x1 <- runif(n)
x2 <- runif(n)
fac <- factor(sample(c("a", "b", "c"), n, replace = TRUE))

f1 <- sin(2 * pi * x1)
f2 <- 0.5 * x2

data_gaussian <- data.frame(x1 = x1, x2 = x2, fac = fac)
data_gaussian$y <- f1 + f2 + rnorm(n, 0, 0.5)

data_binomial <- data.frame(x1 = x1, x2 = x2, fac = fac)
eta_bin <- 2 * f1 + f2 - 0.5
data_binomial$y <- rbinom(n, 1, plogis(eta_bin))

data_poisson <- data.frame(x1 = x1, x2 = x2, fac = fac)
eta_pois <- f1 + 0.5 * f2 + 0.5
data_poisson$y <- rpois(n, exp(eta_pois))

data_gamma <- data.frame(x1 = x1, x2 = x2, fac = fac)
eta_gam <- 0.5 * f1 + 0.3 * f2 + 1
mu_gam <- exp(eta_gam)
data_gamma$y <- rgamma(n, shape = 5, scale = mu_gam / 5)
"""
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        script_full = script
        for fam in FAMILIES:
            csv_path = os.path.join(tmpdir, f"data_{fam}.csv")
            script_full += f'write.csv(data_{fam}, "{csv_path}", row.names = FALSE)\n'

        script_path = os.path.join(tmpdir, "gen_data.R")
        with open(script_path, "w") as f:
            f.write(script_full)

        proc = subprocess.run(
            ["Rscript", script_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"R data generation failed:\n{proc.stderr}")

        datasets = {}
        for fam in FAMILIES:
            df = pd.read_csv(os.path.join(tmpdir, f"data_{fam}.csv"))
            # Convert string columns to Categorical so rpy2 maps them to R factors
            if "fac" in df.columns:
                df["fac"] = pd.Categorical(df["fac"])
            datasets[fam] = df

    return datasets


def load_cached_data() -> dict[str, pd.DataFrame] | None:
    """Load data NPZ files if all four exist. Returns None otherwise."""
    datasets = {}
    for fam in FAMILIES:
        path = OUTPUT_DIR / f"data_{fam}.npz"
        if not path.exists():
            return None
        loaded = np.load(path, allow_pickle=True)
        df = pd.DataFrame({k: loaded[k] for k in loaded.files})
        if "fac" in df.columns:
            df["fac"] = pd.Categorical(df["fac"])
        datasets[fam] = df
    return datasets


def save_model_npz(result: dict, name: str, output_dir: Path) -> None:
    """Save a model result dict as a single .npz file."""
    arrays = {
        "coefficients": result["coefficients"],
        "fitted_values": result["fitted_values"],
        "smoothing_params": result["smoothing_params"],
        "edf": result["edf"],
        "Vp": result["Vp"],
        "deviance": np.array(result["deviance"]),
        "scale": np.array(result["scale"]),
        "reml_score": np.array(result["reml_score"]),
    }

    # Per-smooth basis and penalty matrices
    for i, basis in enumerate(result["basis_matrices"]):
        arrays[f"basis_{i}"] = basis
    for i, penalties in enumerate(result["penalty_matrices"]):
        for j, pen in enumerate(penalties):
            arrays[f"pen_{i}_{j}"] = pen

    np.savez_compressed(output_dir / f"{name}.npz", **arrays)


def build_model_specs() -> list[tuple[str, str, bool]]:
    """Return list of (name, formula, uses_factor) for all 32 models."""
    models: list[tuple[str, str, bool]] = []

    for bs in SMOOTH_TYPES:
        for fam in FAMILIES:
            name = f"{bs}_{fam}"
            formula = f"y ~ s(x1, bs='{bs}', k=10) + s(x2, bs='{bs}', k=10)"
            models.append((name, formula, False))

    for tt in TENSOR_TYPES:
        for fam in FAMILIES:
            name = f"{tt}_{fam}"
            if tt == "te":
                formula = "y ~ te(x1, x2, k=c(5, 5))"
            else:
                formula = "y ~ s(x1, k=10) + s(x2, k=10) + ti(x1, x2, k=c(5, 5))"
            models.append((name, formula, False))

    for bs in SMOOTH_TYPES:
        for fam in FAMILIES:
            name = f"by_{bs}_{fam}"
            formula = f"y ~ fac + s(x1, by=fac, bs='{bs}', k=10)"
            models.append((name, formula, True))

    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate R mgcv reference data for testing."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all files even if they already exist.",
    )
    args = parser.parse_args()

    if not RBridge.available():
        print("ERROR: R with mgcv is not available.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = build_model_specs()

    # Check which models still need fitting
    if args.force:
        pending = models
    else:
        pending = [m for m in models if not (OUTPUT_DIR / f"{m[0]}.npz").exists()]

    if not pending:
        print("All reference data already exists. Use --force to regenerate.")
        return

    # We need data for any pending model — try cached first
    datasets = None if args.force else load_cached_data()

    if datasets is None:
        bridge = RBridge()
        print(f"Using RBridge mode: {bridge.mode}")
        print("Generating synthetic data via R (set.seed(42))...")
        datasets = generate_data(bridge)
        for fam, df in datasets.items():
            arrays = {col: df[col].values for col in df.columns}
            np.savez_compressed(OUTPUT_DIR / f"data_{fam}.npz", **arrays)
        print(f"  Saved data for {len(datasets)} families")
    else:
        bridge = RBridge()
        print(f"Using RBridge mode: {bridge.mode}")
        print("Loaded cached data files.")

    n_skipped = len(models) - len(pending)
    if n_skipped:
        print(f"Skipping {n_skipped} models (already exist).\n")

    print(f"Fitting {len(pending)} model(s)...\n")
    n_success = 0
    n_fail = 0

    for name, formula, _uses_factor in pending:
        fam = name.split("_")[-1]
        data = datasets[fam]

        print(f"  {name}: {formula}")
        try:
            result = bridge.get_smooth_components(
                formula, data, family=fam, method="REML"
            )
            save_model_npz(result, name, OUTPUT_DIR)
            n_success += 1
            print(f"    -> saved {name}.npz")
        except Exception as e:
            n_fail += 1
            print(f"    -> FAILED: {e}")

    print(f"\nDone: {n_success} fitted, {n_skipped} cached, {n_fail} failed.")
    print(f"Reference data: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
