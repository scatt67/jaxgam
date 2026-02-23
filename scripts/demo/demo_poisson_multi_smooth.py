"""Demo script: fit Poisson GAMs and save plots as PNG files.

Models:
1. Multi-smooth additive:  y ~ s(x1) + s(x2)
2. Tensor product:         y ~ te(x1, x2)
3. 2D TPRS:                y ~ s(x1, x2)
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymgcv.api import GAM

OUT = "scripts/demo"

# ── Generate Poisson data with 2D interaction surface ─────────────────
rng = np.random.default_rng(42)
n = 500
x1 = rng.uniform(0, 1, n)
x2 = rng.uniform(0, 1, n)
eta = np.sin(2 * np.pi * x1) + 0.8 * x2 + 0.5 * np.cos(3 * x1 * x2)
mu = np.exp(eta)
y = rng.poisson(mu).astype(float)
data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

print(f"Data: n={n}, y range=[{y.min():.0f}, {y.max():.0f}], mean={y.mean():.2f}")


def fit_and_plot(formula, label, tag):
    """Fit a GAM, print summary, save plot via model.plot()."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  {formula}")
    print(f"{'=' * 60}")

    model = GAM(formula, family="poisson", method="REML")
    model.fit(data)

    print(f"Converged: {model.converged_} in {model.n_iter_} iterations")
    print(f"Deviance:  {model.deviance_:.4f}")
    print(f"EDF:       {model.edf_}")
    model.summary()

    fig, axes = model.plot(pages=1, shade=True, rug=True, se=True)
    fig.suptitle(f"Poisson GAM: {formula}", fontsize=12, y=1.02)
    fig.tight_layout()
    fname = f"{OUT}/poisson_{tag}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)

    return model


# ── Model 1: Additive multi-smooth ───────────────────────────────────
fit_and_plot(
    "y ~ s(x1, k=10, bs='cr') + s(x2, k=10, bs='cr')",
    "Model 1: Additive multi-smooth (1D + 1D)",
    "multi_smooth",
)

# ── Model 2: Tensor product ──────────────────────────────────────────
fit_and_plot(
    "y ~ te(x1, x2, k=8)",
    "Model 2: Tensor product te(x1, x2)",
    "tensor_product",
)

# ── Model 3: 2D TPRS ─────────────────────────────────────────────────
fit_and_plot(
    "y ~ s(x1, x2, k=25)",
    "Model 3: 2D TPRS s(x1, x2)",
    "tprs_2d",
)

print("\nDone — all plots saved.")
