"""Demo script: fit a Negative Binomial GAM and save plots as PNG.

Simulates overdispersed count data with known theta, fits with
estimated theta, and compares the estimate to the true value.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxgam.api import GAM

OUT = "scripts/demo"

# ── Generate NB count data with known theta ──────────────────────────
TRUE_THETA = 2.0
rng = np.random.default_rng(42)
n = 500
x = rng.uniform(0, 1, n)
eta = np.sin(2 * np.pi * x) + 0.5
mu = np.exp(eta)
y = rng.negative_binomial(
    n=TRUE_THETA, p=TRUE_THETA / (mu + TRUE_THETA), size=n
).astype(float)
data = pd.DataFrame({"x": x, "y": y})

print(f"Data: n={n}, y range=[{y.min():.0f}, {y.max():.0f}], mean={y.mean():.2f}")
print(f"True theta: {TRUE_THETA}")

# ── Fit NB GAM with estimated theta ─────────────────────────────────
formula = "y ~ s(x, k=10, bs='cr')"
print(f"\nFitting: {formula}, family='nb'")

model = GAM(formula, family="nb", method="REML")
results = model.fit(data)

est_theta = results.theta
print(f"\nConverged: {results.converged} in {results.n_iter} iterations")
print(f"Estimated theta: {est_theta:.3f}  (true: {TRUE_THETA})")
print(f"Deviance: {results.deviance:.4f}")
results.summary()

# ── Plot 1: smooth component via results.plot() ─────────────────────
fig_smooth, axes_smooth = results.plot(pages=1, shade=True, rug=True, se=True)
fig_smooth.suptitle(f"NB GAM smooth component: {formula}", fontsize=12, y=1.02)
fig_smooth.tight_layout()
fname1 = f"{OUT}/nb_smooth.png"
fig_smooth.savefig(fname1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {fname1}")
plt.close(fig_smooth)

# ── Plot 2: observed counts vs fitted curve ──────────────────────────
x_grid = np.linspace(0, 1, 200)
newdata = pd.DataFrame({"x": x_grid})
pred = results.predict(newdata)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, alpha=0.3, s=10, color="gray", label="Observed")
ax.plot(x_grid, pred, color="#1f77b4", linewidth=2, label="Fitted mean")
ax.set_xlabel("x")
ax.set_ylabel("y (count)")
ax.set_title(
    f"NB GAM: observed vs fitted\n"
    rf"$\hat{{\theta}}$ = {est_theta:.2f} (true = {TRUE_THETA})"
)
ax.legend()
fig.tight_layout()
fname2 = f"{OUT}/nb_demo.png"
fig.savefig(fname2, dpi=150, bbox_inches="tight")
print(f"Saved: {fname2}")
plt.close(fig)

print("\nDone — all plots saved.")
