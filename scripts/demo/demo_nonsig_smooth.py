"""Demo: GAM with one significant and one non-significant (pure noise) smooth.

Tests whether summary() correctly reports a high p-value for the noise term.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymgcv.api import GAM

rng = np.random.default_rng(123)
n = 500

# x1: true signal — sin wave
# x2: pure noise — no relationship with y
x1 = rng.uniform(0, 1, n)
x2 = rng.uniform(0, 1, n)
eta = 1.0 + 2.0 * np.sin(2 * np.pi * x1)  # only x1 matters
y = eta + rng.normal(0, 1.0, n)
data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

print(f"Data: n={n}")
print("True DGP: y = 1 + 2*sin(2*pi*x1) + N(0,1)")
print("x2 is pure noise — should be non-significant\n")

# Fit with both smooths
model = GAM("y ~ s(x1, k=10) + s(x2, k=10)", family="gaussian").fit(data)

# Summary
s = model.summary()

# Check p-values
print("\n--- Verification ---")
for i, name in enumerate(s.s_names):
    edf = s.s_table[i, 0]
    pval = s.s_table[i, 3]
    sig = "SIGNIFICANT (p < 0.05)" if pval < 0.05 else "NOT significant (p >= 0.05)"
    print(f"  {name}: EDF={edf:.3f}, p={pval:.4f} -> {sig}")

# Also compare to R
try:
    from pymgcv.compat.r_bridge import RBridge

    rb = RBridge()
    r_summary = rb.summary_gam(
        "y ~ s(x1, k=10) + s(x2, k=10)",
        data,
        "gaussian",
    )
    print("\n--- R comparison ---")
    r_s_table = r_summary["s_table"]
    for i, name in enumerate(s.s_names):
        r_pval = r_s_table[i, 3]
        py_pval = s.s_table[i, 3]
        print(f"  {name}: Python p={py_pval:.6f}, R p={r_pval:.6f}")
except Exception as e:
    print(f"\n(R comparison skipped: {e})")

# Plot
fig, axes = model.plot(pages=1, shade=True, rug=True, se=True)
fig.suptitle("y ~ s(x1) + s(x2)  [x2 is pure noise]", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("scripts/demo/nonsig_smooth.png", dpi=150, bbox_inches="tight")
print("\nSaved: scripts/demo/nonsig_smooth.png")
plt.close(fig)
