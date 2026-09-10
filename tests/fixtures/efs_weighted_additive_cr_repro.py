"""Pinned EFS weighted-offset additive cubic-regression fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd

FORMULA = "y ~ s(x, bs='cr', k=6) + s(z, bs='cr', k=5)"
SEED = 2026
N = 64


def make_data() -> pd.DataFrame:
    """Return the exact weighted-offset Poisson fixture."""
    rng = np.random.default_rng(SEED)
    x = np.linspace(-1.0, 1.0, N)
    z = rng.uniform(-1.0, 1.0, N)
    offset = rng.normal(0.0, 0.05, N)
    eta = 0.2 + 0.5 * np.sin(3.0 * x) - 0.2 * z + offset
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame(
        {"y": y, "x": x, "z": z, "w": 0.5 + rng.random(N), "off": offset}
    )
