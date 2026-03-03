"""Shared test helpers.

Public API (used directly by tests):
- SEED, N — constants
- r_available() — import-time R check for @pytest.mark.skipif
- r_tolerance() — tolerance tier by family
- make_smooth_spec() — SmoothSpec factory with many call-site variants

Private API (used by conftest fixtures and complex test-local fixtures):
- _generate_family_data() — single-predictor family data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jaxgam.formula.terms import SmoothSpec
from tests.tolerances import LOOSE, MODERATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N = 200


# ---------------------------------------------------------------------------
# R bridge helpers
# ---------------------------------------------------------------------------


def r_available() -> bool:
    """Check if R and mgcv are available with correct versions.

    Safe to call at import time (used in ``@pytest.mark.skipif``).
    """
    try:
        from tests.r_bridge import RBridge

        if not RBridge.available():
            return False
        ok, _ = RBridge.check_versions()
        return ok
    except Exception:
        return False


def r_tolerance(family_name: str):
    """Return tolerance tier for R comparison by family.

    Gaussian: MODERATE (single PIRLS iteration, no compounding).
    GLM families: LOOSE (iterative PIRLS + Newton, differences compound).
    """
    if family_name == "gaussian":
        return MODERATE
    return LOOSE


# ---------------------------------------------------------------------------
# SmoothSpec factory
# ---------------------------------------------------------------------------


def make_smooth_spec(
    variables: list[str],
    bs: str = "tp",
    k: int = 10,
    by: str | None = None,
    smooth_type: str = "s",
    **extra_args: object,
) -> SmoothSpec:
    """Create a ``SmoothSpec`` for testing."""
    return SmoothSpec(
        variables=variables,
        bs=bs,
        k=k,
        by=by,
        smooth_type=smooth_type,
        extra_args=dict(extra_args),
    )


# ---------------------------------------------------------------------------
# Private data generators (used by conftest and test-local fixtures)
# ---------------------------------------------------------------------------


def _generate_family_data(family_name: str, n: int | None = None) -> pd.DataFrame:
    """Generate single-predictor synthetic data for a given family.

    Parameters
    ----------
    family_name : str
        One of "gaussian", "binomial", "poisson", "gamma".
    n : int or None
        Sample size.  If ``None``, defaults to 200 (300 for binomial).
    """
    rng = np.random.default_rng(SEED)
    if n is None:
        n = 200 if family_name != "binomial" else 300
    x = rng.uniform(0, 1, n)

    if family_name == "gaussian":
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        eta = 2 * np.sin(2 * np.pi * x)
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        eta = np.sin(2 * np.pi * x) + 0.5
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family_name == "gamma":
        eta = 0.5 * np.sin(2 * np.pi * x) + 1.0
        mu = np.exp(eta)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "y": y})
