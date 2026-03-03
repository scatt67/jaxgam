"""Shared test fixtures and configuration.

Provides:
- Legacy simple_*_data fixtures (n=200, two-predictor)
- New data-generating fixtures with indirect parametrisation
- R bridge fixture
"""

import numpy as np
import pandas as pd
import pytest

from tests.helpers import SEED, N, _generate_family_data
from tests.r_bridge import RBridge

# ---------------------------------------------------------------------------
# Legacy simple_*_data fixtures (unchanged)
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_gaussian_data() -> pd.DataFrame:
    """Synthetic Gaussian response data (n=200).

    Generating process: y = sin(2*pi*x1) + 0.5*x2 + N(0, 0.25)
    """
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    f1 = np.sin(2 * np.pi * x1)
    f2 = 0.5 * x2
    y = f1 + f2 + rng.normal(0, 0.5, N)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def simple_binomial_data() -> pd.DataFrame:
    """Synthetic binomial response data (n=200).

    Generating process: logit(p) = 2*sin(2*pi*x1) + x2 - 1, y ~ Bernoulli(p)
    """
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    eta = 2 * np.sin(2 * np.pi * x1) + x2 - 1
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p, N).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def simple_poisson_data() -> pd.DataFrame:
    """Synthetic Poisson response data (n=200).

    Generating process: log(mu) = sin(2*pi*x1) + 0.5*x2, y ~ Poisson(mu)
    """
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    eta = np.sin(2 * np.pi * x1) + 0.5 * x2
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def simple_gamma_data() -> pd.DataFrame:
    """Synthetic Gamma response data (n=200).

    Generating process:
        log(mu) = 0.5*sin(2*pi*x1) + 0.3*x2 + 1
        y ~ Gamma(shape=5, scale=mu/5)
    """
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    eta = 0.5 * np.sin(2 * np.pi * x1) + 0.3 * x2 + 1
    mu = np.exp(eta)
    shape = 5.0
    y = rng.gamma(shape, scale=mu / shape, size=N)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


# ---------------------------------------------------------------------------
# R bridge
# ---------------------------------------------------------------------------


@pytest.fixture
def r_bridge():
    """Fixture providing R bridge for reference comparison.

    Skips the test if R/mgcv is not available or version mismatch.
    """
    if not RBridge.available():
        pytest.skip("R with mgcv not available")
    ok, reason = RBridge.check_versions()
    if not ok:
        pytest.skip(f"R version mismatch: {reason}. Use: make test")
    return RBridge()


# ---------------------------------------------------------------------------
# New data-generating fixtures (indirect parametrisation)
# ---------------------------------------------------------------------------


@pytest.fixture
def smooth_1d_data(request):
    """1D smooth data {x: array}. Indirect param = n (default 200)."""
    n = getattr(request, "param", N)
    rng = np.random.default_rng(SEED)
    return {"x": rng.uniform(0, 1, n)}


@pytest.fixture
def smooth_2d_data(request):
    """2D smooth data {x1, x2}. Indirect param = n (default 200)."""
    n = getattr(request, "param", N)
    rng = np.random.default_rng(SEED)
    return {"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)}


@pytest.fixture
def smooth_3d_data(request):
    """3D smooth data {x1, x2, x3}. Indirect param = n (default 200)."""
    n = getattr(request, "param", N)
    rng = np.random.default_rng(SEED)
    return {
        "x1": rng.uniform(0, 1, n),
        "x2": rng.uniform(0, 1, n),
        "x3": rng.uniform(0, 1, n),
    }


@pytest.fixture
def pred_smooth_1d_data(request):
    """Prediction data disjoint from training (burns N draws).

    Indirect param = n (default 50).
    """
    n = getattr(request, "param", 50)
    rng = np.random.default_rng(SEED)
    _ = rng.uniform(0, 1, N)  # burn training draws
    return {"x": rng.uniform(0, 1, n)}


@pytest.fixture
def pred_smooth_2d_data(request):
    """2D prediction data disjoint from training.

    Indirect param = n (default 50).
    """
    n = getattr(request, "param", 50)
    rng = np.random.default_rng(SEED)
    _ = rng.uniform(0, 1, N * 2)  # burn training draws (2 dims)
    return {"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)}


@pytest.fixture
def two_smooth_data():
    """Two-predictor additive data (x1, x2, y) DataFrame."""
    rng = np.random.default_rng(SEED)
    n = 200
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def factor_by_data():
    """Factor-by data (x, fac, y) DataFrame.

    Uses pd.Categorical so rpy2 converts to R factor correctly.
    """
    rng = np.random.default_rng(SEED)
    n = 300
    x = rng.uniform(0, 1, n)
    levels = ["a", "b", "c"]
    fac = rng.choice(levels, n)
    eta = np.where(
        fac == "a",
        np.sin(2 * np.pi * x),
        np.where(fac == "b", 0.5 * x, -0.3 * x),
    )
    y = eta + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "x": x,
            "fac": pd.Categorical(fac, categories=levels),
            "y": y,
        }
    )


@pytest.fixture
def family_data(request):
    """Single-predictor family data.

    Indirect param = family_name or (family_name, n).
    Returns (family_name, DataFrame).
    """
    param = request.param
    if isinstance(param, tuple):
        family_name, n = param
    else:
        family_name, n = param, None
    return family_name, _generate_family_data(family_name, n=n)
