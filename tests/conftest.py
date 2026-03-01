"""Shared test fixtures and configuration.

Provides small synthetic datasets (n=200) for each v1.0 family,
generated with a known process and fixed seed for reproducibility.
"""

import numpy as np
import pandas as pd
import pytest

from tests.r_bridge import RBridge

SEED = 42
N = 200


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
