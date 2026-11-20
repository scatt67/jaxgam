"""Shared test fixtures and configuration.

Provides:
- Legacy simple_*_data fixtures (n=200, two-predictor)
- New data-generating fixtures with indirect parametrisation
- R bridge fixture
- Minimal committed checkout for isolated benchmark-harness children
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers import SEED, N, _generate_family_data
from tests.r_bridge import RBridge
from tests.sharding import SHARD_ENV, parse_shard

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
def re_factor_data():
    """Single-factor data for RE smooth tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    n_groups = 20
    g = rng.choice([f"g{i}" for i in range(n_groups)], size=n)
    return {
        "g": pd.Series(pd.Categorical(g)),
    }


@pytest.fixture
def re_two_factor_data():
    """Two-factor data for RE interaction tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    g1 = rng.choice(["a", "b", "c"], size=n)
    g2 = rng.choice(["x", "y"], size=n)
    return {
        "g1": pd.Series(pd.Categorical(g1)),
        "g2": pd.Series(pd.Categorical(g2)),
    }


@pytest.fixture
def re_numeric_factor_data():
    """Numeric x factor data for random slope tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    x = rng.uniform(0, 1, n)
    g = rng.choice([f"g{i}" for i in range(10)], size=n)
    return {
        "x": x,
        "g": pd.Series(pd.Categorical(g)),
    }


@pytest.fixture
def re_model_data():
    """DataFrame with x, g, and y for RE integration tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    n_groups = 20
    x = rng.uniform(0, 1, n)
    g = rng.choice([f"g{i}" for i in range(n_groups)], size=n)
    group_effects = rng.normal(0, 0.3, n_groups)
    g_idx = np.array([int(gi[1:]) for gi in g])
    y = np.sin(2 * np.pi * x) + group_effects[g_idx] + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "g": pd.Categorical(g), "y": y})


@pytest.fixture
def gp_1d_data():
    """Univariate continuous data for GP smooth tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    return {"x": rng.uniform(0, 1, n)}


@pytest.fixture
def gp_2d_data():
    """Bivariate continuous data for 2D GP tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    return {"x": rng.uniform(0, 1, n), "z": rng.uniform(0, 1, n)}


@pytest.fixture
def gp_explicit_knots_data():
    """Small dataset with explicit knot positions for STRICT R comparison."""
    knots = np.linspace(0.05, 0.95, 10)
    x = np.tile(knots, 10)
    return {
        "data": pd.DataFrame({"x": x}),
        "knots": {"x": knots},
    }


@pytest.fixture
def gp_te_2d_data() -> pd.DataFrame:
    """Ordered bivariate data for GP tensor-margin R comparisons."""
    n = 30
    return pd.DataFrame(
        {
            "x1": np.linspace(0.05, 0.95, n),
            "x2": np.linspace(0.15, 0.85, n),
        }
    )


@pytest.fixture
def large_gp_data():
    """Univariate GP data with more unique rows than the default max_knots."""
    rng = np.random.default_rng(SEED)
    n = 2500
    return {"x": rng.uniform(0, 1, n)}


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


def pytest_collection_modifyitems(config, items):
    """Keep only this shard's tests when ``JAXGAM_TEST_SHARD=i/N`` is set.

    CI runs the suite as ``N`` such shards on parallel runners
    (``make test-cov-shard-ci``); together they run every test exactly once.
    """
    shard = parse_shard(os.environ.get(SHARD_ENV))
    if shard is None:
        return
    kept, dropped = [], []
    for item in items:
        (kept if shard.owns(item.nodeid) else dropped).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept


def pytest_report_header(config):
    """Print the shard, and the CPU and kernel identity for platform-sensitive runs."""
    lines = []
    shard = parse_shard(os.environ.get(SHARD_ENV))
    if shard is not None:
        lines.append(f"shard {shard.index} of {shard.count} ({SHARD_ENV})")
    if "platform_sensitive" in (config.option.markexpr or ""):
        from tests.helpers import platform_report_lines

        lines.append("platform-sensitive run:")
        lines.extend(f"  {line}" for line in platform_report_lines())
    return lines or None


def pytest_terminal_summary(terminalreporter):
    """Print every replay-bound check of the run with its measured count.

    Reviewed replay bounds are per platform class, and x86_64 entries are
    filled from CI's measured counts, so a run reports them even when every
    check passed.
    """
    from tests.helpers import REPLAY_COUNT_RECORDS, cpu_platform_class

    if not REPLAY_COUNT_RECORDS:
        return
    terminalreporter.section(f"replay counts (platform class {cpu_platform_class()})")
    for table, key, platform_class, measured, bound in REPLAY_COUNT_RECORDS:
        reviewed = "no reviewed entry" if bound is None else f"bound {bound}"
        terminalreporter.write_line(
            f"{table} {key} [{platform_class}]: measured {measured}, {reviewed}"
        )


@pytest.fixture
def minimal_git_checkout(tmp_path: Path) -> Path:
    """Build the smallest real checkout needed by isolated harness children.

    The Docker test image deliberately excludes the repository's ``.git``
    directory. The harness must still fail closed when provenance cannot be
    obtained, so real-child tests run against this copied, committed checkout
    rather than mocking source identity or relaxing the production helper.
    """
    checkout = tmp_path / "checkout"
    ignore_bytecode = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(Path.cwd() / "jaxgam", checkout / "jaxgam", ignore=ignore_bytecode)
    (checkout / "tests").mkdir()
    for name in ("__init__.py", "tolerances.py"):
        shutil.copy2(Path.cwd() / "tests" / name, checkout / "tests" / name)
    (checkout / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    subprocess.run(["git", "init", "--quiet"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "add", ".gitignore", "jaxgam", "tests"], cwd=checkout, check=True
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=benchmark-test",
            "-c",
            "user.email=benchmark-test@example.com",
            "commit",
            "--quiet",
            "-m",
            "minimal isolated harness checkout",
        ],
        cwd=checkout,
        check=True,
    )
    return checkout
