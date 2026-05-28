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

from collections.abc import Callable

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
# Assertion collection
# ---------------------------------------------------------------------------


class _AssertCollector:
    """Collect assertion failures across related checks."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, fn: Callable[[], None]) -> None:
        """Run one assertion block and retain its failure message."""
        try:
            fn()
        except AssertionError as exc:
            self.failures.append(f"{name}: {exc}")

    def raise_if_any(self, label: str) -> None:
        """Raise one readable assertion if any collected check failed."""
        if self.failures:
            raise AssertionError(
                f"{label} failed:\n  - " + "\n  - ".join(self.failures)
            )


def check_that(condition: bool, message: str) -> None:
    """Boolean assertion usable inside a lambda passed to ``_AssertCollector``.

    Lambdas cannot contain ``assert`` statements, so non-numeric checks
    (``converged``, ``mu > 0``, ``theta is not None``) raise ``AssertionError``
    through this helper instead of ``np.testing.assert_`` (which ruff flags).
    """
    if not condition:
        raise AssertionError(message)


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
# FittingData construction (used across test modules)
# ---------------------------------------------------------------------------


def _setup_fd(formula: str, data: pd.DataFrame, family):
    """Build FittingData from formula + data.

    Shared helper for tests that need a FittingData without going
    through the full GAM API.
    """
    from jaxgam.fitting.data import FittingData
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    spec = parse_formula(formula)
    setup = ModelSetup.build(spec, data)
    return FittingData.from_setup(setup, family)


# ---------------------------------------------------------------------------
# Private data generators (used by conftest and test-local fixtures)
# ---------------------------------------------------------------------------


def _make_nb_data(
    n: int = 200,
    seed: int = SEED,
    true_theta: float = 2.0,
) -> pd.DataFrame:
    """Generate single-predictor NB count data with known theta.

    Used across NB test modules (fitting, custom_jvp, results).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x) + 0.5
    mu = np.exp(eta)
    y = rng.negative_binomial(
        n=true_theta, p=true_theta / (mu + true_theta), size=n
    ).astype(float)
    return pd.DataFrame({"x": x, "y": y})


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
    elif family_name == "nb":
        eta = np.sin(2 * np.pi * x) + 0.5
        mu = np.exp(eta)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "y": y})
