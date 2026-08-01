"""Shared test helpers.

Public API (used directly by tests):
- SEED, N — constants
- r_available() — import-time R check for @pytest.mark.skipif
- r_tolerance() — tolerance tier by family
- make_smooth_spec() — SmoothSpec factory with many call-site variants

Private API (used by conftest fixtures and complex test-local fixtures):
- _generate_family_data() — single-predictor family data
- _inference_equivalence_cases(), _inference_r_cases() — shared prediction zoos
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


# ---------------------------------------------------------------------------
# Shared inference prediction cases
# ---------------------------------------------------------------------------


def _inference_factor_by_case(n: int = 120) -> tuple[Any, ...]:
    """Binomial factor-by case shared by predictor and result-mode gates."""
    rng = np.random.default_rng(SEED)
    levels = ["a", "b"]
    x = rng.uniform(size=n)
    fac_values = rng.choice(levels, size=n)
    eta = np.where(fac_values == "a", np.sin(2 * np.pi * x), 0.5 * x)
    data = pd.DataFrame(
        {
            "x": x,
            "fac": pd.Categorical(fac_values, categories=levels),
            "y": rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float),
        }
    )
    newdata = pd.DataFrame(
        {
            "x": rng.uniform(size=30),
            "fac": pd.Categorical(rng.choice(levels, size=30), categories=levels),
        }
    )
    return (
        "y ~ s(x, by=fac, k=5, bs='cr') + fac",
        data,
        newdata,
        "binomial",
        [1.0, 1.0],
        None,
        None,
    )


def _inference_tensor_case(kind: str, n: int = 120) -> tuple[Any, ...]:
    """Gaussian tensor case shared by predictor and result-mode gates."""
    rng = np.random.default_rng(SEED + (1 if kind == "te" else 2))
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(scale=0.2, size=n)
    data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    newdata = pd.DataFrame({"x1": rng.uniform(size=30), "x2": rng.uniform(size=30)})
    return (
        f"y ~ {kind}(x1, x2, k=4)",
        data,
        newdata,
        "gaussian",
        [1.0, 1.0],
        None,
        None,
    )


def _inference_one_dimensional_case(
    family: str | Any,
    *,
    offset: bool = False,
) -> tuple[Any, ...]:
    """One-dimensional family case shared by inference behavior gates."""
    family_name = family if isinstance(family, str) else family.family_name.lower()
    data = _generate_family_data(family_name, n=120)
    rng = np.random.default_rng(SEED + 100)
    newdata = pd.DataFrame({"x": rng.uniform(size=30)})
    train_offset = np.linspace(0.1, 0.3, len(data)) if offset else None
    predict_offset = np.linspace(0.2, 0.4, len(newdata)) if offset else None
    return (
        "y ~ s(x, k=6, bs='cr')",
        data,
        newdata,
        family,
        [1.0],
        train_offset,
        predict_offset,
    )


def _inference_equivalence_cases() -> dict[str, tuple[Any, ...]]:
    """Internal full/predictor/inference equivalence zoo."""
    from jaxgam.families.standard import Gamma

    return {
        "gaussian-s": _inference_one_dimensional_case("gaussian"),
        "binomial-factor-by": _inference_factor_by_case(),
        "poisson-offset": _inference_one_dimensional_case("poisson", offset=True),
        "negative-binomial": _inference_one_dimensional_case("nb"),
        "gamma-log": _inference_one_dimensional_case(Gamma(link="log")),
        "tensor-te": _inference_tensor_case("te"),
        "tensor-ti": _inference_tensor_case("ti"),
    }


def _inference_r_cases() -> dict[
    str, tuple[str, str, pd.DataFrame, pd.DataFrame, Any, str]
]:
    """Direct-mgcv inference cases supported by the rpy2 bridge."""
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.families.standard import Gamma

    rng = np.random.default_rng(SEED)
    n = 180
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    tensor_data = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "y": np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.25, n),
        }
    )
    tensor_new = pd.DataFrame({"x1": rng.uniform(size=40), "x2": rng.uniform(size=40)})
    factor_formula, factor_data, factor_new, *_ = _inference_factor_by_case(n=180)
    one_d_new = pd.DataFrame({"x": rng.uniform(size=40)})

    return {
        "tensor": (
            "y ~ te(x1, x2, k=5)",
            "y ~ te(x1, x2, k=c(5,5))",
            tensor_data,
            tensor_new,
            "gaussian",
            "gaussian",
        ),
        "factor-by": (
            factor_formula,
            factor_formula,
            factor_data,
            factor_new,
            "binomial",
            "binomial",
        ),
        "negative-binomial": (
            "y ~ s(x, k=8, bs='cr')",
            "y ~ s(x, k=8, bs='cr')",
            _generate_family_data("nb", n=180),
            one_d_new,
            NegativeBinomial(),
            "nb",
        ),
        "gamma-log": (
            "y ~ s(x, k=8, bs='cr')",
            "y ~ s(x, k=8, bs='cr')",
            _generate_family_data("gamma", n=180),
            one_d_new,
            Gamma(link="log"),
            "gamma_log",
        ),
    }
