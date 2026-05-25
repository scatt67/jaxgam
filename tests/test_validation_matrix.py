"""75-Cell Validation Matrix: systematic R comparison for all smooth x family combos.

Tests cover every cell in the v1.0 surface (design.md §1.2):
- 15 smooth configs (tp, cr, te, ti, tp_by, cr_by, te_by, re, re_slope,
  re_mixed, gp, gp_2d, gp_mixed, gp_te, gp_ti) x 5 families = 75 cells
- Families: gaussian, binomial, poisson, gamma, nb

Plus hard-gate invariants (§18.1) that must hold for all cells without R.

Tolerance rationale (from AGENTS.md §Common Pitfalls, MEMORY.md):
  Gaussian REML: MODERATE (rtol=1e-4, atol=1e-6).
  GLM families: LOOSE (rtol=1e-2, atol=1e-4).
  Tensor products / factor-by: LOOSE for all (flat REML surfaces).
  TPRS: compare fitted values not raw coefficients (sign ambiguity).
  GP: compare fitted values not raw coefficients (eigenvector/SVD ambiguity).
  Factor-by EDF: our architecture stores 1 combined entry vs R's per-level;
    compare total EDF sum.
  RE (re, re_slope): deterministic basis, single sp — direct coef comparison.
  RE mixed (re_mixed): contains TPRS sign ambiguity — compare fitted values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from jax import clear_caches

from jaxgam.api import GAM
from tests.helpers import SEED, _AssertCollector, r_available
from tests.tolerances import LOOSE, MODERATE, STRICT

# ---------------------------------------------------------------------------
# JAX cache teardown
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    """Clear JAX compilation caches after each test to prevent OOM.

    Each GAM fit JIT-compiles functions with shapes specific to the model
    (smooth type, basis size, family). Without clearing, the accumulated LLVM
    artifacts exhaust memory on GH Actions runners (7 GB RAM).
    """
    yield
    clear_caches()  # teardown


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_single_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Single-predictor data for s(x) models."""
    rng = np.random.default_rng(seed)
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
        eta = np.sin(2 * np.pi * x) + 1.0
        mu = np.exp(eta)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "y": y})


def _make_two_smooth_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Two-predictor data for te/ti models, parametrized by family."""
    rng = np.random.default_rng(seed)
    n = 200 if family_name != "binomial" else 300
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)

    eta = np.sin(2 * np.pi * x1) + 0.5 * x2

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_gp_1d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """One-dimensional GP smooth data, parametrized by family."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, n)

    eta = np.sin(3 * np.pi * x) * 0.8 + np.cos(2 * np.pi * x) * 0.4

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "y": y})


def _make_gp_2d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Two-dimensional direct GP smooth data, parametrized by family."""
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)

    eta = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) + 0.3 * (x + z)

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "z": z, "y": y})


def _make_gp_1d_par_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """One-dimensional GP plus parametric linear term data."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, n)

    eta = 0.7 * x + np.sin(3 * np.pi * x) * 0.6

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "y": y})


def _make_gp_te_2d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Two-dimensional tensor GP smooth data, parametrized by family."""
    rng = np.random.default_rng(seed)
    n = 400
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)

    eta = (
        np.sin(2 * np.pi * x1)
        + np.cos(2 * np.pi * x2)
        + 0.5 * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
    )

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_factor_by_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Factor-by data for s(x, by=fac) models, parametrized by family."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, n)
    levels = ["a", "b", "c"]
    fac = rng.choice(levels, n)

    eta = np.where(
        fac == "a",
        np.sin(2 * np.pi * x),
        np.where(fac == "b", 0.5 * x, -0.3 * x),
    )

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-2 * eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame(
        {
            "x": x,
            "fac": pd.Categorical(fac, categories=levels),
            "y": y,
        }
    )


def _make_factor_by_2d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Factor-by data with 2D covariates for te(x1, x2, by=fac) models."""
    rng = np.random.default_rng(seed)
    n = 300
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    levels = ["a", "b", "c"]
    fac = rng.choice(levels, n)

    eta = np.where(
        fac == "a",
        np.sin(2 * np.pi * x1) + 0.5 * x2,
        np.where(fac == "b", 0.5 * x1 + x2, -0.3 * x1 - 0.2 * x2),
    )

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-2 * eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "fac": pd.Categorical(fac, categories=levels),
            "y": y,
        }
    )


def _make_re_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Random effects data: continuous x + factor g with group-level effects.

    Matches the conftest.py re_model_data fixture structure (n=300, 20 groups).
    """
    rng = np.random.default_rng(seed)
    n = 300
    n_groups = 20
    x = rng.uniform(0, 1, n)
    g = rng.choice([f"g{i}" for i in range(n_groups)], size=n)

    # True group effects
    b_intercept = rng.normal(0, 1.0, n_groups)
    group_idx = {f"g{i}": i for i in range(n_groups)}
    group_effect = np.array([b_intercept[group_idx[gi]] for gi in g])

    # Smooth + RE truth
    eta = np.sin(2 * np.pi * x) + group_effect

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.5, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame(
        {
            "x": x,
            "g": pd.Categorical(g),
            "y": y,
        }
    )


# ---------------------------------------------------------------------------
# Smooth configuration registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothConfig:
    """Configuration for one smooth type in the validation matrix."""

    py_formula: str
    r_formula: str
    data_type: str  # dispatched in _get_data


SMOOTH_CONFIGS: dict[str, SmoothConfig] = {
    "tp": SmoothConfig(
        py_formula="y ~ s(x, k=10, bs='tp')",
        r_formula="y ~ s(x, k=10, bs='tp')",
        data_type="single",
    ),
    "cr": SmoothConfig(
        py_formula="y ~ s(x, k=10, bs='cr')",
        r_formula="y ~ s(x, k=10, bs='cr')",
        data_type="single",
    ),
    "te": SmoothConfig(
        py_formula="y ~ te(x1, x2, k=5)",
        r_formula="y ~ te(x1, x2, k=c(5,5))",
        data_type="two_smooth",
    ),
    "ti": SmoothConfig(
        py_formula="y ~ ti(x1, x2, k=5)",
        r_formula="y ~ ti(x1, x2, k=c(5,5))",
        data_type="two_smooth",
    ),
    "tp_by": SmoothConfig(
        py_formula="y ~ s(x, by=fac, k=10, bs='tp') + fac",
        r_formula="y ~ s(x, by=fac, k=10, bs='tp') + fac",
        data_type="factor_by",
    ),
    "cr_by": SmoothConfig(
        py_formula="y ~ s(x, by=fac, k=10, bs='cr') + fac",
        r_formula="y ~ s(x, by=fac, k=10, bs='cr') + fac",
        data_type="factor_by",
    ),
    "te_by": SmoothConfig(
        py_formula="y ~ te(x1, x2, by=fac, k=5) + fac",
        r_formula="y ~ te(x1, x2, by=fac, k=c(5,5)) + fac",
        data_type="factor_by_2d",
    ),
    "re": SmoothConfig(
        py_formula="y ~ s(g, bs='re')",
        r_formula="y ~ s(g, bs='re')",
        data_type="re",
    ),
    "re_slope": SmoothConfig(
        py_formula="y ~ s(x, g, bs='re')",
        r_formula="y ~ s(x, g, bs='re')",
        data_type="re",
    ),
    "re_mixed": SmoothConfig(
        py_formula="y ~ s(x, k=10, bs='tp') + s(g, bs='re')",
        r_formula="y ~ s(x, k=10, bs='tp') + s(g, bs='re')",
        data_type="re",
    ),
    "gp": SmoothConfig(
        py_formula="y ~ s(x, bs='gp')",
        r_formula="y ~ s(x, bs='gp')",
        data_type="gp_1d",
    ),
    "gp_2d": SmoothConfig(
        py_formula="y ~ s(x, z, bs='gp', k=30)",
        r_formula="y ~ s(x, z, bs='gp', k=30)",
        data_type="gp_2d",
    ),
    "gp_mixed": SmoothConfig(
        py_formula="y ~ x + s(x, bs='gp')",
        r_formula="y ~ x + s(x, bs='gp')",
        data_type="gp_1d_par",
    ),
    "gp_te": SmoothConfig(
        py_formula="y ~ te(x1, x2, bs='gp', k=5)",
        r_formula="y ~ te(x1, x2, bs='gp', k=c(5, 5))",
        data_type="gp_te_2d",
    ),
    "gp_ti": SmoothConfig(
        py_formula=(
            "y ~ s(x1, bs='gp', k=5) + s(x2, bs='gp', k=5) + ti(x1, x2, bs='gp', k=5)"
        ),
        r_formula=(
            "y ~ s(x1, bs='gp', k=5) + s(x2, bs='gp', k=5) "
            "+ ti(x1, x2, bs='gp', k=c(5, 5))"
        ),
        data_type="gp_te_2d",
    ),
}

FAMILIES = ["gaussian", "binomial", "poisson", "gamma", "nb"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_data(config: SmoothConfig, family: str) -> pd.DataFrame:
    """Generate data for a given smooth config and family."""
    if config.data_type == "single":
        return _make_single_data(family)
    if config.data_type == "two_smooth":
        return _make_two_smooth_data(family)
    if config.data_type == "factor_by":
        return _make_factor_by_data(family)
    if config.data_type == "factor_by_2d":
        return _make_factor_by_2d_data(family)
    if config.data_type == "re":
        return _make_re_data(family)
    if config.data_type == "gp_1d":
        return _make_gp_1d_data(family)
    if config.data_type == "gp_2d":
        return _make_gp_2d_data(family)
    if config.data_type == "gp_1d_par":
        return _make_gp_1d_par_data(family)
    if config.data_type == "gp_te_2d":
        return _make_gp_te_2d_data(family)
    raise ValueError(f"Unknown data_type: {config.data_type}")


def _r_tol(smooth_key: str, family_name: str):
    """Tolerance for R comparison: MODERATE for Gaussian single-smooth, LOOSE otherwise.

    Tensor products and factor-by always use LOOSE (flat REML surfaces,
    multiple sp). GLM families also use LOOSE (iterative PIRLS compounding).
    """
    if family_name == "gaussian" and smooth_key in (
        "tp",
        "cr",
        "re",
        "re_slope",
        "re_mixed",
        "gp",
        "gp_2d",
    ):
        return MODERATE
    return LOOSE


def _fitted_tol(smooth_key: str, family_name: str):
    """Tolerance for fitted value comparison, wider for flat REML surfaces."""
    # Tensor factor-by with GLM/NB: 6+ sp, very flat REML surface
    if smooth_key in ("te_by",) and family_name in ("binomial", "poisson", "nb"):
        return LOOSE
    # Factor-by with binomial: multiple sp + binary response
    if smooth_key.endswith("_by") and family_name == "binomial":
        return LOOSE
    # Tensor interaction with GLM: flat surface
    if smooth_key in ("ti",) and family_name != "gaussian":
        return LOOSE
    return _r_tol(smooth_key, family_name)


def _compare_fitted_not_coefs(smooth_key: str) -> bool:
    """Whether to compare fitted values instead of raw coefficients.

    TPRS: eigenvector sign ambiguity makes coefficient comparison meaningless.
    Tensor products and factor-by: flat REML surfaces mean different sp can
    give different coefficients that produce equivalent fitted values.
    """
    return smooth_key in (
        "tp",
        "tp_by",
        "te",
        "ti",
        "te_by",
        "cr_by",
        "re_mixed",
        "gp",
        "gp_2d",
        "gp_mixed",
        "gp_te",
        "gp_ti",
    )


# ---------------------------------------------------------------------------
# Cell IDs for parametrization
# ---------------------------------------------------------------------------

CELL_IDS = [
    (smooth_key, family) for smooth_key in SMOOTH_CONFIGS for family in FAMILIES
]


def _cell_id(val):
    """Human-readable test ID: 'tp-gaussian', 'cr_by-binomial', etc."""
    return f"{val[0]}-{val[1]}"


# ---------------------------------------------------------------------------
# A. TestValidationMatrix — R comparison (75 cells)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestValidationMatrix:
    """Systematic R comparison across all smooth type x family cells."""

    @pytest.fixture(scope="class", params=CELL_IDS, ids=[_cell_id(c) for c in CELL_IDS])
    def cell(self, request):
        """Fit Python GAM and R GAM, return both for comparison."""
        from tests.r_bridge import RBridge

        smooth_key, family_name = request.param
        config = SMOOTH_CONFIGS[smooth_key]
        data = _get_data(config, family_name)

        model = GAM(config.py_formula, family=family_name).fit(data)
        bridge = RBridge()
        r_result = bridge.fit_gam(config.r_formula, data, family=family_name)

        return smooth_key, family_name, model, r_result

    def test_matches_r(self, cell):
        """Compare each validation-matrix cell against mgcv."""
        smooth_key, family_name, model, r_result = cell
        cell_id = f"{smooth_key}-{family_name}"
        collector = _AssertCollector()

        def assert_deviance_vs_r() -> None:
            tol = _r_tol(smooth_key, family_name)
            np.testing.assert_allclose(
                model.deviance,
                r_result["deviance"],
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{cell_id}] deviance",
            )

        def assert_fitted_values_vs_r() -> None:
            tol = _fitted_tol(smooth_key, family_name)
            np.testing.assert_allclose(
                model.fitted_values,
                r_result["fitted_values"],
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{cell_id}] fitted values",
            )

        def assert_edf_vs_r() -> None:
            tol = _fitted_tol(smooth_key, family_name)
            if smooth_key == "gp_mixed":
                # y ~ x + s(x, bs='gp') is intentionally rank-deficient:
                # parametric x duplicates the GP linear null-space column.
                # Smooth-block EDF allocation is pivot-dependent, so compare
                # the allocation-invariant total model EDF for this cell.
                py_edf_total = model.edf_total
                r_edf_total = float(r_result["edf_total"])
                edf_label = "total model EDF"
            else:
                py_edf_total = float(np.sum(model.edf))
                r_edf_total = float(np.sum(r_result["edf"]))
                edf_label = "total EDF"
            np.testing.assert_allclose(
                py_edf_total,
                r_edf_total,
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{cell_id}] {edf_label}",
            )

        def assert_scale_vs_r() -> None:
            tol = _r_tol(smooth_key, family_name)
            np.testing.assert_allclose(
                model.scale,
                r_result["scale"],
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{cell_id}] scale",
            )

        def assert_coefficients_vs_r() -> None:
            tol = _r_tol(smooth_key, family_name)
            if _compare_fitted_not_coefs(smooth_key):
                ftol = _fitted_tol(smooth_key, family_name)
                np.testing.assert_allclose(
                    model.fitted_values,
                    r_result["fitted_values"],
                    rtol=ftol.rtol,
                    atol=ftol.atol,
                    err_msg=f"[{cell_id}] fitted values (coef proxy)",
                )
            else:
                np.testing.assert_allclose(
                    model.coefficients,
                    r_result["coefficients"],
                    rtol=tol.rtol,
                    atol=tol.atol,
                    err_msg=f"[{cell_id}] coefficients",
                )

        def assert_self_prediction_roundtrip() -> None:
            pred = model.predict()
            np.testing.assert_allclose(
                pred,
                model.fitted_values,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"[{cell_id}] self-prediction roundtrip",
            )

        def assert_theta_vs_r() -> None:
            r_theta = r_result.get("theta")
            if r_theta is None:
                return
            py_theta = float(model.family.get_theta(transformed=True)[0])
            tol = _r_tol(smooth_key, family_name)
            np.testing.assert_allclose(
                py_theta,
                r_theta,
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{cell_id}] theta",
            )

        collector.check("deviance vs R", assert_deviance_vs_r)
        collector.check("fitted values vs R", assert_fitted_values_vs_r)
        collector.check("EDF vs R", assert_edf_vs_r)
        collector.check("scale vs R", assert_scale_vs_r)
        collector.check("coefficients vs R", assert_coefficients_vs_r)
        collector.check("self prediction roundtrip", assert_self_prediction_roundtrip)
        collector.check("theta vs R", assert_theta_vs_r)
        collector.raise_if_any(cell_id)


# ---------------------------------------------------------------------------
# B. TestHardGateInvariants — structural invariants (no R required)
# ---------------------------------------------------------------------------


class TestHardGateInvariants:
    """Hard-gate invariants (design.md §18.1) for all smooth x family cells.

    These must hold regardless of R comparison and never be waived.
    Eight invariants tested:
    1. Model convergence
    2. Deviance >= 0
    3. Converged beta produces finite eta, mu (no NaN/Inf)
    4. EDF in [0, p] per term, total in [0, n]
    5. Vp symmetric PSD
    6. Penalty S_j symmetric PSD
    7. Estimated theta > 0 for extended families
    8. Rank(X) >= p - null_space_dim
    """

    @pytest.fixture(scope="class", params=CELL_IDS, ids=[_cell_id(c) for c in CELL_IDS])
    def fitted_model(self, request):
        smooth_key, family_name = request.param
        config = SMOOTH_CONFIGS[smooth_key]
        data = _get_data(config, family_name)
        model = GAM(config.py_formula, family=family_name).fit(data)
        return smooth_key, family_name, model

    def test_all_invariants(self, fitted_model):
        """Check every hard-gate invariant for one matrix cell."""
        smooth_key, family_name, model = fitted_model
        cell_id = f"{smooth_key}-{family_name}"
        collector = _AssertCollector()

        def assert_convergence() -> None:
            assert model.converged, f"[{cell_id}] model did not converge"

        def assert_deviance_non_negative() -> None:
            assert model.deviance >= 0, (
                f"[{cell_id}] negative deviance: {model.deviance}"
            )

        def assert_no_nan_in_converged() -> None:
            assert np.all(np.isfinite(model.coefficients)), (
                f"[{cell_id}] NaN/Inf in coefficients"
            )
            assert np.all(np.isfinite(model.fitted_values)), (
                f"[{cell_id}] NaN/Inf in fitted values"
            )
            assert np.all(np.isfinite(model.linear_predictor)), (
                f"[{cell_id}] NaN/Inf in linear predictor"
            )
            assert np.isfinite(model.scale), f"[{cell_id}] non-finite scale"
            assert np.isfinite(model.deviance), f"[{cell_id}] non-finite deviance"

        def assert_edf_bounds() -> None:
            p = model.X.shape[1]
            n = model.n

            assert np.all(model.edf > 0), (
                f"[{cell_id}] non-positive per-smooth EDF: {model.edf}"
            )
            assert model.edf_total <= p, (
                f"[{cell_id}] total EDF {model.edf_total} > p={p}"
            )
            assert model.edf_total <= n + MODERATE.atol, (
                f"[{cell_id}] total EDF {model.edf_total} > n={n}"
            )

        def assert_vp_symmetric_psd() -> None:
            Vp = model.Vp

            np.testing.assert_allclose(
                Vp,
                Vp.T,
                atol=STRICT.atol,
                err_msg=f"[{cell_id}] Vp not symmetric",
            )
            eigvals = np.linalg.eigvalsh(Vp)
            assert np.all(eigvals >= 0.0), (
                f"[{cell_id}] Vp has negative eigenvalue: {eigvals.min()}"
            )

        def assert_penalty_psd() -> None:
            for j, si in enumerate(model.smooth_info):
                for term in model.coef_map.terms:
                    if term.label == si.label and term.term_type != "parametric":
                        smooth_obj = term.smooth
                        if hasattr(smooth_obj, "penalties") and smooth_obj.penalties:
                            for k, S_j in enumerate(smooth_obj.penalties):
                                np.testing.assert_allclose(
                                    S_j,
                                    S_j.T,
                                    atol=STRICT.atol,
                                    err_msg=f"[{cell_id}] S[{j}][{k}] not symmetric",
                                )
                                eigs = np.linalg.eigvalsh(S_j)
                                assert np.all(eigs >= -STRICT.atol), (
                                    f"[{cell_id}] S[{j}][{k}] has negative "
                                    f"eigenvalue: {eigs.min()}"
                                )

        def assert_theta_positive() -> None:
            if not hasattr(model.family, "n_theta") or model.family.n_theta == 0:
                return
            theta = float(model.family.get_theta(transformed=True)[0])
            assert theta > 0, f"[{cell_id}] non-positive theta: {theta}"

        def assert_model_matrix_rank() -> None:
            X = model.X
            total_null_dim = sum(si.n_penalties for si in model.smooth_info)
            rank = np.linalg.matrix_rank(X)
            assert rank >= min(X.shape) - total_null_dim, (
                f"[{cell_id}] rank(X)={rank}, "
                f"expected >= {min(X.shape) - total_null_dim}"
            )

        collector.check("convergence", assert_convergence)
        collector.check("deviance non-negative", assert_deviance_non_negative)
        collector.check("finite converged values", assert_no_nan_in_converged)
        collector.check("EDF bounds", assert_edf_bounds)
        collector.check("Vp symmetric PSD", assert_vp_symmetric_psd)
        collector.check("penalty PSD", assert_penalty_psd)
        collector.check("theta positive", assert_theta_positive)
        collector.check("model matrix rank", assert_model_matrix_rank)
        collector.raise_if_any(cell_id)
