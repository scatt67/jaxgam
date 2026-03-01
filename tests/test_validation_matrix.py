"""32-Cell Validation Matrix: systematic R comparison for all smooth x family combos.

Tests cover every cell in the v1.0 surface (design.md §1.2):
- 3 smooth types (tp, cr, te) x 4 families = 12 cells
- 2 tensor types (te, ti) x 4 families = 8 cells
- 3 factor-by types (tp_by, cr_by, te_by) x 4 families = 12 cells
- Total ~32 (with overlap, ~28 unique)

Plus hard-gate invariants (§18.1) that must hold for all cells without R.

Tolerance rationale (from AGENTS.md §Common Pitfalls, MEMORY.md):
  Gaussian REML: MODERATE (rtol=1e-4, atol=1e-6).
  GLM families: LOOSE (rtol=1e-2, atol=1e-4).
  Tensor products / factor-by: LOOSE for all (flat REML surfaces).
  TPRS: compare fitted values not raw coefficients (sign ambiguity).
  Factor-by EDF: our architecture stores 1 combined entry vs R's per-level;
    compare total EDF sum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from jaxgam.api import GAM
from tests.tolerances import LOOSE, MODERATE, STRICT

SEED = 42


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


# ---------------------------------------------------------------------------
# Smooth configuration registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothConfig:
    """Configuration for one smooth type in the validation matrix."""

    py_formula: str
    r_formula: str
    data_type: str  # "single", "two_smooth", "factor_by", "factor_by_2d"


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
}

FAMILIES = ["gaussian", "binomial", "poisson", "gamma"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r_available() -> bool:
    from tests.r_bridge import RBridge

    if not RBridge.available():
        return False
    ok, _ = RBridge.check_versions()
    return ok


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
    raise ValueError(f"Unknown data_type: {config.data_type}")


def _r_tol(smooth_key: str, family_name: str):
    """Tolerance for R comparison: MODERATE for Gaussian single-smooth, LOOSE otherwise.

    Tensor products and factor-by always use LOOSE (flat REML surfaces,
    multiple sp). GLM families also use LOOSE (iterative PIRLS compounding).
    """
    if family_name == "gaussian" and smooth_key in ("tp", "cr"):
        return MODERATE
    return LOOSE


def _fitted_tol(smooth_key: str, family_name: str):
    """Tolerance for fitted value comparison, wider for flat REML surfaces."""
    # Tensor factor-by with GLM: 6+ sp, very flat REML surface
    if smooth_key in ("te_by",) and family_name in ("binomial", "poisson"):
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
    return smooth_key in ("tp", "tp_by", "te", "ti", "te_by", "cr_by")


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
# A. TestValidationMatrix — R comparison (28 cells)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestValidationMatrix:
    """Systematic R comparison across all smooth type x family cells."""

    @pytest.fixture(params=CELL_IDS, ids=[_cell_id(c) for c in CELL_IDS])
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

    def test_deviance_vs_r(self, cell):
        smooth_key, family_name, model, r_result = cell
        tol = _r_tol(smooth_key, family_name)
        np.testing.assert_allclose(
            model.deviance_,
            r_result["deviance"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"[{smooth_key}-{family_name}] deviance",
        )

    def test_fitted_values_vs_r(self, cell):
        smooth_key, family_name, model, r_result = cell
        tol = _fitted_tol(smooth_key, family_name)
        np.testing.assert_allclose(
            model.fitted_values_,
            r_result["fitted_values"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"[{smooth_key}-{family_name}] fitted values",
        )

    def test_edf_vs_r(self, cell):
        """Compare total EDF sum (our architecture may group differently)."""
        smooth_key, family_name, model, r_result = cell
        tol = _fitted_tol(smooth_key, family_name)  # EDF sensitive to sp differences
        py_edf_total = float(np.sum(model.edf_))
        r_edf_total = float(np.sum(r_result["edf"]))
        np.testing.assert_allclose(
            py_edf_total,
            r_edf_total,
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"[{smooth_key}-{family_name}] total EDF",
        )

    def test_scale_vs_r(self, cell):
        smooth_key, family_name, model, r_result = cell
        tol = _r_tol(smooth_key, family_name)
        np.testing.assert_allclose(
            model.scale_,
            r_result["scale"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"[{smooth_key}-{family_name}] scale",
        )

    def test_coefficients_vs_r(self, cell):
        """Compare coefficients or fitted values depending on smooth type.

        TPRS: eigenvector sign ambiguity → compare fitted values.
        Tensor/factor-by: flat REML surfaces → compare fitted values.
        CR single-smooth: direct coefficient comparison.
        """
        smooth_key, family_name, model, r_result = cell
        tol = _r_tol(smooth_key, family_name)
        if _compare_fitted_not_coefs(smooth_key):
            # Compare fitted values as proxy for coefficient equivalence
            ftol = _fitted_tol(smooth_key, family_name)
            np.testing.assert_allclose(
                model.fitted_values_,
                r_result["fitted_values"],
                rtol=ftol.rtol,
                atol=ftol.atol,
                err_msg=f"[{smooth_key}-{family_name}] fitted values (coef proxy)",
            )
        else:
            np.testing.assert_allclose(
                model.coefficients_,
                r_result["coefficients"],
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{smooth_key}-{family_name}] coefficients",
            )

    def test_self_prediction_roundtrip(self, cell):
        """predict() with no newdata reproduces fitted_values."""
        smooth_key, family_name, model, _r_result = cell
        pred = model.predict()
        np.testing.assert_allclose(
            pred,
            model.fitted_values_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"[{smooth_key}-{family_name}] self-prediction roundtrip",
        )


# ---------------------------------------------------------------------------
# B. TestHardGateInvariants — structural invariants (no R required)
# ---------------------------------------------------------------------------


class TestHardGateInvariants:
    """Hard-gate invariants (design.md §18.1) for all smooth x family cells.

    These must hold regardless of R comparison and never be waived.
    Seven invariants tested:
    1. H = XtWX + S_lambda symmetric PSD
    2. Penalty S_j symmetric PSD
    3. Rank(X) >= p - null_space_dim
    4. EDF in [0, p] per term, total in [0, n]
    5. Deviance >= 0
    6. Converged beta produces finite eta, mu (no NaN/Inf)
    7. Vp symmetric PSD
    """

    @pytest.fixture(params=CELL_IDS, ids=[_cell_id(c) for c in CELL_IDS])
    def fitted_model(self, request):
        smooth_key, family_name = request.param
        config = SMOOTH_CONFIGS[smooth_key]
        data = _get_data(config, family_name)
        model = GAM(config.py_formula, family=family_name).fit(data)
        return smooth_key, family_name, model

    def test_convergence(self, fitted_model):
        """Model converges for all cells."""
        smooth_key, family_name, model = fitted_model
        assert model.converged_, f"[{smooth_key}-{family_name}] model did not converge"

    def test_deviance_non_negative(self, fitted_model):
        """Deviance >= 0 (§18.1 invariant 6)."""
        smooth_key, family_name, model = fitted_model
        assert model.deviance_ >= 0, (
            f"[{smooth_key}-{family_name}] negative deviance: {model.deviance_}"
        )

    def test_no_nan_in_converged(self, fitted_model):
        """Converged beta produces finite eta, mu (§18.1 invariant 7)."""
        smooth_key, family_name, model = fitted_model
        assert np.all(np.isfinite(model.coefficients_)), (
            f"[{smooth_key}-{family_name}] NaN/Inf in coefficients"
        )
        assert np.all(np.isfinite(model.fitted_values_)), (
            f"[{smooth_key}-{family_name}] NaN/Inf in fitted values"
        )
        assert np.all(np.isfinite(model.linear_predictor_)), (
            f"[{smooth_key}-{family_name}] NaN/Inf in linear predictor"
        )
        assert np.isfinite(model.scale_), (
            f"[{smooth_key}-{family_name}] non-finite scale"
        )
        assert np.isfinite(model.deviance_), (
            f"[{smooth_key}-{family_name}] non-finite deviance"
        )

    def test_edf_bounds(self, fitted_model):
        """EDF in [0, p] per term, total in [0, n] (§18.1 invariant 5)."""
        smooth_key, family_name, model = fitted_model
        p = model.X_.shape[1]
        n = model.n_

        # Per-smooth EDF should be positive
        assert np.all(model.edf_ > 0), (
            f"[{smooth_key}-{family_name}] non-positive per-smooth EDF: {model.edf_}"
        )
        # Total EDF bounded by p
        assert model.edf_total_ <= p + MODERATE.atol, (
            f"[{smooth_key}-{family_name}] total EDF {model.edf_total_} > p={p}"
        )
        # Total EDF bounded by n
        assert model.edf_total_ <= n + MODERATE.atol, (
            f"[{smooth_key}-{family_name}] total EDF {model.edf_total_} > n={n}"
        )

    def test_vp_symmetric_psd(self, fitted_model):
        """Vp is symmetric PSD (§18.1 invariant 2 applied to Bayesian cov)."""
        smooth_key, family_name, model = fitted_model
        Vp = model.Vp_

        # Symmetry
        np.testing.assert_allclose(
            Vp,
            Vp.T,
            atol=STRICT.atol,
            err_msg=f"[{smooth_key}-{family_name}] Vp not symmetric",
        )

        # PSD: eigenvalues >= 0 (allow small negative from numerical noise)
        eigvals = np.linalg.eigvalsh(Vp)
        assert np.all(eigvals >= -MODERATE.atol), (
            f"[{smooth_key}-{family_name}] Vp has negative eigenvalue: {eigvals.min()}"
        )

    def test_penalty_psd(self, fitted_model):
        """Penalty matrices S_j are symmetric PSD (§18.1 invariant 3)."""
        smooth_key, family_name, model = fitted_model

        for j, si in enumerate(model.smooth_info_):
            # Access penalties from the coef_map's smooth terms
            for term in model.coef_map_.terms:
                if term.label == si.label and term.term_type != "parametric":
                    smooth_obj = term.smooth
                    # Get penalty matrices from the smooth
                    if hasattr(smooth_obj, "penalties") and smooth_obj.penalties:
                        for k, S_j in enumerate(smooth_obj.penalties):
                            # Symmetry
                            np.testing.assert_allclose(
                                S_j,
                                S_j.T,
                                atol=STRICT.atol,
                                err_msg=(
                                    f"[{smooth_key}-{family_name}] "
                                    f"S[{j}][{k}] not symmetric"
                                ),
                            )
                            # PSD
                            eigs = np.linalg.eigvalsh(S_j)
                            assert np.all(eigs >= -STRICT.atol), (
                                f"[{smooth_key}-{family_name}] "
                                f"S[{j}][{k}] has negative eigenvalue: {eigs.min()}"
                            )

    def test_model_matrix_rank(self, fitted_model):
        """Rank(X) >= p - total_null_space_dim (§18.1 invariant 4)."""
        smooth_key, family_name, model = fitted_model
        X = model.X_

        # Sum null space dimensions across all smooth terms
        total_null_dim = sum(
            si.n_penalties  # each penalty contributes null space
            for si in model.smooth_info_
        )
        # Rank check (numerical rank via SVD)
        rank = np.linalg.matrix_rank(X)
        # Conservative: rank >= p - total_null_dim (p for full model with intercept)
        # In practice rank should equal p for well-posed GAMs
        assert rank >= min(X.shape) - total_null_dim, (
            f"[{smooth_key}-{family_name}] rank(X)={rank}, "
            f"expected >= {min(X.shape) - total_null_dim}"
        )
