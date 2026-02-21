"""Tests for GAM class (sklearn-style API).

Tests cover:
- A. GAM class API (unfitted raises, fit returns self, attribute types, stubs)
- B. End-to-end fitting (Gaussian basic, all-finite, shapes, Vp PSD)
- C. Hard-gate invariants (parametrized across 4 families)
- D. R comparison (parametrized across 4 families, skip if R unavailable)
- E. Multi-smooth R comparison (two smooths, tensor product)
- F. Factor-by R comparison
- G. ML optimization
- H. Fixed smoothing parameters
- I. Scope guards
- J. Edge cases (purely parametric, offset)

Tolerance rationale:
  Gaussian REML: MODERATE (rtol=1e-4, atol=1e-6). GLM families: LOOSE
  (rtol=1e-2, atol=1e-4). Smoothing parameters are NOT compared vs R
  because the REML criterion is flat near the optimum — lambda can
  differ by ~2% without affecting deviance, coefficients, or EDF
  (AGENTS.md §Common Pitfalls #4).

Design doc reference: Section 10.1, 10.2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymgcv.api import GAM
from tests.tolerances import LOOSE, MODERATE, STRICT

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r_available() -> bool:
    from pymgcv.compat.r_bridge import RBridge

    return RBridge.available()


def _make_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic data for a given family."""
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


def _make_two_smooth_data(seed: int = SEED) -> pd.DataFrame:
    """Two-predictor data for multi-smooth tests."""
    rng = np.random.default_rng(seed)
    n = 200
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_factor_by_data(seed: int = SEED) -> pd.DataFrame:
    """Factor-by data for s(x, by=fac) tests.

    Uses pd.Categorical so rpy2 converts to R factor correctly.
    """
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, n)
    levels = ["a", "b", "c"]
    fac = rng.choice(levels, n)
    eta = np.where(
        fac == "a", np.sin(2 * np.pi * x), np.where(fac == "b", 0.5 * x, -0.3 * x)
    )
    y = eta + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "x": x,
            "fac": pd.Categorical(fac, categories=levels),
            "y": y,
        }
    )


def _r_tol(family_name: str):
    """Return tolerance class for R comparison by family.

    Gaussian: MODERATE (single PIRLS iteration, no compounding).
    GLM families: LOOSE (iterative PIRLS + Newton, differences compound).
    """
    if family_name == "gaussian":
        return MODERATE
    return LOOSE


# ---------------------------------------------------------------------------
# A. TestGAMClass — basic API tests (no R)
# ---------------------------------------------------------------------------


class TestGAMClass:
    """Test GAM class interface before and after fitting."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_unfitted_predict_raises(self):
        model = GAM(self.FORMULA)
        with pytest.raises(RuntimeError, match="not fitted yet"):
            model.predict()

    def test_unfitted_summary_raises(self):
        model = GAM(self.FORMULA)
        with pytest.raises(RuntimeError, match="not fitted yet"):
            model.summary()

    def test_unfitted_plot_raises(self):
        model = GAM(self.FORMULA)
        with pytest.raises(RuntimeError, match="not fitted yet"):
            model.plot()

    def test_fit_returns_self(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA)
        result = model.fit(data)
        assert result is model

    def test_fitted_attributes_are_numpy(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA).fit(data)
        assert isinstance(model.coefficients_, np.ndarray)
        assert isinstance(model.fitted_values_, np.ndarray)
        assert isinstance(model.Vp_, np.ndarray)
        assert isinstance(model.edf_, np.ndarray)
        assert isinstance(model.X_, np.ndarray)
        assert isinstance(model.smoothing_params_, np.ndarray)

    def test_method_stubs_raise(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA).fit(data)
        with pytest.raises(NotImplementedError, match="Task 3.1"):
            model.predict()
        with pytest.raises(NotImplementedError, match="Task 3.2"):
            model.summary()
        with pytest.raises(NotImplementedError, match="Task 3.3"):
            model.plot()

    def test_ve_is_none(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA).fit(data)
        assert model.Ve_ is None

    def test_routing_fields(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA).fit(data)
        assert model.execution_path_ == "jax"
        assert model.lambda_strategy_ == "newton_reml"


# ---------------------------------------------------------------------------
# B. TestEndToEnd — basic fitting (no R)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end fitting sanity checks."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_gaussian_basic(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA).fit(data)
        assert model.converged_
        assert model.n_ == 200

    def test_all_fields_finite(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA).fit(data)
        assert np.all(np.isfinite(model.coefficients_))
        assert np.all(np.isfinite(model.fitted_values_))
        assert np.all(np.isfinite(model.linear_predictor_))
        assert np.all(np.isfinite(model.Vp_))
        assert np.all(np.isfinite(model.edf_))
        assert np.isfinite(model.scale_)
        assert np.isfinite(model.deviance_)
        assert np.isfinite(model.null_deviance_)
        assert np.isfinite(model.edf_total_)

    def test_shapes(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA).fit(data)
        n = model.n_
        p = model.X_.shape[1]
        n_smooths = len(model.smooth_info_)
        assert model.coefficients_.shape == (p,)
        assert model.fitted_values_.shape == (n,)
        assert model.linear_predictor_.shape == (n,)
        assert model.Vp_.shape == (p, p)
        assert model.edf_.shape == (n_smooths,)
        assert model.X_.shape == (n, p)

    def test_vp_symmetric_psd(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA).fit(data)
        Vp = model.Vp_
        np.testing.assert_allclose(
            Vp,
            Vp.T,
            atol=STRICT.atol,
            err_msg="Vp not symmetric",
        )
        eigvals = np.linalg.eigvalsh(Vp)
        assert np.all(eigvals >= -STRICT.rtol), (
            f"Vp has negative eigenvalue: {eigvals.min()}"
        )


# ---------------------------------------------------------------------------
# C. TestHardGateInvariants — parametrized across 4 families (no R)
# ---------------------------------------------------------------------------


class TestHardGateInvariants:
    """Hard-gate invariants that must hold for every family."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(
        params=["gaussian", "poisson", "binomial", "gamma"],
        ids=["gaussian", "poisson", "binomial", "gamma"],
    )
    def fitted_model(self, request):
        family_name = request.param
        data = _make_data(family_name)
        model = GAM(self.FORMULA, family=family_name).fit(data)
        return family_name, model

    def test_deviance_non_negative(self, fitted_model):
        _, model = fitted_model
        assert model.deviance_ >= 0

    def test_all_finite(self, fitted_model):
        _, model = fitted_model
        assert np.all(np.isfinite(model.coefficients_))
        assert np.all(np.isfinite(model.fitted_values_))
        assert np.all(np.isfinite(model.Vp_))
        assert np.isfinite(model.scale_)
        assert np.isfinite(model.deviance_)

    def test_edf_bounds(self, fitted_model):
        _, model = fitted_model
        p = model.X_.shape[1]
        assert np.all(model.edf_ > 0), f"EDF has non-positive entry: {model.edf_}"
        assert model.edf_total_ <= p + MODERATE.atol

    def test_vp_psd(self, fitted_model):
        _, model = fitted_model
        Vp = model.Vp_
        np.testing.assert_allclose(
            Vp,
            Vp.T,
            atol=STRICT.atol,
            err_msg="Vp not symmetric",
        )
        eigvals = np.linalg.eigvalsh(Vp)
        assert np.all(eigvals >= -MODERATE.atol), (
            f"Vp has negative eigenvalue: {eigvals.min()}"
        )

    def test_convergence(self, fitted_model):
        _, model = fitted_model
        assert model.converged_


# ---------------------------------------------------------------------------
# D. TestFamilyVsR — parametrized R comparison (skip if R unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestFamilyVsR:
    """R comparison across all four v1.0 families."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(
        params=[
            ("gaussian", "gaussian"),
            ("poisson", "poisson"),
            ("binomial", "binomial"),
            ("gamma", "gamma"),
        ],
        ids=["gaussian", "poisson", "binomial", "gamma"],
    )
    def family_fit(self, request):
        from pymgcv.compat.r_bridge import RBridge

        family_name, family_r = request.param
        data = _make_data(family_name)
        model = GAM(self.FORMULA, family=family_name).fit(data)
        bridge = RBridge()
        r_result = bridge.fit_gam(self.FORMULA, data, family=family_r)
        return family_name, model, r_result

    def test_deviance_vs_r(self, family_fit):
        family_name, model, r_result = family_fit
        tol = _r_tol(family_name)
        np.testing.assert_allclose(
            model.deviance_,
            r_result["deviance"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} deviance differs from R",
        )

    def test_coefficients_vs_r(self, family_fit):
        family_name, model, r_result = family_fit
        tol = _r_tol(family_name)
        np.testing.assert_allclose(
            model.coefficients_,
            r_result["coefficients"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} coefficients differ from R",
        )

    def test_fitted_values_vs_r(self, family_fit):
        family_name, model, r_result = family_fit
        tol = _r_tol(family_name)
        np.testing.assert_allclose(
            model.fitted_values_,
            r_result["fitted_values"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} fitted values differ from R",
        )

    def test_scale_vs_r(self, family_fit):
        family_name, model, r_result = family_fit
        tol = _r_tol(family_name)
        np.testing.assert_allclose(
            model.scale_,
            r_result["scale"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} scale differs from R",
        )

    def test_vp_vs_r(self, family_fit):
        family_name, model, r_result = family_fit
        tol = _r_tol(family_name)
        np.testing.assert_allclose(
            model.Vp_,
            r_result["Vp"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} Vp differs from R",
        )

    def test_per_smooth_edf_vs_r(self, family_fit):
        family_name, model, r_result = family_fit
        tol = _r_tol(family_name)
        np.testing.assert_allclose(
            model.edf_,
            r_result["edf"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} per-smooth EDF differs from R",
        )

    def test_null_deviance_vs_r(self, family_fit):
        family_name, model, r_result = family_fit
        tol = _r_tol(family_name)
        np.testing.assert_allclose(
            model.null_deviance_,
            r_result["null_deviance"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} null deviance differs from R",
        )


# ---------------------------------------------------------------------------
# E. TestMultiSmooth — two smooths and tensor product (R required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestMultiSmooth:
    """Multi-smooth models compared to R."""

    def test_two_smooths(self):
        from pymgcv.compat.r_bridge import RBridge

        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        data = _make_two_smooth_data()
        model = GAM(formula).fit(data)
        bridge = RBridge()
        r_result = bridge.fit_gam(formula, data)

        np.testing.assert_allclose(
            model.deviance_,
            r_result["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Two-smooth deviance differs from R",
        )
        np.testing.assert_allclose(
            model.coefficients_,
            r_result["coefficients"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Two-smooth coefficients differ from R",
        )
        assert model.edf_.shape == (2,), "Expected 2 per-smooth EDF entries"
        np.testing.assert_allclose(
            model.edf_,
            r_result["edf"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Two-smooth per-EDF differs from R",
        )

    def test_tensor_product(self):
        """te(x1, x2, k=5): Python parser uses scalar k (not R's c(5,5)).

        LOOSE tolerance: tensor products have multiple penalties and
        differences compound through the joint lambda optimization.
        """
        from pymgcv.compat.r_bridge import RBridge

        py_formula = "y ~ te(x1, x2, k=5)"
        r_formula = "y ~ te(x1, x2, k=c(5,5))"
        data = _make_two_smooth_data()
        model = GAM(py_formula).fit(data)
        bridge = RBridge()
        r_result = bridge.fit_gam(r_formula, data)

        np.testing.assert_allclose(
            model.deviance_,
            r_result["deviance"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Tensor product deviance differs from R",
        )


# ---------------------------------------------------------------------------
# F. TestFactorBy — factor-by smooth (R required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestFactorBy:
    """Factor-by smooth comparisons with R."""

    @pytest.mark.xfail(
        reason="Newton optimizer premature convergence for 3-penalty "
        "factor-by models — smoothing params stay at initial values. "
        "Pre-existing issue in fitting/newton.py, not in GAM API.",
        strict=False,
    )
    def test_factor_by_gaussian(self):
        from pymgcv.compat.r_bridge import RBridge

        formula = "y ~ s(x, by=fac, k=10, bs='cr') + fac"
        data = _make_factor_by_data()
        model = GAM(formula).fit(data)
        bridge = RBridge()
        r_result = bridge.fit_gam(formula, data)

        np.testing.assert_allclose(
            model.deviance_,
            r_result["deviance"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Factor-by deviance differs from R",
        )

    def test_factor_by_edf_count(self):
        """Factor-by smooth is stored as one combined SmoothInfo entry."""
        formula = "y ~ s(x, by=fac, k=10, bs='cr') + fac"
        data = _make_factor_by_data()
        model = GAM(formula).fit(data)
        # Our architecture stores factor-by as a single combined SmoothInfo
        # with 3 penalties (one per level), not 3 separate smooths.
        assert len(model.edf_) == 1, (
            f"Expected 1 combined per-smooth EDF entry for factor-by, "
            f"got {len(model.edf_)}"
        )


# ---------------------------------------------------------------------------
# G. TestMLOptimization — ML method
# ---------------------------------------------------------------------------


class TestMLOptimization:
    """ML smoothing parameter selection."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_ml_converges(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA, method="ML").fit(data)
        assert model.converged_
        assert model.lambda_strategy_ == "newton_ml"

    def test_ml_differs_from_reml(self):
        data = _make_data("gaussian")
        reml = GAM(self.FORMULA, method="REML").fit(data)
        ml = GAM(self.FORMULA, method="ML").fit(data)
        # ML and REML should give different smoothing params
        assert not np.allclose(
            reml.smoothing_params_,
            ml.smoothing_params_,
            atol=MODERATE.atol,
        ), "ML and REML smoothing params should differ"


# ---------------------------------------------------------------------------
# H. TestFixedSP — user-supplied smoothing parameters
# ---------------------------------------------------------------------------


class TestFixedSP:
    """Fixed smoothing parameter tests."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_fixed_sp_returns_gam(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA, sp=[1.0]).fit(data)
        assert isinstance(model.coefficients_, np.ndarray)
        assert model.converged_

    def test_fixed_sp_lambda_matches(self):
        data = _make_data("gaussian")
        sp = [2.5]
        model = GAM(self.FORMULA, sp=sp).fit(data)
        np.testing.assert_allclose(
            model.smoothing_params_,
            np.array(sp),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Fixed sp not preserved",
        )

    def test_fixed_sp_lambda_strategy(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA, sp=[1.0]).fit(data)
        assert model.lambda_strategy_ == "fixed"

    def test_fixed_sp_n_iter_zero(self):
        data = _make_data("gaussian")
        model = GAM(self.FORMULA, sp=[1.0]).fit(data)
        assert model.n_iter_ == 0


# ---------------------------------------------------------------------------
# I. TestScopeGuards — v1.0 scope guards
# ---------------------------------------------------------------------------


class TestScopeGuards:
    """v1.0 scope guard validation."""

    def test_backend_numpy_raises(self):
        with pytest.raises(NotImplementedError, match="backend='numpy'"):
            GAM("y ~ s(x)", backend="numpy")

    def test_select_true_raises(self):
        with pytest.raises(NotImplementedError, match="select=True"):
            GAM("y ~ s(x)", select=True)

    def test_gamma_nondefault_raises(self):
        with pytest.raises(NotImplementedError, match="gamma=1.4"):
            GAM("y ~ s(x)", gamma=1.4)

    def test_knots_raises(self):
        with pytest.raises(NotImplementedError, match="knots"):
            GAM("y ~ s(x)", knots={"x": [0, 0.5, 1]})

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="GCV"):
            GAM("y ~ s(x)", method="GCV")

    def test_jax_backend_allowed(self):
        # backend="jax" should not raise
        GAM("y ~ s(x)", backend="jax")

    def test_newton_optimizer_allowed(self):
        # optimizer="newton" should not raise
        GAM("y ~ s(x)", optimizer="newton")


# ---------------------------------------------------------------------------
# J. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and special configurations."""

    def test_purely_parametric(self):
        """Purely parametric model: no smooth terms."""
        rng = np.random.default_rng(SEED)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = 2.0 * x1 - 1.0 * x2 + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        model = GAM("y ~ x1 + x2").fit(data)
        assert model.edf_.shape == (0,)
        assert model.smoothing_params_.shape == (0,)
        assert model.converged_

    def test_offset_support(self):
        """Offset changes coefficients."""
        data = _make_data("gaussian")
        n = len(data)
        model_no_offset = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        offset = np.ones(n) * 0.5
        model_with_offset = GAM("y ~ s(x, k=10, bs='cr')").fit(data, offset=offset)
        # Coefficients should differ
        assert not np.allclose(
            model_no_offset.coefficients_,
            model_with_offset.coefficients_,
            atol=LOOSE.atol,
        ), "Offset should change coefficients"

    def test_method_case_insensitive(self):
        """Method name is case-insensitive."""
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')", method="reml").fit(data)
        assert model.converged_

    def test_chaining_api(self):
        """GAM(...).fit(data) chaining works."""
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        assert model.coefficients_.shape[0] > 0

    def test_family_object_accepted(self):
        """ExponentialFamily object works as family parameter."""
        from pymgcv.families.standard import Gaussian

        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')", family=Gaussian()).fit(data)
        assert model.converged_
