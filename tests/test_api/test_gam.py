"""Tests for GAM class (sklearn-style API).

Tests cover:
- A. GAM class API (fit returns GAMResults, attributes)
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

from jaxgam.api import GAM
from jaxgam.results import GAMResults
from tests.helpers import (
    SEED,
    _generate_family_data,
    r_available,
    r_tolerance,
)
from tests.tolerances import LOOSE, MODERATE, STRICT

# ---------------------------------------------------------------------------
# A. TestGAMClass — basic API tests (no R)
# ---------------------------------------------------------------------------


class TestGAMClass:
    """Test GAM class interface."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_fit_returns_gam_results(self):
        data = _generate_family_data("gaussian")
        model = GAM(self.FORMULA)
        result = model.fit(data)
        assert isinstance(result, GAMResults)

    def test_fitted_attributes_are_numpy(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        assert isinstance(results.coefficients, np.ndarray)
        assert isinstance(results.fitted_values, np.ndarray)
        assert isinstance(results.Vp, np.ndarray)
        assert isinstance(results.edf, np.ndarray)
        assert isinstance(results.X, np.ndarray)
        assert isinstance(results.smoothing_params, np.ndarray)

    def test_summary_and_plot_work(self):
        import matplotlib

        matplotlib.use("Agg")
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        s = results.summary()
        assert s is not None
        fig, _axes = results.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_ve_omitted(self):
        """Ve is omitted entirely (design §9 #4)."""
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        assert not hasattr(results, "Ve")
        assert not hasattr(results, "Ve_")

    def test_routing_fields(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        assert results.execution_path == "jax"
        assert results.lambda_strategy == "newton_reml"


# ---------------------------------------------------------------------------
# B. TestEndToEnd — basic fitting (no R)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end fitting sanity checks."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_gaussian_basic(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        assert results.converged
        assert results.n == 200

    def test_all_fields_finite(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        assert np.all(np.isfinite(results.coefficients))
        assert np.all(np.isfinite(results.fitted_values))
        assert np.all(np.isfinite(results.linear_predictor))
        assert np.all(np.isfinite(results.Vp))
        assert np.all(np.isfinite(results.edf))
        assert np.isfinite(results.scale)
        assert np.isfinite(results.deviance)
        assert np.isfinite(results.null_deviance)
        assert np.isfinite(results.edf_total)

    def test_shapes(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        n = results.n
        p = results.X.shape[1]
        n_smooths = len(results.smooth_info)
        assert results.coefficients.shape == (p,)
        assert results.fitted_values.shape == (n,)
        assert results.linear_predictor.shape == (n,)
        assert results.Vp.shape == (p, p)
        assert results.edf.shape == (n_smooths,)
        assert results.X.shape == (n, p)

    def test_vp_symmetric_psd(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        Vp = results.Vp
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
    def fitted_results(self, request):
        family_name = request.param
        data = _generate_family_data(family_name)
        results = GAM(self.FORMULA, family=family_name).fit(data)
        return family_name, results

    def test_deviance_non_negative(self, fitted_results):
        _, results = fitted_results
        assert results.deviance >= 0

    def test_all_finite(self, fitted_results):
        _, results = fitted_results
        assert np.all(np.isfinite(results.coefficients))
        assert np.all(np.isfinite(results.fitted_values))
        assert np.all(np.isfinite(results.Vp))
        assert np.isfinite(results.scale)
        assert np.isfinite(results.deviance)

    def test_edf_bounds(self, fitted_results):
        _, results = fitted_results
        p = results.X.shape[1]
        assert np.all(results.edf > 0), f"EDF has non-positive entry: {results.edf}"
        assert results.edf_total <= p + MODERATE.atol

    def test_vp_psd(self, fitted_results):
        _, results = fitted_results
        Vp = results.Vp
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

    def test_convergence(self, fitted_results):
        _, results = fitted_results
        assert results.converged


# ---------------------------------------------------------------------------
# D. TestFamilyVsR — parametrized R comparison (skip if R unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
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
        from tests.r_bridge import RBridge

        family_name, family_r = request.param
        data = _generate_family_data(family_name)
        results = GAM(self.FORMULA, family=family_name).fit(data)
        bridge = RBridge()
        r_result = bridge.fit_gam(self.FORMULA, data, family=family_r)
        return family_name, results, r_result

    def test_deviance_vs_r(self, family_fit):
        family_name, results, r_result = family_fit
        tol = r_tolerance(family_name)
        np.testing.assert_allclose(
            results.deviance,
            r_result["deviance"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} deviance differs from R",
        )

    def test_coefficients_vs_r(self, family_fit):
        family_name, results, r_result = family_fit
        tol = r_tolerance(family_name)
        np.testing.assert_allclose(
            results.coefficients,
            r_result["coefficients"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} coefficients differ from R",
        )

    def test_fitted_values_vs_r(self, family_fit):
        family_name, results, r_result = family_fit
        tol = r_tolerance(family_name)
        np.testing.assert_allclose(
            results.fitted_values,
            r_result["fitted_values"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} fitted values differ from R",
        )

    def test_scale_vs_r(self, family_fit):
        family_name, results, r_result = family_fit
        tol = r_tolerance(family_name)
        np.testing.assert_allclose(
            results.scale,
            r_result["scale"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} scale differs from R",
        )

    def test_vp_vs_r(self, family_fit):
        family_name, results, r_result = family_fit
        tol = r_tolerance(family_name)
        np.testing.assert_allclose(
            results.Vp,
            r_result["Vp"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} Vp differs from R",
        )

    def test_per_smooth_edf_vs_r(self, family_fit):
        family_name, results, r_result = family_fit
        tol = r_tolerance(family_name)
        np.testing.assert_allclose(
            results.edf,
            r_result["edf"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} per-smooth EDF differs from R",
        )

    def test_null_deviance_vs_r(self, family_fit):
        family_name, results, r_result = family_fit
        tol = r_tolerance(family_name)
        np.testing.assert_allclose(
            results.null_deviance,
            r_result["null_deviance"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} null deviance differs from R",
        )


# ---------------------------------------------------------------------------
# E. TestMultiSmooth — two smooths and tensor product (R required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestMultiSmooth:
    """Multi-smooth models compared to R."""

    def test_two_smooths(self, two_smooth_data):
        from tests.r_bridge import RBridge

        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        results = GAM(formula).fit(two_smooth_data)
        bridge = RBridge()
        r_result = bridge.fit_gam(formula, two_smooth_data)

        np.testing.assert_allclose(
            results.deviance,
            r_result["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Two-smooth deviance differs from R",
        )
        np.testing.assert_allclose(
            results.coefficients,
            r_result["coefficients"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Two-smooth coefficients differ from R",
        )
        assert results.edf.shape == (2,), "Expected 2 per-smooth EDF entries"
        np.testing.assert_allclose(
            results.edf,
            r_result["edf"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Two-smooth per-EDF differs from R",
        )

    def test_tensor_product(self, two_smooth_data):
        """te(x1, x2, k=5): Python parser uses scalar k (not R's c(5,5)).

        With the default basis fix (te defaults to cr, matching R),
        we achieve MODERATE agreement on deviance, coefficients, and
        fitted values.
        """
        from tests.r_bridge import RBridge

        py_formula = "y ~ te(x1, x2, k=5)"
        r_formula = "y ~ te(x1, x2, k=c(5,5))"
        results = GAM(py_formula).fit(two_smooth_data)
        bridge = RBridge()
        r_result = bridge.fit_gam(r_formula, two_smooth_data)

        np.testing.assert_allclose(
            results.deviance,
            r_result["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Tensor product deviance differs from R",
        )
        # LOOSE: one penalty's sp converges on a flat REML landscape,
        # so small sp differences cascade to coefficients (~1e-3 rel).
        np.testing.assert_allclose(
            results.coefficients,
            r_result["coefficients"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Tensor product coefficients differ from R",
        )
        np.testing.assert_allclose(
            results.fitted_values,
            r_result["fitted_values"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Tensor product fitted values differ from R",
        )


# ---------------------------------------------------------------------------
# F. TestFactorBy — factor-by smooth (R required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestFactorBy:
    """Factor-by smooth comparisons with R."""

    def test_factor_by_gaussian(self, factor_by_data):
        from tests.r_bridge import RBridge

        formula = "y ~ s(x, by=fac, k=10, bs='cr') + fac"
        results = GAM(formula).fit(factor_by_data)
        bridge = RBridge()
        r_result = bridge.fit_gam(formula, factor_by_data)

        np.testing.assert_allclose(
            results.deviance,
            r_result["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Factor-by deviance differs from R",
        )
        # LOOSE: one sp is at the lsp_max cap (exp(15) vs R's exp(18)),
        # causing minor coefficient/fitted value differences (~1e-3 rel).
        np.testing.assert_allclose(
            results.coefficients,
            r_result["coefficients"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Factor-by coefficients differ from R",
        )
        np.testing.assert_allclose(
            results.fitted_values,
            r_result["fitted_values"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Factor-by fitted values differ from R",
        )

    def test_factor_by_edf_count(self, factor_by_data):
        """Factor-by smooth is stored as one combined SmoothInfo entry."""
        formula = "y ~ s(x, by=fac, k=10, bs='cr') + fac"
        results = GAM(formula).fit(factor_by_data)
        # Our architecture stores factor-by as a single combined SmoothInfo
        # with 3 penalties (one per level), not 3 separate smooths.
        assert len(results.edf) == 1, (
            f"Expected 1 combined per-smooth EDF entry for factor-by, "
            f"got {len(results.edf)}"
        )


# ---------------------------------------------------------------------------
# G. TestMLOptimization — ML method
# ---------------------------------------------------------------------------


class TestMLOptimization:
    """ML smoothing parameter selection."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_ml_converges(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA, method="ML").fit(data)
        assert results.converged
        assert results.lambda_strategy == "newton_ml"

    def test_ml_differs_from_reml(self):
        data = _generate_family_data("gaussian")
        reml = GAM(self.FORMULA, method="REML").fit(data)
        ml = GAM(self.FORMULA, method="ML").fit(data)
        # ML and REML should give different smoothing params
        assert not np.allclose(
            reml.smoothing_params,
            ml.smoothing_params,
            atol=MODERATE.atol,
        ), "ML and REML smoothing params should differ"


# ---------------------------------------------------------------------------
# H. TestFixedSP — user-supplied smoothing parameters
# ---------------------------------------------------------------------------


class TestFixedSP:
    """Fixed smoothing parameter tests."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_fixed_sp_attributes(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA, sp=[1.0]).fit(data)
        assert isinstance(results.coefficients, np.ndarray)
        assert results.converged

    def test_fixed_sp_lambda_matches(self):
        data = _generate_family_data("gaussian")
        sp = [2.5]
        results = GAM(self.FORMULA, sp=sp).fit(data)
        np.testing.assert_allclose(
            results.smoothing_params,
            np.array(sp),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Fixed sp not preserved",
        )

    def test_fixed_sp_lambda_strategy(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA, sp=[1.0]).fit(data)
        assert results.lambda_strategy == "fixed"

    def test_fixed_sp_n_iter_zero(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA, sp=[1.0]).fit(data)
        assert results.n_iter == 0


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
        with pytest.raises(NotImplementedError, match=r"gamma=1\.4"):
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

        results = GAM("y ~ x1 + x2").fit(data)
        assert results.edf.shape == (0,)
        assert results.smoothing_params.shape == (0,)
        assert results.converged

    def test_offset_support(self):
        """Offset changes coefficients."""
        data = _generate_family_data("gaussian")
        n = len(data)
        results_no_offset = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        offset = np.ones(n) * 0.5
        results_with_offset = GAM("y ~ s(x, k=10, bs='cr')").fit(data, offset=offset)
        # Coefficients should differ
        assert not np.allclose(
            results_no_offset.coefficients,
            results_with_offset.coefficients,
            atol=LOOSE.atol,
        ), "Offset should change coefficients"

    def test_method_case_insensitive(self):
        """Method name is case-insensitive."""
        data = _generate_family_data("gaussian")
        results = GAM("y ~ s(x, k=10, bs='cr')", method="reml").fit(data)
        assert results.converged

    def test_chaining_api(self):
        """GAM(...).fit(data) chaining returns GAMResults."""
        data = _generate_family_data("gaussian")
        results = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        assert isinstance(results, GAMResults)
        assert results.coefficients.shape[0] > 0

    def test_family_object_accepted(self):
        """ExponentialFamily object works as family parameter."""
        from jaxgam.families.standard import Gaussian

        data = _generate_family_data("gaussian")
        results = GAM("y ~ s(x, k=10, bs='cr')", family=Gaussian()).fit(data)
        assert results.converged
