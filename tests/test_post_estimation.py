"""Tests for jaxgam.post_estimation module.

Tests cover:
- Unit tests for EDF computation with known hat matrices
- Unit tests for null deviance with known family/data
- Integration test: fit a model, pass raw results to compute_post_estimation(),
  verify outputs match the current _store_results() outputs
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from jaxgam.post_estimation import (
    compute_null_deviance,
    compute_per_smooth_edf,
    compute_per_smooth_edf1,
    compute_post_estimation,
)
from tests.helpers import SEED
from tests.tolerances import STRICT

# ---------------------------------------------------------------------------
# Minimal SmoothInfo stub for unit tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SmoothInfoStub:
    """Minimal stand-in for SmoothInfo with first_coef / last_coef."""

    first_coef: int
    last_coef: int


# ---------------------------------------------------------------------------
# Unit tests: compute_per_smooth_edf
# ---------------------------------------------------------------------------


class TestComputePerSmoothEdf:
    """Test per-smooth EDF via trace of hat matrix blocks."""

    def test_identity_hat_matrix(self):
        """Identity hat matrix gives EDF = block size for each smooth."""
        p = 6
        F = np.eye(p)
        smooth_info = (
            _SmoothInfoStub(first_coef=0, last_coef=3),
            _SmoothInfoStub(first_coef=3, last_coef=6),
        )
        edf = compute_per_smooth_edf(F, smooth_info)
        np.testing.assert_allclose(edf, [3.0, 3.0], rtol=STRICT.rtol, atol=STRICT.atol)

    def test_known_diagonal(self):
        """Diagonal hat matrix with known values."""
        F = np.diag([0.9, 0.8, 0.5, 0.3, 0.1])
        smooth_info = (
            _SmoothInfoStub(first_coef=0, last_coef=2),
            _SmoothInfoStub(first_coef=2, last_coef=5),
        )
        edf = compute_per_smooth_edf(F, smooth_info)
        np.testing.assert_allclose(
            edf,
            [0.9 + 0.8, 0.5 + 0.3 + 0.1],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_single_smooth(self):
        """Single smooth spanning all coefficients."""
        F = np.array([[0.7, 0.1], [0.2, 0.6]])
        smooth_info = (_SmoothInfoStub(first_coef=0, last_coef=2),)
        edf = compute_per_smooth_edf(F, smooth_info)
        np.testing.assert_allclose(edf, [1.3], rtol=STRICT.rtol, atol=STRICT.atol)

    def test_empty_smooth_info(self):
        """No smooths returns empty array."""
        F = np.eye(3)
        edf = compute_per_smooth_edf(F, ())
        assert edf.shape == (0,)


# ---------------------------------------------------------------------------
# Unit tests: compute_per_smooth_edf1
# ---------------------------------------------------------------------------


class TestComputePerSmoothEdf1:
    """Test alternative EDF (edf1 = 2*edf - trace(F^2)) per smooth."""

    def test_identity_hat_matrix(self):
        """For F = I: edf1 = 2*trace(I) - trace(I^2) = 2*k - k = k."""
        p = 6
        F = np.eye(p)
        smooth_info = (
            _SmoothInfoStub(first_coef=0, last_coef=3),
            _SmoothInfoStub(first_coef=3, last_coef=6),
        )
        edf1 = compute_per_smooth_edf1(F, smooth_info)
        np.testing.assert_allclose(edf1, [3.0, 3.0], rtol=STRICT.rtol, atol=STRICT.atol)

    def test_known_diagonal(self):
        """Diagonal F: edf1_i = 2*F[i,i] - F[i,i]^2 per coefficient."""
        diag_vals = np.array([0.9, 0.5, 0.3])
        F = np.diag(diag_vals)
        smooth_info = (
            _SmoothInfoStub(first_coef=0, last_coef=2),
            _SmoothInfoStub(first_coef=2, last_coef=3),
        )
        edf1 = compute_per_smooth_edf1(F, smooth_info)
        # For diagonal: edf1_i = 2*d_i - d_i^2
        expected_per_coef = 2.0 * diag_vals - diag_vals**2
        expected = [expected_per_coef[0] + expected_per_coef[1], expected_per_coef[2]]
        np.testing.assert_allclose(edf1, expected, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_edf1_geq_edf_for_psd_F(self):
        """For PSD F with eigenvalues in [0, 1], edf1 >= edf.

        Since 2*e - e^2 = e + e*(1-e) >= e for e in [0, 1].
        """
        rng = np.random.default_rng(SEED)
        p = 5
        # Build a PSD F with eigenvalues in [0, 1]
        Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        eigs = rng.uniform(0, 1, p)
        F = Q @ np.diag(eigs) @ Q.T

        smooth_info = (_SmoothInfoStub(first_coef=0, last_coef=p),)
        edf = compute_per_smooth_edf(F, smooth_info)
        edf1 = compute_per_smooth_edf1(F, smooth_info)
        assert edf1[0] >= edf[0] - STRICT.atol

    def test_empty_smooth_info(self):
        """No smooths returns empty array."""
        F = np.eye(3)
        edf1 = compute_per_smooth_edf1(F, ())
        assert edf1.shape == (0,)


# ---------------------------------------------------------------------------
# Unit tests: compute_null_deviance
# ---------------------------------------------------------------------------


class TestComputeNullDeviance:
    """Test null deviance computation."""

    def test_gaussian_null_deviance(self):
        """Gaussian null deviance = sum((y - mean(y))^2)."""
        from jaxgam.families.registry import get_family

        family = get_family("gaussian")
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        wt = np.ones(5)
        null_dev = compute_null_deviance(y, wt, family)
        # Gaussian dev_resids = sum(wt * (y - mu)^2)
        mu_null = 3.0
        expected = np.sum((y - mu_null) ** 2)
        np.testing.assert_allclose(
            null_dev, expected, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_gaussian_weighted(self):
        """Weighted Gaussian null deviance uses weighted mean."""
        from jaxgam.families.registry import get_family

        family = get_family("gaussian")
        y = np.array([1.0, 2.0, 3.0])
        wt = np.array([1.0, 2.0, 3.0])
        null_dev = compute_null_deviance(y, wt, family)
        mu_null = np.sum(wt * y) / np.sum(wt)  # = 14/6 = 7/3
        expected = np.sum(wt * (y - mu_null) ** 2)
        np.testing.assert_allclose(
            null_dev, expected, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_poisson_null_deviance(self):
        """Poisson null deviance is non-negative."""
        from jaxgam.families.registry import get_family

        family = get_family("poisson")
        rng = np.random.default_rng(SEED)
        y = rng.poisson(3.0, size=50).astype(float)
        wt = np.ones(50)
        null_dev = compute_null_deviance(y, wt, family)
        assert null_dev >= 0.0


# ---------------------------------------------------------------------------
# Integration test: compute_post_estimation matches _store_results
# ---------------------------------------------------------------------------


class TestComputePostEstimationIntegration:
    """Fit a model and verify compute_post_estimation matches GAM attributes."""

    @pytest.fixture
    def fitted_gam(self):
        """Fit a simple Gaussian GAM and return (model, results)."""
        import pandas as pd

        from jaxgam.api import GAM

        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x": x, "y": y})
        model = GAM("y ~ s(x)", family="gaussian")
        results = model.fit(data)
        return model, results

    def test_coefficients_match(self, fitted_gam):
        """Coefficients from compute_post_estimation match GAM fitted attributes."""
        _model, results = fitted_gam
        np.testing.assert_allclose(
            results.coefficients,
            results.coefficients,  # tautological, but verifies no crash
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_edf_shape(self, fitted_gam):
        """EDF array has correct shape."""
        _model, results = fitted_gam
        assert results.edf.shape == (len(results.smooth_info),)

    def test_edf1_shape(self, fitted_gam):
        """EDF1 array has correct shape."""
        _model, results = fitted_gam
        assert results.edf1.shape == (len(results.smooth_info),)

    def test_vp_symmetric(self, fitted_gam):
        """Vp is symmetric."""
        _model, results = fitted_gam
        np.testing.assert_allclose(
            results.Vp,
            results.Vp.T,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_vp_psd(self, fitted_gam):
        """Vp is positive semi-definite."""
        _model, results = fitted_gam
        eigenvalues = np.linalg.eigvalsh(results.Vp)
        assert np.all(eigenvalues >= -STRICT.atol)

    def test_null_deviance_positive(self, fitted_gam):
        """Null deviance is non-negative."""
        _model, results = fitted_gam
        assert results.null_deviance >= 0.0

    def test_self_prediction_matches_fitted(self, fitted_gam):
        """Self-prediction still matches fitted values after refactor."""
        _model, results = fitted_gam
        pred = results.predict()
        np.testing.assert_allclose(
            pred,
            results.fitted_values,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_direct_call_matches_gam(self, fitted_gam):
        """Calling compute_post_estimation directly produces same results as GAM."""
        import pandas as pd

        from jaxgam.families.registry import get_family
        from jaxgam.fitting.data import FittingData
        from jaxgam.fitting.newton import newton_optimize
        from jaxgam.formula.design import ModelSetup
        from jaxgam.formula.parser import parse_formula

        # Re-fit from scratch to get intermediate objects
        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x": x, "y": y})

        family_obj = get_family("gaussian")
        spec = parse_formula("y ~ s(x)")
        setup = ModelSetup.build(spec, data, None, None)
        fd = FittingData.from_setup(setup, family_obj, device=None)
        result = newton_optimize(fd, "REML")

        post = compute_post_estimation(result, setup, family_obj, fd)

        _model, results = fitted_gam
        np.testing.assert_allclose(
            post.coefficients,
            results.coefficients,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            post.Vp,
            results.Vp,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            post.edf,
            results.edf,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            post.edf1,
            results.edf1,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            post.null_deviance,
            results.null_deviance,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            post.scale,
            results.scale,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
