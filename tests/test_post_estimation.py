"""Tests for post-estimation helpers in jaxgam.results.

Tests cover:
- Unit tests for EDF computation with known hat matrices
- Unit tests for null deviance with known family/data
- Integration test: fit a model, verify GAMResults post-estimation outputs
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from jaxgam.results import (
    _compute_null_deviance,
    _compute_per_smooth_edf,
    _compute_per_smooth_edf1,
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
# Unit tests: _compute_per_smooth_edf
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
        edf = _compute_per_smooth_edf(F, smooth_info)
        np.testing.assert_allclose(edf, [3.0, 3.0], rtol=STRICT.rtol, atol=STRICT.atol)

    def test_known_diagonal(self):
        """Diagonal hat matrix with known values."""
        F = np.diag([0.9, 0.8, 0.5, 0.3, 0.1])
        smooth_info = (
            _SmoothInfoStub(first_coef=0, last_coef=2),
            _SmoothInfoStub(first_coef=2, last_coef=5),
        )
        edf = _compute_per_smooth_edf(F, smooth_info)
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
        edf = _compute_per_smooth_edf(F, smooth_info)
        np.testing.assert_allclose(edf, [1.3], rtol=STRICT.rtol, atol=STRICT.atol)


# ---------------------------------------------------------------------------
# Unit tests: _compute_per_smooth_edf1
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
        edf1 = _compute_per_smooth_edf1(F, smooth_info)
        np.testing.assert_allclose(edf1, [3.0, 3.0], rtol=STRICT.rtol, atol=STRICT.atol)

    def test_known_diagonal(self):
        """Diagonal F: edf1_i = 2*F[i,i] - F[i,i]^2 per coefficient."""
        diag_vals = np.array([0.9, 0.5, 0.3])
        F = np.diag(diag_vals)
        smooth_info = (
            _SmoothInfoStub(first_coef=0, last_coef=2),
            _SmoothInfoStub(first_coef=2, last_coef=3),
        )
        edf1 = _compute_per_smooth_edf1(F, smooth_info)
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
        edf = _compute_per_smooth_edf(F, smooth_info)
        edf1 = _compute_per_smooth_edf1(F, smooth_info)
        assert edf1[0] >= edf[0] - STRICT.atol


# ---------------------------------------------------------------------------
# Unit tests: _compute_null_deviance
# ---------------------------------------------------------------------------


class TestComputeNullDeviance:
    """Test null deviance computation."""

    def test_gaussian_null_deviance(self):
        """Gaussian null deviance = sum((y - mean(y))^2)."""
        from jaxgam.families.registry import get_family

        family = get_family("gaussian")
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        wt = np.ones(5)
        null_dev = _compute_null_deviance(y, wt, family)
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
        null_dev = _compute_null_deviance(y, wt, family)
        mu_null = np.sum(wt * y) / np.sum(wt)  # = 14/6 = 7/3
        expected = np.sum(wt * (y - mu_null) ** 2)
        np.testing.assert_allclose(
            null_dev, expected, rtol=STRICT.rtol, atol=STRICT.atol
        )


# ---------------------------------------------------------------------------
# Integration tests: post-estimation via GAMResults
# ---------------------------------------------------------------------------


class TestPostEstimationIntegration:
    """Fit a model and verify post-estimation outputs on GAMResults."""

    @pytest.fixture
    def results(self):
        """Fit a simple Gaussian GAM and return GAMResults."""
        import pandas as pd

        from jaxgam.api import GAM

        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x": x, "y": y})
        return GAM("y ~ s(x)", family="gaussian").fit(data)

    def test_edf_shape(self, results):
        """EDF array has correct shape."""
        assert results.edf.shape == (len(results.smooth_info),)

    def test_edf1_shape(self, results):
        """EDF1 array has correct shape."""
        assert results.edf1.shape == (len(results.smooth_info),)
