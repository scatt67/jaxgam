"""Tests for the R bridge (pymgcv.compat.r_bridge).

These tests require R with mgcv installed.
They are automatically skipped if R is not available.
"""

import numpy as np
import pandas as pd
import pytest

from pymgcv.compat.r_bridge import RBridge


@pytest.fixture
def gaussian_data() -> pd.DataFrame:
    """Small Gaussian dataset for R bridge testing."""
    rng = np.random.default_rng(123)
    n = 100
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x": x, "y": y})


class TestRBridgeAvailability:
    def test_available_returns_bool(self) -> None:
        result = RBridge.available()
        assert isinstance(result, bool)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            RBridge(mode="invalid")


@pytest.mark.skipif(not RBridge.available(), reason="R with mgcv not available")
class TestRBridgeFitGam:
    def test_fit_gaussian_returns_expected_keys(
        self, gaussian_data: pd.DataFrame
    ) -> None:
        bridge = RBridge()
        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        expected_keys = {
            "coefficients",
            "fitted_values",
            "smoothing_params",
            "edf",
            "deviance",
            "scale",
            "reml_scale",
            "Vp",
            "reml_score",
        }
        assert set(result.keys()) == expected_keys

    def test_fit_gaussian_types(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        assert isinstance(result["coefficients"], np.ndarray)
        assert result["coefficients"].dtype == np.float64
        assert isinstance(result["fitted_values"], np.ndarray)
        assert result["fitted_values"].dtype == np.float64
        assert isinstance(result["smoothing_params"], np.ndarray)
        assert isinstance(result["edf"], np.ndarray)
        assert isinstance(result["deviance"], float)
        assert isinstance(result["scale"], float)
        assert isinstance(result["Vp"], np.ndarray)
        assert isinstance(result["reml_score"], float)

    def test_fit_gaussian_dimensions(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        n = len(gaussian_data)
        p = len(result["coefficients"])

        assert result["fitted_values"].shape == (n,)
        assert result["Vp"].shape == (p, p)
        assert result["deviance"] >= 0
        assert result["scale"] > 0

    def test_fit_gaussian_no_nan(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        assert np.all(np.isfinite(result["coefficients"]))
        assert np.all(np.isfinite(result["fitted_values"]))
        assert np.isfinite(result["deviance"])

    def test_fit_all_families(self, gaussian_data: pd.DataFrame) -> None:
        """All four v1.0 families should work."""
        bridge = RBridge()
        rng = np.random.default_rng(456)
        n = 100
        x = rng.uniform(0, 1, n)

        # Gaussian
        result = bridge.fit_gam(
            "y ~ s(x)",
            pd.DataFrame({"x": x, "y": np.sin(x) + rng.normal(0, 0.2, n)}),
            family="gaussian",
        )
        assert len(result["coefficients"]) > 0

        # Binomial
        eta = 2 * np.sin(2 * np.pi * x)
        p = 1 / (1 + np.exp(-eta))
        result = bridge.fit_gam(
            "y ~ s(x)",
            pd.DataFrame({"x": x, "y": rng.binomial(1, p, n).astype(float)}),
            family="binomial",
        )
        assert len(result["coefficients"]) > 0

        # Poisson
        result = bridge.fit_gam(
            "y ~ s(x)",
            pd.DataFrame(
                {"x": x, "y": rng.poisson(np.exp(np.sin(x)), n).astype(float)}
            ),
            family="poisson",
        )
        assert len(result["coefficients"]) > 0

        # Gamma
        mu = np.exp(0.5 * np.sin(2 * np.pi * x) + 1)
        result = bridge.fit_gam(
            "y ~ s(x)",
            pd.DataFrame({"x": x, "y": rng.gamma(5, scale=mu / 5, size=n)}),
            family="gamma",
        )
        assert len(result["coefficients"]) > 0

    def test_invalid_family_raises(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        with pytest.raises(ValueError, match="Unknown family"):
            bridge.fit_gam("y ~ s(x)", gaussian_data, family="tweedie")

    def test_vp_is_symmetric(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")
        vp = result["Vp"]
        np.testing.assert_allclose(vp, vp.T, atol=1e-12, err_msg="Vp must be symmetric")


@pytest.mark.skipif(not RBridge.available(), reason="R with mgcv not available")
class TestRBridgeSmoothComponents:
    def test_get_smooth_components_keys(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.get_smooth_components(
            "y ~ s(x)", gaussian_data, family="gaussian"
        )

        # Should have all fit_gam keys plus basis/penalty
        assert "basis_matrices" in result
        assert "penalty_matrices" in result
        assert "coefficients" in result

    def test_basis_matrix_dimensions(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.get_smooth_components(
            "y ~ s(x, k=10)", gaussian_data, family="gaussian"
        )

        assert len(result["basis_matrices"]) == 1  # one smooth
        X_s = result["basis_matrices"][0]
        assert X_s.shape[0] == len(gaussian_data)
        assert X_s.shape[1] == 9  # k-1 due to identifiability constraint

    def test_penalty_matrix_psd(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.get_smooth_components(
            "y ~ s(x, k=10)", gaussian_data, family="gaussian"
        )

        for penalties in result["penalty_matrices"]:
            for S in penalties:
                assert S.shape[0] == S.shape[1], "Penalty must be square"
                eigenvalues = np.linalg.eigvalsh(S)
                assert np.all(eigenvalues >= -1e-10), "Penalty must be PSD"

    def test_multi_smooth_components(self) -> None:
        bridge = RBridge()
        rng = np.random.default_rng(789)
        n = 100
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + 0.5 * x2 + rng.normal(0, 0.2, n)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        result = bridge.get_smooth_components(
            "y ~ s(x1) + s(x2)", data, family="gaussian"
        )

        assert len(result["basis_matrices"]) == 2
        assert len(result["penalty_matrices"]) == 2


@pytest.mark.skipif(not RBridge.available(), reason="R with mgcv not available")
class TestRBridgeSubprocessFallback:
    def test_subprocess_mode_works(self, gaussian_data: pd.DataFrame) -> None:
        """Subprocess mode should produce same structure as rpy2."""
        bridge = RBridge(mode="subprocess")
        assert bridge.mode == "subprocess"

        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        assert "coefficients" in result
        assert isinstance(result["coefficients"], np.ndarray)
        assert result["deviance"] >= 0

    def test_subprocess_invalid_family_raises(
        self, gaussian_data: pd.DataFrame
    ) -> None:
        bridge = RBridge(mode="subprocess")
        with pytest.raises(ValueError, match="Unknown family"):
            bridge.fit_gam("y ~ s(x)", gaussian_data, family="tweedie")

    def test_subprocess_smooth_components(self, gaussian_data: pd.DataFrame) -> None:
        """Subprocess get_smooth_components should return basis/penalty."""
        bridge = RBridge(mode="subprocess")
        result = bridge.get_smooth_components(
            "y ~ s(x, k=10)", gaussian_data, family="gaussian"
        )

        assert "basis_matrices" in result
        assert "penalty_matrices" in result
        assert len(result["basis_matrices"]) == 1
        X_s = result["basis_matrices"][0]
        assert X_s.shape[0] == len(gaussian_data)
        assert X_s.shape[1] == 9  # k-1

    def test_subprocess_smooth_components_matches_rpy2(
        self, gaussian_data: pd.DataFrame
    ) -> None:
        """Subprocess and rpy2 smooth components should match."""
        try:
            bridge_rpy2 = RBridge(mode="rpy2")
        except Exception:
            pytest.skip("rpy2 not available for comparison")

        bridge_sub = RBridge(mode="subprocess")

        r1 = bridge_rpy2.get_smooth_components(
            "y ~ s(x, k=10)", gaussian_data, family="gaussian"
        )
        r2 = bridge_sub.get_smooth_components(
            "y ~ s(x, k=10)", gaussian_data, family="gaussian"
        )

        np.testing.assert_allclose(r1["coefficients"], r2["coefficients"], rtol=1e-10)
        np.testing.assert_allclose(
            r1["basis_matrices"][0], r2["basis_matrices"][0], rtol=1e-10
        )
        for s1, s2 in zip(r1["penalty_matrices"][0], r2["penalty_matrices"][0]):
            np.testing.assert_allclose(s1, s2, rtol=1e-10, atol=1e-14)

    def test_subprocess_matches_rpy2(self, gaussian_data: pd.DataFrame) -> None:
        """Both modes should produce identical results."""
        try:
            bridge_rpy2 = RBridge(mode="rpy2")
        except Exception:
            pytest.skip("rpy2 not available for comparison")

        bridge_sub = RBridge(mode="subprocess")

        r1 = bridge_rpy2.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")
        r2 = bridge_sub.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        np.testing.assert_allclose(r1["coefficients"], r2["coefficients"], rtol=1e-10)
        np.testing.assert_allclose(r1["fitted_values"], r2["fitted_values"], rtol=1e-10)
        np.testing.assert_allclose(r1["deviance"], r2["deviance"], rtol=1e-10)
