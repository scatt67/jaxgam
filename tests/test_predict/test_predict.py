"""Tests for GAM prediction (Task 3.1).

Tests cover:
- A. Self-prediction roundtrip (STRICT)
- B. New data prediction vs R (MODERATE / LOOSE)
- C. SE computation vs R (MODERATE / LOOSE)
- D. Multi-smooth and special smooth types
- E. Edge cases (purely parametric, offset)

Tolerance rationale:
  Self-prediction: STRICT (algebraic roundtrip, no numerical divergence).
  Gaussian new-data vs R: MODERATE (rtol=1e-4). GLM families: LOOSE
  (rtol=1e-2) because iterative PIRLS + Newton differences compound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.api import GAM
from tests.tolerances import LOOSE, MODERATE, STRICT

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r_available() -> bool:
    from tests.r_bridge import RBridge

    if not RBridge.available():
        return False
    ok, _ = RBridge.check_versions()
    return ok


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


def _make_newdata(family_name: str) -> pd.DataFrame:
    """Generate new data (different seed) for prediction tests."""
    rng = np.random.default_rng(SEED + 100)
    n = 50
    x = rng.uniform(0, 1, n)

    if family_name == "binomial":
        # Binomial needs a response column for the formula, but we don't use it
        y = np.zeros(n)
    elif family_name == "poisson":
        y = np.zeros(n)
    elif family_name == "gamma":
        y = np.ones(n)
    else:
        y = np.zeros(n)

    return pd.DataFrame({"x": x, "y": y})


def _make_two_smooth_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 200
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_factor_by_data(seed: int = SEED) -> pd.DataFrame:
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
    if family_name == "gaussian":
        return MODERATE
    return LOOSE


# ---------------------------------------------------------------------------
# A. Self-prediction roundtrip (STRICT)
# ---------------------------------------------------------------------------


class TestSelfPrediction:
    """Self-prediction must reproduce fitted values exactly."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(
        params=["gaussian", "poisson", "binomial", "gamma"],
        ids=["gaussian", "poisson", "binomial", "gamma"],
    )
    def fitted_model(self, request):
        family_name = request.param
        data = _make_data(family_name)
        model = GAM(self.FORMULA, family=family_name).fit(data)
        return family_name, model, data

    def test_predict_response_matches_fitted_values(self, fitted_model):
        _, model, _ = fitted_model
        pred = model.predict()
        np.testing.assert_allclose(
            pred,
            model.fitted_values_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="predict() != fitted_values_",
        )

    def test_predict_link_matches_linear_predictor(self, fitted_model):
        _, model, _ = fitted_model
        pred = model.predict(pred_type="link")
        np.testing.assert_allclose(
            pred,
            model.linear_predictor_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="predict(pred_type='link') != linear_predictor_",
        )

    def test_predict_matrix_times_coefs_matches_eta(self, fitted_model):
        _, model, data = fitted_model
        X_p = model.predict_matrix(data)
        eta_reconstructed = X_p @ model.coefficients_
        np.testing.assert_allclose(
            eta_reconstructed,
            model.linear_predictor_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="X_p @ coefs != linear_predictor_",
        )

    def test_predict_matrix_shape(self, fitted_model):
        _, model, data = fitted_model
        X_p = model.predict_matrix(data)
        assert X_p.shape == model.X_.shape

    def test_predict_matrix_matches_stored_X(self, fitted_model):
        """predict_matrix on training data should match stored X_."""
        _, model, data = fitted_model
        X_p = model.predict_matrix(data)
        np.testing.assert_allclose(
            X_p,
            model.X_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="predict_matrix(train_data) != X_",
        )


# ---------------------------------------------------------------------------
# B. New data prediction vs R (MODERATE / LOOSE)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestNewDataVsR:
    """New data predictions compared to R."""

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
    def prediction_pair(self, request):
        from tests.r_bridge import RBridge

        family_name, family_r = request.param
        train = _make_data(family_name)
        newdata = _make_newdata(family_name)

        model = GAM(self.FORMULA, family=family_name).fit(train)
        bridge = RBridge()

        r_response = bridge.predict_gam(
            self.FORMULA, train, newdata, family=family_r, pred_type="response"
        )
        r_link = bridge.predict_gam(
            self.FORMULA, train, newdata, family=family_r, pred_type="link"
        )

        return family_name, model, newdata, r_response, r_link

    def test_response_vs_r(self, prediction_pair):
        family_name, model, newdata, r_response, _ = prediction_pair
        tol = _r_tol(family_name)
        pred = model.predict(newdata, pred_type="response")
        np.testing.assert_allclose(
            pred,
            r_response["predictions"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} response prediction differs from R",
        )

    def test_link_vs_r(self, prediction_pair):
        family_name, model, newdata, _, r_link = prediction_pair
        tol = _r_tol(family_name)
        pred = model.predict(newdata, pred_type="link")
        np.testing.assert_allclose(
            pred,
            r_link["predictions"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} link prediction differs from R",
        )


# ---------------------------------------------------------------------------
# C. SE computation vs R (MODERATE / LOOSE)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestSEVsR:
    """Standard errors compared to R's predict.gam(se.fit=TRUE)."""

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
    def se_pair(self, request):
        from tests.r_bridge import RBridge

        family_name, family_r = request.param
        train = _make_data(family_name)
        newdata = _make_newdata(family_name)

        model = GAM(self.FORMULA, family=family_name).fit(train)
        bridge = RBridge()
        r_result = bridge.predict_gam(
            self.FORMULA, train, newdata, family=family_r, pred_type="link", se_fit=True
        )

        return family_name, model, newdata, r_result

    def test_se_vs_r(self, se_pair):
        family_name, model, newdata, r_result = se_pair
        tol = _r_tol(family_name)
        _, se = model.predict(newdata, pred_type="link", se_fit=True)
        np.testing.assert_allclose(
            se,
            r_result["se"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} SE differs from R",
        )


# ---------------------------------------------------------------------------
# D. Multi-smooth and special smooth types
# ---------------------------------------------------------------------------


class TestMultiSmoothPrediction:
    """Prediction for multi-smooth models."""

    def test_two_smooth_self_prediction(self):
        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        data = _make_two_smooth_data()
        model = GAM(formula).fit(data)

        pred = model.predict()
        np.testing.assert_allclose(
            pred,
            model.fitted_values_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Two-smooth self-prediction roundtrip failed",
        )

    def test_two_smooth_predict_matrix_shape(self):
        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        data = _make_two_smooth_data()
        model = GAM(formula).fit(data)

        X_p = model.predict_matrix(data)
        assert X_p.shape == model.X_.shape

    def test_tensor_product_self_prediction(self):
        formula = "y ~ te(x1, x2, k=5)"
        data = _make_two_smooth_data()
        model = GAM(formula).fit(data)

        pred = model.predict()
        np.testing.assert_allclose(
            pred,
            model.fitted_values_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Tensor product self-prediction roundtrip failed",
        )

    def test_factor_by_self_prediction(self):
        formula = "y ~ s(x, by=fac, k=10, bs='cr') + fac"
        data = _make_factor_by_data()
        model = GAM(formula).fit(data)

        pred = model.predict()
        np.testing.assert_allclose(
            pred,
            model.fitted_values_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Factor-by self-prediction roundtrip failed",
        )


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestMultiSmoothVsR:
    """Multi-smooth new-data prediction vs R."""

    def test_two_smooth_newdata_vs_r(self):
        from tests.r_bridge import RBridge

        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        train = _make_two_smooth_data()
        rng = np.random.default_rng(SEED + 200)
        newdata = pd.DataFrame(
            {
                "x1": rng.uniform(0, 1, 50),
                "x2": rng.uniform(0, 1, 50),
                "y": np.zeros(50),
            }
        )

        model = GAM(formula).fit(train)
        bridge = RBridge()
        r_result = bridge.predict_gam(formula, train, newdata, pred_type="response")

        pred = model.predict(newdata, pred_type="response")
        np.testing.assert_allclose(
            pred,
            r_result["predictions"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Two-smooth new-data prediction differs from R",
        )

    def test_tensor_product_newdata_vs_r(self):
        from tests.r_bridge import RBridge

        py_formula = "y ~ te(x1, x2, k=5)"
        r_formula = "y ~ te(x1, x2, k=c(5,5))"
        train = _make_two_smooth_data()
        rng = np.random.default_rng(SEED + 200)
        newdata = pd.DataFrame(
            {
                "x1": rng.uniform(0, 1, 50),
                "x2": rng.uniform(0, 1, 50),
                "y": np.zeros(50),
            }
        )

        model = GAM(py_formula).fit(train)
        bridge = RBridge()
        r_result = bridge.predict_gam(r_formula, train, newdata, pred_type="response")

        pred = model.predict(newdata, pred_type="response")
        np.testing.assert_allclose(
            pred,
            r_result["predictions"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Tensor product new-data prediction differs from R",
        )

    def test_factor_by_newdata_vs_r(self):
        from tests.r_bridge import RBridge

        formula = "y ~ s(x, by=fac, k=10, bs='cr') + fac"
        train = _make_factor_by_data()

        rng = np.random.default_rng(SEED + 200)
        n_new = 60
        x_new = rng.uniform(0, 1, n_new)
        fac_new = rng.choice(["a", "b", "c"], n_new)
        newdata = pd.DataFrame(
            {
                "x": x_new,
                "fac": pd.Categorical(fac_new, categories=["a", "b", "c"]),
                "y": np.zeros(n_new),
            }
        )

        model = GAM(formula).fit(train)
        bridge = RBridge()
        r_result = bridge.predict_gam(formula, train, newdata, pred_type="response")

        pred = model.predict(newdata, pred_type="response")
        np.testing.assert_allclose(
            pred,
            r_result["predictions"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Factor-by new-data prediction differs from R",
        )


# ---------------------------------------------------------------------------
# E. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for prediction."""

    def test_purely_parametric_predict(self):
        rng = np.random.default_rng(SEED)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = 2.0 * x1 - 1.0 * x2 + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        model = GAM("y ~ x1 + x2").fit(data)

        # Self-prediction roundtrip
        pred = model.predict()
        np.testing.assert_allclose(
            pred,
            model.fitted_values_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

        # New data prediction
        rng2 = np.random.default_rng(SEED + 300)
        newdata = pd.DataFrame(
            {
                "x1": rng2.uniform(0, 1, 30),
                "x2": rng2.uniform(0, 1, 30),
                "y": np.zeros(30),
            }
        )
        pred_new = model.predict(newdata)
        assert pred_new.shape == (30,)
        assert np.all(np.isfinite(pred_new))

    def test_offset_predict(self):
        data = _make_data("gaussian")
        n = len(data)
        offset = np.ones(n) * 0.5
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data, offset=offset)

        # Self-prediction roundtrip
        pred = model.predict()
        np.testing.assert_allclose(
            pred,
            model.fitted_values_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_predict_with_newdata_offset(self):
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)

        newdata = _make_newdata("gaussian")
        offset = np.ones(len(newdata)) * 0.5
        pred_no_offset = model.predict(newdata, pred_type="link")
        pred_with_offset = model.predict(newdata, pred_type="link", offset=offset)

        np.testing.assert_allclose(
            pred_with_offset,
            pred_no_offset + 0.5,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Offset not applied correctly in predict",
        )

    def test_se_fit_returns_tuple(self):
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        result = model.predict(se_fit=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        pred, se = result
        assert pred.shape == model.fitted_values_.shape
        assert se.shape == model.fitted_values_.shape
        assert np.all(se >= 0), "SE must be non-negative"

    def test_invalid_type_raises(self):
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        with pytest.raises(ValueError, match="pred_type must be"):
            model.predict(pred_type="terms")

    def test_unfitted_predict_matrix_raises(self):
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not fitted yet"):
            model.predict_matrix({"x": np.array([1.0])})
