"""Tests for jaxgam.results.GAMResults.

Tests cover:
- GAMResults construction from mock data
- predict() self-prediction matches fitted_values
- predict() with new data
- predict() with se_fit=True
- predict_matrix() returns correct shape
- summary() returns GAMSummary
- Immutability (assigning to a frozen field raises)
- __repr__() output

Design doc reference: docs/refactor_gam_api/implementation_plan.md Phase 2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.families.registry import get_family
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.newton import newton_optimize
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.results import GAMResults
from tests.helpers import SEED
from tests.tolerances import STRICT

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fit_artifacts():
    """Fit a simple Gaussian GAM and return intermediate artifacts."""
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

    return {
        "result": result,
        "setup": setup,
        "spec": spec,
        "data": data,
        "family": family_obj,
        "fd": fd,
    }


@pytest.fixture
def gam_results(fit_artifacts):
    """Construct a GAMResults via _from_fit()."""
    return GAMResults._from_fit(
        result=fit_artifacts["result"],
        setup=fit_artifacts["setup"],
        spec=fit_artifacts["spec"],
        data=fit_artifacts["data"],
        family=fit_artifacts["family"],
        fd=fit_artifacts["fd"],
        lambda_strategy="newton_reml",
        formula="y ~ s(x)",
        method="REML",
    )


@pytest.fixture
def fit_data(fit_artifacts):
    """Return the training data DataFrame."""
    return fit_artifacts["data"]


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestGAMResultsConstruction:
    """Test GAMResults construction from _from_fit()."""

    def test_construction_succeeds(self, gam_results):
        """_from_fit produces a GAMResults instance."""
        assert isinstance(gam_results, GAMResults)

    def test_formula_stored(self, gam_results):
        """Formula is stored as metadata."""
        assert gam_results.formula == "y ~ s(x)"

    def test_method_stored(self, gam_results):
        """Method is stored as metadata."""
        assert gam_results.method == "REML"

    def test_family_is_object(self, gam_results):
        """Family is an ExponentialFamily object, not a string."""
        from jaxgam.families.base import ExponentialFamily

        assert isinstance(gam_results.family, ExponentialFamily)

    def test_no_trailing_underscores(self, gam_results):
        """No attributes with trailing underscores (design §3.4 #2)."""
        assert hasattr(gam_results, "coefficients")
        assert not hasattr(gam_results, "coefficients_")

    def test_no_ve_placeholder(self, gam_results):
        """Ve is omitted entirely (design §9 #4)."""
        assert not hasattr(gam_results, "Ve")
        assert not hasattr(gam_results, "Ve_")

    def test_coefficients_shape(self, gam_results):
        """Coefficients have correct shape."""
        p = gam_results.X.shape[1]
        assert gam_results.coefficients.shape == (p,)

    def test_edf_shape(self, gam_results):
        """EDF array has one entry per smooth."""
        assert gam_results.edf.shape == (
            len(gam_results.smooth_info),
        )

    def test_n_matches_data(self, gam_results):
        """n matches the number of observations."""
        assert gam_results.n == gam_results.X.shape[0]


# ---------------------------------------------------------------------------
# Immutability tests
# ---------------------------------------------------------------------------


class TestImmutability:
    """Test that GAMResults is frozen."""

    def test_cannot_set_coefficients(self, gam_results):
        """Assigning to a frozen field raises."""
        with pytest.raises(AttributeError):
            gam_results.coefficients = np.zeros(10)

    def test_cannot_set_scale(self, gam_results):
        """Assigning to a frozen scalar field raises."""
        with pytest.raises(AttributeError):
            gam_results.scale = 999.0

    def test_cannot_add_new_attribute(self, gam_results):
        """Adding a new attribute raises."""
        with pytest.raises(AttributeError):
            gam_results.new_attr = "test"


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


class TestPredict:
    """Test GAMResults.predict()."""

    def test_self_prediction_matches_fitted(self, gam_results):
        """predict() with no newdata matches fitted_values."""
        pred = gam_results.predict()
        np.testing.assert_allclose(
            pred,
            gam_results.fitted_values,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_predict_with_new_data(self, gam_results, fit_data):
        """predict() with new data returns correct shape."""
        pred = gam_results.predict(newdata=fit_data)
        assert pred.shape == (len(fit_data),)

    def test_predict_link_scale(self, gam_results):
        """predict(pred_type='link') returns linear predictor."""
        pred_link = gam_results.predict(pred_type="link")
        np.testing.assert_allclose(
            pred_link,
            gam_results.linear_predictor,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_predict_se_fit(self, gam_results):
        """predict(se_fit=True) returns (pred, se) tuple."""
        result = gam_results.predict(se_fit=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        pred, se = result
        assert pred.shape == gam_results.fitted_values.shape
        assert se.shape == gam_results.fitted_values.shape
        assert np.all(se >= 0)

    def test_predict_se_with_new_data(
        self, gam_results, fit_data
    ):
        """predict() with new data and se_fit=True."""
        pred, se = gam_results.predict(
            newdata=fit_data, se_fit=True
        )
        assert pred.shape == (len(fit_data),)
        assert se.shape == (len(fit_data),)
        assert np.all(se >= 0)

    def test_predict_invalid_type(self, gam_results):
        """predict() with invalid pred_type raises."""
        with pytest.raises(ValueError, match="pred_type"):
            gam_results.predict(pred_type="invalid")


# ---------------------------------------------------------------------------
# predict_matrix tests
# ---------------------------------------------------------------------------


class TestPredictMatrix:
    """Test GAMResults.predict_matrix()."""

    def test_shape(self, gam_results, fit_data):
        """predict_matrix() returns correct shape."""
        X_pred = gam_results.predict_matrix(fit_data)
        n_new = len(fit_data)
        p = gam_results.coefficients.shape[0]
        assert X_pred.shape == (n_new, p)


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------


class TestSummary:
    """Test GAMResults.summary()."""

    def test_returns_gam_summary(self, gam_results):
        """summary() returns a GAMSummary."""
        from jaxgam.summary.summary import GAMSummary

        s = gam_results.summary()
        assert isinstance(s, GAMSummary)

    def test_summary_formula(self, gam_results):
        """Summary shows the formula."""
        s = gam_results.summary()
        assert s.formula == "y ~ s(x)"

    def test_summary_method(self, gam_results):
        """Summary shows the method."""
        s = gam_results.summary()
        assert s.method == "REML"


# ---------------------------------------------------------------------------
# Repr tests
# ---------------------------------------------------------------------------


class TestRepr:
    """Test GAMResults.__repr__()."""

    def test_repr_contains_formula(self, gam_results):
        """repr includes formula."""
        r = repr(gam_results)
        assert "y ~ s(x)" in r

    def test_repr_contains_family(self, gam_results):
        """repr includes family name."""
        r = repr(gam_results)
        assert "gaussian" in r.lower()

    def test_repr_contains_converged(self, gam_results):
        """repr includes convergence status."""
        r = repr(gam_results)
        assert "converged=" in r

    def test_repr_contains_deviance_explained(self, gam_results):
        """repr includes deviance explained."""
        r = repr(gam_results)
        assert "deviance_explained=" in r


# ---------------------------------------------------------------------------
# Matches legacy GAM tests
# ---------------------------------------------------------------------------


class TestMatchesLegacyGAM:
    """Verify GAMResults produces same outputs as legacy GAM."""

    @pytest.fixture
    def legacy_gam(self, fit_data):
        """Fit a model using legacy GAM API."""
        from jaxgam.api import GAM

        model = GAM("y ~ s(x)", family="gaussian")
        model.fit(fit_data)
        return model

    def test_coefficients_match(
        self, gam_results, legacy_gam
    ):
        """Coefficients from GAMResults match legacy GAM."""
        np.testing.assert_allclose(
            gam_results.coefficients,
            legacy_gam.coefficients_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_vp_match(self, gam_results, legacy_gam):
        """Vp from GAMResults matches legacy GAM."""
        np.testing.assert_allclose(
            gam_results.Vp,
            legacy_gam.Vp_,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_predict_match(self, gam_results, legacy_gam):
        """Predictions match between GAMResults and legacy GAM."""
        pred_new = gam_results.predict()
        pred_old = legacy_gam.predict()
        np.testing.assert_allclose(
            pred_new,
            pred_old,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
