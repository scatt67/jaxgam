"""Tests for jaxgam.results.GAMResults.

Tests cover:
- GAMResults construction metadata from mock data
- predict() with new data
- predict() with se_fit=True
- predict_matrix() returns correct shape
- Immutability (assigning to a frozen field raises)
- NB post-estimation: theta in results and summary

Design doc reference: docs/refactor_gam_api/implementation_plan.md Phase 2
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.registry import get_family
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.newton import newton_optimize
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.results import GAMResults
from tests.helpers import SEED, _AssertCollector, _make_nb_data, check_that
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

    def test_setup_aliases_are_properties(self, gam_results):
        """Setup-backed values are exposed without duplicate dataclass fields."""
        offset = np.linspace(0.1, 1.0, gam_results.n)
        setup_with_offset = replace(gam_results.setup, offset=offset)
        result_with_offset = replace(gam_results, setup=setup_with_offset)
        aliases = (
            "X",
            "y",
            "weights",
            "offset",
            "coef_map",
            "smooth_info",
            "term_names",
        )
        collector = _AssertCollector()
        for name in aliases:
            collector.check(
                f"{name} is a property",
                lambda name=name: check_that(
                    isinstance(getattr(GAMResults, name), property),
                    f"GAMResults.{name} is not a property",
                ),
            )
            collector.check(
                f"{name} is not stored",
                lambda name=name: check_that(
                    name not in GAMResults.__dataclass_fields__,
                    f"GAMResults.{name} remains a dataclass field",
                ),
            )
            collector.check(
                f"{name} delegates to setup",
                lambda name=name: check_that(
                    getattr(result_with_offset, name)
                    is getattr(result_with_offset.setup, name),
                    f"GAMResults.{name} does not return setup.{name}",
                ),
            )
        collector.raise_if_any("setup-backed GAMResults properties")


# ---------------------------------------------------------------------------
# Immutability tests
# ---------------------------------------------------------------------------


class TestImmutability:
    """Test that GAMResults is frozen."""

    def test_cannot_set_coefficients(self, gam_results):
        """Assigning to a frozen field raises."""
        with pytest.raises(AttributeError):
            gam_results.coefficients = np.zeros(10)


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


class TestPredict:
    """Test GAMResults.predict()."""

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

    def test_predict_se_with_new_data(self, gam_results, fit_data):
        """predict() with new data and se_fit=True."""
        pred, se = gam_results.predict(newdata=fit_data, se_fit=True)
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
# NB post-estimation tests (PR 6)
# ---------------------------------------------------------------------------


def _fit_nb_gam(family_obj=None, formula="y ~ s(x, k=10, bs='cr')"):
    """Fit NB GAM and return GAMResults via the full API pipeline."""
    import copy

    data = _make_nb_data()
    if family_obj is None:
        family_obj = NegativeBinomial()
    family_obj = copy.copy(family_obj)

    spec = parse_formula(formula)
    setup = ModelSetup.build(spec, data, None, None)
    fd = FittingData.from_setup(setup, family_obj, device=None)
    result = newton_optimize(fd, "REML")

    gam_result = GAMResults._from_fit(
        result=result,
        setup=setup,
        spec=spec,
        data=data,
        family=family_obj,
        fd=fd,
        lambda_strategy="newton_reml",
        formula=formula,
        method="REML",
    )
    return gam_result, data


class TestNBPostEstimation:
    """NB post-estimation: theta in results and summary."""

    @pytest.fixture(scope="class")
    def nb_result(self):
        """Fit NB model via _from_fit pipeline."""
        return _fit_nb_gam()

    def test_theta_populated(self, nb_result):
        """result.theta is populated for estimated NB."""
        gam_result, _ = nb_result
        assert gam_result.theta is not None
        assert isinstance(gam_result.theta, float)
        assert gam_result.theta > 0

    def test_theta_matches_family(self, nb_result):
        """result.theta matches the family's stored theta."""
        gam_result, _ = nb_result
        family_theta = float(gam_result.family.get_theta(transformed=True)[0])
        np.testing.assert_allclose(
            gam_result.theta,
            family_theta,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_summary_displays_theta(self, nb_result):
        """Summary output includes theta."""
        gam_result, _ = nb_result
        from jaxgam.summary.summary import GAMSummary

        s = gam_result.summary()
        assert isinstance(s, GAMSummary)
        assert s.theta is not None
        assert s.theta == gam_result.theta
        # Check formatted output contains "Theta:"
        formatted = str(s)
        assert "Theta:" in formatted

    def test_repr_contains_theta(self, nb_result):
        """repr includes theta for NB."""
        gam_result, _ = nb_result
        r = repr(gam_result)
        assert "theta=" in r

    def test_summary_family_name(self, nb_result):
        """Summary shows 'nb' as family name."""
        gam_result, _ = nb_result
        s = gam_result.summary()
        assert s.family_name == "nb"


class TestStandardFamilyThetaNone:
    """Standard families have theta=None in results and summary."""

    def test_gaussian_theta_none(self, gam_results):
        """Gaussian result.theta is None."""
        assert gam_results.theta is None

    def test_gaussian_repr_no_theta(self, gam_results):
        """Gaussian repr does not include theta."""
        r = repr(gam_results)
        assert "theta=" not in r

    def test_gaussian_summary_no_theta(self, gam_results):
        """Gaussian summary does not show theta."""
        s = gam_results.summary()
        assert s.theta is None
        formatted = str(s)
        assert "Theta:" not in formatted


class TestNBFixedThetaPostEstimation:
    """Fixed-theta NB: theta=None in results (n_theta=0, not estimated)."""

    @pytest.fixture(scope="class")
    def fixed_nb_result(self):
        """Fit NB(theta=2) fixed model."""
        return _fit_nb_gam(family_obj=NegativeBinomial(theta=2, fixed=True))

    def test_theta_is_none(self, fixed_nb_result):
        """Fixed-theta NB has result.theta=None (theta not estimated)."""
        gam_result, _ = fixed_nb_result
        assert gam_result.theta is None

    def test_summary_no_theta(self, fixed_nb_result):
        """Fixed-theta NB summary does not show Theta line."""
        gam_result, _ = fixed_nb_result
        s = gam_result.summary()
        assert s.theta is None
        formatted = str(s)
        assert "Theta:" not in formatted
