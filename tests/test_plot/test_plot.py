"""Tests for GAM plotting (Task 3.3).

Tests cover:
- A. Smoke tests: plot() produces a figure without error for each smooth type
- B. Parameter tests: select, pages, se, shade, rug
- C. Return value tests: (fig, axes) tuple shape and type
- D. Edge cases: unfitted model, purely parametric model

Uses Agg backend to avoid display issues in CI.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.figure
import matplotlib.pyplot as plt

from jaxgam.api import GAM
from tests.helpers import (
    SEED,
    _generate_family_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensor_data(seed: int = SEED) -> pd.DataFrame:
    """Generate data for a tensor product model."""
    rng = np.random.default_rng(seed)
    n = 200
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# A. Smoke tests — plot() produces a figure for each smooth type
# ---------------------------------------------------------------------------


class TestSmokeTests:
    """plot() runs without error for each major smooth type."""

    @pytest.fixture(
        params=["gaussian", "poisson", "binomial", "gamma"],
        ids=["gaussian", "poisson", "binomial", "gamma"],
    )
    def single_smooth_model(self, request):
        family_name = request.param
        data = _generate_family_data(family_name, n=200)
        model = GAM("y ~ s(x, k=10, bs='cr')", family=family_name).fit(data)
        return model

    def test_single_smooth_plot(self, single_smooth_model):
        """Single smooth model plots without error."""
        fig, axes = single_smooth_model.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes is not None

    def test_multi_smooth_plot(self, two_smooth_data):
        """Two-smooth model plots without error."""
        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        model = GAM(formula).fit(two_smooth_data)
        fig, _axes = model.plot()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_tensor_product_plot(self):
        """Tensor product model plots without error."""
        data = _make_tensor_data()
        model = GAM("y ~ te(x1, x2, k=5)").fit(data)
        fig, _axes = model.plot()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_factor_by_plot(self, factor_by_data):
        """Factor-by model plots without error."""
        model = GAM("y ~ s(x, by=fac, k=10, bs='cr') + fac").fit(factor_by_data)
        fig, _axes = model.plot()
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# B. Panel count tests
# ---------------------------------------------------------------------------


class TestPanelCounts:
    """Correct number of panels/axes for each model type."""

    def test_single_smooth_has_one_panel(self):
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        _fig, axes = model.plot()
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        assert len(visible_axes) == 1

    def test_two_smooth_has_two_panels(self, two_smooth_data):
        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        model = GAM(formula).fit(two_smooth_data)
        _fig, axes = model.plot()
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        assert len(visible_axes) == 2

    def test_tensor_product_has_one_panel(self):
        data = _make_tensor_data()
        model = GAM("y ~ te(x1, x2, k=5)").fit(data)
        _fig, axes = model.plot()
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        assert len(visible_axes) == 1

    def test_factor_by_has_one_panel_per_level(self, factor_by_data):
        model = GAM("y ~ s(x, by=fac, k=10, bs='cr') + fac").fit(factor_by_data)
        _fig, axes = model.plot()
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        # 3 levels = 3 panels
        assert len(visible_axes) == 3


# ---------------------------------------------------------------------------
# C. Parameter tests
# ---------------------------------------------------------------------------


class TestParameters:
    """Test various plot parameter combinations."""

    @pytest.fixture
    def two_smooth_model(self, two_smooth_data):
        return GAM("y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')").fit(two_smooth_data)

    def test_select_single(self, two_smooth_model):
        """select=0 shows only first smooth."""
        _fig, axes = two_smooth_model.plot(select=0)
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        assert len(visible_axes) == 1

    def test_select_list(self, two_smooth_model):
        """select=[1] shows only second smooth."""
        _fig, axes = two_smooth_model.plot(select=[1])
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        assert len(visible_axes) == 1

    def test_select_both(self, two_smooth_model):
        """select=[0, 1] shows both smooths."""
        _fig, axes = two_smooth_model.plot(select=[0, 1])
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        assert len(visible_axes) == 2

    def test_pages_one(self, two_smooth_model):
        """pages=1 arranges all smooths on one page."""
        _fig, axes = two_smooth_model.plot(pages=1)
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        assert len(visible_axes) == 2

    def test_se_false_no_bands(self):
        """se=False produces no SE bands."""
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        _fig, axes = model.plot(se=False)
        ax = axes.ravel()[0]
        # With se=False, should have no fill_between collections
        poly_collections = [
            c for c in ax.collections if "PolyCollection" in type(c).__name__
        ]
        assert len(poly_collections) == 0, "Expected no SE bands"
        # Should have at least the smooth effect line
        lines = ax.get_lines()
        assert len(lines) >= 1

    def test_se_true_has_bands(self):
        """se=True with shade produces fill_between."""
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        _fig, axes = model.plot(se=True, shade=True)
        ax = axes.ravel()[0]
        # Should have a fill_between collection
        # (FillBetweenPolyCollection in modern matplotlib)
        poly_collections = [
            c for c in ax.collections if "PolyCollection" in type(c).__name__
        ]
        assert len(poly_collections) >= 1, "Expected shaded SE band"

    def test_shade_false_dashed_lines(self):
        """shade=False produces dashed SE lines instead of shading."""
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        _fig, axes = model.plot(se=True, shade=False)
        ax = axes.ravel()[0]
        lines = ax.get_lines()
        # Should have 3 lines: smooth effect + 2 SE boundary lines
        # (plus possibly rug marks)
        dashed = [ln for ln in lines if ln.get_linestyle() == "--"]
        assert len(dashed) == 2, "Expected 2 dashed SE lines"

    def test_rug_true(self):
        """rug=True adds rug marks."""
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        _fig, axes = model.plot(rug=True, se=False)
        ax = axes.ravel()[0]
        lines = ax.get_lines()
        # Should have more than 1 line (smooth + rug)
        assert len(lines) >= 2, "Expected rug marks"

    def test_rug_false_no_rug(self):
        """rug=False produces no rug marks."""
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        _fig, axes = model.plot(rug=False, se=False)
        ax = axes.ravel()[0]
        lines = ax.get_lines()
        # Should have only 1 line (the smooth effect, no rug)
        assert len(lines) == 1, "Expected only the smooth effect line"


# ---------------------------------------------------------------------------
# D. Return value tests
# ---------------------------------------------------------------------------


class TestReturnValues:
    """Test that plot returns the expected types."""

    def test_returns_tuple(self):
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        result = model.plot()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fig_is_figure(self):
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        fig, _axes = model.plot()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_axes_is_ndarray(self):
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        _fig, axes = model.plot()
        assert isinstance(axes, np.ndarray)

    def test_multi_smooth_axes_shape(self, two_smooth_data):
        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        model = GAM(formula).fit(two_smooth_data)
        _fig, axes = model.plot()
        # axes should be 2D array
        assert axes.ndim == 2


# ---------------------------------------------------------------------------
# E. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_unfitted_raises(self):
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not fitted yet"):
            model.plot()

    def test_select_out_of_range_raises(self):
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        with pytest.raises(ValueError, match="out of range"):
            model.plot(select=5)

    def test_purely_parametric_raises(self):
        rng = np.random.default_rng(SEED)
        n = 200
        data = pd.DataFrame(
            {
                "x": rng.uniform(0, 1, n),
                "y": rng.normal(0, 1, n),
            }
        )
        model = GAM("y ~ x").fit(data)
        with pytest.raises(ValueError, match="No smooth terms"):
            model.plot()

    def test_labels_contain_edf(self):
        """Y-axis labels include EDF value."""
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        _fig, axes = model.plot()
        ax = axes.ravel()[0]
        ylabel = ax.get_ylabel()
        # Should be something like "s(x,3.45)"
        assert "s(x," in ylabel
        assert ")" in ylabel

    def test_2d_has_colorbar(self):
        """2D contour plot includes a colorbar."""
        data = _make_tensor_data()
        model = GAM("y ~ te(x1, x2, k=5)").fit(data)
        fig, _axes = model.plot()
        # The colorbar adds an extra axes to the figure
        assert len(fig.axes) > 1

    def test_factor_by_titles_contain_level(self, factor_by_data):
        """Factor-by panels have the level name in the title."""
        model = GAM("y ~ s(x, by=fac, k=10, bs='cr') + fac").fit(factor_by_data)
        _fig, axes = model.plot()
        titles = [ax.get_title() for ax in axes.ravel() if ax.get_visible()]
        # Should have 3 titles: "a", "b", "c"
        assert len(titles) == 3
        assert "a" in titles
        assert "b" in titles
        assert "c" in titles

    def test_training_data_stored(self):
        """Verify that _training_data is stored after fitting."""
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        assert hasattr(model, "_training_data")
        assert "x" in model._training_data
        assert len(model._training_data["x"]) == len(data)
