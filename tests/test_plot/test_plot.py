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

import matplotlib.pyplot as plt

from jaxgam.api import GAM
from tests.helpers import (
    SEED,
    _AssertCollector,
    _generate_family_data,
    check_that,
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
# A. Panel count tests
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
# B. Parameter tests
# ---------------------------------------------------------------------------


class TestParameters:
    """Test various plot parameter combinations."""

    @pytest.fixture
    def two_smooth_model(self, two_smooth_data):
        return GAM("y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')").fit(two_smooth_data)

    @pytest.mark.parametrize(
        ("select", "expected_visible"),
        [(0, 1), ([1], 1), ([0, 1], 2)],
        ids=["single", "list", "both"],
    )
    def test_select(self, two_smooth_model, select, expected_visible):
        """select chooses the expected number of smooth panels."""
        _fig, axes = two_smooth_model.plot(select=select)
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        assert len(visible_axes) == expected_visible

    def test_pages_one(self, two_smooth_model):
        """pages=1 arranges all smooths on one page."""
        _fig, axes = two_smooth_model.plot(pages=1)
        visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
        assert len(visible_axes) == 2

    @pytest.mark.parametrize(
        ("se", "shade", "expected"),
        [(False, True, "none"), (True, True, "shaded"), (True, False, "dashed")],
        ids=["off", "shaded", "dashed"],
    )
    def test_se_display(self, se, shade, expected):
        """SE display follows se/shade settings."""
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        _fig, axes = model.plot(se=se, shade=shade)
        ax = axes.ravel()[0]
        poly_collections = [
            c for c in ax.collections if "PolyCollection" in type(c).__name__
        ]
        lines = ax.get_lines()

        if expected == "none":
            assert len(poly_collections) == 0, "Expected no SE bands"
            assert len(lines) >= 1
        elif expected == "shaded":
            assert len(poly_collections) >= 1, "Expected shaded SE band"
        else:
            dashed = [ln for ln in lines if ln.get_linestyle() == "--"]
            assert len(dashed) == 2, "Expected 2 dashed SE lines"

    @pytest.mark.parametrize(
        ("rug", "expected_line_count"),
        [(True, 2), (False, 1)],
        ids=["on", "off"],
    )
    def test_rug_display(self, rug, expected_line_count):
        """rug toggles rug marks."""
        data = _generate_family_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        _fig, axes = model.plot(rug=rug, se=False)
        ax = axes.ravel()[0]
        lines = ax.get_lines()
        if rug:
            assert len(lines) >= expected_line_count
        else:
            assert len(lines) == expected_line_count


# ---------------------------------------------------------------------------
# C. Return value tests
# ---------------------------------------------------------------------------


class TestReturnValues:
    """Test that plot returns the expected types."""

    def test_multi_smooth_axes_shape(self, two_smooth_data):
        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        model = GAM(formula).fit(two_smooth_data)
        _fig, axes = model.plot()
        # axes should be 2D array
        assert axes.ndim == 2


# ---------------------------------------------------------------------------
# D. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

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
        """Verify that training_data is stored after fitting."""
        data = _generate_family_data("gaussian")
        results = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        assert hasattr(results, "training_data")
        assert "x" in results.training_data
        assert len(results.training_data["x"]) == len(data)


class TestM2PlotRegression:
    """Finding M2: 2D factor-by and random-effect smooths must plot, not crash."""

    def test_2d_factor_by_and_re_plot_without_crash(self):
        """te(x1,x2,by=fac) -> one contour panel per level; s(g,bs='re') -> QQ.

        Before the fix: te-by raised KeyError ('fac' not injected into the 2D
        grid) and s(g,bs='re') raised ValueError (string factor routed through
        the 1D numeric plotter).
        """
        import matplotlib.figure

        collector = _AssertCollector()

        rng = np.random.default_rng(SEED)
        n = 300
        df_te = pd.DataFrame(
            {
                "x1": rng.uniform(0, 1, n),
                "x2": rng.uniform(0, 1, n),
                "fac": pd.Categorical(rng.choice(["a", "b", "c"], n)),
                "y": rng.normal(0, 1, n),
            }
        )
        te_model = GAM("y ~ te(x1, x2, by=fac, k=5) + fac").fit(df_te)

        def _te_plot() -> None:
            fig, axes = te_model.plot()
            check_that(isinstance(fig, matplotlib.figure.Figure), "te-by: no Figure")
            visible = [ax for ax in np.asarray(axes).ravel() if ax.get_visible()]
            check_that(
                len(visible) == 3, f"expected 3 te-by panels, got {len(visible)}"
            )
            check_that(
                any(len(ax.collections) > 0 for ax in visible),
                "te-by panels have no contour collections",
            )
            plt.close(fig)

        collector.check("te_by_2d_plot", _te_plot)

        df_re = pd.DataFrame(
            {
                "g": pd.Categorical(rng.choice(list("abcdef"), 200)),
                "y": rng.normal(0, 1, 200),
            }
        )
        re_model = GAM("y ~ s(g, bs='re')").fit(df_re)

        def _re_plot() -> None:
            fig, axes = re_model.plot()
            check_that(isinstance(fig, matplotlib.figure.Figure), "re: no Figure")
            visible = [ax for ax in np.asarray(axes).ravel() if ax.get_visible()]
            check_that(len(visible) == 1, f"expected 1 RE panel, got {len(visible)}")
            check_that(len(visible[0].get_lines()) >= 1, "RE QQ panel has no lines")
            check_that(
                visible[0].get_xlabel() == "Gaussian quantiles",
                f"RE panel xlabel={visible[0].get_xlabel()!r} (expected QQ)",
            )
            plt.close(fig)

        collector.check("re_qq_plot", _re_plot)
        collector.raise_if_any("M2 plotting regression")
