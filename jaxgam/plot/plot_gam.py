"""Smooth component visualization equivalent to R's plot.gam().

Provides ``plot_gam()`` for visualizing the smooth components of a fitted
GAM model, equivalent to R's ``plot.gam()``.

Supports:
- 1D smooths ``s(x)``: line plot with optional SE bands and rug
- 2D smooths ``te(x1, x2)`` / ``s(x1, x2)``: filled contour plot
- Factor-by smooths ``s(x, by=fac)``: one panel per factor level
- Numeric-by smooths ``s(x, by=z)``: same as 1D smooth

This module is Phase 3 (NumPy + matplotlib only, no JAX imports).

Design doc reference: docs/design.md Section 10
R source reference: R/plots.r plot.gam(), plot.mgcv.smooth()
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from jaxgam.api import GAM
    from jaxgam.smooths.constraints import TermBlock


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def plot_gam(
    model: GAM,
    select: int | list[int] | None = None,
    pages: int = 0,  # noqa: ARG001
    rug: bool = True,
    se: bool = True,
    shade: bool = True,
    n_grid: int = 200,
    n2: int = 40,
    shade_color: str = "#C8C8C8",
    se_mult: float = 2.0,
    line_color: str = "black",
    se_line_color: str = "black",
    figsize: tuple[float, float] | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot smooth components of a fitted GAM.

    Equivalent to R's ``plot.gam()``. For each smooth term in the model,
    produces a plot of the partial effect (on the link scale) with optional
    standard error bands and rug marks.

    Parameters
    ----------
    model : GAM
        A fitted GAM instance.
    select : int, list[int], or None
        Select specific smooth term(s) to plot (0-indexed). If None,
        plots all smooth terms.
    pages : int
        Reserved for multi-page layout (currently unused).
    rug : bool
        Whether to add rug marks at data covariate values.
    se : bool
        Whether to show standard error bands.
    shade : bool
        If True, use shaded bands for SE. If False, use dashed lines.
    n_grid : int
        Number of evaluation points for 1D smooths (default 200).
    n2 : int
        Grid size per dimension for 2D smooths (default 40, giving
        n2 x n2 grid).
    shade_color : str
        Color for shaded SE bands.
    se_mult : float
        SE multiplier for bands. Default 2.0 matches R's ``plot.gam``.
    line_color : str
        Color for the smooth effect line.
    se_line_color : str
        Color for SE band lines (when shade=False).
    figsize : tuple or None
        Figure size ``(width, height)`` in inches. If None, auto-computed.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : numpy.ndarray
        Array of matplotlib Axes objects, one per plotted smooth.

    Raises
    ------
    RuntimeError
        If the model is not fitted.
    ValueError
        If ``select`` references an invalid smooth index, or if
        ``n_grid``, ``n2``, or ``se_mult`` are invalid.
    """
    import matplotlib.pyplot as plt

    from jaxgam.smooths.by_variable import FactorBySmooth

    model._check_fitted()

    # Validate grid/SE parameters
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    if n2 < 2:
        raise ValueError(f"n2 must be >= 2, got {n2}")
    if se_mult < 0:
        raise ValueError(f"se_mult must be non-negative, got {se_mult}")

    # Gather smooth terms (skip parametric)
    smooth_terms = [t for t in model.coef_map_.terms if t.term_type == "smooth"]

    if len(smooth_terms) == 0:
        raise ValueError("No smooth terms to plot (model is purely parametric).")

    # Resolve select
    if select is not None:
        if isinstance(select, int):
            select = [select]
        for idx in select:
            if idx < 0 or idx >= len(smooth_terms):
                raise ValueError(
                    f"select={idx} is out of range. "
                    f"Model has {len(smooth_terms)} smooth term(s) (0-indexed)."
                )
        plot_terms = [smooth_terms[i] for i in select]
    else:
        plot_terms = smooth_terms

    # Expand factor-by smooths: one panel per level
    panels: list[_PlotPanel] = []
    for term in plot_terms:
        smooth = term.smooth
        if isinstance(smooth, FactorBySmooth):
            panels.extend(_make_factor_by_panels(term))
        else:
            panels.append(_PlotPanel(term=term, factor_level=None))

    n_plots = len(panels)

    # Compute layout
    if n_plots == 1:
        n_rows, n_cols = 1, 1
    else:
        n_cols = min(n_plots, 3)
        n_rows = (n_plots + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5.0 * n_cols, 4.0 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    # Hide unused axes
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Get stored training data for rug/grid range
    if not hasattr(model, "_training_data") or model._training_data is None:
        raise RuntimeError(
            "Training data not stored on model. "
            "Cannot generate rug/grid without covariate ranges."
        )
    training_data = model._training_data

    # Plot each panel
    for i, panel in enumerate(panels):
        ax = axes_flat[i]
        smooth = panel.term.smooth

        # Determine smooth dimensionality
        dim = len(smooth.spec.variables)

        if dim == 1:
            if panel.factor_level is not None:
                _plot_factor_by_level(
                    model=model,
                    term=panel.term,
                    level=panel.factor_level,
                    ax=ax,
                    training_data=training_data,
                    n_grid=n_grid,
                    rug=rug,
                    se=se,
                    shade=shade,
                    shade_color=shade_color,
                    se_mult=se_mult,
                    line_color=line_color,
                    se_line_color=se_line_color,
                )
            else:
                _plot_1d_smooth(
                    model=model,
                    term=panel.term,
                    ax=ax,
                    training_data=training_data,
                    n_grid=n_grid,
                    rug=rug,
                    se=se,
                    shade=shade,
                    shade_color=shade_color,
                    se_mult=se_mult,
                    line_color=line_color,
                    se_line_color=se_line_color,
                )
        elif dim == 2:
            _plot_2d_smooth(
                model=model,
                term=panel.term,
                ax=ax,
                training_data=training_data,
                n2=n2,
                rug=rug,
            )
        else:
            warnings.warn(f"{dim}D smooth plotting not yet supported", stacklevel=2)
            ax.text(
                0.5,
                0.5,
                f"{panel.term.label}\n({dim}D smooth: plot not supported)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Internal panel descriptor
# ---------------------------------------------------------------------------


class _PlotPanel:
    """Describes one plot panel (one axes).

    For factor-by smooths, one panel per level. For other smooths,
    one panel per term.
    """

    __slots__ = ("factor_level", "level_idx", "term")

    def __init__(
        self,
        term: TermBlock,
        factor_level: str | None,
        level_idx: int = 0,
    ) -> None:
        """Initialize a plot panel.

        Parameters
        ----------
        term : TermBlock
            The smooth term block to plot.
        factor_level : str or None
            Factor level for factor-by smooths, or None.
        level_idx : int
            Index of this level within the factor-by smooth (default 0).
        """
        self.term = term
        self.factor_level = factor_level
        self.level_idx = level_idx


def _make_factor_by_panels(term: TermBlock) -> list[_PlotPanel]:
    """Create one panel per factor level for a FactorBySmooth.

    Parameters
    ----------
    term : TermBlock
        The factor-by smooth term.

    Returns
    -------
    list[_PlotPanel]
        One panel per factor level.
    """
    from jaxgam.smooths.by_variable import FactorBySmooth

    assert isinstance(term.smooth, FactorBySmooth)
    smooth = term.smooth
    panels = []
    for idx, level in enumerate(smooth.levels):
        panels.append(_PlotPanel(term=term, factor_level=level, level_idx=idx))
    return panels


# ---------------------------------------------------------------------------
# 1D smooth plotting (shared core + thin wrappers)
# ---------------------------------------------------------------------------


def _plot_1d_core(
    model: GAM,
    term: TermBlock,
    pred_data: dict[str, np.ndarray],
    x_grid: np.ndarray,
    x_rug: np.ndarray,
    ax: matplotlib.axes.Axes,
    rug: bool,
    se: bool,
    shade: bool,
    shade_color: str,
    se_mult: float,
    line_color: str,
    se_line_color: str,
    title: str | None = None,
) -> None:
    """Shared core for 1D smooth and factor-by-level plotting.

    Parameters
    ----------
    model : GAM
        Fitted model.
    term : TermBlock
        The smooth term to plot.
    pred_data : dict[str, np.ndarray]
        Prediction data dict (already built by caller).
    x_grid : np.ndarray
        1D evaluation grid for the covariate.
    x_rug : np.ndarray
        Covariate values for rug marks (may be filtered by level).
    ax : matplotlib.axes.Axes
        Axes to plot on.
    rug : bool
        Show rug marks.
    se : bool
        Show SE bands.
    shade : bool
        Shade SE bands (vs dashed lines).
    shade_color : str
        Shade fill color.
    se_mult : float
        SE multiplier.
    line_color : str
        Line color for smooth effect.
    se_line_color : str
        Line color for SE lines (when shade=False).
    title : str or None
        Optional panel title (used for factor-by levels).
    """
    smooth = term.smooth
    var_name = smooth.spec.variables[0]

    # Get prediction matrix for this smooth
    X_raw = smooth.predict_matrix(pred_data)
    X_s = model.coef_map_.transform_X(X_raw, term.label)

    # Get coefficients and Vp block for this term
    col_start = term.col_start
    col_end = col_start + term.n_coefs
    beta_s = model.coefficients_[col_start:col_end]
    Vp_block = model.Vp_[col_start:col_end, col_start:col_end]

    # Compute partial effect
    fit = X_s @ beta_s

    # Compute SE
    se_fit = None
    if se:
        se_fit = np.sqrt(np.maximum(0.0, np.sum((X_s @ Vp_block) * X_s, axis=1)))

    # Get EDF for label
    edf = _get_smooth_edf(model, term)

    # Plot
    ax.plot(x_grid, fit, color=line_color, linewidth=1.5)

    if se and se_fit is not None:
        upper = fit + se_mult * se_fit
        lower = fit - se_mult * se_fit
        if shade:
            ax.fill_between(x_grid, lower, upper, alpha=0.5, color=shade_color)
        else:
            ax.plot(x_grid, upper, color=se_line_color, linestyle="--", linewidth=0.8)
            ax.plot(x_grid, lower, color=se_line_color, linestyle="--", linewidth=0.8)

    if rug:
        ax.plot(
            x_rug,
            np.full_like(x_rug, ax.get_ylim()[0]),
            "|",
            color="black",
            alpha=0.3,
            markersize=3,
        )

    # Labels
    ax.set_xlabel(var_name)
    ylabel = _make_ylabel(term.label, edf)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


def _plot_1d_smooth(
    model: GAM,
    term: TermBlock,
    ax: matplotlib.axes.Axes,
    training_data: dict[str, np.ndarray],
    n_grid: int,
    rug: bool,
    se: bool,
    shade: bool,
    shade_color: str,
    se_mult: float,
    line_color: str,
    se_line_color: str,
) -> None:
    """Plot a 1D smooth term.

    Parameters
    ----------
    model : GAM
        Fitted model.
    term : TermBlock
        The smooth term to plot.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    training_data : dict
        Training data arrays.
    n_grid : int
        Number of grid points.
    rug : bool
        Show rug marks.
    se : bool
        Show SE bands.
    shade : bool
        Shade SE bands (vs dashed lines).
    shade_color : str
        Shade fill color.
    se_mult : float
        SE multiplier.
    line_color : str
        Line color for smooth effect.
    se_line_color : str
        Line color for SE lines (when shade=False).
    """
    from jaxgam.smooths.by_variable import NumericBySmooth

    smooth = term.smooth
    var_name = smooth.spec.variables[0]

    x_raw = np.asarray(training_data[var_name], dtype=np.float64)
    x_grid = np.linspace(x_raw.min(), x_raw.max(), n_grid)

    pred_data: dict[str, np.ndarray] = {var_name: x_grid}
    if isinstance(smooth, NumericBySmooth):
        pred_data[smooth.by_variable] = np.ones(n_grid)

    _plot_1d_core(
        model=model,
        term=term,
        pred_data=pred_data,
        x_grid=x_grid,
        x_rug=x_raw,
        ax=ax,
        rug=rug,
        se=se,
        shade=shade,
        shade_color=shade_color,
        se_mult=se_mult,
        line_color=line_color,
        se_line_color=se_line_color,
    )


# ---------------------------------------------------------------------------
# Factor-by smooth plotting (one level)
# ---------------------------------------------------------------------------


def _plot_factor_by_level(
    model: GAM,
    term: TermBlock,
    level: str,
    ax: matplotlib.axes.Axes,
    training_data: dict[str, np.ndarray],
    n_grid: int,
    rug: bool,
    se: bool,
    shade: bool,
    shade_color: str,
    se_mult: float,
    line_color: str,
    se_line_color: str,
) -> None:
    """Plot one level of a factor-by smooth.

    The factor-by smooth has a block-diagonal structure. For level ``l``
    the relevant coefficients are selected by the indicator-multiplied
    prediction matrix (zero columns for other levels).

    Parameters
    ----------
    model : GAM
        Fitted model.
    term : TermBlock
        The factor-by smooth term block.
    level : str
        The factor level value.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    training_data : dict
        Training data arrays.
    n_grid : int
        Number of grid points.
    rug : bool
        Show rug marks.
    se : bool
        Show SE bands.
    shade : bool
        Shade SE bands.
    shade_color : str
        Shade fill color.
    se_mult : float
        SE multiplier.
    line_color : str
        Line color.
    se_line_color : str
        SE line color.
    """
    from jaxgam.smooths.by_variable import FactorBySmooth

    assert isinstance(term.smooth, FactorBySmooth)
    smooth = term.smooth
    var_name = smooth.spec.variables[0]

    x_raw = np.asarray(training_data[var_name], dtype=np.float64)
    x_grid = np.linspace(x_raw.min(), x_raw.max(), n_grid)

    by_col = np.full(n_grid, level, dtype=str)
    pred_data: dict[str, np.ndarray] = {
        var_name: x_grid,
        smooth.by_variable: by_col,
    }

    # Filter rug to this factor level
    by_raw = training_data[smooth.by_variable]
    mask = np.asarray(by_raw == level)
    x_rug = x_raw[mask]

    _plot_1d_core(
        model=model,
        term=term,
        pred_data=pred_data,
        x_grid=x_grid,
        x_rug=x_rug,
        ax=ax,
        rug=rug,
        se=se,
        shade=shade,
        shade_color=shade_color,
        se_mult=se_mult,
        line_color=line_color,
        se_line_color=se_line_color,
        title=f"{level}",
    )


# ---------------------------------------------------------------------------
# 2D smooth plotting
# ---------------------------------------------------------------------------


def _plot_2d_smooth(
    model: GAM,
    term: TermBlock,
    ax: matplotlib.axes.Axes,
    training_data: dict[str, np.ndarray],
    n2: int,
    rug: bool,
) -> None:
    """Plot a 2D smooth term as a filled contour.

    Parameters
    ----------
    model : GAM
        Fitted model.
    term : TermBlock
        The smooth term to plot.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    training_data : dict
        Training data arrays.
    n2 : int
        Grid size per dimension.
    rug : bool
        Show data points as scatter.
    """
    from jaxgam.smooths.by_variable import NumericBySmooth

    smooth = term.smooth
    var1 = smooth.spec.variables[0]
    var2 = smooth.spec.variables[1]

    x1_raw = np.asarray(training_data[var1], dtype=np.float64)
    x2_raw = np.asarray(training_data[var2], dtype=np.float64)

    x1_grid = np.linspace(x1_raw.min(), x1_raw.max(), n2)
    x2_grid = np.linspace(x2_raw.min(), x2_raw.max(), n2)

    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    xx, yy = X1.ravel(), X2.ravel()

    # Build prediction data
    pred_data: dict[str, np.ndarray] = {var1: xx, var2: yy}

    # Handle by-variable for tensor products with numeric-by
    if isinstance(smooth, NumericBySmooth):
        pred_data[smooth.by_variable] = np.ones(len(xx))

    # Get prediction matrix for this smooth
    X_raw = smooth.predict_matrix(pred_data)
    X_s = model.coef_map_.transform_X(X_raw, term.label)

    # Compute partial effect
    col_start = term.col_start
    col_end = col_start + term.n_coefs
    beta_s = model.coefficients_[col_start:col_end]
    fit = X_s @ beta_s

    # Reshape to grid: contourf(x, y, Z) expects Z.shape == (len(y), len(x))
    Z = fit.reshape(n2, n2)

    # Get EDF for label
    edf = _get_smooth_edf(model, term)

    # Plot filled contour
    contour = ax.contourf(x1_grid, x2_grid, Z, levels=20, cmap="viridis")
    ax.figure.colorbar(contour, ax=ax, shrink=0.8)

    # Add contour lines
    ax.contour(
        x1_grid, x2_grid, Z, levels=20, colors="black", linewidths=0.3, alpha=0.5
    )

    if rug:
        ax.scatter(x1_raw, x2_raw, s=2, color="black", alpha=0.3, zorder=5)

    # Labels
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(_make_ylabel(term.label, edf))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_smooth_edf(model: GAM, term: TermBlock) -> float:
    """Get the effective degrees of freedom for a smooth term.

    Parameters
    ----------
    model : GAM
        Fitted model.
    term : TermBlock
        The term block.

    Returns
    -------
    float
        Sum of per-coefficient EDF for this term.

    Raises
    ------
    ValueError
        If the term label is not found in the model's smooth info.
    """
    for j, si in enumerate(model.smooth_info_):
        if si.label == term.label:
            return float(model.edf_[j])
    raise ValueError(
        f"Smooth term {term.label!r} not found in model smooth info. "
        f"Available: {[si.label for si in model.smooth_info_]}"
    )


def _make_ylabel(label: str, edf: float) -> str:
    """Construct a y-axis label with EDF, matching R's convention.

    R's label format: ``s(x,3.45)`` where 3.45 is the EDF.

    Parameters
    ----------
    label : str
        Term label, e.g. ``"s(x1)"``.
    edf : float
        Effective degrees of freedom.

    Returns
    -------
    str
        Label with EDF, e.g. ``"s(x1,3.45)"``.
    """
    if ")" in label:
        pos = label.rfind(")")
        return f"{label[:pos]},{edf:.2f})"
    return f"{label} (edf={edf:.2f})"
