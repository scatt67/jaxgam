"""GAMResults: frozen dataclass holding all fitted state.

This module defines:
- ``FittedModel`` protocol for summary/plot interop
- ``GAMResults`` frozen dataclass with prediction, summary, and plot methods
- ``_from_fit()`` classmethod for construction from raw fit output

Design doc reference: docs/refactor_gam_api/design.md §3.4, §4.1, §7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from jaxgam.jax_utils import to_numpy
from jaxgam.post_estimation import compute_post_estimation

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd

    from jaxgam.families.base import ExponentialFamily
    from jaxgam.fitting.data import FittingData
    from jaxgam.fitting.newton import NewtonResult
    from jaxgam.formula.design import ModelSetup, SmoothInfo
    from jaxgam.formula.terms import FormulaSpec
    from jaxgam.smooths.constraints import CoefficientMap
    from jaxgam.summary.summary import GAMSummary


# ---------------------------------------------------------------------------
# FittedModel protocol — structural typing for summary/plot interop
# ---------------------------------------------------------------------------


@runtime_checkable
class FittedModel(Protocol):
    """Protocol capturing what summary() and plot() need.

    Both ``GAM`` (with forwarding) and ``GAMResults`` satisfy this
    protocol, enabling summary/plot to work with either during the
    migration period.

    Design doc reference: docs/refactor_gam_api/design.md §2 goal #6,
    docs/refactor_gam_api/implementation_plan.md Step 2.2.
    """

    coefficients: np.ndarray
    Vp: np.ndarray
    scale: float
    edf: np.ndarray
    edf1: np.ndarray
    edf_total: float
    family: ExponentialFamily
    smooth_info: tuple[SmoothInfo, ...]
    term_names: tuple[str, ...]
    coef_map: CoefficientMap
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    n: int
    fitted_values: np.ndarray
    linear_predictor: np.ndarray
    deviance: float
    null_deviance: float
    score: float
    formula: str
    method: str
    training_data: dict[str, np.ndarray]
    setup: ModelSetup


# ---------------------------------------------------------------------------
# GAMResults frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GAMResults:
    """Results from a fitted GAM.

    All attributes are read-only (frozen dataclass). This object is the
    primary interface for post-estimation: prediction, inference, and
    visualization.

    Design doc reference: docs/refactor_gam_api/design.md §3.4
    """

    # -- Core estimates -----------------------------------------------------
    coefficients: np.ndarray  # (p,) fitted coefficients
    fitted_values: np.ndarray  # (n,) response-scale fitted values
    linear_predictor: np.ndarray  # (n,) link-scale linear predictor

    # -- Covariance & scale -------------------------------------------------
    Vp: np.ndarray  # (p, p) Bayesian posterior covariance
    scale: float  # dispersion parameter (phi)

    # -- Degrees of freedom -------------------------------------------------
    edf: np.ndarray  # (n_smooths,) per-smooth EDF
    edf1: np.ndarray  # (n_smooths,) alternative EDF
    edf_total: float  # total model EDF

    # -- Deviance -----------------------------------------------------------
    deviance: float
    null_deviance: float

    # -- Smoothing parameters -----------------------------------------------
    smoothing_params: np.ndarray  # (n_penalties,) estimated lambda

    # -- Convergence --------------------------------------------------------
    converged: bool
    n_iter: int
    score: float  # REML/ML value at convergence

    # -- Model structure (Phase 1 artifacts) --------------------------------
    family: ExponentialFamily
    setup: ModelSetup  # frozen Phase 1 output
    coef_map: CoefficientMap  # Phase 1→3 coefficient mapping
    smooth_info: tuple[SmoothInfo, ...]
    term_names: tuple[str, ...]

    # -- Data references ----------------------------------------------------
    X: np.ndarray  # (n, p) design matrix
    y: np.ndarray  # (n,) response
    weights: np.ndarray  # (n,) prior weights
    offset: np.ndarray | None

    # -- Metadata -----------------------------------------------------------
    n: int
    execution_path: str
    lambda_strategy: str
    formula: str  # echoed from specification
    method: str  # "REML" or "ML" (echoed from specification)
    training_data: dict[str, np.ndarray]  # for plotting

    # ------------------------------------------------------------------
    # Factory classmethod
    # ------------------------------------------------------------------

    @classmethod
    def _from_fit(
        cls,
        result: NewtonResult,
        setup: ModelSetup,
        spec: FormulaSpec,
        data: pd.DataFrame | dict,
        family: ExponentialFamily,
        fd: FittingData,
        lambda_strategy: str,
        formula: str,
        method: str,
    ) -> GAMResults:
        """Construct GAMResults from raw fit output.

        Calls ``compute_post_estimation()`` for derived quantities,
        extracts training data, and assembles all fields.

        Design doc reference: docs/refactor_gam_api/design.md §3.4
        decision #4.

        Parameters
        ----------
        result : NewtonResult
            Raw output from Newton optimization or fixed-sp PIRLS.
        setup : ModelSetup
            Phase 1 model setup.
        spec : FormulaSpec
            Parsed formula (needed for training data extraction).
        data : DataFrame or dict
            Training data (needed for training data extraction).
        family : ExponentialFamily
            Distribution family.
        fd : FittingData
            Phase 1→2 boundary data.
        lambda_strategy : str
            How smoothing parameters were determined.
        formula : str
            Original formula string from the GAM specification.
        method : str
            Smoothing parameter estimation method ("REML" or "ML").
        """
        pr = result.pirls_result

        # Compute derived quantities via post_estimation module
        post = compute_post_estimation(result, setup, family, fd)

        # Phase 2→3: transfer remaining arrays to NumPy
        mu = to_numpy(pr.mu)
        eta = to_numpy(pr.eta)
        deviance = float(to_numpy(pr.deviance))
        smoothing_params = to_numpy(result.smoothing_params)

        # Extract training data for plotting
        training_data = _extract_training_data(spec, data)

        return cls(
            coefficients=post.coefficients,
            fitted_values=mu,
            linear_predictor=eta,
            Vp=post.Vp,
            scale=post.scale,
            edf=post.edf,
            edf1=post.edf1,
            edf_total=post.edf_total,
            deviance=deviance,
            null_deviance=post.null_deviance,
            smoothing_params=smoothing_params,
            converged=result.converged,
            n_iter=result.n_iter,
            score=float(to_numpy(result.score)),
            family=family,
            setup=setup,
            coef_map=setup.coef_map,
            smooth_info=setup.smooth_info,
            term_names=setup.term_names,
            X=setup.X,
            y=setup.y,
            weights=setup.weights,
            offset=setup.offset,
            n=setup.n_obs,
            execution_path="jax",
            lambda_strategy=lambda_strategy,
            formula=formula,
            method=method,
            training_data=training_data,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        newdata: pd.DataFrame | dict | None = None,
        pred_type: str = "response",
        se_fit: bool = False,
        offset: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict from a fitted GAM.

        Parameters
        ----------
        newdata : pandas.DataFrame or dict, optional
            New data for prediction. If None, uses the training data.
        pred_type : str
            Type of prediction: ``'response'`` or ``'link'``.
        se_fit : bool
            Whether to return standard errors.
        offset : array-like, optional
            Offset for new data predictions.

        Returns
        -------
        numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
            Predictions, or ``(predictions, standard_errors)``
            if ``se_fit=True``.
        """
        if pred_type not in ("response", "link"):
            raise ValueError(
                f"pred_type must be 'response' or 'link', "
                f"got {pred_type!r}"
            )

        if newdata is None:
            # Self-prediction: use stored linear predictor
            eta = self.linear_predictor.copy()
            X_p = self.X if se_fit else None
        else:
            X_p = self.setup.build_predict_matrix(newdata)
            eta = X_p @ self.coefficients
            if offset is not None:
                eta = eta + np.asarray(
                    offset, dtype=np.float64
                ).ravel()

        pred = (
            self.family.link.linkinv(eta)
            if pred_type == "response"
            else eta
        )

        if se_fit:
            if X_p is None:
                X_p = self.X
            # se = sqrt(rowSums((X_p @ Vp) * X_p))
            XVp = X_p @ self.Vp
            se = np.sqrt(np.sum(XVp * X_p, axis=1))
            return pred, se

        return pred

    def predict_matrix(
        self, newdata: pd.DataFrame | dict
    ) -> np.ndarray:
        """Build constrained prediction matrix for new data.

        Equivalent to R's ``predict.gam(type="lpmatrix")``.

        Parameters
        ----------
        newdata : DataFrame or dict
            New data for prediction.

        Returns
        -------
        np.ndarray, shape ``(n_new, total_coefs)``
            Constrained prediction matrix.
        """
        return self.setup.build_predict_matrix(newdata)

    # ------------------------------------------------------------------
    # Summary and plot delegation
    # ------------------------------------------------------------------

    def summary(self) -> GAMSummary:
        """Print and return summary of a fitted GAM.

        Computes parametric coefficient significance (z/t tests),
        smooth term significance (Wood 2013 testStat), and model-level
        statistics (R-squared, deviance explained, scale estimate).

        Returns
        -------
        GAMSummary
            Summary object with parametric and smooth term tables.
            The summary is also printed to stdout.
        """
        from jaxgam.summary.summary import summary as _summary

        s = _summary(self)
        print(s)  # noqa: T201
        return s

    def plot(
        self,
        select: int | list | None = None,
        pages: int = 0,
        rug: bool = True,
        se: bool = True,
        shade: bool = True,
        **kwargs,
    ) -> tuple[matplotlib.figure.Figure, np.ndarray]:
        """Plot smooth components of a fitted GAM.

        Equivalent to R's ``plot.gam()``.

        Parameters
        ----------
        select : int, list, or None
            Select specific smooth term(s) to plot (0-indexed).
        pages : int
            Number of pages. 0 means automatic layout.
        rug : bool
            Show rug marks at data covariate values.
        se : bool
            Show standard error bands.
        shade : bool
            If True, use shaded SE bands; if False, use dashed lines.
        **kwargs
            Additional arguments passed to ``plot_gam()``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        axes : numpy.ndarray
            Array of Axes objects.
        """
        from jaxgam.plot import plot_gam

        return plot_gam(
            self,
            select=select,
            pages=pages,
            rug=rug,
            se=se,
            shade=shade,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        family_name = (
            self.family.family_name
            if hasattr(self.family, "family_name")
            else type(self.family).__name__
        )
        dev_explained = (
            1.0 - self.deviance / self.null_deviance
            if self.null_deviance > 0
            else float("nan")
        )
        return (
            f"GAMResults(\n"
            f"  formula='{self.formula}',\n"
            f"  family='{family_name}',\n"
            f"  converged={self.converged},\n"
            f"  deviance_explained={dev_explained:.4f},\n"
            f"  n={self.n}, edf_total={self.edf_total:.2f}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_training_data(
    spec: FormulaSpec,
    data: pd.DataFrame | dict,
) -> dict[str, np.ndarray]:
    """Extract raw training covariate data for plotting.

    Stores all variables referenced in smooth terms (covariates and
    by-variables) so that ``plot()`` can construct evaluation grids
    and rug plots without re-accessing the original data.

    Parameters
    ----------
    spec : FormulaSpec
        Parsed formula specification.
    data : DataFrame or dict
        Training data.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from variable name to raw training data array.
    """
    from jaxgam.smooths.by_variable import is_factor

    training: dict[str, np.ndarray] = {}

    # Collect all variable names from smooth terms
    var_names: set[str] = set()
    for st in spec.smooth_terms:
        for v in st.variables:
            var_names.add(v)
        if st.by is not None:
            var_names.add(st.by)

    for name in var_names:
        col = data[name]
        # Preserve dtype: factors stay as-is, numerics become float64
        if is_factor(col):
            training[name] = np.asarray(col)
        else:
            training[name] = np.asarray(col, dtype=np.float64).ravel()

    return training
