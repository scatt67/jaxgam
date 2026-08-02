"""Full and lean frozen result types for fitted GAMs.

This module defines:
- ``GAMResults`` with prediction, summary, and plot methods
- ``GAMInferenceResult`` with training-data-free prediction state
- ``_from_fit()`` classmethod for construction from raw fit output
- Post-estimation helpers (EDF, covariance, null deviance)

Design doc reference: docs/refactor_gam_api/design.md §3.4, §4.1, §7
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.linalg as sla

from jaxgam.inference._core import finish_prediction, predict_core
from jaxgam.inference.predictor import GAMPredictor
from jaxgam.jax_utils import to_numpy

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


def _offset_was_nonzero(setup: ModelSetup) -> bool:
    """Whether fitting used an external offset that prediction cannot recover."""
    return setup.offset is not None and not np.allclose(setup.offset, 0.0)


# ---------------------------------------------------------------------------
# Shared fit diagnostics and concrete result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FitDiagnostics:
    """Cheap fit diagnostics retained by both result materializations."""

    edf: np.ndarray
    edf1: np.ndarray
    edf_total: float
    deviance: float
    null_deviance: float
    score: float
    scale: float
    theta: float | None
    smoothing_params: np.ndarray
    converged: bool
    n_iter: int
    convergence_info: str
    method: str
    lambda_strategy: str
    execution_path: str
    n: int


@dataclass(frozen=True)
class GAMInferenceResult(_FitDiagnostics):
    """Lean fitted result retaining prediction state but no training arrays.

    This object is directly usable for prediction; calling ``to_predictor()``
    is optional. The method returns the predictor already composed during fit,
    without copying state or reducing retained memory further. Use it only when
    a downstream consumer should receive the narrower prediction-only surface.
    """

    _predictor: GAMPredictor

    @property
    def coefficients(self) -> np.ndarray:
        """Read-only fitted coefficients owned by the predictor."""
        return self._predictor.coefficients

    @property
    def Vp(self) -> np.ndarray:
        """Read-only Bayesian posterior covariance owned by the predictor."""
        return self._predictor.Vp

    @property
    def family(self) -> ExponentialFamily:
        """Post-fit family snapshot, including the fitted link and theta."""
        return self._predictor.family

    @property
    def formula(self) -> str:
        """Original model formula."""
        return self._predictor.formula

    @property
    def smooth_info(self) -> tuple[SmoothInfo, ...]:
        """Per-smooth labels and coefficient ranges for interpreting EDF."""
        return self._predictor._predict_spec.smooth_info

    @property
    def term_names(self) -> tuple[str, ...]:
        """Model-matrix column names retained in prediction metadata."""
        return self._predictor._predict_spec.term_names

    def predict(
        self,
        newdata: pd.DataFrame | dict,
        pred_type: str = "response",
        se_fit: bool = False,
        offset: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict from lean state; unlike ``GAMResults``, new data is required."""
        return self._predictor._predict(
            newdata,
            pred_type=pred_type,
            se_fit=se_fit,
            offset=offset,
            warning_stacklevel=4,
        )

    def predict_matrix(self, newdata: pd.DataFrame | dict) -> np.ndarray:
        """Build the constrained prediction matrix for new data."""
        return self._predictor.predict_matrix(newdata)

    def to_predictor(self) -> GAMPredictor:
        """Return the existing prediction-only core without copying.

        This is an optional interface handoff, not another memory optimization.
        The inference result itself already delegates prediction to this object.
        """
        return self._predictor

    def __repr__(self) -> str:
        return (
            "GAMInferenceResult(\n"
            f"  formula={self.formula!r},\n"
            f"  family={self.family.family_name!r},\n"
            f"  converged={self.converged}, n={self.n}, "
            f"edf_total={self.edf_total:.2f}\n"
            "  Fit with result='full' for summary() and plot().\n"
            ")"
        )


@dataclass(frozen=True)
class GAMResults(_FitDiagnostics):
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

    # -- Covariance ---------------------------------------------------------
    Vp: np.ndarray  # (p, p) Bayesian posterior covariance

    # -- Model structure (Phase 1 artifacts) --------------------------------
    family: ExponentialFamily
    setup: ModelSetup  # frozen Phase 1 output

    # -- Metadata -----------------------------------------------------------
    formula: str  # echoed from specification
    training_data: dict[str, np.ndarray]  # for plotting

    # ------------------------------------------------------------------
    # Factory classmethod
    # ------------------------------------------------------------------

    @classmethod
    def _from_fit(
        cls,
        fit_result: NewtonResult,
        setup: ModelSetup,
        spec: FormulaSpec,
        data: pd.DataFrame | dict,
        family: ExponentialFamily,
        fd: FittingData,
        lambda_strategy: str,
        formula: str,
        method: str,
        result_mode: Literal["full", "inference"],
    ) -> GAMResults | GAMInferenceResult:
        """Construct the requested result materialization from raw fit output.

        Computes derived quantities (covariance, EDF, null deviance),
        extracts training data, and assembles all fields.

        Design doc reference: docs/refactor_gam_api/design.md §3.4
        decision #4.

        Parameters
        ----------
        fit_result : NewtonResult
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
            Smoothing parameter estimation method ("REML"; only REML in v1.0).
        result_mode : {"full", "inference"}
            Whether to retain full training-backed state or lean prediction
            state only.
        """
        pr = fit_result.pirls_result

        # Snapshot after Newton has synchronized any fitted family parameters
        # (notably NB theta) into the fitting family instance.
        family_snapshot = copy.deepcopy(family)

        # Phase 2→3: transfer to NumPy
        coefficients = to_numpy(pr.coefficients)
        scale = float(to_numpy(fit_result.scale))
        edf_total = float(to_numpy(fit_result.edf))

        # Use Fisher-weighted quantities for EDF and Bayesian covariance.
        # For standard families Fisher = Newton; for extended families (NB)
        # these are recomputed post-convergence with expected weights
        # (R's gdi2, gdi.c:2262-2294, gam.fit4.r:564).
        L = to_numpy(pr.L_fisher)
        XtWX = to_numpy(pr.XtWX_fisher)

        # Compute H^{-1} via Cholesky solve (matches R's chol2inv).
        # O(p^3) but p is typically small (< 200 for GAMs).
        p = L.shape[0]
        Z = sla.solve_triangular(L, np.eye(p), lower=True)
        H_inv = Z.T @ Z

        # Per-smooth EDF via hat matrix F = H^{-1} @ XtWX
        # (invariant under repara -- cyclic trace with block-diagonal D)
        F = H_inv @ XtWX
        per_smooth_edf = _compute_per_smooth_edf(F, setup.smooth_info)
        # edf1 = 2*edf - trace(F^2): alternative EDF for significance testing
        # (R's gam.fit3.post.proc, mgcv.r line 966)
        per_smooth_edf1 = _compute_per_smooth_edf1(F, setup.smooth_info)

        # Back-transform from Sl.setup reparameterized space
        if fd.repara_D is not None:
            D = to_numpy(fd.repara_D)
            coefficients = D @ coefficients
            H_inv = D @ H_inv @ D.T

        # Bayesian covariance
        phi = 1.0 if family_snapshot.scale_known else scale
        Vp = phi * H_inv

        # Null deviance
        null_deviance = _compute_null_deviance(
            setup.y, setup.weights, family_snapshot, setup.offset
        )

        # Phase 2→3: transfer remaining arrays to NumPy
        mu = to_numpy(pr.mu)
        eta = to_numpy(pr.eta)
        deviance = float(to_numpy(pr.deviance))
        smoothing_params = to_numpy(fit_result.smoothing_params)

        diagnostics = {
            "edf": per_smooth_edf,
            "edf1": per_smooth_edf1,
            "edf_total": edf_total,
            "deviance": deviance,
            "null_deviance": null_deviance,
            "score": float(to_numpy(fit_result.score)),
            "scale": scale,
            "theta": fit_result.theta,
            "smoothing_params": smoothing_params,
            "converged": fit_result.converged,
            "n_iter": fit_result.n_iter,
            "convergence_info": fit_result.convergence_info,
            "method": method,
            "lambda_strategy": lambda_strategy,
            "execution_path": "jax",
            "n": setup.n_obs,
        }

        if result_mode == "inference":
            predictor = GAMPredictor(
                coefficients=coefficients,
                Vp=Vp,
                family=family_snapshot,
                formula=formula,
                offset_was_nonzero=_offset_was_nonzero(setup),
                _predict_spec=setup._lazy_predict_spec(),
            )
            return GAMInferenceResult(_predictor=predictor, **diagnostics)

        # Plotting data is extracted only for the full diagnostic surface.
        training_data = _extract_training_data(spec, data)
        return cls(
            coefficients=coefficients,
            fitted_values=mu,
            linear_predictor=eta,
            Vp=Vp,
            family=family_snapshot,
            setup=setup,
            formula=formula,
            training_data=training_data,
            **diagnostics,
        )

    # ------------------------------------------------------------------
    # Phase 1 aliases
    # ------------------------------------------------------------------

    @property
    def X(self) -> np.ndarray:
        """Constrained training design matrix owned by ``setup``."""
        return self.setup.X

    @property
    def y(self) -> np.ndarray:
        """Training response owned by ``setup``."""
        return self.setup.y

    @property
    def weights(self) -> np.ndarray:
        """Prior weights owned by ``setup``."""
        return self.setup.weights

    @property
    def offset(self) -> np.ndarray | None:
        """Training offset owned by ``setup``."""
        return self.setup.offset

    @property
    def coef_map(self) -> CoefficientMap:
        """Phase 1→3 coefficient mapping owned by ``setup``."""
        return self.setup.coef_map

    @property
    def smooth_info(self) -> tuple[SmoothInfo, ...]:
        """Per-smooth metadata owned by ``setup``."""
        return self.setup.smooth_info

    @property
    def term_names(self) -> tuple[str, ...]:
        """Model-matrix column names owned by ``setup``."""
        return self.setup.term_names

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
                f"pred_type must be 'response' or 'link', got {pred_type!r}"
            )

        if newdata is None:
            # Self-prediction: use stored linear predictor
            eta = self.linear_predictor.copy()
            return finish_prediction(
                eta,
                self.X,
                self.family.link,
                self.Vp,
                pred_type=pred_type,
                se_fit=se_fit,
            )

        return predict_core(
            self.setup._lazy_predict_spec(),
            self.coefficients,
            self.Vp,
            self.family.link,
            newdata,
            pred_type=pred_type,
            se_fit=se_fit,
            offset=offset,
            offset_was_nonzero=_offset_was_nonzero(self.setup),
        )

    def predict_matrix(self, newdata: pd.DataFrame | dict) -> np.ndarray:
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

    def to_predictor(self) -> GAMPredictor:
        """Build an independent, prediction-only core on demand.

        This does not mutate or slim the full result. Discard the full result if
        its training-backed state is no longer needed after the handoff.
        """
        return GAMPredictor(
            coefficients=self.coefficients,
            Vp=self.Vp,
            family=self.family,
            formula=self.formula,
            offset_was_nonzero=_offset_was_nonzero(self.setup),
            _predict_spec=self.setup._lazy_predict_spec(),
        )

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
        family_name = self.family.family_name
        dev_explained = (
            1.0 - self.deviance / self.null_deviance
            if self.null_deviance > 0
            else float("nan")
        )
        theta_line = ""
        if self.theta is not None:
            theta_line = f"  theta={self.theta:.4f},\n"
        # Surface the optimizer's terminal state when the fit did not
        # converge, so non-convergence is visible without inspecting fields.
        conv_line = f"  converged={self.converged},\n"
        if not self.converged:
            conv_line = f"  converged={self.converged} ({self.convergence_info}),\n"
        return (
            f"GAMResults(\n"
            f"  formula='{self.formula}',\n"
            f"  family='{family_name}',\n"
            f"{theta_line}"
            f"{conv_line}"
            f"  deviance_explained={dev_explained:.4f},\n"
            f"  n={self.n}, edf_total={self.edf_total:.2f}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Post-estimation helpers
# ---------------------------------------------------------------------------


def _compute_per_smooth_edf(
    F: np.ndarray,
    smooth_info: tuple[SmoothInfo, ...],
) -> np.ndarray:
    """Per-smooth effective degrees of freedom.

    Parameters
    ----------
    F : np.ndarray, shape (p, p)
        Hat-like matrix: ``H^{-1} @ XtWX``.
    smooth_info : tuple[SmoothInfo, ...]
        Per-smooth metadata with column ranges.

    Returns
    -------
    np.ndarray, shape (n_smooths,)
        Per-smooth EDF.
    """
    n_smooths = len(smooth_info)
    edf = np.empty(n_smooths, dtype=np.float64)
    for j, si in enumerate(smooth_info):
        cols = slice(si.first_coef, si.last_coef)
        edf[j] = np.trace(F[cols, cols])
    return edf


def _compute_per_smooth_edf1(
    F: np.ndarray,
    smooth_info: tuple[SmoothInfo, ...],
) -> np.ndarray:
    """Alternative per-smooth EDF for significance testing.

    Computes ``edf1 = 2*edf - edf2`` where ``edf2 = trace(F^2)`` per
    smooth block. This is R's ``edf1`` (mgcv gam.fit3.post.proc line 966):
    ``edf1 <- 2*edf - rowSums(t(F)*F)``.

    The per-smooth version sums per-coefficient ``edf1`` values over
    each smooth's column range, matching R's
    ``sum(object$edf1[start:stop])``.

    Parameters
    ----------
    F : np.ndarray, shape (p, p)
        Hat-like matrix: ``H^{-1} @ XtWX``.
    smooth_info : tuple[SmoothInfo, ...]
        Per-smooth metadata with column ranges.

    Returns
    -------
    np.ndarray, shape (n_smooths,)
        Alternative EDF (``edf1``) per smooth, for use as ``Ref.df``
        in Wood (2013) significance tests.
    """
    # Per-coefficient: edf_i = F[i,i], edf2_i = sum(F[i,:] * F[:,i])
    edf_per_coef = np.diag(F)
    edf2_per_coef = np.sum(F.T * F, axis=0)  # rowSums(t(F)*F)
    edf1_per_coef = 2.0 * edf_per_coef - edf2_per_coef

    n_smooths = len(smooth_info)
    edf1 = np.empty(n_smooths, dtype=np.float64)
    for j, si in enumerate(smooth_info):
        cols = slice(si.first_coef, si.last_coef)
        edf1[j] = np.sum(edf1_per_coef[cols])
    return edf1


def _compute_null_deviance(
    y: np.ndarray,
    wt: np.ndarray,
    family: ExponentialFamily,
    offset: np.ndarray | None = None,
) -> float:
    """Null model deviance.

    Without an offset, the null model prediction is the weighted mean of
    ``y`` (the intercept-only MLE for canonical-mean families). With an
    offset, the null model is intercept-only *including* the offset,
    ``mu_i = linkinv(beta0 + offset_i)`` with ``beta0`` fit by IRLS. This
    matches R/mgcv and ``glm()``'s ``null.deviance``, which is offset-aware
    (verified: a Poisson+offset null deviance equals the intercept+offset
    fit, not the offset-free weighted mean).

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Response values.
    wt : np.ndarray, shape (n,)
        Prior weights.
    family : ExponentialFamily
        Family with ``dev_resids()`` method.
    offset : np.ndarray, shape (n,), optional
        Offset vector. If ``None`` or all-zero, the weighted-mean null is
        used.

    Returns
    -------
    float
        Null model deviance.
    """
    if offset is None or np.allclose(offset, 0.0):
        mu_null = np.sum(wt * y) / np.sum(wt)
        mu_null_arr = np.full_like(y, mu_null)
        return float(family.dev_resids(y, mu_null_arr, wt))

    offset = np.asarray(offset, dtype=np.float64).ravel()
    beta0 = _fit_null_intercept(y, wt, family, offset)
    mu_null_arr = np.asarray(family.link.inverse(beta0 + offset))
    return float(family.dev_resids(y, mu_null_arr, wt))


def _fit_null_intercept(
    y: np.ndarray,
    wt: np.ndarray,
    family: ExponentialFamily,
    offset: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> float:
    """IRLS fit of the intercept-only model ``mu = linkinv(beta0 + offset)``.

    Standard GLM IRLS for a single intercept column: at each step the
    working response ``z = beta0 + (y - mu) * g'(mu)`` is regressed on the
    constant with working weights ``W = wt / (V(mu) * g'(mu)^2)``, giving
    ``beta0 <- sum(W z) / sum(W)``. Used only for offset-aware null deviance.
    """
    mu_bar = np.sum(wt * y) / np.sum(wt)
    beta0 = float(family.link.link(np.asarray(mu_bar)))
    for _ in range(max_iter):
        eta = beta0 + offset
        mu = np.asarray(family.link.inverse(eta))
        g_prime = np.asarray(family.link.derivative(mu))
        var = np.asarray(family.variance(mu))
        weight = wt / np.maximum(var * g_prime**2, 1e-300)
        z = beta0 + (y - mu) * g_prime  # working response minus offset
        beta0_new = float(np.sum(weight * z) / np.sum(weight))
        if abs(beta0_new - beta0) <= tol * (abs(beta0_new) + tol):
            return beta0_new
        beta0 = beta0_new
    return beta0


# ---------------------------------------------------------------------------
# Training data extraction
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
    from jaxgam.smooths.utils import is_factor

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
