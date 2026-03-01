"""Top-level fitting orchestration for pymgcv.

Provides the ``GAM`` class (sklearn-style API) that wires together:
- Phase 1: ``parse_formula()`` → ``ModelSetup.build()``
- Phase 2: ``FittingData.from_setup()`` → ``newton_optimize()`` / ``pirls_loop()``
- Phase 3: Post-estimation → fitted attributes on ``GAM`` instance

Design doc reference: docs/design.md Section 10.1, 10.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as sla

from pymgcv.families.base import ExponentialFamily
from pymgcv.families.registry import get_family
from pymgcv.fitting.data import FittingData
from pymgcv.fitting.initialization import initialize_beta
from pymgcv.fitting.newton import NewtonResult, newton_optimize
from pymgcv.fitting.pirls import pirls_loop
from pymgcv.fitting.reml import estimate_edf, estimate_scale
from pymgcv.formula.design import ModelSetup, SmoothInfo
from pymgcv.formula.parser import parse_formula
from pymgcv.formula.terms import FormulaSpec, ParametricTerm
from pymgcv.jax_utils import to_numpy
from pymgcv.smooths.by_variable import get_factor_levels, is_factor

if TYPE_CHECKING:
    import jax
    import matplotlib.figure
    import pandas as pd

    from pymgcv.summary.summary import GAMSummary


# ---------------------------------------------------------------------------
# GAM class
# ---------------------------------------------------------------------------


class GAM:
    """Generalized Additive Model (sklearn-style API).

    Parameters
    ----------
    formula : str
        Model formula in R-style Wilkinson notation, e.g. ``"y ~ s(x)"``.
    family : str or ExponentialFamily
        Distribution family. One of ``'gaussian'``, ``'binomial'``,
        ``'poisson'``, ``'gamma'``, or an ``ExponentialFamily`` instance.
    method : str
        Smoothing parameter estimation method: ``'REML'`` or ``'ML'``.
    sp : np.ndarray or list, optional
        Fixed smoothing parameters. If provided, skips Newton optimization.
    device : str, optional
        Target device: ``'cpu'``, ``'gpu'``, or ``None`` (auto-detect).
        GPU requires ``jax[cuda12]`` (NVIDIA) or ``jax-metal`` (Apple).
    **kwargs
        Additional arguments. Supported scope guards:
        ``backend``, ``optimizer``, ``select``, ``gamma``, ``knots``.

    Attributes (set after ``fit()``)
    --------------------------------
    coefficients_ : np.ndarray
        Fitted coefficient vector.
    fitted_values_ : np.ndarray
        Fitted values (response scale).
    linear_predictor_ : np.ndarray
        Linear predictor (link scale).
    family_ : ExponentialFamily
        Fitted family object.
    Vp_ : np.ndarray
        Bayesian posterior covariance of coefficients.
    scale_ : float
        Estimated or fixed scale parameter (phi).
    edf_ : np.ndarray
        Per-smooth effective degrees of freedom.
    edf_total_ : float
        Total effective degrees of freedom.
    smoothing_params_ : np.ndarray
        Estimated smoothing parameters (original scale).
    deviance_ : float
        Model deviance.
    null_deviance_ : float
        Null model deviance.
    converged_ : bool
        Whether the outer Newton loop converged.
    n_iter_ : int
        Number of outer Newton iterations.

    Examples
    --------
    >>> model = GAM("y ~ s(x)", family="gaussian").fit(data)
    >>> model.coefficients_
    array([...])
    """

    def __init__(
        self,
        formula: str,
        family: str | ExponentialFamily = "gaussian",
        method: str = "REML",
        sp: np.ndarray | list | None = None,
        **kwargs,
    ) -> None:
        _check_scope_guards(method, kwargs)
        self.formula = formula
        self.family = family
        self.method = method.upper()
        self.sp = sp
        self.device = kwargs.get("device")
        self._fitted = False

    def fit(
        self,
        data: pd.DataFrame | dict,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ) -> GAM:
        """Fit the GAM to data.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Data frame containing the variables in the formula.
        weights : np.ndarray, optional
            Prior weights, shape ``(n,)``.
        offset : np.ndarray, optional
            Offset vector, shape ``(n,)``.

        Returns
        -------
        GAM
            Self, for method chaining.
        """
        family_obj = get_family(self.family)

        # Phase 1: parse + build model setup
        spec = parse_formula(self.formula)
        setup = ModelSetup.build(spec, data, weights, offset)

        # Phase 1→2: transfer to JAX device
        jax_device = _resolve_device(self.device)
        fd = FittingData.from_setup(setup, family_obj, device=jax_device)

        # Phase 2: fit
        if self.sp is not None:
            result = _fit_fixed_sp(fd, self.sp)
            lambda_strategy = "fixed"
        else:
            result = newton_optimize(fd, self.method)
            lambda_strategy = f"newton_{self.method.lower()}"

        # Phase 2→3: post-estimation
        self._store_results(result, setup, spec, data, family_obj, fd, lambda_strategy)
        self._fitted = True
        return self

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
            Predictions, or ``(predictions, standard_errors)`` if ``se_fit=True``.
        """
        self._check_fitted()

        if pred_type not in ("response", "link"):
            raise ValueError(
                f"pred_type must be 'response' or 'link', got {pred_type!r}"
            )

        if newdata is None:
            # Self-prediction: use stored linear predictor
            eta = self.linear_predictor_.copy()
            X_p = self.X_ if se_fit else None
        else:
            X_p = self._build_predict_matrix(newdata)
            eta = X_p @ self.coefficients_
            if offset is not None:
                eta = eta + np.asarray(offset, dtype=np.float64).ravel()

        pred = self.family_.link.linkinv(eta) if pred_type == "response" else eta

        if se_fit:
            if X_p is None:
                X_p = self.X_
            # se = sqrt(rowSums((X_p @ Vp) * X_p))
            XVp = X_p @ self.Vp_
            se = np.sqrt(np.sum(XVp * X_p, axis=1))
            return pred, se

        return pred

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
        self._check_fitted()
        return self._build_predict_matrix(newdata)

    def _build_predict_matrix(self, newdata: pd.DataFrame | dict) -> np.ndarray:
        """Build the full constrained prediction matrix for new data.

        Parameters
        ----------
        newdata : DataFrame or dict
            New data containing all required variables.

        Returns
        -------
        np.ndarray, shape ``(n_new, total_coefs)``
        """
        data_dict = ModelSetup._to_dict(newdata)

        if not data_dict:
            raise ValueError("newdata is empty — no variables found.")

        # Determine n_obs from first available variable
        first_key = next(iter(data_dict))
        n_new = len(data_dict[first_key])

        # Build parametric columns
        X_parametric = _build_parametric_predict(
            self.formula_spec_.parametric_terms,
            newdata,
            self.formula_spec_.has_intercept,
            n_new,
            self._factor_info_,
        )

        # Build smooth columns
        blocks: list[np.ndarray] = [X_parametric]
        coef_map = self.coef_map_

        for term in coef_map.terms:
            if term.term_type == "parametric":
                continue
            # Get raw prediction matrix from the smooth
            X_raw = term.smooth.predict_matrix(data_dict)
            # Apply constraint transform (centering + gam_side)
            X_c = coef_map.transform_X(X_raw, term.label)
            blocks.append(X_c)

        X_p = np.column_stack(blocks) if len(blocks) > 1 else blocks[0]
        if X_p.shape[1] != coef_map.total_coefs:
            raise RuntimeError(
                f"Prediction matrix has {X_p.shape[1]} columns but model "
                f"expects {coef_map.total_coefs}."
            )
        return X_p

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
        from pymgcv.summary.summary import summary as _summary

        self._check_fitted()
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

        Equivalent to R's ``plot.gam()``. Produces one panel per smooth
        term (or per factor level for factor-by smooths) showing the
        partial effect on the link scale, with optional SE bands and
        rug marks.

        Parameters
        ----------
        select : int, list, or None
            Select specific smooth term(s) to plot (0-indexed). If None,
            plots all smooth terms.
        pages : int
            Number of pages. 0 means automatic layout.
        rug : bool
            Show rug marks at data covariate values.
        se : bool
            Show standard error bands.
        shade : bool
            If True, use shaded SE bands; if False, use dashed lines.
        **kwargs
            Additional arguments passed to ``plot_gam()``. See
            ``pymgcv.plot.plot_gam.plot_gam`` for full parameter list.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        axes : numpy.ndarray
            Array of Axes objects.
        """
        self._check_fitted()
        from pymgcv.plot import plot_gam

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
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self._fitted:
            raise RuntimeError("This GAM instance is not fitted yet. Call fit() first.")

    def _store_results(
        self,
        result: NewtonResult,
        setup: ModelSetup,
        spec: FormulaSpec,
        data: pd.DataFrame | dict,
        family_obj: ExponentialFamily,
        fd: FittingData,
        lambda_strategy: str,
    ) -> None:
        """Transfer Phase 2 output to fitted attributes.

        Sets all ``*_`` attributes listed in the class docstring's
        Attributes section (coefficients_, Vp_, edf_, etc.), plus
        internal bookkeeping attributes used by predict/summary/plot.
        """
        pr = result.pirls_result

        # Phase 2→3: transfer to NumPy
        coefficients = to_numpy(pr.coefficients)
        mu = to_numpy(pr.mu)
        eta = to_numpy(pr.eta)
        deviance = float(to_numpy(pr.deviance))
        L = to_numpy(pr.L)
        XtWX = to_numpy(pr.XtWX)
        scale = float(to_numpy(result.scale))
        smoothing_params = to_numpy(result.smoothing_params)
        edf_total = float(to_numpy(result.edf))

        X_np = setup.X
        y_np = setup.y
        wt_np = setup.weights

        # Compute H^{-1} via Cholesky solve (matches R's chol2inv).
        # O(p^3) but p is typically small (< 200 for GAMs).
        p = L.shape[0]
        Z = sla.solve_triangular(L, np.eye(p), lower=True)
        H_inv = Z.T @ Z

        # Per-smooth EDF via hat matrix F = H^{-1} @ XtWX
        # (invariant under repara — cyclic trace with block-diagonal D)
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
        phi = 1.0 if family_obj.scale_known else scale
        Vp = phi * H_inv

        # Null deviance
        null_deviance = _compute_null_deviance(y_np, wt_np, family_obj)

        # Store all fitted attributes (trailing underscore convention)
        self.coefficients_ = coefficients
        self.fitted_values_ = mu
        self.linear_predictor_ = eta
        self.family_ = family_obj
        self.Vp_ = Vp
        # Placeholder for frequentist covariance (not yet implemented).
        self.Ve_ = None
        self.scale_ = scale
        self.edf_ = per_smooth_edf
        self.edf1_ = per_smooth_edf1
        self.edf_total_ = edf_total
        self.smoothing_params_ = smoothing_params
        self.deviance_ = deviance
        self.null_deviance_ = null_deviance
        self.n_ = setup.n_obs
        self.converged_ = result.converged
        self.n_iter_ = result.n_iter
        self.X_ = X_np
        self.offset_ = setup.offset
        self.coef_map_ = setup.coef_map
        self.smooth_info_ = setup.smooth_info
        self.term_names_ = setup.term_names
        # Currently always "jax"; reserved for future backend dispatch.
        self.execution_path_ = "jax"
        self.lambda_strategy_ = lambda_strategy
        self.formula_spec_ = spec
        self._factor_info_ = _extract_factor_info(spec.parametric_terms, data)
        self.y_ = y_np
        self.weights_ = wt_np
        self.score_ = float(to_numpy(result.score))
        self._training_data = _extract_training_data(spec, data)


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str | None) -> jax.Device | None:
    """Resolve a device string to a JAX device object."""
    if device is None:
        return None
    import jax

    if device == "cpu":
        return jax.devices("cpu")[0]
    if device == "gpu":
        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError:
            gpu_devices = []
        if not gpu_devices:
            raise RuntimeError(
                "device='gpu' requested but no GPU backend found. "
                "Install jax[cuda12] (NVIDIA) or jax-metal (Apple Silicon)."
            )
        return gpu_devices[0]
    # _check_scope_guards validates device before this is called, so this
    # line is unreachable.  Raise explicitly for defensive clarity.
    raise ValueError(f"Unrecognized device: {device!r}")


def _check_scope_guards(method: str, kwargs: dict) -> None:
    """Validate v1.0 scope guards."""
    method_upper = method.upper()
    if method_upper not in ("REML", "ML"):
        raise ValueError(
            f"method must be 'REML' or 'ML', got {method!r}. "
            "GCV/UBRE is planned for v1.1."
        )

    backend = kwargs.get("backend")
    if backend is not None and backend != "jax":
        raise NotImplementedError(
            f"backend={backend!r} is not supported in v1.0. "
            "Only 'jax' backend is available. See docs/design.md Section 10."
        )

    device = kwargs.get("device")
    if device is not None and device not in ("cpu", "gpu"):
        raise ValueError(
            f"device={device!r} is not recognized. Use 'cpu', 'gpu', or None."
        )

    optimizer = kwargs.get("optimizer")
    if optimizer is not None and optimizer != "newton":
        raise NotImplementedError(
            f"optimizer={optimizer!r} is not supported in v1.0. "
            "Only 'newton' optimizer is available."
        )

    if kwargs.get("select", False):
        raise NotImplementedError(
            "select=True (shrinkage smoothing) is planned for v1.1. "
            "See docs/design.md Section 4.6."
        )

    gamma = kwargs.get("gamma", 1.0)
    if gamma != 1.0:
        raise NotImplementedError(
            f"gamma={gamma} is not supported in v1.0. "
            "Only gamma=1.0 (standard REML/ML) is available."
        )

    if kwargs.get("knots") is not None:
        raise NotImplementedError(
            "User-specified knots are planned for v1.1. See docs/design.md Section 5.2."
        )


def _fit_fixed_sp(fd: FittingData, sp: np.ndarray | list) -> NewtonResult:
    """Fit with user-supplied fixed smoothing parameters.

    Runs a single PIRLS at the given lambda (no Newton optimization).

    Parameters
    ----------
    fd : FittingData
        Phase 1→2 boundary data.
    sp : array-like
        Smoothing parameters on the original scale, shape ``(n_penalties,)``.

    Returns
    -------
    NewtonResult
        Result with ``n_iter=0``, ``convergence_info="fixed sp"``.
    """
    import jax.numpy as jnp

    sp_arr = np.asarray(sp, dtype=np.float64)
    if sp_arr.shape[0] != fd.n_penalties:
        raise ValueError(
            f"sp has {sp_arr.shape[0]} elements but model has "
            f"{fd.n_penalties} penalty terms."
        )

    log_lambda = jnp.log(jnp.array(sp_arr))
    S_lam = fd.S_lambda(log_lambda)

    # Initialize beta and run PIRLS
    beta_init = initialize_beta(
        np.asarray(fd.X),
        np.asarray(fd.y),
        np.asarray(fd.wt),
        fd.family,
        np.asarray(fd.offset) if fd.offset is not None else None,
    )

    pirls_result = pirls_loop(
        fd.X,
        fd.y,
        beta_init,
        S_lam,
        fd.family,
        wt=fd.wt,
        offset=fd.offset,
    )

    # Compute EDF and scale
    edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
    scale = estimate_scale(
        fd.y,
        pirls_result.mu,
        fd.wt,
        fd.family,
        edf,
    )

    return NewtonResult(
        log_lambda=log_lambda,
        smoothing_params=jnp.exp(log_lambda),
        converged=bool(pirls_result.converged),
        n_iter=0,
        score=jnp.array(0.0),
        gradient=jnp.zeros_like(log_lambda),
        edf=edf,
        scale=scale,
        pirls_result=pirls_result,
        convergence_info="fixed sp",
    )


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
) -> float:
    """Null model deviance.

    Uses the weighted mean of y as the null model prediction.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Response values.
    wt : np.ndarray, shape (n,)
        Prior weights.
    family : ExponentialFamily
        Family with ``dev_resids()`` method.

    Returns
    -------
    float
        Null model deviance.
    """
    mu_null = np.sum(wt * y) / np.sum(wt)
    mu_null_arr = np.full_like(y, mu_null)
    return float(family.dev_resids(y, mu_null_arr, wt))


def _extract_factor_info(
    parametric_terms: list[ParametricTerm],
    data: pd.DataFrame | dict,
) -> dict[str, list]:
    """Extract factor level info from parametric terms at training time.

    Parameters
    ----------
    parametric_terms : list[ParametricTerm]
        Parametric terms from the formula.
    data : DataFrame or dict
        Training data.

    Returns
    -------
    dict[str, list]
        Mapping from factor variable name to its ordered levels.
    """

    factor_info: dict[str, list] = {}
    for term in parametric_terms:
        col = data[term.name]
        if is_factor(col):
            factor_info[term.name] = get_factor_levels(col)
    return factor_info


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


def _build_parametric_predict(
    parametric_terms: list[ParametricTerm],
    data: pd.DataFrame | dict,
    has_intercept: bool,
    n_obs: int,
    factor_info: dict[str, list],
) -> np.ndarray:
    """Build the parametric portion of the prediction matrix.

    Uses stored factor levels from training time for consistent encoding.

    Parameters
    ----------
    parametric_terms : list[ParametricTerm]
        Parametric terms from formula.
    data : DataFrame or dict
        New data for prediction.
    has_intercept : bool
        Whether model includes an intercept.
    n_obs : int
        Number of new observations.
    factor_info : dict[str, list]
        Training-time factor levels.

    Returns
    -------
    np.ndarray, shape ``(n_obs, n_parametric_cols)``
    """

    blocks: list[np.ndarray] = []

    if has_intercept:
        blocks.append(np.ones((n_obs, 1), dtype=np.float64))

    for term in parametric_terms:
        col = data[term.name]

        if term.name in factor_info:
            # Use training-time levels for consistent dummy encoding
            levels = factor_info[term.name]
            drop_ref = has_intercept
            dummy, _ = ModelSetup._encode_factor(col, levels, drop_reference=drop_ref)
            blocks.append(dummy)
        else:
            col_arr = np.asarray(col, dtype=np.float64).ravel()
            blocks.append(col_arr[:, np.newaxis])

    if blocks:
        return np.column_stack(blocks)
    return np.empty((n_obs, 0), dtype=np.float64)
