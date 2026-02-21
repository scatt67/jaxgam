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
from pymgcv.fitting.newton import NewtonResult, newton_optimize
from pymgcv.fitting.pirls import pirls_loop
from pymgcv.fitting.reml import estimate_edf, estimate_scale
from pymgcv.formula.design import ModelSetup, SmoothInfo
from pymgcv.formula.parser import parse_formula
from pymgcv.formula.terms import FormulaSpec, ParametricTerm
from pymgcv.jax_utils import to_numpy
from pymgcv.smooths.by_variable import _get_factor_levels, is_factor

if TYPE_CHECKING:
    import pandas as pd


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
    **kwargs
        Additional arguments. Supported v1.0 scope guards:
        ``backend``, ``device``, ``optimizer``, ``select``, ``gamma``,
        ``knots``.

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
        fd = FittingData.from_setup(setup, family_obj)

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
        type: str = "response",
        se_fit: bool = False,
        offset: np.ndarray | None = None,
    ):
        """Predict from a fitted GAM.

        Parameters
        ----------
        newdata : pandas.DataFrame or dict, optional
            New data for prediction. If None, uses the training data.
        type : str
            Type of prediction: ``'response'`` or ``'link'``.
        se_fit : bool
            Whether to return standard errors.
        offset : array-like, optional
            Offset for new data predictions.

        Returns
        -------
        numpy.ndarray or tuple
            Predictions, or ``(predictions, standard_errors)`` if ``se_fit=True``.
        """
        self._check_fitted()

        if type not in ("response", "link"):
            raise ValueError(f"type must be 'response' or 'link', got {type!r}")

        if newdata is None:
            # Self-prediction: use stored linear predictor
            eta = self.linear_predictor_.copy()
            X_p = self.X_ if se_fit else None
        else:
            X_p = self._build_predict_matrix(newdata)
            eta = X_p @ self.coefficients_
            if offset is not None:
                eta = eta + np.asarray(offset, dtype=np.float64).ravel()

        if type == "response":
            pred = self.family_.link.linkinv(eta)
        else:
            pred = eta

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
        assert X_p.shape[1] == coef_map.total_coefs
        return X_p

    def summary(self):
        """Print summary of a fitted GAM."""
        self._check_fitted()
        raise NotImplementedError(
            "summary() is planned for Task 3.2. See IMPLEMENTATION_PLAN.md."
        )

    def plot(
        self,
        select: int | list | None = None,
        pages: int = 0,
        rug: bool = True,
        se: bool = True,
        shade: bool = True,
    ):
        """Plot smooth components of a fitted GAM."""
        self._check_fitted()
        raise NotImplementedError(
            "plot() is planned for Task 3.3. See IMPLEMENTATION_PLAN.md."
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
        """Transfer Phase 2 output to fitted attributes."""
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

        # Compute H^{-1} once, reuse for Vp and per-smooth EDF
        p = L.shape[0]
        Z = sla.solve_triangular(L, np.eye(p), lower=True)
        H_inv = Z.T @ Z

        # Bayesian covariance
        phi = 1.0 if family_obj.scale_known else scale
        Vp = phi * H_inv

        # Per-smooth EDF via hat matrix F = H^{-1} @ XtWX
        F = H_inv @ XtWX
        per_smooth_edf = _compute_per_smooth_edf(F, setup.smooth_info)

        # Null deviance
        null_deviance = _compute_null_deviance(y_np, wt_np, family_obj)

        # Store all fitted attributes (trailing underscore convention)
        self.coefficients_ = coefficients
        self.fitted_values_ = mu
        self.linear_predictor_ = eta
        self.family_ = family_obj
        self.Vp_ = Vp
        self.Ve_ = None
        self.scale_ = scale
        self.edf_ = per_smooth_edf
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
        self.execution_path_ = "jax"
        self.lambda_strategy_ = lambda_strategy
        self.formula_spec_ = spec
        self._factor_info_ = _extract_factor_info(spec.parametric_terms, data)


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------


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
    if device is not None and device == "gpu":
        raise NotImplementedError(
            "GPU execution is planned for v1.1. See docs/design.md Section 10.2."
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

    from pymgcv.fitting.initialization import initialize_beta

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


def _compute_vp(
    L: np.ndarray,
    scale: float,
    family: ExponentialFamily,
) -> np.ndarray:
    """Bayesian covariance matrix: ``phi * H^{-1}``.

    Parameters
    ----------
    L : np.ndarray, shape (p, p)
        Lower Cholesky factor of H = XtWX + S_lambda.
    scale : float
        Estimated dispersion parameter.
    family : ExponentialFamily
        Family (for scale_known check).

    Returns
    -------
    np.ndarray, shape (p, p)
        Bayesian covariance matrix Vp.
    """
    p = L.shape[0]
    Z = sla.solve_triangular(L, np.eye(p), lower=True)
    H_inv = Z.T @ Z
    phi = 1.0 if family.scale_known else scale
    return phi * H_inv


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
    import pandas as pd

    factor_info: dict[str, list] = {}
    for term in parametric_terms:
        if isinstance(data, pd.DataFrame):
            col = data[term.name]
        else:
            col = data[term.name]
        if is_factor(col):
            factor_info[term.name] = _get_factor_levels(col)
    return factor_info


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
    import pandas as pd

    blocks: list[np.ndarray] = []

    if has_intercept:
        blocks.append(np.ones((n_obs, 1), dtype=np.float64))

    for term in parametric_terms:
        if isinstance(data, pd.DataFrame):
            col = data[term.name]
        else:
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
