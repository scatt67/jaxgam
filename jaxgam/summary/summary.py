"""summary.gam equivalent: parametric and smooth term tables.

Port of R's ``summary.gam()`` (mgcv.r lines 3858-4068) and
``testStat()`` (mgcv.r lines 3759-3853).

Phase 3 code: NumPy + SciPy only, no JAX imports.

Key references:
- Wood (2013) "A simple test statistic for testing against a one-sided
  alternative", Biometrika 100(1), 221-228.
- Wood (2017) "Generalized Additive Models: An Introduction with R",
  2nd edition, Section 6.12.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from jaxgam.api import GAM


# ---------------------------------------------------------------------------
# GAMSummary dataclass
# ---------------------------------------------------------------------------


@dataclass
class GAMSummary:
    """Summary of a fitted GAM.

    Contains parametric coefficient table, smooth term table, and
    model-level statistics, matching the structure of R's
    ``summary.gam()`` output.

    Attributes
    ----------
    formula : str
        The model formula.
    family_name : str
        Distribution family name.
    link_name : str
        Link function name.
    method : str
        Smoothing parameter estimation method (REML/ML).
    n : int
        Number of observations.

    p_table : np.ndarray or None
        Parametric coefficient table, shape ``(n_parametric, 4)``.
        Columns: Estimate, Std. Error, z/t value, p-value.
    p_names : list[str]
        Row names for ``p_table``.
    p_test_name : str
        Either ``"z value"`` or ``"t value"``.
    p_pv_name : str
        Either ``"Pr(>|z|)"`` or ``"Pr(>|t|)"``.

    s_table : np.ndarray or None
        Smooth term table, shape ``(n_smooths, 4)``.
        Columns: edf, Ref.df, F/Chi.sq, p-value.
    s_names : list[str]
        Row names for ``s_table``.
    s_test_name : str
        Either ``"F"`` or ``"Chi.sq"``.

    r_sq : float or None
        Adjusted R-squared.
    dev_explained : float
        Proportion of deviance explained.
    scale : float
        Estimated or fixed scale (dispersion) parameter.
    reml_score : float or None
        REML/ML criterion value (sp.criterion in R).
    residual_df : float
        Residual degrees of freedom.
    edf_total : float
        Total effective degrees of freedom.
    """

    formula: str
    family_name: str
    link_name: str
    method: str
    n: int

    p_table: np.ndarray | None
    p_names: list[str]
    p_test_name: str
    p_pv_name: str

    s_table: np.ndarray | None
    s_names: list[str] = field(default_factory=list)
    s_test_name: str = "F"

    r_sq: float | None = None
    dev_explained: float = 0.0
    scale: float = 1.0
    reml_score: float | None = None
    residual_df: float = 0.0
    edf_total: float = 0.0

    def __str__(self) -> str:
        """Formatted summary string matching R's print.summary.gam."""
        return _format_summary(self)


# ---------------------------------------------------------------------------
# Core summary computation
# ---------------------------------------------------------------------------


def summary(gam: GAM) -> GAMSummary:
    """Compute summary statistics for a fitted GAM.

    Port of R's ``summary.gam()`` (mgcv.r lines 3858-4068).

    Parameters
    ----------
    gam : GAM
        A fitted GAM object.

    Returns
    -------
    GAMSummary
        Summary object with parametric and smooth term tables.

    Raises
    ------
    RuntimeError
        If the model is not fitted.
    """
    gam._check_fitted()

    family = gam.family_
    est_disp = not family.scale_known
    dispersion = gam.scale_

    # Bayesian covariance matrix
    covmat = gam.Vp_

    # Parametric coefficients
    se = np.sqrt(np.diag(covmat))
    residual_df = gam.n_ - gam.edf_total_

    # Total parametric columns: count from coef_map
    n_parametric = _count_parametric(gam)

    p_table = None
    p_names: list[str] = []
    p_test_name = "t value" if est_disp else "z value"
    p_pv_name = "Pr(>|t|)" if est_disp else "Pr(>|z|)"

    if n_parametric > 0:
        ind = np.arange(n_parametric)
        p_coeff = gam.coefficients_[ind]
        p_se = se[ind]
        p_t = p_coeff / p_se

        if not est_disp:
            # Known scale: z-test (Normal)
            p_pv = 2.0 * stats.norm.sf(np.abs(p_t))
        else:
            # Unknown scale: t-test
            p_pv = 2.0 * stats.t.sf(np.abs(p_t), df=residual_df)

        p_table = np.column_stack([p_coeff, p_se, p_t, p_pv])

        # Get parametric coefficient names from term_names
        p_names = list(gam.term_names_[:n_parametric])

    # Smooth terms
    smooth_info = gam.smooth_info_
    m = len(smooth_info)

    s_table = None
    s_names: list[str] = []
    s_test_name = "F" if est_disp else "Chi.sq"

    if m > 0:
        edf_arr = np.zeros(m)
        edf1_arr = np.zeros(m)
        ref_df = np.zeros(m)
        chi_sq = np.zeros(m)
        s_pv = np.zeros(m)

        # Use the stored model matrix X
        X = gam.X_

        for i, si in enumerate(smooth_info):
            start = si.first_coef
            stop = si.last_coef
            n_coefs_i = stop - start

            V_i = covmat[start:stop, start:stop]
            p_i = gam.coefficients_[start:stop]
            edf_i = float(gam.edf_[i])
            # Use edf1 (= 2*edf - trace(F^2)) as reference df for the test,
            # matching R's summary.gam (mgcv.r line 4019/4027).
            # edf1_ may be absent on models serialized before edf1 was added
            edf1_i = float(gam.edf1_[i]) if hasattr(gam, "edf1_") else edf_i

            X_i = X[:, start:stop]

            rdf = residual_df if est_disp else -1.0

            res = _test_stat(
                p_i,
                X_i,
                V_i,
                rank=min(n_coefs_i, edf1_i),
                type_=0,
                res_df=rdf,
            )

            ref_df[i] = res["rank"]
            chi_sq[i] = res["stat"]
            s_pv[i] = res["pval"]
            edf_arr[i] = edf_i
            edf1_arr[i] = edf1_i
            s_names.append(si.label)

        if not est_disp:
            s_table = np.column_stack([edf_arr, ref_df, chi_sq, s_pv])
        else:
            # For F-test: divide test stat by ref.df
            f_stat = chi_sq / ref_df
            s_table = np.column_stack([edf_arr, ref_df, f_stat, s_pv])

    # Model-level statistics
    r_sq = _compute_r_squared(gam, residual_df)
    dev_explained = _compute_deviance_explained(gam)

    # REML/ML score
    reml_score = getattr(gam, "score_", None)

    return GAMSummary(
        formula=gam.formula,
        family_name=family.family_name,
        link_name=type(family.link).__name__.replace("Link", "").lower(),
        method=gam.method,
        n=gam.n_,
        p_table=p_table,
        p_names=p_names,
        p_test_name=p_test_name,
        p_pv_name=p_pv_name,
        s_table=s_table,
        s_names=s_names,
        s_test_name=s_test_name,
        r_sq=r_sq,
        dev_explained=dev_explained,
        scale=dispersion,
        reml_score=reml_score,
        residual_df=residual_df,
        edf_total=gam.edf_total_,
    )


# ---------------------------------------------------------------------------
# Wood (2013) testStat -- smooth significance test
# ---------------------------------------------------------------------------


def _test_stat(
    p: np.ndarray,
    X: np.ndarray,
    V: np.ndarray,
    rank: float | None = None,
    type_: int = 0,
    res_df: float = -1.0,
) -> dict[str, float]:
    """Wood (2013) test statistic for smooth significance.

    Port of R's ``testStat`` (mgcv.r lines 3759-3853).

    Parameters
    ----------
    p : np.ndarray, shape (k,)
        Coefficient vector for the smooth.
    X : np.ndarray, shape (n, k)
        Model matrix block for the smooth.
    V : np.ndarray, shape (k, k)
        Bayesian covariance matrix block for the smooth.
    rank : float or None
        EDF estimate (possibly fractional).
    type_ : int
        Truncation type. 0 = fractional pinv (default).
        1 = round to integer.
    res_df : float
        Residual df. <= 0 implies fixed (known) scale.

    Returns
    -------
    dict
        Keys: ``stat`` (test statistic), ``pval`` (p-value),
        ``rank`` (reference degrees of freedom).
    """
    # QR of X to get R factor
    _Q_x, R_x = np.linalg.qr(X, mode="reduced")
    # Transform V into R-space: R @ V @ R^T
    V_r = R_x @ V @ R_x.T
    V_r = (V_r + V_r.T) / 2.0

    # Eigendecomposition of V_r
    eigvals, eigvecs = np.linalg.eigh(V_r)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Remove sign ambiguity (match R: sign of first element)
    siv = np.sign(eigvecs[0, :])
    siv[siv == 0] = 1.0
    eigvecs = eigvecs * siv[np.newaxis, :]

    # Compute k and nu (integer and fractional parts of rank)
    k = max(0, int(np.floor(rank)))
    nu = abs(rank - k)

    if type_ == 1:
        # Round up if more than 0.05 above lower integer
        if rank > k + 0.05 or k == 0:
            k = k + 1
        nu = 0.0
        rank = float(k)

    k1 = k + 1 if nu > 0 else k

    # Check actual numerical rank
    r_est = int(np.sum(eigvals > np.max(eigvals) * np.finfo(float).eps ** 0.9))
    if r_est < k1:
        k1 = r_est
        k = r_est
        nu = 0.0
        rank = float(r_est)

    # Truncate eigenvectors
    vec = eigvecs.copy()
    if k1 < vec.shape[1]:
        vec = vec[:, :k1]

    # Deal with fractional part of the pseudo-inverse
    if nu > 0 and k > 0:
        # Scale the first k-1 eigenvectors by 1/sqrt(eigenvalue)
        if k > 1:
            vec[:, : k - 1] = vec[:, : k - 1] / np.sqrt(eigvals[: k - 1])[np.newaxis, :]

        b12 = 0.5 * nu * (1.0 - nu)
        if b12 < 0:
            b12 = 0.0
        b12 = np.sqrt(b12)
        B = np.array([[1.0, b12], [b12, nu]])

        ev_diag = np.diag(eigvals[k - 1 : k1] ** (-0.5))
        B = ev_diag @ B @ ev_diag

        eb_vals, eb_vecs = np.linalg.eigh(B)
        # Sort descending
        eb_idx = np.argsort(eb_vals)[::-1]
        eb_vals = eb_vals[eb_idx]
        eb_vecs = eb_vecs[:, eb_idx]

        rB = eb_vecs @ np.diag(np.sqrt(np.maximum(eb_vals, 0.0))) @ eb_vecs.T

        vec1 = vec.copy()
        # R: vec1[,k:k1] <- t(rB %*% diag(c(-1,1)) %*% t(vec[,k:k1]))
        # R's vec[,k:k1] selects columns k..k1 (1-based).
        # Python's vec[:, k-1:k1] selects columns k-1..k1-1 (0-based).
        sign_diag = np.diag(np.array([-1.0, 1.0]))
        cols = vec[:, k - 1 : k1]  # shape (p, 2)
        vec1[:, k - 1 : k1] = (rB @ sign_diag @ cols.T).T
        vec[:, k - 1 : k1] = (rB @ cols.T).T
    else:
        # Integer rank case
        if k == 0:
            vec = vec / np.sqrt(eigvals[0])
            vec1 = vec.copy()
        else:
            vec = vec / np.sqrt(eigvals[:k])[np.newaxis, :]
            vec1 = vec.copy()
        if k == 1:
            rank = 1.0

    # Compute test statistics
    Rp = R_x @ p
    d = float(np.sum((vec.T @ Rp) ** 2))
    d1 = float(np.sum((vec1.T @ Rp) ** 2))

    rank1 = rank  # rank for lower tail computation

    use_integer_fallback = True
    pval = 0.0

    if nu > 0:
        # Mixture of chi-squared reference distribution
        if k1 == 1:
            rank1 = 1.0
            val = np.array([1.0])
        else:
            val = np.ones(k1)
            rp = nu + 1.0
            val[k - 1] = (rp + np.sqrt(rp * (2.0 - rp))) / 2.0
            val[k1 - 1] = rp - val[k - 1]

        if res_df <= 0:
            # Known scale: mixture of chi-squared (Davies exact)
            pval = (psum_chisq_davies(d, val) + psum_chisq_davies(d1, val)) / 2.0
        else:
            # Unknown scale: F-like mixture via Davies
            # R's testStat line 3839: difference of two weighted chi-sq sums
            res_df_int = max(1, round(res_df))
            lb_d = np.concatenate([val, [-d / res_df_int]])
            lb_d1 = np.concatenate([val, [-d1 / res_df_int]])
            df_arr = np.concatenate(
                [np.ones(len(val), dtype=int), np.array([res_df_int])]
            )
            pval = (
                psum_chisq_davies(0.0, lb_d, df=df_arr)
                + psum_chisq_davies(0.0, lb_d1, df=df_arr)
            ) / 2.0
        use_integer_fallback = pval > 1.0

    if use_integer_fallback:
        if res_df <= 0:
            # Known scale: chi-squared
            pval = (stats.chi2.sf(d, df=rank1) + stats.chi2.sf(d1, df=rank1)) / 2.0
        else:
            # Unknown scale: F distribution
            pval = (
                stats.f.sf(d / rank1, rank1, res_df)
                + stats.f.sf(d1 / rank1, rank1, res_df)
            ) / 2.0

    return {
        "stat": float(d),
        "pval": float(min(1.0, pval)),
        "rank": float(rank),
    }


# ---------------------------------------------------------------------------
# Weighted sum of chi-squared distributions
# ---------------------------------------------------------------------------


def _liu2(x: float, lam: np.ndarray, h: np.ndarray | None = None) -> float:
    """Liu et al. (2009) / Pearson (1959) approximation.

    Approximates Pr[sum_i lambda_i * chi^2_{h_i} > x] for central
    chi-squared variables.

    Port of R's ``liu2()`` (mgcv.r lines 3500-3554).

    Parameters
    ----------
    x : float
        Test statistic value.
    lam : np.ndarray
        Weights (can be negative).
    h : np.ndarray or None
        Degrees of freedom for each component. Defaults to all ones.

    Returns
    -------
    float
        Upper tail probability.
    """
    if h is None:
        h = np.ones(len(lam))

    lh = lam * h
    muQ = np.sum(lh)

    lh2 = lh * lam
    c2 = np.sum(lh2)

    lh3 = lh2 * lam
    c3 = np.sum(lh3)

    if x <= 0 or c2 <= 0:
        return 1.0

    lh4 = lh3 * lam
    s1 = c3 / c2**1.5
    s2 = np.sum(lh4) / c2**2

    sigQ = np.sqrt(2.0 * c2)
    t = (x - muQ) / sigQ

    if s1**2 > s2:
        a = 1.0 / (s1 - np.sqrt(s1**2 - s2))
        delta = s1 * a**3 - a**2
        ell = a**2 - 2.0 * delta
    else:
        a = 1.0 / s1
        delta = 0.0
        if c3 == 0:
            return 1.0
        ell = c2**3 / c3**2

    muX = ell + delta
    sigX = np.sqrt(2.0) * a
    z = t * sigX + muX

    return float(stats.ncx2.sf(z, df=ell, nc=delta))


def psum_chisq_davies(
    q: float,
    lb: np.ndarray,
    df: np.ndarray | None = None,
    nc: np.ndarray | None = None,
    sigz: float = 0.0,
    tol: float = 2e-5,
    nlim: int = 100_000,
) -> float:
    """Upper tail probability for weighted sum of chi-squared variables.

    Computes ``Pr(sum_j lb[j] * X_j + sigz * Z > q)`` where
    ``X_j ~ chi^2(df[j], nc[j])`` and ``Z ~ N(0, 1)``.

    Uses Davies' exact method with Liu et al. (2009) fallback.

    Port of R's ``psum.chisq()`` (mgcv.r lines 3466-3498).

    Parameters
    ----------
    q : float
        Quantile.
    lb : np.ndarray
        Weights (can be either sign).
    df : np.ndarray or None
        Degrees of freedom (positive integers). Defaults to all ones.
    nc : np.ndarray or None
        Non-centrality parameters. Defaults to all zeros.
    sigz : float
        Standard deviation of the normal component.
    tol : float
        Accuracy tolerance.
    nlim : int
        Maximum number of integration terms.

    Returns
    -------
    float
        Upper tail probability.
    """
    from jaxgam.summary._davies import _davies

    lb = np.asarray(lb, dtype=np.float64)
    r = len(lb)

    if df is None:
        df = np.ones(r, dtype=np.int64)
    else:
        df = np.round(np.asarray(df, dtype=np.float64)).astype(np.int64)

    if nc is None:
        nc = np.zeros(r, dtype=np.float64)
    else:
        nc = np.asarray(nc, dtype=np.float64)

    if sigz < 0:
        sigz = 0.0

    result = _davies(lb, nc, df, sigz, q, lim=nlim, acc=tol)

    if result.ifault in (0, 2):
        return float(1.0 - result.prob)
    # Fallback to Liu approximation
    if np.all(nc == 0):
        return float(_liu2(q, lb, h=df))
    return float(np.nan)


# ---------------------------------------------------------------------------
# Model-level statistics
# ---------------------------------------------------------------------------


def _count_parametric(gam: GAM) -> int:
    """Count the number of parametric coefficient columns.

    Parameters
    ----------
    gam : GAM
        Fitted GAM object.

    Returns
    -------
    int
        Number of parametric columns (intercept + linear + factor dummies).
    """
    coef_map = gam.coef_map_
    for term in coef_map.terms:
        if term.term_type == "parametric":
            return term.n_coefs
    return 0


def _compute_r_squared(gam: GAM, residual_df: float) -> float | None:
    """Compute adjusted R-squared.

    Matches R's formula in ``summary.gam`` (mgcv.r line 4056)::

        r.sq = 1 - var(w*(y - fitted)) * (n-1) /
               (var(w*(y - mean_y)) * residual_df)

    Parameters
    ----------
    gam : GAM
        Fitted GAM object.
    residual_df : float
        Residual degrees of freedom.

    Returns
    -------
    float or None
        Adjusted R-squared, or None if not applicable.
    """
    n = gam.n_
    if residual_df <= 0 or n <= 1:
        return None

    y_vals = getattr(gam, "y_", None)
    if y_vals is None:
        return None

    w = getattr(gam, "weights_", np.ones(n))
    sqrt_w = np.sqrt(w)

    mean_y = np.sum(w * y_vals) / np.sum(w)

    # R uses: var() which divides by (n-1)
    resid = sqrt_w * (y_vals - gam.fitted_values_)
    null = sqrt_w * (y_vals - mean_y)

    var_resid = np.var(resid, ddof=1)
    var_null = np.var(null, ddof=1)

    if var_null == 0:
        return None

    r_sq = 1.0 - var_resid * (n - 1) / (var_null * residual_df)
    return float(r_sq)


def _compute_deviance_explained(gam: GAM) -> float:
    """Compute proportion of deviance explained.

    Formula: ``(null_deviance - deviance) / null_deviance``

    Parameters
    ----------
    gam : GAM
        Fitted GAM object.

    Returns
    -------
    float
        Proportion of deviance explained.
    """
    if gam.null_deviance_ == 0:
        return 0.0
    return float((gam.null_deviance_ - gam.deviance_) / gam.null_deviance_)


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------


def _format_summary(s: GAMSummary) -> str:
    """Format GAMSummary as a string matching R's print.summary.gam.

    Parameters
    ----------
    s : GAMSummary
        Summary object.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines: list[str] = []

    # Family and link
    lines.append(f"Family: {s.family_name}")
    lines.append(f"Link function: {s.link_name}")
    lines.append("")
    lines.append("Formula:")
    lines.append(s.formula)

    # Parametric coefficients
    if s.p_table is not None and len(s.p_names) > 0:
        lines.append("")
        lines.append("Parametric coefficients:")
        # Header
        max_name_len = max(len(n) for n in s.p_names)
        header = (
            f"{'':>{max_name_len}}  "
            f"{'Estimate':>12}  "
            f"{'Std. Error':>12}  "
            f"{s.p_test_name:>12}  "
            f"{s.p_pv_name:>12}"
        )
        lines.append(header)

        for i, name in enumerate(s.p_names):
            row = s.p_table[i]
            sig = _signif_code(row[3])
            lines.append(
                f"{name:>{max_name_len}}  "
                f"{row[0]:>12.5f}  "
                f"{row[1]:>12.5f}  "
                f"{row[2]:>12.3f}  "
                f"{_format_pval(row[3]):>12} {sig}"
            )
        lines.append(_signif_legend())

    # Smooth terms
    if s.s_table is not None and len(s.s_names) > 0:
        lines.append("")
        lines.append("Approximate significance of smooth terms:")
        max_name_len = max(len(n) for n in s.s_names)
        header = (
            f"{'':>{max_name_len}}  "
            f"{'edf':>8}  "
            f"{'Ref.df':>8}  "
            f"{s.s_test_name:>10}  "
            f"{'p-value':>12}"
        )
        lines.append(header)

        for i, name in enumerate(s.s_names):
            row = s.s_table[i]
            sig = _signif_code(row[3])
            lines.append(
                f"{name:>{max_name_len}}  "
                f"{row[0]:>8.3f}  "
                f"{row[1]:>8.3f}  "
                f"{row[2]:>10.3f}  "
                f"{_format_pval(row[3]):>12} {sig}"
            )
        lines.append(_signif_legend())

    # Model statistics
    lines.append("")
    if s.r_sq is not None:
        lines.append(
            f"R-sq.(adj) =  {s.r_sq:.3f}   "
            f"Deviance explained = {s.dev_explained * 100:.1f}%"
        )
    else:
        lines.append(f"Deviance explained = {s.dev_explained * 100:.1f}%")

    if s.reml_score is not None:
        lines.append(
            f"-{s.method} = {s.reml_score:.5f}   "
            f"Scale est. = {s.scale:<8.5f}   n = {s.n}"
        )
    else:
        lines.append(f"Scale est. = {s.scale:<8.5f}   n = {s.n}")

    return "\n".join(lines)


def _format_pval(p: float) -> str:
    """Format a p-value for display."""
    if p < 2e-16:
        return "< 2e-16"
    elif p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"


def _signif_code(p: float) -> str:
    """Return significance code for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    return " "


def _signif_legend() -> str:
    """Return significance legend string."""
    return "---\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
