"""Post-estimation computation for fitted GAMs.

Extracts derived-quantity computation from the fit pipeline into a
standalone, testable module. Sits at the Phase 2→3 boundary: takes
raw JAX fitting output and produces NumPy-based post-estimation
results.

Design doc reference: docs/refactor_gam_api/design.md §4.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as sla

from jaxgam.jax_utils import to_numpy

if TYPE_CHECKING:
    from jaxgam.families.base import ExponentialFamily
    from jaxgam.fitting.data import FittingData
    from jaxgam.fitting.newton import NewtonResult
    from jaxgam.formula.design import ModelSetup, SmoothInfo


@dataclass(frozen=True)
class PostEstimationResults:
    """Derived quantities computed from raw fit output.

    All arrays are NumPy (Phase 3). This dataclass is the output of
    ``compute_post_estimation()`` and is consumed by ``GAMResults._from_fit()``
    (Phase 2 of the refactor).

    Attributes
    ----------
    coefficients : np.ndarray, shape (p,)
        Back-transformed coefficients (undone reparameterization).
    Vp : np.ndarray, shape (p, p)
        Bayesian posterior covariance of coefficients.
    edf : np.ndarray, shape (n_smooths,)
        Per-smooth effective degrees of freedom.
    edf1 : np.ndarray, shape (n_smooths,)
        Alternative per-smooth EDF for significance testing.
    edf_total : float
        Total effective degrees of freedom.
    null_deviance : float
        Null model deviance.
    hat_matrix : np.ndarray, shape (p, p)
        Hat-like matrix F = H^{-1} @ XtWX (kept for debugging).
    scale : float
        Estimated or fixed scale parameter (phi).
    """

    coefficients: np.ndarray
    Vp: np.ndarray
    edf: np.ndarray
    edf1: np.ndarray
    edf_total: float
    null_deviance: float
    hat_matrix: np.ndarray
    scale: float


def compute_post_estimation(
    result: NewtonResult,
    setup: ModelSetup,
    family: ExponentialFamily,
    fd: FittingData,
) -> PostEstimationResults:
    """Compute all derived quantities from raw fit output.

    This function encapsulates the covariance computation, EDF
    calculation, coefficient back-transformation, and null deviance
    -- everything that previously lived in ``GAM._store_results()``.

    Parameters
    ----------
    result : NewtonResult
        Raw output from Newton optimization or fixed-sp PIRLS.
    setup : ModelSetup
        Phase 1 model setup (holds X, y, weights, smooth_info).
    family : ExponentialFamily
        Distribution family with ``dev_resids()`` method.
    fd : FittingData
        Phase 1→2 boundary data (holds ``repara_D``).

    Returns
    -------
    PostEstimationResults
        All derived quantities needed by the results object.
    """
    pr = result.pirls_result

    # Phase 2→3: transfer to NumPy
    coefficients = to_numpy(pr.coefficients)
    L = to_numpy(pr.L)
    XtWX = to_numpy(pr.XtWX)
    scale = float(to_numpy(result.scale))
    edf_total = float(to_numpy(result.edf))

    # Compute H^{-1} via Cholesky solve (matches R's chol2inv).
    # O(p^3) but p is typically small (< 200 for GAMs).
    p = L.shape[0]
    Z = sla.solve_triangular(L, np.eye(p), lower=True)
    H_inv = Z.T @ Z

    # Per-smooth EDF via hat matrix F = H^{-1} @ XtWX
    # (invariant under repara -- cyclic trace with block-diagonal D)
    F = H_inv @ XtWX
    per_smooth_edf = compute_per_smooth_edf(F, setup.smooth_info)
    # edf1 = 2*edf - trace(F^2): alternative EDF for significance testing
    # (R's gam.fit3.post.proc, mgcv.r line 966)
    per_smooth_edf1 = compute_per_smooth_edf1(F, setup.smooth_info)

    # Back-transform from Sl.setup reparameterized space
    if fd.repara_D is not None:
        D = to_numpy(fd.repara_D)
        coefficients = D @ coefficients
        H_inv = D @ H_inv @ D.T

    # Bayesian covariance
    phi = 1.0 if family.scale_known else scale
    Vp = phi * H_inv

    # Null deviance
    null_deviance = compute_null_deviance(setup.y, setup.weights, family)

    return PostEstimationResults(
        coefficients=coefficients,
        Vp=Vp,
        edf=per_smooth_edf,
        edf1=per_smooth_edf1,
        edf_total=edf_total,
        null_deviance=null_deviance,
        hat_matrix=F,
        scale=scale,
    )


def compute_per_smooth_edf(
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


def compute_per_smooth_edf1(
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


def compute_null_deviance(
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
