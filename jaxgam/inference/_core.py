"""Shared NumPy prediction finishing for fitted GAMs."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla

if TYPE_CHECKING:
    from jaxgam.formula.predict_matrix import Data, PredictSpec
    from jaxgam.inference.predictor import PivotedQRFisherFactor
    from jaxgam.links.links import Link


def finish_prediction(
    eta: npt.NDArray[np.floating],
    X_p: npt.NDArray[np.floating],
    link: Link,
    Vp: npt.NDArray[np.floating] | None,
    *,
    pred_type: str,
    se_fit: bool,
    fisher_factor: npt.NDArray[np.floating] | None = None,
    fisher_qr_factor: PivotedQRFisherFactor | None = None,
    fisher_transforms: tuple[tuple[int, int, str, npt.NDArray[np.floating]], ...] = (),
    fisher_scale: float = 1.0,
) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.floating]]:
    """Transform a linear predictor and optionally compute prediction SEs."""
    pred = link.linkinv(eta) if pred_type == "response" else eta
    if not se_fit:
        return pred

    if Vp is not None and fisher_qr_factor is None:
        # Preserve the exact operation order used by GAMResults.predict.
        se = np.sqrt(np.sum((X_p @ Vp) * X_p, axis=1))
    elif fisher_factor is not None or fisher_qr_factor is not None:
        X_fit = X_p.copy()
        for start, stop, kind, values in fisher_transforms:
            if kind == "dense":
                X_fit[:, start:stop] = X_fit[:, start:stop] @ values
            elif kind == "diagonal":
                X_fit[:, start:stop] *= values
        Z = (
            sla.solve_triangular(fisher_factor, X_fit.T, lower=True)
            if fisher_factor is not None
            else fisher_qr_factor.root_transpose_inverse(X_fit.T)
        )
        se = np.sqrt(fisher_scale * np.sum(Z * Z, axis=0))
    else:  # defensive: callers normally reject before reaching here.
        raise RuntimeError(
            "Prediction uncertainty is unavailable: no covariance provider."
        )
    if pred_type == "response":
        se = se * np.abs(np.asarray(link.mu_eta(eta)))
    return pred, se


def predict_core(
    spec: PredictSpec,
    coefficients: npt.NDArray[np.floating],
    Vp: npt.NDArray[np.floating],
    link: Link,
    newdata: Data,
    *,
    pred_type: str = "response",
    se_fit: bool = False,
    offset: npt.ArrayLike | None = None,
    offset_was_nonzero: bool = False,
    warning_stacklevel: int = 3,
) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.floating]]:
    """Build a prediction matrix and finish predictions on the CPU."""
    X_p, eta = prepare_prediction(
        spec,
        coefficients,
        newdata,
        pred_type=pred_type,
        offset=offset,
        offset_was_nonzero=offset_was_nonzero,
        warning_stacklevel=warning_stacklevel + 1,
    )
    return finish_prediction(eta, X_p, link, Vp, pred_type=pred_type, se_fit=se_fit)


def prepare_prediction(
    spec: PredictSpec,
    coefficients: npt.NDArray[np.floating],
    newdata: Data,
    *,
    pred_type: str,
    offset: npt.ArrayLike | None,
    offset_was_nonzero: bool,
    warning_stacklevel: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Use the one prediction encoder and validate external offsets."""
    if pred_type not in ("response", "link"):
        raise ValueError(f"pred_type must be 'response' or 'link', got {pred_type!r}")

    X_p = spec.build_predict_matrix(newdata)
    eta = X_p @ coefficients
    if offset is not None:
        offset_array = np.asarray(offset, dtype=np.float64)
        if offset_array.ndim != 1 or offset_array.shape[0] != X_p.shape[0]:
            raise ValueError(
                "offset must be a one-dimensional array with one value per "
                "prediction row."
            )
        if not np.all(np.isfinite(offset_array)):
            raise ValueError("offset must contain only finite values.")
        eta = eta + offset_array
    elif offset_was_nonzero:
        warnings.warn(
            "This model was fit with an external offset, but no `offset=` "
            "was supplied to predict() on new data. The offset is omitted "
            "from the returned predictions (matching mgcv predict.gam for "
            "external offsets). Pass `offset=` to include it.",
            stacklevel=warning_stacklevel,
        )

    return X_p, eta
