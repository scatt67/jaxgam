"""Shared NumPy prediction finishing for fitted GAMs."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from jaxgam.formula.predict_matrix import Data, PredictSpec
    from jaxgam.links.links import Link


def finish_prediction(
    eta: npt.NDArray[np.floating],
    X_p: npt.NDArray[np.floating],
    link: Link,
    Vp: npt.NDArray[np.floating],
    *,
    pred_type: str,
    se_fit: bool,
) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.floating]]:
    """Transform a linear predictor and optionally compute prediction SEs."""
    pred = link.linkinv(eta) if pred_type == "response" else eta
    if not se_fit:
        return pred

    # Preserve the exact operation order used by GAMResults.predict.
    se = np.sqrt(np.sum((X_p @ Vp) * X_p, axis=1))
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
) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.floating]]:
    """Build a prediction matrix and finish predictions on the CPU."""
    if pred_type not in ("response", "link"):
        raise ValueError(f"pred_type must be 'response' or 'link', got {pred_type!r}")

    X_p = spec.build_predict_matrix(newdata)
    eta = X_p @ coefficients
    if offset is not None:
        eta = eta + np.asarray(offset, dtype=np.float64).ravel()
    elif offset_was_nonzero:
        warnings.warn(
            "This model was fit with an external offset, but no `offset=` "
            "was supplied to predict() on new data. The offset is omitted "
            "from the returned predictions (matching mgcv predict.gam for "
            "external offsets). Pass `offset=` to include it.",
            stacklevel=3,
        )

    return finish_prediction(
        eta,
        X_p,
        link,
        Vp,
        pred_type=pred_type,
        se_fit=se_fit,
    )
