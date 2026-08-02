"""Frozen, picklable prediction core for a fitted GAM."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

import jaxgam
from jaxgam.formula.predict_matrix import PredictSpec
from jaxgam.inference._core import predict_core

if TYPE_CHECKING:
    from jaxgam.families.base import ExponentialFamily
    from jaxgam.formula.predict_matrix import Data


@dataclass(frozen=True)
class GAMPredictor:
    """Lean, training-data-free prediction state for a fitted GAM.

    This is an optional boundary object for consumers that should receive only
    ``predict()`` and ``predict_matrix()`` state. A ``GAMInferenceResult`` is
    already lean and directly usable; its ``to_predictor()`` simply returns the
    predictor it already contains.

    Pickles are intended for trusted, same-version transient handoff. The
    coefficients and posterior covariance are defensively copied and exposed
    as read-only arrays.
    """

    coefficients: npt.NDArray[np.floating]
    Vp: npt.NDArray[np.floating]
    family: ExponentialFamily
    formula: str
    offset_was_nonzero: bool
    _predict_spec: PredictSpec
    _jaxgam_version: str = field(default_factory=lambda: jaxgam.__version__)

    def __post_init__(self) -> None:
        """Own and freeze the two arrays covered by the public contract."""
        object.__setattr__(self, "coefficients", np.array(self.coefficients))
        object.__setattr__(self, "Vp", np.array(self.Vp))
        self.coefficients.setflags(write=False)
        self.Vp.setflags(write=False)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore frozen arrays and report unsupported cross-version loads."""
        self.__dict__.update(state)
        self.coefficients.setflags(write=False)
        self.Vp.setflags(write=False)
        if self._jaxgam_version != jaxgam.__version__:
            warnings.warn(
                f"GAMPredictor was pickled by jaxgam {self._jaxgam_version}, "
                f"loading under {jaxgam.__version__}. Pickles are not a "
                "cross-version format; predictions may be wrong or fail.",
                stacklevel=2,
            )

    def predict(
        self,
        newdata: Data,
        pred_type: str = "response",
        se_fit: bool = False,
        offset: npt.ArrayLike | None = None,
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.floating]]:
        """Predict responses or linear predictors for new data."""
        return self._predict(
            newdata,
            pred_type=pred_type,
            se_fit=se_fit,
            offset=offset,
            warning_stacklevel=4,
        )

    def _predict(
        self,
        newdata: Data,
        *,
        pred_type: str,
        se_fit: bool,
        offset: npt.ArrayLike | None,
        warning_stacklevel: int,
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.floating]]:
        """Predict with an explicit warning depth for delegating wrappers."""
        return predict_core(
            self._predict_spec,
            self.coefficients,
            self.Vp,
            self.family.link,
            newdata,
            pred_type=pred_type,
            se_fit=se_fit,
            offset=offset,
            offset_was_nonzero=self.offset_was_nonzero,
            warning_stacklevel=warning_stacklevel,
        )

    def predict_matrix(self, newdata: Data) -> npt.NDArray[np.floating]:
        """Build the constrained linear-predictor matrix for new data."""
        return self._predict_spec.build_predict_matrix(newdata)
