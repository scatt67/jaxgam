"""Frozen, picklable prediction core for a fitted GAM."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

import jaxgam
from jaxgam.data.source import RowSource
from jaxgam.formula.predict_matrix import PredictSpec
from jaxgam.inference._core import finish_prediction, predict_core, prepare_prediction

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
    Vp: npt.NDArray[np.floating] | None
    family: ExponentialFamily
    formula: str
    offset_was_nonzero: bool
    _predict_spec: PredictSpec
    _fisher_factor: npt.NDArray[np.floating] | None = None
    _fisher_transforms: tuple[tuple[int, int, str, npt.NDArray[np.floating]], ...] = ()
    _fisher_scale: float = 1.0
    _output_budget_bytes: int | None = None
    _memory_budget_bytes: int | None = None
    _jaxgam_version: str = field(default_factory=lambda: jaxgam.__version__)

    def __post_init__(self) -> None:
        """Own and freeze the two arrays covered by the public contract."""
        object.__setattr__(self, "coefficients", np.array(self.coefficients))
        if self.Vp is not None:
            object.__setattr__(self, "Vp", np.array(self.Vp))
        self.coefficients.setflags(write=False)
        if self.Vp is not None:
            self.Vp.setflags(write=False)
        if self._fisher_factor is not None:
            object.__setattr__(self, "_fisher_factor", np.array(self._fisher_factor))
            self._fisher_factor.setflags(write=False)
        transforms = []
        for start, stop, kind, values in self._fisher_transforms:
            owned = np.array(values)
            owned.setflags(write=False)
            transforms.append((start, stop, kind, owned))
        object.__setattr__(self, "_fisher_transforms", tuple(transforms))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore frozen arrays and report unsupported cross-version loads."""
        self.__dict__.update(state)
        self.coefficients.setflags(write=False)
        if self.Vp is not None:
            self.Vp.setflags(write=False)
        if self._fisher_factor is not None:
            self._fisher_factor.setflags(write=False)
        for _, _, _, values in self._fisher_transforms:
            values.setflags(write=False)
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
        _check_prediction_budgets(
            newdata,
            self._predict_spec.total_coefs,
            se_fit,
            self.Vp is not None,
            self._output_budget_bytes,
            self._memory_budget_bytes,
            matrix_output=False,
        )
        if se_fit and self.Vp is None and self._fisher_factor is None:
            raise RuntimeError(
                "Prediction uncertainty is unavailable for this "
                "prediction-only result. "
                "Fit with FitControl(uncertainty='fisher' or 'covariance')."
            )
        if self.Vp is not None:
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
        # The initial bounded adapter shares the exact PredictSpec encoding
        # and link/offset contract; term matvec actions can replace its matrix.
        X_p, eta = prepare_prediction(
            self._predict_spec,
            self.coefficients,
            newdata,
            pred_type=pred_type,
            offset=offset,
            offset_was_nonzero=self.offset_was_nonzero,
            warning_stacklevel=warning_stacklevel,
        )
        return finish_prediction(
            eta,
            X_p,
            self.family.link,
            None,
            pred_type=pred_type,
            se_fit=se_fit,
            fisher_factor=self._fisher_factor,
            fisher_transforms=self._fisher_transforms,
            fisher_scale=self._fisher_scale,
        )

    def predict_matrix(self, newdata: Data) -> npt.NDArray[np.floating]:
        """Build the constrained linear-predictor matrix for new data."""
        _check_prediction_budgets(
            newdata,
            self._predict_spec.total_coefs,
            False,
            self.Vp is not None,
            self._output_budget_bytes,
            self._memory_budget_bytes,
            matrix_output=True,
        )
        return self._predict_spec.build_predict_matrix(newdata)

    def predict_iter(
        self,
        source: RowSource,
        batch_rows: int,
        *,
        pred_type: str = "response",
        se_fit: bool = False,
    ) -> Iterator[
        tuple[
            npt.NDArray[np.intp],
            npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.floating]],
        ]
    ]:
        """Yield bounded predictions for a replayable :class:`RowSource`.

        Source offsets are applied positionally for every batch.  This adapter
        deliberately delegates matrix construction and response/SE finishing
        to the normal prediction core, preserving factor and link semantics.
        """
        if not isinstance(source, RowSource):
            raise TypeError("predict_iter requires a replayable RowSource.")
        if (
            isinstance(batch_rows, bool)
            or not isinstance(batch_rows, int)
            or batch_rows <= 0
        ):
            raise ValueError("batch_rows must be a positive integer.")
        if pred_type not in ("response", "link"):
            raise ValueError(
                f"pred_type must be 'response' or 'link', got {pred_type!r}"
            )
        if se_fit and self.Vp is None and self._fisher_factor is None:
            raise RuntimeError(
                "Prediction uncertainty is unavailable for this "
                "prediction-only result. "
                "Fit with FitControl(uncertainty='fisher' or 'covariance')."
            )
        return self._predict_iter(
            source, batch_rows, pred_type=pred_type, se_fit=se_fit
        )

    def _predict_iter(
        self,
        source: RowSource,
        batch_rows: int,
        *,
        pred_type: str,
        se_fit: bool,
    ) -> Iterator[
        tuple[
            npt.NDArray[np.intp],
            npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.floating]],
        ]
    ]:
        """Generator implementation after eager public validation."""
        source_has_offset = bool(getattr(source, "has_explicit_offset", True))
        warned_missing_offset = False
        for batch in source.scan(batch_rows):
            if not np.all(batch.valid):
                raise NotImplementedError(
                    "predict_iter does not support padded batches."
                )
            offset = batch.offset
            if not source_has_offset and not warned_missing_offset:
                offset = None
                warned_missing_offset = True
            yield (
                batch.row_positions,
                self._predict(
                    dict(batch.columns),
                    pred_type=pred_type,
                    se_fit=se_fit,
                    offset=offset,
                    warning_stacklevel=4,
                ),
            )


def _check_prediction_budgets(
    newdata: Data,
    p: int,
    se_fit: bool,
    has_covariance: bool,
    output_budget: int | None,
    memory_budget: int | None,
    *,
    matrix_output: bool,
) -> None:
    """Check returned output and known NumPy matrix workspace separately.

    This covers the shared dense ``PredictSpec`` adapter only. Smooth-native
    evaluators may allocate implementation-specific temporary buffers, which
    are outside this conservative host-array estimate.
    """
    if hasattr(newdata, "columns"):
        n_rows = len(newdata)
    else:
        values = list(newdata.values())
        n_rows = 0 if not values else len(values[0])
    itemsize = np.dtype(float).itemsize
    output_bytes = n_rows * (p if matrix_output else 2 if se_fit else 1) * itemsize
    if output_budget is not None and output_bytes > output_budget:
        raise MemoryError(
            f"Prediction output requires {output_bytes} bytes, exceeding output budget "
            f"{output_budget} bytes. Use predict_iter() with smaller batches."
        )
    # Point prediction retains X. Dense Vp SE additionally forms X@Vp and its
    # elementwise product (3 B*p arrays); factor SE keeps X, fitting X, solve
    # RHS, and squared solve output (4 B*p arrays).
    multiplier = 1 if not se_fit else 3 if has_covariance else 4
    workspace = multiplier * n_rows * p * itemsize
    if memory_budget is not None and workspace > memory_budget:
        raise MemoryError(
            f"Prediction workspace requires {workspace} bytes, exceeding memory budget "
            f"{memory_budget} bytes. Use predict_iter() with smaller batches."
        )
