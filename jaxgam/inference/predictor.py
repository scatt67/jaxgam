"""Frozen, picklable prediction core for a fitted GAM."""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla

import jaxgam
from jaxgam.data.continuous_request import FrozenRequestGrid, RequestCompressionSpec
from jaxgam.data.source import RowSource
from jaxgam.formula.predict_matrix import PredictSpec, _to_dict
from jaxgam.inference._core import finish_prediction, predict_core, prepare_prediction

if TYPE_CHECKING:
    from jaxgam.families.base import ExponentialFamily
    from jaxgam.formula.predict_matrix import Data
    from jaxgam.inference.operator_geometry import OperatorPredictionGeometry


@dataclass(frozen=True)
class PivotedQRFisherFactor:
    """CPU-owned QR covariance action with ``H = (R P.T).T @ (R P.T)``.

    This is deliberately a Phase-3 NumPy object rather than a JAX fitting
    factor.  It is compact (one triangular root and index maps), picklable,
    and applies all coefficient actions on axis zero.
    """

    R: npt.NDArray[np.floating]
    pivots: npt.NDArray[np.integer]
    keep: npt.NDArray[np.integer]
    original_n_coef: int

    def __post_init__(self) -> None:
        raw_pivots = np.asarray(self.pivots)
        raw_keep = np.asarray(self.keep)
        if (
            raw_pivots.dtype.kind not in "iu"
            or raw_keep.dtype.kind not in "iu"
            or raw_pivots.dtype.kind == "b"
            or raw_keep.dtype.kind == "b"
            or isinstance(self.original_n_coef, bool)
            or not isinstance(self.original_n_coef, (int, np.integer))
        ):
            raise ValueError("QR Fisher index maps and dimension must be integers")
        R = np.array(self.R, dtype=float, copy=True)
        pivots = np.array(self.pivots, dtype=np.intp, copy=True)
        keep = np.array(self.keep, dtype=np.intp, copy=True)
        if (
            R.ndim != 2
            or self.original_n_coef < 0
            or (R.ndim == 2 and R.shape[1] != R.shape[0])
        ):
            raise ValueError("QR Fisher root must be square with a valid dimension")
        rank = R.shape[0]
        if (
            pivots.shape != (rank,)
            or keep.shape != (rank,)
            or self.original_n_coef < rank
            or np.any(pivots < 0)
            or np.any(pivots >= rank)
            or len(np.unique(pivots)) != rank
            or np.any(keep < 0)
            or np.any(keep >= self.original_n_coef)
            or len(np.unique(keep)) != rank
            or not np.all(np.isfinite(R))
            or not np.array_equal(R, np.triu(R))
            or np.any(np.diag(R) == 0.0)
        ):
            raise ValueError("invalid compact pivoted QR Fisher factor")
        for value in (R, pivots, keep):
            value.setflags(write=False)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "pivots", pivots)
        object.__setattr__(self, "keep", keep)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore owned read-only leaves after NumPy's pickle reconstruction."""
        self.__dict__.update(state)
        self.__post_init__()

    @property
    def n_coef(self) -> int:
        """Original fitting-coordinate dimension."""
        return self.original_n_coef

    def _project(self, rhs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        value = np.asarray(rhs, dtype=float)
        if value.ndim not in (1, 2) or value.shape[0] != self.n_coef:
            raise ValueError("QR Fisher RHS must use coefficient-first coordinates")
        return value[self.keep, ...]

    def _reconstruct(
        self, reduced: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        value = np.asarray(reduced, dtype=float)
        if value.ndim not in (1, 2) or value.shape[0] != self.R.shape[0]:
            raise ValueError("QR Fisher reduced RHS has the wrong rank")
        result = np.zeros((self.n_coef, *value.shape[1:]), dtype=value.dtype)
        result[self.keep, ...] = value
        return result

    def root_transpose_inverse(
        self, rhs: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Apply ``B^-T = R^-T P.T`` to full-coordinate RHS columns."""
        return sla.solve_triangular(
            self.R, self._project(rhs)[self.pivots, ...], lower=False, trans="T"
        )

    def hessian_inverse(
        self, rhs: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Apply the compact QR ``H^-1`` action in original coordinates."""
        rows = self.root_transpose_inverse(rhs)
        ordered = sla.solve_triangular(self.R, rows, lower=False)
        reduced = np.zeros_like(ordered)
        reduced[self.pivots, ...] = ordered
        return self._reconstruct(reduced)


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
    _fisher_qr_factor: PivotedQRFisherFactor | None = None
    _fisher_transforms: tuple[tuple[int, int, str, npt.NDArray[np.floating]], ...] = ()
    _fisher_scale: float = 1.0
    _output_budget_bytes: int | None = None
    _memory_budget_bytes: int | None = None
    _prediction_policy: str | None = None
    _request_compression: RequestCompressionSpec | None = None
    _operator_geometry: OperatorPredictionGeometry | None = None
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
        if self._fisher_factor is not None and self._fisher_qr_factor is not None:
            raise ValueError("only one compact Fisher provider may be retained")
        if self._fisher_qr_factor is not None:
            if not isinstance(self._fisher_qr_factor, PivotedQRFisherFactor):
                raise ValueError("compact QR Fisher provider has the wrong type")
            qr = self._fisher_qr_factor
            object.__setattr__(
                self,
                "_fisher_qr_factor",
                PivotedQRFisherFactor(qr.R, qr.pivots, qr.keep, qr.original_n_coef),
            )
        transforms = []
        for start, stop, kind, values in self._fisher_transforms:
            owned = np.array(values)
            owned.setflags(write=False)
            transforms.append((start, stop, kind, owned))
        object.__setattr__(self, "_fisher_transforms", tuple(transforms))
        compressed = self._request_compression is not None
        if compressed and not isinstance(
            self._request_compression, RequestCompressionSpec
        ):
            raise ValueError("request compression must be a RequestCompressionSpec")
        if compressed != (self._prediction_policy == "mgcv_predict_bamd"):
            raise ValueError(
                "request compression requires prediction_policy 'mgcv_predict_bamd'"
                " and that policy requires request compression"
            )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore frozen arrays and report unsupported cross-version loads."""
        state.setdefault("_fisher_qr_factor", None)
        state.setdefault("_prediction_policy", None)
        state.setdefault("_request_compression", None)
        state.setdefault("_operator_geometry", None)
        self.__dict__.update(state)
        self.coefficients.setflags(write=False)
        if self.Vp is not None:
            self.Vp.setflags(write=False)
        if self._fisher_factor is not None:
            self._fisher_factor.setflags(write=False)
        if self._fisher_qr_factor is not None:
            qr = self._fisher_qr_factor
            object.__setattr__(
                self,
                "_fisher_qr_factor",
                PivotedQRFisherFactor(qr.R, qr.pivots, qr.keep, qr.original_n_coef),
            )
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

    @property
    def prediction_policy(self) -> str | None:
        """The effective discrete prediction policy (``None`` off that route).

        ``"mgcv_predict_bamd"`` compresses each whole request first;
        ``"continuous_frozen_basis"`` evaluates every value exactly.
        """
        return self._prediction_policy

    def freeze_request(self, newdata: Data) -> FrozenRequestGrid | None:
        """Freeze ``predict.bamd`` compression over one whole request.

        Returns ``None`` when this predictor evaluates new values exactly.
        """
        if self._request_compression is None:
            return None
        return self._request_compression.freeze(_to_dict(newdata))

    def _compressed(
        self, newdata: Data, request_grid: FrozenRequestGrid | None
    ) -> Data:
        """Apply request-level compression when the policy requires it."""
        if self._request_compression is None:
            return newdata
        # Use the prediction encoder's own column conversion, so compressed
        # and exact requests reach the frozen basis with identical types.
        columns = _to_dict(newdata)
        if request_grid is None:
            request_grid = self._request_compression.freeze(columns)
        return request_grid.apply(columns)

    def _predict(
        self,
        newdata: Data,
        *,
        pred_type: str,
        se_fit: bool,
        offset: npt.ArrayLike | None,
        warning_stacklevel: int,
        request_grid: FrozenRequestGrid | None = None,
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.floating]]:
        """Predict with an explicit warning depth for delegating wrappers.

        Under ``mgcv_predict_bamd`` the request (``newdata`` itself unless a
        frozen ``request_grid`` of a larger request is supplied) is compressed
        before the frozen basis is evaluated.
        """
        # Reject an invalid request before any whole-request pre-pass.
        _check_pred_type(pred_type)
        _check_prediction_budgets(
            newdata,
            len(self.coefficients),
            se_fit,
            self.Vp is not None,
            self._output_budget_bytes,
            self._memory_budget_bytes,
            matrix_output=False,
            has_qr_fisher=self._fisher_qr_factor is not None,
            geometry=self._operator_geometry,
        )
        if (
            se_fit
            and self.Vp is None
            and self._fisher_factor is None
            and self._fisher_qr_factor is None
        ):
            raise RuntimeError(self._no_uncertainty_message())
        newdata = self._compressed(newdata, request_grid)
        if self.Vp is not None and self._fisher_qr_factor is None:
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
            geometry=self._operator_geometry,
        )
        return finish_prediction(
            eta,
            X_p,
            self.family.link,
            None,
            pred_type=pred_type,
            se_fit=se_fit,
            fisher_factor=self._fisher_factor,
            fisher_qr_factor=self._fisher_qr_factor,
            fisher_transforms=self._fisher_transforms,
            fisher_scale=self._fisher_scale,
        )

    def predict_matrix(self, newdata: Data) -> npt.NDArray[np.floating]:
        """Build the constrained linear-predictor matrix for new data.

        Under ``mgcv_predict_bamd`` this is the matrix of the compressed
        request, like ``predict.bamd(type="lpmatrix")``.
        """
        _check_prediction_budgets(
            newdata,
            len(self.coefficients),
            False,
            self.Vp is not None,
            self._output_budget_bytes,
            self._memory_budget_bytes,
            matrix_output=True,
            has_qr_fisher=False,
            geometry=self._operator_geometry,
        )
        if self._operator_geometry is not None:
            return self._operator_geometry.build_predict_matrix(newdata)
        return self._predict_spec.build_predict_matrix(self._compressed(newdata, None))

    def _no_uncertainty_message(self) -> str:
        """Why ``se_fit=True`` cannot be served, and what to do instead."""
        if self._operator_geometry is not None:
            return (
                "Prediction standard errors are not available on the operator "
                "route. Use bulk_link_variance() on the fit result for "
                "approximate link-scale variances."
            )
        return (
            "Prediction uncertainty is unavailable for this "
            "prediction-only result. "
            "Fit with FitControl(uncertainty='fisher' or 'covariance')."
        )

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

        Under ``mgcv_predict_bamd`` the whole source is the prediction
        request: a bounded pre-pass scans it here, before the iterator is
        returned, and freezes ``predict.bamd``'s compression grid.  Every
        yielded batch is compressed with that frozen grid, so results do not
        depend on ``batch_rows``.  The source must not change afterwards.
        """
        if not isinstance(source, RowSource):
            raise TypeError("predict_iter requires a replayable RowSource.")
        if (
            isinstance(batch_rows, bool)
            or not isinstance(batch_rows, int)
            or batch_rows <= 0
        ):
            raise ValueError("batch_rows must be a positive integer.")
        _check_pred_type(pred_type)
        if (
            se_fit
            and self.Vp is None
            and self._fisher_factor is None
            and self._fisher_qr_factor is None
        ):
            raise RuntimeError(self._no_uncertainty_message())
        request_grid = (
            None
            if self._request_compression is None
            else self._request_compression.freeze_source(source, batch_rows)
        )
        return self._predict_iter(
            source,
            batch_rows,
            pred_type=pred_type,
            se_fit=se_fit,
            request_grid=request_grid,
        )

    def _predict_iter(
        self,
        source: RowSource,
        batch_rows: int,
        *,
        pred_type: str,
        se_fit: bool,
        request_grid: FrozenRequestGrid | None = None,
    ) -> Iterator[
        tuple[
            npt.NDArray[np.intp],
            npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.floating]],
        ]
    ]:
        """Generator implementation after eager public validation."""
        source_has_offset = bool(getattr(source, "has_explicit_offset", True))
        warned_missing_offset = False
        _check_request_source(source, request_grid)
        expected = source.n_rows
        if (
            isinstance(expected, bool)
            or not isinstance(expected, numbers.Integral)
            or expected < 0
        ):
            raise ValueError(
                "Prediction source row count must be a nonnegative integer."
            )
        fingerprint = source.fingerprint()
        seen = 0
        for batch in source.scan(batch_rows):
            positions = np.asarray(batch.row_positions)
            n = len(positions)
            if (
                positions.ndim != 1
                or positions.dtype.kind not in "iu"
                or np.any(positions < 0)
                or np.any(positions > np.iinfo(np.int64).max)
            ):
                raise ValueError("Prediction positions must be nonnegative integers.")
            if n > batch_rows or seen + n > expected:
                raise RuntimeError("Prediction source exceeded its declared row bound.")
            if not np.all(batch.valid):
                raise NotImplementedError(
                    "predict_iter does not support padded batches."
                )
            if n == 0:
                continue
            seen += n
            offset = batch.offset
            if not source_has_offset and not warned_missing_offset:
                offset = None
                warned_missing_offset = True
            yield (
                positions.copy(),
                self._predict(
                    (
                        dict(batch.columns)
                        if self._operator_geometry is None
                        else self._operator_geometry.slice_columns(batch.columns, 0, n)
                    ),
                    pred_type=pred_type,
                    se_fit=se_fit,
                    offset=offset,
                    warning_stacklevel=4,
                    request_grid=request_grid,
                ),
            )
        _check_request_source(source, request_grid)
        if seen != expected:
            raise RuntimeError("Prediction source yielded fewer rows than declared.")
        actual = source.n_rows
        if (
            isinstance(actual, bool)
            or not isinstance(actual, numbers.Integral)
            or actual != expected
            or source.fingerprint() != fingerprint
        ):
            raise RuntimeError("Prediction source changed during replay.")


def _check_pred_type(pred_type: str) -> None:
    if pred_type not in ("response", "link"):
        raise ValueError(f"pred_type must be 'response' or 'link', got {pred_type!r}")


def _check_request_source(
    source: RowSource, request_grid: FrozenRequestGrid | None
) -> None:
    """A frozen request grid is valid only for the unchanged source."""
    if request_grid is None or request_grid.source_fingerprint is None:
        return
    if source.fingerprint() != request_grid.source_fingerprint:
        raise RuntimeError(
            "RowSource changed after its prediction request compression was "
            "frozen; restart predict_iter() to freeze the changed request."
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
    has_qr_fisher: bool,
    geometry: OperatorPredictionGeometry | None = None,
) -> None:
    """Check returned output and known NumPy matrix workspace separately.

    This covers the shared dense ``PredictSpec`` adapter only. Smooth-native
    evaluators may allocate implementation-specific temporary buffers, which
    are outside this conservative host-array estimate.
    """
    if hasattr(newdata, "columns"):
        n_rows = len(newdata)
    else:
        n_rows = len(next(iter(newdata.values()), ()))
    itemsize = np.dtype(float).itemsize
    output_bytes = n_rows * (p if matrix_output else 2 if se_fit else 1) * itemsize
    if output_budget is not None and output_bytes > output_budget:
        raise MemoryError(
            f"Prediction output requires {output_bytes} bytes, exceeding output budget "
            f"{output_budget} bytes. Use predict_iter() with smaller batches."
        )
    # Point prediction retains X. Dense Vp SE additionally forms X@Vp and its
    # elementwise product (3 B*p arrays); lower-factor SE keeps X, fitting X,
    # solve RHS, and squared solve output (4 B*p arrays). Pivoted QR makes
    # projection, pivoting, solve, and square work arrays; bound each by p
    # even where its retained rank is smaller (7 B*p arrays).
    multiplier = 1 if not se_fit else 7 if has_qr_fisher else 3 if has_covariance else 4
    workspace = multiplier * n_rows * p * itemsize
    if geometry is not None:
        workspace = geometry.workspace_bound(n_rows, matrix_output=matrix_output)
    if memory_budget is not None and workspace > memory_budget:
        raise MemoryError(
            f"Prediction workspace requires {workspace} bytes, exceeding memory budget "
            f"{memory_budget} bytes. Use predict_iter() with smaller batches."
        )
