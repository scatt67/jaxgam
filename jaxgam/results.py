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
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla

from jaxgam.control import FitControl
from jaxgam.inference._core import finish_prediction, predict_core
from jaxgam.inference.predictor import GAMPredictor
from jaxgam.jax_utils import to_numpy

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd

    from jaxgam.data.source import RowSource
    from jaxgam.families.base import ExponentialFamily
    from jaxgam.fitting.data import FittingData, PreparedFittingMetadata
    from jaxgam.fitting.newton import NewtonResult
    from jaxgam.fitting.state import StreamFitState
    from jaxgam.formula.design import ModelSetup, SmoothInfo
    from jaxgam.formula.predict_matrix import Data
    from jaxgam.formula.prepare import PreparedModel
    from jaxgam.formula.terms import FormulaSpec
    from jaxgam.smooths.constraints import CoefficientMap
    from jaxgam.summary.summary import GAMSummary


def _offset_was_nonzero(setup: ModelSetup) -> bool:
    """Whether fitting used an external offset that prediction cannot recover."""
    return setup.offset is not None and not np.allclose(setup.offset, 0.0)


def _transform_coefficients_cpu(
    fitting_data: FittingData, coefficients: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Map fitting coefficients to public coordinates without device transfer."""
    result = coefficients.copy()
    for block in fitting_data.penalty_structure.blocks:
        transform = block.transform
        if transform.kind == "identity":
            continue
        values = to_numpy(transform.values)
        local = result[block.start : block.stop]
        result[block.start : block.stop] = (
            values @ local if transform.kind == "dense" else values * local
        )
    return result


def _prepared_transform_coefficients_cpu(
    prepared, coefficients: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Map prepared fitting coordinates to public coordinates without rows."""
    from jaxgam.penalties.structure import DenseTransform, DiagonalTransform

    assert prepared.fitting is not None
    result = np.array(coefficients, copy=True)
    for block in prepared.fitting.penalty_structure.blocks:
        transform = block.transform
        local = result[block.start : block.stop]
        if isinstance(transform, DenseTransform):
            result[block.start : block.stop] = transform.matrix @ local
        elif isinstance(transform, DiagonalTransform):
            result[block.start : block.stop] = transform.diagonal * local
    return result


def _prepared_fisher_transforms(
    prepared,
) -> tuple[tuple[int, int, str, npt.NDArray[np.floating]], ...]:
    """Describe frozen local-D transforms for factor-based prediction SEs."""
    from jaxgam.penalties.structure import DenseTransform, DiagonalTransform

    assert prepared.fitting is not None
    transforms = []
    for block in prepared.fitting.penalty_structure.blocks:
        transform = block.transform
        if isinstance(transform, DenseTransform):
            transforms.append((block.start, block.stop, "dense", transform.matrix))
        elif isinstance(transform, DiagonalTransform):
            transforms.append((block.start, block.stop, "diagonal", transform.diagonal))
    return tuple(transforms)


def _prepared_transform_covariance_cpu(
    prepared, covariance: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Apply all local-D transforms, retaining public cross-block covariance."""
    result = np.array(covariance, copy=True)
    for start, stop, kind, values in _prepared_fisher_transforms(prepared):
        if kind == "dense":
            result[start:stop, :] = values @ result[start:stop, :]
            result[:, start:stop] = result[:, start:stop] @ values.T
        else:
            result[start:stop, :] *= values[:, None]
            result[:, start:stop] *= values[None, :]
    return result


def _transform_covariance_cpu(
    fitting_data: FittingData, covariance: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Apply each local transform on both covariance sides on the CPU."""
    result = covariance.copy()
    for block in fitting_data.penalty_structure.blocks:
        transform = block.transform
        if transform.kind == "identity":
            continue
        values = to_numpy(transform.values)
        start, stop = block.start, block.stop
        if transform.kind == "dense":
            result[start:stop, :] = values @ result[start:stop, :]
            result[:, start:stop] = result[:, start:stop] @ values.T
        else:
            result[start:stop, :] = values[:, None] * result[start:stop, :]
            result[:, start:stop] = result[:, start:stop] * values[None, :]
    return result


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
    execution_route: str = field(default="dense", kw_only=True)
    execution_fallback_reason: str | None = field(default=None, kw_only=True)
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
class GAMPredictionResult:
    """Small, picklable result surface for bounded new-data prediction.

    This deliberately has no training rows, setup, EDF vectors, or mandatory
    covariance.  It retains only scalar convergence diagnostics and a frozen
    :class:`GAMPredictor`.
    """

    _predictor: GAMPredictor
    deviance: float
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
    execution_route: str = field(default="dense", kw_only=True)
    execution_fallback_reason: str | None = field(default=None, kw_only=True)
    n: int
    _batch_rows: int = 65_536

    def __post_init__(self) -> None:
        object.__setattr__(self, "smoothing_params", np.array(self.smoothing_params))
        self.smoothing_params.setflags(write=False)

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.smoothing_params.setflags(write=False)

    @property
    def coefficients(self) -> np.ndarray:
        return self._predictor.coefficients

    @property
    def family(self) -> ExponentialFamily:
        return self._predictor.family

    @property
    def formula(self) -> str:
        return self._predictor.formula

    def predict(
        self,
        newdata: pd.DataFrame | dict,
        pred_type: str = "response",
        se_fit: bool = False,
        offset: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict a concrete new-data batch from compact retained state."""
        return self._predictor._predict(
            newdata,
            pred_type=pred_type,
            se_fit=se_fit,
            offset=offset,
            warning_stacklevel=4,
        )

    def predict_iter(
        self,
        source: RowSource,
        batch_rows: int | None = None,
        *,
        pred_type: str = "response",
        se_fit: bool = False,
    ) -> Iterator[tuple[np.ndarray, np.ndarray | tuple[np.ndarray, np.ndarray]]]:
        """Yield ordered bounded prediction batches from a replayable source."""
        return self._predictor.predict_iter(
            source,
            self._batch_rows if batch_rows is None else batch_rows,
            pred_type=pred_type,
            se_fit=se_fit,
        )

    def predict_matrix(self, newdata: Data) -> np.ndarray:
        """Build a budgeted constrained prediction matrix for a concrete batch."""
        return self._predictor.predict_matrix(newdata)

    def to_predictor(self) -> GAMPredictor:
        return self._predictor

    def materialize_covariance(self, memory_budget_bytes: int) -> np.ndarray:
        """Explicitly build public ``Vp`` only after a caller budgets it."""
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes <= 0
        ):
            raise ValueError("memory_budget_bytes must be a positive integer.")
        if self._predictor.Vp is not None:
            return self._predictor.Vp
        L = self._predictor._fisher_factor
        qr_factor = self._predictor._fisher_qr_factor
        if L is None and qr_factor is None:
            raise RuntimeError("No covariance provider was retained for this result.")
        p = L.shape[0] if L is not None else qr_factor.n_coef
        # The legacy lower-factor path retains three p-by-p work arrays.
        # QR covariance action additionally needs projection/permutation,
        # triangular, reconstruction, and output buffers at the same time.
        required_multiplier = 3 if L is not None else 12
        required = required_multiplier * p * p * np.dtype(float).itemsize
        if required > memory_budget_bytes:
            raise MemoryError(
                f"Vp materialization requires {required} workspace bytes, "
                "exceeding budget "
                f"{memory_budget_bytes} bytes."
            )
        if L is not None:
            Z = sla.solve_triangular(L, np.eye(p), lower=True)
            covariance = self._predictor._fisher_scale * (Z.T @ Z)
        else:
            covariance = self._predictor._fisher_scale * qr_factor.hessian_inverse(
                np.eye(p)
            )
        for start, stop, kind, values in self._predictor._fisher_transforms:
            if kind == "dense":
                covariance[start:stop, :] = values @ covariance[start:stop, :]
                covariance[:, start:stop] = covariance[:, start:stop] @ values.T
            elif kind == "diagonal":
                covariance[start:stop, :] *= values[:, None]
                covariance[:, start:stop] *= values[None, :]
        covariance.setflags(write=False)
        return covariance

    @classmethod
    def _from_stream_fit(
        cls,
        *,
        stream_state: StreamFitState,
        prepared: PreparedModel,
        metadata: PreparedFittingMetadata,
        family: ExponentialFamily,
        formula: str,
        method: str,
        control: FitControl,
        execution_route: str = "stream",
    ) -> GAMPredictionResult:
        """Build compact prediction state directly from row-free stream state."""
        from jaxgam.fitting.reml import (
            reml_criterion,
            reml_criterion_with_logdet_hessian,
        )
        from jaxgam.fitting.state import PivotedQRCoefficientFactor

        coefficients_fit = to_numpy(stream_state.coefficients)
        coefficients = _prepared_transform_coefficients_cpu(prepared, coefficients_fit)
        scale = float(to_numpy(stream_state.scale))
        phi = 1.0 if family.scale_known else scale
        if isinstance(stream_state.coefficient_factor, PivotedQRCoefficientFactor):
            score = reml_criterion_with_logdet_hessian(
                stream_state.log_lambda,
                stream_state.coefficients,
                stream_state.deviance,
                stream_state.saturated_loglik,
                metadata.penalty_structure,
                stream_state.score_scale,
                metadata.total_penalty_null_dim,
                metadata.singleton_sp_indices,
                metadata.singleton_ranks,
                metadata.singleton_eig_constants,
                metadata.multi_block_sp_indices,
                metadata.multi_block_ranks,
                metadata.multi_block_proj_S,
                stream_state.coefficient_factor.logdet_hessian(),
                metadata.rank_deficit,
            )
        else:
            # Preserve the legacy Cholesky/default arithmetic exactly.
            score = reml_criterion(
                stream_state.log_lambda,
                stream_state.xtwx,
                stream_state.coefficients,
                stream_state.deviance,
                stream_state.saturated_loglik,
                metadata.penalty_structure,
                stream_state.score_scale,
                metadata.total_penalty_null_dim,
                metadata.singleton_sp_indices,
                metadata.singleton_ranks,
                metadata.singleton_eig_constants,
                metadata.multi_block_sp_indices,
                metadata.multi_block_ranks,
                metadata.multi_block_proj_S,
                metadata.rank_deficit,
            )
        factor = None
        qr_factor = None
        transforms: tuple[tuple[int, int, str, np.ndarray], ...] = ()
        covariance = None
        if control.uncertainty != "none":
            if isinstance(
                stream_state.fisher_coefficient_factor, PivotedQRCoefficientFactor
            ):
                from jaxgam.inference.predictor import PivotedQRFisherFactor

                tagged = stream_state.fisher_coefficient_factor
                qr_factor = PivotedQRFisherFactor(
                    to_numpy(tagged.R),
                    to_numpy(tagged.pivots),
                    to_numpy(tagged.keep),
                    tagged.original_n_coef,
                )
            else:
                factor = to_numpy(stream_state.fisher_factor)
            transforms = _prepared_fisher_transforms(prepared)
            if control.uncertainty == "covariance":
                p = factor.shape[0] if factor is not None else qr_factor.n_coef
                # Keep the established lower-Cholesky contract exact. QR's
                # CPU action has several simultaneous p-by-p index/solve
                # buffers, so its compact covariance materialization needs a
                # deliberately conservative separate budget.
                required_multiplier = 3 if factor is not None else 12
                required = required_multiplier * p * p * np.dtype(float).itemsize
                if required > control.memory_budget_bytes:
                    raise MemoryError(
                        f"Vp requires {required} bytes, exceeding "
                        "FitControl.memory_budget_bytes="
                        f"{control.memory_budget_bytes}."
                    )
                if factor is not None:
                    Z = sla.solve_triangular(factor, np.eye(p), lower=True)
                    covariance_fit = phi * (Z.T @ Z)
                else:
                    covariance_fit = phi * qr_factor.hessian_inverse(np.eye(p))
                covariance = _prepared_transform_covariance_cpu(
                    prepared, covariance_fit
                )
                factor = None
                # Retain the matched QR root alongside requested Vp: forming
                # X @ Vp @ X.T loses its conditioning advantage for SEs.
                if qr_factor is None:
                    transforms = ()
        offset_reduction = prepared.fitting.response
        predictor = GAMPredictor(
            coefficients=coefficients,
            Vp=covariance,
            family=copy.deepcopy(family),
            formula=formula,
            offset_was_nonzero=not np.allclose(
                (offset_reduction.offset_min, offset_reduction.offset_max), 0.0
            ),
            _predict_spec=prepared.predict_spec,
            _fisher_factor=factor,
            _fisher_qr_factor=qr_factor,
            _fisher_transforms=transforms,
            _fisher_scale=phi,
            _output_budget_bytes=control.output_budget_bytes,
            _memory_budget_bytes=control.memory_budget_bytes,
        )
        return cls(
            _predictor=predictor,
            deviance=float(to_numpy(stream_state.deviance)),
            score=float(to_numpy(score)),
            scale=scale,
            theta=None,
            smoothing_params=np.exp(to_numpy(stream_state.log_lambda)),
            converged=stream_state.converged,
            n_iter=stream_state.n_iter,
            convergence_info=(
                "fixed sp streamed PIRLS"
                if stream_state.converged
                else "fixed sp streamed PIRLS did not converge"
            ),
            method=method,
            lambda_strategy="fixed",
            execution_path="jax",
            execution_route=execution_route,
            execution_fallback_reason=None,
            n=prepared.n_obs,
            _batch_rows=control.batch_rows,
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
        result_mode: Literal["full", "inference", "prediction"],
        control: FitControl | None = None,
        execution_route: str = "dense",
        execution_fallback_reason: str | None = None,
    ) -> GAMResults | GAMInferenceResult | GAMPredictionResult:
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
        control = FitControl() if control is None else control

        # Snapshot after Newton has synchronized any fitted family parameters
        # (notably NB theta) into the fitting family instance.
        family_snapshot = copy.deepcopy(family)

        # Phase 2→3: transfer to NumPy
        coefficients = to_numpy(pr.coefficients)
        scale = float(to_numpy(fit_result.scale))

        # This must precede every dense covariance/EDF and row-output transfer.
        # The optimizer already computed total Fisher EDF for unknown-scale
        # reporting; retain that scalar only, not diagnostic vectors.
        if result_mode == "prediction":
            coefficients = _transform_coefficients_cpu(fd, coefficients)
            phi = 1.0 if family_snapshot.scale_known else scale
            factor = None
            transforms: tuple[tuple[int, int, str, np.ndarray], ...] = ()
            covariance = None
            if control.uncertainty != "none":
                factor = to_numpy(pr.L_fisher)
                transforms = tuple(
                    (
                        block.start,
                        block.stop,
                        block.transform.kind,
                        to_numpy(block.transform.values),
                    )
                    for block in fd.penalty_structure.blocks
                    if block.transform.kind != "identity"
                )
                if control.uncertainty == "covariance":
                    p = factor.shape[0]
                    required = 3 * p * p * np.dtype(float).itemsize
                    if required > control.memory_budget_bytes:
                        raise MemoryError(
                            f"Vp requires {required} bytes, exceeding "
                            "FitControl.memory_budget_bytes="
                            f"{control.memory_budget_bytes}."
                        )
                    Z = sla.solve_triangular(factor, np.eye(p), lower=True)
                    covariance = phi * (Z.T @ Z)
                    covariance = _transform_covariance_cpu(fd, covariance)
                    # Dense covariance is the provider in this explicit mode;
                    # do not retain its factor or local transforms as well.
                    factor = None
                    transforms = ()
            predictor = GAMPredictor(
                coefficients=coefficients,
                Vp=covariance,
                family=family_snapshot,
                formula=formula,
                offset_was_nonzero=_offset_was_nonzero(setup),
                _predict_spec=setup._lazy_predict_spec(),
                _fisher_factor=factor,
                _fisher_transforms=transforms,
                _fisher_scale=phi,
                _output_budget_bytes=control.output_budget_bytes,
                _memory_budget_bytes=control.memory_budget_bytes,
            )
            return GAMPredictionResult(
                _predictor=predictor,
                deviance=float(to_numpy(pr.deviance)),
                score=float(to_numpy(fit_result.score)),
                scale=scale,
                theta=fit_result.theta,
                smoothing_params=to_numpy(fit_result.smoothing_params),
                converged=fit_result.converged,
                n_iter=fit_result.n_iter,
                convergence_info=fit_result.convergence_info,
                method=method,
                lambda_strategy=lambda_strategy,
                execution_path="jax",
                execution_route=execution_route,
                execution_fallback_reason=execution_fallback_reason,
                n=setup.n_obs,
                _batch_rows=control.batch_rows,
            )

        # Use Fisher-weighted quantities for EDF and Bayesian covariance.
        # For standard families Fisher = Newton; for extended families (NB)
        # these are recomputed post-convergence with expected weights
        # (R's gdi2, gdi.c:2262-2294, gam.fit4.r:564).
        L = to_numpy(pr.L_fisher)
        edf_total = float(to_numpy(fit_result.edf))
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

        # Phase 3 stays on CPU. Block transforms retain covariance cross-blocks
        # without moving a p-by-p matrix back to the accelerator.
        coefficients = _transform_coefficients_cpu(fd, coefficients)
        H_inv = _transform_covariance_cpu(fd, H_inv)

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
            "execution_route": execution_route,
            "execution_fallback_reason": execution_fallback_reason,
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
