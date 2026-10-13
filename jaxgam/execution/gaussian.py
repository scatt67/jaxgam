"""CPU preparation for the frozen Gaussian QR score kernel.

R mapping: ``bam.r::qr_update`` supplies the weighted QR and residual tail;
``fast-REML.r::Sl.Xprep`` establishes the fitting-coordinate convention before
``Sl.fit`` attaches roots.  This module prepares the analogous immutable
arrays and deliberately does not optimize, route results, or select a rank map.
"""

from __future__ import annotations

import hashlib
import numbers
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import linalg

from jaxgam.execution.qr import reduce_positive_qr
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.gaussian import FrozenGaussianQR
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.fitting_prepare import qr_penalty_roots
from jaxgam.formula.rank_coordinates import IdentifiableCoordinateMap
from jaxgam.links.links import IdentityLink


class GaussianQRIneligible(ValueError):
    """The model is valid, but outside the reviewed Gaussian QR route."""


GAUSSIAN_LBFGS_PENALTY_THRESHOLD = 50
GAUSSIAN_LBFGS_MAXCOR = 10


@dataclass(frozen=True)
class GaussianQRWorkspaceEstimate:
    """Conservative ledger for known streamed Gaussian QR array storage.

    The ledger covers Python-visible NumPy/JAX arrays. SciPy LAPACK and XLA may
    allocate additional opaque native workspace, so this is a prospective
    rejection bound rather than a native allocator hard cap.
    """

    qr_design_bytes: int
    compression_vector_bytes: int
    compression_peak_bytes: int
    compact_host_device_bytes: int
    score_derivative_bytes: int
    outer_hessian_bytes: int
    lbfgs_workspace_bytes: int
    optimization_peak_bytes: int
    required_bytes: int


def _readonly(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_scalar(value: int | float, *, dtype: npt.DTypeLike) -> np.ndarray:
    """Own immutable scalar metadata without coercing count fields to float."""
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FrozenGaussianQRPreparation:
    """Immutable CPU inputs for the JIT score in global orthonormal coordinates."""

    compression: FrozenGaussianQR
    penalty_roots: tuple[npt.NDArray[np.float64], ...]
    coordinate_basis: npt.NDArray[np.float64]
    public_basis: npt.NDArray[np.float64]
    penalty_range_basis: npt.NDArray[np.float64]
    root_sp_indices: tuple[int, ...]
    n_smoothing_parameters: int
    penalty_nullity: int
    penalty_logdet_coordinate: str = "orthonormal_global"
    penalty_logdet_correction: float = 0.0
    source_scans: int = 0
    batches_scanned: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "penalty_roots", tuple(_readonly(x) for x in self.penalty_roots)
        )
        for name in ("coordinate_basis", "public_basis", "penalty_range_basis"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        if (
            self.penalty_logdet_coordinate != "orthonormal_global"
            or self.penalty_logdet_correction != 0.0
        ):
            raise ValueError(
                "Gaussian QR preparation requires an explicit zero-correction "
                "orthonormal pdet convention"
            )
        if self.source_scans < 0 or self.batches_scanned < 0:
            raise ValueError("Gaussian QR scan counters must be non-negative")


def _local_matrix(penalty: object) -> npt.NDArray[np.float64]:
    values, size, kind = (
        np.asarray(penalty.values, dtype=float),
        penalty.size,
        penalty.kind,
    )
    if kind == "dense":
        matrix = values
    elif kind == "diagonal":
        matrix = np.diag(values)
    elif kind == "identity":
        matrix = np.eye(size) * values
    else:
        raise ValueError("unsupported Gaussian QR local penalty")
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError("Gaussian QR local penalty must be finite and square")
    return (matrix + matrix.T) / 2.0


def _local_transform(transform: object) -> npt.NDArray[np.float64]:
    values, size, kind = (
        np.asarray(transform.values, dtype=float),
        transform.size,
        transform.kind,
    )
    if kind == "dense":
        matrix = values
    elif kind == "diagonal":
        matrix = np.diag(values)
    elif kind == "identity":
        matrix = np.eye(size)
    else:
        raise ValueError("unsupported Gaussian QR local transform")
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError("Gaussian QR local transform must be finite and square")
    return matrix


def _compact_root(matrix: npt.ArrayLike, rank: int) -> npt.NDArray[np.float64]:
    values, vectors = linalg.eigh(np.asarray(matrix), driver="evr")
    maximum = float(np.max(np.abs(values), initial=0.0))
    tolerance = maximum * 100.0 * np.finfo(float).eps
    if rank < 0 or rank > len(values) or np.min(values, initial=0.0) < -tolerance:
        raise ValueError("Gaussian QR local penalty rank/PSD metadata is invalid")
    order = np.argsort(values)[::-1]
    if rank and np.any(values[order[:rank]] <= 0.0):
        raise ValueError("Gaussian QR declared rank includes a null direction")
    if np.any(values[order[rank:]] > tolerance):
        raise ValueError("Gaussian QR declared rank omits an active direction")
    return np.sqrt(values[order[:rank]])[:, None] * vectors[:, order[:rank]].T


def _range_basis(
    roots: tuple[npt.NDArray[np.float64], ...], p: int, rank: int
) -> npt.NDArray[np.float64]:
    if not roots:
        if rank:
            raise ValueError("Gaussian QR roots do not span the declared penalty range")
        return np.empty((p, 0))
    stacked = np.vstack(roots)
    q, r, _ = linalg.qr(stacked.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    threshold = diagonal[0] * 100.0 * np.finfo(float).eps
    if (
        rank > min(stacked.shape)
        or (rank and diagonal[rank - 1] <= threshold)
        or (rank < len(diagonal) and diagonal[rank] > threshold)
    ):
        raise ValueError("Gaussian QR compact roots are rank-deficient")
    return q[:, :rank]


def prepare_frozen_gaussian_qr(
    fitting: FittingData, coordinate_map: IdentifiableCoordinateMap
) -> FrozenGaussianQRPreparation:
    """Prepare one frozen QR layout without changing the supplied rank map.

    ``fitting.X`` uses local-D coordinates.  The map's orthonormal global
    transform is bridged by block solves, so roots and data share the frozen
    global coordinate convention used by the pseudo-determinant.
    """
    if (
        fitting.family.family_name != "gaussian"
        or not isinstance(fitting.family.link, IdentityLink)
        or coordinate_map.n_coef != fitting.n_coef
    ):
        raise ValueError(
            "Gaussian QR preparation requires Gaussian identity-link FittingData/map"
        )
    global_transform = np.asarray(coordinate_map.fitting_to_public, dtype=float)
    if not np.allclose(
        global_transform.T @ global_transform,
        np.eye(fitting.n_coef),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Gaussian QR preparation requires an orthonormal global pdet convention"
        )
    X, y, weight = (
        np.asarray(fitting.X, float),
        np.asarray(fitting.y, float),
        np.asarray(fitting.wt, float),
    )
    offset = (
        np.zeros_like(y)
        if fitting.offset is None
        else np.asarray(fitting.offset, float)
    )
    if (
        X.shape != (fitting.n_obs, fitting.n_coef)
        or y.shape != (fitting.n_obs,)
        or weight.shape != (fitting.n_obs,)
        or offset.shape != (fitting.n_obs,)
        or not all(np.all(np.isfinite(x)) for x in (X, y, weight, offset))
        or np.any(weight < 1e-10)
        or np.any(weight > 1e10)
    ):
        raise ValueError(
            "Gaussian QR preparation currently requires finite unclipped "
            "positive weights"
        )
    bridge = np.array(global_transform, copy=True)
    for block in fitting.penalty_structure.blocks:
        bridge[block.start : block.stop] = linalg.solve(
            _local_transform(block.transform),
            bridge[block.start : block.stop],
            assume_a="gen",
        )
    # QR the original weighted data first, then apply the small p-by-p bridge
    # to its unpivoted factor. This retains the data residual tail without a
    # second n-by-p transformed design allocation.
    data_state = reduce_positive_qr(
        ((np.sqrt(weight)[:, None] * X, np.sqrt(weight) * (y - offset)),),
        n_coef=fitting.n_coef,
    )
    data_factor = np.empty_like(data_state.R)
    data_factor[:, data_state.pivots] = data_state.R
    global_state = reduce_positive_qr(
        ((data_factor @ bridge, data_state.qtz),), n_coef=fitting.n_coef
    )
    R = np.empty_like(global_state.R)
    R[:, global_state.pivots] = global_state.R
    compression = FrozenGaussianQR(
        _readonly(R),
        _readonly(global_state.qtz),
        _readonly_scalar(
            data_state.residual_tail + global_state.residual_tail, dtype=np.float64
        ),
        _readonly_scalar(fitting.n_obs, dtype=np.int64),
        _readonly_scalar(fitting.n_obs, dtype=np.int64),
        _readonly_scalar(np.log(weight).sum(), dtype=np.float64),
    )
    ranks = iter(fitting.penalty_ranks)
    roots, indices = [], []
    for block in fitting.penalty_structure.blocks:
        for sp, penalty in zip(block.sp_indices, block.penalties, strict=True):
            try:
                rank = next(ranks)
            except StopIteration as error:
                raise ValueError(
                    "Gaussian QR penalty rank metadata is incomplete"
                ) from error
            root = _compact_root(_local_matrix(penalty), rank)
            if len(root):
                roots.append(root @ bridge[block.start : block.stop])
                indices.append(sp)
    if next(ranks, None) is not None:
        raise ValueError("Gaussian QR penalty rank metadata is excessive")
    compact = tuple(roots)
    return FrozenGaussianQRPreparation(
        compression,
        compact,
        coordinate_map.basis,
        coordinate_map.public_basis,
        _range_basis(compact, fitting.n_coef, fitting.total_penalty_rank),
        tuple(indices),
        fitting.n_penalties,
        fitting.total_penalty_null_dim,
    )


def prepare_streamed_frozen_gaussian_qr(
    stream: StreamDesign, batch_rows: int
) -> FrozenGaussianQRPreparation:
    """Compress one prepared RowSource scan into immutable Gaussian QR state.

    Prepared setup may already have replayed the source to freeze knots,
    constraints, and local penalty coordinates.  This function owns exactly
    one additional replay for the weighted Gaussian sufficient statistics.
    Later smoothing trials consume only the returned coefficient-size arrays.
    """
    if (
        not isinstance(batch_rows, numbers.Integral)
        or isinstance(batch_rows, bool)
        or batch_rows <= 0
    ):
        raise ValueError("batch_rows must be a positive integer")
    prepared = stream.prepared
    fitting = prepared.fitting
    if fitting is None:
        raise ValueError("Gaussian QR RowSource preparation requires fitting metadata")
    if fitting.family_name != "gaussian" or fitting.link_name != "IdentityLink":
        raise GaussianQRIneligible(
            "Gaussian QR requires the exact Gaussian family with identity link"
        )
    if fitting.unpenalized_rank_deficit:
        raise GaussianQRIneligible(
            "Gaussian QR currently requires a full identifiable coefficient map"
        )
    start_fingerprint = stream.source.fingerprint()
    if start_fingerprint != prepared.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again.")

    p = prepared.n_coef
    public_from_fitting = np.eye(p)
    for block in fitting.penalty_structure.blocks:
        public_from_fitting[block.start : block.stop, block.start : block.stop] = (
            block.transform.dense()
        )
    fitting_from_public = linalg.solve(public_from_fitting, np.eye(p), assume_a="gen")

    n_obs = 0
    n_positive_weights = 0
    sum_log_positive_weights = 0.0
    batches_scanned = 0
    row_positions_digest = hashlib.sha256(b"jaxgam-row-positions-v1\0")

    def weighted_batches():
        nonlocal n_obs, n_positive_weights, sum_log_positive_weights, batches_scanned
        for batch in stream.source.scan(int(batch_rows)):
            positions = np.asarray(batch.row_positions)
            batch_n = len(positions)
            if batch_n > batch_rows:
                raise ValueError(
                    "RowSource yielded a batch larger than the requested batch_rows"
                )
            if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
                raise ValueError(
                    "RowSource positions must be a one-dimensional integer array"
                )
            row_positions_digest.update(
                np.ascontiguousarray(positions, dtype=np.int64).view(np.uint8)
            )
            if batch.y is None:
                raise ValueError(
                    "Gaussian QR RowSource preparation requires response values"
                )
            valid = np.asarray(batch.valid, dtype=bool)
            y = np.asarray(batch.y, dtype=float)
            weight = np.asarray(batch.weight, dtype=float)
            offset = np.asarray(batch.offset, dtype=float)
            if any(values.shape != (batch_n,) for values in (valid, y, weight, offset)):
                raise ValueError(
                    "Gaussian QR RowSource batches require matching row vectors"
                )
            if not np.all(valid):
                raise NotImplementedError(
                    "Gaussian QR RowSource preparation does not support padding"
                )
            X = np.asarray(prepared.evaluate_fitting_batch(batch), dtype=float)
            if X.shape != (batch_n, p) or not all(
                np.all(np.isfinite(x)) for x in (X, y, weight, offset)
            ):
                raise ValueError(
                    "Gaussian QR RowSource batches require finite matching arrays"
                )
            if np.any(weight < 1e-10) or np.any(weight > 1e10):
                raise GaussianQRIneligible(
                    "Gaussian QR currently requires unclipped finite positive weights"
                )
            square_root_weight = np.sqrt(weight)
            n_obs += batch_n
            n_positive_weights += int(np.sum(weight > 0.0))
            sum_log_positive_weights += float(np.sum(np.log(weight)))
            batches_scanned += 1
            yield square_root_weight[:, None] * X, square_root_weight * (y - offset)

    data_state = reduce_positive_qr(weighted_batches(), n_coef=p)
    if n_obs != prepared.n_obs:
        raise RuntimeError(
            "RowSource scan changed its valid row count after preparation; "
            "prepare again."
        )
    if row_positions_digest.hexdigest() != prepared.row_positions_fingerprint:
        raise RuntimeError(
            "RowSource scan changed row positions after preparation; prepare again."
        )
    if stream.source.fingerprint() != start_fingerprint:
        raise RuntimeError("RowSource changed during Gaussian QR preparation.")

    data_factor = np.empty_like(data_state.R)
    data_factor[:, data_state.pivots] = data_state.R
    global_state = reduce_positive_qr(
        ((data_factor @ fitting_from_public, data_state.qtz),), n_coef=p
    )
    R = np.empty_like(global_state.R)
    R[:, global_state.pivots] = global_state.R
    compression = FrozenGaussianQR(
        _readonly(R),
        _readonly(global_state.qtz),
        _readonly_scalar(
            data_state.residual_tail + global_state.residual_tail, dtype=np.float64
        ),
        _readonly_scalar(n_obs, dtype=np.int64),
        _readonly_scalar(n_positive_weights, dtype=np.int64),
        _readonly_scalar(sum_log_positive_weights, dtype=np.float64),
    )

    roots: list[npt.NDArray[np.float64]] = []
    indices: list[int] = []
    for local_root in qr_penalty_roots(fitting.penalty_structure):
        if len(local_root.root):
            roots.append(
                np.asarray(local_root.root, dtype=float)
                @ fitting_from_public[local_root.start : local_root.stop]
            )
            indices.append(local_root.sp_index)
    compact = tuple(roots)
    identity = np.eye(p)
    return FrozenGaussianQRPreparation(
        compression=compression,
        penalty_roots=compact,
        coordinate_basis=identity,
        public_basis=identity,
        penalty_range_basis=_range_basis(compact, p, fitting.total_penalty_rank),
        root_sp_indices=tuple(indices),
        n_smoothing_parameters=fitting.penalty_structure.n_penalties,
        penalty_nullity=fitting.total_penalty_null_dim,
        source_scans=1,
        batches_scanned=batches_scanned,
    )


def estimate_streamed_gaussian_qr_workspace(
    prepared: object, batch_rows: int, *, pin_lambda: bool = False
) -> GaussianQRWorkspaceEstimate:
    """Estimate known peak arrays before QR compression or device transfer."""
    if (
        not isinstance(batch_rows, numbers.Integral)
        or isinstance(batch_rows, bool)
        or batch_rows <= 0
    ):
        raise ValueError("batch_rows must be a positive integer")
    fitting = prepared.fitting
    if fitting is None:
        raise ValueError("Gaussian QR workspace estimation requires fitting metadata")
    p = prepared.n_coef
    m = fitting.penalty_structure.n_penalties
    rank = fitting.total_penalty_rank
    root_rows = sum(sum(block.ranks) for block in fitting.penalty_structure.blocks)
    root_entries = root_rows * p
    rows = min(int(batch_rows), prepared.n_obs)
    itemsize = np.dtype(np.float64).itemsize

    # At the second QR merge these four arrays coexist: fitting X, weighted X,
    # [old R; weighted X], and SciPy's raw packed QR copy. The two stacked
    # arrays each contribute p additional rows.
    qr_design_bytes = (4 * rows * p + 2 * p * p) * itemsize
    # Retain old R and its unpivoted copy, plus response/weight/offset,
    # positions, masks, weighted/stacked response, and reflector RHS vectors.
    compression_vector_bytes = (2 * p * p + 12 * rows + 8 * p) * itemsize + rows
    compression_peak_bytes = qr_design_bytes + compression_vector_bytes

    # Host preparation and its device mirror retain R/qtz, two coordinate
    # bases, the penalty-range basis, compact global roots, and two fitting /
    # public transforms.
    compact_host_device_bytes = (
        8 * p * p + 2 * p * rank + 2 * root_entries + 2 * p
    ) * itemsize
    # Conservative explicit-array allowance for a score/value-gradient trial:
    # scaled/augmented roots and QR factors, coefficient systems, result
    # factors, reverse-mode primals/cotangents, and parameter vectors.
    score_derivative_bytes = (
        16 * p * p + 8 * root_entries + 16 * (p + rank + root_rows + m + 1)
    ) * itemsize
    n_outer = m + 1
    if m > GAUSSIAN_LBFGS_PENALTY_THRESHOLD:
        # SciPy's L-BFGS-B work arrays retain 2*m correction vectors plus its
        # documented vector/scalar workspace. Fixed rho is removed from the
        # free vector, leaving only unknown log(phi).
        n_free = 1 if pin_lambda else n_outer
        maxcor = GAUSSIAN_LBFGS_MAXCOR
        lbfgs_workspace_bytes = (
            2 * maxcor * n_free + 5 * n_free + 11 * maxcor * maxcor + 8 * maxcor
        ) * itemsize + 3 * n_free * np.dtype(np.int32).itemsize
        outer_hessian_bytes = 0
    else:
        # Newton returns the full unknown-scale outer Hessian to Python.
        outer_hessian_bytes = n_outer * n_outer * itemsize
        lbfgs_workspace_bytes = 0
    optimization_peak_bytes = (
        compact_host_device_bytes
        + score_derivative_bytes
        + outer_hessian_bytes
        + lbfgs_workspace_bytes
    )
    required_bytes = max(compression_peak_bytes, optimization_peak_bytes)
    return GaussianQRWorkspaceEstimate(
        qr_design_bytes=qr_design_bytes,
        compression_vector_bytes=compression_vector_bytes,
        compression_peak_bytes=compression_peak_bytes,
        compact_host_device_bytes=compact_host_device_bytes,
        score_derivative_bytes=score_derivative_bytes,
        outer_hessian_bytes=outer_hessian_bytes,
        lbfgs_workspace_bytes=lbfgs_workspace_bytes,
        optimization_peak_bytes=optimization_peak_bytes,
        required_bytes=required_bytes,
    )


def preflight_streamed_gaussian_qr_workspace(
    prepared: object,
    batch_rows: int,
    memory_budget_bytes: int,
    *,
    pin_lambda: bool = False,
) -> GaussianQRWorkspaceEstimate:
    """Reject an over-budget Gaussian QR route before compression/device copies."""
    if (
        not isinstance(memory_budget_bytes, numbers.Integral)
        or isinstance(memory_budget_bytes, bool)
        or memory_budget_bytes <= 0
    ):
        raise ValueError("memory_budget_bytes must be a positive integer")
    estimate = estimate_streamed_gaussian_qr_workspace(
        prepared, batch_rows, pin_lambda=pin_lambda
    )
    if estimate.required_bytes > memory_budget_bytes:
        raise MemoryError(
            "Known streamed Gaussian QR workspace requires "
            f"{estimate.required_bytes} bytes, exceeding "
            f"FitControl.memory_budget_bytes={memory_budget_bytes} "
            f"(compression={estimate.compression_peak_bytes}, compact host/device="
            f"{estimate.compact_host_device_bytes}, score/derivative="
            f"{estimate.score_derivative_bytes}, outer Hessian="
            f"{estimate.outer_hessian_bytes}, L-BFGS-B="
            f"{estimate.lbfgs_workspace_bytes}). SciPy LAPACK and XLA may allocate "
            "additional native workspace that this prospective check cannot hard-cap."
        )
    return estimate
