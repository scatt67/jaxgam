"""CPU preparation for the frozen Gaussian QR score kernel.

R mapping: ``bam.r::qr_update`` supplies the weighted QR and residual tail;
``fast-REML.r::Sl.Xprep`` establishes the fitting-coordinate convention before
``Sl.fit`` attaches roots.  This module prepares the analogous immutable
arrays and deliberately does not optimize, route results, or select a rank map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import linalg

from jaxgam.execution.qr import reduce_positive_qr
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.gaussian import FrozenGaussianQR
from jaxgam.formula.rank_coordinates import IdentifiableCoordinateMap
from jaxgam.links.links import IdentityLink


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
