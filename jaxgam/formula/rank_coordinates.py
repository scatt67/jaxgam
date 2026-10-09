"""CPU-only global rank-coordinate construction for Gaussian QR fitting.

Pinned mgcv's ``gdiPK`` selects an identifiable coordinate minor only after
moving public constrained penalties into the global ``U1 %*% Qs`` basis.
This module contains that NumPy/SciPy setup operation.  It deliberately has
no JAX dependency; the corresponding device solve lives in
``jaxgam.fitting.rank_coordinates``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import linalg

from jaxgam.formula.fitting_prepare import (
    GlobalPenaltyCoordinates,
    global_penalty_coordinates,
)
from jaxgam.penalties.structure import PenaltyStructure


def _readonly(value: npt.ArrayLike, *, dtype: type[np.generic]) -> np.ndarray:
    """Copy small CPU map metadata and prevent accidental mutation."""
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class IdentifiableCoordinateMap:
    """A full-rank reduced coordinate basis and its public reconstruction.

    ``basis`` maps reduced coordinates into the selected global fitting
    coordinates. ``fitting_to_public`` maps those coordinates back to public
    constrained coefficients. Every solve must use their product for both
    coefficient and covariance reconstruction.
    """

    basis: npt.NDArray[np.float64]
    fitting_to_public: npt.NDArray[np.float64]
    keep: npt.NDArray[np.int64]
    drop: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=float)
        transform = np.asarray(self.fitting_to_public, dtype=float)
        keep = np.asarray(self.keep)
        drop = np.asarray(self.drop)
        if basis.ndim != 2 or basis.shape[0] == 0 or basis.shape[1] == 0:
            raise ValueError("rank-coordinate basis must be a non-empty matrix")
        if transform.shape != (basis.shape[0], basis.shape[0]):
            raise ValueError("rank-coordinate public transform has incompatible shape")
        if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(transform)):
            raise ValueError("rank-coordinate map must be finite")
        if np.linalg.matrix_rank(basis) != basis.shape[1]:
            raise ValueError("rank-coordinate basis must have full column rank")
        if (
            keep.ndim != 1
            or drop.ndim != 1
            or keep.dtype.kind not in "iu"
            or drop.dtype.kind not in "iu"
        ):
            raise TypeError("rank-coordinate keep/drop indices must be integer vectors")
        all_indices = np.concatenate((keep, drop))
        p = basis.shape[0]
        if (
            len(np.unique(all_indices)) != p
            or not np.array_equal(np.sort(all_indices), np.arange(p))
            or len(keep) != basis.shape[1]
        ):
            raise ValueError("rank-coordinate keep/drop must partition coefficients")
        object.__setattr__(self, "basis", _readonly(basis, dtype=np.float64))
        object.__setattr__(
            self,
            "fitting_to_public",
            _readonly(transform, dtype=np.float64),
        )
        object.__setattr__(self, "keep", _readonly(keep, dtype=np.int64))
        object.__setattr__(self, "drop", _readonly(drop, dtype=np.int64))

    @property
    def n_coef(self) -> int:
        """Number of full fitting coordinates."""
        return self.basis.shape[0]

    @property
    def n_reduced(self) -> int:
        """Number of retained identifiable coordinates."""
        return self.basis.shape[1]

    @property
    def public_basis(self) -> npt.NDArray[np.float64]:
        """Map reduced solve coordinates to public coefficients."""
        return self.fitting_to_public @ self.basis

    def reconstruct_coefficients(
        self, reduced: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Map a reduced coefficient vector to public coordinates."""
        reduced_array = np.asarray(reduced, dtype=float)
        if reduced_array.shape != (self.n_reduced,):
            raise ValueError("reduced coefficient vector has incompatible shape")
        return self.public_basis @ reduced_array

    def reconstruct_covariance(self, reduced: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map a reduced covariance matrix to public coordinates."""
        reduced_array = np.asarray(reduced, dtype=float)
        if reduced_array.shape != (self.n_reduced, self.n_reduced):
            raise ValueError("reduced covariance has incompatible shape")
        return self.public_basis @ reduced_array @ self.public_basis.T


def rank_revealing_coordinate_map(
    weighted_design: npt.ArrayLike,
    balanced_penalty_root: npt.ArrayLike,
    *,
    retained_rank: int,
    fitting_to_public: npt.ArrayLike | None = None,
) -> IdentifiableCoordinateMap:
    """Select gdiPK-style retained columns from a frozen CPU layout."""
    design = np.asarray(weighted_design, dtype=float)
    root = np.asarray(balanced_penalty_root, dtype=float)
    if design.ndim != 2 or root.ndim != 2 or root.shape[1] != design.shape[1]:
        raise ValueError("rank-coordinate design and penalty root are incompatible")
    p = design.shape[1]
    if not 0 < retained_rank <= p:
        raise ValueError("rank-coordinate retained rank is invalid")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(root)):
        raise ValueError("rank-coordinate inputs must be finite")
    _, data_r_pivoted, data_pivots = linalg.qr(design, mode="economic", pivoting=True)
    inverse = np.empty_like(data_pivots)
    inverse[data_pivots] = np.arange(p)
    data_r = data_r_pivoted[:, inverse]
    data_norm = np.linalg.norm(data_r, ord="fro")
    root_norm = np.linalg.norm(root, ord="fro")
    if data_norm == 0 or root_norm == 0:
        raise ValueError("rank-coordinate layout needs data and penalty support")
    layout = np.vstack((data_r / data_norm, root / root_norm))
    pivots = linalg.qr(layout, mode="economic", pivoting=True)[2]
    keep = np.sort(np.asarray(pivots[:retained_rank], dtype=np.int64))
    drop = np.sort(np.asarray(pivots[retained_rank:], dtype=np.int64))
    basis = np.eye(p)[:, keep]
    transform = np.eye(p) if fitting_to_public is None else fitting_to_public
    return IdentifiableCoordinateMap(basis, transform, keep, drop)


def global_rank_coordinate_map(
    weighted_public_design: npt.ArrayLike,
    structure: PenaltyStructure,
    log_lambda: npt.ArrayLike,
    *,
    retained_rank: int,
) -> tuple[IdentifiableCoordinateMap, GlobalPenaltyCoordinates]:
    """Build a current global rank map from public constrained penalties.

    This remains a CPU oracle/setup primitive. A public Phase-2 route must
    provide its own JIT-compatible dynamic stabilization and must not import
    this module during a device score evaluation.
    """
    coordinates = global_penalty_coordinates(structure, log_lambda)
    coordinate_map = rank_revealing_coordinate_map(
        np.asarray(weighted_public_design) @ coordinates.transform,
        coordinates.balanced_root,
        retained_rank=retained_rank,
        fitting_to_public=coordinates.transform,
    )
    return coordinate_map, coordinates
