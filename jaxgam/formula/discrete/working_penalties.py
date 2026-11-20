"""CPU block transforms and roots used by pinned Sl.setup performance REML.

The caller checks its prospective construction budget before invoking this
module. Arrays remain local to coefficient blocks; no global penalty stack
is constructed here. Device transfer belongs to the fitting boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from jaxgam.penalties.structure import PenaltyStructure
from jaxgam.smooths.source_penalty_coordinates import source_symmetric_eigh


@dataclass(frozen=True)
class WorkingPenaltyBlock:
    """One initial coefficient transform and its source penalty range."""

    start: int
    stop: int
    sp_indices: tuple[int, ...]
    transform: np.ndarray
    penalties: tuple[np.ndarray, ...]
    rank: int
    roots: tuple[np.ndarray, ...]


def _source_root(matrix: np.ndarray, requested_rank: int) -> np.ndarray:
    """mroot's upper DPSTRF layout with source tol=0 and requested rank."""
    if requested_rank == 0:
        return np.zeros((len(matrix), 0))
    upper, pivots, rank, info = sla.lapack.dpstrf(matrix, lower=0, tol=0.0)
    if info < 0:
        raise FloatingPointError("Working penalty root factorization failed.")
    upper = np.triu(upper)
    if rank < len(upper):
        upper[rank:, rank:] = 0.0
    inverse_pivots = np.argsort(pivots - 1)
    return upper[:requested_rank, :][:, inverse_pivots].T


def _source_project_penalty(basis: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """Apply Sl.setup's left then right DGEMM before symmetrization."""
    columns = np.asfortranarray(basis)
    source = np.asfortranarray(penalty)
    left = sla.blas.dgemm(1.0, columns, source, trans_a=1)
    local = sla.blas.dgemm(1.0, left, columns)
    return (local.T + local) / 2.0


def source_root_product(root: np.ndarray) -> np.ndarray:
    """Mirror pcrossprod's transposed upper DSYRK on one local root."""
    transposed = np.array(root.T, order="F", copy=True)
    product = sla.blas.dsyrk(1.0, transposed, trans=1, lower=0)
    for column in range(len(product)):
        product[column + 1 :, column] = product[column, column + 1 :]
    return product


def build_working_penalty_blocks(
    structure: PenaltyStructure,
    *,
    source_eigensolver: bool = False,
) -> tuple[WorkingPenaltyBlock, ...]:
    """Port Sl.setup's default non-Cholesky initial local parameterization.

    Singletons use diagonal inverse roots or descending symmetric eigenpairs,
    producing an exact partial identity penalty. Overlapping components use
    the range of their unscaled sum and mroot with the combined block rank.
    """
    result = []
    for block in structure.blocks:
        penalties = block.dense_penalties()
        if not penalties:
            continue
        size = block.size
        if len(penalties) == 1:
            penalty = penalties[0]
            off_diagonal = penalty.copy()
            np.fill_diagonal(off_diagonal, 0.0)
            diagonal_only = not np.any(off_diagonal)
            del off_diagonal
            if diagonal_only:
                diagonal = np.diag(penalty)
                active = diagonal > 0.0
                scales = np.ones(size)
                scales[active] = 1.0 / np.sqrt(diagonal[active])
                transform = np.diag(scales)
                rank = int(np.sum(active))
            else:
                values, vectors = sla.eigh(penalty, driver="evr", check_finite=False)
                values, vectors = values[::-1], vectors[:, ::-1]
                rank = block.ranks[0]
                if np.any(values[:rank] <= 0.0):
                    raise FloatingPointError("Working penalty range is not positive.")
                active = np.arange(size) < rank
                scales = np.ones(size)
                scales[:rank] = 1.0 / np.sqrt(values[:rank])
                transform = vectors * scales
            result.append(
                WorkingPenaltyBlock(
                    block.start,
                    block.stop,
                    block.sp_indices,
                    transform,
                    (np.diag(active.astype(float)),),
                    rank,
                    (),
                )
            )
            continue
        total = penalties[0].copy()
        for component in penalties[1:]:
            total += component
        values, vectors = (
            source_symmetric_eigh(total)
            if source_eigensolver
            else sla.eigh(total, driver="evr", check_finite=False)
        )
        rank = int(np.sum(values > np.max(values) * np.finfo(float).eps ** 0.8))
        transform = vectors[:, ::-1]
        basis = transform[:, :rank]
        projected = []
        roots = []
        for component in penalties:
            local = _source_project_penalty(basis, component)
            roots.append(_source_root(local, rank))
            padded = np.zeros((size, size))
            padded[:rank, :rank] = local
            projected.append(padded)
        result.append(
            WorkingPenaltyBlock(
                block.start,
                block.stop,
                block.sp_indices,
                transform,
                tuple(projected),
                rank,
                tuple(roots),
            )
        )
    return tuple(result)
