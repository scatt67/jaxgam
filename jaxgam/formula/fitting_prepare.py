"""CPU-only fitting-coordinate preparation shared by dense and streamed setup.

The routines here deliberately have no JAX dependency.  They describe the
small coefficient-space state which is frozen after Phase 1: local penalty
rotations, the initial smoothing-parameter scale, and normal-equation inputs
for a starting coefficient vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    DenseTransform,
    DiagonalPenalty,
    DiagonalTransform,
    IdentityPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)

_EPS_TWO_THIRDS = np.finfo(float).eps ** (2.0 / 3.0)
_ACTIVITY_THRESH = np.finfo(float).eps ** 0.8
_MAX_SP_ADJUST_ITERS = 200


@dataclass(frozen=True)
class ResponseReduction:
    """Observation reductions retained by prepared fitting, never y itself."""

    n_obs: int
    total_weight: float
    response_min: float
    response_max: float
    weighted_sum: float
    offset_min: float
    offset_max: float

    @property
    def weighted_mean(self) -> float:
        return self.weighted_sum / self.total_weight


@dataclass(frozen=True)
class FittingPreparation:
    """Frozen fitting-coordinate metadata for a prepared model."""

    penalty_structure: PenaltyStructure
    log_lambda_init: npt.NDArray[np.floating]
    beta_init: npt.NDArray[np.floating]
    gram: npt.NDArray[np.floating]
    rhs: npt.NDArray[np.floating]
    response: ResponseReduction


def _make_transform(D: np.ndarray) -> object:
    diagonal = np.diag(D)
    if np.array_equal(D, np.eye(len(D))):
        return IdentityTransform(len(D))
    if np.array_equal(D, np.diag(diagonal)):
        return DiagonalTransform(diagonal)
    return DenseTransform(D)


def _make_penalty(S: np.ndarray) -> object:
    diagonal = np.diag(S)
    if np.array_equal(S, np.diag(diagonal)):
        if len(diagonal) and np.all(diagonal == diagonal[0]):
            return IdentityPenalty(len(diagonal), float(diagonal[0]))
        return DiagonalPenalty(diagonal)
    return DenseLocalPenalty(0.5 * (S + S.T))


def reparameterize_structure(structure: PenaltyStructure) -> PenaltyStructure:
    """Apply mgcv-style local D transforms without globally padded matrices."""
    result: list[PenaltyBlock] = []
    for block in structure.blocks:
        penalties = block.dense_penalties()
        k = block.size
        if k == 0:
            result.append(block)
            continue
        if len(penalties) == 1:
            eigs, U = np.linalg.eigh(penalties[0])
            active = eigs > max(float(np.max(eigs)), 0.0) * _EPS_TWO_THIRDS
            scale = np.ones(k)
            scale[active] = 1.0 / np.sqrt(eigs[active])
            D = U * scale
        elif penalties_non_overlapping(list(penalties)):
            D = np.eye(k)
            for S in penalties:
                rows = np.flatnonzero(np.sum(np.abs(S), axis=1) > 0)
                if not len(rows):
                    continue
                first, last = int(rows[0]), int(rows[-1]) + 1
                eigs, U = np.linalg.eigh(S[first:last, first:last])
                active = eigs > max(float(np.max(eigs)), 0.0) * _EPS_TWO_THIRDS
                scale = np.ones(last - first)
                scale[active] = 1.0 / np.sqrt(eigs[active])
                D[first:last, first:last] = U * scale
        else:
            D = np.linalg.eigh(np.add.reduce(penalties))[1]
        local = tuple(D.T @ S @ D for S in penalties)
        result.append(
            PenaltyBlock(
                block.start,
                block.stop,
                block.sp_indices,
                tuple(_make_penalty(S) for S in local),
                _make_transform(D),
                block.ranks,
            )
        )
    return PenaltyStructure(structure.n_coef, tuple(result))


def penalties_non_overlapping(penalties: list[np.ndarray]) -> bool:
    """Return whether local penalty *enclosing intervals* do not overlap."""
    intervals: list[tuple[int, int]] = []
    for penalty in penalties:
        support = np.flatnonzero(np.sum(np.abs(penalty), axis=1) > 0)
        if len(support):
            intervals.append((int(support[0]), int(support[-1]) + 1))
    intervals.sort()
    return all(right[0] >= left[1] for left, right in pairwise(intervals))


def apply_transforms_to_design(
    X: npt.NDArray[np.floating], structure: PenaltyStructure
) -> npt.NDArray[np.floating]:
    """Return one fitting-coordinate design batch."""
    result = np.array(X, dtype=float, copy=True)
    for block in structure.blocks:
        if not isinstance(block.transform, IdentityTransform):
            result[:, block.start : block.stop] = (
                result[:, block.start : block.stop] @ block.transform.dense()
            )
    return result


def initial_log_sp_from_diagonal(
    ldxx: npt.NDArray[np.floating], structure: PenaltyStructure
) -> npt.NDArray[np.floating]:
    """R ``initial.sp`` calculation from streamed ``diag(X'WX)``."""
    if structure.n_penalties == 0:
        return np.zeros(0)
    def_sp = np.zeros(structure.n_penalties)
    ldss = np.zeros_like(ldxx)
    pen = np.zeros(len(ldxx), dtype=bool)
    for block in structure.blocks:
        for sp, S in zip(block.sp_indices, block.dense_penalties(), strict=True):
            if S.size == 0:
                continue
            maS = np.max(np.abs(S))
            if maS == 0:
                continue
            active = (
                (np.sum(np.abs(S), axis=1) / len(ldxx) > _ACTIVITY_THRESH * maS)
                & (np.sum(np.abs(S), axis=0) / len(ldxx) > _ACTIVITY_THRESH * maS)
                & (np.abs(np.diag(S)) > _ACTIVITY_THRESH * maS)
            )
            xx, ss = ldxx[block.start : block.stop][active], np.diag(S)[active]
            if len(xx) == 0 or np.mean(xx) <= 0 or np.mean(ss) <= 0:
                continue
            def_sp[sp] = np.mean(xx) / np.mean(ss)
            pen[block.start : block.stop] |= active
            ldss[block.start : block.stop] += def_sp[sp] * np.diag(S)
    index = (ldss > 0) & pen & (ldxx > 0)
    if not np.any(index):
        return np.zeros(structure.n_penalties)
    ldxx_s, ldss_s = ldxx[index].copy(), ldss[index].copy()
    for _ in range(_MAX_SP_ADJUST_ITERS):
        if np.mean(ldxx_s / (ldxx_s + ldss_s)) <= 0.4:
            break
        def_sp *= 10
        ldss_s *= 10
    for _ in range(_MAX_SP_ADJUST_ITERS):
        if np.mean(ldxx_s / (ldxx_s + ldss_s)) >= 0.4:
            break
        def_sp /= 10
        ldss_s /= 10
    return np.log(np.maximum(def_sp, np.finfo(float).tiny))
