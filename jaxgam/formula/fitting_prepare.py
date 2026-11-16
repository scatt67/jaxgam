"""CPU-only fitting-coordinate preparation shared by dense and streamed setup.

The routines here deliberately have no JAX dependency.  They describe the
small coefficient-space state which is frozen after Phase 1: local penalty
rotations, the initial smoothing-parameter scale, and normal-equation inputs
for a starting coefficient vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from scipy import linalg

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
from jaxgam.smooths.source_arithmetic import libm_exp
from jaxgam.smooths.source_penalty_coordinates import (
    scipy_uses_accelerate_lapack,
    source_symmetric_eigh,
)

_EPS_TWO_THIRDS = np.finfo(float).eps ** (2.0 / 3.0)
_ACTIVITY_THRESH = np.finfo(float).eps ** 0.8
_MAX_SP_ADJUST_ITERS = 200


def _owned_readonly(values: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Detach immutable CPU preparation leaves from caller-owned arrays."""
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ResponseReduction:
    """Observation reductions retained by prepared fitting, never y itself."""

    n_obs: int
    total_weight: float
    weight_min: float
    weight_max: float
    response_min: float
    response_max: float
    weighted_sum: float
    offset_min: float
    offset_max: float

    @property
    def weighted_mean(self) -> float:
        return self.weighted_sum / self.total_weight


@dataclass(frozen=True)
class QRLocalPenaltyRoot:
    """Immutable local ``E`` with ``E.T @ E = S`` for positive QR solves."""

    start: int
    stop: int
    sp_index: int
    root: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _owned_readonly(self.root))


@dataclass(frozen=True)
class FittingPreparation:
    """Frozen fitting-coordinate metadata for a prepared model."""

    penalty_structure: PenaltyStructure
    log_lambda_init: npt.NDArray[np.floating]
    beta_init: npt.NDArray[np.floating]
    gram: npt.NDArray[np.floating]
    rhs: npt.NDArray[np.floating]
    response: ResponseReduction
    total_penalty_rank: int
    total_penalty_null_dim: int
    unpenalized_rank_deficit: int
    family_name: str
    link_name: str
    family_execution_static_config: tuple[object, ...] | None
    family_parameter_snapshot: object | None


def qr_penalty_roots(structure: PenaltyStructure) -> tuple[QRLocalPenaltyRoot, ...]:
    """Factor all local PSD penalties once without global zero padding.

    Roots retain every smoothing-parameter slot, including declared rank-zero
    penalties.  The streamed QR route scales these roots by ``exp(rho / 2)``
    for its actual solve while reusing the identical unscaled layout for the
    scale-independent structural-rank system.
    """
    result: list[QRLocalPenaltyRoot] = []
    for block in structure.blocks:
        for sp_index, penalty, declared_rank in zip(
            block.sp_indices, block.dense_penalties(), block.ranks, strict=True
        ):
            raw = np.asarray(penalty, dtype=float)
            matrix = 0.5 * (raw + raw.T)
            diagonal = np.diag(matrix)
            if np.array_equal(matrix, np.diag(diagonal)):
                if np.any(diagonal < 0.0):
                    raise ValueError(
                        "QR penalty roots require positive-semidefinite "
                        "local penalties."
                    )
                if declared_rank < 0 or declared_rank > len(diagonal):
                    raise ValueError(
                        "QR penalty root rank is outside its local penalty dimension."
                    )
                selected = np.argsort(diagonal)[::-1][:declared_rank]
                if declared_rank and np.any(diagonal[selected] <= 0.0):
                    raise ValueError(
                        "QR penalty metadata declares a non-positive active direction."
                    )
                remainder = np.argsort(diagonal)[::-1][declared_rank:]
                tolerance = (
                    (float(np.max(np.abs(diagonal))) if len(diagonal) else 0.0)
                    * 100.0
                    * np.finfo(float).eps
                )
                if len(remainder) and np.any(diagonal[remainder] > tolerance):
                    raise ValueError(
                        "QR penalty root rank disagrees with a positive "
                        "local penalty direction."
                    )
                root = np.zeros((declared_rank, len(diagonal)))
                root[np.arange(declared_rank), selected] = np.sqrt(diagonal[selected])
                result.append(
                    QRLocalPenaltyRoot(block.start, block.stop, sp_index, root)
                )
                continue
            eigenvalues, vectors = np.linalg.eigh(matrix)
            maximum = float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 0.0
            tolerance = maximum * 100.0 * np.finfo(float).eps
            if len(eigenvalues) and float(np.min(eigenvalues)) < -tolerance:
                raise ValueError(
                    "QR penalty roots require positive-semidefinite local penalties."
                )
            if declared_rank < 0 or declared_rank > len(eigenvalues):
                raise ValueError(
                    "QR penalty root rank is outside its local penalty dimension."
                )
            descending = np.argsort(eigenvalues)[::-1]
            selected = descending[:declared_rank]
            remainder = descending[declared_rank:]
            if declared_rank and np.any(eigenvalues[selected] <= 0.0):
                raise ValueError(
                    "QR penalty metadata declares a non-positive active direction."
                )
            if len(remainder) and np.any(eigenvalues[remainder] > tolerance):
                raise ValueError(
                    "QR penalty root rank disagrees with a positive local "
                    "penalty direction."
                )
            root = np.sqrt(eigenvalues[selected])[:, None] * vectors[:, selected].T
            result.append(QRLocalPenaltyRoot(block.start, block.stop, sp_index, root))
    return tuple(result)


@dataclass(frozen=True)
class GlobalPenaltyCoordinates:
    """Current mgcv-style global penalty coordinates for one score state.

    These arrays are CPU preparation metadata.  ``transform`` is the current
    ``U1 %*% blockdiag(Qs, I)`` map from fitting coordinates back to public
    constrained coefficients.  It must be rebuilt for every log-smoothing
    state because ``Qs`` is smoothing-parameter dependent.
    """

    transform: npt.NDArray[np.float64]
    penalty_space: npt.NDArray[np.float64]
    balanced_root: npt.NDArray[np.float64]
    penalty_roots: tuple[npt.NDArray[np.float64], ...]

    def __post_init__(self) -> None:
        transform = np.asarray(self.transform, dtype=float)
        penalty_space = np.asarray(self.penalty_space, dtype=float)
        balanced_root = np.asarray(self.balanced_root, dtype=float)
        if transform.ndim != 2 or transform.shape[0] != transform.shape[1]:
            raise ValueError("global penalty transform must be square")
        if penalty_space.shape != transform.shape:
            raise ValueError("global penalty space has incompatible shape")
        if balanced_root.ndim != 2 or balanced_root.shape[1] != transform.shape[0]:
            raise ValueError("global balanced root has incompatible shape")
        if not all(
            np.all(np.isfinite(value))
            for value in (transform, penalty_space, balanced_root)
        ):
            raise ValueError("global penalty coordinates must be finite")
        roots: list[np.ndarray] = []
        for root in self.penalty_roots:
            root_array = np.asarray(root, dtype=float)
            if root_array.ndim != 2 or root_array.shape[1] != transform.shape[0]:
                raise ValueError("global penalty root has incompatible shape")
            if not np.all(np.isfinite(root_array)):
                raise ValueError("global penalty roots must be finite")
            roots.append(_owned_readonly(root_array))
        object.__setattr__(self, "transform", _owned_readonly(transform))
        object.__setattr__(self, "penalty_space", _owned_readonly(penalty_space))
        object.__setattr__(self, "balanced_root", _owned_readonly(balanced_root))
        object.__setattr__(self, "penalty_roots", tuple(roots))


def _symmetric_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use mgcv's lower-triangle DSYEVR coordinate gauge."""
    if scipy_uses_accelerate_lapack():
        # The audited SciPy Accelerate wheel preserves this coordinate gauge
        # for the selected coupled penalties. R uses its own libRlapack here.
        return linalg.eigh(
            np.asarray(matrix, dtype=np.float64), lower=True, driver="evr"
        )
    return source_symmetric_eigh(np.asarray(matrix, dtype=np.float64))


def _source_penalty_frobenius_norm(penalty: np.ndarray) -> float:
    """Match R's column-major ``sqrt(sum(S * S))`` with bounded scratch."""
    rows, columns = penalty.shape
    if rows == 0 or columns == 0:
        return 0.0
    chunk_rows = min(rows, 4096)
    work = np.empty(chunk_rows + 1, dtype=np.longdouble)
    accumulated = np.longdouble(0.0)
    for column in range(columns):
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            count = stop - start
            work[0] = accumulated
            values = penalty[start:stop, column]
            # R materializes S*S as binary64 before its sequential LDOUBLE sum.
            work[1 : count + 1] = values * values
            np.cumsum(work[: count + 1], dtype=np.longdouble, out=work[: count + 1])
            accumulated = work[count]
    return math.sqrt(float(accumulated))


def _root_columns(matrix: np.ndarray, rank: int | None = None) -> np.ndarray:
    """Compact PSD root columns, matching ``mini.roots``'s rank contract."""
    values, vectors = _symmetric_eigh(matrix)
    if rank is None:
        largest = float(np.max(values, initial=0.0))
        active = values > largest * _EPS_TWO_THIRDS
    else:
        if not 0 <= rank <= len(values):
            raise ValueError("global penalty root has invalid supplied rank")
        active = np.zeros(len(values), dtype=bool)
        if rank:
            active[-rank:] = True
        elif np.linalg.norm(matrix, ord="fro") != 0:
            raise ValueError("global penalty root rank is zero for a nonzero penalty")
        if rank and np.any(values[active] <= 0):
            raise ValueError("global penalty root has non-positive supplied direction")
    return vectors[:, active] * np.sqrt(values[active])


def _stable_range_rotation(
    roots: tuple[np.ndarray, ...], log_lambda: npt.ArrayLike
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Port ``gdi.c::get_stableS``'s orthogonal range-space rotation.

    The grouping and rank cutoffs are deliberately source-specific.  The
    routine returns ``Qs`` and transformed root *columns* in range
    coordinates; callers append the global penalty null space afterwards.
    """
    log_sp = np.asarray(log_lambda, dtype=float)
    if len(roots) != len(log_sp):
        raise ValueError("global penalty roots and smoothing parameters disagree")
    if log_sp.ndim != 1 or not np.all(np.isfinite(log_sp)):
        raise ValueError("global penalty log smoothing parameters must be finite")
    # gdi.c's sp[i] = exp(lsp[i]) is the C library's exp.
    smoothing_parameters = libm_exp(log_sp)
    if not np.all(np.isfinite(smoothing_parameters)) or np.any(
        smoothing_parameters <= 0
    ):
        raise ValueError("global penalty smoothing parameters must be positive")
    if not roots:
        return np.empty((0, 0)), ()
    q = roots[0].shape[0]
    if any(root.ndim != 2 or root.shape[0] != q for root in roots):
        raise ValueError("global penalty roots have incompatible dimensions")
    transformed_roots = [np.array(root, copy=True) for root in roots]
    gamma = np.ones(len(roots), dtype=bool)
    qf = np.eye(q)
    k = 0
    current = q
    d_tol = np.finfo(float).eps ** 0.3
    r_tol = np.finfo(float).eps ** 0.75

    while True:
        norms = np.array(
            [
                np.linalg.norm(root[k:, :].T @ root[k:, :], ord="fro")
                if active
                else 0.0
                for root, active in zip(transformed_roots, gamma, strict=True)
            ]
        )
        weighted_norms = norms * smoothing_parameters
        maximum = float(np.max(weighted_norms, initial=0.0))
        alpha = gamma & (weighted_norms > maximum * d_tol)
        if not np.any(alpha):
            raise ValueError("global penalty rotation has no active penalty")
        gamma_next = gamma & ~alpha
        if np.any(gamma_next):
            scaled = np.zeros((current, current))
            for root, norm, active in zip(transformed_roots, norms, alpha, strict=True):
                if active:
                    tail = root[k:, :]
                    scaled += tail @ tail.T / norm
            values, _ = _symmetric_eigh(scaled)
            rank = 1
            while rank < current and values[current - rank - 1] > values[-1] * r_tol:
                rank += 1
        else:
            rank = current

        if rank == current:
            break

        dominant = np.zeros((current, current))
        for index, root in enumerate(transformed_roots):
            if alpha[index]:
                tail = root[k:, :]
                dominant += smoothing_parameters[index] * tail @ tail.T
        values, vectors = _symmetric_eigh(dominant)
        values, rotation = values[::-1], vectors[:, ::-1]
        qf[:, k:] = qf[:, k:] @ rotation
        for index, root in enumerate(transformed_roots):
            tail = root[k:, :]
            rotated = rotation.T @ tail
            if alpha[index]:
                root[k : k + rank, :] = rotated[:rank, :]
                root[k + rank :, :] = 0.0
            elif gamma_next[index]:
                root[k:, :] = rotated
        gamma = gamma_next
        k += rank
        current -= rank
        if current == 0:
            break

    return qf, tuple(transformed_roots)


def global_penalty_coordinates(
    structure: PenaltyStructure,
    log_lambda: npt.ArrayLike,
) -> GlobalPenaltyCoordinates:
    """Build the current global ``U1/Qs`` coordinate state from public S_j.

    This follows pinned mgcv's ``totalPenaltySpace``, ``mini.roots`` and
    ``gam.reparam`` route.  It intentionally receives the public constrained
    penalty structure, never a block-local D-whitened fitting structure.
    """
    log_sp = np.asarray(log_lambda, dtype=float)
    if structure.n_penalties != len(log_sp):
        raise ValueError("global penalties and smoothing parameters disagree")
    if log_sp.ndim != 1 or not np.all(np.isfinite(log_sp)):
        raise ValueError("global penalty log smoothing parameters must be finite")
    # gdi.c's sp[i] = exp(lsp[i]) is the C library's exp.
    smoothing_parameters = libm_exp(log_sp)
    if not np.all(np.isfinite(smoothing_parameters)) or np.any(
        smoothing_parameters <= 0
    ):
        raise ValueError("global penalty smoothing parameters must be positive")
    if not structure.n_penalties:
        raise ValueError("global penalty coordinates require at least one penalty")
    balanced = np.zeros((structure.n_coef, structure.n_coef))
    components: list[tuple[int, int, int, np.ndarray, int]] = []
    for block in structure.blocks:
        for index, penalty, rank in zip(
            block.sp_indices,
            block.dense_penalties(),
            block.ranks,
            strict=True,
        ):
            # R's sum(S * S) accumulates before the double-precision sqrt.
            # The final rounding matters for tied global penalty directions.
            norm = _source_penalty_frobenius_norm(penalty)
            if norm == 0:
                raise ValueError(
                    "global penalty coordinates do not accept zero penalties"
                )
            balanced[block.start : block.stop, block.start : block.stop] += (
                penalty / norm
            )
            components.append((index, block.start, block.stop, penalty, rank))
    values, vectors = _symmetric_eigh(balanced)
    values, vectors = values[::-1], vectors[:, ::-1]
    active = values > values[0] * np.finfo(float).eps ** 0.66
    y = vectors[:, active]
    z = vectors[:, ~active]
    root_parts: list[list[np.ndarray]] = [[] for _ in range(structure.n_penalties)]
    for index, start, stop, penalty, rank in components:
        root_parts[index].append(y[start:stop, :].T @ _root_columns(penalty, rank))
    range_roots = tuple(
        np.concatenate(parts, axis=1) if parts else np.empty((y.shape[1], 0))
        for parts in root_parts
    )
    qs, stable_roots = _stable_range_rotation(range_roots, log_sp)
    q = structure.n_coef
    mp = z.shape[1]
    rotation = np.eye(q)
    rotation[: q - mp, : q - mp] = qs
    transform = np.column_stack((y, z)) @ rotation
    balanced_root = np.sqrt(values[active])[:, None] * y.T @ transform
    full_roots = tuple(
        np.vstack((root, np.zeros((mp, root.shape[1])))).T for root in stable_roots
    )
    return GlobalPenaltyCoordinates(
        transform,
        np.column_stack((y, z)),
        balanced_root,
        full_roots,
    )


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


def total_penalty_spaces(
    structure: PenaltyStructure,
) -> tuple[
    list[tuple[PenaltyBlock, npt.NDArray[np.floating]]],
    list[tuple[PenaltyBlock, npt.NDArray[np.floating]]],
]:
    """Return local null/range bases with FittingData's relative cutoff."""
    spectra: list[tuple[PenaltyBlock, np.ndarray, np.ndarray]] = []
    largest = 0.0
    for block in structure.blocks:
        total = np.zeros((block.size, block.size))
        for S in block.dense_penalties():
            norm = np.linalg.norm(S, "fro")
            if norm:
                total += S / norm
        eigs, vectors = np.linalg.eigh(total)
        spectra.append((block, eigs, vectors))
        largest = max(largest, float(np.max(eigs)) if len(eigs) else 0.0)
    cutoff = largest * _EPS_TWO_THIRDS
    nulls = [(block, vectors[:, eigs <= cutoff]) for block, eigs, vectors in spectra]
    ranges = [(block, vectors[:, eigs > cutoff]) for block, eigs, vectors in spectra]
    return nulls, ranges
