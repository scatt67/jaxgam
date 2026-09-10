"""Local, coefficient-block penalty descriptors.

The setup phase deliberately keeps a penalty in the coordinates of the term
that owns it.  A model with ``m`` smoothness parameters must not retain ``m``
zero-padded ``p x p`` matrices merely because the direct solver later needs a
single dense Hessian.

This module is Phase 1 code: it uses NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jaxgam.penalties.penalty import Penalty


@dataclass(frozen=True)
class IdentityPenalty:
    """A scaled identity penalty without allocating its dense matrix."""

    size: int
    scale: float = 1.0

    def dense(self) -> npt.NDArray[np.floating]:
        return np.eye(self.size) * self.scale


@dataclass(frozen=True)
class DiagonalPenalty:
    """A diagonal penalty without allocating off-diagonal zeroes."""

    diagonal: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagonal", _owned_readonly(self.diagonal))

    @property
    def size(self) -> int:
        return len(self.diagonal)

    def dense(self) -> npt.NDArray[np.floating]:
        return np.diag(self.diagonal)


@dataclass(frozen=True)
class DenseLocalPenalty:
    """A dense penalty in one coefficient block."""

    matrix: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _owned_readonly(self.matrix))

    @property
    def size(self) -> int:
        return self.matrix.shape[0]

    def dense(self) -> npt.NDArray[np.floating]:
        return self.matrix


LocalPenalty = IdentityPenalty | DiagonalPenalty | DenseLocalPenalty


@dataclass(frozen=True)
class IdentityTransform:
    """Identity fitting reparameterization for a coefficient block."""

    size: int

    def dense(self) -> npt.NDArray[np.floating]:
        return np.eye(self.size)


@dataclass(frozen=True)
class DiagonalTransform:
    """Diagonal fitting reparameterization kept as its diagonal."""

    diagonal: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagonal", _owned_readonly(self.diagonal))

    @property
    def size(self) -> int:
        return len(self.diagonal)

    def dense(self) -> npt.NDArray[np.floating]:
        return np.diag(self.diagonal)


@dataclass(frozen=True)
class DenseTransform:
    """Dense fitting reparameterization for a single coefficient block."""

    matrix: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _owned_readonly(self.matrix))

    @property
    def size(self) -> int:
        return self.matrix.shape[0]

    def dense(self) -> npt.NDArray[np.floating]:
        return self.matrix


LocalTransform = IdentityTransform | DiagonalTransform | DenseTransform


def _as_local_penalty(S: npt.NDArray[np.floating]) -> LocalPenalty:
    S = _owned_readonly(S)
    diagonal = np.diag(S)
    if np.array_equal(S, np.diag(diagonal)):
        if len(diagonal) and np.all(diagonal == diagonal[0]):
            return IdentityPenalty(len(diagonal), float(diagonal[0]))
        return DiagonalPenalty(_owned_readonly(diagonal))
    return DenseLocalPenalty(_owned_readonly(0.5 * (S + S.T)))


@dataclass(frozen=True)
class PenaltyBlock:
    """All local penalties and one fitting transform for a term block.

    ``sp_indices`` retain the global smoothing-parameter order while the
    matrices stay local.  Multiple entries are intentional for tensor terms:
    their penalties overlap and therefore share this one transform.
    """

    start: int
    stop: int
    sp_indices: tuple[int, ...]
    local_penalties: tuple[LocalPenalty, ...]
    transform: LocalTransform
    ranks: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.stop - self.start

    def dense_penalties(self) -> tuple[npt.NDArray[np.floating], ...]:
        return tuple(p.dense() for p in self.local_penalties)


@dataclass(frozen=True)
class PenaltyStructure:
    """All model penalties stored by their owning coefficient block."""

    n_coef: int
    blocks: tuple[PenaltyBlock, ...]

    @property
    def n_penalties(self) -> int:
        return 1 + max(
            (sp for block in self.blocks for sp in block.sp_indices), default=-1
        )

    @property
    def penalties(self) -> tuple[Penalty, ...]:
        """Compatibility view of local penalties; never pads them globally."""
        result: list[Penalty | None] = [None] * self.n_penalties
        for block in self.blocks:
            for sp, penalty, rank in zip(
                block.sp_indices, block.dense_penalties(), block.ranks, strict=True
            ):
                result[sp] = Penalty(penalty, rank=rank)
        return tuple(p for p in result if p is not None)

    def materialize(
        self, log_lambda: npt.NDArray[np.floating] | None = None
    ) -> np.ndarray:
        """Explicit compatibility materialization of the one combined penalty."""
        if log_lambda is None:
            log_lambda = np.zeros(self.n_penalties)
        result = np.zeros((self.n_coef, self.n_coef))
        for block in self.blocks:
            local = np.zeros((block.size, block.size))
            for sp, S in zip(block.sp_indices, block.dense_penalties(), strict=True):
                local += np.exp(log_lambda[sp]) * S
            result[block.start : block.stop, block.start : block.stop] += local
        return result


def make_penalty_structure(
    n_coef: int,
    smooth_blocks: list[object],
    constrained_penalties: list[list[npt.NDArray[np.floating]]],
) -> PenaltyStructure | None:
    """Build local descriptors after constraints without global padding."""
    blocks: list[PenaltyBlock] = []
    for term, penalties in zip(smooth_blocks, constrained_penalties, strict=True):
        if not penalties:
            continue
        dense = tuple(np.array(S, dtype=float, copy=True) for S in penalties)
        ranks = tuple(_penalty_rank(S, n_coef) for S in dense)
        if len(dense) > 1 and _non_overlapping(dense):
            # Factor-by levels are disjoint. Retain each actual coefficient
            # interval, not m copies of a q*k square wrapper penalty.
            for sp, S, rank in zip(term.penalty_indices, dense, ranks, strict=True):
                support = np.flatnonzero(np.sum(np.abs(S), axis=1) > 0)
                if len(support):
                    first, last = int(support[0]), int(support[-1]) + 1
                    local = _as_local_penalty(S[first:last, first:last])
                else:
                    # Retain the smoothing-parameter position even for an
                    # all-zero penalty without overlapping a real subblock.
                    first = last = 0
                    local = DiagonalPenalty(np.empty(0))
                blocks.append(
                    PenaltyBlock(
                        start=term.col_start + first,
                        stop=term.col_start + last,
                        sp_indices=(sp,),
                        local_penalties=(local,),
                        transform=IdentityTransform(last - first),
                        ranks=(rank,),
                    )
                )
        else:
            blocks.append(
                PenaltyBlock(
                    start=term.col_start,
                    stop=term.col_start + term.n_coefs,
                    sp_indices=tuple(term.penalty_indices),
                    local_penalties=tuple(_as_local_penalty(S) for S in dense),
                    transform=IdentityTransform(term.n_coefs),
                    ranks=ranks,
                )
            )
    return PenaltyStructure(n_coef=n_coef, blocks=tuple(blocks)) if blocks else None


def _penalty_rank(S: npt.NDArray[np.floating], n_coef: int) -> int:
    """The rank threshold used by the former globally padded Penalty view."""
    eigenvalues = np.linalg.eigvalsh(S)
    maximum = np.max(np.abs(eigenvalues)) if len(eigenvalues) else 0.0
    return int(np.sum(eigenvalues > maximum * max(n_coef, 1) * np.finfo(float).eps))


def _non_overlapping(penalties: tuple[npt.NDArray[np.floating], ...]) -> bool:
    """Whether every penalty has a disjoint contiguous coefficient interval."""
    intervals: list[tuple[int, int]] = []
    for penalty in penalties:
        support = np.flatnonzero(np.sum(np.abs(penalty), axis=1) > 0)
        if not len(support):
            continue
        intervals.append((int(support[0]), int(support[-1]) + 1))
    return all(
        right[0] >= left[1]
        for index, left in enumerate(sorted(intervals))
        for right in sorted(intervals)[index + 1 :]
    )


def _owned_readonly(values: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Make descriptor arrays independent from mutable smooth setup state."""
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result
