"""Host ``gam.reparam`` log|S|+ terms for the dense REML criterion.

Pinned ``gam.fit3`` calls ``gam.reparam`` (``get_stableS``) at every outer
evaluation of the smoothing parameters (gam.fit3.r:144) and takes the penalty
log-determinant and its first and second derivatives from it
(gam.fit3.r:616-621). ``get_stableS`` branches on the smoothing parameters
(dominance grouping, rank decisions, number of stages), so it runs here on the
host, in NumPy, between jitted evaluations. Its outputs have shapes that do not
depend on the smoothing parameters and enter the jitted criterion as dynamic
arrays (:class:`jaxgam.jax_utils.StableLogDetTerms`). Nothing is
differentiated through this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import multi_block_eigenbasis
from jaxgam.jax_utils import StableLogDetTerms
from jaxgam.penalties.stable_reparam import StableReparam, stable_reparam

_EPS_TWO_THIRDS = float(np.finfo(np.float64).eps ** (2.0 / 3.0))


@dataclass(frozen=True)
class StableBlockRoots:
    """Range-space penalty roots of one multi-penalty block.

    ``roots`` holds a root ``R_j`` with ``R_j R_j' = S_j`` (in the block's
    total-penalty range basis, the coordinates of ``multi_block_proj_S``) for
    each nonzero penalty; ``active`` gives their positions in
    ``sp_indices``. A structurally zero penalty owns its smoothing parameter
    but contributes nothing to log|S|+ or its derivatives.
    """

    sp_indices: tuple[int, ...]
    active: tuple[int, ...]
    roots: tuple[np.ndarray, ...]
    #: Placement and eigenbasis ``U`` of the block in fitting coordinates
    #: (``U[:, -rank:]`` the range, ``U[:, :-rank]`` the null space), present
    #: when the roots were built from a penalty structure.
    start: int = 0
    stop: int = 0
    rank: int = 0
    basis: np.ndarray | None = None


@dataclass(frozen=True)
class StableSystem:
    """One evaluation's ``gam.reparam`` coordinates (gam.fit3.r 144-179).

    ``rotation`` maps stable to fitting coefficients block by block
    (``T = [Y Qs, Z]``, identity outside multi-penalty blocks);
    ``penalty_structure`` and ``penalty_roots`` are the penalties and roots
    in those coordinates (``get_stableS``'s ``rS`` padded with exact zeros
    on the null space); ``log_det_terms`` are the criterion's log|S|+ terms.
    """

    rotation: penalty_ops.JaxCoefficientRotation
    penalty_structure: penalty_ops.JaxPenaltyStructure
    penalty_roots: penalty_ops.JaxPenaltyRoots | None
    log_det_terms: tuple[StableLogDetTerms, ...]

    @property
    def host_rotation(self) -> tuple[tuple[int, int, np.ndarray], ...]:
        """``(start, stop, T)`` per rotated block, as NumPy arrays."""
        return tuple(
            (start, stop, np.asarray(T))
            for start, stop, T in zip(
                self.rotation.starts,
                self.rotation.stops,
                self.rotation.transforms,
                strict=True,
            )
        )


def _range_root(penalty: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (penalty + penalty.T)
    values, vectors = np.linalg.eigh(symmetric)
    largest = float(np.max(values, initial=0.0))
    keep = values > largest * _EPS_TWO_THIRDS if largest > 0.0 else values > 0.0
    return vectors[:, keep] * np.sqrt(values[keep])


def _host_penalty(penalty: penalty_ops.JaxLocalPenalty) -> np.ndarray:
    values = np.asarray(penalty.values, dtype=np.float64)
    if penalty.kind == "dense":
        return values
    if penalty.kind == "diagonal":
        return np.diag(values)
    return np.eye(penalty.size) * values


def _block_geometry(
    structure: penalty_ops.JaxPenaltyStructure | None, sp_indices: tuple[int, ...]
) -> dict:
    if structure is None:
        return {}
    for block in structure.blocks:
        if tuple(block.sp_indices) == sp_indices:
            penalties = [_host_penalty(penalty) for penalty in block.penalties]
            size = block.stop - block.start
            rank, basis = multi_block_eigenbasis(penalties, size)
            return {
                "start": block.start,
                "stop": block.stop,
                "rank": rank,
                "basis": basis,
            }
    raise ValueError(f"no penalty block owns smoothing parameters {sp_indices}")


def stable_block_roots(
    multi_block_sp_indices: Sequence[Sequence[int]],
    multi_block_proj_S: Sequence[Sequence[object]],
    structure: penalty_ops.JaxPenaltyStructure | None = None,
) -> tuple[StableBlockRoots, ...]:
    """Build each multi-penalty block's range-space roots once, at setup.

    With ``structure`` (the fitting-coordinate penalties), each block also
    records the eigenbasis behind ``multi_block_proj_S`` so that
    :func:`stable_system` can place the ``gam.reparam`` rotation.
    """
    blocks = []
    for sp_indices, projections in zip(
        multi_block_sp_indices, multi_block_proj_S, strict=True
    ):
        roots: list[np.ndarray] = []
        active: list[int] = []
        for position, projection in enumerate(projections):
            matrix = np.asarray(projection, dtype=np.float64)
            if matrix.size == 0 or not np.any(matrix):
                continue
            root = _range_root(matrix)
            if root.shape[1] == 0:
                continue
            roots.append(root)
            active.append(position)
        blocks.append(
            StableBlockRoots(
                tuple(sp_indices),
                tuple(active),
                tuple(roots),
                **_block_geometry(structure, tuple(sp_indices)),
            )
        )
    return tuple(blocks)


def _block_reparam(
    block: StableBlockRoots, rho: np.ndarray, deriv: int
) -> tuple[float, np.ndarray, np.ndarray, StableReparam | None]:
    size = len(block.sp_indices)
    det1 = np.zeros(size)
    det2 = np.zeros((size, size))
    if not block.roots:
        return 0.0, det1, det2, None
    active = np.asarray(block.active)
    result = stable_reparam(block.roots, rho[active], deriv=deriv)
    if deriv >= 1:
        det1[active] = result.det1
    if deriv >= 2:
        det2[np.ix_(active, active)] = result.det2
    return float(result.det), det1, det2, result


def _block_terms(
    block: StableBlockRoots, rho: np.ndarray, deriv: int
) -> tuple[float, np.ndarray, np.ndarray]:
    return _block_reparam(block, rho, deriv)[:3]


def stable_log_det_terms(
    blocks: Sequence[StableBlockRoots],
    log_lambda: np.ndarray,
    *,
    deriv: int,
    base_log_lambda: np.ndarray | None = None,
) -> tuple[StableLogDetTerms, ...]:
    """``get_stableS`` terms of every multi-penalty block at ``log_lambda``.

    ``log_lambda`` must be the exact values the jitted criterion will read
    (``StableLogDetTerms`` is NaN anywhere else). With ``base_log_lambda``,
    each term also carries ``det(log_lambda) - det(base_log_lambda)``.
    """
    log_lambda = np.asarray(log_lambda, dtype=np.float64)
    terms = []
    for block in blocks:
        indices = np.asarray(block.sp_indices)
        rho = log_lambda[indices]
        det, det1, det2 = _block_terms(block, rho, deriv)
        change = 0.0
        if base_log_lambda is not None:
            base = np.asarray(base_log_lambda, dtype=np.float64)[indices]
            change = det - _block_terms(block, base, 0)[0]
        terms.append(
            StableLogDetTerms(
                jnp.asarray(rho),
                jnp.asarray(det),
                jnp.asarray(det1),
                jnp.asarray(det2),
                jnp.asarray(change),
            )
        )
    return tuple(terms)


def criterion_multi_block_inputs(
    multi_block_sp_indices: Sequence[Sequence[int]],
    multi_block_proj_S: Sequence[Sequence[object]],
    log_lambda: np.ndarray,
    *,
    deriv: int,
    blocks: Sequence[StableBlockRoots] | None = None,
    base_log_lambda: np.ndarray | None = None,
) -> tuple:
    """The criterion's ``multi_block_proj_S`` argument at ``log_lambda``.

    With multi-penalty blocks, their host ``get_stableS`` terms; otherwise the
    (empty) projections unchanged. ``blocks`` reuses roots built once.
    """
    if not multi_block_sp_indices:
        return tuple(multi_block_proj_S)
    if blocks is None:
        blocks = stable_block_roots(multi_block_sp_indices, multi_block_proj_S)
    return stable_log_det_terms(
        blocks, log_lambda, deriv=deriv, base_log_lambda=base_log_lambda
    )


def _terms(
    rho: np.ndarray, det: float, det1: np.ndarray, det2: np.ndarray, change: float
) -> StableLogDetTerms:
    return StableLogDetTerms(
        jnp.asarray(rho),
        jnp.asarray(det),
        jnp.asarray(det1),
        jnp.asarray(det2),
        jnp.asarray(change),
    )


def stable_system(
    blocks: Sequence[StableBlockRoots],
    structure: penalty_ops.JaxPenaltyStructure,
    roots: penalty_ops.JaxPenaltyRoots | None,
    log_lambda: np.ndarray,
    *,
    deriv: int,
    base_log_lambda: np.ndarray | None = None,
) -> StableSystem:
    """``gam.reparam``'s coordinates of every multi-penalty block at ``log_lambda``.

    Per block, ``get_stableS`` on the range-space roots gives ``Qs``, the
    transformed roots ``rS`` and log|S|+ with its derivatives. The block's
    map to fitting coordinates is ``T = [Y Qs, Z]`` (R's ``U1`` times
    ``blockdiag(Qs, I)``, gam.fit3.r 161-162); its penalty ``j`` becomes
    ``rS_j rS_j'`` on the range coordinates and exact zeros on the null space
    (174-179), and its QR root ``[rS_j', 0]``. Single-penalty blocks are
    untouched. Shapes do not depend on ``log_lambda``.
    """
    log_lambda = np.asarray(log_lambda, dtype=np.float64)
    replaced_penalties: dict[int, tuple[np.ndarray, ...]] = {}
    replaced_roots: dict[tuple[int, int], np.ndarray] = {}
    starts, stops, transforms, terms = [], [], [], []
    for block in blocks:
        if block.basis is None:
            raise ValueError("stable_system needs blocks built with a structure")
        indices = np.asarray(block.sp_indices)
        rho = log_lambda[indices]
        det, det1, det2, result = _block_reparam(block, rho, deriv)
        change = 0.0
        if base_log_lambda is not None:
            base = np.asarray(base_log_lambda, dtype=np.float64)[indices]
            change = det - _block_terms(block, base, 0)[0]
        terms.append(_terms(rho, det, det1, det2, change))

        size, rank = block.stop - block.start, block.rank
        range_basis = block.basis[:, size - rank :]
        null_basis = block.basis[:, : size - rank]
        Qs = np.eye(rank) if result is None else result.Qs
        T = np.hstack([range_basis @ Qs, null_basis])
        starts.append(block.start)
        stops.append(block.stop)
        transforms.append(T)

        penalties = []
        stable_roots = {}
        for position, sp in enumerate(block.sp_indices):
            matrix = np.zeros((size, size))
            if result is not None and position in block.active:
                rS = result.rS[block.active.index(position)]
                square = rS @ rS.T
                matrix[:rank, :rank] = 0.5 * (square + square.T)
                root = np.zeros((rS.shape[1], size))
                root[:, :rank] = rS.T
                stable_roots[sp] = root
            penalties.append(matrix)
        replaced_penalties[block.start] = tuple(penalties)
        for sp, root in stable_roots.items():
            replaced_roots[(block.start, sp)] = root

    rotated_blocks = []
    for block in structure.blocks:
        penalties = replaced_penalties.get(block.start)
        if penalties is None or tuple(block.sp_indices) not in {
            b.sp_indices for b in blocks
        }:
            rotated_blocks.append(block)
            continue
        rotated_blocks.append(
            penalty_ops.JaxPenaltyBlock(
                block.start,
                block.stop,
                block.sp_indices,
                tuple(
                    penalty_ops.JaxLocalPenalty(
                        "dense", jnp.asarray(matrix), block.stop - block.start
                    )
                    for matrix in penalties
                ),
                block.transform,
            )
        )
    rotated_structure = penalty_ops.JaxPenaltyStructure(
        structure.n_coef, tuple(rotated_blocks)
    )

    rotated_roots = None
    if roots is not None:
        rotation_at = dict(zip(starts, transforms, strict=True))
        new_roots = []
        for start, stop, sp, root in zip(
            roots.starts, roots.stops, roots.sp_indices, roots.roots, strict=True
        ):
            if (start, sp) in replaced_roots:
                new_roots.append(jnp.asarray(replaced_roots[(start, sp)]))
            elif start in rotation_at and stop - start == rotation_at[start].shape[0]:
                # A structurally zero penalty: its (zero) root, rotated.
                new_roots.append(jnp.asarray(np.asarray(root) @ rotation_at[start]))
            else:
                new_roots.append(root)
        rotated_roots = penalty_ops.JaxPenaltyRoots(
            roots.n_coef, roots.starts, roots.stops, roots.sp_indices, tuple(new_roots)
        )

    rotation = penalty_ops.JaxCoefficientRotation(
        tuple(starts), tuple(stops), tuple(jnp.asarray(T) for T in transforms)
    )
    return StableSystem(rotation, rotated_structure, rotated_roots, tuple(terms))
