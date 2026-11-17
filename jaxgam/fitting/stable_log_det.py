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

from jaxgam.jax_utils import StableLogDetTerms
from jaxgam.penalties.stable_reparam import stable_reparam

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


def _range_root(penalty: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (penalty + penalty.T)
    values, vectors = np.linalg.eigh(symmetric)
    largest = float(np.max(values, initial=0.0))
    keep = values > largest * _EPS_TWO_THIRDS if largest > 0.0 else values > 0.0
    return vectors[:, keep] * np.sqrt(values[keep])


def stable_block_roots(
    multi_block_sp_indices: Sequence[Sequence[int]],
    multi_block_proj_S: Sequence[Sequence[object]],
) -> tuple[StableBlockRoots, ...]:
    """Build each multi-penalty block's range-space roots once, at setup."""
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
        blocks.append(StableBlockRoots(tuple(sp_indices), tuple(active), tuple(roots)))
    return tuple(blocks)


def _block_terms(
    block: StableBlockRoots, rho: np.ndarray, deriv: int
) -> tuple[float, np.ndarray, np.ndarray]:
    size = len(block.sp_indices)
    det1 = np.zeros(size)
    det2 = np.zeros((size, size))
    if not block.roots:
        return 0.0, det1, det2
    active = np.asarray(block.active)
    result = stable_reparam(block.roots, rho[active], deriv=deriv)
    if deriv >= 1:
        det1[active] = result.det1
    if deriv >= 2:
        det2[np.ix_(active, active)] = result.det2
    return float(result.det), det1, det2


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
