"""Stable similarity reparameterization of a multi-penalty block.

Port of pinned mgcv 1.9-3 ``gam.reparam`` (``R/gam.fit3.r`` lines 9-60) and
the C routine it calls, ``get_stableS`` (``src/gdi.c`` lines 550-794), with
its determinant helper ``qr_ldet_inv`` (``src/gdi.c`` lines 257-296).

For a block whose total penalty is ``S0 = sum_j exp(rho_j) S_j`` with
``S_j = rS_j rS_j'``, forming or factoring ``S0`` in fixed coordinates lets
roundoff of a dominant ``exp(rho_i) S_i`` swamp the subspace on which only a
dominated ``S_j`` acts.  ``get_stableS`` instead builds an orthogonal ``Qs``
term by term, from the dominant penalties downward: at each stage it splits
the remaining penalties into a dominant set (Frobenius-norm test against
``d_tol``), finds the rank ``r`` of the dominant set's range (eigenvalue test
against ``r_tol``), rotates the remaining coordinates onto that range and its
complement, keeps exactly the ``r`` dominant eigenvalues, and recurses onto
the complement with the deferred penalties only.  ``log|S|`` and its first
and second derivatives with respect to ``rho`` are then evaluated from the
transformed ``S`` and the transformed roots.

The C control flow is followed step by step, including mgcv's quirks: the
root of a fixed (unweighted) penalty is never transformed (gdi.c:732-733), and
``r`` starts at one, so a stage always peels off at least one coordinate.

LAPACK provenance: ``mgcv_symeig`` with ``use_dsyevd=0`` calls DSYEVR
(``uplo='L'``, ``range='A'``, ``abstol=0``); ``mgcv_qr`` calls DGEQP3;
``mgcv_qrqy`` calls DORMQR; ``mgcv_backsolve`` calls DTRSM; ``mroot`` calls
R's ``chol(pivot=TRUE, tol=0)``, i.e. DPSTRF.  This module calls the same
routines through SciPy's LAPACK.

This is Phase 1 code: NumPy and SciPy only, no JAX.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.linalg import blas, lapack

from jaxgam.smooths.source_lapack import source_frobenius_norm

# gam.fit3.r:31,33 -- d.tol groups "similar sized" penalties; r.tol decides rank.
D_TOL: float = float(np.finfo(np.float64).eps ** 0.3)
R_TOL: float = float(np.finfo(np.float64).eps ** 0.75)


@dataclass(frozen=True)
class StableReparam:
    """``gam.reparam``'s return value for one penalty block.

    Attributes
    ----------
    S : (q, q) transformed total penalty, ``Qs' S0 Qs`` evaluated stably and
        symmetrized as ``gam.reparam`` does.
    E : (q, q) square root with ``E'E = S`` from the diagonally
        pre-conditioned pivoted Cholesky (``mroot``) of ``S``.
    rS : transformed penalty roots, in input order.  The root of a fixed
        penalty (when ``len(rS) == len(log_sp) + 1``) is returned untransformed,
        exactly as ``get_stableS`` leaves it.
    Qs : (q, q) orthogonal transform with ``S = Qs' S0 Qs``.
    det : ``log|S|`` from the pivoted QR of the transformed ``S``.
    det1 : gradient of ``det`` with respect to ``log_sp`` (``deriv >= 1``).
    det2 : Hessian of ``det`` with respect to ``log_sp`` (``deriv == 2``).
    fixed_penalty : whether the last root is a fixed penalty.
    stage_ranks : the rank ``r`` peeled off by each transforming stage (empty
        when ``Qs`` is the identity).  ``sum(stage_ranks)`` coordinates are
        assigned to dominant ranges; the rest form the final stage's block.
    """

    S: npt.NDArray[np.float64]
    E: npt.NDArray[np.float64]
    rS: tuple[npt.NDArray[np.float64], ...]
    Qs: npt.NDArray[np.float64]
    det: float
    det1: npt.NDArray[np.float64] | None
    det2: npt.NDArray[np.float64] | None
    fixed_penalty: bool
    stage_ranks: tuple[int, ...]


def _symeig_values(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``mgcv_symeig(A, ev, n, use_dsyevd=0, get_vectors=0, descending=0)``.

    DSYEVR on the lower triangle; eigenvalues ascending (mat.c:3467-3540).
    """
    values, _, count, _, info = lapack.dsyevr(
        matrix, compute_v=0, range="A", lower=1, abstol=0.0
    )
    if info != 0 or count != len(matrix):
        raise np.linalg.LinAlgError(f"DSYEVR failed with info={info}.")
    return np.asarray(values[: len(matrix)], dtype=np.float64)


def _symeig_descending(
    matrix: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``mgcv_symeig(A, ev, n, use_dsyevd=0, get_vectors=1, descending=1)``.

    DSYEVR on the lower triangle, then mgcv reverses the eigenvector columns
    and the eigenvalues into descending order (mat.c:3520-3545).
    """
    values, vectors, count, _, info = lapack.dsyevr(
        matrix, compute_v=1, range="A", lower=1, abstol=0.0
    )
    size = len(matrix)
    if info != 0 or count != size:
        raise np.linalg.LinAlgError(f"DSYEVR failed with info={info}.")
    return (
        np.asarray(values[:size][::-1], dtype=np.float64),
        np.asarray(vectors[:, :size][:, ::-1], dtype=np.float64),
    )


def _qr_ldet_inv(
    matrix: npt.NDArray[np.float64], *, get_inverse: bool
) -> tuple[float, npt.NDArray[np.float64] | None]:
    """Port ``gdi.c::qr_ldet_inv`` (lines 257-296).

    DGEQP3 pivoted QR of the square ``matrix``; ``log|X| = sum log|R_ii|``.
    When requested, ``X^{-1} = P R^{-1} Q'``: DORMQR forms ``Q'`` from the
    identity, DTRSM back-solves with ``R`` and the rows are unpivoted.
    """
    size = len(matrix)
    factor, pivots, tau, _, info = lapack.dgeqp3(np.array(matrix, order="F"))
    if info != 0:
        raise np.linalg.LinAlgError(f"DGEQP3 failed with info={info}.")
    log_det = float(np.sum(np.log(np.abs(np.diag(factor)))))
    if not get_inverse:
        return log_det, None
    q_transpose, _, info = lapack.dormqr(
        "L", "T", factor, tau, np.eye(size, order="F"), lwork=max(1, 64 * size)
    )
    if info != 0:
        raise np.linalg.LinAlgError(f"DORMQR failed with info={info}.")
    pivoted_inverse = _dtrsm_upper(factor, q_transpose)
    inverse = np.empty_like(pivoted_inverse)
    # gdi.c:285-290 -- pivoted row i goes to unpivoted row pivot[i].
    inverse[pivots - 1, :] = pivoted_inverse
    return log_det, inverse


def _dtrsm_upper(
    factor: npt.NDArray[np.float64], right_hand_side: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """``mgcv_backsolve(right=0)``: DTRSM('L','U','N','N') with ``alpha = 1``."""
    return np.asarray(
        blas.dtrsm(1.0, factor, right_hand_side, side=0, lower=0, trans_a=0, diag=0),
        dtype=np.float64,
    )


def _trace_btab(
    matrix: npt.NDArray[np.float64], root: npt.NDArray[np.float64]
) -> float:
    """``gdi.c::trBtAB`` (lines 35-48): ``tr(B'AB)``."""
    return float(np.sum(root * (matrix @ root)))


def _trace_ab(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    """``gdi.c::trAB`` (lines 90-98): ``tr(AB)`` for square ``A`` and ``B``."""
    return float(np.sum(left * right.T))


def _penalty_square(root: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``mgcv_mmult(A, B, B, bt=0, ct=1)``: mat.c ``getXXt``.

    DSYRK('L','N') forms the lower triangle of ``B B'`` and mgcv mirrors it
    into the upper triangle, so ``S_i`` is exactly symmetric.
    """
    lower = np.tril(blas.dsyrk(1.0, np.asfortranarray(root), lower=1, trans=0))
    return np.asarray(lower + np.tril(lower, -1).T, dtype=np.float64)


def _mroot_full_rank(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``mroot(A, rank=q)`` with ``method='chol'`` (mgcv.r ``mroot``).

    R's ``chol(A, pivot=TRUE, tol=0)`` (DPSTRF, upper) with the lower
    triangle zero and the trailing block beyond the computed rank zeroed;
    the columns are unpivoted and the result transposed, so ``L L' = A``.
    """
    upper, pivots, rank, info = lapack.dpstrf(np.array(matrix), lower=0, tol=0.0)
    if info < 0:
        raise np.linalg.LinAlgError(f"DPSTRF failed with info={info}.")
    upper = np.triu(upper)
    if rank < len(upper):
        upper[rank:, rank:] = 0.0
    return np.asarray(upper[:, np.argsort(pivots - 1)].T, dtype=np.float64)


def _validated_inputs(
    rS: Sequence[npt.ArrayLike], log_sp: npt.ArrayLike, deriv: int
) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.float64], bool]:
    roots = [np.array(root, dtype=np.float64, order="F") for root in rS]
    rho = np.array(log_sp, dtype=np.float64).reshape(-1)
    if deriv not in (0, 1, 2):
        raise ValueError("deriv must be 0, 1 or 2.")
    if not roots:
        raise ValueError("At least one penalty root is required.")
    if any(root.ndim != 2 or root.shape[1] < 1 for root in roots):
        raise ValueError("Each penalty root must be a two-dimensional q by r_j array.")
    size = roots[0].shape[0]
    if size < 1 or any(root.shape[0] != size for root in roots):
        raise ValueError("All penalty roots must share the same number of rows q.")
    if len(roots) not in (len(rho), len(rho) + 1):
        raise ValueError(
            "Need one root per log smoothing parameter, plus at most one fixed root."
        )
    if not all(np.all(np.isfinite(root)) for root in roots):
        raise ValueError("Penalty roots must be finite.")
    with np.errstate(over="ignore"):
        finite_sp = np.all(np.isfinite(np.exp(rho)))
    if not np.all(np.isfinite(rho)) or not finite_sp:
        raise ValueError("Smoothing parameters exp(log_sp) must be finite.")
    return roots, rho, len(roots) > len(rho)


def stable_reparam(
    rS: Sequence[npt.ArrayLike], log_sp: npt.ArrayLike, deriv: int = 2
) -> StableReparam:
    """Port ``mgcv:::gam.reparam(rS, lsp, deriv)`` for one penalty block.

    Parameters
    ----------
    rS : penalty square roots ``rS_j`` (``q`` by ``r_j``, ``rS_j rS_j' = S_j``),
        as mgcv passes them: one per smoothing parameter, optionally followed
        by the root of a fixed penalty.  mgcv assumes the block is already in
        the range space of its total penalty (``totalPenaltySpace``); if it is
        not, ``S`` is singular and ``det`` is not meaningful.
    log_sp : log smoothing parameters, one per non-fixed root.
    deriv : 0, 1 or 2 -- the highest derivative of ``det`` to return.
    """
    roots, rho, fixed_penalty = _validated_inputs(rS, log_sp, deriv)
    size = roots[0].shape[0]
    n_sp = len(rho)
    n_terms = len(roots)
    sp = np.exp(rho)
    # gdi.c:603-609 -- the fixed term carries weight one.
    spf = np.concatenate([sp, [1.0]]) if fixed_penalty else sp

    # gdi.c:618-622 -- form the S_i explicitly (mgcv_mmult(.., bt=0, ct=1)).
    blocks = [_penalty_square(root) for root in roots]

    total = np.zeros((size, size))
    transform = np.zeros((size, size))
    frob = np.zeros(n_terms)
    gamma = np.ones(n_terms, dtype=bool)
    offset = 0  # K: coefficients already dealt with
    remaining = size  # Q: coefficients left to deal with
    stage_ranks: list[int] = []
    iteration = 0
    while True:  # gdi.c:647 -- the similarity transform loop
        iteration += 1
        # gdi.c:651-656 -- Frobenius norms of the S_i still in gamma.
        max_frob = 0.0
        for i in range(n_terms):
            if gamma[i]:
                frob[i] = source_frobenius_norm(blocks[i])
                max_frob = max(max_frob, frob[i] * spf[i])
        # gdi.c:658-671 -- dominant set alpha and deferred set gamma1.
        alpha = gamma & (frob * spf > max_frob * D_TOL)
        gamma1 = gamma & ~alpha

        # gdi.c:673-687 -- rank of the norm-scaled dominant sum by DSYEVR.
        if np.any(gamma1):
            scaled = np.zeros((remaining, remaining))
            for i in np.flatnonzero(alpha):
                scaled += blocks[i] / frob[i]
            values = _symeig_values(scaled)
            rank = 1
            while (
                rank < remaining
                and values[remaining - rank - 1] > values[remaining - 1] * R_TOL
            ):
                rank += 1
        else:
            rank = remaining

        # gdi.c:692-703 -- nothing left to separate: stop.
        if remaining == rank:
            if iteration == 1:
                for i in range(n_terms):
                    total += blocks[i] * spf[i]
                transform = np.eye(size)
            break

        # gdi.c:706-712 -- eigen-decompose the sp-weighted dominant term.
        dominant = np.zeros((remaining, remaining))
        for i in np.flatnonzero(alpha):
            dominant += blocks[i] * spf[i]
        eigenvalues, vectors = _symeig_descending(dominant)
        stage_ranks.append(rank)

        # gdi.c:714-718 -- accumulate Qf on the columns not yet fixed.
        if iteration == 1:
            transform = vectors.copy()
        else:
            transform[:, offset:] = transform[:, offset:] @ vectors

        # gdi.c:721-725 -- sp-weighted sum of the deferred terms.
        deferred = np.zeros((remaining, remaining))
        for i in np.flatnonzero(gamma1):
            deferred += blocks[i] * spf[i]

        # gdi.c:728-737 -- rotate the upper-right K by Q block of S.
        if offset > 0:
            upper = total[:offset, offset:] @ vectors
            total[:offset, offset:] = upper
            total[offset:, :offset] = upper.T

        # gdi.c:741-749 -- lower-right block U' Sg U plus the r dominant
        # eigenvalues; the dominant term's remaining eigenvalues are dropped.
        lower = vectors.T @ (deferred @ vectors)
        lower[np.arange(rank), np.arange(rank)] += eigenvalues[:rank]
        total[offset:, offset:] = lower

        # gdi.c:752-765 -- transform the non-fixed roots: dominant roots keep
        # their r range rows and are zeroed on the complement; deferred roots
        # are fully rotated.  A fixed root is left untransformed.
        for k in range(n_sp):
            rows = roots[k][offset:, :]
            if alpha[k]:
                roots[k][offset : offset + rank, :] = vectors[:, :rank].T @ rows
                roots[k][offset + rank :, :] = 0.0
            elif gamma1[k]:
                roots[k][offset:, :] = vectors.T @ rows

        # gdi.c:768-773 -- project the deferred S_i onto the complement.
        complement = vectors[:, rank:]
        for i in np.flatnonzero(gamma1):
            blocks[i] = (complement.T @ blocks[i]) @ complement

        # gdi.c:775-777 -- recurse onto the complement.
        offset += rank
        remaining -= rank
        gamma = gamma1

    # gdi.c:782-784 -- log|S| and S^{-1} by pivoted QR.
    log_det, inverse = _qr_ldet_inv(total, get_inverse=deriv > 0)

    det1: npt.NDArray[np.float64] | None = None
    det2: npt.NDArray[np.float64] | None = None
    if deriv > 0:
        assert inverse is not None
        # gdi.c:787-789 -- det1_i = sp_i tr(rS_i' S^{-1} rS_i).
        det1 = np.array([_trace_btab(inverse, roots[i]) * sp[i] for i in range(n_sp)])
    if deriv == 2:
        assert inverse is not None
        assert det1 is not None
        # gdi.c:791-799 -- det2_ij = -sp_i sp_j tr(S^{-1}S_i S^{-1}S_j) + d_ij det1_i.
        products = [(inverse @ roots[i]) @ roots[i].T for i in range(n_sp)]
        det2 = np.empty((n_sp, n_sp))
        for i in range(n_sp):
            for j in range(i, n_sp):
                value = -sp[i] * sp[j] * _trace_ab(products[i], products[j])
                det2[i, j] = det2[j, i] = value
        det2[np.diag_indices(n_sp)] += det1

    # gam.fit3.r:43-48 -- symmetrize, then the pre-conditioned square root.
    total = (total + total.T) * 0.5
    scale = np.abs(np.diag(total)) ** 0.5
    scale[scale == 0.0] = 1.0
    preconditioned = (total / scale[:, None]) / scale[None, :]
    preconditioned = (preconditioned + preconditioned.T) * 0.5
    root = (_mroot_full_rank(preconditioned) * scale[:, None]).T

    return StableReparam(
        S=total,
        E=root,
        rS=tuple(roots),
        Qs=transform,
        det=log_det,
        det1=det1,
        det2=det2,
        fixed_penalty=fixed_penalty,
        stage_ranks=tuple(stage_ranks),
    )
