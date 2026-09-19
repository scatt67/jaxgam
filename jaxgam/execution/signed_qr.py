"""Bounded CPU QR/SVD reduction of signed frozen working systems.

This follows mgcv 1.9-3 ``src/gdi.c::pls_fit1``: factor absolute weights,
augment actual penalty roots, then correct negative rows through an SVD.
The reducer retains two at-most-p-by-p QR states and a p-vector RHS. It
does not retain rows or form the signed normal-equation matrix.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from jaxgam.execution.qr import (
    LocalPenaltyRoot,
    PositiveQRState,
    QRFactorSolve,
    _unpivot,
    qr_update,
    solve_augmented_qr,
)


@dataclass(frozen=True)
class SignedQRState:
    """Observation-independent absolute/negative roots and stable ``X.T Wz``."""

    absolute: PositiveQRState
    negative: PositiveQRState
    rhs: np.ndarray
    direct_rhs: bool = False

    def __post_init__(self) -> None:
        value = np.array(self.rhs, dtype=float, copy=True)
        value.setflags(write=False)
        object.__setattr__(self, "rhs", value)


def signed_qr_update(
    state: SignedQRState | None,
    X: np.ndarray,
    weights: np.ndarray,
    z: np.ndarray,
    *,
    weighted_response: np.ndarray | None = None,
    use_weighted_response: bool = False,
) -> SignedQRState:
    """Merge one finite working block, optionally using source-stable ``Wz``.

    Zero-curvature rows have neutral QR contributions even when their unused
    pseudodata are not finite. Nonzero-weight pseudodata must be finite unless
    ``use_weighted_response`` selects the source's direct ``use.wy`` path.
    ``weighted_response``
    avoids reconstructing a cancellation-sensitive product from ``W`` and
    ``z``. Its entries may be nonzero at zero curvature, as permitted by
    ``pls_fit1``'s direct RHS path. Prior-zero rows must be masked by the
    family working-system owner before this numerical reducer is called.
    """
    X, w, z = np.asarray(X, float), np.asarray(weights, float), np.asarray(z, float)
    if X.ndim != 2 or w.shape != (len(X),) or z.shape != w.shape:
        raise ValueError("matching signed working row shapes required")
    active = w != 0.0
    if use_weighted_response and weighted_response is None:
        raise ValueError("direct RHS requires supplied Wz")
    if (
        not np.all(np.isfinite(X))
        or not np.all(np.isfinite(w))
        or (not use_weighted_response and not np.all(np.isfinite(z[active])))
    ):
        raise ValueError("finite signed working rows required")
    if state is not None and X.shape[1] != state.absolute.n_coef:
        raise ValueError("coefficient dimension changed")
    wz = np.zeros_like(w)
    if weighted_response is None:
        np.multiply(w, z, out=wz, where=active)
    else:
        wz = np.asarray(weighted_response, float)
        if wz.shape != w.shape or not np.all(np.isfinite(wz)):
            raise ValueError("finite matching Wz required")
    sqrt_abs = np.sqrt(np.abs(w))
    A = sqrt_abs[:, None] * X
    response = np.zeros_like(z)
    if not use_weighted_response:
        np.multiply(np.sign(w) * sqrt_abs, z, out=response, where=active)
    absolute = qr_update(
        None if state is None else state.absolute, A, response, n_coef=X.shape[1]
    )
    # Compress only negative rows; this temporary is bounded by the batch.
    negative = qr_update(
        None if state is None else state.negative,
        A[w < 0.0],
        np.zeros(np.count_nonzero(w < 0.0)),
        n_coef=X.shape[1],
    )
    rhs = X.T @ wz
    if state is not None:
        rhs = state.rhs + rhs
    return SignedQRState(
        absolute,
        negative,
        rhs,
        use_weighted_response or (state is not None and state.direct_rhs),
    )


@dataclass(frozen=True)
class SignedQRResult:
    """Coefficient candidate or explicit step-local Fisher recovery request."""

    absolute_factor: QRFactorSolve
    vectors: np.ndarray
    correction: np.ndarray
    coefficients: np.ndarray | None
    fisher_required: bool
    score_admissible: bool
    used_direct_rhs: bool

    def __post_init__(self) -> None:
        for name in ("vectors", "correction", "coefficients"):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)


def solve_signed_qr(
    state: SignedQRState,
    roots: Iterable[LocalPenaltyRoot],
    *,
    identifiable_subspace: Sequence[int] | np.ndarray | None = None,
    balanced_roots: Iterable[LocalPenaltyRoot] | None = None,
) -> SignedQRResult:
    """Apply ``pls_fit1``'s negative correction and stable RHS fallback.

    Indefiniteness uses the literal source ``-100 * machine epsilon`` gate.
    Values in the nonpositive boundary interval are zero pseudoinverse modes,
    never a valid likelihood determinant. Identifiable coordinates remain
    owned by preparation, exactly as for the positive QR solver.
    """
    factor = solve_augmented_qr(
        state.absolute,
        roots,
        identifiable_subspace=identifiable_subspace,
        balanced_roots=balanced_roots,
    )
    R, piv, keep = factor.R, factor.pivots, factor.keep
    rank = len(keep)
    N = _unpivot(state.negative.R, state.negative.pivots)[:, keep][:, piv]
    Z = linalg.solve_triangular(R, N.T, trans="T").T
    # Full coefficient-space right vectors include directions absent in N.
    _, singular, vt = linalg.svd(Z, full_matrices=True)
    correction = np.ones(rank)
    correction[: len(singular)] -= 2.0 * singular**2
    vectors = vt.T
    tolerance = 100.0 * np.finfo(float).eps
    if np.any(correction < -tolerance):
        return SignedQRResult(factor, vectors, correction, None, True, False, False)
    correction = np.maximum(correction, 0.0)
    b = state.rhs[keep][piv]
    q = R @ factor.coefficients[keep][piv]
    discrepancy = R.T @ q - b
    direct = state.direct_rhs or bool(discrepancy @ discrepancy > tolerance * (b @ b))
    if direct:
        q = linalg.solve_triangular(R, b, trans="T")
    inverse = np.zeros(rank)
    np.divide(1.0, correction, out=inverse, where=correction > 0.0)
    q = vectors @ (inverse * (vectors.T @ q))
    ordered = linalg.solve_triangular(R, q)
    reduced = np.zeros(rank)
    reduced[piv] = ordered
    return SignedQRResult(
        factor,
        vectors,
        correction,
        factor.reconstruct(reduced),
        False,
        bool(np.all(correction > 0.0)),
        direct,
    )
