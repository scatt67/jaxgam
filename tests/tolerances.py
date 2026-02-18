"""Tolerance classes and sign-normalization helpers for test comparisons.

Three tolerance tiers for different comparison contexts:
- STRICT: CPU self-consistency, exact algebraic properties
- MODERATE: GPU vs CPU, cross-backend comparisons
- LOOSE: PyMGCV vs R mgcv (different implementations, BLAS, algorithms)

Sign normalization removes LAPACK eigenvector sign ambiguity so that
element-wise comparisons work across different LAPACK implementations
(e.g. macOS Accelerate vs R's bundled reference LAPACK).

Usage::

    import numpy as np
    from tests.tolerances import STRICT, MODERATE, LOOSE, normalize_column_signs

    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)

    # For matrices affected by eigenvector sign ambiguity:
    np.testing.assert_allclose(
        normalize_column_signs(py_X), normalize_column_signs(r_X),
        rtol=MODERATE.rtol, atol=MODERATE.atol,
    )
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToleranceClass:
    """Tolerance specification for numerical comparisons."""

    rtol: float
    atol: float
    label: str


STRICT = ToleranceClass(rtol=1e-10, atol=1e-12, label="strict")
MODERATE = ToleranceClass(rtol=1e-6, atol=1e-8, label="moderate")
LOOSE = ToleranceClass(rtol=1e-3, atol=1e-5, label="loose")


def normalize_column_signs(X: np.ndarray) -> np.ndarray:
    """Normalize column signs so the element with largest absolute value is positive.

    Eigenvectors are only defined up to a ±1 sign per column. Different LAPACK
    implementations (Accelerate, reference LAPACK, MKL, OpenBLAS) may choose
    different signs. This function applies a deterministic convention — for each
    column, the entry with the largest absolute value is made positive — so that
    element-wise comparisons are valid across implementations.

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Matrix whose columns may have arbitrary sign conventions.

    Returns
    -------
    np.ndarray, shape (n, k)
        Copy of *X* with each column sign-normalized.
    """
    X = np.array(X, copy=True)
    if X.ndim == 1:
        if X[np.argmax(np.abs(X))] < 0:
            X = -X
        return X
    max_idx = np.argmax(np.abs(X), axis=0)
    signs = np.sign(X[max_idx, np.arange(X.shape[1])])
    signs[signs == 0] = 1.0
    X *= signs[np.newaxis, :]
    return X


def normalize_symmetric_signs(S: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
    """Normalize a symmetric matrix for sign-robust element-wise comparison.

    Under column sign flips D (diagonal ±1), a penalty matrix transforms as
    S' = D @ S @ D — off-diagonal S[i,j] flips when exactly one of columns
    i,j is sign-flipped, while diagonal entries are unchanged.

    The sign vector D is derived from the design matrix *X* (if given) using
    the same convention as ``normalize_column_signs``. If *X* is not
    provided, falls back to deriving signs from S's own columns (less
    reliable when diagonal entries dominate).

    Parameters
    ----------
    S : np.ndarray, shape (k, k)
        Symmetric matrix (e.g. penalty matrix).
    X : np.ndarray, shape (n, k), optional
        Design matrix from which to derive the sign convention. Recommended
        whenever the corresponding X is available.

    Returns
    -------
    np.ndarray, shape (k, k)
        Copy of *S* with sign convention applied via D @ S @ D.
    """
    ref = X if X is not None else S
    max_idx = np.argmax(np.abs(ref), axis=0)
    signs = np.sign(ref[max_idx, np.arange(ref.shape[1])])
    signs[signs == 0] = 1.0
    return signs[:, None] * S * signs[None, :]
