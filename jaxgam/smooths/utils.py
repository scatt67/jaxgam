"""Shared utilities for smooth term construction.

Factor detection, level extraction, data column access, and row-wise
Kronecker product helpers used across smooth modules (by_variable,
random_effects, tensor, etc.).

This module is Phase 1 (NumPy only, no JAX imports).
"""

from __future__ import annotations

from typing import Any

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd


def is_factor(col: pd.Series | npt.NDArray) -> bool:
    """Detect whether a column should be treated as a factor.

    Matches R's ``is.factor()`` semantics: only explicit categorical or
    string types are factors. Integers are NOT automatically promoted.

    Parameters
    ----------
    col : pd.Series or np.ndarray
        Column to check.

    Returns
    -------
    bool
        True if the column is a factor (categorical/string).
    """
    if isinstance(col, pd.Series):
        if hasattr(col, "cat"):
            return True
        if col.dtype == object or col.dtype.kind in ("U", "S", "T"):
            return True
        # pandas StringDtype (pd.StringDtype())
        return bool(pd.api.types.is_string_dtype(col))

    # numpy array
    return hasattr(col, "dtype") and (
        col.dtype == object or col.dtype.kind in ("U", "S")
    )


def get_factor_levels(col: pd.Series | npt.NDArray) -> list[Any]:
    """Extract sorted factor levels from a column.

    For pandas Categorical, uses ``.cat.categories`` to respect the
    user-defined level ordering. For other types, uses sorted unique values.

    Parameters
    ----------
    col : pd.Series or np.ndarray
        Factor column.

    Returns
    -------
    list[Any]
        Ordered factor levels (strings, ints, or other hashable types).
    """
    if isinstance(col, pd.Series) and hasattr(col, "cat"):
        return list(col.cat.categories)
    return sorted(np.unique(col).tolist())


def is_ordered_factor(col: pd.Series | npt.NDArray) -> bool:
    """Check whether a factor column is ordered.

    Parameters
    ----------
    col : pd.Series or np.ndarray
        Factor column (must already pass ``is_factor``).

    Returns
    -------
    bool
        True if ordered (pandas Categorical with ``ordered=True``).
    """
    if isinstance(col, pd.Series) and hasattr(col, "cat"):
        return col.cat.ordered
    return False


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]))
def row_tensor(
    A: npt.NDArray[np.floating], B: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Row-wise Kronecker product of two matrices.

    For each row i, computes ``A[i, :] ⊗ B[i, :]``.  The second
    argument (B) varies fastest in the output columns.

    Parameters
    ----------
    A : np.ndarray
        Shape ``(n, ka)``.
    B : np.ndarray
        Shape ``(n, kb)``.

    Returns
    -------
    np.ndarray
        Shape ``(n, ka * kb)``.
    """
    n = A.shape[0]
    ka = A.shape[1]
    kb = B.shape[1]
    result = np.empty((n, ka * kb))
    for i in range(ka):
        for j in range(n):
            for k in range(kb):
                result[j, i * kb + k] = A[j, i] * B[j, k]
    return result


@numba.njit(numba.float64[:, :](numba.float64[:, :, :], numba.int64[:]))
def interaction_matrix(
    encodings: npt.NDArray[np.floating],
    widths: npt.NDArray[np.signedinteger],
) -> npt.NDArray[np.floating]:
    """Build an interaction model matrix from per-variable encodings.

    Chains :func:`row_tensor` so the first variable varies fastest and
    the last varies slowest, matching R's ``model.matrix(~v1:v2:…:vN - 1)``.

    Parameters
    ----------
    encodings : np.ndarray, shape ``(n_vars, n, max_k)``
        Per-variable encoding matrices, zero-padded to ``max_k`` columns.
    widths : np.ndarray, shape ``(n_vars,)``
        Actual column count for each variable's encoding.

    Returns
    -------
    np.ndarray
        Interaction matrix, shape ``(n, prod(widths))``.
    """
    n_vars = widths.shape[0]
    result = encodings[0, :, : widths[0]]
    for idx in range(1, n_vars):
        result = row_tensor(encodings[idx, :, : widths[idx]], result)
    return result


@numba.njit(
    numba.types.Tuple((numba.float64[:], numba.float64[:, :]))(
        numba.float64[:, ::1], numba.int64, numba.float64
    ),
    cache=True,
)
def _slanczos_jit(  # pragma: no cover
    A: npt.NDArray[np.floating], k: int, tol: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Numba-compiled Lanczos core; see _slanczos() for documentation."""
    n = A.shape[0]

    # --- Deterministic starting vector (R's LCG) ---
    q0 = np.empty(n)
    jran = 1
    for i in range(n):
        jran = (jran * 106 + 1283) % 6075
        q0[i] = jran / 6075.0 - 0.5
    q0 /= np.linalg.norm(q0)

    # --- Lanczos iteration ---
    Q = np.empty((n, n))
    Q[:, 0] = q0
    alpha = np.empty(n)
    beta = np.empty(n)

    # Convergence check frequency (matching R)
    f_check = k // 2
    if f_check < 10:
        f_check = 10
    kk = n // 10
    if kk < 1:
        kk = 1
    if kk < f_check:
        f_check = kk

    j_final = n
    n_pos = 0
    n_neg = 0
    converged = False
    d = np.zeros(1)
    v_tri = np.zeros((1, 1))

    for j in range(n):
        qj = np.ascontiguousarray(Q[:, j])
        z = A @ qj
        alpha[j] = qj @ z

        if j == 0:
            z -= alpha[0] * qj
        else:
            z -= alpha[j] * qj + beta[j - 1] * np.ascontiguousarray(Q[:, j - 1])
            # Double reorthogonalization (CGS via BLAS gemv)
            Qj = np.ascontiguousarray(Q[:, : j + 1])
            for _pass in range(2):
                z -= Qj @ (Qj.T @ z)

        beta[j] = np.linalg.norm(z)

        if j < n - 1:
            Q[:, j + 1] = z / beta[j]

        # --- Convergence check ---
        if not ((j >= k and j % f_check == 0) or j == n - 1):
            continue

        # Build tridiagonal matrix and eigendecompose
        size = j + 1
        T_mat = np.zeros((size, size))
        for idx in range(size):
            T_mat[idx, idx] = alpha[idx]
        for idx in range(j):
            T_mat[idx, idx + 1] = beta[idx]
            T_mat[idx + 1, idx] = beta[idx]
        d, v_tri = np.linalg.eigh(T_mat)
        # Reverse to descending order
        d = d[::-1].copy()
        v_tri = v_tri[:, ::-1].copy()

        # Error bounds: |beta_j * last component of kth Ritz vector|
        norm_Tj = max(abs(d[0]), abs(d[-1]))
        max_err = norm_Tj * tol
        err = np.abs(beta[j] * v_tri[-1, :])

        # Biggest mode: greedily walk from both ends by magnitude
        pos_idx = 0
        ni = 0
        ok = True
        while pos_idx + ni < k:
            if abs(d[pos_idx]) >= abs(d[j - ni]):
                if err[pos_idx] > max_err:
                    ok = False
                    break
                pos_idx += 1
            else:
                if err[ni] > max_err:
                    ok = False
                    break
                ni += 1

        if ok:
            j_final = j + 1
            n_pos = pos_idx
            n_neg = ni
            converged = True
            break

    if not converged:
        j_final = n
        pos_idx = 0
        ni = 0
        while pos_idx + ni < k:
            if abs(d[pos_idx]) >= abs(d[n - 1 - ni]):
                pos_idx += 1
            else:
                ni += 1
        n_pos = pos_idx
        n_neg = ni

    # --- Build output eigenvalues and Ritz vectors ---
    pos_idx = np.arange(n_pos)
    neg_idx = np.arange(j_final - n_neg, j_final)
    sel = np.concatenate((pos_idx, neg_idx))

    D = d[sel]
    Q_cont = np.ascontiguousarray(Q[:, :j_final])
    U = Q_cont @ np.ascontiguousarray(v_tri[:, sel])

    return D, U


def _slanczos(
    A: npt.NDArray[np.floating],
    k: int,
    tol: float | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Lanczos eigendecomposition matching R's mgcv Rlanczos (biggest mode).

    Reimplements the Rlanczos function from mgcv/src/mat.c with minus=-1
    (largest magnitude eigenvalues). Uses the same deterministic LCG
    starting vector, double reorthogonalization, and convergence
    criteria as R.

    JIT-compiled via Numba for native performance.

    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix, shape ``(n, n)``.
    k : int
        Number of eigenvalues/vectors to compute (largest magnitude).
    tol : float, optional
        Convergence tolerance. Default: ``np.finfo(float).eps ** 0.7``.

    Returns
    -------
    D : np.ndarray
        Eigenvalues, shape ``(k,)``. Positive eigenvalues first
        (descending), then negative eigenvalues.
    U : np.ndarray
        Eigenvectors, shape ``(n, k)``.
    """
    if tol is None:
        tol = np.finfo(float).eps ** 0.7
    return _slanczos_jit(A, k, tol)


def _compute_distance_matrix(
    X1: npt.NDArray[np.floating],
    X2: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute pairwise Euclidean distance matrix.

    Parameters
    ----------
    X1 : np.ndarray
        Shape ``(n1, d)``.
    X2 : np.ndarray
        Shape ``(n2, d)``.

    Returns
    -------
    np.ndarray
        Distance matrix, shape ``(n1, n2)``.
    """
    # Use broadcasting for efficiency
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def _get_unique_rows(
    X: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp]]:
    """Get unique rows and inverse mapping, sorted lexicographically.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n, d)``.

    Returns
    -------
    Xu : np.ndarray
        Unique rows, shape ``(n_unique, d)``, sorted lexicographically.
    inverse : np.ndarray
        Index array such that ``Xu[inverse] == X`` (up to float tolerance).
    """
    # Round to handle floating-point duplicates
    # Use np.unique with axis=0 which sorts lexicographically
    Xu, inverse = np.unique(X, axis=0, return_inverse=True)
    return Xu, inverse


def _subsample_knots(
    Xu: npt.NDArray[np.floating], max_knots: int, seed: int = 1
) -> npt.NDArray[np.floating]:
    """Reproducible knot subsampling. Matches R's mgcv pattern.

    Uses np.random.RandomState (legacy API) intentionally to preserve
    bit-exact reproducibility with TPRS's pre-refactor behavior.
    """
    if Xu.shape[0] <= max_knots:
        return Xu
    rng = np.random.RandomState(seed)
    idx = rng.choice(Xu.shape[0], max_knots, replace=False)
    idx.sort()
    return Xu[idx]


def get_col(
    data: dict[str, npt.NDArray[np.floating]] | pd.DataFrame,
    name: str,
) -> pd.Series | npt.NDArray:
    """Extract a column from data (dict or DataFrame).

    Parameters
    ----------
    data : dict or DataFrame
        Data source.
    name : str
        Column name.

    Returns
    -------
    pd.Series or np.ndarray
        The column.

    Raises
    ------
    KeyError
        If the column is not found.
    """
    if isinstance(data, pd.DataFrame):
        if name not in data.columns:
            raise KeyError(
                f"Variable '{name}' not found in data. Available: {list(data.columns)}"
            )
        return data[name]
    if name not in data:
        raise KeyError(
            f"Variable '{name}' not found in data. Available: {list(data.keys())}"
        )
    return data[name]
