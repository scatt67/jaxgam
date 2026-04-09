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
