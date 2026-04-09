"""Shared utilities for smooth term construction.

Factor detection, level extraction, and data column access helpers
used across smooth modules (by_variable, random_effects, etc.).

This module is Phase 1 (NumPy only, no JAX imports).
"""

from __future__ import annotations

from typing import Any

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
