"""Factor-by and numeric-by smooth expansion.

Implements the ``by`` argument in smooth terms (``s(x, by=z)``). The
behavior depends on whether ``z`` is a factor (categorical) or numeric:

- **Factor by** (``s(x, by=fac)``): Creates one smooth per factor level,
  each with its own smoothing parameter λ. The design matrix is block-
  structured: row *i* has nonzeros only in the block for ``fac[i]``'s level.

- **Numeric by** (``s(x, by=z)``): Multiplies the smooth basis pointwise
  by the numeric variable ``z``. Same penalty, same λ.

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 5.7
R source reference: R/smooth.r smoothCon() by-variable handling
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from pymgcv.formula.terms import SmoothSpec
from pymgcv.penalties.penalty import Penalty
from pymgcv.smooths.base import Smooth

# ---------------------------------------------------------------------------
# Factor detection
# ---------------------------------------------------------------------------


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


def _get_factor_levels(col: pd.Series | npt.NDArray) -> list:
    """Extract sorted factor levels from a column.

    For pandas Categorical, uses ``.cat.categories`` to respect the
    user-defined level ordering. For other types, uses sorted unique values.

    Parameters
    ----------
    col : pd.Series or np.ndarray
        Factor column.

    Returns
    -------
    list
        Ordered factor levels.
    """
    if isinstance(col, pd.Series) and hasattr(col, "cat"):
        return list(col.cat.categories)
    return sorted(np.unique(col).tolist())


def _is_ordered_factor(col: pd.Series | npt.NDArray) -> bool:
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


# ---------------------------------------------------------------------------
# FactorBySmooth
# ---------------------------------------------------------------------------


class FactorBySmooth:
    """Smooth expansion for ``s(x, by=fac)`` where ``fac`` is a factor.

    Creates one smooth per factor level, each with its own smoothing
    parameter. The base smooth is constructed once (on the full data)
    so knots reflect the full covariate range. Each level's design matrix
    is the base design matrix multiplied by an indicator for that level.

    This is NOT a subclass of ``Smooth``. It is a model-assembly mechanism
    that wraps a base smooth and expands it into per-level components.

    Parameters
    ----------
    base_smooth : Smooth
        An already-constructed base smooth (``setup()`` must have been
        called). This defines the basis and penalty structure.
    spec : SmoothSpec
        The original smooth specification (with ``by`` set).
    levels : list
        Factor levels to create smooths for.
    by_variable : str
        Name of the by-variable in the data.
    """

    def __init__(
        self,
        base_smooth: Smooth,
        spec: SmoothSpec,
        levels: list,
        by_variable: str,
    ) -> None:
        base_smooth._require_setup()

        self.base_smooth = base_smooth
        self.spec = spec
        self.levels = list(levels)
        self.by_variable = by_variable
        self.n_levels = len(self.levels)

        # Per-level metadata
        k_per = base_smooth.n_coefs
        self.n_coefs = self.n_levels * k_per
        self.k_per_level = k_per
        self.null_space_dim = self.n_levels * base_smooth.null_space_dim
        self.rank = self.n_levels * base_smooth.rank
        self._is_setup = True

        # Store labels for each level (matching R: "s(x):facLEVEL")
        base_label = _smooth_label(spec)
        self.labels = [f"{base_label}:{by_variable}{lev}" for lev in self.levels]

    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]] | pd.DataFrame
    ) -> npt.NDArray[np.floating]:
        """Build block-structured design matrix.

        Row *i* has nonzeros only in the block corresponding to the
        factor level of observation *i*.

        Parameters
        ----------
        data : dict or DataFrame
            Must contain the by-variable and all smooth covariates.

        Returns
        -------
        np.ndarray
            Shape ``(n, n_levels * k_per_level)``.
        """
        by_col = _get_col(data, self.by_variable)
        n = len(by_col)

        # Get base design matrix for ALL observations
        X_base = self.base_smooth.build_design_matrix(data)

        # Store constraint from full-data base basis (matches R's smoothCon).
        # R computes C = colSums(X_base) BEFORE multiplying by the factor
        # indicator, so all levels share the same constraint direction.
        self._base_constraint = X_base.sum(axis=0)

        # Validate all levels have observations
        for level in self.levels:
            mask = np.asarray(by_col == level)
            if not np.any(mask):
                raise ValueError(
                    f"Factor level '{level}' of by-variable "
                    f"'{self.by_variable}' has zero observations. "
                    f"Remove empty levels or subset data before fitting."
                )

        # Build block-structured matrix
        X = np.zeros((n, self.n_coefs))
        for level_idx, level in enumerate(self.levels):
            mask = np.asarray(by_col == level)
            col_start = level_idx * self.k_per_level
            col_end = col_start + self.k_per_level
            # Only rows where fac==level get the basis values
            X[mask, col_start:col_end] = X_base[mask]

        return X

    def predict_matrix(
        self, new_data: dict[str, npt.NDArray[np.floating]] | pd.DataFrame
    ) -> npt.NDArray[np.floating]:
        """Build prediction matrix for new data.

        Parameters
        ----------
        new_data : dict or DataFrame
            Must contain the by-variable and all smooth covariates.

        Returns
        -------
        np.ndarray
            Shape ``(n_new, n_levels * k_per_level)``.
        """
        by_col = _get_col(new_data, self.by_variable)
        n = len(by_col)

        # Get base prediction matrix for ALL observations
        X_base = self.base_smooth.predict_matrix(new_data)

        X = np.zeros((n, self.n_coefs))
        for level_idx, level in enumerate(self.levels):
            mask = np.asarray(by_col == level)
            col_start = level_idx * self.k_per_level
            col_end = col_start + self.k_per_level
            X[mask, col_start:col_end] = X_base[mask]

        return X

    def build_penalty_matrices(self) -> list[Penalty]:
        """Build per-level penalty matrices.

        Each factor level gets its own copy of the base smooth's
        penalty matrices, embedded in the full coefficient space.
        Each penalty gets its own smoothing parameter λ in the
        REML outer loop.

        Returns
        -------
        list[Penalty]
            One penalty per (level, base_penalty) combination.
            Length = ``n_levels * n_base_penalties``.
        """
        base_penalties = self.base_smooth.build_penalty_matrices()
        penalties = []

        for level_idx in range(self.n_levels):
            col_start = level_idx * self.k_per_level
            for base_pen in base_penalties:
                # Embed the per-level penalty into the full space
                S_global = np.zeros((self.n_coefs, self.n_coefs))
                S_global[
                    col_start : col_start + self.k_per_level,
                    col_start : col_start + self.k_per_level,
                ] = base_pen.S
                penalties.append(
                    Penalty(
                        S_global,
                        rank=base_pen.rank,
                        null_space_dim=self.n_coefs - base_pen.rank,
                    )
                )

        return penalties

    def __repr__(self) -> str:
        return (
            f"FactorBySmooth(by={self.by_variable!r}, "
            f"levels={self.levels}, "
            f"k_per_level={self.k_per_level}, "
            f"n_coefs={self.n_coefs})"
        )


# ---------------------------------------------------------------------------
# NumericBySmooth
# ---------------------------------------------------------------------------


class NumericBySmooth:
    """Smooth expansion for ``s(x, by=z)`` where ``z`` is numeric.

    Multiplies the smooth basis pointwise by the numeric variable.
    The penalty is unchanged (penalizes wiggliness of ``f``, not ``z``).

    Parameters
    ----------
    base_smooth : Smooth
        An already-constructed base smooth (``setup()`` must have been
        called).
    spec : SmoothSpec
        The original smooth specification (with ``by`` set).
    by_variable : str
        Name of the numeric by-variable.
    """

    def __init__(
        self,
        base_smooth: Smooth,
        spec: SmoothSpec,
        by_variable: str,
    ) -> None:
        base_smooth._require_setup()

        self.base_smooth = base_smooth
        self.spec = spec
        self.by_variable = by_variable

        # Same coefficient structure as base
        self.n_coefs = base_smooth.n_coefs
        self.null_space_dim = base_smooth.null_space_dim
        self.rank = base_smooth.rank
        self._is_setup = True

        # Label (matching R: "s(x):z")
        base_label = _smooth_label(spec)
        self.label = f"{base_label}:{by_variable}"

        # Centering constraint is removed for non-constant numeric by
        # (R's behavior: the by variable breaks confounding with intercept)
        self.has_centering_constraint = False

    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]] | pd.DataFrame
    ) -> npt.NDArray[np.floating]:
        """Build design matrix with numeric-by multiplication.

        Parameters
        ----------
        data : dict or DataFrame
            Must contain the by-variable and all smooth covariates.

        Returns
        -------
        np.ndarray
            Shape ``(n, n_coefs)``. Each column of the base design
            matrix is multiplied by the numeric by-variable.
        """
        by_col = np.asarray(_get_col(data, self.by_variable), dtype=float)
        X_base = self.base_smooth.build_design_matrix(data)
        return by_col[:, np.newaxis] * X_base

    def predict_matrix(
        self, new_data: dict[str, npt.NDArray[np.floating]] | pd.DataFrame
    ) -> npt.NDArray[np.floating]:
        """Build prediction matrix for new data.

        Parameters
        ----------
        new_data : dict or DataFrame
            Must contain the by-variable and all smooth covariates.

        Returns
        -------
        np.ndarray
            Shape ``(n_new, n_coefs)``.
        """
        by_col = np.asarray(_get_col(new_data, self.by_variable), dtype=float)
        X_base = self.base_smooth.predict_matrix(new_data)
        return by_col[:, np.newaxis] * X_base

    def build_penalty_matrices(self) -> list[Penalty]:
        """Return base smooth's penalty matrices unchanged.

        Returns
        -------
        list[Penalty]
            Same penalties as the base smooth.
        """
        return self.base_smooth.build_penalty_matrices()

    def __repr__(self) -> str:
        return f"NumericBySmooth(by={self.by_variable!r}, n_coefs={self.n_coefs})"


# ---------------------------------------------------------------------------
# Factory / resolution
# ---------------------------------------------------------------------------


def resolve_by_variable(
    spec: SmoothSpec,
    data: dict[str, npt.NDArray[np.floating]] | pd.DataFrame,
    smooth: Smooth,
) -> Smooth | FactorBySmooth | NumericBySmooth:
    """Resolve the by-variable on a smooth spec into the correct wrapper.

    Called during model matrix assembly (Phase 1), not during parsing.

    Parameters
    ----------
    spec : SmoothSpec
        Smooth specification (from formula parser). ``spec.by`` is the
        by-variable name, or ``None`` for no by-variable.
    data : dict or DataFrame
        Data containing all variables.
    smooth : Smooth
        The base smooth (already set up).

    Returns
    -------
    Smooth or FactorBySmooth or NumericBySmooth
        The unwrapped smooth, or a by-variable wrapper.

    Raises
    ------
    ValueError
        If the by-variable is not found in the data, or has an
        unsupported type.
    """
    if spec.by is None:
        return smooth

    by_col = _get_col(data, spec.by)

    if is_factor(by_col):
        levels = _get_factor_levels(by_col)

        # Ordered factors: skip reference (first) level, matching R
        if _is_ordered_factor(by_col) and len(levels) > 1:
            levels = levels[1:]

        if len(levels) == 0:
            raise ValueError(
                f"Factor by-variable '{spec.by}' has no levels. "
                f"Cannot create factor-by smooth."
            )

        return FactorBySmooth(
            base_smooth=smooth,
            spec=spec,
            levels=levels,
            by_variable=spec.by,
        )
    else:
        # Numeric by
        by_arr = np.asarray(by_col, dtype=float)
        if np.isnan(by_arr).any():
            raise ValueError(f"Numeric by-variable '{spec.by}' contains NaN values.")
        return NumericBySmooth(
            base_smooth=smooth,
            spec=spec,
            by_variable=spec.by,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_col(
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
    else:
        if name not in data:
            raise KeyError(
                f"Variable '{name}' not found in data. Available: {list(data.keys())}"
            )
        return data[name]


def _smooth_label(spec: SmoothSpec) -> str:
    """Construct a smooth label from its specification.

    Parameters
    ----------
    spec : SmoothSpec
        Smooth specification.

    Returns
    -------
    str
        Label like ``"s(x)"`` or ``"te(x1,x2)"``.
    """
    vars_str = ",".join(spec.variables)
    return f"{spec.smooth_type}({vars_str})"
