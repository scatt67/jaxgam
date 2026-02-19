"""Design matrix assembly for GAM model setup.

Assembles the full model matrix X from a parsed formula and data,
applying identifiability constraints and embedding penalty matrices
into the global coefficient space. Python equivalent of R's ``gam.setup()``.

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 13.2
R source reference: R/gam.r gam.setup()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from pymgcv.formula.terms import FormulaSpec, ParametricTerm
from pymgcv.penalties.penalty import CompositePenalty, Penalty
from pymgcv.smooths.by_variable import (
    FactorBySmooth,
    NumericBySmooth,
    _get_factor_levels,
    is_factor,
    resolve_by_variable,
)
from pymgcv.smooths.constraints import CoefficientMap
from pymgcv.smooths.registry import get_smooth_class

if TYPE_CHECKING:
    from pymgcv.smooths.base import Smooth


@dataclass(frozen=True)
class SmoothInfo:
    """Per-smooth metadata in the assembled model.

    Parameters
    ----------
    label : str
        Human-readable label, e.g. ``"s(x1)"``, ``"te(x1,x2)"``.
    term_type : str
        One of ``"s"``, ``"te"``, ``"ti"``.
    variables : tuple[str, ...]
        Covariate names.
    by_variable : str | None
        Factor or numeric by-variable name, or None.
    first_coef : int
        Start column in the constrained model matrix X.
    last_coef : int
        End column (exclusive) in constrained X.
    n_penalties : int
        Number of penalty matrices for this smooth.
    first_penalty : int
        Index of first penalty in the global penalty list.
    null_space_dim : int
        Null space dimension from the smooth object.
    """

    label: str
    term_type: str
    variables: tuple[str, ...]
    by_variable: str | None
    first_coef: int
    last_coef: int
    n_penalties: int
    first_penalty: int
    null_space_dim: int


@dataclass(frozen=True)
class ModelSetup:
    """Assembled GAM model — the output of Phase 1 setup.

    Frozen dataclass containing the full constrained model matrix,
    response, penalties, and coefficient mapping. Created via the
    ``build()`` classmethod factory.

    Parameters
    ----------
    X : np.ndarray
        Full constrained model matrix, shape ``(n, total_p)``.
    y : np.ndarray
        Response vector, shape ``(n,)``.
    n_obs : int
        Number of observations.
    weights : np.ndarray
        Prior weights, shape ``(n,)``.
    offset : np.ndarray | None
        Offset vector, shape ``(n,)``, or None.
    penalties : CompositePenalty | None
        All penalties embedded in ``(total_p, total_p)`` space.
        None if model is purely parametric.
    coef_map : CoefficientMap
        Constraint mapping for predict/summary (Phase 3).
    smooth_info : tuple[SmoothInfo, ...]
        Per-smooth metadata.
    term_names : tuple[str, ...]
        Human-readable names, one per column of X.
    """

    X: npt.NDArray[np.floating]
    y: npt.NDArray[np.floating]
    n_obs: int
    weights: npt.NDArray[np.floating]
    offset: npt.NDArray[np.floating] | None
    penalties: CompositePenalty | None
    coef_map: CoefficientMap
    smooth_info: tuple[SmoothInfo, ...]
    term_names: tuple[str, ...]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        formula_spec: FormulaSpec,
        data: dict[str, npt.NDArray[np.floating]] | pd.DataFrame,
        weights: npt.NDArray[np.floating] | None = None,
        offset: npt.NDArray[np.floating] | None = None,
    ) -> ModelSetup:
        """Assemble the full model from a parsed formula and data.

        Parameters
        ----------
        formula_spec : FormulaSpec
            Parsed formula specification.
        data : dict or DataFrame
            Data containing all formula variables.
        weights : np.ndarray or None
            Prior weights. Defaults to ones.
        offset : np.ndarray or None
            Offset vector. Defaults to None.

        Returns
        -------
        ModelSetup
            Assembled model ready for Phase 2 fitting.

        Raises
        ------
        ValueError
            If required variables are missing from data.
        """
        # Keep original data for factor detection (by-variable needs dtype info)
        original_data = data
        data_dict = cls._to_dict(data)

        # 2a. Validate and extract response
        if formula_spec.response not in data_dict:
            raise ValueError(
                f"Response variable '{formula_spec.response}' not found in data. "
                f"Available: {list(data_dict.keys())}"
            )
        y = np.asarray(data_dict[formula_spec.response], dtype=np.float64).ravel()
        n_obs = len(y)

        # Validate all variables exist
        cls._validate_variables(formula_spec, data_dict, original_data)

        # Default weights and offset
        if weights is None:
            weights = np.ones(n_obs, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()

        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64).ravel()

        # 2b. Build parametric design matrix
        X_parametric, param_names = cls._build_parametric_matrix(
            formula_spec.parametric_terms,
            original_data,
            formula_spec.has_intercept,
            n_obs,
        )
        n_parametric = X_parametric.shape[1]

        # 2c. Construct smooth bases and resolve by-variables
        smooths, X_blocks, S_blocks = cls._build_smooth_components(
            formula_spec.smooth_terms, data_dict, original_data
        )

        # 2d. Apply constraints
        coef_map, X_constrained, S_constrained = CoefficientMap.build(
            smooths,
            X_blocks,
            S_blocks,
            has_intercept=formula_spec.has_intercept,
            n_parametric=n_parametric,
            X_parametric=X_parametric,
        )

        # 2e. Assemble full X
        if X_constrained:
            X = np.column_stack([X_parametric] + X_constrained)
        else:
            X = X_parametric
        assert X.shape[1] == coef_map.total_coefs

        # 2f. Embed penalties
        total_p = coef_map.total_coefs
        embedded_penalties: list[Penalty] = []

        for i, sm in enumerate(smooths):
            term_label = CoefficientMap._smooth_label(sm)
            term_block = coef_map.get_term(term_label)
            col_start = term_block.col_start

            for S_j in S_constrained[i]:
                S_global = CompositePenalty.embed(S_j, col_start, total_p)
                # Compute rank of the per-smooth penalty
                eigvals = np.linalg.eigvalsh(S_j)
                max_eigval = np.max(np.abs(eigvals)) if len(eigvals) > 0 else 0
                if max_eigval > 0:
                    tol = max_eigval * max(S_j.shape[0], 1) * np.finfo(float).eps
                    rank = int(np.sum(eigvals > tol))
                else:
                    rank = 0
                embedded_penalties.append(Penalty(S_global, rank=rank))

        if embedded_penalties:
            composite_penalty = CompositePenalty(embedded_penalties)
        else:
            composite_penalty = None

        # 2g. Build SmoothInfo and term_names
        smooth_infos = cls._build_smooth_info(smooths, coef_map)
        term_names = cls._build_term_names(param_names, smooths, coef_map)

        # 2h. Return frozen ModelSetup
        return cls(
            X=X,
            y=y,
            n_obs=n_obs,
            weights=weights,
            offset=offset,
            penalties=composite_penalty,
            coef_map=coef_map,
            smooth_info=tuple(smooth_infos),
            term_names=term_names,
        )

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def get_smooth(self, label: str) -> SmoothInfo:
        """Look up a smooth by label.

        Parameters
        ----------
        label : str
            Smooth label, e.g. ``"s(x1)"``.

        Returns
        -------
        SmoothInfo

        Raises
        ------
        KeyError
            If no smooth matches the label.
        """
        for info in self.smooth_info:
            if info.label == label:
                return info
        raise KeyError(
            f"No smooth '{label}'. Available: {[si.label for si in self.smooth_info]}"
        )

    def smooth_coef_slice(self, label: str) -> slice:
        """Return slice for a smooth's columns in X.

        Parameters
        ----------
        label : str
            Smooth label.

        Returns
        -------
        slice
        """
        info = self.get_smooth(label)
        return slice(info.first_coef, info.last_coef)

    def smooth_penalty_indices(self, label: str) -> range:
        """Return range of penalty indices for a smooth.

        Parameters
        ----------
        label : str
            Smooth label.

        Returns
        -------
        range
        """
        info = self.get_smooth(label)
        return range(info.first_penalty, info.first_penalty + info.n_penalties)

    # ------------------------------------------------------------------
    # Private static methods (pipeline steps)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(
        data: dict[str, npt.NDArray[np.floating]] | pd.DataFrame,
    ) -> dict[str, npt.NDArray[np.floating]]:
        """Convert data to dict of numpy arrays.

        Parameters
        ----------
        data : dict or DataFrame
            Input data.

        Returns
        -------
        dict[str, np.ndarray]
            Data as dict of arrays.
        """
        if isinstance(data, pd.DataFrame):
            result = {}
            for col in data.columns:
                if is_factor(data[col]):
                    result[col] = np.asarray(data[col])
                else:
                    result[col] = np.asarray(data[col], dtype=np.float64)
            return result
        return dict(data)

    @staticmethod
    def _validate_variables(
        formula_spec: FormulaSpec,
        data_dict: dict[str, npt.NDArray],
        original_data: dict | pd.DataFrame,
    ) -> None:
        """Validate all formula variables exist in data."""
        available = set(data_dict.keys())

        # Check parametric terms
        for term in formula_spec.parametric_terms:
            if term.name not in available:
                raise ValueError(
                    f"Parametric variable '{term.name}' not found in data. "
                    f"Available: {sorted(available)}"
                )

        # Check smooth terms
        for spec in formula_spec.smooth_terms:
            for var in spec.variables:
                if var not in available:
                    raise ValueError(
                        f"Smooth variable '{var}' not found in data. "
                        f"Available: {sorted(available)}"
                    )
            if spec.by is not None and spec.by not in available:
                raise ValueError(
                    f"By-variable '{spec.by}' not found in data. "
                    f"Available: {sorted(available)}"
                )

    @staticmethod
    def _encode_factor(
        col: npt.NDArray | pd.Series,
        levels: list,
        drop_reference: bool,
    ) -> tuple[npt.NDArray[np.floating], list[str]]:
        """Create dummy indicator matrix for a factor column.

        Parameters
        ----------
        col : array-like
            Factor column values.
        levels : list
            Ordered factor levels.
        drop_reference : bool
            If True, drop the first (reference) level column.

        Returns
        -------
        dummy_matrix : np.ndarray
            Shape ``(n, n_levels)`` or ``(n, n_levels - 1)``.
        level_names : list[str]
            Names for each dummy column.
        """
        col_arr = np.asarray(col)
        n = len(col_arr)
        n_levels = len(levels)

        dummy = np.zeros((n, n_levels), dtype=np.float64)
        for j, level in enumerate(levels):
            dummy[:, j] = (col_arr == level).astype(np.float64)

        if drop_reference:
            dummy = dummy[:, 1:]
            level_names = [str(lev) for lev in levels[1:]]
        else:
            level_names = [str(lev) for lev in levels]

        return dummy, level_names

    @staticmethod
    def _build_parametric_matrix(
        parametric_terms: list[ParametricTerm],
        data: dict[str, npt.NDArray] | pd.DataFrame,
        has_intercept: bool,
        n_obs: int,
    ) -> tuple[npt.NDArray[np.floating], list[str]]:
        """Build the parametric portion of the model matrix.

        Parameters
        ----------
        parametric_terms : list[ParametricTerm]
            Parametric terms from formula.
        data : dict or DataFrame
            Original data (preserving dtypes for factor detection).
        has_intercept : bool
            Whether to include an intercept column.
        n_obs : int
            Number of observations.

        Returns
        -------
        X_parametric : np.ndarray
            Shape ``(n, n_parametric_cols)``.
        param_names : list[str]
            Column names.
        """
        blocks: list[npt.NDArray[np.floating]] = []
        names: list[str] = []

        if has_intercept:
            blocks.append(np.ones((n_obs, 1), dtype=np.float64))
            names.append("(Intercept)")

        for term in parametric_terms:
            if isinstance(data, pd.DataFrame):
                col = data[term.name]
            else:
                col = data[term.name]

            if is_factor(col):
                levels = _get_factor_levels(col)
                if len(levels) < 2:
                    raise ValueError(
                        f"Factor variable '{term.name}' has fewer than 2 levels "
                        f"({levels}). Cannot create dummy variables."
                    )
                drop_ref = has_intercept
                dummy, level_names = ModelSetup._encode_factor(
                    col, levels, drop_reference=drop_ref
                )
                blocks.append(dummy)
                for lev in level_names:
                    names.append(f"{term.name}{lev}")
            else:
                col_arr = np.asarray(col, dtype=np.float64).ravel()
                blocks.append(col_arr[:, np.newaxis])
                names.append(term.name)

        if blocks:
            X_parametric = np.column_stack(blocks)
        else:
            X_parametric = np.empty((n_obs, 0), dtype=np.float64)

        return X_parametric, names

    @staticmethod
    def _build_smooth_components(
        smooth_terms: list,
        data_dict: dict[str, npt.NDArray[np.floating]],
        original_data: dict | pd.DataFrame,
    ) -> tuple[
        list[Smooth | FactorBySmooth | NumericBySmooth],
        list[npt.NDArray[np.floating]],
        list[list[npt.NDArray[np.floating]]],
    ]:
        """Construct smooth bases and resolve by-variables.

        Parameters
        ----------
        smooth_terms : list[SmoothSpec]
            Smooth specifications from formula.
        data_dict : dict
            Data as dict of numpy arrays.
        original_data : dict or DataFrame
            Original data (for factor detection in by-variables).

        Returns
        -------
        smooths : list
            Smooth objects (possibly wrapped in FactorBySmooth/NumericBySmooth).
        X_blocks : list[np.ndarray]
            Raw design matrices per smooth.
        S_blocks : list[list[np.ndarray]]
            Raw penalty matrices per smooth.
        """
        smooths: list[Smooth | FactorBySmooth | NumericBySmooth] = []
        X_blocks: list[npt.NDArray[np.floating]] = []
        S_blocks: list[list[npt.NDArray[np.floating]]] = []

        for spec in smooth_terms:
            # Registry key: smooth_type for te/ti, else bs
            if spec.smooth_type in ("te", "ti"):
                key = spec.smooth_type
            else:
                key = spec.bs

            smooth_cls = get_smooth_class(key)
            smooth = smooth_cls(spec)
            smooth.setup(data_dict)

            # Resolve by-variable
            smooth = resolve_by_variable(spec, original_data, smooth)

            # Build design matrix and penalty matrices
            X_s = smooth.build_design_matrix(data_dict)
            penalties = smooth.build_penalty_matrices()
            S_s = [p.S for p in penalties]

            smooths.append(smooth)
            X_blocks.append(X_s)
            S_blocks.append(S_s)

        return smooths, X_blocks, S_blocks

    @staticmethod
    def _build_smooth_info(
        smooths: list[Smooth | FactorBySmooth | NumericBySmooth],
        coef_map: CoefficientMap,
    ) -> list[SmoothInfo]:
        """Build SmoothInfo for each smooth from coef_map.

        Parameters
        ----------
        smooths : list
            Smooth objects.
        coef_map : CoefficientMap
            The coefficient map with term blocks.

        Returns
        -------
        list[SmoothInfo]
        """
        infos: list[SmoothInfo] = []
        for sm in smooths:
            label = CoefficientMap._smooth_label(sm)
            term = coef_map.get_term(label)

            if isinstance(sm, FactorBySmooth):
                by_var: str | None = sm.by_variable
                term_type = sm.spec.smooth_type
                variables = tuple(sm.spec.variables)
                null_space_dim = sm.null_space_dim
            elif isinstance(sm, NumericBySmooth):
                by_var = sm.by_variable
                term_type = sm.spec.smooth_type
                variables = tuple(sm.spec.variables)
                null_space_dim = sm.null_space_dim
            else:
                by_var = None
                term_type = sm.spec.smooth_type
                variables = tuple(sm.spec.variables)
                null_space_dim = sm.null_space_dim

            infos.append(
                SmoothInfo(
                    label=label,
                    term_type=term_type,
                    variables=variables,
                    by_variable=by_var,
                    first_coef=term.col_start,
                    last_coef=term.col_start + term.n_coefs,
                    n_penalties=len(term.penalty_indices),
                    first_penalty=(
                        term.penalty_indices[0] if term.penalty_indices else 0
                    ),
                    null_space_dim=null_space_dim,
                )
            )

        return infos

    @staticmethod
    def _build_term_names(
        param_names: list[str],
        smooths: list[Smooth | FactorBySmooth | NumericBySmooth],
        coef_map: CoefficientMap,
    ) -> tuple[str, ...]:
        """Build human-readable names for each column in X.

        Parameters
        ----------
        param_names : list[str]
            Names for parametric columns.
        smooths : list
            Smooth objects.
        coef_map : CoefficientMap
            Coefficient map with term blocks.

        Returns
        -------
        tuple[str, ...]
            One name per column of X.
        """
        names: list[str] = list(param_names)

        for sm in smooths:
            label = CoefficientMap._smooth_label(sm)
            term = coef_map.get_term(label)
            for j in range(term.n_coefs):
                names.append(f"{label}.{j + 1}")

        return tuple(names)
