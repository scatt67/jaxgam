"""Design matrix assembly for GAM model setup.

Assembles the full model matrix X from a parsed formula and data,
applying identifiability constraints and embedding penalty matrices
into the global coefficient space. Python equivalent of R's ``gam.setup()``.

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 13.2
R source reference: R/gam.r gam.setup()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import linalg

from jaxgam.formula import predict_matrix
from jaxgam.formula.terms import FormulaSpec, ParametricTerm, SmoothSpec
from jaxgam.penalties.penalty import CompositePenalty, Penalty
from jaxgam.smooths.by_variable import (
    FactorBySmooth,
    NumericBySmooth,
    resolve_by_variable,
)
from jaxgam.smooths.constraints import CoefficientMap
from jaxgam.smooths.registry import get_smooth_class
from jaxgam.smooths.utils import get_factor_levels, is_factor, is_ordered_factor

if TYPE_CHECKING:
    from jaxgam.smooths.base import Smooth


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
    is_random : bool
        True for random effect terms (``bs="re"``).  Controls the
        p-value test type in ``summary()`` (``type_=1`` instead of 0).
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
    is_random: bool = False


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
    factor_info : dict[str, list]
        Mapping from factor variable name to ordered levels,
        captured at training time for prediction-time encoding.
    ordered_factors : frozenset[str]
        Names of parametric factor terms that are *ordered* categoricals
        (encoded with ``contr.poly`` contrasts). Captured at training time so
        prediction reproduces the contrasts independent of the newdata dtype.
    has_intercept : bool
        Whether the model includes an intercept.
    parametric_terms : tuple[ParametricTerm, ...]
        Parametric terms from the formula specification.
    parametric_keep_cols : tuple[int, ...]
        Column indices of the *full* parametric block retained after dropping
        aliased/rank-deficient columns (pivoted-QR rank reduction). Empty means
        no reduction was needed; used at predict time to reproduce the drop.
    dropped_param_names : tuple[str, ...]
        Names of parametric columns dropped as aliased (reported NA in
        ``summary()``), matching mgcv's rank-deficient handling.
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
    factor_info: dict[str, list]
    ordered_factors: frozenset[str]
    has_intercept: bool
    parametric_terms: tuple[ParametricTerm, ...]
    parametric_keep_cols: tuple[int, ...] = ()
    dropped_param_names: tuple[str, ...] = ()
    _predict_spec_cache: predict_matrix.PredictSpec | None = field(
        init=False, default=None, compare=False, repr=False
    )

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
        # Steps are numbered 2a-2h matching the design doc (Section 13.2).
        # Step 1 (formula parsing) happens in parse_formula().

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

        # Validate data quality
        if not np.all(np.isfinite(y)):
            n_nan = np.sum(np.isnan(y))
            n_inf = np.sum(np.isinf(y))
            parts = []
            if n_nan:
                parts.append(f"{n_nan} NaN")
            if n_inf:
                parts.append(f"{n_inf} Inf")
            raise ValueError(
                f"Response variable '{formula_spec.response}' contains "
                f"non-finite values ({', '.join(parts)}). "
                f"Remove or impute missing values before fitting."
            )

        if n_obs < 2:
            raise ValueError(
                f"Data has {n_obs} observation(s). "
                f"At least 2 observations are required to fit a GAM."
            )

        # Validate all variables exist
        cls._validate_variables(formula_spec, data_dict)

        # Validate all referenced variables share the response's length, so a
        # short/long covariate raises a clear, variable-named error rather than
        # a cryptic np.column_stack dimension mismatch downstream.
        ref_names: list[str] = [t.name for t in formula_spec.parametric_terms]
        for spec in formula_spec.smooth_terms:
            ref_names.extend(spec.variables)
            if spec.by is not None:
                ref_names.append(spec.by)
        cls._validate_equal_lengths(ref_names, data_dict, n_obs, formula_spec.response)

        # Default weights and offset
        if weights is None:
            weights = np.ones(n_obs, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()
            cls._validate_vector(weights, n_obs, "weights", non_negative=True)
            # Total weight mass must be positive: all-zero (or empty) weights
            # leave the model with no information, producing a degenerate
            # "converged" fit and a NaN null deviance. mgcv errors during
            # smoothing-parameter setup; reject it up front with a clear message.
            if np.sum(weights) <= 0:
                raise ValueError(
                    "weights sum to zero: at least one observation must have a "
                    "positive prior weight."
                )

        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64).ravel()
            cls._validate_vector(offset, n_obs, "offset")

        # Validate covariate values (response is validated above; covariates
        # are validated here so non-finite predictors raise a clear error
        # rather than a downstream LinAlgError).
        cls._validate_finite_covariates(formula_spec, original_data, data_dict)

        # 2b. Build parametric design matrix, then drop any exactly-aliased
        # (rank-deficient) parametric columns so the coefficient table is
        # identifiable and matches mgcv (which reports dropped columns as NA).
        X_parametric, param_names = cls._build_parametric_matrix(
            formula_spec.parametric_terms,
            original_data,
            formula_spec.has_intercept,
            n_obs,
        )
        X_parametric, param_names, dropped_param_names, parametric_keep_cols = (
            cls._drop_aliased_parametric_columns(
                X_parametric, param_names, formula_spec.has_intercept
            )
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
            X = np.column_stack([X_parametric, *X_constrained])
        else:
            X = X_parametric
        if X.shape[1] != coef_map.total_coefs:
            raise RuntimeError(
                f"Model matrix column count ({X.shape[1]}) does not match "
                f"coefficient map total ({coef_map.total_coefs})"
            )

        # 2f. Embed penalties. Look the smooth's columns up POSITIONALLY (the
        # i-th smooth term block), never by label — two smooths can share a
        # label (s(x,k=6) + s(x,k=8)) and a label lookup would embed the second
        # smooth's penalty on the first smooth's columns.
        total_p = coef_map.total_coefs
        embedded_penalties: list[Penalty] = []
        smooth_blocks = [t for t in coef_map.terms if t.term_type == "smooth"]

        for i, _sm in enumerate(smooths):
            col_start = smooth_blocks[i].col_start

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

        # 2g+. Extract factor levels for prediction-time encoding
        factor_info = cls._extract_factor_info(
            formula_spec.parametric_terms, original_data
        )
        ordered_factors = frozenset(
            t.name
            for t in formula_spec.parametric_terms
            if t.name in factor_info and is_ordered_factor(original_data[t.name])
        )

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
            factor_info=factor_info,
            ordered_factors=ordered_factors,
            has_intercept=formula_spec.has_intercept,
            parametric_terms=tuple(formula_spec.parametric_terms),
            parametric_keep_cols=tuple(parametric_keep_cols),
            dropped_param_names=tuple(dropped_param_names),
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

    def build_predict_matrix(
        self,
        newdata: dict[str, npt.NDArray[np.floating]] | pd.DataFrame,
    ) -> npt.NDArray[np.floating]:
        """Build the full constrained prediction matrix for new data.

        Uses stored factor levels from training time for consistent
        parametric encoding. Equivalent to the matrix-building portion
        of R's ``predict.gam(type="lpmatrix")``.

        Parameters
        ----------
        newdata : DataFrame or dict
            New data containing all required variables.

        Returns
        -------
        np.ndarray, shape ``(n_new, total_coefs)``
        """
        return predict_matrix.build_predict_matrix(self._lazy_predict_spec(), newdata)

    def _lazy_predict_spec(self) -> predict_matrix.PredictSpec:
        """Return the cached prediction-only state, building it on first use."""
        spec = self._predict_spec_cache
        if spec is None:
            spec = predict_matrix.build_predict_spec(self)
            object.__setattr__(self, "_predict_spec_cache", spec)
        return spec

    # ------------------------------------------------------------------
    # Private static methods (pipeline steps)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(
        data: dict[str, npt.NDArray[np.floating]] | pd.DataFrame,
    ) -> dict[str, npt.NDArray[np.floating]]:
        """Delegate to the shared Phase-1 data conversion helper."""
        return predict_matrix._to_dict(data)

    @staticmethod
    def _validate_variables(
        formula_spec: FormulaSpec,
        data_dict: dict[str, npt.NDArray],
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
    def _validate_vector(
        vec: npt.NDArray[np.floating],
        n_obs: int,
        name: str,
        non_negative: bool = False,
    ) -> None:
        """Validate a per-observation vector (``weights`` or ``offset``).

        Guards against silently-broadcast wrong-length inputs and
        non-finite values, which otherwise surface as cryptic downstream
        errors or NaN coefficients.
        """
        if vec.shape[0] != n_obs:
            raise ValueError(
                f"{name} has {vec.shape[0]} element(s) but data has "
                f"{n_obs} observations; expected shape ({n_obs},)."
            )
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf).")
        if non_negative and np.any(vec < 0):
            raise ValueError(f"{name} must be non-negative.")

    @staticmethod
    def _validate_equal_lengths(
        names: list[str],
        data_dict: dict[str, npt.NDArray],
        expected: int,
        ref_name: str,
        context: str = "",
    ) -> None:
        """Delegate to the shared Phase-1 length validation helper."""
        predict_matrix._validate_equal_lengths(
            names, data_dict, expected, ref_name, context
        )

    @staticmethod
    def _validate_finite_covariates(
        formula_spec: FormulaSpec,
        original_data: dict[str, npt.NDArray] | pd.DataFrame,
        data_dict: dict[str, npt.NDArray],
    ) -> None:
        """Check that numeric covariate columns contain no NaN/Inf.

        Factor (categorical/string) columns are skipped — only numeric
        predictors are checked. Mirrors the response-variable validation
        so a non-finite predictor raises a clear error at setup time.
        """
        names: list[str] = [t.name for t in formula_spec.parametric_terms]
        for spec in formula_spec.smooth_terms:
            names.extend(spec.variables)
            if spec.by is not None:
                names.append(spec.by)

        for name in dict.fromkeys(names):  # de-dup, preserve order
            col = original_data[name]
            if is_factor(col):
                continue
            values = np.asarray(data_dict[name], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                n_nan = int(np.sum(np.isnan(values)))
                n_inf = int(np.sum(np.isinf(values)))
                parts = []
                if n_nan:
                    parts.append(f"{n_nan} NaN")
                if n_inf:
                    parts.append(f"{n_inf} Inf")
                raise ValueError(
                    f"Covariate '{name}' contains non-finite values "
                    f"({', '.join(parts)}). Remove or impute missing values "
                    f"before fitting."
                )

    @staticmethod
    def _extract_factor_info(
        parametric_terms: list[ParametricTerm] | tuple[ParametricTerm, ...],
        data: dict[str, npt.NDArray] | pd.DataFrame,
    ) -> dict[str, list]:
        """Extract factor level info from parametric terms at training time.

        Parameters
        ----------
        parametric_terms : list or tuple of ParametricTerm
            Parametric terms from the formula.
        data : DataFrame or dict
            Training data.

        Returns
        -------
        dict[str, list]
            Mapping from factor variable name to its ordered levels.
        """
        factor_info: dict[str, list] = {}
        for term in parametric_terms:
            col = data[term.name]
            if is_factor(col):
                factor_info[term.name] = get_factor_levels(col)
        return factor_info

    @staticmethod
    def _contr_poly(n_levels: int) -> tuple[npt.NDArray[np.floating], list[str]]:
        """Delegate to the shared Phase-1 ordered-contrast helper."""
        return predict_matrix._contr_poly(n_levels)

    @staticmethod
    def _build_parametric_matrix(
        parametric_terms: list[ParametricTerm] | tuple[ParametricTerm, ...],
        data: dict[str, npt.NDArray] | pd.DataFrame,
        has_intercept: bool,
        n_obs: int,
        factor_info: dict[str, list] | None = None,
        ordered_factors: frozenset[str] | None = None,
    ) -> tuple[npt.NDArray[np.floating], list[str]]:
        """Delegate to the shared Phase-1 parametric matrix builder."""
        return predict_matrix._build_parametric_matrix(
            parametric_terms,
            data,
            has_intercept,
            n_obs,
            factor_info,
            ordered_factors,
        )

    @staticmethod
    def _drop_aliased_parametric_columns(
        X_parametric: npt.NDArray[np.floating],
        param_names: list[str],
        has_intercept: bool,
    ) -> tuple[npt.NDArray[np.floating], list[str], list[str], list[int]]:
        """Drop exactly-aliased (rank-deficient) parametric columns.

        Mirrors mgcv, which builds the full parametric ``model.matrix`` then
        drops rank-deficient columns in the estimator via a pivoted QR with the
        ``Rrank`` tolerance (``eps**0.9``), reporting the dropped coefficients as
        NA (R/mgcv.r:4-16, 863-919). LAPACK column pivoting keeps the earlier
        column of an aliased pair and drops the later one. The intercept is never
        dropped.

        Returns
        -------
        X_reduced, names_reduced, dropped_names, keep_cols
            ``keep_cols`` are indices into the original full block.
        """
        ncol = X_parametric.shape[1]
        if ncol <= 1:
            return X_parametric, param_names, [], list(range(ncol))

        _, r, piv = linalg.qr(X_parametric, pivoting=True, mode="economic")
        rdiag = np.abs(np.diag(r))
        tol = rdiag[0] * np.finfo(float).eps ** 0.9 if rdiag.size else 0.0
        rank = int(np.sum(rdiag > tol))
        if rank >= ncol:
            return X_parametric, param_names, [], list(range(ncol))

        dropped = {int(p) for p in piv[rank:]}
        # Never drop the intercept (column 0 of a with-intercept block): if it
        # pivoted out, retain it and drop the next-pivoted dependent column.
        if has_intercept and 0 in dropped:
            dropped.discard(0)
            for p in piv[:rank][::-1]:
                if int(p) != 0:
                    dropped.add(int(p))
                    break

        keep_cols = sorted(set(range(ncol)) - dropped)
        dropped_names = [param_names[i] for i in sorted(dropped)]
        names_reduced = [param_names[i] for i in keep_cols]
        return X_parametric[:, keep_cols], names_reduced, dropped_names, keep_cols

    @staticmethod
    def _build_smooth_components(
        smooth_terms: list[SmoothSpec],
        data_dict: dict[str, npt.NDArray[np.floating]],
        original_data: dict[str, npt.NDArray] | pd.DataFrame,
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
            key = spec.smooth_type if spec.smooth_type in ("te", "ti") else spec.bs

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
        # Pair each smooth with its term block POSITIONALLY (the i-th smooth
        # term block), never by label: two smooths can share a label and a label
        # lookup returns the first match, corrupting every other smooth's offsets.
        smooth_blocks = [t for t in coef_map.terms if t.term_type == "smooth"]
        for sm, term in zip(smooths, smooth_blocks, strict=True):
            by_var: str | None = getattr(sm, "by_variable", None)
            base = getattr(sm, "base_smooth", sm)
            is_random = getattr(base, "_random", False)

            if isinstance(sm, FactorBySmooth):
                # mgcv replicates a factor-by smooth once PER LEVEL (one
                # object$smooth entry with its own label, sp, and EDF per level;
                # R/smooth.r:3969-3991). Mirror that here so EDF/summary are
                # per-level. The fit/penalty grouping stays per-smooth (Phase 2
                # iterates coef_map.terms, not smooth_info — see fitting/data.py).
                n_levels = sm.n_levels
                if (
                    term.n_coefs % n_levels != 0
                    or len(term.penalty_indices) % n_levels != 0
                ):
                    raise RuntimeError(
                        "factor-by block is not evenly divisible by n_levels; "
                        "per-level SmoothInfo cannot be derived."
                    )
                per_level_coefs = term.n_coefs // n_levels
                n_base_pen = len(term.penalty_indices) // n_levels
                for level_idx in range(n_levels):
                    start = term.col_start + level_idx * per_level_coefs
                    infos.append(
                        SmoothInfo(
                            label=sm.labels[level_idx],
                            term_type=sm.spec.smooth_type,
                            variables=tuple(sm.spec.variables),
                            by_variable=by_var,
                            first_coef=start,
                            last_coef=start + per_level_coefs,
                            n_penalties=n_base_pen,
                            first_penalty=term.penalty_indices[level_idx * n_base_pen],
                            null_space_dim=base.null_space_dim,
                            is_random=is_random,
                        )
                    )
            else:
                infos.append(
                    SmoothInfo(
                        label=CoefficientMap.smooth_label(sm),
                        term_type=sm.spec.smooth_type,
                        variables=tuple(sm.spec.variables),
                        by_variable=by_var,
                        first_coef=term.col_start,
                        last_coef=term.col_start + term.n_coefs,
                        n_penalties=len(term.penalty_indices),
                        first_penalty=(
                            term.penalty_indices[0] if term.penalty_indices else 0
                        ),
                        null_space_dim=sm.null_space_dim,
                        is_random=is_random,
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

        # Positional pairing (not label lookup) so label-colliding smooths get
        # the correct per-smooth column counts.
        smooth_blocks = [t for t in coef_map.terms if t.term_type == "smooth"]
        for sm, term in zip(smooths, smooth_blocks, strict=True):
            label = CoefficientMap.smooth_label(sm)
            names.extend(f"{label}.{j + 1}" for j in range(term.n_coefs))

        return tuple(names)
