"""Phase-1 prediction state and constrained matrix construction.

This module contains the NumPy-only boundary used to build prediction
matrices.  ``PredictSpec`` deliberately retains only setup metadata needed
for prediction; training design and penalty caches are removed from its
smooth graph by :func:`build_predict_spec`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from jaxgam.formula.terms import ParametricTerm
from jaxgam.smooths.constraints import CoefficientMap
from jaxgam.smooths.utils import (
    get_factor_levels,
    is_factor,
    is_ordered_factor,
)

if TYPE_CHECKING:
    from jaxgam.formula.design import ModelSetup, SmoothInfo


Data = dict[str, npt.NDArray[Any]] | pd.DataFrame


@dataclass(frozen=True)
class PredictSpec:
    """The observation-independent Phase-1 state required for prediction."""

    coef_map: CoefficientMap
    smooth_info: tuple[SmoothInfo, ...]
    parametric_terms: tuple[ParametricTerm, ...]
    factor_info: dict[str, list]
    ordered_factors: frozenset[str]
    has_intercept: bool
    parametric_keep_cols: tuple[int, ...]
    dropped_param_names: tuple[str, ...]
    total_coefs: int

    def build_predict_matrix(self, newdata: Data) -> npt.NDArray[np.floating]:
        """Build the full constrained prediction matrix for ``newdata``."""
        return build_predict_matrix(self, newdata)


def build_predict_spec(setup: ModelSetup) -> PredictSpec:
    """Create prediction-only state without mutating the live model setup.

    Constraint arrays and prediction transforms are shared by reference.  Each
    smooth is shallow-copied through its polymorphic ``copy_for_prediction``
    hook, which removes training design and penalty caches.
    """
    terms = tuple(
        dataclasses.replace(term, smooth=term.smooth.copy_for_prediction())
        if term.smooth is not None
        else term
        for term in setup.coef_map.terms
    )
    coef_map = dataclasses.replace(setup.coef_map, terms=terms)
    return PredictSpec(
        coef_map=coef_map,
        smooth_info=setup.smooth_info,
        parametric_terms=setup.parametric_terms,
        factor_info=setup.factor_info,
        ordered_factors=setup.ordered_factors,
        has_intercept=setup.has_intercept,
        parametric_keep_cols=setup.parametric_keep_cols,
        dropped_param_names=setup.dropped_param_names,
        total_coefs=setup.coef_map.total_coefs,
    )


def build_predict_matrix(
    spec: PredictSpec,
    newdata: Data,
) -> npt.NDArray[np.floating]:
    """Build a constrained prediction matrix from a concrete ``PredictSpec``."""
    data_dict = _to_dict(newdata)

    if not data_dict:
        raise ValueError("newdata is empty — no variables found.")

    first_key = next(iter(data_dict))
    n_new = len(np.asarray(data_dict[first_key]).ravel())

    pred_names: list[str] = [term.name for term in spec.parametric_terms]
    for smooth_info in spec.smooth_info:
        pred_names.extend(smooth_info.variables)
        if smooth_info.by_variable is not None:
            pred_names.append(smooth_info.by_variable)
    _validate_equal_lengths(
        pred_names,
        data_dict,
        n_new,
        first_key,
        context="in newdata",
    )

    X_parametric, _ = _build_parametric_matrix(
        spec.parametric_terms,
        newdata,
        spec.has_intercept,
        n_new,
        factor_info=spec.factor_info,
        ordered_factors=spec.ordered_factors,
    )
    # Reproduce the training-time aliased-column drop so prediction has the
    # same reduced parametric block as the fitted model.
    if spec.dropped_param_names:
        X_parametric = X_parametric[:, list(spec.parametric_keep_cols)]

    blocks: list[npt.NDArray[np.floating]] = [X_parametric]
    coef_map = spec.coef_map
    for term in coef_map.terms:
        if term.term_type == "parametric":
            continue
        X_raw = term.smooth.predict_matrix(data_dict)
        # Pass the TermBlock, not term.label: label-colliding smooths must use
        # their own Z_centering/del_index rather than the first label match.
        blocks.append(coef_map.transform_X(X_raw, term))

    X_p = np.column_stack(blocks) if len(blocks) > 1 else blocks[0]
    if X_p.shape[1] != spec.total_coefs:
        raise RuntimeError(
            f"Prediction matrix has {X_p.shape[1]} columns but model "
            f"expects {spec.total_coefs}."
        )
    return X_p


def _to_dict(data: Data) -> dict[str, npt.NDArray[Any]]:
    """Convert supported data containers to a mapping of array-like columns.

    Factor columns remain categorical-preserving Series objects. Converting an
    integer ``pd.Categorical`` with ``np.asarray`` demotes it to ``int64`` and
    destroys factor identity, corrupting uses such as ``s(g, bs='re')``.
    Resetting the index also preserves positional alignment for factor masks.
    """
    if isinstance(data, pd.DataFrame):
        result = {}
        for col in data.columns:
            if is_factor(data[col]):
                # Preserve categorical identity and positional mask alignment.
                result[col] = data[col].reset_index(drop=True)
            else:
                result[col] = np.asarray(data[col], dtype=np.float64)
        return result
    return dict(data)


def _validate_equal_lengths(
    names: list[str],
    data_dict: dict[str, npt.NDArray[Any]],
    expected: int,
    ref_name: str,
    context: str = "",
) -> None:
    """Check that each referenced, present variable has one common length.

    Raises a variable-named error instead of allowing a later
    ``np.column_stack`` call to fail with an opaque dimension mismatch.
    """
    where = f" {context}" if context else ""
    for name in dict.fromkeys(names):
        if name not in data_dict:
            continue
        length = len(np.asarray(data_dict[name]).ravel())
        if length != expected:
            raise ValueError(
                f"Variable '{name}'{where} has {length} element(s) but "
                f"'{ref_name}' has {expected}; all variables must share "
                f"one length."
            )


def _contr_poly(n_levels: int) -> tuple[npt.NDArray[np.floating], list[str]]:
    """Build orthogonal-polynomial contrasts matching R's ``contr.poly``.

    Returns an ``(n_levels, n_levels - 1)`` matrix with the constant column
    dropped and R-compatible suffixes (``.L``, ``.Q``, ``.C``, ``^4``, ...).
    """
    scores = np.arange(1, n_levels + 1, dtype=np.float64)
    centered = scores - scores.mean()
    vander = np.vander(centered, n_levels, increasing=True)
    q, r = np.linalg.qr(vander)
    raw = q * np.diag(r)
    norms = np.sqrt((raw**2).sum(axis=0))
    contrasts = (raw / norms)[:, 1:]
    suffixes = [f"^{i}" for i in range(n_levels)]
    for pos in range(1, min(4, n_levels)):
        suffixes[pos] = [".L", ".Q", ".C"][pos - 1]
    return contrasts, suffixes[1:]


def _encode_factor(
    col: npt.ArrayLike | pd.Series,
    levels: list,
    drop_reference: bool,
    ordered: bool = False,
) -> tuple[npt.NDArray[np.floating], list[str]]:
    """Encode a factor with treatment or ordered-polynomial contrasts.

    NA factor values produce all-NaN rows, matching ``predict.gam`` rather than
    being silently encoded as the reference level.
    """
    col_arr = np.asarray(col, dtype=object)
    n = len(col_arr)
    n_levels = len(levels)
    na_mask = pd.isna(col_arr)

    if ordered and drop_reference and n_levels >= 2:
        contrasts, suffixes = _contr_poly(n_levels)
        level_index = {level: i for i, level in enumerate(levels)}
        codes = np.array(
            [level_index.get(value, -1) for value in col_arr], dtype=np.intp
        )
        dummy = np.full((n, n_levels - 1), np.nan, dtype=np.float64)
        valid = (codes >= 0) & ~na_mask
        dummy[valid] = contrasts[codes[valid]]
        return dummy, suffixes

    dummy = np.zeros((n, n_levels), dtype=np.float64)
    for j, level in enumerate(levels):
        dummy[:, j] = (col_arr == level).astype(np.float64)
    # R predict.gam propagates an NA factor value to an NA prediction.
    dummy[na_mask, :] = np.nan

    if drop_reference:
        return dummy[:, 1:], [str(level) for level in levels[1:]]
    return dummy, [str(level) for level in levels]


def _build_parametric_matrix(
    parametric_terms: list[ParametricTerm] | tuple[ParametricTerm, ...],
    data: Data,
    has_intercept: bool,
    n_obs: int,
    factor_info: dict[str, list] | None = None,
    ordered_factors: frozenset[str] | None = None,
) -> tuple[npt.NDArray[np.floating], list[str]]:
    """Build the training or prediction-time parametric matrix block.

    Training auto-detects factors; prediction uses the stored levels and
    orderedness so encoding is independent of the newdata dtype.
    """
    blocks: list[npt.NDArray[np.floating]] = []
    names: list[str] = []

    if has_intercept:
        blocks.append(np.ones((n_obs, 1), dtype=np.float64))
        names.append("(Intercept)")

    # Without an intercept, R full-codes only the first factor (which absorbs
    # the missing intercept) and treatment-codes subsequent factors. This keeps
    # multi-factor parametric blocks full rank.
    seen_factor = False
    for term in parametric_terms:
        col = data[term.name]

        if factor_info is not None:
            is_fac = term.name in factor_info
            levels = factor_info.get(term.name)
            ordered = ordered_factors is not None and term.name in ordered_factors
        else:
            is_fac = is_factor(col)
            levels = get_factor_levels(col) if is_fac else None
            ordered = is_fac and is_ordered_factor(col)

        if is_fac:
            if factor_info is None and len(levels) < 2:
                raise ValueError(
                    f"Factor variable '{term.name}' has fewer than 2 levels "
                    f"({levels}). Cannot create dummy variables."
                )
            if factor_info is not None:
                # Reject new levels like predict.gam rather than silently
                # treating them as the reference level.
                known = set(levels)
                col_obj = np.asarray(col, dtype=object)
                # Mask NA before np.unique: mixed strings and NaN cannot be
                # ordered, and NA is not a new factor level.
                na_mask = pd.isna(col_obj)
                observed = np.unique(col_obj[~na_mask]).tolist()
                unseen = sorted(
                    {value for value in observed if value not in known}, key=str
                )
                if unseen:
                    raise ValueError(
                        f"Parametric factor '{term.name}' has new level(s) "
                        f"{unseen} not seen during fitting. Predictions for "
                        f"unseen levels of a parametric factor are undefined."
                    )
            drop_reference = has_intercept or seen_factor
            seen_factor = True
            dummy, level_names = _encode_factor(
                col,
                levels,
                drop_reference=drop_reference,
                ordered=ordered,
            )
            blocks.append(dummy)
            names.extend(f"{term.name}{level}" for level in level_names)
        else:
            col_arr = np.asarray(col, dtype=np.float64).ravel()
            blocks.append(col_arr[:, np.newaxis])
            names.append(term.name)

    X_parametric = (
        np.column_stack(blocks) if blocks else np.empty((n_obs, 0), dtype=np.float64)
    )
    return X_parametric, names
