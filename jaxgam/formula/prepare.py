"""Observation-independent preparation for the initial streamed design path.

The initial exact path is intentionally narrow: additive univariate cubic
smooths and numeric parametric terms.  It scans rows repeatedly for global
reductions and never routes through ``ModelSetup.build``.
"""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import linalg

from jaxgam.data.source import RowBatch, RowSource
from jaxgam.formula import predict_matrix
from jaxgam.formula.design import ModelSetup, SmoothInfo
from jaxgam.formula.fitting_prepare import (
    FittingPreparation,
    ResponseReduction,
    apply_transforms_to_design,
    initial_log_sp_from_diagonal,
    reparameterize_structure,
    total_penalty_spaces,
)
from jaxgam.formula.terms import FormulaSpec
from jaxgam.penalties.structure import PenaltyStructure, make_penalty_structure
from jaxgam.smooths.constraints import CoefficientMap, TermBlock
from jaxgam.smooths.cubic import CubicRegressionSmooth, CubicShrinkageSmooth
from jaxgam.smooths.utils import is_factor


@dataclass(frozen=True)
class PreparedModel:
    """Frozen setup metadata with no observation-aligned training arrays."""

    predict_spec: predict_matrix.PredictSpec
    penalties: PenaltyStructure | None
    source_fingerprint: str
    basis_fingerprint: str
    n_obs: int
    response: str
    fitting: FittingPreparation | None = None

    @property
    def n_coef(self) -> int:
        return self.predict_spec.total_coefs

    def evaluate_batch(self, batch: RowBatch) -> npt.NDArray[np.floating]:
        """Evaluate one bounded public-coordinate design batch."""
        if not batch.columns and self.predict_spec.has_intercept:
            return np.ones((len(batch.row_positions), self.n_coef))
        return self.predict_spec.build_predict_matrix(dict(batch.columns))

    def evaluate_fitting_batch(self, batch: RowBatch) -> npt.NDArray[np.floating]:
        """Evaluate one batch in frozen local-D fitting coordinates."""
        if self.fitting is None:
            raise RuntimeError("Fitting preparation has not been completed.")
        return apply_transforms_to_design(
            self.evaluate_batch(batch), self.fitting.penalty_structure
        )


def _scan_valid(source: RowSource, batch_rows: int):
    """Yield only unpadded Phase-1 batches for the initial exact contract."""
    for batch in source.scan(batch_rows):
        if not np.all(batch.valid):
            raise NotImplementedError(
                "Prepared setup does not yet support padded batches."
            )
        yield batch


def _exact_cubic_knots(
    values: RowSource, variable: str, k: int
) -> npt.NDArray[np.floating]:
    """Return exact rank-interpolated cubic knots without retaining uniques.

    SQLite's on-disk DISTINCT/ORDER BY is deliberately used here instead of a
    fixed-size sketch: cubic knots interpolate ranks through *unique* values.
    """
    with tempfile.TemporaryDirectory(prefix="jaxgam-unique-") as directory:
        db_path = Path(directory) / "values.sqlite"
        with closing(sqlite3.connect(db_path)) as connection, connection:
            # Keep DISTINCT/ORDER BY state on the temporary database, never
            # in SQLite's in-memory temp store for a high-cardinality source.
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("PRAGMA cache_size = -2048")
            connection.execute("CREATE TABLE values_table (value REAL NOT NULL)")
            for batch in _scan_valid(values, 65_536):
                if variable not in batch.columns:
                    raise ValueError(
                        f"Smooth variable '{variable}' not found in source."
                    )
                value = np.asarray(batch.columns[variable], dtype=float)
                if not np.all(np.isfinite(value)):
                    raise ValueError(
                        f"Covariate '{variable}' contains non-finite values."
                    )
                connection.executemany(
                    "INSERT INTO values_table VALUES (?)", ((float(x),) for x in value)
                )
            n_unique = connection.execute(
                "SELECT COUNT(DISTINCT value) FROM values_table"
            ).fetchone()[0]
            if n_unique < k:
                raise ValueError(
                    f"Basis dimension k={k} exceeds number of unique data values "
                    f"({n_unique})."
                )
            ranks = np.linspace(0, n_unique - 1, k)
            needed = sorted(
                {int(np.floor(rank)) for rank in ranks}
                | {int(np.ceil(rank)) for rank in ranks}
            )
            placeholders = ",".join("?" for _ in needed)
            rows = connection.execute(
                "SELECT value FROM (SELECT value, ROW_NUMBER() OVER "
                "(ORDER BY value) - 1 AS position FROM "
                "(SELECT DISTINCT value FROM values_table)) "
                f"WHERE position IN ({placeholders}) ORDER BY position",
                needed,
            ).fetchall()
    lookup = dict(zip(needed, (row[0] for row in rows), strict=True))
    return np.array(
        [
            (1.0 - (rank - np.floor(rank))) * lookup[int(np.floor(rank))]
            + (rank - np.floor(rank)) * lookup[int(np.ceil(rank))]
            for rank in ranks
        ]
    )


def _cubic_from_knots(spec, knots: npt.NDArray[np.floating]) -> CubicRegressionSmooth:
    if spec.bs not in {"cr", "cs"} or len(spec.variables) != 1:
        raise NotImplementedError(
            "Prepared setup currently supports only univariate cr/cs smooths."
        )
    smooth = (
        CubicShrinkageSmooth(spec) if spec.bs == "cs" else CubicRegressionSmooth(spec)
    )
    k = 10 if spec.k == -1 else max(spec.k, smooth._min_k)
    smooth._k = k
    smooth._knots = knots
    smooth._F, smooth._S = smooth._compute_f_and_penalty(smooth._knots)
    smooth._S = smooth._apply_shrinkage(smooth._S)
    smooth.n_coefs = k
    smooth.rank = k if spec.bs == "cs" else k - 2
    smooth.null_space_dim = 0 if spec.bs == "cs" else 2
    smooth._is_setup = True
    return smooth


def _batch_parametric(
    spec: FormulaSpec, columns: dict[str, object], n_rows: int
) -> tuple[np.ndarray, list[str]]:
    return predict_matrix._build_parametric_matrix(
        spec.parametric_terms, columns, spec.has_intercept, n_rows
    )


def _update_exact_parametric_aliases(
    values: np.ndarray, scales: np.ndarray, possible: np.ndarray
) -> None:
    """Track exact proportional numeric columns with coefficient-size state."""
    n_columns = values.shape[1]
    for left in range(n_columns):
        x = values[:, left]
        for right in range(left + 1, n_columns):
            if not possible[left, right]:
                continue
            z = values[:, right]
            nonzero = x != 0.0
            if np.any((~nonzero) & (z != 0.0)):
                possible[left, right] = False
                continue
            if not np.any(nonzero):
                continue
            ratios = z[nonzero] / x[nonzero]
            scale = scales[left, right]
            if np.isnan(scale):
                scale = float(ratios[0])
                scales[left, right] = scale
            if not np.all(ratios == scale):
                possible[left, right] = False


def _canonicalize_exact_parametric_aliases(
    keep_cols: list[int],
    scales: np.ndarray,
    possible: np.ndarray,
    has_intercept: bool,
) -> list[int]:
    """Match dense pivot ties for exact proportional source columns.

    Re-QR of a bounded sequential R factor preserves rank but can reverse an
    equal-norm alias on last-bit differences that depend on batch partition.
    Dense LAPACK keeps the earlier equal-norm column, while a larger-norm
    proportional column wins the pivot. Record that source-level decision
    directly; no row array is retained and the shared dense policy is unchanged.
    """
    retained = set(keep_cols)
    for left in range(scales.shape[0]):
        for right in range(left + 1, scales.shape[1]):
            if not possible[left, right] or np.isnan(scales[left, right]):
                continue
            selected = retained.intersection((left, right))
            if len(selected) != 1:
                continue
            if (has_intercept and left == 0) or abs(scales[left, right]) <= 1.0:
                preferred = left
            else:
                preferred = right
            if preferred not in selected:
                retained.remove(next(iter(selected)))
                retained.add(preferred)
    return sorted(retained)


def prepare_model(
    formula_spec: FormulaSpec, source: RowSource, *, family: Any | None = None
) -> PreparedModel:
    """Prepare exact cubic/numeric metadata using dependency-ordered scans.

    Unsupported by-variables, tensors, and factors fail before a training
    design is materialized. Exactly aliased numeric parametric columns are
    reduced through the same fixed ``CoefficientMap`` contract as dense setup.
    """
    if not isinstance(source, RowSource):
        raise TypeError("Prepared setup requires a restartable RowSource.")
    if source.n_rows <= 0:
        raise ValueError("Prepared setup requires at least one row.")
    start_fingerprint = source.fingerprint()
    if any(
        term.by is not None or term.smooth_type != "s"
        for term in formula_spec.smooth_terms
    ):
        raise NotImplementedError(
            "Prepared setup does not yet support by-variable or tensor smooths."
        )
    try:
        first = next(_scan_valid(source, 1))
    except StopIteration as error:
        raise ValueError("RowSource advertised rows but yielded no batches.") from error
    try:
        replay_first = next(_scan_valid(source, 1))
    except StopIteration as error:
        raise TypeError("Prepared setup requires a replayable RowSource.") from error
    if not np.array_equal(first.row_positions, replay_first.row_positions):
        raise TypeError(
            "Prepared setup requires replayable scans with stable row positions."
        )
    if source.fingerprint() != start_fingerprint:
        raise RuntimeError("RowSource changed during replayability preflight.")
    if first.y is None:
        raise ValueError("Prepared fitting requires a response in the RowSource.")
    required = [term.name for term in formula_spec.parametric_terms]
    required.extend(
        variable for term in formula_spec.smooth_terms for variable in term.variables
    )
    missing = sorted(set(required) - set(first.columns))
    if missing:
        raise ValueError(f"Variables missing from source: {missing}")
    # Parametric factors have a frozen prediction contract but their streamed
    # alias reduction is not implemented yet; reject them before X allocation.
    for term in formula_spec.parametric_terms:
        if is_factor(first.columns[term.name]) or not np.issubdtype(
            np.asarray(first.columns[term.name]).dtype, np.number
        ):
            raise NotImplementedError(
                "Prepared setup currently supports numeric parametric terms only."
            )

    variables = [term.variables[0] for term in formula_spec.smooth_terms]
    if len(set(variables)) != len(variables):
        raise NotImplementedError(
            "Prepared setup does not yet support overlapping cubic terms."
        )
    smooths = []
    for term in formula_spec.smooth_terms:
        k = 10 if term.k == -1 else max(term.k, 3)
        smooths.append(
            _cubic_from_knots(term, _exact_cubic_knots(source, term.variables[0], k))
        )
    param_first, param_names = _batch_parametric(formula_spec, dict(first.columns), 1)
    n_parametric = param_first.shape[1]
    param_R = np.empty((0, n_parametric))
    exact_alias_scales = np.full((n_parametric, n_parametric), np.nan)
    exact_alias_possible = np.triu(
        np.ones((n_parametric, n_parametric), dtype=bool), k=1
    )
    smooth_sums = [np.zeros(smooth.n_coefs) for smooth in smooths]
    smooth_norms = [0.0 for _ in smooths]
    for batch in _scan_valid(source, 65_536):
        columns = dict(batch.columns)
        parametric, _ = _batch_parametric(
            formula_spec, columns, len(batch.row_positions)
        )
        if not np.all(np.isfinite(parametric)):
            raise ValueError("Parametric covariates contain non-finite values.")
        if n_parametric:
            _update_exact_parametric_aliases(
                parametric, exact_alias_scales, exact_alias_possible
            )
            param_R = linalg.qr(np.vstack((param_R, parametric)), mode="economic")[1]
        for index, smooth in enumerate(smooths):
            raw = smooth.predict_matrix(columns)
            smooth_sums[index] += raw.sum(axis=0)
            smooth_norms[index] = max(
                smooth_norms[index], float(np.max(np.sum(np.abs(raw), axis=1)))
            )
    parametric_keep_cols = list(range(n_parametric))
    dropped_param_names: list[str] = []
    if n_parametric:
        # A sequential thin QR has the same column Gram matrix as the original
        # row design. Reusing the dense helper on its bounded R factor preserves
        # the established pivot tolerance, intercept rule, public names and
        # prediction-time CoefficientMap deletion without retaining training X.
        full_param_names = list(param_names)
        (
            _reduced_param_R,
            _reduced_param_names,
            _initial_dropped_param_names,
            parametric_keep_cols,
        ) = ModelSetup._drop_aliased_parametric_columns(
            param_R, full_param_names, formula_spec.has_intercept
        )
        parametric_keep_cols = _canonicalize_exact_parametric_aliases(
            parametric_keep_cols,
            exact_alias_scales,
            exact_alias_possible,
            formula_spec.has_intercept,
        )
        dropped = sorted(set(range(n_parametric)) - set(parametric_keep_cols))
        dropped_param_names = [full_param_names[index] for index in dropped]
        param_names = [full_param_names[index] for index in parametric_keep_cols]
        n_parametric = len(parametric_keep_cols)
    constrained_penalties: list[list[np.ndarray]] = []
    term_blocks: list[TermBlock] = []
    smooth_info: list[SmoothInfo] = []
    offset = n_parametric
    if n_parametric:
        term_blocks.append(
            TermBlock("parametric", 0, n_parametric, n_parametric, "parametric")
        )
    for smooth_index, smooth in enumerate(smooths):
        penalty_offset = smooth_index
        # smoothCon normalization precedes centering and uses the global
        # unweighted infinity norm, accumulated above.
        if smooth_norms[smooth_index] > 0:
            scale = np.linalg.norm(smooth._S, ord=1) / smooth_norms[smooth_index] ** 2
            smooth._S = smooth._S / scale
            smooth._s_scale = scale
        C = smooth_sums[smooth_index]
        Q, _ = np.linalg.qr(C[:, None], mode="complete")
        Z = Q[:, 1:]
        S = Z.T @ smooth._S @ Z
        S = 0.5 * (S + S.T)
        label = CoefficientMap.smooth_label(smooth)
        term_blocks.append(
            TermBlock(
                label,
                offset,
                smooth.n_coefs - 1,
                smooth.n_coefs,
                "smooth",
                smooth,
                (penalty_offset,),
                Z,
            )
        )
        prediction_smooth = smooth.copy_for_prediction()
        term_blocks[-1] = TermBlock(
            label,
            offset,
            smooth.n_coefs - 1,
            smooth.n_coefs,
            "smooth",
            prediction_smooth,
            (penalty_offset,),
            Z,
        )
        smooth_info.append(
            SmoothInfo(
                label,
                "s",
                smooth.spec.variables,
                None,
                offset,
                offset + smooth.n_coefs - 1,
                1,
                penalty_offset,
                smooth.null_space_dim,
            )
        )
        constrained_penalties.append([S])
        offset += smooth.n_coefs - 1
    coef_map = CoefficientMap(
        tuple(term_blocks),
        offset,
        n_parametric + sum(s.n_coefs for s in smooths),
        formula_spec.has_intercept,
    )
    penalties = make_penalty_structure(
        offset,
        [term for term in term_blocks if term.term_type == "smooth"],
        constrained_penalties,
    )
    if source.fingerprint() != start_fingerprint:
        raise RuntimeError("RowSource changed during preparation; prepare again.")
    predict_spec = predict_matrix.PredictSpec(
        coef_map=coef_map,
        smooth_info=tuple(smooth_info),
        term_names=ModelSetup._build_term_names(param_names, smooths, coef_map),
        parametric_terms=tuple(formula_spec.parametric_terms),
        factor_info={},
        ordered_factors=frozenset(),
        has_intercept=formula_spec.has_intercept,
        parametric_keep_cols=tuple(parametric_keep_cols),
        dropped_param_names=tuple(dropped_param_names),
        total_coefs=offset,
    )
    basis_fingerprint = hashlib.sha256(
        repr(
            (
                formula_spec.response,
                formula_spec.has_intercept,
                tuple(
                    (
                        term.variables,
                        term.bs,
                        term.k,
                        term.by,
                        term.smooth_type,
                        term.extra_args,
                    )
                    for term in formula_spec.smooth_terms
                ),
                tuple(term.name for term in formula_spec.parametric_terms),
                tuple(
                    np.asarray(term.smooth._knots).tobytes()
                    for term in term_blocks
                    if term.smooth is not None
                ),
            )
        ).encode()
    ).hexdigest()
    prepared = PreparedModel(
        predict_spec,
        penalties,
        source.fingerprint(),
        basis_fingerprint,
        source.n_rows,
        formula_spec.response,
    )
    return prepare_fitting(prepared, source, family) if family is not None else prepared


def prepare_fitting(
    prepared: PreparedModel, source: RowSource, family: Any
) -> PreparedModel:
    """Freeze CPU fitting metadata by bounded replayable reductions.

    ``family`` is intentionally duck typed: family implementations may be
    Phase-2-aware, but this Phase-1 module never imports JAX.  It only uses
    their established NumPy-compatible initialization/link/domain methods.
    """
    if source.fingerprint() != prepared.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again.")
    static_config_method = getattr(family, "execution_static_config", None)
    parameter_snapshot_method = getattr(family, "execution_parameter_snapshot", None)
    if callable(static_config_method) != callable(parameter_snapshot_method):
        raise TypeError(
            "Family execution contract must provide both static configuration "
            "and parameter snapshot hooks."
        )
    family_static_config = (
        static_config_method() if callable(static_config_method) else None
    )
    family_parameter_snapshot = (
        parameter_snapshot_method() if callable(parameter_snapshot_method) else None
    )
    public = prepared.penalties or PenaltyStructure(prepared.n_coef, ())
    transformed = reparameterize_structure(public)
    gram = np.zeros((prepared.n_coef, prepared.n_coef))
    rhs = np.zeros(prepared.n_coef)
    qr_R = np.empty((0, prepared.n_coef))
    qr_target = np.empty(0)
    ldxx = np.zeros(prepared.n_coef)
    n_obs = 0
    total_weight = 0.0
    weighted_sum = 0.0
    response_min = np.inf
    response_max = -np.inf
    offset_min = np.inf
    offset_max = -np.inf
    for batch in _scan_valid(source, 65_536):
        if batch.y is None:
            raise ValueError("Prepared fitting requires a response in the RowSource.")
        y = np.asarray(batch.y, dtype=float)
        wt = np.asarray(batch.weight, dtype=float)
        offset = np.asarray(batch.offset, dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError("Response contains non-finite values (NaN or Inf).")
        if not np.all(batch.valid):
            raise NotImplementedError(
                "Prepared setup does not yet support padded batches."
            )
        X_public = prepared.evaluate_batch(batch)
        X = apply_transforms_to_design(X_public, transformed)
        # Calling initialize is also the family-owned response-domain check.
        mu = np.asarray(family.initialize(y, wt), dtype=float)
        eta = np.asarray(family.link.link(mu), dtype=float)
        if not np.all(np.isfinite(eta)):
            raise ValueError(
                "Family initialization produced non-finite linear predictors."
            )
        target = eta - offset
        # initialize_beta_cpu is intentionally unweighted.  Keep only its
        # p-by-p normal-equation reductions here; no n-row design is retained.
        gram += X.T @ X
        rhs += X.T @ target
        qr_input = np.vstack((qr_R, X))
        target_input = np.concatenate((qr_target, target))
        Q, qr_R = linalg.qr(qr_input, mode="economic")
        qr_target = Q.T @ target_input
        weighted_public = np.sqrt(wt)[:, None] * X_public
        ldxx += np.sum(weighted_public * weighted_public, axis=0)
        n_obs += len(y)
        total_weight += float(np.sum(wt))
        weighted_sum += float(np.sum(wt * y))
        response_min = min(response_min, float(np.min(y)))
        response_max = max(response_max, float(np.max(y)))
        offset_min = min(offset_min, float(np.min(offset)))
        offset_max = max(offset_max, float(np.max(offset)))
    if n_obs != prepared.n_obs or total_weight <= 0:
        raise RuntimeError("RowSource replay did not provide a valid stable row set.")
    dense_rcond = max(n_obs, prepared.n_coef) * np.finfo(float).eps
    beta_init, _, _, _ = np.linalg.lstsq(qr_R, qr_target, rcond=dense_rcond)
    # Retain the dense initializer's valid-domain fallback without retaining X.
    valid = True
    for batch in _scan_valid(source, 65_536):
        if not np.all(batch.valid):
            raise NotImplementedError(
                "Prepared setup does not yet support padded batches."
            )
        eta = prepared.evaluate_batch(batch)
        eta = apply_transforms_to_design(eta, transformed) @ beta_init + batch.offset
        mu = np.asarray(family.link.inverse(eta), dtype=float)
        if not (
            np.all(np.asarray(family.valid_mu(mu)))
            and np.all(np.asarray(family.valid_eta(eta)))
        ):
            valid = False
            break
    if not valid:
        target = float(family.link.link(np.asarray(weighted_sum / total_weight)))
        rhs_null = np.zeros(prepared.n_coef)
        qr_R = np.empty((0, prepared.n_coef))
        qr_target = np.empty(0)
        for batch in _scan_valid(source, 65_536):
            if not np.all(batch.valid):
                raise NotImplementedError(
                    "Prepared setup does not yet support padded batches."
                )
            X = apply_transforms_to_design(prepared.evaluate_batch(batch), transformed)
            target_batch = target - batch.offset
            rhs_null += X.T @ target_batch
            qr_input = np.vstack((qr_R, X))
            target_input = np.concatenate((qr_target, target_batch))
            Q, qr_R = linalg.qr(qr_input, mode="economic")
            qr_target = Q.T @ target_input
        beta_init, _, _, _ = np.linalg.lstsq(qr_R, qr_target, rcond=dense_rcond)
    null_bases, range_bases = total_penalty_spaces(transformed)
    total_penalty_rank = sum(basis.shape[1] for _, basis in range_bases)
    covered = np.zeros(prepared.n_coef, dtype=bool)
    for block in transformed.blocks:
        covered[block.start : block.stop] = True
    rank_R = np.empty((0, 0))
    rank_columns = 0
    for batch in _scan_valid(source, 65_536):
        if not np.all(batch.valid):
            raise NotImplementedError(
                "Prepared setup does not yet support padded batches."
            )
        X = apply_transforms_to_design(prepared.evaluate_batch(batch), transformed)
        pieces = [X[:, ~covered]]
        pieces.extend(
            X[:, block.start : block.stop] @ basis
            for block, basis in null_bases
            if basis.shape[1]
        )
        null_design = np.column_stack(pieces)
        if rank_columns == 0:
            rank_columns = null_design.shape[1]
            rank_R = np.empty((0, rank_columns))
        Q, rank_R = linalg.qr(np.vstack((rank_R, null_design)), mode="economic")
        del Q
    singular_values = np.linalg.svd(rank_R, compute_uv=False)
    rank_tolerance = (
        (np.max(singular_values) if len(singular_values) else 0.0)
        * max(n_obs, rank_columns)
        * np.finfo(float).eps
    )
    unpenalized_rank_deficit = rank_columns - int(
        np.sum(singular_values > rank_tolerance)
    )
    fitting = FittingPreparation(
        penalty_structure=transformed,
        log_lambda_init=initial_log_sp_from_diagonal(ldxx, public),
        beta_init=beta_init,
        gram=gram,
        rhs=rhs,
        response=ResponseReduction(
            n_obs=n_obs,
            total_weight=total_weight,
            response_min=response_min,
            response_max=response_max,
            weighted_sum=weighted_sum,
            offset_min=offset_min,
            offset_max=offset_max,
        ),
        total_penalty_rank=total_penalty_rank,
        total_penalty_null_dim=prepared.n_coef - total_penalty_rank,
        unpenalized_rank_deficit=unpenalized_rank_deficit,
        family_name=str(family.family_name),
        link_name=type(family.link).__qualname__,
        family_execution_static_config=family_static_config,
        family_parameter_snapshot=family_parameter_snapshot,
    )
    if source.fingerprint() != prepared.source_fingerprint:
        raise RuntimeError(
            "RowSource changed during fitting preparation; prepare again."
        )
    if callable(static_config_method) and (
        static_config_method() != family_static_config
        or parameter_snapshot_method() != family_parameter_snapshot
    ):
        raise RuntimeError(
            "Family or link execution configuration changed during fitting "
            "preparation; prepare again."
        )
    return replace(prepared, fitting=fitting)
