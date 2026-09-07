"""Observation-independent preparation for the initial streamed design path.

The initial exact path is intentionally narrow: additive univariate cubic
smooths and full-rank numeric parametric terms.  It scans rows repeatedly for
global reductions and never routes through ``ModelSetup.build``.
"""

from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import linalg

from jaxgam.data.source import RowBatch, RowSource
from jaxgam.formula import predict_matrix
from jaxgam.formula.design import ModelSetup, SmoothInfo
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
    n_obs: int
    response: str

    @property
    def n_coef(self) -> int:
        return self.predict_spec.total_coefs

    def evaluate_batch(self, batch: RowBatch) -> npt.NDArray[np.floating]:
        """Evaluate one bounded public-coordinate design batch."""
        if not batch.columns and self.predict_spec.has_intercept:
            return np.ones((len(batch.row_positions), self.n_coef))
        return self.predict_spec.build_predict_matrix(dict(batch.columns))


def _exact_cubic_knots(
    values: RowSource, variable: str, k: int
) -> npt.NDArray[np.floating]:
    """Return exact rank-interpolated cubic knots without retaining uniques.

    SQLite's on-disk DISTINCT/ORDER BY is deliberately used here instead of a
    fixed-size sketch: cubic knots interpolate ranks through *unique* values.
    """
    with tempfile.TemporaryDirectory(prefix="jaxgam-unique-") as directory:
        db_path = Path(directory) / "values.sqlite"
        with sqlite3.connect(db_path) as connection:
            # Keep DISTINCT/ORDER BY state on the temporary database, never
            # in SQLite's in-memory temp store for a high-cardinality source.
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("PRAGMA cache_size = -2048")
            connection.execute("CREATE TABLE values_table (value REAL NOT NULL)")
            for batch in values.scan(65_536):
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


def prepare_model(formula_spec: FormulaSpec, source: RowSource) -> PreparedModel:
    """Prepare exact cubic/numeric metadata using dependency-ordered scans.

    Unsupported by-variables, tensors, factors, and rank-deficient parametric
    blocks fail before a training design is materialized.
    """
    start_fingerprint = source.fingerprint()
    if any(
        term.by is not None or term.smooth_type != "s"
        for term in formula_spec.smooth_terms
    ):
        raise NotImplementedError(
            "Prepared setup does not yet support by-variable or tensor smooths."
        )
    first = next(source.scan(1))
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
    smooth_sums = [np.zeros(smooth.n_coefs) for smooth in smooths]
    smooth_norms = [0.0 for _ in smooths]
    for batch in source.scan(65_536):
        columns = dict(batch.columns)
        parametric, _ = _batch_parametric(
            formula_spec, columns, len(batch.row_positions)
        )
        if not np.all(np.isfinite(parametric)):
            raise ValueError("Parametric covariates contain non-finite values.")
        if n_parametric:
            param_R = linalg.qr(np.vstack((param_R, parametric)), mode="economic")[1]
        for index, smooth in enumerate(smooths):
            raw = smooth.predict_matrix(columns)
            smooth_sums[index] += raw.sum(axis=0)
            smooth_norms[index] = max(
                smooth_norms[index], float(np.max(np.sum(np.abs(raw), axis=1)))
            )
    if n_parametric:
        _q, r, _piv = linalg.qr(param_R, pivoting=True, mode="economic")
        diagonal = np.abs(np.diag(r))
        rank = (
            int(np.sum(diagonal > diagonal[0] * np.finfo(float).eps ** 0.9))
            if len(diagonal)
            else 0
        )
        if rank < n_parametric:
            raise NotImplementedError(
                "Prepared setup does not yet support aliased parametric columns."
            )
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
        parametric_keep_cols=(),
        dropped_param_names=(),
        total_coefs=offset,
    )
    return PreparedModel(
        predict_spec,
        penalties,
        source.fingerprint(),
        source.n_rows,
        formula_spec.response,
    )
