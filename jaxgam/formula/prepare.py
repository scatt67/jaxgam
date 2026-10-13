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
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
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


_DEFAULT_SETUP_BATCH_ROWS = 65_536
_FLOAT_BYTES = np.dtype(np.float64).itemsize

#: Fixed canonical row leaves for the narrow QR factors behind discrete setup
#: decisions (parametric aliasing and the unpenalized rank deficit). Leaves are
#: cut by scan ordinal, never by RowSource batch boundaries, like the fixed
#: ``chunk.size`` blocks of mgcv ``bam.fit``/``qr_update`` (R/bam.r).
_DECISION_QR_LEAF_ROWS = 1_024

#: SQLite page cache (KiB) for the on-disk exact unique-value knot table.
_SQLITE_CACHE_KIB = 2_048

#: SQLite-owned memory while one exact knot scan is open. SQLite allocates it
#: outside NumPy/tracemalloc and none of it scales with rows: the configured
#: main page cache above; temp-file ephemeral B-trees for DISTINCT, the window
#: partition and the IN list, each capped by SQLite's default 2000 KiB page
#: cache; the external sorter's in-memory run (at most the main cache size
#: before spilling); and statement/connection state. That is about 12.5 MiB
#: in the worst case; isolated-process RSS increments of whole knot scans
#: stayed below 7 MB at 256-row batches for up to four million unique values.
_SQLITE_SCOPE_BYTES = 16 * 1024 * 1024

#: Python objects, smooth/penalty metadata, LAPACK work arrays and family
#: helpers allocated by any setup scan independently of batch rows and width.
_SETUP_BASE_BYTES = 512 * 1024

#: Per-batch object overhead of one sliced source column (a pandas Series,
#: its index and block manager measure about 1.8 KiB; NumPy views far less).
_SOURCE_COLUMN_BATCH_BYTES = 2 * 1024


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_setup_controls(
    batch_rows: Any, maximum_bytes: Any
) -> tuple[int, int | None]:
    """Reject invalid scan controls before any RowSource access."""
    rows = _positive_integer(batch_rows, "batch_rows")
    if maximum_bytes is None:
        return rows, None
    return rows, _positive_integer(maximum_bytes, "maximum_bytes")


def _scan_valid(source: RowSource, batch_rows: int):
    """Yield only unpadded Phase-1 batches no larger than the requested size."""
    for batch in source.scan(batch_rows):
        if not isinstance(batch, RowBatch):
            raise TypeError("RowSource.scan must yield RowBatch instances.")
        if len(batch.row_positions) > batch_rows:
            raise ValueError(
                f"RowSource yielded {len(batch.row_positions)} rows for a "
                f"setup scan requesting at most {batch_rows}."
            )
        if not np.all(batch.valid):
            raise NotImplementedError(
                "Prepared setup does not yet support padded batches."
            )
        yield batch
        # Do not keep the consumed batch alive while the next one is built.
        del batch


def _source_batch_bytes(batch: RowBatch) -> int:
    """Row-independent object overhead of one yielded source batch."""
    return _SOURCE_COLUMN_BATCH_BYTES * (len(batch.columns) + 1)


def _source_row_bytes(batch: RowBatch) -> int:
    """Bytes one yielded row occupies in a source batch.

    Sources may copy every column on each slice, so all columns count, not
    only formula variables. Narrow numeric columns are charged a float64 cell
    because design evaluation converts them; object columns hold pointers.
    """
    row_bytes = 3 * _FLOAT_BYTES + np.dtype(np.bool_).itemsize
    row_bytes += np.dtype(np.intp).itemsize
    for value in batch.columns.values():
        dtype = value.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            itemsize = value.cat.codes.dtype.itemsize
        else:
            itemsize = getattr(dtype, "itemsize", _FLOAT_BYTES)
        row_bytes += max(int(itemsize), _FLOAT_BYTES)
    return row_bytes


def _decision_tree_depth(n_rows: int) -> int:
    """Upper bound on the retained levels of one canonical decision tree."""
    leaves = max(1, -(-n_rows // _DECISION_QR_LEAF_ROWS))
    return int(leaves - 1).bit_length() + 1


@dataclass(frozen=True)
class SetupWorkspace:
    """Prospective JaxGAM-owned live bytes of the Phase-1 setup scans.

    Scans run sequentially, so one batch size is feasible when the widest
    design scan and, with cubic smooths, one exact SQLite knot scan each fit
    the budget. Row terms count the arrays simultaneously live per batch row:
    source batches, design evaluation (term blocks, ``column_stack`` copy and
    cubic evaluator temporaries), fitting/rank designs and their Fortran QR
    stacks, and row vectors. Fixed terms count coefficient-square arrays,
    local penalty metadata, the canonical decision-QR leaves and tree levels,
    Python/LAPACK work objects, and the SQLite scope during knot scans.
    ``tests/test_formula/test_prepare.py`` checks the bound with tracemalloc.
    Source-owned backing storage, allocator retention and native BLAS/LAPACK
    internals are outside it.
    """

    row_bytes: int
    fixed_bytes: int
    knot_row_bytes: int
    knot_fixed_bytes: int

    @classmethod
    def for_design(
        cls,
        *,
        n_parametric: int,
        basis_dimensions: tuple[int, ...],
        null_dimensions: tuple[int, ...],
        source_row_bytes: int,
        source_batch_bytes: int,
        n_rows: int,
        knot_scans: bool = True,
    ) -> SetupWorkspace:
        """Bound setup scans of the evaluated (pre-alias, raw-basis) design.

        ``n_parametric`` is the evaluated parametric width before alias
        deletion and ``basis_dimensions`` the raw cubic widths before
        centering, so ``width`` bounds both public and fitting designs.
        """
        width = n_parametric + sum(basis_dimensions)
        k_max = max(basis_dimensions, default=0)
        rank_width = n_parametric + sum(null_dimensions)
        design_floats = max(
            # Evaluation: term blocks and their column_stack copy, one previous
            # raw cubic block, and the evaluator's (rows, k) temporaries plus
            # its row vectors and parametric conversions.
            2 * width + 4 * k_max + n_parametric + 16,
            # Fitting/rank passes: two designs (public/fitting, weighted
            # diag(X'WX), or design plus its Fortran QR stack) with local
            # transform products and the QR isfinite mask.
            3 * width,
        )
        # The consumed batch can stay referenced while the next one is built.
        row_bytes = 2 * source_row_bytes + _FLOAT_BYTES * (design_floats + 12)
        depth = _decision_tree_depth(n_rows)
        decision_floats = sum(
            # One canonical leaf and SciPy's copy of a final partial leaf, a
            # merge stack and the retained tree levels of each decision factor.
            2 * _DECISION_QR_LEAF_ROWS * m + (depth + 4) * m * m
            for m in (n_parametric, rank_width)
        )
        coefficient_floats = (
            # Gram, its product temporary, old/new R factors and the p stacked
            # rows of each QR input, plus Q'y-sized vectors.
            8 * width * width
            # Public/local penalties, D transforms, null/range bases, F and S.
            + 6 * sum(k * k for k in basis_dimensions)
            + decision_floats
            + 32 * width
        )
        # Two live source batches also carry per-batch column objects.
        base_bytes = _SETUP_BASE_BYTES + 2 * source_batch_bytes
        fixed_bytes = base_bytes + _FLOAT_BYTES * coefficient_floats
        if knot_scans and basis_dimensions:
            # Source batches, the float conversion and isfinite mask of one
            # variable; SQLite holds its pages in its own fixed scope.
            knot_row_bytes = 2 * source_row_bytes + _FLOAT_BYTES + 1
            knot_fixed_bytes = base_bytes + _SQLITE_SCOPE_BYTES
        else:
            knot_row_bytes = knot_fixed_bytes = 0
        return cls(row_bytes, fixed_bytes, knot_row_bytes, knot_fixed_bytes)

    def _scans(self) -> tuple[tuple[int, int], ...]:
        design = (self.fixed_bytes, self.row_bytes)
        if self.knot_row_bytes:
            return (design, (self.knot_fixed_bytes, self.knot_row_bytes))
        return (design,)

    @property
    def minimum_bytes(self) -> int:
        """Prospective bytes needed to scan one row through every setup pass."""
        return max(fixed + row for fixed, row in self._scans())

    def bytes_for_rows(self, rows: int) -> int:
        """Prospective peak of setup scans yielding ``rows``-row batches."""
        return max(fixed + rows * row for fixed, row in self._scans())

    def batch_rows(self, requested: int, maximum_bytes: int | None) -> int:
        """Largest scan size not above ``requested`` that fits the budget."""
        if maximum_bytes is None:
            return requested
        if maximum_bytes < self.minimum_bytes:
            raise MemoryError(
                "Phase-1 streamed setup requires at least "
                f"{self.minimum_bytes} bytes to scan one row, exceeding the "
                f"setup memory budget of {maximum_bytes} bytes "
                "(FitControl.memory_budget_bytes)."
            )
        return min(
            requested,
            *((maximum_bytes - fixed) // row for fixed, row in self._scans()),
        )


def _fortran_stack(
    top: npt.NDArray[np.floating], bottom: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Stack two row blocks directly into a LAPACK-ready Fortran array.

    SciPy would otherwise copy a C-ordered ``np.vstack`` result before
    ``geqrf``; factoring this owned array in place gives bitwise-identical
    factors with one stacked copy instead of two.
    """
    stacked = np.empty((len(top) + len(bottom), bottom.shape[1]), order="F")
    stacked[: len(top)] = top
    stacked[len(top) :] = bottom
    return stacked


def _r_factor(design: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return the economic Householder R factor, overwriting ``design``."""
    (_, R) = linalg.qr(design, mode="raw", overwrite_a=True)
    return R


class _CanonicalTreeR:
    """Batch-partition-invariant R factor for narrow decision designs.

    Rows enter fixed canonical leaves by scan ordinal and leaf factors merge in
    a binary-counter tree. Every LAPACK call therefore receives the same input
    for any RowSource batch partition, so rank/alias decisions cannot flip with
    ``batch_rows``; roundoff grows with tree depth instead of batch count.
    """

    def __init__(self, n_columns: int) -> None:
        self._n_columns = n_columns
        self._leaf = np.empty((_DECISION_QR_LEAF_ROWS, n_columns), order="F")
        self._filled = 0
        self._levels: list[tuple[int, npt.NDArray[np.floating]]] = []

    def update(self, rows: npt.NDArray[np.floating]) -> None:
        if not len(self._leaf):
            raise RuntimeError("Canonical decision QR was already finalized.")
        start = 0
        while start < len(rows):
            take = min(len(rows) - start, len(self._leaf) - self._filled)
            self._leaf[self._filled : self._filled + take] = rows[start : start + take]
            self._filled += take
            start += take
            if self._filled == len(self._leaf):
                # The full leaf is consumed in place and refilled afterwards.
                self._push(_r_factor(self._leaf))
                self._filled = 0

    def _push(self, R: npt.NDArray[np.floating]) -> None:
        level = 0
        while self._levels and self._levels[-1][0] == level:
            R = _r_factor(_fortran_stack(self._levels.pop()[1], R))
            level += 1
        self._levels.append((level, R))

    def finalize(self) -> npt.NDArray[np.floating]:
        """Return the factor and release the leaf buffer."""
        if self._filled:
            # A partial leaf is not Fortran-contiguous; SciPy copies it.
            self._push(_r_factor(self._leaf[: self._filled]))
            self._filled = 0
        self._leaf = np.empty((0, self._n_columns), order="F")
        if not self._levels:
            return np.empty((0, self._n_columns))
        R = self._levels.pop()[1]
        while self._levels:
            R = _r_factor(_fortran_stack(self._levels.pop()[1], R))
        return R


def _merge_qr(
    stacked: npt.NDArray[np.floating],
    qty: npt.NDArray[np.floating],
    target: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Factor ``[R; X]`` in place and return ``(R, Q'[qty; target])``.

    ``stacked`` comes from :func:`_fortran_stack` and becomes the economic
    ``Q``; that buffer dies when the caller releases ``stacked``.
    """
    Q, R = linalg.qr(stacked, mode="economic", overwrite_a=True)
    return R, Q.T @ np.concatenate((qty, target))


def _exact_cubic_knots(
    values: RowSource,
    variable: str,
    k: int,
    batch_rows: int = _DEFAULT_SETUP_BATCH_ROWS,
) -> npt.NDArray[np.floating]:
    """Return exact rank-interpolated cubic knots without retaining uniques.

    SQLite's on-disk DISTINCT/ORDER BY is deliberately used here instead of a
    fixed-size sketch: cubic knots interpolate ranks through *unique* values.
    """
    batch_rows = _positive_integer(batch_rows, "batch_rows")
    with tempfile.TemporaryDirectory(prefix="jaxgam-unique-") as directory:
        db_path = Path(directory) / "values.sqlite"
        with closing(sqlite3.connect(db_path)) as connection, connection:
            # Keep DISTINCT/ORDER BY state on the temporary database, never
            # in SQLite's in-memory temp store for a high-cardinality source.
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute(f"PRAGMA cache_size = -{_SQLITE_CACHE_KIB}")
            connection.execute("CREATE TABLE values_table (value REAL NOT NULL)")
            for batch in _scan_valid(values, batch_rows):
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


def _cubic_class(spec) -> type[CubicRegressionSmooth]:
    """Validate a prepared cubic term before any data scan."""
    if spec.bs not in {"cr", "cs"} or len(spec.variables) != 1:
        raise NotImplementedError(
            "Prepared setup currently supports only univariate cr/cs smooths."
        )
    return CubicShrinkageSmooth if spec.bs == "cs" else CubicRegressionSmooth


def _cubic_dimension(spec) -> int:
    """Basis dimension shared by knot discovery and construction."""
    return 10 if spec.k == -1 else max(spec.k, _cubic_class(spec)._min_k)


def _cubic_null_dimension(spec) -> int:
    """Raw penalty null-space dimension of a prepared cubic term."""
    return 0 if spec.bs == "cs" else 2


def _cubic_from_knots(spec, knots: npt.NDArray[np.floating]) -> CubicRegressionSmooth:
    smooth = _cubic_class(spec)(spec)
    k = _cubic_dimension(spec)
    smooth._k = k
    smooth._knots = knots
    smooth._F, smooth._S = smooth._compute_f_and_penalty(smooth._knots)
    smooth._S = smooth._apply_shrinkage(smooth._S)
    smooth.n_coefs = k
    smooth.rank = k if spec.bs == "cs" else k - 2
    smooth.null_space_dim = _cubic_null_dimension(spec)
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
    formula_spec: FormulaSpec,
    source: RowSource,
    *,
    family: Any | None = None,
    batch_rows: int = _DEFAULT_SETUP_BATCH_ROWS,
    maximum_bytes: int | None = None,
) -> PreparedModel:
    """Prepare exact cubic/numeric metadata using dependency-ordered scans.

    Unsupported by-variables, tensors, and factors fail before a training
    design is materialized. Exactly aliased numeric parametric columns are
    reduced through the same fixed ``CoefficientMap`` contract as dense setup.

    Every full-data scan requests at most ``batch_rows`` rows. With
    ``maximum_bytes``, that request shrinks to the largest size whose
    prospective :class:`SetupWorkspace` fits; a budget that cannot scan one
    row raises ``MemoryError`` before any full-data scan. Knots, alias and
    rank decisions do not depend on the resulting batch partition.
    """
    if not isinstance(source, RowSource):
        raise TypeError("Prepared setup requires a restartable RowSource.")
    batch_rows, maximum_bytes = _validate_setup_controls(batch_rows, maximum_bytes)
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
    basis_dimensions = tuple(
        _cubic_dimension(term) for term in formula_spec.smooth_terms
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
    # The evaluated parametric block, not the formula term count, fixes the
    # scanned width (an intercept plus any column expansion).
    param_first, param_names = _batch_parametric(formula_spec, dict(first.columns), 1)
    n_parametric = param_first.shape[1]
    workspace = SetupWorkspace.for_design(
        n_parametric=n_parametric,
        basis_dimensions=basis_dimensions,
        null_dimensions=tuple(
            _cubic_null_dimension(term) for term in formula_spec.smooth_terms
        ),
        source_row_bytes=_source_row_bytes(first),
        source_batch_bytes=_source_batch_bytes(first),
        n_rows=source.n_rows,
    )
    setup_rows = workspace.batch_rows(min(batch_rows, source.n_rows), maximum_bytes)
    smooths = [
        _cubic_from_knots(
            term, _exact_cubic_knots(source, term.variables[0], k, setup_rows)
        )
        for term, k in zip(formula_spec.smooth_terms, basis_dimensions, strict=True)
    ]
    param_tree = _CanonicalTreeR(n_parametric) if n_parametric else None
    exact_alias_scales = np.full((n_parametric, n_parametric), np.nan)
    exact_alias_possible = np.triu(
        np.ones((n_parametric, n_parametric), dtype=bool), k=1
    )
    smooth_sums = [np.zeros(smooth.n_coefs) for smooth in smooths]
    smooth_norms = [0.0 for _ in smooths]
    for batch in _scan_valid(source, setup_rows):
        columns = dict(batch.columns)
        parametric, _ = _batch_parametric(
            formula_spec, columns, len(batch.row_positions)
        )
        if not np.all(np.isfinite(parametric)):
            raise ValueError("Parametric covariates contain non-finite values.")
        if param_tree is not None:
            _update_exact_parametric_aliases(
                parametric, exact_alias_scales, exact_alias_possible
            )
            param_tree.update(parametric)
        del parametric
        for index, smooth in enumerate(smooths):
            raw = smooth.predict_matrix(columns)
            smooth_sums[index] += raw.sum(axis=0)
            smooth_norms[index] = max(
                smooth_norms[index], float(np.max(np.sum(np.abs(raw), axis=1)))
            )
            del raw
    parametric_keep_cols = list(range(n_parametric))
    dropped_param_names: list[str] = []
    if n_parametric:
        # A streamed thin QR has the same column Gram matrix as the original
        # row design. Reusing the dense helper on its bounded R factor preserves
        # the established pivot tolerance, intercept rule, public names and
        # prediction-time CoefficientMap deletion without retaining training X.
        # The canonical tree makes that R independent of the batch partition.
        full_param_names = list(param_names)
        (
            _reduced_param_R,
            _reduced_param_names,
            _initial_dropped_param_names,
            parametric_keep_cols,
        ) = ModelSetup._drop_aliased_parametric_columns(
            param_tree.finalize(), full_param_names, formula_spec.has_intercept
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
    return (
        _prepare_fitting(prepared, source, family, setup_rows)
        if family is not None
        else prepared
    )


def prepare_fitting(
    prepared: PreparedModel,
    source: RowSource,
    family: Any,
    *,
    batch_rows: int = _DEFAULT_SETUP_BATCH_ROWS,
    maximum_bytes: int | None = None,
) -> PreparedModel:
    """Freeze CPU fitting metadata by bounded replayable reductions.

    ``family`` is intentionally duck typed: family implementations may be
    Phase-2-aware, but this Phase-1 module never imports JAX.  It only uses
    their established NumPy-compatible initialization/link/domain methods.

    ``batch_rows``/``maximum_bytes`` have the same meaning as in
    :func:`prepare_model`; a budget is checked before any full-data scan.
    """
    batch_rows, maximum_bytes = _validate_setup_controls(batch_rows, maximum_bytes)
    if source.fingerprint() != prepared.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again.")
    batch_rows = min(batch_rows, source.n_rows)
    if maximum_bytes is not None:
        spec = prepared.predict_spec
        smooth_terms = [term for term in spec.coef_map.terms if term.smooth is not None]
        try:
            first = next(_scan_valid(source, 1))
        except StopIteration as error:
            raise RuntimeError(
                "RowSource replay did not provide a valid stable row set."
            ) from error
        # Widths are the evaluated (pre-alias, pre-centering) design widths.
        workspace = SetupWorkspace.for_design(
            n_parametric=len(spec.parametric_keep_cols) + len(spec.dropped_param_names),
            basis_dimensions=tuple(term.n_coefs_raw for term in smooth_terms),
            null_dimensions=tuple(term.smooth.null_space_dim for term in smooth_terms),
            source_row_bytes=_source_row_bytes(first),
            source_batch_bytes=_source_batch_bytes(first),
            n_rows=source.n_rows,
            knot_scans=False,
        )
        batch_rows = workspace.batch_rows(batch_rows, maximum_bytes)
    return _prepare_fitting(prepared, source, family, batch_rows)


def _prepare_fitting(
    prepared: PreparedModel, source: RowSource, family: Any, setup_rows: int
) -> PreparedModel:
    """Run fitting preparation scans with an already budgeted batch size."""
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
    for batch in _scan_valid(source, setup_rows):
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
        weighted_public = np.sqrt(wt)[:, None] * X_public
        weighted_public *= weighted_public
        ldxx += np.sum(weighted_public, axis=0)
        del weighted_public
        X = apply_transforms_to_design(X_public, transformed)
        del X_public
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
        stacked = _fortran_stack(qr_R, X)
        del X
        qr_R, qr_target = _merge_qr(stacked, qr_target, target)
        del stacked
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
    for batch in _scan_valid(source, setup_rows):
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
        for batch in _scan_valid(source, setup_rows):
            if not np.all(batch.valid):
                raise NotImplementedError(
                    "Prepared setup does not yet support padded batches."
                )
            X = apply_transforms_to_design(prepared.evaluate_batch(batch), transformed)
            target_batch = target - batch.offset
            rhs_null += X.T @ target_batch
            stacked = _fortran_stack(qr_R, X)
            del X
            qr_R, qr_target = _merge_qr(stacked, qr_target, target_batch)
            del stacked
        beta_init, _, _, _ = np.linalg.lstsq(qr_R, qr_target, rcond=dense_rcond)
    null_bases, range_bases = total_penalty_spaces(transformed)
    total_penalty_rank = sum(basis.shape[1] for _, basis in range_bases)
    covered = np.zeros(prepared.n_coef, dtype=bool)
    for block in transformed.blocks:
        covered[block.start : block.stop] = True
    null_bases = [(block, basis) for block, basis in null_bases if basis.shape[1]]
    rank_columns = int(np.sum(~covered)) + sum(
        basis.shape[1] for _, basis in null_bases
    )
    # Canonical leaves make the rank decision independent of batch_rows.
    rank_tree = _CanonicalTreeR(rank_columns) if rank_columns else None
    for batch in _scan_valid(source, setup_rows):
        if not np.all(batch.valid):
            raise NotImplementedError(
                "Prepared setup does not yet support padded batches."
            )
        if rank_tree is None:
            continue
        X = apply_transforms_to_design(prepared.evaluate_batch(batch), transformed)
        pieces = [X[:, ~covered]]
        pieces.extend(
            X[:, block.start : block.stop] @ basis for block, basis in null_bases
        )
        del X
        rank_tree.update(np.column_stack(pieces))
        del pieces
    rank_R = rank_tree.finalize() if rank_tree is not None else np.empty((0, 0))
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
