"""Thin plate regression splines (tp, ts basis types).

Implements the TPRS basis and penalty construction following
R's mgcv smooth.construct.tp.smooth.spec() and src/tprs.c.

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 5.2
R source reference: R/smooth.r smooth.construct.tp.smooth.spec(),
                    src/tprs.c construct_tprs()
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, ceil, comb, sqrt

import numpy as np
import numpy.typing as npt

from jaxgam.formula.terms import SmoothSpec
from jaxgam.penalties.penalty import Penalty
from jaxgam.smooths.base import Smooth
from jaxgam.smooths.source_arithmetic import (
    SOURCE_FMA_CONTRACTION,
    column_squares,
    householder_left_transpose,
    householder_right,
    knot_squared_distances,
    matmult_tn,
    qt_factor,
    squared_distances,
)
from jaxgam.smooths.source_lapack import source_lanczos, source_project_rows
from jaxgam.smooths.utils import (
    DISTANCE_BATCH_ROWS,
    _compute_distance_matrix,
    _get_unique_rows,
    _SourceMeanAccumulator,
    _subsample_knots,
)

# ---------------------------------------------------------------------------
# TPS semi-kernel helpers
# ---------------------------------------------------------------------------


def default_penalty_order(d: int) -> int:
    """Default penalty order m for dimension d.

    Parameters
    ----------
    d : int
        Number of covariates (dimension of the smooth).

    Returns
    -------
    int
        Default penalty order: ``ceil((d + 2) / 2)``.
    """
    return ceil((d + 2) / 2)


def null_space_dimension(d: int, m: int) -> int:
    """Dimension M of the polynomial null space.

    Parameters
    ----------
    d : int
        Number of covariates.
    m : int
        Penalty order.

    Returns
    -------
    int
        ``comb(m + d - 1, d)`` — the number of monomials of
        degree < m in d variables.
    """
    return comb(m + d - 1, d)


def eta_const(m: int, d: int) -> float:
    """Constant c_{m,d} in the TPS semi-kernel.

    Parameters
    ----------
    m : int
        Penalty order.
    d : int
        Dimension.

    Returns
    -------
    float
        The constant multiplier for the TPS semi-kernel.
    """
    # Follow src/tprs.c::eta_const's division order. The gamma expression is
    # algebraically equivalent, but differs by one ULP for m=2, d=1; that is
    # enough to change a truncated Lanczos gauge at small TPRS ranks.
    if 2 * m <= d:
        raise ValueError("Thin-plate kernel requires 2m > d.")
    source_pi = asin(1.0) * 2.0
    half = sqrt(source_pi)
    d2 = d // 2
    if d % 2 == 0:
        factor = -1.0 if (m + 1 + d2) % 2 else 1.0
        for _ in range(2 * m - 1):
            factor /= 2.0
        for _ in range(d2):
            factor /= source_pi
        for divisor in range(2, m):
            factor /= divisor
        for divisor in range(2, m - d2 + 1):
            factor /= divisor
    else:
        factor = half
        for index in range(m - (d - 1) // 2):
            factor /= -0.5 - index
        for _ in range(m):
            factor /= 4.0
        for _ in range(d2):
            factor /= source_pi
        factor /= half
        for divisor in range(2, m):
            factor /= divisor
    return factor


def _source_fast_eta(
    squared: npt.NDArray[np.float64], constant: float, m: int, d: int
) -> npt.NDArray[np.float64]:
    """Apply src/tprs.c::fast_eta to squared distances, in its operation order."""
    values = np.zeros(squared.shape, dtype=np.float64)
    positive = squared > 0.0
    distances = squared[positive]
    transformed = np.full(len(distances), constant, dtype=np.float64)
    if d % 2 == 0:
        transformed *= np.log(distances) * 0.5
        for _ in range(m - d // 2):
            transformed *= distances
    else:
        for _ in range(m - d // 2 - 1):
            transformed *= distances
        transformed *= np.sqrt(distances)
    values[positive] = transformed
    return values


def _source_tps_e_matrix(
    knots: npt.NDArray[np.float64], m: int
) -> npt.NDArray[np.float64]:
    """Build the symmetric knot kernel in src/tprs.c::tpsE operation order."""
    d = knots.shape[1]
    squared = knot_squared_distances(
        np.ascontiguousarray(knots, dtype=np.float64), SOURCE_FMA_CONTRACTION
    )
    return _source_fast_eta(squared, eta_const(m, d), m, d)


def _source_tps_rows(
    rows: npt.NDArray[np.float64], knots: npt.NDArray[np.float64], m: int
) -> npt.NDArray[np.float64]:
    """Evaluate radial rows as src/tprs.c::predict_tprs and ``tps_g`` do.

    Each squared distance accumulates ``knot - row`` over covariates in source
    order before ``fast_eta``; this is not the Euclidean-distance formula.
    """
    squared = squared_distances(
        np.ascontiguousarray(rows, dtype=np.float64),
        np.ascontiguousarray(knots, dtype=np.float64),
        SOURCE_FMA_CONTRACTION,
    )
    return _source_fast_eta(squared, eta_const(m, knots.shape[1]), m, knots.shape[1])


def _source_matmult_tn(
    A: npt.NDArray[np.float64], B: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return ``A' B`` with src/matrix.c::matmult(C, A, B, 1, 0) accumulation."""
    return matmult_tn(
        np.ascontiguousarray(A, dtype=np.float64),
        np.ascontiguousarray(B, dtype=np.float64),
        SOURCE_FMA_CONTRACTION,
    )


def _source_qt(
    A: npt.NDArray[np.float64], *, full: bool = False
) -> npt.NDArray[np.float64]:
    """Port src/matrix.c::QT for ``A`` with ``A.r <= A.c``.

    With ``full=False`` the rows of the result hold the scaled Householder
    vectors used by ``HQmult``. With ``full=True`` the explicit ``Q`` with
    ``A Q = [0, T]`` is returned, formed in QT's row-update order.
    """
    return qt_factor(
        np.ascontiguousarray(A, dtype=np.float64), full, SOURCE_FMA_CONTRACTION
    )


def _source_householder_right(
    C: npt.NDArray[np.float64], reflectors: npt.NDArray[np.float64]
) -> None:
    """In-place src/matrix.c::HQmult(C, U, 0, 0): ``C <- C Q``."""
    householder_right(C, reflectors, SOURCE_FMA_CONTRACTION)


def _source_householder_left_transpose(
    C: npt.NDArray[np.float64], reflectors: npt.NDArray[np.float64]
) -> None:
    """In-place src/matrix.c::HQmult(C, U, 1, 1): ``C <- Q' C``."""
    householder_left_transpose(C, reflectors, SOURCE_FMA_CONTRACTION)


class _SourceColumnSquares:
    """Accumulate tprs_setup's column sums of squares in source row order."""

    def __init__(self, width: int) -> None:
        self._sum = np.zeros(width, dtype=np.float64)
        self.count = 0

    def add(self, rows: npt.NDArray[np.float64]) -> None:
        rows = np.ascontiguousarray(rows, dtype=np.float64)
        column_squares(rows, self._sum, SOURCE_FMA_CONTRACTION)
        self.count += len(rows)

    def scales(self) -> npt.NDArray[np.float64]:
        if self.count == 0:
            raise ValueError("Cannot scale a thin-plate basis without rows.")
        scales = np.sqrt(self._sum / self.count)
        scales[scales == 0] = 1.0
        return scales


@dataclass(frozen=True)
class _TPRSKnotBasis:
    """Unscaled src/tprs.c::tprs_setup quantities for one set of knots.

    ``knot_rows`` is ``[U diag(v) Z, T]`` at the knots, the training basis for
    knots taken from the data; it is ``None`` for the pure-knot basis, whose
    rows tprs_setup evaluates through ``tps_g`` instead.
    """

    UZ: npt.NDArray[np.float64]
    S: npt.NDArray[np.float64]
    knot_rows: npt.NDArray[np.float64] | None


def _source_tprs_knot_basis(
    knots: npt.NDArray[np.float64],
    m: int,
    k: int,
    *,
    available_workspace_bytes: int | None = None,
) -> _TPRSKnotBasis:
    """Construct the truncated basis as src/tprs.c::tprs_setup does.

    The Householder factors are applied with ``HQmult`` rather than formed
    explicitly, so ``U diag(v) Z``, ``U Z`` and ``Z' diag(v) Z`` follow the
    source accumulation order. ``available_workspace_bytes`` excludes the
    knots and is charged for the kernel and polynomial matrices first.
    """
    nk = knots.shape[0]
    E = _source_tps_e_matrix(knots, m)
    T = compute_polynomial_basis(knots, m)
    M = T.shape[1]
    wiggly = k - M
    UZ = np.zeros((nk + M, k), dtype=np.float64)
    UZ[nk:, wiggly:] = np.eye(M)
    if nk == k:
        # Pure-knot basis: tprs_setup skips Lanczos and keeps the full Q.
        UZ[:nk, :wiggly] = _source_qt(T.T, full=True)[:, :wiggly]
        reflectors = _source_qt(T.T)
        S = E
        _source_householder_right(S, reflectors)
        _source_householder_left_transpose(S, reflectors)
        S[:, wiggly:] = 0.0
        S[wiggly:, :] = 0.0
        return _TPRSKnotBasis(UZ, S, None)
    available = None
    if available_workspace_bytes is not None:
        available = max(available_workspace_bytes - E.nbytes - T.nbytes, 0)
    values, vectors = source_lanczos(E, k, available_workspace_bytes=available)
    del E
    reflectors = _source_qt(_source_matmult_tn(T, vectors))
    UZ[:nk] = vectors
    _source_householder_right(UZ[:nk], reflectors)
    UZ[:nk, wiggly:] = 0.0
    rows = vectors * values[np.newaxis, :]
    _source_householder_right(rows, reflectors)
    rows[:, wiggly:] = T
    S = np.zeros((k, k), dtype=np.float64)
    np.fill_diagonal(S, values)
    _source_householder_right(S, reflectors)
    _source_householder_left_transpose(S, reflectors)
    S[:, wiggly:] = 0.0
    S[wiggly:, :] = 0.0
    return _TPRSKnotBasis(UZ, S, rows)


def _source_scaled_penalty(
    S: npt.NDArray[np.float64], scales: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Rescale and symmetrize as tprs_setup's column loop and smooth.r do.

    tprs_setup divides row ``i`` and then column ``i`` by ``w_i`` for each
    ``i`` in turn, so entry ``(a, b)`` is divided by ``w_min(a,b)`` first.
    smooth.construct.tp then forms ``(S + t(S)) / 2``.
    """
    index = np.arange(len(scales))
    first = scales[np.minimum.outer(index, index)]
    second = scales[np.maximum.outer(index, index)]
    scaled = S / first / second
    return (scaled + scaled.T) / 2


def _source_knot_indices(
    rows: npt.NDArray[np.float64], knots: npt.NDArray[np.float64]
) -> npt.NDArray[np.intp]:
    """Map centered rows to their identical knot, as tprs.c's yxindex does.

    Xd_strip compares coordinates with ``!=``, so signed zeros are equal. The
    knots are the lexicographically sorted unique rows.
    """
    missing = ValueError(
        "Thin-plate training rows must coincide with the source knots."
    )
    if knots.shape[1] == 1:
        values = knots[:, 0]
        index = np.searchsorted(values, rows[:, 0])
        if np.any(index >= len(values)) or np.any(values[index] != rows[:, 0]):
            raise missing
        return index
    lookup = {tuple(row + 0.0): index for index, row in enumerate(knots)}
    try:
        return np.fromiter(
            (lookup[tuple(row + 0.0)] for row in rows), dtype=np.intp, count=len(rows)
        )
    except KeyError:
        raise missing from None


def tps_semi_kernel(
    r: npt.NDArray[np.floating], m: int, d: int
) -> npt.NDArray[np.floating]:
    """Evaluate the TPS semi-kernel eta_{m,d}(r).

    Parameters
    ----------
    r : np.ndarray
        Distance values (non-negative).
    m : int
        Penalty order.
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        Semi-kernel values, same shape as r.
    """
    c = eta_const(m, d)
    alpha = 2 * m - d

    if alpha % 2 == 0:
        # eta = c * r^alpha * log(r), with 0*log(0) = 0
        result = np.zeros_like(r, dtype=float)
        mask = r > 0
        result[mask] = c * r[mask] ** alpha * np.log(r[mask])
    else:
        # eta = c * r^alpha
        result = c * r**alpha

    return result


def compute_polynomial_basis(
    X: npt.NDArray[np.floating], m: int
) -> npt.NDArray[np.floating]:
    """Compute the polynomial null space basis T.

    Generates all monomials of total degree < m evaluated at the data
    points X, column-ordered by R's ``gen_tps_poly_powers`` odometer
    (mgcv/src/tprs.c): the first variable's power varies fastest. This is
    NOT a total-degree grouping (see ``_monomial_indices``).

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, d)``.
    m : int
        Penalty order.

    Returns
    -------
    np.ndarray
        Polynomial basis, shape ``(n, M)`` where
        ``M = comb(m + d - 1, d)``.
    """
    n, d = X.shape
    M = null_space_dimension(d, m)
    T = np.zeros((n, M))

    # Monomials of total degree < m in gen_tps_poly_powers order, e.g. for
    # d=1: [1, x] when m=2; [1, x, x²] when m=3.
    _fill_polynomial_basis(T, X, m, d)
    return T


def _fill_polynomial_basis(
    T: npt.NDArray[np.floating],
    X: npt.NDArray[np.floating],
    m: int,
    d: int,
) -> None:
    """Fill polynomial basis for general d using the R monomial odometer.

    Fills T in-place with monomials x1^a1 * x2^a2 * ... * xd^ad for all
    (a1, ..., ad) with a1 + ... + ad < m, column-ordered by R's
    ``gen_tps_poly_powers`` odometer (first variable varies fastest;
    see ``_monomial_indices``) — not by total degree.
    """
    n = X.shape[0]
    # tpsT/predict_tprs: x = 1; for each covariate k, multiply by X[,k] once
    # per unit of its power, so higher powers are repeated products.
    indices = _monomial_indices(d, m)
    for col, idx in enumerate(indices):
        val = np.ones(n)
        for j, power in enumerate(idx):
            for _ in range(power):
                val = val * X[:, j]
        T[:, col] = val


def _monomial_indices(d: int, max_degree: int) -> list[tuple[int, ...]]:
    """Generate multi-indices for the polynomial null-space monomials.

    ``max_degree`` is the penalty order ``m``; there are ``comb(m+d-1, d)``
    monomials of total degree ``< m`` in ``d`` variables. The ordering
    reproduces R's ``gen_tps_poly_powers`` odometer (mgcv/src/tprs.c:100-130):
    the FIRST variable's power increments fastest and carries up to higher
    variables. This is NOT a total-degree grouping — e.g. for d=2, m=3 the
    degree-2 ``(2,0)`` precedes the degree-1 ``(0,1)``. Matching R's order
    exactly is required for raw X/S/UZ smoothCon parity at d>=2, m>=3.
    """
    m = max_degree
    M = null_space_dimension(d, m)
    index = [0] * d
    out: list[tuple[int, ...]] = []
    for _ in range(M):
        out.append(tuple(index))
        s = sum(index)
        if s < m - 1:
            index[0] += 1
        else:
            s -= index[0]
            index[0] = 0
            for j in range(1, d):
                index[j] += 1
                s += 1
                if s == m:
                    s -= index[j]
                    index[j] = 0
                else:
                    break
    return out


def _default_k(d: int, M: int) -> int:
    """Default basis dimension k for dimension d.

    Matches R's mgcv ``smooth.construct.tp.smooth.spec`` (R/smooth.r:1316-1318)::

        def.k <- c(8, 27, 100); dd <- min(dim, 3); k <- M + def.k[dd]

    where ``M = null.space.dimension(d, m)``. Dropping the ``M`` term (as the
    previous hardcoded ``{1: 10, 2: 30}`` did) is only coincidentally correct
    for the default penalty order ``m=2`` (``M=2``); for ``m=1``/``m=3`` it
    silently builds a different-sized basis than R.
    """
    def_k = (8, 27, 100)
    return M + def_k[min(d, 3) - 1]


# ---------------------------------------------------------------------------
# TPRSSmooth
# ---------------------------------------------------------------------------


class TPRSSmooth(Smooth):
    """Thin plate regression spline smooth (bs="tp").

    Implements the TPRS basis and penalty construction following
    R's mgcv.

    Parameters
    ----------
    spec : SmoothSpec
        Smooth term specification.
    """

    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)
        self._X: npt.NDArray[np.floating] | None = None
        self._S: npt.NDArray[np.floating] | None = None
        self._UZ: npt.NDArray[np.floating] | None = None
        self._Xu: npt.NDArray[np.floating] | None = None
        self._shift: npt.NDArray[np.floating] | None = None
        self._col_norms: npt.NDArray[np.floating] | None = None
        # Scaled tprs_setup training rows at the knots, when the knots are the
        # data's unique rows. Training rows are not prediction rows: R maps
        # each datum to U diag(v) Z rather than re-evaluating the kernel.
        self._knot_rows: npt.NDArray[np.floating] | None = None
        self._knots_from_data: bool = False
        self._m: int = 0
        self._M: int = 0
        self._k: int = 0
        self._d: int = 0

    def copy_for_prediction(self) -> TPRSSmooth:
        """Drop the training design, penalty and training knot rows."""
        clone = super().copy_for_prediction()
        clone._knot_rows = None
        return clone

    def copy_for_training(self) -> TPRSSmooth:
        """Keep the bounded knot rows that reproduce source training rows."""
        clone = super().copy_for_prediction()
        clone._knot_rows = self._knot_rows
        return clone

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct TPRS basis from data.

        Follows R's smooth.construct.tp.smooth.spec() and tprs.c::tprs_setup:
        1. Centre covariates with R's two-pass mean
        2. Get unique (sorted) knots, subsampling only beyond max_knots
        3. Build E, T, the Lanczos eigenbasis and the T'U Householder factor
        4. Map data rows to the knot rows U diag(v) Z (or evaluate tps_g rows
           for subsampled knots)
        5. Column-scale X, UZ and S and symmetrize S
        6. Apply any shrinkage, then smoothCon's penalty normalization

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Must contain keys matching ``self.spec.variables``.
        """
        # Validate required variables are present
        for v in self.spec.variables:
            if v not in data:
                raise KeyError(
                    f"Variable '{v}' not found in data. Available: {list(data.keys())}"
                )

        # Step 1: Extract and centre covariates
        d = len(self.spec.variables)
        self._d = d
        cols = [np.asarray(data[v], dtype=float) for v in self.spec.variables]
        raw = np.column_stack(cols)
        n = raw.shape[0]

        source_mean = _SourceMeanAccumulator(d)
        for row in raw:
            source_mean.add_first(row)
        source_mean.begin_correction()
        for row in raw:
            source_mean.add_correction(row)
        self._shift = source_mean.result()
        X_centered = raw - self._shift

        # Step 2: Determine penalty order m
        self._m = self.spec.extra_args.get("m", default_penalty_order(d))

        # Step 3: Compute null space dimension M
        self._M = null_space_dimension(d, self._m)

        # Step 4: Determine k
        k = self.spec.k
        if k == -1:
            k = _default_k(d, self._M)
        if k < self._M + 1:
            k = self._M + 1
        self._k = k

        if k > n:
            raise ValueError(
                f"Basis dimension k={k} exceeds number of observations n={n}. "
                f"Reduce k or add more data."
            )

        # Step 5: Get unique knots
        Xu, inverse = _get_unique_rows(X_centered)
        n_unique = Xu.shape[0]

        if n_unique < k:
            raise ValueError(
                f"Number of unique covariate values ({n_unique}) is less than "
                f"basis dimension k={k}. Reduce k or add more distinct data."
            )

        # Subsample knots if too many unique values
        max_knots = self.spec.extra_args.get("max_knots", 2000)
        knots_from_data = n_unique <= max_knots
        if not knots_from_data:
            Xu = _subsample_knots(Xu, max_knots, seed=1)
        self._Xu = Xu

        # Steps 6-10: shared tprs_setup knot construction.
        basis = _source_tprs_knot_basis(Xu, self._m, k)

        # Step 11: training rows. tprs_setup maps every datum, ties included,
        # to its knot row; supplied (subsampled) and pure knots use tps_g.
        if knots_from_data and basis.knot_rows is not None:
            X_design = basis.knot_rows[inverse]
        else:
            X_design = self._project_design_rows(X_centered, Xu, basis.UZ)

        # Step 12: column scaling to unit RMS, in tprs_setup's operation order.
        squares = _SourceColumnSquares(k)
        squares.add(X_design)
        self._apply_source_scaling(basis, squares.scales(), knots_from_data)
        X_design = X_design / self._col_norms[np.newaxis, :]

        # Apply shrinkage before smoothCon normalization (matching R's order:
        # smooth.construct modifies S, then smoothCon normalizes)
        S = self._apply_shrinkage(self._S)

        # Step 13: smoothCon penalty normalization
        [S], self._s_scale = self._smoothcon_normalize(X_design, [S])

        # Step 14: Store results
        self._X = X_design
        self._S = S
        self.n_coefs = k
        self.rank = k - self._M
        self.null_space_dim = self._M
        self._is_setup = True

    def _apply_source_scaling(
        self,
        basis: _TPRSKnotBasis,
        scales: npt.NDArray[np.float64],
        knots_from_data: bool,
    ) -> None:
        """Store tprs_setup's column-scaled UZ, S and training knot rows."""
        self._col_norms = scales
        self._UZ = basis.UZ / scales[np.newaxis, :]
        self._S = _source_scaled_penalty(basis.S, scales)
        self._knots_from_data = knots_from_data and basis.knot_rows is not None
        self._knot_rows = (
            basis.knot_rows / scales[np.newaxis, :] if self._knots_from_data else None
        )

    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Return the design matrix for the given data.

        The setup-owned training matrix is consumed by model assembly, so this
        always evaluates the supplied data through the prediction transform.
        """
        self._require_setup()
        return self.predict_matrix(data)

    def build_penalty_matrices(self) -> list[Penalty]:
        """Return the TPRS penalty matrix.

        Returns
        -------
        list[Penalty]
            Single penalty matrix for the TPRS smooth.
        """
        self._require_setup()
        return [
            Penalty(
                self._S,
                rank=self.rank,
                null_space_dim=self.null_space_dim,
            )
        ]

    def training_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Return source training rows for data drawn from the setup source.

        tprs_setup's model matrix maps each datum to its knot row
        ``U diag(v) Z`` (scaled), which differs from the prediction transform
        by the Lanczos residual. Subsampled knots use the prediction rows, as
        tprs_setup's ``tps_g`` branch does.
        """
        self._require_setup()
        if not self._knots_from_data:
            return self.predict_matrix(data)
        if self._knot_rows is None:
            raise RuntimeError(
                "This thin-plate prediction copy no longer owns its training "
                "knot rows; use copy_for_training() for source replay."
            )
        raw = np.column_stack(
            [np.asarray(data[v], dtype=float) for v in self.spec.variables]
        )
        return self._knot_rows[_source_knot_indices(raw - self._shift, self._Xu)]

    def predict_matrix(
        self, new_data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build prediction matrix for new data.

        Parameters
        ----------
        new_data : dict[str, np.ndarray]
            Must contain keys matching ``self.spec.variables``.

        Returns
        -------
        np.ndarray
            Prediction matrix, shape ``(n_new, k)``.
        """
        self._require_setup()

        raw = np.column_stack(
            [np.asarray(new_data[v], dtype=float) for v in self.spec.variables]
        )
        # Centre by stored shift
        X_centered = raw - self._shift

        return self._project_design_rows(X_centered, self._Xu, self._UZ)

    def _project_design_rows(
        self,
        X_centered: npt.NDArray[np.floating],
        knots: npt.NDArray[np.floating],
        UZ: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Evaluate and project TPS rows without retaining an n-by-r-by-d array."""
        n_rows = X_centered.shape[0]
        X_design = np.empty((n_rows, UZ.shape[1]), dtype=float)
        for start in range(0, n_rows, DISTANCE_BATCH_ROWS):
            stop = min(start + DISTANCE_BATCH_ROWS, n_rows)
            rows = X_centered[start:stop]
            E_rows = _source_tps_rows(rows, knots, self._m)
            T_rows = compute_polynomial_basis(rows, self._m)
            X_design[start:stop] = source_project_rows(
                np.column_stack([E_rows, T_rows]), UZ
            )
        return X_design


def _nearest_knot_indices(
    X: npt.NDArray[np.floating], Xu: npt.NDArray[np.floating]
) -> npt.NDArray[np.intp]:
    """Find nearest knot index for each data point."""
    D = _compute_distance_matrix(X, Xu)
    return np.argmin(D, axis=1)


# ---------------------------------------------------------------------------
# TPRSShrinkageSmooth (ts)
# ---------------------------------------------------------------------------


class TPRSShrinkageSmooth(TPRSSmooth):
    """Thin plate regression spline with shrinkage penalty (bs="ts").

    After the standard TPRS construction, the zero eigenvalues of S
    (corresponding to the polynomial null space) are replaced with
    small positive values, making S full rank. This allows the smooth
    to be penalized to zero, which is useful for model selection.

    Parameters
    ----------
    spec : SmoothSpec
        Smooth term specification.
    """

    #: Shrinkage factor from R's ``attr(object,"shrink") <- 1e-1``.
    _shrink: float = 0.1

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct ts basis: standard TPRS + shrinkage penalty."""
        super().setup(data)
        self.null_space_dim = 0
        self.rank = self._k

    def _apply_shrinkage(self, S: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Apply uniform shrinkage matching R's ts."""
        return self._decompose_and_replace(
            S, self._k, self._shrink, self._uniform_replacement
        )
