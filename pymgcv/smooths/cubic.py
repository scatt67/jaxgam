"""Cubic regression splines (cr, cs, cc basis types).

Implements cubic regression spline basis and penalty construction
following Wood (2006) 'Generalized Additive Models', pp 145-147,
and R's mgcv smooth.construct.cr.smooth.spec().

Basis and penalty matrices are constructed from first principles
using the algorithms described in Wood (2006). No external basis
construction library is used.

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 5.3
R source reference: R/smooth.r smooth.construct.cr.smooth.spec()
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import linalg

from pymgcv.formula.terms import SmoothSpec
from pymgcv.penalties.penalty import Penalty
from pymgcv.smooths.base import Smooth

# ---------------------------------------------------------------------------
# Knot placement
# ---------------------------------------------------------------------------


def _place_knots(x: npt.NDArray[np.floating], k: int) -> npt.NDArray[np.floating]:
    """Place k knots at equally-spaced rank positions through unique values.

    Ports R's ``place.knots()`` exactly: linearly interpolate k
    equally-spaced fractional indices through the sorted unique values.

    Parameters
    ----------
    x : np.ndarray
        1D covariate values.
    k : int
        Number of knots to place.

    Returns
    -------
    np.ndarray
        Sorted knot locations, shape ``(k,)``.
    """
    x_unique = np.sort(np.unique(x))
    n = len(x_unique)
    indices = np.linspace(0, n - 1, k)
    return np.interp(indices, np.arange(n), x_unique)


# ---------------------------------------------------------------------------
# Basis matrix construction (Wood 2006, pp 145-147)
# ---------------------------------------------------------------------------


def _find_knot_intervals(
    x: npt.NDArray[np.floating], knots: npt.NDArray[np.floating]
) -> npt.NDArray[np.intp]:
    """Find the knot interval index for each x value.

    Returns array ``j`` such that ``knots[j[i]] <= x[i] < knots[j[i]+1]``,
    clamped to ``[0, len(knots) - 2]``.
    """
    j = np.searchsorted(knots, x) - 1
    return np.clip(j, 0, len(knots) - 2)


def _compute_base_functions(
    x: npt.NDArray[np.floating], knots: npt.NDArray[np.floating]
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.intp],
]:
    """Compute the 4 base functions for cubic spline evaluation.

    Following Wood (2006) p. 146 and mgcv's ``crspl()`` for
    linear extrapolation beyond the knot range.

    Returns ``(ajm, ajp, cjm, cjp, j)`` where ajm/ajp are linear
    interpolation weights and cjm/cjp are cubic correction weights.
    """
    j = _find_knot_intervals(x, knots)

    h = np.diff(knots)
    hj = h[j]
    xj1_x = knots[j + 1] - x
    x_xj = x - knots[j]

    # Linear interpolation weights
    ajm = xj1_x / hj
    ajp = x_xj / hj

    # Cubic correction weights (linear extrapolation beyond boundaries)
    cjm_3 = xj1_x**3 / (6.0 * hj)
    cjm_3[x > knots[-1]] = 0.0
    cjm_1 = hj * xj1_x / 6.0
    cjm = cjm_3 - cjm_1

    cjp_3 = x_xj**3 / (6.0 * hj)
    cjp_3[x < knots[0]] = 0.0
    cjp_1 = hj * x_xj / 6.0
    cjp = cjp_3 - cjp_1

    return ajm, ajp, cjm, cjp, j


def _natural_f(knots: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Mapping from spline values to 2nd derivatives (natural BC).

    Returns F of shape ``(k, k)`` where ``f''(knots) = F @ f(knots)``.
    First and last rows are zero (natural boundary conditions:
    ``f''`` = 0 at endpoints).

    Wood (2006) pp 145-146.
    """
    h = np.diff(knots)
    k = len(knots)

    # B matrix (k-2, k-2) tridiagonal — banded storage
    diag = (h[:-1] + h[1:]) / 3.0
    off_diag = h[1:-1] / 6.0
    banded_B = np.array(
        [
            np.r_[0.0, off_diag],
            diag,
            np.r_[off_diag, 0.0],
        ]
    )

    # D matrix (k-2, k)
    D = np.zeros((k - 2, k))
    for i in range(k - 2):
        D[i, i] = 1.0 / h[i]
        D[i, i + 1] = -(1.0 / h[i] + 1.0 / h[i + 1])
        D[i, i + 2] = 1.0 / h[i + 1]

    # F_inner = B^{-1} @ D, shape (k-2, k)
    F_inner = linalg.solve_banded((1, 1), banded_B, D)

    # Pad with zero rows for natural boundary conditions
    return np.vstack([np.zeros(k), F_inner, np.zeros(k)])


def _cyclic_f(knots: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Mapping from spline values to 2nd derivatives (cyclic BC).

    Returns F of shape ``(k-1, k-1)`` where
    ``f''(knots[:-1]) = F @ f(knots[:-1])``.
    The last knot wraps to the first.

    Wood (2006) pp 146-147.
    """
    h = np.diff(knots)
    n = len(knots) - 1

    # B and D matrices (n, n) — circulant tridiagonal with wrap-around
    B = np.zeros((n, n))
    D = np.zeros((n, n))

    # Wrap-around corner entries
    B[0, 0] = (h[n - 1] + h[0]) / 3.0
    B[0, n - 1] = h[n - 1] / 6.0
    B[n - 1, 0] = h[n - 1] / 6.0

    D[0, 0] = -1.0 / h[0] - 1.0 / h[n - 1]
    D[0, n - 1] = 1.0 / h[n - 1]
    D[n - 1, 0] = 1.0 / h[n - 1]

    for i in range(1, n):
        B[i, i] = (h[i - 1] + h[i]) / 3.0
        B[i, i - 1] = h[i - 1] / 6.0
        B[i - 1, i] = h[i - 1] / 6.0

        D[i, i] = -1.0 / h[i - 1] - 1.0 / h[i]
        D[i, i - 1] = 1.0 / h[i - 1]
        D[i - 1, i] = 1.0 / h[i - 1]

    return np.linalg.solve(B, D)


def _map_cyclic(
    x: npt.NDArray[np.floating], lbound: float, ubound: float
) -> npt.NDArray[np.floating]:
    """Map values into ``[lbound, ubound]`` cyclically."""
    x = np.copy(x)
    period = ubound - lbound
    mask_above = x > ubound
    mask_below = x < lbound
    x[mask_above] = lbound + (x[mask_above] - ubound) % period
    x[mask_below] = ubound - (lbound - x[mask_below]) % period
    return x


def _build_basis_matrix(
    x: npt.NDArray[np.floating],
    knots: npt.NDArray[np.floating],
    cyclic: bool = False,
) -> npt.NDArray[np.floating]:
    """Build cubic regression spline basis matrix.

    Natural (``cyclic=False``): returns ``(n, k)`` matrix.
    Cyclic (``cyclic=True``): returns ``(n, k-1)`` matrix.

    Wood (2006) p. 145, eq. 4.2.

    Parameters
    ----------
    x : np.ndarray
        Data values, shape ``(n,)``.
    knots : np.ndarray
        Sorted knot locations, shape ``(k,)``.
    cyclic : bool
        If True, build cyclic (periodic) basis.

    Returns
    -------
    np.ndarray
        Basis matrix.
    """
    k = len(knots)
    n_basis = k - 1 if cyclic else k

    if cyclic:
        x = _map_cyclic(x, knots[0], knots[-1])

    ajm, ajp, cjm, cjp, j = _compute_base_functions(x, knots)

    j1 = j + 1
    if cyclic:
        j1[j1 == k - 1] = 0

    # F matrix: maps spline values to 2nd derivatives
    F = _cyclic_f(knots) if cyclic else _natural_f(knots)

    # Build basis: X[i,:] = ajm*e_j + ajp*e_j1 + cjm*F[j,:] + cjp*F[j1,:]
    eye = np.eye(n_basis)
    X = (
        ajm[:, np.newaxis] * eye[j, :]
        + ajp[:, np.newaxis] * eye[j1, :]
        + cjm[:, np.newaxis] * F[j, :]
        + cjp[:, np.newaxis] * F[j1, :]
    )
    return X


# ---------------------------------------------------------------------------
# Penalty matrix construction
# ---------------------------------------------------------------------------


def _build_natural_penalty(
    knots: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Build penalty for natural cubic regression spline.

    Computes ``S = D.T @ B^{-1} @ D`` where D is the ``(k-2, k)``
    second-difference matrix and B is the ``(k-2, k-2)`` tridiagonal
    mass matrix. S has shape ``(k, k)``, rank ``k-2``, and a 2D null
    space spanned by constant and linear functions.
    """
    k = len(knots)
    h = np.diff(knots)

    # D matrix (k-2, k)
    D = np.zeros((k - 2, k))
    for i in range(k - 2):
        D[i, i] = 1.0 / h[i]
        D[i, i + 1] = -(1.0 / h[i] + 1.0 / h[i + 1])
        D[i, i + 2] = 1.0 / h[i + 1]

    # B matrix (k-2, k-2) — banded storage for tridiagonal solve
    diag = (h[:-1] + h[1:]) / 3.0
    off_diag = h[1:-1] / 6.0
    banded_B = np.array(
        [
            np.r_[0.0, off_diag],
            diag,
            np.r_[off_diag, 0.0],
        ]
    )

    # S = D.T @ (B^{-1} @ D)
    F_inner = linalg.solve_banded((1, 1), banded_B, D)
    S = D.T @ F_inner
    S = 0.5 * (S + S.T)
    return S


def _build_cyclic_penalty(
    knots: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Build penalty for cyclic cubic regression spline.

    Computes ``S = D.T @ B^{-1} @ D`` where D and B are ``(k-1, k-1)``
    circulant tridiagonal matrices with wrap-around corners. S has
    shape ``(k-1, k-1)``, rank ``k-2``, and a 1D null space (constants).
    """
    h = np.diff(knots)
    n = len(knots) - 1

    B = np.zeros((n, n))
    D = np.zeros((n, n))

    # Wrap-around corner entries
    B[0, 0] = (h[n - 1] + h[0]) / 3.0
    B[0, n - 1] = h[n - 1] / 6.0
    B[n - 1, 0] = h[n - 1] / 6.0

    D[0, 0] = -1.0 / h[0] - 1.0 / h[n - 1]
    D[0, n - 1] = 1.0 / h[n - 1]
    D[n - 1, 0] = 1.0 / h[n - 1]

    for i in range(1, n):
        B[i, i] = (h[i - 1] + h[i]) / 3.0
        B[i, i - 1] = h[i - 1] / 6.0
        B[i - 1, i] = h[i - 1] / 6.0

        D[i, i] = -1.0 / h[i - 1] - 1.0 / h[i]
        D[i, i - 1] = 1.0 / h[i - 1]
        D[i - 1, i] = 1.0 / h[i - 1]

    # S = D.T @ (B^{-1} @ D)
    F = np.linalg.solve(B, D)
    S = D.T @ F
    S = 0.5 * (S + S.T)
    return S


# ---------------------------------------------------------------------------
# Smooth classes
# ---------------------------------------------------------------------------


class CubicRegressionSmooth(Smooth):
    """Natural cubic regression spline smooth (bs="cr").

    Constructs basis and penalty matrices for a natural cubic
    regression spline with k knots placed at equally-spaced rank
    positions through the unique covariate values.

    Parameters
    ----------
    spec : SmoothSpec
        Smooth term specification.
    """

    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)
        self._X: npt.NDArray[np.floating] | None = None
        self._S: npt.NDArray[np.floating] | None = None
        self._knots: npt.NDArray[np.floating] | None = None
        self._k: int = 0
        self._is_setup: bool = False
        self._cyclic: bool = False

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct cubic regression spline basis from data.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Must contain the key matching ``self.spec.variables[0]``.
        """
        if len(self.spec.variables) != 1:
            raise ValueError(
                f"Cubic regression splines are univariate, got "
                f"{len(self.spec.variables)} variables."
            )

        x = np.asarray(data[self.spec.variables[0]], dtype=float)

        # Determine basis dimension
        k = self.spec.k
        if k == -1:
            k = 10  # R default for cr
        if k < 3:
            raise ValueError(
                f"Basis dimension k={k} must be at least 3 for cubic splines."
            )
        self._k = k

        n_unique = len(np.unique(x))
        if k > n_unique:
            raise ValueError(
                f"Basis dimension k={k} exceeds number of unique data values "
                f"({n_unique}). Reduce k or add more distinct data."
            )

        # Place knots and build basis + penalty
        knots = _place_knots(x, k)
        self._knots = knots
        self._X = _build_basis_matrix(x, knots, cyclic=self._cyclic)
        self._S = self._build_penalty(knots)

        # Apply shrinkage before normalization (matching R's order:
        # smooth.construct modifies S, then smoothCon normalizes)
        self._S = self._apply_shrinkage(self._S)

        # Normalize penalty: S = S / S.scale
        # Replicates R's smoothCon() normalization:
        #   maXX = norm(X, type="I")^2
        #   S.scale = norm(S, type="O") / maXX
        norm_X_inf = np.linalg.norm(self._X, ord=np.inf)
        norm_S_1 = np.linalg.norm(self._S, ord=1)
        maXX = norm_X_inf**2
        if maXX > 0:
            self._s_scale = norm_S_1 / maXX
            self._S = self._S / self._s_scale
        else:
            self._s_scale = 1.0

        # Set dimensions
        if self._cyclic:
            self.n_coefs = k - 1
            self.rank = k - 2
            self.null_space_dim = 1
        else:
            self.n_coefs = k
            self.rank = k - 2
            self.null_space_dim = 2

        self._is_setup = True

    def _apply_shrinkage(self, S: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Hook for shrinkage penalty modification. No-op in base class."""
        return S

    def _build_penalty(
        self, knots: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Build the penalty matrix. Overridden in CyclicCubicSmooth."""
        return _build_natural_penalty(knots)

    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Return the design matrix for the given data."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before build_design_matrix().")
        return self.predict_matrix(data)

    def build_penalty_matrices(self) -> list[Penalty]:
        """Return the cubic spline penalty matrix."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before build_penalty_matrices().")
        return [
            Penalty(
                self._S,
                rank=self.rank,
                null_space_dim=self.null_space_dim,
            )
        ]

    def predict_matrix(
        self, new_data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build prediction matrix for new data."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before predict_matrix().")
        x = np.asarray(new_data[self.spec.variables[0]], dtype=float)
        return _build_basis_matrix(x, self._knots, cyclic=self._cyclic)


class CubicShrinkageSmooth(CubicRegressionSmooth):
    """Cubic regression spline with shrinkage penalty (bs="cs").

    After the standard cr construction, the zero eigenvalues of S
    (corresponding to the polynomial null space) are replaced with
    small positive values, making S full rank. This allows the smooth
    to be penalized to zero for model selection.

    Parameters
    ----------
    spec : SmoothSpec
        Smooth term specification.
    """

    #: Shrinkage factor from R's ``attr(object,"shrink") <- .1``.
    _shrink: float = 0.1

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct cs basis: standard cr + shrinkage penalty."""
        super().setup(data)
        self.null_space_dim = 0
        self.rank = self._k

    def _apply_shrinkage(self, S: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Apply chained geometric shrinkage matching R's cs.

        R source (smooth.r, smooth.construct.cs.smooth.spec):
          es$values[nk-1] <- es$values[nk-2]*shrink
          es$values[nk]   <- es$values[nk-1]*shrink
        Each successive null eigenvalue is shrink times the previous.

        Uses additive update (S + delta) rather than full reconstruction
        to preserve range-space eigenvalues exactly.
        """
        # Use driver='evr' (dsyevr) to match R's eigen(symmetric=TRUE)
        eigvals, eigvecs = linalg.eigh(S, driver="evr")

        tol = np.max(np.abs(eigvals)) * self._k * np.finfo(float).eps
        nonzero_mask = np.abs(eigvals) > tol
        null_rank = int(np.sum(~nonzero_mask))

        if null_rank > 0:
            if np.any(nonzero_mask):
                smallest_nonzero = np.min(np.abs(eigvals[nonzero_mask]))
            else:
                smallest_nonzero = 1.0
            # Chained geometric: ascending order gets
            # [smallest * shrink^null_rank, ..., smallest * shrink]
            factors = self._shrink ** np.arange(null_rank, 0, -1)
            eigvals[:null_rank] = smallest_nonzero * factors

        S_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return 0.5 * (S_new + S_new.T)


class CyclicCubicSmooth(CubicRegressionSmooth):
    """Cyclic cubic regression spline smooth (bs="cc").

    The cyclic variant constrains the spline to be periodic over the
    knot range, reducing the number of coefficients by 1 (k-1 instead
    of k). The penalty null space is 1D (constants only).

    Parameters
    ----------
    spec : SmoothSpec
        Smooth term specification.
    """

    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)
        self._cyclic = True

    def _build_penalty(
        self, knots: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Build the cyclic penalty matrix."""
        return _build_cyclic_penalty(knots)
