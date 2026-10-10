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

import ctypes
import functools
import math
import platform
import sys
import warnings

import numba
import numpy as np
import numpy.typing as npt
from scipy.linalg import blas

from jaxgam.formula.terms import SmoothSpec
from jaxgam.penalties.penalty import Penalty
from jaxgam.smooths.base import Smooth
from jaxgam.smooths.source_lapack import (
    cubic_shrinkage_penalty,
    natural_cubic_system_solve,
)


@functools.lru_cache(maxsize=1)
def _source_fma():
    """Return C's fused multiply-add, including on supported Python 3.11."""
    if hasattr(math, "fma"):
        return math.fma
    fused = ctypes.CDLL(None).fma
    fused.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    fused.restype = ctypes.c_double
    return fused


def _source_getfs_multiply_add():
    """Match the audited pinned R/mgcv Linux binary contraction.

    The audited R 4.5.2/mgcv 1.9-3 x86_64 binary uses separate MULSD/ADDSD;
    the audited aarch64 binary emits FMADD. This build-specific behavior is
    not a promise about arbitrary binaries with the same version strings.
    """
    if sys.platform.startswith("linux") and platform.machine() == "x86_64":
        return lambda a, b, c: a * b + c
    return _source_fma()


@functools.lru_cache(maxsize=1)
def _compiled_source_cubic_basis():
    """Compile the source-ordered FMA row loop once for macOS construction."""
    fused = ctypes.CDLL(None).fma
    fused.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    fused.restype = ctypes.c_double

    @numba.njit
    def fill(x, knots, F, intervals, X):
        n_basis = len(knots)
        for row in range(len(x)):
            xi = x[row]
            interval = int(intervals[row])
            if xi < knots[0]:
                h = knots[1] - knots[0]
                xik = xi - knots[0]
                cjm = -xik * h / 3.0
                cjp = -xik * h / 6.0
                for column in range(n_basis):
                    X[row, column] = fused(cjm, F[0, column], cjp * F[1, column])
                X[row, 0] += 1.0 - xik / h
                X[row, 1] += xik / h
            elif xi > knots[-1]:
                h = knots[-1] - knots[-2]
                xik = xi - knots[-1]
                cjm = xik * h / 6.0
                cjp = xik * h / 3.0
                left = n_basis - 2
                right = n_basis - 1
                for column in range(n_basis):
                    X[row, column] = fused(cjm, F[left, column], cjp * F[right, column])
                X[row, left] += -xik / h
                X[row, right] += 1.0 + xik / h
            else:
                xj = knots[interval]
                xj1 = knots[interval + 1]
                h = xj1 - xj
                ajm = xj1 - xi
                ajp = xi - xj
                cjm = (ajm * (ajm * ajm / h - h)) / 6.0
                cjp = (ajp * (ajp * ajp / h - h)) / 6.0
                ajm /= h
                ajp /= h
                for column in range(n_basis):
                    X[row, column] = fused(
                        cjm, F[interval, column], cjp * F[interval + 1, column]
                    )
                X[row, interval] += ajm
                X[row, interval + 1] += ajp

    return fill


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

    #: Minimum basis dimension. R's smooth.construct.cr.smooth.spec
    #: (R/smooth.r:1458-1462) bumps cr/cs to 3; cc overrides to 4.
    _min_k: int = 3

    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)
        self._X: npt.NDArray[np.floating] | None = None
        self._S: npt.NDArray[np.floating] | None = None
        self._F: npt.NDArray[np.floating] | None = None
        self._knots: npt.NDArray[np.floating] | None = None
        self._k: int = 0
        self._cyclic: bool = False
        # Skip SVD reparameterization in tensor products — cubic bases
        # are already well-conditioned (see base.py._noterp docstring).
        self._noterp: bool = True

    # ------------------------------------------------------------------
    # Static helpers (pure computation, no instance state)
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def _find_knot_intervals(
        x: npt.NDArray[np.floating], knots: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.intp]:
        """Find the knot interval index for each x value.

        Returns array ``j`` such that ``knots[j[i]] <= x[i] < knots[j[i]+1]``,
        clamped to ``[0, len(knots) - 2]``.
        """
        j = np.searchsorted(knots, x) - 1
        return np.clip(j, 0, len(knots) - 2)

    @staticmethod
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
        j = CubicRegressionSmooth._find_knot_intervals(x, knots)

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

    @staticmethod
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

    @staticmethod
    def _compute_natural_f_and_penalty(
        knots: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Compute F matrix and penalty S for natural cubic spline.

        Replays mgcv ``getFS`` (``src/mgcv.c``) and the pinned LAPACK
        DPTTRF/DPTTS2 scalar order. The explicit penalty contractions are
        load-bearing for ``bs="cs"``: last-bit changes in the rank-deficient
        raw penalty can rotate its two null eigenvectors before mgcv assigns
        unequal shrinkage eigenvalues.

        Returns
        -------
        F : np.ndarray
            Second-derivative mapping, shape ``(k, k)``.
        S : np.ndarray
            Penalty matrix ``D.T @ F_inner``, shape ``(k, k)``.
        """
        k = len(knots)
        inner = k - 2
        h = np.empty(k - 1)
        for i in range(1, k):
            h[i - 1] = knots[i] - knots[i - 1]

        # getFS builds D column-major, then DPTSV overwrites it with B^-1 D.
        D = np.zeros((inner, k))
        for i in range(inner):
            D[i, i] = 1.0 / h[i]
            D[i, i + 2] = 1.0 / h[i + 1]
            D[i, i + 1] = -D[i, i] - D[i, i + 2]

        diagonal = np.empty(inner)
        for i in range(inner):
            diagonal[i] = (h[i] + h[i + 1]) / 3.0
        off_diagonal = np.empty(inner - 1)
        for i in range(1, inner):
            off_diagonal[i - 1] = h[i] / 6.0

        D = natural_cubic_system_solve(diagonal, off_diagonal, D)

        # Python prediction consumes rows of F where crspl consumes columns.
        F = np.vstack([np.zeros(k), D, np.zeros(k)])

        # Exact getFS D'B^-1D contraction order, followed by smooth.r's
        # explicit ``(S + t(S))/2`` symmetry operation.
        multiply_add = _source_getfs_multiply_add()
        S = np.zeros((k, k))
        a = 1.0 / h[0]
        for column in range(k):
            S[0, column] = D[0, column] * a
        if k > 3:
            a = -1.0 / h[0] - 1.0 / h[1]
            b = 1.0 / h[1]
            for column in range(k):
                S[1, column] = multiply_add(D[0, column], a, D[1, column] * b)
            for row in range(2, inner):
                a = 1.0 / h[row - 1]
                c = 1.0 / h[row]
                b = -a - c
                for column in range(k):
                    S[row, column] = multiply_add(
                        D[row, column],
                        c,
                        multiply_add(D[row - 2, column], a, D[row - 1, column] * b),
                    )
            row = inner
            a = 1.0 / h[row - 1]
            b = -1.0 / h[row - 1] - 1.0 / h[row]
            for column in range(k):
                S[row, column] = multiply_add(
                    D[inner - 2, column], a, D[inner - 1, column] * b
                )
        else:
            a = -1.0 / h[0] - 1.0 / h[1]
            for column in range(k):
                S[1, column] = D[0, column] * a
        a = 1.0 / h[inner]
        for column in range(k):
            S[k - 1, column] = D[inner - 1, column] * a
        S = (S + S.T) * 0.5

        return F, S

    @staticmethod
    def _compute_cyclic_f_and_penalty(
        knots: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Compute F matrix and penalty S for cyclic cubic spline.

        Constructs B, D once, solves ``F = B^{-1} D`` once, and
        derives both F (for basis construction) and S (penalty) from the
        same intermediates.

        Returns
        -------
        F : np.ndarray
            Second-derivative mapping, shape ``(k-1, k-1)``.
        S : np.ndarray
            Penalty matrix ``D.T @ F``, shape ``(k-1, k-1)``.
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

        # Single solve
        F = np.linalg.solve(B, D)

        S = D.T @ F
        S = 0.5 * (S + S.T)

        return F, S

    # ------------------------------------------------------------------
    # Template methods (overridden by subclasses)
    # ------------------------------------------------------------------

    def _compute_f_and_penalty(
        self, knots: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Compute F and S matrices for this spline type.

        Overridden by CyclicCubicSmooth to use cyclic boundary conditions.

        Returns
        -------
        F : np.ndarray
            Second-derivative mapping matrix.
        S : np.ndarray
            Penalty matrix.
        """
        return self._compute_natural_f_and_penalty(knots)

    def _build_basis_matrix(
        self,
        x: npt.NDArray[np.floating],
        knots: npt.NDArray[np.floating],
        F: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Build cubic regression spline basis matrix.

        Natural: returns ``(n, k)`` matrix.
        Cyclic: returns ``(n, k-1)`` matrix.

        Wood (2006) p. 145, eq. 4.2.

        Parameters
        ----------
        x : np.ndarray
            Data values, shape ``(n,)``.
        knots : np.ndarray
            Sorted knot locations, shape ``(k,)``.
        F : np.ndarray
            Pre-computed second-derivative mapping matrix.

        Returns
        -------
        np.ndarray
            Basis matrix.
        """
        k = len(knots)
        n_basis = k - 1 if self._cyclic else k

        if self._cyclic:
            x = self._map_cyclic(x, knots[0], knots[-1])

        if not self._cyclic:
            j = self._find_knot_intervals(x, knots)
            X = np.zeros((len(x), n_basis))
            if sys.platform != "darwin":
                # Keep the pinned source BLAS path on Linux. Accelerate's
                # two-column GEMM rounds differently from mgcv's crspl FMA.
                left_rows = np.flatnonzero(x < knots[0])
                if left_rows.size:
                    h = knots[1] - knots[0]
                    xik = x[left_rows] - knots[0]
                    cjm = -xik * h / 3.0
                    cjp = -xik * h / 6.0
                    X[left_rows] = blas.dgemm(
                        1.0, np.column_stack((cjp, cjm)), np.vstack((F[1], F[0]))
                    )
                    X[left_rows, 0] += 1.0 - xik / h
                    X[left_rows, 1] += xik / h
                right_rows = np.flatnonzero(x > knots[-1])
                if right_rows.size:
                    h = knots[-1] - knots[-2]
                    xik = x[right_rows] - knots[-1]
                    cjm = xik * h / 6.0
                    cjp = xik * h / 3.0
                    X[right_rows] = blas.dgemm(
                        1.0, np.column_stack((cjp, cjm)), np.vstack((F[-1], F[-2]))
                    )
                    X[right_rows, -2] += -xik / h
                    X[right_rows, -1] += 1.0 + xik / h
                interior = (x >= knots[0]) & (x <= knots[-1])
                for interval in range(n_basis - 1):
                    rows = np.flatnonzero(interior & (j == interval))
                    if rows.size == 0:
                        continue
                    xj = knots[interval]
                    xj1 = knots[interval + 1]
                    h = xj1 - xj
                    ajm = xj1 - x[rows]
                    ajp = x[rows] - xj
                    cjm = (ajm * (ajm * ajm / h - h)) / 6.0
                    cjp = (ajp * (ajp * ajp / h - h)) / 6.0
                    coefficients = np.column_stack((cjp, cjm))
                    factors = np.vstack((F[interval + 1], F[interval]))
                    X[rows] = blas.dgemm(1.0, coefficients, factors)
                    X[rows, interval] += ajm / h
                    X[rows, interval + 1] += ajp / h
                return X

            _compiled_source_cubic_basis()(x, knots, F, j, X)
            return X

        ajm, ajp, cjm, cjp, j = self._compute_base_functions(x, knots)

        j1 = j + 1
        if self._cyclic:
            j1[j1 == k - 1] = 0

        # Build basis: X[i,:] = ajm*e_j + ajp*e_j1 + cjm*F[j,:] + cjp*F[j1,:]
        eye = np.eye(n_basis)
        X = (
            ajm[:, np.newaxis] * eye[j, :]
            + ajp[:, np.newaxis] * eye[j1, :]
            + cjm[:, np.newaxis] * F[j, :]
            + cjp[:, np.newaxis] * F[j1, :]
        )
        return X

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def setup(
        self,
        data: dict[str, npt.NDArray[np.floating]],
        *,
        _normalize_penalty: bool = True,
    ) -> None:
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

        # Determine basis dimension. mgcv applies the default first, then a
        # per-class minimum (cr/cs >= 3, cc >= 4 because the endpoints are
        # identified). Below the minimum R warns-and-bumps rather than erroring
        # (R/smooth.r:1458-1462 for cr, 1608-1611 for cc); a degenerate cc k=3
        # basis would otherwise be a silently non-mgcv model.
        k = self.spec.k
        if k == -1:
            k = 10  # R default for cr/cc
        if k < self._min_k:
            warnings.warn(
                f"basis dimension, k, increased to minimum possible "
                f"(k={k} -> {self._min_k}).",
                stacklevel=2,
            )
            k = self._min_k
        self._k = k

        n_unique = len(np.unique(x))
        if k > n_unique:
            raise ValueError(
                f"Basis dimension k={k} exceeds number of unique data values "
                f"({n_unique}). Reduce k or add more distinct data."
            )

        # Place knots and build F + S together (single solve)
        knots = self._place_knots(x, k)
        self._knots = knots

        F, S = self._compute_f_and_penalty(knots)

        self._F = F
        self._X = self._build_basis_matrix(x, knots, F)
        self._S = S

        # Apply shrinkage before normalization (matching R's order:
        # smooth.construct modifies S, then smoothCon normalizes)
        self._S = self._apply_shrinkage(self._S)

        # te() constructs unconstrained marginals with smooth.construct(),
        # which supplies the raw S directly. Normalizing and then undoing the
        # scale changes its last bits at a tied null-space rank boundary.
        if _normalize_penalty:
            [self._S], self._s_scale = self._smoothcon_normalize(self._X, [self._S])
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

    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Return the design matrix for the given data."""
        self._require_setup()
        return self.predict_matrix(data)

    def build_penalty_matrices(self) -> list[Penalty]:
        """Return the cubic spline penalty matrix."""
        self._require_setup()
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
        self._require_setup()
        x = np.asarray(new_data[self.spec.variables[0]], dtype=float)
        return self._build_basis_matrix(x, self._knots, self._F)


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

    def setup(
        self,
        data: dict[str, npt.NDArray[np.floating]],
        *,
        _normalize_penalty: bool = True,
    ) -> None:
        """Construct cs basis: standard cr + shrinkage penalty."""
        super().setup(data, _normalize_penalty=_normalize_penalty)
        self.null_space_dim = 0
        self.rank = self._k

    def _apply_shrinkage(
        self,
        S: npt.NDArray[np.floating],
        *,
        available_workspace_bytes: int | None = None,
    ) -> npt.NDArray[np.floating]:
        """Apply chained geometric shrinkage matching R's cs."""
        return cubic_shrinkage_penalty(
            np.asarray(S, dtype=np.float64),
            shrink=self._shrink,
            available_workspace_bytes=available_workspace_bytes,
        )


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

    #: Cyclic cubic splines need k>=4 (endpoints identified); R bumps below 4
    #: (R/smooth.r:1608-1611).
    _min_k: int = 4

    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)
        self._cyclic = True

    def _compute_f_and_penalty(
        self, knots: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Compute F and S matrices with cyclic boundary conditions."""
        return self._compute_cyclic_f_and_penalty(knots)
