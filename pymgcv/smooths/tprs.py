"""Thin plate regression splines (tp, ts basis types).

Implements the TPRS basis and penalty construction following
R's mgcv smooth.construct.tp.smooth.spec() and src/tprs.c.

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 5.2
R source reference: R/smooth.r smooth.construct.tp.smooth.spec(),
                    src/tprs.c construct_tprs()
"""

from __future__ import annotations

from math import ceil, comb, factorial, pi

import numba
import numpy as np
import numpy.typing as npt
from scipy.special import gamma as gamma_func

from pymgcv.formula.terms import SmoothSpec
from pymgcv.penalties.penalty import Penalty
from pymgcv.smooths.base import Smooth

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
    alpha = 2 * m - d
    if alpha % 2 == 0:
        # 2m - d is even
        sign = (-1) ** (m + 1 + d // 2)
        denom = (
            2 ** (2 * m - 1) * pi ** (d / 2) * factorial(m - 1) * factorial(m - d // 2)
        )
        return sign / denom
    else:
        # 2m - d is odd
        numer = gamma_func(d / 2 - m)
        denom = 2 ** (2 * m) * pi ** (d / 2) * factorial(m - 1)
        return numer / denom


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

    Generates all monomials of total degree < m evaluated at the
    data points X. The ordering matches R's convention.

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

    # Generate monomials of total degree < m
    # For d=1: [1, x] when m=2; [1, x, x²] when m=3
    # For d=2: [1, x1, x2] when m=2; [1, x1, x2, x1², x1*x2, x2²] when m=3
    col = 0
    if d == 1:
        for p in range(m):
            T[:, col] = X[:, 0] ** p
            col += 1
    else:
        # General d: enumerate monomials by total degree
        _fill_polynomial_basis(T, X, m, d)

    return T


def _fill_polynomial_basis(
    T: npt.NDArray[np.floating],
    X: npt.NDArray[np.floating],
    m: int,
    d: int,
) -> None:
    """Fill polynomial basis for general d using recursive enumeration.

    Fills T in-place with monomials x1^a1 * x2^a2 * ... * xd^ad
    for all (a1, ..., ad) with a1 + ... + ad < m, ordered by
    total degree then lexicographic order.
    """
    n = X.shape[0]
    col = 0

    # Generate all multi-indices with total degree < m
    indices = _monomial_indices(d, m)
    for idx in indices:
        val = np.ones(n)
        for j, power in enumerate(idx):
            if power > 0:
                val = val * X[:, j] ** power
        T[:, col] = val
        col += 1


def _monomial_indices(d: int, max_degree: int) -> list[tuple[int, ...]]:
    """Generate multi-indices for monomials of total degree < max_degree.

    Returns tuples (a1, ..., ad) ordered by total degree, then
    lexicographically within each degree.
    """
    if d == 1:
        return [(p,) for p in range(max_degree)]

    indices: list[tuple[int, ...]] = []
    for total in range(max_degree):
        indices.extend(_partitions(d, total))
    return indices


def _partitions(d: int, total: int) -> list[tuple[int, ...]]:
    """Generate all d-tuples of non-negative integers summing to total.

    Ordered lexicographically (reverse of standard, matching R convention).
    """
    if d == 1:
        return [(total,)]
    result: list[tuple[int, ...]] = []
    for first in range(total, -1, -1):
        for rest in _partitions(d - 1, total - first):
            result.append((first, *rest))
    return result


def _default_k(d: int, M: int) -> int:
    """Default basis dimension k for dimension d.

    R's defaults: M + {8, 27, 100}[d-1] for d in {1, 2, 3+}.
    """
    defaults = {1: 10, 2: 30}
    return defaults.get(d, M + 100)


def _compute_distance_matrix(
    X1: npt.NDArray[np.floating],
    X2: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute pairwise Euclidean distance matrix.

    Parameters
    ----------
    X1 : np.ndarray
        Shape ``(n1, d)``.
    X2 : np.ndarray
        Shape ``(n2, d)``.

    Returns
    -------
    np.ndarray
        Distance matrix, shape ``(n1, n2)``.
    """
    # Use broadcasting for efficiency
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def _get_unique_rows(
    X: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp]]:
    """Get unique rows and inverse mapping, sorted lexicographically.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n, d)``.

    Returns
    -------
    Xu : np.ndarray
        Unique rows, shape ``(n_unique, d)``, sorted lexicographically.
    inverse : np.ndarray
        Index array such that ``Xu[inverse] == X`` (up to float tolerance).
    """
    # Round to handle floating-point duplicates
    # Use np.unique with axis=0 which sorts lexicographically
    Xu, inverse = np.unique(X, axis=0, return_inverse=True)
    return Xu, inverse


@numba.njit(
    numba.types.Tuple((numba.float64[:], numba.float64[:, :]))(
        numba.float64[:, ::1], numba.int64, numba.float64
    ),
    cache=True,
)
def _slanczos_jit(A, k, tol):  # pragma: no cover  # noqa: C901
    n = A.shape[0]

    # --- Deterministic starting vector (R's LCG) ---
    q0 = np.empty(n)
    jran = 1
    for i in range(n):
        jran = (jran * 106 + 1283) % 6075
        q0[i] = jran / 6075.0 - 0.5
    q0 /= np.linalg.norm(q0)

    # --- Lanczos iteration ---
    Q = np.empty((n, n))
    Q[:, 0] = q0
    alpha = np.empty(n)
    beta = np.empty(n)

    # Convergence check frequency (matching R)
    f_check = k // 2
    if f_check < 10:
        f_check = 10
    kk = n // 10
    if kk < 1:
        kk = 1
    if kk < f_check:
        f_check = kk

    j_final = n
    n_pos = 0
    n_neg = 0
    converged = False
    d = np.zeros(1)
    v_tri = np.zeros((1, 1))

    for j in range(n):
        qj = np.ascontiguousarray(Q[:, j])
        z = A @ qj
        alpha[j] = qj @ z

        if j == 0:
            z -= alpha[0] * qj
        else:
            z -= alpha[j] * qj + beta[j - 1] * np.ascontiguousarray(Q[:, j - 1])
            # Double reorthogonalization (CGS via BLAS gemv)
            Qj = np.ascontiguousarray(Q[:, : j + 1])
            for _pass in range(2):
                z -= Qj @ (Qj.T @ z)

        beta[j] = np.linalg.norm(z)

        if j < n - 1:
            Q[:, j + 1] = z / beta[j]

        # --- Convergence check ---
        if not ((j >= k and j % f_check == 0) or j == n - 1):
            continue

        # Build tridiagonal matrix and eigendecompose
        size = j + 1
        T_mat = np.zeros((size, size))
        for idx in range(size):
            T_mat[idx, idx] = alpha[idx]
        for idx in range(j):
            T_mat[idx, idx + 1] = beta[idx]
            T_mat[idx + 1, idx] = beta[idx]
        d, v_tri = np.linalg.eigh(T_mat)
        # Reverse to descending order
        d = d[::-1].copy()
        v_tri = v_tri[:, ::-1].copy()

        # Error bounds: |beta_j * last component of kth Ritz vector|
        norm_Tj = max(abs(d[0]), abs(d[-1]))
        max_err = norm_Tj * tol
        err = np.abs(beta[j] * v_tri[-1, :])

        # Biggest mode: greedily walk from both ends by magnitude
        pi = 0
        ni = 0
        ok = True
        while pi + ni < k:
            if abs(d[pi]) >= abs(d[j - ni]):
                if err[pi] > max_err:
                    ok = False
                    break
                pi += 1
            else:
                if err[ni] > max_err:
                    ok = False
                    break
                ni += 1

        if ok:
            j_final = j + 1
            n_pos = pi
            n_neg = ni
            converged = True
            break

    if not converged:
        j_final = n
        pi = 0
        ni = 0
        while pi + ni < k:
            if abs(d[pi]) >= abs(d[n - 1 - ni]):
                pi += 1
            else:
                ni += 1
        n_pos = pi
        n_neg = ni

    # --- Build output eigenvalues and Ritz vectors ---
    pos_idx = np.arange(n_pos)
    neg_idx = np.arange(j_final - n_neg, j_final)
    sel = np.concatenate((pos_idx, neg_idx))

    D = d[sel]
    Q_cont = np.ascontiguousarray(Q[:, :j_final])
    U = Q_cont @ np.ascontiguousarray(v_tri[:, sel])

    return D, U


def _slanczos(
    A: npt.NDArray[np.floating],
    k: int,
    tol: float | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Lanczos eigendecomposition matching R's mgcv Rlanczos (biggest mode).

    Reimplements the Rlanczos function from mgcv/src/mat.c with minus=-1
    (largest magnitude eigenvalues). Uses the same deterministic LCG
    starting vector, double reorthogonalization, and convergence
    criteria as R.

    JIT-compiled via Numba for native performance.

    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix, shape ``(n, n)``.
    k : int
        Number of eigenvalues/vectors to compute (largest magnitude).
    tol : float, optional
        Convergence tolerance. Default: ``np.finfo(float).eps ** 0.7``.

    Returns
    -------
    D : np.ndarray
        Eigenvalues, shape ``(k,)``. Positive eigenvalues first
        (descending), then negative eigenvalues.
    U : np.ndarray
        Eigenvectors, shape ``(n, k)``.
    """
    if tol is None:
        tol = np.finfo(float).eps ** 0.7
    return _slanczos_jit(A, k, tol)


@numba.njit(numba.float64[:, :](numba.float64[:, :]), cache=True)
def _null_space_basis_r_jit(TU):  # pragma: no cover
    M = TU.shape[0]
    k = TU.shape[1]
    A = TU.copy()

    # --- Phase 1: Householder reflectors (R's QT, fullQ=0) ---
    reflectors = np.zeros((M, k))
    for i in range(M):
        n_elem = k - i
        v = A[i, :n_elem]

        # Scale for numerical stability
        m_scale = 0.0
        for j in range(n_elem):
            av = abs(v[j])
            if av > m_scale:
                m_scale = av
        if m_scale > 0.0:
            for j in range(n_elem):
                v[j] /= m_scale

        lsq = np.linalg.norm(v)
        if v[n_elem - 1] < 0.0:
            lsq = -lsq

        v[n_elem - 1] += lsq
        if lsq != 0.0:
            g = 1.0 / (lsq * v[n_elem - 1])
        else:
            g = 0.0
        lsq *= m_scale

        # Apply reflector to remaining rows of A
        if g != 0.0:
            for j in range(i + 1, M):
                dot_val = 0.0
                for jj in range(n_elem):
                    dot_val += v[jj] * A[j, jj]
                for jj in range(n_elem):
                    A[j, jj] -= g * dot_val * v[jj]

        # Store reflector scaled by sqrt(g)
        sqrt_g = np.sqrt(g) if g > 0.0 else 0.0
        for j in range(n_elem):
            reflectors[i, j] = v[j] * sqrt_g

        # Overwrite row i with R factor
        for j in range(n_elem - 1):
            A[i, j] = 0.0
        A[i, n_elem - 1] = -lsq

    # --- Phase 2: Build Q = H_0 @ H_1 @ ... @ H_{M-1} (R's HQmult) ---
    Q = np.eye(k)
    for i in range(M):
        u = reflectors[i]
        Cu = Q @ u
        Q -= np.outer(Cu, u)

    return Q[:, : k - M]


def _null_space_basis_r(
    TU: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Null space basis of TU via right-Householder QR matching R's mgcv.

    Reimplements R's ``QT`` and ``HQmult`` functions from
    ``mgcv/src/matrix.c`` to produce an identical null space basis.
    R applies Householder reflectors from the right on TU (M x k),
    yielding Q such that TU @ Q = [0 | T]. The first k-M columns
    of Q span the right null space of TU.

    JIT-compiled via Numba for native performance.

    Parameters
    ----------
    TU : np.ndarray
        Constraint projection matrix, shape ``(M, k)`` with ``M < k``.

    Returns
    -------
    Z : np.ndarray
        Null space basis, shape ``(k, k - M)``.
    """
    return _null_space_basis_r_jit(TU)


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
        self._m: int = 0
        self._M: int = 0
        self._k: int = 0
        self._d: int = 0
        self._is_setup: bool = False

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct TPRS basis from data.

        Follows R's smooth.construct.tp.smooth.spec() algorithm:
        1. Centre covariates
        2. Get unique knots
        3. Build kernel matrix E and polynomial basis T
        4. Eigendecompose E
        5. QR factorization for null space absorption
        6. Build design matrix X and penalty S
        7. Column normalize

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Must contain keys matching ``self.spec.variables``.
        """
        # Step 1: Extract and centre covariates
        d = len(self.spec.variables)
        self._d = d
        cols = [np.asarray(data[v], dtype=float) for v in self.spec.variables]
        raw = np.column_stack(cols)
        n = raw.shape[0]

        self._shift = raw.mean(axis=0)
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
        if n_unique > max_knots:
            rng = np.random.RandomState(1)  # seed=1 to match R
            idx = rng.choice(n_unique, max_knots, replace=False)
            idx.sort()
            Xu = Xu[idx]
            n_unique = max_knots
            # Recompute inverse mapping for subsampled knots
            inverse = _nearest_knot_indices(X_centered, Xu)

        self._Xu = Xu
        nk = n_unique

        # Step 6: Build E (kernel matrix at knots)
        R_knots = _compute_distance_matrix(Xu, Xu)
        E = tps_semi_kernel(R_knots, self._m, d)

        # Step 7: Build T (polynomial null space at knots)
        T = compute_polynomial_basis(Xu, self._m)

        # Step 8: Eigendecompose E via Lanczos (matching R's Rlanczos)
        D_k, U_k = _slanczos(E, k)

        # Step 9: Null space via right-Householder QR (matching R's QT)
        TU = T.T @ U_k  # (M, k)
        Z = _null_space_basis_r(TU)  # (k, k-M)

        # Step 10: Build UZ matrix
        # UZ is (nk + M, k) where:
        #   UZ[:nk, :k-M] = U_k @ Z  (wiggly part)
        #   UZ[nk:, k-M:] = I_M      (polynomial part)
        k_wiggly = k - self._M
        UZ = np.zeros((nk + self._M, k))
        UZ[:nk, :k_wiggly] = U_k @ Z
        UZ[nk:, k_wiggly:] = np.eye(self._M)

        # Step 11: Build X (design matrix)
        knots_are_data = (nk == n) and np.array_equal(Xu[inverse], X_centered)

        if knots_are_data:
            # X = [(U_k * D_k) @ Z | T][inverse_map]
            X_wiggly = (U_k * D_k[np.newaxis, :]) @ Z  # (nk, k-M)
            X_full_knots = np.column_stack([X_wiggly, T])  # (nk, k)
            X_design = X_full_knots[inverse]
        else:
            # Build E from data to knots, then T from data
            E_data = tps_semi_kernel(
                _compute_distance_matrix(X_centered, Xu), self._m, d
            )
            T_data = compute_polynomial_basis(X_centered, self._m)
            # X = [E_data | T_data] @ UZ
            ET = np.column_stack([E_data, T_data])  # (n, nk+M)
            X_design = ET @ UZ  # (n, k)

        # Step 12: Build S (penalty matrix)
        # S = Z' @ diag(D_k) @ Z, padded to k×k with zeros for null space
        S_wiggly = Z.T @ np.diag(D_k) @ Z  # (k-M, k-M)
        S = np.zeros((k, k))
        S[:k_wiggly, :k_wiggly] = S_wiggly

        # Step 13: Column normalize (RMS = 1)
        rms = np.sqrt(np.mean(X_design**2, axis=0))
        rms[rms == 0] = 1.0  # avoid division by zero
        X_design = X_design / rms[np.newaxis, :]
        # Apply same scaling to S and UZ
        S = S / np.outer(rms, rms)
        UZ = UZ / rms[np.newaxis, :]

        # Force symmetry of S
        S = 0.5 * (S + S.T)

        # Step 14: smoothCon penalty normalization
        # Replicates R's smoothCon(): S.scale = norm(S,"O") / norm(X,"I")^2
        norm_X_inf = np.linalg.norm(X_design, ord=np.inf)
        norm_S_1 = np.linalg.norm(S, ord=1)
        maXX = norm_X_inf**2
        if maXX > 0:
            self._s_scale = norm_S_1 / maXX
            S = S / self._s_scale
        else:
            self._s_scale = 1.0

        # Step 15: Store results
        self._X = X_design
        self._S = S
        self._UZ = UZ
        self._col_norms = rms
        self.n_coefs = k
        self.rank = k_wiggly
        self.null_space_dim = self._M
        self._is_setup = True

    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Return the design matrix for the given data.

        If ``data`` matches the setup data, returns the stored matrix.
        Otherwise builds a new prediction matrix.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before build_design_matrix().")
        # For the training data, return stored X
        # For new data, use predict_matrix
        cols = [np.asarray(data[v], dtype=float) for v in self.spec.variables]
        raw = np.column_stack(cols)

        if self._X is not None and raw.shape[0] == self._X.shape[0]:
            X_centered = raw - self._shift
            # Check if this is the same data (fast approximate check)
            E_data = tps_semi_kernel(
                _compute_distance_matrix(X_centered, self._Xu), self._m, self._d
            )
            T_data = compute_polynomial_basis(X_centered, self._m)
            ET = np.column_stack([E_data, T_data])
            X_new = ET @ self._UZ
            return X_new

        return self.predict_matrix(data)

    def build_penalty_matrices(self) -> list[Penalty]:
        """Return the TPRS penalty matrix.

        Returns
        -------
        list[Penalty]
            Single penalty matrix for the TPRS smooth.
        """
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
        if not self._is_setup:
            raise RuntimeError("Call setup() before predict_matrix().")

        raw = np.column_stack(
            [np.asarray(new_data[v], dtype=float) for v in self.spec.variables]
        )
        # Centre by stored shift
        X_centered = raw - self._shift

        # E_new = kernel(new_data, knots)
        E_new = tps_semi_kernel(
            _compute_distance_matrix(X_centered, self._Xu), self._m, self._d
        )
        # T_new = polynomial basis at new data
        T_new = compute_polynomial_basis(X_centered, self._m)

        # [E_new | T_new] @ UZ
        ET = np.column_stack([E_new, T_new])
        return ET @ self._UZ


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

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct ts basis: standard TPRS + shrinkage penalty."""
        # First do the standard TPRS setup
        super().setup(data)

        # Make S full rank by replacing zero eigenvalues
        eigvals, eigvecs = np.linalg.eigh(self._S)

        # Find smallest nonzero eigenvalue
        tol = np.max(np.abs(eigvals)) * self._k * np.finfo(float).eps
        nonzero_mask = np.abs(eigvals) > tol
        if np.any(nonzero_mask):
            smallest_nonzero = np.min(np.abs(eigvals[nonzero_mask]))
        else:
            smallest_nonzero = 1.0

        # Replace zero eigenvalues with smallest_nonzero * shrink_factor
        # R uses TRUE which coerces to 1.0
        shrink_factor = self.spec.extra_args.get("shrink", 1.0)
        replacement = smallest_nonzero * shrink_factor

        eigvals[~nonzero_mask] = replacement

        # Reconstruct S
        self._S = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self._S = 0.5 * (self._S + self._S.T)  # force symmetry

        # Update rank and null space
        self.null_space_dim = 0
        self.rank = self._k
