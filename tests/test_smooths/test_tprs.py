"""Tests for TPRS basis and penalty construction.

Validates TPRSSmooth and TPRSShrinkageSmooth from pymgcv.smooths.tprs:
- Helper function unit tests (STRICT)
- Structural invariant tests (STRICT)
- R comparison tests (MODERATE, skip if R unavailable)
- Parameterized tests for various k values
- Edge cases
- Phase boundary guard (no JAX imports)

Design doc reference: docs/design.md Section 5.2
R source reference: R/smooth.r smooth.construct.tp.smooth.spec()
"""

from __future__ import annotations

import importlib
import sys
from math import comb, pi

import numpy as np
import pytest

from pymgcv.formula.terms import SmoothSpec
from pymgcv.penalties.penalty import Penalty
from pymgcv.smooths.tprs import (
    TPRSShrinkageSmooth,
    TPRSSmooth,
    _monomial_indices,
    _nearest_knot_indices,
    compute_polynomial_basis,
    default_penalty_order,
    eta_const,
    null_space_dimension,
    tps_semi_kernel,
)
from tests.tolerances import MODERATE, STRICT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    variables: list[str],
    bs: str = "tp",
    k: int = 10,
    **extra_args: object,
) -> SmoothSpec:
    """Create a SmoothSpec for testing."""
    return SmoothSpec(
        variables=variables,
        bs=bs,
        k=k,
        extra_args=dict(extra_args),
    )


def _make_1d_data(n: int = 200, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate simple 1D test data."""
    rng = np.random.default_rng(seed)
    return {"x": rng.uniform(0, 1, n)}


def _make_2d_data(n: int = 200, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate simple 2D test data."""
    rng = np.random.default_rng(seed)
    return {"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)}


# ===========================================================================
# 1. Helper function unit tests (STRICT)
# ===========================================================================


class TestDefaultPenaltyOrder:
    """Tests for default_penalty_order()."""

    def test_d1(self) -> None:
        assert default_penalty_order(1) == 2

    def test_d2(self) -> None:
        assert default_penalty_order(2) == 2

    def test_d3(self) -> None:
        assert default_penalty_order(3) == 3

    def test_d4(self) -> None:
        assert default_penalty_order(4) == 3


class TestNullSpaceDimension:
    """Tests for null_space_dimension()."""

    def test_d1_m2(self) -> None:
        """d=1, m=2 → M=2 (constant + linear)."""
        assert null_space_dimension(1, 2) == 2

    def test_d2_m2(self) -> None:
        """d=2, m=2 → M=3 (1, x1, x2)."""
        assert null_space_dimension(2, 2) == 3

    def test_d3_m3(self) -> None:
        """d=3, m=3 → M=10."""
        assert null_space_dimension(3, 3) == 10

    def test_d1_m3(self) -> None:
        """d=1, m=3 → M=3 (1, x, x²)."""
        assert null_space_dimension(1, 3) == 3

    def test_formula(self) -> None:
        """General formula: M = comb(m+d-1, d)."""
        for d in range(1, 5):
            for m in range(1, 5):
                assert null_space_dimension(d, m) == comb(m + d - 1, d)


class TestEtaConst:
    """Tests for eta_const() — TPS semi-kernel constant."""

    def test_d1_m2(self) -> None:
        """d=1, m=2 → c = 1/12 (so eta = r³/12)."""
        c = eta_const(2, 1)
        np.testing.assert_allclose(c, 1.0 / 12.0, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_d2_m2(self) -> None:
        """d=2, m=2 → c = 1/(8π) (so eta = r²log(r)/(8π))."""
        c = eta_const(2, 2)
        expected = 1.0 / (8.0 * pi)
        np.testing.assert_allclose(c, expected, rtol=STRICT.rtol, atol=STRICT.atol)


class TestTPSSemiKernel:
    """Tests for tps_semi_kernel()."""

    def test_eta_zero_is_zero(self) -> None:
        """eta(0) = 0 for all m, d."""
        r = np.array([0.0])
        for d in [1, 2]:
            m = default_penalty_order(d)
            result = tps_semi_kernel(r, m, d)
            np.testing.assert_allclose(
                result,
                [0.0],
                atol=STRICT.atol,
                err_msg=f"eta(0) != 0 for d={d}, m={m}",
            )

    def test_d1_m2_known(self) -> None:
        """d=1, m=2: eta(r) = r³/12."""
        r = np.array([0.0, 1.0, 2.0, 3.0])
        result = tps_semi_kernel(r, 2, 1)
        expected = r**3 / 12.0
        np.testing.assert_allclose(result, expected, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_d2_m2_known(self) -> None:
        """d=2, m=2: eta(r) = r²*log(r)/(8π)."""
        r = np.array([0.0, 1.0, 2.0, 3.0])
        c = 1.0 / (8.0 * pi)
        expected = np.zeros_like(r)
        mask = r > 0
        expected[mask] = c * r[mask] ** 2 * np.log(r[mask])
        result = tps_semi_kernel(r, 2, 2)
        np.testing.assert_allclose(result, expected, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_non_negative_r(self) -> None:
        """Kernel is real-valued for non-negative r."""
        rng = np.random.default_rng(0)
        r = rng.uniform(0, 10, 50)
        for d, m in [(1, 2), (2, 2)]:
            result = tps_semi_kernel(r, m, d)
            msg = f"Non-finite kernel values for d={d}, m={m}"
            assert np.all(np.isfinite(result)), msg


class TestComputePolynomialBasis:
    """Tests for compute_polynomial_basis()."""

    def test_d1_m2_shape(self) -> None:
        """d=1, m=2 → T is (n, 2) with columns [1, x]."""
        X = np.array([[1.0], [2.0], [3.0]])
        T = compute_polynomial_basis(X, m=2)
        assert T.shape == (3, 2)
        np.testing.assert_allclose(T[:, 0], [1.0, 1.0, 1.0], atol=STRICT.atol)
        np.testing.assert_allclose(T[:, 1], [1.0, 2.0, 3.0], atol=STRICT.atol)

    def test_d2_m2_shape(self) -> None:
        """d=2, m=2 → T is (n, 3) with columns [1, x1, x2]."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        T = compute_polynomial_basis(X, m=2)
        assert T.shape == (2, 3)
        np.testing.assert_allclose(T[:, 0], [1.0, 1.0], atol=STRICT.atol)
        np.testing.assert_allclose(T[:, 1], [1.0, 3.0], atol=STRICT.atol)
        np.testing.assert_allclose(T[:, 2], [2.0, 4.0], atol=STRICT.atol)

    def test_d1_m3_shape(self) -> None:
        """d=1, m=3 → T is (n, 3) with columns [1, x, x²]."""
        X = np.array([[0.0], [1.0], [2.0]])
        T = compute_polynomial_basis(X, m=3)
        assert T.shape == (3, 3)
        np.testing.assert_allclose(T[:, 0], [1.0, 1.0, 1.0], atol=STRICT.atol)
        np.testing.assert_allclose(T[:, 1], [0.0, 1.0, 2.0], atol=STRICT.atol)
        np.testing.assert_allclose(T[:, 2], [0.0, 1.0, 4.0], atol=STRICT.atol)


# ===========================================================================
# 2. Structural invariant tests (STRICT)
# ===========================================================================


class TestStructuralInvariants:
    """Structural properties that must hold for any TPRS smooth."""

    @pytest.mark.parametrize("k", [5, 10, 15])
    def test_S_symmetric_psd_1d(self, k: int) -> None:
        """S is symmetric PSD for 1D smooth."""
        spec = _make_spec(["x"], k=k)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=200)
        smooth.setup(data)

        penalties = smooth.build_penalty_matrices()
        S = penalties[0].S

        # Symmetric
        np.testing.assert_allclose(S, S.T, rtol=STRICT.rtol, atol=STRICT.atol)
        # PSD
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -STRICT.atol), (
            f"S has negative eigenvalue: {np.min(eigvals):.2e}"
        )

    def test_S_symmetric_psd_2d(self) -> None:
        """S is symmetric PSD for 2D smooth."""
        spec = _make_spec(["x1", "x2"], k=10)
        smooth = TPRSSmooth(spec)
        data = _make_2d_data(n=200)
        smooth.setup(data)

        S = smooth.build_penalty_matrices()[0].S
        np.testing.assert_allclose(S, S.T, rtol=STRICT.rtol, atol=STRICT.atol)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -STRICT.atol)

    @pytest.mark.parametrize("k", [5, 10, 20])
    def test_S_rank_equals_k_minus_M_1d(self, k: int) -> None:
        """For tp, S rank = k - M (M=2 for d=1)."""
        spec = _make_spec(["x"], k=k)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=200)
        smooth.setup(data)

        assert smooth.rank == k - 2
        penalty = smooth.build_penalty_matrices()[0]
        assert penalty.rank == k - 2
        assert penalty.null_space_dim == 2

    def test_S_rank_2d(self) -> None:
        """For tp with d=2, S rank = k - 3."""
        spec = _make_spec(["x1", "x2"], k=10)
        smooth = TPRSSmooth(spec)
        data = _make_2d_data(n=200)
        smooth.setup(data)

        assert smooth.rank == 10 - 3
        assert smooth.null_space_dim == 3

    @pytest.mark.parametrize("k", [5, 10, 15])
    def test_X_shape_1d(self, k: int) -> None:
        """X shape = (n, k) for 1D smooth."""
        n = 200
        spec = _make_spec(["x"], k=k)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=n)
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert X.shape == (n, k)

    def test_X_shape_2d(self) -> None:
        """X shape = (n, k) for 2D smooth."""
        n = 200
        spec = _make_spec(["x1", "x2"], k=15)
        smooth = TPRSSmooth(spec)
        data = _make_2d_data(n=n)
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert X.shape == (n, 15)

    def test_penalty_returns_list_of_penalty(self) -> None:
        """build_penalty_matrices returns list[Penalty]."""
        spec = _make_spec(["x"], k=10)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=200)
        smooth.setup(data)

        penalties = smooth.build_penalty_matrices()
        assert isinstance(penalties, list)
        assert len(penalties) == 1
        assert isinstance(penalties[0], Penalty)

    def test_predict_equals_design_matrix(self) -> None:
        """predict_matrix(original_data) == build_design_matrix(original_data)."""
        spec = _make_spec(["x"], k=10)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=100)
        smooth.setup(data)

        X_design = smooth.build_design_matrix(data)
        X_predict = smooth.predict_matrix(data)
        np.testing.assert_allclose(
            X_predict,
            X_design,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="predict_matrix should equal build_design_matrix for same data",
        )

    def test_penalty_null_space_aligned_with_polynomial(self) -> None:
        """Null space of S spans polynomial space."""
        spec = _make_spec(["x"], k=10)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=200)
        smooth.setup(data)

        S = smooth.build_penalty_matrices()[0].S
        eigvals, eigvecs = np.linalg.eigh(S)

        # Last M=2 eigenvalues should be ~0
        M = smooth.null_space_dim
        null_eigvals = eigvals[:M]
        assert np.all(np.abs(null_eigvals) < STRICT.atol), (
            f"Expected {M} near-zero eigenvalues, got {null_eigvals}"
        )

    def test_n_coefs_matches_k(self) -> None:
        """n_coefs equals k after setup."""
        spec = _make_spec(["x"], k=10)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=200)
        smooth.setup(data)
        assert smooth.n_coefs == 10


# ===========================================================================
# 3. R comparison tests (MODERATE, skip if R unavailable)
# ===========================================================================


def _r_available() -> bool:
    """Check if R bridge is available."""
    try:
        from pymgcv.compat.r_bridge import RBridge

        return RBridge.available()
    except Exception:
        return False


@pytest.mark.skipif(not _r_available(), reason="R with mgcv not available")
class TestRComparison:
    """Compare TPRS construction against R's smoothCon().

    Python reimplements R's Rlanczos eigensolver (mgcv/src/mat.c) to
    produce identical eigenvectors, so all downstream quantities (X, S)
    match R after column and smoothCon normalisation.
    """

    def _setup_1d(
        self,
    ) -> tuple:
        """Shared 1D setup for R comparison."""
        import pandas as pd

        from pymgcv.compat.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        data = pd.DataFrame({"x": x})

        bridge = RBridge()
        r_result = bridge.smooth_construct("s(x, bs='tp', k=10)", data)

        spec = _make_spec(["x"], k=10)
        smooth = TPRSSmooth(spec)
        smooth.setup({"x": x})
        return smooth, r_result, x

    def test_tp_1d_column_space_vs_r(self) -> None:
        """1D tp column space of X matches R (projection matrix)."""
        smooth, r_result, x = self._setup_1d()
        X_py = smooth.build_design_matrix({"x": x})
        X_r = r_result["X"]

        # Projection matrices: P = X @ pinv(X)
        # Using MODERATE tolerance because pinv introduces O(eps*cond) noise.
        P_py = X_py @ np.linalg.pinv(X_py)
        P_r = X_r @ np.linalg.pinv(X_r)

        np.testing.assert_allclose(
            P_py,
            P_r,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="1D tp column spaces differ from R",
        )

    def test_tp_1d_wiggly_subspace_vs_r(self) -> None:
        """Wiggly subspace (UZ columns) spans same space as R.

        Compares principal angles between wiggly column spaces.
        Both sides are normalized to unit-length columns before
        computing the cross-Gram SVD.
        """
        smooth, r_result, _x = self._setup_1d()
        nk = smooth._Xu.shape[0]
        k_w = smooth._k - smooth._M

        # Python: undo smoothCon normalization → U_k @ Z (orthonormal cols)
        UZ_w_py = smooth._UZ[:nk, :k_w] * smooth._col_norms[:k_w]

        # R: normalize columns to unit L2 norm
        UZ_w_r_raw = r_result["UZ"][:nk, :k_w]
        col_norms_r = np.sqrt(np.sum(UZ_w_r_raw**2, axis=0))
        UZ_w_r = UZ_w_r_raw / col_norms_r

        # Principal angles via SVD of cross product
        # All singular values = 1 iff subspaces are identical
        sv = np.linalg.svd(UZ_w_py.T @ UZ_w_r, compute_uv=False)

        np.testing.assert_allclose(
            sv,
            np.ones(k_w),
            atol=MODERATE.atol,
            err_msg="Wiggly subspaces differ between Python and R",
        )

    def test_tp_1d_X_gram_vs_r(self) -> None:
        """X Gram matrix (X @ X.T) matches mgcv.

        Validates _slanczos eigenvalues and eigenvector magnitudes:
        X @ X.T is sign-invariant (column sign flips cancel) but
        depends on the actual eigenvalues and basis vectors. Wrong
        Lanczos output would produce a different Gram matrix.
        """
        smooth, r_result, x = self._setup_1d()
        X_py = smooth.build_design_matrix({"x": x})
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py @ X_py.T,
            X_r @ X_r.T,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="X Gram matrix differs from mgcv",
        )

    def test_tp_1d_UZ_gram_vs_r(self) -> None:
        """UZ Gram matrix (UZ @ UZ.T) matches mgcv.

        UZ encodes the Lanczos eigenvectors (U_k) projected through
        the null space basis (Z), but NOT the eigenvalues D_k. The
        Gram matrix is sign-invariant and validates that _slanczos
        eigenvectors and _null_space_basis_r produce the same
        column geometry as R's smoothCon. D_k is validated
        separately via the X Gram and S eigenvalue tests.
        """
        smooth, r_result, _x = self._setup_1d()
        UZ_py = smooth._UZ
        UZ_r = r_result["UZ"]

        np.testing.assert_allclose(
            UZ_py @ UZ_py.T,
            UZ_r @ UZ_r.T,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="UZ Gram matrix differs from mgcv",
        )

    def test_tp_1d_S_eigenvalues_vs_r(self) -> None:
        """Penalty matrix S eigenvalues match mgcv.

        S eigenvalues are sign-invariant (immune to LAPACK eigenvector
        sign ambiguity) and directly encode the Lanczos eigenvalues D_k
        through S_wiggly = Z' @ diag(D_k) @ Z. This is the strongest
        validation that _slanczos eigenvalues are correct, complementing
        the X Gram test (which tests D_k² through X_wiggly).
        """
        smooth, r_result, _x = self._setup_1d()
        S_py = smooth._S
        S_r = r_result["S"][0]

        eigvals_py = np.linalg.eigvalsh(S_py)
        eigvals_r = np.linalg.eigvalsh(S_r)

        np.testing.assert_allclose(
            eigvals_py,
            eigvals_r,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="S eigenvalues differ from mgcv",
        )

    def test_tp_1d_rank_and_null_space_vs_r(self) -> None:
        """Rank and null_space_dim match R exactly."""
        smooth, r_result, _x = self._setup_1d()
        assert smooth.rank == r_result["rank"]
        assert smooth.null_space_dim == r_result["null_space_dim"]

    def test_tp_1d_shift_vs_r(self) -> None:
        """Centring shift matches R."""
        smooth, r_result, _x = self._setup_1d()
        np.testing.assert_allclose(
            smooth._shift,
            r_result["shift"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Centring shift does not match R",
        )

    def test_tp_1d_knots_vs_r(self) -> None:
        """Knot locations Xu match R."""
        smooth, r_result, _x = self._setup_1d()
        np.testing.assert_allclose(
            smooth._Xu,
            r_result["Xu"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Knot locations do not match R",
        )

    def _setup_2d(
        self,
    ) -> tuple:
        """Shared 2D setup for R comparison."""
        import pandas as pd

        from pymgcv.compat.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        data = pd.DataFrame({"x1": x1, "x2": x2})

        bridge = RBridge()
        r_result = bridge.smooth_construct("s(x1, x2, bs='tp', k=10)", data)

        spec = _make_spec(["x1", "x2"], k=10)
        smooth = TPRSSmooth(spec)
        smooth.setup({"x1": x1, "x2": x2})
        return smooth, r_result, x1, x2

    def test_tp_2d_column_space_vs_r(self) -> None:
        """2D tp column space of X matches R (projection matrix)."""
        smooth, r_result, x1, x2 = self._setup_2d()
        X_py = smooth.build_design_matrix({"x1": x1, "x2": x2})
        X_r = r_result["X"]

        P_py = X_py @ np.linalg.pinv(X_py)
        P_r = X_r @ np.linalg.pinv(X_r)

        np.testing.assert_allclose(
            P_py,
            P_r,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="2D tp column spaces differ from R",
        )

    def test_tp_2d_X_gram_vs_r(self) -> None:
        """2D X Gram matrix (X @ X.T) matches mgcv.

        Validates _slanczos eigenvalues and eigenvector magnitudes
        for the 2D case (M=3 null space, different eigenvalue
        structure than 1D).
        """
        smooth, r_result, x1, x2 = self._setup_2d()
        X_py = smooth.build_design_matrix({"x1": x1, "x2": x2})
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py @ X_py.T,
            X_r @ X_r.T,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="2D X Gram matrix differs from mgcv",
        )

    def test_tp_2d_UZ_gram_vs_r(self) -> None:
        """2D UZ Gram matrix (UZ @ UZ.T) matches mgcv.

        Validates eigenvector/null-space geometry for the 2D case.
        """
        smooth, r_result, _x1, _x2 = self._setup_2d()
        UZ_py = smooth._UZ
        UZ_r = r_result["UZ"]

        np.testing.assert_allclose(
            UZ_py @ UZ_py.T,
            UZ_r @ UZ_r.T,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="2D UZ Gram matrix differs from mgcv",
        )

    def test_tp_2d_S_eigenvalues_vs_r(self) -> None:
        """2D penalty matrix S eigenvalues match mgcv.

        Validates _slanczos eigenvalues for the 2D case where
        M=3 (larger null space than 1D).
        """
        smooth, r_result, _x1, _x2 = self._setup_2d()
        S_py = smooth._S
        S_r = r_result["S"][0]

        eigvals_py = np.linalg.eigvalsh(S_py)
        eigvals_r = np.linalg.eigvalsh(S_r)

        np.testing.assert_allclose(
            eigvals_py,
            eigvals_r,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="2D S eigenvalues differ from mgcv",
        )

    def test_ts_1d_column_space_vs_r(self) -> None:
        """ts column space matches R (same basis as tp)."""
        import pandas as pd

        from pymgcv.compat.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        data = pd.DataFrame({"x": x})

        bridge = RBridge()
        r_result = bridge.smooth_construct("s(x, bs='ts', k=10)", data)

        spec = _make_spec(["x"], bs="ts", k=10)
        smooth = TPRSShrinkageSmooth(spec)
        smooth.setup({"x": x})
        X_py = smooth.build_design_matrix({"x": x})
        X_r = r_result["X"]

        P_py = X_py @ np.linalg.pinv(X_py)
        P_r = X_r @ np.linalg.pinv(X_r)

        np.testing.assert_allclose(
            P_py,
            P_r,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="ts column spaces differ from R",
        )

    def test_ts_1d_rank_vs_r(self) -> None:
        """ts has full rank penalty."""
        import pandas as pd

        from pymgcv.compat.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        data = pd.DataFrame({"x": x})

        bridge = RBridge()
        r_result = bridge.smooth_construct("s(x, bs='ts', k=10)", data)

        spec = _make_spec(["x"], bs="ts", k=10)
        smooth = TPRSShrinkageSmooth(spec)
        smooth.setup({"x": x})

        assert smooth.rank == r_result["rank"]
        assert smooth.null_space_dim == 0


# ===========================================================================
# 4. Parameterized tests
# ===========================================================================


class TestParameterized:
    """Parameterized tests for various k values."""

    @pytest.mark.parametrize("k", [5, 10, 15, 20, 50])
    def test_various_k_1d(self, k: int) -> None:
        """TPRS works for various k values (1D)."""
        spec = _make_spec(["x"], k=k)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=200)
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert X.shape == (200, k)
        assert smooth.n_coefs == k
        assert smooth.rank == k - 2
        assert smooth.null_space_dim == 2

        penalties = smooth.build_penalty_matrices()
        assert len(penalties) == 1
        S = penalties[0].S
        assert S.shape == (k, k)

    def test_k_exceeds_n_raises(self) -> None:
        """k > n raises ValueError."""
        spec = _make_spec(["x"], k=50)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=20)
        with pytest.raises(ValueError, match="exceeds"):
            smooth.setup(data)

    def test_k_auto_increased_below_M_plus_1(self) -> None:
        """k < M+1 is auto-increased to M+1."""
        # For d=1, M=2, so k=2 should be increased to 3
        spec = _make_spec(["x"], k=2)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=200)
        smooth.setup(data)
        assert smooth.n_coefs == 3  # M+1 = 3

    def test_default_k(self) -> None:
        """Default k (k=-1) uses R's default."""
        spec = _make_spec(["x"], k=-1)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=200)
        smooth.setup(data)
        assert smooth.n_coefs == 10  # default for d=1


# ===========================================================================
# 5. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for TPRS smooths."""

    def test_duplicate_data_points(self) -> None:
        """Duplicate data points are handled correctly."""
        x = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3] * 20)
        spec = _make_spec(["x"], k=3)  # Only 3 unique values
        smooth = TPRSSmooth(spec)
        smooth.setup({"x": x})

        X = smooth.build_design_matrix({"x": x})
        assert X.shape == (120, 3)
        assert np.all(np.isfinite(X))

    def test_too_few_unique_values(self) -> None:
        """Raises error when unique values < k."""
        x = np.array([0.1, 0.2] * 50)
        spec = _make_spec(["x"], k=5)
        smooth = TPRSSmooth(spec)
        with pytest.raises(ValueError, match="unique"):
            smooth.setup({"x": x})

    def test_setup_required_for_design_matrix(self) -> None:
        """build_design_matrix before setup raises RuntimeError."""
        spec = _make_spec(["x"], k=10)
        smooth = TPRSSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.build_design_matrix({"x": np.zeros(10)})

    def test_setup_required_for_penalty(self) -> None:
        """build_penalty_matrices before setup raises RuntimeError."""
        spec = _make_spec(["x"], k=10)
        smooth = TPRSSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.build_penalty_matrices()

    def test_setup_required_for_predict(self) -> None:
        """predict_matrix before setup raises RuntimeError."""
        spec = _make_spec(["x"], k=10)
        smooth = TPRSSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.predict_matrix({"x": np.zeros(10)})

    def test_predict_new_data_different_n(self) -> None:
        """predict_matrix works with different n than training data."""
        spec = _make_spec(["x"], k=10)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=100)
        smooth.setup(data)

        new_data = _make_1d_data(n=50, seed=99)
        X_new = smooth.predict_matrix(new_data)
        assert X_new.shape == (50, 10)
        assert np.all(np.isfinite(X_new))

    def test_small_n(self) -> None:
        """Small n with appropriate k works."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 10)
        spec = _make_spec(["x"], k=5)
        smooth = TPRSSmooth(spec)
        smooth.setup({"x": x})

        X = smooth.build_design_matrix({"x": x})
        assert X.shape == (10, 5)


# ===========================================================================
# 6. TPRSShrinkageSmooth (ts) tests
# ===========================================================================


class TestTPRSShrinkageSmooth:
    """Tests for the ts (shrinkage) variant."""

    def test_ts_S_full_rank(self) -> None:
        """ts penalty S has full rank (k, not k-M)."""
        spec = _make_spec(["x"], bs="ts", k=10)
        smooth = TPRSShrinkageSmooth(spec)
        data = _make_1d_data(n=200)
        smooth.setup(data)

        assert smooth.rank == 10
        assert smooth.null_space_dim == 0

        penalty = smooth.build_penalty_matrices()[0]
        assert penalty.rank == 10
        assert penalty.null_space_dim == 0

    def test_ts_S_psd(self) -> None:
        """ts penalty S is PSD (all eigenvalues > 0)."""
        spec = _make_spec(["x"], bs="ts", k=10)
        smooth = TPRSShrinkageSmooth(spec)
        data = _make_1d_data(n=200)
        smooth.setup(data)

        S = smooth.build_penalty_matrices()[0].S
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals > -STRICT.atol), (
            f"ts penalty has negative eigenvalue: {np.min(eigvals):.2e}"
        )
        # All should be strictly positive
        assert np.all(eigvals > 0), "ts penalty should be strictly positive definite"

    def test_ts_basis_matches_tp(self) -> None:
        """ts and tp produce the same basis matrix X."""
        data = _make_1d_data(n=200)

        spec_tp = _make_spec(["x"], bs="tp", k=10)
        smooth_tp = TPRSSmooth(spec_tp)
        smooth_tp.setup(data)
        X_tp = smooth_tp.build_design_matrix(data)

        spec_ts = _make_spec(["x"], bs="ts", k=10)
        smooth_ts = TPRSShrinkageSmooth(spec_ts)
        smooth_ts.setup(data)
        X_ts = smooth_ts.build_design_matrix(data)

        np.testing.assert_allclose(
            X_ts,
            X_tp,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="ts and tp should have the same basis matrix",
        )

    def test_ts_predict_matches_tp(self) -> None:
        """ts predict_matrix equals tp predict_matrix."""
        data = _make_1d_data(n=100)

        spec_tp = _make_spec(["x"], bs="tp", k=10)
        smooth_tp = TPRSSmooth(spec_tp)
        smooth_tp.setup(data)

        spec_ts = _make_spec(["x"], bs="ts", k=10)
        smooth_ts = TPRSShrinkageSmooth(spec_ts)
        smooth_ts.setup(data)

        new_data = _make_1d_data(n=50, seed=99)
        X_tp = smooth_tp.predict_matrix(new_data)
        X_ts = smooth_ts.predict_matrix(new_data)

        np.testing.assert_allclose(
            X_ts,
            X_tp,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


# ===========================================================================
# 7. Registry tests
# ===========================================================================


class TestRegistry:
    """Tests for smooth class registry."""

    def test_tp_lookup(self) -> None:
        from pymgcv.smooths.registry import get_smooth_class

        assert get_smooth_class("tp") is TPRSSmooth

    def test_ts_lookup(self) -> None:
        from pymgcv.smooths.registry import get_smooth_class

        assert get_smooth_class("ts") is TPRSShrinkageSmooth

    def test_unknown_raises(self) -> None:
        from pymgcv.smooths.registry import get_smooth_class

        with pytest.raises(KeyError, match="Unknown"):
            get_smooth_class("xx")

    def test_case_insensitive(self) -> None:
        from pymgcv.smooths.registry import get_smooth_class

        assert get_smooth_class("TP") is TPRSSmooth


# ===========================================================================
# 8. Phase boundary guard (no JAX imports)
# ===========================================================================


class TestNoJaxImport:
    """Verify that pymgcv.smooths does not trigger JAX import."""

    def test_smooths_tprs_import_no_jax(self) -> None:
        """Importing pymgcv.smooths.tprs must not cause jax to be imported."""
        modules_to_remove = [
            key
            for key in sys.modules
            if key == "jax" or key.startswith("jax.") or key.startswith("pymgcv.")
        ]
        saved = {key: sys.modules.pop(key) for key in modules_to_remove}

        try:
            importlib.import_module("pymgcv.smooths.tprs")
            assert "jax" not in sys.modules, (
                "Importing pymgcv.smooths.tprs triggered a jax import. "
                "Phase 1 modules must not depend on JAX."
            )
        finally:
            for key in list(sys.modules):
                if key.startswith("pymgcv."):
                    sys.modules.pop(key, None)
            sys.modules.update(saved)

    def test_smooths_init_import_no_jax(self) -> None:
        """Importing pymgcv.smooths must not cause jax import."""
        modules_to_remove = [
            key
            for key in sys.modules
            if key == "jax" or key.startswith("jax.") or key.startswith("pymgcv.")
        ]
        saved = {key: sys.modules.pop(key) for key in modules_to_remove}

        try:
            importlib.import_module("pymgcv.smooths")
            assert "jax" not in sys.modules, (
                "Importing pymgcv.smooths triggered a jax import."
            )
        finally:
            for key in list(sys.modules):
                if key.startswith("pymgcv."):
                    sys.modules.pop(key, None)
            sys.modules.update(saved)

    def test_smooths_registry_import_no_jax(self) -> None:
        """Importing pymgcv.smooths.registry must not cause jax import."""
        modules_to_remove = [
            key
            for key in sys.modules
            if key == "jax" or key.startswith("jax.") or key.startswith("pymgcv.")
        ]
        saved = {key: sys.modules.pop(key) for key in modules_to_remove}

        try:
            importlib.import_module("pymgcv.smooths.registry")
            assert "jax" not in sys.modules, (
                "Importing pymgcv.smooths.registry triggered a jax import."
            )
        finally:
            for key in list(sys.modules):
                if key.startswith("pymgcv."):
                    sys.modules.pop(key, None)
            sys.modules.update(saved)


class TestCoveragePaths:
    """Tests for code paths not covered by main tests."""

    def test_monomial_indices_d1(self) -> None:
        """_monomial_indices handles d=1 path."""
        result = _monomial_indices(1, 3)
        assert result == [(0,), (1,), (2,)]

    def test_knot_subsampling(self) -> None:
        """Knot subsampling activates when n_unique > max_knots."""
        rng = np.random.RandomState(42)
        n = 100
        x = rng.randn(n)
        spec = _make_spec(["x"], k=5, max_knots=20)
        smooth = TPRSSmooth(spec)
        smooth.setup({"x": x})
        assert smooth._is_setup
        assert smooth._Xu.shape[0] == 20

    def test_nearest_knot_indices(self) -> None:
        """_nearest_knot_indices returns correct indices."""
        X = np.array([[0.0], [1.0], [2.0]])
        Xu = np.array([[0.5], [1.5]])
        idx = _nearest_knot_indices(X, Xu)
        np.testing.assert_array_equal(idx, [0, 0, 1])

    def test_build_design_matrix_different_n(self) -> None:
        """build_design_matrix falls back to predict_matrix for different n."""
        rng = np.random.RandomState(42)
        x_train = rng.randn(50)
        spec = _make_spec(["x"], k=5)
        smooth = TPRSSmooth(spec)
        smooth.setup({"x": x_train})
        x_new = rng.randn(30)
        X_new = smooth.build_design_matrix({"x": x_new})
        assert X_new.shape == (30, 5)

    def test_s_scale_fallback_when_maxx_zero(self) -> None:
        """_s_scale = 1.0 when X_design is all zeros (maXX == 0)."""
        spec = _make_spec(["x"], k=5)
        smooth = TPRSSmooth(spec)
        data = _make_1d_data(n=50)
        # Patch np.linalg.norm to return 0.0 for the X inf-norm call
        # during smoothCon normalization (step 14)
        orig_norm = np.linalg.norm
        call_count = [0]

        def patched_norm(x, ord=None, **kwargs):
            result = orig_norm(x, ord=ord, **kwargs)
            # The smoothCon step computes norm(X, inf) then norm(S, 1)
            # norm(X, inf) is the first call with ord=inf
            if ord == np.inf:
                call_count[0] += 1
                return 0.0
            return result

        import unittest.mock

        with unittest.mock.patch("numpy.linalg.norm", side_effect=patched_norm):
            smooth.setup(data)
        assert smooth._s_scale == 1.0

    def test_ts_smallest_nonzero_fallback(self) -> None:
        """smallest_nonzero = 1.0 when all S eigenvalues are zero."""
        import unittest.mock

        spec = _make_spec(["x"], bs="ts", k=5)
        smooth = TPRSShrinkageSmooth(spec)
        data = _make_1d_data(n=50)

        # Patch TPRSSmooth.setup to zero out S before shrinkage logic runs
        orig_setup = TPRSSmooth.setup

        def patched_setup(self_inner, data_inner):
            orig_setup(self_inner, data_inner)
            self_inner._S = np.zeros_like(self_inner._S)

        with unittest.mock.patch.object(TPRSSmooth, "setup", patched_setup):
            smooth.setup(data)

        # After shrinkage with all-zero input eigenvalues, S should be
        # replacement * I where replacement = 1.0 * shrink_factor
        assert smooth.rank == 5
        assert smooth.null_space_dim == 0
        eigvals = np.linalg.eigvalsh(smooth._S)
        # All eigenvalues should be equal (= 1.0 * shrink_factor)
        np.testing.assert_allclose(eigvals, eigvals[0], rtol=STRICT.rtol)
