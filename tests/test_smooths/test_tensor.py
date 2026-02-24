"""Tests for tensor product smooth basis and penalty construction.

Validates TensorProductSmooth (te) and TensorInteractionSmooth (ti)
from pymgcv.smooths.tensor:
- Row-wise Kronecker product tests (STRICT)
- Constraint absorption tests (STRICT)
- TensorProductSmooth structural tests (STRICT)
- TensorInteractionSmooth structural tests (STRICT)
- Marginal basis type tests
- R comparison tests (MODERATE, skip if R unavailable)
- Edge cases
- Phase boundary guard (no JAX imports)

Design doc reference: docs/design.md Section 5.5
R source reference: R/smooth.r smooth.construct.tensor.smooth.spec()
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from pymgcv.formula.terms import SmoothSpec
from pymgcv.penalties.penalty import Penalty
from pymgcv.smooths.tensor import (
    TensorInteractionSmooth,
    TensorProductSmooth,
    _row_tensor,
)
from tests.tolerances import (
    MODERATE,
    STRICT,
    normalize_column_signs,
    normalize_symmetric_signs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    variables: list[str],
    bs: str = "cr",
    k: int = 5,
    smooth_type: str = "te",
    **extra_args: object,
) -> SmoothSpec:
    """Create a SmoothSpec for testing."""
    return SmoothSpec(
        variables=variables,
        bs=bs,
        k=k,
        smooth_type=smooth_type,
        extra_args=dict(extra_args),
    )


def _make_2d_data(n: int = 200, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate 2D test data."""
    rng = np.random.default_rng(seed)
    return {"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)}


def _make_3d_data(n: int = 200, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate 3D test data."""
    rng = np.random.default_rng(seed)
    return {
        "x1": rng.uniform(0, 1, n),
        "x2": rng.uniform(0, 1, n),
        "x3": rng.uniform(0, 1, n),
    }


# ===========================================================================
# 1. Row-wise Kronecker product tests (STRICT)
# ===========================================================================


class TestRowTensor:
    """Tests for _row_tensor()."""

    def test_known_2x2_times_2x3(self) -> None:
        """Known 2x2 @ 2x3 example matches manual computation."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])

        result = _row_tensor(A, B)

        # Row 0: [1,2] kron [5,6,7] = [1*5, 1*6, 1*7, 2*5, 2*6, 2*7]
        expected_row0 = np.array([5.0, 6.0, 7.0, 10.0, 12.0, 14.0])
        # Row 1: [3,4] kron [8,9,10] = [3*8, 3*9, 3*10, 4*8, 4*9, 4*10]
        expected_row1 = np.array([24.0, 27.0, 30.0, 32.0, 36.0, 40.0])

        np.testing.assert_allclose(
            result[0], expected_row0, rtol=STRICT.rtol, atol=STRICT.atol
        )
        np.testing.assert_allclose(
            result[1], expected_row1, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_shape(self) -> None:
        """Shape: (n, ka) kron (n, kb) -> (n, ka*kb)."""
        rng = np.random.default_rng(42)
        n, ka, kb = 50, 4, 6
        A = rng.standard_normal((n, ka))
        B = rng.standard_normal((n, kb))

        result = _row_tensor(A, B)
        assert result.shape == (n, ka * kb)

    def test_matches_einsum(self) -> None:
        """Matches np.einsum('ni,nj->nij', A, B).reshape(n, ka*kb)."""
        rng = np.random.default_rng(42)
        n, ka, kb = 30, 3, 5
        A = rng.standard_normal((n, ka))
        B = rng.standard_normal((n, kb))

        result = _row_tensor(A, B)
        expected = np.einsum("ni,nj->nij", A, B).reshape(n, ka * kb)
        np.testing.assert_allclose(result, expected, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_associativity_shape(self) -> None:
        """Associativity: chained _row_tensor has correct shape for 3D tensors."""
        rng = np.random.default_rng(42)
        n, ka, kb, kc = 20, 3, 4, 5
        A = rng.standard_normal((n, ka))
        B = rng.standard_normal((n, kb))
        C = rng.standard_normal((n, kc))

        result = _row_tensor(_row_tensor(A, B), C)
        assert result.shape == (n, ka * kb * kc)


# ===========================================================================
# 2. Constraint absorption tests (STRICT)
# ===========================================================================


class TestAbsorbConstraint:
    """Tests for TensorInteractionSmooth._absorb_constraint()."""

    def test_absorbed_basis_shape(self) -> None:
        """Absorbed basis has k-1 columns."""
        rng = np.random.default_rng(42)
        n, k = 100, 10
        X = rng.standard_normal((n, k))
        S = np.eye(k)

        X_c, S_c, Z = TensorInteractionSmooth._absorb_constraint(X, S)
        assert X_c.shape == (n, k - 1)
        assert S_c.shape == (k - 1, k - 1)
        assert Z.shape == (k, k - 1)

    def test_constraint_satisfied(self) -> None:
        """Columns of X_c sum to ~0 (constraint satisfied)."""
        rng = np.random.default_rng(42)
        n, k = 100, 10
        X = rng.standard_normal((n, k))
        S = np.eye(k)

        X_c, _S_c, _Z = TensorInteractionSmooth._absorb_constraint(X, S)
        col_sums = X_c.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.zeros(k - 1), atol=STRICT.atol)

    def test_penalty_symmetric_psd(self) -> None:
        """Constrained penalty remains symmetric PSD."""
        from pymgcv.smooths.cubic import CubicRegressionSmooth

        spec = SmoothSpec(variables=["x"], bs="cr", k=10)
        smooth = CubicRegressionSmooth(spec)
        rng = np.random.default_rng(42)
        data = {"x": rng.uniform(0, 1, 200)}
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        S = smooth.build_penalty_matrices()[0].S * smooth._s_scale

        _X_c, S_c, _Z = TensorInteractionSmooth._absorb_constraint(X, S)

        # Symmetric
        np.testing.assert_allclose(S_c, S_c.T, rtol=STRICT.rtol, atol=STRICT.atol)

        # PSD
        eigvals = np.linalg.eigvalsh(S_c)
        assert np.all(eigvals >= -STRICT.atol), (
            f"Constrained penalty has negative eigenvalue: {np.min(eigvals):.2e}"
        )

    def test_z_orthonormal(self) -> None:
        """Z has orthonormal columns (Z.T @ Z = I)."""
        rng = np.random.default_rng(42)
        n, k = 100, 8
        X = rng.standard_normal((n, k))
        S = np.eye(k)

        _X_c, _S_c, Z = TensorInteractionSmooth._absorb_constraint(X, S)
        np.testing.assert_allclose(
            Z.T @ Z, np.eye(k - 1), rtol=STRICT.rtol, atol=STRICT.atol
        )


# ===========================================================================
# 3. TensorProductSmooth structural tests (STRICT)
# ===========================================================================


class TestTensorProductStructure:
    """Structural properties of TensorProductSmooth."""

    def test_te_shape_cr(self) -> None:
        """te(x1, x2, k=5, bs='cr') produces (n, 25) basis."""
        n = 200
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        data = _make_2d_data(n=n)
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert X.shape == (n, 25)

    def test_n_coefs(self) -> None:
        """n_coefs = product of marginal n_coefs."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(_make_2d_data())

        assert smooth.n_coefs == 5 * 5

    def test_null_space_dim_cr(self) -> None:
        """null_space_dim for cr: 2*2 = 4."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(_make_2d_data())

        assert smooth.null_space_dim == 2 * 2

    def test_penalty_count(self) -> None:
        """Penalty count equals number of marginals."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(_make_2d_data())

        penalties = smooth.build_penalty_matrices()
        assert len(penalties) == 2

    def test_penalty_shape(self) -> None:
        """Each penalty is (n_coefs, n_coefs)."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(_make_2d_data())

        for p in smooth.build_penalty_matrices():
            assert p.shape == (25, 25)

    def test_penalty_symmetry(self) -> None:
        """Penalty matrices are symmetric."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(_make_2d_data())

        for p in smooth.build_penalty_matrices():
            np.testing.assert_allclose(p.S, p.S.T, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_penalty_psd(self) -> None:
        """Penalty matrices are PSD."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(_make_2d_data())

        for p in smooth.build_penalty_matrices():
            eigvals = np.linalg.eigvalsh(p.S)
            assert np.all(eigvals >= -STRICT.atol), (
                f"Penalty has negative eigenvalue: {np.min(eigvals):.2e}"
            )

    def test_penalty_rank(self) -> None:
        """Penalty rank: rank(S_j) * product(d_i for i != j)."""
        k = 5
        spec = _make_spec(["x1", "x2"], bs="cr", k=k)
        smooth = TensorProductSmooth(spec)
        smooth.setup(_make_2d_data())

        penalties = smooth.build_penalty_matrices()
        # cr has rank k-2=3, so each tensor penalty has rank 3*5=15
        for p in penalties:
            assert p.rank == (k - 2) * k

    def test_predict_matches_design(self) -> None:
        """predict_matrix(train_data) matches build_design_matrix(train_data)."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        data = _make_2d_data()
        smooth.setup(data)

        X_design = smooth.build_design_matrix(data)
        X_predict = smooth.predict_matrix(data)
        np.testing.assert_allclose(
            X_predict, X_design, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_returns_penalty_objects(self) -> None:
        """build_penalty_matrices returns list[Penalty]."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(_make_2d_data())

        penalties = smooth.build_penalty_matrices()
        assert isinstance(penalties, list)
        for p in penalties:
            assert isinstance(p, Penalty)


# ===========================================================================
# 4. TensorInteractionSmooth structural tests (STRICT)
# ===========================================================================


class TestTensorInteractionStructure:
    """Structural properties of TensorInteractionSmooth."""

    def test_ti_shape_cr(self) -> None:
        """ti(x1, x2, k=5, bs='cr') produces (n, 16) basis (4*4)."""
        n = 200
        spec = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        data = _make_2d_data(n=n)
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert X.shape == (n, 16)

    def test_smaller_than_te(self) -> None:
        """ti columns < te columns for same k."""
        data = _make_2d_data()

        spec_te = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="te")
        te_smooth = TensorProductSmooth(spec_te)
        te_smooth.setup(data)

        spec_ti = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        ti_smooth = TensorInteractionSmooth(spec_ti)
        ti_smooth.setup(data)

        assert ti_smooth.n_coefs < te_smooth.n_coefs

    def test_subspace_of_te(self) -> None:
        """Column space of ti is a subspace of te (verified via projection)."""
        data = _make_2d_data()

        spec_te = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="te")
        te_smooth = TensorProductSmooth(spec_te)
        te_smooth.setup(data)
        X_te = te_smooth.build_design_matrix(data)

        spec_ti = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        ti_smooth = TensorInteractionSmooth(spec_ti)
        ti_smooth.setup(data)
        X_ti = ti_smooth.build_design_matrix(data)

        # Project X_ti onto column space of X_te
        # If X_ti is a subspace of X_te, then P @ X_ti ≈ X_ti
        Q_te, _ = np.linalg.qr(X_te, mode="reduced")
        X_ti_proj = Q_te @ (Q_te.T @ X_ti)
        np.testing.assert_allclose(
            X_ti_proj,
            X_ti,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="ti column space is not a subspace of te",
        )

    def test_null_space_dim(self) -> None:
        """null_space_dim: product of (marginal_nsd - 1) for each marginal."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(_make_2d_data())

        # cr has nsd=2, so constrained nsd = 2-1 = 1 per marginal
        # product = 1*1 = 1
        assert smooth.null_space_dim == 1

    def test_penalty_count(self) -> None:
        """ti has same number of penalties as marginals."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(_make_2d_data())

        penalties = smooth.build_penalty_matrices()
        assert len(penalties) == 2

    def test_penalty_shape(self) -> None:
        """ti penalty shape matches n_coefs."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(_make_2d_data())

        for p in smooth.build_penalty_matrices():
            assert p.shape == (16, 16)

    def test_penalty_symmetry(self) -> None:
        """ti penalty matrices are symmetric."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(_make_2d_data())

        for p in smooth.build_penalty_matrices():
            np.testing.assert_allclose(p.S, p.S.T, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_penalty_psd(self) -> None:
        """ti penalty matrices are PSD."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(_make_2d_data())

        for p in smooth.build_penalty_matrices():
            eigvals = np.linalg.eigvalsh(p.S)
            assert np.all(eigvals >= -STRICT.atol), (
                f"ti penalty has negative eigenvalue: {np.min(eigvals):.2e}"
            )

    def test_penalty_rank(self) -> None:
        """ti penalty rank: constrained_rank * product(constrained_d_i for i != j)."""
        k = 5
        spec = _make_spec(["x1", "x2"], bs="cr", k=k, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(_make_2d_data())

        penalties = smooth.build_penalty_matrices()
        # cr k=5: rank=3, nsd=2. Constrained: dim=4, nsd=1, rank=3.
        # Each tensor penalty has rank 3*4=12
        constrained_dim = k - 1  # 4
        constrained_rank = k - 2  # 3 (cr rank unchanged by constraint)
        for p in penalties:
            assert p.rank == constrained_rank * constrained_dim

    def test_predict_matches_design(self) -> None:
        """predict_matrix(train_data) matches build_design_matrix(train_data)."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        data = _make_2d_data()
        smooth.setup(data)

        X_design = smooth.build_design_matrix(data)
        X_predict = smooth.predict_matrix(data)
        np.testing.assert_allclose(
            X_predict, X_design, rtol=STRICT.rtol, atol=STRICT.atol
        )


# ===========================================================================
# 5. Marginal basis type tests
# ===========================================================================


class TestMarginalBasisTypes:
    """Tests for different marginal basis types."""

    def test_te_with_tprs(self) -> None:
        """te(x1, x2, bs='tp') works with TPRS marginals."""
        spec = _make_spec(["x1", "x2"], bs="tp", k=5)
        smooth = TensorProductSmooth(spec)
        data = _make_2d_data()
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert X.shape == (200, 25)
        assert np.all(np.isfinite(X))

    def test_te_with_cr(self) -> None:
        """te(x1, x2, bs='cr') works with cubic marginals."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        data = _make_2d_data()
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert X.shape == (200, 25)
        assert np.all(np.isfinite(X))

    def test_te_with_cc(self) -> None:
        """te(x1, x2, bs='cc') works with cyclic cubic marginals."""
        spec = _make_spec(["x1", "x2"], bs="cc", k=5)
        smooth = TensorProductSmooth(spec)
        data = _make_2d_data()
        smooth.setup(data)

        # cc has k-1 = 4 coefs per marginal
        X = smooth.build_design_matrix(data)
        assert X.shape == (200, 16)
        assert np.all(np.isfinite(X))

    def test_null_space_dim_tp(self) -> None:
        """null_space_dim for tp: 2*2 = 4 (for 1D marginals with m=2)."""
        spec = _make_spec(["x1", "x2"], bs="tp", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(_make_2d_data())

        assert smooth.null_space_dim == 2 * 2


# ===========================================================================
# 6. R comparison tests (MODERATE, skip if R unavailable)
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
    """Compare tensor product construction against R's smoothCon().

    With SVD reparameterization implemented, all basis types now match
    R element-wise. Cubic marginals have noterp=True (skip SVD reparam),
    TPRS marginals get SVD reparameterized to match R exactly.
    """

    def _setup_te_cr(self) -> tuple:
        """Setup te(x1, x2, bs='cr', k=5) for R comparison."""
        import pandas as pd

        from pymgcv.compat.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        data_pd = pd.DataFrame({"x1": x1, "x2": x2})

        bridge = RBridge()
        r_result = bridge.smooth_construct("te(x1, x2, bs='cr', k=5)", data_pd)

        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup({"x1": x1, "x2": x2})

        return smooth, r_result, {"x1": x1, "x2": x2}

    def _setup_te_tp(self) -> tuple:
        """Setup te(x1, x2, bs='tp', k=5) for R comparison."""
        import pandas as pd

        from pymgcv.compat.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        data_pd = pd.DataFrame({"x1": x1, "x2": x2})

        bridge = RBridge()
        r_result = bridge.smooth_construct("te(x1, x2, bs='tp', k=5)", data_pd)

        spec = _make_spec(["x1", "x2"], bs="tp", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup({"x1": x1, "x2": x2})

        return smooth, r_result, {"x1": x1, "x2": x2}

    def _setup_ti_cr(self) -> tuple:
        """Setup ti(x1, x2, bs='cr', k=5) for R comparison."""
        import pandas as pd

        from pymgcv.compat.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        data_pd = pd.DataFrame({"x1": x1, "x2": x2})

        bridge = RBridge()
        r_result = bridge.smooth_construct("ti(x1, x2, bs='cr', k=5)", data_pd)

        spec = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup({"x1": x1, "x2": x2})

        return smooth, r_result, {"x1": x1, "x2": x2}

    def _setup_ti_tp(self) -> tuple:
        """Setup ti(x1, x2, bs='tp', k=5) for R comparison."""
        import pandas as pd

        from pymgcv.compat.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        data_pd = pd.DataFrame({"x1": x1, "x2": x2})

        bridge = RBridge()
        r_result = bridge.smooth_construct("ti(x1, x2, bs='tp', k=5)", data_pd)

        spec = _make_spec(["x1", "x2"], bs="tp", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup({"x1": x1, "x2": x2})

        return smooth, r_result, {"x1": x1, "x2": x2}

    def test_te_cr_basis_vs_r(self) -> None:
        """te(x1, x2, k=5, bs='cr'): basis matches R element-wise (STRICT)."""
        smooth, r_result, data = self._setup_te_cr()
        X_py = smooth.build_design_matrix(data)
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py,
            X_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="te(cr) basis differs from R",
        )

    def test_te_tp_basis_vs_r(self) -> None:
        """te(x1, x2, k=5, bs='tp'): basis matches R element-wise (MODERATE).

        TPRS marginals undergo SVD reparameterization in both R and Python.
        Sign normalization handles LAPACK eigenvector sign ambiguity.
        """
        smooth, r_result, data = self._setup_te_tp()
        X_py = smooth.build_design_matrix(data)
        X_r = r_result["X"]

        np.testing.assert_allclose(
            normalize_column_signs(X_py),
            normalize_column_signs(X_r),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="te(tp) basis differs from R",
        )

    def test_te_cr_penalty_vs_r(self) -> None:
        """te(cr) penalty matrices match R element-wise (STRICT)."""
        smooth, r_result, _data = self._setup_te_cr()

        for j, (py_pen, r_S) in enumerate(
            zip(smooth.build_penalty_matrices(), r_result["S"], strict=True)
        ):
            np.testing.assert_allclose(
                py_pen.S,
                r_S,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"te(cr) penalty {j} differs from R",
            )

    def test_te_tp_penalty_vs_r(self) -> None:
        """te(tp) penalty matrices match R element-wise (MODERATE).

        Sign normalization handles LAPACK eigenvector sign ambiguity
        that propagates through SVD reparameterization.
        """
        smooth, r_result, data = self._setup_te_tp()
        X_py = smooth.build_design_matrix(data)
        X_r = r_result["X"]

        for j, (py_pen, r_S) in enumerate(
            zip(smooth.build_penalty_matrices(), r_result["S"], strict=True)
        ):
            np.testing.assert_allclose(
                normalize_symmetric_signs(py_pen.S, X_py),
                normalize_symmetric_signs(r_S, X_r),
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=f"te(tp) penalty {j} differs from R",
            )

    def test_ti_cr_basis_vs_r(self) -> None:
        """ti(x1, x2, k=5, bs='cr'): basis matches R element-wise (STRICT)."""
        smooth, r_result, data = self._setup_ti_cr()
        X_py = smooth.build_design_matrix(data)
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py,
            X_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="ti(cr) basis differs from R",
        )

    def test_ti_tp_basis_vs_r(self) -> None:
        """ti(x1, x2, k=5, bs='tp'): basis matches R element-wise (MODERATE).

        Sign normalization handles LAPACK eigenvector sign ambiguity
        compounded by constraint absorption.
        """
        smooth, r_result, data = self._setup_ti_tp()
        X_py = smooth.build_design_matrix(data)
        X_r = r_result["X"]

        np.testing.assert_allclose(
            normalize_column_signs(X_py),
            normalize_column_signs(X_r),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="ti(tp) basis differs from R",
        )

    def test_ti_cr_penalty_vs_r(self) -> None:
        """ti(cr) penalty matrices match R element-wise (STRICT)."""
        smooth, r_result, _data = self._setup_ti_cr()

        for j, (py_pen, r_S) in enumerate(
            zip(smooth.build_penalty_matrices(), r_result["S"], strict=True)
        ):
            np.testing.assert_allclose(
                py_pen.S,
                r_S,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"ti(cr) penalty {j} differs from R",
            )

    def test_ti_tp_penalty_vs_r(self) -> None:
        """ti(tp) penalty matrices match R element-wise (MODERATE)."""
        smooth, r_result, data = self._setup_ti_tp()
        X_py = smooth.build_design_matrix(data)
        X_r = r_result["X"]

        for j, (py_pen, r_S) in enumerate(
            zip(smooth.build_penalty_matrices(), r_result["S"], strict=True)
        ):
            np.testing.assert_allclose(
                normalize_symmetric_signs(py_pen.S, X_py),
                normalize_symmetric_signs(r_S, X_r),
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=f"ti(tp) penalty {j} differs from R",
            )


# ===========================================================================
# 7. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for tensor product smooths."""

    def test_3d_tensor(self) -> None:
        """3D tensor: te(x1, x2, x3, k=3, bs='cr') — 27 columns, 3 penalties."""
        spec = _make_spec(["x1", "x2", "x3"], bs="cr", k=3)
        smooth = TensorProductSmooth(spec)
        data = _make_3d_data()
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert X.shape == (200, 27)
        assert smooth.n_coefs == 27

        penalties = smooth.build_penalty_matrices()
        assert len(penalties) == 3

    def test_small_k(self) -> None:
        """k=3 for cr marginals (minimum viable)."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=3)
        smooth = TensorProductSmooth(spec)
        data = _make_2d_data()
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert X.shape == (200, 9)
        assert np.all(np.isfinite(X))

    def test_large_n(self) -> None:
        """n=1000 runs without memory issues."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        data = _make_2d_data(n=1000)
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert X.shape == (1000, 25)
        assert np.all(np.isfinite(X))

    def test_predict_different_n(self) -> None:
        """predict_matrix with different data size than training data."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(_make_2d_data(n=200))

        new_data = _make_2d_data(n=50, seed=99)
        X_new = smooth.predict_matrix(new_data)
        assert X_new.shape == (50, 25)
        assert np.all(np.isfinite(X_new))

    def test_ti_predict_different_n(self) -> None:
        """ti predict_matrix with different data size than training data."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(_make_2d_data(n=200))

        new_data = _make_2d_data(n=50, seed=99)
        X_new = smooth.predict_matrix(new_data)
        assert X_new.shape == (50, 16)
        assert np.all(np.isfinite(X_new))

    def test_setup_required(self) -> None:
        """build_design_matrix before setup raises RuntimeError."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.build_design_matrix(_make_2d_data())

    def test_ti_setup_required(self) -> None:
        """ti build_design_matrix before setup raises RuntimeError."""
        spec = _make_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.build_design_matrix(_make_2d_data())

    def test_3d_ti(self) -> None:
        """3D ti: ti(x1, x2, x3, k=3, bs='cr')."""
        spec = _make_spec(["x1", "x2", "x3"], bs="cr", k=3, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        data = _make_3d_data()
        smooth.setup(data)

        # cr k=3 -> 3 coefs, constrained -> 2 each, so 2*2*2=8
        X = smooth.build_design_matrix(data)
        assert X.shape == (200, 8)
        assert smooth.n_coefs == 8

        penalties = smooth.build_penalty_matrices()
        assert len(penalties) == 3


# ===========================================================================
# 8. Phase boundary guard (no JAX imports)
# ===========================================================================


class TestRegistry:
    """Tests for smooth class registry with tensor types."""

    def test_te_lookup(self) -> None:
        from pymgcv.smooths.registry import get_smooth_class

        assert get_smooth_class("te") is TensorProductSmooth

    def test_ti_lookup(self) -> None:
        from pymgcv.smooths.registry import get_smooth_class

        assert get_smooth_class("ti") is TensorInteractionSmooth


class TestNoJaxImport:
    """Verify pymgcv.smooths.tensor does not trigger JAX import."""

    def test_tensor_import_no_jax(self) -> None:
        """Importing pymgcv.smooths.tensor must not cause jax import."""
        modules_to_remove = [
            key
            for key in sys.modules
            if key == "jax" or key.startswith(("jax.", "pymgcv."))
        ]
        saved = {key: sys.modules.pop(key) for key in modules_to_remove}

        try:
            importlib.import_module("pymgcv.smooths.tensor")
            assert "jax" not in sys.modules, (
                "Importing pymgcv.smooths.tensor triggered a jax import. "
                "Phase 1 modules must not depend on JAX."
            )
        finally:
            for key in list(sys.modules):
                if key.startswith("pymgcv."):
                    sys.modules.pop(key, None)
            sys.modules.update(saved)
