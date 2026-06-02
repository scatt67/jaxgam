"""Tests for tensor product smooth basis and penalty construction.

Validates TensorProductSmooth (te) and TensorInteractionSmooth (ti)
from jaxgam.smooths.tensor:
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

import numpy as np
import pytest

from jaxgam.formula.terms import SmoothSpec
from jaxgam.smooths.tensor import (
    TensorInteractionSmooth,
    TensorProductSmooth,
    _row_tensor,
)
from tests.helpers import (
    _AssertCollector,
    check_that,
    make_smooth_spec,
    r_available,
)
from tests.tolerances import (
    MODERATE,
    STRICT,
    normalize_column_signs,
    normalize_symmetric_signs,
)

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
        from jaxgam.smooths.cubic import CubicRegressionSmooth

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


# ===========================================================================
# 3. TensorProductSmooth structural tests (STRICT)
# ===========================================================================


class TestTensorProductStructure:
    """Structural properties of TensorProductSmooth."""

    def test_te_shape_cr(self, smooth_2d_data) -> None:
        """te(x1, x2, k=5, bs='cr') produces (n, 25) basis."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(smooth_2d_data)

        X = smooth.build_design_matrix(smooth_2d_data)
        assert X.shape == (200, 25)

    def test_n_coefs(self, smooth_2d_data) -> None:
        """n_coefs = product of marginal n_coefs."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(smooth_2d_data)

        assert smooth.n_coefs == 5 * 5

    def test_null_space_dim_cr(self, smooth_2d_data) -> None:
        """null_space_dim for cr: 2*2 = 4."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(smooth_2d_data)

        assert smooth.null_space_dim == 2 * 2

    def test_penalty_rank(self, smooth_2d_data) -> None:
        """Penalty rank: rank(S_j) * product(d_i for i != j)."""
        k = 5
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=k)
        smooth = TensorProductSmooth(spec)
        smooth.setup(smooth_2d_data)

        penalties = smooth.build_penalty_matrices()
        # cr has rank k-2=3, so each tensor penalty has rank 3*5=15
        for p in penalties:
            assert p.rank == (k - 2) * k

    def test_predict_matches_design(self, smooth_2d_data) -> None:
        """predict_matrix(train_data) matches build_design_matrix(train_data)."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(smooth_2d_data)

        X_design = smooth.build_design_matrix(smooth_2d_data)
        X_predict = smooth.predict_matrix(smooth_2d_data)
        np.testing.assert_allclose(
            X_predict, X_design, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_default_k_resolves_to_5(self, smooth_2d_data) -> None:
        """Unspecified marginal k defaults to 5 per 1-D margin (R's 5^d).

        Regression for Finding 2: ``te(x1, x2)`` with no k must build a
        5x5 = 25-coef tensor (was 10x10 = 100). Explicit k is still honored.
        """
        # k=-1 means "unspecified" (the parser default), distinct from
        # make_smooth_spec's own k=10 default.
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=-1, smooth_type="te")
        smooth = TensorProductSmooth(spec)
        smooth.setup(smooth_2d_data)
        assert [m.n_coefs for m in smooth._marginals] == [5, 5]
        assert smooth.n_coefs == 25

        # Explicit k must still pass through unchanged.
        spec7 = make_smooth_spec(["x1", "x2"], bs="cr", k=7, smooth_type="te")
        smooth7 = TensorProductSmooth(spec7)
        smooth7.setup(smooth_2d_data)
        assert [m.n_coefs for m in smooth7._marginals] == [7, 7]
        assert smooth7.n_coefs == 49


# ===========================================================================
# 4. TensorInteractionSmooth structural tests (STRICT)
# ===========================================================================


class TestTensorInteractionStructure:
    """Structural properties of TensorInteractionSmooth."""

    def test_ti_shape_cr(self, smooth_2d_data) -> None:
        """ti(x1, x2, k=5, bs='cr') produces (n, 16) basis (4*4)."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(smooth_2d_data)

        X = smooth.build_design_matrix(smooth_2d_data)
        assert X.shape == (200, 16)

    def test_default_k_resolves_to_5(self, smooth_2d_data) -> None:
        """Unspecified ti() marginal k defaults to 5 per margin (Finding 2).

        Each cr margin loses one column to the sum-to-zero constraint, so the
        constrained marginals are 4 columns -> 4x4 = 16-coef interaction.
        """
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=-1, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(smooth_2d_data)
        assert [m.n_coefs for m in smooth._marginals] == [5, 5]
        assert smooth.n_coefs == 16

    def test_smaller_than_te(self, smooth_2d_data) -> None:
        """ti columns < te columns for same k."""
        spec_te = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type="te")
        te_smooth = TensorProductSmooth(spec_te)
        te_smooth.setup(smooth_2d_data)

        spec_ti = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        ti_smooth = TensorInteractionSmooth(spec_ti)
        ti_smooth.setup(smooth_2d_data)

        assert ti_smooth.n_coefs < te_smooth.n_coefs

    def test_subspace_of_te(self, smooth_2d_data) -> None:
        """Column space of ti is a subspace of te (verified via projection)."""
        spec_te = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type="te")
        te_smooth = TensorProductSmooth(spec_te)
        te_smooth.setup(smooth_2d_data)
        X_te = te_smooth.build_design_matrix(smooth_2d_data)

        spec_ti = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        ti_smooth = TensorInteractionSmooth(spec_ti)
        ti_smooth.setup(smooth_2d_data)
        X_ti = ti_smooth.build_design_matrix(smooth_2d_data)

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

    def test_null_space_dim(self, smooth_2d_data) -> None:
        """null_space_dim: product of (marginal_nsd - 1) for each marginal."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(smooth_2d_data)

        # cr has nsd=2, so constrained nsd = 2-1 = 1 per marginal
        # product = 1*1 = 1
        assert smooth.null_space_dim == 1

    def test_penalty_rank(self, smooth_2d_data) -> None:
        """ti penalty rank: constrained_rank * product(constrained_d_i for i != j)."""
        k = 5
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=k, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(smooth_2d_data)

        penalties = smooth.build_penalty_matrices()
        # cr k=5: rank=3, nsd=2. Constrained: dim=4, nsd=1, rank=3.
        # Each tensor penalty has rank 3*4=12
        constrained_dim = k - 1  # 4
        constrained_rank = k - 2  # 3 (cr rank unchanged by constraint)
        for p in penalties:
            assert p.rank == constrained_rank * constrained_dim

    def test_predict_matches_design(self, smooth_2d_data) -> None:
        """predict_matrix(train_data) matches build_design_matrix(train_data)."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(smooth_2d_data)

        X_design = smooth.build_design_matrix(smooth_2d_data)
        X_predict = smooth.predict_matrix(smooth_2d_data)
        np.testing.assert_allclose(
            X_predict, X_design, rtol=STRICT.rtol, atol=STRICT.atol
        )


# ===========================================================================
# 5. Shared tensor penalty structural tests (STRICT)
# ===========================================================================


class TestTensorPenaltyStructure:
    """Structural penalty checks shared by te and ti smooths."""

    @pytest.mark.parametrize(
        ("smooth_type", "smooth_cls"),
        [("te", TensorProductSmooth), ("ti", TensorInteractionSmooth)],
    )
    def test_penalty_symmetry(
        self, smooth_2d_data, smooth_type: str, smooth_cls
    ) -> None:
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type=smooth_type)
        smooth = smooth_cls(spec)
        smooth.setup(smooth_2d_data)

        for penalty in smooth.build_penalty_matrices():
            np.testing.assert_allclose(
                penalty.S, penalty.S.T, rtol=STRICT.rtol, atol=STRICT.atol
            )

    @pytest.mark.parametrize(
        ("smooth_type", "smooth_cls"),
        [("te", TensorProductSmooth), ("ti", TensorInteractionSmooth)],
    )
    def test_penalty_psd(self, smooth_2d_data, smooth_type: str, smooth_cls) -> None:
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type=smooth_type)
        smooth = smooth_cls(spec)
        smooth.setup(smooth_2d_data)

        for penalty in smooth.build_penalty_matrices():
            eigvals = np.linalg.eigvalsh(penalty.S)
            assert np.all(eigvals >= -STRICT.atol), (
                f"{smooth_type} penalty has negative eigenvalue: {np.min(eigvals):.2e}"
            )


# ===========================================================================
# 6. R comparison tests (MODERATE, skip if R unavailable)
# ===========================================================================


@pytest.mark.skipif(not r_available(), reason="R with mgcv not available")
class TestRComparison:
    """Compare tensor product construction against R's smoothCon().

    With SVD reparameterization implemented, all basis types now match
    R element-wise. Cubic marginals have noterp=True (skip SVD reparam),
    TPRS marginals get SVD reparameterized to match R exactly.
    """

    def _setup_te_cr(self) -> tuple:
        """Setup te(x1, x2, bs='cr', k=5) for R comparison."""
        import pandas as pd

        from tests.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        data_pd = pd.DataFrame({"x1": x1, "x2": x2})

        bridge = RBridge()
        r_result = bridge.smooth_construct("te(x1, x2, bs='cr', k=5)", data_pd)

        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup({"x1": x1, "x2": x2})

        return smooth, r_result, {"x1": x1, "x2": x2}

    def _setup_te_tp(self) -> tuple:
        """Setup te(x1, x2, bs='tp', k=5) for R comparison."""
        import pandas as pd

        from tests.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        data_pd = pd.DataFrame({"x1": x1, "x2": x2})

        bridge = RBridge()
        r_result = bridge.smooth_construct("te(x1, x2, bs='tp', k=5)", data_pd)

        spec = make_smooth_spec(["x1", "x2"], bs="tp", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup({"x1": x1, "x2": x2})

        return smooth, r_result, {"x1": x1, "x2": x2}

    def _setup_te_tp_m1(self) -> tuple:
        """Setup te(x1, x2, bs='tp', k=5, m=1) for R comparison (Finding H3)."""
        import pandas as pd

        from tests.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        data_pd = pd.DataFrame({"x1": x1, "x2": x2})

        bridge = RBridge()
        r_result = bridge.smooth_construct("te(x1, x2, bs='tp', k=5, m=1)", data_pd)

        spec = make_smooth_spec(["x1", "x2"], bs="tp", k=5, smooth_type="te", m=1)
        smooth = TensorProductSmooth(spec)
        smooth.setup({"x1": x1, "x2": x2})

        return smooth, r_result, {"x1": x1, "x2": x2}

    def test_te_tp_marginal_m_vs_r(self) -> None:
        """te(tp, m=1) basis + penalty match R's smoothCon element-wise (H3).

        Before the fix the marginal order was dropped, so this fit produced the
        m=2 basis and diverged from R's m=1 reference.
        """
        smooth, r_result, data = self._setup_te_tp_m1()
        X_py = smooth.build_design_matrix(data)
        X_r = r_result["X"]

        collector = _AssertCollector()
        collector.check(
            "basis_vs_r",
            lambda: np.testing.assert_allclose(
                normalize_column_signs(X_py),
                normalize_column_signs(X_r),
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg="te(tp, m=1) basis differs from R",
            ),
        )
        for j, (py_pen, r_S) in enumerate(
            zip(smooth.build_penalty_matrices(), r_result["S"], strict=True)
        ):
            collector.check(
                f"penalty_{j}_vs_r",
                lambda py_pen=py_pen, r_S=r_S, j=j: np.testing.assert_allclose(
                    normalize_symmetric_signs(py_pen.S, X_py),
                    normalize_symmetric_signs(r_S, X_r),
                    rtol=MODERATE.rtol,
                    atol=MODERATE.atol,
                    err_msg=f"te(tp, m=1) penalty {j} differs from R",
                ),
            )
        collector.raise_if_any("te(tp, m=1) vs R")

    def _setup_ti_cr(self) -> tuple:
        """Setup ti(x1, x2, bs='cr', k=5) for R comparison."""
        import pandas as pd

        from tests.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        data_pd = pd.DataFrame({"x1": x1, "x2": x2})

        bridge = RBridge()
        r_result = bridge.smooth_construct("ti(x1, x2, bs='cr', k=5)", data_pd)

        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup({"x1": x1, "x2": x2})

        return smooth, r_result, {"x1": x1, "x2": x2}

    def _setup_ti_tp(self) -> tuple:
        """Setup ti(x1, x2, bs='tp', k=5) for R comparison."""
        import pandas as pd

        from tests.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        data_pd = pd.DataFrame({"x1": x1, "x2": x2})

        bridge = RBridge()
        r_result = bridge.smooth_construct("ti(x1, x2, bs='tp', k=5)", data_pd)

        spec = make_smooth_spec(["x1", "x2"], bs="tp", k=5, smooth_type="ti")
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

    def test_3d_tensor(self, smooth_3d_data) -> None:
        """3D tensor: te(x1, x2, x3, k=3, bs='cr') — 27 columns, 3 penalties."""
        spec = make_smooth_spec(["x1", "x2", "x3"], bs="cr", k=3)
        smooth = TensorProductSmooth(spec)
        smooth.setup(smooth_3d_data)

        X = smooth.build_design_matrix(smooth_3d_data)
        assert X.shape == (200, 27)
        assert smooth.n_coefs == 27

        penalties = smooth.build_penalty_matrices()
        assert len(penalties) == 3

    def test_small_k(self, smooth_2d_data) -> None:
        """k=3 for cr marginals (minimum viable)."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=3)
        smooth = TensorProductSmooth(spec)
        smooth.setup(smooth_2d_data)

        X = smooth.build_design_matrix(smooth_2d_data)
        assert X.shape == (200, 9)
        assert np.all(np.isfinite(X))

    def test_marginal_order_m_threaded_to_tp_margins(self) -> None:
        """te(...,bs='tp',m=1) must differ from m=2 and thread m into marginals.

        Finding H3: ``_create_marginals`` dropped ``spec.extra_args``, so every
        marginal used its default order regardless of the requested ``m`` — m=1
        silently returned the m=2 answer. R replicates a scalar m to all margins
        (R/smooth.r:443) and the tp margin constructor honours it; cr margins
        ignore m in both R and jaxgam, and m-absent keeps the default order.
        """
        rng = np.random.default_rng(42)
        data = {"x1": rng.uniform(0, 1, 200), "x2": rng.uniform(0, 1, 200)}

        def _build_tp(m):
            sm = TensorProductSmooth(
                make_smooth_spec(["x1", "x2"], bs="tp", k=5, smooth_type="te", m=m)
            )
            sm.setup(data)
            return sm, sm.build_design_matrix(data)

        sm1, X1 = _build_tp(1)
        sm2, X2 = _build_tp(2)
        _smd, Xd = _build_tp(None)  # m absent -> default order (==m=2 for tp d=1)

        collector = _AssertCollector()
        collector.check(
            "m1_marginal_order",
            lambda: check_that(
                [mm._m for mm in sm1._marginals] == [1, 1],
                f"te(m=1) marginals _m={[mm._m for mm in sm1._marginals]}, want [1,1]",
            ),
        )
        collector.check(
            "basis_differs",
            lambda: check_that(
                not np.allclose(X1, X2),
                "te(bs='tp',m=1) basis identical to m=2 — marginal order ignored",
            ),
        )
        collector.check(
            "null_space_dim",
            lambda: check_that(
                sm1.null_space_dim == 1 and sm2.null_space_dim == 4,
                f"nsd m=1->{sm1.null_space_dim} (want 1), m=2->{sm2.null_space_dim}"
                " (want 4)",
            ),
        )
        collector.check(
            "default_preserved",
            lambda: check_that(
                np.allclose(Xd, X2),
                "m-absent default must equal m=2 for tp d=1 (preserve prior default)",
            ),
        )
        collector.raise_if_any("te(tp) marginal-order threading (H3)")

    @pytest.mark.parametrize("smooth_2d_data", [1000], indirect=True)
    def test_large_n(self, smooth_2d_data) -> None:
        """n=1000 runs without memory issues."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(smooth_2d_data)

        X = smooth.build_design_matrix(smooth_2d_data)
        assert X.shape == (1000, 25)
        assert np.all(np.isfinite(X))

    def test_predict_different_n(self, smooth_2d_data, pred_smooth_2d_data) -> None:
        """predict_matrix with different data size than training data."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        smooth.setup(smooth_2d_data)

        X_new = smooth.predict_matrix(pred_smooth_2d_data)
        assert X_new.shape == (50, 25)
        assert np.all(np.isfinite(X_new))

    def test_ti_predict_different_n(self, smooth_2d_data, pred_smooth_2d_data) -> None:
        """ti predict_matrix with different data size than training data."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(smooth_2d_data)

        X_new = smooth.predict_matrix(pred_smooth_2d_data)
        assert X_new.shape == (50, 16)
        assert np.all(np.isfinite(X_new))

    def test_setup_required(self, smooth_2d_data) -> None:
        """build_design_matrix before setup raises RuntimeError."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5)
        smooth = TensorProductSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.build_design_matrix(smooth_2d_data)

    def test_ti_setup_required(self, smooth_2d_data) -> None:
        """ti build_design_matrix before setup raises RuntimeError."""
        spec = make_smooth_spec(["x1", "x2"], bs="cr", k=5, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.build_design_matrix(smooth_2d_data)

    def test_3d_ti(self, smooth_3d_data) -> None:
        """3D ti: ti(x1, x2, x3, k=3, bs='cr')."""
        spec = make_smooth_spec(["x1", "x2", "x3"], bs="cr", k=3, smooth_type="ti")
        smooth = TensorInteractionSmooth(spec)
        smooth.setup(smooth_3d_data)

        # cr k=3 -> 3 coefs, constrained -> 2 each, so 2*2*2=8
        X = smooth.build_design_matrix(smooth_3d_data)
        assert X.shape == (200, 8)
        assert smooth.n_coefs == 8

        penalties = smooth.build_penalty_matrices()
        assert len(penalties) == 3
