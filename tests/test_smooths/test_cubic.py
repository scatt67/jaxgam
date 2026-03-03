"""Tests for cubic regression spline basis and penalty construction.

Validates CubicRegressionSmooth (cr), CubicShrinkageSmooth (cs),
and CyclicCubicSmooth (cc) from jaxgam.smooths.cubic:
- Knot placement tests (STRICT)
- Penalty construction unit tests (STRICT)
- Basis matrix structural tests (STRICT)
- R comparison tests (MODERATE, skip if R unavailable)
- Cyclic-specific tests
- Shrinkage tests
- Edge cases
- Phase boundary guard (no JAX imports)
- Parameterized tests

Design doc reference: docs/design.md Section 5.3
R source reference: R/smooth.r smooth.construct.cr.smooth.spec()
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxgam.penalties.penalty import Penalty
from jaxgam.smooths.cubic import (
    CubicRegressionSmooth,
    CubicShrinkageSmooth,
    CyclicCubicSmooth,
)
from tests.helpers import make_smooth_spec, r_available
from tests.tolerances import MODERATE, STRICT

_place_knots = CubicRegressionSmooth._place_knots


# ===========================================================================
# 1. Knot placement tests (STRICT)
# ===========================================================================


class TestKnotPlacement:
    """Tests for _place_knots()."""

    def test_uniform_data(self) -> None:
        """place_knots on uniform data gives equally-spaced knots."""
        x = np.linspace(0, 1, 100)
        knots = _place_knots(x, 10)
        expected = np.linspace(0, 1, 10)
        np.testing.assert_allclose(knots, expected, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_data_with_ties(self) -> None:
        """place_knots handles data with ties (rank-based placement)."""
        x = np.concatenate([np.zeros(50), np.ones(50), np.full(50, 0.5)])
        knots = _place_knots(x, 3)
        # 3 unique values: [0, 0.5, 1], indices [0, 1, 2]
        # linspace(0, 2, 3) = [0, 1, 2] → knots = [0, 0.5, 1]
        np.testing.assert_allclose(
            knots, [0.0, 0.5, 1.0], rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_k_equals_3_minimum(self) -> None:
        """place_knots works with k=3 (minimum)."""
        x = np.linspace(0, 1, 50)
        knots = _place_knots(x, 3)
        assert len(knots) == 3
        np.testing.assert_allclose(knots[0], 0.0, atol=STRICT.atol)
        np.testing.assert_allclose(knots[-1], 1.0, atol=STRICT.atol)

    def test_k_equals_50(self) -> None:
        """place_knots works with large k."""
        x = np.linspace(0, 10, 200)
        knots = _place_knots(x, 50)
        assert len(knots) == 50
        assert np.all(np.diff(knots) >= 0)

    def test_sorted_output(self) -> None:
        """place_knots returns sorted knots for random data."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        knots = _place_knots(x, 10)
        assert np.all(np.diff(knots) >= 0)

    def test_boundary_knots_match_data_range(self) -> None:
        """First and last knots equal data min and max."""
        x = np.linspace(2.0, 7.0, 100)
        knots = _place_knots(x, 10)
        np.testing.assert_allclose(knots[0], 2.0, atol=STRICT.atol)
        np.testing.assert_allclose(knots[-1], 7.0, atol=STRICT.atol)


# ===========================================================================
# 2. Penalty construction unit tests (STRICT)
# ===========================================================================


class TestPenaltyConstruction:
    """Tests for penalty matrix construction."""

    def test_cr_S_symmetric_psd(self, smooth_1d_data) -> None:
        """cr penalty S is symmetric PSD."""
        spec = make_smooth_spec(["x"], k=10)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup(smooth_1d_data)

        S = smooth.build_penalty_matrices()[0].S
        np.testing.assert_allclose(S, S.T, rtol=STRICT.rtol, atol=STRICT.atol)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -STRICT.atol), (
            f"cr S has negative eigenvalue: {np.min(eigvals):.2e}"
        )

    def test_cs_S_symmetric_psd(self, smooth_1d_data) -> None:
        """cs penalty S is symmetric PSD."""
        spec = make_smooth_spec(["x"], bs="cs", k=10)
        smooth = CubicShrinkageSmooth(spec)
        smooth.setup(smooth_1d_data)

        S = smooth.build_penalty_matrices()[0].S
        np.testing.assert_allclose(S, S.T, rtol=STRICT.rtol, atol=STRICT.atol)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -STRICT.atol)

    def test_cc_S_symmetric_psd(self, smooth_1d_data) -> None:
        """cc penalty S is symmetric PSD."""
        spec = make_smooth_spec(["x"], bs="cc", k=10)
        smooth = CyclicCubicSmooth(spec)
        smooth.setup(smooth_1d_data)

        S = smooth.build_penalty_matrices()[0].S
        np.testing.assert_allclose(S, S.T, rtol=STRICT.rtol, atol=STRICT.atol)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -STRICT.atol), (
            f"cc S has negative eigenvalue: {np.min(eigvals):.2e}"
        )

    def test_cr_S_rank(self, smooth_1d_data) -> None:
        """cr penalty rank = k-2."""
        k = 10
        spec = make_smooth_spec(["x"], k=k)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup(smooth_1d_data)

        assert smooth.rank == k - 2
        penalty = smooth.build_penalty_matrices()[0]
        assert penalty.rank == k - 2
        assert penalty.null_space_dim == 2

    def test_cs_S_rank(self, smooth_1d_data) -> None:
        """cs penalty rank = k (full rank)."""
        k = 10
        spec = make_smooth_spec(["x"], bs="cs", k=k)
        smooth = CubicShrinkageSmooth(spec)
        smooth.setup(smooth_1d_data)

        assert smooth.rank == k
        penalty = smooth.build_penalty_matrices()[0]
        assert penalty.rank == k
        assert penalty.null_space_dim == 0

    def test_cc_S_rank(self, smooth_1d_data) -> None:
        """cc penalty rank = k-2, null_space_dim = 1."""
        k = 10
        spec = make_smooth_spec(["x"], bs="cc", k=k)
        smooth = CyclicCubicSmooth(spec)
        smooth.setup(smooth_1d_data)

        assert smooth.rank == k - 2
        penalty = smooth.build_penalty_matrices()[0]
        assert penalty.rank == k - 2
        assert penalty.null_space_dim == 1

    def test_cr_null_space_contains_linear(self, smooth_1d_data) -> None:
        """cr null space is aligned with constant + linear functions."""
        k = 10
        spec = make_smooth_spec(["x"], k=k)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup(smooth_1d_data)

        S = smooth.build_penalty_matrices()[0].S
        # S @ ones ≈ 0 (constant in null space)
        np.testing.assert_allclose(S @ np.ones(k), np.zeros(k), atol=STRICT.atol)
        # S @ knots ≈ 0 (linear in null space)
        np.testing.assert_allclose(S @ smooth._knots, np.zeros(k), atol=STRICT.atol)

    def test_cc_null_space_contains_constant(self, smooth_1d_data) -> None:
        """cc null space contains constant function."""
        k = 10
        spec = make_smooth_spec(["x"], bs="cc", k=k)
        smooth = CyclicCubicSmooth(spec)
        smooth.setup(smooth_1d_data)

        S = smooth.build_penalty_matrices()[0].S
        np.testing.assert_allclose(
            S @ np.ones(k - 1), np.zeros(k - 1), atol=STRICT.atol
        )


# ===========================================================================
# 3. Basis matrix structural tests (STRICT)
# ===========================================================================


class TestBasisMatrixStructure:
    """Structural properties of the basis matrix."""

    def test_cr_X_shape(self, smooth_1d_data) -> None:
        """cr X shape = (n, k)."""
        n, k = 200, 10
        spec = make_smooth_spec(["x"], k=k)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup(smooth_1d_data)

        X = smooth.build_design_matrix(smooth_1d_data)
        assert X.shape == (n, k)

    def test_cs_X_shape(self, smooth_1d_data) -> None:
        """cs X shape = (n, k)."""
        n, k = 200, 10
        spec = make_smooth_spec(["x"], bs="cs", k=k)
        smooth = CubicShrinkageSmooth(spec)
        smooth.setup(smooth_1d_data)

        X = smooth.build_design_matrix(smooth_1d_data)
        assert X.shape == (n, k)

    def test_cc_X_shape(self, smooth_1d_data) -> None:
        """cc X shape = (n, k-1)."""
        n, k = 200, 10
        spec = make_smooth_spec(["x"], bs="cc", k=k)
        smooth = CyclicCubicSmooth(spec)
        smooth.setup(smooth_1d_data)

        X = smooth.build_design_matrix(smooth_1d_data)
        assert X.shape == (n, k - 1)

    @pytest.mark.parametrize("smooth_1d_data", [100], indirect=True)
    def test_predict_equals_design_matrix_cr(self, smooth_1d_data) -> None:
        """predict_matrix == build_design_matrix for cr."""
        spec = make_smooth_spec(["x"], k=10)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup(smooth_1d_data)

        X_design = smooth.build_design_matrix(smooth_1d_data)
        X_predict = smooth.predict_matrix(smooth_1d_data)
        np.testing.assert_allclose(
            X_predict, X_design, rtol=STRICT.rtol, atol=STRICT.atol
        )

    @pytest.mark.parametrize("smooth_1d_data", [100], indirect=True)
    def test_predict_equals_design_matrix_cc(self, smooth_1d_data) -> None:
        """predict_matrix == build_design_matrix for cc."""
        spec = make_smooth_spec(["x"], bs="cc", k=10)
        smooth = CyclicCubicSmooth(spec)
        smooth.setup(smooth_1d_data)

        X_design = smooth.build_design_matrix(smooth_1d_data)
        X_predict = smooth.predict_matrix(smooth_1d_data)
        np.testing.assert_allclose(
            X_predict, X_design, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_penalty_returns_list_of_penalty(self, smooth_1d_data) -> None:
        """build_penalty_matrices returns list[Penalty]."""
        spec = make_smooth_spec(["x"], k=10)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup(smooth_1d_data)

        penalties = smooth.build_penalty_matrices()
        assert isinstance(penalties, list)
        assert len(penalties) == 1
        assert isinstance(penalties[0], Penalty)

    def test_n_coefs_cr(self, smooth_1d_data) -> None:
        """n_coefs = k for cr."""
        spec = make_smooth_spec(["x"], k=10)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup(smooth_1d_data)
        assert smooth.n_coefs == 10

    def test_n_coefs_cs(self, smooth_1d_data) -> None:
        """n_coefs = k for cs."""
        spec = make_smooth_spec(["x"], bs="cs", k=10)
        smooth = CubicShrinkageSmooth(spec)
        smooth.setup(smooth_1d_data)
        assert smooth.n_coefs == 10

    def test_n_coefs_cc(self, smooth_1d_data) -> None:
        """n_coefs = k-1 for cc."""
        spec = make_smooth_spec(["x"], bs="cc", k=10)
        smooth = CyclicCubicSmooth(spec)
        smooth.setup(smooth_1d_data)
        assert smooth.n_coefs == 9


# ===========================================================================
# 4. R comparison tests (MODERATE, skip if R unavailable)
# ===========================================================================


@pytest.mark.skipif(not r_available(), reason="R with mgcv not available")
class TestRComparison:
    """Compare cubic spline construction against R's smoothCon().

    Unlike TPRS, cubic splines involve no eigendecomposition and thus
    have no LAPACK sign ambiguity. Basis matrices X and penalty matrices
    S are fully deterministic and match R element-wise at machine
    precision (~1e-15). All primary tests use STRICT tolerance.
    """

    def _setup_cr(self) -> tuple:
        """Shared cr setup for R comparison."""
        import pandas as pd

        from tests.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        data = pd.DataFrame({"x": x})

        bridge = RBridge()
        r_result = bridge.smooth_construct("s(x, bs='cr', k=10)", data)

        spec = make_smooth_spec(["x"], k=10)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup({"x": x})
        return smooth, r_result, x

    def _setup_cc(self) -> tuple:
        """Shared cc setup for R comparison."""
        import pandas as pd

        from tests.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        data = pd.DataFrame({"x": x})

        bridge = RBridge()
        r_result = bridge.smooth_construct("s(x, bs='cc', k=10)", data)

        spec = make_smooth_spec(["x"], bs="cc", k=10)
        smooth = CyclicCubicSmooth(spec)
        smooth.setup({"x": x})
        return smooth, r_result, x

    def _setup_cs(self) -> tuple:
        """Shared cs setup for R comparison."""
        import pandas as pd

        from tests.r_bridge import RBridge

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        data = pd.DataFrame({"x": x})

        bridge = RBridge()
        r_result = bridge.smooth_construct("s(x, bs='cs', k=10)", data)

        spec = make_smooth_spec(["x"], bs="cs", k=10)
        smooth = CubicShrinkageSmooth(spec)
        smooth.setup({"x": x})
        return smooth, r_result, x

    # --- cr element-wise tests (STRICT) ---

    def test_cr_X_values_vs_r(self) -> None:
        """cr basis matrix X matches R element-wise (STRICT).

        Cubic splines are fully deterministic — no eigendecomposition,
        no sign ambiguity. X should match at machine precision.
        """
        smooth, r_result, x = self._setup_cr()
        X_py = smooth.build_design_matrix({"x": x})
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py,
            X_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="cr X values differ from R",
        )

    def test_cr_S_values_vs_r(self) -> None:
        """cr penalty matrix S matches R element-wise (STRICT)."""
        smooth, r_result, _x = self._setup_cr()
        S_py = smooth.build_penalty_matrices()[0].S
        S_r = r_result["S"][0]

        np.testing.assert_allclose(
            S_py,
            S_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="cr S values differ from R",
        )

    def test_cr_rank_vs_r(self) -> None:
        """cr rank and null_space_dim match R."""
        smooth, r_result, _x = self._setup_cr()
        assert smooth.rank == r_result["rank"]
        assert smooth.null_space_dim == r_result["null_space_dim"]

    def test_cr_knots_vs_r(self) -> None:
        """cr knot locations match R (STRICT)."""
        import pandas as pd
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri

        smooth, _r_result, x = self._setup_cr()

        # Extract knots via rpy2 (smoothCon doesn't export xp for cr)
        with ro.conversion.localconverter(
            ro.default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            r_df = ro.conversion.py2rpy(pd.DataFrame({"x": x}))
        ro.globalenv["dat_input"] = r_df
        r_knots = np.array(
            ro.r(
                """
            library(mgcv)
            dat <- as.data.frame(dat_input)
            spec <- s(x, bs="cr", k=10)
            spec <- eval(spec)
            sm <- smooth.construct(spec, dat, knots=NULL)
            sm$xp
            """
            )
        )
        np.testing.assert_allclose(
            smooth._knots,
            r_knots,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="cr knot locations do not match R",
        )

    # --- cc element-wise tests (STRICT) ---

    def test_cc_X_values_vs_r(self) -> None:
        """cc basis matrix X matches R element-wise (STRICT)."""
        smooth, r_result, x = self._setup_cc()
        X_py = smooth.build_design_matrix({"x": x})
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py,
            X_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="cc X values differ from R",
        )

    def test_cc_S_values_vs_r(self) -> None:
        """cc penalty matrix S matches R element-wise (STRICT)."""
        smooth, r_result, _x = self._setup_cc()
        S_py = smooth.build_penalty_matrices()[0].S
        S_r = r_result["S"][0]

        np.testing.assert_allclose(
            S_py,
            S_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="cc S values differ from R",
        )

    def test_cc_rank_vs_r(self) -> None:
        """cc rank and null_space_dim match R."""
        smooth, r_result, _x = self._setup_cc()
        assert smooth.rank == r_result["rank"]
        assert smooth.null_space_dim == r_result["null_space_dim"]

    def test_cc_knots_vs_r(self) -> None:
        """cc knot locations match R (STRICT)."""
        import pandas as pd
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri

        smooth, _r_result, x = self._setup_cc()

        with ro.conversion.localconverter(
            ro.default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            r_df = ro.conversion.py2rpy(pd.DataFrame({"x": x}))
        ro.globalenv["dat_input"] = r_df
        r_knots = np.array(
            ro.r(
                """
            library(mgcv)
            dat <- as.data.frame(dat_input)
            spec <- s(x, bs="cc", k=10)
            spec <- eval(spec)
            sm <- smooth.construct(spec, dat, knots=NULL)
            sm$xp
            """
            )
        )
        np.testing.assert_allclose(
            smooth._knots,
            r_knots,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="cc knot locations do not match R",
        )

    # --- cs element-wise tests (STRICT) ---

    def test_cs_X_values_vs_r(self) -> None:
        """cs basis matrix X matches R element-wise (STRICT).

        cs and cr share the same basis matrix X (shrinkage only
        modifies S), so this also cross-validates the cr X test.
        """
        smooth, r_result, x = self._setup_cs()
        X_py = smooth.build_design_matrix({"x": x})
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py,
            X_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="cs X values differ from R",
        )

    def test_cs_S_eigenvalues_vs_r(self) -> None:
        """cs penalty S eigenvalues match R.

        cs S is reconstructed from eigenvectors after replacing zero
        eigenvalues, so element-wise comparison may differ due to
        LAPACK eigenvector ordering. Eigenvalue comparison is robust.
        """
        smooth, r_result, _x = self._setup_cs()
        S_py = smooth.build_penalty_matrices()[0].S
        S_r = r_result["S"][0]

        eigvals_py = np.linalg.eigvalsh(S_py)
        eigvals_r = np.linalg.eigvalsh(S_r)

        np.testing.assert_allclose(
            eigvals_py,
            eigvals_r,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="cs S eigenvalues differ from R",
        )

    def test_cs_rank_vs_r(self) -> None:
        """cs has full rank penalty matching R."""
        smooth, r_result, _x = self._setup_cs()
        assert smooth.rank == r_result["rank"]
        assert smooth.null_space_dim == 0


# ===========================================================================
# 5. Cyclic-specific tests
# ===========================================================================


class TestCyclicSpecific:
    """Tests specific to cyclic cubic splines."""

    def test_cc_periodicity(self) -> None:
        """cc basis is periodic: predict at lower_bound ≈ predict at upper_bound."""
        spec = make_smooth_spec(["x"], bs="cc", k=10)
        smooth = CyclicCubicSmooth(spec)
        x = np.linspace(0, 1, 200)
        smooth.setup({"x": x})

        X_low = smooth.predict_matrix({"x": np.array([0.0])})
        X_high = smooth.predict_matrix({"x": np.array([1.0])})
        np.testing.assert_allclose(
            X_low,
            X_high,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="cc basis not periodic at boundaries",
        )

    def test_cc_penalty_penalizes_non_constant(self, smooth_1d_data) -> None:
        """cc penalty penalizes all non-constant functions."""
        k = 10
        spec = make_smooth_spec(["x"], bs="cc", k=k)
        smooth = CyclicCubicSmooth(spec)
        smooth.setup(smooth_1d_data)

        S = smooth.build_penalty_matrices()[0].S
        eigvals = np.sort(np.linalg.eigvalsh(S))
        # First eigenvalue should be ~0 (constant null space)
        assert np.abs(eigvals[0]) < STRICT.atol
        # All others should be positive
        assert np.all(eigvals[1:] > STRICT.atol)


# ===========================================================================
# 6. Shrinkage tests
# ===========================================================================


class TestShrinkage:
    """Tests for CubicShrinkageSmooth."""

    def test_cs_basis_identical_to_cr(self, smooth_1d_data) -> None:
        """cs and cr produce the same basis matrix X."""
        spec_cr = make_smooth_spec(["x"], bs="cr", k=10)
        smooth_cr = CubicRegressionSmooth(spec_cr)
        smooth_cr.setup(smooth_1d_data)
        X_cr = smooth_cr.build_design_matrix(smooth_1d_data)

        spec_cs = make_smooth_spec(["x"], bs="cs", k=10)
        smooth_cs = CubicShrinkageSmooth(spec_cs)
        smooth_cs.setup(smooth_1d_data)
        X_cs = smooth_cs.build_design_matrix(smooth_1d_data)

        np.testing.assert_allclose(
            X_cs,
            X_cr,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="cs and cr should have the same basis matrix",
        )

    def test_cs_penalty_full_rank(self, smooth_1d_data) -> None:
        """cs penalty has full rank."""
        spec = make_smooth_spec(["x"], bs="cs", k=10)
        smooth = CubicShrinkageSmooth(spec)
        smooth.setup(smooth_1d_data)

        penalty = smooth.build_penalty_matrices()[0]
        assert penalty.rank == 10
        assert penalty.null_space_dim == 0

    def test_cs_penalty_psd(self, smooth_1d_data) -> None:
        """cs penalty is strictly positive definite."""
        spec = make_smooth_spec(["x"], bs="cs", k=10)
        smooth = CubicShrinkageSmooth(spec)
        smooth.setup(smooth_1d_data)

        S = smooth.build_penalty_matrices()[0].S
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals > 0), "cs penalty should be strictly positive definite"


# ===========================================================================
# 7. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for cubic splines."""

    def test_duplicate_data_points(self) -> None:
        """Duplicate data points handled correctly."""
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 20)
        spec = make_smooth_spec(["x"], k=5)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup({"x": x})

        X = smooth.build_design_matrix({"x": x})
        assert X.shape == (100, 5)
        assert np.all(np.isfinite(X))

    def test_very_small_n(self) -> None:
        """Small n with appropriate k works."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 10)
        spec = make_smooth_spec(["x"], k=5)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup({"x": x})

        X = smooth.build_design_matrix({"x": x})
        assert X.shape == (10, 5)

    def test_k_exceeds_n_unique_raises(self) -> None:
        """k > n_unique raises ValueError."""
        x = np.array([0.1, 0.2, 0.3] * 30)
        spec = make_smooth_spec(["x"], k=10)
        smooth = CubicRegressionSmooth(spec)
        with pytest.raises(ValueError, match="exceeds"):
            smooth.setup({"x": x})

    def test_k_less_than_3_raises(self, smooth_1d_data) -> None:
        """k < 3 raises ValueError."""
        spec = make_smooth_spec(["x"], k=2)
        smooth = CubicRegressionSmooth(spec)
        with pytest.raises(ValueError, match="at least 3"):
            smooth.setup(smooth_1d_data)

    def test_setup_required_for_design_matrix(self) -> None:
        """build_design_matrix before setup raises RuntimeError."""
        spec = make_smooth_spec(["x"], k=10)
        smooth = CubicRegressionSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.build_design_matrix({"x": np.zeros(10)})

    def test_setup_required_for_penalty(self) -> None:
        """build_penalty_matrices before setup raises RuntimeError."""
        spec = make_smooth_spec(["x"], k=10)
        smooth = CubicRegressionSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.build_penalty_matrices()

    def test_setup_required_for_predict(self) -> None:
        """predict_matrix before setup raises RuntimeError."""
        spec = make_smooth_spec(["x"], k=10)
        smooth = CubicRegressionSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.predict_matrix({"x": np.zeros(10)})

    def test_multivariate_raises(self) -> None:
        """Multi-variable spec raises ValueError."""
        spec = make_smooth_spec(["x1", "x2"], k=10)
        smooth = CubicRegressionSmooth(spec)
        with pytest.raises(ValueError, match="univariate"):
            smooth.setup({"x1": np.zeros(10), "x2": np.zeros(10)})

    @pytest.mark.parametrize("smooth_1d_data", [100], indirect=True)
    def test_predict_new_data_different_n(
        self, smooth_1d_data, pred_smooth_1d_data
    ) -> None:
        """predict_matrix works with different n than training data."""
        spec = make_smooth_spec(["x"], k=10)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup(smooth_1d_data)

        X_new = smooth.predict_matrix(pred_smooth_1d_data)
        assert X_new.shape == (50, 10)
        assert np.all(np.isfinite(X_new))


# ===========================================================================
# 8. Parameterized tests
# ===========================================================================


class TestParameterized:
    """Parameterized tests for various k values."""

    @pytest.mark.parametrize("k", [5, 10, 15, 20])
    def test_cr_various_k(self, k: int, smooth_1d_data) -> None:
        """cr works for various k values."""
        spec = make_smooth_spec(["x"], k=k)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup(smooth_1d_data)

        X = smooth.build_design_matrix(smooth_1d_data)
        assert X.shape == (200, k)
        assert smooth.n_coefs == k
        assert smooth.rank == k - 2
        assert smooth.null_space_dim == 2

        penalties = smooth.build_penalty_matrices()
        S = penalties[0].S
        assert S.shape == (k, k)

    @pytest.mark.parametrize("k", [5, 10, 15, 20])
    def test_cs_various_k(self, k: int, smooth_1d_data) -> None:
        """cs works for various k values."""
        spec = make_smooth_spec(["x"], bs="cs", k=k)
        smooth = CubicShrinkageSmooth(spec)
        smooth.setup(smooth_1d_data)

        X = smooth.build_design_matrix(smooth_1d_data)
        assert X.shape == (200, k)
        assert smooth.n_coefs == k
        assert smooth.rank == k
        assert smooth.null_space_dim == 0

    @pytest.mark.parametrize("k", [5, 10, 15, 20])
    def test_cc_various_k(self, k: int, smooth_1d_data) -> None:
        """cc works for various k values."""
        spec = make_smooth_spec(["x"], bs="cc", k=k)
        smooth = CyclicCubicSmooth(spec)
        smooth.setup(smooth_1d_data)

        X = smooth.build_design_matrix(smooth_1d_data)
        assert X.shape == (200, k - 1)
        assert smooth.n_coefs == k - 1
        assert smooth.rank == k - 2
        assert smooth.null_space_dim == 1

    def test_default_k(self, smooth_1d_data) -> None:
        """Default k (k=-1) uses 10."""
        spec = make_smooth_spec(["x"], k=-1)
        smooth = CubicRegressionSmooth(spec)
        smooth.setup(smooth_1d_data)
        assert smooth.n_coefs == 10


# ===========================================================================
# 10. Registry tests
# ===========================================================================


class TestRegistry:
    """Tests for smooth class registry with cubic types."""

    def test_cr_lookup(self) -> None:
        from jaxgam.smooths.registry import get_smooth_class

        assert get_smooth_class("cr") is CubicRegressionSmooth

    def test_cs_lookup(self) -> None:
        from jaxgam.smooths.registry import get_smooth_class

        assert get_smooth_class("cs") is CubicShrinkageSmooth

    def test_cc_lookup(self) -> None:
        from jaxgam.smooths.registry import get_smooth_class

        assert get_smooth_class("cc") is CyclicCubicSmooth
