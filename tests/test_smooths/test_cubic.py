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

from jaxgam.smooths.cubic import (
    CubicRegressionSmooth,
    CubicShrinkageSmooth,
    CyclicCubicSmooth,
)
from tests.helpers import (
    SEED,
    _AssertCollector,
    check_that,
    make_smooth_spec,
    r_available,
)
from tests.tolerances import MODERATE, STRICT

_place_knots = CubicRegressionSmooth._place_knots
_CUBIC_SMOOTH_CLASSES = {
    "cr": CubicRegressionSmooth,
    "cs": CubicShrinkageSmooth,
    "cc": CyclicCubicSmooth,
}


def _setup_cubic_smooth(bs: str, data, k: int = 10):
    spec = make_smooth_spec(["x"], bs=bs, k=k)
    smooth = _CUBIC_SMOOTH_CLASSES[bs](spec)
    smooth.setup(data)
    return smooth


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

    @pytest.mark.parametrize("bs", ["cr", "cs", "cc"])
    def test_S_symmetric_psd(self, smooth_1d_data, bs: str) -> None:
        """Cubic penalties are symmetric PSD across basis variants."""
        smooth = _setup_cubic_smooth(bs, smooth_1d_data)

        S = smooth.build_penalty_matrices()[0].S
        np.testing.assert_allclose(S, S.T, rtol=STRICT.rtol, atol=STRICT.atol)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -STRICT.atol), (
            f"{bs} S has negative eigenvalue: {np.min(eigvals):.2e}"
        )

    @pytest.mark.parametrize(
        ("bs", "expected_rank", "expected_null_space"),
        [("cr", 8, 2), ("cs", 10, 0), ("cc", 8, 1)],
    )
    def test_S_rank(
        self,
        smooth_1d_data,
        bs: str,
        expected_rank: int,
        expected_null_space: int,
    ) -> None:
        """Cubic penalty ranks match the basis variant."""
        k = 10
        smooth = _setup_cubic_smooth(bs, smooth_1d_data, k=k)

        assert smooth.rank == expected_rank
        penalty = smooth.build_penalty_matrices()[0]
        assert penalty.rank == expected_rank
        assert penalty.null_space_dim == expected_null_space

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

    def test_below_minimum_k_warns_and_bumps(self) -> None:
        """Cubic smooths warn-and-bump below-minimum k (match mgcv); never raise.

        Finding M5: mgcv requires k>=4 for cc (endpoints identified) and k>=3
        for cr/cs, and below the minimum it warns and INCREASES k rather than
        erroring (R/smooth.r:1458-1462 cr, 1608-1611 cc). Before the fix
        ``s(x,bs='cc',k=3)`` silently built a degenerate 2-column basis, and
        k=2 raised ``ValueError`` instead of bumping.
        """
        rng = np.random.default_rng(SEED)
        data = {"x": rng.uniform(0.0, 1.0, 200)}
        collector = _AssertCollector()

        # cc k=3 (the M5 bug): must warn and bump to 4 -> k-1 = 3 columns.
        def _cc_k3() -> None:
            smooth = CyclicCubicSmooth(make_smooth_spec(["x"], bs="cc", k=3))
            with pytest.warns(UserWarning, match="basis dimension"):
                smooth.setup(data)
            check_that(smooth.n_coefs == 3, f"cc k=3 n_coefs={smooth.n_coefs}, want 3")
            check_that(smooth._X.shape == (200, 3), f"cc k=3 X={smooth._X.shape}")
            check_that(smooth.rank == 2, f"cc k=3 rank={smooth.rank}, want 2")
            check_that(
                smooth.null_space_dim == 1, f"cc k=3 nsd={smooth.null_space_dim}"
            )

        collector.check("cc_k3_warns_and_bumps", _cc_k3)

        # cc k=3 must produce the SAME basis as an explicit cc k=4 fit.
        def _cc_k3_equals_k4() -> None:
            s3 = CyclicCubicSmooth(make_smooth_spec(["x"], bs="cc", k=3))
            with pytest.warns(UserWarning, match="basis dimension"):
                s3.setup(data)
            s4 = CyclicCubicSmooth(make_smooth_spec(["x"], bs="cc", k=4))
            s4.setup(data)
            np.testing.assert_allclose(s3._X, s4._X, rtol=STRICT.rtol, atol=STRICT.atol)

        collector.check("cc_k3_matches_explicit_k4", _cc_k3_equals_k4)

        # cr/cs k=2 must warn and bump to 3 (not raise).
        def _cr_k2() -> None:
            smooth = CubicRegressionSmooth(make_smooth_spec(["x"], bs="cr", k=2))
            with pytest.warns(UserWarning, match="basis dimension"):
                smooth.setup(data)
            check_that(smooth.n_coefs == 3, f"cr k=2 n_coefs={smooth.n_coefs}, want 3")

        collector.check("cr_k2_warns_and_bumps", _cr_k2)

        def _cs_k2() -> None:
            smooth = CubicShrinkageSmooth(make_smooth_spec(["x"], bs="cs", k=2))
            with pytest.warns(UserWarning, match="basis dimension"):
                smooth.setup(data)
            check_that(smooth.n_coefs == 3, f"cs k=2 n_coefs={smooth.n_coefs}, want 3")

        collector.check("cs_k2_warns_and_bumps", _cs_k2)

        # Valid k is unaffected: cc k=5 builds 4 columns with no warning.
        def _cc_k5_no_warn() -> None:
            import warnings as _w

            smooth = CyclicCubicSmooth(make_smooth_spec(["x"], bs="cc", k=5))
            with _w.catch_warnings():
                _w.simplefilter("error")
                smooth.setup(data)
            check_that(smooth.n_coefs == 4, f"cc k=5 n_coefs={smooth.n_coefs}, want 4")

        collector.check("cc_k5_no_warning", _cc_k5_no_warn)

        collector.raise_if_any("cubic below-minimum k warn-and-bump (M5)")

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

    @pytest.mark.parametrize("bs", ["cr", "cs", "cc"])
    @pytest.mark.parametrize("k", [5, 10, 15, 20])
    def test_various_k(self, smooth_1d_data, bs: str, k: int) -> None:
        """Cubic smooths work for various k values."""
        smooth = _setup_cubic_smooth(bs, smooth_1d_data, k=k)

        X = smooth.build_design_matrix(smooth_1d_data)
        expected_coefs = k - 1 if bs == "cc" else k
        expected_rank = k if bs == "cs" else k - 2
        expected_null_space = {"cr": 2, "cs": 0, "cc": 1}[bs]

        assert X.shape == (200, expected_coefs)
        assert smooth.n_coefs == expected_coefs
        assert smooth.rank == expected_rank
        assert smooth.null_space_dim == expected_null_space

        S = smooth.build_penalty_matrices()[0].S
        assert S.shape == (expected_coefs, expected_coefs)
