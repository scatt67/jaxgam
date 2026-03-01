"""Tests for identifiability constraints and CoefficientMap.

Validates the constraint pipeline from jaxgam.smooths.constraints:
- CoefficientMap.apply_sum_to_zero: centering constraint absorption (STRICT)
- CoefficientMap.fix_dependence: linear dependence detection (STRICT)
- CoefficientMap.gam_side: inter-term identifiability (STRICT)
- CoefficientMap: coefficient mapping and roundtrip (STRICT)
- Phase boundary guard (no JAX imports)

Design doc reference: docs/design.md Section 5.10
R source reference: R/mgcv.r gam.side(), fixDependence()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.formula.terms import SmoothSpec
from jaxgam.smooths.by_variable import (
    FactorBySmooth,
    NumericBySmooth,
)
from jaxgam.smooths.constraints import CoefficientMap, TermBlock
from jaxgam.smooths.tensor import TensorProductSmooth
from jaxgam.smooths.tprs import TPRSSmooth
from tests.tolerances import (
    MODERATE,
    STRICT,
    normalize_column_signs,
    normalize_symmetric_signs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42
N = 200


def _make_spec(
    variables: list[str],
    bs: str = "tp",
    k: int = 10,
    by: str | None = None,
    smooth_type: str = "s",
    **extra_args: object,
) -> SmoothSpec:
    """Create a SmoothSpec for testing."""
    return SmoothSpec(
        variables=variables,
        bs=bs,
        k=k,
        by=by,
        smooth_type=smooth_type,
        extra_args=dict(extra_args),
    )


def _make_1d_data(n: int = N, seed: int = SEED) -> dict[str, np.ndarray]:
    """Generate simple 1D test data."""
    rng = np.random.default_rng(seed)
    return {"x": rng.uniform(0, 1, n)}


def _make_2d_data(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Generate simple 2D test data with optional factor column."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    fac = pd.Categorical(rng.choice(["a", "b", "c"], n))
    z = rng.uniform(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "fac": fac, "z": z})


def _setup_tprs(var_name: str, data_values: np.ndarray, k: int = 10) -> TPRSSmooth:
    """Create and setup a TPRS smooth with the given variable name."""
    spec = _make_spec([var_name], k=k)
    sm = TPRSSmooth(spec)
    sm.setup({var_name: data_values})
    return sm


def _setup_te(
    var_names: list[str],
    data_dict: dict[str, np.ndarray],
    k: int = 5,
) -> TensorProductSmooth:
    """Create and setup a TensorProductSmooth.

    Uses 'cr' as the marginal basis type (tensor products use marginal bases).
    """
    spec = _make_spec(var_names, bs="cr", k=k, smooth_type="te")
    sm = TensorProductSmooth(spec)
    sm.setup(data_dict)
    return sm


def _get_X_S(smooth, data: dict[str, np.ndarray]):
    """Get design matrix and penalty matrix list from a smooth."""
    X = smooth.build_design_matrix(data)
    S_list = [p.S for p in smooth.build_penalty_matrices()]
    return X, S_list


# ===========================================================================
# Centering constraint tests (STRICT)
# ===========================================================================


class TestApplySumToZero:
    """Tests for sum-to-zero centering constraint absorption."""

    def test_reduces_columns_by_one(self):
        """Centering reduces columns from k to k-1."""
        data = _make_1d_data()
        sm = _setup_tprs("x", data["x"])
        X, S_list = _get_X_S(sm, data)

        X_c, S_c_list, Z = CoefficientMap.apply_sum_to_zero(X, S_list)

        assert X_c.shape == (N, 9), f"Expected (200, 9), got {X_c.shape}"
        assert Z.shape == (10, 9), f"Expected (10, 9), got {Z.shape}"
        for S_c in S_c_list:
            assert S_c.shape == (9, 9)

    def test_constraint_satisfied(self):
        """After centering, column sums of X_c should be near zero."""
        data = _make_1d_data()
        sm = _setup_tprs("x", data["x"])
        X, S_list = _get_X_S(sm, data)

        X_c, _, _ = CoefficientMap.apply_sum_to_zero(X, S_list)

        col_sums = np.sum(X_c, axis=0)
        np.testing.assert_allclose(
            col_sums,
            0.0,
            atol=STRICT.atol,
            err_msg="Column sums should be near zero after centering",
        )

    def test_prediction_roundtrip(self):
        """X @ Z @ beta_c == X_c @ beta_c for any beta_c."""
        data = _make_1d_data()
        sm = _setup_tprs("x", data["x"])
        X, S_list = _get_X_S(sm, data)

        X_c, _, Z = CoefficientMap.apply_sum_to_zero(X, S_list)

        rng = np.random.default_rng(123)
        beta_c = rng.standard_normal(X_c.shape[1])

        pred_via_raw = X @ Z @ beta_c
        pred_via_constrained = X_c @ beta_c

        np.testing.assert_allclose(
            pred_via_raw,
            pred_via_constrained,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_penalty_psd(self):
        """Constrained penalty matrices must remain PSD."""
        data = _make_1d_data()
        sm = _setup_tprs("x", data["x"])
        X, S_list = _get_X_S(sm, data)

        _, S_c_list, _ = CoefficientMap.apply_sum_to_zero(X, S_list)

        for S_c in S_c_list:
            eigvals = np.linalg.eigvalsh(S_c)
            assert np.all(eigvals >= -STRICT.atol), (
                f"Constrained penalty has negative eigenvalue: {np.min(eigvals)}"
            )

    def test_penalty_symmetric(self):
        """Constrained penalty matrices must be symmetric."""
        data = _make_1d_data()
        sm = _setup_tprs("x", data["x"])
        X, S_list = _get_X_S(sm, data)

        _, S_c_list, _ = CoefficientMap.apply_sum_to_zero(X, S_list)

        for S_c in S_c_list:
            np.testing.assert_allclose(
                S_c,
                S_c.T,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            )

    def test_works_with_cr_basis(self):
        """Centering works with cubic regression splines."""
        from jaxgam.smooths.cubic import CubicRegressionSmooth

        data = _make_1d_data()
        spec = _make_spec(["x"], bs="cr", k=10)
        sm = CubicRegressionSmooth(spec)
        sm.setup(data)
        X, S_list = _get_X_S(sm, data)

        X_c, _, _ = CoefficientMap.apply_sum_to_zero(X, S_list)

        assert X_c.shape[1] == X.shape[1] - 1
        col_sums = np.sum(X_c, axis=0)
        np.testing.assert_allclose(col_sums, 0.0, atol=STRICT.atol)

    def test_works_with_cc_basis(self):
        """Centering works with cyclic cubic splines."""
        from jaxgam.smooths.cubic import CyclicCubicSmooth

        data = _make_1d_data()
        spec = _make_spec(["x"], bs="cc", k=10)
        sm = CyclicCubicSmooth(spec)
        sm.setup(data)
        X, S_list = _get_X_S(sm, data)

        X_c, _, _ = CoefficientMap.apply_sum_to_zero(X, S_list)

        assert X_c.shape[1] == X.shape[1] - 1
        col_sums = np.sum(X_c, axis=0)
        np.testing.assert_allclose(col_sums, 0.0, atol=STRICT.atol)

    def test_works_with_multiple_penalties(self):
        """Centering works with tensor product (multiple penalties)."""
        df = _make_2d_data()
        data_dict = {"x1": df["x1"].values, "x2": df["x2"].values}
        sm = _setup_te(["x1", "x2"], data_dict, k=5)
        X, S_list = _get_X_S(sm, data_dict)

        X_c, S_c_list, _Z = CoefficientMap.apply_sum_to_zero(X, S_list)

        assert X_c.shape[1] == X.shape[1] - 1
        assert len(S_c_list) == len(S_list)
        for S_c in S_c_list:
            assert S_c.shape[0] == X_c.shape[1]
            eigvals = np.linalg.eigvalsh(S_c)
            assert np.all(eigvals >= -STRICT.atol)

    def test_Z_orthogonal(self):
        """Z columns must be orthonormal."""
        data = _make_1d_data()
        sm = _setup_tprs("x", data["x"])
        X, S_list = _get_X_S(sm, data)

        _, _, Z = CoefficientMap.apply_sum_to_zero(X, S_list)

        np.testing.assert_allclose(
            Z.T @ Z,
            np.eye(Z.shape[1]),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


# ===========================================================================
# Factor-by centering constraint tests
# ===========================================================================


class TestApplySumToZeroFactorBy:
    """Tests for factor-by centering constraint."""

    def test_reduces_columns_per_level(self):
        """Each level block loses 1 column."""
        df = _make_2d_data()
        sm = _setup_tprs("x1", df["x1"].values)

        spec = _make_spec(["x1"], k=10, by="fac")
        fbs = FactorBySmooth(sm, spec, levels=["a", "b", "c"], by_variable="fac")
        X = fbs.build_design_matrix(df)
        S_list = [p.S for p in fbs.build_penalty_matrices()]

        X_c, _S_c_list, Z = CoefficientMap.apply_sum_to_zero_factor_by(
            X, S_list, n_levels=3, k_per_level=10
        )

        # 3 levels * (10 - 1) = 27
        assert X_c.shape[1] == 27
        assert Z.shape == (30, 27)

    def test_block_diagonal_Z(self):
        """Z should be block-diagonal."""
        df = _make_2d_data()
        sm = _setup_tprs("x1", df["x1"].values)

        spec = _make_spec(["x1"], k=10, by="fac")
        fbs = FactorBySmooth(sm, spec, levels=["a", "b", "c"], by_variable="fac")
        X = fbs.build_design_matrix(df)
        S_list = [p.S for p in fbs.build_penalty_matrices()]

        _, _, Z = CoefficientMap.apply_sum_to_zero_factor_by(
            X, S_list, n_levels=3, k_per_level=10
        )

        # Check block diagonal: off-diagonal blocks should be zero
        for lev_i in range(3):
            for lev_j in range(3):
                if lev_i == lev_j:
                    continue
                row_start = lev_i * 10
                row_end = row_start + 10
                col_start = lev_j * 9
                col_end = col_start + 9
                block = Z[row_start:row_end, col_start:col_end]
                np.testing.assert_allclose(block, 0.0, atol=STRICT.atol)


# ===========================================================================
# Linear dependence detection tests (STRICT)
# ===========================================================================


class TestFixDependence:
    """Tests for linear dependence detection."""

    def test_detects_fully_dependent_columns(self):
        """When X2 columns are linear combinations of X1."""
        rng = np.random.default_rng(SEED)
        X1 = rng.standard_normal((100, 5))
        A = rng.standard_normal((5, 3))
        X2 = X1 @ A

        ind = CoefficientMap.fix_dependence(X1, X2)

        assert ind is not None
        assert len(ind) == 3

    def test_returns_none_when_independent(self):
        """When X2 is fully independent of X1."""
        rng = np.random.default_rng(SEED)
        X1 = rng.standard_normal((100, 5))
        X2 = rng.standard_normal((100, 3))

        ind = CoefficientMap.fix_dependence(X1, X2)

        assert ind is None

    def test_detects_partial_dependence(self):
        """When some columns of X2 are dependent, others not."""
        rng = np.random.default_rng(SEED)
        X1 = rng.standard_normal((100, 5))
        X2_dep = X1 @ rng.standard_normal((5, 1))
        X2_indep = rng.standard_normal((100, 2))
        X2 = np.column_stack([X2_dep, X2_indep])

        ind = CoefficientMap.fix_dependence(X1, X2)

        assert ind is not None
        assert len(ind) == 1

    def test_intercept_column_detection(self):
        """Detects when X2 contains an intercept already in X1."""
        n = 100
        X1 = np.ones((n, 1))
        rng = np.random.default_rng(SEED)
        X2 = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])

        ind = CoefficientMap.fix_dependence(X1, X2)

        assert ind is not None
        assert len(ind) == 1

    def test_tolerance_handling(self):
        """Tight tolerance should detect fewer dependencies."""
        rng = np.random.default_rng(SEED)
        X1 = rng.standard_normal((100, 5))
        A = rng.standard_normal((5, 3))
        noise = rng.standard_normal((100, 3)) * 1e-6
        X2 = X1 @ A + noise

        ind_tight = CoefficientMap.fix_dependence(X1, X2, tol=1e-14)
        ind_loose = CoefficientMap.fix_dependence(X1, X2, tol=1e-3)

        assert ind_tight is None
        assert ind_loose is not None


# ===========================================================================
# Inter-term identifiability tests (STRICT)
# ===========================================================================


class TestGamSide:
    """Tests for inter-term identifiability constraint detection."""

    def _make_two_smooth_setup(self):
        """Set up s(x1) + s(x2) with centered X blocks."""
        df = _make_2d_data()
        sm1 = _setup_tprs("x1", df["x1"].values)
        sm2 = _setup_tprs("x2", df["x2"].values)

        X1, S1 = _get_X_S(sm1, {"x1": df["x1"].values})
        X2, S2 = _get_X_S(sm2, {"x2": df["x2"].values})

        X1_c, S1_c, _ = CoefficientMap.apply_sum_to_zero(X1, S1)
        X2_c, S2_c, _ = CoefficientMap.apply_sum_to_zero(X2, S2)

        X_param = np.ones((N, 1))
        return sm1, sm2, X1_c, X2_c, S1_c, S2_c, X_param, df

    def _make_te_setup(self):
        """Set up s(x1) + s(x2) + te(x1,x2) with centered X blocks."""
        df = _make_2d_data()
        data_dict = {"x1": df["x1"].values, "x2": df["x2"].values}

        sm1 = _setup_tprs("x1", df["x1"].values)
        sm2 = _setup_tprs("x2", df["x2"].values)
        sm_te = _setup_te(["x1", "x2"], data_dict, k=5)

        X1, S1 = _get_X_S(sm1, {"x1": df["x1"].values})
        X2, S2 = _get_X_S(sm2, {"x2": df["x2"].values})
        X_te, S_te = _get_X_S(sm_te, data_dict)

        X1_c, S1_c, _ = CoefficientMap.apply_sum_to_zero(X1, S1)
        X2_c, S2_c, _ = CoefficientMap.apply_sum_to_zero(X2, S2)
        X_te_c, S_te_c, _ = CoefficientMap.apply_sum_to_zero(X_te, S_te)

        X_param = np.ones((N, 1))
        return (sm1, sm2, sm_te, X1_c, X2_c, X_te_c, S1_c, S2_c, S_te_c, X_param, df)

    def test_no_nesting_no_deletion(self):
        """s(x1) + s(x2) -> no columns deleted."""
        sm1, sm2, X1_c, X2_c, S1_c, S2_c, X_param, _ = self._make_two_smooth_setup()

        del_indices = CoefficientMap.gam_side(
            [sm1, sm2],
            [X1_c, X2_c],
            [S1_c, S2_c],
            X_param,
        )

        assert del_indices[0] is None
        assert del_indices[1] is None

    def test_te_nesting_deletes_columns(self):
        """s(x1) + s(x2) + te(x1,x2) -> te columns deleted."""
        (sm1, sm2, sm_te, X1_c, X2_c, X_te_c, S1_c, S2_c, S_te_c, X_param, _) = (
            self._make_te_setup()
        )

        del_indices = CoefficientMap.gam_side(
            [sm1, sm2, sm_te],
            [X1_c, X2_c, X_te_c],
            [S1_c, S2_c, S_te_c],
            X_param,
        )

        # s(x1) and s(x2) should have no deletions
        assert del_indices[0] is None
        assert del_indices[1] is None
        # te(x1,x2) should have columns deleted
        assert del_indices[2] is not None
        assert len(del_indices[2]) > 0

    def test_factor_by_not_nested_with_main(self):
        """s(x1) + s(x1, by=fac): different variable names -> no nesting."""
        df = _make_2d_data()
        sm1 = _setup_tprs("x1", df["x1"].values)

        spec_by = _make_spec(["x1"], k=10, by="fac")
        sm_base = _setup_tprs("x1", df["x1"].values)
        fbs = FactorBySmooth(
            sm_base,
            spec_by,
            levels=["a", "b", "c"],
            by_variable="fac",
        )

        X1, S1 = _get_X_S(sm1, {"x1": df["x1"].values})
        X_by = fbs.build_design_matrix(df)
        S_by = [p.S for p in fbs.build_penalty_matrices()]

        X1_c, S1_c, _ = CoefficientMap.apply_sum_to_zero(X1, S1)
        X_by_c, S_by_c, _ = CoefficientMap.apply_sum_to_zero_factor_by(
            X_by, S_by, n_levels=3, k_per_level=10
        )

        X_param = np.ones((N, 1))

        del_indices = CoefficientMap.gam_side(
            [sm1, fbs],
            [X1_c, X_by_c],
            [S1_c, S_by_c],
            X_param,
        )

        # Variable names differ: x1 vs x1fac, so no nesting detected
        assert del_indices[0] is None
        assert del_indices[1] is None

    def test_processing_order_low_to_high(self):
        """Smooths are processed low->high dimension (1D before 2D)."""
        (sm1, sm2, sm_te, X1_c, X2_c, X_te_c, S1_c, S2_c, S_te_c, X_param, _) = (
            self._make_te_setup()
        )

        # Put te first in the list — should still process 1D first
        del_indices = CoefficientMap.gam_side(
            [sm_te, sm1, sm2],
            [X_te_c.copy(), X1_c.copy(), X2_c.copy()],
            [
                [S.copy() for S in S_te_c],
                [S.copy() for S in S1_c],
                [S.copy() for S in S2_c],
            ],
            X_param,
        )

        # te should have deletions (it's 2D, processed after 1D terms)
        assert del_indices[0] is not None

    def test_deleted_columns_from_X_and_S(self):
        """Deleted columns are actually removed from X and S blocks."""
        (sm1, sm2, sm_te, X1_c, X2_c, X_te_c, S1_c, S2_c, S_te_c, X_param, _) = (
            self._make_te_setup()
        )

        orig_te_cols = X_te_c.shape[1]

        # gam_side mutates X_blocks and S_blocks
        X_blocks = [X1_c, X2_c, X_te_c]
        S_blocks = [S1_c, S2_c, S_te_c]

        del_indices = CoefficientMap.gam_side(
            [sm1, sm2, sm_te],
            X_blocks,
            S_blocks,
            X_param,
        )

        if del_indices[2] is not None:
            n_deleted = len(del_indices[2])
            assert X_blocks[2].shape[1] == orig_te_cols - n_deleted
            for S_c in S_blocks[2]:
                assert S_c.shape == (orig_te_cols - n_deleted, orig_te_cols - n_deleted)

    def test_empty_smooths_list(self):
        """Empty list returns empty list."""
        result = CoefficientMap.gam_side([], [], [], None)
        assert result == []


# ===========================================================================
# CoefficientMap tests (STRICT)
# ===========================================================================


class TestCoefficientMap:
    """Tests for the CoefficientMap dataclass."""

    def _make_simple_coef_map(self):
        """Create a simple CoefficientMap for testing."""
        term1 = TermBlock(
            label="intercept",
            col_start=0,
            n_coefs=1,
            n_coefs_raw=1,
            term_type="parametric",
        )
        Q, _ = np.linalg.qr(np.ones((10, 1)), mode="complete")
        Z = Q[:, 1:]

        term2 = TermBlock(
            label="s(x1)",
            col_start=1,
            n_coefs=9,
            n_coefs_raw=10,
            term_type="smooth",
            Z_centering=Z,
        )
        return CoefficientMap(
            terms=(term1, term2),
            total_coefs=10,
            total_coefs_raw=11,
            has_intercept=True,
        )

    def test_get_term(self):
        """get_term returns correct term."""
        cm = self._make_simple_coef_map()
        t = cm.get_term("s(x1)")
        assert t.label == "s(x1)"
        assert t.n_coefs == 9

    def test_get_term_missing(self):
        """get_term raises KeyError for missing label."""
        cm = self._make_simple_coef_map()
        with pytest.raises(KeyError, match="No term"):
            cm.get_term("s(x_missing)")

    def test_term_slice(self):
        """term_slice returns correct slice."""
        cm = self._make_simple_coef_map()
        s = cm.term_slice("s(x1)")
        assert s == slice(1, 10)

    def test_total_coefs(self):
        """total_coefs reflects constrained space."""
        cm = self._make_simple_coef_map()
        assert cm.total_coefs == 10

    def test_constrained_to_full_roundtrip(self):
        """constrained_to_full produces correct-length output."""
        cm = self._make_simple_coef_map()
        rng = np.random.default_rng(SEED)
        beta_c = rng.standard_normal(cm.total_coefs)

        beta_raw = cm.constrained_to_full(beta_c)
        assert len(beta_raw) == cm.total_coefs_raw

    def test_prediction_equivalence(self):
        """X_c @ beta_c == X_raw @ constrained_to_full(beta_c)."""
        data = _make_1d_data()
        sm = _setup_tprs("x", data["x"])
        X_raw, S_list = _get_X_S(sm, data)

        X_c, _, Z = CoefficientMap.apply_sum_to_zero(X_raw, S_list)

        term = TermBlock(
            label="s(x)",
            col_start=0,
            n_coefs=X_c.shape[1],
            n_coefs_raw=X_raw.shape[1],
            term_type="smooth",
            Z_centering=Z,
        )
        cm = CoefficientMap(
            terms=(term,),
            total_coefs=X_c.shape[1],
            total_coefs_raw=X_raw.shape[1],
            has_intercept=False,
        )

        rng = np.random.default_rng(SEED)
        beta_c = rng.standard_normal(cm.total_coefs)

        pred_constrained = X_c @ beta_c
        beta_raw = cm.constrained_to_full(beta_c)
        pred_raw = X_raw @ beta_raw

        np.testing.assert_allclose(
            pred_raw,
            pred_constrained,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_transform_S_preserves_psd(self):
        """transform_S on a PSD matrix must produce PSD result."""
        data = _make_1d_data()
        sm = _setup_tprs("x", data["x"])
        X_raw, S_list = _get_X_S(sm, data)

        _, _, Z = CoefficientMap.apply_sum_to_zero(X_raw, S_list)

        term = TermBlock(
            label="s(x)",
            col_start=0,
            n_coefs=9,
            n_coefs_raw=10,
            term_type="smooth",
            Z_centering=Z,
        )
        cm = CoefficientMap(
            terms=(term,),
            total_coefs=9,
            total_coefs_raw=10,
            has_intercept=False,
        )

        S_transformed = cm.transform_S(S_list[0], "s(x)")
        eigvals = np.linalg.eigvalsh(S_transformed)
        assert np.all(eigvals >= -STRICT.atol)

    def test_transform_X_consistent_with_manual(self):
        """transform_X matches manual apply_sum_to_zero."""
        data = _make_1d_data()
        sm = _setup_tprs("x", data["x"])
        X_raw, S_list = _get_X_S(sm, data)

        X_c, _, Z = CoefficientMap.apply_sum_to_zero(X_raw, S_list)

        term = TermBlock(
            label="s(x)",
            col_start=0,
            n_coefs=9,
            n_coefs_raw=10,
            term_type="smooth",
            Z_centering=Z,
        )
        cm = CoefficientMap(
            terms=(term,),
            total_coefs=9,
            total_coefs_raw=10,
            has_intercept=False,
        )

        X_transformed = cm.transform_X(X_raw, "s(x)")
        np.testing.assert_allclose(
            X_transformed,
            X_c,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_with_del_index(self):
        """CoefficientMap with del_index handles column deletion."""
        rng = np.random.default_rng(SEED)
        n_raw = 10
        n_del = 2
        del_idx = (3, 7)
        n_final = n_raw - n_del

        term = TermBlock(
            label="s(x)",
            col_start=0,
            n_coefs=n_final,
            n_coefs_raw=n_raw,
            term_type="smooth",
            del_index=del_idx,
        )
        cm = CoefficientMap(
            terms=(term,),
            total_coefs=n_final,
            total_coefs_raw=n_raw,
            has_intercept=False,
        )

        beta_c = rng.standard_normal(n_final)
        beta_raw = cm.constrained_to_full(beta_c)

        assert len(beta_raw) == n_raw
        # Deleted positions should be zero
        assert beta_raw[3] == 0.0
        assert beta_raw[7] == 0.0


# ===========================================================================
# CoefficientMap.build integration tests
# ===========================================================================


class TestBuildCoefficientMap:
    """Integration tests for the full constraint pipeline."""

    def test_simple_two_smooth_model(self):
        """s(x1) + s(x2): centering applied, no gam_side deletion."""
        df = _make_2d_data()

        sm1 = _setup_tprs("x1", df["x1"].values)
        sm2 = _setup_tprs("x2", df["x2"].values)

        X1, S1 = _get_X_S(sm1, {"x1": df["x1"].values})
        X2, S2 = _get_X_S(sm2, {"x2": df["x2"].values})

        X_param = np.ones((N, 1))

        coef_map, _X_blocks, _S_blocks = CoefficientMap.build(
            smooths=[sm1, sm2],
            X_smooth_blocks=[X1, X2],
            S_smooth_blocks=[S1, S2],
            has_intercept=True,
            n_parametric=1,
            X_parametric=X_param,
        )

        # 1 intercept + 9 + 9 = 19
        assert coef_map.total_coefs == 19
        assert coef_map.total_coefs_raw == 21  # 1 + 10 + 10
        assert coef_map.has_intercept
        assert len(coef_map.terms) == 3  # parametric + 2 smooths

    def test_te_model_with_nesting(self):
        """s(x1) + s(x2) + te(x1,x2): te gets side-constrained."""
        df = _make_2d_data()
        data_dict = {"x1": df["x1"].values, "x2": df["x2"].values}

        sm1 = _setup_tprs("x1", df["x1"].values)
        sm2 = _setup_tprs("x2", df["x2"].values)
        sm_te = _setup_te(["x1", "x2"], data_dict, k=5)

        X1, S1 = _get_X_S(sm1, {"x1": df["x1"].values})
        X2, S2 = _get_X_S(sm2, {"x2": df["x2"].values})
        X_te, S_te = _get_X_S(sm_te, data_dict)

        X_param = np.ones((N, 1))

        coef_map, _X_blocks, _S_blocks = CoefficientMap.build(
            smooths=[sm1, sm2, sm_te],
            X_smooth_blocks=[X1, X2, X_te],
            S_smooth_blocks=[S1, S2, S_te],
            has_intercept=True,
            n_parametric=1,
            X_parametric=X_param,
        )

        # te term should have del_index set
        te_term = coef_map.get_term("te(x1,x2)")
        assert len(te_term.del_index) > 0, "te should have columns deleted"
        assert te_term.n_coefs < te_term.n_coefs_raw

    def test_numeric_by_no_centering(self):
        """s(x1, by=z): no centering constraint applied."""
        df = _make_2d_data()
        sm_base = _setup_tprs("x1", df["x1"].values)

        spec = _make_spec(["x1"], k=10, by="z")
        nbs = NumericBySmooth(sm_base, spec, by_variable="z")

        X_nbs = nbs.build_design_matrix(df)
        S_list = [p.S for p in nbs.build_penalty_matrices()]

        coef_map, _X_blocks, _S_blocks = CoefficientMap.build(
            smooths=[nbs],
            X_smooth_blocks=[X_nbs],
            S_smooth_blocks=[S_list],
            has_intercept=True,
            n_parametric=1,
            apply_side=False,
        )

        nbs_term = coef_map.get_term(nbs.label)
        assert nbs_term.n_coefs == 10  # no column removed
        assert nbs_term.Z_centering is None


# ===========================================================================
# R comparison tests (MODERATE tolerance, live RBridge)
# ===========================================================================


class TestRComparison:
    """Compare constraint pipeline against R mgcv reference output.

    Uses RBridge to call R's smoothCon() and gam() live, matching the
    pattern in test_tprs.py, test_cubic.py, etc.
    """

    def _get_bridge(self):
        """Get RBridge, skip test if R/mgcv not available."""
        from tests.r_bridge import RBridge

        if not RBridge.available():
            pytest.skip("R with mgcv not available")
        return RBridge()

    def _get_smoothcon_raw_and_absorbed(self, bridge):
        """Call R's smoothCon with absorb.cons=FALSE and TRUE."""
        rng = np.random.default_rng(SEED)
        x = rng.uniform(0, 1, N)
        data = pd.DataFrame({"x": x})

        raw = bridge.smooth_construct("s(x, bs='tp', k=10)", data, absorb_cons=False)
        absorbed = bridge.smooth_construct(
            "s(x, bs='tp', k=10)", data, absorb_cons=True
        )
        return raw, absorbed

    def test_smoothcon_centering_X(self):
        """apply_sum_to_zero(X_raw_R) matches R smoothCon(absorb.cons=TRUE) X.

        Uses R's own raw X as input, so this tests only the constraint
        absorption, independent of basis construction.
        """
        bridge = self._get_bridge()
        raw, absorbed = self._get_smoothcon_raw_and_absorbed(bridge)

        X_c_py, _, _ = CoefficientMap.apply_sum_to_zero(raw["X"], raw["S"])

        # QR sign ambiguity: normalize column signs before comparing
        np.testing.assert_allclose(
            normalize_column_signs(X_c_py),
            normalize_column_signs(absorbed["X"]),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=(
                "apply_sum_to_zero X does not match R smoothCon(absorb.cons=TRUE)"
            ),
        )

    def test_smoothcon_centering_S(self):
        """apply_sum_to_zero(S_raw_R) matches R smoothCon(absorb.cons=TRUE) S."""
        bridge = self._get_bridge()
        raw, absorbed = self._get_smoothcon_raw_and_absorbed(bridge)

        X_c_py, S_c_py, _ = CoefficientMap.apply_sum_to_zero(raw["X"], raw["S"])

        # Normalize signs using the design matrix for consistent convention
        np.testing.assert_allclose(
            normalize_symmetric_signs(S_c_py[0], X_c_py),
            normalize_symmetric_signs(absorbed["S"][0], absorbed["X"]),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=(
                "apply_sum_to_zero S does not match R smoothCon(absorb.cons=TRUE)"
            ),
        )

    def test_smoothcon_centering_shape(self):
        """Constrained dimensions match R: raw (n,10) -> absorbed (n,9)."""
        bridge = self._get_bridge()
        raw, absorbed = self._get_smoothcon_raw_and_absorbed(bridge)

        X_c_py, _, _ = CoefficientMap.apply_sum_to_zero(raw["X"], raw["S"])

        assert raw["X"].shape[1] == 10
        assert absorbed["X"].shape[1] == 9
        assert X_c_py.shape[1] == 9

    def test_gamside_te_column_counts(self, r_bridge):
        """gam_side te model: per-smooth column counts match R.

        Fits y ~ s(x1,k=10) + s(x2,k=10) + te(x1,x2,k=c(5,5)) in R
        and compares per-smooth constrained column counts.
        """
        rng = np.random.default_rng(SEED)
        x1 = rng.uniform(0, 1, N)
        x2 = rng.uniform(0, 1, N)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.5, N)
        dat = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        formula = (
            "y ~ s(x1, k=10, bs='tp') + s(x2, k=10, bs='tp') + te(x1, x2, k=c(5,5))"
        )
        r_result = r_bridge.get_smooth_components(formula, dat)

        # R column counts per smooth (from basis_matrices shapes)
        r_ncols = [b.shape[1] for b in r_result["basis_matrices"]]

        # Python pipeline
        sm1 = _setup_tprs("x1", x1)
        sm2 = _setup_tprs("x2", x2)
        data_dict = {"x1": x1, "x2": x2}
        sm_te = _setup_te(["x1", "x2"], data_dict, k=5)

        X1, S1 = _get_X_S(sm1, {"x1": x1})
        X2, S2 = _get_X_S(sm2, {"x2": x2})
        X_te, S_te = _get_X_S(sm_te, data_dict)

        coef_map, _, _ = CoefficientMap.build(
            smooths=[sm1, sm2, sm_te],
            X_smooth_blocks=[X1, X2, X_te],
            S_smooth_blocks=[S1, S2, S_te],
            has_intercept=True,
            n_parametric=1,
            X_parametric=np.ones((N, 1)),
        )

        py_ncols = [t.n_coefs for t in coef_map.terms if t.term_type == "smooth"]

        assert py_ncols == r_ncols, (
            f"Python smooth col counts {py_ncols} != R's {r_ncols}"
        )

    def test_gamside_te_del_index_count(self, r_bridge):
        """gam_side te model: te(x1,x2) deletion count matches R."""
        rng = np.random.default_rng(SEED)
        x1 = rng.uniform(0, 1, N)
        x2 = rng.uniform(0, 1, N)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.5, N)
        dat = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        formula = (
            "y ~ s(x1, k=10, bs='tp') + s(x2, k=10, bs='tp') + te(x1, x2, k=c(5,5))"
        )
        r_result = r_bridge.get_smooth_components(formula, dat)

        # R's te column count after constraints
        r_te_ncols = r_result["basis_matrices"][2].shape[1]
        # te raw: 5*5=25, after centering: 24
        r_te_deleted = 24 - r_te_ncols

        # Python pipeline
        sm1 = _setup_tprs("x1", x1)
        sm2 = _setup_tprs("x2", x2)
        data_dict = {"x1": x1, "x2": x2}
        sm_te = _setup_te(["x1", "x2"], data_dict, k=5)

        X1, S1 = _get_X_S(sm1, {"x1": x1})
        X2, S2 = _get_X_S(sm2, {"x2": x2})
        X_te, S_te = _get_X_S(sm_te, data_dict)

        coef_map, _, _ = CoefficientMap.build(
            smooths=[sm1, sm2, sm_te],
            X_smooth_blocks=[X1, X2, X_te],
            S_smooth_blocks=[S1, S2, S_te],
            has_intercept=True,
            n_parametric=1,
            X_parametric=np.ones((N, 1)),
        )

        te_term = coef_map.get_term("te(x1,x2)")
        assert len(te_term.del_index) == r_te_deleted, (
            f"Python deletes {len(te_term.del_index)} te cols, R deletes {r_te_deleted}"
        )

    def test_no_nesting_column_count(self, r_bridge):
        """No-nesting model s(x1)+s(x2): column counts match R, no deletions."""
        rng = np.random.default_rng(SEED)
        x1 = rng.uniform(0, 1, N)
        x2 = rng.uniform(0, 1, N)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.5, N)
        dat = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        formula = "y ~ s(x1, k=10, bs='tp') + s(x2, k=10, bs='tp')"
        r_result = r_bridge.get_smooth_components(formula, dat)

        r_ncols = [b.shape[1] for b in r_result["basis_matrices"]]
        r_total = 1 + sum(r_ncols)  # intercept + smooths

        # Python pipeline
        sm1 = _setup_tprs("x1", x1)
        sm2 = _setup_tprs("x2", x2)

        X1, S1 = _get_X_S(sm1, {"x1": x1})
        X2, S2 = _get_X_S(sm2, {"x2": x2})

        coef_map, _, _ = CoefficientMap.build(
            smooths=[sm1, sm2],
            X_smooth_blocks=[X1, X2],
            S_smooth_blocks=[S1, S2],
            has_intercept=True,
            n_parametric=1,
            X_parametric=np.ones((N, 1)),
        )

        assert coef_map.total_coefs == r_total, (
            f"Total coefs {coef_map.total_coefs} != R's {r_total}"
        )

        for term in coef_map.terms:
            if term.term_type == "smooth":
                assert len(term.del_index) == 0, (
                    f"{term.label} has del_index but R has none"
                )


