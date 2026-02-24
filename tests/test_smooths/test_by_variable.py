"""Tests for factor-by and numeric-by smooth expansion.

Validates FactorBySmooth and NumericBySmooth from pymgcv.smooths.by_variable:
- Factor detection tests (STRICT)
- FactorBySmooth structural invariant tests (STRICT)
- NumericBySmooth structural invariant tests (STRICT)
- R comparison tests (MODERATE, skip if R unavailable)
- Edge cases
- Phase boundary guard (no JAX imports)

Design doc reference: docs/design.md Section 5.7
R source reference: R/smooth.r smoothCon() by-variable handling
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from pymgcv.formula.terms import SmoothSpec
from pymgcv.smooths.by_variable import (
    FactorBySmooth,
    NumericBySmooth,
    is_factor,
    resolve_by_variable,
)
from pymgcv.smooths.tprs import TPRSSmooth
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
    bs: str = "tp",
    k: int = 10,
    by: str | None = None,
    **extra_args: object,
) -> SmoothSpec:
    """Create a SmoothSpec for testing."""
    return SmoothSpec(
        variables=variables,
        bs=bs,
        k=k,
        by=by,
        extra_args=dict(extra_args),
    )


def _make_factor_data(
    n: int = 200,
    n_levels: int = 3,
    seed: int = 42,
    ordered: bool = False,
) -> pd.DataFrame:
    """Generate test data with a factor variable.

    Returns DataFrame with columns: x, fac, z (numeric), y.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    level_names = [f"level{i}" for i in range(n_levels)]
    fac = rng.choice(level_names, n)
    fac = pd.Categorical(fac, categories=level_names, ordered=ordered)
    z = rng.uniform(0.5, 2.0, n)  # numeric by-variable
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "fac": fac, "z": z, "y": y})


def _setup_base_smooth(data: pd.DataFrame, k: int = 10, bs: str = "tp") -> TPRSSmooth:
    """Create and setup a base TPRS smooth on x."""
    spec = _make_spec(["x"], bs=bs, k=k)
    smooth = TPRSSmooth(spec)
    data_dict = {"x": data["x"].values}
    smooth.setup(data_dict)
    return smooth


# ===========================================================================
# 1. Factor detection tests (STRICT)
# ===========================================================================


class TestIsFactorDetection:
    """Tests for is_factor()."""

    def test_pandas_categorical_is_factor(self) -> None:
        """pandas Categorical dtype is a factor."""
        col = pd.Series(pd.Categorical(["a", "b", "c"]))
        assert is_factor(col) is True

    def test_pandas_ordered_categorical_is_factor(self) -> None:
        """pandas ordered Categorical is a factor."""
        col = pd.Series(pd.Categorical(["a", "b", "c"], ordered=True))
        assert is_factor(col) is True

    def test_pandas_object_dtype_is_factor(self) -> None:
        """pandas object dtype (strings) is a factor."""
        col = pd.Series(["a", "b", "c"], dtype=object)
        assert is_factor(col) is True

    def test_pandas_string_dtype_is_factor(self) -> None:
        """pandas string dtype is a factor."""
        col = pd.Series(["a", "b", "c"])
        # Default string columns may be object dtype
        assert is_factor(col) is True

    def test_integer_column_is_not_factor(self) -> None:
        """Integer column is NOT a factor (must be explicit)."""
        col = pd.Series([1, 2, 3], dtype=int)
        assert is_factor(col) is False

    def test_float_column_is_not_factor(self) -> None:
        """Float column is NOT a factor."""
        col = pd.Series([1.0, 2.0, 3.0], dtype=float)
        assert is_factor(col) is False

    def test_numpy_string_array_is_factor(self) -> None:
        """Numpy string array is a factor."""
        arr = np.array(["a", "b", "c"])
        assert is_factor(arr) is True

    def test_numpy_object_array_is_factor(self) -> None:
        """Numpy object array is a factor."""
        arr = np.array(["a", "b", "c"], dtype=object)
        assert is_factor(arr) is True

    def test_numpy_int_array_is_not_factor(self) -> None:
        """Numpy integer array is NOT a factor."""
        arr = np.array([1, 2, 3])
        assert is_factor(arr) is False

    def test_numpy_float_array_is_not_factor(self) -> None:
        """Numpy float array is NOT a factor."""
        arr = np.array([1.0, 2.0, 3.0])
        assert is_factor(arr) is False


# ===========================================================================
# 2. FactorBySmooth structural tests (STRICT)
# ===========================================================================


class TestFactorBySmoothStructure:
    """Structural invariant tests for FactorBySmooth."""

    @pytest.fixture
    def factor_data(self) -> pd.DataFrame:
        return _make_factor_data(n=200, n_levels=3)

    @pytest.fixture
    def base_smooth(self, factor_data: pd.DataFrame) -> TPRSSmooth:
        return _setup_base_smooth(factor_data, k=10)

    @pytest.fixture
    def factor_by(
        self, base_smooth: TPRSSmooth, factor_data: pd.DataFrame
    ) -> FactorBySmooth:
        spec = _make_spec(["x"], k=10, by="fac")
        levels = sorted(factor_data["fac"].cat.categories.tolist())
        return FactorBySmooth(
            base_smooth=base_smooth,
            spec=spec,
            levels=levels,
            by_variable="fac",
        )

    def test_n_coefs_equals_n_levels_times_k(
        self, factor_by: FactorBySmooth, base_smooth: TPRSSmooth
    ) -> None:
        """Total columns = n_levels x k_per_level."""
        assert factor_by.n_coefs == 3 * base_smooth.n_coefs

    def test_k_per_level_matches_base(
        self, factor_by: FactorBySmooth, base_smooth: TPRSSmooth
    ) -> None:
        """Per-level basis dimension matches base smooth."""
        assert factor_by.k_per_level == base_smooth.n_coefs

    def test_n_levels(self, factor_by: FactorBySmooth) -> None:
        """Number of levels is correct."""
        assert factor_by.n_levels == 3

    def test_design_matrix_shape(
        self, factor_by: FactorBySmooth, factor_data: pd.DataFrame
    ) -> None:
        """Design matrix has correct shape."""
        X = factor_by.build_design_matrix(factor_data)
        assert X.shape == (200, factor_by.n_coefs)

    def test_design_matrix_block_diagonal_structure(
        self, factor_by: FactorBySmooth, factor_data: pd.DataFrame
    ) -> None:
        """Each row has nonzeros only in its level's block."""
        X = factor_by.build_design_matrix(factor_data)
        k = factor_by.k_per_level

        for level_idx, level in enumerate(factor_by.levels):
            mask = (factor_data["fac"] == level).values
            other_mask = ~mask

            col_start = level_idx * k
            col_end = col_start + k

            # Rows for THIS level should have zeros in OTHER level blocks
            for other_idx in range(factor_by.n_levels):
                if other_idx == level_idx:
                    continue
                other_start = other_idx * k
                other_end = other_start + k
                np.testing.assert_array_equal(
                    X[mask, other_start:other_end],
                    0.0,
                    err_msg=(f"Level {level} rows should be zero in block {other_idx}"),
                )

            # Rows for OTHER levels should have zeros in THIS level block
            np.testing.assert_array_equal(
                X[other_mask, col_start:col_end],
                0.0,
                err_msg=f"Non-level-{level} rows should be zero in block {level_idx}",
            )

    def test_design_matrix_level_block_matches_base(
        self,
        factor_by: FactorBySmooth,
        base_smooth: TPRSSmooth,
        factor_data: pd.DataFrame,
    ) -> None:
        """Within a level's block, values match the base smooth's basis."""
        X = factor_by.build_design_matrix(factor_data)
        X_base = base_smooth.build_design_matrix({"x": factor_data["x"].values})
        k = factor_by.k_per_level

        for level_idx, level in enumerate(factor_by.levels):
            mask = (factor_data["fac"] == level).values
            col_start = level_idx * k
            col_end = col_start + k

            np.testing.assert_allclose(
                X[mask, col_start:col_end],
                X_base[mask],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"Level {level} block should match base smooth basis",
            )

    def test_penalty_count(self, factor_by: FactorBySmooth) -> None:
        """One penalty per level (for single-penalty base smooth)."""
        penalties = factor_by.build_penalty_matrices()
        # TPRS has 1 penalty, so factor-by with 3 levels has 3 penalties
        assert len(penalties) == 3

    def test_penalty_shape(self, factor_by: FactorBySmooth) -> None:
        """Each penalty is (n_coefs x n_coefs) - embedded in global space."""
        penalties = factor_by.build_penalty_matrices()
        for pen in penalties:
            assert pen.shape == (factor_by.n_coefs, factor_by.n_coefs)

    def test_penalty_symmetry(self, factor_by: FactorBySmooth) -> None:
        """Each penalty matrix is symmetric."""
        penalties = factor_by.build_penalty_matrices()
        for pen in penalties:
            np.testing.assert_allclose(
                pen.S,
                pen.S.T,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg="Penalty matrix must be symmetric",
            )

    def test_penalty_psd(self, factor_by: FactorBySmooth) -> None:
        """Each penalty matrix is positive semi-definite."""
        penalties = factor_by.build_penalty_matrices()
        for pen in penalties:
            eigvals = np.linalg.eigvalsh(pen.S)
            assert np.all(eigvals >= -STRICT.atol), (
                f"Penalty has negative eigenvalue: {eigvals.min()}"
            )

    def test_penalty_block_structure(
        self,
        factor_by: FactorBySmooth,
        base_smooth: TPRSSmooth,
    ) -> None:
        """Each penalty is nonzero only in its level's block."""
        penalties = factor_by.build_penalty_matrices()
        base_penalties = base_smooth.build_penalty_matrices()
        k = factor_by.k_per_level

        for level_idx, pen in enumerate(penalties):
            col_start = level_idx * k
            col_end = col_start + k

            # Check block matches base penalty
            np.testing.assert_allclose(
                pen.S[col_start:col_end, col_start:col_end],
                base_penalties[0].S,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"Penalty block {level_idx} should match base penalty",
            )

            # Check rest is zeros
            S_copy = pen.S.copy()
            S_copy[col_start:col_end, col_start:col_end] = 0.0
            np.testing.assert_allclose(
                S_copy,
                0.0,
                atol=STRICT.atol,
                err_msg=f"Penalty {level_idx} should be zero outside its block",
            )

    def test_labels(self, factor_by: FactorBySmooth) -> None:
        """Labels include by-variable and level names."""
        for label, level in zip(factor_by.labels, factor_by.levels, strict=True):
            assert "fac" in label
            assert str(level) in label
            assert "s(x)" in label

    def test_predict_matrix_matches_design(
        self,
        factor_by: FactorBySmooth,
        factor_data: pd.DataFrame,
    ) -> None:
        """predict_matrix on training data matches build_design_matrix."""
        X_design = factor_by.build_design_matrix(factor_data)
        X_predict = factor_by.predict_matrix(factor_data)
        np.testing.assert_allclose(
            X_design,
            X_predict,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


# ===========================================================================
# 3. NumericBySmooth structural tests (STRICT)
# ===========================================================================


class TestNumericBySmoothStructure:
    """Structural invariant tests for NumericBySmooth."""

    @pytest.fixture
    def factor_data(self) -> pd.DataFrame:
        return _make_factor_data(n=200)

    @pytest.fixture
    def base_smooth(self, factor_data: pd.DataFrame) -> TPRSSmooth:
        return _setup_base_smooth(factor_data, k=10)

    @pytest.fixture
    def numeric_by(self, base_smooth: TPRSSmooth) -> NumericBySmooth:
        spec = _make_spec(["x"], k=10, by="z")
        return NumericBySmooth(
            base_smooth=base_smooth,
            spec=spec,
            by_variable="z",
        )

    def test_n_coefs_same_as_base(
        self, numeric_by: NumericBySmooth, base_smooth: TPRSSmooth
    ) -> None:
        """Numeric-by doesn't change the number of coefficients."""
        assert numeric_by.n_coefs == base_smooth.n_coefs

    def test_design_matrix_shape(
        self, numeric_by: NumericBySmooth, factor_data: pd.DataFrame
    ) -> None:
        """Design matrix shape matches base."""
        X = numeric_by.build_design_matrix(factor_data)
        assert X.shape == (200, numeric_by.n_coefs)

    def test_design_matrix_is_elementwise_product(
        self,
        numeric_by: NumericBySmooth,
        base_smooth: TPRSSmooth,
        factor_data: pd.DataFrame,
    ) -> None:
        """Design matrix is z[:, None] * X_base."""
        X_by = numeric_by.build_design_matrix(factor_data)
        X_base = base_smooth.build_design_matrix({"x": factor_data["x"].values})
        z = factor_data["z"].values
        expected = z[:, np.newaxis] * X_base

        np.testing.assert_allclose(
            X_by,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_penalty_unchanged(
        self,
        numeric_by: NumericBySmooth,
        base_smooth: TPRSSmooth,
    ) -> None:
        """Penalty matrices are the same as base smooth."""
        by_penalties = numeric_by.build_penalty_matrices()
        base_penalties = base_smooth.build_penalty_matrices()
        assert len(by_penalties) == len(base_penalties)
        for by_pen, base_pen in zip(by_penalties, base_penalties, strict=True):
            np.testing.assert_allclose(
                by_pen.S,
                base_pen.S,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            )

    def test_no_centering_constraint(self, numeric_by: NumericBySmooth) -> None:
        """Centering constraint is removed for numeric-by."""
        assert numeric_by.has_centering_constraint is False

    def test_predict_matrix_matches_design(
        self,
        numeric_by: NumericBySmooth,
        factor_data: pd.DataFrame,
    ) -> None:
        """predict_matrix on training data matches build_design_matrix."""
        X_design = numeric_by.build_design_matrix(factor_data)
        X_predict = numeric_by.predict_matrix(factor_data)
        np.testing.assert_allclose(
            X_design,
            X_predict,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_label(self, numeric_by: NumericBySmooth) -> None:
        """Label includes by-variable name."""
        assert "z" in numeric_by.label
        assert "s(x)" in numeric_by.label


# ===========================================================================
# 4. resolve_by_variable tests
# ===========================================================================


class TestResolveByVariable:
    """Tests for the resolve_by_variable() factory function."""

    @pytest.fixture
    def factor_data(self) -> pd.DataFrame:
        return _make_factor_data(n=200, n_levels=3)

    @pytest.fixture
    def base_smooth(self, factor_data: pd.DataFrame) -> TPRSSmooth:
        return _setup_base_smooth(factor_data, k=10)

    def test_no_by_returns_smooth_unchanged(
        self, base_smooth: TPRSSmooth, factor_data: pd.DataFrame
    ) -> None:
        """No by-variable returns the base smooth."""
        spec = _make_spec(["x"], k=10, by=None)
        result = resolve_by_variable(spec, factor_data, base_smooth)
        assert result is base_smooth

    def test_factor_by_returns_factor_by_smooth(
        self, base_smooth: TPRSSmooth, factor_data: pd.DataFrame
    ) -> None:
        """Factor by-variable returns FactorBySmooth."""
        spec = _make_spec(["x"], k=10, by="fac")
        result = resolve_by_variable(spec, factor_data, base_smooth)
        assert isinstance(result, FactorBySmooth)

    def test_numeric_by_returns_numeric_by_smooth(
        self, base_smooth: TPRSSmooth, factor_data: pd.DataFrame
    ) -> None:
        """Numeric by-variable returns NumericBySmooth."""
        spec = _make_spec(["x"], k=10, by="z")
        result = resolve_by_variable(spec, factor_data, base_smooth)
        assert isinstance(result, NumericBySmooth)

    def test_factor_by_levels(
        self, base_smooth: TPRSSmooth, factor_data: pd.DataFrame
    ) -> None:
        """Factor-by has correct levels."""
        spec = _make_spec(["x"], k=10, by="fac")
        result = resolve_by_variable(spec, factor_data, base_smooth)
        assert isinstance(result, FactorBySmooth)
        assert result.levels == ["level0", "level1", "level2"]

    def test_ordered_factor_skips_reference_level(
        self, base_smooth: TPRSSmooth
    ) -> None:
        """Ordered factor skips the first (reference) level."""
        data = _make_factor_data(n=200, n_levels=3, ordered=True)
        spec = _make_spec(["x"], k=10, by="fac")
        result = resolve_by_variable(spec, data, base_smooth)
        assert isinstance(result, FactorBySmooth)
        # First level ("level0") should be skipped
        assert result.levels == ["level1", "level2"]
        assert result.n_levels == 2

    def test_missing_by_variable_raises(
        self, base_smooth: TPRSSmooth, factor_data: pd.DataFrame
    ) -> None:
        """Missing by-variable raises KeyError."""
        spec = _make_spec(["x"], k=10, by="nonexistent")
        with pytest.raises(KeyError, match="nonexistent"):
            resolve_by_variable(spec, factor_data, base_smooth)

    def test_nan_numeric_by_raises(
        self, base_smooth: TPRSSmooth, factor_data: pd.DataFrame
    ) -> None:
        """NaN in numeric by-variable raises ValueError."""
        data = factor_data.copy()
        data.loc[0, "z"] = np.nan
        spec = _make_spec(["x"], k=10, by="z")
        with pytest.raises(ValueError, match="NaN"):
            resolve_by_variable(spec, data, base_smooth)


# ===========================================================================
# 5. Parametrized tests across basis types
# ===========================================================================


class TestFactorByWithDifferentBases:
    """Test factor-by with different basis types."""

    @pytest.fixture
    def factor_data(self) -> pd.DataFrame:
        return _make_factor_data(n=200, n_levels=3)

    @pytest.mark.parametrize("bs", ["tp", "ts"])
    def test_tprs_variants(self, factor_data: pd.DataFrame, bs: str) -> None:
        """Factor-by works with tp and ts basis types."""
        from pymgcv.smooths.registry import get_smooth_class

        spec = _make_spec(["x"], bs=bs, k=10)
        smooth_cls = get_smooth_class(bs)
        smooth = smooth_cls(spec)
        smooth.setup({"x": factor_data["x"].values})

        by_spec = _make_spec(["x"], bs=bs, k=10, by="fac")
        levels = sorted(factor_data["fac"].cat.categories.tolist())
        fbs = FactorBySmooth(
            base_smooth=smooth,
            spec=by_spec,
            levels=levels,
            by_variable="fac",
        )

        X = fbs.build_design_matrix(factor_data)
        assert X.shape == (200, 3 * smooth.n_coefs)
        penalties = fbs.build_penalty_matrices()
        assert len(penalties) == 3

    @pytest.mark.parametrize("bs", ["cr", "cs", "cc"])
    def test_cubic_variants(self, factor_data: pd.DataFrame, bs: str) -> None:
        """Factor-by works with cr, cs, cc basis types."""
        from pymgcv.smooths.registry import get_smooth_class

        spec = _make_spec(["x"], bs=bs, k=10)
        smooth_cls = get_smooth_class(bs)
        smooth = smooth_cls(spec)
        smooth.setup({"x": factor_data["x"].values})

        by_spec = _make_spec(["x"], bs=bs, k=10, by="fac")
        levels = sorted(factor_data["fac"].cat.categories.tolist())
        fbs = FactorBySmooth(
            base_smooth=smooth,
            spec=by_spec,
            levels=levels,
            by_variable="fac",
        )

        X = fbs.build_design_matrix(factor_data)
        assert X.shape == (200, 3 * smooth.n_coefs)
        penalties = fbs.build_penalty_matrices()
        # cr and cc have 1 penalty, cs has 1 penalty too
        assert len(penalties) == 3


# ===========================================================================
# 6. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for by-variable expansion."""

    def test_single_level_factor(self) -> None:
        """Factor with 1 level: works, equivalent to no factor-by."""
        data = _make_factor_data(n=100, n_levels=1)
        smooth = _setup_base_smooth(data, k=8)

        spec = _make_spec(["x"], k=8, by="fac")
        result = resolve_by_variable(spec, data, smooth)
        assert isinstance(result, FactorBySmooth)
        assert result.n_levels == 1
        assert result.n_coefs == smooth.n_coefs

        X = result.build_design_matrix(data)
        X_base = smooth.build_design_matrix({"x": data["x"].values})
        np.testing.assert_allclose(X, X_base, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_many_levels(self) -> None:
        """Factor with many levels (10)."""
        data = _make_factor_data(n=500, n_levels=10)
        smooth = _setup_base_smooth(data, k=8)

        spec = _make_spec(["x"], k=8, by="fac")
        result = resolve_by_variable(spec, data, smooth)
        assert isinstance(result, FactorBySmooth)
        assert result.n_levels == 10
        assert result.n_coefs == 10 * smooth.n_coefs

        X = result.build_design_matrix(data)
        assert X.shape == (500, result.n_coefs)
        penalties = result.build_penalty_matrices()
        assert len(penalties) == 10

    def test_constant_numeric_by(self) -> None:
        """Constant numeric by-variable still works (degenerate case)."""
        data = _make_factor_data(n=100)
        data["z_const"] = 2.0
        smooth = _setup_base_smooth(data, k=8)

        spec = _make_spec(["x"], k=8, by="z_const")
        result = resolve_by_variable(spec, data, smooth)
        assert isinstance(result, NumericBySmooth)

        X = result.build_design_matrix(data)
        X_base = smooth.build_design_matrix({"x": data["x"].values})
        expected = 2.0 * X_base
        np.testing.assert_allclose(X, expected, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_zero_numeric_by(self) -> None:
        """Numeric by-variable that is all zeros produces all-zero design matrix."""
        data = _make_factor_data(n=100)
        data["z_zero"] = 0.0
        smooth = _setup_base_smooth(data, k=8)

        spec = _make_spec(["x"], k=8, by="z_zero")
        result = resolve_by_variable(spec, data, smooth)

        X = result.build_design_matrix(data)
        np.testing.assert_array_equal(X, 0.0)

    def test_base_smooth_must_be_setup(self) -> None:
        """FactorBySmooth requires base smooth to be setup."""
        spec = _make_spec(["x"], k=10, by="fac")
        smooth = TPRSSmooth(_make_spec(["x"], k=10))
        # smooth.setup() NOT called

        with pytest.raises(RuntimeError, match="setup"):
            FactorBySmooth(
                base_smooth=smooth,
                spec=spec,
                levels=["a", "b"],
                by_variable="fac",
            )

    def test_numeric_by_smooth_requires_setup(self) -> None:
        """NumericBySmooth requires base smooth to be setup."""
        spec = _make_spec(["x"], k=10, by="z")
        smooth = TPRSSmooth(_make_spec(["x"], k=10))

        with pytest.raises(RuntimeError, match="setup"):
            NumericBySmooth(
                base_smooth=smooth,
                spec=spec,
                by_variable="z",
            )

    def test_dict_data_input(self) -> None:
        """By-variable works with dict data (not just DataFrame)."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(0, 1, n)
        fac = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
        z = rng.uniform(0.5, 2.0, n)

        data = {"x": x, "fac": fac, "z": z}

        spec = _make_spec(["x"], k=8)
        smooth = TPRSSmooth(spec)
        smooth.setup({"x": x})

        # Factor by with dict
        by_spec = _make_spec(["x"], k=8, by="fac")
        result = resolve_by_variable(by_spec, data, smooth)
        assert isinstance(result, FactorBySmooth)
        X = result.build_design_matrix(data)
        assert X.shape == (n, 3 * smooth.n_coefs)

        # Numeric by with dict
        num_spec = _make_spec(["x"], k=8, by="z")
        result_num = resolve_by_variable(num_spec, data, smooth)
        assert isinstance(result_num, NumericBySmooth)
        X_num = result_num.build_design_matrix(data)
        assert X_num.shape == (n, smooth.n_coefs)

    def test_predict_on_new_data_with_unseen_levels(self) -> None:
        """predict_matrix with data where some levels have no observations."""
        data = _make_factor_data(n=200, n_levels=3)
        smooth = _setup_base_smooth(data, k=8)
        spec = _make_spec(["x"], k=8, by="fac")
        result = resolve_by_variable(spec, data, smooth)
        assert isinstance(result, FactorBySmooth)

        # New data with only level0
        rng = np.random.default_rng(99)
        new_data = pd.DataFrame(
            {
                "x": rng.uniform(0, 1, 50),
                "fac": pd.Categorical(
                    ["level0"] * 50,
                    categories=["level0", "level1", "level2"],
                ),
            }
        )
        X_new = result.predict_matrix(new_data)
        assert X_new.shape == (50, result.n_coefs)

        # Only level0 block should have nonzeros
        k = result.k_per_level
        assert np.any(X_new[:, :k] != 0)  # level0 block
        np.testing.assert_array_equal(X_new[:, k:], 0.0)  # level1 + level2


# ===========================================================================
# 7. R comparison tests (MODERATE, skip if R unavailable)
# ===========================================================================


class TestRComparison:
    """R comparison tests: element-wise numerical comparisons at MODERATE.

    Uses smoothCon() with absorb.cons=FALSE to get pre-constraint basis
    matrices from R, then compares against Python's FactorBySmooth and
    NumericBySmooth outputs.
    """

    @pytest.fixture
    def r_bridge(self):
        from pymgcv.compat.r_bridge import RBridge

        if not RBridge.available():
            pytest.skip("R with mgcv not available")
        return RBridge()

    @staticmethod
    def _make_shared_data(seed: int = 42) -> pd.DataFrame:
        """Generate shared data for R comparison tests."""
        rng = np.random.default_rng(seed)
        n = 200
        x = rng.uniform(0, 1, n)
        fac_vals = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
        z = rng.uniform(0.5, 2.0, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.5, n)
        return pd.DataFrame(
            {
                "x": x,
                "fac": pd.Categorical(fac_vals, categories=["a", "b", "c"]),
                "z": z,
                "y": y,
            }
        )

    def test_factor_by_expansion_count_matches_r(self, r_bridge) -> None:
        """smoothCon(s(x, by=fac)) returns one smooth per factor level."""
        data = self._make_shared_data()
        r_smooths = r_bridge.smooth_construct_list(
            "s(x, by=fac, bs='tp', k=10)", data, absorb_cons=False
        )
        assert len(r_smooths) == 3

    def test_factor_by_levels_match_r(self, r_bridge) -> None:
        """Factor level ordering matches R's levels()."""
        data = self._make_shared_data()
        r_smooths = r_bridge.smooth_construct_list(
            "s(x, by=fac, bs='tp', k=10)", data, absorb_cons=False
        )
        r_levels = [sm["by_level"] for sm in r_smooths]
        # R alphabetical: ["a", "b", "c"]
        assert r_levels == ["a", "b", "c"]

    def test_factor_by_design_matrix_elementwise_vs_r(self, r_bridge) -> None:
        """Each level's design matrix matches R element-wise at MODERATE.

        R's smoothCon(s(x, by=fac)) returns per-level (n x k) matrices
        where rows not belonging to the level are zeroed. Our
        FactorBySmooth assembles these into a single (n x 3k) block.
        Sign normalization removes LAPACK eigenvector sign ambiguity
        (R bundles reference LAPACK; Python may use Accelerate/MKL).
        """
        data = self._make_shared_data()
        k = 10

        # --- R side: get smoothCon output (no constraint absorption) ---
        r_smooths = r_bridge.smooth_construct_list(
            "s(x, by=fac, bs='tp', k=10)", data, absorb_cons=False
        )

        # --- Python side: construct matching FactorBySmooth ---
        spec = _make_spec(["x"], bs="tp", k=k, by="fac")
        smooth = TPRSSmooth(_make_spec(["x"], bs="tp", k=k))
        smooth.setup({"x": data["x"].values})

        fbs = FactorBySmooth(
            base_smooth=smooth,
            spec=spec,
            levels=["a", "b", "c"],
            by_variable="fac",
        )
        X_py = fbs.build_design_matrix(data)

        # --- Compare each level's block ---
        for level_idx, r_sm in enumerate(r_smooths):
            r_X = r_sm["X"]  # (n, k) with zeros for non-level rows
            col_start = level_idx * fbs.k_per_level
            col_end = col_start + fbs.k_per_level

            py_block = X_py[:, col_start:col_end]

            assert py_block.shape == r_X.shape, (
                f"Level {level_idx}: shape mismatch "
                f"py={py_block.shape} vs r={r_X.shape}"
            )

            np.testing.assert_allclose(
                normalize_column_signs(py_block),
                normalize_column_signs(r_X),
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=(
                    f"Factor-by level {r_sm['by_level']}: "
                    f"design matrix does not match R"
                ),
            )

    def test_factor_by_penalty_matches_r(self, r_bridge) -> None:
        """Per-level penalty matrices match R element-wise at MODERATE.

        S transforms as D @ S @ D under column sign flips. The sign
        vector D is derived from X (the design matrix) and applied to
        both Python and R penalties so element-wise comparison is valid.
        """
        data = self._make_shared_data()
        k = 10

        r_smooths = r_bridge.smooth_construct_list(
            "s(x, by=fac, bs='tp', k=10)", data, absorb_cons=False
        )

        spec = _make_spec(["x"], bs="tp", k=k, by="fac")
        smooth = TPRSSmooth(_make_spec(["x"], bs="tp", k=k))
        smooth.setup({"x": data["x"].values})

        fbs = FactorBySmooth(
            base_smooth=smooth,
            spec=spec,
            levels=["a", "b", "c"],
            by_variable="fac",
        )
        X_py = fbs.build_design_matrix(data)
        py_penalties = fbs.build_penalty_matrices()

        # Each level has 1 penalty in R (tp has 1 penalty)
        for level_idx, r_sm in enumerate(r_smooths):
            assert len(r_sm["S"]) == 1, (
                f"R level {level_idx}: expected 1 penalty, got {len(r_sm['S'])}"
            )
            r_X = r_sm["X"]
            r_S = r_sm["S"][0]

            # Extract the per-level block from our embedded penalty
            py_pen = py_penalties[level_idx]
            col_start = level_idx * fbs.k_per_level
            col_end = col_start + fbs.k_per_level
            py_S_block = py_pen.S[col_start:col_end, col_start:col_end]
            py_X_block = X_py[:, col_start:col_end]

            np.testing.assert_allclose(
                normalize_symmetric_signs(py_S_block, py_X_block),
                normalize_symmetric_signs(r_S, r_X),
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=(
                    f"Factor-by level {r_sm['by_level']}: "
                    f"penalty matrix does not match R"
                ),
            )

    def test_factor_by_rank_and_null_dim_match_r(self, r_bridge) -> None:
        """Rank and null space dimension match R for each level."""
        data = self._make_shared_data()

        r_smooths = r_bridge.smooth_construct_list(
            "s(x, by=fac, bs='tp', k=10)", data, absorb_cons=False
        )

        smooth = TPRSSmooth(_make_spec(["x"], bs="tp", k=10))
        smooth.setup({"x": data["x"].values})

        for r_sm in r_smooths:
            assert smooth.rank == r_sm["rank"], (
                f"Level {r_sm['by_level']}: rank py={smooth.rank} != r={r_sm['rank']}"
            )
            assert smooth.null_space_dim == r_sm["null_space_dim"], (
                f"Level {r_sm['by_level']}: "
                f"null_space_dim py={smooth.null_space_dim} "
                f"!= r={r_sm['null_space_dim']}"
            )

    def test_numeric_by_design_matrix_elementwise_vs_r(self, r_bridge) -> None:
        """Numeric-by design matrix matches R element-wise at MODERATE.

        R's smoothCon(s(x, by=z)) returns one smooth whose X is the
        base basis multiplied by z. Sign normalization removes LAPACK
        eigenvector sign ambiguity.
        """
        data = self._make_shared_data()
        k = 10

        r_smooths = r_bridge.smooth_construct_list(
            "s(x, by=z, bs='tp', k=10)", data, absorb_cons=False
        )
        assert len(r_smooths) == 1
        r_X = r_smooths[0]["X"]

        # Python side
        smooth = TPRSSmooth(_make_spec(["x"], bs="tp", k=k))
        smooth.setup({"x": data["x"].values})

        spec = _make_spec(["x"], bs="tp", k=k, by="z")
        nbs = NumericBySmooth(
            base_smooth=smooth,
            spec=spec,
            by_variable="z",
        )
        py_X = nbs.build_design_matrix(data)

        assert py_X.shape == r_X.shape, (
            f"Numeric-by: shape mismatch py={py_X.shape} vs r={r_X.shape}"
        )

        np.testing.assert_allclose(
            normalize_column_signs(py_X),
            normalize_column_signs(r_X),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Numeric-by design matrix does not match R",
        )

    def test_numeric_by_penalty_matches_r(self, r_bridge) -> None:
        """Numeric-by penalty matches R element-wise at MODERATE.

        The sign vector D is derived from X and applied to S via
        D @ S @ D to remove LAPACK eigenvector sign ambiguity.
        """
        data = self._make_shared_data()
        k = 10

        r_smooths = r_bridge.smooth_construct_list(
            "s(x, by=z, bs='tp', k=10)", data, absorb_cons=False
        )
        r_X = r_smooths[0]["X"]
        r_S = r_smooths[0]["S"][0]

        smooth = TPRSSmooth(_make_spec(["x"], bs="tp", k=k))
        smooth.setup({"x": data["x"].values})

        spec = _make_spec(["x"], bs="tp", k=k, by="z")
        nbs = NumericBySmooth(
            base_smooth=smooth,
            spec=spec,
            by_variable="z",
        )
        py_X = nbs.build_design_matrix(data)
        py_S = nbs.build_penalty_matrices()[0].S

        np.testing.assert_allclose(
            normalize_symmetric_signs(py_S, py_X),
            normalize_symmetric_signs(r_S, r_X),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Numeric-by penalty matrix does not match R",
        )

    @pytest.mark.parametrize("bs", ["tp", "cr"])
    def test_factor_by_basis_type_variants_vs_r(self, r_bridge, bs: str) -> None:
        """Factor-by matches R across basis types (tp, cr).

        Sign normalization handles LAPACK eigenvector sign ambiguity
        for TPRS-based bases (tp). Cubic bases (cr) have no sign
        ambiguity so normalization is a no-op.
        """
        from pymgcv.smooths.registry import get_smooth_class

        data = self._make_shared_data()
        k = 10

        r_smooths = r_bridge.smooth_construct_list(
            f"s(x, by=fac, bs='{bs}', k={k})", data, absorb_cons=False
        )
        assert len(r_smooths) == 3

        smooth_cls = get_smooth_class(bs)
        smooth = smooth_cls(_make_spec(["x"], bs=bs, k=k))
        smooth.setup({"x": data["x"].values})

        fbs = FactorBySmooth(
            base_smooth=smooth,
            spec=_make_spec(["x"], bs=bs, k=k, by="fac"),
            levels=["a", "b", "c"],
            by_variable="fac",
        )
        X_py = fbs.build_design_matrix(data)

        for level_idx, r_sm in enumerate(r_smooths):
            col_start = level_idx * fbs.k_per_level
            col_end = col_start + fbs.k_per_level
            py_block = X_py[:, col_start:col_end]

            np.testing.assert_allclose(
                normalize_column_signs(py_block),
                normalize_column_signs(r_sm["X"]),
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=(
                    f"bs={bs}, level {r_sm['by_level']}: design matrix does not match R"
                ),
            )


# ===========================================================================
# 8. Phase boundary guard
# ===========================================================================


class TestPhaseBoundary:
    """Ensure no JAX imports in this Phase 1 module."""

    def test_no_jax_import(self) -> None:
        """Importing by_variable must not trigger JAX import."""
        # Remove any cached by_variable module
        mod_name = "pymgcv.smooths.by_variable"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        # Track loaded modules before import
        pre_modules = set(sys.modules.keys())
        importlib.import_module(mod_name)
        post_modules = set(sys.modules.keys())

        new_modules = post_modules - pre_modules
        jax_modules = {m for m in new_modules if m.startswith("jax")}
        assert not jax_modules, (
            f"Importing {mod_name} triggered JAX imports: {jax_modules}"
        )
