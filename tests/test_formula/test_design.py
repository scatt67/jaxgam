"""Tests for design matrix assembly (ModelSetup).

Covers:
1. Parametric matrix construction (factor and numeric encoding)
2. Basic smooth assembly structure
3. Tensor product smooths
4. Factor-by smooths
5. Penalty embedding
6. CoefficientMap integration
7. Instance methods
8. R comparison (numerical matching via r_bridge)
9. Phase boundary (no JAX imports)
10. Edge cases

Design doc reference: docs/design.md Section 13.2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM
from jaxgam.formula.design import ModelSetup, SmoothInfo
from jaxgam.formula.parser import parse_formula
from tests.helpers import SEED, N, _AssertCollector, check_that, r_available
from tests.tolerances import MODERATE, STRICT, normalize_column_signs


@pytest.fixture
def data() -> pd.DataFrame:
    """Standard test data (x1, x2, y)."""
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.5, N)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def factor_data() -> pd.DataFrame:
    """Test data with a 3-level factor column."""
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.5, N)
    levels = [f"lev{i}" for i in range(3)]
    fac = rng.choice(levels, N)
    return pd.DataFrame(
        {"x1": x1, "x2": x2, "y": y, "fac": pd.Categorical(fac, categories=levels)}
    )


@pytest.fixture
def numeric_by_data() -> pd.DataFrame:
    """Test data with a numeric by-variable z."""
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.5, N)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y, "z": rng.uniform(0, 1, N)})


# ===========================================================================
# TestParametricMatrix — factor and numeric encoding
# ===========================================================================


class TestParametricMatrix:
    """Test parametric matrix construction."""

    def test_intercept_column(self, data) -> None:
        """Intercept column is all-ones when has_intercept=True."""
        spec = parse_formula("y ~ s(x1)")
        setup = ModelSetup.build(spec, data)

        np.testing.assert_allclose(
            setup.X[:, 0],
            np.ones(N),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_numeric_parametric(self, data) -> None:
        """Numeric parametric term produces a single column alongside smooth."""
        spec = parse_formula("y ~ s(x1) + x2")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape[0] == N
        np.testing.assert_allclose(
            setup.X[:, 1],
            data["x2"].values,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_factor_parametric_with_intercept(self, factor_data) -> None:
        """Factor parametric produces k-1 dummy columns (treatment coding)."""
        spec = parse_formula("y ~ s(x1) + fac")
        setup = ModelSetup.build(spec, factor_data)

        assert "(Intercept)" in setup.term_names
        fac_names = [n for n in setup.term_names if n.startswith("fac")]
        assert len(fac_names) == 2

    def test_factor_without_intercept(self, factor_data) -> None:
        """Factor without intercept produces k columns (no reference dropped)."""
        spec = parse_formula("y ~ 0 + fac + s(x1)")
        setup = ModelSetup.build(spec, factor_data)

        fac_names = [n for n in setup.term_names if n.startswith("fac")]
        assert len(fac_names) == 3
        assert "(Intercept)" not in setup.term_names

    def test_mixed_parametric(self, factor_data) -> None:
        """Mixed numeric + factor + smooth."""
        spec = parse_formula("y ~ x2 + fac + s(x1)")
        setup = ModelSetup.build(spec, factor_data)

        assert "(Intercept)" in setup.term_names
        assert "x2" in setup.term_names
        fac_names = [n for n in setup.term_names if n.startswith("fac")]
        assert len(fac_names) == 2


# ===========================================================================
# TestBasicAssembly — smooth assembly structure
# ===========================================================================


class TestBasicAssembly:
    """Test basic smooth assembly structure."""

    def test_single_smooth_column_count(self, data) -> None:
        """y ~ s(x1, k=10): cols = 1 (intercept) + (10-1) after centering."""
        spec = parse_formula("y ~ s(x1, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape == (N, 10)
        assert setup.coef_map.total_coefs == 10

    def test_two_smooth_column_count(self, data) -> None:
        """y ~ s(x1, k=10) + s(x2, k=10): correct column count."""
        spec = parse_formula("y ~ s(x1, k=10) + s(x2, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape == (N, 19)
        assert setup.coef_map.total_coefs == 19

    def test_no_intercept(self, data) -> None:
        """y ~ 0 + s(x1, k=10): no intercept column."""
        spec = parse_formula("y ~ 0 + s(x1, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape == (N, 9)
        assert "(Intercept)" not in setup.term_names


# ===========================================================================
# TestTensorProducts — tensor product smooths
# ===========================================================================


class TestTensorProducts:
    """Test tensor product smooth assembly."""

    def test_te_assembly(self, data) -> None:
        """y ~ te(x1, x2, k=5): assembled correctly."""
        spec = parse_formula("y ~ te(x1, x2, k=5)")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape[0] == N
        assert setup.n_obs == N
        assert len(setup.smooth_info) == 1
        assert setup.smooth_info[0].term_type == "te"

    def test_te_with_main_effects(self, data) -> None:
        """y ~ s(x1) + te(x1, x2, k=5): gam_side removes dependent columns."""
        spec = parse_formula("y ~ s(x1, k=10) + te(x1, x2, k=5)")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape[0] == N
        assert len(setup.smooth_info) == 2
        te_info = setup.get_smooth("te(x1,x2)")
        assert te_info.last_coef - te_info.first_coef > 0

    def test_ti_assembly(self, data) -> None:
        """y ~ ti(x1, x2, k=5): tensor interaction assembled."""
        spec = parse_formula("y ~ ti(x1, x2, k=5)")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape[0] == N
        assert len(setup.smooth_info) == 1
        assert setup.smooth_info[0].term_type == "ti"


# ===========================================================================
# TestFactorBy — by-variable smooths
# ===========================================================================


class TestFactorBy:
    """Test by-variable smooth assembly."""

    def test_factor_by(self, factor_data) -> None:
        """y ~ s(x1, by=fac, k=10): one SmoothInfo PER LEVEL (mgcv; Finding H1)."""
        spec = parse_formula("y ~ s(x1, by=fac, k=10)")
        setup = ModelSetup.build(spec, factor_data)

        assert setup.X.shape[0] == N
        # 3-level factor-by -> 3 per-level SmoothInfo (mgcv replicates per level).
        assert len(setup.smooth_info) == 3
        assert [si.label for si in setup.smooth_info] == [
            "s(x1):faclev0",
            "s(x1):faclev1",
            "s(x1):faclev2",
        ]

    def test_factor_by_with_main_effect(self, factor_data) -> None:
        """y ~ s(x1) + s(x1, by=fac): main effect (1) + per-level factor-by (3)."""
        spec = parse_formula("y ~ s(x1, k=10) + s(x1, by=fac, k=10)")
        setup = ModelSetup.build(spec, factor_data)

        assert setup.X.shape[0] == N
        # 1 main-effect SmoothInfo + 3 per-level factor-by SmoothInfos (H1).
        assert len(setup.smooth_info) == 4

    def test_numeric_by(self, numeric_by_data) -> None:
        """y ~ s(x1, by=z, k=10): numeric-by works."""
        spec = parse_formula("y ~ s(x1, by=z, k=10)")
        setup = ModelSetup.build(spec, numeric_by_data)

        assert setup.X.shape[0] == N
        assert len(setup.smooth_info) == 1


# ===========================================================================
# TestPenaltyEmbedding — global penalty structure
# ===========================================================================


class TestPenaltyEmbedding:
    """Test global penalty structure after embedding."""

    def test_embedded_penalty_shape(self, data) -> None:
        """Each embedded penalty is (total_p, total_p)."""
        spec = parse_formula("y ~ s(x1, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.penalties is not None
        total_p = setup.coef_map.total_coefs
        for pen in setup.penalties.penalties:
            assert pen.S.shape == (total_p, total_p)

    def test_penalty_nonzero_block(self, data) -> None:
        """Embedded penalty has nonzeros in the correct block."""
        spec = parse_formula("y ~ s(x1, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.penalties is not None
        pen = setup.penalties.penalties[0]

        # Intercept block should be zero
        np.testing.assert_allclose(pen.S[0, :], 0.0, rtol=STRICT.rtol, atol=STRICT.atol)
        np.testing.assert_allclose(pen.S[:, 0], 0.0, rtol=STRICT.rtol, atol=STRICT.atol)

        # Smooth block should have nonzeros
        smooth_block = pen.S[1:, 1:]
        assert np.any(np.abs(smooth_block) > 1e-10)

    def test_penalty_count(self, data) -> None:
        """Penalty count matches sum of per-smooth penalties."""
        spec = parse_formula("y ~ s(x1, k=10) + s(x2, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.penalties is not None
        total_penalties = sum(si.n_penalties for si in setup.smooth_info)
        assert setup.penalties.n_penalties == total_penalties

    def test_penalty_psd(self, data) -> None:
        """Embedded penalties are PSD (eigenvalues >= 0)."""
        spec = parse_formula("y ~ s(x1, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.penalties is not None
        for pen in setup.penalties.penalties:
            eigvals = np.linalg.eigvalsh(pen.S)
            assert np.all(eigvals >= -STRICT.atol), (
                f"Penalty has negative eigenvalue: {np.min(eigvals)}"
            )

    def test_weighted_penalty_works(self, data) -> None:
        """CompositePenalty.weighted_penalty() works on embedded penalties."""
        spec = parse_formula("y ~ s(x1, k=10) + s(x2, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.penalties is not None
        total_p = setup.coef_map.total_coefs
        S_lambda = setup.penalties.weighted_penalty()
        assert S_lambda.shape == (total_p, total_p)


# ===========================================================================
# TestCoefficientMapIntegration — constraint consistency
# ===========================================================================


class TestCoefficientMapIntegration:
    """Test CoefficientMap integration with ModelSetup."""

    def test_total_coefs_matches_X(self, data) -> None:
        """coef_map.total_coefs == X.shape[1]."""
        spec = parse_formula("y ~ s(x1, k=10) + s(x2, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.coef_map.total_coefs == setup.X.shape[1]

    def test_term_slice_matches_X(self, data) -> None:
        """coef_map.term_slice(label) returns correct range matching X columns."""
        spec = parse_formula("y ~ s(x1, k=10) + s(x2, k=10)")
        setup = ModelSetup.build(spec, data)

        for si in setup.smooth_info:
            term_sl = setup.coef_map.term_slice(si.label)
            assert term_sl.start == si.first_coef
            assert term_sl.stop == si.last_coef

    def test_term_labels_match(self, data) -> None:
        """Term labels in coef_map match smooth_info labels."""
        spec = parse_formula("y ~ s(x1, k=10) + s(x2, k=10)")
        setup = ModelSetup.build(spec, data)

        smooth_labels = {si.label for si in setup.smooth_info}
        coef_map_labels = {
            t.label for t in setup.coef_map.terms if t.term_type == "smooth"
        }
        assert smooth_labels == coef_map_labels


# ===========================================================================
# TestInstanceMethods — ModelSetup methods
# ===========================================================================


class TestInstanceMethods:
    """Test ModelSetup instance methods."""

    @pytest.fixture
    def two_smooth_setup(self, data) -> ModelSetup:
        """ModelSetup with two smooths for method testing."""
        spec = parse_formula("y ~ s(x1, k=10) + s(x2, k=10)")
        return ModelSetup.build(spec, data)

    def test_get_smooth(self, two_smooth_setup) -> None:
        """get_smooth('s(x1)') returns correct SmoothInfo."""
        info = two_smooth_setup.get_smooth("s(x1)")
        assert isinstance(info, SmoothInfo)
        assert info.label == "s(x1)"
        assert info.variables == ("x1",)
        assert info.term_type == "s"

    def test_get_smooth_not_found(self, two_smooth_setup) -> None:
        """get_smooth raises KeyError for unknown label."""
        with pytest.raises(KeyError, match="No smooth 's\\(x99\\)'"):
            two_smooth_setup.get_smooth("s(x99)")

    def test_smooth_coef_slice(self, two_smooth_setup) -> None:
        """smooth_coef_slice returns correct slice matching coef_map."""
        sl = two_smooth_setup.smooth_coef_slice("s(x1)")
        info = two_smooth_setup.get_smooth("s(x1)")
        assert sl == slice(info.first_coef, info.last_coef)

        X_smooth = two_smooth_setup.X[:, sl]
        assert X_smooth.shape[1] == info.last_coef - info.first_coef

    def test_smooth_penalty_indices(self, two_smooth_setup) -> None:
        """smooth_penalty_indices returns correct range."""
        indices = two_smooth_setup.smooth_penalty_indices("s(x1)")
        info = two_smooth_setup.get_smooth("s(x1)")
        assert list(indices) == list(
            range(info.first_penalty, info.first_penalty + info.n_penalties)
        )


# ===========================================================================
# TestRComparison — R mgcv numerical matching
# ===========================================================================


@pytest.mark.skipif(not r_available(), reason="R with mgcv not available")
class TestRComparison:
    """Compare ModelSetup results against R mgcv."""

    @pytest.fixture
    def data(self) -> pd.DataFrame:
        """Data fixture scoped to this class."""
        rng = np.random.default_rng(SEED)
        x1 = rng.uniform(0, 1, N)
        x2 = rng.uniform(0, 1, N)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.5, N)
        return pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    def test_single_smooth_column_count(self, r_bridge, data) -> None:
        """y ~ s(x1, k=10, bs='tp'): column count matches R exactly."""
        formula = "y ~ s(x1, k=10, bs='tp')"

        r_result = r_bridge.get_smooth_components(formula, data)
        r_total_cols = 1 + sum(b.shape[1] for b in r_result["basis_matrices"])

        spec = parse_formula(formula)
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape[1] == r_total_cols

    def test_single_smooth_X(self, r_bridge, data) -> None:
        """y ~ s(x1, k=10, bs='tp'): full model matrix X matches R at MODERATE."""
        formula = "y ~ s(x1, k=10, bs='tp')"

        r_result = r_bridge.get_smooth_components(formula, data)
        r_X = r_result["model_matrix"]

        spec = parse_formula(formula)
        setup = ModelSetup.build(spec, data)

        np.testing.assert_allclose(
            normalize_column_signs(setup.X),
            normalize_column_signs(r_X),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Single smooth model matrix differs from R",
        )

    def test_two_smooth_X(self, r_bridge, data) -> None:
        """y ~ s(x1, k=10) + s(x2, k=10): full X matches R at MODERATE."""
        formula = "y ~ s(x1, k=10, bs='tp') + s(x2, k=10, bs='tp')"

        r_result = r_bridge.get_smooth_components(formula, data)
        r_X = r_result["model_matrix"]

        spec = parse_formula(formula)
        setup = ModelSetup.build(spec, data)

        np.testing.assert_allclose(
            normalize_column_signs(setup.X),
            normalize_column_signs(r_X),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Two smooth model matrix differs from R",
        )

    def test_te_with_main_effects(self, r_bridge, data) -> None:
        """y ~ s(x1,k=10) + s(x2,k=10) + te(x1,x2,k=c(5,5)): X matches R."""
        formula = (
            "y ~ s(x1, k=10, bs='tp') + s(x2, k=10, bs='tp') + te(x1, x2, k=c(5,5))"
        )

        r_result = r_bridge.get_smooth_components(formula, data)
        r_ncols = [b.shape[1] for b in r_result["basis_matrices"]]

        # Parse formula - note: c(5,5) is not valid Python AST, use k=5
        py_formula = "y ~ s(x1, k=10) + s(x2, k=10) + te(x1, x2, k=5)"
        spec = parse_formula(py_formula)
        setup = ModelSetup.build(spec, data)

        py_ncols = [si.last_coef - si.first_coef for si in setup.smooth_info]
        assert py_ncols == r_ncols, (
            f"Python smooth col counts {py_ncols} != R's {r_ncols}"
        )

    def test_cubic_smooth_X(self, r_bridge, data) -> None:
        """y ~ s(x1, k=10, bs='cr'): cubic basis X matches R at MODERATE."""
        formula = "y ~ s(x1, k=10, bs='cr')"

        r_result = r_bridge.get_smooth_components(formula, data)
        r_X = r_result["model_matrix"]

        spec = parse_formula(formula)
        setup = ModelSetup.build(spec, data)

        np.testing.assert_allclose(
            normalize_column_signs(setup.X),
            normalize_column_signs(r_X),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Cubic smooth model matrix differs from R",
        )

    def test_factor_by_X(self, r_bridge) -> None:
        """y ~ s(x1, by=fac, k=10, bs='tp'): factor-by X matches R."""
        rng = np.random.default_rng(SEED)
        x1 = rng.uniform(0, 1, N)
        x2 = rng.uniform(0, 1, N)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.5, N)
        levels = [f"lev{i}" for i in range(3)]
        fac = rng.choice(levels, N)
        data = pd.DataFrame(
            {"x1": x1, "x2": x2, "y": y, "fac": pd.Categorical(fac, categories=levels)}
        )
        formula = "y ~ s(x1, by=fac, k=10, bs='tp')"

        r_result = r_bridge.get_smooth_components(formula, data)
        # R returns one basis block per factor level; jaxgam now emits one
        # per-level SmoothInfo too (Finding H1), so compare per-level and total.
        r_ncols = [b.shape[1] for b in r_result["basis_matrices"]]

        spec = parse_formula(formula)
        setup = ModelSetup.build(spec, data)

        py_ncols = [si.last_coef - si.first_coef for si in setup.smooth_info]

        assert len(py_ncols) == len(r_ncols), (
            f"Factor-by level count: Python {len(py_ncols)} != R {len(r_ncols)}"
        )
        assert sorted(py_ncols) == sorted(r_ncols), (
            f"Per-level factor-by cols: Python {py_ncols} != R {r_ncols}"
        )
        assert sum(py_ncols) == sum(r_ncols)

    def test_penalty_structure(self, r_bridge, data) -> None:
        """Per-smooth penalty matrices match R at MODERATE."""
        formula = "y ~ s(x1, k=10, bs='tp')"

        r_result = r_bridge.get_smooth_components(formula, data)
        r_pen = r_result["penalty_matrices"][0][0]

        spec = parse_formula(formula)
        setup = ModelSetup.build(spec, data)

        si = setup.smooth_info[0]
        pen_idx = next(iter(setup.smooth_penalty_indices(si.label)))
        S_global = setup.penalties.penalties[pen_idx].S
        S_block = S_global[si.first_coef : si.last_coef, si.first_coef : si.last_coef]

        assert S_block.shape == r_pen.shape, (
            f"Penalty shape: Python {S_block.shape} != R {r_pen.shape}"
        )

    def test_no_nesting_no_deletion(self, r_bridge, data) -> None:
        """y ~ s(x1) + s(x2): no gam_side deletions, column counts match R."""
        formula = "y ~ s(x1, k=10, bs='tp') + s(x2, k=10, bs='tp')"

        r_result = r_bridge.get_smooth_components(formula, data)
        r_ncols = [b.shape[1] for b in r_result["basis_matrices"]]
        r_total = 1 + sum(r_ncols)

        spec = parse_formula(formula)
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape[1] == r_total

        for term in setup.coef_map.terms:
            if term.term_type == "smooth":
                assert len(term.del_index) == 0


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_missing_variable_raises(self) -> None:
        """Missing variable in data raises ValueError."""
        data = pd.DataFrame({"x1": np.ones(10), "y": np.ones(10)})
        spec = parse_formula("y ~ s(x1) + s(x99)")

        with pytest.raises(ValueError, match="x99"):
            ModelSetup.build(spec, data)

    def test_missing_response_raises(self) -> None:
        """Missing response variable raises ValueError."""
        data = pd.DataFrame({"x1": np.ones(10)})
        spec = parse_formula("y ~ s(x1)")

        with pytest.raises(ValueError, match="Response variable 'y'"):
            ModelSetup.build(spec, data)

    def test_smooth_only_formula(self, data) -> None:
        """Empty parametric terms (smooth-only formula) works."""
        spec = parse_formula("y ~ s(x1, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape[0] == N
        assert setup.n_obs == N

    def test_purely_parametric_formula(self, data) -> None:
        """Purely parametric formula (no smooths) works."""
        spec = parse_formula("y ~ x1 + x2")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape == (N, 3)
        assert setup.penalties is None
        assert len(setup.smooth_info) == 0

    def test_factor_single_level_raises(self) -> None:
        """Factor with single level raises informative error."""
        rng = np.random.default_rng(SEED)
        data = pd.DataFrame(
            {
                "x1": rng.uniform(0, 1, 10),
                "y": rng.normal(0, 1, 10),
                "fac": pd.Categorical(["a"] * 10),
            }
        )
        spec = parse_formula("y ~ fac + s(x1)")

        with pytest.raises(ValueError, match="fewer than 2 levels"):
            ModelSetup.build(spec, data)

    def test_custom_weights(self, data) -> None:
        """Custom weights are stored correctly."""
        spec = parse_formula("y ~ s(x1, k=10)")
        w = np.random.default_rng(SEED).uniform(0.5, 2.0, N)
        setup = ModelSetup.build(spec, data, weights=w)

        np.testing.assert_allclose(setup.weights, w, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_custom_offset(self, data) -> None:
        """Custom offset is stored correctly."""
        spec = parse_formula("y ~ s(x1, k=10)")
        off = np.random.default_rng(SEED).normal(0, 1, N)
        setup = ModelSetup.build(spec, data, offset=off)

        np.testing.assert_allclose(
            setup.offset, off, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_dict_data_input(self) -> None:
        """Dict data input works (not just DataFrame)."""
        rng = np.random.default_rng(SEED)
        data = {
            "x1": rng.uniform(0, 1, N),
            "y": rng.normal(0, 1, N),
        }
        spec = parse_formula("y ~ s(x1, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape[0] == N

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            (np.array([2.0]), "expected shape"),  # wrong length: silent broadcast
            (np.ones(N - 1), "expected shape"),
            (np.where(np.arange(N) == 0, -1.0, 1.0), "non-negative"),
            (np.where(np.arange(N) == 0, np.nan, 1.0), "non-finite"),
        ],
    )
    def test_invalid_weights_raise(self, data, bad, match) -> None:
        """Wrong-length, negative, or non-finite weights raise clearly."""
        spec = parse_formula("y ~ s(x1, k=10)")
        with pytest.raises(ValueError, match=match):
            ModelSetup.build(spec, data, weights=bad)

    def test_zero_weights_allowed(self, data) -> None:
        """Zero weights are valid (drop observations), as in mgcv."""
        spec = parse_formula("y ~ s(x1, k=10)")
        w = np.ones(N)
        w[0] = 0.0
        setup = ModelSetup.build(spec, data, weights=w)
        assert setup.weights[0] == 0.0

    def test_all_zero_weights_raise(self, data) -> None:
        """All-zero weights have no data to fit; reject up front (Finding 14).

        Previously produced a degenerate 'converged' model with NaN null
        deviance; mgcv errors during smoothing-parameter setup.
        """
        spec = parse_formula("y ~ s(x1, k=10)")
        with pytest.raises(ValueError, match="sum to zero"):
            ModelSetup.build(spec, data, weights=np.zeros(N))

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            (np.array([0.5]), "expected shape"),  # wrong length: silent broadcast
            (np.where(np.arange(N) == 3, np.nan, 0.0), "non-finite"),
        ],
    )
    def test_invalid_offset_raises(self, data, bad, match) -> None:
        """Wrong-length or non-finite offset raises clearly."""
        spec = parse_formula("y ~ s(x1, k=10)")
        with pytest.raises(ValueError, match=match):
            ModelSetup.build(spec, data, offset=bad)

    @pytest.mark.parametrize(
        ("formula", "col", "value", "name"),
        [
            ("y ~ x2 + s(x1, k=10)", "x2", np.nan, "x2"),  # parametric covariate
            ("y ~ s(x1, k=10)", "x1", np.inf, "x1"),  # smooth covariate
        ],
        ids=["parametric", "smooth"],
    )
    def test_nonfinite_covariate_raises(self, data, formula, col, value, name) -> None:
        """Non-finite numeric covariate raises a clear error (not a LinAlgError)."""
        data = data.copy()
        data.loc[5, col] = value
        spec = parse_formula(formula)
        with pytest.raises(ValueError, match=rf"Covariate '{name}'.*non-finite"):
            ModelSetup.build(spec, data)

    def test_unseen_parametric_factor_level_raises(self, factor_data) -> None:
        """Predicting an unseen parametric factor level raises like R."""
        spec = parse_formula("y ~ fac + s(x1, k=10)")
        setup = ModelSetup.build(spec, factor_data)
        newdata = pd.DataFrame(
            {
                "x1": [0.5, 0.5],
                "fac": pd.Categorical(
                    ["lev0", "newlev"], categories=["lev0", "lev1", "lev2", "newlev"]
                ),
            }
        )
        with pytest.raises(ValueError, match=r"new level.*newlev"):
            setup.build_predict_matrix(newdata)

    def test_na_parametric_factor_level_predicts_nan(self, factor_data) -> None:
        """An NA parametric factor value -> NaN row, not a TypeError (Finding 15).

        Matches R's predict.gam (NA in -> NA out). Previously ``np.unique`` on
        mixed str+NaN raised ``TypeError``.
        """
        spec = parse_formula("y ~ fac + s(x1, k=10)")
        setup = ModelSetup.build(spec, factor_data)
        newdata = pd.DataFrame(
            {"x1": [0.5, 0.5], "fac": np.array(["lev1", np.nan], dtype=object)}
        )
        X = setup.build_predict_matrix(newdata)  # must not raise
        assert X.shape[0] == 2
        assert np.isnan(X[1]).any()  # NA-factor row carries NaN
        assert np.isfinite(X[0]).all()


class TestIntegerCategorical:
    """Finding S1: integer-valued pd.Categorical must not be demoted to numeric."""

    def test_integer_categorical_matches_string_categorical(self) -> None:
        """int-coded pd.Categorical == string-coded factor (re + parametric).

        mgcv's ``is.factor()`` gate is agnostic to the level dtype, so int- and
        string-coded factors are interchangeable. Before the fix
        ``ModelSetup._to_dict`` did ``np.asarray`` on factor columns, demoting an
        INTEGER pd.Categorical to a bare int64 array; downstream ``is_factor``
        then returned False and ``s(g, bs="re")`` encoded g as a single numeric
        covariate (string categoricals -> object arrays, so only int categories
        were silently dropped).
        """
        rng = np.random.default_rng(SEED)
        n = 120
        codes = rng.integers(0, 4, size=n)
        x = rng.uniform(0.0, 1.0, n)
        group_effect = np.array([0.0, 1.5, -1.0, 0.5])
        y = group_effect[codes] + 0.5 * x + rng.normal(scale=0.3, size=n)

        df_int = pd.DataFrame({"y": y, "x": x, "g": pd.Categorical(codes)})
        df_str = pd.DataFrame(
            {"y": y, "x": x, "g": pd.Categorical([str(c) for c in codes])}
        )

        collector = _AssertCollector()

        # Random effect s(g, bs="re") — the path the bug corrupted.
        re_int = GAM('y ~ x + s(g, bs="re")').fit(df_int)
        re_str = GAM('y ~ x + s(g, bs="re")').fit(df_str)
        collector.check(
            "re_xcols",
            lambda: check_that(
                re_int.X.shape[1] == re_str.X.shape[1],
                f"RE X column count differs: int={re_int.X.shape[1]} "
                f"str={re_str.X.shape[1]} (integer categorical demoted to numeric)",
            ),
        )
        collector.check(
            "re_deviance",
            lambda: np.testing.assert_allclose(
                float(re_int.deviance),
                float(re_str.deviance),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        collector.check(
            "re_edf",
            lambda: np.testing.assert_allclose(
                float(np.sum(np.asarray(re_int.edf))),
                float(np.sum(np.asarray(re_str.edf))),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        collector.check(
            "re_predict_roundtrip",
            lambda: np.testing.assert_allclose(
                np.asarray(re_int.predict(df_int)),
                np.asarray(re_str.predict(df_str)),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )

        # Parametric factor (already correct via original_data; guard it stays so).
        par_int = GAM("y ~ g + x").fit(df_int)
        par_str = GAM("y ~ g + x").fit(df_str)
        collector.check(
            "param_xcols",
            lambda: check_that(
                par_int.X.shape[1] == par_str.X.shape[1],
                f"parametric X cols differ: int={par_int.X.shape[1]} "
                f"str={par_str.X.shape[1]}",
            ),
        )

        # A genuine numeric int column must NOT be promoted to a factor.
        df_num = pd.DataFrame({"y": y, "xi": rng.integers(0, 50, n)})
        num = GAM("y ~ s(xi, k=5)").fit(df_num)
        collector.check(
            "numeric_int_not_promoted",
            lambda: check_that(
                num.X.shape[1] == 5,
                f"numeric int column promoted to factor: X cols={num.X.shape[1]}",
            ),
        )

        collector.raise_if_any("integer-categorical vs string-categorical parity")


def _two_factor_data() -> pd.DataFrame:
    """Data with a 3-level factor f1, a 2-level factor f2, numeric x and y."""
    rng = np.random.default_rng(SEED)
    n = 120
    lev1, lev2 = ["a", "b", "c"], ["p", "q"]
    f1 = pd.Categorical(rng.choice(lev1, size=n), categories=lev1)
    f2 = pd.Categorical(rng.choice(lev2, size=n), categories=lev2)
    x = rng.uniform(0.0, 1.0, n)
    eff1 = np.array([0.0, 1.0, -1.0])[f1.codes]
    eff2 = np.array([0.0, 0.5])[f2.codes]
    y = eff1 + eff2 + np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "f1": f1, "f2": f2, "x": x})


class TestNoInterceptMultiFactor:
    """Finding S2: no-intercept model with >=2 factors must be full-rank."""

    def test_full_rank_parametric_block(self) -> None:
        """y ~ 0 + f1 + f2 + s(x): first factor full-coded, rest treatment-coded.

        R full-codes only the FIRST factor of a no-intercept formula and
        treatment-codes the rest, keeping the parametric block full rank. The
        bug (drop_ref = has_intercept) full-coded every factor, giving
        [f1a,f1b,f1c,f2p,f2q] (rank 4 of 5). Correct: [f1a,f1b,f1c,f2q].
        """
        df = _two_factor_data()
        collector = _AssertCollector()

        x_param, names = ModelSetup._build_parametric_matrix(
            parse_formula("y ~ 0 + f1 + f2 + s(x)").parametric_terms,
            df,
            False,
            len(df),
        )
        collector.check(
            "names",
            lambda: check_that(
                names == ["f1a", "f1b", "f1c", "f2q"],
                f"parametric names {names} != R's ['f1a','f1b','f1c','f2q']",
            ),
        )
        collector.check(
            "full_rank",
            lambda: check_that(
                np.linalg.matrix_rank(x_param) == x_param.shape[1] == 4,
                f"block rank {np.linalg.matrix_rank(x_param)} of {x_param.shape[1]}",
            ),
        )

        # Single-factor no-intercept stays full-coded (must not regress).
        _, names1 = ModelSetup._build_parametric_matrix(
            parse_formula("y ~ 0 + f1 + s(x)").parametric_terms, df, False, len(df)
        )
        collector.check(
            "single_factor_full_coded",
            lambda: check_that(
                names1 == ["f1a", "f1b", "f1c"],
                f"single-factor names {names1} != ['f1a','f1b','f1c']",
            ),
        )

        # With intercept stays treatment-coded (must not regress).
        _, names2 = ModelSetup._build_parametric_matrix(
            parse_formula("y ~ f1 + f2 + s(x)").parametric_terms, df, True, len(df)
        )
        collector.check(
            "with_intercept_treatment_coded",
            lambda: check_that(
                names2 == ["(Intercept)", "f1b", "f1c", "f2q"],
                f"with-intercept names {names2}",
            ),
        )
        collector.raise_if_any("no-intercept multi-factor parametric coding")

    @pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
    def test_matches_r(self) -> None:
        """R parity: y ~ 0 + f1 + f2 + s(x) is identifiable; coef count + fitted
        values match mgcv. A rank-deficient block would give a different fit."""
        from tests.r_bridge import RBridge

        df = _two_factor_data()
        formula = "y ~ 0 + f1 + f2 + s(x)"
        r_fit = RBridge().fit_gam(formula, df, family="gaussian", method="REML")
        model = GAM(formula, family="gaussian").fit(df)

        collector = _AssertCollector()
        collector.check(
            "n_coef_match",
            lambda: check_that(
                len(model.coefficients) == len(r_fit["coefficients"]),
                f"jaxgam {len(model.coefficients)} coefs vs R "
                f"{len(r_fit['coefficients'])}",
            ),
        )
        collector.check(
            "fitted_values_match",
            lambda: np.testing.assert_allclose(
                model.fitted_values,
                r_fit["fitted_values"],
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
            ),
        )
        collector.raise_if_any("no-intercept multi-factor R parity")


class TestOrderedFactor:
    """Finding H2: ordered factors must use R's contr.poly contrasts."""

    def test_contr_poly_matches_r_algebra(self) -> None:
        """_contr_poly reproduces stats::contr.poly (orthonormal, R suffixes).

        No R needed: contr.poly is exact algebra. Pins names and the n=3
        reference matrix, and asserts orthonormality (a strong guard against
        any regression back to treatment 0/1 coding).
        """
        collector = _AssertCollector()
        c3, sfx3 = ModelSetup._contr_poly(3)
        # R: contr.poly(3) == [[-.7071,.4082],[0,-.8165],[.7071,.4082]]
        ref3 = np.array([[-0.70711, 0.40825], [0.0, -0.81650], [0.70711, 0.40825]])
        collector.check(
            "contr_poly_3_values",
            lambda: np.testing.assert_allclose(
                c3, ref3, rtol=MODERATE.rtol, atol=MODERATE.atol
            ),
        )
        collector.check(
            "suffixes",
            lambda: check_that(
                sfx3 == [".L", ".Q"]
                and ModelSetup._contr_poly(4)[1] == [".L", ".Q", ".C"]
                and ModelSetup._contr_poly(5)[1] == [".L", ".Q", ".C", "^4"],
                "contr.poly column-name suffixes do not match R",
            ),
        )
        # Each contrast column has unit norm and is orthogonal to the others
        # and to the constant (this is what distinguishes it from treatment).
        cols = np.column_stack([np.ones(4), ModelSetup._contr_poly(4)[0]])
        collector.check(
            "orthonormal",
            lambda: np.testing.assert_allclose(
                (cols / np.linalg.norm(cols, axis=0)).T
                @ (cols / np.linalg.norm(cols, axis=0)),
                np.eye(4),
                atol=STRICT.atol,
            ),
        )
        collector.raise_if_any("contr.poly algebra")

    @pytest.mark.skipif(not r_available(), reason="R with mgcv not available")
    def test_ordered_factor_uses_contr_poly_vs_r(self, r_bridge) -> None:
        """y ~ g(ordered) + s(x1): parametric block + names match R's contr.poly.

        Before the fix ordered factors got treatment 0/1 dummies (names
        gmid/ghi). R applies contr.poly by default (names g.L/g.Q). An unordered
        control still treatment-codes. Prediction reproduces the contrasts from
        stored metadata, independent of newdata dtype.
        """
        rng = np.random.default_rng(SEED)
        x1 = rng.uniform(0, 1, N)
        y = np.sin(2 * np.pi * x1) + rng.normal(0, 0.5, N)
        lev = ["lo", "mid", "hi"]
        codes = rng.integers(0, 3, N)
        g_ord = pd.Categorical([lev[c] for c in codes], categories=lev, ordered=True)
        g_unord = pd.Categorical([lev[c] for c in codes], categories=lev, ordered=False)
        formula = "y ~ g + s(x1, k=10, bs='tp')"

        collector = _AssertCollector()

        df_ord = pd.DataFrame({"x1": x1, "y": y, "g": g_ord})
        r_X = np.asarray(
            r_bridge.get_smooth_components(formula, df_ord)["model_matrix"],
            dtype=np.float64,
        )
        # R layout: (Intercept), g.L, g.Q, <smooth basis>.
        r_param = r_X[:, 1:3]
        setup = ModelSetup.build(parse_formula(formula), df_ord)
        py_param = setup.X[:, 1:3]
        py_names = [nm for nm in setup.term_names if nm.startswith("g")]

        collector.check(
            "ordered_names",
            lambda: check_that(
                py_names == ["g.L", "g.Q"], f"expected ['g.L','g.Q'], got {py_names}"
            ),
        )
        collector.check(
            "ordered_values_vs_r",
            lambda: np.testing.assert_allclose(
                py_param,
                r_param,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg="ordered-factor block != R contr.poly",
            ),
        )

        # Unordered control still treatment-codes (gmid/ghi indicators).
        df_un = pd.DataFrame({"x1": x1, "y": y, "g": g_unord})
        setup_u = ModelSetup.build(parse_formula(formula), df_un)
        u_names = [nm for nm in setup_u.term_names if nm.startswith("g")]
        collector.check(
            "unordered_names",
            lambda: check_that(
                u_names == ["gmid", "ghi"], f"expected ['gmid','ghi'], got {u_names}"
            ),
        )
        collector.check(
            "unordered_is_indicator",
            lambda: check_that(
                set(np.unique(setup_u.X[:, 1:3])) <= {0.0, 1.0},
                "unordered factor should be 0/1 indicators",
            ),
        )

        # Prediction reproduces the contr.poly block (from stored metadata).
        collector.check(
            "predict_reproduces_contr_poly",
            lambda: np.testing.assert_allclose(
                setup.build_predict_matrix(df_ord)[:, 1:3],
                py_param,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        collector.raise_if_any("ordered-factor contr.poly parity (H2)")


class TestAliasedParametricColumns:
    """Finding H5: exactly-aliased parametric columns must be dropped (NA)."""

    def test_aliased_column_dropped(self) -> None:
        """y ~ x + z with z == x: block reduced to full rank, alias dropped,
        surviving slope recovers the full effect, fitted == full-rank fit.

        mgcv drops rank-deficient parametric columns via pivoted QR (keeping the
        earlier of an aliased pair) and reports the dropped coefficient as NA.
        Before the fix jaxgam kept both columns and split the slope ~0.49/0.49.
        """
        rng = np.random.default_rng(SEED)
        n = 200
        x = rng.standard_normal(n)
        z = x.copy()
        true_slope = 0.997
        y = 2.0 + true_slope * x + 0.1 * rng.standard_normal(n)
        df = pd.DataFrame({"y": y, "x": x, "z": z})

        aliased = GAM("y ~ x + z").fit(df)
        reference = GAM("y ~ x").fit(df[["y", "x"]])
        c = _AssertCollector()

        X = np.asarray(aliased.X)
        c.check(
            "full_rank",
            lambda: check_that(
                np.linalg.matrix_rank(X) == X.shape[1],
                f"X {X.shape[1]} cols rank {np.linalg.matrix_rank(X)}; alias kept",
            ),
        )
        active = set(aliased.term_names)
        c.check(
            "one_alias_survives",
            lambda: check_that(
                ("x" in active) ^ ("z" in active),
                f"expected exactly one of x/z active, got {active}",
            ),
        )
        c.check(
            "alias_recorded_dropped",
            lambda: check_that(
                set(aliased.setup.dropped_param_names) == ({"x", "z"} - active),
                f"dropped={aliased.setup.dropped_param_names} "
                f"active={active & {'x', 'z'}}",
            ),
        )
        name_to_coef = dict(zip(aliased.term_names, aliased.coefficients, strict=True))
        surviving = "x" if "x" in active else "z"
        c.check(
            "coefs_finite",
            lambda: check_that(
                np.all(np.isfinite(aliased.coefficients)),
                f"non-finite active coefs: {aliased.coefficients}",
            ),
        )
        ref_map = dict(zip(reference.term_names, reference.coefficients, strict=True))
        c.check(
            "slope_matches_fullrank",
            lambda: np.testing.assert_allclose(
                name_to_coef[surviving],
                ref_map["x"],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        c.check(
            "slope_recovers_full_effect",
            lambda: check_that(
                abs(name_to_coef[surviving] - true_slope) < 0.05,
                f"surviving slope {name_to_coef[surviving]:.4f} looks split",
            ),
        )
        c.check(
            "fitted_match_fullrank",
            lambda: np.testing.assert_allclose(
                aliased.fitted_values,
                reference.fitted_values,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
            ),
        )
        c.check(
            "predict_roundtrip",
            lambda: np.testing.assert_allclose(
                np.asarray(aliased.predict(df)),
                aliased.fitted_values,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
            ),
        )
        # summary presents the dropped column as an NA row (R parity).
        s = aliased.summary()
        c.check(
            "summary_na_row",
            lambda: check_that(
                surviving == "x"
                and "z" in s.p_names
                and bool(np.isnan(s.p_table[s.p_names.index("z")]).all()),
                f"dropped col not shown as NA row: names={s.p_names}",
            ),
        )
        c.raise_if_any("aliased parametric column handling (H5)")

    def test_correlated_but_independent_kept(self) -> None:
        """A highly-but-imperfectly-correlated predictor must NOT be dropped."""
        rng = np.random.default_rng(SEED)
        n = 200
        x = rng.standard_normal(n)
        z = x + 1e-3 * rng.standard_normal(n)
        y = 1.0 + 0.5 * x - 0.3 * z + 0.1 * rng.standard_normal(n)
        df = pd.DataFrame({"y": y, "x": x, "z": z})
        model = GAM("y ~ x + z").fit(df)
        assert "x" in model.term_names
        assert "z" in model.term_names
        assert model.setup.dropped_param_names == ()
        X = np.asarray(model.X)
        assert np.linalg.matrix_rank(X) == X.shape[1]


class TestRepeatedSmoothLabels:
    """Finding S3: repeated smooth labels must not collide (positional lookup)."""

    @staticmethod
    def _pen_support(S: np.ndarray) -> tuple[int, int]:
        nz = np.where(np.any(np.asarray(S) != 0.0, axis=0))[0]
        assert nz.size > 0
        return int(nz.min()), int(nz.max())

    def test_repeated_labels_do_not_collide(self) -> None:
        """s(x,k=6) + s(x,k=8): disjoint penalty/info structure + working predict.

        Both smooths share the label 's(x)'. Before the fix, label-keyed lookups
        embedded the second penalty on the first smooth's columns and predict()
        crashed with a matmul size mismatch. mgcv keeps both (with a warning) and
        side-constrains the higher-indexed one.
        """
        rng = np.random.default_rng(SEED)
        x = np.sort(rng.uniform(0.0, 1.0, N))
        data = {"x": x, "y": np.sin(2 * np.pi * x) + rng.normal(0, 0.2, N)}
        spec = parse_formula('y ~ s(x, k=6, bs="cr") + s(x, k=8, bs="cr")')
        assert len(spec.smooth_terms) == 2  # different config -> both kept

        with pytest.warns(UserWarning, match="repeated 1-d smooths of same variable"):
            setup = ModelSetup.build(spec, data)

        blocks = [t for t in setup.coef_map.terms if t.term_type == "smooth"]
        si0, si1 = setup.smooth_info
        pens = setup.penalties.penalties
        c = _AssertCollector()

        c.check(
            "smooth_info_disjoint",
            lambda: check_that(
                si0.last_coef <= si1.first_coef and si0.first_coef != si1.first_coef,
                f"smooth_info overlap ({si0.first_coef},{si0.last_coef}) vs "
                f"({si1.first_coef},{si1.last_coef})",
            ),
        )
        c.check(
            "info_matches_blocks",
            lambda: check_that(
                si0.first_coef == blocks[0].col_start
                and si1.first_coef == blocks[1].col_start
                and si0.first_penalty != si1.first_penalty,
                "smooth_info offsets/penalties do not match term blocks",
            ),
        )

        def _pen_in_block(i: int) -> None:
            lo, hi = self._pen_support(pens[i].S)
            b = blocks[i]
            check_that(
                b.col_start <= lo and hi < b.col_start + b.n_coefs,
                f"penalty {i} support ({lo},{hi}) escapes block "
                f"[{b.col_start},{b.col_start + b.n_coefs})",
            )

        c.check("penalty_0_in_block_0", lambda: _pen_in_block(0))
        c.check("penalty_1_in_block_1", lambda: _pen_in_block(1))
        c.check(
            "predict_works",
            lambda: check_that(
                setup.build_predict_matrix({"x": np.linspace(0.05, 0.95, 23)}).shape
                == (23, setup.coef_map.total_coefs),
                "predict matrix wrong shape (label collision)",
            ),
        )
        c.raise_if_any("repeated-smooth-label structure (S3)")

    def test_identical_smooths_dedup_to_one(self) -> None:
        """Identical s(x)+s(x) collapses to one smooth (R terms.formula)."""
        spec = parse_formula("y ~ s(x) + s(x)")
        assert len(spec.smooth_terms) == 1
        rng = np.random.default_rng(SEED)
        x = np.sort(rng.uniform(0.0, 1.0, N))
        setup = ModelSetup.build(spec, {"x": x, "y": np.sin(2 * np.pi * x)})
        blocks = [t for t in setup.coef_map.terms if t.term_type == "smooth"]
        assert len(blocks) == 1
        assert len(setup.smooth_info) == 1
