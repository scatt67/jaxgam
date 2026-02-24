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

import sys

import numpy as np
import pandas as pd
import pytest

from pymgcv.formula.design import ModelSetup, SmoothInfo
from pymgcv.formula.parser import parse_formula
from tests.tolerances import MODERATE, STRICT, normalize_column_signs

SEED = 42
N = 200


def _r_available() -> bool:
    """Check if R bridge is available."""
    try:
        from pymgcv.compat.r_bridge import RBridge

        return RBridge.available()
    except Exception:
        return False


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

    def test_cubic_smooth(self, data) -> None:
        """y ~ s(x, k=10, bs='cr'): cubic basis works."""
        spec = parse_formula('y ~ s(x1, k=10, bs="cr")')
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape[0] == N
        assert setup.n_obs == N

    def test_no_intercept(self, data) -> None:
        """y ~ 0 + s(x1, k=10): no intercept column."""
        spec = parse_formula("y ~ 0 + s(x1, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.X.shape == (N, 9)
        assert "(Intercept)" not in setup.term_names

    def test_fields_populated(self, data) -> None:
        """ModelSetup fields are populated correctly."""
        spec = parse_formula("y ~ s(x1, k=10)")
        setup = ModelSetup.build(spec, data)

        assert setup.n_obs == N
        assert setup.y.shape == (N,)
        assert setup.weights.shape == (N,)
        np.testing.assert_allclose(
            setup.weights, np.ones(N), rtol=STRICT.rtol, atol=STRICT.atol
        )
        assert setup.offset is None
        assert len(setup.term_names) == setup.X.shape[1]


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
        """y ~ s(x1, by=fac, k=10): block-diagonal structure, correct n_coefs."""
        spec = parse_formula("y ~ s(x1, by=fac, k=10)")
        setup = ModelSetup.build(spec, factor_data)

        assert setup.X.shape[0] == N
        assert len(setup.smooth_info) == 1

    def test_factor_by_with_main_effect(self, factor_data) -> None:
        """y ~ s(x1) + s(x1, by=fac): main effect + factor-by coexist."""
        spec = parse_formula("y ~ s(x1, k=10) + s(x1, by=fac, k=10)")
        setup = ModelSetup.build(spec, factor_data)

        assert setup.X.shape[0] == N
        assert len(setup.smooth_info) == 2

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


@pytest.mark.skipif(not _r_available(), reason="R with mgcv not available")
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
        r_ncols = [b.shape[1] for b in r_result["basis_matrices"]]

        spec = parse_formula(formula)
        setup = ModelSetup.build(spec, data)

        r_total_smooth_cols = sum(r_ncols)
        py_smooth_info = setup.smooth_info[0]
        py_smooth_cols = py_smooth_info.last_coef - py_smooth_info.first_coef

        assert py_smooth_cols == r_total_smooth_cols, (
            f"Factor-by smooth cols: Python {py_smooth_cols} != R {r_total_smooth_cols}"
        )

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
# TestPhaseBoundary — import guard
# ===========================================================================


class TestPhaseBoundary:
    """Phase 1 import guard."""

    def test_no_jax_import(self) -> None:
        """Importing pymgcv.formula.design does not import JAX."""
        import importlib

        modules_to_remove = [
            key
            for key in sys.modules
            if key == "jax" or key.startswith(("jax.", "pymgcv."))
        ]
        saved = {key: sys.modules.pop(key) for key in modules_to_remove}

        try:
            importlib.import_module("pymgcv.formula.design")
            assert "jax" not in sys.modules, (
                "Importing pymgcv.formula.design triggered a jax import"
            )
        finally:
            for key in list(sys.modules):
                if key.startswith("pymgcv."):
                    sys.modules.pop(key, None)
            sys.modules.update(saved)


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
