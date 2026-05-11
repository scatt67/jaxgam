"""Tests for dense random effects smooth (bs="re").

Validates RandomEffectSmooth from jaxgam.smooths.random_effects:
- Structural tests: shape, rank, null_space_dim, flags
- Penalty tests: symmetric PSD, identity (pre-normalization), full rank
- Factor handling: single factor, factor x factor, numeric x factor, numeric only
- Prediction: unseen levels -> zero rows, predict reproduces design matrix
- R comparison tests (skip if R unavailable)

Design doc reference: docs/dense_random_effects/design.md
R source reference: R/smooth.r smooth.construct.re.smooth.spec()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.api import GAM
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.smooths.random_effects import RandomEffectSmooth
from jaxgam.smooths.registry import get_smooth_class
from tests.helpers import SEED, make_smooth_spec, r_available
from tests.r_bridge import RBridge
from tests.tolerances import MODERATE, STRICT

# ===========================================================================
# 1. Structural tests
# ===========================================================================


class TestStructural:
    """Structural properties of RandomEffectSmooth."""

    def test_rank_equals_n_coefs(self, re_factor_data) -> None:
        """RE smooth has rank = n_coefs (full rank)."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        assert smooth.rank == smooth.n_coefs

    def test_n_coefs_equals_n_levels(self, re_factor_data) -> None:
        """Single factor: n_coefs = number of factor levels."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        n_levels = len(pd.Categorical(re_factor_data["g"]).categories)
        assert smooth.n_coefs == n_levels

    def test_X_shape(self, re_factor_data) -> None:
        """Basis matrix shape = (n, n_levels)."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        X = smooth.build_design_matrix(re_factor_data)
        n = len(re_factor_data["g"])
        assert X.shape == (n, smooth.n_coefs)

    def test_k_argument_ignored(self, re_factor_data) -> None:
        """The k argument is ignored for RE smooths."""
        spec = make_smooth_spec(["g"], bs="re", k=5)
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        n_levels = len(pd.Categorical(re_factor_data["g"]).categories)
        assert smooth.n_coefs == n_levels

    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("build_design_matrix", ({"g": pd.Series(pd.Categorical(["a", "b"]))},)),
            ("build_penalty_matrices", ()),
            ("predict_matrix", ({"g": pd.Series(pd.Categorical(["a", "b"]))},)),
        ],
    )
    def test_setup_required(self, method_name: str, args: tuple) -> None:
        """Matrix and penalty methods require setup."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            getattr(smooth, method_name)(*args)


# ===========================================================================
# 2. Penalty tests
# ===========================================================================


class TestPenalty:
    """Tests for RE penalty construction."""

    def test_penalty_symmetric(self, re_factor_data) -> None:
        """RE penalty S is symmetric."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        S = smooth.build_penalty_matrices()[0].S
        np.testing.assert_allclose(S, S.T, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_penalty_psd(self, re_factor_data) -> None:
        """RE penalty S is positive semi-definite."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        S = smooth.build_penalty_matrices()[0].S
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -STRICT.atol), (
            f"RE S has negative eigenvalue: {np.min(eigvals):.2e}"
        )

    def test_penalty_proportional_to_identity(self, re_factor_data) -> None:
        """RE penalty is a scalar multiple of the identity (post-normalization)."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        S = smooth.build_penalty_matrices()[0].S
        k = S.shape[0]
        # S should be c * I for some constant c
        diag_val = S[0, 0]
        expected = diag_val * np.eye(k)
        np.testing.assert_allclose(S, expected, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_penalty_full_rank(self, re_factor_data) -> None:
        """RE penalty has full rank and null_space_dim = 0."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        penalty = smooth.build_penalty_matrices()[0]
        assert penalty.rank == smooth.n_coefs
        assert penalty.null_space_dim == 0


# ===========================================================================
# 3. Factor handling tests
# ===========================================================================


class TestFactorHandling:
    """Tests for model matrix construction with various variable types."""

    def test_single_factor_one_hot(self, re_factor_data) -> None:
        """Single factor produces correct one-hot indicator matrix."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        X = smooth.build_design_matrix(re_factor_data)
        g = re_factor_data["g"]
        levels = list(pd.Categorical(g).categories)

        # Each row should have exactly one 1.0 and rest 0.0
        assert np.all(X.sum(axis=1) == 1.0)
        assert np.all((X == 0.0) | (X == 1.0))

        # Check indicator correctness
        g_arr = np.asarray(g)
        for i in range(len(g_arr)):
            j = levels.index(g_arr[i])
            assert X[i, j] == 1.0

    def test_two_factor_interaction(self, re_two_factor_data) -> None:
        """Factor x factor produces correct interaction indicator."""
        spec = make_smooth_spec(["g1", "g2"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_two_factor_data)

        X = smooth.build_design_matrix(re_two_factor_data)

        g1 = re_two_factor_data["g1"]
        g2 = re_two_factor_data["g2"]
        levels_g1 = list(pd.Categorical(g1).categories)
        levels_g2 = list(pd.Categorical(g2).categories)

        # n_coefs = L1 * L2
        assert smooth.n_coefs == len(levels_g1) * len(levels_g2)

        # Each row should have exactly one 1.0
        assert np.all(X.sum(axis=1) == 1.0)
        assert np.all((X == 0.0) | (X == 1.0))

        # Check column ordering: g1 varies fastest (R's model.matrix convention)
        g1_arr = np.asarray(g1)
        g2_arr = np.asarray(g2)
        n_g1 = len(levels_g1)
        for i in range(len(g1_arr)):
            j1 = levels_g1.index(g1_arr[i])
            j2 = levels_g2.index(g2_arr[i])
            expected_col = j2 * n_g1 + j1
            assert X[i, expected_col] == 1.0

    def test_numeric_factor_interaction(self, re_numeric_factor_data) -> None:
        """Numeric x factor produces correct weighted indicator."""
        spec = make_smooth_spec(["x", "g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_numeric_factor_data)

        X = smooth.build_design_matrix(re_numeric_factor_data)

        x = re_numeric_factor_data["x"]
        g = re_numeric_factor_data["g"]
        levels = list(pd.Categorical(g).categories)

        # n_coefs = number of factor levels (numeric contributes 1 column)
        assert smooth.n_coefs == len(levels)

        # Each row should have exactly one nonzero = x[i]
        g_arr = np.asarray(g)
        x_arr = np.asarray(x)
        for i in range(len(g_arr)):
            j = levels.index(g_arr[i])
            np.testing.assert_allclose(
                X[i, j], x_arr[i], rtol=STRICT.rtol, atol=STRICT.atol
            )
            # Other columns are zero
            mask = np.ones(smooth.n_coefs, dtype=bool)
            mask[j] = False
            np.testing.assert_allclose(X[i, mask], 0.0, atol=STRICT.atol)

    @pytest.mark.parametrize("n_variables", [1, 2])
    def test_numeric_terms(self, n_variables: int) -> None:
        """Numeric-only terms produce one column with the product of inputs."""
        rng = np.random.default_rng(SEED)
        data = {f"x{i + 1}": rng.uniform(0, 1, 50) for i in range(n_variables)}
        variables = list(data)
        expected = np.prod(np.column_stack([data[var] for var in variables]), axis=1)

        spec = make_smooth_spec(variables, bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert smooth.n_coefs == 1
        assert X.shape == (50, 1)
        np.testing.assert_allclose(
            X[:, 0], expected, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_factor_with_many_levels(self) -> None:
        """Factor with many levels works correctly."""
        rng = np.random.default_rng(SEED)
        n_levels = 100
        n = 500
        g = rng.choice([f"lev{i}" for i in range(n_levels)], size=n)
        data = {"g": pd.Series(pd.Categorical(g))}

        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(data)

        assert smooth.n_coefs == n_levels
        X = smooth.build_design_matrix(data)
        assert X.shape == (n, n_levels)
        assert np.all(X.sum(axis=1) == 1.0)


# ===========================================================================
# 4. Prediction tests
# ===========================================================================


class TestPrediction:
    """Tests for prediction matrix construction."""

    @pytest.mark.parametrize(
        ("fixture_name", "variables"),
        [
            ("re_factor_data", ["g"]),
            ("re_two_factor_data", ["g1", "g2"]),
            ("re_numeric_factor_data", ["x", "g"]),
        ],
    )
    def test_predict_reproduces_design_matrix(
        self, request, fixture_name: str, variables: list[str]
    ) -> None:
        """predict_matrix reproduces build_design_matrix on training data."""
        data = request.getfixturevalue(fixture_name)
        spec = make_smooth_spec(variables, bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(data)

        X_design = smooth.build_design_matrix(data)
        X_predict = smooth.predict_matrix(data)
        np.testing.assert_allclose(
            X_predict, X_design, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_unseen_levels_produce_zero_rows(self, re_factor_data) -> None:
        """Unseen factor levels produce zero rows in prediction matrix."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        # Predict with entirely unseen levels
        new_data = {"g": pd.Series(pd.Categorical(["unseen_a", "unseen_b"]))}
        X_pred = smooth.predict_matrix(new_data)

        assert X_pred.shape == (2, smooth.n_coefs)
        np.testing.assert_allclose(X_pred, 0.0, atol=STRICT.atol)

    def test_mixed_seen_unseen_levels(self, re_factor_data) -> None:
        """Mix of seen and unseen levels: seen rows correct, unseen rows zero."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        levels = list(pd.Categorical(re_factor_data["g"]).categories)
        seen_level = levels[0]
        new_data = {
            "g": pd.Series(pd.Categorical([seen_level, "never_seen", seen_level]))
        }
        X_pred = smooth.predict_matrix(new_data)

        # Row 0 and 2: seen level -> indicator for that level
        j = levels.index(seen_level)
        assert X_pred[0, j] == 1.0
        assert X_pred[2, j] == 1.0

        # Row 1: unseen -> all zeros
        np.testing.assert_allclose(X_pred[1, :], 0.0, atol=STRICT.atol)

    def test_subset_of_training_levels(self, re_factor_data) -> None:
        """Prediction with a subset of training levels works correctly."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        levels = list(pd.Categorical(re_factor_data["g"]).categories)
        # Use only first 3 levels
        subset = levels[:3]
        new_data = {"g": pd.Series(pd.Categorical(subset * 5))}
        X_pred = smooth.predict_matrix(new_data)

        assert X_pred.shape == (15, smooth.n_coefs)
        # Each row should have exactly one 1.0
        assert np.all(X_pred.sum(axis=1) == 1.0)

    def test_single_unseen_level(self, re_factor_data) -> None:
        """Single unseen level contributes zero."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        new_data = {"g": pd.Series(pd.Categorical(["totally_new_level"]))}
        X_pred = smooth.predict_matrix(new_data)

        assert X_pred.shape == (1, smooth.n_coefs)
        np.testing.assert_allclose(X_pred, 0.0, atol=STRICT.atol)

    def test_unseen_level_in_one_variable_of_interaction(self) -> None:
        """Unseen level in one var of a multi-var interaction zeros the row."""
        rng = np.random.default_rng(SEED)
        n = 100
        g1 = rng.choice(["a", "b"], size=n)
        g2 = rng.choice(["x", "y"], size=n)
        train_data = {
            "g1": pd.Series(pd.Categorical(g1)),
            "g2": pd.Series(pd.Categorical(g2)),
        }

        spec = make_smooth_spec(["g1", "g2"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(train_data)

        # g1 known, g2 unseen -> entire row should be zero
        new_data = {
            "g1": pd.Series(pd.Categorical(["a", "b", "a"])),
            "g2": pd.Series(pd.Categorical(["x", "unseen", "unseen"])),
        }
        X_pred = smooth.predict_matrix(new_data)

        # Row 0: both known -> exactly one 1.0
        assert X_pred[0].sum() == 1.0
        # Row 1: g2 unseen -> all zeros
        np.testing.assert_allclose(X_pred[1, :], 0.0, atol=STRICT.atol)
        # Row 2: g2 unseen -> all zeros
        np.testing.assert_allclose(X_pred[2, :], 0.0, atol=STRICT.atol)


# ===========================================================================
# 5. R basis comparison tests (skip if R unavailable)
# ===========================================================================


@pytest.mark.skipif(not r_available(), reason="R with mgcv not available")
class TestRComparison:
    """Compare RE smooth construction against R's smoothCon().

    RE smooths are fully deterministic (no eigendecomposition), so
    basis matrix X and penalty matrix S should match R at machine
    precision. All tests use STRICT tolerance.
    """

    @pytest.fixture(params=["single_factor", "two_factor", "numeric_factor"])
    def r_case(self, request) -> tuple:
        """Shared R/Python smooth setup across RE basis types."""
        case = request.param
        rng = np.random.default_rng(SEED)
        n = 100

        if case == "single_factor":
            n_groups = 20
            g = rng.choice([f"g{i}" for i in range(n_groups)], size=n)
            data = pd.DataFrame({"g": pd.Categorical(g)})
            formula = "s(g, bs='re')"
            spec = make_smooth_spec(["g"], bs="re")
            setup_data = {"g": pd.Series(pd.Categorical(g))}
            matrix_data = {"g": data["g"]}
        elif case == "two_factor":
            g1 = rng.choice(["a", "b", "c"], size=n)
            g2 = rng.choice(["x", "y"], size=n)
            data = pd.DataFrame(
                {
                    "g1": pd.Categorical(g1),
                    "g2": pd.Categorical(g2),
                }
            )
            formula = "s(g1, g2, bs='re')"
            spec = make_smooth_spec(["g1", "g2"], bs="re")
            setup_data = {
                "g1": pd.Series(pd.Categorical(g1)),
                "g2": pd.Series(pd.Categorical(g2)),
            }
            matrix_data = {
                "g1": data["g1"],
                "g2": data["g2"],
            }
        else:
            x = rng.uniform(0, 1, n)
            g = rng.choice([f"g{i}" for i in range(10)], size=n)
            data = pd.DataFrame(
                {
                    "x": x,
                    "g": pd.Categorical(g),
                }
            )
            formula = "s(x, g, bs='re')"
            spec = make_smooth_spec(["x", "g"], bs="re")
            setup_data = {"x": x, "g": pd.Series(pd.Categorical(g))}
            matrix_data = {
                "x": data["x"].values,
                "g": data["g"],
            }

        smooth = RandomEffectSmooth(spec)
        smooth.setup(setup_data)

        r_result = RBridge().smooth_construct(formula, data)
        return case, smooth, r_result, matrix_data

    def test_X_vs_r(self, r_case) -> None:
        """RE basis matrix X matches R across factor/numeric cases."""
        case, smooth, r_result, matrix_data = r_case
        X_py = smooth.build_design_matrix(matrix_data)
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py,
            X_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"RE {case} X differs from R",
        )

    def test_S_vs_r(self, r_case) -> None:
        """RE penalty matrix S matches R across factor/numeric cases."""
        case, smooth, r_result, _matrix_data = r_case
        S_py = smooth.build_penalty_matrices()[0].S
        S_r = r_result["S"][0]

        np.testing.assert_allclose(
            S_py,
            S_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"RE {case} S differs from R",
        )

    def test_rank_vs_r(self, r_case) -> None:
        """RE rank and null_space_dim match R across cases."""
        _case, smooth, r_result, _matrix_data = r_case
        assert smooth.rank == r_result["rank"]
        assert smooth.null_space_dim == r_result["null_space_dim"]


# ===========================================================================
# 6. Integration tests (registry + constraint pipeline)
# ===========================================================================


class TestRegistryIntegration:
    """Tests that bs='re' is wired into the GAM pipeline."""

    def test_registry_has_re(self) -> None:
        """'re' is registered in the smooth registry."""
        cls = get_smooth_class("re")
        assert cls is RandomEffectSmooth

    def test_model_setup_re_only(self, re_model_data) -> None:
        """ModelSetup.build() succeeds with RE-only formula."""
        spec = parse_formula("y ~ s(g, bs='re')")
        setup = ModelSetup.build(spec, re_model_data)

        n_groups = len(re_model_data["g"].cat.categories)
        # Model matrix: intercept (1) + RE (n_groups)
        assert setup.X.shape == (len(re_model_data), 1 + n_groups)
        assert setup.coef_map.total_coefs == 1 + n_groups

    def test_re_no_centering_applied(self, re_model_data) -> None:
        """RE smooth has Z_centering = None (centering skipped)."""
        spec = parse_formula("y ~ s(g, bs='re')")
        setup = ModelSetup.build(spec, re_model_data)

        re_term = setup.coef_map.get_term("s(g)")
        assert re_term.Z_centering is None

    def test_re_no_gam_side_deletion(self, re_model_data) -> None:
        """RE smooth skipped by gam_side even when sharing a variable.

        s(x) and s(x, g, bs='re') share variable x. gam_side would
        normally constrain the higher-dim smooth, but RE's
        side_constrain=False opts it out.
        """
        spec = parse_formula("y ~ s(x) + s(x, g, bs='re')")
        setup = ModelSetup.build(spec, re_model_data)

        re_term = setup.coef_map.get_term("s(x,g)")
        assert re_term.del_index == ()

    def test_re_n_coefs_no_centering_loss(self, re_model_data) -> None:
        """RE smooth keeps all columns (no column lost to centering)."""
        spec = parse_formula("y ~ s(g, bs='re')")
        setup = ModelSetup.build(spec, re_model_data)

        re_term = setup.coef_map.get_term("s(g)")
        n_groups = len(re_model_data["g"].cat.categories)
        assert re_term.n_coefs == re_term.n_coefs_raw
        assert re_term.n_coefs == n_groups

    def test_re_plus_standard_smooth(self, re_model_data) -> None:
        """RE + standard smooth: both terms present with correct dimensions."""
        spec = parse_formula("y ~ s(x) + s(g, bs='re')")
        setup = ModelSetup.build(spec, re_model_data)

        sx_term = setup.coef_map.get_term("s(x)")
        re_term = setup.coef_map.get_term("s(g)")

        assert sx_term.term_type == "smooth"
        assert re_term.term_type == "smooth"

        # RE keeps all columns, standard smooth gets centering
        n_groups = len(re_model_data["g"].cat.categories)
        assert re_term.n_coefs == n_groups
        assert re_term.Z_centering is None
        assert sx_term.Z_centering is not None

        # Total columns: intercept + s(x) constrained + RE
        expected_total = 1 + sx_term.n_coefs + re_term.n_coefs
        assert setup.X.shape[1] == expected_total

    def test_re_slope_model_setup(self, re_model_data) -> None:
        """ModelSetup with s(x, g, bs='re') random slopes succeeds."""
        spec = parse_formula("y ~ s(x, g, bs='re')")
        setup = ModelSetup.build(spec, re_model_data)

        re_term = setup.coef_map.get_term("s(x,g)")
        n_groups = len(re_model_data["g"].cat.categories)
        assert re_term.n_coefs == n_groups
        assert re_term.Z_centering is None


# ===========================================================================
# 7. End-to-end prediction tests (full predict() pipeline)
# ===========================================================================


class TestEndToEndPrediction:
    """Validate prediction through the full GAM pipeline with unseen levels.

    PR 1 ensures unseen levels zero out at the smooth level. These tests
    confirm the same behavior holds end-to-end through ``ModelSetup
    .build_predict_matrix()`` and ``GAMResults.predict()`` — including
    constraint transforms and column placement.
    """

    def _fit_model(self, data: pd.DataFrame, formula: str = "y ~ s(x) + s(g, bs='re')"):
        return GAM(formula, family="gaussian").fit(data)

    def test_predict_matrix_re_columns_zero_for_unseen(self, re_model_data) -> None:
        """RE term columns are zero for unseen-level rows in the prediction matrix."""
        model = self._fit_model(re_model_data)

        levels = list(re_model_data["g"].cat.categories)
        seen = levels[0]
        new_data = pd.DataFrame(
            {
                "x": [0.5, 0.5, 0.5],
                "g": pd.Categorical(
                    [seen, "totally_unseen", "another_unseen"],
                    categories=[*levels, "totally_unseen", "another_unseen"],
                ),
            }
        )

        X_pred = model.predict_matrix(new_data)
        re_slice = model.setup.coef_map.term_slice("s(g)")
        re_block = X_pred[:, re_slice]

        # Row 0 (seen): exactly one 1.0 in the RE block
        assert re_block[0].sum() == pytest.approx(1.0)
        assert ((re_block[0] == 0.0) | (re_block[0] == 1.0)).all()

        # Rows 1, 2 (unseen): all zeros
        np.testing.assert_allclose(re_block[1], 0.0, atol=STRICT.atol)
        np.testing.assert_allclose(re_block[2], 0.0, atol=STRICT.atol)

    def test_predict_unseen_equals_smooth_only(self, re_model_data) -> None:
        """Prediction with unseen level == prediction with RE coefficients zeroed."""
        model = self._fit_model(re_model_data)

        levels = list(re_model_data["g"].cat.categories)
        # Unseen factor level for prediction
        new_data = pd.DataFrame(
            {
                "x": [0.2, 0.5, 0.8],
                "g": pd.Categorical(["new_a"] * 3, categories=[*levels, "new_a"]),
            }
        )

        # Full prediction (RE columns are zero anyway, so RE coefs don't contribute)
        pred = model.predict(new_data, pred_type="link")

        # Manually compute prediction with RE coefficients forced to zero
        X_pred = model.predict_matrix(new_data)
        coefs = model.coefficients.copy()
        re_slice = model.setup.coef_map.term_slice("s(g)")
        coefs_no_re = coefs.copy()
        coefs_no_re[re_slice] = 0.0
        pred_no_re = X_pred @ coefs_no_re

        # The two should match exactly: RE block is zero for unseen rows,
        # so the RE coefficients contribute zero regardless of their value.
        np.testing.assert_allclose(pred, pred_no_re, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_predict_single_unseen_level(self, re_model_data) -> None:
        """Single-row prediction with unseen level: RE contribution is zero."""
        model = self._fit_model(re_model_data)
        levels = list(re_model_data["g"].cat.categories)

        new_data = pd.DataFrame(
            {
                "x": [0.5],
                "g": pd.Categorical(
                    ["never_in_training"],
                    categories=[*levels, "never_in_training"],
                ),
            }
        )

        X_pred = model.predict_matrix(new_data)
        re_slice = model.setup.coef_map.term_slice("s(g)")
        np.testing.assert_allclose(X_pred[0, re_slice], 0.0, atol=STRICT.atol)

        # Full prediction is finite
        pred = model.predict(new_data, pred_type="link")
        assert pred.shape == (1,)
        assert np.isfinite(pred[0])

    def test_predict_seen_diff_equals_re_coefficient(self, re_model_data) -> None:
        """At the same x, pred(seen g_j) - pred(unseen g) == RE coef for level j.

        This pins down the prediction behavior exactly: for random
        intercepts, the RE column for a seen level is 1, so the seen-row
        prediction adds exactly that level's coefficient on top of the
        parametric+smooth contribution. Unseen rows contribute zero.
        """
        model = self._fit_model(re_model_data)
        levels = list(re_model_data["g"].cat.categories)
        x_const = 0.5

        # Two seen levels and one unseen level, all at the same x
        seen_a, seen_b = levels[0], levels[1]
        new_data = pd.DataFrame(
            {
                "x": [x_const, x_const, x_const],
                "g": pd.Categorical(
                    [seen_a, seen_b, "unseen_z"],
                    categories=[*levels, "unseen_z"],
                ),
            }
        )

        pred = model.predict(new_data, pred_type="link")
        re_slice = model.setup.coef_map.term_slice("s(g)")
        re_coefs = model.coefficients[re_slice]

        # The RE block has one column per training level in registration order
        idx_a = levels.index(seen_a)
        idx_b = levels.index(seen_b)

        # pred(seen) - pred(unseen) equals that level's RE coefficient
        np.testing.assert_allclose(
            pred[0] - pred[2],
            re_coefs[idx_a],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            pred[1] - pred[2],
            re_coefs[idx_b],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_predict_missing_factor_column_raises(self, re_model_data) -> None:
        """Predicting without the RE factor column raises an error naming the column."""
        model = self._fit_model(re_model_data)

        new_data = pd.DataFrame({"x": [0.5, 0.6]})  # missing "g"
        with pytest.raises((KeyError, ValueError), match=r"['\"]?g['\"]?.*not found"):
            model.predict(new_data)

    def test_predict_mixed_seen_unseen(self, re_model_data) -> None:
        """Mixed seen/unseen factor levels: seen rows nonzero, unseen rows zero."""
        model = self._fit_model(re_model_data)
        levels = list(re_model_data["g"].cat.categories)

        new_data = pd.DataFrame(
            {
                "x": [0.3, 0.3, 0.3, 0.3],
                "g": pd.Categorical(
                    [levels[0], "unseen_x", levels[1], "unseen_y"],
                    categories=[*levels, "unseen_x", "unseen_y"],
                ),
            }
        )

        X_pred = model.predict_matrix(new_data)
        re_slice = model.setup.coef_map.term_slice("s(g)")

        # Seen rows: one nonzero in RE block
        assert X_pred[0, re_slice].sum() == pytest.approx(1.0)
        assert X_pred[2, re_slice].sum() == pytest.approx(1.0)
        # Unseen rows: all zeros in RE block
        np.testing.assert_allclose(X_pred[1, re_slice], 0.0, atol=STRICT.atol)
        np.testing.assert_allclose(X_pred[3, re_slice], 0.0, atol=STRICT.atol)

    def test_predict_random_slope_unseen_level(self, re_model_data) -> None:
        """Random slopes s(x, g, bs='re'): unseen level row contributes zero.

        For a random-slope basis, column j holds ``x`` for rows where
        ``g == level_j`` and zero elsewhere. So at the same x:
            pred(seen g_j) - pred(unseen g) == x * RE_coef[j]
        """
        model = self._fit_model(re_model_data, formula="y ~ s(x, g, bs='re')")
        levels = list(re_model_data["g"].cat.categories)
        x_const = 0.4
        seen = levels[0]

        new_data = pd.DataFrame(
            {
                "x": [x_const, x_const],
                "g": pd.Categorical(
                    [seen, "unseen_slope"],
                    categories=[*levels, "unseen_slope"],
                ),
            }
        )

        X_pred = model.predict_matrix(new_data)
        re_slice = model.setup.coef_map.term_slice("s(x,g)")

        # Row 0 (seen): RE block has x at position idx_seen, zero elsewhere
        idx_seen = levels.index(seen)
        re_row_seen = X_pred[0, re_slice]
        assert re_row_seen[idx_seen] == pytest.approx(x_const)
        mask = np.ones_like(re_row_seen, dtype=bool)
        mask[idx_seen] = False
        np.testing.assert_allclose(re_row_seen[mask], 0.0, atol=STRICT.atol)

        # Row 1 (unseen): all zeros in RE block
        np.testing.assert_allclose(X_pred[1, re_slice], 0.0, atol=STRICT.atol)

        # Prediction difference equals x * coef[idx_seen]
        pred = model.predict(new_data, pred_type="link")
        re_coefs = model.coefficients[re_slice]
        np.testing.assert_allclose(
            pred[0] - pred[1],
            x_const * re_coefs[idx_seen],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


@pytest.mark.skipif(not r_available(), reason="R with mgcv not available")
class TestEndToEndPredictionRComparison:
    """End-to-end predictions match R's predict.gam() with unseen levels."""

    def _fit_and_predict(
        self,
        train_data: pd.DataFrame,
        new_data: pd.DataFrame,
        formula: str = "y ~ s(x) + s(g, bs='re')",
    ):
        """Fit jaxgam + R, return (py_pred, r_pred)."""
        model = GAM(formula, family="gaussian").fit(train_data)
        py_pred = model.predict(new_data, pred_type="link")

        bridge = RBridge()
        r_result = bridge.predict_gam(
            formula, train_data, new_data, family="gaussian", pred_type="link"
        )
        r_pred = r_result["predictions"]

        return py_pred, r_pred

    def test_predict_seen_levels_vs_r(self, re_model_data) -> None:
        """Predictions for seen levels match R's predict.gam (MODERATE)."""
        levels = list(re_model_data["g"].cat.categories)
        new_data = pd.DataFrame(
            {
                "x": np.array([0.1, 0.4, 0.7, 0.3]),
                "g": pd.Categorical(
                    [levels[0], levels[1], levels[2], levels[0]],
                    categories=levels,
                ),
            }
        )
        py_pred, r_pred = self._fit_and_predict(re_model_data, new_data)

        np.testing.assert_allclose(
            py_pred,
            r_pred,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Predictions for seen levels differ from R",
        )

    def test_predict_unseen_levels_vs_r(self, re_model_data) -> None:
        """Predictions with unseen factor levels match R's predict.gam (MODERATE)."""
        levels = list(re_model_data["g"].cat.categories)
        # Mix of seen and unseen levels
        new_data = pd.DataFrame(
            {
                "x": np.array([0.2, 0.5, 0.8, 0.5]),
                "g": pd.Categorical(
                    [levels[0], "newlevA", levels[1], "newlevB"],
                    categories=[*levels, "newlevA", "newlevB"],
                ),
            }
        )
        py_pred, r_pred = self._fit_and_predict(re_model_data, new_data)

        np.testing.assert_allclose(
            py_pred,
            r_pred,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Predictions with unseen levels differ from R",
        )
