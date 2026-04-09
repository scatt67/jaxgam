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

from jaxgam.penalties.penalty import Penalty
from jaxgam.smooths.random_effects import RandomEffectSmooth
from tests.helpers import SEED, make_smooth_spec, r_available
from tests.tolerances import STRICT

# ===========================================================================
# 1. Structural tests
# ===========================================================================


class TestStructural:
    """Structural properties of RandomEffectSmooth."""

    def test_flags(self, re_factor_data) -> None:
        """RE smooth has correct flags after setup."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        assert smooth.side_constrain is False
        assert smooth._noterp is True
        assert smooth._random is True
        assert smooth._has_centering_constraint is False

    def test_null_space_dim_zero(self, re_factor_data) -> None:
        """RE smooth has null_space_dim = 0."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        assert smooth.null_space_dim == 0

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

    def test_setup_required_for_design_matrix(self) -> None:
        """build_design_matrix before setup raises RuntimeError."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.build_design_matrix({"g": pd.Series(pd.Categorical(["a", "b"]))})

    def test_setup_required_for_penalty(self) -> None:
        """build_penalty_matrices before setup raises RuntimeError."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.build_penalty_matrices()

    def test_setup_required_for_predict(self) -> None:
        """predict_matrix before setup raises RuntimeError."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        with pytest.raises(RuntimeError, match="setup"):
            smooth.predict_matrix({"g": pd.Series(pd.Categorical(["a", "b"]))})


# ===========================================================================
# 2. Penalty tests
# ===========================================================================


class TestPenalty:
    """Tests for RE penalty construction."""

    def test_penalty_returns_list_of_penalty(self, re_factor_data) -> None:
        """build_penalty_matrices returns list[Penalty] with one element."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        penalties = smooth.build_penalty_matrices()
        assert isinstance(penalties, list)
        assert len(penalties) == 1
        assert isinstance(penalties[0], Penalty)

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

    def test_penalty_shape(self, re_factor_data) -> None:
        """RE penalty has shape (n_coefs, n_coefs)."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        S = smooth.build_penalty_matrices()[0].S
        assert S.shape == (smooth.n_coefs, smooth.n_coefs)


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

    def test_numeric_only(self) -> None:
        """Numeric-only produces single column = x values."""
        rng = np.random.default_rng(SEED)
        x = rng.uniform(0, 1, 50)
        data = {"x": x}

        spec = make_smooth_spec(["x"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert smooth.n_coefs == 1
        assert X.shape == (50, 1)
        np.testing.assert_allclose(X[:, 0], x, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_two_numeric_interaction(self) -> None:
        """Two numerics produce single column = x1 * x2."""
        rng = np.random.default_rng(SEED)
        x1 = rng.uniform(0, 1, 50)
        x2 = rng.uniform(0, 1, 50)
        data = {"x1": x1, "x2": x2}

        spec = make_smooth_spec(["x1", "x2"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(data)

        X = smooth.build_design_matrix(data)
        assert smooth.n_coefs == 1
        np.testing.assert_allclose(X[:, 0], x1 * x2, rtol=STRICT.rtol, atol=STRICT.atol)

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

    def test_predict_reproduces_design_matrix(self, re_factor_data) -> None:
        """predict_matrix reproduces build_design_matrix on training data."""
        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_factor_data)

        X_design = smooth.build_design_matrix(re_factor_data)
        X_predict = smooth.predict_matrix(re_factor_data)
        np.testing.assert_allclose(
            X_predict, X_design, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_predict_reproduces_two_factor(self, re_two_factor_data) -> None:
        """predict_matrix reproduces build_design_matrix for two-factor RE."""
        spec = make_smooth_spec(["g1", "g2"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_two_factor_data)

        X_design = smooth.build_design_matrix(re_two_factor_data)
        X_predict = smooth.predict_matrix(re_two_factor_data)
        np.testing.assert_allclose(
            X_predict, X_design, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_predict_reproduces_numeric_factor(self, re_numeric_factor_data) -> None:
        """predict_matrix reproduces build_design_matrix for numeric x factor."""
        spec = make_smooth_spec(["x", "g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(re_numeric_factor_data)

        X_design = smooth.build_design_matrix(re_numeric_factor_data)
        X_predict = smooth.predict_matrix(re_numeric_factor_data)
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

    def _setup_single_factor(self) -> tuple:
        """Shared setup: s(g, bs='re') with 20-level factor."""
        from tests.r_bridge import RBridge

        rng = np.random.default_rng(SEED)
        n = 100
        n_groups = 20
        g = rng.choice([f"g{i}" for i in range(n_groups)], size=n)
        data = pd.DataFrame({"g": pd.Categorical(g)})

        bridge = RBridge()
        r_result = bridge.smooth_construct("s(g, bs='re')", data)

        spec = make_smooth_spec(["g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup({"g": pd.Series(pd.Categorical(g))})

        return smooth, r_result, data

    def _setup_two_factor(self) -> tuple:
        """Shared setup: s(g1, g2, bs='re') with factor interaction."""
        from tests.r_bridge import RBridge

        rng = np.random.default_rng(SEED)
        n = 100
        g1 = rng.choice(["a", "b", "c"], size=n)
        g2 = rng.choice(["x", "y"], size=n)
        data = pd.DataFrame(
            {
                "g1": pd.Categorical(g1),
                "g2": pd.Categorical(g2),
            }
        )

        bridge = RBridge()
        r_result = bridge.smooth_construct("s(g1, g2, bs='re')", data)

        spec = make_smooth_spec(["g1", "g2"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup(
            {
                "g1": pd.Series(pd.Categorical(g1)),
                "g2": pd.Series(pd.Categorical(g2)),
            }
        )

        return smooth, r_result, data

    def _setup_numeric_factor(self) -> tuple:
        """Shared setup: s(x, g, bs='re') with numeric x factor."""
        from tests.r_bridge import RBridge

        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        g = rng.choice([f"g{i}" for i in range(10)], size=n)
        data = pd.DataFrame(
            {
                "x": x,
                "g": pd.Categorical(g),
            }
        )

        bridge = RBridge()
        r_result = bridge.smooth_construct("s(x, g, bs='re')", data)

        spec = make_smooth_spec(["x", "g"], bs="re")
        smooth = RandomEffectSmooth(spec)
        smooth.setup({"x": x, "g": pd.Series(pd.Categorical(g))})

        return smooth, r_result, data

    # --- Single factor ---

    def test_single_factor_X_vs_r(self) -> None:
        """s(g, bs='re') basis matrix X matches R (STRICT)."""
        smooth, r_result, data = self._setup_single_factor()
        X_py = smooth.build_design_matrix({"g": data["g"]})
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py,
            X_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="RE single-factor X differs from R",
        )

    def test_single_factor_S_vs_r(self) -> None:
        """s(g, bs='re') penalty matrix S matches R (STRICT)."""
        smooth, r_result, _data = self._setup_single_factor()
        S_py = smooth.build_penalty_matrices()[0].S
        S_r = r_result["S"][0]

        np.testing.assert_allclose(
            S_py,
            S_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="RE single-factor S differs from R",
        )

    def test_single_factor_rank_vs_r(self) -> None:
        """s(g, bs='re') rank and null_space_dim match R."""
        smooth, r_result, _data = self._setup_single_factor()
        assert smooth.rank == r_result["rank"]
        assert smooth.null_space_dim == r_result["null_space_dim"]

    # --- Factor x factor ---

    def test_two_factor_X_vs_r(self) -> None:
        """s(g1, g2, bs='re') basis matrix X matches R (STRICT)."""
        smooth, r_result, data = self._setup_two_factor()
        X_py = smooth.build_design_matrix(
            {
                "g1": data["g1"],
                "g2": data["g2"],
            }
        )
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py,
            X_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="RE two-factor X differs from R",
        )

    def test_two_factor_S_vs_r(self) -> None:
        """s(g1, g2, bs='re') penalty matrix S matches R (STRICT)."""
        smooth, r_result, _data = self._setup_two_factor()
        S_py = smooth.build_penalty_matrices()[0].S
        S_r = r_result["S"][0]

        np.testing.assert_allclose(
            S_py,
            S_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="RE two-factor S differs from R",
        )

    def test_two_factor_rank_vs_r(self) -> None:
        """s(g1, g2, bs='re') rank and null_space_dim match R."""
        smooth, r_result, _data = self._setup_two_factor()
        assert smooth.rank == r_result["rank"]
        assert smooth.null_space_dim == r_result["null_space_dim"]

    # --- Numeric x factor ---

    def test_numeric_factor_X_vs_r(self) -> None:
        """s(x, g, bs='re') basis matrix X matches R (STRICT)."""
        smooth, r_result, data = self._setup_numeric_factor()
        X_py = smooth.build_design_matrix(
            {
                "x": data["x"].values,
                "g": data["g"],
            }
        )
        X_r = r_result["X"]

        np.testing.assert_allclose(
            X_py,
            X_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="RE numeric-factor X differs from R",
        )

    def test_numeric_factor_S_vs_r(self) -> None:
        """s(x, g, bs='re') penalty matrix S matches R (STRICT)."""
        smooth, r_result, _data = self._setup_numeric_factor()
        S_py = smooth.build_penalty_matrices()[0].S
        S_r = r_result["S"][0]

        np.testing.assert_allclose(
            S_py,
            S_r,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="RE numeric-factor S differs from R",
        )

    def test_numeric_factor_rank_vs_r(self) -> None:
        """s(x, g, bs='re') rank and null_space_dim match R."""
        smooth, r_result, _data = self._setup_numeric_factor()
        assert smooth.rank == r_result["rank"]
        assert smooth.null_space_dim == r_result["null_space_dim"]
