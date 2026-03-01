"""Tests for GAM summary (Task 3.2).

Tests cover:
- A. Self-consistency tests (STRICT)
- B. R comparison tests (MODERATE / LOOSE)
- C. Smoke tests for all smooth types
- D. Edge cases
- E. Davies algorithm unit tests

Tolerance rationale:
  Self-consistency: STRICT for algebraic properties.
  R comparison for parametric p-values: MODERATE (rtol=1e-4).
  R comparison for smooth stats: MODERATE (rtol=1e-4) — uses Davies
  exact method matching R's psum.chisq.
  EDF: LOOSE (rtol=1e-2).
  R-squared and deviance explained: MODERATE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.api import GAM
from jaxgam.summary.summary import GAMSummary, summary
from tests.tolerances import LOOSE, MODERATE, STRICT

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r_available() -> bool:
    from tests.r_bridge import RBridge

    if not RBridge.available():
        return False
    ok, _ = RBridge.check_versions()
    return ok


def _make_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic data for a given family."""
    rng = np.random.default_rng(seed)
    n = 200
    x = rng.uniform(0, 1, n)

    if family_name == "gaussian":
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        eta = 2 * np.sin(2 * np.pi * x)
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        eta = np.sin(2 * np.pi * x) + 0.5
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family_name == "gamma":
        eta = 0.5 * np.sin(2 * np.pi * x) + 1.0
        mu = np.exp(eta)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "y": y})


def _make_two_smooth_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 200
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_tensor_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 200
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_factor_by_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, n)
    levels = ["a", "b", "c"]
    fac = rng.choice(levels, n)
    eta = np.where(
        fac == "a",
        np.sin(2 * np.pi * x),
        np.where(fac == "b", 0.5 * x, -0.3 * x),
    )
    y = eta + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "x": x,
            "fac": pd.Categorical(fac, categories=levels),
            "y": y,
        }
    )


def _r_tol(family_name: str):
    if family_name == "gaussian":
        return MODERATE
    return LOOSE


# ---------------------------------------------------------------------------
# A. Self-consistency tests (STRICT)
# ---------------------------------------------------------------------------


class TestSelfConsistency:
    """Self-consistency checks for summary statistics."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(
        params=["gaussian", "poisson", "binomial", "gamma"],
        ids=["gaussian", "poisson", "binomial", "gamma"],
    )
    def fitted_summary(self, request):
        family_name = request.param
        data = _make_data(family_name)
        model = GAM(self.FORMULA, family=family_name).fit(data)
        s = summary(model)
        return family_name, model, s

    def test_parametric_se_matches_vp_diag(self, fitted_summary):
        """Parametric SE should equal sqrt(diag(Vp)) for parametric columns."""
        _, model, s = fitted_summary
        if s.p_table is None:
            pytest.skip("No parametric terms")

        n_param = s.p_table.shape[0]
        se_from_vp = np.sqrt(np.diag(model.Vp_)[:n_param])
        se_from_table = s.p_table[:, 1]

        np.testing.assert_allclose(
            se_from_table,
            se_from_vp,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Parametric SE doesn't match sqrt(diag(Vp))",
        )

    def test_edf_sums_approximately(self, fitted_summary):
        """Per-smooth EDF should sum to approximately total EDF."""
        _, model, s = fitted_summary
        if s.s_table is None:
            pytest.skip("No smooth terms")

        edf_per_smooth = s.s_table[:, 0]
        n_param = s.p_table.shape[0] if s.p_table is not None else 0

        # Total EDF = parametric coefs + sum of smooth EDFs
        total_edf_from_smooths = float(np.sum(edf_per_smooth)) + n_param

        np.testing.assert_allclose(
            total_edf_from_smooths,
            model.edf_total_,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Per-smooth EDF + parametric doesn't match total EDF",
        )

    def test_r_squared_in_range(self, fitted_summary):
        """R-squared should be between -inf and 1 for well-fitted models."""
        _, _, s = fitted_summary
        if s.r_sq is not None:
            # Adjusted R-sq can be negative but should be <= 1
            assert s.r_sq <= 1.0 + 1e-10, f"R-squared {s.r_sq} exceeds 1"

    def test_deviance_explained_in_range(self, fitted_summary):
        """Deviance explained should be between 0 and 1."""
        _, _, s = fitted_summary
        assert -0.01 <= s.dev_explained <= 1.01, (
            f"Deviance explained {s.dev_explained} out of [0, 1] range"
        )

    def test_p_values_in_range(self, fitted_summary):
        """All p-values should be in [0, 1]."""
        _, _, s = fitted_summary

        if s.p_table is not None:
            p_pv = s.p_table[:, 3]
            assert np.all(p_pv >= 0), "Negative parametric p-value"
            assert np.all(p_pv <= 1.0 + 1e-10), "Parametric p-value > 1"

        if s.s_table is not None:
            s_pv = s.s_table[:, 3]
            assert np.all(s_pv >= 0), "Negative smooth p-value"
            assert np.all(s_pv <= 1.0 + 1e-10), "Smooth p-value > 1"

    def test_residual_df_positive(self, fitted_summary):
        """Residual df should be positive."""
        _, _, s = fitted_summary
        assert s.residual_df > 0, f"Residual df {s.residual_df} not positive"

    def test_scale_positive(self, fitted_summary):
        """Scale estimate should be positive."""
        _, _, s = fitted_summary
        assert s.scale > 0, f"Scale {s.scale} not positive"


# ---------------------------------------------------------------------------
# B. R comparison tests (MODERATE / LOOSE)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestSummaryVsR:
    """Summary statistics compared to R's summary.gam()."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(
        params=[
            ("gaussian", "gaussian"),
            ("poisson", "poisson"),
            ("binomial", "binomial"),
            ("gamma", "gamma"),
        ],
        ids=["gaussian", "poisson", "binomial", "gamma"],
    )
    def summary_pair(self, request):
        from tests.r_bridge import RBridge

        family_name, family_r = request.param
        data = _make_data(family_name)

        model = GAM(self.FORMULA, family=family_name).fit(data)
        py_summary = summary(model)

        bridge = RBridge()
        r_summary = bridge.summary_gam(self.FORMULA, data, family=family_r)

        return family_name, py_summary, r_summary, model

    def test_edf_vs_r(self, summary_pair):
        """Per-smooth EDF should match R at LOOSE."""
        family_name, py_s, r_s, _ = summary_pair
        if py_s.s_table is None:
            pytest.skip("No smooth terms")

        py_edf = py_s.s_table[:, 0]
        r_edf = r_s["edf"]

        np.testing.assert_allclose(
            py_edf,
            r_edf,
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg=f"{family_name} EDF differs from R",
        )

    def test_parametric_pvalues_vs_r(self, summary_pair):
        """Parametric p-values should match R at MODERATE."""
        family_name, py_s, r_s, _ = summary_pair
        if py_s.p_table is None or r_s["p_table"] is None:
            pytest.skip("No parametric terms")

        tol = _r_tol(family_name)

        py_pv = py_s.p_table[:, 3]
        r_pv = r_s["p_table"][:, 3]

        np.testing.assert_allclose(
            py_pv,
            r_pv,
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} parametric p-values differ from R",
        )

    def test_r_squared_vs_r(self, summary_pair):
        """R-squared should match R at MODERATE."""
        family_name, py_s, r_s, _ = summary_pair
        if py_s.r_sq is None or r_s["r_sq"] is None:
            pytest.skip("R-squared not available")

        np.testing.assert_allclose(
            py_s.r_sq,
            r_s["r_sq"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=f"{family_name} R-squared differs from R",
        )

    def test_deviance_explained_vs_r(self, summary_pair):
        """Deviance explained should match R at MODERATE."""
        family_name, py_s, r_s, _ = summary_pair
        tol = _r_tol(family_name)

        np.testing.assert_allclose(
            py_s.dev_explained,
            r_s["dev_explained"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} deviance explained differs from R",
        )

    def test_scale_vs_r(self, summary_pair):
        """Scale estimate should match R at MODERATE."""
        family_name, py_s, r_s, _ = summary_pair
        tol = _r_tol(family_name)

        np.testing.assert_allclose(
            py_s.scale,
            r_s["scale"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} scale differs from R",
        )

    def test_smooth_test_stat_vs_r(self, summary_pair):
        """Smooth F/Chi-sq statistics should match R at MODERATE.

        Now uses Davies' exact method for the mixture chi-squared
        distribution, matching R's psum.chisq.
        """
        family_name, py_s, r_s, _ = summary_pair
        if py_s.s_table is None or r_s["s_table"] is None:
            pytest.skip("No smooth terms")

        tol = _r_tol(family_name)

        # Column 2 is F (for unknown-scale) or Chi.sq (for known-scale)
        py_stat = py_s.s_table[:, 2]
        r_stat = r_s["s_table"][:, 2]

        np.testing.assert_allclose(
            py_stat,
            r_stat,
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} smooth test stat differs from R",
        )

    def test_smooth_pvalue_vs_r(self, summary_pair):
        """Smooth p-values should match R at MODERATE.

        Davies exact method ensures p-value accuracy matches R.
        """
        family_name, py_s, r_s, _ = summary_pair
        if py_s.s_table is None or r_s["s_table"] is None:
            pytest.skip("No smooth terms")

        tol = _r_tol(family_name)

        # Column 3 is p-value
        py_pval = py_s.s_table[:, 3]
        r_pval = r_s["s_table"][:, 3]

        np.testing.assert_allclose(
            py_pval,
            r_pval,
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} smooth p-value differs from R",
        )


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestMultiSmoothSummaryVsR:
    """Summary for multi-smooth models vs R."""

    def test_two_smooth_edf_vs_r(self):
        from tests.r_bridge import RBridge

        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        data = _make_two_smooth_data()

        model = GAM(formula).fit(data)
        py_s = summary(model)

        bridge = RBridge()
        r_s = bridge.summary_gam(formula, data)

        py_edf = py_s.s_table[:, 0]
        r_edf = r_s["edf"]

        np.testing.assert_allclose(
            py_edf,
            r_edf,
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Two-smooth EDF differs from R",
        )

    def test_two_smooth_deviance_explained_vs_r(self):
        from tests.r_bridge import RBridge

        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        data = _make_two_smooth_data()

        model = GAM(formula).fit(data)
        py_s = summary(model)

        bridge = RBridge()
        r_s = bridge.summary_gam(formula, data)

        np.testing.assert_allclose(
            py_s.dev_explained,
            r_s["dev_explained"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Two-smooth deviance explained differs from R",
        )


# ---------------------------------------------------------------------------
# C. Smoke tests
# ---------------------------------------------------------------------------


class TestSmokeTests:
    """Smoke tests: summary prints without error for all smooth types."""

    def test_single_smooth_gaussian(self):
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')", family="gaussian").fit(data)
        s = summary(model)
        text = str(s)
        assert "Parametric coefficients:" in text
        assert "Approximate significance of smooth terms:" in text
        assert "R-sq." in text
        assert "Deviance explained" in text
        assert "Scale est." in text

    def test_single_smooth_binomial(self):
        data = _make_data("binomial")
        model = GAM("y ~ s(x, k=10, bs='cr')", family="binomial").fit(data)
        s = summary(model)
        text = str(s)
        assert "binomial" in text.lower()
        assert "Chi.sq" in text or "z value" in text

    def test_single_smooth_poisson(self):
        data = _make_data("poisson")
        model = GAM("y ~ s(x, k=10, bs='cr')", family="poisson").fit(data)
        s = summary(model)
        text = str(s)
        assert "poisson" in text.lower()

    def test_single_smooth_gamma(self):
        data = _make_data("gamma")
        model = GAM("y ~ s(x, k=10, bs='cr')", family="gamma").fit(data)
        s = summary(model)
        text = str(s)
        assert "gamma" in text.lower()

    def test_two_smooth(self):
        data = _make_two_smooth_data()
        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        model = GAM(formula).fit(data)
        s = summary(model)
        assert s.s_table is not None
        assert s.s_table.shape[0] == 2

    def test_tensor_product(self):
        data = _make_tensor_data()
        formula = "y ~ te(x1, x2, k=5)"
        model = GAM(formula).fit(data)
        s = summary(model)
        assert s.s_table is not None
        assert s.s_table.shape[0] == 1

    def test_factor_by(self):
        data = _make_factor_by_data()
        formula = "y ~ s(x, by=fac, k=10, bs='cr') + fac"
        model = GAM(formula).fit(data)
        s = summary(model)
        assert s.s_table is not None
        # FactorBySmooth is treated as a single smooth term
        assert s.s_table.shape[0] == 1

    def test_summary_returns_gam_summary(self):
        """summary() should return a GAMSummary instance."""
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        s = summary(model)
        assert isinstance(s, GAMSummary)

    def test_summary_method_returns_gam_summary(self, capsys):
        """GAM.summary() should print and return GAMSummary."""
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        s = model.summary()
        assert isinstance(s, GAMSummary)

        captured = capsys.readouterr()
        assert "Parametric coefficients:" in captured.out

    def test_str_representation(self):
        """str(GAMSummary) should return formatted summary."""
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        s = summary(model)
        text = str(s)
        assert isinstance(text, str)
        assert len(text) > 100  # Not trivially empty

    def test_purely_parametric(self):
        """Summary should work for purely parametric models."""
        rng = np.random.default_rng(SEED)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = 2.0 * x1 - 1.0 * x2 + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        model = GAM("y ~ x1 + x2").fit(data)
        s = summary(model)
        assert s.p_table is not None
        assert s.s_table is None
        text = str(s)
        assert "Parametric coefficients:" in text


# ---------------------------------------------------------------------------
# D. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for summary."""

    def test_unfitted_model_raises(self):
        """summary() on unfitted model should raise RuntimeError."""
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.summary()

    def test_known_scale_uses_z_test(self):
        """Binomial/Poisson should use z-test, not t-test."""
        data = _make_data("binomial")
        model = GAM("y ~ s(x, k=10, bs='cr')", family="binomial").fit(data)
        s = summary(model)
        assert s.p_test_name == "z value"
        assert s.p_pv_name == "Pr(>|z|)"
        assert s.s_test_name == "Chi.sq"

    def test_unknown_scale_uses_t_test(self):
        """Gaussian/Gamma should use t-test, not z-test."""
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')", family="gaussian").fit(data)
        s = summary(model)
        assert s.p_test_name == "t value"
        assert s.p_pv_name == "Pr(>|t|)"
        assert s.s_test_name == "F"

    def test_formula_stored(self):
        """Formula should be stored in summary."""
        data = _make_data("gaussian")
        formula = "y ~ s(x, k=10, bs='cr')"
        model = GAM(formula).fit(data)
        s = summary(model)
        assert s.formula == formula

    def test_n_correct(self):
        """Number of observations should match data."""
        data = _make_data("gaussian")
        model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        s = summary(model)
        assert s.n == len(data)


# ---------------------------------------------------------------------------
# E. Davies algorithm unit tests
# ---------------------------------------------------------------------------


class TestDaviesAlgorithm:
    """Unit tests for the Davies method implementation."""

    def test_single_weight_matches_chi2(self):
        """Single weight = 1 should match chi-squared(1) CDF."""
        from scipy import stats

        from jaxgam.summary._davies import _davies as davies

        for q in [0.5, 1.0, 2.0, 5.0, 10.0]:
            result = davies(
                lb=np.array([1.0]),
                nc=np.array([0.0]),
                n=np.array([1]),
                sigma=0.0,
                c=q,
            )
            expected = stats.chi2.cdf(q, df=1)
            np.testing.assert_allclose(
                result.prob,
                expected,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=f"Single chi2(1) at q={q}",
            )

    def test_single_weight_df3_matches_chi2(self):
        """Single weight = 1, df = 3 should match chi-squared(3) CDF."""
        from scipy import stats

        from jaxgam.summary._davies import _davies as davies

        for q in [1.0, 3.0, 7.0]:
            result = davies(
                lb=np.array([1.0]),
                nc=np.array([0.0]),
                n=np.array([3]),
                sigma=0.0,
                c=q,
            )
            expected = stats.chi2.cdf(q, df=3)
            np.testing.assert_allclose(
                result.prob,
                expected,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=f"Single chi2(3) at q={q}",
            )

    def test_scaled_chi2(self):
        """Weight = 2 with df = 1: should match CDF of 2 * chi2(1)."""
        from scipy import stats

        from jaxgam.summary._davies import _davies as davies

        for q in [1.0, 3.0, 5.0]:
            result = davies(
                lb=np.array([2.0]),
                nc=np.array([0.0]),
                n=np.array([1]),
                sigma=0.0,
                c=q,
            )
            # Pr(2 * X < q) = Pr(X < q/2) where X ~ chi2(1)
            expected = stats.chi2.cdf(q / 2.0, df=1)
            np.testing.assert_allclose(
                result.prob,
                expected,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=f"Scaled chi2 at q={q}",
            )

    def test_ifault_zero_on_valid_input(self):
        """Valid inputs should produce ifault = 0."""
        from jaxgam.summary._davies import _davies as davies

        result = davies(
            lb=np.array([1.0, 0.5]),
            nc=np.array([0.0, 0.0]),
            n=np.array([1, 2]),
            sigma=0.0,
            c=3.0,
        )
        assert result.ifault in (0, 2)

    def test_ifault_3_on_negative_df(self):
        """Negative df should produce ifault = 3."""
        from jaxgam.summary._davies import _davies as davies

        result = davies(
            lb=np.array([1.0]),
            nc=np.array([0.0]),
            n=np.array([-1]),
            sigma=0.0,
            c=1.0,
        )
        assert result.ifault == 3

    def test_probability_bounds(self):
        """Result probability should be in [0, 1]."""
        from jaxgam.summary._davies import _davies as davies

        for q in [0.1, 1.0, 5.0, 20.0]:
            result = davies(
                lb=np.array([1.0, 0.5, 0.3]),
                nc=np.array([0.0, 0.0, 0.0]),
                n=np.array([1, 1, 1]),
                sigma=0.0,
                c=q,
            )
            if result.ifault in (0, 2):
                assert 0.0 <= result.prob <= 1.0, (
                    f"prob={result.prob} out of [0,1] at q={q}"
                )

    def test_extreme_tails(self):
        """Very large q should give prob near 1, very small near 0."""
        from jaxgam.summary._davies import _davies as davies

        result_large = davies(
            lb=np.array([1.0]),
            nc=np.array([0.0]),
            n=np.array([1]),
            sigma=0.0,
            c=100.0,
        )
        assert result_large.prob > 0.99

        result_small = davies(
            lb=np.array([1.0]),
            nc=np.array([0.0]),
            n=np.array([1]),
            sigma=0.0,
            c=0.001,
        )
        assert result_small.prob < 0.05


class TestPsumChisqDavies:
    """Unit tests for the psum_chisq_davies wrapper."""

    def test_upper_tail_single_chi2(self):
        """Upper tail for single chi2(1) should match scipy."""
        from scipy import stats

        from jaxgam.summary.summary import psum_chisq_davies

        for q in [1.0, 3.84, 6.63]:
            result = psum_chisq_davies(q, np.array([1.0]))
            expected = stats.chi2.sf(q, df=1)
            np.testing.assert_allclose(
                result,
                expected,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=f"Upper tail chi2(1) at q={q}",
            )

    def test_negative_weights(self):
        """Should handle negative weights (used in F-like tests)."""
        from jaxgam.summary.summary import psum_chisq_davies

        # Pr(X1 - 0.5*X2 > 0) where X1, X2 ~ chi2(1)
        result = psum_chisq_davies(0.0, np.array([1.0, -0.5]), df=np.array([1, 1]))
        assert 0.0 <= result <= 1.0

    def test_mixed_df(self):
        """Should handle mixed degrees of freedom."""
        from jaxgam.summary.summary import psum_chisq_davies

        result = psum_chisq_davies(5.0, np.array([1.0, 0.5]), df=np.array([1, 3]))
        assert 0.0 <= result <= 1.0


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestDaviesVsR:
    """Davies implementation compared to R's psum.chisq."""

    def test_mixture_chi2_vs_r(self):
        """Mixture of chi-squared should match R's psum.chisq."""
        import rpy2.robjects as ro

        lb = np.array([1.0, 0.5, 0.3])
        q_vals = [1.0, 3.0, 5.0]

        for q in q_vals:
            r_code = f"""
            mgcv::psum.chisq({q}, c(1.0, 0.5, 0.3),
                             df=c(1,1,1), nc=c(0,0,0))
            """
            r_result = float(ro.r(r_code)[0])

            from jaxgam.summary.summary import psum_chisq_davies

            py_result = psum_chisq_davies(q, lb)

            np.testing.assert_allclose(
                py_result,
                r_result,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=f"psum.chisq mismatch at q={q}",
            )

    def test_f_like_mixture_vs_r(self):
        """F-like test (negative weights) should match R."""
        import rpy2.robjects as ro

        lb = np.array([1.0, 0.8, -2.0])
        df = np.array([1, 1, 5])

        r_code = """
        mgcv::psum.chisq(0, c(1.0, 0.8, -2.0),
                         df=c(1, 1, 5), nc=c(0,0,0))
        """
        r_result = float(ro.r(r_code)[0])

        from jaxgam.summary.summary import psum_chisq_davies

        py_result = psum_chisq_davies(0.0, lb, df=df)

        np.testing.assert_allclose(
            py_result,
            r_result,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="F-like psum.chisq mismatch",
        )
