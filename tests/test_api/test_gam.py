"""Tests for GAM class (sklearn-style API).

Tests cover:
- A. GAM class API smoke checks
- B. End-to-end fitting shapes
- C. Factor-by API metadata
- D. ML not supported (deferred in v1.0)
- E. Fixed smoothing parameters
- F. Scope guards
- G. Edge cases (purely parametric, offset)

R-parity and hard-gate ownership:
  Broad family x smooth R parity and final-result hard gates live in
  tests/test_validation_matrix.py. This file owns public API orchestration,
  routing, input validation, and fixed-sp behavior.

Design doc reference: Section 10.1, 10.2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.api import GAM
from jaxgam.results import GAMResults
from tests.helpers import (
    SEED,
    _AssertCollector,
    _generate_family_data,
    _make_nb_data,
    check_that,
    r_available,
)
from tests.tolerances import LOOSE, STRICT

# ---------------------------------------------------------------------------
# A. TestGAMClass — basic API tests (no R)
# ---------------------------------------------------------------------------


class TestGAMClass:
    """Test GAM class interface."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_summary_and_plot_work(self):
        import matplotlib

        matplotlib.use("Agg")
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        s = results.summary()
        assert s is not None
        fig, _axes = results.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close("all")


# ---------------------------------------------------------------------------
# B. TestEndToEnd — basic fitting (no R)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end fitting sanity checks."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_shapes(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        n = results.n
        p = results.X.shape[1]
        n_smooths = len(results.smooth_info)
        assert results.coefficients.shape == (p,)
        assert results.fitted_values.shape == (n,)
        assert results.linear_predictor.shape == (n,)
        assert results.Vp.shape == (p, p)
        assert results.edf.shape == (n_smooths,)
        assert results.X.shape == (n, p)
        assert results.execution_path == "jax"
        assert results.lambda_strategy == "newton_reml"

    def test_convergence_info_exposed(self):
        """GAMResults surfaces the optimizer's terminal state.

        ``convergence_info`` was dropped at the Phase 2->3 boundary, so users
        could not tell a step failure from an iteration-limit hit; it is now
        propagated from the NewtonResult.
        """
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        assert results.converged
        assert results.convergence_info == "full convergence"


# ---------------------------------------------------------------------------
# C. TestFactorBy — factor-by API metadata
# ---------------------------------------------------------------------------


class TestFactorBy:
    """Factor-by smooth API behavior."""

    def test_factor_by_edf_count(self, factor_by_data):
        """Factor-by smooth is stored as one combined SmoothInfo entry."""
        formula = "y ~ s(x, by=fac, k=10, bs='cr') + fac"
        results = GAM(formula).fit(factor_by_data)
        # Our architecture stores factor-by as a single combined SmoothInfo
        # with 3 penalties (one per level), not 3 separate smooths.
        assert len(results.edf) == 1, (
            f"Expected 1 combined per-smooth EDF entry for factor-by, "
            f"got {len(results.edf)}"
        )


# ---------------------------------------------------------------------------
# D. TestMLNotSupported — ML is deferred in v1.0
# ---------------------------------------------------------------------------


class TestMLNotSupported:
    """ML is not available in v1.0.

    mgcv's ML criterion uses the penalty range-space projection of
    log|X'WX+S| (MLpenalty1), which differs from REML's full-space
    determinant. ``GAM`` rejects method='ML' rather than fit a wrong
    criterion; only REML is supported.
    """

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_ml_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="ML"):
            GAM(self.FORMULA, method="ML")

    def test_ml_rejected_case_insensitive(self):
        for meth in ("ml", "Ml", "mL"):
            with pytest.raises(NotImplementedError):
                GAM(self.FORMULA, method=meth)

    def test_reml_still_default(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA).fit(data)
        assert results.method == "REML"
        assert results.lambda_strategy == "newton_reml"


# ---------------------------------------------------------------------------
# E. TestFixedSP — user-supplied smoothing parameters
# ---------------------------------------------------------------------------


class TestFixedSP:
    """Fixed smoothing parameter tests."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_fixed_sp_attributes(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA, sp=[1.0]).fit(data)
        assert isinstance(results.coefficients, np.ndarray)
        assert results.converged

    def test_fixed_sp_lambda_matches(self):
        data = _generate_family_data("gaussian")
        sp = [2.5]
        results = GAM(self.FORMULA, sp=sp).fit(data)
        np.testing.assert_allclose(
            results.smoothing_params,
            np.array(sp),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Fixed sp not preserved",
        )

    def test_fixed_sp_lambda_strategy(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA, sp=[1.0]).fit(data)
        assert results.lambda_strategy == "fixed"

    def test_fixed_sp_n_iter_zero(self):
        data = _generate_family_data("gaussian")
        results = GAM(self.FORMULA, sp=[1.0]).fit(data)
        assert results.n_iter == 0

    def test_scalar_sp_accepted(self):
        """A scalar ``sp`` is accepted for a single-penalty model.

        Regression: a scalar ``sp`` (e.g. ``1.0`` or ``np.array(1.0)``)
        previously raised ``IndexError: tuple index out of range`` because
        ``_fit_fixed_sp`` indexed ``sp_arr.shape[0]`` on a 0-d array.
        """
        data = _generate_family_data("gaussian")
        for sp in (1.0, np.array(1.0)):
            results = GAM(self.FORMULA, sp=sp).fit(data)
            np.testing.assert_allclose(
                results.smoothing_params,
                np.array([1.0]),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"scalar sp={sp!r} not handled",
            )


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestFixedSPNegativeBinomial:
    """Fixed sp must still estimate Negative Binomial theta.

    Bug: ``GAM.fit`` routed every fixed-``sp`` fit to a single PIRLS at the
    family's *current* theta and never estimated theta, returning
    ``results.theta=None`` and a theta frozen at its initial value (a silently
    wrong fit). mgcv keeps family parameters in the outer optimization even
    when the smoothing parameters are fixed (``gam.fit4``). Fixed by pinning
    the smoothing parameters and optimizing theta only.

    Owned here (not the validation matrix) because it exercises the public
    fixed-``sp`` routing in ``GAM.fit`` / ``_fit_fixed_sp``.
    """

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_theta_estimated_at_fixed_sp(self):
        from jaxgam.families.negative_binomial import NegativeBinomial
        from tests.r_bridge import RBridge

        data = _make_nb_data(n=250, seed=5, true_theta=3.0)
        bridge = RBridge()
        # At R's REML-optimal sp, the conditional-optimal theta equals the
        # joint-optimal (auto) theta, so the auto fit is the reference.
        r_auto = bridge.fit_gam(self.FORMULA, data, family="nb")
        sp = [float(r_auto["smoothing_params"][0])]

        # Start theta deliberately wrong (9.0 vs truth ~3): a correct fit
        # estimates it toward R's value rather than leaving it at the init.
        res = GAM(self.FORMULA, family=NegativeBinomial(theta=9.0), sp=sp).fit(data)

        collector = _AssertCollector()
        collector.check(
            "theta_estimated",
            lambda: check_that(res.theta is not None, "theta was not estimated (None)"),
        )
        collector.check(
            "theta_not_stuck_at_init",
            lambda: check_that(
                abs(res.theta - 9.0) > 1.0, f"theta stuck near init: {res.theta}"
            ),
        )
        collector.check(
            "theta_vs_r",
            lambda: np.testing.assert_allclose(
                res.theta, r_auto["theta"], rtol=LOOSE.rtol, atol=LOOSE.atol
            ),
        )
        collector.check(
            "deviance_vs_r",
            lambda: np.testing.assert_allclose(
                res.deviance, r_auto["deviance"], rtol=LOOSE.rtol, atol=LOOSE.atol
            ),
        )
        collector.check(
            "sp_pinned",
            lambda: np.testing.assert_allclose(
                res.smoothing_params, np.array(sp), rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        collector.raise_if_any("NB fixed-sp theta estimation")


# ---------------------------------------------------------------------------
# F. TestScopeGuards — v1.0 scope guards
# ---------------------------------------------------------------------------


class TestScopeGuards:
    """v1.0 scope guard validation."""

    def test_backend_numpy_raises(self):
        with pytest.raises(NotImplementedError, match="backend='numpy'"):
            GAM("y ~ s(x)", backend="numpy")

    def test_select_true_raises(self):
        with pytest.raises(NotImplementedError, match="select=True"):
            GAM("y ~ s(x)", select=True)

    def test_gamma_nondefault_raises(self):
        with pytest.raises(NotImplementedError, match=r"gamma=1\.4"):
            GAM("y ~ s(x)", gamma=1.4)

    def test_knots_raises(self):
        with pytest.raises(NotImplementedError, match="knots"):
            GAM("y ~ s(x)", knots={"x": [0, 0.5, 1]})

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="GCV"):
            GAM("y ~ s(x)", method="GCV")

    def test_jax_backend_allowed(self):
        # backend="jax" should not raise
        GAM("y ~ s(x)", backend="jax")

    def test_newton_optimizer_allowed(self):
        # optimizer="newton" should not raise
        GAM("y ~ s(x)", optimizer="newton")


# ---------------------------------------------------------------------------
# G. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and special configurations."""

    def test_purely_parametric(self):
        """Purely parametric model: no smooth terms."""
        rng = np.random.default_rng(SEED)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = 2.0 * x1 - 1.0 * x2 + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        results = GAM("y ~ x1 + x2").fit(data)
        assert results.edf.shape == (0,)
        assert results.smoothing_params.shape == (0,)
        assert results.converged

    def test_offset_support(self):
        """Offset changes coefficients."""
        data = _generate_family_data("gaussian")
        n = len(data)
        results_no_offset = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        offset = np.ones(n) * 0.5
        results_with_offset = GAM("y ~ s(x, k=10, bs='cr')").fit(data, offset=offset)
        # Coefficients should differ
        assert not np.allclose(
            results_no_offset.coefficients,
            results_with_offset.coefficients,
            atol=LOOSE.atol,
        ), "Offset should change coefficients"

    def test_method_case_insensitive(self):
        """Method name is case-insensitive."""
        data = _generate_family_data("gaussian")
        results = GAM("y ~ s(x, k=10, bs='cr')", method="reml").fit(data)
        assert results.converged

    def test_chaining_api(self):
        """GAM(...).fit(data) chaining returns GAMResults."""
        data = _generate_family_data("gaussian")
        results = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
        assert isinstance(results, GAMResults)
        assert results.coefficients.shape[0] > 0

    def test_family_object_accepted(self):
        """ExponentialFamily object works as family parameter."""
        from jaxgam.families.standard import Gaussian

        data = _generate_family_data("gaussian")
        results = GAM("y ~ s(x, k=10, bs='cr')", family=Gaussian()).fit(data)
        assert results.converged
