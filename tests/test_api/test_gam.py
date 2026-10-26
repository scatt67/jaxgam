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
from tests.tolerances import LOOSE, MODERATE, STRICT

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

    def test_factor_by_edf_is_per_level(self, factor_by_data):
        """Factor-by smooth reports one SmoothInfo / EDF entry PER LEVEL (H1).

        mgcv replicates the smooth once per factor level (R/smooth.r:3969-3991),
        so a 3-level factor-by yields 3 per-level EDF entries labeled
        ``s(x):faclevel``, not one lumped entry.
        """
        formula = "y ~ s(x, by=fac, k=10, bs='cr') + fac"
        results = GAM(formula).fit(factor_by_data)
        assert len(results.edf) == 3, (
            f"Expected 3 per-level EDF entries for a 3-level factor-by, "
            f"got {len(results.edf)}"
        )
        assert [si.label for si in results.smooth_info] == [
            "s(x):faca",
            "s(x):facb",
            "s(x):facc",
        ]


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

    def test_fixed_sp_iteration_behavior(self):
        """Known-scale fixed-sp is a single PIRLS (n_iter==0); unknown-scale
        co-optimizes the REML scale (mgcv's scale.as.sp) with sp held fixed.

        Finding H7: a fixed-sp standard-family fit must report the real REML
        score, not the old 0.0 stub. For unknown-scale families (Gaussian) this
        means the outer loop runs to optimize phi while the real sp stay pinned.
        """
        # Poisson (known scale): single PIRLS, no outer iteration.
        results_p = GAM(self.FORMULA, family="poisson", sp=[1.0]).fit(
            _generate_family_data("poisson")
        )
        assert results_p.n_iter == 0
        assert abs(float(results_p.score)) > 1e-6  # not the 0.0 stub
        np.testing.assert_allclose(
            results_p.smoothing_params, [1.0], rtol=STRICT.rtol, atol=STRICT.atol
        )
        # Gaussian (unknown scale): phi co-optimized; sp pinned; real score.
        results_g = GAM(self.FORMULA, sp=[1.0]).fit(_generate_family_data("gaussian"))
        assert results_g.converged
        assert abs(float(results_g.score)) > 1e-6
        np.testing.assert_allclose(
            results_g.smoothing_params, [1.0], rtol=STRICT.rtol, atol=STRICT.atol
        )

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

    @pytest.mark.parametrize("bad_sp", [[-1.0], [0.0], [np.inf], [np.nan]])
    def test_nonpositive_sp_raises_clear_error(self, bad_sp):
        """sp<=0 / non-finite raises a clear ValueError naming the bad value.

        Finding H7: previously sp=[-1.0] hit ``log(-1)`` and failed deep in
        PIRLS with the cryptic "array must not contain infs or NaNs", and
        sp=[0.0] was silently masked by the score=0.0 stub.
        """
        data = _generate_family_data("gaussian")
        with pytest.raises(
            ValueError, match="sp must contain strictly positive"
        ) as exc:
            GAM(self.FORMULA, sp=bad_sp).fit(data)
        msg = str(exc.value).lower()
        assert "must not contain infs or nans" not in msg, (
            f"sp={bad_sp!r} produced the cryptic PIRLS error: {exc.value}"
        )
        assert "sp" in msg, f"error does not mention sp: {exc.value}"
        assert "positive" in msg, f"error does not explain sp>0: {exc.value}"


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestFixedSPScoreVsR:
    """Fixed-sp fits report the real REML score (R's gcv.ubre), not 0.0 (H7).

    Owned here (not the validation matrix) because it exercises the public
    fixed-``sp`` routing in ``GAM.fit`` / ``_fit_fixed_sp``. Pinning jaxgam's sp
    to R's REML-optimal sp makes R's reported ``gcv.ubre`` the REML score at
    exactly that sp, so the fixed-sp ``.score`` must match it.
    """

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def _check(self, family, tol):
        from tests.r_bridge import RBridge

        data = _generate_family_data(family)
        bridge = RBridge()
        r_auto = bridge.fit_gam(self.FORMULA, data, family=family)
        sp = [float(r_auto["smoothing_params"][0])]
        res = GAM(self.FORMULA, family=family, sp=sp).fit(data)

        collector = _AssertCollector()
        collector.check(
            f"{family}_score_not_zero",
            lambda: check_that(
                abs(float(res.score)) > 1e-6,
                f"fixed-sp score is the 0.0 stub: {float(res.score)}",
            ),
        )
        collector.check(
            f"{family}_score_vs_r",
            lambda: np.testing.assert_allclose(
                float(res.score),
                r_auto["reml_score"],
                rtol=tol.rtol,
                atol=max(tol.atol, 0.05),
            ),
        )
        collector.check(
            f"{family}_sp_pinned",
            lambda: np.testing.assert_allclose(
                np.asarray(res.smoothing_params),
                np.asarray(sp),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        collector.raise_if_any(f"fixed-sp score parity ({family})")

    def test_fixed_sp_score_gaussian(self):
        # Unknown-scale family: co-optimized-phi (pin_lambda) path.
        self._check("gaussian", MODERATE)

    def test_fixed_sp_score_poisson(self):
        # Known-scale family: REMLCriterion-at-fixed-sp path.
        self._check("poisson", LOOSE)


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
# E2. Supplied smoothing parameters beyond the optimizer's log-sp box
# ---------------------------------------------------------------------------

_EXTREME_SP_FORMULA = 'y ~ s(x, bs="cr", k=8)'
_EXTREME_SP_FAMILIES = ("gaussian", "gamma", "nb-fixed", "nb-estimated")


def _extreme_sp_case(name: str):
    """Rows, a jaxgam family factory and pinned-R family arguments.

    The rows match ``RBridge.public_fixed_sp_fit``'s ``y ~ s(x, bs="cr",
    k=8)`` model. A negative R theta asks pinned ``nb()`` to estimate theta
    from 2, as ``NegativeBinomial(theta=2.0)`` does.
    """
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.families.standard import Gamma, Gaussian

    rng = np.random.default_rng(4101)
    x = np.sort(rng.uniform(-1.0, 1.0, 120))
    signal = 0.4 + 0.6 * np.sin(3.0 * x)
    y_gaussian = signal + rng.normal(scale=0.2, size=x.size)
    y_count = rng.poisson(np.exp(signal)).astype(float)
    y_gamma = rng.gamma(4.0, np.exp(signal) / 4.0)
    y, family, r_family = {
        "gaussian": (y_gaussian, Gaussian, ("gaussian", "identity", None)),
        "gamma": (y_gamma, lambda: Gamma("log"), ("gamma", "log", None)),
        "nb-fixed": (
            y_count,
            lambda: NegativeBinomial(theta=2.0, fixed=True),
            ("nb", "log", 2.0),
        ),
        "nb-estimated": (
            y_count,
            lambda: NegativeBinomial(theta=2.0),
            ("nb", "log", -2.0),
        ),
    }[name]
    return pd.DataFrame({"x": x, "y": y}), family, r_family


def _pinned_smoothing_params(sp: float) -> np.ndarray:
    """The supplied sp after jaxgam's single log transform at entry."""
    import jax.numpy as jnp

    from jaxgam.api import _fixed_log_smoothing_parameters

    return np.asarray(jnp.exp(_fixed_log_smoothing_parameters([sp], 1)))


class TestFixedSPBeyondOptimizerBox:
    """Supplied sp outside the optimizer's ``|log sp| <= 40`` box stay pinned.

    Bug: the dense pinned-Newton route (unknown scale, or estimated NB theta)
    passed supplied smoothing parameters through the free-coordinate
    ``lsp_max`` clamp, so a supplied log sp of +-41 was fitted and reported
    as +-40; and a NaN REML score passed every convergence test, so NB at
    log sp +41 returned ``score=nan`` with ``converged=True``. Pinned mgcv
    keeps a supplied sp verbatim in ``lsp0`` (mgcv.r ``gam.setup``), and its
    ``newton()`` has no log-sp bound (gam.fit3.r). Owned here because it
    exercises the public fixed-``sp`` routing in ``GAM.fit`` /
    ``_fit_fixed_sp``.
    """

    @pytest.mark.parametrize("family_name", _EXTREME_SP_FAMILIES)
    def test_supplied_sp_stay_pinned_and_convergence_is_truthful(
        self, family_name: str
    ):
        data, family, _ = _extreme_sp_case(family_name)
        checks = _AssertCollector()
        for log_sp in (-60.0, -41.0, 41.0, 60.0):
            sp = float(np.exp(log_sp))
            res = GAM(_EXTREME_SP_FORMULA, family=family(), sp=[sp]).fit(data)
            score = float(res.score)
            label = f"log sp {log_sp:+.0f}"
            report = (
                f"converged={res.converged} score={score!r} "
                f"info={res.convergence_info!r}"
            )
            checks.check(
                f"{label} pinned bitwise",
                lambda res=res, sp=sp: np.testing.assert_array_equal(
                    res.smoothing_params, _pinned_smoothing_params(sp)
                ),
            )
            checks.check(
                f"{label} non-finite score never converged",
                lambda res=res, score=score, report=report: check_that(
                    np.isfinite(score)
                    or (not res.converged and "non-finite" in res.convergence_info),
                    report,
                ),
            )
            if log_sp < 0:
                checks.check(
                    f"{label} converged",
                    lambda res=res, score=score, report=report: check_that(
                        res.converged and np.isfinite(score), report
                    ),
                )
        checks.raise_if_any(f"{family_name} supplied sp beyond the log-sp box")

    def test_known_scale_nonfinite_score_is_not_converged(self, monkeypatch):
        """The single-PIRLS known-scale route reports a NaN score as failed."""
        import jax.numpy as jnp

        import jaxgam.api as api_module

        class _NonFiniteScore(api_module.REMLCriterion):
            def score(self, _log_lambda):
                return jnp.asarray(jnp.nan)

        monkeypatch.setattr(api_module, "REMLCriterion", _NonFiniteScore)
        res = GAM("y ~ s(x, k=10, bs='cr')", family="poisson", sp=[1.0]).fit(
            _generate_family_data("poisson")
        )
        assert not res.converged
        assert res.convergence_info == "fixed sp: non-finite REML score"
        assert np.isnan(res.score)
        np.testing.assert_array_equal(
            res.smoothing_params, _pinned_smoothing_params(1.0)
        )


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestFixedSPBeyondOptimizerBoxVsR:
    """Supplied ``|log sp| > 40`` match pinned mgcv's ``lsp0`` fit at STRICT.

    R reports ``full.sp = exp(lsp0)`` with the supplied sp kept verbatim.
    The R fit uses mgcv's default control (``epsilon=1e-7``, newton
    ``conv.tol=1e-6``), whose tolerances the dense non-Gaussian Newton route
    ports (outer ``1e-6``, inner ``conv.tol / 100``); tighter R control moves
    the estimated NB theta along its flat REML direction. Score, deviance,
    scale, total EDF, theta and the smoothing parameter are compared at
    STRICT. Log sp below -40 is the effectively unpenalized limit, where
    pinned R converges and the clamped fit's score differed by
    ``0.5 * rank * |log sp + 40|``.
    """

    @pytest.mark.parametrize("family_name", _EXTREME_SP_FAMILIES)
    def test_supplied_small_sp_matches_pinned_r(self, family_name: str):
        from tests.r_bridge import RBridge

        data, family, (r_family, link, theta) = _extreme_sp_case(family_name)
        bridge = RBridge(mode="rpy2")
        weight = np.ones(len(data))
        offset = np.zeros(len(data))
        checks = _AssertCollector()
        for log_sp in (-41.0, -60.0):
            sp = float(np.exp(log_sp))
            expected = bridge.public_fixed_sp_fit(
                data,
                weight,
                offset,
                r_family,
                link,
                theta=theta,
                sp=sp,
                r_epsilon=1e-7,
                r_newton_tolerance=1e-6,
            )
            res = GAM(_EXTREME_SP_FORMULA, family=family(), sp=[sp]).fit(data)
            label = f"log sp {log_sp:+.0f}"
            checks.check(
                f"{label} R inner converged",
                lambda expected=expected: check_that(
                    expected["inner_converged"],
                    f"R inner loop did not converge (outer "
                    f"{expected['outer_status']!r})",
                ),
            )
            checks.check(
                f"{label} jaxgam converged",
                lambda res=res: check_that(res.converged, res.convergence_info),
            )
            # Smoothing parameters are compared on the log scale: at sp ~ e^-41
            # STRICT's atol would absorb the whole e^-40 vs e^-41 difference.
            fields = [
                ("R log full.sp", np.log(expected["full_sp"]), [log_sp]),
                (
                    "log smoothing params",
                    np.log(res.smoothing_params),
                    np.log(expected["full_sp"]),
                ),
                ("score", res.score, expected["score"]),
                ("deviance", res.deviance, expected["deviance"]),
                ("scale", res.scale, expected["scale"]),
                ("EDF", res.edf_total, expected["edf"]),
            ]
            if theta is not None:
                fields.append(
                    (
                        "theta",
                        res.family.get_theta(transformed=True)[0],
                        expected["theta"],
                    )
                )
            for name, actual, reference in fields:
                checks.check(
                    f"{label} {name}",
                    lambda actual=actual, reference=reference: (
                        np.testing.assert_allclose(
                            actual, reference, rtol=STRICT.rtol, atol=STRICT.atol
                        )
                    ),
                )
        checks.raise_if_any(f"{family_name} supplied small sp vs pinned R")


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

    def test_linked_smoothing_id_raises_on_every_fit_route(self):
        """id= is never silently ignored: every dense and streamed route raises."""
        from jaxgam.control import FitControl
        from jaxgam.data.source import DataFrameRowSource

        rng = np.random.default_rng(SEED)
        n = 60
        data = pd.DataFrame(
            {
                "x": rng.uniform(0, 1, n),
                "z": rng.uniform(0, 1, n),
                "y": rng.normal(0, 1, n),
            }
        )
        source = DataFrameRowSource(data, response="y")
        stream = FitControl(execution="stream")
        stream_qr = FitControl(execution="stream", linear_solver="qr")
        formulas = (
            "y ~ s(x, id=1) + s(z, id=1)",
            "y ~ te(x, z, id='a')",
            "y ~ ti(x, z, id=1)",
        )
        routes = {
            "dense_reml": lambda f: GAM(f).fit(data),
            "dense_fixed_sp": lambda f: GAM(f, sp=[0.2, 0.2]).fit(data),
            "dense_efs": lambda f: GAM(f, optimizer="efs").fit(data),
            "stream_fixed_sp": lambda f: GAM(f, sp=[0.2, 0.2], control=stream).fit(
                source, result="prediction"
            ),
            "stream_exact_reml": lambda f: GAM(f, control=stream_qr).fit(
                source, result="prediction"
            ),
            "stream_efs": lambda f: GAM(f, optimizer="efs", control=stream_qr).fit(
                source, result="prediction"
            ),
        }
        collector = _AssertCollector()
        for formula in formulas:
            for route, fit in routes.items():

                def _raises(formula=formula, fit=fit) -> None:
                    try:
                        fit(formula)
                    except Exception as exc:  # the exception type is the check
                        check_that(
                            isinstance(exc, NotImplementedError)
                            and "Linked smoothing parameters (id=)" in str(exc),
                            f"raised {type(exc).__name__}: {exc}",
                        )
                    else:
                        check_that(False, "fit without raising")

                collector.check(f"{route}: {formula}", _raises)
        collector.raise_if_any("id= on public fit routes")


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


class TestInputValidation:
    """Findings M3/M4: reject unknown kwargs and wrong-length columns clearly."""

    def test_unknown_kwargs_rejected(self) -> None:
        """M3: unknown GAM kwargs raise TypeError (typos like tol=/max_iter=)."""
        with pytest.raises(TypeError, match="unexpected keyword"):
            GAM("y ~ s(x)", foo=1, max_iter=5, tol=1e-2)

    def test_supported_kwargs_accepted(self) -> None:
        """M3 guard: every genuinely-supported kwarg still constructs."""
        GAM("y ~ s(x)", family="gaussian", method="REML", device="cpu")
        GAM(
            "y ~ s(x)",
            backend="jax",
            optimizer="newton",
            select=False,
            gamma=1.0,
            knots=None,
        )

    def test_length_mismatch_names_variable(self) -> None:
        """M4: wrong-length column raises a ValueError naming the variable and
        its length, at BOTH fit and predict, instead of a cryptic NumPy error."""
        rng = np.random.RandomState(SEED)
        c = _AssertCollector()

        def _fit() -> None:
            # A bare "x" would spuriously match "axis"/"index" in the raw NumPy
            # message; require the quoted name next to its length.
            with pytest.raises(ValueError, match=r"'x'.*40"):
                GAM("y ~ s(x)").fit(
                    {"y": rng.randn(50), "x": np.linspace(0.0, 1.0, 40)}
                )

        c.check("fit_length_mismatch", _fit)

        def _predict() -> None:
            n = 60
            res = GAM("y ~ s(x) + z").fit(
                {"y": rng.randn(n), "x": np.linspace(0.0, 1.0, n), "z": rng.randn(n)}
            )
            with pytest.raises(ValueError, match=r"'z'.*7"):
                res.predict({"x": np.linspace(0.0, 1.0, 10), "z": np.zeros(7)})

        c.check("predict_length_mismatch", _predict)
        c.raise_if_any("input length validation (M4)")
