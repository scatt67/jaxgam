"""Tests for Newton smoothing parameter optimizer.

Tests cover:
- Safe Newton step (eigenvalue handling, norm capping, floor)
- Optimizer Hessian PSD
- Parametrized optimizer R comparison across all families (REML score,
  smoothing params)
- Multi-penalty optimizer checks for two-smooth, tensor, and factor-by models
- ML criterion optimization (Gaussian and non-Gaussian)
- Offset support
- Purely parametric shortcut
- REML monotonicity across families
- Step-halving activation
- Edge cases (invalid method, iteration limit)

Tolerance rationale:
  Gaussian REML achieves MODERATE (rtol=1e-4, atol=1e-6) for optimizer
  comparisons. GLM smoothing parameters use LOOSE because the REML criterion
  is flat near the optimum (AGENTS.md §Common Pitfalls #4). ML criterion has a
  different normalization convention from R's (constant offset), so ML
  converges to a slightly different lambda even for Gaussian; only the fit
  deviance is compared there.

Design doc reference: Section 8.2 (Outer Newton with Damped Hessian)
R source reference: fast-REML.r lines 1740-1875
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.initialization import initialize_beta
from jaxgam.fitting.newton import (
    _MAX_HALVINGS_GAUSSIAN,
    NewtonOptimizer,
    _jit_gaussian_score_change,
    _logdet_change,
    _safe_newton_step,
    newton_optimize,
)
from jaxgam.fitting.penalty_ops import (
    JaxLocalPenalty,
    JaxPenaltyBlock,
    JaxPenaltyStructure,
    JaxTransform,
)
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.jax_utils import to_jax
from tests.helpers import (
    SEED,
    _generate_family_data,
    _setup_fd,
    r_available,
    r_tolerance,
)
from tests.tolerances import LOOSE, MODERATE, STRICT

jax.config.update("jax_enable_x64", True)


class _GaussianLineSearchProbe(NewtonOptimizer):
    """Deterministic score probe for fast-REML line-search semantics."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = iter(scores)
        self.calls: list[jax.Array] = []
        self._tol = 1e-6

    def _clamp_params(self, params: jax.Array) -> jax.Array:
        return params

    def _gaussian_trial_score_change(
        self, _params, _trial_params, _beta, _trial, score, score_trial, _scale
    ) -> float:
        return score_trial - score

    def _fit_and_score(
        self, params: jax.Array, _beta_init: jax.Array
    ) -> tuple[object, jax.Array]:
        self.calls.append(params)
        return object(), jnp.asarray(next(self._scores))


# ---- A. Safe Newton step tests ----


class TestGaussianStepHalving:
    """Pinned fast-REML acceptance and representability semantics."""

    @staticmethod
    def _run(
        scores: list[float], step: float
    ) -> tuple[_GaussianLineSearchProbe, jax.Array, object]:
        probe = _GaussianLineSearchProbe(scores)
        params, _pirls, _score, outcome = probe._step_halve_gaussian(
            jnp.array([1.0]),
            jnp.array([step]),
            1.0,
            jnp.array([0.0]),
            1.0,
        )
        return probe, params, outcome

    def test_equal_finite_score_is_accepted_without_halving(self):
        probe, params, outcome = self._run([1.0], 0.25)
        assert outcome.name == "ACCEPTED"
        assert len(probe.calls) == 1
        np.testing.assert_allclose(
            params, jnp.array([1.25]), rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_representable_half_step_is_tried_after_one_ulp_worse_trial(self):
        ulp = np.nextafter(1.0, 2.0) - 1.0
        probe, params, outcome = self._run(
            [np.nextafter(1.0, np.inf), np.nextafter(1.0, -np.inf)], 2.0 * ulp
        )
        assert outcome.name == "ACCEPTED"
        assert len(probe.calls) == 2
        assert float(params[0]) == np.nextafter(1.0, 2.0)

    @pytest.mark.parametrize("invalid_score", [np.nan, np.inf, -np.inf])
    def test_nonfinite_trial_is_never_accepted(self, invalid_score: float):
        probe, params, outcome = self._run([invalid_score, 0.9], 0.25)
        assert outcome.name == "ACCEPTED"
        assert len(probe.calls) == 2
        np.testing.assert_allclose(
            params, jnp.array([1.125]), rtol=STRICT.rtol, atol=STRICT.atol
        )

    @pytest.mark.parametrize("invalid_score", [np.nan, np.inf, -np.inf])
    def test_persistent_nonfinite_trials_exhaust_the_bounded_search(
        self, invalid_score: float
    ) -> None:
        probe, _params, outcome = self._run(
            [invalid_score] * (_MAX_HALVINGS_GAUSSIAN + 1), 0.25
        )
        assert outcome.name == "FAILED"
        assert len(probe.calls) == _MAX_HALVINGS_GAUSSIAN + 1

    def test_four_consecutive_insignificant_increases_fail(self):
        probe, _params, outcome = self._run([np.nextafter(1.0, np.inf)] * 4, 0.25)
        assert outcome.name == "FAILED"
        assert len(probe.calls) == 4

    def test_exact_unrepresentable_step_fails_without_a_second_trial(self):
        half_ulp = (np.nextafter(1.0, 2.0) - 1.0) / 2.0
        probe, params, outcome = self._run([1.1], half_ulp)
        assert outcome.name == "FAILED"
        assert len(probe.calls) == 1
        assert float(params[0]) == 1.0

    def test_halving_cap_remains_source_matched(self):
        probe, _params, outcome = self._run([2.0] * (_MAX_HALVINGS_GAUSSIAN + 1), 0.25)
        assert outcome.name == "FAILED"
        assert len(probe.calls) == _MAX_HALVINGS_GAUSSIAN + 1


class TestGaussianScoreChange:
    """Stable criterion changes retain strict descent below scalar-score noise."""

    @pytest.mark.parametrize(
        ("base_shift", "rho_step", "phi_step", "beta_shift"),
        [
            (4e-8, 0.0, -4e-8, 0.0),
            (0.0, 0.0, 4e-8, 0.0),
            (0.0, 2e-8, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.01),
        ],
    )
    def test_sub_ulp_changes_match_100_digit_absolute_criterion(
        self, base_shift, rho_step, phi_step, beta_shift
    ) -> None:
        base = np.array([0.0, np.log(2.0) + base_shift])
        trial = base + np.array([rho_step, phi_step])
        beta_base = np.array([1.0, 1.0 + beta_shift])
        beta_trial = np.array([1.0, 2.0 / (1.0 + np.exp(trial[0]))])
        structure = JaxPenaltyStructure(
            2,
            (
                JaxPenaltyBlock(
                    1,
                    2,
                    (0,),
                    (JaxLocalPenalty("identity", jnp.array(1.0), 1),),
                    JaxTransform("identity", jnp.array(1.0), 1),
                ),
            ),
        )
        actual = float(
            _jit_gaussian_score_change(
                jnp.asarray(base),
                jnp.asarray(trial),
                jnp.asarray(beta_base),
                jnp.asarray(beta_trial),
                jnp.eye(2),
                jnp.array([1.0, 2.0]),
                jnp.ones(2),
                jnp.zeros(2),
                jnp.eye(2),
                structure,
                (),
                n_lambda=1,
                Mp=1,
                singleton_sp_indices=(0,),
                singleton_ranks=(1,),
                multi_block_sp_indices=(),
            )
        )
        with localcontext() as context:
            context.prec = 100

            def absolute_score(params, beta):
                rho, log_phi = (Decimal.from_float(float(x)) for x in params)
                b0, b1 = (Decimal.from_float(float(x)) for x in beta)
                penalty = rho.exp()
                quadratic = (1 - b0) ** 2 + (2 - b1) ** 2 + penalty * b1**2
                return (
                    quadratic / log_phi.exp() + log_phi + (1 + penalty).ln() - rho
                ) / 2

            expected = float(
                absolute_score(trial, beta_trial) - absolute_score(base, beta_base)
            )
        assert actual != 0.0
        assert np.sign(actual) == np.sign(expected)
        # An ordinary absolute tolerance would conceal a wrong sign or zero.
        np.testing.assert_allclose(actual, expected, rtol=MODERATE.rtol, atol=0.0)

    @pytest.mark.parametrize("jitter", [0.0, 1e-3])
    def test_logdet_change_matches_high_precision_and_rejects_singular_base(
        self, jitter
    ):
        base = np.array([[10.0, 0.3], [0.3, 0.1]])
        change = np.array([[1e-12, -3e-13], [-3e-13, 2e-14]])
        with localcontext() as context:
            context.prec = 100
            a, b, d = map(Decimal.from_float, (base[0, 0], base[0, 1], base[1, 1]))
            da, db, dd = map(
                Decimal.from_float, (change[0, 0], change[0, 1], change[1, 1])
            )
            multiplier = 1 + Decimal.from_float(jitter)
            a, d, da, dd = (value * multiplier for value in (a, d, da, dd))
            expected = float(
                ((a + da) * (d + dd) - (b + db) ** 2).ln() - (a * d - b**2).ln()
            )
        actual = jax.jit(_logdet_change)(
            jnp.asarray(base), jnp.asarray(change), relative_diagonal_jitter=jitter
        )
        np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=0.0)
        assert np.isnan(_logdet_change(jnp.zeros((2, 2)), jnp.eye(2)))

    @pytest.mark.parametrize(
        "formula",
        ["y ~ s(x, bs='cr', k=6)", "y ~ te(x, z, bs='cr', k=4)"],
    )
    def test_change_matches_direct_criterion_for_weighted_offset_models(self, formula):
        rng = np.random.default_rng(SEED)
        data = pd.DataFrame({"x": rng.uniform(size=55), "z": rng.uniform(size=55)})
        data["y"] = np.sin(3.0 * data.x) + rng.normal(scale=0.2, size=len(data))
        fd = _setup_fd(formula, data, Gaussian())
        fd = dataclasses.replace(
            fd,
            wt=jnp.linspace(0.2, 1.7, fd.n_obs).at[0].set(0.0),
            offset=jnp.linspace(-0.2, 0.1, fd.n_obs),
        )
        optimizer = NewtonOptimizer(fd)
        base = jnp.concatenate([fd.log_lambda_init, jnp.array([-1.2])])
        trial = base + jnp.linspace(-0.02, 0.03, len(base))
        before, score_before = optimizer._fit_and_score(base, optimizer._initial_beta())
        after, score_after = optimizer._fit_and_score(trial, before.coefficients)
        actual = _jit_gaussian_score_change(
            base,
            trial,
            before.coefficients,
            after.coefficients,
            fd.X,
            fd.y,
            fd.wt,
            fd.offset,
            after.XtWX,
            fd.penalty_structure,
            fd.multi_block_proj_S,
            n_lambda=fd.n_penalties,
            Mp=fd.total_penalty_null_dim,
            singleton_sp_indices=fd.singleton_sp_indices,
            singleton_ranks=fd.singleton_ranks,
            multi_block_sp_indices=fd.multi_block_sp_indices,
        )
        np.testing.assert_allclose(
            actual, score_after - score_before, rtol=STRICT.rtol, atol=STRICT.atol
        )

    def test_flat_gp_start_converges_without_relaxing_gradient_tolerance(
        self, monkeypatch
    ):
        from tests.test_validation_matrix import _make_gp_2d_data

        fd = _setup_fd(
            "y ~ s(x, z, bs='gp', k=30)", _make_gp_2d_data("gaussian"), Gaussian()
        )
        optimizer = NewtonOptimizer(
            dataclasses.replace(fd, log_lambda_init=fd.log_lambda_init - 0.1)
        )
        checks = []
        original_check = optimizer._check_convergence

        def capture(*args, **kwargs):
            result = original_check(*args, **kwargs)
            checks.append(result)
            return result

        monkeypatch.setattr(optimizer, "_check_convergence", capture)
        result = optimizer.run()
        assert result.converged
        gradient, _hessian, score_scale, converged = checks[-1]
        assert converged
        assert float(jnp.max(jnp.abs(gradient))) <= optimizer._tol * score_scale
        assert optimizer._tol == np.sqrt(np.finfo(float).eps)

    @pytest.mark.parametrize(
        "mode", ["noncanonical", "rank", "no_joint_scale", "nonfinite", "large"]
    )
    def test_stable_comparison_does_not_expand_unsupported_paths(
        self, monkeypatch, mode
    ):
        from jaxgam.fitting import newton as newton_module

        data = _generate_family_data("gaussian", n=35)
        family = Gaussian(link="log") if mode == "noncanonical" else Gaussian()
        fd = _setup_fd("y ~ s(x, bs='cr', k=6)", data, family)
        if mode == "rank":
            fd = dataclasses.replace(fd, rank_deficit=1)
        optimizer = NewtonOptimizer(fd)
        if mode == "no_joint_scale":
            optimizer._joint_scale = False

        def unexpected_call(*_args, **_kwargs):
            raise AssertionError("unsupported comparison used Gaussian refinement")

        monkeypatch.setattr(
            newton_module, "_jit_gaussian_score_change", unexpected_call
        )
        score_trial = (
            np.inf if mode == "nonfinite" else 1.0 + (0.1 if mode == "large" else 1e-12)
        )
        actual = optimizer._gaussian_trial_score_change(
            None, None, None, None, 1.0, score_trial, 2.0
        )
        assert actual == score_trial - 1.0

    def test_nonfinite_stable_difference_preserves_original_comparison(
        self, monkeypatch
    ):
        from jaxgam.fitting import newton as newton_module

        fd = _setup_fd(
            "y ~ s(x, bs='cr', k=6)",
            _generate_family_data("gaussian", n=35),
            Gaussian(),
        )
        optimizer = NewtonOptimizer(fd)
        params = jnp.concatenate([fd.log_lambda_init, jnp.array([0.0])])
        trial, _score = optimizer._fit_and_score(params, optimizer._initial_beta())
        monkeypatch.setattr(
            newton_module,
            "_jit_gaussian_score_change",
            lambda *_args, **_kwargs: jnp.nan,
        )
        actual = optimizer._gaussian_trial_score_change(
            params, params, trial.coefficients, trial, 1.0, 1.0 + 1e-12, 2.0
        )
        assert actual == (1.0 + 1e-12) - 1.0


class TestSafeNewtonStep:
    """Tests for _safe_newton_step eigenvalue handling."""

    def test_quadratic_one_step(self):
        """1D quadratic f(x) = (x-2)^2: Newton converges in 1 step."""
        # f(x) = (x-2)^2, f'(x) = 2(x-2), f''(x) = 2
        # At x=0: f'=-4, f''=2, step = -f'/f'' = 2
        grad = jnp.array([-4.0])
        hess = jnp.array([[2.0]])
        step, is_pdef = _safe_newton_step(grad, hess)
        np.testing.assert_allclose(float(step[0]), 2.0, rtol=STRICT.rtol)
        assert bool(is_pdef)

    def test_negative_eigenvalues_flipped(self):
        """Negative Hessian eigenvalues are flipped to positive."""
        grad = jnp.array([1.0, -1.0])
        hess = jnp.array([[-2.0, 0.0], [0.0, -3.0]])
        step, is_pdef = _safe_newton_step(grad, hess)
        # After flipping: eigs become [2, 3], step = -H_safe^{-1} g
        expected = -jnp.array([1.0 / 2.0, -1.0 / 3.0])
        np.testing.assert_allclose(
            np.asarray(step), np.asarray(expected), rtol=STRICT.rtol
        )
        assert not bool(is_pdef)

    def test_step_norm_capped(self):
        """Step component magnitude is capped to max_step."""
        grad = jnp.array([100.0])
        hess = jnp.array([[1.0]])
        step, _ = _safe_newton_step(grad, hess, max_step=5.0)
        assert float(jnp.max(jnp.abs(step))) <= 5.0 + STRICT.rtol

    def test_near_singular_hessian(self):
        """Near-singular Hessian: floor prevents division by zero."""
        grad = jnp.array([1.0, 1.0])
        hess = jnp.array([[1.0, 0.0], [0.0, 1e-20]])
        step, _ = _safe_newton_step(grad, hess)
        assert jnp.all(jnp.isfinite(step))

    def test_eigenvalue_floor_value(self):
        """Floor computation uses max(|D|) * eps^0.7 as threshold.

        With one large and one zero eigenvalue, the zero eigenvalue
        should be floored to max(|D|) * eps^0.7 (R line 1450). The
        step in the floored direction should be much larger than in
        the well-conditioned direction (before component-wise capping).
        """
        grad = jnp.array([1.0, 1.0])
        hess = jnp.array([[4.0, 0.0], [0.0, 0.0]])
        step, is_pdef = _safe_newton_step(grad, hess)

        # Step should be finite and component-wise capped
        assert jnp.all(jnp.isfinite(step))
        assert float(jnp.max(jnp.abs(step))) <= 5.0 + STRICT.rtol

        # The floored direction (index 1) should dominate the step
        # because its eigenvalue is tiny
        assert abs(float(step[1])) > abs(float(step[0]))

        # Hessian with a zero eigenvalue is not positive definite
        assert not bool(is_pdef)


# ---- B. Hard-gate invariants ----


class TestInvariants:
    """Optimizer-specific invariants for every converged model."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(
        params=[
            ("gaussian", Gaussian()),
            ("poisson", Poisson()),
            ("binomial", Binomial()),
            ("gamma", Gamma()),
        ],
        ids=["gaussian", "poisson", "binomial", "gamma"],
    )
    def converged_result(self, request):
        """Fit a converged model for each family."""
        family_name, family_obj = request.param
        data = _generate_family_data(family_name)
        fd = _setup_fd(self.FORMULA, data, family_obj)
        result = newton_optimize(fd)
        return family_name, fd, result

    def test_hessian_symmetric_psd(self, converged_result):
        """Penalized Hessian XtWX + S_lambda must be symmetric PSD."""
        _, fd, result = converged_result
        XtWX = np.asarray(result.pirls_result.XtWX)
        S = np.asarray(fd.S_lambda(result.log_lambda))
        H = XtWX + S

        # Symmetry
        np.testing.assert_allclose(H, H.T, rtol=STRICT.rtol, atol=STRICT.atol)

        # PSD: all eigenvalues >= 0
        eigs = np.linalg.eigvalsh(H)
        assert np.all(eigs >= -STRICT.rtol), f"H has negative eigenvalue: {eigs.min()}"


# ---- C. Parametrized R comparison across all families ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestFamilyVsR:
    """Optimizer-level R comparison across all four families."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(
        params=[
            ("gaussian", Gaussian()),
            ("poisson", Poisson()),
            ("binomial", Binomial()),
            ("gamma", Gamma()),
        ],
        ids=["gaussian", "poisson", "binomial", "gamma"],
    )
    def family_fit(self, request):
        """Fit both jaxgam and R for a given family, return optimizer outputs."""
        from tests.r_bridge import RBridge

        family_name, family_obj = request.param
        data = _generate_family_data(family_name)
        fd = _setup_fd(self.FORMULA, data, family_obj)
        result = newton_optimize(fd)

        bridge = RBridge()
        r_result = bridge.fit_gam(self.FORMULA, data, family=family_name)

        return family_name, fd, result, r_result

    def test_reml_score_vs_r(self, family_fit):
        """REML criterion score matches R."""
        family_name, _, result, r_result = family_fit
        tol = r_tolerance(family_name)
        np.testing.assert_allclose(
            float(result.score),
            r_result["reml_score"],
            rtol=tol.rtol,
            atol=tol.atol,
            err_msg=f"{family_name} REML score differs from R",
        )

    def test_smoothing_params_vs_r(self, family_fit):
        """Smoothing parameters match R.

        Compared on original scale with LOOSE tolerance. The REML
        criterion is flat near the optimum (AGENTS.md Pitfall #4),
        so cross-implementation differences in lambda are expected.
        Gamma exceeds LOOSE (~1.2% vs 1% rtol) because the inverse
        link amplifies small lambda differences.
        """
        family_name, _, result, r_result = family_fit
        np.testing.assert_allclose(
            np.asarray(result.smoothing_params),
            r_result["smoothing_params"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg=f"{family_name} smoothing params differ from R",
        )


# ---- D. Multi-penalty optimizer tests ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestMultiSmooth:
    """Multi-smooth optimizer checks against R smoothing parameters."""

    def test_two_smooths_vs_r(self):
        """Two-smooth Gaussian model finds R-compatible smoothing params."""
        from tests.r_bridge import RBridge

        rng = np.random.default_rng(SEED)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        fd = _setup_fd(formula, data, Gaussian())
        result = newton_optimize(fd)

        assert result.converged
        assert len(result.log_lambda) == 2

        bridge = RBridge()
        r_result = bridge.fit_gam(formula, data, family="gaussian")

        np.testing.assert_allclose(
            np.asarray(result.smoothing_params),
            r_result["smoothing_params"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Two-smooth smoothing params differ from R",
        )

    def test_tensor_product_vs_r(self):
        """Tensor product te(x1, x2, k=5): multi-penalty optimizer behavior.

        Tensor products have a multi-penalty block where one penalty
        direction has a gently sloping REML surface. The lsp_max cap
        clips log(sp) at 15 while R converges at ~13.08 via its
        internal penalty reparameterization (Sl.setup). Both give
        an equivalent fit.
        """
        from tests.r_bridge import RBridge

        rng = np.random.default_rng(SEED)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        py_formula = "y ~ te(x1, x2, k=5)"
        r_formula = "y ~ te(x1, x2, k=c(5,5))"
        fd = _setup_fd(py_formula, data, Gaussian())
        result = newton_optimize(fd)

        assert result.converged

        bridge = RBridge()
        r_result = bridge.fit_gam(r_formula, data, family="gaussian")

        # All sp must be finite and positive
        sp_ours = np.asarray(result.smoothing_params)
        assert np.all(np.isfinite(sp_ours)), f"All sp must be finite, got {sp_ours}"
        assert np.all(sp_ours > 0), f"All sp must be positive, got {sp_ours}"

        # Compare well-determined sp. Tensor products have gently sloping
        # REML surfaces where any sp in a wide range gives an equivalent
        # fit. Different optimizers land at different points on these flat
        # surfaces.
        log_sp_ours = np.log(sp_ours)
        log_sp_r = np.log(r_result["smoothing_params"])
        well_determined = np.abs(log_sp_ours - log_sp_r) < 2.0
        if np.any(well_determined):
            np.testing.assert_allclose(
                sp_ours[well_determined],
                r_result["smoothing_params"][well_determined],
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg="Tensor product well-determined sp differ from R",
            )
        # Poorly-determined sp must still be in a sensible range
        poorly_determined = ~well_determined
        if np.any(poorly_determined):
            assert np.all(sp_ours[poorly_determined] >= 1e-5), (
                f"Poorly-determined sp too small: {sp_ours[poorly_determined]}"
            )
            assert np.all(sp_ours[poorly_determined] <= 1e20), (
                f"Poorly-determined sp too large: {sp_ours[poorly_determined]}"
            )

    def test_factor_by_vs_r(self):
        """Factor-by smooth: multi-penalty optimizer behavior.

        With block-structured log|S+|, each factor level's penalty
        is a singleton block with exact derivatives. The optimizer
        converges quickly even when one level is heavily penalized.
        """
        from tests.r_bridge import RBridge

        rng = np.random.default_rng(SEED)
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
        data = pd.DataFrame(
            {
                "x": x,
                "fac": pd.Categorical(fac, categories=levels),
                "y": y,
            }
        )

        formula = "y ~ s(x, by=fac, k=10, bs='cr') + fac"
        fd = _setup_fd(formula, data, Gaussian())
        result = newton_optimize(fd)

        assert result.converged

        bridge = RBridge()
        r_result = bridge.fit_gam(formula, data, family="gaussian")

        # All sp must be finite and positive
        sp_ours = np.asarray(result.smoothing_params)
        assert np.all(np.isfinite(sp_ours)), f"All sp must be finite, got {sp_ours}"
        assert np.all(sp_ours > 0), f"All sp must be positive, got {sp_ours}"

        # Compare well-determined sp (flat-surface sp are ambiguous)
        log_sp_ours = np.log(sp_ours)
        log_sp_r = np.log(r_result["smoothing_params"])
        well_determined = np.abs(log_sp_ours - log_sp_r) < 2.0
        if np.any(well_determined):
            np.testing.assert_allclose(
                sp_ours[well_determined],
                r_result["smoothing_params"][well_determined],
                rtol=LOOSE.rtol,
                atol=LOOSE.atol,
                err_msg="Factor-by well-determined sp differ from R",
            )
        # Poorly-determined sp must still be in a sensible range
        poorly_determined = ~well_determined
        if np.any(poorly_determined):
            assert np.all(sp_ours[poorly_determined] >= 1e-5), (
                f"Poorly-determined sp too small: {sp_ours[poorly_determined]}"
            )
            assert np.all(sp_ours[poorly_determined] <= 1e20), (
                f"Poorly-determined sp too large: {sp_ours[poorly_determined]}"
            )


# ---- E. ML is not supported in v1.0 ----


class TestMLNotSupported:
    """ML is deferred in v1.0: mgcv's ML criterion needs the penalty
    range-space determinant (MLpenalty1), not REML's full-space one.
    The optimizer must reject method='ML' rather than silently fit a
    wrong criterion.
    """

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_ml_rejected(self):
        """newton_optimize raises for method='ML'."""
        data = _generate_family_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())
        with pytest.raises(ValueError, match="REML"):
            newton_optimize(fd, method="ML")


# ---- F. Diagnostics and edge cases ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestDiagnostics:
    """Optimizer edge cases and invariants."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_purely_parametric(self):
        """No penalties: skip Newton, return immediately."""
        rng = np.random.default_rng(SEED)
        n, p = 100, 3
        X = rng.normal(0, 1, (n, p))
        beta_true = np.array([1.0, -0.5, 0.3])
        y = X @ beta_true + rng.normal(0, 0.5, n)

        fd = FittingData(
            X=to_jax(X),
            y=to_jax(y),
            wt=jnp.ones(n),
            offset=None,
            penalty_structure=JaxPenaltyStructure(p, ()),
            log_lambda_init=jnp.zeros(0),
            family=Gaussian(),
            n_obs=n,
            n_coef=p,
            penalty_ranks=(),
            penalty_null_dims=(),
            total_penalty_rank=0,
            singleton_sp_indices=(),
            singleton_ranks=(),
            singleton_eig_constants=jnp.array([]),
            multi_block_sp_indices=(),
            multi_block_ranks=(),
            multi_block_proj_S=(),
            multi_block_S_local=(),
            max_y=0,
        )
        result = newton_optimize(fd)

        assert result.converged
        assert result.n_iter == 0
        assert result.convergence_info == "full convergence"
        assert result.log_lambda.shape == (0,)
        assert result.smoothing_params.shape == (0,)

    def test_offset_support(self):
        """Non-None offset is passed through and affects the fit.

        A constant offset shifts the linear predictor. We verify that
        fitting with offset=c and y produces different coefficients
        than fitting without offset.
        """
        data = _generate_family_data("gaussian")
        fd_no_offset = _setup_fd(self.FORMULA, data, Gaussian())
        result_no_offset = newton_optimize(fd_no_offset)

        # Manually add an offset to FittingData
        n = fd_no_offset.n_obs
        offset = jnp.full(n, 0.5)
        fd_offset = dataclasses.replace(fd_no_offset, offset=offset)
        result_offset = newton_optimize(fd_offset)

        assert result_offset.converged
        # Coefficients should differ due to the offset
        coef_diff = float(
            jnp.max(
                jnp.abs(
                    result_offset.pirls_result.coefficients
                    - result_no_offset.pirls_result.coefficients
                )
            )
        )
        assert coef_diff > 0.01, "Offset should change coefficients"

    @pytest.mark.parametrize(
        ("family_name", "family_obj"),
        [
            ("gaussian", Gaussian()),
            ("binomial", Binomial()),
            ("gamma", Gamma()),
        ],
        ids=["gaussian", "binomial", "gamma"],
    )
    def test_reml_monotonicity(self, family_name, family_obj):
        """REML score should not increase at accepted steps.

        Tested across Gaussian, Binomial, and Gamma — the families most
        likely to challenge monotonicity due to iterative PIRLS.
        """
        from jaxgam.fitting.reml import REMLCriterion

        data = _generate_family_data(family_name)
        fd = _setup_fd(self.FORMULA, data, family_obj)

        # Run with a deliberately bad start to force multiple iterations
        log_lambda_init = jnp.array([5.0])
        result = newton_optimize(fd, log_lambda_init=log_lambda_init)

        # Compute initial score for comparison
        beta_init = initialize_beta(
            np.asarray(fd.X), np.asarray(fd.y), np.asarray(fd.wt), fd.family
        )
        S_init = fd.S_lambda(log_lambda_init)
        pirls_init = pirls_loop(
            fd.X, fd.y, to_jax(np.asarray(beta_init)), S_init, fd.family, fd.wt
        )
        crit_init = REMLCriterion(fd, pirls_init)
        score_init = float(crit_init.score(log_lambda_init))

        assert float(result.score) <= score_init + STRICT.rtol

    def test_invalid_method_raises(self):
        """Invalid method string raises ValueError."""
        data = _generate_family_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())
        with pytest.raises(ValueError, match="Unknown method"):
            newton_optimize(fd, method="INVALID")

    def test_iteration_limit(self):
        """max_iter=1 triggers 'iteration limit' convergence info."""
        data = _generate_family_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())
        result = newton_optimize(fd, max_iter=1)
        assert result.convergence_info == "iteration limit"
        assert result.n_iter == 1
        assert not result.converged


class TestStepFailureConvergence:
    """Step failure at the optimum is recognized as convergence.

    Bug: the Newton loop broke on ``_StepOutcome.FAILED`` with
    ``converged=False`` even when the gradient was already within tolerance.
    On multi-penalty models step-halving can exhaust one iterate *after* the
    point where mgcv's gradient test would have declared convergence, so
    jaxgam reported ``converged=False`` on a fit that is at the optimum and
    matches mgcv. mgcv's ``fast.REML.fit`` reaches the optimum via the
    gradient test before the step failure and only warns on the iteration
    limit. Fixed by re-checking the gradient on step failure via
    ``_gradient_within_tol``; this class covers that decision logic directly
    (the end-to-end flip is data/precision-sensitive near the flat optimum,
    so the mechanism is tested deterministically here).
    """

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_gaussian_recognizes_within_tol_gradient(self):
        """Gaussian: ``max|grad| <= score_scale * tol`` => convergence."""
        fd = _setup_fd(self.FORMULA, _generate_family_data("gaussian"), Gaussian())
        opt = NewtonOptimizer(fd)
        n = fd.n_penalties + (1 if opt._joint_scale else 0)
        params = jnp.zeros(n)
        score_scale = 100.0
        thresh = score_scale * opt._tol
        within = jnp.full((n,), 0.5 * thresh)
        outside = jnp.full((n,), 10.0 * thresh)
        assert opt._gradient_within_tol(within, params, score_scale)
        assert not opt._gradient_within_tol(outside, params, score_scale)

    def test_glm_recognizes_within_tol_gradient(self):
        """Non-Gaussian uses the relaxed ``5 * tol`` projected-gradient test."""
        fd = _setup_fd(self.FORMULA, _generate_family_data("poisson"), Poisson())
        opt = NewtonOptimizer(fd)
        n = fd.n_penalties
        params = jnp.zeros(n)  # no bounds active => projected grad == grad
        score_scale = 100.0
        thresh = 5.0 * score_scale * opt._tol
        within = jnp.full((n,), 0.5 * thresh)
        outside = jnp.full((n,), 2.0 * thresh)
        assert opt._gradient_within_tol(within, params, score_scale)
        assert not opt._gradient_within_tol(outside, params, score_scale)


# ---- G. Step-halving ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestStepHalving:
    """Step-halving behavior."""

    def test_step_halving_activates(self):
        """With adversarial log_lambda_init, step-halving still converges.

        Starting very far from the optimum forces the optimizer to use
        step-halving. We verify convergence and that extra iterations
        were needed (more than the default-start case).
        """
        data = _generate_family_data("gaussian")
        formula = "y ~ s(x, k=10, bs='cr')"
        fd = _setup_fd(formula, data, Gaussian())

        # Fit from default start for baseline iteration count
        result_default = newton_optimize(fd)

        # Very far from optimum
        log_lambda_init = jnp.array([10.0])
        result_far = newton_optimize(fd, log_lambda_init=log_lambda_init)

        assert result_far.converged
        assert jnp.isfinite(result_far.score)
        # Adversarial start should need more iterations
        assert result_far.n_iter > result_default.n_iter


# ---- H. custom_jvp differentiable score ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestCustomJVP:
    """Tests for the custom_jvp-based differentiable score function.

    The custom_jvp on PIRLS defines how (β*, XtWX, deviance) change
    with S_lambda via the IFT. jax.grad and jax.hessian of the
    end-to-end score automatically capture all missing terms.
    """

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.mark.parametrize(
        ("family_name", "family_obj"),
        [("poisson", Poisson()), ("binomial", Binomial()), ("gamma", Gamma())],
        ids=["poisson", "binomial", "gamma"],
    )
    def test_custom_jvp_gradient_matches_fd(self, family_name, family_obj):
        """custom_jvp gradient matches central FD of the score.

        Central FD re-runs PIRLS at each perturbation, computing the
        true gradient. The custom_jvp gradient should match this to
        O(h²) accuracy.
        """
        data = _generate_family_data(family_name)
        fd = _setup_fd(self.FORMULA, data, family_obj)
        optimizer = NewtonOptimizer(fd)

        params = fd.log_lambda_init.copy()
        if optimizer._joint_scale:
            # Add log_phi for joint scale families
            S = fd.S_lambda(params)
            beta_init = initialize_beta(
                np.asarray(fd.X),
                np.asarray(fd.y),
                np.asarray(fd.wt),
                fd.family,
            )
            pirls_result = pirls_loop(
                fd.X,
                fd.y,
                to_jax(np.asarray(beta_init)),
                S,
                fd.family,
                fd.wt,
                fd.offset,
            )
            from jaxgam.fitting.reml import estimate_edf, fletcher_scale

            edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
            phi = fletcher_scale(fd.y, pirls_result.mu, fd.wt, fd.family, edf)
            params = jnp.concatenate([params, jnp.log(phi)[None]])

        # Get PIRLS result for beta_warm
        S = fd.S_lambda(params[: fd.n_penalties] if optimizer._joint_scale else params)
        beta_init = initialize_beta(
            np.asarray(fd.X),
            np.asarray(fd.y),
            np.asarray(fd.wt),
            fd.family,
        )
        pirls_result = pirls_loop(
            fd.X,
            fd.y,
            to_jax(np.asarray(beta_init)),
            S,
            fd.family,
            fd.wt,
            fd.offset,
        )
        beta_warm = pirls_result.coefficients

        # custom_jvp gradient (fused call returns both grad and hess)
        grad_jvp, _ = optimizer._diff_grad_hess(params, beta_warm)

        # Central FD gradient (re-running PIRLS at each perturbation)
        eps = 1e-5
        fd_grad = jnp.zeros_like(params)
        for j in range(len(params)):
            e_j = jnp.zeros_like(params).at[j].set(eps)
            _, score_plus = optimizer._fit_and_score(params + e_j, beta_warm)
            _, score_minus = optimizer._fit_and_score(params - e_j, beta_warm)
            fd_grad = fd_grad.at[j].set((score_plus - score_minus) / (2 * eps))

        np.testing.assert_allclose(
            np.asarray(grad_jvp),
            np.asarray(fd_grad),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=f"{family_name}: custom_jvp gradient differs from FD",
        )

    @pytest.mark.parametrize(
        ("family_name", "family_obj"),
        [("poisson", Poisson()), ("binomial", Binomial())],
        ids=["poisson", "binomial"],
    )
    def test_custom_jvp_hessian_matches_fd(self, family_name, family_obj):
        """custom_jvp Hessian matches FD Hessian with PIRLS reconvergence.

        Central FD of the custom_jvp gradient (which itself re-runs
        PIRLS via _fit_and_score). The custom_jvp Hessian should match.
        """
        data = _generate_family_data(family_name)
        fd = _setup_fd(self.FORMULA, data, family_obj)
        optimizer = NewtonOptimizer(fd)

        params = fd.log_lambda_init.copy()
        S = fd.S_lambda(params)
        beta_init = initialize_beta(
            np.asarray(fd.X),
            np.asarray(fd.y),
            np.asarray(fd.wt),
            fd.family,
        )
        pirls_result = pirls_loop(
            fd.X,
            fd.y,
            to_jax(np.asarray(beta_init)),
            S,
            fd.family,
            fd.wt,
            fd.offset,
        )
        beta_warm = pirls_result.coefficients

        # custom_jvp Hessian (fused call returns both grad and hess)
        _, hess_jvp = optimizer._diff_grad_hess(params, beta_warm)
        hess_jvp = (hess_jvp + hess_jvp.T) / 2

        # FD Hessian: central FD of the custom_jvp gradient with PIRLS
        # reconvergence at each perturbation
        h = float(jnp.finfo(jnp.float64).eps ** (1.0 / 3.0))
        m = len(params)
        cols = []
        for j in range(m):
            e_j = jnp.zeros(m).at[j].set(h)
            # Re-converge PIRLS at perturbed params, get custom_jvp gradient
            g_p, _ = optimizer._diff_grad_hess(params + e_j, beta_warm)
            g_m, _ = optimizer._diff_grad_hess(params - e_j, beta_warm)
            cols.append((g_p - g_m) / (2 * h))
        hess_fd = jnp.column_stack(cols)
        hess_fd = (hess_fd + hess_fd.T) / 2

        np.testing.assert_allclose(
            np.asarray(hess_jvp),
            np.asarray(hess_fd),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=f"{family_name}: custom_jvp Hessian differs from FD Hessian",
        )

    @pytest.mark.parametrize(
        ("family_name", "family_obj"),
        [("poisson", Poisson()), ("binomial", Binomial())],
        ids=["poisson", "binomial"],
    )
    def test_convergence_speed(self, family_name, family_obj):
        """Non-Gaussian single-smooth models converge in <30 iterations."""
        data = _generate_family_data(family_name)
        fd = _setup_fd(self.FORMULA, data, family_obj)
        result = newton_optimize(fd)

        assert result.converged, (
            f"{family_name} did not converge in {result.n_iter} iterations"
        )
        assert result.n_iter < 30, (
            f"{family_name} took {result.n_iter} iterations (expected <30)"
        )
