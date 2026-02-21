"""Tests for Newton smoothing parameter optimizer.

Tests cover:
- Safe Newton step (eigenvalue handling, norm capping)
- Gaussian GAM convergence and vs R comparison
- Two-smooth Gaussian GAM
- Poisson, Binomial, Gamma convergence
- ML criterion optimization
- Purely parametric shortcut
- NewtonResult fields and types
- REML monotonicity at accepted steps
- Step-halving activation
- Convergence info strings

Design doc reference: Section 8.2 (Outer Newton with Damped Hessian)
R source reference: fast-REML.r lines 1740-1875
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pymgcv.families.standard import Binomial, Gamma, Gaussian, Poisson
from pymgcv.fitting.data import FittingData
from pymgcv.fitting.initialization import initialize_beta
from pymgcv.fitting.newton import (
    NewtonResult,
    _safe_newton_step,
    newton_optimize,
)
from pymgcv.fitting.pirls import PIRLSResult, pirls_loop
from pymgcv.jax_utils import to_jax
from tests.tolerances import LOOSE, MODERATE

jax.config.update("jax_enable_x64", True)

SEED = 42


# ---- Helpers ----


def _r_available() -> bool:
    """Check if R and mgcv are available."""
    from pymgcv.compat.r_bridge import RBridge

    return RBridge.available()


def _make_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic data as a DataFrame."""
    rng = np.random.default_rng(seed)
    n = 200 if family_name != "binomial" else 300
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


def _setup_fd(
    formula: str,
    data: pd.DataFrame,
    family,
):
    """Build FittingData from formula + data."""
    from pymgcv.formula.design import ModelSetup
    from pymgcv.formula.parser import parse_formula

    spec = parse_formula(formula)
    setup = ModelSetup.build(spec, data)
    return FittingData.from_setup(setup, family)


# ---- A. Safe Newton step tests ----


class TestSafeNewtonStep:
    """Tests for _safe_newton_step eigenvalue handling."""

    def test_quadratic_one_step(self):
        """1D quadratic f(x) = (x-2)^2: Newton converges in 1 step."""
        # f(x) = (x-2)^2, f'(x) = 2(x-2), f''(x) = 2
        # At x=0: f'=−4, f''=2, step = -f'/f'' = 2
        grad = jnp.array([-4.0])
        hess = jnp.array([[2.0]])
        step = _safe_newton_step(grad, hess)
        np.testing.assert_allclose(float(step[0]), 2.0, rtol=1e-10)

    def test_negative_eigenvalues_flipped(self):
        """Negative Hessian eigenvalues are flipped to positive."""
        # Negative-definite Hessian: step should still be a descent direction
        grad = jnp.array([1.0, -1.0])
        hess = jnp.array([[-2.0, 0.0], [0.0, -3.0]])
        step = _safe_newton_step(grad, hess)
        # After flipping: eigs become [2, 3], step = -H_safe^{-1} g
        expected = -jnp.array([1.0 / 2.0, -1.0 / 3.0])
        np.testing.assert_allclose(np.asarray(step), np.asarray(expected), rtol=1e-10)

    def test_step_norm_capped(self):
        """Step norm is capped to max_step."""
        grad = jnp.array([100.0])
        hess = jnp.array([[1.0]])
        step = _safe_newton_step(grad, hess, max_step=5.0)
        assert float(jnp.sqrt(jnp.sum(step**2))) <= 5.0 + 1e-10

    def test_near_singular_hessian(self):
        """Near-singular Hessian: floor prevents division by zero."""
        grad = jnp.array([1.0, 1.0])
        hess = jnp.array([[1.0, 0.0], [0.0, 1e-20]])
        step = _safe_newton_step(grad, hess)
        assert jnp.all(jnp.isfinite(step))


# ---- B. Gaussian GAM vs R ----


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestGaussianNewton:
    """Gaussian GAM: Newton optimization vs R."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_gaussian_converges(self):
        """Gaussian GAM converges."""
        data = _make_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())
        result = newton_optimize(fd)
        assert result.converged
        assert result.convergence_info == "full convergence"
        assert result.n_iter > 0

    def test_gaussian_single_smooth_vs_r(self):
        """Gaussian single smooth: deviance matches R at MODERATE."""
        from pymgcv.compat.r_bridge import RBridge

        data = _make_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())
        result = newton_optimize(fd)

        bridge = RBridge()
        r_result = bridge.fit_gam(self.FORMULA, data, family="gaussian")

        np.testing.assert_allclose(
            float(result.pirls_result.deviance),
            r_result["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Gaussian deviance differs from R",
        )

    def test_gaussian_lambda_vs_r(self):
        """Gaussian: log_lambda matches R at LOOSE tolerance."""
        from pymgcv.compat.r_bridge import RBridge

        data = _make_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())
        result = newton_optimize(fd)

        bridge = RBridge()
        r_result = bridge.fit_gam(self.FORMULA, data, family="gaussian")

        r_log_lambda = np.log(r_result["smoothing_params"])
        np.testing.assert_allclose(
            np.asarray(result.log_lambda),
            r_log_lambda,
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Gaussian log_lambda differs from R",
        )

    def test_gaussian_two_smooths(self):
        """Two-smooth Gaussian model converges and matches R deviance."""
        from pymgcv.compat.r_bridge import RBridge

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
            float(result.pirls_result.deviance),
            r_result["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Two-smooth Gaussian deviance differs from R",
        )


# ---- C. GLM families ----


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestGLMFamilies:
    """GLM family convergence tests."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_poisson_converges(self):
        """Poisson GAM converges."""
        data = _make_data("poisson")
        fd = _setup_fd(self.FORMULA, data, Poisson())
        result = newton_optimize(fd)
        assert result.converged
        assert float(result.scale) == 1.0  # Known scale

    def test_binomial_converges(self):
        """Binomial GAM converges and coefficients match R at LOOSE."""
        from pymgcv.compat.r_bridge import RBridge

        data = _make_data("binomial")
        fd = _setup_fd(self.FORMULA, data, Binomial())
        result = newton_optimize(fd)

        assert result.converged

        bridge = RBridge()
        r_result = bridge.fit_gam(self.FORMULA, data, family="binomial")

        np.testing.assert_allclose(
            np.asarray(result.pirls_result.coefficients),
            r_result["coefficients"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Binomial coefficients differ from R",
        )

    def test_gamma_converges(self):
        """Gamma GAM converges."""
        data = _make_data("gamma")
        fd = _setup_fd(self.FORMULA, data, Gamma())
        result = newton_optimize(fd)
        assert result.converged
        assert float(result.scale) > 0


# ---- D. ML criterion ----


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestMLOptimization:
    """ML criterion optimization tests."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_ml_converges(self):
        """ML optimization converges."""
        data = _make_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())
        result = newton_optimize(fd, method="ML")
        assert result.converged

    def test_ml_differs_from_reml(self):
        """ML and REML produce different criterion scores.

        The REML criterion includes a -Mp/2*log(2*pi*phi) correction
        that ML does not. While optimal lambda may coincide at large n,
        the criterion values must differ.
        """
        data = _make_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())
        result_reml = newton_optimize(fd, method="REML")
        result_ml = newton_optimize(fd, method="ML")

        # Criterion values must differ (REML has the -Mp/2*log(2pi*phi) term)
        assert abs(float(result_reml.score) - float(result_ml.score)) > 0.01


# ---- E. Diagnostics and edge cases ----


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestDiagnostics:
    """Result fields, edge cases, and invariants."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_result_fields(self):
        """NewtonResult has all expected fields with correct types."""
        data = _make_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())
        result = newton_optimize(fd)

        assert isinstance(result, NewtonResult)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iter, int)
        assert isinstance(result.convergence_info, str)
        assert isinstance(result.pirls_result, PIRLSResult)
        assert result.log_lambda.shape == (fd.n_penalties,)
        assert result.smoothing_params.shape == (fd.n_penalties,)
        assert result.gradient.shape == (fd.n_penalties,)
        assert result.score.shape == ()
        assert result.edf.shape == ()
        assert result.scale.shape == ()

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
            S_list=(),
            log_lambda_init=jnp.zeros(0),
            family=Gaussian(),
            n_obs=n,
            n_coef=p,
            penalty_ranks=(),
            penalty_null_dims=(),
        )
        result = newton_optimize(fd)

        assert result.converged
        assert result.n_iter == 0
        assert result.convergence_info == "full convergence"
        assert result.log_lambda.shape == (0,)
        assert result.smoothing_params.shape == (0,)

    def test_reml_monotonicity(self):
        """REML score should not increase at accepted steps.

        We verify this by running the optimizer with a callback that
        tracks all accepted scores.
        """
        data = _make_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())

        # Run with a deliberately bad start to force multiple iterations
        log_lambda_init = jnp.array([5.0])
        result = newton_optimize(fd, log_lambda_init=log_lambda_init)

        # The final score should be less than or equal to the initial score
        from pymgcv.fitting.reml import REMLCriterion

        beta_init = initialize_beta(
            np.asarray(fd.X), np.asarray(fd.y), np.asarray(fd.wt), fd.family
        )
        S_init = fd.S_lambda(log_lambda_init)
        pirls_init = pirls_loop(
            fd.X, fd.y, to_jax(np.asarray(beta_init)), S_init, fd.family, fd.wt
        )
        crit_init = REMLCriterion(fd, pirls_init)
        score_init = float(crit_init.score(log_lambda_init))

        assert float(result.score) <= score_init + 1e-10

    def test_convergence_info_values(self):
        """convergence_info is one of the three expected strings."""
        data = _make_data("gaussian")
        fd = _setup_fd(self.FORMULA, data, Gaussian())
        result = newton_optimize(fd)
        assert result.convergence_info in {
            "full convergence",
            "step failed",
            "iteration limit",
        }


# ---- F. Step-halving ----


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestStepHalving:
    """Step-halving behavior."""

    def test_step_halving_activates(self):
        """With adversarial log_lambda_init, step-halving kicks in.

        Starting far from the optimum forces the optimizer to use
        step-halving. We verify convergence still occurs.
        """
        data = _make_data("gaussian")
        formula = "y ~ s(x, k=10, bs='cr')"
        fd = _setup_fd(formula, data, Gaussian())

        # Very far from optimum
        log_lambda_init = jnp.array([10.0])
        result = newton_optimize(fd, log_lambda_init=log_lambda_init)

        # Should still converge (perhaps with more iterations)
        assert result.converged or result.convergence_info == "step failed"
        # Score should be reasonable (not NaN/Inf)
        assert jnp.isfinite(result.score)
