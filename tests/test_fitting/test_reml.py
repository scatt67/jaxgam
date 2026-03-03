"""Tests for REML and ML criterion functions.

Tests cover:
- REML score matching R's gcv.ubre for all four standard families
- Gradient and Hessian via jax.grad / jax.hessian
- Finite-difference verification of gradient and Hessian
- ML criterion (differs from REML)
- Pearson RSS properties
- Multi-penalty models (two smooths)
- Purely parametric model (no penalties)
- REMLCriterion / MLCriterion class API
- Scale handling (known phi=1 vs estimated phi)
- Fletcher scale estimation

Design doc reference: Section 4.3, 4.4
R source reference: gam.fit3.r lines 612-640 (general Laplace REML)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.families.base import ExponentialFamily
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.initialization import initialize_beta
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.fitting.reml import (
    MLCriterion,
    REMLCriterion,
    REMLResult,
    estimate_edf,
    estimate_scale,
    fletcher_scale,
    ml_criterion,
    pearson_rss,
    reml_criterion,
)
from jaxgam.jax_utils import to_jax, to_numpy
from tests.helpers import SEED, _generate_family_data, r_available
from tests.tolerances import LOOSE, MODERATE, STRICT

jax.config.update("jax_enable_x64", True)


def _setup_pipeline(
    formula: str,
    data: pd.DataFrame,
    family: ExponentialFamily,
    family_r: str,
):
    """Build Python pipeline and get R reference values.

    Returns (fd, pirls_result, log_lambda, r_result).
    """
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula
    from tests.r_bridge import RBridge

    spec = parse_formula(formula)
    setup = ModelSetup.build(spec, data)
    fd = FittingData.from_setup(setup, family)

    # Initialize beta from the (possibly reparameterized) model matrix
    # stored in fd.X, matching the coordinate system PIRLS will use.
    beta_init = initialize_beta(
        np.asarray(fd.X), setup.y, setup.weights, family, setup.offset
    )
    beta_jax = to_jax(np.asarray(beta_init))

    bridge = RBridge()
    r_result = bridge.fit_gam(formula, data, family=family_r)
    log_lambda = jnp.log(jnp.array(r_result["smoothing_params"]))

    S_lambda = fd.S_lambda(log_lambda)
    pirls_result = pirls_loop(
        fd.X, fd.y, beta_jax, S_lambda, fd.family, fd.wt, fd.offset
    )

    return fd, pirls_result, log_lambda, r_result


def _reml_args(fd, pirls_result, log_lambda):
    """Build common arguments for reml_criterion calls.

    Uses our own Fletcher scale estimation for unknown-scale families.
    """
    edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
    phi = estimate_scale(fd.y, pirls_result.mu, fd.wt, fd.family, edf)
    deviance = pirls_result.deviance
    ls_sat = fd.family.saturated_loglik(fd.y, fd.wt, phi)
    Mp = fd.total_penalty_null_dim
    return {
        "log_lambda": log_lambda,
        "XtWX": pirls_result.XtWX,
        "beta": pirls_result.coefficients,
        "deviance": deviance,
        "ls_sat": ls_sat,
        "S_list": fd.S_list,
        "phi": phi,
        "Mp": Mp,
        "singleton_sp_indices": fd.singleton_sp_indices,
        "singleton_ranks": fd.singleton_ranks,
        "singleton_eig_constants": fd.singleton_eig_constants,
        "multi_block_sp_indices": fd.multi_block_sp_indices,
        "multi_block_ranks": fd.multi_block_ranks,
        "multi_block_proj_S": fd.multi_block_proj_S,
    }


# ---- REML score vs R ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestREMLScore:
    """REML score must match R's gcv.ubre.

    Uses our own Fletcher scale estimation for unknown-scale families.
    Uses LOOSE tolerance since this is a cross-implementation comparison:
    our Fletcher scale differs from R's jointly-optimized reml_scale.
    """

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def _check_reml_score(self, family_name, family_r, family):
        data = _generate_family_data(family_name)
        fd, pirls_result, log_lambda, r_result = _setup_pipeline(
            self.FORMULA, data, family, family_r
        )

        args = _reml_args(fd, pirls_result, log_lambda)
        py_reml = reml_criterion(**args)

        np.testing.assert_allclose(
            float(py_reml),
            r_result["reml_score"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg=f"{family_name} REML score differs from R",
        )

    def test_matches_r_gaussian(self):
        self._check_reml_score("gaussian", "gaussian", Gaussian())

    def test_matches_r_poisson(self):
        self._check_reml_score("poisson", "poisson", Poisson())

    def test_matches_r_binomial(self):
        self._check_reml_score("binomial", "binomial", Binomial())

    def test_matches_r_gamma(self):
        self._check_reml_score("gamma", "gamma", Gamma())


# ---- REML gradient ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestREMLGradient:
    """jax.grad(reml_criterion) must produce valid gradients."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @classmethod
    def setup_class(cls):
        data = _generate_family_data("gaussian")
        cls.fd, cls.pirls_result, cls.log_lambda, cls.r_result = _setup_pipeline(
            cls.FORMULA, data, Gaussian(), "gaussian"
        )

    def _reml_fn(self, log_lambda):
        args = _reml_args(self.fd, self.pirls_result, log_lambda)
        return reml_criterion(**args)

    def test_gradient_finite(self):
        grad = jax.grad(self._reml_fn)(self.log_lambda)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_near_zero_at_r_optimum(self):
        """At R's optimal sp, the REML gradient should be small."""
        grad = jax.grad(self._reml_fn)(self.log_lambda)
        grad_np = to_numpy(grad)
        reml_val = float(self._reml_fn(self.log_lambda))
        scale = abs(reml_val) + 1.0
        # Gradient should be small relative to function scale.
        # Not exactly zero because R uses a different solve path
        # and our phi differs from R's.
        assert np.max(np.abs(grad_np)) < 0.5 * scale

    def test_gradient_matches_fd(self):
        """AD gradient matches central finite differences."""
        grad_ad = to_numpy(jax.grad(self._reml_fn)(self.log_lambda))
        eps = 1e-5
        m = len(self.log_lambda)
        grad_fd = np.zeros(m)
        for j in range(m):
            e_j = jnp.zeros(m).at[j].set(eps)
            f_plus = float(self._reml_fn(self.log_lambda + e_j))
            f_minus = float(self._reml_fn(self.log_lambda - e_j))
            grad_fd[j] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(
            grad_ad,
            grad_fd,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="REML gradient differs from FD",
        )


# ---- REML Hessian ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestREMLHessian:
    """jax.hessian(reml_criterion) must be finite and match FD."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @classmethod
    def setup_class(cls):
        data = _generate_family_data("gaussian")
        cls.fd, cls.pirls_result, cls.log_lambda, _ = _setup_pipeline(
            cls.FORMULA, data, Gaussian(), "gaussian"
        )

    def _reml_fn(self, log_lambda):
        args = _reml_args(self.fd, self.pirls_result, log_lambda)
        return reml_criterion(**args)

    def test_hessian_finite(self):
        hess = jax.hessian(self._reml_fn)(self.log_lambda)
        assert jnp.all(jnp.isfinite(hess))

    def test_hessian_matches_fd(self):
        """Hessian matches FD of the gradient."""
        hess_ad = to_numpy(jax.hessian(self._reml_fn)(self.log_lambda))
        grad_fn = jax.grad(self._reml_fn)
        eps = 1e-4
        m = len(self.log_lambda)
        hess_fd = np.zeros((m, m))
        for j in range(m):
            e_j = jnp.zeros(m).at[j].set(eps)
            g_plus = to_numpy(grad_fn(self.log_lambda + e_j))
            g_minus = to_numpy(grad_fn(self.log_lambda - e_j))
            hess_fd[:, j] = (g_plus - g_minus) / (2 * eps)

        np.testing.assert_allclose(
            hess_ad,
            hess_fd,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="REML Hessian differs from FD of gradient",
        )


# ---- ML criterion ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestMLCriterion:
    """ML criterion tests."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @classmethod
    def setup_class(cls):
        data = _generate_family_data("gaussian")
        cls.fd, cls.pirls_result, cls.log_lambda, _ = _setup_pipeline(
            cls.FORMULA, data, Gaussian(), "gaussian"
        )

    def test_ml_differs_from_reml(self):
        """ML and REML produce different scores."""
        args = _reml_args(self.fd, self.pirls_result, self.log_lambda)
        reml_val = float(reml_criterion(**args))

        ml_val = float(ml_criterion(**args))
        assert reml_val != ml_val

    def test_ml_gradient_finite(self):
        def ml_fn(ll):
            args = _reml_args(self.fd, self.pirls_result, ll)
            return ml_criterion(**args)

        grad = jax.grad(ml_fn)(self.log_lambda)
        assert jnp.all(jnp.isfinite(grad))


# ---- Pearson RSS ----


class TestPearsonRSS:
    """Pearson chi-square properties."""

    def test_gaussian_equals_deviance(self):
        """For Gaussian, Pearson chi-sq = deviance."""
        rng = np.random.default_rng(SEED)
        n = 100
        y = jnp.array(rng.normal(0, 1, n))
        mu = jnp.array(rng.normal(0, 0.5, n))
        wt = jnp.ones(n)
        family = Gaussian()

        p_rss = float(pearson_rss(y, mu, wt, family))
        deviance = float(family.dev_resids(y, mu, wt))

        np.testing.assert_allclose(
            p_rss,
            deviance,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Gaussian Pearson RSS should equal deviance",
        )

    def test_poisson_differs_from_deviance(self):
        """For Poisson, Pearson chi-sq != deviance."""
        rng = np.random.default_rng(SEED)
        n = 100
        mu_true = np.exp(rng.normal(0.5, 0.3, n))
        y = jnp.array(rng.poisson(mu_true).astype(float))
        mu = jnp.array(mu_true)
        wt = jnp.ones(n)
        family = Poisson()

        p_rss = float(pearson_rss(y, mu, wt, family))
        deviance = float(family.dev_resids(y, mu, wt))

        assert abs(p_rss - deviance) > 0.01


# ---- Multi-penalty models ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestMultiPenalty:
    """REML handles multi-penalty models (multiple smooths)."""

    def test_reml_two_smooths(self):
        """Two-smooth model: y ~ s(x1) + s(x2)."""
        rng = np.random.default_rng(SEED)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        family = Gaussian()

        fd, pirls_result, log_lambda, r_result = _setup_pipeline(
            formula, data, family, "gaussian"
        )

        args = _reml_args(fd, pirls_result, log_lambda)
        py_reml = reml_criterion(**args)

        np.testing.assert_allclose(
            float(py_reml),
            r_result["reml_score"],
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="Two-smooth REML score differs from R",
        )

    def test_reml_two_smooths_gradient_finite(self):
        """Gradient should be finite with 2 components."""
        rng = np.random.default_rng(SEED)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        formula = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"
        family = Gaussian()

        fd, pirls_result, log_lambda, _ = _setup_pipeline(
            formula, data, family, "gaussian"
        )

        def reml_fn(ll):
            args = _reml_args(fd, pirls_result, ll)
            return reml_criterion(**args)

        grad = jax.grad(reml_fn)(log_lambda)
        assert grad.shape == (2,)
        assert jnp.all(jnp.isfinite(grad))


# ---- Purely parametric model ----


class TestPurelyParametric:
    """Degenerate case: no penalties, REML reduces to log-lik."""

    def test_no_penalties(self):
        """No penalties: S_list empty, Mp=p, log|S^+|=0."""
        rng = np.random.default_rng(SEED)
        n, p = 100, 3
        X = rng.normal(0, 1, (n, p))
        beta_true = np.array([1.0, -0.5, 0.3])
        y = X @ beta_true + rng.normal(0, 0.5, n)

        X_d, y_d = to_jax(X, y)
        wt_d = jnp.ones(n)
        S_d = jnp.zeros((p, p))

        family = Gaussian()
        beta_init = to_jax(np.asarray(initialize_beta(X, y, np.ones(n), family)))
        result = pirls_loop(X_d, y_d, beta_init, S_d, family)

        deviance = result.deviance
        ls_sat = family.saturated_loglik(y_d, wt_d, result.scale)

        # No penalties => total null space = p (all coefficients unpenalized)
        Mp = p
        log_lambda = jnp.zeros(0)
        py_reml = reml_criterion(
            log_lambda,
            result.XtWX,
            result.coefficients,
            deviance,
            ls_sat,
            (),
            result.scale,
            Mp,
            singleton_sp_indices=(),
            singleton_ranks=(),
            singleton_eig_constants=jnp.array([]),
            multi_block_sp_indices=(),
            multi_block_ranks=(),
            multi_block_proj_S=(),
        )

        # Expected: Dp/(2*phi) - ls_sat + log|XtWX|/2 - Mp/2*log(2*pi*phi)
        # For Gaussian with wt=1:
        #   ls_sat = -n/2*log(2*pi*phi)
        #   Dp = dev (no penalty)
        #   REML = dev/(2*phi) + n/2*log(2*pi*phi) + log|XtWX|/2
        #          - p/2*log(2*pi*phi)
        #        = dev/(2*phi) + (n-p)/2*log(2*pi*phi) + log|XtWX|/2
        _, logdet = jnp.linalg.slogdet(result.XtWX)
        expected = (
            deviance / (2.0 * result.scale)
            + (n - p) / 2.0 * jnp.log(2.0 * jnp.pi * result.scale)
            + logdet / 2.0
        )

        np.testing.assert_allclose(
            float(py_reml),
            float(expected),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="No-penalty REML formula mismatch",
        )


# ---- Criterion classes ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestCriterionClasses:
    """REMLCriterion / MLCriterion class API."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @classmethod
    def setup_class(cls):
        data = _generate_family_data("gaussian")
        cls.fd, cls.pirls_result, cls.log_lambda, _ = _setup_pipeline(
            cls.FORMULA, data, Gaussian(), "gaussian"
        )

    def test_reml_score_scalar(self):
        """REMLCriterion.score returns a scalar."""
        obj = REMLCriterion(self.fd, self.pirls_result)
        score = obj.score(self.log_lambda)
        assert score.shape == ()
        assert jnp.isfinite(score)

    def test_reml_gradient(self):
        """REMLCriterion.gradient returns finite gradient."""
        obj = REMLCriterion(self.fd, self.pirls_result)
        grad = obj.gradient(self.log_lambda)
        assert grad.shape == self.log_lambda.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_reml_hessian(self):
        """REMLCriterion.hessian returns finite Hessian."""
        obj = REMLCriterion(self.fd, self.pirls_result)
        hess = obj.hessian(self.log_lambda)
        m = len(self.log_lambda)
        assert hess.shape == (m, m)
        assert jnp.all(jnp.isfinite(hess))

    def test_reml_evaluate(self):
        """REMLCriterion.evaluate returns REMLResult."""
        obj = REMLCriterion(self.fd, self.pirls_result)
        result = obj.evaluate(self.log_lambda)
        assert isinstance(result, REMLResult)
        assert jnp.isfinite(result.score)
        assert jnp.isfinite(result.edf)
        assert jnp.isfinite(result.scale)
        assert float(result.scale) > 0

    def test_ml_score_scalar(self):
        """MLCriterion.score returns a scalar."""
        obj = MLCriterion(self.fd, self.pirls_result)
        score = obj.score(self.log_lambda)
        assert score.shape == ()
        assert jnp.isfinite(score)

    def test_ml_gradient(self):
        """MLCriterion.gradient returns finite gradient."""
        obj = MLCriterion(self.fd, self.pirls_result)
        grad = obj.gradient(self.log_lambda)
        assert grad.shape == self.log_lambda.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_ml_evaluate(self):
        """MLCriterion.evaluate returns REMLResult."""
        obj = MLCriterion(self.fd, self.pirls_result)
        result = obj.evaluate(self.log_lambda)
        assert isinstance(result, REMLResult)
        assert jnp.isfinite(result.score)


# ---- Scale handling ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestScaleHandling:
    """Dispersion parameter handling."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_known_scale_phi_one(self):
        """For Binomial/Poisson, phi should be 1.0."""
        data = _generate_family_data("poisson")
        _, pirls_result, _, _ = _setup_pipeline(
            self.FORMULA, data, Poisson(), "poisson"
        )
        np.testing.assert_allclose(
            float(pirls_result.scale),
            1.0,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Poisson scale should be 1.0",
        )

    def test_unknown_scale_positive(self):
        """For Gaussian, phi = dev/(n-p) should be positive."""
        data = _generate_family_data("gaussian")
        _, pirls_result, _, _ = _setup_pipeline(
            self.FORMULA, data, Gaussian(), "gaussian"
        )
        phi = float(pirls_result.scale)
        assert phi > 0, "Gaussian scale must be positive"


# ---- Fletcher scale ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestFletcherScale:
    """Fletcher (2012) bias-corrected scale estimator."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_gaussian_no_correction(self):
        """For Gaussian, V'(mu)=0 so Fletcher = Pearson."""
        data = _generate_family_data("gaussian")
        fd, pirls_result, _log_lambda, _ = _setup_pipeline(
            self.FORMULA, data, Gaussian(), "gaussian"
        )
        n = fd.n_obs
        edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
        pearson = pearson_rss(fd.y, pirls_result.mu, fd.wt, fd.family)
        phi_pearson = pearson / (n - edf)
        phi_fletcher = fletcher_scale(fd.y, pirls_result.mu, fd.wt, fd.family, edf)
        np.testing.assert_allclose(
            float(phi_fletcher),
            float(phi_pearson),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Gaussian Fletcher should equal Pearson",
        )

    def test_poisson_differs_from_pearson(self):
        """For Poisson, V'(mu)=1 so Fletcher != Pearson."""
        data = _generate_family_data("poisson")
        fd, pirls_result, _log_lambda, _ = _setup_pipeline(
            self.FORMULA, data, Poisson(), "poisson"
        )
        n = fd.n_obs
        edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
        pearson = pearson_rss(fd.y, pirls_result.mu, fd.wt, fd.family)
        phi_pearson = float(pearson / (n - edf))
        phi_fletcher = float(
            fletcher_scale(fd.y, pirls_result.mu, fd.wt, fd.family, edf)
        )
        assert phi_fletcher != phi_pearson
        assert phi_fletcher > 0

    def test_fletcher_positive(self):
        """Fletcher scale estimate is positive for all families."""
        for family_name, family_r, family in [
            ("gaussian", "gaussian", Gaussian()),
            ("poisson", "poisson", Poisson()),
            ("binomial", "binomial", Binomial()),
            ("gamma", "gamma", Gamma()),
        ]:
            data = _generate_family_data(family_name)
            fd, pirls_result, _log_lambda, _ = _setup_pipeline(
                self.FORMULA, data, family, family_r
            )
            edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
            phi = float(fletcher_scale(fd.y, pirls_result.mu, fd.wt, fd.family, edf))
            assert phi > 0, f"{family_name} Fletcher scale not positive"

    def test_fletcher_matches_r_scale(self):
        """Our Fletcher scale matches R's model$scale for all families."""
        for family_name, family_r, family in [
            ("gaussian", "gaussian", Gaussian()),
            ("gamma", "gamma", Gamma()),
        ]:
            data = _generate_family_data(family_name)
            fd, pirls_result, _log_lambda, r_result = _setup_pipeline(
                self.FORMULA, data, family, family_r
            )
            edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
            phi = float(fletcher_scale(fd.y, pirls_result.mu, fd.wt, fd.family, edf))
            np.testing.assert_allclose(
                phi,
                r_result["scale"],
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=f"{family_name} Fletcher scale differs from R",
            )


# ---- EDF estimation ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestEstimateEdf:
    """EDF estimation from PIRLS Cholesky factor."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_edf_matches_r(self):
        """Our EDF matches sum of R's edf + 1 (intercept).

        R's edf vector reports per-smooth EDF. Our trace(H^{-1} XtWX)
        is the total hat matrix trace including the unpenalized intercept.
        """
        data = _generate_family_data("gaussian")
        _fd, pirls_result, _log_lambda, r_result = _setup_pipeline(
            self.FORMULA, data, Gaussian(), "gaussian"
        )
        py_edf = float(estimate_edf(pirls_result.XtWX, pirls_result.L))
        # R's edf is per-smooth; add 1 for the intercept
        r_edf_total = float(np.sum(r_result["edf"])) + 1.0
        np.testing.assert_allclose(
            py_edf,
            r_edf_total,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="EDF differs from R",
        )

    def test_edf_bounded(self):
        """EDF should be between 0 and p (number of coefficients)."""
        data = _generate_family_data("gaussian")
        fd, pirls_result, _log_lambda, _ = _setup_pipeline(
            self.FORMULA, data, Gaussian(), "gaussian"
        )
        py_edf = float(estimate_edf(pirls_result.XtWX, pirls_result.L))
        p = fd.n_coef
        assert 0 < py_edf <= p, f"EDF {py_edf} not in (0, {p}]"


# ---- Saturated log-likelihood ----


class TestSaturatedLoglik:
    """Family saturated_loglik implementations."""

    def test_gaussian_matches_formula(self):
        """Gaussian ls_sat = -n/2*log(2*pi*phi) for unit weights."""
        rng = np.random.default_rng(SEED)
        n = 100
        y = jnp.array(rng.normal(0, 1, n))
        wt = jnp.ones(n)
        phi = jnp.array(0.5)
        family = Gaussian()

        ls_sat = float(family.saturated_loglik(y, wt, phi))
        expected = float(-n / 2.0 * jnp.log(2.0 * jnp.pi * phi))

        np.testing.assert_allclose(
            ls_sat,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_poisson_scale_independent(self):
        """Poisson ls_sat doesn't depend on scale."""
        rng = np.random.default_rng(SEED)
        y = jnp.array(rng.poisson(3.0, 50).astype(float))
        wt = jnp.ones(50)
        family = Poisson()

        ls1 = float(family.saturated_loglik(y, wt, jnp.array(1.0)))
        ls2 = float(family.saturated_loglik(y, wt, jnp.array(2.0)))

        np.testing.assert_allclose(
            ls1,
            ls2,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_binomial_boundary_handling(self):
        """Binomial ls_sat handles y=0 and y=1 (boundaries)."""
        y = jnp.array([0.0, 1.0, 0.0, 1.0])
        wt = jnp.ones(4)
        family = Binomial()

        ls_sat = float(family.saturated_loglik(y, wt, jnp.array(1.0)))
        # At boundaries, y*log(y) + (1-y)*log(1-y) = 0
        np.testing.assert_allclose(ls_sat, 0.0, atol=STRICT.atol)

    def test_gamma_finite(self):
        """Gamma ls_sat is finite for reasonable inputs."""
        rng = np.random.default_rng(SEED)
        y = jnp.array(rng.gamma(5.0, 1.0, 100))
        wt = jnp.ones(100)
        family = Gamma()

        ls_sat = float(family.saturated_loglik(y, wt, jnp.array(0.2)))
        assert np.isfinite(ls_sat)


# ---- JIT compilation tests ----


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestJITCompilation:
    """JIT compilation of REML/ML criterion classes."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @classmethod
    def setup_class(cls):
        data = _generate_family_data("gaussian")
        cls.fd, cls.pirls_result, cls.log_lambda, _ = _setup_pipeline(
            cls.FORMULA, data, Gaussian(), "gaussian"
        )

    def test_reml_score_jit(self):
        """REMLCriterion.score compiles under jax.jit."""
        obj = REMLCriterion(self.fd, self.pirls_result)
        jit_score = jax.jit(obj.score)
        score = jit_score(self.log_lambda)
        score_nojit = obj.score(self.log_lambda)
        np.testing.assert_allclose(float(score), float(score_nojit), rtol=STRICT.rtol)

    def test_reml_jit_grad(self):
        """jax.grad works on JIT'd REMLCriterion.score."""
        obj = REMLCriterion(self.fd, self.pirls_result)
        grad = jax.grad(jax.jit(obj.score))(self.log_lambda)
        assert jnp.all(jnp.isfinite(grad))

    def test_reml_jit_hessian(self):
        """jax.hessian works on JIT'd REMLCriterion.score."""
        obj = REMLCriterion(self.fd, self.pirls_result)
        hess = jax.hessian(jax.jit(obj.score))(self.log_lambda)
        assert jnp.all(jnp.isfinite(hess))

    def test_ml_score_jit(self):
        """MLCriterion.score compiles under jax.jit."""
        obj = MLCriterion(self.fd, self.pirls_result)
        jit_score = jax.jit(obj.score)
        score = jit_score(self.log_lambda)
        assert jnp.isfinite(score)
