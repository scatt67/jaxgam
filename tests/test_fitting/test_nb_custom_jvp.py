"""Tests for the extended custom_jvp on PIRLS with theta (PR 3).

Validates that ``_diff_score``'s 3-primal custom_jvp correctly computes
derivatives of the REML criterion w.r.t. ``log_theta`` for extended
families.

Tests:
- AD gradient w.r.t. log_theta matches FD (fresh families per perturbation)
- AD lambda gradient from 3-primal path matches 2-primal standard path
- Hessian finite and symmetric
- Hessian theta-theta block matches FD of gradient
- dbeta/d(log_theta) via IFT matches FD PIRLS perturbation
- Standard families (n_theta=0): gradient unchanged (no regression)

Design doc reference: Section 6.2-6.4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax.scipy.linalg import cho_solve

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.initialization import initialize_beta
from jaxgam.fitting.newton import _diff_score, _fit_and_score_impl
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.jax_utils import build_S_lambda, cho_factor
from tests.tolerances import MODERATE, STRICT

jax.config.update("jax_enable_x64", True)

_FORMULA = "y ~ s(x, bs='cr', k=8)"


# ---- Helpers ----


def _setup_fd(formula: str, data: pd.DataFrame, family):
    """Build FittingData from formula + data."""
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    spec = parse_formula(formula)
    setup = ModelSetup.build(spec, data)
    return FittingData.from_setup(setup, family)


def _make_nb_data(n: int = 100, seed: int = 42, true_theta: float = 2.0):
    """Generate NB count data with known theta."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x) + 0.5
    mu = np.exp(eta)
    y = rng.negative_binomial(
        n=true_theta, p=true_theta / (mu + true_theta), size=n
    ).astype(float)
    return pd.DataFrame({"x": x, "y": y})


def _build_diff_score_kwargs(fd: FittingData, joint_theta: bool):
    """Build keyword args dict for calling _diff_score directly."""
    offset = fd.offset if fd.offset is not None else jnp.zeros(fd.n_obs)
    return {
        "X": fd.X,
        "y": fd.y,
        "wt": fd.wt,
        "offset": offset,
        "S_list": fd.S_list,
        "singleton_eig_constants": fd.singleton_eig_constants,
        "multi_block_proj_S": fd.multi_block_proj_S,
        "family": fd.family,
        "pirls_tol": 1e-8,
        "is_reml": True,
        "joint_theta": joint_theta,
        "joint_scale": False,
        "n_lambda": fd.n_penalties,
        "Mp": fd.total_penalty_null_dim,
        "singleton_sp_indices": fd.singleton_sp_indices,
        "singleton_ranks": fd.singleton_ranks,
        "multi_block_sp_indices": fd.multi_block_sp_indices,
        "multi_block_ranks": fd.multi_block_ranks,
        "p": fd.n_coef,
    }


def _converge_pirls(fd):
    """Run PIRLS to convergence, return beta_warm."""
    offset = fd.offset if fd.offset is not None else jnp.zeros(fd.n_obs)
    beta_init = initialize_beta(
        np.asarray(fd.X),
        np.asarray(fd.y),
        np.asarray(fd.wt),
        fd.family,
        np.asarray(offset) if fd.offset is not None else None,
    )
    log_lambda = fd.log_lambda_init.copy()
    S_lambda = build_S_lambda(log_lambda, fd.S_list, fd.n_coef)
    pirls_result = pirls_loop(
        fd.X,
        fd.y,
        beta_init,
        S_lambda,
        fd.family,
        fd.wt,
        offset,
        tol=1e-8,
    )
    return pirls_result, log_lambda


def _score_at_theta(log_theta_val, log_lambda, data, beta_warm):
    """Evaluate REML score at given (log_lambda, log_theta) with a fresh family.

    Creates a fresh NegativeBinomial to avoid JAX compilation caching
    issues with mutable family state. Both log_lambda and log_theta
    are passed through params; the family's stored theta is synced.
    """
    fam = NegativeBinomial()
    fam._log_theta = np.array([float(log_theta_val)])
    fam.n_theta = 1
    fam._max_y = int(np.max(data["y"].values))

    fd = _setup_fd(_FORMULA, data, fam)
    kwargs = _build_diff_score_kwargs(fd, joint_theta=True)
    params = jnp.concatenate(
        [
            jnp.atleast_1d(jnp.asarray(log_lambda)),
            jnp.array([float(log_theta_val)]),
        ]
    )
    return float(_diff_score(params, beta_warm, **kwargs))


# ---- Fixtures ----


@pytest.fixture(scope="module")
def nb_problem():
    """Small NB problem: n=100, 1 CR smooth (k=8), true theta=2."""
    data = _make_nb_data(n=100, seed=42, true_theta=2.0)
    family = NegativeBinomial()  # estimate theta, start at 1
    fd = _setup_fd(_FORMULA, data, family)
    # Set max_y for _lgamma_diff scan (normally done by NewtonOptimizer)
    family._max_y = int(np.max(data["y"].values))

    pirls_result, log_lambda = _converge_pirls(fd)
    log_theta = jnp.asarray(family.get_theta(transformed=False))
    params = jnp.concatenate([log_lambda, log_theta])
    beta_warm = pirls_result.coefficients

    return fd, params, beta_warm, data


# ---- Tests ----


class TestExtendedCustomJVPGradient:
    """AD gradient of _diff_score w.r.t. theta matches finite differences."""

    def test_theta_gradient_matches_fd(self, nb_problem):
        """dScore/d(log_theta) from AD matches central FD.

        LOOSE tolerance because FD requires separate PIRLS runs at
        perturbed theta (each with 1e-8 convergence tolerance), which
        introduces O(1e-3) relative noise in the FD estimate.
        """
        fd, params, beta_warm, data = nb_problem
        kwargs = _build_diff_score_kwargs(fd, joint_theta=True)
        n_lambda = fd.n_penalties

        # AD gradient
        grad_fn = jax.grad(_diff_score, argnums=0)
        ad_grad = grad_fn(params, beta_warm, **kwargs)
        ad_theta_grad = float(ad_grad[n_lambda])

        # Central FD: fresh families avoid JAX caching issues
        eps = 1e-4
        log_lambda = params[:n_lambda]
        log_theta_base = float(params[n_lambda])

        score_plus = _score_at_theta(
            log_theta_base + eps,
            log_lambda,
            data,
            beta_warm,
        )
        score_minus = _score_at_theta(
            log_theta_base - eps,
            log_lambda,
            data,
            beta_warm,
        )
        fd_theta_grad = (score_plus - score_minus) / (2 * eps)

        np.testing.assert_allclose(
            ad_theta_grad,
            fd_theta_grad,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="theta gradient: AD vs FD mismatch",
        )

    def test_lambda_gradient_matches_fd(self, nb_problem):
        """Lambda gradient from 3-primal path matches central FD.

        The extended path uses the observed Hessian (matching R's gam.fit4
        line 562) which is more accurate than the Fisher scoring
        approximation in the standard 2-primal path for NB.
        FD uses fresh families to avoid JAX compilation caching.
        """
        fd, params, beta_warm, data = nb_problem
        n_lambda = fd.n_penalties

        # AD gradient from extended path
        kwargs_ext = _build_diff_score_kwargs(fd, joint_theta=True)
        grad_ext = jax.grad(_diff_score, argnums=0)(
            params,
            beta_warm,
            **kwargs_ext,
        )

        # Central FD for lambda: fresh families per perturbation
        eps = 1e-4
        log_theta = params[n_lambda:]
        for i in range(n_lambda):
            log_lambda_p = params[:n_lambda].at[i].add(eps)
            score_p = _score_at_theta(
                float(log_theta[0]),
                log_lambda_p,
                data,
                beta_warm,
            )
            log_lambda_m = params[:n_lambda].at[i].add(-eps)
            score_m = _score_at_theta(
                float(log_theta[0]),
                log_lambda_m,
                data,
                beta_warm,
            )
            fd_grad_i = (score_p - score_m) / (2 * eps)

            np.testing.assert_allclose(
                float(grad_ext[i]),
                fd_grad_i,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
                err_msg=f"lambda[{i}] gradient: AD vs FD mismatch",
            )

    def test_gradient_all_finite(self, nb_problem):
        """All gradient components are finite (no NaN/Inf)."""
        fd, params, beta_warm, _ = nb_problem
        kwargs = _build_diff_score_kwargs(fd, joint_theta=True)

        grad_fn = jax.grad(_diff_score, argnums=0)
        ad_grad = grad_fn(params, beta_warm, **kwargs)
        assert jnp.all(jnp.isfinite(ad_grad)), f"Non-finite gradient: {ad_grad}"


class TestExtendedCustomJVPHessian:
    """AD Hessian properties and FD validation."""

    def test_hessian_all_finite(self, nb_problem):
        """All Hessian entries are finite."""
        fd, params, beta_warm, _ = nb_problem
        kwargs = _build_diff_score_kwargs(fd, joint_theta=True)

        hess_fn = jax.hessian(_diff_score, argnums=0)
        ad_hess = hess_fn(params, beta_warm, **kwargs)
        assert jnp.all(jnp.isfinite(ad_hess)), "Non-finite Hessian entries"

    def test_hessian_approximately_symmetric(self, nb_problem):
        """Hessian is approximately symmetric.

        ``jax.hessian`` uses ``jacfwd(jacrev)``, which can produce
        small asymmetry in the cross theta-lambda block because the
        forward and reverse passes through the custom_jvp follow
        slightly different numerical paths. The Newton optimizer
        symmetrizes with ``(H + H.T) / 2``.
        """
        fd, params, beta_warm, _ = nb_problem
        kwargs = _build_diff_score_kwargs(fd, joint_theta=True)

        hess_fn = jax.hessian(_diff_score, argnums=0)
        ad_hess = hess_fn(params, beta_warm, **kwargs)
        np.testing.assert_allclose(
            np.asarray(ad_hess),
            np.asarray(ad_hess.T),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Hessian not approximately symmetric",
        )

    def test_hessian_theta_theta_matches_fd(self, nb_problem):
        """d^2Score/d(log_theta)^2 from AD Hessian matches FD of gradient."""
        fd, params, beta_warm, data = nb_problem
        kwargs = _build_diff_score_kwargs(fd, joint_theta=True)
        n_lambda = fd.n_penalties

        # AD Hessian
        hess_fn = jax.hessian(_diff_score, argnums=0)
        ad_hess = hess_fn(params, beta_warm, **kwargs)
        ad_hess_tt = float(ad_hess[n_lambda, n_lambda])

        # FD of gradient: fresh families for each perturbation
        eps = 1e-4
        log_lambda = params[:n_lambda]
        log_theta_base = float(params[n_lambda])

        def _grad_at_theta(lt_val):
            fam = NegativeBinomial()
            fam._log_theta = np.array([lt_val])
            fam.n_theta = 1
            fam._max_y = int(np.max(data["y"].values))
            fd_local = _setup_fd(_FORMULA, data, fam)
            kw = _build_diff_score_kwargs(fd_local, joint_theta=True)
            p = jnp.concatenate([log_lambda, jnp.array([lt_val])])
            g = jax.grad(_diff_score, argnums=0)(p, beta_warm, **kw)
            return float(g[n_lambda])

        grad_plus = _grad_at_theta(log_theta_base + eps)
        grad_minus = _grad_at_theta(log_theta_base - eps)
        fd_hess_tt = (grad_plus - grad_minus) / (2 * eps)

        np.testing.assert_allclose(
            ad_hess_tt,
            fd_hess_tt,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="theta-theta Hessian: AD vs FD mismatch",
        )


class TestIFTDbetaDtheta:
    """Verify dbeta/d(log_theta) from the IFT in the custom_jvp."""

    def test_dbeta_dtheta_matches_fd(self, nb_problem):
        """IFT analytical dbeta/dtheta matches FD PIRLS perturbation."""
        fd, params, beta_warm, _ = nb_problem
        n_lambda = fd.n_penalties
        log_lambda = params[:n_lambda]
        log_theta_base = float(params[n_lambda])
        offset = fd.offset if fd.offset is not None else jnp.zeros(fd.n_obs)

        # Base PIRLS result
        S_lambda = build_S_lambda(log_lambda, fd.S_list, fd.n_coef)
        pirls_base = pirls_loop(
            fd.X,
            fd.y,
            beta_warm,
            S_lambda,
            fd.family,
            fd.wt,
            offset,
            tol=1e-8,
        )

        # Perturbed PIRLS: fresh families, tight tolerance for FD accuracy
        eps = 1e-5
        fam_plus = NegativeBinomial()
        fam_plus._log_theta = np.array([log_theta_base + eps])
        fam_plus.n_theta = 1
        pirls_plus = pirls_loop(
            fd.X,
            fd.y,
            beta_warm,
            S_lambda,
            fam_plus,
            fd.wt,
            offset,
            tol=1e-12,
        )

        fam_minus = NegativeBinomial()
        fam_minus._log_theta = np.array([log_theta_base - eps])
        fam_minus.n_theta = 1
        pirls_minus = pirls_loop(
            fd.X,
            fd.y,
            beta_warm,
            S_lambda,
            fam_minus,
            fd.wt,
            offset,
            tol=1e-12,
        )

        fd_dbeta = (
            np.asarray(pirls_plus.coefficients) - np.asarray(pirls_minus.coefficients)
        ) / (2 * eps)

        # IFT analytical using observed Hessian (matching R's gam.fit4
        # line 562: w <- dd$Deta2 * .5, which uses the observed d²D/dη²)
        dev_fn = fd.family.deviance_fn(fd.y, fd.wt)
        eta = fd.X @ pirls_base.coefficients + offset
        log_theta_arr = jnp.array([log_theta_base])

        # Observed Hessian diagonal: d²D/dη² (diagonal since D is element-wise)
        grad_D_eta = jax.grad(dev_fn, argnums=0)
        _, d2D_deta2 = jax.jvp(
            lambda e: grad_D_eta(e, log_theta_arr),
            (eta,),
            (jnp.ones_like(eta),),
        )
        H_obs = (fd.X.T * d2D_deta2) @ fd.X + 2 * S_lambda
        L_obs, _ = cho_factor(H_obs)

        # Mixed derivative d²D/(dη dθ)
        _, d_grad_D = jax.jvp(
            lambda lt: grad_D_eta(eta, lt),
            (log_theta_arr,),
            (jnp.ones(1),),
        )
        rhs = fd.X.T @ d_grad_D
        ift_dbeta = np.asarray(cho_solve((L_obs, True), -rhs))

        np.testing.assert_allclose(
            ift_dbeta,
            fd_dbeta,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="IFT dbeta/dtheta vs FD PIRLS perturbation mismatch",
        )


class TestStandardFamilyRegression:
    """Standard families (n_theta=0) are unaffected by the theta extension."""

    def test_poisson_gradient_unchanged(self):
        """Poisson gradient with joint_theta=False is finite and correct shape."""
        rng = np.random.default_rng(42)
        n = 80
        x = rng.uniform(0, 1, n)
        eta = np.sin(2 * np.pi * x) + 0.5
        y = rng.poisson(np.exp(eta)).astype(float)
        data = pd.DataFrame({"x": x, "y": y})

        family = Poisson()
        fd = _setup_fd(_FORMULA, data, family)
        _, log_lambda = _converge_pirls(fd)
        pirls_result, _ = _converge_pirls(fd)
        beta_warm = pirls_result.coefficients

        kwargs = _build_diff_score_kwargs(fd, joint_theta=False)
        grad = jax.grad(_diff_score, argnums=0)(log_lambda, beta_warm, **kwargs)

        assert jnp.all(jnp.isfinite(grad)), "Poisson gradient not finite"
        assert grad.shape == (fd.n_penalties,), "Wrong gradient shape"

    def test_nb_fixed_theta_uses_standard_path(self):
        """NB with fixed theta (n_theta=0) uses standard 2-primal path."""
        data = _make_nb_data(n=80, seed=42, true_theta=2.0)
        family = NegativeBinomial(theta=2.0)  # fixed, n_theta=0
        assert family.n_theta == 0

        fd = _setup_fd(_FORMULA, data, family)
        pirls_result, log_lambda = _converge_pirls(fd)
        beta_warm = pirls_result.coefficients

        kwargs = _build_diff_score_kwargs(fd, joint_theta=False)
        grad = jax.grad(_diff_score, argnums=0)(log_lambda, beta_warm, **kwargs)

        assert jnp.all(jnp.isfinite(grad)), "Fixed-theta NB gradient not finite"
        assert grad.shape == (fd.n_penalties,), "Wrong gradient shape"


class TestForwardPassTheta:
    """_fit_and_score_impl correctly parses theta from params."""

    def test_forward_score_finite(self, nb_problem):
        """Forward pass produces finite score with joint_theta=True."""
        fd, params, beta_warm, _ = nb_problem
        kwargs = _build_diff_score_kwargs(fd, joint_theta=True)

        score, pirls_result = _fit_and_score_impl(
            params,
            beta_warm,
            **kwargs,
        )
        assert jnp.isfinite(score), f"Non-finite score: {score}"
        assert jnp.all(jnp.isfinite(pirls_result.coefficients))

    def test_scores_match_between_paths(self, nb_problem):
        """Score from extended path matches standard path (same theta)."""
        fd, params, beta_warm, _ = nb_problem
        n_lambda = fd.n_penalties

        kwargs_ext = _build_diff_score_kwargs(fd, joint_theta=True)
        score_ext = float(_diff_score(params, beta_warm, **kwargs_ext))

        kwargs_std = _build_diff_score_kwargs(fd, joint_theta=False)
        score_std = float(_diff_score(params[:n_lambda], beta_warm, **kwargs_std))

        np.testing.assert_allclose(
            score_ext,
            score_std,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Extended and standard path scores differ",
        )

    def test_forward_score_matches_diff_score(self, nb_problem):
        """Forward pass score matches _diff_score."""
        fd, params, beta_warm, _ = nb_problem
        kwargs = _build_diff_score_kwargs(fd, joint_theta=True)

        score_forward, _ = _fit_and_score_impl(params, beta_warm, **kwargs)
        score_diff = _diff_score(params, beta_warm, **kwargs)

        np.testing.assert_allclose(
            float(score_forward),
            float(score_diff),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Forward pass and _diff_score scores differ",
        )


class TestJITCompilation:
    """Phase 2 code must JIT-compile (AGENTS.md line 149).

    All NB family methods used in newton.py's JIT-compiled paths must
    compile and run without error under ``jax.jit``.
    """

    def test_lgamma_diff_jit(self):
        """_lgamma_diff compiles under jax.jit."""
        from jaxgam.families.negative_binomial import _lgamma_diff

        theta = jnp.array(2.0)
        y = jnp.array([0.0, 1.0, 5.0, 10.0])
        result = jax.jit(_lgamma_diff, static_argnums=(2,))(theta, y, 10)
        assert jnp.all(jnp.isfinite(result))

    def test_lgamma_diff_grad_jit(self):
        """jax.grad through _lgamma_diff compiles under jax.jit."""
        from jaxgam.families.negative_binomial import _lgamma_diff

        y = jnp.array([0.0, 1.0, 5.0, 10.0])

        @jax.jit
        def _grad_fn(theta):
            return jax.grad(lambda t: jnp.sum(_lgamma_diff(t, y, 10)))(theta)

        grad = _grad_fn(jnp.array(2.0))
        assert jnp.isfinite(grad)

    def test_lgamma_diff_hessian_jit(self):
        """jax.hessian through _lgamma_diff compiles under jax.jit."""
        from jaxgam.families.negative_binomial import _lgamma_diff

        y = jnp.array([0.0, 1.0, 5.0, 10.0])

        @jax.jit
        def _hess_fn(theta):
            return jax.hessian(lambda t: jnp.sum(_lgamma_diff(t, y, 10)))(theta)

        hess = _hess_fn(jnp.array(2.0))
        assert jnp.isfinite(hess)

    def test_saturated_loglik_theta_jit(self, nb_problem):
        """saturated_loglik_theta compiles under jax.jit."""
        fd, _, _, _ = nb_problem
        family = fd.family

        @jax.jit
        def _fn(log_theta):
            return family.saturated_loglik_theta(
                fd.y,
                fd.wt,
                1.0,
                log_theta,
            )

        result = _fn(jnp.array([0.0]))
        assert jnp.isfinite(result)

    def test_saturated_loglik_theta_grad_jit(self, nb_problem):
        """jax.grad of saturated_loglik_theta compiles under jax.jit."""
        fd, _, _, _ = nb_problem
        family = fd.family

        @jax.jit
        def _grad_fn(log_theta):
            return jax.grad(
                lambda lt: family.saturated_loglik_theta(
                    fd.y,
                    fd.wt,
                    1.0,
                    lt,
                )
            )(log_theta)

        grad = _grad_fn(jnp.array([0.0]))
        assert jnp.all(jnp.isfinite(grad))

    def test_deviance_fn_jit(self, nb_problem):
        """deviance_fn pure function compiles under jax.jit."""
        fd, _, _, _ = nb_problem
        dev_fn = fd.family.deviance_fn(fd.y, fd.wt)

        eta = jnp.ones(fd.n_obs)
        log_theta = jnp.array([0.0])
        result = jax.jit(dev_fn)(eta, log_theta)
        assert jnp.isfinite(result)

    def test_deviance_fn_grad_jit(self, nb_problem):
        """jax.grad of deviance_fn compiles under jax.jit."""
        fd, _, _, _ = nb_problem
        dev_fn = fd.family.deviance_fn(fd.y, fd.wt)

        eta = jnp.ones(fd.n_obs)

        @jax.jit
        def _grad_fn(log_theta):
            return jax.grad(dev_fn, argnums=1)(eta, log_theta)

        grad = _grad_fn(jnp.array([0.0]))
        assert jnp.all(jnp.isfinite(grad))

    def test_working_weights_fn_jit(self, nb_problem):
        """working_weights_fn pure function compiles under jax.jit."""
        fd, _, _, _ = nb_problem
        ww_fn = fd.family.working_weights_fn(fd.wt)

        eta = jnp.ones(fd.n_obs)
        log_theta = jnp.array([0.0])
        result = jax.jit(ww_fn)(eta, log_theta)
        assert jnp.all(jnp.isfinite(result))

    def test_diff_score_grad_jit(self, nb_problem):
        """Full _diff_score gradient compiles under jax.jit (end-to-end)."""
        fd, params, beta_warm, _ = nb_problem
        kwargs = _build_diff_score_kwargs(fd, joint_theta=True)

        @jax.jit
        def _grad_fn(p):
            return jax.grad(_diff_score, argnums=0)(p, beta_warm, **kwargs)

        grad = _grad_fn(params)
        assert jnp.all(jnp.isfinite(grad))
