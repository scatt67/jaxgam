"""Tests for the PIRLS inner loop.

Tests cover:
- Convergence for all four standard families (Gaussian, Binomial, Poisson, Gamma)
- Step-halving activation under adversarial initialization
- Monotonic decrease of penalized deviance
- JIT compilation compatibility
- Offset handling
- Penalized (ridge) shrinkage
- R comparison via RBridge against mgcv::gam(sp=...)
- Single PIRLS step correctness

Design doc reference: Section 7.2
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pymgcv.families.base import ExponentialFamily
from pymgcv.families.standard import Binomial, Gamma, Gaussian, Poisson
from pymgcv.fitting.initialization import initialize_beta
from pymgcv.fitting.pirls import (
    PIRLSResult,
    _penalized_deviance,
    _pirls_step,
    pirls_loop,
)
from pymgcv.jax_utils import to_jax, to_numpy
from tests.tolerances import MODERATE, STRICT

jax.config.update("jax_enable_x64", True)

# ---- Seed and helpers ----

SEED = 42


def _make_polynomial_data(
    n: int = 200, p: int = 3, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create polynomial design matrix X = [1, x, x^2, ...] and Gaussian y."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    X = np.column_stack([x**j for j in range(p)])
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + rng.normal(0, 0.1, n)
    return X, y, beta_true


def _make_glm_data(
    family_name: str, n: int = 200, p: int = 2, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, y) for a GLM with known linear predictor.

    X is a polynomial basis [1, x, x^2, ...] with *p* columns.
    The response y is generated from a simple linear predictor
    (intercept + slope), so higher-order columns act as noise
    that the penalty should regularise toward zero.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.05, 0.95, n)
    X = np.column_stack([x**j for j in range(p)])

    if family_name == "gaussian":
        eta = 1.0 + 2.0 * x
        y = eta + rng.normal(0, 0.5, n)
    elif family_name == "binomial":
        eta = -3.0 + 6.0 * x
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        eta = 0.5 + 1.5 * x
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(float)
    elif family_name == "gamma":
        eta = 0.5 + 1.0 * x  # positive, suitable for inverse link
        mu = 1.0 / eta
        shape = 5.0
        y = rng.gamma(shape, scale=mu / shape, size=n)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return X, y


def _init_and_transfer(X, y, wt, family, S_lambda_np, offset=None):
    """Initialize beta (Phase 1, CPU) then transfer to device (Phase 1→2).

    Returns (X_d, y_d, beta_d, S_d, wt_d) on device, plus offset_d if given.
    """
    beta_init = initialize_beta(X, y, wt, family, offset=offset)
    X_d, y_d, S_d, wt_d = to_jax(X, y, S_lambda_np, wt)
    # beta_init is already a JAX array from initialize_beta
    beta_d = to_jax(np.asarray(beta_init))
    if offset is not None:
        offset_d = to_jax(offset)
        return X_d, y_d, beta_d, S_d, wt_d, offset_d
    return X_d, y_d, beta_d, S_d, wt_d


# ---- Test classes ----


class TestGaussianConverges:
    """Gaussian PIRLS should converge in very few iterations."""

    @classmethod
    def setup_class(cls):
        cls.X, cls.y, cls.beta_true = _make_polynomial_data(n=200, p=3)
        cls.family = Gaussian()
        wt = np.ones(len(cls.y))
        S_lambda_np = np.zeros((3, 3))
        X_d, y_d, beta_d, S_d, _wt_d = _init_and_transfer(
            cls.X, cls.y, wt, cls.family, S_lambda_np
        )
        cls.result = pirls_loop(X_d, y_d, beta_d, S_d, cls.family)

    def test_converges(self):
        assert isinstance(self.result, PIRLSResult)
        assert self.result.converged
        assert self.result.n_iter <= 5

    def test_coefficients_match_lstsq(self):
        """Unpenalized Gaussian PIRLS should match OLS."""
        beta_ols, _, _, _ = np.linalg.lstsq(self.X, self.y, rcond=None)
        np.testing.assert_allclose(
            to_numpy(self.result.coefficients),
            beta_ols,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Gaussian PIRLS coefficients should match OLS",
        )


class TestBinomialConverges:
    @classmethod
    def setup_class(cls):
        cls.X, cls.y = _make_glm_data("binomial", p=6)
        cls.family = Binomial()
        wt = np.ones(len(cls.y))
        S_lambda_np = np.eye(6)
        X_d, y_d, beta_d, S_d, _wt_d = _init_and_transfer(
            cls.X, cls.y, wt, cls.family, S_lambda_np
        )
        cls.result = pirls_loop(X_d, y_d, beta_d, S_d, cls.family)

    def test_converges(self):
        assert self.result.converged
        assert self.result.n_iter < 25

    def test_mu_in_bounds(self):
        mu = to_numpy(self.result.mu)
        assert np.all(mu > 0)
        assert np.all(mu < 1)


class TestPoissonConverges:
    @classmethod
    def setup_class(cls):
        cls.X, cls.y = _make_glm_data("poisson", p=6)
        cls.family = Poisson()
        wt = np.ones(len(cls.y))
        S_lambda_np = np.eye(6)
        X_d, y_d, beta_d, S_d, _wt_d = _init_and_transfer(
            cls.X, cls.y, wt, cls.family, S_lambda_np
        )
        cls.result = pirls_loop(X_d, y_d, beta_d, S_d, cls.family)

    def test_converges(self):
        assert self.result.converged
        assert self.result.n_iter < 25

    def test_mu_positive(self):
        assert np.all(to_numpy(self.result.mu) > 0)


class TestGammaConverges:
    @classmethod
    def setup_class(cls):
        cls.X, cls.y = _make_glm_data("gamma", p=6)
        cls.family = Gamma()
        wt = np.ones(len(cls.y))
        S_lambda_np = np.eye(6)
        X_d, y_d, beta_d, S_d, _wt_d = _init_and_transfer(
            cls.X, cls.y, wt, cls.family, S_lambda_np
        )
        cls.result = pirls_loop(X_d, y_d, beta_d, S_d, cls.family)

    def test_converges(self):
        assert self.result.converged
        assert self.result.n_iter < 25

    def test_mu_positive(self):
        assert np.all(to_numpy(self.result.mu) > 0)


class TestStepHalving:
    """Step-halving should rescue PIRLS from poor initial values."""

    def test_converges_from_adversarial_init(self):
        X, y = _make_glm_data("binomial", n=200)
        family = Binomial()
        S_lambda_np = np.zeros((2, 2))

        X_d, y_d, S_d = to_jax(X, y, S_lambda_np)
        # Adversarial initialization: very far from truth
        beta_init = to_jax(np.array([10.0, -20.0]))
        result = pirls_loop(X_d, y_d, beta_init, S_d, family, max_iter=200)

        assert result.converged


class TestMonotonicity:
    """Penalized deviance should not increase (within tolerance)."""

    @pytest.mark.parametrize(
        ("family_name", "family"),
        [
            ("gaussian", Gaussian()),
            ("binomial", Binomial()),
            ("poisson", Poisson()),
            ("gamma", Gamma()),
        ],
    )
    def test_final_pen_dev_leq_initial(self, family_name, family):
        X, y = _make_glm_data(family_name, p=6)
        n = len(y)
        p = X.shape[1]
        wt = np.ones(n)
        S_lambda_np = np.eye(p)

        X_d, y_d, beta_d, S_d, wt_d = _init_and_transfer(X, y, wt, family, S_lambda_np)

        # Compute initial penalized deviance
        eta_init = X_d @ beta_d
        mu_init = family.link.inverse(eta_init)
        pen_dev_init = _penalized_deviance(beta_d, mu_init, y_d, wt_d, S_d, family)

        result = pirls_loop(X_d, y_d, beta_d, S_d, family)

        assert float(result.penalized_deviance) <= float(pen_dev_init) + MODERATE.atol


class TestJITCompilation:
    """PIRLS should work under jax.jit for all families."""

    def test_jit_gaussian(self):
        X, y, _ = _make_polynomial_data(n=50, p=2)
        family = Gaussian()
        wt = np.ones(len(y))
        S_lambda_np = np.zeros((2, 2))

        X_d, y_d, beta_d, S_d, wt_d = _init_and_transfer(X, y, wt, family, S_lambda_np)

        jit_pirls = jax.jit(pirls_loop, static_argnames=("family", "max_iter", "tol"))
        result = jit_pirls(X_d, y_d, beta_d, S_d, family, wt=wt_d)

        assert result.converged

    def test_jit_poisson_penalized(self):
        X, y = _make_glm_data("poisson", n=50, p=4)
        family = Poisson()
        wt = np.ones(len(y))
        p = X.shape[1]
        S_lambda_np = np.eye(p)

        X_d, y_d, beta_d, S_d, wt_d = _init_and_transfer(X, y, wt, family, S_lambda_np)

        jit_pirls = jax.jit(pirls_loop, static_argnames=("family", "max_iter", "tol"))
        result = jit_pirls(X_d, y_d, beta_d, S_d, family, wt=wt_d)

        assert result.converged


class TestOffset:
    """Offset should be handled correctly."""

    def test_offset_equivalence(self):
        """Fit with offset should give same fitted values as without."""
        rng = np.random.default_rng(SEED)
        n = 200
        x = np.linspace(0, 1, n)
        offset_val = 2.0
        beta_true = np.array([1.0, 3.0])
        # y = 1 + 3x + 2 + noise = 3 + 3x + noise
        y = (
            np.column_stack([np.ones(n), x]) @ beta_true
            + offset_val
            + rng.normal(0, 0.1, n)
        )

        family = Gaussian()
        wt = np.ones(n)
        X = np.column_stack([np.ones(n), x])
        S_lambda_np = np.zeros((2, 2))
        offset = np.full(n, offset_val)

        # Fit WITHOUT offset
        X_d, y_d, beta_no, S_d, _wt_d = _init_and_transfer(
            X, y, wt, family, S_lambda_np
        )
        result_no = pirls_loop(X_d, y_d, beta_no, S_d, family)

        # Fit WITH offset
        *_, beta_off, _S_d2, _wt_d2, offset_d = _init_and_transfer(
            X, y, wt, family, S_lambda_np, offset=offset
        )
        result_off = pirls_loop(X_d, y_d, beta_off, S_d, family, offset=offset_d)

        # Fitted values (eta) should match
        np.testing.assert_allclose(
            to_numpy(result_off.eta),
            to_numpy(result_no.eta),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Fitted etas should match with/without offset",
        )

        # Intercept difference should equal the offset
        coef_no = to_numpy(result_no.coefficients)
        coef_off = to_numpy(result_off.coefficients)
        np.testing.assert_allclose(
            coef_no[0] - coef_off[0],
            offset_val,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="Intercept difference should equal offset",
        )


class TestPenalized:
    """Ridge penalty should shrink coefficients for all families."""

    @pytest.mark.parametrize(
        ("family_name", "family"),
        [
            ("gaussian", Gaussian()),
            ("binomial", Binomial()),
            ("poisson", Poisson()),
            ("gamma", Gamma()),
        ],
    )
    def test_ridge_shrinkage(self, family_name, family):
        X, y = _make_glm_data(family_name, n=200, p=5)
        wt = np.ones(len(y))
        p = X.shape[1]

        beta_init = initialize_beta(X, y, wt, family)
        X_d, y_d, beta_d = to_jax(X, y, np.asarray(beta_init))

        # Unpenalized
        S_zero = to_jax(np.zeros((p, p)))
        result_unpen = pirls_loop(X_d, y_d, beta_d, S_zero, family)

        # Heavily penalized (ridge)
        S_ridge = to_jax(100.0 * np.eye(p))
        result_pen = pirls_loop(X_d, y_d, beta_d, S_ridge, family)

        # Penalized coefficients should be smaller in norm
        norm_unpen = float(jnp.linalg.norm(result_unpen.coefficients))
        norm_pen = float(jnp.linalg.norm(result_pen.coefficients))
        assert norm_pen < norm_unpen, (
            f"Ridge penalty should shrink {family_name} coefficient norm"
        )


class TestPIRLSStep:
    """Test a single PIRLS step for correctness."""

    def test_single_step_gaussian(self):
        """A single Gaussian PIRLS step should produce valid XtWX."""
        X, y, _ = _make_polynomial_data(n=100, p=3)
        family = Gaussian()
        n = len(y)
        wt = np.ones(n)
        S_lambda_np = np.zeros((3, 3))

        beta_init = initialize_beta(X, y, wt, family)
        X_d, y_d, wt_d, S_d, beta_d = to_jax(
            X, y, wt, S_lambda_np, np.asarray(beta_init)
        )
        offset_d = to_jax(np.zeros(n))

        eta = X_d @ beta_d + offset_d
        mu = family.link.inverse(eta)

        _beta_new, XtWX, _L, W = _pirls_step(X_d, y_d, wt_d, beta_d, mu, S_d, family)

        # XtWX should be symmetric positive definite for Gaussian
        XtWX_np = to_numpy(XtWX)
        np.testing.assert_allclose(
            XtWX_np, XtWX_np.T, rtol=STRICT.rtol, atol=STRICT.atol
        )
        eigvals = np.linalg.eigvalsh(XtWX_np)
        assert np.all(eigvals > 0), "XtWX should be positive definite"

        # For Gaussian with identity link, W should be all 1s (= wt)
        np.testing.assert_allclose(
            to_numpy(W), np.ones(n), rtol=STRICT.rtol, atol=STRICT.atol
        )


# ---- R comparison tests ----


def _r_available() -> bool:
    """Check if R and mgcv are available with correct versions."""
    from tests.r_bridge import RBridge

    if not RBridge.available():
        return False
    ok, _ = RBridge.check_versions()
    return ok


@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestVsR:
    """Compare PIRLS output against mgcv::gam() with REML-estimated sp.

    Uses the full Python pipeline (parse_formula → ModelSetup → FittingData
    → pirls_loop) and compares coefficients against R's gam() output.
    R's smoothing parameters are used to build S_lambda so we isolate
    PIRLS numerics from REML optimization.
    """

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @staticmethod
    def _make_data(family_name: str) -> pd.DataFrame:
        """Generate synthetic data as a DataFrame."""
        rng = np.random.default_rng(SEED)
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

    def _setup(self, family_name: str, family_r: str, family: ExponentialFamily):
        """Build Python pipeline and get R reference values.

        Returns (fd, beta_jax, log_lambda, r_ref) where fd is a FittingData
        from our full Python pipeline, log_lambda uses R's REML-estimated
        smoothing parameters, and r_ref is a dict with R's coefficients,
        deviance, and fitted_values.
        """
        from pymgcv.fitting.data import FittingData
        from pymgcv.formula.design import ModelSetup
        from pymgcv.formula.parser import parse_formula
        from tests.r_bridge import RBridge

        data = self._make_data(family_name)

        # Python pipeline: parse → setup → transfer
        spec = parse_formula(self.FORMULA)
        setup = ModelSetup.build(spec, data)
        fd = FittingData.from_setup(setup, family)

        # Initialize beta from the (possibly reparameterized) model matrix
        # stored in fd.X, matching the coordinate system PIRLS will use.
        beta_init = initialize_beta(
            np.asarray(fd.X), setup.y, setup.weights, family, setup.offset
        )
        beta_jax = to_jax(np.asarray(beta_init))

        # R reference: fit gam and extract sp + coefficients + deviance + fitted
        bridge = RBridge()
        r_result = bridge.fit_gam(self.FORMULA, data, family=family_r)
        log_lambda = jnp.log(jnp.array(r_result["smoothing_params"]))

        r_ref = {
            "coefficients": r_result["coefficients"],
            "deviance": r_result["deviance"],
            "fitted_values": r_result["fitted_values"],
        }
        return fd, beta_jax, log_lambda, r_ref

    @staticmethod
    def _run_pirls(fd, beta_jax, log_lambda):
        """Run PIRLS using FittingData and R's smoothing parameters."""
        S_lambda = fd.S_lambda(log_lambda)
        return pirls_loop(fd.X, fd.y, beta_jax, S_lambda, fd.family, fd.wt, fd.offset)

    def _check_vs_r(self, result, r_ref, label, fd=None):
        """Compare PIRLS result against R reference for coefficients,
        deviance, and fitted values."""
        coefs = to_numpy(result.coefficients)
        if fd is not None and fd.repara_D is not None:
            coefs = to_numpy(fd.repara_D) @ coefs
        np.testing.assert_allclose(
            coefs,
            r_ref["coefficients"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=f"{label} PIRLS coefficients differ from R",
        )
        np.testing.assert_allclose(
            float(result.deviance),
            r_ref["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=f"{label} PIRLS deviance differs from R",
        )
        np.testing.assert_allclose(
            to_numpy(result.mu),
            r_ref["fitted_values"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=f"{label} PIRLS fitted values differ from R",
        )

    def test_vs_r_gaussian(self):
        fd, beta_jax, log_lambda, r_ref = self._setup(
            "gaussian", "gaussian", Gaussian()
        )
        result = self._run_pirls(fd, beta_jax, log_lambda)
        self._check_vs_r(result, r_ref, "Gaussian", fd)

    def test_vs_r_binomial(self):
        fd, beta_jax, log_lambda, r_ref = self._setup(
            "binomial", "binomial", Binomial()
        )
        result = self._run_pirls(fd, beta_jax, log_lambda)
        self._check_vs_r(result, r_ref, "Binomial", fd)

    def test_vs_r_poisson(self):
        fd, beta_jax, log_lambda, r_ref = self._setup("poisson", "poisson", Poisson())
        result = self._run_pirls(fd, beta_jax, log_lambda)
        self._check_vs_r(result, r_ref, "Poisson", fd)

    def test_vs_r_gamma(self):
        fd, beta_jax, log_lambda, r_ref = self._setup("gamma", "gamma", Gamma())
        result = self._run_pirls(fd, beta_jax, log_lambda)
        self._check_vs_r(result, r_ref, "Gamma", fd)
