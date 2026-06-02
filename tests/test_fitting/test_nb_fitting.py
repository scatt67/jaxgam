"""Tests for end-to-end Negative Binomial fitting (PR 4).

Tests cover:
- Simple NB fit: convergence, theta estimation, deviance finite
- Fixed theta: no theta estimation
- Multiple smooths: y ~ s(x1) + s(x2) with NB
- Hard-gate invariants: deviance >= 0, no NaN, EDF bounds, H PSD
- Poisson limit: NB with large theta matches Poisson
- Numerical edge cases: zero-inflated, extreme overdispersion, large counts,
  constant response, mu near epsilon
- Standard family regression: existing families unaffected
- GAM API integration: full pipeline works

Design doc reference: docs/nb_implementation/design.md Sections 8-9
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.newton import NewtonResult, newton_optimize
from tests.helpers import (
    SEED,
    _AssertCollector,
    _make_nb_data,
    _setup_fd,
    check_that,
    r_available,
)
from tests.tolerances import LOOSE, STRICT

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _back_transform_coefs(result: NewtonResult, fd: FittingData) -> np.ndarray:
    """Back-transform coefficients from Sl.setup reparameterized space."""
    coefs = np.asarray(result.pirls_result.coefficients)
    if fd.repara_D is not None:
        coefs = np.asarray(fd.repara_D) @ coefs
    return coefs


def _make_poisson_data(
    n: int = 200,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate Poisson count data for Poisson limit tests."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x) + 0.5
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"x": x, "y": y})


# Relative tolerance for theta estimation bounds: estimated theta must be
# within [true * (1 - THETA_RTOL), true * (1 + THETA_RTOL)].
THETA_RTOL = 0.5

# ---------------------------------------------------------------------------
# Guard: max_y is computed from data in FittingData
# ---------------------------------------------------------------------------


class TestMaxYFromData:
    """Verify max_y is computed by FittingData.from_setup and passed through."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_fitting_data_computes_max_y(self):
        """FittingData.from_setup computes max_y from response vector."""
        data = _make_nb_data(true_theta=2.0)
        family = NegativeBinomial()
        fd = _setup_fd(self.FORMULA, data, family)
        expected = int(np.max(data["y"].values))
        assert fd.max_y == expected

    def test_max_y_correct_gives_nonzero_grad(self):
        """saturated_loglik_theta with correct max_y has nonzero theta grad."""
        family = NegativeBinomial()
        y = jnp.array([1.0, 2.0, 3.0])
        wt = jnp.ones(3)
        log_theta = jnp.array([0.0])
        max_y = 3
        grad = jax.grad(
            lambda lt: family.saturated_loglik_theta(y, wt, 1.0, lt, max_y=max_y)
        )(log_theta)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.abs(grad[0])) > 0


# ---------------------------------------------------------------------------
# A. Simple end-to-end NB fit
# ---------------------------------------------------------------------------


class TestNBSimpleFit:
    """Simple NB fitting: convergence, theta estimation, deviance."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(scope="class")
    def nb_fit(self):
        """Fit NB model with estimated theta."""
        data = _make_nb_data(true_theta=2.0)
        family = NegativeBinomial()
        fd = _setup_fd(self.FORMULA, data, family)
        result = newton_optimize(fd)
        return fd, result

    def test_converges(self, nb_fit):
        """NB fit converges."""
        _, result = nb_fit
        assert result.converged
        assert result.convergence_info == "full convergence"

    def test_theta_in_range(self, nb_fit):
        """Estimated theta is within THETA_RTOL of true theta=2."""
        true_theta = 2.0
        lo, hi = true_theta * (1 - THETA_RTOL), true_theta * (1 + THETA_RTOL)
        _, result = nb_fit
        assert result.theta is not None
        assert lo < result.theta < hi, f"theta={result.theta} outside [{lo}, {hi}]"

    def test_deviance_finite(self, nb_fit):
        """Deviance is finite and non-negative."""
        _, result = nb_fit
        dev = float(result.pirls_result.deviance)
        assert np.isfinite(dev)
        assert dev >= 0


# ---------------------------------------------------------------------------
# B. Fixed theta
# ---------------------------------------------------------------------------


class TestNBFixedTheta:
    """NB fitting with fixed theta (n_theta=0 path)."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(scope="class")
    def fixed_fit(self):
        """Fit NB model with fixed theta=2."""
        data = _make_nb_data(true_theta=2.0)
        family = NegativeBinomial(theta=2, fixed=True)
        fd = _setup_fd(self.FORMULA, data, family)
        result = newton_optimize(fd)
        return fd, result, family

    def test_converges(self, fixed_fit):
        """Fixed-theta NB converges."""
        _, result, _ = fixed_fit
        assert result.converged

    def test_result_theta_is_none(self, fixed_fit):
        """result.theta is None for fixed-theta NB (n_theta=0)."""
        _, result, _ = fixed_fit
        assert result.theta is None


# ---------------------------------------------------------------------------
# D. Hard-gate invariants
# ---------------------------------------------------------------------------


class TestNBHardGateInvariants:
    """Hard-gate invariants for NB models.

    Per CLAUDE.md: deviance >= 0, no NaN, EDF bounds, H PSD.
    """

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(
        params=[
            NegativeBinomial(),
            NegativeBinomial(theta=2, fixed=True),
        ],
        ids=["nb_estimated", "nb_fixed"],
    )
    def converged_result(self, request):
        """Fit NB model for hard-gate invariant testing."""
        family = request.param
        data = _make_nb_data(true_theta=2.0)
        fd = _setup_fd(self.FORMULA, data, family)
        result = newton_optimize(fd)
        return fd, result

    def test_deviance_non_negative(self, converged_result):
        """Deviance >= 0."""
        _, result = converged_result
        assert float(result.pirls_result.deviance) >= 0

    def test_no_nan_in_converged(self, converged_result):
        """All output arrays are finite when converged."""
        _, result = converged_result
        assert result.converged
        assert jnp.all(jnp.isfinite(result.pirls_result.coefficients))
        assert jnp.all(jnp.isfinite(result.pirls_result.mu))
        assert jnp.all(jnp.isfinite(result.pirls_result.eta))
        assert jnp.isfinite(result.pirls_result.deviance)
        assert jnp.isfinite(result.score)
        assert jnp.all(jnp.isfinite(result.gradient))
        assert jnp.isfinite(result.edf)
        assert jnp.isfinite(result.scale)
        assert jnp.all(jnp.isfinite(result.log_lambda))

    def test_edf_bounds(self, converged_result):
        """EDF satisfies 0 < edf <= n_coef."""
        fd, result = converged_result
        edf = float(result.edf)
        assert edf > 0, f"EDF {edf} must be positive"
        assert edf <= fd.n_coef, f"EDF {edf} exceeds n_coef {fd.n_coef}"

    def test_hessian_symmetric_psd(self, converged_result):
        """Penalized Hessian XtWX + S_lambda is symmetric PSD."""
        fd, result = converged_result
        XtWX = np.asarray(result.pirls_result.XtWX)
        S = np.asarray(fd.S_lambda(result.log_lambda))
        H = XtWX + S

        # Symmetry
        np.testing.assert_allclose(H, H.T, rtol=STRICT.rtol, atol=STRICT.atol)

        # PSD: all eigenvalues >= 0
        eigs = np.linalg.eigvalsh(H)
        assert np.all(eigs >= -STRICT.rtol), f"H has negative eigenvalue: {eigs.min()}"


# ---------------------------------------------------------------------------
# E. Poisson limit
# ---------------------------------------------------------------------------


class TestNBPoissonLimit:
    """NB with large theta should match Poisson."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(scope="class")
    def poisson_reference(self):
        """Fit Poisson model as reference."""
        data = _make_poisson_data()
        fd = _setup_fd(self.FORMULA, data, Poisson())
        result = newton_optimize(fd)
        return data, fd, result

    def test_fixed_large_theta_matches_poisson(self, poisson_reference):
        """NB(theta=1e4) on Poisson data ~ Poisson fit (LOOSE).

        NB deviance formula includes theta-dependent terms that don't
        vanish exactly at theta=1e4, so LOOSE (not MODERATE) is appropriate
        for cross-family comparisons.
        """
        data, pois_fd, pois_result = poisson_reference

        nb_family = NegativeBinomial(theta=1e4, fixed=True)
        nb_fd = _setup_fd(self.FORMULA, data, nb_family)
        nb_result = newton_optimize(nb_fd)

        assert nb_result.converged

        # Deviance
        np.testing.assert_allclose(
            float(nb_result.pirls_result.deviance),
            float(pois_result.pirls_result.deviance),
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="NB(1e4) deviance differs from Poisson",
        )

        # Coefficients
        nb_coefs = _back_transform_coefs(nb_result, nb_fd)
        pois_coefs = _back_transform_coefs(pois_result, pois_fd)
        np.testing.assert_allclose(
            nb_coefs,
            pois_coefs,
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="NB(1e4) coefficients differ from Poisson",
        )

    def test_estimated_theta_on_poisson_data(self, poisson_reference):
        """NB() on Poisson data: converges, theta > 1, all finite.

        Joint Newton over [log_lambda, log_theta] converges when the
        REML gradient is small.  For Poisson data the criterion is
        relatively flat in the theta direction, so Newton may converge
        at a moderate theta rather than pushing to infinity.

        The fixed-theta test above validates exact Poisson-limit
        behavior (NB(theta=1e4) ≈ Poisson).  This test verifies
        that NB with estimated theta handles equi-dispersed data
        gracefully: no divergence, positive theta, finite outputs.
        """
        data, _, _ = poisson_reference

        nb_family = NegativeBinomial()
        nb_fd = _setup_fd(self.FORMULA, data, nb_family)
        nb_result = newton_optimize(nb_fd)

        assert nb_result.converged
        assert nb_result.theta is not None
        assert nb_result.theta > 1, (
            f"theta={nb_result.theta} should be >1 for Poisson data"
        )
        assert np.isfinite(nb_result.theta)
        assert float(nb_result.pirls_result.deviance) >= 0
        assert jnp.all(jnp.isfinite(nb_result.pirls_result.coefficients))
        assert jnp.all(jnp.isfinite(nb_result.pirls_result.mu))


# ---------------------------------------------------------------------------
# F. Numerical edge cases
# ---------------------------------------------------------------------------


class TestNBEdgeCases:
    """Numerical edge cases for NB fitting."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_zero_inflated(self):
        """60%+ zeros: converges, theta > 0, deviance >= 0, mu > 0."""
        rng = np.random.default_rng(SEED)
        n = 300
        x = rng.uniform(0, 1, n)
        # Low eta -> small mu -> many zeros
        eta = -2.0 + 0.5 * np.sin(2 * np.pi * x)
        mu = np.exp(eta)
        theta = 0.5
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
        data = pd.DataFrame({"x": x, "y": y})
        assert np.mean(y == 0) >= 0.60, f"Need 60%+ zeros, got {np.mean(y == 0):.1%}"

        family = NegativeBinomial()
        fd = _setup_fd(self.FORMULA, data, family)
        result = newton_optimize(fd)

        assert result.converged
        assert result.theta is not None
        assert result.theta > 0
        assert np.isfinite(result.theta)
        assert float(result.pirls_result.deviance) >= 0
        assert jnp.all(jnp.isfinite(result.pirls_result.coefficients))
        assert jnp.all(jnp.isfinite(result.pirls_result.mu))
        # Fitted mu should be positive for all observations
        assert jnp.all(result.pirls_result.mu > 0)

    def test_extreme_overdispersion(self):
        """theta=0.1: converges, theta estimated near true value.

        With extreme overdispersion (theta=0.1), the high-variance data
        is challenging but n=500 provides enough signal for Newton to
        recover theta.  Uses a wider tolerance (50%) because the REML
        surface is flatter in the theta direction at small theta.
        """
        rng = np.random.default_rng(SEED)
        n = 500
        x = rng.uniform(0, 1, n)
        eta = np.sin(2 * np.pi * x) + 0.5
        mu = np.exp(eta)
        true_theta = 0.1
        y = rng.negative_binomial(
            n=true_theta, p=true_theta / (mu + true_theta), size=n
        ).astype(float)
        data = pd.DataFrame({"x": x, "y": y})

        family = NegativeBinomial()
        fd = _setup_fd(self.FORMULA, data, family)
        result = newton_optimize(fd)

        assert result.converged
        assert result.theta is not None
        # Wider tolerance for extreme overdispersion: REML surface is
        # flatter at small theta, so precision is lower than theta=2 case.
        extreme_rtol = 0.5
        lo = true_theta * (1 - extreme_rtol)
        hi = true_theta * (1 + extreme_rtol)
        assert lo < result.theta < hi, (
            f"theta={result.theta} outside [{lo}, {hi}] (true={true_theta})"
        )
        assert float(result.pirls_result.deviance) >= 0
        assert jnp.all(jnp.isfinite(result.pirls_result.coefficients))

    def test_large_counts(self):
        """max(y) > 500: completes, deviance finite, theta in range.

        Large counts stress the ``_lgamma_diff`` scan (runs for max_y
        iterations).  A higher theta gives moderate overdispersion and
        a more tractable REML surface.
        """
        rng = np.random.default_rng(SEED)
        n = 300
        x = rng.uniform(0, 1, n)
        # Large eta -> large mu -> large counts
        eta = np.sin(2 * np.pi * x) + 5.0
        mu = np.exp(eta)
        true_theta = 10.0
        y = rng.negative_binomial(
            n=true_theta, p=true_theta / (mu + true_theta), size=n
        ).astype(float)
        data = pd.DataFrame({"x": x, "y": y})
        assert np.max(y) > 500, f"Need max(y) > 500, got {np.max(y)}"

        family = NegativeBinomial()
        fd = _setup_fd(self.FORMULA, data, family)
        result = newton_optimize(fd)

        # Fit completes (converged or step-failed, not NaN crash)
        assert result.convergence_info in (
            "full convergence",
            "step failed",
        )
        dev = float(result.pirls_result.deviance)
        assert np.isfinite(dev)
        assert dev >= 0
        assert result.theta is not None
        assert result.theta > 0
        assert np.isfinite(result.theta)

    def test_constant_response(self):
        """All y=5: completes without divergence, theta finite, deviance near zero."""
        rng = np.random.default_rng(SEED)
        n = 200
        x = rng.uniform(0, 1, n)
        y = np.full(n, 5.0)
        data = pd.DataFrame({"x": x, "y": y})

        family = NegativeBinomial()
        fd = _setup_fd(self.FORMULA, data, family)
        result = newton_optimize(fd)

        # Fit completes (converged or step-failed, not stuck in infinite loop)
        assert result.convergence_info in (
            "full convergence",
            "step failed",
        )
        assert result.theta is not None
        assert result.theta > 0
        assert np.isfinite(result.theta)
        # Deviance should be near zero (perfect constant fit)
        assert float(result.pirls_result.deviance) < 1.0

    def test_sparse_predictor_mu_near_epsilon(self):
        """Wide mu range: no NaN/Inf, _MU_EPS guard works."""
        rng = np.random.default_rng(SEED)
        n = 200
        x = rng.uniform(0, 1, n)
        # eta from -6 to +3 gives mu from ~0.0025 to ~20
        eta = -6.0 + 9.0 * x
        mu = np.exp(eta)
        true_theta = 2.0
        y = rng.negative_binomial(
            n=true_theta, p=true_theta / (mu + true_theta), size=n
        ).astype(float)
        data = pd.DataFrame({"x": x, "y": y})

        family = NegativeBinomial()
        fd = _setup_fd(self.FORMULA, data, family)
        result = newton_optimize(fd)

        # Fit completes without NaN crash
        assert result.convergence_info in (
            "full convergence",
            "step failed",
        )
        assert jnp.all(jnp.isfinite(result.pirls_result.coefficients))
        assert jnp.all(jnp.isfinite(result.pirls_result.mu))
        assert float(result.pirls_result.deviance) >= 0


# ---------------------------------------------------------------------------
# G. GAM API integration
# ---------------------------------------------------------------------------


class TestNBGAMAPI:
    """Full GAM API integration for NB."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    def test_gam_fit_nb_instance(self):
        """GAM.fit() with NegativeBinomial() instance works."""
        from jaxgam.api import GAM

        data = _make_nb_data(true_theta=2.0)
        result = GAM(self.FORMULA, family=NegativeBinomial()).fit(data)

        assert result.converged
        assert np.isfinite(result.deviance)
        assert result.deviance >= 0
        # Theta accessible via family object
        theta = float(result.family.get_theta(transformed=True)[0])
        assert theta > 0
        assert np.isfinite(theta)

    def test_predict_roundtrip(self):
        """predict(training_data) reproduces fitted_values."""
        from jaxgam.api import GAM

        data = _make_nb_data(true_theta=2.0)
        result = GAM(self.FORMULA, family=NegativeBinomial()).fit(data)

        pred = result.predict()
        np.testing.assert_allclose(
            pred,
            result.fitted_values,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Predict roundtrip failed",
        )


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestNBNonCanonicalLink:
    """Finding H4: NB with a non-canonical link must match mgcv.

    The observed-information XtWX feeding the REML log|H| used positive-clamped
    weights, discarding the negative per-observation weights mgcv keeps
    (gdi.c:2481-2498), and the inner PIRLS solve dropped those observations too.
    For the canonical log link observed weights stay positive (no-op, so log
    already matched); identity/sqrt go negative and diverged (identity Δdev~20.8).
    The fix uses signed observed information in both the log|H| and the inner
    Newton WLS (mgcv gam.fit4).
    """

    @staticmethod
    def _nb_signal(n: int = 200, true_theta: float = 4.0) -> pd.DataFrame:
        rng = np.random.default_rng(SEED)
        x = np.sort(rng.uniform(0.0, 1.0, n))
        mu = 3.0 + 2.0 * np.sin(2.0 * np.pi * x) ** 2  # in [3,5] -> identity valid
        p = true_theta / (true_theta + mu)
        y = rng.negative_binomial(true_theta, p).astype(float)
        return pd.DataFrame({"x": x, "y": y})

    def test_nb_noncanonical_link_matches_r(self) -> None:
        """NB deviance/theta for log, identity and sqrt links all match mgcv."""
        from tests.r_bridge import RBridge

        df = self._nb_signal()
        formula = "y ~ s(x, k=10, bs='cr')"
        bridge = RBridge()
        cases = {"log": "nb", "identity": "nb_identity", "sqrt": "nb_sqrt"}

        coll = _AssertCollector()
        for link, r_family in cases.items():
            m = GAM(formula, family=NegativeBinomial(theta=1.0, link=link)).fit(df)
            r = bridge.fit_gam(formula, df, family=r_family)
            coll.check(
                f"{link}: converged",
                lambda m=m, link=link: check_that(
                    m.converged, f"NB {link} did not converge"
                ),
            )
            coll.check(
                f"{link}: deviance vs R",
                lambda m=m, r=r: np.testing.assert_allclose(
                    float(m.deviance),
                    r["deviance"],
                    rtol=LOOSE.rtol,
                    atol=LOOSE.atol,
                ),
            )
            coll.check(
                f"{link}: theta vs R",
                lambda m=m, r=r, link=link: check_that(
                    m.theta is not None
                    and abs(float(m.theta) - float(r["theta"]))
                    <= LOOSE.atol + LOOSE.rtol * abs(float(r["theta"])),
                    f"theta jaxgam={m.theta} vs R={r['theta']} (link={link})",
                ),
            )
        coll.raise_if_any("NB non-canonical link R parity (H4)")
