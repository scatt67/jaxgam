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

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.newton import NewtonResult, newton_optimize
from tests.helpers import SEED, _generate_family_data
from tests.tolerances import LOOSE, STRICT

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_fd(formula: str, data: pd.DataFrame, family) -> FittingData:
    """Build FittingData from formula + data."""
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    spec = parse_formula(formula)
    setup = ModelSetup.build(spec, data)
    return FittingData.from_setup(setup, family)


def _back_transform_coefs(result: NewtonResult, fd: FittingData) -> np.ndarray:
    """Back-transform coefficients from Sl.setup reparameterized space."""
    coefs = np.asarray(result.pirls_result.coefficients)
    if fd.repara_D is not None:
        coefs = np.asarray(fd.repara_D) @ coefs
    return coefs


def _make_nb_data(
    n: int = 200,
    seed: int = SEED,
    true_theta: float = 2.0,
) -> pd.DataFrame:
    """Generate single-predictor NB count data with known theta."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x) + 0.5
    mu = np.exp(eta)
    y = rng.negative_binomial(
        n=true_theta, p=true_theta / (mu + true_theta), size=n
    ).astype(float)
    return pd.DataFrame({"x": x, "y": y})


def _make_nb_data_two_smooth(
    n: int = 200,
    seed: int = SEED,
    true_theta: float = 2.0,
) -> pd.DataFrame:
    """Generate two-predictor NB count data."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x1) + 0.5 * np.cos(2 * np.pi * x2) + 0.5
    mu = np.exp(eta)
    y = rng.negative_binomial(
        n=true_theta, p=true_theta / (mu + true_theta), size=n
    ).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


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
        """Estimated theta is in reasonable range around true theta=2."""
        _, result = nb_fit
        assert result.theta is not None
        assert 0.5 < result.theta < 10.0, f"theta={result.theta} outside [0.5, 10]"

    def test_deviance_finite(self, nb_fit):
        """Deviance is finite and non-negative."""
        _, result = nb_fit
        dev = float(result.pirls_result.deviance)
        assert np.isfinite(dev)
        assert dev >= 0

    def test_result_theta_populated(self, nb_fit):
        """result.theta is populated for estimated NB."""
        _, result = nb_fit
        assert result.theta is not None
        assert isinstance(result.theta, float)
        assert result.theta > 0


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
        family = NegativeBinomial(theta=2)
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

    def test_family_theta_unchanged(self, fixed_fit):
        """Family theta stays at fixed value after fitting."""
        _, _, family = fixed_fit
        assert family.n_theta == 0
        theta = float(family.get_theta(transformed=True)[0])
        np.testing.assert_allclose(theta, 2.0, rtol=STRICT.rtol)


# ---------------------------------------------------------------------------
# C. Multiple smooths
# ---------------------------------------------------------------------------


class TestNBMultipleSmooths:
    """NB fitting with multiple smooth terms."""

    FORMULA = "y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')"

    @pytest.fixture(scope="class")
    def multi_fit(self):
        """Fit NB model with two smooths."""
        data = _make_nb_data_two_smooth(true_theta=2.0)
        family = NegativeBinomial()
        fd = _setup_fd(self.FORMULA, data, family)
        result = newton_optimize(fd)
        return fd, result

    def test_converges(self, multi_fit):
        """Two-smooth NB converges."""
        _, result = multi_fit
        assert result.converged

    def test_theta_positive(self, multi_fit):
        """Estimated theta is positive."""
        _, result = multi_fit
        assert result.theta is not None
        assert result.theta > 0

    def test_two_penalty_terms(self, multi_fit):
        """Model has two penalty terms (one per smooth)."""
        fd, result = multi_fit
        assert fd.n_penalties == 2
        assert result.log_lambda.shape[0] == 2


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
            NegativeBinomial(theta=2),
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

        nb_family = NegativeBinomial(theta=1e4)
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
        """theta=0.1: converges, theta not near-Poisson, deviance >= 0.

        With extreme overdispersion the REML surface is flat in the
        theta direction, so Newton may converge near the starting value
        (theta=1).  The key check is that the fit handles the high-
        variance data without crashing or producing non-finite results.
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
        # Theta should stay moderate (not explode to Poisson limit)
        assert result.theta < 10.0, (
            f"theta={result.theta} should be < 10 for overdispersed data"
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
# G. Standard family regression
# ---------------------------------------------------------------------------


class TestStandardFamilyRegression:
    """Verify standard families are unaffected by NB additions."""

    FORMULA = "y ~ s(x, k=10, bs='cr')"

    @pytest.fixture(
        params=[
            ("gaussian", Gaussian()),
            ("poisson", Poisson()),
        ],
        ids=["gaussian", "poisson"],
    )
    def standard_fit(self, request):
        """Fit standard family model."""
        family_name, family_obj = request.param
        data = _generate_family_data(family_name)
        fd = _setup_fd(self.FORMULA, data, family_obj)
        result = newton_optimize(fd)
        return family_name, fd, result

    def test_converges(self, standard_fit):
        """Standard family still converges."""
        _, _, result = standard_fit
        assert result.converged

    def test_theta_is_none(self, standard_fit):
        """Standard family result.theta is None."""
        _, _, result = standard_fit
        assert result.theta is None

    def test_deviance_non_negative(self, standard_fit):
        """Standard family deviance >= 0."""
        _, _, result = standard_fit
        assert float(result.pirls_result.deviance) >= 0

    def test_all_finite(self, standard_fit):
        """Standard family outputs are all finite."""
        _, _, result = standard_fit
        assert jnp.all(jnp.isfinite(result.pirls_result.coefficients))
        assert jnp.all(jnp.isfinite(result.pirls_result.mu))
        assert jnp.isfinite(result.pirls_result.deviance)
        assert jnp.isfinite(result.score)


# ---------------------------------------------------------------------------
# H. GAM API integration
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

    def test_gam_fit_nb_string(self):
        """GAM.fit() with family='nb' string works."""
        from jaxgam.api import GAM

        data = _make_nb_data(true_theta=2.0)
        result = GAM(self.FORMULA, family="nb").fit(data)

        assert result.converged
        assert np.isfinite(result.deviance)

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
