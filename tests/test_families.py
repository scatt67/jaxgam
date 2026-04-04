"""Tests for jaxgam.families.

Coverage:
1. TestVariance — V(mu) correct for each family at STRICT tolerance
2. TestDevianceResids — deviance residuals match R at STRICT on synthetic data
3. TestWorkingWeights — 1/(V(mu)*g'(mu)^2) computed correctly
4. TestInitialization — family.initialize(y, wt) produces valid starting mu
5. TestEdgeCases — Binomial y=0/y=1, Poisson y=0, Gamma small mu
6. TestRegistry — get_family("gaussian") returns Gaussian, etc.
7. TestNoJaxImports — importing jaxgam.families doesn't trigger jax import
8. TestExtendedFamilyContract — generic contract for all ExtendedFamily subclasses
9. TestNBSpecific — NB-specific tests (constructor, alpha, Poisson limit)
10. TestExtendedFamilyAD — finite-difference validation of AD through extended families
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.families import (
    Binomial,
    ExponentialFamily,
    ExtendedFamily,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    get_family,
)
from jaxgam.links.links import LogLink
from tests.tolerances import LOOSE, MODERATE, STRICT

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 123


def _r_available() -> bool:
    """Check if R is available with correct versions."""
    try:
        from tests.r_bridge import RBridge

        if not RBridge.available():
            return False
        ok, _ = RBridge.check_versions()
        return ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test 0: ResponseSupport
# ---------------------------------------------------------------------------


class TestResponseSupport:
    """Verify ResponseSupport bounds checking handles inclusive/exclusive correctly."""

    def test_real_accepts_anything(self) -> None:
        from jaxgam.families.base import REAL

        assert REAL.check(np.array([-1e10, 0.0, 1e10]))

    def test_non_negative_accepts_zero(self) -> None:
        from jaxgam.families.base import NON_NEGATIVE

        assert NON_NEGATIVE.check(np.array([0.0, 1.0, 100.0]))

    def test_non_negative_rejects_negative(self) -> None:
        from jaxgam.families.base import NON_NEGATIVE

        assert not NON_NEGATIVE.check(np.array([0.0, -0.001]))

    def test_positive_rejects_zero(self) -> None:
        from jaxgam.families.base import POSITIVE

        assert not POSITIVE.check(np.array([0.0, 1.0]))

    def test_positive_accepts_small(self) -> None:
        from jaxgam.families.base import POSITIVE

        assert POSITIVE.check(np.array([1e-300, 1.0]))

    def test_unit_interval_accepts_boundaries(self) -> None:
        from jaxgam.families.base import UNIT_INTERVAL

        assert UNIT_INTERVAL.check(np.array([0.0, 0.5, 1.0]))

    def test_unit_interval_rejects_above(self) -> None:
        from jaxgam.families.base import UNIT_INTERVAL

        assert not UNIT_INTERVAL.check(np.array([0.5, 1.001]))

    def test_unit_interval_rejects_below(self) -> None:
        from jaxgam.families.base import UNIT_INTERVAL

        assert not UNIT_INTERVAL.check(np.array([-0.001, 0.5]))

    def test_str_representation(self) -> None:
        from jaxgam.families.base import NON_NEGATIVE, POSITIVE, REAL, UNIT_INTERVAL

        assert str(REAL) == "[-inf, inf]"
        assert str(NON_NEGATIVE) == "[0, inf]"
        assert str(POSITIVE) == "(0, inf]"
        assert str(UNIT_INTERVAL) == "[0, 1]"


# ---------------------------------------------------------------------------
# Test 1: Variance functions
# ---------------------------------------------------------------------------


class TestVariance:
    """V(mu) must match the known analytical form at STRICT tolerance."""

    def test_gaussian_variance(self) -> None:
        """Gaussian V(mu) = 1."""
        mu = np.array([0.1, 0.5, 1.0, 5.0, 100.0])
        fam = Gaussian()
        v = fam.variance(mu)
        np.testing.assert_allclose(
            v,
            np.ones_like(mu),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_binomial_variance(self) -> None:
        """Binomial V(mu) = mu * (1 - mu)."""
        mu = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        fam = Binomial()
        v = fam.variance(mu)
        expected = mu * (1.0 - mu)
        np.testing.assert_allclose(
            v,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_poisson_variance(self) -> None:
        """Poisson V(mu) = mu."""
        mu = np.array([0.01, 0.1, 1.0, 5.0, 100.0])
        fam = Poisson()
        v = fam.variance(mu)
        np.testing.assert_allclose(
            v,
            mu,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_gamma_variance(self) -> None:
        """Gamma V(mu) = mu^2."""
        mu = np.array([0.01, 0.1, 1.0, 5.0, 100.0])
        fam = Gamma()
        v = fam.variance(mu)
        expected = mu**2
        np.testing.assert_allclose(
            v,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_nb_variance(self) -> None:
        """NB V(mu) = mu + mu^2/theta."""
        mu = np.array([0.01, 0.1, 1.0, 5.0, 100.0])
        for theta_val in [0.5, 2.0, 10.0, 100.0]:
            fam = NegativeBinomial(theta=theta_val)
            v = fam.variance(mu)
            expected = mu + mu**2 / theta_val
            np.testing.assert_allclose(
                v,
                expected,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"NB variance at theta={theta_val}",
            )

    def test_nb_dvar(self) -> None:
        """NB V'(mu) = 1 + 2*mu/theta."""
        mu = jnp.array([0.01, 0.1, 1.0, 5.0, 100.0])
        for theta_val in [0.5, 2.0, 10.0]:
            fam = NegativeBinomial(theta=theta_val)
            dv = fam.dvar(mu)
            expected = 1.0 + 2.0 * mu / theta_val
            np.testing.assert_allclose(
                dv,
                expected,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"NB dvar at theta={theta_val}",
            )


# ---------------------------------------------------------------------------
# Test 2: Deviance residuals vs R
# ---------------------------------------------------------------------------


def _compute_r_dev_resids(
    family_r: str, y: np.ndarray, mu: np.ndarray, wt: np.ndarray
) -> np.ndarray:
    """Compute R's family$dev.resids(y, mu, wt) via subprocess."""
    with tempfile.TemporaryDirectory() as tmpdir:
        y_path = os.path.join(tmpdir, "y.csv")
        mu_path = os.path.join(tmpdir, "mu.csv")
        wt_path = os.path.join(tmpdir, "wt.csv")
        out_path = os.path.join(tmpdir, "result.json")

        np.savetxt(y_path, y, delimiter=",")
        np.savetxt(mu_path, mu, delimiter=",")
        np.savetxt(wt_path, wt, delimiter=",")

        script = f"""\
y <- scan("{y_path}", sep=",", quiet=TRUE)
mu <- scan("{mu_path}", sep=",", quiet=TRUE)
wt <- scan("{wt_path}", sep=",", quiet=TRUE)
fam <- {family_r}
dr <- fam$dev.resids(y, mu, wt)
cat(sprintf('[%s]', paste(format(dr, digits=17), collapse=",")),
    file="{out_path}")
"""
        script_path = os.path.join(tmpdir, "compute.R")
        with open(script_path, "w") as f:
            f.write(script)

        proc = subprocess.run(
            ["Rscript", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            pytest.skip(f"R script failed: {proc.stderr}")

        with open(out_path) as f:
            data = json.loads(f.read())

        return np.array(data, dtype=np.float64)


@pytest.mark.skipif(not _r_available(), reason="R not available")
class TestDevianceResidsVsR:
    """Deviance residuals match R's family$dev.resids() at STRICT tolerance.

    R's dev.resids returns the per-observation UNIT deviance (wt * d_i),
    not the signed residuals. Our deviance_resids returns signed residuals
    whose squares equal wt * d_i. So we compare squares.
    """

    def test_gaussian_dev_resids(self) -> None:
        rng = np.random.default_rng(SEED)
        y = rng.normal(2.0, 1.0, 50)
        mu = rng.normal(2.0, 0.5, 50)
        wt = np.ones(50)

        r_dr = _compute_r_dev_resids("gaussian()", y, mu, wt)
        fam = Gaussian()
        py_dr = fam.deviance_resids(y, mu, wt)

        # R returns unit deviance components (wt * (y-mu)^2)
        # Our deviance_resids returns signed sqrt; compare squares
        np.testing.assert_allclose(
            py_dr**2,
            r_dr,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Gaussian deviance residuals^2 vs R dev.resids",
        )

    def test_binomial_dev_resids(self) -> None:
        rng = np.random.default_rng(SEED)
        n = 50
        mu = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
        y = rng.binomial(1, mu).astype(float)
        wt = np.ones(n)

        r_dr = _compute_r_dev_resids("binomial()", y, mu, wt)
        fam = Binomial()
        py_dr = fam.deviance_resids(y, mu, wt)

        np.testing.assert_allclose(
            py_dr**2,
            r_dr,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Binomial deviance residuals^2 vs R dev.resids",
        )

    def test_poisson_dev_resids(self) -> None:
        rng = np.random.default_rng(SEED)
        n = 50
        mu = rng.uniform(0.5, 5.0, n)
        y = rng.poisson(mu).astype(float)
        wt = np.ones(n)

        r_dr = _compute_r_dev_resids("poisson()", y, mu, wt)
        fam = Poisson()
        py_dr = fam.deviance_resids(y, mu, wt)

        np.testing.assert_allclose(
            py_dr**2,
            r_dr,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Poisson deviance residuals^2 vs R dev.resids",
        )

    def test_gamma_dev_resids(self) -> None:
        rng = np.random.default_rng(SEED)
        n = 50
        mu = rng.uniform(0.5, 5.0, n)
        y = rng.gamma(5.0, scale=mu / 5.0)
        wt = np.ones(n)

        r_dr = _compute_r_dev_resids("Gamma()", y, mu, wt)
        fam = Gamma()
        py_dr = fam.deviance_resids(y, mu, wt)

        np.testing.assert_allclose(
            py_dr**2,
            r_dr,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Gamma deviance residuals^2 vs R dev.resids",
        )


class TestDevianceResidsSelfConsistency:
    """Self-consistency checks for deviance residuals (no R needed)."""

    def test_gaussian_dev_resids_formula(self) -> None:
        """Gaussian: dev_resid^2 == wt * (y - mu)^2."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        wt = np.array([1.0, 2.0, 1.0, 0.5, 1.0])
        fam = Gaussian()
        dr = fam.deviance_resids(y, mu, wt)
        expected = wt * (y - mu) ** 2
        np.testing.assert_allclose(
            dr**2,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_binomial_dev_resids_formula(self) -> None:
        """Binomial: check against explicit formula for interior y."""
        y = np.array([0.3, 0.5, 0.7])
        mu = np.array([0.2, 0.6, 0.8])
        wt = np.ones(3)
        fam = Binomial()
        dr = fam.deviance_resids(y, mu, wt)
        expected = 2.0 * (y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu)))
        np.testing.assert_allclose(
            dr**2,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_poisson_dev_resids_formula(self) -> None:
        """Poisson: check against explicit formula for y > 0."""
        y = np.array([1.0, 3.0, 5.0, 10.0])
        mu = np.array([1.5, 2.5, 5.5, 8.0])
        wt = np.ones(4)
        fam = Poisson()
        dr = fam.deviance_resids(y, mu, wt)
        expected = 2.0 * (y * np.log(y / mu) - (y - mu))
        np.testing.assert_allclose(
            dr**2,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_gamma_dev_resids_formula(self) -> None:
        """Gamma: check against explicit formula."""
        y = np.array([0.5, 1.0, 2.0, 5.0])
        mu = np.array([0.8, 1.2, 1.5, 4.0])
        wt = np.ones(4)
        fam = Gamma()
        dr = fam.deviance_resids(y, mu, wt)
        expected = 2.0 * (-np.log(y / mu) + (y - mu) / mu)
        np.testing.assert_allclose(
            dr**2,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_total_deviance(self) -> None:
        """dev_resids() returns sum of deviance_resids()^2."""
        rng = np.random.default_rng(SEED)
        y = rng.normal(0, 1, 20)
        mu = rng.normal(0, 0.5, 20)
        wt = np.ones(20)
        fam = Gaussian()
        total = fam.dev_resids(y, mu, wt)
        dr = fam.deviance_resids(y, mu, wt)
        np.testing.assert_allclose(
            total,
            np.sum(dr**2),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_deviance_non_negative(self) -> None:
        """Total deviance must be non-negative for all families."""
        rng = np.random.default_rng(SEED)
        families_and_data: list[tuple[ExponentialFamily, np.ndarray, np.ndarray]] = [
            (Gaussian(), rng.normal(0, 1, 50), rng.normal(0, 0.5, 50)),
            (
                Binomial(),
                rng.binomial(1, 0.5, 50).astype(float),
                np.clip(rng.uniform(0.1, 0.9, 50), 0.01, 0.99),
            ),
            (
                Poisson(),
                rng.poisson(3.0, 50).astype(float),
                rng.uniform(1.0, 5.0, 50),
            ),
            (
                Gamma(),
                rng.gamma(5.0, 1.0, 50),
                rng.uniform(0.5, 5.0, 50),
            ),
        ]
        for fam, y_data, mu_data in families_and_data:
            wt = np.ones(50)
            total = fam.dev_resids(y_data, mu_data, wt)
            assert total >= 0, f"{fam.family_name} deviance is negative: {total}"


# ---------------------------------------------------------------------------
# Test 3: Working weights
# ---------------------------------------------------------------------------


class TestWorkingWeights:
    """Working weights W = wt / (V(mu) * g'(mu)^2)."""

    def test_gaussian_identity_weights(self) -> None:
        """Gaussian + identity: W = wt / (1 * 1^2) = wt."""
        mu = np.array([0.5, 1.0, 2.0])
        wt = np.array([1.0, 2.0, 0.5])
        fam = Gaussian()
        w = fam.working_weights(mu, wt)
        np.testing.assert_allclose(
            w,
            wt,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_binomial_logit_weights(self) -> None:
        """Binomial + logit: W = wt * mu*(1-mu).

        V(mu) = mu(1-mu), g'(mu) = 1/(mu(1-mu)).
        So V(mu) * g'(mu)^2 = 1/(mu(1-mu)).
        W = wt * mu * (1-mu).
        """
        mu = np.array([0.2, 0.5, 0.8])
        wt = np.ones(3)
        fam = Binomial()
        w = fam.working_weights(mu, wt)
        expected = wt * mu * (1.0 - mu)
        np.testing.assert_allclose(
            w,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_poisson_log_weights(self) -> None:
        """Poisson + log: W = wt * mu.

        V(mu) = mu, g'(mu) = 1/mu.
        So V(mu) * g'(mu)^2 = mu * (1/mu)^2 = 1/mu.
        W = wt * mu.
        """
        mu = np.array([0.5, 1.0, 5.0])
        wt = np.ones(3)
        fam = Poisson()
        w = fam.working_weights(mu, wt)
        expected = wt * mu
        np.testing.assert_allclose(
            w,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_gamma_inverse_weights(self) -> None:
        """Gamma + inverse: W = wt * mu^4 / mu^2 = wt * mu^2.

        V(mu) = mu^2, g'(mu) = -1/mu^2.
        So V(mu) * g'(mu)^2 = mu^2 * (1/mu^4) = 1/mu^2.
        W = wt * mu^2.
        """
        mu = np.array([0.5, 1.0, 3.0])
        wt = np.ones(3)
        fam = Gamma()
        w = fam.working_weights(mu, wt)
        expected = wt * mu**2
        np.testing.assert_allclose(
            w,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_nb_log_weights(self) -> None:
        """NB + log: W = wt * mu * theta / (mu + theta).

        V(mu) = mu + mu^2/theta, g'(mu) = 1/mu.
        V(mu) * g'(mu)^2 = (mu + mu^2/theta) / mu^2 = (1/mu + 1/theta).
        W = wt / (1/mu + 1/theta) = wt * mu * theta / (mu + theta).
        """
        mu = np.array([0.5, 1.0, 2.0, 5.0])
        wt = np.ones_like(mu)
        fam = NegativeBinomial(theta=2)
        w = fam.working_weights(mu, wt)
        theta = 2.0
        expected = wt * mu * theta / (mu + theta)
        np.testing.assert_allclose(
            w,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_working_weights_generic(self) -> None:
        """Generic check: W = wt / (V(mu) * g'(mu)^2) for all families."""
        families: list[tuple[ExponentialFamily, np.ndarray]] = [
            (Gaussian(), np.array([0.5, 1.0, 2.0])),
            (Binomial(), np.array([0.2, 0.5, 0.8])),
            (Poisson(), np.array([0.5, 1.0, 5.0])),
            (Gamma(), np.array([0.5, 1.0, 3.0])),
            (NegativeBinomial(theta=2), np.array([0.5, 1.0, 3.0])),
        ]
        wt = np.ones(3)
        for fam, mu in families:
            w = fam.working_weights(mu, wt)
            v = fam.variance(mu)
            g_prime = fam.link.derivative(mu)
            expected = wt / (v * g_prime**2)
            np.testing.assert_allclose(
                w,
                expected,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"Working weights mismatch for {fam.family_name}",
            )


# ---------------------------------------------------------------------------
# Test 4: Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """family.initialize(y, wt) produces valid starting mu."""

    def test_gaussian_initialize(self) -> None:
        """Gaussian: mu = y."""
        y = np.array([-1.0, 0.0, 1.0, 2.5])
        wt = np.ones_like(y)
        fam = Gaussian()
        mu = fam.initialize(y, wt)
        np.testing.assert_allclose(mu, y, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_binomial_initialize(self) -> None:
        """Binomial: mu = (y + 0.5) / 2."""
        y = np.array([0.0, 1.0, 0.0, 1.0])
        wt = np.ones_like(y)
        fam = Binomial()
        mu = fam.initialize(y, wt)
        expected = (y + 0.5) / 2.0
        np.testing.assert_allclose(mu, expected, rtol=STRICT.rtol, atol=STRICT.atol)
        # All initialized mu must be valid
        assert np.all(fam.valid_mu(mu))

    def test_poisson_initialize(self) -> None:
        """Poisson: mu = y + 0.1 where y == 0, else mu = y."""
        y = np.array([0.0, 1.0, 0.0, 5.0, 0.0])
        wt = np.ones_like(y)
        fam = Poisson()
        mu = fam.initialize(y, wt)
        expected = np.where(y == 0, 0.1, y)
        np.testing.assert_allclose(mu, expected, rtol=STRICT.rtol, atol=STRICT.atol)
        # All initialized mu must be valid
        assert np.all(fam.valid_mu(mu))

    def test_gamma_initialize(self) -> None:
        """Gamma: mu = max(y, eps) for strictly positive y."""
        y = np.array([0.01, 0.5, 1.0, 5.0])
        wt = np.ones_like(y)
        fam = Gamma()
        mu = fam.initialize(y, wt)
        assert np.all(mu > 0), "Gamma initialize must produce positive mu"
        np.testing.assert_allclose(
            mu,
            y,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        assert np.all(fam.valid_mu(mu))

    def test_nb_initialize(self) -> None:
        """NB: mu = y + (y == 0) / 6."""
        y = np.array([0.0, 1.0, 5.0, 0.0, 10.0])
        wt = np.ones_like(y)
        fam = NegativeBinomial()
        mu = fam.initialize(y, wt)
        expected = np.where(y == 0, 1.0 / 6.0, y)
        np.testing.assert_allclose(
            mu,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        assert np.all(fam.valid_mu(mu))

    def test_all_families_produce_valid_mu(self) -> None:
        """For each family, initialize produces mu that passes valid_mu."""
        rng = np.random.default_rng(SEED)

        # Gaussian
        y_gauss = rng.normal(0, 1, 100)
        fam_gauss = Gaussian()
        mu_gauss = fam_gauss.initialize(y_gauss, np.ones(100))
        assert np.all(fam_gauss.valid_mu(mu_gauss))

        # Binomial
        y_binom = rng.binomial(1, 0.5, 100).astype(float)
        fam_binom = Binomial()
        mu_binom = fam_binom.initialize(y_binom, np.ones(100))
        assert np.all(fam_binom.valid_mu(mu_binom))

        # Poisson
        y_pois = rng.poisson(3.0, 100).astype(float)
        fam_pois = Poisson()
        mu_pois = fam_pois.initialize(y_pois, np.ones(100))
        assert np.all(fam_pois.valid_mu(mu_pois))

        # Gamma
        y_gam = rng.gamma(5.0, 1.0, 100)
        fam_gam = Gamma()
        mu_gam = fam_gam.initialize(y_gam, np.ones(100))
        assert np.all(fam_gam.valid_mu(mu_gam))


# ---------------------------------------------------------------------------
# Test 5: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case handling: boundary y values and extreme mu."""

    def test_binomial_y_zero(self) -> None:
        """Binomial with y=0: deviance residuals finite and non-negative."""
        y = np.array([0.0, 0.0, 0.0])
        mu = np.array([0.3, 0.5, 0.9])
        wt = np.ones(3)
        fam = Binomial()
        dr = fam.deviance_resids(y, mu, wt)
        assert np.all(np.isfinite(dr)), "Binomial dev resids not finite for y=0"
        assert np.all(dr**2 >= 0), "Binomial dev resids^2 negative for y=0"

    def test_binomial_y_one(self) -> None:
        """Binomial with y=1: deviance residuals finite and non-negative."""
        y = np.array([1.0, 1.0, 1.0])
        mu = np.array([0.1, 0.5, 0.9])
        wt = np.ones(3)
        fam = Binomial()
        dr = fam.deviance_resids(y, mu, wt)
        assert np.all(np.isfinite(dr)), "Binomial dev resids not finite for y=1"
        assert np.all(dr**2 >= 0), "Binomial dev resids^2 negative for y=1"

    def test_binomial_y_equals_mu(self) -> None:
        """Binomial with y=mu: deviance residuals should be zero."""
        mu = np.array([0.2, 0.5, 0.8])
        y = mu.copy()
        wt = np.ones(3)
        fam = Binomial()
        dr = fam.deviance_resids(y, mu, wt)
        np.testing.assert_allclose(
            dr**2,
            np.zeros(3),
            atol=STRICT.atol,
        )

    def test_poisson_y_zero(self) -> None:
        """Poisson with y=0: deviance residuals finite and correct."""
        y = np.array([0.0, 0.0, 0.0])
        mu = np.array([0.5, 1.0, 5.0])
        wt = np.ones(3)
        fam = Poisson()
        dr = fam.deviance_resids(y, mu, wt)
        assert np.all(np.isfinite(dr)), "Poisson dev resids not finite for y=0"
        # When y=0: unit deviance = 2*(0 - (0-mu)) = 2*mu
        expected = 2.0 * mu
        np.testing.assert_allclose(
            dr**2,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_poisson_y_equals_mu(self) -> None:
        """Poisson with y=mu: deviance residuals should be zero."""
        mu = np.array([1.0, 3.0, 10.0])
        y = mu.copy()
        wt = np.ones(3)
        fam = Poisson()
        dr = fam.deviance_resids(y, mu, wt)
        np.testing.assert_allclose(
            dr**2,
            np.zeros(3),
            atol=STRICT.atol,
        )

    def test_gamma_small_mu(self) -> None:
        """Gamma with small mu: results should be finite."""
        y = np.array([0.001, 0.01, 0.1])
        mu = np.array([0.001, 0.01, 0.1])
        wt = np.ones(3)
        fam = Gamma()
        dr = fam.deviance_resids(y, mu, wt)
        assert np.all(np.isfinite(dr)), "Gamma dev resids not finite for small mu"
        v = fam.variance(mu)
        assert np.all(np.isfinite(v)), "Gamma variance not finite for small mu"
        assert np.all(v > 0), "Gamma variance not positive for small mu"

    def test_gamma_y_equals_mu(self) -> None:
        """Gamma with y=mu: deviance residuals should be zero."""
        mu = np.array([0.5, 1.0, 5.0])
        y = mu.copy()
        wt = np.ones(3)
        fam = Gamma()
        dr = fam.deviance_resids(y, mu, wt)
        np.testing.assert_allclose(
            dr**2,
            np.zeros(3),
            atol=STRICT.atol,
        )

    def test_binomial_extreme_mu(self) -> None:
        """Binomial with mu near 0 and 1: variance should be near zero but finite."""
        mu = np.array([1e-10, 1 - 1e-10])
        fam = Binomial()
        v = fam.variance(mu)
        assert np.all(np.isfinite(v)), "Binomial variance not finite at extreme mu"
        assert np.all(v >= 0), "Binomial variance negative at extreme mu"

    def test_nb_y_zero(self) -> None:
        """NB: y=0 produces finite deviance residuals."""
        y = np.array([0.0, 0.0, 0.0])
        mu = np.array([1.0, 5.0, 0.01])
        wt = np.ones(3)
        fam = NegativeBinomial(theta=2)
        dr = fam.deviance_resids(y, mu, wt)
        assert np.all(np.isfinite(dr)), f"NaN/Inf in NB y=0 dev resids: {dr}"

    def test_nb_mu_near_zero(self) -> None:
        """NB: mu near zero produces finite deviance residuals."""
        y = np.array([1.0, 5.0])
        mu = np.array([1e-8, 1e-10])
        wt = np.ones(2)
        fam = NegativeBinomial(theta=2)
        dr = fam.deviance_resids(y, mu, wt)
        assert np.all(np.isfinite(dr)), f"NaN/Inf in NB near-zero mu: {dr}"

    def test_nb_poisson_limit_variance(self) -> None:
        """NB: as theta -> inf, V(mu) -> mu (Poisson)."""
        mu = np.array([0.5, 1.0, 5.0, 10.0])
        fam = NegativeBinomial(theta=1e8)
        v = fam.variance(mu)
        np.testing.assert_allclose(
            v,
            mu,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="NB variance should approach Poisson at large theta",
        )

    def test_nb_poisson_limit_deviance(self) -> None:
        """NB: as theta -> inf, deviance -> Poisson deviance."""
        y = np.array([1.0, 3.0, 5.0, 10.0])
        mu = np.array([1.5, 2.5, 5.5, 8.0])
        wt = np.ones(4)
        nb_fam = NegativeBinomial(theta=1e8)
        poi_fam = Poisson()
        nb_dev = nb_fam.dev_resids(y, mu, wt)
        poi_dev = poi_fam.dev_resids(y, mu, wt)
        np.testing.assert_allclose(
            float(nb_dev),
            float(poi_dev),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="NB deviance should approach Poisson at large theta",
        )

    def test_nb_negative_y_raises(self) -> None:
        """NB: negative y values raise ValueError (R: efam.r line 278)."""
        y = np.array([1.0, -1.0, 3.0])
        wt = np.ones(3)
        fam = NegativeBinomial()
        with pytest.raises(ValueError, match=r"requires response values in"):
            fam.initialize(y, wt)

    def test_poisson_negative_y_raises(self) -> None:
        """Poisson: negative y raises ValueError (R: 'negative values not allowed')."""
        y = np.array([1.0, -1.0, 3.0])
        wt = np.ones(3)
        fam = Poisson()
        with pytest.raises(ValueError, match=r"requires response values in"):
            fam.initialize(y, wt)

    def test_gamma_non_positive_y_raises(self) -> None:
        """Gamma: y=0 raises ValueError (R: 'non-positive values not allowed')."""
        y = np.array([1.0, 0.0, 3.0])
        wt = np.ones(3)
        fam = Gamma()
        with pytest.raises(ValueError, match=r"requires response values in"):
            fam.initialize(y, wt)

    def test_gamma_negative_y_raises(self) -> None:
        """Gamma: y<0 raises ValueError."""
        y = np.array([1.0, -1.0, 3.0])
        wt = np.ones(3)
        fam = Gamma()
        with pytest.raises(ValueError, match=r"requires response values in"):
            fam.initialize(y, wt)

    def test_binomial_out_of_range_raises(self) -> None:
        """Binomial: y>1 raises ValueError (R: 'y values must be 0 <= y <= 1')."""
        y = np.array([0.0, 1.5, 0.5])
        wt = np.ones(3)
        fam = Binomial()
        with pytest.raises(ValueError, match=r"requires response values in"):
            fam.initialize(y, wt)

    def test_binomial_negative_y_raises(self) -> None:
        """Binomial: y<0 raises ValueError."""
        y = np.array([-0.1, 0.5, 1.0])
        wt = np.ones(3)
        fam = Binomial()
        with pytest.raises(ValueError, match=r"requires response values in"):
            fam.initialize(y, wt)

    def test_gaussian_accepts_any_real(self) -> None:
        """Gaussian: any real y is valid (no validation error)."""
        y = np.array([-100.0, 0.0, 100.0])
        wt = np.ones(3)
        fam = Gaussian()
        mu = fam.initialize(y, wt)  # should not raise
        assert np.all(np.isfinite(mu))


# ---------------------------------------------------------------------------
# Test 6: Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """get_family() returns correct family instances."""

    @pytest.mark.parametrize(
        ("name", "expected_cls"),
        [
            ("gaussian", Gaussian),
            ("binomial", Binomial),
            ("poisson", Poisson),
            ("gamma", Gamma),
            ("nb", NegativeBinomial),
        ],
    )
    def test_get_family_by_name(
        self, name: str, expected_cls: type[ExponentialFamily]
    ) -> None:
        fam = get_family(name)
        assert isinstance(fam, expected_cls)

    def test_get_family_case_insensitive(self) -> None:
        """get_family should be case-insensitive."""
        fam = get_family("Gaussian")
        assert isinstance(fam, Gaussian)
        fam = get_family("POISSON")
        assert isinstance(fam, Poisson)
        fam = get_family("Gamma")
        assert isinstance(fam, Gamma)

    def test_get_family_passthrough(self) -> None:
        """If already an ExponentialFamily instance, return it as-is."""
        fam_in = Gaussian()
        fam_out = get_family(fam_in)
        assert fam_out is fam_in

    def test_get_family_unknown_raises(self) -> None:
        """Unknown family name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown family"):
            get_family("nonexistent_family")

    def test_get_family_wrong_type_raises(self) -> None:
        """Non-string, non-family argument raises TypeError."""
        with pytest.raises(TypeError):
            get_family(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 7: Family properties and link integration
# ---------------------------------------------------------------------------


class TestFamilyProperties:
    """Test family_name, default_link, scale_known, and repr."""

    def test_gaussian_properties(self) -> None:
        fam = Gaussian()
        assert fam.family_name == "gaussian"
        assert not fam.scale_known
        from jaxgam.links import IdentityLink

        assert isinstance(fam.link, IdentityLink)

    def test_binomial_properties(self) -> None:
        fam = Binomial()
        assert fam.family_name == "binomial"
        assert fam.scale_known
        from jaxgam.links import LogitLink

        assert isinstance(fam.link, LogitLink)

    def test_poisson_properties(self) -> None:
        fam = Poisson()
        assert fam.family_name == "poisson"
        assert fam.scale_known
        from jaxgam.links import LogLink

        assert isinstance(fam.link, LogLink)

    def test_gamma_properties(self) -> None:
        fam = Gamma()
        assert fam.family_name == "Gamma"
        assert not fam.scale_known
        from jaxgam.links import InverseLink

        assert isinstance(fam.link, InverseLink)

    def test_custom_link(self) -> None:
        """Families accept non-default link functions."""
        fam = Poisson(link="identity")
        from jaxgam.links import IdentityLink

        assert isinstance(fam.link, IdentityLink)

    def test_repr(self) -> None:
        fam = Gaussian()
        assert "Gaussian" in repr(fam)
        assert "IdentityLink" in repr(fam)

    def test_valid_mu_and_eta(self) -> None:
        """valid_mu and valid_eta produce boolean arrays."""
        mu = np.array([0.0, 0.5, 1.0, -1.0, np.nan, np.inf])
        eta = np.array([-10.0, 0.0, 10.0, np.nan, np.inf, -np.inf])

        for fam in [Gaussian(), Binomial(), Poisson(), Gamma()]:
            vm = fam.valid_mu(mu)
            ve = fam.valid_eta(eta)
            assert vm.dtype == bool
            assert ve.dtype == bool
            assert vm.shape == mu.shape
            assert ve.shape == eta.shape


# ---------------------------------------------------------------------------
# Test 9: Working response
# ---------------------------------------------------------------------------


class TestWorkingResponse:
    """PIRLS working response: z = eta + (y - mu) * g'(mu)."""

    def test_gaussian_identity_working_response(self) -> None:
        """Gaussian + identity: z = eta + (y - mu) * 1 = y (since eta = mu)."""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([0.9, 2.1, 2.8])
        eta = mu.copy()  # identity link: eta = mu
        fam = Gaussian()
        z = fam.working_response(y, mu, eta)
        expected = eta + (y - mu) * 1.0  # g'(mu) = 1 for identity
        np.testing.assert_allclose(z, expected, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_working_response_generic(self) -> None:
        """Check working response formula z = eta + (y - mu) * g'(mu)."""
        families: list[tuple[ExponentialFamily, np.ndarray]] = [
            (Gaussian(), np.array([0.5, 1.0, 2.0])),
            (Binomial(), np.array([0.2, 0.5, 0.8])),
            (Poisson(), np.array([0.5, 1.0, 5.0])),
            (Gamma(), np.array([0.5, 1.0, 3.0])),
        ]
        rng = np.random.default_rng(SEED)
        for fam, mu in families:
            eta = fam.link.link(mu)
            y = mu + rng.normal(0, 0.01, len(mu))
            if isinstance(fam, Binomial):
                y = np.clip(y, 0.01, 0.99)
            elif isinstance(fam, (Poisson, Gamma)):
                y = np.maximum(y, 0.01)
            z = fam.working_response(y, mu, eta)
            g_prime = fam.link.derivative(mu)
            expected = eta + (y - mu) * g_prime
            np.testing.assert_allclose(
                z,
                expected,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"Working response mismatch for {fam.family_name}",
            )


# ---------------------------------------------------------------------------
# Test 10: JAX compatibility — family PIRLS methods accept JAX arrays
# ---------------------------------------------------------------------------


def _jax_family_test_data(
    fam: ExponentialFamily,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return (y, mu, wt, eta) as JAX arrays for a given family."""
    if isinstance(fam, Gaussian):
        y = jnp.array([0.5, 1.2, 2.0, 3.5, 4.1])
        mu = jnp.array([0.6, 1.0, 2.2, 3.3, 4.0])
    elif isinstance(fam, Binomial):
        y = jnp.array([0.0, 0.3, 0.5, 0.7, 1.0])
        mu = jnp.array([0.15, 0.35, 0.5, 0.65, 0.85])
    elif isinstance(fam, Poisson):
        y = jnp.array([0.0, 1.0, 2.0, 5.0, 10.0])
        mu = jnp.array([0.5, 1.2, 2.5, 4.0, 9.0])
    elif isinstance(fam, Gamma):
        y = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        mu = jnp.array([0.8, 1.2, 1.5, 4.0, 8.0])
    else:
        raise ValueError(f"Unknown family: {fam}")
    wt = jnp.ones_like(y)
    eta = fam.link.link(mu)
    return y, mu, wt, eta


FAMILIES = [Gaussian(), Binomial(), Poisson(), Gamma()]
FAMILY_IDS = ["gaussian", "binomial", "poisson", "gamma"]


class TestFamilyJAXCompat:
    """JAX compatibility: PIRLS-path methods accept JAX arrays."""

    @pytest.mark.parametrize("fam", FAMILIES, ids=FAMILY_IDS)
    def test_variance_jax_matches_numpy(self, fam: ExponentialFamily) -> None:
        _, jax_mu, _, _ = _jax_family_test_data(fam)
        np_mu = np.asarray(jax_mu)

        jax_v = fam.variance(jax_mu)
        np_v = fam.variance(np_mu)
        np.testing.assert_allclose(
            np.asarray(jax_v),
            np_v,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"variance() JAX vs NumPy for {fam.family_name}",
        )

    @pytest.mark.parametrize("fam", FAMILIES, ids=FAMILY_IDS)
    def test_deviance_resids_jax_matches_numpy(self, fam: ExponentialFamily) -> None:
        jax_y, jax_mu, jax_wt, _ = _jax_family_test_data(fam)
        np_y, np_mu, np_wt = (
            np.asarray(jax_y),
            np.asarray(jax_mu),
            np.asarray(jax_wt),
        )

        jax_dr = fam.deviance_resids(jax_y, jax_mu, jax_wt)
        np_dr = fam.deviance_resids(np_y, np_mu, np_wt)
        np.testing.assert_allclose(
            np.asarray(jax_dr),
            np_dr,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"deviance_resids() JAX vs NumPy for {fam.family_name}",
        )

    @pytest.mark.parametrize("fam", FAMILIES, ids=FAMILY_IDS)
    def test_dev_resids_jax_matches_numpy(self, fam: ExponentialFamily) -> None:
        """Total deviance (scalar) matches between backends."""
        jax_y, jax_mu, jax_wt, _ = _jax_family_test_data(fam)
        np_y, np_mu, np_wt = (
            np.asarray(jax_y),
            np.asarray(jax_mu),
            np.asarray(jax_wt),
        )

        jax_total = fam.dev_resids(jax_y, jax_mu, jax_wt)
        np_total = fam.dev_resids(np_y, np_mu, np_wt)
        np.testing.assert_allclose(
            float(jax_total),
            float(np_total),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"dev_resids() JAX vs NumPy for {fam.family_name}",
        )

    @pytest.mark.parametrize("fam", FAMILIES, ids=FAMILY_IDS)
    def test_working_weights_jax_matches_numpy(self, fam: ExponentialFamily) -> None:
        _, jax_mu, jax_wt, _ = _jax_family_test_data(fam)
        np_mu, np_wt = np.asarray(jax_mu), np.asarray(jax_wt)

        jax_w = fam.working_weights(jax_mu, jax_wt)
        np_w = fam.working_weights(np_mu, np_wt)
        np.testing.assert_allclose(
            np.asarray(jax_w),
            np_w,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"working_weights() JAX vs NumPy for {fam.family_name}",
        )

    @pytest.mark.parametrize("fam", FAMILIES, ids=FAMILY_IDS)
    def test_working_response_jax_matches_numpy(self, fam: ExponentialFamily) -> None:
        jax_y, jax_mu, _, jax_eta = _jax_family_test_data(fam)
        np_y = np.asarray(jax_y)
        np_mu = np.asarray(jax_mu)
        np_eta = np.asarray(jax_eta)

        jax_z = fam.working_response(jax_y, jax_mu, jax_eta)
        np_z = fam.working_response(np_y, np_mu, np_eta)
        np.testing.assert_allclose(
            np.asarray(jax_z),
            np_z,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=(f"working_response() JAX vs NumPy for {fam.family_name}"),
        )

    @pytest.mark.parametrize("fam", FAMILIES, ids=FAMILY_IDS)
    def test_pirls_methods_jit_compile(self, fam: ExponentialFamily) -> None:
        """variance, deviance_resids, working_weights, working_response
        all JIT-compile without error."""
        jax_y, jax_mu, jax_wt, jax_eta = _jax_family_test_data(fam)

        jit_var = jax.jit(fam.variance)
        jit_dr = jax.jit(fam.deviance_resids)
        jit_ww = jax.jit(fam.working_weights)
        jit_wr = jax.jit(fam.working_response)

        v = jit_var(jax_mu)
        dr = jit_dr(jax_y, jax_mu, jax_wt)
        ww = jit_ww(jax_mu, jax_wt)
        wr = jit_wr(jax_y, jax_mu, jax_eta)

        assert jnp.all(jnp.isfinite(v)), (
            f"JIT variance() non-finite for {fam.family_name}"
        )
        assert jnp.all(jnp.isfinite(dr)), (
            f"JIT deviance_resids() non-finite for {fam.family_name}"
        )
        assert jnp.all(jnp.isfinite(ww)), (
            f"JIT working_weights() non-finite for {fam.family_name}"
        )
        assert jnp.all(jnp.isfinite(wr)), (
            f"JIT working_response() non-finite for {fam.family_name}"
        )


# ---------------------------------------------------------------------------
# Test 8: ExtendedFamily contract tests
# ---------------------------------------------------------------------------


def _extended_family_test_data(
    fam: ExtendedFamily,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, mu, wt) appropriate for the family's response domain.

    Each extended family needs data in its valid range. Adding a new
    family here is the ONLY change needed to get full contract test
    coverage -- all tests below use this function.

    Returns
    -------
    y : np.ndarray, shape (6,)
    mu : np.ndarray, shape (6,)
    wt : np.ndarray, shape (6,)
    """
    if isinstance(fam, NegativeBinomial):
        # Count data with zeros -- valid for NB
        return (
            np.array([0.0, 1.0, 2.0, 5.0, 10.0, 50.0]),
            np.array([0.5, 1.0, 2.0, 4.0, 8.0, 30.0]),
            np.ones(6),
        )
    # Future families: add elif branches here.
    # elif isinstance(fam, Tweedie):
    #     # Continuous non-negative with exact zeros
    #     return (
    #         np.array([0.0, 0.0, 0.5, 1.2, 3.0, 10.0]),
    #         np.array([0.3, 0.5, 1.0, 2.0, 4.0, 8.0]),
    #         np.ones(6),
    #     )
    # elif isinstance(fam, Beta):
    #     # (0, 1) support
    #     return (
    #         np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.95]),
    #         np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.8]),
    #         np.ones(6),
    #     )
    raise NotImplementedError(
        f"Add test data for {type(fam).__name__} in _extended_family_test_data()"
    )


class TestExtendedFamilyContract:
    """Contract tests every ExtendedFamily must pass.

    Parametrized by family instance. Each family provides its own valid
    test data via ``_extended_family_test_data()``. To add a new
    extended family:
    1. Add it to the ``efamily`` fixture params
    2. Add a branch in ``_extended_family_test_data()``

    All contract tests run automatically -- no new test code needed.
    """

    @pytest.fixture(
        params=[
            NegativeBinomial(),
            NegativeBinomial(theta=2, fixed=True),
            NegativeBinomial(theta=0.5, fixed=True),
            # future: Tweedie(p=1.5), Beta(), ...
        ],
        ids=["nb_estimated", "nb_fixed_2", "nb_fixed_0.5"],
    )
    def efamily(self, request):
        return request.param

    @pytest.fixture
    def test_data(self, efamily):
        """Family-appropriate (y, mu, wt) for the current efamily."""
        y, mu, wt = _extended_family_test_data(efamily)
        # Set _max_y for _lgamma_diff scan (normally done by NewtonOptimizer)
        if hasattr(efamily, "_max_y"):
            efamily._max_y = int(np.max(y))
        return y, mu, wt

    def test_is_extended_and_exponential(self, efamily) -> None:
        assert isinstance(efamily, ExtendedFamily)
        assert isinstance(efamily, ExponentialFamily)

    def test_standard_families_not_extended(self) -> None:
        """Standard families must NOT be ExtendedFamily instances."""
        for fam in [Gaussian(), Poisson(), Binomial(), Gamma()]:
            assert not isinstance(fam, ExtendedFamily), fam.family_name

    def test_n_theta_non_negative(self, efamily) -> None:
        assert efamily.n_theta >= 0

    def test_get_theta_shape(self, efamily) -> None:
        """get_theta returns array of shape (n_theta,) or (1,)."""
        lt = efamily.get_theta()
        assert lt.ndim == 1
        assert lt.shape[0] >= 1

    def test_get_put_roundtrip(self, efamily) -> None:
        original = efamily.get_theta().copy()
        efamily.put_theta(original + 1.0)
        np.testing.assert_allclose(
            efamily.get_theta(),
            original + 1.0,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        efamily.put_theta(original)  # restore

    def test_deviance_fn_returns_callable(self, efamily, test_data) -> None:
        y, _mu, wt = test_data
        dev_fn = efamily.deviance_fn(jnp.array(y), jnp.array(wt))
        assert callable(dev_fn)

    def test_working_weights_fn_returns_callable(self, efamily, test_data) -> None:
        _y, _mu, wt = test_data
        ww_fn = efamily.working_weights_fn(jnp.array(wt))
        assert callable(ww_fn)

    def test_deviance_fn_consistency(self, efamily, test_data) -> None:
        """deviance_fn(eta, log_theta) matches dev_resids at stored theta."""
        y, mu, wt = test_data
        y, mu, wt = jnp.array(y), jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        dev_fn = efamily.deviance_fn(y, wt)
        fn_val = float(dev_fn(eta, log_theta))
        std_val = float(efamily.dev_resids(y, mu, wt))
        np.testing.assert_allclose(
            fn_val,
            std_val,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="deviance_fn inconsistent with dev_resids",
        )

    def test_working_weights_fn_consistency(self, efamily, test_data) -> None:
        """working_weights_fn(eta, log_theta) matches working_weights."""
        _y, mu, wt = test_data
        mu, wt = jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        ww_fn = efamily.working_weights_fn(wt)
        fn_val = ww_fn(eta, log_theta)
        std_val = efamily.working_weights(mu, wt)
        np.testing.assert_allclose(
            fn_val,
            std_val,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="working_weights_fn inconsistent with working_weights",
        )

    def test_saturated_loglik_theta_consistency(self, efamily, test_data) -> None:
        """saturated_loglik_theta matches saturated_loglik at stored theta."""
        y, _mu, wt = test_data
        y, wt = jnp.array(y), jnp.array(wt)
        log_theta = jnp.array(efamily.get_theta())

        ls_std = float(efamily.saturated_loglik(y, wt, 1.0))
        ls_theta = float(efamily.saturated_loglik_theta(y, wt, 1.0, log_theta))
        np.testing.assert_allclose(
            ls_theta,
            ls_std,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="saturated_loglik_theta inconsistent",
        )

    def test_ad_deviance_fn_grad_eta_finite(self, efamily, test_data) -> None:
        y, mu, wt = test_data
        y, mu, wt = jnp.array(y), jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        dev_fn = efamily.deviance_fn(y, wt)
        grad_eta = jax.grad(dev_fn, argnums=0)(eta, log_theta)
        assert jnp.all(jnp.isfinite(grad_eta)), "dD/deta not finite"

    def test_ad_deviance_fn_grad_theta_finite(self, efamily, test_data) -> None:
        y, mu, wt = test_data
        y, mu, wt = jnp.array(y), jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        dev_fn = efamily.deviance_fn(y, wt)
        grad_theta = jax.grad(dev_fn, argnums=1)(eta, log_theta)
        assert jnp.all(jnp.isfinite(grad_theta)), "dD/d(log_theta) not finite"

    def test_ad_mixed_derivative_finite(self, efamily, test_data) -> None:
        """d^2D/(deta d(log_theta)) must be finite -- custom_jvp needs this."""
        y, mu, wt = test_data
        y, mu, wt = jnp.array(y), jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        dev_fn = efamily.deviance_fn(y, wt)
        grad_D_eta = jax.grad(dev_fn, argnums=0)
        _, mixed = jax.jvp(
            lambda lt: grad_D_eta(eta, lt),
            (log_theta,),
            (jnp.ones_like(log_theta),),
        )
        assert jnp.all(jnp.isfinite(mixed)), "mixed d2D/(deta dtheta) not finite"

    def test_ad_working_weights_jvp_finite(self, efamily, test_data) -> None:
        _y, mu, wt = test_data
        mu, wt = jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        ww_fn = efamily.working_weights_fn(wt)
        deta = jnp.ones_like(eta) * 0.01
        dlt = jnp.ones_like(log_theta) * 0.01
        _, dW = jax.jvp(ww_fn, (eta, log_theta), (deta, dlt))
        assert jnp.all(jnp.isfinite(dW)), "working_weights JVP not finite"

    def test_ad_saturated_loglik_theta_grad_finite(self, efamily, test_data) -> None:
        y, _mu, wt = test_data
        y, wt = jnp.array(y), jnp.array(wt)
        log_theta = jnp.array(efamily.get_theta())

        grad_ls = jax.grad(efamily.saturated_loglik_theta, argnums=3)(
            y, wt, 1.0, log_theta
        )
        assert jnp.all(jnp.isfinite(grad_ls)), "d(ls_sat)/d(log_theta) not finite"


# ---------------------------------------------------------------------------
# Test 9: NB-specific tests
# ---------------------------------------------------------------------------


class TestNBSpecific:
    """NB-only tests that don't generalize to other extended families."""

    def test_constructor_default(self) -> None:
        """NegativeBinomial() -> estimate theta, start at 1."""
        fam = NegativeBinomial()
        assert fam.n_theta == 1
        np.testing.assert_allclose(fam.get_theta(), [0.0])
        np.testing.assert_allclose(fam.get_theta(transformed=True), [1.0])

    def test_constructor_fixed(self) -> None:
        """NegativeBinomial(theta=2, fixed=True) -> fixed theta = 2."""
        fam = NegativeBinomial(theta=2, fixed=True)
        assert fam.n_theta == 0
        np.testing.assert_allclose(
            fam.get_theta(transformed=True),
            [2.0],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_constructor_estimated_with_start(self) -> None:
        """NegativeBinomial(theta=3) -> estimate, start at 3."""
        fam = NegativeBinomial(theta=3)
        assert fam.n_theta == 1
        np.testing.assert_allclose(
            fam.get_theta(transformed=True),
            [3.0],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_constructor_rejects_nonpositive(self) -> None:
        """NegativeBinomial(theta<=0) raises ValueError."""
        with pytest.raises(ValueError, match="theta must be positive"):
            NegativeBinomial(theta=0)
        with pytest.raises(ValueError, match="theta must be positive"):
            NegativeBinomial(theta=-3)

    def test_alpha_property(self) -> None:
        fam = NegativeBinomial(theta=2)
        np.testing.assert_allclose(fam.alpha, 0.5, rtol=STRICT.rtol)
        fam2 = NegativeBinomial(theta=10)
        np.testing.assert_allclose(fam2.alpha, 0.1, rtol=STRICT.rtol)

    def test_scale_known(self) -> None:
        assert NegativeBinomial().scale_known is True

    def test_default_link_is_log(self) -> None:
        fam = NegativeBinomial()
        assert isinstance(fam.link, LogLink)

    def test_family_name(self) -> None:
        assert NegativeBinomial().family_name == "nb"

    def test_valid_mu(self) -> None:
        fam = NegativeBinomial()
        mu = np.array([-1.0, 0.0, 0.001, 1.0, 100.0])
        expected = np.array([False, False, True, True, True])
        np.testing.assert_array_equal(fam.valid_mu(mu), expected)

    def test_repr(self) -> None:
        fam = NegativeBinomial(theta=2, fixed=True)
        r = repr(fam)
        assert "NegativeBinomial" in r
        assert "fixed" in r
        assert "LogLink" in r


# ---------------------------------------------------------------------------
# Test 10: ExtendedFamily AD finite-difference validation
# ---------------------------------------------------------------------------


def _central_fd_grad(f, x, eps=1e-5):
    """Central finite-difference gradient of scalar-valued f at array x.

    Parameters
    ----------
    f : callable
        Scalar-valued function of a single JAX array argument.
    x : jax.Array
        Point at which to evaluate the gradient.
    eps : float
        Perturbation size for central differences.

    Returns
    -------
    np.ndarray
        Gradient estimate, same shape as x.
    """
    x_np = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x_np)
    for i in range(x_np.size):
        x_plus = x_np.copy()
        x_minus = x_np.copy()
        x_plus.flat[i] += eps
        x_minus.flat[i] -= eps
        grad.flat[i] = (float(f(jnp.array(x_plus))) - float(f(jnp.array(x_minus)))) / (
            2 * eps
        )
    return grad


class TestExtendedFamilyAD:
    """Finite-difference validation of AD through extended family factories.

    Parametrized by family instance and theta regime. Verifies jax.grad
    output matches central finite differences to MODERATE tolerance.

    PR 1's TestExtendedFamilyContract checks that AD produces *finite*
    values. This class checks that the AD values are *numerically correct*
    by comparing against central finite differences.
    """

    FD_EPS = 1e-5

    @pytest.fixture(
        params=[
            (NegativeBinomial(theta=2, fixed=True), "moderate_theta"),
            (NegativeBinomial(theta=0.01, fixed=True), "high_overdispersion"),
            (NegativeBinomial(theta=10000, fixed=True), "near_poisson"),
            # future: Tweedie(p=1.5), Beta(), ...
        ],
        ids=["nb_theta2", "nb_theta0.01", "nb_theta10000"],
    )
    def efamily_regime(self, request):
        return request.param

    @pytest.fixture
    def efamily(self, efamily_regime):
        return efamily_regime[0]

    @pytest.fixture
    def test_data(self, efamily):
        y, mu, wt = _extended_family_test_data(efamily)
        if hasattr(efamily, "_max_y"):
            efamily._max_y = int(np.max(y))
        return y, mu, wt

    # ------------------------------------------------------------------
    # saturated_loglik_theta
    # ------------------------------------------------------------------

    def test_saturated_loglik_theta_grad(self, efamily, test_data) -> None:
        """d(ls_sat)/d(log_theta) via AD matches central FD."""
        y, _mu, wt = test_data
        y, wt = jnp.array(y), jnp.array(wt)
        log_theta = jnp.array(efamily.get_theta())

        ad_grad = np.asarray(
            jax.grad(efamily.saturated_loglik_theta, argnums=3)(y, wt, 1.0, log_theta)
        )
        # Larger eps to reduce cancellation error when |ls_sat| >> |gradient|
        # (near-Poisson regime: ls_sat ~ 82000, gradient ~ 0.003).
        fd_grad = _central_fd_grad(
            lambda lt: efamily.saturated_loglik_theta(y, wt, 1.0, lt),
            log_theta,
            eps=1e-4,
        )
        np.testing.assert_allclose(
            ad_grad,
            fd_grad,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="saturated_loglik_theta: AD grad vs FD",
        )

    def test_saturated_loglik_theta_hessian(self, efamily, test_data) -> None:
        """d^2(ls_sat)/d(log_theta)^2 via AD matches FD of gradient."""
        y, _mu, wt = test_data
        y, wt = jnp.array(y), jnp.array(wt)
        log_theta = jnp.array(efamily.get_theta())

        def ls_fn(lt):
            return efamily.saturated_loglik_theta(y, wt, 1.0, lt)

        ad_hess = np.asarray(jax.hessian(ls_fn)(log_theta))

        # FD of gradient: Jacobian of grad = Hessian.
        grad_fn = jax.grad(ls_fn)
        n = log_theta.shape[0]
        fd_hess = np.zeros((n, n))
        eps = self.FD_EPS
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = eps
            g_plus = np.asarray(grad_fn(log_theta + jnp.array(e_i)))
            g_minus = np.asarray(grad_fn(log_theta - jnp.array(e_i)))
            fd_hess[:, i] = (g_plus - g_minus) / (2 * eps)

        # LOOSE tolerance: at large theta the gradient involves
        # catastrophic cancellation (large log/digamma terms summing
        # to a small residual), limiting FD precision for second
        # derivatives regardless of step size.
        np.testing.assert_allclose(
            ad_hess,
            fd_hess,
            rtol=LOOSE.rtol,
            atol=LOOSE.atol,
            err_msg="saturated_loglik_theta: AD hessian vs FD",
        )

    # ------------------------------------------------------------------
    # deviance_fn
    # ------------------------------------------------------------------

    def test_deviance_fn_grad_eta(self, efamily, test_data) -> None:
        """dD/deta via AD matches central FD."""
        y, mu, wt = test_data
        y, mu, wt = jnp.array(y), jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        dev_fn = efamily.deviance_fn(y, wt)
        ad_grad = np.asarray(jax.grad(dev_fn, argnums=0)(eta, log_theta))
        fd_grad = _central_fd_grad(lambda e: dev_fn(e, log_theta), eta, eps=self.FD_EPS)
        np.testing.assert_allclose(
            ad_grad,
            fd_grad,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="deviance_fn: dD/deta AD vs FD",
        )

    def test_deviance_fn_grad_theta(self, efamily, test_data) -> None:
        """dD/d(log_theta) via AD matches central FD."""
        y, mu, wt = test_data
        y, mu, wt = jnp.array(y), jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        dev_fn = efamily.deviance_fn(y, wt)
        ad_grad = np.asarray(jax.grad(dev_fn, argnums=1)(eta, log_theta))
        fd_grad = _central_fd_grad(
            lambda lt: dev_fn(eta, lt), log_theta, eps=self.FD_EPS
        )
        np.testing.assert_allclose(
            ad_grad,
            fd_grad,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="deviance_fn: dD/d(log_theta) AD vs FD",
        )

    def test_deviance_fn_mixed_derivative(self, efamily, test_data) -> None:
        """d^2D/(deta d(log_theta)) via JVP matches FD of gradient."""
        y, mu, wt = test_data
        y, mu, wt = jnp.array(y), jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        dev_fn = efamily.deviance_fn(y, wt)
        grad_D_eta = jax.grad(dev_fn, argnums=0)

        # AD: JVP of dD/deta w.r.t. log_theta
        _, mixed_ad = jax.jvp(
            lambda lt: grad_D_eta(eta, lt),
            (log_theta,),
            (jnp.ones_like(log_theta),),
        )

        # FD: perturb log_theta, recompute dD/deta
        eps = self.FD_EPS
        g_plus = grad_D_eta(eta, log_theta + eps)
        g_minus = grad_D_eta(eta, log_theta - eps)
        mixed_fd = (g_plus - g_minus) / (2 * eps)

        np.testing.assert_allclose(
            mixed_ad,
            mixed_fd,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="deviance_fn: mixed d2D/(deta dtheta) AD vs FD",
        )

    @pytest.mark.parametrize(
        ("y_val", "mu_val"),
        [
            (0.0, 0.5),  # y=0 boundary
            (1000.0, 500.0),  # large y
            (1.0, 0.001),  # mu near zero
            (1.0, 100.0),  # mu large
        ],
        ids=["y_zero", "y_large", "mu_small", "mu_large"],
    )
    def test_deviance_fn_grad_extreme(self, efamily, y_val, mu_val) -> None:
        """AD gradients at extreme (y, mu) are finite and match FD."""
        y = jnp.array([y_val])
        mu = jnp.array([mu_val])
        wt = jnp.ones(1)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        dev_fn = efamily.deviance_fn(y, wt)

        # Check finiteness
        ad_eta = np.asarray(jax.grad(dev_fn, argnums=0)(eta, log_theta))
        ad_theta = np.asarray(jax.grad(dev_fn, argnums=1)(eta, log_theta))
        assert np.all(np.isfinite(ad_eta)), (
            f"dD/deta not finite at y={y_val}, mu={mu_val}"
        )
        assert np.all(np.isfinite(ad_theta)), (
            f"dD/d(log_theta) not finite at y={y_val}, mu={mu_val}"
        )

        # Check accuracy vs FD
        fd_eta = _central_fd_grad(lambda e: dev_fn(e, log_theta), eta, eps=self.FD_EPS)
        fd_theta = _central_fd_grad(
            lambda lt: dev_fn(eta, lt), log_theta, eps=self.FD_EPS
        )
        np.testing.assert_allclose(
            ad_eta,
            fd_eta,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=f"dD/deta vs FD at y={y_val}, mu={mu_val}",
        )
        np.testing.assert_allclose(
            ad_theta,
            fd_theta,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=f"dD/d(log_theta) vs FD at y={y_val}, mu={mu_val}",
        )

    # ------------------------------------------------------------------
    # working_weights_fn
    # ------------------------------------------------------------------

    def test_working_weights_fn_jvp_eta(self, efamily, test_data) -> None:
        """JVP of W w.r.t. eta matches central FD."""
        _y, mu, wt = test_data
        mu, wt = jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        ww_fn = efamily.working_weights_fn(wt)
        deta = jnp.ones_like(eta) * 0.01

        # AD JVP (eta only)
        _, jvp_ad = jax.jvp(
            ww_fn,
            (eta, log_theta),
            (deta, jnp.zeros_like(log_theta)),
        )

        # FD JVP
        eps = self.FD_EPS
        jvp_fd = (
            ww_fn(eta + eps * deta, log_theta) - ww_fn(eta - eps * deta, log_theta)
        ) / (2 * eps)

        np.testing.assert_allclose(
            jvp_ad,
            jvp_fd,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="working_weights_fn: JVP w.r.t. eta AD vs FD",
        )

    def test_working_weights_fn_jvp_theta(self, efamily, test_data) -> None:
        """JVP of W w.r.t. log_theta matches central FD."""
        _y, mu, wt = test_data
        mu, wt = jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        ww_fn = efamily.working_weights_fn(wt)
        dlt = jnp.ones_like(log_theta) * 0.01

        # AD JVP (theta only)
        _, jvp_ad = jax.jvp(
            ww_fn,
            (eta, log_theta),
            (jnp.zeros_like(eta), dlt),
        )

        # FD JVP
        eps = self.FD_EPS
        jvp_fd = (
            ww_fn(eta, log_theta + eps * dlt) - ww_fn(eta, log_theta - eps * dlt)
        ) / (2 * eps)

        np.testing.assert_allclose(
            jvp_ad,
            jvp_fd,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="working_weights_fn: JVP w.r.t. theta AD vs FD",
        )

    def test_working_weights_fn_jvp_joint_linearity(self, efamily, test_data) -> None:
        """Joint JVP equals sum of individual JVPs (derivative linearity)."""
        _y, mu, wt = test_data
        mu, wt = jnp.array(mu), jnp.array(wt)
        eta = efamily.link.link(mu)
        log_theta = jnp.array(efamily.get_theta())

        ww_fn = efamily.working_weights_fn(wt)
        deta = jnp.ones_like(eta) * 0.01
        dlt = jnp.ones_like(log_theta) * 0.01

        # Joint
        _, jvp_joint = jax.jvp(ww_fn, (eta, log_theta), (deta, dlt))

        # Individual
        _, jvp_eta = jax.jvp(ww_fn, (eta, log_theta), (deta, jnp.zeros_like(log_theta)))
        _, jvp_theta = jax.jvp(ww_fn, (eta, log_theta), (jnp.zeros_like(eta), dlt))

        np.testing.assert_allclose(
            jvp_joint,
            jvp_eta + jvp_theta,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Joint JVP != sum of individual JVPs",
        )
