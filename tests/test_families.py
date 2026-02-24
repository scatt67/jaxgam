"""Tests for pymgcv.families.

Coverage:
1. TestVariance — V(mu) correct for each family at STRICT tolerance
2. TestDevianceResids — deviance residuals match R at STRICT on synthetic data
3. TestWorkingWeights — 1/(V(mu)*g'(mu)^2) computed correctly
4. TestInitialization — family.initialize(y, wt) produces valid starting mu
5. TestEdgeCases — Binomial y=0/y=1, Poisson y=0, Gamma small mu
6. TestRegistry — get_family("gaussian") returns Gaussian, etc.
7. TestNoJaxImports — importing pymgcv.families doesn't trigger jax import
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

from pymgcv.families import (
    Binomial,
    ExponentialFamily,
    Gamma,
    Gaussian,
    Poisson,
    get_family,
)
from tests.tolerances import STRICT

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 123


def _r_available() -> bool:
    """Check if Rscript is available."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", "cat('ok')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


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

    def test_working_weights_generic(self) -> None:
        """Generic check: W = wt / (V(mu) * g'(mu)^2) for all families."""
        families: list[tuple[ExponentialFamily, np.ndarray]] = [
            (Gaussian(), np.array([0.5, 1.0, 2.0])),
            (Binomial(), np.array([0.2, 0.5, 0.8])),
            (Poisson(), np.array([0.5, 1.0, 5.0])),
            (Gamma(), np.array([0.5, 1.0, 3.0])),
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
        """Gamma: mu = max(y, eps)."""
        y = np.array([0.0, 0.5, 1.0, 5.0])
        wt = np.ones_like(y)
        fam = Gamma()
        mu = fam.initialize(y, wt)
        assert np.all(mu > 0), "Gamma initialize must produce positive mu"
        # For y > 0, mu should equal y
        np.testing.assert_allclose(
            mu[y > 0], y[y > 0], rtol=STRICT.rtol, atol=STRICT.atol
        )
        # All initialized mu must be valid
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
        from pymgcv.links import IdentityLink

        assert isinstance(fam.link, IdentityLink)

    def test_binomial_properties(self) -> None:
        fam = Binomial()
        assert fam.family_name == "binomial"
        assert fam.scale_known
        from pymgcv.links import LogitLink

        assert isinstance(fam.link, LogitLink)

    def test_poisson_properties(self) -> None:
        fam = Poisson()
        assert fam.family_name == "poisson"
        assert fam.scale_known
        from pymgcv.links import LogLink

        assert isinstance(fam.link, LogLink)

    def test_gamma_properties(self) -> None:
        fam = Gamma()
        assert fam.family_name == "Gamma"
        assert not fam.scale_known
        from pymgcv.links import InverseLink

        assert isinstance(fam.link, InverseLink)

    def test_custom_link(self) -> None:
        """Families accept non-default link functions."""
        fam = Poisson(link="identity")
        from pymgcv.links import IdentityLink

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
