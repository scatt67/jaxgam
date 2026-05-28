"""Tests for jaxgam.families.

Coverage:
1. TestVariance — V(mu) correct for each family at STRICT tolerance
2. TestDevianceResidsVsR — deviance residuals match R at STRICT
3. TestWorkingWeights — 1/(V(mu)*g'(mu)^2) computed correctly
4. TestInitialization — family.initialize(y, wt) produces valid starting mu
5. TestEdgeCases — boundary values and numerical-stability cases
6. TestRegistry — get_family("gaussian") returns Gaussian, etc.
7. TestFamilyJAXCompat — consolidated JAX parity and JIT checks
8. TestExtendedFamilyContract — generic ExtendedFamily contract checks
9. TestNBSpecific — NB-specific scientific correctness checks
10. TestFamilyStaticCacheKey/TestNBJITCacheReuse — NB JIT cache regressions
11. TestExtendedFamilyAD — finite-difference validation through extended families
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
    """Verify representative ResponseSupport bounds behavior."""

    @pytest.mark.parametrize(
        ("support_name", "values"),
        [
            ("REAL", np.array([-1e10, 0.0, 1e10])),
            ("NON_NEGATIVE", np.array([0.0, 1.0, 100.0])),
            ("POSITIVE", np.array([1e-300, 1.0])),
            ("UNIT_INTERVAL", np.array([0.0, 0.5, 1.0])),
        ],
    )
    def test_accepts_values_inside_support(
        self, support_name: str, values: np.ndarray
    ) -> None:
        import jaxgam.families.base as base

        assert getattr(base, support_name).check(values)

    @pytest.mark.parametrize(
        ("support_name", "values"),
        [
            ("NON_NEGATIVE", np.array([0.0, -0.001])),
            ("POSITIVE", np.array([0.0, 1.0])),
            ("UNIT_INTERVAL", np.array([-0.001, 0.5, 1.001])),
        ],
    )
    def test_rejects_values_outside_support(
        self, support_name: str, values: np.ndarray
    ) -> None:
        import jaxgam.families.base as base

        assert not getattr(base, support_name).check(values)


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


# ---------------------------------------------------------------------------
# Test 3: Working weights
# ---------------------------------------------------------------------------


class TestWorkingWeights:
    """Working weights W = wt / (V(mu) * g'(mu)^2)."""

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

    def test_all_families_produce_valid_mu(self) -> None:
        """For each family, initialize produces mu that passes valid_mu."""
        rng = np.random.default_rng(SEED)
        cases: list[tuple[ExponentialFamily, np.ndarray]] = [
            (Gaussian(), rng.normal(0, 1, 100)),
            (Binomial(), rng.binomial(1, 0.5, 100).astype(float)),
            (Poisson(), rng.poisson(3.0, 100).astype(float)),
            (Gamma(), rng.gamma(5.0, 1.0, 100)),
            (NegativeBinomial(), rng.negative_binomial(2.0, 0.4, 100).astype(float)),
        ]

        for fam, y in cases:
            mu = fam.initialize(y, np.ones_like(y))
            assert np.all(fam.valid_mu(mu)), fam.family_name


# ---------------------------------------------------------------------------
# Test 5: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case handling: boundary y values and extreme mu."""

    @pytest.mark.parametrize(
        ("y_val", "mu"),
        [
            (0.0, np.array([0.3, 0.5, 0.9])),
            (1.0, np.array([0.1, 0.5, 0.9])),
        ],
    )
    def test_binomial_boundary_y(self, y_val: float, mu: np.ndarray) -> None:
        """Binomial with y at the boundary: residuals finite and non-negative."""
        y = np.full_like(mu, y_val)
        wt = np.ones(3)
        fam = Binomial()
        dr = fam.deviance_resids(y, mu, wt)
        assert np.all(np.isfinite(dr)), f"Binomial dev resids not finite for y={y_val}"
        assert np.all(dr**2 >= 0), f"Binomial dev resids^2 negative for y={y_val}"

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

    def test_nb_poisson_limit(self) -> None:
        """NB approaches Poisson variance and deviance as theta -> inf."""
        mu = np.array([0.5, 1.0, 5.0, 10.0])
        nb_fam = NegativeBinomial(theta=1e8)
        poi_fam = Poisson()
        v = nb_fam.variance(mu)
        np.testing.assert_allclose(
            v,
            mu,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="NB variance should approach Poisson at large theta",
        )

        y = np.array([1.0, 3.0, 5.0, 10.0])
        mu = np.array([1.5, 2.5, 5.5, 8.0])
        wt = np.ones(4)
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

    def test_get_family_unknown_raises(self) -> None:
        """Unknown family name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown family"):
            get_family("nonexistent_family")

    def test_get_family_wrong_type_raises(self) -> None:
        """Non-string, non-family argument raises TypeError."""
        with pytest.raises(TypeError):
            get_family(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 9: Working response
# ---------------------------------------------------------------------------


class TestWorkingResponse:
    """PIRLS working response: z = eta + (y - mu) * g'(mu)."""

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


class TestFamilyJAXCompat:
    """JAX compatibility: PIRLS-path methods accept JAX arrays."""

    def test_pirls_methods_jax_match_numpy(self) -> None:
        """PIRLS-path methods agree for JAX and NumPy inputs."""
        for fam in FAMILIES:
            jax_y, jax_mu, jax_wt, jax_eta = _jax_family_test_data(fam)
            np_y = np.asarray(jax_y)
            np_mu = np.asarray(jax_mu)
            np_wt = np.asarray(jax_wt)
            np_eta = np.asarray(jax_eta)

            np.testing.assert_allclose(
                np.asarray(fam.variance(jax_mu)),
                fam.variance(np_mu),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"variance() JAX vs NumPy for {fam.family_name}",
            )
            np.testing.assert_allclose(
                np.asarray(fam.deviance_resids(jax_y, jax_mu, jax_wt)),
                fam.deviance_resids(np_y, np_mu, np_wt),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"deviance_resids() JAX vs NumPy for {fam.family_name}",
            )
            np.testing.assert_allclose(
                float(fam.dev_resids(jax_y, jax_mu, jax_wt)),
                float(fam.dev_resids(np_y, np_mu, np_wt)),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"dev_resids() JAX vs NumPy for {fam.family_name}",
            )
            np.testing.assert_allclose(
                np.asarray(fam.working_weights(jax_mu, jax_wt)),
                fam.working_weights(np_mu, np_wt),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"working_weights() JAX vs NumPy for {fam.family_name}",
            )
            np.testing.assert_allclose(
                np.asarray(fam.working_response(jax_y, jax_mu, jax_eta)),
                fam.working_response(np_y, np_mu, np_eta),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"working_response() JAX vs NumPy for {fam.family_name}",
            )

    def test_pirls_methods_jit_compile(self) -> None:
        """variance, deviance_resids, working_weights, working_response
        all JIT-compile without error."""
        for fam in FAMILIES:
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
        return y, mu, wt

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
        max_y = int(np.max(np.asarray(y)))

        ls_std = float(efamily.saturated_loglik(y, wt, 1.0, max_y=max_y))
        ls_theta = float(
            efamily.saturated_loglik_theta(y, wt, 1.0, log_theta, max_y=max_y)
        )
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
        max_y = int(np.max(np.asarray(y)))

        grad_ls = jax.grad(
            lambda lt: efamily.saturated_loglik_theta(y, wt, 1.0, lt, max_y=max_y)
        )(log_theta)
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

    def test_valid_mu(self) -> None:
        fam = NegativeBinomial()
        mu = np.array([-1.0, 0.0, 0.001, 1.0, 100.0])
        expected = np.array([False, False, True, True, True])
        np.testing.assert_array_equal(fam.valid_mu(mu), expected)


# ---------------------------------------------------------------------------
# Test 9b: JIT static-arg cache key (__hash__ / __eq__)
# ---------------------------------------------------------------------------


class TestFamilyStaticCacheKey:
    """Family ``__hash__`` / ``__eq__`` are used as JIT static-arg cache keys.

    Two instances that produce identical compiled code must compare equal
    and hash equal; instances that would bake different values into the
    trace must differ. Without this, ``copy.deepcopy`` (used in ``GAM.fit``
    to protect mutable extended-family state) causes a JIT cache miss on
    every fit.
    """

    def test_standard_family_singletons_equal(self) -> None:
        a = Gaussian()
        b = Gaussian()
        assert a == b
        assert hash(a) == hash(b)

    def test_different_standard_families_not_equal(self) -> None:
        assert Gaussian() != Poisson()
        assert hash(Gaussian()) != hash(Poisson())

    def test_estimated_nb_deepcopy_equal(self) -> None:
        """Deep-copied estimated NB must hash-equal the original."""
        import copy

        fam = NegativeBinomial(theta=1.0)
        fam_copy = copy.deepcopy(fam)
        assert fam == fam_copy
        assert hash(fam) == hash(fam_copy)

    def test_estimated_nb_theta_value_does_not_affect_key(self) -> None:
        """Estimated theta is a dynamic JAX argument — value must not key the trace."""
        a = NegativeBinomial(theta=1.0)
        b = NegativeBinomial(theta=5.0)
        assert a == b
        assert hash(a) == hash(b)

    def test_fixed_nb_theta_value_keys_trace(self) -> None:
        """Fixed theta is baked into the trace — different values must differ."""
        a = NegativeBinomial(theta=2.0, fixed=True)
        b = NegativeBinomial(theta=10.0, fixed=True)
        assert a != b
        assert hash(a) != hash(b)

    def test_fixed_nb_same_theta_equal(self) -> None:
        a = NegativeBinomial(theta=2.0, fixed=True)
        b = NegativeBinomial(theta=2.0, fixed=True)
        assert a == b
        assert hash(a) == hash(b)

    def test_fixed_vs_estimated_nb_not_equal(self) -> None:
        """``n_theta`` selects different code paths in PIRLS."""
        a = NegativeBinomial(theta=1.0, fixed=True)
        b = NegativeBinomial(theta=1.0, fixed=False)
        assert a != b
        assert hash(a) != hash(b)

    def test_different_links_not_equal(self) -> None:
        """Link type changes the trace (different link.inverse / derivative)."""
        from jaxgam.links.links import IdentityLink

        a = NegativeBinomial(theta=1.0, link="log")
        b = NegativeBinomial(theta=1.0, link=IdentityLink())
        assert a != b
        assert hash(a) != hash(b)

    def test_parameterized_custom_link_uses_identity(self) -> None:
        """Custom links default to identity-based cache key.

        Two instances of the same parameterized custom link class must
        not collide in the JIT cache, since they may bake different
        constants into the trace. Built-in stateless links opt into
        type-based keys via ``_stateless = True``.
        """
        from jaxgam.links.links import Link

        class ScaledLogLink(Link):
            def __init__(self, scale: float) -> None:
                self.scale = scale

            def link(self, mu):
                return self.scale * jnp.log(jnp.maximum(mu, 1e-10))

            def inverse(self, eta):
                return jnp.exp(eta / self.scale)

            def derivative(self, mu):
                return self.scale / jnp.maximum(mu, 1e-10)

        a = NegativeBinomial(theta=1.0, link=ScaledLogLink(1.0))
        b = NegativeBinomial(theta=1.0, link=ScaledLogLink(2.0))
        assert a != b
        assert hash(a) != hash(b)

    def test_stateless_builtin_links_share_key(self) -> None:
        """Two ``LogLink`` instances must share a cache key (stateless)."""
        from jaxgam.links.links import LogLink

        a = NegativeBinomial(theta=1.0, link=LogLink())
        b = NegativeBinomial(theta=1.0, link=LogLink())
        assert a == b
        assert hash(a) == hash(b)


class TestNBJITCacheReuse:
    """Repeated ``GAM.fit`` calls with NB must reuse JIT cache."""

    @staticmethod
    def _make_nb_theta_cache_data():
        import pandas as pd

        rng = np.random.default_rng(7)
        n = 80
        x = np.linspace(0.0, 1.0, n)
        mu = np.exp(0.6 + 0.5 * np.sin(2 * np.pi * x))
        y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
        return pd.DataFrame({"x": x, "y": y})

    def test_repeated_nb_fits_do_not_grow_jit_cache(self) -> None:
        import pandas as pd

        from jaxgam import GAM
        from jaxgam.fitting import newton as newton_mod

        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(0, 1, n)
        mu = np.exp(0.5 * np.sin(2 * np.pi * x) + 0.5)
        y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
        data = pd.DataFrame({"x": x, "y": y})

        # Warm up — first NB fit compiles.
        GAM('y ~ s(x, k=8, bs="cr")', family="nb").fit(data)
        size_after_warmup = newton_mod._jit_fit_and_score._cache_size()
        grad_size_after_warmup = newton_mod._jit_diff_grad_hess._cache_size()

        # Subsequent fits with the same shape/family must hit cache.
        for _ in range(3):
            GAM('y ~ s(x, k=8, bs="cr")', family="nb").fit(data)

        assert newton_mod._jit_fit_and_score._cache_size() == size_after_warmup, (
            "NB fit triggered _jit_fit_and_score recompile — likely a "
            "static-arg identity mismatch (see family.__hash__)."
        )
        assert newton_mod._jit_diff_grad_hess._cache_size() == grad_size_after_warmup, (
            "NB fit triggered _jit_diff_grad_hess recompile — likely a "
            "static-arg identity mismatch (see family.__hash__)."
        )

    def test_parametric_nb_theta_changes_are_dynamic_on_pirls_cache_hit(self) -> None:
        from jaxgam import GAM
        from jaxgam.fitting.pirls import pirls_loop

        data = self._make_nb_theta_cache_data()

        def fit(theta: float):
            return GAM(
                "y ~ x",
                family=NegativeBinomial(theta=theta, fixed=False),
            ).fit(data)

        pirls_loop.clear_cache()
        fresh_high = fit(10.0)

        pirls_loop.clear_cache()
        cached_low = fit(2.0)
        cached_high = fit(10.0)

        assert abs(float(cached_low.deviance - fresh_high.deviance)) > 1e-3
        np.testing.assert_allclose(
            cached_high.deviance,
            fresh_high.deviance,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            cached_high.coefficients,
            fresh_high.coefficients,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_fixed_sp_nb_estimates_theta_init_independent(self) -> None:
        """Fixed-`sp` NB estimates theta and is independent of the theta init.

        A fixed ``sp`` with an *estimated*-theta NB family
        (``fixed=False``) now optimizes theta — fixing the smoothing
        parameters does not fix theta (mgcv ``gam.fit4`` semantics). The
        converged fit is therefore independent of the theta *init*. Previously
        fixed-``sp`` ran a single PIRLS at the init theta (a bug), which this
        test originally documented by asserting that different inits gave
        different results.

        The dynamic-theta PIRLS cache is still exercised here: the second fit
        starts from a different theta init on a warm cache and theta estimation
        evaluates many thetas through one compiled kernel. The cache-size guard
        proper lives in ``test_repeated_nb_fits_do_not_grow_jit_cache``.
        """
        from jaxgam import GAM
        from jaxgam.fitting.pirls import pirls_loop

        data = self._make_nb_theta_cache_data()

        def fit(theta_init: float):
            return GAM(
                'y ~ s(x, k=8, bs="cr")',
                family=NegativeBinomial(theta=theta_init, fixed=False),
                sp=[1.0],
            ).fit(data)

        pirls_loop.clear_cache()
        fit_high = fit(10.0)
        fit_low = fit(2.0)  # different init, warm dynamic-theta cache

        # Theta is estimated, so the converged fit is independent of the init
        # (agreement is at optimizer-tolerance level, hence MODERATE).
        np.testing.assert_allclose(
            fit_low.theta, fit_high.theta, rtol=MODERATE.rtol, atol=MODERATE.atol
        )
        np.testing.assert_allclose(
            fit_low.deviance, fit_high.deviance, rtol=MODERATE.rtol, atol=MODERATE.atol
        )
        np.testing.assert_allclose(
            fit_low.coefficients,
            fit_high.coefficients,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )


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
        return y, mu, wt

    # ------------------------------------------------------------------
    # saturated_loglik_theta
    # ------------------------------------------------------------------

    def test_saturated_loglik_theta_grad(self, efamily, test_data) -> None:
        """d(ls_sat)/d(log_theta) via AD matches central FD."""
        y, _mu, wt = test_data
        y, wt = jnp.array(y), jnp.array(wt)
        log_theta = jnp.array(efamily.get_theta())
        max_y = int(np.max(np.asarray(y)))

        def _ls(lt):
            return efamily.saturated_loglik_theta(y, wt, 1.0, lt, max_y=max_y)

        ad_grad = np.asarray(jax.grad(_ls)(log_theta))
        # Larger eps to reduce cancellation error when |ls_sat| >> |gradient|
        # (near-Poisson regime: ls_sat ~ 82000, gradient ~ 0.003).
        fd_grad = _central_fd_grad(_ls, log_theta, eps=1e-4)
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
        max_y = int(np.max(np.asarray(y)))

        def ls_fn(lt):
            return efamily.saturated_loglik_theta(y, wt, 1.0, lt, max_y=max_y)

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
