"""Tests for pymgcv.links.

Coverage:
1. TestLinkRoundtrip — linkinv(link(mu)) ≈ mu for each link
2. TestMuEta — mu_eta(eta) matches finite differences
3. TestLinkVsR — link values match R's make.link() output
4. TestLinkRegistry — from_name returns correct class, unknown raises KeyError
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

from pymgcv.links import (
    CloglogLink,
    IdentityLink,
    InverseLink,
    InverseSquaredLink,
    Link,
    LogitLink,
    LogLink,
    ProbitLink,
    SqrtLink,
)
from tests.tolerances import MODERATE, STRICT

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Fixtures: mu test points for each link family
# ---------------------------------------------------------------------------


def _mu_unit_interval() -> np.ndarray:
    """mu values in (0, 1) including near boundaries."""
    return np.array([1e-8, 1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-4])


def _mu_positive() -> np.ndarray:
    """mu values in (0, ∞) for log/sqrt/inverse links."""
    return np.array([1e-6, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])


def _mu_positive_moderate() -> np.ndarray:
    """mu values in moderate range — avoids extremes where FD is ill-conditioned."""
    return np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])


def _mu_real() -> np.ndarray:
    """mu values spanning ℝ for identity link."""
    return np.array([-10.0, -1.0, 0.0, 0.5, 1.0, 5.0, 100.0])


# ---------------------------------------------------------------------------
# Map each link to appropriate mu test values
# ---------------------------------------------------------------------------

LINK_MU_MAP: list[tuple[Link, np.ndarray]] = [
    (IdentityLink(), _mu_real()),
    (LogLink(), _mu_positive()),
    (LogitLink(), _mu_unit_interval()),
    (InverseLink(), _mu_positive()),
    (ProbitLink(), _mu_unit_interval()),
    (CloglogLink(), _mu_unit_interval()),
    (SqrtLink(), _mu_positive()),
    (InverseSquaredLink(), _mu_positive()),
]

LINK_IDS = [
    "identity",
    "log",
    "logit",
    "inverse",
    "probit",
    "cloglog",
    "sqrt",
    "inverse_squared",
]

# Finite difference test uses well-conditioned mu ranges to avoid
# regions where curvature makes FD inherently inaccurate.
FD_LINK_MU_MAP: list[tuple[Link, np.ndarray]] = [
    (IdentityLink(), _mu_real()),
    (LogLink(), _mu_positive()),
    (LogitLink(), _mu_unit_interval()),
    (InverseLink(), _mu_positive()),
    (ProbitLink(), _mu_unit_interval()),
    (CloglogLink(), _mu_unit_interval()),
    (SqrtLink(), _mu_positive()),
    (InverseSquaredLink(), _mu_positive_moderate()),
]


# ---------------------------------------------------------------------------
# Test 1: Roundtrip — linkinv(link(mu)) ≈ mu
# ---------------------------------------------------------------------------


class TestLinkRoundtrip:
    @pytest.mark.parametrize("link_obj,mu", LINK_MU_MAP, ids=LINK_IDS)
    def test_roundtrip(self, link_obj: Link, mu: np.ndarray) -> None:
        eta = link_obj.link(mu)
        mu_recovered = link_obj.linkinv(eta)
        np.testing.assert_allclose(
            mu_recovered,
            mu,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"Roundtrip failed for {type(link_obj).__name__}",
        )

    @pytest.mark.parametrize("link_obj,mu", LINK_MU_MAP, ids=LINK_IDS)
    def test_inverse_alias(self, link_obj: Link, mu: np.ndarray) -> None:
        """linkinv and inverse return the same thing."""
        eta = link_obj.link(mu)
        np.testing.assert_array_equal(
            link_obj.linkinv(eta),
            link_obj.inverse(eta),
        )


# ---------------------------------------------------------------------------
# Test 2: mu_eta matches finite differences of linkinv
# ---------------------------------------------------------------------------


class TestMuEta:
    """Verify mu_eta against finite differences.

    Uses MODERATE tolerance because central finite differences have
    inherent O(h²) + O(eps/h) error that limits achievable accuracy.
    The analytical mu_eta correctness is validated to STRICT by the
    R comparison tests (TestLinkVsR).
    """

    @pytest.mark.parametrize("link_obj,mu", FD_LINK_MU_MAP, ids=LINK_IDS)
    def test_mu_eta_vs_finite_diff(self, link_obj: Link, mu: np.ndarray) -> None:
        eta = link_obj.link(mu)
        # Optimal step for central differences: h = eps^(1/3) * scale
        eps_third = np.float64(np.finfo(np.float64).eps) ** (1.0 / 3.0)
        h = eps_third * np.maximum(1.0, np.abs(eta))
        fd = (link_obj.linkinv(eta + h) - link_obj.linkinv(eta - h)) / (2 * h)
        mu_eta_val = link_obj.mu_eta(eta)
        np.testing.assert_allclose(
            mu_eta_val,
            fd,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg=f"mu_eta mismatch for {type(link_obj).__name__}",
        )


# ---------------------------------------------------------------------------
# Test 3: R comparison — link values match R's make.link()
# ---------------------------------------------------------------------------


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


def _compute_r_link_values(
    link_name: str, mu_values: np.ndarray
) -> dict[str, np.ndarray]:
    """Call R's make.link() and return link, linkinv, mu.eta values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mu_path = os.path.join(tmpdir, "mu.csv")
        out_path = os.path.join(tmpdir, "result.json")

        np.savetxt(mu_path, mu_values, delimiter=",")

        # R script that computes link, linkinv, and mu.eta via make.link
        script = f"""\
mu <- scan("{mu_path}", sep=",", quiet=TRUE)
lnk <- make.link("{link_name}")
eta <- lnk$linkfun(mu)
mu_back <- lnk$linkinv(eta)
mu_eta <- lnk$mu.eta(eta)
result <- list(eta=eta, mu_back=mu_back, mu_eta=mu_eta)
# Write as JSON-like text
cat(sprintf('{{"eta": [%s], "mu_back": [%s], "mu_eta": [%s]}}',
    paste(format(eta, digits=17), collapse=","),
    paste(format(mu_back, digits=17), collapse=","),
    paste(format(mu_eta, digits=17), collapse=",")),
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

        return {
            "eta": np.array(data["eta"], dtype=np.float64),
            "mu_back": np.array(data["mu_back"], dtype=np.float64),
            "mu_eta": np.array(data["mu_eta"], dtype=np.float64),
        }


# R's make.link supports these names
R_LINK_MAP: list[tuple[str, Link, np.ndarray]] = [
    ("identity", IdentityLink(), _mu_real()),
    ("log", LogLink(), _mu_positive()),
    ("logit", LogitLink(), _mu_unit_interval()),
    ("inverse", InverseLink(), _mu_positive()),
    ("probit", ProbitLink(), _mu_unit_interval()),
    ("cloglog", CloglogLink(), _mu_unit_interval()),
    ("sqrt", SqrtLink(), _mu_positive()),
    # Note: R's make.link doesn't have "inverse_squared" by that name,
    # but 1/mu^2 is the Inverse Gaussian link. We test it via custom R code.
    ("1/mu^2", InverseSquaredLink(), _mu_positive()),
]


@pytest.mark.skipif(not _r_available(), reason="R not available")
class TestLinkVsR:
    @pytest.mark.parametrize(
        "r_name,link_obj,mu",
        R_LINK_MAP,
        ids=[
            "identity",
            "log",
            "logit",
            "inverse",
            "probit",
            "cloglog",
            "sqrt",
            "inverse_squared",
        ],
    )
    def test_link_values_match_r(
        self, r_name: str, link_obj: Link, mu: np.ndarray
    ) -> None:
        r_vals = _compute_r_link_values(r_name, mu)

        # Compare link(mu) = eta
        eta_py = link_obj.link(mu)
        np.testing.assert_allclose(
            eta_py,
            r_vals["eta"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"link() mismatch vs R for {r_name}",
        )

        # Compare linkinv(eta)
        mu_back_py = link_obj.linkinv(r_vals["eta"])
        np.testing.assert_allclose(
            mu_back_py,
            r_vals["mu_back"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"linkinv() mismatch vs R for {r_name}",
        )

        # Compare mu_eta(eta)
        mu_eta_py = link_obj.mu_eta(r_vals["eta"])
        np.testing.assert_allclose(
            mu_eta_py,
            r_vals["mu_eta"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"mu_eta() mismatch vs R for {r_name}",
        )


# ---------------------------------------------------------------------------
# Test 4: Registry
# ---------------------------------------------------------------------------


class TestLinkRegistry:
    @pytest.mark.parametrize(
        "name,expected_cls",
        [
            ("identity", IdentityLink),
            ("log", LogLink),
            ("logit", LogitLink),
            ("inverse", InverseLink),
            ("probit", ProbitLink),
            ("cloglog", CloglogLink),
            ("sqrt", SqrtLink),
            ("inverse_squared", InverseSquaredLink),
        ],
    )
    def test_from_name(self, name: str, expected_cls: type) -> None:
        link_obj = Link.from_name(name)
        assert isinstance(link_obj, expected_cls)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            Link.from_name("nonexistent_link")


# ---------------------------------------------------------------------------
# Test 5: JAX compatibility — links accept JAX arrays and JIT-compile
# ---------------------------------------------------------------------------

# JAX-compatible mu test points (same values as NumPy helpers above)
_JAX_MU_UNIT = jnp.array([1e-8, 1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-4])
_JAX_MU_POS = jnp.array([1e-6, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
_JAX_MU_REAL = jnp.array([-10.0, -1.0, 0.0, 0.5, 1.0, 5.0, 100.0])

JAX_LINK_MU_MAP: list[tuple[Link, jax.Array, np.ndarray]] = [
    (IdentityLink(), _JAX_MU_REAL, _mu_real()),
    (LogLink(), _JAX_MU_POS, _mu_positive()),
    (LogitLink(), _JAX_MU_UNIT, _mu_unit_interval()),
    (InverseLink(), _JAX_MU_POS, _mu_positive()),
    (ProbitLink(), _JAX_MU_UNIT, _mu_unit_interval()),
    (CloglogLink(), _JAX_MU_UNIT, _mu_unit_interval()),
    (SqrtLink(), _JAX_MU_POS, _mu_positive()),
    (InverseSquaredLink(), _JAX_MU_POS, _mu_positive()),
]


class TestLinkJAXCompat:
    """JAX compatibility: link methods accept JAX arrays and JIT-compile."""

    @pytest.mark.parametrize("link_obj,jax_mu,np_mu", JAX_LINK_MU_MAP, ids=LINK_IDS)
    def test_link_jax_matches_numpy(
        self, link_obj: Link, jax_mu: jax.Array, np_mu: np.ndarray
    ) -> None:
        """link(jax_mu) matches link(np_mu)."""
        jax_eta = link_obj.link(jax_mu)
        np_eta = link_obj.link(np_mu)
        np.testing.assert_allclose(
            np.asarray(jax_eta),
            np_eta,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"link() JAX vs NumPy for {type(link_obj).__name__}",
        )

    @pytest.mark.parametrize("link_obj,jax_mu,np_mu", JAX_LINK_MU_MAP, ids=LINK_IDS)
    def test_inverse_jax_matches_numpy(
        self, link_obj: Link, jax_mu: jax.Array, np_mu: np.ndarray
    ) -> None:
        """inverse(link(jax_mu)) matches inverse(link(np_mu))."""
        jax_eta = link_obj.link(jax_mu)
        np_eta = link_obj.link(np_mu)
        jax_mu_back = link_obj.inverse(jax_eta)
        np_mu_back = link_obj.inverse(np_eta)
        np.testing.assert_allclose(
            np.asarray(jax_mu_back),
            np_mu_back,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"inverse() JAX vs NumPy for {type(link_obj).__name__}",
        )

    @pytest.mark.parametrize("link_obj,jax_mu,np_mu", JAX_LINK_MU_MAP, ids=LINK_IDS)
    def test_derivative_jax_matches_numpy(
        self, link_obj: Link, jax_mu: jax.Array, np_mu: np.ndarray
    ) -> None:
        """derivative(jax_mu) matches derivative(np_mu)."""
        jax_deriv = link_obj.derivative(jax_mu)
        np_deriv = link_obj.derivative(np_mu)
        np.testing.assert_allclose(
            np.asarray(jax_deriv),
            np_deriv,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"derivative() JAX vs NumPy for {type(link_obj).__name__}",
        )

    @pytest.mark.parametrize("link_obj,jax_mu,np_mu", JAX_LINK_MU_MAP, ids=LINK_IDS)
    def test_mu_eta_jax_matches_numpy(
        self, link_obj: Link, jax_mu: jax.Array, np_mu: np.ndarray
    ) -> None:
        """mu_eta(link(jax_mu)) matches mu_eta(link(np_mu))."""
        jax_eta = link_obj.link(jax_mu)
        np_eta = link_obj.link(np_mu)
        jax_me = link_obj.mu_eta(jax_eta)
        np_me = link_obj.mu_eta(np_eta)
        np.testing.assert_allclose(
            np.asarray(jax_me),
            np_me,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"mu_eta() JAX vs NumPy for {type(link_obj).__name__}",
        )

    @pytest.mark.parametrize("link_obj,jax_mu,_np_mu", JAX_LINK_MU_MAP, ids=LINK_IDS)
    def test_link_jit_compiles(
        self, link_obj: Link, jax_mu: jax.Array, _np_mu: np.ndarray
    ) -> None:
        """All 4 methods JIT-compile without error."""
        jit_link = jax.jit(link_obj.link)
        jit_inverse = jax.jit(link_obj.inverse)
        jit_deriv = jax.jit(link_obj.derivative)
        jit_mu_eta = jax.jit(link_obj.mu_eta)

        eta = jit_link(jax_mu)
        mu_back = jit_inverse(eta)
        deriv = jit_deriv(jax_mu)
        me = jit_mu_eta(eta)

        # Sanity: results are finite
        assert jnp.all(jnp.isfinite(eta)), "JIT link() produced non-finite"
        assert jnp.all(jnp.isfinite(mu_back)), "JIT inverse() produced non-finite"
        assert jnp.all(jnp.isfinite(deriv)), "JIT derivative() produced non-finite"
        assert jnp.all(jnp.isfinite(me)), "JIT mu_eta() produced non-finite"

    @pytest.mark.parametrize("link_obj,jax_mu,_np_mu", JAX_LINK_MU_MAP, ids=LINK_IDS)
    def test_roundtrip_jax(
        self, link_obj: Link, jax_mu: jax.Array, _np_mu: np.ndarray
    ) -> None:
        """linkinv(link(jax_mu)) ≈ jax_mu."""
        eta = link_obj.link(jax_mu)
        mu_recovered = link_obj.linkinv(eta)
        np.testing.assert_allclose(
            np.asarray(mu_recovered),
            np.asarray(jax_mu),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"JAX roundtrip failed for {type(link_obj).__name__}",
        )
