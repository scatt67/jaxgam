"""Raw stats Gaussian AIC diagnostics stay distinct from score likelihood."""

import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.families.standard import Gaussian
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


def _cases():
    y = np.array([0.2, 1.0, -0.5, 2.0])
    mu = np.array([0.0, 1.1, -0.4, 1.7])
    return [
        (y, mu, np.ones(4)),
        (y, mu, np.array([0.2, 0.7, 1.5, 4.0])),
        (y, mu, np.array([0.0, 1.0, 2.0, 0.0])),
        (y, mu, np.zeros(4)),
        (y, y, np.ones(4)),
        (y, y, np.array([0.0, 1.0, 2.0, 0.0])),
        (np.empty(0), np.empty(0), np.empty(0)),
        # Preserve the discovered finite-vs-Inf public-method discrepancy.
        (np.array([0.0, 1.0]), np.array([0.1, 0.9]), np.array([0.0, 1.0])),
    ]


@pytest.mark.usefixtures("r_bridge")
def test_gaussian_aic_cpu_matches_raw_pinned_weight_boundaries(tmp_path):
    cases = _cases()
    for i, (y, mu, w) in enumerate(cases):
        for name, value in (("y", y), ("mu", mu), ("w", w)):
            np.savetxt(tmp_path / f"{name}{i}", value, fmt="%.17g")
    script = r"""
stopifnot(getRversion()=="4.5.2")
d <- commandArgs(TRUE)[1]; family <- gaussian()
for (i in 0:7) {
 read <- function(name) scan(file.path(d,paste0(name,i)),quiet=TRUE)
 y <- read("y"); mu <- read("mu"); w <- read("w")
 dev <- sum(family$dev.resids(y,mu,w))
 value <- family$aic(y,rep(1,length(y)),mu,w,dev)
 writeBin(as.double(value),file.path(d,paste0("reference",i)),size=8,endian="little")
}
"""
    subprocess.run(
        ["Rscript", "-e", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    checks = _AssertCollector()
    for i, (y, mu, w) in enumerate(cases):
        oracle = np.fromfile(tmp_path / f"reference{i}", dtype="<f8")[0]
        for scale in (0.2, 1.0, 7.0):
            actual = Gaussian().aic(y, mu, w, scale)
            checks.check(
                f"case{i}/scale{scale}",
                lambda a=actual, r=oracle: np.testing.assert_allclose(
                    a, r, rtol=STRICT.rtol, atol=STRICT.atol, equal_nan=True
                ),
            )
    checks.raise_if_any("Pinned raw Gaussian AIC diagnostics")
    assert np.isposinf(Gaussian().aic(*cases[-1], scale=1.0))


@pytest.mark.usefixtures("r_bridge")
def test_gaussian_zero_prior_saturated_likelihood_remains_finite_cpu_jit(tmp_path):
    y = np.array([0.2, 1.0, -0.5, 2.0])
    w = np.array([0.2, 0.0, 0.7, 1.5])
    phi = 0.7
    for name, value in (("y", y), ("w", w)):
        np.savetxt(tmp_path / name, value, fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
d <- commandArgs(TRUE)[1]
y <- scan(file.path(d,"y"),quiet=TRUE); w <- scan(file.path(d,"w"),quiet=TRUE)
fam <- mgcv:::fix.family.ls(gaussian())
writeBin(as.double(fam$ls(y,w,rep(1,length(y)),.7)),file.path(d,"reference"),size=8,endian="little")
"""
    subprocess.run(
        ["Rscript", "-e", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    oracle = np.fromfile(tmp_path / "reference", dtype="<f8")
    family = Gaussian()

    def likelihood(scale):
        return family.saturated_loglik(jnp.asarray(y), jnp.asarray(w), scale)

    def derivatives(scale):
        return jnp.array(
            [
                likelihood(scale),
                jax.grad(likelihood)(scale),
                jax.grad(jax.grad(likelihood))(scale),
            ]
        )

    for eager in (False, True):
        with jax.disable_jit(eager):
            actual = jax.jit(derivatives)(phi)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, oracle, rtol=STRICT.rtol, atol=STRICT.atol)
