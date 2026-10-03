"""Pinned single-response likelihood and valid-domain Binomial curvature."""

import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.families.standard import Binomial
from tests.tolerances import STRICT


@pytest.mark.usefixtures("r_bridge")
def test_binomial_fractional_trial_weights_follow_pinned_dbinom(tmp_path):
    weights = np.r_[np.geomspace(0.05, 100.0, 81), 0.8, 0.0]
    y = np.resize(
        np.array([0.0, 0.6, 1.0, np.nextafter(1.0, 0.0), 1.0 - 1e-12]), len(weights)
    )
    y[-1] = 0.0
    mu = np.linspace(0.13, 0.87, len(y))
    np.savetxt(tmp_path / "input", np.column_stack((y, weights, mu)), fmt="%.17g")
    script = r"""
    library(mgcv)
    stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
    d <- commandArgs(TRUE)[1]
    x <- as.matrix(read.table(file.path(d,"input")))
    fam <- mgcv:::fix.family.ls(binomial())
    options(digits=17)
    write.table(c(fam$ls(x[,1],x[,2],rep(1,nrow(x)),1)[1],
      binomial()$aic(x[,1],rep(1,nrow(x)),x[,3],x[,2],0)),
      file.path(d,"reference"),row.names=FALSE,col.names=FALSE)
    """
    subprocess.run(
        ["Rscript", "-e", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    reference = np.loadtxt(tmp_path / "reference")
    family = Binomial()
    for disabled in (False, True):
        with jax.disable_jit(disabled):
            likelihood = jax.jit(lambda yy, ww: family.saturated_loglik(yy, ww, 1.0))(
                jnp.asarray(y), jnp.asarray(weights)
            )
        np.testing.assert_allclose(
            likelihood, reference[0], rtol=STRICT.rtol, atol=STRICT.atol
        )
    np.testing.assert_allclose(
        family.aic(y, mu, weights, 1.0),
        reference[1],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.usefixtures("r_bridge")
def test_binomial_valid_boundary_means_retain_source_deviance_and_curvature(tmp_path):
    mu = np.array([1e-12, 1e-11, 0.2, 1.0 - 1e-12, np.nextafter(1.0, 0.0)])
    y = np.array([0.0, 0.1, 0.7, 1.0 - 1e-12, 1.0])
    weight = np.array([0.3, 0.8, 1.2, 2.0, 0.7])
    np.savetxt(tmp_path / "input", np.column_stack((y, weight, mu)), fmt="%.17g")
    script = r"""
    library(mgcv)
    stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
    d <- commandArgs(TRUE)[1]
    x <- as.matrix(read.table(file.path(d,"input")))
    y <- x[,1]; w <- x[,2]; mu <- x[,3]
    # Half the second derivative of stats binomial's direct deviance in mu.
    curvature <- w*(y/mu^2+(1-y)/(1-mu)^2)
    options(digits=17)
    write.table(cbind(binomial()$dev.resids(y,mu,w),curvature),
      file.path(d,"reference"),row.names=FALSE,col.names=FALSE)
    """
    subprocess.run(
        ["Rscript", "-e", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    reference = np.loadtxt(tmp_path / "reference")
    family = Binomial()
    np.testing.assert_allclose(
        family.deviance_contributions(y, mu, weight),
        reference[:, 0],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        family.deviance_resids(y, mu, weight) ** 2,
        reference[:, 0],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )

    def direct(mean):
        return jnp.sum(
            family.deviance_derivative_contributions(
                jnp.asarray(y), mean, jnp.asarray(weight)
            )
        )

    kernel = jax.jit(lambda mean: 0.5 * jax.hessian(direct)(mean))
    for disabled in (False, True):
        with jax.disable_jit(disabled):
            hessian = kernel(jnp.asarray(mu))
        np.testing.assert_allclose(
            hessian, np.diag(reference[:, 1]), rtol=STRICT.rtol, atol=STRICT.atol
        )
    assert np.all(reference[:, 1] > 0)


@pytest.mark.usefixtures("r_bridge")
def test_binomial_cpu_aic_preserves_pinned_tail_endpoint_and_zero_trial_terms(tmp_path):
    y = np.array([0.0, 1.0, 0.6, 1.0, 0.0, 1.0, 777.0, -777.0, 1.0, 0.0])
    weight = np.array([0.3, 0.8, 2.3, 15.9, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    mu = np.array(
        [
            1e-12,
            1.0 - 1e-12,
            0.4,
            np.nextafter(1.0, 0.0),
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ]
    )
    np.savetxt(tmp_path / "input", np.column_stack((y, weight, mu)), fmt="%.17g")
    script = r"""
    stopifnot(getRversion()=="4.5.2")
    d <- commandArgs(TRUE)[1]
    x <- as.matrix(read.table(file.path(d,"input")))
    reference <- vapply(seq_len(nrow(x)),function(i) {
      binomial()$aic(x[i,1],1,x[i,3],x[i,2],0)
    },numeric(1))
    options(digits=17)
    write.table(reference,file.path(d,"reference"),row.names=FALSE,col.names=FALSE)
    """
    subprocess.run(
        ["Rscript", "-e", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    reference = np.loadtxt(tmp_path / "reference")
    family = Binomial()
    actual = np.array(
        [
            family.aic(y[i : i + 1], mu[i : i + 1], weight[i : i + 1], 1.0)
            for i in range(len(y))
        ]
    )
    np.testing.assert_allclose(actual, reference, rtol=STRICT.rtol, atol=STRICT.atol)
    assert np.all(np.isfinite(actual[:-2]))
    assert np.all(np.isposinf(actual[-2:]))
    assert np.all(actual[4:8] == 0.0)
