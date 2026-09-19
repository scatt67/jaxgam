"""Global Gaussian fix.family starts and source-owned reporting."""

import subprocess
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.control import FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs import (
    _efs_regular_start,
    dense_efs_unknown_scale,
    efs_initial_log_lambda,
    efs_initial_log_scale,
)
from jaxgam.execution.regular_stream import fit_regular_streamed_pirls
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Gaussian
from jaxgam.fitting.data import FittingData, PreparedFittingMetadata
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    batch_execution_summary,
    finalize_execution_summary,
    merge_execution_summaries,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.results import GAMPredictionResult
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize("link", ["log", "inverse"])
def test_global_patched_gaussian_mustart_cpu_jit_and_batch_parity(tmp_path, link):
    y = np.array([-0.2, 0.0, 0.3, 0.8, 1.4, 0.0, 2.0, 0.4, 0.7])
    weights = np.array([1.0, 0.0, 0.5, 2.0, 0.0, 0.8, 1.0, 2.0, 3.0])
    # Positive global mean; response SD includes real zero-prior observations.
    np.savetxt(tmp_path / "y", y, fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
d <- commandArgs(TRUE)[1]; y <- scan(file.path(d,"y"),quiet=TRUE)
family <- mgcv:::fix.family(gaussian(commandArgs(TRUE)[2]))
nobs <- length(y); eval(family$initialize)
writeBin(as.double(c(sd(y),mustart)),file.path(d,"reference"),size=8,endian="little")
"""
    subprocess.run(
        ["Rscript", "-e", script, str(tmp_path), link],
        check=True,
        capture_output=True,
        text=True,
    )
    oracle = np.fromfile(tmp_path / "reference", dtype="<f8")
    family = Gaussian(link)
    context = FamilyExecutionContext.from_family(family)
    for eager in (False, True):
        for B in (1, 3, 20):
            summary = None
            with jax.disable_jit(eager):
                # Empty and invalid padding do not contribute global moments.
                padded = batch_execution_summary(
                    jnp.array([jnp.nan]),
                    jnp.array([-1.0]),
                    jnp.array([False]),
                    family,
                    context,
                )
                summary = padded
                for start in range(0, len(y), B):
                    batch = batch_execution_summary(
                        jnp.asarray(y[start : start + B]),
                        jnp.asarray(weights[start : start + B]),
                        jnp.ones(len(y[start : start + B]), bool),
                        family,
                        context,
                    )
                    summary = merge_execution_summaries(summary, batch, family, context)
            assert all(np.ndim(leaf) == 0 for leaf in jax.tree.leaves(summary))
            metadata = finalize_execution_summary(summary, family)
            np.testing.assert_allclose(
                metadata["response_sd"], oracle[0], rtol=STRICT.rtol, atol=STRICT.atol
            )
            for start in range(0, len(y), B):
                state = family.initial_working_state_cpu(
                    y[start : start + B],
                    weights[start : start + B],
                    np.ones(len(y[start : start + B]), bool),
                    summary=metadata,
                )
                assert state.input_ok
                assert state.domain_ok
                np.testing.assert_allclose(
                    state.mustart,
                    oracle[1 + start : 1 + start + B],
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                )


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize("link", ["log", "inverse"])
def test_patched_gaussian_start_matches_actual_public_gam(tmp_path, link):
    rng = np.random.default_rng(7321)
    x = np.linspace(-0.7, 0.8, 83)
    y = np.exp(0.2 + 0.3 * x) + rng.normal(0.0, 0.08, len(x))
    if link == "log":
        y[[2, 15]] = [-0.2, -0.05]
    else:
        y[[2, 15]] = 0.0
    weight = 0.4 + rng.uniform(size=len(x))
    weight[8] = 0.0
    offset = 0.03 * np.sin(x)
    np.savetxt(tmp_path / "data", np.column_stack((y, x, weight, offset)), fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
d <- commandArgs(TRUE)[1]
z <- as.data.frame(read.table(file.path(d,"data")))
names(z) <- c("y","x","w","off")
fit <- gam(y~x,data=z,weights=w,offset=off,family=gaussian(commandArgs(TRUE)[2]),
 method="REML",control=gam.control(epsilon=1e-12))
stopifnot(fit$converged)
writeBin(as.double(c(coef(fit),fit$deviance,fit$sig2,sum(fit$edf),fit$gcv.ubre)),
 file.path(d,"reference"),size=8,endian="little")
writeBin(as.double(fit$fitted.values),file.path(d,"fitted"),size=8,endian="little")
writeBin(as.double(fit$Vp),file.path(d,"covariance"),size=8,endian="little")
writeBin(as.double(fit$reml.scale),file.path(d,"score_phi"),size=8,endian="little")
"""
    subprocess.run(
        ["Rscript", "-e", script, str(tmp_path), link],
        check=True,
        capture_output=True,
        text=True,
    )
    oracle = np.fromfile(tmp_path / "reference", dtype="<f8")
    # gam.outer reports scale.est as sig2; the REML objective retains its
    # distinct optimized reml.scale, notably with real zero-prior rows.
    score_phi = float(np.fromfile(tmp_path / "score_phi", dtype="<f8")[0])
    oracle_covariance = np.fromfile(tmp_path / "covariance", dtype="<f8").reshape(2, 2)
    family = Gaussian(link)
    source = DataFrameRowSource(
        pd.DataFrame({"x": x, "y": y}), response="y", weights=weight, offset=offset
    )
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    stream = StreamDesign(prepared, source)
    X = np.column_stack((np.ones(len(x)), x))
    checks = _AssertCollector()
    for B in (1, 7, 200):
        result = fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=10000000,
            score_scale=score_phi,
            control=StreamPIRLSControl(
                batch_rows=B,
                tol=1e-12,
                max_iter=100,
                max_halvings=100,
                solver_policy="qr",
            ),
        )
        state = result.state
        assert state.converged
        prediction = GAMPredictionResult._from_stream_fit(
            stream_state=state,
            prepared=prepared,
            metadata=PreparedFittingMetadata.from_prepared(prepared, family),
            family=family,
            formula="y ~ x",
            method="REML",
            control=FitControl(),
        )
        covariance = state.scale * np.asarray(
            state.fisher_coefficient_factor.hessian_inverse(jnp.eye(2))
        )
        for leaf, value, reference in (
            (
                "coef_dev_scale_edf_reml",
                np.r_[
                    state.coefficients,
                    state.deviance,
                    state.scale,
                    state.edf,
                    prediction.score,
                ],
                oracle,
            ),
            (
                "fitted",
                family.link.inverse(X @ np.asarray(state.coefficients) + offset),
                np.fromfile(tmp_path / "fitted", dtype="<f8"),
            ),
            ("fisher_covariance", covariance, oracle_covariance),
            (
                "standard_error",
                np.sqrt(np.einsum("ij,jk,ik->i", X, covariance, X)),
                np.sqrt(np.einsum("ij,jk,ik->i", X, oracle_covariance, X)),
            ),
        ):
            checks.check(
                f"{link}/B{B}/{leaf}",
                lambda a=value, b=reference: np.testing.assert_allclose(
                    a, b, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
    checks.raise_if_any("Public Gaussian patched-start parity")


@pytest.mark.parametrize("link", ["log", "inverse"])
def test_dense_efs_null_start_consumes_shared_global_patched_initializer(link):
    y = np.array([0.0, 0.8, 1.2, 1.4, 2.0])
    if link == "log":
        y[0] = -0.2
    x = np.linspace(-0.5, 0.5, len(y))
    family = Gaussian(link)
    setup = ModelSetup.build(parse_formula("y ~ x"), pd.DataFrame({"x": x, "y": y}))
    fd = FittingData.from_setup(setup, family)
    seed = jnp.zeros(2)
    null_beta, null_eta, initial_eta = _efs_regular_start(fd, seed, start_present=False)
    sd = np.std(y, ddof=1)
    mustart = np.maximum(y, 0.01 * sd) if link == "log" else y + (y == 0) * sd * 0.01
    np.testing.assert_allclose(
        initial_eta,
        family.link.initial_link_cpu(mustart),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    expected_null = np.linalg.lstsq(
        np.asarray(fd.X),
        np.full(len(y), family.link.initial_link_cpu(np.asarray(np.mean(y)))),
        rcond=None,
    )[0]
    np.testing.assert_allclose(
        null_beta, expected_null, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        null_eta, np.asarray(fd.X) @ expected_null, rtol=STRICT.rtol, atol=STRICT.atol
    )
    _, _, retained_eta = _efs_regular_start(fd, jnp.ones(2), start_present=True)
    np.testing.assert_array_equal(retained_eta, fd.X @ jnp.ones(2))


def test_gaussian_global_metadata_is_bounded_and_preserves_unpatched_defaults():
    for link in ("identity", "sqrt"):
        family = Gaussian(link)
        raw = family.execution_summary_from_batch(
            np.array([0.3, 0.8]), np.ones(2), np.ones(2, bool)
        )
        assert len(raw) == 4
        assert "response_sd" not in family.finalize_execution_summary(raw)
    family = Gaussian("log")
    one = family.execution_summary_from_batch(
        np.array([0.3]), np.ones(1), np.ones(1, bool)
    )
    metadata = family.finalize_execution_summary(one)
    assert np.isnan(metadata["response_sd"])
    invalid = family.initial_working_state_cpu(
        np.array([0.3]), np.ones(1), np.ones(1, bool), summary=metadata
    )
    assert invalid.input_ok
    assert not invalid.domain_ok
    with pytest.raises(ValueError, match="summary shape"):
        family.finalize_execution_summary((1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="summary shape"):
        family.merge_execution_summaries(one, one[:4])


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize("link", ["log", "inverse"])
def test_efs_initial_smoothing_uses_patched_global_start_and_neutral_zero_weight(
    tmp_path, link
):
    x = np.linspace(-0.7, 0.8, 83)
    y = np.exp(0.2 + 0.3 * x)
    y[[2, 15]] = [-0.2, -0.05] if link == "log" else 0.0
    weights = np.linspace(0.4, 1.2, len(x))
    weights[8] = 0.0
    data = pd.DataFrame({"x": x, "y": y, "w": weights})
    data.to_csv(tmp_path / "data.csv", index=False)
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
d <- commandArgs(TRUE)[1]; z <- read.csv(file.path(d,"data.csv"))
G <- gam(y~s(x,bs="cr",k=7),data=z,weights=w,
 family=gaussian(commandArgs(TRUE)[2]),fit=FALSE)
G$family <- mgcv:::fix.family(G$family)
sp <- mgcv:::initial.spg(G$X,G$y,G$w,G$family,G$S,G$rank,G$off)
scale <- mgcv:::get.null.coef(G)$null.scale / 10
writeBin(as.double(c(log(sp),log(scale))),file.path(d,"reference"),size=8,endian="little")
"""
    subprocess.run(
        ["Rscript", "-e", script, str(tmp_path), link],
        check=True,
        capture_output=True,
        text=True,
    )
    family = Gaussian(link)
    setup = ModelSetup.build(
        parse_formula("y ~ s(x, bs='cr', k=7)"), data, weights=weights
    )
    np.testing.assert_allclose(
        np.r_[
            efs_initial_log_lambda(setup, family), efs_initial_log_scale(setup, family)
        ],
        np.fromfile(tmp_path / "reference", dtype="<f8"),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    with pytest.raises(ValueError, match="no finite informative"):
        efs_initial_log_lambda(replace(setup, weights=np.zeros(len(x))), family)
    with pytest.raises(ValueError, match="globally informative"):
        efs_initial_log_scale(replace(setup, weights=np.zeros(len(x))), family)
    for bad in (-1.0, np.nan, np.inf):
        invalid = weights.copy()
        invalid[0] = bad
        with pytest.raises(ValueError, match="prior weights"):
            efs_initial_log_lambda(replace(setup, weights=invalid), family)
        with pytest.raises(ValueError, match="finite nonnegative"):
            efs_initial_log_scale(replace(setup, weights=invalid), family)


def test_unknown_scale_efs_zero_prior_admission_keeps_source_and_bounds_explicit():
    x = np.linspace(-0.7, 0.8, 20)
    data = pd.DataFrame({"x": x, "y": np.exp(0.2 + 0.3 * x)})
    setup = ModelSetup.build(parse_formula("y ~ s(x, bs='cr', k=5)"), data)
    for link in ("identity", "log", "inverse"):
        fd = FittingData.from_setup(setup, Gaussian(link))
        for bad in (-1.0, np.nan, np.inf, 1e-11, 1e11):
            weights = np.ones(len(x))
            weights[0] = bad
            with pytest.raises(ValueError, match="clipping bounds"):
                dense_efs_unknown_scale(replace(fd, wt=jnp.asarray(weights)))
        if link == "identity":
            weights = np.ones(len(x))
            weights[0] = 0.0
            with pytest.raises(ValueError, match="clipping bounds"):
                dense_efs_unknown_scale(replace(fd, wt=jnp.asarray(weights)))
        else:
            with pytest.raises(ValueError, match="globally informative"):
                dense_efs_unknown_scale(replace(fd, wt=jnp.zeros(len(x))))
