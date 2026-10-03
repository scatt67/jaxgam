"""Retained regular starts, natural null projection and all-link source systems."""

import hashlib
import subprocess
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.control import FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.regular_stream import (
    fit_regular_streamed_pirls,
    preflight_regular_stream_workspace,
)
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.results import GAMPredictionResult
from tests.helpers import _AssertCollector
from tests.tolerances import MODERATE, STRICT

_LINKS = (
    "identity",
    "log",
    "inverse",
    "sqrt",
    "logit",
    "probit",
    "cloglog",
    "inverse_squared",
)
_ANCHORS = {
    "identity": 0.55,
    "log": -0.65,
    "inverse": 2.0,
    "sqrt": 0.7,
    "logit": -0.2,
    "probit": -0.2,
    "cloglog": -0.6,
    "inverse_squared": 2.0,
}


def _case(family_class, link):
    x = np.linspace(-0.6, 0.7, 83)
    family = family_class(link)
    start = np.array([_ANCHORS[link], 0.08])
    mu = np.asarray(family.link.inverse(start[0] + start[1] * x))
    rng = np.random.default_rng(73300 + _LINKS.index(link))
    if family_class is Gaussian:
        y = mu + rng.normal(0, 0.025, len(x))
    elif family_class is Gamma:
        y = mu * np.exp(rng.normal(0, 0.08, len(x)))
    elif family_class is Poisson:
        y = rng.poisson(mu).astype(float)
    else:
        y = rng.binomial(1, mu).astype(float)
    weight = 0.4 + rng.uniform(size=len(x))
    weight[3] = 0.0
    offset = 0.01 * np.sin(x)
    return family, pd.DataFrame({"x": x, "y": y}), weight, offset, start


def _reference(
    tmp_path,
    family,
    link,
    X,
    y,
    weight,
    offset,
    start=None,
    *,
    tolerance=1e-7,
    max_iter=200,
    require_convergence=True,
):
    for name, value in (("X", X), ("y", y), ("w", weight), ("off", offset)):
        np.savetxt(tmp_path / name, value, fmt="%.17g")
    if start is not None:
        np.savetxt(tmp_path / "start", start, fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
a <- commandArgs(TRUE); d <- a[1]
read <- function(name) as.matrix(read.table(file.path(d,name)))
X <- read("X"); y <- c(read("y")); w <- c(read("w")); off <- c(read("off"))
fam <- get(a[2],asNamespace("stats"))(link=a[3]); fam <- mgcv:::fix.family(fam)
fam <- mgcv:::fix.family.link(fam); fam <- mgcv:::fix.family.var(fam)
fam <- mgcv:::fix.family.ls(fam); family <- fam; nobs <- length(y); weights <- w
# get.null.coef normalizes response first and uses an unweighted projection.
eval(fam$initialize)
null <- qr.coef(qr(X),rep(fam$linkfun(mean(y)),length(y)))
null[is.na(null)] <- 0
known <- a[2] %in% c("binomial","poisson")
start <- if (file.exists(file.path(d,"start"))) c(read("start")) else NULL
fit <- mgcv:::gam.fit3(x=X,y=y,sp=if (known) numeric() else log(.7),
 Eb=0,UrS=list(),weights=w,offset=off,U1=diag(ncol(X)),Mp=ncol(X),
 family=fam,control=gam.control(epsilon=as.numeric(a[4]),maxit=as.integer(a[5])),deriv=0,
 scale=if (known) 1 else 0,scoreType="REML",null.coef=null,start=start)
writeBin(as.double(c(fit$converged,fit$iter)),file.path(d,"status"),size=8,endian="little")
if (a[6]=="TRUE") stopifnot(fit$converged)
writeBin(as.double(c(fit$coefficients,fit$deviance,
 if (known) 1 else fit$scale.est,fit$trA,fit$REML,null,
 as.vector(tcrossprod(fit$rV)))),file.path(d,"reference"),size=8,endian="little")
"""
    completed = subprocess.run(
        [
            "Rscript",
            "-e",
            script,
            str(tmp_path),
            family.family_name,
            "1/mu^2" if link == "inverse_squared" else link,
            repr(tolerance),
            str(max_iter),
            "TRUE" if require_convergence else "FALSE",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"Pinned source oracle failed: {completed.stderr}"
    p = X.shape[1]
    raw = np.fromfile(tmp_path / "reference", dtype="<f8")
    return (
        raw[: p + 4],
        raw[p + 4 : 2 * p + 4],
        raw[2 * p + 4 :].reshape(p, p, order="F"),
    )


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize("family_class", [Gaussian, Gamma, Poisson, Binomial])
@pytest.mark.parametrize("link", _LINKS)
def test_regular_retained_start_all_32_source_cells(tmp_path, family_class, link):
    family, data, weight, offset, start = _case(family_class, link)
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula("y~x"), source, family=family)
    X = np.column_stack((np.ones(len(data)), data.x))
    reference, null, covariance = _reference(
        tmp_path,
        family,
        link,
        X,
        data.y,
        weight,
        offset,
        start,
    )
    # Root-reviewed four-correction record applies only to these original
    # seeded 83-row fixtures. It does not select production family policy.
    boundary_fixture = (family_class, link) in (
        (Poisson, "identity"),
        (Binomial, "log"),
    )
    fit_tolerance = MODERATE if boundary_fixture else STRICT
    score_tolerance = MODERATE if (family_class, link) == (Binomial, "log") else STRICT
    reference_mu = np.asarray(family.link.inverse(X @ reference[:2] + offset))
    reference_se = np.sqrt(reference[3] * np.einsum("ij,jk,ik->i", X, covariance, X))
    original_inputs = tuple(
        np.array(value, copy=True) for value in (X, data.y, weight, offset, start)
    )
    initial_input_hash = hashlib.sha256(
        b"".join(np.asarray(value, dtype="<f8").tobytes() for value in original_inputs)
    ).hexdigest()
    checks = _AssertCollector()
    for B in (1, 17, 200):
        result = fit_regular_streamed_pirls(
            StreamDesign(prepared, source),
            family,
            np.empty(0),
            maximum_bytes=10000000,
            score_scale=1.0 if family.scale_known else 0.7,
            initial_coefficients=start,
            control=StreamPIRLSControl(
                batch_rows=B, tol=1e-7, max_iter=200, solver_policy="qr"
            ),
        )
        state = result.state
        assert state.converged
        assert result.initial_coefficients_present
        np.testing.assert_array_equal(start, np.array([_ANCHORS[link], 0.08]))
        assert not result.null_coefficients.flags.writeable
        metadata = PreparedFittingMetadata.from_prepared(prepared, family)
        prediction = GAMPredictionResult._from_stream_fit(
            stream_state=state,
            prepared=prepared,
            metadata=metadata,
            family=family,
            formula="y~x",
            method="REML",
            control=FitControl(),
        )
        value = np.r_[
            np.asarray(state.coefficients),
            state.deviance,
            state.scale,
            state.edf,
            prediction.score,
        ]
        if boundary_fixture:
            np.testing.assert_array_equal(
                np.fromfile(tmp_path / "status", dtype="<f8"), [1.0, 4.0]
            )
            assert state.n_iter == 4
        for field, observed, expected, tolerance in (
            ("beta", value[:2], reference[:2], fit_tolerance),
            ("deviance/scale/EDF", value[2:5], reference[2:5], STRICT),
            ("REML", value[5], reference[5], score_tolerance),
            (
                "fitted means",
                family.link.inverse(X @ np.asarray(state.coefficients) + offset),
                reference_mu,
                fit_tolerance,
            ),
        ):
            checks.check(
                f"B{B}/{field}",
                lambda a=observed, b=expected, t=tolerance: np.testing.assert_allclose(
                    a, b, rtol=t.rtol, atol=t.atol
                ),
            )
        checks.check(
            f"B{B}/null",
            lambda r=result: np.testing.assert_allclose(
                r.null_coefficients, null, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        actual_covariance = np.asarray(
            state.fisher_coefficient_factor.hessian_inverse(jnp.eye(prepared.n_coef))
        )
        checks.check(
            f"B{B}/Fisher covariance",
            lambda a=actual_covariance: np.testing.assert_allclose(
                a,
                covariance,
                rtol=fit_tolerance.rtol,
                atol=fit_tolerance.atol,
            ),
        )
        actual_se = np.sqrt(
            float(state.scale) * np.einsum("ij,jk,ik->i", X, actual_covariance, X)
        )
        checks.check(
            f"B{B}/link SE",
            lambda a=actual_se: np.testing.assert_allclose(
                a,
                reference_se,
                rtol=fit_tolerance.rtol,
                atol=fit_tolerance.atol,
            ),
        )
        assert result.source_score.score_phi == (1.0 if family.scale_known else 0.7)
        assert result.source_score.reported_phi == float(state.scale)
        assert result.source_score.penalized_deviance == float(state.penalized_deviance)
        assert (
            result.source_score.stopping_penalized_deviance
            == result.accepted_penalized_history[-1]
        )
        for original, current in zip(
            original_inputs, (X, data.y, weight, offset, start), strict=True
        ):
            np.testing.assert_array_equal(current, original)
        assert (
            hashlib.sha256(
                b"".join(
                    np.asarray(value, dtype="<f8").tobytes()
                    for value in (X, data.y, weight, offset, start)
                )
            ).hexdigest()
            == initial_input_hash
        )
    checks.raise_if_any(f"Retained {family.family_name}/{link} source system")


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize("alias", [False, True])
def test_regular_no_intercept_natural_anchor_and_full_map(tmp_path, alias):
    rng = np.random.default_rng(73401)
    x = np.linspace(0.3, 1.3, 83)
    y = (0.8 * x + 0.25 * x * x) * np.exp(rng.normal(0, 0.1, len(x)))
    data = pd.DataFrame({"x": x, "z": -x if alias else x * x, "y": y})
    weight = 0.4 + rng.uniform(size=len(x))
    weight[5] = 0.0
    offset = 0.01 * np.sin(x)
    family = Gamma("identity")
    formula = "y~0+x+z"
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula(formula), source, family=family)
    dense = ModelSetup.build(parse_formula(formula), data)
    assert prepared.predict_spec.parametric_keep_cols == tuple(
        dense.parametric_keep_cols
    )
    X = dense.X
    reference, null, covariance = _reference(
        tmp_path, family, "identity", X, y, weight, offset
    )
    for B in (1, 17, 200):
        result = fit_regular_streamed_pirls(
            StreamDesign(prepared, source),
            family,
            np.empty(0),
            maximum_bytes=10000000,
            score_scale=0.7,
            control=StreamPIRLSControl(
                batch_rows=B, tol=1e-7, max_iter=200, solver_policy="qr"
            ),
        )
        assert result.state.converged
        assert not result.initial_coefficients_present
        np.testing.assert_allclose(
            result.null_coefficients, null, rtol=STRICT.rtol, atol=STRICT.atol
        )
        metadata = PreparedFittingMetadata.from_prepared(prepared, family)
        prediction = GAMPredictionResult._from_stream_fit(
            stream_state=result.state,
            prepared=prepared,
            metadata=metadata,
            family=family,
            formula=formula,
            method="REML",
            control=FitControl(),
        )
        state = result.state
        np.testing.assert_allclose(
            np.r_[
                np.asarray(state.coefficients),
                state.deviance,
                state.scale,
                state.edf,
                prediction.score,
            ],
            reference,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            state.fisher_coefficient_factor.hessian_inverse(jnp.eye(prepared.n_coef)),
            covariance,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


def test_regular_retained_start_validation_and_anchor_workspace():
    family, data, weight, offset, _ = _case(Gamma, "identity")
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula("y~x"), source, family=family)
    stream = StreamDesign(prepared, source)
    control = StreamPIRLSControl(batch_rows=17, solver_policy="qr")
    ledger = preflight_regular_stream_workspace(stream, control, 10000000)
    assert ledger.null_projection_bytes >= 8 * (4 * 17 * 2 + 8 * 2 * 2)
    for start in (np.ones(3), np.array([1.0, np.nan])):
        with pytest.raises(ValueError, match="fitting p-vector"):
            fit_regular_streamed_pirls(
                stream,
                family,
                np.empty(0),
                maximum_bytes=10000000,
                score_scale=0.7,
                initial_coefficients=start,
                control=control,
            )
    without_anchor = replace(ledger, null_projection_bytes=0).required_bytes
    with pytest.raises(MemoryError, match="known workspace"):
        fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=without_anchor,
            score_scale=0.7,
            control=control,
        )


@pytest.mark.usefixtures("r_bridge")
def test_regular_retained_start_domain_shrink_matches_actual_source(tmp_path):
    family, data, weight, offset, _ = _case(Gamma, "identity")
    start = np.array([-0.25, 0.0])
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula("y~x"), source, family=family)
    X = np.column_stack((np.ones(len(data)), data.x))
    reference, null, covariance = _reference(
        tmp_path, family, "identity", X, data.y, weight, offset, start
    )
    for B in (1, 17, 200):
        result = fit_regular_streamed_pirls(
            StreamDesign(prepared, source),
            family,
            np.empty(0),
            maximum_bytes=10000000,
            score_scale=0.7,
            initial_coefficients=start,
            control=StreamPIRLSControl(
                batch_rows=B, tol=1e-7, max_iter=200, solver_policy="qr"
            ),
        )
        assert result.state.converged
        assert result.initial_coefficients_present
        assert result.initial_shrinks > 0
        np.testing.assert_allclose(
            result.null_coefficients, null, rtol=STRICT.rtol, atol=STRICT.atol
        )
        metadata = PreparedFittingMetadata.from_prepared(prepared, family)
        prediction = GAMPredictionResult._from_stream_fit(
            stream_state=result.state,
            prepared=prepared,
            metadata=metadata,
            family=family,
            formula="y~x",
            method="REML",
            control=FitControl(),
        )
        state = result.state
        np.testing.assert_allclose(
            np.r_[
                np.asarray(state.coefficients),
                state.deviance,
                state.scale,
                state.edf,
                prediction.score,
            ],
            reference,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            state.fisher_coefficient_factor.hessian_inverse(jnp.eye(2)),
            covariance,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


@pytest.mark.usefixtures("r_bridge")
def test_preserved_poisson_identity_tight_control_exhaustion(tmp_path):
    family, data, weight, offset, start = _case(Poisson, "identity")
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula("y~x"), source, family=family)
    X = np.column_stack((np.ones(len(data)), data.x))
    _reference(
        tmp_path,
        family,
        "identity",
        X,
        data.y,
        weight,
        offset,
        start,
        tolerance=1e-12,
        max_iter=100,
        require_convergence=False,
    )
    source_status = np.fromfile(tmp_path / "status", dtype="<f8")
    assert source_status[0] == 0.0
    assert source_status[1] == 100.0
    result = fit_regular_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.empty(0),
        maximum_bytes=10000000,
        score_scale=1.0,
        initial_coefficients=start,
        control=StreamPIRLSControl(
            batch_rows=17, tol=1e-12, max_iter=100, solver_policy="qr"
        ),
    )
    assert not result.state.converged
    assert result.state.n_iter == 100
    assert not result.state.line_search_failed
    assert np.all(np.isfinite(result.state.coefficients))
