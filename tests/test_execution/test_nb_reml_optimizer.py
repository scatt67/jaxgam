"""Joint exact streamed REML optimizer gates for Negative Binomial."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import jaxgam.execution.nb_reml as nb_reml_execution
import jaxgam.execution.reml as reml_execution
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.nb_reml import (
    evaluate_nb_stream_reml,
    optimize_nb_stream_reml,
)
from jaxgam.execution.reml import StreamREMLControl
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.newton import _diff_score
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.fitting_prepare import qr_penalty_roots
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from tests.helpers import _AssertCollector, r_available
from tests.test_execution.test_nb_reml import (
    _CONTROL,
    _LINKS,
    _MAXIMUM_BYTES,
    _THETA,
    _case,
    _dense_score_arguments,
)
from tests.test_execution.test_regular_reml import (
    _CountedSource,
    _retained_numeric_bytes,
)
from tests.tolerances import MODERATE, STRICT

_OPTIMIZER_CONTROL = StreamREMLControl(
    max_iter=50,
    maxfun=180,
    gtol=1e-6,
    ftol=1e-14,
    history_size=12,
)
_OPTIMIZER_PIRLS_CONTROL = type(_CONTROL)(
    batch_rows=17,
    solver_policy="qr",
    tol=1e-11,
    max_iter=200,
)
_OPTIMIZER_MAXIMUM_BYTES = 64_000_000
_OPTIMIZER_REVIEW_SHA256 = (
    "0528a07a8c1b12d4ae628b8063cabd1a27d656aac42bff53a580c4e78327baba"
)
_OPTIMIZER_INPUT_SHA256 = (
    "6f78b9ac6c859994b5eb9fcc0a3a172b1810f77472e2d9021e9dc1cabac4f558"
)
_OPTIMIZER_CSV_SHA256 = (
    "525561fc0ac0702e3d95574ab8edb8252b396a69d9abde0f2e025b704f9ccd58"
)
_OPTIMIZER_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "pr82_nb_optimizer_seed8803.csv"
)
_OPTIMIZER_MODEL_METADATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "pr82_nb_optimizer_model_metadata.npz"
)
_OPTIMIZER_MODEL_METADATA_SHA256 = (
    "103185a4abf21ec23848ed8a1468b840874243e4ad87fa6edf96b00a1f55ff0f"
)
_SELECTED_GATE_DIGESTS = {
    ("log", False): "5817af18b2bd67b9fd1869c65db36d2d3d316a161b43669edf1a34b7663872a5",
    ("log", True): "0d0c601e5be809ba9e0705d06fc14abd973dea9ffa488905e7988c7e0fc105e5",
    (
        "identity",
        False,
    ): "c014ff97560384d87f58cd49f70e1f896e3f5f07829e7d249f387fae6713f336",
    (
        "identity",
        True,
    ): "d726eec69ff706a51b50ccb2961f2e7ea08ccfcb1b0d79c9899367675ffab057",
    ("sqrt", False): "2c52cd9c4b59b24ea21517975863470d6833b1cc0d37bd91640e3b1fdd0a537f",
    ("sqrt", True): "0d1a730dce4274a258af0e837564989a66143c7a1b99742a7b8c0c3f92b552fb",
}


def _optimizer_case(link: str, *, estimated: bool, smooth: bool = True):
    assert hashlib.sha256(_OPTIMIZER_FIXTURE_PATH.read_bytes()).hexdigest() == (
        _OPTIMIZER_CSV_SHA256
    )
    saved = pd.read_csv(_OPTIMIZER_FIXTURE_PATH, float_precision="round_trip")
    x = saved.x.to_numpy()
    y = saved.y.to_numpy(dtype=np.float64)
    weight = saved.weight.to_numpy()
    offset = saved.offset.to_numpy()
    family = NegativeBinomial(
        theta=_THETA,
        fixed=not estimated,
        link=link,
    )
    source = DataFrameRowSource(
        pd.DataFrame({"x": x, "y": y}),
        response="y",
        weights=weight,
        offset=offset,
    )
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=8)' if smooth else "y ~ x"),
        source,
        family=family,
    )
    params = np.r_[
        np.full(prepared.fitting.penalty_structure.n_penalties, np.log(0.5)),
        [np.log(_THETA)] if estimated else [],
    ]
    return StreamDesign(prepared, _CountedSource(source)), family, params


def _optimizer_gate_digest(stream, family, params) -> str:
    source = getattr(stream.source, "source", stream.source)
    batch = next(source.scan(stream.prepared.n_obs))
    raw = hashlib.sha256()
    for name, value in (
        ("x", batch.columns["x"]),
        ("y", batch.y),
        ("weight", batch.weight),
        ("offset", batch.offset),
    ):
        array = np.asarray(value, dtype="<f8")
        raw.update(name.encode())
        raw.update(array.tobytes())
    assert raw.hexdigest() == _OPTIMIZER_INPUT_SHA256
    smooth_blocks = tuple(
        block
        for block in stream.prepared.predict_spec.coef_map.terms
        if block.smooth is not None
    )
    assert len(smooth_blocks) == 1
    smooth = smooth_blocks[0].smooth.spec
    assert len(smooth.variables) == 1
    assert smooth.by is None
    assert not smooth.extra_args
    formula = (
        f"{stream.prepared.response} ~ s({smooth.variables[0]}, "
        f'bs="{smooth.bs}", k={smooth.k})'
    )
    review = raw.copy()
    review.update(formula.encode())
    review.update(float(_THETA).hex().encode())
    if formula == 'y ~ s(x, bs="cr", k=8)':
        assert review.hexdigest() == _OPTIMIZER_REVIEW_SHA256
    penalty = stream.prepared.fitting.penalty_structure
    assert hashlib.sha256(_OPTIMIZER_MODEL_METADATA_PATH.read_bytes()).hexdigest() == (
        _OPTIMIZER_MODEL_METADATA_SHA256
    )
    assert len(penalty.blocks) == 1
    assert len(penalty.blocks[0].local_penalties) == 1
    with np.load(_OPTIMIZER_MODEL_METADATA_PATH, allow_pickle=False) as metadata:
        np.testing.assert_allclose(
            penalty.blocks[0].local_penalties[0].matrix,
            metadata["penalty"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        actual_transform = penalty.blocks[0].transform.matrix.T
        expected_transform = metadata["transform"].T
        np.testing.assert_allclose(
            actual_transform[:, :, None] * actual_transform[:, None, :],
            expected_transform[:, :, None] * expected_transform[:, None, :],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    penalty_coordinates = [
        {
            "local_penalties": [
                {
                    "shape": item.matrix.shape,
                    "type": type(item).__qualname__,
                }
                for item in block.local_penalties
            ],
            "ranks": block.ranks,
            "sp_indices": block.sp_indices,
            "start": block.start,
            "stop": block.stop,
            "transform_shape": block.transform.matrix.shape,
            "transform_type": type(block.transform).__qualname__,
        }
        for block in penalty.blocks
    ]
    payload = {
        "basis_fingerprint": stream.prepared.basis_fingerprint,
        "family": family.family_name,
        "family_class": type(family).__qualname__,
        "family_n_theta": family.n_theta,
        "family_theta_hex": [
            float(value).hex() for value in family.get_theta(transformed=False)
        ],
        "formula": formula,
        "initial_params_hex": [float(value).hex() for value in params],
        "link": type(family.link).__qualname__,
        "maximum_bytes": _OPTIMIZER_MAXIMUM_BYTES,
        "model_metadata_sha256": _OPTIMIZER_MODEL_METADATA_SHA256,
        "n_coef": stream.prepared.n_coef,
        "n_obs": stream.prepared.n_obs,
        "optimizer_control": asdict(_OPTIMIZER_CONTROL),
        "penalty_coordinates": penalty_coordinates,
        "pirls_control": asdict(_OPTIMIZER_PIRLS_CONTROL),
        "r_controls": {
            "epsilon": 1e-11,
            "max_iter": 200,
            "newton_conv_tol": 1e-6,
            "newton_max_half": 30,
            "newton_max_n_step": 5,
            "newton_max_s_step": 2,
        },
        "review_fixture_sha256": review.hexdigest(),
        "source_fingerprint": stream.prepared.source_fingerprint,
        "theta_hex": float(_THETA).hex(),
    }
    digest = hashlib.sha256()
    digest.update(raw.digest())
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _optimize(link: str, *, estimated: bool, pin_lambda: bool = False):
    stream, family, params = _optimizer_case(link, estimated=estimated)
    if pin_lambda:
        params = params.copy()
        params[0] = -41.25
    result = optimize_nb_stream_reml(
        stream,
        family,
        params,
        maximum_bytes=_OPTIMIZER_MAXIMUM_BYTES,
        pin_lambda=pin_lambda,
        pirls_control=_OPTIMIZER_PIRLS_CONTROL,
        control=_OPTIMIZER_CONTROL,
    )
    return stream, family, params, result


def _dense_score(stream, family, trial, link):
    source = getattr(stream.source, "source", stream.source)
    batch = next(source.scan(stream.prepared.n_obs))
    data = pd.DataFrame({"x": batch.columns["x"], "y": batch.y})
    setup = ModelSetup.build(
        parse_formula('y ~ s(x, bs="cr", k=8)'),
        data,
        weights=batch.weight,
        offset=batch.offset,
    )
    fitting = FittingData.from_setup(setup, family)
    arguments = _dense_score_arguments(fitting)
    params = np.asarray(trial.params)
    if not family.n_theta:
        dynamic_family = NegativeBinomial(
            theta=_THETA,
            fixed=False,
            link=link,
        )
        fitting = FittingData.from_setup(setup, dynamic_family)
        arguments = _dense_score_arguments(fitting)
        params = np.r_[params, np.log(_THETA)]
    dense_score, dense_gradient = jax.value_and_grad(_diff_score)(
        jnp.asarray(params),
        jnp.asarray(trial.fit_result.state.coefficients),
        **arguments,
    )
    if not family.n_theta:
        dense_gradient = dense_gradient[: len(trial.params)]
    return dense_score, dense_gradient


def _pinned_selected_reference(
    tmp_path,
    stream,
    link,
    params,
    *,
    estimated,
    epsilon=1e-11,
    max_iter=200,
):
    source = getattr(stream.source, "source", stream.source)
    batch = next(source.scan(stream.prepared.n_obs))
    X = stream.prepared.evaluate_fitting_batch(batch)
    roots = qr_penalty_roots(stream.prepared.fitting.penalty_structure)
    E = np.zeros((len(roots[0].root), stream.prepared.n_coef))
    root = roots[0]
    E[:, root.start : root.stop] = root.root
    for name, value in (
        ("X", X),
        ("E", E),
        ("y", batch.y),
        ("weight", batch.weight),
        ("offset", batch.offset),
        ("params", params),
    ):
        np.savetxt(tmp_path / name, value, fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
a <- commandArgs(TRUE); d <- a[1]; link <- a[2]; estimated <- a[3]=="TRUE"
epsilon <- as.double(a[4]); maxit <- as.integer(a[5])
read <- function(name) as.matrix(read.table(file.path(d,name)))
X <- read("X"); E <- read("E"); y <- c(read("y")); wt <- c(read("weight"))
off <- c(read("offset")); params <- c(read("params")); theta <- 2.7
fam <- mgcv:::fix.family.link(do.call(mgcv::nb,
 list(theta=if(estimated) -exp(params[length(params)]) else theta,link=link)))
q <- ncol(X); rank <- nrow(E)
U1 <- eigen(crossprod(E),symmetric=TRUE)$vectors
UrS <- list(t(U1[,seq_len(rank),drop=FALSE]) %*% t(E))
null <- qr.coef(qr(X),rep(fam$linkfun(mean(y)),length(y))); null[is.na(null)] <- 0
sp <- if(estimated) c(params[2],params[1]) else params[1]
fit <- mgcv:::gam.fit4(x=X,y=y,sp=sp,Eb=E,UrS=UrS,weights=wt,offset=off,
 U1=U1,Mp=q-rank,family=fam,
 control=mgcv::gam.control(epsilon=epsilon,maxit=maxit),deriv=1,scale=1,
 scoreType="REML",null.coef=null)
stopifnot(fit$converged)
writeBin(as.double(c(fit$REML,fit$iter,fit$deviance,
 fit$coefficients,fit$REML1)),file.path(d,"reference"),size=8,endian="little")
"""
    completed = subprocess.run(
        [
            "Rscript",
            "-e",
            script,
            str(tmp_path),
            link,
            str(estimated).upper(),
            f"{epsilon:.17g}",
            str(max_iter),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    raw = np.fromfile(tmp_path / "reference", dtype="<f8")
    p = stream.prepared.n_coef
    gradient = raw[3 + p :]
    if estimated:
        gradient = gradient[[1, 0]]
    return {
        "score": raw[0],
        "iter": int(raw[1]),
        "deviance": raw[2],
        "beta": raw[3 : 3 + p],
        "gradient": gradient,
    }


def _pinned_outer_reference(
    tmp_path,
    stream,
    link,
    *,
    estimated,
    pinned_rho=None,
):
    source = getattr(stream.source, "source", stream.source)
    batch = next(source.scan(stream.prepared.n_obs))
    for name, value in (
        ("x", batch.columns["x"]),
        ("y", batch.y),
        ("weight", batch.weight),
        ("offset", batch.offset),
    ):
        np.savetxt(tmp_path / name, value, fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
a<-commandArgs(TRUE);d<-a[1];link<-a[2];estimated<-a[3]=="TRUE"
pinned<-a[4]!="NONE";rho<-if(pinned)as.double(a[4])else 0
v<-function(n)scan(file.path(d,n),quiet=TRUE)
dat<-data.frame(x=v("x"),y=v("y"));w<-v("weight");o<-v("offset")
fam<-do.call(nb,list(theta=if(estimated)-2.7 else 2.7,link=link))
fit<-gam(y~s(x,bs="cr",k=8),data=dat,weights=w,offset=o,family=fam,
 sp=if(pinned)exp(rho)else NULL,method="REML",optimizer=c("outer","newton"),
 control=gam.control(epsilon=1e-11,maxit=200,
 newton=list(conv.tol=1e-6,maxNstep=5,maxSstep=2,maxHalf=30)))
outer.full<-identical(fit$outer.info$conv,"full convergence")
rho.out<-if(pinned)rho else log(fit$sp)
writeBin(as.double(c(rho.out,fit$family$getTheta(),fit$gcv.ubre,
 fit$fitted.values,fit$deviance,sum(fit$edf),fit$sig2,fit$converged,
 outer.full,fit$outer.info$iter)),file.path(d,"outer"),size=8,endian="little")
"""
    completed = subprocess.run(
        [
            "Rscript",
            "-e",
            script,
            str(tmp_path),
            link,
            str(estimated).upper(),
            "NONE" if pinned_rho is None else f"{pinned_rho:.17g}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    raw = np.fromfile(tmp_path / "outer", dtype="<f8")
    n = stream.prepared.n_obs
    return {
        "rho": raw[0],
        "log_theta": raw[1],
        "score": raw[2],
        "fitted_values": raw[3 : 3 + n],
        "deviance": raw[-6],
        "edf": raw[-5],
        "scale": raw[-4],
        "inner_converged": bool(raw[-3]),
        "outer_converged": bool(raw[-2]),
        "outer_iterations": int(raw[-1]),
    }


@pytest.mark.parametrize("estimated", [False, True], ids=["fixed", "dynamic"])
@pytest.mark.parametrize("link", _LINKS)
def test_nb_optimizer_selected_trial_stationarity_and_nearby_dense_derivatives(
    link, estimated
):
    stream, family, _params, result = _optimize(link, estimated=estimated)
    optimizer_scans = stream.source.scans
    selected = np.asarray(result.trial.params)
    repeated = evaluate_nb_stream_reml(
        stream,
        family,
        selected,
        maximum_bytes=_MAXIMUM_BYTES,
        control=_OPTIMIZER_PIRLS_CONTROL,
    )
    dense_score, dense_gradient = _dense_score(stream, family, result.trial, link)

    assert result.converged
    assert result.status == 0
    assert result.n_accepted == result.n_iter
    assert result.n_evaluations >= result.n_accepted + 1
    assert result.cumulative_source_scans == optimizer_scans
    batches_per_scan = int(
        np.ceil(stream.prepared.n_obs / _OPTIMIZER_PIRLS_CONTROL.batch_rows)
    )
    assert result.cumulative_batches_scanned == optimizer_scans * batches_per_scan
    assert result.cumulative_source_scans >= result.trial.source_scans
    assert result.projected_gradient_inf <= _OPTIMIZER_CONTROL.gtol
    assert result.required_bytes < 64_000_000
    assert np.all(np.diff(result.accepted_score_history) <= STRICT.atol)
    np.testing.assert_allclose(
        result.trial.score, repeated.score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.trial.gradient,
        repeated.gradient,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.trial.fit_result.state.coefficients,
        repeated.fit_result.state.coefficients,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.trial.score, dense_score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    # At the stationary point, backend reduction residuals of at most 4.7e-11
    # can dominate components whose true magnitude is only O(1e-7). Preserve
    # the truthful stationarity checks above, and separately prove the actual
    # dense/streamed JVP at a fixed nonstationary point in the selected
    # coordinate neighborhood.
    assert np.max(np.abs(dense_gradient)) <= _OPTIMIZER_CONTROL.gtol
    nearby_params = selected.copy()
    nearby_params[0] += 0.1
    if estimated:
        nearby_params[-1] += 0.5
    nearby = evaluate_nb_stream_reml(
        stream,
        family,
        nearby_params,
        maximum_bytes=_MAXIMUM_BYTES,
        control=_OPTIMIZER_PIRLS_CONTROL,
        warm_start=result.trial,
    )
    nearby_dense_score, nearby_dense_gradient = _dense_score(
        stream,
        family,
        nearby,
        link,
    )
    np.testing.assert_allclose(
        nearby.score,
        nearby_dense_score,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        nearby.gradient,
        nearby_dense_gradient,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    if estimated:
        np.testing.assert_array_equal(
            result.trial.fit_result.log_theta,
            selected[-1:],
        )
    np.testing.assert_array_equal(family.get_theta(), [np.log(_THETA)])
    retained, history, lbfgs = reml_execution._parameterized_optimizer_workspace_bytes(
        stream.prepared.n_coef,
        len(selected),
        _OPTIMIZER_CONTROL,
        _OPTIMIZER_PIRLS_CONTROL,
    )
    assert result.outer_workspace_bytes == retained + history + lbfgs
    assert 0 < _retained_numeric_bytes(result.trial) <= retained


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("estimated", [False, True], ids=["fixed", "dynamic"])
@pytest.mark.parametrize("link", _LINKS)
def test_nb_optimizer_selected_state_matches_pinned_gam_fit4(tmp_path, link, estimated):
    stream, _family, _params, result = _optimize(link, estimated=estimated)
    expected = _pinned_selected_reference(
        tmp_path,
        stream,
        link,
        np.asarray(result.trial.params),
        estimated=estimated,
    )
    collector = _AssertCollector()
    for name, actual in (
        ("score", result.trial.score),
        ("gradient", result.trial.gradient),
        ("beta", result.trial.fit_result.state.coefficients),
        ("deviance", result.trial.fit_result.state.deviance),
    ):
        collector.check(
            name,
            lambda name=name, actual=actual: np.testing.assert_allclose(
                actual,
                expected[name],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
    collector.raise_if_any("NB selected exact source state")


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("estimated", [False, True], ids=["fixed", "dynamic"])
@pytest.mark.parametrize("link", _LINKS)
def test_nb_optimizer_independently_selected_fit_matches_pinned_outer_source(
    tmp_path, link, estimated
):
    """The reviewed six-cell outer fit retains STRICT score agreement."""
    stream, family, initial, result = _optimize(link, estimated=estimated)
    assert (
        _optimizer_gate_digest(stream, family, initial)
        == _SELECTED_GATE_DIGESTS[(link, estimated)]
    )
    expected = _pinned_outer_reference(
        tmp_path,
        stream,
        link,
        estimated=estimated,
    )
    source = getattr(stream.source, "source", stream.source)
    batch = next(source.scan(stream.prepared.n_obs))
    X = stream.prepared.evaluate_fitting_batch(batch)
    state = result.trial.fit_result.state
    fitted = np.asarray(
        family.link.inverse(X @ np.asarray(state.coefficients) + batch.offset)
    )

    assert result.converged
    assert expected["inner_converged"]
    assert expected["outer_converged"]
    assert expected["outer_iterations"] > 0
    for value in (
        result.trial.params,
        result.trial.gradient,
        result.trial.score,
        state.coefficients,
        fitted,
        state.deviance,
        state.edf,
        state.scale,
        expected["rho"],
        expected["log_theta"],
        expected["score"],
        expected["fitted_values"],
        expected["deviance"],
        expected["edf"],
        expected["scale"],
    ):
        assert np.all(np.isfinite(np.asarray(value)))

    selected = np.asarray(result.trial.params)
    np.testing.assert_allclose(
        selected[0], expected["rho"], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    if estimated:
        np.testing.assert_allclose(
            selected[-1],
            expected["log_theta"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        np.testing.assert_array_equal(result.trial.fit_result.log_theta, selected[-1:])
    else:
        np.testing.assert_allclose(
            expected["log_theta"],
            np.log(_THETA),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            result.trial.fit_result.log_theta,
            [np.log(_THETA)],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    np.testing.assert_allclose(
        result.trial.score,
        expected["score"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    for actual, reference in (
        (fitted, expected["fitted_values"]),
        (state.deviance, expected["deviance"]),
        (state.edf, expected["edf"]),
    ):
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
    np.testing.assert_allclose(
        state.scale,
        expected["scale"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_selected_fit_moderate_gate_binds_input_model_start_and_controls(
    monkeypatch,
):
    stream, family, initial = _optimizer_case("log", estimated=True)
    expected = _SELECTED_GATE_DIGESTS[("log", True)]
    assert _optimizer_gate_digest(stream, family, initial) == expected

    changed_start = initial.copy()
    changed_start[0] = np.nextafter(changed_start[0], np.inf)
    assert _optimizer_gate_digest(stream, family, changed_start) != expected

    with monkeypatch.context() as control_patch:
        control_patch.setattr(
            __name__ + "._OPTIMIZER_CONTROL",
            replace(_OPTIMIZER_CONTROL, ftol=1e-13),
        )
        assert _optimizer_gate_digest(stream, family, initial) != expected

    source = getattr(stream.source, "source", stream.source)
    batch = next(source.scan(stream.prepared.n_obs))
    changed_x = np.asarray(batch.columns["x"]).copy()
    changed_x[0] = np.nextafter(changed_x[0], np.inf)
    changed_source = DataFrameRowSource(
        pd.DataFrame({"x": changed_x, "y": batch.y}),
        response="y",
        weights=batch.weight,
        offset=batch.offset,
    )
    changed_stream = StreamDesign(
        prepare_model(
            parse_formula('y ~ s(x, bs="cr", k=8)'),
            changed_source,
            family=family,
        ),
        changed_source,
    )
    with pytest.raises(AssertionError):
        _optimizer_gate_digest(changed_stream, family, initial)

    changed_basis_source = DataFrameRowSource(
        pd.DataFrame({"x": batch.columns["x"], "y": batch.y}),
        response="y",
        weights=batch.weight,
        offset=batch.offset,
    )
    changed_basis_stream = StreamDesign(
        prepare_model(
            parse_formula('y ~ s(x, bs="cs", k=8)'),
            changed_basis_source,
            family=family,
        ),
        changed_basis_source,
    )
    with pytest.raises(AssertionError):
        _optimizer_gate_digest(changed_basis_stream, family, initial)


@pytest.mark.parametrize("estimated", [False, True], ids=["fixed", "dynamic"])
@pytest.mark.parametrize("link", _LINKS)
def test_nb_optimizer_pins_supplied_rho_outside_estimated_bounds(link, estimated):
    stream, family, supplied, result = _optimize(
        link,
        estimated=estimated,
        pin_lambda=True,
    )
    assert result.converged
    assert np.asarray(result.trial.params)[0] == supplied[0] == -41.25
    if estimated:
        assert result.n_iter > 0
        assert result.n_evaluations > 1
        assert result.projected_gradient_inf <= _OPTIMIZER_CONTROL.gtol
    else:
        reference = evaluate_nb_stream_reml(
            stream,
            family,
            supplied,
            maximum_bytes=_MAXIMUM_BYTES,
            control=_OPTIMIZER_PIRLS_CONTROL,
        )
        assert result.n_iter == result.n_accepted == 0
        assert result.n_evaluations == 1
        assert result.projected_gradient_inf == 0.0
        assert result.status == 0
        assert "exact evaluation" in result.message
        np.testing.assert_array_equal(result.trial.params, reference.params)
        np.testing.assert_allclose(
            result.trial.score,
            reference.score,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            result.trial.fit_result.state.coefficients,
            reference.fit_result.state.coefficients,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", _LINKS)
def test_nb_optimizer_dynamic_pinned_rho_matches_tight_pinned_source_state(
    tmp_path, link
):
    stream, _family, supplied, result = _optimize(
        link,
        estimated=True,
        pin_lambda=True,
    )
    expected = _pinned_selected_reference(
        tmp_path,
        stream,
        link,
        np.asarray(result.trial.params),
        estimated=True,
        epsilon=1e-12,
        max_iter=400,
    )
    state = result.trial.fit_result.state
    assert result.converged
    np.testing.assert_array_equal(np.asarray(result.trial.params)[0], supplied[0])
    for actual, reference in (
        (result.trial.score, expected["score"]),
        (result.trial.gradient, expected["gradient"]),
        (state.coefficients, expected["beta"]),
        (state.deviance, expected["deviance"]),
    ):
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


@pytest.mark.parametrize("estimated", [False, True], ids=["fixed", "dynamic"])
@pytest.mark.parametrize("link", _LINKS)
def test_nb_optimizer_supports_parametric_theta_only_and_zero_coordinate_cases(
    link, estimated
):
    stream, family, params = _optimizer_case(link, estimated=estimated, smooth=False)
    result = optimize_nb_stream_reml(
        stream,
        family,
        params,
        maximum_bytes=64_000_000,
        pirls_control=_OPTIMIZER_PIRLS_CONTROL,
        control=_OPTIMIZER_CONTROL,
    )
    assert result.converged
    assert np.asarray(result.trial.params).shape == ((1,) if estimated else (0,))
    if estimated:
        assert result.n_iter > 0
        assert result.projected_gradient_inf <= _OPTIMIZER_CONTROL.gtol
        np.testing.assert_array_equal(
            result.trial.fit_result.log_theta,
            result.trial.params,
        )
    else:
        assert result.n_iter == result.n_accepted == 0
        assert result.n_evaluations == 1
        assert result.projected_gradient_inf == 0.0
        assert "exact evaluation" in result.message


def test_nb_optimizer_preflights_retention_and_keeps_accepted_on_failed_candidate(
    monkeypatch,
):
    stream, family, params = _case("log", estimated=True)
    with (
        patch.object(nb_reml_execution, "evaluate_nb_stream_reml") as evaluate,
        pytest.raises(MemoryError, match="retention"),
    ):
        optimize_nb_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=1,
            pirls_control=_CONTROL,
        )
    evaluate.assert_not_called()

    initial = SimpleNamespace(
        params=np.array([0.0, 0.0]),
        score=jnp.array(2.0),
        gradient=jnp.array([1.0, 1.0]),
        source_scans=3,
        batches_scanned=9,
        source_fingerprint="source",
    )
    candidate = SimpleNamespace(
        params=np.array([1.0, 1.0]),
        score=jnp.array(1.0),
        gradient=jnp.array([0.5, 0.5]),
        source_scans=5,
        batches_scanned=15,
        source_fingerprint="source",
    )
    calls = iter((initial, candidate))
    monkeypatch.setattr(
        nb_reml_execution,
        "evaluate_nb_stream_reml",
        lambda *_args, **_kwargs: next(calls),
    )
    objective = nb_reml_execution._AcceptedNBTrialObjective(
        SimpleNamespace(source=SimpleNamespace(fingerprint=lambda: "source")),
        NegativeBinomial(theta=2.7),
        np.array([0.0, 0.0]),
        _CONTROL,
        _OPTIMIZER_CONTROL,
        1_000_000,
        None,
        None,
    )

    def failed_line_search(objective, params, **_kwargs):
        objective(np.array([1.0, 1.0]))
        return SimpleNamespace(
            x=np.asarray(params),
            success=False,
            message="line search failed",
            status=2,
            nit=0,
        )

    monkeypatch.setattr(reml_execution, "minimize", failed_line_search)
    optimized = reml_execution._run_parameterized_stream_reml(
        objective,
        np.array([0.0, 0.0]),
        [(-40.0, 40.0), (None, None)],
        _OPTIMIZER_CONTROL,
    )
    assert optimized.trial is initial
    assert not optimized.converged
    assert optimized.status == 2
    assert optimized.n_evaluations == 2
    assert optimized.n_accepted == 0
    assert optimized.cumulative_source_scans == 8
    assert optimized.cumulative_batches_scanned == 24


@pytest.mark.parametrize("pin_lambda", [0, 1, "yes", None])
def test_nb_optimizer_rejects_non_boolean_pin_lambda_before_trial(
    pin_lambda,
):
    stream, family, params = _case("log", estimated=True)
    with (
        patch.object(nb_reml_execution, "evaluate_nb_stream_reml") as evaluate,
        pytest.raises(ValueError, match="pin_lambda must be bool"),
    ):
        optimize_nb_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=64_000_000,
            pin_lambda=pin_lambda,
            pirls_control=_CONTROL,
        )
    evaluate.assert_not_called()
