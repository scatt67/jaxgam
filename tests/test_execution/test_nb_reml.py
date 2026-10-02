"""Exact fixed-state streamed NB REML host score and gradient gates."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.nb_reml import evaluate_nb_stream_reml
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Gamma
from jaxgam.fitting.data import FittingData, PreparedFittingMetadata
from jaxgam.fitting.newton import _diff_score
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.fitting_prepare import qr_penalty_roots
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from tests.helpers import _AssertCollector, r_available
from tests.test_execution.test_nb_stream import _fixture
from tests.tolerances import STRICT

_LINKS = ("log", "identity", "sqrt")
_MAXIMUM_BYTES = 10_000_000
_THETA = 2.7
_RHO = float(np.log(0.35))
_CONTROL = StreamPIRLSControl(
    batch_rows=11,
    solver_policy="qr",
    tol=1e-11,
    max_iter=200,
)


def _case(link: str, *, estimated: bool):
    stream, family = _fixture(
        link,
        _THETA,
        estimated=estimated,
        smooth=True,
        response_theta=_THETA,
    )
    rho = np.full_like(stream.prepared.fitting.log_lambda_init, _RHO)
    params = np.concatenate((rho, [np.log(_THETA)])) if estimated else rho
    return stream, family, params


def _evaluate(link: str, *, estimated: bool):
    stream, family, params = _case(link, estimated=estimated)
    trial = evaluate_nb_stream_reml(
        stream,
        family,
        params,
        maximum_bytes=_MAXIMUM_BYTES,
        control=_CONTROL,
    )
    return stream, family, params, trial


def _pinned_reference(tmp_path, stream, link: str, *, estimated: bool):
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
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
    ):
        np.savetxt(tmp_path / name, value, fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
a <- commandArgs(TRUE); d <- a[1]; link <- a[2]; estimated <- a[3]=="TRUE"
read <- function(name) as.matrix(read.table(file.path(d,name)))
X <- read("X"); E <- read("E"); y <- c(read("y")); wt <- c(read("weight"))
off <- c(read("offset")); theta <- 2.7
fam <- mgcv:::fix.family.link(do.call(mgcv::nb,
 list(theta=if(estimated) -theta else theta,link=link)))
q <- ncol(X); rank <- nrow(E)
U1 <- eigen(crossprod(E),symmetric=TRUE)$vectors
UrS <- list(t(U1[,seq_len(rank),drop=FALSE]) %*% t(E))
null <- qr.coef(qr(X),rep(fam$linkfun(mean(y)),length(y))); null[is.na(null)] <- 0
sp <- if(estimated) c(log(theta),log(.35)) else log(.35)
fit <- mgcv:::gam.fit4(x=X,y=y,sp=sp,Eb=E,UrS=UrS,weights=wt,offset=off,
 U1=U1,Mp=q-rank,family=fam,
 control=mgcv::gam.control(epsilon=1e-11,maxit=200),deriv=1,scale=1,
 scoreType="REML",null.coef=null)
stopifnot(fit$converged)
writeBin(as.double(c(fit$REML,fit$iter,fit$coefficients,fit$REML1)),
 file.path(d,"reference"),size=8,endian="little")
"""
    completed = subprocess.run(
        [
            "Rscript",
            "-e",
            script,
            str(tmp_path),
            link,
            str(estimated).upper(),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    raw = np.fromfile(tmp_path / "reference", dtype="<f8")
    p = stream.prepared.n_coef
    gradient = raw[2 + p :]
    if estimated:
        gradient = gradient[[1, 0]]
    return {
        "score": raw[0],
        "iter": int(raw[1]),
        "beta": raw[2 : 2 + p],
        "gradient": gradient,
    }


def _dense_score_arguments(fitting: FittingData) -> dict[str, object]:
    prefix = fitting.count_prefix_plan
    assert prefix is not None
    return {
        "X": fitting.X,
        "y": fitting.y,
        "wt": fitting.wt,
        "offset": fitting.offset,
        "penalty_structure": fitting.penalty_structure,
        "singleton_eig_constants": fitting.singleton_eig_constants,
        "multi_block_proj_S": fitting.multi_block_proj_S,
        "count_indices": prefix.indices,
        "family": fitting.family,
        "pirls_tol": _CONTROL.tol,
        "joint_theta": True,
        "joint_scale": False,
        "n_lambda": fitting.n_penalties,
        "Mp": fitting.total_penalty_null_dim,
        "singleton_sp_indices": fitting.singleton_sp_indices,
        "singleton_ranks": fitting.singleton_ranks,
        "multi_block_sp_indices": fitting.multi_block_sp_indices,
        "multi_block_ranks": fitting.multi_block_ranks,
        "p": fitting.n_coef,
        "max_y": fitting.max_y,
        "integer_counts": prefix.integer_counts,
    }


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("estimated", [False, True], ids=["fixed", "dynamic"])
@pytest.mark.parametrize("link", _LINKS)
def test_nb_host_score_gradient_and_state_match_pinned_gam_fit4(
    tmp_path, link, estimated
):
    """gam.fit4 REML/REML1 and coefficient state agree at identical params."""
    stream, family, params, trial = _evaluate(link, estimated=estimated)
    expected = _pinned_reference(tmp_path, stream, link, estimated=estimated)
    collector = _AssertCollector()
    for name, actual in (
        ("score", trial.score),
        ("gradient", trial.gradient),
        ("beta", trial.fit_result.state.coefficients),
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
    collector.check(
        "iteration",
        lambda: np.testing.assert_equal(
            trial.fit_result.state.n_iter, expected["iter"]
        ),
    )
    collector.check(
        "parameters",
        lambda: np.testing.assert_array_equal(trial.params, params),
    )
    collector.raise_if_any("NB exact streamed REML")
    np.testing.assert_array_equal(family.get_theta(), [np.log(_THETA)])


@pytest.mark.parametrize("link", _LINKS)
def test_nb_dynamic_host_gradient_matches_five_point_full_refits(link):
    """Both rho and log-theta derivatives include reconverged-beta effects."""
    stream, family, params, trial = _evaluate(link, estimated=True)
    step = 1e-3
    finite_difference = []
    for coordinate in range(len(params)):
        scores = []
        for multiplier in (2.0, 1.0, -1.0, -2.0):
            candidate = params.copy()
            candidate[coordinate] += multiplier * step
            scores.append(
                float(
                    evaluate_nb_stream_reml(
                        stream,
                        family,
                        candidate,
                        maximum_bytes=_MAXIMUM_BYTES,
                        control=_CONTROL,
                    ).score
                )
            )
        finite_difference.append(
            (-scores[0] + 8 * scores[1] - 8 * scores[2] + scores[3]) / (12 * step)
        )
    np.testing.assert_allclose(
        trial.gradient,
        finite_difference,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize("link", _LINKS)
def test_nb_dynamic_host_matches_existing_dense_custom_jvp(link):
    """The streamed adjoint agrees with dense joint-theta score derivatives."""
    stream, family, params, trial = _evaluate(link, estimated=True)
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    data = pd.DataFrame({"x": batch.columns["x"], "y": batch.y})
    setup = ModelSetup.build(
        parse_formula('y ~ s(x, bs="cr", k=6)'),
        data,
        weights=batch.weight,
        offset=batch.offset,
    )
    fitting = FittingData.from_setup(setup, family)
    np.testing.assert_allclose(
        fitting.X,
        stream.prepared.evaluate_fitting_batch(batch),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    dense_score, dense_gradient = jax.value_and_grad(_diff_score)(
        jnp.asarray(params),
        jnp.asarray(trial.fit_result.state.coefficients),
        **_dense_score_arguments(fitting),
    )
    np.testing.assert_allclose(
        trial.score, dense_score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        trial.gradient, dense_gradient, rtol=STRICT.rtol, atol=STRICT.atol
    )


@pytest.mark.parametrize("link", _LINKS)
def test_fixed_theta_reuses_joint_host_without_a_theta_coordinate(link):
    """Fixed theta shares score arithmetic while omitting the free coordinate."""
    _, fixed_family, fixed_params, fixed = _evaluate(link, estimated=False)
    _, dynamic_family, dynamic_params, dynamic = _evaluate(link, estimated=True)
    assert fixed.gradient.shape == fixed_params.shape == (1,)
    assert dynamic.gradient.shape == dynamic_params.shape == (2,)
    np.testing.assert_allclose(
        fixed.score, dynamic.score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        fixed.gradient,
        dynamic.gradient[:1],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        fixed.fit_result.state.coefficients,
        dynamic.fit_result.state.coefficients,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_array_equal(fixed_family.get_theta(), [np.log(_THETA)])
    np.testing.assert_array_equal(dynamic_family.get_theta(), [np.log(_THETA)])


def test_dynamic_theta_isolation_warm_start_and_no_conditional_update():
    """One family evaluates theta trials without mutation or EFS substitution."""
    stream, family, params = _case("identity", estimated=True)
    with patch(
        "jaxgam.execution.nb_stream.conditional_theta_stream",
        side_effect=AssertionError("joint REML must not run conditional theta"),
    ):
        first = evaluate_nb_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=_MAXIMUM_BYTES,
            control=_CONTROL,
        )
        changed_params = params.copy()
        changed_params[-1] = np.log(1.3)
        changed = evaluate_nb_stream_reml(
            stream,
            family,
            changed_params,
            maximum_bytes=_MAXIMUM_BYTES,
            control=_CONTROL,
            warm_start=first,
        )
        repeated = evaluate_nb_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=_MAXIMUM_BYTES,
            control=_CONTROL,
            warm_start=changed,
        )
    assert first.theta_free
    assert changed.theta_free
    assert repeated.theta_free
    assert not np.array_equal(first.gradient, changed.gradient)
    np.testing.assert_allclose(
        repeated.score, first.score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        repeated.gradient, first.gradient, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_array_equal(family.get_theta(), [np.log(_THETA)])


def test_nb_host_rejects_wrong_solver_parameters_family_and_warm_lineage():
    """Invalid host coordinates and stale warm states fail before dispatch."""
    stream, family, params = _case("log", estimated=True)
    with pytest.raises(ValueError, match="solver_policy='qr'"):
        evaluate_nb_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=_MAXIMUM_BYTES,
            control=StreamPIRLSControl(),
        )
    for bad in (params[:-1], np.array([params[0], np.nan]), np.array([params[0], 1e3])):
        with (
            np.errstate(over="ignore"),
            pytest.raises(ValueError, match=r"shape|finite|positive theta"),
        ):
            evaluate_nb_stream_reml(
                stream,
                family,
                bad,
                maximum_bytes=_MAXIMUM_BYTES,
                control=_CONTROL,
            )
    with pytest.raises(TypeError, match="requires NegativeBinomial"):
        evaluate_nb_stream_reml(
            stream,
            Gamma(),
            params,
            maximum_bytes=_MAXIMUM_BYTES,
            control=_CONTROL,
        )

    valid = evaluate_nb_stream_reml(
        stream,
        family,
        params,
        maximum_bytes=_MAXIMUM_BYTES,
        control=_CONTROL,
    )
    stale = replace(valid, link_name="stale.Link")
    with pytest.raises(ValueError, match="compatible converged NB state"):
        evaluate_nb_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=_MAXIMUM_BYTES,
            control=_CONTROL,
            warm_start=stale,
        )


def test_nb_host_preflights_memory_and_reports_complete_scan_counts():
    """Known workspace rejects before fit allocation; successful counts are honest."""
    stream, family, params = _case("sqrt", estimated=True)
    with (
        patch.object(PreparedFittingMetadata, "from_prepared") as metadata,
        patch("jaxgam.execution.nb_reml.fit_nb_streamed_pirls") as fit,
        pytest.raises(MemoryError, match="known workspace"),
    ):
        evaluate_nb_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=1,
            control=_CONTROL,
        )
    metadata.assert_not_called()
    fit.assert_not_called()

    trial = evaluate_nb_stream_reml(
        stream,
        family,
        params,
        maximum_bytes=_MAXIMUM_BYTES,
        control=_CONTROL,
    )
    batches = int(np.ceil(stream.prepared.n_obs / _CONTROL.batch_rows))
    assert trial.workspace.required_bytes <= _MAXIMUM_BYTES
    assert trial.source_scans == trial.fit_result.state.source_scans + 2
    assert trial.batches_scanned == (
        trial.fit_result.state.batches_scanned + 2 * batches
    )
    assert trial.source_factor_residual < 1e-12
    assert trial.observed_factor_residual < 1e-12
    assert not trial.fit_result.source_solve_coefficients.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        trial.fit_result.source_solve_coefficients[0] = 0.0


def test_large_count_prefix_is_charged_before_fit_or_derivative_allocation():
    """The differentiated recurrence table cannot bypass the host budget."""
    original, _, _ = _case("log", estimated=True)
    batch = next(original.source.source.scan(original.prepared.n_obs))
    y = np.array(batch.y, copy=True)
    y[0] = 100_000.0
    family = NegativeBinomial(theta=_THETA, fixed=False, link="log")
    source = DataFrameRowSource(
        pd.DataFrame({"x": batch.columns["x"], "y": y}),
        response="y",
        weights=batch.weight,
        offset=batch.offset,
    )
    stream = StreamDesign(
        prepare_model(parse_formula('y ~ s(x, bs="cr", k=6)'), source, family=family),
        source,
    )
    params = np.concatenate(
        (
            np.full_like(stream.prepared.fitting.log_lambda_init, _RHO),
            [np.log(_THETA)],
        )
    )
    with (
        patch("jaxgam.execution.nb_reml.fit_nb_streamed_pirls") as fit,
        pytest.raises(MemoryError, match="count-prefix workspace"),
    ):
        evaluate_nb_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=2_000_000,
            control=_CONTROL,
        )
    fit.assert_not_called()
