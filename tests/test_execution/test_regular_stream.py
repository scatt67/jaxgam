"""Regular source controller: accounting, information consumers and R fits."""

import inspect
import subprocess
import sys
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import jaxgam.execution.regular_stream as regular_controller
from jaxgam.control import FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.regular_stream import (
    _positive_signed_state,
    fit_regular_streamed_pirls,
    preflight_regular_stream_workspace,
    regular_working_scan,
)
from jaxgam.execution.signed_qr import solve_signed_qr
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Gamma
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.fitting.family_execution import (
    FamilyExecutionLineage,
    FamilyExecutionParameters,
)
from jaxgam.fitting.signed_qr import SignedQRCoefficientFactor
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.fitting_prepare import qr_penalty_roots
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.results import GAMPredictionResult
from tests.tolerances import STRICT


class _CountedSource:
    def __init__(self, source):
        self.source = source
        self.scans = 0
        self.batches = 0

    @property
    def n_rows(self):
        return self.source.n_rows

    def fingerprint(self):
        return self.source.fingerprint()

    def scan(self, batch_rows):
        self.scans += 1
        for batch in self.source.scan(batch_rows):
            self.batches += 1
            yield batch


class _EmptyCountedSource(_CountedSource):
    def scan(self, batch_rows):
        self.scans += 1
        for index, batch in enumerate(self.source.scan(batch_rows)):
            if index == 0:
                self.batches += 1
                yield replace(
                    batch,
                    columns={name: value[:0] for name, value in batch.columns.items()},
                    y=batch.y[:0],
                    weight=batch.weight[:0],
                    offset=batch.offset[:0],
                    valid=batch.valid[:0],
                    row_positions=batch.row_positions[:0],
                )
            self.batches += 1
            yield batch


def _fixture(link="identity"):
    rng = np.random.default_rng(731)
    x = np.linspace(-0.6, 0.7, 83)
    y = (2.0 + 0.8 * x) * rng.uniform(0.2, 1.8, len(x))
    w = 0.3 + rng.uniform(size=len(x))
    off = 0.1 * np.sin(2.0 * x)
    data = pd.DataFrame({"x": x, "y": y})
    family = Gamma(link=link)
    source = DataFrameRowSource(data, response="y", weights=w, offset=off)
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    counted = _CountedSource(source)
    return StreamDesign(prepared, counted), family, data, w, off


@pytest.mark.parametrize("batch_rows", [1, 7, 200])
def test_regular_gamma_information_consumers_and_measured_scans(batch_rows):
    stream, family, data, w, off = _fixture()
    result = fit_regular_streamed_pirls(
        stream,
        family,
        np.empty(0),
        maximum_bytes=10000000,
        score_scale=0.7,
        control=StreamPIRLSControl(
            batch_rows=batch_rows, tol=1e-12, solver_policy="qr"
        ),
    )
    state = result.state
    assert state.converged
    assert state.source_scans == stream.source.scans
    assert state.batches_scanned == stream.source.batches
    assert result.final_refit_accepted
    assert isinstance(state.coefficient_factor, SignedQRCoefficientFactor)
    X = np.column_stack((np.ones(len(data)), data.x))
    mu_info = X @ result.information_coefficients + off
    observed = X.T @ (
        (w * (2.0 * data.y.to_numpy() / mu_info - 1.0) / mu_info**2)[:, None] * X
    )
    fisher = X.T @ ((w / mu_info**2)[:, None] * X)
    np.testing.assert_allclose(
        state.coefficient_factor.hessian_inverse(jnp.eye(2)),
        np.linalg.inv(observed),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        state.fisher_coefficient_factor.hessian_inverse(jnp.eye(2)),
        np.linalg.inv(fisher),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert np.linalg.norm(observed - fisher) > 0.1
    metadata = PreparedFittingMetadata.from_prepared(stream.prepared, family)
    prediction = GAMPredictionResult._from_stream_fit(
        stream_state=state,
        prepared=stream.prepared,
        metadata=metadata,
        family=family,
        formula="y ~ x",
        method="REML",
        control=FitControl(uncertainty="fisher"),
    )
    assert np.isfinite(prediction.score)
    invalid = replace(state.coefficient_factor, correction=jnp.zeros(2))
    with pytest.raises(FloatingPointError, match="determinant"):
        GAMPredictionResult._from_stream_fit(
            stream_state=replace(state, coefficient_factor=invalid),
            prepared=stream.prepared,
            metadata=metadata,
            family=family,
            formula="y ~ x",
            method="REML",
            control=FitControl(),
        )


def test_regular_budget_rejection_is_prospective():
    stream, family, *_ = _fixture()
    control = StreamPIRLSControl(batch_rows=17, solver_policy="qr")
    ledger = preflight_regular_stream_workspace(stream, control, 10000000)
    with pytest.raises(MemoryError, match="known workspace"):
        fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=ledger.required_bytes - 1,
            score_scale=0.7,
            control=control,
        )
    assert stream.source.scans == 0
    assert stream.source.batches == 0


def test_regular_iteration_history_is_prospectively_budgeted():
    stream, family, *_ = _fixture()
    short = StreamPIRLSControl(batch_rows=17, max_iter=2, solver_policy="qr")
    long = replace(short, max_iter=1000000)
    short_ledger = preflight_regular_stream_workspace(stream, short, 100000000)
    long_ledger = preflight_regular_stream_workspace(stream, long, 100000000)
    assert long_ledger.history_bytes >= 8 * (long.max_iter + 1)
    assert (
        long_ledger.required_bytes - short_ledger.required_bytes
        == long_ledger.history_bytes - short_ledger.history_bytes
    )
    with pytest.raises(MemoryError, match="known workspace"):
        fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=short_ledger.required_bytes,
            score_scale=0.7,
            control=long,
        )
    assert stream.source.scans == 0


def test_regular_empty_zero_and_repeated_source_blocks_have_measured_global_counts():
    _, family, data, weights, offset = _fixture()
    weights[:7] = 0.0
    positions = np.r_[np.arange(7), np.arange(82, 6, -1), [21, 21, 3]]
    source = DataFrameRowSource(
        data,
        response="y",
        weights=weights,
        offset=offset,
        row_selection=positions,
    )
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    control = StreamPIRLSControl(batch_rows=7, tol=1e-12, solver_policy="qr")
    ordinary = fit_regular_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.empty(0),
        maximum_bytes=10000000,
        score_scale=0.7,
        control=control,
    )
    counted = _EmptyCountedSource(source)
    with_empty = fit_regular_streamed_pirls(
        StreamDesign(prepared, counted),
        family,
        np.empty(0),
        maximum_bytes=10000000,
        score_scale=0.7,
        control=control,
    )
    assert with_empty.state.converged
    assert with_empty.state.source_scans == counted.scans
    assert with_empty.state.batches_scanned == counted.batches
    np.testing.assert_allclose(
        with_empty.state.coefficients,
        ordinary.state.coefficients,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert not with_empty.information_coefficients.flags.writeable


def _scan_inputs(stream, family, batch_rows):
    lineage = FamilyExecutionLineage.from_prepared(stream.prepared, family)
    parameters = FamilyExecutionParameters.from_snapshot(lineage.parameters)
    return (
        lineage,
        parameters,
        StreamPIRLSControl(batch_rows=batch_rows, solver_policy="qr"),
    )


@pytest.mark.parametrize("batch_rows", [1, 7, 200])
def test_regular_working_qr_retains_only_source_good_rows(batch_rows):
    _, family, data, weights, offset = _fixture()
    weights[:7] = 0.0
    source = DataFrameRowSource(data, response="y", weights=weights, offset=offset)
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    stream = StreamDesign(prepared, source)
    lineage, parameters, control = _scan_inputs(stream, family, batch_rows)
    scan = regular_working_scan(
        stream,
        family,
        lineage,
        parameters,
        control,
        lambda batch, X: X @ np.array([2.0, 0.1]) + batch.offset,
        score_system=True,
    )
    good_rows = int(np.count_nonzero(weights > 0.0))
    assert scan.informative_count == good_rows
    assert scan.newton.absolute.n_data_rows == good_rows
    assert scan.fisher.n_data_rows == good_rows
    assert scan.observed.absolute.n_data_rows == good_rows
    # Source lineage and reporting still retain all real observation rows.
    assert stream.prepared.n_obs == len(data)


def test_regular_real_negative_working_system_requests_fisher_recovery():
    stream, family, *_ = _fixture()
    lineage, parameters, control = _scan_inputs(stream, family, 7)
    # A valid Gamma mean far above every response yields genuinely negative
    # observed curvature. Its selected Newton system is allowed to reach the
    # signed factor; the resulting indefiniteness requests step-local Fisher.
    scan = regular_working_scan(
        stream,
        family,
        lineage,
        parameters,
        control,
        lambda _batch, X: np.full(len(X), 20.0),
    )
    assert scan.newton is not None
    assert solve_signed_qr(scan.newton, ()).fisher_required
    recovered = solve_signed_qr(_positive_signed_state(scan), ())
    assert recovered.score_admissible
    assert np.all(np.isfinite(recovered.coefficients))
    np.testing.assert_allclose(
        recovered.coefficients,
        np.linalg.solve(scan.fisher_G, scan.fisher_rhs),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_regular_finite_working_rows_cannot_converge_with_overflowed_statistics():
    data = pd.DataFrame(
        {"x": np.linspace(1000.0, 1200.0, 11), "y": np.full(11, 1e-152)}
    )
    family = Gamma("identity")
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    stream = StreamDesign(prepared, source)
    lineage, parameters, control = _scan_inputs(stream, family, 7)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(FloatingPointError, match="statistics overflow"),
    ):
        regular_working_scan(
            stream,
            family,
            lineage,
            parameters,
            control,
            lambda _batch, X: np.full(len(X), 1e-152),
        )


def test_regular_qr_live_design_allocations_fit_the_workspace_ledger(monkeypatch):
    from scipy import linalg

    rng = np.random.default_rng(735)
    n, p, B = 4096, 20, 2048
    data = pd.DataFrame({f"x{j}": rng.normal(size=n) for j in range(1, p)})
    data["y"] = 2.0
    family = Gamma("identity")
    source = DataFrameRowSource(data, response="y")
    formula = "y ~ " + " + ".join(f"x{j}" for j in range(1, p))
    prepared = prepare_model(parse_formula(formula), source, family=family)
    stream = StreamDesign(prepared, source)
    lineage, parameters, control = _scan_inputs(stream, family, B)
    ledger = preflight_regular_stream_workspace(stream, control, 100000000)
    original = linalg.qr
    measured = []

    def observe_qr(*args, **kwargs):
        result = original(*args, **kwargs)
        candidates = [args[0], result[0][0], result[1]]
        frame = inspect.currentframe().f_back
        while frame is not None and "jaxgam/" in frame.f_code.co_filename:
            candidates.extend(
                value
                for value in frame.f_locals.values()
                if isinstance(value, np.ndarray)
                and value.ndim == 2
                and value.shape[1] == p
            )
            frame = frame.f_back
        unique = []
        for value in candidates:
            if not any(np.shares_memory(value, old) for old in unique):
                unique.append(value)
        measured.append(sum(value.nbytes for value in unique))
        return result

    monkeypatch.setattr(linalg, "qr", observe_qr)
    regular_working_scan(
        stream,
        family,
        lineage,
        parameters,
        control,
        lambda _batch, X: np.full(len(X), 2.0),
        score_system=True,
    )
    # This observes Python-visible live design buffers at the packed-QR
    # boundary, including the caller's X and the QR's output copy. It is a
    # measured lower bound, not a claim to measure opaque LAPACK/XLA scratch.
    assert max(measured) >= 4 * B * p * 8
    assert max(measured) <= ledger.host_batch_bytes + ledger.host_coefficient_bytes


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_regular_gamma_fixed_trial_scale_matches_pinned_gam_fit3(tmp_path, link):
    stream, family, data, w, off = _fixture(link)
    X = np.column_stack((np.ones(len(data)), data.x))
    for name, array in (("X", X), ("y", data.y), ("w", w), ("off", off)):
        np.savetxt(tmp_path / name, array, fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
d <- commandArgs(TRUE)[1]
read <- function(name) as.matrix(read.table(file.path(d,name)))
X <- read("X"); y <- c(read("y")); w <- c(read("w")); off <- c(read("off"))
fam <- mgcv:::fix.family.link(Gamma(link=commandArgs(TRUE)[2]))
fam <- mgcv:::fix.family.var(fam); fam <- mgcv:::fix.family.ls(fam)
null <- c(fam$linkfun(mean(y)),0)
fit <- mgcv:::gam.fit3(x=X,y=y,sp=log(.7),Eb=0,UrS=list(),weights=w,
 offset=off,U1=diag(2),Mp=2,family=fam,control=gam.control(epsilon=1e-12),
 deriv=0,scale=0,scoreType="REML",null.coef=null)
options(digits=17)
write.table(c(fit$coefficients,fit$deviance,fit$scale.est,fit$trA,fit$REML),
 file.path(d,"reference"),row.names=FALSE,col.names=FALSE)
"""
    subprocess.run(
        ["Rscript", "-e", script, str(tmp_path), link],
        capture_output=True,
        text=True,
        check=True,
    )
    reference = np.loadtxt(tmp_path / "reference")
    for batch_rows in (1, 7, 200):
        result = fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=10000000,
            score_scale=0.7,
            control=StreamPIRLSControl(
                batch_rows=batch_rows, tol=1e-12, solver_policy="qr"
            ),
        )
        state = result.state
        metadata = PreparedFittingMetadata.from_prepared(stream.prepared, family)
        prediction = GAMPredictionResult._from_stream_fit(
            stream_state=state,
            prepared=stream.prepared,
            metadata=metadata,
            family=family,
            formula="y ~ x",
            method="REML",
            control=FitControl(),
        )
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


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize(
    "controlled_invalid_refit",
    [False, True],
    ids=["source", "controlled_source_fallback"],
)
def test_regular_penalized_gamma_identity_matches_pinned_signed_score_and_fisher(
    tmp_path, monkeypatch, controlled_invalid_refit
):
    rng = np.random.default_rng(737)
    x = np.linspace(-1.0, 1.0, 127)
    y = (3.0 + 0.8 * np.cos(3.0 * x) + 0.3 * x) * rng.gamma(6.0, 1.0 / 6.0, len(x))
    weight = 0.4 + rng.uniform(size=len(x))
    offset = 0.05 * np.sin(2.0 * x)
    data = pd.DataFrame({"x": x, "y": y})
    family = Gamma("identity")
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    formula = 'y ~ s(x, bs="cr", k=7)'
    prepared = prepare_model(parse_formula(formula), source, family=family)
    stream = StreamDesign(prepared, source)
    batch = next(source.scan(len(data)))
    X = prepared.evaluate_fitting_batch(batch)
    layout = qr_penalty_roots(prepared.fitting.penalty_structure)
    assert len(layout) == 1
    root = layout[0]
    E = np.zeros((len(root.root), prepared.n_coef))
    E[:, root.start : root.stop] = root.root
    rho = np.log(np.array([0.35]))
    for name, array in (("X", X), ("E", E), ("y", y), ("w", weight), ("off", offset)):
        np.savetxt(tmp_path / name, array, fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
d <- commandArgs(TRUE)[1]
read <- function(name) as.matrix(read.table(file.path(d,name)))
X <- read("X"); E <- read("E"); y <- c(read("y"))
w <- c(read("w")); off <- c(read("off"))
q <- ncol(X); rank <- nrow(E)
# gam.fit3 accepts roots in the total penalty's range space, with U1
# putting that space first. Return coefficients and rV to the supplied
# fitting coordinates after the source's internal reparameterization.
U1 <- eigen(crossprod(E),symmetric=TRUE)$vectors
UrS <- list(t(U1[,seq_len(rank),drop=FALSE]) %*% t(E))
fam <- mgcv:::fix.family.link(Gamma(link="identity"))
fam <- mgcv:::fix.family.var(fam); fam <- mgcv:::fix.family.ls(fam)
null <- c(fam$linkfun(mean(y)),rep(0,q-1))
# Clone only the actual source's final C_gdi1 return boundary. The ordinary
# case records its untouched output. The controlled case substitutes an
# internally consistent invalid candidate and its penalty, leaving all
# source iterations, working systems, determinants and Fisher factors real.
diag <- new.env(parent=emptyenv())
controlled <- commandArgs(TRUE)[2]=="TRUE"
walk <- function(e) {
 if (is.call(e) && identical(e[[1]],as.name("<-")) &&
     identical(e[[2]],as.name("oo")) && is.call(e[[3]]) &&
     identical(e[[3]][[1]],as.name(".C")) &&
     identical(e[[3]][[2]],as.name("C_gdi1"))) {
   return(bquote({
     oo <- .(e[[3]])
     diag$raw_deviance <- dev
     diag$stopping_pdev <- pdev
     diag$score_phi <- scale
     if (controlled) {
       bad <- rep(0,ncol(x));bad[1] <- -10;bad[2] <- 3
       oo$beta <- c(crossprod(T,bad))
       oo$conv.tol <- drop(t(oo$beta) %*% St %*% oo$beta)
     }
     candidate_eta <- drop(x %*% oo$beta + offset)
     diag$candidate_valid <- valideta(candidate_eta) && validmu(linkinv(candidate_eta))
     diag$solve_penalty <- oo$conv.tol
     oo
   }))
 }
 if (!is.call(e)) return(e)
 as.call(lapply(as.list(e),walk))
}
source_clone <- mgcv:::gam.fit3;body(source_clone) <- walk(body(source_clone))
environment(source_clone) <- list2env(list(diag=diag,controlled=controlled),
 parent=environment(mgcv:::gam.fit3))
fit <- source_clone(x=X,y=y,sp=c(log(.35),log(.7)),Eb=E,UrS=UrS,
 weights=w,offset=off,U1=U1,Mp=q-rank,family=fam,
 control=gam.control(epsilon=1e-12),deriv=0,scale=0,scoreType="REML",null.coef=null)
stopifnot(fit$converged)
options(digits=17)
write.table(c(fit$coefficients,fit$deviance,fit$scale.est,fit$trA,fit$REML),
 file.path(d,"reference"),row.names=FALSE,col.names=FALSE)
write.table(tcrossprod(fit$rV),file.path(d,"fisher_inverse"),row.names=FALSE,col.names=FALSE)
writeBin(as.double(c(diag$raw_deviance,diag$stopping_pdev,diag$solve_penalty,
 diag$candidate_valid,diag$score_phi,fit$scale.est)),file.path(d,"source_payload"),
 size=8,endian="little")
"""
    subprocess.run(
        [
            "Rscript",
            "-e",
            script,
            str(tmp_path),
            "TRUE" if controlled_invalid_refit else "FALSE",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    reference = np.loadtxt(tmp_path / "reference")
    fisher_inverse = np.loadtxt(tmp_path / "fisher_inverse")
    source_payload = np.fromfile(tmp_path / "source_payload", dtype="<f8")
    assert source_payload[3] == float(not controlled_invalid_refit)
    assert not np.isclose(source_payload[4], source_payload[5])
    if controlled_invalid_refit:
        original_solve = regular_controller.solve_signed_qr
        bad = np.zeros(prepared.n_coef)
        bad[0], bad[1] = -10.0, 3.0

        def controlled_final_return(source, *args, **kwargs):
            solved = original_solve(source, *args, **kwargs)
            caller = sys._getframe(1)
            if caller.f_locals.get("refit_source") is source:
                return replace(solved, coefficients=bad)
            return solved

        monkeypatch.setattr(
            regular_controller, "solve_signed_qr", controlled_final_return
        )
    for B in (1, 17, 200):
        result = fit_regular_streamed_pirls(
            stream,
            family,
            rho,
            maximum_bytes=10000000,
            score_scale=0.7,
            control=StreamPIRLSControl(batch_rows=B, tol=1e-12, solver_policy="qr"),
        )
        state = result.state
        assert state.converged
        assert result.final_refit_accepted == (not controlled_invalid_refit)
        payload = result.source_score
        np.testing.assert_allclose(
            [
                payload.raw_deviance,
                payload.stopping_penalized_deviance,
                payload.solve_penalty,
                float(payload.candidate_valid),
                payload.score_phi,
                payload.reported_phi,
            ],
            source_payload,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        if controlled_invalid_refit:
            returned_penalty = 0.35 * np.sum((E @ np.asarray(state.coefficients)) ** 2)
            assert not np.isclose(payload.solve_penalty, returned_penalty)
        metadata = PreparedFittingMetadata.from_prepared(prepared, family)
        prediction = GAMPredictionResult._from_stream_fit(
            stream_state=state,
            prepared=prepared,
            metadata=metadata,
            family=family,
            formula=formula,
            method="REML",
            control=FitControl(),
        )
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
            fisher_inverse,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
