"""Fixed/trial-theta host controller, source stopping and bounded scans."""

import json
import logging
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.nb_stream import fit_nb_streamed_pirls, nb_working_scan
from jaxgam.execution.regular_stream import preflight_regular_stream_workspace
from jaxgam.execution.signed_qr import solve_signed_qr
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Gamma
from jaxgam.fitting.family_execution import (
    FamilyExecutionLineage,
    FamilyExecutionParameters,
)
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.fitting_prepare import qr_penalty_roots
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.links.links import ProbitLink
from tests.helpers import _AssertCollector, r_available
from tests.test_execution.test_regular_stream import _CountedSource, _EmptyCountedSource
from tests.tolerances import STRICT


def _fixture(
    link="log", theta=2.7, *, estimated=False, smooth=False, response_theta=None
):
    rng = np.random.default_rng(1201)
    x = np.linspace(-0.6, 0.7, 79)
    mean = 3 + 0.7 * np.cos(3 * x) + 0.4 * x
    response_theta = theta if response_theta is None else response_theta
    y = rng.negative_binomial(
        response_theta, response_theta / (response_theta + mean)
    ).astype(float)
    weight = 0.5 + rng.uniform(size=len(x))
    weight[::13] = 0
    offset = 0.5 + 0.1 * np.sin(2 * x)
    data = pd.DataFrame({"x": x, "y": y})
    family = NegativeBinomial(theta=theta, fixed=not estimated, link=link)
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    formula = 'y ~ s(x, bs="cr", k=6)' if smooth else "y ~ x"
    prepared = prepare_model(parse_formula(formula), source, family=family)
    return StreamDesign(prepared, _CountedSource(source)), family


def _fit(stream, family, batch_rows=11, parameters=None, **kwargs):
    rho = np.full_like(stream.prepared.fitting.log_lambda_init, np.log(0.35))
    return fit_nb_streamed_pirls(
        stream,
        family,
        rho,
        maximum_bytes=10_000_000,
        parameters=parameters,
        control=StreamPIRLSControl(
            batch_rows=batch_rows, solver_policy="qr", tol=1e-11
        ),
        **kwargs,
    )


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_fixed_theta_batch_invariance_and_measured_source_counts(link):
    stream, family = _fixture(link, smooth=True)
    results = []
    for B in (1, 5, 200):
        before = stream.source.scans, stream.source.batches
        result = _fit(stream, family, B)
        state = result.state
        assert state.converged
        assert state.source_scans == stream.source.scans - before[0]
        assert state.batches_scanned == stream.source.batches - before[1]
        assert state.stationarity < 1e-10
        assert result.log_theta == tuple(family.get_theta())
        assert result.integer_counts
        assert result.max_count >= 1
        history = np.asarray(result.accepted_penalized_history)
        tolerance = 10 * (0.1 + np.abs(history[:-1])) * np.sqrt(np.finfo(float).eps)
        assert np.all(np.diff(history) <= tolerance)
        results.append(np.r_[np.asarray(state.coefficients), state.deviance, state.edf])
    for result in results[1:]:
        np.testing.assert_allclose(
            result, results[0], rtol=STRICT.rtol, atol=STRICT.atol
        )


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_trial_theta_isolation_repeated_fixed_equivalence_and_jit(link):
    stream, family = _fixture(link, estimated=True)
    original = family.get_theta().copy()
    outputs = []
    for theta in (0.1, 2.7, 1e6, 0.1):
        params = FamilyExecutionParameters(jnp.asarray([np.log(theta)]))
        result = _fit(stream, family, parameters=params)
        assert result.state.converged
        assert result.log_theta == (np.log(theta),)
        np.testing.assert_array_equal(family.get_theta(), original)
        outputs.append(np.asarray(result.state.coefficients))
    np.testing.assert_array_equal(outputs[0], outputs[-1])
    assert np.max(np.abs(outputs[0] - outputs[1])) > 1e-4
    # All batch primitives compile with trial theta as a dynamic array leaf.
    assert isinstance(result.state.coefficients, jax.Array)
    with pytest.raises(ValueError, match="explicit trial theta"):
        _fit(stream, family)


def test_prospective_budget_and_control_parameter_rejections():
    stream, family = _fixture()
    control = StreamPIRLSControl(batch_rows=11, solver_policy="qr")
    ledger = preflight_regular_stream_workspace(stream, control, 10_000_000)
    with pytest.raises(MemoryError, match="known workspace"):
        fit_nb_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=ledger.required_bytes - 1,
            control=control,
        )
    assert stream.source.scans == 0
    for value in (
        np.empty(0),
        np.array([np.nan]),
        np.array([1000.0]),
        np.array([-1000.0]),
    ):
        with (
            np.errstate(over="ignore"),
            pytest.raises(ValueError, match="trial log_theta"),
        ):
            _fit(
                stream, family, parameters=FamilyExecutionParameters(jnp.asarray(value))
            )
    with pytest.raises(ValueError, match="solver_policy"):
        fit_nb_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=10_000_000,
            control=StreamPIRLSControl(),
        )


def test_empty_batches_are_neutral_and_accounted():
    stream, family = _fixture("identity")
    expected = _fit(stream, family)
    empty = StreamDesign(stream.prepared, _EmptyCountedSource(stream.source.source))
    actual = _fit(empty, family)
    np.testing.assert_allclose(
        actual.state.coefficients,
        expected.state.coefficients,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert actual.state.source_scans == empty.source.scans
    assert actual.state.batches_scanned == empty.source.batches


def test_parameter_family_smoothing_and_lineage_fail_before_cached_dispatch():
    stream, family = _fixture()
    for rho in (np.ones(1), np.array([np.nan])):
        with pytest.raises(ValueError, match="smoothing parameters"):
            fit_nb_streamed_pirls(stream, family, rho, maximum_bytes=10_000_000)
    with pytest.raises(TypeError, match="NegativeBinomial"):
        fit_nb_streamed_pirls(stream, Gamma(), np.empty(0), maximum_bytes=10_000_000)
    family.link = ProbitLink()
    with pytest.raises(NotImplementedError, match="log, identity, sqrt"):
        _fit(stream, family)
    assert stream.source.scans == 0
    stream, family = _fixture()
    lineage = FamilyExecutionLineage.from_prepared(stream.prepared, family)
    params = FamilyExecutionParameters.from_snapshot(lineage.parameters)
    control = StreamPIRLSControl(batch_rows=11, solver_policy="qr")
    with pytest.raises(ValueError, match="one value per source row"):
        nb_working_scan(
            stream, family, lineage, params, control, lambda _batch, _X: np.ones(1)
        )

    def mutate_family(_batch, X):
        family.put_theta(np.log([7.0]))
        return np.ones(len(X))

    with pytest.raises(RuntimeError, match="changed"):
        nb_working_scan(stream, family, lineage, params, control, mutate_family)


def test_invalid_working_domain_and_finite_row_statistics_overflow_reject():
    stream, family = _fixture("identity")
    lineage = FamilyExecutionLineage.from_prepared(stream.prepared, family)
    params = FamilyExecutionParameters.from_snapshot(lineage.parameters)
    control = StreamPIRLSControl(batch_rows=11, solver_policy="qr")
    with pytest.raises(ValueError, match="outside its domain"):
        nb_working_scan(
            stream, family, lineage, params, control, lambda _batch, X: -np.ones(len(X))
        )
    data = pd.DataFrame({"x": np.linspace(1000.0, 1200.0, 11), "y": np.ones(11)})
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    stream = StreamDesign(prepared, source)
    lineage = FamilyExecutionLineage.from_prepared(prepared, family)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(FloatingPointError, match="statistics overflow"),
    ):
        nb_working_scan(
            stream,
            family,
            lineage,
            params,
            control,
            lambda _batch, X: np.full(len(X), 1e-152),
        )


def test_iteration_limit_cannot_report_convergence():
    stream, family = _fixture("identity", smooth=True)
    result = fit_nb_streamed_pirls(
        stream,
        family,
        np.log([0.35]),
        maximum_bytes=10_000_000,
        control=StreamPIRLSControl(
            batch_rows=11, solver_policy="qr", max_iter=1, tol=1e-11
        ),
    )
    assert not result.state.converged
    assert result.state.n_iter == 1
    assert not result.state.line_search_failed


def test_exact_zero_residual_reporting_keeps_source_objective_separate():
    stream, family = _parametric_fixture("log", np.full(36, 2.0))
    result = _fit(stream, family)
    assert result.state.converged
    assert result.state.deviance >= 0
    assert result.state.penalized_deviance >= 0
    assert abs(result.source_deviance) < STRICT.atol
    assert (
        result.score_penalized_deviance == result.source_deviance + result.gdi_penalty
    )
    assert np.isfinite(result.reml_score)


@pytest.mark.parametrize("raw_deviance", [-1.0, np.nan])
def test_invalid_raw_deviance_cannot_become_valid_fit(monkeypatch, raw_deviance):
    import jaxgam.execution.nb_stream as controller

    stream, family = _parametric_fixture("log", np.full(36, 2.0))
    original = controller._working

    def invalid_deviance(*args, **kwargs):
        return original(*args, **kwargs)._replace(deviance=jnp.asarray(raw_deviance))

    monkeypatch.setattr(controller, "_working", invalid_deviance)
    with pytest.raises(FloatingPointError, match=r"negative or nonfinite|nonfinite"):
        _fit(stream, family)


def test_fresh_process_trial_theta_uses_identical_saved_inputs(tmp_path):
    """Fresh compiled caches reproduce binary saved inputs and trial theta."""
    stream, family = _fixture("sqrt", estimated=True)
    expected = np.asarray(
        _fit(
            stream,
            family,
            parameters=FamilyExecutionParameters(jnp.asarray([np.log(0.1)])),
        ).state.coefficients
    )
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    path = tmp_path / "input.npz"
    np.savez(
        path, x=batch.columns["x"], y=batch.y, weight=batch.weight, offset=batch.offset
    )
    script = """
import jax
jax.devices()
import json,numpy as np,pandas as pd,sys
from tests.test_execution.test_nb_stream import _fit
from jaxgam.data.source import DataFrameRowSource
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.formula.prepare import prepare_model
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.fitting.family_execution import FamilyExecutionParameters
saved=np.load(sys.argv[1])
family=NegativeBinomial(theta=2.7,link="sqrt")
source=DataFrameRowSource(pd.DataFrame({"x":saved["x"],"y":saved["y"]}),
    response="y",weights=saved["weight"],offset=saved["offset"])
prepared=prepare_model(parse_formula("y ~ x"),source,family=family)
stream=StreamDesign(prepared,source)
original=family.get_theta().copy()
outputs=[]
for theta in (.1,7.0,.1):
    result=_fit(stream,family,parameters=FamilyExecutionParameters(
        jax.numpy.asarray([np.log(theta)])))
    assert result.state.converged
    assert np.array_equal(original,family.get_theta())
    outputs.append(np.asarray(result.state.coefficients).tolist())
print(json.dumps(outputs))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "isolated-cache"),
        },
    )
    outputs = np.asarray(json.loads(process.stdout))
    np.testing.assert_array_equal(outputs[0], outputs[-1])
    np.testing.assert_allclose(outputs[0], expected, rtol=STRICT.rtol, atol=STRICT.atol)
    assert np.max(np.abs(outputs[0] - outputs[1])) > 1e-4


def test_warm_coefficient_memory_is_bounded_as_source_rows_increase(tmp_path):
    """Measure process high water after preparation, with fixed CPU B/p.

    Linux RSS is sampled every 5ms; shorter peaks can escape sampling.
    High-water increases exclude source/preparation and may hide allocations
    below an earlier peak. The retained-buffer and workspace checks separately
    forbid training-row retention; this is not a general memory/speed claim.
    """
    script = """
import jax
jax.config.update("jax_platform_name","cpu")
jax.devices()
import json,resource,sys,dataclasses,threading,os,numpy as np,pandas as pd
from jaxgam.data.source import DataFrameRowSource
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.formula.prepare import prepare_model
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.execution.nb_stream import fit_nb_streamed_pirls
from jaxgam.execution.stream import StreamPIRLSControl
def arrays(value):
    if isinstance(value,jax.Array):
        return [np.asarray(value)]
    if dataclasses.is_dataclass(value):
        return [a for f in dataclasses.fields(value)
                  for a in arrays(getattr(value,f.name))]
    return []
unit=1024 if sys.platform.startswith("linux") else 1
records=[]
for n in (256,65536):
    x=np.linspace(.5,1.5,n)
    family=NegativeBinomial(theta=2.7,fixed=True)
    source=DataFrameRowSource(pd.DataFrame({"x":x,"y":np.full(n,2.)}),
                             response="y",offset=np.full(n,.2))
    prepared=prepare_model(parse_formula("y ~ x"),source,family=family)
    before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit
    def resident_bytes():
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1])*os.sysconf("SC_PAGE_SIZE")
    samples=[]
    stop=threading.Event()
    if sys.platform.startswith("linux"):
        samples.append(resident_bytes())
        def sample_rss():
            while not stop.wait(.005):
                samples.append(resident_bytes())
        monitor=threading.Thread(target=sample_rss,daemon=True)
        monitor.start()
    result=fit_nb_streamed_pirls(StreamDesign(prepared,source),family,np.empty(0),
        maximum_bytes=10_000_000,control=StreamPIRLSControl(
            batch_rows=256,solver_policy="qr",tol=1e-7))
    if sys.platform.startswith("linux"):
        samples.append(resident_bytes())
        stop.set()
        monitor.join()
    leaves=arrays(result.state)
    after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit
    records.append(dict(n=n,p=2,B=256,converged=result.state.converged,
        additional_high_water_bytes=after-before,peak_bytes=after,
        baseline_rss_bytes=samples[0] if samples else None,
        sampled_peak_rss_bytes=max(samples) if samples else None,
        additional_sampled_rss_bytes=max(samples)-samples[0] if samples else None,
        rss_samples=len(samples),rss_sample_interval_seconds=.005,
        sampled_rss_unavailable_reason=None if samples else "requires Linux /proc",
        retained_device_numeric_bytes=sum(a.nbytes for a in leaves),
        max_retained_entries=max(a.size for a in leaves),
        known_workspace_bytes=result.workspace.required_bytes,
        source_scans=result.state.source_scans,
        batches_scanned=result.state.batches_scanned,
        deviance=float(result.state.deviance),source_deviance=result.source_deviance))
print(json.dumps(records),flush=True)
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "memory-cache")},
    )
    records = json.loads(process.stdout)
    logging.getLogger(__name__).info("NB_MEMORY_OBSERVATION %s", json.dumps(records))
    assert all(value["converged"] for value in records)
    assert all(value["deviance"] >= 0 for value in records)
    assert (
        records[0]["retained_device_numeric_bytes"]
        == records[1]["retained_device_numeric_bytes"]
    )
    assert records[0]["known_workspace_bytes"] == records[1]["known_workspace_bytes"]
    assert records[1]["max_retained_entries"] <= 4
    assert records[1]["additional_high_water_bytes"] < 64 * 1024 * 1024
    if sys.platform.startswith("linux"):
        assert records[1]["rss_samples"] > 2
        assert records[1]["additional_sampled_rss_bytes"] < 64 * 1024 * 1024


def _parametric_fixture(link, y, *, no_intercept=False):
    x = np.linspace(0.5, 1.5, len(y))
    data = pd.DataFrame({"x": x, "y": y})
    family = NegativeBinomial(theta=2.7, fixed=True, link=link)
    source = DataFrameRowSource(
        data,
        response="y",
        weights=np.linspace(0.7, 1.3, len(y)),
        offset=np.full(len(y), 0.2),
    )
    formula = "y ~ 0 + x" if no_intercept else "y ~ 1"
    prepared = prepare_model(parse_formula(formula), source, family=family)
    return StreamDesign(prepared, _CountedSource(source)), family


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("kind", ["fractional", "count_tail", "no_intercept"])
def test_pinned_fractional_count_tail_and_no_intercept_controller_layers(link, kind):
    """Fractional R pmax constants are compared directly, including score."""
    y = (
        np.resize([0, 0.25, 0.5, 0.75, 1.5, 2.0], 36)
        if kind == "fractional"
        else np.resize([0, 1, 2, 101, 0, 3], 36)
        if kind == "count_tail"
        else np.random.default_rng(1439).poisson(4, size=36).astype(float)
    )
    stream, family = _parametric_fixture(link, y, no_intercept=kind == "no_intercept")
    expected, X = _pinned_reference(stream, link, 2.7)
    assert expected["converged"][0]
    for B in (1, 7, 100):
        result = _fit(stream, family, B)
        assert result.state.converged
        assert result.integer_counts == bool(np.all(y == np.floor(y)))
        assert result.max_count == np.max(y)
        np.testing.assert_allclose(
            result.state.coefficients,
            expected["beta"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            [
                result.state.deviance,
                result.state.edf,
                result.gdi_penalty,
                result.reml_score,
            ],
            [expected[key][0] for key in ("dev", "edf", "gdi", "score")],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            result.state.fisher_coefficient_factor.hessian_inverse(jnp.eye(X.shape[1])),
            expected["V"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


def test_zero_curvature_direct_rhs_reaches_penalized_solve():
    data = pd.DataFrame({"y": [0.25], "x": [0.0]})
    family = NegativeBinomial(theta=2, fixed=True, link="identity")
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(parse_formula("y ~ 1"), source, family=family)
    stream = StreamDesign(prepared, source)
    lineage = FamilyExecutionLineage.from_prepared(prepared, family)
    scan = nb_working_scan(
        stream,
        family,
        lineage,
        FamilyExecutionParameters.from_snapshot(lineage.parameters),
        StreamPIRLSControl(batch_rows=1, solver_policy="qr"),
        lambda _batch, X: np.ones(len(X)),
    )
    assert scan.good_count == 1
    assert scan.informative_count == 0
    assert scan.used_direct_response
    np.testing.assert_array_equal(scan.gradient_rhs, np.zeros(1))
    result = solve_signed_qr(scan.observed, ((slice(0, 1), np.array([[np.sqrt(2)]])),))
    np.testing.assert_allclose(
        result.coefficients, [-0.25], rtol=STRICT.rtol, atol=STRICT.atol
    )


def _sparse_fixture(mean=0.3):
    rng = np.random.default_rng(883)
    x = np.linspace(-1, 1, 96)
    theta = 0.8
    y = rng.negative_binomial(theta, theta / (theta + mean + 0.01 * np.sin(x)))
    family = NegativeBinomial(theta=theta, fixed=True, link="identity")
    source = DataFrameRowSource(
        pd.DataFrame({"x": x, "y": y}), response="y", offset=np.ones(len(x))
    )
    prepared = prepare_model(
        parse_formula('y ~ s(x,bs="cr",k=5)'), source, family=family
    )
    return StreamDesign(prepared, _CountedSource(source)), family


def test_real_indefinite_steps_recover_positive_observed_and_backtrack():
    stream, family = _sparse_fixture()
    result = _fit(stream, family)
    assert result.state.converged
    assert result.positive_observed_recoveries == 3
    assert result.state.backtracks > 0
    assert result.state.n_iter > result.positive_observed_recoveries
    np.testing.assert_array_equal(family.get_theta(), [np.log(0.8)])


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
def test_sparse_positive_observed_recovery_matches_source_once():
    stream, family = _sparse_fixture()
    expected, _X = _pinned_reference(stream, "identity", 0.8)
    result = _fit(stream, family)
    assert result.positive_observed_recoveries == expected["recovery"][0]
    assert result.state.n_iter == expected["iter"][0]
    np.testing.assert_allclose(
        result.state.coefficients, expected["beta"], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        [
            result.state.deviance,
            result.state.edf,
            result.gdi_penalty,
            result.reml_score,
        ],
        [expected[key][0] for key in ("dev", "edf", "gdi", "score")],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
def test_preserved_sparse_theta_point_one_identity_source_recovery_boundary():
    """Original theta=.1 response fixture is retained, not replaced by a gate."""
    embedded = pytest.importorskip("rpy2.rinterface_lib.embedded")
    stream, family = _fixture("identity", 0.1, smooth=True)
    with pytest.raises(
        embedded.RRuntimeError,
        match="inner loop 1; can't correct step size",
    ):
        _pinned_reference(stream, "identity", 0.1)
    # Failure of this input does not close identity-link capability. It must
    # remain nonconverged/rejected by the bounded source controller too.
    try:
        result = _fit(stream, family)
    except (FloatingPointError, np.linalg.LinAlgError):
        return
    assert not result.state.converged


def _pinned_reference(stream, link, theta):
    ro = pytest.importorskip("rpy2.robjects")
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    X = stream.prepared.evaluate_fitting_batch(batch)
    roots = qr_penalty_roots(stream.prepared.fitting.penalty_structure)
    E = np.zeros((len(roots[0].root) if roots else 0, stream.prepared.n_coef))
    if roots:
        root = roots[0]
        E[:, root.start : root.stop] = root.root

    def matrix(value):
        return ro.r.matrix(
            ro.FloatVector(value.ravel(order="F")),
            nrow=value.shape[0],
            ncol=value.shape[1],
        )

    oracle = ro.r("""function(X,E,y,wt,off,link,theta) {
      fam <- mgcv:::fix.family.link(do.call(mgcv::nb,list(theta=theta,link=link)))
      q <- ncol(X); rank <- nrow(E)
      U1 <- if(rank) eigen(crossprod(E),symmetric=TRUE)$vectors else diag(q)
      UrS <- if(rank) list(t(U1[,seq_len(rank),drop=FALSE]) %*% t(E)) else list()
      null <- qr.coef(qr(X),rep(fam$linkfun(mean(y)),length(y)))
      null[is.na(null)] <- 0
      fitfun <- mgcv:::gam.fit4
      last <- length(body(fitfun))
      body(fitfun)[[last]] <- substitute({value <- RETURN;
          value$gdi.penalty <- oo$P;value},list(RETURN=body(fitfun)[[last]]))
      trace <- capture.output(fit <- fitfun(x=X,y=y,
        sp=if(rank) log(.35) else numeric(0),Eb=E,
        UrS=UrS,weights=wt,offset=off,U1=U1,Mp=q-rank,family=fam,
        control=mgcv::gam.control(epsilon=1e-11,maxit=100,trace=TRUE),
        deriv=0,scale=1,scoreType="REML",null.coef=null))
      dd <- mgcv:::dDeta(y,fit$fitted.values,wt,log(theta),fam,0)
      G <- crossprod(X,.5*dd$Deta2*X)
      F <- crossprod(X,.5*dd$EDeta2*X)
      V <- tcrossprod(fit$rV)
      list(beta=fit$coefficients, dev=fit$deviance, G=G,F=F,V=V,
           edf=sum(diag(V%*%F)), score=fit$REML,gdi=fit$gdi.penalty,
           iter=fit$iter,converged=fit$converged,
           recovery=sum(grepl("using positive weights",trace,fixed=TRUE)))
    }""")(
        matrix(X),
        matrix(E),
        ro.FloatVector(batch.y),
        ro.FloatVector(batch.weight),
        ro.FloatVector(batch.offset),
        link,
        theta,
    )
    expected = {name: np.asarray(oracle.rx2(name)) for name in oracle.names}
    return expected, X


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("theta", [0.1, 2.7, 1e6])
def test_pinned_gam_fit4_fixed_theta_prepared_basis_stopping_and_information(
    link, theta
):
    """Layer parity fixes the response across theta and supplies identical X/S."""
    stream, family = _fixture(link, theta, smooth=True, response_theta=2.7)
    expected, X = _pinned_reference(stream, link, theta)
    assert expected["converged"][0]
    for B in (5, 200):
        result = _fit(stream, family, B)
        state = result.state
        collector = _AssertCollector()
        for name, actual in (
            ("beta", state.coefficients),
            ("dev", state.deviance),
            ("G", state.xtwx),
            ("F", state.xtwx_fisher),
            ("V", state.fisher_coefficient_factor.hessian_inverse(jnp.eye(X.shape[1]))),
            ("edf", state.edf),
            ("gdi", result.gdi_penalty),
            ("score", result.reml_score),
        ):
            collector.check(
                name,
                lambda actual=actual, name=name: np.testing.assert_allclose(
                    actual,
                    np.squeeze(expected[name]),
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
        collector.check(
            "source outer iteration",
            lambda state=state: np.testing.assert_equal(
                state.n_iter, expected["iter"][0]
            ),
        )
        collector.check(
            "positive observed recoveries",
            lambda result=result: np.testing.assert_equal(
                result.positive_observed_recoveries, expected["recovery"][0]
            ),
        )
        collector.raise_if_any("gam.fit4 coefficient layer")
