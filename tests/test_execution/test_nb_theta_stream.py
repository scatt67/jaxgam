"""Global conditional theta Newton and its in-PIRLS source timing."""

import jax
import numpy as np
import pytest

from jaxgam.execution.nb_stream import fit_nb_streamed_pirls
from jaxgam.execution.nb_theta_stream import (
    NBThetaStreamControl,
    conditional_theta_stream,
)
from jaxgam.execution.regular_stream import preflight_regular_stream_workspace
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.fitting.efs_theta import conditional_theta_newton
from jaxgam.fitting.family_execution import (
    FamilyExecutionLineage,
    FamilyExecutionParameters,
)
from jaxgam.formula.fitting_prepare import qr_penalty_roots
from tests.helpers import _AssertCollector, r_available
from tests.test_execution.test_nb_stream import _fit, _fixture
from tests.tolerances import MODERATE, STRICT


def _controls(B=11, tol=1e-11):
    return StreamPIRLSControl(batch_rows=B, solver_policy="qr", tol=tol)


def _theta_fit(stream, family, B=11, **kwargs):
    return _fit(
        stream,
        family,
        B,
        parameters=FamilyExecutionParameters(np.array([np.log(0.7)])),
        estimate_theta=True,
        **kwargs,
    )


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_global_conditional_newton_matches_dense_trajectory_and_counted_replay(link):
    stream, family = _fixture(link, estimated=True)
    lineage = FamilyExecutionLineage.from_prepared(stream.prepared, family)
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    X = stream.prepared.evaluate_fitting_batch(batch)
    beta = np.array([0.6, 0.05]) if link != "identity" else np.array([2.0, 0.05])
    eta = X @ beta + batch.offset
    max_y = int(np.max(batch.y))
    start = np.array([np.log(0.7)])
    dense = conditional_theta_newton(
        start,
        eta,
        batch.y,
        batch.weight,
        batch.y.astype(np.int64),
        family,
        max_y=max_y,
        integer_counts=True,
    )
    original = family.get_theta().copy()
    for B in (1, 11, 200):
        scans, batches = stream.source.scans, stream.source.batches
        result = conditional_theta_stream(
            stream,
            family,
            lineage,
            beta,
            start,
            _controls(B),
            max_y=max_y,
            integer_counts=True,
        )
        assert result.converged
        assert result.n_iter == int(dense.n_iter)
        assert result.source_scans == stream.source.scans - scans
        assert result.batches_scanned == stream.source.batches - batches
        np.testing.assert_allclose(
            result.log_theta, dense.log_theta, rtol=STRICT.rtol, atol=STRICT.atol
        )
        np.testing.assert_allclose(
            result.nll_history,
            np.asarray(dense.nll_history)[: int(dense.n_history)],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            [result.nll, result.gradient, result.hessian],
            [dense.nll, dense.gradient, dense.hessian],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        assert abs(result.gradient) <= 1e-7 * (abs(result.nll) + 1)
        np.testing.assert_array_equal(family.get_theta(), original)


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_conditional_theta_inside_pirls_retained_rejected_trials_and_counts(link):
    stream, family = _fixture(link, estimated=True, smooth=True)
    original = family.get_theta().copy()
    before = stream.source.scans, stream.source.batches
    first = _theta_fit(stream, family)
    assert first.state.converged
    assert first.theta_status == 0
    assert first.theta_n_iter > 0
    assert first.theta_source_scans > 0
    assert len(first.theta_history) == first.state.n_iter + 1
    assert first.state.source_scans == stream.source.scans - before[0]
    assert first.state.batches_scanned == stream.source.batches - before[1]
    retained = _theta_fit(
        stream, family, beta_start=np.asarray(first.state.coefficients)
    )
    assert retained.initial_start_retained
    assert retained.state.converged
    rejected = _theta_fit(
        stream, family, beta_start=np.full(stream.prepared.n_coef, 100.0)
    )
    assert not rejected.initial_start_retained
    assert rejected.state.converged
    # Retained starts legitimately follow a different source stopping trajectory;
    # owning pinned gates compare each against R with the identical start.
    assert retained.state.stationarity < _controls().tol * (
        abs(retained.stopping_penalized_deviance) + 1
    )
    np.testing.assert_array_equal(rejected.state.coefficients, first.state.coefficients)
    np.testing.assert_array_equal(rejected.log_theta, first.log_theta)
    np.testing.assert_array_equal(family.get_theta(), original)
    repeated = _theta_fit(stream, family)
    np.testing.assert_array_equal(repeated.state.coefficients, first.state.coefficients)
    np.testing.assert_array_equal(repeated.log_theta, first.log_theta)


def test_invalid_controls_starts_conditional_modes_and_prospective_budget():
    stream, family = _fixture(estimated=True)
    for kwargs in (
        {"estimate_theta": 1},
        {"start_is_absent": 1},
        {"start_is_absent": False},
        {"beta_start": np.zeros(3)},
        {"beta_old_init": np.zeros(3)},
    ):
        with pytest.raises(ValueError, match=r"bool|start|anchor"):
            _fit(
                stream,
                family,
                parameters=FamilyExecutionParameters(np.array([0.0])),
                **kwargs,
            )
    base = preflight_regular_stream_workspace(stream, _controls(), 10_000_000)
    with pytest.raises(MemoryError, match="conditional theta known workspace"):
        fit_nb_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=base.required_bytes + 8 * 64 * base.batch_rows - 1,
            parameters=FamilyExecutionParameters(np.zeros(1)),
            control=_controls(),
            estimate_theta=True,
        )
    _, fixed = _fixture()
    with pytest.raises(ValueError, match="estimated NB"):
        _fit(stream, fixed, estimate_theta=True)
    for kwargs in (
        {"tolerance": True},
        {"max_iter": 0},
        {"max_step": np.nan},
        {"max_halvings": -1},
    ):
        with pytest.raises(ValueError, match=r"conditional theta"):
            NBThetaStreamControl(**kwargs)


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("retained", [False, True])
def test_pinned_theta_controller_matches_raw_score_timing_factors_and_trajectory(
    link, retained
):
    stream, family = _fixture(link, estimated=True, smooth=True)
    start = (
        np.asarray(_theta_fit(stream, family).state.coefficients) if retained else None
    )
    result = _theta_fit(stream, family, beta_start=start)
    oracle = _pinned_controller(stream, link, start)
    collector = _AssertCollector()
    fields = {
        "beta": result.state.coefficients,
        "theta": result.log_theta,
        "source_deviance": result.source_deviance,
        "final_deviance": result.final_theta_deviance,
        "source_theta": result.source_deviance_log_theta,
        "gdi_penalty": result.gdi_penalty,
        "score": result.reml_score,
        "stopping_pdev": result.stopping_penalized_deviance,
        "G": result.state.xtwx,
        "F": result.state.xtwx_fisher,
        "V": result.state.fisher_coefficient_factor.hessian_inverse(
            jax.numpy.eye(stream.prepared.n_coef)
        ),
        "theta_history": result.theta_history,
    }
    for field, actual in fields.items():
        collector.check(
            field,
            lambda field=field, actual=actual: np.testing.assert_allclose(
                actual, oracle[field], rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("streamed in-PIRLS theta/source parity")
    assert result.state.n_iter == oracle["iter"]
    assert result.theta_n_iter == oracle["theta_iter"]
    assert result.positive_observed_recoveries == oracle["recoveries"]
    assert result.initial_start_retained == retained
    assert result.state.converged


def _pinned_controller(stream, link, start, tol=1e-11):
    """Instrument local source copies; binary R arrays avoid CSV float loss."""
    ro = pytest.importorskip("rpy2.robjects")
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    prepared = stream.prepared
    X = prepared.evaluate_fitting_batch(batch)
    X_public = prepared.evaluate_batch(batch)
    p = prepared.n_coef
    D = np.eye(p)
    for block in prepared.fitting.penalty_structure.blocks:
        D[block.start : block.stop, block.start : block.stop] = block.transform.dense()
    roots = qr_penalty_roots(prepared.fitting.penalty_structure)
    E = np.zeros((sum(len(root.root) for root in roots), p))
    at = 0
    for root in roots:
        E[at : at + len(root.root), root.start : root.stop] = root.root
        at += len(root.root)

    def matrix(value):
        return ro.r.matrix(
            ro.FloatVector(value.ravel(order="F")),
            nrow=value.shape[0],
            ncol=value.shape[1],
        )

    function = ro.r("""function(X,Xpublic,D,E,y,wt,off,link,start,rank,tol) {
      stopifnot(as.character(getRversion())=="4.5.2",packageVersion("mgcv")==package_version("1.9.3"))
      fam <- mgcv:::fix.family.link(do.call(mgcv::nb,list(theta=-.7,link=link)))
      q <- ncol(X)
      U1 <- if(rank) eigen(crossprod(E),symmetric=TRUE)$vectors else diag(q)
      UrS <- if(rank) list(t(U1[,seq_len(rank),drop=FALSE]) %*% t(E)) else list()
      null <- qr.coef(qr(Xpublic),rep(fam$linkfun(mean(y)),length(y)))
      null[is.na(null)] <- 0; null <- solve(D,null)
      theta_count <- 0L; theta_path <- log(.7); last_pre_theta <- log(.7)
      lines <- deparse(mgcv:::estimate.theta,width.cutoff=500L)
      lines[1] <- sub("function", "traced <- function",lines[1],fixed=TRUE)
      anchor <- "theta <- theta + step"
      stopifnot(sum(grepl(anchor,lines,fixed=TRUE))==1L)
      insertion <- paste0(anchor,"; theta_count <<- theta_count+1L")
      lines <- sub(anchor,insertion,lines,fixed=TRUE)
      eval(parse(text=lines))
      env <- new.env(parent=environment(mgcv:::gam.fit4))
      env$estimate.theta <- function(theta,...) {
         last_pre_theta <<- theta
         result <- traced(theta,...)
         theta_path <<- c(theta_path,result)
         result
      }
      fitfun <- mgcv:::gam.fit4; environment(fitfun) <- env
      last <- length(body(fitfun))
      body(fitfun)[[last]] <- substitute({value <- RETURN;
         value$gdi.penalty <- oo$P; value$stopping <- pdev; value},
         list(RETURN=body(fitfun)[[last]]))
      trace <- capture.output(fit <- fitfun(x=X,y=y,
         sp=c(log(.7),if(rank) log(.35) else numeric(0)),
         Eb=E,UrS=UrS,weights=wt,offset=off,start=if(length(start)) start else NULL,
         U1=U1,Mp=q-rank,family=fam,control=mgcv::gam.control(epsilon=tol,maxit=100,trace=TRUE),
         deriv=0,scale=1,scoreType="EFS",null.coef=null))
      theta <- fam$getTheta()
      dd <- mgcv:::dDeta(y,fit$fitted.values,wt,theta,fam,0)
      list(beta=fit$coefficients,theta=theta,source_deviance=fit$deviance,
         final_deviance=sum(fam$dev.resids(y,fit$fitted.values,wt,theta)),
         source_theta=last_pre_theta,gdi_penalty=fit$gdi.penalty,score=fit$REML,
         stopping_pdev=fit$stopping,G=crossprod(X,.5*dd$Deta2*X),F=crossprod(X,.5*dd$EDeta2*X),
         V=tcrossprod(fit$rV),theta_history=theta_path,iter=fit$iter,theta_iter=theta_count,
         null_coefficients=null,
         recoveries=sum(grepl("using positive weights",trace,fixed=TRUE)))
    }""")
    result = function(
        matrix(X),
        matrix(X_public),
        matrix(D),
        matrix(E),
        ro.FloatVector(batch.y),
        ro.FloatVector(batch.weight),
        ro.FloatVector(batch.offset),
        link,
        ro.FloatVector([] if start is None else start),
        prepared.fitting.total_penalty_rank,
        tol,
    )
    return {name: np.asarray(result.rx2(name)) for name in result.names}


@pytest.mark.parametrize(
    ("case", "status"),
    [
        ("initial", 1),
        ("curvature", 2),
        ("zero", 3),
        ("overflow", 4),
        ("line", 5),
        ("limit", 6),
        ("post", 7),
        ("repair", 0),
        ("infinite", 0),
    ],
)
def test_global_source_newton_failure_retention_and_recoverable_infinite_trials(
    monkeypatch, case, status
):
    from jaxgam.execution import nb_theta_stream
    from jaxgam.fitting.nb_theta_stream import NBConditionalThetaBatch

    stream, family = _fixture(estimated=True)
    lineage = FamilyExecutionLineage.from_prepared(stream.prepared, family)

    def fake_batch(theta, *_args, **_kwargs):
        t = float(np.asarray(theta)[0])
        if case == "initial":
            return NBConditionalThetaBatch(np.nan, 1.0, 1.0, False)
        if case == "curvature":
            return NBConditionalThetaBatch(10.0, 1.0, np.nan, False)
        if case == "zero":
            return NBConditionalThetaBatch(10.0, 1.0, 0.0, True)
        if case == "overflow":
            return NBConditionalThetaBatch(10.0, 1e308, 1e-308, True)
        if case == "line":
            return NBConditionalThetaBatch(1.0 + t * t, -1.0, 1.0, True)
        if case == "limit":
            return NBConditionalThetaBatch(10.0, 1.0, 1.0, True)
        if case == "post":
            return NBConditionalThetaBatch(
                10.0 - t, -1.0, 1.0 if t == 0 else np.nan, t == 0
            )
        if case == "repair":
            return NBConditionalThetaBatch(
                10.0 - t, -1.0 if t == 0 else 0.0, -1.0, True
            )
        return NBConditionalThetaBatch(
            10.0 if t == 0 else (np.inf if t > 1 else 9.0),
            -10.0 if t == 0 else 0.0,
            1.0,
            t <= 1,
        )

    monkeypatch.setattr(nb_theta_stream, "_batch", fake_batch)
    result = conditional_theta_stream(
        stream,
        family,
        lineage,
        np.array([0.6, 0.05]),
        np.zeros(1),
        _controls(200),
        max_y=100,
        integer_counts=True,
        control=NBThetaStreamControl(
            max_iter=1, max_halvings=0 if case == "line" else 25
        ),
    )
    assert result.status == status
    assert result.converged == (status == 0)
    assert result.source_scans == stream.source.scans
    assert result.batches_scanned == stream.source.batches
    if status in (1, 2, 3, 4, 5, 7):
        assert result.log_theta == (0.0,)
        assert len(result.nll_history) == 1
    if case == "infinite":
        assert result.log_theta == (1.0,)
        assert result.halvings == 2
    if case == "repair":
        assert result.stabilized_curvatures == 1


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
def test_converged_theta_transition_keeps_source_score_and_reported_state():
    stream, family = _fixture("log", estimated=True, smooth=True)
    result = fit_nb_streamed_pirls(
        stream,
        family,
        np.log([0.35]),
        maximum_bytes=10_000_000,
        parameters=FamilyExecutionParameters(np.array([np.log(0.7)])),
        control=_controls(tol=0.5),
        estimate_theta=True,
    )
    oracle = _pinned_controller(stream, "log", None, tol=0.5)
    assert result.state.converged
    assert result.state.n_iter == int(oracle["iter"][0]) == 2
    assert abs(result.source_deviance - result.final_theta_deviance) > 1.0
    assert abs(result.source_deviance_log_theta[0] - result.log_theta[0]) > 0.05
    assert result.gdi_penalty > 0.0
    collector = _AssertCollector()
    for key, actual in {
        "source_deviance": result.source_deviance,
        "final_deviance": result.final_theta_deviance,
        "source_theta": result.source_deviance_log_theta,
        "theta": result.log_theta,
        "gdi_penalty": result.gdi_penalty,
        "score": result.reml_score,
        "stopping_pdev": result.stopping_penalized_deviance,
        "G": result.state.xtwx,
        "F": result.state.xtwx_fisher,
    }.items():
        collector.check(
            key,
            lambda key=key, actual=actual: np.testing.assert_allclose(
                actual, oracle[key], rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("source pre-theta score/post-theta factor transition")
    assert (
        result.score_penalized_deviance == result.source_deviance + result.gdi_penalty
    )
    assert float(result.state.deviance) == result.final_theta_deviance


def test_count_budget_is_checked_after_summary_before_theta_dispatch(
    monkeypatch,
):
    import jaxgam.execution.nb_theta_stream as engine

    stream, family = _fixture(estimated=True)
    control, theta_control = _controls(), NBThetaStreamControl()
    ledger = preflight_regular_stream_workspace(stream, control, 10_000_000)
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    capacity = max(1, int(np.ceil(np.max(batch.y))))
    required = (
        ledger.required_bytes
        + 8 * 64 * ledger.batch_rows
        + 128 * (theta_control.max_iter + control.max_iter + 2)
        + 8 * 16 * (capacity + 1)
    )

    def forbid_derivative(*_args, **_kwargs):
        raise AssertionError("conditional derivative dispatched before count budget")

    monkeypatch.setattr(engine, "_batch", forbid_derivative)
    before = stream.source.scans
    with pytest.raises(MemoryError, match="count-prefix workspace"):
        fit_nb_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=required - 1,
            parameters=FamilyExecutionParameters(np.zeros(1)),
            control=control,
            estimate_theta=True,
        )
    assert stream.source.scans == before + 1


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_no_intercept_null_projection_counts_zero_priors_and_releases_qr(
    monkeypatch, link
):
    import weakref

    import pandas as pd

    import jaxgam.execution.nb_stream as controller
    from jaxgam.data.source import DataFrameRowSource
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.formula.design_provider import StreamDesign
    from jaxgam.formula.parser import parse_formula
    from jaxgam.formula.prepare import prepare_model
    from tests.test_execution.test_regular_stream import _CountedSource

    x = np.linspace(0.6, 1.4, 61)
    rng = np.random.default_rng(4019)
    mu, theta = 3.0 + 0.2 * x, 1.2
    y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
    weight = 0.6 + rng.uniform(size=len(y))
    weight[::7] = 0.0
    source = DataFrameRowSource(
        pd.DataFrame({"x": x, "y": y}),
        response="y",
        weights=weight,
        offset=0.2 + 0.1 * x,
    )
    family = NegativeBinomial(theta=0.7, link=link)
    prepared = prepare_model(parse_formula("y ~ 0 + x"), source, family=family)
    stream = StreamDesign(prepared, _CountedSource(source))
    original_project = controller.project_null_coefficients
    original_scan = controller.nb_working_scan
    saved = []

    def projection(state, eta):
        assert state.n_data_rows == len(y)  # zero priors are real source rows
        result = original_project(state, eta)
        saved.append((weakref.ref(state.R), result.coefficients.copy()))
        return result

    def working(*args, **kwargs):
        assert saved
        assert saved[-1][0]() is None
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(controller, "project_null_coefficients", projection)
    monkeypatch.setattr(controller, "nb_working_scan", working)
    oracle = _pinned_controller(stream, link, None)
    assert np.any(weight == 0.0)
    for B in (1, 11, 200):
        before = stream.source.scans, stream.source.batches
        result = _theta_fit(stream, family, B)
        assert result.state.converged
        np.testing.assert_allclose(
            saved[-1][1],
            oracle["null_coefficients"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        checks = _AssertCollector()
        for field, actual in {
            "beta": result.state.coefficients,
            "theta": result.log_theta,
            "source_deviance": result.source_deviance,
            "source_theta": result.source_deviance_log_theta,
            "final_deviance": result.final_theta_deviance,
            "gdi_penalty": result.gdi_penalty,
            "score": result.reml_score,
            "G": result.state.xtwx,
            "F": result.state.xtwx_fisher,
        }.items():
            checks.check(
                field,
                lambda field=field, actual=actual: np.testing.assert_allclose(
                    actual,
                    oracle[field],
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
        checks.raise_if_any("no-intercept/real-zero-prior source theta fit")
        assert result.state.n_iter == int(oracle["iter"][0])
        assert result.theta_n_iter == int(oracle["theta_iter"][0])
        assert result.state.source_scans == stream.source.scans - before[0]
        assert result.state.batches_scanned == stream.source.batches - before[1]


def _pinned_six_row_theta(start, y, mu, weight, link):
    """Binary source contractions and a traced unmodified theta update loop."""
    ro = pytest.importorskip("rpy2.robjects")
    function = ro.r("""function(start,y,mu,w,link) {
      stopifnot(as.character(getRversion())=="4.5.2",packageVersion("mgcv")==package_version("1.9.3"))
      fam <- do.call(mgcv::nb,list(theta=-exp(start),link=link))
      fields <- function(theta) {
        ls <- fam$ls(y,w=w,theta=theta,scale=1)
        dd <- fam$Dd(y,mu,theta,wt=w,level=2)
        c(sum(fam$dev.resids(y,mu,w,theta))/2-ls$ls,
          sum(dd$Dth)/2-ls$lsth1[1],sum(dd$Dth2)/2-as.matrix(ls$lsth2)[1,1])
      }
      theta_path <- start; halving_count <- 0L
      lines <- deparse(mgcv:::estimate.theta,width.cutoff=500L)
      lines[1] <- sub("function","traced <- function",lines[1],fixed=TRUE)
      anchor <- "theta <- theta + step"
      stopifnot(sum(grepl(anchor,lines,fixed=TRUE))==1L)
      insertion <- paste0(anchor,"; theta_path <<- c(theta_path,theta)")
      lines <- sub(anchor,insertion,lines,fixed=TRUE)
      halving_anchor <- "iter <- iter + 1"
      stopifnot(sum(grepl(halving_anchor,lines,fixed=TRUE))==1L)
      insertion <- paste0(halving_anchor,"; halving_count <<- halving_count+1L")
      lines <- sub(halving_anchor,insertion,lines,fixed=TRUE)
      eval(parse(text=lines))
      end <- traced(start,fam,y,mu,scale=1,wt=w)
      list(initial=fields(start),final=c(end,fields(end)),path=theta_path,
           halvings=halving_count)
    }""")
    result = function(
        ro.FloatVector(start),
        ro.FloatVector(y),
        ro.FloatVector(mu),
        ro.FloatVector(weight),
        link,
    )
    return {name: np.asarray(result.rx2(name)) for name in result.names}


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("variant", ["integer", "fractional_ge_one"])
def test_exact_six_row_theta_boundary_has_reviewed_field_specific_source_gates(
    link, variant
):
    """Exact hashed fixtures: reviewed cancellation fields, no path relaxation.

    See docs/scale_jaxgam/efs52_nb_conditional_theta_numerical_review.md.
    All ordinary controller and identical-coordinate end contractions stay STRICT.
    """
    import hashlib
    import json
    from pathlib import Path

    import jax.numpy as jnp
    import pandas as pd

    from jaxgam.data.source import DataFrameRowSource
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.fitting.nb_theta_stream import nb_conditional_theta_batch
    from jaxgam.formula.design_provider import StreamDesign
    from jaxgam.formula.parser import parse_formula
    from jaxgam.formula.prepare import prepare_model

    path = (
        Path(__file__).parents[1]
        / "fixtures"
        / ("efs52_nb_theta_six_row_" + variant + ".json")
    )
    raw = path.read_bytes()
    expected_hash = {
        "integer": "da0bc32e17c97b4decc43fa523c00315fd4f975a0fa73c7fadbdc0f3e5e16fcc",
        "fractional_ge_one": (
            "82c7333e5f8e0e862875afc4257e25abdd9aff3f5d2ab2ed656ea44e73a8d6a3"
        ),
    }[variant]
    assert hashlib.sha256(raw).hexdigest() == expected_hash
    fixture = json.loads(raw)
    mu, y, weight = [np.asarray(fixture[key]) for key in ("mu", "y", "weight")]
    start = np.asarray(fixture["log_theta"])
    eta = np.log(mu) if link == "log" else mu if link == "identity" else np.sqrt(mu)
    family = NegativeBinomial(theta=0.7, link=link)
    compiled = jax.jit(
        nb_conditional_theta_batch,
        static_argnames=("family", "max_y", "integer_counts"),
    )

    def fields(theta):
        values = []
        for where in (slice(0, 2), slice(2, 5), slice(5, 6)):
            result = compiled(
                jnp.asarray(theta),
                jnp.asarray(eta[where]),
                jnp.asarray(y[where]),
                jnp.asarray(weight[where]),
                jnp.ones_like(y[where], dtype=bool),
                family,
                max_y=fixture["max_y"],
                integer_counts=fixture["integer_counts"],
            )
            assert result.admissible
            values.append(np.asarray(result[:3]))
        return np.sum(values, axis=0)

    source = DataFrameRowSource(
        pd.DataFrame({"x": eta, "y": y}), response="y", weights=weight
    )
    prepared = prepare_model(parse_formula("y ~ 0 + x"), source, family=family)
    stream = StreamDesign(prepared, source)
    end = conditional_theta_stream(
        stream,
        family,
        FamilyExecutionLineage.from_prepared(prepared, family),
        np.ones(1),
        start,
        _controls(2),
        max_y=fixture["max_y"],
        integer_counts=fixture["integer_counts"],
    )
    oracle = _pinned_six_row_theta(start, y, mu, weight, link)
    initial = fields(start)
    # Both exact hashed inputs have reviewed objective-cancellation gates.
    # The integer gate became MODERATE when the published stable saturated
    # likelihood removed the source gamma cancellation; see the numerical
    # review. Every derivative, final-state and trajectory assertion below
    # retains its independently reviewed class.
    objective_tolerance = MODERATE
    theta_tolerance = MODERATE if variant == "integer" else STRICT
    np.testing.assert_allclose(
        initial[0],
        oracle["initial"][0],
        rtol=objective_tolerance.rtol,
        atol=objective_tolerance.atol,
    )
    np.testing.assert_allclose(
        initial[1:],
        oracle["initial"][1:],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    np.testing.assert_allclose(
        end.log_theta,
        oracle["final"][0],
        rtol=theta_tolerance.rtol,
        atol=theta_tolerance.atol,
    )
    np.testing.assert_allclose(
        end.nll,
        oracle["final"][1],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    # Compare derivatives at the identical R-selected coordinate, separately
    # from the legitimately differing selected theta and early trajectory.
    np.testing.assert_allclose(
        fields(oracle["final"][:1]),
        oracle["final"][1:],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert end.converged
    assert end.n_iter == len(oracle["path"]) - 1
    assert abs(initial[1]) > 1e-7 * (abs(initial[0]) + 1)
    assert abs(oracle["initial"][1]) > 1e-7 * (abs(oracle["initial"][0]) + 1)
    assert abs(end.gradient) <= 1e-7 * (abs(end.nll) + 1)
    assert abs(oracle["final"][2]) <= 1e-7 * (abs(oracle["final"][1]) + 1)
    assert end.hessian > 0
    assert oracle["final"][3] > 0
    assert end.halvings == int(oracle["halvings"][0])
    assert end.halvings <= 25 * end.n_iter
    assert end.stabilized_curvatures > 0
    allowed = np.finfo(float).eps ** 0.75 * np.abs(end.nll_history[:-1])
    assert np.all(np.diff(end.nll_history) <= allowed)
    np.testing.assert_array_equal(family.get_theta(), [np.log(0.7)])


def test_conditional_n_growth_keeps_numeric_buffers_histories_and_ledger_bounded(
    monkeypatch,
):
    import dataclasses
    import weakref

    import pandas as pd

    import jaxgam.execution.nb_stream as controller
    from jaxgam.data.source import DataFrameRowSource
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.formula.design_provider import StreamDesign
    from jaxgam.formula.parser import parse_formula
    from jaxgam.formula.prepare import prepare_model
    from tests.test_execution.test_regular_stream import _CountedSource

    def arrays(value):
        if isinstance(value, (jax.Array, np.ndarray)):
            return [np.asarray(value)]
        if dataclasses.is_dataclass(value):
            return [
                array
                for field in dataclasses.fields(value)
                for array in arrays(getattr(value, field.name))
            ]
        if isinstance(value, (tuple, list)):
            return [array for item in value for array in arrays(item)]
        return []

    previous = []
    original_theta = controller.conditional_theta_stream

    def conditional(*args, **kwargs):
        if previous:
            assert previous[-1]() is None
        result = original_theta(*args, **kwargs)
        previous.append(weakref.ref(result))
        return result

    monkeypatch.setattr(controller, "conditional_theta_stream", conditional)
    records = []
    for n in (128, 1024):
        previous.clear()  # a prior caller-owned fit is outside this fit's ledger
        y = np.resize([0.0, 1.0, 2.0, 9.0], n)
        weight = np.resize([0.0, 0.8, 1.2, 1.0], n)
        family = NegativeBinomial(theta=0.7)
        source = DataFrameRowSource(
            pd.DataFrame({"y": y}), response="y", weights=weight, offset=np.full(n, 0.4)
        )
        prepared = prepare_model(parse_formula("y ~ 1"), source, family=family)
        stream = StreamDesign(prepared, _CountedSource(source))
        result = _theta_fit(stream, family, 32)
        assert result.state.converged
        assert result.state.source_scans == stream.source.scans
        assert result.state.batches_scanned == stream.source.batches
        assert result.theta_source_scans >= result.state.n_iter
        assert len(result.theta_history) <= _controls().max_iter + 1
        assert len(result.last_theta_result.theta_history) <= 101
        assert len(result.last_theta_result.nll_history) <= 101
        buffers = arrays(result)
        assert max(array.size for array in buffers) <= prepared.n_coef**2
        assert result.max_count == 9
        records.append(
            (result.workspace.required_bytes, sum(a.nbytes for a in buffers))
        )
    assert records[0] == records[1]


def test_initial_batch_cap_fails_before_design_or_null_qr(monkeypatch):
    import jaxgam.execution.nb_stream as controller

    stream, family = _fixture(estimated=True)
    original = controller._source_batches

    def oversized(stream, family, lineage, _batch_rows):
        return original(stream, family, lineage, stream.prepared.n_obs)

    monkeypatch.setattr(controller, "_source_batches", oversized)
    with pytest.raises(
        ValueError, match="initial source exceeds prospective batch cap"
    ):
        _theta_fit(stream, family)
    assert stream.source.scans == 1
