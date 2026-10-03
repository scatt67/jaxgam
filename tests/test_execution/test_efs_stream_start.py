"""Source initial-sp/null-scale reductions and explicit first-fit requests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from jaxgam.control import EFSControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs_stream_provider import (
    NBStreamEFSProvider,
    RegularStreamEFSProvider,
)
from jaxgam.execution.efs_stream_start import prepare_stream_efs_start
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.fitting_prepare import initial_log_sp_from_diagonal
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from tests.helpers import _AssertCollector, r_available
from tests.test_execution.test_nb_stream import _fixture
from tests.test_execution.test_regular_stream import _CountedSource, _EmptyCountedSource
from tests.tolerances import STRICT


def _close(collector, label, actual, expected, tolerance):
    collector.check(
        label,
        lambda: np.testing.assert_allclose(
            actual, expected, rtol=tolerance.rtol, atol=tolerance.atol
        ),
    )


def _regular(family, *, no_intercept=False, empty=False, constant=False):
    x = np.linspace(-0.6, 0.7, 79)
    rng = np.random.default_rng(17019)
    if isinstance(family, Poisson):
        y = rng.poisson(2 + np.sin(x)).astype(float)
    elif isinstance(family, Binomial):
        y = rng.binomial(1, 0.5 + 0.1 * x).astype(float)
    else:
        y = 0.5 + 0.1 * np.sin(x) + rng.uniform(-0.05, 0.05, len(x))
    if constant:
        y[:] = 2
    weights = rng.uniform(0.5, 1.5, len(x))
    weights[::9] = 0
    source = DataFrameRowSource(
        pd.DataFrame({"x": x, "y": y}),
        response="y",
        weights=weights,
        offset=0.2 + 0.05 * x,
    )
    formula = 'y ~ 0 + s(x, bs="cr", k=6)' if no_intercept else 'y ~ s(x, bs="cr", k=6)'
    prepared = prepare_model(parse_formula(formula), source, family=family)
    return StreamDesign(
        prepared, (_EmptyCountedSource if empty else _CountedSource)(source)
    )


def _provider(stream, family, B=11):
    cls = (
        NBStreamEFSProvider if family.family_name == "nb" else RegularStreamEFSProvider
    )
    return cls.create(
        stream,
        family,
        maximum_bytes=10_000_000,
        batch_rows=B,
        control=EFSControl(pirls_tolerance=1e-11),
    )


@pytest.mark.parametrize("family", [Gaussian(), Gamma(), Poisson(), Binomial()])
@pytest.mark.parametrize("no_intercept", [False, True])
def test_regular_startup_matches_full_public_diagonal_and_null_projection(
    family, no_intercept
):
    stream = _regular(family, no_intercept=no_intercept)
    provider = _provider(stream, family)
    before = stream.source.scans, stream.source.batches
    start = prepare_stream_efs_start(provider)
    assert start.source_scans == stream.source.scans - before[0] == 2
    assert start.batches_scanned == stream.source.batches - before[1] == 16
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    X = stream.prepared.evaluate_batch(batch)
    response = family.execution_initial_response_cpu(batch.y, batch.weight)
    summary = family.finalize_execution_summary(
        family.execution_summary_from_batch(response, batch.weight, batch.valid)
    )
    initial = family.initial_working_state_cpu(
        batch.y, batch.weight, batch.valid, summary=summary
    )
    working = (
        batch.weight
        * family.link.mu_eta(initial.eta) ** 2
        / family.variance(initial.mustart)
    )
    expected = initial_log_sp_from_diagonal(
        np.sum(working[:, None] * X**2, axis=0), stream.prepared.penalties
    )
    beta = np.linalg.lstsq(
        X, np.full(len(X), family.link.initial_link_cpu(np.mean(response))), rcond=None
    )[0]
    for block in stream.prepared.fitting.penalty_structure.blocks:
        where = slice(block.start, block.stop)
        beta[where] = np.linalg.solve(block.transform.dense(), beta[where])
    null_deviance = float(
        family.dev_resids(
            response, np.full(len(response), np.mean(response)), batch.weight
        )
    )
    check = _AssertCollector()
    _close(check, "public initial.sp", start.log_lambda, expected, STRICT)
    _close(check, "unweighted null beta", start.null_coefficients, beta, STRICT)
    _close(check, "source null deviance", start.null_deviance, null_deviance, STRICT)
    _close(
        check,
        "incoming phi",
        start.score_phi,
        1 if family.scale_known else null_deviance / len(response) / 10,
        STRICT,
    )
    check.raise_if_any("streamed EFS startup")
    assert start.initial_weight_policy == "regular"
    assert not start.log_lambda.flags.writeable
    assert not start.null_coefficients.flags.writeable
    assert start.workspace_bytes + provider.provider_bytes <= provider.maximum_bytes
    request = start.initial_request(estimated_theta=False)
    np.testing.assert_array_equal(request.log_lambda, start.log_lambda + 2.5)
    np.testing.assert_array_equal(request.beta_start, start.null_coefficients)
    assert request.regular_start_present is False
    assert request.start_is_absent is True
    assert request.log_theta_start is None


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("estimated", [False, True])
def test_nb_startup_global_expected_selection_batch_invariance_and_theta_isolation(
    link, estimated
):
    stream, family = _fixture(link, estimated=estimated, smooth=True)
    original = family.get_theta().copy()
    outputs = []
    for B in (1, 11, 200):
        provider = _provider(stream, family, B)
        before = stream.source.scans, stream.source.batches
        start = prepare_stream_efs_start(provider)
        assert start.source_scans == stream.source.scans - before[0] == 2
        assert start.batches_scanned == stream.source.batches - before[1]
        assert start.initial_weight_policy == "expected-Dmu2"
        assert start.score_phi == 1
        assert start.log_theta == tuple(original)
        outputs.append(start)
        request = start.initial_request(estimated_theta=estimated)
        if estimated:
            np.testing.assert_array_equal(request.log_theta_start, original)
            np.testing.assert_array_equal(
                request.beta_old_init, start.null_coefficients
            )
        else:
            assert request.log_theta_start is None
    for start in outputs[1:]:
        np.testing.assert_allclose(
            start.log_lambda, outputs[0].log_lambda, rtol=STRICT.rtol, atol=STRICT.atol
        )
        np.testing.assert_allclose(
            start.null_coefficients,
            outputs[0].null_coefficients,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    np.testing.assert_array_equal(family.get_theta(), original)


def test_empty_batches_are_neutral_and_zero_prior_rows_remain_in_null_projection():
    family = Gaussian()
    stream = _regular(family, no_intercept=True, empty=True)
    provider = _provider(stream, family)
    start = prepare_stream_efs_start(provider)
    assert start.batches_scanned == 18
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    X = stream.prepared.evaluate_batch(batch)
    beta = np.linalg.lstsq(X, np.full(len(X), np.mean(batch.y)), rcond=None)[0]
    for block in stream.prepared.fitting.penalty_structure.blocks:
        where = slice(block.start, block.stop)
        beta[where] = np.linalg.solve(block.transform.dense(), beta[where])
    np.testing.assert_allclose(
        start.null_coefficients, beta, rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_startup_budget_rejects_before_source_or_qr(monkeypatch):
    import jaxgam.execution.efs_stream_start as module

    stream, family = _fixture(smooth=True)
    provider = replace(_provider(stream, family), maximum_bytes=1)
    before = stream.source.scans

    def forbidden(*_args, **_kwargs):
        raise AssertionError("QR allocated before startup preflight")

    monkeypatch.setattr(module, "qr_update", forbidden)
    with pytest.raises(MemoryError, match="startup known workspace"):
        prepare_stream_efs_start(provider)
    assert stream.source.scans == before


def test_startup_bound_covers_shared_balancing_dense_tuple_for_many_local_penalties():
    """Exercise actual expansions, rather than assume they are references."""
    import jaxgam.execution.efs_stream_start as module
    from jaxgam.penalties.structure import DiagonalPenalty

    stream, _family = _fixture(smooth=True)
    block = stream.prepared.penalties.blocks[0]
    count = 100
    many = replace(
        block,
        sp_indices=tuple(range(count)),
        local_penalties=tuple(
            DiagonalPenalty(np.linspace(1, 2, block.size)) for _ in range(count)
        ),
        ranks=(block.size,) * count,
    )
    structure = replace(stream.prepared.penalties, blocks=(many,))
    prepared = replace(stream.prepared, penalties=structure)
    bounded = replace(stream, prepared=prepared)
    # The shared routine constructs this entire tuple before its inner zip.
    expanded = many.dense_penalties()
    assert sum(value.nbytes for value in expanded) == 8 * count * block.size**2
    bytes_needed = module._startup_workspace_bytes(bounded, 1)
    old_vector_only_bound = 8 * (
        16 * prepared.n_coef**2
        + 32 * prepared.n_coef
        + 8 * prepared.n_coef
        + 64
        + 8 * count
    )
    assert bytes_needed - old_vector_only_bound >= sum(
        value.nbytes for value in expanded
    )
    rho = initial_log_sp_from_diagonal(np.ones(prepared.n_coef), structure)
    assert rho.shape == (count,)
    assert np.all(np.isfinite(rho))


def test_nonpositive_null_scale_is_rejected_without_clipping():
    family = Gaussian()
    provider = _provider(_regular(family, constant=True), family)
    with pytest.raises(ValueError, match="positive incoming scale"):
        prepare_stream_efs_start(provider)


def test_regular_first_request_has_no_estimated_theta_snapshot():
    family = Gaussian()
    start = prepare_stream_efs_start(_provider(_regular(family), family))
    with pytest.raises(ValueError, match="attributed log theta"):
        start.initial_request(estimated_theta=True)


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_nb_nonnegative_observed_start_weights_match_pinned_source(link):
    stream, family = _fixture(link, smooth=True)
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    source = DataFrameRowSource(
        pd.DataFrame({"x": batch.columns["x"], "y": batch.y + 1}),
        response="y",
        weights=batch.weight,
        offset=batch.offset,
    )
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=6)'), source, family=family
    )
    provider = _provider(StreamDesign(prepared, _CountedSource(source)), family)
    start = prepare_stream_efs_start(provider)
    assert start.initial_weight_policy == "observed-Dmu2"
    rho, beta, phi = _r_start(provider)
    check = _AssertCollector()
    _close(check, "observed initial.spg", start.log_lambda, rho, STRICT)
    _close(check, "observed null beta", start.null_coefficients, beta, STRICT)
    _close(check, "known scale", start.score_phi, phi, STRICT)
    check.raise_if_any("nonnegative NB observed initializer")


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize("kind", ["integer", "fractional", "finite_tail"])
def test_nb_raw_null_deviance_uses_source_pmax_and_finite_tail_ratio(kind):
    """Value layers do not enable a fractional-below-one coefficient route."""
    import jaxgam.execution.efs_stream_start as module

    ro = pytest.importorskip("rpy2.robjects")
    y = np.array([0.0, 1.0, 4.0])
    if kind == "fractional":
        y = np.array([0.25, 0.5, 0.75])
    mu = 1e20 if kind == "finite_tail" else 2.0
    prior = np.array([0.8, 1.2, 0.0])
    theta = 2.7
    function = ro.r("""function(y,mu,wt,theta) {
      fam <- mgcv::nb(theta=theta)
      sum(fam$dev.resids(y,rep(mu,length(y)),wt,log(theta)))
    }""")
    expected = float(function(ro.FloatVector(y), mu, ro.FloatVector(prior), theta)[0])
    actual = module._nb_null_deviance(y, mu, prior, theta)
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)


def _r_start(provider):
    ro = pytest.importorskip("rpy2.robjects")

    batch = next(provider.stream.source.source.scan(provider.stream.prepared.n_obs))
    prepared = provider.stream.prepared
    X = prepared.evaluate_batch(batch)

    def matrix(value):
        return ro.r.matrix(ro.FloatVector(value.ravel(order="F")), nrow=value.shape[0])

    penalties, ranks, offsets = [], [], []
    for block in prepared.penalties.blocks:
        for penalty, rank in zip(block.dense_penalties(), block.ranks, strict=True):
            penalties.append(matrix(penalty))
            ranks.append(rank)
            offsets.append(block.start + 1)
    family = provider.family
    name = "Gamma" if isinstance(family, Gamma) else family.family_name
    link = type(family.link).__name__.removesuffix("Link").lower()
    link = {"inversesquared": "1/mu^2"}.get(link, link)
    theta = float(np.exp(family.get_theta()[0])) if name == "nb" else 1
    function = ro.r("""function(X,y,w,S,rank,off,name,link,theta) {
      fam <- if(name=="nb") do.call(mgcv::nb,list(theta=theta,link=link)) else
             get(name)(link=stats::make.link(link))
      fam <- mgcv:::fix.family(fam)
      G <- list(X=X,y=y,w=w,n=length(y),family=fam)
      null <- mgcv:::get.null.coef(G)
      sp <- mgcv:::initial.spg(X,y,w,fam,S,rank,off)
      phi <- if(name %in% c("poisson","binomial","nb")) 1 else null$null.scale/10
      list(rho=log(sp),beta=null$null.coef,phi=phi)
    }""")
    result = function(
        matrix(X),
        ro.FloatVector(batch.y),
        ro.FloatVector(batch.weight),
        ro.ListVector([(str(i), value) for i, value in enumerate(penalties)]),
        ro.IntVector(ranks),
        ro.IntVector(offsets),
        name,
        link,
        theta,
    )
    beta = np.array(result.rx2("beta"))
    for block in prepared.fitting.penalty_structure.blocks:
        where = slice(block.start, block.stop)
        beta[where] = np.linalg.solve(block.transform.dense(), beta[where])
    return np.array(result.rx2("rho")), beta, float(result.rx2("phi")[0])


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize("family", [Gaussian(), Gamma(), Poisson(), Binomial()])
def test_regular_startup_matches_live_pinned_initial_spg_and_get_null_coef(family):
    provider = _provider(_regular(family, no_intercept=True), family)
    start = prepare_stream_efs_start(provider)
    rho, beta, phi = _r_start(provider)
    check = _AssertCollector()
    _close(check, "pinned initial.spg", start.log_lambda, rho, STRICT)
    _close(check, "pinned get.null.coef", start.null_coefficients, beta, STRICT)
    _close(check, "pinned null.scale/10", start.score_phi, phi, STRICT)
    check.raise_if_any("streamed EFS startup")


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_nb_startup_matches_live_pinned_global_Dmu2_selection(link):
    stream, family = _fixture(link, smooth=True)
    provider = _provider(stream, family)
    start = prepare_stream_efs_start(provider)
    rho, beta, phi = _r_start(provider)
    check = _AssertCollector()
    _close(check, "pinned initial.spg", start.log_lambda, rho, STRICT)
    _close(check, "pinned get.null.coef", start.null_coefficients, beta, STRICT)
    _close(check, "pinned known scale", start.score_phi, phi, STRICT)
    check.raise_if_any("streamed EFS startup")


_REGULAR_CELLS = [
    (cls, link)
    for cls in (Gaussian, Gamma, Poisson, Binomial)
    for link in (
        "identity",
        "log",
        "logit",
        "probit",
        "cloglog",
        "inverse",
        "inverse_squared",
        "sqrt",
    )
]


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize(("family_cls", "link"), _REGULAR_CELLS)
def test_all_regular_constructor_link_startup_or_source_input_boundary(
    family_cls, link
):
    """Record the bounded-count initializer boundary for this integer fixture."""
    family = family_cls(link=link)
    provider = _provider(_regular(family), family)
    if family_cls is Poisson and link in ("logit", "probit", "cloglog"):
        pytest.importorskip("rpy2.robjects")
        from rpy2.rinterface_lib.embedded import RRuntimeError

        with pytest.raises((ValueError, FloatingPointError)):
            prepare_stream_efs_start(provider)
        with pytest.raises(RRuntimeError):
            _r_start(provider)
        return
    start = prepare_stream_efs_start(provider)
    rho, beta, phi = _r_start(provider)
    check = _AssertCollector()
    _close(check, "constructor-link initial.spg", start.log_lambda, rho, STRICT)
    _close(check, "constructor-link null anchor", start.null_coefficients, beta, STRICT)
    _close(check, "constructor-link incoming phi", start.score_phi, phi, STRICT)
    check.raise_if_any("regular constructor-link startup")
