"""Actual bounded NB fits and accepted/trial provider attribution."""

import copy
import weakref
from dataclasses import replace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.control import EFSControl
from jaxgam.execution import efs_stream_provider
from jaxgam.execution.efs import _run_efs_known_scale
from jaxgam.execution.efs_provider import EFSFitRequest
from jaxgam.execution.efs_stream_provider import (
    NBStreamEFSProvider,
    RegularStreamEFSProvider,
)
from jaxgam.execution.nb_stream import fit_nb_streamed_pirls
from jaxgam.families.standard import Poisson
from jaxgam.fitting.family_execution import FamilyExecutionParameters
from jaxgam.formula.fitting_prepare import qr_penalty_roots
from tests.helpers import _AssertCollector, r_available
from tests.test_execution.test_nb_stream import _fixture
from tests.test_execution.test_nb_theta_stream import _pinned_controller
from tests.tolerances import STRICT


def _request(provider, estimated, *, absent=True):
    stream = provider.stream
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    public = stream.prepared.evaluate_batch(batch)
    constant = float(provider.family.link.link(np.mean(batch.y)))
    old = np.linalg.lstsq(public, np.full(len(batch.y), constant), rcond=None)[0]
    for block in stream.prepared.fitting.penalty_structure.blocks:
        sl = slice(block.start, block.stop)
        old[sl] = np.linalg.solve(block.transform.dense(), old[sl])
    return EFSFitRequest(
        jnp.full((provider.fitting.n_penalties,), np.log(0.35)),
        jnp.asarray(old),
        log_theta_start=jnp.asarray([np.log(0.7)]) if estimated else None,
        beta_old_init=jnp.asarray(old) if estimated else None,
        start_is_absent=absent,
    )


@pytest.fixture(
    scope="module",
    params=[
        (link, mode) for link in ("log", "identity", "sqrt") for mode in (False, True)
    ],
)
def fitted(request):
    link, estimated = request.param
    stream, family = _fixture(link, estimated=estimated, smooth=True)
    provider = NBStreamEFSProvider.create(
        stream,
        family,
        maximum_bytes=10_000_000,
        batch_rows=11,
        control=EFSControl(pirls_tolerance=1e-11, pirls_max_iter=100),
    )
    initial = _request(provider, estimated)
    original = efs_stream_provider.fit_nb_streamed_pirls
    captured = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    with patch.object(efs_stream_provider, "fit_nb_streamed_pirls", capture):
        fit = provider(initial)
    return provider, initial, fit, link, estimated, captured[0]


def test_actual_provider_matches_coefficient_controller_and_exact_statistics(fitted):
    provider, request, fit, _link, estimated, direct = fitted
    before = provider.stream.source.scans, provider.stream.source.batches
    direct = fit_nb_streamed_pirls(
        provider.stream,
        provider.family,
        np.asarray(request.log_lambda),
        maximum_bytes=provider.maximum_bytes - provider.provider_bytes,
        control=provider.control,
        parameters=FamilyExecutionParameters(
            request.log_theta_start
            if estimated
            else jnp.asarray(provider.lineage.parameters.log_theta)
        ),
        beta_start=np.asarray(request.beta_start),
        beta_old_init=np.asarray(request.beta_old_init) if estimated else None,
        start_is_absent=True,
        estimate_theta=estimated,
    )
    assert fit.valid
    assert direct.state.converged
    assert direct.state.source_scans == provider.stream.source.scans - before[0]
    assert direct.state.batches_scanned == provider.stream.source.batches - before[1]
    np.testing.assert_array_equal(
        fit.pirls_result.coefficients, direct.state.coefficients
    )
    np.testing.assert_array_equal(fit.score, direct.reml_score)
    assert fit.pirls_result.n_iter == direct.state.n_iter
    assert (
        int(fit.positive_curvature_retry_count) == direct.positive_observed_recoveries
    )
    if estimated:
        np.testing.assert_array_equal(fit.log_theta, direct.log_theta)
        assert int(fit.theta_n_iter) == direct.theta_n_iter
    else:
        assert fit.log_theta is None
    H = np.array(direct.state.xtwx_fisher)
    p = len(H)
    t, q = 0.0, 0.0
    for root in qr_penalty_roots(provider.stream.prepared.fitting.penalty_structure):
        B = np.zeros((p, len(root.root)))
        B[root.start : root.stop] = root.root.T
        H += np.exp(request.log_lambda[root.sp_index]) * B @ B.T
        q += np.sum((B.T @ np.asarray(direct.state.coefficients)) ** 2)
    for root in qr_penalty_roots(provider.stream.prepared.fitting.penalty_structure):
        B = np.zeros((p, len(root.root)))
        B[root.start : root.stop] = root.root.T
        t += np.sum(B * np.linalg.solve(H, B))
    np.testing.assert_allclose(
        [fit.statistics.fisher_trace[0], fit.statistics.quadratic[0]],
        [t, q],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert float(fit.score_phi) == float(fit.update_phi) == 1.0
    assert provider.provider_bytes > 8 * 16 * p * p
    assert provider.persistent_bytes <= provider.persistent_upper_bytes
    assert provider.fitting.family is provider.family


@pytest.mark.skipif(not r_available, reason="pinned R unavailable")
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_live_source_provider_score_theta_and_fisher_contractions(link):
    stream, family = _fixture(link, estimated=True, smooth=True)
    provider = NBStreamEFSProvider.create(
        stream,
        family,
        maximum_bytes=10_000_000,
        batch_rows=11,
        control=EFSControl(pirls_tolerance=1e-11, pirls_max_iter=100),
    )
    fit = provider(_request(provider, True))
    oracle = _pinned_controller(provider.stream, link, None)
    p = provider.fitting.n_coef
    collector = _AssertCollector()
    for name, actual, expected in (
        ("source score", fit.score, oracle["score"]),
        ("selected theta", fit.log_theta, oracle["theta"]),
        ("Fisher information", fit.pirls_result.xtwx_fisher, oracle["F"]),
    ):
        collector.check(
            name,
            lambda actual=actual, expected=expected: np.testing.assert_allclose(
                actual,
                expected,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
    t, q = 0.0, 0.0
    for root in qr_penalty_roots(provider.stream.prepared.fitting.penalty_structure):
        B = np.zeros((p, len(root.root)))
        B[root.start : root.stop] = root.root.T
        t += np.sum(B * (oracle["V"] @ B))
        q += np.sum((B.T @ oracle["beta"]) ** 2)
    collector.check(
        "source Fisher t/q",
        lambda: np.testing.assert_allclose(
            [fit.statistics.fisher_trace[0], fit.statistics.quadratic[0]],
            [t, q],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any("NB streamed EFS provider layer")


@pytest.mark.parametrize("field", ["rho", "phi", "theta", "generating_theta"])
def test_misattributed_fit_rejected_before_valid_state(monkeypatch, fitted, field):
    provider, request, fit, _link, _estimated, direct = fitted
    # Reuse one real reconverged result; inject the stale field at the
    # coefficient/provider boundary, never substitute a fabricated valid fit.
    if field == "rho":
        direct = replace(
            direct, state=replace(direct.state, log_lambda=request.log_lambda + 0.1)
        )
    elif field == "phi":
        direct = replace(
            direct, state=replace(direct.state, score_scale=jnp.asarray(2.0))
        )
    elif field == "theta":
        direct = replace(direct, log_theta=(direct.log_theta[0] + 0.1,))
    else:
        direct = replace(
            direct,
            source_deviance_log_theta=(direct.source_deviance_log_theta[0] + 0.1,),
        )
    monkeypatch.setattr(
        efs_stream_provider, "fit_nb_streamed_pirls", lambda *_args, **_kwargs: direct
    )
    with pytest.raises(RuntimeError, match="misattributed"):
        provider(request)
    assert fit.valid


@pytest.mark.parametrize(
    "field",
    ["source", "actual_source", "basis", "metadata_source", "metadata_basis", "family"],
)
def test_stale_lineage_rejected_before_dispatch(monkeypatch, fitted, field):
    provider, request, _fit_value, _link, _estimated, _direct = fitted
    if field == "source":
        stream = replace(
            provider.stream,
            prepared=replace(provider.stream.prepared, source_fingerprint="stale"),
        )
        provider = replace(provider, stream=stream)
    elif field == "actual_source":
        from types import SimpleNamespace

        stream = replace(
            provider.stream, source=SimpleNamespace(fingerprint=lambda: "stale")
        )
        provider = replace(provider, stream=stream)
    elif field == "basis":
        stream = replace(
            provider.stream,
            prepared=replace(provider.stream.prepared, basis_fingerprint="stale"),
        )
        provider = replace(provider, stream=stream)
    elif field.startswith("metadata"):
        name = "source_fingerprint" if field.endswith("source") else "basis_fingerprint"
        provider = replace(
            provider, fitting=replace(provider.fitting, **{name: "stale"})
        )
    else:
        family = copy.deepcopy(provider.family)
        family.put_theta(np.array([0.9]))
        provider = replace(provider, family=family)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("stale provider dispatched")

    monkeypatch.setattr(efs_stream_provider, "fit_nb_streamed_pirls", forbidden)
    with pytest.raises(RuntimeError):
        provider(request)


def test_source_change_during_fit_rejected_before_return(monkeypatch, fitted):
    provider, request, _fit_value, _link, _estimated, raw = fitted

    class ChangingSource:
        changed = False

        def fingerprint(self):
            return "stale" if self.changed else provider.lineage.source_fingerprint

    source = ChangingSource()
    provider = replace(provider, stream=replace(provider.stream, source=source))

    def changed_fit(*_args, **_kwargs):
        source.changed = True
        return raw

    monkeypatch.setattr(efs_stream_provider, "fit_nb_streamed_pirls", changed_fit)
    with pytest.raises(RuntimeError, match="changed during"):
        provider(request)


def test_actual_outer_controller_uses_streamed_provider_and_bounded_diagnostics():
    stream, family = _fixture("log", estimated=True, smooth=True)
    control = EFSControl(outer_limit=4, pirls_tolerance=1e-11, history_limit=3)
    provider = NBStreamEFSProvider.create(
        stream, family, maximum_bytes=10_000_000, batch_rows=11, control=control
    )
    snapshot = family.get_theta().copy()
    initial = _request(provider, True)
    before = stream.source.scans, stream.source.batches
    result = _run_efs_known_scale(provider, provider.context, initial, control)
    assert np.all(np.isfinite(result.pirls_result.coefficients))
    assert result.theta is not None
    assert len(result.score_history) <= control.history_limit
    assert result.optimizer_diagnostics.trace_method == "exact-streamed-fisher"
    assert (
        result.optimizer_diagnostics.provider_source_scans
        == stream.source.scans - before[0]
    )
    assert (
        result.optimizer_diagnostics.provider_batches_scanned
        == stream.source.batches - before[1]
    )
    np.testing.assert_array_equal(family.get_theta(), snapshot)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("winning_extension", [False, True])
def test_outer_cost_totals_include_rejected_and_extended_trials_or_remain_unknown(
    missing, winning_extension
):
    from jaxgam.execution.efs_provider import EFSControllerContext
    from tests.test_execution.test_efs_provider import _state

    scores = [10.0, 9.0, 8.0 if winning_extension else 9.1, 11.0, 10.5]
    scans = [2, 3, 5, 7, 11]
    states = []

    def provider(request):
        index = len(states)
        state = replace(
            _state(request, index, scores[index], 1.0),
            source_scans=scans[index],
            batches_scanned=None if missing and index == 2 else 4 * scans[index],
        )
        states.append(state)
        return state

    result = _run_efs_known_scale(
        provider,
        EFSControllerContext(),
        EFSFitRequest(jnp.zeros(1), jnp.zeros(2)),
        EFSControl(outer_limit=2, score_tolerance=0),
    )
    assert len(states) == (5 if winning_extension else 4)
    diagnostics = result.optimizer_diagnostics
    if missing:
        assert diagnostics.provider_source_scans is None
        assert diagnostics.provider_batches_scanned is None
    else:
        assert diagnostics.provider_source_scans == sum(scans[: len(states)])
        assert diagnostics.provider_batches_scanned == 4 * sum(scans[: len(states)])
        assert diagnostics.provider_source_scans != scans[len(states) - 1]


def test_outer_extension_retains_three_prior_fits_during_next_provider_call():
    """Charge actual controller lifetimes without changing source decisions."""
    from jaxgam.execution.efs_provider import EFSControllerContext
    from tests.test_execution.test_efs_provider import _state

    references = []
    scores = [10.0, 9.0, 9.1, 8.0, 8.1]

    def provider(request):
        index = len(references)
        if index == 4:
            # Previous accepted, losing extension, and this iteration's
            # candidate coexist while the next provider call is evaluated.
            assert sum(reference() is not None for reference in references) == 3
            assert all(references[i]() is not None for i in (1, 2, 3))
        state = _state(request, index, scores[index], 1.0)
        references.append(weakref.ref(state))
        return state

    _run_efs_known_scale(
        provider,
        EFSControllerContext(),
        EFSFitRequest(jnp.zeros(1), jnp.zeros(2)),
        EFSControl(outer_limit=2, score_tolerance=0),
    )
    assert len(references) == 5


def test_configured_outer_histories_are_preflighted_before_metadata(monkeypatch):
    stream, family = _fixture(smooth=True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("allocated metadata before outer-history preflight")

    monkeypatch.setattr(
        efs_stream_provider.PreparedFittingMetadata, "from_prepared", forbidden
    )
    with pytest.raises(MemoryError, match="known buffers"):
        NBStreamEFSProvider.create(
            stream,
            family,
            maximum_bytes=1_000_000,
            control=EFSControl(outer_limit=100_000, history_limit=100_000),
        )


@pytest.mark.parametrize("budget", [False, 0, 1.0, 1])
def test_provider_budget_rejects_before_scanning(budget):
    stream, family = _fixture(smooth=True)
    before = stream.source.scans
    with pytest.raises((ValueError, MemoryError)):
        NBStreamEFSProvider.create(stream, family, maximum_bytes=budget)
    assert stream.source.scans == before


def test_tiny_budget_preflights_before_metadata_transfer_or_root_build(monkeypatch):
    stream, family = _fixture(smooth=True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("allocated provider metadata before preflight")

    monkeypatch.setattr(
        efs_stream_provider.PreparedFittingMetadata, "from_prepared", forbidden
    )
    monkeypatch.setattr(efs_stream_provider, "prepare_efs_statistics", forbidden)
    with pytest.raises(MemoryError, match="known buffers"):
        NBStreamEFSProvider.create(stream, family, maximum_bytes=1)


def test_prospective_prior_allowance_charges_penalty_parameter_arrays():
    stream, _family = _fixture(smooth=True)
    original = efs_stream_provider._provider_memory_bounds(stream)
    block = stream.prepared.fitting.penalty_structure.blocks[0]
    count = 100
    many = replace(
        block,
        sp_indices=tuple(range(count)),
        local_penalties=block.local_penalties * count,
        ranks=block.ranks * count,
    )
    structure = replace(stream.prepared.fitting.penalty_structure, blocks=(many,))
    fitting = replace(stream.prepared.fitting, penalty_structure=structure)
    prepared = replace(stream.prepared, fitting=fitting)
    bound = efs_stream_provider._provider_memory_bounds(
        replace(stream, prepared=prepared)
    )
    assert bound[2] - original[2] == 8 * 32 * (count - 1)


def test_provider_type_and_request_validation_before_scanning(fitted):
    provider, request, _fit_value, _link, estimated, _direct = fitted
    before = provider.stream.source.scans
    with pytest.raises(TypeError, match="NegativeBinomial"):
        NBStreamEFSProvider.create(provider.stream, Poisson(), maximum_bytes=10_000_000)
    with pytest.raises(TypeError, match="EFSControl"):
        NBStreamEFSProvider.create(
            provider.stream, provider.family, maximum_bytes=10_000_000, control=True
        )
    invalid = [
        replace(request, log_lambda=jnp.asarray([np.nan])),
        replace(request, beta_start=jnp.zeros(1)),
        replace(request, score_phi=jnp.asarray(2.0)),
    ]
    if estimated:
        invalid.extend(
            [
                replace(request, log_theta_start=None),
                replace(request, beta_old_init=None),
            ]
        )
    else:
        invalid.append(replace(request, log_theta_start=jnp.asarray([0.9])))
    for candidate in invalid:
        with pytest.raises(ValueError, match=r"must|requires|finite|accept|positive"):
            provider(candidate)
    assert provider.stream.source.scans == before


@pytest.fixture(scope="module", params=["gaussian", "gamma", "poisson", "binomial"])
def regular_fitted(request):
    import pandas as pd

    from jaxgam.data.source import DataFrameRowSource
    from jaxgam.families.standard import Binomial, Gamma, Gaussian
    from jaxgam.formula.design_provider import StreamDesign
    from jaxgam.formula.parser import parse_formula
    from jaxgam.formula.prepare import prepare_model
    from tests.test_execution.test_regular_stream import _CountedSource

    family_class = {
        "gaussian": Gaussian,
        "gamma": Gamma,
        "poisson": Poisson,
        "binomial": Binomial,
    }[request.param]
    family = family_class()
    rng = np.random.default_rng(780132)
    x = np.linspace(-0.6, 0.7, 83)
    if request.param == "gaussian":
        y = 2 + x + 0.3 * np.cos(5 * x) + rng.normal(0, 0.05, len(x))
    elif request.param == "gamma":
        y = (2 + x + 0.3 * np.cos(5 * x)) * rng.uniform(0.6, 1.4, len(x))
    elif request.param == "poisson":
        y = rng.poisson(2 + x + 0.3 * np.cos(5 * x)).astype(float)
    else:
        y = rng.binomial(1, 0.5 + 0.1 * np.sin(x)).astype(float)
    data = pd.DataFrame({"x": x, "y": y})
    weight = 0.4 + rng.uniform(size=len(x))
    weight[::9] = 0.0
    source = DataFrameRowSource(
        data, response="y", weights=weight, offset=0.01 * np.sin(x)
    )
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=6)'), source, family=family
    )
    stream = StreamDesign(prepared, _CountedSource(source))
    provider = RegularStreamEFSProvider.create(
        stream,
        family,
        maximum_bytes=10_000_000,
        batch_rows=11,
        control=EFSControl(pirls_tolerance=1e-10),
    )
    request_value = EFSFitRequest(
        jnp.full((provider.fitting.n_penalties,), np.log(0.35)),
        jnp.zeros(provider.fitting.n_coef),
        score_phi=jnp.asarray(0.7) if not family.scale_known else None,
        regular_start_present=False,
    )
    original = efs_stream_provider.fit_regular_streamed_pirls
    captured = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    with patch.object(efs_stream_provider, "fit_regular_streamed_pirls", capture):
        fit = provider(request_value)
    return provider, request_value, fit, captured[0]


def test_regular_provider_preserves_raw_score_and_factor_provenance(regular_fitted):
    provider, request, fit, raw = regular_fitted
    assert fit.valid
    assert fit.pirls_result is raw.state
    assert float(fit.pre_gdi1_deviance) == raw.source_score.raw_deviance
    assert float(fit.gdi1_penalty) == raw.source_score.solve_penalty
    assert (
        float(fit.pre_gdi1_penalized_deviance)
        == raw.source_score.stopping_penalized_deviance
    )
    assert bool(fit.gdi1_candidate_valid) == raw.source_score.candidate_valid
    assert float(fit.score_phi) == raw.source_score.score_phi
    assert float(fit.update_phi) == raw.source_score.reported_phi
    assert float(fit.carried_phi) == float(fit.reported_phi)
    p = provider.fitting.n_coef
    covariance = np.asarray(
        raw.state.fisher_coefficient_factor.hessian_inverse(jnp.eye(p))
    )
    t, q = 0.0, 0.0
    for root in qr_penalty_roots(provider.stream.prepared.fitting.penalty_structure):
        B = np.zeros((p, len(root.root)))
        B[root.start : root.stop] = root.root.T
        t += np.sum(B * (covariance @ B))
        q += np.sum((B.T @ np.asarray(raw.state.coefficients)) ** 2)
    np.testing.assert_allclose(
        [fit.statistics.fisher_trace[0], fit.statistics.quadratic[0]],
        [t, q],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert provider.context.estimated_theta is False
    assert fit.log_theta is None
    assert int(fit.positive_curvature_retry_count) == raw.fisher_recoveries
    np.testing.assert_array_equal(fit.log_lambda, request.log_lambda)


@pytest.mark.parametrize("field", ["rho", "score_phi", "reported_phi"])
def test_regular_misattribution_rejected(monkeypatch, regular_fitted, field):
    provider, request, _fit_value, raw = regular_fitted
    if field == "rho":
        raw = replace(
            raw, state=replace(raw.state, log_lambda=raw.state.log_lambda + 0.1)
        )
    else:
        raw = replace(
            raw,
            source_score=replace(
                raw.source_score, **{field: getattr(raw.source_score, field) + 0.1}
            ),
        )
    monkeypatch.setattr(
        efs_stream_provider, "fit_regular_streamed_pirls", lambda *_args, **_kwargs: raw
    )
    with pytest.raises(RuntimeError, match="misattributed"):
        provider(request)


def test_unknown_phi_is_not_replaced_by_reported_phi(regular_fitted):
    provider, request, fit, raw = regular_fitted
    if provider.family.scale_known:
        assert float(fit.score_phi) == float(fit.update_phi) == 1.0
    else:
        assert float(fit.score_phi) == 0.7
        assert abs(float(fit.update_phi) - 0.7) > 0.05
        before = provider.stream.source.scans
        with pytest.raises(ValueError, match="explicit score_phi"):
            provider(replace(request, score_phi=None))
        assert provider.stream.source.scans == before
    assert raw.source_score.score_phi == float(fit.score_phi)


def test_reporting_fallback_keeps_candidate_score_penalty(monkeypatch, regular_fitted):
    provider, request, fit, raw = regular_fitted
    # The coefficient-controller's actual-source/controlled invalid-refit
    # gates own this branch. Here distinguish its payload from reporting beta
    # at the adapter seam without changing any fit/factor/response arrays.
    increase = 0.3
    changed = replace(
        raw,
        source_score=replace(
            raw.source_score,
            solve_penalty=raw.source_score.solve_penalty + increase,
            candidate_valid=False,
        ),
    )
    monkeypatch.setattr(
        efs_stream_provider,
        "fit_regular_streamed_pirls",
        lambda *_args, **_kwargs: changed,
    )
    returned = provider(request)
    assert returned.valid
    assert not bool(returned.gdi1_candidate_valid)
    np.testing.assert_array_equal(
        returned.pirls_result.coefficients, fit.pirls_result.coefficients
    )
    np.testing.assert_array_equal(
        returned.statistics.quadratic, fit.statistics.quadratic
    )
    np.testing.assert_allclose(
        returned.score - fit.score,
        increase / (2 * float(fit.score_phi)),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_regular_provider_request_and_family_mode_rejections(regular_fitted):
    provider, request, _fit_value, _raw = regular_fitted
    before = provider.stream.source.scans
    for candidate in (
        replace(request, log_theta_start=jnp.asarray([0.0])),
        replace(request, regular_start_present=1),
        replace(request, score_phi=jnp.asarray(-1.0)),
    ):
        with pytest.raises(ValueError, match=r"must|requires|finite|accept|positive"):
            provider(candidate)
    assert provider.stream.source.scans == before
    with pytest.raises(TypeError):
        RegularStreamEFSProvider.create(provider.stream, True, maximum_bytes=10_000_000)
    from jaxgam.families.negative_binomial import NegativeBinomial

    with pytest.raises(ValueError, match="no nuisance theta"):
        RegularStreamEFSProvider.create(
            provider.stream, NegativeBinomial(theta=2.7), maximum_bytes=10_000_000
        )
