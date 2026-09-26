"""Source-timed host adjoints for the regular streamed controller."""

import sys
from collections import deque
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgam.execution.reml as reml_execution
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.reml import (
    StreamREMLControl,
    evaluate_regular_stream_reml,
    optimize_regular_stream_reml,
)
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.formula.design_provider import StreamDesign
from tests.test_execution.test_regular_penalized_matrix import (
    _APPROVED_DIGESTS,
    _fixture_digest,
    _rank_one_preparation,
    _source_reference,
)
from tests.test_execution.test_regular_starts import _LINKS, _case
from tests.tolerances import MODERATE, STRICT


class _CountedSource:
    def __init__(self, source):
        self.source = source
        self.scans = 0

    @property
    def n_rows(self):
        return self.source.n_rows

    def fingerprint(self):
        return self.source.fingerprint()

    def scan(self, batch_rows):
        self.scans += 1
        yield from self.source.scan(batch_rows)


def _problem(family_class, link):
    family, data, weight, offset, start = _case(family_class, link)
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared, rho = _rank_one_preparation(source, family)
    control = StreamPIRLSControl(
        batch_rows=17, tol=1e-7, max_iter=200, solver_policy="qr"
    )
    params = rho if family.scale_known else np.r_[rho, np.log(0.7)]
    return family, data, weight, offset, start, source, prepared, params, control


def _retained_numeric_bytes(value, seen=None):
    """Count distinct immutable NumPy/JAX array bytes in a result tree."""
    seen = set() if seen is None else seen
    if isinstance(value, (np.ndarray, jax.Array)):
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        return int(value.size * value.dtype.itemsize)
    if is_dataclass(value):
        return sum(
            _retained_numeric_bytes(getattr(value, field.name), seen)
            for field in fields(value)
        )
    if isinstance(value, (tuple, list)):
        return sum(_retained_numeric_bytes(item, seen) for item in value)
    return 0


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize("family_class", [Gaussian, Gamma, Poisson, Binomial])
@pytest.mark.parametrize("link", _LINKS)
def test_regular_source_gradient_matches_all_32_pinned_gdi_cells(family_class, link):
    (
        family,
        data,
        weight,
        offset,
        start,
        source,
        prepared,
        params,
        control,
    ) = _problem(family_class, link)
    X = np.column_stack([np.ones(len(data)), data.x])
    rho = params[:1]
    oracle = _source_reference(
        family,
        link,
        X,
        data.y,
        weight,
        offset,
        start,
        derivatives=True,
    )
    fields = oracle["reference"]
    _covariance = oracle["covariance"]
    expected = oracle["gradient"]
    trial = evaluate_regular_stream_reml(
        StreamDesign(prepared, source),
        family,
        params,
        maximum_bytes=64_000_000,
        control=control,
        initial_coefficients=start,
    )
    approved_boundary = (family_class, link) == (Binomial, "log")
    if approved_boundary:
        assert (
            _fixture_digest(
                family_class,
                link,
                X,
                data.y,
                weight,
                offset,
                start,
                prepared.fitting.penalty_structure,
                rho,
                prepared,
                control,
            )
            == _APPROVED_DIGESTS[(Binomial, "log")]
        )
        assert trial.alpha_roundoff_recovered
        assert trial.fit_result.state.n_iter == 4
        assert trial.fit_result.source_score.candidate_valid
    else:
        assert not trial.alpha_roundoff_recovered
    tolerance = MODERATE if approved_boundary else STRICT
    np.testing.assert_allclose(
        np.asarray(trial.gradient),
        expected,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
    )
    np.testing.assert_allclose(
        trial.score,
        fields[5],
        rtol=tolerance.rtol if approved_boundary else STRICT.rtol,
        atol=tolerance.atol if approved_boundary else STRICT.atol,
    )
    assert trial.source_factor_residual < STRICT.atol
    assert trial.observed_factor_residual < STRICT.atol


def test_regular_source_gradient_matches_refitted_scalar_finite_difference():
    family, _data, _weight, _offset, start, source, prepared, params, control = (
        _problem(Gaussian, "identity")
    )
    stream = StreamDesign(prepared, source)
    trial = evaluate_regular_stream_reml(
        stream,
        family,
        params,
        maximum_bytes=64_000_000,
        control=control,
        initial_coefficients=start,
    )
    step = 5e-5
    finite_difference = np.empty_like(params)
    for index in range(len(params)):
        direction = np.zeros_like(params)
        direction[index] = step
        plus = evaluate_regular_stream_reml(
            stream,
            family,
            params + direction,
            maximum_bytes=64_000_000,
            control=control,
            warm_start=trial,
        )
        minus = evaluate_regular_stream_reml(
            stream,
            family,
            params - direction,
            maximum_bytes=64_000_000,
            control=control,
            warm_start=trial,
        )
        finite_difference[index] = (float(plus.score) - float(minus.score)) / (
            2.0 * step
        )
    np.testing.assert_allclose(
        np.asarray(trial.gradient),
        finite_difference,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_regular_source_gradient_rejects_rank_deficit_before_scanning():
    family, _data, _weight, _offset, start, source, prepared, params, control = (
        _problem(Gaussian, "identity")
    )
    prepared = replace(
        prepared,
        fitting=replace(prepared.fitting, unpenalized_rank_deficit=1),
    )
    with pytest.raises(np.linalg.LinAlgError, match="full-rank"):
        evaluate_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            params,
            maximum_bytes=64_000_000,
            control=control,
            initial_coefficients=start,
        )


def test_regular_source_gradient_rejects_nonconverged_and_near_domain_states():
    family, _data, _weight, _offset, start, source, prepared, params, control = (
        _problem(Gamma, "inverse")
    )
    with pytest.raises(RuntimeError, match="converged"):
        evaluate_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            params,
            maximum_bytes=64_000_000,
            control=replace(control, max_iter=1),
            initial_coefficients=start,
        )
    with pytest.raises(ValueError, match="domain"):
        evaluate_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            params,
            maximum_bytes=64_000_000,
            control=control,
            initial_coefficients=np.array([-1e300, 0.0]),
        )


def test_regular_source_gradient_rejects_exact_success_and_factor_mismatch(
    monkeypatch,
):
    family, data, weight, offset, start = _case(Binomial, "log")
    data = data.copy()
    data["y"] = 1.0
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared, rho = _rank_one_preparation(source, family)
    control = StreamPIRLSControl(
        batch_rows=17, tol=1e-7, max_iter=200, solver_policy="qr"
    )
    with pytest.raises((RuntimeError, FloatingPointError)) as exact_failure:
        evaluate_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            rho,
            maximum_bytes=64_000_000,
            control=control,
            initial_coefficients=start,
        )
    assert "converged" in str(exact_failure.value) or "determinant" in str(
        exact_failure.value
    )

    (
        family,
        _data,
        _weight,
        _offset,
        start,
        source,
        prepared,
        params,
        control,
    ) = _problem(Gaussian, "identity")
    stream = StreamDesign(prepared, source)
    real = reml_execution.fit_regular_streamed_pirls(
        stream,
        family,
        params[:1],
        maximum_bytes=64_000_000,
        score_scale=np.exp(params[1]),
        control=control,
        initial_coefficients=start,
    )
    invalid_factor = replace(
        real.source_coefficient_factor,
        correction=jnp.zeros_like(real.source_coefficient_factor.correction),
    )

    def invalid_fit(*_args, **_kwargs):
        return replace(real, source_coefficient_factor=invalid_factor)

    monkeypatch.setattr(
        reml_execution,
        "fit_regular_streamed_pirls",
        invalid_fit,
    )
    with pytest.raises(np.linalg.LinAlgError, match="positive"):
        evaluate_regular_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=64_000_000,
            control=control,
            initial_coefficients=start,
        )


def test_regular_source_gradient_workspace_rejects_before_device_metadata(
    monkeypatch,
):
    family, _data, _weight, _offset, start, source, prepared, params, control = (
        _problem(Gaussian, "identity")
    )
    counted = _CountedSource(source)
    stream = StreamDesign(prepared, counted)
    metadata_calls = 0

    def forbidden_metadata(*_args, **_kwargs):
        nonlocal metadata_calls
        metadata_calls += 1
        raise AssertionError("device metadata must not be constructed")

    monkeypatch.setattr(
        reml_execution.PreparedFittingMetadata,
        "from_prepared",
        classmethod(forbidden_metadata),
    )
    with pytest.raises(MemoryError, match="workspace"):
        evaluate_regular_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=1,
            control=control,
            initial_coefficients=start,
        )
    assert metadata_calls == 0
    assert counted.scans == 0

    controller = reml_execution.preflight_regular_stream_workspace(
        stream, control, 64_000_000
    )
    with pytest.raises(MemoryError, match="Regular streamed REML needs"):
        evaluate_regular_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=controller.required_bytes,
            control=control,
            initial_coefficients=start,
        )
    assert metadata_calls == 0
    assert counted.scans == 0


def test_regular_source_gradient_rejects_underflowed_phi_before_device_metadata(
    monkeypatch,
):
    family, _data, _weight, _offset, start, source, prepared, params, control = (
        _problem(Gaussian, "identity")
    )
    counted = _CountedSource(source)
    stream = StreamDesign(prepared, counted)
    params = np.asarray(params, dtype=np.float64).copy()
    params[-1] = -1000.0
    metadata_calls = 0

    def forbidden_metadata(*_args, **_kwargs):
        nonlocal metadata_calls
        metadata_calls += 1
        raise AssertionError("device metadata must not be constructed")

    monkeypatch.setattr(
        reml_execution.PreparedFittingMetadata,
        "from_prepared",
        classmethod(forbidden_metadata),
    )
    with pytest.raises(ValueError, match="finite positive score scale"):
        evaluate_regular_stream_reml(
            stream,
            family,
            params,
            maximum_bytes=64_000_000,
            control=control,
            initial_coefficients=start,
        )
    assert metadata_calls == 0
    assert counted.scans == 0


def test_regular_stream_reml_optimizer_preserves_joint_scale_and_bounds_rho_only(
    monkeypatch,
):
    calls = []
    trial = SimpleNamespace(
        params=np.array([40.0, -100.0]),
        score=jnp.array(1.0),
        gradient=jnp.array([0.0, 0.0]),
        source_scans=3,
        batches_scanned=9,
        source_fingerprint="source",
        workspace=SimpleNamespace(required_bytes=128),
    )

    def evaluate(*_args, **kwargs):
        calls.append(kwargs)
        return trial

    def minimize_call(objective, params, **kwargs):
        score, gradient = objective(params)
        assert score == 1.0
        np.testing.assert_array_equal(gradient, np.zeros(2))
        assert kwargs["bounds"] == [(-40.0, 40.0), (None, None)]
        return SimpleNamespace(
            x=np.asarray(params),
            success=True,
            message="converged",
            status=0,
            nit=0,
        )

    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", evaluate)
    monkeypatch.setattr(reml_execution, "minimize", minimize_call)
    fitting = SimpleNamespace(
        penalty_structure=SimpleNamespace(n_penalties=1),
    )
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
        prepared=SimpleNamespace(fitting=fitting, n_coef=2),
    )
    family = SimpleNamespace(scale_known=False)
    result = optimize_regular_stream_reml(
        stream,
        family,
        np.array([41.0, -100.0]),
        maximum_bytes=np.int64(1_000_000),
        control=StreamREMLControl(history_size=2),
    )
    np.testing.assert_array_equal(result.trial.params, np.array([40.0, -100.0]))
    assert result.converged
    assert isinstance(calls[0]["maximum_bytes"], int)
    assert calls[0]["maximum_bytes"] < 1_000_000
    assert result.required_bytes < 1_000_000


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize(
    ("family_class", "link"),
    [(Gaussian, "identity"), (Poisson, "log")],
)
def test_regular_stream_reml_optimizer_converges_and_matches_final_source_refit(
    family_class, link
) -> None:
    family, data, weight, offset, start, source, prepared, params, control = _problem(
        family_class, link
    )
    counted = _CountedSource(source)
    result = optimize_regular_stream_reml(
        StreamDesign(prepared, counted),
        family,
        params,
        maximum_bytes=64_000_000,
        pirls_control=control,
        control=StreamREMLControl(
            max_iter=30,
            maxfun=100,
            gtol=1e-6,
            ftol=1e-14,
            history_size=8,
        ),
        initial_coefficients=start,
    )
    history = np.asarray(result.accepted_score_history)
    assert result.converged
    assert result.projected_gradient_inf <= 1e-6
    assert result.n_accepted == result.n_iter
    assert result.n_evaluations >= result.n_accepted + 1
    assert result.cumulative_source_scans == counted.scans
    assert result.cumulative_source_scans >= result.trial.source_scans
    assert result.required_bytes < 64_000_000
    assert np.all(np.diff(history) <= 1e-10)
    n_params = 1 + int(not family.scale_known)
    assert np.asarray(result.trial.params).shape == (n_params,)
    retained_bound, retained_history_bound, _ = (
        reml_execution._regular_optimizer_workspace_bytes(
            prepared.n_coef,
            n_params,
            StreamREMLControl(
                max_iter=30,
                maxfun=100,
                gtol=1e-6,
                ftol=1e-14,
                history_size=8,
            ),
            control,
        )
    )
    assert 0 < _retained_numeric_bytes(result.trial) <= retained_bound
    accepted_history = result.trial.fit_result.accepted_penalized_history
    accepted_history_bytes = sys.getsizeof(accepted_history) + sum(
        sys.getsizeof(value) for value in accepted_history
    )
    assert accepted_history_bytes <= retained_history_bound

    selected = np.asarray(result.trial.params)
    score_phi = 1.0 if family.scale_known else float(np.exp(selected[-1]))
    X = np.column_stack((np.ones(len(data)), data.x))
    oracle = _source_reference(
        family,
        link,
        X,
        data.y,
        weight,
        offset,
        start,
        rho=selected[:1],
        score_phi=score_phi,
    )
    fields = oracle["reference"]
    state = result.trial.fit_result.state
    for actual, expected in (
        (state.coefficients, fields[:2]),
        (state.deviance, fields[2]),
        (state.scale, fields[3]),
        (state.edf, fields[4]),
        (result.trial.score, fields[5]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)


@pytest.mark.parametrize("family_class", [Gaussian, Gamma, Poisson, Binomial])
@pytest.mark.parametrize("link", _LINKS)
def test_regular_stream_reml_optimizer_all_32_cells_report_iteration_limit(
    family_class, link
) -> None:
    family, _data, _weight, _offset, start, source, prepared, params, control = (
        _problem(family_class, link)
    )
    result = optimize_regular_stream_reml(
        StreamDesign(prepared, source),
        family,
        params,
        maximum_bytes=64_000_000,
        pirls_control=control,
        control=StreamREMLControl(
            max_iter=1,
            maxfun=12,
            gtol=1e-7,
            ftol=1e-14,
            history_size=2,
        ),
        initial_coefficients=start,
    )
    assert not result.converged
    assert result.status == 1
    assert result.n_iter == 1
    assert result.n_accepted == 1
    assert result.n_evaluations >= 2
    assert result.cumulative_source_scans >= result.trial.source_scans
    assert np.all(np.diff(result.accepted_score_history) <= 1e-10)
    expected_params = 1 + int(not family.scale_known)
    assert np.asarray(result.trial.params).shape == (expected_params,)


def test_regular_stream_reml_optimizer_rejects_retention_before_trial(
    monkeypatch,
) -> None:
    family, _data, _weight, _offset, start, source, prepared, params, control = (
        _problem(Gaussian, "identity")
    )
    evaluations = 0

    def forbidden(*_args, **_kwargs):
        nonlocal evaluations
        evaluations += 1
        raise AssertionError("trial evaluation must not start")

    monkeypatch.setattr(
        reml_execution,
        "evaluate_regular_stream_reml",
        forbidden,
    )
    with pytest.raises(MemoryError, match="retention"):
        optimize_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            params,
            maximum_bytes=1,
            pirls_control=control,
            initial_coefficients=start,
        )
    assert evaluations == 0


def test_regular_stream_reml_optimizer_charges_retained_inner_history(
    monkeypatch,
) -> None:
    family, _data, _weight, _offset, start, source, prepared, params, control = (
        _problem(Gaussian, "identity")
    )
    long_control = replace(control, max_iter=500)
    workspace_parts = reml_execution._regular_optimizer_workspace_bytes(
        prepared.n_coef,
        len(params),
        StreamREMLControl(history_size=3),
        long_control,
    )
    short_parts = reml_execution._regular_optimizer_workspace_bytes(
        prepared.n_coef,
        len(params),
        StreamREMLControl(history_size=3),
        replace(control, max_iter=1),
    )
    expected_delta = (500 - 1) * (sys.getsizeof(0.0) + np.dtype(np.intp).itemsize)
    assert workspace_parts[1] - short_parts[1] == expected_delta
    large_history = 10_000
    large_outer = StreamREMLControl(history_size=large_history)
    *_, large_lbfgs_bytes = reml_execution._regular_optimizer_workspace_bytes(
        prepared.n_coef,
        len(params),
        large_outer,
        control,
    )
    retained_scores = deque(
        (float(index) for index in range(large_history)), maxlen=large_history
    )
    returned_scores = tuple(retained_scores)
    actual_score_history_bytes = (
        sys.getsizeof(retained_scores)
        + sys.getsizeof(returned_scores)
        + sum(sys.getsizeof(value) for value in retained_scores)
    )
    assert actual_score_history_bytes <= large_lbfgs_bytes
    evaluations = 0

    def forbidden(*_args, **_kwargs):
        nonlocal evaluations
        evaluations += 1
        raise AssertionError("trial evaluation must not start")

    monkeypatch.setattr(
        reml_execution,
        "evaluate_regular_stream_reml",
        forbidden,
    )
    with pytest.raises(MemoryError, match="retention"):
        optimize_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            params,
            maximum_bytes=sum(workspace_parts),
            pirls_control=long_control,
            control=StreamREMLControl(history_size=3),
            initial_coefficients=start,
        )
    assert evaluations == 0


@pytest.mark.parametrize("maximum_bytes", [True, 0, -1, 1.5, np.nan])
def test_regular_stream_reml_optimizer_validates_budget_before_trial(
    monkeypatch, maximum_bytes
) -> None:
    family, _data, _weight, _offset, start, source, prepared, params, control = (
        _problem(Gaussian, "identity")
    )
    evaluations = 0

    def forbidden(*_args, **_kwargs):
        nonlocal evaluations
        evaluations += 1
        raise AssertionError("trial evaluation must not start")

    monkeypatch.setattr(
        reml_execution,
        "evaluate_regular_stream_reml",
        forbidden,
    )
    with pytest.raises(ValueError, match="positive integer"):
        optimize_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            params,
            maximum_bytes=maximum_bytes,
            pirls_control=control,
            initial_coefficients=start,
        )
    assert evaluations == 0


def test_regular_stream_reml_optimizer_does_not_accept_failed_trial(
    monkeypatch,
) -> None:
    initial = SimpleNamespace(
        params=np.array([0.0]),
        score=jnp.array(1.0),
        gradient=jnp.array([0.5]),
        source_scans=4,
        batches_scanned=12,
        source_fingerprint="source",
    )
    calls = 0

    def evaluate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return initial
        raise FloatingPointError("candidate overflow")

    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", evaluate)
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
    )
    objective = reml_execution._AcceptedRegularTrialObjective(
        stream,
        SimpleNamespace(),
        np.array([0.0]),
        StreamPIRLSControl(solver_policy="qr"),
        StreamREMLControl(),
        1_000_000,
        None,
        None,
    )
    with pytest.raises(FloatingPointError, match="candidate overflow"):
        objective(np.array([1.0]))
    assert objective.accepted is initial
    assert objective.candidate is None
    assert objective.n_evaluations == 1
    assert objective.n_accepted == 0
    assert objective.cumulative_source_scans == 4
    assert objective.cumulative_batches_scanned == 12


def test_regular_stream_reml_optimizer_keeps_accepted_on_line_search_failure(
    monkeypatch,
) -> None:
    initial = SimpleNamespace(
        params=np.array([0.0]),
        score=jnp.array(2.0),
        gradient=jnp.array([1.0]),
        source_scans=3,
        batches_scanned=9,
        source_fingerprint="source",
        workspace=SimpleNamespace(required_bytes=128),
    )
    candidate = SimpleNamespace(
        params=np.array([1.0]),
        score=jnp.array(1.0),
        gradient=jnp.array([0.5]),
        source_scans=5,
        batches_scanned=15,
        source_fingerprint="source",
        workspace=SimpleNamespace(required_bytes=128),
    )
    calls = iter((initial, candidate))
    monkeypatch.setattr(
        reml_execution,
        "evaluate_regular_stream_reml",
        lambda *_args, **_kwargs: next(calls),
    )

    def failed_line_search(objective, params, **_kwargs):
        objective(np.array([1.0]))
        return SimpleNamespace(
            x=np.asarray(params),
            success=False,
            message="line search failed",
            status=2,
            nit=0,
        )

    monkeypatch.setattr(reml_execution, "minimize", failed_line_search)
    fitting = SimpleNamespace(
        penalty_structure=SimpleNamespace(n_penalties=1),
    )
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
        prepared=SimpleNamespace(fitting=fitting, n_coef=2),
    )
    result = optimize_regular_stream_reml(
        stream,
        SimpleNamespace(scale_known=True),
        np.array([0.0]),
        maximum_bytes=1_000_000,
    )
    assert result.trial is initial
    assert not result.converged
    assert result.status == 2
    assert result.n_evaluations == 2
    assert result.n_accepted == 0
    assert result.cumulative_source_scans == 8
    assert result.cumulative_batches_scanned == 24


def test_regular_stream_reml_optimizer_rejects_source_mutation(monkeypatch) -> None:
    fingerprint = ["source"]
    initial = SimpleNamespace(
        params=np.array([0.0]),
        score=jnp.array(1.0),
        gradient=jnp.array([0.5]),
        source_scans=3,
        batches_scanned=9,
        source_fingerprint="source",
    )
    monkeypatch.setattr(
        reml_execution,
        "evaluate_regular_stream_reml",
        lambda *_args, **_kwargs: initial,
    )
    objective = reml_execution._AcceptedRegularTrialObjective(
        SimpleNamespace(
            source=SimpleNamespace(fingerprint=lambda: fingerprint[0]),
        ),
        SimpleNamespace(),
        np.array([0.0]),
        StreamPIRLSControl(solver_policy="qr"),
        StreamREMLControl(),
        1_000_000,
        None,
        None,
    )
    fingerprint[0] = "changed"
    with pytest.raises(RuntimeError, match="RowSource changed"):
        objective(np.array([0.0]))
