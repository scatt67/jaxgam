"""Source-timed host adjoints for the regular streamed controller."""

import hashlib
import sys
from collections import deque
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgam.execution.stream.reml as reml_execution
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.stream.pirls import StreamPIRLSControl
from jaxgam.execution.stream.reml import (
    StreamREMLControl,
    evaluate_regular_stream_reml,
    optimize_regular_stream_reml,
)
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
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


def _problem(family_class, link):
    family, data, weight, offset, start = _case(family_class, link)
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared, rho = _rank_one_preparation(source, family)
    control = StreamPIRLSControl(
        batch_rows=17, tol=1e-7, max_iter=200, solver_policy="qr"
    )
    params = rho if family.scale_known else np.r_[rho, np.log(0.7)]
    return family, data, weight, offset, start, source, prepared, params, control


def _coupled_penalty_problem():
    """Two overlapping rank-one penalties with one retained null direction."""
    family, data, weight, offset, start = _case(Poisson, "log")
    x = np.asarray(data.x)
    data = data.assign(z=np.sin(2.3 * x) + 0.17 * x**2)
    start = np.r_[start, 0.0]
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula("y~x+z"), source, family=family)
    first_direction = np.array([0.0, 1.0, 0.0])
    second_direction = np.array([0.0, 1.0, -1.0])
    structure = PenaltyStructure(
        n_coef=3,
        blocks=(
            PenaltyBlock(
                start=0,
                stop=3,
                sp_indices=(0, 1),
                local_penalties=(
                    DenseLocalPenalty(np.outer(first_direction, first_direction)),
                    DenseLocalPenalty(np.outer(second_direction, second_direction)),
                ),
                transform=IdentityTransform(3),
                ranks=(1, 1),
            ),
        ),
    )
    fitting = replace(
        prepared.fitting,
        penalty_structure=structure,
        log_lambda_init=np.zeros(2),
        total_penalty_rank=2,
        total_penalty_null_dim=1,
    )
    fingerprint = hashlib.sha256(
        (prepared.basis_fingerprint + ":coupled-rank-one-penalties").encode()
    ).hexdigest()
    prepared = replace(
        prepared,
        penalties=structure,
        fitting=fitting,
        basis_fingerprint=fingerprint,
    )
    return family, start, source, prepared


def _rotated_null_penalty_problem():
    """Dense rank-one penalty whose two-dimensional nullspace is not axial."""
    family, data, weight, offset, start = _case(Poisson, "log")
    x = np.asarray(data.x)
    data = data.assign(z=np.sin(2.3 * x) + 0.17 * x**2)
    start = np.r_[start, 0.0]
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula("y~x+z"), source, family=family)
    penalized_direction = np.ones(3) / np.sqrt(3.0)
    dense_penalty = np.outer(penalized_direction, penalized_direction)
    structure = PenaltyStructure(
        n_coef=3,
        blocks=(
            PenaltyBlock(
                start=0,
                stop=3,
                sp_indices=(0,),
                local_penalties=(DenseLocalPenalty(dense_penalty),),
                transform=IdentityTransform(3),
                ranks=(1,),
            ),
        ),
    )
    fitting = replace(
        prepared.fitting,
        penalty_structure=structure,
        log_lambda_init=np.zeros(1),
        total_penalty_rank=1,
        total_penalty_null_dim=2,
    )
    fingerprint = hashlib.sha256(
        (prepared.basis_fingerprint + ":dense-rotated-null-penalty").encode()
    ).hexdigest()
    prepared = replace(
        prepared,
        penalties=structure,
        fitting=fitting,
        basis_fingerprint=fingerprint,
    )
    return family, start, source, prepared, penalized_direction


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
    approved_fit_boundary = (family_class, link) in _APPROVED_DIGESTS
    if approved_fit_boundary:
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
            == _APPROVED_DIGESTS[(family_class, link)]
        )
    if approved_boundary:
        assert trial.alpha_roundoff_recovered
        assert trial.fit_result.state.n_iter == 4
        assert trial.fit_result.source_score.candidate_valid
        np.testing.assert_array_equal(oracle["status"], [1.0, 4.0])
        np.testing.assert_array_equal(oracle["candidate_valid"], [1.0])
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

    state = trial.fit_result.state
    actual_covariance = np.asarray(
        state.fisher_coefficient_factor.hessian_inverse(jnp.eye(prepared.n_coef))
    )
    actual_mu = np.asarray(
        family.link.inverse(X @ np.asarray(state.coefficients) + offset)
    )
    expected_mu = np.asarray(family.link.inverse(X @ fields[:2] + offset))
    actual_se = np.sqrt(
        float(state.scale) * np.einsum("ij,jk,ik->i", X, actual_covariance, X)
    )
    expected_se = np.sqrt(fields[3] * np.einsum("ij,jk,ik->i", X, _covariance, X))
    fit_tolerance = MODERATE if approved_fit_boundary else STRICT
    for actual, expected in (
        (state.coefficients, fields[:2]),
        (actual_mu, expected_mu),
        (actual_covariance, _covariance),
        (actual_se, expected_se),
    ):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=fit_tolerance.rtol,
            atol=fit_tolerance.atol,
        )
    for actual, expected in (
        (state.deviance, fields[2]),
        (state.scale, fields[3]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    # Reuse the coefficient layer's reviewed native-AMD64 EDF allowance for
    # this same digest-guarded Binomial/log state after four source corrections.
    # Deviance, scale and every other family's EDF remain STRICT.
    edf_tolerance = MODERATE if approved_boundary else STRICT
    np.testing.assert_allclose(
        state.edf, fields[4], rtol=edf_tolerance.rtol, atol=edf_tolerance.atol
    )
    assert trial.source_factor_residual < STRICT.atol
    assert trial.observed_factor_residual < STRICT.atol


def test_regular_derivative_timer_starts_after_real_fit_device_barrier(
    monkeypatch,
) -> None:
    family, _data, _weight, _offset, start, source, prepared, params, control = (
        _problem(Gaussian, "identity")
    )
    events = []
    actual_fit = reml_execution.fit_regular_streamed_pirls
    actual_sync = reml_execution._synchronize_fitted_state
    actual_block = jax.block_until_ready
    actual_clock = reml_execution.time.perf_counter
    actual_cotangents = reml_execution.regular_score_cotangents

    def fit(*args, **kwargs):
        result = actual_fit(*args, **kwargs)
        events.append("fit_returned")
        return result

    def synchronize(*args):
        assert events == ["fit_returned"]
        state, source_factor = args
        required = {
            id(value)
            for field in fields(state)
            if isinstance(value := getattr(state, field.name), jax.Array)
        }
        for factor in (
            state.coefficient_factor,
            state.fisher_coefficient_factor,
            source_factor,
        ):
            required.update(id(leaf) for leaf in jax.tree_util.tree_leaves(factor))

        def block(values):
            observed = {id(leaf) for leaf in jax.tree_util.tree_leaves(values)}
            assert required <= observed
            return actual_block(values)

        with monkeypatch.context() as scoped:
            scoped.setattr(jax, "block_until_ready", block)
            actual_sync(*args)
        events.append("fit_synchronized")

    def clock():
        events.append("timer")
        return actual_clock()

    def cotangents(*args, **kwargs):
        assert events == ["fit_returned", "fit_synchronized", "timer"]
        events.append("derivative")
        return actual_cotangents(*args, **kwargs)

    monkeypatch.setattr(reml_execution, "fit_regular_streamed_pirls", fit)
    monkeypatch.setattr(reml_execution, "_synchronize_fitted_state", synchronize)
    monkeypatch.setattr(reml_execution, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(reml_execution, "regular_score_cotangents", cotangents)
    trial = evaluate_regular_stream_reml(
        StreamDesign(prepared, source),
        family,
        params,
        maximum_bytes=64_000_000,
        control=control,
        initial_coefficients=start,
    )

    assert events == [
        "fit_returned",
        "fit_synchronized",
        "timer",
        "derivative",
        "timer",
    ]
    assert trial.derivative_elapsed_s > 0.0


@pytest.mark.parametrize("family_class", [Gaussian, Gamma, Poisson, Binomial])
@pytest.mark.parametrize("link", _LINKS)
def test_regular_source_gradient_matches_all_32_refitted_finite_differences(
    family_class, link
):
    family, data, weight, offset, start, source, prepared, params, control = _problem(
        family_class, link
    )
    tighter_cells = {
        (Gaussian, "probit"),
        (Gaussian, "inverse_squared"),
        (Poisson, "log"),
        (Poisson, "inverse_squared"),
        (Binomial, "inverse"),
        (Binomial, "inverse_squared"),
    }
    if (family_class, link) in tighter_cells:
        control = replace(control, tol=1e-9)
    elif (family_class, link) == (Poisson, "inverse"):
        control = replace(control, tol=1e-10)
    stream = StreamDesign(prepared, source)
    trial = evaluate_regular_stream_reml(
        stream,
        family,
        params,
        maximum_bytes=64_000_000,
        control=control,
        initial_coefficients=start,
    )
    approved_boundary = (family_class, link) in {
        (Poisson, "identity"),
        (Binomial, "log"),
    }
    if approved_boundary:
        X = np.column_stack((np.ones(len(data)), data.x))
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
                params[:1],
                prepared,
                control,
            )
            == _APPROVED_DIGESTS[(family_class, link)]
        )
        assert trial.fit_result.state.n_iter == 4
        assert trial.fit_result.source_score.candidate_valid
    step = {
        (Poisson, "identity"): 1e-3,
        (Binomial, "log"): 1e-2,
    }.get((family_class, link), 1e-3)
    finite_difference = np.empty_like(params)
    for index in range(len(params)):
        scores = []
        for multiplier in (-2.0, -1.0, 1.0, 2.0):
            direction = np.zeros_like(params)
            direction[index] = multiplier * step
            evaluated = evaluate_regular_stream_reml(
                stream,
                family,
                params + direction,
                maximum_bytes=64_000_000,
                control=control,
                warm_start=trial,
            )
            scores.append(float(evaluated.score))
        finite_difference[index] = (
            scores[0] - 8.0 * scores[1] + 8.0 * scores[2] - scores[3]
        ) / (12.0 * step)
    tolerance = MODERATE if approved_boundary else STRICT
    np.testing.assert_allclose(
        np.asarray(trial.gradient),
        finite_difference,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
    )
    if approved_boundary:
        assert np.all(
            np.abs(np.asarray(trial.gradient) - finite_difference) < tolerance.atol
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


def test_regular_stream_reml_step_windows_reopen_from_exact_accepted_trial(
    monkeypatch,
) -> None:
    evaluated: list[float] = []

    def evaluate(_stream, _family, params, **_kwargs):
        rho = float(np.asarray(params)[0])
        evaluated.append(rho)
        return SimpleNamespace(
            params=np.asarray(params).copy(),
            score=jnp.array((rho - 12.0) ** 2),
            gradient=jnp.array([2.0 * (rho - 12.0)]),
            source_scans=2,
            batches_scanned=4,
            source_fingerprint="source",
            workspace=SimpleNamespace(required_bytes=128),
        )

    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", evaluate)
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
        prepared=SimpleNamespace(
            fitting=SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1)),
            n_coef=2,
        ),
    )
    result = optimize_regular_stream_reml(
        stream,
        SimpleNamespace(scale_known=True),
        np.array([0.0]),
        maximum_bytes=1_000_000,
        max_step=5.0,
    )
    assert result.converged
    assert result.message.endswith("3 source step window(s).")
    np.testing.assert_allclose(
        result.trial.params, [12.0], rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert max(evaluated) <= 15.0
    assert result.cumulative_source_scans == 2 * len(evaluated)
    assert result.cumulative_batches_scanned == 4 * len(evaluated)
    assert result.n_evaluations == len(evaluated)
    assert result.n_accepted >= 3
    assert result.required_bytes < 1_000_000


def test_regular_stream_reml_reopens_artificial_edge_with_other_free_gradient(
    monkeypatch,
) -> None:
    """A free scale gradient must not strand rho at a source step cap."""
    accepted: list[np.ndarray] = []
    window_bounds: list[list[tuple[float | None, float | None]]] = []

    def evaluate(_stream, _family, params, **_kwargs):
        rho, log_phi = np.asarray(params)
        return SimpleNamespace(
            params=np.asarray(params).copy(),
            score=jnp.array((rho - 12.0) ** 2 + (log_phi - 1.0) ** 2),
            gradient=jnp.array([2.0 * (rho - 12.0), 2.0 * (log_phi - 1.0)]),
            source_scans=2,
            batches_scanned=4,
            source_fingerprint="source",
            workspace=SimpleNamespace(required_bytes=128),
        )

    def minimize_call(objective, _center, *, bounds, callback, **_kwargs):
        window_bounds.append(bounds)
        rho = min(float(bounds[0][1]), 12.0)
        # The first artificial rho edge has a residual free-scale gradient
        # above gtol, as in the official 100,000-row Gaussian fit.
        log_phi = 1.0 + 4.6e-6 if not accepted else 1.0
        point = np.array([rho, log_phi])
        objective(point)
        callback(SimpleNamespace(x=point))
        accepted.append(point)
        return SimpleNamespace(
            x=point,
            success=True,
            message="relative reduction",
            status=0,
            nit=1,
        )

    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", evaluate)
    monkeypatch.setattr(reml_execution, "minimize", minimize_call)
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
        prepared=SimpleNamespace(
            fitting=SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1)),
            n_coef=2,
        ),
    )
    result = optimize_regular_stream_reml(
        stream,
        SimpleNamespace(scale_known=False),
        np.array([0.0, 1.0]),
        maximum_bytes=1_000_000,
        max_step=5.0,
    )
    assert result.converged
    assert result.message.endswith("3 source step window(s).")
    assert window_bounds[0][0] == (-5.0, 5.0)
    np.testing.assert_allclose(result.trial.params, [12.0, 1.0])
    assert result.projected_gradient_inf == 0.0


def _quadratic_window_case(monkeypatch, optimum: float):
    """Exact-trial stub with optimum ``optimum`` and an edge-walking minimizer.

    The stub ``minimize`` spends its full per-window iteration budget, moves
    to the nearer of its window edge and the optimum, and records the
    options it was given, so the test sees each window's budget.
    """
    evaluated: list[float] = []
    windows: list[dict[str, object]] = []

    def evaluate(_stream, _family, params, **_kwargs):
        rho = float(np.asarray(params)[0])
        evaluated.append(rho)
        return SimpleNamespace(
            params=np.asarray(params).copy(),
            score=jnp.array((rho - optimum) ** 2),
            gradient=jnp.array([2.0 * (rho - optimum)]),
            source_scans=2,
            batches_scanned=4,
            source_fingerprint="source",
            workspace=SimpleNamespace(required_bytes=128),
        )

    def minimize_call(objective, _center, *, bounds, callback, options, **_kwargs):
        windows.append(dict(options))
        point = np.array([min(float(bounds[0][1]), optimum)])
        objective(point)
        callback(SimpleNamespace(x=point))
        return SimpleNamespace(
            x=point,
            success=True,
            message="CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
            status=0,
            nit=options["maxiter"],
        )

    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", evaluate)
    monkeypatch.setattr(reml_execution, "minimize", minimize_call)
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
        prepared=SimpleNamespace(
            fitting=SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1)),
            n_coef=2,
        ),
    )
    return stream, evaluated, windows


def test_regular_stream_reml_each_step_window_has_its_own_budget(
    monkeypatch,
) -> None:
    """Every window gets the full max_iter/maxfun; none is starved."""
    stream, evaluated, windows = _quadratic_window_case(monkeypatch, 37.0)
    control = StreamREMLControl(max_iter=3, maxfun=7)
    result = optimize_regular_stream_reml(
        stream,
        SimpleNamespace(scale_known=True),
        np.array([0.0]),
        maximum_bytes=1_000_000,
        control=control,
        max_step=5.0,
    )
    # 37 is reached in the eighth window; each spends its whole 3-iteration
    # budget, which a single shared budget of 3 would have exhausted at once.
    assert len(windows) == 8
    assert all(w["maxiter"] == 3 and w["maxfun"] == 7 for w in windows)
    assert result.converged
    assert result.message.endswith("8 source step window(s).")
    assert result.n_iter == 8 * 3
    np.testing.assert_array_equal(result.trial.params, [37.0])
    assert result.n_evaluations == len(evaluated)
    assert result.cumulative_source_scans == 2 * len(evaluated)
    assert result.cumulative_batches_scanned == 4 * len(evaluated)


def test_regular_stream_reml_window_cap_stops_truthfully(monkeypatch) -> None:
    """Descent past the last allowed window is reported, never converged."""
    stream, evaluated, windows = _quadratic_window_case(monkeypatch, 30.0)
    result = optimize_regular_stream_reml(
        stream,
        SimpleNamespace(scale_known=True),
        np.array([0.0]),
        maximum_bytes=1_000_000,
        max_step=1.0,
    )
    cap = reml_execution.MAX_SOURCE_STEP_WINDOWS
    assert len(windows) == cap
    assert not result.converged
    assert result.status == 0
    assert f"{cap} source step window(s). Window limit {cap} reached" in (
        result.message
    )
    assert "exceeds gtol" in result.message
    np.testing.assert_array_equal(result.trial.params, [float(cap)])
    assert result.projected_gradient_inf == 2.0 * (30.0 - cap)
    assert result.cumulative_source_scans == 2 * len(evaluated)


def test_regular_stream_reml_roundoff_window_stalls_truthfully(monkeypatch) -> None:
    calls = 0

    def evaluate(_stream, _family, params, **_kwargs):
        return SimpleNamespace(
            params=np.asarray(params).copy(),
            score=jnp.array(1.0),
            gradient=jnp.array([1.0]),
            source_scans=2,
            batches_scanned=4,
            source_fingerprint="source",
            workspace=SimpleNamespace(required_bytes=128),
        )

    def stationary_on_rounded_window(_objective, params, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["bounds"] == [(40.0, 40.0)]
        return SimpleNamespace(
            x=np.asarray(params),
            success=True,
            message="local projected gradient",
            status=0,
            nit=0,
        )

    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", evaluate)
    monkeypatch.setattr(reml_execution, "minimize", stationary_on_rounded_window)
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
        prepared=SimpleNamespace(
            fitting=SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1)),
            n_coef=2,
        ),
    )
    result = optimize_regular_stream_reml(
        stream,
        SimpleNamespace(scale_known=True),
        np.array([40.0]),
        maximum_bytes=1_000_000,
        max_step=1e-300,
    )
    assert calls == 1
    assert not result.converged
    assert result.projected_gradient_inf == 1.0
    assert result.n_evaluations == 1
    assert result.cumulative_source_scans == 2


@pytest.mark.parametrize("max_step", [0.0, np.inf, True])
def test_regular_stream_reml_step_window_rejects_invalid_before_source_scan(
    max_step,
) -> None:
    stream = SimpleNamespace(
        prepared=SimpleNamespace(
            fitting=SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1)),
            n_coef=2,
        ),
    )
    with pytest.raises(ValueError, match="max_step must be finite and positive"):
        optimize_regular_stream_reml(
            stream,
            SimpleNamespace(scale_known=True),
            np.array([0.0]),
            maximum_bytes=1_000_000,
            max_step=max_step,
        )


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize(
    ("family_class", "link"),
    [(Gaussian, "identity"), (Poisson, "log")],
)
def test_regular_stream_reml_optimizer_converges_and_matches_final_source_refit(
    family_class, link, monkeypatch
) -> None:
    family, data, weight, offset, start, source, prepared, params, control = _problem(
        family_class, link
    )
    counted = _CountedSource(source)
    actual_fit = reml_execution.fit_regular_streamed_pirls
    evaluated_inner_iterations = []

    def counted_fit(*args, **kwargs):
        fit = actual_fit(*args, **kwargs)
        evaluated_inner_iterations.append(fit.state.n_iter)
        return fit

    monkeypatch.setattr(reml_execution, "fit_regular_streamed_pirls", counted_fit)
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
    assert result.cumulative_inner_iterations == sum(evaluated_inner_iterations)
    assert result.cumulative_final_refit_solves == len(evaluated_inner_iterations)
    assert len(evaluated_inner_iterations) == result.n_evaluations
    assert result.cumulative_inner_iterations >= result.trial.fit_result.state.n_iter
    assert result.cumulative_derivative_s > 0.0
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
    curvature_bytes = 8 * (len(params) ** 2 + 24 * len(params))
    with pytest.raises(MemoryError, match="retention"):
        optimize_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            params,
            maximum_bytes=sum(workspace_parts) + curvature_bytes,
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


@pytest.mark.parametrize("max_step", [None, 5.0])
def test_regular_stream_reml_optimizer_keeps_accepted_on_line_search_failure(
    monkeypatch,
    max_step,
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
        max_step=max_step,
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


def test_regular_stream_reml_coupled_penalties_are_partition_invariant() -> None:
    """Exercise overlapping penalties, their gradient, and outer optimization."""
    family, start, source, prepared = _coupled_penalty_problem()
    fixed_trials = {}
    optimized = []
    for batch_rows in (17, 200):
        control = StreamPIRLSControl(
            solver_policy="qr",
            batch_rows=batch_rows,
            max_iter=100,
            tol=1e-9,
        )
        for rho in (np.zeros(2), np.full(2, 30.0)):
            counted = _CountedSource(source)
            trial = evaluate_regular_stream_reml(
                StreamDesign(prepared, counted),
                family,
                rho,
                maximum_bytes=64_000_000,
                control=control,
                initial_coefficients=start,
            )
            assert trial.source_scans == counted.scans
            assert trial.batches_scanned == counted.batches
            assert trial.source_factor_residual < STRICT.rtol
            assert trial.observed_factor_residual < STRICT.rtol
            fixed_trials[(batch_rows, float(rho[0]))] = trial

        counted = _CountedSource(source)
        result = optimize_regular_stream_reml(
            StreamDesign(prepared, counted),
            family,
            np.zeros(2),
            maximum_bytes=64_000_000,
            pirls_control=control,
            control=StreamREMLControl(
                max_iter=40,
                maxfun=100,
                gtol=1e-6,
                ftol=1e-14,
                history_size=16,
            ),
            initial_coefficients=start,
        )
        assert result.converged
        assert result.n_iter == result.n_accepted > 0
        assert result.n_evaluations >= result.n_accepted + 1
        assert result.projected_gradient_inf <= 1e-6
        assert result.cumulative_source_scans == counted.scans
        assert result.cumulative_batches_scanned == counted.batches
        assert np.all(np.diff(result.accepted_score_history) <= STRICT.atol)
        optimized.append(result)

    for rho in (0.0, 30.0):
        left = fixed_trials[(17, rho)]
        right = fixed_trials[(200, rho)]
        for left_value, right_value in (
            (left.score, right.score),
            (left.gradient, right.gradient),
            (
                left.fit_result.state.coefficients,
                right.fit_result.state.coefficients,
            ),
            (left.fit_result.state.deviance, right.fit_result.state.deviance),
            (left.fit_result.state.edf, right.fit_result.state.edf),
        ):
            np.testing.assert_allclose(
                left_value,
                right_value,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            )

    left_result, right_result = optimized
    assert left_result.n_iter == right_result.n_iter
    assert left_result.n_evaluations == right_result.n_evaluations
    for left_value, right_value in (
        (left_result.trial.params, right_result.trial.params),
        (left_result.trial.score, right_result.trial.score),
        (left_result.trial.gradient, right_result.trial.gradient),
        (
            left_result.trial.fit_result.state.coefficients,
            right_result.trial.fit_result.state.coefficients,
        ),
        (left_result.accepted_score_history, right_result.accepted_score_history),
    ):
        np.testing.assert_allclose(
            left_value,
            right_value,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


def test_regular_stream_reml_coupled_gradient_and_high_rho_failure() -> None:
    """Validate the coupled derivative and keep a rejected boundary truthful."""
    family, start, source, prepared = _coupled_penalty_problem()
    control = StreamPIRLSControl(
        solver_policy="qr", batch_rows=17, max_iter=100, tol=1e-9
    )
    rho = np.zeros(2)
    trial = evaluate_regular_stream_reml(
        StreamDesign(prepared, source),
        family,
        rho,
        maximum_bytes=64_000_000,
        control=control,
        initial_coefficients=start,
    )
    h = 1e-3
    finite_difference = []
    for index in range(2):
        scores = []
        for multiplier in (-2.0, -1.0, 1.0, 2.0):
            params = rho.copy()
            params[index] += multiplier * h
            evaluated = evaluate_regular_stream_reml(
                StreamDesign(prepared, source),
                family,
                params,
                maximum_bytes=64_000_000,
                control=control,
                initial_coefficients=start,
            )
            scores.append(float(evaluated.score))
        finite_difference.append(
            (scores[0] - 8.0 * scores[1] + 8.0 * scores[2] - scores[3]) / (12.0 * h)
        )
    np.testing.assert_allclose(
        trial.gradient,
        finite_difference,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )

    with pytest.raises(
        RuntimeError,
        match="converged, valid final source state",
    ):
        evaluate_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            np.full(2, 40.0),
            maximum_bytes=64_000_000,
            control=control,
            initial_coefficients=start,
        )


def test_regular_stream_reml_rejects_high_rho_rotated_nullspace_roundoff() -> None:
    """Never return a plausible derivative after dense nullspace contamination."""
    family, start, source, prepared, penalized_direction = (
        _rotated_null_penalty_problem()
    )
    structure = prepared.fitting.penalty_structure
    assert prepared.fitting.total_penalty_rank == 1
    assert prepared.fitting.total_penalty_null_dim == 2
    null_basis = np.column_stack(
        (
            np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
            np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0),
        )
    )
    np.testing.assert_allclose(
        penalized_direction @ null_basis,
        np.zeros(2),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )

    controls = {
        batch_rows: StreamPIRLSControl(
            solver_policy="qr", batch_rows=batch_rows, max_iter=100, tol=1e-9
        )
        for batch_rows in (17, 200)
    }
    regular_trials = []
    for control in controls.values():
        trial = evaluate_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            np.zeros(1),
            maximum_bytes=64_000_000,
            control=control,
            initial_coefficients=start,
        )
        assert trial.source_factor_residual < STRICT.rtol
        assert trial.observed_factor_residual < STRICT.rtol
        regular_trials.append(trial)
    for left, right in (
        (regular_trials[0].score, regular_trials[1].score),
        (regular_trials[0].gradient, regular_trials[1].gradient),
        (
            regular_trials[0].fit_result.state.coefficients,
            regular_trials[1].fit_result.state.coefficients,
        ),
    ):
        np.testing.assert_allclose(left, right, rtol=STRICT.rtol, atol=STRICT.atol)

    # Center the score differences before applying stencil weights.  The
    # absolute REML scores are about 85, so weighting them directly needlessly
    # amplifies platform-dependent last-bit cancellation.  A sixth-order
    # stencil at this step keeps truncation and score-quantization errors below
    # STRICT on the source-aligned analytic derivative.
    step = 1e-2
    center_score = float(regular_trials[0].score)
    scores = []
    for multiplier in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0):
        evaluated = evaluate_regular_stream_reml(
            StreamDesign(prepared, source),
            family,
            np.array([multiplier * step]),
            maximum_bytes=64_000_000,
            control=controls[17],
            warm_start=regular_trials[0],
        )
        scores.append(float(evaluated.score) - center_score)
    finite_difference = (
        -scores[0]
        + 9.0 * scores[1]
        - 45.0 * scores[2]
        + 45.0 * scores[3]
        - 9.0 * scores[4]
        + scores[5]
    ) / (60.0 * step)
    np.testing.assert_allclose(
        regular_trials[0].gradient,
        np.array([finite_difference]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )

    high_rho = np.array([40.0])
    materialized = structure.materialize(high_rho)
    # Mathematically S_lambda annihilates this basis. At exp(40), forming the
    # dense matrix injects order-one products into it; this is the PR9
    # high-rho penalty-nullspace roundoff mechanism, not an axial zero row.
    assert np.max(np.abs(materialized @ null_basis)) >= 1.0
    assert np.all(np.isfinite(materialized))
    for control in controls.values():
        with pytest.raises(
            RuntimeError,
            match="converged, valid final source state",
        ):
            evaluate_regular_stream_reml(
                StreamDesign(prepared, source),
                family,
                high_rho,
                maximum_bytes=64_000_000,
                control=control,
                initial_coefficients=start,
            )


def _floor_case(monkeypatch, *, offset: float, message: str, status: int):
    """Stub exact trials around an optimum at 12 with an O(1000) score.

    ``minimize`` stops at ``12 + offset`` with the given exit, as L-BFGS-B
    does when no further decrease is representable.
    """

    def evaluate(_stream, _family, params, **_kwargs):
        rho = float(np.asarray(params)[0])
        return SimpleNamespace(
            params=np.asarray(params).copy(),
            score=jnp.array(1000.0 + (rho - 12.0) ** 2),
            gradient=jnp.array([2.0 * (rho - 12.0)]),
            source_scans=2,
            batches_scanned=4,
            source_fingerprint="source",
            workspace=SimpleNamespace(required_bytes=128),
        )

    def stop(objective, _center, *, callback, **_kwargs):
        point = np.array([12.0 + offset])
        objective(point)
        callback(SimpleNamespace(x=point))
        return SimpleNamespace(
            x=point,
            success=status == 0,
            message=message,
            status=status,
            nit=7,
        )

    monkeypatch.setattr(reml_execution, "evaluate_regular_stream_reml", evaluate)
    monkeypatch.setattr(reml_execution, "minimize", stop)
    # Isolate the classification from the curvature completion.
    monkeypatch.setattr(reml_execution, "_CURVATURE_COMPLETION_MAX_FREE", 0)
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "source"),
        prepared=SimpleNamespace(
            fitting=SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1)),
            n_coef=2,
        ),
    )
    return optimize_regular_stream_reml(
        stream,
        SimpleNamespace(scale_known=True),
        # Start one step away, so the last accepted score change is as small
        # as at a real function-reduction exit.
        np.array([12.0 + 2.0 * offset]),
        maximum_bytes=1_000_000,
        control=StreamREMLControl(ftol=0.0),
    )


_FUNCTION_REDUCTION_EXIT = "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"


def test_function_reduction_exit_within_mgcv_relative_test_converges(
    monkeypatch,
) -> None:
    """Gradient 2e-4 > gtol 1e-6 but <= 5e-6 * score.scale (~1000)."""
    result = _floor_case(
        monkeypatch, offset=1e-4, message=_FUNCTION_REDUCTION_EXIT, status=0
    )
    assert result.converged
    assert result.projected_gradient_inf == 2.0 * ((12.0 + 1e-4) - 12.0)
    assert result.projected_gradient_inf > StreamREMLControl().gtol
    assert "converged by mgcv newton()'s relative test" in result.message
    assert "<= 5*conv.tol*score.scale 5.000e-03" in result.message
    assert result.message.startswith(_FUNCTION_REDUCTION_EXIT)


def test_function_reduction_exit_beyond_mgcv_relative_test_stays_unconverged(
    monkeypatch,
) -> None:
    """Gradient 0.2 exceeds 5e-6 * score.scale: truthfully not converged."""
    result = _floor_case(
        monkeypatch, offset=0.1, message=_FUNCTION_REDUCTION_EXIT, status=0
    )
    assert not result.converged
    assert "relative test" not in result.message
    assert "exceeds gtol" in result.message


@pytest.mark.parametrize(
    ("message", "status"),
    [
        ("ABNORMAL: ", 2),
        ("STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT", 1),
        ("STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT", 1),
    ],
    ids=["abnormal", "iteration-limit", "evaluation-limit"],
)
def test_other_lbfgsb_exits_keep_their_meaning(monkeypatch, message, status) -> None:
    """Only the function-reduction exit is classified by mgcv's test."""
    result = _floor_case(monkeypatch, offset=1e-4, message=message, status=status)
    assert not result.converged
    assert "relative test" not in result.message


def test_mgcv_relative_test_forms_score_scale_as_newton_does() -> None:
    """score.scale = |log(scale.est)| + |score|; 5*conv.tol on the gradient."""
    from jaxgam.execution.stream.reml import (
        MGCV_NEWTON_CONV_TOL,
        mgcv_relative_convergence,
    )

    assert MGCV_NEWTON_CONV_TOL == 1e-6
    estimated = mgcv_relative_convergence(
        -100.0, 1e-4, scale_estimate=np.exp(-2.0), score_change=1e-5
    )
    assert estimated.score_scale == 102.0
    assert estimated.gradient_limit == 5.0 * 1e-6 * 102.0
    assert estimated.score_change_limit == 1e-6 * 102.0
    assert estimated.met
    known = mgcv_relative_convergence(
        -100.0, 1e-4, scale_estimate=None, score_change=0.0
    )
    assert known.score_scale == 100.0
    assert not mgcv_relative_convergence(
        -100.0, 1e-4, scale_estimate=None, score_change=2e-4
    ).met
    assert not mgcv_relative_convergence(
        -100.0, 6e-4, scale_estimate=None, score_change=0.0
    ).met
    with pytest.raises(ValueError, match="finite and positive"):
        mgcv_relative_convergence(1.0, 0.0, scale_estimate=0.0, score_change=0.0)


def test_public_exact_stream_poisson_ten_smooths_converges_across_windows() -> None:
    """Scaling case M3 at n = 2,000: ten Poisson smooths, two of them
    unresolvable at k = 10 so their log sp climb +5 per source step window.

    With one shared 100-iteration budget this stopped at the iteration limit
    after four windows; with a budget per window it converges.
    """
    import pandas as pd

    from jaxgam import GAM
    from jaxgam.control import FitControl
    from jaxgam.data.source import ArrayRowSource

    rng = np.random.default_rng(42)
    names = [f"x{index}" for index in range(10)]
    columns = {name: rng.uniform(0.0, 1.0, 2000) for name in names}
    signal = sum(
        0.4 * np.sin(2.0 * np.pi * (index + 1) * columns[name])
        for index, name in enumerate(names)
    )
    y = rng.poisson(np.exp(0.2 + signal))
    data = pd.DataFrame({**columns, "y": y})
    formula = "y ~ " + " + ".join(f"s({name}, k=10, bs='cr')" for name in names)
    inner = ArrayRowSource({name: data[name].to_numpy() for name in names}, y=y)
    scans_started = []

    class _CountingSource:
        n_rows = inner.n_rows

        def fingerprint(self):
            return inner.fingerprint()

        def scan(self, batch_rows):
            scans_started.append(batch_rows)
            return inner.scan(batch_rows)

    source = _CountingSource()
    control = FitControl(
        execution="stream", linear_solver="qr", batch_rows=65536, uncertainty="fisher"
    )
    fit = GAM(formula, family="poisson", control=control).fit(
        source, result="prediction"
    )
    diagnostics = fit.exact_optimizer_diagnostics
    assert fit.converged, fit.convergence_info
    # Converged under the route's rule, by whichever branch meets it: the
    # projected gradient reaches gtol, or (native Linux x86_64 CI: a
    # function-reduction exit at 2.1e-6) L-BFGS-B exits on relative
    # reduction and pinned newton()'s relative test classifies the exit.
    gtol = StreamREMLControl().gtol
    if fit.convergence_info.startswith(
        "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
    ):
        assert diagnostics.projected_gradient_inf <= gtol
    else:
        assert fit.convergence_info.startswith(
            "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
        )
        assert "converged by mgcv newton()'s relative test" in (fit.convergence_info)
    assert "source step window(s)" in fit.convergence_info
    windows = int(fit.convergence_info.split("; ")[1].split(" ")[0])
    assert windows > 1
    assert fit.n_iter > StreamREMLControl().max_iter
    # Every exact trial scans the source at least once (its adjoint pass), and
    # the reported fit scans are a subset of the scans the source served
    # (the remainder are setup scans).
    assert (
        fit.source_scans >= diagnostics.startup_source_scans + diagnostics.n_evaluations
    )
    assert len(scans_started) >= fit.source_scans
