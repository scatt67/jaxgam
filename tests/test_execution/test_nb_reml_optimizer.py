"""Joint exact streamed REML optimizer gates for Negative Binomial."""

from __future__ import annotations

from dataclasses import replace
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
from tests.helpers import _AssertCollector, nb_optimizer_case_data, r_available
from tests.r_bridge import RBridge
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
# These controls and the named seeded model are the reviewed selection profile.
# A candidate test must match them before a field-specific MODERATE comparison.
_REVIEWED_OPTIMIZER_CONTROL = _OPTIMIZER_CONTROL
_REVIEWED_PIRLS_CONTROL = _OPTIMIZER_PIRLS_CONTROL


def _optimizer_case(link: str, *, estimated: bool, smooth: bool = True):
    data, weight, offset = nb_optimizer_case_data()
    family = NegativeBinomial(theta=_THETA, fixed=not estimated, link=link)
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
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


def _assert_optimizer_case(stream, family, params, *, link, estimated) -> None:
    """Bind MODERATE only to the original generated model and controls."""
    source = getattr(stream.source, "source", stream.source)
    data, weight, offset = nb_optimizer_case_data()
    expected_source = DataFrameRowSource(
        data, response="y", weights=weight, offset=offset
    )
    assert stream.prepared.n_obs == source.n_rows == expected_source.n_rows == 180
    assert source.fingerprint() == expected_source.fingerprint()
    actual = next(source.scan(stream.prepared.n_obs))
    for observed, expected in (
        (actual.columns["x"], data.x.to_numpy()),
        (actual.y, data.y.to_numpy()),
        (actual.weight, weight),
        (actual.offset, offset),
    ):
        np.testing.assert_array_equal(observed, expected)
    assert type(family) is NegativeBinomial
    assert family.family_name == "nb"
    expected_family = NegativeBinomial(theta=_THETA, fixed=not estimated, link=link)
    assert type(family.link) is type(expected_family.link)
    assert family.n_theta == int(estimated)
    np.testing.assert_array_equal(family.get_theta(transformed=True), [_THETA])
    assert len(params) == 1 + family.n_theta
    np.testing.assert_array_equal(np.asarray(params)[0], np.log(0.5))
    if family.n_theta:
        np.testing.assert_array_equal(np.asarray(params)[-1], np.log(_THETA))
    assert _OPTIMIZER_CONTROL == _REVIEWED_OPTIMIZER_CONTROL
    assert _OPTIMIZER_PIRLS_CONTROL == _REVIEWED_PIRLS_CONTROL
    assert _OPTIMIZER_MAXIMUM_BYTES == 64_000_000
    smooth_blocks = tuple(
        block
        for block in stream.prepared.predict_spec.coef_map.terms
        if block.smooth is not None
    )
    assert len(smooth_blocks) == 1
    smooth = smooth_blocks[0].smooth.spec
    assert tuple(smooth.variables) == ("x",)
    assert smooth.bs == "cr"
    assert smooth.k == 8
    assert smooth.by is None
    assert not smooth.extra_args
    assert stream.prepared.response == "y"
    assert stream.prepared.n_coef == 8
    penalty = stream.prepared.fitting.penalty_structure
    assert penalty.n_penalties == 1
    assert len(penalty.blocks) == 1
    assert len(penalty.blocks[0].local_penalties) == 1
    assert penalty.blocks[0].ranks == (6,)
    # A fresh preparation of the same generated rows protects local-D basis,
    # penalty and transform coordinates without retaining numeric snapshots.
    reference = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=8)'), expected_source, family=family
    )
    assert stream.prepared.basis_fingerprint == reference.basis_fingerprint
    np.testing.assert_allclose(
        penalty.blocks[0].local_penalties[0].matrix,
        reference.fitting.penalty_structure.blocks[0].local_penalties[0].matrix,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    actual_transform = penalty.blocks[0].transform.matrix.T
    expected_transform = reference.fitting.penalty_structure.blocks[
        0
    ].transform.matrix.T
    np.testing.assert_allclose(
        actual_transform[:, :, None] * actual_transform[:, None, :],
        expected_transform[:, :, None] * expected_transform[:, None, :],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


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
    return RBridge(mode="rpy2").nb_selected_fit(
        X,
        E,
        batch.y,
        batch.weight,
        batch.offset,
        params,
        link,
        estimated=estimated,
        epsilon=epsilon,
        max_iter=max_iter,
    )


def _pinned_outer_reference(
    stream,
    link,
    *,
    estimated,
    pinned_rho=None,
):
    source = getattr(stream.source, "source", stream.source)
    batch = next(source.scan(stream.prepared.n_obs))
    return RBridge(mode="rpy2").nb_cubic_outer_fit(
        pd.DataFrame({"x": batch.columns["x"], "y": batch.y}),
        batch.weight,
        batch.offset,
        link,
        estimated=estimated,
        pinned_rho=pinned_rho,
    )


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
def test_nb_optimizer_selected_state_matches_pinned_gam_fit4(link, estimated):
    stream, _family, _params, result = _optimize(link, estimated=estimated)
    expected = _pinned_selected_reference(
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
    link, estimated
):
    """The reviewed six-cell outer fit retains STRICT score agreement."""
    stream, family, initial, result = _optimize(link, estimated=estimated)
    _assert_optimizer_case(stream, family, initial, link=link, estimated=estimated)
    expected = _pinned_outer_reference(
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
    _assert_optimizer_case(stream, family, initial, link="log", estimated=True)

    changed_start = initial.copy()
    changed_start[0] = np.nextafter(changed_start[0], np.inf)
    with pytest.raises(AssertionError):
        _assert_optimizer_case(
            stream, family, changed_start, link="log", estimated=True
        )

    with pytest.raises(AssertionError):
        _assert_optimizer_case(stream, family, initial, link="sqrt", estimated=True)
    with pytest.raises(AssertionError):
        _assert_optimizer_case(stream, family, initial, link="log", estimated=False)

    with monkeypatch.context() as control_patch:
        control_patch.setattr(
            __name__ + "._OPTIMIZER_CONTROL",
            replace(_OPTIMIZER_CONTROL, ftol=1e-13),
        )
        with pytest.raises(AssertionError):
            _assert_optimizer_case(stream, family, initial, link="log", estimated=True)

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
        _assert_optimizer_case(
            changed_stream, family, initial, link="log", estimated=True
        )

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
        _assert_optimizer_case(
            changed_basis_stream, family, initial, link="log", estimated=True
        )


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
def test_nb_optimizer_generated_basis_and_penalty_match_live_pinned_r():
    """Validate model coordinates through gam.setup instead of saved matrices."""
    stream, family, initial = _optimizer_case("log", estimated=True)
    _assert_optimizer_case(stream, family, initial, link="log", estimated=True)
    source = getattr(stream.source, "source", stream.source)
    batch = next(source.scan(stream.prepared.n_obs))
    data = pd.DataFrame({"x": batch.columns["x"], "y": batch.y})
    expected_X, expected_S, metadata = RBridge(mode="rpy2").nb_cubic_setup(
        data, batch.weight, batch.offset
    )
    np.testing.assert_array_equal(metadata, [2.0, 6.0])
    np.testing.assert_allclose(
        stream.prepared.evaluate_batch(batch),
        expected_X,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        stream.prepared.penalties.blocks[0].local_penalties[0].matrix,
        expected_S,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


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
def test_nb_optimizer_dynamic_pinned_rho_matches_tight_pinned_source_state(link):
    stream, _family, supplied, result = _optimize(
        link,
        estimated=True,
        pin_lambda=True,
    )
    expected = _pinned_selected_reference(
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


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
def test_nb_fixed_theta_and_pinned_smoothing_extracts_source_fields():
    """The no-outer-step R fit still reports its own deviance, EDF and scale."""
    stream, _family, _supplied, result = _optimize(
        "log", estimated=False, pin_lambda=True
    )
    expected = _pinned_outer_reference(
        stream,
        "log",
        estimated=False,
        pinned_rho=float(np.asarray(result.trial.params)[0]),
    )
    state = result.trial.fit_result.state
    assert expected["inner_converged"]
    assert expected["outer_converged"]
    assert expected["outer_iterations"] == 0
    assert expected["scale"] == 1.0
    np.testing.assert_allclose(
        [expected["deviance"], expected["edf"]],
        [float(state.deviance), float(state.edf)],
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
