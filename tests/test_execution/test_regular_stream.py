"""Regular source controller: accounting, information consumers and R fits."""

import inspect
import sys
from dataclasses import replace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import jaxgam.execution.regular_stream as regular_controller
from jaxgam.control import FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.null_coefficient import NullCoefficientProjection
from jaxgam.execution.regular_stream import (
    _positive_signed_state,
    fit_regular_streamed_pirls,
    preflight_regular_stream_workspace,
    regular_working_scan,
)
from jaxgam.execution.signed_qr import solve_signed_qr
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Gamma, Poisson
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
from tests.helpers import r_available
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


def test_explicit_source_null_rejects_complex_before_scanning() -> None:
    stream, family, *_ = _fixture()
    with pytest.raises(ValueError, match="finite fitting p-vector"):
        fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=10_000_000,
            score_scale=0.7,
            control=StreamPIRLSControl(batch_rows=11, solver_policy="qr"),
            null_coefficients=np.zeros(stream.prepared.n_coef, dtype=np.complex128),
        )
    assert stream.source.scans == 0


def test_explicit_zero_null_preserves_raw_source_baseline_semantics() -> None:
    x = np.linspace(-0.6, 0.7, 61)
    data = pd.DataFrame({"x": x, "y": np.full(len(x), 2.0)})
    control = StreamPIRLSControl(batch_rows=11, solver_policy="qr")

    poisson = Poisson("identity")
    poisson_source = DataFrameRowSource(data, response="y")
    poisson_prepared = prepare_model(
        parse_formula("y ~ x"), poisson_source, family=poisson
    )
    fitted = fit_regular_streamed_pirls(
        StreamDesign(poisson_prepared, poisson_source),
        poisson,
        np.empty(0),
        maximum_bytes=10_000_000,
        score_scale=1.0,
        control=control,
        null_coefficients=np.zeros(poisson_prepared.n_coef),
    )
    assert fitted.state.converged
    assert np.isposinf(fitted.accepted_penalized_history[0])

    gamma = Gamma("identity")
    gamma_source = DataFrameRowSource(data, response="y")
    gamma_prepared = prepare_model(parse_formula("y ~ x"), gamma_source, family=gamma)
    with pytest.raises(ValueError, match="null coefficient baseline"):
        fit_regular_streamed_pirls(
            StreamDesign(gamma_prepared, gamma_source),
            gamma,
            np.empty(0),
            maximum_bytes=10_000_000,
            score_scale=0.7,
            control=control,
            null_coefficients=np.zeros(gamma_prepared.n_coef),
        )


def test_default_null_preserves_original_domain_rejection() -> None:
    """Only an explicit source anchor may retain a raw +Inf baseline."""
    stream, family, *_ = _fixture()
    p = stream.prepared.n_coef
    invalid_projection = NullCoefficientProjection(
        np.zeros(p), p, np.arange(p, dtype=np.int64)
    )
    with (
        patch.object(
            regular_controller,
            "project_null_coefficients",
            return_value=invalid_projection,
        ),
        pytest.raises(ValueError, match="anchor leaves the family domain"),
    ):
        fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=10_000_000,
            score_scale=0.7,
            control=StreamPIRLSControl(batch_rows=11, solver_policy="qr"),
        )


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_explicit_zero_null_raw_baseline_matches_pinned_gam_fit3(r_bridge) -> None:
    versions_match, reason = r_bridge.check_versions()
    assert versions_match, reason
    x = np.linspace(-0.6, 0.7, 61)
    data = pd.DataFrame({"x": x, "y": np.full(len(x), 2.0)})
    family = Poisson("identity")
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    fitted = fit_regular_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.empty(0),
        maximum_bytes=10_000_000,
        score_scale=1.0,
        control=StreamPIRLSControl(batch_rows=11, solver_policy="qr"),
        null_coefficients=np.zeros(prepared.n_coef),
    )
    reference = r_bridge.regular_source_zero_null(x, data.y.to_numpy())
    np.testing.assert_allclose(
        fitted.state.coefficients,
        reference["beta"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        fitted.state.deviance,
        reference["deviance"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert reference["gamma_failed"]


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
    assert isinstance(result.source_coefficient_factor, SignedQRCoefficientFactor)
    assert result.source_coefficient_factor is not state.coefficient_factor
    assert not result.source_solve_coefficients.flags.writeable
    np.testing.assert_allclose(
        result.source_solve_coefficients,
        state.coefficients,
        rtol=0.0,
        atol=0.0,
    )
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


def test_explicit_source_null_anchor_is_distinct_and_prospectively_charged():
    stream, family, *_ = _fixture()
    control = StreamPIRLSControl(batch_rows=17, solver_policy="qr", tol=1e-11)
    ledger = preflight_regular_stream_workspace(stream, control, 10000000)
    anchor = np.zeros(stream.prepared.n_coef)
    anchor[0] = 0.2
    with pytest.raises(MemoryError, match="explicit regular null anchor"):
        fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=ledger.required_bytes + 8 * len(anchor) - 1,
            score_scale=0.7,
            control=control,
            null_coefficients=anchor,
        )
    assert stream.source.scans == stream.source.batches == 0
    with pytest.raises(ValueError, match="finite fitting p-vector"):
        fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=10000000,
            score_scale=0.7,
            control=control,
            null_coefficients=np.full(len(anchor), np.nan),
        )
    assert stream.source.scans == stream.source.batches == 0

    with patch.object(
        regular_controller,
        "project_null_coefficients",
        side_effect=AssertionError("explicit anchor must not be reprojected"),
    ):
        result = fit_regular_streamed_pirls(
            stream,
            family,
            np.empty(0),
            maximum_bytes=10000000,
            score_scale=0.7,
            control=control,
            null_coefficients=anchor,
        )
    np.testing.assert_array_equal(result.null_coefficients, anchor)
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    X = stream.prepared.evaluate_fitting_batch(batch)
    expected = float(
        family.dev_resids(
            batch.y, family.link.linkinv(X @ anchor + batch.offset), batch.weight
        )
    )
    np.testing.assert_allclose(
        result.accepted_penalized_history[0],
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_regular_source_adjoint_state_is_prospectively_budgeted():
    stream, family, *_ = _fixture()
    control = StreamPIRLSControl(batch_rows=17, solver_policy="qr")
    ledger = preflight_regular_stream_workspace(stream, control, 10000000)
    p = stream.prepared.n_coef
    assert ledger.source_adjoint_bytes == 8 * (4 * p * p + 10 * p)
    assert ledger.required_bytes > ledger.source_adjoint_bytes
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
def test_regular_gamma_fixed_trial_scale_matches_pinned_gam_fit3(r_bridge, link):
    stream, family, data, w, off = _fixture(link)
    X = np.column_stack((np.ones(len(data)), data.x))
    reference = r_bridge.regular_source_gamma_fixed_trial(
        link, X, data.y.to_numpy(), w, off
    )
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
    r_bridge, monkeypatch, controlled_invalid_refit
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
    reference_fit = r_bridge.regular_source_penalized_gamma_score(
        X,
        E,
        y,
        weight,
        offset,
        controlled_invalid_refit=controlled_invalid_refit,
    )
    reference = reference_fit["reference"]
    fisher_inverse = reference_fit["fisher_inverse"]
    source_payload = reference_fit["source_payload"]
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
        assert not result.source_solve_coefficients.flags.writeable
        if controlled_invalid_refit:
            np.testing.assert_array_equal(result.source_solve_coefficients, bad)
            assert not np.array_equal(
                result.source_solve_coefficients, np.asarray(state.coefficients)
            )
        else:
            np.testing.assert_allclose(
                result.source_solve_coefficients,
                state.coefficients,
                rtol=0.0,
                atol=0.0,
            )
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
