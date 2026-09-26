"""Source-timed host adjoints for the regular streamed controller."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgam.execution.reml as reml_execution
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.reml import evaluate_regular_stream_reml
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


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize("family_class", [Gaussian, Gamma, Poisson, Binomial])
@pytest.mark.parametrize("link", _LINKS)
def test_regular_source_gradient_matches_all_32_pinned_gdi_cells(
    tmp_path, family_class, link
):
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
    fields, _null, _covariance, expected = _source_reference(
        tmp_path,
        family,
        link,
        X,
        data.y,
        weight,
        offset,
        start,
        derivatives=True,
    )
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
