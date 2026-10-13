"""RowSource integration gates for frozen Gaussian QR smoothing."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import pandas as pd
import pytest

import jaxgam.api as api_module
import jaxgam.execution.gaussian_optimizer as gaussian_optimizer_module
import jaxgam.fitting.newton as newton_module
from jaxgam import GAM, FitControl
from jaxgam.api import _preflight_stream_workspace
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.gaussian import (
    estimate_streamed_gaussian_qr_workspace,
    preflight_streamed_gaussian_qr_workspace,
)
from jaxgam.execution.gaussian_optimizer import build_stream_gaussian_qr_provider
from jaxgam.families.standard import Gaussian
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.fitting.newton import newton_optimize
from jaxgam.fitting.stream_kernels import coefficient_stationarity
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from tests.tolerances import STRICT


def _inputs(n: int = 149):
    rng = np.random.default_rng(9814)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    weight = rng.uniform(0.3, 1.9, n)
    offset = 0.09 * np.sin(2.0 * z)
    y = (
        0.4
        + np.sin(2.7 * x)
        - 0.6 * np.cos(3.1 * z)
        + offset
        + rng.normal(scale=0.2, size=n)
    )
    data = pd.DataFrame({"x": x, "z": z, "y": y})
    formula = "y ~ s(x, bs='cr', k=7) + s(z, bs='cr', k=6)"
    return formula, data, weight, offset


def _large_m_inputs():
    n = 153
    base = np.linspace(-1.0, 1.0, n)
    columns = {f"x{index:02d}": np.roll(base, 3 * index) for index in range(51)}
    y = sum(np.sin(2.0 * values) for values in columns.values()) / np.sqrt(51.0)
    y += np.tile(np.array([0.1, -0.05, 0.02]), 51)
    data = pd.DataFrame({**columns, "y": y})
    formula = "y ~ " + " + ".join(
        f"s(x{index:02d}, bs='cr', k=3)" for index in range(51)
    )
    return formula, data


class _CountingSource:
    """Transparent source proxy exposing replay counts for hard gates."""

    def __init__(self, source) -> None:
        self.source = source
        self.scan_calls = 0
        self.batch_calls = 0
        self.max_batch_rows = 0

    @property
    def n_rows(self) -> int:
        return self.source.n_rows

    def fingerprint(self) -> str:
        return self.source.fingerprint()

    def scan(self, batch_rows: int):
        self.scan_calls += 1
        for batch in self.source.scan(batch_rows):
            self.batch_calls += 1
            self.max_batch_rows = max(self.max_batch_rows, len(batch.row_positions))
            yield batch


def _prepared_provider(batch_rows: int = 17):
    formula, data, weight, offset = _inputs()
    base = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula(formula), base, family=Gaussian())
    source = _CountingSource(base)
    metadata = PreparedFittingMetadata.from_prepared(prepared, Gaussian())
    provider = build_stream_gaussian_qr_provider(
        StreamDesign(prepared, source), metadata, batch_rows=batch_rows
    )
    return formula, data, weight, offset, source, metadata, provider


def _jaxpr_shapes(closed_jaxpr) -> list[tuple[int, ...]]:
    shapes = [np.shape(value) for value in jax.tree.leaves(closed_jaxpr.consts)]

    def visit(jaxpr) -> None:
        for variable in (
            *jaxpr.constvars,
            *jaxpr.invars,
            *jaxpr.outvars,
            *(variable for equation in jaxpr.eqns for variable in equation.invars),
            *(variable for equation in jaxpr.eqns for variable in equation.outvars),
        ):
            aval = getattr(variable, "aval", None)
            if aval is not None and hasattr(aval, "shape"):
                shapes.append(tuple(aval.shape))
        for equation in jaxpr.eqns:
            for value in jax.tree.leaves(equation.params):
                if hasattr(value, "jaxpr") and hasattr(value, "consts"):
                    shapes.extend(np.shape(constant) for constant in value.consts)
                    visit(value.jaxpr)
                elif hasattr(value, "eqns") and hasattr(value, "invars"):
                    visit(value)

    visit(closed_jaxpr.jaxpr)
    return shapes


def test_one_preparation_scan_and_no_smoothing_or_finalization_rescan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    *_, source, metadata, provider = _prepared_provider(batch_rows=17)
    expected_batches = math.ceil(metadata.n_obs / 17)
    assert source.scan_calls == provider.source_scans == 1
    assert source.batch_calls == provider.batches_scanned == expected_batches
    assert source.max_batch_rows <= 17
    assert not hasattr(provider, "_stream")

    def fail_row_initializer(*_args, **_kwargs):
        raise AssertionError("compressed optimization called the dense row initializer")

    class _AuditedJnp:
        def __init__(self) -> None:
            self.zeros_shapes = []

        def zeros(self, shape, *args, **kwargs):
            self.zeros_shapes.append(shape)
            if shape == metadata.n_obs or shape == (metadata.n_obs,):
                raise AssertionError("compressed optimization allocated an n-vector")
            return jax.numpy.zeros(shape, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(jax.numpy, name)

    audited_jnp = _AuditedJnp()
    monkeypatch.setattr(newton_module, "initialize_beta", fail_row_initializer)
    monkeypatch.setattr(newton_module, "jnp", audited_jnp)

    result = newton_optimize(metadata, score_provider=provider)
    after_optimization = (source.scan_calls, source.batch_calls)
    state = provider.to_stream_state(result)
    assert (source.scan_calls, source.batch_calls) == after_optimization
    assert state.source_scans == 1
    assert state.batches_scanned == expected_batches
    assert provider.finalize_calls == 1
    assert audited_jnp.zeros_shapes == [metadata.n_coef]
    expected_stationarity = coefficient_stationarity(
        state.coefficient_factor.lower @ state.coefficient_factor.lower.T,
        provider._data_rhs,
        state.coefficients,
    )
    np.testing.assert_allclose(
        state.stationarity,
        expected_stationarity,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert state.stationarity < 1e-12
    # StreamFitState counts inner coefficient halvings. The Gaussian QR
    # coefficient solve is direct; outer Newton status is carried separately.
    assert state.backtracks == 0
    assert not state.line_search_failed


def test_stream_provider_jaxprs_and_retained_state_have_no_row_dimension() -> None:
    *_, source, metadata, provider = _prepared_provider(batch_rows=19)
    params = provider.initial_parameters(metadata.log_lambda_init)
    trial, _ = provider.evaluate(params)
    score_jaxpr = jax.make_jaxpr(lambda value: provider.evaluate(value)[1])(params)
    derivative_jaxpr = jax.make_jaxpr(
        lambda value: provider.differentiate(value, trial)
    )(params)
    shapes = _jaxpr_shapes(score_jaxpr) + _jaxpr_shapes(derivative_jaxpr)
    assert shapes
    assert all(metadata.n_obs not in shape for shape in shapes)

    result = newton_optimize(metadata, score_provider=provider)
    state = provider.to_stream_state(result)
    retained = [
        provider._compression.R,
        provider._compression.qtz,
        *provider._penalty_roots,
        state.coefficients,
        state.xtwx,
        state.coefficient_factor.lower,
    ]
    assert all(metadata.n_obs not in np.shape(value) for value in retained)
    assert source.scan_calls == 1


@pytest.mark.parametrize("fixed_sp", [None, [0.7, 1.3]], ids=["estimated", "fixed"])
def test_public_stream_gaussian_qr_matches_dense_strict(fixed_sp) -> None:
    formula, data, weight, offset = _inputs()
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    streamed = GAM(
        formula,
        family="gaussian",
        sp=fixed_sp,
        control=FitControl(
            execution="stream", gaussian_compression=True, batch_rows=23
        ),
    ).fit(source, result="prediction")
    dense = GAM(
        formula,
        family="gaussian",
        sp=fixed_sp,
        control=FitControl(gaussian_compression=True),
    ).fit(data, weights=weight, offset=offset, result="prediction")

    assert streamed.execution_route == "stream_gaussian_qr"
    assert streamed.execution_fallback_reason is None
    for name in ("coefficients", "deviance", "score", "scale", "smoothing_params"):
        np.testing.assert_allclose(
            getattr(streamed, name),
            getattr(dense, name),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    np.testing.assert_allclose(
        streamed.predict(data[["x", "z"]], offset=offset),
        dense.predict(data[["x", "z"]], offset=offset),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_real_large_m_stream_uses_first_order_route_and_bounded_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formula, data = _large_m_inputs()
    source = DataFrameRowSource(data, response="y")
    captured: dict[str, object] = {}
    original_builder = gaussian_optimizer_module.build_stream_gaussian_qr_provider

    def audited_builder(stream, fitting, *, batch_rows):
        estimate = estimate_streamed_gaussian_qr_workspace(stream.prepared, batch_rows)
        provider = original_builder(stream, fitting, batch_rows=batch_rows)
        captured.update(estimate=estimate, prepared=stream.prepared, provider=provider)
        return provider

    def fail_hessian(*_args, **_kwargs):
        raise AssertionError("large-m streamed Gaussian fit formed an outer Hessian")

    monkeypatch.setattr(
        gaussian_optimizer_module,
        "build_stream_gaussian_qr_provider",
        audited_builder,
    )
    monkeypatch.setattr(
        gaussian_optimizer_module, "jit_gaussian_qr_reml_hessian", fail_hessian
    )
    streamed = GAM(
        formula,
        control=FitControl(
            execution="stream",
            gaussian_compression=True,
            batch_rows=37,
        ),
    ).fit(source, result="prediction")

    estimate = captured["estimate"]
    provider = captured["provider"]
    assert provider._fitting.n_penalties == 51
    assert estimate.outer_hessian_bytes == 0
    n_free = 52
    maxcor = 10
    expected_lbfgs = (
        2 * maxcor * n_free + 5 * n_free + 11 * maxcor * maxcor + 8 * maxcor
    ) * np.dtype(np.float64).itemsize + 3 * n_free * np.dtype(np.int32).itemsize
    assert estimate.lbfgs_workspace_bytes == expected_lbfgs
    fixed_estimate = estimate_streamed_gaussian_qr_workspace(
        captured["prepared"], 37, pin_lambda=True
    )
    assert fixed_estimate.outer_hessian_bytes == 0
    assert 0 < fixed_estimate.lbfgs_workspace_bytes < estimate.lbfgs_workspace_bytes
    assert provider.source_scans == 1
    assert provider.finalize_calls == 1
    assert streamed.execution_route == "stream_gaussian_qr"
    assert streamed.lambda_strategy == "lbfgs_reml"
    assert streamed.converged

    dense = GAM(
        formula,
        control=FitControl(gaussian_compression=True),
    ).fit(data, result="prediction")
    for actual, expected in (
        (streamed.coefficients, dense.coefficients),
        (streamed.deviance, dense.deviance),
        (streamed.score, dense.score),
        (streamed.scale, dense.scale),
        (streamed.predict(data), dense.predict(data)),
    ):
        np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    # This fixture's true signal lies in the shared unpenalized subspace, so
    # all 51 smoothing optima are effectively infinite. The raw finite lambda
    # representatives can drift while fitted quantities remain STRICT.
    assert np.min(streamed.smoothing_params) > 1e8
    assert np.min(dense.smoothing_params) > 1e8


@pytest.mark.parametrize("bad_weight", [0.0, 1e-12, 1e11])
def test_fixed_stream_ineligible_weights_fall_back_with_original_likelihood(
    bad_weight: float,
) -> None:
    formula, data, weight, offset = _inputs(83)
    weight[7] = bad_weight
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    fallback = GAM(
        formula,
        sp=[0.8, 1.2],
        control=FitControl(
            execution="stream", gaussian_compression=True, batch_rows=13
        ),
    ).fit(source, result="prediction")
    ordinary = GAM(
        formula,
        sp=[0.8, 1.2],
        control=FitControl(execution="stream", batch_rows=13),
    ).fit(source, result="prediction")
    assert fallback.execution_route == "stream"
    assert "unclipped finite positive weights" in fallback.execution_fallback_reason
    for name in ("coefficients", "deviance", "score", "scale"):
        np.testing.assert_allclose(
            getattr(fallback, name),
            getattr(ordinary, name),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


def test_estimated_stream_ineligible_weights_fail_with_specific_reason() -> None:
    formula, data, weight, offset = _inputs(83)
    weight[0] = 0.0
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    with pytest.raises(NotImplementedError, match="unclipped finite positive weights"):
        GAM(
            formula,
            control=FitControl(
                execution="stream", gaussian_compression=True, batch_rows=13
            ),
        ).fit(source, result="prediction")


def test_stream_preparation_rejects_oversized_batches() -> None:
    formula, data, weight, offset = _inputs(83)
    base = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula(formula), base, family=Gaussian())
    metadata = PreparedFittingMetadata.from_prepared(prepared, Gaussian())

    class OversizedSource:
        n_rows = base.n_rows

        def fingerprint(self) -> str:
            return base.fingerprint()

        def scan(self, _batch_rows: int):
            yield from base.scan(base.n_rows)

    with pytest.raises(ValueError, match="larger than the requested batch_rows"):
        build_stream_gaussian_qr_provider(
            StreamDesign(prepared, OversizedSource()), metadata, batch_rows=11
        )


# FitControl.memory_budget_bytes also budgets Phase-1 setup, whose exact
# cubic-knot scan holds a fixed SQLite scope (about 17.3 MB for these models)
# independent of the row count. A public budget below that scope is rejected
# by setup before any fit-stage preflight. The single-batch source below keeps
# its Gaussian QR peak above a setup-admitting budget while its ordinary
# streamed PIRLS workspace stays within it.
_SETUP_ADMITTING_BUDGET = 20_000_000


def _setup_admitting_large_batch_case(*, zero_weight: bool):
    n = batch_rows = 32_768
    x = np.linspace(-1.0, 1.0, n)
    data = pd.DataFrame({"x": x, "y": np.sin(3.0 * x) + 0.1 * np.cos(11.0 * x)})
    weight = None
    if zero_weight:
        weight = np.ones(n)
        weight[0] = 0.0
    source = DataFrameRowSource(data, response="y", weights=weight)
    formula = "y ~ s(x, bs='cr', k=20)"
    prepared = prepare_model(
        parse_formula(formula),
        source,
        family=Gaussian(),
        batch_rows=batch_rows,
        maximum_bytes=_SETUP_ADMITTING_BUDGET,
    )
    ordinary = _preflight_stream_workspace(
        prepared.n_coef, batch_rows, _SETUP_ADMITTING_BUDGET
    )
    assert ordinary <= _SETUP_ADMITTING_BUDGET
    assert (
        estimate_streamed_gaussian_qr_workspace(prepared, batch_rows).required_bytes
        > _SETUP_ADMITTING_BUDGET
    )
    return formula, source, batch_rows


def test_gaussian_workspace_rejects_four_buffer_qr_peak_before_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 4096
    batch_rows = 2048
    x = np.linspace(-1.0, 1.0, n)
    data = pd.DataFrame({"x": x, "y": np.sin(3.0 * x) + 0.1 * np.cos(11.0 * x)})
    source = DataFrameRowSource(data, response="y")
    formula = "y ~ s(x, bs='cr', k=20)"
    prepared = prepare_model(parse_formula(formula), source, family=Gaussian())
    p = prepared.n_coef
    generic_budget = (
        12 * p * p + 3 * batch_rows * p + 8 * batch_rows + 4 * p
    ) * np.dtype(np.float64).itemsize
    _preflight_stream_workspace(p, batch_rows, generic_budget)

    estimate = estimate_streamed_gaussian_qr_workspace(prepared, batch_rows)
    expected_four_design_arrays = (4 * batch_rows * p + 2 * p * p) * np.dtype(
        np.float64
    ).itemsize
    assert expected_four_design_arrays == 1_317_120
    assert estimate.qr_design_bytes == expected_four_design_arrays
    assert estimate.outer_hessian_bytes == 2 * 2 * np.dtype(np.float64).itemsize
    assert estimate.required_bytes > generic_budget
    with pytest.raises(MemoryError, match="Known streamed Gaussian QR workspace"):
        preflight_streamed_gaussian_qr_workspace(prepared, batch_rows, generic_budget)
    accepted = preflight_streamed_gaussian_qr_workspace(
        prepared, batch_rows, estimate.required_bytes
    )
    assert accepted == estimate

    def fail_device(*_args, **_kwargs):
        raise AssertionError("device resolution preceded Gaussian QR preflight")

    monkeypatch.setattr(api_module, "_resolve_device", fail_device)
    with pytest.raises(MemoryError, match="Phase-1 streamed setup requires"):
        GAM(
            formula,
            control=FitControl(
                execution="stream",
                gaussian_compression=True,
                batch_rows=batch_rows,
                memory_budget_bytes=generic_budget,
            ),
        ).fit(source, result="prediction")
    large_formula, large_source, large_batch_rows = _setup_admitting_large_batch_case(
        zero_weight=False
    )
    with pytest.raises(MemoryError, match="additional native workspace"):
        GAM(
            large_formula,
            control=FitControl(
                execution="stream",
                gaussian_compression=True,
                batch_rows=large_batch_rows,
                memory_budget_bytes=_SETUP_ADMITTING_BUDGET,
            ),
        ).fit(large_source, result="prediction")


def test_fixed_ineligible_weights_use_ordinary_budget_before_qr_preflight() -> None:
    n = 4096
    batch_rows = 2048
    x = np.linspace(-1.0, 1.0, n)
    data = pd.DataFrame({"x": x, "y": np.sin(3.0 * x) + 0.1 * np.cos(11.0 * x)})
    weight = np.ones(n)
    weight[0] = 0.0
    source = DataFrameRowSource(data, response="y", weights=weight)
    formula = "y ~ s(x, bs='cr', k=20)"
    prepared = prepare_model(parse_formula(formula), source, family=Gaussian())
    p = prepared.n_coef
    generic_budget = (
        12 * p * p + 3 * batch_rows * p + 8 * batch_rows + 4 * p
    ) * np.dtype(np.float64).itemsize
    assert generic_budget == 1_153_152
    _preflight_stream_workspace(p, batch_rows, generic_budget)
    assert (
        estimate_streamed_gaussian_qr_workspace(prepared, batch_rows).required_bytes
        > generic_budget
    )

    with pytest.raises(MemoryError, match="Phase-1 streamed setup requires"):
        GAM(
            formula,
            sp=[1.0],
            control=FitControl(
                execution="stream",
                gaussian_compression=True,
                batch_rows=batch_rows,
                memory_budget_bytes=generic_budget,
            ),
        ).fit(source, result="prediction")

    large_formula, large_source, large_batch_rows = _setup_admitting_large_batch_case(
        zero_weight=True
    )
    ordinary = GAM(
        large_formula,
        sp=[1.0],
        control=FitControl(
            execution="stream",
            batch_rows=large_batch_rows,
            memory_budget_bytes=_SETUP_ADMITTING_BUDGET,
        ),
    ).fit(large_source, result="prediction")
    fallback = GAM(
        large_formula,
        sp=[1.0],
        control=FitControl(
            execution="stream",
            gaussian_compression=True,
            batch_rows=large_batch_rows,
            memory_budget_bytes=_SETUP_ADMITTING_BUDGET,
        ),
    ).fit(large_source, result="prediction")

    assert fallback.execution_route == "stream"
    assert "unclipped finite positive weights" in fallback.execution_fallback_reason
    for name in ("coefficients", "deviance", "score", "scale"):
        np.testing.assert_allclose(
            getattr(fallback, name),
            getattr(ordinary, name),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


@pytest.mark.parametrize(
    "field",
    ["row_positions", "valid", "y", "weight", "offset"],
)
def test_stream_preparation_rejects_mismatched_batch_vectors(field: str) -> None:
    formula, data, weight, offset = _inputs(83)
    base = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula(formula), base, family=Gaussian())
    metadata = PreparedFittingMetadata.from_prepared(prepared, Gaussian())

    class MismatchedSource:
        n_rows = base.n_rows

        def fingerprint(self) -> str:
            return base.fingerprint()

        def scan(self, batch_rows: int):
            for index, batch in enumerate(base.scan(batch_rows)):
                if index == 0:
                    values = np.asarray(getattr(batch, field))[:-1]
                    contents = {
                        name: getattr(batch, name)
                        for name in (
                            "columns",
                            "y",
                            "weight",
                            "offset",
                            "valid",
                            "row_positions",
                        )
                    }
                    contents[field] = values
                    batch = SimpleNamespace(**contents)
                yield batch

    with pytest.raises(ValueError, match="matching row vectors"):
        build_stream_gaussian_qr_provider(
            StreamDesign(prepared, MismatchedSource()),
            metadata,
            batch_rows=11,
        )


@pytest.mark.parametrize("replacement", [0, 10_000], ids=["duplicate", "missing"])
def test_stream_preparation_rejects_changed_row_position_replay(
    replacement: int,
) -> None:
    formula, data, weight, offset = _inputs(83)
    base = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared = prepare_model(parse_formula(formula), base, family=Gaussian())
    metadata = PreparedFittingMetadata.from_prepared(prepared, Gaussian())

    class ChangedPositionsSource:
        n_rows = base.n_rows

        def fingerprint(self) -> str:
            return base.fingerprint()

        def scan(self, batch_rows: int):
            for index, batch in enumerate(base.scan(batch_rows)):
                if index == 1:
                    positions = np.array(batch.row_positions, copy=True)
                    positions[0] = replacement
                    batch = replace(batch, row_positions=positions)
                yield batch

    with pytest.raises(RuntimeError, match="changed row positions"):
        build_stream_gaussian_qr_provider(
            StreamDesign(prepared, ChangedPositionsSource()),
            metadata,
            batch_rows=11,
        )


def test_reordered_repeated_source_selection_remains_supported() -> None:
    formula, data, weight, offset = _inputs(83)
    positions = np.r_[np.arange(8, 48), [17, 9, 44, 17]]
    source = DataFrameRowSource(
        data,
        response="y",
        weights=weight,
        offset=offset,
        row_selection=positions,
    )
    streamed = GAM(
        formula,
        sp=[0.8, 1.2],
        control=FitControl(execution="stream", gaussian_compression=True, batch_rows=7),
    ).fit(source, result="prediction")
    selected = data.iloc[positions].reset_index(drop=True)
    dense = GAM(
        formula,
        sp=[0.8, 1.2],
        control=FitControl(gaussian_compression=True),
    ).fit(
        selected,
        weights=weight[positions],
        offset=offset[positions],
        result="prediction",
    )
    np.testing.assert_allclose(
        streamed.coefficients,
        dense.coefficients,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
