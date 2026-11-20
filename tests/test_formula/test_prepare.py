"""Tests for observation-independent cubic preparation."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import ArrayRowSource, DataFrameRowSource
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import DenseDesign, StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import (
    _SQLITE_CACHE_KIB,
    _SQLITE_SCOPE_BYTES,
    SetupWorkspace,
    _exact_cubic_knots,
    _exact_cubic_knots_in_memory,
    _source_batch_bytes,
    _source_row_bytes,
    prepare_fitting,
    prepare_model,
)
from jaxgam.smooths.cubic import CubicRegressionSmooth
from tests.helpers import _AssertCollector, check_that, r_available
from tests.tolerances import MODERATE, STRICT, normalize_column_signs


def _data() -> pd.DataFrame:
    x = np.repeat(np.linspace(0.0, 1.0, 12), 3)
    return pd.DataFrame({"x": x, "z": x**2, "y": np.sin(2 * np.pi * x)})


class _CappedSource(DataFrameRowSource):
    """Force preparation through a chosen physical batch partition."""

    def __init__(self, data: pd.DataFrame, cap: int) -> None:
        super().__init__(data, response="y")
        self._cap = cap

    def scan(self, batch_rows: int):
        return super().scan(min(batch_rows, self._cap))


def _coefficient_map_signature(coef_map) -> tuple:
    return (
        coef_map.total_coefs,
        coef_map.total_coefs_raw,
        coef_map.has_intercept,
        tuple(
            (
                term.label,
                term.col_start,
                term.n_coefs,
                term.n_coefs_raw,
                term.term_type,
                term.penalty_indices,
                term.del_index,
                term.Z_centering is None,
            )
            for term in coef_map.terms
        ),
    )


def test_prepared_cubic_matches_dense_setup_and_batch_boundaries() -> None:
    data = _data()
    spec = parse_formula('y ~ z + s(x, bs="cr", k=6)')
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(spec, source)
    streamed = StreamDesign(prepared, source)

    X_a = DenseDesign.materialize(streamed, 5).X
    X_b = DenseDesign.materialize(streamed, 17).X
    dense = ModelSetup.build(spec, data)
    np.testing.assert_allclose(X_a, X_b, rtol=STRICT.rtol, atol=STRICT.atol)
    np.testing.assert_allclose(X_a, dense.X, rtol=STRICT.rtol, atol=STRICT.atol)
    assert prepared.penalties is not None
    np.testing.assert_allclose(
        prepared.penalties.materialize(),
        dense.penalties.materialize(),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_prepared_numeric_alias_uses_dense_coefficient_map_contract() -> None:
    spec = parse_formula("y ~ x + z")
    for seed in (881, 882, 883):
        x = np.random.default_rng(seed).normal(size=83)
        for scale in (-1.0, 1.0, 2.0):
            data = pd.DataFrame({"x": x, "z": scale * x, "y": 1.0 + 0.3 * x})
            dense = ModelSetup.build(spec, data)
            for cap in (1, 7, 100):
                source = _CappedSource(data, cap)
                prepared = prepare_model(spec, source, family=Gaussian())
                assert prepared.predict_spec.parametric_keep_cols == tuple(
                    dense.parametric_keep_cols
                )
                assert prepared.predict_spec.dropped_param_names == tuple(
                    dense.dropped_param_names
                )
                assert prepared.predict_spec.term_names == tuple(dense.term_names)
                assert _coefficient_map_signature(
                    prepared.predict_spec.coef_map
                ) == _coefficient_map_signature(dense.coef_map)
                streamed_X = DenseDesign.materialize(
                    StreamDesign(prepared, source), 113
                ).X
                np.testing.assert_allclose(
                    streamed_X, dense.X, rtol=STRICT.rtol, atol=STRICT.atol
                )
                assert prepared.fitting is not None
                assert prepared.fitting.unpenalized_rank_deficit == 0


def test_two_cubics_with_distinct_k_and_ranges_match_dense_metadata() -> None:
    rng = np.random.default_rng(81)
    x = np.repeat(np.linspace(-2.0, 3.0, 17), 2)
    z = np.repeat(
        np.array(
            [
                -4.0,
                -3.0,
                -1.0,
                -0.5,
                0.3,
                0.9,
                1.7,
                2.1,
                4.0,
                5.0,
                7.0,
                9.0,
                10.0,
                12.0,
                16.0,
                18.0,
                20.0,
            ]
        ),
        2,
    )
    rng.shuffle(z)
    data = pd.DataFrame({"x": x, "z": z, "y": np.sin(x) + np.cos(z)})
    spec = parse_formula('y ~ s(x, bs="cr", k=6) + s(z, bs="cs", k=5)')
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(spec, source)
    dense = ModelSetup.build(spec, data)
    X = DenseDesign.materialize(StreamDesign(prepared, source), 9).X

    np.testing.assert_allclose(X, dense.X, rtol=STRICT.rtol, atol=STRICT.atol)
    np.testing.assert_allclose(
        prepared.penalties.materialize(),
        dense.penalties.materialize(),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    prepared_smooths = [
        t.smooth
        for t in prepared.predict_spec.coef_map.terms
        if t.term_type == "smooth"
    ]
    dense_smooths = [t.smooth for t in dense.coef_map.terms if t.term_type == "smooth"]
    for got, expected in zip(prepared_smooths, dense_smooths, strict=True):
        np.testing.assert_allclose(
            got._knots, expected._knots, rtol=STRICT.rtol, atol=STRICT.atol
        )


def test_prepared_rejects_unsupported_before_design_materialization() -> None:
    source = DataFrameRowSource(_data(), response="y")
    with pytest.raises(NotImplementedError, match="by-variable"):
        prepare_model(parse_formula('y ~ s(x, by=z, bs="cr", k=6)'), source)
    with pytest.raises(NotImplementedError, match="cr/cs"):
        prepare_model(parse_formula('y ~ s(x, bs="tp", k=6)'), source)


def test_stream_design_rejects_source_mutation() -> None:
    data = _data()
    columns = {"x": data["x"].to_numpy().copy()}
    source = ArrayRowSource(columns, y=data["y"].to_numpy())
    prepared = prepare_model(parse_formula('y ~ s(x, bs="cr", k=6)'), source)
    columns["x"][0] = -1.0
    with pytest.raises(RuntimeError, match="changed"):
        list(StreamDesign(prepared, source).batches(8))


def test_exact_sqlite_knots_match_dense_unique_rank_interpolation() -> None:
    x = np.repeat(np.array([0.2, 0.5, 1.3, 2.0, 4.5, 8.0, 9.0]), [4, 1, 7, 2, 3, 1, 8])
    source = DataFrameRowSource(pd.DataFrame({"x": x, "y": x}), response="y")
    knots = _exact_cubic_knots(source, "x", 4)
    expected = CubicRegressionSmooth._place_knots(np.unique(x), 4)
    np.testing.assert_allclose(knots, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    assert knots[0] == x.min()
    assert knots[-1] == x.max()


def test_fitting_preparation_matches_dense_cpu_coordinate_setup() -> None:
    data = _data()
    weights = np.linspace(0.5, 2.0, len(data))
    offset = np.linspace(-0.2, 0.3, len(data))
    spec = parse_formula('y ~ z + s(x, bs="cr", k=6)')
    family = Gaussian()
    source = DataFrameRowSource(data, response="y", weights=weights, offset=offset)
    prepared = prepare_model(spec, source, family=family)
    dense = FittingData.from_setup(
        ModelSetup.build(spec, data, weights=weights, offset=offset), family
    )

    assert prepared.fitting is not None
    np.testing.assert_allclose(
        prepared.fitting.log_lambda_init,
        np.asarray(dense.log_lambda_init),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        prepared.fitting.beta_init,
        np.asarray(dense.beta_init),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    reduction = prepared.fitting.response
    assert reduction.n_obs == len(data)
    assert reduction.total_weight == pytest.approx(float(weights.sum()))
    assert reduction.weight_min == pytest.approx(float(weights.min()))
    assert reduction.weight_max == pytest.approx(float(weights.max()))
    assert reduction.weighted_mean == pytest.approx(
        np.average(data["y"], weights=weights)
    )
    X_fit = prepared.evaluate_fitting_batch(next(source.scan(len(data))))
    np.testing.assert_allclose(
        X_fit,
        np.asarray(dense.X),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert prepared.fitting.total_penalty_rank == dense.total_penalty_rank
    assert prepared.fitting.total_penalty_null_dim == dense.total_penalty_null_dim
    assert prepared.fitting.unpenalized_rank_deficit == dense.rank_deficit


def test_prepared_basis_fingerprint_changes_with_basis_configuration() -> None:
    source = DataFrameRowSource(_data(), response="y")
    a = prepare_model(parse_formula('y ~ s(x, bs="cr", k=5)'), source)
    b = prepare_model(parse_formula('y ~ s(x, bs="cr", k=6)'), source)
    assert a.basis_fingerprint != b.basis_fingerprint


def test_prepared_prediction_smooth_drops_penalty_cache() -> None:
    source = DataFrameRowSource(_data(), response="y")
    prepared = prepare_model(parse_formula('y ~ s(x, bs="cr", k=6)'), source)
    smooth = next(
        term.smooth
        for term in prepared.predict_spec.coef_map.terms
        if term.term_type == "smooth"
    )
    assert smooth._S is None


@pytest.mark.skipif(not r_available(), reason="R with mgcv not available")
@pytest.mark.parametrize(("basis", "k"), [("cr", 6), ("cs", 5)])
def test_prepared_cubic_design_and_cr_penalty_match_r_components(
    r_bridge, basis: str, k: int
) -> None:
    """Both bases match R; cs retains the dense null-space orientation caveat."""
    data = _data()
    formula = f'y ~ s(x, bs="{basis}", k={k})'
    prepared = prepare_model(
        parse_formula(formula), DataFrameRowSource(data, response="y")
    )
    r_result = r_bridge.get_smooth_components(formula, data)
    X = DenseDesign.materialize(
        StreamDesign(prepared, DataFrameRowSource(data, response="y")), 7
    ).X
    r_X = np.column_stack((np.ones(len(data)), r_result["basis_matrices"][0]))
    np.testing.assert_allclose(
        normalize_column_signs(X),
        normalize_column_signs(r_X),
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    if basis == "cr":
        assert prepared.penalties is not None
        np.testing.assert_allclose(
            prepared.penalties.blocks[0].dense_penalties()[0],
            r_result["penalty_matrices"][0][0],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )


def test_exact_knot_scan_configures_its_sqlite_page_scope(monkeypatch) -> None:
    """SQLite pages sit outside tracemalloc; their bounded scope is configured."""
    statements: list[str] = []
    original_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    x = np.repeat(np.linspace(0.0, 1.0, 13), 3)
    knots = _exact_cubic_knots(ArrayRowSource({"x": x}), "x", 6, batch_rows=4)
    np.testing.assert_array_equal(
        knots, _exact_cubic_knots(ArrayRowSource({"x": x}), "x", 6)
    )
    assert "PRAGMA temp_store = FILE" in statements
    assert f"PRAGMA cache_size = -{_SQLITE_CACHE_KIB}" in statements
    # Main cache, four default-size ephemeral caches and the sorter run fit.
    assert _SQLITE_SCOPE_BYTES >= 1024 * (2 * _SQLITE_CACHE_KIB + 4 * 2_000)


@pytest.mark.parametrize("failure", [None, "unique", "nonfinite"])
def test_exact_knots_close_database_on_success_and_failure(monkeypatch, failure):
    """The transaction context alone does not close its SQLite connection."""
    connections = []
    original_connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    x = np.linspace(0.0, 1.0, 12)
    if failure == "nonfinite":
        x[-1] = np.nan
    source = ArrayRowSource({"x": x})
    if failure is None:
        _exact_cubic_knots(source, "x", 6)
    else:
        match = "unique data values" if failure == "unique" else "non-finite"
        with pytest.raises(ValueError, match=match):
            _exact_cubic_knots(source, "x", 20 if failure == "unique" else 6)
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connections[0].execute("SELECT 1")


# ---------------------------------------------------------------------------
# Phase-1 scan size and memory-budget control
# ---------------------------------------------------------------------------


class _RecordingSource(DataFrameRowSource):
    """Record each requested scan size and the batch lengths it yielded."""

    def __init__(self, data: pd.DataFrame, **kwargs) -> None:
        super().__init__(data, response="y", **kwargs)
        self.scans: list[tuple[int, list[int]]] = []

    def scan(self, batch_rows: int):
        lengths: list[int] = []
        self.scans.append((batch_rows, lengths))
        for batch in super().scan(batch_rows):
            lengths.append(len(batch.row_positions))
            yield batch

    def full_scans(self) -> list[tuple[int, list[int]]]:
        """Scans after prepare_model's two one-row replayability/schema peeks."""
        assert [rows for rows, _ in self.scans[:2]] == [1, 1]
        return self.scans[2:]


def _budget_data(n: int = 103, seed: int = 518) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    x2 = rng.uniform(-2.0, 3.0, size=n)
    z = rng.normal(size=n)
    y = 0.3 + np.sin(2 * np.pi * x) + 0.2 * z + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x": x, "x2": x2, "z": z, "w": -z, "y": y})


_BUDGET_FORMULA = 'y ~ z + w + s(x, bs="cr", k=6) + s(x2, bs="cs", k=5)'


def _capture_workspaces(monkeypatch) -> list:
    """Spy on the workspace each preparation entry point actually sized."""
    from jaxgam.formula import prepare as prepare_module

    captured = []
    original = prepare_module.SetupWorkspace.for_design.__func__

    def spy(cls, **kwargs):
        workspace = original(cls, **kwargs)
        captured.append((kwargs, workspace))
        return workspace

    monkeypatch.setattr(prepare_module.SetupWorkspace, "for_design", classmethod(spy))
    return captured


def test_every_setup_scan_honors_the_requested_batch_rows() -> None:
    data = _budget_data()
    source = _RecordingSource(data)
    prepared = prepare_model(
        parse_formula(_BUDGET_FORMULA), source, family=Gaussian(), batch_rows=7
    )
    # Two one-row peeks, one in-memory exact knot scan for both cubic
    # covariates (no budget), centering/alias, initial beta with the rank
    # factor, and valid-domain scans.
    assert [rows for rows, _ in source.scans] == [1, 1] + [7] * 4
    for rows, lengths in source.full_scans():
        assert max(lengths) <= rows
        assert sum(lengths) == len(data)

    direct = _RecordingSource(data)
    unfitted = prepare_model(parse_formula(_BUDGET_FORMULA), direct, batch_rows=5)
    direct.scans.clear()
    refitted = prepare_fitting(unfitted, direct, Gaussian(), batch_rows=5)
    assert [rows for rows, _ in direct.scans] == [5] * 2
    np.testing.assert_allclose(
        refitted.fitting.beta_init,
        prepared.fitting.beta_init,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )

    default = _RecordingSource(data)
    prepare_model(parse_formula(_BUDGET_FORMULA), default, family=Gaussian())
    # Standalone callers keep the 65,536-row default, capped at the source.
    assert {rows for rows, _ in default.full_scans()} == {len(data)}


def test_setup_workspace_uses_the_scanned_design_widths(monkeypatch) -> None:
    """Prospective widths are the evaluated pre-alias/pre-centering widths."""
    captured = _capture_workspaces(monkeypatch)
    data = _budget_data()
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(
        parse_formula(_BUDGET_FORMULA), source, family=Gaussian(), batch_rows=9
    )
    spec = prepared.predict_spec
    smooths = [term for term in spec.coef_map.terms if term.smooth is not None]
    (kwargs, workspace) = captured[0]
    # The w = -z alias is dropped only after scanning: both are evaluated.
    assert spec.dropped_param_names == ("w",)
    assert kwargs["n_parametric"] == 3
    assert kwargs["basis_dimensions"] == tuple(t.n_coefs_raw for t in smooths)
    assert kwargs["basis_dimensions"] == (6, 5)
    assert kwargs["null_dimensions"] == (2, 0)
    first = next(source.scan(1))
    assert kwargs["source_row_bytes"] == _source_row_bytes(first)
    # x, x2, z and w are all sliced per batch, not only formula variables.
    assert kwargs["source_batch_bytes"] == _source_batch_bytes(first)
    assert _source_row_bytes(first) == 3 * 8 + 1 + 8 + 4 * 8
    assert kwargs["n_rows"] == len(data)
    assert prepared.n_coef < kwargs["n_parametric"] + sum(kwargs["basis_dimensions"])
    assert workspace.knot_row_bytes > 0

    captured.clear()
    prepare_fitting(
        prepare_model(parse_formula(_BUDGET_FORMULA), source),
        source,
        Gaussian(),
        maximum_bytes=workspace.minimum_bytes + workspace.row_bytes,
    )
    (fitting_kwargs, fitting_workspace) = captured[-1]
    assert fitting_kwargs["n_parametric"] == kwargs["n_parametric"]
    assert fitting_kwargs["basis_dimensions"] == kwargs["basis_dimensions"]
    assert fitting_kwargs["null_dimensions"] == kwargs["null_dimensions"]
    # Fitting preparation runs no SQLite knot scan.
    assert fitting_workspace.knot_row_bytes == 0
    assert fitting_workspace.row_bytes == workspace.row_bytes


def test_setup_budget_shrinks_every_scan(monkeypatch) -> None:
    captured = _capture_workspaces(monkeypatch)
    data = _budget_data()
    spec = parse_formula(_BUDGET_FORMULA)
    prepare_model(spec, DataFrameRowSource(data, response="y"), batch_rows=64)
    workspace = captured[0][1]
    for rows in (1, 4, 11):
        # Any budget below the next row's prospective peak selects ``rows``.
        budget = workspace.bytes_for_rows(rows + 1) - 1
        source = _RecordingSource(data)
        prepared = prepare_model(
            spec, source, family=Gaussian(), batch_rows=64, maximum_bytes=budget
        )
        assert workspace.bytes_for_rows(rows) <= budget
        assert {requested for requested, _ in source.full_scans()} == {rows}
        assert all(max(lengths) <= rows for _, lengths in source.full_scans())
        assert prepared.fitting is not None
        direct = _RecordingSource(data)
        unfitted = prepare_model(spec, direct, batch_rows=64)
        direct.scans.clear()
        prepare_fitting(unfitted, direct, Gaussian(), maximum_bytes=budget)
        # A schema peek, then fitting scans sized by the knot-free workspace.
        fitting_rows = captured[-1][1].batch_rows(len(data), budget)
        assert fitting_rows >= rows
        assert [requested for requested, _ in direct.scans] == [1] + [fitting_rows] * 2


def test_setup_budget_below_one_row_is_rejected_before_full_scans(
    monkeypatch,
) -> None:
    captured = _capture_workspaces(monkeypatch)
    data = _budget_data()
    spec = parse_formula(_BUDGET_FORMULA)
    prepare_model(spec, DataFrameRowSource(data, response="y"))
    workspace = captured[0][1]
    # The SQLite knot scope is part of the one-row minimum for cubic models.
    assert workspace.minimum_bytes > _SQLITE_SCOPE_BYTES

    def no_sqlite(*_args, **_kwargs):
        raise AssertionError("knot discovery started despite a rejected budget")

    monkeypatch.setattr(sqlite3, "connect", no_sqlite)
    source = _RecordingSource(data)
    with pytest.raises(MemoryError, match=r"at least \d+ bytes to scan one row"):
        prepare_model(
            spec, source, family=Gaussian(), maximum_bytes=workspace.minimum_bytes - 1
        )
    assert [rows for rows, _ in source.scans] == [1, 1]

    monkeypatch.undo()
    unfitted = prepare_model(spec, DataFrameRowSource(data, response="y"))
    captured = _capture_workspaces(monkeypatch)
    direct = _RecordingSource(data)
    with pytest.raises(MemoryError, match="requires at least"):
        prepare_fitting(unfitted, direct, Gaussian(), maximum_bytes=4096)
    assert [rows for rows, _ in direct.scans] == [1]
    assert captured[0][1].minimum_bytes > 4096

    class EmptyReplay(DataFrameRowSource):
        def scan(self, _batch_rows: int):
            return iter(())

    with pytest.raises(RuntimeError, match="stable row set"):
        prepare_fitting(
            unfitted,
            EmptyReplay(data, response="y"),
            Gaussian(),
            maximum_bytes=workspace.minimum_bytes,
        )


def test_source_row_bytes_charge_every_sliced_column() -> None:
    data = pd.DataFrame(
        {
            "f": [1.0, 2.0],
            "i": np.array([1, 2], dtype=np.int8),
            "g": pd.Categorical(["a", "b"]),
            "o": ["u", "v"],
            "y": [0.0, 1.0],
        }
    )
    batch = next(DataFrameRowSource(data, response="y").scan(1))
    vectors = 3 * 8 + 1 + np.dtype(np.intp).itemsize  # y/weight/offset, valid, rows
    # Narrow numeric, categorical-code and object-pointer cells count as float64.
    assert _source_row_bytes(batch) == vectors + 4 * 8
    assert _source_batch_bytes(batch) == 2 * 1024 * (4 + 1)
    wide = next(ArrayRowSource({"s": np.array(["abc", "def"])}).scan(1))
    assert _source_row_bytes(wide) == vectors + np.dtype("<U3").itemsize


def test_setup_without_unpenalized_columns_has_no_rank_factor() -> None:
    x = np.linspace(0.0, 1.0, 40)
    source = _RecordingSource(pd.DataFrame({"x": x, "y": np.sin(6 * x)}))
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cs", k=5) - 1'),
        source,
        family=Gaussian(),
        batch_rows=7,
    )
    assert prepared.fitting.total_penalty_null_dim == 0
    assert prepared.fitting.unpenalized_rank_deficit == 0
    # Knot, centering, initial-beta (with the empty rank pass) and
    # valid-domain scans; every one validates every batch.
    assert [rows for rows, _ in source.full_scans()] == [7] * 4


def test_valid_domain_fallback_scan_honors_setup_batch_rows() -> None:
    """The dense initializer's fallback refit is batched like every pass."""
    x = np.linspace(0.0, 1.0, 60)
    data = pd.DataFrame({"x": x, "y": np.where(x > 0.8, 40.0, 0.0)})
    spec = parse_formula("y ~ x")
    family = Poisson(link="identity")
    dense = FittingData.from_setup(ModelSetup.build(spec, data), family)
    for batch_rows in (len(data), 7):
        source = _RecordingSource(data)
        prepared = prepare_model(spec, source, family=family, batch_rows=batch_rows)
        # Centering, initial beta with the rank factor, valid-domain and
        # fallback passes.
        assert [rows for rows, _ in source.full_scans()] == [batch_rows] * 4
        np.testing.assert_allclose(
            prepared.fitting.beta_init,
            np.asarray(dense.beta_init),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    # The fallback fits the link of the weighted response mean.
    np.testing.assert_allclose(
        prepared.fitting.beta_init, [8.0, 0.0], rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_prepare_fitting_rejects_a_different_source_before_scanning() -> None:
    data = _budget_data()
    prepared = prepare_model(
        parse_formula(_BUDGET_FORMULA), DataFrameRowSource(data, response="y")
    )
    changed = data.assign(y=data.y + 1.0)
    source = _RecordingSource(changed)
    with pytest.raises(RuntimeError, match="changed after preparation"):
        prepare_fitting(prepared, source, Gaussian(), maximum_bytes=1 << 30)
    assert source.scans == []


class _UntouchableSource(ArrayRowSource):
    """Fail if a setup entry point reads the source before validating controls."""

    def fingerprint(self) -> str:
        raise AssertionError("source fingerprint read before control validation")

    def scan(self, _batch_rows: int):
        raise AssertionError("source scanned before control validation")


@pytest.mark.parametrize(
    ("batch_rows", "maximum_bytes", "name"),
    [
        (0, None, "batch_rows"),
        (-3, None, "batch_rows"),
        (True, None, "batch_rows"),
        (2.0, None, "batch_rows"),
        ("8", None, "batch_rows"),
        (8, 0, "maximum_bytes"),
        (8, -1, "maximum_bytes"),
        (8, False, "maximum_bytes"),
        (8, 1e6, "maximum_bytes"),
    ],
)
def test_invalid_setup_controls_fail_before_source_access(
    batch_rows, maximum_bytes, name
) -> None:
    spec = parse_formula('y ~ s(x, bs="cr", k=6)')
    data = _data()
    source = _UntouchableSource({"x": data.x.to_numpy()}, y=data.y.to_numpy())
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        prepare_model(
            spec,
            source,
            family=Gaussian(),
            batch_rows=batch_rows,
            maximum_bytes=maximum_bytes,
        )
    prepared = prepare_model(spec, DataFrameRowSource(data, response="y"))
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        prepare_fitting(
            prepared,
            source,
            Gaussian(),
            batch_rows=batch_rows,
            maximum_bytes=maximum_bytes,
        )
    # NumPy integers are valid integral controls, like RowSource.scan accepts.
    prepare_model(
        spec,
        DataFrameRowSource(data, response="y"),
        batch_rows=np.int64(5),
        maximum_bytes=np.int64(512 * 1024 * 1024),
    )


class _OversizedSource(DataFrameRowSource):
    """Malicious source ignoring requested sizes after ``honest_scans`` scans."""

    def __init__(self, data: pd.DataFrame, honest_scans: int) -> None:
        super().__init__(data, response="y")
        self._honest_scans = honest_scans
        self.calls = 0

    def scan(self, batch_rows: int):
        self.calls += 1
        if self.calls <= self._honest_scans:
            return super().scan(batch_rows)
        return super().scan(self.n_rows)


@pytest.mark.parametrize("honest_scans", [0, 2, 3, 5])
def test_setup_rejects_batches_larger_than_requested(honest_scans: int) -> None:
    """Peeks, knot, centering and fitting scans each enforce their request."""
    source = _OversizedSource(_budget_data(), honest_scans)
    with pytest.raises(ValueError, match="setup scan requesting at most"):
        prepare_model(
            parse_formula(_BUDGET_FORMULA), source, family=Gaussian(), batch_rows=8
        )
    assert source.calls == honest_scans + 1


def test_setup_rejects_non_rowbatch_scan_items() -> None:
    data = _data()
    base = DataFrameRowSource(data, response="y")

    class DuckSource:
        n_rows = base.n_rows

        def fingerprint(self) -> str:
            return base.fingerprint()

        def scan(self, batch_rows: int):
            for batch in base.scan(batch_rows):
                yield SimpleNamespace(**batch.__dict__)

    with pytest.raises(TypeError, match="RowBatch"):
        prepare_model(parse_formula('y ~ s(x, bs="cr", k=6)'), DuckSource())


def _partition_fields(prepared) -> tuple[dict, dict]:
    """Split prepared state into exact decisions and STRICT reductions."""
    fitting = prepared.fitting
    spec = prepared.predict_spec
    terms = [t for t in spec.coef_map.terms if t.smooth is not None]
    exact = {
        "knots": tuple(term.smooth._knots.tobytes() for term in terms),
        "basis_fingerprint": prepared.basis_fingerprint,
        "keep": spec.parametric_keep_cols,
        "dropped": spec.dropped_param_names,
        "coef_map": _coefficient_map_signature(spec.coef_map),
        "deficit": fitting.unpenalized_rank_deficit,
        "penalty_rank": fitting.total_penalty_rank,
    }
    reductions = {
        **{f"Z{i}": term.Z_centering for i, term in enumerate(terms)},
        "penalties": prepared.penalties.materialize(),
        "beta_init": fitting.beta_init,
        "log_lambda_init": fitting.log_lambda_init,
        "gram": fitting.gram,
        "rhs": fitting.rhs,
        "weighted_sum": fitting.response.weighted_sum,
        "total_weight": fitting.response.total_weight,
    }
    return exact, reductions


@pytest.mark.parametrize("family", [Gaussian(), Poisson()], ids=["gaussian", "poisson"])
def test_prepared_model_is_invariant_to_setup_batch_partition(family) -> None:
    """Knots and decisions are exact; reductions agree at STRICT."""
    n = 1_500  # more than one canonical decision-QR leaf
    data = _budget_data(n=n, seed=519)
    if isinstance(family, Poisson):
        mu = np.exp(0.2 + 0.5 * np.sin(2 * np.pi * data.x.to_numpy()))
        data["y"] = np.random.default_rng(520).poisson(mu).astype(float)
    weights = np.random.default_rng(521).uniform(0.2, 2.0, size=n)
    offset = 0.1 * np.cos(data.x.to_numpy())
    spec = parse_formula(_BUDGET_FORMULA)
    dense = ModelSetup.build(spec, data, weights=weights, offset=offset)
    partitions = {}
    for batch_rows in (n, 256, 7, 1):
        source = DataFrameRowSource(data, response="y", weights=weights, offset=offset)
        partitions[batch_rows] = _partition_fields(
            prepare_model(spec, source, family=family, batch_rows=batch_rows)
        )
    reference_exact, reference = partitions.pop(n)
    reference_gram = reference.pop("gram")
    assert reference_exact["keep"] == tuple(dense.parametric_keep_cols)
    collector = _AssertCollector()
    for batch_rows, (exact, reductions) in partitions.items():
        gram = reductions.pop("gram")
        for name, value in exact.items():
            collector.check(
                f"{name} at batch_rows={batch_rows}",
                lambda value=value, name=name: check_that(
                    value == reference_exact[name], f"{name} changed"
                ),
            )
        for name, value in reductions.items():
            collector.check(
                f"{name} at batch_rows={batch_rows}",
                lambda value=value, name=name: np.testing.assert_allclose(
                    value, reference[name], rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
        # The unweighted Gram has structural zeros (intercept against centered
        # columns) whose values are pure summation roundoff; compare it by
        # normwise relative error instead of elementwise at that noise floor.
        collector.check(
            f"gram at batch_rows={batch_rows}",
            lambda gram=gram: check_that(
                np.linalg.norm(gram - reference_gram)
                <= STRICT.rtol * np.linalg.norm(reference_gram),
                "normwise Gram difference exceeds STRICT.rtol",
            ),
        )
    collector.raise_if_any("setup batch-partition invariance")


@pytest.mark.parametrize("scale", [-1.0, 1.0])
def test_exact_parametric_alias_decision_survives_tiny_setup_batches(
    scale: float,
) -> None:
    """Thousands of sequential re-QR merges no longer hide an exact alias.

    With one flat per-batch QR merge (the pre-budget implementation), 2,858
    seven-row merges accumulate enough roundoff on this seed that the dense
    eps**0.9 pivot tolerance keeps both columns. Canonical leaves merged in a
    tree keep the decision identical to dense for every batch size.
    """
    n = 20_000
    x = np.random.default_rng(20_000).normal(size=n)
    data = pd.DataFrame({"x": x, "z": scale * x, "y": 1.0 + 0.3 * x})
    spec = parse_formula("y ~ x + z")
    dense = ModelSetup.build(spec, data)
    for batch_rows in (7, n):
        prepared = prepare_model(
            spec,
            DataFrameRowSource(data, response="y"),
            family=Gaussian(),
            batch_rows=batch_rows,
        )
        assert prepared.predict_spec.parametric_keep_cols == tuple(
            dense.parametric_keep_cols
        )
        assert prepared.fitting.unpenalized_rank_deficit == 0


def test_canonical_tree_r_is_bitwise_invariant_to_row_chunking() -> None:
    from jaxgam.formula.prepare import _DECISION_QR_LEAF_ROWS, _CanonicalTreeR

    rng = np.random.default_rng(523)
    rows = rng.normal(size=(3 * _DECISION_QR_LEAF_ROWS + 17, 4))
    rows[:, 3] = 2.5 * rows[:, 1]
    factors = []
    for chunk in (1, 7, _DECISION_QR_LEAF_ROWS, len(rows)):
        tree = _CanonicalTreeR(rows.shape[1])
        for start in range(0, len(rows), chunk):
            tree.update(rows[start : start + chunk])
        factors.append(tree.finalize())
    for factor in factors[1:]:
        assert np.array_equal(factor, factors[0])
    np.testing.assert_allclose(
        factors[0].T @ factors[0], rows.T @ rows, rtol=STRICT.rtol, atol=STRICT.atol
    )
    empty = _CanonicalTreeR(3)
    assert empty.finalize().shape == (0, 3)
    with pytest.raises(RuntimeError, match="already finalized"):
        empty.update(rows[:2, :3])


def test_setup_workspace_bounds_traced_setup_allocations(monkeypatch) -> None:
    """NumPy/LAPACK-wrapper allocations stay inside the prospective bound.

    tracemalloc sees NumPy buffers, SciPy work arrays and Python objects. It
    cannot see SQLite's own heap, so the knot-scan allowance is excluded here.
    """
    import tracemalloc

    captured = _capture_workspaces(monkeypatch)
    n = 6_000
    data = _budget_data(n=n, seed=524)
    spec = parse_formula(_BUDGET_FORMULA)
    # Size the workspace and warm lazy imports outside the traced window.
    prepare_model(spec, DataFrameRowSource(data, response="y"), family=Gaussian())
    workspace = captured[-1][1]
    for rows in (64, 1_500):
        budget = workspace.bytes_for_rows(rows)
        source = DataFrameRowSource(data, response="y")
        tracemalloc.start()
        try:
            start, _ = tracemalloc.get_traced_memory()
            prepare_model(
                spec, source, family=Gaussian(), batch_rows=n, maximum_bytes=budget
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert captured[-1][1] == workspace
        assert workspace.batch_rows(n, budget) == rows
        # The budget leaves room to gather both cubic covariates in memory.
        knot_columns = workspace.in_memory_knot_columns(rows, budget, 2)
        assert knot_columns == 2
        traced_bound = max(
            workspace.fixed_bytes + rows * workspace.row_bytes,
            workspace.knot_fixed_bytes
            - _SQLITE_SCOPE_BYTES
            + rows * workspace.knot_row_bytes,
            workspace.in_memory_knot_bytes(rows, knot_columns),
        )
        assert peak - start <= traced_bound


# ---------------------------------------------------------------------------
# In-memory exact knots
# ---------------------------------------------------------------------------


def _knot_columns(case: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1_119)
    if case == "ties":
        x = np.repeat(rng.normal(size=40), rng.integers(1, 9, size=40))
    elif case == "few":
        # Exactly six distinct values, each repeated many times.
        x = rng.integers(0, 6, size=900) * 0.1
    elif case == "signed_zero":
        # SQLite returns +0.0 for a stored -0.0; subnormals and integral
        # values beyond 2**53 round-trip exactly.
        x = np.array(
            [-0.0, 0.0, 1e-310, -1e-310, 5e-324, 2.0**60, -3.5, 1.0, -0.0, 0.0, 7.25]
            * 3
        )
    else:
        x = np.tile(np.array([3.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0]), 50)
    x = rng.permutation(x)
    return {"x": x, "z": np.round(x[::-1] * 3.0) / 3.0}


@pytest.mark.parametrize("case", ["ties", "few", "signed_zero", "repeats"])
def test_in_memory_knots_are_bitwise_equal_to_sqlite_knots(case: str) -> None:
    columns = _knot_columns(case)
    source = ArrayRowSource(columns)
    n_unique = min(len(np.unique(value)) for value in columns.values())
    collector = _AssertCollector()
    for k in sorted({3, min(6, n_unique), n_unique}):
        for batch_rows in (1, 7, len(columns["x"])):
            in_memory = _exact_cubic_knots_in_memory(
                source, ["x", "z"], [k, k], batch_rows
            )
            for variable, knots in zip(("x", "z"), in_memory, strict=True):
                sqlite_knots = _exact_cubic_knots(source, variable, k, batch_rows)
                collector.check(
                    f"{variable} k={k} batch_rows={batch_rows}",
                    lambda a=knots, b=sqlite_knots: check_that(
                        a.tobytes() == b.tobytes(), f"{a!r} != {b!r}"
                    ),
                )
                collector.check(
                    f"{variable} k={k} matches np.unique",
                    lambda a=knots, v=columns[variable], k=k: (
                        np.testing.assert_allclose(
                            a,
                            CubicRegressionSmooth._place_knots(v, k),
                            rtol=STRICT.rtol,
                            atol=STRICT.atol,
                        )
                    ),
                )
    collector.raise_if_any(f"in-memory knots, {case}")


@pytest.mark.parametrize(
    ("x_failure", "z_failure"),
    [
        ("nonfinite", None),
        ("unique", None),
        (None, "nonfinite"),
        (None, "unique"),
        ("unique", "nonfinite"),
        ("nonfinite", "unique"),
    ],
)
def test_in_memory_knots_raise_what_one_sqlite_scan_per_covariate_raises(
    x_failure: str | None, z_failure: str | None
) -> None:
    def column(failure: str | None) -> np.ndarray:
        value = np.linspace(0.0, 1.0, 24)
        if failure == "unique":
            value = np.repeat(value[:4], 6)
        elif failure == "nonfinite":
            value[17] = np.inf
        return value

    source = ArrayRowSource({"x": column(x_failure), "z": column(z_failure)})

    def sqlite_error() -> str:
        try:
            for variable in ("x", "z"):
                _exact_cubic_knots(source, variable, 6, 5)
        except ValueError as error:
            return str(error)
        raise AssertionError("the SQLite scans accepted invalid covariates")

    with pytest.raises(ValueError, match=f"^{re.escape(sqlite_error())}$"):
        _exact_cubic_knots_in_memory(source, ["x", "z"], [6, 6], 5)


def test_in_memory_knots_reject_a_source_whose_replay_changes_length() -> None:
    class ShortReplay(ArrayRowSource):
        def scan(self, batch_rows):
            batches = list(super().scan(batch_rows))
            yield from batches[:-1]

    source = ShortReplay({"x": np.linspace(0.0, 1.0, 30)})
    with pytest.raises(RuntimeError, match="valid stable row set"):
        _exact_cubic_knots_in_memory(source, ["x"], [6], 7)


def test_setup_workspace_accounts_the_in_memory_knot_path() -> None:
    kwargs = {
        "n_parametric": 1,
        "basis_dimensions": (6, 5),
        "null_dimensions": (2, 0),
        "source_row_bytes": 64,
        "source_batch_bytes": 4_096,
        "n_rows": 1_000_000,
    }
    workspace = SetupWorkspace.for_design(**kwargs)
    assert workspace.knot_column_bytes == 8 * 1_000_000
    assert workspace.knot_reduce_bytes == (1 + np.dtype(np.intp).itemsize) * 1_000_000
    rows = 512
    one = workspace.in_memory_knot_bytes(rows, 1)
    assert one == (
        workspace.knot_fixed_bytes
        - _SQLITE_SCOPE_BYTES
        + rows * workspace.knot_row_bytes
        + workspace.knot_column_bytes
        + workspace.knot_reduce_bytes
    )
    assert workspace.in_memory_knot_columns(rows, one - 1, 2) == 0
    assert workspace.in_memory_knot_columns(rows, one, 2) == 1
    two = workspace.in_memory_knot_bytes(rows, 2)
    assert workspace.in_memory_knot_columns(rows, two, 2) == 2
    assert workspace.in_memory_knot_columns(rows, 10 * two, 2) == 2
    assert workspace.in_memory_knot_columns(rows, None, 2) == 2
    # The SQLite scan's bound does not grow with rows; the in-memory one does.
    larger = SetupWorkspace.for_design(**{**kwargs, "n_rows": 2_000_000})
    assert larger.bytes_for_rows(rows) == workspace.bytes_for_rows(rows)
    assert larger.in_memory_knot_bytes(rows, 1) > one
    fitting = SetupWorkspace.for_design(**kwargs, knot_scans=False)
    assert fitting.knot_column_bytes == fitting.knot_reduce_bytes == 0
    assert fitting.in_memory_knot_columns(rows, None, 2) == 0


@pytest.mark.parametrize("fitting_columns", [0, 1, 2])
def test_setup_budget_gate_picks_sqlite_when_the_columns_do_not_fit(
    monkeypatch, fitting_columns: int
) -> None:
    """Above the budget each covariate keeps its SQLite scan; knots agree.

    Each gathered column is charged as if it held 32 MiB, so the default
    one-row-feasible budget can be set exactly at the gate.
    """
    from jaxgam.formula import prepare as prepare_module

    data = _budget_data()
    spec = parse_formula(_BUDGET_FORMULA)
    reference = prepare_model(spec, DataFrameRowSource(data, response="y"))
    column_bytes = 32 * 1024 * 1024
    original = prepare_module.SetupWorkspace.for_design.__func__
    sized = []

    def heavy_columns(cls, **kwargs):
        workspace = original(cls, **kwargs)
        workspace = replace(workspace, knot_column_bytes=column_bytes)
        sized.append(workspace)
        return workspace

    monkeypatch.setattr(
        prepare_module.SetupWorkspace, "for_design", classmethod(heavy_columns)
    )
    prepare_model(spec, DataFrameRowSource(data, response="y"))
    rows = 64
    workspace = sized[-1]
    budget = workspace.in_memory_knot_bytes(rows, max(fitting_columns, 1))
    if fitting_columns == 0:
        budget -= 1
    assert workspace.bytes_for_rows(rows) <= budget

    connections = []
    in_memory_scans = []
    original_connect = sqlite3.connect
    original_in_memory = prepare_module._exact_cubic_knots_in_memory

    def counted_connect(*args, **kwargs):
        connections.append(1)
        return original_connect(*args, **kwargs)

    def counted_in_memory(source, variables, *args):
        in_memory_scans.append(tuple(variables))
        return original_in_memory(source, variables, *args)

    monkeypatch.setattr(sqlite3, "connect", counted_connect)
    monkeypatch.setattr(
        prepare_module, "_exact_cubic_knots_in_memory", counted_in_memory
    )
    prepared = prepare_model(
        spec,
        DataFrameRowSource(data, response="y"),
        batch_rows=rows,
        maximum_bytes=budget,
    )
    if fitting_columns == 0:
        assert len(connections) == 2
        assert in_memory_scans == []
    elif fitting_columns == 1:
        assert connections == []
        assert in_memory_scans == [("x",), ("x2",)]
    else:
        assert connections == []
        assert in_memory_scans == [("x", "x2")]
    for observed, expected in zip(
        prepared.predict_spec.coef_map.terms,
        reference.predict_spec.coef_map.terms,
        strict=True,
    ):
        if observed.smooth is not None:
            assert (
                np.asarray(observed.smooth._knots).tobytes()
                == np.asarray(expected.smooth._knots).tobytes()
            )
    assert prepared.basis_fingerprint == reference.basis_fingerprint


def test_in_memory_knot_scan_stays_inside_its_traced_bound() -> None:
    import tracemalloc

    n = 200_000
    rng = np.random.default_rng(526)
    source = ArrayRowSource(
        {"x": rng.uniform(size=n), "z": np.round(rng.normal(size=n), 2)},
        y=rng.normal(size=n),
    )
    first = next(source.scan(1))
    workspace = SetupWorkspace.for_design(
        n_parametric=1,
        basis_dimensions=(10, 10),
        null_dimensions=(2, 2),
        source_row_bytes=_source_row_bytes(first),
        source_batch_bytes=_source_batch_bytes(first),
        n_rows=n,
    )
    rows = 4_096
    _exact_cubic_knots_in_memory(source, ["x"], [10], rows)  # warm imports
    tracemalloc.start()
    try:
        start, _ = tracemalloc.get_traced_memory()
        _exact_cubic_knots_in_memory(source, ["x", "z"], [10, 10], rows)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The gathered columns dominate: the bound is not vacuous at this size.
    assert peak - start >= 2 * workspace.knot_column_bytes
    assert peak - start <= workspace.in_memory_knot_bytes(rows, 2)


def test_prepared_batches_read_only_the_formula_columns(monkeypatch) -> None:
    from jaxgam.data import source as source_module

    rng = np.random.default_rng(527)
    n = 50
    columns = {f"unused{index}": rng.normal(size=n) for index in range(6)}
    columns["x"] = rng.uniform(size=n)
    columns["z"] = rng.normal(size=n)
    source = ArrayRowSource(columns, y=rng.normal(size=n))
    read: list[int] = []
    slice_column = source_module._slice

    def counted_slice(value, positions):
        read.append(id(value))
        return slice_column(value, positions)

    monkeypatch.setattr(source_module, "_slice", counted_slice)
    prepared = prepare_model(
        parse_formula('y ~ z + s(x, bs="cr", k=6)'), source, family=Gaussian()
    )
    used = {id(columns["x"]), id(columns["z"])}
    unused = {id(value) for name, value in columns.items() if name.startswith("unused")}
    # The two one-row schema peeks still size every column of the source.
    assert sum(identity in unused for identity in read) <= 2 * len(unused)
    read.clear()
    for batch in source.scan(7):
        prepared.evaluate_fitting_batch(batch)
    assert read
    assert set(read) <= used
