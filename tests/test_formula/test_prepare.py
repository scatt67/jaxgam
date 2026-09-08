"""Tests for observation-independent cubic preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import ArrayRowSource, DataFrameRowSource
from jaxgam.families.standard import Gaussian
from jaxgam.fitting.data import FittingData
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import DenseDesign, StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import _exact_cubic_knots, prepare_model
from jaxgam.smooths.cubic import CubicRegressionSmooth
from tests.helpers import r_available
from tests.tolerances import MODERATE, STRICT, normalize_column_signs


def _data() -> pd.DataFrame:
    x = np.repeat(np.linspace(0.0, 1.0, 12), 3)
    return pd.DataFrame({"x": x, "z": x**2, "y": np.sin(2 * np.pi * x)})


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
def test_prepared_cubic_design_and_penalty_match_r_components(
    r_bridge, basis: str, k: int
) -> None:
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
