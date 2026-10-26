"""Indexed preparation: dense/source parity and prospective memory gates."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import ArrayRowSource, DataFrameRowSource
from jaxgam.formula import operator_prepare
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.penalties.structure import IdentityPenalty, IdentityTransform
from jaxgam.smooths.constraints import CoefficientMap
from jaxgam.smooths.random_effects import RandomEffectSmooth
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


def _frame():
    n = 48
    x = np.linspace(-1, 1, n)
    return pd.DataFrame(
        {
            "x": x,
            "u": x - 0.4,
            "y": np.sin(x),
            "g": pd.Categorical(
                np.resize(["a", "b", "c"], n), categories=["c", "a", "b", "unused"]
            ),
            "h": pd.Categorical(np.resize(["j", "i"], n), categories=["j", "i"]),
        }
    )


def _expanded(rows):
    """Small test-only explicit oracle; production never expands RE rows."""
    n = len(rows.modest_values)
    X = np.zeros((n, rows.n_coef))
    X[:, rows.modest_columns] = rows.modest_values
    for ids, amplitude, start, _ in rows.indexed_rows:
        X[np.arange(n), start + ids] += amplitude
    return X


@pytest.mark.parametrize(
    "formula",
    [
        "y~s(g,bs='re')",
        "y~0+s(u,bs='re')",
        "y~s(u,g,h,bs='re')",
        "y~s(g,bs='re')+s(x,bs='cr',k=5)+s(u,h,bs='re')+s(g,h,bs='re')",
        "y~x+s(x,bs='cs',k=5)",
        "y~s(g,bs='re')+s(x,bs='cc',k=6)+s(u,h,bs='re')",
    ],
)
def test_indexed_preparation_matches_dense_original_placement_and_penalties(formula):
    frame = _frame()
    spec = parse_formula(formula)
    dense = ModelSetup.build(spec, frame)
    collector = _AssertCollector()
    for size in (1, 7, 19):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=size)
        )
        X = np.vstack([_expanded(model.evaluate_batch(b)) for b in source.scan(size)])
        collector.check(
            f"B{size}/design",
            lambda X=X: np.testing.assert_allclose(
                X, dense.X, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        for logsp in (
            np.zeros(model.penalties.n_penalties),
            np.linspace(-3, 2, model.penalties.n_penalties),
        ):
            collector.check(
                f"B{size}/penalty",
                lambda logsp=logsp, model=model: np.testing.assert_allclose(
                    model.penalties.materialize(logsp),
                    dense.penalties.materialize(logsp),
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
        assert model.completed_preparation_scans >= 2
        assert model.preparation_peak_bound_bytes <= 128 * 1024 * 1024
    collector.raise_if_any(formula)


def test_cyclic_metadata_and_wrapped_replay_match_pinned_source(r_bridge):
    frame = _frame()
    expression = "s(x,bs='cc',k=6)"
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~0+" + expression),
        source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    expected = r_bridge.smooth_construct(expression, frame, absorb_cons=True)
    X = np.vstack([model.evaluate_batch(b).modest_values for b in source.scan(7)])
    smooth = model.modest.predict_spec.coef_map.terms[0].smooth
    collector = _AssertCollector()
    for name, actual, reference in (
        ("centered original basis", X, expected["X"]),
        ("normalized cyclic penalty", model.penalties.materialize(), expected["S"][0]),
        ("cyclic rank", smooth.rank, 4),
        ("cyclic null space", smooth.null_space_dim, 1),
    ):
        collector.check(
            name,
            lambda actual=actual, reference=reference: np.testing.assert_allclose(
                actual, reference, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    period = smooth._knots[-1] - smooth._knots[0]
    heldout = {"x": np.array([-0.83, 0.17, 0.64])}
    wrapped = {"x": heldout["x"] + 3 * period}
    collector.check(
        "held-out periodic replay",
        lambda: np.testing.assert_allclose(
            model.modest.predict_spec.build_predict_matrix(heldout),
            model.modest.predict_spec.build_predict_matrix(wrapped),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    assert smooth._X is None
    collector.raise_if_any(expression)


def test_cyclic_minimum_width_is_charged_before_modest_preparation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        operator_prepare, "prepare_model", lambda *args: calls.append(args)
    )
    with pytest.raises(MemoryError, match="width 5"):
        prepare_operator_model(
            parse_formula("y~s(x,bs='cc',k=3)"),
            DataFrameRowSource(_frame(), response="y"),
            control=OperatorPreparationControl(max_modest_coefficients=4),
        )
    assert not calls


@pytest.mark.parametrize(
    "expression", ["s(g,bs='re')", "s(g,h,bs='re')", "s(u,g,h,bs='re')", "s(u,bs='re')"]
)
def test_indexed_re_matches_pinned_constructor_and_normalization(r_bridge, expression):
    frame = _frame()
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(parse_formula("y~0+" + expression), source)
    expected = r_bridge.smooth_construct(expression, frame, absorb_cons=True)
    X = _expanded(model.evaluate_batch(next(source.scan(len(frame)))))
    collector = _AssertCollector()
    for name, actual, reference in (
        ("interaction ordering", X, expected["X"]),
        ("smoothCon normalization", model.penalties.materialize(), expected["S"][0]),
        ("full rank", model.random_effects[0].n_coef, expected["rank"]),
        ("no null space", 0, expected["null_space_dim"]),
    ):
        collector.check(
            name,
            lambda actual=actual, reference=reference: np.testing.assert_allclose(
                actual, reference, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any(expression)


def test_selection_repeated_lineage_pickle_and_unseen_levels():
    frame = _frame()
    selection = np.array([47, 2, 7, 2, 0, 30])
    columns = {name: frame[name] for name in ("g", "h", "u")}
    source = ArrayRowSource(columns, y=frame.y, row_selection=selection)
    spec = parse_formula("y~s(g,h,bs='re')+s(u,g,bs='re')")
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=2)
    )
    dense = ModelSetup.build(spec, frame.iloc[selection])
    np.testing.assert_allclose(
        np.vstack([_expanded(model.evaluate_batch(b)) for b in source.scan(2)]),
        dense.X,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert model.random_effects[0].factors[0] is model.random_effects[1].factors[0]
    restored = pickle.loads(pickle.dumps(model))
    assert (
        restored.random_effects[0].factors[0] is restored.random_effects[1].factors[0]
    )
    with pytest.raises(TypeError):
        restored.random_effects[0].factors[0].lookup["bad"] = 3
    columns = {
        "g": np.array(["a", "novel", "c"]),
        "h": np.array(["j", "i", "novel"]),
        "u": np.array([2.0, 3.0, np.nan]),
    }
    term = restored.random_effects[0]
    ids, amplitude = term.evaluate(columns)
    assert ids[0] == 1  # g is the first, fastest-varying margin.
    np.testing.assert_array_equal(amplitude, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        restored.random_effects[1].evaluate(columns)[1], [2.0, 0.0, 0.0]
    )


def test_symbolic_large_cartesian_dimension_never_uses_dense_re_setup(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Forbidden dense random-effect/global preparation")

    for cls, name in (
        (RandomEffectSmooth, "setup"),
        (IdentityPenalty, "dense"),
        (IdentityTransform, "dense"),
        (CoefficientMap, "build"),
    ):
        monkeypatch.setattr(cls, name, forbidden)
    n_levels = 1000
    frame = pd.DataFrame(
        {
            "g": pd.Categorical(
                ["g1", "g0"], categories=[f"g{i}" for i in range(n_levels)]
            ),
            "h": pd.Categorical(
                ["h2", "h1"], categories=[f"h{i}" for i in range(n_levels)]
            ),
            "y": [1.0, 2.0],
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(parse_formula("y~s(g,h,bs='re')"), source)
    assert model.n_coef == 1_000_001
    assert model.penalties.blocks[0].local_penalties[0].size == 1_000_000
    ids, amplitude = model.random_effects[0].evaluate(frame)
    np.testing.assert_array_equal(ids, [2001, 1000])
    np.testing.assert_array_equal(amplitude, 1.0)
    assert model.preparation_peak_bound_bytes < 3_000_000


def test_hard_modest_and_label_budgets_reject_before_preparation(monkeypatch):
    monkeypatch.setattr(
        operator_prepare,
        "prepare_model",
        lambda *_a, **_k: pytest.fail("Predictor/preparation was called"),
    )
    frame = _frame()
    source = DataFrameRowSource(frame, response="y")
    with pytest.raises(MemoryError, match="hard cap"):
        prepare_operator_model(parse_formula("y~s(x,bs='cr',k=100000)"), source)
    with pytest.raises(MemoryError):
        prepare_operator_model(
            parse_formula("y~s(g,bs='re')"),
            source,
            control=OperatorPreparationControl(memory_budget_bytes=1),
        )
    long = DataFrameRowSource(
        pd.DataFrame({"g": ["x" * 5000], "y": [1.0]}), response="y"
    )
    with pytest.raises(MemoryError):
        prepare_operator_model(
            parse_formula("y~s(g,bs='re')"),
            long,
            control=OperatorPreparationControl(batch_rows=1, memory_budget_bytes=12000),
        )


def test_batch_budget_rejects_before_predictor_and_preserves_bound(monkeypatch):
    source = DataFrameRowSource(_frame(), response="y")
    model = prepare_operator_model(
        parse_formula("y~s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=3),
    )
    monkeypatch.setattr(
        type(model.modest),
        "evaluate_batch",
        lambda *_a: pytest.fail("Predictor evaluated"),
    )
    with pytest.raises(MemoryError, match="workspace"):
        model.evaluate_batch(next(source.scan(3)), available_workspace_bytes=1)
    with pytest.raises(MemoryError, match="row bound"):
        model.evaluate_batch(next(source.scan(4)))


@pytest.mark.parametrize("field", ["y", "weight", "offset", "positions", "column"])
def test_metadata_prefix_replay_checks_payload_before_yield(field):
    class PrefixChanged(DataFrameRowSource):
        def scan(self, batch_rows):
            for batch in super().scan(batch_rows):
                if batch_rows == 1:
                    if field == "positions":
                        batch = replace(batch, row_positions=batch.row_positions + 1)
                    elif field == "column":
                        columns = dict(batch.columns)
                        columns["irrelevant"] = columns["irrelevant"] + 1.0
                        batch = replace(batch, columns=columns)
                    else:
                        batch = replace(batch, **{field: getattr(batch, field) + 1.0})
                yield batch

        def fingerprint(self):
            return "dishonestly-stable-prefix-fingerprint"

    frame = _frame()
    frame["irrelevant"] = np.linspace(0, 1, len(frame))
    with pytest.raises(RuntimeError, match="changed its operator replay"):
        prepare_operator_model(
            parse_formula("y~s(g,bs='re')+s(x,bs='cr',k=5)"),
            PrefixChanged(frame, response="y"),
            control=OperatorPreparationControl(batch_rows=7),
        )


def test_prefix_header_reserve_rejects_before_serializing_payload(monkeypatch):
    source = DataFrameRowSource(_frame(), response="y")
    batch = next(source.scan(1))
    row_bound = 512 * (len(batch.columns) + 4) + sum(
        16 * len(name) for name in batch.columns
    )
    checked = operator_prepare._CheckedSource(
        source,
        OperatorPreparationControl(
            batch_rows=1,
            memory_budget_bytes=row_bound + operator_prepare._REPLAY_HEADER_BYTES - 1,
        ),
        0,
    )
    monkeypatch.setattr(
        operator_prepare.pickle,
        "dumps",
        lambda *_args, **_kwargs: pytest.fail("unbudgeted payload serialization"),
    )
    with pytest.raises(MemoryError, match="preparation"):
        list(checked.scan(1))


def test_completed_replay_header_is_constant_metadata_and_validates_prefixes():
    source = DataFrameRowSource(_frame(), response="y")
    checked = operator_prepare._CheckedSource(source, OperatorPreparationControl(), 0)
    list(checked.scan(7))
    signature = checked.first_row_digest
    assert isinstance(signature, str)
    assert len(signature) == 64
    for B in (1, 4, 9):
        prefix = next(checked.scan(B))
        assert len(prefix.row_positions) == B
        assert checked.first_row_digest is signature
    list(checked.scan(11))
    assert checked.first_row_digest == signature
    assert checked.completed_scans == 2


class _BrokenSource(DataFrameRowSource):
    def __init__(self, frame, mode):
        super().__init__(frame, response="y")
        self.mode = mode
        self.calls = 0

    def scan(self, batch_rows):
        self.calls += 1
        call = self.calls
        batches = list(super().scan(batch_rows))
        if self.mode == "under":
            batches = batches[:-1]
        if self.mode == "over":
            batches += batches[-1:]
        for b in batches:
            if self.mode == "float_positions":
                b = replace(b, row_positions=b.row_positions.astype(float))
            elif self.mode == "negative_positions":
                b = replace(b, row_positions=-np.ones(len(b.row_positions), dtype=int))
            elif self.mode == "changed" and call > 1:
                b = replace(b, y=b.y + 1.0)
            elif self.mode == "schema" and call > 1:
                b = replace(
                    b, columns={**b.columns, "x": b.columns["x"].astype(np.float32)}
                )
            elif self.mode == "invalid_rows":
                b = replace(b, valid=np.zeros(len(b.row_positions), dtype=bool))
            elif self.mode == "invalid_weight":
                b = replace(b, weight=np.full(len(b.row_positions), np.nan))
            yield b

    def fingerprint(self):
        return "dishonestly-stable-fingerprint"


@pytest.mark.parametrize(
    "mode",
    [
        "under",
        "over",
        "float_positions",
        "negative_positions",
        "changed",
        "schema",
        "invalid_rows",
        "invalid_weight",
    ],
)
def test_source_count_metadata_and_changed_replay_reject(mode):
    with pytest.raises((ValueError, RuntimeError)):
        prepare_operator_model(
            parse_formula("y~s(g,bs='re')"),
            _BrokenSource(_frame(), mode),
            control=OperatorPreparationControl(batch_rows=7),
        )


@pytest.mark.parametrize(
    "formula",
    [
        "y~s(g,by=u,bs='re')",
        "y~s(g,bs='re',id=1)",
        "y~g",
    ],
)
def test_unimplemented_geometry_rejects_explicitly(formula):
    with pytest.raises(NotImplementedError):
        prepare_operator_model(
            parse_formula(formula), DataFrameRowSource(_frame(), response="y")
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"batch_rows": False},
        {"memory_budget_bytes": 0},
        {"max_modest_coefficients": 1.5},
    ],
)
def test_preparation_controls_validate(arguments):
    with pytest.raises(ValueError, match="positive integer"):
        OperatorPreparationControl(**arguments)


def test_mixed_declared_factor_values_keep_types_and_shared_positional_terms():
    frame = pd.DataFrame(
        {
            "g": pd.Categorical(["a", 12, 3.5, 12], categories=["a", 12, 3.5]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(g,bs='re')+s(g,bs='re',k=2)"), source
    )
    first, second = model.random_effects
    assert first.term_id == 0
    assert second.term_id == 1
    assert first.sp_index == 0
    assert second.sp_index == 1
    assert first.factors[0] is second.factors[0]
    assert first.factors[0].levels == ("a", 12, 3.5)
    np.testing.assert_array_equal(first.evaluate(frame)[0], [0, 1, 2, 1])


@pytest.mark.parametrize("case", ["missing", "nan", "infinity", "overflow", "empty"])
def test_invalid_re_input_rejects(case):
    frame = _frame()
    formula = "y~s(g,bs='re')"
    if case == "missing":
        frame = frame.drop(columns="g")
    elif case == "nan":
        frame.loc[0, "g"] = np.nan
    elif case in ("infinity", "overflow"):
        frame["u"] = np.inf if case == "infinity" else 1e200
        formula = "y~s(u,bs='re')"
    elif case == "empty":
        frame = frame.iloc[:0]
    with pytest.raises(ValueError, match=r"missing|finite|overflow|nonempty"):
        prepare_operator_model(
            parse_formula(formula), DataFrameRowSource(frame, response="y")
        )


def test_zero_numeric_re_uses_unscaled_identity():
    frame = _frame()
    frame["u"] = 0.0
    model = prepare_operator_model(
        parse_formula("y~0+s(u,g,bs='re')"), DataFrameRowSource(frame, response="y")
    )
    assert model.random_effects[0].penalty_scale == 1.0
    np.testing.assert_array_equal(model.random_effects[0].evaluate(frame)[1], 0.0)


def test_fixed_string_replay_budget_rejects_before_serialization(monkeypatch):
    source = ArrayRowSource({"g": np.array(["x" * 5000])}, y=[1.0])
    monkeypatch.setattr(
        operator_prepare.pickle,
        "dumps",
        lambda *_a, **_k: pytest.fail("Replay serialization allocated"),
    )
    with pytest.raises(MemoryError, match="budget"):
        prepare_operator_model(
            parse_formula("y~s(g,bs='re')"),
            source,
            control=OperatorPreparationControl(batch_rows=1, memory_budget_bytes=12000),
        )


def test_plain_string_level_discovery_matches_dense_without_categorical_schema():
    frame = _frame()
    frame["g"] = frame.g.astype(str)
    frame["h"] = frame.h.astype(str)
    spec = parse_formula("y~s(u,g,h,bs='re')")
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(spec, source)
    dense = ModelSetup.build(spec, frame)
    assert model.random_effects[0].factors[0].levels == ("a", "b", "c")
    np.testing.assert_allclose(
        _expanded(model.evaluate_batch(next(source.scan(len(frame))))),
        dense.X,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_replay_streams_many_categorical_columns_without_batch_materialization(
    monkeypatch,
):
    n = 100
    frame = pd.DataFrame({"y": np.arange(n, dtype=float)})
    extras = {
        f"unused_{i}": pd.Series(pd.Categorical(np.resize(["a", "b"], n)))
        for i in range(200)
    }
    frame = pd.concat([frame, pd.DataFrame(extras)], axis=1)

    class Source(DataFrameRowSource):
        def fingerprint(self):
            return "stable-external-fingerprint"

    source = Source(frame, response="y")
    original = pd.Series.__array__

    def forbidden_conversion(series, *args, **kwargs):
        if str(series.name).startswith("unused_"):
            pytest.fail("Replay allocated an irrelevant categorical batch array")
        return original(series, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "__array__", forbidden_conversion)
    checked = operator_prepare._CheckedSource(
        source, OperatorPreparationControl(batch_rows=n, memory_budget_bytes=140000), 0
    )
    list(checked.scan(n))
    first_digest = checked.digest
    list(checked.scan(7))
    assert checked.digest == first_digest
    assert checked.peak_bound_bytes <= 140000
    # Irrelevant columns remain part of the complete replay contract.
    source._columns["unused_0"].iloc[0] = "b"
    with pytest.raises(RuntimeError, match="replay"):
        list(checked.scan(13))


def test_replay_bounds_serialized_column_names_before_payload_allocation(monkeypatch):
    source = ArrayRowSource({"x" * 10000: np.array([1.0])}, y=[2.0])
    monkeypatch.setattr(
        operator_prepare.pickle,
        "dumps",
        lambda *_a, **_k: pytest.fail("Oversize replay payload allocated"),
    )
    checked = operator_prepare._CheckedSource(
        source, OperatorPreparationControl(batch_rows=1, memory_budget_bytes=12000), 0
    )
    with pytest.raises(MemoryError, match="budget"):
        list(checked.scan(1))


@pytest.mark.parametrize("changed_count", [3.0, True, 2])
def test_replay_rejects_changed_row_count_metadata_without_lossy_coercion(
    changed_count,
):
    original = ArrayRowSource({"x": np.arange(3.0)}, y=np.ones(3))

    class Source:
        n_rows = 3

        def fingerprint(self):
            return "external-stable-version"

        def scan(self, rows):
            yield from original.scan(rows)

    source = Source()
    checked = operator_prepare._CheckedSource(
        source, OperatorPreparationControl(batch_rows=2), 0
    )
    list(checked.scan(2))
    source.n_rows = changed_count
    with pytest.raises(RuntimeError, match="row count metadata"):
        list(checked.scan(2))
