"""Bounded cubic/by/tensor geometry, source penalties and expanded layout."""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from jaxgam import GAM, FitControl, OperatorControl
from jaxgam.data.source import ArrayRowSource, DataFrameRowSource
from jaxgam.formula import operator_modest
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.inference.operator_geometry import OperatorPredictionGeometry
from jaxgam.smooths.source_lapack import (
    cubic_shrinkage_workspace_bound,
    scipy_uses_accelerate_lapack,
)
from tests.helpers import _AssertCollector, r_links_pinned_openblas
from tests.tolerances import STRICT


def _frame(ordered=False):
    x = np.linspace(-1, 1, 53)
    return pd.DataFrame(
        {
            "x": x,
            "y": np.sin(x),
            "z": 1.2 + x**2,
            "fac": pd.Categorical(
                np.resize(["b", "a", "c"], len(x)),
                categories=["c", "b", "a"],
                ordered=ordered,
            ),
            "g": pd.Categorical(np.resize(["g1", "g2"], len(x))),
        }
    )


def _matrix(model, batch):
    rows = model.evaluate_batch(batch)
    result = np.zeros((len(batch.row_positions), model.n_coef))
    result[:, rows.modest_columns] = rows.modest_values
    for ids, amplitude, start, _ in rows.indexed_rows:
        result[np.arange(len(result)), start + ids] += amplitude
    return result


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("by", ["z", "fac"])
def test_by_geometry_and_global_penalty_order_match_dense(ordered, by, monkeypatch):
    frame = _frame(ordered)
    formula = f"y~s(g,bs='re')+s(x,bs='cs',k=5,by={by})+s(x,bs='cr',k=4)"
    spec = parse_formula(formula)
    expected = ModelSetup.build(spec, frame)
    monkeypatch.setattr(
        ModelSetup, "build", lambda *_args, **_kwargs: pytest.fail("global dense setup")
    )
    collector = _AssertCollector()
    for B in (1, 7, 19):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        actual = np.vstack([_matrix(model, batch) for batch in source.scan(B)])
        rho = np.linspace(-2, 1, expected.penalties.n_penalties)
        assert model.penalties.n_penalties == expected.penalties.n_penalties
        for name, value, reference in (
            ("original basis", actual, expected.X),
            (
                "expanded sp ordering",
                model.penalties.materialize(rho),
                expected.penalties.materialize(rho),
            ),
        ):
            collector.check(
                f"B{B}/{name}",
                lambda value=value, reference=reference: np.testing.assert_allclose(
                    value, reference, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
        assert model.preparation_peak_bound_bytes <= 128 * 1024 * 1024
    collector.raise_if_any(formula)


@pytest.mark.parametrize("constant", [0.0, 3.0, -3.0])
@pytest.mark.parametrize("basis", ["cr", "cs"])
def test_constant_numeric_by_constraint_and_penalty_match_pinned(
    r_bridge, constant, basis
):
    frame = _frame()
    frame.z = constant
    expression = f"s(x,bs='{basis}',k=5,by=z)"
    expected = r_bridge.smooth_construct(expression, frame, absorb_cons=True)
    # A cs penalty depends on the basis R's LAPACK returns for the penalty
    # null space (see test_cubic_shrinkage_raw_penalties_match_pinned_r_...
    # in tests/test_smooths/test_source_lapack.py, which checks what does not
    # depend on it). Its entries are compared with R's only when this side
    # runs R's provider.
    penalty_is_comparable = (
        basis != "cs"
        or r_links_pinned_openblas(r_bridge)
        or scipy_uses_accelerate_lapack()
    )
    collector = _AssertCollector()
    for B in (1, 7, 19):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            parse_formula("y~0+" + expression),
            source,
            control=OperatorPreparationControl(batch_rows=B),
        )
        actual = np.vstack(
            [model.evaluate_batch(b).modest_values for b in source.scan(B)]
        )
        fields = [("base constraint before by", actual, expected["X"])]
        if penalty_is_comparable:
            fields.append(
                ("source penalty", model.penalties.materialize(), expected["S"][0])
            )
        for name, value, reference in fields:
            collector.check(
                f"{constant}/B{B}/{name}",
                lambda value=value, reference=reference: np.testing.assert_allclose(
                    value, reference, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
    collector.raise_if_any(expression)


def test_factor_expansion_rejects_before_local_basis_allocation(monkeypatch):
    frame = _frame()
    frame.fac = pd.Categorical(
        np.resize(["l1", "l2", "l3"], len(frame)),
        categories=[f"l{i}" for i in range(1000)],
    )
    calls = []
    monkeypatch.setattr(
        operator_modest, "_cubic_from_knots", lambda *args: calls.append(args)
    )
    with pytest.raises(MemoryError, match="hard cap"):
        prepare_operator_model(
            parse_formula("y~s(x,bs='cs',k=5,by=fac)"),
            DataFrameRowSource(frame, response="y"),
            control=OperatorPreparationControl(batch_rows=7),
        )
    assert not calls


def test_cs_provider_workspace_rejects_before_basis_construction(monkeypatch):
    term = parse_formula("y~s(x,bs='cs',k=65)").smooth_terms[0]
    knots = np.linspace(-1.0, 1.0, 65)
    retained = 4096
    workspace = cubic_shrinkage_workspace_bound(65)
    source = SimpleNamespace(reserved_bytes=retained, peak_bound_bytes=retained)
    calls = []
    monkeypatch.setattr(
        operator_modest,
        "_cubic_from_knots",
        lambda *args, **_kwargs: calls.append(args),
    )
    with pytest.raises(MemoryError, match="needs at most"):
        operator_modest._cubic_with_budget(
            term,
            knots,
            source,
            OperatorPreparationControl(
                batch_rows=1, memory_budget_bytes=retained + workspace - 1
            ),
        )
    assert not calls


@pytest.mark.parametrize(
    ("by", "ordered"), [("z", False), ("fac", False), ("fac", True)]
)
def test_public_by_fit_retains_actual_penalty_layout_and_predictions(
    r_bridge, by, ordered
):
    frame = _frame(ordered)
    frame.y = 1.0 + 0.2 * np.sin(frame.x)
    frame["w"] = np.linspace(0.6, 1.4, len(frame))
    frame.loc[::7, "w"] = 0.0
    frame["o"] = 0.01 * frame.x
    formula = f"y~s(g,bs='re')+s(x,bs='cr',k=5,by={by})"
    sp = (
        np.array([0.8, 1.2])
        if by == "z"
        else np.array([0.8, 1.2, 0.6])
        if ordered
        else np.array([0.8, 1.2, 0.6, 0.4])
    )
    reference = r_bridge.fit_fixed_sp(
        formula, frame, "gaussian", sp, weights="w", offset="o"
    )
    collector = _AssertCollector()
    for B in (1, 7, 19):
        result = GAM(
            formula,
            sp=sp,
            control=FitControl(
                execution="operator",
                linear_solver="pcg",
                batch_rows=B,
                operator=OperatorControl(cg_rtol=1e-13, cg_atol=1e-14),
            ),
        ).fit(
            DataFrameRowSource(frame, response="y", weights=frame.w, offset=frame.o),
            result="prediction",
        )
        assert result.converged
        for name, actual, expected in (
            ("coefficients", result.coefficients, reference["coefficients"]),
            ("deviance", result.deviance, reference["deviance"]),
            (
                "means",
                result.predict(frame, offset=frame.o),
                reference["fitted_values"],
            ),
        ):
            collector.check(
                f"{by}/{ordered}/B{B}/{name}",
                lambda actual=actual, expected=expected: np.testing.assert_allclose(
                    actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
    collector.raise_if_any("Public bounded by fit")


@pytest.mark.parametrize(
    "expression",
    [
        "te(x,z,bs='cr',k=4)",
        "ti(x,z,bs='cr',k=4)",
        "te(x,z,bs='cc',k=4)",
        "ti(x,z,bs='cc',k=4)",
        "te(x,z,bs='cr',k=4,by=fac)",
        "ti(x,z,bs='cr',k=4,by=z)",
    ],
)
def test_capped_tensor_metadata_and_penalty_layout_match_dense(expression):
    frame = _frame()
    formula = "y~s(g,bs='re')+" + expression
    spec = parse_formula(formula)
    dense = ModelSetup.build(spec, frame)
    collector = _AssertCollector()
    for B in (1, 7, 19):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        X = np.vstack([_matrix(model, batch) for batch in source.scan(B)])
        rho = np.linspace(-2, 1, dense.penalties.n_penalties)
        assert model.penalties.n_penalties == dense.penalties.n_penalties
        for name, actual, expected in (
            ("tensor original basis", X, dense.X),
            (
                "margin and level sp ordering",
                model.penalties.materialize(rho),
                dense.penalties.materialize(rho),
            ),
        ):
            collector.check(
                f"B{B}/{name}",
                lambda actual=actual, expected=expected: np.testing.assert_allclose(
                    actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
    collector.raise_if_any(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "te(x,z,bs='cr',k=4)",
        "ti(x,z,bs='cr',k=4)",
        "te(x,z,bs='cc',k=4)",
        "ti(x,z,bs='cc',k=4)",
        "te(x,z,bs='cr',k=4,by=fac)",
        "ti(x,z,bs='cr',k=4,by=z)",
    ],
)
def test_tensor_original_basis_and_penalties_match_pinned(r_bridge, expression):
    frame = _frame()
    expected = r_bridge.smooth_construct_list(expression, frame, absorb_cons=True)
    expected_X = np.column_stack([smooth["X"] for smooth in expected])
    expected_S = linalg.block_diag(*[sum(smooth["S"]) for smooth in expected])
    collector = _AssertCollector()
    for B in (1, 7, 19):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            parse_formula("y~0+" + expression),
            source,
            control=OperatorPreparationControl(batch_rows=B),
        )
        assert model.penalties.n_penalties == sum(len(s["S"]) for s in expected)
        actual_X = np.vstack(
            [model.evaluate_batch(batch).modest_values for batch in source.scan(B)]
        )
        for name, actual, reference in (
            ("original constrained tensor basis", actual_X, expected_X),
            ("level and margin penalties", model.penalties.materialize(), expected_S),
        ):
            collector.check(
                f"{expression}/B{B}/{name}",
                lambda actual=actual, reference=reference: np.testing.assert_allclose(
                    actual, reference, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
    collector.raise_if_any("Pinned bounded tensor setup")


@pytest.mark.parametrize(
    ("expression", "sp"),
    [
        ("te(x,z,bs='cr',k=4)", [0.8, 1.2, 0.4]),
        ("ti(x,z,bs='cr',k=4)", [0.8, 1.2, 0.4]),
        ("te(x,z,bs='cc',k=4)", [0.8, 1.2, 0.4]),
        ("ti(x,z,bs='cr',k=4,by=z)", [0.8, 1.2, 0.4]),
        ("te(x,z,bs='cr',k=4,by=fac)", [0.8, 1.2, 0.4, 0.6, 0.3, 0.9, 0.7]),
    ],
)
def test_public_tensor_fixed_sp_uses_actual_local_layout(r_bridge, expression, sp):
    frame = _frame()
    frame.y = 1.0 + 0.2 * np.sin(frame.x)
    frame["w"] = np.linspace(0.6, 1.4, len(frame))
    frame.loc[::7, "w"] = 0.0
    frame["o"] = 0.01 * frame.x
    formula = "y~s(g,bs='re')+" + expression
    reference = r_bridge.fit_fixed_sp(
        formula, frame, "gaussian", np.asarray(sp), weights="w", offset="o"
    )
    collector = _AssertCollector()
    for B in (1, 7, 19):
        result = GAM(
            formula,
            sp=sp,
            control=FitControl(
                execution="operator",
                linear_solver="pcg",
                batch_rows=B,
                operator=OperatorControl(cg_rtol=1e-13, cg_atol=1e-14),
            ),
        ).fit(
            DataFrameRowSource(frame, response="y", weights=frame.w, offset=frame.o),
            result="prediction",
        )
        assert result.converged
        for name, actual, expected in (
            ("coefficients", result.coefficients, reference["coefficients"]),
            ("deviance", result.deviance, reference["deviance"]),
            (
                "means",
                result.predict(frame, offset=frame.o),
                reference["fitted_values"],
            ),
        ):
            collector.check(
                f"{expression}/B{B}/{name}",
                lambda actual=actual, expected=expected: np.testing.assert_allclose(
                    actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
    collector.raise_if_any("Public bounded tensor fit")


@pytest.mark.parametrize(
    "expression",
    [
        "s(x,bs='cr',k=5,by=fac)",
        "te(x,z,bs='cr',k=4)",
        "ti(x,z,bs='cc',k=4,by=fac)",
    ],
)
def test_selected_repeated_by_tensor_rows_and_frozen_prediction_ownership(expression):
    frame = _frame(ordered=True)
    selection = np.array([52, 0, 39, 3, 17, 3, 40, 7, 22, 10, 29, 22, 45, 13, 33])
    spec = parse_formula("y~s(g,bs='re')+" + expression)
    expected = ModelSetup.build(spec, frame.iloc[selection])
    collector = _AssertCollector()
    for B in (1, 4, 7):
        source = ArrayRowSource(
            {name: frame[name] for name in ("x", "z", "fac", "g")},
            y=frame.y,
            row_selection=selection,
        )
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        batches = list(source.scan(B))
        np.testing.assert_array_equal(
            np.concatenate([batch.row_positions for batch in batches]), selection
        )
        actual = np.vstack([_matrix(model, batch) for batch in batches])
        collector.check(
            f"{expression}/B{B}/selected original basis",
            lambda actual=actual: np.testing.assert_allclose(
                actual, expected.X, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        geometry = OperatorPredictionGeometry(
            model.modest.predict_spec,
            model.modest_columns,
            model.random_effects,
            model.n_coef,
            B,
            model.modest_raw_width,
            model.metadata_bound_bytes,
        )
        coefficients = np.linspace(-0.3, 0.5, model.n_coef)
        heldout = frame.iloc[[1, 30, 51, 2, 31]].copy()
        expected_predict = geometry.forward(heldout, coefficients)
        restored = pickle.loads(pickle.dumps(geometry))
        assert restored.modest is not geometry.modest
        for term in restored.modest.coef_map.terms:
            if term.Z_centering is not None:
                assert not term.Z_centering.flags.writeable
            if term.smooth is not None:
                base = getattr(term.smooth, "base_smooth", term.smooth)
                assert base._X is None
        np.testing.assert_allclose(
            restored.forward(heldout, coefficients),
            expected_predict,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            restored.build_predict_matrix(heldout) @ coefficients,
            expected_predict,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        if "by=fac" in expression:
            unknown = heldout.iloc[:3].copy()
            unknown.fac = ["novel", "c", "a"]
            with pytest.warns(UserWarning, match="not in original fit"):
                prediction = restored.forward(unknown, coefficients)
            assert np.isnan(prediction[0])
            assert np.all(np.isfinite(prediction[1:]))
    collector.raise_if_any("Selected operator by/tensor replay")


@pytest.mark.parametrize(
    "expression",
    ["te(x,z,bs='cr',k=20)", "ti(x,z,bs='cc',k=20,by=fac)"],
)
def test_tensor_cap_rejects_before_any_margin_constructor(monkeypatch, expression):
    calls = []
    monkeypatch.setattr(
        operator_modest, "_cubic_from_knots", lambda *args: calls.append(args)
    )
    with pytest.raises(MemoryError, match="hard cap"):
        prepare_operator_model(
            parse_formula("y~s(g,bs='re')+" + expression),
            DataFrameRowSource(_frame(), response="y"),
            control=OperatorPreparationControl(batch_rows=7),
        )
    assert not calls


@pytest.mark.parametrize("mixed", [False, True])
def test_object_discovery_and_mixed_declared_by_levels_preserve_dense_values(mixed):
    frame = _frame()
    if mixed:
        # NumPy's sequence coercion would stringify the numeric categories.
        values = np.empty(len(frame), dtype=object)
        values[0::3] = "a"
        values[1::3] = 12
        values[2::3] = 3.5
        frame.fac = pd.Categorical(values, categories=["a", 12, 3.5])
    else:
        frame.fac = frame.fac.astype(object)
    spec = parse_formula("y~s(g,bs='re')+s(x,bs='cr',k=5,by=fac)")
    dense = ModelSetup.build(spec, frame)
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=7)
    )
    np.testing.assert_allclose(
        np.vstack([_matrix(model, batch) for batch in source.scan(7)]),
        dense.X,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_nonfinite_numeric_by_rejects_before_geometry_reduction():
    frame = _frame()
    frame.loc[20, "z"] = np.inf
    with pytest.raises(ValueError, match="Numeric by covariates must be finite"):
        prepare_operator_model(
            parse_formula("y~s(g,bs='re')+s(x,bs='cr',k=5,by=z)"),
            DataFrameRowSource(frame, response="y"),
            control=OperatorPreparationControl(batch_rows=7),
        )


def test_factor_tensor_penalty_overlap_budget_rejects_before_margin_constructor(
    monkeypatch,
):
    frame = _frame()
    spec = parse_formula("y~te(x,z,bs='cr',k=4,by=fac)")
    calls = []
    monkeypatch.setattr(
        operator_modest, "_cubic_from_knots", lambda *args: calls.append(args)
    )
    with pytest.raises(MemoryError, match="preparation"):
        prepare_operator_model(
            spec,
            DataFrameRowSource(frame, response="y"),
            control=OperatorPreparationControl(
                batch_rows=7, memory_budget_bytes=400_000
            ),
        )
    assert not calls


def test_actual_tensor_penalty_layout_rejects_wrong_public_sp_before_fitting(
    monkeypatch,
):
    import jaxgam.execution.operator_fit as operator_fit

    calls = []
    monkeypatch.setattr(
        operator_fit,
        "fit_operator_fixed_sp",
        lambda *args, **_kwargs: calls.append(args),
    )
    with pytest.raises(ValueError, match="3"):
        GAM(
            "y~s(g,bs='re')+te(x,z,bs='cr',k=4)",
            sp=[0.8, 1.2],
            control=FitControl(execution="operator", linear_solver="pcg", batch_rows=7),
        ).fit(DataFrameRowSource(_frame(), response="y"), result="prediction")
    assert not calls
