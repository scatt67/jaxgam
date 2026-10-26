"""Source-faithful bounded TP/TS/GP operator preparation."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from jaxgam.data.source import ArrayRowSource, DataFrameRowSource
from jaxgam.formula import operator_radial
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    _CheckedSource,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.inference.operator_geometry import OperatorPredictionGeometry
from jaxgam.smooths import source_lapack
from tests.helpers import _AssertCollector
from tests.tolerances import (
    MODERATE,
    STRICT,
    normalize_column_signs,
    normalize_symmetric_signs,
)


def _frame(n: int = 61) -> pd.DataFrame:
    x = np.linspace(-1.3, 1.7, n)
    z = np.sin(np.linspace(-0.8, 2.2, n)) + np.linspace(0, 0.1, n)
    return pd.DataFrame({"x": x, "z": z, "y": np.cos(1.7 * x) + 0.2 * z})


def _operator_matrix(formula: str, frame: pd.DataFrame, batch_rows: int):
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula(formula),
        source,
        control=OperatorPreparationControl(batch_rows=batch_rows),
    )
    rows = np.vstack(
        [model.evaluate_batch(batch).modest_values for batch in source.scan(batch_rows)]
    )
    return model, rows


def _align_column_signs(
    actual: np.ndarray, expected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align eigenvector signs without choosing among tied largest entries."""
    products = np.sum(actual * expected, axis=0)
    signs = np.where(products < 0.0, -1.0, 1.0)
    return actual * signs[np.newaxis, :], signs


def _assert_geometry_prediction(
    geometry,
    frame: pd.DataFrame,
    coefficients: np.ndarray,
    expected: np.ndarray,
    *,
    explicit: bool,
) -> None:
    actual = (
        geometry.build_predict_matrix(frame) @ coefficients
        if explicit
        else geometry.forward(frame, coefficients)
    )
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)


@pytest.mark.parametrize(
    "expression",
    [
        "s(x,bs='tp',k=6)",
        "s(x,bs='ts',k=6)",
        "s(x,bs='gp',k=6)",
        "te(x,z,bs='tp',k=4)",
        "ti(x,z,bs='tp',k=4)",
        "te(x,z,bs='gp',k=4)",
    ],
)
def test_radial_operator_matches_existing_dense_geometry(expression: str) -> None:
    frame = _frame()
    formula = "y~0+" + expression
    dense = ModelSetup.build(parse_formula(formula), frame)
    collector = _AssertCollector()
    reference_model = None
    reference_rows = None
    for batch_rows in (1, 7, 19):
        model, rows = _operator_matrix(formula, frame, batch_rows)
        if reference_model is None:
            reference_model, reference_rows = model, rows
        collector.check(
            f"B{batch_rows}/basis",
            lambda rows=rows, reference_rows=reference_rows: np.testing.assert_allclose(
                rows,
                reference_rows,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        collector.check(
            f"B{batch_rows}/penalty",
            lambda model=model, reference_model=reference_model: (
                np.testing.assert_allclose(
                    model.penalties.materialize(),
                    reference_model.penalties.materialize(),
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                )
            ),
        )
        assert model.preparation_peak_bound_bytes <= model.memory_budget_bytes
        for term in model.modest.predict_spec.coef_map.terms:
            if term.term_type == "smooth":
                assert term.smooth._X is None
    collector.raise_if_any(expression)
    # Alternative LAPACK eigenvector coordinates are allowed, but the fitted
    # training space must equal the existing dense constructor's space.
    if "te(" not in expression and "ti(" not in expression:
        q_operator = np.linalg.qr(reference_rows, mode="reduced")[0]
        q_dense = np.linalg.qr(dense.X, mode="reduced")[0]
        np.testing.assert_allclose(
            q_operator @ q_operator.T,
            q_dense @ q_dense.T,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


@pytest.mark.parametrize(
    ("python_expression", "r_expression"),
    [
        ("s(x,bs='tp',k=6)", "s(x,bs='tp',k=6)"),
        ("s(x,bs='ts',k=6)", "s(x,bs='ts',k=6)"),
        ("s(x,bs='gp',k=6)", "s(x,bs='gp',k=6)"),
        ("te(x,z,bs='tp',k=4)", "te(x,z,bs='tp',k=c(4,4))"),
        ("te(x,z,bs='tp',k=5)", "te(x,z,bs='tp',k=c(5,5))"),
        ("te(x,z,bs='tp',k=6)", "te(x,z,bs='tp',k=c(6,6))"),
        ("ti(x,z,bs='tp',k=4)", "ti(x,z,bs='tp',k=c(4,4))"),
        ("ti(x,z,bs='tp',k=5)", "ti(x,z,bs='tp',k=c(5,5))"),
        ("ti(x,z,bs='tp',k=6)", "ti(x,z,bs='tp',k=c(6,6))"),
        ("te(x,z,bs='gp',k=4)", "te(x,z,bs='gp',k=c(4,4))"),
        (
            "s(x,z,bs='gp',k=6,xt={'max_knots':11,'seed':1})",
            "s(x,z,bs='gp',k=6,xt=list(max.knots=11,seed=1))",
        ),
    ],
)
def test_radial_operator_matches_pinned_smoothcon(
    r_bridge, python_expression: str, r_expression: str
) -> None:
    frame = _frame()
    model, rows = _operator_matrix("y~0+" + python_expression, frame, 7)
    expected = r_bridge.smooth_construct(r_expression, frame, absorb_cons=True)
    collector = _AssertCollector()
    collector.check(
        "basis",
        lambda: np.testing.assert_allclose(
            rows, expected["X"], rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    actual_penalty = model.penalties.materialize()
    expected_penalty = sum(expected["S"])
    collector.check(
        "summed penalties",
        lambda: np.testing.assert_allclose(
            actual_penalty,
            expected_penalty,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any(python_expression)


def test_random_ti_k6_retains_exact_fixture_and_narrow_moderate_basis(r_bridge) -> None:
    """The sole conditioned cross-provider residue keeps its exact fixture."""
    rng = np.random.default_rng(92612)
    x = rng.normal(size=73)
    z = 0.3 * x + rng.normal(size=73)
    y = np.sin(x) - 0.1 * z
    expression = "ti(x,z,bs='tp',k=6)"
    formula = "y~0+" + expression
    raw = np.column_stack((x, z)).astype("<f8", copy=False)
    control = json.dumps(
        {"batch_rows": 7, "formula": formula, "n": 73},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(raw.tobytes(order="C") + control).hexdigest() == (
        "b7eff021138f903308f759f17d02576598f3675fe746f94e3c1c214a5e14b92b"
    )
    frame = pd.DataFrame({"x": x, "z": z, "y": y})
    model, rows = _operator_matrix(formula, frame, 7)
    provider = source_lapack._source_lapack().module
    assert provider.__version__ == "0.3.26.0.4"
    assert provider.get_openblas_config().startswith("OpenBLAS 0.3.26 ")
    expected = r_bridge.smooth_construct(
        "ti(x,z,bs='tp',k=c(6,6))", frame, absorb_cons=True
    )
    np.testing.assert_allclose(
        rows, expected["X"], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        model.penalties.materialize(),
        sum(expected["S"]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_r_sampled_tp_knots_and_transform_match_pinned_source(r_bridge) -> None:
    frame = _frame(83)
    spec = parse_formula("y~0+s(x,z,bs='tp',k=6,max_knots=11,seed=1)")
    model = prepare_operator_model(
        spec,
        DataFrameRowSource(frame, response="y"),
        control=OperatorPreparationControl(batch_rows=9),
    )
    smooth = model.modest.predict_spec.coef_map.terms[0].smooth
    expected = r_bridge.smooth_construct(
        "s(x,z,bs='tp',k=6,xt=list(max.knots=11,seed=1))",
        frame,
        absorb_cons=False,
    )
    np.testing.assert_allclose(
        smooth._Xu, expected["Xu"], rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_radial_budget_rejects_before_kernel_eigendecomposition(monkeypatch) -> None:
    calls: list[object] = []

    def record_call(*args, **_kwargs):
        calls.append(args)

    monkeypatch.setattr(operator_radial, "source_lanczos", record_call)
    with pytest.raises(MemoryError):
        prepare_operator_model(
            parse_formula("y~s(x,z,bs='tp',k=6,max_knots=2000)"),
            DataFrameRowSource(_frame(300), response="y"),
            control=OperatorPreparationControl(
                batch_rows=31, memory_budget_bytes=180_000
            ),
        )
    assert calls == []


def test_gp_retains_signed_source_eigenvalues(monkeypatch) -> None:
    frame = _frame(23)
    spec = parse_formula("y~0+s(x,bs='gp',k=5)").smooth_terms[0]
    source = _CheckedSource(
        DataFrameRowSource(frame, response="y"),
        OperatorPreparationControl(batch_rows=5),
        0,
    )

    def signed(matrix, k, **_kwargs):
        values, vectors = np.linalg.eigh(matrix)
        chosen = np.argsort(np.abs(values))[-k:][::-1]
        retained = values[chosen]
        retained[0] = -abs(retained[0])
        return retained, vectors[:, chosen]

    monkeypatch.setattr(operator_radial, "source_lanczos", signed)
    smooth, _ = operator_radial.prepare_radial_smooth(spec, source, source.control)
    assert smooth._S[0, 0] < 0


@pytest.mark.parametrize("basis", ["tp", "gp"])
def test_numeric_by_radial_basis_and_penalty_match_pinned_source(
    r_bridge, basis: str
) -> None:
    frame = _frame()
    frame["by"] = np.linspace(0.4, 1.4, len(frame))
    expression = f"s(x,bs='{basis}',k=6,by=by)"
    model, rows = _operator_matrix("y~0+" + expression, frame, 7)
    expected = r_bridge.smooth_construct(expression, frame, absorb_cons=True)
    np.testing.assert_allclose(
        normalize_column_signs(rows),
        normalize_column_signs(expected["X"]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        normalize_symmetric_signs(model.penalties.materialize(), rows),
        normalize_symmetric_signs(sum(expected["S"]), expected["X"]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize("basis", ["tp", "gp"])
def test_factor_by_radial_blocks_match_pinned_source(r_bridge, basis: str) -> None:
    frame = _frame()
    frame["group"] = pd.Categorical(
        np.resize(["a", "b", "c"], len(frame)), categories=["a", "b", "c"]
    )
    expression = f"s(x,bs='{basis}',k=6,by=group)"
    model, rows = _operator_matrix("y~0+" + expression, frame, 7)
    expected = r_bridge.smooth_construct_list(expression, frame, absorb_cons=True)
    expected_rows = np.column_stack([item["X"] for item in expected])
    expected_penalty = linalg.block_diag(*(item["S"][0] for item in expected))
    aligned_rows, signs = _align_column_signs(rows, expected_rows)
    np.testing.assert_allclose(
        aligned_rows,
        expected_rows,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        signs[:, np.newaxis] * model.penalties.materialize() * signs[np.newaxis, :],
        expected_penalty,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_radial_setup_rejects_a_changed_first_row_before_later_scans() -> None:
    class PrefixChanged(DataFrameRowSource):
        def scan(self, batch_rows):
            for batch in super().scan(batch_rows):
                if batch_rows == 1:
                    batch = replace(batch, y=batch.y + 1.0)
                yield batch

        def fingerprint(self):
            return "dishonestly-stable-radial-prefix"

    with pytest.raises(RuntimeError, match="changed its operator replay"):
        prepare_operator_model(
            parse_formula("y~s(x,bs='tp',k=6)"),
            PrefixChanged(_frame(), response="y"),
            control=OperatorPreparationControl(batch_rows=7),
        )


@pytest.mark.parametrize("basis", ["tp", "gp"])
def test_selected_repeated_radial_replay_and_prediction_ownership(basis: str) -> None:
    frame = _frame()
    selection = np.array([52, 0, 39, 3, 17, 3, 40, 7, 22, 10, 22, 45, 13, 33])
    spec = parse_formula(f"y~s(x,bs='{basis}',k=6)")
    dense = ModelSetup.build(spec, frame.iloc[selection])
    collector = _AssertCollector()
    for batch_rows in (1, 4, 7):
        source = ArrayRowSource({"x": frame.x}, y=frame.y, row_selection=selection)
        model = prepare_operator_model(
            spec,
            source,
            control=OperatorPreparationControl(batch_rows=batch_rows),
        )
        batches = list(source.scan(batch_rows))
        collector.check(
            f"{basis}/B{batch_rows}/lineage",
            lambda batches=batches: np.testing.assert_array_equal(
                np.concatenate([batch.row_positions for batch in batches]), selection
            ),
        )
        rows = np.vstack(
            [model.evaluate_batch(batch).modest_values for batch in batches]
        )
        collector.check(
            f"{basis}/B{batch_rows}/selected basis",
            lambda rows=rows: np.testing.assert_allclose(
                rows, dense.X, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        geometry = OperatorPredictionGeometry(
            model.modest.predict_spec,
            model.modest_columns,
            model.random_effects,
            model.n_coef,
            batch_rows,
            model.modest_raw_width,
            model.metadata_bound_bytes,
        )
        coefficients = np.linspace(-0.3, 0.5, model.n_coef)
        heldout = frame.iloc[[1, 30, 51, 2, 31]].copy()
        expected = geometry.forward(heldout, coefficients)
        restored = pickle.loads(pickle.dumps(geometry))
        smooth = restored.modest.coef_map.terms[-1].smooth
        assert smooth is not None
        assert smooth._X is None
        knots = getattr(smooth, "_Xu", None)
        if knots is None:
            knots = smooth._knt
        assert knots is not None
        assert not knots.flags.writeable
        collector.check(
            f"{basis}/B{batch_rows}/pickle forward",
            partial(
                _assert_geometry_prediction,
                restored,
                heldout,
                coefficients,
                expected,
                explicit=False,
            ),
        )
        collector.check(
            f"{basis}/B{batch_rows}/explicit matrix",
            partial(
                _assert_geometry_prediction,
                restored,
                heldout,
                coefficients,
                expected,
                explicit=True,
            ),
        )
    collector.raise_if_any("Selected radial replay")
