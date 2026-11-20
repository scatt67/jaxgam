"""Continuous compact indices adapted to frozen discrete operators."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.discretize import DiscretizationBudget, ExactIndexTable
from jaxgam.data.source import ArrayRowSource
from jaxgam.formula.continuous_design import build_continuous_discrete_design
from jaxgam.formula.continuous_setup import (
    ContinuousDiscreteSetup,
    build_continuous_discrete_setup,
)
from jaxgam.formula.discrete_design import (
    DiscreteDesignControl,
    DiscreteDesignUnavailable,
    LookupBlock,
    TensorLookupBlock,
)
from jaxgam.formula.parser import parse_formula
from tests.tolerances import STRICT


def _frame(n_rows: int = 72) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": np.linspace(0.2, 1.8, n_rows),
            "x": np.resize(np.array((0.0, 0.2, 0.5, 0.8, 1.0)), n_rows),
            "z": np.linspace(-1.0, 1.0, n_rows),
            "w": np.resize(np.array((-1.0, 0.0, 1.0)), n_rows),
            "b": np.resize(np.array((0.5, 1.0, 1.5)), n_rows),
            "g": pd.Series(
                pd.Categorical(
                    np.resize(np.array(("low", "mid", "high")), n_rows),
                    categories=("low", "mid", "high"),
                    ordered=True,
                )
            ),
        }
    )


class _CountingSource(ArrayRowSource):
    def __init__(self, frame, **kwargs):
        super().__init__({name: frame[name] for name in frame}, **kwargs)
        self.scans = 0

    def scan(self, batch_rows):
        self.scans += 1
        yield from super().scan(batch_rows)


def _setup(formula: str, source, **kwargs) -> ContinuousDiscreteSetup:
    result = build_continuous_discrete_setup(
        parse_formula(formula), source, batch_rows=7, **kwargs
    )
    assert isinstance(result, ContinuousDiscreteSetup)
    return result


def _snapped_frame(setup: ContinuousDiscreteSetup) -> pd.DataFrame:
    columns = {}
    for group in setup.compact.groups:
        selector = group.index.selector
        for name, values in zip(group.names, group.index.values, strict=True):
            columns[name] = values[selector]
    return pd.DataFrame(columns)


def test_descriptor_matches_explicit_snapped_design_without_another_scan() -> None:
    frame = _frame()
    formula = (
        "y ~ g + z + s(x, bs='cr', k=4) + "
        "te(x, w, bs='cr', k=3, by=g) + s(z, bs='cr', k=4, by=b)"
    )
    source = _CountingSource(frame)
    setup = _setup(formula, source, factor_names=("g",), m=(6, 5, 6))
    scans = source.scans
    design = build_continuous_discrete_design(setup, source)
    assert not isinstance(design, DiscreteDesignUnavailable)
    assert source.scans == scans
    expected = setup.predict_spec.build_predict_matrix(_snapped_frame(setup))
    actual = np.vstack(
        [
            design.evaluate_batch(np.arange(start, min(start + 11, len(frame))))
            for start in range(0, len(frame), 11)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    assert design.lineage.origin == "continuous"


def test_shared_compact_index_is_reused_across_terms_and_batch_gather(
    monkeypatch,
) -> None:
    frame = _frame()
    source = _CountingSource(frame)
    setup = _setup(
        "y ~ 0 + s(x, bs='cr', k=4) + te(x, w, bs='cr', k=3)",
        source,
        m=6,
    )
    design = build_continuous_discrete_design(setup, source)
    assert not isinstance(design, DiscreteDesignUnavailable)
    ordinary_number, ordinary = next(
        (number, block)
        for number, block in enumerate(design.blocks)
        if isinstance(block, LookupBlock) and block.index.names == ("x",)
    )
    tensor_number, tensor = next(
        (number, block)
        for number, block in enumerate(design.blocks)
        if isinstance(block, TensorLookupBlock)
    )
    assert ordinary.index is tensor.margin_indices[0]
    assert ordinary.index.selector is setup.compact.selector_for("x")

    calls = 0
    original = ExactIndexTable.batch_selector

    def counted(self, positions):
        nonlocal calls
        if self is ordinary.index:
            calls += 1
        return original(self, positions)

    monkeypatch.setattr(ExactIndexTable, "batch_selector", counted)
    gathered = design.batch_indices(np.arange(7))
    assert calls == 1
    assert gathered[ordinary_number][0] is gathered[tensor_number][0]


def test_parametric_terms_keep_separate_grids_and_frozen_column_order() -> None:
    frame = _frame()
    source = _CountingSource(frame)
    setup = _setup("y ~ g + z + w", source, factor_names=("g",), m=5)
    design = build_continuous_discrete_design(setup, source)
    assert not isinstance(design, DiscreteDesignUnavailable)
    assert all(isinstance(block, LookupBlock) for block in design.blocks)
    # Intercept, ordered-factor contrasts, z, and w use four independent
    # compact tables but retain the frozen coefficient ordering.
    assert [block.col_start for block in design.blocks] == [0, 1, 3, 4]
    assert [block.n_coef for block in design.blocks] == [1, 2, 1, 1]
    np.testing.assert_allclose(
        design.evaluate_batch(np.arange(len(frame))),
        setup.predict_spec.build_predict_matrix(_snapped_frame(setup)),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_numeric_parametric_reuses_its_named_joint_smooth_column() -> None:
    rng = np.random.default_rng(14981)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=75),
            "x": rng.uniform(-2.0, 2.0, size=75),
            "z": rng.uniform(1.0, 5.0, size=75),
        }
    )
    source = _CountingSource(frame)
    setup = _setup("y ~ z + s(x, z, bs='tp', k=6)", source, m=9)
    design = build_continuous_discrete_design(setup, source)
    assert not isinstance(design, DiscreteDesignUnavailable)
    assert setup.compact.groups[0].names == ("x", "z")

    parametric = next(
        block
        for block in design.blocks
        if isinstance(block, LookupBlock) and block.col_start == 1
    )
    smooth = next(
        block
        for block in design.blocks
        if isinstance(block, LookupBlock) and block.col_start > 1
    )
    assert parametric.index is smooth.index
    np.testing.assert_array_equal(
        parametric.table[:, 0],
        parametric.index.values[parametric.index.names.index("z")],
    )
    np.testing.assert_allclose(
        design.evaluate_batch(np.arange(len(frame))),
        setup.predict_spec.build_predict_matrix(_snapped_frame(setup)),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_factor_parametric_reuses_its_named_joint_random_effect_column() -> None:
    frame = pd.DataFrame(
        {
            "y": np.linspace(0.1, 2.0, 45),
            "x": np.resize(np.array((1.0, 2.0, 4.0, 7.0, 11.0)), 45),
            "g": pd.Series(
                pd.Categorical(
                    np.resize(np.array(("a", "b", "c")), 45),
                    categories=("a", "b", "c"),
                )
            ),
        }
    )
    source = _CountingSource(frame)
    setup = _setup("y ~ g + s(x, g, bs='re')", source, factor_names=("g",), m=9)
    design = build_continuous_discrete_design(setup, source)
    assert not isinstance(design, DiscreteDesignUnavailable)
    assert setup.compact.groups[0].names == ("x", "g")

    parametric = next(
        block
        for block in design.blocks
        if isinstance(block, LookupBlock) and block.col_start == 1
    )
    smooth = next(
        block
        for block in design.blocks
        if isinstance(block, LookupBlock) and block.col_start > 2
    )
    assert parametric.index is smooth.index
    np.testing.assert_allclose(
        design.evaluate_batch(np.arange(len(frame))),
        setup.predict_spec.build_predict_matrix(_snapped_frame(setup)),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize(
    ("formula", "factors"),
    [
        ("y ~ s(x, bs='tp', k=4)", ()),
        ("y ~ s(x, bs='gp', k=4)", ()),
        ("y ~ ti(x, w, bs='cr', k=3)", ()),
        ("y ~ s(g, bs='re')", ("g",)),
        ("y ~ s(x, bs='cr', k=4, by=g)", ("g",)),
        ("y ~ s(x, bs='cr', k=4, by=b)", ()),
    ],
)
def test_supported_compact_blocks_match_explicit_rows(formula, factors) -> None:
    frame = _frame()
    source = _CountingSource(frame)
    setup = _setup(formula, source, factor_names=factors, m=6)
    design = build_continuous_discrete_design(setup, source)
    assert not isinstance(design, DiscreteDesignUnavailable)
    np.testing.assert_allclose(
        design.evaluate_batch(np.arange(len(frame))),
        setup.predict_spec.build_predict_matrix(_snapped_frame(setup)),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_combined_retained_and_action_budgets_return_typed_fallbacks() -> None:
    frame = _frame()
    source = _CountingSource(frame)
    setup = _setup("y ~ s(x, bs='cr', k=4)", source, m=6)
    retained = build_continuous_discrete_design(
        setup,
        source,
        control=DiscreteDesignControl(
            budget=DiscretizationBudget(max_table_bytes=setup.retained_table_bytes - 1)
        ),
    )
    assert isinstance(retained, DiscreteDesignUnavailable)
    assert "retained compact setup" in retained.reason
    action = build_continuous_discrete_design(
        setup,
        source,
        control=DiscreteDesignControl(
            batch_rows=8,
            budget=DiscretizationBudget(max_batch_bytes=32),
        ),
    )
    assert isinstance(action, DiscreteDesignUnavailable)
    assert "operator batch" in action.reason


def test_descriptor_rejects_changed_or_incompatible_source() -> None:
    frame = _frame()
    source = _CountingSource(frame)
    setup = _setup("y ~ s(x, bs='cr', k=4)", source, m=6)
    changed = _CountingSource(frame.assign(x=frame.x + 0.01))
    with pytest.raises(RuntimeError, match="changed before"):
        build_continuous_discrete_design(setup, changed)
    short = _CountingSource(frame.iloc[:-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="different row counts"):
        build_continuous_discrete_design(setup, short)


def test_selected_reordered_repeated_rows_replay_across_batches() -> None:
    frame = _frame()
    selection = np.array((7, 2, 40, 1, 7, 55, 3))
    source = _CountingSource(frame, row_selection=selection)
    setup = _setup(
        "y ~ g + s(x, bs='cr', k=4, by=b)",
        source,
        factor_names=("g",),
        m=5,
    )
    design = build_continuous_discrete_design(
        setup, source, control=DiscreteDesignControl(batch_rows=3)
    )
    assert not isinstance(design, DiscreteDesignUnavailable)
    actual = np.vstack(
        [
            design.evaluate_batch(np.arange(start, min(start + 3, len(selection))))
            for start in range(0, len(selection), 3)
        ]
    )
    np.testing.assert_allclose(
        actual,
        setup.predict_spec.build_predict_matrix(_snapped_frame(setup)),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_continuous_descriptor_import_does_not_import_jax() -> None:
    root = Path(__file__).resolve().parents[2] / "jaxgam"
    script = f"""
import sys
import types
from pathlib import Path
root = Path({str(root)!r})
package = types.ModuleType('jaxgam')
package.__path__ = [str(root)]
sys.modules['jaxgam'] = package
import jaxgam.formula.continuous_design
assert not any(name == 'jax' or name.startswith('jax.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_one_margin_by_blocks_run_all_device_actions() -> None:
    from jaxgam.fitting.discrete_ops import (
        discrete_xb_transpose,
        discrete_xbd,
        discrete_xwxd,
        to_device_discrete_design,
    )

    frame = _frame(24)
    source = _CountingSource(frame)
    setup = _setup(
        "y ~ s(x, bs='cr', k=4, by=g)",
        source,
        factor_names=("g",),
        m=6,
    )
    design = build_continuous_discrete_design(
        setup, source, control=DiscreteDesignControl(batch_rows=len(frame))
    )
    assert not isinstance(design, DiscreteDesignUnavailable)
    explicit = design.evaluate_batch(np.arange(len(frame)))
    device = to_device_discrete_design(design)
    selectors = design.batch_indices(np.arange(len(frame)))
    beta = np.linspace(-0.3, 0.7, design.n_coef)
    values = np.linspace(0.1, 1.1, len(frame))
    weights = np.linspace(0.5, 1.5, len(frame))
    np.testing.assert_allclose(
        discrete_xbd(device, selectors, beta), explicit @ beta, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        discrete_xb_transpose(device, selectors, values),
        explicit.T @ values,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        discrete_xwxd(device, selectors, weights),
        explicit.T @ (weights[:, None] * explicit),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fixed_sp_gaussian_solution_matches_identical_explicit_design() -> None:
    from jaxgam.fitting.discrete_ops import (
        discrete_xwxd,
        discrete_xwyd,
        to_device_discrete_design,
    )

    frame = _frame(72)
    weights = np.linspace(0.7, 1.3, len(frame))
    offset = np.linspace(-0.1, 0.2, len(frame))
    source = ArrayRowSource(
        {name: frame[name] for name in frame if name != "y"},
        y=frame.y,
        weights=weights,
        offset=offset,
    )
    formula = "y ~ z + s(x, bs='cr', k=4)"
    setup = _setup(formula, source, m=6)
    design = build_continuous_discrete_design(
        setup, source, control=DiscreteDesignControl(batch_rows=13)
    )
    assert not isinstance(design, DiscreteDesignUnavailable)
    device = to_device_discrete_design(design)
    gram = np.zeros((design.n_coef, design.n_coef))
    rhs = np.zeros(design.n_coef)
    cursor = 0
    for batch in source.scan(13):
        rows = np.arange(cursor, cursor + len(batch.row_positions))
        selectors = design.batch_indices(rows)
        gram += np.asarray(discrete_xwxd(device, selectors, batch.weight))
        assert batch.y is not None
        rhs += np.asarray(
            discrete_xwyd(device, selectors, batch.weight, batch.y - batch.offset)
        )
        cursor += len(rows)
    assert cursor == len(frame)
    assert setup.penalties is not None
    log_lambda = np.log(np.linspace(0.4, 1.1, setup.penalties.n_penalties))
    penalty = setup.penalties.materialize(log_lambda)
    beta_discrete = np.linalg.solve(gram + penalty, rhs)

    explicit = setup.predict_spec.build_predict_matrix(_snapped_frame(setup))
    beta_explicit = np.linalg.solve(
        explicit.T @ (weights[:, None] * explicit) + penalty,
        explicit.T @ (weights * (frame.y.to_numpy() - offset)),
    )
    np.testing.assert_allclose(
        beta_discrete, beta_explicit, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        explicit @ beta_discrete,
        explicit @ beta_explicit,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
