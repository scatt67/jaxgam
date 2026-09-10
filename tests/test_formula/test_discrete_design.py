"""Frozen Phase-1 discrete design construction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from jaxgam.data.discretize import (
    DiscretizationBudget,
    ExactIndexTable,
    ExactIndexUnavailable,
    exact_index_columns,
)
from jaxgam.data.source import DataFrameRowSource
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.discrete_design import (
    DiscreteDesignControl,
    DiscreteDesignLineage,
    DiscreteDesignUnavailable,
    LookupBlock,
    TensorLookupBlock,
    _basis_token,
    build_discrete_design_from_predict_spec,
    build_discrete_design_from_prepared,
    build_discrete_design_from_setup,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.predict_matrix import build_predict_spec
from jaxgam.formula.prepare import prepare_model
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


def _data() -> pd.DataFrame:
    n = 48
    x = np.resize(np.array((0.1, 0.35, 0.7, 0.9)), n)
    z = np.resize(np.array((0.15, 0.45, 0.75, 0.95)), n)
    by = np.resize(np.array((0.5, 1.0, 1.5)), n)
    group = pd.Categorical(np.resize(np.array(("a", "b", "c")), n))
    return pd.DataFrame({"x": x, "z": z, "by": by, "group": group, "y": np.sin(x) + z})


def test_dense_setup_descriptor_replays_frozen_factor_by_and_tensor_design() -> None:
    data = _data()
    formula = (
        "y ~ group + s(x, bs='cr', k=4, by=by) "
        "+ s(z, bs='cr', k=4, by=group) + te(x, z, bs='cr', k=4)"
    )
    setup = ModelSetup.build(parse_formula(formula), data)
    source = DataFrameRowSource(data, response="y")
    descriptor = build_discrete_design_from_setup(setup, source)
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    collector = _AssertCollector()
    collector.check(
        "frozen_design_equivalence",
        lambda: np.testing.assert_allclose(
            descriptor.evaluate_batch(np.arange(len(data))),
            setup.X,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "tensor_margins_are_separate",
        lambda: (
            (_ for _ in ()).throw(AssertionError("missing tensor lookup"))
            if not any(
                isinstance(block, TensorLookupBlock) for block in descriptor.blocks
            )
            else None
        ),
    )
    collector.check(
        "no_training_design_field",
        lambda: (
            (_ for _ in ()).throw(AssertionError("descriptor retained training X"))
            if any(name in descriptor.__dict__ for name in ("X", "design_matrix"))
            else None
        ),
    )
    collector.raise_if_any("frozen discrete descriptor")


def test_prepared_adapter_stays_with_existing_additive_cubic_contract() -> None:
    data = _data()
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(parse_formula("y ~ x + s(x, bs='cr', k=4)"), source)
    descriptor = build_discrete_design_from_prepared(prepared, source)
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    expected = prepared.evaluate_batch(next(source.scan(len(data))))
    np.testing.assert_allclose(
        descriptor.evaluate_batch(np.arange(len(data))),
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    descriptor.validate_source(source)
    changed = DataFrameRowSource(data.assign(x=data.x + 0.01), response="y")
    with np.testing.assert_raises(RuntimeError):
        descriptor.validate_source(changed)


def test_selected_source_replays_reordered_repeated_rows_across_batches() -> None:
    data = _data()
    selection = np.array((7, 2, 40, 1, 7))
    selected = data.iloc[selection].reset_index(drop=True)
    setup = ModelSetup.build(parse_formula("y ~ s(x, bs='cr', k=4)"), selected)
    source = DataFrameRowSource(data, response="y", row_selection=selection)
    descriptor = build_discrete_design_from_setup(
        setup, source, control=DiscreteDesignControl(batch_rows=2)
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    actual = np.vstack(
        [
            descriptor.evaluate_batch(np.arange(start, min(start + 2, len(selection))))
            for start in range(0, len(selection), 2)
        ]
    )
    np.testing.assert_allclose(actual, setup.X, rtol=STRICT.rtol, atol=STRICT.atol)
    descriptor.validate_source(source)
    changed = DataFrameRowSource(data, response="y", row_selection=selection[::-1])
    with np.testing.assert_raises(RuntimeError):
        descriptor.validate_source(changed)


def test_descriptor_returns_typed_budget_fallback_before_dense_table() -> None:
    data = _data()
    setup = ModelSetup.build(parse_formula("y ~ s(x, bs='cr', k=4)"), data)
    result = build_discrete_design_from_setup(
        setup,
        DataFrameRowSource(data, response="y"),
        control=DiscreteDesignControl(budget=DiscretizationBudget(max_table_bytes=8)),
    )
    assert isinstance(result, DiscreteDesignUnavailable)
    assert result.lineage.origin == "dense_setup"


def test_failed_later_block_cleans_only_owned_selector_memmaps(tmp_path) -> None:
    data = _data().assign(z=np.linspace(0.1, 0.9, len(_data())))
    setup = ModelSetup.build(
        parse_formula("y ~ s(x, bs='cr', k=4) + s(z, bs='cr', k=4)"), data
    )
    caller_file = tmp_path / "caller-kept.txt"
    caller_file.write_text("do not remove")
    result = build_discrete_design_from_setup(
        setup,
        DataFrameRowSource(data, response="y"),
        control=DiscreteDesignControl(
            budget=DiscretizationBudget(max_table_bytes=10_000),
            selector_memmap_directory=tmp_path,
        ),
    )
    assert isinstance(result, DiscreteDesignUnavailable)
    assert caller_file.read_text() == "do not remove"
    assert not list(tmp_path.glob("jaxgam-discrete-*.npy"))


def test_repeated_covariates_share_exact_index_storage_and_batch_gather(
    tmp_path, monkeypatch
) -> None:
    data = _data()

    class CountingSource(DataFrameRowSource):
        scans = 0

        def scan(self, batch_rows):
            self.scans += 1
            yield from super().scan(batch_rows)

    setup = ModelSetup.build(
        parse_formula("y ~ 0 + s(x, bs='cr', k=4) + te(x, z, bs='cr', k=4)"),
        data,
    )
    source = CountingSource(data, response="y")
    success_dir = tmp_path / "success"
    success_dir.mkdir()
    descriptor = build_discrete_design_from_setup(
        setup,
        source,
        control=DiscreteDesignControl(selector_memmap_directory=success_dir),
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    # x is requested by both the ordinary smooth and the tensor margin, while
    # z is requested once.  Both requests scan and allocate exactly once.
    assert source.scans == 2
    assert len(list(success_dir.glob("jaxgam-discrete-*.npy"))) == 2
    ordinary_number, ordinary = next(
        (number, block)
        for number, block in enumerate(descriptor.blocks)
        if isinstance(block, LookupBlock) and block.index.names == ("x",)
    )
    tensor_number, tensor = next(
        (number, block)
        for number, block in enumerate(descriptor.blocks)
        if isinstance(block, TensorLookupBlock)
    )
    assert ordinary.index is tensor.margin_indices[0]
    assert ordinary.index.values[0] is tensor.margin_indices[0].values[0]
    assert ordinary.index.selector is tensor.margin_indices[0].selector
    # Frozen term/margin evaluators still own distinct basis lookup tables.
    assert ordinary.table is not tensor.margin_tables[0]

    calls = 0
    original_batch_selector = ExactIndexTable.batch_selector

    def traced_batch_selector(self, positions):
        nonlocal calls
        if self is ordinary.index:
            calls += 1
        return original_batch_selector(self, positions)

    monkeypatch.setattr(ExactIndexTable, "batch_selector", traced_batch_selector)
    selectors = descriptor.batch_indices(np.arange(7))
    assert calls == 1
    assert selectors[ordinary_number][0] is selectors[tensor_number][0]
    plan = descriptor.memory_plan
    assert plan is not None
    assert plan.selector_retained_bytes == 2 * len(data) * np.dtype(np.int32).itemsize
    expected_host_batch = (
        descriptor.control.batch_rows * np.dtype(np.intp).itemsize
        + descriptor.control.batch_rows * 2 * np.dtype(np.int32).itemsize
        + (2 + 3) * 4 * np.dtype(np.intp).itemsize
    )
    assert plan.host_batch_bytes == expected_host_batch

    failure_dir = tmp_path / "failure"
    failure_dir.mkdir()
    failed = build_discrete_design_from_setup(
        setup,
        CountingSource(data, response="y"),
        control=DiscreteDesignControl(
            batch_rows=1,
            selector_memmap_directory=failure_dir,
            budget=DiscretizationBudget(max_batch_bytes=100),
        ),
    )
    assert isinstance(failed, DiscreteDesignUnavailable)
    assert not list(failure_dir.glob("jaxgam-discrete-*.npy"))


def test_mixed_factor_scalar_types_replay_parametric_and_factor_by_designs() -> None:
    n = 24
    data = pd.DataFrame(
        {
            "group": pd.Categorical(
                np.resize(np.array(("a", 12, 3.5), dtype=object), n)
            ),
            "x": np.linspace(0.0, 1.0, n),
            "y": np.linspace(1.0, 2.0, n),
        }
    )
    formulas = (
        "y ~ group + s(x, bs='cr', k=4)",
        "y ~ group + s(x, by=group, bs='cr', k=4)",
    )
    collector = _AssertCollector()
    for formula in formulas:
        setup = ModelSetup.build(parse_formula(formula), data)
        descriptor = build_discrete_design_from_setup(
            setup, DataFrameRowSource(data, response="y")
        )
        collector.check(
            formula,
            lambda descriptor=descriptor, setup=setup: (
                (_ for _ in ()).throw(
                    AssertionError(f"unexpected fallback: {descriptor}")
                )
                if isinstance(descriptor, DiscreteDesignUnavailable)
                else np.testing.assert_allclose(
                    descriptor.evaluate_batch(np.arange(n)),
                    setup.X,
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                )
            ),
        )
    collector.raise_if_any("mixed factor exact replay")


def test_tensor_interaction_and_batch_workspace_keep_the_frozen_contract() -> None:
    data = _data()
    source = DataFrameRowSource(data, response="y")
    setup = ModelSetup.build(parse_formula("y ~ ti(x, z, bs='cr', k=4)"), data)
    descriptor = build_discrete_design_from_setup(setup, source)
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    np.testing.assert_allclose(
        descriptor.evaluate_batch(np.arange(len(data))),
        setup.X,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    too_wide = build_discrete_design_from_setup(
        setup,
        source,
        control=DiscreteDesignControl(
            batch_rows=64,
            budget=DiscretizationBudget(max_batch_bytes=8),
        ),
    )
    assert isinstance(too_wide, DiscreteDesignUnavailable)
    assert "batch" in too_wide.reason


def test_predict_spec_entry_requires_declared_source_lineage() -> None:
    data = _data()
    source = DataFrameRowSource(data, response="y")
    setup = ModelSetup.build(parse_formula("y ~ s(x, bs='cr', k=4)"), data)
    with np.testing.assert_raises(RuntimeError):
        build_discrete_design_from_predict_spec(
            build_predict_spec(setup),
            source,
            lineage=DiscreteDesignLineage("not-the-source", "frozen", "predict_spec"),
        )


def test_predict_spec_default_lineage_hashes_frozen_arrays_and_dense_rows() -> None:
    data = _data()
    source = DataFrameRowSource(data, response="y")
    setup = ModelSetup.build(parse_formula("y ~ s(x, bs='cr', k=4)"), data)
    descriptor = build_discrete_design_from_predict_spec(
        build_predict_spec(setup), source
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    assert descriptor.lineage.origin == "predict_spec"
    assert len(descriptor.lineage.basis_fingerprint) == 64
    short_source = DataFrameRowSource(
        data.iloc[:-1].reset_index(drop=True), response="y"
    )
    with np.testing.assert_raises(ValueError):
        build_discrete_design_from_setup(setup, short_source)


def test_frozen_fingerprint_owns_object_categories_and_prediction_transforms() -> None:
    data = _data()
    formula = "y ~ group + s(x, bs='cr', k=4)"
    first = build_predict_spec(ModelSetup.build(parse_formula(formula), data))
    second = build_predict_spec(ModelSetup.build(parse_formula(formula), data.copy()))
    assert _basis_token(first) == _basis_token(second)
    reordered = data.copy()
    reordered["group"] = pd.Categorical(reordered["group"], categories=("c", "b", "a"))
    category_spec = build_predict_spec(
        ModelSetup.build(parse_formula(formula), reordered)
    )
    assert _basis_token(first) != _basis_token(category_spec)
    changed = data.assign(x=data.x + np.linspace(0.0, 0.02, len(data)))
    knot_spec = build_predict_spec(ModelSetup.build(parse_formula(formula), changed))
    assert _basis_token(first) != _basis_token(knot_spec)


def test_intercept_lookup_and_descriptor_validation_fail_closed() -> None:
    data = _data()
    source = DataFrameRowSource(data, response="y")
    setup = ModelSetup.build(parse_formula("y ~ 1"), data)
    descriptor = build_discrete_design_from_setup(setup, source)
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    np.testing.assert_allclose(
        descriptor.evaluate_batch(np.arange(len(data))),
        setup.X,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    block = descriptor.blocks[0]
    assert isinstance(block, LookupBlock)
    with np.testing.assert_raises(ValueError):
        LookupBlock(-1, block.table, block.index)
    with np.testing.assert_raises(ValueError):
        DiscreteDesignControl(batch_rows=0)
    with np.testing.assert_raises(ValueError):
        DiscreteDesignControl(pair_policy="unbounded")  # type: ignore[arg-type]


def test_factor_tensor_replays_deleted_column_and_ordered_reference() -> None:
    data = _data()
    data["group"] = pd.Categorical(data["group"], ordered=True)
    setup = ModelSetup.build(
        parse_formula("y ~ te(x, z, bs='cr', k=3, by=group)"), data
    )
    spec = build_predict_spec(setup)
    smooth_number = next(
        number
        for number, term in enumerate(spec.coef_map.terms)
        if term.term_type == "smooth"
    )
    smooth_term = spec.coef_map.terms[smooth_number]
    assert smooth_term.Z_centering is not None
    # Ordered factors omit the declared reference level from active smooths.
    assert smooth_term.Z_centering.shape[0] == 2 * 3 * 3
    deleted = 3
    replacement = replace(
        smooth_term,
        n_coefs=smooth_term.n_coefs - 1,
        del_index=(deleted,),
    )
    terms = list(spec.coef_map.terms)
    terms[smooth_number] = replacement
    coef_map = replace(
        spec.coef_map,
        terms=tuple(terms),
        total_coefs=spec.coef_map.total_coefs - 1,
    )
    frozen = replace(spec, coef_map=coef_map, total_coefs=spec.total_coefs - 1)
    # Replay prediction rows where the ordered reference level is absent.  The
    # frozen level order and expanded centering transform must still apply.
    subset = data.loc[data.group != "a"].reset_index(drop=True)
    descriptor = build_discrete_design_from_predict_spec(
        frozen, DataFrameRowSource(subset, response="y")
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    tensor = next(
        block for block in descriptor.blocks if isinstance(block, TensorLookupBlock)
    )
    assert tensor.n_levels == 2
    assert deleted not in tensor.keep
    expected = np.delete(
        setup.X[data.group != "a"], smooth_term.col_start + deleted, axis=1
    )
    np.testing.assert_allclose(
        descriptor.evaluate_batch(np.arange(len(subset))),
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_memory_plan_counts_retained_metadata_combined_peaks_and_action_output() -> (
    None
):
    data = _data()
    setup = ModelSetup.build(
        parse_formula("y ~ te(x, z, bs='cr', k=4, by=group)"), data
    )
    source = DataFrameRowSource(data, response="y")
    descriptor = build_discrete_design_from_setup(setup, source)
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    plan = descriptor.memory_plan
    assert plan is not None
    tensor = next(
        block for block in descriptor.blocks if isinstance(block, TensorLookupBlock)
    )
    metadata_bytes = tensor.keep.nbytes
    assert tensor.centering is not None
    metadata_bytes += tensor.centering.nbytes
    assert tensor.level_codes is not None
    metadata_bytes += tensor.level_codes.nbytes
    assert plan.host_retained_bytes >= metadata_bytes
    assert plan.construction_peak_bytes > plan.host_retained_bytes
    assert plan.selector_retained_bytes >= plan.selector_resident_bytes
    assert plan.device_action_bytes >= setup.X.shape[1] ** 2 * 8
    assert plan.host_batch_bytes >= descriptor.control.batch_rows * 8
    assert plan.host_peak_bytes == (
        plan.host_retained_bytes + plan.selector_resident_bytes + plan.host_batch_bytes
    )
    assert plan.device_peak_bytes == (
        plan.device_retained_bytes + plan.device_action_bytes
    )

    construction_limited = build_discrete_design_from_setup(
        setup,
        source,
        control=DiscreteDesignControl(
            budget=DiscretizationBudget(
                max_table_bytes=plan.construction_peak_bytes - 1
            )
        ),
    )
    assert isinstance(construction_limited, DiscreteDesignUnavailable)
    assert construction_limited.required_bytes > construction_limited.budget_bytes

    action_limited = build_discrete_design_from_setup(
        setup,
        source,
        control=DiscreteDesignControl(
            budget=DiscretizationBudget(max_batch_bytes=plan.device_action_bytes - 1)
        ),
    )
    assert isinstance(action_limited, DiscreteDesignUnavailable)
    assert action_limited.required_bytes == plan.device_action_bytes


def test_budget_rejects_before_smooth_predictor_evaluation(monkeypatch) -> None:
    n = 120
    data = pd.DataFrame(
        {
            "x": np.resize(np.linspace(0.0, 1.0, 20), n),
            "group": pd.Categorical(np.resize(np.array(("a", "b", "c")), n)),
            "y": np.zeros(n),
        }
    )
    source = DataFrameRowSource(data, response="y")
    setup = ModelSetup.build(
        parse_formula("y ~ 0 + s(x, bs='cr', k=20, by=group)"), data
    )
    term = setup.coef_map.terms[0]
    indexed = exact_index_columns(
        source,
        ("x", "group"),
        budget=DiscretizationBudget(),
    )
    assert not isinstance(indexed, ExactIndexUnavailable)
    prospective = indexed.n_unique * term.n_coefs_raw * 8 * 8
    # Leave enough space for exact discovery and the estimated retained table,
    # but not for the predictor's raw factor-expanded evaluation workspace.
    limit = max(
        indexed.construction_peak_bytes,
        indexed.n_unique * term.n_coefs * 8,
    )
    assert limit < indexed.value_bytes + prospective
    smooth_type = type(term.smooth)
    original = smooth_type.predict_matrix
    called = False

    def spy(self, values):
        nonlocal called
        called = True
        return original(self, values)

    monkeypatch.setattr(smooth_type, "predict_matrix", spy)
    result = build_discrete_design_from_setup(
        setup,
        source,
        control=DiscreteDesignControl(
            budget=DiscretizationBudget(max_table_bytes=limit)
        ),
    )
    assert isinstance(result, DiscreteDesignUnavailable)
    assert "predictor workspace" in result.reason
    assert not called


def test_knot_predictors_charge_frozen_distance_scratch_before_evaluation(
    monkeypatch,
) -> None:
    training_x = np.linspace(0.01, 0.99, 193)
    training = pd.DataFrame({"x": training_x, "y": np.sin(training_x)})
    replay = DataFrameRowSource(
        pd.DataFrame({"x": np.repeat(0.31, 9), "y": np.repeat(0.5, 9)}),
        response="y",
    )
    for basis in ("tp", "gp"):
        setup = ModelSetup.build(
            parse_formula(f"y ~ 0 + s(x, bs='{basis}', k=5)"), training
        )
        spec = build_predict_spec(setup)
        smooth = spec.coef_map.terms[0].smooth
        assert smooth is not None
        smooth_type = type(smooth)
        original = smooth_type.predict_matrix
        calls = 0

        def spy(self, values, _original=original):
            nonlocal calls
            calls += 1
            return _original(self, values)

        with monkeypatch.context() as context:
            context.setattr(smooth_type, "predict_matrix", spy)
            result = build_discrete_design_from_predict_spec(
                spec,
                replay,
                control=DiscreteDesignControl(
                    batch_rows=9,
                    budget=DiscretizationBudget(max_table_bytes=1400),
                ),
            )
        assert isinstance(result, DiscreteDesignUnavailable)
        assert "predictor workspace" in result.reason
        assert result.required_bytes > 1544
        assert calls == 0
