"""JIT indexed discrete operators against the explicit frozen design."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxgam.data.source import DataFrameRowSource
from jaxgam.fitting.discrete_ops import (
    DeviceDiscreteDesign,
    DeviceLookupBlock,
    DeviceTensorLookupBlock,
    discrete_diag_xvxd,
    discrete_xb_transpose,
    discrete_xbd,
    discrete_xwxd,
    discrete_xwyd,
    lookup_pair_histogram_crossproduct,
    pair_reduction_plan,
    to_device_discrete_design,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.discrete_design import (
    DiscreteDesignUnavailable,
    LookupBlock,
    build_discrete_design_from_predict_spec,
    build_discrete_design_from_setup,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.predict_matrix import build_predict_spec
from tests.helpers import _AssertCollector
from tests.r_bridge import DiscreteOperatorLayout
from tests.tolerances import STRICT


def _descriptor():
    n = 45
    x = np.resize(np.array((0.1, 0.3, 0.6, 0.9)), n)
    z = np.resize(np.array((0.2, 0.5, 0.8)), n)
    group = pd.Categorical(np.resize(np.array(("a", "b", "c")), n))
    data = pd.DataFrame({"x": x, "z": z, "group": group, "y": np.sin(x) + z})
    setup = ModelSetup.build(
        parse_formula("y ~ group + s(x, bs='cr', k=4) + te(x, z, bs='cr', k=3)"), data
    )
    descriptor = build_discrete_design_from_setup(
        setup, DataFrameRowSource(data, response="y")
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    return setup, descriptor


def _nested_jaxpr_shapes(value: object) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    if isinstance(value, dict):
        for item in value.values():
            shapes.extend(_nested_jaxpr_shapes(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            shapes.extend(_nested_jaxpr_shapes(item))
    elif hasattr(value, "jaxpr"):
        shapes.extend(_nested_jaxpr_shapes(value.jaxpr))
    elif hasattr(value, "eqns"):
        for equation in value.eqns:
            shapes.extend(
                tuple(variable.aval.shape)
                for variable in equation.outvars
                if hasattr(variable, "aval")
            )
            shapes.extend(_nested_jaxpr_shapes(equation.params))
    return shapes


def _nested_jaxpr_primitives(value: object) -> list[str]:
    primitives: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            primitives.extend(_nested_jaxpr_primitives(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            primitives.extend(_nested_jaxpr_primitives(item))
    elif hasattr(value, "jaxpr"):
        primitives.extend(_nested_jaxpr_primitives(value.jaxpr))
    elif hasattr(value, "eqns"):
        for equation in value.eqns:
            primitives.append(equation.primitive.name)
            primitives.extend(_nested_jaxpr_primitives(equation.params))
    return primitives


def _selectors(descriptor, positions):
    return tuple(
        tuple(jnp.asarray(value) for value in block)
        for block in descriptor.batch_indices(positions)
    )


def test_jitted_actions_match_explicit_design_across_batches_and_are_adjoint() -> None:
    setup, descriptor = _descriptor()
    device = to_device_discrete_design(descriptor)
    n, p = setup.X.shape
    beta = np.linspace(-0.4, 0.7, p)
    values = np.linspace(-1.0, 1.0, n)
    weights = np.linspace(0.2, 1.3, n)
    covariance = np.eye(p) + 0.03 * np.ones((p, p))
    xbd_parts = []
    transpose = np.zeros(p)
    xwyd = np.zeros(p)
    xwxd = np.zeros((p, p))
    diag_parts = []
    for start in range(0, n, 7):
        positions = np.arange(start, min(n, start + 7))
        selectors = _selectors(descriptor, positions)
        xbd_parts.append(np.asarray(discrete_xbd(device, selectors, jnp.asarray(beta))))
        transpose += np.asarray(
            discrete_xb_transpose(device, selectors, jnp.asarray(values[positions]))
        )
        xwyd += np.asarray(
            discrete_xwyd(
                device,
                selectors,
                jnp.asarray(weights[positions]),
                jnp.asarray(values[positions]),
            )
        )
        xwxd += np.asarray(
            discrete_xwxd(device, selectors, jnp.asarray(weights[positions]))
        )
        diag_parts.append(
            np.asarray(discrete_diag_xvxd(device, selectors, jnp.asarray(covariance)))
        )
    collector = _AssertCollector()
    collector.check(
        "jit_compiled",
        lambda: (
            (_ for _ in ()).throw(AssertionError("Xbd was not jitted"))
            if discrete_xbd._cache_size() == 0
            else None
        ),
    )
    collector.check(
        "xbd",
        lambda: np.testing.assert_allclose(
            np.concatenate(xbd_parts),
            setup.X @ beta,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "adjoint",
        lambda: np.testing.assert_allclose(
            transpose, setup.X.T @ values, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "xwyd",
        lambda: np.testing.assert_allclose(
            xwyd, setup.X.T @ (weights * values), rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "xwxd",
        lambda: np.testing.assert_allclose(
            xwxd,
            setup.X.T @ (weights[:, None] * setup.X),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "diag_xvxd",
        lambda: np.testing.assert_allclose(
            np.concatenate(diag_parts),
            np.sum((setup.X @ covariance) * setup.X, axis=1),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any("exact JIT discrete actions")


def test_bounded_pair_histogram_matches_direct_and_falls_back() -> None:
    _setup, descriptor = _descriptor()
    ordinary = [block for block in descriptor.blocks if isinstance(block, LookupBlock)]
    left, right = ordinary[:2]
    positions = np.arange(descriptor.n_rows)
    weights = np.linspace(0.3, 1.0, descriptor.n_rows)
    plan = pair_reduction_plan(left.index.n_unique, right.index.n_unique, 1_000_000)
    assert plan.method == "histogram"
    actual = lookup_pair_histogram_crossproduct(
        jnp.asarray(left.table),
        jnp.asarray(right.table),
        jnp.asarray(left.index.batch_selector(positions)),
        jnp.asarray(right.index.batch_selector(positions)),
        jnp.asarray(weights),
    )
    expected = left.table[left.index.selector].T @ (
        weights[:, None] * right.table[right.index.selector]
    )
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert pair_reduction_plan(100_000, 100_000, 1024).method == "direct"


def test_jitted_actions_cover_numeric_by_factor_by_and_ti() -> None:
    n = 45
    x = np.resize(np.array((0.1, 0.3, 0.6, 0.9)), n)
    z = np.resize(np.array((0.2, 0.5, 0.8)), n)
    group = pd.Categorical(np.resize(np.array(("a", "b", "c")), n))
    data = pd.DataFrame({"x": x, "z": z, "group": group, "y": np.sin(x) + z})
    setup = ModelSetup.build(
        parse_formula(
            "y ~ group + s(x, by=z, bs='cr', k=4) "
            "+ s(x, by=group, bs='cr', k=4) + ti(x, z, bs='cr', k=3)"
        ),
        data,
    )
    descriptor = build_discrete_design_from_setup(
        setup, DataFrameRowSource(data, response="y")
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    device = to_device_discrete_design(descriptor)
    selectors = _selectors(descriptor, np.arange(n))
    beta = np.linspace(-0.4, 0.6, setup.X.shape[1])
    weights = np.linspace(0.2, 1.2, n)
    values = np.linspace(-1.0, 1.0, n)
    covariance = np.eye(setup.X.shape[1]) + 0.02
    collector = _AssertCollector()
    collector.check(
        "numeric_factor_by_ti_xbd",
        lambda: np.testing.assert_allclose(
            np.asarray(discrete_xbd(device, selectors, jnp.asarray(beta))),
            setup.X @ beta,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "numeric_factor_by_ti_adjoint",
        lambda: np.testing.assert_allclose(
            np.asarray(discrete_xb_transpose(device, selectors, jnp.asarray(values))),
            setup.X.T @ values,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "numeric_factor_by_ti_crossproduct",
        lambda: np.testing.assert_allclose(
            np.asarray(discrete_xwxd(device, selectors, jnp.asarray(weights))),
            setup.X.T @ (weights[:, None] * setup.X),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "numeric_factor_by_ti_diag",
        lambda: np.testing.assert_allclose(
            np.asarray(discrete_diag_xvxd(device, selectors, jnp.asarray(covariance))),
            np.sum((setup.X @ covariance) * setup.X, axis=1),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any("by/factor/tensor JIT actions")


def test_jitted_numeric_tensor_by_actions_keep_separate_margins() -> None:
    n = 36
    x = np.resize(np.array((0.1, 0.3, 0.6, 0.9)), n)
    z = np.resize(np.array((0.2, 0.5, 0.8)), n)
    by = np.resize(np.array((0.4, 1.0, 1.6)), n)
    data = pd.DataFrame({"x": x, "z": z, "by": by, "y": np.sin(x) + z})
    setup = ModelSetup.build(parse_formula("y ~ te(x, z, bs='cr', k=3, by=by)"), data)
    descriptor = build_discrete_design_from_setup(
        setup, DataFrameRowSource(data, response="y")
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    tensor = next(
        block for block in descriptor.blocks if not isinstance(block, LookupBlock)
    )
    assert tensor.multiplier_index is not None
    device = to_device_discrete_design(descriptor)
    selectors = _selectors(descriptor, np.arange(n))
    beta = np.linspace(-0.3, 0.5, setup.X.shape[1])
    values = np.linspace(-1.0, 1.0, n)
    weights = np.linspace(0.2, 1.2, n)
    np.testing.assert_allclose(
        np.asarray(discrete_xbd(device, selectors, jnp.asarray(beta))),
        setup.X @ beta,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        np.asarray(discrete_xb_transpose(device, selectors, jnp.asarray(values))),
        setup.X.T @ values,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        np.asarray(discrete_xwxd(device, selectors, jnp.asarray(weights))),
        setup.X.T @ (weights[:, None] * setup.X),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_jitted_factor_tensor_by_including_ordered_reference() -> None:
    n = 36
    x = np.resize(np.array((0.1, 0.3, 0.6, 0.9)), n)
    z = np.resize(np.array((0.2, 0.5, 0.8)), n)
    for ordered in (False, True):
        levels = pd.Categorical(
            np.resize(np.array(("a", "b", "c")), n), ordered=ordered
        )
        data = pd.DataFrame({"x": x, "z": z, "g": levels, "y": np.sin(x) + z})
        setup = ModelSetup.build(
            parse_formula("y ~ te(x, z, bs='cr', k=3, by=g)"), data
        )
        descriptor = build_discrete_design_from_setup(
            setup, DataFrameRowSource(data, response="y")
        )
        assert not isinstance(descriptor, DiscreteDesignUnavailable)
        device = to_device_discrete_design(descriptor)
        selectors = _selectors(descriptor, np.arange(n))
        beta = np.linspace(-0.3, 0.5, setup.X.shape[1])
        values = np.linspace(-1.0, 1.0, n)
        weights = np.linspace(0.2, 1.2, n)
        np.testing.assert_allclose(
            np.asarray(discrete_xbd(device, selectors, jnp.asarray(beta))),
            setup.X @ beta,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            np.asarray(discrete_xb_transpose(device, selectors, jnp.asarray(values))),
            setup.X.T @ values,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            np.asarray(discrete_xwxd(device, selectors, jnp.asarray(weights))),
            setup.X.T @ (weights[:, None] * setup.X),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        if ordered:
            reference = np.asarray(levels == "a")
            smooth_only = beta.copy()
            smooth_only[0] = 0.0
            np.testing.assert_allclose(
                np.asarray(discrete_xbd(device, selectors, jnp.asarray(smooth_only)))[
                    reference
                ],
                0.0,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            )


def test_factor_tensor_jaxpr_has_no_expanded_batch_level_tensor() -> None:
    n = 24
    data = pd.DataFrame(
        {
            "x": np.resize(np.array((0.1, 0.4, 0.8)), n),
            "z": np.resize(np.array((0.2, 0.6, 0.9)), n),
            "g": pd.Categorical(np.resize(np.array(("a", "b", "c")), n)),
            "y": np.zeros(n),
        }
    )
    setup = ModelSetup.build(parse_formula("y ~ te(x, z, bs='cr', k=3, by=g)"), data)
    descriptor = build_discrete_design_from_setup(
        setup, DataFrameRowSource(data, response="y")
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    tensor = next(
        block for block in descriptor.blocks if not isinstance(block, LookupBlock)
    )
    device = to_device_discrete_design(descriptor)
    selectors = _selectors(descriptor, np.arange(n))
    forbidden = (n, tensor.n_levels * tensor.raw_n_coef)
    closed = jax.make_jaxpr(discrete_xbd)(device, selectors, jnp.ones(setup.X.shape[1]))
    crossproduct = jax.make_jaxpr(discrete_xwxd)(device, selectors, jnp.ones(n))
    shapes = _nested_jaxpr_shapes((closed, crossproduct))
    assert forbidden not in shapes
    primitives = _nested_jaxpr_primitives(crossproduct)
    assert "scan" in primitives
    assert primitives.count("dot_general") < 40


def test_factor_tensor_predict_spec_replays_subset_and_rejects_novel_level() -> None:
    n = 30
    data = pd.DataFrame(
        {
            "x": np.resize(np.array((0.1, 0.4, 0.8)), n),
            "z": np.resize(np.array((0.2, 0.6, 0.9)), n),
            "g": pd.Categorical(np.resize(np.array(("a", "b", "c")), n)),
            "y": np.zeros(n),
        }
    )
    setup = ModelSetup.build(parse_formula("y ~ te(x, z, bs='cr', k=3, by=g)"), data)
    subset = data.loc[data.g != "c"].reset_index(drop=True)
    source = DataFrameRowSource(subset, response="y")
    descriptor = build_discrete_design_from_predict_spec(
        build_predict_spec(setup), source
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    tensor = next(
        block for block in descriptor.blocks if not isinstance(block, LookupBlock)
    )
    assert tensor.n_levels == 3
    np.testing.assert_allclose(
        descriptor.evaluate_batch(np.arange(len(subset))),
        setup.X[data.g != "c"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    novel = DataFrameRowSource(subset.assign(g="novel"), response="y")
    with np.testing.assert_raises(ValueError):
        build_discrete_design_from_predict_spec(build_predict_spec(setup), novel)


def test_crossproduct_histogram_and_direct_device_paths_match() -> None:
    setup, descriptor = _descriptor()
    device = to_device_discrete_design(descriptor)
    selectors = _selectors(descriptor, np.arange(descriptor.n_rows))
    weights = np.linspace(0.2, 1.2, descriptor.n_rows)
    direct = replace(device, pair_policy="direct_only")
    histogram = np.asarray(discrete_xwxd(device, selectors, jnp.asarray(weights)))
    direct_result = np.asarray(discrete_xwxd(direct, selectors, jnp.asarray(weights)))
    np.testing.assert_allclose(
        histogram, direct_result, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        histogram,
        setup.X.T @ (weights[:, None] * setup.X),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_pinned_r_compact_actions_match_explicit_lookup_layout(r_bridge) -> None:
    """Exercise mgcv's Xd/kd/ks/ts/dt/v/qc translation directly."""
    n = 36
    x = np.resize(np.array((0.1, 0.4, 0.8)), n)
    z = np.resize(np.array((0.2, 0.6, 0.9)), n)
    data = pd.DataFrame({"x": x, "z": z, "y": np.sin(x) + z})
    setup = ModelSetup.build(
        parse_formula("y ~ s(x, bs='cr', k=3) + s(z, bs='cr', k=3)"), data
    )
    descriptor = build_discrete_design_from_setup(
        setup, DataFrameRowSource(data, response="y")
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    blocks = [block for block in descriptor.blocks if isinstance(block, LookupBlock)]
    layout = DiscreteOperatorLayout(
        marginal_tables=tuple(block.table for block in blocks),
        row_indices=np.column_stack([block.index.selector for block in blocks]),
        index_spans=np.column_stack(
            (np.arange(len(blocks)), np.arange(1, len(blocks) + 1))
        ).astype(np.int32),
        term_starts=tuple(range(len(blocks))),
        term_dimensions=(1,) * len(blocks),
        constraint_vectors=(),
        constraint_codes=np.zeros(len(blocks), dtype=np.int32),
        drop=None,
        r_to_public=np.arange(setup.X.shape[1], dtype=np.int32),
    )
    weights = np.linspace(0.2, 1.1, n)
    response = data.y.to_numpy()
    beta = np.linspace(-0.3, 0.5, setup.X.shape[1])
    covariance = np.eye(setup.X.shape[1]) + 0.01
    actual = r_bridge.discrete_operators(layout, weights, response, beta, covariance)
    collector = _AssertCollector()
    collector.check(
        "xbd",
        lambda: np.testing.assert_allclose(
            actual["xbd"], setup.X @ beta, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "xwyd",
        lambda: np.testing.assert_allclose(
            actual["xwyd"],
            setup.X.T @ (weights * response),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "xwxd",
        lambda: np.testing.assert_allclose(
            actual["xwxd"],
            setup.X.T @ (weights[:, None] * setup.X),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "diag",
        lambda: np.testing.assert_allclose(
            actual["diag_xvxd"],
            np.sum((setup.X @ covariance) * setup.X, axis=1),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any("pinned compact discrete operations")


def test_pinned_r_tensor_constraint_drop_and_permutation_match_device_actions(
    r_bridge,
) -> None:
    """Exercise mgcv qc/v/drop and a different compact-versus-public order."""
    first = np.array(((1.0, 0.2), (0.5, 1.1), (1.3, -0.4)))
    second = np.array(((0.7, 1.0), (1.2, -0.3), (-0.2, 0.8)))
    ordinary = np.array(((1.0, -0.4), (0.3, 0.9), (-0.5, 0.6)))
    selector_a = np.array((0, 1, 2, 0, 2, 1), dtype=np.int32)
    selector_b = np.array((2, 1, 0, 2, 0, 1), dtype=np.int32)
    selector_o = np.array((1, 2, 0, 1, 0, 2), dtype=np.int32)
    # mgcv's qc>0 contract uses Z = (I - vv') with first column discarded.
    v = np.full(4, 1.0 / np.sqrt(2.0))
    z_matrix = (np.eye(4) - np.outer(v, v))[:, 1:]
    np.testing.assert_allclose(z_matrix.T @ z_matrix, np.eye(3), atol=STRICT.atol)
    tensor_raw = np.einsum("bi,bj->bij", first[selector_a], second[selector_b]).reshape(
        len(selector_a), -1
    )
    # Public coefficients place the ordinary block first and retain compact
    # tensor coordinates 0 and 2 after dropping R coordinate 1.
    explicit = np.column_stack((ordinary[selector_o], tensor_raw @ z_matrix[:, (0, 2)]))
    device = DeviceDiscreteDesign(
        (
            DeviceLookupBlock(jnp.asarray(ordinary), 0),
            DeviceTensorLookupBlock(
                (jnp.asarray(first), jnp.asarray(second)),
                jnp.asarray(z_matrix),
                jnp.asarray(np.array((0, 2), dtype=np.int32)),
                2,
            ),
        ),
        4,
        "histogram_or_direct",
        1024,
        len(selector_a),
    )
    selectors = (
        (jnp.asarray(selector_o),),
        (jnp.asarray(selector_a), jnp.asarray(selector_b)),
    )
    layout = DiscreteOperatorLayout(
        marginal_tables=(first, second, ordinary),
        row_indices=np.column_stack((selector_a, selector_b, selector_o)),
        index_spans=np.array(((0, 1), (1, 2), (2, 3)), dtype=np.int32),
        term_starts=(0, 2),
        term_dimensions=(2, 1),
        constraint_vectors=(v, np.empty(0)),
        constraint_codes=np.array((1, 0), dtype=np.int32),
        drop=np.array((1,), dtype=np.int32),
        r_to_public=np.array((2, 3, 0, 1), dtype=np.int32),
    )
    weights = np.linspace(0.3, 1.1, len(selector_a))
    response = np.linspace(-0.8, 1.0, len(selector_a))
    beta = np.linspace(-0.4, 0.6, 4)
    covariance = np.eye(4) + 0.03
    actual = r_bridge.discrete_operators(layout, weights, response, beta, covariance)
    collector = _AssertCollector()
    collector.check(
        "frozen_tensor_xbd",
        lambda: np.testing.assert_allclose(
            np.asarray(discrete_xbd(device, selectors, jnp.asarray(beta))),
            explicit @ beta,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "r_tensor_xbd",
        lambda: np.testing.assert_allclose(
            actual["xbd"], explicit @ beta, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "r_tensor_xwyd",
        lambda: np.testing.assert_allclose(
            actual["xwyd"],
            explicit.T @ (weights * response),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "r_tensor_xwxd",
        lambda: np.testing.assert_allclose(
            actual["xwxd"],
            explicit.T @ (weights[:, None] * explicit),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "r_tensor_diag",
        lambda: np.testing.assert_allclose(
            actual["diag_xvxd"],
            np.sum((explicit @ covariance) * explicit, axis=1),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any("pinned tensor compact metadata")
