"""Operator adjoints, frozen weighted/penalty actions and no-p² geometry."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.fitting.data import _to_jax_structure
from jaxgam.fitting.operators import (
    accumulate_rmatvec_batch,
    matvec_batch,
    normal_matvec_batch,
    penalty_matvec,
    rmatvec_batch,
    to_device_operator_batch,
)
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.penalties.structure import IdentityPenalty, IdentityTransform
from tests.helpers import _AssertCollector
from tests.test_formula.test_operator_prepare import _expanded, _frame
from tests.tolerances import STRICT


@pytest.fixture
def geometry():
    frame = _frame()
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(g,bs='re')+s(x,bs='cr',k=5)+s(u,h,bs='re')+s(g,h,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    batch = model.evaluate_batch(next(source.scan(7)))
    return model, batch


def test_batch_forward_adjoint_fused_normal_and_penalty_jit(geometry):
    model, rows = geometry
    device = to_device_operator_batch(rows, memory_budget_bytes=1_000_000)
    X = _expanded(rows)
    rng = np.random.default_rng(8547)
    v = rng.normal(size=model.n_coef)
    u = rng.normal(size=len(X))
    weight = np.linspace(0.4, 1.3, len(X))
    weight[2] = 0.0
    rho = np.linspace(-3, 2, model.penalties.n_penalties)
    structure = _to_jax_structure(model.penalties, None)
    collector = _AssertCollector()
    for name, actual, reference in (
        ("forward", matvec_batch(device, v), X @ v),
        ("adjoint", rmatvec_batch(device, u), X.T @ u),
        (
            "adjoint identity",
            jnp.vdot(matvec_batch(device, v), u),
            jnp.vdot(v, rmatvec_batch(device, u)),
        ),
        (
            "fused normal",
            normal_matvec_batch(device, v, weight),
            X.T @ (weight * (X @ v)),
        ),
        (
            "penalty",
            penalty_matvec(structure, v, rho),
            model.penalties.materialize(rho) @ v,
        ),
        (
            "symmetry",
            jnp.vdot(v, normal_matvec_batch(device, v[::-1].copy(), weight)),
            jnp.vdot(normal_matvec_batch(device, v, weight), v[::-1].copy()),
        ),
    ):
        collector.check(
            name,
            lambda actual=actual, reference=reference: np.testing.assert_allclose(
                actual, reference, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    # Explicit outer compilation also exercises all PyTree reconstruction paths.
    compiled = jax.jit(
        lambda b, x, w, s, r: normal_matvec_batch(b, x, w) + penalty_matvec(s, x, r)
    )
    np.testing.assert_allclose(
        compiled(device, v, weight, structure, rho),
        X.T @ (weight * (X @ v)) + model.penalties.materialize(rho) @ v,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    collector.raise_if_any("batch operator algebra")


def test_high_cardinality_jaxpr_has_no_expanded_design_or_square_system(monkeypatch):
    import pandas as pd

    n_levels = 6000
    frame = pd.DataFrame(
        {
            "g": pd.Categorical(
                ["g1", "g0", "g2"], categories=[f"g{i}" for i in range(n_levels)]
            ),
            "y": [1.0, 2.0, 3.0],
        }
    )
    model = prepare_operator_model(
        parse_formula("y~s(g,bs='re')"), DataFrameRowSource(frame, response="y")
    )
    for cls in (IdentityPenalty, IdentityTransform):
        monkeypatch.setattr(
            cls, "dense", lambda *_a: pytest.fail("Identity materialized")
        )
    structure = _to_jax_structure(model.penalties, None)
    source = DataFrameRowSource(frame, response="y")
    rows = to_device_operator_batch(
        model.evaluate_batch(next(source.scan(3))), memory_budget_bytes=1000
    )
    traced = jax.make_jaxpr(
        lambda b, v, w, s, r: normal_matvec_batch(b, v, w) + penalty_matvec(s, v, r)
    )(rows, jnp.ones(model.n_coef), jnp.ones(3), structure, jnp.zeros(1))

    def inspect_graph(closed):
        graph = closed.jaxpr if hasattr(closed, "jaxpr") else closed
        for eq in graph.eqns:
            for var in (*eq.invars, *eq.outvars):
                shape = getattr(getattr(var, "aval", None), "shape", ())
                assert not (len(shape) >= 2 and sum(d >= n_levels for d in shape) >= 2)
                assert shape not in ((3, n_levels), (3, model.n_coef))
            for value in eq.params.values():
                if hasattr(value, "jaxpr") or hasattr(value, "eqns"):
                    inspect_graph(value)

    inspect_graph(traced)
    result = normal_matvec_batch(rows, jnp.ones(model.n_coef), jnp.ones(3))
    assert result.shape == (model.n_coef,)
    assert float(result[0]) == 6.0
    assert rows.modest_values.shape == (3, 1)


def test_transfer_budget_rejects_before_device_put(monkeypatch, geometry):
    _, rows = geometry
    monkeypatch.setattr(
        jax, "device_put", lambda *_a, **_k: pytest.fail("Device transfer was called")
    )
    with pytest.raises(MemoryError):
        to_device_operator_batch(rows, memory_budget_bytes=1)


@pytest.mark.parametrize("budget", [False, 0, 1.5])
def test_transfer_budget_type_validation(geometry, budget):
    with pytest.raises(ValueError, match="positive integer"):
        to_device_operator_batch(geometry[1], memory_budget_bytes=budget)


@pytest.mark.parametrize(
    "invalid", ["columns", "values", "ids", "amplitudes", "overlap", "interval"]
)
def test_invalid_batch_geometry_rejects(geometry, invalid):
    _, rows = geometry
    if invalid == "columns":
        rows = replace(rows, modest_columns=(rows.n_coef,) * len(rows.modest_columns))
    elif invalid == "values":
        rows = replace(rows, modest_values=rows.modest_values + np.nan)
    else:
        items = list(rows.indexed_rows)
        ids, amplitude, start, size = items[0]
        if invalid == "ids":
            ids = ids.astype(float)
        elif invalid == "amplitudes":
            amplitude = amplitude + np.inf
        elif invalid == "overlap":
            start = rows.modest_columns[0]
        elif invalid == "interval":
            start = rows.n_coef
        items[0] = (ids, amplitude, start, size)
        rows = replace(rows, indexed_rows=tuple(items))
    with pytest.raises(ValueError, match=r"operator|Operator"):
        to_device_operator_batch(rows, memory_budget_bytes=1_000_000)


def test_static_vector_shape_validation(geometry):
    model, rows = geometry
    batch = to_device_operator_batch(rows, memory_budget_bytes=1_000_000)
    with pytest.raises(ValueError, match=r"dimension|batch|dimensions"):
        matvec_batch(batch, jnp.ones(model.n_coef + 1))
    with pytest.raises(ValueError, match=r"dimension|batch|dimensions"):
        rmatvec_batch(batch, jnp.ones(len(rows.modest_values) + 1))
    with pytest.raises(ValueError, match=r"dimension|batch|dimensions"):
        normal_matvec_batch(
            batch, jnp.ones(model.n_coef), jnp.ones((len(rows.modest_values), 1))
        )
    with pytest.raises(ValueError, match=r"dimension|batch|dimensions"):
        penalty_matvec(
            _to_jax_structure(model.penalties, None),
            jnp.ones(model.n_coef),
            jnp.zeros(model.penalties.n_penalties + 1),
        )


def test_donated_adjoint_accumulation_reuses_rhs_storage_and_validates_shapes(geometry):
    model, rows = geometry
    batch = to_device_operator_batch(rows, memory_budget_bytes=1_000_000)
    X = _expanded(rows)
    values = jnp.sin(jnp.arange(len(X), dtype=jnp.float64))
    accumulator = jnp.ones(model.n_coef)
    before = accumulator.unsafe_buffer_pointer()
    result = accumulate_rmatvec_batch(accumulator, batch, values)
    result.block_until_ready()
    assert result.unsafe_buffer_pointer() == before
    result = accumulate_rmatvec_batch(result, batch, 2 * values)
    np.testing.assert_allclose(
        result, 1 + 3 * X.T @ np.asarray(values), rtol=STRICT.rtol, atol=STRICT.atol
    )
    for initial, row_values, message in (
        (jnp.ones(model.n_coef + 1), values, "coefficient dimension"),
        (jnp.ones(model.n_coef), jnp.ones(len(X) + 1), "batch row count"),
    ):
        with pytest.raises(ValueError, match=message):
            accumulate_rmatvec_batch(initial, batch, row_values)
