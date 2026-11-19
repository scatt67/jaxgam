"""Frozen source actions, host PCG statuses, scan and memory ownership."""

import pickle
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import ArrayRowSource, DataFrameRowSource
from jaxgam.execution.operator import (
    FrozenScanSystem,
    PCGControl,
    solve_host_pcg,
)
from jaxgam.fitting.iterative import MAX_ITERATIONS, NONFINITE, NONPOSITIVE_CURVATURE
from jaxgam.fitting.preconditioners import prepare_preconditioner
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    PreparedOperatorModel,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    IdentityPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
from tests.helpers import _AssertCollector
from tests.r_bridge import DiscreteOperatorLayout
from tests.tolerances import STRICT


def _frame(n=83):
    x = np.linspace(-1, 1, n)
    return pd.DataFrame(
        {
            "x": x,
            "y": np.sin(2 * x),
            "g": pd.Categorical(
                np.resize(["a", "c", "b"], n), categories=["c", "b", "a", "unused"]
            ),
            "h": pd.Categorical(np.resize(["u", "v", "v", "u"], n)),
            "u": np.cos(x),
        }
    )


@pytest.mark.parametrize(
    "formula",
    [
        "y~0+s(g,bs='re')",
        "y~x+s(g,h,bs='re')",
        "y~s(g,bs='re')+s(x,bs='cr',k=6)+s(u,h,bs='re')",
    ],
)
def test_frozen_scanned_system_matches_direct_actions_solve_and_scan_counts(formula):
    frame = _frame()
    spec = parse_formula(formula)
    dense = ModelSetup.build(spec, frame)
    weights = np.linspace(0.4, 1.6, len(frame))
    weights[::7] = 0
    rng = np.random.default_rng(10391)
    v = rng.normal(size=dense.X.shape[1])
    rho = np.linspace(-1.7, 0.4, dense.penalties.n_penalties)
    H = dense.X.T @ (weights[:, None] * dense.X) + dense.penalties.materialize(rho)
    rhs = rng.normal(size=len(v))
    collector = _AssertCollector()
    for B in (1, 13, 29):
        source = DataFrameRowSource(frame, response="y", weights=weights)
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        system = FrozenScanSystem(
            model, source, rho, memory_budget_bytes=8 * 1024 * 1024
        )
        original_rho = rho.copy()
        # Constructor owns one full preconditioner/certificate scan.
        assert system.diagnostics.completed_scans == 1
        assert system.diagnostics.normal_actions == 0
        action = system.matvec(jnp.asarray(v))
        collector.check(
            f"B{B}/normal action",
            lambda action=action: np.testing.assert_allclose(
                action, H @ v, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        assert system.donated_accumulator_bytes == 8 * len(v)
        before = replace(system.diagnostics)
        result = system.solve(rhs, control=PCGControl(rtol=1e-12))
        assert result.converged
        assert result.relative_residual <= 1e-12
        collector.check(
            f"B{B}/PCG",
            lambda result=result: np.testing.assert_allclose(
                result.solution,
                np.linalg.solve(H, rhs),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        assert (
            system.diagnostics.completed_scans - before.completed_scans
            == result.matvec_count
        )
        assert (
            system.diagnostics.normal_actions - before.normal_actions
            == result.matvec_count
        )
        assert (
            system.diagnostics.yielded_rows - before.yielded_rows
            == result.matvec_count * len(frame)
        )
        # Each action performs explicit entry plus before/after fingerprints.
        assert (
            system.diagnostics.fingerprint_checks - before.fingerprint_checks
            == 3 * result.matvec_count
        )
        assert not result.solution.flags.writeable
        assert result.vector_peak_bound_bytes == 112 * len(v)
        assert (
            result.additional_peak_bound_bytes
            == result.vector_peak_bound_bytes + result.executable_temporary_bytes
        )
        assert system._solver_reserve == 0
        if B == 13:
            adjoint_rhs = np.sin(np.arange(len(v)))
            adjoint = system.solve_adjoint(adjoint_rhs, control=PCGControl(rtol=1e-12))
            collector.check(
                "explicit host adjoint",
                lambda adjoint=adjoint, adjoint_rhs=adjoint_rhs: (
                    np.testing.assert_allclose(
                        adjoint.solution,
                        np.linalg.solve(H, adjoint_rhs),
                        rtol=STRICT.rtol,
                        atol=STRICT.atol,
                    )
                ),
            )
        np.testing.assert_array_equal(rho, original_rho)
    collector.raise_if_any(formula)


def test_selected_repeated_lineage_and_frozen_curvature_replay():
    frame = _frame(41)
    selection = np.array([40, 2, 2, 8, 1, 30, 0, 9])
    source = ArrayRowSource(
        {"g": frame.g, "x": frame.x}, y=frame.y, row_selection=selection
    )
    spec = parse_formula("y~x+s(g,bs='re')")
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=3)
    )
    dense = ModelSetup.build(spec, frame.iloc[selection])
    factor = [1.5]
    system = FrozenScanSystem(
        model,
        source,
        np.array([-0.3]),
        curvature=lambda batch: batch.weight * factor[0],
        memory_budget_bytes=1024 * 1024,
    )
    H = 1.5 * dense.X.T @ dense.X + dense.penalties.materialize(np.array([-0.3]))
    np.testing.assert_allclose(
        system.matvec(jnp.ones(model.n_coef)),
        H @ np.ones(model.n_coef),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    factor[0] = 2
    with pytest.raises(RuntimeError, match="curvature changed"):
        system.matvec(jnp.ones(model.n_coef))
    factor[0] = 1.5
    source._y = source._y.copy()
    source._y[2] += 1
    with pytest.raises(RuntimeError, match="source changed"):
        system.matvec(jnp.ones(model.n_coef))


def test_spd_certificate_rejects_invalid_actual_curvature_and_unidentified_nulls():
    frame = _frame()
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=11),
    )
    for curvature, message in (
        (lambda b: b.weight * jnp.nan, "finite row"),
        (lambda _b: np.ones(1), "finite row"),
    ):
        with pytest.raises(ValueError, match=message):
            FrozenScanSystem(
                model,
                source,
                np.zeros(1),
                curvature=curvature,
                memory_budget_bytes=1024 * 1024,
            )
    with pytest.raises(np.linalg.LinAlgError, match="signed system"):
        FrozenScanSystem(
            model,
            source,
            np.zeros(1),
            curvature=lambda b: -b.weight,
            memory_budget_bytes=1024 * 1024,
        )
    with pytest.raises(np.linalg.LinAlgError, match="Schur SPD"):
        FrozenScanSystem(
            model,
            source,
            np.zeros(1),
            curvature=lambda b: b.weight * 0,
            memory_budget_bytes=1024 * 1024,
        )


@pytest.mark.parametrize(
    ("formula", "negative_weight", "negative_rows", "uses_downdate"),
    [
        ("y~s(g,bs='re')", -0.01, 1, False),
        ("y~s(x,bs='cr',k=6)+s(g,bs='re')", -0.01, 1, False),
        ("y~s(g,bs='re')+s(h,bs='re')", -0.1, 1, True),
        ("y~s(g,bs='re')+s(h,bs='re')", -0.1, 2, True),
        ("y~s(g,bs='re')+s(h,bs='re')", -0.1, 9, True),
    ],
)
def test_signed_rows_with_spd_total_observed_system_use_actual_weights(
    formula, negative_weight, negative_rows, uses_downdate
):
    frame = _frame(41)
    source = DataFrameRowSource(frame, response="y")
    spec = parse_formula(formula)
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=7)
    )
    dense = ModelSetup.build(spec, frame)
    weights = np.ones(len(frame))
    weights[:negative_rows] = negative_weight
    rho = np.zeros(dense.penalties.n_penalties)
    H = dense.X.T @ (weights[:, None] * dense.X) + dense.penalties.materialize(rho)
    assert np.min(np.linalg.eigvalsh(H)) > 0
    system = FrozenScanSystem(
        model,
        source,
        rho,
        curvature=lambda batch: weights[batch.row_positions],
        memory_budget_bytes=8 * 1024 * 1024,
    )
    assert (system.certification_actions > 0) is uses_downdate
    assert system.construction_peak_bound_bytes <= 8 * 1024 * 1024
    vector = np.linspace(-0.5, 0.7, model.n_coef)
    np.testing.assert_allclose(
        system.matvec(jnp.asarray(vector)),
        H @ vector,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    rhs = np.sin(np.arange(model.n_coef))
    result = system.solve(rhs, control=PCGControl(rtol=1e-12))
    assert result.converged
    np.testing.assert_allclose(
        result.solution,
        np.linalg.solve(H, rhs),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize(
    ("levels", "negative_rows", "batch_rows", "rank"),
    [(4, 9, 1, 6), (5, 16, 7, 8)],
)
def test_distinct_crossed_signed_directions_keep_spd_total_system(
    levels, negative_rows, batch_rows, rank
):
    pairs = [(g, h) for g in range(levels) for h in range(levels)]
    frame = pd.DataFrame(
        {
            "y": np.sin(np.arange(len(pairs))),
            "g": pd.Categorical([f"g{g}" for g, _ in pairs]),
            "h": pd.Categorical([f"h{h}" for _, h in pairs]),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    spec = parse_formula("y~s(g,bs='re')+s(h,bs='re')")
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=batch_rows)
    )
    dense = ModelSetup.build(spec, frame)
    weights = np.ones(len(pairs))
    weights[:negative_rows] = -0.05
    rho = np.zeros(2)
    H = dense.X.T @ (weights[:, None] * dense.X) + dense.penalties.materialize(rho)
    assert np.min(np.linalg.eigvalsh(H)) > 0
    system = FrozenScanSystem(
        model,
        source,
        rho,
        curvature=lambda batch: weights[batch.row_positions],
        memory_budget_bytes=8 * 1024 * 1024,
    )
    assert system.certification_actions > 0
    assert system.certification_rank == rank
    assert system.diagnostics.completed_scans == system.certification_actions + 3
    assert system.construction_peak_bound_bytes <= system.memory_budget_bytes
    vector = np.sin(np.arange(model.n_coef))
    np.testing.assert_allclose(
        system.matvec(jnp.asarray(vector)),
        H @ vector,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    solved = system.solve(
        np.cos(np.arange(model.n_coef)), control=PCGControl(rtol=1e-12)
    )
    assert solved.converged
    np.testing.assert_allclose(
        solved.solution,
        np.linalg.solve(H, np.cos(np.arange(model.n_coef))),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_signed_certificate_cap_is_not_a_claim_of_indefiniteness():
    levels = 7
    pairs = [(g, h) for g in range(levels) for h in range(levels)]
    frame = pd.DataFrame(
        {
            "y": np.sin(np.arange(len(pairs))),
            "g": pd.Categorical([f"g{g}" for g, _ in pairs]),
            "h": pd.Categorical([f"h{h}" for _, h in pairs]),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    spec = parse_formula("y~s(g,bs='re')+s(h,bs='re')")
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=7)
    )
    dense = ModelSetup.build(spec, frame)
    weights = np.ones(len(pairs))
    weights[:16] = -0.05
    H = dense.X.T @ (weights[:, None] * dense.X) + dense.penalties.materialize(
        np.zeros(2)
    )
    assert np.min(np.linalg.eigvalsh(H)) > 0
    with pytest.raises(
        np.linalg.LinAlgError, match="no sufficient Schur SPD certificate"
    ):
        FrozenScanSystem(
            model,
            source,
            np.zeros(2),
            curvature=lambda batch: weights[batch.row_positions],
            memory_budget_bytes=8 * 1024 * 1024,
        )


def test_memory_preflight_rejects_before_source_scan_predictor_or_transfer(monkeypatch):
    frame = _frame()
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=11),
    )
    monkeypatch.setattr(
        source, "scan", lambda *_a: pytest.fail("prospective scan started")
    )
    monkeypatch.setattr(
        jax, "device_put", lambda *_a, **_k: pytest.fail("prospective transfer started")
    )
    with pytest.raises(MemoryError, match="Frozen operator requires"):
        FrozenScanSystem(model, source, np.zeros(1), memory_budget_bytes=1)


def test_host_pcg_statuses_and_budget_rejection_before_actions(monkeypatch):
    inverse = prepare_preconditioner(np.ones(8), memory_budget_bytes=100000)
    rhs = np.arange(1, 9, dtype=float)
    for action, status in (
        (lambda v: -v, NONPOSITIVE_CURVATURE),
        (lambda v: v * jnp.nan, NONFINITE),
    ):
        result = solve_host_pcg(action, inverse, rhs, available_memory_bytes=100000)
        assert result.status == status
        assert not result.converged
        assert result.matvec_count >= 2
    limited = solve_host_pcg(
        lambda v: jnp.arange(1, 9) * v,
        inverse,
        rhs,
        available_memory_bytes=100000,
        control=PCGControl(max_iterations=1, rtol=1e-14),
    )
    assert limited.status == MAX_ITERATIONS
    assert limited.matvec_count == 3
    monkeypatch.setattr(
        jax, "device_put", lambda *_a, **_k: pytest.fail("premature vector transfer")
    )
    with pytest.raises(MemoryError, match="vector ownership"):
        solve_host_pcg(
            lambda _v: pytest.fail("premature operator action"),
            inverse,
            rhs,
            available_memory_bytes=112 * 8 - 1,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"rtol": -1},
        {"atol": np.nan},
        {"rtol": False},
        {"rtol": 0, "atol": 0},
        {"max_iterations": 0},
        {"max_iterations": 1.5},
    ],
)
def test_invalid_pcg_control(options):
    with pytest.raises(ValueError, match=r"tolerance|finite|positive integer"):
        PCGControl(**options)


def test_host_validation_and_invalid_matvec_output():
    inverse = prepare_preconditioner(np.ones(4), memory_budget_bytes=100000)
    with pytest.raises(ValueError, match="dimension"):
        solve_host_pcg(lambda v: v, inverse, np.ones(3), available_memory_bytes=100000)
    for rhs, initial, message in (
        (np.ones(4, dtype=np.float32), None, "rhs"),
        (np.ones(4), np.ones(3), "initial"),
        (np.array([np.nan, 1, 1, 1]), None, "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            solve_host_pcg(
                lambda v: v,
                inverse,
                rhs,
                initial=initial,
                available_memory_bytes=100000,
            )
    with pytest.raises(ValueError, match="CPU float64 coefficient"):
        solve_host_pcg(
            lambda _v: jnp.ones(3), inverse, np.ones(4), available_memory_bytes=100000
        )
    with pytest.raises(ValueError, match="positive integer"):
        solve_host_pcg(lambda v: v, inverse, np.ones(4), available_memory_bytes=False)
    with pytest.raises(TypeError, match="PCGControl"):
        solve_host_pcg(
            lambda v: v,
            inverse,
            np.ones(4),
            control=None,
            available_memory_bytes=100000,
        )


def test_frozen_modest_constraints_owned_and_host_solution_pickle_reconstruction():
    frame = _frame()
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(x,bs='cr',k=6)+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=17),
    )
    rho = np.array([-0.3, 0.4])
    system = FrozenScanSystem(model, source, rho, memory_budget_bytes=4 * 1024 * 1024)
    expected = np.asarray(system.matvec(jnp.ones(model.n_coef))).copy()
    rho[:] = 12
    for term in model.modest.predict_spec.coef_map.terms:
        if term.Z_centering is not None:
            term.Z_centering[:] = 0
    np.testing.assert_allclose(
        system.matvec(jnp.ones(model.n_coef)),
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    for term in system.model.modest.predict_spec.coef_map.terms:
        if term.Z_centering is not None:
            assert not term.Z_centering.flags.writeable
    rhs = np.cos(np.arange(model.n_coef))
    result = system.solve(rhs, initial=np.ones(model.n_coef))
    restored = pickle.loads(pickle.dumps(result))
    assert restored.converged
    assert not restored.solution.flags.writeable
    np.testing.assert_array_equal(restored.solution, result.solution)


def test_symbolic_product_storage_and_runtime_without_global_square(monkeypatch):
    original_dense = IdentityPenalty.dense

    def guarded_dense(penalty):
        if penalty.size > 64:
            pytest.fail("Unbounded identity materialization")
        return original_dense(penalty)

    monkeypatch.setattr(IdentityPenalty, "dense", guarded_dense)
    monkeypatch.setattr(
        PenaltyStructure,
        "materialize",
        lambda *_a, **_k: pytest.fail("Global penalty materialization"),
    )
    reports = []
    for levels in (80, 160):
        frame = pd.DataFrame(
            {
                "y": np.sin(np.arange(43)),
                "g": pd.Categorical(
                    np.resize(["g0", "g1"], 43),
                    categories=[f"g{i}" for i in range(levels)],
                ),
                "h": pd.Categorical(
                    np.resize(["h0", "h1", "h2"], 43),
                    categories=[f"h{i}" for i in range(levels)],
                ),
            }
        )
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            parse_formula("y~s(g,h,bs='re')"),
            source,
            control=OperatorPreparationControl(batch_rows=13),
        )
        system = FrozenScanSystem(
            model, source, np.zeros(1), memory_budget_bytes=32 * 1024 * 1024
        )
        assert model.n_coef == levels**2 + 1
        assert system.donated_accumulator_bytes == 8 * model.n_coef
        pointers = []
        compiled = system._normal

        def observed(accumulator, *args, compiled=compiled, pointers=pointers):
            before = accumulator.unsafe_buffer_pointer()
            updated = compiled(accumulator, *args)
            updated.block_until_ready()
            pointers.append((before, updated.unsafe_buffer_pointer()))
            return updated

        system._normal = observed
        result = system.solve(
            np.cos(np.arange(model.n_coef)), control=PCGControl(rtol=1e-11)
        )
        assert result.converged
        assert all(old == new for old, new in pointers)
        assert system.diagnostics.completed_scans == result.matvec_count + 1
        reports.append(
            (
                model.n_coef,
                system.executable_temporary_bytes,
                result.executable_temporary_bytes,
                result.additional_peak_bound_bytes,
            )
        )
    for report in reports:
        p, action_scratch, solve_scratch, peak = report
        assert action_scratch < 32 * p + 65536
        assert solve_scratch < 64 * p + 65536
        assert peak < 200 * p + 65536
    assert reports[1][3] <= 4.2 * reports[0][3]


def test_no_modest_large_factor_signed_construction_budget_preflights_before_scan(
    monkeypatch,
):
    levels = 4096
    frame = pd.DataFrame(
        {
            "y": np.sin(np.arange(57)),
            "g": pd.Categorical(
                np.resize(["g0", "g1"], 57),
                categories=[f"g{i}" for i in range(levels)],
            ),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~0+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=13),
    )
    assert not model.modest_columns
    assert model.n_coef == levels
    B, p = model.batch_rows, model.n_coef
    row_bound = model.batch_workspace_bound(B) + 32 * B * (2 + 8)
    construction = (
        model.metadata_bound_bytes
        + row_bound
        + 65 * p
        + 256 * 8 * (len(model.random_effects) + 2)
        + 4096
        + 2 * 64 * model.modest_raw_width**2
    )
    original_scan = source.scan
    curvature = np.ones(len(frame))
    curvature[0] = -0.1
    monkeypatch.setattr(
        source, "scan", lambda *_a: pytest.fail("underbudget setup started a scan")
    )
    with pytest.raises(MemoryError, match="Frozen operator requires"):
        FrozenScanSystem(
            model,
            source,
            np.zeros(1),
            curvature=lambda batch: curvature[batch.row_positions],
            memory_budget_bytes=construction - 1,
        )
    monkeypatch.setattr(source, "scan", original_scan)
    system = FrozenScanSystem(
        model,
        source,
        np.zeros(1),
        curvature=lambda batch: curvature[batch.row_positions],
        memory_budget_bytes=8 * 1024 * 1024,
    )
    assert system.construction_peak_bound_bytes >= construction
    assert system.construction_peak_bound_bytes <= 8 * 1024 * 1024
    vector = np.linspace(-0.2, 0.3, p)
    actual = np.asarray(system.matvec(jnp.asarray(vector)))
    expected = vector.copy()
    expected[0] += np.sum(curvature[frame.g == "g0"]) * vector[0]
    expected[1] += np.sum(curvature[frame.g == "g1"]) * vector[1]
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)


def test_frozen_host_solver_against_pinned_same_input_discrete_actions(r_bridge):
    frame = _frame()
    weights = np.linspace(0.5, 1.5, len(frame))
    weights[::9] = 0
    spec = parse_formula("y~s(x,bs='cr',k=6)+s(g,h,bs='re')")
    source = DataFrameRowSource(frame, response="y", weights=weights)
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=13)
    )
    dense = ModelSetup.build(spec, frame)
    X = dense.X
    p = X.shape[1]
    rho = np.array([-0.8, 0.3])
    beta = np.cos(np.arange(p))
    layout = DiscreteOperatorLayout(
        (X,),
        np.arange(len(frame), dtype=np.int64)[:, None],
        np.array([[0, 1]], dtype=np.int64),
        (0,),
        (1,),
        (),
        np.zeros(1, dtype=np.int32),
    )
    expected = r_bridge.discrete_operators(
        layout, weights, frame.y.to_numpy(), beta, np.eye(p)
    )
    penalty = dense.penalties.materialize(rho)
    system = FrozenScanSystem(model, source, rho, memory_budget_bytes=4 * 1024 * 1024)
    collector = _AssertCollector()
    normal = system.matvec(jnp.asarray(beta))
    collector.check(
        "pinned XWXd normal action",
        lambda: np.testing.assert_allclose(
            normal,
            expected["xwxd"] @ beta + penalty @ beta,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    result = system.solve(expected["xwyd"], control=PCGControl(rtol=1e-12))
    assert result.converged
    collector.check(
        "same-input pinned system solution",
        lambda: np.testing.assert_allclose(
            result.solution,
            np.linalg.solve(expected["xwxd"] + penalty, expected["xwyd"]),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any("frozen operator R/C action contract")


@pytest.mark.parametrize("modest_sp", [1.0, 1e13])
def test_certificate_uses_weighted_penalty_and_indexed_coupling(modest_sp):
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0], "g": pd.Categorical(["a"] * 3)})
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=2),
    )
    # A rounded PSD boundary can contain a tiny negative direction. It must
    # remain in the actual operator; its unweighted size cannot certify SPD.
    modest = PenaltyBlock(
        0,
        1,
        (0,),
        (DenseLocalPenalty(np.array([[-1e-13]])),),
        IdentityTransform(1),
        (0,),
    )
    indexed = replace(model.penalties.blocks[0], sp_indices=(1,))
    model = replace(model, penalties=PenaltyStructure(2, (modest, indexed)))
    rho = np.log(np.array([modest_sp, 0.1]))
    H = np.array([[3.0 - modest_sp * 1e-13, 3.0], [3.0, 3.1]])
    if modest_sp == 1e13:
        assert H[0, 0] > 0  # A modest-only Cholesky would pass incorrectly.
        assert np.linalg.eigvalsh(H)[0] < 0
        with pytest.raises(np.linalg.LinAlgError, match="Schur SPD"):
            FrozenScanSystem(model, source, rho, memory_budget_bytes=1024 * 1024)
    else:
        system = FrozenScanSystem(model, source, rho, memory_budget_bytes=1024 * 1024)
        actual = system.matvec(jnp.ones(2))
        np.testing.assert_allclose(
            actual, H @ np.ones(2), rtol=STRICT.rtol, atol=STRICT.atol
        )
        assert system.model.penalties.blocks[0].local_penalties[0].dense()[0, 0] < 0


def test_curvature_from_owned_rows_reuses_batch_geometry_and_keeps_true_system(
    monkeypatch,
):
    frame = _frame()
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~x+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=13),
    )
    original = PreparedOperatorModel.evaluate_batch
    evaluations = []

    def observed(self, batch, **options):
        evaluations.append(len(batch.row_positions))
        return original(self, batch, **options)

    monkeypatch.setattr(PreparedOperatorModel, "evaluate_batch", observed)

    def actual(batch, rows):
        return batch.weight * (1 + rows.modest_values[:, 1] ** 2)

    system = FrozenScanSystem(
        model,
        source,
        np.zeros(1),
        curvature_from_rows=actual,
        memory_budget_bytes=2 * 1024 * 1024,
    )
    vector = jnp.cos(jnp.arange(model.n_coef))
    result = system.matvec(vector)
    dense = ModelSetup.build(parse_formula("y~x+s(g,bs='re')"), frame)
    weight = 1 + np.asarray(frame.x) ** 2
    expected = dense.X.T @ (weight * (dense.X @ np.asarray(vector)))
    expected += dense.penalties.materialize(np.zeros(1)) @ np.asarray(vector)
    np.testing.assert_allclose(result, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    assert len(evaluations) == system.diagnostics.yielded_batches
    assert sum(evaluations) == system.diagnostics.yielded_rows
    with pytest.raises(ValueError, match="Choose one"):
        FrozenScanSystem(
            model,
            source,
            np.zeros(1),
            curvature=lambda b: b.weight,
            curvature_from_rows=actual,
            memory_budget_bytes=2 * 1024 * 1024,
        )
    with pytest.raises(TypeError, match="callable"):
        FrozenScanSystem(
            model,
            source,
            np.zeros(1),
            curvature_from_rows=1,
            memory_budget_bytes=2 * 1024 * 1024,
        )
