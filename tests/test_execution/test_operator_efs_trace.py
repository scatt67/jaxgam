"""Bounded exact Fisher contractions against pinned mgcv and source work."""

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator import FrozenScanSystem, PCGControl
from jaxgam.execution.operator_efs_trace import (
    ExactOperatorPenaltyStatistics,
    ExactPenaltyTraceSolveError,
    StochasticOperatorPenaltyStatistics,
    StochasticPenaltyTraceSolveError,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.penalties.structure import PenaltyStructure
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


def _assert_strict(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)


def _model(frame, formula, *, batch_rows=7):
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula(formula),
        source,
        control=OperatorPreparationControl(batch_rows=batch_rows),
    )
    return source, model


@pytest.mark.parametrize(
    "expression", ["s(x,bs='cr',k=6)+s(g,bs='re')", "te(x,z,bs='cr',k=4)"]
)
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_exact_operator_fisher_trace_and_quadratic_match_pinned_r(
    r_bridge, expression, family
):
    row = np.arange(47)
    frame = pd.DataFrame(
        {
            "y": (
                0.8 + np.sin(row / 9) + 0.02 * np.cos(row)
                if family == "gaussian"
                else np.resize([1, 3, 2, 4, 2, 5], len(row))
            ),
            "x": np.linspace(0.1, 1.5, len(row)),
            "z": ((17 * row) % len(row)) / len(row),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    formula = f"y~{expression}"
    sp = np.array([0.7, 1.1])
    reference = r_bridge.fit_fixed_sp(formula, frame, family, sp, with_uncertainty=True)
    collector = _AssertCollector()
    for B in (1, 11):
        source, model = _model(frame, formula, batch_rows=B)
        curvature = (
            None
            if family == "gaussian"
            else lambda batch, mu=reference["fitted_values"]: mu[batch.row_positions]
        )
        system = FrozenScanSystem(
            model,
            source,
            np.log(sp),
            curvature=curvature,
            memory_budget_bytes=8 * 1024 * 1024,
        )
        result = ExactOperatorPenaltyStatistics(
            system, memory_budget_bytes=16 * 1024 * 1024
        ).statistics(reference["coefficients"])
        expected_trace = np.zeros(len(sp))
        expected_quadratic = np.zeros(len(sp))
        beta = reference["coefficients"]
        for block in model.penalties.blocks:
            for index, penalty in zip(
                block.sp_indices, block.local_penalties, strict=True
            ):
                local = penalty.dense()
                interval = slice(block.start, block.stop)
                expected_trace[index] += (
                    np.trace(reference["Vp"][interval, interval] @ local)
                    / reference["scale"]
                )
                expected_quadratic[index] += beta[interval] @ local @ beta[interval]
        collector.check(
            f"{expression}/B{B}/Fisher trace",
            lambda actual=result.traces, expected=expected_trace: _assert_strict(
                actual, expected
            ),
        )
        q_expected = expected_quadratic
        collector.check(
            f"{expression}/B{B}/quadratic",
            lambda actual=result.quadratics, expected=q_expected: _assert_strict(
                actual, expected
            ),
        )
        assert result.diagnostics.completed_coordinates == sum(
            block.size for block in model.penalties.blocks
        )
        assert result.diagnostics.normal_actions == result.diagnostics.source_scans
        assert not result.traces.flags.writeable
    collector.raise_if_any("Exact operator Fisher penalty statistics")


def test_exact_trace_rejects_uncapped_work_and_budget_before_solve(monkeypatch):
    row = np.arange(29)
    frame = pd.DataFrame(
        {"y": 1 + row / 29, "g": pd.Categorical(np.resize(["a", "b"], 29))}
    )
    source, model = _model(frame, "y~s(g,bs='re')")
    system = FrozenScanSystem(
        model, source, np.array([0.0]), memory_budget_bytes=1024 * 1024
    )
    monkeypatch.setattr(
        system, "solve", lambda *_a, **_k: pytest.fail("Fisher solve started")
    )
    beta = np.zeros(model.n_coef)
    with pytest.raises(ValueError, match="coordinate cap"):
        ExactOperatorPenaltyStatistics(
            system, memory_budget_bytes=2 * 1024 * 1024, max_trace_coordinates=1
        ).statistics(beta)
    with pytest.raises(MemoryError, match="require"):
        ExactOperatorPenaltyStatistics(
            system, memory_budget_bytes=system.memory_budget_bytes + 1
        ).statistics(beta)


def test_exact_trace_preserves_failed_solve_status_and_no_dense_penalty(monkeypatch):
    row = np.arange(39)
    frame = pd.DataFrame(
        {
            "y": 0.1 * row,
            "x": np.linspace(0, 1, len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    source, model = _model(frame, "y~x+s(g,bs='re')")
    system = FrozenScanSystem(
        model, source, np.array([0.0]), memory_budget_bytes=1024 * 1024
    )
    monkeypatch.setattr(
        PenaltyStructure,
        "materialize",
        lambda *_a: pytest.fail("Global penalty materialized"),
    )
    provider = ExactOperatorPenaltyStatistics(
        system,
        memory_budget_bytes=2 * 1024 * 1024,
        control=PCGControl(rtol=1e-14, max_iterations=1),
    )
    with pytest.raises(ExactPenaltyTraceSolveError) as failure:
        provider.statistics(np.zeros(model.n_coef))
    assert failure.value.coordinate == model.penalties.blocks[0].start
    assert failure.value.iterations == 1


def test_shared_probes_trace_large_diagonal_system_without_coefficient_square(
    monkeypatch,
):
    row = np.arange(51)
    frame = pd.DataFrame(
        {
            "y": 1.0 + row / 51,
            "g": pd.Categorical(
                [f"g{index}" for index in row % 3],
                categories=[f"g{index}" for index in range(1536)],
            ),
        }
    )
    source, model = _model(frame, "y~0+s(g,bs='re')", batch_rows=11)
    assert model.n_coef == 1536
    system = FrozenScanSystem(
        model, source, np.log(np.array([0.8])), memory_budget_bytes=4 * 1024 * 1024
    )
    monkeypatch.setattr(
        PenaltyStructure,
        "materialize",
        lambda *_a, **_k: pytest.fail("Global penalty materialized"),
    )
    beta = np.zeros(model.n_coef, dtype=np.float64)
    beta[:3] = [0.2, -0.1, 0.3]
    exact = ExactOperatorPenaltyStatistics(
        system, memory_budget_bytes=8 * 1024 * 1024
    ).statistics(beta)
    assert exact.diagnostics.strategy == "structured_diagonal"
    assert exact.diagnostics.completed_coordinates == 0
    result = StochasticOperatorPenaltyStatistics(
        system,
        memory_budget_bytes=8 * 1024 * 1024,
        probes=4,
        seed=17,
    ).statistics(beta)
    scale = model.penalties.blocks[0].local_penalties[0].scale
    counts = np.zeros(model.n_coef)
    counts[:3] = [17, 17, 17]
    expected_trace = np.sum(scale / (counts + 0.8 * scale))
    expected_quadratic = scale * np.dot(beta, beta)
    _assert_strict(exact.traces, [expected_trace])
    _assert_strict(exact.quadratics, [expected_quadratic])
    _assert_strict(result.traces, [expected_trace])
    _assert_strict(result.quadratics, [expected_quadratic])
    _assert_strict(result.per_probe[:, 0], expected_trace)
    assert result.diagnostics.normal_actions == result.diagnostics.source_scans
    assert result.diagnostics.peak_bound_bytes < 8 * 1024 * 1024
    assert not result.per_probe.flags.writeable
    assert 8 * model.n_coef**2 > 8 * 1024 * 1024


def test_shared_probe_calibration_across_independent_seeds():
    row = np.arange(43)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.1 * np.sin(row),
            "x": np.linspace(-1.0, 1.0, len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    formula = "y~x+s(g,bs='re')"
    source, model = _model(frame, formula)
    rho = np.log(np.array([0.7]))
    system = FrozenScanSystem(model, source, rho, memory_budget_bytes=4 * 1024 * 1024)
    dense = ModelSetup.build(parse_formula(formula), frame)
    S = dense.penalties.materialize(rho)
    H = dense.X.T @ dense.X + S
    local = model.penalties.blocks[0].local_penalties[0].dense()
    interval = slice(model.penalties.blocks[0].start, model.penalties.blocks[0].stop)
    exact = np.trace(np.linalg.inv(H)[interval, interval] @ local)
    beta = np.linspace(-0.2, 0.2, model.n_coef)
    fingerprints = []
    for seed in (3, 7, 11, 19):
        result = StochasticOperatorPenaltyStatistics(
            system, memory_budget_bytes=8 * 1024 * 1024, probes=32, seed=seed
        ).statistics(beta)
        error = abs(result.traces[0] - exact)
        assert error <= 4 * result.monte_carlo_standard_error[0]
        assert result.diagnostics.maximum_relative_residual < 1e-8
        fingerprints.append(result.diagnostics.probe_fingerprint)
    assert len(set(fingerprints)) == 4


def test_shared_probe_failure_and_budget_are_explicit():
    row = np.arange(39)
    frame = pd.DataFrame(
        {
            "y": 0.1 * row,
            "x": np.linspace(0, 1, len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    source, model = _model(frame, "y~x+s(g,bs='re')")
    system = FrozenScanSystem(
        model, source, np.array([0.0]), memory_budget_bytes=1024 * 1024
    )
    beta = np.zeros(model.n_coef)
    with pytest.raises(MemoryError, match="require"):
        StochasticOperatorPenaltyStatistics(
            system, memory_budget_bytes=system.memory_budget_bytes + 1
        ).statistics(beta)
    provider = StochasticOperatorPenaltyStatistics(
        system,
        memory_budget_bytes=2 * 1024 * 1024,
        control=PCGControl(rtol=1e-14, max_iterations=1),
    )
    with pytest.raises(StochasticPenaltyTraceSolveError) as failure:
        provider.statistics(beta)
    assert failure.value.probe == 0
    assert failure.value.iterations == 1
