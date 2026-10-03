"""Bounded exact Fisher contractions against pinned mgcv and source work."""

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator import FrozenScanSystem, PCGControl
from jaxgam.execution.operator_efs_trace import (
    ExactOperatorPenaltyStatistics,
    ExactPenaltyTraceSolveError,
)
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
def test_exact_operator_fisher_trace_and_quadratic_match_pinned_r(r_bridge, expression):
    row = np.arange(47)
    frame = pd.DataFrame(
        {
            "y": 0.8 + np.sin(row / 9) + 0.02 * np.cos(row),
            "x": np.linspace(0.1, 1.5, len(row)),
            "z": ((17 * row) % len(row)) / len(row),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    formula = f"y~{expression}"
    sp = np.array([0.7, 1.1])
    reference = r_bridge.fit_fixed_sp(
        formula, frame, "gaussian", sp, with_uncertainty=True
    )
    collector = _AssertCollector()
    for B in (1, 11):
        source, model = _model(frame, formula, batch_rows=B)
        system = FrozenScanSystem(
            model,
            source,
            np.log(sp),
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
