"""Checked arrowhead source algebra at small and release-sized dimensions."""

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator import FrozenScanSystem
from jaxgam.execution.operator_arrowhead import ExactArrowheadFactorization
from jaxgam.execution.operator_efs_trace import ExactOperatorPenaltyStatistics
from jaxgam.execution.operator_logdet import FrozenOperatorLogDet
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from tests.tolerances import STRICT


def _system(levels: int, *, batch_rows: int = 17):
    n = 2 * levels
    row = np.arange(n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.1 * np.sin(row),
            "x": np.linspace(-0.9, 1.1, n),
            "g": pd.Categorical(np.resize([f"g{i}" for i in range(levels)], n)),
        }
    )
    formula = "y~x+s(g,bs='re')"
    source = DataFrameRowSource(frame, response="y")
    spec = parse_formula(formula)
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=batch_rows)
    )
    rho = np.array([np.log(0.8)], dtype=np.float64)

    def curvature(batch):
        return 1.0 + 0.2 * frame.x.to_numpy()[batch.row_positions]

    system = FrozenScanSystem(
        model,
        source,
        rho,
        curvature=curvature,
        memory_budget_bytes=8 * 1024 * 1024,
    )
    dense = ModelSetup.build(spec, frame)
    weight = 1.0 + 0.2 * frame.x.to_numpy()
    hessian = dense.X.T @ (weight[:, None] * dense.X)
    hessian += dense.penalties.materialize(rho)
    return system, hessian


@pytest.mark.parametrize("levels", [5, 300])
def test_arrowhead_logdet_and_inverse_match_dense_at_coupled_sizes(levels):
    system, hessian = _system(levels)
    assert system.has_exact_arrowhead
    prior = system.diagnostics.completed_scans
    factor = ExactArrowheadFactorization.from_system(
        system, memory_budget_bytes=16 * 1024 * 1024
    )
    modest = np.array(system.model.modest_columns)
    indexed = np.arange(system.model.random_effects[0].col_start, hessian.shape[0])
    expected_inverse = np.linalg.inv(hessian)
    dense_lower = np.linalg.cholesky(hessian)
    modest_diag, indexed_diag = factor.inverse_diagonal()
    np.testing.assert_allclose(
        factor.log_determinant,
        2 * np.sum(np.log(np.diag(dense_lower))),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        modest_diag,
        np.diag(expected_inverse)[modest],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        indexed_diag,
        np.diag(expected_inverse)[indexed],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        factor.modest_inverse(),
        expected_inverse[np.ix_(modest, modest)],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert factor.source_scans == system.diagnostics.completed_scans - prior == 1
    assert factor.coupling.shape == (levels, len(modest))
    assert not factor.coupling.flags.writeable
    assert factor.peak_bound_bytes < 16 * 1024 * 1024
    eigenvalues = np.linalg.eigvalsh(hessian)
    assert 0 < factor.spectral_lower_bound <= eigenvalues[0] * (1 + 1e-10)
    assert factor.spectral_condition_bound >= (eigenvalues[-1] / eigenvalues[0]) * (
        1 - 1e-10
    )


@pytest.mark.parametrize("levels", [5, 300])
def test_arrowhead_logdet_and_fisher_penalties_share_exact_dense_target(levels):
    system, hessian = _system(levels)
    coefficients = np.linspace(-0.2, 0.2, system.model.n_coef)
    observed = FrozenOperatorLogDet(
        system, memory_budget_bytes=16 * 1024 * 1024
    ).exact_structured_arrowhead()
    fisher = ExactOperatorPenaltyStatistics(
        system, memory_budget_bytes=16 * 1024 * 1024
    ).statistics(coefficients)
    inverse = np.linalg.inv(hessian)
    expected_trace = np.zeros(system.model.penalties.n_penalties)
    expected_quadratic = np.zeros_like(expected_trace)
    for block in system.model.penalties.blocks:
        interval = slice(block.start, block.stop)
        for sp, penalty in zip(block.sp_indices, block.local_penalties, strict=True):
            matrix = penalty.dense()
            expected_trace[sp] += np.trace(inverse[interval, interval] @ matrix)
            beta = coefficients[interval]
            expected_quadratic[sp] += beta @ matrix @ beta
    np.testing.assert_allclose(
        observed.value,
        2 * np.sum(np.log(np.diag(np.linalg.cholesky(hessian)))),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        fisher.traces, expected_trace, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        fisher.quadratics,
        expected_quadratic,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert observed.diagnostics.strategy == "structured_arrowhead"
    assert observed.diagnostics.rounding_error_bound is None
    assert observed.diagnostics.spectral_lower_bound is not None
    assert observed.diagnostics.spectral_condition_bound is not None
    assert observed.diagnostics.completed_actions == 0
    assert observed.diagnostics.source_scans == 1
    assert fisher.diagnostics.strategy == "structured_arrowhead"
    assert fisher.diagnostics.completed_coordinates == 0
    assert fisher.diagnostics.source_scans == 1


def test_arrowhead_budget_preflights_before_replay(monkeypatch):
    system, _ = _system(8)
    monkeypatch.setattr(
        system,
        "exact_arrowhead_coupling",
        lambda **_: pytest.fail("source replay began before memory preflight"),
    )
    with pytest.raises(MemoryError, match="budget"):
        ExactArrowheadFactorization.from_system(
            system, memory_budget_bytes=system.memory_budget_bytes + 1
        )


def test_arrowhead_replay_detects_changed_curvature(monkeypatch):
    system, _ = _system(8)
    original = system._weight

    def changed_weight(batch, rows):
        return 1.01 * original(batch, rows)

    monkeypatch.setattr(system, "_weight", changed_weight)
    with pytest.raises(RuntimeError, match="curvature changed"):
        ExactArrowheadFactorization.from_system(
            system, memory_budget_bytes=16 * 1024 * 1024
        )


def test_diagonal_system_is_not_arrowhead():
    frame = pd.DataFrame(
        {
            "y": np.ones(12),
            "g": pd.Categorical(np.resize(["a", "b", "c"], 12)),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~0+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    system = FrozenScanSystem(
        model,
        source,
        np.array([0.0]),
        memory_budget_bytes=8 * 1024 * 1024,
    )
    assert not system.has_exact_arrowhead
    with pytest.raises(ValueError, match="arrowhead"):
        ExactArrowheadFactorization.from_system(
            system, memory_budget_bytes=16 * 1024 * 1024
        )
