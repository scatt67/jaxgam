"""Local operator penalty pseudo-determinants across exact supported blocks."""

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator.penalty_determinant import (
    LocalOperatorPenaltyDeterminant,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator.prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    DiagonalPenalty,
    IdentityPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
from tests.tolerances import STRICT


def _compare_model(formula: str, frame: pd.DataFrame, sp: np.ndarray) -> None:
    spec = parse_formula(formula)
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=7)
    )
    rho = np.log(sp)
    actual = LocalOperatorPenaltyDeterminant(
        model.penalties, memory_budget_bytes=2 * 1024 * 1024
    ).evaluate(rho)
    dense = ModelSetup.build(spec, frame)
    eigenvalues = np.linalg.eigvalsh(dense.penalties.materialize(rho))
    positive = eigenvalues > (
        max(np.max(eigenvalues), 1.0) * np.finfo(float).eps ** (2.0 / 3.0)
    )
    expected = float(np.sum(np.log(eigenvalues[positive])))
    np.testing.assert_allclose(
        actual.value, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert actual.rank == int(np.count_nonzero(positive))
    assert actual.null_dimension == model.n_coef - actual.rank
    assert not actual.block_values.flags.writeable
    assert actual.derivative is not None
    weighted = dense.penalties.materialize(rho)
    inverse = np.linalg.pinv(weighted, hermitian=True)
    expected_derivative = np.zeros(model.penalties.n_penalties)
    for block in model.penalties.blocks:
        local_inverse = inverse[block.start : block.stop, block.start : block.stop]
        for sp_index, penalty in zip(
            block.sp_indices, block.local_penalties, strict=True
        ):
            expected_derivative[sp_index] += np.exp(rho[sp_index]) * np.trace(
                local_inverse @ penalty.dense()
            )
    np.testing.assert_allclose(
        actual.derivative, expected_derivative, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert not actual.derivative.flags.writeable


@pytest.mark.parametrize(
    ("formula", "sp"),
    [
        ("y~s(x,bs='cr',k=6)+s(g,bs='re')", np.array([0.7, 1.1])),
        ("y~te(x,z,bs='cr',k=4)", np.array([0.5, 1.3])),
    ],
)
def test_local_penalty_determinant_matches_direct_supported_blocks(formula, sp):
    row = np.arange(47)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.1 * np.sin(row),
            "x": np.linspace(0.1, 1.5, len(row)),
            "z": ((17 * row) % len(row)) / len(row),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    _compare_model(formula, frame, sp)


def test_large_indexed_identity_needs_no_coefficient_square(monkeypatch):
    row = np.arange(97)
    frame = pd.DataFrame(
        {
            "y": 1.0 + row / 97,
            "g": pd.Categorical(
                [f"g{i}" for i in row % 3],
                categories=[f"g{i}" for i in range(1024)],
            ),
            "h": pd.Categorical(
                [f"h{i}" for i in row % 4],
                categories=[f"h{i}" for i in range(1024)],
            ),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(g,bs='re')+s(h,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=13),
    )
    assert model.n_coef == 2049
    monkeypatch.setattr(
        PenaltyStructure,
        "materialize",
        lambda *_a, **_k: pytest.fail("Global penalty materialized"),
    )
    actual = LocalOperatorPenaltyDeterminant(
        model.penalties, memory_budget_bytes=2 * 1024 * 1024
    ).evaluate(np.log(np.array([0.8, 1.1])))
    expected = sum(
        block.size * (np.log(sp) + np.log(block.local_penalties[0].scale))
        for block, sp in zip(model.penalties.blocks, [0.8, 1.1], strict=True)
    )
    np.testing.assert_allclose(
        actual.value, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert actual.rank == 2048
    assert actual.null_dimension == 1
    np.testing.assert_array_equal(actual.derivative, [1024.0, 1024.0])
    assert actual.peak_bound_bytes < 2 * 1024 * 1024
    assert 8 * model.n_coef**2 > 2 * 1024 * 1024


def test_diagonal_penalty_range_and_budget_preflight():
    structure = PenaltyStructure(
        n_coef=4,
        blocks=(
            PenaltyBlock(
                0,
                4,
                (0, 1),
                (
                    DiagonalPenalty(np.array([1.0, 0.0, 3.0, 0.0])),
                    DiagonalPenalty(np.array([0.0, 2.0, 1.0, 0.0])),
                ),
                IdentityTransform(4),
                (2, 2),
            ),
        ),
    )
    evaluator = LocalOperatorPenaltyDeterminant(structure, memory_budget_bytes=100_000)
    result = evaluator.evaluate(np.log(np.array([0.7, 1.2])))
    expected = np.log(0.7) + np.log(2.4) + np.log(3 * 0.7 + 1.2)
    np.testing.assert_allclose(
        result.value, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert result.rank == 3
    assert result.null_dimension == 1
    np.testing.assert_allclose(
        result.derivative,
        [1.0 + 2.1 / 3.3, 1.0 + 1.2 / 3.3],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    with pytest.raises(MemoryError, match="requires"):
        LocalOperatorPenaltyDeterminant(structure, memory_budget_bytes=1).evaluate(
            np.zeros(2, dtype=np.float64)
        )
    with pytest.raises(ValueError, match="finite float64"):
        evaluator.evaluate(np.array([0.0, np.inf]))


def test_dense_local_cap_rejects_before_eigenwork():
    structure = PenaltyStructure(
        n_coef=65,
        blocks=(
            PenaltyBlock(
                0,
                65,
                (0,),
                (DenseLocalPenalty(np.eye(65)),),
                IdentityTransform(65),
                (65,),
            ),
        ),
    )
    with pytest.raises(MemoryError, match="hard size cap"):
        LocalOperatorPenaltyDeterminant(
            structure, memory_budget_bytes=8 * 1024 * 1024
        ).evaluate(np.zeros(1, dtype=np.float64))


def test_zero_and_invalid_local_penalties_do_not_produce_fictitious_logdet():
    zero = PenaltyStructure(
        n_coef=2,
        blocks=(
            PenaltyBlock(
                0, 2, (0,), (IdentityPenalty(2, 0.0),), IdentityTransform(2), (0,)
            ),
        ),
    )
    result = LocalOperatorPenaltyDeterminant(
        zero, memory_budget_bytes=100_000
    ).evaluate(np.zeros(1, dtype=np.float64))
    assert result.value == 0.0
    assert result.rank == 0
    assert result.null_dimension == 2
    np.testing.assert_array_equal(result.derivative, [0.0])
    negative = PenaltyStructure(
        n_coef=2,
        blocks=(
            PenaltyBlock(
                0,
                2,
                (0,),
                (IdentityPenalty(2, -1.0),),
                IdentityTransform(2),
                (2,),
            ),
        ),
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        LocalOperatorPenaltyDeterminant(negative, memory_budget_bytes=100_000).evaluate(
            np.zeros(1, dtype=np.float64)
        )


def test_singleton_identity_uses_dense_efs_log_eigenvalue_floor():
    structure = PenaltyStructure(
        n_coef=3,
        blocks=(
            PenaltyBlock(
                0,
                3,
                (0,),
                (IdentityPenalty(3, 1e-40),),
                IdentityTransform(3),
                (3,),
            ),
        ),
    )
    actual = LocalOperatorPenaltyDeterminant(
        structure, memory_budget_bytes=100_000
    ).evaluate(np.array([np.log(0.7)]))
    np.testing.assert_allclose(
        actual.value,
        3 * (np.log(0.7) + np.log(1e-30)),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_array_equal(actual.derivative, [3.0])
    with pytest.raises(FloatingPointError, match="nonfinite"):
        LocalOperatorPenaltyDeterminant(
            structure, memory_budget_bytes=100_000
        ).evaluate(np.array([1e308]))


def test_dense_multi_penalty_preflight_charges_each_live_local_matrix():
    penalties = tuple(DenseLocalPenalty(np.eye(64)) for _ in range(20))
    structure = PenaltyStructure(
        n_coef=64,
        blocks=(
            PenaltyBlock(
                0,
                64,
                tuple(range(20)),
                penalties,
                IdentityTransform(64),
                (64,) * 20,
            ),
        ),
    )
    evaluator = LocalOperatorPenaltyDeterminant(
        structure, memory_budget_bytes=1_500_000
    )
    with pytest.raises(MemoryError, match="requires"):
        evaluator.evaluate(np.zeros(20, dtype=np.float64))


def test_coupled_local_derivative_matches_pinned_mgcv_algebra(r_bridge):
    first = np.array([[2.0, 0.4], [0.4, 1.0]])
    second = np.array([[1.0, -0.3], [-0.3, 3.0]])
    structure = PenaltyStructure(
        n_coef=3,
        blocks=(
            PenaltyBlock(
                1,
                3,
                (0, 1),
                (DenseLocalPenalty(first), DenseLocalPenalty(second)),
                IdentityTransform(2),
                (2, 2),
            ),
        ),
    )
    rho = np.array([0.3, -0.5], dtype=np.float64)
    actual = LocalOperatorPenaltyDeterminant(
        structure, memory_budget_bytes=100_000
    ).evaluate(rho)
    roots = [np.linalg.cholesky(first), np.linalg.cholesky(second)]
    oracle = r_bridge.efs_statistics_algebra(
        rho,
        roots,
        [np.vstack((np.zeros((1, 2)), root)) for root in roots],
        np.array([0.2, -0.8, 1.1]),
        np.linalg.cholesky(np.eye(3) * 4),
    )
    np.testing.assert_allclose(
        actual.derivative, oracle["d"], rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert actual.null_dimension == 1


@pytest.mark.parametrize(
    "penalties",
    [
        (IdentityPenalty(2, 1e308), IdentityPenalty(2, 1e308)),
        (
            DenseLocalPenalty(np.eye(2) * 1e308),
            DenseLocalPenalty(np.eye(2) * 1e308),
        ),
        (IdentityPenalty(2, 1e-308), IdentityPenalty(2, 1e-308)),
        (
            DenseLocalPenalty(np.eye(2) * 1e-308),
            DenseLocalPenalty(np.eye(2) * 1e-308),
        ),
    ],
)
def test_coupled_local_range_remains_active_across_finite_scales(penalties):
    structure = PenaltyStructure(
        n_coef=2,
        blocks=(PenaltyBlock(0, 2, (0, 1), penalties, IdentityTransform(2), (2, 2)),),
    )
    result = LocalOperatorPenaltyDeterminant(
        structure, memory_budget_bytes=100_000
    ).evaluate(np.zeros(2, dtype=np.float64))
    expected = 2 * (np.log(penalties[0].dense()[0, 0]) + np.log(2.0))
    np.testing.assert_allclose(
        result.value, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert result.rank == 2
    assert result.null_dimension == 0
    np.testing.assert_allclose(
        result.derivative, [1.0, 1.0], rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_finite_block_determinants_cannot_overflow_total_score_component():
    structure = PenaltyStructure(
        n_coef=2,
        blocks=tuple(
            PenaltyBlock(
                index,
                index + 1,
                (index,),
                (IdentityPenalty(1),),
                IdentityTransform(1),
                (1,),
            )
            for index in range(2)
        ),
    )
    with pytest.raises(FloatingPointError, match="Total"):
        LocalOperatorPenaltyDeterminant(
            structure, memory_budget_bytes=100_000
        ).evaluate(np.array([1e308, 1e308]))
