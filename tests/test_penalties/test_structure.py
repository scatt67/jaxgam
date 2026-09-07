"""CPU local penalty descriptor tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    DiagonalPenalty,
    IdentityPenalty,
    make_penalty_structure,
)


def _term(start: int, n_coefs: int, sp_indices: tuple[int, ...]):
    return SimpleNamespace(col_start=start, n_coefs=n_coefs, penalty_indices=sp_indices)


def test_additive_penalties_remain_local_and_materialize_explicitly() -> None:
    structure = make_penalty_structure(
        8,
        [_term(1, 3, (0,)), _term(4, 4, (1,))],
        [[np.diag([2.0, 3.0, 4.0])], [np.eye(4)]],
    )
    assert structure is not None
    assert [block.size for block in structure.blocks] == [3, 4]
    assert isinstance(structure.blocks[0].local_penalties[0], DiagonalPenalty)
    assert isinstance(structure.blocks[1].local_penalties[0], IdentityPenalty)
    materialized = structure.materialize(np.log([2.0, 5.0]))
    assert materialized.shape == (8, 8)
    np.testing.assert_allclose(materialized[1:4, 1:4], np.diag([4.0, 6.0, 8.0]))
    np.testing.assert_allclose(materialized[4:, 4:], 5.0 * np.eye(4))


def test_disjoint_factor_by_levels_split_but_interleaved_support_does_not() -> None:
    split = make_penalty_structure(
        7,
        [_term(1, 6, (0, 1))],
        [[np.diag([1.0, 1.0, 1.0, 0, 0, 0]), np.diag([0, 0, 0, 2.0, 2.0, 2.0])]],
    )
    assert split is not None
    assert [(block.start, block.stop) for block in split.blocks] == [(1, 4), (4, 7)]
    interleaved = make_penalty_structure(
        4,
        [_term(0, 4, (0, 1))],
        [[np.diag([1.0, 0, 1.0, 0]), np.diag([0, 2.0, 0, 2.0])]],
    )
    assert interleaved is not None
    assert len(interleaved.blocks) == 1
    assert len(interleaved.blocks[0].local_penalties) == 2


def test_zero_penalty_keeps_lambda_position_and_values_are_owned_readonly() -> None:
    values = np.eye(2)
    descriptor = DenseLocalPenalty(values)
    values[0, 0] = 4
    assert descriptor.matrix[0, 0] == 1
    assert not descriptor.matrix.flags.writeable
    structure = make_penalty_structure(
        4, [_term(1, 3, (0, 1))], [[np.zeros((3, 3)), np.eye(3)]]
    )
    assert structure is not None
    assert structure.n_penalties == 2
    assert structure.blocks[0].size == 0
    assert structure.blocks[0].sp_indices == (0,)
