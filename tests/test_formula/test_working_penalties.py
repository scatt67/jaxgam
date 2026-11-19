"""Owning CPU Sl.setup initial-transform and mroot source gates."""

from __future__ import annotations

import numpy as np
import pytest

from jaxgam.formula.working_penalties import (
    _source_project_penalty,
    build_working_penalty_blocks,
)
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
from jaxgam.smooths.source_lapack import scipy_uses_accelerate_lapack
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


def _structure(penalties, ranks):
    size = len(penalties[0])
    return PenaltyStructure(
        size,
        (
            PenaltyBlock(
                0,
                size,
                tuple(range(len(penalties))),
                tuple(DenseLocalPenalty(S) for S in penalties),
                IdentityTransform(size),
                tuple(ranks),
            ),
        ),
    )


@pytest.mark.parametrize("kind", ["diagonal", "dense", "coupled"])
def test_initial_working_penalty_transform_matches_pinned_sl_setup(r_bridge, kind):
    assert r_bridge.check_versions()[0]
    vectors, _ = np.linalg.qr(np.random.default_rng(71423).normal(size=(6, 6)))
    if kind == "diagonal":
        penalties = [np.diag([0.0, 1.0, 2.0, 0.0, 4.0, 8.0])]
        ranks = [4]
    elif kind == "dense":
        penalties = [
            (vectors[:, :4] * np.array([1.0, 2.0, 4.0, 8.0])) @ vectors[:, :4].T
        ]
        ranks = [4]
    else:
        penalties = [
            (vectors[:, :3] * np.array([1.0, 2.0, 4.0])) @ vectors[:, :3].T,
            (vectors[:, 2:5] * np.array([3.0, 5.0, 7.0])) @ vectors[:, 2:5].T,
        ]
        ranks = [3, 3]
    expected = r_bridge.source_working_penalty_setup(tuple(penalties), tuple(ranks))
    (observed,) = build_working_penalty_blocks(_structure(penalties, ranks))
    collector = _AssertCollector()
    assert observed.rank == expected[1]
    # Sl.setup's private eigenvector signs and degenerate null basis are
    # nonunique. Compare their coordinate-invariant operators at STRICT;
    # actual public proposal/full-fit parity remains a separate hard gate.
    inverse = np.linalg.inv(observed.transform)
    expected_inverse = np.linalg.inv(expected[0])
    comparisons = [
        (
            "initial transform covariance",
            observed.transform @ observed.transform.T,
            expected[0] @ expected[0].T,
        )
    ]
    for i, (actual, target) in enumerate(
        zip(observed.penalties, expected[2], strict=True)
    ):
        comparisons.append(
            (
                f"public penalty{i}",
                inverse.T @ actual @ inverse,
                expected_inverse.T @ target @ expected_inverse,
            )
        )
    for i, (actual, target) in enumerate(zip(observed.roots, expected[3], strict=True)):
        actual_root = observed.transform[:, : observed.rank] @ actual
        target_root = expected[0][:, : expected[1]] @ target
        comparisons.append(
            (
                f"public root operator{i}",
                actual_root @ actual_root.T,
                target_root @ target_root.T,
            )
        )
    for name, actual, target in comparisons:
        collector.check(
            name,
            lambda actual=actual, target=target: np.testing.assert_allclose(
                actual, target, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any(kind)


def test_empty_and_zero_working_penalty_blocks():
    assert build_working_penalty_blocks(PenaltyStructure(3, ())) == ()
    (observed,) = build_working_penalty_blocks(_structure([np.zeros((3, 3))], [0]))
    assert observed.rank == 0
    np.testing.assert_array_equal(observed.transform, np.eye(3))
    np.testing.assert_array_equal(observed.penalties[0], np.zeros((3, 3)))


def test_source_ordered_coupled_projection_matches_pinned_sl_setup(r_bridge):
    assert r_bridge.check_versions()[0]
    vectors, _ = np.linalg.qr(np.random.default_rng(71423).normal(size=(6, 6)))
    penalties = (
        (vectors[:, :3] * np.array([1.0, 2.0, 4.0])) @ vectors[:, :3].T,
        (vectors[:, 2:5] * np.array([3.0, 5.0, 7.0])) @ vectors[:, 2:5].T,
    )
    transform, rank, projected, _ = r_bridge.source_working_penalty_setup(
        penalties, (3, 3)
    )
    basis = np.array(transform[:, :rank], copy=True)
    for source, target in zip(penalties, projected, strict=True):
        actual = _source_project_penalty(basis, source)
        expected = np.array(target[:rank, :rank], copy=True)
        if scipy_uses_accelerate_lapack():
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            )


def test_nonpositive_structural_working_penalty_rejects():
    S = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(FloatingPointError, match="range"):
        build_working_penalty_blocks(_structure([S], [2]))
