"""CPU-only fitting-preparation helpers."""

from __future__ import annotations

import numpy as np

from jaxgam.formula.fitting_prepare import (
    ResponseReduction,
    apply_transforms_to_design,
    initial_log_sp_from_diagonal,
    penalties_non_overlapping,
    qr_penalty_roots,
    reparameterize_structure,
)
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
from tests.tolerances import STRICT


def _structure(*penalties: np.ndarray) -> PenaltyStructure:
    return PenaltyStructure(
        3,
        (
            PenaltyBlock(
                0,
                3,
                tuple(range(len(penalties))),
                tuple(DenseLocalPenalty(S) for S in penalties),
                IdentityTransform(3),
                tuple(np.linalg.matrix_rank(S) for S in penalties),
            ),
        ),
    )


def test_initial_sp_and_singleton_transform_are_cpu_local() -> None:
    S = np.diag([0.0, 2.0, 8.0])
    transformed = reparameterize_structure(_structure(S))
    X = np.arange(12.0).reshape(4, 3)
    X_fit = apply_transforms_to_design(X, transformed)
    np.testing.assert_allclose(
        X_fit,
        X @ transformed.blocks[0].transform.dense(),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    log_sp = initial_log_sp_from_diagonal(np.array([5.0, 8.0, 13.0]), _structure(S))
    assert log_sp.shape == (1,)
    assert np.isfinite(log_sp[0])


def test_disjoint_intervals_take_separate_local_rotations() -> None:
    left = np.diag([2.0, 0.0, 0.0])
    right = np.diag([0.0, 0.0, 3.0])
    assert penalties_non_overlapping([left, right])
    transformed = reparameterize_structure(_structure(left, right))
    assert transformed.blocks[0].transform.size == 3


def test_interleaved_supports_remain_coupled_and_zero_penalty_is_safe() -> None:
    interleaved_a = np.diag([1.0, 0.0, 1.0])
    interleaved_b = np.diag([0.0, 2.0, 0.0])
    assert not penalties_non_overlapping([interleaved_a, interleaved_b])
    transformed = reparameterize_structure(_structure(interleaved_a, interleaved_b))
    assert transformed.blocks[0].transform.size == 3
    zero = _structure(np.zeros((3, 3)))
    np.testing.assert_array_equal(initial_log_sp_from_diagonal(np.ones(3), zero), [0.0])


def test_response_reduction_weighted_mean() -> None:
    reduction = ResponseReduction(3, 4.0, 1.0, 6.0, 10.0, -1.0, 2.0)
    assert reduction.weighted_mean == 2.5


def test_qr_penalty_roots_are_local_psd_owned_and_keep_zero_layout() -> None:
    left = np.diag([4.0, 0.0, 1.0])
    zero = np.zeros((3, 3))
    roots = qr_penalty_roots(_structure(left, zero))
    assert [(root.start, root.stop, root.sp_index) for root in roots] == [
        (0, 3, 0),
        (0, 3, 1),
    ]
    np.testing.assert_allclose(roots[0].root.T @ roots[0].root, left)
    assert roots[1].root.shape == (0, 3)
    with np.testing.assert_raises(ValueError):
        roots[0].root[0, 0] = 0.0
    with np.testing.assert_raises(ValueError):
        qr_penalty_roots(_structure(np.diag([1.0, -0.5, 0.0])))


def test_qr_roots_honor_supplied_rank_without_global_dimension_cutoff() -> None:
    matrix = np.diag([1.0, 1e-14])
    structure = PenaltyStructure(
        101,
        (
            PenaltyBlock(
                0,
                2,
                (0,),
                (DenseLocalPenalty(matrix),),
                IdentityTransform(2),
                (2,),
            ),
        ),
    )
    root = qr_penalty_roots(structure)[0]
    assert root.root.shape == (2, 2)
    np.testing.assert_allclose(
        root.root.T @ root.root,
        matrix,
        rtol=STRICT.rtol,
        atol=100.0 * np.finfo(float).eps,
    )


def test_qr_roots_honor_rotated_declared_rank() -> None:
    """A small declared positive eigendirection is not discarded by global p."""
    rotation, _ = np.linalg.qr(
        np.array([[1.0, -2.0, 0.5], [2.0, 1.0, -1.0], [0.0, 1.0, 2.0]])
    )
    matrix = rotation @ np.diag([3.0, 1e-14, 0.0]) @ rotation.T
    structure = PenaltyStructure(
        137,
        (
            PenaltyBlock(
                4,
                7,
                (0,),
                (DenseLocalPenalty(matrix),),
                IdentityTransform(3),
                (2,),
            ),
        ),
    )
    root = qr_penalty_roots(structure)[0]
    assert root.root.shape == (2, 3)
    np.testing.assert_allclose(
        root.root.T @ root.root,
        matrix,
        rtol=STRICT.rtol,
        atol=100.0 * np.finfo(float).eps,
    )
