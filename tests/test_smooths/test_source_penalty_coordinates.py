"""CPU-only source coordinates retained for Gaussian penalty determinants."""

import numpy as np
import pytest

from jaxgam.smooths.source_penalty_coordinates import (
    source_centered_penalties,
    source_symmetric_eigh,
)
from tests.tolerances import STRICT


def test_source_centering_preserves_the_constrained_quadratic_form() -> None:
    design = np.array(
        [
            [1.0, 0.0, 2.0, -1.0],
            [1.0, 1.0, -1.0, 0.0],
            [1.0, 2.0, 1.0, 1.0],
            [1.0, 3.0, -2.0, 2.0],
        ]
    )
    diagonal = np.diag([0.0, 2.0, 3.0, 5.0])
    coupled = np.array(
        [
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    actual = source_centered_penalties(design, (diagonal, coupled))

    mean = design.mean(axis=0)
    norm = np.copysign(np.linalg.norm(mean), mean[0])
    reflector = mean / norm
    reflector[0] += 1.0
    Q = np.eye(len(mean)) - np.outer(reflector, reflector) / reflector[0]
    for observed, source in zip(actual, (diagonal, coupled), strict=True):
        expected = Q[1:, :] @ source @ Q[1:, :].T
        np.testing.assert_allclose(
            observed, expected, rtol=STRICT.rtol, atol=STRICT.atol
        )
        assert observed.shape == (3, 3)


def test_source_eigensolver_reconstructs_full_symmetric_input() -> None:
    matrix = np.array([[2.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 3.0]])
    values, vectors = source_symmetric_eigh(matrix)
    assert np.all(np.diff(values) >= 0)
    np.testing.assert_allclose(
        vectors @ np.diag(values) @ vectors.T,
        matrix,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        vectors.T @ vectors,
        np.eye(3),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_source_coordinate_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty smooth design"):
        source_centered_penalties(np.zeros((0, 3)), (np.eye(3),))
    with pytest.raises(ValueError, match="zero length"):
        source_centered_penalties(np.zeros((2, 3)), (np.eye(3),))
    with pytest.raises(ValueError, match="float64 square"):
        source_symmetric_eigh(np.ones((3, 2)))
    with pytest.raises(ValueError, match="entries are invalid"):
        source_symmetric_eigh(np.diag([1.0, np.nan]))
