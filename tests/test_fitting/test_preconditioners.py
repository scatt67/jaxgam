"""Budgeted SPD diagonal, small-block and coarse inverse actions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.fitting.preconditioners import apply_preconditioner, prepare_preconditioner
from tests.tolerances import STRICT


def test_inverse_blocks_coarse_action_and_owned_inputs():
    diagonal = np.arange(1, 33, dtype=float)
    cols = np.array([3, 0, 7], dtype=np.int64)
    block = np.array([[3.0, 0.2, 0.4], [0.2, 2.0, 0.3], [0.4, 0.3, 4.0]])
    coarse_cols = np.array([0, 3, 7, 10], dtype=np.int64)
    Q = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    coarse_matrix = Q.T @ np.diag(diagonal[coarse_cols]) @ Q
    prepared = prepare_preconditioner(
        diagonal,
        blocks=((cols, block),),
        coarse=(coarse_cols, Q, coarse_matrix),
        memory_budget_bytes=100000,
    )
    v = np.sin(np.arange(32))
    expected = v / diagonal
    expected[cols] = np.linalg.solve(block, v[cols])
    expected[coarse_cols] += Q @ np.linalg.solve(coarse_matrix, Q.T @ v[coarse_cols])
    diagonal[:] = 99
    cols[:] = 20
    block[:] = 0
    Q[:] = 0
    coarse_matrix[:] = 0
    actual = jax.jit(apply_preconditioner)(prepared.inverse, jnp.asarray(v))
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    u = jnp.cos(jnp.arange(32))
    np.testing.assert_allclose(
        jnp.vdot(u, apply_preconditioner(prepared.inverse, jnp.asarray(v))),
        jnp.vdot(apply_preconditioner(prepared.inverse, u), v),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert jnp.vdot(u, apply_preconditioner(prepared.inverse, u)) > 0
    assert (
        prepared.retained_device_bytes
        < prepared.construction_peak_bound_bytes
        <= 100000
    )
    with pytest.raises(ValueError, match="dimension"):
        apply_preconditioner(prepared.inverse, jnp.ones(31))


def test_large_diagonal_budget_is_checked_before_transfer_or_factor(monkeypatch):
    diagonal = np.ones(10000)
    monkeypatch.setattr(
        jax, "device_put", lambda *_a, **_k: pytest.fail("premature transfer")
    )
    monkeypatch.setattr(
        np.linalg, "cholesky", lambda *_a, **_k: pytest.fail("premature factor")
    )
    with pytest.raises(MemoryError, match="construction"):
        prepare_preconditioner(diagonal, memory_budget_bytes=170000 - 1)
    with pytest.raises(MemoryError, match="hard modest"):
        prepare_preconditioner(
            diagonal,
            blocks=((np.arange(10000), np.broadcast_to(1.0, (10000, 10000))),),
            memory_budget_bytes=10**12,
        )
    with pytest.raises(MemoryError, match="hard cap"):
        prepare_preconditioner(
            diagonal,
            coarse=(
                np.arange(10000),
                np.broadcast_to(1.0, (10000, 1)),
                np.ones((1, 1)),
            ),
            memory_budget_bytes=10**12,
        )


@pytest.mark.parametrize(
    "diagonal",
    [
        np.ones(3, dtype=np.float32),
        np.ones((3, 1)),
        np.zeros(3),
        np.array([1.0, np.nan]),
        np.array([]),
    ],
)
def test_invalid_diagonal(diagonal):
    with pytest.raises(ValueError, match="diagonal"):
        prepare_preconditioner(diagonal, memory_budget_bytes=100000)


@pytest.mark.parametrize(
    ("columns", "matrix"),
    [
        (np.array([0.0, 1.0]), np.eye(2)),
        (np.array([0, 0]), np.eye(2)),
        (np.array([-1, 0]), np.eye(2)),
        (np.array([0, 3]), np.eye(2)),
        (np.array([0, 1]), np.array([[1.0, 2.0], [0.0, 1.0]])),
        (np.array([0, 1]), np.array([[1.0, np.nan], [np.nan, 1.0]])),
        (np.array([0, 1]), np.eye(2, dtype=np.float32)),
        (np.array([0, 1]), np.zeros((2, 2))),
        (np.array([], dtype=np.int64), np.empty((0, 0))),
    ],
)
def test_invalid_small_blocks(columns, matrix):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        prepare_preconditioner(
            np.ones(3), blocks=((columns, matrix),), memory_budget_bytes=100000
        )


def test_overlapping_blocks_invalid_controls_and_coarse():
    with pytest.raises(ValueError, match="overlap"):
        prepare_preconditioner(
            np.ones(3),
            blocks=((np.array([0, 1]), np.eye(2)), (np.array([1, 2]), np.eye(2))),
            memory_budget_bytes=100000,
        )
    for options in (
        {"memory_budget_bytes": False},
        {"memory_budget_bytes": 0},
        {"memory_budget_bytes": 100000, "max_modest_coefficients": False},
    ):
        with pytest.raises(ValueError, match="positive integer"):
            prepare_preconditioner(np.ones(3), **options)
    for directions, matrix in (
        (np.ones(2), np.eye(1)),
        (np.ones((2, 1), dtype=np.float32), np.eye(1)),
        (np.array([[np.nan], [1.0]]), np.eye(1)),
    ):
        with pytest.raises(ValueError, match="directions"):
            prepare_preconditioner(
                np.ones(3),
                coarse=(np.array([0, 1]), directions, matrix),
                memory_budget_bytes=100000,
            )


def test_diagonal_storage_does_not_create_global_matrix():
    p = 20000
    prepared = prepare_preconditioner(np.linspace(1, 2, p), memory_budget_bytes=17 * p)
    assert prepared.retained_device_bytes == 8 * p
    np.testing.assert_allclose(
        apply_preconditioner(prepared.inverse, jnp.ones(p)),
        1 / np.linspace(1, 2, p),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    compiled = (
        jax.jit(apply_preconditioner).lower(prepared.inverse, jnp.ones(p)).compile()
    )
    sizes = compiled.memory_analysis()
    assert sizes.temp_size_in_bytes < 32 * p
