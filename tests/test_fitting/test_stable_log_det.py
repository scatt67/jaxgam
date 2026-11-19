"""Host get_stableS terms in the dense REML criterion's log|S|+."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.fitting.stable_log_det import (
    criterion_multi_block_inputs,
    stable_block_roots,
    stable_log_det_terms,
)
from jaxgam.jax_utils import StableLogDetTerms, block_log_det_S
from jaxgam.penalties.stable_reparam import stable_reparam
from tests.tolerances import STRICT


def _difference(k: int, order: int) -> np.ndarray:
    matrix = np.eye(k)
    for _ in range(order):
        matrix = matrix[1:] - matrix[:-1]
    return matrix


def _te_block() -> tuple[np.ndarray, np.ndarray]:
    """Two Kronecker penalties of a 4x3 te block in their total range basis."""
    first = np.kron(_difference(4, 2).T @ _difference(4, 2), np.eye(3))
    second = np.kron(np.eye(4), _difference(3, 1).T @ _difference(3, 1))
    total = first / np.linalg.norm(first) + second / np.linalg.norm(second)
    values, vectors = np.linalg.eigh(total)
    basis = vectors[:, values > values[-1] * 1e-10]
    return basis.T @ first @ basis, basis.T @ second @ basis


def _close(actual, expected) -> None:
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)


@pytest.mark.parametrize("rho", [(0.0, 0.0), (20.0, -3.0), (-6.0, 40.0)])
def test_jitted_log_det_value_gradient_and_hessian_are_get_stables(rho) -> None:
    """Under jax.jit the term's value, grad and Hessian are det, det1, det2."""
    projections = _te_block()
    blocks = stable_block_roots(((0, 1),), (projections,))
    rho = np.asarray(rho)
    terms = stable_log_det_terms(blocks, rho, deriv=2)
    expected = stable_reparam(blocks[0].roots, rho, deriv=2)

    def log_det(log_lambda, multi):
        return block_log_det_S(log_lambda, (), (), jnp.zeros(0), ((0, 1),), (0,), multi)

    value = jax.jit(log_det)(jnp.asarray(rho), terms)
    gradient = jax.jit(jax.grad(log_det))(jnp.asarray(rho), terms)
    hessian = jax.jit(jax.hessian(log_det))(jnp.asarray(rho), terms)
    _close(value, expected.det)
    _close(gradient, expected.det1)
    _close(hessian, expected.det2)
    off_point = jax.jit(log_det)(jnp.asarray(rho + np.array([1e-9, 0.0])), terms)
    assert np.isnan(off_point)


def test_terms_handle_zero_penalties_and_score_changes() -> None:
    first, second = _te_block()
    zero = np.zeros_like(first)
    blocks = stable_block_roots(((0, 1, 2),), ((first, zero, second),))
    assert blocks[0].active == (0, 2)
    rho = np.array([3.0, 1.0, -2.0])
    (terms,) = stable_log_det_terms(blocks, rho, deriv=2)
    expected = stable_reparam(blocks[0].roots, rho[[0, 2]], deriv=2)
    _close(terms.det, expected.det)
    _close(np.asarray(terms.det1), [expected.det1[0], 0.0, expected.det1[1]])
    assert np.all(np.asarray(terms.det2)[1] == 0.0)
    assert float(terms.det_change) == 0.0
    base = np.array([2.0, 1.0, -1.0])
    (change,) = stable_log_det_terms(blocks, rho, deriv=0, base_log_lambda=base)
    base_det = stable_reparam(blocks[0].roots, base[[0, 2]], deriv=0).det
    _close(change.det_change, expected.det - base_det)
    empty = stable_block_roots(((0, 1),), ((zero, zero),))
    (nothing,) = stable_log_det_terms(empty, np.zeros(2), deriv=2)
    assert float(nothing.det) == 0.0


def test_criterion_inputs_pass_through_without_multi_blocks() -> None:
    assert criterion_multi_block_inputs((), (), np.zeros(1), deriv=2) == ()
    (term,) = criterion_multi_block_inputs(
        ((0, 1),), (_te_block(),), np.array([1.0, 2.0]), deriv=0
    )
    assert isinstance(term, StableLogDetTerms)
