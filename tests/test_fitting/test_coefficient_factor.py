"""Tagged coefficient-factor orientation and matrix-RHS contracts."""

import pickle

import jax
import numpy as np
import pytest

from jaxgam.fitting.state import (
    CholeskyCoefficientFactor,
    PivotedQRCoefficientFactor,
    StreamFitState,
)
from jaxgam.inference.predictor import PivotedQRFisherFactor
from tests.tolerances import STRICT


def test_cholesky_factor_actions_follow_b_transpose_convention() -> None:
    lower = np.array([[2.0, 0.0, 0.0], [0.4, 3.0, 0.0], [-0.3, 0.5, 1.5]])
    factor = CholeskyCoefficientFactor(lower, 3)
    rhs = np.array([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]])
    B = lower.T
    H = B.T @ B
    np.testing.assert_allclose(
        factor.root_inverse(rhs),
        np.linalg.solve(B, rhs),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        factor.root_transpose_inverse(rhs),
        np.linalg.solve(B.T, rhs),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        factor.hessian_inverse(rhs),
        np.linalg.solve(H, rhs),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        factor.logdet_hessian(),
        np.linalg.slogdet(H)[1],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert jax.jit(factor.hessian_inverse)(rhs).shape == rhs.shape


def test_pivoted_qr_factor_actions_embed_project_matrix_rhs() -> None:
    R = np.array([[3.0, -1.0, 0.4], [0.0, 2.0, 0.7], [0.0, 0.0, 1.5]])
    pivots = np.array([2, 0, 1])
    keep = np.array([0, 2, 3])
    factor = PivotedQRCoefficientFactor(R, pivots, keep, 4)
    P = np.eye(3)[:, pivots]
    B = R @ P.T
    H_reduced = B.T @ B
    row_rhs = np.array([[1.0, 2.0], [-0.5, 1.5], [3.0, -1.0]])
    original_rhs = np.array([[1.0, -2.0], [7.0, 4.0], [-0.5, 1.5], [3.0, -1.0]])

    expected_root = np.zeros_like(original_rhs)
    expected_root[keep] = np.linalg.solve(B, row_rhs)
    np.testing.assert_allclose(
        factor.root_inverse(row_rhs), expected_root, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        factor.root_transpose_inverse(original_rhs),
        np.linalg.solve(B.T, original_rhs[keep]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    expected_hessian = np.zeros_like(original_rhs)
    expected_hessian[keep] = np.linalg.solve(H_reduced, original_rhs[keep])
    np.testing.assert_allclose(
        factor.hessian_inverse(original_rhs),
        expected_hessian,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        factor.logdet_hessian(),
        np.linalg.slogdet(H_reduced)[1],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert jax.jit(factor.hessian_inverse)(original_rhs).shape == original_rhs.shape
    state = StreamFitState(
        coefficients=np.zeros(4),
        log_lambda=np.zeros(0),
        deviance=np.array(0.0),
        penalized_deviance=np.array(0.0),
        scale=np.array(1.0),
        score_scale=np.array(1.0),
        saturated_loglik=np.array(0.0),
        edf=np.array(0.0),
        xtwx=np.zeros((4, 4)),
        xtwx_fisher=np.zeros((4, 4)),
        coefficient_factor=factor,
        fisher_coefficient_factor=factor,
        n_iter=1,
        converged=True,
        line_search_failed=False,
        backtracks=0,
        stationarity=0.0,
        source_scans=1,
        batches_scanned=1,
    )
    with np.testing.assert_raises(RuntimeError):
        _ = state.factor
    with np.testing.assert_raises(RuntimeError):
        _ = state.fisher_factor


def test_cpu_qr_fisher_factor_is_owned_picklable_and_action_correct() -> None:
    R = np.array([[3.0, -1.0, 0.4], [0.0, 2.0, 0.7], [0.0, 0.0, 1.5]])
    pivots = np.array([2, 0, 1])
    keep = np.array([0, 2, 3])
    factor = PivotedQRFisherFactor(R, pivots, keep, 4)
    R[:, :] = 99.0
    pivots[:] = 0
    keep[:] = 0
    P = np.eye(3)[:, np.array([2, 0, 1])]
    B = np.array([[3.0, -1.0, 0.4], [0.0, 2.0, 0.7], [0.0, 0.0, 1.5]]) @ P.T
    rhs = np.array([[1.0, -2.0], [7.0, 4.0], [-0.5, 1.5], [3.0, -1.0]])
    np.testing.assert_allclose(
        factor.root_transpose_inverse(rhs),
        np.linalg.solve(B.T, rhs[[0, 2, 3]]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    expected = np.zeros_like(rhs)
    expected[[0, 2, 3]] = np.linalg.solve(B.T @ B, rhs[[0, 2, 3]])
    np.testing.assert_allclose(
        factor.hessian_inverse(rhs), expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    with np.testing.assert_raises(ValueError):
        factor.R[0, 0] = 0.0
    restored = pickle.loads(pickle.dumps(factor))
    np.testing.assert_allclose(
        restored.hessian_inverse(rhs), expected, rtol=STRICT.rtol
    )
    assert not restored.R.flags.writeable
    assert not restored.pivots.flags.writeable
    assert not restored.keep.flags.writeable
    with np.testing.assert_raises(ValueError):
        restored.R[0, 0] = 0.0


@pytest.mark.parametrize(
    ("R", "pivots", "keep", "n_coef"),
    [
        (np.ones((2, 3)), np.array([0, 1]), np.array([0, 1]), 2),
        (
            np.array([[1.0, 0.0], [0.2, 1.0]]),
            np.array([0, 1]),
            np.array([0, 1]),
            2,
        ),
        (np.diag([1.0, 0.0]), np.array([0, 1]), np.array([0, 1]), 2),
        (np.empty((0, 0)), np.array([], dtype=int), np.array([], dtype=int), 0),
    ],
)
def test_cpu_qr_fisher_factor_validates_root_layout(
    R: np.ndarray, pivots: np.ndarray, keep: np.ndarray, n_coef: int
) -> None:
    if R.shape == (0, 0):
        factor = PivotedQRFisherFactor(R, pivots, keep, n_coef)
        assert factor.n_coef == 0
    else:
        with np.testing.assert_raises(ValueError):
            PivotedQRFisherFactor(R, pivots, keep, n_coef)
