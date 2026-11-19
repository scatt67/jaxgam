"""Coefficient-only positive QR reducer tests."""

import jax
import numpy as np

from jaxgam.execution.qr import qr_update, reduce_positive_qr, solve_augmented_qr
from jaxgam.fitting.qr import qr_hessian_inverse, qr_root_inverse
from tests.tolerances import STRICT


def test_qr_local_root_solve_and_tail_match_dense_least_squares() -> None:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(31, 4))
    z = rng.normal(size=31)
    root = np.array([[1.0, 0.0], [0.0, 2.0]])
    state = reduce_positive_qr(((X[:1], z[:1]), (X[1:7], z[1:7]), (X[7:], z[7:])))
    solved = solve_augmented_qr(state, ((slice(1, 3), root),))
    augmented = np.vstack((X, np.pad(root, ((0, 0), (1, 1)))))
    expected = np.linalg.lstsq(augmented, np.r_[z, np.zeros(len(root))], rcond=None)[0]
    np.testing.assert_allclose(
        solved.coefficients, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    residual = z - X @ solved.coefficients
    assert state.residual_tail >= 0.0
    assert np.isfinite(residual @ residual)


def test_pivoted_triangular_solve_jits_and_unpivots() -> None:
    R = np.array([[3.0, 1.0], [0.0, 2.0]])
    pivots = np.array([1, 0])
    rhs = np.array([4.0, 5.0])
    actual = qr_root_inverse(R, rhs, pivots)
    expected_pivot = np.linalg.solve(R, rhs)
    expected = np.empty(2)
    expected[pivots] = expected_pivot
    np.testing.assert_allclose(actual, expected)
    assert jax.jit(qr_root_inverse)(R, rhs, pivots).shape == (2,)
    P = np.eye(2)[:, pivots]
    H = P @ R.T @ R @ P.T
    np.testing.assert_allclose(
        qr_hessian_inverse(R, rhs, pivots), np.linalg.solve(H, rhs)
    )
    matrix_rhs = np.column_stack((rhs, 2 * rhs))
    np.testing.assert_allclose(
        qr_hessian_inverse(R, matrix_rhs, pivots), np.linalg.solve(H, matrix_rhs)
    )


def test_rank_deficient_data_is_identified_by_local_penalty() -> None:
    X = np.column_stack((np.ones(5), np.zeros(5)))
    z = np.arange(5.0)
    state = reduce_positive_qr(((X, z),))
    solved = solve_augmented_qr(state, ((slice(1, 2), np.array([[2.0]])),))
    expected = np.linalg.lstsq(np.vstack((X, [[0.0, 2.0]])), np.r_[z, 0.0], rcond=None)[
        0
    ]
    np.testing.assert_allclose(solved.coefficients, expected)


def test_joint_alias_requires_fixed_subspace_and_all_zero_data_can_be_penalized() -> (
    None
):
    state = reduce_positive_qr(
        ((np.column_stack((np.ones(4), np.zeros(4))), np.ones(4)),)
    )
    with np.testing.assert_raises(np.linalg.LinAlgError):
        solve_augmented_qr(state, ())
    solved = solve_augmented_qr(state, (), identifiable_subspace=[0])
    np.testing.assert_allclose(solved.coefficients, [1.0, 0.0])
    zero = reduce_positive_qr((), n_coef=2)
    penalized = solve_augmented_qr(zero, ((slice(0, 2), np.eye(2)),))
    np.testing.assert_allclose(penalized.coefficients, [0.0, 0.0])


def test_subspace_rank_indices_and_owned_factor_arrays() -> None:
    X = np.column_stack((np.ones(4), np.zeros(4), np.zeros(4)))
    z = np.ones(4)
    root = np.array([[1.0]])
    state = reduce_positive_qr(((X, z),))
    roots = ((slice(1, 2), root),)
    with np.testing.assert_raises(np.linalg.LinAlgError):
        solve_augmented_qr(state, roots, identifiable_subspace=[0, 2])
    for invalid in ([0.0, 1.0], [True, False], [0, 0], [0, 3]):
        with np.testing.assert_raises(ValueError):
            solve_augmented_qr(state, roots, identifiable_subspace=invalid)
    solved = solve_augmented_qr(state, roots, identifiable_subspace=[0, 1])
    rhs = np.column_stack((solved.coefficients, 2 * solved.coefficients))
    np.testing.assert_allclose(solved.reconstruct(solved.project(rhs)), rhs)
    augmented = np.vstack((X, np.array([[0.0, 1.0, 0.0]])))
    reduced_rhs = solved.project(rhs)
    expected = np.linalg.solve(
        (augmented.T @ augmented)[np.ix_(solved.keep, solved.keep)], reduced_rhs
    )
    actual = qr_hessian_inverse(solved.R, reduced_rhs, solved.pivots)
    np.testing.assert_allclose(solved.reconstruct(actual), solved.reconstruct(expected))

    snapshots = {
        "state_r": state.R.copy(),
        "state_qtz": state.qtz.copy(),
        "state_pivots": state.pivots.copy(),
        "factor_r": solved.R.copy(),
        "factor_pivots": solved.pivots.copy(),
        "factor_coefficients": solved.coefficients.copy(),
        "factor_keep": solved.keep.copy(),
    }
    X[0, 0] = 99.0
    z[0] = -99.0
    root[0, 0] = 99.0
    for output in (
        state.R,
        state.qtz,
        state.pivots,
        solved.R,
        solved.pivots,
        solved.coefficients,
        solved.keep,
    ):
        with np.testing.assert_raises(ValueError):
            output.flat[0] = 0
    np.testing.assert_allclose(state.R, snapshots["state_r"])
    np.testing.assert_allclose(state.qtz, snapshots["state_qtz"])
    np.testing.assert_array_equal(state.pivots, snapshots["state_pivots"])
    np.testing.assert_allclose(solved.R, snapshots["factor_r"])
    np.testing.assert_array_equal(solved.pivots, snapshots["factor_pivots"])
    np.testing.assert_allclose(solved.coefficients, snapshots["factor_coefficients"])
    np.testing.assert_array_equal(solved.keep, snapshots["factor_keep"])


def test_empty_updates_and_rank_zero_roots_are_neutral_or_rejected() -> None:
    empty = qr_update(None, np.empty((0, 2)), np.empty(0), n_coef=2)
    assert empty.n_data_rows == 0
    with np.testing.assert_raises(np.linalg.LinAlgError):
        solve_augmented_qr(empty, ())
    rank_zero = solve_augmented_qr(empty, (), identifiable_subspace=[])
    assert rank_zero.R.shape == (0, 0)
    np.testing.assert_allclose(rank_zero.coefficients, [0.0, 0.0])
    with np.testing.assert_raises(ValueError):
        qr_update(empty, np.ones((1, 2)), np.ones(1), n_coef=3)


def test_local_root_slice_validation_and_zero_row_roots() -> None:
    state = reduce_positive_qr(((np.eye(2), np.ones(2)),))
    baseline = solve_augmented_qr(state, ())
    neutral = solve_augmented_qr(state, ((slice(0, 2), np.empty((0, 2))),))
    np.testing.assert_allclose(neutral.coefficients, baseline.coefficients)
    for invalid in (slice(1.0, 2), slice(False, 1), slice(1, 0), slice(0, 3)):
        with np.testing.assert_raises(ValueError):
            solve_augmented_qr(state, ((invalid, np.empty((0, 0))),))


def test_balanced_roots_keep_structural_rank_separate_from_lambda_scale() -> None:
    """Tiny actual roots are conditioning failures, never structural aliases."""
    state = reduce_positive_qr((), n_coef=2)
    balanced_roots = ((slice(0, 2), np.eye(2)),)
    stable = solve_augmented_qr(
        state,
        ((slice(0, 2), np.diag([4.0, 5.0])),),
        balanced_roots=balanced_roots,
    )
    np.testing.assert_allclose(stable.coefficients, [0.0, 0.0])
    with np.testing.assert_raises(ValueError):
        solve_augmented_qr(
            state,
            ((slice(0, 2), np.eye(2)),),
            balanced_roots=((slice(0, 1), np.ones((1, 1))),),
        )
    with np.testing.assert_raises_regex(
        np.linalg.LinAlgError, "actual augmented QR is numerically ill-conditioned"
    ):
        solve_augmented_qr(
            state,
            ((slice(0, 2), np.diag([1e10, 1e-10])),),
            balanced_roots=balanced_roots,
        )
