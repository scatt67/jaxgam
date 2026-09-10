"""Matched-state tests for the pure JIT EFS statistics kernel."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.fitting.efs import (
    EFSRoot,
    EFSStatistics,
    EFSStatisticsPlan,
    efs_raw_update,
    efs_statistics,
    prepare_efs_statistics,
)
from jaxgam.fitting.penalty_ops import (
    JaxLocalPenalty,
    JaxPenaltyBlock,
    JaxPenaltyStructure,
    JaxTransform,
)
from tests.helpers import r_available
from tests.r_bridge import RBridge
from tests.tolerances import STRICT


def _plan() -> EFSStatisticsPlan:
    # First term is a singleton; the following two noncommuting PSD matrices
    # share the exact same range and exercise the coupled determinant path.
    s1 = np.array([[2.0, 0.4], [0.4, 1.0]])
    s2 = np.array([[1.0, -0.3], [-0.3, 3.0]])
    return EFSStatisticsPlan(
        4,
        3,
        (
            EFSRoot(
                0, 2, 0, "dense", jnp.asarray(np.array([[0.0], [np.sqrt(2.0)]])), ()
            ),
            EFSRoot(2, 4, 1, "dense", jnp.asarray(np.linalg.cholesky(s1)), ()),
            EFSRoot(2, 4, 2, "dense", jnp.asarray(np.linalg.cholesky(s2)), ()),
        ),
        (0,),
        (1,),
        ((1, 2),),
        (2,),
        ((jnp.asarray(s1), jnp.asarray(s2)),),
    )


def test_statistics_match_dense_independent_algebra_and_jit() -> None:
    plan = _plan()
    rho = jnp.array([-0.2, 0.4, -0.3])
    beta = jnp.array([0.5, -1.0, 0.2, 0.7])
    fisher = jnp.array(
        [
            [4.0, 0.2, 0.1, 0.0],
            [0.2, 3.0, 0.0, 0.1],
            [0.1, 0.0, 2.5, 0.4],
            [0.0, 0.1, 0.4, 3.5],
        ]
    )
    penalty = np.zeros((4, 4))
    matrices = [np.diag([0.0, 2.0, 0.0, 0.0]), np.zeros((4, 4)), np.zeros((4, 4))]
    matrices[1][2:, 2:] = np.array([[2.0, 0.4], [0.4, 1.0]])
    matrices[2][2:, 2:] = np.array([[1.0, -0.3], [-0.3, 3.0]])
    for value, matrix in zip(np.exp(np.asarray(rho)), matrices, strict=True):
        penalty += value * matrix
    hessian = np.asarray(fisher) + penalty
    inverse = np.linalg.inv(hessian)
    expected_q = np.array(
        [np.asarray(beta) @ matrix @ np.asarray(beta) for matrix in matrices]
    )
    expected_t = np.array([np.trace(inverse @ matrix) for matrix in matrices])
    coupled = (
        np.exp(float(rho[1])) * matrices[1][2:, 2:]
        + np.exp(float(rho[2])) * matrices[2][2:, 2:]
    )
    expected_d = np.array(
        [
            1.0,
            np.exp(float(rho[1]))
            * np.trace(np.linalg.solve(coupled, matrices[1][2:, 2:])),
            np.exp(float(rho[2]))
            * np.trace(np.linalg.solve(coupled, matrices[2][2:, 2:])),
        ]
    )
    actual = jax.jit(efs_statistics)(
        plan, beta, jnp.linalg.cholesky(fisher + jnp.asarray(penalty)), rho
    )
    np.testing.assert_allclose(
        actual.quadratic, expected_q, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        actual.fisher_trace, expected_t, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        actual.determinant_derivative, expected_d, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert bool(actual.determinant_valid)


def test_dynamic_plan_leaves_change_statistics_without_recompile_contract() -> None:
    plan = _plan()
    beta = jnp.ones(4)
    L = jnp.linalg.cholesky(jnp.eye(4) * 5)
    compiled = jax.jit(efs_statistics)
    first = compiled(plan, beta, L, jnp.zeros(3))
    changed = EFSStatisticsPlan(
        plan.n_coef,
        plan.n_penalties,
        plan.roots,
        plan.singleton_sp_indices,
        plan.singleton_ranks,
        plan.multi_block_sp_indices,
        plan.multi_block_ranks,
        ((plan.multi_block_proj_S[0][0] * 2, plan.multi_block_proj_S[0][1]),),
    )
    second = compiled(changed, beta, L, jnp.zeros(3))
    assert not np.allclose(
        np.asarray(first.determinant_derivative),
        np.asarray(second.determinant_derivative),
    )


def test_zero_member_and_singular_common_range_are_visible() -> None:
    zero_member = EFSStatisticsPlan(
        2,
        2,
        (
            EFSRoot(0, 2, 0, "dense", jnp.array([[1.0], [0.0]]), ()),
            EFSRoot(0, 2, 1, "dense", jnp.empty((2, 0)), ()),
        ),
        (),
        (),
        ((0, 1),),
        (1,),
        ((jnp.ones((1, 1)), jnp.zeros((1, 1))),),
    )
    good = efs_statistics(zero_member, jnp.ones(2), jnp.eye(2), jnp.zeros(2))
    np.testing.assert_allclose(
        good.determinant_derivative, [1.0, 0.0], rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert bool(good.determinant_valid)
    invalid = EFSStatisticsPlan(
        2, 1, (), (), (), ((0,),), (1,), ((jnp.zeros((1, 1)),),)
    )
    bad = jax.jit(efs_statistics)(invalid, jnp.ones(2), jnp.eye(2), jnp.zeros(1))
    assert not bool(bad.determinant_valid)
    assert np.isnan(np.asarray(bad.determinant_derivative)[0])


def test_raw_ratio_preserves_efsudr_boundary_ordering() -> None:
    stats = EFSStatistics(
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
        jnp.array([0.0, 0.0, 1.0]),
        jnp.array(True),
        jnp.array(True),
    )
    raw = jax.jit(efs_raw_update)(
        jnp.zeros(3), stats, jnp.array(1.0), jnp.array(1.0), jnp.array(15.0)
    )
    # positive / zero -> non-finite replacement; zero / zero -> one;
    # zero / positive -> literal zero and hence a -infinity trial.
    np.testing.assert_allclose(
        raw.ratio[:2], [1e6, 1.0], rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert raw.ratio[2] == 0
    assert np.isneginf(np.asarray(raw.log_smoothing_trial)[2])
    assert not bool(raw.finite_positive)


def test_prepare_roots_remains_local_and_recovers_quadratic() -> None:
    structure = JaxPenaltyStructure(
        3,
        (
            JaxPenaltyBlock(
                1,
                3,
                (0,),
                (JaxLocalPenalty("diagonal", jnp.array([2.0, 5.0]), 2),),
                JaxTransform("identity", jnp.array(1.0), 2),
            ),
        ),
    )
    fd = SimpleNamespace(
        n_coef=3,
        n_penalties=1,
        penalty_structure=structure,
        singleton_sp_indices=(0,),
        singleton_ranks=(2,),
        multi_block_sp_indices=(),
        multi_block_ranks=(),
        multi_block_proj_S=(),
    )
    plan = prepare_efs_statistics(fd)
    assert plan.roots[0].values.shape == (2,)
    assert plan.roots[0].kind == "diagonal"
    result = efs_statistics(plan, jnp.array([4.0, 2.0, -1.0]), jnp.eye(3), jnp.zeros(1))
    np.testing.assert_allclose(
        result.quadratic, [13.0], rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_scaled_identity_and_empty_diagonal_keep_bounded_rhs_contract() -> None:
    structure = JaxPenaltyStructure(
        40,
        (
            JaxPenaltyBlock(
                0,
                40,
                (0,),
                (JaxLocalPenalty("identity", jnp.array(3.0), 40),),
                JaxTransform("identity", jnp.array(1.0), 40),
            ),
            JaxPenaltyBlock(
                0,
                0,
                (1,),
                (JaxLocalPenalty("diagonal", jnp.empty((0,)), 0),),
                JaxTransform("identity", jnp.array(1.0), 0),
            ),
        ),
    )
    fd = SimpleNamespace(
        n_coef=40,
        n_penalties=2,
        penalty_structure=structure,
        singleton_sp_indices=(0, 1),
        singleton_ranks=(40, 0),
        multi_block_sp_indices=(),
        multi_block_ranks=(),
        multi_block_proj_S=(),
    )
    plan = prepare_efs_statistics(fd)
    actual = jax.jit(efs_statistics)(plan, jnp.ones(40), jnp.eye(40), jnp.zeros(2))
    np.testing.assert_allclose(
        actual.quadratic, [120.0, 0.0], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        actual.fisher_trace, [120.0, 0.0], rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_preparation_rejects_indefinite_penalty_and_raw_validity_masks() -> None:
    structure = JaxPenaltyStructure(
        2,
        (
            JaxPenaltyBlock(
                0,
                2,
                (0,),
                (JaxLocalPenalty("dense", jnp.diag(jnp.array([1.0, -2.0])), 2),),
                JaxTransform("identity", jnp.array(1.0), 2),
            ),
        ),
    )
    fd = SimpleNamespace(
        n_coef=2,
        n_penalties=1,
        penalty_structure=structure,
        singleton_sp_indices=(0,),
        singleton_ranks=(1,),
        multi_block_sp_indices=(),
        multi_block_ranks=(),
        multi_block_proj_S=(),
    )
    with np.testing.assert_raises_regex(ValueError, "positive semidefinite"):
        prepare_efs_statistics(fd)
    for kind, values in (
        ("identity", jnp.array(jnp.nan)),
        ("diagonal", jnp.array([jnp.nan])),
        ("dense", jnp.array([[jnp.nan]])),
    ):
        malformed = JaxPenaltyStructure(
            1,
            (
                JaxPenaltyBlock(
                    0,
                    1,
                    (0,),
                    (JaxLocalPenalty(kind, values, 1),),
                    JaxTransform("identity", jnp.array(1.0), 1),
                ),
            ),
        )
        bad_fd = SimpleNamespace(
            n_coef=1,
            n_penalties=1,
            penalty_structure=malformed,
            singleton_sp_indices=(0,),
            singleton_ranks=(0,),
            multi_block_sp_indices=(),
            multi_block_ranks=(),
            multi_block_proj_S=(),
        )
        with np.testing.assert_raises_regex(ValueError, "finite"):
            prepare_efs_statistics(bad_fd)
    invalid_stats = EFSStatistics(
        jnp.array([1.0]),
        jnp.array([jnp.nan]),
        jnp.array([0.0]),
        jnp.array(True),
        jnp.array(False),
    )
    raw = efs_raw_update(
        jnp.zeros(1), invalid_stats, jnp.array(1.0), jnp.array(1.0), jnp.array(15.0)
    )
    assert raw.ratio[0] == 1e6
    assert not bool(raw.finite_positive)


@pytest.mark.skipif(not r_available(), reason="pinned R with mgcv not available")
def test_pinned_r_gam_reparam_and_covariance_root_algebra() -> None:
    s1 = np.array([[2.0, 0.4], [0.4, 1.0]])
    s2 = np.array([[1.0, -0.3], [-0.3, 3.0]])
    roots = (np.linalg.cholesky(s1), np.linalg.cholesky(s2))
    plan = EFSStatisticsPlan(
        3,
        2,
        (
            EFSRoot(1, 3, 0, "dense", jnp.asarray(roots[0]), ()),
            EFSRoot(1, 3, 1, "dense", jnp.asarray(roots[1]), ()),
        ),
        (),
        (),
        ((0, 1),),
        (2,),
        ((jnp.asarray(s1), jnp.asarray(s2)),),
    )
    rho = jnp.array([0.3, -0.5])
    beta = jnp.array([0.2, -0.8, 1.1])
    L = jnp.linalg.cholesky(
        jnp.array([[4.0, 0.2, 0.1], [0.2, 3.0, 0.4], [0.1, 0.4, 2.0]])
    )
    actual = efs_statistics(plan, beta, L, rho)
    covariance_roots = [
        np.vstack((np.zeros((1, root.shape[1])), root)) for root in roots
    ]
    oracle = RBridge(mode="subprocess").efs_statistics_algebra(
        np.asarray(rho), list(roots), covariance_roots, np.asarray(beta), np.asarray(L)
    )
    np.testing.assert_allclose(
        actual.determinant_derivative, oracle["d"], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        actual.fisher_trace, oracle["t"], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        actual.quadratic, oracle["q"], rtol=STRICT.rtol, atol=STRICT.atol
    )
