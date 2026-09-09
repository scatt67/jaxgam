"""Algebra tests for local Phase-2 penalty descriptors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.fitting.penalty_ops import (
    JaxLocalPenalty,
    JaxPenaltyBlock,
    JaxPenaltyStructure,
    JaxTransform,
    add_to_dense,
    apply,
    log_pdet,
    materialize_difference,
    parameter_vjp,
    quadratic,
    transform_covariance,
)
from tests.tolerances import STRICT


def _structure() -> JaxPenaltyStructure:
    """One diagonal and two coupled dense penalties in disjoint blocks."""
    return JaxPenaltyStructure(
        5,
        (
            JaxPenaltyBlock(
                1,
                3,
                (0,),
                (JaxLocalPenalty("diagonal", jnp.array([2.0, 3.0]), 2),),
                JaxTransform("diagonal", jnp.array([2.0, 4.0]), 2),
            ),
            JaxPenaltyBlock(
                3,
                5,
                (1, 2),
                (
                    JaxLocalPenalty("dense", jnp.array([[1.0, 0.2], [0.2, 2.0]]), 2),
                    JaxLocalPenalty("identity", jnp.array(0.5), 2),
                ),
                JaxTransform("identity", jnp.array(1.0), 2),
            ),
        ),
    )


def _expected(rho: jax.Array) -> jax.Array:
    result = jnp.zeros((5, 5))
    result = result.at[1:3, 1:3].set(jnp.diag(jnp.exp(rho[0]) * jnp.array([2.0, 3.0])))
    result = result.at[3:5, 3:5].set(
        jnp.exp(rho[1]) * jnp.array([[1.0, 0.2], [0.2, 2.0]])
        + jnp.exp(rho[2]) * 0.5 * jnp.eye(2)
    )
    return result


def test_actions_quadratic_dense_and_jit_agree() -> None:
    structure = _structure()
    rho = jnp.array([0.3, -0.2, 0.7])
    beta = jnp.array([1.0, -2.0, 0.5, 1.2, -0.4])
    expected = _expected(rho)
    np.testing.assert_allclose(
        add_to_dense(structure, jnp.zeros((5, 5)), rho),
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        apply(structure, beta, rho), expected @ beta, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        quadratic(structure, beta, rho),
        beta @ expected @ beta,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        jax.jit(apply)(structure, beta, rho),
        expected @ beta,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_parameter_vjp_matches_autodiff() -> None:
    structure = _structure()
    rho = jnp.array([0.3, -0.2, 0.7])
    beta = jnp.array([1.0, -2.0, 0.5, 1.2, -0.4])
    adjoint = jnp.array([-0.5, 0.8, 0.2, -1.0, 0.6])
    expected = jax.grad(lambda x: jnp.vdot(adjoint, apply(structure, beta, x)))(rho)
    np.testing.assert_allclose(
        parameter_vjp(structure, beta, adjoint, rho),
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_penalty_difference_preserves_tiny_signed_changes_under_jit() -> None:
    structure = _structure()
    base = jnp.array([0.3, -0.2, 0.7])
    trial = base + jnp.array([1e-12, -1e-12, 2e-12])
    multipliers = np.exp(np.asarray(base)) * np.expm1(np.asarray(trial - base))
    expected = np.zeros((5, 5))
    expected[1:3, 1:3] = multipliers[0] * np.diag([2.0, 3.0])
    expected[3:5, 3:5] = multipliers[1] * np.array(
        [[1.0, 0.2], [0.2, 2.0]]
    ) + multipliers[2] * 0.5 * np.eye(2)
    actual = jax.jit(materialize_difference)(structure, base, trial)
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=0.0)
    np.testing.assert_array_equal(materialize_difference(structure, base, base), 0.0)


def test_log_pdet_uses_multi_block_singularity_sentinel() -> None:
    rho = jnp.array([0.2, -0.4])
    good = log_pdet(
        rho,
        (0,),
        (1,),
        jnp.array([jnp.log(3.0)]),
        ((1,),),
        (1,),
        ((jnp.array([[2.0]]),),),
    )
    np.testing.assert_allclose(
        good,
        rho[0] + jnp.log(3.0) + rho[1] + jnp.log(2.0),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    singular = log_pdet(
        rho, (), (), jnp.array([]), ((1,),), (1,), ((jnp.array([[0.0]]),),)
    )
    assert singular == -1e10 + rho[1]


def test_covariance_transform_preserves_off_diagonal_blocks() -> None:
    structure = _structure()
    covariance = jnp.arange(25.0).reshape(5, 5)
    actual = transform_covariance(structure, covariance)
    D = np.eye(5)
    D[1:3, 1:3] = np.diag([2.0, 4.0])
    np.testing.assert_allclose(
        actual, D @ covariance @ D.T, rtol=STRICT.rtol, atol=STRICT.atol
    )
