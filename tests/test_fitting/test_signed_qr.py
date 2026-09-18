"""Pure compiled signed-factor actions and determinant admissibility."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.fitting.signed_qr import SignedQRCoefficientFactor
from jaxgam.fitting.state import PivotedQRCoefficientFactor
from tests.tolerances import STRICT


@pytest.mark.parametrize("matrix_rhs", [False, True])
def test_signed_factor_actions_preserve_embedded_coordinates(matrix_rhs):
    R = np.array([[2.0, 0.3], [0.0, 1.2]])
    vectors = np.array([[0.6, -0.8], [0.8, 0.6]])
    correction = np.array([0.2, 1.0])
    factor = SignedQRCoefficientFactor(
        PivotedQRCoefficientFactor(
            jnp.asarray(R), jnp.array([1, 0]), jnp.array([0, 2]), 3
        ),
        jnp.asarray(vectors),
        jnp.asarray(correction),
    )
    B = np.sqrt(correction)[:, None] * (vectors.T @ R[:, [1, 0]])
    H = B.T @ B
    rhs = np.arange(6.0).reshape(3, 2) if matrix_rhs else np.arange(3.0)
    result = jax.jit(lambda f, b: f.hessian_inverse(b))(factor, jnp.asarray(rhs))
    expected = np.zeros_like(rhs)
    expected[[0, 2]] = np.linalg.solve(H, rhs[[0, 2]])
    np.testing.assert_allclose(result, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    determinant = jax.jit(lambda f: f.logdet_hessian())(factor)
    np.testing.assert_allclose(
        determinant, np.linalg.slogdet(H)[1], rtol=STRICT.rtol, atol=STRICT.atol
    )
    # The pytree contains only coefficient-space factors; tracing root actions
    # needs no response, weights, or design closure.
    assert [leaf.shape for leaf in jax.tree.leaves(factor)] == [
        (2, 2),
        (2,),
        (2,),
        (2, 2),
        (2,),
    ]


def test_signed_factor_zero_mode_is_a_solve_but_not_a_score():
    factor = SignedQRCoefficientFactor(
        PivotedQRCoefficientFactor(jnp.eye(2), jnp.arange(2), jnp.arange(2), 2),
        jnp.eye(2),
        jnp.array([0.0, 1.0]),
    )
    value, determinant, admissible = jax.jit(
        lambda f: (
            f.hessian_inverse(jnp.eye(2)),
            f.logdet_hessian(),
            f.score_admissible,
        )
    )(factor)
    np.testing.assert_array_equal(value, np.diag([0.0, 1.0]))
    assert np.isneginf(determinant)
    assert not bool(admissible)
    with pytest.raises(ValueError, match="rank"):
        factor.root_inverse(jnp.ones((2, 1, 1)))
