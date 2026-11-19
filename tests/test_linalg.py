"""Tests for jaxgam.linalg.

Coverage:
1. TestChoFactor — plain Cholesky with diagonal-relative fallback
2. TestPenalizedCholesky — penalized Hessian factorization and solve
3. TestNumericalRank — rank estimation via pivoted QR
4. TestJITCompilation — every function runs under jax.jit
5. TestDifferentiability — jax.grad through key functions
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np
import pytest

from jaxgam.jax_utils import (
    _CHOLESKY_LARGE_RELATIVE_JITTER,
    _CHOLESKY_SMALL_RELATIVE_JITTER,
    cho_factor,
    numerical_rank,
    penalized_cholesky,
    penalized_solve,
)
from tests.tolerances import STRICT

jax.config.update("jax_enable_x64", True)

SEED = 123


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pd_matrix() -> jnp.ndarray:
    """Well-conditioned 10x10 positive-definite matrix."""
    rng = np.random.default_rng(SEED)
    A = rng.standard_normal((10, 10))
    return jnp.array(A.T @ A + 5.0 * np.eye(10))


@pytest.fixture
def near_singular_matrix() -> jnp.ndarray:
    """10x10 matrix with a negative eigenvalue (not PD, triggers large jitter)."""
    rng = np.random.default_rng(SEED)
    Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    eigvals = np.array([-0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0])
    return jnp.array(Q @ np.diag(eigvals) @ Q.T)


@pytest.fixture
def realistic_penalized_system() -> dict:
    """GAM-like penalized system: XtWX + S_lambda with rank-deficient penalty."""
    rng = np.random.default_rng(SEED)
    n, p = 50, 10
    X = rng.standard_normal((n, p))
    w = np.abs(rng.standard_normal(n)) + 0.1
    XtWX = jnp.array(X.T @ np.diag(w) @ X)
    # Rank-deficient penalty (penalizes only first 8 of 10 basis functions)
    S = np.zeros((p, p))
    S[:8, :8] = np.eye(8) * 2.0
    S_lambda = jnp.array(S)
    rhs = jnp.array(X.T @ (w * rng.standard_normal(n)))
    return {"XtWX": XtWX, "S_lambda": S_lambda, "rhs": rhs}


# ---------------------------------------------------------------------------
# TestChoFactor
# ---------------------------------------------------------------------------


class TestChoFactor:
    """Tests for cho_factor: plain factor, regularized only on failure."""

    def test_reconstruction(self, pd_matrix: jnp.ndarray) -> None:
        """L @ L.T ≈ H + jitter * I."""
        L, jitter = cho_factor(pd_matrix)
        p = pd_matrix.shape[0]
        reconstructed = L @ L.T
        expected = pd_matrix + jitter * jnp.eye(p)
        np.testing.assert_allclose(
            np.array(reconstructed),
            np.array(expected),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_plain_factor_unless_not_positive_definite(
        self,
        pd_matrix: jnp.ndarray,
        near_singular_matrix: jnp.ndarray,
    ) -> None:
        """A PD matrix gets the plain factor; fallbacks are diagonal-relative.

        No jitter is added while the plain Cholesky succeeds (mgcv's QR-based
        penalized fits never add one). A singular PSD matrix needs the
        ``1e-12 * |diag(H)|`` fallback, and an indefinite matrix reports the
        ``1e-6`` level even when that still cannot make it factorizable.
        """
        L, level = cho_factor(pd_matrix)
        assert float(level) == 0.0
        np.testing.assert_array_equal(
            np.array(L), np.array(jnp.linalg.cholesky(pd_matrix))
        )

        rng = np.random.default_rng(SEED)
        basis = rng.standard_normal((10, 6))
        singular = jnp.array(basis @ basis.T)
        L_singular, singular_level = cho_factor(singular)
        assert float(singular_level) == _CHOLESKY_SMALL_RELATIVE_JITTER
        regularized = singular + _CHOLESKY_SMALL_RELATIVE_JITTER * jnp.diag(
            jnp.abs(jnp.diag(singular))
        )
        np.testing.assert_allclose(
            np.array(L_singular @ L_singular.T),
            np.array(regularized),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

        _, indefinite_level = cho_factor(near_singular_matrix)
        assert float(indefinite_level) == _CHOLESKY_LARGE_RELATIVE_JITTER

    def test_cho_solve_roundtrip(self, pd_matrix: jnp.ndarray) -> None:
        """cho_solve(cho_factor(H), b) ≈ H^{-1} b."""
        b = jnp.arange(1.0, pd_matrix.shape[0] + 1.0)
        L, _ = cho_factor(pd_matrix)
        x = jsla.cho_solve((L, True), b)
        np.testing.assert_allclose(
            np.array(pd_matrix @ x),
            np.array(b),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


# ---------------------------------------------------------------------------
# TestPenalizedCholesky
# ---------------------------------------------------------------------------


class TestPenalizedCholesky:
    """Tests for penalized_cholesky and penalized_solve."""

    def test_penalized_system_reconstructs_and_solves(
        self, realistic_penalized_system: dict
    ) -> None:
        """Penalized factorization reconstructs H and solves H beta = rhs."""
        sys = realistic_penalized_system
        L, jitter = penalized_cholesky(sys["XtWX"], sys["S_lambda"])
        beta, L_solve, jitter_solve = penalized_solve(
            sys["XtWX"], sys["S_lambda"], sys["rhs"]
        )
        H = sys["XtWX"] + sys["S_lambda"]
        p = H.shape[0]
        np.testing.assert_allclose(
            np.array(L @ L.T),
            np.array(H + jitter * jnp.eye(p)),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        H_jittered = (
            sys["XtWX"] + sys["S_lambda"] + jitter * jnp.eye(sys["rhs"].shape[0])
        )
        np.testing.assert_allclose(
            np.array(H_jittered @ beta),
            np.array(sys["rhs"]),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_array_equal(np.array(L), np.array(L_solve))
        np.testing.assert_array_equal(float(jitter), float(jitter_solve))

    def test_positive_definite_system_is_unregularized(
        self, realistic_penalized_system: dict
    ) -> None:
        """A positive-definite penalized Hessian is factored without jitter."""
        sys = realistic_penalized_system
        L, level = penalized_cholesky(sys["XtWX"], sys["S_lambda"])
        assert float(level) == 0.0
        np.testing.assert_array_equal(
            np.array(L), np.array(jnp.linalg.cholesky(sys["XtWX"] + sys["S_lambda"]))
        )


# ---------------------------------------------------------------------------
# TestNumericalRank
# ---------------------------------------------------------------------------


class TestNumericalRank:
    """Tests for numerical_rank via pivoted QR."""

    def test_standard_ranks(self) -> None:
        """Full-rank and rank-one matrices have expected numerical ranks."""
        rng = np.random.default_rng(SEED)
        cases = [
            (jnp.array(rng.standard_normal((5, 3))), 3),
            (jnp.array([[1.0], [2.0], [3.0]]) @ jnp.array([[1.0, 2.0, 3.0]]), 1),
        ]

        for A, expected_rank in cases:
            assert int(numerical_rank(A)) == expected_rank

    def test_rank_deficient(self) -> None:
        """Rank-deficient matrix detected correctly."""
        # Columns 2 = 2 * column 1
        A = jnp.array([[1.0, 2.0, 2.0], [3.0, 4.0, 6.0], [5.0, 6.0, 10.0]])
        assert int(numerical_rank(A)) == 2

    def test_explicit_tol(self) -> None:
        """Explicit tolerance overrides default."""
        rng = np.random.default_rng(SEED)
        A = jnp.array(rng.standard_normal((5, 5)))
        # Very large tol should reduce apparent rank
        rank_large_tol = int(numerical_rank(A, tol=1e10))
        rank_default = int(numerical_rank(A))
        assert rank_large_tol < rank_default

    def test_numpy_rank_match(self) -> None:
        """Matches numpy.linalg.matrix_rank for standard cases."""
        rng = np.random.default_rng(SEED)
        A_np = rng.standard_normal((8, 5))
        A_jax = jnp.array(A_np)
        assert int(numerical_rank(A_jax)) == int(np.linalg.matrix_rank(A_np))


# ---------------------------------------------------------------------------
# TestJITCompilation
# ---------------------------------------------------------------------------


class TestJITCompilation:
    """Verify every function compiles and runs under jax.jit."""

    @pytest.mark.parametrize(
        "function_name",
        [
            "cho_factor",
            "penalized_cholesky",
            "penalized_solve",
            "numerical_rank",
            "slogdet",
            "solve_triangular",
        ],
    )
    def test_key_functions_jit(
        self,
        function_name: str,
        pd_matrix: jnp.ndarray,
        realistic_penalized_system: dict,
    ) -> None:
        """Each key linear algebra function compiles and produces finite output."""
        sys = realistic_penalized_system
        if function_name == "cho_factor":
            L, _jitter = jax.jit(cho_factor)(pd_matrix)
            assert jnp.all(jnp.isfinite(L))
        elif function_name == "penalized_cholesky":
            L, _jitter = jax.jit(penalized_cholesky)(sys["XtWX"], sys["S_lambda"])
            assert jnp.all(jnp.isfinite(L))
        elif function_name == "penalized_solve":
            beta, _L, _jitter = jax.jit(penalized_solve)(
                sys["XtWX"], sys["S_lambda"], sys["rhs"]
            )
            assert jnp.all(jnp.isfinite(beta))
        elif function_name == "numerical_rank":
            A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            rank = jax.jit(numerical_rank)(A)
            assert int(rank) == 2
        elif function_name == "slogdet":
            _sign, logdet = jax.jit(jnp.linalg.slogdet)(pd_matrix)
            assert jnp.isfinite(logdet)
        elif function_name == "solve_triangular":
            L, _ = cho_factor(pd_matrix)
            b = jnp.ones(pd_matrix.shape[0])
            x = jax.jit(jsla.solve_triangular, static_argnames=("lower",))(
                L, b, lower=True
            )
            assert jnp.all(jnp.isfinite(x))


# ---------------------------------------------------------------------------
# TestDifferentiability
# ---------------------------------------------------------------------------


class TestDifferentiability:
    """Verify jax.grad produces finite gradients through key functions."""

    def test_slogdet_grad(self, pd_matrix: jnp.ndarray) -> None:
        """jax.grad through slogdet produces finite gradients."""

        def logdet_fn(H: jax.Array) -> jax.Array:
            _, logdet = jnp.linalg.slogdet(H)
            return logdet

        grad = jax.grad(logdet_fn)(pd_matrix)
        assert jnp.all(jnp.isfinite(grad))

    def test_cho_solve_grad(self, pd_matrix: jnp.ndarray) -> None:
        """jax.grad through cho_solve produces finite gradients."""

        def solve_norm(H: jax.Array) -> jax.Array:
            L = jnp.linalg.cholesky(H)
            b = jnp.ones(H.shape[0])
            x = jsla.cho_solve((L, True), b)
            return jnp.sum(x**2)

        grad = jax.grad(solve_norm)(pd_matrix)
        assert jnp.all(jnp.isfinite(grad))

    def test_penalized_solve_grad(self, realistic_penalized_system: dict) -> None:
        """jax.grad through penalized_solve produces finite gradients.

        Differentiates w.r.t. rhs (the typical PIRLS use case).
        """
        sys = realistic_penalized_system

        def objective(rhs: jax.Array) -> jax.Array:
            beta, _, _ = penalized_solve(sys["XtWX"], sys["S_lambda"], rhs)
            return jnp.sum(beta**2)

        grad = jax.grad(objective)(sys["rhs"])
        assert jnp.all(jnp.isfinite(grad))
