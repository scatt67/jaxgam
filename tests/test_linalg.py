"""Tests for jaxgam.linalg.

Coverage:
1. TestChoFactor — jittered Cholesky factorization
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
import scipy.linalg as sla

from jaxgam.jax_utils import (
    cho_factor,
    numerical_rank,
    penalized_cholesky,
    penalized_solve,
)
from tests.tolerances import MODERATE, STRICT

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
    """Tests for cho_factor with jitter stabilization."""

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

    def test_lower_triangular(self, pd_matrix: jnp.ndarray) -> None:
        """L must be lower-triangular."""
        L, _ = cho_factor(pd_matrix)
        upper = jnp.triu(L, k=1)
        assert jnp.allclose(upper, 0.0)

    def test_small_jitter_well_conditioned(self, pd_matrix: jnp.ndarray) -> None:
        """Well-conditioned matrix uses eps_small jitter."""
        _, jitter = cho_factor(pd_matrix)
        p = pd_matrix.shape[0]
        trace_H = jnp.trace(pd_matrix)
        eps_small = jnp.maximum(1e-12 * trace_H / p, 1e-10)
        np.testing.assert_allclose(float(jitter), float(eps_small))

    def test_large_jitter_near_singular(
        self, near_singular_matrix: jnp.ndarray
    ) -> None:
        """Near-singular matrix triggers eps_large jitter."""
        _, jitter = cho_factor(near_singular_matrix)
        p = near_singular_matrix.shape[0]
        trace_H = jnp.trace(near_singular_matrix)
        eps_small = float(jnp.maximum(1e-12 * trace_H / p, 1e-10))
        # Jitter should be larger than eps_small
        assert float(jitter) > eps_small

    def test_scipy_match(self, pd_matrix: jnp.ndarray) -> None:
        """Solve via cho_factor matches scipy on well-conditioned system."""
        L, _ = cho_factor(pd_matrix)
        b = jnp.ones(pd_matrix.shape[0])
        x_jax = jsla.cho_solve((L, True), b)

        H_np = np.array(pd_matrix)
        c, low = sla.cho_factor(H_np, lower=True)
        x_scipy = sla.cho_solve((c, low), np.ones(H_np.shape[0]))

        np.testing.assert_allclose(
            np.array(x_jax),
            x_scipy,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

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

    def test_multiple_rhs(self, pd_matrix: jnp.ndarray) -> None:
        """cho_solve works with matrix RHS."""
        p = pd_matrix.shape[0]
        B = jnp.eye(p)
        L, _ = cho_factor(pd_matrix)
        X = jsla.cho_solve((L, True), B)
        # X should be approximately H^{-1}
        np.testing.assert_allclose(
            np.array(pd_matrix @ X),
            np.array(B),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )


# ---------------------------------------------------------------------------
# TestPenalizedCholesky
# ---------------------------------------------------------------------------


class TestPenalizedCholesky:
    """Tests for penalized_cholesky and penalized_solve."""

    def test_penalized_cholesky_reconstruction(
        self, realistic_penalized_system: dict
    ) -> None:
        """L @ L.T ≈ XtWX + S_lambda + jitter * I."""
        sys = realistic_penalized_system
        L, jitter = penalized_cholesky(sys["XtWX"], sys["S_lambda"])
        H = sys["XtWX"] + sys["S_lambda"]
        p = H.shape[0]
        np.testing.assert_allclose(
            np.array(L @ L.T),
            np.array(H + jitter * jnp.eye(p)),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_penalized_solve_roundtrip(self, realistic_penalized_system: dict) -> None:
        """penalized_solve produces beta such that (XtWX + S_lambda) @ beta ≈ rhs."""
        sys = realistic_penalized_system
        beta, _L, jitter = penalized_solve(sys["XtWX"], sys["S_lambda"], sys["rhs"])
        H = sys["XtWX"] + sys["S_lambda"] + jitter * jnp.eye(sys["rhs"].shape[0])
        np.testing.assert_allclose(
            np.array(H @ beta),
            np.array(sys["rhs"]),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_penalized_solve_scipy_match(
        self, realistic_penalized_system: dict
    ) -> None:
        """penalized_solve matches scipy cho_solve."""
        sys = realistic_penalized_system
        beta, _, _ = penalized_solve(sys["XtWX"], sys["S_lambda"], sys["rhs"])

        H_np = np.array(sys["XtWX"] + sys["S_lambda"])
        c, low = sla.cho_factor(H_np, lower=True)
        beta_scipy = sla.cho_solve((c, low), np.array(sys["rhs"]))

        np.testing.assert_allclose(
            np.array(beta),
            beta_scipy,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_penalized_solve_returns_L(self, realistic_penalized_system: dict) -> None:
        """penalized_solve returns the same L as penalized_cholesky."""
        sys = realistic_penalized_system
        L_direct, jitter_direct = penalized_cholesky(sys["XtWX"], sys["S_lambda"])
        _, L_solve, jitter_solve = penalized_solve(
            sys["XtWX"], sys["S_lambda"], sys["rhs"]
        )
        np.testing.assert_array_equal(np.array(L_direct), np.array(L_solve))
        np.testing.assert_array_equal(float(jitter_direct), float(jitter_solve))

    def test_jitter_recorded(self, realistic_penalized_system: dict) -> None:
        """Jitter value is finite and positive."""
        sys = realistic_penalized_system
        _, jitter = penalized_cholesky(sys["XtWX"], sys["S_lambda"])
        assert jnp.isfinite(jitter)
        assert float(jitter) > 0


# ---------------------------------------------------------------------------
# TestNumericalRank
# ---------------------------------------------------------------------------


class TestNumericalRank:
    """Tests for numerical_rank via pivoted QR."""

    def test_full_rank(self) -> None:
        """Full-rank matrix has rank = min(m, n)."""
        rng = np.random.default_rng(SEED)
        A = jnp.array(rng.standard_normal((5, 3)))
        assert int(numerical_rank(A)) == 3

    def test_rank_deficient(self) -> None:
        """Rank-deficient matrix detected correctly."""
        # Columns 2 = 2 * column 1
        A = jnp.array([[1.0, 2.0, 2.0], [3.0, 4.0, 6.0], [5.0, 6.0, 10.0]])
        assert int(numerical_rank(A)) == 2

    def test_rank_one(self) -> None:
        """Rank-1 matrix."""
        v = jnp.array([[1.0], [2.0], [3.0]])
        A = v @ v.T
        assert int(numerical_rank(A)) == 1

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
# TestSlogdet
# ---------------------------------------------------------------------------


class TestSlogdet:
    """Tests for jnp.linalg.slogdet (used directly, no wrapper)."""

    def test_known_determinant(self) -> None:
        """slogdet of diagonal matrix matches known log-det."""
        D = jnp.diag(jnp.array([2.0, 3.0, 5.0]))
        sign, logdet = jnp.linalg.slogdet(D)
        assert float(sign) == 1.0
        np.testing.assert_allclose(float(logdet), np.log(30.0), rtol=STRICT.rtol)

    def test_numpy_match(self, pd_matrix: jnp.ndarray) -> None:
        """slogdet matches numpy."""
        sign_jax, logdet_jax = jnp.linalg.slogdet(pd_matrix)
        sign_np, logdet_np = np.linalg.slogdet(np.array(pd_matrix))
        assert float(sign_jax) == sign_np
        np.testing.assert_allclose(
            float(logdet_jax),
            logdet_np,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_positive_sign_for_pd(self, pd_matrix: jnp.ndarray) -> None:
        """PD matrix has positive determinant (sign = +1)."""
        sign, _ = jnp.linalg.slogdet(pd_matrix)
        assert float(sign) == 1.0


# ---------------------------------------------------------------------------
# TestSolveTriangular
# ---------------------------------------------------------------------------


class TestSolveTriangular:
    """Tests for jax.scipy.linalg.solve_triangular (used directly)."""

    def test_forward_solve(self, pd_matrix: jnp.ndarray) -> None:
        """L @ solve_triangular(L, b) ≈ b."""
        L, _ = cho_factor(pd_matrix)
        b = jnp.ones(pd_matrix.shape[0])
        x = jsla.solve_triangular(L, b, lower=True)
        np.testing.assert_allclose(
            np.array(L @ x),
            np.array(b),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_inverse_identity(self, pd_matrix: jnp.ndarray) -> None:
        """L @ L^{-1} ≈ I."""
        L, _ = cho_factor(pd_matrix)
        p = L.shape[0]
        L_inv = jsla.solve_triangular(L, jnp.eye(p), lower=True)
        np.testing.assert_allclose(
            np.array(L @ L_inv),
            np.eye(p),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_scipy_match(self, pd_matrix: jnp.ndarray) -> None:
        """Matches scipy.linalg.solve_triangular."""
        L_jax, _ = cho_factor(pd_matrix)
        L_np = np.array(L_jax)
        b = np.ones(pd_matrix.shape[0])

        x_jax = jsla.solve_triangular(L_jax, jnp.array(b), lower=True)
        x_scipy = sla.solve_triangular(L_np, b, lower=True)

        np.testing.assert_allclose(
            np.array(x_jax),
            x_scipy,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


# ---------------------------------------------------------------------------
# TestJITCompilation
# ---------------------------------------------------------------------------


class TestJITCompilation:
    """Verify every function compiles and runs under jax.jit."""

    def test_cho_factor_jit(self, pd_matrix: jnp.ndarray) -> None:
        L, _jitter = jax.jit(cho_factor)(pd_matrix)
        assert jnp.all(jnp.isfinite(L))

    def test_penalized_cholesky_jit(self, realistic_penalized_system: dict) -> None:
        sys = realistic_penalized_system
        L, _jitter = jax.jit(penalized_cholesky)(sys["XtWX"], sys["S_lambda"])
        assert jnp.all(jnp.isfinite(L))

    def test_penalized_solve_jit(self, realistic_penalized_system: dict) -> None:
        sys = realistic_penalized_system
        beta, _L, _jitter = jax.jit(penalized_solve)(
            sys["XtWX"], sys["S_lambda"], sys["rhs"]
        )
        assert jnp.all(jnp.isfinite(beta))

    def test_numerical_rank_jit(self) -> None:
        A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rank = jax.jit(numerical_rank)(A)
        assert int(rank) == 2

    def test_slogdet_jit(self, pd_matrix: jnp.ndarray) -> None:
        _sign, logdet = jax.jit(jnp.linalg.slogdet)(pd_matrix)
        assert jnp.isfinite(logdet)

    def test_solve_triangular_jit(self, pd_matrix: jnp.ndarray) -> None:
        L, _ = cho_factor(pd_matrix)
        b = jnp.ones(pd_matrix.shape[0])
        x = jax.jit(jsla.solve_triangular, static_argnames=("lower",))(L, b, lower=True)
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
