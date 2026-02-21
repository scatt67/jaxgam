"""Tests for penalty matrix classes.

Validates Penalty and CompositePenalty from pymgcv.penalties:
- PSD checks (eigenvalues >= 0)
- Symmetry validation
- Weighted penalty computation
- Embedding into global coefficient space
- Rank and null space dimension
- Edge cases (single penalty, multiple penalties, extreme log_lambda)
- No JAX imports (Phase 1 boundary guard)

Design doc reference: docs/design.md Section 10.2
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from pymgcv.penalties import CompositePenalty, Penalty
from tests.tolerances import STRICT

# ---------------------------------------------------------------------------
# Helpers — construct known penalty matrices
# ---------------------------------------------------------------------------


def _second_derivative_penalty(k: int) -> np.ndarray:
    """Build a rank-deficient second-derivative penalty matrix.

    This is the standard wiggliness penalty for a cubic spline with k
    knots. The null space (rank deficiency) is 2 — the constant and
    linear functions are unpenalised.

    The matrix is tridiagonal-like: D2.T @ D2 where D2 is the
    second-order finite-difference operator.
    """
    D2 = np.zeros((k - 2, k))
    for i in range(k - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2.T @ D2


def _full_rank_penalty(k: int, rng: np.random.Generator) -> np.ndarray:
    """Build a random symmetric PSD matrix with full rank."""
    A = rng.standard_normal((k, k))
    return A.T @ A + 0.01 * np.eye(k)


def _diagonal_penalty(k: int) -> np.ndarray:
    """Build a diagonal penalty matrix."""
    d = np.arange(1, k + 1, dtype=float)
    return np.diag(d)


# ---------------------------------------------------------------------------
# Tests — Penalty class
# ---------------------------------------------------------------------------


class TestPenalty:
    """Tests for the Penalty class."""

    def test_psd_check_second_derivative(self) -> None:
        """Eigenvalues of a second-derivative penalty are all >= 0."""
        k = 10
        S = _second_derivative_penalty(k)
        penalty = Penalty(S)
        eigvals = np.linalg.eigvalsh(penalty.S)
        # All eigenvalues must be >= 0 within STRICT tolerance
        assert np.all(eigvals >= -STRICT.atol), (
            f"Penalty matrix has negative eigenvalue(s): "
            f"min eigenvalue = {np.min(eigvals):.2e}"
        )

    def test_psd_check_full_rank(self) -> None:
        """Eigenvalues of a full-rank penalty are all > 0."""
        rng = np.random.default_rng(42)
        k = 8
        S = _full_rank_penalty(k, rng)
        penalty = Penalty(S)
        eigvals = np.linalg.eigvalsh(penalty.S)
        assert np.all(eigvals > 0), (
            f"Full-rank penalty should have all positive eigenvalues, "
            f"min = {np.min(eigvals):.2e}"
        )

    def test_psd_check_zero_matrix(self) -> None:
        """Zero matrix is PSD with rank 0."""
        k = 5
        S = np.zeros((k, k))
        penalty = Penalty(S)
        eigvals = np.linalg.eigvalsh(penalty.S)
        assert np.all(eigvals >= -STRICT.atol)
        assert penalty.rank == 0
        assert penalty.null_space_dim == k

    def test_psd_check_diagonal(self) -> None:
        """Diagonal penalty with positive entries is PSD."""
        k = 6
        S = _diagonal_penalty(k)
        penalty = Penalty(S)
        eigvals = np.linalg.eigvalsh(penalty.S)
        assert np.all(eigvals >= -STRICT.atol)

    def test_symmetry_at_strict_tolerance(self) -> None:
        """Stored penalty matrix is symmetric at STRICT tolerance."""
        k = 10
        S = _second_derivative_penalty(k)
        penalty = Penalty(S)
        np.testing.assert_allclose(
            penalty.S,
            penalty.S.T,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Penalty matrix is not symmetric at STRICT tolerance.",
        )

    def test_symmetry_forced_exact(self) -> None:
        """Even with tiny floating-point asymmetry, stored S is exactly symmetric."""
        rng = np.random.default_rng(123)
        k = 6
        A = rng.standard_normal((k, k))
        S = A.T @ A
        # Introduce tiny asymmetry (within tolerance)
        S_noisy = S.copy()
        S_noisy[0, 1] += 1e-15
        S_noisy[1, 0] -= 1e-15
        penalty = Penalty(S_noisy)
        # After construction, S should be exactly symmetric
        assert np.array_equal(penalty.S, penalty.S.T)

    def test_symmetry_rejects_asymmetric(self) -> None:
        """Constructor rejects clearly asymmetric matrices."""
        S = np.array([[1.0, 0.5], [0.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            Penalty(S)

    def test_rank_second_derivative(self) -> None:
        """Second-derivative penalty has rank k - 2."""
        k = 10
        S = _second_derivative_penalty(k)
        penalty = Penalty(S)
        assert penalty.rank == k - 2, f"Expected rank {k - 2}, got {penalty.rank}"

    def test_null_space_dim_second_derivative(self) -> None:
        """Second-derivative penalty has null space dim = 2."""
        k = 10
        S = _second_derivative_penalty(k)
        penalty = Penalty(S)
        assert penalty.null_space_dim == 2, (
            f"Expected null_space_dim 2, got {penalty.null_space_dim}"
        )

    def test_rank_full_rank_matrix(self) -> None:
        """Full-rank PSD matrix has rank = k."""
        rng = np.random.default_rng(42)
        k = 8
        S = _full_rank_penalty(k, rng)
        penalty = Penalty(S)
        assert penalty.rank == k
        assert penalty.null_space_dim == 0

    def test_rank_zero_matrix(self) -> None:
        """Zero matrix has rank 0 and null_space_dim = k."""
        k = 5
        S = np.zeros((k, k))
        penalty = Penalty(S)
        assert penalty.rank == 0
        assert penalty.null_space_dim == k

    def test_rank_explicit_override(self) -> None:
        """Explicit rank and null_space_dim override auto-computation."""
        k = 6
        S = _second_derivative_penalty(k)
        penalty = Penalty(S, rank=3, null_space_dim=3)
        assert penalty.rank == 3
        assert penalty.null_space_dim == 3

    def test_shape_property(self) -> None:
        """Shape property returns the matrix shape."""
        k = 7
        S = _second_derivative_penalty(k)
        penalty = Penalty(S)
        assert penalty.shape == (k, k)

    def test_size_property(self) -> None:
        """Size property returns k."""
        k = 7
        S = _second_derivative_penalty(k)
        penalty = Penalty(S)
        assert penalty.size == k

    def test_rejects_non_2d(self) -> None:
        """Constructor rejects non-2D input."""
        with pytest.raises(ValueError, match="2D"):
            Penalty(np.array([1.0, 2.0, 3.0]))

    def test_rejects_non_square(self) -> None:
        """Constructor rejects non-square matrix."""
        with pytest.raises(ValueError, match="square"):
            Penalty(np.ones((3, 4)))

    def test_repr(self) -> None:
        """repr contains useful information."""
        k = 5
        S = _second_derivative_penalty(k)
        penalty = Penalty(S)
        r = repr(penalty)
        assert "Penalty" in r
        assert str(k) in r


# ---------------------------------------------------------------------------
# Tests — CompositePenalty class
# ---------------------------------------------------------------------------


class TestCompositePenalty:
    """Tests for the CompositePenalty class."""

    def test_n_penalties_single(self) -> None:
        """Single penalty: n_penalties = 1."""
        S = _second_derivative_penalty(8)
        cp = CompositePenalty([Penalty(S)])
        assert cp.n_penalties == 1

    def test_n_penalties_multiple(self) -> None:
        """Multiple penalties: n_penalties matches list length."""
        rng = np.random.default_rng(42)
        k = 6
        penalties = [
            Penalty(_second_derivative_penalty(k)),
            Penalty(_full_rank_penalty(k, rng)),
            Penalty(_diagonal_penalty(k)),
        ]
        cp = CompositePenalty(penalties)
        assert cp.n_penalties == 3

    def test_weighted_penalty_known_values(self) -> None:
        """weighted_penalty with known S and lambda produces correct result."""
        k = 5
        S1 = np.eye(k) * 2.0
        S2 = np.eye(k) * 3.0
        p1 = Penalty(S1)
        p2 = Penalty(S2)
        cp = CompositePenalty([p1, p2])

        log_lambda = np.array([np.log(1.0), np.log(2.0)])
        result = cp.weighted_penalty(log_lambda)
        expected = 1.0 * S1 + 2.0 * S2
        np.testing.assert_allclose(
            result,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="weighted_penalty does not match expected linear combination.",
        )

    def test_weighted_penalty_single_penalty(self) -> None:
        """Single penalty: weighted_penalty = exp(log_lambda) * S."""
        k = 6
        S = _second_derivative_penalty(k)
        p = Penalty(S)
        cp = CompositePenalty([p])

        log_lambda = np.array([2.0])
        result = cp.weighted_penalty(log_lambda)
        expected = np.exp(2.0) * S
        np.testing.assert_allclose(
            result,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_weighted_penalty_zero_log_lambda(self) -> None:
        """Zero log_lambda means lambda = 1.0."""
        k = 5
        S1 = np.eye(k)
        S2 = np.eye(k) * 2.0
        cp = CompositePenalty([Penalty(S1), Penalty(S2)])

        log_lambda = np.array([0.0, 0.0])
        result = cp.weighted_penalty(log_lambda)
        expected = S1 + S2
        np.testing.assert_allclose(
            result,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_weighted_penalty_large_log_lambda(self) -> None:
        """Large log_lambda produces large weights without numerical issues."""
        k = 4
        S = np.eye(k)
        cp = CompositePenalty([Penalty(S)])

        log_lambda = np.array([20.0])
        result = cp.weighted_penalty(log_lambda)
        expected = np.exp(20.0) * S
        np.testing.assert_allclose(
            result,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_weighted_penalty_negative_log_lambda(self) -> None:
        """Negative log_lambda produces small weights."""
        k = 4
        S = np.eye(k)
        cp = CompositePenalty([Penalty(S)])

        log_lambda = np.array([-10.0])
        result = cp.weighted_penalty(log_lambda)
        expected = np.exp(-10.0) * S
        np.testing.assert_allclose(
            result,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_weighted_penalty_defaults_to_stored_params(self) -> None:
        """weighted_penalty with no argument uses stored log_smoothing_params."""
        k = 4
        S = np.eye(k)
        log_sp = np.array([1.5])
        cp = CompositePenalty([Penalty(S)], log_smoothing_params=log_sp)

        result = cp.weighted_penalty()
        expected = np.exp(1.5) * S
        np.testing.assert_allclose(
            result,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_weighted_penalty_result_is_psd(self) -> None:
        """Weighted penalty of PSD matrices is PSD."""
        rng = np.random.default_rng(99)
        k = 6
        S1 = _second_derivative_penalty(k)
        S2 = _full_rank_penalty(k, rng)
        cp = CompositePenalty([Penalty(S1), Penalty(S2)])

        log_lambda = np.array([1.0, -1.0])
        result = cp.weighted_penalty(log_lambda)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals >= -STRICT.atol), (
            f"Weighted penalty is not PSD: min eigenvalue = {np.min(eigvals):.2e}"
        )

    def test_weighted_penalty_result_is_symmetric(self) -> None:
        """Weighted penalty is symmetric."""
        rng = np.random.default_rng(99)
        k = 6
        S1 = _second_derivative_penalty(k)
        S2 = _full_rank_penalty(k, rng)
        cp = CompositePenalty([Penalty(S1), Penalty(S2)])

        log_lambda = np.array([1.0, -1.0])
        result = cp.weighted_penalty(log_lambda)
        np.testing.assert_allclose(
            result,
            result.T,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Weighted penalty is not symmetric.",
        )

    def test_weighted_penalty_wrong_shape_raises(self) -> None:
        """Wrong log_lambda shape raises ValueError."""
        k = 5
        cp = CompositePenalty([Penalty(np.eye(k)), Penalty(np.eye(k))])
        with pytest.raises(ValueError, match="shape"):
            cp.weighted_penalty(np.array([1.0]))  # Should be length 2

    def test_weighted_penalty_three_penalties(self) -> None:
        """Three penalties with distinct weights."""
        S1 = np.diag([1.0, 0.0, 0.0, 0.0])
        S2 = np.diag([0.0, 2.0, 0.0, 0.0])
        S3 = np.diag([0.0, 0.0, 3.0, 4.0])
        cp = CompositePenalty([Penalty(S1), Penalty(S2), Penalty(S3)])

        log_lambda = np.array([0.0, np.log(2.0), np.log(0.5)])
        result = cp.weighted_penalty(log_lambda)
        expected = 1.0 * S1 + 2.0 * S2 + 0.5 * S3
        np.testing.assert_allclose(
            result,
            expected,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


# ---------------------------------------------------------------------------
# Tests — Embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    """Tests for CompositePenalty.embed()."""

    def test_embed_block_position(self) -> None:
        """Embedded penalty has S_j in the correct block."""
        k = 3
        total_p = 10
        col_start = 4
        S_j = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 1.0]])

        S_global = CompositePenalty.embed(S_j, col_start, total_p)

        # Check the block
        block = S_global[col_start : col_start + k, col_start : col_start + k]
        np.testing.assert_allclose(
            block,
            S_j,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Embedded block does not match original S_j.",
        )

    def test_embed_zeros_elsewhere(self) -> None:
        """Embedded penalty has zeros outside the block."""
        k = 3
        total_p = 10
        col_start = 4
        S_j = np.eye(k) * 5.0

        S_global = CompositePenalty.embed(S_j, col_start, total_p)

        # Zero out the block and check everything else is zero
        S_check = S_global.copy()
        S_check[col_start : col_start + k, col_start : col_start + k] = 0.0
        np.testing.assert_allclose(
            S_check,
            np.zeros((total_p, total_p)),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Nonzeros found outside the embedded block.",
        )

    def test_embed_roundtrip(self) -> None:
        """Extract the block from embedded matrix and recover original."""
        rng = np.random.default_rng(77)
        k = 5
        total_p = 20
        col_start = 7
        A = rng.standard_normal((k, k))
        S_j = A.T @ A  # PSD

        S_global = CompositePenalty.embed(S_j, col_start, total_p)
        recovered = S_global[col_start : col_start + k, col_start : col_start + k]
        np.testing.assert_allclose(
            recovered,
            S_j,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Roundtrip: extracted block != original S_j.",
        )

    def test_embed_at_start(self) -> None:
        """Embedding at col_start=0 works correctly."""
        k = 4
        total_p = 10
        S_j = np.eye(k)

        S_global = CompositePenalty.embed(S_j, 0, total_p)
        assert S_global.shape == (total_p, total_p)
        np.testing.assert_allclose(
            S_global[:k, :k],
            S_j,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_embed_at_end(self) -> None:
        """Embedding at the end of the global matrix works."""
        k = 3
        total_p = 10
        col_start = total_p - k
        S_j = np.eye(k) * 2.0

        S_global = CompositePenalty.embed(S_j, col_start, total_p)
        np.testing.assert_allclose(
            S_global[col_start:, col_start:],
            S_j,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_embed_preserves_symmetry(self) -> None:
        """Embedded matrix is symmetric if S_j is symmetric."""
        k = 4
        total_p = 12
        col_start = 3
        S_j = _second_derivative_penalty(k)

        S_global = CompositePenalty.embed(S_j, col_start, total_p)
        np.testing.assert_allclose(
            S_global,
            S_global.T,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_embed_shape(self) -> None:
        """Embedded matrix has shape (total_p, total_p)."""
        S_j = np.eye(3)
        S_global = CompositePenalty.embed(S_j, 2, 8)
        assert S_global.shape == (8, 8)

    def test_embed_rejects_overflow(self) -> None:
        """Embedding that exceeds total_p raises ValueError."""
        S_j = np.eye(5)
        with pytest.raises(ValueError, match="extends beyond"):
            CompositePenalty.embed(S_j, 8, 10)

    def test_embed_rejects_negative_col_start(self) -> None:
        """Negative col_start raises ValueError."""
        S_j = np.eye(3)
        with pytest.raises(ValueError, match="non-negative"):
            CompositePenalty.embed(S_j, -1, 10)

    def test_embed_rejects_non_square(self) -> None:
        """Non-square S_j raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            CompositePenalty.embed(np.ones((3, 4)), 0, 10)


# ---------------------------------------------------------------------------
# Tests — Rank and null space
# ---------------------------------------------------------------------------


class TestRankAndNullSpace:
    """Tests for rank and null space dimension computation."""

    def test_second_derivative_rank_various_k(self) -> None:
        """Second-derivative penalty rank is k-2 for various k."""
        for k in [5, 8, 10, 15, 20, 50]:
            S = _second_derivative_penalty(k)
            penalty = Penalty(S)
            assert penalty.rank == k - 2, (
                f"k={k}: expected rank {k - 2}, got {penalty.rank}"
            )
            assert penalty.null_space_dim == 2, (
                f"k={k}: expected null_space_dim 2, got {penalty.null_space_dim}"
            )

    def test_identity_rank(self) -> None:
        """Identity matrix has full rank."""
        for k in [3, 5, 10]:
            S = np.eye(k)
            penalty = Penalty(S)
            assert penalty.rank == k
            assert penalty.null_space_dim == 0

    def test_rank_one_matrix(self) -> None:
        """Rank-1 matrix (outer product) has rank = 1."""
        v = np.array([1.0, 2.0, 3.0, 4.0])
        S = np.outer(v, v)
        penalty = Penalty(S)
        assert penalty.rank == 1
        assert penalty.null_space_dim == 3

    def test_block_diagonal_rank(self) -> None:
        """Block diagonal penalty: rank = sum of block ranks."""
        k = 8
        # Create a penalty with rank 4 (two blocks of rank 2 each)
        S = np.zeros((k, k))
        block = _second_derivative_penalty(4)  # rank 2
        S[:4, :4] = block
        S[4:, 4:] = block
        penalty = Penalty(S)
        assert penalty.rank == 4
        assert penalty.null_space_dim == 4


# ---------------------------------------------------------------------------
# Tests — Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for penalty classes."""

    def test_single_penalty_in_composite(self) -> None:
        """CompositePenalty with one penalty works correctly."""
        k = 5
        S = _second_derivative_penalty(k)
        cp = CompositePenalty([Penalty(S)])
        assert cp.n_penalties == 1
        result = cp.weighted_penalty(np.array([0.0]))
        np.testing.assert_allclose(result, S, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_empty_penalty_list_raises(self) -> None:
        """Empty penalty list raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            CompositePenalty([])

    def test_log_smoothing_params_default(self) -> None:
        """Default log_smoothing_params are zeros."""
        k = 4
        cp = CompositePenalty([Penalty(np.eye(k))])
        np.testing.assert_array_equal(cp.log_smoothing_params, np.array([0.0]))

    def test_log_smoothing_params_custom(self) -> None:
        """Custom log_smoothing_params are stored correctly."""
        k = 4
        log_sp = np.array([1.0, -1.0])
        cp = CompositePenalty(
            [Penalty(np.eye(k)), Penalty(np.eye(k))],
            log_smoothing_params=log_sp,
        )
        np.testing.assert_array_equal(cp.log_smoothing_params, log_sp)

    def test_log_smoothing_params_wrong_shape_raises(self) -> None:
        """Wrong shape for log_smoothing_params raises ValueError."""
        k = 4
        with pytest.raises(ValueError, match="shape"):
            CompositePenalty(
                [Penalty(np.eye(k))],
                log_smoothing_params=np.array([1.0, 2.0]),
            )

    def test_very_large_log_lambda(self) -> None:
        """Very large log_lambda (lambda ~ 1e87) does not produce inf/nan."""
        k = 3
        S = np.eye(k)
        cp = CompositePenalty([Penalty(S)])
        log_lambda = np.array([200.0])
        result = cp.weighted_penalty(log_lambda)
        assert np.all(np.isfinite(result)), "Result contains inf or nan."

    def test_very_negative_log_lambda(self) -> None:
        """Very negative log_lambda (lambda ~ 0) produces near-zero penalty."""
        k = 3
        S = np.eye(k)
        cp = CompositePenalty([Penalty(S)])
        log_lambda = np.array([-200.0])
        result = cp.weighted_penalty(log_lambda)
        assert np.all(np.isfinite(result)), "Result contains inf or nan."
        np.testing.assert_allclose(
            result,
            np.zeros((k, k)),
            atol=STRICT.atol,
        )

    def test_small_matrix_2x2(self) -> None:
        """2x2 penalty matrix works."""
        S = np.array([[1.0, -0.5], [-0.5, 1.0]])
        penalty = Penalty(S)
        assert penalty.shape == (2, 2)
        assert penalty.rank == 2
        assert penalty.null_space_dim == 0

    def test_1x1_penalty(self) -> None:
        """1x1 penalty matrix works."""
        S = np.array([[5.0]])
        penalty = Penalty(S)
        assert penalty.shape == (1, 1)
        assert penalty.rank == 1
        assert penalty.null_space_dim == 0

    def test_repr_composite(self) -> None:
        """CompositePenalty repr contains useful info."""
        k = 4
        cp = CompositePenalty([Penalty(np.eye(k)), Penalty(np.eye(k))])
        r = repr(cp)
        assert "CompositePenalty" in r
        assert "n_penalties=2" in r


# ---------------------------------------------------------------------------
# Tests — No JAX imports (Phase 1 boundary guard)
# ---------------------------------------------------------------------------


class TestNoJaxImport:
    """Verify that pymgcv.penalties does not trigger JAX import."""

    def test_penalties_import_no_jax(self) -> None:
        """Importing pymgcv.penalties must not cause jax to be imported."""
        # Remove jax and pymgcv modules from sys.modules
        modules_to_remove = [
            key
            for key in sys.modules
            if key == "jax" or key.startswith("jax.") or key.startswith("pymgcv.")
        ]
        saved = {key: sys.modules.pop(key) for key in modules_to_remove}

        try:
            importlib.import_module("pymgcv.penalties")
            assert "jax" not in sys.modules, (
                "Importing pymgcv.penalties triggered a jax import. "
                "Phase 1 modules must not depend on JAX."
            )
        finally:
            # Restore original sys.modules state
            for key in list(sys.modules):
                if key.startswith("pymgcv."):
                    sys.modules.pop(key, None)
            sys.modules.update(saved)

    def test_penalty_module_import_no_jax(self) -> None:
        """Importing pymgcv.penalties.penalty must not cause jax import."""
        modules_to_remove = [
            key
            for key in sys.modules
            if key == "jax" or key.startswith("jax.") or key.startswith("pymgcv.")
        ]
        saved = {key: sys.modules.pop(key) for key in modules_to_remove}

        try:
            importlib.import_module("pymgcv.penalties.penalty")
            assert "jax" not in sys.modules, (
                "Importing pymgcv.penalties.penalty triggered a jax import."
            )
        finally:
            for key in list(sys.modules):
                if key.startswith("pymgcv."):
                    sys.modules.pop(key, None)
            sys.modules.update(saved)
