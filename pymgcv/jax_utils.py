"""JAX utilities: array dispatch, device transfer, and linear algebra.

Provides:
- ``array_module`` / ``is_jax_array``: runtime NumPy/JAX dispatch for
  backend-agnostic code (links, families).
- ``to_jax`` / ``to_numpy``: Phase 1→2 and Phase 2→3 device transfer
  boundaries (design.md §1.3).
- ``cho_factor``: Cholesky with scale-relative jitter (design.md §4.8).
- ``penalized_cholesky`` / ``penalized_solve``: penalized Hessian solves
  for PIRLS.
- ``numerical_rank``: rank estimation via pivoted QR.

All linalg functions are JIT-compiled, float64, and pure. Other JAX linalg
functions (cho_solve, slogdet, solve_triangular) should be imported directly
from ``jax.scipy.linalg`` / ``jax.numpy.linalg``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np

# ---------------------------------------------------------------------------
# Device transfer (Phase 1→2 and Phase 2→3 boundaries, design.md §1.3)
# ---------------------------------------------------------------------------


def to_jax(
    *arrays: np.ndarray,
    device: jax.Device | None = None,
) -> tuple[jax.Array, ...] | jax.Array:
    """Transfer NumPy arrays to a JAX device.

    This is the Phase 1→2 boundary: after basis/penalty construction
    (NumPy, CPU) and before PIRLS/REML (JAX, JIT-compiled).

    Parameters
    ----------
    *arrays : np.ndarray
        One or more NumPy arrays to transfer.
    device : jax.Device, optional
        Target device. If ``None``, uses JAX's default device
        (GPU/Metal if available, otherwise CPU).

    Returns
    -------
    jax.Array or tuple[jax.Array, ...]
        Single array if one input, tuple if multiple.

    Examples
    --------
    >>> X_jax, y_jax, S_jax = to_jax(X, y, S_lambda)
    >>> result = pirls_loop(X_jax, y_jax, beta_init, S_jax, family)
    """
    converted = tuple(
        jax.device_put(jnp.asarray(a, dtype=jnp.float64), device) for a in arrays
    )
    if len(converted) == 1:
        return converted[0]
    return converted


def to_numpy(*arrays: jax.Array) -> tuple[np.ndarray, ...] | np.ndarray:
    """Transfer JAX arrays back to NumPy on CPU.

    This is the Phase 2→3 boundary: after PIRLS/REML (JAX) and
    before post-estimation (NumPy, CPU).

    Parameters
    ----------
    *arrays : jax.Array
        One or more JAX arrays to transfer.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, ...]
        Single array if one input, tuple if multiple.
    """
    converted = tuple(np.asarray(a) for a in arrays)
    if len(converted) == 1:
        return converted[0]
    return converted


# ---------------------------------------------------------------------------
# Array dispatch
# ---------------------------------------------------------------------------


def is_jax_array(x: object) -> bool:
    """Check if x is a JAX array."""
    return type(x).__module__.startswith(("jax", "jaxlib"))


def array_module(x: object):
    """Return ``jax.numpy`` if *x* is a JAX array, else ``numpy``."""
    if is_jax_array(x):
        return jnp
    return np


# ---------------------------------------------------------------------------
# Linear algebra primitives
# ---------------------------------------------------------------------------


def build_S_lambda(
    log_lambda: jax.Array,
    S_list: tuple[jax.Array, ...],
    p: int,
) -> jax.Array:
    """Build S_lambda = sum_j exp(log_lambda[j]) * S_j.

    Pure JAX, differentiable w.r.t. log_lambda.

    Parameters
    ----------
    log_lambda : jax.Array, shape (m,)
        Log smoothing parameters.
    S_list : tuple[jax.Array, ...]
        Per-penalty (p, p) matrices.
    p : int
        Number of coefficients.

    Returns
    -------
    jax.Array, shape (p, p)
        Combined weighted penalty matrix.
    """
    S_lambda = jnp.zeros((p, p))
    for j, S_j in enumerate(S_list):
        S_lambda = S_lambda + jnp.exp(log_lambda[j]) * S_j
    return S_lambda


def log_pseudo_det(S: jax.Array, n_zero: int = 0) -> jax.Array:
    """Log pseudo-determinant of S (product of non-zero eigenvalues).

    Adds a tiny asymmetric diagonal perturbation before eigendecomposition
    to break eigenvalue degeneracy.  Without this, ``jax.hessian`` through
    ``eigvalsh`` produces NaN when eigenvalues are degenerate — which
    occurs for factor-by models whose block-diagonal penalties share
    identical eigenvalue structure.  The perturbation (scale ~1e-14) is
    negligible relative to the eigenvalues.

    Parameters
    ----------
    S : jax.Array, shape (p, p)
        Symmetric positive semi-definite matrix.
    n_zero : int
        Number of eigenvalues known to be zero (the penalty null space
        dimension ``Mp``).  When provided, the bottom ``n_zero``
        eigenvalues are excluded by index rather than by a relative
        threshold.  This is critical for multi-penalty models where
        eigenvalues span many orders of magnitude — a relative
        threshold ``1e-10 * max(eig)`` can incorrectly exclude
        eigenvalues from weaker penalties when one smoothing parameter
        dominates.

    Returns
    -------
    jax.Array, scalar
        Log of the product of non-zero eigenvalues.
    """
    p = S.shape[0]
    # Asymmetric diagonal jitter breaks eigenvalue degeneracy so that
    # second derivatives through eigvalsh remain finite.
    #
    # The jitter scales proportionally to each diagonal entry of S,
    # ensuring the relative perturbation is ~1e-14 regardless of the
    # eigenvalue magnitude.  This is critical for multi-penalty models
    # (tensor products) where eigenvalues span many orders of magnitude:
    # a global scale ``1e-14 * max|S|`` would corrupt the small eigenvalues
    # from the weaker penalty when one smoothing parameter dominates.
    diag_S = jnp.abs(jnp.diag(S))
    # Floor prevents zero jitter in the null space (where diag ≈ 0).
    # The floor is negligible relative to any non-null eigenvalue.
    per_entry_scale = jnp.maximum(diag_S, 1e-30)
    jitter = 1e-14 * per_entry_scale * jnp.arange(1, p + 1, dtype=S.dtype)
    S = S + jnp.diag(jitter)
    eigs = jnp.linalg.eigvalsh(S)  # ascending order
    safe_eigs = jnp.maximum(eigs, 1e-30)
    # Select top p - n_zero eigenvalues (the non-null ones).
    # eigvalsh returns ascending order, so null eigenvalues are first.
    mask = jnp.arange(p) >= n_zero
    return jnp.sum(jnp.where(mask, jnp.log(safe_eigs), 0.0))


def block_log_det_S(
    log_lambda: jax.Array,
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    singleton_eig_constants: jax.Array,
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
) -> jax.Array:
    """Block-structured log pseudo-determinant of S_lambda.

    For models where penalties from different smooths occupy non-overlapping
    columns, S_lambda is block-diagonal and log|S+| decomposes as a sum
    over blocks.

    Singleton blocks (one penalty per smooth) use the exact formula
    ``log|S+| = rank * rho + const``, whose derivative w.r.t. rho is
    exactly ``rank`` — no matrix operations, zero numerical error.

    Multi-penalty blocks (tensor products) factor out ``exp(rho_max)``
    for numerical stability, then use slogdet on a well-conditioned matrix.

    Parameters
    ----------
    log_lambda : jax.Array, shape (m,)
        Log smoothing parameters.
    singleton_sp_indices : tuple[int, ...]
        Index into ``log_lambda`` for each singleton block.
    singleton_ranks : tuple[int, ...]
        Rank of each singleton's penalty.
    singleton_eig_constants : jax.Array, shape (n_singletons,)
        Precomputed ``sum(log(nonzero_eigenvalues))`` for each singleton.
    multi_block_sp_indices : tuple[tuple[int, ...], ...]
        ``log_lambda`` indices for each multi-penalty block.
    multi_block_ranks : tuple[int, ...]
        Combined penalty rank for each multi-penalty block.
    multi_block_proj_S : tuple[tuple[jax.Array, ...], ...]
        Range-space-projected penalty matrices for each multi-penalty block.

    Returns
    -------
    jax.Array, scalar
        Log pseudo-determinant of S_lambda.
    """
    log_det = jnp.array(0.0)

    # Singleton blocks: exact formula (derivative = rank, no numerical error)
    for i, sp_idx in enumerate(singleton_sp_indices):
        log_det = (
            log_det
            + singleton_ranks[i] * log_lambda[sp_idx]
            + singleton_eig_constants[i]
        )

    # Multi-penalty blocks: scaled slogdet for well-conditioned derivatives
    for i, sp_indices in enumerate(multi_block_sp_indices):
        rho_block = jnp.stack([log_lambda[j] for j in sp_indices])
        rho_max = jnp.max(rho_block)
        S_adj = jnp.zeros_like(multi_block_proj_S[i][0])
        for j, S_j_proj in enumerate(multi_block_proj_S[i]):
            S_adj = S_adj + jnp.exp(rho_block[j] - rho_max) * S_j_proj
        sign, logdet = jnp.linalg.slogdet(S_adj)
        log_det = (
            log_det
            + multi_block_ranks[i] * rho_max
            + jnp.where(sign > 0, logdet, -1e10)
        )

    return log_det


def stable_log_pseudo_det(S: jax.Array, U_range: jax.Array) -> jax.Array:
    """Log pseudo-determinant via projection into range space + slogdet.

    Projects S into its range space using a precomputed orthogonal
    basis, then computes slogdet on the resulting full-rank PD matrix.
    This avoids eigendecomposition of rank-deficient matrices, which
    has unstable derivatives under JAX autodiff.

    Parameters
    ----------
    S : jax.Array, shape (p, p)
        Symmetric positive semi-definite matrix.
    U_range : jax.Array, shape (p, r)
        Orthogonal basis for the range space of the total penalty.

    Returns
    -------
    jax.Array, scalar
        Log of the product of non-zero eigenvalues.
    """
    S_proj = U_range.T @ S @ U_range  # (r, r), full rank PD
    sign, logdet = jnp.linalg.slogdet(S_proj)
    return jnp.where(sign > 0, logdet, -1e10)


@jax.jit
def cho_factor(
    H: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Cholesky factorization with two-level scale-relative jitter.

    Follows the stabilization strategy from design.md §4.8 (lines 3190-3212):
    always applies a small jitter ``eps_small * trace(H)/p``, and falls back
    to a larger jitter ``eps_large * trace(H)/p`` if the first attempt
    produces NaN (indicating non-positive-definiteness under JIT).

    Uses ``jax.lax.cond`` for the fallback, which is gradient-safe
    (only differentiates the executed branch).

    Parameters
    ----------
    H : jax.Array, shape (p, p)
        Symmetric positive-definite (or nearly PD) matrix.

    Returns
    -------
    L : jax.Array, shape (p, p)
        Lower-triangular Cholesky factor. Pass ``(L, True)`` to
        ``jax.scipy.linalg.cho_solve``.
    jitter_applied : jax.Array
        Scalar jitter level that was actually used (for diagnostics).
    """
    p = H.shape[0]
    I_p = jnp.eye(p)
    trace_H = jnp.trace(H)

    eps_small = jnp.maximum(1e-12 * trace_H / p, 1e-10)
    eps_large = jnp.maximum(1e-6 * trace_H / p, 1e-10)

    L_small = jnp.linalg.cholesky(H + eps_small * I_p)
    has_nan = jnp.any(jnp.isnan(L_small))

    def _use_large_jitter(_: None) -> tuple[jax.Array, jax.Array]:
        L_large = jnp.linalg.cholesky(H + eps_large * I_p)
        return L_large, eps_large

    def _use_small_jitter(_: None) -> tuple[jax.Array, jax.Array]:
        return L_small, eps_small

    L, jitter_applied = jax.lax.cond(
        has_nan, _use_large_jitter, _use_small_jitter, None
    )

    return L, jitter_applied


@jax.jit
def penalized_cholesky(
    XtWX: jax.Array, S_lambda: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Cholesky factor of the penalized Hessian ``XtWX + S_lambda``.

    Parameters
    ----------
    XtWX : jax.Array, shape (p, p)
        Weighted cross-product matrix.
    S_lambda : jax.Array, shape (p, p)
        Combined weighted penalty matrix.

    Returns
    -------
    L : jax.Array, shape (p, p)
        Lower-triangular Cholesky factor.
    jitter_applied : jax.Array
        Jitter level used (for diagnostics).
    """
    H = XtWX + S_lambda
    return cho_factor(H)


@jax.jit
def penalized_solve(
    XtWX: jax.Array, S_lambda: jax.Array, rhs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve ``(XtWX + S_lambda) @ beta = rhs`` via penalized Cholesky.

    This is the core PIRLS solve: ``beta = (XtWX + S_lambda)^{-1} rhs``.

    Parameters
    ----------
    XtWX : jax.Array, shape (p, p)
        Weighted cross-product matrix.
    S_lambda : jax.Array, shape (p, p)
        Combined weighted penalty matrix.
    rhs : jax.Array, shape (p,) or (p, k)
        Right-hand side (typically ``X^T W z``).

    Returns
    -------
    beta : jax.Array, shape (p,) or (p, k)
        Solution coefficients.
    L : jax.Array, shape (p, p)
        Lower-triangular Cholesky factor, retained for downstream use
        (e.g. EDF computation, log-determinant).
    jitter_applied : jax.Array
        Jitter level used.
    """
    L, jitter_applied = penalized_cholesky(XtWX, S_lambda)
    beta = jsla.cho_solve((L, True), rhs)
    return beta, L, jitter_applied


@jax.jit(static_argnames=("tol",))
def numerical_rank(A: jax.Array, tol: float | None = None) -> jax.Array:
    """Estimate the numerical rank of a matrix via pivoted QR.

    Parameters
    ----------
    A : jax.Array, shape (m, n)
    tol : float, optional
        Threshold for diagonal elements of R. Defaults to
        ``eps * max(m, n) * max(|diag(R)|)``.

    Returns
    -------
    rank : jax.Array
        Scalar integer estimate of rank.
    """
    _, R, _ = jsla.qr(A, pivoting=True)
    m, n = A.shape
    diag_R = jnp.abs(jnp.diag(R[: min(m, n), : min(m, n)]))
    if tol is None:
        tol = jnp.finfo(A.dtype).eps * max(m, n) * jnp.max(diag_R)
    return jnp.sum(diag_R > tol)
