"""JAX utilities: array dispatch, device transfer, and linear algebra.

Provides:
- ``array_module`` / ``is_jax_array``: runtime NumPy/JAX dispatch for
  backend-agnostic code (links, families).
- ``to_jax`` / ``to_numpy``: Phase 1→2 and Phase 2→3 device transfer
  boundaries (design.md §1.3).
- ``cho_factor``: plain Cholesky with a diagonal-relative fallback jitter.
- ``penalized_cholesky`` / ``penalized_solve``: penalized Hessian solves
  for PIRLS.
- ``numerical_rank``: rank estimation via pivoted QR.

All linalg functions are JIT-compiled, float64, and pure. Other JAX linalg
functions (cho_solve, slogdet, solve_triangular) should be imported directly
from ``jax.scipy.linalg`` / ``jax.numpy.linalg``.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import overload

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np

__all__ = [
    "array_module",
    "block_log_det_S",
    "build_S_lambda",
    "cho_factor",
    "is_jax_array",
    "numerical_rank",
    "penalized_cholesky",
    "penalized_solve",
    "stable_log_pseudo_det",
    "to_jax",
    "to_numpy",
]

# ---------------------------------------------------------------------------
# Device transfer (Phase 1→2 and Phase 2→3 boundaries, design.md §1.3)
# ---------------------------------------------------------------------------


@overload
def to_jax(array: np.ndarray, /, *, device: jax.Device | None = ...) -> jax.Array: ...


@overload
def to_jax(
    *arrays: np.ndarray, device: jax.Device | None = ...
) -> tuple[jax.Array, ...]: ...


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
    if not arrays:
        raise TypeError("to_jax() requires at least one array argument.")
    converted = tuple(
        jax.device_put(jnp.asarray(a, dtype=jnp.float64), device) for a in arrays
    )
    if len(converted) == 1:
        return converted[0]
    return converted


@overload
def to_numpy(array: jax.Array, /) -> np.ndarray: ...


@overload
def to_numpy(*arrays: jax.Array) -> tuple[np.ndarray, ...]: ...


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
    if not arrays:
        raise TypeError("to_numpy() requires at least one array argument.")
    converted = tuple(np.asarray(a) for a in arrays)
    if len(converted) == 1:
        return converted[0]
    return converted


# ---------------------------------------------------------------------------
# Array dispatch
# ---------------------------------------------------------------------------


def is_jax_array(x: object) -> bool:
    """Check if x is a JAX array."""
    return isinstance(x, jax.Array)


def array_module(x: object) -> types.ModuleType:
    """Return ``jax.numpy`` if *x* is a JAX array, else ``numpy``."""
    if is_jax_array(x):
        return jnp
    return np


@jax.custom_jvp
def _materialize_source_jax(value: jax.Array) -> jax.Array:
    return jnp.nextafter(value, value)


@_materialize_source_jax.defjvp
def _materialize_source_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (value,) = primals
    (tangent,) = tangents
    return _materialize_source_jax(value), tangent


def _materialize_source_operation(
    value: jax.Array | np.ndarray,
) -> jax.Array | np.ndarray:
    """Preserve one rounded source temporary with its identity derivative.

    nextafter(x, x) is exactly x. Its bit-level lowering prevents XLA from
    contracting arithmetic across a source vector-operation boundary; the
    custom JVP preserves the derivative of this identity operation.
    """
    if is_jax_array(value):
        return _materialize_source_jax(value)
    return np.nextafter(value, value)


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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StableLogDetTerms:
    """Host ``gam.reparam`` log|S|+ terms of one multi-penalty block at ``rho0``.

    ``det``, ``det1`` and ``det2`` are ``get_stableS``'s log|S|+ and its
    gradient and Hessian in the block's log smoothing parameters, computed on
    the host at ``rho0`` (pinned gam.fit3.r:144 calls gam.reparam at every
    evaluation; REML uses rp$det, rp$det1, rp$det2 at 616-621).  In the
    criterion they enter as ``det + det1.d + d'det2 d / 2`` with
    ``d = rho - rho0``: at ``rho0`` its value, ``jax.grad`` and
    ``jax.hessian`` are exactly det, det1 and det2.  Evaluated at any other
    rho the term is NaN, so no code can silently use it away from the host's
    point.  ``det_change`` carries ``det(rho_trial) - det(rho0)`` for the
    Gaussian score-change comparison (zero otherwise).
    """

    rho0: jax.Array
    det: jax.Array
    det1: jax.Array
    det2: jax.Array
    det_change: jax.Array

    def tree_flatten(self):
        return (self.rho0, self.det, self.det1, self.det2, self.det_change), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)

    def log_det(self, rho_block: jax.Array) -> jax.Array:
        """The quadratic expansion at ``rho_block``; NaN unless it is ``rho0``."""
        delta = rho_block - self.rho0
        value = self.det + self.det1 @ delta + 0.5 * delta @ self.det2 @ delta
        at_point = jnp.all(jax.lax.stop_gradient(delta) == 0.0)
        return jnp.where(at_point, value, jnp.nan)


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
    multi_block_proj_S : tuple[tuple[jax.Array, ...] | StableLogDetTerms, ...]
        Range-space-projected penalty matrices for each multi-penalty block,
        or that block's host :class:`StableLogDetTerms` at the current rho.

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

    # Multi-penalty blocks: host get_stableS terms when supplied, otherwise
    # the scaled slogdet in the fixed range basis.
    for i, sp_indices in enumerate(multi_block_sp_indices):
        rho_block = jnp.stack([log_lambda[j] for j in sp_indices])
        if isinstance(multi_block_proj_S[i], StableLogDetTerms):
            log_det = log_det + multi_block_proj_S[i].log_det(rho_block)
            continue
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
    # -1e10 sentinel: if the projected matrix is not PD (sign ≤ 0), return
    # a large negative value.  This acts as a soft penalty in the REML
    # criterion, steering the optimizer away from degenerate regions.
    # Raising an exception is not possible inside JIT-compiled code.
    return jnp.where(sign > 0, logdet, -1e10)


_CHOLESKY_SMALL_RELATIVE_JITTER = 1e-12
_CHOLESKY_LARGE_RELATIVE_JITTER = 1e-6


@jax.jit
def cho_factor(
    H: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Plain Cholesky factorization, regularized only when it fails.

    mgcv never adds a diagonal jitter to a penalized fit: ``pls_fit1`` and
    ``gdiPK`` (gdi.c) factor by pivoted QR with explicit rank handling.  A
    jitter proportional to ``trace(H)/p`` is dominated by the penalty at large
    smoothing parameters and then swamps the data curvature of unpenalized
    directions, so the factor here is the *plain* Cholesky of ``H`` whenever
    that is finite.  Only if it fails (``H`` not numerically positive
    definite) is a diagonal-relative jitter ``eps * |diag(H)|`` added, first
    ``eps = 1e-12`` and then ``eps = 1e-6``.  That regularization is
    scale-free in every coordinate (a zero diagonal entry uses unit scale).

    A plain factor whose smallest relative pivot ``L_ii**2 / |H_ii|`` is below
    ``1e-12`` counts as failed too: ``H`` is then numerically singular (for
    example aliased unpenalized columns, which ``pls_fit1`` would drop by its
    rank test), the pivot is roundoff and the solve would be garbage in that
    direction.  The relative pivot is scale-free, so a large smoothing
    parameter alone never triggers the fallback.

    ``jax.lax.cond`` selects the branch, which is gradient-safe (only the
    executed branch is differentiated).

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
        Relative diagonal jitter level used: ``0`` for the plain factor,
        otherwise the fallback ``eps``.  Callers surface a nonzero level as an
        explicit regularization diagnostic.
    """
    L_plain = jnp.linalg.cholesky(H)
    magnitude = jnp.abs(jnp.diag(H))
    scale = jnp.where(magnitude > 0.0, magnitude, 1.0)
    relative_pivot = jnp.diag(L_plain) ** 2 / scale
    plain_ok = jnp.all(jnp.isfinite(L_plain)) & jnp.all(
        relative_pivot >= _CHOLESKY_SMALL_RELATIVE_JITTER
    )
    reference = jnp.diag(scale)
    zero = jnp.zeros((), dtype=H.dtype)

    def _use_plain(_: None) -> tuple[jax.Array, jax.Array]:
        return L_plain, zero

    def _use_fallback(_: None) -> tuple[jax.Array, jax.Array]:
        small = jnp.asarray(_CHOLESKY_SMALL_RELATIVE_JITTER, dtype=H.dtype)
        large = jnp.asarray(_CHOLESKY_LARGE_RELATIVE_JITTER, dtype=H.dtype)
        L_small = jnp.linalg.cholesky(H + small * reference)

        def _use_large(_: None) -> tuple[jax.Array, jax.Array]:
            return jnp.linalg.cholesky(H + large * reference), large

        def _use_small(_: None) -> tuple[jax.Array, jax.Array]:
            return L_small, small

        return jax.lax.cond(
            jnp.all(jnp.isfinite(L_small)), _use_small, _use_large, None
        )

    return jax.lax.cond(plain_ok, _use_plain, _use_fallback, None)


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
        Relative diagonal jitter level of the ``cho_factor`` fallback, or
        ``0`` when the plain factor succeeded.
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
        Relative diagonal jitter level of the ``cho_factor`` fallback, or
        ``0`` when the plain factor succeeded.
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
