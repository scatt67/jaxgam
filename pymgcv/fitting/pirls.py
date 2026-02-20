"""Penalized iteratively reweighted least squares (PIRLS) inner loop.

Given fixed smoothing parameters (encoded in ``S_lambda``), PIRLS finds
the penalized maximum likelihood coefficients by iterating a weighted
least-squares solve with step-halving on penalized deviance.

Fisher scoring only (not full Newton).

The loop is implemented with ``jax.lax.while_loop`` so the entire
iteration compiles to a single fused XLA kernel when JIT-compiled.

Design doc reference: Section 7.2
R source reference: gam.fit3() lines 296-468
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp

from pymgcv.families.base import ExponentialFamily
from pymgcv.jax_utils import penalized_solve


@dataclass(frozen=True)
class PIRLSResult:
    """Result of the PIRLS inner loop.

    Attributes
    ----------
    coefficients : jax.Array, shape (p,)
        Fitted coefficient vector.
    mu : jax.Array, shape (n,)
        Fitted mean response.
    eta : jax.Array, shape (n,)
        Linear predictor (including offset).
    deviance : jax.Array
        Scalar unpenalized deviance.
    penalized_deviance : jax.Array
        Scalar penalized deviance: deviance + beta^T S_lambda beta.
    n_iter : jax.Array
        Number of iterations used.
    converged : jax.Array
        Whether the convergence criterion was met.
    scale : jax.Array
        Estimated scale parameter.
    XtWX : jax.Array, shape (p, p)
        Final weighted cross-product matrix (needed for REML).
    L : jax.Array, shape (p, p)
        Final Cholesky factor of penalized Hessian (needed for REML).
    working_weights : jax.Array, shape (n,)
        Final working weights.
    """

    coefficients: jax.Array
    mu: jax.Array
    eta: jax.Array
    deviance: jax.Array
    penalized_deviance: jax.Array
    n_iter: jax.Array
    converged: jax.Array
    scale: jax.Array
    XtWX: jax.Array
    L: jax.Array
    working_weights: jax.Array


# Register as JAX pytree so PIRLSResult can be returned from jax.jit
_PIRLS_FIELDS = [f.name for f in fields(PIRLSResult)]

jax.tree_util.register_pytree_node(
    PIRLSResult,
    lambda r: ([getattr(r, f) for f in _PIRLS_FIELDS], None),
    lambda _, children: PIRLSResult(**dict(zip(_PIRLS_FIELDS, children))),
)


def _pirls_step(
    X: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    beta: jax.Array,
    mu: jax.Array,
    offset: jax.Array,
    S_lambda: jax.Array,
    family: ExponentialFamily,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One PIRLS iteration: working quantities → penalized WLS solve.

    Parameters
    ----------
    X : jax.Array, shape (n, p)
    y : jax.Array, shape (n,)
    wt : jax.Array, shape (n,)
    beta : jax.Array, shape (p,)
    mu : jax.Array, shape (n,)
    offset : jax.Array, shape (n,)
    S_lambda : jax.Array, shape (p, p)
    family : ExponentialFamily

    Returns
    -------
    beta_new : jax.Array, shape (p,)
    XtWX : jax.Array, shape (p, p)
    L : jax.Array, shape (p, p)
    W : jax.Array, shape (n,)
    """
    eta_no_offset = X @ beta

    W = family.working_weights(mu, wt)
    W = jnp.clip(W, 1e-10, 1e10)

    z = family.working_response(y, mu, eta_no_offset)

    W_sqrt = jnp.sqrt(W)
    WX = W_sqrt[:, None] * X
    XtWX = WX.T @ WX
    XtWz = WX.T @ (W_sqrt * z)

    beta_new, L, _ = penalized_solve(XtWX, S_lambda, XtWz)
    return beta_new, XtWX, L, W


def _penalized_deviance(
    beta: jax.Array,
    mu: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    S_lambda: jax.Array,
    family: ExponentialFamily,
) -> jax.Array:
    """Compute penalized deviance: dev(y, mu, wt) + beta^T S_lambda beta."""
    dev = family.dev_resids(y, mu, wt)
    penalty = beta @ S_lambda @ beta
    return dev + penalty


def pirls_loop(
    X: jax.Array,
    y: jax.Array,
    beta_init: jax.Array,
    S_lambda: jax.Array,
    family: ExponentialFamily,
    wt: jax.Array | None = None,
    offset: jax.Array | None = None,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> PIRLSResult:
    """Run the PIRLS inner loop to convergence.

    Finds coefficients ``beta`` that minimize the penalized deviance::

        dev(y, mu, wt) + beta^T @ S_lambda @ beta

    where ``mu = linkinv(X @ beta + offset)``.

    All array inputs must be JAX arrays on the target device.
    Use ``jax_utils.to_jax()`` to transfer NumPy arrays before
    calling this function (design.md §1.3 Phase 1→2 boundary).

    Parameters
    ----------
    X : jax.Array, shape (n, p)
        Model matrix (on device).
    y : jax.Array, shape (n,)
        Response values (on device).
    beta_init : jax.Array, shape (p,)
        Starting coefficients (on device).
    S_lambda : jax.Array, shape (p, p)
        Combined weighted penalty matrix (on device).
    family : ExponentialFamily
        Family with link attached.
    wt : jax.Array, shape (n,), optional
        Prior weights (on device). Defaults to ones.
    offset : jax.Array, shape (n,), optional
        Offset term (on device). Defaults to zeros.
    max_iter : int
        Maximum PIRLS iterations.
    tol : float
        Convergence tolerance for both deviance and coefficient criteria.

    Returns
    -------
    PIRLSResult
        Fitted result with coefficients, diagnostics, and quantities
        needed by the REML outer loop. All arrays are JAX arrays
        on device; use ``jax_utils.to_numpy()`` to transfer back
        to CPU for post-estimation.
    """
    n, p = X.shape

    if wt is None:
        wt = jnp.ones(n)
    if offset is None:
        offset = jnp.zeros(n)

    # Initial mu from beta_init
    eta_init = X @ beta_init + offset
    mu_init = family.link.inverse(eta_init)

    # Initialize loop state
    # (i, beta, beta_old, mu, pen_dev, pen_dev_prev,
    #  converged, XtWX, L, W)
    init_state = (
        jnp.int32(0),  # i
        beta_init,  # beta
        jnp.zeros_like(beta_init),  # beta_old
        mu_init,  # mu
        jnp.array(jnp.inf),  # pen_dev
        jnp.array(jnp.inf),  # pen_dev_prev
        jnp.bool_(False),  # converged
        jnp.zeros((p, p)),  # XtWX
        jnp.eye(p),  # L
        jnp.ones(n),  # W
    )

    def _cond(state):
        i, _, _, _, _, _, converged, _, _, _ = state
        return (i < max_iter) & (~converged)

    def _body(state):
        i, beta, _beta_old, mu, pen_dev, pen_dev_prev, _converged, _XtWX, _L, _W = state

        # One PIRLS step
        beta_new, XtWX, L, W = _pirls_step(X, y, wt, beta, mu, offset, S_lambda, family)

        # Step-halving on penalized deviance
        # Use nested while_loop for JIT compatibility
        is_first_iter = i == 0

        # Initial step-halving state: (k, step, beta_try, pen_dev_try, mu_try, accepted)
        eta_new = X @ beta_new + offset
        mu_new = family.link.inverse(eta_new)
        pen_dev_new = _penalized_deviance(beta_new, mu_new, y, wt, S_lambda, family)

        # First iteration: unconditionally accept
        first_ok = is_first_iter & jnp.isfinite(pen_dev_new)
        subsequent_ok = (
            (~is_first_iter)
            & jnp.isfinite(pen_dev_new)
            & (pen_dev_new <= pen_dev + 1e-7 * jnp.abs(pen_dev))
        )
        accepted = first_ok | subsequent_ok

        sh_init = (
            jnp.int32(0),  # k (halving counter)
            beta_new,  # beta_try
            pen_dev_new,  # pen_dev_try
            mu_new,  # mu_try
            accepted,  # accepted
        )

        def _sh_cond(sh_state):
            k, _, _, _, acc = sh_state
            return (k < 25) & (~acc)

        def _sh_body(sh_state):
            k, _beta_try, _pen_dev_try, _mu_try, _acc = sh_state
            step = 0.5 ** (k + 2)  # 0.25, 0.125, ...
            bt = beta + step * (beta_new - beta)
            eta_t = X @ bt + offset
            mu_t = family.link.inverse(eta_t)
            pd_t = _penalized_deviance(bt, mu_t, y, wt, S_lambda, family)

            ok = jnp.isfinite(pd_t) & (pd_t <= pen_dev + 1e-7 * jnp.abs(pen_dev))
            # On first iteration, accept any finite value
            ok = ok | (is_first_iter & jnp.isfinite(pd_t))

            return (k + 1, bt, pd_t, mu_t, ok)

        sh_final = jax.lax.while_loop(_sh_cond, _sh_body, sh_init)
        _k, beta_acc, pen_dev_acc, mu_acc, was_accepted = sh_final

        # If nothing was accepted (all 25 halvings failed), keep beta unchanged
        beta_next = jnp.where(was_accepted, beta_acc, beta)
        pen_dev_next = jnp.where(was_accepted, pen_dev_acc, pen_dev)
        mu_next = jnp.where(was_accepted, mu_acc, mu)

        # Convergence check (skip first 3 iterations)
        dev_change = jnp.abs(pen_dev_next - pen_dev) / (0.1 + jnp.abs(pen_dev_next))
        coef_change = jnp.max(jnp.abs(beta_next - beta)) / (
            0.1 + jnp.max(jnp.abs(beta_next))
        )
        converged = (i >= 3) & (dev_change < tol) & (coef_change < tol)

        return (
            i + 1,
            beta_next,
            beta,  # beta_old = previous beta
            mu_next,
            pen_dev_next,
            pen_dev,  # pen_dev_prev
            converged,
            XtWX,
            L,
            W,
        )

    final_state = jax.lax.while_loop(_cond, _body, init_state)
    (
        n_iter,
        beta_final,
        _beta_old,
        mu_final,
        pen_dev_final,
        _pen_dev_prev,
        converged,
        XtWX_final,
        L_final,
        W_final,
    ) = final_state

    eta_final = X @ beta_final + offset
    dev_final = family.dev_resids(y, mu_final, wt)
    scale = jnp.where(
        family.scale_known,
        1.0,
        dev_final / jnp.maximum(n - p, 1),
    )

    return PIRLSResult(
        coefficients=beta_final,
        mu=mu_final,
        eta=eta_final,
        deviance=dev_final,
        penalized_deviance=pen_dev_final,
        n_iter=n_iter,
        converged=converged,
        scale=scale,
        XtWX=XtWX_final,
        L=L_final,
        working_weights=W_final,
    )
