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
class _PIRLSState:
    """Internal while_loop state for PIRLS. Registered as JAX pytree."""

    i: jax.Array
    beta: jax.Array
    beta_old: jax.Array
    mu: jax.Array
    pen_dev: jax.Array
    pen_dev_prev: jax.Array
    converged: jax.Array
    XtWX: jax.Array
    L: jax.Array
    W: jax.Array


_PIRLS_STATE_FIELDS = [f.name for f in fields(_PIRLSState)]

jax.tree_util.register_pytree_node(
    _PIRLSState,
    lambda s: ([getattr(s, f) for f in _PIRLS_STATE_FIELDS], None),
    lambda _, children: _PIRLSState(
        **dict(zip(_PIRLS_STATE_FIELDS, children, strict=True))
    ),
)


@dataclass(frozen=True)
class _StepHalvingState:
    """Internal while_loop state for step-halving. Registered as JAX pytree."""

    k: jax.Array
    beta_try: jax.Array
    pen_dev_try: jax.Array
    mu_try: jax.Array
    accepted: jax.Array


_SH_STATE_FIELDS = [f.name for f in fields(_StepHalvingState)]

jax.tree_util.register_pytree_node(
    _StepHalvingState,
    lambda s: ([getattr(s, f) for f in _SH_STATE_FIELDS], None),
    lambda _, children: _StepHalvingState(
        **dict(zip(_SH_STATE_FIELDS, children, strict=True))
    ),
)


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
    lambda _, children: PIRLSResult(**dict(zip(_PIRLS_FIELDS, children, strict=True))),
)


def _pirls_step(
    X: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    beta: jax.Array,
    mu: jax.Array,
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


@jax.jit(static_argnames=("family", "max_iter", "tol"))
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
    init_state = _PIRLSState(
        i=jnp.int32(0),
        beta=beta_init,
        beta_old=jnp.zeros_like(beta_init),
        mu=mu_init,
        pen_dev=jnp.array(jnp.inf),
        pen_dev_prev=jnp.array(jnp.inf),
        converged=jnp.bool_(False),
        XtWX=jnp.zeros((p, p)),
        L=jnp.eye(p),
        W=jnp.ones(n),
    )

    def _cond(state: _PIRLSState):
        return (state.i < max_iter) & (~state.converged)

    def _body(state: _PIRLSState):
        # One PIRLS step
        beta_new, XtWX, L, W = _pirls_step(
            X, y, wt, state.beta, state.mu, S_lambda, family
        )

        # Step-halving on penalized deviance
        is_first_iter = state.i == 0

        eta_new = X @ beta_new + offset
        mu_new = family.link.inverse(eta_new)
        pen_dev_new = _penalized_deviance(beta_new, mu_new, y, wt, S_lambda, family)

        # First iteration: unconditionally accept
        first_ok = is_first_iter & jnp.isfinite(pen_dev_new)
        subsequent_ok = (
            (~is_first_iter)
            & jnp.isfinite(pen_dev_new)
            & (pen_dev_new <= state.pen_dev + 1e-7 * jnp.abs(state.pen_dev))
        )
        accepted = first_ok | subsequent_ok

        sh_init = _StepHalvingState(
            k=jnp.int32(0),
            beta_try=beta_new,
            pen_dev_try=pen_dev_new,
            mu_try=mu_new,
            accepted=accepted,
        )

        def _sh_cond(sh: _StepHalvingState):
            return (sh.k < 25) & (~sh.accepted)

        def _sh_body(sh: _StepHalvingState):
            step = 0.5 ** (sh.k + 2)  # 0.25, 0.125, ...
            bt = state.beta + step * (beta_new - state.beta)
            eta_t = X @ bt + offset
            mu_t = family.link.inverse(eta_t)
            pd_t = _penalized_deviance(bt, mu_t, y, wt, S_lambda, family)

            ok = jnp.isfinite(pd_t) & (
                pd_t <= state.pen_dev + 1e-7 * jnp.abs(state.pen_dev)
            )
            # On first iteration, accept any finite value
            ok = ok | (is_first_iter & jnp.isfinite(pd_t))

            return _StepHalvingState(
                k=sh.k + 1, beta_try=bt, pen_dev_try=pd_t, mu_try=mu_t, accepted=ok
            )

        sh_final = jax.lax.while_loop(_sh_cond, _sh_body, sh_init)

        # If nothing was accepted (all 25 halvings failed), keep beta unchanged
        beta_next = jnp.where(sh_final.accepted, sh_final.beta_try, state.beta)
        pen_dev_next = jnp.where(sh_final.accepted, sh_final.pen_dev_try, state.pen_dev)
        mu_next = jnp.where(sh_final.accepted, sh_final.mu_try, state.mu)

        # Convergence check (skip first 3 iterations)
        dev_change = jnp.abs(pen_dev_next - state.pen_dev) / (
            0.1 + jnp.abs(pen_dev_next)
        )
        coef_change = jnp.max(jnp.abs(beta_next - state.beta)) / (
            0.1 + jnp.max(jnp.abs(beta_next))
        )
        converged = (state.i >= 3) & (dev_change < tol) & (coef_change < tol)

        return _PIRLSState(
            i=state.i + 1,
            beta=beta_next,
            beta_old=state.beta,
            mu=mu_next,
            pen_dev=pen_dev_next,
            pen_dev_prev=state.pen_dev,
            converged=converged,
            XtWX=XtWX,
            L=L,
            W=W,
        )

    final = jax.lax.while_loop(_cond, _body, init_state)

    eta_final = X @ final.beta + offset
    dev_final = family.dev_resids(y, final.mu, wt)
    scale = jnp.where(
        family.scale_known,
        1.0,
        dev_final / jnp.maximum(n - p, 1),
    )

    return PIRLSResult(
        coefficients=final.beta,
        mu=final.mu,
        eta=eta_final,
        deviance=dev_final,
        penalized_deviance=final.pen_dev,
        n_iter=final.i,
        converged=final.converged,
        scale=scale,
        XtWX=final.XtWX,
        L=final.L,
        working_weights=final.W,
    )
