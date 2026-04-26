"""Penalized iteratively reweighted least squares (PIRLS) inner loop.

Given fixed smoothing parameters (encoded in ``S_lambda``), PIRLS finds
the penalized maximum likelihood coefficients by iterating a weighted
least-squares solve with step-halving on penalized deviance.

Standard exponential families use Fisher scoring (``gam.fit3``).
Extended families (NB) use Newton scoring with observed weights
(``gam.fit4``): ``w = 0.5 * d²D/dη²``.  After convergence, Fisher-
weighted ``XtWX_fisher`` / ``L_fisher`` are computed for EDF and
Bayesian covariance (matching R's ``gdi2``, ``gdi.c:2262-2294``).

The loop is implemented with ``jax.lax.while_loop`` so the entire
iteration compiles to a single fused XLA kernel when JIT-compiled.

Design doc reference: Section 7.2
R source reference: gam.fit3() lines 296-468, gam.fit4() lines 367-564
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp

from jaxgam.families.base import ExponentialFamily
from jaxgam.jax_utils import penalized_cholesky, penalized_solve

# Working weight bounds to prevent numerical overflow/underflow.
# R's gam.fit3 uses similar implicit bounds via sqrt(W) clamping.
_W_MIN = 1e-10
_W_MAX = 1e10

# Step acceptance tolerance relative to current penalized deviance.
# Distinct from the convergence tolerance parameter.
_PEN_DEV_REL_TOL = 1e-7

# Maximum step-halving iterations before giving up.
# Matches R's gam.fit3.r step-halving limit.
_MAX_HALVINGS = 25


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
        Final weighted cross-product matrix (Newton weights for extended
        families, Fisher weights for standard families). Used for REML
        criterion ``log|H|`` computation.
    L : jax.Array, shape (p, p)
        Final Cholesky factor of penalized Hessian (Newton/Fisher
        weights, matching ``XtWX``). Used for REML criterion.
    working_weights : jax.Array, shape (n,)
        Final working weights (Newton for extended, Fisher for standard).
    XtWX_fisher : jax.Array, shape (p, p)
        Fisher-weighted cross-product matrix. For standard families this
        equals ``XtWX``. For extended families this is recomputed with
        Fisher weights after convergence (R's ``gdi2``, gdi.c:2262-2294).
        Used for EDF and Bayesian covariance.
    L_fisher : jax.Array, shape (p, p)
        Cholesky factor of Fisher-weighted penalized Hessian. For
        standard families this equals ``L``. Used for EDF and Bayesian
        covariance.
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
    XtWX_fisher: jax.Array
    L_fisher: jax.Array


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
    W = jnp.clip(W, _W_MIN, _W_MAX)

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
    """Compute penalized deviance: dev(y, mu, wt) + beta^T S_lambda beta.

    Parameters
    ----------
    beta : jax.Array, shape (p,)
        Coefficient vector.
    mu : jax.Array, shape (n,)
        Fitted mean values.
    y : jax.Array, shape (n,)
        Response values.
    wt : jax.Array, shape (n,)
        Prior weights.
    S_lambda : jax.Array, shape (p, p)
        Combined weighted penalty matrix.
    family : ExponentialFamily
        Family with dev_resids method.

    Returns
    -------
    jax.Array, scalar
        Penalized deviance.
    """
    dev = family.dev_resids(y, mu, wt)
    penalty = beta @ S_lambda @ beta
    return dev + penalty


@jax.jit(static_argnames=("family", "max_iter", "tol"))
def _pirls_loop_jit(
    X: jax.Array,
    y: jax.Array,
    beta_init: jax.Array,
    S_lambda: jax.Array,
    family: ExponentialFamily,
    wt: jax.Array | None = None,
    offset: jax.Array | None = None,
    max_iter: int = 100,
    tol: float = 1e-7,
    log_theta: jax.Array | None = None,
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
    log_theta : jax.Array, shape (n_theta,), optional
        Extra distributional parameter for extended families (e.g.
        log-theta for NB).  When provided for a family with
        ``n_theta > 0``, working weights and deviance are computed
        via the family's pure-function factories (``working_weights_fn``,
        ``deviance_fn``) with ``log_theta`` as a **dynamic** JAX
        argument.  This avoids baking theta into the JIT cache as a
        static constant, so a single compiled kernel handles all
        theta values without recompilation.

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

    # ---- Theta-aware compute functions ----
    # Python ``if`` on ``family.n_theta`` is resolved at trace time
    # (``family`` is a static JIT arg).  For extended families the
    # pure-function factories take ``log_theta`` as a dynamic JAX
    # vector of shape ``(n_theta,)`` — generic over NB (1), Tweedie (2), etc.
    #
    # Extended families use **observed** weights ``w = 0.5 * d²D/dη²``
    # and observed working response ``z = η - (dD/dη)/(d²D/dη²)``
    # matching R's ``gam.fit4`` (gam.fit4.r lines 367-370).  Standard
    # families use Fisher weights via ``family.working_weights``,
    # matching R's ``gam.fit3``.
    if family.n_theta > 0 and log_theta is not None:
        _dev_fn = family.deviance_fn(y, wt)
        _grad_D_eta = jax.grad(_dev_fn, argnums=0)

        def _compute_W_and_z(mu, eta):  # noqa: ARG001  mu unused: observed weights come from d²D/dη², not V(mu)
            """Observed weights and working response from d²D/dη²."""
            dD_deta = _grad_D_eta(eta, log_theta)
            _, d2D_deta2 = jax.jvp(
                lambda e: _grad_D_eta(e, log_theta),
                (eta,),
                (jnp.ones_like(eta),),
            )
            # Observed weights: w = 0.5 * d²D/dη².  Unlike Fisher weights,
            # d²D/dη² can be negative for extended families.  Clamp to
            # _W_MIN so negative or near-zero values don't cause division
            # instability in the working response z.
            w = d2D_deta2 * 0.5
            d2D_safe = jnp.where(d2D_deta2 > _W_MIN, d2D_deta2, _W_MIN)
            z = (eta - offset) - dD_deta / d2D_safe
            return w, z

        def _compute_dev(mu, eta):  # noqa: ARG001  mu unused: deviance computed from eta via pure-function factory
            return _dev_fn(eta, log_theta)
    else:

        def _compute_W_and_z(mu, eta):
            W = family.working_weights(mu, wt)
            eta_no_offset = eta - offset
            z = family.working_response(y, mu, eta_no_offset)
            return W, z

        def _compute_dev(mu, eta):  # noqa: ARG001
            return family.dev_resids(y, mu, wt)

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
        # ---- Working quantities ----
        eta_cur = X @ state.beta + offset
        W, z = _compute_W_and_z(state.mu, eta_cur)
        W = jnp.clip(W, _W_MIN, _W_MAX)

        W_sqrt = jnp.sqrt(W)
        WX = W_sqrt[:, None] * X
        XtWX = WX.T @ WX
        XtWz = WX.T @ (W_sqrt * z)
        beta_new, L, _ = penalized_solve(XtWX, S_lambda, XtWz)

        # ---- Step-halving on penalized deviance ----
        is_first_iter = state.i == 0

        eta_new = X @ beta_new + offset
        mu_new = family.link.inverse(eta_new)
        dev_new = _compute_dev(mu_new, eta_new)
        pen_dev_new = dev_new + beta_new @ S_lambda @ beta_new

        # First iteration: unconditionally accept
        first_ok = is_first_iter & jnp.isfinite(pen_dev_new)
        subsequent_ok = (
            (~is_first_iter)
            & jnp.isfinite(pen_dev_new)
            & (pen_dev_new <= state.pen_dev + _PEN_DEV_REL_TOL * jnp.abs(state.pen_dev))
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
            return (sh.k < _MAX_HALVINGS) & (~sh.accepted)

        def _sh_body(sh: _StepHalvingState):
            step = 0.5 ** (sh.k + 2)  # 0.25, 0.125, ...
            bt = state.beta + step * (beta_new - state.beta)
            eta_t = X @ bt + offset
            mu_t = family.link.inverse(eta_t)
            dev_t = _compute_dev(mu_t, eta_t)
            pd_t = dev_t + bt @ S_lambda @ bt

            ok = jnp.isfinite(pd_t) & (
                pd_t <= state.pen_dev + _PEN_DEV_REL_TOL * jnp.abs(state.pen_dev)
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
        # Skip convergence check during first 3 warm-up iterations (R's gam.fit3.r)
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

    # Recompute curvature at final mu for consistency (R's gam.fit3 §7.2).
    eta_final = X @ final.beta + offset
    W_final, _ = _compute_W_and_z(final.mu, eta_final)
    W_final = jnp.clip(W_final, _W_MIN, _W_MAX)
    W_sqrt_final = jnp.sqrt(W_final)
    WX_final = W_sqrt_final[:, None] * X
    XtWX_final = WX_final.T @ WX_final
    L_final, _ = penalized_cholesky(XtWX_final, S_lambda)

    # For extended families: recompute XtWX and L using Fisher weights
    # for EDF and Bayesian covariance.  R's gam.fit4 uses Newton weights
    # for PIRLS but Fisher weights for EDF (gdi.c:2262-2294,
    # gam.fit4.r:564: ``wf = pmax(0, dd$EDeta2 * .5)``).
    # For standard families, Fisher = Newton, so just alias.
    if family.n_theta > 0 and log_theta is not None:
        _ww_fisher_fn = family.working_weights_fn(wt)
        W_fisher = _ww_fisher_fn(eta_final, log_theta)
        W_fisher = jnp.clip(W_fisher, _W_MIN, _W_MAX)
        W_sqrt_fisher = jnp.sqrt(W_fisher)
        WX_fisher = W_sqrt_fisher[:, None] * X
        XtWX_fisher = WX_fisher.T @ WX_fisher
        L_fisher, _ = penalized_cholesky(XtWX_fisher, S_lambda)
    else:
        XtWX_fisher = XtWX_final
        L_fisher = L_final

    dev_final = _compute_dev(final.mu, eta_final)
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
        XtWX=XtWX_final,
        L=L_final,
        working_weights=W_final,
        XtWX_fisher=XtWX_fisher,
        L_fisher=L_fisher,
    )


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
    log_theta: jax.Array | None = None,
) -> PIRLSResult:
    """Run PIRLS, passing estimated-family theta as dynamic JAX data.

    ``family`` is a JIT static argument, so any mutable family state read by
    the jitted implementation is baked into the compiled executable. For
    estimated extended families, default ``log_theta`` from the family state
    here, before JIT dispatch, so theta participates in the cache as a
    regular array argument instead of as static Python object state.
    """
    if family.n_theta > 0 and log_theta is None:
        log_theta = jnp.asarray(family.get_theta(transformed=False))

    return _pirls_loop_jit(
        X,
        y,
        beta_init,
        S_lambda,
        family,
        wt,
        offset,
        max_iter=max_iter,
        tol=tol,
        log_theta=log_theta,
    )


pirls_loop.clear_cache = _pirls_loop_jit.clear_cache  # type: ignore[attr-defined]
pirls_loop._cache_size = _pirls_loop_jit._cache_size  # type: ignore[attr-defined]
