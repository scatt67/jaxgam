"""Penalized iteratively reweighted least squares (PIRLS) inner loop.

Given fixed smoothing parameters (encoded in ``S_lambda``), PIRLS finds
the penalized maximum likelihood coefficients by iterating a weighted
least-squares solve with step-halving on penalized deviance.

Standard exponential families run the PIRLS solve with Fisher weights
(``gam.fit3``); extended families (NB) use Newton scoring with observed
weights (``gam.fit4``): ``w = 0.5 * d²D/dη²``. The converged coefficients
are identical under either weighting (both find the penalized-MLE
stationary point).

The REML ``log|H|`` curvature ``XtWX`` follows mgcv's information split:
observed (Newton) information for **non-canonical** links, Fisher
(expected) for canonical links (``gam.fit3.r:118``, ``gdi.c:2481-2498``).
After convergence, Fisher-weighted ``XtWX_fisher`` / ``L_fisher`` are
always computed for EDF and Bayesian covariance (``gdi.c:2262-2294``).
Canonical-ness is the static ``family.is_canonical`` property, so the
choice is resolved at trace time and canonical fits are byte-identical
to the pure-Fisher path.

The loop is implemented with ``jax.lax.while_loop`` so the entire
iteration compiles to a single fused XLA kernel when JIT-compiled.

Design doc reference: Section 7.2
R source reference: gam.fit3() lines 296-468, gam.fit4() lines 367-564
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral, Real

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np

from jaxgam.families.base import ExponentialFamily
from jaxgam.families.extended import ExtendedFamily
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.efs_theta import _conditional_theta_newton_jit, _efs_nb_log_deviance
from jaxgam.jax_utils import penalized_cholesky, penalized_solve
from jaxgam.links.links import LogLink

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

# ``gam.fit4`` uses a distinct divergence policy for the EFS extended-family
# branch (lines 484-504): its threshold is 10 * (.1 + |old.pdev|) * sqrt(eps)
# and it permits 100 halvings.  Keep the long-standing ordinary PIRLS policy
# separate, so default Newton and fixed-theta execution is unchanged.
_EFS_DIVERGENCE_ABS = 10.0 * 0.1 * jnp.sqrt(jnp.finfo(jnp.float64).eps)
_EFS_DIVERGENCE_REL = 10.0 * jnp.sqrt(jnp.finfo(jnp.float64).eps)
_EFS_MAX_HALVINGS = 100

_EFS_STATUS_CONVERGED = 0
_EFS_STATUS_BETA_STEP_FAILED = 1
_EFS_STATUS_INVALID_WORKING_FACTORS = 2
_EFS_STATUS_THETA_FAILED = 3
_EFS_STATUS_NONFINITE_STATIONARITY = 4
_EFS_STATUS_ITERATION_LIMIT = 5
_EFS_STATUS_INVALID_INPUT = 6
_EFS_STATUS_RETAINED_START_INVALID_TRIAL = 7
_EFS_STATUS_NONFINITE_RECOVERY_FAILED = 8
_EFS_STATUS_DOMAIN_RECOVERY_FAILED = 9
_EFS_STATUS_DIVERGENCE_RECOVERY_FAILED = 10


def canonical_working_quantities(
    family: ExponentialFamily,
    y: jax.Array,
    mu: jax.Array,
    eta: jax.Array,
    wt: jax.Array,
    offset: jax.Array,
    log_theta: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return standard-family Fisher weights and offset-free working response.

    Both dense PIRLS and streamed reductions use this family-owned arithmetic;
    callers apply their own bounded-row masking after weight clipping.
    """
    if log_theta is None:
        # Preserve the dense default operation order exactly.
        weight = family.working_weights(mu, wt)
    else:
        weight = family.working_weights_for_parameters(mu, eta, wt, log_theta)
    return weight, family.working_response(y, mu, eta - offset)


def accepted_penalized_step(
    trial_penalized_deviance: jax.Array,
    current_penalized_deviance: jax.Array,
    domain_ok: jax.Array,
    first_iteration: bool | jax.Array,
) -> jax.Array:
    """Dense PIRLS's finite/domain/decrease acceptance criterion."""
    finite_valid = jnp.isfinite(trial_penalized_deviance) & domain_ok
    return finite_valid & (
        jnp.asarray(first_iteration)
        | (
            trial_penalized_deviance
            <= current_penalized_deviance
            + _PEN_DEV_REL_TOL * jnp.abs(current_penalized_deviance)
        )
    )


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
class _BetaStepResult:
    """One immutable PIRLS beta proposal evaluated at one fixed theta."""

    beta: jax.Array
    mu: jax.Array
    eta: jax.Array
    penalized_deviance: jax.Array
    accepted: jax.Array
    factors_valid: jax.Array
    solver_valid: jax.Array
    XtWX: jax.Array
    L: jax.Array
    W: jax.Array
    proposal_valid: jax.Array | None = None


_BETA_STEP_FIELDS = [f.name for f in fields(_BetaStepResult)]

jax.tree_util.register_pytree_node(
    _BetaStepResult,
    lambda s: ([getattr(s, f) for f in _BETA_STEP_FIELDS], None),
    lambda _, children: _BetaStepResult(
        **dict(zip(_BETA_STEP_FIELDS, children, strict=True))
    ),
)


def _beta_step(
    *,
    X: jax.Array,
    S_lambda: jax.Array,
    offset: jax.Array,
    family: ExponentialFamily,
    beta: jax.Array,
    beta_old: jax.Array,
    mu: jax.Array,
    eta_current: jax.Array | None,
    penalized_deviance: jax.Array,
    iteration: jax.Array,
    compute_W_and_z,
    form_wls,
    compute_dev,
    divergence_absolute: float,
    divergence_relative: float,
    max_halvings: int,
    first_iteration_accepts_any: bool,
    require_valid_factors: bool,
) -> _BetaStepResult:
    """Take one WLS proposal and beta-only step-halving at fixed theta.

    The functions carrying the dynamic theta are closed over by the caller.
    This is deliberately the only shared EFS/ordinary-PIRLS iteration piece:
    the EFS loop can change theta *after* this accepted beta step without
    duplicating the WLS or beta-halving algorithm.
    """
    # Ordinary PIRLS derives eta from beta exactly as before. The EFS NB
    # initializer may instead supply R's link(mustart), which intentionally
    # has no coefficient representation on the first iteration.
    eta_cur = X @ beta + offset if eta_current is None else eta_current
    W, z = compute_W_and_z(mu, eta_cur)
    factors_valid = jnp.all(jnp.isfinite(W)) & jnp.all(jnp.isfinite(z))
    XtWX, XtWz = form_wls(W, z)
    beta_new, L, _ = penalized_solve(XtWX, S_lambda, XtWz)
    solver_valid = (
        jnp.all(jnp.isfinite(XtWX))
        & jnp.all(jnp.isfinite(XtWz))
        & jnp.all(jnp.isfinite(beta_new))
        & jnp.all(jnp.isfinite(L))
        & jnp.all(jnp.diag(L) > 0.0)
    )

    eta_new = X @ beta_new + offset
    mu_new = family.link.inverse(eta_new)
    dev_new = compute_dev(mu_new, eta_new)
    pen_dev_new = dev_new + beta_new @ S_lambda @ beta_new
    valid_new = _is_valid_trial(family, mu_new, eta_new)
    is_first_iter = iteration == 0
    threshold = divergence_absolute + divergence_relative * jnp.abs(penalized_deviance)
    decreasing = pen_dev_new <= penalized_deviance + threshold
    valid_factors = jnp.where(
        require_valid_factors, factors_valid & solver_valid, jnp.array(True)
    )
    accepted = (
        valid_factors
        & jnp.isfinite(pen_dev_new)
        & valid_new
        & ((is_first_iter & first_iteration_accepts_any) | decreasing)
    )

    halving_initial = _StepHalvingState(
        k=jnp.int32(0),
        beta_try=beta_new,
        pen_dev_try=pen_dev_new,
        mu_try=mu_new,
        accepted=accepted,
    )

    def halving_condition(state: _StepHalvingState) -> jax.Array:
        return (state.k < max_halvings) & ~state.accepted

    def halving_body(state: _StepHalvingState) -> _StepHalvingState:
        step = 0.5 ** (state.k + 1)
        beta_try = beta_old + step * (beta_new - beta_old)
        eta_try = X @ beta_try + offset
        mu_try = family.link.inverse(eta_try)
        dev_try = compute_dev(mu_try, eta_try)
        pen_dev_try = dev_try + beta_try @ S_lambda @ beta_try
        valid_try = _is_valid_trial(family, mu_try, eta_try)
        accepted_try = (
            valid_factors
            & jnp.isfinite(pen_dev_try)
            & valid_try
            & (
                (is_first_iter & first_iteration_accepts_any)
                | (pen_dev_try <= penalized_deviance + threshold)
            )
        )
        return _StepHalvingState(
            k=state.k + 1,
            beta_try=beta_try,
            pen_dev_try=pen_dev_try,
            mu_try=mu_try,
            accepted=accepted_try,
        )

    halving_final = jax.lax.while_loop(halving_condition, halving_body, halving_initial)
    beta_next = jnp.where(halving_final.accepted, halving_final.beta_try, beta)
    mu_next = jnp.where(halving_final.accepted, halving_final.mu_try, mu)
    eta_next = X @ beta_next + offset
    pdev_next = jnp.where(
        halving_final.accepted, halving_final.pen_dev_try, penalized_deviance
    )
    return _BetaStepResult(
        beta=beta_next,
        mu=mu_next,
        eta=eta_next,
        penalized_deviance=pdev_next,
        accepted=halving_final.accepted,
        factors_valid=factors_valid,
        solver_valid=solver_valid,
        proposal_valid=jnp.isfinite(pen_dev_new) & valid_new,
        XtWX=XtWX,
        L=L,
        W=W,
    )


def _is_valid_trial(
    family: ExponentialFamily, mu: jax.Array, eta: jax.Array
) -> jax.Array:
    """JIT-safe R ``validmu``/``valideta`` counterpart for a trial."""
    return jnp.all(family.valid_mu(mu)) & jnp.all(family.valid_eta(eta))


def _efs_nb_observed_working_quantities(
    eta: jax.Array,
    log_theta: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    offset: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return EFS NB/log observed WLS quantities at dynamic theta.

    ``gam.fit4`` divides by its observed second deviance derivative directly.
    Preserve every finite nonzero derivative here; the generic PIRLS floor
    would materially perturb a valid retained-start recovery at small
    curvature. Pinned R can instead use its finite ``wz`` representation when
    z is nonfinite. That direct-wz route is not yet ported: a positive-weight
    exact-zero curvature is therefore made explicitly invalid. A zero prior
    weight remains a finite zero-information row.
    """

    def dev_fn(current_eta: jax.Array) -> jax.Array:
        return _efs_nb_log_deviance(current_eta, log_theta, y, wt)

    gradient = jax.grad(dev_fn)
    d1 = gradient(eta)
    _, d2 = jax.jvp(gradient, (eta,), (jnp.ones_like(eta),))
    z = _efs_nb_observed_working_response(eta, offset, wt, d1, d2)
    return 0.5 * d2, z


def _efs_nb_observed_working_response(
    eta: jax.Array,
    offset: jax.Array,
    wt: jax.Array,
    d1: jax.Array,
    d2: jax.Array,
) -> jax.Array:
    """Form observed z while rejecting unported positive-weight zero curvature."""
    d2_safe = jnp.where(d2 != 0.0, d2, 1.0)
    z = (eta - offset) - d1 / d2_safe
    zero_positive_curvature = (d2 == 0.0) & (wt > 0.0)
    return jnp.where(zero_positive_curvature, jnp.nan, z)


@dataclass(frozen=True)
class _EFSRecoveryState:
    """One bounded EFS beta recovery stage with explicit eta midpoints."""

    beta: jax.Array
    eta: jax.Array
    mu: jax.Array
    deviance: jax.Array
    penalized_deviance: jax.Array
    n_halvings: jax.Array


_EFS_RECOVERY_FIELDS = [field.name for field in fields(_EFSRecoveryState)]
jax.tree_util.register_pytree_node(
    _EFSRecoveryState,
    lambda state: ([getattr(state, field) for field in _EFS_RECOVERY_FIELDS], None),
    lambda _, values: _EFSRecoveryState(
        **dict(zip(_EFS_RECOVERY_FIELDS, values, strict=True))
    ),
)


@dataclass(frozen=True)
class _EFSBetaRecoveryResult:
    """EFS-only raw WLS proposal after R's three sequential recoveries."""

    beta: jax.Array
    eta: jax.Array
    mu: jax.Array
    penalized_deviance: jax.Array
    accepted: jax.Array
    factors_valid: jax.Array
    solver_valid: jax.Array
    failure_status: jax.Array


_EFS_BETA_RECOVERY_FIELDS = [field.name for field in fields(_EFSBetaRecoveryResult)]
jax.tree_util.register_pytree_node(
    _EFSBetaRecoveryResult,
    lambda result: (
        [getattr(result, field) for field in _EFS_BETA_RECOVERY_FIELDS],
        None,
    ),
    lambda _, values: _EFSBetaRecoveryResult(
        **dict(zip(_EFS_BETA_RECOVERY_FIELDS, values, strict=True))
    ),
)


def _efs_beta_step_with_recovery(
    *,
    X: jax.Array,
    S_lambda: jax.Array,
    offset: jax.Array,
    family: NegativeBinomial,
    beta: jax.Array,
    eta: jax.Array,
    mu: jax.Array,
    beta_old: jax.Array,
    eta_old: jax.Array,
    null_beta: jax.Array,
    null_eta: jax.Array,
    initial_start_retained: jax.Array,
    iteration: jax.Array,
    baseline: jax.Array,
    compute_W_and_z,
    form_wls,
    compute_dev,
    max_recovery_halvings: int,
) -> _EFSBetaRecoveryResult:
    """Apply EFS-only ``gam.fit4`` recovery in its source order.

    The standard PIRLS beta helper remains unchanged.  Here nonfinite and
    domain recovery each move to the accepted start for the current iteration;
    only an initially retained start has a distinct beta/eta origin.  A first
    iteration pdev divergence then resets independently to the explicit null
    anchor.  Both beta and eta are retained in every midpoint because a
    ``mustart`` eta need not have a projected beta representation.
    """
    W, z = compute_W_and_z(mu, eta)
    factors_valid = jnp.all(jnp.isfinite(W)) & jnp.all(jnp.isfinite(z))
    XtWX, XtWz = form_wls(W, z)
    # EFS recovery follows ``gam.fit4``'s unregularized WLS proposal. The
    # shared solve intentionally adds scale-relative jitter for ordinary
    # PIRLS, but at a valid tiny observed curvature that can shift a raw beta
    # enough to change the bounded midpoint sequence. This exact EFS-only
    # factorization fails closed if the current observed system is not SPD.
    H = XtWX + S_lambda
    L = jnp.linalg.cholesky(H)
    beta_raw = jsla.cho_solve((L, True), XtWz)
    solver_valid = (
        jnp.all(jnp.isfinite(XtWX))
        & jnp.all(jnp.isfinite(XtWz))
        & jnp.all(jnp.isfinite(beta_raw))
        & jnp.all(jnp.isfinite(L))
        & jnp.all(jnp.diag(L) > 0.0)
    )
    solve_valid = factors_valid & solver_valid

    def solve(_: None) -> _EFSBetaRecoveryResult:
        eta_raw = X @ beta_raw + offset
        mu_raw = family.link.inverse(eta_raw)
        dev_raw = compute_dev(mu_raw, eta_raw)
        pdev_raw = dev_raw + beta_raw @ S_lambda @ beta_raw
        raw = _EFSRecoveryState(
            beta=beta_raw,
            eta=eta_raw,
            mu=mu_raw,
            deviance=dev_raw,
            penalized_deviance=pdev_raw,
            n_halvings=jnp.array(0, dtype=jnp.int32),
        )
        first_iteration = iteration == 0
        first_recovery_beta = jnp.where(
            first_iteration & initial_start_retained, beta, beta_old
        )
        first_recovery_eta = jnp.where(
            first_iteration & initial_start_retained, eta, eta_old
        )

        def nonfinite_condition(state: _EFSRecoveryState) -> jax.Array:
            return ~jnp.isfinite(state.deviance) & (
                state.n_halvings < max_recovery_halvings
            )

        def midpoint(
            state: _EFSRecoveryState, anchor_beta: jax.Array, anchor_eta: jax.Array
        ) -> _EFSRecoveryState:
            beta_next = (state.beta + anchor_beta) / 2.0
            eta_next = (state.eta + anchor_eta) / 2.0
            mu_next = family.link.inverse(eta_next)
            dev_next = compute_dev(mu_next, eta_next)
            return _EFSRecoveryState(
                beta=beta_next,
                eta=eta_next,
                mu=mu_next,
                deviance=dev_next,
                penalized_deviance=dev_next + beta_next @ S_lambda @ beta_next,
                n_halvings=state.n_halvings + 1,
            )

        def nonfinite_body(state: _EFSRecoveryState) -> _EFSRecoveryState:
            return midpoint(state, first_recovery_beta, first_recovery_eta)

        after_nonfinite = jax.lax.while_loop(nonfinite_condition, nonfinite_body, raw)
        nonfinite_failed = ~jnp.isfinite(after_nonfinite.deviance)

        def recover_domain(_: None) -> _EFSRecoveryState:
            domain_initial = _EFSRecoveryState(
                beta=after_nonfinite.beta,
                eta=after_nonfinite.eta,
                mu=after_nonfinite.mu,
                deviance=after_nonfinite.deviance,
                penalized_deviance=after_nonfinite.penalized_deviance,
                n_halvings=jnp.array(0, dtype=jnp.int32),
            )

            def domain_condition(state: _EFSRecoveryState) -> jax.Array:
                return ~_is_valid_trial(family, state.mu, state.eta) & (
                    state.n_halvings < max_recovery_halvings
                )

            def domain_body(state: _EFSRecoveryState) -> _EFSRecoveryState:
                return midpoint(state, first_recovery_beta, first_recovery_eta)

            return jax.lax.while_loop(domain_condition, domain_body, domain_initial)

        after_domain = jax.lax.cond(
            nonfinite_failed,
            lambda _: after_nonfinite,
            recover_domain,
            operand=None,
        )
        domain_nonfinite = ~jnp.isfinite(after_domain.deviance) | ~jnp.isfinite(
            after_domain.penalized_deviance
        )
        domain_failed = ~_is_valid_trial(family, after_domain.mu, after_domain.eta)

        # ``gam.fit4`` overwrites coefold/etaold with null state only when its
        # first iteration diverges.  Retained-start recovery above must not
        # leak into this later, independent divergence stage.
        divergence_beta = jnp.where(first_iteration, null_beta, beta_old)
        divergence_eta = jnp.where(first_iteration, null_eta, eta_old)
        divergence_threshold = _EFS_DIVERGENCE_ABS + _EFS_DIVERGENCE_REL * jnp.abs(
            baseline
        )

        def recover_divergence(_: None) -> _EFSRecoveryState:
            divergence_initial = _EFSRecoveryState(
                beta=after_domain.beta,
                eta=after_domain.eta,
                mu=after_domain.mu,
                deviance=after_domain.deviance,
                penalized_deviance=after_domain.penalized_deviance,
                n_halvings=jnp.array(0, dtype=jnp.int32),
            )

            def divergence_condition(state: _EFSRecoveryState) -> jax.Array:
                return (state.penalized_deviance - baseline > divergence_threshold) & (
                    state.n_halvings < _EFS_MAX_HALVINGS
                )

            def divergence_body(state: _EFSRecoveryState) -> _EFSRecoveryState:
                return midpoint(state, divergence_beta, divergence_eta)

            return jax.lax.while_loop(
                divergence_condition, divergence_body, divergence_initial
            )

        after_divergence = jax.lax.cond(
            nonfinite_failed | domain_nonfinite | domain_failed,
            lambda _: after_domain,
            recover_divergence,
            operand=None,
        )
        final_finite = (
            jnp.all(jnp.isfinite(after_divergence.beta))
            & jnp.all(jnp.isfinite(after_divergence.eta))
            & jnp.all(jnp.isfinite(after_divergence.mu))
            & jnp.isfinite(after_divergence.deviance)
            & jnp.isfinite(after_divergence.penalized_deviance)
        )
        final_domain_valid = _is_valid_trial(
            family, after_divergence.mu, after_divergence.eta
        )
        divergence_failed = (
            after_divergence.penalized_deviance - baseline > divergence_threshold
        )
        recovered = (
            ~nonfinite_failed
            & ~domain_nonfinite
            & ~domain_failed
            & final_finite
            & final_domain_valid
            & ~divergence_failed
        )
        failure_status = jnp.where(
            ~final_finite | nonfinite_failed | domain_nonfinite,
            jnp.array(_EFS_STATUS_NONFINITE_RECOVERY_FAILED, dtype=jnp.int32),
            jnp.where(
                ~final_domain_valid | domain_failed,
                jnp.array(_EFS_STATUS_DOMAIN_RECOVERY_FAILED, dtype=jnp.int32),
                jnp.where(
                    divergence_failed,
                    jnp.array(_EFS_STATUS_DIVERGENCE_RECOVERY_FAILED, dtype=jnp.int32),
                    jnp.array(_EFS_STATUS_CONVERGED, dtype=jnp.int32),
                ),
            ),
        )
        return _EFSBetaRecoveryResult(
            beta=after_divergence.beta,
            eta=after_divergence.eta,
            mu=after_divergence.mu,
            penalized_deviance=after_divergence.penalized_deviance,
            accepted=recovered,
            factors_valid=factors_valid,
            solver_valid=solver_valid,
            failure_status=failure_status,
        )

    def invalid(_: None) -> _EFSBetaRecoveryResult:
        return _EFSBetaRecoveryResult(
            beta=beta,
            eta=eta,
            mu=mu,
            penalized_deviance=baseline,
            accepted=jnp.array(False),
            factors_valid=factors_valid,
            solver_valid=solver_valid,
            failure_status=jnp.array(
                _EFS_STATUS_INVALID_WORKING_FACTORS, dtype=jnp.int32
            ),
        )

    return jax.lax.cond(solve_valid, solve, invalid, operand=None)


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


def _observed_weights(
    family: ExponentialFamily,
    X: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    beta: jax.Array,
    offset: jax.Array,
) -> jax.Array:
    """Per-observation observed (Newton) information weight ``0.5 d²D/dη²``.

    For an exponential-dispersion family the observed information per
    observation equals half the second derivative of the unit deviance w.r.t.
    eta. mgcv uses these (not Fisher) weights in the REML ``log|H|`` for
    non-canonical links (gam.fit3.r:118, gdi.c:2481-2498). The total deviance
    is separable across observations, so the eta-Hessian is diagonal and a
    single JVP of ``grad(D)`` in the all-ones direction recovers the diagonal.
    """
    eta = X @ beta + offset

    def _dev_sum(e: jax.Array) -> jax.Array:
        return family.dev_resids(y, family.link.inverse(e), wt)

    grad_D = jax.grad(_dev_sum)
    _, d2 = jax.jvp(grad_D, (eta,), (jnp.ones_like(eta),))
    return 0.5 * d2


def _signed_XtWX(w_signed: jax.Array, X: jax.Array) -> jax.Array:
    """``X' diag(w) X`` with SIGNED weights for the observed-information log|H|.

    Non-canonical links produce per-observation observed (Newton) weights that
    can be negative; mgcv keeps them in the REML ``log|H|`` via a sign-aware
    factorization (gdi.c:2481-2498). Only the magnitude is capped (to avoid
    overflow); negatives are NOT floored to a positive value — that flooring is
    exactly what corrupts ``log|H|`` for NB identity/sqrt links. ``H = XtWX + S``
    stays positive-definite at a valid penalized optimum because the penalty
    dominates the indefinite directions (the same signed observed Hessian is
    already used by the Newton IFT in ``newton.py``).
    """
    w = jnp.clip(w_signed, -_W_MAX, _W_MAX)
    return (w[:, None] * X).T @ X


@jax.jit(static_argnames=("family", "max_iter", "tol", "extended_observed"))
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
    extended_observed: bool = False,
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
    extended_observed : bool
        Static EFS-only opt-in for a fixed extended-family parameter. The
        default preserves the established ``n_theta > 0`` PIRLS/Newton route;
        EFS uses this only for fixed-theta NB to mirror ``gam.fit4``'s
        observed scoring and Fisher EDF split without adding a theta
        optimization coordinate.

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
    # Python ``if`` on this static controller mode is resolved at trace time
    # (``family`` is a static JIT arg).  For extended families the
    # pure-function factories take ``log_theta`` as a dynamic JAX
    # vector of shape ``(n_theta,)`` — generic over NB (1), Tweedie (2), etc.
    #
    # Extended families use **observed** weights ``w = 0.5 * d²D/dη²``
    # and observed working response ``z = η - (dD/dη)/(d²D/dη²)``
    # matching R's ``gam.fit4`` (gam.fit4.r lines 367-370).  Standard
    # families use Fisher weights via ``family.working_weights``,
    # matching R's ``gam.fit3``.
    use_extended_observed = family.n_theta > 0 or extended_observed
    if use_extended_observed and log_theta is not None:
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
            # d²D/dη² can be NEGATIVE for non-canonical extended-family links;
            # keep the sign (mgcv gam.fit4 / gdi.c). Only floor the *magnitude*
            # of the working-response denominator so a near-zero curvature does
            # not blow up z, without flipping a genuine negative to positive
            # (that flip corrupts z for NB identity/sqrt).
            w = d2D_deta2 * 0.5
            d2D_safe = jnp.where(jnp.abs(d2D_deta2) > _W_MIN, d2D_deta2, _W_MIN)
            z = (eta - offset) - dD_deta / d2D_safe
            return w, z

        def _compute_dev(mu, eta):  # noqa: ARG001  mu unused: deviance computed from eta via pure-function factory
            return _dev_fn(eta, log_theta)

        def _form_wls(W, z):
            """Sign-aware (Newton) penalized WLS for extended families.

            Observed weights can be negative for non-canonical links; mgcv keeps
            them in the WLS (gam.fit4) rather than flooring to a positive value.
            Build ``X' diag(W) X`` and ``X' diag(W) z`` directly (no ``sqrt(W)``)
            with the magnitude capped; for canonical links (all W >= 0) this is
            numerically identical to the sqrt form.
            """
            Wc = jnp.clip(W, -_W_MAX, _W_MAX)
            return (Wc[:, None] * X).T @ X, X.T @ (Wc * z)
    else:

        def _compute_W_and_z(mu, eta):
            return canonical_working_quantities(family, y, mu, eta, wt, offset)

        def _compute_dev(mu, eta):  # noqa: ARG001
            return family.dev_resids(y, mu, wt)

        def _form_wls(W, z):
            """Fisher-scoring penalized WLS for standard families (positive W)."""
            Wc = jnp.clip(W, _W_MIN, _W_MAX)
            w_sqrt = jnp.sqrt(Wc)
            wx = w_sqrt[:, None] * X
            return wx.T @ wx, wx.T @ (w_sqrt * z)

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
        beta_step = _beta_step(
            X=X,
            S_lambda=S_lambda,
            offset=offset,
            family=family,
            beta=state.beta,
            beta_old=state.beta,
            mu=state.mu,
            eta_current=None,
            penalized_deviance=state.pen_dev,
            iteration=state.i,
            compute_W_and_z=_compute_W_and_z,
            form_wls=_form_wls,
            compute_dev=_compute_dev,
            divergence_absolute=0.0,
            divergence_relative=_PEN_DEV_REL_TOL,
            max_halvings=_MAX_HALVINGS,
            first_iteration_accepts_any=True,
            require_valid_factors=False,
        )
        beta_next = beta_step.beta
        pen_dev_next = beta_step.penalized_deviance
        mu_next = beta_step.mu

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
            XtWX=beta_step.XtWX,
            L=beta_step.L,
            W=beta_step.W,
        )

    final = jax.lax.while_loop(_cond, _body, init_state)

    # Recompute curvature at final mu (R's gam.fit3 §7.2). Split of information
    # matrices (gdi.c): the REML log|H| uses OBSERVED (Newton) information for
    # non-canonical links — INCLUDING its negative per-observation weights
    # (gdi.c:2481-2498 is sign-aware) — while EDF/Bayesian covariance always use
    # FISHER information (gdi.c:2262-2294). For canonical links observed ==
    # Fisher. The converged beta is identical under either weighting.
    eta_final = X @ final.beta + offset

    if use_extended_observed and log_theta is not None:
        # Extended families (NB): observed (signed) weights for the REML log|H|.
        W_final, _ = _compute_W_and_z(final.mu, eta_final)  # 0.5 d²D/dη², signed
        XtWX_final = _signed_XtWX(W_final, X)
        L_final, _ = penalized_cholesky(XtWX_final, S_lambda)
        # Fisher (nonnegative) for EDF and Bayesian covariance.
        _ww_fisher_fn = family.working_weights_fn(wt)
        W_fisher = jnp.clip(_ww_fisher_fn(eta_final, log_theta), _W_MIN, _W_MAX)
        WX_fisher = jnp.sqrt(W_fisher)[:, None] * X
        XtWX_fisher = WX_fisher.T @ WX_fisher
        L_fisher, _ = penalized_cholesky(XtWX_fisher, S_lambda)
    elif not family.is_canonical:
        # Non-canonical standard family: the PIRLS solve used Fisher weights
        # (nonnegative) -> keep for EDF/covariance; rebuild the REML-log|H| XtWX
        # from OBSERVED (Newton) weights, preserving their sign (matching mgcv).
        W_final, _ = _compute_W_and_z(final.mu, eta_final)  # Fisher for standard
        W_final = jnp.clip(W_final, _W_MIN, _W_MAX)
        WX_fisher = jnp.sqrt(W_final)[:, None] * X
        XtWX_fisher = WX_fisher.T @ WX_fisher
        L_fisher, _ = penalized_cholesky(XtWX_fisher, S_lambda)
        W_obs = _observed_weights(family, X, y, wt, final.beta, offset)  # signed
        XtWX_final = _signed_XtWX(W_obs, X)
        L_final, _ = penalized_cholesky(XtWX_final, S_lambda)
    else:
        # Canonical standard family: Fisher == Newton (weights nonnegative).
        W_final, _ = _compute_W_and_z(final.mu, eta_final)
        W_final = jnp.clip(W_final, _W_MIN, _W_MAX)
        WX_final = jnp.sqrt(W_final)[:, None] * X
        XtWX_final = WX_final.T @ WX_final
        L_final, _ = penalized_cholesky(XtWX_final, S_lambda)
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


@dataclass(frozen=True)
class EFSThetaPIRLSResult:
    """Immutable result for the NB/log EFS in-loop conditional-theta path.

    ``stopping_penalized_deviance`` is the pre-theta value used by the pinned
    R stopping test. ``pirls_result.penalized_deviance`` is deliberately
    recomputed at ``log_theta`` so every returned fit quantity is consistent
    with its reported theta. This is a documented stronger final-state rule,
    not a substitution for R's stopping predicate.
    """

    pirls_result: PIRLSResult
    log_theta: jax.Array
    theta_status: jax.Array
    theta_n_iter: jax.Array
    status: jax.Array
    stopping_penalized_deviance: jax.Array
    post_theta_penalized_deviance: jax.Array


_EFS_THETA_PIRLS_FIELDS = [f.name for f in fields(EFSThetaPIRLSResult)]

jax.tree_util.register_pytree_node(
    EFSThetaPIRLSResult,
    lambda result: (
        [getattr(result, field) for field in _EFS_THETA_PIRLS_FIELDS],
        None,
    ),
    lambda _, values: EFSThetaPIRLSResult(
        **dict(zip(_EFS_THETA_PIRLS_FIELDS, values, strict=True))
    ),
)


@dataclass(frozen=True)
class _EFSThetaPIRLSState:
    """Pytree state carrying beta and theta together inside one compiled loop."""

    i: jax.Array
    beta: jax.Array
    beta_old: jax.Array
    eta_old: jax.Array
    eta: jax.Array
    mu: jax.Array
    log_theta: jax.Array
    baseline: jax.Array
    stopping_pdev: jax.Array
    post_theta_pdev: jax.Array
    converged: jax.Array
    failed: jax.Array
    status: jax.Array
    theta_status: jax.Array
    theta_n_iter: jax.Array


_EFS_THETA_STATE_FIELDS = [f.name for f in fields(_EFSThetaPIRLSState)]

jax.tree_util.register_pytree_node(
    _EFSThetaPIRLSState,
    lambda state: ([getattr(state, field) for field in _EFS_THETA_STATE_FIELDS], None),
    lambda _, values: _EFSThetaPIRLSState(
        **dict(zip(_EFS_THETA_STATE_FIELDS, values, strict=True))
    ),
)


@jax.jit(static_argnames=("family", "max_y", "integer_counts", "max_iter", "tol"))
def _efs_theta_pirls_loop_jit(
    X: jax.Array,
    y: jax.Array,
    beta_init: jax.Array,
    S_lambda: jax.Array,
    family: NegativeBinomial,
    wt: jax.Array,
    offset: jax.Array,
    log_theta_init: jax.Array,
    beta_old_init: jax.Array,
    initial_eta: jax.Array,
    initial_start_retained: jax.Array,
    count_indices: jax.Array,
    *,
    max_y: int,
    integer_counts: bool,
    max_iter: int,
    tol: float,
) -> EFSThetaPIRLSResult:
    """Pinned EFS NB/log beta/theta alternation in one JAX while-loop.

    Each beta proposal and its divergence control are evaluated at the
    carried, incoming theta. Only after beta is accepted is conditional theta
    Newton run at fixed eta. The next beta iteration gets the post-theta
    deviance baseline and working quantities, matching ``gam.fit4.r``
    lines 486-547 without a host-side fit/update alternation.
    """

    # Estimated-theta NB EFS deliberately uses its own stable eta-space
    # deviance. Ordinary/default and fixed-theta NB retain the family helper.
    def dev_fn(eta: jax.Array, log_theta: jax.Array) -> jax.Array:
        return _efs_nb_log_deviance(eta, log_theta, y, wt)

    def ops(log_theta: jax.Array):
        def compute_W_and_z(mu: jax.Array, eta: jax.Array):  # noqa: ARG001
            return _efs_nb_observed_working_quantities(
                eta,
                log_theta,
                y,
                wt,
                offset,
            )

        def form_wls(W: jax.Array, z: jax.Array):
            Wc = jnp.clip(W, -_W_MAX, _W_MAX)
            return (Wc[:, None] * X).T @ X, X.T @ (Wc * z)

        def compute_dev(mu: jax.Array, eta: jax.Array):  # noqa: ARG001
            return dev_fn(eta, log_theta)

        return compute_W_and_z, form_wls, compute_dev

    eta_init = initial_eta
    mu_init = family.link.inverse(eta_init)
    eta_old_init = X @ beta_old_init + offset
    mu_old_init = family.link.inverse(eta_old_init)
    _, _, compute_initial_dev = ops(log_theta_init)
    initial_dev = compute_initial_dev(mu_old_init, eta_old_init)
    initial_pdev = initial_dev + beta_old_init @ S_lambda @ beta_old_init
    input_valid = (
        jnp.all(jnp.isfinite(y))
        & jnp.all(y >= 0.0)
        & jnp.all(jnp.isfinite(wt))
        & jnp.all(wt >= 0.0)
        & jnp.all(jnp.isfinite(log_theta_init))
        & jnp.isfinite(jnp.exp(log_theta_init[0]))
        & (jnp.exp(log_theta_init[0]) > 0.0)
        & jnp.all(count_indices >= 0)
        & jnp.all(count_indices <= max_y)
    )
    integer_prefix_valid = jnp.all(y == jnp.floor(y)) & jnp.all(
        count_indices == y.astype(count_indices.dtype)
    )
    input_valid = input_valid & jnp.where(
        integer_counts, integer_prefix_valid, jnp.all(count_indices == 0)
    )
    initial_valid = (
        input_valid
        & jnp.all(jnp.isfinite(S_lambda))
        & jnp.all(jnp.isfinite(beta_init))
        & jnp.all(jnp.isfinite(beta_old_init))
        & jnp.all(jnp.isfinite(mu_init))
        & jnp.all(jnp.isfinite(mu_old_init))
        & _is_valid_trial(family, mu_init, eta_init)
        & _is_valid_trial(family, mu_old_init, eta_old_init)
        & jnp.isfinite(initial_pdev)
    )
    initial_status = jnp.where(
        ~input_valid,
        jnp.array(_EFS_STATUS_INVALID_INPUT, dtype=jnp.int32),
        jnp.where(
            initial_valid,
            jnp.array(_EFS_STATUS_CONVERGED, dtype=jnp.int32),
            jnp.array(_EFS_STATUS_INVALID_WORKING_FACTORS, dtype=jnp.int32),
        ),
    )
    state = _EFSThetaPIRLSState(
        i=jnp.array(0, dtype=jnp.int32),
        beta=beta_init,
        beta_old=beta_old_init,
        eta_old=eta_old_init,
        eta=eta_init,
        mu=mu_init,
        log_theta=log_theta_init,
        baseline=initial_pdev,
        stopping_pdev=initial_pdev,
        post_theta_pdev=initial_pdev,
        converged=jnp.array(False),
        failed=~initial_valid,
        status=initial_status,
        theta_status=jnp.array(0, dtype=jnp.int32),
        theta_n_iter=jnp.array(0, dtype=jnp.int32),
    )

    def condition(state: _EFSThetaPIRLSState) -> jax.Array:
        return (state.i < max_iter) & ~state.converged & ~state.failed

    def body(state: _EFSThetaPIRLSState) -> _EFSThetaPIRLSState:
        compute_W_and_z, form_wls, compute_dev = ops(state.log_theta)
        beta_step = _efs_beta_step_with_recovery(
            X=X,
            S_lambda=S_lambda,
            offset=offset,
            family=family,
            beta=state.beta,
            eta=state.eta,
            mu=state.mu,
            beta_old=state.beta_old,
            eta_old=state.eta_old,
            null_beta=beta_old_init,
            null_eta=eta_old_init,
            initial_start_retained=initial_start_retained,
            iteration=state.i,
            baseline=state.baseline,
            compute_W_and_z=compute_W_and_z,
            form_wls=form_wls,
            compute_dev=compute_dev,
            max_recovery_halvings=max_iter,
        )

        def beta_failed(_: None) -> _EFSThetaPIRLSState:
            status = jnp.where(
                ~beta_step.factors_valid | ~beta_step.solver_valid,
                jnp.array(_EFS_STATUS_INVALID_WORKING_FACTORS, dtype=jnp.int32),
                jnp.where(
                    beta_step.failure_status
                    == jnp.array(_EFS_STATUS_CONVERGED, dtype=jnp.int32),
                    jnp.array(_EFS_STATUS_BETA_STEP_FAILED, dtype=jnp.int32),
                    beta_step.failure_status,
                ),
            )
            return _EFSThetaPIRLSState(
                i=state.i + 1,
                beta=state.beta,
                beta_old=state.beta_old,
                eta_old=state.eta_old,
                eta=state.eta,
                mu=state.mu,
                log_theta=state.log_theta,
                baseline=state.baseline,
                stopping_pdev=state.stopping_pdev,
                post_theta_pdev=state.post_theta_pdev,
                converged=jnp.array(False),
                failed=jnp.array(True),
                status=status,
                theta_status=state.theta_status,
                theta_n_iter=state.theta_n_iter,
            )

        def beta_accepted(_: None) -> _EFSThetaPIRLSState:
            theta_result = _conditional_theta_newton_jit(
                state.log_theta,
                beta_step.eta,
                y,
                wt,
                count_indices,
                family,
                max_y=max_y,
                integer_counts=integer_counts,
                tolerance=1e-7,
                max_iter=100,
                max_step=4.0,
                max_halvings=25,
            )
            theta_ok = theta_result.converged & (theta_result.status == 0)
            _, _, post_theta_dev_fn = ops(theta_result.log_theta)
            post_theta_dev = post_theta_dev_fn(beta_step.mu, beta_step.eta)
            post_theta_pdev = (
                post_theta_dev + beta_step.beta @ S_lambda @ beta_step.beta
            )
            # R rebuilds dd at the accepted theta before checking the score
            # equations (gam.fit4.r:516-528). The pdev part remains pre-theta.
            next_compute_W_and_z, _, _ = ops(theta_result.log_theta)
            next_W, next_z = next_compute_W_and_z(beta_step.mu, beta_step.eta)
            next_factors_valid = jnp.all(jnp.isfinite(next_W)) & jnp.all(
                jnp.isfinite(next_z)
            )
            gradient = (
                2.0 * X.T @ (next_W * (X @ beta_step.beta) - next_W * next_z)
                + 2.0 * S_lambda @ beta_step.beta
            )
            gradient_finite = jnp.all(jnp.isfinite(gradient))
            pdev_change = jnp.abs(beta_step.penalized_deviance - state.baseline) / (
                0.1 + jnp.abs(beta_step.penalized_deviance)
            )
            pdev_small = pdev_change < tol
            stationary = jnp.max(jnp.abs(gradient)) <= tol * (
                jnp.abs(beta_step.penalized_deviance) + 1.0
            )
            valid_post_theta = (
                theta_ok
                & jnp.isfinite(post_theta_pdev)
                & next_factors_valid
                & gradient_finite
            )
            converged = valid_post_theta & pdev_small & stationary
            status = jnp.where(
                ~theta_ok,
                jnp.array(_EFS_STATUS_THETA_FAILED, dtype=jnp.int32),
                jnp.where(
                    ~next_factors_valid | ~jnp.isfinite(post_theta_pdev),
                    jnp.array(_EFS_STATUS_INVALID_WORKING_FACTORS, dtype=jnp.int32),
                    jnp.where(
                        ~gradient_finite,
                        jnp.array(_EFS_STATUS_NONFINITE_STATIONARITY, dtype=jnp.int32),
                        jnp.array(_EFS_STATUS_CONVERGED, dtype=jnp.int32),
                    ),
                ),
            )
            return _EFSThetaPIRLSState(
                i=state.i + 1,
                beta=beta_step.beta,
                beta_old=beta_step.beta,
                eta_old=beta_step.eta,
                eta=beta_step.eta,
                mu=beta_step.mu,
                log_theta=theta_result.log_theta,
                baseline=post_theta_pdev,
                stopping_pdev=beta_step.penalized_deviance,
                post_theta_pdev=post_theta_pdev,
                converged=converged,
                failed=~valid_post_theta,
                status=status,
                theta_status=theta_result.status,
                theta_n_iter=theta_result.n_iter,
            )

        return jax.lax.cond(
            beta_step.accepted,
            beta_accepted,
            beta_failed,
            operand=None,
        )

    final = jax.lax.while_loop(condition, body, state)
    final_status = jnp.where(
        ~final.converged & ~final.failed,
        jnp.array(_EFS_STATUS_ITERATION_LIMIT, dtype=jnp.int32),
        final.status,
    )

    # Named final-state consistency rule: R can break before its post-theta
    # pdev refresh, leaving a stale returned deviance. Keep the exact pre-theta
    # stopping value above, but return all fit/curvature data at final theta.
    eta_final = X @ final.beta + offset
    mu_final = family.link.inverse(eta_final)
    final_compute_W_and_z, _, final_compute_dev = ops(final.log_theta)
    W_final, _ = final_compute_W_and_z(mu_final, eta_final)
    XtWX_final = _signed_XtWX(W_final, X)
    L_final, _ = penalized_cholesky(XtWX_final, S_lambda)
    W_fisher = jnp.clip(
        family.working_weights_fn(wt)(eta_final, final.log_theta), _W_MIN, _W_MAX
    )
    WX_fisher = jnp.sqrt(W_fisher)[:, None] * X
    XtWX_fisher = WX_fisher.T @ WX_fisher
    L_fisher, _ = penalized_cholesky(XtWX_fisher, S_lambda)
    dev_final = final_compute_dev(mu_final, eta_final)
    pdev_final = dev_final + final.beta @ S_lambda @ final.beta
    final_valid = (
        _is_valid_trial(family, mu_final, eta_final)
        & jnp.isfinite(dev_final)
        & jnp.isfinite(pdev_final)
        & jnp.all(jnp.isfinite(W_final))
        & jnp.all(jnp.isfinite(W_fisher))
        & jnp.all(jnp.isfinite(XtWX_final))
        & jnp.all(jnp.isfinite(XtWX_fisher))
        & jnp.all(jnp.isfinite(L_final))
        & jnp.all(jnp.isfinite(L_fisher))
        & jnp.all(jnp.diag(L_final) > 0.0)
        & jnp.all(jnp.diag(L_fisher) > 0.0)
    )
    final_status = jnp.where(
        (final_status == _EFS_STATUS_CONVERGED) & ~final_valid,
        jnp.array(_EFS_STATUS_INVALID_WORKING_FACTORS, dtype=jnp.int32),
        final_status,
    )
    pirls_result = PIRLSResult(
        coefficients=final.beta,
        mu=mu_final,
        eta=eta_final,
        deviance=dev_final,
        penalized_deviance=pdev_final,
        n_iter=final.i,
        converged=final.converged & (final_status == _EFS_STATUS_CONVERGED),
        scale=jnp.array(1.0),
        XtWX=XtWX_final,
        L=L_final,
        working_weights=W_final,
        XtWX_fisher=XtWX_fisher,
        L_fisher=L_fisher,
    )
    return EFSThetaPIRLSResult(
        pirls_result=pirls_result,
        log_theta=final.log_theta,
        theta_status=final.theta_status,
        theta_n_iter=final.theta_n_iter,
        status=final_status,
        stopping_penalized_deviance=final.stopping_pdev,
        post_theta_penalized_deviance=final.post_theta_pdev,
    )


def efs_theta_pirls_loop(
    X: jax.Array,
    y: jax.Array,
    beta_init: jax.Array,
    S_lambda: jax.Array,
    family: NegativeBinomial,
    wt: jax.Array,
    offset: jax.Array | None,
    log_theta_init: jax.Array,
    count_indices: jax.Array,
    *,
    beta_old_init: jax.Array | None = None,
    initial_eta: jax.Array | None = None,
    initial_start_retained: bool = False,
    max_y: int,
    integer_counts: bool,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> EFSThetaPIRLSResult:
    """Run the EFS-only NB/log in-loop conditional-theta PIRLS solver.

    ``beta_old_init`` is the explicit null coefficient state for the first R
    divergence comparison. ``initial_eta`` supplies ``link(mustart)`` when R
    starts or resets with no retained coefficient vector; it is intentionally
    distinct from ``X @ beta_init + offset``. A retained first start first
    recovers nonfinite/domain proposals toward its retained beta/eta state;
    an ensuing first-iteration pdev divergence instead moves toward the null
    beta/eta anchor, as in ``gam.fit4``.

    This internal entry point is intentionally separate from ``pirls_loop``;
    neither default joint-Newton NB nor fixed-theta EFS routes opt into it.

    Shapes and dtypes are rejected before JIT dispatch: X is nonempty ``(n,p)``
    float data, y/wt/offset/count_indices are aligned ``(n,)`` arrays,
    beta is ``(p,)``, S_lambda is ``(p,p)``, and count indices are integral.
    FittingData is the upstream owner of count-prefix construction and must
    supply indices matching nonnegative integer responses (or zero indices for
    fractional metadata). Dynamic malformed y, weights, theta, prefix values,
    return ``INVALID_INPUT`` rather than claiming a safe/converged fit.
    Non-finite penalty algebra is separately reported as
    ``INVALID_WORKING_FACTORS``.
    """
    if not isinstance(family, NegativeBinomial):
        raise TypeError("EFS theta PIRLS requires NegativeBinomial")
    if family.n_theta != 1:
        raise ValueError("EFS theta PIRLS requires an estimated NB theta")
    if not isinstance(family.link, LogLink):
        raise NotImplementedError("EFS theta PIRLS currently supports NB/log")
    if isinstance(max_y, bool) or not isinstance(max_y, Integral) or max_y < 0:
        raise ValueError("EFS theta PIRLS max_y must be an integer >= 0")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("EFS theta PIRLS max_iter must be an integer >= 1")
    if (
        isinstance(tol, bool)
        or not isinstance(tol, Real)
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError("EFS theta PIRLS tol must be finite and positive")
    if not isinstance(integer_counts, bool):
        raise ValueError("EFS theta PIRLS integer_counts must be bool")
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("EFS theta PIRLS X must be a nonempty two-dimensional array")
    n, p = X.shape
    for name, value, shape in (
        ("y", y, (n,)),
        ("wt", wt, (n,)),
        ("count_indices", count_indices, (n,)),
        ("beta_init", beta_init, (p,)),
        ("S_lambda", S_lambda, (p, p)),
    ):
        if value.shape != shape:
            raise ValueError(f"EFS theta PIRLS {name} must have shape {shape}")
    for name, value in (
        ("X", X),
        ("y", y),
        ("wt", wt),
        ("beta_init", beta_init),
        ("S_lambda", S_lambda),
    ):
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError(f"EFS theta PIRLS {name} must have floating dtype")
    if not jnp.issubdtype(count_indices.dtype, jnp.integer):
        raise ValueError("EFS theta PIRLS count_indices must have integer dtype")
    if log_theta_init.shape != (1,):
        raise ValueError("EFS theta PIRLS log_theta_init must have shape (1,)")
    if not jnp.issubdtype(log_theta_init.dtype, jnp.floating):
        raise ValueError("EFS theta PIRLS log_theta_init must have floating dtype")
    if beta_old_init is None:
        beta_old_init = beta_init
    if beta_old_init.shape != beta_init.shape:
        raise ValueError("EFS theta PIRLS beta_old_init must align with beta_init")
    if not jnp.issubdtype(beta_old_init.dtype, jnp.floating):
        raise ValueError("EFS theta PIRLS beta_old_init must have floating dtype")
    if y.ndim != 1 or wt.ndim != 1 or count_indices.ndim != 1:
        raise ValueError("EFS theta PIRLS requires X 2-D and aligned 1-D vectors")
    if (
        X.shape[0] != y.shape[0]
        or y.shape != wt.shape
        or y.shape != count_indices.shape
    ):
        raise ValueError("EFS theta PIRLS X/y/wt/count_indices must align")
    if offset is None:
        offset = jnp.zeros_like(y)
    if offset.shape != y.shape:
        raise ValueError("EFS theta PIRLS offset must align with y")
    if not jnp.issubdtype(offset.dtype, jnp.floating):
        raise ValueError("EFS theta PIRLS offset must have floating dtype")
    if not isinstance(initial_start_retained, bool):
        raise ValueError("EFS theta PIRLS initial_start_retained must be bool")
    if initial_eta is None:
        initial_eta = X @ beta_init + offset
    if initial_eta.shape != y.shape:
        raise ValueError("EFS theta PIRLS initial_eta must align with y")
    if not jnp.issubdtype(initial_eta.dtype, jnp.floating):
        raise ValueError("EFS theta PIRLS initial_eta must have floating dtype")
    return _efs_theta_pirls_loop_jit(
        X,
        y,
        beta_init,
        S_lambda,
        family,
        wt,
        offset,
        log_theta_init,
        beta_old_init,
        initial_eta,
        jnp.asarray(initial_start_retained),
        count_indices,
        max_y=max_y,
        integer_counts=integer_counts,
        max_iter=max_iter,
        tol=tol,
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
    extended_observed: bool = False,
) -> PIRLSResult:
    """Run PIRLS, passing estimated theta as dynamic JAX data.

    ``family`` is a JIT static argument, so any mutable family state read by
    the jitted implementation is baked into the compiled executable. For
    estimated extended families default ``log_theta`` from the family state
    here, before JIT dispatch, so theta participates in the cache as a regular
    array argument instead of as static Python object state. ``extended_observed``
    is an EFS-only fixed-theta NB opt-in; default fitting behavior is unchanged.
    """
    if extended_observed and not isinstance(family, ExtendedFamily):
        raise ValueError("extended_observed requires an ExtendedFamily")
    if (family.n_theta > 0 or extended_observed) and log_theta is None:
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
        extended_observed=extended_observed,
    )


pirls_loop.clear_cache = _pirls_loop_jit.clear_cache  # type: ignore[attr-defined]
pirls_loop._cache_size = _pirls_loop_jit._cache_size  # type: ignore[attr-defined]
