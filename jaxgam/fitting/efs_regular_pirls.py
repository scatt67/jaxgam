"""EFS-only regular-family PIRLS with pinned ``gam.fit3`` scoring.

The ordinary dense PIRLS loop deliberately retains its historical Fisher
coefficient steps.  This module is the opt-in EFS counterpart: noncanonical
regular links use the pinned full-Newton ``alpha`` system, while the final
result retains observed score factors and Fisher EDF/covariance factors.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting.pirls import PIRLSResult, _is_valid_trial, _signed_XtWX

EFS_REGULAR_STATUS_CONVERGED = 0
EFS_REGULAR_STATUS_ITERATION_LIMIT = 1
EFS_REGULAR_STATUS_INVALID_START = 2
EFS_REGULAR_STATUS_NONFINITE_COEFFICIENTS = 3
EFS_REGULAR_STATUS_NONFINITE_RECOVERY_FAILED = 4
EFS_REGULAR_STATUS_DOMAIN_RECOVERY_FAILED = 5
EFS_REGULAR_STATUS_DIVERGENCE_RECOVERY_FAILED = 6
EFS_REGULAR_STATUS_INVALID_WORKING_FACTORS = 7
EFS_REGULAR_STATUS_RANK_BOUNDARY_UNSUPPORTED = 8

_DIVERGENCE_ABS = 10.0 * 0.1 * jnp.sqrt(jnp.finfo(jnp.float64).eps)
_DIVERGENCE_REL = 10.0 * jnp.sqrt(jnp.finfo(jnp.float64).eps)
_DIVERGENCE_MAX_HALVINGS = 100
_INITIAL_START_MAX_HALVINGS = 20


@dataclass(frozen=True)
class EFSRegularPIRLSResult:
    """A compatible PIRLS result plus EFS-specific inner-loop provenance."""

    pirls_result: PIRLSResult
    status: jax.Array
    used_fisher_fallback: jax.Array
    n_fisher_fallbacks: jax.Array
    pre_gdi1_coefficients: jax.Array
    pre_gdi1_deviance: jax.Array
    pre_gdi1_penalized_deviance: jax.Array
    gdi1_coefficients: jax.Array
    gdi1_penalty: jax.Array
    gdi1_candidate_valid: jax.Array
    gdi1_coefficients_selected: jax.Array


_EFS_REGULAR_RESULT_FIELDS = [field.name for field in fields(EFSRegularPIRLSResult)]
jax.tree_util.register_pytree_node(
    EFSRegularPIRLSResult,
    lambda result: (
        [getattr(result, field) for field in _EFS_REGULAR_RESULT_FIELDS],
        None,
    ),
    lambda _, values: EFSRegularPIRLSResult(
        **dict(zip(_EFS_REGULAR_RESULT_FIELDS, values, strict=True))
    ),
)


@dataclass(frozen=True)
class _State:
    iteration: jax.Array
    beta: jax.Array
    eta: jax.Array
    mu: jax.Array
    anchor_beta: jax.Array
    anchor_eta: jax.Array
    baseline: jax.Array
    converged: jax.Array
    failed: jax.Array
    status: jax.Array
    used_fisher_fallback: jax.Array
    n_fisher_fallbacks: jax.Array


_STATE_FIELDS = [field.name for field in fields(_State)]
jax.tree_util.register_pytree_node(
    _State,
    lambda state: ([getattr(state, field) for field in _STATE_FIELDS], None),
    lambda _, values: _State(**dict(zip(_STATE_FIELDS, values, strict=True))),
)


@dataclass(frozen=True)
class _RecoveryState:
    beta: jax.Array
    eta: jax.Array
    mu: jax.Array
    deviance: jax.Array
    penalized_deviance: jax.Array
    n_halvings: jax.Array


_RECOVERY_FIELDS = [field.name for field in fields(_RecoveryState)]
jax.tree_util.register_pytree_node(
    _RecoveryState,
    lambda state: ([getattr(state, field) for field in _RECOVERY_FIELDS], None),
    lambda _, values: _RecoveryState(
        **dict(zip(_RECOVERY_FIELDS, values, strict=True))
    ),
)


def _regular_factors(
    family: ExponentialFamily,
    y: jax.Array,
    wt: jax.Array,
    offset: jax.Array,
    mu: jax.Array,
    eta: jax.Array,
    *,
    fisher: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return R ``gam.fit3`` W/z and its informative-row validity mask."""
    variance = family.variance(mu)
    mu_eta = family.link.mu_eta(eta)
    informative = (wt > 0.0) & (mu_eta != 0.0)
    safe_variance = jnp.where(informative, variance, 1.0)
    safe_mu_eta = jnp.where(informative, mu_eta, 1.0)
    safe_mu = jnp.where(informative, mu, 0.0)
    safe_y = jnp.where(informative, y, 0.0)
    fisher_weight = wt * safe_mu_eta**2 / safe_variance
    fisher_response = (eta - offset) + (safe_y - safe_mu) / safe_mu_eta
    if fisher:
        return (
            jnp.where(informative, fisher_weight, 0.0),
            jnp.where(informative, fisher_response, 0.0),
            informative,
        )
    alpha_raw = 1.0 + (safe_y - safe_mu) * (
        family.dvar(safe_mu) / safe_variance
        + family.link.second_derivative(safe_mu) * safe_mu_eta
    )
    alpha = jnp.where(alpha_raw == 0.0, jnp.finfo(jnp.float64).eps, alpha_raw)
    weight = fisher_weight * alpha
    response = (eta - offset) + (safe_y - safe_mu) / (safe_mu_eta * alpha)
    return (
        jnp.where(informative, weight, 0.0),
        jnp.where(informative, response, 0.0),
        informative,
    )


def _form_system(
    X: jax.Array, weight: jax.Array, response: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Form direct signed WLS normal equations without ordinary jitter."""
    return (weight[:, None] * X).T @ X, X.T @ (weight * response)


def _checked_exact_solve(
    XtWX: jax.Array, S_lambda: jax.Array, rhs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve the first full-rank EFS layer and classify its signed curvature.

    ``pls_fit1`` can retain tiny nonpositive directions through a pseudoinverse.
    That rank-boundary behavior is not implemented in this dense first layer:
    it reports a distinct status.  A material negative eigenvalue triggers the
    source's same-step Fisher retry; a bare Cholesky failure never does.
    """
    H = XtWX + S_lambda
    eig = jnp.linalg.eigvalsh(H)
    finite = jnp.all(jnp.isfinite(H)) & jnp.all(jnp.isfinite(rhs))
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(eig)))
    rank_tol = 100.0 * jnp.finfo(jnp.float64).eps
    indefinite = finite & (jnp.min(eig) < -rank_tol * scale)
    rank_boundary = finite & ~indefinite & (jnp.min(eig) <= 0.0)
    L = jnp.linalg.cholesky(H)
    beta = jsla.cho_solve((L, True), rhs)
    solved = (
        finite
        & ~indefinite
        & ~rank_boundary
        & jnp.all(jnp.isfinite(L))
        & jnp.all(jnp.diag(L) > 0.0)
        & jnp.all(jnp.isfinite(beta))
    )
    return beta, L, solved, indefinite


def _recover(
    raw: _RecoveryState,
    *,
    y: jax.Array,
    wt: jax.Array,
    family: ExponentialFamily,
    S_lambda: jax.Array,
    anchor_beta: jax.Array,
    anchor_eta: jax.Array,
    max_halvings: int,
    require_finite_deviance: bool,
) -> _RecoveryState:
    """Halve explicit beta/eta coordinates toward one pinned-R anchor."""

    def midpoint(state: _RecoveryState) -> _RecoveryState:
        beta = (state.beta + anchor_beta) / 2.0
        eta = (state.eta + anchor_eta) / 2.0
        mu = family.link.inverse(eta)
        deviance = family.dev_resids(y, mu, wt)
        return _RecoveryState(
            beta=beta,
            eta=eta,
            mu=mu,
            deviance=deviance,
            penalized_deviance=deviance + beta @ S_lambda @ beta,
            n_halvings=state.n_halvings + 1,
        )

    if require_finite_deviance:

        def condition(state: _RecoveryState) -> jax.Array:
            return ~jnp.isfinite(state.deviance) & (state.n_halvings < max_halvings)

    else:

        def condition(state: _RecoveryState) -> jax.Array:
            return ~_is_valid_trial(family, state.mu, state.eta) & (
                state.n_halvings < max_halvings
            )

    return jax.lax.while_loop(condition, midpoint, raw)


@jax.jit(
    static_argnames=("family", "start_present", "max_iter", "tol"),
)
def _efs_regular_pirls_loop_jit(
    X: jax.Array,
    y: jax.Array,
    beta_start: jax.Array,
    initial_eta: jax.Array,
    null_beta: jax.Array,
    null_eta: jax.Array,
    S_lambda: jax.Array,
    family: ExponentialFamily,
    wt: jax.Array,
    offset: jax.Array,
    scale: jax.Array,
    *,
    start_present: bool,
    max_iter: int,
    tol: float,
) -> EFSRegularPIRLSResult:
    """Compiled EFS regular loop for a static family/start-origin policy."""
    initial_mu = family.link.inverse(initial_eta)
    null_mu = family.link.inverse(null_eta)
    null_dev = family.dev_resids(y, null_mu, wt)
    null_pdev = null_dev + null_beta @ S_lambda @ null_beta
    # gam.fit3:288-294 shrinks an invalid initial eta by .9/.1 toward the
    # null anchor. A supplied beta follows the same coordinates; an absent
    # start retains its beta placeholder while eta has no beta representation.
    initial = _RecoveryState(
        beta=beta_start,
        eta=initial_eta,
        mu=initial_mu,
        deviance=family.dev_resids(y, initial_mu, wt),
        penalized_deviance=jnp.array(0.0),
        n_halvings=jnp.array(0, dtype=jnp.int32),
    )

    def start_condition(state: _RecoveryState) -> jax.Array:
        return ~_is_valid_trial(family, state.mu, state.eta) & (
            state.n_halvings < _INITIAL_START_MAX_HALVINGS
        )

    def start_body(state: _RecoveryState) -> _RecoveryState:
        beta = 0.9 * state.beta + 0.1 * null_beta if start_present else state.beta
        eta = 0.9 * state.eta + 0.1 * null_eta
        mu = family.link.inverse(eta)
        deviance = family.dev_resids(y, mu, wt)
        return _RecoveryState(
            beta=beta,
            eta=eta,
            mu=mu,
            deviance=deviance,
            penalized_deviance=deviance + beta @ S_lambda @ beta,
            n_halvings=state.n_halvings + 1,
        )

    started = jax.lax.while_loop(start_condition, start_body, initial)
    start_ok = _is_valid_trial(family, started.mu, started.eta)
    state = _State(
        iteration=jnp.array(0, dtype=jnp.int32),
        beta=started.beta,
        eta=started.eta,
        mu=started.mu,
        # R gam.fit3 resets this anchor to null.coef/null.eta even when a
        # retained coefficient start was supplied (lines 275-285).
        anchor_beta=null_beta,
        anchor_eta=null_eta,
        baseline=null_pdev,
        converged=jnp.array(False),
        failed=~start_ok,
        status=jnp.where(
            start_ok,
            jnp.array(EFS_REGULAR_STATUS_ITERATION_LIMIT, dtype=jnp.int32),
            jnp.array(EFS_REGULAR_STATUS_INVALID_START, dtype=jnp.int32),
        ),
        used_fisher_fallback=jnp.array(False),
        n_fisher_fallbacks=jnp.array(0, dtype=jnp.int32),
    )

    def condition(current: _State) -> jax.Array:
        return (current.iteration < max_iter) & ~current.converged & ~current.failed

    def body(current: _State) -> _State:
        use_fisher = family.is_canonical
        W_obs, z_obs, good = _regular_factors(
            family, y, wt, offset, current.mu, current.eta, fisher=use_fisher
        )
        factors_valid = (
            jnp.any(good) & jnp.all(jnp.isfinite(W_obs)) & jnp.all(jnp.isfinite(z_obs))
        )
        XtWX_obs, rhs_obs = _form_system(X, W_obs, z_obs)
        beta_obs, _L_obs, obs_solved, obs_indefinite = _checked_exact_solve(
            XtWX_obs, S_lambda, rhs_obs
        )

        W_fisher, z_fisher, _ = _regular_factors(
            family, y, wt, offset, current.mu, current.eta, fisher=True
        )
        XtWX_fisher, rhs_fisher = _form_system(X, W_fisher, z_fisher)
        beta_fisher, _L_fisher, fisher_solved, _ = _checked_exact_solve(
            XtWX_fisher, S_lambda, rhs_fisher
        )
        retry_fisher = jnp.asarray(not use_fisher) & obs_indefinite
        beta_raw = jnp.where(retry_fisher, beta_fisher, beta_obs)
        W_used = jnp.where(retry_fisher, W_fisher, W_obs)
        z_used = jnp.where(retry_fisher, z_fisher, z_obs)
        solve_ok = jnp.where(retry_fisher, fisher_solved, obs_solved)
        solve_ok = factors_valid & solve_ok

        eta_raw = X @ beta_raw + offset
        mu_raw = family.link.inverse(eta_raw)
        dev_raw = family.dev_resids(y, mu_raw, wt)
        raw = _RecoveryState(
            beta=beta_raw,
            eta=eta_raw,
            mu=mu_raw,
            deviance=dev_raw,
            penalized_deviance=dev_raw + beta_raw @ S_lambda @ beta_raw,
            n_halvings=jnp.array(0, dtype=jnp.int32),
        )

        # Pinned order: a non-finite coefficient terminates before any
        # deviance recovery. Otherwise recover non-finite deviance, then
        # domain, then penalized-deviance divergence.
        coefficients_finite = jnp.all(jnp.isfinite(beta_raw))
        after_nonfinite = _recover(
            raw,
            y=y,
            wt=wt,
            family=family,
            S_lambda=S_lambda,
            anchor_beta=current.anchor_beta,
            anchor_eta=current.anchor_eta,
            max_halvings=max_iter,
            require_finite_deviance=True,
        )
        nonfinite_failed = ~jnp.isfinite(after_nonfinite.deviance)
        domain_initial = _RecoveryState(
            beta=after_nonfinite.beta,
            eta=after_nonfinite.eta,
            mu=after_nonfinite.mu,
            deviance=after_nonfinite.deviance,
            penalized_deviance=after_nonfinite.penalized_deviance,
            n_halvings=jnp.array(0, dtype=jnp.int32),
        )
        after_domain = _recover(
            domain_initial,
            y=y,
            wt=wt,
            family=family,
            S_lambda=S_lambda,
            anchor_beta=current.anchor_beta,
            anchor_eta=current.anchor_eta,
            max_halvings=max_iter,
            require_finite_deviance=False,
        )
        domain_failed = ~_is_valid_trial(family, after_domain.mu, after_domain.eta)
        divergence_threshold = _DIVERGENCE_ABS + _DIVERGENCE_REL * jnp.abs(
            current.baseline
        )
        divergence_initial = _RecoveryState(
            beta=after_domain.beta,
            eta=after_domain.eta,
            mu=after_domain.mu,
            deviance=after_domain.deviance,
            penalized_deviance=after_domain.penalized_deviance,
            n_halvings=jnp.array(0, dtype=jnp.int32),
        )

        def divergence_condition(value: _RecoveryState) -> jax.Array:
            return (
                value.penalized_deviance - current.baseline > divergence_threshold
            ) & (value.n_halvings < _DIVERGENCE_MAX_HALVINGS)

        def divergence_body(value: _RecoveryState) -> _RecoveryState:
            beta = (value.beta + current.anchor_beta) / 2.0
            eta = (value.eta + current.anchor_eta) / 2.0
            mu = family.link.inverse(eta)
            deviance = family.dev_resids(y, mu, wt)
            return _RecoveryState(
                beta=beta,
                eta=eta,
                mu=mu,
                deviance=deviance,
                penalized_deviance=deviance + beta @ S_lambda @ beta,
                n_halvings=value.n_halvings + 1,
            )

        after_divergence = jax.lax.while_loop(
            divergence_condition, divergence_body, divergence_initial
        )
        divergence_failed = (
            after_divergence.penalized_deviance - current.baseline
            > divergence_threshold
        )
        valid_final = (
            jnp.all(jnp.isfinite(after_divergence.beta))
            & jnp.all(jnp.isfinite(after_divergence.eta))
            & jnp.all(jnp.isfinite(after_divergence.mu))
            & jnp.isfinite(after_divergence.deviance)
            & jnp.isfinite(after_divergence.penalized_deviance)
            & _is_valid_trial(family, after_divergence.mu, after_divergence.eta)
        )
        accepted = solve_ok & coefficients_finite & ~nonfinite_failed & ~domain_failed
        accepted &= ~divergence_failed & valid_final
        rank_boundary = ~obs_solved & ~obs_indefinite & ~retry_fisher
        status = jnp.where(
            ~factors_valid,
            jnp.array(EFS_REGULAR_STATUS_INVALID_WORKING_FACTORS, dtype=jnp.int32),
            jnp.where(
                ~coefficients_finite,
                jnp.array(EFS_REGULAR_STATUS_NONFINITE_COEFFICIENTS, dtype=jnp.int32),
                jnp.where(
                    rank_boundary,
                    jnp.array(
                        EFS_REGULAR_STATUS_RANK_BOUNDARY_UNSUPPORTED,
                        dtype=jnp.int32,
                    ),
                    jnp.where(
                        nonfinite_failed,
                        jnp.array(
                            EFS_REGULAR_STATUS_NONFINITE_RECOVERY_FAILED,
                            dtype=jnp.int32,
                        ),
                        jnp.where(
                            domain_failed,
                            jnp.array(
                                EFS_REGULAR_STATUS_DOMAIN_RECOVERY_FAILED,
                                dtype=jnp.int32,
                            ),
                            jnp.where(
                                divergence_failed,
                                jnp.array(
                                    EFS_REGULAR_STATUS_DIVERGENCE_RECOVERY_FAILED,
                                    dtype=jnp.int32,
                                ),
                                jnp.array(
                                    EFS_REGULAR_STATUS_ITERATION_LIMIT,
                                    dtype=jnp.int32,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

        # gam.fit3's convergence gradient is built from the same selected
        # system used for this iteration, before fresh final factors.
        fitted_without_offset = X @ after_divergence.beta
        gradient = (
            2.0 * X.T @ (W_used * (fitted_without_offset - z_used))
            + 2.0 * S_lambda @ after_divergence.beta
        )
        pdev_change = jnp.abs(after_divergence.penalized_deviance - current.baseline)
        pdev_scale = tol * (
            jnp.abs(scale) + jnp.abs(after_divergence.penalized_deviance)
        )
        gradient_scale = tol * (
            jnp.abs(after_divergence.penalized_deviance) + jnp.abs(scale)
        )
        strictly_additive = family.family_name == "gaussian" and family.is_canonical
        converged = accepted & (
            strictly_additive
            | (
                (pdev_change < pdev_scale)
                & (jnp.max(jnp.abs(gradient)) <= gradient_scale)
            )
        )
        next_anchor_beta = jnp.where(
            accepted, after_divergence.beta, current.anchor_beta
        )
        next_anchor_eta = jnp.where(accepted, after_divergence.eta, current.anchor_eta)
        return _State(
            iteration=current.iteration + 1,
            beta=jnp.where(accepted, after_divergence.beta, current.beta),
            eta=jnp.where(accepted, after_divergence.eta, current.eta),
            mu=jnp.where(accepted, after_divergence.mu, current.mu),
            anchor_beta=next_anchor_beta,
            anchor_eta=next_anchor_eta,
            baseline=jnp.where(
                accepted, after_divergence.penalized_deviance, current.baseline
            ),
            converged=converged,
            failed=~accepted,
            status=jnp.where(
                converged,
                jnp.array(EFS_REGULAR_STATUS_CONVERGED, dtype=jnp.int32),
                status,
            ),
            used_fisher_fallback=current.used_fisher_fallback | retry_fisher,
            n_fisher_fallbacks=current.n_fisher_fallbacks
            + retry_fisher.astype(jnp.int32),
        )

    final = jax.lax.while_loop(condition, body, state)
    # gam.fit3:470-581 freezes the loop deviance, rebuilds W/z at that
    # pre-gdi1 state, and lets gdi1 perform one last penalized WLS solve.
    # Determinant, EDF, and covariance factors all belong to these pre-polish
    # weights.  Only the returned coefficient/eta/mu state can move afterward.
    pre_gdi1_beta = final.beta
    pre_gdi1_eta = final.eta
    pre_gdi1_mu = final.mu
    pre_gdi1_deviance = family.dev_resids(y, pre_gdi1_mu, wt)
    pre_gdi1_penalty = pre_gdi1_beta @ S_lambda @ pre_gdi1_beta
    W_obs, z_obs, _ = _regular_factors(
        family, y, wt, offset, final.mu, final.eta, fisher=family.is_canonical
    )
    W_fisher, _z_fisher, _ = _regular_factors(
        family, y, wt, offset, final.mu, final.eta, fisher=True
    )
    XtWX = _signed_XtWX(W_obs, X)
    XtWX_fisher = _signed_XtWX(W_fisher, X)
    L = jnp.linalg.cholesky(XtWX + S_lambda)
    L_fisher = jnp.linalg.cholesky(XtWX_fisher + S_lambda)
    gdi1_beta, _gdi1_L, gdi1_solved, _gdi1_indefinite = _checked_exact_solve(
        XtWX, S_lambda, X.T @ (W_obs * z_obs)
    )
    gdi1_eta = X @ gdi1_beta + offset
    gdi1_mu = family.link.inverse(gdi1_eta)
    gdi1_candidate_valid = (
        gdi1_solved
        & jnp.all(jnp.isfinite(gdi1_beta))
        & _is_valid_trial(family, gdi1_mu, gdi1_eta)
    )
    # gam.fit3:581-593 returns the pre-gdi1 feasible coordinates if the gdi1
    # candidate leaves the family domain.  C_gdi1's bSb nevertheless remains
    # the candidate penalty used in Dp, so retain it independently.
    selected_beta = jnp.where(gdi1_candidate_valid, gdi1_beta, pre_gdi1_beta)
    selected_eta = jnp.where(gdi1_candidate_valid, gdi1_eta, pre_gdi1_eta)
    selected_mu = jnp.where(gdi1_candidate_valid, gdi1_mu, pre_gdi1_mu)
    gdi1_penalty = gdi1_beta @ S_lambda @ gdi1_beta
    final_status = jnp.where(
        final.converged & ~gdi1_solved,
        jnp.array(EFS_REGULAR_STATUS_RANK_BOUNDARY_UNSUPPORTED, dtype=jnp.int32),
        final.status,
    )
    final_converged = final.converged & gdi1_solved
    pirls_result = PIRLSResult(
        coefficients=selected_beta,
        mu=selected_mu,
        eta=selected_eta,
        # R's public deviance remains the pre-gdi1 loop deviance.  Its REML
        # Dp instead adds the separately returned gdi1 candidate penalty.
        deviance=pre_gdi1_deviance,
        penalized_deviance=pre_gdi1_deviance + gdi1_penalty,
        n_iter=final.iteration,
        converged=final_converged,
        scale=pre_gdi1_deviance / jnp.maximum(X.shape[0] - X.shape[1], 1),
        XtWX=XtWX,
        L=L,
        working_weights=W_obs,
        XtWX_fisher=XtWX_fisher,
        L_fisher=L_fisher,
    )
    return EFSRegularPIRLSResult(
        pirls_result=pirls_result,
        status=final_status,
        used_fisher_fallback=final.used_fisher_fallback,
        n_fisher_fallbacks=final.n_fisher_fallbacks,
        pre_gdi1_coefficients=pre_gdi1_beta,
        pre_gdi1_deviance=pre_gdi1_deviance,
        pre_gdi1_penalized_deviance=pre_gdi1_deviance + pre_gdi1_penalty,
        gdi1_coefficients=gdi1_beta,
        gdi1_penalty=gdi1_penalty,
        gdi1_candidate_valid=gdi1_candidate_valid,
        gdi1_coefficients_selected=gdi1_candidate_valid,
    )


def efs_regular_pirls_loop(
    X: jax.Array,
    y: jax.Array,
    beta_start: jax.Array,
    initial_eta: jax.Array,
    null_beta: jax.Array,
    null_eta: jax.Array,
    S_lambda: jax.Array,
    family: ExponentialFamily,
    wt: jax.Array,
    offset: jax.Array | None,
    scale: jax.Array,
    *,
    start_present: bool,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> EFSRegularPIRLSResult:
    """Run the dedicated JIT-compatible EFS regular-family PIRLS loop."""
    if offset is None:
        offset = jnp.zeros(X.shape[0], dtype=X.dtype)
    return _efs_regular_pirls_loop_jit(
        X,
        y,
        beta_start,
        initial_eta,
        null_beta,
        null_eta,
        S_lambda,
        family,
        wt,
        offset,
        scale,
        start_present=start_present,
        max_iter=max_iter,
        tol=tol,
    )
