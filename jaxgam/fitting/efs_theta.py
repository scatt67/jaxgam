"""Pure conditional theta Newton kernel for estimated-NB EFS.

This module ports the one-parameter ``estimate.theta`` branch used by
``mgcv::gam.fit4(scoreType='EFS')``.  It deliberately does not invoke PIRLS or
mutate a family: callers provide the current fixed linear predictor and the
theta coordinate explicitly.  EFS integration owns when this kernel runs.

R source reference: efam.r ``estimate.theta`` (mgcv 1.9-3).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from numbers import Integral, Real

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.links.links import IdentityLink, LogLink, SqrtLink

_EPS_075 = np.finfo(np.float64).eps ** 0.75
_CURVATURE_FLOOR_RELATIVE = 1e-5
_MAX_STEP = 4.0
_MAX_HALVINGS = 25
_MAX_ITER = 100
_TOLERANCE = 1e-7

_STATUS_CONVERGED = 0
_STATUS_INITIAL_NONFINITE = 1
_STATUS_NONFINITE_CURVATURE = 2
_STATUS_ZERO_CURVATURE = 3
_STATUS_NONFINITE_STEP = 4
_STATUS_LINE_SEARCH_FAILED = 5
_STATUS_ITERATION_LIMIT = 6
_STATUS_POST_ACCEPT_NONFINITE = 7
_STATUS_UNSUPPORTED_FRACTIONAL_RESPONSE = 8

# ``log1p((y - mu) / (mu + theta))`` is accurate near the mean, but the
# quotient rounds to exactly -1 when ``mu`` is much larger than ``theta``.
# Keep it a full half-unit away from that singular boundary. The tail uses
# the literal mgcv log-ratio in log space instead.
_NEAR_LOG1P_MINIMUM = -0.5
_LOG_MIN_SUBNORMAL = np.log(np.nextafter(np.float64(0.0), np.float64(1.0)))
_LOG_MAX_FLOAT = np.log(np.finfo(np.float64).max)


@dataclass(frozen=True)
class EFSThetaResult:
    """Bounded diagnostics from one fixed-eta conditional theta solve.

    ``status`` values are 0 (converged), 1 (nonfinite initial objective), 2
    (nonfinite curvature), 3 (zero curvature after the R repair), 4
    (nonfinite step), 5 (line-search failure), 6 (iteration limit), and 7
    (nonfinite derivatives at an otherwise accepted trial). The latter retains
    the prior valid theta and objective. Status 8 rejects fractional responses
    in ``(0, 1)``: their inherited NB deviance convention differs from
    mgcv's, so the R controller thresholds are not comparable.
    ``nll_history`` includes the initial objective followed by accepted trials;
    entries after ``n_history`` are unused zeros.
    """

    log_theta: jax.Array
    nll: jax.Array
    gradient: jax.Array
    hessian: jax.Array
    n_iter: jax.Array
    converged: jax.Array
    status: jax.Array
    nll_history: jax.Array
    n_history: jax.Array


_RESULT_FIELDS = [field.name for field in fields(EFSThetaResult)]
jax.tree_util.register_pytree_node(
    EFSThetaResult,
    lambda result: ([getattr(result, field) for field in _RESULT_FIELDS], None),
    lambda _, values: EFSThetaResult(**dict(zip(_RESULT_FIELDS, values, strict=True))),
)


def _require_estimated_nb(family: NegativeBinomial) -> None:
    if not isinstance(family, NegativeBinomial):
        raise TypeError("EFS conditional theta requires NegativeBinomial")
    if family.n_theta != 1:
        raise ValueError("EFS conditional theta requires an estimated NB theta")
    if not isinstance(family.link, (LogLink, IdentityLink, SqrtLink)):
        raise NotImplementedError(
            "EFS conditional theta supports the pinned NB links log, identity, and sqrt"
        )


def _validate_static_inputs(
    log_theta: jax.Array,
    eta: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    count_indices: jax.Array,
    *,
    max_y: int,
    integer_counts: bool,
    tolerance: float | None = None,
    max_iter: int | None = None,
    max_step: float | None = None,
    max_halvings: int | None = None,
) -> None:
    if log_theta.shape != (1,):
        raise ValueError("conditional theta log_theta must have shape (1,)")
    if eta.ndim != 1 or y.ndim != 1 or wt.ndim != 1 or count_indices.ndim != 1:
        raise ValueError("conditional theta inputs must be one-dimensional")
    if not (eta.shape == y.shape == wt.shape == count_indices.shape):
        raise ValueError("conditional theta eta/y/wt/count_indices must align")
    for name, value, minimum in (
        ("max_y", max_y, 0),
        ("max_iter", max_iter, 1),
        ("max_halvings", max_halvings, 0),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(
                f"conditional theta {name} must be an integer >= {minimum}"
            )
    if not isinstance(integer_counts, bool):
        raise ValueError("conditional theta integer_counts must be bool")
    for name, value in (("tolerance", tolerance), ("max_step", max_step)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"conditional theta {name} must be finite and positive")


def _efs_nb_log_deviance(
    eta: jax.Array,
    log_theta: jax.Array,
    y: jax.Array,
    wt: jax.Array,
) -> jax.Array:
    """EFS-only NB/log deviance with the pinned mgcv validity contract.

    This is ``efam.r``'s ``nb()$dev.resids`` in eta coordinates. Around the
    mean it retains the stable ``log1p`` identity; for the large-mu tail it
    evaluates the literal log ratio as ``log(y + theta) - logaddexp(eta,
    log_theta)``. The latter avoids the inherited family's cancellation to
    ``log1p(-1)`` while preserving R's non-finite recovery trigger whenever
    the actual log-link mean is non-finite or non-positive. XLA flushes
    subnormal link means on the supported accelerator path, so those remain a
    deliberately fail-closed boundary rather than claimed R parity.

    It is intentionally private to estimated-theta EFS. Default Newton and
    fixed-theta NB continue to use :meth:`NegativeBinomial.deviance_fn`.
    """
    fractional_below_one = jnp.any((y > 0.0) & (y < 1.0))

    def inherited(_: None) -> jax.Array:
        """Existing family expression retained under the status-8 guard."""
        theta = jnp.exp(log_theta[0])
        mu_safe = jnp.maximum(jnp.exp(eta), 1e-10)
        y_safe = jnp.where(y > 0.0, y, 1.0)
        return jnp.sum(
            2.0
            * wt
            * (
                y * jnp.log(y_safe / mu_safe)
                - (y + theta) * jnp.log1p((y - mu_safe) / (mu_safe + theta))
            )
        )

    def supported(_: None) -> jax.Array:
        theta_raw = jnp.exp(log_theta[0])
        theta_valid = jnp.isfinite(theta_raw) & (theta_raw > 0.0)
        theta = jnp.where(theta_valid, theta_raw, 1.0)
        log_theta_safe = jnp.where(theta_valid, log_theta[0], 0.0)

        eta_finite = jnp.isfinite(eta)
        eta_safe = jnp.where(eta_finite, eta, 0.0)
        mu_raw = jnp.exp(eta_safe)
        mu_valid = eta_finite & jnp.isfinite(mu_raw) & (mu_raw > 0.0)
        mu = jnp.where(mu_valid, mu_raw, 1.0)
        eta_for_value = jnp.where(mu_valid, eta_safe, 0.0)

        y_valid = jnp.isfinite(y) & (y >= 0.0)
        wt_valid = jnp.isfinite(wt) & (wt >= 0.0)
        y_safe = jnp.where(y_valid, y, 0.0)
        wt_safe = jnp.where(wt_valid, wt, 0.0)

        y_theta_raw = y_safe + theta
        denominator_raw = mu + theta
        y_theta_valid = jnp.isfinite(y_theta_raw) & (y_theta_raw > 0.0)
        denominator_valid = jnp.isfinite(denominator_raw) & (denominator_raw > 0.0)
        y_theta = jnp.where(y_theta_valid, y_theta_raw, 1.0)
        denominator = jnp.where(denominator_valid, denominator_raw, 1.0)

        # R evaluates both ratios literally before applying log. Test those
        # machine operations after replacing invalid inputs with finite,
        # positive placeholders. XLA can flush a *subnormal quotient* to zero
        # despite retaining its normal mean, so additionally admit a
        # log-domain representable quotient. A flushed/non-finite mean itself
        # is never rescued: it remains a recovery signal.
        literal_first_ratio = jnp.maximum(1.0, y_safe) / mu
        literal_second_ratio = y_theta / denominator
        literal_ratio_representable = (
            jnp.isfinite(literal_first_ratio)
            & (literal_first_ratio > 0.0)
            & jnp.isfinite(literal_second_ratio)
            & (literal_second_ratio > 0.0)
        )
        first_log_ratio = jnp.log(jnp.maximum(1.0, y_safe)) - eta_for_value
        second_log_ratio = jnp.log(y_theta) - jnp.logaddexp(
            eta_for_value, log_theta_safe
        )
        log_ratio_representable = (
            jnp.isfinite(first_log_ratio)
            & (first_log_ratio >= _LOG_MIN_SUBNORMAL)
            & (first_log_ratio <= _LOG_MAX_FLOAT)
            & jnp.isfinite(second_log_ratio)
            & (second_log_ratio >= _LOG_MIN_SUBNORMAL)
            & (second_log_ratio <= _LOG_MAX_FLOAT)
        )
        ratio_representable = literal_ratio_representable | log_ratio_representable
        # ``(y - mu) / (mu + theta) > -.5`` iff ``mu < 2*y + theta``.
        # Determine the branch without differentiating a tail quotient: at a
        # finite eta near 709 its quotient derivatives form ``mu**2`` and
        # overflow even though that inactive branch is not selected. Only the
        # near branch receives its real numerator and denominator.
        near_threshold = y_safe + y_theta
        use_near_log1p = ratio_representable & (mu < near_threshold)
        near_numerator = jnp.where(use_near_log1p, y_safe - mu, 0.0)
        near_denominator = jnp.where(use_near_log1p, denominator, 1.0)
        near_log_ratio = jnp.log1p(near_numerator / near_denominator)
        tail_log_ratio = second_log_ratio
        stable_log_ratio = jnp.where(
            use_near_log1p,
            near_log_ratio,
            tail_log_ratio,
        )
        first_term = y_safe * (jnp.log(jnp.maximum(1.0, y_safe)) - eta_for_value)
        deviance = jnp.sum(2.0 * wt_safe * (first_term - y_theta * stable_log_ratio))
        input_valid = (
            theta_valid
            & jnp.all(mu_valid)
            & jnp.all(y_valid)
            & jnp.all(wt_valid)
            & jnp.all(y_theta_valid)
            & jnp.all(denominator_valid)
            & jnp.all(ratio_representable)
        )
        return jnp.where(input_valid, deviance, jnp.array(jnp.inf, dtype=eta.dtype))

    return jax.lax.cond(
        fractional_below_one,
        inherited,
        supported,
        operand=None,
    )


def _efs_nb_deviance(
    eta: jax.Array,
    log_theta: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    family: NegativeBinomial,
) -> jax.Array:
    """Evaluate pinned NB deviance for any constructor-supported link.

    The log link keeps its range-safe eta-space implementation. Identity and
    square-root links use the same explicit-theta deviance expression after a
    strict eta/mu domain check, matching ``nb()$dev.resids`` while retaining
    nonfinite values as recovery signals for ``gam.fit4``.
    """
    if isinstance(family.link, LogLink):
        return _efs_nb_log_deviance(eta, log_theta, y, wt)
    mu_raw = family.link.inverse(eta)
    valid = (
        jnp.all(jnp.isfinite(eta))
        & jnp.all(jnp.isfinite(mu_raw))
        & jnp.all(family.valid_eta(eta))
        & jnp.all(family.valid_mu(mu_raw))
    )
    mu = jnp.where(jnp.isfinite(mu_raw) & (mu_raw > 0.0), mu_raw, 1.0)
    theta = jnp.exp(log_theta[0])
    # Source uses ``pmax(1, y)`` in the first log term, including 0 < y < 1.
    y_safe = jnp.maximum(1.0, y)
    # Keep the source expression as separated logarithms.  Rewriting the
    # second ratio with log1p makes its argument round to -1 for large finite
    # mu, while ``log(y + theta) - log(mu + theta)`` remains representable.
    contributions = (
        2.0
        * wt
        * (
            y * (jnp.log(y_safe) - jnp.log(mu))
            - (y + theta) * (jnp.log(y + theta) - jnp.log(mu + theta))
        )
    )
    deviance = jnp.sum(contributions)
    return jnp.where(valid & jnp.isfinite(deviance), deviance, jnp.inf)


@partial(
    jax.jit,
    static_argnames=("family", "max_y", "integer_counts"),
)
def _conditional_theta_nll_jit(
    log_theta: jax.Array,
    eta: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    count_indices: jax.Array,
    family: NegativeBinomial,
    *,
    max_y: int,
    integer_counts: bool,
) -> jax.Array:
    """Inherited-family conditional NB objective at fixed linear predictor.

    For integer responses and fractional responses at least one, this is the
    pinned mgcv ``estimate.theta`` objective. The inherited NB deviance
    convention differs from mgcv for ``0 < y < 1``; the controller rejects
    that domain rather than claiming R-compatible convergence there.
    """
    deviance = _efs_nb_deviance(eta, log_theta, y, wt, family)
    saturated = family.saturated_loglik_theta(
        y,
        wt,
        1.0,
        log_theta,
        max_y=max_y,
        count_indices=count_indices,
        integer_counts=integer_counts,
    )
    return deviance / 2.0 - saturated


def conditional_theta_nll(
    log_theta: jax.Array,
    eta: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    count_indices: jax.Array,
    family: NegativeBinomial,
    *,
    max_y: int,
    integer_counts: bool,
) -> jax.Array:
    """Evaluate the inherited-family conditional NB objective.

    See ``_conditional_theta_nll_jit`` for the explicit pinned-R parity domain
    and fractional-response compatibility restriction.
    """
    _require_estimated_nb(family)
    _validate_static_inputs(
        log_theta,
        eta,
        y,
        wt,
        count_indices,
        max_y=max_y,
        integer_counts=integer_counts,
    )
    return _conditional_theta_nll_jit(
        log_theta,
        eta,
        y,
        wt,
        count_indices,
        family,
        max_y=max_y,
        integer_counts=integer_counts,
    )


@partial(
    jax.jit,
    static_argnames=(
        "family",
        "max_y",
        "integer_counts",
        "tolerance",
        "max_iter",
        "max_step",
        "max_halvings",
    ),
)
def _conditional_theta_newton_jit(
    log_theta: jax.Array,
    eta: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    count_indices: jax.Array,
    family: NegativeBinomial,
    *,
    max_y: int,
    integer_counts: bool,
    tolerance: float,
    max_iter: int,
    max_step: float,
    max_halvings: int,
) -> EFSThetaResult:
    """JIT implementation of mgcv's scalar safeguarded theta Newton loop."""

    def nll_fn(theta):
        return _conditional_theta_nll_jit(
            theta,
            eta,
            y,
            wt,
            count_indices,
            family,
            max_y=max_y,
            integer_counts=integer_counts,
        )

    value_grad_hessian = jax.value_and_grad(nll_fn), jax.hessian(nll_fn)

    def evaluate(theta):
        value, gradient = value_grad_hessian[0](theta)
        hessian = value_grad_hessian[1](theta)
        return value, gradient[0], hessian[0, 0]

    nll0, gradient0, hessian0 = evaluate(log_theta)
    initial_objective_finite = jnp.isfinite(nll0) & jnp.isfinite(gradient0)
    finite_initial = initial_objective_finite & jnp.isfinite(hessian0)
    unsupported_fractional_response = jnp.any((y > 0.0) & (y < 1.0))
    active0 = jnp.abs(gradient0) > tolerance * (jnp.abs(nll0) + 1.0)
    history = jnp.zeros((max_iter + 1,), dtype=nll0.dtype).at[0].set(nll0)

    state = (
        log_theta,
        nll0,
        gradient0,
        hessian0,
        jnp.array(0, dtype=jnp.int32),
        active0 & finite_initial & ~unsupported_fractional_response,
        ~finite_initial | unsupported_fractional_response,
        jnp.where(
            unsupported_fractional_response,
            jnp.array(_STATUS_UNSUPPORTED_FRACTIONAL_RESPONSE, dtype=jnp.int32),
            jnp.where(
                initial_objective_finite,
                jnp.where(
                    jnp.isfinite(hessian0),
                    jnp.array(_STATUS_CONVERGED, dtype=jnp.int32),
                    jnp.array(_STATUS_NONFINITE_CURVATURE, dtype=jnp.int32),
                ),
                jnp.array(_STATUS_INITIAL_NONFINITE, dtype=jnp.int32),
            ),
        ),
        history,
        jnp.array(1, dtype=jnp.int32),
    )

    def condition(state):
        _, _, _, _, iteration, active, failed, _, _, _ = state
        return (iteration < max_iter) & active & ~failed

    def body(state):
        (
            theta,
            nll,
            gradient,
            hessian,
            iteration,
            _,
            _,
            _,
            history,
            n_history,
        ) = state
        hessian_finite = jnp.isfinite(hessian)
        abs_hessian = jnp.abs(hessian)
        repaired_hessian = jnp.where(
            hessian > 0.0,
            hessian,
            jnp.maximum(abs_hessian, abs_hessian * _CURVATURE_FLOOR_RELATIVE),
        )
        zero_curvature = repaired_hessian == 0.0
        usable_curvature = hessian_finite & ~zero_curvature
        raw_step = -gradient / jnp.where(usable_curvature, repaired_hessian, 1.0)
        raw_step_finite = jnp.isfinite(raw_step)
        step = jnp.clip(raw_step, -max_step, max_step)
        usable_step = usable_curvature & raw_step_finite & jnp.isfinite(step)

        def line_search(step):
            candidate_theta = theta + step
            candidate_nll = nll_fn(candidate_theta)
            line_state = (
                step,
                candidate_nll,
                jnp.array(0, dtype=jnp.int32),
                jnp.isnan(candidate_nll),
            )

            def line_condition(line_state):
                _, trial_nll, _, failed = line_state
                worse = trial_nll - nll > _EPS_075 * jnp.abs(nll)
                return worse & ~failed

            def line_body(line_state):
                trial_step, trial_nll, halvings, _ = line_state
                next_step = trial_step / 2.0
                next_halvings = halvings + 1
                unchanged = jnp.all(theta == theta + next_step)
                failed = unchanged | (next_halvings > max_halvings)
                next_nll = jax.lax.cond(
                    failed,
                    lambda _: trial_nll,
                    lambda _: nll_fn(theta + next_step),
                    operand=None,
                )
                # R's comparison keeps halving a recoverable +Inf trial;
                # only NaN has no ordering and must fail closed.
                failed = failed | jnp.isnan(next_nll)
                return next_step, next_nll, next_halvings, failed

            return jax.lax.while_loop(line_condition, line_body, line_state)

        step, _, _, line_failed = jax.lax.cond(
            usable_step,
            line_search,
            lambda initial_step: (
                initial_step,
                nll,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(True),
            ),
            step,
        )
        status = jnp.where(
            ~hessian_finite,
            _STATUS_NONFINITE_CURVATURE,
            jnp.where(
                zero_curvature,
                _STATUS_ZERO_CURVATURE,
                jnp.where(
                    ~raw_step_finite, _STATUS_NONFINITE_STEP, _STATUS_LINE_SEARCH_FAILED
                ),
            ),
        ).astype(jnp.int32)
        failed = ~usable_step | line_failed

        def accept(_: None):
            next_theta = theta + step
            next_nll, next_gradient, next_hessian = evaluate(next_theta)
            next_finite = (
                jnp.isfinite(next_nll)
                & jnp.isfinite(next_gradient)
                & jnp.isfinite(next_hessian)
            )

            def keep_trial(_: None):
                next_history = history.at[n_history].set(next_nll)
                next_active = jnp.abs(next_gradient) > tolerance * (
                    jnp.abs(next_nll) + 1.0
                )
                return (
                    next_theta,
                    next_nll,
                    next_gradient,
                    next_hessian,
                    iteration + 1,
                    next_active,
                    jnp.array(False),
                    jnp.array(_STATUS_CONVERGED, dtype=jnp.int32),
                    next_history,
                    n_history + 1,
                )

            def retain_prior(_: None):
                return (
                    theta,
                    nll,
                    gradient,
                    hessian,
                    iteration + 1,
                    jnp.array(False),
                    jnp.array(True),
                    jnp.array(_STATUS_POST_ACCEPT_NONFINITE, dtype=jnp.int32),
                    history,
                    n_history,
                )

            return jax.lax.cond(next_finite, keep_trial, retain_prior, operand=None)

        def reject(_: None):
            return (
                theta,
                nll,
                gradient,
                hessian,
                iteration + 1,
                jnp.array(False),
                jnp.array(True),
                status,
                history,
                n_history,
            )

        return jax.lax.cond(failed, reject, accept, operand=None)

    (
        theta,
        nll,
        gradient,
        hessian,
        iteration,
        active,
        failed,
        status,
        history,
        n_history,
    ) = jax.lax.while_loop(condition, body, state)
    reached_limit = active & ~failed & (iteration == max_iter)
    status = jnp.where(reached_limit, _STATUS_ITERATION_LIMIT, status)
    return EFSThetaResult(
        theta,
        nll,
        gradient,
        hessian,
        iteration,
        ~active & ~failed,
        status,
        history,
        n_history,
    )


def conditional_theta_newton(
    log_theta: jax.Array,
    eta: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    count_indices: jax.Array,
    family: NegativeBinomial,
    *,
    max_y: int,
    integer_counts: bool,
    tolerance: float = _TOLERANCE,
    max_iter: int = _MAX_ITER,
    max_step: float = _MAX_STEP,
    max_halvings: int = _MAX_HALVINGS,
) -> EFSThetaResult:
    """Safeguard one NB log-theta coordinate at fixed eta, without mutation."""
    _require_estimated_nb(family)
    _validate_static_inputs(
        log_theta,
        eta,
        y,
        wt,
        count_indices,
        max_y=max_y,
        integer_counts=integer_counts,
        tolerance=tolerance,
        max_iter=max_iter,
        max_step=max_step,
        max_halvings=max_halvings,
    )
    return _conditional_theta_newton_jit(
        log_theta,
        eta,
        y,
        wt,
        count_indices,
        family,
        max_y=max_y,
        integer_counts=integer_counts,
        tolerance=tolerance,
        max_iter=max_iter,
        max_step=max_step,
        max_halvings=max_halvings,
    )
