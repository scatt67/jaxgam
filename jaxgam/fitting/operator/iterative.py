"""Pure vector recurrences for frozen-system preconditioned CG.

The resident solver is a new SPD algorithm, not an mgcv solver port.
Host row-source orchestration uses the same recurrences and owns every
operator scan. Recursive residuals only request verification; convergence
always requires a fresh action of the original frozen system.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

READY = 0
CONVERGED = 1
MAX_ITERATIONS = 2
NONFINITE = 3
NONPOSITIVE_CURVATURE = 4
VERIFY_RESIDUAL = 5


class PCGState(NamedTuple):
    solution: jax.Array
    residual: jax.Array
    preconditioned: jax.Array
    direction: jax.Array
    residual_product: jax.Array
    residual_norm: jax.Array
    iteration: jax.Array
    status: jax.Array


class PCGResult(NamedTuple):
    solution: jax.Array
    iteration: jax.Array
    status: jax.Array
    absolute_residual: jax.Array
    relative_residual: jax.Array
    matvec_count: jax.Array


def initialize_pcg(
    solution: jax.Array,
    residual: jax.Array,
    preconditioned: jax.Array,
    threshold: jax.Array,
) -> PCGState:
    """Initialize from an actual residual; reject unusable inverse actions."""
    norm = jnp.linalg.norm(residual)
    product = jnp.vdot(residual, preconditioned)
    finite = (
        jnp.all(jnp.isfinite(solution))
        & jnp.all(jnp.isfinite(residual))
        & jnp.all(jnp.isfinite(preconditioned))
        & jnp.isfinite(product)
        & jnp.isfinite(norm)
        & jnp.isfinite(threshold)
        & (threshold >= 0)
    )
    status = jnp.where(
        ~finite,
        NONFINITE,
        jnp.where(
            norm <= threshold,
            VERIFY_RESIDUAL,
            jnp.where(product > 0, READY, NONPOSITIVE_CURVATURE),
        ),
    )
    return PCGState(
        solution,
        residual,
        preconditioned,
        preconditioned,
        product,
        norm,
        jnp.asarray(0, jnp.int64),
        status.astype(jnp.int32),
    )


def advance_pcg(
    state: PCGState,
    action: jax.Array,
    inverse_action: Callable[[jax.Array], jax.Array],
    threshold: jax.Array,
) -> PCGState:
    """One recurrence with finite and positive-curvature gates."""
    curvature = jnp.vdot(state.direction, action)
    finite = jnp.all(jnp.isfinite(action)) & jnp.isfinite(curvature)

    def step(_):
        alpha = state.residual_product / curvature
        solution = state.solution + alpha * state.direction
        residual = state.residual - alpha * action
        preconditioned = inverse_action(residual)
        product = jnp.vdot(residual, preconditioned)
        norm = jnp.linalg.norm(residual)
        direction = (
            preconditioned + (product / state.residual_product) * state.direction
        )
        all_finite = (
            jnp.all(jnp.isfinite(solution))
            & jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(preconditioned))
            & jnp.all(jnp.isfinite(direction))
            & jnp.isfinite(product)
            & jnp.isfinite(norm)
        )
        status = jnp.where(
            ~all_finite,
            NONFINITE,
            jnp.where(
                norm <= threshold,
                VERIFY_RESIDUAL,
                jnp.where(product > 0, READY, NONPOSITIVE_CURVATURE),
            ),
        )
        return PCGState(
            solution,
            residual,
            preconditioned,
            direction,
            product,
            norm,
            state.iteration + 1,
            status.astype(jnp.int32),
        )

    def breakdown(_):
        return state._replace(
            status=jnp.where(finite, NONPOSITIVE_CURVATURE, NONFINITE).astype(jnp.int32)
        )

    return jax.lax.cond(finite & (curvature > 0), step, breakdown, operand=None)


def verify_pcg(
    state: PCGState,
    true_residual: jax.Array,
    inverse_action: Callable[[jax.Array], jax.Array],
    threshold: jax.Array,
) -> PCGState:
    """Accept a true residual or restart CG without resetting its iteration."""
    verified = initialize_pcg(
        state.solution, true_residual, inverse_action(true_residual), threshold
    )
    status = jnp.where(verified.status == VERIFY_RESIDUAL, CONVERGED, verified.status)
    return verified._replace(iteration=state.iteration, status=status)


def solve_resident_pcg(
    matvec: Callable[[jax.Array], jax.Array],
    inverse_action: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    initial: jax.Array,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iterations: int = 1000,
) -> PCGResult:
    """JIT-compatible frozen SPD solve, including counted true-residual checks.

    The caller owns positive-system proof and memory preflight. A curvature
    check during CG alone does not establish that the entire system is SPD.
    No differentiation rule is asserted for an unconverged iteration trace.
    """
    if rhs.ndim != 1 or initial.shape != rhs.shape:
        raise ValueError("PCG vectors must have matching one-dimensional shapes.")
    rhs_norm = jnp.linalg.norm(rhs)
    threshold = jnp.maximum(atol, rtol * rhs_norm)
    residual = rhs - matvec(initial)
    state = initialize_pcg(initial, residual, inverse_action(residual), threshold)

    def condition(carry):
        current, _ = carry
        return ((current.status == READY) | (current.status == VERIFY_RESIDUAL)) & (
            current.iteration < max_iterations
        )

    def body(carry):
        current, count = carry

        def verify(_):
            actual = rhs - matvec(current.solution)
            return verify_pcg(current, actual, inverse_action, threshold), count + 1

        def advance(_):
            action = matvec(current.direction)
            return advance_pcg(current, action, inverse_action, threshold), count + 1

        return jax.lax.cond(current.status == VERIFY_RESIDUAL, verify, advance, None)

    state, count = jax.lax.while_loop(
        condition, body, (state, jnp.asarray(1, jnp.int64))
    )

    def final_verification(carry):
        current, count = carry
        actual = rhs - matvec(current.solution)

        def incomplete(_):
            verified = verify_pcg(current, actual, inverse_action, threshold)
            return verified._replace(
                status=jnp.where(
                    verified.status == READY, MAX_ITERATIONS, verified.status
                )
            )

        def failed(_):
            norm = jnp.linalg.norm(actual)
            return current._replace(
                residual_norm=norm,
                status=jnp.where(jnp.isfinite(norm), current.status, NONFINITE),
            )

        current = jax.lax.cond(
            (current.status == READY) | (current.status == VERIFY_RESIDUAL),
            incomplete,
            failed,
            None,
        )
        return current, count + 1

    state, count = jax.lax.cond(
        state.status != CONVERGED,
        final_verification,
        lambda carry: carry,
        (state, count),
    )
    relative = state.residual_norm / jnp.where(rhs_norm > 0, rhs_norm, 1.0)
    return PCGResult(
        state.solution,
        state.iteration,
        state.status,
        state.residual_norm,
        relative,
        count,
    )


def solve_resident_implicit(
    matvec: Callable[[jax.Array], jax.Array],
    inverse_action: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iterations: int = 1000,
) -> PCGResult:
    """Implicit derivatives of a converged resident SPD linear system.

    Host RowSource I/O is not traced. Unconverged forward or adjoint solves
    return nonfinite values rather than differentiating an iteration history.
    The caller still owns SPD proof and explicit executable memory preflight.
    """

    def solve(action, values):
        result = solve_resident_pcg(
            action,
            inverse_action,
            values,
            jnp.zeros_like(values),
            rtol=rtol,
            atol=atol,
            max_iterations=max_iterations,
        )
        solution = jnp.where(result.status == CONVERGED, result.solution, jnp.nan)
        auxiliary = (
            result.iteration,
            result.status,
            result.absolute_residual,
            result.relative_residual,
            result.matvec_count,
        )
        return solution, auxiliary

    solution, auxiliary = jax.lax.custom_linear_solve(
        matvec, rhs, solve, symmetric=True, has_aux=True
    )
    iteration, status, absolute, relative, count = auxiliary
    # custom_linear_solve can have an easy adjoint even when its primal failed.
    # A nonfinite external multiplier makes that failed primal invalidate all
    # cotangents too; a zero branch would silently return finite derivatives.
    solution = solution * jnp.where(status == CONVERGED, 1.0, jnp.nan)
    return PCGResult(solution, iteration, status, absolute, relative, count)
