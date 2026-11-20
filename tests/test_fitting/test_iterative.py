"""PCG recurrence, actual-residual and failure-status gates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.fitting.operator.iterative import (
    CONVERGED,
    MAX_ITERATIONS,
    NONFINITE,
    NONPOSITIVE_CURVATURE,
    READY,
    VERIFY_RESIDUAL,
    advance_pcg,
    initialize_pcg,
    solve_resident_implicit,
    solve_resident_pcg,
    verify_pcg,
)
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


def test_resident_pcg_matches_direct_solve_and_counts_actual_residual_verification():
    rng = np.random.default_rng(23951)
    Q, _ = np.linalg.qr(rng.normal(size=(24, 24)))
    H = Q @ np.diag(np.geomspace(0.3, 300.0, 24)) @ Q.T
    rhs = rng.normal(size=24)
    matrix = jnp.asarray(H)
    compiled = jax.jit(
        lambda b, x: solve_resident_pcg(
            lambda v: matrix @ v,
            lambda v: v / jnp.diag(matrix),
            b,
            x,
            rtol=1e-12,
            max_iterations=120,
        )
    )
    result = compiled(jnp.asarray(rhs), jnp.zeros(24))
    collector = _AssertCollector()
    collector.check(
        "SPD solution",
        lambda: np.testing.assert_allclose(
            result.solution, np.linalg.solve(H, rhs), rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "verified relative residual",
        lambda: np.testing.assert_allclose(
            result.relative_residual,
            np.linalg.norm(rhs - H @ result.solution) / np.linalg.norm(rhs),
            rtol=1e-3,
            atol=1e-14,
        ),
    )
    assert result.status == CONVERGED
    assert result.relative_residual <= 1e-12
    assert result.matvec_count >= result.iteration + 2
    collector.raise_if_any("resident PCG")


@pytest.mark.parametrize(
    ("rhs", "initial"),
    [
        (np.zeros(8), np.zeros(8)),
        (np.arange(8, dtype=float), np.arange(8, dtype=float)),
    ],
)
def test_already_solved_system_requires_true_verification(rhs, initial):
    result = jax.jit(lambda b, x: solve_resident_pcg(lambda v: v, lambda v: v, b, x))(
        jnp.asarray(rhs), jnp.asarray(initial)
    )
    assert result.status == CONVERGED
    assert result.iteration == 0
    assert result.matvec_count == 2
    assert result.absolute_residual == 0


@pytest.mark.parametrize(
    ("system", "inverse", "expected"),
    [
        (lambda v: -v, lambda v: v, NONPOSITIVE_CURVATURE),
        (lambda v: v, lambda v: -v, NONPOSITIVE_CURVATURE),
        (lambda v: v * jnp.nan, lambda v: v, NONFINITE),
        (lambda v: v, lambda v: v * jnp.inf, NONFINITE),
    ],
)
def test_breakdowns_are_finite_statuses_not_convergence(system, inverse, expected):
    result = jax.jit(
        lambda b: solve_resident_pcg(system, inverse, b, jnp.zeros_like(b))
    )(jnp.ones(4))
    assert result.status == expected


def test_iteration_limit_preserves_actual_residual_and_truthful_nonconvergence():
    diagonal = jnp.arange(1, 13, dtype=jnp.float64)
    rhs = jnp.ones(12)
    result = jax.jit(
        lambda b: solve_resident_pcg(
            lambda v: diagonal * v,
            lambda v: v,
            b,
            jnp.zeros_like(b),
            rtol=1e-14,
            max_iterations=1,
        )
    )(rhs)
    assert result.status == MAX_ITERATIONS
    assert result.iteration == 1
    assert result.matvec_count == 3
    np.testing.assert_allclose(
        result.absolute_residual, np.linalg.norm(rhs - diagonal * result.solution)
    )


def test_recursive_residual_cannot_claim_convergence_and_false_candidate_restarts():
    rhs = jnp.ones(5)
    initial = jax.jit(initialize_pcg)(jnp.zeros(5), rhs, rhs, jnp.asarray(1e-8))
    assert initial.status == READY
    candidate = jax.jit(
        lambda s: advance_pcg(s, s.direction, lambda v: v, jnp.asarray(1e-8))
    )(initial)
    assert candidate.status == VERIFY_RESIDUAL
    restarted = jax.jit(lambda s: verify_pcg(s, rhs, lambda v: v, jnp.asarray(1e-8)))(
        candidate
    )
    assert restarted.status == READY
    assert restarted.iteration == 1
    np.testing.assert_array_equal(restarted.direction, rhs)
    failed = jax.jit(
        lambda s: advance_pcg(s, s.direction * jnp.nan, lambda v: v, jnp.asarray(1e-8))
    )(initial)
    assert failed.status == NONFINITE
    invalid = jax.jit(
        lambda s: advance_pcg(s, s.direction, lambda v: v * jnp.nan, jnp.asarray(1e-8))
    )(initial)
    assert invalid.status == NONFINITE


def test_resident_pcg_shape_error_and_compiled_storage_grows_linearly():
    with pytest.raises(ValueError, match="matching"):
        solve_resident_pcg(lambda v: v, lambda v: v, jnp.ones(3), jnp.ones(2))
    measurements = []
    for p in (1024, 4096):
        compiled = (
            jax.jit(
                lambda b: solve_resident_pcg(
                    lambda v: 2 * v, lambda v: v / 2, b, jnp.zeros_like(b)
                )
            )
            .lower(jnp.ones(p))
            .compile()
        )
        sizes = compiled.memory_analysis()
        measurements.append(
            sizes.argument_size_in_bytes
            + sizes.output_size_in_bytes
            + sizes.temp_size_in_bytes
        )
        result = compiled(jnp.ones(p))
        assert result.status == CONVERGED
        assert result.iteration == 1
    assert measurements[1] <= 4.5 * measurements[0]
    assert measurements[1] < 128 * 4096


def test_resident_implicit_gradient_matches_converged_system_not_iteration_trace():
    diagonal = jnp.geomspace(0.7, 9.0, 19)
    rhs = jnp.sin(jnp.arange(19, dtype=jnp.float64))

    def solution(log_shift, values):
        total = diagonal + jnp.exp(log_shift)
        return solve_resident_implicit(
            lambda v: total * v, lambda v: v / total, values, rtol=1e-13
        ).solution

    shift = jnp.asarray(-0.4)
    actual = jax.jit(
        jax.grad(lambda rho: jnp.vdot(solution(rho, rhs), solution(rho, rhs)))
    )(shift)
    total = diagonal + jnp.exp(shift)
    expected = jnp.sum(-2 * rhs**2 * jnp.exp(shift) / total**3)
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    gradient = jax.jit(jax.grad(lambda values: jnp.sum(solution(shift, values))))(rhs)
    np.testing.assert_allclose(gradient, 1 / total, rtol=STRICT.rtol, atol=STRICT.atol)
    failed = jax.jit(
        lambda values: solve_resident_implicit(
            lambda v: diagonal * v, lambda v: v, values, rtol=1e-14, max_iterations=1
        )
    )(rhs)
    assert failed.status == MAX_ITERATIONS
    assert np.all(np.isnan(failed.solution))


@pytest.mark.parametrize("forward_easy", [False, True])
def test_implicit_derivatives_require_both_forward_and_adjoint_success(forward_easy):
    diagonal = jnp.array([1.0, 2.0, 4.0])
    rhs = jnp.array([1.0, 0.0, 0.0]) if forward_easy else jnp.ones(3)

    def solve(values):
        return solve_resident_implicit(
            lambda v: diagonal * v,
            lambda v: v,
            values,
            rtol=1e-14,
            max_iterations=1,
        )

    result = jax.jit(solve)(rhs)
    assert result.status == (CONVERGED if forward_easy else MAX_ITERATIONS)
    # A coordinate cotangent is easy while the all-ones primal is not;
    # reversing these choices makes the primal easy and the adjoint fail.
    objective = (
        (lambda values: jnp.sum(solve(values).solution))
        if forward_easy
        else (lambda values: solve(values).solution[0])
    )
    gradient = jax.jit(jax.grad(objective))(rhs)
    assert np.all(np.isnan(gradient))
