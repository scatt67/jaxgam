"""Dense inner fit through the pinned mgcv coefficient loops.

Standard families run the ``gam.fit3`` port (``efs_regular_pirls``): Fisher
steps for canonical links, full Newton (``alpha``-weighted) steps with a
same-step Fisher retry when the observed system is indefinite for
noncanonical links (gam.fit3.r 317-352), R's sequential non-finite ->
validity -> divergence step halving (gam.fit3.r 372-443) and its convergence
test with the penalized-deviance gradient check (gam.fit3.r 447-467).

``NegativeBinomial`` runs the ``gam.fit4`` port (the in-loop theta solver of
``pirls.py`` with theta held fixed): observed weights ``0.5 d2D/deta2``, the
positive-weight retry for an indefinite system (gam.fit4.r 391-416), the same
recovery order (gam.fit4.r 432-505) and gam.fit4's convergence test.  An
estimated theta enters as dynamic data, so a REML evaluation reuses one
compiled loop for every theta.

A recovery that cannot correct its step is reported by status, never as
convergence (R stops with "inner loop 1/2/3; can't correct step size").
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from jaxgam.families.base import ExponentialFamily
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting import efs_regular_pirls as regular
from jaxgam.fitting import pirls
from jaxgam.fitting.pirls import PIRLSResult

SOURCE_REGULAR = "regular"
SOURCE_NB = "nb"

#: Default convergence epsilon of the source loops: R's tight pinned control
#: (gam.control(epsilon=1e-11)), a documented deviation from R's default
#: 1e-7, whose gradient test stops about 1e-9 short of the stationary point.
#: It is an explicit internal parameter of ``source_pirls`` (``epsilon``);
#: the outer optimizer's PIRLS tolerance is not reinterpreted as it.
SOURCE_EPSILON = 1e-11

_REGULAR_MESSAGES = {
    regular.EFS_REGULAR_STATUS_ITERATION_LIMIT: "iteration limit reached",
    regular.EFS_REGULAR_STATUS_INVALID_START: "no valid starting values",
    regular.EFS_REGULAR_STATUS_NONFINITE_COEFFICIENTS: "non-finite coefficients",
    regular.EFS_REGULAR_STATUS_NONFINITE_RECOVERY_FAILED: (
        "inner loop 1; can't correct step size (non-finite deviance)"
    ),
    regular.EFS_REGULAR_STATUS_DOMAIN_RECOVERY_FAILED: (
        "inner loop 2; can't correct step size (invalid mu or eta)"
    ),
    regular.EFS_REGULAR_STATUS_DIVERGENCE_RECOVERY_FAILED: (
        "inner loop 3; can't correct step size (penalized deviance diverged)"
    ),
    regular.EFS_REGULAR_STATUS_INVALID_WORKING_FACTORS: "invalid working factors",
    regular.EFS_REGULAR_STATUS_RANK_BOUNDARY_UNSUPPORTED: (
        "singular penalized Hessian (pls_fit1 rank boundary)"
    ),
}
_NB_MESSAGES = {
    pirls._EFS_STATUS_BETA_STEP_FAILED: "coefficient step failed",
    pirls._EFS_STATUS_INVALID_WORKING_FACTORS: "invalid working factors",
    pirls._EFS_STATUS_THETA_FAILED: "theta update failed",
    pirls._EFS_STATUS_NONFINITE_STATIONARITY: "non-finite stationarity gradient",
    pirls._EFS_STATUS_ITERATION_LIMIT: "iteration limit reached",
    pirls._EFS_STATUS_INVALID_INPUT: "invalid input",
    pirls._EFS_STATUS_RETAINED_START_INVALID_TRIAL: "no valid starting values",
    pirls._EFS_STATUS_NONFINITE_RECOVERY_FAILED: (
        "inner loop 1; can't correct step size (non-finite deviance)"
    ),
    pirls._EFS_STATUS_DOMAIN_RECOVERY_FAILED: (
        "inner loop 2; can't correct step size (invalid mu or eta)"
    ),
    pirls._EFS_STATUS_DIVERGENCE_RECOVERY_FAILED: (
        "inner loop 3; can't correct step size (penalized deviance diverged)"
    ),
}


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SourcePIRLSOutcome:
    """A ``PIRLSResult`` plus the source loop's termination status."""

    pirls_result: PIRLSResult
    status: jax.Array
    source: str

    def tree_flatten(self):
        return (self.pirls_result, self.status), self.source

    @classmethod
    def tree_unflatten(cls, source, children):
        return cls(children[0], children[1], source)


def uses_source_loop(family: ExponentialFamily) -> bool:
    """Whether the dense inner fit runs a pinned mgcv loop for this family."""
    return isinstance(family, (Gaussian, Binomial, Poisson, Gamma, NegativeBinomial))


def status_message(family: ExponentialFamily, status: int) -> str:
    """Human-readable reason for a non-converged source inner fit."""
    if int(status) == 0:
        return "no convergence within the iteration limit"
    messages = (
        _NB_MESSAGES if isinstance(family, NegativeBinomial) else _REGULAR_MESSAGES
    )
    return messages.get(int(status), f"inner-loop status {int(status)}")


def source_pirls(
    X: jax.Array,
    y: jax.Array,
    beta_start: jax.Array,
    S_lambda: jax.Array,
    family: ExponentialFamily,
    wt: jax.Array,
    offset: jax.Array | None,
    null_beta: jax.Array,
    *,
    scale: jax.Array | float = 1.0,
    log_theta: jax.Array | None = None,
    count_indices: jax.Array | None = None,
    max_y: int = 0,
    integer_counts: bool = True,
    penalty_root: jax.Array | None = None,
    max_iter: int = 200,
    epsilon: float = SOURCE_EPSILON,
) -> SourcePIRLSOutcome:
    """Fit coefficients at fixed smoothing (and NB theta) by the source loop.

    ``beta_start`` is the retained start (``start=`` in R) and ``null_beta``
    R's ``null.coef`` anchor for the first-iteration divergence and validity
    recovery.  ``scale`` is the dispersion in gam.fit3's convergence scale;
    ``epsilon`` is gam.control's ``epsilon`` for both loops' convergence
    tests (default ``SOURCE_EPSILON``).
    """
    if offset is None:
        offset = jnp.zeros_like(y)
    initial_eta = X @ beta_start + offset
    if isinstance(family, NegativeBinomial):
        if log_theta is None:
            log_theta = jnp.asarray(family.get_theta(transformed=False))
        if count_indices is None:
            # No count-prefix plan: the theta-free loop does not use the
            # prefix, and the zero indices are the non-integer layout.
            count_indices = jnp.zeros(y.shape, dtype=jnp.int64)
            integer_counts = False
        result = pirls._efs_theta_pirls_loop_jit(
            X,
            y,
            beta_start,
            S_lambda,
            family,
            wt,
            offset,
            jnp.reshape(jnp.asarray(log_theta, dtype=X.dtype), (1,)),
            null_beta,
            initial_eta,
            jnp.asarray(True),
            count_indices,
            max_y=max_y,
            integer_counts=integer_counts,
            estimate_theta=False,
            max_iter=max_iter,
            tol=epsilon,
            penalty_root=penalty_root,
        )
        return SourcePIRLSOutcome(
            replace(result.pirls_result, inner_status=result.status),
            result.status,
            SOURCE_NB,
        )
    result = regular.efs_regular_pirls_loop(
        X,
        y,
        beta_start,
        initial_eta,
        null_beta,
        X @ null_beta + offset,
        S_lambda,
        family,
        wt,
        offset,
        jnp.asarray(scale, dtype=X.dtype),
        start_present=True,
        max_iter=max_iter,
        tol=epsilon,
        penalty_root=penalty_root,
    )
    return SourcePIRLSOutcome(
        replace(result.pirls_result, inner_status=result.status),
        result.status,
        SOURCE_REGULAR,
    )
