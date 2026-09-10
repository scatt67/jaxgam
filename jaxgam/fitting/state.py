"""Observation-independent state produced by streamed fitting.

The streamed execution path deliberately keeps only coefficient-space arrays.
Row-aligned fitted values, working weights, and responses remain owned by a
``RowSource`` and are recomputed in a later scan when they are requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from jaxgam.fitting.qr import (
    qr_hessian_inverse,
    qr_root_inverse,
    qr_root_transpose_inverse,
)

if TYPE_CHECKING:
    from jaxgam.fitting.pirls import PIRLSResult


class ConsumedFitResult(Protocol):
    """Optimizer-neutral fields consumed by result materialization."""

    smoothing_params: jax.Array
    converged: bool
    n_iter: int
    score: jax.Array
    edf: jax.Array
    scale: jax.Array
    pirls_result: PIRLSResult
    convergence_info: str
    theta: float | None


@dataclass(frozen=True)
class EFSOptimizerDiagnostics:
    """Compact immutable trace for the host EFS outer controller.

    The histories contain accepted scalar states only.  Trial fit arrays and
    optimizer-shaped gradients are deliberately absent from this Phase 3
    contract.
    """

    reference_profile: str
    trace_method: str
    step_policy: str
    stop_reason: str
    outer_iterations: int
    inner_iterations: int
    theta_iterations: int
    accepted_score_history: tuple[float, ...]
    accepted_score_phi_history: tuple[float, ...]
    multiplier: float
    final_update_residual: tuple[float, ...]
    max_proposed_movement: float
    max_accepted_movement: float
    numerator_clamp_count: int
    ratio_replacement_count: int
    log_lambda_cap_count: int
    invalid_fit_seen: bool
    stabilized_solve_seen: bool


def _coefficient_rhs(rhs: jax.Array, n_coef: int) -> jax.Array:
    """Validate the shared coefficient-first ``(p, ...)`` action contract."""
    value = jnp.asarray(rhs)
    if value.ndim not in (1, 2) or value.shape[0] != n_coef:
        raise ValueError(
            f"coefficient RHS must have shape ({n_coef},) or ({n_coef}, n_rhs)"
        )
    return value


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CholeskyCoefficientFactor:
    """Positive factor with ``H = B.T @ B`` and ``B = lower.T``.

    All actions use coefficient-first vectors or matrices.  ``root_inverse``
    is ``B^-1 = lower^-T`` and ``root_transpose_inverse`` is
    ``B^-T = lower^-1``; keeping that ordering explicit prevents treating a
    lower Cholesky factor as the QR root convention.
    """

    lower: jax.Array
    n_coef: int

    def tree_flatten(self):
        return (self.lower,), self.n_coef

    @classmethod
    def tree_unflatten(
        cls, n_coef: int, children: tuple[jax.Array]
    ) -> CholeskyCoefficientFactor:
        return cls(children[0], n_coef)

    def root_inverse(self, rhs_rows: jax.Array) -> jax.Array:
        """Apply ``B^-1 = lower^-T`` to a ``(p, ...)`` row-coordinate RHS."""
        return jsla.solve_triangular(
            self.lower, _coefficient_rhs(rhs_rows, self.n_coef), lower=True, trans="T"
        )

    def root_transpose_inverse(self, rhs_original: jax.Array) -> jax.Array:
        """Apply ``B^-T = lower^-1`` to a ``(p, ...)`` coefficient RHS."""
        return jsla.solve_triangular(
            self.lower, _coefficient_rhs(rhs_original, self.n_coef), lower=True
        )

    def hessian_inverse(self, rhs_original: jax.Array) -> jax.Array:
        """Apply ``H^-1 = B^-1 B^-T`` to a ``(p, ...)`` coefficient RHS."""
        return self.root_inverse(self.root_transpose_inverse(rhs_original))

    def logdet_hessian(self) -> jax.Array:
        """Return ``log|H| = 2 sum(log(diag(lower)))``."""
        return 2.0 * jnp.sum(jnp.log(jnp.diag(self.lower)))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PivotedQRCoefficientFactor:
    """Rank-aware QR factor with reduced ``B = R @ P.T``.

    ``keep`` maps retained fitting coordinates into ``original_n_coef``.
    ``root_inverse`` consumes rank-sized QR-row RHS and returns original
    fitting coordinates; ``root_transpose_inverse`` consumes original fitting
    coordinates and returns rank-sized QR-row RHS.  ``hessian_inverse``
    projects and reconstructs along axis zero for either ``(p,)`` or
    ``(p, n_rhs)`` inputs.
    """

    R: jax.Array
    pivots: jax.Array
    keep: jax.Array
    original_n_coef: int

    def tree_flatten(self):
        return (self.R, self.pivots, self.keep), self.original_n_coef

    @classmethod
    def tree_unflatten(
        cls, original_n_coef: int, children: tuple[jax.Array, jax.Array, jax.Array]
    ) -> PivotedQRCoefficientFactor:
        return cls(*children, original_n_coef)

    @property
    def n_coef(self) -> int:
        return self.original_n_coef

    @property
    def rank(self) -> int:
        return self.R.shape[0]

    def project(self, rhs_original: jax.Array) -> jax.Array:
        """Select retained entries from an original-coordinate ``(p, ...)`` RHS."""
        return _coefficient_rhs(rhs_original, self.n_coef)[self.keep, ...]

    def reconstruct(self, rhs_reduced: jax.Array) -> jax.Array:
        """Embed a rank-sized ``(rank, ...)`` array in fitting coordinates."""
        value = jnp.asarray(rhs_reduced)
        if value.ndim not in (1, 2) or value.shape[0] != self.rank:
            raise ValueError(
                f"reduced QR RHS must have shape ({self.rank},) or ({self.rank}, n_rhs)"
            )
        result = jnp.zeros((self.n_coef, *value.shape[1:]), dtype=value.dtype)
        return result.at[self.keep, ...].set(value)

    def root_inverse(self, rhs_rows: jax.Array) -> jax.Array:
        """Apply embedded ``B^-1 = P R^-1`` to rank-sized QR-row RHS."""
        value = jnp.asarray(rhs_rows)
        if value.ndim not in (1, 2) or value.shape[0] != self.rank:
            raise ValueError("QR row RHS does not match retained factor rank")
        return self.reconstruct(qr_root_inverse(self.R, value, self.pivots))

    def root_transpose_inverse(self, rhs_original: jax.Array) -> jax.Array:
        """Apply ``B^-T = R^-T P.T`` to original-coordinate RHS."""
        return qr_root_transpose_inverse(
            self.R, self.project(rhs_original), self.pivots
        )

    def hessian_inverse(self, rhs_original: jax.Array) -> jax.Array:
        """Apply embedded ``H^-1`` to original-coordinate coefficient RHS."""
        return self.reconstruct(
            qr_hessian_inverse(self.R, self.project(rhs_original), self.pivots)
        )

    def logdet_hessian(self) -> jax.Array:
        """Return the retained-subspace ``log|H|`` without pivot dependence."""
        return 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(self.R))))


CoefficientFactor: TypeAlias = CholeskyCoefficientFactor | PivotedQRCoefficientFactor


@dataclass(frozen=True)
class StreamFitState:
    """Final coefficient-space state for a fixed-smoothing streamed fit."""

    coefficients: jax.Array
    log_lambda: jax.Array
    deviance: jax.Array
    penalized_deviance: jax.Array
    scale: jax.Array
    score_scale: jax.Array
    saturated_loglik: jax.Array
    edf: jax.Array
    xtwx: jax.Array
    xtwx_fisher: jax.Array
    coefficient_factor: CoefficientFactor
    fisher_coefficient_factor: CoefficientFactor
    n_iter: int
    converged: bool
    line_search_failed: bool
    backtracks: int
    stationarity: float
    source_scans: int
    batches_scanned: int

    @property
    def factor(self) -> jax.Array:
        """Legacy lower-factor accessor; QR callers must use tagged actions."""
        if isinstance(self.coefficient_factor, CholeskyCoefficientFactor):
            return self.coefficient_factor.lower
        raise RuntimeError("Pivoted QR factor is not a lower Cholesky factor.")

    @property
    def fisher_factor(self) -> jax.Array:
        """Legacy Fisher lower-factor accessor; reject QR rather than mislabel R."""
        if isinstance(self.fisher_coefficient_factor, CholeskyCoefficientFactor):
            return self.fisher_coefficient_factor.lower
        raise RuntimeError("Pivoted QR Fisher factor requires hessian_inverse actions.")
