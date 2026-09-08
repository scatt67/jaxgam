"""Pure JIT algebra used by the dense extended Fellner--Schall path.

This module deliberately stops short of an outer optimizer.  It supplies the
three matched-state EFS quantities: the log-smoothing determinant derivative
``d``, Fisher inverse trace ``t``, and fitting-coordinate quadratic ``q``.
Penalty roots are made once on the host and retained locally; a trace solve
only scatters one root into a ``p x r`` right-hand side at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np

from jaxgam.fitting import penalty_ops

if TYPE_CHECKING:
    from jaxgam.fitting.data import FittingData


_ROOT_EPS = np.finfo(float).eps ** (2.0 / 3.0)
_RHS_COLUMN_BUDGET = 32


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EFSRoot:
    """A local root ``B`` satisfying ``S_j = B B.T`` on its fixed support."""

    start: int
    stop: int
    sp_index: int
    kind: str
    values: jax.Array
    indices: tuple[int, ...]

    def tree_flatten(self):
        return (self.values,), (
            self.start,
            self.stop,
            self.sp_index,
            self.kind,
            self.indices,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        start, stop, sp_index, kind, indices = aux
        return cls(start, stop, sp_index, kind, children[0], indices)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EFSStatisticsPlan:
    """Static placement plus dynamic leaves for the EFS statistics kernel."""

    n_coef: int
    n_penalties: int
    roots: tuple[EFSRoot, ...]
    singleton_sp_indices: tuple[int, ...]
    singleton_ranks: tuple[int, ...]
    multi_block_sp_indices: tuple[tuple[int, ...], ...]
    multi_block_ranks: tuple[int, ...]
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...]

    def tree_flatten(self):
        return (
            *self.roots,
            *[x for group in self.multi_block_proj_S for x in group],
        ), (
            self.n_coef,
            self.n_penalties,
            len(self.roots),
            self.singleton_sp_indices,
            self.singleton_ranks,
            self.multi_block_sp_indices,
            self.multi_block_ranks,
            tuple(len(group) for group in self.multi_block_proj_S),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            n_coef,
            n_penalties,
            n_roots,
            singleton_sp_indices,
            singleton_ranks,
            multi_indices,
            multi_ranks,
            group_lengths,
        ) = aux
        roots = tuple(children[:n_roots])
        values = iter(children[n_roots:])
        projected = tuple(
            tuple(next(values) for _ in range(length)) for length in group_lengths
        )
        return cls(
            n_coef,
            n_penalties,
            roots,
            singleton_sp_indices,
            singleton_ranks,
            multi_indices,
            multi_ranks,
            projected,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EFSStatistics:
    """EFS matched-state statistics; invalid determinant blocks return NaN d."""

    determinant_derivative: jax.Array
    fisher_trace: jax.Array
    quadratic: jax.Array
    determinant_valid: jax.Array
    input_valid: jax.Array

    def tree_flatten(self):
        return (
            self.determinant_derivative,
            self.fisher_trace,
            self.quadratic,
            self.determinant_valid,
            self.input_valid,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EFSRawUpdate:
    """Raw mgcv ratio arithmetic, kept separate from production-domain policy."""

    numerator: jax.Array
    ratio: jax.Array
    log_smoothing_trial: jax.Array
    finite_positive: jax.Array

    def tree_flatten(self):
        return (
            self.numerator,
            self.ratio,
            self.log_smoothing_trial,
            self.finite_positive,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


def _root(matrix: np.ndarray) -> np.ndarray:
    """Build a CPU root with the same fixed numerical range convention."""
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    symmetric = 0.5 * (matrix + matrix.T)
    if not np.all(np.isfinite(symmetric)):
        raise ValueError("EFS dense penalties must be finite")
    eigvals, vectors = np.linalg.eigh(symmetric)
    threshold = np.max(np.abs(eigvals)) * _ROOT_EPS
    if np.any(eigvals < -threshold):
        raise ValueError(
            "EFS penalties must be positive semidefinite on their fixed range"
        )
    keep = eigvals > threshold
    return vectors[:, keep] * np.sqrt(eigvals[keep])[None, :]


def prepare_efs_statistics(fitting_data: FittingData) -> EFSStatisticsPlan:
    """Prepare local penalty roots once at the CPU-to-JAX boundary.

    The returned plan is a pytree: roots and projected matrices remain dynamic
    device leaves, while intervals and parameter positions stay compilation
    metadata.  It never forms a per-penalty global matrix.
    """
    roots: list[EFSRoot] = []
    for block in fitting_data.penalty_structure.blocks:
        for sp, penalty in zip(block.sp_indices, block.penalties, strict=True):
            if penalty.kind == "identity":
                scale = float(np.asarray(penalty.values))
                if not np.isfinite(scale) or scale < 0:
                    raise ValueError(
                        "EFS identity penalties must be finite and nonnegative"
                    )
                roots.append(
                    EFSRoot(
                        block.start,
                        block.stop,
                        sp,
                        "identity",
                        jnp.asarray(np.sqrt(scale)),
                        tuple(range(block.stop - block.start)),
                    )
                )
            elif penalty.kind == "diagonal":
                values = np.asarray(penalty.values)
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(
                        "EFS diagonal penalties must be finite and nonnegative"
                    )
                active = tuple(np.flatnonzero(values > 0).tolist())
                roots.append(
                    EFSRoot(
                        block.start,
                        block.stop,
                        sp,
                        "diagonal",
                        jnp.asarray(np.sqrt(values[list(active)])),
                        active,
                    )
                )
            else:
                roots.append(
                    EFSRoot(
                        block.start,
                        block.stop,
                        sp,
                        "dense",
                        jnp.asarray(_root(np.asarray(penalty.values))),
                        (),
                    )
                )
    return EFSStatisticsPlan(
        fitting_data.n_coef,
        fitting_data.n_penalties,
        tuple(roots),
        fitting_data.singleton_sp_indices,
        fitting_data.singleton_ranks,
        fitting_data.multi_block_sp_indices,
        fitting_data.multi_block_ranks,
        fitting_data.multi_block_proj_S,
    )


def _determinant_derivative(
    plan: EFSStatisticsPlan, rho: jax.Array
) -> tuple[jax.Array, jax.Array]:
    d = jax.grad(penalty_ops.log_pdet)(
        rho,
        plan.singleton_sp_indices,
        plan.singleton_ranks,
        jnp.zeros((len(plan.singleton_sp_indices),), dtype=rho.dtype),
        plan.multi_block_sp_indices,
        plan.multi_block_ranks,
        plan.multi_block_proj_S,
    )
    valid = jnp.all(jnp.isfinite(rho))
    invalid = ~jnp.isfinite(rho)
    for sp_indices, rank, projected in zip(
        plan.multi_block_sp_indices,
        plan.multi_block_ranks,
        plan.multi_block_proj_S,
        strict=True,
    ):
        if rank == 0:
            continue
        rho_block = jnp.stack([rho[sp] for sp in sp_indices])
        shift = jnp.max(rho_block)
        matrix = jnp.zeros_like(projected[0])
        for value, penalty in zip(rho_block, projected, strict=True):
            matrix = matrix + jnp.exp(value - shift) * penalty
        eigenvalues = jnp.linalg.eigvalsh(matrix)
        block_valid = jnp.all(jnp.isfinite(eigenvalues)) & jnp.all(eigenvalues > 0)
        for sp in sp_indices:
            invalid = invalid.at[sp].set(~block_valid)
        valid = valid & block_valid
    return jnp.where(invalid, jnp.nan, d), valid


def efs_statistics(
    plan: EFSStatisticsPlan, beta: jax.Array, L_fisher: jax.Array, rho: jax.Array
) -> EFSStatistics:
    """Compute d/t/q without an inverse or an m-by-p-by-p temporary."""
    d, valid = _determinant_derivative(plan, rho)
    t = jnp.zeros((plan.n_penalties,), dtype=beta.dtype)
    q = jnp.zeros((plan.n_penalties,), dtype=beta.dtype)
    for root in plan.roots:
        local_beta = beta[root.start : root.stop]
        if root.kind == "dense":
            projected_beta = root.values.T @ local_beta
            n_columns = root.values.shape[1]
        else:
            selected = jnp.asarray(root.indices, dtype=jnp.int32)
            projected_beta = root.values * local_beta[selected]
            n_columns = len(root.indices)
        q = q.at[root.sp_index].add(jnp.vdot(projected_beta, projected_beta).real)
        for start in range(0, n_columns, _RHS_COLUMN_BUDGET):
            stop = min(start + _RHS_COLUMN_BUDGET, n_columns)
            rhs = jnp.zeros((plan.n_coef, stop - start), dtype=beta.dtype)
            if root.kind == "dense":
                rhs = rhs.at[root.start : root.stop, :].set(root.values[:, start:stop])
            else:
                rows = root.start + jnp.asarray(
                    root.indices[start:stop], dtype=jnp.int32
                )
                values = (
                    root.values if root.kind == "identity" else root.values[start:stop]
                )
                rhs = rhs.at[rows, jnp.arange(stop - start)].set(values)
            solved = jsla.solve_triangular(L_fisher, rhs, lower=True)
            t = t.at[root.sp_index].add(jnp.vdot(solved, solved).real)
    input_valid = (
        jnp.all(jnp.isfinite(beta))
        & jnp.all(jnp.isfinite(L_fisher))
        & jnp.all(jnp.diag(L_fisher) > 0)
        & jnp.all(jnp.isfinite(d))
        & jnp.all(jnp.isfinite(t))
        & jnp.all(jnp.isfinite(q))
    )
    return EFSStatistics(d, t, q, valid, input_valid)


def efs_raw_update(
    rho: jax.Array,
    statistics: EFSStatistics,
    phi_update: jax.Array,
    multiplier: jax.Array,
    log_lambda_max: jax.Array,
) -> EFSRawUpdate:
    """Reproduce ``efsudr`` ratio branch ordering, including its boundaries."""
    numerator = jnp.maximum(
        0.0, jnp.exp(-rho) * statistics.determinant_derivative - statistics.fisher_trace
    )
    ratio = phi_update * numerator / jnp.maximum(0.0, statistics.quadratic)
    ratio = jnp.where((numerator == 0) & (statistics.quadratic == 0), 1.0, ratio)
    ratio = jnp.where(jnp.isfinite(ratio), ratio, 1e6)
    trial = jnp.minimum(rho + jnp.log(ratio) * multiplier, log_lambda_max)
    return EFSRawUpdate(
        numerator,
        ratio,
        trial,
        statistics.determinant_valid
        & statistics.input_valid
        & jnp.all(jnp.isfinite(rho))
        & jnp.isfinite(phi_update)
        & (phi_update > 0)
        & jnp.isfinite(multiplier)
        & (multiplier > 0)
        & jnp.isfinite(log_lambda_max)
        & jnp.all(jnp.isfinite(ratio) & (ratio > 0))
        & jnp.all(jnp.isfinite(trial)),
    )
