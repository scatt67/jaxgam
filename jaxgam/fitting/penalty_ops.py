"""JAX algebra for local penalty blocks.

Only ``add_to_dense`` creates a p-by-p array, and it creates the *combined*
penalty requested by a dense direct-solver consumer.  Individual smoothing
penalties never acquire global zero padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxLocalPenalty:
    """One local penalty; ``kind`` is static and ``values`` is a device leaf."""

    kind: str  # dense | diagonal | identity
    values: jax.Array
    size: int

    def tree_flatten(self):
        return (self.values,), (self.kind, self.size)

    @classmethod
    def tree_unflatten(cls, aux, children):
        kind, size = aux
        return cls(kind, children[0], size)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxTransform:
    """One local coefficient transform; its representation is static."""

    kind: str  # dense | diagonal | identity
    values: jax.Array
    size: int

    def tree_flatten(self):
        return (self.values,), (self.kind, self.size)

    @classmethod
    def tree_unflatten(cls, aux, children):
        kind, size = aux
        return cls(kind, children[0], size)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxPenaltyBlock:
    """Local penalties sharing one coefficient interval and transform."""

    start: int
    stop: int
    sp_indices: tuple[int, ...]
    penalties: tuple[JaxLocalPenalty, ...]
    transform: JaxTransform

    def tree_flatten(self):
        return (*self.penalties, self.transform), (
            self.start,
            self.stop,
            self.sp_indices,
            len(self.penalties),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        start, stop, sp_indices, n_penalties = aux
        return cls(start, stop, sp_indices, tuple(children[:n_penalties]), children[-1])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxPenaltyStructure:
    """Pytree of local penalties and static coefficient placement metadata."""

    n_coef: int
    blocks: tuple[JaxPenaltyBlock, ...]

    def tree_flatten(self):
        return self.blocks, self.n_coef

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux, tuple(children))

    @property
    def n_penalties(self) -> int:
        return 1 + max(
            (sp for block in self.blocks for sp in block.sp_indices), default=-1
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxPenaltyRoots:
    """Unscaled local penalty roots ``E_j`` (``E_j' E_j = S_j``), one per sp.

    Placement metadata is static; the roots are device leaves.  These are the
    fitting-coordinate roots of the structural penalties, so their null
    columns are exactly zero.
    """

    n_coef: int
    starts: tuple[int, ...]
    stops: tuple[int, ...]
    sp_indices: tuple[int, ...]
    roots: tuple[jax.Array, ...]

    def tree_flatten(self):
        return self.roots, (self.n_coef, self.starts, self.stops, self.sp_indices)

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_coef, starts, stops, sp_indices = aux
        return cls(n_coef, starts, stops, sp_indices, tuple(children))

    @property
    def n_rows(self) -> int:
        return sum(root.shape[0] for root in self.roots)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxCoefficientRotation:
    """Block-local orthogonal maps ``b = T b_s`` into gam.reparam coordinates.

    ``gam.fit3`` fits in ``T``'s coordinates (gam.fit3.r 161-179) and maps the
    coefficients back (812-813). Placement is static; each ``T`` is a device
    leaf computed on the host at every evaluation of the smoothing parameters.
    """

    starts: tuple[int, ...]
    stops: tuple[int, ...]
    transforms: tuple[jax.Array, ...]

    def tree_flatten(self):
        return self.transforms, (self.starts, self.stops)

    @classmethod
    def tree_unflatten(cls, aux, children):
        starts, stops = aux
        return cls(starts, stops, tuple(children))

    def _blocks(self):
        return zip(self.starts, self.stops, self.transforms, strict=True)

    def design(self, X: jax.Array) -> jax.Array:
        """``X T`` (gam.fit3.r 169)."""
        for start, stop, T in self._blocks():
            X = X.at[:, start:stop].set(X[:, start:stop] @ T)
        return X

    def to_stable(self, beta: jax.Array) -> jax.Array:
        """``T' b`` (gam.fit3.r 164-166)."""
        for start, stop, T in self._blocks():
            beta = beta.at[start:stop].set(T.T @ beta[start:stop])
        return beta

    def from_stable(self, beta: jax.Array) -> jax.Array:
        """``T b_s`` (gam.fit3.r 812)."""
        for start, stop, T in self._blocks():
            beta = beta.at[start:stop].set(T @ beta[start:stop])
        return beta

    def from_stable_matrix(self, M: jax.Array) -> jax.Array:
        """``T M T'`` on both sides, keeping cross-blocks."""
        for start, stop, T in self._blocks():
            M = M.at[start:stop, :].set(T @ M[start:stop, :])
            M = M.at[:, start:stop].set(M[:, start:stop] @ T.T)
        return M

    def to_stable_matrix(self, M: jax.Array) -> jax.Array:
        """``T' M T`` on both sides, keeping cross-blocks."""
        for start, stop, T in self._blocks():
            M = M.at[start:stop, :].set(T.T @ M[start:stop, :])
            M = M.at[:, start:stop].set(M[:, start:stop] @ T)
        return M


def augmented_root(roots: JaxPenaltyRoots, rho: jax.Array) -> jax.Array:
    """Return ``E(rho)`` with ``E' E = S_lambda``, stacking scaled local roots.

    Each penalty keeps its own rows, scaled by ``exp(rho_j / 2)``, so
    penalties of very different size are never summed before factorization
    (the machine-zero leakage that mgcv's ``gam.reparam`` avoids).
    """
    blocks = [
        jnp.zeros((root.shape[0], roots.n_coef), dtype=rho.dtype)
        .at[:, start:stop]
        .set(jnp.exp(0.5 * rho[sp]) * root)
        for start, stop, sp, root in zip(
            roots.starts, roots.stops, roots.sp_indices, roots.roots, strict=True
        )
        if root.shape[0]
    ]
    if not blocks:
        return jnp.zeros((0, roots.n_coef), dtype=rho.dtype)
    return jnp.concatenate(blocks, axis=0)


def _apply_local(penalty: JaxLocalPenalty, vector: jax.Array) -> jax.Array:
    if penalty.kind == "dense":
        return penalty.values @ vector
    if penalty.kind == "diagonal":
        return penalty.values * vector
    return penalty.values * vector


def _dense_local(penalty: JaxLocalPenalty) -> jax.Array:
    if penalty.kind == "dense":
        return penalty.values
    if penalty.kind == "diagonal":
        return jnp.diag(penalty.values)
    return jnp.eye(penalty.size, dtype=penalty.values.dtype) * penalty.values


def _smoothing(rho: jax.Array, smoothing: jax.Array | None, sp: int) -> jax.Array:
    """``exp(rho[sp])``, or a caller-owned exponential of the same state."""
    return jnp.exp(rho[sp]) if smoothing is None else smoothing[sp]


def apply(
    structure: JaxPenaltyStructure,
    beta: jax.Array,
    rho: jax.Array,
    *,
    smoothing: jax.Array | None = None,
) -> jax.Array:
    """Return ``S_lambda beta`` without forming any individual global S_j.

    ``smoothing`` optionally supplies ``exp(rho)`` evaluated elsewhere.
    """
    result = jnp.zeros_like(beta)
    for block in structure.blocks:
        local_beta = beta[block.start : block.stop]
        local_result = jnp.zeros_like(local_beta)
        for sp, penalty in zip(block.sp_indices, block.penalties, strict=True):
            local_result = local_result + _smoothing(rho, smoothing, sp) * _apply_local(
                penalty, local_beta
            )
        result = result.at[block.start : block.stop].set(local_result)
    return result


def quadratic(
    structure: JaxPenaltyStructure, beta: jax.Array, rho: jax.Array
) -> jax.Array:
    """Return ``beta.T @ S_lambda @ beta`` using local actions."""
    return jnp.vdot(beta, apply(structure, beta, rho)).real


def add_to_dense(
    structure: JaxPenaltyStructure, G: jax.Array, rho: jax.Array
) -> jax.Array:
    """Add the combined penalty to a dense coefficient-space matrix.

    This is the explicit dense-direct-solver boundary.  It deliberately does
    not expose a per-penalty global materialization API.
    """
    result = G
    for block in structure.blocks:
        local = jnp.zeros(
            (block.stop - block.start, block.stop - block.start), dtype=G.dtype
        )
        for sp, penalty in zip(block.sp_indices, block.penalties, strict=True):
            local = local + jnp.exp(rho[sp]) * _dense_local(penalty)
        result = result.at[block.start : block.stop, block.start : block.stop].add(
            local
        )
    return result


def materialize(structure: JaxPenaltyStructure, rho: jax.Array) -> jax.Array:
    """Explicit dense materialization of the one combined ``S_lambda``."""
    return add_to_dense(structure, jnp.zeros((structure.n_coef, structure.n_coef)), rho)


def materialize_difference(
    structure: JaxPenaltyStructure, rho_base: jax.Array, rho_trial: jax.Array
) -> jax.Array:
    """Form S_trial - S_base without subtracting nearly equal penalties."""
    multipliers = jnp.exp(rho_base) * jnp.expm1(rho_trial - rho_base)
    result = jnp.zeros((structure.n_coef, structure.n_coef), dtype=rho_base.dtype)
    for block in structure.blocks:
        local = jnp.zeros((block.stop - block.start,) * 2, dtype=rho_base.dtype)
        for sp, penalty in zip(block.sp_indices, block.penalties, strict=True):
            local = local + multipliers[sp] * _dense_local(penalty)
        result = result.at[block.start : block.stop, block.start : block.stop].add(
            local
        )
    return result


def parameter_vjp(
    structure: JaxPenaltyStructure,
    beta: jax.Array,
    adjoint: jax.Array,
    rho: jax.Array,
) -> jax.Array:
    """VJP of ``S_lambda beta`` with respect to log smoothing parameters."""
    result = jnp.zeros_like(rho)
    for block in structure.blocks:
        local_beta = beta[block.start : block.stop]
        local_adjoint = adjoint[block.start : block.stop]
        for sp, penalty in zip(block.sp_indices, block.penalties, strict=True):
            value = (
                jnp.exp(rho[sp])
                * jnp.vdot(local_adjoint, _apply_local(penalty, local_beta)).real
            )
            result = result.at[sp].add(value)
    return result


def log_pdet(
    rho: jax.Array,
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    singleton_eig_constants: jax.Array,
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
) -> jax.Array:
    """Use the repository's single block log-pseudodeterminant engine."""
    # Import lazily: jax_utils is deliberately independent of fitting modules.
    from jaxgam.jax_utils import block_log_det_S

    return block_log_det_S(
        rho,
        singleton_sp_indices,
        singleton_ranks,
        singleton_eig_constants,
        multi_block_sp_indices,
        multi_block_ranks,
        multi_block_proj_S,
    )


def apply_transform(transform: JaxTransform, beta: jax.Array) -> jax.Array:
    """Map fitting coefficients to public coefficients for one block."""
    if transform.kind == "dense":
        return transform.values @ beta
    if transform.kind == "diagonal":
        return transform.values * beta
    return beta


def transform_coefficients(
    structure: JaxPenaltyStructure, beta: jax.Array
) -> jax.Array:
    """Apply all local transforms to a coefficient vector."""
    result = beta
    for block in structure.blocks:
        result = result.at[block.start : block.stop].set(
            apply_transform(block.transform, beta[block.start : block.stop])
        )
    return result


def transform_covariance(
    structure: JaxPenaltyStructure, covariance: jax.Array
) -> jax.Array:
    """Apply block transforms on both covariance sides, preserving cross-blocks."""
    result = covariance
    for block in structure.blocks:
        start, stop = block.start, block.stop
        T = block.transform
        if T.kind == "identity":
            continue
        if T.kind == "diagonal":
            result = result.at[start:stop, :].set(
                T.values[:, None] * result[start:stop, :]
            )
            result = result.at[:, start:stop].set(
                result[:, start:stop] * T.values[None, :]
            )
        else:
            result = result.at[start:stop, :].set(T.values @ result[start:stop, :])
            result = result.at[:, start:stop].set(result[:, start:stop] @ T.values.T)
    return result
