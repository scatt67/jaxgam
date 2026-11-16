"""Phase 1 to Phase 2 fitting data with local penalty storage."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.initialization import initialize_beta_cpu, r_null_coefficients
from jaxgam.formula.fitting_prepare import (
    apply_transforms_to_design,
    initial_log_sp_from_diagonal,
    qr_penalty_roots,
    reparameterize_structure,
    total_penalty_spaces,
)
from jaxgam.jax_utils import to_jax
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    DenseTransform,
    DiagonalPenalty,
    DiagonalTransform,
    IdentityPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)

if TYPE_CHECKING:
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.prepare import PreparedModel


_EPS_TWO_THIRDS = np.finfo(float).eps ** (2.0 / 3.0)
_LOG_FLOOR = 1e-30

# Keep initial-sp reductions bounded without changing its dense result.
_INITIAL_SP_BATCH_ROWS = 8192


@dataclass(frozen=True)
class CountPrefixPlan:
    """Reusable NB count metadata prepared once at the CPU→device boundary.

    ``indices`` is dynamic device data; the capacity and integer-domain flag
    remain static so they key only the small compiled prefix-table shape.
    """

    indices: jax.Array
    max_count: int
    capacity: int
    integer_counts: bool


@dataclass(frozen=True)
class PreparedFittingMetadata:
    """Observation-independent Phase-2 metadata for prepared backends.

    This deliberately contains neither a design matrix nor response, weights,
    or offsets.  It is the compact fitting metadata needed by stream routes to
    form penalty and REML statistics without retaining training rows.
    """

    penalty_structure: penalty_ops.JaxPenaltyStructure
    log_lambda_init: jax.Array
    family: ExponentialFamily
    n_obs: int
    n_coef: int
    total_penalty_rank: int
    singleton_sp_indices: tuple[int, ...]
    singleton_ranks: tuple[int, ...]
    singleton_eig_constants: jax.Array
    multi_block_sp_indices: tuple[tuple[int, ...], ...]
    multi_block_ranks: tuple[int, ...]
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...]
    source_fingerprint: str
    basis_fingerprint: str
    family_name: str
    link_name: str
    rank_deficit: int = 0

    @property
    def n_penalties(self) -> int:
        return self.penalty_structure.n_penalties

    @property
    def total_penalty_null_dim(self) -> int:
        return self.n_coef - self.total_penalty_rank

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedModel,
        family: ExponentialFamily,
        device: jax.Device | None = None,
    ) -> PreparedFittingMetadata:
        """Transfer only frozen prepared fitting metadata to the device."""
        fitting = prepared.fitting
        if fitting is None:
            raise ValueError("Prepared fitting metadata requires fitting preparation.")
        metadata = _build_block_metadata(fitting.penalty_structure, device)
        return cls(
            penalty_structure=_to_jax_structure(fitting.penalty_structure, device),
            log_lambda_init=to_jax(fitting.log_lambda_init, device=device),
            family=family,
            n_obs=prepared.n_obs,
            n_coef=prepared.n_coef,
            total_penalty_rank=fitting.total_penalty_rank,
            singleton_sp_indices=metadata["singleton_sp_indices"],
            singleton_ranks=metadata["singleton_ranks"],
            singleton_eig_constants=metadata["singleton_eig_constants"],
            multi_block_sp_indices=metadata["multi_block_sp_indices"],
            multi_block_ranks=metadata["multi_block_ranks"],
            multi_block_proj_S=metadata["multi_block_proj_S"],
            source_fingerprint=prepared.source_fingerprint,
            basis_fingerprint=prepared.basis_fingerprint,
            family_name=fitting.family_name,
            link_name=fitting.link_name,
            rank_deficit=fitting.unpenalized_rank_deficit,
        )


@dataclass(frozen=True)
class FittingData:
    """Device data and local penalty algebra for dense fitting.

    The direct solver still materializes one combined dense ``S_lambda`` when
    constructing its Hessian. This object never owns m global zero-padded
    penalties or a global fitting transform.
    """

    X: jax.Array
    y: jax.Array
    wt: jax.Array
    offset: jax.Array | None
    penalty_structure: penalty_ops.JaxPenaltyStructure
    log_lambda_init: jax.Array
    family: ExponentialFamily
    n_obs: int
    n_coef: int
    penalty_ranks: tuple[int, ...]
    penalty_null_dims: tuple[int, ...]
    total_penalty_rank: int
    singleton_sp_indices: tuple[int, ...]
    singleton_ranks: tuple[int, ...]
    singleton_eig_constants: jax.Array
    multi_block_sp_indices: tuple[tuple[int, ...], ...]
    multi_block_ranks: tuple[int, ...]
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...]
    multi_block_S_local: tuple[tuple[jax.Array, ...], ...]
    max_y: int
    count_prefix_plan: CountPrefixPlan | None = None
    rank_deficit: int = 0

    # Computed once in transformed fitting coordinates at the CPU-to-JAX
    # boundary. Manual FittingData fixtures retain the compatibility fallback.
    beta_init: jax.Array | None = None

    # Unscaled structural penalty roots when the lambda-free pls_fit1 rank
    # test finds the data/penalty system structurally full rank; the dense
    # inner fit then solves by QR. None keeps the Cholesky solve (manual
    # fixtures, unpenalized models and structurally rank-deficient models).
    penalty_roots: penalty_ops.JaxPenaltyRoots | None = None

    # R's get.null.coef anchor in fitting coordinates, used by the pinned
    # gam.fit3/gam.fit4 inner loops; None for manual fixtures.
    null_beta: jax.Array | None = None

    @property
    def uses_source_loop(self) -> bool:
        """Whether the dense inner fit runs the pinned mgcv loop.

        Structurally rank-deficient systems keep the regularized solve:
        pls_fit1's unidentifiable-column drop is not ported.
        """
        from jaxgam.fitting.source_pirls import uses_source_loop

        return (
            self.null_beta is not None
            and self.rank_deficit == 0
            and uses_source_loop(self.family)
        )

    def penalty_root(self, log_lambda: jax.Array) -> jax.Array | None:
        """``E(rho)`` for the QR inner solve, or ``None`` for Cholesky."""
        if self.penalty_roots is None:
            return None
        return penalty_ops.augmented_root(self.penalty_roots, log_lambda)

    @property
    def n_penalties(self) -> int:
        return self.penalty_structure.n_penalties

    @property
    def total_penalty_null_dim(self) -> int:
        return self.n_coef - self.total_penalty_rank

    def S_lambda(self, log_lambda: jax.Array) -> jax.Array:
        """Explicit one-matrix materialization at the direct-solver boundary."""
        return penalty_ops.materialize(self.penalty_structure, log_lambda)

    @classmethod
    def from_setup(
        cls,
        setup: ModelSetup,
        family: ExponentialFamily,
        device: jax.Device | None = None,
    ) -> FittingData:
        if setup.X.ndim != 2:
            raise ValueError(f"Expected 2-D model matrix X, got ndim={setup.X.ndim}")
        structure = setup.penalties or PenaltyStructure(setup.X.shape[1], ())
        log_sp_init = cls._initial_sp(setup.X, structure, setup.weights)
        transformed = reparameterize_structure(structure)
        X_np = apply_transforms_to_design(setup.X, transformed)
        beta_init = to_jax(
            initialize_beta_cpu(X_np, setup.y, setup.weights, family, setup.offset),
            device=device,
        )
        metadata = _build_block_metadata(transformed, device)
        ranks = tuple(rank for block in structure.blocks for rank in block.ranks)
        # Compatibility metadata historically came from a p-by-p Penalty, so
        # its nullity includes coefficient directions outside this local block.
        null_dims = tuple(setup.X.shape[1] - rank for rank in ranks)

        # Only NB needs count metadata. Keeping max_y neutral for all other
        # families prevents response maxima from needlessly splitting JIT
        # cache entries. Fractional NB responses use the exact polygamma
        # derivative path; they are never rounded into recurrence indices.
        count_prefix_plan: CountPrefixPlan | None = None
        max_y = 0
        if family.family_name == "nb":
            y_np = np.asarray(setup.y, dtype=np.float64)
            integer_counts = bool(
                np.all(np.isfinite(y_np)) and np.all(y_np == np.floor(y_np))
            )
            if integer_counts:
                max_y = int(np.max(y_np)) if y_np.size else 0
                int64_max = np.iinfo(np.int64).max
                if max_y > int64_max:
                    raise ValueError(
                        "NB count-prefix indices require responses representable "
                        f"as int64, got maximum count {max_y}."
                    )
                capacity = max(1, max_y)  # y_safe=1 is evaluated under where.
                indices_np = y_np.astype(np.int64)
            else:
                capacity = 0
                indices_np = np.zeros(y_np.shape, dtype=np.int64)
            count_prefix_plan = CountPrefixPlan(
                indices=jax.device_put(
                    jnp.asarray(indices_np, dtype=jnp.int64), device
                ),
                max_count=max_y,
                capacity=capacity,
                integer_counts=integer_counts,
            )

        return cls(
            X=to_jax(X_np, device=device),
            y=to_jax(setup.y, device=device),
            wt=to_jax(setup.weights, device=device),
            offset=None
            if setup.offset is None
            else to_jax(setup.offset, device=device),
            penalty_structure=_to_jax_structure(transformed, device),
            log_lambda_init=to_jax(log_sp_init, device=device),
            family=family,
            n_obs=setup.n_obs,
            n_coef=setup.X.shape[1],
            penalty_ranks=ranks,
            penalty_null_dims=null_dims,
            total_penalty_rank=_total_penalty_rank(transformed),
            singleton_sp_indices=metadata["singleton_sp_indices"],
            singleton_ranks=metadata["singleton_ranks"],
            singleton_eig_constants=metadata["singleton_eig_constants"],
            multi_block_sp_indices=metadata["multi_block_sp_indices"],
            multi_block_ranks=metadata["multi_block_ranks"],
            multi_block_proj_S=metadata["multi_block_proj_S"],
            multi_block_S_local=metadata["multi_block_S_local"],
            max_y=max_y,
            count_prefix_plan=count_prefix_plan,
            rank_deficit=cls._unpenalized_rank_deficit(transformed, X_np),
            beta_init=beta_init,
            penalty_roots=_structural_penalty_roots(
                transformed, X_np, np.asarray(setup.weights, dtype=np.float64), device
            ),
            null_beta=_null_beta(X_np, setup.y, family, device),
        )

    @staticmethod
    def _initial_sp(
        X: np.ndarray, structure: PenaltyStructure, weights: np.ndarray
    ) -> np.ndarray:
        """R ``initial.sp`` scaling, retaining its old global-padding cutoff."""
        if structure.n_penalties == 0:
            return np.zeros(0)
        ldxx = FittingData._weighted_crossproduct_diag(X, weights)
        return initial_log_sp_from_diagonal(ldxx, structure)

    @staticmethod
    def _initial_sp_from_crossproduct_diag(
        X: np.ndarray, structure: PenaltyStructure, ldxx: np.ndarray
    ) -> np.ndarray:
        """Preserve the EFS adapter through shared CPU initial-sp balancing."""
        if ldxx.shape != (X.shape[1],):
            raise ValueError("Initial-sp diagonal must match the design columns")
        return initial_log_sp_from_diagonal(ldxx, structure)

    @staticmethod
    def _weighted_crossproduct_diag(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute diag(X' diag(weights) X) in bounded row batches."""
        diagonal = np.zeros(X.shape[1], dtype=np.float64)
        for start in range(0, X.shape[0], _INITIAL_SP_BATCH_ROWS):
            stop = min(start + _INITIAL_SP_BATCH_ROWS, X.shape[0])
            weighted = np.sqrt(weights[start:stop])[:, None] * X[start:stop]
            diagonal += np.sum(weighted * weighted, axis=0)
        return diagonal

    @staticmethod
    def _unpenalized_rank_deficit(structure: PenaltyStructure, X: np.ndarray) -> int:
        null_bases, _ = _total_penalty_spaces(structure)
        if not structure.blocks:
            return int(X.shape[1] - np.linalg.matrix_rank(X))
        nonempty = [(block, basis) for block, basis in null_bases if basis.shape[1]]
        covered = np.zeros(X.shape[1], dtype=bool)
        for block in structure.blocks:
            covered[block.start : block.stop] = True
        unpenalized = X[:, ~covered]
        if not nonempty:
            design_null = unpenalized
        else:
            design_null = np.column_stack(
                [
                    unpenalized,
                    *(
                        X[:, block.start : block.stop] @ basis
                        for block, basis in nonempty
                    ),
                ]
            )
        if design_null.shape[1] == 0:
            return 0
        return design_null.shape[1] - int(np.linalg.matrix_rank(design_null))


def _null_beta(
    X: np.ndarray, y: np.ndarray, family: ExponentialFamily, device
) -> jax.Array | None:
    """R's ``null.coef`` anchor, or None when the link mean is not finite."""
    try:
        return to_jax(r_null_coefficients(X, y, family), device=device)
    except ValueError:
        return None


def _structural_penalty_roots(
    structure: PenaltyStructure,
    X: np.ndarray,
    weights: np.ndarray,
    device: jax.Device | None,
) -> penalty_ops.JaxPenaltyRoots | None:
    """Device roots for the QR inner solve when the system is full rank.

    ``pls_fit1`` decides identifiability once on the lambda-free system
    ``[R1/|R1|; Es/|Es|]`` (gdi.c, ``R_cond`` with ``rank.tol = 100 eps``),
    with ``R1`` the factor of the prior-weighted design.  A structurally
    deficient system would have its unidentifiable columns dropped by
    ``pls_fit1``; that drop is not ported, so such models keep the Cholesky
    solve and its explicit fallback diagnostic.
    """
    if not structure.blocks or structure.n_penalties == 0:
        return None
    # Imported here: jaxgam.execution imports streamed routes that import
    # this module.
    from jaxgam.execution.qr import structural_rank

    roots = qr_penalty_roots(structure)
    weighted = np.sqrt(np.maximum(weights, 0.0))[:, None] * X
    data_R = np.linalg.qr(weighted, mode="r")
    local = [(slice(root.start, root.stop), root.root) for root in roots]
    if structural_rank(data_R, local, X.shape[1]) < X.shape[1]:
        return None
    return penalty_ops.JaxPenaltyRoots(
        n_coef=X.shape[1],
        starts=tuple(root.start for root in roots),
        stops=tuple(root.stop for root in roots),
        sp_indices=tuple(root.sp_index for root in roots),
        roots=tuple(to_jax(root.root, device=device) for root in roots),
    )


def _to_jax_penalty(
    penalty: object, device: jax.Device | None
) -> penalty_ops.JaxLocalPenalty:
    if isinstance(penalty, DenseLocalPenalty):
        return penalty_ops.JaxLocalPenalty(
            "dense", to_jax(penalty.matrix, device=device), penalty.size
        )
    if isinstance(penalty, DiagonalPenalty):
        return penalty_ops.JaxLocalPenalty(
            "diagonal", to_jax(penalty.diagonal, device=device), penalty.size
        )
    assert isinstance(penalty, IdentityPenalty)
    return penalty_ops.JaxLocalPenalty(
        "identity", to_jax(np.asarray(penalty.scale), device=device), penalty.size
    )


def _to_jax_transform(
    transform: object, device: jax.Device | None
) -> penalty_ops.JaxTransform:
    if isinstance(transform, DenseTransform):
        return penalty_ops.JaxTransform(
            "dense", to_jax(transform.matrix, device=device), transform.size
        )
    if isinstance(transform, DiagonalTransform):
        return penalty_ops.JaxTransform(
            "diagonal", to_jax(transform.diagonal, device=device), transform.size
        )
    assert isinstance(transform, IdentityTransform)
    return penalty_ops.JaxTransform(
        "identity", to_jax(np.asarray(1.0), device=device), transform.size
    )


def _to_jax_structure(
    structure: PenaltyStructure, device: jax.Device | None
) -> penalty_ops.JaxPenaltyStructure:
    return penalty_ops.JaxPenaltyStructure(
        structure.n_coef,
        tuple(
            penalty_ops.JaxPenaltyBlock(
                block.start,
                block.stop,
                block.sp_indices,
                tuple(_to_jax_penalty(p, device) for p in block.local_penalties),
                _to_jax_transform(block.transform, device),
            )
            for block in structure.blocks
        ),
    )


def _penalties_non_overlapping(penalties: list[np.ndarray]) -> bool:
    intervals: list[tuple[int, int]] = []
    for penalty in penalties:
        support = np.flatnonzero(np.sum(np.abs(penalty), axis=1) > 0)
        if len(support):
            intervals.append((int(support[0]), int(support[-1]) + 1))
    intervals.sort()
    return all(right[0] >= left[1] for left, right in pairwise(intervals))


def _total_penalty_spaces(
    structure: PenaltyStructure,
) -> tuple[
    list[tuple[PenaltyBlock, np.ndarray]], list[tuple[PenaltyBlock, np.ndarray]]
]:
    """Compatibility wrapper for the shared CPU penalty-space routine."""
    return total_penalty_spaces(structure)


def _total_penalty_rank(structure: PenaltyStructure) -> int:
    return sum(basis.shape[1] for _, basis in _total_penalty_spaces(structure)[1])


def _build_block_metadata(
    structure: PenaltyStructure, device: jax.Device | None
) -> dict[str, Any]:
    singletons: list[tuple[int, int, float]] = []
    multis: list[tuple[tuple[int, ...], int, list[np.ndarray], list[np.ndarray]]] = []
    for block in structure.blocks:
        penalties = list(block.dense_penalties())
        if len(penalties) == 1:
            _append_singleton(singletons, block.sp_indices[0], penalties[0])
        elif _penalties_non_overlapping(penalties):
            for sp, S in zip(block.sp_indices, penalties, strict=True):
                _append_singleton(singletons, sp, S)
        else:
            # A structurally present zero penalty still owns an sp index, but
            # it contributes neither a range direction nor a determinant
            # factor.  mgcv's norm-based combined range construction likewise
            # cannot normalize a zero matrix.
            normalized = [
                S / norm for S in penalties if (norm := np.linalg.norm(S, "fro")) > 0
            ]
            total = (
                np.add.reduce(normalized)
                if normalized
                else np.zeros((block.size, block.size))
            )
            eigs, U = np.linalg.eigh(total)
            rank = int(np.sum(eigs > np.max(eigs) * _EPS_TWO_THIRDS))
            range_basis = U[:, -rank:] if rank else U[:, :0]
            multis.append(
                (
                    block.sp_indices,
                    rank,
                    [range_basis.T @ S @ range_basis for S in penalties],
                    penalties,
                )
            )
    return {
        "singleton_sp_indices": tuple(x[0] for x in singletons),
        "singleton_ranks": tuple(x[1] for x in singletons),
        "singleton_eig_constants": to_jax(
            np.asarray([x[2] for x in singletons]), device=device
        ),
        "multi_block_sp_indices": tuple(x[0] for x in multis),
        "multi_block_ranks": tuple(x[1] for x in multis),
        "multi_block_proj_S": tuple(
            tuple(to_jax(S, device=device) for S in x[2]) for x in multis
        ),
        "multi_block_S_local": tuple(
            tuple(to_jax(S, device=device) for S in x[3]) for x in multis
        ),
    }


def _append_singleton(
    target: list[tuple[int, int, float]], sp: int, S: np.ndarray
) -> None:
    if S.size == 0:
        target.append((sp, 0, 0.0))
        return
    eigs = np.linalg.eigvalsh(S)
    active = eigs > np.max(np.abs(eigs)) * _EPS_TWO_THIRDS
    target.append(
        (
            sp,
            int(np.sum(active)),
            float(np.sum(np.log(np.maximum(eigs[active], _LOG_FLOOR)))),
        )
    )
