"""Phase 1→2 boundary container for PIRLS/REML fitting.

``FittingData`` encapsulates device transfer and penalty structure in one
place. Created from a ``ModelSetup`` (Phase 1, NumPy) via the
``from_setup()`` factory, it holds all JAX arrays and metadata that
PIRLS and REML need.

Key design points:
- Not a JAX pytree (contains Python objects like ``family``). Used as an
  orchestration container that provides arrays to fitting functions.
- ``S_lambda()`` is pure JAX and differentiable — REML will differentiate
  through it via ``jax.grad``.
- ``coef_map`` and ``smooth_info`` bypass Phase 2 entirely (Python objects
  that can't cross the JAX boundary). They stay on ``ModelSetup`` and
  flow directly to Phase 3.

Design doc reference: docs/design.md §1.3 (phase boundaries), §4.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from pymgcv.families.base import ExponentialFamily
from pymgcv.jax_utils import build_S_lambda, to_jax

if TYPE_CHECKING:
    from pymgcv.formula.design import ModelSetup


@dataclass(frozen=True)
class FittingData:
    """Phase 1→2 boundary: on-device data for PIRLS/REML.

    Created via ``FittingData.from_setup(model_setup, family)``.
    Not a JAX pytree (contains Python objects). Used as an
    orchestration container that provides arrays to fitting functions.

    Attributes
    ----------
    X : jax.Array, shape (n, p)
        Model matrix on device.
    y : jax.Array, shape (n,)
        Response vector on device.
    wt : jax.Array, shape (n,)
        Prior weights on device.
    offset : jax.Array or None
        Offset vector on device, shape (n,), or None.
    S_list : tuple[jax.Array, ...]
        Per-penalty (p, p) matrices on device.
    log_lambda_init : jax.Array, shape (n_penalties,)
        Initial log smoothing parameters.
    family : ExponentialFamily
        Family with link attached (Python object, not traced).
    n_obs : int
        Number of observations.
    n_coef : int
        Number of coefficients (columns of X).
    penalty_ranks : tuple[int, ...]
        Rank of each penalty matrix (for REML log|S^+|).
    penalty_null_dims : tuple[int, ...]
        Null space dimension of each penalty (for REML).
    """

    # Arrays on device
    X: jax.Array
    y: jax.Array
    wt: jax.Array
    offset: jax.Array | None
    S_list: tuple[jax.Array, ...]
    log_lambda_init: jax.Array

    # Python objects
    family: ExponentialFamily

    # Static metadata
    n_obs: int
    n_coef: int
    penalty_ranks: tuple[int, ...]
    penalty_null_dims: tuple[int, ...]
    penalty_range_basis: jax.Array | None  # shape (p, r), r = p - Mp

    # Block-structured log|S+| metadata
    singleton_sp_indices: tuple[int, ...]  # index into log_lambda per singleton
    singleton_ranks: tuple[int, ...]  # rank of each singleton's penalty
    singleton_eig_constants: jax.Array  # (n_singletons,) precomputed constants
    multi_block_sp_indices: tuple[tuple[int, ...], ...]  # log_lambda indices per block
    multi_block_ranks: tuple[int, ...]  # combined penalty rank per block
    multi_block_proj_S: tuple[
        tuple[jax.Array, ...], ...
    ]  # projected penalties per block
    multi_block_S_local: tuple[
        tuple[jax.Array, ...], ...
    ]  # block-local (unprojected) penalties for adaptive reparam

    # Sl.setup reparameterization (R's fast-REML.r lines 68-429)
    repara_D: jax.Array | None  # (p, p) back-transform matrix, or None

    @property
    def n_penalties(self) -> int:
        """Number of penalty matrices."""
        return len(self.S_list)

    @property
    def total_penalty_null_dim(self) -> int:
        """Null space dimension of the total penalty matrix.

        This is R's ``Mp`` — the dimension of the kernel of
        ``S_total = Σ S_j / ||S_j||`` (R's ``totalPenaltySpace``,
        gam.fit3.r line 2661). It differs from ``sum(penalty_null_dims)``
        when there are multiple penalties with overlapping null spaces.

        Used in the REML criterion's ``-Mp/2·log(2πφ)`` term.
        """
        if self.penalty_range_basis is None:
            return self.n_coef
        return self.n_coef - self.penalty_range_basis.shape[1]

    @classmethod
    def from_setup(
        cls,
        setup: ModelSetup,
        family: ExponentialFamily,
        device: jax.Device | None = None,
    ) -> FittingData:
        """Create FittingData from a Phase 1 ModelSetup.

        Transfers arrays to the JAX device and extracts penalty metadata.

        Parameters
        ----------
        setup : ModelSetup
            Phase 1 output (NumPy arrays, penalty structure).
        family : ExponentialFamily
            Family with link attached.
        device : jax.Device, optional
            Target device. If None, uses JAX's default device.

        Returns
        -------
        FittingData
            On-device data ready for PIRLS/REML.
        """
        y_jax = to_jax(setup.y, device=device)
        wt_jax = to_jax(setup.weights, device=device)

        offset_jax: jax.Array | None = None
        if setup.offset is not None:
            offset_jax = to_jax(setup.offset, device=device)

        # Extract per-penalty matrices and metadata (NumPy only;
        # transfer to JAX after Sl.setup reparameterization below).
        penalty_arrays: list[np.ndarray] = []
        ranks: list[int] = []
        null_dims: list[int] = []

        if setup.penalties is not None:
            for penalty in setup.penalties.penalties:
                penalty_arrays.append(penalty.S)
                ranks.append(penalty.rank)
                null_dims.append(penalty.null_space_dim)
            log_lambda_init = to_jax(
                setup.penalties.log_smoothing_params, device=device
            )
        else:
            # Purely parametric model — no penalties
            log_lambda_init = jnp.zeros(0)

        # -- Sl.setup reparameterization (R's fast-REML.r lines 68-429) --
        # For each smooth, reparameterize so that singleton penalties
        # become partial identities (D^T S D = I_r). This makes
        # S_lambda = lambda * I_r, yielding a well-conditioned Hessian
        # and enabling fast Newton convergence (~10 iterations vs 100+).
        # The REML criterion is invariant (2*log|D| cancels between
        # log|H| and log|S+|). After optimization, coefficients and Vp
        # are back-transformed: beta_orig = D @ beta_repara.
        n_coef = setup.X.shape[1]
        X_np = setup.X
        repara_D_jax: jax.Array | None = None

        if penalty_arrays and setup.smooth_info:
            D_global = _compute_repara_D(penalty_arrays, setup.smooth_info, n_coef)
            if D_global is not None:
                X_np = setup.X @ D_global
                penalty_arrays = [D_global.T @ S @ D_global for S in penalty_arrays]
                repara_D_jax = to_jax(D_global, device=device)

        # Transfer model matrix and penalties to device
        X_jax = to_jax(X_np, device=device)
        S_list: tuple[jax.Array, ...] = tuple(
            to_jax(S, device=device) for S in penalty_arrays
        )

        # Compute range space basis for stable log|S+| computation.
        # Eigendecompose the normalized total penalty (R's totalPenaltySpace)
        # and keep the eigenvectors for non-zero eigenvalues.
        penalty_range_basis: jax.Array | None = None
        if penalty_arrays:
            St = np.zeros((n_coef, n_coef))
            for S_np in penalty_arrays:
                norm_j = np.sqrt(np.sum(S_np * S_np))
                if norm_j > 0:
                    St += S_np / norm_j
            eigs, vecs = np.linalg.eigh(St)
            threshold = np.max(eigs) * np.finfo(float).eps ** (2.0 / 3.0)
            Mp = int(np.sum(eigs <= threshold))
            U_range = vecs[:, Mp:]  # (p, r) orthogonal basis for range space
            penalty_range_basis = to_jax(U_range, device=device)

        # Block-structured log|S+| metadata.
        # Classify penalties into singleton blocks (exact derivative) and
        # multi-penalty blocks (scaled slogdet). Since penalties from
        # different smooths occupy non-overlapping columns, S_lambda is
        # block-diagonal and log|S+| = sum(log|S+_block|).
        eps_thresh = np.finfo(float).eps ** (2.0 / 3.0)
        singletons: list[tuple[int, int, float]] = []  # (sp_idx, rank, eig_const)
        multi_blocks: list[
            tuple[tuple[int, ...], int, list[np.ndarray]]
        ] = []  # (sp_indices, rank, [S_proj, ...])

        if penalty_arrays and setup.smooth_info:
            for si in setup.smooth_info:
                if si.n_penalties == 0:
                    continue
                col_start = si.first_coef
                col_stop = si.last_coef
                sp_indices = tuple(
                    range(si.first_penalty, si.first_penalty + si.n_penalties)
                )

                if si.n_penalties == 1:
                    # Singleton: exact formula log|S+| = rank * rho + const
                    sp_idx = sp_indices[0]
                    S_local = penalty_arrays[sp_idx][
                        col_start:col_stop, col_start:col_stop
                    ]
                    eig_vals = np.linalg.eigvalsh(S_local)
                    thresh = np.max(np.abs(eig_vals)) * eps_thresh
                    nonzero = eig_vals > thresh
                    rank = int(np.sum(nonzero))
                    eig_const = float(
                        np.sum(np.log(np.maximum(eig_vals[nonzero], 1e-30)))
                    )
                    singletons.append((sp_idx, rank, eig_const))
                else:
                    # Multi-penalty: check non-overlapping (factor-by) vs
                    # overlapping (tensor product)
                    S_locals = [
                        penalty_arrays[j][col_start:col_stop, col_start:col_stop]
                        for j in sp_indices
                    ]
                    non_overlapping = True
                    for j in range(len(S_locals)):
                        for k in range(j + 1, len(S_locals)):
                            if np.any(np.abs(S_locals[j]) * np.abs(S_locals[k]) > 0):
                                non_overlapping = False
                                break
                        if not non_overlapping:
                            break

                    if non_overlapping:
                        # Factor-by: split into singleton sub-blocks
                        for idx, sp_idx in enumerate(sp_indices):
                            eig_vals = np.linalg.eigvalsh(S_locals[idx])
                            thresh = np.max(np.abs(eig_vals)) * eps_thresh
                            nonzero = eig_vals > thresh
                            rank = int(np.sum(nonzero))
                            eig_const = float(
                                np.sum(np.log(np.maximum(eig_vals[nonzero], 1e-30)))
                            )
                            singletons.append((sp_idx, rank, eig_const))
                    else:
                        # Tensor product: scaled slogdet in range space
                        St_local = np.zeros_like(S_locals[0])
                        for S in S_locals:
                            norm = np.linalg.norm(S, "fro")
                            if norm > 0:
                                St_local += S / norm
                        eig_vals, vecs = np.linalg.eigh(St_local)
                        thresh = np.max(eig_vals) * eps_thresh
                        rank = int(np.sum(eig_vals > thresh))
                        U_local = vecs[:, -rank:]  # (block_size, rank)
                        S_projs = [U_local.T @ S @ U_local for S in S_locals]
                        multi_blocks.append((sp_indices, rank, S_projs, S_locals))

        # Pack block metadata
        singleton_sp_indices = tuple(s[0] for s in singletons)
        singleton_ranks_tup = tuple(s[1] for s in singletons)
        eig_consts_np = (
            np.array([s[2] for s in singletons]) if singletons else np.array([])
        )
        singleton_eig_constants = to_jax(eig_consts_np, device=device)

        multi_block_sp_indices = tuple(mb[0] for mb in multi_blocks)
        multi_block_ranks_tup = tuple(mb[1] for mb in multi_blocks)
        multi_block_proj_S = tuple(
            tuple(to_jax(S, device=device) for S in mb[2]) for mb in multi_blocks
        )
        multi_block_S_local = tuple(
            tuple(to_jax(S, device=device) for S in mb[3]) for mb in multi_blocks
        )

        return cls(
            X=X_jax,
            y=y_jax,
            wt=wt_jax,
            offset=offset_jax,
            S_list=S_list,
            log_lambda_init=log_lambda_init,
            family=family,
            n_obs=setup.n_obs,
            n_coef=n_coef,
            penalty_ranks=tuple(ranks),
            penalty_null_dims=tuple(null_dims),
            penalty_range_basis=penalty_range_basis,
            singleton_sp_indices=singleton_sp_indices,
            singleton_ranks=singleton_ranks_tup,
            singleton_eig_constants=singleton_eig_constants,
            multi_block_sp_indices=multi_block_sp_indices,
            multi_block_ranks=multi_block_ranks_tup,
            multi_block_proj_S=multi_block_proj_S,
            multi_block_S_local=multi_block_S_local,
            repara_D=repara_D_jax,
        )

    def S_lambda(self, log_lambda: jax.Array) -> jax.Array:
        """Compute S_lambda = sum_j exp(log_lambda[j]) * S_j.

        JAX-traceable: REML differentiates through this via ``jax.grad``.

        Parameters
        ----------
        log_lambda : jax.Array, shape (n_penalties,)
            Log smoothing parameters.

        Returns
        -------
        jax.Array, shape (n_coef, n_coef)
            Combined weighted penalty matrix.
        """
        if self.n_penalties == 0:
            return jnp.zeros((self.n_coef, self.n_coef))

        return build_S_lambda(log_lambda, self.S_list, self.n_coef)


def _compute_repara_D(
    penalty_arrays: list[np.ndarray],
    smooth_info: tuple,
    n_coef: int,
) -> np.ndarray | None:
    """Compute Sl.setup reparameterization matrix.

    Builds a block-diagonal ``D_global`` so that in the reparameterized
    coordinate system, singleton penalties become partial identities
    (``D^T S D = I_r``) and tensor product penalties are rotated into
    the eigenspace of their total penalty.

    Parameters
    ----------
    penalty_arrays : list[np.ndarray]
        Per-penalty (p, p) matrices (NumPy, not yet on device).
    smooth_info : tuple of SmoothInfo
        Per-smooth metadata with column ranges and penalty counts.
    n_coef : int
        Total number of coefficients.

    Returns
    -------
    np.ndarray or None
        (p, p) reparameterization matrix, or None if no transform needed.
    """
    eps_23 = np.finfo(float).eps ** (2.0 / 3.0)
    D_global = np.eye(n_coef)
    modified = False

    for si in smooth_info:
        if si.n_penalties == 0:
            continue

        col_start = si.first_coef
        col_stop = si.last_coef
        block_size = col_stop - col_start
        sp_indices = list(range(si.first_penalty, si.first_penalty + si.n_penalties))

        if si.n_penalties == 1:
            # Singleton: eigendecompose and scale so D^T S D = I_r
            sp_idx = sp_indices[0]
            S_local = penalty_arrays[sp_idx][col_start:col_stop, col_start:col_stop]
            eigs, U = np.linalg.eigh(S_local)
            threshold = max(eigs.max(), 0) * eps_23
            D_diag = np.ones(block_size)
            mask = eigs > threshold
            D_diag[mask] = 1.0 / np.sqrt(eigs[mask])
            D_block = U * D_diag  # U @ diag(D_diag)
            D_global[col_start:col_stop, col_start:col_stop] = D_block
            modified = True
        else:
            # Multi-penalty: check overlapping vs non-overlapping
            S_locals = [
                penalty_arrays[j][col_start:col_stop, col_start:col_stop]
                for j in sp_indices
            ]
            non_overlapping = True
            for j in range(len(S_locals)):
                for k in range(j + 1, len(S_locals)):
                    if np.any(np.abs(S_locals[j]) * np.abs(S_locals[k]) > 0):
                        non_overlapping = False
                        break
                if not non_overlapping:
                    break

            if non_overlapping:
                # Factor-by: apply singleton treatment per sub-block
                D_block = np.eye(block_size)
                for S_local in S_locals:
                    row_sums = np.sum(np.abs(S_local), axis=1)
                    nonzero_rows = np.where(row_sums > 0)[0]
                    if len(nonzero_rows) == 0:
                        continue
                    sub_start = int(nonzero_rows[0])
                    sub_stop = int(nonzero_rows[-1]) + 1
                    S_sub = S_local[sub_start:sub_stop, sub_start:sub_stop]
                    sub_k = sub_stop - sub_start

                    eigs, U = np.linalg.eigh(S_sub)
                    threshold = max(eigs.max(), 0) * eps_23
                    D_diag = np.ones(sub_k)
                    mask = eigs > threshold
                    D_diag[mask] = 1.0 / np.sqrt(eigs[mask])
                    D_block[sub_start:sub_stop, sub_start:sub_stop] = U * D_diag

                D_global[col_start:col_stop, col_start:col_stop] = D_block
                modified = True
            else:
                # Tensor product: rotate into eigenspace of total penalty
                St = sum(S_locals)
                _eigs, U = np.linalg.eigh(St)
                D_global[col_start:col_stop, col_start:col_stop] = U
                modified = True

    return D_global if modified else None
