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

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.base import ExponentialFamily
from jaxgam.jax_utils import build_S_lambda, to_jax

if TYPE_CHECKING:
    from jaxgam.formula.design import ModelSetup, SmoothInfo

# Eigenvalue rank threshold: eigenvalues below max(eig) * _EPS_TWO_THIRDS
# are treated as zero. Matches R's Sl.setup / totalPenaltySpace convention.
_EPS_TWO_THIRDS = np.finfo(float).eps ** (2.0 / 3.0)

# Floor for log of eigenvalues to prevent log(0).
_LOG_FLOOR = 1e-30

# R's initial.sp threshold for identifying active penalty entries.
_ACTIVITY_THRESH = np.finfo(float).eps ** 0.8

# Maximum iterations for the initial.sp scaling loop.
_MAX_SP_ADJUST_ITERS = 200


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
    penalty_range_basis : jax.Array or None
        Orthogonal basis for the range space of the total penalty,
        shape (p, r) where r = p - Mp. Used for stable log|S+|.
    singleton_sp_indices : tuple[int, ...]
        Index into log_lambda for each singleton penalty block.
    singleton_ranks : tuple[int, ...]
        Rank of each singleton penalty's local block.
    singleton_eig_constants : jax.Array, shape (n_singletons,)
        Precomputed log-eigenvalue constants for singletons.
    multi_block_sp_indices : tuple[tuple[int, ...], ...]
        Log_lambda indices for each multi-penalty block.
    multi_block_ranks : tuple[int, ...]
        Combined penalty rank for each multi-penalty block.
    multi_block_proj_S : tuple[tuple[jax.Array, ...], ...]
        Range-space-projected penalties for each multi-penalty block.
    multi_block_S_local : tuple[tuple[jax.Array, ...], ...]
        Block-local (unprojected) penalties for adaptive reparam.
    repara_D : jax.Array or None
        Sl.setup reparameterization matrix (p, p), or None if no
        reparameterization was applied.
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
    penalty_range_basis: jax.Array | None

    # Block-structured log|S+| metadata
    singleton_sp_indices: tuple[int, ...]
    singleton_ranks: tuple[int, ...]
    singleton_eig_constants: jax.Array
    multi_block_sp_indices: tuple[tuple[int, ...], ...]
    multi_block_ranks: tuple[int, ...]
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...]
    multi_block_S_local: tuple[tuple[jax.Array, ...], ...]

    # Sl.setup reparameterization (R's fast-REML.r lines 68-429)
    repara_D: jax.Array | None

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
        if setup.X.ndim != 2:
            raise ValueError(f"Expected 2-D model matrix X, got ndim={setup.X.ndim}")

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
            log_sp_init = cls._initial_sp(setup.X, penalty_arrays, setup.weights)
            log_lambda_init = to_jax(log_sp_init, device=device)
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

        penalty_range_basis = cls._penalty_range_basis(penalty_arrays, n_coef, device)

        block_meta = _build_block_metadata(penalty_arrays, setup.smooth_info, device)

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
            singleton_sp_indices=block_meta["singleton_sp_indices"],
            singleton_ranks=block_meta["singleton_ranks"],
            singleton_eig_constants=block_meta["singleton_eig_constants"],
            multi_block_sp_indices=block_meta["multi_block_sp_indices"],
            multi_block_ranks=block_meta["multi_block_ranks"],
            multi_block_proj_S=block_meta["multi_block_proj_S"],
            multi_block_S_local=block_meta["multi_block_S_local"],
            repara_D=repara_D_jax,
        )

    @staticmethod
    def _initial_sp(
        X: np.ndarray,
        S_list: list[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        """Initial log smoothing parameters via R's ``initial.sp`` (mgcv.r:4626).

        Balances diag(X'WX) against each penalty diagonal so that the
        effective degrees of freedom are ~40% of the unpenalized model.
        Must use the original (pre-reparameterization) X and S.
        """
        n_penalties = len(S_list)
        if n_penalties == 0:
            return np.zeros(0)

        w = np.sqrt(np.maximum(weights, 0.0))
        wX = w[:, None] * X

        ldxx = np.sum(wX * wX, axis=0)
        def_sp = np.zeros(n_penalties)
        ldss = np.zeros_like(ldxx)
        pen = np.zeros(len(ldxx), dtype=bool)

        for i, S in enumerate(S_list):
            maS = np.max(np.abs(S))
            if maS == 0:
                continue
            rsS = np.mean(np.abs(S), axis=1)
            csS = np.mean(np.abs(S), axis=0)
            dS = np.abs(np.diag(S))
            thresh = _ACTIVITY_THRESH * maS
            ind = (rsS > thresh) & (csS > thresh) & (dS > thresh)

            ss = np.diag(S)[ind]
            xx = ldxx[ind]
            sizeXX = np.mean(xx) if len(xx) > 0 else 0.0
            sizeS = np.mean(ss) if len(ss) > 0 else 0.0

            if sizeS <= 0 or sizeXX <= 0:
                continue

            def_sp[i] = sizeXX / sizeS
            pen |= ind
            ldss += def_sp[i] * np.diag(S)

        idx = (ldss > 0) & pen & (ldxx > 0)
        if not np.any(idx):
            return np.zeros(n_penalties)

        ldxx_s = ldxx[idx].copy()
        ldss_s = ldss[idx].copy()

        for _ in range(_MAX_SP_ADJUST_ITERS):
            if np.mean(ldxx_s / (ldxx_s + ldss_s)) <= 0.4:
                break
            def_sp *= 10
            ldss_s *= 10
        for _ in range(_MAX_SP_ADJUST_ITERS):
            if np.mean(ldxx_s / (ldxx_s + ldss_s)) >= 0.4:
                break
            def_sp /= 10
            ldss_s /= 10

        def_sp = np.maximum(def_sp, np.finfo(float).tiny)
        return np.log(def_sp)

    @staticmethod
    def _penalty_range_basis(
        penalty_arrays: list[np.ndarray],
        n_coef: int,
        device: jax.Device | None,
    ) -> jax.Array | None:
        """Orthogonal basis for the range space of the total penalty.

        Eigendecomposes the normalized total penalty (R's totalPenaltySpace)
        and returns eigenvectors for non-zero eigenvalues. Used for
        stable ``log|S+|`` computation via range-space projection.
        """
        if not penalty_arrays:
            return None

        St = np.zeros((n_coef, n_coef))
        for S_np in penalty_arrays:
            norm_j = np.sqrt(np.sum(S_np * S_np))
            if norm_j > 0:
                St += S_np / norm_j
        eigs, vecs = np.linalg.eigh(St)
        threshold = np.max(eigs) * _EPS_TWO_THIRDS
        Mp = int(np.sum(eigs <= threshold))
        U_range = vecs[:, Mp:]
        return to_jax(U_range, device=device)

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


def _build_block_metadata(
    penalty_arrays: list[np.ndarray],
    smooth_info: tuple[SmoothInfo, ...] | None,
    device: jax.Device | None,
) -> dict[str, Any]:
    """Classify penalties into singleton and multi-penalty blocks.

    Penalties from different smooths occupy non-overlapping columns, so
    ``S_lambda`` is block-diagonal and ``log|S+| = sum(log|S+_block|)``.

    Singletons (one penalty per smooth) get an exact analytical
    derivative: ``log|S+| = rank * rho + const``.  Multi-penalty blocks
    (tensor products with overlapping penalties) use a scaled slogdet in
    the range space.  Factor-by smooths (non-overlapping multi-penalty)
    are split into independent singletons.

    Parameters
    ----------
    penalty_arrays : list[np.ndarray]
        Per-penalty (p, p) matrices (NumPy, not yet on device).
    smooth_info : tuple[SmoothInfo, ...] or None
        Per-smooth metadata with column ranges and penalty counts.
    device : jax.Device or None
        Target JAX device for array transfer.

    Returns
    -------
    dict[str, Any]
        Block metadata with keys:

        - ``singleton_sp_indices``: index into log_lambda per singleton
        - ``singleton_ranks``: rank of each singleton's penalty
        - ``singleton_eig_constants``: precomputed log-eigenvalue sums
        - ``multi_block_sp_indices``: log_lambda indices per block
        - ``multi_block_ranks``: combined penalty rank per block
        - ``multi_block_proj_S``: range-space-projected penalties
        - ``multi_block_S_local``: block-local unprojected penalties
    """
    singletons: list[tuple[int, int, float]] = []
    multi_blocks: list[tuple[tuple[int, ...], int, list, list]] = []

    if penalty_arrays and smooth_info:
        for si in smooth_info:
            if si.n_penalties == 0:
                continue
            col_start = si.first_coef
            col_stop = si.last_coef
            sp_indices = tuple(
                range(si.first_penalty, si.first_penalty + si.n_penalties)
            )

            if si.n_penalties == 1:
                sp_idx = sp_indices[0]
                S_local = penalty_arrays[sp_idx][col_start:col_stop, col_start:col_stop]
                eig_vals = np.linalg.eigvalsh(S_local)
                thresh = np.max(np.abs(eig_vals)) * _EPS_TWO_THIRDS
                nonzero = eig_vals > thresh
                rank = int(np.sum(nonzero))
                eig_const = float(
                    np.sum(np.log(np.maximum(eig_vals[nonzero], _LOG_FLOOR)))
                )
                singletons.append((sp_idx, rank, eig_const))
            else:
                S_locals = [
                    penalty_arrays[j][col_start:col_stop, col_start:col_stop]
                    for j in sp_indices
                ]
                non_overlapping = _penalties_non_overlapping(S_locals)

                if non_overlapping:
                    for idx, sp_idx in enumerate(sp_indices):
                        eig_vals = np.linalg.eigvalsh(S_locals[idx])
                        thresh = np.max(np.abs(eig_vals)) * _EPS_TWO_THIRDS
                        nonzero = eig_vals > thresh
                        rank = int(np.sum(nonzero))
                        eig_const = float(
                            np.sum(np.log(np.maximum(eig_vals[nonzero], _LOG_FLOOR)))
                        )
                        singletons.append((sp_idx, rank, eig_const))
                else:
                    St_local = np.zeros_like(S_locals[0])
                    for S in S_locals:
                        norm = np.linalg.norm(S, "fro")
                        if norm > 0:
                            St_local += S / norm
                    eig_vals, vecs = np.linalg.eigh(St_local)
                    thresh = np.max(eig_vals) * _EPS_TWO_THIRDS
                    rank = int(np.sum(eig_vals > thresh))
                    U_local = vecs[:, -rank:]
                    S_projs = [U_local.T @ S @ U_local for S in S_locals]
                    multi_blocks.append((sp_indices, rank, S_projs, S_locals))

    eig_consts_np = np.array([s[2] for s in singletons]) if singletons else np.array([])
    return {
        "singleton_sp_indices": tuple(s[0] for s in singletons),
        "singleton_ranks": tuple(s[1] for s in singletons),
        "singleton_eig_constants": to_jax(eig_consts_np, device=device),
        "multi_block_sp_indices": tuple(mb[0] for mb in multi_blocks),
        "multi_block_ranks": tuple(mb[1] for mb in multi_blocks),
        "multi_block_proj_S": tuple(
            tuple(to_jax(S, device=device) for S in mb[2]) for mb in multi_blocks
        ),
        "multi_block_S_local": tuple(
            tuple(to_jax(S, device=device) for S in mb[3]) for mb in multi_blocks
        ),
    }


def _penalties_non_overlapping(S_locals: list[np.ndarray]) -> bool:
    """Check if multi-penalty matrices have non-overlapping supports.

    Two penalties have non-overlapping supports if there is no position
    where both have nonzero entries (i.e. their element-wise product
    is zero everywhere). This is the case for factor-by smooths where
    each level's penalty occupies a different sub-block.
    """
    for j in range(len(S_locals)):
        for k in range(j + 1, len(S_locals)):
            if np.any(np.abs(S_locals[j]) * np.abs(S_locals[k]) > 0):
                return False
    return True


def _compute_repara_D(
    penalty_arrays: list[np.ndarray],
    smooth_info: tuple[SmoothInfo, ...],
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
    smooth_info : tuple[SmoothInfo, ...]
        Per-smooth metadata with column ranges and penalty counts.
    n_coef : int
        Total number of coefficients.

    Returns
    -------
    np.ndarray or None
        (p, p) reparameterization matrix, or None if no transform needed.
    """
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
            threshold = max(eigs.max(), 0) * _EPS_TWO_THIRDS
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

            if _penalties_non_overlapping(S_locals):
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
                    threshold = max(eigs.max(), 0) * _EPS_TWO_THIRDS
                    D_diag = np.ones(sub_k)
                    mask = eigs > threshold
                    D_diag[mask] = 1.0 / np.sqrt(eigs[mask])
                    D_block[sub_start:sub_stop, sub_start:sub_stop] = U * D_diag

                D_global[col_start:col_stop, col_start:col_stop] = D_block
                modified = True
            else:
                # Tensor product: rotate into eigenspace of total penalty
                St = functools.reduce(np.add, S_locals)
                _eigs, U = np.linalg.eigh(St)
                D_global[col_start:col_stop, col_start:col_stop] = U
                modified = True

    return D_global if modified else None
