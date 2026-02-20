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

from pymgcv.families.base import ExponentialFamily
from pymgcv.jax_utils import to_jax

if TYPE_CHECKING:
    from pymgcv.formula.design import ModelSetup

jax.config.update("jax_enable_x64", True)


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

    @property
    def n_penalties(self) -> int:
        """Number of penalty matrices."""
        return len(self.S_list)

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
        X_jax, y_jax, wt_jax = to_jax(setup.X, setup.y, setup.weights, device=device)

        offset_jax: jax.Array | None = None
        if setup.offset is not None:
            offset_jax = to_jax(setup.offset, device=device)

        # Transfer per-penalty matrices and extract metadata
        S_list: list[jax.Array] = []
        ranks: list[int] = []
        null_dims: list[int] = []

        if setup.penalties is not None:
            for penalty in setup.penalties.penalties:
                S_list.append(to_jax(penalty.S, device=device))
                ranks.append(penalty.rank)
                null_dims.append(penalty.null_space_dim)
            log_lambda_init = to_jax(
                setup.penalties.log_smoothing_params, device=device
            )
        else:
            # Purely parametric model — no penalties
            log_lambda_init = jnp.zeros(0)

        return cls(
            X=X_jax,
            y=y_jax,
            wt=wt_jax,
            offset=offset_jax,
            S_list=tuple(S_list),
            log_lambda_init=log_lambda_init,
            family=family,
            n_obs=setup.n_obs,
            n_coef=setup.X.shape[1],
            penalty_ranks=tuple(ranks),
            penalty_null_dims=tuple(null_dims),
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

        result = jnp.zeros((self.n_coef, self.n_coef))
        for j, S_j in enumerate(self.S_list):
            result = result + jnp.exp(log_lambda[j]) * S_j
        return result
