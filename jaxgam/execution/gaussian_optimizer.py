"""Host integration for frozen Gaussian QR statistics and Newton fitting.

The score provider owns only coefficient-sized device arrays during outer
optimization.  Training rows are retained solely by the finalizer needed to
construct the existing :class:`~jaxgam.fitting.pirls.PIRLSResult` contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.standard import Gaussian
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.gaussian import (
    GaussianQRScore,
    jit_gaussian_qr_reml_hessian,
    jit_gaussian_qr_reml_score,
    jit_gaussian_qr_reml_value_and_grad,
)
from jaxgam.fitting.pirls import PIRLSResult
from jaxgam.formula.rank_coordinates import IdentifiableCoordinateMap
from jaxgam.links.links import IdentityLink

from .gaussian import FrozenGaussianQRPreparation, prepare_frozen_gaussian_qr


class GaussianQRIneligible(ValueError):
    """The dense fit is valid, but outside the reviewed Gaussian QR route."""


@dataclass(frozen=True)
class GaussianQROuterState:
    """One coefficient-space trial with Newton's required scalar metadata."""

    score_state: GaussianQRScore
    coefficients: jax.Array
    fisher_factor: jax.Array
    xtwx: jax.Array
    edf: jax.Array
    scale: jax.Array


def _dense_transform(fd: FittingData) -> np.ndarray:
    """Return the block-local map from fitting to public coefficients."""
    result = np.eye(fd.n_coef)
    for block in fd.penalty_structure.blocks:
        transform = block.transform
        if transform.kind == "dense":
            values = np.asarray(transform.values, dtype=float)
        elif transform.kind == "diagonal":
            values = np.diag(np.asarray(transform.values, dtype=float))
        elif transform.kind == "identity":
            values = np.eye(transform.size)
        else:  # pragma: no cover - JaxTransform validates this upstream.
            raise ValueError("unsupported Gaussian QR fitting transform")
        result[block.start : block.stop, block.start : block.stop] = values
    return result


class DenseGaussianQROuterProvider:
    """Four-operation host seam consumed by :class:`NewtonOptimizer`.

    ``initial_parameters``, ``evaluate``, and ``differentiate`` touch only the
    frozen QR/penalty statistics.  ``finalize`` is the single post-optimization
    training-row evaluation needed by the legacy result surface.
    """

    def __init__(
        self,
        fitting: FittingData,
        preparation: FrozenGaussianQRPreparation,
    ) -> None:
        identity = np.eye(fitting.n_coef)
        if preparation.coordinate_basis.shape != identity.shape or not np.allclose(
            preparation.coordinate_basis, identity, rtol=0.0, atol=0.0
        ):
            raise GaussianQRIneligible(
                "The initial Gaussian QR optimizer requires full identity "
                "coordinate_basis."
            )
        if preparation.public_basis.shape != identity.shape or not np.allclose(
            preparation.public_basis, identity, rtol=0.0, atol=0.0
        ):
            raise GaussianQRIneligible(
                "The initial Gaussian QR optimizer requires full identity public_basis."
            )
        self._fitting = fitting
        self._preparation = preparation
        device = fitting.X.device
        self._compression = jax.device_put(preparation.compression, device)
        self._penalty_roots = tuple(
            jax.device_put(root, device) for root in preparation.penalty_roots
        )
        self._coordinate_basis = jax.device_put(preparation.coordinate_basis, device)
        self._public_basis = jax.device_put(preparation.public_basis, device)
        self._penalty_range_basis = jax.device_put(
            preparation.penalty_range_basis, device
        )
        public_from_fitting = _dense_transform(fitting)
        self._public_from_fitting = jax.device_put(public_from_fitting, device)
        self._fitting_from_public = jax.device_put(
            np.linalg.solve(public_from_fitting, np.eye(fitting.n_coef)), device
        )
        self._fixed_scale = jax.device_put(jnp.array(1.0, dtype=jnp.float64), device)
        self._kwargs = {
            "root_sp_indices": preparation.root_sp_indices,
            "n_smoothing_parameters": preparation.n_smoothing_parameters,
            "penalty_nullity": preparation.penalty_nullity,
            "scale_known": fitting.family.scale_known,
        }
        self._finalize_calls = 0

    @property
    def finalize_calls(self) -> int:
        """Number of final row materializations, exposed for execution gates."""
        return self._finalize_calls

    def initial_parameters(self, log_lambda: jax.Array) -> jax.Array:
        """Initialize unknown Gaussian scale from the frozen weighted QR."""
        log_lambda = jnp.asarray(log_lambda, dtype=jnp.float64)
        if self._fitting.family.scale_known:
            return log_lambda
        unit_scale = jnp.concatenate((log_lambda, jnp.zeros(1, log_lambda.dtype)))
        state, _ = self.evaluate(unit_scale)
        tiny = jnp.finfo(log_lambda.dtype).tiny
        log_scale = jnp.log(jnp.maximum(state.scale, tiny))
        return jnp.concatenate((log_lambda, log_scale[None]))

    def evaluate(self, params: jax.Array) -> tuple[GaussianQROuterState, jax.Array]:
        """Evaluate a trial without accepting any observation-sized input."""
        score_state = jit_gaussian_qr_reml_score(
            params,
            self._compression,
            self._penalty_roots,
            self._coordinate_basis,
            self._public_basis,
            self._penalty_range_basis,
            self._fixed_scale,
            **self._kwargs,
        )
        coefficients = self._fitting_from_public @ score_state.coefficients

        # Transform the stable public-coordinate QR factor to local-D fitting
        # coordinates with another QR.  This preserves H=L L' without forming
        # H or introducing the dense solver's normal-equation jitter.
        _, upper = jnp.linalg.qr(
            score_state.augmented_factor @ self._public_from_fitting,
            mode="reduced",
        )
        fisher_factor = upper.T
        data_factor = self._compression.R @ self._public_from_fitting
        xtwx = data_factor.T @ data_factor
        transformed_data = jax.scipy.linalg.solve_triangular(
            upper.T, data_factor.T, lower=True
        )
        edf = jnp.sum(transformed_data * transformed_data)
        denominator = jnp.maximum(
            self._compression.n_obs - edf, jnp.finfo(params.dtype).eps
        )
        # Gaussian has V(mu)=1 and dV/dmu=0, so Fletcher equals weighted
        # RSS/(n-edf).  The original n is retained by the compression.
        reported_scale = score_state.residual_sum_squares / denominator
        state = GaussianQROuterState(
            score_state=score_state,
            coefficients=coefficients,
            fisher_factor=fisher_factor,
            xtwx=xtwx,
            edf=edf,
            scale=reported_scale,
        )
        return state, score_state.score

    def differentiate(
        self, params: jax.Array, state: GaussianQROuterState
    ) -> tuple[jax.Array, jax.Array]:
        """Return exact score derivatives from the same frozen statistics."""
        del state
        _, gradient = jit_gaussian_qr_reml_value_and_grad(
            params,
            self._compression,
            self._penalty_roots,
            self._coordinate_basis,
            self._public_basis,
            self._penalty_range_basis,
            self._fixed_scale,
            **self._kwargs,
        )
        hessian = jit_gaussian_qr_reml_hessian(
            params,
            self._compression,
            self._penalty_roots,
            self._coordinate_basis,
            self._public_basis,
            self._penalty_range_basis,
            self._fixed_scale,
            **self._kwargs,
        )
        return gradient, (hessian + hessian.T) / 2.0

    def finalize(self, params: jax.Array, state: GaussianQROuterState) -> PIRLSResult:
        """Materialize the legacy final fit exactly once after optimization."""
        del params
        self._finalize_calls += 1
        fd = self._fitting
        offset = (
            jnp.zeros(fd.n_obs, dtype=fd.X.dtype) if fd.offset is None else fd.offset
        )
        eta = fd.X @ state.coefficients + offset
        mu = eta
        working_weights = jnp.clip(fd.wt, 1e-10, 1e10)
        return PIRLSResult(
            coefficients=state.coefficients,
            mu=mu,
            eta=eta,
            deviance=state.score_state.residual_sum_squares,
            penalized_deviance=state.score_state.penalized_residual_sum_squares,
            n_iter=jnp.array(1, dtype=jnp.int32),
            converged=jnp.array(True),
            scale=state.scale,
            XtWX=state.xtwx,
            L=state.fisher_factor,
            working_weights=working_weights,
            XtWX_fisher=state.xtwx,
            L_fisher=state.fisher_factor,
        )


def build_dense_gaussian_qr_provider(
    fitting: FittingData,
) -> DenseGaussianQROuterProvider:
    """Build the initial full-coordinate Gaussian QR route or fail explicitly."""
    if type(fitting.family) is not Gaussian or not isinstance(
        fitting.family.link, IdentityLink
    ):
        raise GaussianQRIneligible(
            "Gaussian QR requires the exact Gaussian family with identity link"
        )
    if fitting.n_penalties == 0:
        raise GaussianQRIneligible(
            "Gaussian QR smoothing optimization requires at least one penalty"
        )
    if fitting.rank_deficit:
        raise GaussianQRIneligible(
            "Gaussian QR currently requires a full identifiable coefficient map"
        )
    arrays = (fitting.X, fitting.y, fitting.wt)
    if fitting.offset is not None:
        arrays += (fitting.offset,)
    if not all(np.all(np.isfinite(np.asarray(value))) for value in arrays):
        raise GaussianQRIneligible("Gaussian QR requires finite dense fitting data")
    weight = np.asarray(fitting.wt)
    if np.any(weight < 1e-10) or np.any(weight > 1e10):
        raise GaussianQRIneligible(
            "Gaussian QR currently requires unclipped finite positive weights"
        )

    indices = np.arange(fitting.n_coef, dtype=np.int64)
    coordinate_map = IdentifiableCoordinateMap(
        basis=np.eye(fitting.n_coef),
        fitting_to_public=np.eye(fitting.n_coef),
        keep=indices,
        drop=np.empty(0, dtype=np.int64),
    )
    preparation = prepare_frozen_gaussian_qr(fitting, coordinate_map)
    return DenseGaussianQROuterProvider(fitting, preparation)
