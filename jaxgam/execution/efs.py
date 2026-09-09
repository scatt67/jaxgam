"""Dense regular-family extended Fellner--Schall execution adapter.

This is intentionally an internal controller. It supports known-scale
Poisson/log, binomial/logit, and NB/log (fixed theta or an explicitly staged
conditional-theta route), plus regular unknown-scale Gaussian/identity and
Gamma inverse/log. Accepted and trial states remain separate so a rejected
refit can never leak into the next iteration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.base import ExponentialFamily
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Gamma, Gaussian
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.efs import (
    EFSRawUpdate,
    EFSStatistics,
    EFSStatisticsPlan,
    efs_raw_update,
    efs_statistics,
    prepare_efs_statistics,
)
from jaxgam.fitting.pirls import (
    _EFS_STATUS_DIVERGENCE_RECOVERY_FAILED,
    _EFS_STATUS_DOMAIN_RECOVERY_FAILED,
    _EFS_STATUS_NONFINITE_RECOVERY_FAILED,
    _W_MAX,
    _W_MIN,
    PIRLSResult,
    efs_theta_pirls_loop,
    pirls_loop,
)
from jaxgam.fitting.reml import estimate_edf, fletcher_scale, reml_criterion
from jaxgam.links.links import IdentityLink, InverseLink, LogitLink, LogLink

if TYPE_CHECKING:
    from jaxgam.formula.design import ModelSetup


def efs_initial_log_lambda(setup: ModelSetup, family: ExponentialFamily) -> jax.Array:
    """Prepare mgcv ``initial.spg`` start only for supported EFS families.

    Regular fitting never incurs this initialized-mean/working-weight work.
    The shared initial-sp routine keeps its bounded original-coordinate
    weighted-crossproduct reduction.
    """
    structure = setup.penalties
    if structure is None or structure.n_penalties == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    ldxx = np.zeros(setup.X.shape[1], dtype=np.float64)
    nb_observed_ldxx: np.ndarray | None = None
    nb_fisher_ldxx: np.ndarray | None = None
    nb_has_negative_observed = False
    nb_observed_valid = True
    nb_fisher_valid = True
    if isinstance(family, NegativeBinomial):
        nb_observed_ldxx = np.zeros_like(ldxx)
        nb_fisher_ldxx = np.zeros_like(ldxx)
    batch_rows = 8192
    for start in range(0, setup.X.shape[0], batch_rows):
        stop = min(start + batch_rows, setup.X.shape[0])
        y = setup.y[start:stop]
        prior = setup.weights[start:stop]
        mu = np.asarray(family.initialize(y, prior), dtype=np.float64)
        eta = np.asarray(family.link.link(mu), dtype=np.float64)
        mu_eta = np.asarray(family.link.mu_eta(eta), dtype=np.float64)
        if isinstance(family, NegativeBinomial):
            # initial.spg's extended-family branch (mgcv.r:4787-4794) uses
            # half Dmu2 times mu.eta², falling back to expected curvature if
            # *any* observed start weight is negative. Accumulate both
            # bounded coordinate diagonals before selecting R's global branch;
            # do not let the batch boundary affect this decision.
            theta = float(family.get_theta(transformed=True)[0])
            dmu2 = -2.0 * prior * ((y + theta) / (mu + theta) ** 2 - y / mu**2)
            observed = 0.5 * dmu2 * mu_eta**2
            variance = np.asarray(family.variance(mu), dtype=np.float64)
            fisher = prior * mu_eta**2 / variance
            nb_has_negative_observed |= bool(np.any(observed < 0))
            nb_observed_valid &= bool(
                np.all(np.isfinite(observed)) and np.all(observed > 0)
            )
            nb_fisher_valid &= bool(np.all(np.isfinite(fisher)) and np.all(fisher > 0))
            X_batch = setup.X[start:stop]
            assert nb_observed_ldxx is not None
            assert nb_fisher_ldxx is not None
            nb_observed_ldxx += np.sum(observed[:, None] * X_batch**2, axis=0)
            nb_fisher_ldxx += np.sum(fisher[:, None] * X_batch**2, axis=0)
            continue
        else:
            variance = np.asarray(family.variance(mu), dtype=np.float64)
            working = prior * mu_eta**2 / variance
        if not np.all(np.isfinite(working)) or np.any(working <= 0):
            raise ValueError(
                "EFS initial.spg working weights must be finite and positive"
            )
        weighted_X = np.sqrt(working)[:, None] * setup.X[start:stop]
        ldxx += np.sum(weighted_X * weighted_X, axis=0)
    if isinstance(family, NegativeBinomial):
        if nb_has_negative_observed:
            valid = nb_fisher_valid
            assert nb_fisher_ldxx is not None
            ldxx = nb_fisher_ldxx
        else:
            valid = nb_observed_valid
            assert nb_observed_ldxx is not None
            ldxx = nb_observed_ldxx
        if not valid:
            raise ValueError(
                "EFS initial.spg working weights must be finite and positive"
            )
    return jnp.asarray(
        FittingData._initial_sp_from_crossproduct_diag(setup.X, structure, ldxx)
    )


def efs_initial_log_scale(setup: ModelSetup, family: ExponentialFamily) -> jax.Array:
    """Return mgcv EFS's incoming unknown-scale value.

    This is deliberately separate from the Newton scale initialization.  In
    pinned ``estimate.gam`` the EFS parameter starts at ``null.scale / 10``;
    ``get.null.coef`` forms that scale from the *unweighted* response mean,
    but evaluates the family deviance with prior weights and divides by the
    original number of rows.  In particular it is not ``GAMResults``'s null
    deviance and it is not an RSS/(n-p) shortcut.
    """
    if family.scale_known:
        return jnp.array(0.0, dtype=jnp.float64)
    return _efs_initial_log_scale_from_arrays(setup.y, setup.weights, family)


def _efs_initial_log_scale_from_arrays(
    y: np.ndarray, wt: np.ndarray, family: ExponentialFamily
) -> jax.Array:
    """Array-level implementation shared by setup and dense fit adapters."""
    if y.ndim != 1 or y.size == 0 or wt.shape != y.shape:
        raise ValueError(
            "EFS initial scale requires nonempty aligned response and weights"
        )
    # Keep all reductions bounded by the initialization batch.  This is also
    # where Python family initialization performs response-domain validation,
    # before we reproduce get.null.coef's unweighted mean convention.
    response_sum = 0.0
    n_obs = int(y.shape[0])
    batch_rows = 8192
    for start in range(0, n_obs, batch_rows):
        stop = min(start + batch_rows, n_obs)
        y_batch = np.asarray(y[start:stop], dtype=np.float64)
        wt_batch = np.asarray(wt[start:stop], dtype=np.float64)
        if (
            not np.all(np.isfinite(y_batch))
            or not np.all(np.isfinite(wt_batch))
            or np.any(wt_batch <= 0)
        ):
            raise ValueError(
                "EFS initial scale requires finite strictly positive weights"
            )
        family.initialize(y_batch, wt_batch)
        response_sum += float(np.sum(y_batch))
    response_mean = response_sum / n_obs
    deviance = 0.0
    for start in range(0, n_obs, batch_rows):
        stop = min(start + batch_rows, n_obs)
        y_batch = np.asarray(y[start:stop], dtype=np.float64)
        wt_batch = np.asarray(wt[start:stop], dtype=np.float64)
        mu_batch = np.full(y_batch.shape, response_mean, dtype=np.float64)
        deviance += float(np.asarray(family.dev_resids(y_batch, mu_batch, wt_batch)))
    null_scale = deviance / n_obs
    incoming_phi = null_scale / 10.0
    if not np.isfinite(incoming_phi) or incoming_phi <= 0:
        raise ValueError(
            "EFS initial unknown scale null.scale / 10 must be finite and positive"
        )
    return jnp.asarray(np.log(incoming_phi), dtype=jnp.float64)


@dataclass(frozen=True)
class EFSControl:
    """Pinned ``efsudr`` controls for the dense regular-family path."""

    outer_limit: int = 200
    log_lambda_max: float = 15.0
    score_tolerance: float = 0.1
    pirls_tolerance: float = 1e-7
    pirls_max_iter: int = 200
    history_limit: int = 200

    def __post_init__(self) -> None:
        for name, value in (
            ("outer_limit", self.outer_limit),
            ("pirls_max_iter", self.pirls_max_iter),
            ("history_limit", self.history_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"EFS {name} must be a positive integer")
        if not np.isfinite(self.log_lambda_max):
            raise ValueError("EFS log_lambda_max must be finite")
        if (
            not np.isfinite(self.score_tolerance)
            or not np.isfinite(self.pirls_tolerance)
            or self.score_tolerance < 0
            or self.pirls_tolerance <= 0
        ):
            raise ValueError("EFS tolerances must be finite and positive")


DEFAULT_EFS_CONTROL = EFSControl()

# Module-level compiled kernels deliberately avoid constructing a fresh closure
# for every accepted/trial refit.  The statistics plan is a dynamic pytree;
# only family/range metadata keys the scalar-score executable.
_jit_efs_statistics = jax.jit(efs_statistics)
_jit_reml_score = jax.jit(
    reml_criterion,
    static_argnames=(
        "Mp",
        "rank_deficit",
        "singleton_sp_indices",
        "singleton_ranks",
        "multi_block_sp_indices",
        "multi_block_ranks",
    ),
)


@dataclass(frozen=True)
class EFSFitState:
    """One immutable accepted or trial fit, all in fitting coordinates."""

    log_lambda: jax.Array
    pirls_result: PIRLSResult
    score: jax.Array
    edf: jax.Array
    statistics: EFSStatistics
    raw_update: EFSRawUpdate | None
    valid: bool
    inner_converged: bool
    # ``efsudr`` scores a fit at the incoming scale, then writes the fit's
    # Fletcher estimate into the next lsp.  Keeping all three names avoids a
    # tempting, but incorrect, score re-evaluation at the reported scale.
    score_phi: jax.Array | None = None
    update_phi: jax.Array | None = None
    reported_phi: jax.Array | None = None
    carried_phi: jax.Array | None = None
    # Estimated NB theta is an explicit immutable fit result.  In particular,
    # it is never read back from a mutable family after a rejected EFS trial.
    log_theta: jax.Array | None = None
    theta_status: jax.Array | None = None
    theta_loop_status: jax.Array | None = None
    theta_n_iter: jax.Array | None = None
    stopping_penalized_deviance: jax.Array | None = None


@dataclass(frozen=True)
class EFSResult:
    """Optimizer-neutral fields consumed by result materialization.

    ``update_residual`` is deliberately not exposed as a REML gradient.
    """

    log_lambda: jax.Array
    smoothing_params: jax.Array
    converged: bool
    n_iter: int
    score: jax.Array
    edf: jax.Array
    scale: jax.Array
    pirls_result: PIRLSResult
    convergence_info: str
    theta: float | None = None
    update_residual: jax.Array | None = None
    score_history: tuple[float, ...] = ()
    multiplier: float = 1.0
    score_phi: jax.Array | None = None
    update_phi: jax.Array | None = None
    reported_phi: jax.Array | None = None
    carried_phi: jax.Array | None = None
    score_phi_history: tuple[float, ...] = ()


class ConsumedFitResult(Protocol):
    """The subset of an optimizer result used by ``GAMResults._from_fit``."""

    smoothing_params: jax.Array
    converged: bool
    n_iter: int
    score: jax.Array
    edf: jax.Array
    scale: jax.Array
    pirls_result: PIRLSResult
    convergence_info: str
    theta: float | None


def _known_scale_family_supported(fd: FittingData) -> bool:
    family = fd.family
    return family.scale_known and (
        (family.family_name == "poisson" and isinstance(family.link, LogLink))
        or (family.family_name == "binomial" and isinstance(family.link, LogitLink))
        or (
            isinstance(family, NegativeBinomial)
            and family.n_theta in (0, 1)
            and isinstance(family.link, LogLink)
        )
    )


def _fixed_theta(fd: FittingData) -> float | None:
    """Return a fixed NB size without changing the family instance."""
    family = fd.family
    if isinstance(family, NegativeBinomial) and family.n_theta == 0:
        return float(family.get_theta(transformed=True)[0])
    return None


def _estimated_theta_nb(fd: FittingData) -> bool:
    """Whether this fit uses the staged, EFS-only conditional NB theta loop."""
    family = fd.family
    return (
        isinstance(family, NegativeBinomial)
        and family.n_theta == 1
        and isinstance(family.link, LogLink)
    )


def _result_theta(state: EFSFitState, fd: FittingData) -> float | None:
    """Return only the selected fit's theta, never mutable-family state."""
    if state.log_theta is not None:
        return float(np.asarray(jnp.exp(state.log_theta[0])))
    return _fixed_theta(fd)


def _efs_fit_failure_label(state: EFSFitState) -> str:
    """Expose EFS beta-loop recovery failures without masking theta status."""
    if state.theta_loop_status is not None:
        recovery_labels = {
            _EFS_STATUS_NONFINITE_RECOVERY_FAILED: "nonfinite_recovery_failed",
            _EFS_STATUS_DOMAIN_RECOVERY_FAILED: "domain_recovery_failed",
            _EFS_STATUS_DIVERGENCE_RECOVERY_FAILED: "divergence_recovery_failed",
        }
        label = recovery_labels.get(int(np.asarray(state.theta_loop_status)))
        if label is not None:
            return label
    return "inner_failure" if not state.inner_converged else "invalid_trial"


@dataclass(frozen=True)
class _EFSThetaStart:
    """One immutable R ``gam.fit4`` coefficient/eta initialization choice."""

    beta: jax.Array
    initial_eta: jax.Array
    retained: bool


def _select_efs_theta_start(
    fd: FittingData,
    penalty: jax.Array,
    beta_start: jax.Array,
    beta_null: jax.Array,
    log_theta_start: jax.Array,
    *,
    start_is_absent: bool,
) -> _EFSThetaStart:
    """Match R's retained-start check or its ``mustart`` reset.

    ``gam.fit4`` uses ``link(initialize(y, wt))`` immediately when its start
    is absent. With a supplied start it computes both penalized deviances at
    the incoming theta, discarding a worse vector before the first WLS step.
    Both reset paths retain the explicit null coefficient anchor for the first
    divergence comparison. Do not substitute a projected coefficient
    initializer for that per-row state.
    """
    assert isinstance(fd.family, NegativeBinomial)
    beta_start = jnp.asarray(beta_start)
    beta_null = jnp.asarray(beta_null)
    log_theta_start = jnp.asarray(log_theta_start)
    expected_beta_shape = (fd.n_coef,)
    if (
        beta_start.shape != expected_beta_shape
        or beta_null.shape != expected_beta_shape
    ):
        raise ValueError(
            "Estimated NB EFS beta start and null anchor must match n_coef"
        )
    if log_theta_start.shape != (1,):
        raise ValueError("Estimated NB EFS initial log theta must have shape (1,)")
    if not bool(
        np.all(np.isfinite(np.asarray(beta_start)))
        and np.all(np.isfinite(np.asarray(beta_null)))
        and np.all(np.isfinite(np.asarray(log_theta_start)))
        and np.all(np.exp(np.asarray(log_theta_start)) > 0.0)
    ):
        raise ValueError("Estimated NB EFS beta starts and log theta must be finite")
    # Validate all host-visible inputs before the dense pdev matrix products.
    if not bool(np.all(np.isfinite(np.asarray(penalty)))):
        raise ValueError(
            "Estimated NB EFS penalty must be finite before start selection"
        )
    offset = jnp.zeros_like(fd.y) if fd.offset is None else fd.offset
    if not start_is_absent:
        deviance_fn = fd.family.deviance_fn(fd.y, fd.wt)
        eta_start = fd.X @ beta_start + offset
        eta_null = fd.X @ beta_null + offset
        start_pdev = (
            deviance_fn(eta_start, log_theta_start) + beta_start @ penalty @ beta_start
        )
        null_pdev = (
            deviance_fn(eta_null, log_theta_start) + beta_null @ penalty @ beta_null
        )
        if not bool(np.asarray(jnp.isfinite(null_pdev))):
            raise ValueError(
                "Estimated NB EFS null-anchor penalized deviance must be finite"
            )
        reset = bool(np.asarray(~jnp.isfinite(start_pdev) | (start_pdev > null_pdev)))
        if not reset:
            return _EFSThetaStart(beta_start, eta_start, True)
    mustart = fd.family.initialize(np.asarray(fd.y), np.asarray(fd.wt))
    initial_eta = jnp.asarray(fd.family.link.link(mustart), dtype=fd.X.dtype)
    if not bool(np.all(np.isfinite(np.asarray(initial_eta)))):
        raise ValueError("Estimated NB EFS mustart initialization must be finite")
    return _EFSThetaStart(beta_null, initial_eta, False)


def _unknown_scale_family_supported(fd: FittingData) -> bool:
    """Initial regular-family EFS scope, mirroring the validated R route."""
    family = fd.family
    return (not family.scale_known) and (
        (isinstance(family, Gaussian) and isinstance(family.link, IdentityLink))
        or (
            isinstance(family, Gamma)
            and isinstance(family.link, (InverseLink, LogLink))
        )
    )


def _scalar_score(
    fd: FittingData,
    rho: jax.Array,
    pr: PIRLSResult,
    score_phi: jax.Array,
    log_theta: jax.Array | None = None,
) -> jax.Array:
    """Evaluate only the scalar REML criterion; no score derivatives."""
    if fd.family.family_name == "nb" and fd.count_prefix_plan is not None:
        if _estimated_theta_nb(fd):
            if log_theta is None:
                raise ValueError("Estimated NB EFS score requires explicit log theta")
            assert isinstance(fd.family, NegativeBinomial)
            ls_sat = fd.family.saturated_loglik_theta(
                fd.y,
                fd.wt,
                score_phi,
                log_theta,
                max_y=fd.max_y,
                count_indices=fd.count_prefix_plan.indices,
                integer_counts=fd.count_prefix_plan.integer_counts,
            )
        else:
            ls_sat = fd.family.saturated_loglik(
                fd.y,
                fd.wt,
                score_phi,
                max_y=fd.max_y,
                count_indices=fd.count_prefix_plan.indices,
                integer_counts=fd.count_prefix_plan.integer_counts,
            )
    else:
        ls_sat = fd.family.saturated_loglik(fd.y, fd.wt, score_phi, max_y=fd.max_y)
    return _jit_reml_score(
        rho,
        pr.XtWX,
        pr.coefficients,
        pr.deviance,
        ls_sat,
        fd.penalty_structure,
        score_phi,
        fd.total_penalty_null_dim,
        fd.singleton_sp_indices,
        fd.singleton_ranks,
        fd.singleton_eig_constants,
        fd.multi_block_sp_indices,
        fd.multi_block_ranks,
        fd.multi_block_proj_S,
        fd.rank_deficit,
    )


def _fit_state(
    fd: FittingData,
    plan: EFSStatisticsPlan,
    rho: jax.Array,
    beta_start: jax.Array,
    control: EFSControl,
    score_phi: jax.Array | None = None,
    *,
    log_theta_start: jax.Array | None = None,
    beta_old_init: jax.Array | None = None,
    start_is_absent: bool = False,
) -> EFSFitState:
    if score_phi is None:
        score_phi = jnp.array(1.0)
    penalty = penalty_ops.materialize(fd.penalty_structure, rho)
    log_theta: jax.Array | None = None
    theta_status: jax.Array | None = None
    theta_loop_status: jax.Array | None = None
    theta_n_iter: jax.Array | None = None
    stopping_pdev: jax.Array | None = None
    if _estimated_theta_nb(fd):
        if log_theta_start is None or beta_old_init is None:
            raise ValueError(
                "Estimated NB EFS requires explicit log_theta_start and beta_old_init"
            )
        if fd.count_prefix_plan is None:
            raise ValueError("Estimated NB EFS requires count-prefix metadata")
        assert isinstance(fd.family, NegativeBinomial)
        start = _select_efs_theta_start(
            fd,
            penalty,
            beta_start,
            beta_old_init,
            log_theta_start,
            start_is_absent=start_is_absent,
        )
        theta_result = efs_theta_pirls_loop(
            fd.X,
            fd.y,
            start.beta,
            penalty,
            fd.family,
            fd.wt,
            fd.offset,
            log_theta_start,
            fd.count_prefix_plan.indices,
            beta_old_init=beta_old_init,
            initial_eta=start.initial_eta,
            initial_start_retained=start.retained,
            max_y=fd.max_y,
            integer_counts=fd.count_prefix_plan.integer_counts,
            max_iter=control.pirls_max_iter,
            tol=control.pirls_tolerance,
        )
        pr = theta_result.pirls_result
        log_theta = theta_result.log_theta
        theta_status = theta_result.theta_status
        theta_loop_status = theta_result.status
        theta_n_iter = theta_result.theta_n_iter
        stopping_pdev = theta_result.stopping_penalized_deviance
    else:
        pr = pirls_loop(
            fd.X,
            fd.y,
            beta_start,
            penalty,
            fd.family,
            fd.wt,
            fd.offset,
            max_iter=control.pirls_max_iter,
            tol=control.pirls_tolerance,
            extended_observed=(
                isinstance(fd.family, NegativeBinomial) and fd.family.n_theta == 0
            ),
        )
    statistics = _jit_efs_statistics(plan, pr.coefficients, pr.L_fisher, rho)
    score = _scalar_score(fd, rho, pr, score_phi, log_theta)
    edf = estimate_edf(pr.XtWX_fisher, pr.L_fisher)
    update_phi = (
        jnp.array(1.0)
        if fd.family.scale_known
        else fletcher_scale(fd.y, pr.mu, fd.wt, fd.family, edf)
    )
    finite = bool(
        np.asarray(jnp.isfinite(score) & jnp.all(jnp.isfinite(pr.coefficients)))
    )
    inner = bool(np.asarray(pr.converged))
    theta_ok = (
        True
        if theta_status is None or log_theta is None
        else bool(
            np.asarray(
                (theta_status == 0)
                & jnp.all(jnp.isfinite(log_theta))
                & jnp.all(jnp.isfinite(jnp.exp(log_theta)))
                & jnp.all(jnp.exp(log_theta) > 0)
            )
        )
    )
    valid = (
        finite
        and inner
        and theta_ok
        and bool(np.asarray(statistics.input_valid))
        and bool(np.asarray(jnp.isfinite(score_phi) & (score_phi > 0)))
        and bool(np.asarray(jnp.isfinite(update_phi) & (update_phi > 0)))
    )
    return EFSFitState(
        rho,
        pr,
        score,
        edf,
        statistics,
        None,
        valid,
        inner,
        score_phi,
        update_phi,
        update_phi,
        update_phi,
        log_theta,
        theta_status,
        theta_loop_status,
        theta_n_iter,
        stopping_pdev,
    )


def dense_efs_known_scale(
    fitting_data: FittingData,
    *,
    initial_log_lambda: jax.Array | None = None,
    initial_log_theta: jax.Array | None = None,
    beta_init: jax.Array | None = None,
    beta_old_init: jax.Array | None = None,
    control: EFSControl = DEFAULT_EFS_CONTROL,
) -> EFSResult:
    """Run the pinned EFS accepted/trial policy for known-scale families.

    The routine is opt-in and internal; it never dispatches from ``GAM.fit``.
    A caller may pass a matched retained beta/null anchor for R-parity tests.
    Estimated NB otherwise starts at R EFS's zero null anchor with a separate
    per-row ``link(mustart)`` state; it never repurposes the ordinary projected
    ``FittingData.beta_init`` as a ``gam.fit4`` initializer.
    """
    if not _known_scale_family_supported(fitting_data):
        raise NotImplementedError(
            "Dense EFS currently supports only known-scale Poisson/log, "
            "binomial/logit, and NB/log."
        )
    if fitting_data.n_penalties == 0:
        raise ValueError("EFS bypasses models without estimated penalties")
    for start in range(0, fitting_data.n_obs, 8192):
        prior_weights = np.asarray(fitting_data.wt[start : start + 8192])
        if not np.all(np.isfinite(prior_weights)) or np.any(
            (prior_weights < _W_MIN) | (prior_weights > _W_MAX)
        ):
            raise ValueError(
                "EFS known-scale path requires finite positive prior weights "
                "inside PIRLS clipping bounds"
            )
    if fitting_data.rank_deficit:
        raise ValueError(
            "EFS known-scale path requires an identifiable penalized system"
        )
    plan = prepare_efs_statistics(fitting_data)
    rho0 = (
        fitting_data.log_lambda_init
        if initial_log_lambda is None
        else initial_log_lambda
    )
    estimated_theta = _estimated_theta_nb(fitting_data)
    initial_start_is_absent = estimated_theta and beta_init is None
    if initial_log_lambda is None and isinstance(fitting_data.family, NegativeBinomial):
        raise ValueError(
            "NB EFS requires efs_initial_log_lambda(setup, family); "
            "its extended-family initial.spg weights differ from regular fitting"
        )
    beta0 = fitting_data.beta_init if beta_init is None else beta_init
    if beta0 is None:
        beta0 = jnp.zeros((fitting_data.n_coef,), dtype=fitting_data.X.dtype)
    theta0: jax.Array | None = None
    if estimated_theta:
        assert isinstance(fitting_data.family, NegativeBinomial)
        # efsudr reaches gam.fit4 without get.null.coef in this pinned route,
        # so the actual default null anchor is exactly the zero vector.
        beta0 = (
            jnp.zeros((fitting_data.n_coef,), dtype=fitting_data.X.dtype)
            if beta_init is None
            else jnp.asarray(beta_init)
        )
        beta_old_init = (
            jnp.zeros_like(beta0)
            if beta_old_init is None
            else jnp.asarray(beta_old_init)
        )
        theta0 = (
            jnp.asarray(fitting_data.family.get_theta(transformed=False))
            if initial_log_theta is None
            else jnp.asarray(initial_log_theta)
        )
        if theta0.shape != (1,) or not bool(
            np.all(np.isfinite(np.asarray(theta0)))
            and np.all(np.isfinite(np.exp(np.asarray(theta0))))
            and np.all(np.exp(np.asarray(theta0)) > 0.0)
        ):
            raise ValueError(
                "Estimated NB EFS initial log theta must be finite with shape (1,)"
            )
        if beta_old_init.shape != beta0.shape:
            raise ValueError("Estimated NB EFS beta_old_init must align with beta_init")
    rho0 = jnp.asarray(rho0) + 2.5
    if rho0.shape != (fitting_data.n_penalties,) or not bool(
        np.all(np.isfinite(np.asarray(rho0)))
    ):
        raise ValueError(
            "EFS initial log smoothing parameters must be finite and match penalties"
        )
    if estimated_theta:
        accepted = _fit_state(
            fitting_data,
            plan,
            rho0,
            beta0,
            control,
            log_theta_start=theta0,
            beta_old_init=beta_old_init,
            start_is_absent=initial_start_is_absent,
        )
    else:
        # Keep the established known-scale call signature byte-for-byte for
        # Poisson, binomial, and fixed-theta NB scripted/default paths.
        accepted = _fit_state(fitting_data, plan, rho0, beta0, control)
    if not accepted.valid:
        return EFSResult(
            rho0,
            jnp.exp(rho0),
            False,
            0,
            accepted.score,
            accepted.edf,
            jnp.array(1.0),
            accepted.pirls_result,
            (
                _efs_fit_failure_label(accepted)
                if accepted.theta_status is not None
                else "inner_failure"
                if not accepted.inner_converged
                else "invalid_initial"
            ),
            _result_theta(accepted, fitting_data),
        )

    multiplier = 1.0
    history: deque[float] = deque(maxlen=max(4, control.history_limit))
    old_deviance: float | None = None
    stop = "iteration_limit"
    update_residual: jax.Array | None = None

    def refit(rho: jax.Array, old: EFSFitState) -> EFSFitState:
        """Refit from the immutable old accepted beta/theta state only."""
        if estimated_theta:
            assert old.log_theta is not None
            assert beta_old_init is not None
            return _fit_state(
                fitting_data,
                plan,
                rho,
                old.pirls_result.coefficients,
                control,
                log_theta_start=old.log_theta,
                beta_old_init=beta_old_init,
            )
        return _fit_state(
            fitting_data,
            plan,
            rho,
            old.pirls_result.coefficients,
            control,
        )

    for iteration in range(1, control.outer_limit + 1):
        raw = efs_raw_update(
            accepted.log_lambda,
            accepted.statistics,
            jnp.array(1.0),
            jnp.asarray(multiplier),
            jnp.asarray(control.log_lambda_max),
        )
        update_residual = raw.log_smoothing_trial - accepted.log_lambda
        if not bool(np.asarray(raw.finite_positive)):
            stop = "invalid_update"
            break
        old = accepted
        original_max_step = float(np.max(np.abs(np.asarray(update_residual))))
        candidate = refit(raw.log_smoothing_trial, old)
        if not candidate.valid:
            stop = _efs_fit_failure_label(candidate)
            break
        if float(np.asarray(candidate.score)) <= float(np.asarray(old.score)):
            if original_max_step < 0.05:
                extension_rho = jnp.minimum(
                    old.log_lambda + jnp.log(raw.ratio) * (multiplier * 2.0),
                    control.log_lambda_max,
                )
                extension = refit(extension_rho, old)
                if extension.valid and float(np.asarray(extension.score)) < float(
                    np.asarray(candidate.score)
                ):
                    accepted = extension
                    multiplier *= 2.0
                else:
                    accepted = candidate
            else:
                accepted = candidate
        else:
            while (
                float(np.asarray(candidate.score)) > float(np.asarray(old.score))
                and multiplier > 1.0
            ):
                multiplier /= 2.0
                rho = jnp.minimum(
                    old.log_lambda + jnp.log(raw.ratio) * multiplier,
                    control.log_lambda_max,
                )
                candidate = refit(rho, old)
                if not candidate.valid:
                    stop = _efs_fit_failure_label(candidate)
                    break
            if stop != "iteration_limit":
                break
            accepted = candidate
            multiplier = max(multiplier, 1.0)
        history.append(float(np.asarray(accepted.score)))
        if (
            iteration > 3
            and original_max_step < 0.05
            and max(abs(np.diff(tuple(history)[-4:]))) < control.score_tolerance
        ):
            stop = "score_window"
            break
        dev = float(np.asarray(accepted.pirls_result.deviance))
        if old_deviance is not None and abs(
            old_deviance - dev
        ) < 100.0 * control.pirls_tolerance * abs(dev):
            stop = "deviance_change"
            break
        old_deviance = dev
    else:
        iteration = control.outer_limit
    # efsudr labels an iteration-200 return as an iteration limit even when
    # one of its stop predicates first becomes true on that final iteration.
    if iteration == control.outer_limit and stop in {"score_window", "deviance_change"}:
        stop = "iteration_limit"
    converged = stop in {"score_window", "deviance_change"}
    label = "iteration limit reached" if stop == "iteration_limit" else stop
    return EFSResult(
        accepted.log_lambda,
        jnp.exp(accepted.log_lambda),
        converged,
        iteration,
        accepted.score,
        accepted.edf,
        jnp.array(1.0),
        accepted.pirls_result,
        label,
        _result_theta(accepted, fitting_data),
        update_residual,
        tuple(history)[-control.history_limit :],
        multiplier,
    )


def dense_efs_unknown_scale(
    fitting_data: FittingData,
    *,
    initial_log_lambda: jax.Array | None = None,
    initial_log_scale: jax.Array | None = None,
    beta_init: jax.Array | None = None,
    control: EFSControl = DEFAULT_EFS_CONTROL,
) -> EFSResult:
    """Run regular unknown-scale Gaussian/Gamma EFS without Newton in phi.

    The scale passed to each score is an explicit input parameter.  Its
    replacement is the current fit's Fletcher estimate, as in pinned
    ``efsudr``.  Alternative extension/contraction fits are intentionally
    formed from the old accepted beta *and old accepted input parameter
    state*, never from a rejected trial's reported scale.  Pinned R has one
    deliberate exception: after a winning extension it carries the candidate
    scale into the next score while retaining the extension scale for its next
    EFS ratio; ``carried_phi`` records that distinct full-parameter state.
    """
    if not _unknown_scale_family_supported(fitting_data):
        raise NotImplementedError(
            "Dense unknown-scale EFS currently supports Gaussian/identity and "
            "Gamma/inverse or Gamma/log."
        )
    if fitting_data.n_penalties == 0:
        raise ValueError("EFS bypasses models without estimated penalties")
    for start in range(0, fitting_data.n_obs, 8192):
        prior_weights = np.asarray(fitting_data.wt[start : start + 8192])
        if not np.all(np.isfinite(prior_weights)) or np.any(
            (prior_weights < _W_MIN) | (prior_weights > _W_MAX)
        ):
            raise ValueError(
                "EFS unknown-scale path requires finite positive prior weights "
                "inside PIRLS clipping bounds"
            )
    if fitting_data.rank_deficit:
        raise ValueError(
            "EFS unknown-scale path requires an identifiable penalized system"
        )
    rho0 = (
        fitting_data.log_lambda_init
        if initial_log_lambda is None
        else initial_log_lambda
    )
    rho0 = jnp.asarray(rho0) + 2.5
    if rho0.shape != (fitting_data.n_penalties,) or not bool(
        np.all(np.isfinite(np.asarray(rho0)))
    ):
        raise ValueError(
            "EFS initial log smoothing parameters must be finite and match penalties"
        )
    if initial_log_scale is None:
        initial_log_scale = _efs_initial_log_scale_from_arrays(
            fitting_data.y, fitting_data.wt, fitting_data.family
        )
    score_phi0 = jnp.exp(jnp.asarray(initial_log_scale))
    if not bool(np.asarray(jnp.isfinite(score_phi0) & (score_phi0 > 0))):
        raise ValueError("EFS initial unknown scale must be finite and positive")
    beta0 = fitting_data.beta_init if beta_init is None else beta_init
    if beta0 is None:
        beta0 = jnp.zeros((fitting_data.n_coef,), dtype=fitting_data.X.dtype)

    plan = prepare_efs_statistics(fitting_data)
    accepted = _fit_state(fitting_data, plan, rho0, beta0, control, score_phi0)
    if not accepted.valid:
        return EFSResult(
            rho0,
            jnp.exp(rho0),
            False,
            0,
            accepted.score,
            accepted.edf,
            accepted.reported_phi,
            accepted.pirls_result,
            "inner_failure" if not accepted.inner_converged else "invalid_initial",
            score_phi=accepted.score_phi,
            update_phi=accepted.update_phi,
            reported_phi=accepted.reported_phi,
            carried_phi=accepted.carried_phi,
        )

    multiplier = 1.0
    history: deque[float] = deque(maxlen=max(4, control.history_limit))
    score_phi_history: deque[float] = deque(maxlen=max(4, control.history_limit))
    old_deviance: float | None = None
    stop = "iteration_limit"
    update_residual: jax.Array | None = None
    for iteration in range(1, control.outer_limit + 1):
        # ``update_phi`` is the new Fletcher scale from the *accepted* fit,
        # while ``score_phi`` remains the scale at which that cached score was
        # evaluated.  The ratio specifically uses the former.
        assert accepted.update_phi is not None
        raw = efs_raw_update(
            accepted.log_lambda,
            accepted.statistics,
            accepted.update_phi,
            jnp.asarray(multiplier),
            jnp.asarray(control.log_lambda_max),
        )
        update_residual = raw.log_smoothing_trial - accepted.log_lambda
        if not bool(np.asarray(raw.finite_positive)):
            stop = "invalid_update"
            break
        old = accepted
        # In R this is ``max(abs(lsp1-lsp))`` before the trial returns its
        # scale.  The full proposal's phi component is unchanged at this
        # point, so retaining rho's maximum is the exact nonzero component.
        original_max_step = float(np.max(np.abs(np.asarray(update_residual))))
        assert old.carried_phi is not None
        candidate = _fit_state(
            fitting_data,
            plan,
            raw.log_smoothing_trial,
            old.pirls_result.coefficients,
            control,
            old.carried_phi,
        )
        if not candidate.valid:
            stop = "inner_failure" if not candidate.inner_converged else "invalid_trial"
            break
        if float(np.asarray(candidate.score)) <= float(np.asarray(old.score)):
            if original_max_step < 0.05:
                extension_rho = jnp.minimum(
                    old.log_lambda + jnp.log(raw.ratio) * (multiplier * 2.0),
                    control.log_lambda_max,
                )
                extension = _fit_state(
                    fitting_data,
                    plan,
                    extension_rho,
                    old.pirls_result.coefficients,
                    control,
                    old.carried_phi,
                )
                if extension.valid and float(np.asarray(extension.score)) < float(
                    np.asarray(candidate.score)
                ):
                    # Pinned efsudr writes lsp2's scale from ``fit`` (the
                    # first candidate), not fit2 (the extension), before it
                    # swaps ``fit <- fit2``.  Therefore next score uses the
                    # candidate scale while the next ratio uses extension's
                    # Fletcher scale retained in ``accepted.update_phi``.
                    assert candidate.update_phi is not None
                    accepted = replace(extension, carried_phi=candidate.update_phi)
                    multiplier *= 2.0
                else:
                    accepted = candidate
            else:
                accepted = candidate
        else:
            while (
                float(np.asarray(candidate.score)) > float(np.asarray(old.score))
                and multiplier > 1.0
            ):
                multiplier /= 2.0
                rho = jnp.minimum(
                    old.log_lambda + jnp.log(raw.ratio) * multiplier,
                    control.log_lambda_max,
                )
                candidate = _fit_state(
                    fitting_data,
                    plan,
                    rho,
                    old.pirls_result.coefficients,
                    control,
                    old.carried_phi,
                )
                if not candidate.valid:
                    stop = (
                        "inner_failure"
                        if not candidate.inner_converged
                        else "invalid_trial"
                    )
                    break
            if stop != "iteration_limit":
                break
            accepted = candidate
            multiplier = max(multiplier, 1.0)
        history.append(float(np.asarray(accepted.score)))
        assert accepted.score_phi is not None
        score_phi_history.append(float(np.asarray(accepted.score_phi)))
        if (
            iteration > 3
            and original_max_step < 0.05
            and max(abs(np.diff(tuple(history)[-4:]))) < control.score_tolerance
        ):
            stop = "score_window"
            break
        dev = float(np.asarray(accepted.pirls_result.deviance))
        if old_deviance is not None and abs(
            old_deviance - dev
        ) < 100.0 * control.pirls_tolerance * abs(dev):
            stop = "deviance_change"
            break
        old_deviance = dev
    else:
        iteration = control.outer_limit
    if iteration == control.outer_limit and stop in {"score_window", "deviance_change"}:
        stop = "iteration_limit"
    converged = stop in {"score_window", "deviance_change"}
    label = "iteration limit reached" if stop == "iteration_limit" else stop
    return EFSResult(
        accepted.log_lambda,
        jnp.exp(accepted.log_lambda),
        converged,
        iteration,
        accepted.score,
        accepted.edf,
        accepted.reported_phi,
        accepted.pirls_result,
        label,
        None,
        update_residual,
        tuple(history)[-control.history_limit :],
        multiplier,
        accepted.score_phi,
        accepted.update_phi,
        accepted.reported_phi,
        accepted.carried_phi,
        tuple(score_phi_history)[-control.history_limit :],
    )
