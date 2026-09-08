"""Dense, known-scale extended Fellner--Schall execution adapter.

This is intentionally an internal controller.  It supports the first EFS
regime only (Poisson/log and binomial/logit), and keeps accepted and trial
states separate so a rejected refit can never leak into the next iteration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

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
from jaxgam.fitting.pirls import _W_MAX, _W_MIN, PIRLSResult, pirls_loop
from jaxgam.fitting.reml import estimate_edf, fletcher_scale, reml_criterion
from jaxgam.links.links import IdentityLink, InverseLink, LogitLink, LogLink


def efs_initial_log_lambda(setup, family) -> jax.Array:
    """Prepare mgcv ``initial.spg`` regular-family start only for EFS.

    Regular fitting never incurs this initialized-mean/working-weight work.
    The shared initial-sp routine keeps its bounded original-coordinate
    weighted-crossproduct reduction.
    """
    structure = setup.penalties
    if structure is None or structure.n_penalties == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    ldxx = np.zeros(setup.X.shape[1], dtype=np.float64)
    batch_rows = 8192
    for start in range(0, setup.X.shape[0], batch_rows):
        stop = min(start + batch_rows, setup.X.shape[0])
        y = setup.y[start:stop]
        prior = setup.weights[start:stop]
        mu = np.asarray(family.initialize(y, prior), dtype=np.float64)
        eta = np.asarray(family.link.link(mu), dtype=np.float64)
        mu_eta = np.asarray(family.link.mu_eta(eta), dtype=np.float64)
        variance = np.asarray(family.variance(mu), dtype=np.float64)
        working = prior * mu_eta**2 / variance
        if not np.all(np.isfinite(working)) or np.any(working <= 0):
            raise ValueError(
                "EFS initial.spg working weights must be finite and positive"
            )
        weighted_X = np.sqrt(working)[:, None] * setup.X[start:stop]
        ldxx += np.sum(weighted_X * weighted_X, axis=0)
    return jnp.asarray(
        FittingData._initial_sp_from_crossproduct_diag(setup.X, structure, ldxx)
    )


def efs_initial_log_scale(setup, family) -> jax.Array:
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


def _efs_initial_log_scale_from_arrays(y, wt, family) -> jax.Array:
    """Array-level implementation shared by setup and dense fit adapters."""
    y = np.asarray(y, dtype=np.float64)
    wt = np.asarray(wt, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or wt.shape != y.shape:
        raise ValueError(
            "EFS initial scale requires nonempty aligned response and weights"
        )
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(wt)) or np.any(wt <= 0):
        raise ValueError("EFS initial scale requires finite strictly positive weights")
    # Retain family response validation before using the same response mean as
    # get.null.coef().  ``initialize`` may perform support checks.
    family.initialize(y, wt)
    mu = np.full_like(y, np.mean(y))
    null_scale = float(np.asarray(family.dev_resids(y, mu, wt))) / y.size
    incoming_phi = null_scale / 10.0
    if not np.isfinite(incoming_phi) or incoming_phi <= 0:
        raise ValueError(
            "EFS initial unknown scale null.scale / 10 must be finite and positive"
        )
    return jnp.asarray(np.log(incoming_phi), dtype=jnp.float64)


@dataclass(frozen=True)
class EFSControl:
    """Pinned ``efsudr`` controls for the known-scale dense path."""

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
    )


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
    fd: FittingData, rho: jax.Array, pr: PIRLSResult, score_phi: jax.Array
) -> jax.Array:
    """Evaluate only the scalar REML criterion; no score derivatives."""
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
) -> EFSFitState:
    if score_phi is None:
        score_phi = jnp.array(1.0)
    penalty = penalty_ops.materialize(fd.penalty_structure, rho)
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
    )
    statistics = _jit_efs_statistics(plan, pr.coefficients, pr.L_fisher, rho)
    score = _scalar_score(fd, rho, pr, score_phi)
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
    valid = (
        finite
        and inner
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
    )


def dense_efs_known_scale(
    fitting_data: FittingData,
    *,
    initial_log_lambda: jax.Array | None = None,
    beta_init: jax.Array | None = None,
    control: EFSControl = DEFAULT_EFS_CONTROL,
) -> EFSResult:
    """Run the pinned EFS accepted/trial policy for known-scale families.

    The routine is opt-in and internal; it never dispatches from ``GAM.fit``.
    A caller may pass a matched initial state for R-parity tests.  In ordinary
    use the existing fitting-boundary start is used, then EFS applies its
    required one-time +2.5 smoothing shift.
    """
    if not _known_scale_family_supported(fitting_data):
        raise NotImplementedError(
            "Dense EFS currently supports only known-scale Poisson/log and "
            "binomial/logit."
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
    beta0 = fitting_data.beta_init if beta_init is None else beta_init
    if beta0 is None:
        beta0 = jnp.zeros((fitting_data.n_coef,), dtype=fitting_data.X.dtype)
    rho0 = jnp.asarray(rho0) + 2.5
    if rho0.shape != (fitting_data.n_penalties,) or not bool(
        np.all(np.isfinite(np.asarray(rho0)))
    ):
        raise ValueError(
            "EFS initial log smoothing parameters must be finite and match penalties"
        )
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
            "inner_failure" if not accepted.inner_converged else "invalid_initial",
            None,
        )

    multiplier = 1.0
    history: deque[float] = deque(maxlen=max(4, control.history_limit))
    old_deviance: float | None = None
    stop = "iteration_limit"
    update_residual: jax.Array | None = None
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
        candidate = _fit_state(
            fitting_data,
            plan,
            raw.log_smoothing_trial,
            old.pirls_result.coefficients,
            control,
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
                )
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
                candidate = _fit_state(
                    fitting_data, plan, rho, old.pirls_result.coefficients, control
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
        None,
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
    state*, never from a rejected trial's reported scale.
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
        assert old.update_phi is not None
        candidate = _fit_state(
            fitting_data,
            plan,
            raw.log_smoothing_trial,
            old.pirls_result.coefficients,
            control,
            old.update_phi,
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
                    old.update_phi,
                )
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
                candidate = _fit_state(
                    fitting_data,
                    plan,
                    rho,
                    old.pirls_result.coefficients,
                    control,
                    old.update_phi,
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
        tuple(score_phi_history)[-control.history_limit :],
    )
