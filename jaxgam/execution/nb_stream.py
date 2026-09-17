"""Bounded fixed/trial-theta NB coefficient fitting from gam.fit4.

This internal controller optionally estimates theta conditionally inside PIRLS;
it does not enable public streaming. Source iteration stays on the host;
explicit-parameter batch mathematics is
compiled independently. Positive-observed recovery is separate from Fisher
reporting and never changes the accepted family object.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import jax
import numpy as np

from jaxgam.execution.nb_theta_stream import (
    NBStreamThetaResult,
    NBThetaStreamControl,
    conditional_theta_stream,
)
from jaxgam.execution.null_coefficient import project_null_coefficients
from jaxgam.execution.qr import PositiveQRState, qr_update, solve_augmented_qr
from jaxgam.execution.regular_stream import (
    EtaFactory,
    RegularStreamWorkspace,
    _tag_absolute,
    preflight_regular_stream_workspace,
)
from jaxgam.execution.signed_qr import SignedQRState, signed_qr_update, solve_signed_qr
from jaxgam.execution.stream import StreamPIRLSControl, _source_batches
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.fitting.family_execution import (
    FamilyExecutionLineage,
    FamilyExecutionParameters,
    batch_saturated_loglikelihood,
)
from jaxgam.fitting.nb_stream_kernels import (
    merge_nb_working_summaries,
    nb_positive_observed_retry,
    nb_selected_working_rows,
    nb_working_batch,
    nb_working_summary,
)
from jaxgam.fitting.reml import reml_criterion_from_penalized_deviance
from jaxgam.fitting.signed_qr import SignedQRCoefficientFactor
from jaxgam.fitting.state import StreamFitState
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.fitting_prepare import qr_penalty_roots
from jaxgam.links.links import IdentityLink, LogLink, SqrtLink

_working = partial(jax.jit, static_argnames=("link",))(nb_working_batch)
_retry = jax.jit(nb_positive_observed_retry)
_rows = jax.jit(nb_selected_working_rows)
_summary = jax.jit(nb_working_summary)
_merge = jax.jit(merge_nb_working_summaries)


def _link_name(family: NegativeBinomial) -> str:
    if not isinstance(family, NegativeBinomial):
        raise TypeError("NB coefficient controller requires NegativeBinomial")
    for cls, name in ((LogLink, "log"), (IdentityLink, "identity"), (SqrtLink, "sqrt")):
        if type(family.link) is cls:
            return name
    raise NotImplementedError("NB coefficient streaming supports log, identity, sqrt")


@dataclass(frozen=True)
class NBWorkingScan:
    """Small observed solve plus separately accumulated Fisher information."""

    observed: SignedQRState
    observed_G: np.ndarray
    observed_rhs: np.ndarray
    fisher: PositiveQRState
    fisher_G: np.ndarray
    good_count: int
    informative_count: int
    batches_scanned: int
    deviance: float
    used_direct_response: bool
    gradient_G: np.ndarray
    gradient_rhs: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "observed_G",
            "observed_rhs",
            "fisher_G",
            "gradient_G",
            "gradient_rhs",
        ):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def nb_working_scan(
    stream: StreamDesign,
    family: NegativeBinomial,
    lineage: FamilyExecutionLineage,
    parameters: FamilyExecutionParameters,
    control: StreamPIRLSControl,
    eta_for_batch: EtaFactory,
    *,
    positive_observed: bool = False,
    score_system: bool = False,
    device: jax.Device | None = None,
) -> NBWorkingScan:
    """Reduce one source scan without retaining working rows or theta caches.

    A later batch requiring direct Wz selects the accumulated direct RHS
    globally through SignedQRState. The source good-row gate counts finite
    zero-curvature rows; coefficient/penalty rank is checked by the solver.
    """
    link = _link_name(family)
    p = stream.prepared.n_coef
    observed = fisher = summary = None
    G, F, rhs = np.zeros((p, p)), np.zeros((p, p)), np.zeros(p)
    gradient_G, gradient_rhs = np.zeros((p, p)), np.zeros(p)
    batches = 0
    for batch, y, weight, offset, valid in _source_batches(
        stream, family, lineage, control.batch_rows
    ):
        if len(y) > min(control.batch_rows, stream.prepared.n_obs):
            raise ValueError("NB source exceeds prospective batch cap")
        X = (
            stream.prepared.evaluate_fitting_batch(batch)
            if len(y)
            else np.empty((0, p))
        )
        if X.shape != (len(y), p) or not np.all(np.isfinite(X)):
            raise ValueError("finite matching NB fitting design required")
        eta = np.asarray(eta_for_batch(batch, X), float)
        if eta.shape != y.shape:
            raise ValueError("NB predictor must return one value per source row")
        lineage.validate(stream.prepared, family)
        working = _working(
            *[
                jax.device_put(value, device)
                for value in (eta, y, weight, offset, valid)
            ],
            parameters,
            link=link,
        )
        if positive_observed:
            working = _retry(working)
        batch_summary = _summary(working)
        summary = batch_summary if summary is None else _merge(summary, batch_summary)
        if not bool(np.asarray(working.domain_ok)):
            raise ValueError("NB working predictor or theta is outside its domain")
        direct = score_system or bool(np.asarray(working.requires_direct_response))
        w, z, wz = map(np.asarray, _rows(working, use_weighted_response=direct))
        observed = signed_qr_update(
            observed, X, w, z, weighted_response=wz, use_weighted_response=direct
        )
        # Supplying Wz on every batch preserves the direct RHS when a later
        # batch globally requests it, without replaying earlier QR rows.
        G += X.T @ (w[:, None] * X)
        rhs += X.T @ wz
        # gam.fit4 recomputes normal finite-z good rows before its stopping
        # gradient. This mask is distinct from a direct-Wz coefficient solve.
        normal = np.asarray(working.normal_rows)
        wn = np.where(normal, np.asarray(working.observed_weight), 0.0)
        wzn = np.where(normal, np.asarray(working.weighted_response), 0.0)
        gradient_G += X.T @ (wn[:, None] * X)
        gradient_rhs += X.T @ wzn
        wf = np.asarray(working.fisher_weight)
        if not np.all(np.isfinite(wf)) or np.any(wf < 0):
            raise FloatingPointError("NB Fisher information is not finite positive")
        fisher = qr_update(fisher, np.sqrt(wf)[:, None] * X, np.zeros(len(y)), n_coef=p)
        F += X.T @ (wf[:, None] * X)
        batches += 1
    if summary is None or observed is None or fisher is None:
        raise ValueError("NB source contains no batches")
    direct = score_system or bool(np.asarray(summary.requires_direct_response))
    good = int(
        np.asarray(summary.direct_good_count if direct else summary.normal_good_count)
    )
    informative = int(
        np.asarray(
            summary.direct_informative_count
            if direct
            else summary.normal_informative_count
        )
    )
    if good == 0:
        raise ValueError("no good NB data in iteration")
    if not all(
        np.all(np.isfinite(value)) for value in (G, F, rhs, gradient_G, gradient_rhs)
    ):
        raise FloatingPointError("NB working statistics overflow")
    return NBWorkingScan(
        observed,
        G,
        rhs,
        fisher,
        F,
        good,
        informative,
        batches,
        float(np.asarray(summary.deviance)),
        direct,
        gradient_G,
        gradient_rhs,
    )


@dataclass(frozen=True)
class NBStreamResult:
    """Compact coefficient result with exact immutable generating theta."""

    state: StreamFitState
    positive_observed_recoveries: int
    accepted_penalized_history: tuple[float, ...]
    workspace: RegularStreamWorkspace
    log_theta: tuple[float, ...]
    integer_counts: bool
    max_count: float
    gdi_penalty: float
    score_penalized_deviance: float
    reml_score: float
    source_deviance: float
    initial_start_retained: bool = False
    theta_n_iter: int = 0
    theta_status: int = 0
    theta_source_scans: int = 0
    stopping_penalized_deviance: float | None = None
    final_theta_deviance: float | None = None
    theta_history: tuple[float, ...] = ()
    last_theta_result: NBStreamThetaResult | None = None
    source_deviance_log_theta: tuple[float, ...] = ()


def fit_nb_streamed_pirls(
    stream: StreamDesign,
    family: NegativeBinomial,
    log_lambda: np.ndarray,
    *,
    maximum_bytes: int,
    parameters: FamilyExecutionParameters | None = None,
    control: StreamPIRLSControl | None = None,
    device: jax.Device | None = None,
    beta_start: np.ndarray | None = None,
    beta_old_init: np.ndarray | None = None,
    start_is_absent: bool | None = None,
    estimate_theta: bool = False,
    theta_control: NBThetaStreamControl | None = None,
) -> NBStreamResult:
    """Fit coefficients at supplied fixed/trial theta, leaving family untouched.

    Estimated-family callers explicitly supply trial parameters. With
    estimate_theta=True, conditional theta Newton runs inside each accepted
    coefficient iteration. This is an internal EFS provider component, not a
    public estimated-theta route or exact REML joint-theta driver.
    """
    control = StreamPIRLSControl(solver_policy="qr") if control is None else control
    if control.solver_policy != "qr":
        raise ValueError("NB source controller requires solver_policy='qr'")
    _link_name(family)
    if not isinstance(estimate_theta, bool) or (
        start_is_absent is not None and not isinstance(start_is_absent, bool)
    ):
        raise ValueError("NB start and conditional theta flags must be bool")
    if estimate_theta and family.n_theta != 1:
        raise ValueError("conditional theta requires an estimated NB family")
    if theta_control is not None and not isinstance(
        theta_control, NBThetaStreamControl
    ):
        raise TypeError("NB theta_control must be NBThetaStreamControl")
    theta_control = NBThetaStreamControl() if theta_control is None else theta_control
    ledger = preflight_regular_stream_workspace(stream, control, maximum_bytes)
    if estimate_theta:
        # Stage one: count capacity is unknown until the bounded summary pass.
        # Charge batch AD arrays and both scalar histories before that pass.
        ledger = replace(
            ledger,
            device_visible_bytes=ledger.device_visible_bytes
            + 8 * 64 * ledger.batch_rows,
            history_bytes=ledger.history_bytes
            + 128 * (theta_control.max_iter + control.max_iter + 2),
        )
        if ledger.required_bytes > maximum_bytes:
            raise MemoryError(
                "NB conditional theta known workspace exceeds maximum_bytes"
            )
    lineage = FamilyExecutionLineage.from_prepared(stream.prepared, family)
    if parameters is None:
        if family.n_theta:
            raise ValueError(
                "estimated NB coefficient subproblem requires explicit trial theta"
            )
        parameters = FamilyExecutionParameters.from_snapshot(lineage.parameters)
    theta = np.array(parameters.log_theta, dtype=float, copy=True)
    if (
        theta.shape != (1,)
        or not np.all(np.isfinite(theta))
        or not np.isfinite(np.exp(theta[0]))
        or np.exp(theta[0]) <= 0
    ):
        raise ValueError("NB trial log_theta must have one finite positive exponent")
    parameters = jax.device_put(
        FamilyExecutionParameters(log_theta=theta.copy()), device
    )
    prepared = stream.prepared
    assert prepared.fitting is not None
    p = prepared.n_coef
    rho = np.asarray(log_lambda, float)
    if rho.shape != prepared.fitting.log_lambda_init.shape or not np.all(
        np.isfinite(rho)
    ):
        raise ValueError("finite matching NB smoothing parameters required")
    layout = qr_penalty_roots(prepared.fitting.penalty_structure)
    balanced = tuple((slice(root.start, root.stop), root.root) for root in layout)
    actual = tuple(
        (slice(root.start, root.stop), root.root * np.exp(0.5 * rho[root.sp_index]))
        for root in layout
    )
    if not all(np.all(np.isfinite(root)) for _, root in actual):
        raise FloatingPointError("NB actual penalty roots overflow")
    scans = batches_scanned = 0

    def penalty(beta):
        result = np.zeros(p)
        for where, root in actual:
            result[where] += root.T @ (root @ beta[where])
        return result

    sum_y = 0.0
    count = 0
    summary = None
    null_projection = None
    for batch, y, weight, _offset, valid in _source_batches(
        stream, family, lineage, control.batch_rows
    ):
        if len(y) > ledger.batch_rows:
            raise ValueError("NB initial source exceeds prospective batch cap")
        X = prepared.evaluate_fitting_batch(batch) if len(y) else np.empty((0, p))
        if (
            prepared.predict_spec.has_intercept
            and len(y)
            and not np.all(X[:, 0] == 1.0)
        ):
            raise NotImplementedError(
                "NB null anchor requires literal intercept coordinate"
            )
        if not prepared.predict_spec.has_intercept:
            public_X = prepared.evaluate_batch(batch) if len(y) else np.empty((0, p))
            null_projection = qr_update(
                null_projection, public_X, np.ones(len(y)), n_coef=p
            )
        initial = family.initial_working_state_cpu(y, weight, valid)
        if not initial.input_ok or not initial.domain_ok:
            raise ValueError("NB initial mustart predictor is invalid")
        if len(y):
            item = family.execution_summary_from_batch(y, weight, valid)
            summary = (
                item
                if summary is None
                else family.merge_execution_summaries(summary, item)
            )
        sum_y += float(np.sum(y))
        count += len(y)
        batches_scanned += 1
    scans += 1
    if (
        summary is None
        or count != prepared.n_obs
        or not family.execution_summary_input_ok(summary)
    ):
        raise ValueError("NB initial global source summary is invalid")
    metadata = family.finalize_execution_summary(summary)
    beta = np.zeros(p)
    constant_eta = float(
        np.asarray(family.link.initial_link_cpu(np.asarray(sum_y / count)))
    )
    if prepared.predict_spec.has_intercept:
        beta[0] = constant_eta
    else:
        # get.null.coef projects link(mean(y)) without weights or offsets;
        # its null anchor stays distinct from the per-row mustart predictor.
        beta = project_null_coefficients(
            null_projection, constant_eta
        ).coefficients.copy()
        for block in prepared.fitting.penalty_structure.blocks:
            beta[block.start : block.stop] = np.linalg.solve(
                block.transform.dense(), beta[block.start : block.stop]
            )

    # Initial source rows and public null-projection QR are no longer needed.
    # Release them before any old/current/candidate working scan can coexist;
    # the initial phase's two designs and compact QR/helper fit its own peak.
    if not prepared.predict_spec.has_intercept:
        del public_X
    del null_projection, X, batch, y, weight, _offset, valid, initial

    if beta_old_init is not None:
        beta = np.array(beta_old_init, dtype=float, copy=True)
        if beta.shape != (p,) or not np.all(np.isfinite(beta)):
            raise ValueError("finite matching NB null anchor required")
    beta_null = beta.copy()
    if estimate_theta:
        capacity = max(1, int(np.ceil(metadata["max_count"])))
        # The inherited table-selection policy budgets four capacity arrays up
        # to 8MiB. Charge sixteen arrays for this combined value/gradient/
        # Hessian graph; the recurrence branch has only batch-vector storage.
        table_policy_bytes = 8 * 4 * (capacity + 1)
        prefix_bytes = 8 * 16 * (capacity + 1)
        prefix_bytes = (
            prefix_bytes
            if metadata["integer_counts"] and table_policy_bytes <= 8 << 20
            else 0
        )
        ledger = replace(
            ledger, device_visible_bytes=ledger.device_visible_bytes + prefix_bytes
        )
        if ledger.required_bytes > maximum_bytes:
            raise MemoryError(
                "NB conditional count-prefix workspace exceeds maximum_bytes"
            )

    def scan_at(factory, *, positive=False, score=False):
        nonlocal scans, batches_scanned
        scan = nb_working_scan(
            stream,
            family,
            lineage,
            parameters,
            control,
            factory,
            positive_observed=positive,
            score_system=score,
            device=device,
        )
        scans += 1
        batches_scanned += scan.batches_scanned
        return scan

    def trial(candidate):
        # Trial reduction uses the same source-owned NB deviance arithmetic;
        # no ordinary family reporting clamp or tiny-mu derivative clip.
        nonlocal scans, batches_scanned
        deviance, domain = 0.0, True
        if not np.all(np.isfinite(candidate)):
            return np.inf, False
        for batch, y, weight, offset, valid in _source_batches(
            stream, family, lineage, control.batch_rows
        ):
            X = prepared.evaluate_fitting_batch(batch) if len(y) else np.empty((0, p))
            working = _working(
                *[
                    jax.device_put(value, device)
                    for value in (X @ candidate + offset, y, weight, offset, valid)
                ],
                parameters,
                link=_link_name(family),
            )
            deviance += float(np.asarray(working.deviance))
            domain = domain and bool(np.asarray(working.domain_ok))
            batches_scanned += 1
        scans += 1
        return deviance, domain and np.isfinite(deviance)

    deviance, domain = trial(beta)
    deviance_theta = tuple(np.asarray(parameters.log_theta))
    if not np.isfinite(deviance):
        raise FloatingPointError("NB null source deviance is nonfinite")
    if not domain:
        raise ValueError("NB null coefficient anchor leaves the family domain")
    old_pdev = deviance + float(beta @ penalty(beta))
    if not np.isfinite(old_pdev):
        raise FloatingPointError("NB null penalized deviance is nonfinite")
    initial_retained = False
    if beta_start is not None and start_is_absent is not True:
        retained = np.array(beta_start, dtype=float, copy=True)
        if retained.shape != (p,) or not np.all(np.isfinite(retained)):
            raise ValueError("finite matching NB retained start required")
        start_dev, start_domain = trial(retained)
        start_pdev = start_dev + float(retained @ penalty(retained))
        if start_domain and np.isfinite(start_pdev) and start_pdev <= old_pdev:
            beta = retained
            initial_retained = True
    elif beta_start is None and start_is_absent is False:
        raise ValueError("NB retained start flag requires beta_start")
    history = [old_pdev]
    theta_iterations = theta_scans = theta_status = 0
    theta_history = [float(np.asarray(parameters.log_theta)[0])]
    last_theta_result = None
    stopping_pdev = old_pdev
    recoveries = backtracks = 0
    converged = failed = False
    candidate_scan = None
    for iteration in range(control.max_iter):
        eta_factory = (
            (
                lambda batch, _X: (
                    family.initial_working_state_cpu(
                        batch.y, batch.weight, batch.valid
                    ).eta
                )
            )
            if iteration == 0 and not initial_retained
            else (lambda batch, X, beta=beta: X @ beta + batch.offset)
        )
        current = scan_at(eta_factory) if iteration == 0 else candidate_scan
        proposal = solve_signed_qr(current.observed, actual, balanced_roots=balanced)
        original_posdef = not proposal.fisher_required
        if not original_posdef:
            recoveries += 1
            current = scan_at(eta_factory, positive=True)
            proposal = solve_signed_qr(
                current.observed, actual, balanced_roots=balanced
            )
        if (
            proposal.fisher_required
            or proposal.coefficients is None
            or not np.all(np.isfinite(proposal.coefficients))
        ):
            raise FloatingPointError("NB positive-observed coefficient recovery failed")
        candidate = proposal.coefficients
        threshold = 10.0 * (0.1 + abs(old_pdev)) * np.sqrt(np.finfo(float).eps)
        accepted = False
        if estimate_theta or beta_start is not None:
            # gam.fit4 safeguards finite deviance, then domain, then divergence.
            # The first two use coefold (a retained start when present), while
            # immediate divergence switches to null.coef even for a retained start.
            candidate_dev, candidate_domain = trial(candidate)
            guard_halvings = 0
            while not np.isfinite(candidate_dev):
                if guard_halvings >= control.max_iter:
                    break
                candidate = 0.5 * (candidate + beta)
                guard_halvings += 1
                candidate_dev, candidate_domain = trial(candidate)
            domain_halvings = 0
            while np.isfinite(candidate_dev) and not candidate_domain:
                if domain_halvings >= control.max_iter:
                    break
                candidate = 0.5 * (candidate + beta)
                domain_halvings += 1
                candidate_dev, candidate_domain = trial(candidate)
            candidate_pdev = candidate_dev + float(candidate @ penalty(candidate))
            divergence_halvings = 0
            while (
                candidate_domain
                and np.isfinite(candidate_pdev)
                and candidate_pdev - old_pdev > threshold
            ):
                if divergence_halvings >= 100:
                    break
                candidate = 0.5 * (candidate + (beta_null if iteration == 0 else beta))
                divergence_halvings += 1
                candidate_dev, candidate_domain = trial(candidate)
                candidate_pdev = candidate_dev + float(candidate @ penalty(candidate))
            accepted = (
                candidate_domain
                and np.isfinite(candidate_pdev)
                and candidate_pdev - old_pdev <= threshold
            )
            backtracks += guard_halvings + domain_halvings + divergence_halvings
        else:
            for halving in range(control.max_halvings + 1):
                candidate_dev, candidate_domain = trial(candidate)
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate_pdev = candidate_dev + float(
                        candidate @ penalty(candidate)
                    )
                if (
                    candidate_domain
                    and np.isfinite(candidate_pdev)
                    and candidate_pdev - old_pdev <= threshold
                ):
                    accepted = True
                    backtracks += halving
                    break
                candidate = 0.5 * (candidate + (beta_null if iteration == 0 else beta))
        if not accepted:
            failed = True
            break
        candidate_theta = tuple(np.asarray(parameters.log_theta))
        if estimate_theta:
            # Counts and carried theta are already separate. Do not retain the
            # preceding conditional histories during the next replay/return.
            last_theta_result = None
            last_theta_result = conditional_theta_stream(
                stream,
                family,
                lineage,
                candidate,
                np.asarray(parameters.log_theta),
                control,
                max_y=int(np.ceil(metadata["max_count"])),
                integer_counts=bool(metadata["integer_counts"]),
                control=theta_control,
                device=device,
            )
            scans += last_theta_result.source_scans
            batches_scanned += last_theta_result.batches_scanned
            theta_scans += last_theta_result.source_scans
            theta_iterations += last_theta_result.n_iter
            theta_status = last_theta_result.status
            parameters = jax.device_put(
                FamilyExecutionParameters(np.asarray(last_theta_result.log_theta)),
                device,
            )
            theta_history.append(last_theta_result.log_theta[0])
            if not last_theta_result.converged:
                deviance_theta = candidate_theta
                beta, deviance, stopping_pdev = candidate, candidate_dev, candidate_pdev
                failed = True
                break
        candidate_scan = scan_at(
            lambda batch, X, candidate=candidate: X @ candidate + batch.offset
        )
        residual = (
            candidate_scan.gradient_G @ candidate
            + penalty(candidate)
            - candidate_scan.gradient_rhs
        )
        change_ok = (
            abs(candidate_pdev - old_pdev) / (0.1 + abs(candidate_pdev)) < control.tol
        )
        gradient_ok = np.max(np.abs(2.0 * residual)) <= control.tol * (
            abs(candidate_pdev) + 1.0
        )
        deviance_theta = candidate_theta
        stopping_pdev = candidate_pdev
        beta, deviance, old_pdev = candidate, candidate_dev, candidate_pdev
        history.append(old_pdev)
        if original_posdef and change_ok and gradient_ok:
            converged = True
            break
        if estimate_theta:
            # Source refresh happens only after a continuing iteration; the
            # stopping comparison above deliberately used the pre-theta pdev.
            deviance, refresh_domain = trial(beta)
            deviance_theta = tuple(np.asarray(parameters.log_theta))
            if not refresh_domain:
                raise FloatingPointError("NB post-theta objective refresh is invalid")
            old_pdev = deviance + float(beta @ penalty(beta))
    final = scan_at(lambda batch, X: X @ beta + batch.offset, score=True)
    observed = solve_signed_qr(final.observed, actual, balanced_roots=balanced)
    if not observed.score_admissible or observed.fisher_required:
        raise FloatingPointError("NB final observed determinant is inadmissible")
    # gdi2's get_bSb receives PKtz from the final observed weighted solve.
    # gam.fit4 reports the accepted PIRLS beta and pre-gdi deviance, but its
    # score uses this solve's penalty. Keep that provenance explicit.
    gdi_penalty = float(observed.coefficients @ penalty(observed.coefficients))
    score_pdev = deviance + gdi_penalty
    observed_factor = SignedQRCoefficientFactor(
        _tag_absolute(observed.absolute_factor, device),
        jax.device_put(observed.vectors, device),
        jax.device_put(observed.correction, device),
    )
    fisher_factor = _tag_absolute(
        solve_augmented_qr(final.fisher, actual, balanced_roots=balanced), device
    )
    # gdi2 first computes the observed score factor, then rebuilds rV/K from
    # expected W at lines 2262-2299. Reported EDF/covariance use that Fisher
    # solve; the observed factor remains exclusively the likelihood factor.
    edf = float(
        np.asarray(
            np.trace(
                fisher_factor.hessian_inverse(jax.device_put(final.fisher_G, device))
            )
        )
    )
    saturated = 0.0
    for _batch, y, weight, _offset, valid in _source_batches(
        stream, family, lineage, control.batch_rows
    ):
        value, admissible = batch_saturated_loglikelihood(
            *[jax.device_put(x, device) for x in (y, weight, valid, 1.0)],
            parameters,
            family,
            lineage.context,
            max_y=int(np.ceil(metadata["max_count"])),
        )
        if not bool(np.asarray(admissible)):
            raise FloatingPointError("NB saturated likelihood is inadmissible")
        saturated += float(np.asarray(value))
        batches_scanned += 1
    scans += 1
    fitting = PreparedFittingMetadata.from_prepared(prepared, family, device)
    score = float(
        np.asarray(
            reml_criterion_from_penalized_deviance(
                jax.device_put(rho, device),
                jax.device_put(final.observed_G, device),
                jax.device_put(score_pdev, device),
                jax.device_put(saturated, device),
                fitting.penalty_structure,
                jax.device_put(1.0, device),
                fitting.total_penalty_null_dim,
                fitting.singleton_sp_indices,
                fitting.singleton_ranks,
                fitting.singleton_eig_constants,
                fitting.multi_block_sp_indices,
                fitting.multi_block_ranks,
                fitting.multi_block_proj_S,
                fitting.rank_deficit,
            )
        )
    )
    H_beta = final.observed_G @ beta + penalty(beta)
    stationarity = float(
        np.max(np.abs(H_beta - final.observed_rhs))
        / (1 + np.max(np.abs(H_beta)) + np.max(np.abs(final.observed_rhs)))
    )
    if not all(
        np.isfinite(x)
        for x in (edf, saturated, stationarity, old_pdev, score, gdi_penalty)
    ):
        raise FloatingPointError("NB final coefficient statistics are nonfinite")
    # This reporting bound covers accumulated roundoff at an exact fit only.
    # Raw source deviance remains authoritative for trials and score provenance;
    # materially negative values (including fractional-response boundaries)
    # must not become an apparently valid nonnegative fit.
    reporting_roundoff = 64 * np.finfo(np.float64).eps * (1 + count + sum_y)
    if not np.isfinite(deviance) or deviance < -reporting_roundoff:
        raise FloatingPointError("NB raw deviance is materially negative or nonfinite")
    final_theta_deviance = final.deviance if estimate_theta else deviance
    if (
        not np.isfinite(final_theta_deviance)
        or final_theta_deviance < -reporting_roundoff
    ):
        raise FloatingPointError(
            "NB final-theta deviance is materially negative or nonfinite"
        )
    reported_deviance = max(0.0, final_theta_deviance)
    state = StreamFitState(
        coefficients=jax.device_put(beta, device),
        log_lambda=jax.device_put(rho, device),
        deviance=jax.device_put(reported_deviance, device),
        penalized_deviance=jax.device_put(
            reported_deviance + float(beta @ penalty(beta)), device
        ),
        scale=jax.device_put(1.0, device),
        score_scale=jax.device_put(1.0, device),
        saturated_loglik=jax.device_put(saturated, device),
        edf=jax.device_put(edf, device),
        xtwx=jax.device_put(final.observed_G, device),
        xtwx_fisher=jax.device_put(final.fisher_G, device),
        coefficient_factor=observed_factor,
        fisher_coefficient_factor=fisher_factor,
        n_iter=iteration + 1,
        converged=converged and not failed,
        line_search_failed=failed,
        backtracks=backtracks,
        stationarity=stationarity,
        source_scans=scans,
        batches_scanned=batches_scanned,
    )
    return NBStreamResult(
        state,
        recoveries,
        tuple(history),
        ledger,
        tuple(np.asarray(parameters.log_theta)),
        bool(metadata["integer_counts"]),
        float(metadata["max_count"]),
        gdi_penalty,
        score_pdev,
        score,
        deviance,
        initial_retained,
        theta_iterations,
        theta_status,
        theta_scans,
        stopping_pdev,
        final_theta_deviance,
        tuple(theta_history),
        last_theta_result,
        deviance_theta,
    )
