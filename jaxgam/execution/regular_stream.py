"""Source-style regular working scans for the signed streamed controller.

These opt-in execution primitives do not widen the public stream release
policy. Extended-family ``gam.fit4`` positive-observed-weight recovery is
distinct from regular ``gam.fit3`` step-local Fisher recovery.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np

from jaxgam.data.source import RowBatch
from jaxgam.execution.family_initialization import select_initial_working_state_cpu
from jaxgam.execution.null_coefficient import project_null_coefficients
from jaxgam.execution.qr import PositiveQRState, qr_update
from jaxgam.execution.signed_qr import SignedQRState, signed_qr_update, solve_signed_qr
from jaxgam.execution.stream import StreamPIRLSControl, _source_batches
from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting.family_execution import (
    FamilyExecutionLineage,
    FamilyExecutionParameters,
    batch_initial_working_quantities,
    batch_regular_fletcher_statistics,
    batch_saturated_loglikelihood,
    finalize_initial_working_status,
    finalize_regular_fletcher_scale,
    initial_working_status,
    merge_initial_working_status,
    merge_regular_fletcher_statistics,
)
from jaxgam.fitting.regular_stream import regular_trial_deviance
from jaxgam.fitting.signed_qr import SignedQRCoefficientFactor
from jaxgam.fitting.state import PivotedQRCoefficientFactor, StreamFitState
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.fitting_prepare import qr_penalty_roots


@dataclass(frozen=True)
class RegularStreamWorkspace:
    """Conservative ledger of known host/device numeric workspace.

    Source-owned training storage and opaque native/compiled library scratch
    are excluded. This is a numerical-workspace limit, not a process RSS cap.
    QR packed copies and negative-row selections are counted independently.
    """

    batch_rows: int
    n_coef: int
    host_batch_bytes: int
    host_coefficient_bytes: int
    device_visible_bytes: int
    penalty_bytes: int
    history_bytes: int
    null_projection_bytes: int

    @property
    def required_bytes(self) -> int:
        return (
            self.host_batch_bytes
            + self.host_coefficient_bytes
            + self.device_visible_bytes
            + self.penalty_bytes
            + self.history_bytes
            + self.null_projection_bytes
        )


def preflight_regular_stream_workspace(
    stream: StreamDesign,
    control: StreamPIRLSControl,
    maximum_bytes: int,
) -> RegularStreamWorkspace:
    """Reject known oversized work before scans, QR or device transfers.

    Eight batch-design buffers cover source design, weighted/negative rows,
    separate QR stack/packed copies, statistics products and one transfer.
    Forty coefficient matrices cover old/new working statistics, the three
    QR states, balanced/actual root augmentation, SVD output/copies and
    retained final factors. Sixty-four batch vectors cover explicit inputs,
    family working outputs and packed-Q RHS copies. These are conservative
    simultaneous-array bounds; opaque XLA/BLAS/LAPACK scratch is not counted.
    """
    if (
        not isinstance(maximum_bytes, int)
        or isinstance(maximum_bytes, bool)
        or maximum_bytes <= 0
    ):
        raise ValueError("maximum_bytes must be a positive integer")
    prepared = stream.prepared
    if prepared.fitting is None:
        raise ValueError("regular streaming requires fitting preparation")
    p, B = prepared.n_coef, min(control.batch_rows, prepared.n_obs)
    structure = prepared.fitting.penalty_structure
    # Count local CPU penalty matrices, local coordinate transforms, roots
    # and their actual/balanced copies, without globally padding m matrices.
    penalty_entries = sum(
        block.size**2 * (len(block.local_penalties) + 1)
        + 3 * block.size * sum(block.ranks)
        for block in structure.blocks
    )
    ledger = RegularStreamWorkspace(
        B,
        p,
        8 * (8 * B * p + 64 * B),
        8 * (40 * p * p + 64 * p),
        8 * (B * p + 24 * B + 8 * p * p + 16 * p),
        8 * penalty_entries,
        # Retained float objects, an overallocated list (at most 2N+8
        # references), and the final tuple coexist at return. Charge those
        # visible Python buffers as well as the numerical history values.
        (control.max_iter + 1) * (sys.getsizeof(0.0) + 3 * np.dtype(np.intp).itemsize)
        + sys.getsizeof([])
        + sys.getsizeof(())
        + 8 * np.dtype(np.intp).itemsize,
        # Unweighted public QR is reduced during the initial summary scan.
        # Charge its state, stack/packed copies, natural-order helper copy,
        # Householder vectors, alias-column shift temporary and local-D
        # conversion while the final source batch can remain alive.
        8 * (4 * B * p + 8 * p * p + 16 * B + 16 * p),
    )
    if ledger.required_bytes > maximum_bytes:
        raise MemoryError(
            f"regular signed QR known workspace requires {ledger.required_bytes} "
            f"bytes, exceeding maximum_bytes={maximum_bytes}"
        )
    return ledger


@dataclass(frozen=True)
class RegularWorkingScan:
    """Distinct selected Newton, observed-score and Fisher reporting state."""

    newton: SignedQRState | None
    fisher: PositiveQRState | None
    observed: SignedQRState | None
    selected_G: np.ndarray
    observed_G: np.ndarray | None
    fisher_G: np.ndarray | None
    selected_rhs: np.ndarray
    informative_count: int
    batches_scanned: int
    observed_information_ok: bool
    fisher_system_ok: bool
    fisher_rhs: np.ndarray | None


EtaFactory = Callable[[RowBatch, np.ndarray], np.ndarray]


def regular_working_scan(
    stream: StreamDesign,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    parameters: FamilyExecutionParameters,
    control: StreamPIRLSControl,
    eta_for_batch: EtaFactory,
    *,
    score_system: bool = False,
    device: jax.Device | None = None,
) -> RegularWorkingScan:
    """Replay one globally admissible source eta without row retention.

    A raw observed diagnostic does not veto an otherwise admissible selected
    coefficient step. An explicitly requested final score scan requires its
    own finite observed system. Empty/zero-prior blocks are neutral; the
    informative-count requirement is applied only after the complete scan.
    """
    if lineage.context.capabilities.dynamic_theta:
        raise NotImplementedError("regular working scans exclude dynamic theta")
    lineage.validate(stream.prepared, family)
    p = stream.prepared.n_coef
    fisher = None
    newton = None
    observed = None
    status = None
    selected_G = np.zeros((p, p))
    fisher_G = np.zeros((p, p))
    observed_G = np.zeros((p, p)) if score_system else None
    selected_rhs = np.zeros(p)
    fisher_rhs = np.zeros(p)
    observed_ok = True
    fisher_ok = True
    batches = 0
    valid_count = 0
    is_newton = lineage.context.capabilities.coefficient_system == "observed"
    for batch, y, weight, offset, valid in _source_batches(
        stream, family, lineage, control.batch_rows
    ):
        rows = len(valid)
        if (
            rows > min(control.batch_rows, stream.prepared.n_obs)
            or valid_count + rows > stream.prepared.n_obs
        ):
            raise ValueError("regular scan exceeds source row/batch cap")
        valid_count += rows
        X = stream.prepared.evaluate_fitting_batch(batch) if rows else np.empty((0, p))
        if X.shape != (rows, p) or not np.all(np.isfinite(X)):
            raise ValueError("finite matching fitting design required")
        eta = np.asarray(eta_for_batch(batch, X), dtype=float)
        if eta.shape != (rows,):
            raise ValueError("eta_for_batch must return one predictor per row")
        lineage.validate(stream.prepared, family)
        working = batch_initial_working_quantities(
            jax.device_put(y, device),
            jax.device_put(weight, device),
            jax.device_put(offset, device),
            jax.device_put(valid, device),
            jax.device_put(eta, device),
            parameters,
            family,
            lineage.context,
            source_signed_recovery=True,
        )
        batch_status = initial_working_status(working)
        status = (
            batch_status
            if status is None
            else merge_initial_working_status(status, batch_status)
        )
        jax.block_until_ready(working)
        # Fail an invalid selected block before QR, while deferring the
        # globally informative requirement across neutral blocks.
        if not bool(np.asarray(batch_status[0])):
            raise ValueError("regular selected working system is inadmissible")
        # gam.fit3 subsets ``good = weights > 0 & mu.eta != 0`` before
        # pls_fit1. Explicit neutral rows change packed-QR rounding, even
        # though their contribution to the mathematical system is zero.
        # Keep the shared kernel's mask rather than infer support from
        # curvature: a good row may have exactly zero observed curvature.
        good = np.asarray(working.informative_mask, dtype=bool)
        X = X[good]
        wf, zf = (
            np.asarray(working.fisher_weight)[good],
            np.asarray(working.fisher_response)[good],
        )
        batch_fisher_ok = bool(np.asarray(working.fisher_system_ok)) and np.all(
            wf >= 0.0
        )
        fisher_ok = fisher_ok and batch_fisher_ok
        if batch_fisher_ok:
            sqrt_fisher = np.sqrt(wf)
            fisher = qr_update(
                fisher, sqrt_fisher[:, None] * X, sqrt_fisher * zf, n_coef=p
            )
            fisher_G += X.T @ (wf[:, None] * X)
            fisher_rhs += X.T @ (wf * zf)
        if is_newton:
            ws, zs = (
                np.asarray(working.newton_weight)[good],
                np.asarray(working.newton_response)[good],
            )
            newton = signed_qr_update(newton, X, ws, zs)
        else:
            ws, zs = wf, zf
        selected_G += X.T @ (ws[:, None] * X)
        selected_rhs += X.T @ (ws * zs)
        observed_ok = observed_ok and bool(np.asarray(working.observed_information_ok))
        if score_system:
            # gam.fit3 retains the Fisher system for a canonical link. Raw
            # AD curvature remains a diagnostic; it must not silently select
            # a different final likelihood determinant in that branch.
            canonical_score = (
                lineage.context.capabilities.fisher_equals_observed_for_score
            )
            if (canonical_score and not fisher_ok) or (
                not canonical_score and not observed_ok
            ):
                raise FloatingPointError("nonfinite observed score information")
            wo = wf if canonical_score else np.asarray(working.observed_weight)[good]
            observed = signed_qr_update(observed, X, wo, np.zeros(len(X)))
            assert observed_G is not None
            observed_G += X.T @ (wo[:, None] * X)
        # Finite rows can overflow coefficient-space products. Never allow
        # that to reach convergence or likelihood scoring.
        arrays = [selected_G, selected_rhs, fisher_G, fisher_rhs]
        if observed_G is not None:
            arrays.append(observed_G)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise FloatingPointError("regular coefficient statistics overflow")
        batches += 1
    if status is None or not bool(np.asarray(finalize_initial_working_status(status))):
        raise ValueError("regular source has no globally admissible informative rows")
    return RegularWorkingScan(
        newton,
        fisher if fisher_ok else None,
        observed,
        selected_G,
        observed_G,
        fisher_G if fisher_ok else None,
        selected_rhs,
        int(np.asarray(status[1])),
        batches,
        observed_ok,
        fisher_ok,
        fisher_rhs if fisher_ok else None,
    )


@dataclass(frozen=True)
class RegularStreamResult:
    """Compact fit plus measured regular-controller recovery metadata."""

    state: StreamFitState
    fisher_recoveries: int
    initial_shrinks: int
    accepted_penalized_history: tuple[float, ...]
    workspace: RegularStreamWorkspace
    final_refit_accepted: bool
    information_coefficients: np.ndarray
    null_coefficients: np.ndarray
    initial_coefficients_present: bool
    source_score: RegularSourceScore

    def __post_init__(self) -> None:
        for name in ("information_coefficients", "null_coefficients"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RegularSourceScore:
    """Source bookkeeping before and after the final coefficient refit.

    gam.fit3 scores pre-gdi1 deviance plus gdi1's candidate solve penalty,
    even if its domain recovery returns the previous feasible coefficients.
    Factors remain derived at information_coefficients; reporting and q
    may use the returned coefficients. Incoming trial phi and reported
    Fletcher phi therefore have separate identities.
    """

    raw_deviance: float
    stopping_penalized_deviance: float
    solve_penalty: float
    candidate_valid: bool
    score_phi: float
    reported_phi: float

    @property
    def penalized_deviance(self) -> float:
        return self.raw_deviance + self.solve_penalty


def _positive_signed_state(scan: RegularWorkingScan) -> SignedQRState:
    if scan.fisher is None or scan.fisher_rhs is None:
        raise FloatingPointError(
            "regular Fisher recovery/reporting system is unavailable"
        )
    p = scan.fisher.n_coef
    neutral = PositiveQRState(np.empty((0, p)), np.empty(0), 0.0, 0, np.arange(p))
    return SignedQRState(scan.fisher, neutral, scan.fisher_rhs)


def _tag_absolute(factor, device: jax.Device | None) -> PivotedQRCoefficientFactor:
    return PivotedQRCoefficientFactor(
        jax.device_put(factor.R, device),
        jax.device_put(factor.pivots, device),
        jax.device_put(factor.keep, device),
        factor.original_n_coef,
    )


def fit_regular_streamed_pirls(
    stream: StreamDesign,
    family: ExponentialFamily,
    log_lambda: np.ndarray,
    *,
    maximum_bytes: int,
    score_scale: float | None = None,
    control: StreamPIRLSControl | None = None,
    device: jax.Device | None = None,
    initial_coefficients: np.ndarray | None = None,
) -> RegularStreamResult:
    """Fit a regular fixed-sp/trial-scale system with source initialization.

    This internal engine requires unpadded prepared batches. A retained
    p-vector start is expressed in local-D fitting coordinates. The separate
    source null anchor is projected in public coordinates with R's natural
    column policy, then converted to local-D coordinates.
    Unknown-scale fits require an explicit trial score scale; Fletcher
    reporting does not profile an outer REML scale. Public release and the
    all-family outer controller remain separate validation milestones.
    """
    control = StreamPIRLSControl(solver_policy="qr") if control is None else control
    if control.solver_policy != "qr":
        raise ValueError("regular source controller requires solver_policy='qr'")
    ledger = preflight_regular_stream_workspace(stream, control, maximum_bytes)
    prepared = stream.prepared
    assert prepared.fitting is not None
    lineage = FamilyExecutionLineage.from_prepared(prepared, family)
    capabilities = lineage.context.capabilities
    if capabilities.dynamic_theta or not all(
        (
            capabilities.row_separable,
            capabilities.fisher_working_system,
            capabilities.observed_information,
            capabilities.direct_deviance,
            capabilities.saturated_loglikelihood,
        )
    ):
        raise NotImplementedError(
            "regular controller requires the static regular-family execution contract"
        )
    policy = lineage.context.reduction_policy
    if policy.reported_scale not in {
        "known_one",
        "regular_fletcher",
        "gaussian_fisher_edf_deviance",
    }:
        raise NotImplementedError("regular reported-scale policy is unavailable")
    phi = 1.0 if family.scale_known and score_scale is None else score_scale
    if phi is None:
        raise ValueError(
            "unknown-scale regular fit requires an explicit trial score_scale"
        )
    phi = float(phi)
    if not np.isfinite(phi) or phi <= 0.0 or (family.scale_known and phi != 1.0):
        raise ValueError(
            "score_scale must be finite, positive and respect the known scale"
        )
    rho = np.asarray(log_lambda, float)
    if rho.shape != prepared.fitting.log_lambda_init.shape or not np.all(
        np.isfinite(rho)
    ):
        raise ValueError("finite matching log smoothing parameters required")
    layout = qr_penalty_roots(prepared.fitting.penalty_structure)
    balanced = tuple((slice(root.start, root.stop), root.root) for root in layout)
    actual = tuple(
        (slice(root.start, root.stop), root.root * np.exp(0.5 * rho[root.sp_index]))
        for root in layout
    )
    if not all(np.all(np.isfinite(root)) for _, root in actual):
        raise FloatingPointError("regular actual penalty roots overflow")
    parameters = jax.device_put(
        FamilyExecutionParameters.from_snapshot(lineage.parameters), device
    )
    scans = 0
    batches_scanned = 0
    p = prepared.n_coef
    retained_start = None
    if initial_coefficients is not None:
        retained_start = np.array(initial_coefficients, dtype=float, copy=True)
        if retained_start.shape != (p,) or not np.all(np.isfinite(retained_start)):
            raise ValueError("initial_coefficients must be a finite fitting p-vector")

    def checked_X(batch: RowBatch) -> np.ndarray:
        rows = len(batch.valid)
        if rows > ledger.batch_rows:
            raise ValueError("regular source exceeds prospective batch cap")
        X = prepared.evaluate_fitting_batch(batch) if rows else np.empty((0, p))
        if X.shape != (rows, p) or not np.all(np.isfinite(X)):
            raise ValueError("finite matching regular fitting design required")
        return X

    def penalty_apply(beta: np.ndarray, *, require_finite: bool = True) -> np.ndarray:
        result = np.zeros(p)
        for where, root in actual:
            result[where] += root.T @ (root @ beta[where])
        if require_finite and not np.all(np.isfinite(result)):
            raise FloatingPointError("regular penalty action overflow")
        return result

    def penalized(beta: np.ndarray, deviance: float) -> float:
        value = float(deviance + beta @ penalty_apply(beta))
        if not np.isfinite(value):
            raise FloatingPointError("regular penalized deviance is nonfinite")
        return value

    def trial(beta: np.ndarray) -> tuple[float, bool]:
        nonlocal scans, batches_scanned
        if not np.all(np.isfinite(beta)):
            return np.inf, False
        deviance = 0.0
        domain = True
        for batch, y, weight, offset, valid in _source_batches(
            stream, family, lineage, control.batch_rows
        ):
            X = checked_X(batch)
            lineage.validate(prepared, family)
            value, admissible = regular_trial_deviance(
                jax.device_put(X, device),
                jax.device_put(y, device),
                jax.device_put(weight, device),
                jax.device_put(offset, device),
                jax.device_put(valid, device),
                jax.device_put(beta, device),
                parameters,
                family,
                lineage.context,
            )
            deviance += float(np.asarray(value))
            domain = domain and bool(np.asarray(admissible))
            batches_scanned += 1
        scans += 1
        return deviance, domain and np.isfinite(deviance)

    # get.null.coef uses unweighted PUBLIC X and mean(initialized y), not a
    # weighted mean or a projected mustart predictor. Real prior-zero rows
    # participate in both reductions; source padding is already excluded.
    sum_y = 0.0
    count = 0
    summary = None
    null_qr = None
    for batch, y, weight, _offset, valid in _source_batches(
        stream, family, lineage, control.batch_rows
    ):
        rows = len(valid)
        if rows > ledger.batch_rows:
            raise ValueError("regular source exceeds prospective batch cap")
        X_public = prepared.evaluate_batch(batch) if rows else np.empty((0, p))
        if X_public.shape != (rows, p) or not np.all(np.isfinite(X_public)):
            raise ValueError("finite matching public null-anchor design required")
        null_qr = qr_update(null_qr, X_public, np.ones(rows), n_coef=p)
        normalized_y = family.execution_initial_response_cpu(y, weight)
        if (
            not family.response_support.check(normalized_y)
            or not np.all(np.isfinite(normalized_y))
            or not np.all(np.isfinite(weight) & (weight >= 0.0))
        ):
            raise ValueError("regular family NULL-start input is invalid")
        sum_y += float(np.sum(normalized_y))
        count += len(y)
        batch_summary = family.execution_summary_from_batch(normalized_y, weight, valid)
        summary = (
            batch_summary
            if summary is None
            else family.merge_execution_summaries(summary, batch_summary)
        )
        batches_scanned += 1
    scans += 1
    if (
        count != prepared.n_obs
        or not np.isfinite(sum_y)
        or summary is None
        or not family.execution_summary_input_ok(summary)
    ):
        raise FloatingPointError("regular initial source summary is invalid")
    metadata = family.finalize_execution_summary(summary)
    constant_eta = float(
        np.asarray(family.link.initial_link_cpu(np.asarray(sum_y / count)))
    )
    assert null_qr is not None
    null_public = project_null_coefficients(null_qr, constant_eta)
    beta = np.array(null_public.coefficients, copy=True)
    for block in prepared.fitting.penalty_structure.blocks:
        where = slice(block.start, block.stop)
        beta[where] = np.linalg.solve(block.transform.dense(), beta[where])
    if not np.all(np.isfinite(beta)):
        raise FloatingPointError("regular null coefficient is nonfinite")

    def null_eta(batch: RowBatch) -> np.ndarray:
        return checked_X(batch) @ beta + batch.offset

    def retained_eta(batch: RowBatch) -> np.ndarray:
        assert retained_start is not None
        return checked_X(batch) @ retained_start + batch.offset

    selection = select_initial_working_state_cpu(
        stream,
        family,
        lineage,
        batch_rows=control.batch_rows,
        null_eta_for_batch=null_eta,
        summary=metadata,
        eta_for_batch=retained_eta if retained_start is not None else None,
    )
    scans += selection.source_scans
    batches_scanned += selection.batches_scanned
    if not selection.input_ok or not selection.domain_ok:
        raise ValueError("regular family initial predictor cannot enter its domain")
    null_beta = beta.copy()

    def initial_eta(batch: RowBatch, X: np.ndarray) -> np.ndarray:
        if retained_start is None:
            initial = family.initial_working_state_cpu(
                batch.y, batch.weight, batch.valid, summary=metadata
            )
            eta = initial.eta
        else:
            eta = X @ retained_start + batch.offset
        anchor = X @ null_beta + batch.offset
        for _ in range(selection.shrink_count):
            eta = 0.9 * eta + 0.1 * anchor
        return eta

    current_deviance, domain = trial(beta)
    if not domain:
        raise ValueError("regular null coefficient anchor leaves the family domain")
    old_pdev = penalized(beta, current_deviance)
    history = [old_pdev]
    converged = False
    line_search_failed = False
    backtracks = 0
    recoveries = 0
    selected_scan = None

    def scan_at(factory: EtaFactory, score_system: bool = False) -> RegularWorkingScan:
        nonlocal scans, batches_scanned
        result = regular_working_scan(
            stream,
            family,
            lineage,
            parameters,
            control,
            factory,
            score_system=score_system,
            device=device,
        )
        scans += 1
        batches_scanned += result.batches_scanned
        return result

    def coefficient_proposal(scan: RegularWorkingScan):
        nonlocal recoveries
        source = (
            scan.newton if scan.newton is not None else _positive_signed_state(scan)
        )
        result = solve_signed_qr(source, actual, balanced_roots=balanced)
        G, rhs = scan.selected_G, scan.selected_rhs
        if result.fisher_required:
            recoveries += 1
            result = solve_signed_qr(
                _positive_signed_state(scan), actual, balanced_roots=balanced
            )
            G, rhs = scan.fisher_G, scan.fisher_rhs
        if result.coefficients is None or not np.all(np.isfinite(result.coefficients)):
            raise FloatingPointError("regular coefficient candidate is nonfinite")
        return result.coefficients, G, rhs

    for iteration in range(control.max_iter):
        selected_scan = scan_at(
            initial_eta
            if iteration == 0
            else lambda batch, X, beta=beta: X @ beta + batch.offset
        )
        proposal, G, rhs = coefficient_proposal(selected_scan)
        accepted = False
        candidate = proposal
        threshold = 10.0 * (0.1 + abs(old_pdev)) * np.sqrt(np.finfo(float).eps)
        for halving in range(control.max_halvings + 1):
            candidate_deviance, candidate_domain = trial(candidate)
            # Finite coefficients can still overflow a trial quadratic.
            # Such a trial is rejected and halved, rather than aborting an
            # otherwise recoverable source step.
            with np.errstate(over="ignore", invalid="ignore"):
                candidate_pdev = candidate_deviance + float(
                    candidate @ penalty_apply(candidate, require_finite=False)
                )
            if (
                candidate_domain
                and np.isfinite(candidate_pdev)
                and candidate_pdev - old_pdev <= threshold
            ):
                accepted = True
                backtracks += halving
                break
            candidate = 0.5 * (candidate + beta)
        if not accepted:
            line_search_failed = True
            break
        residual = G @ candidate + penalty_apply(candidate) - rhs
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("regular accepted-state gradient is nonfinite")
        change_ok = abs(candidate_pdev - old_pdev) < control.tol * (
            abs(phi) + abs(candidate_pdev)
        )
        gradient_ok = np.max(np.abs(2.0 * residual), initial=0.0) <= control.tol * (
            abs(candidate_pdev) + abs(phi)
        )
        beta = candidate
        current_deviance = candidate_deviance
        old_pdev = candidate_pdev
        history.append(old_pdev)
        if change_ok and gradient_ok:
            converged = True
            break

    final = scan_at(lambda batch, X: X @ beta + batch.offset, score_system=True)
    information_beta = beta.copy()
    # gam.fit3's final gdi1 recomputes coefficients at the last working
    # system. Its observed determinant and Fisher EDF/covariance retain that
    # system; the subsequent mu/Fletcher reduction uses the refitted beta.
    # A refit leaving the family domain is replaced by the accepted PIRLS
    # coefficient, matching the explicit gam.fit3 recovery at lines 580-589.
    refit_source = (
        final.newton if final.newton is not None else _positive_signed_state(final)
    )
    refit = solve_signed_qr(refit_source, actual, balanced_roots=balanced)
    if (
        refit.fisher_required
        or refit.coefficients is None
        or not np.all(np.isfinite(refit.coefficients))
    ):
        raise FloatingPointError("regular final source refit is inadmissible")
    _, refit_domain = trial(refit.coefficients)
    solve_penalty = penalized(refit.coefficients, 0.0)
    if refit_domain:
        beta = refit.coefficients
    if final.observed is None or final.fisher is None or final.fisher_G is None:
        raise FloatingPointError(
            "regular final observed/Fisher systems are unavailable"
        )
    observed = solve_signed_qr(final.observed, actual, balanced_roots=balanced)
    if not observed.score_admissible or observed.fisher_required:
        raise FloatingPointError("regular final observed determinant is inadmissible")
    observed_factor = SignedQRCoefficientFactor(
        _tag_absolute(observed.absolute_factor, device),
        jax.device_put(observed.vectors, device),
        jax.device_put(observed.correction, device),
    )
    fisher = solve_signed_qr(
        _positive_signed_state(final), actual, balanced_roots=balanced
    )
    fisher_factor = _tag_absolute(fisher.absolute_factor, device)
    unpivoted = np.empty_like(final.fisher.R)
    unpivoted[:, final.fisher.pivots] = final.fisher.R
    whitened = fisher_factor.root_transpose_inverse(jax.device_put(unpivoted.T, device))
    edf = float(np.asarray(np.sum(whitened * whitened)))
    if not np.isfinite(edf):
        raise FloatingPointError("regular Fisher EDF is nonfinite")
    # Source deviance was reduced before gdi1's final coefficient refit.
    # Preserve that bookkeeping rather than silently recomputing its score
    # at a different working system.
    final_deviance = current_deviance
    final_pdev = final_deviance + solve_penalty
    if not np.isfinite(final_pdev):
        raise FloatingPointError("regular final source score is nonfinite")
    fletcher = None
    saturated = 0.0
    for batch, y, weight, offset, valid in _source_batches(
        stream, family, lineage, control.batch_rows
    ):
        X = checked_X(batch)
        normalized_y = family.execution_initial_response_cpu(y, weight)
        if policy.reported_scale == "regular_fletcher":
            lineage.validate(prepared, family)
            stats = batch_regular_fletcher_statistics(
                jax.device_put(X, device),
                jax.device_put(normalized_y, device),
                jax.device_put(weight, device),
                jax.device_put(offset, device),
                jax.device_put(valid, device),
                jax.device_put(beta, device),
                parameters,
                family,
                lineage.context,
            )
            fletcher = (
                stats
                if fletcher is None
                else merge_regular_fletcher_statistics(
                    fletcher, stats, family, lineage.context
                )
            )
        lineage.validate(prepared, family)
        value, admissible = batch_saturated_loglikelihood(
            jax.device_put(normalized_y, device),
            jax.device_put(weight, device),
            jax.device_put(valid, device),
            jax.device_put(phi, device),
            parameters,
            family,
            lineage.context,
            max_y=int(metadata.get("max_y", 0)),
        )
        if not bool(np.asarray(admissible)):
            raise FloatingPointError("regular saturated likelihood is inadmissible")
        saturated += float(np.asarray(value))
        batches_scanned += 1
    scans += 1
    if policy.reported_scale == "known_one":
        scale = 1.0
    elif policy.reported_scale == "regular_fletcher":
        reported = finalize_regular_fletcher_scale(
            fletcher, edf, family, lineage.context
        )
        if not bool(np.asarray(reported.stream_admissible)):
            raise FloatingPointError("regular Fletcher scale is inadmissible")
        scale = float(np.asarray(reported.scale))
    else:
        scale = final_deviance / (prepared.n_obs - edf)
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(saturated):
        raise FloatingPointError("regular final scale/likelihood is nonfinite")
    H_beta = final.selected_G @ beta + penalty_apply(beta)
    residual = H_beta - final.selected_rhs
    stationarity = float(
        np.max(np.abs(residual), initial=0.0)
        / (
            1.0
            + np.max(np.abs(final.selected_rhs), initial=0.0)
            + np.max(np.abs(H_beta), initial=0.0)
        )
    )
    if not np.isfinite(stationarity):
        raise FloatingPointError("regular final coefficient stationarity is nonfinite")
    state = StreamFitState(
        coefficients=jax.device_put(beta, device),
        log_lambda=jax.device_put(rho, device),
        deviance=jax.device_put(final_deviance, device),
        penalized_deviance=jax.device_put(final_pdev, device),
        scale=jax.device_put(scale, device),
        score_scale=jax.device_put(phi, device),
        saturated_loglik=jax.device_put(saturated, device),
        edf=jax.device_put(edf, device),
        xtwx=jax.device_put(final.observed_G, device),
        xtwx_fisher=jax.device_put(final.fisher_G, device),
        coefficient_factor=observed_factor,
        fisher_coefficient_factor=fisher_factor,
        n_iter=iteration + 1,
        converged=converged and not line_search_failed,
        line_search_failed=line_search_failed,
        backtracks=backtracks,
        stationarity=stationarity,
        source_scans=scans,
        batches_scanned=batches_scanned,
    )
    return RegularStreamResult(
        state,
        recoveries,
        selection.shrink_count,
        tuple(history),
        ledger,
        refit_domain,
        information_beta,
        null_beta,
        retained_start is not None,
        RegularSourceScore(
            final_deviance, old_pdev, solve_penalty, refit_domain, phi, scale
        ),
    )
