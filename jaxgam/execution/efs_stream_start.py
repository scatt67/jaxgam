"""Bounded source initialization for the streamed EFS fitting boundary."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from jaxgam.execution.efs_provider import EFSFitRequest
from jaxgam.execution.efs_stream_provider import NBStreamEFSProvider
from jaxgam.execution.null_coefficient import project_null_coefficients
from jaxgam.execution.qr import qr_update
from jaxgam.execution.stream import _source_batches
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.fitting_prepare import initial_log_sp_from_diagonal


def _nb_null_deviance(
    y: np.ndarray, mu: float, prior: np.ndarray, theta: float
) -> float:
    """Pinned nb$dev.resids value, distinct from ordinary defensive defaults."""
    return float(
        np.sum(
            2
            * prior
            * (
                y * np.log(np.maximum(1, y) / mu)
                - (y + theta) * np.log((y + theta) / (mu + theta))
            )
        )
    )


def _startup_workspace_bytes(stream: StreamDesign, batch_rows: int) -> int:
    """Descriptor-only lifetime bound before source evaluation or expansion."""
    p, m = stream.prepared.n_coef, stream.prepared.penalties.n_penalties
    # initial.sp's zip owns the whole expanded local tuple; previous S,
    # identity-construction and abs/activity temporaries can overlap it.
    expanded_scratch = max(
        (
            (len(block.local_penalties) + 4) * block.size**2
            for block in stream.prepared.penalties.blocks
        ),
        default=0,
    ) + max((block.size**2 for block in stream.prepared.penalties.blocks), default=0)
    return 8 * (
        16 * p * p
        + 32 * p
        + 8 * batch_rows * p
        + 64 * batch_rows
        + 8 * m
        + expanded_scratch
    )


@dataclass(frozen=True)
class EFSStreamStartup:
    """Unshifted source starts and separately measured initialization cost.

    ``initial_request`` applies efsudr's +2.5 once. The returned request is
    for the generic accepted/trial controller, which never adds that shift.
    Source storage, prepared bases and opaque native scratch are excluded
    from ``workspace_bytes``; no training rows survive initialization.
    """

    log_lambda: np.ndarray
    null_coefficients: np.ndarray
    score_phi: float
    log_theta: tuple[float, ...]
    null_deviance: float
    initial_weight_policy: str
    source_fingerprint: str
    basis_fingerprint: str
    source_scans: int
    batches_scanned: int
    workspace_bytes: int

    def __post_init__(self) -> None:
        for name in ("log_lambda", "null_coefficients"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def initial_request(self, *, estimated_theta: bool) -> EFSFitRequest:
        """Construct one explicit first-fit request from immutable starts."""
        beta = jnp.asarray(self.null_coefficients)
        if estimated_theta and len(self.log_theta) != 1:
            raise ValueError("Estimated NB startup requires attributed log theta")
        return EFSFitRequest(
            jnp.asarray(self.log_lambda) + 2.5,
            beta,
            jnp.asarray(self.score_phi),
            log_theta_start=jnp.asarray(self.log_theta) if estimated_theta else None,
            beta_old_init=beta if estimated_theta else None,
            start_is_absent=True,
            regular_start_present=False,
        )


def prepare_stream_efs_start(provider: NBStreamEFSProvider) -> EFSStreamStartup:
    """Reduce pinned initial.spg/get.null.coef without materializing X or y.

    PUBLIC-coordinate diagonal weighting precedes the shared initial.sp
    balancing operation. NB selects expected Dmu2 weights globally if any
    observed start weight is negative; this is separate from PIRLS Deta2.
    Null coefficients use unweighted PUBLIC QR, including real zero priors.
    """
    stream, family = provider.stream, provider.family
    lineage, prepared = provider.lineage, provider.stream.prepared
    lineage.validate(prepared, family)
    p = prepared.n_coef
    B = min(provider.control.batch_rows, prepared.n_obs)
    # Compact QR state, stack/packed/immutable copies and natural-column
    # helper plus local-D conversion: <=16p²+32p entries. Design/weighted
    # copies and source/family vectors: <=8Bp+64B. Three diagonals and sp
    # balancing vectors add <=8m. The two scans do not overlap a PIRLS fit.
    workspace = _startup_workspace_bytes(stream, B)
    if workspace + provider.provider_bytes > provider.maximum_bytes:
        raise MemoryError("EFS startup known workspace exceeds maximum_bytes")
    is_nb = isinstance(family, NegativeBinomial)
    log_theta = tuple(lineage.parameters.log_theta) if is_nb else ()
    theta = float(np.exp(log_theta[0])) if is_nb else None
    projection = summary = None
    response_sum = 0.0
    n_rows = positive = batches = 0
    for batch, y, prior, _offset, valid in _source_batches(stream, family, lineage, B):
        if len(y) > B:
            raise ValueError("EFS startup source exceeds prospective batch cap")
        response = family.execution_initial_response_cpu(y, prior)
        if not np.all(np.isfinite(response)):
            raise ValueError("EFS startup initialized response is invalid")
        X = prepared.evaluate_batch(batch) if len(y) else np.empty((0, p))
        if not np.all(np.isfinite(X)):
            raise ValueError("EFS startup PUBLIC design is nonfinite")
        projection = qr_update(projection, X, np.ones(len(y)), n_coef=p)
        if len(y):
            item = family.execution_summary_from_batch(response, prior, valid)
            summary = (
                item
                if summary is None
                else family.merge_execution_summaries(summary, item)
            )
        response_sum += float(np.sum(response))
        n_rows += len(response)
        positive += int(np.count_nonzero(prior > 0))
        batches += 1
    if n_rows != prepared.n_obs or not positive:
        raise ValueError("EFS startup requires a globally informative source")
    metadata = family.finalize_execution_summary(summary)
    if not metadata["input_ok"]:
        raise ValueError("EFS startup global summary is invalid")
    response_mean = response_sum / n_rows
    constant_eta = float(family.link.initial_link_cpu(np.asarray(response_mean)))
    if not np.isfinite(constant_eta):
        raise ValueError("EFS startup requires a finite null link mean")
    null = project_null_coefficients(projection, constant_eta).coefficients.copy()
    # Release the public QR before the second scan and device transfer.
    del projection, X, batch
    for block in prepared.fitting.penalty_structure.blocks:
        where = slice(block.start, block.stop)
        null[where] = np.linalg.solve(block.transform.dense(), null[where])
    observed = np.zeros(p)
    fisher = np.zeros(p)
    negative_observed = False
    observed_ok = fisher_ok = True
    null_deviance = 0.0
    for batch, y, prior, _offset, valid in _source_batches(stream, family, lineage, B):
        if len(y) > B:
            raise ValueError("EFS startup source exceeds prospective batch cap")
        initial = family.initial_working_state_cpu(y, prior, valid, summary=metadata)
        if not initial.input_ok or not initial.domain_ok:
            raise ValueError("EFS startup mustart predictor is invalid in pinned R")
        response = family.execution_initial_response_cpu(y, prior)
        mu, eta = initial.mustart, initial.eta
        derivative = np.asarray(family.link.mu_eta(eta))
        expected = prior * derivative**2 / np.asarray(family.variance(mu))
        if is_nb:
            dmu2 = (
                -2 * prior * ((response + theta) / (mu + theta) ** 2 - response / mu**2)
            )
            working = 0.5 * dmu2 * derivative**2
        else:
            working = expected
        negative_observed |= bool(np.any(working < 0))
        observed_ok &= bool(np.all(np.isfinite(working)))
        fisher_ok &= bool(np.all(np.isfinite(expected) & (expected >= 0)))
        X = prepared.evaluate_batch(batch) if len(y) else np.empty((0, p))
        if not np.all(np.isfinite(X)):
            raise ValueError("EFS startup PUBLIC design is nonfinite")
        # initial.spg passes sqrt(w)*X to initial.sp. Negative observed
        # entries are never selected: their global presence selects Fisher.
        weighted = np.sqrt(np.maximum(working, 0))[:, None] * X
        observed += np.sum(weighted * weighted, axis=0)
        weighted = np.sqrt(expected)[:, None] * X
        fisher += np.sum(weighted * weighted, axis=0)
        null_deviance += (
            _nb_null_deviance(response, response_mean, prior, theta)
            if is_nb
            else float(
                family.dev_resids(response, np.full(len(y), response_mean), prior)
            )
        )
        batches += 1
    expected_selected = is_nb and negative_observed
    diagonal = fisher if expected_selected else observed
    if (
        not (fisher_ok if expected_selected else observed_ok)
        or (negative_observed and not is_nb)
        or not np.all(np.isfinite(diagonal))
        or not np.any(diagonal > 0)
    ):
        raise ValueError("EFS startup has no finite informative initial.spg diagonal")
    rho = initial_log_sp_from_diagonal(diagonal, prepared.penalties)
    phi = 1.0 if family.scale_known else null_deviance / n_rows / 10
    if not np.all(np.isfinite(rho)) or not np.isfinite(phi) or phi <= 0:
        raise ValueError("EFS startup requires finite positive incoming scale")
    return EFSStreamStartup(
        rho,
        null,
        phi,
        log_theta,
        null_deviance,
        "expected-Dmu2"
        if expected_selected
        else "observed-Dmu2"
        if is_nb
        else "regular",
        lineage.source_fingerprint,
        lineage.basis_fingerprint,
        2,
        batches,
        workspace,
    )
