"""Explicit-theta NB batch factors for streamed coefficient fitting.

Pinned source: mgcv 1.9-3 ``nb$Dd``, ``dDeta`` and ``gam.fit4``. These
are coefficient/working-system kernels, not conditional-theta or joint REML
objectives. Hosts own scans and global direct-response selection. Nothing
here prepares or retains an entire response or a count-prefix table.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxgam.fitting.family_execution import FamilyExecutionParameters
from jaxgam.jax_utils import _materialize_source_operation as rounded


class NBWorkingBatch(NamedTuple):
    """Bounded row factors; raw pseudo-responses may be nonfinite.

    ``requires_direct_response`` is OR-reduced over the whole scan before
    selecting ``normal_rows`` versus ``direct_rows``. A finite direct RHS
    remains usable at zero curvature. Fisher factors are separate and must
    never stand in for gam.fit4's positive-observed retry.
    """

    eta: jax.Array
    mu: jax.Array
    eta_minus_offset: jax.Array
    observed_weight: jax.Array
    fisher_weight: jax.Array
    response: jax.Array
    weighted_response: jax.Array
    derivative_rhs: jax.Array
    valid_rows: jax.Array
    normal_rows: jax.Array
    direct_rows: jax.Array
    requires_direct_response: jax.Array
    domain_ok: jax.Array
    deviance: jax.Array
    log_theta: jax.Array


class NBWorkingSummary(NamedTuple):
    """Constant-size scan metadata, independent of response/count length."""

    requires_direct_response: jax.Array
    domain_ok: jax.Array
    normal_count: jax.Array
    direct_count: jax.Array
    deviance: jax.Array


def nb_working_batch(
    eta: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    valid: jax.Array,
    parameters: FamilyExecutionParameters,
    *,
    link: str,
) -> NBWorkingBatch:
    """Evaluate fixed or trial theta without reading a mutable family.

    ``eta`` is the actual frozen working predictor, including offsets; it
    need not equal the null-coefficient predictor. Padding is sanitized
    before arithmetic. Fractional responses use nb's literal pmax(1,y)
    deviance convention, separately from saturated-likelihood conventions.
    """
    if link not in ("log", "identity", "sqrt"):
        raise NotImplementedError("streamed NB supports log, identity, and sqrt")
    if parameters.log_theta.shape != (1,):
        raise ValueError("NB log_theta must have shape (1,)")
    if eta.ndim != 1 or any(
        value.shape != eta.shape for value in (y, prior_weight, offset, valid)
    ):
        raise ValueError("NB batch vectors must be aligned and one-dimensional")
    theta_raw = jnp.exp(parameters.log_theta[0])
    theta_ok = jnp.isfinite(theta_raw) & (theta_raw > 0)
    theta = jnp.where(theta_ok, theta_raw, 1.0)
    eta_safe = jnp.where(valid & jnp.isfinite(eta), eta, 1.0)
    mu_raw = (
        jnp.exp(eta_safe)
        if link == "log"
        else eta_safe**2
        if link == "sqrt"
        else eta_safe
    )
    row_ok = (
        jnp.isfinite(eta)
        & jnp.isfinite(offset)
        & jnp.isfinite(y)
        & (y >= 0)
        & jnp.isfinite(prior_weight)
        & (prior_weight >= 0)
        & jnp.isfinite(mu_raw)
        & (mu_raw > 0)
    )
    active = valid & row_ok & (prior_weight > 0)
    mu = jnp.where(active, mu_raw, 1.0)
    response_y = jnp.where(active, y, 0.0)
    wt = jnp.where(active, prior_weight, 0.0)
    eta_work = jnp.where(active, eta_safe, 0.0)
    offset_work = jnp.where(active, offset, 0.0)
    y_theta = response_y + theta
    mu_theta = mu + theta
    dmu = rounded(
        rounded(2 * wt)
        * rounded(rounded(y_theta / mu_theta) - rounded(response_y / mu))
    )
    dmu2 = rounded(
        rounded(-2 * wt)
        * rounded(
            rounded(y_theta / rounded(mu_theta**2))
            - rounded(response_y / rounded(mu**2))
        )
    )
    expected = rounded(
        rounded(2 * wt) * rounded(rounded(1 / mu) - rounded(1 / mu_theta))
    )
    if link == "identity":
        d1, d2, ed2 = dmu, dmu2, expected
        derivative_ratio = dmu / dmu2
    else:
        # dDeta uses mu.eta(linkfun(mu)), including sqrt's positive root.
        ig1 = mu if link == "log" else 2 * jnp.sqrt(mu)
        g2g = -1.0 if link == "log" else -1 / jnp.sqrt(mu)
        d1 = rounded(dmu * ig1)
        d2 = rounded(
            rounded(dmu2 * rounded(ig1**2)) - rounded(rounded(dmu * g2g) * ig1)
        )
        ed2 = rounded(expected * rounded(ig1**2))
        derivative_ratio = dmu / rounded(rounded(dmu2 * ig1) - rounded(dmu * g2g))
    weight = 0.5 * d2
    fisher = 0.5 * ed2
    derivative_rhs = -0.5 * d1
    wz = weight * (eta_work - offset_work) + derivative_rhs
    z = eta_work - offset_work - derivative_ratio
    normal = valid & row_ok & jnp.isfinite(weight) & jnp.isfinite(z)
    direct = valid & row_ok & jnp.isfinite(weight) & jnp.isfinite(wz)
    requires_direct = jnp.any(valid & ~normal)
    ratio = (response_y - mu) / mu_theta
    log_ratio = jnp.where(
        ratio > -0.5,
        jnp.log1p(jnp.maximum(ratio, -0.5)),
        jnp.log(y_theta) - jnp.log(mu_theta),
    )
    dev = (
        2
        * wt
        * (
            response_y * (jnp.log(jnp.maximum(1.0, response_y)) - jnp.log(mu))
            - y_theta * log_ratio
        )
    )
    return NBWorkingBatch(
        eta_work,
        jnp.where(valid & row_ok, mu_raw, 1.0),
        eta_work - offset_work,
        weight,
        fisher,
        z,
        wz,
        derivative_rhs,
        valid & row_ok,
        normal,
        direct,
        requires_direct,
        theta_ok & jnp.all(~valid | row_ok),
        jnp.sum(dev),
        parameters.log_theta,
    )


def nb_positive_observed_retry(batch: NBWorkingBatch) -> NBWorkingBatch:
    """gam.fit4 recovery: zero nonpositive observed W, retain derivative RHS."""
    weight = jnp.where(
        jnp.isfinite(batch.observed_weight) & (batch.observed_weight > 0),
        batch.observed_weight,
        0.0,
    )
    wz = weight * batch.eta_minus_offset + batch.derivative_rhs
    normal = batch.valid_rows & jnp.isfinite(batch.response)
    direct = batch.valid_rows & jnp.isfinite(wz)
    return batch._replace(
        observed_weight=weight,
        weighted_response=wz,
        normal_rows=normal,
        direct_rows=direct,
        requires_direct_response=jnp.any(batch.valid_rows & ~normal),
    )


def nb_selected_working_rows(
    batch: NBWorkingBatch,
    *,
    use_weighted_response: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Finite signed-QR inputs after the scan-wide direct-response decision."""
    good = jnp.where(use_weighted_response, batch.direct_rows, batch.normal_rows)
    return (
        jnp.where(good, batch.observed_weight, 0.0),
        jnp.where(good & jnp.isfinite(batch.response), batch.response, 0.0),
        jnp.where(good, batch.weighted_response, 0.0),
    )


def nb_working_summary(batch: NBWorkingBatch) -> NBWorkingSummary:
    """Informative counts ignore zero curvature but retain direct RHS rows."""
    informative = batch.observed_weight != 0
    return NBWorkingSummary(
        batch.requires_direct_response,
        batch.domain_ok,
        jnp.sum(batch.normal_rows & informative),
        jnp.sum(batch.direct_rows & informative),
        batch.deviance,
    )


def merge_nb_working_summaries(
    left: NBWorkingSummary,
    right: NBWorkingSummary,
) -> NBWorkingSummary:
    """Merge without retaining batches; host decides recovery after the scan."""
    return NBWorkingSummary(
        left.requires_direct_response | right.requires_direct_response,
        left.domain_ok & right.domain_ok,
        left.normal_count + right.normal_count,
        left.direct_count + right.direct_count,
        left.deviance + right.deviance,
    )


def nb_working_statistics(
    X: jax.Array,
    batch: NBWorkingBatch,
    *,
    use_weighted_response: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Small signed G/RHS for an already globally selected response policy."""
    good = jnp.where(use_weighted_response, batch.direct_rows, batch.normal_rows)
    weight = jnp.where(good, batch.observed_weight, 0.0)
    rhs = jnp.where(
        use_weighted_response,
        batch.weighted_response,
        weight * jnp.where(good, batch.response, 0.0),
    )
    safe_X = jnp.where(good[:, None], X, 0.0)
    return safe_X.T @ (weight[:, None] * safe_X), safe_X.T @ jnp.where(good, rhs, 0)
