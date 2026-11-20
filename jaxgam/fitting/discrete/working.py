"""Per-batch ``bgam.fitd`` working statistics on the compact discrete design.

Pinned mgcv 1.9-3 ``bgam.fitd`` (``R/bam.r`` lines 585-651) forms the
working model of a regular exponential family from IRLS/Fisher weights::

    mu.eta.val <- mu.eta(eta); good <- mu.eta.val != 0
    mu.eta.val[!good] <- .1
    z <- (eta - offset) + (y - mu)/mu.eta.val
    w <- (G$w * mu.eta.val^2)/variance(mu)

and accumulates ``y.norm2 = sum(w*z^2)``, ``XWXd`` and ``XWyd``. These are
the discrete method's semantics for every link: unlike ``gam.fit3`` there is
no observed (Newton) curvature, no weight clipping and no ``good`` row
filtering. The full-data ``sum(dev.resids)`` drives coefficient step
control; kernels return it per row so the host can reduce it in R's order.

Kernels here only evaluate one batch. Family arithmetic is owned by the
family/link objects; the host controller in ``execution/`` owns iteration,
step control and every cross-batch reduction.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting.discrete.ops import (
    DeviceDiscreteDesign,
    discrete_xbd,
    discrete_xwxd,
    discrete_xwyd,
)

jax.config.update("jax_enable_x64", True)


class DiscreteWorkingBatch(NamedTuple):
    """One batch's deviance rows and Fisher working-model cross-products.

    ``deviance_rows`` are pinned ``dev.resids`` per row with R's arithmetic
    outside the family domain (``nan`` where R's formula is undefined, for
    example a log of a negative ratio; finite where it is defined, such as a
    nonpositive Poisson mean at a zero count). ``bgam.fitd`` checks neither
    ``validmu`` nor ``valideta`` between iterations, so its step control
    sees exactly this ``sum(dev.resids)``. The host reduces the rows in row
    order, as R's ``sum`` does. ``gram``/``rhs``/
    ``response_square_sum`` are ``X'WX``, ``X'Wz`` and ``sum(w*z^2)``; they
    are zero-sized when only the deviance is requested.
    """

    deviance_rows: jax.Array
    gram: jax.Array
    rhs: jax.Array
    response_square_sum: jax.Array


def _fisher_working_quantities(
    family: ExponentialFamily,
    eta: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``mu`` and ``bgam.fitd``'s Fisher ``w``/``z`` for one batch."""
    mu = family.link.inverse(eta)
    mu_eta = family.link.mu_eta(eta)
    # bgam.fitd replaces a zero mu.eta by 0.1 *before* forming both w and z.
    mu_eta = jnp.where(mu_eta != 0.0, mu_eta, 0.1)
    response = (eta - offset) + (y - mu) / mu_eta
    weight = (prior_weight * mu_eta**2) / family.variance(mu)
    return mu, weight, response


def _working_batch(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    eta: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    family: ExponentialFamily,
    statistics: bool,
) -> DiscreteWorkingBatch:
    if not (eta.shape == y.shape == prior_weight.shape == offset.shape):
        raise ValueError("eta, y, prior weights and offset must share one shape.")
    mu, weight, response = _fisher_working_quantities(
        family, eta, y, prior_weight, offset
    )
    # Pinned dev.resids arithmetic, with no validmu/valideta mask: R checks
    # neither between iterations, so only a non-finite sum stops a step.
    deviance = family.source_dev_resids(y, mu, prior_weight)
    if not statistics:
        empty = jnp.zeros((0,), dtype=eta.dtype)
        return DiscreteWorkingBatch(
            deviance, empty.reshape(0, 0), empty, jnp.zeros((), eta.dtype)
        )
    return DiscreteWorkingBatch(
        deviance,
        discrete_xwxd(design, selectors, weight),
        discrete_xwyd(design, selectors, weight, response),
        jnp.sum(weight * response**2),
    )


@partial(jax.jit, static_argnames=("family", "statistics"))
def discrete_working_batch(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    coefficients: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    *,
    family: ExponentialFamily,
    statistics: bool = True,
) -> DiscreteWorkingBatch:
    """Evaluate one batch at ``eta = Xbd(coefficients) + offset``.

    ``statistics=False`` evaluates only the deviance (a rejected or final
    coefficient trial needs no cross-products).
    """
    eta = discrete_xbd(design, selectors, coefficients) + offset
    return _working_batch(
        design, selectors, eta, y, prior_weight, offset, family, statistics
    )


@partial(jax.jit, static_argnames=("family",))
def discrete_initial_working_batch(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    eta: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    *,
    family: ExponentialFamily,
) -> DiscreteWorkingBatch:
    """Evaluate one batch at a supplied full predictor ``linkfun(mustart)``.

    ``bgam.fitd``'s first iteration has no coefficient vector: its predictor
    is the family initializer's ``mustart`` mapped through the link. The
    supplied ``eta`` already includes any offset.
    """
    return _working_batch(design, selectors, eta, y, prior_weight, offset, family, True)


# --------------------------------------------------------------------------
# Extended family (NB): bgam.fitd's dDeta working model at explicit theta.
# --------------------------------------------------------------------------


class DiscreteThetaBatch(NamedTuple):
    """One batch's rows for ``estimate.theta``'s ``nlogl`` at fixed ``mu``.

    ``deviance_rows`` are ``dev.resids``; ``theta_rows``/``theta2_rows`` are
    ``Dd$Dth``/``Dd$Dth2``; ``ls_rows``, ``lsth1_rows`` and ``lsth2_rows`` are
    the summands of ``ls``'s ``-sum``, ``sum`` and ``-sum``. Derivative rows
    are zero-sized when only the objective is requested.
    """

    deviance_rows: jax.Array
    theta_rows: jax.Array
    theta2_rows: jax.Array
    ls_rows: jax.Array
    lsth1_rows: jax.Array
    lsth2_rows: jax.Array


def _extended_working_batch(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    eta: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    theta: jax.Array,
    family: ExponentialFamily,
    statistics: bool,
) -> DiscreteWorkingBatch:
    if not (eta.shape == y.shape == prior_weight.shape == offset.shape):
        raise ValueError("eta, y, prior weights and offset must share one shape.")
    mu = family.link.inverse(eta)
    deviance = family.source_dev_resids_at(y, mu, prior_weight, theta)
    if not statistics:
        empty = jnp.zeros((0,), dtype=eta.dtype)
        return DiscreteWorkingBatch(
            deviance, empty.reshape(0, 0), empty, jnp.zeros((), eta.dtype)
        )
    weight, response = family.source_newton_working(
        y, mu, eta, offset, prior_weight, theta
    )
    return DiscreteWorkingBatch(
        deviance,
        discrete_xwxd(design, selectors, weight),
        discrete_xwyd(design, selectors, weight, response),
        jnp.sum(weight * response**2),
    )


@partial(jax.jit, static_argnames=("family", "statistics"))
def discrete_extended_working_batch(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    coefficients: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    theta: jax.Array,
    *,
    family: ExponentialFamily,
    statistics: bool = True,
) -> DiscreteWorkingBatch:
    """Extended-family batch at ``eta = Xbd(coefficients) + offset``.

    ``theta`` is the natural-scale family parameter (``exp(log theta)``
    formed on the host). Working weights are ``bgam.fitd``'s ``dDeta``
    Newton weights ``Deta2/2`` (``rho == 0``), not Fisher weights.
    """
    eta = discrete_xbd(design, selectors, coefficients) + offset
    return _extended_working_batch(
        design, selectors, eta, y, prior_weight, offset, theta, family, statistics
    )


@partial(jax.jit, static_argnames=("family",))
def discrete_extended_initial_working_batch(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    eta: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    theta: jax.Array,
    *,
    family: ExponentialFamily,
) -> DiscreteWorkingBatch:
    """Extended-family batch at ``eta = linkfun(mustart)`` (iteration one)."""
    return _extended_working_batch(
        design, selectors, eta, y, prior_weight, offset, theta, family, True
    )


@partial(jax.jit, static_argnames=("family", "derivatives"))
def discrete_theta_batch(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    coefficients: jax.Array,
    y: jax.Array,
    prior_weight: jax.Array,
    offset: jax.Array,
    log_theta: jax.Array,
    theta: jax.Array,
    *,
    family: ExponentialFamily,
    derivatives: bool = True,
) -> DiscreteThetaBatch:
    """``estimate.theta``'s per-row ``nlogl`` terms at fixed coefficients.

    ``derivatives=False`` is ``nlogl(deriv=0)``: deviance and ``ls`` only.
    """
    eta = discrete_xbd(design, selectors, coefficients) + offset
    mu = family.link.inverse(eta)
    deviance = family.source_dev_resids_at(y, mu, prior_weight, theta)
    ls_rows, lsth1_rows, lsth2_rows = family.source_ls_rows(
        y, prior_weight, log_theta, theta
    )
    if not derivatives:
        empty = jnp.zeros((0,), dtype=eta.dtype)
        return DiscreteThetaBatch(deviance, empty, empty, ls_rows, empty, empty)
    terms = family.source_dd(y, mu, prior_weight, theta)
    return DiscreteThetaBatch(
        deviance, terms.theta, terms.theta2, ls_rows, lsth1_rows, lsth2_rows
    )
