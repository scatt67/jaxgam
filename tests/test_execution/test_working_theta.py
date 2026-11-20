"""Host port of pinned efam.r::estimate.theta for one NB theta."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jaxgam.execution.discrete.regular_working_model import source_sum
from jaxgam.execution.discrete.working_theta import (
    ThetaObjective,
    source_estimate_theta,
)
from jaxgam.families.negative_binomial import NegativeBinomial
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


class _Quadratic:
    """nll = a (t - t0)^2 / 2 + c, recording every evaluation."""

    def __init__(self, curvature, minimum, offset=10.0, hessian=None):
        self.curvature = curvature
        self.minimum = minimum
        self.offset = offset
        self.hessian = curvature if hessian is None else hessian
        self.calls: list[tuple[float, bool]] = []

    def __call__(self, theta, derivatives):
        self.calls.append((theta, derivatives))
        nll = self.curvature * (theta - self.minimum) ** 2 / 2 + self.offset
        if not derivatives:
            return ThetaObjective(nll, np.nan, np.nan)
        return ThetaObjective(
            nll, self.curvature * (theta - self.minimum), self.hessian
        )


def test_newton_converges_and_counts_scans():
    objective = _Quadratic(2.0, 1.25)
    update = source_estimate_theta(0.0, objective)
    assert update.log_theta == 1.25
    assert update.newton_iterations == 1
    assert update.evaluations == len(objective.calls) == 2
    assert not update.step_failed


def test_already_converged_makes_no_step():
    objective = _Quadratic(2.0, 1e-9)
    update = source_estimate_theta(0.0, objective)
    assert update.log_theta == 0.0
    assert update.newton_iterations == 0
    assert update.evaluations == 1


def test_step_is_capped_at_four_and_hessian_made_positive():
    # A wrong-signed curvature: R uses |H|; the step towards 40 is capped at 4.
    objective = _Quadratic(1.0, 40.0, hessian=-1.0)
    update = source_estimate_theta(0.0, objective)
    first_trial = objective.calls[1][0]
    assert first_trial == 4.0
    assert update.log_theta > 4.0


def test_rising_trial_is_halved_then_failure_warns_and_keeps_theta():
    # The gradient claims a decrease, but the objective rises linearly in
    # every direction: 25 halvings (objective-only scans) cannot recover it.
    calls = []

    def objective(theta, derivatives):
        calls.append(derivatives)
        nll = 10.0 + 1e3 * abs(theta)
        if not derivatives:
            return ThetaObjective(nll, np.nan, np.nan)
        return ThetaObjective(nll, -1.0, 1.0)

    with pytest.warns(RuntimeWarning, match="step failure in theta estimation"):
        update = source_estimate_theta(0.0, objective)
    assert update.step_failed
    assert update.log_theta == 0.0
    assert calls.count(False) == 25


def test_non_finite_objective_fails_like_r():
    def objective(_theta, _derivatives):
        return ThetaObjective(np.nan, np.nan, np.nan)

    with pytest.raises(FloatingPointError, match=r"estimate\.theta objective"):
        source_estimate_theta(0.0, objective)

    def infinite_step(theta, _derivatives):
        return ThetaObjective(1.0 + theta, 1.0, 0.0)

    with pytest.raises(FloatingPointError, match="theta Newton step"):
        source_estimate_theta(0.0, infinite_step)


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_estimate_theta_matches_pinned_r(r_bridge, link):
    """Same y, mu, weights and start: pinned R's estimate.theta at STRICT.

    The objective uses the family's pinned ``nb()`` arithmetic with R's own
    sums; starts below, near and above the optimum take different Newton
    paths, which R reproduces to its own stopping point.
    """
    rng = np.random.default_rng(3)
    n = 60
    family = NegativeBinomial(theta=2.0, link=link)
    mu = rng.uniform(0.5, 9.0, n)
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
    weight = np.linspace(0.5, 1.5, n)
    weight[7] = 0.0

    def nlogl(log_theta, derivatives):
        theta = math.exp(log_theta)
        deviance = float(source_sum(family.source_dev_resids_at(y, mu, weight, theta)))
        ls_rows, first, second = family.source_ls_rows(y, weight, log_theta, theta)
        nll = deviance / 2.0 + float(source_sum(ls_rows))
        if not derivatives:
            return ThetaObjective(nll, np.nan, np.nan)
        terms = family.source_dd(y, mu, weight, theta)
        return ThetaObjective(
            nll,
            float(source_sum(terms.theta)) / 2.0 - float(source_sum(first)),
            float(source_sum(terms.theta2)) / 2.0 + float(source_sum(second)),
        )

    collector = _AssertCollector()
    for start in (-1.0, 0.0, 2.5):
        ours = source_estimate_theta(start, nlogl)
        expected = r_bridge.nb_estimate_theta(y, mu, weight, start, link=link)
        collector.check(
            f"start {start}",
            lambda ours=ours, expected=expected: np.testing.assert_allclose(
                ours.log_theta, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any(f"nb/{link} estimate.theta")
