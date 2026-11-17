"""Negative binomial bgam.fitd performance iteration on the discrete route.

Pinned mgcv 1.9-3 ``bam(discrete=TRUE)`` fits ``nb()`` by ``bgam.fitd``'s
extended-family branch: ``dDeta`` Newton working weights (``rho == 0``),
``dev.resids`` at the current theta for step control, and, when theta is
estimated, ``estimate.theta`` at the step-controlled coefficients of every
iteration from two, before the working model is formed. Gates:

* fixed sp on the identical explicit design (the same controller and
  ``estimate.theta`` port on dense rows), every field and theta STRICT, and
  for a fixed theta the fixed point of JaxGAM's dense PIRLS;
* fixed sp and working-model REML against pinned ``bam`` for the log,
  identity and sqrt links, theta fixed and estimated, every field, theta and
  smoothing parameters STRICT;
* matched-state oracles: every theta update, working model, ``Sl.fitChol``
  trial and controller decision from pinned R's exact state;
* the identity-link domain exit both implementations stop at.

Step-control ties follow the policy of ``test_discrete_regular.py``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution import regular_working_model as regular
from jaxgam.execution.discrete_regular import (
    fit_discrete_regular,
    regular_working_components,
)
from jaxgam.execution.working_theta import ThetaObjective, source_estimate_theta
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.initialization import initialize_beta
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector, check_that
from tests.test_execution.test_discrete_regular import (
    _SEPARATE,
    _TENSOR_BY,
    _check_public_fit,
    _close,
    _control,
    _follow,
    _label,
    _our_decisions,
    _overrides,
    _RecordingRows,
    _snapped_design,
    _source,
    _source_decisions,
    check_matched_state,
)
from tests.tolerances import STRICT

_LEVELS = ("a", "b", "c", "d")
_THETA = {"fixed": 2.5, "estimated": 1.5}
_SP = {
    _SEPARATE: np.array([0.6, 1.5, 0.8]),
    _TENSOR_BY: np.array([0.5, 1.2, 0.7, 1.1, 0.9, 1.6]),
}
# The identity link's coefficient scale needs firmer fixed smoothing: with the
# separate-smooth defaults the first proposal leaves the NB domain.
_IDENTITY_SP = np.array([5.0, 10.0, 1.0])
_CASES = [
    pytest.param(link, _SEPARATE, id=f"{link}-separate_re")
    for link in ("log", "identity", "sqrt")
] + [pytest.param("log", _TENSOR_BY, id="log-tensor_by")]
_MODES = ("fixed", "estimated")


def _nb_frame(link: str, n: int = 160, seed: int = 5, theta: float = 3.0):
    """Runtime NB counts with weights, an offset and well-identified smooths."""
    position = np.arange(n, dtype=float)
    x = (position + 0.2 * np.sin(position * 0.7)) / n
    z = np.cos(position * 0.23) + 0.07 * np.sin(position * 0.91)
    w = np.sin(position * 0.37) + 0.5 * position / n
    g = pd.Categorical(np.resize(np.array(_LEVELS), n), categories=_LEVELS)
    code = np.asarray(g.codes)
    eta = (
        0.3
        + 1.2 * np.sin(5.0 * x)
        + 0.7 * np.cos(3.0 * z)
        + 0.6 * np.array([-1.0, 0.4, 1.0, -0.6])[code]
    )
    mean = 3.0 * np.exp(eta) + 6.0 if link == "identity" else np.exp(eta)
    rng = np.random.default_rng(seed)
    y = rng.negative_binomial(theta, theta / (theta + mean)).astype(float)
    weights = np.linspace(0.6, 1.4, n)
    offset = np.linspace(-0.1, 0.2, n)
    frame = pd.DataFrame({"x": x, "z": z, "w": w, "g": g, "y": y})
    return frame, weights, offset


def _nb_family(link: str, mode: str) -> NegativeBinomial:
    return NegativeBinomial(theta=_THETA[mode], fixed=mode == "fixed", link=link)


def _fixed_sp(link: str, formula: str) -> np.ndarray:
    return _IDENTITY_SP if link == "identity" else _SP[formula]


def _trace(r_bridge, link, formula, mode, sp, frame, weights, offset, **kwargs):
    return r_bridge.trace_bam_regular_working_model(
        formula,
        frame,
        family="nb",
        link=link,
        sp=sp,
        factor_names=("g",),
        weights=weights,
        offset=offset,
        theta=_THETA[mode],
        theta_fixed=mode == "fixed",
        **kwargs,
    )


# --------------------------------------------------------------------------
# Gate 1: fixed sp on the identical explicit design.
# --------------------------------------------------------------------------


class _ExplicitNBRows:
    """NB ``bgam.fitd`` rows on the explicit snapped design (NumPy).

    The same pinned ``nb()`` arithmetic (``dev.resids``, ``dDeta`` weights,
    ``Dd``/``ls`` for ``estimate.theta``) on dense ``X @ beta``.
    """

    def __init__(self, X, frame, weights, offset, family):
        self.X = X
        self.y = frame.y.to_numpy()
        self.weights = weights
        self.offset = offset
        self.family = family
        self.set_log_theta(float(family.get_theta()[0]))

    def set_log_theta(self, log_theta):
        self.log_theta = float(log_theta)
        self.theta = math.exp(self.log_theta)

    def _evaluate(self, eta, statistics):
        family = self.family
        mu = np.asarray(family.link.inverse(eta))
        with np.errstate(divide="ignore", invalid="ignore"):
            deviance = family.source_dev_resids_at(self.y, mu, self.weights, self.theta)
            if not statistics:
                return regular.WorkingEvaluation(
                    float(regular.source_sum(deviance)), None
                )
            W, z = family.source_newton_working(
                self.y, mu, eta, self.offset, self.weights, self.theta
            )
        return regular.WorkingEvaluation(
            float(regular.source_sum(deviance)),
            regular.WorkingStatistics(
                self.X.T @ (W[:, None] * self.X),
                self.X.T @ (W * z),
                float(np.sum(W * z**2)),
            ),
        )

    def initial(self):
        state = self.family.initial_working_state_cpu(
            self.y, self.weights, np.ones(len(self.y), dtype=bool)
        )
        return self._evaluate(np.asarray(state.eta), True)

    def evaluate(self, coefficients, *, statistics=True):
        return self._evaluate(self.X @ coefficients + self.offset, statistics)

    def response_variance(self):
        return float(np.var(self.y, ddof=1))

    def objective(self, coefficients, log_theta, derivatives):
        family = self.family
        theta = math.exp(log_theta)
        mu = np.asarray(family.link.inverse(self.X @ coefficients + self.offset))
        deviance = float(
            regular.source_sum(
                family.source_dev_resids_at(self.y, mu, self.weights, theta)
            )
        )
        ls_rows, first_rows, second_rows = family.source_ls_rows(
            self.y, self.weights, log_theta, theta
        )
        nll = deviance / 2.0 + float(regular.source_sum(ls_rows))
        if not derivatives:
            return ThetaObjective(nll, np.nan, np.nan)
        terms = family.source_dd(self.y, mu, self.weights, theta)
        return ThetaObjective(
            nll,
            float(regular.source_sum(terms.theta)) / 2.0
            - float(regular.source_sum(first_rows)),
            float(regular.source_sum(terms.theta2)) / 2.0
            + float(regular.source_sum(second_rows)),
        )


class _ExplicitTheta:
    def __init__(self, rows):
        self.rows = rows

    def update(self, coefficients):
        beta = np.array(coefficients, dtype=float)
        update = source_estimate_theta(
            self.rows.log_theta,
            lambda log_theta, derivatives: self.rows.objective(
                beta, log_theta, derivatives
            ),
        )
        self.rows.set_log_theta(update.log_theta)
        return update


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(("link", "formula"), _CASES)
@pytest.mark.platform_sensitive
def test_fixed_sp_nb_matches_the_identical_explicit_design(link, formula, mode):
    """The discrete NB fit is the explicit-design ``bgam.fitd`` NB fit.

    The same controller, ``Sl.fitChol`` kernel and ``estimate.theta`` port,
    driven by dense rows with the same pinned ``nb()`` arithmetic, give the
    discrete fit's coefficients, fitted values, deviance, EDF, score and
    theta at STRICT, and every discrete working model equals the explicit
    ``X'WX``/``X'Wz`` with ``dDeta`` weights at that iteration's theta. For a
    fixed theta, the discrete fit iterated to its rounding floor reaches the
    penalized deviance of JaxGAM's dense PIRLS fixed point.
    """
    frame, weights, offset = _nb_frame(link)
    sp = _fixed_sp(link, formula)
    fam = _nb_family(link, mode)
    X, penalties = _snapped_design(formula, frame)
    explicit_rows = _ExplicitNBRows(X, frame, weights, offset, fam)
    explicit_theta = _ExplicitTheta(explicit_rows) if mode == "estimated" else None
    collector = _AssertCollector()
    with regular_working_components(
        parse_formula(formula),
        _source(frame, weights, offset),
        fam,
        sp,
        control=_control(False),
    ) as parts:
        explicit_fit = parts.run(explicit_rows, nuisance=explicit_theta)
        explicit = parts.result(explicit_fit, log_theta=explicit_rows.log_theta)
        recording = _RecordingRows(parts.rows)
        thetas = []
        final_theta = {}

        def run(replay):
            parts.rows.set_log_theta(float(fam.get_theta()[0]))
            recording.records.clear()
            thetas.clear()
            recorded_nuisance = None
            if parts.nuisance is not None:

                class _Recorded:
                    def update(self, coefficients):
                        update = parts.nuisance.update(coefficients)
                        thetas.append(update.log_theta)
                        return update

                recorded_nuisance = _Recorded()
            fit = parts.run(
                recording,
                nuisance=recorded_nuisance,
                _oracle_replay_halvings=replay,
            )
            final_theta[id(fit)] = parts.rows.log_theta
            return fit

        followed = _follow(run, _our_decisions(explicit_fit))
        discrete = parts.result(followed.fit, log_theta=final_theta[id(followed.fit)])
        public = (
            None
            if followed.public is None
            else parts.result(
                followed.public, log_theta=final_theta[id(followed.public)]
            )
        )
        floor = None
        # JaxGAM's dense pirls_loop does not converge for NB/sqrt at this fixed
        # theta (its penalized deviance still moves after 2000 iterations), so
        # the dense fixed point is compared for the log and identity links.
        if mode == "fixed" and link != "sqrt":
            floor = regular.fit_regular_working_model(
                parts.rows,
                parts.propose,
                parts.penalty,
                X.shape[1],
                scale_estimated=False,
                epsilon=1e-300,
                max_iterations=100,
                start=parts.start,
            )
    collector.check(
        "decisions",
        lambda: check_that(not followed.problems, "; ".join(followed.problems)),
    )
    collector.check(
        "replay bound",
        lambda: check_that(
            len(followed.replayed)
            <= _REVIEWED_NB_TIE_REPLAYS.get(
                ("explicit", f"{link}-{_label(formula)}-{mode}"), 0
            ),
            f"{len(followed.replayed)} replayed decisions",
        ),
    )

    def fields(fit):
        return [
            ("coefficients", fit.coefficients, explicit.coefficients),
            (
                "fitted values",
                fam.link.inverse(X @ fit.coefficients + offset),
                fam.link.inverse(X @ explicit.coefficients + offset),
            ),
            ("deviance", fit.deviance, explicit.deviance),
            ("edf", fit.edf_total, explicit.edf_total),
            ("score", fit.score, explicit.score),
            ("theta", math.exp(fit.log_theta), math.exp(explicit.log_theta)),
        ]

    for name, observed, expected in fields(discrete):
        collector.check(name, _close(observed, expected, STRICT))
    _check_public_fit(collector, replace(followed, public=public), fields)
    # Every discrete working model, at the theta it was formed with.
    statistic_thetas = iter([float(fam.get_theta()[0]), *thetas])
    theta = next(statistic_thetas)
    for index, (beta, evaluation) in enumerate(recording.records):
        if evaluation.statistics is None:
            continue
        if beta is not None and mode == "estimated":
            theta = next(statistic_thetas)
        explicit_rows.set_log_theta(theta)
        expected = (
            explicit_rows.initial() if beta is None else explicit_rows.evaluate(beta)
        )
        for name, observed, target in (
            ("X'WX", evaluation.statistics.gram, expected.statistics.gram),
            ("X'Wz", evaluation.statistics.rhs, expected.statistics.rhs),
        ):
            collector.check(
                f"evaluation {index} {name}", _close(observed, target, STRICT)
            )
    if floor is not None:
        y = frame.y.to_numpy()
        S = penalties.materialize(np.log(sp))
        beta0 = initialize_beta(X, y, weights, fam, offset)
        dense = pirls_loop(
            X, y, beta0, S, fam, weights, offset, max_iter=400, tol=1e-14
        )
        collector.check(
            "shared fixed point",
            lambda: (
                check_that(bool(dense.converged), "dense PIRLS converges"),
                _close(
                    floor.penalized_deviance, float(dense.penalized_deviance), STRICT
                )(),
            ),
        )
    collector.raise_if_any(f"nb/{link} {mode} theta discrete vs explicit design")


# --------------------------------------------------------------------------
# Gate 2: end to end against pinned bam(discrete=TRUE, family=nb()).
# --------------------------------------------------------------------------

# Replayed step-control decisions per fit, as in test_discrete_regular.py:
# reviewed per-case maxima (larger of the macOS host and Linux arm64 counts).
_REVIEWED_NB_TIE_REPLAYS = {
    ("explicit", "log-separate_re-estimated"): 1,  # host 1, Linux 0
    ("fixed", "log-separate_re-estimated"): 1,  # host 1, Linux 0
    ("fixed", "identity-separate_re-estimated"): 2,  # host 2, Linux 0
}


@pytest.mark.parametrize("selected", [False, True], ids=["fixed_sp", "selected_sp"])
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(("link", "formula"), _CASES)
@pytest.mark.platform_sensitive
def test_nb_matches_pinned_bam(r_bridge, link, formula, mode, selected):
    frame, weights, offset = _nb_frame(link)
    sp = None if selected else _fixed_sp(link, formula)
    fam = _nb_family(link, mode)
    trace = _trace(r_bridge, link, formula, mode, sp, frame, weights, offset)

    def run(replay):
        return fit_discrete_regular(
            parse_formula(formula),
            _source(frame, weights, offset),
            fam,
            sp,
            control=_control(selected),
            _oracle_replay_halvings=replay,
        )

    followed = _follow(run, _source_decisions(trace))
    fit = followed.fit
    X, _ = _snapped_design(formula, frame)
    expected = trace.fit
    collector = _AssertCollector()
    key = ("selected" if selected else "fixed", f"{link}-{_label(formula)}-{mode}")
    bound = _REVIEWED_NB_TIE_REPLAYS.get(key, 0)
    collector.check(
        "decisions",
        lambda: check_that(not followed.problems, "; ".join(followed.problems)),
    )
    collector.check(
        "replay bound",
        lambda: check_that(
            len(followed.replayed) <= bound,
            f"{len(followed.replayed)} replayed decisions for {key}; bound {bound}",
        ),
    )
    collector.check(
        "iterations and convergence",
        lambda: check_that(
            fit.converged
            and expected.mgcv_converged
            and fit.scale_iterations == expected.n_iter,
            f"{fit.converged=} {fit.scale_iterations=} vs pinned {expected.n_iter}",
        ),
    )

    def fields(result):
        return [
            ("coefficients", result.coefficients, expected.coefficients),
            (
                "fitted values",
                np.asarray(fam.link.inverse(X @ result.coefficients + offset)),
                expected.fitted_values,
            ),
            ("deviance", result.deviance, expected.deviance),
            ("edf", result.edf_total, expected.edf_total),
            ("scale", result.scale, expected.scale),
            ("score", result.score, expected.score),
            ("theta", math.exp(result.log_theta), math.exp(expected.log_theta)),
        ]

    for name, observed, target in fields(fit):
        collector.check(name, _close(observed, target, STRICT))
    if selected:
        collector.check(
            "smoothing parameters",
            _close(fit.smoothing_params, expected.smoothing_params, STRICT),
        )
    _check_public_fit(collector, followed, fields)
    collector.raise_if_any(
        f"nb/{link} {mode} theta {'selected' if selected else 'fixed'} sp vs bam"
    )


# --------------------------------------------------------------------------
# Matched-state oracles.
# --------------------------------------------------------------------------

_MATCHED = {
    "log-separate-estimated-selected": ("log", _SEPARATE, "estimated", True),
    "log-separate-estimated-fixed": ("log", _SEPARATE, "estimated", False),
    "identity-separate-fixed-selected": ("identity", _SEPARATE, "fixed", True),
    "sqrt-separate-estimated-fixed": ("sqrt", _SEPARATE, "estimated", False),
    "log-tensor-estimated-selected": ("log", _TENSOR_BY, "estimated", True),
}
# No matched-state NB iteration needed a replay on the macOS host or Linux.
_REVIEWED_NB_MATCHED_REPLAYS: dict[str, int] = {}


@pytest.mark.parametrize("case", list(_MATCHED))
@pytest.mark.platform_sensitive
def test_nb_matched_state_theta_statistics_proposals_and_controller(r_bridge, case):
    """Every pinned NB iteration from R's exact state.

    Beyond the regular-family checks (working models, ``Sl.fitChol`` trials,
    controller replays through the converging iteration): each
    ``estimate.theta`` call is replayed from R's previous theta at R's
    step-controlled coefficients and must give R's theta, the working
    models use R's updated theta, and the final theta and the final deviance
    at it are R's.
    """
    link, formula, mode, selected = _MATCHED[case]
    frame, weights, offset = _nb_frame(link)
    sp = None if selected else _fixed_sp(link, formula)
    fam = _nb_family(link, mode)
    trace = _trace(r_bridge, link, formula, mode, sp, frame, weights, offset)
    collector = _AssertCollector()
    collector.check(
        "theta traced",
        lambda: check_that(
            all(trial.log_theta is not None for trial in trace.trials)
            and trace.final_log_theta is not None,
            "every NB trial reports theta",
        ),
    )
    with regular_working_components(
        parse_formula(formula),
        _source(frame, weights, offset),
        fam,
        sp,
        control=_control(selected),
    ) as parts:
        replayed = check_matched_state(
            r_bridge, collector, trace, parts, fam=fam, selected=selected
        )
    bound = _REVIEWED_NB_MATCHED_REPLAYS.get(case, 0)
    collector.check(
        "replay bound",
        lambda: check_that(
            len(replayed) <= bound,
            f"{len(replayed)} replayed matched-state decisions; bound {bound}",
        ),
    )
    collector.raise_if_any(f"{case} matched NB bgam.fitd state")


# --------------------------------------------------------------------------
# Boundary, public route and controller order.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _MODES)
def test_identity_domain_exit_stops_like_pinned_bam(r_bridge, mode):
    """The identity link's first proposal leaves the NB domain.

    With the separate-smooth default smoothing the iteration-two predictor is
    negative somewhere. Pinned bam stops at iteration two, inside
    ``estimate.theta`` (a non-finite ``nlogl``) when theta is estimated and at
    ``stop("Non-finite deviance")`` when it is fixed, after its iteration-one
    trials only; the discrete route raises at the same point.
    """
    frame, weights, offset = _nb_frame("identity")
    sp = _SP[_SEPARATE]
    trace = _trace(
        r_bridge,
        "identity",
        _SEPARATE,
        mode,
        sp,
        frame,
        weights,
        offset,
        allow_failure=True,
    )
    assert trace.failure is not None
    assert trace.fit is None
    assert {trial.iteration for trial in trace.trials} == {1}
    expected = (
        "estimate.theta objective" if mode == "estimated" else "deviance at iteration 2"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(FloatingPointError, match=expected):
            fit_discrete_regular(
                parse_formula(_SEPARATE),
                _source(frame, weights, offset),
                _nb_family("identity", mode),
                sp,
                control=_control(False),
            )


def test_public_nb_route_reports_theta_and_strategy():
    frame, weights, offset = _nb_frame("log", n=120)
    family = NegativeBinomial(theta=1.5)
    result = GAM(
        _SEPARATE,
        family=family,
        control=_control(True, uncertainty="fisher"),
    ).fit(
        DataFrameRowSource(frame, response="y", weights=weights, offset=offset),
        result="prediction",
    )
    assert result.execution_route == "discrete"
    assert result.lambda_strategy == "working_model_reml"
    assert result.converged
    assert "dDeta Newton performance iteration" in result.convergence_info
    assert "estimated theta" in result.convergence_info
    assert result.theta is not None
    assert result.theta != 1.5
    np.testing.assert_allclose(
        result.family.get_theta(transformed=True), [result.theta], rtol=STRICT.rtol
    )
    # The caller's family is not mutated by the discrete route.
    np.testing.assert_array_equal(family.get_theta(transformed=True), [1.5])
    held_out = pd.DataFrame(
        {
            "x": [0.2, 0.7],
            "z": [-0.5, 0.4],
            "g": pd.Categorical(["a", "c"], categories=_LEVELS),
        }
    )
    response = result.predict(held_out, pred_type="response", offset=np.zeros(2))
    assert np.all(response > 0)
    fixed = GAM(
        _SEPARATE,
        family=NegativeBinomial(theta=2.0, fixed=True, link="sqrt"),
        sp=[1.0, 2.0, 1.0],
        control=_control(False),
    ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
    assert fixed.theta == 2.0
    assert "fixed theta" in fixed.convergence_info


class _ScriptedNuisanceRows:
    """Scripted deviances; records which coefficients and theta were seen."""

    def __init__(self, deviances):
        self.deviances = list(deviances)
        self.theta = 0.0
        self.seen: list[tuple[str, float, bool]] = []

    def initial(self):
        return regular.WorkingEvaluation(
            self.deviances.pop(0), regular.WorkingStatistics(np.eye(2), np.ones(2), 3.0)
        )

    def evaluate(self, coefficients, *, statistics=True):
        self.seen.append((str(np.round(coefficients, 3)), self.theta, statistics))
        return regular.WorkingEvaluation(
            self.deviances.pop(0),
            regular.WorkingStatistics(np.eye(2), np.ones(2), 3.0)
            if statistics
            else None,
        )

    def response_variance(self):
        return 1.0


def test_controller_orders_dev0_step_control_theta_update_and_statistics():
    """bam.r:575-621: dev0 and step control at the entering theta, then
    ``estimate.theta`` at the step-controlled coefficients, then the working
    model at the new theta; the penalized deviance keeps the entering theta.
    """
    rows = _ScriptedNuisanceRows([5.0, 7.0, 2.0, 1.5, 4.0])
    updates = []

    class _Nuisance:
        def update(self, coefficients):
            updates.append((np.array(coefficients), rows.theta))
            rows.theta += 1.0
            return "theta record"

    state = regular.RegularWorkingModelState(
        2,
        np.zeros(0),
        np.zeros(0),
        np.array([1.0, 1.0]),
        np.array([0.0, 0.0]),
        123.0,
        10.0,
        1.0,
    )
    advance = regular.advance_regular_working_model(
        state,
        rows,
        lambda *_args: _proposal([0.5, 0.5]),
        lambda beta, _trial: float(beta @ beta),
        scale_estimated=False,
        epsilon=1e-7,
        start=lambda *_args: np.zeros(0),
        nuisance=_Nuisance(),
    )
    # dev0 re-evaluated (5.0, ignoring the carried 123.0), trial 7 + 2 > 5:
    # halve once, trial 2 + 0.5 <= 5: accept; then theta, then statistics.
    assert [entry[2] for entry in rows.seen] == [False, False, False, True]
    assert {entry[1] for entry in rows.seen[:3]} == {0.0}
    assert rows.seen[3][1] == 1.0
    np.testing.assert_array_equal(updates[0][0], [0.5, 0.5])
    assert advance.coefficient_halvings == 1
    assert advance.penalized_deviance == 2.0 + 0.5
    assert advance.iteration.nuisance == "theta record"
    assert _overrides([advance.iteration]) == []


def _proposal(coefficients):
    from tests.test_execution.test_regular_working_model import _proposal as make

    return make(coefficients)
