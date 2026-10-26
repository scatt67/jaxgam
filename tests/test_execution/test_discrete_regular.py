"""Regular-family bgam.fitd performance iteration on the discrete route.

Pinned mgcv 1.9-3 ``bam(discrete=TRUE)`` fits Poisson, Binomial and Gamma by
``bgam.fitd``'s Fisher performance iteration. The gates here are:

* fixed sp on the identical explicit design: the same controller driven by
  rows that evaluate the explicit snapped design with dense PIRLS's own
  Fisher arithmetic reproduces every field of the discrete fit, every
  discrete statistic/proposal equals the explicit X'WX/X'Wz and penalized
  Fisher solve, and JaxGAM's dense PIRLS converges to the same penalized
  deviance;
* fixed sp and working-model REML against pinned ``bam`` end to end;
* matched-state oracles: every iteration's working statistics, every
  ``Sl.fitChol`` trial and every outer-iteration controller decision
  evaluated from pinned R's exact state;
* one boundary case per family, and a domain exit without step control.

Every comparison is STRICT except two documented flat smoothing-parameter
vectors (selected Binomial/log and the selected Poisson low-count boundary
fit), each with one log sp drifting past 16.

Ties. ``bgam.fitd`` halves a coefficient step when ``dev0 + bSb0 < dev +
bSb``. Near convergence (and for Gamma while the log scale is still moving)
the two penalized deviances can agree to rounding, and pinned R's own outcome
there flips under a one-ULP offset perturbation. An outcome that differs
between two implementations is admitted only when its relative margin is
unresolvable at STRICT; the comparison then replays the reference's
step-control outcomes (``_oracle_replay_halvings``), checks again that every
replayed outcome our rule would not have taken is such a tie, and gates every
field at STRICT.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

from jaxgam import ContinuousDiscretizationControl, FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution import regular_working_model as regular
from jaxgam.execution.discrete_regular import (
    fit_discrete_regular,
    regular_working_components,
)
from jaxgam.execution.working_model import evaluate_gaussian_working_model_proposal
from jaxgam.families.standard import Binomial, Gamma, Poisson
from jaxgam.fitting.initialization import initialize_beta
from jaxgam.fitting.pirls import canonical_working_quantities, pirls_loop
from jaxgam.formula.continuous_setup import build_continuous_discrete_setup
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector, check_that
from tests.test_fitting.test_working_model import (
    check_newton_step,
    ldet_tolerance,
    source_working_metadata,
)
from tests.tolerances import MODERATE, STRICT

_LEVELS = ("a", "b", "c", "d")
_SEPARATE = "y ~ s(x,bs='cr',k=6) + s(z,bs='cr',k=5) + s(g,bs='re')"
_TENSOR_BY = "y ~ g + te(x,z,bs='cr',k=3) + s(w,bs='cr',k=5,by=g)"
_FIXED_SP = {
    _SEPARATE: np.array([0.6, 1.5, 0.8]),
    _TENSOR_BY: np.array([0.5, 1.2, 0.7, 1.1, 0.9, 1.6]),
}
_BOUNDARY_FORMULA = "y ~ g + s(x,bs='cr',k=6) + s(z,bs='cr',k=5)"
_BOUNDARY_SP = np.array([0.8, 1.3])
_FAMILIES = {"poisson": Poisson, "binomial": Binomial, "gamma": Gamma}
_CANONICAL = (("poisson", "log"), ("binomial", "logit"), ("gamma", "inverse"))
_NONCANONICAL = (
    ("poisson", "sqrt"),
    ("poisson", "identity"),
    ("binomial", "probit"),
    ("binomial", "cloglog"),
    ("binomial", "log"),
    ("gamma", "log"),
    ("gamma", "identity"),
)


def _cases(selected: bool):
    cases = [
        pytest.param(family, link, formula, id=f"{family}-{link}-{label}")
        for family, link in _CANONICAL
        for formula, label in ((_SEPARATE, "separate_re"), (_TENSOR_BY, "tensor_by"))
        # Pinned bam stops on this one (see the domain-exit test).
        if not (selected and family == "gamma" and formula == _TENSOR_BY)
    ]
    return cases + [
        pytest.param(family, link, _SEPARATE, id=f"{family}-{link}-separate_re")
        for family, link in _NONCANONICAL
    ]


def _label(formula: str) -> str:
    return {_SEPARATE: "separate_re", _TENSOR_BY: "tensor_by"}[formula]


def _frame(family: str, link: str, n: int = 120, seed: int = 11):
    """Runtime data with weights, an offset, a factor and link-safe means."""
    position = np.arange(n, dtype=float)
    x = (position + 0.2 * np.sin(position * 0.7)) / n
    z = np.cos(position * 0.23) + 0.07 * np.sin(position * 0.91)
    w = np.sin(position * 0.37) + 0.5 * position / n
    g = pd.Categorical(np.resize(np.array(_LEVELS), n), categories=_LEVELS)
    code = np.asarray(g.codes)
    eta = (
        0.1
        + (0.7 + 0.25 * code) * np.sin(4.5 * x)
        + 0.6 * np.cos(3.0 * z)
        + np.array([-0.25, 0.1, 0.3, -0.15])[code]
        + 0.3 * np.sin(3 * x) * np.cos(2 * z)
        + (0.2 + 0.15 * code) * np.cos(2.0 * w)
    )
    rng = np.random.default_rng(seed)
    weights = np.linspace(0.6, 1.4, n)
    if family == "poisson":
        y = rng.poisson(np.exp(eta) + (6.0 if link == "identity" else 0.0))
    elif family == "binomial":
        # Integer trial counts as prior weights; y is the success proportion.
        weights = rng.integers(1, 5, size=n).astype(float)
        p = 0.05 * np.exp(0.5 * eta) if link == "log" else 1 / (1 + np.exp(-eta))
        y = rng.binomial(weights.astype(int), p) / weights
    else:
        mean = np.exp(eta) + (0.5 if link == "identity" else 0.0)
        y = rng.gamma(8.0, mean / 8.0)
    offset = np.linspace(-0.1, 0.2, n)
    frame = pd.DataFrame({"x": x, "z": z, "w": w, "g": g, "y": np.asarray(y, float)})
    return frame, weights, offset


def _control(selected: bool, batch_rows: int = 17, **kwargs) -> FitControl:
    return FitControl(
        execution="discrete",
        batch_rows=batch_rows,
        smoothing_strategy="working_model_reml" if selected else "exact_reml",
        discretization=ContinuousDiscretizationControl(m=30, factor_names=("g",)),
        **kwargs,
    )


def _source(frame, weights, offset) -> DataFrameRowSource:
    return DataFrameRowSource(frame, response="y", weights=weights, offset=offset)


def _snapped_design(formula: str, frame: pd.DataFrame):
    """The explicit design at the discretized covariates, and its penalties."""
    setup = build_continuous_discrete_setup(
        parse_formula(formula),
        DataFrameRowSource(frame, response="y"),
        m=30,
        factor_names=("g",),
    )
    try:
        snapped = {
            name: values[group.index.selector]
            for group in setup.compact.groups
            for name, values in zip(group.names, group.index.values, strict=True)
        }
        X = setup.predict_spec.build_predict_matrix(pd.DataFrame(snapped))
        return X, setup.penalties
    finally:
        setup.cleanup()


def _family(family: str, link: str):
    return _FAMILIES[family](link=link)


def _close(observed, expected, tolerance):
    return lambda: np.testing.assert_allclose(
        observed, expected, rtol=tolerance.rtol, atol=tolerance.atol
    )


# --------------------------------------------------------------------------
# Step-control ties.
# --------------------------------------------------------------------------

# bgam.fitd's ``kk < 30``: at most 29 coefficient step halvings.
_MAX_COEFFICIENT_TRIALS = 30


def _records(fit):
    """Outer-iteration records of a controller fit or a discrete fit state."""
    if isinstance(fit, regular.RegularWorkingModelFit):
        return fit.trace
    return fit.working_model_trace


def _our_decisions(fit) -> list[tuple[int, int]]:
    """Per outer iteration: (coefficient halvings, Sl.fitChol trials)."""
    return [
        (record.coefficient_halvings, record.trial_evaluations)
        for record in _records(fit)
    ]


def _source_decisions(trace) -> list[tuple[int, int]]:
    groups: dict[int, list] = {}
    for trial in trace.trials:
        groups.setdefault(trial.iteration, []).append(trial)
    decisions = [
        (trials[0].coefficient_trials - 1, len(trials)) for trials in groups.values()
    ]
    if trace.converged:
        decisions.append((trace.final_coefficient_trials - 1, 0))
    return decisions


def _schedule(decisions: list[tuple[int, int]]) -> dict[int, int]:
    """Replayable halvings: iterations with step control (from three)."""
    return {
        iteration: halvings
        for iteration, (halvings, _) in enumerate(decisions, start=1)
        if iteration > 2
    }


def _is_tie(margin: float) -> bool:
    """Two penalized deviances equal at STRICT cannot be ordered at STRICT."""
    return bool(abs(margin) <= STRICT.rtol)


def _overrides(records) -> list[tuple[int, int, float]]:
    """Replayed decisions: trials whose outcome differs from our own rule.

    Each entry is (outer iteration, step-control trial, margin). Without a
    replay the list is empty by construction.
    """
    replayed = []
    for record in records:
        if record.iteration < 3:
            continue
        for trial, margin in enumerate(record.step_control_margins):
            ours = (not np.isfinite(margin) or margin > 0) and (
                trial + 1 < _MAX_COEFFICIENT_TRIALS
            )
            if ours != (trial < record.coefficient_halvings):
                replayed.append((record.iteration, trial, float(margin)))
    return replayed


@dataclass(frozen=True)
class _Followed:
    """A fit gated at STRICT and, when ties were replayed, the public fit.

    ``fit`` follows the reference's step-control outcomes at STRICT ties;
    ``public`` is the unreplayed fit a user gets (``None`` when no tie
    differed) and ``replayed`` lists the replayed decisions.
    """

    fit: object
    public: object | None
    replayed: tuple[tuple[int, int, float], ...]
    problems: tuple[str, ...]


def _follow(run, reference: list[tuple[int, int]]) -> _Followed:
    """Fit, then replay the reference's ties if (and only if) they differ.

    ``run(_oracle_replay_halvings)`` fits. Problems are a first difference
    that is not a coefficient step-control tie, or a replayed fit that still
    differs or overrides a resolvable margin.
    """
    fit = run(None)
    ours = _our_decisions(fit)
    if ours == reference:
        return _Followed(fit, None, (), ())
    index = next(
        (
            k
            for k, pair in enumerate(zip(ours, reference, strict=False))
            if pair[0] != pair[1]
        ),
        min(len(ours), len(reference)),
    )
    if index == min(len(ours), len(reference)):
        problem = f"iteration counts differ: {ours} vs reference {reference}"
        return _Followed(fit, None, (), (problem,))
    (our_halvings, _), (reference_halvings, _) = ours[index], reference[index]
    if our_halvings == reference_halvings:
        problem = (
            f"sp/scale trial counts differ at iteration {index + 1}: "
            f"{ours[index]} vs reference {reference[index]}"
        )
        return _Followed(fit, None, (), (problem,))
    margin = _records(fit)[index].step_control_margins[
        min(our_halvings, reference_halvings)
    ]
    if not _is_tie(margin):
        problem = (
            f"iteration {index + 1}: {our_halvings} vs reference "
            f"{reference_halvings} halvings at a resolvable margin {margin:.3e}"
        )
        return _Followed(fit, None, (), (problem,))
    replayed = run(_schedule(reference))
    decisions = tuple(_overrides(_records(replayed)))
    problems = [
        f"iteration {iteration} trial {trial}: replay overrides margin {value:.3e}"
        for iteration, trial, value in decisions
        if not _is_tie(value)
    ]
    if _our_decisions(replayed) != reference:
        problems.append(
            f"replayed decisions {_our_decisions(replayed)} vs reference {reference}"
        )
    return _Followed(replayed, fit, decisions, tuple(problems))


# Replayed step-control decisions per fit (root decision 2026-10-24): replay
# resolves only STRICT ties, and no more of them than reviewed for the case
# (the larger of the macOS host and pinned Linux arm64 counts). Every other
# case must need none, so replay cannot silently cover a systematic
# difference.
_REVIEWED_TIE_REPLAYS = {
    ("explicit", "gamma-inverse-separate_re"): 2,  # host 2, Linux 1
    ("explicit", "gamma-inverse-tensor_by"): 1,  # host 1, Linux 1
    ("fixed", "gamma-inverse-separate_re"): 3,  # host 3, Linux 0
    ("fixed", "gamma-inverse-tensor_by"): 4,  # host 4, Linux 1
    ("selected", "gamma-inverse-separate_re"): 1,  # host 1, Linux 0
    ("boundary-fixed", "gamma-near-zero-y"): 67,  # host 67, Linux 38
}


def _check_replays(collector, key, followed: _Followed) -> None:
    bound = _REVIEWED_TIE_REPLAYS.get(key, 0)
    collector.check(
        "decisions",
        lambda: check_that(not followed.problems, "; ".join(followed.problems)),
    )
    collector.check(
        "replay bound",
        lambda: check_that(
            len(followed.replayed) <= bound,
            f"{len(followed.replayed)} replayed decisions for {key}; reviewed "
            f"bound {bound}",
        ),
    )


def _check_public_fit(collector, followed: _Followed, fields) -> None:
    """The unreplayed public fit's physical fields, at MODERATE.

    Only for a fit that needed a tie replay. Pinned R's own fit is not
    reproducible at STRICT through these ties: a one-ULP offset perturbation
    moves pinned R's Gamma/inverse fixed-sp coefficients 67.9x STRICT, the
    same gap the unreplayed fit shows. ``fields(fit)`` lists (name, observed,
    expected) for coefficients, fitted values, deviance, EDF, scale, score.
    """
    if followed.public is None:
        return
    for name, observed, expected in fields(followed.public):
        collector.check(f"public {name}", _close(observed, expected, MODERATE))


# --------------------------------------------------------------------------
# Gate 1: fixed sp on the identical explicit design.
# --------------------------------------------------------------------------


class _ExplicitRows:
    """``bgam.fitd`` rows on the explicit snapped design, dense PIRLS arithmetic.

    Working weights/pseudodata are ``canonical_working_quantities`` (dense
    PIRLS's Fisher quantities) and the deviance is ``family.dev_resids``.
    """

    def __init__(self, X, frame, weights, offset, family):
        self.X = X
        self.y_raw = frame.y.to_numpy()
        self.weights = weights
        self.offset = offset
        self.family = family
        self.y = np.asarray(family.execution_initial_response_cpu(self.y_raw, weights))

    def _evaluate(self, eta, statistics):
        family = self.family
        mu = np.asarray(family.link.inverse(eta))
        W, z = (
            np.asarray(value)
            for value in canonical_working_quantities(
                family, self.y, mu, eta, self.weights, self.offset
            )
        )
        deviance = float(family.dev_resids(self.y, mu, self.weights))
        return regular.WorkingEvaluation(
            deviance,
            regular.WorkingStatistics(
                self.X.T @ (W[:, None] * self.X),
                self.X.T @ (W * z),
                float(np.sum(W * z**2)),
            )
            if statistics
            else None,
        )

    def initial(self):
        state = self.family.initial_working_state_cpu(
            self.y_raw, self.weights, np.ones(len(self.y), dtype=bool)
        )
        return self._evaluate(np.asarray(state.eta), True)

    def evaluate(self, coefficients, *, statistics=True):
        return self._evaluate(self.X @ coefficients + self.offset, statistics)

    def response_variance(self):
        return float(np.var(self.y, ddof=1))


class _RecordingRows:
    """Forward to the discrete rows; keep every evaluated (beta, statistics)."""

    def __init__(self, rows):
        self.rows = rows
        self.records: list[tuple[np.ndarray | None, object]] = []

    def initial(self):
        evaluation = self.rows.initial()
        self.records.append((None, evaluation))
        return evaluation

    def evaluate(self, coefficients, *, statistics=True):
        evaluation = self.rows.evaluate(coefficients, statistics=statistics)
        self.records.append((np.array(coefficients), evaluation))
        return evaluation

    def response_variance(self):
        return self.rows.response_variance()


_EXPLICIT = [
    pytest.param(family, link, _SEPARATE, id=f"{family}-{link}-separate_re")
    for family, link in (*_CANONICAL, *_NONCANONICAL)
] + [
    pytest.param(family, link, _TENSOR_BY, id=f"{family}-{link}-tensor_by")
    for family, link in _CANONICAL
]


@pytest.mark.parametrize(("family", "link", "formula"), _EXPLICIT)
def test_fixed_sp_matches_the_identical_explicit_design(family, link, formula):
    """The discrete fit is the explicit-design Fisher fit, field by field.

    (1) The same ``bgam.fitd`` controller and ``Sl.fitChol`` kernel, driven
    by explicit-design rows with dense PIRLS's Fisher arithmetic, gives the
    discrete fit's coefficients, fitted values, deviance, EDF, scale and
    score at STRICT. (2) At every coefficient vector the discrete controller
    evaluates, its indexed statistics equal the explicit X'WX/X'Wz and each
    proposal equals the penalized Fisher solve. (3) Iterated to its rounding
    floor, the discrete fit reaches the penalized deviance of JaxGAM's dense
    ``pirls_loop`` fixed point. (Final coefficients of the two loops are not
    compared: ``bgam.fitd`` stops on a penalized deviance change, which is
    second order in the remaining step, while ``pirls_loop`` stops on a
    coefficient change.)
    """
    frame, weights, offset = _frame(family, link)
    sp = _FIXED_SP[formula]
    fam = _family(family, link)
    X, penalties = _snapped_design(formula, frame)
    S = penalties.materialize(np.log(sp))
    explicit_rows = _ExplicitRows(X, frame, weights, offset, fam)
    collector = _AssertCollector()
    with regular_working_components(
        parse_formula(formula),
        _source(frame, weights, offset),
        fam,
        sp,
        control=_control(False),
    ) as parts:
        explicit_fit = parts.run(explicit_rows)
        explicit = parts.result(explicit_fit)
        proposals = []

        def propose(trial, tolerance, statistics):
            proposal = parts.propose(trial, tolerance, statistics)
            proposals.append((statistics, np.asarray(proposal.coefficients)))
            return proposal

        recorded = replace(parts, propose=propose)
        recording = _RecordingRows(parts.rows)

        def run(replay):
            recording.records.clear()
            proposals.clear()
            return recorded.run(recording, _oracle_replay_halvings=replay)

        followed = _follow(run, _our_decisions(explicit_fit))
        discrete = parts.result(followed.fit)
        public = None if followed.public is None else parts.result(followed.public)
        floor = regular.fit_regular_working_model(
            parts.rows,
            parts.propose,
            parts.penalty,
            X.shape[1],
            scale_estimated=parts.scale_estimated,
            epsilon=1e-300,
            max_iterations=100,
            start=parts.start,
        )
    _check_replays(
        collector, ("explicit", f"{family}-{link}-{_label(formula)}"), followed
    )
    collector.check(
        "iterations and convergence",
        lambda: check_that(
            discrete.converged
            and explicit.converged
            and discrete.scale_iterations == explicit.scale_iterations,
            f"{discrete.scale_iterations=} vs {explicit.scale_iterations=}",
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
            ("scale", fit.scale, explicit.scale),
            ("score", fit.score, explicit.score),
        ]

    for name, observed, expected in fields(discrete):
        collector.check(name, _close(observed, expected, STRICT))
    _check_public_fit(collector, replace(followed, public=public), fields)
    y = explicit_rows.y
    for index, (beta, evaluation) in enumerate(recording.records):
        eta = (
            fam.initial_working_state_cpu(
                frame.y.to_numpy(), weights, np.ones(len(y), dtype=bool)
            ).eta
            if beta is None
            else X @ beta + offset
        )
        mu = np.asarray(fam.link.inverse(eta))
        W, z = (
            np.asarray(value)
            for value in canonical_working_quantities(fam, y, mu, eta, weights, offset)
        )
        collector.check(
            f"evaluation {index} deviance",
            _close(
                evaluation.deviance,
                float(np.sum(fam.deviance_contributions(y, mu, weights))),
                STRICT,
            ),
        )
        if evaluation.statistics is None:
            continue
        collector.check(
            f"evaluation {index} X'WX",
            _close(evaluation.statistics.gram, X.T @ (W[:, None] * X), STRICT),
        )
        collector.check(
            f"evaluation {index} X'Wz",
            _close(evaluation.statistics.rhs, X.T @ (W * z), STRICT),
        )
    for index, (statistics, coefficients) in enumerate(proposals):
        collector.check(
            f"proposal {index}",
            _close(
                coefficients,
                np.linalg.solve(statistics.gram + S, statistics.rhs),
                STRICT,
            ),
        )
    beta0 = initialize_beta(X, y, weights, fam, offset)
    dense = pirls_loop(X, y, beta0, S, fam, weights, offset, max_iter=400, tol=1e-14)
    collector.check(
        "shared fixed point",
        lambda: (
            check_that(bool(dense.converged), "dense PIRLS converges"),
            _close(floor.penalized_deviance, float(dense.penalized_deviance), STRICT)(),
        ),
    )
    collector.raise_if_any(f"{family}/{link} discrete vs explicit design")


# --------------------------------------------------------------------------
# Gate 2: end to end against pinned bam(discrete=TRUE).
# --------------------------------------------------------------------------


def _input_digest(frame: pd.DataFrame, weights, offset) -> str:
    """SHA-256 of the generated input bytes.

    Every column in order (little-endian float64 values; for a factor its
    int64 codes and levels), then the prior weights and the offset.
    """
    digest = hashlib.sha256()
    for name in frame.columns:
        column = frame[name]
        digest.update(name.encode())
        if isinstance(column.dtype, pd.CategoricalDtype):
            digest.update(np.asarray(column.cat.codes, dtype="<i8").tobytes())
            digest.update("\x1f".join(map(str, column.cat.categories)).encode())
        else:
            digest.update(np.asarray(column, dtype="<f8").tobytes())
    digest.update(np.asarray(weights, dtype="<f8").tobytes())
    digest.update(np.asarray(offset, dtype="<f8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class _ReviewedFlatSp:
    """One reviewed MODERATE smoothing-parameter vector and its exact case."""

    digest: str
    formula: str
    family_class: type
    link: str
    control: FitControl


# Root decision 2026-10-24: the selected smoothing-parameter vector of these
# two fits (one log sp drifting past 16, where pinned R itself moves 0.2-8.6x
# STRICT under one-ULP offset noise) is MODERATE after the four correction
# attempts in docs/scale_jaxgam/pr11_working_model_plan.md, on the exact
# generated bytes, model and controls reviewed. Libm differences change the
# low bits of the trigonometric generators across platforms: the Binomial/log
# bytes are pinned Linux arm64's and the Poisson low-count bytes the macOS
# arm64 host's, where each vector exceeded STRICT; other bytes stay STRICT.
# Every physical field is STRICT everywhere.
_REVIEWED_FLAT_SP = {
    "binomial-log-separate_re": _ReviewedFlatSp(
        "ce5962e805e08fa9bc39b91f6369d24340d0caf32487c007045200d8f406de5a",
        _SEPARATE,
        Binomial,
        "log",
        _control(True),
    ),
    "poisson-low-counts": _ReviewedFlatSp(
        "cd4b75b8f0e40cc7663b11db3a05ae278b93cdf0ee55eb838bd5d64500d1318e",
        _BOUNDARY_FORMULA,
        Poisson,
        "log",
        _control(True),
    ),
}


def _flat_sp_tolerance(case, frame, weights, offset, formula, family, control, sp):
    """MODERATE for a reviewed sp vector on its exact bytes and controls only.

    Other bytes keep STRICT; reviewed bytes with any changed model or control
    fail instead of inheriting the allowance.
    """
    reviewed = _REVIEWED_FLAT_SP.get(case)
    if reviewed is None or _input_digest(frame, weights, offset) != reviewed.digest:
        return STRICT
    assert formula == reviewed.formula
    assert type(family) is reviewed.family_class
    assert type(family.link) is type(reviewed.family_class(link=reviewed.link).link)
    assert sp is None
    assert control.smoothing_strategy == "working_model_reml"
    assert control == reviewed.control
    return MODERATE


def _fit_against(trace, formula, frame, weights, offset, fam, sp, selected, **kwargs):
    """Fit, following pinned R's step-control ties (see the module note)."""

    def run(replay):
        return fit_discrete_regular(
            parse_formula(formula),
            _source(frame, weights, offset),
            fam,
            sp,
            control=_control(selected, **kwargs),
            _oracle_replay_halvings=replay,
        )

    return _follow(run, _source_decisions(trace))


def _gate_against_bam(r_bridge, family, link, formula, *, selected):
    frame, weights, offset = _frame(family, link)
    sp = None if selected else _FIXED_SP[formula]
    fam = _family(family, link)
    trace = r_bridge.trace_bam_regular_working_model(
        formula,
        frame,
        family=family,
        link=link,
        sp=sp,
        factor_names=("g",),
        weights=weights,
        offset=offset,
    )
    followed = _fit_against(trace, formula, frame, weights, offset, fam, sp, selected)
    fit = followed.fit
    X, _ = _snapped_design(formula, frame)
    expected = trace.fit
    collector = _AssertCollector()
    case = f"{family}-{link}-{_label(formula)}"
    _check_replays(collector, ("selected" if selected else "fixed", case), followed)
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
        ]

    for name, observed, target in fields(fit):
        collector.check(name, _close(observed, target, STRICT))
    if selected:
        tolerance = _flat_sp_tolerance(
            case, frame, weights, offset, formula, fam, _control(selected), sp
        )
        collector.check(
            "smoothing parameters",
            _close(fit.smoothing_params, expected.smoothing_params, tolerance),
        )
    _check_public_fit(collector, followed, fields)
    collector.raise_if_any(
        f"{family}/{link} {'selected' if selected else 'fixed'} vs pinned bam"
    )


@pytest.mark.parametrize(("family", "link", "formula"), _cases(selected=False))
def test_fixed_sp_matches_pinned_bam(r_bridge, family, link, formula):
    _gate_against_bam(r_bridge, family, link, formula, selected=False)


@pytest.mark.parametrize(("family", "link", "formula"), _cases(selected=True))
def test_working_model_reml_matches_pinned_bam(r_bridge, family, link, formula):
    _gate_against_bam(r_bridge, family, link, formula, selected=True)


# --------------------------------------------------------------------------
# Matched-state oracles: pinned R's exact state fed to every JAX component.
# --------------------------------------------------------------------------


def _public(beta: np.ndarray, blocks) -> np.ndarray:
    """``Sl.initial.repara(beta, inverse=TRUE)`` for one coefficient vector."""
    out = np.array(beta, dtype=float)
    for block in blocks:
        if block.reparameterized:
            out[block.start : block.stop] = (
                block.transform @ out[block.start : block.stop]
            )
    return out


def _boundary_frame(case: str, n: int = 150, seed: int = 5):
    position = np.arange(n, dtype=float)
    x = (position + 0.3 * np.sin(position)) / n
    z = np.cos(position * 0.19)
    g = pd.Categorical(np.resize(np.array(_LEVELS), n), categories=_LEVELS)
    eta = (
        np.sin(4.0 * x)
        + 0.6 * np.cos(3.0 * z)
        + np.array([-0.3, 0, 0.2, 0.1])[np.asarray(g.codes)]
    )
    rng = np.random.default_rng(seed)
    weights = np.linspace(0.7, 1.3, n)
    offset = np.linspace(-0.05, 0.05, n)
    if case == "poisson_low_counts":
        y = rng.poisson(0.08 * np.exp(eta)).astype(float)
    elif case == "binomial_near_0_1":
        weights = np.ones(n)
        weights[::17] = 0.0
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-6.0 * eta))).astype(float)
    elif case == "gamma_near_zero":
        y = rng.gamma(8.0, 1e-4 * np.exp(eta) / 8.0)
    else:
        raise ValueError(case)
    frame = pd.DataFrame({"x": x, "z": z, "g": g, "y": y})
    return frame, weights, offset


_MATCHED = {
    "poisson-log-selected": ("poisson", "log", _SEPARATE, True, None),
    "binomial-probit-fixed": ("binomial", "probit", _SEPARATE, False, None),
    # Ties at iteration seven (pinned R halves three times).
    "gamma-inverse-fixed": ("gamma", "inverse", _TENSOR_BY, False, None),
    "gamma-inverse-separate-selected": ("gamma", "inverse", _SEPARATE, True, None),
    "gamma-log-selected": ("gamma", "log", _SEPARATE, True, None),
    "binomial-logit-selected": ("binomial", "logit", _TENSOR_BY, True, None),
    # A flat, drifting log sp (17.8) and a long run of rounding-level ties.
    "poisson-low-counts-selected": (
        "poisson",
        "log",
        _BOUNDARY_FORMULA,
        True,
        "poisson_low_counts",
    ),
    "gamma-near-zero-fixed": (
        "gamma",
        "log",
        _BOUNDARY_FORMULA,
        False,
        "gamma_near_zero",
    ),
}


def _advance_following(parts, state, source_halvings, collector, label, replayed):
    """One controller iteration from R's state, replaying a STRICT tie.

    Every replayed decision is appended to ``replayed``.
    """
    arguments = {
        "scale_estimated": parts.scale_estimated,
        "epsilon": 1e-7,
        "start": parts.start,
    }
    advance = regular.advance_regular_working_model(
        state, parts.rows, parts.propose, parts.penalty, **arguments
    )
    if advance.coefficient_halvings == source_halvings:
        return advance
    margin = advance.iteration.step_control_margins[
        min(advance.coefficient_halvings, source_halvings)
    ]
    collector.check(
        f"{label} halving tie",
        lambda: check_that(
            _is_tie(margin), f"halving decision differs with margin {margin:.3e}"
        ),
    )
    advance = regular.advance_regular_working_model(
        state,
        parts.rows,
        parts.propose,
        parts.penalty,
        _oracle_replay_halvings=source_halvings,
        **arguments,
    )
    for iteration, trial, other in _overrides([advance.iteration]):
        replayed.append((iteration, trial, other))
        collector.check(
            f"{label} replayed trial {trial} tie",
            lambda other=other: check_that(
                _is_tie(other), f"replay overrides margin {other:.3e}"
            ),
        )
    return advance


# Replayed matched-state decisions per case: the larger of the macOS host and
# pinned Linux arm64 counts; every other case must need none.
_REVIEWED_MATCHED_REPLAYS = {
    "gamma-inverse-fixed": 2,  # host 1, Linux 2
    "gamma-near-zero-fixed": 50,  # host 50, Linux 37
}


@pytest.mark.parametrize("case", list(_MATCHED))
def test_matched_state_statistics_proposals_and_controller(r_bridge, case):
    """Every pinned bgam.fitd iteration and Sl.fitChol trial, from R's state.

    (1) The discrete working statistics at R's step-controlled coefficients
    equal R's ``qrx$R``/``qrx$f``/``y.norm2``. (2) Each proposal kernel, given
    R's ``Sl``, reparameterized moments, exp(rho), log(phi) and tolerance,
    reproduces ``Sl.fitChol`` (ldetS follows ``ldet_tolerance``; steps use the
    conditioning-aware gates). (3) Restarted from R's carried state at every
    iteration, including the converging one, the controller makes R's
    decisions (a differing halving must be a STRICT tie, then is replayed)
    and reaches R's next state and final coefficients.
    """
    family, link, formula, selected, boundary = _MATCHED[case]
    frame, weights, offset = (
        _frame(family, link) if boundary is None else _boundary_frame(boundary)
    )
    sp = None if selected else (_FIXED_SP.get(formula, _BOUNDARY_SP))
    fam = _family(family, link)
    trace = r_bridge.trace_bam_regular_working_model(
        formula,
        frame,
        family=family,
        link=link,
        sp=sp,
        factor_names=("g",),
        weights=weights,
        offset=offset,
    )
    blocks = trace.trials[0].state.blocks
    metadata = source_working_metadata(trace.trials[0].state)
    ldet = ldet_tolerance(r_bridge)
    groups: dict[int, list] = {}
    for trial in trace.trials:
        groups.setdefault(trial.iteration, []).append(trial)
    collector = _AssertCollector()
    with regular_working_components(
        parse_formula(formula),
        _source(frame, weights, offset),
        fam,
        sp,
        control=_control(selected),
    ) as parts:
        for k, trials in groups.items():
            first = trials[0]
            evaluation = (
                parts.rows.initial()
                if k == 1
                else parts.rows.evaluate(first.coefficients)
            )
            statistics = evaluation.statistics
            for name, observed, expected in (
                ("X'WX", statistics.gram, first.gram),
                ("X'Wz", statistics.rhs, first.rhs),
                (
                    "y.norm2",
                    statistics.response_square_sum,
                    first.state.response_square_sum,
                ),
            ):
                collector.check(
                    f"iteration {k} {name}", _close(observed, expected, STRICT)
                )
            for index, trial in enumerate(trials):
                proposal = trial.proposal
                parameters = (
                    proposal.log_smoothing
                    if fam.scale_known
                    else np.r_[proposal.log_smoothing, proposal.log_scale]
                )
                result = evaluate_gaussian_working_model_proposal(
                    parameters,
                    trial.state.gram,
                    trial.state.rhs,
                    trial.state.response_square_sum,
                    metadata,
                    scale_is_fixed=fam.scale_known,
                    fixed_smoothing=not selected,
                    log_scale=proposal.log_scale,
                    tolerance=proposal.tolerance,
                    smoothing=proposal.smoothing,
                )
                label = f"iteration {k} trial {index}"
                for name, observed, expected, tolerance in (
                    (
                        "coefficients",
                        result.coefficients,
                        proposal.coefficients,
                        STRICT,
                    ),
                    ("gradient", result.gradient, proposal.gradient, STRICT),
                    ("hessian", result.hessian, proposal.hessian, STRICT),
                    ("ldetS", result.logdet_penalty, proposal.logdet_penalty, ldet),
                    ("ldetXXS", result.logdet_hessian, proposal.logdet_hessian, STRICT),
                ):
                    collector.check(
                        f"{label} {name}", _close(observed, expected, tolerance)
                    )
                if len(proposal.gradient):
                    check_newton_step(
                        collector,
                        label,
                        (result.gradient, result.hessian, result.step),
                        (proposal.gradient, proposal.hessian, proposal.step),
                        proposal.tolerance,
                    )
                else:
                    collector.check(
                        f"{label} empty step",
                        lambda result=result: check_that(
                            result.step.shape == (0,), "fixed sp and scale: no step"
                        ),
                    )
        iterations = sorted(groups)
        criterion = np.nan
        replayed: list[tuple[int, int, float]] = []
        for position, k in enumerate(iterations[1:], start=1):
            first = groups[k][0]
            accepted = None if k == 2 else groups[k - 1][0].coefficients
            state = regular.RegularWorkingModelState(
                k - 1,
                first.accepted,
                first.step,
                _public(groups[k - 1][-1].proposal.coefficients, blocks),
                accepted,
                np.nan
                if accepted is None
                else parts.rows.evaluate(accepted, statistics=False).deviance,
                first.previous_deviance,
                first.criterion,
            )
            label = f"replay iteration {k}"
            advance = _advance_following(
                parts, state, first.coefficient_trials - 1, collector, label, replayed
            )
            last = groups[k][-1]
            checks = [
                ("top coefficients", advance.coefficients, first.coefficients),
                ("penalized deviance", advance.penalized_deviance, first.deviance),
            ]
            collector.check(
                f"{label} decisions",
                lambda advance=advance, k=k: check_that(
                    not advance.converged
                    and advance.proposal.evaluations == len(groups[k]),
                    f"converged={advance.converged}, trials "
                    f"{advance.proposal and advance.proposal.evaluations} vs "
                    f"pinned {len(groups[k])}",
                ),
            )
            if advance.converged:
                continue
            criterion = advance.state.criterion
            checks += [
                (
                    "accepted parameters",
                    advance.state.parameters,
                    last.accepted + last.step,
                ),
                (
                    "next coefficients",
                    advance.state.coefficients,
                    _public(last.proposal.coefficients, blocks),
                ),
            ]
            if position + 1 < len(iterations):
                following = groups[iterations[position + 1]][0]
                checks.append(
                    ("criterion", advance.state.criterion, following.criterion)
                )
            for name, observed, expected in checks:
                collector.check(f"{label} {name}", _close(observed, expected, STRICT))
            if len(last.proposal.step):
                # The accepted trial's own system, re-evaluated on this
                # iteration's statistics, against R's for the next Nstep.
                ours = parts.propose(
                    advance.state.parameters,
                    last.proposal.tolerance,
                    advance.statistics,
                )
                collector.check(
                    f"{label} next step is the accepted proposal's",
                    _close(np.asarray(ours.step), advance.state.step, STRICT),
                )
                check_newton_step(
                    collector,
                    f"{label} next step",
                    (ours.gradient, ours.hessian, ours.step),
                    (last.proposal.gradient, last.proposal.hessian, last.proposal.step),
                    last.proposal.tolerance,
                )
        # The converging iteration: R's state after its last proposal.
        k = iterations[-1]
        last, first = groups[k][-1], groups[k][0]
        state = regular.RegularWorkingModelState(
            k,
            last.accepted + last.step,
            last.proposal.step,
            _public(last.proposal.coefficients, blocks),
            first.coefficients,
            parts.rows.evaluate(first.coefficients, statistics=False).deviance,
            first.deviance,
            criterion,
        )
        label = f"replay converging iteration {k + 1}"
        advance = _advance_following(
            parts,
            state,
            trace.final_coefficient_trials - 1,
            collector,
            label,
            replayed,
        )
        collector.check(
            f"{label} converges",
            lambda: check_that(
                trace.converged
                and trace.final_iteration == k + 1
                and advance.converged,
                f"pinned conv={trace.converged} iter={trace.final_iteration}; "
                f"replay converged={advance.converged}",
            ),
        )
        for name, observed, expected in (
            ("final coefficients", advance.coefficients, trace.fit.coefficients),
            ("final deviance", advance.deviance, trace.fit.deviance),
        ):
            collector.check(f"{label} {name}", _close(observed, expected, STRICT))
    bound = _REVIEWED_MATCHED_REPLAYS.get(case, 0)
    collector.check(
        "replay bound",
        lambda: check_that(
            len(replayed) <= bound,
            f"{len(replayed)} replayed matched-state decisions; reviewed bound {bound}",
        ),
    )
    collector.raise_if_any(f"{case} matched bgam.fitd state")


# --------------------------------------------------------------------------
# Family boundaries.
# --------------------------------------------------------------------------

_BOUNDARIES = [
    pytest.param("poisson_low_counts", "poisson", "log", id="poisson-low-counts"),
    pytest.param("binomial_near_0_1", "binomial", "logit", id="binomial-near-0-1"),
    pytest.param("gamma_near_zero", "gamma", "log", id="gamma-near-zero-y"),
]
_BOUNDARY_IDS = {param.values[0]: param.id for param in _BOUNDARIES}


@pytest.mark.parametrize(("case", "family", "link"), _BOUNDARIES)
@pytest.mark.parametrize("selected", [False, True], ids=["fixed", "selected"])
def test_family_boundary_matches_pinned_bam(r_bridge, case, family, link, selected):
    """Poisson mostly-zero counts, Binomial probabilities near 0 and 1 with
    zero-weight rows (whose response binomial()$initialize sets to 0), and
    Gamma responses of order 1e-4. Each must match pinned bam, keep its
    fitted means strictly inside the family domain and stay finite.

    The selected Poisson low-count sp vector (one log sp at 17.8) is the
    reviewed MODERATE vector of ``_REVIEWED_FLAT_SP`` on its exact bytes;
    every other field is STRICT.
    """
    frame, weights, offset = _boundary_frame(case)
    fam = _family(family, link)
    sp = None if selected else _BOUNDARY_SP
    trace = r_bridge.trace_bam_regular_working_model(
        _BOUNDARY_FORMULA,
        frame,
        family=family,
        link=link,
        sp=sp,
        factor_names=("g",),
        weights=weights,
        offset=offset,
    )
    followed = _fit_against(
        trace, _BOUNDARY_FORMULA, frame, weights, offset, fam, sp, selected
    )
    fit = followed.fit
    X, _ = _snapped_design(_BOUNDARY_FORMULA, frame)
    fitted = np.asarray(fam.link.inverse(X @ fit.coefficients + offset))
    expected = trace.fit
    collector = _AssertCollector()
    case_id = _BOUNDARY_IDS[case]
    _check_replays(
        collector,
        (f"boundary-{'selected' if selected else 'fixed'}", case_id),
        followed,
    )
    collector.check(
        "boundary data",
        lambda: check_that(
            {
                "poisson_low_counts": np.mean(frame.y == 0) > 0.6,
                "binomial_near_0_1": np.mean((fitted < 0.02) | (fitted > 0.98)) > 0.2,
                "gamma_near_zero": frame.y.max() < 1e-2,
            }[case],
            f"{case} data must exercise its boundary",
        ),
    )
    collector.check(
        "fitted means inside the family domain",
        lambda: check_that(
            bool(np.all(np.isfinite(fitted)) and np.all(fam.valid_mu(fitted))),
            "fitted means must be finite and valid",
        ),
    )
    collector.check(
        "convergence",
        lambda: check_that(
            fit.converged
            and expected.mgcv_converged
            and fit.scale_iterations == expected.n_iter,
            f"{fit.scale_iterations=} vs pinned {expected.n_iter}",
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
        ]

    for name, observed, target in fields(fit):
        collector.check(name, _close(observed, target, STRICT))
    if selected:
        tolerance = _flat_sp_tolerance(
            case_id,
            frame,
            weights,
            offset,
            _BOUNDARY_FORMULA,
            fam,
            _control(selected),
            sp,
        )
        collector.check(
            "sp", _close(fit.smoothing_params, expected.smoothing_params, tolerance)
        )
    _check_public_fit(collector, followed, fields)
    collector.raise_if_any(f"{case} {'selected' if selected else 'fixed'}")


def test_domain_exit_without_step_control_stops_like_pinned_bam(r_bridge):
    """Iteration two has no coefficient step control in bgam.fitd.

    The Gamma inverse-link selected tensor/factor-by fit's first proposal
    gives a negative mean, so ``sum(dev.resids)`` is ``NaN`` at the top of
    iteration two: pinned bam stops (``"Non-finite deviance"``) after its
    iteration-one trials only, and the discrete route raises at the same
    iteration rather than clipping the mean.
    """
    family, link = "gamma", "inverse"
    frame, weights, offset = _frame(family, link)
    trace = r_bridge.trace_bam_regular_working_model(
        _TENSOR_BY,
        frame,
        family=family,
        link=link,
        factor_names=("g",),
        weights=weights,
        offset=offset,
        allow_failure=True,
    )
    assert trace.failure is not None
    assert "Non-finite deviance" in trace.failure
    assert trace.fit is None
    assert {trial.iteration for trial in trace.trials} == {1}
    with pytest.raises(FloatingPointError, match="deviance at iteration 2"):
        fit_discrete_regular(
            parse_formula(_TENSOR_BY),
            _source(frame, weights, offset),
            _family(family, link),
            None,
            control=_control(True),
        )


def test_tie_replay_is_explicit_and_bounded():
    """``_oracle_replay_halvings`` exists only where bgam.fitd has step control."""
    frame, weights, offset = _frame("poisson", "log")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for schedule in ({2: 0}, {3: 30}, {3: -1}):
            with pytest.raises(ValueError, match="_oracle_replay_halvings"):
                fit_discrete_regular(
                    parse_formula(_SEPARATE),
                    _source(frame, weights, offset),
                    Poisson(),
                    _FIXED_SP[_SEPARATE],
                    control=_control(False),
                    _oracle_replay_halvings=schedule,
                )


def _flat_sp_case_inputs(case: str):
    if case == "poisson-low-counts":
        return _boundary_frame("poisson_low_counts")
    return _frame("binomial", "log")


@pytest.mark.parametrize("case", list(_REVIEWED_FLAT_SP))
def test_flat_sp_allowance_needs_reviewed_bytes_model_and_controls(monkeypatch, case):
    """A changed row loses the MODERATE vector; a changed model/control fails.

    The reviewed record is rebound to this platform's bytes so the guard runs
    on every platform.
    """
    reviewed = _REVIEWED_FLAT_SP[case]
    frame, weights, offset = _flat_sp_case_inputs(case)
    monkeypatch.setitem(
        _REVIEWED_FLAT_SP,
        case,
        replace(reviewed, digest=_input_digest(frame, weights, offset)),
    )
    family = reviewed.family_class(link=reviewed.link)
    control = _control(True)

    def tolerance(
        frame=frame,
        weights=weights,
        offset=offset,
        formula=reviewed.formula,
        family=family,
        control=control,
        sp=None,
        case=case,
    ):
        return _flat_sp_tolerance(
            case, frame, weights, offset, formula, family, control, sp
        )

    assert tolerance() is MODERATE
    assert tolerance(case="poisson-log-separate_re") is STRICT
    changed_y = frame.copy()
    changed_y.loc[3, "y"] = np.nextafter(changed_y.loc[3, "y"], np.inf)
    changed_x = frame.copy()
    changed_x.loc[0, "x"] = np.nextafter(changed_x.loc[0, "x"], np.inf)
    changed_g = frame.copy()
    changed_g.loc[[0, 1], "g"] = changed_g.loc[[1, 0], "g"].to_numpy()
    changed_weights = weights.copy()
    changed_weights[1] = np.nextafter(changed_weights[1], np.inf)
    changed_offset = offset.copy()
    changed_offset[2] = np.nextafter(changed_offset[2], -np.inf)
    for changed in (
        {"frame": changed_y},
        {"frame": changed_x},
        {"frame": changed_g},
        {"weights": changed_weights},
        {"offset": changed_offset},
    ):
        assert tolerance(**changed) is STRICT
    discretization = control.discretization
    for changed in (
        {"formula": reviewed.formula.replace("k=5", "k=6")},
        {"family": reviewed.family_class(link="identity")},
        {"sp": np.ones(3)},
        {"control": replace(control, batch_rows=18)},
        {"control": replace(control, working_model_epsilon=1e-8)},
        {"control": replace(control, uncertainty="fisher")},
        {"control": replace(control, discretization=replace(discretization, m=31))},
        {"control": _control(False)},
    ):
        with pytest.raises(AssertionError):
            tolerance(**changed)
