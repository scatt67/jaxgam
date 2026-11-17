"""Regular-family bgam.fitd controller semantics and the public discrete route."""

from __future__ import annotations

import inspect
import pickle
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

import jaxgam.execution.discrete_regular as discrete_regular
from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution import regular_working_model as regular
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.working_model import GaussianWorkingModelProposal
from tests.helpers import _AssertCollector, check_that
from tests.tolerances import STRICT

# --------------------------------------------------------------------------
# Controller semantics with scripted rows and proposals (no data, no R).
# --------------------------------------------------------------------------

_STATISTICS = regular.WorkingStatistics(np.eye(2), np.ones(2), 3.0)


class _ScriptedRows:
    """Return scripted deviances; record every evaluated coefficient vector."""

    def __init__(self, deviances, statistics=_STATISTICS):
        self.deviances = list(deviances)
        self.statistics = statistics
        self.evaluated: list[np.ndarray] = []

    def initial(self):
        return regular.WorkingEvaluation(self.deviances.pop(0), self.statistics)

    def evaluate(self, coefficients, *, statistics=True):
        self.evaluated.append(np.array(coefficients))
        return regular.WorkingEvaluation(
            self.deviances.pop(0), self.statistics if statistics else None
        )

    def response_variance(self):
        return 0.5


def _proposal(coefficients, step=(), gradient=None, criterion=1.0):
    step = np.asarray(step, dtype=float)
    gradient = np.zeros_like(step) if gradient is None else np.asarray(gradient)
    return GaussianWorkingModelProposal(
        np.asarray(coefficients, dtype=float),
        np.asarray(criterion),
        gradient,
        np.eye(len(step)),
        step,
        np.eye(2),
        np.arange(2, dtype=np.int32),
        np.asarray(2, dtype=np.int32),
        np.ones(2),
        np.asarray(0.5),
        np.asarray(0.25),
    )


def _penalty(beta, _trial):
    return float(beta @ beta)


def _state(completed=2, coefficients=(1.0, 1.0), accepted=(0.0, 0.0), deviance=4.0):
    return regular.RegularWorkingModelState(
        completed,
        np.zeros(0),
        np.zeros(0),
        np.asarray(coefficients, dtype=float),
        np.asarray(accepted, dtype=float),
        deviance,
        10.0,
        1.0,
    )


def _advance(state, rows, propose=None, **kwargs):
    if propose is None:

        def propose(_trial, _tolerance, _statistics):
            return _proposal([2.0, 2.0])

    return regular.advance_regular_working_model(
        state,
        rows,
        propose,
        _penalty,
        scale_estimated=kwargs.pop("scale_estimated", False),
        epsilon=kwargs.pop("epsilon", 1e-7),
        start=kwargs.pop("start", lambda _statistics, _variance: np.zeros(0)),
        _oracle_replay_halvings=kwargs.pop("_oracle_replay_halvings", None),
    )


def test_coefficient_step_halves_towards_accepted_until_not_worse():
    # dev0 + bSb0 = 4 + 0. Trial penalized deviances 5+2, 4.6+0.5, then 3+0.125.
    rows = _ScriptedRows([5.0, 4.6, 3.0])
    advance = _advance(_state(), rows)
    assert advance.coefficient_halvings == 2
    np.testing.assert_array_equal(advance.coefficients, [0.25, 0.25])
    np.testing.assert_array_equal(rows.evaluated[1], [0.5, 0.5])
    assert advance.penalized_deviance == 3.125
    margins = advance.iteration.step_control_margins
    np.testing.assert_allclose(margins, [0.75, 0.275, -0.21875], rtol=STRICT.rtol)
    assert advance.state.accepted_coefficients is advance.coefficients
    assert advance.state.accepted_deviance == 3.0


def test_non_finite_trial_deviance_is_halved_like_r():
    rows = _ScriptedRows([np.nan, 1.0])
    advance = _advance(_state(), rows)
    assert advance.coefficient_halvings == 1
    assert advance.iteration.step_control_margins[0] == np.inf


def test_step_control_stops_after_29_halvings_and_accepts():
    rows = _ScriptedRows([100.0] * 31)
    advance = _advance(_state(), rows)
    assert advance.coefficient_halvings == 29
    assert len(rows.evaluated) == 30
    assert len(rows.deviances) == 1


def test_replayed_halvings_follow_the_source_outcome_and_keep_margins():
    # Our rule would accept the first trial (5 + 2 <= 4 + 10 is false: it
    # halves); a replay of zero accepts it, a replay of two halves twice.
    rows = _ScriptedRows([5.0])
    advance = _advance(_state(), rows, _oracle_replay_halvings=0)
    assert advance.coefficient_halvings == 0
    np.testing.assert_array_equal(advance.coefficients, [1.0, 1.0])
    np.testing.assert_allclose(
        advance.iteration.step_control_margins, [0.75], rtol=STRICT.rtol
    )
    rows = _ScriptedRows([1.0, 1.0, 1.0])
    advance = _advance(_state(), rows, _oracle_replay_halvings=2)
    assert advance.coefficient_halvings == 2
    np.testing.assert_array_equal(advance.coefficients, [0.25, 0.25])
    assert len(advance.iteration.step_control_margins) == 3
    for halvings, completed in ((1, 1), (30, 2), (-1, 2)):
        with pytest.raises(ValueError, match="_oracle_replay_halvings"):
            _advance(
                _state(completed=completed),
                _ScriptedRows([1.0]),
                _oracle_replay_halvings=halvings,
            )


def test_fit_replays_only_the_scheduled_iterations():
    rows = _ScriptedRows([5.0, 4.0, 3.9, 3.0, 2.5])
    fit = regular.fit_regular_working_model(
        rows,
        lambda *_args: _proposal([0.5, 0.5]),
        _penalty,
        2,
        scale_estimated=False,
        epsilon=1e-7,
        max_iterations=3,
        start=lambda _statistics, _variance: np.zeros(0),
        _oracle_replay_halvings={3: 1},
    )
    assert [record.coefficient_halvings for record in fit.trace] == [0, 0, 1]
    # dev0 + bSb0 = 4.0 + 0.5; the replayed first trial (3.9 + 0.5) was not
    # worse, so its margin is negative: an override our rule would not take.
    assert fit.trace[2].step_control_margins[0] < 0.0


def test_iteration_two_has_no_step_control_and_non_finite_deviance_stops():
    rows = _ScriptedRows([100.0])
    advance = _advance(_state(completed=1, accepted=(0.0, 0.0)), rows)
    assert advance.coefficient_halvings == 0
    assert advance.iteration.step_control_margins == ()
    with pytest.raises(FloatingPointError, match="iteration 2"):
        _advance(_state(completed=1), _ScriptedRows([np.inf]))
    with pytest.raises(FloatingPointError, match="statistics"):
        _advance(
            _state(completed=1),
            _ScriptedRows(
                [1.0], regular.WorkingStatistics(np.full((2, 2), np.nan), np.ones(2), 1)
            ),
        )


def test_convergence_precedes_any_proposal_and_needs_the_scale_step():
    state = regular.RegularWorkingModelState(
        3,
        np.array([0.2, -1.0]),
        np.array([0.0, 0.5]),
        np.zeros(2),
        np.zeros(2),
        1.0,
        1.0,
        1.0,
    )

    def refuse(*_args):
        raise AssertionError("no proposal after convergence")

    converged = _advance(state, _ScriptedRows([1.0]), refuse)
    assert converged.converged
    assert converged.iteration.converged
    assert converged.iteration.trial_evaluations == 0
    # An estimated scale whose last step is large keeps iterating.
    advance = _advance(
        state,
        _ScriptedRows([1.0]),
        lambda *_args: _proposal([0.0, 0.0], step=[0.0, 0.0]),
        scale_estimated=True,
    )
    assert not advance.converged
    assert advance.proposal.evaluations == 1


def test_iteration_one_uses_start_and_rejects_non_finite_start():
    state = regular.RegularWorkingModelState.initial(2)
    seen = []

    def propose(trial, tolerance, statistics):
        seen.append((trial.copy(), tolerance, statistics))
        return _proposal([1.0, 2.0], step=[0.3])

    advance = _advance(
        state,
        _ScriptedRows([8.0]),
        propose,
        start=lambda statistics, _variance: np.array(
            [float(statistics.response_square_sum)]
        ),
    )
    ((trial, tolerance, statistics),) = seen
    np.testing.assert_array_equal(trial, [3.0])
    assert tolerance == 8.0 * np.sqrt(np.finfo(float).eps)
    assert statistics is _STATISTICS
    assert advance.state.accepted_coefficients is None
    assert np.isnan(advance.state.accepted_deviance)
    # crit <- (dev/exp(log.phi) - ldetS + ldetXXS)/2 with a known scale.
    assert advance.state.criterion == (8.0 - 0.25 + 0.5) / 2.0
    with pytest.raises(ValueError, match="start parameters"):
        _advance(
            state,
            _ScriptedRows([8.0]),
            propose,
            start=lambda _statistics, _variance: np.array([np.nan]),
        )


def test_iteration_limit_reports_truthfully_and_scans_final_deviance():
    rows = _ScriptedRows([5.0, 4.0, 3.0, 2.5])
    fit = regular.fit_regular_working_model(
        rows,
        lambda *_args: _proposal([0.5, 0.5]),
        _penalty,
        2,
        scale_estimated=False,
        epsilon=1e-7,
        max_iterations=3,
        start=lambda _statistics, _variance: np.zeros(0),
    )
    assert not fit.converged
    assert fit.n_iter == 3
    assert fit.deviance == 2.5
    assert len(fit.trace) == 3
    assert not fit.coefficients.flags.writeable
    rows = _ScriptedRows([5.0, 4.0, np.nan])
    with pytest.raises(FloatingPointError, match="final"):
        regular.fit_regular_working_model(
            rows,
            lambda *_args: _proposal([0.5, 0.5]),
            _penalty,
            2,
            scale_estimated=False,
            epsilon=1e-7,
            max_iterations=2,
            start=lambda _statistics, _variance: np.zeros(0),
        )


@pytest.mark.platform_sensitive
def test_source_sum_is_left_to_right_in_long_double():
    values = np.array([1.0, 1e-16, 1e-16, -1.0])
    expected = np.longdouble(0)
    for value in values:
        expected += np.longdouble(value)
    assert regular.source_sum(values) == expected
    assert regular.source_sum(values[2:], regular.source_sum(values[:2])) == expected
    assert regular.source_sum(np.empty(0), np.longdouble(2.5)) == 2.5
    assert np.isnan(regular.source_sum(np.array([1.0, np.nan])))


# --------------------------------------------------------------------------
# Public route (no R).
# --------------------------------------------------------------------------


def _frame(n=90):
    position = np.arange(n, dtype=float)
    x = (position + 0.2 * np.sin(position)) / n
    z = np.cos(position * 0.23)
    eta = 0.3 + np.sin(4.0 * x) + 0.5 * np.cos(3.0 * z)
    y = np.random.default_rng(2).poisson(np.exp(eta)).astype(float)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def _control(batch_rows=11, **kwargs):
    values = {
        "execution": "discrete",
        "batch_rows": batch_rows,
        "discretization": ContinuousDiscretizationControl(m=30),
    }
    values.update(kwargs)
    return FitControl(**values)


_FORMULA = "y ~ s(x,bs='cr',k=6) + s(z,bs='cr',k=5)"


def test_public_regular_fit_is_batch_invariant_and_pickles():
    frame = _frame()
    weights = np.linspace(0.5, 1.5, len(frame))
    offset = np.linspace(-0.1, 0.1, len(frame))
    collector = _AssertCollector()
    results = {}
    for batch_rows in (7, 90):
        for uncertainty in ("none", "fisher", "covariance"):
            results[batch_rows, uncertainty] = GAM(
                _FORMULA,
                family="poisson",
                control=_control(
                    batch_rows,
                    uncertainty=uncertainty,
                    smoothing_strategy="working_model_reml",
                ),
            ).fit(
                DataFrameRowSource(frame, response="y", weights=weights, offset=offset),
                result="prediction",
            )
    baseline = results[7, "none"]
    for key, result in results.items():
        for field in (
            "coefficients",
            "smoothing_params",
            "deviance",
            "scale",
            "edf_total",
            "score",
        ):
            collector.check(
                f"{key}/{field}",
                lambda result=result, field=field: np.testing.assert_allclose(
                    getattr(result, field),
                    getattr(baseline, field),
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
    collector.check(
        "route and strategy",
        lambda: check_that(
            baseline.execution_route == "discrete"
            and baseline.smoothing_strategy == "working_model_reml"
            and baseline.lambda_strategy == "working_model_reml"
            and baseline.execution_path
            == "jax_actions_performance_iteration_numpy_postestimation"
            and "poisson discrete Fisher performance iteration"
            in baseline.convergence_info
            and baseline.converged
            and baseline.source_scans >= baseline.n_iter
            and baseline.batches_scanned
            == baseline.source_scans * int(np.ceil(len(frame) / 7))
            and baseline.fit_peak_bytes <= baseline.fit_memory_budget_bytes,
            "regular discrete result metadata",
        ),
    )
    held_out = pd.DataFrame({"x": [0.05, 0.4, 0.95], "z": [-0.9, 0.1, 0.8]})
    fisher, covariance = results[7, "fisher"], results[7, "covariance"]
    fit_f, se_f = fisher.predict(held_out, pred_type="link", se_fit=True)
    fit_c, se_c = covariance.predict(held_out, pred_type="link", se_fit=True)
    collector.check(
        "fisher vs covariance prediction",
        lambda: (
            np.testing.assert_allclose(
                fit_f, fit_c, rtol=STRICT.rtol, atol=STRICT.atol
            ),
            np.testing.assert_allclose(se_f, se_c, rtol=STRICT.rtol, atol=STRICT.atol),
        ),
    )
    collector.check(
        "response predictions are positive rates",
        lambda: check_that(
            bool(np.all(baseline.predict(held_out, pred_type="response") > 0)),
            "Poisson response predictions must be positive",
        ),
    )
    restored = pickle.loads(pickle.dumps(covariance))
    collector.check(
        "pickle",
        lambda: np.testing.assert_allclose(
            restored.predict(held_out, pred_type="link", se_fit=True)[1],
            se_c,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any("public regular discrete route")


def test_public_fixed_sp_route_names_its_strategy():
    frame = _frame()
    result = GAM(
        _FORMULA, family=Gamma(link="log"), sp=[0.7, 1.4], control=_control()
    ).fit(
        DataFrameRowSource(frame.assign(y=frame.y + 0.5), response="y"),
        result="prediction",
    )
    assert result.smoothing_strategy == result.lambda_strategy == "fixed"
    assert "Gamma discrete Fisher performance iteration (fixed sp)" in (
        result.convergence_info
    )
    np.testing.assert_array_equal(result.smoothing_params, [0.7, 1.4])
    assert result.scale > 0.0
    assert result.working_model_proposal_evaluations >= result.n_iter - 1


def test_iteration_limit_warns_and_reports_non_convergence():
    frame = _frame()
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = GAM(
            _FORMULA,
            family="binomial",
            control=_control(
                smoothing_strategy="working_model_reml", working_model_max_iterations=2
            ),
        ).fit(
            DataFrameRowSource(
                frame.assign(y=(frame.y > 1).astype(float)), response="y"
            ),
            result="prediction",
        )
    assert not result.converged
    assert result.n_iter == 2
    assert "did not converge" in result.convergence_info
    # Two iteration scans plus the final deviance-only scan.
    assert result.source_scans == 3


class _NoScanSource(DataFrameRowSource):
    def scan(self, _batch_rows):
        raise AssertionError("an unsupported family must reject before scanning")


class _UnportedExtendedFamily(NegativeBinomial):
    """An extended family without the discrete performance-iteration port."""

    def execution_capabilities(self):
        return replace(
            super().execution_capabilities(), extended_performance_iteration=False
        )


@pytest.mark.parametrize(
    ("family", "message"),
    [
        (_UnportedExtendedFamily(), "negative binomial"),
        (Poisson(link="logit"), "Poisson"),
        (Gaussian(link="log"), "Gaussian identity"),
        (Binomial(link="sqrt"), "Binomial"),
    ],
    ids=["unported-extended", "poisson-logit", "gaussian-log", "binomial-sqrt"],
)
def test_unsupported_families_reject_before_scanning(family, message):
    with pytest.raises(NotImplementedError, match=message):
        GAM(_FORMULA, family=family, sp=[1.0, 1.0], control=_control()).fit(
            _NoScanSource(_frame(), response="y"), result="prediction"
        )


@pytest.mark.parametrize(
    ("family", "response", "error", "message"),
    [
        ("poisson", -1.0, ValueError, "initialization rejects"),
        ("gamma", 0.0, ValueError, "initialization rejects"),
        ("binomial", 1.5, ValueError, "initialization rejects"),
    ],
)
def test_invalid_responses_are_initialization_errors(family, response, error, message):
    frame = _frame()
    frame.loc[3, "y"] = response
    if family != "poisson":
        frame["y"] = np.where(frame.index == 3, response, 0.5)
    with pytest.raises(error, match=message):
        GAM(_FORMULA, family=family, sp=[1.0, 1.0], control=_control()).fit(
            DataFrameRowSource(frame, response="y"), result="prediction"
        )


def test_budget_rejects_before_device_transfer(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("device transfer must not run")

    monkeypatch.setattr(discrete_regular, "to_device_discrete_design", refuse)
    with pytest.raises(MemoryError, match="Discrete fit workspace"):
        GAM(
            _FORMULA,
            family="poisson",
            sp=[1.0, 1.0],
            control=_control(memory_budget_bytes=1),
        ).fit(DataFrameRowSource(_frame(), response="y"), result="prediction")


def test_constant_gamma_response_has_no_scale_start():
    """var(y) = 0 makes R's log(var(G$y)*.05) start -Inf: fail explicitly."""
    with pytest.raises(FloatingPointError, match=r"log\(var\(y\)\*0\.05\)"):
        GAM(_FORMULA, family="gamma", sp=[1.0, 1.0], control=_control()).fit(
            DataFrameRowSource(_frame().assign(y=2.0), response="y"),
            result="prediction",
        )


def test_oracle_replay_hook_is_unreachable_from_the_public_api(monkeypatch):
    """``_oracle_replay_halvings`` is a test/oracle hook only.

    No public signature, control or result field carries it, and the public
    route never passes it, so every public fit decides each step itself.
    """
    for signature in (inspect.signature(GAM.__init__), inspect.signature(GAM.fit)):
        assert not any("replay" in name for name in signature.parameters)
    for control in (FitControl, ContinuousDiscretizationControl):
        assert not any("replay" in field.name for field in fields(control))
    calls = []
    original = discrete_regular.fit_discrete_regular

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(discrete_regular, "fit_discrete_regular", spy)
    result = GAM(_FORMULA, family="poisson", sp=[1.0, 1.0], control=_control()).fit(
        DataFrameRowSource(_frame(), response="y"), result="prediction"
    )
    assert calls
    assert all(
        "_oracle_replay_halvings" not in kwargs and "replay" not in str(kwargs)
        for kwargs in calls
    )
    assert not any("replay" in name for name in vars(result))
    assert "replay" not in result.convergence_info
