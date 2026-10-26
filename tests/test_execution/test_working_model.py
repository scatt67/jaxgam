"""Gaussian bgam.fitd orchestration, live array budgets and result ownership."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import jaxgam.execution.working_model as working
from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.formula.continuous_setup import build_continuous_discrete_setup
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector, check_that
from tests.test_fitting.test_working_model import (
    check_newton_step,
    ldet_tolerance,
    penalty_determinant,
    source_working_metadata,
)
from tests.tolerances import LOOSE, MODERATE, STRICT

_COUPLED = "y ~ te(x,z,bs='cr',k=3)"


def _frame():
    position = np.arange(97, dtype=float)
    x = (position + 0.2 * np.sin(position * 0.7)) / len(position)
    z = np.cos(position * 0.23) + 0.07 * np.sin(position * 0.91)
    y = 0.4 - 0.3 * z + np.sin(4.5 * x) + 0.05 * np.cos(position)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def _control(batch_rows=13, **kwargs):
    values = {
        "execution": "discrete",
        "smoothing_strategy": "working_model_reml",
        "discretization": ContinuousDiscretizationControl(m=30),
        "batch_rows": batch_rows,
    }
    values.update(kwargs)
    return FitControl(**values)


@pytest.mark.parametrize(
    "case",
    [
        "single",
        "weighted",
        "zero",
        "separate",
        "separate_identified",
        "coupled",
        "coupled_identified",
        "numeric_by",
    ],
)
def test_gaussian_performance_fit_matches_pinned_bam(r_bridge, case):
    frame = _frame()
    formula = "y ~ z + s(x,bs='cr',k=6)"
    weights = None
    offset = None
    if case in ("weighted", "zero"):
        weights = np.linspace(0.6, 1.4, len(frame))
        if case == "zero":
            weights[np.arange(len(frame)) % 7 == 0] = 0.0
        offset = np.linspace(-0.1, 0.2, len(frame))
    elif case in ("separate", "separate_identified"):
        if case == "separate_identified":
            frame["y"] += 0.35 * np.sin(3 * frame.z)
        formula = "y ~ s(x,bs='cr',k=6)+s(z,bs='cr',k=6)"
    elif case in ("coupled", "coupled_identified"):
        if case == "coupled_identified":
            frame["y"] += 0.35 * np.sin(3 * frame.z) + 0.4 * np.sin(
                3 * frame.x
            ) * np.cos(2 * frame.z)
        formula = "y ~ te(x,z,bs='cr',k=3)"
    elif case == "numeric_by":
        formula = "y ~ z + s(x,by=z,bs='cr',k=6)"
    expected = r_bridge.fit_bam_discrete_gaussian(
        formula,
        frame,
        m=30,
        weights=weights,
        offset=offset,
    )
    result = GAM(formula, control=_control()).fit(
        DataFrameRowSource(frame, response="y", weights=weights, offset=offset),
        result="prediction",
    )
    # The coupled te() and separate s(z) smoothing vectors are flat in their
    # large sp (1.8e5 and 1.8e7): a one-ULP change of pinned R's own start
    # moves R's coupled sp 1.4e5x STRICT (0.14x MODERATE), and pinned Linux R
    # lands 1.1x STRICT from JaxGAM's separate sp. Their controller arithmetic
    # received the four documented source passes noted in
    # tests/test_fitting/test_working_model.py. Physical fields stay STRICT.
    smoothing = MODERATE if case in ("coupled", "separate") else STRICT
    collector = _AssertCollector()
    for name, observed, target, tolerance in (
        ("coefficients", result.coefficients, expected.coefficients, STRICT),
        (
            "smoothing parameters",
            result.smoothing_params,
            expected.smoothing_params,
            smoothing,
        ),
        *[
            (field, getattr(result, field), getattr(expected, field), STRICT)
            for field in ("deviance", "scale", "edf_total", "score")
        ],
    ):
        collector.check(
            name,
            lambda observed=observed, target=target, tolerance=tolerance: (
                np.testing.assert_allclose(
                    observed,
                    target,
                    rtol=tolerance.rtol,
                    atol=tolerance.atol,
                )
            ),
        )
    collector.check(
        "route and convergence",
        lambda: check_that(
            result.converged
            and expected.mgcv_converged
            and result.n_iter == expected.n_iter
            and result.smoothing_strategy
            == result.lambda_strategy
            == "working_model_reml"
            and result.execution_route == "discrete"
            and result.fitting_source_scans == 2
            and result.working_model_proposal_evaluations >= result.n_iter - 1,
            "Gaussian route must report actual finite iterations and strategy",
        ),
    )
    collector.raise_if_any(case)


@pytest.fixture
def statistics():
    frame = _frame()
    formula = "y ~ z + s(x,bs='cr',k=6)"
    setup = build_continuous_discrete_setup(
        parse_formula(formula),
        DataFrameRowSource(frame, response="y"),
        m=30,
    )
    snapped = {
        name: values[group.index.selector]
        for group in setup.compact.groups
        for name, values in zip(group.names, group.index.values, strict=True)
    }
    X = setup.predict_spec.build_predict_matrix(pd.DataFrame(snapped))
    penalties = setup.penalties
    setup.cleanup()
    y = frame.y.to_numpy()
    return (
        X.T @ X,
        X.T @ y,
        float(y @ y),
        float(np.var(y, ddof=1)),
        penalties,
        len(frame),
    )


def _fit_statistics(statistics, available=512 * 1024 * 1024, control=None, **kwargs):
    return working.fit_gaussian_working_model_statistics(
        *statistics,
        control=_control() if control is None else control,
        available_workspace_bytes=available,
        **kwargs,
    )


def test_gaussian_iteration_limit_is_truthful_and_fixed_sp_owns_mode(statistics):
    selected = _fit_statistics(
        statistics, control=_control(working_model_max_iterations=1)
    )
    assert not selected.converged
    assert selected.n_iter == selected.proposal_evaluations == len(selected.trace) == 1
    frame = _frame()
    formula = "y ~ z + s(x,bs='cr',k=6)"
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = GAM(formula, control=_control(working_model_max_iterations=1)).fit(
            DataFrameRowSource(frame, response="y"),
            result="prediction",
        )
    assert not result.converged
    assert "did not converge" in result.convergence_info
    fixed = GAM(formula, sp=[0.55], control=_control()).fit(
        DataFrameRowSource(frame, response="y"),
        result="prediction",
    )
    assert fixed.smoothing_strategy == fixed.lambda_strategy == "fixed"
    assert fixed.working_model_proposal_evaluations == 0


def test_gaussian_budget_rejects_before_argument_transfer(monkeypatch, statistics):
    def reject(*_args, **_kwargs):
        raise AssertionError("budget must reject before working metadata transfer")

    monkeypatch.setattr(working.WorkingModelCriterionMetadata, "from_structure", reject)
    with pytest.raises(MemoryError, match="arguments require"):
        _fit_statistics(statistics, available=1)


def test_gaussian_compiled_budget_rejects_before_proposal(monkeypatch, statistics):
    selected = _fit_statistics(statistics)
    memory = selected.memory_plan
    preliminary = (
        memory.metadata_host_peak_bytes
        + memory.metadata_device_peak_bytes
        + memory.statistic_device_bytes
        + memory.state_and_trace_bytes
    )
    assert memory.compiled_temporary_bytes > 0
    assert memory.additional_peak_bytes > preliminary
    original = working.gaussian_working_model_proposal
    executions = []

    class Wrapper:
        def lower(self, *args, **kwargs):
            executable = original.lower(*args, **kwargs).compile()

            class Executable:
                def memory_analysis(self):
                    return executable.memory_analysis()

                def __call__(self, *args, **kwargs):
                    executions.append(True)
                    return executable(*args, **kwargs)

            return SimpleNamespace(compile=lambda: Executable())

    monkeypatch.setattr(working, "gaussian_working_model_proposal", Wrapper())
    with pytest.raises(MemoryError, match="compiled workspace"):
        _fit_statistics(statistics, available=preliminary)
    assert executions == []


@pytest.mark.parametrize(
    "field",
    [
        "coefficients",
        "criterion",
        "gradient",
        "hessian",
        "step",
        "memory_analysis",
        "factor_upper",
        "factor_diagonal",
        "factor_rank",
        "logdet_hessian",
        "logdet_penalty",
    ],
)
def test_gaussian_nonfinite_proposals_never_converge(monkeypatch, statistics, field):
    original = working.gaussian_working_model_proposal

    class Wrapper:
        def lower(self, *args, **kwargs):
            executable = original.lower(*args, **kwargs).compile()

            class Executable:
                def memory_analysis(self):
                    return (
                        None
                        if field == "memory_analysis"
                        else executable.memory_analysis()
                    )

                def __call__(self, *args, **kwargs):
                    proposal = executable(*args, **kwargs)
                    return proposal._replace(
                        **{
                            field: np.full_like(
                                getattr(proposal, field),
                                -1 if field == "factor_rank" else np.nan,
                            )
                        }
                    )

            return SimpleNamespace(compile=lambda: Executable())

    monkeypatch.setattr(working, "gaussian_working_model_proposal", Wrapper())
    error = NotImplementedError if field == "memory_analysis" else FloatingPointError
    with pytest.raises(error, match=r"report compiled memory|Non-finite"):
        _fit_statistics(statistics)


@pytest.mark.parametrize("field", ["gram", "rhs", "yy", "variance", "shape"])
def test_gaussian_invalid_statistics_reject(statistics, field):
    values = list(statistics)
    if field == "shape":
        values[0] = np.eye(3)
    else:
        index = {"gram": 0, "rhs": 1, "yy": 2, "variance": 3}[field]
        values[index] = np.full_like(values[index], np.nan)
    with pytest.raises((ValueError, FloatingPointError), match="statistics"):
        _fit_statistics(tuple(values))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smoothing_strategy": "bad"},
        {"working_model_max_iterations": True},
        {"working_model_max_iterations": 0},
        {"working_model_epsilon": True},
        {"working_model_epsilon": np.inf},
        {"working_model_epsilon": 0.0},
    ],
)
def test_gaussian_public_control_validates(kwargs):
    with pytest.raises(ValueError, match=r"smoothing_strategy|working_model"):
        _control(**kwargs)


def test_gaussian_results_covariance_pickle_and_batch_scan_ownership():
    frame = _frame()
    formula = "y ~ z + s(x,bs='cr',k=6)"
    collector = _AssertCollector()
    baseline = None
    for uncertainty in ("none", "fisher", "covariance"):
        for batch_rows in (5, 17):
            result = GAM(
                formula, control=_control(batch_rows, uncertainty=uncertainty)
            ).fit(
                DataFrameRowSource(frame, response="y"),
                result="prediction",
            )
            if baseline is None:
                baseline = result
            for field in (
                "coefficients",
                "smoothing_params",
                "deviance",
                "scale",
                "score",
                "edf_total",
            ):
                collector.check(
                    f"{uncertainty}/B{batch_rows}/{field}",
                    lambda field=field, result=result, baseline=baseline: (
                        np.testing.assert_allclose(
                            getattr(result, field),
                            getattr(baseline, field),
                            rtol=STRICT.rtol,
                            atol=STRICT.atol,
                        )
                    ),
                )
            assert result.fitting_source_scans == 2
            assert result.fitting_batches_scanned == 2 * int(
                np.ceil(len(frame) / batch_rows)
            )
            restored = pickle.loads(pickle.dumps(result))
            assert not restored.coefficients.flags.writeable
            assert not np.shares_memory(restored.coefficients, result.coefficients)
            collector.check(
                "pickle prediction",
                lambda result=result, restored=restored: np.testing.assert_allclose(
                    restored.predict(frame),
                    result.predict(frame),
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
            if uncertainty != "none":
                fixed = GAM(
                    formula,
                    sp=result.smoothing_params,
                    control=_control(batch_rows, uncertainty=uncertainty),
                ).fit(
                    DataFrameRowSource(frame, response="y"),
                    result="prediction",
                )
                fit, se = result.predict(frame, se_fit=True)
                expected_fit, expected_se = fixed.predict(frame, se_fit=True)
                collector.check(
                    "fixed accepted-sp prediction",
                    lambda fit=fit, expected_fit=expected_fit: (
                        np.testing.assert_allclose(
                            fit,
                            expected_fit,
                            rtol=STRICT.rtol,
                            atol=STRICT.atol,
                        )
                    ),
                )
                # The source iteration owns finite scale. Remove its scale
                # difference to compare the exact retained inverse-H geometry.
                collector.check(
                    "accepted-sp SE geometry",
                    lambda se=se, expected_se=expected_se, result=result, fixed=fixed: (
                        np.testing.assert_allclose(
                            se / np.sqrt(result.scale),
                            expected_se / np.sqrt(fixed.scale),
                            rtol=STRICT.rtol,
                            atol=STRICT.atol,
                        )
                    ),
                )
                collector.check(
                    "accepted-sp covariance geometry",
                    lambda result=result, fixed=fixed: np.testing.assert_allclose(
                        result.materialize_covariance(1_000_000) / result.scale,
                        fixed.materialize_covariance(1_000_000) / fixed.scale,
                        rtol=STRICT.rtol,
                        atol=STRICT.atol,
                    ),
                )
    collector.raise_if_any("Gaussian working-model ownership")


def test_gaussian_nearly_exact_deviance_uses_selected_residuals():
    x = np.linspace(-1.0, 1.0, 97)
    frame = pd.DataFrame({"x": x, "y": 2.0 + 0.5 * x + 1e-7 * np.sin(7 * x)})
    result = GAM(
        "y~x",
        control=_control(
            discretization=ContinuousDiscretizationControl(m=200),
        ),
    ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
    residual = frame.y.to_numpy() - result.predict(frame)
    expected = float(residual @ residual)
    np.testing.assert_allclose(
        result.deviance, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    # Multiplication order differs between the selected JAX action and CPU
    # prediction; this absolute gate still detects cancellation of RSS.
    assert abs(result.deviance - expected) < 1e-20
    assert 0.0 < result.deviance < 1e-11
    assert result.fitting_source_scans == 2


def test_gaussian_selected_repeated_source_preserves_scan_ordinals():
    from jaxgam.data.source import ArrayRowSource

    frame = _frame()
    selection = np.r_[np.arange(96, -1, -2), np.arange(1, 97, 2), [7, 2, 40, 1, 7]]
    source = ArrayRowSource(
        {"x": frame.x.to_numpy(), "z": frame.z.to_numpy()},
        y=frame.y.to_numpy(),
        row_selection=selection,
    )
    selected_frame = frame.iloc[selection].reset_index(drop=True)
    collector = _AssertCollector()
    baseline = GAM("y~z+s(x,bs='cr',k=6)", control=_control(17)).fit(
        DataFrameRowSource(selected_frame, response="y"),
        result="prediction",
    )
    for batch_rows in (5, 13):
        actual = GAM("y~z+s(x,bs='cr',k=6)", control=_control(batch_rows)).fit(
            source,
            result="prediction",
        )
        for field in ("coefficients", "smoothing_params", "scale", "score", "deviance"):
            collector.check(
                f"B{batch_rows}/{field}",
                lambda actual=actual, field=field: np.testing.assert_allclose(
                    getattr(actual, field),
                    getattr(baseline, field),
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
        assert actual.n == len(selection)
        assert actual.fitting_source_scans == 2
    collector.raise_if_any("selected repeated working-model source")


def test_rank_dropped_numeric_by_covariance_prediction_and_pickle(r_bridge):
    frame = _frame()
    formula = "y~z+s(x,by=z,bs='cr',k=6)"
    held_out = pd.DataFrame(
        {"x": [0.07, 0.31, 0.66, 0.91], "z": [-0.8, 0.2, 1.1, -0.4]}
    )
    expected = r_bridge.fit_bam_discrete_gaussian(
        formula,
        frame,
        m=30,
        newdata=held_out,
    )
    collector = _AssertCollector()
    baseline = None
    for uncertainty in ("none", "fisher", "covariance"):
        for batch_rows in (5, 17):
            result = GAM(
                formula, control=_control(batch_rows, uncertainty=uncertainty)
            ).fit(
                DataFrameRowSource(frame, response="y"),
                result="prediction",
            )
            restored = pickle.loads(pickle.dumps(result))
            if baseline is None:
                baseline = result
            for field in (
                "coefficients",
                "smoothing_params",
                "scale",
                "score",
                "edf_total",
            ):
                collector.check(
                    f"{uncertainty}/B{batch_rows}/{field}",
                    lambda field=field, result=result: np.testing.assert_allclose(
                        getattr(result, field),
                        getattr(expected, field),
                        rtol=STRICT.rtol,
                        atol=STRICT.atol,
                    ),
                )
            assert result.converged
            assert result.fitting_source_scans == 2
            assert not restored.coefficients.flags.writeable
            assert not np.shares_memory(result.coefficients, restored.coefficients)
            collector.check(
                f"{uncertainty}/B{batch_rows}/heldout mean",
                lambda restored=restored: np.testing.assert_allclose(
                    restored.predict(held_out, pred_type="link"),
                    expected.prediction,
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
            if uncertainty == "none":
                continue
            collector.check(
                f"{uncertainty}/B{batch_rows}/covariance",
                lambda restored=restored: np.testing.assert_allclose(
                    restored.materialize_covariance(memory_budget_bytes=1_000_000),
                    expected.covariance,
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
            collector.check(
                f"{uncertainty}/B{batch_rows}/heldout SE",
                lambda restored=restored: np.testing.assert_allclose(
                    restored.predict(held_out, pred_type="link", se_fit=True)[1],
                    expected.prediction_se,
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
            if uncertainty == "fisher":
                factor = restored._predictor._fisher_qr_factor
                assert factor is not None
                assert len(factor.R) < len(result.coefficients)
                assert not factor.R.flags.writeable
                assert not factor.keep.flags.writeable
                assert not factor.pivots.flags.writeable
                assert all(
                    not values.flags.writeable
                    for _, _, _, values in restored._predictor._fisher_transforms
                )
    collector.raise_if_any("source rank-drop public ownership")


def test_rejected_trial_factor_is_released_before_next_execution(
    monkeypatch, statistics
):
    import weakref

    references = []
    live_at_execution = []
    p = statistics[0].shape[0]
    real_asarray = np.asarray
    original = working.gaussian_working_model_proposal

    class TrackedFactor(np.ndarray):
        def copy(self, *args, **kwargs):
            result = super().copy(*args, **kwargs)
            references.append(weakref.ref(result))
            return result

    def track_factor(value, *args, **kwargs):
        result = real_asarray(value, *args, **kwargs)
        if isinstance(value, jax.Array) and result.shape == (p, p):
            return result.view(TrackedFactor)
        return result

    class Wrapper:
        def lower(self, *args, **kwargs):
            executable = original.lower(*args, **kwargs).compile()

            class Executable:
                def memory_analysis(self):
                    return executable.memory_analysis()

                def __call__(self, *args, **kwargs):
                    live_at_execution.append(
                        sum(reference() is not None for reference in references)
                    )
                    return executable(*args, **kwargs)

            return SimpleNamespace(compile=lambda: Executable())

    monkeypatch.setattr(working, "gaussian_working_model_proposal", Wrapper())
    monkeypatch.setattr(working.np, "asarray", track_factor)
    selected = _fit_statistics(statistics)
    assert any(iteration.trial_evaluations > 1 for iteration in selected.trace)
    assert max(live_at_execution) == 1
    assert not selected.factor_upper.flags.writeable
    assert not selected.factor_pivots.flags.writeable
    assert not selected.factor_diagonal.flags.writeable
    assert all(
        not values.flags.writeable
        for _, _, _, values in selected.coefficient_transforms
    )


def _extreme_start_inputs(zero_weights):
    frame = _frame()
    weights = np.linspace(0.6, 1.4, len(frame))
    if zero_weights:
        weights[np.arange(len(frame)) % 7 == 0] = 0.0
    return frame, weights, np.linspace(-0.1, 0.2, len(frame))


_EXTREME_STARTS = {"low": (1e-10, 100.0), "high": (1e-4, 1e4)}


_EXTREME_START_CASES = [
    pytest.param(
        zero_weights,
        start,
        id=f"{'zeros' if zero_weights else 'ordinary'}-{label}",
    )
    for zero_weights in (False, True)
    for label, start in _EXTREME_STARTS.items()
]


def _trace_extreme_start(r_bridge, fixed, zero_weights, start):
    frame, weights, offset = _extreme_start_inputs(zero_weights)
    trace = r_bridge.trace_bam_working_model(
        _COUPLED,
        frame,
        initial_sp=np.asarray(start),
        initial_scale=float(np.exp(-3.0)),
        scale=float(np.exp(-3.0)) if fixed else -1.0,
        weights=weights,
        offset=offset,
    )
    statistics = (
        jnp.asarray(trace.state.gram),
        jnp.asarray(trace.state.rhs),
        jnp.asarray(trace.state.response_square_sum),
    )
    return trace, source_working_metadata(trace.state), statistics


@pytest.mark.parametrize("fixed", [False, True], ids=["free", "fixed"])
@pytest.mark.parametrize(("zero_weights", "start"), _EXTREME_START_CASES)
def test_extreme_start_trials_match_pinned_sl_fitchol_at_matched_state(
    r_bridge, fixed, zero_weights, start
):
    """Every bgam.fitd Sl.fitChol trial from R's complete frozen state.

    The kernels and host DPSTRF receive R's own Sl.setup S/rS/pcrossprod(rS),
    reparameterized qrx moments, y.norm2, trial rho/log(phi), tolerance and
    exp(rho); nothing comes from JaxGAM's setup. Gradient and Hessian are
    MODERATE (their trace terms run through R's own GEMM and long-double sums
    on penalized Hessians with condition up to 2.7e7, after the four
    documented source passes); ldetS follows ``ldet_tolerance``; coefficients
    and ldetXXS stay STRICT; steps use the conditioning-aware gates.
    """
    trace, metadata, statistics = _trace_extreme_start(
        r_bridge, fixed, zero_weights, start
    )
    ldet = ldet_tolerance(r_bridge)
    collector = _AssertCollector()
    for index, trial in enumerate(trace.trials):
        proposal = trial.proposal
        result = working.evaluate_gaussian_working_model_proposal(
            proposal.log_smoothing
            if fixed
            else np.r_[proposal.log_smoothing, proposal.log_scale],
            *statistics,
            metadata,
            scale_is_fixed=fixed,
            log_scale=proposal.log_scale,
            tolerance=proposal.tolerance,
            smoothing=proposal.smoothing,
        )
        _, first, second = penalty_determinant(
            proposal.log_smoothing, metadata, proposal.smoothing
        )
        label = f"trial {index} (iteration {trial.iteration})"
        for name, observed, target, tolerance in (
            ("coefficients", result.coefficients, proposal.coefficients, STRICT),
            ("gradient", result.gradient, proposal.gradient, MODERATE),
            ("hessian", result.hessian, proposal.hessian, MODERATE),
            ("ldetS", result.logdet_penalty, proposal.logdet_penalty, ldet),
            ("ldetS first derivative", first, trial.penalty_first, ldet),
            ("ldetS second derivative", second, trial.penalty_second, ldet),
            ("ldetXXS", result.logdet_hessian, proposal.logdet_hessian, STRICT),
        ):
            collector.check(
                f"{label} {name}",
                lambda observed=observed, target=target, tolerance=tolerance: (
                    np.testing.assert_allclose(
                        observed, target, rtol=tolerance.rtol, atol=tolerance.atol
                    )
                ),
            )
        check_newton_step(
            collector,
            label,
            (result.gradient, result.hessian, result.step),
            (proposal.gradient, proposal.hessian, proposal.step),
            proposal.tolerance,
        )
    collector.raise_if_any(
        f"matched Sl.fitChol trials {fixed=} {zero_weights=} {start}"
    )


@pytest.mark.parametrize(("zero_weights", "start"), _EXTREME_START_CASES)
def test_extreme_start_controller_replays_pinned_bgam_fitd(
    r_bridge, zero_weights, start
):
    """Restart the JAX controller from R's carried state at every iteration.

    At outer iteration k the controller starts from R's lsp0, Nstep, coef,
    devold and crit with R's complete frozen state and exp(rho), and must
    make R's convergence/halving decisions and reach R's state for k+1.
    Proposal-field parity is the matched-state trial gate above.
    """
    trace, metadata, statistics = _trace_extreme_start(
        r_bridge, False, zero_weights, start
    )
    state = trace.state
    trials = {
        np.r_[t.proposal.log_smoothing, t.proposal.log_scale].tobytes(): t
        for t in trace.trials
    }

    def propose(trial, tolerance):
        record = trials.get(np.asarray(trial).tobytes())
        check_that(record is not None, f"controller trial {trial} is not pinned R's")
        check_that(
            tolerance == record.proposal.tolerance,
            f"tolerance {tolerance} != pinned {record.proposal.tolerance}",
        )
        return working.evaluate_gaussian_working_model_proposal(
            trial,
            *statistics,
            metadata,
            tolerance=tolerance,
            smoothing=record.proposal.smoothing,
        )

    def advance(carried):
        return working.advance_gaussian_working_model(
            carried,
            propose,
            state.rhs,
            state.response_square_sum,
            state.n_obs,
            state.null_dimension,
            1e-7,
        )

    collector = _AssertCollector()
    iterations = sorted({trial.iteration for trial in trace.trials})
    groups = {k: [t for t in trace.trials if t.iteration == k] for k in iterations}
    coefficients = np.zeros(len(state.rhs))
    for position, k in enumerate(iterations):
        first, last = groups[k][0], groups[k][-1].proposal
        carried = working.WorkingModelState(
            k - 1,
            first.accepted,
            first.step,
            coefficients,
            first.previous_deviance,
            first.criterion,
        )
        coefficients = last.coefficients
        outcome = []
        collector.check(
            f"iteration {k} replay",
            lambda outcome=outcome, carried=carried: outcome.append(advance(carried)),
        )
        if not outcome:
            continue
        (result,) = outcome
        collector.check(
            f"iteration {k} decisions",
            lambda result=result, k=k: check_that(
                not result.converged and result.proposal_evaluations == len(groups[k]),
                f"converged={result.converged}, trials="
                f"{result.proposal_evaluations} vs pinned {len(groups[k])}",
            ),
        )
        checks = [
            ("penalized deviance", result.deviance, first.deviance, STRICT),
            (
                "accepted parameters",
                result.state.parameters,
                np.r_[last.log_smoothing, last.log_scale],
                STRICT,
            ),
            ("coefficients", result.state.coefficients, last.coefficients, STRICT),
        ]
        if position + 1 < len(iterations):
            following = groups[iterations[position + 1]][0]
            # crit carries ldetS/ldetXXS from near-singular penalty states,
            # where R's own provider rounding remains after the four documented
            # source passes; it only sets the next active-coordinate tolerance.
            checks.append(
                ("criterion", result.state.criterion, following.criterion, MODERATE)
            )
        for name, observed, target, tolerance in checks:
            collector.check(
                f"iteration {k} {name}",
                lambda observed=observed, target=target, tolerance=tolerance: (
                    np.testing.assert_allclose(
                        observed, target, rtol=tolerance.rtol, atol=tolerance.atol
                    )
                ),
            )
    last_trial = groups[iterations[-1]][-1]
    last = last_trial.proposal
    # crit only sets the next proposal's tolerance, which convergence skips.
    criterion = (
        last_trial.deviance / np.exp(last.log_scale)
        - last.logdet_penalty
        + last.logdet_hessian
    ) / 2.0
    final = advance(
        working.WorkingModelState(
            iterations[-1],
            np.r_[last.log_smoothing, last.log_scale],
            last.step,
            last.coefficients,
            last_trial.deviance,
            criterion,
        )
    )
    collector.check(
        "convergence",
        lambda: check_that(
            final.converged
            and trace.fit.mgcv_converged
            and iterations[-1] + 1 == trace.fit.n_iter,
            f"converged={final.converged} at {iterations[-1] + 1}; pinned "
            f"{trace.fit.mgcv_converged} at {trace.fit.n_iter}",
        ),
    )
    collector.raise_if_any(f"bgam.fitd controller replay {zero_weights=} {start}")


@pytest.mark.parametrize(("zero_weights", "start"), _EXTREME_START_CASES)
def test_extreme_start_working_model_matches_pinned_bam(r_bridge, zero_weights, start):
    """JAX's own controller run from bgam.fitd's in.out extreme start."""
    frame, weights, offset = _extreme_start_inputs(zero_weights)
    expected = r_bridge.fit_bam_discrete_gaussian(
        _COUPLED,
        frame,
        initial_sp=np.asarray(start),
        initial_scale=float(np.exp(-3.0)),
        scale=-1.0,
        m=30,
        weights=weights,
        offset=offset,
    )
    setup = build_continuous_discrete_setup(
        parse_formula(_COUPLED),
        DataFrameRowSource(frame, response="y", weights=weights, offset=offset),
        m=30,
    )
    try:
        snapped = {
            name: values[group.index.selector]
            for group in setup.compact.groups
            for name, values in zip(group.names, group.index.values, strict=True)
        }
        X = setup.predict_spec.build_predict_matrix(pd.DataFrame(snapped))
        response = frame.y.to_numpy() - offset
        selected = working.fit_gaussian_working_model_statistics(
            X.T @ (weights[:, None] * X),
            X.T @ (weights * response),
            float(np.sum(weights * response**2)),
            float(np.var(frame.y.to_numpy(), ddof=1)),
            setup.penalties,
            len(frame),
            determinant_penalties=setup.source_determinant_penalties,
            control=_control(),
            available_workspace_bytes=512 * 1024 * 1024,
            initial_parameters=np.r_[np.log(start), -3.0],
        )
    finally:
        setup.cleanup()
    fitted = X @ selected.coefficients + offset
    deviance = float(np.sum(weights * (frame.y.to_numpy() - fitted) ** 2))
    collector = _AssertCollector()
    collector.check(
        "iterations and convergence",
        lambda: check_that(
            selected.converged
            and expected.mgcv_converged
            and selected.n_iter == expected.n_iter,
            f"{selected.converged=} {selected.n_iter=} vs pinned {expected.n_iter}",
        ),
    )
    # The final sp vector is flat in its large sp: pinned R itself moves it
    # up to 12.1x MODERATE (0.12x LOOSE) for a one-ULP in.out start, with
    # fitted values within 5.3e-11, and 1.8e4-5.6e5x MODERATE for epsilon
    # 1e-9. LOOSE is authorized for this vector only (decision 2026-10-11).
    # JaxGAM's own setup rounding drives the 1e-11 physical differences;
    # the four documented passes in tests/test_fitting/test_working_model.py
    # covered these endpoint runs, so coefficients/fitted/scale are MODERATE.
    for name, observed, target, tolerance in (
        (
            "smoothing parameters",
            np.exp(selected.log_smoothing),
            expected.smoothing_params,
            LOOSE,
        ),
        ("coefficients", selected.coefficients, expected.coefficients, MODERATE),
        ("fitted values", fitted, expected.fitted_values, MODERATE),
        ("deviance", deviance, expected.deviance, STRICT),
        ("scale", selected.scale, expected.scale, MODERATE),
    ):
        collector.check(
            name,
            lambda observed=observed, target=target, tolerance=tolerance: (
                np.testing.assert_allclose(
                    observed, target, rtol=tolerance.rtol, atol=tolerance.atol
                )
            ),
        )
    collector.raise_if_any(f"extreme start {zero_weights=} {start}")


def test_working_model_in_out_start_validates(statistics):
    with pytest.raises(ValueError, match="log\\(sp\\) and log\\(scale\\)"):
        _fit_statistics(statistics, initial_parameters=np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        _fit_statistics(statistics, initial_parameters=np.array([np.nan, 0.0]))


def test_advance_rejects_nonfinite_deviance_before_any_proposal():
    state = working.WorkingModelState(
        1, np.zeros(2), np.zeros(2), np.zeros(3), 0.0, 0.0
    )

    def propose(*_args):
        raise AssertionError("no proposal may follow a non-finite deviance")

    with pytest.raises(FloatingPointError, match="penalized deviance"):
        working.advance_gaussian_working_model(
            state, propose, np.ones(3), np.inf, 10, 1, 1e-7
        )
