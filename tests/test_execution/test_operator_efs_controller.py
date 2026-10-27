"""Accepted/trial and adaptive score gates for internal operator EFS."""

import weakref
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jaxgam.control import EFSControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator_efs_controller import (
    SourceOperatorEFSFitProvider,
    _score_window_certified,
    run_operator_efs_gaussian_scale,
    run_operator_efs_known_scale,
)
from jaxgam.execution.operator_logdet import OperatorLogDetControl
from jaxgam.execution.operator_score import OperatorScoreEstimate
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.fitting.efs_operator_score import operator_reml_score
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula


class _ControlledProvider:
    def __init__(self, *, direction: float, bound: float | None, ratio: float = 2.0):
        self.direction = direction
        self.bound = bound
        self.ratio = ratio
        self.fit_calls = []
        self.evaluations = []

    def reserve(self, held, *, operation, pending_fit=None, controller_bytes=0):
        assert operation in {"fit", "evaluate"}
        assert isinstance(held, tuple)
        assert (pending_fit is None) == (operation == "fit")
        assert controller_bytes >= 8192

    def fit(self, rho, beta_start):
        assert rho.dtype == np.float64
        self.fit_calls.append(
            (rho.copy(), None if beta_start is None else beta_start.copy())
        )
        return SimpleNamespace(
            log_lambda=rho.copy(),
            coefficients=np.array([float(rho[0])]),
            deviance=float(1 + rho[0] ** 2),
            converged=True,
        )

    def evaluate(self, fit, control):
        self.evaluations.append(
            (fit.log_lambda.copy(), control.probes, control.lanczos_depth)
        )
        rho = float(fit.log_lambda[0])
        value = (
            self.direction(rho) if callable(self.direction) else self.direction * rho
        )
        score = OperatorScoreEstimate(
            value=value,
            per_probe=np.full(control.probes, value),
            monte_carlo_standard_error=0.0,
            maximum_depth_difference=0.0,
            source_scans=1,
            seed=control.seed,
            lanczos_depth=control.lanczos_depth,
            n_coef=1,
            probe_fingerprint="a" * 64 if control.probes == 2 else "b" * 64,
        )
        components = SimpleNamespace(
            score=score,
            penalty=SimpleNamespace(
                derivative=np.array([np.exp(fit.log_lambda[0]) * self.ratio])
            ),
            fisher=SimpleNamespace(traces=np.array([0.0]), quadratics=np.array([1.0])),
        )
        return SimpleNamespace(
            fit=fit,
            components=components,
            score_error_bound=self.bound,
            score_phi=1.0,
            update_phi=1.0,
            reported_phi=1.0,
            carried_phi=1.0,
        )


def _run(provider, *, outer_limit=1, maximum_probes=2, maximum_depth=2):
    return run_operator_efs_known_scale(
        provider,
        np.array([0.0], dtype=np.float64),
        control=EFSControl(outer_limit=outer_limit),
        initial_accuracy=OperatorLogDetControl(probes=2, lanczos_depth=2),
        maximum_probes=maximum_probes,
        maximum_depth=maximum_depth,
    )


def test_gaussian_scale_controller_carries_incoming_phi_across_trial():
    n = 45
    group = np.arange(n) % 3
    frame = pd.DataFrame(
        {
            "y": 1.2 + 0.4 * np.sin(np.arange(n) / 6),
            "g": pd.Categorical(group),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~0+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(
            batch_rows=7, memory_budget_bytes=2 * 1024 * 1024
        ),
    )

    class _RecordingProvider(SourceOperatorEFSFitProvider):
        def __init__(self):
            super().__init__(
                model,
                source,
                Gaussian(),
                fit_control=OperatorControl(),
                fit_budget_bytes=8 * 1024 * 1024,
                components_budget_bytes=16 * 1024 * 1024,
                aggregate_budget_bytes=64 * 1024 * 1024,
            )
            self.phi_inputs = []

        def evaluate(self, fit, control, *, score_phi=1.0):
            self.phi_inputs.append(score_phi)
            return super().evaluate(fit, control, score_phi=score_phi)

    provider = _RecordingProvider()
    result = run_operator_efs_gaussian_scale(
        provider,
        np.log(np.array([0.8])),
        control=EFSControl(outer_limit=3),
        initial_accuracy=OperatorLogDetControl(probes=2, lanczos_depth=2),
        maximum_probes=2,
        maximum_depth=2,
    )
    assert result.stop_reason == "indeterminate_score"
    assert result.attempted_fits == 2
    initial = result.accepted
    np.testing.assert_allclose(
        initial.score_phi,
        initial.fit.diagnostics.initial_null_scale / 10,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        provider.phi_inputs, [initial.score_phi, initial.update_phi]
    )


def test_gaussian_extension_refinement_preserves_candidate_carried_phi():
    """Refining an accepted extension cannot replace its inherited scale."""
    ratio = 1.02
    candidate_rho = 2.5 + np.log(ratio)
    extension_rho = 2.5 + 2 * np.log(ratio)

    @dataclass(frozen=True)
    class _ScaleState:
        fit: object
        components: object
        score_error_bound: float
        score_phi: float
        update_phi: float
        reported_phi: float
        carried_phi: float

    class _ScaleProvider(_ControlledProvider):
        family = Gaussian()

        def __init__(self):
            super().__init__(direction=0.0, bound=0.0, ratio=ratio)
            self.phi_inputs = []

        def evaluate(self, fit, control, *, score_phi=1.0):
            rho = float(fit.log_lambda[0])
            self.phi_inputs.append((rho, score_phi, control.probes))
            score_value = (rho - extension_rho) ** 2
            estimate = OperatorScoreEstimate(
                value=score_value,
                per_probe=np.full(control.probes, score_value),
                monte_carlo_standard_error=0.0,
                maximum_depth_difference=0.0,
                source_scans=0,
                seed=control.seed,
                lanczos_depth=control.lanczos_depth,
                n_coef=1,
                probe_fingerprint="a" * 64 if control.probes == 2 else "b" * 64,
            )
            components = SimpleNamespace(
                score=estimate,
                penalty=SimpleNamespace(derivative=np.array([np.exp(rho) * ratio])),
                fisher=SimpleNamespace(
                    traces=np.array([0.0]), quadratics=np.array([1.0])
                ),
            )
            update_phi = (
                2.0
                if np.isclose(rho, candidate_rho)
                else 3.0
                if np.isclose(rho, extension_rho)
                else 1.0
            )
            uncertain = len(self.fit_calls) >= 4 and control.probes == 2
            return _ScaleState(
                fit,
                components,
                10.0 if uncertain else 0.0,
                score_phi,
                update_phi,
                update_phi,
                update_phi,
            )

    provider = _ScaleProvider()
    result = run_operator_efs_gaussian_scale(
        provider,
        np.array([0.0], dtype=np.float64),
        control=EFSControl(outer_limit=2),
        initial_accuracy=OperatorLogDetControl(probes=2, lanczos_depth=2),
        initial_score_phi=0.5,
        maximum_probes=4,
        maximum_depth=4,
    )
    assert result.accuracy_refinements >= 1
    assert result.attempted_fits == 5
    assert result.multiplier == 1.0
    np.testing.assert_allclose(provider.phi_inputs[3][1], 2.0)
    np.testing.assert_allclose(provider.phi_inputs[-1][1], 2.0)
    np.testing.assert_allclose(result.accepted.score_phi, 2.0)


def test_operator_efs_applies_initial_shift_and_accepts_proven_improvement():
    provider = _ControlledProvider(direction=-1.0, bound=0.0)
    result = _run(provider)
    np.testing.assert_allclose(provider.fit_calls[0][0], [2.5])
    assert provider.fit_calls[0][1] is None
    np.testing.assert_allclose(provider.fit_calls[1][1], [2.5])
    np.testing.assert_allclose(result.accepted.fit.log_lambda, [2.5 + np.log(2.0)])
    assert result.attempted_fits == 2
    assert result.stop_reason == "iteration_limit"


def test_operator_efs_refines_same_fits_then_retains_accepted_on_indeterminate():
    provider = _ControlledProvider(direction=-1.0, bound=None)
    result = _run(provider, maximum_probes=4, maximum_depth=4)
    assert result.stop_reason == "indeterminate_score"
    assert not result.converged
    assert result.attempted_fits == 2
    assert result.accuracy_refinements == 1
    assert result.indeterminate_comparisons == 1
    np.testing.assert_allclose(result.accepted.fit.log_lambda, [2.5])
    assert [entry[1:] for entry in provider.evaluations] == [
        (2, 2),
        (2, 2),
        (4, 4),
        (4, 4),
    ]


def test_refinement_reserves_caller_and_new_large_vector_generations():
    vector_bytes = 8 * 2048

    class _BoundedProvider(_ControlledProvider):
        max_held = 0

        def fit(self, rho, beta_start):
            fit = super().fit(rho, beta_start)
            fit.coefficients = np.zeros(2048, dtype=np.float64)
            return fit

        def reserve(self, held, *, operation, pending_fit=None, controller_bytes=0):
            super().reserve(
                held,
                operation=operation,
                pending_fit=pending_fit,
                controller_bytes=controller_bytes,
            )
            unique = {id(state) for state in held}
            self.max_held = max(self.max_held, len(unique))
            if (len(unique) + 1) * vector_bytes > 70 * 1024:
                raise MemoryError("Combined retained vectors exceed the budget.")

    provider = _BoundedProvider(direction=-1.0, bound=None)
    with pytest.raises(MemoryError, match="retained vectors"):
        _run(provider, maximum_probes=8, maximum_depth=8)
    assert provider.max_held == 4


def test_replaced_initial_fit_is_released_before_later_refits():
    class _WeakFit(SimpleNamespace):
        pass

    class _LifetimeProvider(_ControlledProvider):
        first_reference = None

        def fit(self, rho, beta_start):
            fit = _WeakFit(**vars(super().fit(rho, beta_start)))
            if len(self.fit_calls) == 1:
                self.first_reference = weakref.ref(fit)
            elif len(self.fit_calls) == 3:
                assert self.first_reference() is None
            return fit

    provider = _LifetimeProvider(direction=-1.0, bound=0.0)
    result = _run(provider, outer_limit=2)
    assert result.attempted_fits == 3


def test_history_ownership_is_reserved_before_first_fit():
    class _HistoryBudgetProvider(_ControlledProvider):
        def reserve(self, held, *, operation, pending_fit=None, controller_bytes=0):
            super().reserve(
                held,
                operation=operation,
                pending_fit=pending_fit,
                controller_bytes=controller_bytes,
            )
            if controller_bytes > 1024 * 1024:
                raise MemoryError("Controller history exceeds shared budget.")

    provider = _HistoryBudgetProvider(direction=-1.0, bound=0.0)
    with pytest.raises(MemoryError, match="Controller history"):
        run_operator_efs_known_scale(
            provider,
            np.array([0.0], dtype=np.float64),
            control=EFSControl(outer_limit=5000, history_limit=5000),
            initial_accuracy=OperatorLogDetControl(probes=2, lanczos_depth=2),
        )
    assert not provider.fit_calls


def test_operator_efs_rejects_empty_smoothing_before_provider_work():
    provider = _ControlledProvider(direction=-1.0, bound=0.0)
    with pytest.raises(ValueError, match="rho"):
        run_operator_efs_known_scale(
            provider,
            np.empty(0, dtype=np.float64),
            control=EFSControl(),
            initial_accuracy=OperatorLogDetControl(probes=2, lanczos_depth=2),
        )
    assert not provider.fit_calls


def test_operator_efs_keeps_pinned_minimum_multiplier_worsening_policy():
    provider = _ControlledProvider(direction=1.0, bound=0.0)
    result = _run(provider)
    np.testing.assert_allclose(result.accepted.fit.log_lambda, [2.5 + np.log(2.0)])
    assert result.multiplier == 1.0


def test_operator_efs_retains_accepted_state_after_failed_trial_fit():
    class _FailingProvider(_ControlledProvider):
        def fit(self, rho, beta_start):
            fit = super().fit(rho, beta_start)
            if beta_start is not None:
                fit.converged = False
            return fit

    provider = _FailingProvider(direction=-1.0, bound=0.0)
    result = _run(provider)
    assert result.stop_reason == "inner_failure"
    assert result.attempted_fits == 2
    np.testing.assert_allclose(result.accepted.fit.log_lambda, [2.5])


def test_operator_efs_extends_proven_small_step_from_same_old_fit():
    provider = _ControlledProvider(direction=-1.0, bound=0.0, ratio=1.02)
    result = _run(provider)
    assert result.attempted_fits == 3
    np.testing.assert_allclose(result.accepted.fit.log_lambda, [2.5 + 2 * np.log(1.02)])
    np.testing.assert_allclose(provider.fit_calls[1][1], provider.fit_calls[2][1])
    assert result.multiplier == 2.0


def test_operator_efs_accepts_proven_equal_trial_but_not_equal_extension():
    provider = _ControlledProvider(direction=0.0, bound=0.0, ratio=1.02)
    result = _run(provider)
    assert result.attempted_fits == 3
    np.testing.assert_allclose(result.accepted.fit.log_lambda, [2.5 + np.log(1.02)])
    assert result.multiplier == 1.0


def test_score_window_requires_uncertainty_to_fit_inside_tolerance():
    scores = (1.0, 1.0004, 1.0008, 1.0012)
    assert _score_window_certified(scores, (0.0,) * 4, 0.001)
    assert not _score_window_certified(scores, (0.0004,) * 4, 0.001)
    assert not _score_window_certified(scores, (None,) * 4, 0.001)


@pytest.mark.parametrize("wrong_call", [3, 5])
def test_operator_efs_rejects_wrong_rho_on_extension_and_contraction(wrong_call):
    optimum = 2.5 + 2 * np.log(1.02)

    class _WrongRhoProvider(_ControlledProvider):
        def fit(self, rho, beta_start):
            fit = super().fit(rho, beta_start)
            if len(self.fit_calls) == wrong_call:
                fit.log_lambda[0] += 0.01
            return fit

    provider = _WrongRhoProvider(
        direction=lambda rho: (rho - optimum) ** 2,
        bound=0.0,
        ratio=1.02,
    )
    with pytest.raises(ValueError, match="mismatched rho"):
        _run(provider, outer_limit=2)


def test_operator_efs_contracts_worse_extended_trial_then_accepts_minimum_step():
    optimum = 2.5 + 2 * np.log(1.02)
    provider = _ControlledProvider(
        direction=lambda rho: (rho - optimum) ** 2,
        bound=0.0,
        ratio=1.02,
    )
    result = _run(provider, outer_limit=2)
    assert result.attempted_fits == 5
    assert result.multiplier == 1.0
    np.testing.assert_allclose(result.accepted.fit.log_lambda, [optimum + np.log(1.02)])
    np.testing.assert_allclose(provider.fit_calls[3][1], provider.fit_calls[4][1])


class _ScriptedBoundProvider(_ControlledProvider):
    """Controlled states with a per-rho bound and a chosen score strategy."""

    def __init__(self, *, bound_at=None, strategy="stochastic_lanczos", **kwargs):
        super().__init__(**kwargs)
        self.bound_at = bound_at
        self.strategy = strategy

    def evaluate(self, fit, control):
        state = super().evaluate(fit, control)
        state.components.score = replace(state.components.score, strategy=self.strategy)
        if self.bound_at is not None:
            state.score_error_bound = self.bound_at(float(fit.log_lambda[0]))
        return state


def test_undecided_contraction_to_unit_multiplier_accepts_minimum_step():
    optimum = 2.5 + 2 * np.log(1.02)
    minimum_step = optimum + np.log(1.02)
    provider = _ScriptedBoundProvider(
        direction=lambda rho: (rho - optimum) ** 2,
        bound=0.0,
        ratio=1.02,
        bound_at=lambda rho: 1.0 if np.isclose(rho, minimum_step) else 0.0,
    )
    result = _run(provider, outer_limit=2)
    assert result.stop_reason == "iteration_limit"
    assert result.indeterminate_comparisons == 1
    assert result.attempted_fits == 5
    assert result.multiplier == 1.0
    np.testing.assert_allclose(result.accepted.fit.log_lambda, [minimum_step])


def test_undecided_contraction_above_unit_multiplier_still_stops():
    step = np.log(1.02)
    optimum = 2.5 + 7 * step
    # Two accepted extensions reach multiplier 4; the worse 2.5 + 10 step
    # trial contracts to multiplier 2, where efsudr's action depends on sign.
    provider = _ScriptedBoundProvider(
        direction=lambda rho: (rho - optimum) ** 2,
        bound=0.0,
        ratio=1.02,
        bound_at=lambda rho: 1.0 if np.isclose(rho, 2.5 + 8 * step) else 0.0,
    )
    result = _run(provider, outer_limit=3)
    assert result.stop_reason == "indeterminate_score"
    assert not result.converged
    assert result.attempted_fits == 7
    assert result.indeterminate_comparisons == 1
    np.testing.assert_allclose(result.accepted.fit.log_lambda, [2.5 + 6 * step])


def test_structured_scores_are_not_refined_when_undecided():
    provider = _ScriptedBoundProvider(
        direction=-1.0, bound=None, strategy="structured_diagonal"
    )
    result = _run(provider, maximum_probes=4, maximum_depth=4)
    assert result.stop_reason == "indeterminate_score"
    assert result.accuracy_refinements == 0
    assert len(provider.evaluations) == 2


def test_operator_efs_source_refit_matches_dense_score_bound_at_moderate_p():
    row = np.arange(27)
    frame = pd.DataFrame(
        {
            "y": np.asarray(row % 4, dtype=float),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    weights = np.resize([0.0, 1.0, 2.0, 0.5], len(row))
    source = DataFrameRowSource(frame, response="y", weights=weights)
    spec = parse_formula("y~0+s(g,bs='re')")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(
            batch_rows=7, memory_budget_bytes=2 * 1024 * 1024
        ),
    )
    dense = ModelSetup.build(spec, frame)
    oracle_calls = []

    def oracle_bound(fit, components):
        X = dense.X
        mu = np.exp(X @ fit.coefficients)
        H = X.T @ ((weights * mu)[:, None] * X)
        H += dense.penalties.materialize(fit.log_lambda)
        exact = float(
            operator_reml_score(
                components.score_rows.terms.penalized_deviance,
                components.score_rows.terms.saturated_loglik,
                1.0,
                np.linalg.slogdet(H)[1],
                components.penalty.value,
                components.penalty.null_dimension,
            )
        )
        bound = abs(exact - components.score.value) + 1e-10
        oracle_calls.append((fit.log_lambda.copy(), exact, bound))
        return bound

    provider = SourceOperatorEFSFitProvider(
        model,
        source,
        Poisson(),
        fit_control=OperatorControl(),
        fit_budget_bytes=8 * 1024 * 1024,
        components_budget_bytes=16 * 1024 * 1024,
        aggregate_budget_bytes=32 * 1024 * 1024,
        score_error_bound=oracle_bound,
    )
    result = run_operator_efs_known_scale(
        provider,
        np.zeros(1, dtype=np.float64),
        control=EFSControl(outer_limit=1),
        initial_accuracy=OperatorLogDetControl(probes=4, lanczos_depth=3),
        maximum_probes=4,
        maximum_depth=3,
    )
    assert result.attempted_fits >= 2
    assert len(oracle_calls) == result.attempted_fits
    assert np.isfinite(result.accepted.components.score.value)
    assert result.accepted.components.source_scans > 0
    assert result.accepted.fit.converged
    assert result.accepted.fit.log_lambda.shape == (1,)
    assert result.stop_reason in {"iteration_limit", "indeterminate_score"}
    provider.score_error_bound = None
    uncalibrated = run_operator_efs_known_scale(
        provider,
        np.zeros(1, dtype=np.float64),
        control=EFSControl(outer_limit=1),
        initial_accuracy=OperatorLogDetControl(probes=4, lanczos_depth=3),
        maximum_probes=4,
        maximum_depth=3,
    )
    assert uncalibrated.stop_reason == "indeterminate_score"
    np.testing.assert_allclose(uncalibrated.accepted.fit.log_lambda, [2.5])
    assert provider.aggregate_peak_bound_bytes < provider.aggregate_budget_bytes
    provider.aggregate_budget_bytes = provider.components_budget_bytes
    with pytest.raises(MemoryError, match="combined bytes"):
        provider.reserve(
            (result.accepted,),
            operation="evaluate",
            pending_fit=result.accepted.fit,
        )


def test_operator_efs_source_provider_requires_penalty():
    frame = pd.DataFrame({"y": [1.0, 2.0, 1.0, 3.0]})
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(parse_formula("y~1"), source)
    with pytest.raises(ValueError, match="at least one penalty"):
        SourceOperatorEFSFitProvider(
            model,
            source,
            Poisson(),
            fit_control=OperatorControl(),
            fit_budget_bytes=8 * 1024 * 1024,
            components_budget_bytes=8 * 1024 * 1024,
            aggregate_budget_bytes=16 * 1024 * 1024,
        )
