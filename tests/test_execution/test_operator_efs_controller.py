"""Accepted/trial and adaptive score gates for internal operator EFS."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from jaxgam.control import EFSControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator_efs_controller import (
    SourceOperatorEFSFitProvider,
    run_operator_efs_known_scale,
)
from jaxgam.execution.operator_logdet import OperatorLogDetControl
from jaxgam.execution.operator_score import OperatorScoreEstimate
from jaxgam.families.standard import Poisson
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
            fit=fit, components=components, score_error_bound=self.bound
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
