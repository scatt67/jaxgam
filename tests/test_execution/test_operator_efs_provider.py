"""Matched observed-score and Fisher-statistic operator EFS components."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln, xlogy

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator_efs_controller import OperatorEFSState
from jaxgam.execution.operator_efs_provider import MatchedOperatorEFSProvider
from jaxgam.execution.operator_fit import (
    FrozenOperatorWorkingState,
    fit_operator_fixed_sp,
)
from jaxgam.execution.operator_logdet import OperatorLogDetControl
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Gamma, Gaussian, Poisson
from jaxgam.fitting.efs_operator_score import operator_reml_score
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from tests.tolerances import STRICT


def _case(family, *, batch_rows=7):
    row = np.arange(27)
    frame = pd.DataFrame(
        {
            "y": (
                1.2 + 0.2 * np.sin(row)
                if isinstance(family, Gaussian)
                else np.asarray(row % 4, dtype=float)
            ),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    weights = np.resize([0.0, 1.0, 2.0, 0.5], len(row))
    offsets = 0.05 * np.sin(row)
    source = DataFrameRowSource(frame, response="y", weights=weights, offset=offsets)
    spec = parse_formula("y~0+s(g,bs='re')")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(
            batch_rows=batch_rows, memory_budget_bytes=2 * 1024 * 1024
        ),
    )
    beta = np.array([0.1, 0.3, -0.2], dtype=np.float64)
    state = FrozenOperatorWorkingState(
        model,
        family,
        beta,
        row_memory_budget_bytes=2 * 1024 * 1024,
        state_memory_budget_bytes=2 * 1024 * 1024,
    )
    return frame, weights, offsets, source, spec, state


@pytest.mark.parametrize(("family", "phi"), [(Gaussian(), 1.3), (Poisson(), 1.0)])
def test_matched_components_agree_with_dense_fixed_state_strict(family, phi):
    frame, weights, offsets, source, spec, state = _case(family)
    rho = np.array([np.log(1.3)], dtype=np.float64)
    result = MatchedOperatorEFSProvider(
        state,
        source,
        rho,
        score_phi=phi,
        memory_budget_bytes=16 * 1024 * 1024,
        logdet_control=OperatorLogDetControl(probes=4, lanczos_depth=3, seed=17),
    ).evaluate()
    dense = ModelSetup.build(spec, frame)
    X = dense.X
    eta = X @ state.beta + offsets
    mu = np.asarray(family.link.inverse(eta))
    working = weights if isinstance(family, Gaussian) else weights * mu
    S = dense.penalties.materialize(rho)
    H = X.T @ (working[:, None] * X) + S
    inverse = np.linalg.inv(H)
    S_unscaled = S / np.exp(rho[0])
    expected_trace = np.trace(inverse @ S_unscaled)
    expected_quadratic = state.beta @ S_unscaled @ state.beta
    expected_deviance = float(
        np.sum(family.deviance_contributions(frame.y.to_numpy(), mu, weights))
    )
    expected_saturated = float(
        np.asarray(family.saturated_loglik(frame.y.to_numpy(), weights, phi))
    )
    expected_score = (
        (expected_deviance + np.exp(rho[0]) * expected_quadratic) / (2 * phi)
        - expected_saturated
        + (np.linalg.slogdet(H)[1] - result.penalty.value) / 2
    )
    np.testing.assert_allclose(
        [result.score.value, result.fisher.traces[0], result.fisher.quadratics[0]],
        [expected_score, expected_trace, expected_quadratic],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.edf_total,
        state.model.n_coef - np.exp(rho[0]) * expected_trace,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_array_equal(result.penalty.derivative, [3.0])
    assert result.trace_method == "exact"
    assert result.source_scans > 1
    assert result.peak_bound_bytes < 16 * 1024 * 1024


def test_fixed_theta_nb_large_p_observed_score_and_fisher_trace_match_dense():
    p, n, theta = 300, 600, 2.7
    rng = np.random.default_rng(419)
    group = np.arange(n) % p
    mean = np.exp(0.2 + 0.3 * np.sin(group / 31.0))
    frame = pd.DataFrame(
        {
            "y": rng.negative_binomial(theta, theta / (theta + mean)),
            "g": pd.Categorical(
                [f"g{index}" for index in group],
                categories=[f"g{index}" for index in range(p)],
            ),
        }
    )
    spec = parse_formula("y~0+s(g,bs='re')")
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(
            batch_rows=64, memory_budget_bytes=4 * 1024 * 1024
        ),
    )
    family = NegativeBinomial(theta=theta, fixed=True)
    rho = np.log(np.array([0.8])) + 2.5
    fit = fit_operator_fixed_sp(
        model, source, family, np.exp(rho), memory_budget_bytes=16 * 1024 * 1024
    )
    assert fit.converged
    result = MatchedOperatorEFSProvider.from_fit(
        fit,
        source,
        score_phi=1.0,
        memory_budget_bytes=32 * 1024 * 1024,
        max_y=32,
    ).evaluate()
    dense = ModelSetup.build(spec, frame)
    X = dense.X
    beta = fit.coefficients
    y = frame.y.to_numpy(dtype=float)
    mu = np.exp(X @ beta)
    deviance = float(np.sum(family.deviance_contributions(y, mu, np.ones(n))))
    saturated = float(
        np.sum(
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta / (theta + y))
            + xlogy(y, y / (theta + y))
        )
    )
    penalty = dense.penalties.materialize(rho)
    observed = theta * mu * (theta + y) / (theta + mu) ** 2
    fisher = theta * mu / (theta + mu)
    H_observed = X.T @ (observed[:, None] * X) + penalty
    H_fisher = X.T @ (fisher[:, None] * X) + penalty
    logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(H_observed))))
    penalty_logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(penalty))))
    exact = (
        (deviance + beta @ penalty @ beta) / 2
        - saturated
        + (logdet - penalty_logdet) / 2
    )
    expected_trace = np.sum(1 / np.diag(H_fisher))
    np.testing.assert_allclose(
        result.score.value, exact, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.fisher.traces[0], expected_trace, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert result.score.strategy == "structured_diagonal"
    assert result.fisher.diagnostics.strategy == "structured_diagonal"


def test_stochastic_trace_is_labeled_and_keeps_sampling_error_separate():
    _, _, _, source, _, state = _case(Poisson())
    result = MatchedOperatorEFSProvider(
        state,
        source,
        np.zeros(1, dtype=np.float64),
        score_phi=1.0,
        memory_budget_bytes=16 * 1024 * 1024,
        trace_method="stochastic",
        trace_probes=4,
        trace_seed=31,
        logdet_control=OperatorLogDetControl(probes=4, lanczos_depth=3, seed=13),
    ).evaluate()
    assert result.trace_method == "stochastic"
    assert result.fisher.per_probe.shape == (4, 1)
    assert result.fisher.monte_carlo_standard_error.shape == (1,)
    assert (
        result.fisher.diagnostics.normal_actions
        == result.fisher.diagnostics.source_scans
    )


def test_noncanonical_gamma_keeps_observed_score_and_fisher_trace_distinct():
    row = np.arange(27)
    frame = pd.DataFrame(
        {
            "y": 1.2 + 0.2 * np.sin(row),
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
    family = Gamma(link="log")
    beta = np.array([0.1, 0.3, -0.2], dtype=np.float64)
    rho = np.array([np.log(1.3)], dtype=np.float64)
    phi = 1.7
    state = FrozenOperatorWorkingState(
        model, family, beta, 2 * 1024 * 1024, 2 * 1024 * 1024
    )
    result = MatchedOperatorEFSProvider(
        state,
        source,
        rho,
        score_phi=phi,
        memory_budget_bytes=16 * 1024 * 1024,
        logdet_control=OperatorLogDetControl(probes=4, lanczos_depth=3),
    ).evaluate()
    dense = ModelSetup.build(spec, frame)
    X = dense.X
    mu = np.exp(X @ beta)
    S = dense.penalties.materialize(rho)
    observed = X.T @ ((weights * frame.y.to_numpy() / mu)[:, None] * X) + S
    fisher = X.T @ (weights[:, None] * X) + S
    S_unscaled = S / np.exp(rho[0])
    expected_trace = np.trace(np.linalg.inv(fisher) @ S_unscaled)
    expected_score = (
        result.score_rows.terms.penalized_deviance / (2 * phi)
        - result.score_rows.terms.saturated_loglik
        + (np.linalg.slogdet(observed)[1] - result.penalty.value) / 2
    )
    np.testing.assert_allclose(
        [result.score.value, result.fisher.traces[0]],
        [expected_score, expected_trace],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert not np.isclose(np.linalg.slogdet(observed)[1], np.linalg.slogdet(fisher)[1])


def test_matched_provider_preflights_rho_and_total_budget_before_replay(monkeypatch):
    _, _, _, source, _, state = _case(Poisson())
    monkeypatch.setattr(source, "scan", lambda *_: pytest.fail("source replay started"))
    with pytest.raises(ValueError, match="rho"):
        MatchedOperatorEFSProvider(
            state,
            source,
            np.zeros(1, dtype=np.float32),
            score_phi=1.0,
            memory_budget_bytes=16 * 1024 * 1024,
        )
    with pytest.raises(MemoryError, match="total budget"):
        MatchedOperatorEFSProvider(
            state,
            source,
            np.zeros(1, dtype=np.float64),
            score_phi=1.0,
            memory_budget_bytes=1,
        )


def test_from_fit_binds_rho_family_and_convergence_to_matched_components():
    _, _, _, source, _, state = _case(Gaussian())
    sp = np.array([1.3], dtype=np.float64)
    fitted = fit_operator_fixed_sp(
        state.model,
        source,
        Gaussian(),
        sp,
        memory_budget_bytes=8 * 1024 * 1024,
    )
    assert fitted.converged
    np.testing.assert_array_equal(fitted.log_lambda, np.log(sp))
    assert not fitted.log_lambda.flags.writeable
    provider = MatchedOperatorEFSProvider.from_fit(
        fitted,
        source,
        score_phi=1.3,
        memory_budget_bytes=16 * 1024 * 1024,
        logdet_control=OperatorLogDetControl(probes=4, lanczos_depth=3),
    )
    result = provider.evaluate()
    assert np.isfinite(result.score.value)
    assert result.source_scans > 1
    assert OperatorEFSState(fitted, result, None).components is result
    shifted_beta = fitted.coefficients.copy()
    shifted_beta[0] += 0.01
    with pytest.raises(ValueError, match="do not match"):
        OperatorEFSState(replace(fitted, coefficients=shifted_beta), result, None)
    with pytest.raises(ValueError, match="do not match"):
        OperatorEFSState(
            replace(fitted, log_lambda=np.log(np.array([1.4]))), result, None
        )
    with pytest.raises(ValueError, match="converged finite fit"):
        MatchedOperatorEFSProvider.from_fit(
            replace(fitted, converged=False),
            source,
            score_phi=1.3,
            memory_budget_bytes=16 * 1024 * 1024,
        )


def test_large_single_index_fit_uses_exact_diagonal_score_and_fisher_trace():
    row = np.arange(51)
    frame = pd.DataFrame(
        {
            "y": 1.0 + row / 51,
            "g": pd.Categorical(
                [f"g{index}" for index in row % 3],
                categories=[f"g{index}" for index in range(300)],
            ),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~0+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(
            batch_rows=11, memory_budget_bytes=2 * 1024 * 1024
        ),
    )
    fitted = fit_operator_fixed_sp(
        model,
        source,
        Gaussian(),
        np.array([0.8], dtype=np.float64),
        memory_budget_bytes=8 * 1024 * 1024,
    )
    assert fitted.converged
    assert model.n_coef == 300
    result = MatchedOperatorEFSProvider.from_fit(
        fitted,
        source,
        score_phi=1.3,
        memory_budget_bytes=16 * 1024 * 1024,
        logdet_control=OperatorLogDetControl(probes=4, lanczos_depth=4),
    ).evaluate()
    scale = model.penalties.blocks[0].local_penalties[0].scale
    diagonal = np.full(model.n_coef, 0.8 * scale)
    diagonal[:3] += 17.0
    expected_logdet = float(np.sum(np.log(diagonal)))
    np.testing.assert_allclose(
        result.fisher.traces,
        [scale * np.sum(1.0 / diagonal)],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert result.score.strategy == "structured_diagonal"
    assert result.fisher.diagnostics.strategy == "structured_diagonal"
    assert result.fisher.diagnostics.normal_actions == 0
    assert result.score.source_scans == 0
    assert result.score.determinant_rounding_bound < 1e-10
    expected_score = operator_reml_score(
        result.score_rows.terms.penalized_deviance,
        result.score_rows.terms.saturated_loglik,
        1.3,
        expected_logdet,
        result.penalty.value,
        result.penalty.null_dimension,
    )
    np.testing.assert_allclose(
        result.score.value, expected_score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert result.peak_bound_bytes < 16 * 1024 * 1024
