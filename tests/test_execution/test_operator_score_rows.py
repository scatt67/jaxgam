"""Bounded family-owned source reductions for matched operator scores."""

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator import FrozenScanSystem
from jaxgam.execution.operator_fit import FrozenOperatorWorkingState
from jaxgam.execution.operator_logdet import FrozenOperatorLogDet, OperatorLogDetControl
from jaxgam.execution.operator_penalty_determinant import (
    LocalOperatorPenaltyDeterminant,
)
from jaxgam.execution.operator_score import assemble_operator_score
from jaxgam.execution.operator_score_rows import MatchedOperatorScoreReducer
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
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
    source = DataFrameRowSource(frame, response="y")
    spec = parse_formula("y~0+s(g,bs='re')")
    model = prepare_operator_model(
        spec,
        source,
        control=OperatorPreparationControl(
            batch_rows=batch_rows, memory_budget_bytes=2 * 1024 * 1024
        ),
    )
    beta = np.array(
        [2.0, 2.2, 1.8] if isinstance(family, Gamma) else [0.1, 0.3, -0.2],
        dtype=np.float64,
    )
    state = FrozenOperatorWorkingState(
        model,
        family,
        beta,
        row_memory_budget_bytes=2 * 1024 * 1024,
        state_memory_budget_bytes=2 * 1024 * 1024,
    )
    return frame, source, spec, state


@pytest.mark.parametrize(("family", "phi"), [(Gaussian(), 1.7), (Poisson(), 1.0)])
def test_matched_reductions_match_dense_family_terms_strict(family, phi):
    frame, source, spec, state = _case(family)
    rho = np.array([np.log(1.3)], dtype=np.float64)
    result = MatchedOperatorScoreReducer(
        state, source, rho, score_phi=phi, memory_budget_bytes=8 * 1024 * 1024
    ).reduce()
    dense = ModelSetup.build(spec, frame)
    beta = state.beta
    assert beta is not None
    eta = dense.X @ beta
    mu = np.asarray(family.link.inverse(eta))
    y = frame["y"].to_numpy()
    weight = np.ones(len(frame))
    deviance = float(np.sum(family.deviance_contributions(y, mu, weight)))
    penalty = float(beta @ dense.penalties.materialize(rho) @ beta)
    saturated = float(np.asarray(family.saturated_loglik(y, weight, phi)))
    np.testing.assert_allclose(
        [
            result.raw_deviance,
            result.penalty_quadratic,
            result.terms.penalized_deviance,
            result.terms.saturated_loglik,
        ],
        [deviance, penalty, deviance + penalty, saturated],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert result.source_scans == 1
    assert result.source_batches == 4
    assert result.peak_bound_bytes < 8 * 1024 * 1024


def test_reducer_preflight_rejects_budget_before_replaying(monkeypatch):
    _, source, _, state = _case(Gaussian())
    monkeypatch.setattr(source, "scan", lambda *_: pytest.fail("source replay started"))
    reducer = MatchedOperatorScoreReducer(
        state,
        source,
        np.zeros(1, dtype=np.float64),
        score_phi=1.0,
        memory_budget_bytes=1,
    )
    with pytest.raises(MemoryError, match="budget"):
        reducer.reduce()


@pytest.mark.parametrize("rho", [1000.0, -1000.0])
def test_reducer_rejects_unrepresentable_penalty_before_source_replay(monkeypatch, rho):
    _, source, _, state = _case(Poisson())
    monkeypatch.setattr(source, "scan", lambda *_: pytest.fail("source replay started"))
    with pytest.raises(ValueError, match="finite positive penalties"):
        MatchedOperatorScoreReducer(
            state,
            source,
            np.array([rho], dtype=np.float64),
            score_phi=1.0,
            memory_budget_bytes=8 * 1024 * 1024,
        )


def test_reducer_rejects_wrong_source_scale_and_rho():
    frame, source, _, state = _case(Poisson())
    with pytest.raises(ValueError, match="Known-scale"):
        MatchedOperatorScoreReducer(
            state,
            source,
            np.zeros(1),
            score_phi=1.1,
            memory_budget_bytes=8 * 1024 * 1024,
        )
    with pytest.raises(ValueError, match="rho"):
        MatchedOperatorScoreReducer(
            state,
            source,
            np.array([np.inf]),
            score_phi=1.0,
            memory_budget_bytes=8 * 1024 * 1024,
        )
    different_source = DataFrameRowSource(frame.assign(y=frame.y + 1), response="y")
    with pytest.raises(RuntimeError, match="source changed"):
        MatchedOperatorScoreReducer(
            state,
            different_source,
            np.zeros(1),
            score_phi=1.0,
            memory_budget_bytes=8 * 1024 * 1024,
        )


@pytest.mark.parametrize(
    ("family", "response", "score_phi"),
    [
        (Gaussian(), 0.8 + 0.2 * np.sin(np.arange(27)), 1.3),
        (Poisson(), np.arange(27) % 5, 1.0),
        (Binomial(), np.arange(27) % 2, 1.0),
        (Gamma(), 0.8 + 0.2 * np.sin(np.arange(27)), 1.3),
        (NegativeBinomial(theta=2.3, fixed=True), np.arange(27) % 5, 1.0),
    ],
)
def test_weighted_offset_reductions_are_batch_invariant_and_match_dense(
    family, response, score_phi
):
    frame = pd.DataFrame(
        {
            "y": np.asarray(response, dtype=np.float64),
            "g": pd.Categorical(np.resize(["a", "b", "c"], 27)),
        }
    )
    weights = np.resize([0.0, 1.0, 2.0, 0.5], 27)
    offsets = 0.05 * np.sin(np.arange(27))
    source = DataFrameRowSource(frame, response="y", weights=weights, offset=offsets)
    spec = parse_formula("y~0+s(g,bs='re')")
    beta = np.array(
        [2.0, 2.2, 1.8] if isinstance(family, Gamma) else [0.1, 0.3, -0.2],
        dtype=np.float64,
    )
    rho = np.array([np.log(1.3)], dtype=np.float64)
    results = []
    for batch_rows in (1, 7, 27):
        model = prepare_operator_model(
            spec,
            source,
            control=OperatorPreparationControl(
                batch_rows=batch_rows, memory_budget_bytes=2 * 1024 * 1024
            ),
        )
        state = FrozenOperatorWorkingState(
            model,
            family,
            beta,
            row_memory_budget_bytes=2 * 1024 * 1024,
            state_memory_budget_bytes=2 * 1024 * 1024,
        )
        results.append(
            MatchedOperatorScoreReducer(
                state,
                source,
                rho,
                score_phi=score_phi,
                memory_budget_bytes=8 * 1024 * 1024,
                max_y=4,
            ).reduce()
        )
    dense = ModelSetup.build(spec, frame)
    mu = np.asarray(family.link.inverse(dense.X @ beta + offsets))
    expected_deviance = float(
        np.sum(family.deviance_contributions(frame.y.to_numpy(), mu, weights))
    )
    expected_penalty = float(beta @ dense.penalties.materialize(rho) @ beta)
    expected_saturated = float(
        np.asarray(
            family.saturated_loglik(frame.y.to_numpy(), weights, score_phi, max_y=4)
        )
    )
    for result in results:
        np.testing.assert_allclose(
            [
                result.raw_deviance,
                result.penalty_quadratic,
                result.terms.penalized_deviance,
                result.terms.saturated_loglik,
            ],
            [
                expected_deviance,
                expected_penalty,
                expected_deviance + expected_penalty,
                expected_saturated,
            ],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        assert result.source_scans == 1


def test_reducer_rechecks_frozen_family_and_source_on_replay():
    frame, source, _, state = _case(Poisson())
    reducer = MatchedOperatorScoreReducer(
        state,
        source,
        np.zeros(1, dtype=np.float64),
        score_phi=1.0,
        memory_budget_bytes=8 * 1024 * 1024,
    )
    frame.loc[0, "y"] = 10.0
    with pytest.raises(RuntimeError, match="changed its operator replay"):
        reducer.reduce()


def test_observed_operator_score_matches_dense_at_one_frozen_poisson_state():
    row = np.arange(27)
    frame = pd.DataFrame(
        {
            "y": np.asarray(row % 4, dtype=np.float64),
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
            batch_rows=7, memory_budget_bytes=2 * 1024 * 1024
        ),
    )
    beta = np.array([0.1, 0.3, -0.2], dtype=np.float64)
    rho = np.array([np.log(1.3)], dtype=np.float64)
    family = Poisson()
    state = FrozenOperatorWorkingState(
        model,
        family,
        beta,
        row_memory_budget_bytes=2 * 1024 * 1024,
        state_memory_budget_bytes=2 * 1024 * 1024,
    )
    reductions = MatchedOperatorScoreReducer(
        state,
        source,
        rho,
        score_phi=1.0,
        memory_budget_bytes=8 * 1024 * 1024,
    ).reduce()
    observed = FrozenScanSystem(
        model,
        source,
        rho,
        curvature_from_rows=state.curvature,
        memory_budget_bytes=4 * 1024 * 1024,
    )
    logdet = FrozenOperatorLogDet(
        observed,
        memory_budget_bytes=8 * 1024 * 1024,
        control=OperatorLogDetControl(probes=4, lanczos_depth=3, seed=13),
    ).estimate()
    penalty = LocalOperatorPenaltyDeterminant(
        model.penalties, memory_budget_bytes=2 * 1024 * 1024
    ).evaluate(rho)
    score = assemble_operator_score(reductions.terms, logdet, penalty)

    dense = ModelSetup.build(spec, frame)
    X = dense.X
    mu = np.exp(X @ beta + offsets)
    H_observed = X.T @ ((weights * mu)[:, None] * X)
    H_observed += dense.penalties.materialize(rho)
    expected = (
        reductions.terms.penalized_deviance / 2
        - reductions.terms.saturated_loglik
        + (np.linalg.slogdet(H_observed)[1] - penalty.value) / 2
    )
    np.testing.assert_allclose(
        score.value, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert logdet.diagnostics.source_scans == logdet.diagnostics.completed_actions
