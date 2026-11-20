"""Finite matched score, common-probe comparison and indeterminate status."""

import math

import numpy as np
import pytest

from jaxgam.execution.operator.logdet import (
    OperatorLogDetDiagnostics,
    OperatorLogDetEstimate,
)
from jaxgam.execution.operator.penalty_determinant import OperatorPenaltyLogDet
from jaxgam.execution.operator.score import (
    OperatorScoreTerms,
    assemble_operator_score,
    compare_operator_scores,
)
from tests.tolerances import STRICT


def _logdet(values, *, seed=13, depth=4, depth_error=0.02, n_coef=4, fingerprint=None):
    values = np.asarray(values, dtype=np.float64)
    return OperatorLogDetEstimate(
        float(np.mean(values)),
        values,
        OperatorLogDetDiagnostics(
            len(values),
            depth,
            seed,
            8,
            8,
            0,
            float(np.std(values, ddof=1) / np.sqrt(len(values))),
            depth_error,
            4096,
            n_coef,
            fingerprint or f"{seed:032x}{n_coef:032x}",
        ),
    )


def _score(values, *, seed=13, depth=4, null_dimension=1, n_coef=4, fingerprint=None):
    terms = OperatorScoreTerms(16.0, -18.4, 1.7, n_coef)
    penalty = OperatorPenaltyLogDet(
        2.1, n_coef - null_dimension, null_dimension, np.array([2.1]), 1024
    )
    return assemble_operator_score(
        terms,
        _logdet(
            values,
            seed=seed,
            depth=depth,
            n_coef=n_coef,
            fingerprint=fingerprint,
        ),
        penalty,
    )


def test_matched_score_uses_observed_logdet_and_penalty_null_dimension():
    score = _score([4.0, 4.4, 4.8, 5.2])
    expected = 16.0 / (2 * 1.7) + 18.4 + (4.6 - 2.1) / 2
    expected -= math.log(2 * math.pi * 1.7) / 2
    np.testing.assert_allclose(
        score.value, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        score.per_probe - score.value,
        (np.array([4.0, 4.4, 4.8, 5.2]) - 4.6) / 2,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert score.monte_carlo_standard_error > 0
    assert score.maximum_depth_difference == 0.01
    assert score.source_scans == 8
    assert not score.per_probe.flags.writeable


def test_matched_score_rejects_mixed_coordinates_or_nonfinite_terms():
    with pytest.raises(ValueError, match="coordinates"):
        assemble_operator_score(
            OperatorScoreTerms(16.0, -18.4, 1.7, 4),
            _logdet([4.0, 5.0]),
            OperatorPenaltyLogDet(2.1, 2, 1, np.array([2.1]), 1024),
        )
    with pytest.raises(ValueError, match="score_phi"):
        OperatorScoreTerms(16.0, -18.4, 0.0, 4)
    with pytest.raises(ValueError, match="finite"):
        OperatorScoreTerms(float("nan"), -18.4, 1.7, 4)
    with pytest.raises(ValueError, match="n_coef"):
        OperatorScoreTerms(16.0, -18.4, 1.7, 0)
    with pytest.raises(ValueError, match="components"):
        assemble_operator_score(
            OperatorScoreTerms(16.0, -18.4, 1.7, 4),
            _logdet([float("nan"), 5.0]),
            OperatorPenaltyLogDet(2.1, 3, 1, np.array([2.1]), 1024),
        )


def test_common_probe_difference_reports_noise_but_waits_for_calibration():
    accepted = _score([4.0, 4.4, 4.8, 5.2])
    trial = _score([3.0, 3.8, 4.6, 5.4])
    unknown = compare_operator_scores(accepted, trial)
    assert unknown.status == "indeterminate"
    assert unknown.paired_standard_error > 0
    assert unknown.paired_standard_error < (
        accepted.monte_carlo_standard_error + trial.monte_carlo_standard_error
    )
    assert (
        compare_operator_scores(
            accepted, trial, accepted_error_bound=0.09, trial_error_bound=0.09
        ).status
        == "improvement"
    )
    assert (
        compare_operator_scores(
            accepted, trial, accepted_error_bound=0.3, trial_error_bound=0.3
        ).status
        == "indeterminate"
    )
    assert (
        compare_operator_scores(
            trial, accepted, accepted_error_bound=0.09, trial_error_bound=0.09
        ).status
        == "worsening"
    )
    assert (
        compare_operator_scores(
            accepted, accepted, accepted_error_bound=0.0, trial_error_bound=0.0
        ).status
        == "equal"
    )
    assert (
        compare_operator_scores(
            accepted, accepted, accepted_error_bound=0.01, trial_error_bound=0.0
        ).status
        == "indeterminate"
    )


def test_unpaired_probe_or_invalid_bound_rejects_before_decision():
    accepted = _score([4.0, 4.4, 4.8, 5.2])
    with pytest.raises(ValueError, match="identical probes"):
        compare_operator_scores(accepted, _score([3.0, 3.8, 4.6, 5.4], seed=14))
    with pytest.raises(ValueError, match="identical probes"):
        compare_operator_scores(accepted, _score([3.0, 3.8, 4.6, 5.4], depth=3))
    with pytest.raises(ValueError, match="identical probes"):
        compare_operator_scores(accepted, _score([3.0, 3.8, 4.6, 5.4], n_coef=5))
    with pytest.raises(ValueError, match="identical probes"):
        compare_operator_scores(
            accepted,
            _score([3.0, 3.8, 4.6, 5.4], fingerprint="f" * 64),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        compare_operator_scores(accepted, accepted, trial_error_bound=-1.0)


def test_direct_score_estimates_reject_nonfinite_or_inconsistent_samples():
    score = _score([4.0, 4.4, 4.8, 5.2])
    values = dict(vars(score))
    with pytest.raises(ValueError, match="finite"):
        type(score)(**{**values, "value": float("inf")})
    with pytest.raises(ValueError, match="finite"):
        type(score)(**{**values, "per_probe": np.array([4.0, float("nan")])})
    with pytest.raises(ValueError, match="probe mean"):
        type(score)(**{**values, "value": score.value + 1.0})
    with pytest.raises(ValueError, match="provenance"):
        type(score)(**{**values, "probe_fingerprint": "bad"})


def test_finite_inputs_with_overflowing_score_difference_cannot_decide():
    score = _score([4.0, 4.4, 4.8, 5.2])
    values = dict(vars(score))
    high = type(score)(**{**values, "value": 1e308, "per_probe": np.full(4, 1e308)})
    low = type(score)(**{**values, "value": -1e308, "per_probe": np.full(4, -1e308)})
    with pytest.raises(FloatingPointError, match="nonfinite"):
        compare_operator_scores(
            high, low, accepted_error_bound=0.0, trial_error_bound=0.0
        )
