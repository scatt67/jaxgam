"""Owning BAM score and retained-uncertainty gates for the PR11 fixed fit."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import jaxgam.execution.discrete as execution
from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data.discretize import DiscretizationBudget
from jaxgam.data.source import DataFrameRowSource
from jaxgam.formula.continuous_setup import (
    ContinuousDiscreteSetup,
    build_continuous_discrete_setup,
)
from jaxgam.formula.fitting_prepare import total_penalty_spaces
from jaxgam.formula.parser import parse_formula
from jaxgam.penalties.structure import (
    DiagonalPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
from tests.helpers import _AssertCollector, check_that
from tests.r_bridge import RBridge
from tests.tolerances import MODERATE, STRICT


def _frame() -> pd.DataFrame:
    position = np.arange(97, dtype=float)
    x = (position + 0.2 * np.sin(position * 0.7)) / len(position)
    z = np.cos(position * 0.23) + 0.07 * np.sin(position * 0.91)
    y = 0.4 - 0.3 * z + np.sin(4.5 * x) + 0.05 * np.cos(position)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def _control(batch_rows: int, uncertainty: str = "none") -> FitControl:
    return FitControl(
        execution="discrete",
        batch_rows=batch_rows,
        uncertainty=uncertainty,
        discretization=ContinuousDiscretizationControl(m=30),
    )


def _check_owned_pickle(provider, original_provider, restored, result):
    check_that(
        provider is not None
        and not provider.flags.writeable
        and not np.shares_memory(provider, original_provider)
        and not restored.coefficients.flags.writeable
        and not np.shares_memory(restored.coefficients, result.coefficients),
        "pickle must retain independent read-only arrays",
    )


@pytest.mark.parametrize(
    "case", ["weighted", "zero", "separate", "coupled", "coupled_control"]
)
def test_fixed_score_owns_bam_weight_and_structural_rank_conventions(
    r_bridge, case: str
) -> None:
    frame = _frame()
    formula = "y ~ z + s(x,bs='cr',k=6)"
    smoothing = np.array([0.55])
    weights = np.ones(len(frame))
    if case == "weighted":
        weights = np.linspace(0.6, 1.4, len(frame))
    elif case == "zero":
        weights[np.arange(len(frame)) % 7 == 0] = 0.0
    elif case == "separate":
        formula = "y ~ s(x,bs='cr',k=6)+s(z,bs='cr',k=6)"
        smoothing = np.array([1e-10, 100.0])
    elif case in ("coupled", "coupled_control"):
        formula = "y ~ te(x,z,bs='cr',k=3)"
        smoothing = (
            np.array([1e-10, 100.0]) if case == "coupled" else np.array([0.35, 0.7])
        )
    offset = np.linspace(-0.1, 0.2, len(frame))
    expected = r_bridge.fit_bam_discrete_fixed_sp(
        formula, frame, smoothing, m=30, weights=weights, offset=offset
    )
    result = GAM(formula, sp=smoothing, control=_control(13)).fit(
        DataFrameRowSource(frame, response="y", weights=weights, offset=offset),
        result="prediction",
    )
    collector = _AssertCollector()
    collector.check(
        "coefficients",
        lambda: np.testing.assert_allclose(
            result.coefficients,
            expected.coefficients,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    for field in ("deviance", "scale", "edf_total", "score"):
        # Only this ill-conditioned coupled determinant uses MODERATE after
        # five documented source/solver correction passes.  Other fields and
        # all three original root regressions remain STRICT.
        tolerance = MODERATE if case == "coupled" and field == "score" else STRICT
        collector.check(
            field,
            lambda field=field, tolerance=tolerance: np.testing.assert_allclose(
                getattr(result, field),
                getattr(expected, field),
                rtol=tolerance.rtol,
                atol=tolerance.atol,
            ),
        )
    collector.check(
        "pinned convergence fields",
        lambda: check_that(
            expected.converged and expected.mgcv_converged,
            "supported BAM oracle case must pass both convergence fields",
        ),
    )
    collector.raise_if_any(case)


@pytest.fixture
def uncertainty_reference(r_bridge):
    frame = _frame()
    formula = "y ~ z + s(x,bs='cr',k=6)"
    smoothing = np.array([0.55])
    setup = build_continuous_discrete_setup(
        parse_formula(formula), DataFrameRowSource(frame, response="y"), m=30
    )
    assert isinstance(setup, ContinuousDiscreteSetup)
    snapped = {}
    for group in setup.compact.groups:
        for name, values in zip(group.names, group.index.values, strict=True):
            snapped[name] = values[group.index.selector]
    X = setup.predict_spec.build_predict_matrix(pd.DataFrame(snapped))
    assert setup.penalties is not None
    penalty = setup.penalties.materialize(np.log(smoothing))
    hessian = X.T @ X + penalty
    beta = np.linalg.solve(hessian, X.T @ frame.y.to_numpy())
    rank = sum(basis.shape[1] for _, basis in total_penalty_spaces(setup.penalties)[1])
    null_dimension = X.shape[1] - rank
    residual = frame.y.to_numpy() - X @ beta
    optimum_scale = float(
        (residual @ residual + beta @ penalty @ beta) / (len(frame) - null_dimension)
    )
    # BAM's finite log-scale Newton sequence owns the reported scale, rather
    # than the closed-form optimum. Keep the independent explicit design and
    # check that the oracle satisfies its source stopping bound as well.
    oracle = r_bridge.fit_bam_discrete_fixed_sp(formula, frame, smoothing, m=30)
    scale = oracle.scale
    np.testing.assert_allclose(
        oracle.coefficients, beta, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert abs(1.0 - scale / optimum_scale) <= 1e-7 * (abs(np.log(scale)) + 1.0)
    newdata = pd.DataFrame(
        {"x": np.linspace(-0.1, 1.1, 9), "z": np.linspace(-0.7, 0.8, 9)}
    )
    prediction_matrix = setup.predict_spec.build_predict_matrix(newdata)
    setup.cleanup()
    return frame, formula, smoothing, hessian, beta, scale, newdata, prediction_matrix


def test_uncertainty_prediction_pickle_and_batch_contract(uncertainty_reference):
    frame, formula, smoothing, hessian, beta, scale, newdata, Xnew = (
        uncertainty_reference
    )
    collector = _AssertCollector()
    for uncertainty in ("fisher", "covariance"):
        for batch_rows in (5, 17):
            label = f"{uncertainty}/B={batch_rows}"
            result = GAM(
                formula, sp=smoothing, control=_control(batch_rows, uncertainty)
            ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
            Vp = scale * np.linalg.inv(hessian)
            expected_se = np.sqrt(np.einsum("ij,jk,ik->i", Xnew, Vp, Xnew))
            observed_fit, observed_se = result.predict(newdata, se_fit=True)
            covariance = result.materialize_covariance(32 * hessian.size)
            restored = pickle.loads(pickle.dumps(result))
            streamed = list(
                restored.predict_iter(
                    DataFrameRowSource(newdata), batch_rows=2, se_fit=True
                )
            )
            fields = {
                "coefficients": (result.coefficients, beta),
                "continuous prediction": (observed_fit, Xnew @ beta),
                "standard error": (observed_se, expected_se),
                "covariance materialization": (covariance, Vp),
                "batch scale": (result.scale, scale),
                "pickle prediction": (restored.predict(newdata), Xnew @ beta),
                "streamed standard error": (
                    np.concatenate([output[1] for _, output in streamed]),
                    expected_se,
                ),
            }
            if uncertainty == "fisher":
                factor = result.to_predictor()._fisher_factor
                assert factor is not None
                assert result.to_predictor().Vp is None
                fields["retained factor"] = (factor @ factor.T, hessian)
            else:
                assert result.to_predictor().Vp is not None
                assert result.to_predictor()._fisher_factor is None
            for name, (observed, expected) in fields.items():
                collector.check(
                    f"{label} {name}",
                    lambda observed=observed, expected=expected: (
                        np.testing.assert_allclose(
                            observed, expected, rtol=STRICT.rtol, atol=STRICT.atol
                        )
                    ),
                )
            provider = (
                restored.to_predictor()._fisher_factor
                if uncertainty == "fisher"
                else restored.to_predictor().Vp
            )
            original_provider = (
                result.to_predictor()._fisher_factor
                if uncertainty == "fisher"
                else result.to_predictor().Vp
            )
            collector.check(
                f"{label} ownership",
                lambda owned=(provider, original_provider, restored, result): (
                    _check_owned_pickle(*owned)
                ),
            )
    collector.raise_if_any("discrete uncertainty")


def test_centered_response_variance_survives_large_location(monkeypatch):
    frame = _frame()
    frame["y"] += 1e8
    observed = []
    original = execution._pinned_fixed_scale

    def capture(penalized, n_rows, null_dimension, centered_square_sum):
        observed.append(centered_square_sum / (n_rows - 1))
        return original(penalized, n_rows, null_dimension, centered_square_sum)

    monkeypatch.setattr(execution, "_pinned_fixed_scale", capture)
    for batch_rows in (1, 13, 97):
        GAM("y ~ z + s(x,bs='cr',k=6)", sp=[0.55], control=_control(batch_rows)).fit(
            DataFrameRowSource(frame, response="y"), result="prediction"
        )
    centered = frame.y.to_numpy() - frame.y.iloc[0]
    np.testing.assert_allclose(
        observed, np.var(centered, ddof=1), rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_structural_logdet_keeps_weak_and_zero_penalty_directions():
    block = PenaltyBlock(
        0,
        3,
        (0, 1),
        (
            DiagonalPenalty(np.array([2.0, 0.0, 0.0])),
            DiagonalPenalty(np.array([0.0, 3.0, 0.0])),
        ),
        IdentityTransform(3),
        (1, 1),
    )
    structure = PenaltyStructure(3, (block,))
    observed = execution._log_pdet(structure, np.array([-80.0, 80.0]))
    np.testing.assert_allclose(
        observed, np.log(6.0), rtol=STRICT.rtol, atol=STRICT.atol
    )
    empty = PenaltyStructure(3, ())
    assert execution._log_pdet(empty, np.empty(0)) == 0.0
    with pytest.raises(FloatingPointError, match="structural rank"):
        execution._positive_logdet(np.diag([1.0, 0.0]), 2)


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"budget": None}, TypeError),
        ({"prediction_policy": "training_grid"}, ValueError),
    ],
)
def test_public_policy_rejects_false_metadata(kwargs, exception):
    with pytest.raises(exception):
        ContinuousDiscretizationControl(**kwargs)
    assert isinstance(ContinuousDiscretizationControl().budget, DiscretizationBudget)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": np.ones(3)},
        {"offset": np.ones(3)},
        {"weights": np.full(97, -1.0)},
        {"offset": np.full(97, np.nan)},
        {"factor_names": ("x", "x")},
        {"factor_names": ("missing",)},
        {"m": True},
        {"m": (1,)},
    ],
)
def test_bridge_invalid_inputs_reject_before_r_process(monkeypatch, kwargs):
    monkeypatch.setattr(RBridge, "check_versions", staticmethod(lambda: (True, "")))

    def reject_process(*_args, **_kwargs):
        raise AssertionError("invalid inputs must reject before launching R")

    monkeypatch.setattr("tests.r_bridge.subprocess.run", reject_process)
    with pytest.raises(ValueError, match=r"weights|offset|factor_names|m "):
        RBridge().fit_bam_discrete_fixed_sp(
            "y ~ s(x,bs='cr',k=6)", _frame(), np.array([0.55]), **kwargs
        )
