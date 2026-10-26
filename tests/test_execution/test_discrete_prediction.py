"""``predict.bamd``-compatible prediction from discrete fits vs pinned mgcv."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data.source import ArrayRowSource, DataFrameRowSource
from tests.helpers import _AssertCollector, check_that
from tests.tolerances import STRICT

_LEVELS = ("low", "mid", "high")


def _frame(n_rows: int = 97) -> pd.DataFrame:
    position = np.arange(n_rows, dtype=float)
    x = (position + 0.2 * np.sin(position * 0.7)) / n_rows
    z = np.cos(position * 0.23) + 0.07 * np.sin(position * 0.91)
    y = 0.4 - 0.3 * z + np.sin(4.5 * x) + 0.4 * np.sin(3 * x) * np.cos(2 * z)
    return pd.DataFrame({"x": x, "z": z, "y": y + 0.05 * np.cos(position)})


def _factor_frame(n_rows: int = 72) -> pd.DataFrame:
    position = np.arange(1, n_rows + 1, dtype=float)
    x = (position - 0.5) / n_rows + 0.03 * np.sin(position)
    w = np.cos(position * 0.71) + position / n_rows
    g = pd.Series(
        pd.Categorical(
            np.resize(np.array(_LEVELS), n_rows), categories=_LEVELS, ordered=True
        )
    )
    y = 0.2 + np.sin(2.5 * x) + np.where(g == "mid", 0.3 * w, -0.2 * w)
    return pd.DataFrame({"y": y, "x": x, "w": w, "g": g})


def _requests(seed: int, columns: dict[str, tuple[float, float]]):
    """A held-out request (exact under predict.bamd) and a rounded one.

    Both extend beyond the training range.  The large request has more than
    1000 distinct values per variable, so pinned ``compress.df`` rounds it.
    """
    rng = np.random.default_rng(seed)
    held_out = pd.DataFrame(
        {
            name: np.r_[low - 0.15, rng.uniform(low, high, 6), high + 0.2]
            for name, (low, high) in columns.items()
        }
    )
    large = pd.DataFrame(
        {
            name: rng.uniform(low - 0.15, high + 0.2, 1300)
            for name, (low, high) in columns.items()
        }
    )
    return held_out, large


def _control(policy: str, **kwargs) -> FitControl:
    discretization = ContinuousDiscretizationControl(
        m=30,
        factor_names=kwargs.pop("factor_names", ()),
        prediction_policy=policy,
    )
    values = {
        "execution": "discrete",
        "batch_rows": 13,
        "uncertainty": "fisher",
        "discretization": discretization,
    }
    values.update(kwargs)
    return FitControl(**values)


def _close(observed, expected):
    return lambda: np.testing.assert_allclose(
        observed, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )


_CASES = {
    "smooth+parametric": "y ~ z + s(x,bs='cr',k=6)",
    "tensor margins": "y ~ te(x,z,bs='cr',k=3)",
    "numeric by": "y ~ z + s(x,by=z,bs='cr',k=6)",
    "joint tp": "y ~ s(x,z,bs='tp',k=12)",
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_working_model_prediction_matches_pinned_predict_bam(r_bridge, case) -> None:
    formula = _CASES[case]
    frame = _frame(150 if case == "joint tp" else 97)
    held_out, large = _requests(17, {"x": (0.0, 1.0), "z": (-1.0, 1.0)})
    if case == "joint tp":
        # A joint 2-D group rounds only above 100 x 100 distinct rows.
        rng = np.random.default_rng(5)
        large = pd.DataFrame(
            {
                "x": rng.uniform(-0.15, 1.2, 10_050),
                "z": rng.uniform(-1.15, 1.2, 10_050),
            }
        )
    results = {
        policy: GAM(
            formula, control=_control(policy, smoothing_strategy="working_model_reml")
        ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
        for policy in ("continuous_frozen_basis", "mgcv_predict_bamd")
    }
    collector = _AssertCollector()
    for request_label, request in (("held-out", held_out), ("large", large)):
        exact = r_bridge.fit_bam_discrete_gaussian(
            formula, frame, m=30, newdata=request
        )
        compressed = r_bridge.fit_bam_discrete_gaussian(
            formula, frame, m=30, newdata=request, discrete_prediction=True
        )
        for policy, expected in (
            ("continuous_frozen_basis", exact),
            ("mgcv_predict_bamd", compressed),
        ):
            result = results[policy]
            fit, se = result.predict(request, pred_type="link", se_fit=True)
            label = f"{case}/{request_label}/{policy}"
            collector.check(
                f"{label}/coefficients",
                _close(result.coefficients, expected.coefficients),
            )
            collector.check(f"{label}/fit", _close(fit, expected.prediction))
            collector.check(f"{label}/se", _close(se, expected.prediction_se))
            # Bounded batches of the same request reproduce R exactly.
            batches = [
                output
                for _, output in result.predict_iter(
                    DataFrameRowSource(request), 211, pred_type="link", se_fit=True
                )
            ]
            collector.check(
                f"{label}/batched fit",
                _close(np.concatenate([b[0] for b in batches]), expected.prediction),
            )
            collector.check(
                f"{label}/batched se",
                _close(np.concatenate([b[1] for b in batches]), expected.prediction_se),
            )
        if request_label == "held-out":
            # Fewer distinct rows than mgcv's grid: predict.bamd is exact.
            collector.check(
                f"{case}/held-out identity",
                _close(compressed.prediction, exact.prediction),
            )
        else:
            collector.check(
                f"{case}/large request is rounded",
                lambda compressed=compressed, exact=exact: check_that(
                    not np.allclose(
                        compressed.prediction,
                        exact.prediction,
                        rtol=STRICT.rtol,
                        atol=STRICT.atol,
                    ),
                    "request rounding must change predict.bam's predictions",
                ),
            )
    collector.raise_if_any("pinned predict.bam parity")


def test_factor_by_tensor_prediction_matches_pinned_predict_bamd(r_bridge) -> None:
    frame = _factor_frame()
    formula = "y ~ te(x, w, bs='cr', k=3, by=g)"
    smoothing = np.array([0.35, 0.7, 0.45, 0.8])
    held_out, large = _requests(23, {"x": (0.0, 1.0), "w": (-0.9, 2.0)})
    rng = np.random.default_rng(29)
    collector = _AssertCollector()
    for request_label, request in (("held-out", held_out), ("large", large)):
        request = request.assign(
            g=pd.Categorical(
                rng.choice(np.array(_LEVELS), len(request)),
                categories=_LEVELS,
                ordered=True,
            )
        )
        expected = r_bridge.fit_bam_discrete_fixed_sp(
            formula,
            frame,
            smoothing,
            factor_names=("g",),
            m=30,
            newdata=request,
            discrete_prediction=True,
        )
        for uncertainty in ("fisher", "covariance"):
            result = GAM(
                formula,
                sp=smoothing,
                control=_control(
                    "mgcv_predict_bamd",
                    factor_names=("g",),
                    uncertainty=uncertainty,
                    batch_rows=9,
                ),
            ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
            fit, se = result.predict(request, pred_type="link", se_fit=True)
            label = f"{request_label}/{uncertainty}"
            collector.check(f"{label}/fit", _close(fit, expected.prediction))
            collector.check(f"{label}/se", _close(se, expected.prediction_se))
    collector.raise_if_any("factor-by predict.bamd parity")


def _bamd_result():
    frame = _frame()
    return GAM(
        "y ~ z + s(x,bs='cr',k=6)",
        control=_control("mgcv_predict_bamd", smoothing_strategy="working_model_reml"),
    ).fit(DataFrameRowSource(frame, response="y"), result="prediction")


def test_frozen_request_grid_is_invariant_to_prediction_batches() -> None:
    result = _bamd_result()
    _, request = _requests(31, {"x": (0.0, 1.0), "z": (-1.0, 1.0)})
    expected_fit, expected_se = result.predict(request, se_fit=True)
    collector = _AssertCollector()
    for batch_rows in (1, 37, 499, len(request)):
        positions, fits, ses = [], [], []
        for rows, (fit, se) in result.predict_iter(
            DataFrameRowSource(request), batch_rows, se_fit=True
        ):
            positions.append(rows)
            fits.append(fit)
            ses.append(se)
        collector.check(
            f"B{batch_rows}/positions",
            lambda positions=positions: np.testing.assert_array_equal(
                np.concatenate(positions), np.arange(len(request))
            ),
        )
        collector.check(
            f"B{batch_rows}/fit", _close(np.concatenate(fits), expected_fit)
        )
        collector.check(f"B{batch_rows}/se", _close(np.concatenate(ses), expected_se))
    collector.raise_if_any("frozen request batching")
    # Rediscretizing each batch as its own request changes the answer.
    independent = np.concatenate(
        [result.predict(request.iloc[start : start + 499]) for start in (0, 499, 998)]
    )
    assert not np.allclose(
        independent, expected_fit, rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_prediction_matrix_and_pickle_keep_the_policy() -> None:
    result = _bamd_result()
    _, request = _requests(37, {"x": (0.0, 1.0), "z": (-1.0, 1.0)})
    predictor = result.to_predictor()
    frozen = predictor.freeze_request(request)
    assert frozen is not None
    assert frozen.rounded_names == ("x", "z")
    matrix = result.predict_matrix(request)
    snapped = predictor._predict_spec.build_predict_matrix(frozen.apply(request))
    np.testing.assert_array_equal(matrix, snapped)
    np.testing.assert_allclose(
        matrix @ result.coefficients,
        result.predict(request, pred_type="link"),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.prediction_policy == "mgcv_predict_bamd"
    assert restored.to_predictor().prediction_policy == "mgcv_predict_bamd"
    np.testing.assert_array_equal(restored.predict(request), result.predict(request))
    exact = GAM(
        "y ~ z + s(x,bs='cr',k=6)",
        control=_control(
            "continuous_frozen_basis", smoothing_strategy="working_model_reml"
        ),
    ).fit(DataFrameRowSource(_frame(), response="y"), result="prediction")
    assert exact.to_predictor().freeze_request(request) is None
    assert exact.prediction_policy == "continuous_frozen_basis"
    assert exact.to_predictor().prediction_policy == "continuous_frozen_basis"
    # Legacy predictor pickles without the discrete policy fields load as
    # exact, policy-free predictors.
    state = dict(restored.to_predictor().__dict__)
    del state["_request_compression"], state["_prediction_policy"]
    legacy = object.__new__(type(predictor))
    legacy.__setstate__(state)
    assert legacy.prediction_policy is None
    assert legacy.freeze_request(request) is None


def test_changed_source_after_request_freeze_is_rejected() -> None:
    result = _bamd_result()
    _, request = _requests(41, {"x": (0.0, 1.0), "z": (-1.0, 1.0)})
    columns = {name: request[name].to_numpy().copy() for name in request}
    source = ArrayRowSource(columns)
    batches = result.predict_iter(source, 400)
    columns["x"][0] = 0.5
    with pytest.raises(RuntimeError, match="changed after its prediction request"):
        next(batches)
    late = result.predict_iter(ArrayRowSource(columns), 400)
    next(late)
    columns["x"][-1] = 0.25
    with pytest.raises(RuntimeError, match="changed after its prediction request"):
        list(late)


def test_model_without_smooths_records_exact_effective_policy() -> None:
    frame = _frame()
    result = GAM(
        "y ~ x + z", sp=[], control=_control("mgcv_predict_bamd", batch_rows=11)
    ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
    _, request = _requests(43, {"x": (0.0, 1.0), "z": (-1.0, 1.0)})
    # predict.bam has no dinfo without smooth terms: prediction is exact,
    # and the result says so while keeping the requested policy.
    assert result.prediction_policy == "continuous_frozen_basis"
    assert result.to_predictor().prediction_policy == "continuous_frozen_basis"
    metadata = result.discretization
    assert metadata.prediction_policy == "continuous_frozen_basis"
    assert metadata.requested_prediction_policy == "mgcv_predict_bamd"
    assert "no smooth terms" in metadata.prediction_policy_note
    assert result.to_predictor().freeze_request(request) is None
    design = np.column_stack((np.ones(len(request)), request.x, request.z))
    np.testing.assert_allclose(
        result.predict(request, pred_type="link"),
        design @ result.coefficients,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_prediction_policy_is_explicit_and_validated() -> None:
    assert ContinuousDiscretizationControl().prediction_policy == (
        "continuous_frozen_basis"
    )
    with pytest.raises(ValueError, match="prediction_policy must be"):
        ContinuousDiscretizationControl(prediction_policy="training_grid")


def test_predictor_rejects_inconsistent_request_compression() -> None:
    result = _bamd_result()
    predictor = result.to_predictor()
    with pytest.raises(ValueError, match="requires prediction_policy"):
        dataclasses.replace(predictor, _prediction_policy="continuous_frozen_basis")
    with pytest.raises(ValueError, match="RequestCompressionSpec"):
        dataclasses.replace(predictor, _request_compression=object())
    with pytest.raises(ValueError, match="requires request compression"):
        dataclasses.replace(predictor, _request_compression=None)
    smooth = result.discretization
    assert smooth.requested_prediction_policy == smooth.prediction_policy
    assert smooth.prediction_policy_note is None


def _factor_results(formula: str, *, ordered: bool, **fit) -> dict[str, object]:
    """The same discrete fit under both prediction policies."""
    frame = _factor_frame()
    if not ordered:
        frame["g"] = frame["g"].cat.as_unordered()
    return {
        policy: GAM(
            formula,
            control=_control(policy, factor_names=("g",), **fit.get("control", {})),
            sp=fit.get("sp"),
        ).fit(DataFrameRowSource(frame, response="y"), result="prediction")
        for policy in ("mgcv_predict_bamd", "continuous_frozen_basis")
    } | {"frame": frame}


def _factor_requests(seed: int, size: int, columns: dict[str, tuple[float, float]]):
    """Requests with valid, missing (None) and unseen ("new") factor values."""
    rng = np.random.default_rng(seed)
    values = np.array([*_LEVELS, None, "new"], dtype=object)
    request = pd.DataFrame(
        {name: rng.uniform(low, high, size) for name, (low, high) in columns.items()}
    )
    request["g"] = values[np.resize(np.arange(len(values)), size)]
    return request


_RE_FORMULA = "y ~ s(x,bs='cr',k=6) + s(g,bs='re')"


def test_random_effect_missing_and_unseen_levels_contribute_zero(r_bridge):
    """User decision: a missing or unseen RE group gives the prior mean (0)."""
    results = _factor_results(
        _RE_FORMULA,
        ordered=False,
        control={"smoothing_strategy": "working_model_reml"},
    )
    bamd = results["mgcv_predict_bamd"]
    exact = results["continuous_frozen_basis"]
    (re_info,) = (
        info
        for info in bamd.to_predictor()._predict_spec.smooth_info
        if info.label == "s(g)"
    )
    re_columns = slice(re_info.first_coef, re_info.last_coef)
    collector = _AssertCollector()
    for label, request in (
        ("held-out", _factor_requests(51, 10, {"x": (-0.1, 1.1)})),
        ("rounded", _factor_requests(53, 1300, {"x": (-0.1, 1.1)})),
    ):
        excluded = ~request["g"].isin(_LEVELS).to_numpy()
        fit, se = bamd.predict(request, pred_type="link", se_fit=True)
        collector.check(
            f"{label}/finite",
            lambda fit=fit, se=se: check_that(
                bool(np.all(np.isfinite(fit)) and np.all(np.isfinite(se))),
                "missing/unseen random-effect rows must stay finite",
            ),
        )
        # The same covariates with the random-effect contribution set to 0.
        zeroed = bamd.predict_matrix(request.assign(g="low"))
        zeroed[:, re_columns] = 0.0
        collector.check(
            f"{label}/zero contribution",
            _close(fit[excluded], (zeroed @ bamd.coefficients)[excluded]),
        )
        expected = r_bridge.fit_bam_discrete_gaussian(
            _RE_FORMULA,
            results["frame"],
            factor_names=("g",),
            m=30,
            newdata=request,
            discrete_prediction=True,
        )
        defined = np.isfinite(expected.prediction)
        missing = request["g"].isna().to_numpy()
        # Pinned predict.bamd is NA for missing groups (the intended
        # divergence) and zeroes unseen ones, exactly as this policy does.
        collector.check(
            f"{label}/R defined rows",
            lambda defined=defined, missing=missing: check_that(
                bool(np.all(defined == ~missing)),
                "R is NA exactly for missing random-effect groups",
            ),
        )
        collector.check(
            f"{label}/R fit", _close(fit[defined], expected.prediction[defined])
        )
        collector.check(
            f"{label}/R se", _close(se[defined], expected.prediction_se[defined])
        )
        if label == "held-out":
            # No metric rounding: identical to the default exact path.
            exact_fit, exact_se = exact.predict(request, pred_type="link", se_fit=True)
            collector.check(f"{label}/default fit", _close(fit, exact_fit))
            collector.check(f"{label}/default se", _close(se, exact_se))
        batches = np.concatenate(
            [
                output
                for _, output in bamd.predict_iter(
                    DataFrameRowSource(request), 97, pred_type="link"
                )
            ]
        )
        collector.check(f"{label}/batched", _close(batches, fit))
    collector.raise_if_any("random-effect missing/unseen levels")


def test_factor_by_and_parametric_levels_keep_default_semantics() -> None:
    by = _factor_results(
        "y ~ te(x, w, bs='cr', k=3, by=g)",
        ordered=True,
        sp=np.array([0.35, 0.7, 0.45, 0.8]),
    )
    request = _factor_requests(57, 10, {"x": (0.0, 1.0), "w": (-0.9, 2.0)})
    outputs = []
    for policy in ("mgcv_predict_bamd", "continuous_frozen_basis"):
        with pytest.warns(UserWarning, match="not in original fit"):
            outputs.append(by[policy].predict(request, pred_type="link"))
    np.testing.assert_allclose(
        outputs[0], outputs[1], rtol=STRICT.rtol, atol=STRICT.atol, equal_nan=True
    )
    invalid = ~request["g"].isin(_LEVELS).to_numpy()
    assert np.all(np.isnan(outputs[0][invalid]))
    assert np.all(np.isfinite(outputs[0][~invalid]))

    parametric = _factor_results(
        "y ~ g + s(x,bs='cr',k=6)",
        ordered=False,
        control={"smoothing_strategy": "working_model_reml"},
    )
    request = _factor_requests(59, 8, {"x": (0.0, 1.0)})
    missing = request[request["g"].ne("new").to_numpy()].reset_index(drop=True)
    outputs = [
        parametric[policy].predict(missing, pred_type="link")
        for policy in ("mgcv_predict_bamd", "continuous_frozen_basis")
    ]
    np.testing.assert_allclose(
        outputs[0], outputs[1], rtol=STRICT.rtol, atol=STRICT.atol, equal_nan=True
    )
    assert np.all(np.isnan(outputs[0][missing["g"].isna().to_numpy()]))
    for policy in ("mgcv_predict_bamd", "continuous_frozen_basis"):
        with pytest.raises(ValueError, match="new level"):
            parametric[policy].predict(request)


def test_invalid_pred_type_rejects_before_request_prepass(monkeypatch) -> None:
    from jaxgam.data.continuous_request import RequestCompressionSpec

    result = _bamd_result()
    _, request = _requests(47, {"x": (0.0, 1.0), "z": (-1.0, 1.0)})

    def no_prepass(*_args, **_kwargs):
        raise AssertionError("pred_type must be validated before the pre-pass")

    monkeypatch.setattr(RequestCompressionSpec, "freeze", no_prepass)
    monkeypatch.setattr(RequestCompressionSpec, "freeze_source", no_prepass)

    class NoScanSource(DataFrameRowSource):
        def scan(self, _batch_rows):
            raise AssertionError("pred_type must be validated before any scan")

    with pytest.raises(ValueError, match="pred_type must be"):
        result.predict(request, pred_type="terms")
    with pytest.raises(ValueError, match="pred_type must be"):
        result.predict_iter(NoScanSource(request), 100, pred_type="terms")
