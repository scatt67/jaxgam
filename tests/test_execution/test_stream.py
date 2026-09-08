"""Fixed-sp host-streamed PIRLS integration and bounded-kernel tests."""

from __future__ import annotations

import pickle
from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.control import FitControl
from jaxgam.data.source import DataFrameRowSource, RowBatch
from jaxgam.execution import stream as stream_execution
from jaxgam.execution.stream import StreamPIRLSControl, fit_streamed_pirls
from jaxgam.families.base import REAL, ExponentialFamily
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import FittingData, PreparedFittingMetadata, _to_jax_structure
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionParameters,
)
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.fitting.reml import estimate_edf
from jaxgam.fitting.state import (
    CholeskyCoefficientFactor,
    PivotedQRCoefficientFactor,
    StreamFitState,
)
from jaxgam.fitting.stream_kernels import (
    accumulate_working_statistics,
    empty_statistics,
    positive_qr_working_rows,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.inference._core import finish_prediction
from jaxgam.jax_utils import array_module
from jaxgam.links.links import IdentityLink, Link
from jaxgam.results import GAMPredictionResult
from tests.helpers import r_available
from tests.tolerances import MODERATE, STRICT


def _fixture(family_name: str, *, n: int = 83) -> tuple[pd.DataFrame, object]:
    rng = np.random.default_rng(189)
    x = np.linspace(-0.9, 1.1, n)
    offset = 0.15 * np.sin(2 * x)
    weights = 0.2 + rng.random(n)
    eta = 0.3 + 0.8 * x + offset
    if family_name == "gaussian":
        y = eta + rng.normal(scale=0.2, size=n)
        family = Gaussian()
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta))
        family = Poisson()
    else:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
        family = Binomial()
    return pd.DataFrame({"x": x, "y": y, "w": weights, "off": offset}), family


class _UnregisteredGaussianLike(ExponentialFamily):
    """A truthful Gaussian-like custom family, absent from all registries."""

    family_name = "test_unregistered_gaussian_like"
    scale_known = True
    response_support = REAL
    canonical_link_cls = IdentityLink

    @property
    def default_link(self) -> Link:
        return IdentityLink()

    def variance(self, mu):
        return array_module(mu).ones_like(mu)

    def dvar(self, mu):
        return array_module(mu).zeros_like(mu)

    def saturated_loglik(self, y, wt, scale, *, max_y: int = 0):  # noqa: ARG002
        return jnp.sum(
            jnp.where(wt > 0, -0.5 * jnp.log(2.0 * jnp.pi * scale / wt), 0.0)
        )

    def deviance_resids(self, y, mu, wt):
        xp = array_module(y)
        return xp.sign(y - mu) * xp.sqrt(wt * (y - mu) ** 2)

    def deviance_contributions(self, y, mu, wt):
        return wt * (y - mu) ** 2

    def deviance_derivative_contributions(self, y, mu, wt):
        return wt * (y - mu) ** 2

    def aic(self, y, mu, wt, scale):  # noqa: ARG002
        return float(np.sum(wt * (y - mu) ** 2))

    def _initialize_impl(self, y, wt):  # noqa: ARG002
        return y.copy()

    def valid_mu(self, mu):
        return array_module(mu).isfinite(mu)

    def valid_eta(self, eta):
        return array_module(eta).isfinite(eta)

    def execution_summary_from_batch(self, y, wt, valid):
        """Append a vector leaf to prove the host scanner is pytree-generic."""
        base = super().execution_summary_from_batch(y, wt, valid)
        return (*base, jnp.array([jnp.sum(jnp.where(valid, y, 0.0))]))

    def merge_execution_summaries(self, left, right):
        return (
            left[0] + right[0],
            left[1] + right[1],
            left[2] + right[2],
            jnp.logical_and(left[3], right[3]),
            left[4] + right[4],
        )

    def finalize_execution_summary(self, summary):
        result = super().finalize_execution_summary(summary[:4])
        result["custom_response_sum"] = float(summary[4][0])
        return result


@pytest.mark.parametrize("family_name", ["gaussian", "poisson", "binomial"])
@pytest.mark.parametrize("batch_rows", [1, 7, 29])
@pytest.mark.parametrize("solver_policy", ["cholesky", "qr"])
def test_streamed_fixed_sp_matches_dense_with_explicit_weights_and_offsets(
    family_name: str, batch_rows: int, solver_policy: str
) -> None:
    data, family = _fixture(family_name)
    spec = parse_formula('y ~ s(x, bs="cr", k=6)')
    source = DataFrameRowSource(
        data, response="y", weights=data.w.to_numpy(), offset=data.off.to_numpy()
    )
    prepared = prepare_model(spec, source, family=family)
    streamed = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.array([0.2]),
        control=StreamPIRLSControl(batch_rows=batch_rows, solver_policy=solver_policy),
    )
    dense = FittingData.from_setup(
        ModelSetup.build(
            spec,
            data,
            weights=data.w.to_numpy(),
            offset=data.off.to_numpy(),
        ),
        family,
    )
    dense_result = pirls_loop(
        dense.X,
        dense.y,
        dense.beta_init,
        dense.S_lambda(jnp.array([0.2])),
        family,
        dense.wt,
        dense.offset,
    )
    np.testing.assert_allclose(
        np.asarray(streamed.coefficients),
        np.asarray(dense_result.coefficients),
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    np.testing.assert_allclose(
        np.asarray(streamed.deviance),
        np.asarray(dense_result.deviance),
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    np.testing.assert_allclose(
        np.asarray(streamed.edf),
        np.asarray(estimate_edf(dense_result.XtWX_fisher, dense_result.L_fisher)),
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    if solver_policy == "qr":
        assert isinstance(
            streamed.fisher_coefficient_factor, PivotedQRCoefficientFactor
        )
        expected_inverse = np.asarray(
            streamed.scale
            * jnp.asarray(
                np.linalg.solve(
                    np.asarray(dense_result.L_fisher).T,
                    np.linalg.solve(
                        np.asarray(dense_result.L_fisher),
                        np.eye(prepared.n_coef),
                    ),
                )
            )
        )
        actual_inverse = np.asarray(
            streamed.scale
            * streamed.fisher_coefficient_factor.hessian_inverse(
                jnp.eye(prepared.n_coef)
            )
        )
        np.testing.assert_allclose(
            actual_inverse,
            expected_inverse,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        expected_se = np.linalg.norm(
            np.linalg.solve(np.asarray(dense_result.L_fisher), np.asarray(dense.X).T),
            axis=0,
        )
        actual_se = np.linalg.norm(
            np.asarray(
                streamed.fisher_coefficient_factor.root_transpose_inverse(
                    jnp.asarray(dense.X).T
                )
            ),
            axis=0,
        )
        np.testing.assert_allclose(
            actual_se, expected_se, rtol=MODERATE.rtol, atol=MODERATE.atol
        )
        with np.testing.assert_raises(RuntimeError):
            _ = streamed.fisher_factor
    assert streamed.converged
    assert not streamed.line_search_failed
    assert streamed.stationarity < 1e-7


def test_padded_nan_tail_is_sanitized_then_masked_after_weight_floor() -> None:
    family = Poisson()
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [jnp.nan, jnp.nan]])
    y = jnp.array([2.0, 3.0, jnp.nan])
    weight = jnp.array([1.0, 0.0, jnp.nan])
    offset = jnp.array([0.0, 0.0, jnp.nan])
    beta = jnp.array([0.1, 0.2])
    padded = accumulate_working_statistics(
        empty_statistics(2),
        X,
        y,
        weight,
        offset,
        jnp.array([True, True, False]),
        beta,
        parameters,
        family,
        context,
    )
    reference = accumulate_working_statistics(
        empty_statistics(2),
        X[:2],
        y[:2],
        weight[:2],
        offset[:2],
        jnp.array([True, True]),
        beta,
        parameters,
        family,
        context,
    )
    for got, expected in zip(padded, reference, strict=True):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(expected), rtol=STRICT.rtol, atol=STRICT.atol
        )
    qr_rows = positive_qr_working_rows(
        X,
        y,
        weight,
        offset,
        jnp.array([True, True, False]),
        beta,
        parameters,
        family,
        context,
    )
    qr_reference = positive_qr_working_rows(
        X[:2],
        y[:2],
        weight[:2],
        offset[:2],
        jnp.array([True, True]),
        beta,
        parameters,
        family,
        context,
    )
    np.testing.assert_allclose(
        np.asarray(qr_rows[0][:2]),
        np.asarray(qr_reference[0]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        np.asarray(qr_rows[1][:2]),
        np.asarray(qr_reference[1]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_array_equal(np.asarray(qr_rows[0][2]), np.zeros(2))
    assert qr_rows[0].shape == X.shape
    assert positive_qr_working_rows._cache_size() >= 1
    assert accumulate_working_statistics._cache_size() >= 1


@pytest.mark.parametrize("delta", [1e-6, 1e-8])
def test_streamed_qr_ill_conditioned_edf_and_se_use_final_root(delta: float) -> None:
    """QR EDF/SE remain stable where ``trace(G @ H^-1)`` is not."""
    rng = np.random.default_rng(9272)
    n = 503
    x = rng.normal(size=n)
    z = x + delta * rng.normal(size=n)
    X = np.column_stack((np.ones(n), x, z))
    weight = rng.uniform(0.5, 1.6, n)
    offset = 0.1 * np.sin(x)
    y = X @ np.array([0.3, 2.0, -1.0]) + offset + 1e-9 * rng.normal(size=n)
    source = DataFrameRowSource(
        pd.DataFrame({"x": x, "z": z, "y": y}),
        response="y",
        weights=weight,
        offset=offset,
    )
    family = Gaussian()
    prepared = prepare_model(parse_formula("y ~ x + z"), source, family=family)
    streamed = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.empty(0),
        control=StreamPIRLSControl(batch_rows=13, solver_policy="qr"),
    )
    weighted_X = np.sqrt(weight)[:, None] * X
    Q, R = np.linalg.qr(weighted_X)
    expected_beta = np.linalg.solve(R, Q.T @ (np.sqrt(weight) * (y - offset)))
    expected_se = np.linalg.norm(np.linalg.solve(R.T, X.T), axis=0)
    actual_se = np.linalg.norm(
        np.asarray(streamed.fisher_coefficient_factor.root_transpose_inverse(X.T)),
        axis=0,
    )
    np.testing.assert_allclose(
        np.asarray(streamed.coefficients),
        expected_beta,
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    np.testing.assert_allclose(
        actual_se, expected_se, rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        np.asarray(streamed.edf), 3.0, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert streamed.converged


def test_qr_penalty_roots_are_built_once_per_fixed_sp_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, family = _fixture("gaussian", n=41)
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=5)'), source, family=family
    )
    original = stream_execution.qr_penalty_roots
    calls = 0

    def count_roots(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(stream_execution, "qr_penalty_roots", count_roots)
    fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.array([0.1]),
        control=StreamPIRLSControl(batch_rows=7, solver_policy="qr"),
    )
    assert calls == 1


def test_qr_source_scan_accounting_matches_replayed_source_calls() -> None:
    """QR has one source replay per recorded scan; roots add no hidden scan."""
    data, family = _fixture("gaussian", n=41)
    base = DataFrameRowSource(data, response="y")
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=5)'), base, family=family
    )

    class CountingSource:
        n_rows = base.n_rows

        def __init__(self) -> None:
            self.scan_calls = 0

        def fingerprint(self) -> str:
            return base.fingerprint()

        def scan(self, batch_rows: int):
            self.scan_calls += 1
            yield from base.scan(batch_rows)

    source = CountingSource()
    state = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.array([0.1]),
        control=StreamPIRLSControl(batch_rows=7, solver_policy="qr"),
    )
    assert source.scan_calls == state.source_scans


@pytest.mark.parametrize("n", [53, 509])
def test_qr_retained_fit_state_is_coefficient_sized_as_source_grows(n: int) -> None:
    """This checks retained state shape, not process peak allocation."""
    data, family = _fixture("gaussian", n=n)
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    state = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.empty(0),
        control=StreamPIRLSControl(batch_rows=11, solver_policy="qr"),
    )
    p = prepared.n_coef
    assert state.coefficient_factor.R.shape == (p, p)
    assert state.xtwx.shape == (p, p)
    assert state.coefficients.shape == (p,)


@pytest.mark.parametrize("uncertainty", ["none", "fisher", "covariance"])
def test_qr_compact_result_uses_tagged_logdet_and_cpu_fisher_provider(
    monkeypatch: pytest.MonkeyPatch, uncertainty: str
) -> None:
    """The result score/SE retain QR actions instead of rebuilding ``G``'s factor."""
    rng = np.random.default_rng(9927)
    n = 211
    x = rng.normal(size=n)
    z = x + 1e-8 * rng.normal(size=n)
    X = np.column_stack((np.ones(n), x, z))
    weight = rng.uniform(0.5, 1.5, n)
    offset = 0.1 * np.sin(x)
    y = X @ np.array([0.2, 1.5, -0.7]) + offset + 0.05 * rng.normal(size=n)
    source = DataFrameRowSource(
        pd.DataFrame({"x": x, "z": z, "y": y}),
        response="y",
        weights=weight,
        offset=offset,
    )
    family = Gaussian()
    formula = "y ~ x + z"
    prepared = prepare_model(parse_formula(formula), source, family=family)
    state = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.empty(0),
        control=StreamPIRLSControl(batch_rows=13, solver_policy="qr"),
    )
    metadata = PreparedFittingMetadata.from_prepared(prepared, family)
    result = GAMPredictionResult._from_stream_fit(
        stream_state=state,
        prepared=prepared,
        metadata=metadata,
        family=family,
        formula=formula,
        method="REML",
        control=FitControl(execution="stream", batch_rows=13, uncertainty=uncertainty),
    )
    weighted_X = np.sqrt(weight)[:, None] * X
    _Q, R = np.linalg.qr(weighted_X)
    residual = np.sqrt(weight) * (y - offset - X @ np.asarray(state.coefficients))
    scale = np.sum(residual * residual) / (n - 3)
    saturated = -0.5 * np.sum(np.log(2.0 * np.pi * scale / weight))
    expected_score = (
        np.asarray(state.deviance) / (2.0 * scale)
        - saturated
        + np.sum(np.log(np.abs(np.diag(R))))
        - 3.0 / 2.0 * np.log(2.0 * np.pi * scale)
    )
    np.testing.assert_allclose(
        result.score, expected_score, rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    if uncertainty == "none":
        with pytest.raises(RuntimeError, match="uncertainty is unavailable"):
            result.predict({"x": x[:17], "z": z[:17]}, offset=offset[:17], se_fit=True)
        return
    _prediction, se = result.predict(
        {"x": x[:17], "z": z[:17]}, offset=offset[:17], se_fit=True
    )
    expected_se = np.sqrt(scale) * np.linalg.norm(
        np.linalg.solve(R.T, X[:17].T), axis=0
    )
    np.testing.assert_allclose(se, expected_se, rtol=MODERATE.rtol, atol=MODERATE.atol)
    restored = pickle.loads(pickle.dumps(result.to_predictor()))
    _restored_prediction, restored_se = restored.predict(
        {"x": x[:17], "z": z[:17]}, offset=offset[:17], se_fit=True
    )
    np.testing.assert_allclose(
        restored_se, expected_se, rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    assert restored._fisher_qr_factor is not None
    assert (restored.Vp is not None) is (uncertainty == "covariance")
    assert not restored._fisher_qr_factor.R.flags.writeable
    assert not restored._fisher_qr_factor.pivots.flags.writeable
    assert not restored._fisher_qr_factor.keep.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        restored._fisher_qr_factor.R[0, 0] = 0.0
    limited = replace(
        restored, _memory_budget_bytes=6 * 17 * 3 * np.dtype(float).itemsize
    )
    with pytest.raises(MemoryError, match="Prediction workspace"):
        limited.predict({"x": x[:17], "z": z[:17]}, offset=offset[:17], se_fit=True)
    monkeypatch.setattr(
        jax,
        "device_put",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected JAX")
        ),
    )
    _cpu_prediction, cpu_se = restored.predict(
        {"x": x[:17], "z": z[:17]}, offset=offset[:17], se_fit=True
    )
    np.testing.assert_allclose(
        cpu_se, expected_se, rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    if uncertainty == "fisher":
        with pytest.raises(MemoryError, match="Vp materialization"):
            result.materialize_covariance(1)
        with pytest.raises(MemoryError, match="Vp materialization"):
            result.materialize_covariance(3 * 3 * 3 * np.dtype(float).itemsize)
    covariance = result.materialize_covariance(1_000_000)
    expected_covariance = (
        scale * np.linalg.solve(R, np.eye(3)) @ np.linalg.solve(R.T, np.eye(3))
    )
    np.testing.assert_allclose(
        covariance,
        expected_covariance,
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )


def test_qr_covariance_retains_transforms_and_beats_shared_vp_precedence() -> None:
    """QR covariance results retain transformed-factor SEs, not ``X @ Vp``."""
    rng = np.random.default_rng(8077)
    n = 103
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    data = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "y": np.sin(2.0 * x) + 0.3 * z + rng.normal(scale=0.05, size=n),
        }
    )
    family = Gaussian()
    formula = 'y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)'
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(parse_formula(formula), source, family=family)
    state = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.array([np.log(0.2), np.log(0.4)]),
        control=StreamPIRLSControl(batch_rows=13, solver_policy="qr"),
    )
    metadata = PreparedFittingMetadata.from_prepared(prepared, family)

    def compact(uncertainty: str) -> GAMPredictionResult:
        return GAMPredictionResult._from_stream_fit(
            stream_state=state,
            prepared=prepared,
            metadata=metadata,
            family=family,
            formula=formula,
            method="REML",
            control=FitControl(
                execution="stream", batch_rows=13, uncertainty=uncertainty
            ),
        )

    fisher = compact("fisher")
    covariance = compact("covariance")
    newdata = {"x": x[:19], "z": z[:19]}
    _mean_fisher, se_fisher = fisher.predict(newdata, se_fit=True)
    _mean_covariance, se_covariance = covariance.predict(newdata, se_fit=True)
    np.testing.assert_allclose(
        se_covariance, se_fisher, rtol=STRICT.rtol, atol=STRICT.atol
    )
    predictor = covariance.to_predictor()
    assert predictor.Vp is not None
    assert predictor._fisher_qr_factor is not None
    assert predictor._fisher_transforms
    X_p = predictor.predict_matrix(newdata)
    eta = X_p @ predictor.coefficients
    stable = finish_prediction(
        eta,
        X_p,
        predictor.family.link,
        None,
        pred_type="response",
        se_fit=True,
        fisher_qr_factor=predictor._fisher_qr_factor,
        fisher_transforms=predictor._fisher_transforms,
        fisher_scale=predictor._fisher_scale,
    )
    both = finish_prediction(
        eta,
        X_p,
        predictor.family.link,
        np.zeros_like(predictor.Vp),
        pred_type="response",
        se_fit=True,
        fisher_qr_factor=predictor._fisher_qr_factor,
        fisher_transforms=predictor._fisher_transforms,
        fisher_scale=predictor._fisher_scale,
    )
    np.testing.assert_allclose(both[1], stable[1], rtol=STRICT.rtol, atol=STRICT.atol)


def test_failed_line_search_is_not_convergence(monkeypatch: pytest.MonkeyPatch) -> None:
    data, family = _fixture("poisson", n=31)
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=5)'), source, family=family
    )

    def fail_trial(*_args, **_kwargs):
        return jnp.array(jnp.inf), jnp.array(False), 1

    monkeypatch.setattr(stream_execution, "_trial_scan", fail_trial)
    result = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.array([0.0]),
        control=StreamPIRLSControl(batch_rows=8, max_halvings=1),
    )
    assert result.line_search_failed
    assert not result.converged
    assert result.n_iter == 1


def test_final_cholesky_offdiagonal_nonfinite_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default final validity retains the legacy full-lower finiteness guard."""
    data, family = _fixture("gaussian", n=31)
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    original = stream_execution._working_factor_scan
    calls = 0

    def corrupt_final_factor(*args, **kwargs):
        nonlocal calls
        calls += 1
        statistics, proposal, factor, H, batches, qr_state = original(*args, **kwargs)
        if calls == 2:
            assert isinstance(factor, CholeskyCoefficientFactor)
            lower = np.asarray(factor.lower).copy()
            lower[1, 0] = np.nan
            factor = CholeskyCoefficientFactor(jnp.asarray(lower), factor.n_coef)
        return statistics, proposal, factor, H, batches, qr_state

    monkeypatch.setattr(stream_execution, "_working_factor_scan", corrupt_final_factor)
    with pytest.raises(FloatingPointError, match="final state is non-finite"):
        fit_streamed_pirls(
            StreamDesign(prepared, source),
            family,
            np.empty(0),
            control=StreamPIRLSControl(batch_rows=7, max_iter=1),
        )


@pytest.mark.parametrize("solver_policy", ["cholesky", "qr"])
def test_backtracking_is_recorded_before_acceptance(
    monkeypatch: pytest.MonkeyPatch, solver_policy: str
) -> None:
    data, family = _fixture("poisson", n=31)
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=5)'), source, family=family
    )
    original = stream_execution._trial_scan
    calls = 0

    def reject_first_trial(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return jnp.array(jnp.inf), jnp.array(False), 1
        return original(*args, **kwargs)

    monkeypatch.setattr(stream_execution, "_trial_scan", reject_first_trial)
    result = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.array([0.0]),
        control=StreamPIRLSControl(batch_rows=8, solver_policy=solver_policy),
    )
    assert result.converged
    assert result.backtracks >= 1
    assert result.stationarity < 1e-7


def test_preflight_rejects_noncanonical_or_missing_fitting_preparation() -> None:
    data, _ = _fixture("gaussian", n=25)
    source = DataFrameRowSource(data, response="y")
    spec = parse_formula('y ~ s(x, bs="cr", k=5)')
    without_fitting = StreamDesign(prepare_model(spec, source), source)
    with pytest.raises(ValueError, match="prepare_model"):
        fit_streamed_pirls(without_fitting, Gaussian(), np.array([0.0]))

    from jaxgam.links.links import LogLink

    noncanonical = Gaussian(link=LogLink())
    prepared = StreamDesign(prepare_model(spec, source, family=noncanonical), source)
    with pytest.raises(NotImplementedError, match="Fisher and observed"):
        fit_streamed_pirls(prepared, noncanonical, np.array([0.0]))


def test_preflight_rejects_unknown_scale_and_noncanonical_score_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, _ = _fixture("gaussian", n=25)
    data = data.copy()
    data["y"] = np.exp(data["y"])
    source = DataFrameRowSource(data, response="y")
    spec = parse_formula('y ~ s(x, bs="cr", k=5)')
    gamma_stream = StreamDesign(prepare_model(spec, source, family=Gamma()), source)
    source_scans = 0
    original_scan = source.scan

    def count_scan(batch_rows: int):
        nonlocal source_scans
        source_scans += 1
        yield from original_scan(batch_rows)

    monkeypatch.setattr(source, "scan", count_scan)
    with pytest.raises(NotImplementedError, match="not released"):
        fit_streamed_pirls(gamma_stream, Gamma(), np.array([0.0]))
    assert source_scans == 0

    poisson_data, _ = _fixture("poisson", n=25)
    poisson_source = DataFrameRowSource(poisson_data, response="y")
    poisson = Poisson(link=IdentityLink())
    poisson_stream = StreamDesign(
        prepare_model(spec, poisson_source, family=poisson), poisson_source
    )
    with pytest.raises(NotImplementedError, match="Fisher and observed"):
        fit_streamed_pirls(poisson_stream, poisson, np.array([0.0]))


def test_preflight_rejects_family_link_mutated_after_preparation() -> None:
    data, _ = _fixture("poisson", n=25)
    source = DataFrameRowSource(data, response="y")
    family = Poisson()
    prepared = StreamDesign(
        prepare_model(parse_formula('y ~ s(x, bs="cr", k=5)'), source, family=family),
        source,
    )
    family.link = IdentityLink()
    with pytest.raises(RuntimeError, match="static configuration changed"):
        fit_streamed_pirls(prepared, family, np.array([0.0]))


@pytest.mark.parametrize("mutation", ["source", "family", "basis"])
def test_scanner_rejects_runtime_source_basis_and_family_mutation(
    mutation: str,
) -> None:
    data, _ = _fixture("poisson", n=25)
    base = DataFrameRowSource(data, response="y")
    family = Poisson()
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=5)'), base, family=family
    )
    stream: StreamDesign | None = None

    class MutatingSource:
        n_rows = base.n_rows

        def __init__(self) -> None:
            self.changed = False

        def fingerprint(self) -> str:
            if self.changed and mutation == "source":
                return "changed-source"
            return base.fingerprint()

        def scan(self, batch_rows: int):
            for index, batch in enumerate(base.scan(batch_rows)):
                if index == 1:
                    self.changed = True
                    if mutation == "family":
                        family.link = IdentityLink()
                    elif mutation == "basis":
                        assert stream is not None
                        object.__setattr__(
                            stream,
                            "prepared",
                            replace(prepared, basis_fingerprint="changed-basis"),
                        )
                yield batch

    source = MutatingSource()
    stream = StreamDesign(prepared, source)
    with pytest.raises(RuntimeError, match="changed"):
        fit_streamed_pirls(
            stream,
            family,
            np.array([0.0]),
            control=StreamPIRLSControl(batch_rows=5),
        )


def test_unregistered_gaussian_like_family_uses_the_stream_scanner() -> None:
    data, _ = _fixture("gaussian", n=47)
    source = DataFrameRowSource(
        data, response="y", weights=data.w.to_numpy(), offset=data.off.to_numpy()
    )
    family = _UnregisteredGaussianLike()
    spec = parse_formula('y ~ s(x, bs="cr", k=5)')
    prepared = prepare_model(spec, source, family=family)
    streamed = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.array([0.1]),
        control=StreamPIRLSControl(batch_rows=9),
    )
    dense = FittingData.from_setup(
        ModelSetup.build(
            spec,
            data,
            weights=data.w.to_numpy(),
            offset=data.off.to_numpy(),
        ),
        family,
    )
    dense_result = pirls_loop(
        dense.X,
        dense.y,
        dense.beta_init,
        dense.S_lambda(jnp.array([0.1])),
        family,
        dense.wt,
        dense.offset,
    )
    np.testing.assert_allclose(
        np.asarray(streamed.coefficients),
        np.asarray(dense_result.coefficients),
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    assert streamed.converged


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_rows": True},
        {"batch_rows": 1.5},
        {"max_iter": np.inf},
        {"max_halvings": -1},
        {"tol": np.nan},
        {"tol": 0.0},
    ],
)
def test_internal_stream_controls_reject_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"must be|finite|positive"):
        StreamPIRLSControl(**kwargs)


def test_host_rejects_padded_rows_before_design_evaluation() -> None:
    data, family = _fixture("poisson", n=25)
    base = DataFrameRowSource(data, response="y")
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=5)'), base, family=family
    )

    class PaddedSource:
        n_rows = len(data)

        def fingerprint(self) -> str:
            return base.fingerprint()

        def scan(self, _batch_rows: int):
            yield RowBatch(
                columns={"x": np.r_[data.x.to_numpy(), np.nan]},
                y=np.r_[data.y.to_numpy(), np.nan],
                weight=np.r_[np.ones(len(data)), np.nan],
                offset=np.r_[np.zeros(len(data)), np.nan],
                valid=np.r_[np.ones(len(data), dtype=bool), False],
                row_positions=np.arange(len(data) + 1),
            )

    with pytest.raises(NotImplementedError, match="padded"):
        fit_streamed_pirls(
            StreamDesign(prepared, PaddedSource()), family, np.array([0.0])
        )


def test_preflight_rejects_prepared_rank_deficit() -> None:
    data, family = _fixture("gaussian", n=25)
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=5)'), source, family=family
    )
    assert prepared.fitting is not None
    rank_deficient = replace(
        prepared, fitting=replace(prepared.fitting, unpenalized_rank_deficit=1)
    )
    with pytest.raises(np.linalg.LinAlgError, match="full rank"):
        fit_streamed_pirls(
            StreamDesign(rank_deficient, source), family, np.array([0.0])
        )


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([np.inf])])
def test_nonfinite_smoothing_parameter_is_rejected(bad: np.ndarray) -> None:
    data, family = _fixture("poisson", n=25)
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=5)'), source, family=family
    )
    with pytest.raises(ValueError, match="log_lambda"):
        fit_streamed_pirls(StreamDesign(prepared, source), family, bad)


def test_nonfinite_or_domain_invalid_warm_start_is_rejected() -> None:
    data, family = _fixture("poisson", n=25)
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(
        parse_formula('y ~ s(x, bs="cr", k=5)'), source, family=family
    )
    with pytest.raises(ValueError, match="beta_init"):
        fit_streamed_pirls(
            StreamDesign(prepared, source),
            family,
            np.array([0.0]),
            beta_init=np.array([np.nan] * prepared.n_coef),
        )
    with pytest.raises(ValueError, match="family domain"):
        fit_streamed_pirls(
            StreamDesign(prepared, source),
            family,
            np.array([0.0]),
            beta_init=np.full(prepared.n_coef, 1e4),
        )


def test_zero_and_extreme_real_weights_match_dense_clipping() -> None:
    data, family = _fixture("gaussian", n=49)
    data.loc[0, "w"] = 0.0
    data.loc[1, "w"] = 1e-12
    # Large enough to exercise conditioning/weight clipping without making
    # dense's legacy always-on Cholesky jitter the dominant perturbation.
    data.loc[2, "w"] = 1e5
    spec = parse_formula('y ~ s(x, bs="cr", k=6)')
    source = DataFrameRowSource(data, response="y", weights=data.w.to_numpy())
    prepared = prepare_model(spec, source, family=family)
    streamed = fit_streamed_pirls(
        StreamDesign(prepared, source), family, np.array([0.0])
    )
    dense = FittingData.from_setup(
        ModelSetup.build(spec, data, weights=data.w.to_numpy()), family
    )
    dense_result = pirls_loop(
        dense.X,
        dense.y,
        dense.beta_init,
        dense.S_lambda(jnp.array([0.0])),
        family,
        dense.wt,
        dense.offset,
    )
    np.testing.assert_allclose(
        np.asarray(streamed.coefficients),
        np.asarray(dense_result.coefficients),
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )


def test_stream_fit_state_contains_no_observation_aligned_fields() -> None:
    names = {field.name for field in fields(StreamFitState)}
    assert not {"X", "y", "eta", "mu", "working_weights", "offset"} & names


@pytest.mark.parametrize(
    ("formula", "log_lambda"),
    [
        ('y ~ s(x, bs="cr", k=6)', np.array([np.log(0.2)])),
        ("y ~ x", np.array([])),
    ],
)
def test_gaussian_score_scale_uses_positive_weight_count_and_no_penalty_case(
    formula: str, log_lambda: np.ndarray
) -> None:
    """Fletcher reporting scale is distinct from fixed-sp REML score scale."""
    data, family = _fixture("gaussian", n=49)
    weights = data.w.to_numpy().copy()
    weights[::9] = 0.0
    source = DataFrameRowSource(data, response="y", weights=weights)
    prepared = prepare_model(parse_formula(formula), source, family=family)
    result = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        log_lambda,
        control=StreamPIRLSControl(batch_rows=7),
    )
    assert prepared.fitting is not None
    if len(log_lambda) == 0:
        expected_score_scale = result.scale
    else:
        score_denominator = np.count_nonzero(weights > 0.0) - (
            prepared.n_coef - prepared.fitting.total_penalty_rank
        )
        expected_score_scale = result.penalized_deviance / score_denominator
    np.testing.assert_allclose(
        np.asarray(result.score_scale),
        np.asarray(expected_score_scale),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    if len(log_lambda):
        assert not np.isclose(float(result.scale), float(result.score_scale))


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
@pytest.mark.parametrize("family_name", ["gaussian", "poisson", "binomial"])
@pytest.mark.parametrize("solver_policy", ["cholesky", "qr"])
def test_streamed_fixed_sp_matches_pinned_r_with_basis_held_fixed(
    family_name: str, solver_policy: str
) -> None:
    """Use R's fitted sp as a fixed input to both frozen-basis solvers."""
    from tests.r_bridge import RBridge

    data, family = _fixture(family_name, n=71)
    data = data[["x", "y"]]
    formula = 'y ~ s(x, bs="cr", k=6)'
    r_result = RBridge().fit_gam(formula, data, family=family_name)
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(parse_formula(formula), source, family=family)
    result = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.log(r_result["smoothing_params"]),
        control=StreamPIRLSControl(batch_rows=9, solver_policy=solver_policy),
    )
    assert prepared.fitting is not None
    public = penalty_ops.transform_coefficients(
        _to_jax_structure(prepared.fitting.penalty_structure, None),
        result.coefficients,
    )
    np.testing.assert_allclose(
        np.asarray(public),
        r_result["coefficients"],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
@pytest.mark.parametrize("solver_policy", ["cholesky", "qr"])
def test_strong_smoothing_scale_and_saturated_likelihood_match_dense_and_r(
    solver_policy: str,
) -> None:
    """Unknown Gaussian scale uses n - Fisher EDF, not n - coefficient count."""
    from tests.r_bridge import RBridge

    data, family = _fixture("gaussian", n=71)
    data = data[["x", "y"]]
    formula = 'y ~ s(x, bs="cr", k=6)'
    r_result = RBridge().fit_gam(formula, data, family="gaussian")
    source = DataFrameRowSource(data, response="y")
    prepared = prepare_model(parse_formula(formula), source, family=family)
    result = fit_streamed_pirls(
        StreamDesign(prepared, source),
        family,
        np.log(r_result["smoothing_params"]),
        control=StreamPIRLSControl(batch_rows=9, solver_policy=solver_policy),
    )
    dense = FittingData.from_setup(
        ModelSetup.build(parse_formula(formula), data), family
    )
    dense_result = pirls_loop(
        dense.X,
        dense.y,
        dense.beta_init,
        dense.S_lambda(jnp.log(jnp.asarray(r_result["smoothing_params"]))),
        family,
        dense.wt,
        dense.offset,
    )
    dense_edf = estimate_edf(dense_result.XtWX_fisher, dense_result.L_fisher)
    dense_scale = dense_result.deviance / (len(data) - dense_edf)
    dense_score_scale = (
        dense_result.deviance
        + dense_result.coefficients
        @ dense.S_lambda(jnp.log(jnp.asarray(r_result["smoothing_params"])))
        @ dense_result.coefficients
    ) / (np.count_nonzero(np.asarray(dense.wt) > 0.0) - dense.total_penalty_null_dim)
    dense_saturated = family.saturated_loglik(
        dense.y, dense.wt, dense_score_scale, max_y=dense.max_y
    )
    assert float(r_result["edf_total"]) < dense.n_coef - 1
    np.testing.assert_allclose(
        np.asarray(result.edf),
        np.asarray(dense_edf),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.scale),
        np.asarray(dense_scale),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.scale),
        r_result["scale"],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.score_scale),
        dense_score_scale,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.saturated_loglik),
        np.asarray(dense_saturated),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.deviance),
        r_result["deviance"],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
