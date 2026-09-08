"""Exact known-scale streamed REML adjoint gates."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.data.source import ArrayRowSource
from jaxgam.execution.reml import evaluate_stream_reml
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Binomial, Gaussian, Poisson
from jaxgam.fitting.data import FittingData, PreparedFittingMetadata
from jaxgam.fitting.newton import NewtonOptimizer
from jaxgam.fitting.stream_reml import (
    batch_is_adjoint_interior,
    batch_statistics_beta_vjp,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from tests.helpers import _AssertCollector
from tests.tolerances import MODERATE, STRICT


def _poisson_problem():
    rng = np.random.default_rng(5926)
    n = 149
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    weight = rng.uniform(0.5, 1.5, n)
    offset = 0.1 * np.cos(3.0 * z)
    y = rng.poisson(np.exp(0.2 + 0.3 * np.sin(3.0 * x) - 0.2 * z + offset))
    formula = "y ~ s(x, bs='cr', k=6) + s(z, bs='cr', k=5)"
    family = Poisson()
    source = ArrayRowSource({"x": x, "z": z}, y=y, weights=weight, offset=offset)
    prepared = prepare_model(parse_formula(formula), source, family=family)
    metadata = PreparedFittingMetadata.from_prepared(prepared, family)
    dense = FittingData.from_setup(
        ModelSetup.build(
            parse_formula(formula),
            {"x": x, "z": z, "y": y},
            weights=weight,
            offset=offset,
        ),
        family,
    )
    return family, source, prepared, metadata, dense


def _binomial_problem():
    rng = np.random.default_rng(112)
    n = 91
    x = rng.uniform(-0.8, 0.9, n)
    z = rng.uniform(-0.7, 0.8, n)
    offset = 0.08 * np.sin(2.0 * z)
    probability = 1.0 / (1.0 + np.exp(-(0.1 + 0.2 * x - 0.15 * z + offset)))
    y = rng.binomial(1, probability)
    formula = "y ~ s(x, bs='cr', k=6) + s(z, bs='cr', k=5)"
    family = Binomial()
    source = ArrayRowSource({"x": x, "z": z}, y=y, offset=offset)
    prepared = prepare_model(parse_formula(formula), source, family=family)
    metadata = PreparedFittingMetadata.from_prepared(prepared, family)
    dense = FittingData.from_setup(
        ModelSetup.build(
            parse_formula(formula), {"x": x, "z": z, "y": y}, offset=offset
        ),
        family,
    )
    return family, source, prepared, metadata, dense


def test_stream_reml_adjoint_matches_dense_custom_jvp_and_refits() -> None:
    """Three rho trials and batch sizes prove the source VJP identity."""
    family, source, prepared, _metadata, dense = _poisson_problem()
    oracle = NewtonOptimizer(dense)
    stream = StreamDesign(prepared, source)
    collector = _AssertCollector()
    for rho in (
        jnp.array([-2.0, 1.0]),
        jnp.array([0.3, -0.5]),
        jnp.array([3.0, 4.0]),
    ):
        dense_fit, dense_score = oracle._fit_and_score(rho, dense.beta_init)
        dense_gradient, _ = oracle._diff_grad_hess(rho, dense_fit.coefficients)
        delta = 1e-4
        finite_difference = np.array(
            [
                (
                    float(
                        oracle._fit_and_score(
                            rho.at[index].add(delta), dense_fit.coefficients
                        )[1]
                    )
                    - float(
                        oracle._fit_and_score(
                            rho.at[index].add(-delta), dense_fit.coefficients
                        )[1]
                    )
                )
                / (2.0 * delta)
                for index in range(len(rho))
            ]
        )
        for batch_rows in (1, 13, 64):
            trial = evaluate_stream_reml(
                stream,
                family,
                rho,
                control=StreamPIRLSControl(batch_rows=batch_rows, tol=1e-9),
            )
            prefix = f"rho={np.asarray(rho).tolist()}, B={batch_rows}"
            collector.check(
                f"{prefix}: score",
                lambda a=trial.score, e=dense_score: np.testing.assert_allclose(
                    a, e, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
            collector.check(
                f"{prefix}: custom_jvp",
                lambda a=trial.gradient, e=dense_gradient: np.testing.assert_allclose(
                    a, e, rtol=MODERATE.rtol, atol=MODERATE.atol
                ),
            )
            collector.check(
                f"{prefix}: refit finite difference",
                lambda a=trial.gradient, e=finite_difference: (
                    np.testing.assert_allclose(
                        a, e, rtol=MODERATE.rtol, atol=MODERATE.atol
                    )
                ),
            )
            assert trial.fit_state.converged
            assert trial.fit_state.stationarity < 1e-9
            assert np.array_equal(np.asarray(trial.rho), np.asarray(rho))
    collector.raise_if_any("streamed REML adjoint proof")


def test_stream_reml_trial_warm_start_reconverges_and_jits() -> None:
    family, source, prepared, _metadata, _dense = _poisson_problem()
    stream = StreamDesign(prepared, source)
    first = evaluate_stream_reml(
        stream,
        family,
        jnp.array([-0.4, 0.2]),
        control=StreamPIRLSControl(batch_rows=17, tol=1e-9),
    )
    second = evaluate_stream_reml(
        stream,
        family,
        jnp.array([0.1, -0.3]),
        control=StreamPIRLSControl(batch_rows=17, tol=1e-9),
        warm_start=first,
    )
    assert second.fit_state.converged
    assert not np.array_equal(np.asarray(first.rho), np.asarray(second.rho))
    assert not np.array_equal(
        np.asarray(first.fit_state.coefficients),
        np.asarray(second.fit_state.coefficients),
    )
    assert batch_is_adjoint_interior._cache_size() >= 1


def test_stream_reml_binomial_gradient_matches_dense_custom_jvp() -> None:
    family, source, prepared, _metadata, dense = _binomial_problem()
    rho = jnp.array([-0.3, 0.4])
    trial = evaluate_stream_reml(
        StreamDesign(prepared, source),
        family,
        rho,
        control=StreamPIRLSControl(batch_rows=11, tol=1e-9),
    )
    oracle = NewtonOptimizer(dense)
    dense_fit, dense_score = oracle._fit_and_score(rho, dense.beta_init)
    dense_gradient, _ = oracle._diff_grad_hess(rho, dense_fit.coefficients)
    np.testing.assert_allclose(
        trial.score, dense_score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        trial.gradient,
        dense_gradient,
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )


def test_stream_reml_preflight_rejects_unsupported_derivative_regimes() -> None:
    family, source, prepared, _metadata, _dense = _poisson_problem()
    stream = StreamDesign(prepared, source)
    with pytest.raises(ValueError, match="finite"):
        evaluate_stream_reml(stream, family, np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        evaluate_stream_reml(stream, family, np.array([0.0]))
    assert prepared.fitting is not None
    rank_deficient = replace(
        prepared, fitting=replace(prepared.fitting, unpenalized_rank_deficit=1)
    )
    with pytest.raises(np.linalg.LinAlgError, match="full-rank"):
        evaluate_stream_reml(
            StreamDesign(rank_deficient, source), family, np.array([0.0, 0.0])
        )
    with pytest.raises(NotImplementedError, match=r"Poisson log.*Binomial logit"):
        evaluate_stream_reml(stream, Gaussian(), np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="different family or link"):
        evaluate_stream_reml(stream, Binomial(), np.array([0.0, 0.0]))


def test_adjoint_interior_rejects_clipped_weight() -> None:
    family = Poisson()
    X = jnp.array([[1.0]])
    assert not bool(
        np.asarray(
            batch_is_adjoint_interior(
                jnp.array([30.0]), X, jnp.array([1.0]), jnp.array([0.0]), family
            )
        )
    )


def test_poisson_exact_intercept_fit_has_finite_direct_deviance_vjp() -> None:
    """Direct deviance avoids sqrt-residual AD singularities at y == mu."""
    family = Poisson()
    X = jnp.ones((8, 1))
    beta_vjp = batch_statistics_beta_vjp(
        jnp.array([0.0]),
        X,
        jnp.ones(8),
        jnp.ones(8),
        jnp.zeros(8),
        jnp.zeros((1, 1)),
        jnp.array(1.0),
        family,
    )
    assert np.all(np.isfinite(np.asarray(beta_vjp)))
    np.testing.assert_allclose(
        beta_vjp, jnp.zeros(1), rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_stream_reml_provider_reports_source_and_inner_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family, source, prepared, _metadata, _dense = _poisson_problem()
    stream = StreamDesign(prepared, source)
    control = StreamPIRLSControl(batch_rows=13, tol=1e-9)
    valid = evaluate_stream_reml(stream, family, np.array([0.1, -0.2]), control=control)

    changed_source = ArrayRowSource(
        {"x": np.zeros(source.n_rows), "z": np.zeros(source.n_rows)},
        y=np.zeros(source.n_rows),
    )
    with pytest.raises(RuntimeError, match="changed after preparation"):
        evaluate_stream_reml(
            StreamDesign(prepared, changed_source),
            family,
            np.array([0.1, -0.2]),
        )

    bad_warm = replace(valid, fit_state=replace(valid.fit_state, converged=False))
    with pytest.raises(ValueError, match="Warm-start"):
        evaluate_stream_reml(stream, family, np.array([0.2, -0.1]), warm_start=bad_warm)
    mismatched_warm = replace(valid, basis_fingerprint="different-basis")
    with pytest.raises(ValueError, match="Warm-start"):
        evaluate_stream_reml(
            stream, family, np.array([0.2, -0.1]), warm_start=mismatched_warm
        )

    import jaxgam.execution.reml as stream_reml_execution

    with monkeypatch.context() as scoped:
        scoped.setattr(
            stream_reml_execution,
            "batch_is_adjoint_interior",
            lambda *_args, **_kwargs: jnp.array(False),
        )
        with pytest.raises(NotImplementedError, match="working-weight clipping"):
            evaluate_stream_reml(stream, family, np.array([0.1, -0.2]), control=control)

    failed = replace(valid.fit_state, converged=False)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            stream_reml_execution,
            "fit_streamed_pirls",
            lambda *_args, **_kwargs: failed,
        )
        with pytest.raises(RuntimeError, match="inner PIRLS did not converge"):
            evaluate_stream_reml(stream, family, np.array([0.1, -0.2]), control=control)
