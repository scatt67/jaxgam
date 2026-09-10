"""Exact known-scale streamed REML adjoint gates."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import ArrayRowSource
from jaxgam.execution.reml import (
    StreamREMLControl,
    _AcceptedTrialObjective,
    _projected_gradient,
    evaluate_stream_reml,
    optimize_stream_reml,
)
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Binomial, Gaussian, Poisson
from jaxgam.fitting.data import FittingData, PreparedFittingMetadata
from jaxgam.fitting.newton import NewtonOptimizer, newton_optimize
from jaxgam.fitting.stream_reml import (
    batch_is_adjoint_interior,
    batch_statistics_beta_vjp,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from tests.helpers import _AssertCollector, r_available
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


def _optimizer_problem(family_name: str):
    """Well-conditioned weighted/offset data with a non-flat REML optimum."""
    rng = np.random.default_rng(7841)
    n = 227
    x, z = rng.uniform(-1.0, 1.0, (2, n))
    weight = rng.uniform(0.6, 1.4, n)
    offset = 0.07 * np.sin(2.0 * z)
    family = Poisson() if family_name == "poisson" else Binomial()
    eta = 0.3 + 1.5 * np.sin(3.0 * x) + 1.2 * np.cos(3.0 * z) + offset
    mean = np.asarray(family.link.inverse(eta))
    y = rng.poisson(mean) if family_name == "poisson" else rng.binomial(1, mean)
    formula = "y ~ s(x, bs='cr', k=6) + s(z, bs='cr', k=5)"
    spec = parse_formula(formula)
    data = pd.DataFrame({"x": x, "z": z, "y": y, "w": weight, "off": offset})
    source = ArrayRowSource({"x": x, "z": z}, y=y, weights=weight, offset=offset)
    prepared = prepare_model(spec, source, family=family)
    dense = FittingData.from_setup(
        ModelSetup.build(spec, {"x": x, "z": z, "y": y}, weights=weight, offset=offset),
        family,
    )
    return family, source, prepared, dense, data, formula, offset


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


def test_stream_reml_full_exact_poisson_fit_has_finite_refit_gradient() -> None:
    """The complete provider remains differentiable when every deviance is zero."""
    rng = np.random.default_rng(72038)
    n = 113
    x, z = rng.uniform(-1.0, 1.0, (2, n))
    family = Poisson()
    formula = parse_formula("y ~ s(x, bs='cr', k=6) + s(z, bs='cr', k=5)")
    source = ArrayRowSource({"x": x, "z": z}, y=np.ones(n))
    stream = StreamDesign(prepare_model(formula, source, family=family), source)
    rho = jnp.array([0.3, -0.5])
    control = StreamPIRLSControl(batch_rows=19, tol=1e-9)
    trial = evaluate_stream_reml(stream, family, rho, control=control)
    delta = 1e-4
    finite_difference = np.array(
        [
            (
                float(
                    evaluate_stream_reml(
                        stream,
                        family,
                        rho.at[index].add(delta),
                        control=control,
                        warm_start=trial,
                    ).score
                )
                - float(
                    evaluate_stream_reml(
                        stream,
                        family,
                        rho.at[index].add(-delta),
                        control=control,
                        warm_start=trial,
                    ).score
                )
            )
            / (2.0 * delta)
            for index in range(len(rho))
        ]
    )
    collector = _AssertCollector()
    collector.check(
        "zero deviance",
        lambda: np.testing.assert_allclose(
            trial.fit_state.deviance, 0.0, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "finite score and gradient",
        lambda: np.testing.assert_equal(
            np.all(np.isfinite(np.asarray([trial.score, *trial.gradient]))), True
        ),
    )
    collector.check(
        "reconverged central difference",
        lambda: np.testing.assert_allclose(
            trial.gradient,
            finite_difference,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any("exact Poisson streamed REML provider")


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


def test_stream_reml_lbfgsb_uses_only_callback_accepted_warm_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected line-search candidate cannot become a later warm start."""
    import jaxgam.execution.reml as stream_reml_execution

    calls: list[tuple[np.ndarray, object | None]] = []

    def fake_trial(
        _stream: object,
        _family: object,
        rho: np.ndarray,
        *,
        warm_start: object | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        rho_array = np.asarray(rho, dtype=np.float64)
        calls.append((rho_array, warm_start))
        return SimpleNamespace(
            rho=rho_array,
            score=jnp.sum(jnp.asarray(rho_array) ** 2),
            gradient=2.0 * jnp.asarray(rho_array),
            source_scans=3,
            batches_scanned=9,
            source_fingerprint="test-source",
        )

    monkeypatch.setattr(stream_reml_execution, "evaluate_stream_reml", fake_trial)
    objective = _AcceptedTrialObjective(
        SimpleNamespace(source=SimpleNamespace(fingerprint=lambda: "test-source")),
        object(),
        np.array([0.0]),
        None,
        StreamREMLControl(history_size=2),
        None,
    )
    rejected = objective(np.array([2.0]))
    assert rejected[0] == 4.0
    candidate = objective.candidate
    assert candidate is not None
    with pytest.raises(RuntimeError, match="matching exact"):
        objective.callback(SimpleNamespace(x=np.array([1.0])))

    objective.callback(SimpleNamespace(x=np.array([2.0])))
    objective(np.array([3.0]))
    assert calls[-1][1] is candidate
    assert len(objective.accepted_score_history) == 2


def test_stream_reml_lbfgsb_reports_bounded_nonconvergence_and_kkt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine iteration stop returns diagnostics, while bounds use KKT signs."""
    import jaxgam.execution.reml as stream_reml_execution

    initial = SimpleNamespace(
        rho=np.array([-40.0]),
        score=jnp.array(1.0),
        gradient=jnp.array([2.0]),
        source_scans=3,
        batches_scanned=9,
        source_fingerprint="test-source",
    )
    monkeypatch.setattr(
        stream_reml_execution,
        "evaluate_stream_reml",
        lambda *_args, **_kwargs: initial,
    )
    monkeypatch.setattr(
        stream_reml_execution,
        "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(
            x=np.array([-40.0]),
            success=False,
            message="iteration limit",
            status=1,
            nit=3,
        ),
    )
    fitting = SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1))
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "test-source"),
        prepared=SimpleNamespace(fitting=fitting),
    )
    result = optimize_stream_reml(
        stream,
        object(),
        np.array([-41.0]),
        control=StreamREMLControl(max_iter=3, history_size=1),
    )
    assert not result.converged
    assert result.message == "iteration limit"
    assert result.projected_gradient_inf == 0.0
    assert result.n_evaluations == 1
    assert result.n_accepted == 0
    assert result.cumulative_source_scans == 3
    assert result.cumulative_batches_scanned == 9
    np.testing.assert_array_equal(
        _projected_gradient(
            np.array([-40.0, 40.0, 0.0]),
            np.array([2.0, -3.0, 4.0]),
            lower=-40.0,
            upper=40.0,
        ),
        np.array([0.0, 0.0, 4.0]),
    )


def test_stream_reml_lbfgsb_rechecks_cached_source_and_ftol_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jaxgam.execution.reml as stream_reml_execution

    trial = SimpleNamespace(
        rho=np.array([0.0]),
        score=jnp.array(1.0),
        gradient=jnp.array([2.0]),
        source_scans=3,
        batches_scanned=9,
        source_fingerprint="prepared-source",
    )
    monkeypatch.setattr(
        stream_reml_execution,
        "evaluate_stream_reml",
        lambda *_args, **_kwargs: trial,
    )
    changed_stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "changed-source"),
        prepared=SimpleNamespace(source_fingerprint="prepared-source"),
    )
    objective = _AcceptedTrialObjective(
        changed_stream,
        object(),
        np.array([0.0]),
        None,
        StreamREMLControl(),
        None,
    )
    with pytest.raises(RuntimeError, match="RowSource changed"):
        objective(np.array([0.0]))

    fitting = SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1))
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "prepared-source"),
        prepared=SimpleNamespace(fitting=fitting),
    )
    monkeypatch.setattr(
        stream_reml_execution,
        "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(
            x=np.array([0.0]),
            success=True,
            message="relative reduction",
            status=0,
            nit=1,
        ),
    )
    result = optimize_stream_reml(
        stream,
        object(),
        np.array([0.0]),
        control=StreamREMLControl(gtol=1e-8),
    )
    assert not result.converged
    assert "projected gradient" in result.message


def test_stream_reml_lbfgsb_rejects_zero_penalty_and_callback_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jaxgam.execution.reml as stream_reml_execution

    zero_fitting = SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=0))
    zero_stream = SimpleNamespace(prepared=SimpleNamespace(fitting=zero_fitting))
    with pytest.raises(NotImplementedError, match="at least one"):
        optimize_stream_reml(zero_stream, object(), np.zeros(0))

    initial = SimpleNamespace(
        rho=np.array([0.0]),
        score=jnp.array(1.0),
        gradient=jnp.array([1.0]),
        source_scans=3,
        batches_scanned=9,
        source_fingerprint="test-source",
    )
    monkeypatch.setattr(
        stream_reml_execution,
        "evaluate_stream_reml",
        lambda *_args, **_kwargs: initial,
    )
    monkeypatch.setattr(
        stream_reml_execution,
        "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(
            x=np.array([1.0]), success=False, message="bad", status=2, nit=0
        ),
    )
    fitting = SimpleNamespace(penalty_structure=SimpleNamespace(n_penalties=1))
    stream = SimpleNamespace(
        source=SimpleNamespace(fingerprint=lambda: "test-source"),
        prepared=SimpleNamespace(fitting=fitting),
    )
    with pytest.raises(RuntimeError, match="matching accepted"):
        optimize_stream_reml(stream, object(), np.array([0.0]))


def test_stream_reml_lbfgsb_real_trials_decrease_from_callback_accepts() -> None:
    family, source, prepared, _metadata, _dense = _poisson_problem()
    result = optimize_stream_reml(
        StreamDesign(prepared, source),
        family,
        np.array([-1.0, 0.5]),
        pirls_control=StreamPIRLSControl(batch_rows=19, tol=1e-9),
        control=StreamREMLControl(max_iter=4, maxfun=30, gtol=1e-5, history_size=4),
    )
    history = np.asarray(result.accepted_score_history)
    assert np.all(np.diff(history) <= 0.0)
    assert np.array_equal(
        np.asarray(result.trial.rho), np.asarray(result.trial.fit_state.log_lambda)
    )
    assert result.n_evaluations >= 1
    assert result.n_accepted == result.n_iter


def test_stream_reml_lbfgsb_null_smooth_fails_closed_at_high_conditioning() -> None:
    """Do not report a residue-driven large-rho optimum as a valid fit."""
    rng = np.random.default_rng(72)
    x, z = rng.uniform(-1.0, 1.0, (2, 113))
    family = Poisson()
    source = ArrayRowSource({"x": x, "z": z}, y=np.ones(113))
    stream = StreamDesign(
        prepare_model(
            parse_formula("y ~ s(x, bs='cr', k=6) + s(z, bs='cr', k=5)"),
            source,
            family=family,
        ),
        source,
    )
    with pytest.raises(np.linalg.LinAlgError, match="observed-adjoint"):
        optimize_stream_reml(
            stream,
            family,
            np.zeros(2),
            pirls_control=StreamPIRLSControl(batch_rows=19),
            control=StreamREMLControl(max_iter=30, maxfun=100, gtol=1e-7),
        )


@pytest.mark.parametrize("family_name", ["poisson", "binomial"])
@pytest.mark.parametrize("batch_rows", [17, 256])
def test_stream_reml_lbfgsb_converges_to_dense_reml_optimum(
    family_name: str, batch_rows: int
) -> None:
    """Accepted streamed trials converge to the dense exact REML solution."""
    family, source, prepared, dense, _data, _formula, offset = _optimizer_problem(
        family_name
    )
    dense_result = newton_optimize(dense, tol=1e-9)
    result = optimize_stream_reml(
        StreamDesign(prepared, source),
        family,
        dense.log_lambda_init,
        pirls_control=StreamPIRLSControl(batch_rows=batch_rows),
        control=StreamREMLControl(gtol=1e-7, ftol=1e-14, history_size=5),
    )
    fitted = np.asarray(
        family.link.inverse(dense.X @ result.trial.fit_state.coefficients + offset)
    )
    collector = _AssertCollector()
    collector.check(
        "converged", lambda: np.testing.assert_equal(result.converged, True)
    )
    collector.check(
        "accepted scores decrease",
        lambda: np.testing.assert_array_less(
            np.diff(result.accepted_score_history), 1e-10
        ),
    )
    collector.check(
        "score",
        lambda: np.testing.assert_allclose(
            result.trial.score,
            dense_result.score,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "fitted means",
        lambda: np.testing.assert_allclose(
            fitted,
            dense_result.pirls_result.mu,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any(f"{family_name} B={batch_rows} streamed L-BFGS-B")
    assert result.projected_gradient_inf <= 1e-7
    assert result.trial.fit_state.stationarity < 1e-9
    assert result.n_accepted == result.n_iter
    assert len(result.accepted_score_history) <= 5
    assert result.cumulative_source_scans >= result.trial.source_scans


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
@pytest.mark.parametrize("family_name", ["poisson", "binomial"])
def test_stream_reml_lbfgsb_matches_pinned_r_reml(
    family_name: str,
) -> None:
    """The internal optimized route matches pinned R with explicit w/offset."""

    family, source, prepared, dense, data, formula, offset = _optimizer_problem(
        family_name
    )
    result = optimize_stream_reml(
        StreamDesign(prepared, source),
        family,
        dense.log_lambda_init,
        pirls_control=StreamPIRLSControl(batch_rows=17),
        control=StreamREMLControl(gtol=1e-7, ftol=1e-14),
    )

    def r_vector(values: np.ndarray) -> str:
        return "c(" + ",".join(repr(float(value)) for value in values) + ")"

    r_code = "\n".join(
        [
            "library(mgcv)",
            'stopifnot(getRversion() == "4.5.2", packageVersion("mgcv") == "1.9-3")',
            (
                "d <- data.frame("
                f"x={r_vector(data.x.to_numpy())},"
                f"z={r_vector(data.z.to_numpy())},"
                f"y={r_vector(data.y.to_numpy())},"
                f"w={r_vector(data.w.to_numpy())},"
                f"off={r_vector(data.off.to_numpy())})"
            ),
            (
                f"g <- gam({formula}, data=d, weights=w, offset=off, "
                f'family={family_name}(), method="REML", '
                "control=gam.control(newton=list(conv.tol=1e-8)))"
            ),
            "write.table(c(g$gcv.ubre, g$deviance, fitted(g)), "
            "row.names=FALSE, col.names=FALSE)",
        ]
    )
    r_process = subprocess.run(
        ["Rscript", "-"],
        input=r_code,
        capture_output=True,
        check=True,
        text=True,
    )
    r_values = np.fromstring(r_process.stdout, sep="\n")
    assert r_values.shape == (len(data) + 2,)
    fitted = np.asarray(
        family.link.inverse(dense.X @ result.trial.fit_state.coefficients + offset)
    )
    collector = _AssertCollector()
    collector.check(
        "converged", lambda: np.testing.assert_equal(result.converged, True)
    )
    collector.check(
        "REML score",
        lambda: np.testing.assert_allclose(
            result.trial.score,
            r_values[0],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "deviance",
        lambda: np.testing.assert_allclose(
            result.trial.fit_state.deviance,
            r_values[1],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "fitted means",
        lambda: np.testing.assert_allclose(
            fitted,
            r_values[2:],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any(f"{family_name} streamed L-BFGS-B pinned R REML")
