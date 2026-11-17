"""Newton integration gates for the frozen Gaussian QR provider."""

from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

import jaxgam.execution.gaussian_optimizer as gaussian_optimizer_module
from jaxgam.api import GAM, _fit_fixed_sp
from jaxgam.control import FitControl
from jaxgam.execution.gaussian import prepare_frozen_gaussian_qr
from jaxgam.execution.gaussian_optimizer import (
    DenseGaussianQROuterProvider,
    GaussianQRIneligible,
    build_dense_gaussian_qr_provider,
    optimize_gaussian_qr_lbfgs,
    use_gaussian_qr_lbfgs,
)
from jaxgam.execution.reml import StreamREMLControl
from jaxgam.families.standard import Gaussian
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.newton import NewtonOptimizer, newton_optimize
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.rank_coordinates import IdentifiableCoordinateMap
from tests.tolerances import STRICT


def _problem_inputs():
    rng = np.random.default_rng(9507)
    n = 137
    x, z = rng.uniform(-1.0, 1.0, (2, n))
    weight = rng.uniform(0.4, 1.7, n)
    offset = 0.08 * np.sin(2.0 * z)
    y = (
        0.3
        + 1.2 * np.sin(3.0 * x)
        - 0.7 * np.cos(2.0 * z)
        + offset
        + rng.normal(scale=0.25, size=n)
    )
    formula = "y ~ s(x, bs='cr', k=7) + s(z, bs='cr', k=6)"
    data = {"x": x, "z": z, "y": y}
    return formula, data, weight, offset


def _problem() -> FittingData:
    formula, data, weight, offset = _problem_inputs()
    spec = parse_formula(formula)
    setup = ModelSetup.build(
        spec,
        data,
        weights=weight,
        offset=offset,
    )
    return FittingData.from_setup(setup, Gaussian())


def _large_m_inputs():
    rng = np.random.default_rng(731)
    n_levels = 51
    repeats = 4
    x = np.tile(np.linspace(-1.0, 1.0, repeats), n_levels)
    group = pd.Categorical(
        np.repeat([f"g{index:02d}" for index in range(n_levels)], repeats)
    )
    y = np.sin(2.0 * x) + np.repeat(rng.normal(scale=0.2, size=n_levels), repeats)
    y += rng.normal(scale=0.05, size=len(x))
    data = pd.DataFrame({"x": x, "group": group, "y": y})
    formula = "y ~ group + s(x, by=group, bs='cr', k=3)"
    return formula, data


def _large_m_problem() -> FittingData:
    formula, data = _large_m_inputs()
    setup = ModelSetup.build(parse_formula(formula), data)
    fitting = FittingData.from_setup(setup, Gaussian())
    assert fitting.n_penalties == 51
    assert fitting.rank_deficit == 0
    return fitting


def _jaxpr_shapes(closed_jaxpr) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = [
        np.shape(value) for value in jax.tree.leaves(closed_jaxpr.consts)
    ]

    def visit(jaxpr) -> None:
        for variable in (
            *jaxpr.constvars,
            *jaxpr.invars,
            *jaxpr.outvars,
            *(variable for equation in jaxpr.eqns for variable in equation.invars),
            *(variable for equation in jaxpr.eqns for variable in equation.outvars),
        ):
            aval = getattr(variable, "aval", None)
            if aval is not None and hasattr(aval, "shape"):
                shapes.append(tuple(aval.shape))
        for equation in jaxpr.eqns:
            for value in jax.tree.leaves(equation.params):
                if hasattr(value, "jaxpr") and hasattr(value, "consts"):
                    shapes.extend(np.shape(constant) for constant in value.consts)
                    visit(value.jaxpr)
                elif hasattr(value, "eqns") and hasattr(value, "invars"):
                    visit(value)

    visit(closed_jaxpr.jaxpr)
    return shapes


def _assert_no_observation_dimension(closed_jaxpr, n_obs: int) -> None:
    shapes = _jaxpr_shapes(closed_jaxpr)
    assert shapes
    assert all(n_obs not in shape for shape in shapes)


def test_provider_score_derivatives_and_optimizer_match_dense_strict() -> None:
    fitting = _problem()
    provider = build_dense_gaussian_qr_provider(fitting)
    params = provider.initial_parameters(fitting.log_lambda_init)
    trial, score = provider.evaluate(params)
    gradient, hessian = provider.differentiate(params, trial)

    dense = NewtonOptimizer(fitting)
    dense_trial, dense_score = dense._fit_and_score(params, fitting.beta_init)
    dense_gradient, dense_hessian = dense._diff_grad_hess(
        params, dense_trial.coefficients
    )
    strict = {"rtol": STRICT.rtol, "atol": STRICT.atol}
    np.testing.assert_allclose(score, dense_score, **strict)
    np.testing.assert_allclose(gradient, dense_gradient, **strict)
    np.testing.assert_allclose(hessian, dense_hessian, **strict)

    dense_result = dense.run()
    compressed_result = NewtonOptimizer(fitting, score_provider=provider).run()
    np.testing.assert_allclose(
        compressed_result.smoothing_params,
        dense_result.smoothing_params,
        **strict,
    )
    compressed_public = penalty_ops.transform_coefficients(
        fitting.penalty_structure, compressed_result.pirls_result.coefficients
    )
    dense_public = penalty_ops.transform_coefficients(
        fitting.penalty_structure, dense_result.pirls_result.coefficients
    )
    np.testing.assert_allclose(compressed_public, dense_public, **strict)
    np.testing.assert_allclose(compressed_result.score, dense_result.score, **strict)
    np.testing.assert_allclose(compressed_result.edf, dense_result.edf, **strict)
    np.testing.assert_allclose(compressed_result.scale, dense_result.scale, **strict)
    assert provider.finalize_calls == 1


def test_fixed_sp_provider_and_public_route_preserve_result_contract() -> None:
    fitting = _problem()
    fixed = np.array([1.7, 3.2])
    dense_fixed = _fit_fixed_sp(fitting, fixed)
    provider = build_dense_gaussian_qr_provider(fitting)
    compressed_fixed = _fit_fixed_sp(fitting, fixed, score_provider=provider)
    strict = {"rtol": STRICT.rtol, "atol": STRICT.atol}
    np.testing.assert_allclose(
        compressed_fixed.smoothing_params, dense_fixed.smoothing_params, **strict
    )
    np.testing.assert_allclose(compressed_fixed.score, dense_fixed.score, **strict)
    np.testing.assert_allclose(compressed_fixed.edf, dense_fixed.edf, **strict)
    assert provider.finalize_calls == 1

    formula, data, weight, offset = _problem_inputs()
    dense = GAM(formula, family="gaussian").fit(
        data, weights=weight, offset=offset, result="inference"
    )
    compressed = GAM(
        formula,
        family="gaussian",
        control=FitControl(gaussian_compression=True),
    ).fit(data, weights=weight, offset=offset, result="inference")
    assert dense.execution_route == "dense"
    assert compressed.execution_route == "gaussian_qr"
    assert compressed.execution_fallback_reason is None
    np.testing.assert_allclose(compressed.coefficients, dense.coefficients, **strict)
    np.testing.assert_allclose(
        compressed.smoothing_params, dense.smoothing_params, **strict
    )
    np.testing.assert_allclose(compressed.score, dense.score, **strict)
    np.testing.assert_allclose(compressed.edf_total, dense.edf_total, **strict)
    np.testing.assert_allclose(compressed.scale, dense.scale, **strict)


def test_provider_score_and_derivative_jaxprs_have_no_observation_arrays() -> None:
    fitting = _problem()
    provider = build_dense_gaussian_qr_provider(fitting)
    params = provider.initial_parameters(fitting.log_lambda_init)
    trial, _ = provider.evaluate(params)

    score_jaxpr = jax.make_jaxpr(lambda value: provider.evaluate(value)[1])(params)
    derivative_jaxpr = jax.make_jaxpr(
        lambda value: provider.differentiate(value, trial)
    )(params)
    _assert_no_observation_dimension(score_jaxpr, fitting.n_obs)
    _assert_no_observation_dimension(derivative_jaxpr, fitting.n_obs)
    assert provider.finalize_calls == 0


def test_lbfgs_matches_newton_meaningful_fit_and_shared_gradient_strict() -> None:
    fitting = _problem()
    newton_provider = build_dense_gaussian_qr_provider(fitting)
    lbfgs_provider = build_dense_gaussian_qr_provider(fitting)
    params = lbfgs_provider.initial_parameters(fitting.log_lambda_init)
    state, _ = lbfgs_provider.evaluate(params)
    first_order, no_hessian = lbfgs_provider.differentiate(
        params, state, second_order=False
    )
    second_order, _ = lbfgs_provider.differentiate(params, state)
    assert no_hessian is None
    np.testing.assert_allclose(
        first_order, second_order, rtol=STRICT.rtol, atol=STRICT.atol
    )

    newton = NewtonOptimizer(fitting, score_provider=newton_provider).run()
    lbfgs = optimize_gaussian_qr_lbfgs(fitting, lbfgs_provider)
    assert newton.converged
    assert lbfgs.converged
    for actual, expected in (
        (lbfgs.score, newton.score),
        (lbfgs.edf, newton.edf),
        (lbfgs.scale, newton.scale),
        (lbfgs.pirls_result.coefficients, newton.pirls_result.coefficients),
        (lbfgs.pirls_result.eta, newton.pirls_result.eta),
        (lbfgs.pirls_result.deviance, newton.pirls_result.deviance),
    ):
        np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    assert newton_provider.finalize_calls == lbfgs_provider.finalize_calls == 1


@pytest.mark.platform_sensitive
def test_real_large_m_provider_uses_no_outer_hessian_or_square_m_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitting = _large_m_problem()
    provider = build_dense_gaussian_qr_provider(fitting)
    params = provider.initial_parameters(fitting.log_lambda_init)
    state, _ = provider.evaluate(params)
    n_outer = len(params)
    assert use_gaussian_qr_lbfgs(fitting.n_penalties)

    def fail_hessian(*_args, **_kwargs):
        raise AssertionError("large-m Gaussian optimization formed an outer Hessian")

    monkeypatch.setattr(
        gaussian_optimizer_module, "jit_gaussian_qr_reml_hessian", fail_hessian
    )
    gradient_jaxpr = jax.make_jaxpr(
        lambda value: provider.differentiate(value, state, second_order=False)[0]
    )(params)
    shapes = _jaxpr_shapes(gradient_jaxpr)
    # The 51x51 penalty-range determinant is part of the scalar REML score.
    # The forbidden outer Hessian is 52x52 here: 51 rho plus log(phi).
    assert (n_outer, n_outer) not in shapes

    result = optimize_gaussian_qr_lbfgs(
        fitting,
        provider,
        control=StreamREMLControl(
            max_iter=1,
            maxfun=10,
            maxcor=3,
            maxls=5,
            gtol=1e-8,
            ftol=0.0,
        ),
    )
    retained = (
        provider._compression.R,
        provider._compression.qtz,
        *provider._penalty_roots,
        result.log_lambda,
        result.gradient,
        result.pirls_result.coefficients,
        result.pirls_result.L,
    )
    assert all(np.shape(value) != (n_outer, n_outer) for value in retained)
    assert provider.finalize_calls == 1

    formula, data = _large_m_inputs()
    public = GAM(formula, control=FitControl(gaussian_compression=True)).fit(
        data, result="prediction"
    )
    assert public.execution_route == "gaussian_qr"
    assert public.lambda_strategy == "lbfgs_reml"
    assert public.converged


def test_lbfgs_selection_threshold_is_strictly_above_fifty() -> None:
    assert not use_gaussian_qr_lbfgs(50)
    assert use_gaussian_qr_lbfgs(51)


def test_fixed_log_lambda_outside_estimated_bounds_is_preserved_strict() -> None:
    fitting = _problem()
    fixed_log_lambda = jax.numpy.array([41.0, 0.4], dtype=jax.numpy.float64)
    reference_provider = build_dense_gaussian_qr_provider(fitting)
    actual_provider = build_dense_gaussian_qr_provider(fitting)
    reference = newton_optimize(
        fitting,
        "REML",
        log_lambda_init=fixed_log_lambda,
        pin_lambda=True,
        score_provider=reference_provider,
    )
    actual = optimize_gaussian_qr_lbfgs(
        fitting,
        actual_provider,
        log_lambda_init=fixed_log_lambda,
        pin_lambda=True,
    )

    np.testing.assert_array_equal(actual.log_lambda, fixed_log_lambda)
    for observed, expected in (
        (actual.smoothing_params, reference.smoothing_params),
        (actual.score, reference.score),
        (actual.edf, reference.edf),
        (actual.scale, reference.scale),
        (actual.pirls_result.coefficients, reference.pirls_result.coefficients),
    ):
        np.testing.assert_allclose(
            observed, expected, rtol=STRICT.rtol, atol=STRICT.atol
        )


def test_public_large_m_selection_and_fixed_sp_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formula, data, weight, offset = _problem_inputs()
    control = FitControl(gaussian_compression=True)
    newton = GAM(formula, control=control).fit(
        data, weights=weight, offset=offset, result="prediction"
    )
    fixed = GAM(formula, sp=[0.7, 1.2], control=control).fit(
        data, weights=weight, offset=offset, result="prediction"
    )
    monkeypatch.setattr(
        gaussian_optimizer_module, "GAUSSIAN_LBFGS_PENALTY_THRESHOLD", 1
    )
    lbfgs = GAM(formula, control=control).fit(
        data, weights=weight, offset=offset, result="prediction"
    )

    def fail_hessian(*_args, **_kwargs):
        raise AssertionError("fixed large-m Gaussian fit formed an outer Hessian")

    monkeypatch.setattr(
        gaussian_optimizer_module, "jit_gaussian_qr_reml_hessian", fail_hessian
    )
    fixed_lbfgs = GAM(formula, sp=[0.7, 1.2], control=control).fit(
        data, weights=weight, offset=offset, result="prediction"
    )
    assert newton.lambda_strategy == "newton_reml"
    assert lbfgs.lambda_strategy == "lbfgs_reml"
    assert fixed.lambda_strategy == fixed_lbfgs.lambda_strategy == "fixed"
    for actual, expected in (
        (lbfgs.coefficients, newton.coefficients),
        (lbfgs.deviance, newton.deviance),
        (lbfgs.score, newton.score),
        (lbfgs.scale, newton.scale),
        (fixed_lbfgs.coefficients, fixed.coefficients),
        (fixed_lbfgs.smoothing_params, fixed.smoothing_params),
        (fixed_lbfgs.score, fixed.score),
        (fixed_lbfgs.scale, fixed.scale),
    ):
        np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)


def test_lbfgs_rejects_nonfinite_real_provider_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitting = _problem()
    provider = build_dense_gaussian_qr_provider(fitting)
    original = provider.differentiate

    def nonfinite_gradient(params, state, *, second_order=True):
        gradient, hessian = original(params, state, second_order=second_order)
        return jax.numpy.full_like(gradient, jax.numpy.nan), hessian

    monkeypatch.setattr(provider, "differentiate", nonfinite_gradient)
    with pytest.raises(FloatingPointError, match="non-finite frozen-statistics trial"):
        optimize_gaussian_qr_lbfgs(fitting, provider)
    assert provider.finalize_calls == 0


@pytest.mark.parametrize(
    ("status", "gtol", "expected_info"),
    [(2, 1e-20, "step failed"), (1, 1e10, "iteration limit")],
    ids=["line-search-above-gradient-tolerance", "iteration-limit"],
)
def test_lbfgs_terminal_failure_policy_uses_exact_final_gradient(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    gtol: float,
    expected_info: str,
) -> None:
    fitting = _problem()
    provider = build_dense_gaussian_qr_provider(fitting)

    def terminal_failure(objective, initial, **_kwargs):
        objective(initial)
        return OptimizeResult(
            x=np.asarray(initial),
            success=False,
            status=status,
            nit=4,
            message=expected_info,
        )

    monkeypatch.setattr(gaussian_optimizer_module, "minimize", terminal_failure)
    result = optimize_gaussian_qr_lbfgs(
        fitting,
        provider,
        control=StreamREMLControl(gtol=gtol, ftol=0.0),
    )
    assert not result.converged
    assert result.convergence_info == expected_info
    if status == 2:
        assert float(np.max(np.abs(result.gradient))) > gtol
    else:
        assert float(np.max(np.abs(result.gradient))) < gtol
    assert provider.finalize_calls == 1


def test_lbfgs_abnormal_line_search_profiles_gaussian_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitting = _problem()
    provider = build_dense_gaussian_qr_provider(fitting)
    newton = NewtonOptimizer(
        fitting, score_provider=build_dense_gaussian_qr_provider(fitting)
    ).run()
    assert newton.converged
    reference = np.r_[np.asarray(newton.log_lambda), 0.0]
    state, _ = provider.evaluate(jax.numpy.asarray(reference))
    score_state = state.score_state
    scale_df = (
        float(score_state.n_positive_weights) - provider._kwargs["penalty_nullity"]
    )
    optimal_scale = float(score_state.penalized_residual_sum_squares) / scale_df
    stalled = np.r_[reference[:-1], np.log(optimal_scale) + 1e-8]

    def abnormal_line_search(objective, _initial, **_kwargs):
        _, gradient = objective(stalled)
        assert abs(gradient[-1]) > 1e-10 * (1.0 + abs(float(newton.score)))
        return OptimizeResult(
            x=stalled, success=False, status=2, nit=17, message="ABNORMAL"
        )

    monkeypatch.setattr(gaussian_optimizer_module, "minimize", abnormal_line_search)
    result = optimize_gaussian_qr_lbfgs(fitting, provider)
    assert result.converged
    assert result.convergence_info == "full convergence"
    np.testing.assert_allclose(result.log_lambda, stalled[:-1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.score, newton.score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert provider.finalize_calls == 1


def test_provider_direct_constructor_rejects_nonidentity_coordinate_map() -> None:
    fitting = _problem()
    p = fitting.n_coef
    rotation, _ = np.linalg.qr(np.arange(1.0, p * p + 1.0).reshape(p, p))
    coordinate_map = IdentifiableCoordinateMap(
        basis=rotation,
        fitting_to_public=np.eye(p),
        keep=np.arange(p),
        drop=np.empty(0, dtype=np.int64),
    )
    preparation = prepare_frozen_gaussian_qr(fitting, coordinate_map)
    with pytest.raises(GaussianQRIneligible, match="identity coordinate_basis"):
        DenseGaussianQROuterProvider(fitting, preparation)

    reduced = replace(preparation, coordinate_basis=rotation[:, :-1])
    with pytest.raises(GaussianQRIneligible, match="identity coordinate_basis"):
        DenseGaussianQROuterProvider(fitting, reduced)


@pytest.mark.parametrize("edge_weight", [0.0, 1e-12, 1e11])
def test_public_route_preserves_dense_solve_and_original_deviance_weights(
    edge_weight: float,
) -> None:
    formula, data, weight, offset = _problem_inputs()
    weight = weight.copy()
    weight[0] = edge_weight
    control = FitControl(gaussian_compression=True)
    dense = GAM(formula, family="gaussian", sp=[1.7, 3.2]).fit(
        data, weights=weight, offset=offset, result="inference"
    )
    fallback = GAM(
        formula,
        family="gaussian",
        sp=[1.7, 3.2],
        control=control,
    ).fit(data, weights=weight, offset=offset, result="inference")
    assert fallback.execution_route == "dense"
    assert "unclipped finite positive weights" in fallback.execution_fallback_reason
    strict = {"rtol": STRICT.rtol, "atol": STRICT.atol}
    np.testing.assert_allclose(fallback.coefficients, dense.coefficients, **strict)
    np.testing.assert_allclose(fallback.deviance, dense.deviance, **strict)
    np.testing.assert_allclose(fallback.score, dense.score, **strict)
    np.testing.assert_allclose(fallback.scale, dense.scale, **strict)


_RELATIVE_REDUCTION = "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"


def _stationary_reference():
    """Newton's converged smoothing parameters and the closed-form log scale."""
    fitting = _problem()
    provider = build_dense_gaussian_qr_provider(fitting)
    newton = NewtonOptimizer(
        fitting, score_provider=build_dense_gaussian_qr_provider(fitting)
    ).run()
    assert newton.converged
    reference = np.r_[np.asarray(newton.log_lambda), 0.0]
    state, _ = provider.evaluate(jax.numpy.asarray(reference))
    score_state = state.score_state
    scale_df = (
        float(score_state.n_positive_weights) - provider._kwargs["penalty_nullity"]
    )
    log_scale = np.log(float(score_state.penalized_residual_sum_squares) / scale_df)
    return fitting, provider, newton, np.r_[reference[:-1], log_scale]


def _forced_stop(stalled: np.ndarray, threshold: float, *, scale_only: bool):
    """A deterministic L-BFGS-B relative-reduction (ftol=0) roundoff stop."""

    def relative_reduction_stop(objective, _initial, **_kwargs):
        _, gradient = objective(stalled)
        exceeds = np.abs(gradient) > threshold
        assert exceeds[-1] if scale_only else exceeds[:-1].any()
        if scale_only:
            assert not exceeds[:-1].any()
        return OptimizeResult(
            x=stalled, success=True, status=0, nit=34, message=_RELATIVE_REDUCTION
        )

    return relative_reduction_stop


def test_lbfgs_relative_reduction_stop_completes_gaussian_log_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only log(phi) above gtol after a roundoff stop: completed on gradients."""
    fitting, provider, newton, stationary = _stationary_reference()
    threshold = 1e-10 * (1.0 + abs(float(newton.score)))
    stalled = np.r_[stationary[:-1], stationary[-1] + 4e-10]
    monkeypatch.setattr(
        gaussian_optimizer_module,
        "minimize",
        _forced_stop(stalled, threshold, scale_only=True),
    )
    result = optimize_gaussian_qr_lbfgs(fitting, provider)
    assert result.converged
    assert result.convergence_info == "full convergence"
    np.testing.assert_array_equal(result.log_lambda, stalled[:-1])
    assert float(np.max(np.abs(result.gradient))) <= threshold
    np.testing.assert_allclose(
        result.score, newton.score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert provider.finalize_calls == 1


def test_lbfgs_fixed_smoothing_roundoff_stop_completes_log_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitting, provider, newton, stationary = _stationary_reference()
    threshold = 1e-10 * (1.0 + abs(float(newton.score)))
    stalled = np.array([stationary[-1] + 4e-10])

    def relative_reduction_stop(objective, _initial, **_kwargs):
        _, gradient = objective(stalled)
        assert abs(gradient[-1]) > threshold
        return OptimizeResult(
            x=stalled, success=True, status=0, nit=3, message=_RELATIVE_REDUCTION
        )

    monkeypatch.setattr(gaussian_optimizer_module, "minimize", relative_reduction_stop)
    result = optimize_gaussian_qr_lbfgs(
        fitting,
        provider,
        log_lambda_init=jax.numpy.asarray(stationary[:-1]),
        pin_lambda=True,
    )
    assert result.converged
    assert result.convergence_info == "full convergence"
    np.testing.assert_array_equal(result.log_lambda, stationary[:-1])


def test_lbfgs_roundoff_stop_names_the_smoothing_coordinate_left_unconverged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A smoothing coordinate above gtol stays truthfully not converged."""
    fitting, provider, newton, stationary = _stationary_reference()
    threshold = 1e-10 * (1.0 + abs(float(newton.score)))
    stalled = stationary.copy()
    stalled[1] += 1e-4
    monkeypatch.setattr(
        gaussian_optimizer_module,
        "minimize",
        _forced_stop(stalled, threshold, scale_only=False),
    )
    result = optimize_gaussian_qr_lbfgs(fitting, provider)
    assert not result.converged
    assert result.convergence_info.startswith(
        "step failed: log smoothing parameter 1 projected gradient "
    )
    assert "after an L-BFGS-B roundoff stop" in result.convergence_info
    assert "log-scale completion" not in result.convergence_info
    np.testing.assert_array_equal(result.log_lambda, stalled[:-1])
    assert provider.finalize_calls == 1


def test_lbfgs_rejected_scale_completion_keeps_the_accepted_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitting, provider, newton, stationary = _stationary_reference()
    threshold = 1e-10 * (1.0 + abs(float(newton.score)))
    stalled = np.r_[stationary[:-1], stationary[-1] + 4e-10]
    monkeypatch.setattr(
        gaussian_optimizer_module,
        "minimize",
        _forced_stop(stalled, threshold, scale_only=True),
    )
    monkeypatch.setattr(
        gaussian_optimizer_module,
        "_newton_model_rejection",
        lambda *_args: "quadratic model error",
    )
    _, expected_score = provider.evaluate(jax.numpy.asarray(stalled))
    result = optimize_gaussian_qr_lbfgs(fitting, provider)
    assert not result.converged
    assert result.convergence_info.startswith(
        "step failed: log scale projected gradient "
    )
    assert result.convergence_info.endswith(
        "log-scale completion: quadratic model error, 0 accepted step(s)"
    )
    assert float(result.score) == float(expected_score)


def test_gaussian_scale_rejection_reasons_and_profile_validation() -> None:
    """Each acceptance test of the scale completion rejects on its own."""
    from dataclasses import replace

    fitting, provider, _newton, stationary = _stationary_reference()
    objective = gaussian_optimizer_module._GaussianLBFGSObjective(provider, None)
    stalled = np.r_[stationary[:-1], stationary[-1] + 4e-10]
    _, old_free = objective(stalled)
    accepted = objective.trial
    _, new_free = objective(stationary)
    candidate = objective.trial
    nullity = provider._kwargs["penalty_nullity"]
    _, curvature = gaussian_optimizer_module._gaussian_scale_profile(
        accepted.state, nullity
    )
    step = float(stationary[-1] - stalled[-1])

    def reason(candidate=candidate, old=old_free, new=new_free):
        return gaussian_optimizer_module._gaussian_scale_rejection(
            accepted,
            candidate,
            stalled,
            stationary,
            old,
            new,
            step,
            curvature,
            n_bounded=fitting.n_penalties,
            penalty_nullity=nullity,
        )

    assert reason() is None
    assert reason(candidate=replace(candidate, score=jax.numpy.nan)) == (
        "nonfinite trial"
    )
    assert reason(new=2.0 * old_free) == "no gradient reduction"
    worse_smoothing = new_free.copy()
    worse_smoothing[0] = 10.0 * abs(old_free[-1])
    assert reason(new=worse_smoothing) == "no projected gradient reduction"

    score_state = accepted.state.score_state
    no_df = replace(
        accepted.state,
        score_state=replace(score_state, n_positive_weights=jax.numpy.asarray(nullity)),
    )
    with pytest.raises(FloatingPointError, match="degrees of freedom"):
        gaussian_optimizer_module._gaussian_scale_profile(no_df, nullity)
    no_rss = replace(
        accepted.state,
        score_state=replace(
            score_state, penalized_residual_sum_squares=jax.numpy.asarray(0.0)
        ),
    )
    with pytest.raises(FloatingPointError, match="not positive"):
        gaussian_optimizer_module._gaussian_scale_profile(no_rss, nullity)
