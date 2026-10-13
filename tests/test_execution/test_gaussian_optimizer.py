"""Newton integration gates for the frozen Gaussian QR provider."""

from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest

from jaxgam.api import GAM, _fit_fixed_sp
from jaxgam.control import FitControl
from jaxgam.execution.gaussian import prepare_frozen_gaussian_qr
from jaxgam.execution.gaussian_optimizer import (
    DenseGaussianQROuterProvider,
    GaussianQRIneligible,
    build_dense_gaussian_qr_provider,
)
from jaxgam.families.standard import Gaussian
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.newton import NewtonOptimizer
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


def _assert_no_observation_dimension(closed_jaxpr, n_obs: int) -> None:
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
