"""Frozen Gaussian QR CPU preparation tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.execution.gaussian_qr.preparation import prepare_frozen_gaussian_qr
from jaxgam.families.standard import Gaussian
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.gaussian_qr.kernel import gaussian_qr_reml_score
from jaxgam.fitting.newton import NewtonOptimizer
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.rank_coordinates import IdentifiableCoordinateMap
from jaxgam.links.links import LogLink
from tests.tolerances import MODERATE, STRICT


def _fixture() -> tuple[ModelSetup, FittingData, IdentifiableCoordinateMap]:
    rng = np.random.default_rng(48)
    n = 31
    data = {"x": rng.uniform(size=n), "z": rng.uniform(size=n)}
    offset = np.linspace(-0.2, 0.3, n)
    weights = 0.4 + (np.arange(n) % 7) / 5
    data["y"] = (
        0.2
        + np.sin(3 * data["x"])
        + np.cos(2 * data["z"])
        + offset
        + rng.normal(0, 0.1, n)
    )
    setup = ModelSetup.build(
        parse_formula("y ~ te(x, z, k=4)"), data, weights=weights, offset=offset
    )
    fitting = FittingData.from_setup(setup, Gaussian())
    p = fitting.n_coef
    angle = 0.31
    transform = np.eye(p)
    transform[:2, :2] = (
        (np.cos(angle), -np.sin(angle)),
        (np.sin(angle), np.cos(angle)),
    )
    coordinate_map = IdentifiableCoordinateMap(
        np.eye(p), transform, np.arange(p), np.array([], dtype=int)
    )
    return setup, fitting, coordinate_map


def test_preparation_bridges_local_d_to_independent_public_tensor_penalties() -> None:
    setup, fitting, coordinate_map = _fixture()
    assert setup.penalties is not None
    prepared = prepare_frozen_gaussian_qr(fitting, coordinate_map)
    # This uses retained public setup objects, rather than reconstructing the
    # expectation from FittingData's local transforms.
    target = np.asarray(setup.y) - np.asarray(setup.offset)
    weighted_public = (
        np.sqrt(np.asarray(setup.weights))[:, None]
        * np.asarray(setup.X)
        @ coordinate_map.fitting_to_public
    )
    weighted_target = np.sqrt(np.asarray(setup.weights)) * target
    expected_gram = weighted_public.T @ weighted_public
    np.testing.assert_allclose(
        prepared.compression.R.T @ prepared.compression.R,
        expected_gram,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        prepared.compression.R.T @ prepared.compression.qtz,
        weighted_public.T @ weighted_target,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    roots = prepared.penalty_roots
    expected = (
        coordinate_map.fitting_to_public.T
        @ setup.penalties.materialize()
        @ coordinate_map.fitting_to_public
    )
    actual = sum(root.T @ root for root in roots)
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    assert prepared.penalty_range_basis.shape[1] == fitting.total_penalty_rank
    assert prepared.penalty_nullity == fitting.total_penalty_null_dim
    assert prepared.compression.n_obs == fitting.n_obs
    assert prepared.compression.n_positive_weights == fitting.n_obs
    for value in (
        prepared.compression.R,
        prepared.compression.qtz,
        prepared.compression.residual_tail,
        prepared.compression.n_obs,
        prepared.compression.n_positive_weights,
        prepared.compression.sum_log_positive_weights,
    ):
        assert not value.flags.writeable


def test_preparation_rejects_unsettled_weight_and_pdet_conventions() -> None:
    _setup, fitting, coordinate_map = _fixture()
    altered = fitting.__class__(**{**fitting.__dict__, "wt": fitting.wt.at[0].set(0.0)})
    with pytest.raises(ValueError, match="unclipped positive weights"):
        prepare_frozen_gaussian_qr(altered, coordinate_map)
    nonorthogonal = IdentifiableCoordinateMap(
        np.eye(fitting.n_coef),
        np.diag(np.linspace(1.0, 2.0, fitting.n_coef)),
        np.arange(fitting.n_coef),
        np.array([], dtype=int),
    )
    with pytest.raises(ValueError, match="orthonormal global"):
        prepare_frozen_gaussian_qr(fitting, nonorthogonal)
    log_fitting = fitting.__class__(
        **{**fitting.__dict__, "family": Gaussian(link=LogLink())}
    )
    with pytest.raises(ValueError, match="identity-link"):
        prepare_frozen_gaussian_qr(log_fitting, coordinate_map)


def test_prepared_tensor_score_gradient_and_hessian_match_dense_refits() -> None:
    """Compare each prepared fixed-rho score to an independent dense refit."""
    _setup, fitting, coordinate_map = _fixture()
    prepared = prepare_frozen_gaussian_qr(fitting, coordinate_map)
    roots = tuple(jnp.asarray(root) for root in prepared.penalty_roots)
    compression = prepared.compression

    def qr_value(params: jax.Array) -> jax.Array:
        return gaussian_qr_reml_score(
            params,
            compression,
            roots,
            jnp.asarray(prepared.coordinate_basis),
            jnp.asarray(prepared.public_basis),
            jnp.asarray(prepared.penalty_range_basis),
            jnp.array(1.0),
            root_sp_indices=prepared.root_sp_indices,
            n_smoothing_parameters=prepared.n_smoothing_parameters,
            penalty_nullity=prepared.penalty_nullity,
            scale_known=False,
        ).score

    def dense_value(params: jax.Array) -> jax.Array:
        rho, phi = params[:-1], jnp.exp(params[-1])
        scaled = tuple(
            jnp.exp(rho[index] / 2.0) * root
            for root, index in zip(roots, prepared.root_sp_indices, strict=True)
        )
        R, qtz = jnp.asarray(compression.R), jnp.asarray(compression.qtz)
        S = sum(root.T @ root for root in scaled)
        beta = jnp.linalg.solve(R.T @ R + S, R.T @ qtz)
        rss = compression.residual_tail + jnp.sum((qtz - R @ beta) ** 2)
        p_rows = jnp.vstack(
            tuple(root @ jnp.asarray(prepared.penalty_range_basis) for root in scaled)
        )
        log_s = 2 * jnp.sum(
            jnp.log(jnp.abs(jnp.diag(jnp.linalg.qr(p_rows, mode="reduced")[1])))
        )
        log_h = jnp.linalg.slogdet(R.T @ R + S)[1]
        ls_sat = (
            -compression.n_positive_weights * jnp.log(2 * jnp.pi * phi) / 2
            + compression.sum_log_positive_weights / 2
        )
        return (
            (rss + beta @ S @ beta) / (2 * phi)
            - ls_sat
            + (log_h - log_s) / 2
            - prepared.penalty_nullity * jnp.log(2 * jnp.pi * phi) / 2
        )

    for params in (
        jnp.array((-2.0, 1.0, np.log(0.8))),
        jnp.array((1.2, -1.1, np.log(1.3))),
    ):
        np.testing.assert_allclose(
            qr_value(params), dense_value(params), rtol=STRICT.rtol, atol=STRICT.atol
        )
        np.testing.assert_allclose(
            jax.grad(qr_value)(params),
            jax.grad(dense_value)(params),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        np.testing.assert_allclose(
            jax.hessian(qr_value)(params),
            jax.hessian(dense_value)(params),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )


def test_prepared_tensor_matches_current_dense_newton_derivatives() -> None:
    """Match the live dense Newton score and its custom-JVP derivatives."""
    _setup, fitting, coordinate_map = _fixture()
    prepared = prepare_frozen_gaussian_qr(fitting, coordinate_map)
    roots = tuple(jnp.asarray(root) for root in prepared.penalty_roots)

    def qr_value(params: jax.Array) -> jax.Array:
        return gaussian_qr_reml_score(
            params,
            prepared.compression,
            roots,
            jnp.asarray(prepared.coordinate_basis),
            jnp.asarray(prepared.public_basis),
            jnp.asarray(prepared.penalty_range_basis),
            jnp.array(1.0),
            root_sp_indices=prepared.root_sp_indices,
            n_smoothing_parameters=prepared.n_smoothing_parameters,
            penalty_nullity=prepared.penalty_nullity,
            scale_known=False,
        ).score

    optimizer = NewtonOptimizer(fitting)
    assert optimizer._diff_grad_hess is not None
    beta_warm = optimizer._initial_beta()
    for params in (
        jnp.array((-2.0, 1.0, np.log(0.8))),
        jnp.array((1.2, -1.1, np.log(1.3))),
    ):
        # _fit_and_score dispatches the current _jit_fit_and_score with the
        # optimizer's live FittingData and _jit_kwargs.  Its companion
        # _diff_grad_hess differentiates _diff_score, including the refitted
        # beta sensitivity absent from a fixed-beta criterion derivative.
        dense_fit, dense_score = optimizer._fit_and_score(params, beta_warm)
        dense_grad, dense_hess = optimizer._diff_grad_hess(
            params, dense_fit.coefficients
        )
        np.testing.assert_allclose(
            qr_value(params), dense_score, rtol=STRICT.rtol, atol=STRICT.atol
        )
        np.testing.assert_allclose(
            jax.grad(qr_value)(params),
            dense_grad,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        np.testing.assert_allclose(
            jax.hessian(qr_value)(params),
            dense_hess,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )


def test_prepared_tensor_matches_pinned_sl_fit() -> None:
    """Compare the same public weighted tensor fixture to pinned ``Sl.fit``."""
    from tests.r_bridge import RBridge

    bridge = RBridge()
    ok, reason = bridge.check_versions()
    if not ok or bridge.mode != "rpy2":
        pytest.skip(reason if not ok else "rpy2 required for Sl.fit oracle")
    setup, fitting, coordinate_map = _fixture()
    prepared = prepare_frozen_gaussian_qr(fitting, coordinate_map)
    roots = tuple(jnp.asarray(root) for root in prepared.penalty_roots)

    def value(params: jax.Array) -> jax.Array:
        return gaussian_qr_reml_score(
            params,
            prepared.compression,
            roots,
            jnp.asarray(prepared.coordinate_basis),
            jnp.asarray(prepared.public_basis),
            jnp.asarray(prepared.penalty_range_basis),
            jnp.array(1.0),
            root_sp_indices=prepared.root_sp_indices,
            n_smoothing_parameters=prepared.n_smoothing_parameters,
            penalty_nullity=prepared.penalty_nullity,
            scale_known=False,
        ).score

    # ModelSetup deliberately does not retain raw inputs; recover exactly the
    # fixture arrays from the construction instead of synthesizing a new draw.
    rng = np.random.default_rng(48)
    n = fitting.n_obs
    x = rng.uniform(size=n)
    z = rng.uniform(size=n)
    offset = np.linspace(-0.2, 0.3, n)
    weights = 0.4 + (np.arange(n) % 7) / 5
    y = 0.2 + np.sin(3 * x) + np.cos(2 * z) + offset + rng.normal(0, 0.1, n)
    points = ((-2.0, 1.0, np.log(0.8)), (1.2, -1.1, np.log(1.3)))
    for point in points:
        reference = bridge.source_gaussian_sl_fit(
            pd.DataFrame({"x": x, "z": z, "y": y}),
            weights,
            offset,
            np.asarray(point[:2]),
            point[2],
        )
        np.testing.assert_allclose(
            np.asarray(setup.X),
            reference["X"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        for python_penalty, r_penalty in zip(
            setup.penalties.penalties, reference["S"], strict=True
        ):
            np.testing.assert_allclose(
                python_penalty.S,
                r_penalty,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
            )
        params = jnp.asarray(point)
        expected_score = reference["score"] - np.log(weights).sum() / 2
        np.testing.assert_allclose(
            value(params), expected_score, rtol=MODERATE.rtol, atol=MODERATE.atol
        )
        np.testing.assert_allclose(
            jax.grad(value)(params),
            reference["gradient"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        np.testing.assert_allclose(
            jax.hessian(value)(params),
            reference["hessian"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
