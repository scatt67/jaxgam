"""Tests for FittingData Phase 1→2 boundary container.

Covers:
1. from_setup: array transfer, shapes, values, offset handling
2. Purely parametric models (no penalties)
3. Penalty metadata (ranks, null space dims)
4. S_lambda: single penalty, multi-penalty, JAX traceability, zero penalties
5. n_penalties property
6. End-to-end: ModelSetup.build → FittingData.from_setup → pirls_loop
7. Device placement

Design doc reference: docs/design.md §1.3, §4.4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pymgcv.families.standard import Gaussian, Poisson
from pymgcv.fitting.data import FittingData
from pymgcv.fitting.initialization import initialize_beta
from pymgcv.fitting.pirls import pirls_loop
from pymgcv.formula.design import ModelSetup
from pymgcv.formula.parser import parse_formula
from pymgcv.jax_utils import to_jax, to_numpy
from tests.tolerances import STRICT

jax.config.update("jax_enable_x64", True)

SEED = 42
N = 200


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian_data() -> pd.DataFrame:
    """Standard Gaussian test data with one smooth."""
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 1, N)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, N)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def gaussian_setup(gaussian_data) -> ModelSetup:
    """ModelSetup for y ~ s(x, bs='cr', k=10)."""
    spec = parse_formula("y ~ s(x, bs='cr', k=10)")
    return ModelSetup.build(spec, gaussian_data)


@pytest.fixture
def two_smooth_data() -> pd.DataFrame:
    """Test data with two covariates for multi-smooth model."""
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, N)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def two_smooth_setup(two_smooth_data) -> ModelSetup:
    """ModelSetup for y ~ s(x1, bs='cr', k=8) + s(x2, bs='cr', k=8)."""
    spec = parse_formula("y ~ s(x1, bs='cr', k=8) + s(x2, bs='cr', k=8)")
    return ModelSetup.build(spec, two_smooth_data)


@pytest.fixture
def tensor_data() -> pd.DataFrame:
    """Test data for tensor product model."""
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    y = np.sin(2 * np.pi * x1) * x2 + rng.normal(0, 0.3, N)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def tensor_setup(tensor_data) -> ModelSetup:
    """ModelSetup for y ~ te(x1, x2, bs='cr', k=5)."""
    spec = parse_formula("y ~ te(x1, x2, bs='cr', k=5)")
    return ModelSetup.build(spec, tensor_data)


@pytest.fixture
def parametric_data() -> pd.DataFrame:
    """Purely parametric test data (no smooths)."""
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 1, N)
    y = 1.0 + 2.0 * x + rng.normal(0, 0.3, N)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def parametric_setup(parametric_data) -> ModelSetup:
    """ModelSetup for y ~ x (purely parametric)."""
    spec = parse_formula("y ~ x")
    return ModelSetup.build(spec, parametric_data)


# ---------------------------------------------------------------------------
# TestFromSetup — array transfer and shapes
# ---------------------------------------------------------------------------


class TestFromSetupBasic:
    """Test that from_setup transfers arrays correctly."""

    def test_shapes_match(self, gaussian_setup):
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)

        assert fd.X.shape == gaussian_setup.X.shape
        assert fd.y.shape == gaussian_setup.y.shape
        assert fd.wt.shape == gaussian_setup.weights.shape
        assert fd.n_obs == gaussian_setup.n_obs
        assert fd.n_coef == gaussian_setup.X.shape[1]

    def test_arrays_are_jax(self, gaussian_setup):
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)

        assert isinstance(fd.X, jax.Array)
        assert isinstance(fd.y, jax.Array)
        assert isinstance(fd.wt, jax.Array)
        assert isinstance(fd.log_lambda_init, jax.Array)
        for S_j in fd.S_list:
            assert isinstance(S_j, jax.Array)

    def test_values_preserved(self, gaussian_setup):
        """Values must match original NumPy arrays at STRICT tolerance.

        X may be reparameterized (X_repara = X_orig @ D) for better
        Hessian conditioning. Check the original-space relationship.
        """
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)

        if fd.repara_D is not None:
            # X_repara = X_orig @ D, so X_orig = X_repara @ D^{-1}
            D = to_numpy(fd.repara_D)
            X_recovered = to_numpy(fd.X) @ np.linalg.inv(D)
            np.testing.assert_allclose(
                X_recovered,
                gaussian_setup.X,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg="X values must be recoverable via D inverse",
            )
        else:
            np.testing.assert_allclose(
                to_numpy(fd.X),
                gaussian_setup.X,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg="X values must match after transfer",
            )
        np.testing.assert_allclose(
            to_numpy(fd.y),
            gaussian_setup.y,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="y values must match after transfer",
        )
        np.testing.assert_allclose(
            to_numpy(fd.wt),
            gaussian_setup.weights,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="weights must match after transfer",
        )

    def test_family_stored(self, gaussian_setup):
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)
        assert fd.family is family


class TestFromSetupOffset:
    """Test offset handling in from_setup."""

    def test_with_offset(self, gaussian_data):
        spec = parse_formula("y ~ s(x, bs='cr', k=10)")
        offset = np.ones(N) * 0.5
        setup = ModelSetup.build(spec, gaussian_data, offset=offset)

        family = Gaussian()
        fd = FittingData.from_setup(setup, family)

        assert fd.offset is not None
        assert isinstance(fd.offset, jax.Array)
        np.testing.assert_allclose(
            to_numpy(fd.offset),
            offset,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_no_offset(self, gaussian_setup):
        """offset=None when ModelSetup has no offset."""
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)
        assert fd.offset is None


class TestFromSetupPurelyParametric:
    """Purely parametric model: no penalties → empty S_list."""

    def test_empty_penalties(self, parametric_setup):
        family = Gaussian()
        fd = FittingData.from_setup(parametric_setup, family)

        assert fd.S_list == ()
        assert fd.penalty_ranks == ()
        assert fd.penalty_null_dims == ()
        assert fd.n_penalties == 0
        assert fd.log_lambda_init.shape == (0,)


# ---------------------------------------------------------------------------
# TestPenaltyMetadata
# ---------------------------------------------------------------------------


class TestPenaltyMetadata:
    """Penalty ranks and null_space_dims must match Penalty objects."""

    def test_single_smooth(self, gaussian_setup):
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)

        assert len(fd.penalty_ranks) == fd.n_penalties
        assert len(fd.penalty_null_dims) == fd.n_penalties

        # Cross-check against CompositePenalty
        for j, penalty in enumerate(gaussian_setup.penalties.penalties):
            assert fd.penalty_ranks[j] == penalty.rank
            assert fd.penalty_null_dims[j] == penalty.null_space_dim

    def test_multi_smooth(self, two_smooth_setup):
        family = Gaussian()
        fd = FittingData.from_setup(two_smooth_setup, family)

        assert len(fd.penalty_ranks) == fd.n_penalties
        for j, penalty in enumerate(two_smooth_setup.penalties.penalties):
            assert fd.penalty_ranks[j] == penalty.rank
            assert fd.penalty_null_dims[j] == penalty.null_space_dim

    def test_tensor_product(self, tensor_setup):
        """Tensor products have multiple penalties — check all transferred."""
        family = Gaussian()
        fd = FittingData.from_setup(tensor_setup, family)

        # te() should have >1 penalty (one per marginal)
        assert fd.n_penalties >= 2
        assert len(fd.S_list) == fd.n_penalties
        for j, penalty in enumerate(tensor_setup.penalties.penalties):
            assert fd.penalty_ranks[j] == penalty.rank
            assert fd.penalty_null_dims[j] == penalty.null_space_dim


# ---------------------------------------------------------------------------
# TestSLambda
# ---------------------------------------------------------------------------


class TestSLambdaSinglePenalty:
    """S_lambda with a single penalty matrix."""

    def test_matches_manual(self, gaussian_setup):
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)

        log_lam = jnp.array([2.0])
        S_combined = fd.S_lambda(log_lam)

        # Manual: exp(2.0) * S_list[0]
        expected = jnp.exp(2.0) * fd.S_list[0]
        np.testing.assert_allclose(
            to_numpy(S_combined),
            to_numpy(expected),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_shape(self, gaussian_setup):
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)
        log_lam = fd.log_lambda_init
        S_combined = fd.S_lambda(log_lam)
        assert S_combined.shape == (fd.n_coef, fd.n_coef)


class TestSLambdaMultiPenalty:
    """S_lambda with multiple penalties (te() or multi-smooth)."""

    def test_matches_manual(self, tensor_setup):
        family = Gaussian()
        fd = FittingData.from_setup(tensor_setup, family)

        rng = np.random.default_rng(SEED)
        log_lam = jnp.array(rng.standard_normal(fd.n_penalties))

        S_combined = fd.S_lambda(log_lam)

        # Manual sum
        expected = jnp.zeros((fd.n_coef, fd.n_coef))
        for j, S_j in enumerate(fd.S_list):
            expected = expected + jnp.exp(log_lam[j]) * S_j

        np.testing.assert_allclose(
            to_numpy(S_combined),
            to_numpy(expected),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


class TestSLambdaJAXTraceable:
    """S_lambda must be differentiable via jax.grad for REML."""

    def test_grad_finite(self, gaussian_setup):
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)

        def scalar_fn(log_lam):
            return jnp.sum(fd.S_lambda(log_lam))

        log_lam = fd.log_lambda_init
        grad = jax.grad(scalar_fn)(log_lam)

        assert grad.shape == log_lam.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_multi_penalty(self, tensor_setup):
        """Gradient through multi-penalty S_lambda."""
        family = Gaussian()
        fd = FittingData.from_setup(tensor_setup, family)

        def scalar_fn(log_lam):
            return jnp.sum(fd.S_lambda(log_lam))

        log_lam = fd.log_lambda_init
        grad = jax.grad(scalar_fn)(log_lam)

        assert grad.shape == log_lam.shape
        assert jnp.all(jnp.isfinite(grad))
        # Gradient should be sum of S_j elements times exp(log_lam_j)
        for j in range(fd.n_penalties):
            expected_j = jnp.exp(log_lam[j]) * jnp.sum(fd.S_list[j])
            np.testing.assert_allclose(
                float(grad[j]),
                float(expected_j),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            )


class TestSLambdaZeroPenalties:
    """Purely parametric → S_lambda returns zero matrix."""

    def test_zero_matrix(self, parametric_setup):
        family = Gaussian()
        fd = FittingData.from_setup(parametric_setup, family)
        S_combined = fd.S_lambda(fd.log_lambda_init)
        assert S_combined.shape == (fd.n_coef, fd.n_coef)
        np.testing.assert_allclose(
            to_numpy(S_combined),
            np.zeros((fd.n_coef, fd.n_coef)),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


# ---------------------------------------------------------------------------
# TestNPenalties
# ---------------------------------------------------------------------------


class TestNPenalties:
    """n_penalties property."""

    def test_matches_s_list(self, gaussian_setup):
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)
        assert fd.n_penalties == len(fd.S_list)

    def test_parametric(self, parametric_setup):
        family = Gaussian()
        fd = FittingData.from_setup(parametric_setup, family)
        assert fd.n_penalties == 0


# ---------------------------------------------------------------------------
# TestEndToEnd — full pipeline through PIRLS
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """ModelSetup.build → FittingData.from_setup → pirls_loop."""

    def test_pirls_converges_gaussian(self, gaussian_data):
        spec = parse_formula("y ~ s(x, bs='cr', k=10)")
        setup = ModelSetup.build(spec, gaussian_data)
        family = Gaussian()

        fd = FittingData.from_setup(setup, family)

        # Initialize beta (Phase 1 → Phase 2 boundary)
        beta_init = initialize_beta(
            setup.X, setup.y, setup.weights, family, setup.offset
        )
        beta_jax = to_jax(np.asarray(beta_init))

        # Compute S_lambda from initial sp
        S_combined = fd.S_lambda(fd.log_lambda_init)

        # Run PIRLS
        result = pirls_loop(
            fd.X, fd.y, beta_jax, S_combined, fd.family, fd.wt, fd.offset
        )

        assert result.converged
        assert jnp.all(jnp.isfinite(result.coefficients))
        assert float(result.deviance) > 0

    def test_pirls_converges_poisson(self):
        rng = np.random.default_rng(SEED)
        x = rng.uniform(0, 1, N)
        eta = np.sin(2 * np.pi * x) + 0.5
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(float)
        data = pd.DataFrame({"x": x, "y": y})

        spec = parse_formula("y ~ s(x, bs='cr', k=10)")
        setup = ModelSetup.build(spec, data)
        family = Poisson()

        fd = FittingData.from_setup(setup, family)
        beta_init = initialize_beta(
            setup.X, setup.y, setup.weights, family, setup.offset
        )
        beta_jax = to_jax(np.asarray(beta_init))
        S_combined = fd.S_lambda(fd.log_lambda_init)

        result = pirls_loop(
            fd.X, fd.y, beta_jax, S_combined, fd.family, fd.wt, fd.offset
        )

        assert result.converged
        assert jnp.all(jnp.isfinite(result.coefficients))


# ---------------------------------------------------------------------------
# TestDevicePlacement
# ---------------------------------------------------------------------------


class TestDevicePlacement:
    """Arrays should be on the expected device."""

    def test_default_device(self, gaussian_setup):
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)

        default_device = jax.devices()[0]
        assert fd.X.devices() == {default_device}
        assert fd.y.devices() == {default_device}
        assert fd.wt.devices() == {default_device}
        for S_j in fd.S_list:
            assert S_j.devices() == {default_device}
