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

from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.initialization import initialize_beta
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.fitting.reml import REMLCriterion, reml_criterion
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.jax_utils import build_S_lambda, to_jax, to_numpy
from tests.helpers import SEED, N
from tests.tolerances import MODERATE, STRICT

jax.config.update("jax_enable_x64", True)


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
            # Condition number must be bounded for the inverse to be reliable
            cond = np.linalg.cond(D)
            assert cond < 1e10, f"D condition number {cond:.2e} exceeds 1e10"
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


# ---------------------------------------------------------------------------
# TestComputeReparaD — Sl.setup reparameterization unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def factor_by_data() -> pd.DataFrame:
    """Factor-by test data with two levels."""
    rng = np.random.default_rng(SEED)
    n = 300
    x = rng.uniform(0, 1, n)
    fac = rng.choice(["a", "b"], n)
    y = np.where(fac == "a", np.sin(2 * np.pi * x), 0.5 * x) + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "x": x,
            "fac": pd.Categorical(fac, categories=["a", "b"]),
            "y": y,
        }
    )


@pytest.fixture
def factor_by_setup(factor_by_data) -> ModelSetup:
    """ModelSetup for y ~ s(x, by=fac, k=8, bs='cr') + fac."""
    spec = parse_formula("y ~ s(x, by=fac, k=8, bs='cr') + fac")
    return ModelSetup.build(spec, factor_by_data)


class TestComputeReparaD:
    """Unit tests for _compute_repara_D reparameterization.

    Directly tests the three code paths:
    - Singleton: D^T S D = I_r (partial identity)
    - Non-overlapping multi-penalty (factor-by): each sub-block → I_r
    - Overlapping multi-penalty (tensor product): orthogonal rotation
    """

    def test_singleton_eigenvalues_are_one(self, gaussian_setup):
        """After singleton repara, non-zero eigenvalues of S are 1.0."""
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)
        assert fd.repara_D is not None

        si = gaussian_setup.smooth_info[0]
        S_repara = to_numpy(fd.S_list[0])
        S_local = S_repara[si.first_coef : si.last_coef, si.first_coef : si.last_coef]
        eigs = np.linalg.eigvalsh(S_local)

        eps_23 = np.finfo(float).eps ** (2.0 / 3.0)
        thresh = max(eigs.max(), 0) * eps_23
        nonzero = eigs > thresh

        np.testing.assert_allclose(
            eigs[nonzero],
            np.ones(np.sum(nonzero)),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Singleton: non-zero eigenvalues should be 1.0 after repara",
        )

    def test_singleton_partial_identity(self, gaussian_setup):
        """D^T S D is a partial identity for singleton penalties."""
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)
        assert fd.repara_D is not None

        D = to_numpy(fd.repara_D)
        si = gaussian_setup.smooth_info[0]
        S_orig = gaussian_setup.penalties.penalties[0].S
        S_local = S_orig[si.first_coef : si.last_coef, si.first_coef : si.last_coef]
        D_block = D[si.first_coef : si.last_coef, si.first_coef : si.last_coef]

        DtSD = D_block.T @ S_local @ D_block
        eigs = np.sort(np.linalg.eigvalsh(DtSD))[::-1]
        rank = gaussian_setup.penalties.penalties[0].rank

        np.testing.assert_allclose(
            eigs[:rank],
            np.ones(rank),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Range-space eigenvalues of D^T S D should be 1.0",
        )
        np.testing.assert_allclose(
            eigs[rank:],
            np.zeros(len(eigs) - rank),
            atol=STRICT.atol,
            err_msg="Null-space eigenvalues of D^T S D should be 0.0",
        )

    def test_factor_by_subblock_identity(self, factor_by_setup):
        """Factor-by: each sub-block satisfies D^T S D = I_r."""
        family = Gaussian()
        fd = FittingData.from_setup(factor_by_setup, family)
        assert fd.repara_D is not None

        eps_23 = np.finfo(float).eps ** (2.0 / 3.0)
        D = to_numpy(fd.repara_D)

        for si in factor_by_setup.smooth_info:
            if si.n_penalties <= 1:
                continue

            for p_offset in range(si.n_penalties):
                p_idx = si.first_penalty + p_offset
                S_full = factor_by_setup.penalties.penalties[p_idx].S
                S_local = S_full[
                    si.first_coef : si.last_coef, si.first_coef : si.last_coef
                ]
                D_block = D[si.first_coef : si.last_coef, si.first_coef : si.last_coef]

                # Find the sub-range where this penalty acts
                row_sums = np.sum(np.abs(S_local), axis=1)
                nonzero_rows = np.where(row_sums > 0)[0]
                if len(nonzero_rows) == 0:
                    continue

                sub_start = int(nonzero_rows[0])
                sub_stop = int(nonzero_rows[-1]) + 1
                S_sub = S_local[sub_start:sub_stop, sub_start:sub_stop]
                D_sub = D_block[sub_start:sub_stop, sub_start:sub_stop]

                DtSD = D_sub.T @ S_sub @ D_sub
                eigs = np.sort(np.linalg.eigvalsh(DtSD))[::-1]

                # Count rank from eigenvalues
                orig_eigs = np.linalg.eigvalsh(S_sub)
                thresh = max(orig_eigs.max(), 0) * eps_23
                rank = int(np.sum(orig_eigs > thresh))

                np.testing.assert_allclose(
                    eigs[:rank],
                    np.ones(rank),
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                    err_msg=f"Factor-by penalty {p_idx}: range eigs should be 1.0",
                )

    def test_tensor_product_orthogonal(self, tensor_setup):
        """Tensor product: D block is orthogonal (D^T D = I)."""
        family = Gaussian()
        fd = FittingData.from_setup(tensor_setup, family)
        assert fd.repara_D is not None

        D = to_numpy(fd.repara_D)
        si = tensor_setup.smooth_info[0]
        D_block = D[si.first_coef : si.last_coef, si.first_coef : si.last_coef]

        k = D_block.shape[0]
        np.testing.assert_allclose(
            D_block.T @ D_block,
            np.eye(k),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="Tensor product D block should be orthogonal",
        )

    def test_block_diagonal_isolation(self, two_smooth_setup):
        """D is block-diagonal: no mixing between smooth column ranges."""
        family = Gaussian()
        fd = FittingData.from_setup(two_smooth_setup, family)
        assert fd.repara_D is not None

        D = to_numpy(fd.repara_D)
        infos = two_smooth_setup.smooth_info

        for i, si_i in enumerate(infos):
            for j, si_j in enumerate(infos):
                if i == j:
                    continue
                off_block = D[
                    si_i.first_coef : si_i.last_coef,
                    si_j.first_coef : si_j.last_coef,
                ]
                np.testing.assert_allclose(
                    off_block,
                    np.zeros_like(off_block),
                    atol=STRICT.atol,
                    err_msg=f"D should not mix smooth {i} and smooth {j}",
                )

    def test_no_penalties_returns_none(self, parametric_setup):
        """No penalties → repara_D is None."""
        family = Gaussian()
        fd = FittingData.from_setup(parametric_setup, family)
        assert fd.repara_D is None

    def test_condition_number_bounded(self, gaussian_setup):
        """D condition number should be reasonable for a typical model."""
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)
        if fd.repara_D is None:
            pytest.skip("No reparameterization")

        D = to_numpy(fd.repara_D)
        cond = np.linalg.cond(D)
        assert cond < 1e10, f"D condition number {cond:.2e} exceeds 1e10"

    def test_condition_number_bounded_tensor(self, tensor_setup):
        """D condition number bounded for tensor product models."""
        family = Gaussian()
        fd = FittingData.from_setup(tensor_setup, family)
        if fd.repara_D is None:
            pytest.skip("No reparameterization")

        D = to_numpy(fd.repara_D)
        cond = np.linalg.cond(D)
        assert cond < 1e10, f"D condition number {cond:.2e} exceeds 1e10"

    def test_condition_number_bounded_factor_by(self, factor_by_setup):
        """D condition number bounded for factor-by models."""
        family = Gaussian()
        fd = FittingData.from_setup(factor_by_setup, family)
        if fd.repara_D is None:
            pytest.skip("No reparameterization")

        D = to_numpy(fd.repara_D)
        cond = np.linalg.cond(D)
        assert cond < 1e10, f"D condition number {cond:.2e} exceeds 1e10"


# ---------------------------------------------------------------------------
# TestREMLInvariance — REML criterion invariance under reparameterization
# ---------------------------------------------------------------------------


class TestREMLInvariance:
    """REML criterion is invariant under Sl.setup reparameterization.

    Tests exercise production functions (pirls_loop, REMLCriterion,
    reml_criterion, build_S_lambda) rather than reimplementing math.
    """

    def test_mu_invariant(self, gaussian_setup):
        """pirls_loop produces the same mu in both coordinate spaces."""
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)
        if fd.repara_D is None:
            pytest.skip("No reparameterization")

        log_lambda = jnp.array([2.0])

        # Reparameterized space: production pirls_loop with fd arrays
        S_lam_repara = fd.S_lambda(log_lambda)
        beta_init_repara = initialize_beta(
            np.asarray(fd.X), np.asarray(fd.y), np.asarray(fd.wt), family, None
        )
        pirls_repara = pirls_loop(
            fd.X,
            fd.y,
            to_jax(np.asarray(beta_init_repara)),
            S_lam_repara,
            family,
            fd.wt,
            fd.offset,
        )

        # Original space: production pirls_loop with setup arrays
        X_orig = to_jax(gaussian_setup.X)
        S_orig_list = tuple(to_jax(p.S) for p in gaussian_setup.penalties.penalties)
        n_coef = gaussian_setup.X.shape[1]
        S_lam_orig = build_S_lambda(log_lambda, S_orig_list, n_coef)
        beta_init_orig = initialize_beta(
            gaussian_setup.X,
            gaussian_setup.y,
            gaussian_setup.weights,
            family,
            None,
        )
        pirls_orig = pirls_loop(
            X_orig,
            fd.y,
            to_jax(np.asarray(beta_init_orig)),
            S_lam_orig,
            family,
            fd.wt,
            fd.offset,
        )

        # The two spaces solve different Cholesky systems (different
        # conditioning), so floating-point paths diverge at ~sqrt(eps).
        # MODERATE captures this: genuine invariance, different numerics.
        np.testing.assert_allclose(
            to_numpy(pirls_repara.mu),
            to_numpy(pirls_orig.mu),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
            err_msg="pirls_loop mu must be the same in both spaces",
        )

    def test_reml_score_invariant(self, gaussian_setup):
        """Production REML score is invariant under reparameterization.

        Reparameterized space uses REMLCriterion.score() (the full
        production path). Original space uses reml_criterion() with
        original-space penalties and block metadata.
        """
        family = Gaussian()
        fd = FittingData.from_setup(gaussian_setup, family)
        if fd.repara_D is None:
            pytest.skip("No reparameterization")

        log_lambda = jnp.array([2.0])

        # -- Reparameterized: production pirls + REMLCriterion --
        S_lam_repara = fd.S_lambda(log_lambda)
        beta_init = initialize_beta(
            np.asarray(fd.X), np.asarray(fd.y), np.asarray(fd.wt), family, None
        )
        pirls_repara = pirls_loop(
            fd.X,
            fd.y,
            to_jax(np.asarray(beta_init)),
            S_lam_repara,
            family,
            fd.wt,
            fd.offset,
        )
        criterion_repara = REMLCriterion(fd, pirls_repara)
        score_repara = float(criterion_repara.score(log_lambda))

        # -- Original space: production pirls + reml_criterion() --
        X_orig = to_jax(gaussian_setup.X)
        S_orig_list = tuple(to_jax(p.S) for p in gaussian_setup.penalties.penalties)
        n_coef = gaussian_setup.X.shape[1]
        S_lam_orig = build_S_lambda(log_lambda, S_orig_list, n_coef)
        beta_init_orig = initialize_beta(
            gaussian_setup.X,
            gaussian_setup.y,
            gaussian_setup.weights,
            family,
            None,
        )
        pirls_orig = pirls_loop(
            X_orig,
            fd.y,
            to_jax(np.asarray(beta_init_orig)),
            S_lam_orig,
            family,
            fd.wt,
            fd.offset,
        )

        # Block metadata for original-space singleton penalty.
        # This is the only non-production computation — we need it
        # because from_setup always reparameterizes before computing
        # block metadata, so no production path exists for original
        # space metadata.
        si = gaussian_setup.smooth_info[0]
        S_np = gaussian_setup.penalties.penalties[0].S
        S_local = S_np[si.first_coef : si.last_coef, si.first_coef : si.last_coef]
        eig_vals = np.linalg.eigvalsh(S_local)
        eps_23 = np.finfo(float).eps ** (2.0 / 3.0)
        thresh = max(np.abs(eig_vals).max(), 0) * eps_23
        nonzero = eig_vals > thresh
        rank = int(np.sum(nonzero))
        eig_const = float(np.sum(np.log(np.maximum(eig_vals[nonzero], 1e-30))))

        # Use production reml_criterion with original-space arrays
        score_orig = float(
            reml_criterion(
                log_lambda,
                XtWX=pirls_orig.XtWX,
                beta=pirls_orig.coefficients,
                deviance=pirls_orig.deviance,
                ls_sat=family.saturated_loglik(fd.y, fd.wt, criterion_repara.scale),
                S_list=S_orig_list,
                phi=criterion_repara.scale,
                Mp=fd.total_penalty_null_dim,
                singleton_sp_indices=(0,),
                singleton_ranks=(rank,),
                singleton_eig_constants=jnp.array([eig_const]),
                multi_block_sp_indices=(),
                multi_block_ranks=(),
                multi_block_proj_S=(),
            )
        )

        np.testing.assert_allclose(
            score_repara,
            score_orig,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg="REML score must be invariant under reparameterization",
        )
