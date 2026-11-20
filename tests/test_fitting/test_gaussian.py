"""Frozen Gaussian QR REML kernel tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.fitting.gaussian_qr.kernel import (
    FrozenGaussianQR,
    GaussianQRScore,
    gaussian_qr_reml_score,
    jit_gaussian_qr_reml_hessian,
    jit_gaussian_qr_reml_score,
    jit_gaussian_qr_reml_value_and_grad,
)
from jaxgam.fitting.penalty_ops import (
    JaxLocalPenalty,
    JaxPenaltyBlock,
    JaxPenaltyStructure,
    JaxTransform,
)
from jaxgam.fitting.reml import reml_criterion
from tests.tolerances import MODERATE, STRICT


def _compressed_system(
    *,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    noise_scale: float = 0.12,
) -> tuple[FrozenGaussianQR, tuple[jax.Array, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Build a full-rank weighted QR system with two compact roots."""
    rng = np.random.default_rng(813)
    x = rng.normal(size=(13, 3))
    beta = np.array((0.7, -1.1, 0.35))
    offset_value = np.zeros(len(x)) if offset is None else offset
    y = x @ beta + offset_value + rng.normal(scale=noise_scale, size=len(x))
    weights_value = np.ones(len(x)) if weights is None else weights
    weighted_x = np.sqrt(weights_value)[:, None] * x
    weighted_z = np.sqrt(weights_value) * (y - offset_value)
    q, r = np.linalg.qr(weighted_x, mode="reduced")
    qtz = q.T @ weighted_z
    tail = np.sum((weighted_z - q @ qtz) ** 2)
    compression = FrozenGaussianQR(
        R=jnp.asarray(r),
        qtz=jnp.asarray(qtz),
        residual_tail=jnp.asarray(tail),
        n_obs=jnp.asarray(len(x)),
        n_positive_weights=jnp.asarray(np.count_nonzero(weights_value > 0)),
        sum_log_positive_weights=jnp.asarray(
            np.log(weights_value[weights_value > 0]).sum()
        ),
    )
    roots = (
        jnp.asarray(((0.0, 1.0, 0.0),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
    )
    return compression, roots, x, y, weights_value


def _score(
    params: jax.Array,
    compression: FrozenGaussianQR,
    roots: tuple[jax.Array, ...],
    *,
    scale_known: bool = False,
) -> GaussianQRScore:
    return gaussian_qr_reml_score(
        params,
        compression,
        roots,
        jnp.eye(3),
        jnp.eye(3),
        jnp.eye(3)[:, 1:],
        jnp.asarray(1.7),
        root_sp_indices=(0, 1),
        n_smoothing_parameters=2,
        penalty_nullity=1,
        scale_known=scale_known,
    )


def _numpy_refit_score(
    params: np.ndarray, compression: FrozenGaussianQR, roots: tuple[jax.Array, ...]
) -> float:
    """Independent dense refit of the frozen compressed system."""
    rho, log_phi = params[:2], params[2]
    r = np.asarray(compression.R)
    qtz = np.asarray(compression.qtz)
    root_arrays = tuple(np.asarray(root) for root in roots)
    penalty = sum(
        np.exp(rho[index]) * root.T @ root for index, root in enumerate(root_arrays)
    )
    beta = np.linalg.solve(r.T @ r + penalty, r.T @ qtz)
    rss = float(np.asarray(compression.residual_tail)) + np.sum((qtz - r @ beta) ** 2)
    penalty_quadratic = beta @ penalty @ beta
    penalty_range = np.array(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    weighted_range = np.diag(np.exp(rho / 2.0)) @ penalty_range[:, 1:]
    log_det_penalty = (
        2.0 * np.log(np.abs(np.diag(np.linalg.qr(weighted_range)[1]))).sum()
    )
    log_det_hessian = np.linalg.slogdet(r.T @ r + penalty)[1]
    phi = np.exp(log_phi)
    ls_sat = (
        -float(np.asarray(compression.n_positive_weights))
        * np.log(2.0 * np.pi * phi)
        / 2.0
        + float(np.asarray(compression.sum_log_positive_weights)) / 2.0
    )
    return float(
        (rss + penalty_quadratic) / (2.0 * phi)
        - ls_sat
        + (log_det_hessian - log_det_penalty) / 2.0
        - np.log(2.0 * np.pi * phi) / 2.0
    )


def test_frozen_gaussian_qr_reml_matches_refit_and_jits() -> None:
    compression, roots, _x, _y, _weights = _compressed_system()
    params = jnp.array((np.log(1.8), np.log(0.45), np.log(0.9)))
    result = _score(params, compression, roots)
    jitted = jit_gaussian_qr_reml_score(
        params,
        compression,
        roots,
        jnp.eye(3),
        jnp.eye(3),
        jnp.eye(3)[:, 1:],
        jnp.asarray(1.7),
        root_sp_indices=(0, 1),
        n_smoothing_parameters=2,
        penalty_nullity=1,
        scale_known=False,
    )
    expected = _numpy_refit_score(np.asarray(params), compression, roots)
    np.testing.assert_allclose(
        result.score, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        jitted.score, result.score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    value_and_grad = jit_gaussian_qr_reml_value_and_grad(
        params,
        compression,
        roots,
        jnp.eye(3),
        jnp.eye(3),
        jnp.eye(3)[:, 1:],
        jnp.asarray(1.7),
        root_sp_indices=(0, 1),
        n_smoothing_parameters=2,
        penalty_nullity=1,
        scale_known=False,
    )
    hessian = jit_gaussian_qr_reml_hessian(
        params,
        compression,
        roots,
        jnp.eye(3),
        jnp.eye(3),
        jnp.eye(3)[:, 1:],
        jnp.asarray(1.7),
        root_sp_indices=(0, 1),
        n_smoothing_parameters=2,
        penalty_nullity=1,
        scale_known=False,
    )
    np.testing.assert_allclose(
        value_and_grad[0], result.score, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert hessian.shape == (3, 3)
    np.testing.assert_allclose(
        result.penalized_residual_sum_squares,
        result.augmented_residual_sum_squares,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert result.augmented_factor.shape == (3, 3)
    assert result.penalty_factor.shape == (2, 2)


def test_shared_reordered_roots_and_nonidentity_public_map() -> None:
    """QR pdet retains the full range when initial root rows overlap."""
    compression, _roots, _x, _y, _weights = _compressed_system()
    roots = (
        jnp.array(((0.0, 1.0, 0.0),)),
        jnp.array(((0.0, 1.0, 0.0),)),
        jnp.array(((0.0, 0.0, 1.0),)),
    )
    # These are intentionally permuted: roots one and two share different
    # declared smoothing slots, and the first two stacked rows are dependent.
    indices = (1, 0, 0)
    angle = 0.41
    coordinate_basis = jnp.array(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    fitting_to_public = jnp.array(((1.0, 0.2, 0.0), (0.0, 1.0, -0.3), (0.1, 0.0, 1.0)))
    public_basis = fitting_to_public @ coordinate_basis
    params = jnp.array((np.log(0.7), np.log(3.2), np.log(1.1)))
    result = gaussian_qr_reml_score(
        params,
        compression,
        roots,
        coordinate_basis,
        public_basis,
        jnp.eye(3)[:, 1:],
        jnp.asarray(1.0),
        root_sp_indices=indices,
        n_smoothing_parameters=2,
        penalty_nullity=1,
        scale_known=False,
    )
    r = np.asarray(compression.R)
    qtz = np.asarray(compression.qtz)
    lambda_zero, lambda_one = np.exp(np.asarray(params[:2]))
    penalty = np.diag((0.0, lambda_zero + lambda_one, lambda_zero))
    original_beta = np.linalg.solve(r.T @ r + penalty, r.T @ qtz)
    expected_public = np.asarray(fitting_to_public) @ original_beta
    np.testing.assert_allclose(
        result.coefficients, expected_public, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.log_det_penalty,
        np.log((lambda_zero + lambda_one) * lambda_zero),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )

    def value(rho: jax.Array) -> jax.Array:
        return gaussian_qr_reml_score(
            rho,
            compression,
            roots,
            coordinate_basis,
            public_basis,
            jnp.eye(3)[:, 1:],
            jnp.asarray(1.0),
            root_sp_indices=indices,
            n_smoothing_parameters=2,
            penalty_nullity=1,
            scale_known=False,
        ).score

    assert jax.hessian(value)(params).shape == (3, 3)


def test_frozen_qr_score_matches_current_dense_reml_at_its_refit() -> None:
    """The QR kernel preserves the current dense REML arithmetic at a fit."""
    compression, roots, _x, _y, _weights = _compressed_system()
    params = jnp.array((np.log(1.8), np.log(0.45), np.log(0.9)))
    result = _score(params, compression, roots)
    structure = JaxPenaltyStructure(
        3,
        (
            JaxPenaltyBlock(
                1,
                2,
                (0,),
                (JaxLocalPenalty("identity", jnp.array(1.0), 1),),
                JaxTransform("identity", jnp.array(1.0), 1),
            ),
            JaxPenaltyBlock(
                2,
                3,
                (1,),
                (JaxLocalPenalty("identity", jnp.array(1.0), 1),),
                JaxTransform("identity", jnp.array(1.0), 1),
            ),
        ),
    )
    dense_score = reml_criterion(
        params[:2],
        compression.R.T @ compression.R,
        result.reduced_coefficients,
        result.residual_sum_squares,
        -compression.n_positive_weights * jnp.log(2.0 * jnp.pi * result.scale) / 2.0
        + compression.sum_log_positive_weights / 2.0,
        structure,
        result.scale,
        1,
        singleton_sp_indices=(0, 1),
        singleton_ranks=(1, 1),
        singleton_eig_constants=jnp.zeros(2),
        multi_block_sp_indices=(),
        multi_block_ranks=(),
        multi_block_proj_S=(),
    )
    np.testing.assert_allclose(
        result.score, dense_score, rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_frozen_gaussian_qr_gradient_and_hessian_match_refits() -> None:
    compression, roots, _x, _y, _weights = _compressed_system()
    params = jnp.array((np.log(1.2), np.log(0.7), np.log(1.1)))

    def value(rho: jax.Array) -> jax.Array:
        return _score(rho, compression, roots).score

    gradient = np.asarray(jax.grad(value)(params))
    hessian = np.asarray(jax.hessian(value)(params))
    # Five-point diagonal differences at this step are within 3e-8 of AD in
    # this frozen refit, so the project MODERATE cross-implementation class is
    # a conservative gate rather than an ad-hoc Hessian allowance.
    step = 5e-4
    params_np = np.asarray(params)
    numerical_gradient = np.empty_like(params_np)
    numerical_hessian = np.empty((len(params_np), len(params_np)))
    center = _numpy_refit_score(params_np, compression, roots)
    for i in range(len(params_np)):
        direction_i = np.zeros_like(params_np)
        direction_i[i] = step
        numerical_gradient[i] = (
            _numpy_refit_score(params_np + direction_i, compression, roots)
            - _numpy_refit_score(params_np - direction_i, compression, roots)
        ) / (2.0 * step)
        numerical_hessian[i, i] = (
            -_numpy_refit_score(params_np + 2.0 * direction_i, compression, roots)
            + 16.0 * _numpy_refit_score(params_np + direction_i, compression, roots)
            - 30.0 * center
            + 16.0 * _numpy_refit_score(params_np - direction_i, compression, roots)
            - _numpy_refit_score(params_np - 2.0 * direction_i, compression, roots)
        ) / (12.0 * step**2)
        for j in range(i):
            direction_j = np.zeros_like(params_np)
            direction_j[j] = step
            numerical_hessian[i, j] = numerical_hessian[j, i] = (
                _numpy_refit_score(
                    params_np + direction_i + direction_j, compression, roots
                )
                - _numpy_refit_score(
                    params_np + direction_i - direction_j, compression, roots
                )
                - _numpy_refit_score(
                    params_np - direction_i + direction_j, compression, roots
                )
                + _numpy_refit_score(
                    params_np - direction_i - direction_j, compression, roots
                )
            ) / (4.0 * step**2)
    np.testing.assert_allclose(
        gradient, numerical_gradient, rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        hessian, numerical_hessian, rtol=MODERATE.rtol, atol=MODERATE.atol
    )


def test_zero_weight_offset_metadata_and_near_perfect_tail_are_preserved() -> None:
    weights = np.array(
        (1.0, 0.0, 0.4, 1.7, 0.0, 0.8, 1.1, 0.3, 1.4, 1.0, 0.5, 0.0, 2.0)
    )
    offset = np.linspace(-0.4, 0.3, len(weights))
    compression, roots, _x, _y, _weights = _compressed_system(
        weights=weights, offset=offset, noise_scale=1e-12
    )
    result = _score(jnp.array((np.log(1e-10), np.log(1e-10), 0.0)), compression, roots)
    assert int(result.n_obs) == len(weights)
    assert int(result.n_positive_weights) == np.count_nonzero(weights)
    assert 0.0 < compression.residual_tail < 1e-20
    assert result.residual_sum_squares >= compression.residual_tail
    assert result.penalty_quadratic >= 0.0


def test_rectangular_compression_is_identified_by_penalty_rows() -> None:
    compression = FrozenGaussianQR(
        R=jnp.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        qtz=jnp.array((2.0, -1.0)),
        residual_tail=jnp.array(1e-24),
        n_obs=jnp.array(2),
        n_positive_weights=jnp.array(2),
        sum_log_positive_weights=jnp.array(0.0),
    )
    roots = (jnp.array(((0.0, 0.0, 1.0),)),)
    result = gaussian_qr_reml_score(
        jnp.array((np.log(2.0),)),
        compression,
        roots,
        jnp.eye(3),
        jnp.eye(3),
        jnp.array(((0.0,), (0.0,), (1.0,))),
        jnp.array(1.0),
        root_sp_indices=(0,),
        n_smoothing_parameters=1,
        penalty_nullity=2,
        scale_known=True,
    )
    np.testing.assert_allclose(
        result.coefficients, (2.0, -1.0, 0.0), rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.residual_sum_squares,
        compression.residual_tail,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize("index", [-1, 2])
def test_invalid_declared_smoothing_index_is_rejected(index: int) -> None:
    compression, roots, _x, _y, _weights = _compressed_system()
    with pytest.raises(ValueError, match="declared range"):
        gaussian_qr_reml_score(
            jnp.zeros(3),
            compression,
            roots,
            jnp.eye(3),
            jnp.eye(3),
            jnp.eye(3)[:, 1:],
            jnp.asarray(1.0),
            root_sp_indices=(index, 1),
            n_smoothing_parameters=2,
            penalty_nullity=1,
            scale_known=False,
        )


def test_underidentified_augmented_layout_is_rejected() -> None:
    compression = FrozenGaussianQR(
        R=jnp.array(((1.0, 0.0, 0.0),)),
        qtz=jnp.array((1.0,)),
        residual_tail=jnp.array(0.0),
        n_obs=jnp.array(1),
        n_positive_weights=jnp.array(1),
        sum_log_positive_weights=jnp.array(0.0),
    )
    with pytest.raises(ValueError, match="cannot identify"):
        gaussian_qr_reml_score(
            jnp.array((0.0,)),
            compression,
            (jnp.array(((0.0, 1.0, 0.0),)),),
            jnp.eye(3),
            jnp.eye(3),
            jnp.array(((0.0,), (1.0,), (0.0,))),
            jnp.array(1.0),
            root_sp_indices=(0,),
            n_smoothing_parameters=1,
            penalty_nullity=2,
            scale_known=True,
        )


def test_zero_penalty_qr_supports_a_full_rank_gaussian_data_system() -> None:
    """This internal kernel accepts the no-smooth Gaussian base case only."""
    compression, _roots, _x, _y, _weights = _compressed_system()
    result = gaussian_qr_reml_score(
        jnp.array((np.log(0.9),)),
        compression,
        (),
        jnp.eye(3),
        jnp.eye(3),
        jnp.empty((3, 0)),
        jnp.asarray(1.0),
        root_sp_indices=(),
        n_smoothing_parameters=0,
        penalty_nullity=3,
        scale_known=False,
    )
    assert result.penalty_factor.shape == (0, 0)
    np.testing.assert_allclose(result.penalty_quadratic, 0.0, atol=STRICT.atol)


def test_pinned_r_refit_finite_differences_match_gradient_and_hessian() -> None:
    """Pin an independent fixed-rho QR refit and its finite differences to R."""
    from tests.r_bridge import RBridge

    bridge = RBridge()
    ok, reason = bridge.check_versions()
    if not ok or bridge.mode != "rpy2":
        pytest.skip(reason if not ok else "rpy2 required for fixed-rho QR oracle")
    compression, roots, _x, _y, _weights = _compressed_system()
    params = jnp.array((np.log(1.8), np.log(0.45), np.log(0.9)))

    def r_refit(value: np.ndarray) -> float:
        return bridge.source_gaussian_fixed_rho_qr_refit(
            np.asarray(compression.R),
            np.asarray(compression.qtz),
            float(np.asarray(compression.residual_tail)),
            value[:2],
            float(np.exp(value[2])),
            int(np.asarray(compression.n_positive_weights)),
            float(np.asarray(compression.sum_log_positive_weights)),
        )

    params_np = np.asarray(params)
    expected = r_refit(params_np)
    np.testing.assert_allclose(
        _score(params, compression, roots).score,
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    step = 5e-4
    r_gradient = np.empty_like(params_np)
    r_hessian = np.empty((len(params_np), len(params_np)))
    for i in range(len(params_np)):
        direction_i = np.zeros_like(params_np)
        direction_i[i] = step
        r_gradient[i] = (
            r_refit(params_np + direction_i) - r_refit(params_np - direction_i)
        ) / (2.0 * step)
        r_hessian[i, i] = (
            -r_refit(params_np + 2.0 * direction_i)
            + 16.0 * r_refit(params_np + direction_i)
            - 30.0 * expected
            + 16.0 * r_refit(params_np - direction_i)
            - r_refit(params_np - 2.0 * direction_i)
        ) / (12.0 * step**2)
        for j in range(i):
            direction_j = np.zeros_like(params_np)
            direction_j[j] = step
            r_hessian[i, j] = r_hessian[j, i] = (
                r_refit(params_np + direction_i + direction_j)
                - r_refit(params_np + direction_i - direction_j)
                - r_refit(params_np - direction_i + direction_j)
                + r_refit(params_np - direction_i - direction_j)
            ) / (4.0 * step**2)

    def value(rho: jax.Array) -> jax.Array:
        return _score(rho, compression, roots).score

    np.testing.assert_allclose(
        jax.grad(value)(params), r_gradient, rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        jax.hessian(value)(params), r_hessian, rtol=MODERATE.rtol, atol=MODERATE.atol
    )


def _compact_declared_root(root: np.ndarray, rank: int) -> np.ndarray:
    """Keep the declared rows of an R square-root penalty.

    ``gam.reparam`` can export a square factor whose nominal null rows carry
    round-off-scale singular values.  Those rows are harmless near the
    preparation rho but are amplified when another smoothing parameter
    dominates.  The QR kernel's input contract is a compact root, so reduce
    this once during the host-side frozen-layout preparation.
    """
    _left, singular_values, right_vectors = np.linalg.svd(root, full_matrices=False)
    discarded = singular_values[rank:]
    if discarded.size:
        assert discarded[0] <= singular_values[0] * 1e-7
    return singular_values[:rank, None] * right_vectors[:rank]


def test_pinned_sl_fit_tensor_score_matches_frozen_qr_kernel() -> None:
    """Match a real weighted, offset, unknown-scale tensor ``Sl.fit`` score.

    The R side freezes its data QR, coordinate maps, and compact component
    roots at one rho, then compares independent ``Sl.fit`` refits across
    balanced and strongly-dominant tensor smoothing pairs.  This is a fixed
    statistic kernel test, not a claim that this slice supplies the later
    public rank-map/optimizer route.
    """
    from tests.r_bridge import RBridge

    bridge = RBridge()
    ok, reason = bridge.check_versions()
    if not ok or bridge.mode != "rpy2":
        pytest.skip(reason if not ok else "rpy2 required for Sl.fit oracle")
    points = (
        (-2.0, 1.0, np.log(0.8)),
        (-12.0, 10.0, np.log(0.8)),
        (10.0, -12.0, np.log(1.4)),
        (0.3, -0.7, np.log(0.4)),
    )
    frozen = bridge.source_gaussian_tensor_sl_fit(
        np.asarray(points[0][:2]), points[0][2]
    )
    r_factor = frozen["R"]
    root = frozen["root"]
    penalty_range = np.linalg.qr(root.T, mode="reduced")[0]
    compression = FrozenGaussianQR(
        R=jnp.asarray(r_factor),
        qtz=jnp.asarray(frozen["qtz"]),
        residual_tail=jnp.asarray(float(frozen["residual_tail"])),
        n_obs=jnp.asarray(int(frozen["n_obs"])),
        n_positive_weights=jnp.asarray(int(frozen["n_positive_weights"])),
        sum_log_positive_weights=jnp.asarray(float(frozen["sum_log_positive_weights"])),
    )
    matched_result = gaussian_qr_reml_score(
        jnp.array((0.0, points[0][2])),
        compression,
        (jnp.asarray(root),),
        jnp.eye(r_factor.shape[1]),
        jnp.eye(r_factor.shape[1]),
        jnp.asarray(penalty_range),
        jnp.asarray(1.0),
        root_sp_indices=(0,),
        n_smoothing_parameters=1,
        penalty_nullity=int(frozen["penalty_nullity"]),
        scale_known=False,
    )
    # Sl.fit's fREML expression omits Gaussian's prior-weight normalizer.
    # Current JaxGAM's saturated likelihood retains it, so the two fixed-rho
    # scores differ by this documented data-only constant.
    normalizer_correction = float(frozen["sum_log_positive_weights"]) / 2.0
    expected_score = float(frozen["score"]) - normalizer_correction
    np.testing.assert_allclose(
        matched_result.score,
        expected_score,
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )

    roots = tuple(
        _compact_declared_root(value, rank)
        for value, rank in zip(
            (value for value in frozen["roots"]),
            np.asarray(frozen["root_ranks"], dtype=int),
            strict=True,
        )
    )

    def value(rho: jax.Array) -> jax.Array:
        return gaussian_qr_reml_score(
            rho,
            compression,
            tuple(jnp.asarray(item) for item in roots),
            jnp.eye(r_factor.shape[1]),
            jnp.eye(r_factor.shape[1]),
            jnp.asarray(penalty_range),
            jnp.asarray(1.0),
            root_sp_indices=(0, 1),
            n_smoothing_parameters=2,
            penalty_nullity=int(frozen["penalty_nullity"]),
            scale_known=False,
        ).score

    for point in points:
        reference = bridge.source_gaussian_tensor_sl_fit(
            np.asarray(point[:2]), point[2]
        )
        np.testing.assert_array_equal(reference["R"], r_factor)
        params = jnp.asarray(point)
        result = gaussian_qr_reml_score(
            params,
            compression,
            tuple(jnp.asarray(item) for item in roots),
            jnp.eye(r_factor.shape[1]),
            jnp.eye(r_factor.shape[1]),
            jnp.asarray(penalty_range),
            jnp.asarray(1.0),
            root_sp_indices=(0, 1),
            n_smoothing_parameters=2,
            penalty_nullity=int(reference["penalty_nullity"]),
            scale_known=False,
        )
        expected_score = float(reference["score"]) - normalizer_correction
        np.testing.assert_allclose(
            result.score, expected_score, rtol=MODERATE.rtol, atol=MODERATE.atol
        )
        np.testing.assert_allclose(
            jax.grad(value)(params),
            np.asarray(reference["gradient"], dtype=float),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        np.testing.assert_allclose(
            jax.hessian(value)(params),
            reference["hessian"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        np.testing.assert_allclose(
            r_factor @ result.coefficients,
            np.asarray(reference["prediction"], dtype=float),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
