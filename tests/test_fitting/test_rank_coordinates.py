"""Rank-coordinate Gaussian QR tests.

The fixture is intentionally singular in the full coordinates.  Its selected
minor has a different determinant from the orthonormal pseudo-determinant,
which catches the original rank-count-only implementation error.
"""

from __future__ import annotations

import tracemalloc

import jax
import numpy as np
import pytest

from jaxgam.fitting.rank_coordinates import jit_reduced_augmented_qr
from jaxgam.formula.fitting_prepare import (
    _source_penalty_frobenius_norm,
    global_penalty_coordinates,
)
from jaxgam.formula.rank_coordinates import (
    IdentifiableCoordinateMap,
    global_rank_coordinate_map,
    rank_revealing_coordinate_map,
)
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    DiagonalPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
from tests.r_bridge import RBridge
from tests.tolerances import STRICT

# Outcome can depend on the CPU's SIMD maths or BLAS kernel family.
pytestmark = pytest.mark.platform_sensitive


def test_global_penalty_scale_matches_live_pinned_r_sum() -> None:
    """A last-bit scale difference can rotate tied global penalty vectors."""
    ok, reason = RBridge.check_versions()
    if not ok:
        pytest.skip(reason)
    random = np.random.default_rng(1909)
    matrices = [random.normal(size=(24, 24)) for _ in range(2)]
    expected = RBridge().source_penalty_frobenius_norms(tuple(matrices))
    actual = np.array([_source_penalty_frobenius_norm(m) for m in matrices])
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)


def test_global_penalty_scale_bounds_scratch_for_wide_penalty() -> None:
    """A wide coefficient block must not allocate another full matrix."""
    penalty = np.ones((513, 513), dtype=np.float64)
    tracemalloc.start()
    try:
        observed = _source_penalty_frobenius_norm(penalty)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert observed == 513.0
    assert peak < 1_000_000


def _singular_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return data and a rank-one penalty with one shared null alias."""
    x = np.linspace(-1.0, 1.0, 19)
    data = np.column_stack((np.ones_like(x), x, x, x**2))
    target = 0.3 + 1.1 * x - 0.4 * x**2
    root = np.array(((0.0, 0.0, 0.0, 1.0),))
    # This orthogonal map stands in for mgcv's global U1 %*% Qs map.  It is
    # intentionally separate from the reduced-coordinate basis below.
    transform = np.array(
        (
            (0.0, 1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 0.0),
        )
    )
    return data, target, root, transform


def test_reduced_qr_reconstructs_coefficients_and_covariance_through_one_map() -> None:
    data, target, root, transform = _singular_system()
    # gdiPK drops the aliased third fitting coordinate.  The reduced basis and
    # public map are supplied together; a local-D map cannot be substituted.
    coordinate_map = IdentifiableCoordinateMap(
        basis=np.eye(4)[:, (0, 1, 3)],
        fitting_to_public=transform,
        keep=np.array((0, 1, 3)),
        drop=np.array((2,)),
    )
    log_lambda = jax.numpy.array((np.log(2.3),))
    beta, covariance, factor, logdet = map(
        np.asarray,
        jit_reduced_augmented_qr(
            data,
            target,
            (root,),
            log_lambda,
            (0,),
            coordinate_map.basis,
            coordinate_map.public_basis,
        ),
    )

    keep = coordinate_map.keep
    hessian = data.T @ data + 2.3 * root.T @ root
    rhs = data.T @ target
    expected_reduced_beta = np.linalg.solve(hessian[np.ix_(keep, keep)], rhs[keep])
    expected_reduced_covariance = np.linalg.inv(hessian[np.ix_(keep, keep)])
    expected_public_basis = transform[:, keep]
    np.testing.assert_allclose(
        beta,
        expected_public_basis @ expected_reduced_beta,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        covariance,
        expected_public_basis @ expected_reduced_covariance @ expected_public_basis.T,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        factor.T @ factor,
        hessian[np.ix_(keep, keep)],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        logdet,
        np.linalg.slogdet(hessian[np.ix_(keep, keep)])[1],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )

    nonzero_eigenvalues = np.linalg.eigvalsh(hessian)[1:]
    assert not np.isclose(logdet, np.log(nonzero_eigenvalues).sum())


def test_rank_revealing_map_uses_the_frozen_balanced_layout() -> None:
    data, _target, root, transform = _singular_system()
    coordinate_map = rank_revealing_coordinate_map(
        data,
        root,
        retained_rank=3,
        fitting_to_public=transform,
    )
    np.testing.assert_array_equal(
        np.sort(np.concatenate((coordinate_map.keep, coordinate_map.drop))),
        np.arange(4),
    )
    assert coordinate_map.n_reduced == 3
    with np.testing.assert_raises(ValueError):
        coordinate_map.basis[0, 0] = 0.0
    with np.testing.assert_raises(ValueError):
        coordinate_map.fitting_to_public[0, 0] = 0.0


def test_jitted_reduced_qr_accepts_a_new_global_map_for_each_lambda_state() -> None:
    """The compiled kernel does not capture the first ``U1 %*% Qs`` map.

    ``gam.reparam`` makes ``Qs`` smoothing-parameter dependent.  A caller
    therefore supplies the current transformed design, roots, and public map
    together for every score evaluation.  The kernel may reuse compilation,
    but cannot reuse a map selected at the initial smoothing parameter.
    """
    data, target, root, _transform = _singular_system()
    keep = np.array((0, 1, 3))
    basis = np.eye(4)[:, keep]
    angle = 0.37
    transform = np.array(
        (
            (np.cos(angle), -np.sin(angle), 0.0, 0.0),
            (np.sin(angle), np.cos(angle), 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    params = jax.numpy.array((np.log(2.3),))
    first = jit_reduced_augmented_qr(data, target, (root,), params, (0,), basis, basis)
    second = jit_reduced_augmented_qr(
        data @ transform,
        target,
        (root @ transform,),
        params,
        (0,),
        basis,
        transform @ basis,
    )
    np.testing.assert_allclose(
        np.asarray(first[0]), np.asarray(second[0]), rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        np.asarray(first[1]), np.asarray(second[1]), rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_global_penalty_map_reconstructs_in_public_coordinates() -> None:
    """The U1 route, rather than a local D map, owns public covariance."""
    data, target, root, _transform = _singular_system()
    structure = PenaltyStructure(
        4,
        (
            PenaltyBlock(
                0,
                4,
                (0,),
                (DiagonalPenalty(np.array((0.0, 0.0, 0.0, 1.0))),),
                IdentityTransform(4),
                (1,),
            ),
        ),
    )
    coordinate_map, coordinates = global_rank_coordinate_map(
        data, structure, np.array((np.log(2.3),)), retained_rank=3
    )
    beta, covariance, factor, logdet = map(
        np.asarray,
        jit_reduced_augmented_qr(
            data @ coordinates.transform,
            target,
            tuple(root_row for root_row in coordinates.penalty_roots),
            np.array((np.log(2.3),)),
            (0,),
            coordinate_map.basis,
            coordinate_map.public_basis,
        ),
    )
    hessian = data.T @ data + 2.3 * root.T @ root
    transformed_hessian = coordinates.transform.T @ hessian @ coordinates.transform
    transformed_rhs = coordinates.transform.T @ data.T @ target
    keep = coordinate_map.keep
    reduced_beta = np.linalg.solve(
        transformed_hessian[np.ix_(keep, keep)], transformed_rhs[keep]
    )
    reduced_covariance = np.linalg.inv(transformed_hessian[np.ix_(keep, keep)])
    expected_public_basis = coordinates.transform[:, keep]
    np.testing.assert_allclose(
        beta,
        expected_public_basis @ reduced_beta,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        covariance,
        expected_public_basis @ reduced_covariance @ expected_public_basis.T,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        factor.T @ factor,
        transformed_hessian[np.ix_(keep, keep)],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        logdet,
        np.linalg.slogdet(transformed_hessian[np.ix_(keep, keep)])[1],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_global_qs_rotation_transforms_each_overlapping_penalty() -> None:
    """The C_get_stableS port preserves the full lambda-weighted penalty."""
    first = np.array(
        (
            (4.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    second = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 2.0),
            (0.0, 2.0, 9.0),
        )
    )
    structure = PenaltyStructure(
        3,
        (
            PenaltyBlock(
                0,
                3,
                (0, 1),
                (DenseLocalPenalty(first), DenseLocalPenalty(second)),
                IdentityTransform(3),
                (2, 2),
            ),
        ),
    )
    log_lambda = np.array((-6.0, 3.0))
    coordinates = global_penalty_coordinates(structure, log_lambda)
    transformed = sum(
        np.exp(log_lambda[index]) * root.T @ root
        for index, root in enumerate(coordinates.penalty_roots)
    )
    expected = (
        coordinates.transform.T
        @ (np.exp(log_lambda[0]) * first + np.exp(log_lambda[1]) * second)
        @ coordinates.transform
    )
    np.testing.assert_allclose(
        coordinates.transform.T @ coordinates.transform,
        np.eye(3),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        transformed, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    with np.testing.assert_raises(ValueError):
        coordinates.transform[0, 0] = 0.0
    with np.testing.assert_raises(ValueError):
        coordinates.penalty_roots[0][0, 0] = 0.0
    with np.testing.assert_raises_regex(ValueError, "finite"):
        global_penalty_coordinates(structure, np.array((np.nan, 0.0)))
    with np.testing.assert_raises_regex(ValueError, "positive"):
        global_penalty_coordinates(structure, np.array((-1000.0, 0.0)))
