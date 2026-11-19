"""Pinned mgcv public covariance gate for the Python global rank map."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from jaxgam.families.standard import Gaussian
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.rank_coordinates import jit_reduced_augmented_qr
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.fitting_prepare import GlobalPenaltyCoordinates
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.rank_coordinates import (
    IdentifiableCoordinateMap,
    global_rank_coordinate_map,
)
from tests.helpers import _AssertCollector, check_that
from tests.r_bridge import RBridge
from tests.test_validation_matrix import SMOOTH_CONFIGS, _get_data
from tests.tolerances import MODERATE, STRICT

# Outcome can depend on the CPU's SIMD maths or BLAS kernel family.
pytestmark = pytest.mark.platform_sensitive


def _pinned_public_gaussian_rank_fit(
    bridge: RBridge,
    data: object,
    formula: str,
    smoothing_parameter: object,
    weights: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return pinned ``gam.fit3`` public coefficients and covariance."""
    result = bridge.source_gaussian_rank_fit(
        formula, data, np.atleast_1d(smoothing_parameter), weights, offset
    )
    return (
        result["coefficients"],
        result["covariance"],
        result["score"],
    )


def _gaussian_reml_score(
    beta: np.ndarray,
    logdet_h: float,
    coordinates: object,
    data: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    log_lambda: np.ndarray,
) -> float:
    """Pinned ``gam.fit3`` Gaussian REML scalar in its selected coordinates."""
    fitting_beta = coordinates.transform.T @ beta
    penalty = sum(
        np.exp(log_lambda[index]) * np.sum((root @ fitting_beta) ** 2)
        for index, root in enumerate(coordinates.penalty_roots)
    )
    residual = target - data @ beta
    n_range = coordinates.balanced_root.shape[0]
    stable_penalty = sum(
        np.exp(log_lambda[index]) * root[:, :n_range].T @ root[:, :n_range]
        for index, root in enumerate(coordinates.penalty_roots)
    )
    sign, logdet_s = np.linalg.slogdet(stable_penalty)
    assert sign > 0
    ls_sat = -len(target) * np.log(2.0 * np.pi) / 2.0 + np.log(weights).sum() / 2.0
    mp = coordinates.transform.shape[0] - n_range
    return (
        (np.sum(weights * residual**2) + penalty) / 2.0
        - ls_sat
        + logdet_h / 2.0
        - logdet_s / 2.0
        - mp * np.log(2.0 * np.pi) / 2.0
    )


def _layout_cut_ratio(
    design: np.ndarray, balanced_root: np.ndarray, retained: int
) -> float:
    """|R_rr| / |R_(r-1)(r-1)| of gdiPK's stacked layout QR at the rank cut."""
    from jaxgam.smooths.source_lapack import source_frobenius_norm, source_pivoted_qr

    p = design.shape[1]
    data_r_pivoted, data_pivots = source_pivoted_qr(np.asarray(design, order="F"))
    inverse = np.empty_like(data_pivots)
    inverse[data_pivots] = np.arange(p)
    data_r = data_r_pivoted[:, inverse]
    layout = np.vstack(
        (
            data_r / source_frobenius_norm(data_r),
            balanced_root / source_frobenius_norm(balanced_root),
        )
    )
    upper, _ = source_pivoted_qr(np.asarray(layout, order="F"))
    diagonal = np.abs(np.diag(upper))
    return float(diagonal[retained] / diagonal[retained - 1])


def _retained_projector(transform: np.ndarray, keep: np.ndarray) -> np.ndarray:
    kept = np.asarray(transform)[:, np.asarray(keep)]
    return kept @ kept.T


def _check_rank_gauge(
    collector: _AssertCollector,
    *,
    design: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    structure: object,
    log_lambda: np.ndarray,
    sp_indices: tuple[int, ...],
    retained: int,
    r_fit: dict,
    gauge_tolerance: object,
    score_tolerance: object,
) -> str:
    """Gate a rank-deficient fit on the tie-replay rule; return the branch.

    Which aliased coordinate gdiPK zeroes is decided by roundoff (the basis
    DSYEVR picks inside degenerate penalty eigenspaces and the stacked QR's
    tied pivots), and pinned R's own choice moves with the CPU and BLAS. The
    statistical fit does not depend on it:

    1. always: the gauge-invariant fit at STRICT (fitted values, linear-
       predictor covariance X V X', EDF, deviance and the scale estimate);
    2. if JaxGAM retains the same public subspace as R: coefficients,
       covariance and the REML score (which mgcv computes on the retained
       coordinates, so it is gauge-dependent) at the test's tolerances;
    3. otherwise: the decision must be a rounding-level tie (the stacked-QR
       diagonal at the cut, relative to the last retained one, below STRICT's
       rtol, and the same penalty range and null spaces as R), and JaxGAM's
       solve replayed in R's own fitting coordinates and dropped set must give
       every field at the test's tolerances. The replay is built here from
       R's state; no production route can reach it.
    """
    sqrt_weights = np.sqrt(weights)
    weighted = sqrt_weights[:, None] * design

    def solve(coordinate_map, coordinates):
        beta, covariance, _factor, logdet = map(
            np.asarray,
            jit_reduced_augmented_qr(
                weighted @ coordinates.transform,
                sqrt_weights * target,
                tuple(coordinates.penalty_roots),
                log_lambda,
                sp_indices,
                coordinate_map.basis,
                coordinate_map.public_basis,
            ),
        )
        score = _gaussian_reml_score(
            beta, float(logdet), coordinates, design, target, weights, log_lambda
        )
        return beta, covariance, score

    coordinate_map, coordinates = global_rank_coordinate_map(
        weighted, structure, log_lambda, retained_rank=retained
    )
    beta, covariance, score = solve(coordinate_map, coordinates)
    X_r = r_fit["X"]
    gram = design.T @ (weights[:, None] * design)
    gram_r = X_r.T @ (weights[:, None] * X_r)
    for name, observed, expected in (
        (
            "fitted values (gauge-invariant)",
            design @ beta + (r_fit["fitted_values"] - X_r @ r_fit["coefficients"]),
            r_fit["fitted_values"],
        ),
        (
            "linear-predictor covariance (gauge-invariant)",
            design @ covariance @ design.T,
            X_r @ r_fit["covariance"] @ X_r.T,
        ),
        ("EDF (gauge-invariant)", np.trace(covariance @ gram), r_fit["edf"]),
        (
            "EDF against R's covariance",
            np.trace(covariance @ gram),
            np.trace(r_fit["covariance"] @ gram_r),
        ),
        (
            "deviance (gauge-invariant)",
            np.sum(weights * (target - design @ beta) ** 2),
            r_fit["deviance"],
        ),
        (
            "scale estimate (gauge-invariant)",
            np.sum(weights * (target - design @ beta) ** 2)
            / (len(target) - np.trace(covariance @ gram)),
            r_fit["scale"],
        ),
    ):
        collector.check(
            name,
            lambda observed=observed, expected=expected: np.testing.assert_allclose(
                observed, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    q = coordinates.transform.shape[0]
    drop_r = np.asarray(r_fit["drop"], dtype=np.int64)
    keep_r = np.setdiff1d(np.arange(q), drop_r)
    collector.check(
        "R's dropped set has the rank deficit",
        lambda: check_that(
            len(drop_r) == q - retained,
            f"R drops {list(drop_r)} for retained rank {retained} of {q}",
        ),
    )
    same_gauge = np.allclose(
        _retained_projector(coordinates.transform, coordinate_map.keep),
        _retained_projector(r_fit["transform"], keep_r),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )

    def check_gauge_fields(prefix, observed_beta, observed_covariance, observed):
        for name, value, expected, tolerance in (
            (
                "public coefficients",
                observed_beta,
                r_fit["coefficients"],
                gauge_tolerance,
            ),
            (
                "public covariance",
                observed_covariance,
                r_fit["covariance"],
                gauge_tolerance,
            ),
            ("REML scalar", observed, r_fit["score"], score_tolerance),
        ):
            collector.check(
                f"{prefix}{name}",
                lambda value=value, expected=expected, tolerance=tolerance: (
                    np.testing.assert_allclose(
                        value, expected, rtol=tolerance.rtol, atol=tolerance.atol
                    )
                ),
            )

    if same_gauge:
        check_gauge_fields("", beta, covariance, score)
        return "same retained subspace as R"
    n_range = coordinates.balanced_root.shape[0]
    n_range_r = r_fit["balanced_root"].shape[0]
    ratio = _layout_cut_ratio(
        weighted @ coordinates.transform, coordinates.balanced_root, retained
    )
    collector.check(
        "gauge decision is a rounding-level tie",
        lambda: check_that(
            ratio <= STRICT.rtol and n_range == n_range_r,
            f"stacked-QR cut ratio {ratio:.3e}; range ranks {n_range}/{n_range_r}",
        ),
    )
    for name, columns, columns_r in (
        (
            "penalty range space",
            coordinates.penalty_space[:, :n_range],
            r_fit["penalty_space"][:, :n_range_r],
        ),
        (
            "penalty null space",
            coordinates.penalty_space[:, n_range:],
            r_fit["penalty_space"][:, n_range_r:],
        ),
    ):
        collector.check(
            f"{name} equals R's",
            lambda columns=columns, columns_r=columns_r: np.testing.assert_allclose(
                columns @ columns.T,
                columns_r @ columns_r.T,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
    replay_coordinates = GlobalPenaltyCoordinates(
        r_fit["transform"],
        r_fit["penalty_space"],
        r_fit["balanced_root"],
        tuple(root.T for root in r_fit["fitting_roots"]),
    )
    replay_map = IdentifiableCoordinateMap(
        np.eye(q)[:, keep_r], r_fit["transform"], keep_r, drop_r
    )
    check_gauge_fields("replayed ", *solve(replay_map, replay_coordinates))
    return "rounding-level tie replayed in R's coordinates"


@pytest.mark.parametrize(
    "smoothing_parameter", [(1e-4,), (0.0013187722550607138,), (1.0,)]
)
def test_python_global_rank_map_matches_pinned_public_gp_covariance(
    smoothing_parameter: tuple[float],
) -> None:
    """Exercise U1/Qs construction, selected minor, and public reconstruction."""
    smoothing_parameter = smoothing_parameter[0]
    bridge = RBridge()
    ok, reason = bridge.check_versions()
    if not ok:
        pytest.skip(reason)
    data = _get_data(SMOOTH_CONFIGS["gp_mixed"], "gaussian")
    formula = "y ~ x + s(x, bs='gp')"
    weights = 0.3 + (np.arange(len(data)) % 9) / 7.0
    offset = 0.2 * np.sin(np.arange(len(data)) / 11.0)
    setup = ModelSetup.build(
        parse_formula(formula), data.to_dict("list"), weights=weights, offset=offset
    )
    fitting = FittingData.from_setup(setup, Gaussian())
    assert setup.penalties is not None
    target = np.asarray(setup.y) - offset
    r_setup = bridge.source_gaussian_rank_fit(
        formula, data, smoothing_parameter, weights, offset
    )
    collector = _AssertCollector()
    collector.check(
        "public setup X",
        lambda: np.testing.assert_allclose(
            setup.X, r_setup["X"], rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "public setup S",
        lambda: np.testing.assert_allclose(
            setup.penalties.blocks[0].dense_penalties()[0],
            r_setup["S"][0],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    branch = _check_rank_gauge(
        collector,
        design=np.asarray(setup.X),
        weights=weights,
        target=target,
        structure=setup.penalties,
        log_lambda=np.array((np.log(smoothing_parameter),)),
        sp_indices=(0,),
        retained=fitting.n_coef - fitting.rank_deficit,
        r_fit=r_setup,
        gauge_tolerance=MODERATE,
        score_tolerance=MODERATE,
    )
    warnings.warn(f"weighted GP rank-map gate branch: {branch}", stacklevel=1)
    collector.raise_if_any(f"pinned weighted GP rank map [{branch}]")


@pytest.mark.parametrize(
    "smoothing_parameters", [(np.array((1e-6, 20.0)),), (np.array((20.0, 1e-6)),)]
)
def test_python_global_rank_map_matches_pinned_public_coupled_covariance(
    smoothing_parameters: tuple[np.ndarray],
) -> None:
    """The range-space Qs rotation handles overlapping tensor penalties."""
    bridge = RBridge()
    ok, reason = bridge.check_versions()
    if not ok:
        pytest.skip(reason)
    config = SMOOTH_CONFIGS["te"]
    data = _get_data(config, "gaussian")
    setup = ModelSetup.build(parse_formula(config.py_formula), data.to_dict("list"))
    fitting = FittingData.from_setup(setup, Gaussian())
    assert setup.penalties is not None
    smoothing_parameters = smoothing_parameters[0]
    coordinate_map, coordinates = global_rank_coordinate_map(
        np.asarray(setup.X),
        setup.penalties,
        np.log(smoothing_parameters),
        retained_rank=fitting.n_coef - fitting.rank_deficit,
    )
    beta, covariance, _factor, _logdet = map(
        np.asarray,
        jit_reduced_augmented_qr(
            np.asarray(setup.X) @ coordinates.transform,
            np.asarray(setup.y),
            tuple(coordinates.penalty_roots),
            np.log(smoothing_parameters),
            tuple(range(fitting.n_penalties)),
            coordinate_map.basis,
            coordinate_map.public_basis,
        ),
    )
    (
        expected_beta,
        expected_covariance,
        _expected_score,
    ) = _pinned_public_gaussian_rank_fit(
        bridge,
        data,
        config.r_formula,
        smoothing_parameters,
        np.ones(len(data)),
        np.zeros(len(data)),
    )
    collector = _AssertCollector()
    collector.check(
        "public coefficients",
        lambda: np.testing.assert_allclose(
            beta, expected_beta, rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        "public covariance",
        lambda: np.testing.assert_allclose(
            covariance,
            expected_covariance,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any("pinned coupled rank map")
    assert not np.allclose(coordinates.transform, coordinates.penalty_space)


def test_coupled_alias_uses_pinned_public_rank_gauge() -> None:
    """A tied tensor null space: gauge-invariant fit, then R's gauge or a replay.

    x1 and x2 alias te's linear null functions, and which pair is zeroed is
    decided by the basis DSYEVR picks inside the six-dimensional penalty null
    space. Pinned R's own choice moves with the CPU (its REML scalar is
    211.14208 on an AMD EPYC 7763 and 211.12792 on an Intel Xeon for these
    inputs), so the gate follows _check_rank_gauge's three branches.
    """
    bridge = RBridge()
    ok, reason = bridge.check_versions()
    if not ok:
        pytest.skip(reason)
    data = _get_data(SMOOTH_CONFIGS["te"], "gaussian")
    formula = "y ~ x1 + x2 + te(x1,x2,k=5)"
    r_formula = "y ~ x1 + x2 + te(x1,x2,k=c(5,5))"
    smoothing = np.array((0.7, 1.1))
    setup = ModelSetup.build(parse_formula(formula), data.to_dict("list"))
    fitting = FittingData.from_setup(setup, Gaussian())
    assert setup.penalties is not None
    r_setup = bridge.source_gaussian_rank_fit(
        r_formula, data, smoothing, np.ones(len(data)), np.zeros(len(data))
    )
    collector = _AssertCollector()
    collector.check(
        "public setup X",
        lambda: np.testing.assert_allclose(
            setup.X,
            r_setup["X"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    for index, (penalty, expected) in enumerate(
        zip(
            setup.penalties.blocks[0].dense_penalties(),
            r_setup["S"],
            strict=True,
        )
    ):
        collector.check(
            f"public setup S[{index}]",
            lambda penalty=penalty, expected=expected: np.testing.assert_allclose(
                penalty,
                expected,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
    branch = _check_rank_gauge(
        collector,
        design=np.asarray(setup.X),
        weights=np.ones(len(data)),
        target=np.asarray(setup.y),
        structure=setup.penalties,
        log_lambda=np.log(smoothing),
        sp_indices=(0, 1),
        retained=fitting.n_coef - fitting.rank_deficit,
        r_fit=r_setup,
        gauge_tolerance=STRICT,
        score_tolerance=STRICT,
    )
    warnings.warn(f"coupled-alias rank gauge branch: {branch}", stacklevel=1)
    collector.raise_if_any(f"pinned coupled-alias rank gauge [{branch}]")
