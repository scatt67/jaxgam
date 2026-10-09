"""Pinned mgcv public covariance gate for the Python global rank map."""

from __future__ import annotations

import numpy as np
import pytest

from jaxgam.families.standard import Gaussian
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.rank_coordinates import jit_reduced_augmented_qr
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.rank_coordinates import global_rank_coordinate_map
from tests.helpers import _AssertCollector
from tests.r_bridge import RBridge
from tests.test_validation_matrix import SMOOTH_CONFIGS, _get_data
from tests.tolerances import MODERATE, STRICT


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
    coordinate_map, coordinates = global_rank_coordinate_map(
        np.sqrt(weights)[:, None] * np.asarray(setup.X),
        setup.penalties,
        np.array((np.log(smoothing_parameter),)),
        retained_rank=fitting.n_coef - fitting.rank_deficit,
    )
    target = np.asarray(setup.y) - offset
    beta, covariance, _factor, logdet = map(
        np.asarray,
        jit_reduced_augmented_qr(
            np.sqrt(weights)[:, None] * np.asarray(setup.X) @ coordinates.transform,
            np.sqrt(weights) * target,
            tuple(coordinates.penalty_roots),
            np.array((np.log(smoothing_parameter),)),
            (0,),
            coordinate_map.basis,
            coordinate_map.public_basis,
        ),
    )
    r_setup = bridge.source_gaussian_rank_fit(
        formula, data, smoothing_parameter, weights, offset
    )
    expected_beta = r_setup["coefficients"]
    expected_covariance = r_setup["covariance"]
    expected_score = r_setup["score"]
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
    collector.check(
        "REML scalar",
        lambda: np.testing.assert_allclose(
            _gaussian_reml_score(
                beta,
                float(logdet),
                coordinates,
                np.asarray(setup.X),
                target,
                weights,
                np.array((np.log(smoothing_parameter),)),
            ),
            expected_score,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any("pinned weighted GP rank map")


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
    """A tied tensor null space must retain R's public beta, Vp and REML."""
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
    coordinate_map, coordinates = global_rank_coordinate_map(
        np.asarray(setup.X),
        setup.penalties,
        np.log(smoothing),
        retained_rank=fitting.n_coef - fitting.rank_deficit,
    )
    beta, covariance, _factor, logdet = map(
        np.asarray,
        jit_reduced_augmented_qr(
            np.asarray(setup.X) @ coordinates.transform,
            np.asarray(setup.y),
            coordinates.penalty_roots,
            np.log(smoothing),
            (0, 1),
            coordinate_map.basis,
            coordinate_map.public_basis,
        ),
    )
    expected_beta, expected_covariance, expected_score = (
        _pinned_public_gaussian_rank_fit(
            bridge,
            data,
            r_formula,
            smoothing,
            np.ones(len(data)),
            np.zeros(len(data)),
        )
    )
    r_setup = bridge.source_gaussian_rank_fit(
        r_formula, data, smoothing, np.ones(len(data)), np.zeros(len(data))
    )
    score = _gaussian_reml_score(
        beta,
        float(logdet),
        coordinates,
        np.asarray(setup.X),
        np.asarray(setup.y),
        np.ones(len(data)),
        np.log(smoothing),
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
    collector.check(
        "public coefficients",
        lambda: np.testing.assert_allclose(
            beta, expected_beta, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "public covariance",
        lambda: np.testing.assert_allclose(
            covariance, expected_covariance, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "REML scalar",
        lambda: np.testing.assert_allclose(
            score, expected_score, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.raise_if_any("pinned coupled-alias rank gauge")
