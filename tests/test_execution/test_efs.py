"""Dense known-scale EFS execution adapter tests."""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import pytest

from jaxgam.execution.efs import (
    EFSControl,
    _fit_state,
    dense_efs_known_scale,
    efs_initial_log_lambda,
)
from jaxgam.families.standard import Binomial, Gaussian, Poisson
from jaxgam.fitting.efs import prepare_efs_statistics
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector, r_available
from tests.r_bridge import RBridge
from tests.tolerances import MODERATE, STRICT


def _build(formula: str, data: pd.DataFrame, family):
    from jaxgam.fitting.data import FittingData

    setup = ModelSetup.build(parse_formula(formula), data)
    return setup, FittingData.from_setup(setup, family)


def _oracle_data(family_name: str, *, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, 80)
    z = rng.uniform(-1.0, 1.0, len(x))
    eta = 0.15 + 0.45 * np.sin(2.5 * x) + 0.2 * x * z
    if family_name == "poisson":
        y = rng.poisson(np.exp(eta))
    else:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    return pd.DataFrame({"y": y, "x": x, "z": z})


@pytest.mark.parametrize("family", [Poisson(), Binomial()])
def test_dense_known_scale_efs_runs_from_one_time_shift(family) -> None:
    rng = np.random.default_rng(320)
    x = np.linspace(-1.0, 1.0, 48)
    eta = 0.2 + 0.5 * np.sin(3.0 * x)
    if family.family_name == "poisson":
        y = rng.poisson(np.exp(eta))
    else:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    _, fd = _build("y ~ s(x, bs='cr', k=6)", pd.DataFrame({"x": x, "y": y}), family)
    result = dense_efs_known_scale(fd, control=EFSControl(outer_limit=2))
    assert result.n_iter == 2
    assert result.convergence_info == "iteration limit reached"
    assert result.scale == 1.0
    assert result.score_history
    assert np.all(np.isfinite(np.asarray(result.smoothing_params)))
    assert result.update_residual is not None


def test_dense_efs_rejects_out_of_scope_family_before_any_newton_dispatch() -> None:
    x = np.linspace(0.0, 1.0, 20)
    _, fd = _build(
        "y ~ s(x, bs='cr', k=5)",
        pd.DataFrame({"x": x, "y": x}),
        Gaussian(),
    )
    with pytest.raises(NotImplementedError, match="Poisson/log"):
        dense_efs_known_scale(fd)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outer_limit": True},
        {"pirls_max_iter": 1.5},
        {"history_limit": 0},
        {"score_tolerance": np.nan},
        {"pirls_tolerance": np.inf},
    ],
)
def test_efs_control_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError, match="EFS"):
        EFSControl(**kwargs)


def test_existing_efs_statistics_kernel_compiles_and_executes() -> None:
    # This regression belongs near the execution adapter because every outer
    # proposal consumes its JIT statistics kernel.
    from jaxgam.fitting.efs import EFSStatistics, efs_raw_update

    stats = EFSStatistics(
        jax.numpy.array([1.0]),
        jax.numpy.array([0.25]),
        jax.numpy.array([1.0]),
        jax.numpy.array(True),
        jax.numpy.array(True),
    )
    result = jax.jit(efs_raw_update)(
        jax.numpy.array([0.0]),
        stats,
        jax.numpy.array(1.0),
        jax.numpy.array(1.0),
        jax.numpy.array(15.0),
    )
    assert bool(result.finite_positive)


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    ("family", "family_name"), [(Poisson(), "poisson"), (Binomial(), "binomial")]
)
def test_known_scale_efs_matches_pinned_r_from_matched_initial_state(
    family, family_name: str
) -> None:
    """Real optimizer parity, not a default-Newton substitute."""
    formula = "y ~ s(x, bs='cr', k=6)"
    data = _oracle_data(family_name)
    setup, fd = _build(formula, data, family)
    rho = efs_initial_log_lambda(setup, family)
    oracle = RBridge(mode="subprocess")
    r_fit = oracle.fit_efs(
        formula, data, family_name, initial_smoothing=np.exp(np.asarray(rho))
    )
    j_fit = dense_efs_known_scale(fd, initial_log_lambda=rho)
    collector = _AssertCollector()
    collector.check(
        "fitted values",
        lambda: np.testing.assert_allclose(
            np.asarray(j_fit.pirls_result.mu),
            r_fit["fitted_values"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "deviance",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.deviance,
            r_fit["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "criterion",
        lambda: np.testing.assert_allclose(
            j_fit.score, r_fit["reml_score"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        "smoothing parameters",
        lambda: np.testing.assert_allclose(
            j_fit.smoothing_params,
            r_fit["smoothing_params"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any("known-scale pinned EFS parity")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize("term", ["te(x,z,k=5)", "ti(x,z,k=5)"])
def test_coupled_efs_statistics_and_fit_match_pinned_r(term: str) -> None:
    """Coupled penalty trace/update state and final fit share one R run."""
    formula = f"y ~ {term}"
    data = _oracle_data("poisson", seed=44)
    family = Poisson()
    setup, fd = _build(formula, data, family)
    rho = efs_initial_log_lambda(setup, family)
    bridge = RBridge(mode="subprocess")
    diagnostic = bridge.efs_diagnostics(
        formula, data, "poisson", initial_smoothing=np.exp(np.asarray(rho))
    )
    state = _fit_state(
        fd, prepare_efs_statistics(fd), rho + 2.5, fd.beta_init, EFSControl()
    )
    first = diagnostic["statistics"].query("call == 1").sort_values("parameter")
    r_fit = bridge.fit_efs(
        formula, data, "poisson", initial_smoothing=np.exp(np.asarray(rho))
    )
    j_fit = dense_efs_known_scale(fd, initial_log_lambda=rho)
    collector = _AssertCollector()
    collector.check(
        "initial d",
        lambda: np.testing.assert_allclose(
            state.statistics.determinant_derivative,
            first["ldetS1"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "initial t",
        lambda: np.testing.assert_allclose(
            state.statistics.fisher_trace,
            first["trVS"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "initial q",
        lambda: np.testing.assert_allclose(
            state.statistics.quadratic,
            first["bSb"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "final deviance",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.deviance,
            r_fit["deviance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "final criterion",
        lambda: np.testing.assert_allclose(
            j_fit.score, r_fit["reml_score"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.raise_if_any(f"pinned coupled EFS parity ({term})")
