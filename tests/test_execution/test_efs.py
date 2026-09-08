"""Dense known-scale EFS execution adapter tests."""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import pytest

from jaxgam.execution import efs as execution_efs
from jaxgam.execution.efs import (
    EFSControl,
    _fit_state,
    dense_efs_known_scale,
    efs_initial_log_lambda,
)
from jaxgam.families.standard import Binomial, Gaussian, Poisson
from jaxgam.fitting.efs import EFSRawUpdate, prepare_efs_statistics
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests.fixtures.efs_weighted_additive_cr_repro import FORMULA, make_data
from tests.helpers import _AssertCollector, r_available
from tests.r_bridge import RBridge
from tests.tolerances import MODERATE, STRICT


def _build(
    formula: str,
    data: pd.DataFrame,
    family,
    *,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
):
    from jaxgam.fitting.data import FittingData

    setup = ModelSetup.build(
        parse_formula(formula), data, weights=weights, offset=offset
    )
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


def test_production_controller_extension_and_contraction_keep_old_warm_start(
    monkeypatch,
) -> None:
    """The actual host controller, not the EFS1 oracle, owns trial state."""
    data = _oracle_data("poisson")
    _, fd = _build("y ~ s(x, bs='cr', k=6)", data, Poisson())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
    )
    scores = iter([10.0, 9.0, 8.0, 11.0, 10.5])
    starts: list[np.ndarray] = []

    def scripted(_fd, _plan, rho, beta, _control):
        starts.append(np.asarray(beta))
        return execution_efs.EFSFitState(
            rho,
            base.pirls_result,
            jax.numpy.asarray(next(scores)),
            base.edf,
            base.statistics,
            None,
            True,
            True,
        )

    def update(rho, statistics, phi, multiplier, cap):
        del statistics, phi
        ratio = jax.numpy.exp(jax.numpy.ones_like(rho) * 0.01)
        trial = jax.numpy.minimum(rho + 0.01 * multiplier, cap)
        return EFSRawUpdate(
            jax.numpy.ones_like(rho), ratio, trial, jax.numpy.array(True)
        )

    monkeypatch.setattr(execution_efs, "_fit_state", scripted)
    monkeypatch.setattr(execution_efs, "efs_raw_update", update)
    result = execution_efs.dense_efs_known_scale(fd, control=EFSControl(outer_limit=2))
    assert result.convergence_info == "iteration limit reached"
    assert result.multiplier == 1.0
    # Every candidate/extension/contraction starts from the accepted beta;
    # no rejected trial can become a warm start.
    assert all(np.array_equal(start, starts[1]) for start in starts[1:])


def test_production_controller_preserves_failure_at_iteration_boundary(
    monkeypatch,
) -> None:
    data = _oracle_data("poisson")
    _, fd = _build("y ~ s(x, bs='cr', k=6)", data, Poisson())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
    )

    def invalid_trial(_fd, _plan, rho, _beta, _control):
        return execution_efs.EFSFitState(
            rho,
            base.pirls_result,
            base.score,
            base.edf,
            base.statistics,
            None,
            False,
            False,
        )

    monkeypatch.setattr(execution_efs, "_fit_state", invalid_trial)
    result = execution_efs.dense_efs_known_scale(fd, control=EFSControl(outer_limit=1))
    assert result.convergence_info == "inner_failure"


def test_production_controller_losing_extension_keeps_multiplier(monkeypatch) -> None:
    data = _oracle_data("poisson")
    _, fd = _build("y ~ s(x, bs='cr', k=6)", data, Poisson())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
    )
    scores = iter([10.0, 9.0, 9.1])

    def scripted(_fd, _plan, rho, _beta, _control):
        return execution_efs.EFSFitState(
            rho,
            base.pirls_result,
            jax.numpy.asarray(next(scores)),
            base.edf,
            base.statistics,
            None,
            True,
            True,
        )

    monkeypatch.setattr(execution_efs, "_fit_state", scripted)
    result = execution_efs.dense_efs_known_scale(fd, control=EFSControl(outer_limit=1))
    assert result.multiplier == 1.0
    assert result.convergence_info == "iteration limit reached"


def test_production_controller_reports_invalid_raw_update(monkeypatch) -> None:
    data = _oracle_data("poisson")
    _, fd = _build("y ~ s(x, bs='cr', k=6)", data, Poisson())
    base = _fit_state(
        fd,
        prepare_efs_statistics(fd),
        fd.log_lambda_init + 2.5,
        fd.beta_init,
        EFSControl(),
    )
    monkeypatch.setattr(execution_efs, "_fit_state", lambda *_: base)
    monkeypatch.setattr(
        execution_efs,
        "efs_raw_update",
        lambda rho, *_: EFSRawUpdate(
            jax.numpy.ones_like(rho),
            jax.numpy.ones_like(rho),
            rho,
            jax.numpy.array(False),
        ),
    )
    result = execution_efs.dense_efs_known_scale(fd, control=EFSControl(outer_limit=1))
    assert result.convergence_info == "invalid_update"


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


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    ("family", "family_name"), [(Poisson(), "poisson"), (Binomial(), "binomial")]
)
def test_weighted_offset_additive_cr_matches_pinned_r(family, family_name: str) -> None:
    """Ordinary two-smooth case; setup must receive the same w/off as R."""
    data = make_data()
    if family_name == "binomial":
        probability = 1.0 / (1.0 + np.exp(-0.3 * data["x"]))
        data = data.assign(y=np.random.default_rng(93).binomial(1, probability))
    setup, fd = _build(
        FORMULA,
        data,
        family,
        weights=data["w"].to_numpy(),
        offset=data["off"].to_numpy(),
    )
    rho = efs_initial_log_lambda(setup, family)
    bridge = RBridge(mode="subprocess")
    r_diag = bridge.efs_diagnostics(
        FORMULA,
        data,
        family_name,
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
    )
    initial = _fit_state(
        fd, prepare_efs_statistics(fd), rho + 2.5, fd.beta_init, EFSControl()
    )
    first = r_diag["statistics"].query("call == 1").sort_values("parameter")
    r_fit = bridge.fit_efs(
        FORMULA,
        data,
        family_name,
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(rho)),
    )
    j_fit = dense_efs_known_scale(fd, initial_log_lambda=rho)
    collector = _AssertCollector()
    collector.check(
        "initial lsp",
        lambda: np.testing.assert_allclose(
            rho + 2.5, first["log_smoothing"], rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "initial deviance",
        lambda: np.testing.assert_allclose(
            initial.pirls_result.deviance,
            first["deviance"].iloc[0],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "initial d",
        lambda: np.testing.assert_allclose(
            initial.statistics.determinant_derivative,
            first["ldetS1"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "initial t",
        lambda: np.testing.assert_allclose(
            initial.statistics.fisher_trace,
            first["trVS"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "initial q",
        lambda: np.testing.assert_allclose(
            initial.statistics.quadratic,
            first["bSb"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "final fitted",
        lambda: np.testing.assert_allclose(
            j_fit.pirls_result.mu,
            r_fit["fitted_values"],
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
        "final score",
        lambda: np.testing.assert_allclose(
            j_fit.score, r_fit["reml_score"], rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.raise_if_any(f"weighted-offset additive CR EFS parity ({family_name})")
