"""Conditional NB theta Newton kernel tests against pinned mgcv."""

from __future__ import annotations

import subprocess
import tempfile
from decimal import Decimal, localcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting import efs_theta
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.efs_theta import (
    _efs_nb_log_deviance,
    conditional_theta_newton,
    conditional_theta_nll,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests.helpers import r_available
from tests.tolerances import MODERATE, STRICT


def _inputs(
    *,
    fractional: bool = False,
    fractional_above_one: bool = False,
    large_counts: bool = False,
):
    rng = np.random.default_rng(5051)
    n = 48
    x = np.linspace(-1.0, 1.0, n)
    offset = 0.08 * np.cos(2.0 * x)
    weights = 0.6 + rng.random(n)
    eta = offset + 0.2 + 0.35 * np.sin(2.3 * x)
    mu = np.exp(eta)
    theta = 2.4
    y = rng.negative_binomial(theta, theta / (theta + mu)).astype(np.float64)
    if large_counts:
        y += 20 + 7 * (np.arange(n) % 13)
    if fractional:
        y += np.where(np.arange(n) % 2, 0.25, 0.5) + (
            1.0 if fractional_above_one else 0.0
        )
    data = pd.DataFrame({"y": y, "x": x, "w": weights, "off": offset})
    family = NegativeBinomial(theta=1.1)
    setup = ModelSetup.build(
        parse_formula("y ~ s(x, bs='cr', k=6)"),
        data,
        weights=weights,
        offset=offset,
    )
    fitting_data = FittingData.from_setup(setup, family)
    assert fitting_data.count_prefix_plan is not None
    return (
        family,
        fitting_data,
        jnp.asarray(eta),
        jnp.asarray(np.array([np.log(0.7)])),
        y,
        weights,
    )


def _solve(family, fitting_data, eta, log_theta, **kwargs):
    plan = fitting_data.count_prefix_plan
    assert plan is not None
    return conditional_theta_newton(
        log_theta,
        eta,
        fitting_data.y,
        fitting_data.wt,
        plan.indices,
        family,
        max_y=fitting_data.max_y,
        integer_counts=plan.integer_counts,
        **kwargs,
    )


def _r_conditional_oracle(
    y: np.ndarray,
    eta: np.ndarray,
    wt: np.ndarray,
    start: float,
    *,
    mu: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Evaluate and instrument pinned ``estimate.theta`` without patching mgcv."""
    start = float(start)
    with tempfile.TemporaryDirectory() as tmpdir:
        directory = Path(tmpdir)
        data_path = directory / "data.csv"
        derivative_path = directory / "derivatives.csv"
        trace_path = directory / "trace.csv"
        final_path = directory / "final.csv"
        oracle_mu = np.exp(eta) if mu is None else np.asarray(mu)
        pd.DataFrame({"y": y, "mu": oracle_mu, "wt": wt}).to_csv(data_path, index=False)
        script = f"""
library(mgcv)
if (as.character(getRversion()) != "4.5.2" ||
    packageVersion("mgcv") != package_version("1.9.3")) {{
  stop("conditional theta oracle requires R 4.5.2 and mgcv 1.9-3")
}}
d <- read.csv({str(data_path)!r})
family <- nb(theta=-exp({start!r}))
nlogl <- function(theta, deriv=2) {{
  dev <- sum(family$dev.resids(d$y, d$mu, d$wt, theta))
  ls <- family$ls(d$y, w=d$wt, theta=theta, scale=1)
  nll <- dev/2 - ls$ls
  if (deriv == 0) return(c(nll=nll))
  Dd <- family$Dd(d$y, d$mu, theta, wt=d$wt, level=deriv)
  g <- colSums(as.matrix(Dd$Dth))/2 - ls$lsth1[1]
  h <- colSums(as.matrix(Dd$Dth2))/2 - as.matrix(ls$lsth2)[1,1]
  c(nll=nll, gradient=g, hessian=h)
}}
source_lines <- capture.output(mgcv:::estimate.theta)
source_lines <- source_lines[!grepl("^<", source_lines)]
source_lines[1] <- sub("^function", "estimate.theta.trace <- function", source_lines[1])
anchor <- "theta <- theta + step"
if (sum(grepl(anchor, source_lines, fixed=TRUE)) != 1) stop("trace anchor changed")
source_lines <- sub(
  anchor, paste0(anchor, "; theta_trace <<- c(theta_trace, theta)"),
  source_lines, fixed=TRUE
)
eval(parse(text=source_lines))
initial <- nlogl(c({start!r}), deriv=2)
theta_trace <- numeric()
final_theta <- estimate.theta.trace(c({start!r}), family, d$y, d$mu, scale=1, wt=d$wt)
final <- nlogl(final_theta, deriv=2)
write.csv(as.data.frame(t(initial)), {str(derivative_path)!r}, row.names=FALSE)
trace_nll <- vapply(theta_trace, function(theta) nlogl(theta, deriv=0)[1], 0.0)
write.csv(
  data.frame(log_theta=theta_trace, nll=trace_nll), {str(trace_path)!r},
  row.names=FALSE
)
write.csv(
  as.data.frame(t(c(log_theta=final_theta, final))), {str(final_path)!r},
  row.names=FALSE
)
"""
        script_path = directory / "conditional_theta.R"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True, timeout=30
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        trace = pd.read_csv(trace_path)
        return {
            "initial": pd.read_csv(derivative_path).iloc[0].to_numpy(),
            "trace": trace["log_theta"].to_numpy(),
            "trace_nll": trace["nll"].to_numpy(),
            "final": pd.read_csv(final_path).iloc[0].to_numpy(),
        }


def _r_nb_log_deviance_oracle(
    y: np.ndarray, eta: np.ndarray, wt: np.ndarray, log_theta: float
) -> dict[str, np.ndarray | float]:
    """Pinned ``nb()$dev.resids`` and ``Dd`` in eta/log-theta coordinates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        directory = Path(tmpdir)
        data_path = directory / "data.csv"
        output_path = directory / "output.csv"
        pd.DataFrame({"y": y, "eta": eta, "wt": wt}).to_csv(data_path, index=False)
        script = f"""
library(mgcv)
if (as.character(getRversion()) != "4.5.2" ||
    packageVersion("mgcv") != package_version("1.9.3")) {{
  stop("NB deviance oracle requires R 4.5.2 and mgcv 1.9-3")
}}
d <- read.csv({str(data_path)!r})
theta <- {float(log_theta)!r}
mu <- exp(d$eta)
family <- nb(theta=-exp(theta))
value <- sum(family$dev.resids(d$y, mu, d$wt, theta))
Dd <- family$Dd(d$y, mu, theta, wt=d$wt, level=2)
d_eta <- Dd$Dmu * mu
h_eta <- Dd$Dmu2 * mu^2 + Dd$Dmu * mu
mixed <- Dd$Dmuth * mu
write.csv(data.frame(
  value=rep(value, length(d$y)), d_eta=d_eta, h_eta=h_eta,
  mixed=mixed, d_theta=rep(sum(Dd$Dth), length(d$y)),
  h_theta=rep(sum(Dd$Dth2), length(d$y))
), {str(output_path)!r}, row.names=FALSE)
"""
        script_path = directory / "nb_deviance.R"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True, timeout=30
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        output = pd.read_csv(output_path)
        return {
            "value": float(output["value"].iloc[0]),
            "d_eta": output["d_eta"].to_numpy(),
            "h_eta": output["h_eta"].to_numpy(),
            "mixed": output["mixed"].to_numpy(),
            "d_theta": float(output["d_theta"].iloc[0]),
            "h_theta": float(output["h_theta"].iloc[0]),
        }


def _r_nb_log_deviance_is_finite(
    y: float, eta: float, theta: float, wt: float = 1.0
) -> bool:
    """Whether literal pinned ``dev.resids`` remains a finite scalar."""
    script = f"""
library(mgcv)
if (as.character(getRversion()) != "4.5.2" ||
    packageVersion("mgcv") != package_version("1.9.3")) {{
  stop("NB deviance oracle requires R 4.5.2 and mgcv 1.9-3")
}}
family <- nb(theta=-{theta!r})
value <- sum(family$dev.resids({y!r}, exp({eta!r}), {wt!r}, log({theta!r})))
write(as.character(is.finite(value)), stdout())
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "nb_deviance_is_finite.R"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True, timeout=30
        )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return completed.stdout.strip() == "TRUE"


def _zero_count_tail_oracle(eta: float, theta: float) -> tuple[float, ...]:
    """100-digit analytic NB/log tail value, gradient, and full Hessian."""
    with localcontext() as context:
        context.prec = 100
        eta_decimal = Decimal(str(eta))
        theta_decimal = Decimal(str(theta))
        mu = eta_decimal.exp()
        ratio = mu / (mu + theta_decimal)
        value = 2 * theta_decimal * (1 + mu / theta_decimal).ln()
        d_eta = 2 * theta_decimal * ratio
        d_log_theta = 2 * theta_decimal * ((1 + mu / theta_decimal).ln() - ratio)
        d_eta2 = 2 * theta_decimal * ratio * (1 - ratio)
        mixed = 2 * theta_decimal * ratio * ratio
        d_log_theta2 = d_log_theta - mixed
    return tuple(
        float(item) for item in (value, d_eta, d_log_theta, d_eta2, mixed, d_log_theta2)
    )


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize("theta", [1e-3, 0.8, 10.0, 1e6])
def test_efs_nb_log_deviance_matches_pinned_r_value_and_derivatives(theta: float):
    """Stable EFS algebra agrees with ``nb()$dev.resids`` and ``Dd``."""
    y = np.array([0.0, 1.0, 7.0, 20.0, 1e6])
    eta = np.array([-20.0, 20.0, 40.0, 60.0, 350.0])
    wt = np.array([0.0, 0.7, 1.3, 0.9, 2.1])
    log_theta = jnp.array([np.log(theta)])
    eta_jax = jnp.asarray(eta)
    y_jax = jnp.asarray(y)
    wt_jax = jnp.asarray(wt)
    oracle = _r_nb_log_deviance_oracle(y, eta, wt, float(log_theta[0]))

    def objective(e, t):
        return _efs_nb_log_deviance(e, t, y_jax, wt_jax)

    eta_gradient = jax.grad(objective, argnums=0)(eta_jax, log_theta)
    eta_hessian = jax.hessian(objective, argnums=0)(eta_jax, log_theta)
    theta_gradient = jax.grad(objective, argnums=1)(eta_jax, log_theta)
    theta_hessian = jax.hessian(objective, argnums=1)(eta_jax, log_theta)
    mixed = jax.jacrev(jax.grad(objective, argnums=1), argnums=0)(eta_jax, log_theta)

    np.testing.assert_allclose(
        objective(eta_jax, log_theta),
        oracle["value"],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    np.testing.assert_allclose(
        eta_gradient, oracle["d_eta"], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        jnp.diag(eta_hessian),
        oracle["h_eta"],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    np.testing.assert_allclose(
        eta_hessian - jnp.diag(jnp.diag(eta_hessian)), 0.0, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        theta_gradient[0],
        oracle["d_theta"],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    np.testing.assert_allclose(
        theta_hessian[0, 0],
        oracle["h_theta"],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    np.testing.assert_allclose(
        mixed[0], oracle["mixed"], rtol=MODERATE.rtol, atol=MODERATE.atol
    )


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_efs_nb_log_deviance_near_mean_and_branch_boundary_match_pinned_r():
    """Exercise exact/nextafter mean and the safe-log1p branch boundary."""
    theta = 0.8
    y = np.array([1.0, 5.0, 11.0])
    boundary = np.log(2.0 * y[2] + theta)
    eta = np.array(
        [
            np.log(y[0]),
            np.nextafter(np.log(y[1]), np.inf),
            np.nextafter(boundary, -np.inf),
            boundary,
            np.nextafter(boundary, np.inf),
        ]
    )
    y = np.array([y[0], y[1], y[2], y[2], y[2]])
    wt = np.array([1.0, 0.4, 1.7, 0.9, 1.1])
    log_theta = jnp.array([np.log(theta)])
    oracle = _r_nb_log_deviance_oracle(y, eta, wt, float(log_theta[0]))
    actual = _efs_nb_log_deviance(
        jnp.asarray(eta), log_theta, jnp.asarray(y), jnp.asarray(wt)
    )
    np.testing.assert_allclose(
        actual, oracle["value"], rtol=MODERATE.rtol, atol=MODERATE.atol
    )


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    ("y", "eta", "theta", "expected_finite"),
    [
        (1.0, 350.0, 0.8, True),
        (1.0, 709.0, 0.8, True),
        (0.0, 709.0, 0.8, True),
        (1.0, -720.0, 0.8, False),
        (0.0, -720.0, 0.8, False),
        (1.0, -744.0, 0.8, False),
        (0.0, -744.0, 0.8, False),
        (1e308, 350.0, 1e-3, False),
    ],
)
def test_efs_nb_log_deviance_preserves_pinned_literal_validity_domain(
    y: float, eta: float, theta: float, expected_finite: bool
):
    """The stable tail keeps the literal R representability recovery signal."""
    oracle_finite = _r_nb_log_deviance_is_finite(y, eta, theta)
    assert oracle_finite is expected_finite
    actual = _efs_nb_log_deviance(
        jnp.array([eta]),
        jnp.array([np.log(theta)]),
        jnp.array([y]),
        jnp.array([1.0]),
    )
    assert bool(jnp.isfinite(actual)) is oracle_finite


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_efs_nb_log_deviance_fails_closed_for_flushed_subnormal_mean(y: float):
    """XLA flushes ``exp(-709)`` unlike R; retain the recovery signal."""
    assert _r_nb_log_deviance_is_finite(y, -709.0, 0.8)
    actual = _efs_nb_log_deviance(
        jnp.array([-709.0]),
        jnp.array([np.log(0.8)]),
        jnp.array([y]),
        jnp.array([1.0]),
    )
    assert jnp.isinf(actual)


@pytest.mark.parametrize("eta", [708.0, 709.0])
def test_efs_nb_log_deviance_tail_hessian_is_finite_after_inactive_branch_sanitize(
    eta: float,
):
    """A normal mean with an FTZ quotient must not poison inactive AD paths."""
    theta = 0.8
    y = jnp.array([0.0])
    wt = jnp.array([1.0])

    def objective(values):
        return _efs_nb_log_deviance(values[:1], values[1:], y, wt)

    values = jnp.array([eta, np.log(theta)])
    compiled_value_gradient = jax.jit(jax.value_and_grad(objective))
    compiled_hessian = jax.jit(jax.hessian(objective))
    value, gradient = compiled_value_gradient(values)
    hessian = compiled_hessian(values)
    expected = _zero_count_tail_oracle(eta, theta)
    expected_gradient = np.array(expected[1:3])
    expected_hessian = np.array(
        [[expected[3], expected[4]], [expected[4], expected[5]]]
    )
    assert np.all(np.isfinite(value))
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    np.testing.assert_allclose(
        value, expected[0], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        hessian, expected_hessian, rtol=MODERATE.rtol, atol=MODERATE.atol
    )


def test_efs_nb_log_deviance_preserves_invalid_log_link_recovery_signal_and_jits():
    """Log-space tails never turn overflowing or underflowing means valid."""
    log_theta = jnp.array([np.log(0.8)])
    y = jnp.array([1.0])
    wt = jnp.array([1.0])
    compiled = jax.jit(_efs_nb_log_deviance)
    finite = compiled(jnp.array([350.0]), log_theta, y, wt)
    overflow = compiled(jnp.array([710.0]), log_theta, y, wt)
    underflow = compiled(jnp.array([-750.0]), log_theta, y, wt)
    literal_overflow = compiled(jnp.array([-720.0]), log_theta, y, wt)
    zero_weight = compiled(jnp.array([60.0]), log_theta, y, jnp.array([0.0]))
    assert np.isfinite(finite)
    assert np.isinf(overflow)
    assert np.isinf(underflow)
    assert np.isinf(literal_overflow)
    assert zero_weight == pytest.approx(0.0)


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_conditional_theta_derivatives_and_iterations_match_pinned_r() -> None:
    family, fitting_data, eta, start, y, weights = _inputs()
    plan = fitting_data.count_prefix_plan
    assert plan is not None

    def objective(theta):
        return conditional_theta_nll(
            theta,
            eta,
            fitting_data.y,
            fitting_data.wt,
            plan.indices,
            family,
            max_y=fitting_data.max_y,
            integer_counts=plan.integer_counts,
        )

    result = _solve(family, fitting_data, eta, start)
    oracle = _r_conditional_oracle(y, np.asarray(eta), weights, float(start[0]))
    initial = np.array(
        [
            objective(start),
            jax.grad(objective)(start)[0],
            jax.hessian(objective)(start)[0, 0],
        ]
    )
    np.testing.assert_allclose(
        initial, oracle["initial"], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    assert int(result.n_iter) == len(oracle["trace"])
    history = np.asarray(result.nll_history[: result.n_history])
    assert len(history) == len(oracle["trace_nll"]) + 1
    np.testing.assert_allclose(
        history[1:], oracle["trace_nll"], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        result.log_theta[0], oracle["final"][0], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    assert np.all(
        np.diff(history) <= np.finfo(np.float64).eps ** 0.75 * np.abs(history[:-1])
    )
    assert bool(result.converged)


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    ("fractional_above_one", "large_counts", "start"),
    [(False, True, np.log(0.03)), (True, False, np.log(1e4))],
)
def test_conditional_theta_matches_pinned_r_on_supported_count_modes(
    fractional_above_one: bool, large_counts: bool, start: float
) -> None:
    family, fitting_data, eta, _, y, weights = _inputs(
        fractional=fractional_above_one,
        fractional_above_one=fractional_above_one,
        large_counts=large_counts,
    )
    theta = jnp.array([start])
    plan = fitting_data.count_prefix_plan
    assert plan is not None

    def objective(log_theta):
        return conditional_theta_nll(
            log_theta,
            eta,
            fitting_data.y,
            fitting_data.wt,
            plan.indices,
            family,
            max_y=fitting_data.max_y,
            integer_counts=plan.integer_counts,
        )

    result = _solve(family, fitting_data, eta, theta)
    oracle = _r_conditional_oracle(y, np.asarray(eta), weights, start)
    initial = np.array(
        [
            objective(theta),
            jax.grad(objective)(theta)[0],
            jax.hessian(objective)(theta)[0, 0],
        ]
    )
    np.testing.assert_allclose(
        initial, oracle["initial"], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    history = np.asarray(result.nll_history[: result.n_history])
    assert len(history) == len(oracle["trace_nll"]) + 1
    np.testing.assert_allclose(
        history[1:], oracle["trace_nll"], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    np.testing.assert_allclose(
        result.log_theta[0], oracle["final"][0], rtol=MODERATE.rtol, atol=MODERATE.atol
    )
    assert plan.integer_counts is (not fractional_above_one)
    if large_counts:
        assert fitting_data.max_y > 100


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_conditional_theta_rejects_fractional_response_nll_offset() -> None:
    """Keep the inherited fractional-NB convention visible, not R-supported."""
    family, fitting_data, eta, _, y, weights = _inputs(fractional=True)
    plan = fitting_data.count_prefix_plan
    assert plan is not None
    theta = jnp.array([np.log(1e4)])
    result = _solve(family, fitting_data, eta, theta)
    oracle = _r_conditional_oracle(y, np.asarray(eta), weights, float(theta[0]))
    raw_nll = conditional_theta_nll(
        theta,
        eta,
        fitting_data.y,
        fitting_data.wt,
        plan.indices,
        family,
        max_y=fitting_data.max_y,
        integer_counts=plan.integer_counts,
    )
    fractional = (y > 0.0) & (y < 1.0)
    r_nll_offset = -np.sum(weights[fractional] * y[fractional] * np.log(y[fractional]))
    np.testing.assert_allclose(
        raw_nll + r_nll_offset,
        oracle["initial"][0],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
    assert int(result.status) == efs_theta._STATUS_UNSUPPORTED_FRACTIONAL_RESPONSE
    assert not bool(result.converged)
    assert int(result.n_iter) == 0
    assert int(result.n_history) == 1


def test_conditional_theta_kernel_jits_with_dynamic_theta() -> None:
    family, fitting_data, eta, start, _, _ = _inputs()
    plan = fitting_data.count_prefix_plan
    assert plan is not None
    compiled = jax.jit(
        lambda theta: conditional_theta_newton(
            theta,
            eta,
            fitting_data.y,
            fitting_data.wt,
            plan.indices,
            family,
            max_y=fitting_data.max_y,
            integer_counts=plan.integer_counts,
        )
    )
    first = compiled(start)
    second = compiled(start + 0.2)
    assert np.isfinite(first.nll)
    assert np.isfinite(second.nll)
    assert first.log_theta.shape == second.log_theta.shape == (1,)


def test_conditional_theta_reports_zero_curvature_and_nonfinite_objective(
    monkeypatch,
) -> None:
    family, fitting_data, eta, start, _, _ = _inputs()
    plan = fitting_data.count_prefix_plan
    assert plan is not None

    def linear_objective(theta, *args, **kwargs):
        del args, kwargs
        return theta[0]

    monkeypatch.setattr(efs_theta, "_conditional_theta_nll_jit", linear_objective)
    efs_theta._conditional_theta_newton_jit.clear_cache()
    zero = _solve(family, fitting_data, eta, start)
    assert int(zero.status) == efs_theta._STATUS_ZERO_CURVATURE
    monkeypatch.undo()
    efs_theta._conditional_theta_newton_jit.clear_cache()

    nonfinite = _solve(family, fitting_data, jnp.full_like(eta, jnp.nan), start)
    assert int(nonfinite.status) == efs_theta._STATUS_INITIAL_NONFINITE


def test_conditional_theta_recovers_an_infinite_trial_by_halving(monkeypatch) -> None:
    family, fitting_data, eta, _, _, _ = _inputs()
    plan = fitting_data.count_prefix_plan
    assert plan is not None

    def recoverable_infinite_trial(theta, *args, **kwargs):
        del args, kwargs
        finite_value = theta[0] ** 2 - 2.0 * theta[0]
        return jnp.where(theta[0] >= 0.75, jnp.inf, finite_value)

    monkeypatch.setattr(
        efs_theta, "_conditional_theta_nll_jit", recoverable_infinite_trial
    )
    efs_theta._conditional_theta_newton_jit.clear_cache()
    result = _solve(
        family,
        fitting_data,
        eta,
        jnp.array([0.0]),
        max_iter=1,
    )
    assert int(result.status) == efs_theta._STATUS_ITERATION_LIMIT
    assert int(result.n_history) == 2
    assert result.nll_history[1] == pytest.approx(-0.75)
    monkeypatch.undo()
    efs_theta._conditional_theta_newton_jit.clear_cache()


def test_conditional_theta_reports_line_search_failure() -> None:
    y = jnp.array(
        [0, 1, 2, 1, 6, 0, 3, 2, 1, 2, 3, 2, 2, 1, 4, 4, 2, 4, 1, 2],
        dtype=jnp.float64,
    )
    eta = jnp.array(
        [
            -3.9166786485581433,
            -5.636708018382006,
            1.7488353066002702,
            3.740833789167816,
            -1.9353286523100799,
            3.270792451256682,
            2.22856993753366,
            -7.632976698625011,
            7.914303751098403,
            7.5543545985734095,
            -1.5240893122791217,
            -3.043881295304381,
            1.8139513845075435,
            -7.604154899194157,
            -0.1430485889146258,
            0.7774680457331158,
            5.567537211500943,
            5.027848840219194,
            1.4554000106541576,
            -5.699305319353627,
        ]
    )
    family = NegativeBinomial(theta=1.0)
    result = conditional_theta_newton(
        jnp.array([-4.207454150261791]),
        eta,
        y,
        jnp.ones_like(y),
        y.astype(jnp.int64),
        family,
        max_y=6,
        integer_counts=True,
        max_halvings=0,
        max_iter=3,
    )
    assert int(result.status) == efs_theta._STATUS_LINE_SEARCH_FAILED


def test_conditional_theta_rejects_fixed_or_non_nb_family() -> None:
    _, fitting_data, eta, start, _, _ = _inputs()
    plan = fitting_data.count_prefix_plan
    assert plan is not None
    with pytest.raises(ValueError, match="estimated NB theta"):
        conditional_theta_newton(
            start,
            eta,
            fitting_data.y,
            fitting_data.wt,
            plan.indices,
            NegativeBinomial(theta=1.0, fixed=True),
            max_y=fitting_data.max_y,
            integer_counts=plan.integer_counts,
        )
    with pytest.raises(TypeError, match="NegativeBinomial"):
        conditional_theta_newton(
            start,
            eta,
            fitting_data.y,
            fitting_data.wt,
            plan.indices,
            family=None,  # type: ignore[arg-type]
            max_y=fitting_data.max_y,
            integer_counts=plan.integer_counts,
        )


@pytest.mark.parametrize(
    "link", ["logit", "probit", "cloglog", "inverse", "inverse_squared"]
)
def test_conditional_theta_rejects_links_rejected_by_pinned_nb(link: str) -> None:
    _, fitting_data, eta, start, _, _ = _inputs()
    plan = fitting_data.count_prefix_plan
    assert plan is not None
    with pytest.raises(NotImplementedError, match="links log, identity, and sqrt"):
        conditional_theta_newton(
            start,
            eta,
            fitting_data.y,
            fitting_data.wt,
            plan.indices,
            NegativeBinomial(theta=1.0, link=link),
            max_y=fitting_data.max_y,
            integer_counts=plan.integer_counts,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_y": True}, "max_y"),
        ({"max_y": -1}, "max_y"),
        ({"max_iter": 1.5}, "max_iter"),
        ({"max_halvings": -1}, "max_halvings"),
        ({"integer_counts": 1}, "integer_counts"),
        ({"tolerance": np.nan}, "tolerance"),
        ({"max_step": False}, "max_step"),
    ],
)
def test_conditional_theta_rejects_invalid_static_controls(
    kwargs, message: str
) -> None:
    family, fitting_data, eta, start, _, _ = _inputs()
    plan = fitting_data.count_prefix_plan
    assert plan is not None
    arguments = {
        "max_y": fitting_data.max_y,
        "integer_counts": plan.integer_counts,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        conditional_theta_newton(
            start,
            eta,
            fitting_data.y,
            fitting_data.wt,
            plan.indices,
            family,
            **arguments,
        )


@pytest.mark.parametrize(
    ("log_theta", "eta", "count_indices", "message"),
    [
        (jnp.zeros(2), None, None, "log_theta"),
        (None, jnp.zeros(3), None, "must align"),
        (None, None, jnp.zeros(3, dtype=jnp.int64), "must align"),
    ],
)
def test_conditional_theta_rejects_invalid_static_shapes(
    log_theta, eta, count_indices, message: str
) -> None:
    family, fitting_data, default_eta, start, _, _ = _inputs()
    plan = fitting_data.count_prefix_plan
    assert plan is not None
    with pytest.raises(ValueError, match=message):
        conditional_theta_newton(
            start if log_theta is None else log_theta,
            default_eta if eta is None else eta,
            fitting_data.y,
            fitting_data.wt,
            plan.indices if count_indices is None else count_indices,
            family,
            max_y=fitting_data.max_y,
            integer_counts=plan.integer_counts,
        )


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize("link", ["identity", "sqrt"])
def test_conditional_theta_nonlog_links_match_pinned_r_strict(link: str) -> None:
    _, fitting_data, _, start, y, weights = _inputs()
    plan = fitting_data.count_prefix_plan
    assert plan is not None
    family = NegativeBinomial(theta=1.0, link=link)
    mu = 2.0 + 0.4 * np.sin(np.linspace(-1.0, 1.0, len(y)))
    eta = jnp.asarray(family.link.link(mu))
    result = conditional_theta_newton(
        start,
        eta,
        fitting_data.y,
        fitting_data.wt,
        plan.indices,
        family,
        max_y=fitting_data.max_y,
        integer_counts=plan.integer_counts,
    )
    oracle = _r_conditional_oracle(
        y,
        np.asarray(eta),
        weights,
        float(start[0]),
        mu=mu,
    )
    np.testing.assert_allclose(
        result.nll_history[: result.n_history],
        np.concatenate(([oracle["initial"][0]], oracle["trace_nll"])),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        result.log_theta[0],
        oracle["final"][0],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
