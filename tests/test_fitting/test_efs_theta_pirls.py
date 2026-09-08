"""EFS-only in-loop conditional-theta PIRLS tests.

These tests intentionally exercise the internal NB/log route directly. EFS
outer-controller wiring belongs to a later change: this module establishes the
immutable beta/theta state and the pinned ``gam.fit4`` ordering first.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting import pirls as pirls_module
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.pirls import (
    _EFS_STATUS_BETA_STEP_FAILED,
    _EFS_STATUS_INVALID_INPUT,
    _EFS_STATUS_INVALID_WORKING_FACTORS,
    _EFS_STATUS_ITERATION_LIMIT,
    _EFS_STATUS_RETAINED_START_INVALID_TRIAL,
    _EFS_STATUS_THETA_FAILED,
    _BetaStepResult,
    efs_theta_pirls_loop,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests.helpers import r_available
from tests.tolerances import MODERATE, STRICT


def _fixture(*, fractional_below_one: bool = False):
    rng = np.random.default_rng(5053)
    n = 42
    x = np.linspace(-1.0, 1.0, n)
    offset = 0.08 * np.cos(2.1 * x)
    wt = 0.5 + rng.random(n)
    mu = np.exp(offset + 0.25 + 0.4 * np.sin(2.2 * x))
    y = rng.negative_binomial(2.2, 2.2 / (2.2 + mu)).astype(np.float64)
    if fractional_below_one:
        y[0] = 0.25
    data = pd.DataFrame({"y": y, "x": x})
    setup = ModelSetup.build(
        parse_formula("y ~ s(x, bs='cr', k=6)"), data, weights=wt, offset=offset
    )
    fd = FittingData.from_setup(setup, NegativeBinomial(theta=0.8))
    assert fd.count_prefix_plan is not None
    S_lambda = jnp.eye(fd.X.shape[1]) * 0.15
    return fd, S_lambda


def _fit(fd: FittingData, S_lambda: jax.Array, **kwargs):
    plan = fd.count_prefix_plan
    assert plan is not None
    return efs_theta_pirls_loop(
        fd.X,
        fd.y,
        fd.beta_init,
        S_lambda,
        fd.family,
        fd.wt,
        fd.offset,
        jnp.asarray([np.log(0.8)]),
        plan.indices,
        max_y=fd.max_y,
        integer_counts=plan.integer_counts,
        **kwargs,
    )


def _call(fd: FittingData, S_lambda: jax.Array, **kwargs):
    plan = fd.count_prefix_plan
    assert plan is not None
    defaults = {
        "X": fd.X,
        "y": fd.y,
        "beta_init": fd.beta_init,
        "S_lambda": S_lambda,
        "family": fd.family,
        "wt": fd.wt,
        "offset": fd.offset,
        "log_theta_init": jnp.asarray([np.log(0.8)]),
        "count_indices": plan.indices,
        "max_y": fd.max_y,
        "integer_counts": plan.integer_counts,
    }
    defaults.update(kwargs)
    return efs_theta_pirls_loop(**defaults)


def test_efs_theta_pirls_jit_rebuilds_final_quantities_at_returned_theta():
    fd, S_lambda = _fixture()
    result = _fit(fd, S_lambda)
    pr = result.pirls_result

    assert bool(np.asarray(pr.converged))
    assert int(np.asarray(result.status)) == 0
    assert float(np.asarray(pr.scale)) == 1.0
    assert np.all(np.isfinite(np.asarray(pr.coefficients)))
    assert np.all(np.isfinite(np.asarray(pr.XtWX)))
    assert np.all(np.isfinite(np.asarray(pr.XtWX_fisher)))

    deviance = fd.family.deviance_fn(fd.y, fd.wt)(pr.eta, result.log_theta)
    expected_pdev = deviance + pr.coefficients @ S_lambda @ pr.coefficients
    np.testing.assert_allclose(
        pr.deviance, deviance, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        pr.penalized_deviance, expected_pdev, rtol=STRICT.rtol, atol=STRICT.atol
    )

    W_fisher = fd.family.working_weights_fn(fd.wt)(pr.eta, result.log_theta)
    expected_fisher = (W_fisher[:, None] * fd.X).T @ fd.X
    np.testing.assert_allclose(
        pr.XtWX_fisher, expected_fisher, rtol=STRICT.rtol, atol=STRICT.atol
    )

    # The wrapped route is itself JIT-safe and theta remains a dynamic input.
    compiled = jax.jit(
        efs_theta_pirls_loop,
        static_argnames=(
            "family",
            "max_y",
            "integer_counts",
            "max_iter",
            "tol",
            "initial_start_retained",
        ),
    )
    plan = fd.count_prefix_plan
    assert plan is not None
    compiled_result = compiled(
        fd.X,
        fd.y,
        fd.beta_init,
        S_lambda,
        fd.family,
        fd.wt,
        fd.offset,
        jnp.asarray([np.log(1.7)]),
        plan.indices,
        max_y=fd.max_y,
        integer_counts=plan.integer_counts,
    )
    assert np.all(np.isfinite(np.asarray(compiled_result.log_theta)))
    # A default/reset EFS start supplies link(mustart), which need not equal
    # the projected coefficient predictor. It remains dynamic JIT input.
    mustart = fd.family.initialize(fd.y, fd.wt)
    mustart_eta = fd.family.link.link(mustart)
    reset_result = compiled(
        fd.X,
        fd.y,
        fd.beta_init,
        S_lambda,
        fd.family,
        fd.wt,
        fd.offset,
        jnp.asarray([np.log(1.7)]),
        plan.indices,
        initial_eta=mustart_eta,
        initial_start_retained=False,
        max_y=fd.max_y,
        integer_counts=plan.integer_counts,
    )
    assert np.all(np.isfinite(np.asarray(reset_result.log_theta)))


def test_efs_theta_pirls_rejects_fractional_response_before_convergence():
    fd, S_lambda = _fixture(fractional_below_one=True)
    result = _fit(fd, S_lambda)

    assert not bool(np.asarray(result.pirls_result.converged))
    assert int(np.asarray(result.status)) == _EFS_STATUS_THETA_FAILED
    assert int(np.asarray(result.theta_status)) == 8


def test_efs_theta_first_step_uses_explicit_old_baseline_and_origin():
    fd, S_lambda = _fixture()
    beta_old = jnp.zeros_like(fd.beta_init)
    result = _fit(fd, S_lambda, beta_old_init=beta_old, max_iter=1)
    eta_old = fd.X @ beta_old + fd.offset
    initial_pdev = (
        fd.family.deviance_fn(fd.y, fd.wt)(eta_old, jnp.asarray([np.log(0.8)]))
        + beta_old @ S_lambda @ beta_old
    )

    # Unlike ordinary PIRLS, EFS does not accept its first proposal merely
    # because it is finite: gam.fit4 compares it to old.pdev/null.coef.
    threshold = 10.0 * (0.1 + abs(float(initial_pdev))) * np.sqrt(np.finfo(float).eps)
    assert (
        float(np.asarray(result.stopping_penalized_deviance))
        <= float(initial_pdev) + threshold
    )


def test_efs_theta_input_validation_is_static_before_jit():
    fd, S_lambda = _fixture()
    plan = fd.count_prefix_plan
    assert plan is not None
    with pytest.raises(ValueError, match="max_iter"):
        efs_theta_pirls_loop(
            fd.X,
            fd.y,
            fd.beta_init,
            S_lambda,
            fd.family,
            fd.wt,
            fd.offset,
            jnp.asarray([0.0]),
            plan.indices,
            max_y=fd.max_y,
            integer_counts=plan.integer_counts,
            max_iter=True,
        )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("beta_init", jnp.zeros(1), "beta_init"),
        ("S_lambda", jnp.eye(1), "S_lambda"),
        ("count_indices", jnp.zeros(42, dtype=jnp.float64), "count_indices"),
        ("X", jnp.zeros((0, 2)), "nonempty"),
    ],
)
def test_efs_theta_pirls_rejects_static_shape_and_prefix_dtype(name, value, message):
    fd, S_lambda = _fixture()
    call = (
        (lambda: _call(fd, value))
        if name == "S_lambda"
        else lambda: _call(fd, S_lambda, **{name: value})
    )
    with pytest.raises(ValueError, match=message):
        call()


def test_efs_theta_pirls_reports_dynamic_malformed_inputs_without_convergence():
    fd, S_lambda = _fixture()
    bad_weights = _call(fd, S_lambda, wt=-fd.wt)
    bad_prefix = _call(fd, S_lambda, count_indices=jnp.full(fd.y.shape, -1))
    for result in (bad_weights, bad_prefix):
        assert not bool(np.asarray(result.pirls_result.converged))
        assert int(np.asarray(result.status)) == _EFS_STATUS_INVALID_INPUT


def test_efs_theta_pirls_reports_invalid_factor_and_iteration_limit():
    fd, S_lambda = _fixture()
    invalid_factor = _call(fd, jnp.full_like(S_lambda, jnp.nan))
    limited = _call(fd, S_lambda, max_iter=1)
    assert not bool(np.asarray(invalid_factor.pirls_result.converged))
    assert int(np.asarray(invalid_factor.status)) == _EFS_STATUS_INVALID_WORKING_FACTORS
    assert not bool(np.asarray(limited.pirls_result.converged))
    assert int(np.asarray(limited.status)) == _EFS_STATUS_ITERATION_LIMIT


def test_efs_theta_pirls_distinguishes_unrecoverable_beta_step(monkeypatch):
    """A valid NB/log old baseline is eventually reached by binary halving.

    The beta-step failure status is therefore exercised with an isolated pure
    step double, rather than fabricating invalid NB data and mislabelling that
    as an ordinary divergence.
    """
    fd, S_lambda = _fixture()

    def rejected_step(**kwargs):
        beta = kwargs["beta"]
        X = kwargs["X"]
        offset = kwargs["offset"]
        return _BetaStepResult(
            beta=beta,
            mu=kwargs["mu"],
            eta=X @ beta + offset,
            penalized_deviance=kwargs["penalized_deviance"],
            accepted=jnp.array(False),
            factors_valid=jnp.array(True),
            solver_valid=jnp.array(True),
            XtWX=jnp.eye(X.shape[1]),
            L=jnp.eye(X.shape[1]),
            W=jnp.ones(X.shape[0]),
        )

    monkeypatch.setattr(pirls_module, "_beta_step", rejected_step)
    pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    try:
        result = _call(fd, S_lambda)
    finally:
        pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    assert not bool(np.asarray(result.pirls_result.converged))
    assert int(np.asarray(result.status)) == _EFS_STATUS_BETA_STEP_FAILED


def test_efs_theta_pirls_names_retained_first_start_invalid_trial(monkeypatch):
    """A null-origin halving cannot rescue an earlier retained-start failure."""
    fd, S_lambda = _fixture()

    def invalid_retained_step(**kwargs):
        beta = kwargs["beta"]
        X = kwargs["X"]
        offset = kwargs["offset"]
        return _BetaStepResult(
            beta=beta,
            mu=kwargs["mu"],
            eta=X @ beta + offset,
            penalized_deviance=kwargs["penalized_deviance"],
            # Model a shared helper that halved around the null anchor and
            # found an acceptable point. EFS must reject it before theta is
            # updated because gam.fit4's earlier recovery starts at retained
            # etaold, not the later null-divergence origin.
            accepted=jnp.array(True),
            factors_valid=jnp.array(True),
            solver_valid=jnp.array(True),
            proposal_valid=jnp.array(False),
            XtWX=jnp.eye(X.shape[1]),
            L=jnp.eye(X.shape[1]),
            W=jnp.ones(X.shape[0]),
        )

    monkeypatch.setattr(pirls_module, "_beta_step", invalid_retained_step)
    pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    try:
        result = _call(
            fd,
            S_lambda,
            beta_old_init=jnp.zeros_like(fd.beta_init),
            initial_eta=fd.X @ fd.beta_init + fd.offset,
            initial_start_retained=True,
        )
    finally:
        pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    assert not bool(np.asarray(result.pirls_result.converged))
    assert int(np.asarray(result.status)) == _EFS_STATUS_RETAINED_START_INVALID_TRIAL


def test_efs_theta_pirls_final_factor_guard_overrides_stale_convergence(monkeypatch):
    """Final curvature failure cannot be reported as a selected-theta fit."""
    fd, S_lambda = _fixture()

    def nonfinite_factor(XtWX, _S):
        return jnp.full_like(XtWX, jnp.nan), jnp.array(jnp.nan)

    monkeypatch.setattr(pirls_module, "penalized_cholesky", nonfinite_factor)
    pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    try:
        result = _call(fd, S_lambda)
    finally:
        pirls_module._efs_theta_pirls_loop_jit.clear_cache()
    assert not bool(np.asarray(result.pirls_result.converged))
    assert int(np.asarray(result.status)) == _EFS_STATUS_INVALID_WORKING_FACTORS


def _r_efs_inner_oracle(
    X: np.ndarray,
    y: np.ndarray,
    wt: np.ndarray,
    offset: np.ndarray,
    penalty: float,
    log_theta: float,
) -> dict[str, np.ndarray | float]:
    """Trace pinned ``gam.fit4(scoreType='EFS')`` without namespace edits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_path = root / "data.csv"
        output_path = root / "output.csv"
        pre_path = root / "pre.csv"
        post_path = root / "post.csv"
        theta_path = root / "theta.csv"
        beta_path = root / "beta.csv"
        np.savetxt(
            data_path,
            np.column_stack((X, y, wt, offset)),
            delimiter=",",
            header=",".join([*(f"x{i}" for i in range(X.shape[1])), "y", "wt", "off"]),
            comments="",
        )
        x_columns = ", ".join(f"d[, {i + 1}]" for i in range(X.shape[1]))
        pre_anchor = 'if (scoreType == "EFS" && family$n.theta > 0) {'
        pre_replacement = (
            "trace$pre <- c(trace$pre, pdev); "
            "trace$theta_in <- c(trace$theta_in, theta[1]); "
            "trace$beta <- rbind(trace$beta, start); "
            'if (scoreType == "EFS" && family$n.theta > 0) {'
        )
        script = f"""
library(mgcv)
if (as.character(getRversion()) != "4.5.2" ||
    packageVersion("mgcv") != package_version("1.9.3")) {{
  stop("EFS inner oracle requires R 4.5.2 and mgcv 1.9-3")
}}
d <- as.matrix(read.csv({str(data_path)!r}, check.names=FALSE))
x <- cbind({x_columns})
y <- d[, {X.shape[1] + 1}]
wt <- d[, {X.shape[1] + 2}]
off <- d[, {X.shape[1] + 3}]
trace_env <- new.env(parent=asNamespace("mgcv"))
trace_env$trace <- new.env(parent=emptyenv())
trace <- trace_env$trace
trace$pre <- numeric(); trace$post <- numeric(); trace$theta_in <- numeric()
trace$theta_out <- numeric(); trace$beta <- matrix(numeric(), 0, ncol(x))
source_lines <- capture.output(mgcv:::gam.fit4)
source_lines <- source_lines[!grepl("^<", source_lines)]
source_lines[1] <- sub("^function", "gam.fit4.trace <- function", source_lines[1])
pre_anchor <- {pre_anchor!r}
pre_replacement <- {pre_replacement!r}
theta_anchor <- "family$putTheta(theta)"
replace_once <- function(old, new) {{
  if (sum(grepl(old, source_lines, fixed=TRUE)) != 1) stop(paste("anchor changed", old))
  source_lines <<- sub(old, new, source_lines, fixed=TRUE)
}}
if (sum(grepl(pre_anchor, source_lines, fixed=TRUE)) != 2)
  stop("pre anchor changed")
source_lines <- sub(pre_anchor, pre_replacement, source_lines, fixed=TRUE)
if (sum(grepl(theta_anchor, source_lines, fixed=TRUE)) != 2)
  stop("theta anchor changed")
source_lines <- gsub(
  theta_anchor,
  "family$putTheta(theta); trace$theta_out <- c(trace$theta_out, theta[1])",
  source_lines, fixed=TRUE
)
replace_once(
  'old.pdev <- pdev <- dev + penalty',
  'old.pdev <- pdev <- dev + penalty; trace$post <- c(trace$post, pdev)'
)
eval(parse(text=source_lines), envir=trace_env)
family <- nb(theta=-exp({float(log_theta)!r}))
family <- mgcv:::fix.family.ls(mgcv:::fix.family.var(mgcv:::fix.family.link(family)))
fit <- trace_env$gam.fit4.trace(
  x=x, y=y, sp=c({float(log_theta)!r}, log({float(penalty)!r})), Eb=diag(ncol(x)),
  UrS=list(diag(ncol(x))), weights=wt, offset=off, U1=diag(ncol(x)), Mp=0,
  family=family, control=gam.control(epsilon=1e-7, maxit=100), deriv=0,
  scoreType="EFS", scale=1, start=rep(0, ncol(x)), null.coef=rep(0, ncol(x))
)
write.csv(data.frame(coefficient=as.numeric(fit$coefficients)),
          {str(output_path)!r}, row.names=FALSE)
write.csv(data.frame(value=trace$pre), {str(pre_path)!r}, row.names=FALSE)
write.csv(data.frame(value=trace$post), {str(post_path)!r}, row.names=FALSE)
write.csv(data.frame(value=trace$theta_out),
          {str(theta_path)!r}, row.names=FALSE)
write.csv(as.data.frame(trace$beta), {str(beta_path)!r}, row.names=FALSE)
"""
        script_path = root / "oracle.R"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True, timeout=30
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        return {
            "coefficients": np.loadtxt(output_path, delimiter=",", skiprows=1),
            "pre": np.loadtxt(pre_path, delimiter=",", skiprows=1, ndmin=1),
            "post": np.loadtxt(post_path, delimiter=",", skiprows=1, ndmin=1),
            "theta": np.loadtxt(theta_path, delimiter=",", skiprows=1, ndmin=1),
            "beta": np.loadtxt(beta_path, delimiter=",", skiprows=1, ndmin=2),
        }


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_efs_theta_pirls_matches_pinned_fixed_penalty_inner_ordering():
    """Compare the dedicated loop to real pinned ``gam.fit4`` EFS internals."""
    rng = np.random.default_rng(5061)
    n = 38
    x = np.linspace(-1.0, 1.0, n)
    X = np.column_stack((np.ones(n), x))
    offset = 0.07 * np.cos(2.0 * x)
    wt = 0.6 + rng.random(n)
    mu = np.exp(offset + 0.2 + 0.35 * x)
    y = rng.negative_binomial(2.1, 2.1 / (2.1 + mu)).astype(np.float64)
    penalty = 0.18
    log_theta = np.log(0.8)
    oracle = _r_efs_inner_oracle(X, y, wt, offset, penalty, log_theta)

    assert oracle["beta"].shape[0] == oracle["pre"].size
    # The pinned EFS branch reaches the first theta condition twice per
    # accepted beta update (before/after its score bookkeeping). The paired
    # rows must be identical; theta changes only on the first row of each pair.
    theta_rows = np.arange(0, oracle["pre"].size, 2)
    paired_rows = theta_rows[:-1] + 1
    np.testing.assert_allclose(
        oracle["pre"][theta_rows[:-1]], oracle["pre"][paired_rows]
    )
    np.testing.assert_allclose(
        oracle["beta"][theta_rows[:-1]], oracle["beta"][paired_rows]
    )
    assert oracle["theta"].size == theta_rows.size + 1

    def fit_prefix(iteration: int):
        return efs_theta_pirls_loop(
            jnp.asarray(X),
            jnp.asarray(y),
            jnp.zeros(X.shape[1]),
            jnp.eye(X.shape[1]) * penalty,
            NegativeBinomial(theta=np.exp(log_theta)),
            jnp.asarray(wt),
            jnp.asarray(offset),
            jnp.asarray([log_theta]),
            jnp.asarray(y, dtype=jnp.int64),
            beta_old_init=jnp.zeros(X.shape[1]),
            max_y=int(np.max(y)),
            integer_counts=True,
            max_iter=iteration,
        )

    prefix_results = [
        fit_prefix(iteration) for iteration in range(1, theta_rows.size + 1)
    ]
    for index, result in enumerate(prefix_results):
        np.testing.assert_allclose(
            result.pirls_result.coefficients,
            oracle["beta"][theta_rows[index]],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        np.testing.assert_allclose(
            result.log_theta[0],
            oracle["theta"][index + 1],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        # ``stopping_penalized_deviance`` intentionally retains the pre-theta
        # value used by R's convergence predicate, separate from returned pdev.
        np.testing.assert_allclose(
            result.stopping_penalized_deviance,
            oracle["pre"][theta_rows[index]],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
    for index, value in enumerate(oracle["post"]):
        np.testing.assert_allclose(
            prefix_results[index].post_theta_penalized_deviance,
            value,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
    final = prefix_results[-1]
    assert bool(np.asarray(final.pirls_result.converged))
    np.testing.assert_allclose(
        final.pirls_result.coefficients,
        oracle["coefficients"],
        rtol=MODERATE.rtol,
        atol=MODERATE.atol,
    )
