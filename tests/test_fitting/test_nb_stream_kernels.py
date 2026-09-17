"""Pinned NB same-state factors and bounded streamed reductions."""

import json
import subprocess
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.family_execution import FamilyExecutionParameters
from jaxgam.fitting.nb_stream_kernels import (
    merge_nb_working_summaries,
    nb_positive_observed_retry,
    nb_selected_working_rows,
    nb_working_batch,
    nb_working_statistics,
    nb_working_summary,
)
from tests.helpers import r_available
from tests.tolerances import STRICT


def _batch(link, theta, mu, y, wt=None, offset=None, valid=None):
    mu = np.asarray(mu, dtype=float)
    eta = np.log(mu) if link == "log" else np.sqrt(mu) if link == "sqrt" else mu
    n = len(mu)
    return jax.jit(partial(nb_working_batch, link=link))(
        jnp.asarray(eta),
        jnp.asarray(y, dtype=float),
        jnp.asarray(np.ones(n) if wt is None else wt),
        jnp.asarray(np.zeros(n) if offset is None else offset),
        jnp.asarray(np.ones(n, dtype=bool) if valid is None else valid),
        FamilyExecutionParameters(jnp.asarray([np.log(theta)])),
    )


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("theta", [0.1, 2.7, 1e6])
def test_same_state_source_factors(link, theta):
    """Binary R extraction covers tiny means, count tails and fractional rows."""
    ro = pytest.importorskip("rpy2.robjects")

    mu = np.array([1e-12, 0.1, 0.8, 2, 9, 1e16, 1e20])
    y = np.array([0, 0.25, 0.75, 1.5, 101, 0, 0])
    wt = np.array([0.8, 1, 1.2, 0.7, 2, 0.9, 1.1])
    offset = np.linspace(0.1, 0.7, len(mu))
    batch = _batch(link, theta, mu, y, wt, offset)
    oracle = ro.r("""function(link, theta, mu, eta, y, wt, offset) {
      fam <- mgcv:::fix.family.link(do.call(mgcv::nb,list(theta=theta,link=link)))
      dd <- mgcv:::dDeta(y,mu,wt,log(theta),fam,0)
      w <- .5*dd$Deta2
      list(w=w, fisher=.5*dd$EDeta2,
           wz=w*(eta-offset)-.5*dd$Deta,
           z=eta-offset-dd$Deta.Deta2,
           dev=sum(fam$dev.resids(y,mu,wt)))
    }""")(
        link,
        theta,
        ro.FloatVector(np.asarray(batch.mu)),
        ro.FloatVector(np.asarray(batch.eta)),
        ro.FloatVector(y),
        ro.FloatVector(wt),
        ro.FloatVector(offset),
    )
    for name, actual in (
        ("w", batch.observed_weight),
        ("fisher", batch.fisher_weight),
        ("wz", batch.weighted_response),
        ("z", batch.response),
        ("dev", batch.deviance),
    ):
        np.testing.assert_allclose(
            actual,
            np.asarray(oracle.rx2(name)),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
            err_msg=f"{link}/{theta}/{name}",
        )
    assert batch.domain_ok


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_batch_reduction_theta_isolation_and_padding(link):
    """One-row and padded scans equal the same full signed system under JIT."""
    mu = np.array([0.1, 0.7, 2, 5])
    y = np.array([0.0, 1, 3, 100])
    X = jnp.asarray([[1.0, -1], [1, -0.3], [1, 0.5], [1, 2]])
    full = _batch(link, 0.8, mu, y)
    expected = nb_working_statistics(X, full, use_weighted_response=jnp.asarray(True))
    total = [np.zeros((2, 2)), np.zeros(2)]
    summary = None
    for i in range(4):
        current = _batch(
            link,
            0.8,
            [mu[i], np.nan],
            [y[i], np.nan],
            [1.0, np.nan],
            [0.0, np.inf],
            [True, False],
        )
        stats = jax.jit(nb_working_statistics)(
            jnp.asarray([np.asarray(X[i]), [np.nan, np.inf]]),
            current,
            use_weighted_response=jnp.asarray(True),
        )
        total = [a + np.asarray(b) for a, b in zip(total, stats, strict=True)]
        leaf = nb_working_summary(current)
        summary = leaf if summary is None else merge_nb_working_summaries(summary, leaf)
    for actual, wanted in zip(total, expected, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=STRICT.rtol, atol=STRICT.atol)
    np.testing.assert_allclose(
        summary.deviance, full.deviance, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert summary.domain_ok
    assert len(jax.tree_util.tree_leaves(summary)) == 7
    changed = _batch(link, 100.0, mu, y)
    assert not np.allclose(changed.fisher_weight, full.fisher_weight)
    repeated = _batch(link, 0.8, mu, y)
    np.testing.assert_array_equal(repeated.observed_weight, full.observed_weight)
    np.testing.assert_array_equal(full.log_theta, [np.log(0.8)])


def test_positive_observed_retry_retains_derivative_rhs_under_jit():
    """NB retry is positive observed curvature, with RHS on discarded rows."""
    batch = _batch("identity", 2.7, [0.1, 2], [0, 4], offset=[0.3, 0.5])
    assert batch.observed_weight[0] < 0
    retry = jax.jit(nb_positive_observed_retry)(batch)
    assert retry.observed_weight[0] == 0
    np.testing.assert_array_equal(retry.weighted_response[0], batch.derivative_rhs[0])
    np.testing.assert_array_equal(retry.fisher_weight, batch.fisher_weight)
    _, rhs = jax.jit(nb_working_statistics)(
        jnp.eye(2), retry, use_weighted_response=jnp.asarray(True)
    )
    np.testing.assert_array_equal(rhs, retry.weighted_response)


def test_zero_weight_global_direct_selection_and_empty_neutrality():
    batch = _batch("identity", 2.7, [1.0, 2.0], [0.0, 1.0], wt=[0.0, 1.0])
    assert batch.requires_direct_response
    assert batch.direct_rows[0]
    assert not batch.normal_rows[0]
    _, response, wz = jax.jit(nb_selected_working_rows)(
        batch,
        use_weighted_response=jnp.asarray(True),
    )
    assert np.all(np.isfinite(response))
    assert np.all(np.isfinite(wz))
    normal = nb_working_statistics(
        jnp.eye(2), batch, use_weighted_response=jnp.asarray(False)
    )
    direct = nb_working_statistics(
        jnp.eye(2), batch, use_weighted_response=jnp.asarray(True)
    )
    for a, b in zip(normal, direct, strict=True):
        np.testing.assert_allclose(a, b, rtol=STRICT.rtol, atol=STRICT.atol)
    empty = _batch("log", 1.0, [], [])
    leaf = nb_working_summary(empty)
    assert leaf.domain_ok
    assert leaf.deviance == 0
    assert leaf.direct_informative_count == 0
    assert not leaf.requires_direct_response
    all_zero = _batch("sqrt", 2.0, [1.0, 2.0], [0.0, 7.0], wt=[0.0, 0.0])
    zero_summary = nb_working_summary(all_zero)
    assert zero_summary.normal_informative_count == 0
    assert zero_summary.direct_informative_count == 0
    assert zero_summary.deviance == 0
    merged = jax.jit(merge_nb_working_summaries)(leaf, zero_summary)
    assert merged.requires_direct_response
    assert merged.direct_informative_count == 0


def test_validation_fail_closed():
    bad = _batch("identity", 1.0, [-1.0, 2.0], [0, -1])
    assert not bad.domain_ok
    args = [jnp.ones(2)] * 4 + [jnp.ones(2, dtype=bool)]
    with pytest.raises(NotImplementedError, match="supports"):
        nb_working_batch(*args, FamilyExecutionParameters(jnp.zeros(1)), link="probit")
    with pytest.raises(ValueError, match="shape"):
        nb_working_batch(*args, FamilyExecutionParameters(jnp.zeros(2)), link="log")
    with pytest.raises(ValueError, match="aligned"):
        nb_working_batch(
            jnp.ones((2, 1)),
            *args[1:],
            FamilyExecutionParameters(jnp.zeros(1)),
            link="log",
        )
    batch = nb_working_batch(
        *args, FamilyExecutionParameters(jnp.asarray([jnp.inf])), link="log"
    )
    assert not batch.domain_ok


def test_zero_curvature_direct_rhs_and_family_count_metadata():
    """Direct use.wy survives nonfinite z; count planning retains no response."""
    batch = _batch("identity", 2.0, [1.0], [0.25], offset=[0.7])
    assert batch.observed_weight[0] == 0
    assert not np.isfinite(batch.response[0])
    assert batch.requires_direct_response
    assert batch.direct_rows[0]
    G, rhs = jax.jit(nb_working_statistics)(
        jnp.ones((1, 1)),
        batch,
        use_weighted_response=jnp.asarray(True),
    )
    assert G[0, 0] == 0
    np.testing.assert_allclose(rhs, [-0.5], rtol=STRICT.rtol, atol=STRICT.atol)
    assert nb_working_summary(batch).direct_informative_count == 0
    assert nb_working_summary(batch).direct_good_count == 1
    # Metadata belongs to the family; neither theta nor count magnitude
    # allocates a prefix/response copy inside these coefficient kernels.
    family = NegativeBinomial(theta=2.0, fixed=True)
    first = family.execution_summary_from_batch(
        np.array([0.0, 1e9]),
        np.ones(2),
        np.ones(2, dtype=bool),
    )
    second = family.execution_summary_from_batch(
        np.array([0.25, np.nan]),
        np.ones(2),
        np.array([True, False]),
    )
    merged = family.merge_execution_summaries(first, second)
    metadata = family.finalize_execution_summary(merged)
    assert metadata["max_count"] == 1e9
    assert not metadata["integer_counts"]
    assert len(merged) == 6
    assert all(np.ndim(leaf) == 0 for leaf in merged)
    huge = _batch("log", 1e6, [2.0], [1e9])
    assert all(leaf.size <= 1 for leaf in jax.tree_util.tree_leaves(huge))


def test_explicit_theta_isolated_in_fresh_process():
    """A compiled kernel takes theta dynamically, even with a fresh cache."""
    code = """
import json
from functools import partial
import jax
import jax.numpy as jnp
from jaxgam.fitting.family_execution import FamilyExecutionParameters
from jaxgam.fitting.nb_stream_kernels import nb_working_batch
run=jax.jit(partial(nb_working_batch,link="identity"))
args=(jnp.asarray([.5,2.]),jnp.asarray([0.,3.]),jnp.ones(2),jnp.zeros(2),jnp.ones(2,dtype=bool))
results=[]
for theta in [.1,1e6,.1]:
    value=run(*args,FamilyExecutionParameters(jnp.log(jnp.asarray([theta]))))
    results.append(value.fisher_weight.tolist())
print(json.dumps(results))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    actual = json.loads(completed.stdout)
    np.testing.assert_array_equal(actual[0], actual[2])
    for theta, weights in zip([0.1, 1e6], actual[:2], strict=True):
        expected = _batch("identity", theta, [0.5, 2.0], [0.0, 3.0]).fisher_weight
        np.testing.assert_allclose(
            weights, expected, rtol=STRICT.rtol, atol=STRICT.atol
        )


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
def test_source_direct_switch_and_positive_retry_at_zero_curvature():
    """Source use.wy and dropped-curvature RHS match at an exact singular row."""
    ro = pytest.importorskip("rpy2.robjects")
    batch = _batch("identity", 2.0, [0.1, 1.0], [0.0, 0.25], offset=[0.3, 0.7])
    oracle = ro.r("""function() {
      fam <- mgcv:::fix.family.link(mgcv::nb(theta=2,link="identity"))
      eta <- c(.1,1); offset <- c(.3,.7)
      dd <- mgcv:::dDeta(c(0,.25),eta,c(1,1),log(2),fam,0)
      w <- .5*dd$Deta2; z <- eta-offset-dd$Deta.Deta2
      use.wy <- any(!is.finite(w) | !is.finite(z))
      w[!is.finite(w) | w<=0] <- 0
      wz <- w*(eta-offset)-.5*dd$Deta
      good <- is.finite(w) & is.finite(wz)
      X <- matrix(c(1,2),ncol=1); S <- matrix(2,nrow=1)
      beta <- solve(crossprod(X,w*X)+S,crossprod(X,wz))
      list(use.wy=use.wy,w=w,wz=wz,good.count=sum(good),beta=beta)
    }""")()
    assert batch.requires_direct_response == bool(oracle.rx2("use.wy")[0])
    retry = jax.jit(nb_positive_observed_retry)(batch)
    for name, actual in (("w", retry.observed_weight), ("wz", retry.weighted_response)):
        np.testing.assert_allclose(
            actual, np.asarray(oracle.rx2(name)), rtol=STRICT.rtol, atol=STRICT.atol
        )
    summary = nb_working_summary(retry)
    assert summary.direct_informative_count == 0
    assert summary.direct_good_count == int(oracle.rx2("good.count")[0])
    assert summary.direct_good_count == 2
    G, rhs = jax.jit(nb_working_statistics)(
        jnp.asarray([[1.0], [2.0]]),
        retry,
        use_weighted_response=jnp.asarray(True),
    )
    beta = jax.jit(jnp.linalg.solve)(G + jnp.asarray([[2.0]]), rhs)
    np.testing.assert_allclose(
        beta, np.asarray(oracle.rx2("beta")).ravel(), rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert np.all(np.asarray(retry.weighted_response) != 0)
