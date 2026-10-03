"""Frozen signed systems, retained coordinates, and step-local recovery."""

import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.execution.signed_qr import signed_qr_update, solve_signed_qr
from jaxgam.fitting.signed_qr import SignedQRCoefficientFactor
from jaxgam.fitting.state import PivotedQRCoefficientFactor
from tests.tolerances import STRICT


def _reduce(X, w, z, batch_rows, weighted_response=None, use_weighted_response=False):
    state = None
    for start in range(0, len(X), batch_rows):
        stop = start + batch_rows
        state = signed_qr_update(
            state,
            X[start:stop],
            w[start:stop],
            z[start:stop],
            weighted_response=None
            if weighted_response is None
            else weighted_response[start:stop],
            use_weighted_response=use_weighted_response,
        )
    return state


def _factor(result):
    absolute = result.absolute_factor
    return SignedQRCoefficientFactor(
        PivotedQRCoefficientFactor(
            jnp.asarray(absolute.R),
            jnp.asarray(absolute.pivots),
            jnp.asarray(absolute.keep),
            absolute.original_n_coef,
        ),
        jnp.asarray(result.vectors),
        jnp.asarray(result.correction),
    )


def _gamma_identity_system():
    """Actual noncanonical Gamma observed weights, not an abs-weight model."""
    x = np.linspace(-1.0, 1.0, 67)
    X = np.column_stack((np.ones(len(x)), x, np.sin(2.0 * x)))
    mu = 2.0 + 0.2 * x
    y = mu * (0.7 + 0.4 * np.sin(7.0 * x))
    prior = np.linspace(0.5, 2.0, len(x))
    alpha = 2.0 * y / mu - 1.0
    w = prior * alpha / mu**2
    z = mu + (y - mu) / alpha
    E = np.diag([0.1, 0.6, 0.9])
    assert np.any(w < 0.0)
    return X, w, z, E


@pytest.mark.parametrize("batch_rows", [1, 7, 100])
def test_mixed_signed_system_matches_solve_and_jit_actions(batch_rows):
    X, w, z, E = _gamma_identity_system()
    state = _reduce(X, w, z, batch_rows)
    result = solve_signed_qr(state, [(slice(0, 3), E)])
    H = X.T @ (w[:, None] * X) + E.T @ E
    assert not result.fisher_required
    assert result.score_admissible
    np.testing.assert_allclose(
        result.coefficients,
        np.linalg.solve(H, X.T @ (w * z)),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    factor = _factor(result)
    rhs = jnp.asarray(np.column_stack((np.ones(3), np.arange(3))))
    solved, determinant, admissible = jax.jit(
        lambda f, b: (f.hessian_inverse(b), f.logdet_hessian(), f.score_admissible)
    )(factor, rhs)
    np.testing.assert_allclose(
        solved, np.linalg.solve(H, rhs), rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        determinant, np.linalg.slogdet(H)[1], rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert bool(admissible)
    root_inverse = jax.jit(lambda f: f.root_inverse(jnp.eye(f.rank)))(factor)
    np.testing.assert_allclose(
        root_inverse @ root_inverse.T,
        np.linalg.inv(H),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert state.absolute.R.shape == (3, 3)
    assert state.negative.R.shape[0] <= 3
    assert state.rhs.shape == (3,)


def test_indefinite_observed_requests_fisher_recovery():
    X = np.eye(3)
    observed = solve_signed_qr(_reduce(X, -np.ones(3), np.ones(3), 1), ())
    assert observed.fisher_required
    assert not observed.score_admissible
    assert observed.coefficients is None
    fisher = solve_signed_qr(_reduce(X, np.ones(3), np.ones(3), 1), ())
    assert not fisher.fisher_required
    assert fisher.score_admissible
    np.testing.assert_array_equal(fisher.coefficients, np.ones(3))


def test_zero_correction_keeps_coefficient_solve_separate_from_score():
    # Choose a negative rank-one root giving a tiny negative correction
    # within pls_fit1's zero-mode gate (not its indefinite rejection gate).
    X = np.ones((2, 1))
    w = np.array([1.0, -(1.0 + 16.0 * np.finfo(float).eps)])
    result = solve_signed_qr(_reduce(X, w, np.zeros(2), 1), ())
    assert not result.fisher_required
    assert not result.score_admissible
    np.testing.assert_array_equal(result.coefficients, [0.0])
    factor = _factor(result)
    value, determinant = jax.jit(
        lambda f: (f.hessian_inverse(jnp.ones(1)), f.logdet_hessian())
    )(factor)
    np.testing.assert_array_equal(value, [0.0])
    assert np.isneginf(determinant)


def test_stable_weighted_response_and_fixed_subspace():
    X = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    w = np.array([1.0, -0.1, 1.0])
    z = np.ones(3)
    wz = np.array([2.0, -0.3, 4.0])
    state = _reduce(X, w, z, 1, wz)
    result = solve_signed_qr(state, (), identifiable_subspace=[0, 1])
    assert result.used_direct_rhs
    np.testing.assert_array_equal(result.absolute_factor.keep, [0, 1])
    expected = np.linalg.solve(X[:, :2].T @ (w[:, None] * X[:, :2]), X[:, :2].T @ wz)
    np.testing.assert_allclose(
        result.coefficients[:2], expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert result.coefficients[2] == 0.0
    factor = _factor(result)
    value = jax.jit(lambda f: f.hessian_inverse(jnp.eye(3)))(factor)
    np.testing.assert_array_equal(value[2], np.zeros(3))
    np.testing.assert_array_equal(value[:, 2], np.zeros(3))


def test_empty_zero_and_invalid_rows():
    empty = signed_qr_update(None, np.empty((0, 2)), np.empty(0), np.empty(0))
    zero = signed_qr_update(empty, np.ones((1, 2)), np.zeros(1), np.array([np.inf]))
    np.testing.assert_array_equal(zero.rhs, np.zeros(2))
    with pytest.raises(ValueError, match="shapes"):
        signed_qr_update(None, np.ones((2, 2)), np.ones(1), np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        signed_qr_update(None, np.ones((1, 2)), np.ones(1), np.array([np.inf]))
    with pytest.raises(ValueError, match="dimension"):
        signed_qr_update(empty, np.ones((1, 3)), np.ones(1), np.ones(1))
    with pytest.raises(ValueError, match="Wz"):
        signed_qr_update(
            None,
            np.ones((1, 2)),
            np.zeros(1),
            np.zeros(1),
            weighted_response=np.array([np.nan]),
        )
    factor = _factor(solve_signed_qr(_reduce(np.eye(2), np.ones(2), np.ones(2), 1), ()))
    with pytest.raises(ValueError, match="rank"):
        factor.root_inverse(jnp.ones(3))
    with pytest.raises(ValueError, match="coefficient RHS"):
        factor.root_transpose_inverse(jnp.ones(3))


def test_explicit_direct_rhs_allows_source_zero_curvature_pseudodata():
    X = np.eye(2)
    state = signed_qr_update(
        None,
        X,
        np.array([0.0, 1.0]),
        np.array([np.inf, np.nan]),
        weighted_response=np.array([2.0, 3.0]),
        use_weighted_response=True,
    )
    result = solve_signed_qr(state, [(slice(0, 2), np.eye(2))])
    assert result.used_direct_rhs
    np.testing.assert_allclose(
        result.coefficients, [2.0, 1.5], rtol=STRICT.rtol, atol=STRICT.atol
    )
    with pytest.raises(ValueError, match="requires supplied Wz"):
        signed_qr_update(None, X, np.ones(2), np.ones(2), use_weighted_response=True)


def test_badly_scaled_pseudodata_select_stable_rhs_without_changing_weights():
    X = np.ones((3, 1))
    w = np.array([1e-300, 1.0, 1.0])
    z = np.array([1e300, 1.0, 0.0])
    fallbacks = []
    for batch_rows in (1, 2, 10):
        result = solve_signed_qr(_reduce(X, w, z, batch_rows), ())
        np.testing.assert_allclose(
            result.coefficients, [1.0], rtol=STRICT.rtol, atol=STRICT.atol
        )
        fallbacks.append(result.used_direct_rhs)
    assert any(fallbacks)


@pytest.mark.parametrize("n", [31, 3001])
def test_reducer_retains_only_coefficient_space(n):
    rng = np.random.default_rng(7419)
    X = rng.normal(size=(n, 4))
    w = np.where(np.arange(n) % 5 == 0, -0.01, 1.0)
    state = _reduce(X, w, rng.normal(size=n), 17)
    assert state.absolute.R.shape == (4, 4)
    assert state.negative.R.shape == (4, 4)
    assert state.rhs.shape == (4,)
    assert not state.rhs.flags.writeable


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize("case", ["observed", "indefinite", "direct", "scaled"])
def test_frozen_system_matches_pinned_pls_fit1(tmp_path, case):
    """Call the actual pinned signed fitter, including its recovery signal."""
    X, w, z, E = _gamma_identity_system()
    if case == "scaled":
        X = np.ones((3, 1))
        w = np.array([1e-300, 1.0, 1.0])
        z = np.array([1e300, 1.0, 0.0])
        E = np.array([[0.1]])
    if case == "indefinite":
        w = -np.ones(len(w))
    wz = w * z
    direct = case == "direct"
    if direct:
        # A zero observed-curvature row can still carry a finite derivative
        # RHS. Explicit use.wy bypasses its unavailable pseudodata in source.
        w[0] = 0.0
        z = np.zeros_like(z)
    for name, value in (
        ("X", X),
        ("w", w),
        ("z", z),
        ("wz", wz),
        ("E", E),
        ("direct", [int(direct)]),
    ):
        np.savetxt(tmp_path / name, value, fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
d <- commandArgs(TRUE)[1]
read <- function(name) as.matrix(read.table(file.path(d,name)))
X <- read("X"); w <- c(read("w")); z <- c(read("z")); E <- read("E")
wy <- c(read("wz")); direct <- c(read("direct"))
n <- nrow(X); p <- ncol(X)
oo <- .C(mgcv:::C_pls_fit1,y=as.double(z),X=as.double(X),w=as.double(w),
 wy=as.double(wy),
 E=as.double(E),Es=as.double(E),n=as.integer(n),q=as.integer(p),rE=as.integer(nrow(E)),
 eta=as.double(z),penalty=as.double(1),rank.tol=as.double(100*.Machine$double.eps),
 nt=as.integer(1),use.wy=as.integer(direct))
options(digits=17)
write.table(c(oo$n,oo$use.wy,oo$y[seq_len(p)]),file.path(d,"result"),
 row.names=FALSE,col.names=FALSE)
"""
    completed = subprocess.run(
        ["Rscript", "-e", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.returncode == 0
    reference = np.loadtxt(tmp_path / "result")
    for batch_rows in (1, 7, 100):
        state = _reduce(X, w, z, batch_rows, wz, direct)
        result = solve_signed_qr(state, [(slice(0, X.shape[1]), E)])
        assert result.fisher_required == (reference[0] < 0)
        if case != "indefinite":
            np.testing.assert_allclose(
                result.coefficients, reference[2:], rtol=STRICT.rtol, atol=STRICT.atol
            )
            if case != "scaled":
                assert result.used_direct_rhs == bool(reference[1])
            else:
                assert bool(reference[1])
