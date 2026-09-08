"""CPU packed-reflector QR reducer behavior."""

import subprocess
import tracemalloc

import numpy as np
import pytest

from jaxgam.execution.qr import (
    _mgcv_r_cond,
    _triangular_rank,
    reduce_positive_qr,
    solve_augmented_qr,
)
from tests.helpers import _AssertCollector, r_available
from tests.tolerances import MODERATE, STRICT


def test_packed_reflector_tail_survives_huge_response_cancellation() -> None:
    """Tail is Q' residual energy, not c - ||Q'y|| squared."""
    rng = np.random.default_rng(900720)
    n, p = 5003, 7
    X = np.column_stack((np.ones(n), rng.normal(size=(n, p - 1))))
    response = X @ np.r_[1e8, rng.normal(size=p - 1)] + rng.normal(scale=1e-3, size=n)
    beta = np.linalg.lstsq(X, response, rcond=None)[0]
    expected = np.sum((response - X @ beta) ** 2)
    for batch_rows in (1, 17, 8192):
        state = reduce_positive_qr(
            (X[start : start + batch_rows], response[start : start + batch_rows])
            for start in range(0, n, batch_rows)
        )
        np.testing.assert_allclose(
            state.residual_tail,
            expected,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        # The old c - ||Q'y||^2 identity loses all meaningful digits here.
        assert abs(response @ response - state.qtz @ state.qtz) > 1.0


def test_augmented_qr_preserves_ill_conditioned_identifiable_systems() -> None:
    """A condition-1e8 positive system is not discarded as a structural alias."""
    rng = np.random.default_rng(900720)
    n, p = 137, 7
    left, _ = np.linalg.qr(rng.normal(size=(n, p)))
    right, _ = np.linalg.qr(rng.normal(size=(p, p)))
    X = left @ np.diag(np.geomspace(1.0, 1e-8, p)) @ right.T
    expected = np.linspace(1.0, 2.0, p)
    response = X @ expected
    checks = _AssertCollector()
    for batch_rows in (1, 3, 31, 256):
        state = reduce_positive_qr(
            (X[start : start + batch_rows], response[start : start + batch_rows])
            for start in range(0, n, batch_rows)
        )
        factor = solve_augmented_qr(state, ())
        checks.check(
            f"batch_rows={batch_rows}",
            lambda factor=factor: np.testing.assert_allclose(
                factor.coefficients,
                expected,
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
            ),
        )
        assert state.R.shape == (p, p)
        assert state.qtz.shape == (p,)
    checks.raise_if_any("ill-conditioned positive QR")
    normal_equation_beta = np.linalg.solve(X.T @ X, X.T @ response)
    assert np.max(np.abs(normal_equation_beta - expected)) > 1e-3


def test_reducer_workspace_is_bounded_as_row_count_grows() -> None:
    """A replay generator keeps retained state and traced reducer peak bounded."""
    p = 5

    def batches(n_rows: int):
        rng = np.random.default_rng(123)
        for start in range(0, n_rows, 3):
            rows = min(3, n_rows - start)
            yield rng.normal(size=(rows, p)), rng.normal(size=rows)

    tracemalloc.start()
    small = reduce_positive_qr(batches(13))
    _, small_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    large = reduce_positive_qr(batches(5003))
    _, large_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert small.n_data_rows == 13
    assert large.n_data_rows == 5003
    assert small.R.shape == large.R.shape == (p, p)
    assert small.qtz.shape == large.qtz.shape == (p,)
    # NumPy/SciPy temporaries are a function of p and batch size, not n.
    assert large_peak <= small_peak + 1_000_000


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv not available")
def test_pinned_bam_qr_update_fixed_system_oracle() -> None:
    """Compare normal equations and tail with mgcv's exact ``bam::qr_update``."""
    from tests.r_bridge import RBridge

    assert RBridge.check_versions()[0]
    X = np.array(
        [
            [1.0, -2.0, 0.5],
            [1.0, -1.0, -0.2],
            [1.0, 0.0, 1.3],
            [1.0, 0.5, 0.1],
            [1.0, 1.0, -1.4],
            [1.0, 2.0, 0.7],
        ]
    )
    response = np.array([1.1, -0.3, 0.8, 2.4, -1.2, 0.9])
    state = reduce_positive_qr(((X[:2], response[:2]), (X[2:], response[2:])))
    r_script = """
        suppressPackageStartupMessages(library(mgcv))
        update <- getFromNamespace("qr_update", "mgcv")
        X <- matrix(c(1,-2,.5, 1,-1,-.2, 1,0,1.3, 1,.5,.1,
                      1,1,-1.4, 1,2,.7), ncol=3, byrow=TRUE)
        y <- c(1.1,-.3,.8,2.4,-1.2,.9)
        q <- update(X[1:2,,drop=FALSE], y[1:2])
        q <- update(X[3:6,,drop=FALSE], y[3:6], q$R, q$f, q$y.norm2)
        root <- rbind(c(0, 1.5, 0), c(0, 0, 2))
        q_pen <- update(root, c(0, 0), q$R, q$f, q$y.norm2)
        fmt <- function(x) paste(sprintf("%.17g", x), collapse=",")
        cat(fmt(c(q$R)), "\\n", sep="")
        cat(fmt(q$f), "\\n", sep="")
        cat(sprintf("%.17g", q$y.norm2), "\\n", sep="")
        cat(fmt(solve(crossprod(q_pen$R), crossprod(q_pen$R, q_pen$f))), "\\n", sep="")
    """
    output = subprocess.run(
        ["Rscript", "-e", r_script],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    r_R = np.fromstring(output[-4], sep=",").reshape((3, 3), order="F")
    r_f = np.fromstring(output[-3], sep=",")
    r_norm = float(output[-2])
    r_penalized_coefficients = np.fromstring(output[-1], sep=",")
    python_R = np.empty_like(state.R)
    python_R[:, state.pivots] = state.R
    checks = _AssertCollector()
    checks.check(
        "crossproduct",
        lambda: np.testing.assert_allclose(
            python_R.T @ python_R, r_R.T @ r_R, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    checks.check(
        "crossproduct_rhs",
        lambda: np.testing.assert_allclose(
            python_R.T @ state.qtz,
            r_R.T @ r_f,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    checks.check(
        "residual_tail",
        lambda: np.testing.assert_allclose(
            state.residual_tail,
            r_norm - r_f @ r_f,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    factor = solve_augmented_qr(state, ((slice(1, 3), np.diag([1.5, 2.0])),))
    checks.check(
        "penalized_coefficients",
        lambda: np.testing.assert_allclose(
            factor.coefficients,
            r_penalized_coefficients,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    checks.raise_if_any("pinned bam qr_update")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv not available")
def test_pinned_r_cond_rank_policy_near_threshold() -> None:
    """Use mgcv's Cline--Moler--Stewart--Wilkinson estimator, not ``cond``."""
    threshold = 100.0 * np.finfo(float).eps
    matrices = (
        np.diag([1.0, threshold / 0.99]),
        np.diag([1.0, threshold / 1.01]),
    )
    r_script = """
        suppressPackageStartupMessages(library(mgcv))
        Rrank <- getFromNamespace("Rrank", "mgcv")
        tol <- 100 * .Machine$double.eps
        cat(Rrank(diag(c(1, tol / .99)), tol=tol), "\\n")
        cat(Rrank(diag(c(1, tol / 1.01)), tol=tol), "\\n")
    """
    output = subprocess.run(
        ["Rscript", "-e", r_script],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    r_ranks = [int(value) for value in output[-2:]]
    assert [_triangular_rank(matrix) for matrix in matrices] == r_ranks
    assert _mgcv_r_cond(matrices[0]) < 1.0 / threshold
    assert _mgcv_r_cond(matrices[1]) > 1.0 / threshold
