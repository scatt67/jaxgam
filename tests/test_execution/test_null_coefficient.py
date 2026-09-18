"""Pinned natural-column null projection and bounded compact state."""

import subprocess
from dataclasses import replace

import numpy as np
import pytest

from jaxgam.execution.null_coefficient import project_null_coefficients
from jaxgam.execution.qr import PositiveQRState, qr_update
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


def _compress(X, batch_rows):
    state = None
    for start in range(0, len(X), batch_rows):
        batch = X[start : start + batch_rows]
        state = qr_update(state, batch, np.ones(len(batch)))
    return state


@pytest.mark.usefixtures("r_bridge")
def test_compact_null_projection_preserves_pinned_rank_columns_and_coefficients(
    tmp_path,
):
    x = np.linspace(0.2, 1.6, 83)
    cases = [
        np.column_stack((x, x**2, np.exp(0.2 * x))),
        np.column_stack((np.ones(len(x)), x, -x, x**2)),
        np.column_stack((x, 2.0 * x)),
        np.zeros((len(x), 3)),
        np.column_stack((np.zeros(len(x)), x, 2.0 * x, np.ones(len(x)))),
        np.column_stack((np.ones(len(x)), x, x + 1e-8 * np.sin(x), x**2)),
        np.column_stack((np.ones(len(x)), x, x + 0.01 * np.sin(4.0 * x), x**2)),
        np.random.default_rng(881).normal(size=(4, 6)),
        np.array([[2.0, 4.0, 0.0]]),
    ]
    for index, X in enumerate(cases):
        np.savetxt(tmp_path / f"X{index}", X, fmt="%.17g")
    script = r"""
stopifnot(getRversion()=="4.5.2")
d <- commandArgs(TRUE)[1]
for (i in 0:8) {
 X <- as.matrix(read.table(file.path(d,paste0("X",i))))
 factor <- qr(X); coefficient <- qr.coef(factor,rep(2.3,nrow(X)))
 coefficient[is.na(coefficient)] <- 0
 writeBin(as.double(c(factor$rank,factor$pivot-1,coefficient)),
  file.path(d,paste0("reference",i)),size=8,endian="little")
}
"""
    subprocess.run(
        ["Rscript", "-e", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    checks = _AssertCollector()
    for index, X in enumerate(cases):
        p = X.shape[1]
        oracle = np.fromfile(tmp_path / f"reference{index}", dtype="<f8")
        for batch_rows in (1, 7, 100):
            state = _compress(X, batch_rows)
            result = project_null_coefficients(state, 2.3)
            checks.check(
                f"case{index}/B{batch_rows}/rank",
                lambda r=result, expected=int(oracle[0]): np.testing.assert_equal(
                    r.rank, expected
                ),
            )
            checks.check(
                f"case{index}/B{batch_rows}/full_column_order",
                lambda r=result, expected=oracle[1 : p + 1]: (
                    np.testing.assert_array_equal(r.pivots, expected)
                ),
            )
            checks.check(
                f"case{index}/B{batch_rows}/public_coefficients",
                lambda r=result, expected=oracle[p + 1 :]: np.testing.assert_allclose(
                    r.coefficients,
                    expected,
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
            assert result.coefficients.shape == (p,)
            assert state.R.shape[0] <= p
            assert not result.coefficients.flags.writeable
            assert not result.pivots.flags.writeable
    checks.raise_if_any("Pinned get.null.coef compact natural-column projection")


def test_null_projection_owns_finite_and_compact_statistics_validation():
    state = _compress(np.column_stack((np.ones(8), np.linspace(0.2, 1.0, 8))), 3)
    for constant in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite aligned"):
            project_null_coefficients(state, constant)
    for tolerance in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="finite aligned"):
            project_null_coefficients(state, 1.0, tolerance=tolerance)
    for bad in (
        replace(state, R=np.ones(2)),
        replace(state, R=np.ones((3, 2))),
        replace(state, qtz=np.ones(1)),
        replace(state, pivots=np.array([0, 0])),
        replace(state, pivots=np.array([0.0, 1.0])),
        replace(state, n_data_rows=0),
        replace(state, R=np.full((2, 2), np.inf)),
        replace(state, qtz=np.full(2, np.nan)),
    ):
        with pytest.raises(ValueError, match="null projection"):
            project_null_coefficients(bad, 1.0)
    huge_rhs = replace(state, qtz=np.full(2, 1e308))
    with pytest.raises(ValueError, match="RHS overflowed"):
        project_null_coefficients(huge_rhs, 10.0)
    huge_norm = replace(state, R=np.full((2, 2), 1.7e308))
    with pytest.raises(ValueError, match="norms overflowed"):
        project_null_coefficients(huge_norm, 1.0)
    empty_model = PositiveQRState(
        np.empty((0, 0)), np.empty(0), 0.0, 8, np.empty(0, int)
    )
    empty = project_null_coefficients(empty_model, 1.0)
    assert empty.rank == 0
    assert empty.coefficients.shape == empty.pivots.shape == (0,)
