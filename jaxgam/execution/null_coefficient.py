"""CPU source-column null projection from bounded unweighted QR statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import blas

from jaxgam.execution.qr import PositiveQRState


@dataclass(frozen=True)
class NullCoefficientProjection:
    """Public-coordinate coefficients and R's natural-column retained rank."""

    coefficients: np.ndarray
    rank: int
    pivots: np.ndarray

    def __post_init__(self) -> None:
        for name in ("coefficients", "pivots"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def project_null_coefficients(
    state: PositiveQRState,
    constant_eta: float,
    *,
    tolerance: float = 1e-7,
) -> NullCoefficientProjection:
    """Reproduce ``get.null.coef``'s unweighted ``qr.coef`` column policy.

    ``state`` must compress public-coordinate X and an all-ones response.
    Its pivoted compression is undone before R 4.5.2's ``dqrdc2`` policy:
    cycle relatively negligible columns to the end, preserving the order of
    retained columns. ``dqrsl`` transforms the constant RHS and back-solves;
    source NA coefficients become zero. The caller owns row lineage and the
    subsequent conversion to local fitting coordinates.

    Work retains only O(p²) coefficient arrays and O(p) vectors. It creates
    no row-sized array, explicit Q, normal matrix or regularization jitter.
    LAPACK/BLAS implementation scratch is outside this visible-array bound.
    """
    R = np.asarray(state.R, dtype=np.float64)
    qtz = np.asarray(state.qtz, dtype=np.float64)
    pivots = np.asarray(state.pivots)
    if R.ndim != 2:
        raise ValueError("null projection requires a compact QR matrix")
    n, p = R.shape
    if (
        n != min(state.n_data_rows, p)
        or state.n_data_rows < n
        or state.n_data_rows <= 0
        or qtz.shape != (n,)
        or pivots.shape != (p,)
        or not np.issubdtype(pivots.dtype, np.integer)
        or not np.array_equal(np.sort(pivots), np.arange(p))
        or not np.all(np.isfinite(R))
        or not np.all(np.isfinite(qtz))
        or not np.isfinite(constant_eta)
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("null projection requires finite aligned QR statistics")
    x = np.empty((n, p), dtype=np.float64, order="F")
    x[:, pivots] = R
    with np.errstate(over="ignore", invalid="ignore"):
        rhs = qtz * constant_eta
    if not np.all(np.isfinite(rhs)):
        raise ValueError("null projection constant RHS overflowed")
    original_norm = np.array([blas.dnrm2(x[:, j]) for j in range(p)])
    if not np.all(np.isfinite(original_norm)):
        raise ValueError("null projection column norms overflowed")
    norm = original_norm.copy()
    original_norm[original_norm == 0.0] = 1.0
    order = np.arange(p)
    active = p
    # dqrdc2 checks residual norms relative to each original column norm,
    # rather than sorting by the largest residual norm as LAPACK does.
    for column in range(n):
        while column < active and norm[column] < original_norm[column] * tolerance:
            dropped = x[:, column].copy()
            x[:, column:-1] = x[:, column + 1 :]
            x[:, -1] = dropped
            for vector in (order, norm, original_norm):
                value = vector[column]
                vector[column:-1] = vector[column + 1 :]
                vector[-1] = value
            active -= 1
        if column == n - 1:
            continue
        length = blas.dnrm2(x[column:, column])
        if length == 0.0:
            continue
        if x[column, column] != 0.0:
            length = np.copysign(length, x[column, column])
        x[column:, column] = blas.dscal(1.0 / length, x[column:, column])
        x[column, column] += 1.0
        householder = x[column:, column]
        for other in range(column + 1, p):
            weight = -blas.ddot(householder, x[column:, other]) / householder[0]
            x[column:, other] = blas.daxpy(householder, x[column:, other], a=weight)
            if norm[other] != 0.0:
                reduction = max(1.0 - (abs(x[column, other]) / norm[other]) ** 2, 0.0)
                norm[other] = (
                    norm[other] * np.sqrt(reduction)
                    if reduction >= 1e-6
                    else blas.dnrm2(x[column + 1 :, other])
                )
        # dqrsl applies these same Householders to Q'ones * constant_eta.
        weight = -blas.ddot(householder, rhs[column:]) / householder[0]
        rhs[column:] = blas.daxpy(householder, rhs[column:], a=weight)
        x[column, column] = -length
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(rhs)):
        raise ValueError("source null projection has nonfinite factor statistics")
    rank = min(active, n)
    reduced = rhs[:rank].copy()
    for column in range(rank - 1, -1, -1):
        if x[column, column] == 0.0:
            raise np.linalg.LinAlgError(
                "source null projection retained a zero diagonal"
            )
        reduced[column] /= x[column, column]
        if column:
            reduced[:column] = blas.daxpy(
                x[:column, column], reduced[:column], a=-reduced[column]
            )
    if not np.all(np.isfinite(reduced)):
        raise ValueError("source null projection has nonfinite coefficients")
    coefficients = np.zeros(p)
    coefficients[order[:rank]] = reduced
    return NullCoefficientProjection(coefficients, rank, order)
