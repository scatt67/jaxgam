"""Pinned source penalty coordinates for Gaussian determinants and rank maps.

This CPU-only module consumes the accepted local LP64 source_lapack provider.
It also preserves the lower-triangle eigenvector gauge used by pinned mgcv
when a rank-deficient Gaussian fit needs canonical public coefficients.
"""

from __future__ import annotations

import ctypes

import numpy as np
import numpy.typing as npt
from scipy import __config__ as scipy_config
from scipy.linalg import blas

from jaxgam.smooths.source_lapack import (
    _LAPACK_COL_MAJOR,
    _LAPACK_INT_MAX,
    _query_length,
    _source_lapack,
)


def scipy_uses_accelerate_lapack() -> bool:
    """Inspect the installed SciPy LAPACK provider, independent of the OS."""
    lapack = (
        getattr(scipy_config, "CONFIG", {})
        .get("Build Dependencies", {})
        .get("lapack", {})
    )
    return lapack.get("name") == "Accelerate"


def _source_centering_reflector(design: npt.NDArray[np.float64]) -> np.ndarray:
    """Build R's one-row smooth constraint from extended-precision means."""
    n_rows, width = design.shape
    if n_rows < 1 or width < 2:
        raise ValueError("Source centering requires a nonempty smooth design.")
    # R's colMeans walks each column in row order with an LDOUBLE
    # accumulator. NumPy's sum may use a pairwise reduction instead.
    totals = np.zeros(width, dtype=np.longdouble)
    for row in design:
        np.add(totals, row, out=totals)
    means = np.asarray(totals / n_rows, dtype=np.float64)
    norm = blas.dnrm2(means)
    if means[0] != 0.0:
        norm = np.copysign(norm, means[0])
    if norm == 0.0:
        raise ValueError("Source centering constraint has zero length.")
    reflector = blas.dscal(1.0 / norm, means)
    reflector[0] += 1.0
    return reflector


def _source_centering_qty(
    reflector: np.ndarray, values: npt.NDArray[np.float64]
) -> np.ndarray:
    """Apply R's single Householder reflector by column."""
    result = np.array(values, dtype=np.float64, order="F", copy=True)
    for column in range(result.shape[1]):
        weight = -blas.ddot(reflector, result[:, column]) / reflector[0]
        result[:, column] = blas.daxpy(reflector, result[:, column], a=weight)
    return result


def _source_absorb_penalties(
    reflector: np.ndarray, penalties: tuple[npt.NDArray[np.float64], ...]
) -> tuple[np.ndarray, ...]:
    result = []
    for penalty in penalties:
        left = _source_centering_qty(reflector, penalty)[1:, :]
        result.append(_source_centering_qty(reflector, left.T)[1:, :].T)
    return tuple(result)


def source_centered_penalties(
    design: npt.NDArray[np.float64],
    penalties: tuple[npt.NDArray[np.float64], ...],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Absorb a source smooth constraint for determinant-only coordinates.

    R's ``colMeans`` accumulates extended precision by column. Its
    ``qr.qty`` then applies one packed Householder vector using BLAS. The
    resulting penalty coordinates are orthogonally equivalent to the fit
    coordinates, but preserve the source determinant's small eigenvalues.
    """
    reflector = _source_centering_reflector(design)
    return _source_absorb_penalties(reflector, penalties)


def source_centered_smooth(
    design: npt.NDArray[np.float64],
    penalties: tuple[npt.NDArray[np.float64], ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Absorb a te() centering constraint in R's penalty coordinate gauge.

    The design uses a matrix product with the source Householder map to avoid
    one Python loop per observation. Penalties retain R's ``qr.qty`` scalar
    order, which determines the canonical gauge at coupled rank ties.
    """
    reflector = _source_centering_reflector(design)
    width = design.shape[1]
    transform = _source_centering_qty(reflector, np.eye(width))[:, 1:]
    constrained = np.asarray(design) @ transform
    absorbed = _source_absorb_penalties(reflector, penalties)
    return constrained, absorbed, transform


def _source_symmetric_eigen(
    matrix: npt.NDArray[np.float64], *, vectors: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Use pinned LP64 DSYEVR on the lower triangle, preserving caller input."""
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype != np.float64
        or matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 1
    ):
        raise ValueError(
            "Source eigensolver requires a nonempty float64 square matrix."
        )
    k = matrix.shape[0]
    if k > _LAPACK_INT_MAX or not np.all(np.isfinite(matrix)):
        raise ValueError("Source eigensolver dimension or entries are invalid.")
    provider = _source_lapack()
    lapack_int = ctypes.c_int
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(lapack_int)
    a = np.array(matrix, dtype=np.float64, order="F", copy=True)
    values = np.empty(k, dtype=np.float64)
    eigenvectors = np.empty((k, k) if vectors else (1, 1), dtype=np.float64, order="F")
    support = np.empty(2 * k, dtype=np.int32)
    count = lapack_int(0)
    work_query = np.empty(1, dtype=np.float64)
    iwork_query = np.empty(1, dtype=np.int32)

    def arguments(work: np.ndarray, iwork: np.ndarray, lwork: int, liwork: int):
        return (
            _LAPACK_COL_MAJOR,
            b"V" if vectors else b"N",
            b"A",
            b"L",
            k,
            a.ctypes.data_as(double_pointer),
            k,
            0.0,
            0.0,
            0,
            0,
            0.0,
            ctypes.byref(count),
            values.ctypes.data_as(double_pointer),
            eigenvectors.ctypes.data_as(double_pointer),
            k if vectors else 1,
            support.ctypes.data_as(int_pointer),
            work.ctypes.data_as(double_pointer),
            lwork,
            iwork.ctypes.data_as(int_pointer),
            liwork,
        )

    status = provider.dsyevr_work(*arguments(work_query, iwork_query, -1, -1))
    if status != 0:
        raise np.linalg.LinAlgError(f"Source DSYEVR query failed with info={status}.")
    work = np.empty(_query_length(work_query[0], name="lwork", cap=64 * k))
    iwork = np.empty(
        _query_length(iwork_query[0], name="liwork", cap=32 * k), dtype=np.int32
    )
    np.copyto(a, matrix)
    count.value = 0
    status = provider.dsyevr_work(*arguments(work, iwork, len(work), len(iwork)))
    if status != 0 or count.value != k:
        raise np.linalg.LinAlgError(
            f"Source DSYEVR returned info={status}, eigenpairs={count.value}."
        )
    if not np.all(np.isfinite(values)) or (
        vectors and not np.all(np.isfinite(eigenvectors))
    ):
        raise np.linalg.LinAlgError("Source DSYEVR returned nonfinite eigenpairs.")
    return values, eigenvectors if vectors else None


def source_symmetric_eigh(
    matrix: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ascending eigenpairs using pinned mgcv's LP64 DSYEVR provider."""
    values, vectors = _source_symmetric_eigen(matrix, vectors=True)
    assert vectors is not None
    return values, vectors


def source_symmetric_max_eigenvalue(matrix: npt.NDArray[np.float64]) -> float:
    """Return R's value-only largest symmetric eigenvalue."""
    values, _ = _source_symmetric_eigen(matrix, vectors=False)
    return float(values[-1])
