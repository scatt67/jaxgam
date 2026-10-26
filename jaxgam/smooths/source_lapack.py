"""Pinned LAPACK provider for source-exact cubic and thin-plate constructors.

The cubic shrinkage and thin-plate constructors are sensitive to their numeric
provider's null-space coordinates.  This module binds only the required
LAPACKE/CBLAS entry points from two declared packages.
``scipy-openblas32==0.3.26.0.4`` is the OpenBLAS release the pinned Linux R
links: it serves the cubic shrinkage eigentransform and the thin-plate ports
of mgcv's ``Rlanczos`` and ``predict_tprs``, so those calls reproduce R's.  The
``scipy-openblas64==0.3.30.0.2`` provider serves the retained GP Lanczos
route.  It never searches for R or a system BLAS/LAPACK and does not replace
NumPy/SciPy's backend.

Threading: both wheels are threaded OpenBLAS builds whose default thread count
follows ``OPENBLAS_NUM_THREADS`` or the machine's core count, and threaded
reductions change last-bit results.  When each wheel library is loaded, this
module sets that library's own thread count to one
(``scipy_openblas_set_num_threads`` / ``scipy_openblas_set_num_threads64_``)
and verifies it, so every provider result is the single-thread arithmetic
validated against pinned R, regardless of the caller's environment.  The
setting is library-local: the wheels are loaded ``RTLD_LOCAL`` and are
separate from the BLAS that NumPy and SciPy vendor, whose thread counts are
unchanged.  It is made once at load, never toggled around calls, so
concurrent Python threads see one fixed value; these threaded builds lock
internally, so concurrent provider calls are safe.
"""

from __future__ import annotations

import ctypes
import functools
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import __config__ as scipy_config
from scipy import linalg

from jaxgam.smooths.source_arithmetic import (
    SOURCE_FMA_CONTRACTION,
    lanczos_start,
    lanczos_three_term,
    ritz_accumulate,
    sequential_dot,
)

_PROVIDER_VERSION = "0.3.26.0.4"
_OPENBLAS_PREFIX = "OpenBLAS 0.3.26 "
_LANCZOS_PROVIDER_VERSION = "0.3.30.0.2"
_LANCZOS_OPENBLAS_PREFIX = "OpenBLAS 0.3.30 "
_LAPACK_ROW_MAJOR = 101
_LAPACK_COL_MAJOR = 102
_CBLAS_ROW_MAJOR = 101
_CBLAS_NO_TRANS = 111
_CBLAS_TRANS = 112
_CBLAS_UPPER = 121
_LAPACK_INT_MAX = np.iinfo(np.int32).max
_LANCZOS_INT_MAX = np.iinfo(np.int64).max


def scipy_uses_accelerate_lapack() -> bool:
    """Inspect the installed SciPy LAPACK provider, independent of the OS."""
    lapack = (
        getattr(scipy_config, "CONFIG", {})
        .get("Build Dependencies", {})
        .get("lapack", {})
    )
    return lapack.get("name") == "Accelerate"


@dataclass(frozen=True)
class _DeclaredProvider:
    """Validated package metadata without executing its eager global loader."""

    package: str
    version: str
    config: str
    library_path: Path
    library_name: str
    num_threads: Callable[[], int]

    @property
    def __version__(self) -> str:
        return self.version

    def get_openblas_config(self) -> str:
        return self.config

    def get_lib_dir(self) -> str:
        return str(self.library_path.parent)

    def get_library(self) -> str:
        return self.library_name


@dataclass(frozen=True)
class _SourceLapack:
    """Validated LP64 function handles owned by the declared provider.

    OpenBLAS 0.3.26 is the BLAS/LAPACK linked by the pinned Linux R, so the
    thin-plate Rlanczos and prediction ports call the same routines R calls
    (``dsymv``/``ddot``/``daxpy``/``dstedc``/``dgemv``).
    """

    module: _DeclaredProvider
    library: ctypes.CDLL
    dsyevr_work: object
    dgemm: object
    dptsv: object
    dgeqp3_work: object
    dsymv: object
    ddot: object
    daxpy: object
    dgemv: object
    dnrm2: object
    dscal: object
    dstedc_work: object
    dgesdd_work: object


@dataclass(frozen=True)
class _SourceLanczos:
    """Pinned provider handles for the retained Lanczos implementation."""

    module: _DeclaredProvider
    library: ctypes.CDLL
    dsyevd_work: object
    dgemv: object
    dgemm: object
    ddot: object
    dnrm2: object


def source_lanczos_workspace_bound(n: int, k: int) -> int:
    """Bound workspace beyond the retained symmetric input matrix.

    The peak owns the retained matrix copy, full Lanczos vectors, a copied
    row-major prefix, DSYEVD eigenvectors/work (including the LAPACKE layout
    conversion), output Ritz vectors, scalar work vectors, and integer work.
    The caller separately owns the symmetric input matrix.  The bound is
    deliberately independent of convergence timing.
    """
    for name, value in (("n", n), ("k", k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Source Lanczos {name} must be a positive integer.")
    if k > n:
        raise ValueError("Source Lanczos k cannot exceed n.")
    if n > _LANCZOS_INT_MAX:
        raise ValueError("Source Lanczos dimension exceeds the ILP64 integer range.")
    return 8 * (7 * n * n + 2 * n * k + 19 * n + 35)


def cubic_shrinkage_workspace_bound(k: int) -> int:
    """Return a conservative peak byte bound beyond the retained input S.

    The eigensolve peak owns the overwritten input copy, eigenvectors, values,
    support indices, and capped DSYEVR work arrays on the pinned provider path.
    SciPy's Accelerate EVR path has the same quadratic matrix ownership.  The
    reconstruction peak owns eigenvectors, the scaled transposed factor, and
    the returned penalty.  Both providers' work queries are checked against
    the caps before either full work array is allocated.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 3:
        raise ValueError("Cubic shrinkage dimension must be an integer >= 3.")
    eigensolve_peak = 16 * k * k + (8 + 8 + 8 * 64 + 4 * 32) * k + 16
    reconstruction_peak = 24 * k * k + 16 * k
    return max(eigensolve_peak, reconstruction_peak)


def _provider_library_path(package: str, *, darwin_name: str, linux_name: str) -> Path:
    # Distribution names use hyphens while their import packages use
    # underscores.  ``find_spec`` locates the package without executing its
    # eager ``__init__`` (which would load OpenBLAS with ``RTLD_GLOBAL``).
    spec = importlib_util.find_spec(package.replace("-", "_"))
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Declared numeric provider {package!r} is unavailable.")
    library_dir = (Path(spec.origin).resolve().parent / "lib").resolve()
    if sys.platform == "darwin":
        library = library_dir / darwin_name
    elif sys.platform.startswith("linux"):
        library = library_dir / linux_name
    elif sys.platform == "win32":
        libraries = sorted(library_dir.glob("*.dll"))
        if len(libraries) != 1:
            raise RuntimeError(f"{package} must contain exactly one package-local DLL.")
        library = libraries[0]
    else:
        raise RuntimeError(f"No declared {package} wheel supports this platform.")
    library = library.resolve()
    if library.parent != library_dir or not library.is_file():
        raise RuntimeError(f"Pinned OpenBLAS library is unavailable at {library}.")
    return library


def _declared_provider(
    package: str,
    *,
    version: str,
    prefix: str,
    darwin_name: str,
    linux_name: str,
    library_name: str,
    config_symbol: str,
    ilp64: bool,
    thread_symbol_suffix: str = "",
) -> tuple[_DeclaredProvider, ctypes.CDLL]:
    try:
        actual_version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Declared numeric provider {package!r} is unavailable."
        ) from exc
    if actual_version != version:
        raise RuntimeError(f"{package}=={version} is required; found {actual_version}.")
    path = _provider_library_path(
        package, darwin_name=darwin_name, linux_name=linux_name
    )
    library = ctypes.CDLL(str(path), ctypes.RTLD_LOCAL)
    try:
        config_function = getattr(library, config_symbol)
    except AttributeError as exc:
        raise RuntimeError(f"{package} does not export {config_symbol}.") from exc
    config_function.argtypes = []
    config_function.restype = ctypes.c_char_p
    raw_config = config_function()
    if raw_config is None:
        raise RuntimeError(f"{package} returned an empty OpenBLAS configuration.")
    config = raw_config.decode()
    if not config.startswith(prefix):
        raise RuntimeError(f"{package} returned unexpected config {config!r}.")
    config_is_ilp64 = "USE64BITINT" in config.split()
    if config_is_ilp64 != ilp64:
        expected = "ILP64" if ilp64 else "LP64"
        raise RuntimeError(f"{package} does not provide the required {expected} ABI.")
    num_threads = _pin_single_thread(library, package, thread_symbol_suffix)
    provider = _DeclaredProvider(
        package, actual_version, config, path, library_name, num_threads
    )
    return provider, library


def _pin_single_thread(
    library: ctypes.CDLL, package: str, suffix: str
) -> Callable[[], int]:
    """Fix this wheel library's own OpenBLAS thread count at one, once."""
    try:
        set_threads = getattr(library, f"scipy_openblas_set_num_threads{suffix}")
        get_threads = getattr(library, f"scipy_openblas_get_num_threads{suffix}")
    except AttributeError as exc:
        raise RuntimeError(
            f"{package} does not export its OpenBLAS thread-count controls."
        ) from exc
    set_threads.argtypes = [ctypes.c_int]
    set_threads.restype = None
    get_threads.argtypes = []
    get_threads.restype = ctypes.c_int
    set_threads(1)
    if get_threads() != 1:
        raise RuntimeError(f"{package} did not accept a single-thread setting.")
    return get_threads


@functools.lru_cache(maxsize=1)
def _source_lapack() -> _SourceLapack:
    provider, library = _declared_provider(
        "scipy-openblas32",
        version=_PROVIDER_VERSION,
        prefix=_OPENBLAS_PREFIX,
        darwin_name="libscipy_openblas.dylib",
        linux_name="libscipy_openblas.so",
        library_name="scipy_openblas",
        config_symbol="scipy_openblas_get_config",
        ilp64=False,
    )
    lapack_int = ctypes.c_int
    double = ctypes.c_double
    int_pointer = ctypes.POINTER(lapack_int)
    double_pointer = ctypes.POINTER(double)
    try:
        dsyevr_work = library.scipy_LAPACKE_dsyevr_work
        dgemm = library.scipy_cblas_dgemm
        dptsv = library.scipy_LAPACKE_dptsv
        dgeqp3_work = library.scipy_LAPACKE_dgeqp3_work
        dsymv = library.scipy_cblas_dsymv
        ddot = library.scipy_cblas_ddot
        daxpy = library.scipy_cblas_daxpy
        dgemv = library.scipy_cblas_dgemv
        dnrm2 = library.scipy_cblas_dnrm2
        dscal = library.scipy_cblas_dscal
        dstedc_work = library.scipy_LAPACKE_dstedc_work
        dgesdd_work = library.scipy_LAPACKE_dgesdd_work
    except AttributeError as exc:
        raise RuntimeError(
            "Pinned OpenBLAS provider does not export the required LP64 "
            "LAPACKE/CBLAS symbols."
        ) from exc
    dsyevr_work.argtypes = [
        lapack_int,
        ctypes.c_char,
        ctypes.c_char,
        ctypes.c_char,
        lapack_int,
        double_pointer,
        lapack_int,
        double,
        double,
        lapack_int,
        lapack_int,
        double,
        int_pointer,
        double_pointer,
        double_pointer,
        lapack_int,
        int_pointer,
        double_pointer,
        lapack_int,
        int_pointer,
        lapack_int,
    ]
    dsyevr_work.restype = lapack_int
    dgemm.argtypes = [
        lapack_int,
        lapack_int,
        lapack_int,
        lapack_int,
        lapack_int,
        lapack_int,
        double,
        double_pointer,
        lapack_int,
        double_pointer,
        lapack_int,
        double,
        double_pointer,
        lapack_int,
    ]
    dgemm.restype = None
    dptsv.argtypes = [
        lapack_int,
        lapack_int,
        lapack_int,
        double_pointer,
        double_pointer,
        double_pointer,
        lapack_int,
    ]
    dptsv.restype = lapack_int
    dgeqp3_work.argtypes = [
        lapack_int,
        lapack_int,
        lapack_int,
        double_pointer,
        lapack_int,
        int_pointer,
        double_pointer,
        double_pointer,
        lapack_int,
    ]
    dgeqp3_work.restype = lapack_int
    dsymv.argtypes = [
        lapack_int,
        lapack_int,
        lapack_int,
        double,
        double_pointer,
        lapack_int,
        double_pointer,
        lapack_int,
        double,
        double_pointer,
        lapack_int,
    ]
    dsymv.restype = None
    ddot.argtypes = [lapack_int, double_pointer, lapack_int, double_pointer, lapack_int]
    ddot.restype = double
    daxpy.argtypes = [
        lapack_int,
        double,
        double_pointer,
        lapack_int,
        double_pointer,
        lapack_int,
    ]
    daxpy.restype = None
    dgemv.argtypes = [
        lapack_int,
        lapack_int,
        lapack_int,
        lapack_int,
        double,
        double_pointer,
        lapack_int,
        double_pointer,
        lapack_int,
        double,
        double_pointer,
        lapack_int,
    ]
    dgemv.restype = None
    dnrm2.argtypes = [lapack_int, double_pointer, lapack_int]
    dnrm2.restype = double
    dscal.argtypes = [lapack_int, double, double_pointer, lapack_int]
    dscal.restype = None
    dstedc_work.argtypes = [
        lapack_int,
        ctypes.c_char,
        lapack_int,
        double_pointer,
        double_pointer,
        double_pointer,
        lapack_int,
        double_pointer,
        lapack_int,
        int_pointer,
        lapack_int,
    ]
    dstedc_work.restype = lapack_int
    dgesdd_work.argtypes = [
        lapack_int,
        ctypes.c_char,
        lapack_int,
        lapack_int,
        double_pointer,
        lapack_int,
        double_pointer,
        double_pointer,
        lapack_int,
        double_pointer,
        lapack_int,
        double_pointer,
        lapack_int,
        int_pointer,
    ]
    dgesdd_work.restype = lapack_int
    return _SourceLapack(
        provider,
        library,
        dsyevr_work,
        dgemm,
        dptsv,
        dgeqp3_work,
        dsymv,
        ddot,
        daxpy,
        dgemv,
        dnrm2,
        dscal,
        dstedc_work,
        dgesdd_work,
    )


@functools.lru_cache(maxsize=1)
def _source_lanczos() -> _SourceLanczos:
    """Load only the source Rlanczos entry points from the pinned provider."""
    provider, library = _declared_provider(
        "scipy-openblas64",
        version=_LANCZOS_PROVIDER_VERSION,
        prefix=_LANCZOS_OPENBLAS_PREFIX,
        darwin_name="libscipy_openblas64_.dylib",
        linux_name="libscipy_openblas64_.so",
        library_name="scipy_openblas64_",
        config_symbol="scipy_openblas_get_config64_",
        ilp64=True,
        thread_symbol_suffix="64_",
    )
    enum_int = ctypes.c_int
    lapack_int = ctypes.c_int64
    double = ctypes.c_double
    int_pointer = ctypes.POINTER(lapack_int)
    double_pointer = ctypes.POINTER(double)
    try:
        dsyevd_work = library.scipy_LAPACKE_dsyevd_work64_
        dgemv = library.scipy_cblas_dgemv64_
        dgemm = library.scipy_cblas_dgemm64_
        ddot = library.scipy_cblas_ddot64_
        dnrm2 = library.scipy_cblas_dnrm264_
    except AttributeError as exc:
        raise RuntimeError(
            "Pinned OpenBLAS provider does not export the source Lanczos "
            "ILP64 LAPACKE/CBLAS symbols."
        ) from exc
    dsyevd_work.argtypes = [
        enum_int,
        ctypes.c_char,
        ctypes.c_char,
        lapack_int,
        double_pointer,
        lapack_int,
        double_pointer,
        double_pointer,
        lapack_int,
        int_pointer,
        lapack_int,
    ]
    dsyevd_work.restype = lapack_int
    dgemv.argtypes = [
        enum_int,
        enum_int,
        lapack_int,
        lapack_int,
        double,
        double_pointer,
        lapack_int,
        double_pointer,
        lapack_int,
        double,
        double_pointer,
        lapack_int,
    ]
    dgemv.restype = None
    dgemm.argtypes = [
        enum_int,
        enum_int,
        enum_int,
        lapack_int,
        lapack_int,
        lapack_int,
        double,
        double_pointer,
        lapack_int,
        double_pointer,
        lapack_int,
        double,
        double_pointer,
        lapack_int,
    ]
    dgemm.restype = None
    ddot.argtypes = [
        lapack_int,
        double_pointer,
        lapack_int,
        double_pointer,
        lapack_int,
    ]
    ddot.restype = double
    dnrm2.argtypes = [lapack_int, double_pointer, lapack_int]
    dnrm2.restype = double
    return _SourceLanczos(provider, library, dsyevd_work, dgemv, dgemm, ddot, dnrm2)


def source_lanczos(
    matrix: npt.NDArray[np.float64],
    k: int,
    *,
    tol: float | None = None,
    available_workspace_bytes: int | None = None,
    tprs_tridiagonal: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run the retained Rlanczos port on the declared numeric provider.

    The general GP route retains its pinned block reorthogonalization and
    dense tridiagonal eigensolve. ``tprs_tridiagonal=True`` is the exact
    mat.c::Rlanczos port that tprs_setup calls (pinned OpenBLAS 0.3.26 BLAS
    and DSTEDC, source loop order and contraction): its Ritz vectors define
    the thin-plate basis. Both routes use the same conservative knot-local
    workspace bound.
    """
    if not isinstance(matrix, np.ndarray) or matrix.dtype != np.dtype(np.float64):
        raise TypeError("Source Lanczos matrix must be a float64 NumPy array.")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Source Lanczos matrix must be square.")
    n = matrix.shape[0]
    required = source_lanczos_workspace_bound(n, k)
    if available_workspace_bytes is not None:
        if (
            isinstance(available_workspace_bytes, bool)
            or not isinstance(available_workspace_bytes, int)
            or available_workspace_bytes < 0
        ):
            raise ValueError("available_workspace_bytes must be a nonnegative integer.")
        if required > available_workspace_bytes:
            raise MemoryError(
                f"Source Lanczos requires {required} workspace bytes; "
                f"{available_workspace_bytes} available."
            )
    if tol is None:
        tol = np.finfo(float).eps ** 0.7
    if not np.isfinite(tol) or tol <= 0.0 or tol > 0.01:
        raise ValueError("Source Lanczos tolerance must lie in (0, 0.01].")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Source Lanczos matrix must contain only finite values.")

    if tprs_tridiagonal:
        return _source_tprs_lanczos(matrix, k, tol)

    source = _source_lanczos()
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int64)
    one = 1
    # Match the established Numba implementation's C-contiguous operands.
    a_matrix = np.array(matrix, dtype=np.float64, order="C", copy=True)
    q = np.empty((n, n), dtype=np.float64, order="C")
    jran = 1
    squared = 0.0
    for i in range(n):
        jran = (jran * 106 + 1283) % 6075
        value = jran / 6075.0 - 0.5
        q[i, 0] = value
        squared += value * value
    length = np.sqrt(squared)
    for i in range(n):
        q[i, 0] /= length

    alpha = np.empty(n, dtype=np.float64)
    beta = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    f_check = max(10, k // 2)
    f_check = min(f_check, max(1, n // 10))
    d = np.empty(0, dtype=np.float64)
    vectors = np.empty((0, 0), dtype=np.float64)
    j_final = n
    n_positive = 0
    n_negative = 0
    converged = False

    for j in range(n):
        qj = np.ascontiguousarray(q[:, j])
        source.dgemv(
            _CBLAS_ROW_MAJOR,
            _CBLAS_NO_TRANS,
            n,
            n,
            1.0,
            a_matrix.ctypes.data_as(double_pointer),
            n,
            qj.ctypes.data_as(double_pointer),
            one,
            0.0,
            z.ctypes.data_as(double_pointer),
            one,
        )
        value = source.ddot(
            n,
            qj.ctypes.data_as(double_pointer),
            one,
            z.ctypes.data_as(double_pointer),
            one,
        )
        del qj
        alpha[j] = value
        if j == 0:
            for i in range(n):
                z[i] -= value * q[i, j]
        else:
            previous = beta[j - 1]
            for i in range(n):
                z[i] -= value * q[i, j] + previous * q[i, j - 1]
            coefficients = np.empty(j + 1, dtype=np.float64)
            q_columns = np.ascontiguousarray(q[:, : j + 1])
            for _ in range(2):
                source.dgemv(
                    _CBLAS_ROW_MAJOR,
                    _CBLAS_TRANS,
                    n,
                    j + 1,
                    1.0,
                    q_columns.ctypes.data_as(double_pointer),
                    j + 1,
                    z.ctypes.data_as(double_pointer),
                    one,
                    0.0,
                    coefficients.ctypes.data_as(double_pointer),
                    one,
                )
                source.dgemv(
                    _CBLAS_ROW_MAJOR,
                    _CBLAS_NO_TRANS,
                    n,
                    j + 1,
                    -1.0,
                    q_columns.ctypes.data_as(double_pointer),
                    j + 1,
                    coefficients.ctypes.data_as(double_pointer),
                    one,
                    1.0,
                    z.ctypes.data_as(double_pointer),
                    one,
                )
            del q_columns, coefficients
        beta[j] = source.dnrm2(n, z.ctypes.data_as(double_pointer), one)
        if j < n - 1:
            for i in range(n):
                q[i, j + 1] = z[i] / beta[j]
        if not ((j >= k and j % f_check == 0) or j == n - 1):
            continue

        size = j + 1
        tridiagonal = np.zeros((size, size), dtype=np.float64, order="C")
        for index in range(size):
            tridiagonal[index, index] = alpha[index]
        for index in range(j):
            tridiagonal[index, index + 1] = beta[index]
            tridiagonal[index + 1, index] = beta[index]
        d = np.empty(size, dtype=np.float64)
        work_query = np.empty(1, dtype=np.float64)
        iwork_query = np.empty(1, dtype=np.int64)
        status = source.dsyevd_work(
            _LAPACK_ROW_MAJOR,
            b"V",
            b"L",
            size,
            tridiagonal.ctypes.data_as(double_pointer),
            size,
            d.ctypes.data_as(double_pointer),
            work_query.ctypes.data_as(double_pointer),
            -1,
            iwork_query.ctypes.data_as(int_pointer),
            -1,
        )
        if status != 0:
            raise np.linalg.LinAlgError(
                f"DSYEVD workspace query failed with info={status}."
            )
        lwork = _query_length(
            work_query[0],
            name="lwork",
            cap=max(2 * size * size + 6 * size + 1, 34 * size),
            integer_max=_LANCZOS_INT_MAX,
        )
        liwork = _query_length(
            iwork_query[0],
            name="liwork",
            cap=5 * size + 3,
            integer_max=_LANCZOS_INT_MAX,
        )
        work = np.empty(lwork, dtype=np.float64)
        iwork = np.empty(liwork, dtype=np.int64)
        status = source.dsyevd_work(
            _LAPACK_ROW_MAJOR,
            b"V",
            b"L",
            size,
            tridiagonal.ctypes.data_as(double_pointer),
            size,
            d.ctypes.data_as(double_pointer),
            work.ctypes.data_as(double_pointer),
            lwork,
            iwork.ctypes.data_as(int_pointer),
            liwork,
        )
        if status != 0:
            raise np.linalg.LinAlgError(f"DSYEVD failed with info={status}.")
        del work_query, iwork_query, work, iwork
        d = d[::-1].copy()
        vectors = tridiagonal[:, ::-1].copy(order="C")
        del tridiagonal
        norm = max(abs(d[0]), abs(d[-1]))
        errors = np.abs(beta[j] * vectors[-1, :])
        positive = negative = 0
        ok = True
        while positive + negative < k:
            if abs(d[positive]) >= abs(d[j - negative]):
                if errors[positive] > norm * tol:
                    ok = False
                    break
                positive += 1
            else:
                # This index is intentionally source ordered (mat.c Rlanczos).
                if errors[negative] > norm * tol:
                    ok = False
                    break
                negative += 1
        if ok:
            j_final = size
            n_positive = positive
            n_negative = negative
            converged = True
            break
        if j < n - 1:
            # Failed intermediate Ritz vectors are not needed by later iterations.
            d = np.empty(0, dtype=np.float64)
            vectors = np.empty((0, 0), dtype=np.float64)

    if not converged:
        positive = negative = 0
        while positive + negative < k:
            if abs(d[positive]) >= abs(d[n - 1 - negative]):
                positive += 1
            else:
                negative += 1
        n_positive = positive
        n_negative = negative
    selected = list(range(n_positive)) + list(range(j_final - n_negative, j_final))
    values = np.empty(k, dtype=np.float64)
    selected_vectors = np.empty((j_final, k), dtype=np.float64, order="C")
    for output_column, input_column in enumerate(selected):
        values[output_column] = d[input_column]
        selected_vectors[:, output_column] = vectors[:, input_column]
    del a_matrix, alpha, beta, z, d, vectors
    q_columns = np.ascontiguousarray(q[:, :j_final])
    output = np.empty((n, k), dtype=np.float64, order="C")
    source.dgemm(
        _CBLAS_ROW_MAJOR,
        _CBLAS_NO_TRANS,
        _CBLAS_NO_TRANS,
        n,
        k,
        j_final,
        1.0,
        q_columns.ctypes.data_as(double_pointer),
        j_final,
        selected_vectors.ctypes.data_as(double_pointer),
        k,
        0.0,
        output.ctypes.data_as(double_pointer),
        k,
    )
    return values, output


def _double_pointer(array: npt.NDArray[np.float64]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def source_trisymeig(
    diagonal: npt.NDArray[np.float64], off_diagonal: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Port mat.c::mgcv_trisymeig(d, g, v, n, getvec=1, descending=1).

    Calls the pinned DSTEDC with ``compz='I'`` and mgcv's workspace rounding,
    then reverses the ascending eigenpairs into descending order.
    """
    provider = _source_lapack()
    size = len(diagonal)
    values = np.array(diagonal, dtype=np.float64, copy=True)
    side = np.zeros(max(size - 1, 1), dtype=np.float64)
    side[: size - 1] = off_diagonal
    vectors = np.zeros((size, size), dtype=np.float64, order="F")
    work_query = np.empty(1, dtype=np.float64)
    iwork_query = np.empty(1, dtype=np.int32)
    int_pointer = ctypes.POINTER(ctypes.c_int)

    def call(work, lwork, iwork, liwork) -> int:
        return int(
            provider.dstedc_work(
                _LAPACK_COL_MAJOR,
                b"I",
                size,
                _double_pointer(values),
                _double_pointer(side),
                _double_pointer(vectors),
                size,
                _double_pointer(work),
                lwork,
                iwork.ctypes.data_as(int_pointer),
                liwork,
            )
        )

    if call(work_query, -1, iwork_query, -1) != 0:
        raise np.linalg.LinAlgError("DSTEDC workspace query failed.")
    # mgcv: lwork = floor(work1), rounded up when the fraction exceeds 0.5.
    lwork = int(np.floor(work_query[0]))
    if work_query[0] - lwork > 0.5:
        lwork += 1
    lwork = _query_length(
        float(lwork), name="lwork", cap=2 * size * size + 6 * size + 1
    )
    liwork = _query_length(iwork_query[0], name="liwork", cap=5 * size + 3)
    status = call(
        np.empty(lwork, dtype=np.float64),
        lwork,
        np.empty(liwork, dtype=np.int32),
        liwork,
    )
    if status != 0:
        raise np.linalg.LinAlgError(f"DSTEDC failed with info={status}.")
    return values[::-1].copy(), np.asfortranarray(vectors[:, ::-1])


def _source_tprs_lanczos(
    matrix: npt.NDArray[np.float64], k: int, tol: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Port mat.c::Rlanczos(A, U, D, n, m=k, lm=-1, tol, nt=1) exactly.

    Tprs_setup asks for the ``k`` largest-magnitude eigenpairs.  The BLAS and
    LAPACK calls (``dsymv``/``ddot``/``daxpy``/``dstedc``) go to the pinned
    OpenBLAS 0.3.26 that pinned R links, and the C loops keep mgcv's order and
    its compiled multiply-add rounding (see :mod:`source_arithmetic`).
    """
    provider = _source_lapack()
    contract = SOURCE_FMA_CONTRACTION
    n = matrix.shape[0]
    if n > _LAPACK_INT_MAX:
        raise ValueError("Source Lanczos dimension exceeds the LP64 integer range.")
    A = np.array(matrix, dtype=np.float64, order="F", copy=True)
    basis = np.zeros((n, n), dtype=np.float64, order="F")
    basis[:, 0] = lanczos_start(n, contract)
    diagonal = np.zeros(n, dtype=np.float64)
    off_diagonal = np.zeros(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    f_check = max(k // 2, 10)
    f_check = min(f_check, max(n // 10, 1))
    converged = False
    values = vectors = None
    upper = 0
    lower = 0
    size = n
    for j in range(n):
        current = basis[:, j]
        provider.dsymv(
            _LAPACK_COL_MAJOR,
            _CBLAS_UPPER,
            n,
            1.0,
            _double_pointer(A),
            n,
            _double_pointer(current),
            1,
            0.0,
            _double_pointer(z),
            1,
        )
        alpha = sequential_dot(current, z, 0.0, contract)
        diagonal[j] = alpha
        previous = basis[:, j - 1] if j else current
        beta = off_diagonal[j - 1] if j else 0.0
        lanczos_three_term(z, alpha, current, beta, previous, j == 0, contract)
        if j:
            # Full reorthogonalization, applied twice, as in Rlanczos.
            for _ in range(2):
                for i in range(j + 1):
                    prior = basis[:, i]
                    coefficient = -provider.ddot(
                        n, _double_pointer(z), 1, _double_pointer(prior), 1
                    )
                    provider.daxpy(
                        n, coefficient, _double_pointer(prior), 1, _double_pointer(z), 1
                    )
        off_diagonal[j] = np.sqrt(sequential_dot(z, z, 0.0, contract))
        if j < n - 1:
            basis[:, j + 1] = z / off_diagonal[j]
        if not ((j >= k and j % f_check == 0) or j == n - 1):
            continue
        values, vectors = source_trisymeig(diagonal[: j + 1], off_diagonal[:j])
        norm = max(abs(values[0]), abs(values[j]))
        errors = np.abs(off_diagonal[j] * vectors[j, :])
        if j >= k:
            max_error = norm * tol
            upper = lower = 0
            converged = True
            while upper + lower < k:
                if abs(values[upper]) >= abs(values[j - lower]):
                    if errors[upper] > max_error:
                        converged = False
                        break
                    upper += 1
                else:
                    # mat.c tests err[ni], not err[j-ni]; kept verbatim.
                    if errors[lower] > max_error:
                        converged = False
                        break
                    lower += 1
            if converged:
                size = j + 1
                break
    if not converged:
        # Without convergence Rlanczos keeps m=k, lm=0: the k leading pairs.
        upper, lower, size = k, 0, n
    selected = list(range(upper)) + [
        size - (upper + lower - c) for c in range(upper, k)
    ]
    output = np.zeros((n, k), dtype=np.float64)
    ritz_accumulate(
        output,
        np.asfortranarray(basis[:, :size]),
        np.ascontiguousarray(vectors[:size, selected]),
        contract,
    )
    return values[selected].copy(), output


def source_project_rows(
    rows: npt.NDArray[np.float64], UZ: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return ``rows @ UZ`` as predict_tprs/tps_g form it: one dgemv per row.

    mgcv evaluates ``a = t(UZ) b`` with ``dgemv('T', ...)`` for each datum;
    OpenBLAS' matrix-vector kernel rounds differently from a matrix product.
    """
    provider = _source_lapack()
    UZ = np.asfortranarray(UZ, dtype=np.float64)
    m, k = UZ.shape
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != m:
        raise ValueError("Projected rows must match the UZ row dimension.")
    output = np.empty((rows.shape[0], k), dtype=np.float64)
    target = np.empty(k, dtype=np.float64)
    uz = _double_pointer(UZ)
    out = _double_pointer(target)
    for i in range(rows.shape[0]):
        provider.dgemv(
            _LAPACK_COL_MAJOR,
            _CBLAS_TRANS,
            m,
            k,
            1.0,
            uz,
            m,
            _double_pointer(rows[i]),
            1,
            0.0,
            out,
            1,
        )
        output[i] = target
    return output


def _check_input_shape(S: npt.NDArray[np.float64]) -> int:
    if not isinstance(S, np.ndarray) or S.dtype != np.dtype(np.float64):
        raise TypeError("Cubic shrinkage penalty must be a float64 NumPy array.")
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] < 3:
        raise ValueError("Cubic shrinkage penalty must be square with k >= 3.")
    if S.shape[0] > _LAPACK_INT_MAX:
        raise ValueError("Cubic shrinkage dimension exceeds the LP64 integer range.")
    return S.shape[0]


def _check_input_finite(S: npt.NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(S)):
        raise ValueError("Cubic shrinkage penalty must contain only finite values.")


def source_pivoted_qr(
    matrix: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Run mgcv's ``mgcv_qr`` (LAPACK DGEQP3, all columns free) on the provider.

    Returns the ``min(m, n)`` by ``n`` upper-triangular factor in pivoted
    column order and the zero-based pivots, as ``mgcv_qr`` leaves them.  The
    caller's matrix is not modified.  Pinned mgcv's ``gdiPK`` breaks exactly
    tied (aliased) columns by DGEQP3's own norm downdating, so the call goes
    through the declared single-thread provider rather than SciPy's LAPACK.
    """
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype != np.float64
        or matrix.ndim != 2
        or 0 in matrix.shape
    ):
        raise ValueError("Source pivoted QR requires a nonempty float64 matrix.")
    rows, columns = matrix.shape
    if rows > _LAPACK_INT_MAX or columns > _LAPACK_INT_MAX:
        raise ValueError("Source pivoted QR dimension exceeds the LP64 integer range.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Source pivoted QR matrix must contain only finite values.")
    provider = _source_lapack()
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    factor = np.array(matrix, dtype=np.float64, order="F", copy=True)
    pivots = np.zeros(columns, dtype=np.int32)
    tau = np.zeros(min(rows, columns), dtype=np.float64)
    query = np.empty(1, dtype=np.float64)

    def call(work: np.ndarray, lwork: int) -> int:
        return provider.dgeqp3_work(
            _LAPACK_COL_MAJOR,
            rows,
            columns,
            factor.ctypes.data_as(double_pointer),
            rows,
            pivots.ctypes.data_as(int_pointer),
            tau.ctypes.data_as(double_pointer),
            work.ctypes.data_as(double_pointer),
            lwork,
        )

    status = call(query, -1)
    if status != 0:
        raise np.linalg.LinAlgError(
            f"DGEQP3 workspace query failed with info={status}."
        )
    lwork = _query_length(
        query[0], name="lwork", cap=3 * columns + 1 + 64 * (columns + 1)
    )
    work = np.empty(lwork, dtype=np.float64)
    status = call(work, lwork)
    if status != 0:
        raise np.linalg.LinAlgError(f"DGEQP3 failed with info={status}.")
    size = min(rows, columns)
    upper = np.triu(factor[:size, :])
    return upper, pivots.astype(np.int64) - 1


def source_frobenius_norm(matrix: npt.NDArray[np.float64]) -> float:
    """Match ``gdi.c::frobenius_norm``: one column-major running double sum."""
    values = np.asarray(matrix, dtype=np.float64).ravel(order="F")
    if values.size == 0:
        return 0.0
    squares = values * values
    return math.sqrt(float(np.cumsum(squares)[-1]))


def natural_cubic_system_solve(
    diagonal: npt.NDArray[np.float64],
    off_diagonal: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve pinned ``getFS``'s positive-definite tridiagonal system.

    Pinned mgcv calls Netlib ``DPTSV`` directly for this operation.  Calling
    the same entry point preserves its factorization and multiple-RHS
    arithmetic instead of approximating it with algebraically equivalent
    Python loops.
    """
    if not isinstance(diagonal, np.ndarray) or diagonal.dtype != np.float64:
        raise TypeError("Natural cubic diagonal must be a float64 NumPy array.")
    if not isinstance(off_diagonal, np.ndarray) or off_diagonal.dtype != np.float64:
        raise TypeError("Natural cubic off-diagonal must be a float64 NumPy array.")
    if not isinstance(rhs, np.ndarray) or rhs.dtype != np.float64:
        raise TypeError("Natural cubic right-hand side must be a float64 NumPy array.")
    if diagonal.ndim != 1 or diagonal.size < 1:
        raise ValueError("Natural cubic diagonal must be nonempty and one-dimensional.")
    n = diagonal.size
    if n > _LAPACK_INT_MAX:
        raise ValueError("Natural cubic dimension exceeds the LP64 integer range.")
    if off_diagonal.shape != (n - 1,):
        raise ValueError("Natural cubic off-diagonal has the wrong shape.")
    if rhs.ndim != 2 or rhs.shape[0] != n or rhs.shape[1] < 1:
        raise ValueError("Natural cubic right-hand side has the wrong shape.")
    if rhs.shape[1] > _LAPACK_INT_MAX:
        raise ValueError("Natural cubic right-hand side count exceeds LP64 range.")
    if not (
        np.all(np.isfinite(diagonal))
        and np.all(np.isfinite(off_diagonal))
        and np.all(np.isfinite(rhs))
    ):
        raise ValueError("Natural cubic system must contain only finite values.")

    diagonal_copy = np.array(diagonal, dtype=np.float64, order="C", copy=True)
    off_diagonal_copy = np.array(off_diagonal, dtype=np.float64, order="C", copy=True)
    solution = np.array(rhs, dtype=np.float64, order="F", copy=True)
    provider = _source_lapack()
    double_pointer = ctypes.POINTER(ctypes.c_double)
    info = provider.dptsv(
        _LAPACK_COL_MAJOR,
        n,
        solution.shape[1],
        diagonal_copy.ctypes.data_as(double_pointer),
        off_diagonal_copy.ctypes.data_as(double_pointer),
        solution.ctypes.data_as(double_pointer),
        n,
    )
    if info < 0:
        raise RuntimeError(f"Pinned DPTSV rejected argument {-info}.")
    if info > 0:
        raise np.linalg.LinAlgError(
            "Natural cubic knot system is not positive definite."
        )
    return solution


def _query_length(
    value: float | np.integer,
    *,
    name: str,
    cap: int,
    integer_max: int = _LAPACK_INT_MAX,
) -> int:
    numeric = float(value)
    if (
        not np.isfinite(numeric)
        or not numeric.is_integer()
        or numeric < 1.0
        or numeric > integer_max
    ):
        raise RuntimeError(
            f"Pinned DSYEVR returned an invalid LP64 {name} query: {value!r}."
        )
    result = int(numeric)
    if result > cap:
        raise RuntimeError(
            "Pinned DSYEVR returned workspace outside the declared bound: "
            f"{name}={result}, cap={cap}."
        )
    return result


def cubic_shrinkage_penalty(
    S: npt.NDArray[np.float64],
    *,
    shrink: float = 0.1,
    available_workspace_bytes: int | None = None,
) -> npt.NDArray[np.float64]:
    """Apply mgcv's source-ordered cubic shrinkage eigentransform.

    The provider returns ascending eigenpairs.  R's ``eigen`` returns them in
    descending order, and that order is load-bearing for repeated null values
    and GEMM accumulation.  The two null replacements are also intentionally
    assigned sequentially, as in ``smooth.r`` lines 1495--1498.
    """
    k = _check_input_shape(S)
    required = cubic_shrinkage_workspace_bound(k)
    if available_workspace_bytes is not None:
        if (
            isinstance(available_workspace_bytes, bool)
            or not isinstance(available_workspace_bytes, int)
            or available_workspace_bytes < 0
        ):
            raise ValueError("available_workspace_bytes must be a nonnegative integer.")
        if required > available_workspace_bytes:
            raise MemoryError(
                "Cubic shrinkage requires "
                f"{required} workspace bytes; {available_workspace_bytes} available."
            )
    if not np.isfinite(shrink) or shrink <= 0.0:
        raise ValueError("Cubic shrinkage factor must be finite and positive.")
    _check_input_finite(S)

    provider = _source_lapack()
    double = ctypes.c_double
    double_pointer = ctypes.POINTER(double)
    if scipy_uses_accelerate_lapack():
        # Accelerate EVR preserves R's separate libRlapack cubic null gauge
        # more closely.  The installed SciPy build selects this path.
        lwork, liwork, status = linalg.lapack.dsyevr_lwork(k, lower=1)
        if status != 0:
            raise np.linalg.LinAlgError(
                f"Accelerate DSYEVR workspace query failed with info={status}."
            )
        _query_length(lwork, name="lwork", cap=64 * k)
        _query_length(liwork, name="liwork", cap=32 * k)
        a = np.array(S, dtype=np.float64, order="F", copy=True)
        values, vectors = linalg.eigh(
            a, lower=True, driver="evr", overwrite_a=True, check_finite=False
        )
        del a
    else:
        lapack_int = ctypes.c_int
        int_pointer = ctypes.POINTER(lapack_int)
        a = np.array(S, dtype=np.float64, order="F", copy=True)
        values = np.empty(k, dtype=np.float64)
        vectors = np.empty((k, k), dtype=np.float64, order="F")
        support = np.empty(2 * k, dtype=np.int32)
        count = lapack_int(0)
        work_query = np.empty(1, dtype=np.float64)
        iwork_query = np.empty(1, dtype=np.int32)

        def arguments(
            matrix: np.ndarray,
            support_array: np.ndarray,
            work: np.ndarray,
            iwork: np.ndarray,
            lwork: int,
            liwork: int,
        ):
            return (
                _LAPACK_COL_MAJOR,
                b"V",
                b"A",
                b"L",
                k,
                matrix.ctypes.data_as(double_pointer),
                k,
                0.0,
                0.0,
                0,
                0,
                0.0,
                ctypes.byref(count),
                values.ctypes.data_as(double_pointer),
                vectors.ctypes.data_as(double_pointer),
                k,
                support_array.ctypes.data_as(int_pointer),
                work.ctypes.data_as(double_pointer),
                lwork,
                iwork.ctypes.data_as(int_pointer),
                liwork,
            )

        status = provider.dsyevr_work(
            *arguments(a, support, work_query, iwork_query, -1, -1)
        )
        if status != 0:
            raise np.linalg.LinAlgError(
                f"DSYEVR workspace query failed with info={status}."
            )
        lwork = _query_length(work_query[0], name="lwork", cap=64 * k)
        liwork = _query_length(iwork_query[0], name="liwork", cap=32 * k)
        work = np.empty(lwork, dtype=np.float64)
        iwork = np.empty(liwork, dtype=np.int32)
        np.copyto(a, S)
        count.value = 0
        status = provider.dsyevr_work(
            *arguments(a, support, work, iwork, lwork, liwork)
        )
        if status != 0:
            raise np.linalg.LinAlgError(f"DSYEVR failed with info={status}.")
        if count.value != k:
            raise np.linalg.LinAlgError(
                f"DSYEVR returned {count.value} eigenpairs for a {k} by {k} penalty."
            )
        del a, support, work_query, iwork_query, work, iwork
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise np.linalg.LinAlgError("DSYEVR returned non-finite eigenpairs.")

    values = values[::-1].copy()
    for left in range(k // 2):
        right_column = k - 1 - left
        temporary = vectors[:, left].copy()
        vectors[:, left] = vectors[:, right_column]
        vectors[:, right_column] = temporary
    values[-2] = values[-3] * shrink
    values[-1] = values[-2] * shrink
    right = np.empty((k, k), dtype=np.float64, order="F")
    np.multiply(values[:, None], vectors.T, out=right)
    penalty = np.empty((k, k), dtype=np.float64, order="F")
    provider.dgemm(
        _LAPACK_COL_MAJOR,
        _CBLAS_NO_TRANS,
        _CBLAS_NO_TRANS,
        k,
        k,
        k,
        1.0,
        vectors.ctypes.data_as(double_pointer),
        k,
        right.ctypes.data_as(double_pointer),
        k,
        0.0,
        penalty.ctypes.data_as(double_pointer),
        k,
    )
    return penalty
