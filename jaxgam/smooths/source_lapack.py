"""Pinned LAPACK provider for source-exact cubic and thin-plate constructors.

The cubic shrinkage and thin-plate constructors are sensitive to their numeric
provider's null-space coordinates.  This module binds only the required
LAPACKE/CBLAS entry points from one declared package,
``scipy-openblas32==0.3.26.0.4``, the OpenBLAS release the pinned Linux R
links: it serves the cubic shrinkage eigentransform, getFS's DPTSV, gdiPK's
DGEQP3, the source penalty-coordinate DSYEVR, the working model's pivoted
Cholesky (R ``chol(pivot=TRUE)``'s DPSTRF) and the ports of mgcv's
``Rlanczos`` (every source Lanczos call, Gaussian-process and thin-plate
bases alike), ``predict_tprs``, R's ``%*%`` and the tensor-margin
``qr``/``svd``/``eigen`` calls, so those calls reproduce R's.
It never searches for R or a system BLAS/LAPACK and does not replace
NumPy/SciPy's backend.

Threading: the wheel is a threaded OpenBLAS build whose default thread count
follows ``OPENBLAS_NUM_THREADS`` or the machine's core count, and threaded
reductions change last-bit results.  When the wheel library is loaded, this
module sets that library's own thread count to one
(``scipy_openblas_set_num_threads``) and verifies it, so every provider
result is the single-thread arithmetic validated against pinned R,
regardless of the caller's environment.  The setting is library-local: the
wheel is loaded ``RTLD_LOCAL`` and is separate from the BLAS that NumPy and
SciPy vendor, whose thread counts are unchanged.  It is made once at load,
never toggled around calls, so concurrent Python threads see one fixed value;
the threaded build locks internally, so concurrent provider calls are safe.
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
from numpy import __config__ as numpy_config
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
_LAPACK_COL_MAJOR = 102
_CBLAS_NO_TRANS = 111
_CBLAS_TRANS = 112
_CBLAS_UPPER = 121
_LAPACK_INT_MAX = np.iinfo(np.int32).max


def _build_uses_accelerate(config: object) -> bool:
    """Whether a NumPy/SciPy ``__config__`` module was built on Accelerate."""
    lapack = (
        getattr(config, "CONFIG", {}).get("Build Dependencies", {}).get("lapack", {})
    )
    return str(lapack.get("name", "")).lower() == "accelerate"


def scipy_uses_accelerate_lapack() -> bool:
    """Inspect the installed SciPy LAPACK provider, independent of the OS."""
    return _build_uses_accelerate(scipy_config)


def numpy_uses_accelerate_lapack() -> bool:
    """Inspect the installed NumPy LAPACK provider, independent of the OS."""
    return _build_uses_accelerate(numpy_config)


def accelerate_lapack_in_process() -> bool:
    """Whether NumPy or SciPy brings Apple Accelerate's LAPACK into the process.

    JAX's CPU LAPACK kernels take their function pointers from SciPy, and
    NumPy's linear algebra runs in the same process, so either build exposes
    the process to Accelerate.  Routing decisions that depend on SciPy's own
    LAPACK keep using :func:`scipy_uses_accelerate_lapack`.
    """
    return scipy_uses_accelerate_lapack() or numpy_uses_accelerate_lapack()


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
    thin-plate Rlanczos, prediction and tensor-margin ports call the same
    routines R calls (``dsymv``/``ddot``/``daxpy``/``dstedc``/``dgemv``/...).
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
    dpstrf_work: object


@dataclass(frozen=True)
class SourcePivotedCholesky:
    """R ``chol(pivot=TRUE)`` storage from the pinned provider's DPSTRF.

    ``upper`` is the upper triangle of DPSTRF's output with the lower
    triangle zeroed, as R's ``La_chol`` returns it; after a rank stop the
    trailing upper block keeps LAPACK's uncompleted residual. ``pivots`` are
    zero-based and ``rank`` is DPSTRF's completed rank.
    """

    upper: npt.NDArray[np.float64]
    pivots: npt.NDArray[np.int32]
    rank: int


def source_lanczos_workspace_bound(n: int, k: int) -> int:
    """Bound workspace beyond the retained symmetric input matrix.

    The peak owns the retained matrix copy, full Lanczos vectors, DSTEDC
    tridiagonal eigenvectors/work, the selected Ritz coefficients, output Ritz
    vectors, scalar work vectors, and integer work.  The caller separately
    owns the symmetric input matrix.  The bound is deliberately independent
    of convergence timing.
    """
    for name, value in (("n", n), ("k", k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Source Lanczos {name} must be a positive integer.")
    if k > n:
        raise ValueError("Source Lanczos k cannot exceed n.")
    if n > _LAPACK_INT_MAX:
        raise ValueError("Source Lanczos dimension exceeds the LP64 integer range.")
    return 8 * (7 * n * n + 2 * n * k + 19 * n + 35)


def source_pivoted_cholesky_workspace_bound(n: int) -> int:
    """Bound DPSTRF workspace beyond the caller's retained input matrix.

    The peak owns the column-major input copy that DPSTRF overwrites (and
    that is returned as the upper factor), its one-based integer pivots,
    LAPACK's ``2n`` real work array and the returned zero-based pivots.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("Pivoted Cholesky dimension must be a nonnegative integer.")
    if n > _LAPACK_INT_MAX:
        raise ValueError("Pivoted Cholesky dimension exceeds the LP64 integer range.")
    return 8 * n * n + 4 * n + 16 * n + 4 * n


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
        dpstrf_work = library.scipy_LAPACKE_dpstrf_work
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
    dpstrf_work.argtypes = [
        lapack_int,
        ctypes.c_char,
        lapack_int,
        double_pointer,
        lapack_int,
        int_pointer,
        int_pointer,
        double,
        double_pointer,
    ]
    dpstrf_work.restype = lapack_int
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
        dpstrf_work,
    )


def source_lanczos(
    matrix: npt.NDArray[np.float64],
    k: int,
    *,
    tol: float | None = None,
    available_workspace_bytes: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return mgcv's ``slanczos(A, k, kl=-1, tol)``: the exact Rlanczos port.

    smooth.construct.gp.smooth.spec and tprs_setup both call mat.c::Rlanczos
    for the ``k`` largest-magnitude eigenpairs; its Ritz vectors define their
    bases.  The port's BLAS/LAPACK calls go to the pinned OpenBLAS 0.3.26 and
    its loops keep mgcv's order and compiled contraction (see
    :func:`_source_rlanczos`).  The workspace bound is checked before any
    allocation.
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

    return _source_rlanczos(matrix, k, tol)


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


def _source_rlanczos(
    matrix: npt.NDArray[np.float64], k: int, tol: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Port mat.c::Rlanczos(A, U, D, n, m=k, lm=-1, tol, nt=1) exactly.

    slanczos asks for the ``k`` largest-magnitude eigenpairs.  The BLAS and
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


@dataclass(frozen=True)
class SourceConstraintQR:
    """LINPACK ``dqrdc2`` factor of one constraint column ``t(C)``.

    ``vector`` is the stored Householder column with ``qraux`` restored in its
    first entry, as ``dqrsl`` uses it; ``qraux == 0`` means ``Q = I``.
    """

    vector: npt.NDArray[np.float64]
    qraux: float

    @classmethod
    def from_column(cls, column: npt.NDArray[np.float64]) -> SourceConstraintQR:
        """Port ``qr(t(C))``: dqrdc2 with one column and R's default tolerance."""
        provider = _source_lapack()
        x = np.array(column, dtype=np.float64, copy=True)
        if x.ndim != 1 or len(x) < 2:
            raise ValueError("A source constraint column needs at least two rows.")
        n = len(x)
        norm = provider.dnrm2(n, _double_pointer(x), 1)
        if norm == 0.0:
            return cls(x, 0.0)
        if x[0] != 0.0:
            norm = float(np.copysign(norm, x[0]))
        provider.dscal(n, 1.0 / norm, _double_pointer(x), 1)
        x[0] = 1.0 + x[0]
        return cls(x, float(x[0]))

    def qty_rows(self, rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return ``t(qr.qty(qrc, t(rows)))``: ``Q'`` applied to every row."""
        provider = _source_lapack()
        out = np.array(rows, dtype=np.float64, order="C", copy=True)
        if self.qraux == 0.0:
            return out
        n = len(self.vector)
        if out.ndim != 2 or out.shape[1] != n:
            raise ValueError("Constraint rows must match the constraint length.")
        u = _double_pointer(self.vector)
        for i in range(out.shape[0]):
            y = _double_pointer(out[i])
            t = -provider.ddot(n, u, 1, y, 1) / self.vector[0]
            provider.daxpy(n, t, u, 1, y, 1)
        return out

    def absorb_rows(self, rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """smoothCon's ``t(qr.qty(qrc, t(X))[(j+1):k, ])``."""
        return self.qty_rows(rows)[:, 1:].copy()

    def basis(self) -> npt.NDArray[np.float64]:
        """Explicit null-space basis ``Z = Q[, -1]`` (``Q`` is one reflector)."""
        return self.qty_rows(np.eye(len(self.vector)))[:, 1:].copy()

    def absorb_penalty(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """smoothCon's two-sided ``Z'SZ`` via ``qr.qty`` (not symmetrized)."""
        left = self.qty_rows(np.asarray(S).T)[:, 1:].T
        return self.qty_rows(left)[:, 1:].copy()


def source_svd(
    matrix: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Port R's ``svd(x)``: La_svd's ``dgesdd`` with ``jobz='S'``.

    Returns ``(u, d, vt)``; La_svd truncates the workspace query to ``int``.
    """
    provider = _source_lapack()
    a = np.array(matrix, dtype=np.float64, order="F", copy=True)
    m, n = a.shape
    p = min(m, n)
    d = np.empty(p, dtype=np.float64)
    u = np.empty((m, p), dtype=np.float64, order="F")
    vt = np.empty((p, n), dtype=np.float64, order="F")
    iwork = np.empty(8 * p, dtype=np.int32)
    int_pointer = ctypes.POINTER(ctypes.c_int)

    def call(work, lwork) -> int:
        return int(
            provider.dgesdd_work(
                _LAPACK_COL_MAJOR,
                b"S",
                m,
                n,
                _double_pointer(a),
                m,
                _double_pointer(d),
                _double_pointer(u),
                m,
                _double_pointer(vt),
                p,
                _double_pointer(work),
                lwork,
                iwork.ctypes.data_as(int_pointer),
            )
        )

    query = np.empty(1, dtype=np.float64)
    if call(query, -1) != 0:
        raise np.linalg.LinAlgError("DGESDD workspace query failed.")
    # DGESDD's optimal query includes ILAENV block sizes; bound it generously.
    lwork = _query_length(
        float(int(query[0])),
        name="lwork",
        cap=4 * p * p + 7 * p + 64 * (m + n) + 4096,
    )
    status = call(np.empty(lwork, dtype=np.float64), lwork)
    if status != 0:
        raise np.linalg.LinAlgError(f"DGESDD failed with info={status}.")
    return u, d, vt


def source_matprod(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Port R's matrix-matrix ``%*%``: one column-major ``dgemm('N','N')``."""
    provider = _source_lapack()
    a = np.asfortranarray(left, dtype=np.float64)
    b = np.asfortranarray(right, dtype=np.float64)
    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError("Matrix product dimensions do not conform.")
    n = b.shape[1]
    out = np.zeros((m, n), dtype=np.float64, order="F")
    provider.dgemm(
        _LAPACK_COL_MAJOR,
        _CBLAS_NO_TRANS,
        _CBLAS_NO_TRANS,
        m,
        n,
        k,
        1.0,
        _double_pointer(a),
        max(m, 1),
        _double_pointer(b),
        max(k, 1),
        0.0,
        _double_pointer(out),
        max(m, 1),
    )
    return np.ascontiguousarray(out)


def source_r_matprod(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Port R's ``%*%`` (array.c ``matprod``) for finite double matrices.

    R's default matrix product calls ``dgemv('N')`` when the right operand has
    one column, ``dgemv('T')`` on the transposed right operand when the left
    operand has one row, and ``dgemm('N','N')`` otherwise; zero-extent products
    are zero.  Its NaN/Inf branch (a long-double triple loop) is not ported, so
    non-finite operands are rejected.
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Matrix product dimensions do not conform.")
    m, k = a.shape
    n = b.shape[1]
    if 0 in (m, k, n):
        return np.zeros((m, n), dtype=np.float64)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("Source matrix product operands must be finite.")
    if n != 1 and m != 1:
        return source_matprod(a, b)
    provider = _source_lapack()
    if n == 1:
        matrix = np.asfortranarray(a)
        vector = np.ascontiguousarray(b[:, 0])
        rows, columns, transpose, size = m, k, _CBLAS_NO_TRANS, m
    else:
        matrix = np.asfortranarray(b)
        vector = np.ascontiguousarray(a[0])
        rows, columns, transpose, size = k, n, _CBLAS_TRANS, n
    out = np.zeros(size, dtype=np.float64)
    provider.dgemv(
        _LAPACK_COL_MAJOR,
        transpose,
        rows,
        columns,
        1.0,
        _double_pointer(matrix),
        rows,
        _double_pointer(vector),
        1,
        0.0,
        _double_pointer(out),
        1,
    )
    return out.reshape(m, n)


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


def source_pivoted_cholesky(
    matrix: npt.NDArray[np.float64],
    *,
    tolerance: float = -1.0,
    available_workspace_bytes: int | None = None,
) -> SourcePivotedCholesky:
    """Pinned DPSTRF factor of a symmetric matrix's upper triangle.

    This is R's ``chol(x, pivot=TRUE)`` (``La_chol``) on the provider that
    pinned R's OpenBLAS 0.3.26 LAPACK uses: negative ``tolerance`` selects
    LAPACK's ``n * eps * max(diag)`` stop. The lower triangle is not read.
    The factor is a host operation with no derivatives; callers own any
    analytic trace or implicit-function derivatives.
    """
    if not isinstance(matrix, np.ndarray) or matrix.dtype != np.dtype(np.float64):
        raise TypeError("Pivoted Cholesky input must be a float64 NumPy array.")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Pivoted Cholesky input must be square.")
    n = matrix.shape[0]
    required = source_pivoted_cholesky_workspace_bound(n)
    if available_workspace_bytes is not None:
        if (
            isinstance(available_workspace_bytes, bool)
            or not isinstance(available_workspace_bytes, int)
            or available_workspace_bytes < 0
        ):
            raise ValueError("available_workspace_bytes must be a nonnegative integer.")
        if required > available_workspace_bytes:
            raise MemoryError(
                f"Pivoted Cholesky requires {required} workspace bytes; "
                f"{available_workspace_bytes} available."
            )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float, np.floating))
        or not np.isfinite(tolerance)
    ):
        raise ValueError("Pivoted Cholesky tolerance must be a finite real scalar.")
    if not np.all(np.isfinite(matrix[np.triu_indices(n)])):
        raise ValueError("Pivoted Cholesky upper triangle must be finite.")
    if n == 0:
        return SourcePivotedCholesky(np.empty((0, 0)), np.empty(0, dtype=np.int32), 0)
    provider = _source_lapack()
    lapack_int = ctypes.c_int
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(lapack_int)
    a = np.array(matrix, dtype=np.float64, order="F", copy=True)
    pivots = np.empty(n, dtype=np.int32)
    rank = lapack_int(0)
    work = np.empty(2 * n, dtype=np.float64)
    info = provider.dpstrf_work(
        _LAPACK_COL_MAJOR,
        b"U",
        n,
        a.ctypes.data_as(double_pointer),
        n,
        pivots.ctypes.data_as(int_pointer),
        ctypes.byref(rank),
        float(tolerance),
        work.ctypes.data_as(double_pointer),
    )
    del work
    if info < 0:
        raise RuntimeError(f"Pinned DPSTRF rejected argument {-info}.")
    # info > 0 is DPSTRF's rank stop, which R's chol(pivot=TRUE) only warns on.
    upper = np.triu(a)
    del a
    if not np.all(np.isfinite(upper)):
        raise FloatingPointError("Pinned DPSTRF produced a non-finite factor.")
    pivots -= 1
    return SourcePivotedCholesky(upper, pivots, int(rank.value))


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
