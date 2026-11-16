"""Source-faithful scalar loops ported from mgcv's compiled C code.

The pinned mgcv 1.9-3 ``tprs.c``/``mat.c``/``matrix.c`` loops accumulate
``s += a*b`` and update ``c -= a*b`` in fixed index order.  How those
statements round depends on how R's mgcv shared object was compiled: on
AArch64, gcc -O2 contracts them into fused multiply-adds (``objdump`` of the
pinned build shows ``fmadd``/``fmsub`` in ``Rlanczos``, ``HQmult``, ``QT``,
``matmult``, ``tpsE``, ``predict_tprs`` and ``tprs_setup``), while the x86-64
build of the same source contains no fused instructions.  These kernels keep
the source order and apply the same rounding: one rounding for a contracted
statement, two otherwise.

The kernels are Numba-compiled without ``fastmath``, so LLVM never reorders
or contracts the plain expressions; the fused form uses the exact
``llvm.fma`` intrinsic.  Phase 1 module: NumPy/Numba only, no JAX.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import platform

import numba
import numpy as np
import numpy.typing as npt
from llvmlite import ir
from numba import types
from numba.extending import intrinsic

#: Whether R's compiled mgcv loops contract ``a*b + c`` on this machine.
#:
#: Static evidence (objdump -d of the pinned mgcv 1.9-3 ``mgcv.so``, 2026-10-24):
#: the aarch64 Linux build (gcc -std=gnu2x -O2, R 4.5.2 rocker image) contains
#: fmadd/fmsub in Rlanczos (mat.c:3689, 3743, 3749, 3753, 3780, 3856, 3863),
#: HQmult (matrix.c:355-386), QT (matrix.c:420-439), matmult (matrix.c:140-170),
#: tpsE (tprs.c:92), tprs_setup (tprs.c:494) and predict_tprs (tprs.c:591);
#: the macOS arm64 CRAN build also contracts these loops (23 fused instructions
#: in Rlanczos). The amd64 Linux build of the same source contains no
#: vfmadd/vfmsub/vfnmadd/vfnmsub instruction anywhere. Both modes are tested
#: against scalar transcriptions and a correctly rounded rational reference;
#: native x86-64 CI is the authoritative amd64 parity check.
SOURCE_FMA_CONTRACTION: bool = platform.machine().lower() in {"aarch64", "arm64"}


@intrinsic
def _fused(_typingctx, a, b, c):  # noqa: ARG001  # pragma: no cover - Numba typing
    """Exact ``fma(a, b, c)`` with a single rounding."""
    signature = types.float64(types.float64, types.float64, types.float64)

    def codegen(context, builder, sig, args):  # noqa: ARG001
        double = ir.DoubleType()
        function = builder.module.declare_intrinsic(
            "llvm.fma", [double], ir.FunctionType(double, [double] * 3)
        )
        return builder.call(function, args)

    return signature, codegen


@numba.njit(cache=True)
def _madd(a: float, b: float, c: float, contract: bool) -> float:  # pragma: no cover
    """Source ``c + a*b`` (``c += a*b``)."""
    if contract:
        return _fused(a, b, c)
    return c + a * b


@numba.njit(cache=True)
def _msub(c: float, a: float, b: float, contract: bool) -> float:  # pragma: no cover
    """Source ``c - a*b`` (``c -= a*b``)."""
    if contract:
        return _fused(-a, b, c)
    return c - a * b


@numba.njit(cache=True)
def fused_multiply_add(a: float, b: float, c: float) -> float:  # pragma: no cover
    """Exact single-rounding ``a*b + c`` (exposed for verification)."""
    return _fused(a, b, c)


@numba.njit(cache=True)
def sequential_dot(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    start: float,
    contract: bool,
) -> float:  # pragma: no cover
    """``for i: s += x[i]*y[i]`` starting from ``start``."""
    total = start
    for i in range(x.shape[0]):
        total = _madd(x[i], y[i], total, contract)
    return total


@numba.njit(cache=True)
def column_squares(
    rows: npt.NDArray[np.float64],
    totals: npt.NDArray[np.float64],
    contract: bool,
) -> None:  # pragma: no cover
    """tprs_setup's ``w += X[j][i]*X[j][i]``, rows in source order, in place."""
    for j in range(rows.shape[0]):
        for i in range(rows.shape[1]):
            totals[i] = _madd(rows[j, i], rows[j, i], totals[i], contract)


@numba.njit(cache=True)
def squared_distances(
    rows: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    contract: bool,
) -> npt.NDArray[np.float64]:  # pragma: no cover
    """predict_tprs/tps_g ``z = knot - x; r += z*z`` over covariates in order."""
    n, nk, d = rows.shape[0], knots.shape[0], knots.shape[1]
    out = np.zeros((n, nk))
    for i in range(n):
        for j in range(nk):
            total = 0.0
            for column in range(d):
                delta = knots[j, column] - rows[i, column]
                total = _madd(delta, delta, total, contract)
            out[i, j] = total
    return out


@numba.njit(cache=True)
def knot_squared_distances(
    knots: npt.NDArray[np.float64], contract: bool
) -> npt.NDArray[np.float64]:  # pragma: no cover
    """tpsE ``x = X[i]-X[j]; r += x*x`` for ``j < i``, mirrored."""
    n, d = knots.shape
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            total = 0.0
            for column in range(d):
                delta = knots[i, column] - knots[j, column]
                total = _madd(delta, delta, total, contract)
            out[i, j] = total
            out[j, i] = total
    return out


@numba.njit(cache=True)
def matmult_tn(
    A: npt.NDArray[np.float64], B: npt.NDArray[np.float64], contract: bool
) -> npt.NDArray[np.float64]:  # pragma: no cover
    """matrix.c::matmult(C, A, B, 1, 0): ``C = A' B``."""
    C = np.zeros((A.shape[1], B.shape[1]))
    for k in range(A.shape[0]):
        for i in range(A.shape[1]):
            temp = A[k, i]
            for p in range(B.shape[1]):
                C[i, p] = _madd(temp, B[k, p], C[i, p], contract)
    return C


@numba.njit(cache=True)
def qt_factor(
    A: npt.NDArray[np.float64], full: bool, contract: bool
) -> npt.NDArray[np.float64]:  # pragma: no cover
    """matrix.c::QT on a copy of ``A`` (``A.r <= A.c``).

    Returns the scaled Householder rows (``full=False``) or the explicit ``Q``
    with ``A Q = [0, T]`` (``full=True``).
    """
    A = A.copy()
    rows, columns = A.shape
    Q = np.eye(columns) if full else np.zeros((rows, columns))
    for i in range(rows):
        width = columns - i
        scale = 0.0
        for j in range(width):
            value = abs(A[i, j])
            if value > scale:
                scale = value
        if scale != 0.0:
            for j in range(width):
                A[i, j] /= scale
        length = 0.0
        for j in range(width):
            length = _madd(A[i, j], A[i, j], length, contract)
        length = np.sqrt(length)
        if A[i, width - 1] < 0.0:
            length = -length
        A[i, width - 1] += length
        multiplier = 1.0 / (length * A[i, width - 1]) if length != 0.0 else 0.0
        length *= scale
        for j in range(i + 1, rows):
            dot = 0.0
            for k in range(width):
                dot = _madd(A[i, k], A[j, k], dot, contract)
            dot *= multiplier
            for k in range(width):
                A[j, k] = _msub(A[j, k], dot, A[i, k], contract)
        if full:
            for j in range(columns):
                dot = 0.0
                for k in range(width):
                    dot = _madd(A[i, k], Q[j, k], dot, contract)
                dot *= multiplier
                for k in range(width):
                    Q[j, k] = _msub(Q[j, k], dot, A[i, k], contract)
        else:
            root = np.sqrt(multiplier)
            for j in range(width):
                Q[i, j] = A[i, j] * root
        A[i, width - 1] = -length
        for j in range(width - 1):
            A[i, j] = 0.0
    return Q


@numba.njit(cache=True)
def householder_right(
    C: npt.NDArray[np.float64],
    reflectors: npt.NDArray[np.float64],
    contract: bool,
) -> None:  # pragma: no cover
    """In-place matrix.c::HQmult(C, U, 0, 0): ``C <- C Q``."""
    rows, columns = C.shape
    product = np.empty(rows)
    for k in range(reflectors.shape[0]):
        for i in range(rows):
            total = 0.0
            for j in range(columns):
                total = _madd(C[i, j], reflectors[k, j], total, contract)
            product[i] = total
        for i in range(rows):
            for j in range(columns):
                C[i, j] = _msub(C[i, j], product[i], reflectors[k, j], contract)


@numba.njit(cache=True)
def householder_left_transpose(
    C: npt.NDArray[np.float64],
    reflectors: npt.NDArray[np.float64],
    contract: bool,
) -> None:  # pragma: no cover
    """In-place matrix.c::HQmult(C, U, 1, 1): ``C <- Q' C``."""
    rows, columns = C.shape
    product = np.empty(columns)
    for k in range(reflectors.shape[0]):
        for i in range(columns):
            total = 0.0
            for j in range(rows):
                total = _madd(C[j, i], reflectors[k, j], total, contract)
            product[i] = total
        for i in range(rows):
            for j in range(columns):
                C[i, j] = _msub(C[i, j], product[j], reflectors[k, i], contract)


@numba.njit(cache=True)
def lanczos_start(
    n: int, contract: bool
) -> npt.NDArray[np.float64]:  # pragma: no cover
    """Rlanczos's deterministic normalized start vector."""
    start = np.empty(n)
    state = 1
    squared = 0.0
    for i in range(n):
        state = (state * 106 + 1283) % 6075
        start[i] = state / 6075.0 - 0.5
        squared = _madd(start[i], start[i], squared, contract)
    norm = np.sqrt(squared)
    for i in range(n):
        start[i] /= norm
    return start


@numba.njit(cache=True)
def lanczos_three_term(
    z: npt.NDArray[np.float64],
    alpha: float,
    current: npt.NDArray[np.float64],
    beta: float,
    previous: npt.NDArray[np.float64],
    first: bool,
    contract: bool,
) -> None:  # pragma: no cover
    """Rlanczos ``z -= a*q[j]`` or ``z -= a*q[j] + b*q[j-1]``, in place."""
    for i in range(z.shape[0]):
        if first:
            z[i] = _msub(z[i], alpha, current[i], contract)
        else:
            z[i] = z[i] - _madd(alpha, current[i], beta * previous[i], contract)


@numba.njit(cache=True)
def ritz_accumulate(
    out: npt.NDArray[np.float64],
    basis: npt.NDArray[np.float64],
    coefficients: npt.NDArray[np.float64],
    contract: bool,
) -> None:  # pragma: no cover
    """Rlanczos ``U[,k] += q[l] * v[l,k]`` for ``l`` in order, in place."""
    for column in range(out.shape[1]):
        for step in range(coefficients.shape[0]):
            scale = coefficients[step, column]
            for i in range(out.shape[0]):
                out[i, column] = _madd(basis[i, step], scale, out[i, column], contract)


@functools.lru_cache(maxsize=1)
def _libm() -> ctypes.CDLL:
    """The C math library that R and Python's ``math`` module call.

    glibc's ``libm.so.6`` on Linux and libSystem's on macOS. NumPy's own
    ``exp``/``log``/``power`` loops may instead dispatch to SIMD
    implementations (AVX-512 on x86_64) whose last bits differ from libm.
    """
    name = ctypes.util.find_library("m")
    library = ctypes.CDLL(name) if name else ctypes.CDLL(None)
    for symbol, arguments in (("exp", 1), ("log", 1), ("log10", 1), ("pow", 2)):
        function = getattr(library, symbol)
        function.argtypes = [ctypes.c_double] * arguments
        function.restype = ctypes.c_double
    return library


@functools.lru_cache(maxsize=1)
def _compiled_libm_loops():
    """Compile scalar loops that call libm through opaque function pointers.

    An external call through a ctypes pointer is never vectorized by LLVM,
    so every element is libm's own result on every CPU. Not cached on disk:
    the pointers are process addresses.
    """
    library = _libm()
    c_exp, c_log, c_log10, c_pow = library.exp, library.log, library.log10, library.pow

    @numba.njit
    def exp_loop(values, out):  # pragma: no cover - Numba
        for i in range(values.size):
            out[i] = c_exp(values[i])

    @numba.njit
    def log_loop(values, out):  # pragma: no cover - Numba
        for i in range(values.size):
            out[i] = c_log(values[i])

    @numba.njit
    def log10_loop(values, out):  # pragma: no cover - Numba
        for i in range(values.size):
            out[i] = c_log10(values[i])

    @numba.njit
    def r_pow_loop(values, exponent, out):  # pragma: no cover - Numba
        for i in range(values.size):
            x = values[i]
            if exponent == 2.0:
                out[i] = x * x
            elif x == 1.0 or exponent == 0.0:
                out[i] = 1.0
            elif x == 0.0:
                out[i] = 0.0 if exponent > 0.0 else np.inf
            else:
                out[i] = c_pow(x, exponent)

    return exp_loop, log_loop, r_pow_loop, log10_loop


def _flat_float64(values: npt.ArrayLike) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return array.reshape(-1), array.shape


def libm_exp(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Elementwise C-library ``exp``, as R's ``exp`` evaluates it."""
    flat, shape = _flat_float64(values)
    out = np.empty_like(flat)
    _compiled_libm_loops()[0](flat, out)
    return out.reshape(shape)


def libm_log(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Elementwise C-library ``log`` (R's ``R_log`` for positive values)."""
    flat, shape = _flat_float64(values)
    if np.any(~(flat > 0.0)) or not np.all(np.isfinite(flat)):
        raise ValueError("libm_log is defined here for positive finite values.")
    out = np.empty_like(flat)
    _compiled_libm_loops()[1](flat, out)
    return out.reshape(shape)


def libm_log10(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Elementwise C-library ``log10`` (R's number formatting uses it)."""
    flat, shape = _flat_float64(values)
    if np.any(~(flat > 0.0)) or not np.all(np.isfinite(flat)):
        raise ValueError("libm_log10 is defined here for positive finite values.")
    out = np.empty_like(flat)
    _compiled_libm_loops()[3](flat, out)
    return out.reshape(shape)


def r_pow(values: npt.ArrayLike, exponent: float) -> npt.NDArray[np.float64]:
    """R's ``x ^ y`` (arithmetic.c ``R_pow``) for finite ``x`` and scalar ``y``.

    ``R_pow`` returns ``x * x`` when ``y == 2``, 1 when ``x == 1`` or
    ``y == 0``, 0 for ``x == 0`` and ``y > 0`` (``Inf`` for ``y < 0``), and
    otherwise the C library's ``pow(x, y)`` (``powl`` only on 64-bit Windows).
    """
    exponent = float(exponent)
    flat, shape = _flat_float64(values)
    if not np.isfinite(exponent) or not np.all(np.isfinite(flat)):
        raise ValueError("r_pow is ported here for finite operands only.")
    out = np.empty_like(flat)
    _compiled_libm_loops()[2](flat, exponent, out)
    return out.reshape(shape)
