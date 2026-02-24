"""Davies (1980) exact method for weighted sums of chi-squared variables.

Numba JIT-compiled port of the C implementation in mgcv/src/davies.c.

Computes ``Pr(Q < c)`` where ``Q = sum_j lb[j] * X_j + sigma * X_0``,
with ``X_j ~ chi2(n_j, nc_j)`` and ``X_0 ~ N(0,1)``.

All internal helpers are eagerly compiled at module import time via
``numba.njit`` with explicit type signatures, achieving C-native
performance. See ``_slanczos_jit`` in ``pymgcv/smooths/tprs.py``
for the same pattern.

Phase 3 code: NumPy + Numba only, no JAX imports.

References
----------
Davies, R.B. (1980) "The Distribution of a Linear Combination of
chi^2 Random Variables", J.R. Statist. Soc. C 29, 323-333.

Davies, R.B. (1973) "Numerical inversion of a characteristic function",
Biometrika 60(2), 415-417.
"""

import math

import numba
import numpy as np

# ---------------------------------------------------------------------------
# JIT-compiled internal helpers (translated from C)
# ---------------------------------------------------------------------------


@numba.njit(
    numba.types.Tuple((numba.float64, numba.float64))(
        numba.float64,
        numba.float64,
        numba.int64[:],
        numba.float64[:],
        numba.float64[:],
    ),
    cache=True,
)
def _errbd_jit(u, sigsq, n, lb, nc):  # pragma: no cover
    """Tail probability bound. Returns (bound, cx)."""
    r = len(lb)
    cx = u * sigsq
    sum1 = u * cx
    u2 = 2.0 * u
    for j in range(r - 1, -1, -1):
        nj = n[j]
        lj = lb[j]
        ncj = nc[j]
        x = u2 * lj
        y = 1.0 - x
        cx += lj * (ncj / y + nj) / y
        xy = x / y
        # log1pmx(-x) = log1p(-x) - (-x) = log1p(-x) + x
        sum1 += ncj * xy * xy + nj * (x * xy + math.log1p(-x) + x)
    return math.exp(-0.5 * sum1), cx


@numba.njit(
    numba.types.Tuple((numba.float64, numba.float64))(
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.int64[:],
        numba.float64[:],
        numba.float64[:],
    ),
    cache=True,
)
def _ctff_jit(accx, upn, mean, lmin, lmax, sigsq, n, lb, nc):  # pragma: no cover
    """Find cutoff so that Pr(qf > ctff) < accx. Returns (c2, upn)."""
    u2 = upn
    u1 = 0.0
    c1 = mean
    rb = 2.0 * lmax if u2 > 0.0 else 2.0 * lmin

    while True:
        bound, c2 = _errbd_jit(u2 / (1.0 + u2 * rb), sigsq, n, lb, nc)
        if bound <= accx:
            break
        u1 = u2
        c1 = c2
        u2 = u2 * 2.0

    while True:
        if c2 == mean or (c1 - mean) / (c2 - mean) >= 0.9:
            break
        u = (u1 + u2) * 0.5
        bound, cst = _errbd_jit(u / (1.0 + u * rb), sigsq, n, lb, nc)
        if bound > accx:
            u1 = u
            c1 = cst
        else:
            u2 = u
            c2 = cst

    return c2, u2


@numba.njit(
    numba.float64(
        numba.float64,
        numba.float64,
        numba.float64,
        numba.int64[:],
        numba.float64[:],
        numba.float64[:],
    ),
    cache=True,
)
def _truncation_jit(u, tausq, sigsq, n, lb, nc):  # pragma: no cover
    """Bound the integration error due to truncation."""
    pi = math.pi
    r = len(lb)
    sum1 = 0.0
    prod2 = 0.0
    prod3 = 0.0
    s = 0
    sum2 = (sigsq + tausq) * u * u
    prod1 = 2.0 * sum2
    u2 = 2.0 * u

    for j in range(r):
        lj = lb[j]
        ncj = nc[j]
        nj = n[j]
        x = u2 * lj
        x2 = x * x
        sum1 += ncj * x2 / (1.0 + x2)
        if x2 > 1.0:
            prod2 += nj * math.log(x2)
            prod3 += nj * math.log1p(x2)
            s += nj
        else:
            prod1 += nj * math.log1p(x2)

    sum1 *= 0.5
    prod2 += prod1
    prod3 += prod1
    x = math.exp(-sum1 - 0.25 * prod2) / pi
    y = math.exp(-sum1 - 0.25 * prod3) / pi

    err1 = 1.0 if s == 0 else 2.0 * x / s
    err2 = 2.5 * y if prod3 > 1.0 else 1.0
    if err2 < err1:
        err1 = err2

    x = 0.5 * sum2
    err2 = 1.0 if x <= y else y / x

    if err1 < err2:
        return err1
    return err2


@numba.njit(
    numba.float64(
        numba.float64,
        numba.float64,
        numba.float64,
        numba.int64[:],
        numba.float64[:],
        numba.float64[:],
    ),
    cache=True,
)
def _findu_jit(utx, accx, sigsq, n, lb, nc):  # pragma: no cover
    """Find u s.t. truncation(u) < accx and truncation(u/1.2) > accx."""
    ut = utx
    u = ut * 0.25
    if _truncation_jit(u, 0.0, sigsq, n, lb, nc) > accx:
        while _truncation_jit(ut, 0.0, sigsq, n, lb, nc) > accx:
            ut *= 4.0
    else:
        ut = u
        u = u / 4.0
        while _truncation_jit(u, 0.0, sigsq, n, lb, nc) <= accx:
            ut = u
            u /= 4.0

    for a in (2.0, 1.4, 1.2, 1.1):
        u = ut / a
        if _truncation_jit(u, 0.0, sigsq, n, lb, nc) <= accx:
            ut = u

    return ut


@numba.njit(
    numba.types.Tuple((numba.float64, numba.float64))(
        numba.int64,
        numba.float64,
        numba.float64,
        numba.boolean,
        numba.float64,
        numba.float64,
        numba.int64[:],
        numba.float64[:],
        numba.float64[:],
    ),
    cache=True,
)
def _integrate_jit(nterm, interv, tausq, main, c, sigsq, n, lb, nc):  # pragma: no cover
    """Fourier inversion integral. Returns (intl, ersm)."""
    pi = math.pi
    r = len(lb)
    inpi = interv / pi
    intl = 0.0
    ersm = 0.0

    for k in range(nterm, -1, -1):
        u = (k + 0.5) * interv
        sum1 = -2.0 * u * c
        sum2 = abs(sum1)
        sum3 = -0.5 * sigsq * u * u

        for j in range(r - 1, -1, -1):
            nj = n[j]
            x = 2.0 * lb[j] * u
            y = x * x
            sum3 -= 0.25 * nj * math.log1p(y)
            y_nc = nc[j] * x / (1.0 + y)
            z = nj * math.atan(x) + y_nc
            sum1 += z
            sum2 += abs(z)
            sum3 -= 0.5 * x * y_nc

        x = inpi * math.exp(sum3) / u
        if not main:
            x *= 1.0 - math.exp(-0.5 * tausq * u * u)
        intl += math.sin(0.5 * sum1) * x
        ersm += 0.5 * sum2 * x

    return intl, ersm


@numba.njit(
    numba.types.Tuple((numba.float64, numba.boolean))(
        numba.float64,
        numba.int64[:],
        numba.int64[:],
        numba.float64[:],
        numba.float64[:],
    ),
    cache=True,
)
def _cfe_jit(x, th, n, lb, nc):  # pragma: no cover
    """Convergence factor efficiency coefficient. Returns (coef, fail)."""
    pi = math.pi
    ln28 = math.log(2.0) / 8.0
    r = len(lb)
    axl = abs(x)
    sxl = -1 if x < 0.0 else 1

    sum1 = 0.0
    for j in range(r - 1, -1, -1):
        t = th[j]
        if lb[t] * sxl > 0.0:
            lj = abs(lb[t])
            axl1 = axl - lj * (n[t] + nc[t])
            axl2 = lj / ln28
            if axl1 > axl2:
                axl = axl1
            else:
                if axl > axl2:
                    axl = axl2
                sum1 = (axl - axl1) / lj
                for k in range(j - 1, -1, -1):
                    sum1 += n[th[k]] + nc[th[k]]
                break

    if sum1 > 100.0:
        return 1.0, True
    return 2.0 ** (sum1 * 0.25) / (pi * axl * axl), False


# ---------------------------------------------------------------------------
# Main Davies algorithm (JIT-compiled)
# ---------------------------------------------------------------------------


@numba.njit(
    numba.types.Tuple((numba.float64, numba.int64, numba.float64[:]))(
        numba.float64[:],
        numba.float64[:],
        numba.int64[:],
        numba.float64,
        numba.float64,
        numba.int64,
        numba.float64,
    ),
    cache=True,
)
def _davies_jit(lb, nc, n, sigma, c, lim, acc):  # pragma: no cover
    """Core Davies algorithm. Returns (prob, ifault, trace)."""
    pi = math.pi
    r = len(lb)
    trace = np.zeros(7)

    # Sort indices by descending |lb|
    th = np.argsort(-np.abs(lb)).astype(np.int64)

    # Compute mean, sd, lmin, lmax
    sigsq = sigma * sigma
    sd = sigsq
    mean = 0.0
    lmin = 0.0
    lmax = 0.0

    for j in range(r):
        nj = float(n[j])
        lj = lb[j]
        ncj = nc[j]
        sd += lj * lj * (2.0 * nj + 4.0 * ncj)
        mean += lj * (nj + ncj)
        if lj > lmax:
            lmax = lj
        elif lj < lmin:
            lmin = lj

    if sd == 0.0:
        prob = 1.0 if c > 0.0 else 0.0
        return prob, numba.int64(0), trace

    if lmin == 0.0 and lmax == 0.0 and sigma == 0.0:
        return 0.0, numba.int64(3), trace

    sd = math.sqrt(sd)
    almx = max(lmax, -lmin)

    # Starting values
    utx = 16.0 / sd
    up = 4.5 / sd
    un = -up

    intl = 0.0
    ersm = 0.0
    acc1 = acc
    lim_remaining = lim

    # Truncation point without convergence factor
    utx = _findu_jit(utx, 0.5 * acc1, sigsq, n, lb, nc)

    # Does convergence factor help?
    if c != 0.0 and almx > 0.07 * sd:
        coef, fail = _cfe_jit(c, th, n, lb, nc)
        tausq = 0.25 * acc1 / coef
        if not fail and _truncation_jit(utx, tausq, sigsq, n, lb, nc) < 0.2 * acc1:
            sigsq += tausq
            utx = _findu_jit(utx, 0.25 * acc1, sigsq, n, lb, nc)
            trace[5] = math.sqrt(tausq)

    trace[4] = utx
    acc1 *= 0.5

    # Find range of distribution
    intv = 0.0  # will be set inside the loop
    nt = 0
    while True:
        c2_up, up = _ctff_jit(acc1, up, mean, lmin, lmax, sigsq, n, lb, nc)
        d1 = c2_up - c
        if d1 < 0.0:
            return 1.0, numba.int64(0), trace

        c2_dn, un = _ctff_jit(acc1, un, mean, lmin, lmax, sigsq, n, lb, nc)
        d2 = c - c2_dn
        if d2 < 0.0:
            return 0.0, numba.int64(0), trace

        # Integration interval
        intv = 2.0 * pi / d1 if d1 > d2 else 2.0 * pi / d2

        # Number of terms for main and auxiliary integrations
        # C-style rounding: floor(x); if frac > 0.5 then increment
        x_nt = utx / intv
        nt = math.floor(x_nt)
        if x_nt - nt > 0.5:
            nt += 1

        x_ntm = 3.0 / math.sqrt(acc1)
        ntm = math.floor(x_ntm)
        if x_ntm - ntm > 0.5:
            ntm += 1

        if nt <= int(ntm * 1.5):
            break

        # Parameters for auxiliary integration
        intv1 = utx / ntm
        x = 2.0 * pi / intv1
        if x <= abs(c):
            break

        # Convergence factor
        coef1, fail1 = _cfe_jit(c - x, th, n, lb, nc)
        coef2, fail2 = _cfe_jit(c + x, th, n, lb, nc)
        if fail1 or fail2:
            break
        tausq = 0.33 * acc1 / (1.1 * (coef1 + coef2))
        acc1 *= 0.67

        if ntm > lim_remaining:
            return -1.0, numba.int64(1), trace

        # Auxiliary integration
        intl_add, ersm_add = _integrate_jit(
            numba.int64(ntm), intv1, tausq, False, c, sigsq, n, lb, nc
        )
        intl += intl_add
        ersm += ersm_add

        lim_remaining -= ntm
        sigsq += tausq
        trace[2] += 1.0
        trace[1] += ntm + 1

        # New truncation point
        utx = _findu_jit(utx, 0.25 * acc1, sigsq, n, lb, nc)
        acc1 *= 0.75

    # Main integration
    trace[3] = intv
    if nt > lim_remaining:
        return -1.0, numba.int64(1), trace

    intl_add, ersm_add = _integrate_jit(
        numba.int64(nt), intv, 0.0, True, c, sigsq, n, lb, nc
    )
    intl += intl_add
    ersm += ersm_add
    trace[2] += 1
    trace[1] += nt + 1

    prob = 0.5 - intl
    trace[0] = ersm

    # Test whether round-off error could be significant
    ifault = numba.int64(0)
    x = ersm + acc / 10.0
    for i in range(4):
        j = 2**i
        if j * x == j * ersm:
            ifault = numba.int64(2)

    return prob, ifault, trace


# ---------------------------------------------------------------------------
# Public API (thin Python wrappers around JIT core)
# ---------------------------------------------------------------------------


class DaviesResult:
    """Result from the Davies algorithm.

    Attributes
    ----------
    prob : float
        Probability ``Pr(Q < c)``.
    ifault : int
        Fault indicator: 0 = OK, 1 = accuracy not obtained,
        2 = round-off possibly significant, 3 = invalid parameters,
        4 = unable to locate integration parameters.
    trace : np.ndarray
        7-element diagnostic vector.
    """

    __slots__ = ("ifault", "prob", "trace")

    def __init__(self, prob: float, ifault: int, trace: np.ndarray) -> None:
        self.prob = prob
        self.ifault = ifault
        self.trace = trace

    def __repr__(self) -> str:
        return f"DaviesResult(prob={self.prob:.8e}, ifault={self.ifault})"


def _davies(
    lb: np.ndarray,
    nc: np.ndarray,
    n: np.ndarray,
    sigma: float,
    c: float,
    lim: int = 100_000,
    acc: float = 2e-5,
) -> DaviesResult:
    """Evaluate ``Pr(Q < c)`` via Davies' method.

    ``Q = sum_j lb[j] * X_j + sigma * X_0`` where
    ``X_j ~ chi^2(n[j], nc[j])`` and ``X_0 ~ N(0, 1)``.

    The core algorithm is JIT-compiled via Numba for C-native performance.

    Parameters
    ----------
    lb : np.ndarray
        Weights for the chi-squared components.
    nc : np.ndarray
        Non-centrality parameters (delta_j^2).
    n : np.ndarray
        Degrees of freedom (positive integers).
    sigma : float
        Standard deviation of the normal component.
    c : float
        Cutoff value.
    lim : int
        Upper bound on integration terms.
    acc : float
        Desired accuracy.

    Returns
    -------
    DaviesResult
        Contains ``prob`` (= Pr(Q < c)), ``ifault``, and ``trace``.
    """
    lb = np.asarray(lb, dtype=np.float64)
    nc = np.asarray(nc, dtype=np.float64)
    n = np.asarray(n, dtype=np.int64)

    # Validate parameters (before entering JIT)
    if np.any(n < 0) or np.any(nc < 0):
        return DaviesResult(prob=0.0, ifault=3, trace=np.zeros(7))

    prob, ifault, trace = _davies_jit(
        lb, nc, n, float(sigma), float(c), int(lim), float(acc)
    )
    return DaviesResult(prob=float(prob), ifault=int(ifault), trace=trace)
