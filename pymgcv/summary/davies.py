"""Davies (1980) exact method for weighted sums of chi-squared variables.

Pure NumPy port of the C implementation in mgcv/src/davies.c.

Computes ``Pr(Q < c)`` where ``Q = sum_j lb[j] * X_j + sigma * X_0``,
with ``X_j ~ chi2(n_j, nc_j)`` and ``X_0 ~ N(0,1)``.

Phase 3 code: NumPy + SciPy only, no JAX imports.

References
----------
Davies, R.B. (1980) "The Distribution of a Linear Combination of
chi^2 Random Variables", J.R. Statist. Soc. C 29, 323-333.

Davies, R.B. (1973) "Numerical inversion of a characteristic function",
Biometrika 60(2), 415-417.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers (translated from C)
# ---------------------------------------------------------------------------

_LN28 = np.log(2.0) / 8.0
_PI = np.pi


def _log1pmx(x: float) -> float:
    """Compute ``log(1 + x) - x`` accurately for small x."""
    return float(np.log1p(x)) - x


def _errbd(
    u: float,
    sigsq: float,
    n: np.ndarray,
    lb: np.ndarray,
    nc: np.ndarray,
) -> tuple[float, float]:
    """Tail probability bound.

    Parameters
    ----------
    u : float
        Evaluation point.
    sigsq : float
        Variance of the normal component (sigma^2 or augmented).
    n, lb, nc : np.ndarray
        Degrees of freedom, weights, and non-centrality params.

    Returns
    -------
    bound : float
        Tail probability bound.
    cx : float
        Cutoff value.
    """
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
        sum1 += ncj * xy * xy + nj * (x * xy + _log1pmx(-x))
    return float(np.exp(-0.5 * sum1)), cx


def _ctff(
    accx: float,
    upn: float,
    mean: float,
    lmin: float,
    lmax: float,
    sigsq: float,
    n: np.ndarray,
    lb: np.ndarray,
    nc: np.ndarray,
) -> tuple[float, float]:
    """Find cutoff so that ``Pr(qf > ctff) < accx`` if ``upn > 0``.

    Parameters
    ----------
    accx : float
        Accuracy target.
    upn : float
        Starting value for u (modified in place semantics → returned).
    mean, lmin, lmax : float
        Mean, min and max of weights.
    sigsq : float
        Variance parameter.
    n, lb, nc : np.ndarray
        Degrees of freedom, weights, non-centrality params.

    Returns
    -------
    c2 : float
        Cutoff value.
    upn : float
        Updated u parameter.
    """
    u2 = upn
    u1 = 0.0
    c1 = mean
    rb = 2.0 * lmax if u2 > 0 else 2.0 * lmin

    while True:
        bound, c2 = _errbd(u2 / (1.0 + u2 * rb), sigsq, n, lb, nc)
        if bound <= accx:
            break
        u1 = u2
        c1 = c2
        u2 = u2 * 2.0

    while True:
        if c2 == mean or (c1 - mean) / (c2 - mean) >= 0.9:
            break
        u = (u1 + u2) * 0.5
        bound, cst = _errbd(u / (1.0 + u * rb), sigsq, n, lb, nc)
        if bound > accx:
            u1 = u
            c1 = cst
        else:
            u2 = u
            c2 = cst

    return c2, u2


def _truncation(
    u: float,
    tausq: float,
    sigsq: float,
    n: np.ndarray,
    lb: np.ndarray,
    nc: np.ndarray,
) -> float:
    """Bound the integration error due to truncation.

    Parameters
    ----------
    u : float
        Truncation point candidate.
    tausq : float
        Convergence factor variance.
    sigsq : float
        Normal component variance.
    n, lb, nc : np.ndarray
        Degrees of freedom, weights, non-centrality params.

    Returns
    -------
    float
        Truncation error bound.
    """
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
            prod2 += nj * np.log(x2)
            prod3 += nj * np.log1p(x2)
            s += nj
        else:
            prod1 += nj * np.log1p(x2)

    sum1 *= 0.5
    prod2 += prod1
    prod3 += prod1
    x = np.exp(-sum1 - 0.25 * prod2) / _PI
    y = np.exp(-sum1 - 0.25 * prod3) / _PI

    err1 = 1.0 if s == 0 else 2.0 * x / s
    err2 = 2.5 * y if prod3 > 1.0 else 1.0
    err1 = min(err1, err2)

    x = 0.5 * sum2
    err2 = 1.0 if x <= y else y / x

    return min(err1, err2)


def _findu(
    utx: float,
    accx: float,
    sigsq: float,
    n: np.ndarray,
    lb: np.ndarray,
    nc: np.ndarray,
) -> float:
    """Find u such that truncation(u) < accx and truncation(u/1.2) > accx.

    Parameters
    ----------
    utx : float
        Starting value.
    accx : float
        Accuracy target.
    sigsq : float
        Variance parameter.
    n, lb, nc : np.ndarray
        Degrees of freedom, weights, non-centrality params.

    Returns
    -------
    float
        Truncation point.
    """
    ut = utx
    u = ut * 0.25
    if _truncation(u, 0.0, sigsq, n, lb, nc) > accx:
        while _truncation(ut, 0.0, sigsq, n, lb, nc) > accx:
            ut *= 4.0
    else:
        ut = u
        u = u / 4.0
        while _truncation(u, 0.0, sigsq, n, lb, nc) <= accx:
            ut = u
            u /= 4.0

    for a in [2.0, 1.4, 1.2, 1.1]:
        u = ut / a
        if _truncation(u, 0.0, sigsq, n, lb, nc) <= accx:
            ut = u

    return ut


def _integrate(
    nterm: int,
    interv: float,
    tausq: float,
    main: bool,
    c: float,
    sigsq: float,
    n: np.ndarray,
    lb: np.ndarray,
    nc: np.ndarray,
) -> tuple[float, float]:
    """Fourier inversion integral.

    Parameters
    ----------
    nterm : int
        Number of terms.
    interv : float
        Step size.
    tausq : float
        Convergence factor variance.
    main : bool
        If False, multiply integrand by ``1 - exp(-0.5 * tausq * u^2)``.
    c : float
        Cutoff point.
    sigsq : float
        Variance parameter.
    n, lb, nc : np.ndarray
        Degrees of freedom, weights, non-centrality params.

    Returns
    -------
    intl : float
        Integral value.
    ersm : float
        Error sum.
    """
    r = len(lb)
    inpi = interv / _PI
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
            sum3 -= 0.25 * nj * np.log1p(y)
            y_nc = nc[j] * x / (1.0 + y)
            z = nj * np.arctan(x) + y_nc
            sum1 += z
            sum2 += abs(z)
            sum3 -= 0.5 * x * y_nc

        x = inpi * np.exp(sum3) / u
        if not main:
            x *= 1.0 - np.exp(-0.5 * tausq * u * u)
        intl += np.sin(0.5 * sum1) * x
        ersm += 0.5 * sum2 * x

    return intl, ersm


def _cfe(
    x: float,
    th: np.ndarray,
    n: np.ndarray,
    lb: np.ndarray,
    nc: np.ndarray,
) -> tuple[float, bool]:
    """Coefficient of tausq in error when convergence factor is used.

    Parameters
    ----------
    x : float
        Evaluation point.
    th : np.ndarray
        Sort order (indices into lb, sorted by descending |lb|).
    n, lb, nc : np.ndarray
        Degrees of freedom, weights, non-centrality params.

    Returns
    -------
    coef : float
        Convergence factor coefficient.
    fail : bool
        True if computation failed.
    """
    r = len(lb)
    axl = abs(x)
    sxl = -1 if x < 0.0 else 1

    sum1 = 0.0
    for j in range(r - 1, -1, -1):
        t = th[j]
        if lb[t] * sxl > 0.0:
            lj = abs(lb[t])
            axl1 = axl - lj * (n[t] + nc[t])
            axl2 = lj / _LN28
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
    return 2.0 ** (sum1 * 0.25) / (_PI * axl * axl), False


# ---------------------------------------------------------------------------
# Main Davies entry point
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

    __slots__ = ("prob", "ifault", "trace")

    def __init__(self, prob: float, ifault: int, trace: np.ndarray) -> None:
        self.prob = prob
        self.ifault = ifault
        self.trace = trace

    def __repr__(self) -> str:
        return f"DaviesResult(prob={self.prob:.8e}, ifault={self.ifault})"


def davies(
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
    r = len(lb)
    lb = np.asarray(lb, dtype=float)
    nc = np.asarray(nc, dtype=float)
    n = np.asarray(n, dtype=int)
    trace = np.zeros(7)

    # Sort indices by descending |lb|
    th = np.argsort(-np.abs(lb))

    # Validate parameters
    if np.any(n < 0) or np.any(nc < 0):
        return DaviesResult(prob=0.0, ifault=3, trace=trace)

    # Compute mean, sd, lmin, lmax
    sigsq = sigma * sigma
    sd = sigsq
    mean = 0.0
    lmin = 0.0
    lmax = 0.0

    for j in range(r):
        nj = float(n[j])
        lj = float(lb[j])
        ncj = float(nc[j])
        sd += lj * lj * (2.0 * nj + 4.0 * ncj)
        mean += lj * (nj + ncj)
        if lj > lmax:
            lmax = lj
        elif lj < lmin:
            lmin = lj

    if sd == 0.0:
        prob = 1.0 if c > 0.0 else 0.0
        return DaviesResult(prob=prob, ifault=0, trace=trace)

    if lmin == 0.0 and lmax == 0.0 and sigma == 0.0:
        return DaviesResult(prob=0.0, ifault=3, trace=trace)

    sd = np.sqrt(sd)
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
    utx = _findu(utx, 0.5 * acc1, sigsq, n, lb, nc)

    # Does convergence factor help?
    if c != 0.0 and almx > 0.07 * sd:
        coef, fail = _cfe(c, th, n, lb, nc)
        tausq = 0.25 * acc1 / coef
        if not fail:
            if _truncation(utx, tausq, sigsq, n, lb, nc) < 0.2 * acc1:
                sigsq += tausq
                utx = _findu(utx, 0.25 * acc1, sigsq, n, lb, nc)
                trace[5] = np.sqrt(tausq)

    trace[4] = utx
    acc1 *= 0.5

    # Find range of distribution
    while True:
        c2_up, up = _ctff(acc1, up, mean, lmin, lmax, sigsq, n, lb, nc)
        d1 = c2_up - c
        if d1 < 0.0:
            return DaviesResult(prob=1.0, ifault=0, trace=trace)

        c2_dn, un = _ctff(acc1, un, mean, lmin, lmax, sigsq, n, lb, nc)
        d2 = c - c2_dn
        if d2 < 0.0:
            return DaviesResult(prob=0.0, ifault=0, trace=trace)

        # Integration interval
        intv = 2.0 * _PI / d1 if d1 > d2 else 2.0 * _PI / d2

        # Number of terms for main and auxiliary integrations
        nt = round(utx / intv)
        ntm = round(3.0 / np.sqrt(acc1))

        if nt <= ntm * 1.5:
            break

        # Parameters for auxiliary integration
        intv1 = utx / ntm
        x = 2.0 * _PI / intv1
        if x <= abs(c):
            break

        # Convergence factor
        coef1, fail1 = _cfe(c - x, th, n, lb, nc)
        coef2, fail2 = _cfe(c + x, th, n, lb, nc)
        if fail1 or fail2:
            break
        tausq = 0.33 * acc1 / (1.1 * (coef1 + coef2))
        acc1 *= 0.67

        if ntm > lim_remaining:
            return DaviesResult(prob=-1.0, ifault=1, trace=trace)

        # Auxiliary integration
        intl_add, ersm_add = _integrate(ntm, intv1, tausq, False, c, sigsq, n, lb, nc)
        intl += intl_add
        ersm += ersm_add

        lim_remaining -= ntm
        sigsq += tausq
        trace[2] += 1.0
        trace[1] += ntm + 1

        # New truncation point
        utx = _findu(utx, 0.25 * acc1, sigsq, n, lb, nc)
        acc1 *= 0.75

    # Main integration
    trace[3] = intv
    if nt > lim_remaining:
        return DaviesResult(prob=-1.0, ifault=1, trace=trace)

    intl_add, ersm_add = _integrate(nt, intv, 0.0, True, c, sigsq, n, lb, nc)
    intl += intl_add
    ersm += ersm_add
    trace[2] += 1
    trace[1] += nt + 1

    prob = 0.5 - intl
    trace[0] = ersm

    # Test for round-off error
    ifault = 0
    x = ersm + acc / 10.0
    for i in range(4):
        j = 2**i
        if j * x == j * ersm:
            ifault = 2

    return DaviesResult(prob=prob, ifault=ifault, trace=trace)


# ---------------------------------------------------------------------------
# Public wrapper matching R's psum.chisq()
# ---------------------------------------------------------------------------


def psum_chisq_davies(
    q: float,
    lb: np.ndarray,
    df: np.ndarray | None = None,
    nc: np.ndarray | None = None,
    sigz: float = 0.0,
    tol: float = 2e-5,
    nlim: int = 100_000,
) -> float:
    """Upper tail probability for weighted sum of chi-squared variables.

    Computes ``Pr(sum_j lb[j] * X_j + sigz * Z > q)`` where
    ``X_j ~ chi^2(df[j], nc[j])`` and ``Z ~ N(0, 1)``.

    Uses Davies' exact method with Liu et al. (2009) fallback.

    Port of R's ``psum.chisq()`` (mgcv.r lines 3466-3498).

    Parameters
    ----------
    q : float
        Quantile.
    lb : np.ndarray
        Weights (can be either sign).
    df : np.ndarray or None
        Degrees of freedom (positive integers). Defaults to all ones.
    nc : np.ndarray or None
        Non-centrality parameters. Defaults to all zeros.
    sigz : float
        Standard deviation of the normal component.
    tol : float
        Accuracy tolerance.
    nlim : int
        Maximum number of integration terms.

    Returns
    -------
    float
        Upper tail probability.
    """
    from pymgcv.summary.summary import _liu2

    lb = np.asarray(lb, dtype=float)
    r = len(lb)

    if df is None:
        df = np.ones(r, dtype=int)
    else:
        df = np.asarray(df, dtype=int)
        df = np.round(df).astype(int)

    if nc is None:
        nc = np.zeros(r)
    else:
        nc = np.asarray(nc, dtype=float)

    if sigz < 0:
        sigz = 0.0

    result = davies(lb, nc, df, sigz, q, lim=nlim, acc=tol)

    if result.ifault == 0 or result.ifault == 2:
        return float(1.0 - result.prob)
    else:
        # Fallback to Liu approximation
        if np.all(nc == 0):
            return float(_liu2(q, lb, h=df))
        return float(np.nan)
