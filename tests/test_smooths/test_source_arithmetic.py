"""Source-faithful contracted arithmetic kernels (mgcv C-loop ports)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from jaxgam.smooths import source_arithmetic as sa
from tests.helpers import _AssertCollector


def _exact_fma(a: float, b: float, c: float) -> float:
    """Correctly rounded ``a*b + c`` (``float(Fraction)`` rounds to nearest)."""
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _madd(a: float, b: float, c: float, contract: bool) -> float:
    return _exact_fma(a, b, c) if contract else c + a * b


def _msub(c: float, a: float, b: float, contract: bool) -> float:
    return _exact_fma(-a, b, c) if contract else c - a * b


def _values(rng, size, spread=6):
    return rng.standard_normal(size) * 10.0 ** rng.integers(-spread, spread, size)


def test_contraction_policy_follows_compiled_mgcv_architecture() -> None:
    """AArch64 builds of mgcv contract multiply-adds; x86-64 builds do not."""
    expected = sa.platform.machine().lower() in {"aarch64", "arm64"}
    assert expected == sa.SOURCE_FMA_CONTRACTION


def test_fused_multiply_add_is_single_rounding() -> None:
    rng = np.random.default_rng(20261024)
    a, b, c = (_values(rng, 2000) for _ in range(3))
    # Cancellation-heavy triples where one and two roundings differ.
    a = np.r_[a, 1.0 + 2.0**-30, 0.1, 3.0]
    b = np.r_[b, 1.0 - 2.0**-30, 0.1, 1.0 / 3.0]
    c = np.r_[c, -1.0, -0.01, -1.0]
    fused = np.array(
        [sa.fused_multiply_add(x, y, z) for x, y, z in zip(a, b, c, strict=True)]
    )
    exact = np.array([_exact_fma(x, y, z) for x, y, z in zip(a, b, c, strict=True)])
    np.testing.assert_array_equal(fused, exact)
    assert np.any(fused != a * b + c)


@pytest.mark.parametrize("contract", [False, True])
def test_sequential_kernels_match_source_loops(contract: bool) -> None:
    rng = np.random.default_rng(7)
    x, y = _values(rng, 37), _values(rng, 37)
    expected = 0.25
    for left, right in zip(x, y, strict=True):
        expected = _madd(left, right, expected, contract)
    assert sa.sequential_dot(x, y, 0.25, contract) == expected

    rows = _values(rng, 24).reshape(8, 3)
    totals = np.array([1.0, 2.0, 3.0])
    expected_totals = totals.copy()
    for row in rows:
        for i, value in enumerate(row):
            expected_totals[i] = _madd(value, value, expected_totals[i], contract)
    sa.column_squares(rows, totals, contract)
    np.testing.assert_array_equal(totals, expected_totals)


@pytest.mark.parametrize("contract", [False, True])
def test_distance_and_matmult_kernels_match_source_loops(contract: bool) -> None:
    rng = np.random.default_rng(11)
    knots = rng.standard_normal((6, 3))
    rows = rng.standard_normal((4, 3))
    expected = np.zeros((4, 6))
    for i in range(4):
        for j in range(6):
            total = 0.0
            for column in range(3):
                delta = knots[j, column] - rows[i, column]
                total = _madd(delta, delta, total, contract)
            expected[i, j] = total
    np.testing.assert_array_equal(sa.squared_distances(rows, knots, contract), expected)
    E = sa.knot_squared_distances(knots, contract)
    for i in range(6):
        for j in range(i):
            total = 0.0
            for column in range(3):
                delta = knots[i, column] - knots[j, column]
                total = _madd(delta, delta, total, contract)
            assert E[i, j] == E[j, i] == total
    np.testing.assert_array_equal(np.diag(E), 0.0)

    A, B = rng.standard_normal((5, 2)), rng.standard_normal((5, 4))
    C = np.zeros((2, 4))
    for k in range(5):
        for i in range(2):
            for p in range(4):
                C[i, p] = _madd(A[k, i], B[k, p], C[i, p], contract)
    np.testing.assert_array_equal(sa.matmult_tn(A, B, contract), C)


def _reference_qt(A, full, contract):
    """Literal scalar transcription of matrix.c::QT."""
    A = A.copy()
    rows, columns = A.shape
    Q = np.eye(columns) if full else np.zeros((rows, columns))
    for i in range(rows):
        width = columns - i
        scale = max(abs(v) for v in A[i, :width])
        if scale:
            A[i, :width] /= scale
        length = 0.0
        for j in range(width):
            length = _madd(A[i, j], A[i, j], length, contract)
        length = float(np.sqrt(length))
        if A[i, width - 1] < 0.0:
            length = -length
        A[i, width - 1] += length
        g = 1.0 / (length * A[i, width - 1]) if length else 0.0
        length *= scale
        targets = range(i + 1, rows)
        for j in targets:
            x = 0.0
            for k in range(width):
                x = _madd(A[i, k], A[j, k], x, contract)
            x *= g
            for k in range(width):
                A[j, k] = _msub(A[j, k], x, A[i, k], contract)
        if full:
            for j in range(columns):
                x = 0.0
                for k in range(width):
                    x = _madd(A[i, k], Q[j, k], x, contract)
                x *= g
                for k in range(width):
                    Q[j, k] = _msub(Q[j, k], x, A[i, k], contract)
        else:
            Q[i, :width] = A[i, :width] * np.sqrt(g)
        A[i, width - 1] = -length
        A[i, : width - 1] = 0.0
    return Q


@pytest.mark.parametrize("contract", [False, True])
def test_householder_kernels_match_source_loops(contract: bool) -> None:
    rng = np.random.default_rng(13)
    TU = rng.standard_normal((2, 6))
    reflectors = sa.qt_factor(TU, False, contract)
    np.testing.assert_array_equal(reflectors, _reference_qt(TU, False, contract))
    np.testing.assert_array_equal(
        sa.qt_factor(TU, True, contract), _reference_qt(TU, True, contract)
    )
    # QT annihilates the leading columns of A: A Q = [0, T].
    Q = sa.qt_factor(TU, True, contract)
    np.testing.assert_allclose((TU @ Q)[:, :4], 0.0, atol=1e-14)

    C = rng.standard_normal((5, 6))
    expected = C.copy()
    for u in reflectors:
        product = np.zeros(5)
        for i in range(5):
            for j in range(6):
                product[i] = _madd(expected[i, j], u[j], product[i], contract)
        for i in range(5):
            for j in range(6):
                expected[i, j] = _msub(expected[i, j], product[i], u[j], contract)
    actual = C.copy()
    sa.householder_right(actual, reflectors, contract)
    np.testing.assert_array_equal(actual, expected)

    S = rng.standard_normal((6, 3))
    expected = S.copy()
    for u in reflectors:
        product = np.zeros(3)
        for i in range(3):
            for j in range(6):
                product[i] = _madd(expected[j, i], u[j], product[i], contract)
        for i in range(6):
            for j in range(3):
                expected[i, j] = _msub(expected[i, j], product[j], u[i], contract)
    actual = S.copy()
    sa.householder_left_transpose(actual, reflectors, contract)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("contract", [False, True])
def test_lanczos_kernels_match_source_loops(contract: bool) -> None:
    start = sa.lanczos_start(9, contract)
    state, values, squared = 1, [], 0.0
    for _ in range(9):
        state = (state * 106 + 1283) % 6075
        values.append(state / 6075.0 - 0.5)
        squared = _madd(values[-1], values[-1], squared, contract)
    norm = float(np.sqrt(squared))
    np.testing.assert_array_equal(start, np.array(values) / norm)

    rng = np.random.default_rng(17)
    z, q, p = rng.standard_normal(9), rng.standard_normal(9), rng.standard_normal(9)
    first = z.copy()
    sa.lanczos_three_term(first, 0.3, q, 0.7, p, True, contract)
    np.testing.assert_array_equal(
        first, [_msub(zi, 0.3, qi, contract) for zi, qi in zip(z, q, strict=True)]
    )
    later = z.copy()
    sa.lanczos_three_term(later, 0.3, q, 0.7, p, False, contract)
    np.testing.assert_array_equal(
        later,
        [
            zi - _madd(0.3, qi, 0.7 * pi, contract)
            for zi, qi, pi in zip(z, q, p, strict=True)
        ],
    )

    basis = rng.standard_normal((9, 4))
    coefficients = rng.standard_normal((4, 2))
    out = np.zeros((9, 2))
    sa.ritz_accumulate(out, basis, coefficients, contract)
    expected = np.zeros((9, 2))
    for column in range(2):
        for step in range(4):
            for i in range(9):
                expected[i, column] = _madd(
                    basis[i, step],
                    coefficients[step, column],
                    expected[i, column],
                    contract,
                )
    np.testing.assert_array_equal(out, expected)


def _transcendental_inputs() -> np.ndarray:
    """Adversarial arguments: 1-ulp neighbourhoods, tiny, huge and random."""
    rng = np.random.default_rng(20261116)
    anchors = np.array(
        [
            1e-300,
            5e-324,
            2.2250738585072014e-308,
            1e-10,
            0.5,
            np.log(2.0),
            1.0,
            2.0,
            3.0,
            10.0,
            100.0,
            700.0,
            709.78,
        ]
    )
    neighbours = np.concatenate(
        [anchors, np.nextafter(anchors, 0), np.nextafter(anchors, np.inf)]
    )
    return np.concatenate(
        [
            neighbours,
            rng.uniform(0.0, 5.0, 20_000),
            10.0 ** rng.uniform(-30, 30, 5_000),
            np.arange(1, 2000) / 997.0,
        ]
    )


def test_libm_helpers_equal_the_c_library_bitwise() -> None:
    """The helpers are libm's own results on every CPU, never a SIMD variant.

    Python's ``math`` functions call the same C library. A future build that
    vectorized these loops (or routed them through a vector math library)
    would differ in the last bit on some CPUs and fail here.
    """
    values = _transcendental_inputs()
    collector = _AssertCollector()
    collector.check(
        "exp",
        lambda: np.testing.assert_array_equal(
            sa.libm_exp(-values),
            np.array([math.exp(-value) for value in values]),
        ),
    )
    collector.check(
        "exp of positive arguments",
        lambda: np.testing.assert_array_equal(
            sa.libm_exp(values[values < 700]),
            np.array([math.exp(value) for value in values[values < 700]]),
        ),
    )
    positive = values[values > 0]
    collector.check(
        "log",
        lambda: np.testing.assert_array_equal(
            sa.libm_log(positive),
            np.array([math.log(value) for value in positive]),
        ),
    )
    collector.check(
        "log10",
        lambda: np.testing.assert_array_equal(
            sa.libm_log10(positive),
            np.array([math.log10(value) for value in positive]),
        ),
    )
    for exponent in (3.0, 1.5, 0.5, 0.3, 7.0, -1.5):
        finite = values[(values > 1e-100) & (values < 1e40)]
        collector.check(
            f"pow {exponent}",
            lambda exponent=exponent, finite=finite: np.testing.assert_array_equal(
                sa.r_pow(finite, exponent),
                np.array([math.pow(value, exponent) for value in finite]),
            ),
        )
    collector.raise_if_any("libm helpers")


def test_r_pow_follows_r_special_cases_and_keeps_shape() -> None:
    x = np.array([[0.0, -0.0, 1.0], [2.5, -3.0, 1e-200]])
    np.testing.assert_array_equal(sa.r_pow(x, 2.0), x * x)
    cube = sa.r_pow(x, 3.0)
    assert cube.shape == x.shape
    # R_pow returns +0 for a zero base with y > 0 (C pow gives -0 for -0^3).
    assert np.signbit(cube[0, 1]) == np.False_
    assert cube[0, 2] == 1.0
    assert cube[1, 1] == math.pow(-3.0, 3.0)
    np.testing.assert_array_equal(sa.r_pow(x, 0.0), np.ones_like(x))
    assert sa.r_pow(np.array([0.0]), -1.0)[0] == np.inf
    with pytest.raises(ValueError, match="finite"):
        sa.r_pow(np.array([np.inf]), 2.0)
    with pytest.raises(ValueError, match="finite"):
        sa.r_pow(np.array([1.0]), np.nan)
    with pytest.raises(ValueError, match="positive"):
        sa.libm_log(np.array([0.0]))
    with pytest.raises(ValueError, match="positive"):
        sa.libm_log10(np.array([-1.0]))
    assert sa.libm_exp(np.zeros((2, 3))).shape == (2, 3)
