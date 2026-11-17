"""Exact model of pinned R's numeric ``paste0`` label (test-only).

R 4.5.2 formats a double for ``as.character``/``paste0`` with
``R_print.digits = DBL_DIG`` (coerce.c), ``formatReal`` and ``scientific``
(format.c) and ``EncodeRealDrop0`` (printutils.c). ``scientific`` chooses the
significant digits in the platform's ``long double`` when it is wider than
``double`` and in ``double`` otherwise, so the label of a rounding tie can
depend on the platform. This module replays that sequence in exact rational
arithmetic, rounding every floating operation to nearest-even at a chosen
significand width:

- 53 bits: ``double`` (macOS arm64, Windows): ``KP_MAX`` 22;
- 64 bits: x87 extended (Linux x86_64): ``KP_MAX`` 27;
- 113 bits: binary128 (Linux aarch64): ``KP_MAX`` 27.

The scaling table is ``static const long double tbl[] = {1e00, ..., 1e27}``.
Its unsuffixed literals are C ``double`` constants, so entries above 1e22
hold the double nearest the power, widened, not the power itself.

Calls into the C library are not arithmetic R controls, so the model takes
them from the C library too: ``log10`` (``kp``) and, in the double branch,
``Rexp10(x)``, which ``Defn.h`` defines as ``pow(10.0, x)``. The long-double
branch divides by ``powl(10, kp)`` when ``|kp| > 27``; that value is supplied
by ``long_double_power`` (default: the power rounded to nearest at the
model's width).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
from collections.abc import Callable
from fractions import Fraction

DIGITS = 15  # R_print.digits = DBL_DIG while coercing doubles to strings
# machine.c: R_dec_min_exponent = (int) floor(log10(DBL_MIN)), i.e. -308.
R_DEC_MIN_EXPONENT = -308
DOUBLE_BITS = 53


def round_to_bits(value: Fraction, bits: int) -> Fraction:
    """Round to nearest, ties to even, with a ``bits``-bit significand.

    The exponent range is unbounded; every quantity rounded here is a
    normal number in the formats modelled.
    """
    if value == 0:
        return Fraction(0)
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    scale = Fraction(2) ** (bits - 1 - exponent)
    scaled = magnitude * scale
    floor = scaled.numerator // scaled.denominator
    remainder = scaled - floor
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and floor % 2):
        floor += 1
    return sign * Fraction(floor) / scale


def _round_half_even(value: Fraction) -> Fraction:
    """``nearbyint``/``nearbyintl`` in the default rounding mode."""
    floor = value.numerator // value.denominator
    remainder = value - floor
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and floor % 2):
        floor += 1
    return Fraction(floor)


def _c_math() -> ctypes.CDLL:
    library = ctypes.CDLL(
        ctypes.util.find_library("m") or ctypes.util.find_library("c")
    )
    for name in ("log10", "pow"):
        function = getattr(library, name)
        function.restype = ctypes.c_double
        function.argtypes = [ctypes.c_double] * (1 if name == "log10" else 2)
    return library


_LIBM = _c_math()


def scaling_table(bits: int) -> tuple[list[Fraction], int]:
    """R's ``tbl`` and ``KP_MAX`` in the branch the significand width selects."""
    kp_max = 22 if bits == DOUBLE_BITS else 27
    # Each entry is the C double constant 1e<k> (exact through 1e22).
    return [Fraction(float(f"1e{power}")) for power in range(kp_max + 1)], kp_max


def scientific(
    x: float,
    bits: int,
    *,
    long_double_power: Callable[[int], Fraction] | None = None,
) -> tuple[int, int, int, bool]:
    """``scientific()`` for one finite nonzero double: neg, kpower, nsig, widens."""
    table, kp_max = scaling_table(bits)

    def rnd(value: Fraction) -> Fraction:
        return round_to_bits(value, bits)

    def rnd_double(value: Fraction) -> Fraction:
        return round_to_bits(value, DOUBLE_BITS)

    negative = int(x < 0.0)
    r = Fraction(abs(x))
    kp = math.floor(_LIBM.log10(abs(x))) - DIGITS + 1
    if abs(kp) <= kp_max:
        if kp > 0:
            r_prec = rnd(r / table[kp])
        elif kp < 0:
            r_prec = rnd(r * table[-kp])
        else:
            r_prec = r
    elif bits != DOUBLE_BITS:
        power = (
            long_double_power(kp)
            if long_double_power is not None
            else rnd(Fraction(10) ** kp)
        )
        r_prec = rnd(r / power)
    elif kp <= R_DEC_MIN_EXPONENT:
        shifted = rnd_double(r * Fraction(1e303))
        r_prec = rnd_double(shifted / Fraction(_LIBM.pow(10.0, float(kp + 303))))
    else:
        r_prec = rnd_double(r / Fraction(_LIBM.pow(10.0, float(kp))))
    if r_prec < table[DIGITS - 1]:
        r_prec = rnd(r_prec * 10)
        kp -= 1
    # alpha = (double) nearbyintl(r_prec): an integer below 2^53, so exact.
    alpha = _round_half_even(r_prec)
    nsig = DIGITS
    for _ in range(DIGITS):
        alpha = rnd_double(alpha / 10)
        if alpha == math.floor(alpha):
            nsig -= 1
        else:
            break
    if nsig == 0:
        nsig = 1
        kp += 1
    kpower = kp + DIGITS - 1
    right_cut = min(max(DIGITS - kpower, 0), kp_max)
    fuzz = rnd_double(Fraction(1, 2) / table[right_cut])
    # r < tbl[kpower] - fuzz is evaluated in the table's (long double) type.
    widens = 0 < kpower <= kp_max and r < rnd(table[kpower] - fuzz)
    return negative, kpower, nsig, widens


def r_number_label(
    x: float,
    bits: int,
    *,
    long_double_power: Callable[[int], Fraction] | None = None,
) -> str:
    """Pinned R's ``paste0(x)`` for one double at a significand width."""
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0.0:
        return "0"
    negative, kpower, nsig, widens = scientific(
        x, bits, long_double_power=long_double_power
    )
    # formatReal for a single value (scipen 0, nsmall 0).
    left = kpower + 1 - int(widens)
    signed_left = negative + (1 if left <= 0 else left)
    right = max(nsig - left, 0)
    fixed_width = signed_left + right + (right != 0)
    exponent_digits = 2 if (left > 100 or left <= -99) else 1
    decimals = nsig - 1
    scientific_width = negative + (decimals > 0) + decimals + 4 + exponent_digits
    if fixed_width <= scientific_width:
        text = format(x, f".{right}f")
    else:
        text = format(x, f"#.{decimals}e" if decimals else ".0e")
    return _drop_trailing_zeros(text)


def _drop_trailing_zeros(text: str) -> str:
    """EncodeRealDrop0: drop zeros (and a bare '.') after the decimal mark."""
    point = text.find(".")
    if point < 0:
        return text
    end = point + 1
    while end < len(text) and text[end].isdigit():
        end += 1
    digits = text[point + 1 : end].rstrip("0")
    head = text[:point] + ("." + digits if digits else "")
    return head + text[end:]
