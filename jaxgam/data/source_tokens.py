"""Pinned R 4.5 numeric tokens and sampling for bounded Phase 1 setup."""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from jaxgam.smooths.source_arithmetic import libm_log10


class _RRandom:
    """R 4.5's default Mersenne Twister plus rejection index sampler."""

    def __init__(self, seed: int) -> None:
        state = seed & 0xFFFFFFFF
        for _ in range(50):
            state = (69069 * state + 1) & 0xFFFFFFFF
        words = np.empty(625, dtype=np.uint32)
        for index in range(625):
            state = (69069 * state + 1) & 0xFFFFFFFF
            words[index] = state
        # R discards the first initialized word by replacing i_seed[0] with
        # the MT position 624.  NumPy's legacy MT state exposes the same 624
        # words and position, while its high-level float generator does not.
        self._generator = np.random.RandomState()
        self._generator.set_state(("MT19937", words[1:], 624, 0, 0.0))

    def _word(self) -> int:
        return int(self._generator.randint(0, 2**32, dtype=np.uint32))

    def index(self, upper: int) -> int:
        if upper <= 0:
            return 0
        bits = (upper - 1).bit_length()
        while True:
            value = 0
            for _ in range(0, bits + 1, 16):
                value = 65536 * value + (self._word() >> 16)
            value &= (1 << bits) - 1
            if value < upper:
                return value


def _r_sample_without_replacement(
    size: int, seed: int, generator: _RRandom | None = None
) -> np.ndarray:
    """Port ``sample.int(size, size)`` for R 4.5's default RNG kinds."""
    if generator is None:
        generator = _RRandom(seed)
    pool = np.arange(size, dtype=np.int32)
    result = np.empty(size, dtype=np.int32)
    remaining = size
    for output in range(size):
        chosen = generator.index(remaining)
        result[output] = pool[chosen]
        remaining -= 1
        pool[chosen] = pool[remaining]
    return result


def _r_number_text(value: Any) -> str:
    """Format a finite scalar as R's numeric ``paste0`` representation."""
    if isinstance(value, (bool, np.bool_, int, np.integer)):
        return str(int(value))
    numeric = float(value)
    if np.isnan(numeric):
        return "NaN"
    if np.isposinf(numeric):
        return "Inf"
    if np.isneginf(numeric):
        return "-Inf"
    if numeric == 0.0:
        return "0"

    # StringFromReal temporarily sets R_print.digits to DBL_DIG (15), then
    # calls scientific() and formatReal() to choose the representation.  The
    # significant-digit calculation selects fixed versus exponential output;
    # it does *not* round the value passed to the final ``%f`` conversion.
    # That distinction preserves all integer digits of exactly represented
    # large doubles (for example 8956883870379507).
    digits = 15
    magnitude = abs(numeric)
    # scientific(): kp = (int) floor(log10(r)) - R + 1, with the C library's
    # log10. A vectorized NumPy log10 can differ by an ulp next to a power of
    # ten and move kp; libm_log10 is the same symbol R calls.
    k_pos = math.floor(float(libm_log10(np.array([magnitude]))[0])) - digits + 1
    has_extended_precision = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
    if has_extended_precision:
        magnitude_precise: float | np.longdouble = np.longdouble(magnitude)
        if abs(k_pos) <= 27:
            scale = np.longdouble(f"1e{abs(k_pos)}")
            if k_pos > 0:
                magnitude_precise /= scale
            elif k_pos < 0:
                magnitude_precise *= scale
        else:
            magnitude_precise /= np.longdouble(10.0) ** np.longdouble(k_pos)
    else:
        # R's non-long-double branch rescales subnormal values before the
        # power-of-ten division.  Directly evaluating 10**k_pos underflows on
        # platforms where longdouble is the same width as double (macOS).
        magnitude_precise = magnitude
        if abs(k_pos) <= 22:
            scale = float(f"1e{abs(k_pos)}")
            if k_pos > 0:
                magnitude_precise /= scale
            elif k_pos < 0:
                magnitude_precise *= scale
        elif k_pos <= sys.float_info.min_10_exp:
            magnitude_precise = (magnitude_precise * 1e303) / math.pow(
                10.0, k_pos + 303
            )
        else:
            magnitude_precise /= math.pow(10.0, k_pos)
    digit_floor = np.longdouble("1e14") if has_extended_precision else 1e14
    if magnitude_precise < digit_floor:
        magnitude_precise *= np.longdouble(10.0)
        k_pos -= 1

    alpha = float(np.rint(magnitude_precise))
    n_significant = digits
    for _ in range(digits):
        alpha /= 10.0
        if alpha == np.floor(alpha):
            n_significant -= 1
        else:
            break
    if n_significant == 0:
        n_significant = 1
        k_pos += 1
    k_power = k_pos + digits - 1

    right_cut = min(max(digits - k_power, 0), 27)
    fuzz = 0.5 / float(np.longdouble(f"1e{right_cut}"))
    rounding_widens = (
        0 < k_power <= 27 and magnitude < float(np.longdouble(f"1e{k_power}")) - fuzz
    )

    negative = int(numeric < 0.0)
    left = k_power + 1 - int(rounding_widens)
    signed_left = negative + (1 if left <= 0 else left)
    right = n_significant - left
    fixed_digits = max(right, 0)
    if left < 0:
        signed_left = 1 + negative
    fixed_width = signed_left + fixed_digits + int(fixed_digits != 0)

    exponent_digits_minus_one = 2 if left > 100 or left <= -99 else 1
    scientific_digits = n_significant - 1
    scientific_width = (
        negative
        + int(scientific_digits > 0)
        + scientific_digits
        + 4
        + exponent_digits_minus_one
    )
    if fixed_width <= scientific_width:
        return (
            format(numeric, f".{fixed_digits}f").rstrip("0").rstrip(".")
            if fixed_digits
            else format(numeric, ".0f")
        )

    encoded = format(numeric, f"#{'.' + str(scientific_digits)}e")
    mantissa, exponent = encoded.split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_value = int(exponent)
    exponent_width = exponent_digits_minus_one + 1
    return f"{mantissa}e{exponent_value:+0{exponent_width + 1}d}"
