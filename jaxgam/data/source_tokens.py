"""Pinned R 4.5 numeric tokens and sampling for bounded Phase 1 setup."""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np


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


def _r_sample_int(size: int, count: int, seed: int) -> np.ndarray:
    """Port ``set.seed(seed); sample.int(size, count)`` (zero-based draws).

    R 4.5's ``sample.int`` uses its hash sampler (``sample2``: draw an index
    and keep it unless already drawn) when ``size > 1e7`` and
    ``count <= size / 2``, and otherwise the partial Fisher-Yates draw, whose
    ``count`` draws are the first ``count`` of ``sample.int(size, size)``.
    Only ``count`` draws are made either way.
    """
    if size < 0 or not 0 <= count <= size:
        raise ValueError("sample.int needs 0 <= count <= size.")
    generator = _RRandom(seed)
    result = np.empty(count, dtype=np.int64)
    if size > 10_000_000 and count <= size / 2:
        drawn: set[int] = set()
        output = 0
        while output < count:
            value = generator.index(size)
            if value not in drawn:
                drawn.add(value)
                result[output] = value
                output += 1
        return result
    pool: dict[int, int] = {}
    remaining = size
    for output in range(count):
        chosen = generator.index(remaining)
        result[output] = pool.get(chosen, chosen)
        remaining -= 1
        pool[chosen] = pool.get(remaining, remaining)
    return result


def _libm_log10(values: np.ndarray) -> np.ndarray:
    """The C library's log10 (:func:`jaxgam.smooths.source_arithmetic.libm_log10`).

    Imported on use: ``jaxgam.data`` re-exports these helpers while the
    package loads, and importing ``jaxgam.smooths`` there would load the
    smooth constructors, which import this module.
    """
    from jaxgam.smooths.source_arithmetic import libm_log10

    return libm_log10(values)


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
    k_pos = math.floor(float(_libm_log10(np.array([magnitude]))[0])) - digits + 1
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


_FUZZ_POWERS = tuple(float(np.longdouble(f"1e{power}")) for power in range(28))


def _r_number_texts(values: np.ndarray) -> list[str]:
    """Vectorized :func:`_r_number_text` for a float64 array.

    Every arithmetic step of the scalar port (its long-double or double
    rescaling, ``rint``, significant-digit count and width choice) runs
    elementwise in the same precision, so each label is identical to the
    scalar port's; only the final string formatting is per value.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    texts = [""] * len(values)
    for index in np.flatnonzero(~np.isfinite(values) | (values == 0.0)):
        texts[index] = _r_number_text(float(values[index]))
    regular = np.flatnonzero(np.isfinite(values) & (values != 0.0))
    if not len(regular):
        return texts
    numeric = values[regular]
    digits = 15
    magnitude = np.abs(numeric)
    # kp from the C library's log10, as scientific() computes it; NumPy's
    # log10 ufunc can dispatch to SIMD maths that differ by an ulp.
    k_pos = np.floor(_libm_log10(magnitude)).astype(np.int64) - digits + 1
    has_extended_precision = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
    if has_extended_precision:
        precise = magnitude.astype(np.longdouble)
        scales = np.array(
            [np.longdouble(f"1e{power}") for power in range(28)], dtype=np.longdouble
        )
        small = np.abs(k_pos) <= 27
        scale = scales[np.minimum(np.abs(k_pos), 27)]
        down, up = small & (k_pos > 0), small & (k_pos < 0)
        precise[down] = precise[down] / scale[down]
        precise[up] = precise[up] * scale[up]
        # powl(10, kp) once per distinct exponent, the scalar port's operation.
        for power in np.unique(k_pos[~small]).tolist():
            rows = k_pos == power
            precise[rows] = precise[rows] / (
                np.longdouble(10.0) ** np.longdouble(power)
            )
        digit_floor = np.longdouble("1e14")
    else:
        precise = magnitude.copy()
        scales_double = np.array([float(f"1e{power}") for power in range(23)])
        small = np.abs(k_pos) <= 22
        scale_double = scales_double[np.minimum(np.abs(k_pos), 22)]
        down, up = small & (k_pos > 0), small & (k_pos < 0)
        precise[down] = precise[down] / scale_double[down]
        precise[up] = precise[up] * scale_double[up]
        # Rexp10(x) is C pow(10.0, x) (Defn.h); math.pow is that call, once
        # per distinct exponent, so no SIMD power loop is involved.
        for power in np.unique(k_pos[~small]).tolist():
            rows = k_pos == power
            if power <= sys.float_info.min_10_exp:
                precise[rows] = (precise[rows] * 1e303) / math.pow(10.0, power + 303)
            else:
                precise[rows] = precise[rows] / math.pow(10.0, power)
        digit_floor = 1e14
    low = precise < digit_floor
    precise[low] = precise[low] * np.longdouble(10.0)
    k_pos = k_pos - low
    alpha = np.rint(precise).astype(np.float64)
    n_significant = np.full(len(numeric), digits, dtype=np.int64)
    trailing = np.ones(len(numeric), dtype=bool)
    for _ in range(digits):
        alpha = alpha / 10.0
        trailing &= alpha == np.floor(alpha)
        n_significant -= trailing
    no_digits = n_significant == 0
    n_significant[no_digits] = 1
    k_pos = k_pos + no_digits
    k_power = k_pos + digits - 1
    fuzz_powers = np.array(_FUZZ_POWERS)
    right_cut = np.clip(digits - k_power, 0, 27)
    fuzz = 0.5 / fuzz_powers[right_cut]
    rounding_widens = (
        (k_power > 0)
        & (k_power <= 27)
        & (magnitude < fuzz_powers[np.clip(k_power, 0, 27)] - fuzz)
    )
    negative = (numeric < 0.0).astype(np.int64)
    left = k_power + 1 - rounding_widens
    signed_left = np.where(left < 0, 1 + negative, negative + np.maximum(left, 1))
    fixed_digits = np.maximum(n_significant - left, 0)
    fixed_width = signed_left + fixed_digits + (fixed_digits != 0)
    exponent_digits_minus_one = np.where((left > 100) | (left <= -99), 2, 1)
    scientific_digits = n_significant - 1
    scientific_width = (
        negative
        + (scientific_digits > 0)
        + scientific_digits
        + 4
        + exponent_digits_minus_one
    )
    use_fixed = fixed_width <= scientific_width
    for position, index in enumerate(regular.tolist()):
        number = float(numeric[position])
        if use_fixed[position]:
            places = int(fixed_digits[position])
            texts[index] = (
                format(number, f".{places}f").rstrip("0").rstrip(".")
                if places
                else format(number, ".0f")
            )
            continue
        encoded = format(number, f"#.{int(scientific_digits[position])}e")
        mantissa, exponent = encoded.split("e")
        if "." in mantissa:
            mantissa = mantissa.rstrip("0").rstrip(".")
        width = int(exponent_digits_minus_one[position]) + 1
        texts[index] = f"{mantissa}e{int(exponent):+0{width + 1}d}"
    return texts


def r_row_token(row: Any) -> str:
    """Return mgcv ``uniquecombs``'s label of one multi-column row.

    ``paste0`` of each value's R numeric text, separated by ``"*"``.
    """
    return "*".join(_r_number_text(value) for value in row)


def r_uniquecombs_ordered(x: np.ndarray) -> np.ndarray:
    """Port the unique rows of ``uniquecombs(x, ordered=TRUE)`` in R's order.

    One column follows R's ``sort(unique(x))``. Several columns are
    deduplicated by their :func:`r_row_token` label, keeping each label's
    first row in data order, and ordered by the labels' C-locale collation
    (byte order), as ``uniquecombs`` does under ``Sys.setlocale("C")``.
    Exactly equal rows always share a label, so labels are formatted once
    per exactly unique row.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("uniquecombs needs a nonempty two-dimensional array.")
    if x.shape[1] == 1:
        return np.unique(x[:, 0])[:, np.newaxis]
    exact, first_index = np.unique(x, axis=0, return_index=True)
    labels = None
    for number, column in enumerate(exact.T):
        distinct, inverse = np.unique(column, return_inverse=True)
        texts = np.array(_r_number_texts(distinct), dtype=np.bytes_)[inverse]
        labels = texts if number == 0 else np.char.add(np.char.add(labels, b"*"), texts)
    # C collation is byte order; labels contain no NUL bytes, so NumPy's
    # byte-string order is it. Among exactly unique rows sharing a label,
    # uniquecombs keeps the first in data order.
    order = np.lexsort((first_index, labels))
    labels, first_index = labels[order], first_index[order]
    keep = np.ones(len(labels), dtype=bool)
    keep[1:] = labels[1:] != labels[:-1]
    return x[first_index[keep]].copy()


def r_sample_knot_rows(
    unique_rows: np.ndarray, max_knots: int, seed: int
) -> np.ndarray:
    """Port smooth.construct.{gp,tp}'s large-data knot sample.

    ``temp.seed(seed)`` (R's default RNG kinds), then
    ``xu[sample(1:nu, max.knots, replace=FALSE), ]``, i.e. the draws of
    ``sample.int(nu, max.knots)`` (:func:`_r_sample_int`) in draw order.
    """
    draws = _r_sample_int(len(unique_rows), max_knots, seed)
    return np.asarray(unique_rows)[draws].copy()
