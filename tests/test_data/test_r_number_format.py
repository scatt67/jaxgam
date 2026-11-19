"""R's numeric labels against an exact model of format.c's scientific()."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from jaxgam.data.source_tokens import _r_number_text
from tests.r_number_format import r_number_label, round_to_bits

# Outcome depends on the platform's long double and C library.
pytestmark = pytest.mark.platform_sensitive

# Significand width of this platform's long double: 53 where it is double
# (macOS arm64, Windows), 64 for x87 extended (Linux x86_64), 113 for
# binary128 (Linux aarch64). R's scientific() follows the same type.
PLATFORM_BITS = int(np.finfo(np.longdouble).nmant) + 1


def _platform_powl(kp: int) -> Fraction:
    """powl(10, kp) from this platform's C library, as R calls it."""
    return Fraction(*(np.longdouble(10.0) ** np.longdouble(kp)).as_integer_ratio())


def _model_labels(values: np.ndarray, bits: int = PLATFORM_BITS) -> list[str]:
    power = _platform_powl if bits == PLATFORM_BITS and bits > 53 else None
    return [r_number_label(float(v), bits, long_double_power=power) for v in values]


def _label_stress_values() -> np.ndarray:
    """Doubles where the label depends on each step of scientific()."""
    rng = np.random.default_rng(20261117)
    ties = []
    # 15-digit rounding ties (a 5 in the 16th significant digit) and their
    # neighbouring doubles, for kp around both ends of the scaling table and
    # beyond it, where powl/pow take over.
    for kp in [*range(-36, -18), *range(18, 31), -60, 45]:
        for _ in range(12):
            mantissa = int(rng.integers(10**14, 10**15))
            value = float(f"{mantissa}5e{kp - 1}")
            ties += [value, np.nextafter(value, 0.0), np.nextafter(value, np.inf)]
    # kp = -307 and -308: R shifts by 1e303 only at or below -308.
    band = np.concatenate(
        [rng.uniform(1, 10, 300) * 1e-293, rng.uniform(1, 10, 150) * 1e-294]
    )
    # The widening test r < tbl[kpower] - fuzz at its boundaries.
    widening = []
    for k in range(1, 28):
        edge = float(f"1e{k}") - 0.5 / float(f"1e{min(max(15 - k, 0), 27)}")
        widening += [edge, np.nextafter(edge, 0.0), np.nextafter(edge, np.inf)]
    powers = np.array([float(f"1e{k}") for k in range(-30, 31)])
    values = np.concatenate(
        [
            ties,
            band,
            widening,
            powers,
            np.nextafter(powers, 0.0),
            np.nextafter(powers, np.inf),
            [5e-324, 2.2250738585072014e-308, 1.7976931348623157e308],
        ]
    )
    return np.concatenate([values, -values])


def test_model_rounds_to_nearest_even_at_each_width():
    one = Fraction(1)
    for bits in (53, 64, 113):
        ulp = Fraction(1, 2 ** (bits - 1))
        assert round_to_bits(one + ulp / 2, bits) == one
        assert round_to_bits(one + 3 * ulp / 2, bits) == one + 2 * ulp
        assert round_to_bits(-(one + ulp / 3), bits) == -one
    assert round_to_bits(Fraction(0), 64) == 0


def test_model_follows_the_scaling_table_and_the_platform_width():
    # A 15-digit tie at kp = -23, where the long-double branch multiplies by
    # tbl[23], the double nearest 1e23 (not 1e23 itself); scaling by the exact
    # power gives ...561 on 64- and 113-bit platforms instead of R's ...560.
    for bits in (53, 64, 113):
        assert r_number_label(2.341695031835605e-09, bits) == "2.3416950318356e-09"
    # A tie whose 15-digit label depends on the width R's scientific() uses.
    assert r_number_label(2.093902118201595e-14, 53) == "2.0939021182016e-14"
    assert r_number_label(2.093902118201595e-14, 64) == "2.09390211820159e-14"
    assert r_number_label(2.093902118201595e-14, 113) == "2.09390211820159e-14"


def test_scalar_labels_equal_the_exact_model_on_this_platform():
    values = _label_stress_values()
    assert [_r_number_text(float(v)) for v in values] == _model_labels(values)


def test_exact_model_equals_pinned_r_paste0(r_bridge):
    if r_bridge.mode != "rpy2":
        pytest.skip("R label oracle requires rpy2.")
    values = _label_stress_values()
    expected = [str(label) for label in r_bridge.source_numeric_tokens(values)]
    assert _model_labels(values) == expected
