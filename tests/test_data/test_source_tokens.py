"""Pinned R token and sampler helpers shared by radial and grid setup."""

import numpy as np
import pytest

from jaxgam.data.source_tokens import (
    _r_number_text,
    _r_number_texts,
    _r_sample_int,
    _r_sample_without_replacement,
    _RRandom,
    r_row_token,
    r_sample_knot_rows,
    r_uniquecombs_ordered,
)

# Outcome can depend on the CPU's SIMD maths or BLAS kernel family.
pytestmark = pytest.mark.platform_sensitive


def test_source_number_tokens_cover_r_boundary_values():
    assert _r_number_text(8956883870379507.0) == "8956883870379507"
    assert _r_number_text(np.nextafter(0.0, 1.0)) == "4.94065645841247e-324"
    assert _r_number_text(-0.0) == "0"
    assert _r_number_text(7) == "7"
    assert _r_number_text(np.int64(8)) == "8"
    assert _r_number_text(np.nan) == "NaN"
    assert _r_number_text(np.inf) == "Inf"
    assert _r_number_text(-np.inf) == "-Inf"
    assert _r_number_text(1e-100) == "1e-100"
    assert _RRandom(7).index(0) == 0


def test_sampler_matches_pinned_r_seed_and_keeps_caller_state(r_bridge):
    expected = r_bridge.sample_int_permutation(13, 7)
    rng = _RRandom(7)
    np.testing.assert_array_equal(
        _r_sample_without_replacement(13, 7, rng) + 1, expected
    )
    assert sorted(_r_sample_without_replacement(13, 7).tolist()) == list(range(13))


def _powers_of_ten_and_neighbours() -> np.ndarray:
    """Powers of ten and their one-ulp neighbours, where log10 decides kp."""
    values = []
    for exponent in range(-30, 31):
        power = float(f"1e{exponent}")
        values += [np.nextafter(power, 0.0), power, np.nextafter(power, np.inf)]
    return np.asarray(values)


def test_tokens_next_to_powers_of_ten_match_pinned_r(r_bridge):
    """scientific()'s kp comes from C log10; ulp neighbours of 10^k decide it."""
    values = _powers_of_ten_and_neighbours()
    expected = r_bridge.source_numeric_tokens(values)
    actual = np.asarray([_r_number_text(value) for value in values])
    np.testing.assert_array_equal(actual, expected)


def test_uniquecombs_orders_one_column_numerically_and_rows_by_c_labels():
    np.testing.assert_array_equal(
        r_uniquecombs_ordered(np.array([[2.0], [-1.0], [2.0], [0.5]])),
        [[-1.0], [0.5], [2.0]],
    )
    rows = np.array([[-0.1, 1.0], [-0.2, 1.0], [10.0, 0.0], [9.0, 0.0], [-0.1, 1.0]])
    # C collation of the labels: "-0.1*1" < "-0.2*1" < "10*0" < "9*0".
    np.testing.assert_array_equal(
        r_uniquecombs_ordered(rows), [[-0.1, 1.0], [-0.2, 1.0], [10.0, 0.0], [9.0, 0.0]]
    )
    assert r_row_token((-0.1, 1.0)) == "-0.1*1"


def test_uniquecombs_merges_rows_equal_to_fifteen_digits_keeping_the_first():
    later = 0.1 + 2.0**-55  # prints as 0.1 at 15 significant digits
    rows = np.array([[later, 1.0], [0.3, 2.0], [0.1, 1.0]])
    unique = r_uniquecombs_ordered(rows)
    np.testing.assert_array_equal(unique, [[later, 1.0], [0.3, 2.0]])
    with pytest.raises(ValueError, match="two-dimensional"):
        r_uniquecombs_ordered(np.empty((0, 2)))


def test_knot_sample_is_the_pinned_r_draw_prefix(r_bridge):
    rows = np.arange(26.0).reshape(13, 2)
    expected = r_bridge.sample_int_permutation(13, 5)[:4] - 1
    np.testing.assert_array_equal(r_sample_knot_rows(rows, 4, 5), rows[expected])


def test_vectorized_number_labels_equal_the_scalar_port():
    rng = np.random.default_rng(123)
    powers = 10.0 ** np.arange(-320, 309, dtype=float)
    powers = powers[np.isfinite(powers) & (powers > 0)]
    values = np.concatenate(
        [
            rng.normal(0, 8, 4000),
            rng.uniform(-1, 1, 4000) * 10.0 ** rng.integers(-30, 30, 4000),
            np.round(rng.normal(0, 100, 2000), 3),
            powers,
            np.nextafter(powers, 0),
            np.nextafter(powers, np.inf),
            np.arange(1, 2000) / 1000.0,
            [5e-324, 8956883870379507.0, 999999999999999.9, 0.0, -0.0],
            [np.nan, np.inf, -np.inf, 1 / 3, 99999.99999999999],
        ]
    )
    values = np.concatenate([values, -values])
    assert _r_number_texts(values) == [_r_number_text(float(v)) for v in values]
    assert _r_number_texts(np.array([])) == []


@pytest.mark.parametrize(("size", "count"), [(13, 13), (13, 4), (10_000_001, 5)])
def test_sample_int_matches_pinned_r_draws(r_bridge, size, count):
    """Partial Fisher-Yates, and R's hash sampler beyond 1e7 at size/2."""
    expected = r_bridge.sample_int_draws(size, count, 11) - 1
    np.testing.assert_array_equal(_r_sample_int(size, count, 11), expected)
    with pytest.raises(ValueError, match="count"):
        _r_sample_int(3, 4, 1)


def _adversarial_doubles() -> np.ndarray:
    """A few thousand doubles where R's 15-digit labels are hardest to port."""
    rng = np.random.default_rng(20261116)
    ties = []
    for _ in range(600):
        # Decimal values with a 5 in the 16th significant digit, and their
        # neighbouring doubles: 15-digit rounding ties in every direction.
        mantissa = rng.integers(10**14, 10**15)
        exponent = int(rng.integers(-20, 25))
        value = float(f"{mantissa}5e{exponent - 15}")
        ties += [value, np.nextafter(value, 0), np.nextafter(value, np.inf)]
    large = [1e15, 1e15 - 1, 1e15 + 1, 999999999999999.5, 2.0**53, 2.0**53 + 2]
    large += [1e16, 1e21, 1e22, 1e23, 8956883870379507.0, 1.7976931348623157e308]
    subnormal = [5e-324, 1e-320, 2.2250738585072009e-308, 2.2250738585072014e-308]
    subnormal += [1e-310, 4.9406564584124654e-324 * 12345]
    powers = 10.0 ** np.arange(-300, 300, 7, dtype=float)
    scattered = rng.uniform(-1, 1, 1500) * 10.0 ** rng.integers(-30, 30, 1500)
    values = np.concatenate(
        [
            ties,
            large,
            subnormal,
            powers,
            np.nextafter(powers, 0),
            np.nextafter(powers, np.inf),
            scattered,
            np.round(rng.normal(0, 100, 300), 3),
        ]
    )
    return np.concatenate([values, -values])


def test_number_labels_match_pinned_r_paste0(r_bridge):
    """Both label ports equal R's own paste0 text, incl. long-double branches."""
    if r_bridge.mode != "rpy2":
        pytest.skip("R label oracle requires rpy2.")
    values = _adversarial_doubles()
    expected = r_bridge.paste0_number_labels(values)
    assert _r_number_texts(values) == expected
    assert [_r_number_text(float(value)) for value in values] == expected
