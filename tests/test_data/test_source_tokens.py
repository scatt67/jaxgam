"""Pinned R token and sampler helpers shared by radial and grid setup."""

import numpy as np
import pytest

from jaxgam.data.source_tokens import (
    _r_number_text,
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
