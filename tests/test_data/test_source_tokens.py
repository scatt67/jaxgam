"""Pinned R token and sampler helpers shared by radial and grid setup."""

import numpy as np

from jaxgam.data.source_tokens import (
    _r_number_text,
    _r_sample_without_replacement,
    _RRandom,
)


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
