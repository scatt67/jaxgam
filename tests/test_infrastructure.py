"""Tests for test infrastructure: tolerances and fixtures."""

import numpy as np

from tests.tolerances import LOOSE, MODERATE, STRICT


class TestToleranceClasses:
    def test_strict_values(self) -> None:
        assert STRICT.rtol == 1e-10
        assert STRICT.atol == 1e-12
        assert STRICT.label == "strict"

    def test_moderate_values(self) -> None:
        assert MODERATE.rtol == 1e-4
        assert MODERATE.atol == 1e-6
        assert MODERATE.label == "moderate"

    def test_loose_values(self) -> None:
        assert LOOSE.rtol == 1e-2
        assert LOOSE.atol == 1e-4
        assert LOOSE.label == "loose"

    def test_frozen(self) -> None:
        """Tolerance instances should be immutable."""
        import pytest

        with pytest.raises(AttributeError):
            STRICT.rtol = 0.1  # type: ignore[misc]

    def test_tolerance_ordering(self) -> None:
        """STRICT < MODERATE < LOOSE in permissiveness."""
        assert STRICT.rtol < MODERATE.rtol < LOOSE.rtol
        assert STRICT.atol < MODERATE.atol < LOOSE.atol

    def test_usable_with_assert_allclose(self) -> None:
        """Tolerances work with np.testing.assert_allclose."""
        a = np.array([1.0, 2.0, 3.0])
        b = a + 1e-13
        np.testing.assert_allclose(a, b, rtol=STRICT.rtol, atol=STRICT.atol)


class TestFixtureReproducibility:
    """Fixtures must produce identical data across calls (seeded RNG)."""

    def test_gaussian_reproducible(self, simple_gaussian_data) -> None:
        df = simple_gaussian_data
        assert len(df) == 200
        assert list(df.columns) == ["x1", "x2", "y"]
        # Check first value is deterministic
        expected_x1_0 = np.random.default_rng(42).uniform(0, 1, 200)[0]
        assert df["x1"].iloc[0] == expected_x1_0

    def test_binomial_valid_response(self, simple_binomial_data) -> None:
        df = simple_binomial_data
        assert len(df) == 200
        assert set(df["y"].unique()).issubset({0.0, 1.0})

    def test_poisson_valid_response(self, simple_poisson_data) -> None:
        df = simple_poisson_data
        assert len(df) == 200
        assert (df["y"] >= 0).all()
        assert (df["y"] == df["y"].astype(int)).all()

    def test_gamma_valid_response(self, simple_gamma_data) -> None:
        df = simple_gamma_data
        assert len(df) == 200
        assert (df["y"] > 0).all()
