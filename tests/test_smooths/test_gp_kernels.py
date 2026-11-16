"""Tests for Gaussian process correlation kernels and registry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jaxgam.smooths.gp_kernels import (
    GPKernel,
    Matern32Kernel,
    Matern52Kernel,
    Matern72Kernel,
    PowerExponentialKernel,
    SphericalKernel,
    gp_kernel_registry,
)
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


def _closed_form(
    kernel_cls: type[GPKernel],
    e: np.ndarray,
    power: float,
) -> np.ndarray:
    if kernel_cls is SphericalKernel:
        return (1.0 - 1.5 * e + 0.5 * e**3) * (e <= 1.0)
    if kernel_cls is PowerExponentialKernel:
        return np.exp(-(e**power))
    if kernel_cls is Matern32Kernel:
        return (1.0 + e) * np.exp(-e)
    if kernel_cls is Matern52Kernel:
        return (1.0 + e + e**2 / 3.0) * np.exp(-e)
    if kernel_cls is Matern72Kernel:
        return (1.0 + e + 0.4 * e**2 + e**3 / 15.0) * np.exp(-e)
    raise AssertionError(f"Unhandled kernel class {kernel_cls!r}")


class TestKernelMath:
    """Closed-form match for each kernel class."""

    @pytest.mark.parametrize(
        ("kernel_cls", "power"),
        [
            (SphericalKernel, 1.0),
            (PowerExponentialKernel, 1.0),
            (PowerExponentialKernel, 2.0),
            (Matern32Kernel, 1.0),
            (Matern52Kernel, 1.0),
            (Matern72Kernel, 1.0),
        ],
    )
    def test_kernel_matches_closed_form(
        self,
        kernel_cls: type[GPKernel],
        power: float,
    ) -> None:
        """STRICT closed-form match at a fixed grid of scaled distances."""
        kernel = kernel_cls()
        e = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
        result = kernel.evaluate(e, power=power)
        expected = _closed_form(kernel_cls, e, power)
        collector = _AssertCollector()

        def assert_kernel_values() -> None:
            np.testing.assert_allclose(
                result,
                expected,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            )

        def assert_zero_past_unit_distance() -> None:
            np.testing.assert_allclose(result[e > 1.0], 0.0, atol=STRICT.atol)

        def assert_squared_exponential_form() -> None:
            np.testing.assert_allclose(
                result,
                np.exp(-(e**2)),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            )

        collector.check("kernel values", assert_kernel_values)
        if kernel_cls is SphericalKernel:
            collector.check("zero past e=1", assert_zero_past_unit_distance)
        if kernel_cls is PowerExponentialKernel and power == 2.0:
            collector.check("squared-exponential form", assert_squared_exponential_form)
        collector.raise_if_any(f"kernel {kernel_cls.__name__}")


class TestPowerValidation:
    """``validate`` is only meaningful for PowerExponentialKernel."""

    def test_power_validation(self) -> None:
        collector = _AssertCollector()

        def assert_valid_powers_accepted() -> None:
            kernel = PowerExponentialKernel()
            kernel.validate(0.5)
            kernel.validate(1.0)
            kernel.validate(2.0)

        def assert_power_zero_raises() -> None:
            with pytest.raises(ValueError, match=r"\(0, 2\]"):
                PowerExponentialKernel().validate(0.0)

        def assert_power_too_large_raises() -> None:
            with pytest.raises(ValueError, match=r"\(0, 2\]"):
                PowerExponentialKernel().validate(2.5)

        def assert_matern_ignores_power() -> None:
            for cls in (
                Matern32Kernel,
                Matern52Kernel,
                Matern72Kernel,
                SphericalKernel,
            ):
                cls().validate(2.5)

        collector.check("valid powers accepted", assert_valid_powers_accepted)
        collector.check("power=0 raises", assert_power_zero_raises)
        collector.check("power=2.5 raises", assert_power_too_large_raises)
        collector.check(
            "non-power-exp kernels ignore power",
            assert_matern_ignores_power,
        )
        collector.raise_if_any("power validation")


class TestRegistry:
    """The registry resolves all five canonical names; nothing more."""

    def test_registry_contents(self) -> None:
        expected = {
            "spherical": SphericalKernel,
            "power_exponential": PowerExponentialKernel,
            "matern_3_2": Matern32Kernel,
            "matern_5_2": Matern52Kernel,
            "matern_7_2": Matern72Kernel,
        }
        collector = _AssertCollector()

        def assert_keys_are_canonical_set() -> None:
            assert set(gp_kernel_registry.available) == set(expected)

        def assert_each_key_resolves_to_class() -> None:
            for key, cls in expected.items():
                assert gp_kernel_registry.get_class(key) is cls

        def assert_unknown_kernel_raises() -> None:
            with pytest.raises(KeyError, match="Unknown GP kernel"):
                gp_kernel_registry.get_class("matern32")

        collector.check("registry keys", assert_keys_are_canonical_set)
        collector.check("class resolution", assert_each_key_resolves_to_class)
        collector.check("unknown kernel raises", assert_unknown_kernel_raises)
        collector.raise_if_any("gp_kernel_registry")


def _r_pow_scalar(x: float, y: float) -> float:
    """R's ``x ^ y`` (``R_pow``) for finite non-negative ``x``."""
    if y == 2.0:
        return x * x
    if x == 1.0 or y == 0.0:
        return 1.0
    if x == 0.0:
        return 0.0
    return math.pow(x, y)


_GPE_SCALAR = {
    # gpE (smooth.r): exp is the C library's, E^k is R_pow.
    "spherical": lambda e, _k: (1 - 1.5 * e + 0.5 * _r_pow_scalar(e, 3.0)) * (e <= 1),
    "power_exponential": lambda e, k: math.exp(-_r_pow_scalar(e, k)),
    "matern_3_2": lambda e, _k: (1 + e) * math.exp(-e),
    "matern_5_2": lambda e, _k: math.exp(-e) + (e * math.exp(-e)) * (1 + e / 3),
    "matern_7_2": lambda e, _k: (
        math.exp(-e) + (e * math.exp(-e)) * (1 + 0.4 * e + _r_pow_scalar(e, 2.0) / 15)
    ),
}


@pytest.mark.parametrize("name", sorted(_GPE_SCALAR))
@pytest.mark.parametrize("power", [1.0, 1.5, 2.0])
def test_kernels_evaluate_gpe_with_libm_bitwise(name: str, power: float) -> None:
    """Each kernel is gpE's own expression with libm exp/pow, on every CPU."""
    rng = np.random.default_rng(23)
    e = np.concatenate(
        [rng.uniform(0.0, 4.0, 4000), [0.0, 1.0, np.nextafter(1.0, 2.0), 1e-300]]
    )
    kernel = gp_kernel_registry.get_instance(name)
    expected = np.array([_GPE_SCALAR[name](float(v), power) for v in e])
    np.testing.assert_array_equal(kernel.evaluate(e, power=power), expected)
