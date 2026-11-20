"""Gaussian process correlation kernels.

This module is NumPy-only (Phase 1). It defines the kernel ABC, the five
concrete kernel classes used by ``GaussianProcessSmooth``, and the
``gp_kernel_registry`` that maps a kernel name string to its class. Per
docs/gaussian_process/design.md §5.3, the smooth class itself owns kernel resolution
(``__init__`` reads ``spec.extra_args["kernel"]`` and looks up the
class) and the kernel-evaluation / null-space helpers; this module is
just the kernel surface.

Every kernel follows pinned ``gpE`` (smooth.r) operation by operation: R's
``exp`` is the C library's and ``E^k`` is ``R_pow`` (``E*E`` for ``k == 2``,
otherwise C ``pow``), so the transcendental steps go through
``source_arithmetic``'s libm helpers rather than NumPy's ufuncs, whose SIMD
dispatch (AVX-512 on x86_64) can differ from libm in the last bit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from jaxgam.registry import Registry
from jaxgam.smooths.source_arithmetic import libm_exp, r_pow


class GPKernel(ABC):
    """Correlation kernel used by a Gaussian process smooth."""

    @abstractmethod
    def evaluate(
        self,
        e: npt.NDArray[np.floating],
        *,
        power: float = 1.0,
    ) -> npt.NDArray[np.floating]:
        """Evaluate the kernel at scaled distances ``e``."""
        raise NotImplementedError

    def validate(self, power: float) -> None:
        """Validate kernel-specific arguments."""
        _ = power


class SphericalKernel(GPKernel):
    """Compactly supported spherical correlation kernel."""

    def evaluate(
        self,
        e: npt.NDArray[np.floating],
        *,
        power: float = 1.0,
    ) -> npt.NDArray[np.floating]:
        _ = power
        return (1.0 - 1.5 * e + 0.5 * r_pow(e, 3.0)) * (e <= 1.0)


class PowerExponentialKernel(GPKernel):
    """Power-exponential correlation kernel."""

    def evaluate(
        self,
        e: npt.NDArray[np.floating],
        *,
        power: float = 1.0,
    ) -> npt.NDArray[np.floating]:
        return libm_exp(-r_pow(e, power))

    def validate(self, power: float) -> None:
        if not (0.0 < power <= 2.0):
            raise ValueError(
                f"GP power-exponential `power` must be in (0, 2], got {power!r}."
            )


class Matern32Kernel(GPKernel):
    """Matern 3/2 correlation kernel."""

    def evaluate(
        self,
        e: npt.NDArray[np.floating],
        *,
        power: float = 1.0,
    ) -> npt.NDArray[np.floating]:
        _ = power
        return (1.0 + e) * libm_exp(-e)


class Matern52Kernel(GPKernel):
    """Matern 5/2 correlation kernel."""

    def evaluate(
        self,
        e: npt.NDArray[np.floating],
        *,
        power: float = 1.0,
    ) -> npt.NDArray[np.floating]:
        _ = power
        exp_neg_e = libm_exp(-e)
        return exp_neg_e + (e * exp_neg_e) * (1.0 + e / 3.0)


class Matern72Kernel(GPKernel):
    """Matern 7/2 correlation kernel."""

    def evaluate(
        self,
        e: npt.NDArray[np.floating],
        *,
        power: float = 1.0,
    ) -> npt.NDArray[np.floating]:
        _ = power
        exp_neg_e = libm_exp(-e)
        return exp_neg_e + (e * exp_neg_e) * (1.0 + 0.4 * e + r_pow(e, 2.0) / 15.0)


gp_kernel_registry: Registry[GPKernel] = Registry(
    {
        "spherical": SphericalKernel,
        "power_exponential": PowerExponentialKernel,
        "matern_3_2": Matern32Kernel,
        "matern_5_2": Matern52Kernel,
        "matern_7_2": Matern72Kernel,
    },
    name="GP kernel",
)
