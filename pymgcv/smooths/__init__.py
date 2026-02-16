"""Smooth basis and penalty construction (Phase 1 — NumPy only, no JAX).

Provides the Smooth abstract base class, TPRS and cubic spline
implementations, and the smooth class registry.
"""

from pymgcv.smooths.base import Smooth
from pymgcv.smooths.cubic import (
    CubicRegressionSmooth,
    CubicShrinkageSmooth,
    CyclicCubicSmooth,
)
from pymgcv.smooths.registry import get_smooth_class
from pymgcv.smooths.tprs import TPRSShrinkageSmooth, TPRSSmooth

__all__ = [
    "Smooth",
    "TPRSSmooth",
    "TPRSShrinkageSmooth",
    "CubicRegressionSmooth",
    "CubicShrinkageSmooth",
    "CyclicCubicSmooth",
    "get_smooth_class",
]
