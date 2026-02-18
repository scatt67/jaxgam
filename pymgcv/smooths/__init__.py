"""Smooth basis and penalty construction (Phase 1 — NumPy only, no JAX).

Provides the Smooth abstract base class, TPRS, cubic spline, tensor
product, and by-variable implementations, and the smooth class registry.
"""

from pymgcv.smooths.base import Smooth
from pymgcv.smooths.by_variable import (
    FactorBySmooth,
    NumericBySmooth,
    is_factor,
    resolve_by_variable,
)
from pymgcv.smooths.cubic import (
    CubicRegressionSmooth,
    CubicShrinkageSmooth,
    CyclicCubicSmooth,
)
from pymgcv.smooths.registry import get_smooth_class
from pymgcv.smooths.tensor import TensorInteractionSmooth, TensorProductSmooth
from pymgcv.smooths.tprs import TPRSShrinkageSmooth, TPRSSmooth

__all__ = [
    "Smooth",
    "TPRSSmooth",
    "TPRSShrinkageSmooth",
    "CubicRegressionSmooth",
    "CubicShrinkageSmooth",
    "CyclicCubicSmooth",
    "TensorProductSmooth",
    "TensorInteractionSmooth",
    "FactorBySmooth",
    "NumericBySmooth",
    "is_factor",
    "resolve_by_variable",
    "get_smooth_class",
]
