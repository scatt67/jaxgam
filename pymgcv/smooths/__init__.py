"""Smooth basis and penalty construction (Phase 1 — NumPy only, no JAX).

Provides the Smooth abstract base class, TPRS, cubic spline, tensor
product, and by-variable implementations, the smooth class registry,
and identifiability constraints via CoefficientMap.
"""

from pymgcv.smooths.base import Smooth
from pymgcv.smooths.by_variable import (
    FactorBySmooth,
    NumericBySmooth,
    is_factor,
    resolve_by_variable,
)
from pymgcv.smooths.constraints import CoefficientMap, TermBlock
from pymgcv.smooths.cubic import (
    CubicRegressionSmooth,
    CubicShrinkageSmooth,
    CyclicCubicSmooth,
)
from pymgcv.smooths.registry import get_smooth_class
from pymgcv.smooths.tensor import TensorInteractionSmooth, TensorProductSmooth
from pymgcv.smooths.tprs import TPRSShrinkageSmooth, TPRSSmooth

__all__ = [
    "CoefficientMap",
    "CubicRegressionSmooth",
    "CubicShrinkageSmooth",
    "CyclicCubicSmooth",
    "FactorBySmooth",
    "NumericBySmooth",
    "Smooth",
    "TPRSShrinkageSmooth",
    "TPRSSmooth",
    "TensorInteractionSmooth",
    "TensorProductSmooth",
    "TermBlock",
    "get_smooth_class",
    "is_factor",
    "resolve_by_variable",
]
