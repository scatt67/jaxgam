"""Smooth basis and penalty construction (Phase 1 — NumPy only, no JAX).

Provides the Smooth abstract base class, TPRS, cubic spline, tensor
product, and by-variable implementations, the smooth class registry,
and identifiability constraints via CoefficientMap.
"""

from jaxgam.smooths.base import Smooth
from jaxgam.smooths.by_variable import (
    FactorBySmooth,
    NumericBySmooth,
    is_factor,
    resolve_by_variable,
)
from jaxgam.smooths.constraints import CoefficientMap, TermBlock
from jaxgam.smooths.cubic import (
    CubicRegressionSmooth,
    CubicShrinkageSmooth,
    CyclicCubicSmooth,
)
from jaxgam.smooths.registry import get_smooth_class
from jaxgam.smooths.tensor import TensorInteractionSmooth, TensorProductSmooth
from jaxgam.smooths.tprs import TPRSShrinkageSmooth, TPRSSmooth

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
