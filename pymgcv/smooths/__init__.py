"""Smooth basis and penalty construction (Phase 1 — NumPy only, no JAX).

Provides the Smooth abstract base class, TPRS implementations,
and the smooth class registry.
"""

from pymgcv.smooths.base import Smooth
from pymgcv.smooths.registry import get_smooth_class
from pymgcv.smooths.tprs import TPRSShrinkageSmooth, TPRSSmooth

__all__ = [
    "Smooth",
    "TPRSSmooth",
    "TPRSShrinkageSmooth",
    "get_smooth_class",
]
