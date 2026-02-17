"""Smooth class registry and dispatch.

Provides ``get_smooth_class()`` which maps basis type strings
(e.g. ``"tp"``, ``"ts"``) to the corresponding Smooth subclass.

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 5.1
"""

from __future__ import annotations

from pymgcv.smooths.base import Smooth
from pymgcv.smooths.cubic import (
    CubicRegressionSmooth,
    CubicShrinkageSmooth,
    CyclicCubicSmooth,
)
from pymgcv.smooths.tensor import TensorInteractionSmooth, TensorProductSmooth
from pymgcv.smooths.tprs import TPRSShrinkageSmooth, TPRSSmooth

# Canonical basis type -> smooth class mapping
_SMOOTH_REGISTRY: dict[str, type[Smooth]] = {
    "tp": TPRSSmooth,
    "ts": TPRSShrinkageSmooth,
    "cr": CubicRegressionSmooth,
    "cs": CubicShrinkageSmooth,
    "cc": CyclicCubicSmooth,
    "te": TensorProductSmooth,
    "ti": TensorInteractionSmooth,
}


def get_smooth_class(bs_name: str) -> type[Smooth]:
    """Look up and return a Smooth class by basis type name.

    Parameters
    ----------
    bs_name : str
        Basis type name (e.g. ``"tp"``, ``"ts"``).

    Returns
    -------
    type[Smooth]
        The corresponding Smooth subclass (not an instance).

    Raises
    ------
    KeyError
        If the basis type is not in the registry.

    Examples
    --------
    >>> cls = get_smooth_class("tp")
    >>> cls.__name__
    'TPRSSmooth'
    """
    key = bs_name.lower()
    if key not in _SMOOTH_REGISTRY:
        available = ", ".join(sorted(_SMOOTH_REGISTRY.keys()))
        raise KeyError(f"Unknown basis type: {bs_name!r}. Available: {available}")
    return _SMOOTH_REGISTRY[key]
