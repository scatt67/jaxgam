"""Smooth class registry and dispatch.

Provides ``get_smooth_class()`` which maps basis type strings
(e.g. ``"tp"``, ``"ts"``) to the corresponding Smooth subclass.

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 5.1
"""

from __future__ import annotations

from jaxgam.registry import Registry
from jaxgam.smooths.base import Smooth
from jaxgam.smooths.cubic import (
    CubicRegressionSmooth,
    CubicShrinkageSmooth,
    CyclicCubicSmooth,
)
from jaxgam.smooths.tensor import TensorInteractionSmooth, TensorProductSmooth
from jaxgam.smooths.tprs import TPRSShrinkageSmooth, TPRSSmooth

smooth_registry: Registry[Smooth] = Registry(
    {
        "tp": TPRSSmooth,
        "ts": TPRSShrinkageSmooth,
        "cr": CubicRegressionSmooth,
        "cs": CubicShrinkageSmooth,
        "cc": CyclicCubicSmooth,
        "te": TensorProductSmooth,
        "ti": TensorInteractionSmooth,
    },
    name="basis type",
)


def get_smooth_class(bs_name: str) -> type[Smooth]:
    """Look up and return a Smooth class by basis type name.

    Thin wrapper around ``smooth_registry.get_class()`` for backward compatibility.

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
    return smooth_registry.get_class(bs_name)
