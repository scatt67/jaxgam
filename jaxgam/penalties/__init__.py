"""Penalty matrix construction and manipulation.

This module provides the Penalty and CompositePenalty classes for
representing smoothness penalties in GAMs.

This is Phase 1 code (NumPy only, no JAX imports).
"""

from jaxgam.penalties.penalty import CompositePenalty, Penalty

__all__ = ["CompositePenalty", "Penalty"]
