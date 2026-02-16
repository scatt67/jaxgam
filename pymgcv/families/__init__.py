"""Distribution families for GAMs.

Exports the base class, four standard families, and the registry
function for string-based family lookup.
"""

from pymgcv.families.base import ExponentialFamily
from pymgcv.families.registry import get_family
from pymgcv.families.standard import Binomial, Gamma, Gaussian, Poisson

__all__ = [
    "ExponentialFamily",
    "Gaussian",
    "Binomial",
    "Poisson",
    "Gamma",
    "get_family",
]
