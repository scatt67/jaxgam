"""Distribution families for GAMs.

Exports the base class, four standard families, and the registry
function for string-based family lookup.
"""

from jaxgam.families.base import ExponentialFamily
from jaxgam.families.registry import get_family
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson

__all__ = [
    "Binomial",
    "ExponentialFamily",
    "Gamma",
    "Gaussian",
    "Poisson",
    "get_family",
]
