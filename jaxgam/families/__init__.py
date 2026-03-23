"""Distribution families for GAMs.

Exports the base classes, standard families, extended families,
and the registry function for string-based family lookup.
"""

from jaxgam.families.base import (
    NON_NEGATIVE,
    POSITIVE,
    REAL,
    UNIT_INTERVAL,
    ExponentialFamily,
    ResponseSupport,
)
from jaxgam.families.extended import ExtendedFamily
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.registry import family_registry, get_family
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson

__all__ = [
    "NON_NEGATIVE",
    "POSITIVE",
    "REAL",
    "UNIT_INTERVAL",
    "Binomial",
    "ExponentialFamily",
    "ExtendedFamily",
    "Gamma",
    "Gaussian",
    "NegativeBinomial",
    "Poisson",
    "ResponseSupport",
    "family_registry",
    "get_family",
]
