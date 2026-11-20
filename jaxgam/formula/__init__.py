"""Formula parsing and term representation.

Public API
----------
parse_formula : Parse R-style formula strings.
FormulaSpec : Parsed formula specification.
SmoothSpec : Smooth term specification.
ParametricTerm : Parametric term specification.

Layout: the modules at this level parse the formula and build the design and
the prepared model that every route starts from. ``operator`` and
``discrete`` hold the setup of those two routes.
"""

from jaxgam.formula.parser import parse_formula
from jaxgam.formula.terms import FormulaSpec, ParametricTerm, SmoothSpec

__all__ = [
    "FormulaSpec",
    "ParametricTerm",
    "SmoothSpec",
    "parse_formula",
]
