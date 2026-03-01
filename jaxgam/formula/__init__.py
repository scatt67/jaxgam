"""Formula parsing and term representation.

Public API
----------
parse_formula : Parse R-style formula strings.
FormulaSpec : Parsed formula specification.
SmoothSpec : Smooth term specification.
ParametricTerm : Parametric term specification.
"""

from jaxgam.formula.parser import parse_formula
from jaxgam.formula.terms import FormulaSpec, ParametricTerm, SmoothSpec

__all__ = [
    "FormulaSpec",
    "ParametricTerm",
    "SmoothSpec",
    "parse_formula",
]
