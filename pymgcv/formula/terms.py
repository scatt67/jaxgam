"""Term representation: smooth and parametric terms.

Data classes for formula terms extracted by the AST-based parser.
These are the structured representations used by the design matrix
assembly layer.

Design doc reference: Section 13.1
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SmoothSpec:
    """Specification for a smooth term (s, te, or ti).

    Parameters
    ----------
    variables : list[str]
        Covariate names. For ``s(x1)`` this is ``["x1"]``;
        for ``te(x1, x2)`` this is ``["x1", "x2"]``.
    bs : str
        Basis type. Default is ``"tp"`` for thin plate regression splines.
    k : int
        Basis dimension (number of knots). Default ``-1`` means "auto"
        (let the smooth constructor choose).
    by : str | None
        By-variable name for factor-by or numeric-by smooths.
    smooth_type : str
        One of ``"s"``, ``"te"``, ``"ti"``.
    extra_args : dict
        Any additional keyword arguments not captured above.
    """

    variables: list[str]
    bs: str = "tp"
    k: int = -1
    by: str | None = None
    smooth_type: str = "s"
    extra_args: dict = field(default_factory=dict)


@dataclass
class ParametricTerm:
    """Specification for a parametric (linear) term.

    Parameters
    ----------
    name : str
        Variable name as it appears in the formula.
    """

    name: str


@dataclass
class FormulaSpec:
    """Parsed formula specification.

    Contains all information extracted from an R-style formula string,
    split into response, smooth terms, parametric terms, and intercept.

    Parameters
    ----------
    response : str
        Response variable name (left-hand side of ``~``).
    smooth_terms : list[SmoothSpec]
        Parsed smooth terms (s, te, ti calls).
    parametric_terms : list[ParametricTerm]
        Parsed parametric (linear) terms.
    has_intercept : bool
        Whether the formula includes an intercept. Default ``True``.
        Set to ``False`` when ``0 +`` or ``- 1`` appears in the formula.
    """

    response: str
    smooth_terms: list[SmoothSpec] = field(default_factory=list)
    parametric_terms: list[ParametricTerm] = field(default_factory=list)
    has_intercept: bool = True
