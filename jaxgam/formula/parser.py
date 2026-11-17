"""AST-based formula parser for R-style Wilkinson notation.

Parses formulas like ``"y ~ s(x1) + s(x2, k=20) + x3"`` into a
structured ``FormulaSpec`` using Python's ``ast`` module (no regex).

Design doc reference: Section 13.1

Supported patterns
------------------
- ``y ~ s(x1)`` -- basic smooth
- ``y ~ s(x1) + s(x2)`` -- multiple smooths
- ``y ~ s(x1, k=20)`` -- with basis dimension
- ``y ~ s(x1, bs="cr")`` -- with basis type
- ``y ~ s(x1, by=fac)`` -- factor-by variable
- ``y ~ s(x1) + x2 + x3`` -- parametric + smooth
- ``y ~ te(x1, x2)`` -- tensor product
- ``y ~ ti(x1, x2)`` -- tensor product interaction
- ``y ~ 0 + s(x1)`` or ``y ~ s(x1) - 1`` -- no intercept
- ``y ~ x1`` -- purely parametric
"""

from __future__ import annotations

import ast
from typing import Any

from jaxgam.formula.terms import (
    LINKED_SMOOTHING_PARAMETERS_MESSAGE,
    FormulaSpec,
    ParametricTerm,
    SmoothSpec,
)

_SMOOTH_FUNCTIONS = frozenset({"s", "te", "ti"})

_DEFAULT_BASIS: dict[str, str] = {
    "s": "tp",
    "te": "cr",  # R's te() defaults to cr, not tp
    "ti": "cr",  # R's ti() defaults to cr, not tp
}


def parse_formula(formula_str: str) -> FormulaSpec:
    """Parse an R-style formula string into a ``FormulaSpec``.

    Parameters
    ----------
    formula_str : str
        Formula in R-style Wilkinson notation, e.g.
        ``"y ~ s(x1) + s(x2, k=20) + x3"``.

    Returns
    -------
    FormulaSpec
        Parsed formula with response, smooth terms, parametric terms,
        and intercept flag.

    Raises
    ------
    ValueError
        If the formula is malformed (missing ``~``, empty response or
        RHS, smooth function with no arguments, etc.).
    NotImplementedError
        If a smooth uses a deferred argument (``fx=True``, per-term ``sp=``
        or a non-``None`` ``id=``).
    """
    # --- Split on ~ --------------------------------------------------------
    if "~" not in formula_str:
        raise ValueError(f"Formula must contain '~' separator. Got: {formula_str!r}")

    parts = formula_str.split("~", 1)
    response = parts[0].strip()
    rhs_raw = parts[1].strip()

    if not response:
        raise ValueError(
            f"Formula has empty response (left of '~'). Got: {formula_str!r}"
        )
    if not rhs_raw:
        raise ValueError(
            f"Formula has empty right-hand side (right of '~'). Got: {formula_str!r}"
        )

    # --- Parse RHS as a Python expression -----------------------------------
    # R's `-` for removing the intercept (e.g. `s(x1) - 1`) is valid Python
    # arithmetic, so `ast.parse` handles it directly.
    try:
        tree = ast.parse(rhs_raw, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Cannot parse formula RHS as an expression: {rhs_raw!r}"
        ) from exc

    # --- Walk AST to collect terms ------------------------------------------
    collector = _TermCollector()
    collector.visit(tree.body)

    return FormulaSpec(
        response=response,
        smooth_terms=collector.smooth_terms,
        parametric_terms=collector.parametric_terms,
        has_intercept=collector.has_intercept,
    )


class _TermCollector:
    """Walks an AST expression tree and collects formula terms.

    Note: this is a custom visitor, not an ``ast.NodeVisitor`` subclass,
    because we only handle expression nodes (not statements) and want
    explicit control over dispatch.
    """

    def __init__(self) -> None:
        self.smooth_terms: list[SmoothSpec] = []
        self.parametric_terms: list[ParametricTerm] = []
        self.has_intercept: bool = True

    def visit(self, node: ast.expr) -> None:
        """Dispatch on node type to collect terms."""
        if isinstance(node, ast.BinOp):
            self._visit_binop(node)
        elif isinstance(node, ast.Call):
            self._visit_call(node)
        elif isinstance(node, ast.Name):
            self._visit_name(node)
        elif isinstance(node, ast.Constant):
            # A bare constant like 0 or 1 — handle intercept removal
            self._visit_constant(node)
        elif isinstance(node, ast.UnaryOp):
            self._visit_unaryop(node)
        else:
            raise ValueError(
                f"Unsupported expression node in formula: {ast.dump(node)}"
            )

    def _visit_binop(self, node: ast.BinOp) -> None:
        """Handle ``+`` and ``-`` operators connecting terms.

        ``a + b + c`` parses left-nested, so a formula with m terms is an
        m-deep chain. The chain is flattened iteratively and its links are
        processed left to right, exactly the order of the former recursive
        walk, so the depth no longer grows with the number of terms.
        Parenthesized sub-chains (``a + (b + c)``) recurse once per
        parenthesis level, which Python's own parser bounds.
        """
        if not isinstance(node.op, (ast.Add, ast.Sub)):
            raise ValueError(
                f"Unsupported operator in formula: {type(node.op).__name__}. "
                f"Only '+' and '-' are supported. Got: {_unparse(node)}"
            )
        leftmost, links = _additive_chain(node)
        self.visit(leftmost)
        for link in links:
            if isinstance(link.op, ast.Add):
                self.visit(link.right)
            elif isinstance(link.right, ast.Constant) and link.right.value == 1:
                # ``expr - 1`` removes the intercept
                self.has_intercept = False
            else:
                raise ValueError(
                    "Subtraction in formula RHS is only supported as "
                    "'- 1' to remove the intercept. "
                    f"Got: {_unparse(link)}"
                )

    def _visit_call(self, node: ast.Call) -> None:
        """Handle function calls: s(), te(), ti()."""
        func_name = self._get_func_name(node)
        if func_name is None:
            raise ValueError(
                f"Cannot resolve function name in formula: {_unparse(node)}"
            )

        if func_name not in _SMOOTH_FUNCTIONS:
            raise ValueError(
                f"Unknown function '{func_name}()' in formula. "
                f"Supported: {sorted(_SMOOTH_FUNCTIONS)}"
            )

        if not node.args:
            raise ValueError(
                f"Smooth function '{func_name}()' requires at least one "
                f"variable argument. Got: {_unparse(node)}"
            )

        spec = _parse_smooth_call(func_name, node)
        # R's terms.formula collapses identical term labels, so `y ~ s(x) + s(x)`
        # yields a single smooth (interpret.gam0, mgcv.r:302). SmoothSpec equality
        # compares variables+bs+k+by+smooth_type+extra_args, matching that. A
        # *different*-config repeat (s(x,k=6)+s(x,k=8)) is NOT identical and is
        # kept, like mgcv (which warns but keeps both).
        if spec not in self.smooth_terms:
            self.smooth_terms.append(spec)

    def _visit_name(self, node: ast.Name) -> None:
        """Handle plain variable names as parametric terms (de-duplicated).

        R's ``model.matrix`` collapses a repeated parametric term (``y ~ x + x``)
        to a single column; without de-duplication the design matrix gains an
        aliased column, becomes rank-deficient, and splits the coefficient.
        """
        if any(t.name == node.id for t in self.parametric_terms):
            return
        self.parametric_terms.append(ParametricTerm(name=node.id))

    def _visit_constant(self, node: ast.Constant) -> None:
        """Handle constants: 0 removes intercept, 1 is explicit intercept."""
        if node.value == 0:
            self.has_intercept = False
        elif node.value == 1:
            # Explicit ``+ 1`` re-adds the intercept. R formula semantics are
            # order-sensitive and last-token-wins: ``y ~ 0 + x + 1`` and
            # ``y ~ x - 1 + 1`` both restore the intercept, while
            # ``y ~ x + 1 - 1`` removes it. The AST is walked left-to-right,
            # so simply setting the flag here reproduces R's behavior.
            self.has_intercept = True
        else:
            raise ValueError(
                f"Unexpected constant in formula: {node.value}. "
                f"Only 0 (no intercept) and 1 (intercept) are valid."
            )

    def _visit_unaryop(self, node: ast.UnaryOp) -> None:
        """Handle unary operators (e.g. ``-1`` for no intercept)."""
        if (
            isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and node.operand.value == 1
        ):
            self.has_intercept = False
        else:
            raise ValueError(
                f"Unsupported unary expression in formula: {_unparse(node)}"
            )

    @staticmethod
    def _get_func_name(node: ast.Call) -> str | None:
        """Extract function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        return None


def _parse_smooth_call(func_name: str, node: ast.Call) -> SmoothSpec:
    """Extract a ``SmoothSpec`` from an AST ``Call`` node.

    Parameters
    ----------
    func_name : str
        The smooth function name (``"s"``, ``"te"``, ``"ti"``).
    node : ast.Call
        The AST call node.

    Returns
    -------
    SmoothSpec
    """
    # Positional arguments are variable names
    variables: list[str] = []
    for arg in node.args:
        if isinstance(arg, ast.Name):
            variables.append(arg.id)
        else:
            raise ValueError(
                f"Positional arguments to {func_name}() must be variable "
                f"names. Got: {_unparse(arg)}"
            )

    # Keyword arguments
    bs = _DEFAULT_BASIS[func_name]
    k = -1
    by: str | None = None
    extra_args: dict[str, Any] = {}

    for kw in node.keywords:
        key = kw.arg
        if key == "k":
            k = _eval_kwarg_value(kw.value, key, func_name)
            if not isinstance(k, int):
                raise ValueError(
                    f"Argument 'k' in {func_name}() must be an integer. "
                    f"Got: {_unparse(kw.value)}"
                )
        elif key == "bs":
            bs = _eval_kwarg_value(kw.value, key, func_name)
            if not isinstance(bs, str):
                raise ValueError(
                    f"Argument 'bs' in {func_name}() must be a string. "
                    f"Got: {_unparse(kw.value)}"
                )
        elif key == "by":
            if isinstance(kw.value, ast.Name):
                by = kw.value.id
            elif isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                by = kw.value.value
            else:
                raise ValueError(
                    f"Argument 'by' in {func_name}() must be a variable "
                    f"name or string. Got: {_unparse(kw.value)}"
                )
        elif key == "fx":
            # fx=True (fixed-df, unpenalized regression spline) is a deferred
            # feature: silently fitting a penalized smooth instead would be a
            # wrong result, so follow CLAUDE.md's deferred-feature policy.
            if _eval_kwarg_value(kw.value, key, func_name):
                raise NotImplementedError(
                    "fx=True (fixed-df unpenalized regression spline) is not "
                    "supported in v1.0. mgcv fixes the smoothing parameter and "
                    "uses an unpenalized basis (EDF = basis dim). Planned for "
                    "v1.1. See docs/design.md Section 5.2."
                )
            # fx=False is the default; accept and ignore it.
        elif key == "sp":
            # Per-term sp= inside s()/te()/ti() is deferred. The model-level
            # GAM(formula, sp=[...]) argument IS honored; only the in-formula
            # per-term form is unsupported (would otherwise be silently dropped).
            if _eval_kwarg_value(kw.value, key, func_name) is not None:
                raise NotImplementedError(
                    "Per-term sp= inside s()/te()/ti() is not supported in "
                    "v1.0. Use the model-level GAM(formula, sp=[...]) argument "
                    "to fix all smoothing parameters, or omit sp to estimate "
                    "them by REML. Planned for v1.1. See docs/design.md "
                    "Section 4.3."
                )
            # sp=None is the R default; accept and ignore it.
        elif key == "id":
            # R tests is.null(id) (smooth.r s()/te()), so only Python None, the
            # spelling of R's NULL default, links nothing. Any other value,
            # NA included, is an id, and an unevaluable one is still an id.
            if not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                raise NotImplementedError(LINKED_SMOOTHING_PARAMETERS_MESSAGE)
        else:
            extra_args[key] = _eval_kwarg_value(kw.value, key, func_name)

    return SmoothSpec(
        variables=variables,
        bs=bs,
        k=k,
        by=by,
        smooth_type=func_name,
        extra_args=extra_args,
    )


def _eval_kwarg_value(node: ast.expr, key: str, func_name: str) -> Any:
    """Safely evaluate a keyword argument value from the AST.

    Uses ``ast.literal_eval`` for constants. Raises ``ValueError`` for
    complex expressions that cannot be evaluated at parse time (e.g.
    ``k=int(np.log(n))``).

    Parameters
    ----------
    node : ast.expr
        The AST node for the keyword value.
    key : str
        The keyword name (for error messages).
    func_name : str
        The smooth function name (for error messages).

    Returns
    -------
    Any
        The evaluated value.
    """
    # Normalize R's literal tokens (TRUE/FALSE/NA), which parse as ast.Name and
    # would otherwise fail ast.literal_eval with a misleading "Cannot evaluate
    # argument" error. A user porting an mgcv formula naturally writes
    # `s(x, fx=FALSE)` (the default, identical to plain s(x)); normalizing here
    # lets the fx handler accept FALSE / raise NotImplementedError on TRUE, and
    # sends bs=TRUE to the proper "must be a string" error.
    if isinstance(node, ast.Name):
        if node.id == "TRUE":
            return True
        if node.id == "FALSE":
            return False
        if node.id == "NA":
            return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        raise ValueError(
            f"Cannot evaluate argument '{key}' in {func_name}() at parse "
            f"time. Only literal values (integers, strings, floats) are "
            f"supported. Got: {_unparse(node)}"
        ) from None


# Left-associative binary operators of equal precedence: a left-nested run of
# them is printed and evaluated as one flat chain. ``**`` is right-associative
# and is never flattened.
_PRECEDENCE_GROUPS: tuple[tuple[type[ast.operator], ...], ...] = (
    (ast.Add, ast.Sub),
    (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.MatMult),
    (ast.LShift, ast.RShift),
    (ast.BitAnd,),
    (ast.BitXor,),
    (ast.BitOr,),
)


def _additive_chain(node: ast.BinOp) -> tuple[ast.expr, list[ast.BinOp]]:
    """Flatten a left-nested ``+``/``-`` chain without recursion.

    Returns the leftmost operand and the chain's ``BinOp`` links in source
    order; each link's ``right`` is the term it adds or subtracts.
    """
    return _operator_chain(node, (ast.Add, ast.Sub))


def _operator_chain(
    node: ast.BinOp, group: tuple[type[ast.operator], ...]
) -> tuple[ast.expr, list[ast.BinOp]]:
    """Flatten the left-nested run of ``group`` operators rooted at ``node``."""
    links: list[ast.BinOp] = []
    current: ast.expr = node
    while isinstance(current, ast.BinOp) and isinstance(current.op, group):
        links.append(current)
        current = current.left
    links.reverse()
    return current, links


def _precedence_group(op: ast.operator) -> tuple[type[ast.operator], ...]:
    """The left-associative group of ``op``; empty for ``**``."""
    for group in _PRECEDENCE_GROUPS:
        if isinstance(op, group):
            return group
    return ()


# Placeholder operand that lets ``ast.unparse`` spell one operator and decide
# its parentheses without rendering the operand it stands for.
_SLOT = "_jaxgam_formula_slot_"


def _unparse(node: ast.expr) -> str:
    """``ast.unparse(node)`` without recursion along operator chains.

    ``ast.unparse`` recurses once per nested operator and overflows the
    interpreter stack on a formula of a few hundred terms. This renderer walks
    left-nested chains of equal-precedence operators iteratively (``+``/``-``
    terms, or a malformed ``*`` run) and asks ``ast.unparse`` about one
    operator at a time, with each binary operand replaced by a shallow
    stand-in that has the same operator and therefore the same precedence, so
    the operator spelling and parenthesization are ``ast.unparse``'s own.
    """
    if isinstance(node, ast.BinOp):
        group = _precedence_group(node.op)
        if group:
            leftmost, links = _operator_chain(node, group)
        else:
            leftmost, links = node.left, [node]
        text = _operand_text(leftmost, links[0].op, right=False)
        for link in links:
            text += _operand_text(link.right, link.op, right=True)
        return text
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.BinOp):
        probe = ast.unparse(ast.UnaryOp(op=node.op, operand=_stand_in(node.operand)))
        return _substitute(probe, node.operand, at_end=True)
    return ast.unparse(node)


def _stand_in(node: ast.BinOp) -> ast.BinOp:
    """A one-operator node with the same operator, hence precedence, as ``node``."""
    return ast.BinOp(left=ast.Name(id="a"), op=node.op, right=ast.Name(id="b"))


def _substitute(probe: str, operand: ast.BinOp, *, at_end: bool) -> str:
    """Replace the rendered stand-in at one end of ``probe`` by ``operand``."""
    stand_in = ast.unparse(_stand_in(operand))
    inner = _unparse(operand)
    for rendered, replacement in ((f"({stand_in})", f"({inner})"), (stand_in, inner)):
        if at_end and probe.endswith(rendered):
            return probe[: -len(rendered)] + replacement
        if not at_end and probe.startswith(rendered):
            return replacement + probe[len(rendered) :]
    raise AssertionError(f"Unexpected rendering of a formula operand: {probe!r}")


def _operand_text(operand: ast.expr, op: ast.operator, *, right: bool) -> str:
    """Render one operand of ``op`` exactly as ``ast.unparse`` places it.

    A left operand is returned alone; a right operand is returned with the
    operator text before it, so a chain's text is its leftmost operand
    followed by each link's right-hand piece.
    """
    slot = ast.Name(id=_SLOT)
    stand_in = _stand_in(operand) if isinstance(operand, ast.BinOp) else operand
    if right:
        piece = ast.unparse(ast.BinOp(left=slot, op=op, right=stand_in))[len(_SLOT) :]
    else:
        probe = ast.unparse(ast.BinOp(left=stand_in, op=op, right=slot))
        operator = ast.unparse(ast.BinOp(left=slot, op=op, right=slot))[len(_SLOT) :]
        piece = probe[: -len(operator)]
    if isinstance(operand, ast.BinOp):
        return _substitute(piece, operand, at_end=right)
    return piece
