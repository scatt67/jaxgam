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

from pymgcv.formula.terms import FormulaSpec, ParametricTerm, SmoothSpec

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
    """Walks an AST expression tree and collects formula terms."""

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
        """Handle ``+`` and ``-`` operators connecting terms."""
        if isinstance(node.op, ast.Add):
            self.visit(node.left)
            self.visit(node.right)
        elif isinstance(node.op, ast.Sub):
            # ``expr - 1`` removes the intercept
            self.visit(node.left)
            if isinstance(node.right, ast.Constant) and node.right.value == 1:
                self.has_intercept = False
            else:
                raise ValueError(
                    "Subtraction in formula RHS is only supported as "
                    "'- 1' to remove the intercept. "
                    f"Got: {ast.unparse(node)}"
                )
        else:
            raise ValueError(
                f"Unsupported operator in formula: {type(node.op).__name__}. "
                f"Only '+' and '-' are supported. Got: {ast.unparse(node)}"
            )

    def _visit_call(self, node: ast.Call) -> None:
        """Handle function calls: s(), te(), ti()."""
        func_name = self._get_func_name(node)
        if func_name is None:
            raise ValueError(
                f"Cannot resolve function name in formula: {ast.unparse(node)}"
            )

        if func_name not in _SMOOTH_FUNCTIONS:
            raise ValueError(
                f"Unknown function '{func_name}()' in formula. "
                f"Supported: {sorted(_SMOOTH_FUNCTIONS)}"
            )

        if not node.args:
            raise ValueError(
                f"Smooth function '{func_name}()' requires at least one "
                f"variable argument. Got: {ast.unparse(node)}"
            )

        spec = _parse_smooth_call(func_name, node)
        self.smooth_terms.append(spec)

    def _visit_name(self, node: ast.Name) -> None:
        """Handle plain variable names as parametric terms."""
        self.parametric_terms.append(ParametricTerm(name=node.id))

    def _visit_constant(self, node: ast.Constant) -> None:
        """Handle constants: 0 removes intercept, 1 is explicit intercept."""
        if node.value == 0:
            self.has_intercept = False
        elif node.value == 1:
            # Explicit intercept — default behavior, nothing to do
            pass
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
                f"Unsupported unary expression in formula: {ast.unparse(node)}"
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
                f"names. Got: {ast.unparse(arg)}"
            )

    # Keyword arguments
    bs = _DEFAULT_BASIS[func_name]
    k = -1
    by: str | None = None
    extra_args: dict = {}

    for kw in node.keywords:
        key = kw.arg
        if key == "k":
            k = _eval_kwarg_value(kw.value, key, func_name)
            if not isinstance(k, int):
                raise ValueError(
                    f"Argument 'k' in {func_name}() must be an integer. "
                    f"Got: {ast.unparse(kw.value)}"
                )
        elif key == "bs":
            bs = _eval_kwarg_value(kw.value, key, func_name)
            if not isinstance(bs, str):
                raise ValueError(
                    f"Argument 'bs' in {func_name}() must be a string. "
                    f"Got: {ast.unparse(kw.value)}"
                )
        elif key == "by":
            if isinstance(kw.value, ast.Name):
                by = kw.value.id
            elif isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                by = kw.value.value
            else:
                raise ValueError(
                    f"Argument 'by' in {func_name}() must be a variable "
                    f"name or string. Got: {ast.unparse(kw.value)}"
                )
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


def _eval_kwarg_value(node: ast.expr, key: str, func_name: str) -> object:
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
    object
        The evaluated value.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        raise ValueError(
            f"Cannot evaluate argument '{key}' in {func_name}() at parse "
            f"time. Only literal values (integers, strings, floats) are "
            f"supported. Got: {ast.unparse(node)}"
        ) from None
