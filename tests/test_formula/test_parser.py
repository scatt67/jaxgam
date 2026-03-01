"""Tests for the AST-based formula parser.

Covers:
1. Basic parsing
2. Multiple smooths
3. Keyword arguments (k, bs)
4. By-variable
5. Tensor products (te, ti)
6. Mixed smooth + parametric terms
7. No-intercept formulas
8. Complex multi-term formulas
9. Error cases
10. No JAX imports (Phase 1 boundary)
"""

import sys

import pytest

from jaxgam.formula import FormulaSpec, ParametricTerm, SmoothSpec, parse_formula


class TestBasicParsing:
    """Test 1: basic formula parsing."""

    def test_basic_smooth(self) -> None:
        """y ~ s(x1) produces FormulaSpec with one SmoothSpec."""
        result = parse_formula("y ~ s(x1)")

        assert isinstance(result, FormulaSpec)
        assert result.response == "y"
        assert len(result.smooth_terms) == 1
        assert len(result.parametric_terms) == 0
        assert result.has_intercept is True

        smooth = result.smooth_terms[0]
        assert isinstance(smooth, SmoothSpec)
        assert smooth.variables == ["x1"]
        assert smooth.smooth_type == "s"
        assert smooth.bs == "tp"
        assert smooth.k == -1
        assert smooth.by is None
        assert smooth.extra_args == {}

    def test_purely_parametric(self) -> None:
        """y ~ x1 produces one parametric term, no smooths."""
        result = parse_formula("y ~ x1")

        assert result.response == "y"
        assert len(result.smooth_terms) == 0
        assert len(result.parametric_terms) == 1
        assert result.parametric_terms[0].name == "x1"
        assert result.has_intercept is True

    def test_response_extraction(self) -> None:
        """Response variable is correctly extracted."""
        result = parse_formula("response_var ~ s(x)")
        assert result.response == "response_var"

    def test_whitespace_handling(self) -> None:
        """Extra whitespace around ~ is handled correctly."""
        result = parse_formula("  y  ~  s(x1)  ")
        assert result.response == "y"
        assert len(result.smooth_terms) == 1
        assert result.smooth_terms[0].variables == ["x1"]


class TestMultipleSmooths:
    """Test 2: multiple smooth terms."""

    def test_two_smooths(self) -> None:
        """y ~ s(x1) + s(x2) produces two SmoothSpecs."""
        result = parse_formula("y ~ s(x1) + s(x2)")

        assert len(result.smooth_terms) == 2
        assert result.smooth_terms[0].variables == ["x1"]
        assert result.smooth_terms[1].variables == ["x2"]

    def test_three_smooths(self) -> None:
        """Three smooth terms parsed correctly."""
        result = parse_formula("y ~ s(x1) + s(x2) + s(x3)")

        assert len(result.smooth_terms) == 3
        assert result.smooth_terms[0].variables == ["x1"]
        assert result.smooth_terms[1].variables == ["x2"]
        assert result.smooth_terms[2].variables == ["x3"]


class TestKwargs:
    """Test 3: keyword arguments (k, bs)."""

    def test_k_argument(self) -> None:
        """y ~ s(x1, k=20) has k=20."""
        result = parse_formula("y ~ s(x1, k=20)")

        smooth = result.smooth_terms[0]
        assert smooth.k == 20

    def test_bs_argument(self) -> None:
        """y ~ s(x1, bs='cr') has bs='cr'."""
        result = parse_formula('y ~ s(x1, bs="cr")')

        smooth = result.smooth_terms[0]
        assert smooth.bs == "cr"

    def test_multiple_kwargs(self) -> None:
        """y ~ s(x1, k=20, bs='cr') has both k and bs set."""
        result = parse_formula('y ~ s(x1, k=20, bs="cr")')

        smooth = result.smooth_terms[0]
        assert smooth.k == 20
        assert smooth.bs == "cr"

    def test_default_k(self) -> None:
        """Default k is -1 when not specified."""
        result = parse_formula("y ~ s(x1)")
        assert result.smooth_terms[0].k == -1

    def test_default_bs(self) -> None:
        """Default bs is 'tp' for s()."""
        result = parse_formula("y ~ s(x1)")
        assert result.smooth_terms[0].bs == "tp"

    def test_extra_kwargs(self) -> None:
        """Extra kwargs are captured in extra_args."""
        result = parse_formula('y ~ s(x1, m=2, xt="cs")')

        smooth = result.smooth_terms[0]
        assert smooth.extra_args == {"m": 2, "xt": "cs"}


class TestByVariable:
    """Test 4: by-variable."""

    def test_by_as_name(self) -> None:
        """y ~ s(x1, by=fac) has by='fac' (unquoted name)."""
        result = parse_formula("y ~ s(x1, by=fac)")

        smooth = result.smooth_terms[0]
        assert smooth.by == "fac"

    def test_by_as_string(self) -> None:
        """y ~ s(x1, by='fac') has by='fac' (quoted string)."""
        result = parse_formula('y ~ s(x1, by="fac")')

        smooth = result.smooth_terms[0]
        assert smooth.by == "fac"

    def test_by_with_other_kwargs(self) -> None:
        """by-variable works alongside k and bs."""
        result = parse_formula('y ~ s(x1, k=15, bs="cr", by=group)')

        smooth = result.smooth_terms[0]
        assert smooth.by == "group"
        assert smooth.k == 15
        assert smooth.bs == "cr"


class TestTensorProducts:
    """Test 5: tensor product smooths."""

    def test_te_basic(self) -> None:
        """te(x1, x2) produces SmoothSpec with two variables."""
        result = parse_formula("y ~ te(x1, x2)")

        assert len(result.smooth_terms) == 1
        smooth = result.smooth_terms[0]
        assert smooth.variables == ["x1", "x2"]
        assert smooth.smooth_type == "te"
        assert smooth.bs == "cr"

    def test_ti_basic(self) -> None:
        """ti(x1, x2) produces SmoothSpec with smooth_type='ti'."""
        result = parse_formula("y ~ ti(x1, x2)")

        assert len(result.smooth_terms) == 1
        smooth = result.smooth_terms[0]
        assert smooth.variables == ["x1", "x2"]
        assert smooth.smooth_type == "ti"

    def test_te_with_kwargs(self) -> None:
        """te() supports keyword arguments."""
        result = parse_formula("y ~ te(x1, x2, k=10)")

        smooth = result.smooth_terms[0]
        assert smooth.variables == ["x1", "x2"]
        assert smooth.k == 10

    def test_te_three_variables(self) -> None:
        """te() with three variables."""
        result = parse_formula("y ~ te(x1, x2, x3)")

        smooth = result.smooth_terms[0]
        assert smooth.variables == ["x1", "x2", "x3"]

    def test_default_bs_tensor(self) -> None:
        """Default bs is 'cr' for te() and ti(), matching R."""
        result_te = parse_formula("y ~ te(x1, x2)")
        assert result_te.smooth_terms[0].bs == "cr"
        result_ti = parse_formula("y ~ ti(x1, x2)")
        assert result_ti.smooth_terms[0].bs == "cr"

    def test_smooth_plus_interaction(self) -> None:
        """y ~ s(x1) + s(x2) + ti(x1, x2) parses all three terms."""
        result = parse_formula("y ~ s(x1) + s(x2) + ti(x1, x2)")

        assert len(result.smooth_terms) == 3
        assert result.smooth_terms[0].smooth_type == "s"
        assert result.smooth_terms[0].variables == ["x1"]
        assert result.smooth_terms[1].smooth_type == "s"
        assert result.smooth_terms[1].variables == ["x2"]
        assert result.smooth_terms[2].smooth_type == "ti"
        assert result.smooth_terms[2].variables == ["x1", "x2"]


class TestMixedTerms:
    """Test 6: mixed smooth and parametric terms."""

    def test_smooth_plus_parametric(self) -> None:
        """y ~ s(x1) + x2 produces one smooth and one parametric."""
        result = parse_formula("y ~ s(x1) + x2")

        assert len(result.smooth_terms) == 1
        assert len(result.parametric_terms) == 1
        assert result.smooth_terms[0].variables == ["x1"]
        assert result.parametric_terms[0].name == "x2"

    def test_smooth_plus_multiple_parametric(self) -> None:
        """y ~ s(x1) + x2 + x3 produces one smooth and two parametric."""
        result = parse_formula("y ~ s(x1) + x2 + x3")

        assert len(result.smooth_terms) == 1
        assert len(result.parametric_terms) == 2
        assert result.parametric_terms[0].name == "x2"
        assert result.parametric_terms[1].name == "x3"

    def test_multiple_smooth_and_parametric(self) -> None:
        """Multiple smooths and parametric terms together."""
        result = parse_formula("y ~ s(x1) + x2 + s(x3) + x4")

        assert len(result.smooth_terms) == 2
        assert len(result.parametric_terms) == 2


class TestNoIntercept:
    """Test 7: no-intercept formulas."""

    def test_zero_plus(self) -> None:
        """y ~ 0 + s(x1) has has_intercept=False."""
        result = parse_formula("y ~ 0 + s(x1)")

        assert result.has_intercept is False
        assert len(result.smooth_terms) == 1

    def test_minus_one(self) -> None:
        """y ~ s(x1) - 1 has has_intercept=False."""
        result = parse_formula("y ~ s(x1) - 1")

        assert result.has_intercept is False
        assert len(result.smooth_terms) == 1

    def test_default_intercept(self) -> None:
        """Default formula has has_intercept=True."""
        result = parse_formula("y ~ s(x1)")
        assert result.has_intercept is True

    def test_explicit_intercept(self) -> None:
        """y ~ 1 + s(x1) has has_intercept=True."""
        result = parse_formula("y ~ 1 + s(x1)")
        assert result.has_intercept is True

    def test_zero_plus_parametric(self) -> None:
        """y ~ 0 + x1 removes intercept with parametric terms."""
        result = parse_formula("y ~ 0 + x1")

        assert result.has_intercept is False
        assert len(result.parametric_terms) == 1
        assert result.parametric_terms[0].name == "x1"


class TestComplexFormula:
    """Test 8: complex multi-term formulas."""

    def test_complex_formula(self) -> None:
        """Complex formula with multiple smooth types and parametric terms."""
        result = parse_formula('y ~ s(x1) + s(x2, k=15, bs="cr") + te(x1, x2) + x3')

        assert result.response == "y"
        assert result.has_intercept is True
        assert len(result.smooth_terms) == 3
        assert len(result.parametric_terms) == 1

        # First smooth: s(x1)
        s1 = result.smooth_terms[0]
        assert s1.smooth_type == "s"
        assert s1.variables == ["x1"]
        assert s1.bs == "tp"
        assert s1.k == -1

        # Second smooth: s(x2, k=15, bs="cr")
        s2 = result.smooth_terms[1]
        assert s2.smooth_type == "s"
        assert s2.variables == ["x2"]
        assert s2.bs == "cr"
        assert s2.k == 15

        # Third smooth: te(x1, x2)
        s3 = result.smooth_terms[2]
        assert s3.smooth_type == "te"
        assert s3.variables == ["x1", "x2"]

        # Parametric: x3
        assert result.parametric_terms[0].name == "x3"

    def test_complex_with_by_and_interaction(self) -> None:
        """Formula with by-variable, interaction, and parametric."""
        result = parse_formula("y ~ s(x1, by=fac) + s(x2) + ti(x1, x2) + x3")

        assert len(result.smooth_terms) == 3
        assert result.smooth_terms[0].by == "fac"
        assert result.smooth_terms[2].smooth_type == "ti"
        assert len(result.parametric_terms) == 1


class TestErrorCases:
    """Test 9: malformed formulas raise informative errors."""

    def test_missing_tilde(self) -> None:
        """Formula without ~ raises ValueError."""
        with pytest.raises(ValueError, match="must contain '~'"):
            parse_formula("y + s(x1)")

    def test_empty_response(self) -> None:
        """Empty LHS raises ValueError."""
        with pytest.raises(ValueError, match="empty response"):
            parse_formula("~ s(x1)")

    def test_empty_rhs(self) -> None:
        """Empty RHS raises ValueError."""
        with pytest.raises(ValueError, match="empty right-hand side"):
            parse_formula("y ~")

    def test_empty_rhs_whitespace(self) -> None:
        """Whitespace-only RHS raises ValueError."""
        with pytest.raises(ValueError, match="empty right-hand side"):
            parse_formula("y ~   ")

    def test_unknown_function(self) -> None:
        """Unknown smooth function raises ValueError."""
        with pytest.raises(ValueError, match="Unknown function 'foo\\(\\)'"):
            parse_formula("y ~ foo(x1)")

    def test_smooth_no_args(self) -> None:
        """s() with no arguments raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one"):
            parse_formula("y ~ s()")

    def test_non_literal_k(self) -> None:
        """k=int(np.log(n)) raises ValueError (not a literal)."""
        with pytest.raises(ValueError, match="Cannot evaluate argument 'k'"):
            parse_formula("y ~ s(x1, k=int(np.log(n)))")

    def test_invalid_syntax(self) -> None:
        """Completely invalid RHS raises ValueError."""
        with pytest.raises(ValueError, match="Cannot parse formula RHS"):
            parse_formula("y ~ @@@")

    def test_positional_arg_not_name(self) -> None:
        """Non-name positional arg (e.g. s(1)) raises ValueError."""
        with pytest.raises(ValueError, match="must be variable names"):
            parse_formula("y ~ s(1)")

    def test_unsupported_operator(self) -> None:
        """Multiplication operator raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            parse_formula("y ~ s(x1) * s(x2)")

    def test_subtraction_non_one(self) -> None:
        """Subtraction of non-1 value raises ValueError."""
        with pytest.raises(ValueError, match="only supported as"):
            parse_formula("y ~ s(x1) - 2")


class TestNoJaxImport:
    """Test 10: Phase 1 boundary -- no JAX imports."""

    def test_no_jax_in_modules(self) -> None:
        """Importing jaxgam.formula does not trigger jax import."""
        import importlib

        # Force reimport by removing cached formula modules
        formula_modules = [
            key for key in list(sys.modules) if key.startswith("jaxgam.formula")
        ]
        saved = {}
        for mod in formula_modules:
            saved[mod] = sys.modules.pop(mod)

        try:
            # Also remove jax if it was loaded by other test modules
            jax_mods_to_remove = [
                key
                for key in list(sys.modules)
                if key == "jax" or key.startswith("jax.")
            ]
            saved_jax = {}
            for mod in jax_mods_to_remove:
                saved_jax[mod] = sys.modules.pop(mod)

            importlib.import_module("jaxgam.formula")

            jax_modules_after = {
                key for key in sys.modules if key == "jax" or key.startswith("jax.")
            }
            assert not jax_modules_after, (
                f"Importing jaxgam.formula triggered JAX imports: {jax_modules_after}"
            )
        finally:
            # Restore modules to avoid breaking other tests
            sys.modules.update(saved)
            sys.modules.update(saved_jax)


class TestDataclassProperties:
    """Additional tests for dataclass behavior."""

    def test_smooth_spec_defaults(self) -> None:
        """SmoothSpec has correct defaults."""
        spec = SmoothSpec(variables=["x1"])
        assert spec.bs == "tp"
        assert spec.k == -1
        assert spec.by is None
        assert spec.smooth_type == "s"
        assert spec.extra_args == {}

    def test_smooth_spec_custom(self) -> None:
        """SmoothSpec accepts custom values."""
        spec = SmoothSpec(
            variables=["x1", "x2"],
            bs="cr",
            k=20,
            by="group",
            smooth_type="te",
            extra_args={"m": 2},
        )
        assert spec.variables == ["x1", "x2"]
        assert spec.bs == "cr"
        assert spec.k == 20
        assert spec.by == "group"
        assert spec.smooth_type == "te"
        assert spec.extra_args == {"m": 2}

    def test_parametric_term(self) -> None:
        """ParametricTerm stores name correctly."""
        term = ParametricTerm(name="x1")
        assert term.name == "x1"

    def test_formula_spec_defaults(self) -> None:
        """FormulaSpec has correct defaults."""
        spec = FormulaSpec(response="y")
        assert spec.response == "y"
        assert spec.smooth_terms == []
        assert spec.parametric_terms == []
        assert spec.has_intercept is True
