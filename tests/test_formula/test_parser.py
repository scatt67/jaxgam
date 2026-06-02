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

import pytest

from jaxgam.formula import FormulaSpec, SmoothSpec, parse_formula
from tests.helpers import _AssertCollector, check_that


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

    def test_whitespace_handling(self) -> None:
        """Extra whitespace around ~ is handled correctly."""
        result = parse_formula("  y  ~  s(x1)  ")
        assert result.response == "y"
        assert len(result.smooth_terms) == 1
        assert result.smooth_terms[0].variables == ["x1"]


class TestMultipleSmooths:
    """Test 2: multiple smooth terms."""

    @pytest.mark.parametrize("n_smooths", [2, 3])
    def test_multiple_smooths(self, n_smooths: int) -> None:
        """Multiple smooth terms are parsed in order."""
        terms = [f"s(x{i})" for i in range(1, n_smooths + 1)]
        result = parse_formula("y ~ " + " + ".join(terms))

        assert len(result.smooth_terms) == n_smooths
        for i, smooth in enumerate(result.smooth_terms, start=1):
            assert smooth.variables == [f"x{i}"]


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


class TestTensorProducts:
    """Test 5: tensor product smooths."""

    @pytest.mark.parametrize(
        ("formula", "variables"),
        [("y ~ te(x1, x2)", ["x1", "x2"]), ("y ~ te(x1, x2, x3)", ["x1", "x2", "x3"])],
        ids=["two_variables", "three_variables"],
    )
    def test_te_variables(self, formula: str, variables: list[str]) -> None:
        """te() produces SmoothSpec with the supplied variables."""
        result = parse_formula(formula)

        assert len(result.smooth_terms) == 1
        smooth = result.smooth_terms[0]
        assert smooth.variables == variables
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


class TestGaussianProcessParsing:
    """Gaussian process smooth parser surface."""

    def test_gp_kwargs_parse(self) -> None:
        """GP kwargs are Python literals carried through to SmoothSpec."""
        collector = _AssertCollector()

        def smooth_for(term: str) -> SmoothSpec:
            result = parse_formula(f"y ~ {term}")
            assert len(result.smooth_terms) == 1
            return result.smooth_terms[0]

        def default_gp() -> None:
            smooth = smooth_for('s(x, bs="gp")')
            assert smooth.variables == ["x"]
            assert smooth.bs == "gp"
            assert smooth.k == -1
            assert smooth.smooth_type == "s"
            assert smooth.extra_args == {}

        def direct_multivariate_gp() -> None:
            smooth = smooth_for('s(x, z, bs="gp", k=50)')
            assert smooth.variables == ["x", "z"]
            assert smooth.bs == "gp"
            assert smooth.k == 50
            assert smooth.smooth_type == "s"

        def tensor_product_gp() -> None:
            smooth = smooth_for('te(x1, x2, bs="gp", k=5)')
            assert smooth.variables == ["x1", "x2"]
            assert smooth.bs == "gp"
            assert smooth.k == 5
            assert smooth.smooth_type == "te"

        def tensor_interaction_gp() -> None:
            smooth = smooth_for('ti(x1, x2, bs="gp", k=5)')
            assert smooth.variables == ["x1", "x2"]
            assert smooth.bs == "gp"
            assert smooth.k == 5
            assert smooth.smooth_type == "ti"

        def kernel_kwarg() -> None:
            smooth = smooth_for('s(x, bs="gp", kernel="matern_3_2")')
            assert smooth.extra_args["kernel"] == "matern_3_2"

        def power_exponential_kwargs() -> None:
            smooth = smooth_for(
                's(x, bs="gp", kernel="power_exponential", rho=0.5, power=2.0)'
            )
            assert smooth.extra_args["kernel"] == "power_exponential"
            assert smooth.extra_args["rho"] == 0.5
            assert smooth.extra_args["power"] == 2.0

        def stationary_kwarg() -> None:
            smooth = smooth_for('s(x, bs="gp", stationary=True)')
            assert smooth.extra_args["stationary"] is True

        def xt_kwarg() -> None:
            smooth = smooth_for('s(x, bs="gp", xt={"max_knots": 500, "seed": 42})')
            assert smooth.extra_args["xt"] == {"max_knots": 500, "seed": 42}

        def m_parses_but_is_not_interpreted() -> None:
            smooth = smooth_for('s(x, bs="gp", m=[3, 0.5])')
            assert smooth.extra_args["m"] == [3, 0.5]

        collector.check('s(x, bs="gp")', default_gp)
        collector.check('s(x, z, bs="gp", k=50)', direct_multivariate_gp)
        collector.check('te(x1, x2, bs="gp", k=5)', tensor_product_gp)
        collector.check('ti(x1, x2, bs="gp", k=5)', tensor_interaction_gp)
        collector.check("kernel kwarg", kernel_kwarg)
        collector.check("power-exponential kwargs", power_exponential_kwargs)
        collector.check("stationary kwarg", stationary_kwarg)
        collector.check("xt kwarg", xt_kwarg)
        collector.check(
            "m parses without GP semantics",
            m_parses_but_is_not_interpreted,
        )
        collector.raise_if_any("gp parser kwargs")

    def test_gp_r_style_call_rejected(self) -> None:
        """R-style c(...) stays outside the current parser surface."""
        with pytest.raises(ValueError, match="Cannot evaluate argument 'kernel'"):
            parse_formula('y ~ s(x, bs="gp", kernel=c("matern_3_2"))')


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

    def test_intercept_token_order_last_wins(self) -> None:
        """Intercept is last-token-wins, left-to-right, matching R terms().

        Regression for Finding 1: ``+ 1`` after intercept removal re-adds the
        intercept. These are exact R ``terms.formula`` outputs (analytic ground
        truth, verified against R 4.5.2), so no tolerance is needed.
        """
        cases = {
            "y ~ 0 + x + 1": True,
            "y ~ x - 1 + 1": True,
            "y ~ x + 0 + 1": True,
            "y ~ -1 + x + 1": True,
            "y ~ 1 + x - 1": False,
            "y ~ x + 1 - 1": False,
            "y ~ x - 1 + 1 - 1": False,
            "y ~ 1 - 1 + x": False,
            "y ~ x + 0": False,
            "y ~ x": True,
        }
        collector = _AssertCollector()
        for formula, expected in cases.items():
            collector.check(
                formula,
                lambda f=formula, e=expected: check_that(
                    parse_formula(f).has_intercept is e,
                    f"{f}: has_intercept should be {e}",
                ),
            )
        collector.raise_if_any("intercept token order")


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


class TestDeferredAndDuplicateArgs:
    """Findings 4, 5, 13: deferred smooth kwargs and duplicate-term de-dup."""

    def test_fx_true_raises_not_implemented(self) -> None:
        """s(..., fx=True) is a deferred feature; raise rather than silently fit."""
        with pytest.raises(NotImplementedError, match="fx=True"):
            parse_formula("y ~ s(x1, fx=True)")

    def test_fx_false_accepted_and_not_stored(self) -> None:
        """fx=False is the default; accept it and do not leak it into extra_args."""
        result = parse_formula("y ~ s(x1, fx=False)")
        assert len(result.smooth_terms) == 1
        assert "fx" not in result.smooth_terms[0].extra_args

    def test_r_spelled_booleans_normalized(self) -> None:
        """Finding L3: R's TRUE/FALSE tokens are accepted in DSL kwargs.

        A user porting an mgcv formula writes `s(x, fx=FALSE)` (the default,
        equivalent to plain s(x)); R booleans parse as ast.Name and previously
        hit the misleading "Cannot evaluate argument 'fx'" error before the fx
        handler ran. fx=TRUE must still reach NotImplementedError, and bs=TRUE
        the "must be a string" error.
        """
        collector = _AssertCollector()

        def _fx_false_equals_plain() -> None:
            plain = parse_formula("y ~ s(x1)").smooth_terms[0]
            r_false = parse_formula("y ~ s(x1, fx=FALSE)")
            sm = r_false.smooth_terms[0]
            check_that(
                len(r_false.smooth_terms) == 1
                and sm.bs == plain.bs
                and sm.variables == plain.variables
                and "fx" not in sm.extra_args,
                "s(x, fx=FALSE) must parse equivalently to plain s(x)",
            )

        collector.check("fx_FALSE_equals_plain", _fx_false_equals_plain)

        def _fx_true_not_implemented() -> None:
            with pytest.raises(NotImplementedError, match="fx=True"):
                parse_formula("y ~ s(x1, fx=TRUE)")

        collector.check("fx_TRUE_not_implemented", _fx_true_not_implemented)

        def _bs_true_must_be_string() -> None:
            with pytest.raises(ValueError, match="must be a string"):
                parse_formula("y ~ s(x1, bs=TRUE)")

        collector.check("bs_TRUE_must_be_string", _bs_true_must_be_string)
        collector.raise_if_any("R-spelled boolean normalization (L3)")

    def test_identical_smooths_dedup(self) -> None:
        """Finding S3: identical s(x)+s(x) collapses to one smooth (R terms.formula);
        a different-config repeat is kept."""
        assert len(parse_formula("y ~ s(x)").smooth_terms) == 1
        assert len(parse_formula("y ~ s(x) + s(x)").smooth_terms) == 1
        # Different config (k differs) is NOT identical -> both kept.
        assert len(parse_formula("y ~ s(x, k=6) + s(x, k=8)").smooth_terms) == 2

    def test_per_term_sp_raises_not_implemented(self) -> None:
        """Per-term sp= inside s()/te()/ti() is deferred; raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Per-term sp"):
            parse_formula("y ~ s(x1, sp=0.1)")

    def test_per_term_sp_none_accepted(self) -> None:
        """sp=None is the R default; accept and do not store it."""
        result = parse_formula("y ~ s(x1, sp=None)")
        assert "sp" not in result.smooth_terms[0].extra_args

    def test_duplicate_parametric_dedup(self) -> None:
        """y ~ x + x collapses to a single parametric term (matches R)."""
        result = parse_formula("y ~ x + x")
        assert [t.name for t in result.parametric_terms] == ["x"]

        ordered = parse_formula("y ~ x1 + x2 + x1")
        assert [t.name for t in ordered.parametric_terms] == ["x1", "x2"]
