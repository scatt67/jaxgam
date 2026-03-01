"""Edge case and robustness tests for jaxgam.

Tests cover pathological inputs and boundary conditions that should
either succeed gracefully or produce clear, actionable error messages.
No silent NaN propagation or unhandled exceptions.

Test cases:
- A. Near-separation in Binomial (step-halving saves it)
- B. All-zero response in Poisson
- C. k > n (basis dimension exceeds sample size)
- D. Single-observation data
- E. Constant covariate
- F. Missing values (NaN) in data
- G. Factor-by with only 1 level
- H. Factor-by with empty level (no observations for some level)
- I. Very large k (k=500)
- J. Lambda at boundary (log_lambda = +/-20)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.api import GAM

SEED = 123


# ---------------------------------------------------------------------------
# A. Near-separation in Binomial
# ---------------------------------------------------------------------------


class TestNearSeparation:
    """Near-separation: y nearly perfectly predicted by x.

    Step-halving in PIRLS should keep coefficients from diverging.
    The model should either converge or return a non-NaN result.
    """

    def test_near_separation_no_nan(self):
        """Near-separation produces finite results, no NaN."""
        rng = np.random.default_rng(SEED)
        n = 200
        x = np.linspace(0, 1, n)
        # Near-perfect separation: prob is 0 or 1 except near x=0.5
        eta = 20.0 * (x - 0.5)
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = (prob > 0.5).astype(float)
        # Add a few "noise" flips near the boundary
        boundary = np.abs(x - 0.5) < 0.05
        y[boundary] = rng.binomial(1, prob[boundary]).astype(float)

        data = pd.DataFrame({"x": x, "y": y})
        model = GAM("y ~ s(x, k=10, bs='cr')", family="binomial").fit(data)

        assert np.all(np.isfinite(model.coefficients_))
        assert np.all(np.isfinite(model.fitted_values_))
        assert np.isfinite(model.deviance_)
        assert np.isfinite(model.scale_)


# ---------------------------------------------------------------------------
# B. All-zero response in Poisson
# ---------------------------------------------------------------------------


class TestAllZeroPoisson:
    """All-zero response in Poisson.

    This is a valid edge case: all counts are zero. The Poisson
    initialization maps y=0 to mu=0.1 to avoid log(0). The model
    should fit with very high smoothing (penalized toward constant
    near-zero rate).
    """

    def test_all_zero_poisson_fits(self):
        """All-zero Poisson response: model fits with finite results."""
        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        y = np.zeros(n)
        data = pd.DataFrame({"x": x, "y": y})

        model = GAM("y ~ s(x, k=10, bs='cr')", family="poisson").fit(data)

        assert np.all(np.isfinite(model.coefficients_))
        assert np.all(np.isfinite(model.fitted_values_))
        assert np.all(model.fitted_values_ >= 0)
        assert np.isfinite(model.deviance_)


# ---------------------------------------------------------------------------
# C. k > n (basis dimension exceeds sample size)
# ---------------------------------------------------------------------------


class TestKGreaterThanN:
    """k > n: basis dimension exceeds sample size.

    Should raise a clear error since the unconstrained design matrix
    would have more columns than rows.
    """

    def test_k_greater_than_n_raises(self):
        """k > n raises ValueError with clear message."""
        n = 10
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x)
        data = pd.DataFrame({"x": x, "y": y})

        with pytest.raises(
            ValueError, match=r"k.*larger.*n|k.*exceed|basis.*dimension"
        ):
            GAM("y ~ s(x, k=20, bs='cr')").fit(data)


# ---------------------------------------------------------------------------
# D. Single-observation data
# ---------------------------------------------------------------------------


class TestSingleObservation:
    """Single observation: n=1.

    Not enough data to fit any smooth. Should raise a clear error.
    """

    def test_single_obs_raises(self):
        """n=1 raises ValueError with clear message."""
        data = pd.DataFrame({"x": [0.5], "y": [1.0]})

        with pytest.raises(ValueError, match=r"[Aa]t least|too few|observation"):
            GAM("y ~ s(x, k=10, bs='cr')").fit(data)


# ---------------------------------------------------------------------------
# E. Constant covariate
# ---------------------------------------------------------------------------


class TestConstantCovariate:
    """Constant covariate: all x values are the same.

    A smooth over a constant is meaningless. Should raise a clear error
    or produce a degenerate-but-finite fit.
    """

    def test_constant_covariate_raises(self):
        """Constant covariate raises ValueError."""
        n = 50
        x = np.ones(n) * 0.5
        y = np.random.default_rng(SEED).normal(0, 1, n)
        data = pd.DataFrame({"x": x, "y": y})

        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            GAM("y ~ s(x, k=10, bs='cr')").fit(data)


# ---------------------------------------------------------------------------
# F. Missing values (NaN) in data
# ---------------------------------------------------------------------------


class TestMissingValues:
    """Missing values: NaN in response, predictors, or both.

    Should raise a clear error, not silently propagate NaN.
    """

    def test_nan_in_response_raises(self):
        """NaN in response raises ValueError."""
        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        y[10] = np.nan
        data = pd.DataFrame({"x": x, "y": y})

        with pytest.raises(ValueError, match=r"[Nn]a[Nn]|missing|NA"):
            GAM("y ~ s(x, k=10, bs='cr')").fit(data)

    def test_nan_in_predictor_raises(self):
        """NaN in predictor raises ValueError."""
        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        x[5] = np.nan
        data = pd.DataFrame({"x": x, "y": y})

        with pytest.raises(ValueError, match=r"[Nn]a[Nn]|missing|NA"):
            GAM("y ~ s(x, k=10, bs='cr')").fit(data)

    def test_inf_in_response_raises(self):
        """Inf in response raises ValueError."""
        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        y[0] = np.inf
        data = pd.DataFrame({"x": x, "y": y})

        with pytest.raises(ValueError, match=r"non-finite|Inf"):
            GAM("y ~ s(x, k=10, bs='cr')").fit(data)


# ---------------------------------------------------------------------------
# G. Factor-by with only 1 level
# ---------------------------------------------------------------------------


class TestFactorByOneLevel:
    """Factor-by with a single level.

    Should work: equivalent to no factor-by interaction.
    """

    def test_single_level_fits(self):
        """Factor-by with 1 level fits and produces finite results."""
        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        fac = pd.Categorical(["a"] * n, categories=["a"])
        data = pd.DataFrame({"x": x, "fac": fac, "y": y})

        model = GAM("y ~ s(x, by=fac, k=10, bs='cr')", family="gaussian").fit(data)

        assert np.all(np.isfinite(model.coefficients_))
        assert np.all(np.isfinite(model.fitted_values_))
        assert model.converged_


# ---------------------------------------------------------------------------
# H. Factor-by with empty level
# ---------------------------------------------------------------------------


class TestFactorByEmptyLevel:
    """Factor-by where one level has zero observations.

    Should raise a clear error since the design matrix block for
    that level would be all zeros.
    """

    def test_empty_level_raises(self):
        """Factor-by with empty level raises ValueError."""
        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        # All observations are level "a", but category includes "b" and "c"
        fac = pd.Categorical(["a"] * n, categories=["a", "b", "c"])
        data = pd.DataFrame({"x": x, "fac": fac, "y": y})

        with pytest.raises(ValueError, match=r"empty|no observation|zero.*observation"):
            GAM("y ~ s(x, by=fac, k=10, bs='cr') + fac").fit(data)


# ---------------------------------------------------------------------------
# I. Very large k
# ---------------------------------------------------------------------------


class TestVeryLargeK:
    """Very large basis dimension (k=500).

    Should work but be slower. Check it fits without OOM and
    produces finite results.
    """

    @pytest.mark.slow
    def test_large_k_fits(self):
        """k=500 fits and produces finite results."""
        rng = np.random.default_rng(SEED)
        n = 1000
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x": x, "y": y})

        model = GAM("y ~ s(x, k=500, bs='cr')").fit(data)

        assert np.all(np.isfinite(model.coefficients_))
        assert np.all(np.isfinite(model.fitted_values_))
        assert np.isfinite(model.deviance_)
        assert model.coefficients_.shape[0] > 400  # intercept + ~499 basis functions


# ---------------------------------------------------------------------------
# J. Lambda at boundary
# ---------------------------------------------------------------------------


class TestLambdaAtBoundary:
    """Smoothing parameter at extreme values (log_lambda = +/-20).

    REML should still evaluate and gradient should be finite.
    """

    def test_very_large_lambda(self):
        """Very large lambda (heavy smoothing): model fits with finite results."""
        rng = np.random.default_rng(SEED)
        n = 200
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x": x, "y": y})

        # exp(20) ~ 4.8e8: extreme smoothing, nearly linear fit
        sp = [np.exp(20.0)]
        model = GAM("y ~ s(x, k=10, bs='cr')", sp=sp).fit(data)

        assert np.all(np.isfinite(model.coefficients_))
        assert np.all(np.isfinite(model.fitted_values_))
        assert np.isfinite(model.deviance_)

    def test_very_small_lambda(self):
        """Very small lambda (minimal smoothing): model fits with finite results."""
        rng = np.random.default_rng(SEED)
        n = 200
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        data = pd.DataFrame({"x": x, "y": y})

        # exp(-20) ~ 2e-9: almost no smoothing, interpolation-like
        sp = [np.exp(-20.0)]
        model = GAM("y ~ s(x, k=10, bs='cr')", sp=sp).fit(data)

        assert np.all(np.isfinite(model.coefficients_))
        assert np.all(np.isfinite(model.fitted_values_))
        assert np.isfinite(model.deviance_)
