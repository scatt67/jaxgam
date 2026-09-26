"""75-Cell Validation Matrix: systematic R comparison for all smooth x family combos.

Tests cover every cell in the v1.0 surface (design.md §1.2):
- 15 smooth configs (tp, cr, te, ti, tp_by, cr_by, te_by, re, re_slope,
  re_mixed, gp, gp_2d, gp_mixed, gp_te, gp_ti) x 5 families = 75 cells
- Families: gaussian, binomial, poisson, gamma, nb

Plus hard-gate invariants (§18.1) that must hold for all cells without R.

Tolerance rationale (from AGENTS.md §Common Pitfalls, MEMORY.md):
  Gaussian REML: MODERATE (rtol=1e-4, atol=1e-6).
  GLM families: LOOSE (rtol=1e-2, atol=1e-4).
  Tensor products / factor-by: LOOSE for all (flat REML surfaces).
  TPRS: compare fitted values not raw coefficients (sign ambiguity).
  GP: compare fitted values not raw coefficients (eigenvector/SVD ambiguity).
  Factor-by EDF: our architecture stores 1 combined entry vs R's per-level;
    compare total EDF sum.
  RE (re, re_slope): deterministic basis, single sp — direct coef comparison.
  RE mixed (re_mixed): contains TPRS sign ambiguity — compare fitted values.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import pytest
from jax import clear_caches

from jaxgam.api import GAM
from jaxgam.control import FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs import efs_initial_log_lambda, efs_initial_log_scale
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.fitting.newton import NewtonOptimizer
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from tests.helpers import SEED, _AssertCollector, check_that, r_available
from tests.r_bridge import RBridge, RBridgeError
from tests.tolerances import LOOSE, MODERATE, STRICT, ToleranceClass

# ---------------------------------------------------------------------------
# JAX cache teardown
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    """Clear JAX compilation caches after each test to prevent OOM.

    Each GAM fit JIT-compiles functions with shapes specific to the model
    (smooth type, basis size, family). Without clearing, the accumulated LLVM
    artifacts exhaust memory on GH Actions runners (7 GB RAM).
    """
    yield
    clear_caches()  # teardown


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_single_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Single-predictor data for s(x) models."""
    rng = np.random.default_rng(seed)
    n = 200 if family_name != "binomial" else 300
    x = rng.uniform(0, 1, n)

    if family_name == "gaussian":
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        eta = 2 * np.sin(2 * np.pi * x)
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        eta = np.sin(2 * np.pi * x) + 0.5
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family_name == "gamma":
        eta = 0.5 * np.sin(2 * np.pi * x) + 1.0
        mu = np.exp(eta)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        eta = np.sin(2 * np.pi * x) + 1.0
        mu = np.exp(eta)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "y": y})


def _make_two_smooth_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Two-predictor data for te/ti models, parametrized by family."""
    rng = np.random.default_rng(seed)
    n = 200 if family_name != "binomial" else 300
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)

    eta = np.sin(2 * np.pi * x1) + 0.5 * x2

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_gp_1d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """One-dimensional GP smooth data, parametrized by family."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, n)

    eta = np.sin(3 * np.pi * x) * 0.8 + np.cos(2 * np.pi * x) * 0.4

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "y": y})


def _make_gp_2d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Two-dimensional direct GP smooth data, parametrized by family."""
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)

    eta = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) + 0.3 * (x + z)

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "z": z, "y": y})


def _make_gp_1d_par_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """One-dimensional GP plus parametric linear term data."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, n)

    eta = 0.7 * x + np.sin(3 * np.pi * x) * 0.6

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "y": y})


def _make_gp_te_2d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Two-dimensional tensor GP smooth data, parametrized by family."""
    rng = np.random.default_rng(seed)
    n = 400
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)

    eta = (
        np.sin(2 * np.pi * x1)
        + np.cos(2 * np.pi * x2)
        + 0.5 * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
    )

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _make_factor_by_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Factor-by data for s(x, by=fac) models, parametrized by family."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, n)
    levels = ["a", "b", "c"]
    fac = rng.choice(levels, n)

    eta = np.where(
        fac == "a",
        np.sin(2 * np.pi * x),
        np.where(fac == "b", 0.5 * x, -0.3 * x),
    )

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-2 * eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame(
        {
            "x": x,
            "fac": pd.Categorical(fac, categories=levels),
            "y": y,
        }
    )


def _make_factor_by_2d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Factor-by data with 2D covariates for te(x1, x2, by=fac) models."""
    rng = np.random.default_rng(seed)
    n = 300
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    levels = ["a", "b", "c"]
    fac = rng.choice(levels, n)

    eta = np.where(
        fac == "a",
        np.sin(2 * np.pi * x1) + 0.5 * x2,
        np.where(fac == "b", 0.5 * x1 + x2, -0.3 * x1 - 0.2 * x2),
    )

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-2 * eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "fac": pd.Categorical(fac, categories=levels),
            "y": y,
        }
    )


def _make_re_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Random effects data: continuous x + factor g with group-level effects.

    Matches the conftest.py re_model_data fixture structure (n=300, 20 groups).
    """
    rng = np.random.default_rng(seed)
    n = 300
    n_groups = 20
    x = rng.uniform(0, 1, n)
    g = rng.choice([f"g{i}" for i in range(n_groups)], size=n)

    # True group effects
    b_intercept = rng.normal(0, 1.0, n_groups)
    group_idx = {f"g{i}": i for i in range(n_groups)}
    group_effect = np.array([b_intercept[group_idx[gi]] for gi in g])

    # Smooth + RE truth
    eta = np.sin(2 * np.pi * x) + group_effect

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.5, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame(
        {
            "x": x,
            "g": pd.Categorical(g),
            "y": y,
        }
    )


# ---------------------------------------------------------------------------
# Smooth configuration registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothConfig:
    """Configuration for one smooth type in the validation matrix."""

    py_formula: str
    r_formula: str
    data_type: str  # dispatched in _get_data


SMOOTH_CONFIGS: dict[str, SmoothConfig] = {
    "tp": SmoothConfig(
        py_formula="y ~ s(x, k=10, bs='tp')",
        r_formula="y ~ s(x, k=10, bs='tp')",
        data_type="single",
    ),
    "cr": SmoothConfig(
        py_formula="y ~ s(x, k=10, bs='cr')",
        r_formula="y ~ s(x, k=10, bs='cr')",
        data_type="single",
    ),
    "te": SmoothConfig(
        py_formula="y ~ te(x1, x2, k=5)",
        r_formula="y ~ te(x1, x2, k=c(5,5))",
        data_type="two_smooth",
    ),
    "ti": SmoothConfig(
        py_formula="y ~ ti(x1, x2, k=5)",
        r_formula="y ~ ti(x1, x2, k=c(5,5))",
        data_type="two_smooth",
    ),
    "tp_by": SmoothConfig(
        py_formula="y ~ s(x, by=fac, k=10, bs='tp') + fac",
        r_formula="y ~ s(x, by=fac, k=10, bs='tp') + fac",
        data_type="factor_by",
    ),
    "cr_by": SmoothConfig(
        py_formula="y ~ s(x, by=fac, k=10, bs='cr') + fac",
        r_formula="y ~ s(x, by=fac, k=10, bs='cr') + fac",
        data_type="factor_by",
    ),
    "te_by": SmoothConfig(
        py_formula="y ~ te(x1, x2, by=fac, k=5) + fac",
        r_formula="y ~ te(x1, x2, by=fac, k=c(5,5)) + fac",
        data_type="factor_by_2d",
    ),
    "re": SmoothConfig(
        py_formula="y ~ s(g, bs='re')",
        r_formula="y ~ s(g, bs='re')",
        data_type="re",
    ),
    "re_slope": SmoothConfig(
        py_formula="y ~ s(x, g, bs='re')",
        r_formula="y ~ s(x, g, bs='re')",
        data_type="re",
    ),
    "re_mixed": SmoothConfig(
        py_formula="y ~ s(x, k=10, bs='tp') + s(g, bs='re')",
        r_formula="y ~ s(x, k=10, bs='tp') + s(g, bs='re')",
        data_type="re",
    ),
    "gp": SmoothConfig(
        py_formula="y ~ s(x, bs='gp')",
        r_formula="y ~ s(x, bs='gp')",
        data_type="gp_1d",
    ),
    "gp_2d": SmoothConfig(
        py_formula="y ~ s(x, z, bs='gp', k=30)",
        r_formula="y ~ s(x, z, bs='gp', k=30)",
        data_type="gp_2d",
    ),
    "gp_mixed": SmoothConfig(
        py_formula="y ~ x + s(x, bs='gp')",
        r_formula="y ~ x + s(x, bs='gp')",
        data_type="gp_1d_par",
    ),
    "gp_te": SmoothConfig(
        py_formula="y ~ te(x1, x2, bs='gp', k=5)",
        r_formula="y ~ te(x1, x2, bs='gp', k=c(5, 5))",
        data_type="gp_te_2d",
    ),
    "gp_ti": SmoothConfig(
        py_formula=(
            "y ~ s(x1, bs='gp', k=5) + s(x2, bs='gp', k=5) + ti(x1, x2, bs='gp', k=5)"
        ),
        r_formula=(
            "y ~ s(x1, bs='gp', k=5) + s(x2, bs='gp', k=5) "
            "+ ti(x1, x2, bs='gp', k=c(5, 5))"
        ),
        data_type="gp_te_2d",
    ),
}

FAMILIES = ["gaussian", "binomial", "poisson", "gamma", "nb"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_data(config: SmoothConfig, family: str) -> pd.DataFrame:
    """Generate data for a given smooth config and family."""
    if config.data_type == "single":
        return _make_single_data(family)
    if config.data_type == "two_smooth":
        return _make_two_smooth_data(family)
    if config.data_type == "factor_by":
        return _make_factor_by_data(family)
    if config.data_type == "factor_by_2d":
        return _make_factor_by_2d_data(family)
    if config.data_type == "re":
        return _make_re_data(family)
    if config.data_type == "gp_1d":
        return _make_gp_1d_data(family)
    if config.data_type == "gp_2d":
        return _make_gp_2d_data(family)
    if config.data_type == "gp_1d_par":
        return _make_gp_1d_par_data(family)
    if config.data_type == "gp_te_2d":
        return _make_gp_te_2d_data(family)
    raise ValueError(f"Unknown data_type: {config.data_type}")


def _r_tol(smooth_key: str, family_name: str):
    """Tolerance for R comparison: MODERATE for Gaussian single-smooth, LOOSE otherwise.

    Tensor products and factor-by always use LOOSE (flat REML surfaces,
    multiple sp). GLM families also use LOOSE (iterative PIRLS compounding).
    """
    if family_name == "gaussian" and smooth_key in (
        "tp",
        "cr",
        "re",
        "re_slope",
        "re_mixed",
        "gp",
        "gp_2d",
    ):
        return MODERATE
    return LOOSE


def _fitted_tol(smooth_key: str, family_name: str):
    """Tolerance for fitted value comparison, wider for flat REML surfaces."""
    # Tensor factor-by with GLM/NB: 6+ sp, very flat REML surface
    if smooth_key in ("te_by",) and family_name in ("binomial", "poisson", "nb"):
        return LOOSE
    # Factor-by with binomial: multiple sp + binary response
    if smooth_key.endswith("_by") and family_name == "binomial":
        return LOOSE
    # Tensor interaction with GLM: flat surface
    if smooth_key in ("ti",) and family_name != "gaussian":
        return LOOSE
    return _r_tol(smooth_key, family_name)


def _compare_fitted_not_coefs(smooth_key: str) -> bool:
    """Whether to compare fitted values instead of raw coefficients.

    TPRS: eigenvector sign ambiguity makes coefficient comparison meaningless.
    Tensor products and factor-by: flat REML surfaces mean different sp can
    give different coefficients that produce equivalent fitted values.
    """
    return smooth_key in (
        "tp",
        "tp_by",
        "te",
        "ti",
        "te_by",
        "cr_by",
        "re_mixed",
        "gp",
        "gp_2d",
        "gp_mixed",
        "gp_te",
        "gp_ti",
    )


# ---------------------------------------------------------------------------
# Cell IDs for parametrization
# ---------------------------------------------------------------------------

CELL_IDS = [
    (smooth_key, family) for smooth_key in SMOOTH_CONFIGS for family in FAMILIES
]


def _cell_id(val):
    """Human-readable test ID: 'tp-gaussian', 'cr_by-binomial', etc."""
    return f"{val[0]}-{val[1]}"


def _fit_matrix_model(smooth_key, family_name, config, data):
    """Retain exact optimizer diagnostics for the intermittent native GP gate."""
    if (smooth_key, family_name) != ("gp_2d", "gaussian"):
        return GAM(config.py_formula, family=family_name).fit(data)

    trace = []
    original_fit = NewtonOptimizer._fit_and_score
    original_check = NewtonOptimizer._check_convergence
    original_step = NewtonOptimizer._step_halve_gaussian
    original_change = NewtonOptimizer._gaussian_trial_score_change

    def traced_fit(optimizer, params, beta_warm):
        result = original_fit(optimizer, params, beta_warm)
        trace.append(
            {
                "event": "trial",
                "params": np.asarray(params).tolist(),
                "score": float(result[1]),
                "inner_converged": bool(result[0].converged),
            }
        )
        return result

    def traced_check(optimizer, criterion, params, score, score_old, **kwargs):
        result = original_check(
            optimizer, criterion, params, score, score_old, **kwargs
        )
        gradient, hessian, score_scale, converged = result
        trace.append(
            {
                "event": "check",
                "params": np.asarray(params).tolist(),
                "score": float(score),
                "previous_score": float(score_old),
                "gradient": np.asarray(gradient).tolist(),
                "hessian": np.asarray(hessian).tolist(),
                "score_scale": float(score_scale),
                "tol": optimizer._tol,
                "converged": bool(converged),
                "design_sha256": hashlib.sha256(
                    np.asarray(optimizer._fd.X).tobytes()
                ).hexdigest(),
            }
        )
        return result

    def traced_step(optimizer, params, step, score, beta_warm, score_scale):
        trace.append({"event": "step", "step": np.asarray(step).tolist()})
        result = original_step(optimizer, params, step, score, beta_warm, score_scale)
        trace.append({"event": "outcome", "outcome": result[3].name})
        return result

    def traced_change(
        optimizer, params, params_trial, beta, trial, score, score_trial, scale
    ):
        result = original_change(
            optimizer, params, params_trial, beta, trial, score, score_trial, scale
        )
        trace.append(
            {"event": "score_change", "raw": score_trial - score, "compared": result}
        )
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(NewtonOptimizer, "_fit_and_score", traced_fit)
        patch.setattr(NewtonOptimizer, "_check_convergence", traced_check)
        patch.setattr(NewtonOptimizer, "_step_halve_gaussian", traced_step)
        patch.setattr(NewtonOptimizer, "_gaussian_trial_score_change", traced_change)
        try:
            return GAM(config.py_formula, family=family_name).fit(data)
        finally:
            print(  # noqa: T201 - retain native CI failure diagnostics
                "GP_GAUSSIAN_TRACE " + json.dumps(trace), flush=True
            )


# ---------------------------------------------------------------------------
# A. TestValidationMatrix — R comparison (75 cells)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
class TestValidationMatrix:
    """Systematic R comparison across all smooth type x family cells."""

    @pytest.fixture(scope="class", params=CELL_IDS, ids=[_cell_id(c) for c in CELL_IDS])
    def cell(self, request):
        """Fit Python GAM and R GAM, return both for comparison."""
        from tests.r_bridge import RBridge

        smooth_key, family_name = request.param
        config = SMOOTH_CONFIGS[smooth_key]
        data = _get_data(config, family_name)

        model = _fit_matrix_model(smooth_key, family_name, config, data)
        bridge = RBridge()
        r_result = bridge.fit_gam(config.r_formula, data, family=family_name)

        return smooth_key, family_name, model, r_result

    def test_matches_r(self, cell):
        """Compare each validation-matrix cell against mgcv."""
        smooth_key, family_name, model, r_result = cell
        cell_id = f"{smooth_key}-{family_name}"
        collector = _AssertCollector()

        def assert_deviance_vs_r() -> None:
            tol = _r_tol(smooth_key, family_name)
            np.testing.assert_allclose(
                model.deviance,
                r_result["deviance"],
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{cell_id}] deviance",
            )

        def assert_fitted_values_vs_r() -> None:
            tol = _fitted_tol(smooth_key, family_name)
            np.testing.assert_allclose(
                model.fitted_values,
                r_result["fitted_values"],
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{cell_id}] fitted values",
            )

        def assert_edf_vs_r() -> None:
            tol = _fitted_tol(smooth_key, family_name)
            if smooth_key == "gp_mixed":
                # y ~ x + s(x, bs='gp') is intentionally rank-deficient:
                # parametric x duplicates the GP linear null-space column.
                # Smooth-block EDF allocation is pivot-dependent, so compare
                # the allocation-invariant total model EDF for this cell.
                py_edf_total = model.edf_total
                r_edf_total = float(r_result["edf_total"])
                edf_label = "total model EDF"
            else:
                py_edf_total = float(np.sum(model.edf))
                r_edf_total = float(np.sum(r_result["edf"]))
                edf_label = "total EDF"
            np.testing.assert_allclose(
                py_edf_total,
                r_edf_total,
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{cell_id}] {edf_label}",
            )

        def assert_scale_vs_r() -> None:
            tol = _r_tol(smooth_key, family_name)
            np.testing.assert_allclose(
                model.scale,
                r_result["scale"],
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{cell_id}] scale",
            )

        def assert_coefficients_vs_r() -> None:
            tol = _r_tol(smooth_key, family_name)
            if _compare_fitted_not_coefs(smooth_key):
                ftol = _fitted_tol(smooth_key, family_name)
                np.testing.assert_allclose(
                    model.fitted_values,
                    r_result["fitted_values"],
                    rtol=ftol.rtol,
                    atol=ftol.atol,
                    err_msg=f"[{cell_id}] fitted values (coef proxy)",
                )
            else:
                np.testing.assert_allclose(
                    model.coefficients,
                    r_result["coefficients"],
                    rtol=tol.rtol,
                    atol=tol.atol,
                    err_msg=f"[{cell_id}] coefficients",
                )

        def assert_self_prediction_roundtrip() -> None:
            pred = model.predict()
            np.testing.assert_allclose(
                pred,
                model.fitted_values,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
                err_msg=f"[{cell_id}] self-prediction roundtrip",
            )

        def assert_theta_vs_r() -> None:
            r_theta = r_result.get("theta")
            if r_theta is None:
                return
            py_theta = float(model.family.get_theta(transformed=True)[0])
            tol = _r_tol(smooth_key, family_name)
            np.testing.assert_allclose(
                py_theta,
                r_theta,
                rtol=tol.rtol,
                atol=tol.atol,
                err_msg=f"[{cell_id}] theta",
            )

        collector.check("deviance vs R", assert_deviance_vs_r)
        collector.check("fitted values vs R", assert_fitted_values_vs_r)
        collector.check("EDF vs R", assert_edf_vs_r)
        collector.check("scale vs R", assert_scale_vs_r)
        collector.check("coefficients vs R", assert_coefficients_vs_r)
        collector.check("self prediction roundtrip", assert_self_prediction_roundtrip)
        collector.check("theta vs R", assert_theta_vs_r)
        collector.raise_if_any(cell_id)


# ---------------------------------------------------------------------------
# B. TestHardGateInvariants — structural invariants (no R required)
# ---------------------------------------------------------------------------


def _public_regular_efs_profile_digest(
    prepared, batch, offset, family_name, link, formula, control, startup
) -> str:
    """Bind a reviewed tolerance to raw rows, coordinates, starts and controls."""
    digest = hashlib.sha256()
    for name, value, dtype in (
        ("x", batch.columns["x"], "<f8"),
        ("y", batch.y, "<f8"),
        ("weight", batch.weight, "<f8"),
        ("offset", offset, "<f8"),
        ("valid", batch.valid, "u1"),
        ("row_positions", batch.row_positions, "<i8"),
    ):
        array = np.asarray(value, dtype=dtype)
        digest.update(name.encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    structure = prepared.fitting.penalty_structure
    blocks = []
    for block_index, block in enumerate(structure.blocks):
        transform = np.asarray(block.transform.dense(), dtype="<f8")
        digest.update(f"transform-{block_index}".encode())
        digest.update(repr(transform.shape).encode())
        digest.update(transform.tobytes(order="C"))
        for penalty_index, penalty in enumerate(block.dense_penalties()):
            penalty = np.asarray(penalty, dtype="<f8")
            digest.update(f"penalty-{block_index}-{penalty_index}".encode())
            digest.update(repr(penalty.shape).encode())
            digest.update(penalty.tobytes(order="C"))
        blocks.append(
            {
                "start": block.start,
                "stop": block.stop,
                "sp_indices": block.sp_indices,
                "ranks": block.ranks,
            }
        )
    payload = {
        "basis_fingerprint": prepared.basis_fingerprint,
        "blocks": blocks,
        "control": asdict(control),
        "family": family_name,
        "formula": formula,
        "initial_scale_hex": float(startup.score_phi).hex(),
        "initial_smoothing_hex": tuple(
            float(value).hex() for value in np.exp(startup.log_lambda)
        ),
        "link": link,
        "n_coef": prepared.n_coef,
        "n_obs": prepared.n_obs,
        "source_fingerprint": prepared.source_fingerprint,
    }
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def test_public_regular_efs_profile_digest_binds_basis_controls_and_starts() -> None:
    """A reviewed numerical profile cannot migrate to a nearby model."""
    from jaxgam.control import EFSControl
    from jaxgam.execution.efs_stream_provider import RegularStreamEFSProvider
    from jaxgam.execution.efs_stream_start import prepare_stream_efs_start
    from jaxgam.formula.design_provider import StreamDesign
    from jaxgam.formula.prepare import prepare_model
    from tests.test_execution.test_efs_stream_start import _regular

    family = Poisson("identity")
    original = _regular(family)
    batch = next(original.source.source.scan(original.prepared.n_obs))
    data = pd.DataFrame({"x": np.asarray(batch.columns["x"]), "y": batch.y})
    source = DataFrameRowSource(
        data,
        response="y",
        weights=np.asarray(batch.weight),
        offset=np.asarray(batch.offset),
    )
    formula = 'y ~ s(x, bs="cr", k=6)'
    control = FitControl(
        execution="stream",
        linear_solver="qr",
        batch_rows=11,
        uncertainty="fisher",
    )
    prepared = prepare_model(parse_formula(formula), source, family=family)
    startup = prepare_stream_efs_start(
        RegularStreamEFSProvider.create(
            StreamDesign(prepared, source),
            family,
            maximum_bytes=10_000_000,
            batch_rows=11,
            control=control.efs,
        )
    )

    def digest(model, model_formula, model_control, model_startup):
        return _public_regular_efs_profile_digest(
            model,
            batch,
            np.asarray(batch.offset),
            "poisson",
            "identity",
            model_formula,
            model_control,
            model_startup,
        )

    expected = digest(prepared, formula, control, startup)
    assert (
        expected == "71bc41060f5f138cb00bd8a2328e726718d98454b02eb71634470699bd6ce0a8"
    )
    changed_control = replace(control, efs=EFSControl(score_tolerance=0.01))
    assert digest(prepared, formula, changed_control, startup) != expected
    assert (
        digest(
            prepared,
            formula,
            control,
            replace(startup, log_lambda=startup.log_lambda + 0.01),
        )
        != expected
    )
    assert (
        digest(
            prepared,
            formula,
            control,
            replace(startup, score_phi=startup.score_phi + 0.01),
        )
        != expected
    )
    changed_formula = 'y ~ s(x, bs="cs", k=6)'
    changed_prepared = prepare_model(
        parse_formula(changed_formula), source, family=family
    )
    assert digest(changed_prepared, changed_formula, control, startup) != expected


@pytest.mark.parametrize("family_name", ["gaussian", "gamma", "poisson", "binomial"])
@pytest.mark.parametrize(
    "link",
    [
        "identity",
        "log",
        "logit",
        "probit",
        "cloglog",
        "inverse",
        "inverse_squared",
        "sqrt",
    ],
)
def test_public_streamed_efs_regular_family_link_inventory(
    family_name: str, link: str
) -> None:
    """Every public regular cell has selected R parity or its pinned boundary."""
    from jaxgam.execution.efs_stream_provider import RegularStreamEFSProvider
    from jaxgam.execution.efs_stream_start import prepare_stream_efs_start
    from jaxgam.families.standard import Binomial, Gamma
    from jaxgam.formula.design_provider import StreamDesign
    from jaxgam.formula.prepare import prepare_model
    from tests.test_execution.test_efs_stream_start import _regular

    family = {
        "gaussian": Gaussian,
        "gamma": Gamma,
        "poisson": Poisson,
        "binomial": Binomial,
    }[family_name](link=link)
    stream = _regular(family)
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    data = pd.DataFrame({"x": np.asarray(batch.columns["x"]), "y": np.asarray(batch.y)})
    offset = np.asarray(batch.offset)
    if family_name == "binomial" and link == "log":
        offset = np.full(len(data), -0.5)
    elif family_name == "binomial" and link in {"inverse", "inverse_squared"}:
        offset = np.full(len(data), 2.0)
    formula = 'y ~ s(x, bs="cr", k=6)'
    control = FitControl(
        execution="stream",
        linear_solver="qr",
        batch_rows=11,
        uncertainty="fisher",
    )
    model = GAM(
        formula,
        family=family,
        optimizer="efs",
        control=control,
    )
    source = DataFrameRowSource(
        data,
        response="y",
        weights=np.asarray(batch.weight),
        offset=offset,
    )
    if family_name == "poisson" and link in {"logit", "probit", "cloglog"}:
        with pytest.raises(ValueError, match="finite null link mean"):
            model.fit(source, result="prediction")
        if r_available():
            oracle_data = data.assign(w=np.asarray(batch.weight), off=offset)
            expected_r_error = (
                r"Value 1\.1 out of range"
                if link == "logit"
                else "missing value where TRUE/FALSE needed"
            )
            with pytest.raises(RBridgeError, match=expected_r_error):
                RBridge(mode="rpy2").fit_efs(
                    formula,
                    oracle_data,
                    f"{family_name}_{link}",
                    weights="w",
                    offset="off",
                    initial_smoothing=np.array([0.4]),
                    scale=1.0,
                )
        return

    result = model.fit(source, result="prediction")
    assert result.converged
    assert result.lambda_strategy == "efs_reml"
    assert result.execution_route == "stream_efs_qr"
    assert result.optimizer_diagnostics is not None
    assert result.optimizer_diagnostics.provider_source_scans > 0
    assert np.all(np.isfinite(result.coefficients))
    assert np.isfinite(result.deviance)
    assert np.isfinite(result.score)
    assert result.deviance >= -STRICT.atol
    if not r_available():
        return

    prepared = prepare_model(parse_formula(formula), source, family=family)
    startup = prepare_stream_efs_start(
        RegularStreamEFSProvider.create(
            StreamDesign(prepared, source),
            family,
            maximum_bytes=10_000_000,
            batch_rows=11,
            control=control.efs,
        )
    )
    oracle_data = data.assign(w=np.asarray(batch.weight), off=offset)
    oracle_arguments: dict[str, object] = {
        "weights": "w",
        "offset": "off",
        "initial_smoothing": np.exp(startup.log_lambda),
    }
    if family.scale_known:
        oracle_arguments["scale"] = 1.0
    else:
        oracle_arguments["initial_scale"] = startup.score_phi
    bridge = RBridge(mode="rpy2")
    if family_name == "binomial" and link in {"inverse", "inverse_squared"}:
        # The selected efsudr fit is valid. Pinned estimate.gam subsequently
        # refits an offset-only GLM solely to replace null.deviance, and that
        # auxiliary bounded-link GLM rejects this input. Preserve both facts.
        with pytest.raises(RBridgeError, match="no valid set of coefficients"):
            bridge.fit_efs(
                formula,
                oracle_data,
                f"{family_name}_{link}",
                **oracle_arguments,
            )
        reference = bridge.fit_efs_selected_before_offset_null_deviance(
            formula,
            oracle_data,
            f"{family_name}_{link}",
            **oracle_arguments,
        )
    else:
        reference = bridge.fit_efs(
            formula,
            oracle_data,
            f"{family_name}_{link}",
            **oracle_arguments,
        )
    prediction, prediction_se = result.predict(data, se_fit=True, offset=offset)
    prediction_matrix = result.predict_matrix(data)
    reference_link_se = np.sqrt(
        np.maximum(
            np.sum((prediction_matrix @ reference["Vp"]) * prediction_matrix, axis=1),
            0.0,
        )
    )
    reference_eta = np.asarray(family.link.link(reference["fitted_values"]))
    reference_response_se = reference_link_se * np.abs(
        np.asarray(family.link.mu_eta(reference_eta))
    )
    profile_digest = _public_regular_efs_profile_digest(
        prepared,
        batch,
        offset,
        family_name,
        link,
        formula,
        control,
        startup,
    )
    reviewed_profile = {
        ("poisson", "identity"): (
            "71bc41060f5f138cb00bd8a2328e726718d98454b02eb71634470699bd6ce0a8",
            "fc7066fc896ad7131ad8a105979163a6da1db6c30361397d8a3f72ef361b4a9f",
            4,
            {
                "coefficients",
                "fitted values",
                "smoothing",
                "total EDF",
                "prediction SE",
            },
        ),
        ("binomial", "log"): (
            "1b92d5b8e29458e873c702d93e22e1a639ed34e27711f4e6e08a6b4850f01ef8",
            "b85057b3ab7ac3340e364cfa1407f3887f5d00b5f53ee3e2af00c976f191c552",
            15,
            {
                "coefficients",
                "fitted values",
                "deviance",
                "score",
                "smoothing",
                "total EDF",
                "prediction SE",
            },
        ),
    }.get((family_name, link))
    moderate_fields: set[str] = set()
    if reviewed_profile is not None:
        (
            expected_profile_digest,
            expected_data_hash,
            expected_outer,
            moderate_fields,
        ) = reviewed_profile
        # These are the two exact, reviewed B=11 profiles in the durable
        # EFS5.2 numerical record. A fixture, formula, control or iteration
        # change must not silently inherit their field-specific tolerance.
        assert profile_digest == expected_profile_digest
        assert reference["provenance"]["data_hash"] == expected_data_hash
        assert result.n_iter == reference["outer_iterations"] == expected_outer
    collector = _AssertCollector()
    for field, actual, expected in (
        ("coefficients", result.coefficients, reference["coefficients"]),
        ("fitted values", prediction, reference["fitted_values"]),
        ("deviance", result.deviance, reference["deviance"]),
        ("score", result.score, reference["reml_score"]),
        ("smoothing", result.smoothing_params, reference["smoothing_params"]),
        ("scale", result.scale, reference["scale"]),
        ("total EDF", result.edf_total, reference["edf_total"]),
        ("prediction SE", prediction_se, reference_response_se),
    ):
        tolerance = MODERATE if field in moderate_fields else STRICT
        collector.check(
            field,
            lambda a=actual, e=expected, t=tolerance: np.testing.assert_allclose(
                a, e, rtol=t.rtol, atol=t.atol
            ),
        )
    collector.check(
        "outer iterations",
        lambda: np.testing.assert_equal(result.n_iter, reference["outer_iterations"]),
    )
    collector.check(
        "source convergence",
        lambda: np.testing.assert_equal(reference["convergence"], "full convergence"),
    )
    collector.raise_if_any(f"public streamed {family_name}/{link} EFS")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    "case",
    ["gaussian", "gamma", "poisson", "binomial"]
    + [
        f"nb-{link}-{mode}"
        for link in ("log", "identity", "sqrt")
        for mode in ("fixed", "estimated")
    ],
)
def test_public_streamed_default_start_efs_matches_pinned_selected_fit(r_bridge, case):
    """Public row-source results preserve the reviewed internal/R selected fit."""
    from jaxgam.execution.efs_stream import fit_streamed_efs
    from jaxgam.families.standard import Binomial, Gamma
    from jaxgam.results import _prepared_transform_coefficients_cpu
    from tests.test_execution.test_efs_stream_start import _regular
    from tests.test_execution.test_nb_stream import _fixture

    versions_match, reason = r_bridge.check_versions()
    assert versions_match, reason
    if case.startswith("nb-"):
        _, link, mode = case.split("-")
        stream, family = _fixture(link, estimated=mode == "estimated", smooth=True)
        r_name = "nb"
        theta = -2.7 if mode == "estimated" else 2.7
    else:
        family = {
            "gaussian": Gaussian,
            "gamma": Gamma,
            "poisson": Poisson,
            "binomial": Binomial,
        }[case]()
        stream = _regular(family)
        r_name, link, theta = (
            case,
            {
                "gaussian": "identity",
                "gamma": "inverse",
                "poisson": "log",
                "binomial": "logit",
            }[case],
            1.0,
        )
    result = fit_streamed_efs(stream, family, maximum_bytes=10_000_000, batch_rows=11)
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    reference = RBridge(mode="rpy2").efs_streamed_default_start_reference(
        batch.columns["x"],
        batch.y,
        batch.weight,
        batch.offset,
        family=r_name,
        link=link,
        theta=theta,
        smoothing=np.exp(result.startup.log_lambda),
        scale=result.startup.score_phi,
    )
    fit = result.fit
    beta = _prepared_transform_coefficients_cpu(
        stream.prepared, np.asarray(fit.pirls_result.coefficients)
    )
    mu = result.family.link.linkinv(
        stream.prepared.evaluate_batch(batch) @ beta + batch.offset
    )
    public_data = pd.DataFrame(
        {"x": np.asarray(batch.columns["x"]), "y": np.asarray(batch.y)}
    )
    public_family = copy.deepcopy(family)
    public = GAM(
        'y ~ s(x, bs="cr", k=6)',
        family=public_family,
        optimizer="efs",
        control=FitControl(
            execution="stream",
            linear_solver="qr",
            batch_rows=11,
        ),
    ).fit(
        DataFrameRowSource(
            public_data,
            response="y",
            weights=np.asarray(batch.weight),
            offset=np.asarray(batch.offset),
        ),
        result="prediction",
    )
    collector = _AssertCollector()
    reference_fields = {
        "beta": "coefficients",
        "mu": "fitted_values",
        "D": "deviance",
        "score": "reml_score",
        "sp": "smoothing_params",
        "edf": "edf_total",
        "scale": "scale",
    }
    for field, actual in (
        ("beta", beta),
        ("mu", mu),
        ("D", fit.pirls_result.deviance),
        ("score", fit.score),
        ("sp", fit.smoothing_params),
        ("edf", fit.edf),
        ("scale", fit.scale),
    ):
        expected = np.asarray(reference[reference_fields[field]], dtype=float)
        collector.check(
            field,
            lambda a=actual, e=expected: np.testing.assert_allclose(
                a, e, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.check(
        "outer iterations",
        lambda: np.testing.assert_equal(fit.n_iter, reference["outer_iterations"]),
    )
    collector.check(
        "source convergence",
        lambda: np.testing.assert_equal(reference["convergence"], "full convergence"),
    )
    collector.check(
        "stream convergence", lambda: np.testing.assert_equal(fit.converged, True)
    )
    for field, actual, expected in (
        ("public beta", public.coefficients, beta),
        (
            "public mu",
            public.predict(public_data, offset=np.asarray(batch.offset)),
            mu,
        ),
        ("public deviance", public.deviance, fit.pirls_result.deviance),
        ("public score", public.score, fit.score),
        ("public sp", public.smoothing_params, fit.smoothing_params),
        ("public scale", public.scale, fit.scale),
    ):
        collector.check(
            field,
            lambda a=actual, e=expected: np.testing.assert_allclose(
                a, e, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.check(
        "public outer iterations",
        lambda: np.testing.assert_equal(public.n_iter, fit.n_iter),
    )
    collector.check(
        "public strategy",
        lambda: np.testing.assert_equal(public.lambda_strategy, "efs_reml"),
    )
    collector.check(
        "public diagnostics",
        lambda: check_that(
            public.optimizer_diagnostics is not None
            and public.optimizer_diagnostics.provider_source_scans > 0
            and public.optimizer_diagnostics.startup_source_scans == 2,
            "public streamed EFS omitted source-cost diagnostics",
        ),
    )
    if case.startswith("nb-"):
        collector.check(
            "theta",
            lambda: np.testing.assert_allclose(
                fit.theta,
                np.asarray(reference["theta"]),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        collector.check(
            "public theta",
            lambda: np.testing.assert_allclose(
                public.theta,
                fit.theta,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        collector.check(
            "public selected family",
            lambda: np.testing.assert_allclose(
                public.family.get_theta(transformed=True)[0],
                fit.theta,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
    collector.raise_if_any(f"default-start streamed EFS {case}")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_streamed_regular_recovery_anchor_matches_pinned_efs_default(r_bridge):
    """mgcv.r omits G$null.coef when dispatching every efsudr refit."""
    versions_match, reason = r_bridge.check_versions()
    assert versions_match, reason
    from jaxgam.execution.efs_stream import fit_streamed_efs
    from tests.test_execution.test_efs_stream_start import _regular

    family = Gaussian()
    stream = _regular(family)
    execution = fit_streamed_efs(
        stream, family, maximum_bytes=10_000_000, batch_rows=11
    )
    batch = next(stream.source.source.scan(stream.prepared.n_obs))
    reference = RBridge(mode="rpy2").efs_streamed_default_start_reference(
        batch.columns["x"],
        batch.y,
        batch.weight,
        batch.offset,
        family="gaussian",
        link="identity",
        theta=1.0,
        smoothing=np.exp(execution.startup.log_lambda),
        scale=execution.startup.score_phi,
        trace_null_coef=True,
    )
    trace = reference["null_coef_trace_summary"]
    assert all(reference["null_coef_omitted"])
    assert trace[0] > 1
    assert trace[1] == stream.prepared.n_coef
    assert trace[2] == 0


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize("link", ["log", "inverse"])
def test_public_dense_efs_gaussian_patched_response_matches_pinned_r(
    r_bridge, link: str
) -> None:
    """Global fix.family starts must finish through the public EFS path."""
    rng = np.random.default_rng(7321)
    x = np.linspace(-0.7, 0.8, 83)
    y = np.exp(0.2 + 0.3 * x) + rng.normal(0.0, 0.08, len(x))
    y[[2, 15]] = [-0.2, -0.05] if link == "log" else 0.0
    weights = 0.4 + rng.uniform(size=len(x))
    weights[8] = 0.0
    offset = 0.03 * np.sin(x)
    data = pd.DataFrame({"x": x, "y": y, "w": weights, "off": offset})
    formula = "y ~ s(x, bs='cr', k=7)"
    family = Gaussian(link)
    setup = ModelSetup.build(
        parse_formula(formula), data, weights=weights, offset=offset
    )
    initial = efs_initial_log_lambda(setup, family)
    initial_scale = efs_initial_log_scale(setup, family)
    result = GAM(formula, family=family, optimizer="efs").fit(
        data, weights=weights, offset=offset
    )
    reference = r_bridge.fit_efs(
        formula,
        data,
        f"gaussian_{link}",
        weights="w",
        offset="off",
        initial_smoothing=np.exp(np.asarray(initial)),
        initial_scale=float(np.exp(initial_scale)),
    )
    collector = _AssertCollector()
    for label, actual, expected in (
        ("coefficients", result.coefficients, reference["coefficients"]),
        ("fitted values", result.fitted_values, reference["fitted_values"]),
        ("deviance", result.deviance, reference["deviance"]),
        ("score", result.score, reference["reml_score"]),
        ("smoothing", result.smoothing_params, reference["smoothing_params"]),
        ("edf", result.edf, reference["edf"]),
        ("edf total", result.edf_total, reference["edf_total"]),
        ("covariance", result.Vp, reference["Vp"]),
        ("null deviance", result.null_deviance, reference["null_deviance"]),
        ("reported Fletcher scale", result.scale, reference["scale"]),
    ):
        collector.check(
            label,
            lambda a=actual, e=expected: np.testing.assert_allclose(
                a, e, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.check(
        "strategy", lambda: np.testing.assert_equal(result.lambda_strategy, "efs_reml")
    )
    collector.check(
        "convergence", lambda: np.testing.assert_equal(result.converged, True)
    )
    collector.raise_if_any(f"public dense Gaussian/{link} patched-response EFS")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_public_dense_efs_gaussian_inverse_result_matches_pinned_r(r_bridge) -> None:
    """Broad public-result parity belongs in the optimizer-aware matrix."""
    rng = np.random.default_rng(7712)
    x = np.linspace(-1.0, 1.0, 120)
    wave = np.sin(5.0 * x) + 0.3 * np.cos(8.0 * x)
    mu = 1.0 / (1.0 + 0.22 * wave)
    data = pd.DataFrame(
        {"x": x, "y": np.maximum(mu + rng.normal(0.0, 0.06, len(x)), 0.05)}
    )
    formula = "y ~ s(x, bs='cr', k=7)"
    family = Gaussian("inverse")
    setup = ModelSetup.build(parse_formula(formula), data)
    initial = efs_initial_log_lambda(setup, family)
    initial_scale = efs_initial_log_scale(setup, family)
    result = GAM(formula, family=family, optimizer="efs").fit(data)
    reference = r_bridge.fit_efs(
        formula,
        data,
        "gaussian_inverse",
        initial_smoothing=np.exp(np.asarray(initial)),
        initial_scale=float(np.exp(initial_scale)),
        null_coef=True,
    )
    collector = _AssertCollector()
    for label, actual, expected in (
        ("coefficients", result.coefficients, reference["coefficients"]),
        ("fitted values", result.fitted_values, reference["fitted_values"]),
        ("deviance", result.deviance, reference["deviance"]),
        ("score", result.score, reference["reml_score"]),
        ("smoothing", result.smoothing_params, reference["smoothing_params"]),
        ("edf", result.edf, reference["edf"]),
        ("edf total", result.edf_total, reference["edf_total"]),
        ("covariance", result.Vp, reference["Vp"]),
        ("null deviance", result.null_deviance, reference["null_deviance"]),
        ("scale", result.scale, reference["scale"]),
    ):
        collector.check(
            label,
            lambda a=actual, e=expected: np.testing.assert_allclose(
                a, e, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.check(
        "strategy",
        lambda: np.testing.assert_equal(result.lambda_strategy, "efs_reml"),
    )
    collector.check(
        "convergence", lambda: np.testing.assert_equal(result.converged, True)
    )
    collector.raise_if_any("public dense Gaussian/inverse EFS")


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
@pytest.mark.parametrize(
    ("link", "selected_tolerance"),
    [("identity", MODERATE), ("sqrt", STRICT)],
)
def test_estimated_nb_nonlog_positive_offset_public_efs_matches_pinned_r(
    link: str, selected_tolerance: ToleranceClass
) -> None:
    """Exercise the source-valid default anchor and positive-curvature retry.

    Identity's selected coefficient/smoothing residual uses MODERATE after
    four source, adapter, and stopping correction passes. Its
    deviance, score, theta, and outer count remain STRICT; sqrt is STRICT for
    every fitted quantity. No explicit coefficient or theta start is supplied.
    """
    rng = np.random.default_rng(901)
    n = 160
    x = np.linspace(-1.0, 1.0, n)
    theta = 2.7
    mu = np.exp(1.2 + np.sin(3.2 * x) + 0.35 * np.cos(6.0 * x))
    data = pd.DataFrame(
        {
            "x": x,
            "y": rng.negative_binomial(theta, theta / (theta + mu)),
            "off": np.ones(n),
        }
    )
    formula = "y ~ s(x, bs='cr', k=7)"
    family = NegativeBinomial(theta=theta, link=link)
    setup = ModelSetup.build(
        parse_formula(formula), data, offset=data["off"].to_numpy()
    )
    initial_log_lambda = efs_initial_log_lambda(setup, family)
    result = GAM(formula, family=family, optimizer="efs").fit(
        data, offset=data["off"].to_numpy()
    )
    reference = RBridge(mode="rpy2").fit_efs(
        formula,
        data,
        f"nb_{link}",
        offset="off",
        initial_smoothing=np.exp(np.asarray(initial_log_lambda)),
        scale=1.0,
    )

    for actual, expected in (
        (result.coefficients, reference["coefficients"]),
        (result.fitted_values, reference["fitted_values"]),
        (result.smoothing_params, reference["smoothing_params"]),
    ):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=selected_tolerance.rtol,
            atol=selected_tolerance.atol,
        )
    for actual, expected in (
        (result.deviance, reference["deviance"]),
        (result.score, reference["reml_score"]),
        (result.theta, reference["theta"]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    assert result.converged
    assert result.n_iter == reference["outer_iterations"]
    assert result.optimizer_diagnostics is not None
    assert not result.optimizer_diagnostics.stabilized_solve_seen


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_public_dense_efs_poisson_log_seed1033_result_matches_pinned_r(
    r_bridge,
) -> None:
    """Preserve the reviewed seed-1033 public Poisson default-profile gate.

    After four source, adapter, and stopping correction passes, the public
    and direct controller agree at STRICT, while the pinned R terminal fit has
    coefficient/mean residuals below 4.1e-10. Those fitted quantities use the
    reviewed MODERATE gate; scalar score/deviance/scale checks remain STRICT.
    """
    data = _make_single_data("poisson", seed=SEED + 991).iloc[:96].copy()
    formula = "y ~ s(x, bs='cr', k=6)"
    family = Poisson()
    setup = ModelSetup.build(parse_formula(formula), data)
    initial = efs_initial_log_lambda(setup, family)
    result = GAM(formula, family=family, optimizer="efs").fit(data)
    reference = r_bridge.fit_efs(
        formula,
        data,
        "poisson",
        initial_smoothing=np.exp(np.asarray(initial)),
        scale=1.0,
        null_coef=True,
    )
    collector = _AssertCollector()
    for label, actual, expected in (
        ("coefficients", result.coefficients, reference["coefficients"]),
        ("fitted values", result.fitted_values, reference["fitted_values"]),
        ("smoothing", result.smoothing_params, reference["smoothing_params"]),
        ("edf", result.edf, reference["edf"]),
        ("edf total", result.edf_total, reference["edf_total"]),
        ("covariance", result.Vp, reference["Vp"]),
    ):
        collector.check(
            label,
            lambda a=actual, e=expected: np.testing.assert_allclose(
                a, e, rtol=MODERATE.rtol, atol=MODERATE.atol
            ),
        )
    for label, actual, expected in (
        ("deviance", result.deviance, reference["deviance"]),
        ("score", result.score, reference["reml_score"]),
        ("null deviance", result.null_deviance, reference["null_deviance"]),
        ("scale", result.scale, reference["scale"]),
    ):
        collector.check(
            label,
            lambda a=actual, e=expected: np.testing.assert_allclose(
                a, e, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.check(
        "outer iterations",
        lambda: np.testing.assert_equal(result.n_iter, reference["outer_iterations"]),
    )
    collector.check(
        "strategy",
        lambda: np.testing.assert_equal(result.lambda_strategy, "efs_reml"),
    )
    collector.check(
        "compact diagnostics",
        lambda: np.testing.assert_equal(
            result.optimizer_diagnostics.outer_iterations, result.n_iter
        ),
    )
    collector.raise_if_any("public dense Poisson/log EFS seed1033")


class TestHardGateInvariants:
    """Hard-gate invariants (design.md §18.1) for all smooth x family cells.

    These must hold regardless of R comparison and never be waived.
    Eight invariants tested:
    1. Model convergence
    2. Deviance >= 0
    3. Converged beta produces finite eta, mu (no NaN/Inf)
    4. EDF in [0, p] per term, total in [0, n]
    5. Vp symmetric PSD
    6. Penalty S_j symmetric PSD
    7. Estimated theta > 0 for extended families
    8. Rank(X) >= p - null_space_dim
    """

    @pytest.fixture(scope="class", params=CELL_IDS, ids=[_cell_id(c) for c in CELL_IDS])
    def fitted_model(self, request):
        smooth_key, family_name = request.param
        config = SMOOTH_CONFIGS[smooth_key]
        data = _get_data(config, family_name)
        model = _fit_matrix_model(smooth_key, family_name, config, data)
        return smooth_key, family_name, model

    def test_all_invariants(self, fitted_model):
        """Check every hard-gate invariant for one matrix cell."""
        smooth_key, family_name, model = fitted_model
        cell_id = f"{smooth_key}-{family_name}"
        collector = _AssertCollector()

        def assert_convergence() -> None:
            assert model.converged, f"[{cell_id}] model did not converge"

        def assert_deviance_non_negative() -> None:
            assert model.deviance >= 0, (
                f"[{cell_id}] negative deviance: {model.deviance}"
            )

        def assert_no_nan_in_converged() -> None:
            assert np.all(np.isfinite(model.coefficients)), (
                f"[{cell_id}] NaN/Inf in coefficients"
            )
            assert np.all(np.isfinite(model.fitted_values)), (
                f"[{cell_id}] NaN/Inf in fitted values"
            )
            assert np.all(np.isfinite(model.linear_predictor)), (
                f"[{cell_id}] NaN/Inf in linear predictor"
            )
            assert np.isfinite(model.scale), f"[{cell_id}] non-finite scale"
            assert np.isfinite(model.deviance), f"[{cell_id}] non-finite deviance"

        def assert_edf_bounds() -> None:
            p = model.X.shape[1]
            n = model.n

            assert np.all(model.edf > 0), (
                f"[{cell_id}] non-positive per-smooth EDF: {model.edf}"
            )
            assert model.edf_total <= p, (
                f"[{cell_id}] total EDF {model.edf_total} > p={p}"
            )
            assert model.edf_total <= n + MODERATE.atol, (
                f"[{cell_id}] total EDF {model.edf_total} > n={n}"
            )

        def assert_vp_symmetric_psd() -> None:
            Vp = model.Vp

            np.testing.assert_allclose(
                Vp,
                Vp.T,
                atol=STRICT.atol,
                err_msg=f"[{cell_id}] Vp not symmetric",
            )
            eigvals = np.linalg.eigvalsh(Vp)
            assert np.all(eigvals >= 0.0), (
                f"[{cell_id}] Vp has negative eigenvalue: {eigvals.min()}"
            )

        def assert_penalty_psd() -> None:
            for j, si in enumerate(model.smooth_info):
                for term in model.coef_map.terms:
                    if term.label == si.label and term.term_type != "parametric":
                        smooth_obj = term.smooth
                        if hasattr(smooth_obj, "penalties") and smooth_obj.penalties:
                            for k, S_j in enumerate(smooth_obj.penalties):
                                np.testing.assert_allclose(
                                    S_j,
                                    S_j.T,
                                    atol=STRICT.atol,
                                    err_msg=f"[{cell_id}] S[{j}][{k}] not symmetric",
                                )
                                eigs = np.linalg.eigvalsh(S_j)
                                assert np.all(eigs >= -STRICT.atol), (
                                    f"[{cell_id}] S[{j}][{k}] has negative "
                                    f"eigenvalue: {eigs.min()}"
                                )

        def assert_theta_positive() -> None:
            if not hasattr(model.family, "n_theta") or model.family.n_theta == 0:
                return
            theta = float(model.family.get_theta(transformed=True)[0])
            assert theta > 0, f"[{cell_id}] non-positive theta: {theta}"

        def assert_model_matrix_rank() -> None:
            X = model.X
            total_null_dim = sum(si.n_penalties for si in model.smooth_info)
            rank = np.linalg.matrix_rank(X)
            assert rank >= min(X.shape) - total_null_dim, (
                f"[{cell_id}] rank(X)={rank}, "
                f"expected >= {min(X.shape) - total_null_dim}"
            )

        collector.check("convergence", assert_convergence)
        collector.check("deviance non-negative", assert_deviance_non_negative)
        collector.check("finite converged values", assert_no_nan_in_converged)
        collector.check("EDF bounds", assert_edf_bounds)
        collector.check("Vp symmetric PSD", assert_vp_symmetric_psd)
        collector.check("penalty PSD", assert_penalty_psd)
        collector.check("theta positive", assert_theta_positive)
        collector.check("model matrix rank", assert_model_matrix_rank)
        collector.raise_if_any(cell_id)
