"""Discretization approximation study for the PR11 discrete route.

For one family, model and training size the harness fits the discrete route
at a ladder of covariate discretization levels and compares each fit with the
undiscretized dense JaxGAM fit of the same formula on the same rows.

Smoothing strategies (``--modes``):

* ``fixed_sp``: every fit at the dense REML fit's smoothing parameters (and,
  for the negative binomial with estimated theta, at its theta), compared
  with the dense fit at those values, so only the discrete representation
  differs. That representation includes mgcv's penalty normalization, which
  ``bam(discrete=TRUE)`` computes on the compact frame, so one ``sp`` is not
  one effective penalty at every level;
* ``fixed_sp_matched``: as ``fixed_sp`` with each smoothing parameter divided
  by its penalty's Frobenius-norm ratio (discrete over dense), so that
  ``sp_j * ||S_j||`` equals the dense value: the covariate rounding and basis
  change without the normalization change;
* ``working_model_reml``: the discrete working-model (``bgam.fitd``
  performance iteration) selection, compared with the dense nested exact REML
  selection. This comparison mixes two method changes, so each level is also
  compared with the same strategy at the ``exact`` level, which rounds no
  covariate (the discretization effect alone);
* ``exact_reml``: nested exact REML on the discrete design. The discrete route
  does not expose it; the harness records ``unsupported`` when it raises
  ``NotImplementedError`` and compares it like the others if it ever runs.

A level is a per-margin grid size ``m`` (``ContinuousDiscretizationControl.m``,
one value for every smooth), ``"default"`` (mgcv ``compress.df``: 1000/100/25
points for one/two/three-plus jointly discretized variables) or ``"exact"``
(``m`` equal to the training row count, so no covariate is rounded).

Method error and implementation error are kept apart. Every level records the
discretization (method) metrics: held-out link/response predictions, held-out
deviance, per-term centred curve shapes on grids, EDF per term and in total,
standard-error ratios and interval coverage, the covariate rounding error, and
the basis change (knot displacement and penalty-norm ratios). Coefficients are
not compared: different bases need not agree. The ``exact`` level is the
implementation check: with fixed smoothing parameters the discrete and dense
fits share one model, and optional pinned-R anchors (``bam(discrete=TRUE)`` at
the same grid and ``sp``; ``gam(method="REML")`` for the dense reference)
attribute any remaining gap to a route.

Every case runs in a fresh child process with a task-owned JAX cache. The
harness refuses a dirty checkout, a checkout at an unexpected commit, a child
whose loaded package or source identity differs from the parent's, and an
output path inside the checkout. It tracks no results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import benchmark_scaling
from scripts.benchmark_scaling import (
    _array_sha256,
    _runtime_metadata,
    isolated_cache_dir,
    loaded_source_identity,
    source_identity,
)

FAMILIES = ("gaussian", "poisson", "binomial", "gamma", "nb", "nb_fixed")
MODES = ("fixed_sp", "fixed_sp_matched", "working_model_reml", "exact_reml")
DEFAULT_MODES = ("fixed_sp", "fixed_sp_matched", "working_model_reml")
FIXED_MODES = ("fixed_sp", "fixed_sp_matched")
MODELS = {
    # Two smooths, their tensor interaction, a factor-by smooth with its
    # factor main effect, and a random effect.
    "study": (
        "y ~ fac + s(x0,bs='cr',k=10) + s(x1,bs='cr',k=10)"
        " + ti(x0,x1,bs='cr',k=5) + s(x2,bs='cr',k=8,by=fac) + s(g,bs='re')"
    ),
    # A full tensor product and a factor-by smooth.
    "tensor": "y ~ fac + te(x0,x1,bs='cr',k=6) + s(x2,bs='cr',k=8,by=fac)",
    # One smooth, for fast harness tests.
    "toy": "y ~ s(x0,bs='cr',k=8)",
}
MODEL_FACTORS = {"study": ("fac", "g"), "tensor": ("fac",), "toy": ()}
DEFAULT_LEVELS = ("10", "30", "100", "300", "default", "3000", "exact")
FACTOR_NAMES = ("fac", "g")
_FAC_LEVELS = ("a", "b", "c")
_G_LEVELS = tuple(f"g{index:02d}" for index in range(20))
# Generating negative binomial theta; ``nb_fixed`` fixes every fit at it.
_NB_THETA = 3.0
# Normal quantile of a nominal 95% interval.
_Z_95 = 1.959963984540054
# Construction budget admitting the "exact" level at study sizes. It changes
# no numerics, only which representations are admitted.
_TABLE_BUDGET_BYTES = 1024**3
_COVARIANCE_BUDGET_BYTES = 256 * 1024**2


class ProvenanceError(RuntimeError):
    """The checkout, the loaded package or the output path is not admissible."""


@dataclass(frozen=True)
class ApproximationCase:
    """One family, model, training size and discretization ladder."""

    family: str
    n_rows: int
    n_test: int = 2_000
    seed: int = 20_261_025
    levels: tuple[str, ...] = DEFAULT_LEVELS
    modes: tuple[str, ...] = DEFAULT_MODES
    model: str = "study"
    r_reference: bool = False
    batch_rows: int = 65_536

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "modes", tuple(self.modes))
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}.")
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {tuple(MODELS)}.")
        if self.n_rows < 50 or self.n_test < 10:
            raise ValueError("n_rows must be at least 50 and n_test at least 10.")
        if self.batch_rows <= 0:
            raise ValueError("batch_rows must be positive.")
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError("levels must be non-empty and distinct.")
        for level in self.levels:
            level_m(level, self.n_rows)
        if (
            not self.modes
            or len(set(self.modes)) != len(self.modes)
            or not set(self.modes) <= set(MODES)
        ):
            raise ValueError(f"modes must be distinct members of {MODES}.")

    @property
    def factor_names(self) -> tuple[str, ...]:
        """The model's factor covariates (``discrete.mf`` keeps their values)."""
        return MODEL_FACTORS[self.model]


def level_m(level: str, n_rows: int) -> int | None:
    """The ``ContinuousDiscretizationControl.m`` of one ladder level.

    ``"exact"`` uses ``m = n_rows``. For a one-dimensional group the grid then
    has at least as many points as unique values; a ``d``-variable joint group
    gets ``round(m**(1/d)) + 1`` points per variable, so its grid also exceeds
    the row count. Nothing is rounded either way (recorded per covariate).
    """
    if level == "default":
        return None
    if level == "exact":
        return int(n_rows)
    try:
        value = int(level)
    except ValueError as error:
        raise ValueError(
            f"level {level!r} is not an integer, 'default' or 'exact'."
        ) from error
    if value < 2 or str(value) != level:
        raise ValueError(f"level {level!r} must be an integer of at least 2.")
    return value


# --------------------------------------------------------------------------
# Provenance.
# --------------------------------------------------------------------------


def require_clean_checkout(root: Path, expect_commit: str | None = None) -> dict:
    """Return the checkout identity, refusing a dirty or unexpected tree."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProvenanceError(f"{root} is not a readable git checkout.") from error
    if status.strip():
        raise ProvenanceError(
            f"{root} has uncommitted or untracked changes; commit them first:\n{status}"
        )
    identity = source_identity(root)
    if expect_commit is not None and not identity["commit"].startswith(expect_commit):
        raise ProvenanceError(
            f"{root} is at {identity['commit']}, not the expected {expect_commit}."
        )
    return identity


def harness_identity() -> dict[str, str]:
    """Hash this script and the scaling helpers it imports.

    Both must come from one checkout: with a shared environment whose
    editable install points elsewhere, run with ``PYTHONPATH`` at the
    checkout.
    """
    path = Path(__file__).resolve()
    helpers = Path(benchmark_scaling.__file__).resolve()
    if helpers.parent != path.parent:
        raise ProvenanceError(
            f"imported helpers {helpers} are not beside {path}; "
            "set PYTHONPATH to the checkout."
        )
    digest = hashlib.sha256()
    for source in (path, helpers):
        digest.update(source.name.encode() + b"\0" + source.read_bytes())
    return {"harness_path": str(path), "harness_sha256": digest.hexdigest()}


def require_output_outside(output: Path, root: Path) -> Path:
    """Refuse an output path inside the checkout: results are never tracked."""
    resolved = output.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return resolved
    raise ProvenanceError(
        f"output {resolved} is inside the checkout {root}; write results elsewhere."
    )


def require_matching_child(parent: dict, child: dict) -> None:
    """A child must have run the parent's exact commit, tree and harness."""
    keys = ("commit", "dirty_sha256", "harness_sha256")
    mismatched = [key for key in keys if parent.get(key) != child.get(key)]
    if mismatched:
        raise ProvenanceError(
            f"child provenance differs from the parent in {mismatched}."
        )


# --------------------------------------------------------------------------
# Data and models.
# --------------------------------------------------------------------------


def _family(name: str, *, theta: float | None = None):
    """The JaxGAM family; ``theta`` fixes the ``nb`` family's theta."""
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson

    if name == "nb_fixed":
        return NegativeBinomial(theta=_NB_THETA, fixed=True)
    if name == "nb":
        if theta is None:
            return NegativeBinomial()
        return NegativeBinomial(theta=theta, fixed=True)
    return {
        "gaussian": Gaussian,
        "poisson": Poisson,
        "binomial": Binomial,
        "gamma": lambda: Gamma(link="log"),
    }[name]()


def make_data(
    case: ApproximationCase, *, test: bool
) -> tuple[pd.DataFrame, np.ndarray]:
    """Runtime rows and their true linear predictor (train or held-out)."""
    n = case.n_test if test else case.n_rows
    rng = np.random.default_rng(case.seed + (1 if test else 0))
    x0, x1, x2 = rng.uniform(size=(3, n))
    fac = pd.Categorical(rng.choice(_FAC_LEVELS, n), categories=_FAC_LEVELS)
    g = pd.Categorical(rng.choice(_G_LEVELS, n), categories=_G_LEVELS)
    if case.model == "toy":
        eta = np.sin(2.0 * np.pi * x0) + 0.5 * x0
    else:
        code = np.asarray(fac.codes)
        group_effect = np.random.default_rng(case.seed + 2).normal(
            0.0, 0.3, len(_G_LEVELS)
        )
        eta = (
            0.4 * np.sin(2.0 * np.pi * x0)
            + 2.0 * (x1 - 0.5) ** 2
            + 0.6 * np.sin(np.pi * x0) * np.cos(2.0 * np.pi * x1)
            + np.array([0.0, 0.3, -0.2])[code]
            + np.select(
                [code == 0, code == 1],
                [np.sin(3.0 * x2), -x2],
                0.5 * np.cos(4.0 * x2),
            )
        )
        if case.model == "study":
            eta = eta + group_effect[np.asarray(g.codes)]
    if case.family == "gaussian":
        y = eta + rng.normal(0.0, 0.5, n)
    elif case.family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif case.family == "binomial":
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    elif case.family == "gamma":
        y = rng.gamma(2.0, np.exp(eta) / 2.0)
    else:
        mean = np.exp(eta)
        y = rng.negative_binomial(_NB_THETA, _NB_THETA / (_NB_THETA + mean)).astype(
            float
        )
    frame = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "fac": fac, "g": g, "y": y})
    return frame, eta


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for name in frame.columns:
        values = frame[name]
        digest.update(name.encode() + b"\0")
        if isinstance(values.dtype, pd.CategoricalDtype):
            digest.update(_array_sha256(np.asarray(values.cat.codes)).encode())
            digest.update(repr(tuple(values.cat.categories)).encode())
        else:
            digest.update(_array_sha256(values.to_numpy()).encode())
    return digest.hexdigest()


def _discrete_control(case: ApproximationCase, m: int | None, mode: str):
    from jaxgam.control import ContinuousDiscretizationControl, FitControl
    from jaxgam.data.discretize import DiscretizationBudget

    return FitControl(
        execution="discrete",
        batch_rows=case.batch_rows,
        uncertainty="fisher",
        smoothing_strategy=(
            "working_model_reml" if mode == "working_model_reml" else "exact_reml"
        ),
        discretization=ContinuousDiscretizationControl(
            m=m,
            factor_names=case.factor_names,
            budget=DiscretizationBudget(max_table_bytes=_TABLE_BUDGET_BYTES),
        ),
    )


# --------------------------------------------------------------------------
# Fits and their comparable views.
# --------------------------------------------------------------------------


@dataclass
class FitView:
    """What the comparisons need from a dense or discrete fit.

    ``fit`` provides ``predict``, ``predict_matrix``, ``coefficients`` and
    ``family``. ``edf`` is per smooth term: the dense result's own values, or
    the harness reconstruction for a discrete fit (see ``discrete_edf``).
    """

    fit: Any
    edf: np.ndarray
    edf_total: float
    covariance: np.ndarray
    smoothing_params: np.ndarray
    scale: float
    theta: float | None


def dense_view(fit) -> FitView:
    return FitView(
        fit=fit,
        edf=np.asarray(fit.edf, dtype=float),
        edf_total=float(fit.edf_total),
        covariance=np.asarray(fit.Vp, dtype=float),
        smoothing_params=np.asarray(fit.smoothing_params, dtype=float),
        scale=float(fit.scale),
        theta=None if fit.theta is None else float(fit.theta),
    )


def discrete_view(fit, per_term_edf: Sequence[float]) -> FitView:
    return FitView(
        fit=fit,
        edf=np.asarray(per_term_edf, dtype=float),
        edf_total=float(fit.edf_total),
        covariance=np.asarray(fit.materialize_covariance(_COVARIANCE_BUDGET_BYTES)),
        smoothing_params=np.asarray(fit.smoothing_params, dtype=float),
        scale=float(fit.scale),
        theta=None if fit.theta is None else float(fit.theta),
    )


def _timed(function):
    start = time.perf_counter()
    value = function()
    return value, time.perf_counter() - start


def _fit_dense(case, formula, train, *, sp=None, theta=None):
    from jaxgam import GAM

    family = _family(case.family, theta=theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GAM(formula, family=family, sp=None if sp is None else list(sp)).fit(
            train, result="inference"
        )


def _fit_discrete(case, formula, train, m, mode, *, sp=None, theta=None):
    from jaxgam import GAM
    from jaxgam.data.source import DataFrameRowSource

    family = _family(case.family, theta=theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GAM(
            formula,
            family=family,
            sp=None if sp is None else list(sp),
            control=_discrete_control(case, m, mode),
        ).fit(DataFrameRowSource(train, response="y"), result="prediction")


# --------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "max_abs": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
    }


def _relative(observed: float, reference: float) -> float:
    return float((observed - reference) / max(abs(reference), np.finfo(float).tiny))


def _ratio_summary(numerator: np.ndarray, denominator: np.ndarray) -> dict:
    keep = denominator > 0.0
    if not np.any(keep):
        return {"min": None, "median": None, "max": None}
    ratio = numerator[keep] / denominator[keep]
    return {
        "min": float(np.min(ratio)),
        "median": float(np.median(ratio)),
        "max": float(np.max(ratio)),
    }


def _deviance(family, y: np.ndarray, mu: np.ndarray) -> float:
    return float(family.dev_resids(np.asarray(y), mu, np.ones(len(y))))


def held_out_metrics(
    view: FitView, reference: FitView, test: pd.DataFrame, eta_true: np.ndarray
) -> dict[str, Any]:
    """Prediction, deviance and interval agreement on held-out rows."""
    eta, se = view.fit.predict(test, pred_type="link", se_fit=True)
    eta_ref, se_ref = reference.fit.predict(test, pred_type="link", se_fit=True)
    eta, se, eta_ref, se_ref = (
        np.asarray(value, dtype=float) for value in (eta, se, eta_ref, se_ref)
    )
    mu = np.asarray(view.fit.family.link.inverse(eta))
    mu_ref = np.asarray(reference.fit.family.link.inverse(eta_ref))
    y = test.y.to_numpy()
    deviance = _deviance(view.fit.family, y, mu)
    deviance_ref = _deviance(reference.fit.family, y, mu_ref)
    low, high = eta - _Z_95 * se, eta + _Z_95 * se
    low_ref, high_ref = eta_ref - _Z_95 * se_ref, eta_ref + _Z_95 * se_ref
    overlap = np.clip(np.minimum(high, high_ref) - np.maximum(low, low_ref), 0.0, None)
    union = np.maximum(high, high_ref) - np.minimum(low, low_ref)
    return {
        "reference_link_max_abs": float(np.max(np.abs(eta_ref))),
        "link_difference": _summary(eta - eta_ref),
        "response_difference": _summary(mu - mu_ref),
        "link_difference_over_reference_se": _summary((eta - eta_ref) / se_ref),
        "deviance": {
            "reference": deviance_ref,
            "discrete": deviance,
            "difference": deviance - deviance_ref,
            "relative_difference": _relative(deviance, deviance_ref),
            "per_observation_difference": (deviance - deviance_ref) / len(test),
        },
        "intervals": {
            "nominal": 0.95,
            "se_ratio": _ratio_summary(se, se_ref),
            "coverage_of_reference_estimate": float(
                np.mean((low <= eta_ref) & (eta_ref <= high))
            ),
            "mean_interval_jaccard": float(np.mean(overlap / union)),
            "truth_coverage": {
                "reference": float(
                    np.mean((low_ref <= eta_true) & (eta_true <= high_ref))
                ),
                "discrete": float(np.mean((low <= eta_true) & (eta_true <= high))),
            },
        },
    }


def _grid_frames(train: pd.DataFrame, info) -> list[pd.DataFrame]:
    """Grid frames over one smooth's covariates, others held at a base row.

    A factor ``by`` smooth gets one frame per level of its factor; a random
    effect gets one row per level.
    """
    base = {
        "x0": float(np.median(train.x0)),
        "x1": float(np.median(train.x1)),
        "x2": float(np.median(train.x2)),
        "fac": _FAC_LEVELS[0],
        "g": _G_LEVELS[0],
    }
    variables = tuple(info.variables)
    if info.is_random:
        grid = {variables[0]: list(_G_LEVELS)}
    elif len(variables) == 1:
        name = variables[0]
        grid = {name: np.linspace(train[name].min(), train[name].max(), 200)}
    else:
        first, second = variables
        a, b = np.meshgrid(
            np.linspace(train[first].min(), train[first].max(), 30),
            np.linspace(train[second].min(), train[second].max(), 30),
            indexing="ij",
        )
        grid = {first: a.ravel(), second: b.ravel()}
    rows = len(next(iter(grid.values())))
    by_levels = _FAC_LEVELS if info.by_variable in FACTOR_NAMES else (None,)
    frames = []
    for level in by_levels:
        columns = {name: np.repeat(value, rows) for name, value in base.items()}
        columns.update(grid)
        if level is not None:
            columns[info.by_variable] = np.repeat(level, rows)
        frame = pd.DataFrame(columns)
        frame["fac"] = pd.Categorical(frame.fac, categories=_FAC_LEVELS)
        frame["g"] = pd.Categorical(frame.g, categories=_G_LEVELS)
        frames.append(frame[[c for c in train.columns if c != "y"]])
    return frames


def _term_curve(view: FitView, frame: pd.DataFrame, columns: slice, centered: bool):
    """A term's partial effect on a grid and its pointwise standard error.

    Centred curves (grid mean removed) are invariant to where a fit puts the
    term's constant, so fits with different centring constraints compare.
    """
    X = np.asarray(view.fit.predict_matrix(frame))[:, columns]
    if centered:
        X = X - X.mean(axis=0)
    beta = np.asarray(view.fit.coefficients)[columns]
    covariance = view.covariance[columns, columns]
    variance = np.einsum("ij,jk,ik->i", X, covariance, X)
    return X @ beta, np.sqrt(np.clip(variance, 0.0, None))


def partial_effect_metrics(
    view: FitView, reference: FitView, smooth_info, train: pd.DataFrame
) -> list[dict]:
    """Per-term shape and interval differences on grids.

    Smooth terms are compared after removing each curve's grid mean, because
    pinned ``bam(discrete=TRUE)`` derives centring constraints from the
    compact frame; random effects are compared directly.
    """
    records = []
    for info in smooth_info:
        columns = slice(info.first_coef, info.last_coef)
        centered = not info.is_random
        for frame in _grid_frames(train, info):
            if not np.any(np.asarray(reference.fit.predict_matrix(frame))[:, columns]):
                continue  # a by-level frame that does not select this term
            ref, se_ref = _term_curve(reference, frame, columns, centered)
            ours, se = _term_curve(view, frame, columns, centered)
            difference = ours - ref
            reference_rms = float(np.sqrt(np.mean(ref**2)))
            records.append(
                {
                    "label": info.label,
                    "by_level": (
                        None
                        if info.by_variable not in FACTOR_NAMES
                        else str(frame[info.by_variable].iloc[0])
                    ),
                    "grid_points": len(frame),
                    "centered": centered,
                    "l2": float(np.sqrt(np.mean(difference**2))),
                    "max_abs": float(np.max(np.abs(difference))),
                    "reference_rms": reference_rms,
                    "relative_l2": float(
                        np.sqrt(np.mean(difference**2))
                        / max(reference_rms, np.finfo(float).tiny)
                    ),
                    "se_ratio": _ratio_summary(se, se_ref),
                    "reference_curve_coverage": float(
                        np.mean(np.abs(difference) <= _Z_95 * se)
                    ),
                }
            )
    return records


@dataclass
class _CompactLevel:
    """The fit's own compact setup at one level, for snapping and penalties."""

    covariates: dict[str, dict[str, Any]]
    snapped: pd.DataFrame
    penalties: Any
    knots: dict[str, list[np.ndarray]]


def compact_level(
    case: ApproximationCase, formula: str, train: pd.DataFrame, m: int | None
) -> _CompactLevel:
    """Rebuild the compact setup the discrete fit uses at ``m``.

    The same pure Phase-1 construction with the same inputs and seed: its
    selectors give each training row's discretized covariates, and its
    penalties are the discrete model's.
    """
    from jaxgam.data.discretize import DiscretizationBudget
    from jaxgam.data.source import DataFrameRowSource
    from jaxgam.formula.continuous_setup import build_continuous_discrete_setup
    from jaxgam.formula.parser import parse_formula

    setup = build_continuous_discrete_setup(
        parse_formula(formula),
        DataFrameRowSource(train, response="y"),
        factor_names=case.factor_names,
        m=m,
        budget=DiscretizationBudget(max_table_bytes=_TABLE_BUDGET_BYTES),
        batch_rows=case.batch_rows,
    )
    if not hasattr(setup, "compact"):
        raise RuntimeError(f"compact setup unavailable: {setup}")
    try:
        covariates: dict[str, dict[str, Any]] = {}
        snapped: dict[str, Any] = {}
        for group in setup.compact.groups:
            index = group.index
            for name, values in zip(group.names, index.values, strict=True):
                row_values = np.asarray(values)[np.asarray(index.selector)]
                snapped[name] = row_values
                if name in index.metric_names:
                    original = train[name].to_numpy()
                    error = np.asarray(row_values, dtype=float) - original
                    span = float(np.ptp(original))
                    covariates[name] = {
                        "group": list(group.names),
                        "grid_points": int(index.points_per_metric),
                        "rounded": bool(index.rounded),
                        "compact_rows": int(index.n_unique),
                        "unique_values": int(train[name].nunique()),
                        "max_abs_error": float(np.max(np.abs(error))),
                        "rms_error": float(np.sqrt(np.mean(error**2))),
                        "max_abs_error_over_range": float(np.max(np.abs(error)) / span),
                    }
        frame = pd.DataFrame(
            {
                name: snapped.get(name, train[name].to_numpy())
                for name in train.columns
                if name != "y"
            }
        )
        for name in FACTOR_NAMES:
            frame[name] = pd.Categorical(
                frame[name], categories=train[name].cat.categories
            )
        return _CompactLevel(
            covariates,
            frame,
            setup.penalties,
            term_knots(setup.predict_spec.coef_map),
        )
    finally:
        setup.cleanup()


def _smooth_knots(smooth) -> list[np.ndarray]:
    """Knot vectors of a smooth, its ``by`` base smooth or tensor margins."""
    if getattr(smooth, "base_smooth", None) is not None:
        return _smooth_knots(smooth.base_smooth)
    marginals = getattr(smooth, "_marginals", None)
    if marginals:
        return [knots for margin in marginals for knots in _smooth_knots(margin)]
    knots = getattr(smooth, "_knots", None)
    return [] if knots is None else [np.asarray(knots, dtype=float)]


def term_knots(coef_map) -> dict[str, list[np.ndarray]]:
    """Knot vectors per smooth term (terms without knots are omitted)."""
    records = {}
    for term in coef_map.terms:
        if term.smooth is not None and (knots := _smooth_knots(term.smooth)):
            records[term.label] = knots
    return records


def knot_displacement(
    knots: dict[str, list[np.ndarray]], reference: dict[str, list[np.ndarray]]
) -> dict[str, float | None]:
    """Largest knot move per term, relative to the reference knot range."""
    records: dict[str, float | None] = {}
    for label, vectors in reference.items():
        ours = knots.get(label)
        if ours is None or [v.shape for v in ours] != [v.shape for v in vectors]:
            records[label] = None
            continue
        records[label] = max(
            float(np.max(np.abs(a - b)) / max(np.ptp(b), np.finfo(float).tiny))
            for a, b in zip(ours, vectors, strict=True)
        )
    return records


def _working_weights(family, y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """``bgam.fitd``'s working weights at ``eta`` (no prior weights)."""
    mu = np.asarray(family.link.inverse(eta))
    if family.execution_capabilities().extended_performance_iteration:
        # The dDeta Newton working weights (rho == 0) at the fitted theta.
        theta = float(np.exp(family.get_theta()[0]))
        weight, _ = family.source_newton_working(
            y, mu, eta, np.zeros_like(eta), np.ones_like(eta), theta
        )
        return np.asarray(weight)
    return np.asarray(family.link.mu_eta(eta)) ** 2 / np.asarray(family.variance(mu))


def discrete_edf(fit, level: _CompactLevel, train: pd.DataFrame, smooth_info):
    """Per-term EDF of a discrete fit at its discretized design.

    ``diag((X'WX + S)^-1 X'WX)`` with working weights at the final
    coefficients, the discretized design and the discrete model's penalties,
    summed per term. ``bgam.fitd`` reports the EDF of its last working model,
    whose weights come from the previous iterate, so for an iterated family
    the reconstructed total differs from the fit's by about the convergence
    tolerance; ``self_consistency`` records that, and the deviance at the
    discretized design against the fit's own deviance.
    """
    X = np.asarray(fit.predict_matrix(level.snapped))
    beta = np.asarray(fit.coefficients)
    eta = X @ beta
    y = train.y.to_numpy()
    weight = _working_weights(fit.family, y, eta)
    gram = X.T @ (weight[:, None] * X)
    penalty = level.penalties.materialize(np.log(np.asarray(fit.smoothing_params)))
    diagonal = np.diag(np.linalg.solve(gram + penalty, gram))
    per_term = [
        float(diagonal[info.first_coef : info.last_coef].sum()) for info in smooth_info
    ]
    deviance = _deviance(fit.family, y, np.asarray(fit.family.link.inverse(eta)))
    return per_term, {
        "harness_edf_total": float(diagonal.sum()),
        "fit_edf_total": float(fit.edf_total),
        "edf_total_relative_difference": _relative(
            float(diagonal.sum()), float(fit.edf_total)
        ),
        "harness_deviance": deviance,
        "fit_deviance": float(fit.deviance),
        "deviance_relative_difference": _relative(deviance, float(fit.deviance)),
    }


def _penalty_norms(penalties) -> np.ndarray:
    norms = np.zeros(penalties.n_penalties)
    for block in penalties.blocks:
        for sp_index, matrix in zip(
            block.sp_indices, block.dense_penalties(), strict=True
        ):
            norms[sp_index] = float(np.linalg.norm(matrix))
    return norms


def compare(
    view: FitView,
    reference: FitView,
    smooth_info,
    train: pd.DataFrame,
    test: pd.DataFrame,
    eta_test: np.ndarray,
    *,
    theta_reference: float | None,
) -> dict[str, Any]:
    """Method metrics of ``view`` against ``reference`` (same formula and rows)."""
    if len(view.fit.coefficients) != len(reference.fit.coefficients):
        raise RuntimeError("compared coefficient layouts differ")
    return {
        "log10_sp_ratio": np.log10(
            view.smoothing_params / reference.smoothing_params
        ).tolist(),
        "scale_relative_difference": _relative(view.scale, reference.scale),
        "theta_relative_difference": (
            None
            if view.theta is None or theta_reference is None
            else _relative(view.theta, theta_reference)
        ),
        "edf": {
            "per_term": [
                {
                    "label": info.label,
                    "reference": float(ref),
                    "discrete": float(value),
                    "difference": float(value - ref),
                }
                for info, ref, value in zip(
                    smooth_info, reference.edf, view.edf, strict=True
                )
            ],
            "total": {
                "reference": reference.edf_total,
                "discrete": view.edf_total,
                "difference": view.edf_total - reference.edf_total,
            },
        },
        "held_out": held_out_metrics(view, reference, test, eta_test),
        "partial_effects": partial_effect_metrics(view, reference, smooth_info, train),
    }


# --------------------------------------------------------------------------
# Pinned-R anchors (optional; typed rpy2 RBridge only).
# --------------------------------------------------------------------------

_R_GAM_FAMILIES = {
    "gaussian": "gaussian",
    "poisson": "poisson",
    "binomial": "binomial",
    "gamma": "gamma_log",
    "nb": "nb",
}
_R_BAM_FAMILIES = {
    "gaussian": ("gaussian", None),
    "poisson": ("poisson", None),
    "binomial": ("binomial", None),
    "gamma": ("gamma", "log"),
    "nb": ("nb", None),
    "nb_fixed": ("nb", None),
}


def _r_bridge() -> tuple[Any, str]:
    """A version-checked rpy2 bridge, or ``None`` and the reason."""
    try:
        from tests.r_bridge import RBridge
    except ImportError as error:
        return None, str(error)
    if not RBridge.available():
        return None, "rpy2 or R not available"
    ok, message = RBridge.check_versions()
    if not ok:
        return None, message
    return RBridge(mode="rpy2"), "R 4.5.2 / mgcv 1.9-3"


def r_gam_anchor(
    bridge, case: ApproximationCase, formula: str, train: pd.DataFrame, reference
) -> dict[str, Any]:
    """Pinned R ``gam(method="REML")`` against the dense JaxGAM reference."""
    family = _R_GAM_FAMILIES.get(case.family)
    if family is None:
        return {"status": "unsupported", "reason": "RBridge.fit_gam has no fixed nb"}
    r_fit = bridge.fit_gam(formula, train, family=family)
    fitted = np.asarray(reference.fit.predict(train, pred_type="response"))
    r_fitted = np.asarray(r_fit["fitted_values"])
    return {
        "status": "ok",
        "fitted_values_max_abs_difference": float(np.max(np.abs(fitted - r_fitted))),
        "fitted_values_max_relative_difference": float(
            np.max(
                np.abs(fitted - r_fitted)
                / np.maximum(np.abs(r_fitted), np.finfo(float).tiny)
            )
        ),
        "deviance_relative_difference": _relative(
            _deviance(reference.fit.family, train.y.to_numpy(), fitted),
            float(r_fit["deviance"]),
        ),
        "log10_sp_ratio": np.log10(
            reference.smoothing_params / np.asarray(r_fit["smoothing_params"])
        ).tolist(),
        "edf_difference": (reference.edf - np.asarray(r_fit["edf"])).tolist(),
        "theta_relative_difference": (
            None
            if r_fit.get("theta") is None or reference.theta is None
            else _relative(reference.theta, float(r_fit["theta"]))
        ),
    }


def r_bam_anchor(
    bridge,
    case: ApproximationCase,
    formula: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    m: int | None,
    sp: np.ndarray,
    theta: float | None,
    *,
    dense: FitView,
    discrete: FitView,
) -> dict[str, Any]:
    """Pinned ``bam(discrete=TRUE)`` at the same grid, sp and theta.

    Splits a discrete-versus-dense gap into each route's own difference from
    pinned mgcv: training fitted values and held-out exact (``predict.gam``)
    link predictions.
    """
    family, link = _R_BAM_FAMILIES[case.family]
    nb_theta = None
    if family == "nb":
        nb_theta = _NB_THETA if case.family == "nb_fixed" else theta
    r_fit = bridge.fit_bam_discrete(
        formula,
        train,
        family=family,
        link=link,
        sp=np.asarray(sp, dtype=float),
        factor_names=case.factor_names,
        m=m,
        newdata=test,
        theta=nb_theta,
        theta_fixed=nb_theta is not None,
    )
    r_fitted = np.asarray(r_fit.fitted_values)
    r_link = np.asarray(r_fit.prediction)

    def against(view: FitView) -> dict[str, float]:
        fitted = np.asarray(view.fit.predict(train, pred_type="response"))
        link_prediction = np.asarray(view.fit.predict(test, pred_type="link"))
        return {
            "fitted_values_max_abs_difference": float(
                np.max(np.abs(fitted - r_fitted))
            ),
            "held_out_link_max_abs_difference": float(
                np.max(np.abs(link_prediction - r_link))
            ),
        }

    return {
        "status": "ok",
        "r_converged": bool(r_fit.mgcv_converged),
        "r_edf_total": float(r_fit.edf_total),
        "discrete": {
            **against(discrete),
            "edf_total_difference": discrete.edf_total - float(r_fit.edf_total),
        },
        "dense": {
            **against(dense),
            "edf_total_difference": dense.edf_total - float(r_fit.edf_total),
        },
    }


# --------------------------------------------------------------------------
# One case.
# --------------------------------------------------------------------------


def _reference_record(view: FitView, test: pd.DataFrame, seconds: float) -> dict:
    fit = view.fit
    eta = np.asarray(fit.predict(test, pred_type="link"))
    return {
        "converged": bool(fit.converged),
        "fit_seconds": seconds,
        "smoothing_params": view.smoothing_params.tolist(),
        "edf": view.edf.tolist(),
        "edf_total": view.edf_total,
        "scale": view.scale,
        "theta": view.theta,
        "test_deviance": _deviance(
            fit.family, test.y.to_numpy(), np.asarray(fit.family.link.inverse(eta))
        ),
    }


def _fit_status(fit, seconds: float) -> dict[str, Any]:
    return {
        "status": "ok",
        "converged": bool(fit.converged),
        "n_iter": int(fit.n_iter),
        "convergence_info": str(fit.convergence_info),
        "fit_seconds": seconds,
        "smoothing_params": np.asarray(fit.smoothing_params).tolist(),
        "scale": float(fit.scale),
        "theta": None if fit.theta is None else float(fit.theta),
    }


def _ordered_levels(levels: Sequence[str]) -> list[str]:
    """Compute ``exact`` first so other levels can use it as a reference."""
    return sorted(levels, key=lambda level: level != "exact")


def run_case(case: ApproximationCase, root: Path) -> dict[str, Any]:
    """Run one family/size case in this process (a child of ``run_isolated``)."""
    import jax

    import jaxgam

    loaded = loaded_source_identity(jaxgam, root)
    case_start = time.perf_counter()
    formula = MODELS[case.model]
    train, _ = make_data(case, test=False)
    test, eta_test = make_data(case, test=True)

    reml_fit, reml_seconds = _timed(lambda: _fit_dense(case, formula, train))
    reml = dense_view(reml_fit)
    smooth_info = reml_fit.smooth_info
    sp_reference = reml.smoothing_params
    theta_reference = _NB_THETA if case.family == "nb_fixed" else reml.theta
    references = {"working_model_reml": reml, "exact_reml": reml}
    record: dict[str, Any] = {
        "case": asdict(case),
        "formula": formula,
        "input": {
            "generator": f"{case.model}_{case.family}",
            "seed": case.seed,
            "nb_theta": _NB_THETA if case.family.startswith("nb") else None,
            "train_sha256": _frame_sha256(train),
            "test_sha256": _frame_sha256(test),
            "unique_values": {
                name: int(train[name].nunique()) for name in ("x0", "x1", "x2")
            },
        },
        "reference": {"reml": _reference_record(reml, test, reml_seconds)},
    }
    if set(FIXED_MODES) & set(case.modes):
        fixed_fit, fixed_seconds = _timed(
            lambda: _fit_dense(
                case, formula, train, sp=sp_reference, theta=theta_reference
            )
        )
        references["fixed_sp"] = dense_view(fixed_fit)
        references["fixed_sp_matched"] = references["fixed_sp"]
        record["reference"]["fixed_sp"] = _reference_record(
            references["fixed_sp"], test, fixed_seconds
        )

    bridge, bridge_status = _r_bridge() if case.r_reference else (None, "")
    record["r_anchors"] = {
        "status": (
            "not_requested"
            if not case.r_reference
            else "ok"
            if bridge is not None
            else "unavailable"
        ),
        "versions_or_reason": bridge_status,
        "gam_reml": (
            None if bridge is None else r_gam_anchor(bridge, case, formula, train, reml)
        ),
    }

    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    dense_setup = ModelSetup.build(parse_formula(formula), train)
    dense_norms = _penalty_norms(dense_setup.penalties)
    dense_knots = term_knots(dense_setup.coef_map)

    exact_views: dict[str, FitView] = {}
    level_records: dict[str, dict[str, Any]] = {}
    for level_name in _ordered_levels(case.levels):
        m = level_m(level_name, case.n_rows)
        try:
            level = compact_level(case, formula, train, m)
        except (MemoryError, RuntimeError, ValueError) as error:
            # For example a grid coarser than a smooth's basis dimension.
            failure = {"status": "error", "error": f"{type(error).__name__}: {error}"}
            level_records[level_name] = {
                "level": level_name,
                "m": m,
                "covariates": {},
                "penalty_frobenius_ratio": None,
                "knot_displacement_over_range": None,
                "modes": dict.fromkeys(case.modes, failure),
            }
            continue
        penalty_ratio = _penalty_norms(level.penalties) / dense_norms
        level_record: dict[str, Any] = {
            "level": level_name,
            "m": m,
            "covariates": level.covariates,
            "penalty_frobenius_ratio": penalty_ratio.tolist(),
            "knot_displacement_over_range": knot_displacement(level.knots, dense_knots),
            "modes": {},
        }
        for mode in case.modes:
            fixed = mode in FIXED_MODES
            sp = None
            if fixed:
                sp = sp_reference
                if mode == "fixed_sp_matched":
                    sp = sp_reference / penalty_ratio
            try:
                fit, seconds = _timed(
                    lambda m=m, mode=mode, sp=sp, fixed=fixed: _fit_discrete(
                        case,
                        formula,
                        train,
                        m,
                        mode,
                        sp=sp,
                        theta=theta_reference if fixed else None,
                    )
                )
            except NotImplementedError as error:
                level_record["modes"][mode] = {
                    "status": "unsupported",
                    "error": f"{type(error).__name__}: {error}",
                }
                continue
            except (ArithmeticError, MemoryError, RuntimeError, ValueError) as error:
                level_record["modes"][mode] = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                continue
            per_term, consistency = discrete_edf(fit, level, train, smooth_info)
            view = discrete_view(fit, per_term)
            mode_record = _fit_status(fit, seconds)
            mode_record["self_consistency"] = consistency
            mode_record["vs_dense"] = compare(
                view,
                references[mode],
                smooth_info,
                train,
                test,
                eta_test,
                theta_reference=theta_reference,
            )
            if level_name == "exact":
                exact_views[mode] = view
                if mode == "fixed_sp" and bridge is not None:
                    mode_record["r_bam_anchor"] = r_bam_anchor(
                        bridge,
                        case,
                        formula,
                        train,
                        test,
                        m,
                        sp_reference,
                        theta_reference,
                        dense=references[mode],
                        discrete=view,
                    )
            elif mode in exact_views:
                mode_record["vs_exact_level"] = compare(
                    view,
                    exact_views[mode],
                    smooth_info,
                    train,
                    test,
                    eta_test,
                    theta_reference=exact_views[mode].theta,
                )
            level_record["modes"][mode] = mode_record
        level_records[level_name] = level_record
    record["levels"] = [level_records[name] for name in case.levels]
    record["case_seconds"] = time.perf_counter() - case_start
    record["runtime"] = {
        **source_identity(root),
        **loaded,
        **harness_identity(),
        **_runtime_metadata(jax, jaxgam),
    }
    return record


def run_isolated(case: ApproximationCase, root: Path, parent: dict) -> dict:
    """Run one case in a fresh child process with a task-owned JAX cache."""
    with isolated_cache_dir() as cache:
        environment = os.environ | {
            "JAX_COMPILATION_CACHE_DIR": cache,
            "PYTHONPATH": str(root),
        }
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                json.dumps(asdict(case)),
                "--checkout",
                str(root.resolve()),
            ],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(
            f"child failed for {case.family} n={case.n_rows}:\n{completed.stderr}"
        )
    record = json.loads(completed.stdout)
    require_matching_child(parent, record["runtime"])
    return record


# --------------------------------------------------------------------------
# Summary.
# --------------------------------------------------------------------------


def _max_shape(metrics: dict) -> float:
    return max(p["relative_l2"] for p in metrics["partial_effects"])


def _min_curve_coverage(metrics: dict) -> float:
    return min(p["reference_curve_coverage"] for p in metrics["partial_effects"])


def summary_rows(payload: dict[str, Any], against: str = "vs_dense") -> list[dict]:
    """One row per case, mode and level with the headline metrics.

    ``against`` is ``"vs_dense"`` (the undiscretized dense reference) or
    ``"vs_exact_level"`` (the same strategy at the exact level).
    """
    rows = []
    for record in payload["cases"]:
        case = record["case"]
        for level in record["levels"]:
            covariate_error = max(
                (c["max_abs_error_over_range"] for c in level["covariates"].values()),
                default=0.0,
            )
            for mode, metrics in level["modes"].items():
                if against not in metrics and (
                    metrics["status"] == "ok" or against != "vs_dense"
                ):
                    continue
                row: dict[str, Any] = {
                    "model": case["model"],
                    "family": case["family"],
                    "n": case["n_rows"],
                    "mode": mode,
                    "level": level["level"],
                    "covariate_error": covariate_error,
                    "status": metrics["status"],
                }
                if metrics["status"] == "ok":
                    comparison = metrics[against]
                    held = comparison["held_out"]
                    row.update(
                        {
                            "link_max": held["link_difference"]["max_abs"],
                            "link_rms": held["link_difference"]["rms"],
                            "link_max_over_se": held[
                                "link_difference_over_reference_se"
                            ]["max_abs"],
                            "deviance_relative": held["deviance"][
                                "relative_difference"
                            ],
                            "edf_total_difference": comparison["edf"]["total"][
                                "difference"
                            ],
                            "shape_relative_l2": _max_shape(comparison),
                            "se_ratio_median": held["intervals"]["se_ratio"]["median"],
                            "coverage_of_reference": held["intervals"][
                                "coverage_of_reference_estimate"
                            ],
                            "curve_coverage_min": _min_curve_coverage(comparison),
                        }
                    )
                rows.append(row)
    return rows


def implementation_rows(payload: dict[str, Any]) -> list[dict]:
    """Exact-level fixed-sp checks and the optional pinned-R anchors."""
    rows = []
    for record in payload["cases"]:
        case = record["case"]
        gam = (record.get("r_anchors") or {}).get("gam_reml") or {}
        for level in record["levels"]:
            metrics = level["modes"].get("fixed_sp", {})
            if level["level"] != "exact" or metrics.get("status") != "ok":
                continue
            anchor = metrics.get("r_bam_anchor") or {}
            rows.append(
                {
                    "model": case["model"],
                    "family": case["family"],
                    "n": case["n_rows"],
                    "discrete_vs_dense_link_max": metrics["vs_dense"]["held_out"][
                        "link_difference"
                    ]["max_abs"],
                    "discrete_vs_bam": (anchor.get("discrete") or {}).get(
                        "held_out_link_max_abs_difference"
                    ),
                    "dense_vs_bam": (anchor.get("dense") or {}).get(
                        "held_out_link_max_abs_difference"
                    ),
                    "dense_vs_gam_fitted": gam.get("fitted_values_max_abs_difference"),
                    "edf_self_consistency": metrics["self_consistency"][
                        "edf_total_relative_difference"
                    ],
                    "deviance_self_consistency": metrics["self_consistency"][
                        "deviance_relative_difference"
                    ],
                }
            )
    return rows


def _number(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def format_summary(payload: dict[str, Any]) -> str:
    """Markdown tables of ``summary_rows`` and ``implementation_rows``."""
    header = (
        "| model | family | n | mode | level | max cov. err / range "
        "| max abs d eta | rms d eta | max d eta / se | rel. d test dev | d EDF "
        "| max shape rel. L2 | median SE ratio | ref. covered | min curve cov. |"
    )
    sections = []
    for against, title in (
        ("vs_dense", "Against the undiscretized dense fit"),
        ("vs_exact_level", "Against the same strategy at the exact level"),
    ):
        lines = [f"### {title}", "", header, "|" + " --- |" * 15]
        for row in summary_rows(payload, against):
            prefix = (
                f"| {row['model']} | {row['family']} | {row['n']} | {row['mode']} "
                f"| {row['level']} | {row['covariate_error']:.1e} "
            )
            if row["status"] != "ok":
                lines.append(prefix + f"| {row['status']} |" + " |" * 9)
                continue
            lines.append(
                prefix + f"| {row['link_max']:.1e} | {row['link_rms']:.1e} "
                f"| {row['link_max_over_se']:.2g} | {row['deviance_relative']:.1e} "
                f"| {row['edf_total_difference']:+.2f} "
                f"| {row['shape_relative_l2']:.1e} "
                f"| {_number(row['se_ratio_median'], '.3f')} "
                f"| {row['coverage_of_reference']:.3f} "
                f"| {row['curve_coverage_min']:.3f} |"
            )
        sections.append("\n".join(lines))
    lines = [
        "### Exact-level implementation checks (fixed sp)",
        "",
        "| model | family | n | discrete vs dense | discrete vs bam | dense vs bam "
        "| dense vs gam fitted | EDF self-cons. | deviance self-cons. |",
        "|" + " --- |" * 9,
    ]
    lines.extend(
        f"| {row['model']} | {row['family']} | {row['n']} "
        f"| {row['discrete_vs_dense_link_max']:.1e} "
        f"| {_number(row['discrete_vs_bam'], '.1e')} "
        f"| {_number(row['dense_vs_bam'], '.1e')} "
        f"| {_number(row['dense_vs_gam_fitted'], '.1e')} "
        f"| {row['edf_self_consistency']:.1e} "
        f"| {row['deviance_self_consistency']:.1e} |"
        for row in implementation_rows(payload)
    )
    sections.append("\n".join(lines))
    return "\n\n".join(sections)


# --------------------------------------------------------------------------
# Command line.
# --------------------------------------------------------------------------


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--models", default="study", help="comma-separated models")
    parser.add_argument("--n", default="10000", help="comma-separated row counts")
    parser.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--n-test", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_261_025)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument(
        "--r-reference-max-n",
        type=int,
        default=20_000,
        help="also fit pinned R anchors when n is at most this (0 disables)",
    )
    parser.add_argument("--output", type=Path, help="JSON path outside the checkout")
    parser.add_argument("--expect-commit")
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--summarize", type=Path, help="print a results table")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = (args.checkout or Path(__file__).resolve().parents[1]).resolve()

    if args.summarize is not None:
        print(format_summary(json.loads(args.summarize.read_text())))
        return
    if args.child is not None:
        case_fields = json.loads(args.child)
        record = run_case(ApproximationCase(**case_fields), root)
        print(json.dumps(record, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required (a path outside the checkout)")
    output = require_output_outside(args.output, root)
    identity = require_clean_checkout(root, args.expect_commit)
    parent = {**identity, **harness_identity()}
    cases = [
        ApproximationCase(
            family=family,
            n_rows=int(n),
            n_test=args.n_test,
            seed=args.seed,
            levels=tuple(_split(args.levels)),
            modes=tuple(_split(args.modes)),
            model=model,
            r_reference=int(n) <= args.r_reference_max_n,
            batch_rows=args.batch_rows,
        )
        for model in _split(args.models)
        for family in _split(args.families)
        for n in _split(args.n)
    ]
    payload: dict[str, Any] = {
        "study": "pr11_discrete_approximation",
        "argv": list(sys.argv if argv is None else argv),
        "provenance": parent,
        "cases": [],
    }
    for case in cases:
        print(
            f"case {case.model} {case.family} n={case.n_rows}",
            file=sys.stderr,
            flush=True,
        )
        payload["cases"].append(run_isolated(case, root, parent))
        output.write_text(json.dumps(payload, sort_keys=True) + "\n")
    print(format_summary(payload))
    print(f"results: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
