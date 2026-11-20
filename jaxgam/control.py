"""Validated execution and retention controls for :class:`jaxgam.GAM`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

from jaxgam.data.discretize import DiscretizationBudget

PREDICTION_POLICIES = ("continuous_frozen_basis", "mgcv_predict_bamd")


@dataclass(frozen=True)
class EFSControl:
    """Limits and tolerances of the extended Fellner--Schall optimizer.

    Read by ``optimizer="efs"`` on the dense, streamed and operator routes.
    The numerical defaults are those of mgcv's ``gam.control`` (``maxit=200``,
    ``epsilon=1e-7``, ``efs.lspmax=15``, ``efs.tol=0.1``).
    """

    outer_limit: int = 200
    log_lambda_max: float = 15.0
    score_tolerance: float = 0.1
    pirls_tolerance: float = 1e-7
    pirls_max_iter: int = 200
    history_limit: int = 200

    def __post_init__(self) -> None:
        for name in ("outer_limit", "pirls_max_iter", "history_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"EFS {name} must be a positive integer")
        for name in ("log_lambda_max", "score_tolerance", "pirls_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"EFS {name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"EFS {name} must be finite")
        if self.score_tolerance < 0:
            raise ValueError("EFS score_tolerance must be nonnegative")
        if self.pirls_tolerance <= 0:
            raise ValueError("EFS pirls_tolerance must be positive")


@dataclass(frozen=True)
class ContinuousDiscretizationControl:
    """Explicit Phase-1 grid and indexed-action policy.

    ``prediction_policy`` selects, and the fitted result records, how new
    data are evaluated.  Both policies reuse the fitted knots, constraints
    and transforms; neither refits a basis.

    ``"continuous_frozen_basis"`` (default)
        Every new value is evaluated exactly by the frozen basis, like
        ``predict.gam``.  Rows are independent, so results never depend on
        the prediction request or its batching.
    ``"mgcv_predict_bamd"``
        Pinned ``predict.bamd`` semantics: the *whole* prediction request is
        first compressed with ``discrete.mf``'s default policy
        (``compress.df`` with ``m=NULL``: 1000/100/25 range-grid points per
        metric for one/two/three-plus jointly discretized variables, rounding
        only when a group has more unique rows than that grid), and the frozen
        basis is then evaluated on the compressed values.  The request grid is
        frozen by a bounded pre-pass before any batch is evaluated or
        yielded, so results are invariant to prediction batch boundaries but
        do depend on the composition of the request, exactly as in mgcv.
        As in ``predict.bam``, a model without smooth terms has no discrete
        prediction information and is evaluated exactly; its result then
        records ``prediction_policy="continuous_frozen_basis"`` and keeps the
        request in ``discretization.requested_prediction_policy``.  Only metric
        discretization changes; factor semantics are the default path's: a
        missing or unseen random-effect level contributes zero (100% weight
        on the prior, an intended divergence from mgcv's ``NA``), a missing
        or unseen ``by`` level gives ``NaN`` with a warning, and a parametric
        factor gives ``NaN`` for a missing value and rejects unseen levels.

    ``max_retained_representatives`` bounds the per-group representative
    values copied into the fitted result's discretization metadata.  Groups
    with more compact rows retain only their bounded descriptor (range, grid
    resolution, rounding decision and row count).
    """

    m: int | tuple[int, ...] | None = None
    factor_names: tuple[str, ...] = ()
    budget: DiscretizationBudget = field(default_factory=DiscretizationBudget)
    selector_memmap_directory: Path | None = None
    pair_policy: Literal["histogram_or_direct", "direct_only"] = "histogram_or_direct"
    max_pair_histogram_bytes: int = 32 * 1024 * 1024
    seed: int = 8547
    prediction_policy: Literal["continuous_frozen_basis", "mgcv_predict_bamd"] = (
        "continuous_frozen_basis"
    )
    max_retained_representatives: int = 10_000

    def __post_init__(self) -> None:
        if isinstance(self.m, tuple):
            if not self.m or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 2
                for value in self.m
            ):
                raise ValueError("m tuple entries must be integers of at least 2.")
        elif self.m is not None and (
            isinstance(self.m, bool) or not isinstance(self.m, int) or self.m < 2
        ):
            raise ValueError("m must be an integer of at least 2, a tuple, or None.")
        if not isinstance(self.factor_names, tuple) or not all(
            isinstance(name, str) and name for name in self.factor_names
        ):
            raise ValueError("factor_names must be a tuple of non-empty names.")
        if len(self.factor_names) != len(set(self.factor_names)):
            raise ValueError("factor_names must not contain duplicates.")
        if not isinstance(self.budget, DiscretizationBudget):
            raise TypeError("budget must be a DiscretizationBudget.")
        if self.pair_policy not in {"histogram_or_direct", "direct_only"}:
            raise ValueError(
                "pair_policy must be 'histogram_or_direct' or 'direct_only'."
            )
        if (
            isinstance(self.max_pair_histogram_bytes, bool)
            or not isinstance(self.max_pair_histogram_bytes, int)
            or self.max_pair_histogram_bytes <= 0
        ):
            raise ValueError("max_pair_histogram_bytes must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        if self.prediction_policy not in PREDICTION_POLICIES:
            raise ValueError(
                "prediction_policy must be 'continuous_frozen_basis' or "
                "'mgcv_predict_bamd'."
            )
        if (
            isinstance(self.max_retained_representatives, bool)
            or not isinstance(self.max_retained_representatives, int)
            or self.max_retained_representatives < 0
        ):
            raise ValueError(
                "max_retained_representatives must be a non-negative integer."
            )


@dataclass(frozen=True)
class OperatorControl:
    """Explicit limits for the operator route.

    They bound its coefficient fit and conjugate-gradient solves, with fixed
    smoothing parameters or under ``optimizer="efs"``.
    """

    max_modest_coefficients: int = 64
    max_iterations: int = 200
    tolerance: float = 1e-7
    cg_max_iterations: int = 1000
    cg_rtol: float = 1e-10
    cg_atol: float = 1e-12
    use_coarse: bool = False

    def __post_init__(self) -> None:
        for name in ("max_modest_coefficients", "max_iterations", "cg_max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"Operator {name} must be a positive integer.")
        for name in ("tolerance", "cg_rtol", "cg_atol"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"Operator {name} must be finite and nonnegative.")
        if self.tolerance <= 0:
            raise ValueError("Operator tolerance must be strictly positive.")
        if self.cg_rtol == 0 and self.cg_atol == 0:
            raise ValueError("Operator CG requires a positive residual tolerance.")
        if type(self.use_coarse) is not bool:
            raise ValueError("Operator use_coarse must be boolean.")


@dataclass(frozen=True)
class FitControl:
    """Execution route and explicit resource policy for a fit.

    ``execution`` selects the route; the other fields set batching, memory
    budgets and what a prediction result retains.

    ``uncertainty='none'`` is deliberately the default for prediction-only
    results.  It retains no covariance.  ``'fisher'`` retains a triangular
    factor (a Cholesky factor, or a pivoted QR factor on the QR-solver
    routes) for exact requested-row standard errors; ``'covariance'`` also
    materializes the public dense covariance, subject to ``memory_budget_bytes``.

    The dense fitting backend is not memory-budgeted: these limits apply to
    prediction matrices/SE workspace and explicit covariance materialization.
    The streamed backend additionally rejects known live PIRLS workspaces
    (coefficient reductions plus one design batch) above
    ``memory_budget_bytes``. Phase-1 streamed preparation scans at most
    ``batch_rows`` rows at a time and shrinks that size until its prospective
    setup workspace (JaxGAM-owned arrays plus the exact-knot SQLite scope) fits
    ``memory_budget_bytes``; a budget too small for one row is rejected before
    any full-data scan. When the budget also holds them at that size, exact
    cubic knots are found from covariate columns gathered in memory (8 bytes
    per row per column, as many columns per scan as fit) instead of the
    row-count-independent SQLite scan; the knots are identical either way.
    Source-owned storage and allocator overhead remain
    outside these bounds. Gaussian QR adds a route-specific ledger for its
    merge buffers and compact score/derivative arrays, including either the
    small-penalty Newton Hessian or the bounded large-penalty L-BFGS-B work
    arrays. Native LAPACK/XLA workspace is not an allocator hard cap.

    ``gaussian_compression=True`` opts a Gaussian identity-link fit into one
    weighted QR reduction followed by coefficient-space REML trials.  With
    ``execution='stream'`` the reduction consumes one bounded replay after
    prepared setup and retains no training rows.  The initial route requires
    finite weights in the PIRLS unclipped interval and a full identifiable
    coefficient map.  Ineligible dense models use the ordinary dense route and
    report the reason in ``execution_fallback_reason``.

    ``execution='discrete'`` fits mgcv's compact discretized design
    (``bam(discrete=TRUE)``) from a replayable source under the explicit
    ``discretization`` policy. It combines retained compact setup, indexed
    action, and solver workspaces in a conservative fit peak; its Phase-1
    construction allocations are separately bounded by the discretization
    policy. The discrete route owns its solves, so ``linear_solver='qr'`` is
    rejected there, and a ``gaussian_compression`` request is reported unmet
    in ``execution_fallback_reason``.

    ``smoothing_strategy='working_model_reml'`` (discrete only) selects the
    smoothing parameters by pinned mgcv ``bam(discrete=TRUE)``'s performance
    iteration (``bgam.fitd``): one limited ``Sl.fitChol`` Newton proposal on
    each iteration's working model, stopped when the penalized deviance (and
    an estimated log scale) changes by less than ``working_model_epsilon``
    within ``working_model_max_iterations``. Gaussian identity models use one
    frozen working model; Poisson, Binomial and Gamma re-form a Fisher
    working model (IRLS weights, no observed-curvature split) at every
    iteration, with the scale fixed at one or, for Gamma, estimated. The
    negative binomial uses ``bgam.fitd``'s ``dDeta`` Newton working model and,
    when theta is estimated, re-estimates it each iteration by
    ``estimate.theta`` at the current mean. This is not the nested exact REML
    optimization of the dense route. An explicit ``sp`` instead runs the same
    iteration with every smoothing parameter fixed.
    """

    execution: Literal["dense", "stream", "operator", "discrete"] = "dense"
    batch_rows: int = 65_536
    memory_budget_bytes: int = 512 * 1024 * 1024
    output_budget_bytes: int = 512 * 1024 * 1024
    uncertainty: Literal["none", "fisher", "covariance"] = "none"
    linear_solver: Literal["cholesky", "qr", "pcg"] = "cholesky"
    gaussian_compression: bool = False
    efs: EFSControl = field(default_factory=EFSControl)
    operator: OperatorControl = field(default_factory=OperatorControl)
    discretization: ContinuousDiscretizationControl | None = None
    smoothing_strategy: Literal["exact_reml", "working_model_reml"] = "exact_reml"
    working_model_max_iterations: int = 200
    working_model_epsilon: float = 1e-7

    def __post_init__(self) -> None:
        if self.execution not in ("dense", "stream", "operator", "discrete"):
            raise ValueError(
                "execution must be 'dense', 'stream', 'operator', or 'discrete'."
            )
        for name in ("batch_rows", "memory_budget_bytes", "output_budget_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.uncertainty not in ("none", "fisher", "covariance"):
            raise ValueError("uncertainty must be 'none', 'fisher', or 'covariance'.")
        if self.linear_solver not in ("cholesky", "qr", "pcg"):
            raise ValueError("linear_solver must be 'cholesky', 'qr', or 'pcg'.")
        if self.smoothing_strategy not in ("exact_reml", "working_model_reml"):
            raise ValueError("Unsupported smoothing_strategy.")
        if (
            self.smoothing_strategy == "working_model_reml"
            and self.execution != "discrete"
        ):
            raise ValueError("working_model_reml requires execution='discrete'.")
        if (
            isinstance(self.working_model_max_iterations, bool)
            or not isinstance(self.working_model_max_iterations, int)
            or self.working_model_max_iterations < 1
        ):
            raise ValueError("working_model_max_iterations must be a positive integer.")
        if (
            isinstance(self.working_model_epsilon, bool)
            or not isinstance(self.working_model_epsilon, Real)
            or not math.isfinite(self.working_model_epsilon)
            or self.working_model_epsilon <= 0.0
        ):
            raise ValueError("working_model_epsilon must be positive and finite.")
        if not isinstance(self.gaussian_compression, bool):
            raise ValueError("gaussian_compression must be a boolean.")
        if not isinstance(self.operator, OperatorControl):
            raise ValueError("operator must be an OperatorControl instance.")
        if not isinstance(self.efs, EFSControl):
            raise ValueError("efs must be an EFSControl instance.")
        if self.discretization is not None and not isinstance(
            self.discretization, ContinuousDiscretizationControl
        ):
            raise TypeError(
                "discretization must be a ContinuousDiscretizationControl or None."
            )
        if self.execution == "discrete" and self.discretization is None:
            raise ValueError(
                "execution='discrete' requires an explicit discretization policy."
            )
        if self.execution != "discrete" and self.discretization is not None:
            raise ValueError("A discretization policy requires execution='discrete'.")
