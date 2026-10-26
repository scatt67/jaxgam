"""Validated execution and retention controls for :class:`jaxgam.GAM`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Literal

from jaxgam.data.discretize import DiscretizationBudget


@dataclass(frozen=True)
class EFSControl:
    """Pinned dense Fellner--Schall controller limits and tolerances."""

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

    The fitted prediction policy deliberately evaluates new values against the
    frozen basis metadata.  ``predict.bamd`` request-wise recompression is a
    separate PR11 component.
    """

    m: int | tuple[int, ...] | None = None
    factor_names: tuple[str, ...] = ()
    budget: DiscretizationBudget = field(default_factory=DiscretizationBudget)
    selector_memmap_directory: Path | None = None
    pair_policy: Literal["histogram_or_direct", "direct_only"] = "histogram_or_direct"
    max_pair_histogram_bytes: int = 32 * 1024 * 1024
    seed: int = 8547
    prediction_policy: Literal["continuous_frozen_basis"] = "continuous_frozen_basis"

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
        if self.prediction_policy != "continuous_frozen_basis":
            raise ValueError("prediction_policy must be 'continuous_frozen_basis'.")


@dataclass(frozen=True)
class FitControl:
    """Explicit resource policy for prediction retention and batching.

    ``uncertainty='none'`` is deliberately the default for prediction-only
    results.  It retains no covariance.  ``'fisher'`` retains a Cholesky
    factor for exact requested-row standard errors; ``'covariance'`` also
    materializes the public dense covariance, subject to ``memory_budget_bytes``.

    The dense fitting backend is not memory-budgeted: these limits apply to
    prediction matrices/SE workspace and explicit covariance materialization.
    The streamed backend additionally rejects known live PIRLS workspaces
    (coefficient reductions plus one design batch) above
    ``memory_budget_bytes``. Phase-1 streamed preparation scans at most
    ``batch_rows`` rows at a time and shrinks that size until its prospective
    setup workspace (JaxGAM-owned arrays plus the exact-knot SQLite scope) fits
    ``memory_budget_bytes``; a budget too small for one row is rejected before
    any full-data scan. Source-owned storage and allocator overhead remain
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
    """

    execution: Literal["dense", "stream", "discrete"] = "dense"
    batch_rows: int = 65_536
    memory_budget_bytes: int = 512 * 1024 * 1024
    output_budget_bytes: int = 512 * 1024 * 1024
    uncertainty: Literal["none", "fisher", "covariance"] = "none"
    linear_solver: Literal["cholesky", "qr"] = "cholesky"
    gaussian_compression: bool = False
    efs: EFSControl = field(default_factory=EFSControl)
    discretization: ContinuousDiscretizationControl | None = None

    def __post_init__(self) -> None:
        if self.execution not in ("dense", "stream", "discrete"):
            raise ValueError("execution must be 'dense', 'stream', or 'discrete'.")
        for name in ("batch_rows", "memory_budget_bytes", "output_budget_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.uncertainty not in ("none", "fisher", "covariance"):
            raise ValueError("uncertainty must be 'none', 'fisher', or 'covariance'.")
        if self.linear_solver not in ("cholesky", "qr"):
            raise ValueError("linear_solver must be 'cholesky' or 'qr'.")
        if not isinstance(self.gaussian_compression, bool):
            raise ValueError("gaussian_compression must be a boolean.")
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
