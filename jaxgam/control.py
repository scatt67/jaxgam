"""Validated execution and retention controls for :class:`jaxgam.GAM`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal


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
    ``memory_budget_bytes``. CPU basis preparation remains outside that
    conservative accounting.
    """

    execution: Literal["dense", "stream"] = "dense"
    batch_rows: int = 65_536
    memory_budget_bytes: int = 512 * 1024 * 1024
    output_budget_bytes: int = 512 * 1024 * 1024
    uncertainty: Literal["none", "fisher", "covariance"] = "none"
    linear_solver: Literal["cholesky", "qr"] = "cholesky"
    gaussian_compression: bool = False
    efs: EFSControl = field(default_factory=EFSControl)

    def __post_init__(self) -> None:
        if self.execution not in ("dense", "stream"):
            raise ValueError("execution must be 'dense' or 'stream'.")
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
        if self.gaussian_compression:
            raise NotImplementedError(
                "gaussian_compression=True is not available for this execution path."
            )
