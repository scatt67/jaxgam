"""Validated execution and retention controls for :class:`jaxgam.GAM`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FitControl:
    """Explicit resource policy for prediction retention and batching.

    ``uncertainty='none'`` is deliberately the default for prediction-only
    results.  It retains no covariance.  ``'fisher'`` retains a Cholesky
    factor for exact requested-row standard errors; ``'covariance'`` also
    materializes the public dense covariance, subject to ``memory_budget_bytes``.

    The current dense fitting backend is not memory-budgeted: these limits
    apply to prediction matrices/SE workspace and explicit covariance
    materialization only.  A future streamed execution backend will extend
    the policy to fitting reductions.
    """

    execution: Literal["dense"] = "dense"
    batch_rows: int = 65_536
    memory_budget_bytes: int = 512 * 1024 * 1024
    output_budget_bytes: int = 512 * 1024 * 1024
    uncertainty: Literal["none", "fisher", "covariance"] = "none"
    gaussian_compression: bool = False

    def __post_init__(self) -> None:
        if self.execution != "dense":
            raise NotImplementedError(
                "Only execution='dense' is available until streamed execution lands."
            )
        for name in ("batch_rows", "memory_budget_bytes", "output_budget_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.uncertainty not in ("none", "fisher", "covariance"):
            raise ValueError("uncertainty must be 'none', 'fisher', or 'covariance'.")
        if not isinstance(self.gaussian_compression, bool):
            raise ValueError("gaussian_compression must be a boolean.")
        if self.gaussian_compression:
            raise NotImplementedError(
                "gaussian_compression=True is not available for dense execution."
            )
