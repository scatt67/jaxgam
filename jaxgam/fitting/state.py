"""Observation-independent state produced by streamed fitting.

The streamed execution path deliberately keeps only coefficient-space arrays.
Row-aligned fitted values, working weights, and responses remain owned by a
``RowSource`` and are recomputed in a later scan when they are requested.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class StreamFitState:
    """Final coefficient-space state for a fixed-smoothing streamed fit."""

    coefficients: jax.Array
    log_lambda: jax.Array
    deviance: jax.Array
    penalized_deviance: jax.Array
    scale: jax.Array
    score_scale: jax.Array
    saturated_loglik: jax.Array
    edf: jax.Array
    xtwx: jax.Array
    xtwx_fisher: jax.Array
    factor: jax.Array
    fisher_factor: jax.Array
    n_iter: int
    converged: bool
    line_search_failed: bool
    backtracks: int
    stationarity: float
    source_scans: int
    batches_scanned: int
