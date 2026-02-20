"""Fitting algorithms: PIRLS, REML, Newton optimizer (Phase 2 — JAX)."""

from pymgcv.fitting.data import FittingData
from pymgcv.fitting.initialization import initialize_beta
from pymgcv.fitting.pirls import PIRLSResult, pirls_loop

__all__ = ["FittingData", "initialize_beta", "PIRLSResult", "pirls_loop"]
