"""Fitting algorithms: PIRLS, REML, Newton optimizer (Phase 2 -- JAX)."""

from pymgcv.fitting.data import FittingData
from pymgcv.fitting.initialization import initialize_beta
from pymgcv.fitting.pirls import PIRLSResult, pirls_loop
from pymgcv.fitting.reml import (
    MLCriterion,
    REMLCriterion,
    REMLResult,
    ml_criterion,
    reml_criterion,
)

__all__ = [
    "FittingData",
    "MLCriterion",
    "PIRLSResult",
    "REMLCriterion",
    "REMLResult",
    "initialize_beta",
    "ml_criterion",
    "pirls_loop",
    "reml_criterion",
]
