"""Fitting algorithms: PIRLS, REML, Newton optimizer (Phase 2 -- JAX)."""

from pymgcv.fitting.data import FittingData
from pymgcv.fitting.initialization import initialize_beta
from pymgcv.fitting.newton import NewtonOptimizer, NewtonResult, newton_optimize
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
    "NewtonOptimizer",
    "NewtonResult",
    "PIRLSResult",
    "REMLCriterion",
    "REMLResult",
    "initialize_beta",
    "ml_criterion",
    "newton_optimize",
    "pirls_loop",
    "reml_criterion",
]
