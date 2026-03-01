"""Fitting algorithms: PIRLS, REML, Newton optimizer (Phase 2 -- JAX).

Key entry points:

- ``newton_optimize``: Full outer Newton loop for smoothing parameter selection.
- ``pirls_loop``: Inner PIRLS loop for fixed smoothing parameters.
- ``FittingData``: Phase 1→2 boundary container.
- ``REMLCriterion`` / ``MLCriterion``: Criterion wrappers for the Newton loop.
- ``JointREMLCriterion`` / ``JointMLCriterion``: Joint (log_lambda, log_phi)
  criterion wrappers for unknown-scale families.
"""

from pymgcv.fitting.data import FittingData
from pymgcv.fitting.initialization import initialize_beta
from pymgcv.fitting.newton import NewtonOptimizer, NewtonResult, newton_optimize
from pymgcv.fitting.pirls import PIRLSResult, pirls_loop
from pymgcv.fitting.reml import (
    JointMLCriterion,
    JointREMLCriterion,
    MLCriterion,
    REMLCriterion,
    REMLResult,
    ml_criterion,
    reml_criterion,
)

__all__ = [
    "FittingData",
    "JointMLCriterion",
    "JointREMLCriterion",
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
