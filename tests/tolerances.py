"""Tolerance classes for test comparisons.

Three tolerance tiers for different comparison contexts:
- STRICT: CPU self-consistency, exact algebraic properties
- MODERATE: GPU vs CPU, cross-backend comparisons
- LOOSE: PyMGCV vs R mgcv (different implementations, BLAS, algorithms)

Usage::

    import numpy as np
    from tests.tolerances import STRICT, MODERATE, LOOSE

    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToleranceClass:
    """Tolerance specification for numerical comparisons."""

    rtol: float
    atol: float
    label: str


STRICT = ToleranceClass(rtol=1e-10, atol=1e-12, label="strict")
MODERATE = ToleranceClass(rtol=1e-6, atol=1e-8, label="moderate")
LOOSE = ToleranceClass(rtol=1e-3, atol=1e-5, label="loose")
