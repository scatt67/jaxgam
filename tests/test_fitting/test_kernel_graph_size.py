"""Graph size of the discrete working-model kernels against the smooth count.

Both kernels used to unroll Python loops over pairs of smooths: at 100
smooths their jaxprs held 156,790 and 125,251 equations here and XLA
compiled them for minutes. Stacked by block shape, ``discrete_xwxd`` and the
proposal's pair Hessian (``d.detXXS``) depend on the number of distinct
shapes only. The proposal keeps three linear per-block loops (coefficient
back-transform, ``Sl.iftChol``'s penalty actions and the singleton log
determinant) for their rounding, so it grows linearly, never quadratically.
The tests trace and lower at 10, 25, 50 and 100 equal-shape smooths; they
never compile or run the kernels.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from jaxgam.execution.working_model import _factor_shapes
from jaxgam.fitting.discrete_ops import (
    DeviceDiscreteDesign,
    DeviceLookupBlock,
    discrete_xwxd,
)
from jaxgam.fitting.working_model import (
    WorkingModelCriterionMetadata,
    _source_hessian_determinant_derivatives,
    gaussian_working_model_proposal,
)
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)

jax.config.update("jax_enable_x64", True)

_SMOOTH_COUNTS = (10, 25, 50, 100)
_K = 9
# A stacked kernel's equation count is constant in the smooth count up to
# the trace-time stacking of per-block arguments; allow that growth, not m^2.
_GROWTH_FACTOR = 2


def _equation_count(jaxpr) -> int:
    total = 0
    for equation in jaxpr.eqns:
        total += 1
        for value in equation.params.values():
            for item in value if isinstance(value, (list, tuple)) else (value,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    total += _equation_count(inner)
    return total


def _penalty_structure(m: int) -> PenaltyStructure:
    difference = np.diff(np.eye(_K), n=2, axis=0)
    penalty = difference.T @ difference
    return PenaltyStructure(
        1 + m * _K,
        tuple(
            PenaltyBlock(
                1 + j * _K,
                1 + (j + 1) * _K,
                (j,),
                (DenseLocalPenalty(penalty),),
                IdentityTransform(_K),
                (_K - 2,),
            )
            for j in range(m)
        ),
    )


def _proposal_equations(m: int) -> int:
    metadata = WorkingModelCriterionMetadata.from_structure(
        _penalty_structure(m), 10_000
    )
    p = 1 + m * _K
    scalar = jax.ShapeDtypeStruct((), np.float64)
    traced = gaussian_working_model_proposal.trace(
        jax.ShapeDtypeStruct((m + 1,), np.float64),
        jax.ShapeDtypeStruct((p, p), np.float64),
        jax.ShapeDtypeStruct((p,), np.float64),
        scalar,
        metadata,
        _factor_shapes(metadata),
        smoothing=jax.ShapeDtypeStruct((m,), np.float64),
        tolerance=scalar,
    )
    traced.lower()
    return _equation_count(traced.jaxpr.jaxpr)


def _xwxd_equations(m: int) -> int:
    n_rows, q = 4096, 200
    tables = np.linspace(0.0, 1.0, q * _K).reshape(q, _K)
    design = DeviceDiscreteDesign(
        tuple(DeviceLookupBlock(jax.numpy.asarray(tables), j * _K) for j in range(m)),
        m * _K,
        "histogram_or_direct",
        32 * 1024 * 1024,
        n_rows,
    )
    selector = jax.ShapeDtypeStruct((n_rows,), np.int32)
    traced = discrete_xwxd.trace(
        design,
        tuple((selector,) for _ in range(m)),
        jax.ShapeDtypeStruct((n_rows,), np.float64),
    )
    traced.lower()
    return _equation_count(traced.jaxpr.jaxpr)


def _hessian_derivative_equations(m: int) -> int:
    metadata = WorkingModelCriterionMetadata.from_structure(
        _penalty_structure(m), 10_000
    )
    p = 1 + m * _K
    traced = jax.jit(_source_hessian_determinant_derivatives).trace(
        jax.ShapeDtypeStruct((p, p), np.float64),
        metadata.stacked,
        jax.ShapeDtypeStruct((m,), np.float64),
        jax.ShapeDtypeStruct((m,), np.float64),
    )
    traced.lower()
    return _equation_count(traced.jaxpr.jaxpr)


@pytest.mark.parametrize(
    "kernel",
    [_hessian_derivative_equations, _xwxd_equations],
    ids=["pair_hessian", "xwxd"],
)
def test_pair_kernel_graph_does_not_grow_with_smooth_count(kernel) -> None:
    counts = {m: kernel(m) for m in _SMOOTH_COUNTS}
    assert max(counts.values()) <= _GROWTH_FACTOR * counts[_SMOOTH_COUNTS[0]], counts


def test_proposal_graph_grows_at_most_linearly() -> None:
    counts = {m: _proposal_equations(m) for m in _SMOOTH_COUNTS}
    first, second, *_, last = _SMOOTH_COUNTS
    early = (counts[second] - counts[first]) / (second - first)
    late = (counts[last] - counts[50]) / (last - 50)
    # Equations per added smooth must not rise with the smooth count.
    assert late <= early, counts
