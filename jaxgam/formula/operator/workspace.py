"""Prospective host workspace bounds for compact operator prediction.

The ordinary prediction classes intentionally expose only NumPy semantics.
Operator execution additionally needs a hard allocation contract: a small
coefficient block can still retain many radial knots, so its evaluator scratch
is not bounded by the returned design width.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from jaxgam.smooths.by_variable import FactorBySmooth, NumericBySmooth
from jaxgam.smooths.gaussian_process import GaussianProcessSmooth
from jaxgam.smooths.tensor import TensorProductSmooth
from jaxgam.smooths.tprs import TPRSSmooth, null_space_dimension
from jaxgam.smooths.utils import DISTANCE_BATCH_ROWS

if TYPE_CHECKING:
    from jaxgam.formula.predict_matrix import PredictSpec


def _radial_workspace_bound(smooth: Any, n_rows: int) -> int:
    """Bound one radial predictor, including knot-dependent kernel scratch.

    TPRS' masked even-power kernel can coexist with the distance input, output,
    boolean mask and four full-size arithmetic temporaries. GP kernels retain
    the unscaled distance while their scaled input and up to seven arithmetic
    temporaries coexist. Ten float64 knot-pair arrays conservatively cover both
    paths. Row inputs, polynomial/null columns, projected output and the final
    assignment are charged separately.
    """
    if isinstance(smooth, TPRSSmooth):
        knots = smooth._Xu
        if knots is None:
            raise RuntimeError("Frozen radial predictor is missing its knot table.")
        dimension = int(smooth._d)
        null_width = null_space_dimension(dimension, int(smooth._m))
    elif isinstance(smooth, GaussianProcessSmooth):
        knots = smooth._knt
        if knots is None:
            raise RuntimeError("Frozen radial predictor is missing its knot table.")
        dimension = len(smooth.spec.variables)
        null_width = int(smooth.n_coefs - smooth.rank)
    else:
        return 0
    chunk_rows = min(n_rows, DISTANCE_BATCH_ROWS)
    knot_count = int(knots.shape[0])
    pair_scratch = 10 * np.dtype(np.float64).itemsize * chunk_rows * knot_count
    row_arrays = (
        8 * n_rows * (4 * dimension + 3 * int(smooth.n_coefs) + 2 * null_width + 8)
    )
    return pair_scratch + row_arrays


def smooth_evaluator_workspace_bound(smooth: Any, n_rows: int) -> int:
    """Conservatively bound one frozen smooth prediction call."""
    if n_rows < 0:
        raise ValueError("Operator evaluator row count must be nonnegative.")
    if isinstance(smooth, (FactorBySmooth, NumericBySmooth)):
        base = smooth.base_smooth
        return smooth_evaluator_workspace_bound(base, n_rows) + 8 * n_rows * (
            3 * int(base.n_coefs) + 3 * int(smooth.n_coefs) + 8
        )
    if isinstance(smooth, TensorProductSmooth):
        margins = tuple(smooth._marginals)
        # predict_matrix retains every marginal row matrix, optional Z/XP
        # products, and chained tensor outputs. Summing marginal peak bounds is
        # conservative and keeps the contract independent of evaluation order.
        margin_peak = sum(
            smooth_evaluator_workspace_bound(margin, n_rows) for margin in margins
        )
        widths = sum(int(margin.n_coefs) for margin in margins)
        transform_widths = sum(
            int(Z.shape[1]) for Z in getattr(smooth, "_Z_list", ())
        ) + sum(
            int(XP.shape[1]) for XP in getattr(smooth, "_XP_list", ()) if XP is not None
        )
        return margin_peak + 8 * n_rows * (
            4 * widths + 3 * transform_widths + 3 * int(smooth.n_coefs) + 8
        )
    radial = _radial_workspace_bound(smooth, n_rows)
    if radial:
        return radial
    return 8 * n_rows * (3 * int(getattr(smooth, "n_coefs", 0)) + 8)


def predict_spec_evaluator_workspace_bound(spec: PredictSpec, n_rows: int) -> int:
    """Bound all smooth-native temporaries in a compact PredictSpec call."""
    smooths = tuple(
        term.smooth
        for term in spec.coef_map.terms
        if term.term_type == "smooth" and term.smooth is not None
    )
    # build_predict_matrix retains transformed term blocks until column_stack;
    # charging every smooth peak plus raw/transformed/final rows bounds that
    # overlap without depending on Python's array-release timing.
    native = sum(smooth_evaluator_workspace_bound(smooth, n_rows) for smooth in smooths)
    widths = sum(int(smooth.n_coefs) for smooth in smooths)
    transforms = sum(
        int(term.n_coefs) for term in spec.coef_map.terms if term.term_type == "smooth"
    )
    return native + 8 * n_rows * (3 * widths + 2 * transforms + spec.total_coefs + 8)
