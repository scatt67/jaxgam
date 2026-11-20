"""Memory accounting of bounded TP/GP operator preparation against distinct rows.

The scan holds the distinct-row maps (linear in the distinct rows q); the
knot setup works on at most ``max_knots`` sampled rows after the maps are
released. The preparation bound must follow that, stay above what is
allocated, and report the full requirement once the distinct rows are known.
"""

from __future__ import annotations

import re
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.formula.operator.prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula

_TP = "y~s(x0,x1,bs='tp',k=10)"
_GP = "y~s(x0,bs='gp',k=10)"


def _distinct_frame(n: int, seed: int = 3) -> pd.DataFrame:
    """``n`` rows with ``n`` distinct covariate rows."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x0": rng.uniform(0, 1, n), "x1": rng.uniform(0, 1, n)})
    frame["y"] = np.sin(2 * np.pi * frame.x0) + rng.normal(0, 0.3, n)
    return frame


def _prepare(formula: str, frame: pd.DataFrame, budget: int = 1 << 36):
    return prepare_operator_model(
        parse_formula(formula),
        DataFrameRowSource(frame, response="y"),
        control=OperatorPreparationControl(batch_rows=1024, memory_budget_bytes=budget),
    )


def test_bound_grows_linearly_in_distinct_rows_beyond_the_knot_cap():
    # With 100 knots the setup is fixed beyond 100 distinct rows; only the
    # distinct-row maps and their ordering grow. The former charge added
    # about 80 bytes times q squared (3.1 GB between these two sizes).
    formula = "y~s(x0,x1,bs='tp',k=10,max_knots=100)"
    bounds = {
        q: _prepare(formula, _distinct_frame(q)).preparation_peak_bound_bytes
        for q in (1600, 6400)
    }
    assert bounds[6400] - bounds[1600] < 4096 * (6400 - 1600)


def test_bound_levels_off_at_the_default_knot_cap():
    # Above max_knots = 2000 distinct rows the knot setup no longer grows;
    # the measured bound is 340.7 MB from 2,048 to 16,384 rows (host).
    at_cap = _prepare(_TP, _distinct_frame(2048)).preparation_peak_bound_bytes
    above = _prepare(_TP, _distinct_frame(8192)).preparation_peak_bound_bytes
    assert above <= at_cap + 4096 * (8192 - 2048)


@pytest.mark.parametrize(
    ("formula", "q"), [(_TP, 1024), (_TP, 16384), (_GP, 4096), (_GP, 16384)]
)
def test_bound_covers_the_traced_preparation_peak(formula: str, q: int):
    _prepare(formula, _distinct_frame(300, seed=9))  # compile outside the trace
    frame = _distinct_frame(q)
    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        model = _prepare(formula, frame)
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        tracemalloc.stop()
    assert peak <= model.preparation_peak_bound_bytes


def test_eight_thousand_distinct_rows_fit_within_the_default_budget():
    control = FitControl(execution="operator", linear_solver="pcg", batch_rows=1024)
    assert control.memory_budget_bytes == 512 * 1024 * 1024
    result = GAM(_TP, sp=[1.0], control=control).fit(
        DataFrameRowSource(_distinct_frame(8192), response="y"), result="prediction"
    )
    assert result.converged


def test_rejection_reports_the_full_requirement_after_the_scan():
    frame = _distinct_frame(8192)
    requirement = _prepare(_TP, frame).preparation_peak_bound_bytes
    with pytest.raises(MemoryError) as raised:
        _prepare(_TP, frame, budget=requirement - 1)
    assert str(raised.value) == (
        f"Operator preparation needs at most {requirement} bytes; "
        f"budget is {requirement - 1}. "
        "Lower batch_rows or raise the memory budget."
    )


def test_rejection_during_the_scan_says_it_is_a_lower_bound():
    with pytest.raises(MemoryError) as raised:
        _prepare(
            "y~s(x0,x1,bs='tp',k=10,max_knots=100)",
            _distinct_frame(20_000),
            budget=3_000_000,
        )
    message = str(raised.value)
    assert re.fullmatch(
        r"Operator preparation needs at least \d+ bytes \(distinct covariate rows "
        r"are still being counted\); budget is 3000000\. "
        r"Lower batch_rows or raise the memory budget\.",
        message,
    )
