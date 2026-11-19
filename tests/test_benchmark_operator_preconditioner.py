"""The paired preconditioner benchmark checks its solves by residual norm."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.tolerances import STRICT

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_operator_preconditioner.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "benchmark_operator_preconditioner", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_case_records_dense_residual_norm_within_strict():
    """Both policies converge and satisfy the dense system in norm."""
    case = _load()._case(40, -5.0, 8 * 1024 * 1024)

    assert case["n_coef"] == 46
    assert set(case["outcomes"]) == {"false", "true"}
    for outcome in case["outcomes"].values():
        assert outcome["dense_relative_residual"] <= STRICT.rtol
        assert outcome["source_scans"] == outcome["matvec_count"] + 1
