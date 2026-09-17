"""Smoke and disagreement gates for the dense EFS benchmark harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import benchmark_efs
from tests.helpers import r_available
from tests.tolerances import MODERATE, STRICT


@pytest.fixture(autouse=True)
def isolated_direct_worker_cache(tmp_path: Path):
    import jax

    previous = jax.config.jax_compilation_cache_dir
    jax.config.update("jax_compilation_cache_dir", str(tmp_path / "jax-cache"))
    yield
    jax.config.update("jax_compilation_cache_dir", previous)


def _finite_result() -> dict:
    return {
        "coefficients": [1.0],
        "fitted_values": [2.0],
        "smoothing_params": [3.0],
        "edf": [0.5],
        "deviance": 4.0,
        "score": 5.0,
        "scale": 1.0,
        "outer_iterations": 2,
        "converged": True,
    }


def _config(tmp_path: Path) -> Path:
    args = benchmark_efs._parser().parse_args(
        ["--smoke", "--warm-repeats", "1", "--output-dir", str(tmp_path)]
    )
    tmp_path.mkdir(exist_ok=True)
    benchmark_efs._prepare_config(args, tmp_path)
    return tmp_path / "config.json"


def test_worker_output_larger_than_pipe_capacity_completes() -> None:
    # Use a bounded outer process so a reintroduced pipe deadlock fails the
    # test instead of hanging the full suite. Both output streams exceed any
    # usual pipe capacity, as a sufficiently large fitted-value report can.
    code = """
import os
import sys
from scripts.benchmark_efs import _run_worker
worker = (
    "import json,sys; sys.stderr.write('e'*1048576); "
    "print(json.dumps({'payload':'x'*1048576}))"
)
result, memory = _run_worker([sys.executable, '-c', worker], os.environ.copy())
assert len(result['payload']) == 1048576
assert 'peak_process_tree_rss_bytes' in memory
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr


def test_comparison_flags_numerical_disagreement() -> None:
    comparison = benchmark_efs._comparison(np.array([1.0, 2.0]), np.array([1.0, 2.01]))
    assert not comparison["passed"]
    assert comparison["max_absolute_error"] > 0.009


@pytest.mark.parametrize("nonfinite", [np.inf, -np.inf, np.nan])
def test_matching_nonfinite_fields_never_pass(nonfinite: float) -> None:
    assert not benchmark_efs._comparison([nonfinite], [nonfinite])["passed"]
    assert not benchmark_efs._comparison([1.0], [1.0, 2.0])["passed"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("converged", False),
        ("coefficients", [np.inf]),
        ("score", np.nan),
        ("deviance", -1.0),
        ("scale", 0.0),
        ("smoothing_params", [0.0]),
        ("covariance_finite", False),
        ("linear_predictor_finite", False),
    ],
)
def test_invalid_fits_block_comparison(field: str, value: object) -> None:
    result = _finite_result()
    result[field] = value
    validity = benchmark_efs._fit_validity(result)
    assert not validity["passed"]
    payload = {"result": result}
    assert not benchmark_efs._numerical_comparisons(payload, payload)["passed"]


def test_strict_and_moderate_are_reported_without_implicit_acceptance() -> None:
    assert benchmark_efs._TOLERANCES["STRICT"] == (STRICT.rtol, STRICT.atol)
    assert benchmark_efs._TOLERANCES["MODERATE"] == (MODERATE.rtol, MODERATE.atol)
    actual, expected = _finite_result(), _finite_result()
    actual["coefficients"] = [1.0 + 1e-7]
    assert not benchmark_efs._numerical_comparisons(
        {"result": actual}, {"result": expected}, "STRICT"
    )["passed"]
    assert benchmark_efs._numerical_comparisons(
        {"result": actual}, {"result": expected}, "MODERATE"
    )["passed"]


def test_saved_roundtrip_input_drives_starts_and_checks_hash(tmp_path: Path) -> None:
    config = json.loads(_config(tmp_path).read_text())
    saved = benchmark_efs._load_input(config)
    expected = pd.read_csv(config["data_path"], float_precision="round_trip")
    np.testing.assert_array_equal(saved.to_numpy(), expected.to_numpy())
    assert config["initialization_input_parser"].endswith("on saved CSV")
    Path(config["data_path"]).write_text("x1,y\n0,1\n")
    with pytest.raises(ValueError, match="hash changed"):
        benchmark_efs._load_input(config)


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_direct_saved_input_workers_and_original_strict_boundary(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    newton = benchmark_efs._python_worker(config_path, "newton")
    efs = benchmark_efs._python_worker(config_path, "efs")
    reference = benchmark_efs._r_worker(config_path)
    assert newton["controls"]["optimizer"]["convergence_tolerance"] == 1e-6
    assert newton["controls"]["optimizer"]["pirls_tolerance"] == 1e-8
    assert newton["controls"]["optimizer"]["pirls_max_iter"] == 100
    assert efs["controls"]["optimizer"]["pirls_tolerance"] == 1e-7
    assert not efs["controls"]["fit_control"]["gaussian_compression"]
    assert all(item["passed"] for item in newton["measured_fit_validity"])
    assert all(item["passed"] for item in efs["measured_fit_validity"])
    assert all(item["passed"] for item in reference["measured_fit_validity"])
    assert newton["input_sha256"] == efs["input_sha256"] == reference["input_sha256"]
    assert (
        efs["input_sha256"]
        == "8a49df584d593df11ed6ffc44e66ec72ab37f2352ab37b353e1c42c4232a26b9"
    )
    strict = benchmark_efs._numerical_comparisons(efs, reference, "STRICT")
    moderate = benchmark_efs._numerical_comparisons(efs, reference, "MODERATE")
    assert not strict["passed"]
    assert moderate["passed"]  # default STRICT selection still rejects it


def test_driver_rejects_invalid_measured_fit_and_strict_failure(
    tmp_path: Path, monkeypatch
) -> None:
    args = benchmark_efs._parser().parse_args(
        ["--smoke", "--warm-repeats", "1", "--output-dir", str(tmp_path)]
    )

    def worker(command, _environment):
        result = _finite_result()
        if command[command.index("--worker") + 1] == "efs":
            result["coefficients"] = [1.0 + 1e-7]
        return {"result": result, "measured_fit_validity": [{"passed": True}]}, {}

    monkeypatch.setattr(benchmark_efs, "_run_worker", worker)
    assert benchmark_efs._driver(args) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["efs_vs_pinned_r"]["STRICT"]["passed"]
    assert report["efs_vs_pinned_r"]["MODERATE"]["passed"]
    assert report["selected_comparison_tolerance"]["name"] == "STRICT"
    args.comparison_tolerance = "MODERATE"
    assert benchmark_efs._driver(args) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    selection = report["selected_comparison_tolerance"]
    assert selection["name"] == "MODERATE"
    assert selection["reviewed_scope"]["data_sha256"] == (
        "8a49df584d593df11ed6ffc44e66ec72ab37f2352ab37b353e1c42c4232a26b9"
    )
    assert not report["efs_vs_pinned_r"]["STRICT"]["passed"]
    args.python_only = True

    def invalid_worker(_command, _environment):
        return {
            "result": _finite_result(),
            "measured_fit_validity": [{"passed": False}],
        }, {}

    monkeypatch.setattr(benchmark_efs, "_run_worker", invalid_worker)
    assert benchmark_efs._driver(args) == 2


@pytest.mark.parametrize("changed", ["data_sha256", "formula", "efs_control"])
def test_moderate_selection_rejects_unreviewed_input_model_or_controls(
    tmp_path: Path, changed: str
) -> None:
    config = json.loads(_config(tmp_path).read_text())
    assert benchmark_efs._comparison_selection(config, "MODERATE")["name"] == "MODERATE"
    config[changed] = {} if changed == "efs_control" else "different"
    with pytest.raises(ValueError, match="only for the exact saved smoke"):
        benchmark_efs._comparison_selection(config, "MODERATE")


def test_moderate_comparison_preserves_iteration_and_validity_requirements() -> None:
    actual, expected = _finite_result(), _finite_result()
    actual["outer_iterations"] += 1
    comparison = benchmark_efs._numerical_comparisons(
        {"result": actual}, {"result": expected}, "MODERATE"
    )
    assert not comparison["passed"]
    assert not comparison["fields"]["outer_iterations"]["passed"]
    actual["outer_iterations"] = expected["outer_iterations"]
    actual["converged"] = expected["converged"] = False
    assert not benchmark_efs._numerical_comparisons(
        {"result": actual}, {"result": expected}, "MODERATE"
    )["passed"]


def test_failed_worker_reports_both_streams() -> None:
    with pytest.raises(RuntimeError, match="stdout"):
        benchmark_efs._run_worker(
            [
                sys.executable,
                "-c",
                "import sys; print('output'); sys.stderr.write('error'); sys.exit(3)",
            ],
            os.environ.copy(),
        )


def test_r_failed_fit_is_reported(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        benchmark_efs.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, "", "pinned failure"
        ),
    )
    with pytest.raises(RuntimeError, match="pinned failure"):
        benchmark_efs._r_worker(config)


def test_python_smoke_runs_saved_data_through_isolated_workers(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_efs.py",
            "--smoke",
            "--python-only",
            "--warm-repeats",
            "1",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text())
    saved = output / "input.csv"
    assert saved.exists()
    assert report["input"]["data_sha256"] == benchmark_efs._sha256(saved)
    assert report["input"]["n"] == 128
    assert report["input"]["m"] == 3
    assert report["methods"]["newton"]["result"]["converged"]
    assert report["methods"]["efs"]["result"]["converged"]
    assert report["methods"]["efs"]["result"]["efs_diagnostics"] is not None
    assert report["pinned_r_efs"] is None
    assert report["efs_vs_pinned_r"] is None
