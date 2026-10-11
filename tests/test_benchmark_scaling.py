"""Unit tests for scaling-evidence metadata and cache isolation."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import benchmark_scaling
from scripts.benchmark_scaling import ScalingCase, isolated_cache_dir, source_identity
from tests.helpers import _AssertCollector, check_that, r_available
from tests.tolerances import MODERATE


@pytest.fixture
def minimal_git_checkout(tmp_path: Path) -> Path:
    """Build the smallest real checkout needed by isolated harness children.

    The Docker test image deliberately excludes the repository's ``.git``
    directory. The harness must still fail closed when provenance cannot be
    obtained, so real-child tests run against this copied, committed checkout
    rather than mocking source identity or relaxing the production helper.
    """
    checkout = tmp_path / "checkout"
    ignore_bytecode = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(Path.cwd() / "jaxgam", checkout / "jaxgam", ignore=ignore_bytecode)
    (checkout / "tests").mkdir()
    for name in ("__init__.py", "tolerances.py"):
        shutil.copy2(Path.cwd() / "tests" / name, checkout / "tests" / name)
    (checkout / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    subprocess.run(["git", "init", "--quiet"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "add", ".gitignore", "jaxgam", "tests"], cwd=checkout, check=True
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=benchmark-test",
            "-c",
            "user.email=benchmark-test@example.com",
            "commit",
            "--quiet",
            "-m",
            "minimal isolated harness checkout",
        ],
        cwd=checkout,
        check=True,
    )
    return checkout


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_rows": 0, "basis_dimension": 10},
        {"n_rows": 1, "basis_dimension": 2},
        {"n_rows": 1, "basis_dimension": 10, "batch_rows": 0},
        {"n_rows": 1, "basis_dimension": 10, "repeats": 0},
        {"n_rows": 1, "basis_dimension": 10, "smooth_count": 0},
        {"n_rows": 1, "basis_dimension": 10, "execution": "bam"},
        {"n_rows": 1, "basis_dimension": 10, "result_mode": "lean"},
        {"n_rows": 1, "basis_dimension": 10, "execution": "stream"},
        {"n_rows": 1, "basis_dimension": 10, "execution": "stream-exact"},
    ],
)
def test_case_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"positive|at least|execution|result_mode"):
        ScalingCase(**kwargs)  # type: ignore[arg-type]


def test_isolated_cache_is_temporary_and_task_owned() -> None:
    with isolated_cache_dir() as cache:
        path = Path(cache)
        assert path.exists()
        assert path.parent == Path(tempfile.gettempdir())
        assert path.name.startswith("jaxgam-benchmark-jax-")
    assert not path.exists()


def test_directory_bytes_reports_only_existing_regular_files(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "entry").write_bytes(b"abc")
    (cache / "nested").mkdir()
    (cache / "nested" / "other").write_bytes(b"de")

    assert benchmark_scaling._directory_bytes(cache) == 5
    assert benchmark_scaling._directory_bytes(cache / "missing") is None


def test_source_identity_is_commit_and_dirty_hash(minimal_git_checkout: Path) -> None:
    record = source_identity(minimal_git_checkout)

    assert len(record["commit"]) == 40
    assert len(record["dirty_sha256"]) == 64


def test_source_identity_includes_untracked_content(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )

    clean = source_identity(tmp_path)
    (tmp_path / "untracked.py").write_text("value = 2\n")

    assert source_identity(tmp_path)["dirty_sha256"] != clean["dirty_sha256"]


def test_loaded_source_identity_hashes_current_checkout() -> None:
    import jaxgam

    record = benchmark_scaling.loaded_source_identity(jaxgam, Path.cwd())

    assert record["loaded_package"].endswith("jaxgam")
    assert len(record["loaded_source_sha256"]) == 64


def test_loaded_source_identity_rejects_another_checkout(tmp_path: Path) -> None:
    foreign_package = tmp_path / "foreign" / "jaxgam"
    foreign_package.mkdir(parents=True)
    module = SimpleNamespace(__file__=foreign_package / "__init__.py")

    with pytest.raises(RuntimeError, match="not requested checkout"):
        benchmark_scaling.loaded_source_identity(module, Path.cwd())


def test_typed_reference_imports_are_bound_to_selected_checkout(tmp_path: Path) -> None:
    benchmark_scaling._check_reference_imports(Path.cwd(), r_parity=True)
    with pytest.raises(RuntimeError, match="not selected checkout"):
        benchmark_scaling._check_reference_imports(tmp_path, r_parity=True)


def test_cli_child_selects_checkout_before_tolerance_import(
    tmp_path: Path, minimal_git_checkout: Path
) -> None:
    decoy = tmp_path / "stale" / "tests"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("")
    (decoy / "tolerances.py").write_text(
        'raise RuntimeError("stale tolerance import")\n'
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "scripts" / "benchmark_scaling.py"),
            "--checkout",
            str(minimal_git_checkout),
            "--child",
            "setup",
            "--case",
            json.dumps({"n_rows": 12, "basis_dimension": 4}),
        ],
        cwd=minimal_git_checkout,
        env=os.environ | {"PYTHONPATH": str(decoy.parent)},
        capture_output=True,
        text=True,
        check=True,
    )
    record = json.loads(completed.stdout)
    assert record["runtime"]["loaded_package"] == str(minimal_git_checkout / "jaxgam")


def test_main_smoke_uses_selected_current_checkout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        benchmark_scaling,
        "run_public_smoke",
        lambda root: {"checkout": str(root)},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_scaling.py",
            "--smoke",
            "--checkout",
            str(tmp_path),
            "--current-checkout",
            str(Path.cwd()),
        ],
    )
    benchmark_scaling.main()
    assert json.loads(capsys.readouterr().out) == {"checkout": str(Path.cwd())}


def test_harness_identity_hashes_the_executed_script() -> None:
    record = benchmark_scaling.harness_identity()

    assert record["harness_path"].endswith("scripts/benchmark_scaling.py")
    assert len(record["harness_sha256"]) == 64


def test_array_hash_includes_shape_and_values() -> None:
    assert benchmark_scaling._array_sha256(np.array([1.0])) != (
        benchmark_scaling._array_sha256(np.array([[1.0]]))
    )


def test_canonical_penalty_contract_matches_legacy_global_and_local_storage() -> None:
    from jaxgam.penalties.structure import (
        DenseLocalPenalty,
        IdentityTransform,
        PenaltyBlock,
        PenaltyStructure,
    )

    local = np.array([[2.0, -1.0], [-1.0, 2.0]])
    global_penalty = np.zeros((3, 3))
    global_penalty[1:, 1:] = local
    legacy = SimpleNamespace(penalties=(SimpleNamespace(S=global_penalty, rank=2),))
    current = PenaltyStructure(
        3,
        (
            PenaltyBlock(
                start=1,
                stop=3,
                sp_indices=(0,),
                local_penalties=(DenseLocalPenalty(local),),
                transform=IdentityTransform(2),
                ranks=(2,),
            ),
        ),
    )

    assert benchmark_scaling._canonical_penalty_contract(3, legacy) == (
        benchmark_scaling._canonical_penalty_contract(3, current)
    )


@pytest.mark.parametrize(
    ("execution", "adapter", "result_mode"),
    [
        ("dense", "public", "full"),
        ("stream", "public", "prediction"),
        ("stream-qr", "public", "prediction"),
        ("efs", "public", "prediction"),
    ],
)
def test_route_contracts_explicitly_name_adapter_and_result_semantics(
    execution: str, adapter: str, result_mode: str
) -> None:
    route = benchmark_scaling._route_contract(
        ScalingCase(
            4,
            3,
            execution=execution,
            result_mode="prediction" if execution != "dense" else "full",
        )
    )

    assert route["requested_execution"] == execution
    assert route["adapter"] == adapter
    assert route["result_mode"] == result_mode


def test_input_provenance_includes_case_specific_response_generator() -> None:
    gaussian = ScalingCase(5, 3, execution="dense")
    efs = ScalingCase(5, 3, execution="efs", result_mode="prediction")

    gaussian_provenance = benchmark_scaling._input_provenance(
        benchmark_scaling._make_data(gaussian), gaussian
    )
    efs_provenance = benchmark_scaling._input_provenance(
        benchmark_scaling._make_data(efs), efs
    )

    assert gaussian_provenance["generator"] == "gaussian_sine"
    assert efs_provenance["generator"] == "poisson_sine"
    assert gaussian_provenance["frame_sha256"] != efs_provenance["frame_sha256"]


def test_one_smooth_formula_and_input_remain_backward_compatible() -> None:
    case = ScalingCase(5, 3)
    rng = np.random.default_rng(case.seed)
    x = rng.uniform(0.0, 1.0, case.n_rows)
    expected_y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.3, case.n_rows)

    data = benchmark_scaling._make_data(case)

    assert benchmark_scaling._formula(case) == "y ~ s(x, k=3, bs='cr')"
    np.testing.assert_array_equal(data.x.to_numpy(), x)
    np.testing.assert_array_equal(data.y.to_numpy(), expected_y)


def test_additive_formula_and_fixed_sp_order_are_stable() -> None:
    case = ScalingCase(5, 3, smooth_count=3)

    assert benchmark_scaling._formula(case) == (
        "y ~ s(x0, k=3, bs='cr') + s(x1, k=3, bs='cr') + s(x2, k=3, bs='cr')"
    )
    assert benchmark_scaling._smoothing_parameter_order(case) == [
        "s(x0)",
        "s(x1)",
        "s(x2)",
    ]
    assert benchmark_scaling._fixed_smoothing_parameters(case) == [1.0, 1.0, 1.0]


def test_additive_input_formula_and_sp_order_are_shared_by_fixed_sp_routes() -> None:
    cases = [
        ScalingCase(19, 5, execution=execution, result_mode=result_mode, smooth_count=2)
        for execution, result_mode in (
            ("dense", "prediction"),
            ("stream", "prediction"),
            ("stream-qr", "prediction"),
        )
    ]
    inputs = [
        benchmark_scaling._input_provenance(benchmark_scaling._make_data(case), case)
        for case in cases
    ]

    assert {benchmark_scaling._formula(case) for case in cases} == {
        "y ~ s(x0, k=5, bs='cr') + s(x1, k=5, bs='cr')"
    }
    assert len({record["frame_sha256"] for record in inputs}) == 1
    assert all(
        benchmark_scaling._route_contract(case)["smoothing_parameter_order"]
        == ["s(x0)", "s(x1)"]
        and benchmark_scaling._route_contract(case)["fixed_smoothing_parameter_values"]
        == [1.0, 1.0]
        for case in cases
    )


def test_numeric_comparison_reports_strict_and_moderate_outcomes() -> None:
    comparison = benchmark_scaling._numeric_comparison(
        np.array([1.0]), np.array([1.0 + 1e-8])
    )

    assert comparison["available"]
    assert not comparison["strict_allclose"]
    assert comparison["moderate_allclose"]


def test_numeric_comparison_fails_closed_on_shape_mismatch() -> None:
    comparison = benchmark_scaling._numeric_comparison(np.array([1.0]), np.eye(1))

    assert not comparison["available"]
    assert not comparison["same_shape"]


def test_timed_returns_value_and_nonnegative_elapsed_time() -> None:
    value, elapsed = benchmark_scaling._timed(lambda: "complete")

    assert value == "complete"
    assert elapsed >= 0.0


def test_build_fit_constructs_dense_and_stream_public_routes() -> None:
    data = pd.DataFrame({"x": [0.1, 0.2], "y": [0.0, 1.0]})

    dense_plan = benchmark_scaling._build_fit(
        ScalingCase(2, 3, execution="dense"), data
    )
    stream_plan = benchmark_scaling._build_fit(
        ScalingCase(2, 3, execution="stream", result_mode="prediction"), data
    )

    assert callable(dense_plan.run)
    assert callable(stream_plan.run)
    assert dense_plan.formula == stream_plan.formula
    assert dense_plan.source is None
    assert stream_plan.source is not None


def test_tiny_public_stream_qr_adapter_uses_published_route(
    minimal_git_checkout: Path,
) -> None:
    record = benchmark_scaling.run_isolated(
        ScalingCase(
            19, 5, execution="stream-qr", batch_rows=7, result_mode="prediction"
        ),
        minimal_git_checkout,
    )

    result = record["fit"]["result"]

    assert record["within_run_consistency"]["valid"]
    assert record["setup"]["route"]["adapter"] == "public"
    assert result["execution_route"] == "stream_qr"
    assert result["converged"]
    assert result["source_scans"]["available"]
    assert result["source_scans"]["cold_fit_delta"] > 0


def test_tiny_public_exact_route_records_source_bytes_and_optimizer(
    minimal_git_checkout: Path,
) -> None:
    case = ScalingCase(
        31, 6, execution="stream-exact", batch_rows=7, result_mode="prediction"
    )
    record = benchmark_scaling.run_isolated(case, minimal_git_checkout)

    assert record["within_run_consistency"]["valid"]
    assert record["setup"]["route"]["adapter"] == "public"
    assert record["setup"]["route"]["fixed_smoothing_parameter_values"] is None
    assert record["setup"]["source_scans"]["setup_batch_bytes"] > 0
    assert record["setup"]["source_scans"]["setup_batch_rows_requested"] == 7
    assert record["setup"]["source_scans"]["setup_max_batch_rows"] == 7
    assert (
        record["setup"]["memory"]["rss_high_water_after_setup_bytes"]
        >= record["setup"]["memory"]["rss_high_water_before_setup_bytes"]
    )
    assert "prepare_model" in record["setup"]["memory"]["rss_scope"]
    result = record["fit"]["result"]
    assert result["execution_route"] == "stream_exact_reml_qr"
    assert result["converged"]
    assert result["source_scans"]["cold_fit_delta"] > 0
    assert result["source_scans"]["cold_fit_batch_bytes"] > 0
    assert result["source_scans"]["warm_fit_batch_bytes"][0] > 0
    assert result["source_scans"]["training_source_batch_bytes_during_prediction"] == 0
    assert result["exact_optimizer_diagnostics"]["n_evaluations"] > 0
    assert result["outer_iterations"] == result["n_iter"]
    assert (
        result["inner_iterations"]
        >= result["exact_optimizer_diagnostics"]["n_evaluations"]
    )
    assert (
        result["final_refit_solves"]
        == result["exact_optimizer_diagnostics"]["n_evaluations"]
    )
    assert result["exact_optimizer_diagnostics"]["cumulative_derivative_s"] > 0.0
    assert record["fit"]["timings"]["derivative_s"] > 0.0
    assert result["n_iter_semantics"] == "outer exact-REML optimizer iterations"
    if r_available():
        parity = benchmark_scaling.compare_pinned_r_exact(case, record)
        assert parity["reference"].endswith("RBridge(mode='rpy2')")
        assert parity["fields"]["score"]["available"]
        assert parity["fields"]["smoothing"]["available"]


def test_pinned_r_exact_parity_rejects_unsupported_or_oversized_case() -> None:
    with pytest.raises(ValueError, match="stream-exact"):
        benchmark_scaling.compare_pinned_r_exact(
            ScalingCase(31, 6, execution="stream", result_mode="prediction"), {}
        )
    with pytest.raises(ValueError, match="bounded"):
        benchmark_scaling.compare_pinned_r_exact(
            ScalingCase(10_001, 6, execution="stream-exact", result_mode="prediction"),
            {},
        )


def test_pinned_r_exact_parity_uses_typed_bridge_and_reports_every_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import r_bridge

    case = ScalingCase(31, 6, execution="stream-exact", result_mode="prediction")
    response = {
        "coefficients": [0.5, 1.5],
        "fitted_values": np.array([0.25, 0.75]),
        "smoothing_params": [2.0],
        "reml_score": 3.0,
        "deviance": 4.0,
        "scale": 5.0,
        "reml_scale": 6.0,
        "edf_total": 1.5,
        "inner_converged": True,
        "outer_converged": True,
        "outer_status": "full convergence",
    }

    class PinnedBridge:
        @staticmethod
        def check_versions() -> tuple[bool, str]:
            return True, "pinned"

        def __init__(self, *, mode: str) -> None:
            assert mode == "rpy2"

        def fit_gam(self, formula: str, data: pd.DataFrame, family: str, method: str):
            assert formula == benchmark_scaling._formula(case)
            assert len(data) == case.n_rows
            assert (family, method) == ("gaussian", "REML")
            return response

    monkeypatch.setattr(r_bridge, "RBridge", PinnedBridge)
    record = {
        "within_run_consistency": {"valid": True},
        "fit": {
            "result": {
                "comparison_vectors": {
                    "coefficient_values": response["coefficients"],
                    "prediction_values": response["fitted_values"],
                },
                "smoothing_params": response["smoothing_params"],
                "score": response["reml_score"],
                "deviance": response["deviance"],
                "scale": response["scale"],
                "score_scale": response["reml_scale"],
                "edf_total": response["edf_total"],
                "converged": True,
            }
        },
    }

    parity = benchmark_scaling.compare_pinned_r_exact(case, record)
    assert parity["reference"].endswith("RBridge(mode='rpy2')")
    assert set(parity["fields"]) == {
        "coefficients",
        "prediction_first_128",
        "smoothing",
        "score",
        "deviance",
        "scale",
        "score_scale",
        "edf_total",
    }
    assert all(field["strict_allclose"] for field in parity["fields"].values())
    assert parity["strict_complete"]
    assert parity["moderate_numerical_match"]
    assert parity["r_inner_converged"]
    assert parity["r_outer_converged"]
    assert parity["within_run_consistent"]
    assert "four recorded" in parity["moderate_acceptance_status"]
    record["fit"]["result"]["converged"] = False
    nonconverged = benchmark_scaling.compare_pinned_r_exact(case, record)
    assert not nonconverged["strict_complete"]
    assert not nonconverged["moderate_numerical_match"]
    record["fit"]["result"]["converged"] = True
    record["within_run_consistency"]["valid"] = False
    assert not benchmark_scaling.compare_pinned_r_exact(case, record)["strict_complete"]
    record["within_run_consistency"]["valid"] = True
    response["outer_converged"] = False
    assert not benchmark_scaling.compare_pinned_r_exact(case, record)["strict_complete"]
    response["outer_converged"] = True
    response["inner_converged"] = False
    assert not benchmark_scaling.compare_pinned_r_exact(case, record)["strict_complete"]


def test_pinned_r_exact_parity_requires_pinned_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import r_bridge

    class WrongVersionBridge:
        @staticmethod
        def check_versions() -> tuple[bool, str]:
            return False, "wrong mgcv version"

    monkeypatch.setattr(r_bridge, "RBridge", WrongVersionBridge)
    case = ScalingCase(31, 6, execution="stream-exact", result_mode="prediction")
    with pytest.raises(r_bridge.RBridgeError, match="wrong mgcv version"):
        benchmark_scaling.compare_pinned_r_exact(case, {})


def test_memory_pair_subtracts_each_input_runtime_baseline() -> None:
    def record(n_rows: int, before: int, peak: int, batch_bytes: int) -> dict:
        return {
            "within_run_consistency": {"valid": True},
            "fit": {
                "case": {
                    "execution": "stream-exact",
                    "basis_dimension": 6,
                    "smooth_count": 1,
                    "batch_rows": 7,
                    "n_rows": n_rows,
                },
                "memory": {
                    "input_bytes": 16 * n_rows,
                    "rss_current_before_bytes": before,
                    "rss_high_water_after_cold_bytes": peak,
                },
                "result": {
                    "n_coef": 6,
                    "known_workspace_bytes": 90_000,
                    "source_scans": {
                        "cold_fit_delta": 8,
                        "cold_fit_batch_bytes": batch_bytes,
                    },
                },
            },
        }

    small = record(31, 100_000, 125_000, 4_000)
    large = record(62, 120_000, 148_000, 8_000)
    comparison = benchmark_scaling.compare_memory_growth(small, large)
    assert comparison["smaller"]["fit_rss_above_baseline_bytes"] == 25_000
    assert comparison["larger"]["fit_rss_above_baseline_bytes"] == 28_000
    assert comparison["known_workspace_plateau"]
    assert comparison["larger"]["cold_fit_batch_bytes"] == 8_000
    large["fit"]["case"]["batch_rows"] = 8
    with pytest.raises(ValueError, match="fixed B/p"):
        benchmark_scaling.compare_memory_growth(small, large)


def test_multi_smooth_dense_and_stream_contracts_match_in_public_coordinates() -> None:
    """Exercise real additive CR setup across legacy-global and local penalties."""
    from jaxgam.data.source import ArrayRowSource
    from jaxgam.families.registry import get_family
    from jaxgam.fitting.data import FittingData
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula
    from jaxgam.formula.prepare import prepare_model

    case = ScalingCase(19, 5, smooth_count=2)
    data = benchmark_scaling._make_data(case)
    spec = parse_formula(benchmark_scaling._formula(case))
    dense_setup = ModelSetup.build(spec, data)
    dense_fitting = FittingData.from_setup(dense_setup, get_family("gaussian"))
    source = benchmark_scaling._CountingArrayRowSource(
        ArrayRowSource(
            {
                name: data[name].to_numpy()
                for name in benchmark_scaling._smooth_variables(case)
            },
            y=data.y.to_numpy(),
        )
    )
    prepared = prepare_model(spec, source, family=get_family("gaussian"))

    assert prepared.fitting is not None
    assert source.scan_calls > 0
    assert dense_setup.X.shape[1] == prepared.n_coef
    assert dense_setup.penalties is not None
    assert prepared.penalties is not None
    assert benchmark_scaling._canonical_penalty_contract(
        prepared.n_coef, dense_setup.penalties
    ) == benchmark_scaling._canonical_penalty_contract(
        prepared.n_coef, prepared.penalties
    )
    assert dense_fitting.rank_deficit == prepared.fitting.unpenalized_rank_deficit


def test_tiny_additive_fixed_sp_routes_have_bounded_tolerant_observables() -> None:
    case = ScalingCase(19, 5, batch_rows=7, result_mode="prediction", smooth_count=2)
    data = benchmark_scaling._make_data(case)
    plans = {
        execution: benchmark_scaling._build_fit(
            ScalingCase(
                case.n_rows,
                case.basis_dimension,
                execution=execution,
                batch_rows=case.batch_rows,
                result_mode="prediction",
                smooth_count=case.smooth_count,
            ),
            data,
        )
        for execution in ("dense", "stream", "stream-qr")
    }
    results = {execution: plan.run() for execution, plan in plans.items()}
    dense = results["dense"]

    for execution in ("stream", "stream-qr"):
        candidate = results[execution]
        assert (
            candidate.coefficients.size
            <= benchmark_scaling._MAX_COMPARISON_COEFFICIENTS
        )
        np.testing.assert_allclose(
            candidate.coefficients,
            dense.coefficients,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )
        np.testing.assert_allclose(
            candidate.predict(data.iloc[:8]),
            dense.predict(data.iloc[:8]),
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        )


@pytest.mark.parametrize("execution", ["stream", "stream-qr"])
def test_tiny_additive_stream_routes_record_fit_scan_deltas_only(
    execution: str, minimal_git_checkout: Path
) -> None:
    record = benchmark_scaling.run_isolated(
        ScalingCase(
            19,
            5,
            execution=execution,
            batch_rows=7,
            repeats=2,
            result_mode="prediction",
            smooth_count=2,
        ),
        minimal_git_checkout,
    )

    scans = record["fit"]["result"]["source_scans"]
    assert record["within_run_consistency"]["valid"]
    assert record["setup"]["formula"] == (
        "y ~ s(x0, k=5, bs='cr') + s(x1, k=5, bs='cr')"
    )
    assert record["setup"]["route"]["smoothing_parameter_order"] == [
        "s(x0)",
        "s(x1)",
    ]
    assert record["setup"]["source_scans"]["available"]
    assert record["setup"]["source_scans"]["setup_delta"] > 0
    assert scans["available"]
    assert scans["cold_fit_delta"] > 0
    assert "includes preparation" in scans["cold_fit_scope"]
    assert all(delta > 0 for delta in scans["warm_fit_deltas"])
    assert scans["training_source_scan_delta_during_prediction"] == 0
    assert "original training RowSource" in scans["prediction_source_scope"]


def test_tiny_public_efs_adapter_uses_published_route(
    minimal_git_checkout: Path,
) -> None:
    record = benchmark_scaling.run_isolated(
        ScalingCase(19, 5, execution="efs", result_mode="prediction"),
        minimal_git_checkout,
    )

    result = record["fit"]["result"]

    assert record["within_run_consistency"]["valid"]
    assert record["setup"]["route"]["adapter"] == "public"
    assert result["execution_route"] == "dense"
    assert result["n_penalties"] == 1


def test_tiny_additive_efs_adapter_preserves_term_order(
    minimal_git_checkout: Path,
) -> None:
    record = benchmark_scaling.run_isolated(
        ScalingCase(
            19,
            5,
            execution="efs",
            result_mode="prediction",
            smooth_count=2,
        ),
        minimal_git_checkout,
    )

    assert record["within_run_consistency"]["valid"]
    assert record["setup"]["formula"] == (
        "y ~ s(x0, k=5, bs='cr') + s(x1, k=5, bs='cr')"
    )
    assert record["fit"]["result"]["n_penalties"] == 2


def test_tiny_dense_full_adapter_preserves_full_result_contract(
    minimal_git_checkout: Path,
) -> None:
    record = benchmark_scaling.run_isolated(ScalingCase(19, 5), minimal_git_checkout)

    assert record["within_run_consistency"]["valid"]
    assert record["setup"]["route"]["result_mode"] == "full"
    assert record["fit"]["result"]["execution_route"] == "dense"


def test_runtime_metadata_includes_numeric_library_and_blas_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax

    import jaxgam

    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "1")
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    monkeypatch.delenv("JAX_NUM_THREADS", raising=False)
    metadata = benchmark_scaling._runtime_metadata(jax, jaxgam)

    assert metadata["jaxlib"]
    assert metadata["numpy"] == np.__version__
    assert metadata["scipy"]
    assert metadata["blas"]["blas_name"]
    assert metadata["requested_thread_environment"]["veclib_maximum_threads"] == "1"
    assert metadata["requested_thread_environment"]["xla_flags"] == (
        "--xla_force_host_platform_device_count=1"
    )
    assert metadata["requested_thread_environment"]["jax_num_threads"] is None
    assert "do not prove" in metadata["requested_thread_environment_status"]


def test_counting_source_proxy_delegates_and_counts_scan_calls() -> None:
    batch = SimpleNamespace(
        columns={"x": np.array([0.1, 0.2])},
        y=np.array([1.0, 2.0]),
        weight=np.ones(2),
        offset=np.zeros(2),
        valid=np.ones(2, dtype=bool),
        row_positions=np.arange(2),
    )
    source = SimpleNamespace(
        n_rows=2,
        fingerprint=lambda: "fingerprint",
        scan=lambda _batch_rows: iter([batch]),
    )
    proxy = benchmark_scaling._CountingArrayRowSource(source)

    assert proxy.n_rows == 2
    assert proxy.fingerprint() == "fingerprint"
    assert list(proxy.scan(7)) == [batch]
    assert proxy.scan_calls == 1
    assert proxy.rows_yielded == 2
    assert proxy.batch_bytes == sum(
        np.asarray(value).nbytes
        for value in (
            batch.columns["x"],
            batch.y,
            batch.weight,
            batch.offset,
            batch.valid,
            batch.row_positions,
        )
    )


def test_counting_source_proxy_records_largest_yielded_batch() -> None:
    from jaxgam.data.source import ArrayRowSource

    proxy = benchmark_scaling._CountingArrayRowSource(
        ArrayRowSource({"x": np.linspace(0.0, 1.0, 10)}, y=np.arange(10.0))
    )

    assert proxy.max_batch_rows == 0
    assert [len(batch.row_positions) for batch in proxy.scan(4)] == [4, 4, 2]
    assert proxy.max_batch_rows == 4
    assert [len(batch.row_positions) for batch in proxy.scan(3)] == [3, 3, 3, 1]
    assert proxy.max_batch_rows == 4
    list(proxy.scan(6))
    assert proxy.max_batch_rows == 6
    assert proxy.rows_yielded == 30


def _pin_setup_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark_scaling,
        "source_identity",
        lambda _root: {"commit": "a" * 40, "dirty_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        benchmark_scaling,
        "loaded_source_identity",
        lambda _module, _root: {
            "loaded_package": "jaxgam",
            "loaded_source_sha256": "c" * 64,
        },
    )


@pytest.mark.parametrize("execution", ["stream", "stream-qr", "stream-exact"])
def test_run_setup_stream_uses_case_batch_and_memory_controls(
    execution: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup runs under the batch and budget controls of the public fit."""
    from jaxgam.control import FitControl
    from jaxgam.formula import prepare as prepare_module

    real_prepare = prepare_module.prepare_model
    calls: list[dict[str, object]] = []

    def recording_prepare(spec, source, **kwargs):
        calls.append(kwargs)
        return real_prepare(spec, source, **kwargs)

    monkeypatch.setattr(prepare_module, "prepare_model", recording_prepare)
    _pin_setup_provenance(monkeypatch)
    case = ScalingCase(
        31, 6, execution=execution, batch_rows=7, result_mode="prediction"
    )
    record = benchmark_scaling.run_setup(case, Path.cwd())

    # GAM.fit threads these FitControl fields into prepare_model.
    control = FitControl(execution="stream", batch_rows=case.batch_rows)
    assert len(calls) == 1
    assert calls[0]["batch_rows"] == control.batch_rows == case.batch_rows
    assert calls[0]["maximum_bytes"] == control.memory_budget_bytes
    scans = record["source_scans"]
    assert scans["setup_batch_rows_requested"] == case.batch_rows
    assert scans["setup_memory_budget_bytes"] == control.memory_budget_bytes
    # Thirty-one rows exceed seven, so every full scan is cut at the case's
    # batch size; the former 65,536-row setup default yielded all 31 at once.
    assert scans["setup_max_batch_rows"] == case.batch_rows
    assert scans["setup_delta"] > 0


def test_run_setup_rejects_batches_larger_than_the_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaxgam.formula import prepare as prepare_module

    def oversized_prepare(_spec, source, **kwargs):
        for _batch in source.scan(int(kwargs["batch_rows"]) + 1):
            pass
        # No prepared-model fields: the harness must reject before using it.
        return SimpleNamespace()

    monkeypatch.setattr(prepare_module, "prepare_model", oversized_prepare)
    _pin_setup_provenance(monkeypatch)
    case = ScalingCase(
        31, 6, execution="stream", batch_rows=7, result_mode="prediction"
    )

    with pytest.raises(
        RuntimeError, match="8-row batch, exceeding the case's batch_rows=7"
    ):
        benchmark_scaling.run_setup(case, Path.cwd())


def test_run_setup_dense_uses_model_matrix_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaxgam.fitting.data import FittingData
    from jaxgam.formula.design import ModelSetup

    fake_setup = SimpleNamespace(
        X=np.arange(12.0).reshape(4, 3),
        penalties=SimpleNamespace(n_penalties=2),
    )
    events: list[str] = []

    def build_setup(_cls, _spec, _data):
        events.append("build")
        return fake_setup

    def build_fitting(_cls, _setup, _family):
        events.append("fitting")
        return SimpleNamespace(
            penalty_structure=SimpleNamespace(n_penalties=2), rank_deficit=0
        )

    def peak_rss() -> int:
        events.append("peak")
        return 100 if events.count("peak") == 1 else 120

    def current_rss() -> int:
        events.append("current")
        return 90 if events.count("current") == 1 else 110

    monkeypatch.setattr(ModelSetup, "build", classmethod(build_setup))
    monkeypatch.setattr(FittingData, "from_setup", classmethod(build_fitting))
    monkeypatch.setattr(benchmark_scaling, "peak_rss_bytes", peak_rss)
    monkeypatch.setattr(benchmark_scaling, "current_rss_bytes", current_rss)
    monkeypatch.setattr(
        benchmark_scaling,
        "_canonical_penalty_contract",
        lambda _n_coef, _penalties: {
            "penalty_public_coordinate_sha256": "p" * 64,
            "ranks": [2],
        },
    )
    monkeypatch.setattr(
        benchmark_scaling,
        "source_identity",
        lambda _root: {"commit": "a" * 40, "dirty_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        benchmark_scaling,
        "loaded_source_identity",
        lambda _module, _root: {
            "loaded_package": "jaxgam",
            "loaded_source_sha256": "c" * 64,
        },
    )

    record = benchmark_scaling.run_setup(ScalingCase(4, 3), Path.cwd())

    assert record["model"]["n_coef"] == 3
    assert record["model"]["n_penalties"] == 2
    assert record["model"]["basis_fingerprint_kind"] == "dense_model_matrix_sha256"
    assert events[:5] == ["peak", "current", "build", "peak", "current"]
    assert events.index("fitting") > 4
    assert record["memory"]["rss_high_water_before_setup_bytes"] == 100
    assert record["memory"]["rss_current_before_setup_bytes"] == 90
    assert record["memory"]["rss_high_water_after_setup_bytes"] == 120
    assert record["memory"]["rss_current_after_setup_bytes"] == 110
    assert "excludes later FittingData" in record["memory"]["rss_scope"]
    assert record["source_scans"]["setup_max_batch_rows"] is None
    assert record["source_scans"]["setup_memory_budget_bytes"] is None


@dataclass(frozen=True)
class _Result:
    coefficients: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0]))
    smoothing_params: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    converged: bool = True
    convergence_info: str = "fixed sp"
    n_iter: int = 2
    execution_route: str = "stream"
    score: float = 1.5
    deviance: float = 2.0
    scale: float = 1.0
    score_scale: float = 1.0
    edf_total: float = 1.5

    def predict(self, data: pd.DataFrame, *, se_fit: bool = False):
        fitted = np.zeros(len(data))
        return (fitted, np.ones(len(data))) if se_fit else fitted


def test_run_case_records_completed_timing_and_honest_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pd.DataFrame({"x": [0.1, 0.2], "y": [0.0, 1.0]})
    monkeypatch.setattr(benchmark_scaling, "_make_data", lambda _case: data)
    monkeypatch.setattr(
        benchmark_scaling,
        "_build_fit",
        lambda _case, _data: benchmark_scaling._FitPlan(
            lambda: _Result(), "y ~ s(x, k=3, bs='cr')", None
        ),
    )
    monkeypatch.setattr(
        benchmark_scaling,
        "source_identity",
        lambda _root: {"commit": "a" * 40, "dirty_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        benchmark_scaling,
        "loaded_source_identity",
        lambda _module, _root: {
            "loaded_package": "jaxgam",
            "loaded_source_sha256": "c" * 64,
        },
    )

    record = benchmark_scaling.run_case(
        ScalingCase(
            n_rows=2,
            basis_dimension=3,
            execution="stream",
            result_mode="prediction",
        ),
        Path.cwd(),
    )

    assert record["result"]["execution_route"] == "stream"
    assert record["result"]["cold_fit_validity"]["valid"]
    assert all(item["valid"] for item in record["result"]["warm_fit_validity"])
    assert record["result"]["route"]["strategy"] == "fixed-sp streamed Cholesky PIRLS"
    assert (
        record["result"]["partial_public_dataclass_array_inventory"]["array_bytes"]
        == 24
    )
    assert record["timings"]["result_finish_s"] is None
    assert record["basis_comparison"]["status"] == "not_checked"
    assert not record["result"]["source_scans"]["available"]


@pytest.mark.parametrize(
    "invalid",
    [
        SimpleNamespace(**{**_Result().__dict__, "converged": False}),
        SimpleNamespace(**{**_Result().__dict__, "score": float("nan")}),
    ],
)
def test_timed_fit_validity_rejects_nonconvergence_and_nonfinite_fields(
    invalid: SimpleNamespace,
) -> None:
    with pytest.raises(RuntimeError, match="warm fit 1 returned an invalid fit"):
        benchmark_scaling._require_valid_timed_fit(invalid, label="warm fit 1")


def _paired_record() -> dict[str, object]:
    return {
        "setup": {
            "formula": "y ~ s(x)",
            "route": {
                "requested_execution": "dense",
                "adapter": "public",
                "strategy": "fixed-sp dense PIRLS",
                "result_mode": "prediction",
            },
            "input": {"generator": "gaussian_sine", "seed": 42, "frame_sha256": "x"},
            "runtime": {
                "commit": "a" * 40,
                "dirty_sha256": "b" * 64,
                "loaded_source_sha256": "c" * 64,
                "harness_sha256": "d" * 64,
            },
            "model": {
                "basis_fingerprint_kind": "prepared",
                "basis_fingerprint": "basis",
                "n_coef": 4,
                "n_penalties": 1,
                "frozen_contract": {
                    "basis_fingerprint": "basis",
                    "penalty_transform_rank_sha256": "penalty",
                    "unpenalized_rank_deficit": 0,
                },
            },
        },
        "fit": {
            "formula": "y ~ s(x)",
            "route": {
                "requested_execution": "dense",
                "adapter": "public",
                "strategy": "fixed-sp dense PIRLS",
                "result_mode": "prediction",
            },
            "input": {"generator": "gaussian_sine", "seed": 42, "frame_sha256": "x"},
            "runtime": {
                "commit": "a" * 40,
                "dirty_sha256": "b" * 64,
                "loaded_source_sha256": "c" * 64,
                "harness_sha256": "d" * 64,
            },
            "result": {
                "n_coef": 4,
                "n_penalties": 1,
                "coefficient_sha256": "coefficients",
                "prediction_sha256": "prediction",
                "score": 1.0,
                "comparison_vectors": {
                    "coefficient_values": [1.0],
                    "prediction_values": [2.0],
                },
            },
        },
        "within_run_consistency": {"valid": True},
    }


def test_compare_checkouts_requires_all_parity_observables() -> None:
    record = _paired_record()
    comparison = benchmark_scaling.compare_checkouts(record, record)
    assert comparison["status"] == "comparable"
    assert comparison["same_basis_fingerprint"]
    assert comparison["numerical_parity"]["coefficients"]["strict_allclose"]


def test_compare_checkouts_fails_closed_for_different_route_contract() -> None:
    baseline = _paired_record()
    current = _paired_record()
    current["setup"]["route"]["strategy"] = "fixed-sp streamed pivoted-QR PIRLS"  # type: ignore[index]

    comparison = benchmark_scaling.compare_checkouts(baseline, current)

    assert comparison["status"] == "incompatible"
    assert not comparison["same_route_contract"]


def test_compare_checkouts_fails_closed_for_changed_frozen_penalty_contract() -> None:
    baseline = _paired_record()
    current = _paired_record()
    current["setup"]["model"]["frozen_contract"][  # type: ignore[index]
        "penalty_transform_rank_sha256"
    ] = "changed"

    comparison = benchmark_scaling.compare_checkouts(baseline, current)

    assert comparison["status"] == "incompatible"


def test_tiny_public_dense_stream_smoke_has_route_scan_and_tolerant_parity(
    tmp_path: Path, minimal_git_checkout: Path
) -> None:
    """This is an API smoke, not a reported performance measurement."""
    completed = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "scripts" / "benchmark_scaling.py"),
            "--smoke",
            "--checkout",
            str(minimal_git_checkout),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "jax-cache"),
            "PYTHONPATH": str(minimal_git_checkout),
        },
    )
    record = json.loads(completed.stdout)

    collector = _AssertCollector()
    collector.check(
        "dense route",
        lambda: check_that(
            record["dense_execution_route"] == "dense", "wrong dense route"
        ),
    )
    collector.check(
        "stream route",
        lambda: check_that(
            record["stream_execution_route"] == "stream", "wrong streamed route"
        ),
    )
    collector.check(
        "source scans",
        lambda: check_that(record["source_scan_calls"] > 0, "source was never scanned"),
    )
    collector.check(
        "public scan count",
        lambda: check_that(
            record["source_scans_publicly_available"]
            and 0 < record["source_scans_public"] <= record["source_scan_calls"],
            "public fit scan count is inconsistent with measured total scans",
        ),
    )
    collector.check(
        "coefficients",
        lambda: np.testing.assert_allclose(
            record["coefficient_max_abs_difference"],
            0.0,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "prediction",
        lambda: np.testing.assert_allclose(
            record["prediction_max_abs_difference"],
            0.0,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "score",
        lambda: np.testing.assert_allclose(
            record["score_abs_difference"],
            0.0,
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any("tiny public dense/stream harness smoke")


def test_run_child_uses_task_owned_cache_and_explicit_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        captured["args"] = args
        return SimpleNamespace(stdout='{"ok": true}', stderr="", returncode=0)

    monkeypatch.setattr(benchmark_scaling.subprocess, "run", fake_run)

    record = benchmark_scaling._run_child(ScalingCase(4, 3), Path.cwd(), "fit")

    assert record == {"ok": True}
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"] == str(Path.cwd())
    assert "jaxgam-benchmark-jax-" in environment["JAX_COMPILATION_CACHE_DIR"]


def test_run_child_preserves_actionable_failure_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark_scaling.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7, stdout="child stdout", stderr="child stderr"
        ),
    )

    with pytest.raises(benchmark_scaling.ChildRunError) as error:
        benchmark_scaling._run_child(ScalingCase(4, 3), Path.cwd(), "setup")

    assert error.value.stdout == "child stdout"
    assert error.value.stderr == "child stderr"
    assert error.value.returncode == 7
    assert "exit 7" in str(error.value)
    assert "child stderr" in str(error.value)


def test_failed_child_diagnostics_keep_bounded_error_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark_scaling.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2,
            stdout="ignored setup output\n",
            stderr="old log line\n" * 1_000 + "projected gradient 23.5 > 1e-6\n",
        ),
    )

    with pytest.raises(benchmark_scaling.ChildRunError) as error:
        benchmark_scaling._run_child(ScalingCase(4, 3), Path.cwd(), "fit")

    assert "projected gradient 23.5 > 1e-6" in str(error.value)
    assert "earlier characters omitted" in error.value.stderr
    assert len(error.value.stderr) < 4_200
    assert "old log line" not in error.value.stderr.splitlines()[0]


def test_cli_reports_failed_child_stderr(minimal_git_checkout: Path) -> None:
    (minimal_git_checkout / "jaxgam" / "__init__.py").write_text(
        'raise RuntimeError("CHILD_DIAGNOSTIC")\n'
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "scripts" / "benchmark_scaling.py"),
            "--n",
            "8",
            "--k",
            "3",
            "--checkout",
            str(minimal_git_checkout),
        ],
        cwd=minimal_git_checkout,
        env=os.environ | {"PYTHONPATH": str(minimal_git_checkout)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "setup child failed" in completed.stderr
    assert "child stderr:" in completed.stderr
    assert "CHILD_DIAGNOSTIC" in completed.stderr


def test_run_isolated_delegates_setup_and_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    setup = _paired_record()["setup"]
    fit = _paired_record()["fit"]
    monkeypatch.setattr(
        benchmark_scaling,
        "_run_child",
        lambda _case, _root, mode: setup if mode == "setup" else fit,
    )

    record = benchmark_scaling.run_isolated(ScalingCase(4, 3), Path.cwd())
    assert record["within_run_consistency"]["valid"]


def test_main_child_and_paired_modes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    child_record = {"child": True}
    monkeypatch.setattr(
        benchmark_scaling, "run_setup", lambda _case, _root: child_record
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_scaling.py",
            "--child",
            "setup",
            "--case",
            '{"n_rows": 4, "basis_dimension": 3}',
        ],
    )
    benchmark_scaling.main()
    assert '"child": true' in capsys.readouterr().out

    record = _paired_record()
    monkeypatch.setattr(benchmark_scaling, "run_isolated", lambda _case, _root: record)
    output = tmp_path / "record.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_scaling.py",
            "--n",
            "4",
            "--k",
            "3",
            "--baseline-checkout",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    benchmark_scaling.main()

    assert '"parity"' in output.read_text()


def test_main_memory_pair_runs_ordered_isolated_cases(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    seen_rows: list[int] = []

    def run(case: ScalingCase, _root: Path) -> dict:
        seen_rows.append(case.n_rows)
        return {"case": {"n_rows": case.n_rows}}

    monkeypatch.setattr(benchmark_scaling, "run_isolated", run)
    monkeypatch.setattr(
        benchmark_scaling,
        "compare_memory_growth",
        lambda small, large: {
            "small": small["case"]["n_rows"],
            "large": large["case"]["n_rows"],
        },
    )
    output = tmp_path / "memory.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_scaling.py",
            "--n",
            "31",
            "--k",
            "6",
            "--execution",
            "stream-exact",
            "--result-mode",
            "prediction",
            "--memory-pair-n",
            "31",
            "62",
            "--output",
            str(output),
        ],
    )
    benchmark_scaling.main()
    assert seen_rows == [31, 62]
    assert json.loads(output.read_text())["memory_pair"]["comparison"] == {
        "small": 31,
        "large": 62,
    }
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())


def test_main_r_parity_attaches_only_explicit_typed_reference(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen_cases: list[ScalingCase] = []
    monkeypatch.setattr(
        benchmark_scaling, "run_isolated", lambda _case, _root: {"fit": "measured"}
    )

    def compare(case: ScalingCase, record: dict) -> dict:
        seen_cases.append(case)
        assert record == {"fit": "measured"}
        return {"reference": "typed pinned RBridge"}

    monkeypatch.setattr(benchmark_scaling, "compare_pinned_r_exact", compare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_scaling.py",
            "--n",
            "31",
            "--k",
            "6",
            "--execution",
            "stream-exact",
            "--result-mode",
            "prediction",
            "--r-parity",
        ],
    )
    benchmark_scaling.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["pinned_r_parity"]["reference"] == "typed pinned RBridge"
    assert seen_cases[0].execution == "stream-exact"
