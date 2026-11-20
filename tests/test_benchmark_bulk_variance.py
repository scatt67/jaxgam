"""Operator bulk Fisher variance benchmark (scripts/benchmark_bulk_variance.py)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from jaxgam.execution.operator.bulk_variance import equivalent_normal_deviates
from jaxgam.results import BULK_CALIBRATION_MAX_STANDARD_ERRORS
from scripts import benchmark_bulk_variance


def test_clopper_pearson_matches_its_binomial_tail_definition() -> None:
    from scipy.stats import binom

    lower, upper = benchmark_bulk_variance.clopper_pearson(2, 240)
    assert binom.sf(1, 240, lower) == pytest.approx(0.025, rel=1e-9)
    assert binom.cdf(2, 240, upper) == pytest.approx(0.025, rel=1e-9)
    none_lower, none_upper = benchmark_bulk_variance.clopper_pearson(0, 40)
    assert none_lower == 0.0
    assert binom.cdf(0, 40, none_upper) == pytest.approx(0.025, rel=1e-9)
    assert benchmark_bulk_variance.clopper_pearson(5, 5)[1] == 1.0


def test_superseded_statistic_matches_the_old_provider_rule() -> None:
    estimated = np.array([2.0, 1.0, 3.0, 4.0])
    exact = np.array([1.0, 1.0, 3.0, 2.0])
    error = np.array([0.5, 0.0, 0.0, 4.0])

    standardized = benchmark_bulk_variance.superseded_standardized_errors(
        estimated, exact, error
    )

    # |e - x| / se; a zero residual is 0; a residual over a zero error is inf.
    np.testing.assert_array_equal(standardized, [2.0, 0.0, 0.0, 0.5])
    infinite = benchmark_bulk_variance.superseded_standardized_errors(
        np.array([2.0]), np.array([1.0]), np.array([0.0])
    )
    assert np.isinf(infinite[0])


def test_summary_counts_both_rules_and_checks_the_chi_square_ratio() -> None:
    from scipy.stats import chi2

    probes = 48
    # Exact chi-square quantiles: an ideal sample, with no random draw.
    quantiles = chi2.ppf((np.arange(400) + 0.5) / 400, probes) / probes
    ratios = np.column_stack([quantiles, quantiles[::-1], np.roll(quantiles, 7)])
    superseded = np.r_[np.full(398, 1.0), 6.0, 5.5]
    current = np.r_[np.full(399, 2.0), 5.25]
    seeds = [
        {
            "superseded_maximum_standardized_error": old,
            "maximum_absolute_deviate": new,
            "ratio": ratio.tolist(),
        }
        for old, new, ratio in zip(superseded, current, ratios, strict=True)
    ]
    summary = benchmark_bulk_variance.summarize(seeds, probes, 5.0)

    assert summary["limit_standard_errors"] == 5.0
    old_rule, new_rule = summary["superseded_rule"], summary["current_rule"]
    assert old_rule["exceedances"] == 2
    assert old_rule["rate"] == 2 / 400
    assert old_rule["maximum_quantiles"]["1.0"] == 6.0
    assert new_rule["exceedances"] == 1
    assert new_rule["rate"] == 1 / 400
    assert new_rule["rate_clopper_pearson_95"] == (
        benchmark_bulk_variance.clopper_pearson(1, 400)
    )
    assert "chi-square pivot" in new_rule["statistic"]
    assert summary["ratio_mean"] == pytest.approx(1.0, rel=1e-3)
    assert summary["ratio_variance"] == pytest.approx(2 / probes, rel=0.05)
    assert all(row["p_value"] > 0.99 for row in summary["ratio_ks_per_row"])
    shifted = [{**seed, "ratio": [2 * r for r in seed["ratio"]]} for seed in seeds]
    assert (
        benchmark_bulk_variance.summarize(shifted, probes, 5.0)["ratio_ks_pooled_rows"][
            "p_value"
        ]
        < 1e-6
    )


def test_task_owned_cache_restores_the_caller_setting(monkeypatch) -> None:
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/caller/cache")
    with benchmark_bulk_variance.task_owned_cache() as cache:
        assert os.environ["JAX_COMPILATION_CACHE_DIR"] == cache
        assert Path(cache).is_dir()
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == "/caller/cache"
    assert not Path(cache).exists()
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR")
    with benchmark_bulk_variance.task_owned_cache():
        pass
    assert "JAX_COMPILATION_CACHE_DIR" not in os.environ


@pytest.mark.parametrize(
    ("arguments", "model_fields"),
    [
        (
            ["--family", "gaussian"],
            {"levels": None, "fisher_scale_source": "supplied (--scale)"},
        ),
        (
            [
                "--model",
                "re",
                "--levels",
                "15",
                "--rows-per-level",
                "3",
                "--family",
                "poisson",
            ],
            {"levels": 15, "fisher_scale_source": "fitted", "n_rows": 45},
        ),
    ],
)
def test_cli_records_every_seed_row_and_diagnostic(
    tmp_path: Path, arguments: list[str], model_fields: dict
) -> None:
    output = tmp_path / "bulk.json"
    benchmark_bulk_variance.main(
        [*arguments, "--seeds", "3", "--probes", "8", "--output", str(output)]
    )
    record = json.loads(output.read_text())

    for key, value in model_fields.items():
        assert record["case"][key] == value
    assert len(record["seeds"]) == 3
    assert [seed["seed"] for seed in record["seeds"]] == [20000, 20001, 20002]
    for seed in record["seeds"]:
        exact = np.asarray(seed["exact"])
        estimated = np.asarray(seed["estimated"])
        error = np.asarray(seed["monte_carlo_se"])
        np.testing.assert_allclose(seed["ratio"], estimated / exact)
        np.testing.assert_allclose(
            seed["superseded_standardized_error"], np.abs(estimated - exact) / error
        )
        assert seed["superseded_maximum_standardized_error"] == pytest.approx(
            max(seed["superseded_standardized_error"])
        )
        assert seed["superseded_rule_would_raise"] == (
            seed["superseded_maximum_standardized_error"]
            > BULK_CALIBRATION_MAX_STANDARD_ERRORS
        )
        np.testing.assert_allclose(
            seed["equivalent_deviates"],
            equivalent_normal_deviates(estimated, exact, 8),
        )
        assert seed["maximum_absolute_deviate"] == pytest.approx(
            max(abs(value) for value in seed["equivalent_deviates"])
        )
        assert seed["current_rule_would_raise"] == (
            seed["maximum_absolute_deviate"] > BULK_CALIBRATION_MAX_STANDARD_ERRORS
        )
        diagnostics = seed["diagnostics"]
        assert diagnostics["probes"] == 8
        assert diagnostics["completed_solves"] >= 8
        assert diagnostics["training_source_scans"] > 0
        assert diagnostics["maximum_relative_residual"] < 1e-8
    summary = record["summary"]
    assert summary["seeds"] == 3
    assert summary["superseded_rule"]["exceedances"] == sum(
        seed["superseded_rule_would_raise"] for seed in record["seeds"]
    )
    assert summary["current_rule"]["exceedances"] == sum(
        seed["current_rule_would_raise"] for seed in record["seeds"]
    )
    assert record["schema"] == "jaxgam-bulk-variance-benchmark-v2"
    assert record["bulk_pass"]["rows"] == record["case"]["n_rows"]
    assert record["bulk_pass"]["finite"]
    assert record["runtime"]["compilation_cache"]


def test_cli_rejects_degenerate_controls(tmp_path: Path) -> None:
    for arguments in (["--probes", "1"], ["--seeds", "0"], ["--levels", "0"]):
        with pytest.raises(SystemExit):
            benchmark_bulk_variance.main(
                [*arguments, "--output", str(tmp_path / "x.json")]
            )


def test_script_imports_its_own_checkout(tmp_path: Path) -> None:
    """Run as a script, the record names this checkout's package."""
    import subprocess
    import sys

    root = Path(benchmark_bulk_variance.__file__).resolve().parents[1]
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    output = tmp_path / "bulk.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "benchmark_bulk_variance.py"),
            "--seeds",
            "1",
            "--probes",
            "2",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
    )
    loaded = json.loads(output.read_text())["runtime"]["loaded_package"]
    assert Path(loaded) == root / "jaxgam"
