"""Output gate for the operator EFS calibration harness."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import benchmark_operator_efs_calibration as harness


def test_main_writes_matched_and_complete_reports(tmp_path: Path, capsys):
    harness.main(
        [
            "--regime",
            "indexed_poisson",
            "--levels",
            "16",
            "--smoothing",
            "1.0",
            "--complete",
            "--bound-source",
            "formula",
            "--label",
            "tiny",
            "--output-dir",
            str(tmp_path),
        ]
    )
    reports = [
        json.loads(line) for line in (tmp_path / "tiny.jsonl").read_text().splitlines()
    ]
    assert len(reports) == 1
    report = reports[0]
    assert report["schema"] == harness.SCHEMA
    assert report["case"]["n_coef"] == 16
    (matched,) = report["matched"]
    assert matched["admitted_bound"] is None
    assert matched["admitted_method"] == "unavailable"
    assert matched["envelope_total"] > 0
    assert matched["errors"]["score_total"] <= matched["envelope_total"]
    complete = report["complete"]
    assert complete["operator"]["final_bound_method"] == "external"
    assert complete["formula_violations"] == 0
    assert complete["dense"]["strict"]["score"]
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["complete"]["formula_violations"] == 0
