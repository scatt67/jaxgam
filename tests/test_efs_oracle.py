"""Tests for pinned EFS fixture generation and branch-forced controller traces."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.efs_oracle import (
    EFSControl,
    ScriptedFit,
    canonical_data_hash,
    make_efs_fixture,
    run_pinned_r_scripted_efs,
    run_scripted_efs_controller,
    write_efs_fixture,
)
from tests.helpers import r_available
from tests.tolerances import STRICT


def _provider(*fits: ScriptedFit):
    pending = iter(fits)

    def provide(
        _phase: str, _lsp: np.ndarray, _iteration: int, _multiplier: float
    ) -> ScriptedFit:
        return next(pending)

    return provide


def test_fixture_generator_is_deterministic_and_records_provenance(tmp_path) -> None:
    data_a, fixture_a = make_efs_fixture(seed=5, n=12)
    data_b, fixture_b = make_efs_fixture(seed=5, n=12)

    assert fixture_a == fixture_b
    assert fixture_a.data_hash == canonical_data_hash(data_a)
    np.testing.assert_allclose(
        data_a.to_numpy(), data_b.to_numpy(), rtol=STRICT.rtol, atol=STRICT.atol
    )

    path = tmp_path / "oracle.csv"
    write_efs_fixture(path, data_a, fixture_a)
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["data_hash"] == canonical_data_hash(data_a)
    assert metadata["source_commit"] == fixture_a.source_commit
    assert path.exists()


def test_fixture_generator_validates_supported_shape_and_family() -> None:
    with pytest.raises(ValueError, match="at least"):
        make_efs_fixture(n=7)
    with pytest.raises(ValueError, match="family"):
        make_efs_fixture(family="nb")


def test_initial_shift_and_candidate_cap_preserve_pinned_timing() -> None:
    trace = run_scripted_efs_controller(
        np.array([0.7]),
        np.array([1.0]),
        _provider(ScriptedFit(10, 10), ScriptedFit(9, 9)),
        control=EFSControl(lspmax=3.0, outer_limit=1),
    )
    assert trace.events[0].phase == "initial"
    np.testing.assert_allclose(
        trace.events[0].log_smoothing, [3.2], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        trace.events[1].log_smoothing, [3.0], rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_extension_accepts_only_a_strictly_better_refit() -> None:
    winning = run_scripted_efs_controller(
        np.array([0.0]),
        np.array([0.01]),
        _provider(ScriptedFit(10, 10), ScriptedFit(9, 9), ScriptedFit(8, 8)),
        control=EFSControl(outer_limit=1),
    )
    assert [event.phase for event in winning.events] == [
        "initial",
        "candidate",
        "extension",
    ]
    assert winning.multiplier == 2.0
    np.testing.assert_allclose(
        winning.accepted_log_smoothing, [2.52], rtol=STRICT.rtol, atol=STRICT.atol
    )

    losing = run_scripted_efs_controller(
        np.array([0.0]),
        np.array([0.01]),
        _provider(ScriptedFit(10, 10), ScriptedFit(9, 9), ScriptedFit(9.1, 8)),
        control=EFSControl(outer_limit=1),
    )
    assert losing.multiplier == 1.0
    np.testing.assert_allclose(
        losing.accepted_log_smoothing, [2.51], rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_contraction_stops_at_one_and_accepts_finite_worsening() -> None:
    trace = run_scripted_efs_controller(
        np.array([0.0]),
        np.array([0.01]),
        _provider(
            ScriptedFit(10, 10),  # initial
            ScriptedFit(9, 9),  # first candidate
            ScriptedFit(8, 8),  # first extension; multiplier becomes two
            ScriptedFit(11, 7),  # second candidate, worsens
            ScriptedFit(10.5, 6),  # contraction at one, still worsens and accepted
        ),
        control=EFSControl(outer_limit=2),
    )
    contraction = [event for event in trace.events if event.phase == "contraction"]
    assert len(contraction) == 1
    assert contraction[0].multiplier == 1.0
    assert trace.multiplier == 1.0
    assert trace.accepted_score == 10.5


def test_original_max_step_gates_extension_before_multiplier_changes() -> None:
    trace = run_scripted_efs_controller(
        np.array([0.0]),
        np.array([0.03]),
        _provider(
            ScriptedFit(10, 10),
            ScriptedFit(9, 9),
            ScriptedFit(8, 8),  # first extension wins; multiplier is two
            ScriptedFit(7, 7),  # original proposed step is .06: no extension
        ),
        control=EFSControl(outer_limit=2),
    )
    assert [event.phase for event in trace.events].count("extension") == 1


def test_score_window_deviance_and_limit_stops_match_efsudr_ordering() -> None:
    score_window = run_scripted_efs_controller(
        np.array([0.0]),
        np.array([0.01]),
        _provider(
            ScriptedFit(5.0, 10),
            ScriptedFit(4.99, 9),
            ScriptedFit(5.0, 8),
            ScriptedFit(4.98, 7),
            ScriptedFit(5.0, 6),
            ScriptedFit(4.97, 5),
            ScriptedFit(5.0, 4),
            ScriptedFit(4.96, 3),
            ScriptedFit(5.0, 2),
        ),
        control=EFSControl(outer_limit=10),
    )
    assert score_window.stop_reason == "score_window"

    deviance = run_scripted_efs_controller(
        np.array([0.0]),
        np.array([0.1]),
        _provider(ScriptedFit(5, 10), ScriptedFit(4, 9), ScriptedFit(3, 9)),
        control=EFSControl(outer_limit=10),
    )
    assert deviance.stop_reason == "deviance"

    # mgcv's partial control$eps match is 1e-7, so this is below the
    # 100 * eps * |deviance| boundary but far above machine-epsilon noise.
    epsilon_boundary = run_scripted_efs_controller(
        np.array([0.0]),
        np.array([0.1]),
        _provider(ScriptedFit(5, 10.0), ScriptedFit(4, 9.0), ScriptedFit(3, 9.00005)),
        control=EFSControl(outer_limit=10),
    )
    assert EFSControl().deviance_epsilon == 1e-7
    assert epsilon_boundary.stop_reason == "deviance"

    limit = run_scripted_efs_controller(
        np.array([0.0]),
        np.array([0.1]),
        lambda _phase, _lsp, iteration, _multiplier: ScriptedFit(
            100 - iteration, 100 - iteration
        ),
        control=EFSControl(outer_limit=200),
    )
    assert limit.stop_reason == "iteration_limit"
    assert sum(event.phase == "candidate" for event in limit.events) == 200
    # Once the cap makes the original proposal zero, pinned efsudr still
    # evaluates its losing extension at the same capped value.
    assert sum(event.phase == "extension" for event in limit.events) == 75


def test_controller_rejects_invalid_parameter_vectors_and_controls() -> None:
    provider = _provider(ScriptedFit(1, 1))
    with pytest.raises(ValueError, match="equal-length"):
        run_scripted_efs_controller(np.array([0.0]), np.array([0.0, 1.0]), provider)
    with pytest.raises(ValueError, match="positive"):
        run_scripted_efs_controller(
            np.array([0.0]),
            np.array([0.0]),
            provider,
            control=EFSControl(outer_limit=0),
        )


@pytest.mark.skipif(not r_available(), reason="pinned R with mgcv not available")
def test_pinned_r_scripted_extension_then_contraction_matches_python() -> None:
    fits = [
        ScriptedFit(10, 10),
        ScriptedFit(9, 9),
        ScriptedFit(8, 8),
        ScriptedFit(11, 7),
        ScriptedFit(10.5, 8),
    ]
    python = run_scripted_efs_controller(
        np.array([0.0]),
        np.array([0.01]),
        _provider(*fits),
        control=EFSControl(outer_limit=2),
    )
    oracle = run_pinned_r_scripted_efs(np.array([0.0]), np.array([0.01]), fits)
    np.testing.assert_allclose(
        oracle["proposals"],
        [event.log_smoothing for event in python.events],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        oracle["accepted_sp"],
        np.exp(python.accepted_log_smoothing),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        oracle["score_history"],
        [8.0, 10.5],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert python.stop_reason == "deviance"
    assert oracle["final_score"] == python.accepted_score


@pytest.mark.skipif(not r_available(), reason="pinned R with mgcv not available")
def test_pinned_r_scripted_helper_supports_two_parameters() -> None:
    fits = [
        ScriptedFit(10, 10),
        ScriptedFit(9, 9),
        ScriptedFit(8, 8),
        ScriptedFit(11, 7),
        ScriptedFit(10.5, 8),
    ]
    python = run_scripted_efs_controller(
        np.array([0.0, 0.25]),
        np.array([0.01, 0.02]),
        _provider(*fits),
        control=EFSControl(outer_limit=2),
    )
    oracle = run_pinned_r_scripted_efs(
        np.array([0.0, 0.25]), np.array([0.01, 0.02]), fits
    )

    np.testing.assert_allclose(
        oracle["proposals"],
        [event.log_smoothing for event in python.events],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        oracle["accepted_sp"],
        np.exp(python.accepted_log_smoothing),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert python.stop_reason == "deviance"


@pytest.mark.skipif(not r_available(), reason="pinned R with mgcv not available")
def test_pinned_r_scripted_rejected_extension_returns_accepted_state() -> None:
    fits = [
        ScriptedFit(10, 10),
        ScriptedFit(9, 9),
        ScriptedFit(9.1, 8),
        ScriptedFit(8, 9),
        ScriptedFit(9, 8),
    ]
    oracle = run_pinned_r_scripted_efs(np.array([0.0]), np.array([0.01]), fits)

    np.testing.assert_allclose(
        oracle["proposals"][-2], [2.52], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        oracle["proposals"][-1], [2.53], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        oracle["accepted_sp"],
        np.exp([2.52]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
