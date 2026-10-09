"""Bounded frozen-system log determinant and explicit uncertainty accounting."""

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator import FrozenScanSystem
from jaxgam.execution.operator_logdet import (
    FrozenOperatorLogDet,
    OperatorLogDetControl,
    OperatorLogDetError,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.penalties.structure import PenaltyStructure
from tests.tolerances import STRICT


def _system(formula: str, frame: pd.DataFrame, *, batch_rows: int = 7):
    source = DataFrameRowSource(frame, response="y")
    spec = parse_formula(formula)
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=batch_rows)
    )
    rho = np.zeros(model.penalties.n_penalties, dtype=np.float64)
    system = FrozenScanSystem(model, source, rho, memory_budget_bytes=2 * 1024 * 1024)
    dense = ModelSetup.build(spec, frame)
    matrix = dense.X.T @ dense.X + dense.penalties.materialize(rho)
    return system, matrix


@pytest.mark.parametrize("batch_rows", [1, 7])
def test_diagonal_operator_logdet_matches_direct_strict(batch_rows):
    frame = pd.DataFrame(
        {
            "y": np.linspace(0.1, 1.0, 20),
            "g": pd.Categorical(np.resize(["a", "b", "c", "d"], 20)),
        }
    )
    system, matrix = _system("y~0+s(g,bs='re')", frame, batch_rows=batch_rows)
    control = OperatorLogDetControl(probes=4, lanczos_depth=8, seed=11)
    result = FrozenOperatorLogDet(
        system, memory_budget_bytes=4 * 1024 * 1024, control=control
    ).estimate()
    expected = float(np.linalg.slogdet(matrix)[1])
    np.testing.assert_allclose(
        result.per_probe, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.value, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert result.diagnostics.exact_krylov_probes == control.probes
    assert result.diagnostics.completed_actions == result.diagnostics.source_scans
    assert result.diagnostics.peak_bound_bytes < 4 * 1024 * 1024
    assert not result.per_probe.flags.writeable


def test_each_fixed_probe_matches_dense_matrix_log_and_seed_replay():
    row = np.arange(31)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.1 * np.sin(row),
            "x": np.linspace(-1, 1, len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    system, matrix = _system("y~x+s(g,bs='re')", frame)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    assert np.min(eigenvalues) > 0
    matrix_log = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    control = OperatorLogDetControl(probes=8, lanczos_depth=matrix.shape[0], seed=13)
    provider = FrozenOperatorLogDet(
        system, memory_budget_bytes=4 * 1024 * 1024, control=control
    )
    first = provider.estimate()
    second = provider.estimate()
    rng = np.random.default_rng(control.seed)
    expected = np.array(
        [
            signs @ matrix_log @ signs
            for signs in (
                2 * rng.integers(0, 2, size=matrix.shape[0], dtype=np.int8) - 1
                for _ in range(control.probes)
            )
        ]
    )
    np.testing.assert_allclose(
        first.per_probe, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_array_equal(first.per_probe, second.per_probe)
    assert first.diagnostics.n_coef == matrix.shape[0]
    assert first.diagnostics.probe_fingerprint == second.diagnostics.probe_fingerprint
    assert first.diagnostics.exact_krylov_probes == control.probes
    assert first.diagnostics.completed_actions == first.diagnostics.source_scans
    different = FrozenOperatorLogDet(
        system,
        memory_budget_bytes=4 * 1024 * 1024,
        control=OperatorLogDetControl(
            probes=control.probes, lanczos_depth=control.lanczos_depth, seed=14
        ),
    ).estimate()
    assert np.any(different.per_probe != first.per_probe)
    assert (
        different.diagnostics.probe_fingerprint != first.diagnostics.probe_fingerprint
    )
    true_logdet = float(np.linalg.slogdet(matrix)[1])
    assert abs(first.value - true_logdet) < max(
        4 * first.diagnostics.monte_carlo_standard_error, STRICT.atol
    )


def test_same_seed_and_depth_on_different_dimensions_has_different_probes():
    control = OperatorLogDetControl(probes=4, lanczos_depth=3, seed=29)
    fingerprints = []
    for levels in (3, 4):
        frame = pd.DataFrame(
            {
                "y": np.ones(24),
                "g": pd.Categorical(
                    np.resize([f"g{index}" for index in range(levels)], 24)
                ),
            }
        )
        system, _ = _system("y~0+s(g,bs='re')", frame)
        estimate = FrozenOperatorLogDet(
            system, memory_budget_bytes=4 * 1024 * 1024, control=control
        ).estimate()
        assert estimate.diagnostics.n_coef == levels
        fingerprints.append(estimate.diagnostics.probe_fingerprint)
    assert fingerprints[0] != fingerprints[1]


def test_logdet_preflights_budget_and_controls_before_actions(monkeypatch):
    frame = pd.DataFrame(
        {"y": np.ones(12), "g": pd.Categorical(np.resize(["a", "b"], 12))}
    )
    system, _ = _system("y~0+s(g,bs='re')", frame)
    monkeypatch.setattr(
        system, "matvec", lambda *_: pytest.fail("logdet action started")
    )
    with pytest.raises(MemoryError, match="requires"):
        FrozenOperatorLogDet(
            system, memory_budget_bytes=system.memory_budget_bytes + 1
        ).estimate()
    with pytest.raises(ValueError, match="lanczos_depth"):
        OperatorLogDetControl(lanczos_depth=129)
    with pytest.raises(ValueError, match="seed"):
        OperatorLogDetControl(seed=-1)


def test_logdet_failed_action_preserves_probe_step_and_scan_status(monkeypatch):
    frame = pd.DataFrame(
        {
            "y": np.arange(24) / 24,
            "x": np.linspace(-1, 1, 24),
            "g": pd.Categorical(np.resize(["a", "b", "c"], 24)),
        }
    )
    system, _ = _system("y~x+s(g,bs='re')", frame)
    original = system.matvec
    calls = 0

    def fail_second(vector):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("source action failed")
        return original(vector)

    monkeypatch.setattr(system, "matvec", fail_second)
    provider = FrozenOperatorLogDet(
        system,
        memory_budget_bytes=4 * 1024 * 1024,
        control=OperatorLogDetControl(probes=2, lanczos_depth=5),
    )
    with pytest.raises(OperatorLogDetError, match="probe 0") as failure:
        provider.estimate()
    assert failure.value.step == 1
    assert failure.value.completed_actions == 1
    assert failure.value.source_scans == 1


def test_depth_limited_error_is_separate_from_monte_carlo_error():
    row = np.arange(25)
    frame = pd.DataFrame(
        {
            "y": row / 25,
            "x": np.linspace(-1, 1, len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    system, _ = _system("y~x+s(g,bs='re')", frame)
    result = FrozenOperatorLogDet(
        system,
        memory_budget_bytes=4 * 1024 * 1024,
        control=OperatorLogDetControl(probes=4, lanczos_depth=2, seed=5),
    ).estimate()
    assert result.diagnostics.exact_krylov_probes == 0
    assert result.diagnostics.maximum_depth_difference > 0
    assert np.isfinite(result.diagnostics.monte_carlo_standard_error)
    assert result.diagnostics.completed_actions == 8


def test_nonpositive_ritz_value_rejects_without_partial_result(monkeypatch):
    frame = pd.DataFrame(
        {"y": np.ones(12), "g": pd.Categorical(np.resize(["a", "b"], 12))}
    )
    system, _ = _system("y~0+s(g,bs='re')", frame)
    monkeypatch.setattr(system, "matvec", lambda vector: -vector)
    with pytest.raises(OperatorLogDetError, match="probe 0") as failure:
        FrozenOperatorLogDet(
            system,
            memory_budget_bytes=4 * 1024 * 1024,
            control=OperatorLogDetControl(probes=2, lanczos_depth=2),
        ).estimate()
    assert isinstance(failure.value.__cause__, np.linalg.LinAlgError)
    assert failure.value.completed_actions == 0
    assert failure.value.source_scans == 0


def test_large_symbolic_logdet_stays_below_global_square(monkeypatch):
    row = np.arange(97)
    frame = pd.DataFrame(
        {
            "y": 1.0 + row / 97,
            "g": pd.Categorical(
                [f"g{i}" for i in row % 3],
                categories=[f"g{i}" for i in range(1024)],
            ),
            "h": pd.Categorical(
                [f"h{i}" for i in row % 4],
                categories=[f"h{i}" for i in range(1024)],
            ),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(g,bs='re')+s(h,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=13),
    )
    assert model.n_coef == 2049
    system = FrozenScanSystem(
        model,
        source,
        np.zeros(2, dtype=np.float64),
        memory_budget_bytes=8 * 1024 * 1024,
    )
    monkeypatch.setattr(
        PenaltyStructure,
        "materialize",
        lambda *_a, **_k: pytest.fail("Global penalty materialized"),
    )
    result = FrozenOperatorLogDet(
        system,
        memory_budget_bytes=16 * 1024 * 1024,
        control=OperatorLogDetControl(probes=2, lanczos_depth=2, seed=23),
    ).estimate()
    assert np.isfinite(result.value)
    assert result.diagnostics.completed_actions == 4
    assert result.diagnostics.source_scans == 4
    assert result.diagnostics.peak_bound_bytes < 16 * 1024 * 1024
    assert 8 * model.n_coef**2 > 16 * 1024 * 1024


def test_nontrivial_spd_logdet_obeys_tiny_and_large_scale_identity(monkeypatch):
    row = np.arange(31)
    frame = pd.DataFrame(
        {
            "y": 1.0 + row / 31,
            "x": np.linspace(-1, 1, len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    system, matrix = _system("y~x+s(g,bs='re')", frame)
    original_matvec = system.matvec
    control = OperatorLogDetControl(probes=4, lanczos_depth=matrix.shape[0], seed=17)
    probe_results = {}
    for scale in (1e-20, 1.0, 1e20):
        monkeypatch.setattr(
            system,
            "matvec",
            lambda vector, scale=scale: scale * original_matvec(vector),
        )
        result = FrozenOperatorLogDet(
            system, memory_budget_bytes=4 * 1024 * 1024, control=control
        ).estimate()
        assert result.diagnostics.exact_krylov_probes == control.probes
        assert result.diagnostics.maximum_depth_difference == 0.0
        assert result.diagnostics.completed_actions == result.diagnostics.source_scans
        probe_results[scale] = result.per_probe
    for scale in (1e-20, 1e20):
        np.testing.assert_allclose(
            probe_results[scale] - probe_results[1.0],
            matrix.shape[0] * np.log(scale),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


def test_depth_one_nontrivial_spectrum_has_indeterminate_truncation_error():
    row = np.arange(25)
    frame = pd.DataFrame(
        {
            "y": row / 25,
            "x": np.linspace(-1, 1, len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    system, _ = _system("y~x+s(g,bs='re')", frame)
    result = FrozenOperatorLogDet(
        system,
        memory_budget_bytes=4 * 1024 * 1024,
        control=OperatorLogDetControl(probes=4, lanczos_depth=1, seed=5),
    ).estimate()
    assert np.isfinite(result.value)
    assert result.diagnostics.exact_krylov_probes == 0
    assert np.isinf(result.diagnostics.maximum_depth_difference)
    assert result.diagnostics.completed_actions == 4
