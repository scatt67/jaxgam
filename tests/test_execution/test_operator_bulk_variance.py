"""Seeded Fisher bulk variance and selected exact calibration."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, chi2, norm

from jaxgam import GAM, FitControl
from jaxgam.control import OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator.bulk_variance import (
    BulkOperatorVariance,
    BulkVarianceSolveError,
    _penalty_noise,
    equivalent_normal_deviates,
)
from jaxgam.execution.operator.fit import (
    FrozenOperatorWorkingState,
    fit_operator_fixed_sp,
)
from jaxgam.execution.operator.system import FrozenScanSystem, PCGControl
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator.prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.inference.operator_geometry import OperatorPredictionGeometry
from jaxgam.links.links import IdentityLink
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    DiagonalPenalty,
    IdentityPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
from jaxgam.results import BULK_CALIBRATION_MAX_STANDARD_ERRORS
from tests.tolerances import STRICT

# Monte Carlo estimates are gated by their chi-square pivot as an equivalent
# standard normal deviate, at the public bulk_link_variance default.
_MAX_SE = BULK_CALIBRATION_MAX_STANDARD_ERRORS
# Requested PCG relative tolerance; bitwise 1e-11 (STRICT.rtol / 10 rounds to
# 1.0000000000000001e-11, so the bitwise-equal STRICT composition is used).
_PCG_RTOL = STRICT.atol * 10


def _fisher_provider(frame, formula, family, sp, *, scale=1.0, probes=32, seed=18):
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula(formula),
        source,
        control=OperatorPreparationControl(batch_rows=11),
    )
    fitted = fit_operator_fixed_sp(
        model, source, family, sp, memory_budget_bytes=8 * 1024 * 1024
    )
    assert fitted.converged
    state = FrozenOperatorWorkingState(
        model, family, fitted.coefficients, 8 * 1024 * 1024, 1024 * 1024
    )
    system = FrozenScanSystem(
        model,
        source,
        np.log(sp),
        curvature_from_rows=state.fisher_curvature,
        memory_budget_bytes=8 * 1024 * 1024,
    )
    provider = BulkOperatorVariance(
        system,
        scale=scale,
        memory_budget_bytes=16 * 1024 * 1024,
        probes=probes,
        seed=seed,
        control=PCGControl(rtol=_PCG_RTOL, max_iterations=200),
    )
    return source, fitted, provider


@pytest.mark.parametrize("family", [Gaussian(), Poisson()])
def test_bulk_fisher_approximation_has_exact_and_pinned_r_calibration(r_bridge, family):
    # Branch on the family class: a name literal ("Gaussian" against the
    # reported "gaussian") previously left both Gaussian branches dead.
    gaussian = isinstance(family, Gaussian)
    row = np.arange(47)
    x = np.linspace(0.1, 1.4, len(row))
    frame = pd.DataFrame(
        {
            "y": (
                1.3 + np.sin(2 * x) + 0.1 * np.cos(row)
                if gaussian
                else np.resize([1, 3, 2, 4, 2, 5], len(row))
            ),
            "x": x,
            "g": pd.Categorical(np.resize(["a", "b", "c", "d"], len(row))),
        }
    )
    formula = "y~s(x,bs='cr',k=6)+s(g,bs='re')"
    sp = np.array([0.7, 1.1])
    reference = r_bridge.fit_fixed_sp(
        formula,
        frame,
        family.family_name.lower(),
        sp,
        with_uncertainty=True,
    )
    _source, fitted, provider = _fisher_provider(
        frame, formula, family, sp, scale=reference["scale"], probes=48
    )
    # The approximation is assessed against both exact operator solves and
    # the independent pinned mgcv covariance, with its measured MC error.
    provider.build()
    selected = frame.iloc[[0, 2, 13, 32, 46]].copy()
    calibration = provider.calibrate(selected)
    np.testing.assert_allclose(
        calibration.exact,
        reference["link_se"][[0, 2, 13, 32, 46]] ** 2,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_array_equal(
        calibration.equivalent_deviates,
        equivalent_normal_deviates(
            calibration.estimated, calibration.exact, provider.probes
        ),
    )
    assert np.all(np.abs(calibration.equivalent_deviates) <= _MAX_SE)
    assert calibration.maximum_absolute_deviate == np.max(
        np.abs(calibration.equivalent_deviates)
    )
    assert provider.diagnostics.calibration_rows == len(selected)
    assert provider.diagnostics.calibration_source_scans == (
        calibration.selected_source_scans
    )
    if gaussian:
        # Multiple independent seeds must calibrate against the same pinned
        # oracle; one favorable random draw is not the acceptance evidence.
        for held_out_seed in (9157, 27182, 31415):
            held_out = BulkOperatorVariance(
                provider.system,
                scale=provider.scale,
                memory_budget_bytes=16 * 1024 * 1024,
                probes=48,
                seed=held_out_seed,
                control=PCGControl(rtol=_PCG_RTOL, max_iterations=200),
            ).build()
            held_out_calibration = held_out.calibrate(selected)
            assert held_out.diagnostics.fisher_construction_scans == (
                provider.diagnostics.fisher_construction_scans
            )
            np.testing.assert_allclose(
                held_out_calibration.exact,
                calibration.exact,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            )
            assert held_out_calibration.maximum_absolute_deviate <= _MAX_SE
            assert not np.array_equal(
                held_out_calibration.estimated, calibration.estimated
            )
    prediction = DataFrameRowSource(frame, response="y")
    batches = list(provider.estimate_iter(prediction))
    assert sum(len(batch.positions) for batch in batches) == len(frame)
    bulk_variance = np.concatenate([batch.variance for batch in batches])
    bulk_monte_carlo_se = np.concatenate([batch.monte_carlo_se for batch in batches])
    assert np.all(bulk_monte_carlo_se > 0)
    assert np.all(
        np.abs(
            equivalent_normal_deviates(
                bulk_variance, reference["link_se"] ** 2, provider.probes
            )
        )
        <= _MAX_SE
    )
    np.testing.assert_allclose(
        np.concatenate([batch.link_se**2 for batch in batches]),
        bulk_variance,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert all(not batch.variance.flags.writeable for batch in batches)
    assert all(not batch.monte_carlo_se.flags.writeable for batch in batches)
    assert provider.diagnostics.prediction_source_scans == 1
    assert provider.diagnostics.training_source_scans == (
        provider.probes + provider.diagnostics.normal_actions
    )
    assert provider.diagnostics.maximum_relative_residual <= provider.control.rtol
    assert fitted.coefficients.shape == (provider.geometry.n_coef,)


def test_bulk_symbolic_large_p_preflights_and_never_materializes_covariance(
    monkeypatch,
):
    n = 43
    row = np.arange(n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.1 * np.sin(row),
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
    state = FrozenOperatorWorkingState(
        model, Gaussian(), np.zeros(model.n_coef), 8 * 1024 * 1024, 1024 * 1024
    )
    system = FrozenScanSystem(
        model,
        source,
        np.log(np.array([0.8, 1.1])),
        curvature_from_rows=state.fisher_curvature,
        memory_budget_bytes=8 * 1024 * 1024,
    )
    monkeypatch.setattr(
        PenaltyStructure,
        "materialize",
        lambda *_a, **_k: pytest.fail("Global penalty materialized"),
    )
    monkeypatch.setattr(
        OperatorPredictionGeometry,
        "build_predict_matrix",
        lambda *_a: pytest.fail("Expanded prediction matrix materialized"),
    )
    underbudgeted = system.memory_budget_bytes + 1
    with pytest.raises(MemoryError, match="requires"):
        BulkOperatorVariance(
            system,
            scale=1,
            memory_budget_bytes=underbudgeted,
            probes=4,
            seed=3,
        )
    provider = BulkOperatorVariance(
        system,
        scale=1,
        memory_budget_bytes=16 * 1024 * 1024,
        probes=4,
        seed=3,
    ).build()
    assert provider.diagnostics.peak_bound_bytes < 16 * 1024 * 1024
    with pytest.raises(MemoryError, match="requires"):
        BulkOperatorVariance(
            system,
            scale=1,
            memory_budget_bytes=provider.peak_bound_bytes - 1,
            probes=4,
            seed=3,
        )
    batches = list(provider.estimate_iter(source))
    assert len(np.concatenate([batch.variance for batch in batches])) == n
    assert provider.diagnostics.normal_actions == (
        provider.diagnostics.training_source_scans - provider.probes
    )
    repeated = DataFrameRowSource(
        frame, response="y", row_selection=np.array([11, 0, 11, 42])
    )
    selected_batches = list(provider.estimate_iter(repeated))
    np.testing.assert_array_equal(
        np.concatenate([batch.positions for batch in selected_batches]),
        [11, 0, 11, 42],
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.variance for batch in selected_batches])[[0, 2]],
        np.repeat(selected_batches[0].variance[0], 2),
    )


def test_public_operator_bulk_result_replays_fit_and_reports_approximation(monkeypatch):
    row = np.arange(37)
    frame = pd.DataFrame(
        {
            "y": np.resize([1, 3, 2, 4, 2, 5], len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    result = GAM(
        "y~s(g,bs='re')",
        family=Poisson(),
        sp=np.array([1.1]),
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=11,
            operator=OperatorControl(cg_rtol=_PCG_RTOL),
        ),
    ).fit(source, result="prediction")
    selected = frame.iloc[[0, 3, 9, 20]].copy()
    control = PCGControl(rtol=_PCG_RTOL, max_iterations=200)
    seen_controls = []
    solve = FrozenScanSystem.solve

    def capture(self, rhs, *, initial=None, control):
        seen_controls.append(control)
        return solve(self, rhs, initial=initial, control=control)

    monkeypatch.setattr(FrozenScanSystem, "solve", capture)
    provider, calibration = result.bulk_link_variance(
        source,
        selected,
        probes=32,
        seed=29,
        memory_budget_bytes=16 * 1024 * 1024,
        control=control,
    )
    assert provider.control is control
    assert len(seen_controls) == 32 + len(selected)
    assert all(observed is control for observed in seen_controls)
    assert provider.diagnostics.approximation == "fisher_covariance_monte_carlo"
    assert provider.diagnostics.replay_preparation_scans >= 1
    assert provider.diagnostics.fisher_construction_scans >= 1
    assert provider.system.diagnostics.completed_scans == (
        provider.diagnostics.fisher_construction_scans
        + provider.diagnostics.training_source_scans
        + provider.diagnostics.calibration_source_scans
    )
    assert calibration.maximum_absolute_deviate <= _MAX_SE
    with pytest.raises(ValueError, match="at least one row"):
        result.bulk_link_variance(
            source,
            frame.iloc[[]],
            probes=2,
            seed=29,
            memory_budget_bytes=16 * 1024 * 1024,
        )
    chunks = list(provider.estimate_iter(source))
    assert sum(len(chunk.positions) for chunk in chunks) == len(frame)
    assert provider.diagnostics.prediction_source_scans == 1
    changed = frame.copy()
    changed.loc[0, "y"] += 1
    with pytest.raises(RuntimeError, match="differs"):
        result.bulk_link_variance(
            DataFrameRowSource(changed, response="y"),
            selected,
            probes=2,
            seed=29,
            memory_budget_bytes=16 * 1024 * 1024,
        )
    result.family.link = IdentityLink()
    with pytest.raises(RuntimeError, match="Family, link or theta changed"):
        result.bulk_link_variance(
            source,
            selected,
            probes=2,
            seed=29,
            memory_budget_bytes=16 * 1024 * 1024,
        )


def test_bulk_probe_failure_keeps_partial_sketch_unavailable():
    row = np.arange(31)
    frame = pd.DataFrame(
        {
            "y": 1.3 + np.sin(row / 9),
            "x": np.linspace(0.1, 1.4, len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    _, _, reference = _fisher_provider(
        frame,
        "y~s(x,bs='cr',k=6)+s(g,bs='re')",
        Gaussian(),
        np.array([0.7, 1.1]),
        probes=2,
    )
    provider = BulkOperatorVariance(
        reference.system,
        scale=1,
        memory_budget_bytes=16 * 1024 * 1024,
        probes=2,
        seed=8,
        control=PCGControl(rtol=1e-15, max_iterations=1),
    )
    with pytest.raises(BulkVarianceSolveError) as failure:
        provider.build()
    assert failure.value.probe == 0
    with pytest.raises(RuntimeError, match="not been completed"):
        _ = provider.diagnostics
    with pytest.raises(RuntimeError, match="Build"):
        list(provider.estimate_iter(DataFrameRowSource(frame, response="y")))


def test_bulk_pcg_bias_is_separate_from_monte_carlo_error(monkeypatch):
    row = np.arange(31)
    frame = pd.DataFrame(
        {
            "y": 1.3 + np.sin(row / 9),
            "x": np.linspace(0.1, 1.4, len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    formula = "y~s(x,bs='cr',k=6)+s(g,bs='re')"
    sp = np.array([0.7, 1.1])
    _, _, provider = _fisher_provider(
        frame, formula, Gaussian(), sp, probes=8, seed=991
    )
    right_hand_sides = []
    solve = provider.system.solve

    def capture(rhs, *, control):
        right_hand_sides.append(rhs.copy())
        return solve(rhs, control=control)

    monkeypatch.setattr(provider.system, "solve", capture)
    provider.build()
    setup = ModelSetup.build(parse_formula(formula), frame)
    hessian = setup.X.T @ setup.X + setup.penalties.materialize(np.log(sp))
    exact_samples = np.linalg.solve(hessian, np.array(right_hand_sides).T).T
    approximate_samples = provider._samples
    assert approximate_samples is not None
    relative_solution_error = np.max(
        np.linalg.norm(approximate_samples - exact_samples, axis=1)
        / np.linalg.norm(exact_samples, axis=1)
    )
    # A residual-based condition bound addresses PCG bias independently of
    # the across-probe Monte Carlo standard error reported for variance.
    condition_bound = (
        np.linalg.cond(hessian)
        * provider.diagnostics.maximum_relative_residual
        * (1 + 64 * np.finfo(float).eps)
    )
    assert relative_solution_error <= condition_bound + STRICT.atol
    assert provider.diagnostics.maximum_relative_residual <= provider.control.rtol


def test_local_penalty_noise_preserves_null_spaces_and_rejects_negative_energy():
    dense = PenaltyBlock(
        0,
        2,
        (0,),
        (DenseLocalPenalty(np.array([[1.0, -1.0], [-1.0, 1.0]])),),
        IdentityTransform(2),
        (1,),
    )
    rng = np.random.default_rng(337)
    draws = np.array([_penalty_noise(dense, np.log([2.0]), rng) for _ in range(2048)])
    np.testing.assert_allclose(draws.sum(axis=1), 0, atol=1e-13)
    observed = np.mean(draws[:, 0] ** 2)
    expected = 2.0
    empirical_standard_error = np.std(draws[:, 0] ** 2, ddof=1) / np.sqrt(len(draws))
    assert abs(observed - expected) <= _MAX_SE * empirical_standard_error

    diagonal = PenaltyBlock(
        0,
        2,
        (0, 1),
        (DiagonalPenalty(np.array([0.0, 3.0])), IdentityPenalty(2, 0.5)),
        IdentityTransform(2),
        (1, 2),
    )
    one = _penalty_noise(diagonal, np.log([2.0, 4.0]), rng)
    assert one.shape == (2,)
    assert np.all(np.isfinite(one))
    empty = PenaltyBlock(0, 0, (0,), (IdentityPenalty(0),), IdentityTransform(0), (0,))
    assert _penalty_noise(empty, np.array([0.0]), rng).shape == (0,)
    negative = PenaltyBlock(
        0,
        1,
        (0, 1),
        (DiagonalPenalty(np.array([-2.0])), IdentityPenalty(1, 0.5)),
        IdentityTransform(1),
        (0, 1),
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        _penalty_noise(negative, np.array([0.0, 0.0]), rng)


def test_bulk_prediction_rejects_broken_replay_and_invalid_positions():
    row = np.arange(19)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.1 * row,
            "g": pd.Categorical(np.resize(["a", "b"], len(row))),
        }
    )
    source, _, provider = _fisher_provider(
        frame, "y~s(g,bs='re')", Gaussian(), np.array([1.0]), probes=2
    )
    provider.build()

    class ShortReplay:
        @property
        def n_rows(self):
            return source.n_rows + 1

        def fingerprint(self):
            return source.fingerprint()

        def scan(self, batch_rows):
            yield from source.scan(batch_rows)

    with pytest.raises(RuntimeError, match="changed during replay"):
        list(provider.estimate_iter(ShortReplay()))

    class NegativePositionReplay(ShortReplay):
        @property
        def n_rows(self):
            return source.n_rows

        def scan(self, batch_rows):
            for batch in source.scan(batch_rows):
                yield replace(batch, row_positions=-batch.row_positions - 1)

    with pytest.raises(ValueError, match="nonnegative integers"):
        list(provider.estimate_iter(NegativePositionReplay()))


def test_public_tensor_tprs_bulk_variance_uses_source_matched_basis(r_bridge):
    """The composed source TPRS chain also supports public bulk variance."""
    from jaxgam import GAM, FitControl

    n = 61
    x = np.linspace(-1.3, 1.7, n)
    z = np.sin(np.linspace(-0.8, 2.2, n)) + np.linspace(0, 0.1, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "y": np.cos(1.7 * x) + 0.2 * z,
            "w": np.linspace(0.6, 1.4, n),
            "o": 0.05 * x,
        }
    )
    frame.loc[::9, "w"] = 0.0
    sp = np.array([0.7, 0.7])
    reference = r_bridge.fit_fixed_sp(
        "y~ti(x,z,bs='tp',k=c(4,4))",
        frame,
        "gaussian",
        sp,
        weights="w",
        offset="o",
        with_uncertainty=True,
    )
    source = DataFrameRowSource(
        frame, response="y", weights=frame.w.to_numpy(), offset=frame.o.to_numpy()
    )
    result = GAM(
        "y~ti(x,z,bs='tp',k=4)",
        family=Gaussian(),
        sp=sp,
        control=FitControl(execution="operator", linear_solver="pcg", batch_rows=7),
    ).fit(source, result="prediction")
    assert result.converged
    np.testing.assert_allclose(
        result.coefficients,
        reference["coefficients"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    positions = np.array([0, 11, 29, 60])
    provider, calibration = result.bulk_link_variance(
        source,
        frame.iloc[positions],
        scale=reference["scale"],
        probes=64,
        seed=2901,
        memory_budget_bytes=16 * 1024 * 1024,
    )
    np.testing.assert_allclose(
        calibration.exact,
        reference["link_se"][positions] ** 2,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    prediction = DataFrameRowSource(frame.iloc[[0, 11, 0, 60]])
    batches = list(provider.estimate_iter(prediction))
    variance = np.concatenate([batch.variance for batch in batches])
    error = np.concatenate([batch.monte_carlo_se for batch in batches])
    assert np.all(
        np.abs(variance - reference["link_se"][[0, 11, 0, 60]] ** 2)
        <= _MAX_SE * error + STRICT.atol
    )
    np.testing.assert_array_equal(variance[[0, 2]], variance[0])


def _calibration_model():
    row = np.arange(47)
    x = np.linspace(0.1, 1.4, len(row))
    frame = pd.DataFrame(
        {
            "y": 1.3 + np.sin(2 * x) + 0.1 * np.cos(row),
            "x": x,
            "g": pd.Categorical(np.resize(["a", "b", "c", "d"], len(row))),
        }
    )
    return frame, "y~s(x,bs='cr',k=6)+s(g,bs='re')", np.array([0.7, 1.1])


def test_equivalent_normal_deviates_follow_the_chi_square_pivot():
    """Both tails to the double-precision limit; saturation beyond it; zero
    exact variance; invalid inputs."""
    probes = 32
    lower = np.array([1e-300, 1e-12, 0.5])
    upper = np.array([1e-12, 1e-300])
    pivot = np.concatenate([chi2.ppf(lower, probes), chi2.isf(upper, probes)])
    exact = np.full(len(pivot), 2.0)
    deviates = equivalent_normal_deviates(pivot * exact / probes, exact, probes)
    np.testing.assert_allclose(
        deviates,
        np.concatenate([norm.ppf(lower), norm.isf(upper)]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    saturated = equivalent_normal_deviates(
        np.array([0.0, 1e4 * 2.0 / probes]), np.full(2, 2.0), probes
    )
    np.testing.assert_array_equal(saturated, [-np.inf, np.inf])
    np.testing.assert_array_equal(
        equivalent_normal_deviates(np.array([0.0, 1.0]), np.zeros(2), probes),
        [0.0, np.inf],
    )
    for estimated, reference, count in (
        (np.ones(2), np.ones(3), probes),
        (np.ones(2), np.ones(2), 0),
        (np.array([np.nan, 1.0]), np.ones(2), probes),
        (np.array([-1.0, 1.0]), np.ones(2), probes),
    ):
        with pytest.raises((ValueError, FloatingPointError)):
            equivalent_normal_deviates(estimated, reference, count)


# Seeds for the pivot's sampling test, fixed so the test is deterministic.
_PIVOT_SEEDS = tuple(range(1000, 1240))
_PIVOT_PROBES = 8
_PIVOT_LEVELS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


@pytest.mark.slow
def test_calibration_pivot_is_standard_normal_over_seeds():
    """Over 240 fixed seeds (8 probes each, so the pivot is chi-square with 8
    degrees of freedom) each level's equivalent deviate has normal
    quantiles: the count below each normal quantile lies in its two-sided
    binomial band at a Bonferroni-corrected level of 1e-4 over the rows and
    levels tested. No call exceeds the default limit."""
    # The smallest operator model here (three random-effect levels), so the
    # real probe and solve pipeline runs a few thousand solves quickly.
    row = np.arange(37)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.3 * np.cos(row) + np.resize([-0.4, 0.1, 0.5], len(row)),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(row))),
        }
    )
    _source, _fitted, provider = _fisher_provider(
        frame, "y~s(g,bs='re')", Gaussian(), np.array([1.1]), probes=_PIVOT_PROBES
    )
    selected = frame.iloc[[0, 1, 2]].copy()  # one row per level
    exact = provider.build().calibrate(selected).exact
    deviates = []
    for seed in _PIVOT_SEEDS:
        estimated, _ = (
            BulkOperatorVariance(
                provider.system,
                scale=1.0,
                memory_budget_bytes=16 * 1024 * 1024,
                probes=_PIVOT_PROBES,
                seed=seed,
                control=provider.control,
            )
            .build()
            ._batch_estimate(selected)
        )
        deviates.append(equivalent_normal_deviates(estimated, exact, _PIVOT_PROBES))
    deviates = np.array(deviates)
    assert np.max(np.abs(deviates)) <= _MAX_SE
    seeds = len(_PIVOT_SEEDS)
    alpha = 1e-4 / (len(_PIVOT_LEVELS) * len(selected))
    for row in range(len(selected)):
        for level in _PIVOT_LEVELS:
            below = int(np.sum(deviates[:, row] < norm.ppf(level)))
            low = binom.ppf(alpha / 2, seeds, level)
            high = binom.isf(alpha / 2, seeds, level)
            assert low <= below <= high, (row, level, below, low, high)


def test_biased_bulk_sketch_fails_public_calibration(monkeypatch):
    """A sketch whose variance is scaled by four is rejected."""
    frame, formula, sp = _calibration_model()
    source = DataFrameRowSource(frame, response="y")
    result = GAM(
        formula,
        sp=sp,
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=11,
            operator=OperatorControl(cg_rtol=_PCG_RTOL),
        ),
    ).fit(source, result="prediction")
    selected = frame.iloc[[0, 2, 13, 32, 46]].copy()
    arguments = {
        "probes": 32,
        "seed": 29,
        "memory_budget_bytes": 16 * 1024 * 1024,
        "control": PCGControl(rtol=_PCG_RTOL, max_iterations=200),
        "scale": 1.0,
    }
    _provider, honest = result.bulk_link_variance(source, selected, **arguments)
    assert honest.maximum_absolute_deviate <= _MAX_SE
    build = BulkOperatorVariance.build

    def biased(self):
        built = build(self)
        scaled = 2.0 * built._samples
        scaled.setflags(write=False)
        built._samples = scaled
        return built

    monkeypatch.setattr(BulkOperatorVariance, "build", biased)
    with pytest.raises(RuntimeError, match="equivalent standard normal deviates"):
        result.bulk_link_variance(source, selected, **arguments)
