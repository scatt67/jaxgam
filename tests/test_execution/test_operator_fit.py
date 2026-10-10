"""Family initialization and bounded fixed-sp operator controller contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from jaxgam.api import GAM
from jaxgam.control import FitControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator_fit import (
    FrozenOperatorWorkingState,
    _evaluate_geometry,
    _initial_null_beta,
    fit_operator_fixed_sp,
)
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.links.links import InverseLink, LogLink
from jaxgam.penalties.structure import PenaltyStructure
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


class _EmptyBatches:
    def __init__(self, source):
        self.source = source

    @property
    def n_rows(self):
        return self.source.n_rows

    def fingerprint(self):
        return self.source.fingerprint()

    def scan(self, batch_rows):
        for batch in self.source.scan(batch_rows):
            yield replace(
                batch,
                columns={name: value.iloc[:0] for name, value in batch.columns.items()},
                y=batch.y[:0],
                weight=batch.weight[:0],
                offset=batch.offset[:0],
                valid=batch.valid[:0],
                row_positions=batch.row_positions[:0],
            )
            yield batch


def test_public_fixed_sp_operator_counts_short_source_partitions_and_probes():
    n = 36
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1, 1, n),
            "y": np.resize([0, 1, 2, 1], n),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
        }
    )

    class PartitionedSource(DataFrameRowSource):
        def __init__(self):
            super().__init__(frame, response="y")
            self.scans = 0
            self.batches = 0

        def scan(self, batch_rows):
            self.scans += 1
            for batch in super().scan(min(batch_rows, 3)):
                self.batches += 1
                yield batch

    source = PartitionedSource()
    fitted = GAM(
        "y~x+s(g,bs='re')",
        family="poisson",
        sp=[0.8],
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=8,
            memory_budget_bytes=32 << 20,
        ),
    ).fit(source, result="prediction")
    assert fitted.converged
    assert fitted.source_scans == source.scans
    assert fitted.batches_scanned == source.batches
    assert fitted.batches_scanned > fitted.source_scans * ((n + 7) // 8)
    assert fitted.operator_diagnostics is not None
    assert fitted.source_scans > (
        fitted.operator_diagnostics.preparation_scans
        + fitted.operator_diagnostics.source.started_scans
    )


def test_public_fixed_sp_operator_solves_active_correlated_large_factor(monkeypatch):
    levels = 2048
    group = np.tile(np.arange(levels), 2)
    row = np.arange(2 * levels)
    x = np.sin(group / 29.0) + 0.05 * (row // levels - 0.5)
    response = 1.0 + 0.4 * x + 0.2 * np.cos(group / 17.0)
    response += 0.01 * np.sin(row / 5.0)
    frame = pd.DataFrame(
        {
            "y": response,
            "x": x,
            "g": pd.Categorical(
                [f"g{index}" for index in group],
                categories=[f"g{index}" for index in range(levels)],
            ),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~x+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(
            batch_rows=256, memory_budget_bytes=24 << 20
        ),
    )
    assert model.n_coef == levels + 2
    scale = model.penalties.blocks[0].local_penalties[0].scale
    diagonal = np.bincount(group, minlength=levels) + 0.8 * scale
    cross = np.column_stack(
        (np.bincount(group, minlength=levels), np.bincount(group, weights=x))
    )
    modest = np.column_stack((np.ones(len(x)), x))
    grouped_response = np.bincount(group, weights=response, minlength=levels)
    schur = modest.T @ modest - cross.T @ (cross / diagonal[:, None])
    rhs = modest.T @ response - cross.T @ (grouped_response / diagonal)
    modest_coefficients = np.linalg.solve(schur, rhs)
    indexed_coefficients = (grouped_response - cross @ modest_coefficients) / diagonal
    expected = np.concatenate((modest_coefficients, indexed_coefficients))
    monkeypatch.setattr(
        PenaltyStructure,
        "materialize",
        lambda *_a, **_k: pytest.fail("Global coefficient-square penalty built"),
    )
    fitted = GAM(
        "y~x+s(g,bs='re')",
        family="gaussian",
        sp=[0.8],
        control=FitControl(
            execution="operator",
            linear_solver="pcg",
            batch_rows=256,
            memory_budget_bytes=24 << 20,
            operator=OperatorControl(tolerance=1e-10, cg_rtol=1e-12),
        ),
    ).fit(source, result="prediction")
    assert fitted.converged
    assert fitted.operator_diagnostics is not None
    assert fitted.operator_diagnostics.completed_solves > 0
    assert fitted.operator_diagnostics.krylov_iterations < 1000
    assert fitted.operator_diagnostics.peak_bound_bytes < 24 << 20
    np.testing.assert_allclose(
        fitted.coefficients, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        fitted.predict(frame.iloc[:64]),
        (modest[:64] @ modest_coefficients) + indexed_coefficients[group[:64]],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize("link", [LogLink(), InverseLink()])
def test_global_gaussian_start_and_empty_batches_are_batch_invariant(link):
    y = np.array([-2.0, 0.0, 1.0, 4.0, 0.0, 3.0, 7.0, 2.0, 1.0])
    weights = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    frame = pd.DataFrame({"y": y, "g": pd.Categorical(np.resize(["a", "b"], len(y)))})
    family = Gaussian(link=link)
    expected_sd = np.std(y, ddof=1)
    expected_mu = (
        np.maximum(y, 0.01 * expected_sd)
        if isinstance(link, LogLink)
        else y + (y == 0) * 0.01 * expected_sd
    )
    collector = _AssertCollector()
    for B in (1, 4, 7):
        base = DataFrameRowSource(frame, response="y", weights=weights)
        source = _EmptyBatches(base)
        model = prepare_operator_model(
            parse_formula("y~s(g,bs='re')"),
            source,
            control=OperatorPreparationControl(batch_rows=B),
        )
        beta, summary, _, _ = _initial_null_beta(
            model,
            source,
            family,
            reserved=model.metadata_bound_bytes,
            budget=8 * 1024 * 1024,
        )
        collector.check(
            f"B{B}/global SD",
            lambda summary=summary: np.testing.assert_allclose(
                summary["response_sd"], expected_sd, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        assert np.all(np.isfinite(beta))
        state = FrozenOperatorWorkingState(
            model,
            family,
            None,
            8 * 1024 * 1024,
            8 * 1024 * 1024,
            summary,
        )
        seen = 0
        for batch in source.scan(B):
            rows = _evaluate_geometry(model, batch)
            _, quantities = state.evaluate(batch, rows)
            n = len(batch.row_positions)
            collector.check(
                f"B{B}/rows {seen}",
                lambda quantities=quantities, n=n, seen=seen: (
                    np.testing.assert_allclose(
                        np.asarray(quantities.mu)[:n],
                        expected_mu[seen : seen + n],
                        rtol=STRICT.rtol,
                        atol=STRICT.atol,
                    )
                ),
            )
            assert bool(quantities.input_ok)
            assert bool(quantities.domain_ok)
            seen += n
        assert seen == len(y)
    collector.raise_if_any("Gaussian global starts")


def test_gaussian_operator_fit_matches_weighted_dense_system_across_batches():
    n = 53
    x = np.linspace(-1, 1, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "u": np.cos(x),
            "y": np.sin(2 * x) + 0.2 * x,
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "h": pd.Categorical(np.resize(["v", "v", "u"], n)),
        }
    )
    weights = np.linspace(0.6, 1.4, n)
    weights[::7] = 0
    offset = 0.1 * x
    spec = parse_formula("y~s(x,bs='cr',k=6)+s(g,h,bs='re')+s(u,g,bs='re')")
    dense = ModelSetup.build(spec, frame)
    sp = np.array([0.3, 0.7, 1.2])
    H = dense.X.T @ (weights[:, None] * dense.X) + dense.penalties.materialize(
        np.log(sp)
    )
    expected = np.linalg.solve(H, dense.X.T @ (weights * (frame.y.to_numpy() - offset)))
    expected_dev = np.sum(
        weights * (frame.y.to_numpy() - offset - dense.X @ expected) ** 2
    )
    collector = _AssertCollector()
    for B in (1, 13, 29):
        source = DataFrameRowSource(frame, response="y", weights=weights, offset=offset)
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        fitted = fit_operator_fixed_sp(
            model, source, Gaussian(), sp, memory_budget_bytes=8 * 1024 * 1024
        )
        assert fitted.converged
        assert fitted.n_iter == 1
        assert not fitted.coefficients.flags.writeable
        collector.check(
            f"B{B}/coefficients",
            lambda fitted=fitted: np.testing.assert_allclose(
                fitted.coefficients, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        collector.check(
            f"B{B}/deviance",
            lambda fitted=fitted: np.testing.assert_allclose(
                fitted.deviance, expected_dev, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        diagnostics = fitted.diagnostics
        mean = float(np.mean(frame.y))
        expected_null_scale = float(
            np.sum(weights * (frame.y.to_numpy() - mean) ** 2) / len(frame)
        )
        np.testing.assert_allclose(
            diagnostics.initial_null_scale,
            expected_null_scale,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        assert diagnostics.certificate_scans == 1
        assert diagnostics.completed_solves == 1
        assert diagnostics.source.completed_scans == (
            diagnostics.initialization_scans
            + diagnostics.rhs_scans
            + diagnostics.objective_scans
            + diagnostics.certificate_scans
            + diagnostics.krylov_actions
        )
        assert diagnostics.peak_bound_bytes <= 8 * 1024 * 1024
    collector.raise_if_any("Operator Gaussian fit")


def test_no_intercept_modest_null_projection_and_fit_match_dense_reference():
    n = 53
    x = np.linspace(0.2, 1.6, n)
    z = np.sin(2.1 * x)
    frame = pd.DataFrame({"x": x, "z": z, "y": 1.2 + 0.4 * x + 0.1 * z})
    spec = parse_formula("y~0+x+s(z,bs='cr',k=6)")
    dense = ModelSetup.build(spec, frame)
    scalar = float(frame.y.mean())
    expected_null = np.linalg.lstsq(dense.X, np.full(n, scalar), rcond=None)[0]
    sp = np.array([0.7])
    H = dense.X.T @ dense.X + dense.penalties.materialize(np.log(sp))
    expected_fit = np.linalg.solve(H, dense.X.T @ frame.y.to_numpy())
    collector = _AssertCollector()
    for B in (1, 7, 19):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        null_beta, _, _, _ = _initial_null_beta(
            model,
            source,
            Gaussian(),
            reserved=model.metadata_bound_bytes,
            budget=8 * 1024 * 1024,
        )
        collector.check(
            f"B{B}/null coefficients",
            lambda actual=null_beta: np.testing.assert_allclose(
                actual, expected_null, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        fitted = fit_operator_fixed_sp(
            model, source, Gaussian(), sp, memory_budget_bytes=8 * 1024 * 1024
        )
        assert fitted.converged
        assert fitted.diagnostics.initialization_scans == 3  # QR plus null scale
        collector.check(
            f"B{B}/fit coefficients",
            lambda actual=fitted.coefficients: np.testing.assert_allclose(
                actual, expected_fit, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("No-intercept modest source projection")


def test_no_intercept_numeric_and_indexed_null_anchor_matches_pinned_fit(r_bridge):
    n = 47
    x = np.linspace(0.2, 1.8, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": 1.4 + 0.2 * x + 0.04 * np.sin(3 * x),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
        }
    )
    formula = "y~0+x+s(g,bs='re')"
    spec = parse_formula(formula)
    dense = ModelSetup.build(spec, frame)
    sp = np.array([0.8])
    reference = r_bridge.fit_fixed_sp(formula, frame, "gaussian", sp)
    collector = _AssertCollector()
    for B in (1, 7, 17):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        null_beta, _, mean, _ = _initial_null_beta(
            model,
            source,
            Gaussian(),
            reserved=model.metadata_bound_bytes,
            budget=8 * 1024 * 1024,
        )
        expected_null = np.linalg.lstsq(dense.X, np.full(n, mean), rcond=None)[0]
        collector.check(
            f"B{B}/source null coefficients",
            lambda actual=null_beta, expected=expected_null: np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        actual_penalty = null_beta @ model.penalties.materialize(np.log(sp)) @ null_beta
        expected_penalty = (
            expected_null @ dense.penalties.materialize(np.log(sp)) @ expected_null
        )
        collector.check(
            f"B{B}/null anchor penalty",
            lambda actual=actual_penalty, expected=expected_penalty: (
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                )
            ),
        )
        fitted = fit_operator_fixed_sp(
            model, source, Gaussian(), sp, memory_budget_bytes=8 * 1024 * 1024
        )
        assert fitted.converged
        collector.check(
            f"B{B}/pinned coefficients",
            lambda actual=fitted.coefficients: np.testing.assert_allclose(
                actual,
                reference["coefficients"],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
    collector.raise_if_any("No-intercept mixed source projection")


@pytest.mark.parametrize(
    "formula",
    [
        "y~0+x+s(g,bs='re')",
        "y~0+x+z+s(g,bs='re')",
        "y~0+x+s(g,bs='re')+s(u,bs='cr',k=6)",
    ],
)
def test_rank_deficient_mixed_no_intercept_null_uses_source_order(r_bridge, formula):
    groups = np.resize(["a", "b", "c"], 39)
    frame = pd.DataFrame(
        {
            "x": np.array([{"a": 1.0, "b": 2.0, "c": 3.0}[g] for g in groups]),
            "z": np.array([{"a": 1.0, "b": 4.0, "c": 9.0}[g] for g in groups]),
            "u": np.linspace(0.1, 1.8, 39),
            "g": pd.Categorical(groups),
            "y": np.linspace(1.0, 2.0, 39),
        }
    )
    spec = parse_formula(formula)
    dense = ModelSetup.build(spec, frame)
    expected = r_bridge.qr_constant_coefficients(dense.X, 1.5)
    sp = np.array([0.8, 1.1]) if "s(u" in formula else np.array([0.8])
    reference = r_bridge.fit_fixed_sp(formula, frame, "gaussian", sp)
    collector = _AssertCollector()
    for B in (1, 7, 17):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        actual, _, _, _ = _initial_null_beta(
            model,
            source,
            Gaussian(),
            reserved=model.metadata_bound_bytes,
            budget=8 * 1024 * 1024,
        )
        collector.check(
            f"B{B}/rank-deficient mixed source coefficients",
            lambda actual=actual: np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        fitted = fit_operator_fixed_sp(
            model, source, Gaussian(), sp, memory_budget_bytes=8 * 1024 * 1024
        )
        collector.check(
            f"B{B}/rank-deficient mixed pinned fit",
            lambda actual=fitted.coefficients: np.testing.assert_allclose(
                actual,
                reference["coefficients"],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
    collector.raise_if_any("Rank-deficient mixed source QR")


def test_mixed_no_intercept_constant_modest_column_precedes_indexed_qr(r_bridge):
    n = 41
    frame = pd.DataFrame(
        {
            "z": np.ones(n),
            "x": np.linspace(0.2, 1.8, n),
            "y": np.linspace(1.0, 2.0, n),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "h": pd.Categorical(np.resize(["u", "v", "w", "v"], n)),
        }
    )
    spec = parse_formula("y~0+z+s(g,bs='re')+s(h,x,bs='re')")
    dense = ModelSetup.build(spec, frame)
    expected = r_bridge.qr_constant_coefficients(dense.X, 1.5)
    collector = _AssertCollector()
    for B in (1, 7, 17):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        actual, _, _, _ = _initial_null_beta(
            model,
            source,
            Gaussian(),
            reserved=model.metadata_bound_bytes,
            budget=8 * 1024 * 1024,
        )
        collector.check(
            f"B{B}/rank-deficient mixed source coordinates",
            lambda actual=actual: np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("Constant modest mixed null QR")


@pytest.mark.parametrize(
    "formula",
    [
        "y~0+x+s(g,bs='re')+s(h,bs='re')",
        "y~0+s(g,bs='re')+s(h,x,bs='re')",
        "y~0+x+s(g,bs='re')+s(h,x,bs='re')",
        "y~0+s(g,bs='re')+s(x,bs='cr',k=6)",
    ],
)
def test_crossed_indicator_no_intercept_anchor_matches_pinned_qr(r_bridge, formula):
    n = 41
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.2, 1.8, n),
            "y": np.linspace(1.0, 2.0, n),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "h": pd.Categorical(np.resize(["u", "v", "w", "v"], n)),
        }
    )
    spec = parse_formula(formula)
    dense = ModelSetup.build(spec, frame)
    expected = r_bridge.qr_constant_coefficients(dense.X, 1.5)
    collector = _AssertCollector()
    for B in (1, 7, 17):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        actual, _, _, _ = _initial_null_beta(
            model,
            source,
            Gaussian(),
            reserved=model.metadata_bound_bytes,
            budget=8 * 1024 * 1024,
        )
        collector.check(
            f"B{B}/source QR coefficients",
            lambda actual=actual: np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("Crossed no-intercept source QR")


@pytest.mark.parametrize(
    "formula",
    [
        "y~0+s(g,x,bs='re')",
        "y~0+z+s(g,x,bs='re')",
        "y~0+x+s(g,x,bs='re')",
        "y~0+x+z+s(g,x,bs='re')",
        "y~0+s(g,u,bs='re')+s(h,v,bs='re')",
    ],
)
def test_numeric_indexed_no_intercept_anchor_matches_pinned_qr(r_bridge, formula):
    n = 41
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.2, 1.8, n),
            "z": np.sin(np.linspace(0.2, 1.8, n)),
            "u": np.sin(0.33 * np.arange(n)),
            "v": np.cos(0.71 * np.arange(n)),
            "y": np.linspace(1.0, 2.0, n),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "h": pd.Categorical(np.resize(["u", "v", "w", "t"], n)),
        }
    )
    if formula == "y~0+x+z+s(g,x,bs='re')":
        frame["z"] = frame.x.to_numpy() * np.resize([1.0, 2.0, 3.0], n)
    spec = parse_formula(formula)
    dense = ModelSetup.build(spec, frame)
    expected = r_bridge.qr_constant_coefficients(dense.X, 1.5)
    crossed = "s(h,v,bs='re')" in formula
    rank_mixed_numeric = formula in {
        "y~0+x+s(g,x,bs='re')",
        "y~0+x+z+s(g,x,bs='re')",
    }
    sp = np.array([0.8, 1.1]) if crossed else np.array([0.8])
    reference = (
        r_bridge.fit_fixed_sp(formula, frame, "gaussian", sp)
        if crossed or rank_mixed_numeric
        else None
    )
    collector = _AssertCollector()
    for B in (1, 7, 17):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        actual, _, _, _ = _initial_null_beta(
            model,
            source,
            Gaussian(),
            reserved=model.metadata_bound_bytes,
            budget=8 * 1024 * 1024,
        )
        collector.check(
            f"B{B}/source QR coefficients",
            lambda actual=actual: np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        if crossed or rank_mixed_numeric:
            fitted = fit_operator_fixed_sp(
                model, source, Gaussian(), sp, memory_budget_bytes=8 * 1024 * 1024
            )
            diagnostics = fitted.diagnostics
            if crossed:
                assert diagnostics.initialization_scans > 3
            assert diagnostics.source.completed_scans == (
                diagnostics.initialization_scans
                + diagnostics.rhs_scans
                + diagnostics.objective_scans
                + diagnostics.certificate_scans
                + diagnostics.krylov_actions
            )
            collector.check(
                f"B{B}/numeric indexed fixed-sp coefficients",
                lambda actual=fitted.coefficients: np.testing.assert_allclose(
                    actual,
                    reference["coefficients"],
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
    collector.raise_if_any("Numeric indexed no-intercept source QR")


def test_crossed_numeric_large_levels_fit_without_global_coefficient_square(
    monkeypatch,
):
    n = 97
    row = np.arange(n)
    group = row % 3
    crossed = row % 4
    first_amplitude = np.sin(0.33 * row)
    second_amplitude = np.cos(0.71 * row)
    frame = pd.DataFrame(
        {
            "y": 1.0 + row / n,
            "x": first_amplitude,
            "z": second_amplitude,
            "g": pd.Categorical(
                [f"g{i}" for i in group], categories=[f"g{i}" for i in range(1024)]
            ),
            "h": pd.Categorical(
                [f"h{i}" for i in crossed],
                categories=[f"h{i}" for i in range(1024)],
            ),
        }
    )
    monkeypatch.setattr(
        PenaltyStructure,
        "materialize",
        lambda *_a, **_k: pytest.fail("Global coefficient-square penalty built"),
    )
    source = DataFrameRowSource(frame, response="y")
    spec = parse_formula("y~0+s(g,x,bs='re')+s(h,z,bs='re')")
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=13)
    )
    assert model.n_coef == 2048
    budget = 8 * 1024 * 1024
    assert budget < 8 * model.n_coef**2
    fitted = fit_operator_fixed_sp(
        model,
        source,
        Gaussian(),
        np.array([0.8, 1.1]),
        memory_budget_bytes=budget,
    )
    assert fitted.converged
    assert fitted.diagnostics.peak_bound_bytes <= budget
    active = np.zeros((n, 7))
    active[row, group] = first_amplitude
    active[row, 3 + crossed] = second_amplitude
    penalty = np.diag(
        [0.8 * model.random_effects[0].penalty_scale] * 3
        + [1.1 * model.random_effects[1].penalty_scale] * 4
    )
    expected = np.linalg.solve(active.T @ active + penalty, active.T @ frame.y)
    actual = fitted.coefficients[np.r_[0:3, 1024:1028]]
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    assert np.count_nonzero(fitted.coefficients) == 7


def test_mixed_numeric_large_levels_fit_without_global_coefficient_square(
    monkeypatch,
):
    n = 97
    row = np.arange(n)
    group = row % 3
    crossed = row % 4
    modest = np.sin(0.11 * row) + 0.2 * np.cos(0.37 * row)
    first_amplitude = np.sin(0.33 * row)
    second_amplitude = np.cos(0.71 * row)
    frame = pd.DataFrame(
        {
            "y": 1.0 + row / n,
            "x": modest,
            "u": first_amplitude,
            "v": second_amplitude,
            "g": pd.Categorical(
                [f"g{i}" for i in group], categories=[f"g{i}" for i in range(1024)]
            ),
            "h": pd.Categorical(
                [f"h{i}" for i in crossed],
                categories=[f"h{i}" for i in range(1024)],
            ),
        }
    )
    monkeypatch.setattr(
        PenaltyStructure,
        "materialize",
        lambda *_a, **_k: pytest.fail("Global coefficient-square penalty built"),
    )
    source = DataFrameRowSource(frame, response="y")
    spec = parse_formula("y~0+x+s(g,u,bs='re')+s(h,v,bs='re')")
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=13)
    )
    assert model.n_coef == 2049
    budget = 8 * 1024 * 1024
    assert budget < 8 * model.n_coef**2
    fitted = fit_operator_fixed_sp(
        model,
        source,
        Gaussian(),
        np.array([0.8, 1.1]),
        memory_budget_bytes=budget,
    )
    assert fitted.converged
    assert fitted.diagnostics.peak_bound_bytes <= budget
    active = np.zeros((n, 8))
    active[:, 0] = modest
    active[row, 1 + group] = first_amplitude
    active[row, 4 + crossed] = second_amplitude
    penalty = np.diag(
        [0.0]
        + [0.8 * model.random_effects[0].penalty_scale] * 3
        + [1.1 * model.random_effects[1].penalty_scale] * 4
    )
    expected = np.linalg.solve(active.T @ active + penalty, active.T @ frame.y)
    actual = fitted.coefficients[np.r_[0:4, 1025:1029]]
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
    assert np.count_nonzero(fitted.coefficients) == 8


def test_three_numeric_indexed_source_null_coordinates_and_fit_match_r(r_bridge):
    combinations = [(g, h, q) for g in range(3) for h in range(3) for q in range(3)]
    rows = [(g, h, q, repeat) for g, h, q in combinations for repeat in range(3)]
    frame = pd.DataFrame(
        {
            "y": 1.2 + 0.15 * np.sin(np.arange(len(rows))),
            "g": pd.Categorical([f"g{g}" for g, _, _, _ in rows]),
            "h": pd.Categorical([f"h{h}" for _, h, _, _ in rows]),
            "q": pd.Categorical([f"q{q}" for _, _, q, _ in rows]),
            "u": [1.0 if repeat == 0 else 0.02 for _, _, _, repeat in rows],
            "v": [1.0 if repeat == 1 else 0.02 for _, _, _, repeat in rows],
            "w": [1.0 if repeat == 2 else 0.02 for _, _, _, repeat in rows],
        }
    )
    formula = "y~0+s(g,u,bs='re')+s(h,v,bs='re')+s(q,w,bs='re')"
    spec = parse_formula(formula)
    dense = ModelSetup.build(spec, frame)
    expected = r_bridge.qr_constant_coefficients(dense.X, 1.0)
    reference = r_bridge.fit_fixed_sp(
        formula, frame, "gaussian", np.array([0.8, 1.1, 1.4])
    )
    collector = _AssertCollector()
    for B in (1, 7, 19):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        actual, _, _, _ = _initial_null_beta(
            model,
            source,
            Gaussian(),
            reserved=model.metadata_bound_bytes,
            budget=8 * 1024 * 1024,
        )
        collector.check(
            f"B{B}/source QR coordinates",
            lambda actual=actual: np.testing.assert_allclose(
                actual / np.mean(frame.y),
                expected,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        fitted = fit_operator_fixed_sp(
            model,
            source,
            Gaussian(),
            np.array([0.8, 1.1, 1.4]),
            memory_budget_bytes=8 * 1024 * 1024,
        )
        collector.check(
            f"B{B}/fixed-sp coefficients",
            lambda fitted=fitted: np.testing.assert_allclose(
                fitted.coefficients,
                reference["coefficients"],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        assert fitted.diagnostics.initialization_scans > 3
    collector.raise_if_any("Three numeric indexed source coordinates")


@pytest.mark.parametrize(
    "formula",
    [
        "y~0+s(g,u,bs='re')+s(x,bs='cr',k=6)",
        "y~0+x+s(g,u,bs='re')+s(h,v,bs='re')",
    ],
)
def test_full_rank_mixed_source_null_and_fixed_sp_match_pinned_r(r_bridge, formula):
    n = 63
    row = np.arange(n)
    frame = pd.DataFrame(
        {
            "y": 1.1 + 0.1 * np.sin(row),
            "x": np.linspace(0.2, 1.4, n),
            "u": np.sin(0.41 * row),
            "v": np.cos(0.27 * row),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "h": pd.Categorical(np.resize(["d", "e", "f", "j"], n)),
        }
    )
    spec = parse_formula(formula)
    dense = ModelSetup.build(spec, frame)
    expected_null = r_bridge.qr_constant_coefficients(dense.X, np.mean(frame.y))
    sp = np.array([0.8, 1.1])
    reference = r_bridge.fit_fixed_sp(formula, frame, "gaussian", sp)
    collector = _AssertCollector()
    for B in (1, 7, 17):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        actual, _, _, _ = _initial_null_beta(
            model,
            source,
            Gaussian(),
            reserved=model.metadata_bound_bytes,
            budget=8 * 1024 * 1024,
        )
        collector.check(
            f"B{B}/natural source QR",
            lambda actual=actual: np.testing.assert_allclose(
                actual, expected_null, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        fitted = fit_operator_fixed_sp(
            model, source, Gaussian(), sp, memory_budget_bytes=8 * 1024 * 1024
        )
        collector.check(
            f"B{B}/fixed-sp coefficients",
            lambda fitted=fitted: np.testing.assert_allclose(
                fitted.coefficients,
                reference["coefficients"],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        collector.check(
            f"B{B}/deviance",
            lambda fitted=fitted: np.testing.assert_allclose(
                fitted.deviance,
                reference["deviance"],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        assert fitted.diagnostics.initialization_scans > 3
    collector.raise_if_any("Full-rank mixed source null and fixed sp")


def test_zero_constant_null_for_rank_deficient_crossed_numeric_matches_r(r_bridge):
    n = 40
    frame = pd.DataFrame(
        {
            "y": np.tile([1.0, -1.0], n // 2),
            "x": np.ones(n),
            "z": np.ones(n),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "h": pd.Categorical(np.resize(["u", "v", "w", "t"], n)),
        }
    )
    formula = "y~0+s(g,x,bs='re')+s(h,z,bs='re')"
    for B in (1, 7, 8, 19):
        source = DataFrameRowSource(frame, response="y")
        model = prepare_operator_model(
            parse_formula(formula),
            source,
            control=OperatorPreparationControl(batch_rows=B),
        )
        null_beta, _, mean, _ = _initial_null_beta(
            model,
            source,
            Gaussian(),
            reserved=model.metadata_bound_bytes,
            budget=8 * 1024 * 1024,
        )
        assert mean == 0.0
        np.testing.assert_array_equal(null_beta, np.zeros(model.n_coef))
    sp = np.array([0.8, 1.1])
    reference = r_bridge.fit_fixed_sp(formula, frame, "gaussian", sp)
    fitted = fit_operator_fixed_sp(
        model, source, Gaussian(), sp, memory_budget_bytes=8 * 1024 * 1024
    )
    np.testing.assert_allclose(
        fitted.coefficients,
        reference["coefficients"],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.parametrize(
    ("python_expression", "r_expression"),
    [
        ("s(x,bs='tp',k=6)", "s(x,bs='tp',k=6)"),
        ("s(x,bs='gp',k=6)", "s(x,bs='gp',k=6)"),
        ("te(x,z,bs='tp',k=4)", "te(x,z,bs='tp',k=c(4,4))"),
        ("ti(x,z,bs='tp',k=4)", "ti(x,z,bs='tp',k=c(4,4))"),
        ("ti(x,z,bs='tp',k=6)", "ti(x,z,bs='tp',k=c(6,6))"),
    ],
)
def test_radial_fixed_sp_gaussian_fit_matches_pinned_source_across_batches(
    r_bridge, python_expression, r_expression
):
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
    formula = "y~" + python_expression
    sp = np.full(
        2 if python_expression.startswith(("te(", "ti(")) else 1,
        0.7,
    )
    expected = r_bridge.fit_fixed_sp(
        "y~" + r_expression,
        frame,
        "gaussian",
        sp,
        weights="w",
        offset="o",
    )
    collector = _AssertCollector()
    for batch_rows in (1, 7, 19):
        source = DataFrameRowSource(
            frame,
            response="y",
            weights=frame.w.to_numpy(),
            offset=frame.o.to_numpy(),
        )
        model = prepare_operator_model(
            parse_formula(formula),
            source,
            control=OperatorPreparationControl(batch_rows=batch_rows),
        )
        fitted = fit_operator_fixed_sp(
            model,
            source,
            Gaussian(),
            sp,
            memory_budget_bytes=128 * 1024 * 1024,
        )
        rows = np.vstack(
            [
                model.evaluate_batch(batch).modest_values
                for batch in source.scan(batch_rows)
            ]
        )
        means = rows @ fitted.coefficients + frame.o.to_numpy()
        assert fitted.converged
        collector.check(
            f"B{batch_rows}/means",
            lambda means=means: np.testing.assert_allclose(
                means,
                expected["fitted_values"],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        collector.check(
            f"B{batch_rows}/deviance",
            lambda fitted=fitted: np.testing.assert_allclose(
                fitted.deviance,
                expected["deviance"],
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
        assert fitted.diagnostics.peak_bound_bytes <= 128 * 1024 * 1024
    collector.raise_if_any(python_expression)


@pytest.mark.parametrize(
    ("family", "key"),
    [
        (Gaussian(), "gaussian"),
        (Gaussian("log"), "gaussian_log"),
        (Gaussian("inverse"), "gaussian_inverse"),
        (Poisson(), "poisson"),
        (Poisson("identity"), "poisson_identity"),
        (Poisson("sqrt"), "poisson_sqrt"),
        (Binomial(), "binomial"),
        (Binomial("probit"), "binomial_probit"),
        (Binomial("cloglog"), "binomial_cloglog"),
        (Binomial("log"), "binomial_log"),
        (Gamma(), "gamma"),
        (Gamma("log"), "gamma_log"),
        (Gamma("identity"), "gamma_identity"),
        (NegativeBinomial(3.0, fixed=True), "nb"),
        (NegativeBinomial(3.0, fixed=True, link="identity"), "nb_identity"),
        (NegativeBinomial(3.0, fixed=True, link="sqrt"), "nb_sqrt"),
    ],
)
def test_operator_observed_fixed_sp_fit_matches_pinned_family_reference(
    r_bridge, family, key
):
    n = 67
    x = np.linspace(-1, 1, n)
    y = (
        np.resize([0.0, 1.0, 1.0, 0.0, 1.0], n)
        if key.startswith("binomial")
        else np.resize([1.0, 2.0, 3.0, 2.0, 1.0], n)
        if key.startswith(("poisson", "nb"))
        else 1.0 + 0.1 * np.sin(2 * x) + 0.02 * np.cos(3 * x)
    )
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "h": pd.Categorical(np.resize(["u", "u", "v", "v"], n)),
            "w": np.linspace(0.6, 1.4, n),
            "o": 0.01 * x,
        }
    )
    frame.loc[::7, "w"] = 0.0
    if key == "binomial_log":
        # Bernoulli 0/1 data drive the pinned noncanonical log fit to its
        # boundary without a converged oracle. Four-trial proportions keep
        # the same supported link inside its valid mean domain.
        frame["y"] = np.resize([0.25, 0.5, 0.75, 0.5], n)
        frame["w"] = 4.0
    formula = "y~s(x,bs='cr',k=6)+s(g,bs='re')+s(h,bs='re')"
    spec = parse_formula(formula)
    sp = np.array([1.2, 0.8, 0.6])
    reference = r_bridge.fit_fixed_sp(
        formula,
        frame,
        key,
        sp,
        weights="w",
        offset="o",
        theta=3.0 if key.startswith("nb") else None,
    )
    assert reference["converged"]
    dense = ModelSetup.build(spec, frame)
    collector = _AssertCollector()
    for B in (7, 19):
        source = DataFrameRowSource(
            frame, response="y", weights=frame.w, offset=frame.o
        )
        model = prepare_operator_model(
            spec, source, control=OperatorPreparationControl(batch_rows=B)
        )
        fitted = fit_operator_fixed_sp(
            model,
            source,
            family,
            sp,
            control=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-14),
            memory_budget_bytes=8 * 1024 * 1024,
        )
        assert fitted.converged
        for name, actual, expected in (
            ("coefficients", fitted.coefficients, reference["coefficients"]),
            ("deviance", fitted.deviance, reference["deviance"]),
            (
                "means",
                family.link.inverse(dense.X @ fitted.coefficients + frame.o.to_numpy()),
                reference["fitted_values"],
            ),
        ):
            collector.check(
                f"{key}/B{B}/{name}",
                lambda actual=actual, expected=expected: np.testing.assert_allclose(
                    actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
    collector.raise_if_any("Operator pinned observed family fits")


def test_binomial_log_bernoulli_boundary_oracle_is_nonconvergent(r_bridge):
    n = 67
    x = np.linspace(-1, 1, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": np.resize([0.0, 1.0, 1.0, 0.0, 1.0], n),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "h": pd.Categorical(np.resize(["u", "u", "v", "v"], n)),
            "w": np.linspace(0.6, 1.4, n),
            "o": 0.01 * x,
        }
    )
    frame.loc[::7, "w"] = 0.0
    reference = r_bridge.fit_fixed_sp(
        "y~s(x,bs='cr',k=6)+s(g,bs='re')+s(h,bs='re')",
        frame,
        "binomial_log",
        np.array([1.2, 0.8, 0.6]),
        weights="w",
        offset="o",
    )
    assert not reference["converged"]
    assert np.isfinite(reference["deviance"])


def test_negative_nb_identity_rows_fit_with_total_observed_spd(r_bridge):
    n = 31
    frame = pd.DataFrame(
        {
            "y": np.full(n, 2.0),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
        }
    )
    frame.loc[0, "y"] = 0.0
    formula = "y~s(g,bs='re')"
    reference = r_bridge.fit_fixed_sp(
        formula, frame, "nb_identity", np.array([1.0]), theta=3.0
    )
    assert reference["converged"]
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula(formula),
        source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    family = NegativeBinomial(3.0, fixed=True, link="identity")
    _, summary, _, _ = _initial_null_beta(
        model,
        source,
        family,
        reserved=model.metadata_bound_bytes,
        budget=8 * 1024 * 1024,
    )
    initial = FrozenOperatorWorkingState(
        model, family, None, 8 * 1024 * 1024, 8 * 1024 * 1024, summary
    )
    observed = np.concatenate(
        [
            initial.curvature(batch, _evaluate_geometry(model, batch))
            for batch in source.scan(model.batch_rows)
        ]
    )
    assert observed[0] < 0
    dense = ModelSetup.build(parse_formula(formula), frame)
    H = dense.X.T @ (observed[:, None] * dense.X)
    H += dense.penalties.materialize(np.zeros(1))
    assert np.min(np.linalg.eigvalsh(H)) > 0
    fitted = fit_operator_fixed_sp(
        model,
        source,
        family,
        np.array([1.0]),
        memory_budget_bytes=8 * 1024 * 1024,
    )
    assert fitted.converged
    collector = _AssertCollector()
    collector.check(
        "signed NB coefficients",
        lambda: np.testing.assert_allclose(
            fitted.coefficients,
            reference["coefficients"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "signed NB means",
        lambda: np.testing.assert_allclose(
            dense.X @ fitted.coefficients,
            reference["fitted_values"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "signed NB deviance",
        lambda: np.testing.assert_allclose(
            fitted.deviance,
            reference["deviance"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any("Signed NB identity total SPD")


def test_cyclic_fixed_sp_fit_and_prediction_match_pinned_source(r_bridge):
    from jaxgam import GAM, FitControl

    n = 67
    x = np.linspace(-1, 1, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": 1.0 + 0.1 * np.sin(np.pi * x) + 0.02 * np.cos(2 * np.pi * x),
            "g": pd.Categorical(np.resize(["a", "b", "c"], n)),
            "w": np.linspace(0.6, 1.4, n),
            "o": 0.01 * x,
        }
    )
    frame.loc[::7, "w"] = 0.0
    formula = "y~s(x,bs='cc',k=6)+s(g,bs='re')"
    sp = np.array([1.2, 0.8])
    reference = r_bridge.fit_fixed_sp(
        formula, frame, "gaussian", sp, weights="w", offset="o"
    )
    collector = _AssertCollector()
    for B in (1, 13, 29):
        result = GAM(
            formula,
            family=Gaussian(),
            sp=sp,
            control=FitControl(
                execution="operator",
                linear_solver="pcg",
                batch_rows=B,
                operator=OperatorControl(cg_rtol=1e-13, cg_atol=1e-14),
            ),
        ).fit(
            DataFrameRowSource(frame, response="y", weights=frame.w, offset=frame.o),
            result="prediction",
        )
        assert result.converged
        for name, actual, expected in (
            ("coefficients", result.coefficients, reference["coefficients"]),
            ("deviance", result.deviance, reference["deviance"]),
            (
                "means",
                result.predict(frame, offset=frame.o),
                reference["fitted_values"],
            ),
        ):
            collector.check(
                f"B{B}/{name}",
                lambda actual=actual, expected=expected: np.testing.assert_allclose(
                    actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
        heldout = frame.iloc[:5].copy()
        wrapped = heldout.copy()
        wrapped.x += 6.0
        collector.check(
            f"B{B}/cyclic held-out prediction",
            lambda result=result, heldout=heldout, wrapped=wrapped: (
                np.testing.assert_allclose(
                    result.predict(heldout, offset=np.zeros(5)),
                    result.predict(wrapped, offset=np.zeros(5)),
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                )
            ),
        )
    collector.raise_if_any("Cyclic compact fixed-sp fit")


def test_operator_pcg_failure_returns_last_finite_accepted_state_and_true_status():
    x = np.linspace(-1, 1, 31)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": np.sin(2 * x),
            "g": pd.Categorical(np.resize(["a", "b", "c"], len(x))),
        }
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~s(x,bs='cr',k=6)+s(g,bs='re')"),
        source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    result = fit_operator_fixed_sp(
        model,
        source,
        Gaussian(),
        np.array([0.1, 0.3]),
        control=OperatorControl(cg_max_iterations=1, cg_rtol=1e-14),
        memory_budget_bytes=8 * 1024 * 1024,
    )
    assert not result.converged
    assert "PCG failed" in result.convergence_info
    assert np.all(np.isfinite(result.coefficients))
    assert np.isfinite(result.deviance)
    assert result.diagnostics.completed_solves == 0
    assert result.diagnostics.inner_status != 1


@pytest.mark.parametrize("formula", ["y~0+s(g,bs='re')", "y~s(g,bs='re')"])
def test_provable_null_coordinate_layouts_match_complete_gaussian_fit(formula):
    frame = pd.DataFrame(
        {
            "y": np.resize([1.0, 2.0, 3.0], 31),
            "g": pd.Categorical(
                np.resize(["a", "b"], 31), categories=["a", "b", "unused"]
            ),
        }
    )
    spec = parse_formula(formula)
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        spec, source, control=OperatorPreparationControl(batch_rows=7)
    )
    dense = ModelSetup.build(spec, frame)
    penalty = dense.penalties.materialize(np.log(np.array([0.8])))
    expected = np.linalg.solve(
        dense.X.T @ dense.X + penalty, dense.X.T @ frame.y.to_numpy()
    )
    fitted = fit_operator_fixed_sp(
        model, source, Gaussian(), np.array([0.8]), memory_budget_bytes=8 * 1024 * 1024
    )
    assert fitted.converged
    np.testing.assert_allclose(
        fitted.coefficients, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_fit_combined_budget_rejects_before_initial_source_pass():
    frame = pd.DataFrame({"y": np.arange(13, dtype=float), "x": np.linspace(0, 1, 13)})
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~x"), source, control=OperatorPreparationControl(batch_rows=4)
    )

    class Untouched:
        n_rows = len(frame)

        def fingerprint(self):
            return source.fingerprint()

        def scan(self, _batch_rows):
            pytest.fail("Initialization ran above the combined budget.")

    with pytest.raises(MemoryError, match="Combined"):
        fit_operator_fixed_sp(
            model, Untouched(), Gaussian(), np.empty(0), memory_budget_bytes=1000
        )


def test_gaussian_near_perfect_fit_uses_direct_residual_deviance():
    x = np.linspace(-1, 1, 47)
    frame = pd.DataFrame({"x": x, "y": 1.0 + 2 * x + 1e-10 * np.cos(3 * x)})
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~x"), source, control=OperatorPreparationControl(batch_rows=7)
    )
    result = fit_operator_fixed_sp(
        model, source, Gaussian(), np.empty(0), memory_budget_bytes=8 * 1024 * 1024
    )
    residual = (
        frame.y.to_numpy() - np.column_stack((np.ones(len(x)), x)) @ result.coefficients
    )
    assert result.converged
    assert result.deviance > 0
    assert result.deviance < 1e-15
    np.testing.assert_allclose(
        result.deviance, np.sum(residual**2), rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_working_snapshot_owns_beta_summary_and_validates_cached_dispatch():
    frame = pd.DataFrame(
        {"y": np.arange(1, 14, dtype=float), "x": np.linspace(0, 1, 13)}
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~x"), source, control=OperatorPreparationControl(batch_rows=4)
    )
    family = Gaussian()
    beta = np.array([1.0, 2.0])
    summary = {"input_ok": True}
    state = FrozenOperatorWorkingState(
        model, family, beta, 1024 * 1024, 1024 * 1024, summary
    )
    beta[:] = -10
    summary["input_ok"] = False
    assert np.array_equal(state.beta, [1.0, 2.0])
    assert state.summary["input_ok"]
    assert not state.beta.flags.writeable
    batch = next(source.scan(4))
    rows = _evaluate_geometry(model, batch)
    assert bool(state.evaluate(batch, rows)[1].finite_system)
    state.summary["input_ok"] = False
    with pytest.raises(RuntimeError, match="summary changed"):
        state.evaluate(batch, rows)
    state.summary["input_ok"] = True
    family.link = LogLink()
    with pytest.raises(RuntimeError, match="configuration changed"):
        state.evaluate(batch, rows)
    with pytest.raises(MemoryError, match="ownership"):
        FrozenOperatorWorkingState(model, Gaussian(), np.ones(2), 1024, 1)
    with pytest.raises(ValueError, match="budgets"):
        FrozenOperatorWorkingState(model, Gaussian(), None, True, 1024)
    with pytest.raises(ValueError, match="float64"):
        FrozenOperatorWorkingState(model, Gaussian(), np.ones(3), 1024, 1024)
    with pytest.raises(ValueError, match="finite"):
        FrozenOperatorWorkingState(model, Gaussian(), np.full(2, np.nan), 1024, 1024)


@pytest.mark.parametrize("family", [Poisson(), Gaussian(link=LogLink())])
def test_nonconstant_outer_iteration_limit_keeps_finite_last_accepted_fit(family):
    frame = pd.DataFrame(
        {"y": np.resize([1.0, 2.0, 3.0], 31), "x": np.linspace(-1, 1, 31)}
    )
    source = DataFrameRowSource(frame, response="y")
    model = prepare_operator_model(
        parse_formula("y~x"), source, control=OperatorPreparationControl(batch_rows=7)
    )
    fitted = fit_operator_fixed_sp(
        model,
        source,
        family,
        np.empty(0),
        control=OperatorControl(max_iterations=1),
        memory_budget_bytes=8 * 1024 * 1024,
    )
    assert not fitted.converged
    assert fitted.n_iter == 1
    assert "Maximum" in fitted.convergence_info
    assert np.all(np.isfinite(fitted.coefficients))
    assert np.isfinite(fitted.deviance)


def _exact_penalized_wls(X, S, weights, response):
    """Exact rational solution of (X'WX + S) beta = X'W r for float64 inputs."""
    from fractions import Fraction

    p = X.shape[1]
    system = [[Fraction(float(S[a, b])) for b in range(p)] for a in range(p)]
    rhs = [Fraction(0)] * p
    for row, weight, value in zip(X, weights, response, strict=True):
        weight = Fraction(float(weight))
        if weight == 0:
            continue
        row = [Fraction(float(entry)) for entry in row]
        value = Fraction(float(value))
        for a in range(p):
            rhs[a] += weight * row[a] * value
            for b in range(p):
                system[a][b] += weight * row[a] * row[b]
    augmented = [system[a] + [rhs[a]] for a in range(p)]
    for column in range(p):
        pivot = max(range(column, p), key=lambda k: abs(augmented[k][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for k in range(p):
            if k != column and augmented[k][column] != 0:
                factor = augmented[k][column] / augmented[column][column]
                augmented[k] = [
                    augmented[k][j] - factor * augmented[column][j]
                    for j in range(p + 1)
                ]
    return np.array([float(augmented[a][p] / augmented[a][a]) for a in range(p)])


def test_strictly_additive_solve_is_refined_to_the_exact_solution():
    """The single Gaussian identity solve is refined on its true residual.

    With one-row batches the frozen normal-form reductions alone left the
    PCG solution 2.3e-12 from the exact solution (cond 3.3e4, 1.25x STRICT).
    Residual refinement reaches it at STRICT for every batch size.
    """
    x = np.linspace(-1, 1, 53)
    frame = pd.DataFrame(
        {
            "x": x,
            "z": 1.2 + x**2,
            "g": pd.Categorical(np.resize(["g1", "g2"], len(x))),
            "y": 1.0 + 0.2 * np.sin(x),
            "w": np.linspace(0.6, 1.4, len(x)),
            "o": 0.01 * x,
        }
    )
    frame.loc[::7, "w"] = 0.0
    formula = "y~s(g,bs='re')+s(x,bs='cr',k=5,by=z)"
    sp = np.array([0.8, 1.2])
    setup = ModelSetup.build(parse_formula(formula), frame)
    exact = _exact_penalized_wls(
        np.asarray(setup.X),
        np.asarray(setup.penalties.materialize(np.log(sp))),
        frame.w.to_numpy(),
        (frame.y - frame.o).to_numpy(),
    )
    collector = _AssertCollector()
    for B in (1, 7, 19):
        source = DataFrameRowSource(
            frame, response="y", weights=frame.w, offset=frame.o
        )
        model = prepare_operator_model(
            parse_formula(formula),
            source,
            control=OperatorPreparationControl(batch_rows=B),
        )
        fitted = fit_operator_fixed_sp(
            model,
            source,
            Gaussian(),
            sp,
            control=OperatorControl(cg_rtol=1e-13, cg_atol=1e-14),
            memory_budget_bytes=64 * 1024 * 1024,
        )
        assert fitted.converged
        # One RHS, at least one residual refinement and the final gradient.
        assert fitted.diagnostics.rhs_scans >= 3
        assert fitted.diagnostics.completed_solves == 1
        collector.check(
            f"B{B}",
            lambda fitted=fitted: np.testing.assert_allclose(
                fitted.coefficients, exact, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("refined strictly additive operator solve")
