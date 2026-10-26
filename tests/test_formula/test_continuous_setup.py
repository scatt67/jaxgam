"""Compact basis setup and combined-memory contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import jaxgam.formula.continuous_setup as continuous_setup
from jaxgam.data.continuous_frame import CompactSetupFrame, build_compact_setup_frame
from jaxgam.data.discretize import DiscretizationBudget
from jaxgam.data.source import ArrayRowSource
from jaxgam.formula.continuous_setup import (
    ContinuousDiscreteSetup,
    ContinuousSetupUnavailable,
    build_continuous_discrete_setup,
)
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.predict_matrix import build_predict_spec


def _frame(n_rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": np.sin(np.linspace(-1.0, 2.0, n_rows)),
            "x": np.resize(np.array((0.0, 0.25, 0.5, 0.75, 1.0)), n_rows),
            "z": np.linspace(-2.0, 2.0, n_rows),
            "w": np.resize(np.array((-1.0, 0.0, 1.0)), n_rows),
            "b": np.resize(np.array((0.5, 1.5)), n_rows),
            "g": pd.Series(
                pd.Categorical(
                    np.resize(np.array(("a", "b", "c")), n_rows),
                    categories=("a", "b", "c"),
                    ordered=True,
                )
            ),
        }
    )


def _source(frame: pd.DataFrame) -> ArrayRowSource:
    return ArrayRowSource({name: frame[name] for name in frame})


def _build(formula: str, frame: pd.DataFrame, **kwargs) -> ContinuousDiscreteSetup:
    result = build_continuous_discrete_setup(
        parse_formula(formula), _source(frame), batch_rows=7, **kwargs
    )
    assert isinstance(result, ContinuousDiscreteSetup)
    return result


def test_setup_runs_dense_basis_only_on_padded_compact_frame(monkeypatch) -> None:
    frame = _frame()
    formula = parse_formula("y ~ s(x, bs='cr', k=4) + z")
    seen = []
    original = ModelSetup.build

    def spy(spec, data, weights=None, offset=None):
        seen.append(data)
        return original(spec, data, weights=weights, offset=offset)

    monkeypatch.setattr(continuous_setup.ModelSetup, "build", staticmethod(spy))
    result = build_continuous_discrete_setup(formula, _source(frame), m=6, batch_rows=7)
    assert isinstance(result, ContinuousDiscreteSetup)
    assert seen == [result.compact.frame]
    assert len(seen[0]) == 6
    assert len(seen[0]) < len(frame)
    assert result.n_obs == len(frame)


def test_frozen_basis_and_penalties_equal_direct_compact_setup() -> None:
    frame = _frame()
    formula = parse_formula("y ~ s(x, bs='cr', k=4) + z")
    result = _build("y ~ s(x, bs='cr', k=4) + z", frame, m=6)
    direct = ModelSetup.build(formula, result.compact.frame)
    direct_spec = build_predict_spec(direct)
    np.testing.assert_allclose(
        result.predict_spec.build_predict_matrix(result.compact.frame),
        direct_spec.build_predict_matrix(result.compact.frame),
        rtol=0,
        atol=0,
    )
    assert result.penalties is not None
    assert direct.penalties is not None
    np.testing.assert_allclose(
        result.penalties.materialize(), direct.penalties.materialize(), rtol=0, atol=0
    )
    assert result.basis_fingerprint


def test_factor_tensor_and_numeric_by_freeze_full_prediction_domain() -> None:
    frame = _frame()
    formula = "y ~ te(x, w, bs='cr', k=3, by=g) + s(z, bs='cr', k=4, by=b) + g"
    result = _build(formula, frame, factor_names=("g",), m=(5, 6))
    direct = ModelSetup.build(parse_formula(formula), result.compact.frame)
    expected = build_predict_spec(direct)
    probe = frame.iloc[[11, 2, 40, 1]].reset_index(drop=True)
    np.testing.assert_allclose(
        result.predict_spec.build_predict_matrix(probe),
        expected.build_predict_matrix(probe),
        rtol=0,
        atol=0,
    )
    assert result.predict_spec.factor_info == {"g": ["a", "b", "c"]}
    assert result.predict_spec.ordered_factors == frozenset(("g",))


@pytest.mark.parametrize(
    ("formula", "factors"),
    [
        ("y ~ s(x, bs='tp', k=4)", ()),
        ("y ~ s(x, bs='ts', k=4)", ()),
        ("y ~ s(x, bs='gp', k=4)", ()),
        ("y ~ s(x, bs='cs', k=4)", ()),
        ("y ~ s(x, bs='cc', k=4)", ()),
        ("y ~ ti(x, w, bs='cr', k=3)", ()),
        ("y ~ s(g, bs='re')", ("g",)),
    ],
)
def test_supported_smooth_domain_freezes_compact_metadata(formula, factors) -> None:
    result = _build(formula, _frame(), factor_names=factors, m=6)
    direct = build_predict_spec(
        ModelSetup.build(parse_formula(formula), result.compact.frame)
    )
    np.testing.assert_allclose(
        result.predict_spec.build_predict_matrix(result.compact.frame),
        direct.build_predict_matrix(result.compact.frame),
        rtol=0,
        atol=0,
    )


def test_combined_budget_rejects_before_model_setup(monkeypatch) -> None:
    frame = _frame()
    formula = parse_formula("y ~ te(x, z, bs='cr', k=8) + g")
    source = _source(frame)
    compact = build_compact_setup_frame(
        formula, source, factor_names=("g",), m=6, batch_rows=7
    )
    assert isinstance(compact, CompactSetupFrame)
    prospective = continuous_setup._prospective_basis_bytes(formula, compact.frame)
    table_limit = max(
        compact.construction_peak_bytes,
        compact.retained_table_bytes + prospective - 1,
    )
    compact.cleanup()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ModelSetup.build must not run beyond the budget")

    monkeypatch.setattr(continuous_setup.ModelSetup, "build", staticmethod(forbidden))
    result = build_continuous_discrete_setup(
        formula,
        source,
        factor_names=("g",),
        m=6,
        batch_rows=7,
        budget=DiscretizationBudget(max_table_bytes=table_limit),
    )
    assert isinstance(result, ContinuousSetupUnavailable)
    assert result.stage == "basis"
    assert result.budget_scope == "table"
    assert result.required_bytes == table_limit + 1


def test_frame_failure_is_translated_without_basis_setup(monkeypatch) -> None:
    frame = _frame()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("basis setup must not follow frame failure")

    monkeypatch.setattr(continuous_setup.ModelSetup, "build", staticmethod(forbidden))
    result = build_continuous_discrete_setup(
        parse_formula("y ~ s(x)"),
        _source(frame),
        m=6,
        budget=DiscretizationBudget(max_table_bytes=64),
    )
    assert isinstance(result, ContinuousSetupUnavailable)
    assert result.stage == "frame"


def test_success_ledger_and_memmap_cleanup(tmp_path) -> None:
    result = _build(
        "y ~ s(x, bs='cr', k=4) + z",
        _frame(),
        m=6,
        selector_memmap_dir=tmp_path,
    )
    paths = result.compact.selector_paths
    assert paths
    assert all(path.exists() for path in paths)
    assert result.retained_table_bytes >= (
        result.compact.retained_table_bytes + result.retained_basis_bytes
    )
    assert result.construction_peak_bytes >= result.retained_table_bytes
    assert result.prospective_basis_bytes >= result.retained_basis_bytes
    result.cleanup()
    assert all(not path.exists() for path in paths)


def test_basis_exception_cleans_owned_selector_memmaps(monkeypatch, tmp_path) -> None:
    def broken(*_args, **_kwargs):
        raise RuntimeError("basis failure")

    monkeypatch.setattr(continuous_setup.ModelSetup, "build", staticmethod(broken))
    with pytest.raises(RuntimeError, match="basis failure"):
        build_continuous_discrete_setup(
            parse_formula("y ~ s(x, bs='cr', k=4)"),
            _source(_frame()),
            m=6,
            selector_memmap_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_source_change_after_basis_setup_is_rejected(monkeypatch) -> None:
    frame = _frame()
    source = _source(frame)
    original = ModelSetup.build

    def mutate(spec, data, weights=None, offset=None):
        result = original(spec, data, weights=weights, offset=offset)
        source._columns["x"][0] = 99.0
        return result

    monkeypatch.setattr(continuous_setup.ModelSetup, "build", staticmethod(mutate))
    with pytest.raises(RuntimeError, match="changed during compact basis setup"):
        build_continuous_discrete_setup(
            parse_formula("y ~ s(x, bs='cr', k=4)"), source, m=6
        )


def test_compact_setup_import_does_not_import_jax() -> None:
    root = Path(__file__).resolve().parents[2] / "jaxgam"
    script = f"""
import sys
import types
from pathlib import Path
root = Path({str(root)!r})
package = types.ModuleType('jaxgam')
package.__path__ = [str(root)]
sys.modules['jaxgam'] = package
import jaxgam.formula.continuous_setup
assert not any(name == 'jax' or name.startswith('jax.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
