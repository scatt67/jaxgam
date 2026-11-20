"""Per-group and per-term discretization metadata against pinned ``bam``."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data.discretize import DiscretizationBudget
from jaxgam.data.source import DataFrameRowSource
from jaxgam.formula.discrete.continuous_metadata import (
    ContinuousDiscretizationMetadata,
    DiscreteTermMetadata,
    DiscretizationGroupMetadata,
    build_discretization_metadata,
    build_request_compression,
)
from jaxgam.formula.discrete.continuous_setup import (
    ContinuousDiscreteSetup,
    build_continuous_discrete_setup,
)
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector, check_that
from tests.tolerances import STRICT

_FORMULA = "y ~ u + te(x, w, bs='cr', k=3, by=g) + s(x, bs='cr', k=5) + s(v, q, k=8)"


def _frame(n_rows: int = 90) -> pd.DataFrame:
    position = np.arange(1, n_rows + 1, dtype=float)
    x = (position - 0.5) / n_rows + 0.03 * np.sin(position)
    w = np.cos(position * 0.71) + position / n_rows
    v = np.sin(position * 0.37)
    q = np.cos(position * 0.53) + 0.1 * position / n_rows
    u = np.resize(np.linspace(-1.0, 1.0, 12), n_rows)
    g = pd.Series(
        pd.Categorical(
            np.resize(np.array(("low", "mid", "high")), n_rows),
            categories=("low", "mid", "high"),
            ordered=True,
        )
    )
    y = 0.2 + np.sin(2.5 * x) + np.where(g == "mid", 0.3 * w, -0.2 * w) + v * q
    return pd.DataFrame(
        {"y": y + 0.1 * u, "u": u, "x": x, "w": w, "g": g, "v": v, "q": q}
    )


def _setup(
    frame: pd.DataFrame, m=30, formula: str = _FORMULA
) -> ContinuousDiscreteSetup:
    setup = build_continuous_discrete_setup(
        parse_formula(formula),
        DataFrameRowSource(frame, response="y"),
        factor_names=("g",) if "g" in frame else (),
        m=m,
        batch_rows=7,
    )
    assert isinstance(setup, ContinuousDiscreteSetup)
    return setup


def _metadata(setup, *, bound: int = 10_000) -> ContinuousDiscretizationMetadata:
    return build_discretization_metadata(
        setup,
        prediction_policy="mgcv_predict_bamd",
        max_retained_representatives=bound,
        shuffle_seed=8547,
    )


def _check_against_bam(collector, label, metadata, reference, frame) -> None:
    """Compare every group and term with pinned ``bam``'s own discrete.mf."""
    collector.check(
        f"{label}/variables",
        lambda: check_that(
            {name for group in metadata.groups for name in group.variables}
            == set(reference.variables),
            "metadata and bam discretize different variables",
        ),
    )
    for group in metadata.groups:
        spans = {reference.variables[name].span for name in group.variables}
        collector.check(
            f"{label}/{group.variables}/joint span",
            lambda spans=spans: check_that(
                len(spans) == 1, "jointly discretized variables need one bam span"
            ),
        )
        for name in group.variables:
            expected = reference.variables[name]
            collector.check(
                f"{label}/{name}/q",
                lambda group=group, expected=expected: check_that(
                    group.n_unique == expected.n_unique,
                    f"q {group.n_unique} != bam nr {expected.n_unique}",
                ),
            )
            observed = group.representatives_of(name)
            if name in group.factor_variables:
                collector.check(
                    f"{label}/{name}/levels",
                    lambda observed=observed, expected=expected: (
                        np.testing.assert_array_equal(
                            observed.astype(str), expected.representatives
                        )
                    ),
                )
                continue
            collector.check(
                f"{label}/{name}/representatives",
                lambda observed=observed, expected=expected: np.testing.assert_allclose(
                    observed,
                    expected.representatives,
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
            collector.check(
                f"{label}/{name}/range",
                lambda group=group, name=name, expected=expected: (
                    np.testing.assert_allclose(
                        group.range_of(name),
                        expected.data_range,
                        rtol=STRICT.rtol,
                        atol=STRICT.atol,
                    )
                ),
            )
            # bam rounded this variable exactly when its representatives are
            # not all observed values.
            bam_rounded = not set(expected.representatives) <= set(frame[name])
            collector.check(
                f"{label}/{name}/rounded",
                lambda group=group, bam_rounded=bam_rounded: check_that(
                    group.discretized == bam_rounded,
                    f"discretized={group.discretized}, bam rounded={bam_rounded}",
                ),
            )
            if group.discretized:
                collector.check(
                    f"{label}/{name}/on grid",
                    lambda group=group, name=name, observed=observed: check_that(
                        set(observed) <= set(group.grid(name)),
                        "rounded representatives must lie on the range grid",
                    ),
                )
    expected_terms = {
        name: tuple(margin for margin in margins if margin != ("(Intercept)",))
        for name, margins in reference.term_margins.items()
    }
    observed_terms = {term.label: term.margins for term in metadata.terms}
    collector.check(
        f"{label}/term margins",
        lambda: check_that(
            observed_terms == expected_terms,
            f"terms {observed_terms} != bam {expected_terms}",
        ),
    )


@pytest.mark.parametrize("m", [30, (30, 4, 12)])
def test_group_and_term_metadata_match_pinned_bam(r_bridge, m) -> None:
    frame = _frame()
    setup = _setup(frame, m=m)
    try:
        metadata = _metadata(setup)
    finally:
        setup.cleanup()
    reference = r_bridge.bam_discretization(_FORMULA, frame, factor_names=("g",), m=m)
    collector = _AssertCollector()
    _check_against_bam(collector, f"m={m}", metadata, reference, frame)
    collector.raise_if_any("discretization metadata")
    roles = {group.variables: group.role for group in metadata.groups}
    assert roles == {
        ("g",): "by",
        ("x",): "tensor_margin",
        ("w",): "tensor_margin",
        ("v", "q"): "smooth",
        ("y",): "response",
        ("u",): "parametric",
    }
    joint = metadata.group_of("q")
    assert joint.variables == ("v", "q")
    assert joint.points_per_metric == (6 if m == 30 else 4)
    assert joint.unique_threshold == joint.points_per_metric**2
    assert joint.metric_variables == ("v", "q")
    parametric = metadata.group_of("u")
    assert not parametric.discretized
    assert parametric.grid("u") is None
    assert parametric.requested_m == 30
    term = metadata.term("te(x,w):gmid")
    assert term.margin_roles == ("by", "tensor_margin", "tensor_margin")
    assert metadata.term("u").margin_roles == ("parametric",)
    assert metadata.term("(Intercept)").kind == "intercept"


def test_public_discrete_result_records_metadata_and_policy(r_bridge) -> None:
    frame = _frame()
    formula = "y ~ u + te(x, w, bs='cr', k=3, by=g)"
    control = FitControl(
        execution="discrete",
        batch_rows=11,
        discretization=ContinuousDiscretizationControl(
            m=30, factor_names=("g",), prediction_policy="mgcv_predict_bamd"
        ),
    )
    result = GAM(formula, sp=[0.35, 0.7, 0.45, 0.8], control=control).fit(
        DataFrameRowSource(frame, response="y"), result="prediction"
    )
    metadata = result.discretization
    assert metadata is not None
    assert metadata.strategy == result.discretization_strategy == "mgcv_range_grid"
    assert metadata.prediction_policy == result.prediction_policy
    assert result.prediction_policy == "mgcv_predict_bamd"
    assert result.to_predictor().prediction_policy == "mgcv_predict_bamd"
    reference = r_bridge.bam_discretization(formula, frame, factor_names=("g",), m=30)
    collector = _AssertCollector()
    _check_against_bam(collector, "public", metadata, reference, frame)
    collector.raise_if_any("public discretization metadata")
    restored = pickle.loads(pickle.dumps(result))
    assert restored.discretization.terms == metadata.terms
    # Records owning NumPy arrays compare by identity, never elementwise.
    assert restored.discretization != metadata
    assert restored.discretization.groups[0] != metadata.groups[0]
    assert metadata.groups[0] == metadata.groups[0]
    assert restored.to_predictor().prediction_policy == "mgcv_predict_bamd"
    for group in restored.discretization.groups:
        assert all(not values.flags.writeable for values in group.representatives)


def test_representatives_are_bounded_and_do_not_scale_with_rows() -> None:
    small = _setup(_frame(90))
    large = _setup(_frame(3000))
    try:
        full = [_metadata(setup) for setup in (small, large)]
        bounded = _metadata(large, bound=29)
        none = _metadata(large, bound=0)
    finally:
        small.cleanup()
        large.cleanup()
    # Both requests are rounded to the same grids, so q (and the retained
    # representative storage) is independent of n.
    for name in ("x", "w", "y"):
        assert full[1].group_of(name).discretized
        assert full[1].group_of(name).n_unique <= 30
    assert full[1].retained_bytes < 40 * 1024
    for group in full[1].groups:
        assert group.representatives is not None
        assert all(len(values) == group.n_unique for values in group.representatives)
    assert full[0].retained_bytes > 0
    for group in bounded.groups:
        assert group.representatives_retained is (group.n_unique <= 29)
        if not group.representatives_retained:
            assert group.representatives_of(group.variables[0]) is None
    assert all(group.representatives is None for group in none.groups)
    assert none.retained_bytes == 0
    assert none.max_retained_representatives == 0
    descriptor = none.group_of("x")
    low, high = descriptor.range_of("x")
    grid = descriptor.grid("x")
    assert grid[0] == low
    assert grid[-1] == high
    assert len(grid) == descriptor.points_per_metric


def test_request_compression_follows_fitted_groups_without_response() -> None:
    setup = _setup(_frame())
    try:
        spec = build_request_compression(setup, DiscretizationBudget())
        assert spec is not None
        assert [group.names for group in spec.groups] == [
            ("g",),
            ("x",),
            ("w",),
            ("v", "q"),
            ("u",),
        ]
        assert spec.groups[0].factor_names == ("g",)
        # Factor-only groups pass values through; only mixed groups need levels.
        assert all(group.factor_levels == () for group in spec.groups)
    finally:
        setup.cleanup()
    parametric = _setup(_frame().drop(columns="g"), formula="y ~ u + x")
    try:
        assert build_request_compression(parametric, DiscretizationBudget()) is None
    finally:
        parametric.cleanup()


def test_mixed_group_spec_keeps_fitted_levels() -> None:
    frame = pd.DataFrame(
        {
            "y": np.linspace(0.1, 2.0, 45),
            "x": np.resize(np.array((1.0, 2.0, 4.0, 7.0, 11.0)), 45),
            "g": pd.Series(
                pd.Categorical(
                    np.resize(np.array(("a", "b", "c")), 45),
                    categories=("a", "b", "c", "unused"),
                )
            ),
        }
    )
    setup = _setup(frame, m=9, formula="y ~ s(x, g, bs='re')")
    try:
        spec = build_request_compression(setup, DiscretizationBudget())
        metadata = _metadata(setup)
    finally:
        setup.cleanup()
    assert spec is not None
    (group,) = spec.groups
    assert group.names == ("x", "g")
    # mgcv's model frame drops unused levels: only observed levels are kept.
    assert group.factor_levels == (("a", "b", "c"),)
    joint = metadata.group_of("g")
    assert joint.factor_level_counts == (("g", 3),)
    assert joint.unique_threshold == 3 * joint.points_per_metric


def test_metadata_validation() -> None:
    def group(**overrides):
        values = {
            "variables": ("x",),
            "factor_variables": (),
            "role": "smooth",
            "policy": "mgcv_compress_df",
            "requested_m": 30,
            "points_per_metric": 30,
            "unique_threshold": 30,
            "discretized": False,
            "n_unique": 3,
            "metric_ranges": (("x", 0.0, 1.0),),
            "factor_level_counts": (),
            "representatives": (np.array([0.0, 0.5, 1.0]),),
        }
        values.update(overrides)
        return DiscretizationGroupMetadata(**values)

    assert group().representatives_of("x") is not None
    with pytest.raises(ValueError, match="distinct variables"):
        group(variables=("x", "x"))
    with pytest.raises(ValueError, match="cover variables"):
        group(metric_ranges=())
    with pytest.raises(ValueError, match="positive"):
        group(n_unique=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        group(n_unique=31, representatives=None)
    with pytest.raises(ValueError, match="one column per variable"):
        group(representatives=())
    with pytest.raises(ValueError, match="q rows"):
        group(representatives=(np.array([0.0]),))
    with pytest.raises(KeyError, match="not a metric variable"):
        group().range_of("z")
    with pytest.raises(ValueError, match="must align"):
        DiscreteTermMetadata("s(x)", "smooth", (("x",),), (), (0,))
    metadata = ContinuousDiscretizationMetadata(
        "mgcv_range_grid", "continuous_frozen_basis", (group(),), (), 1, 8547
    )
    with pytest.raises(KeyError, match="No discretization group"):
        metadata.group_of("z")
    with pytest.raises(KeyError, match="No discrete term"):
        metadata.term("s(z)")
    setup = _setup(_frame())
    try:
        with pytest.raises(ValueError, match="non-negative"):
            build_discretization_metadata(
                setup,
                prediction_policy="continuous_frozen_basis",
                max_retained_representatives=-1,
                shuffle_seed=8547,
            )
    finally:
        setup.cleanup()


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_representative_bound_control_is_validated(invalid) -> None:
    with pytest.raises(ValueError, match="max_retained_representatives"):
        ContinuousDiscretizationControl(max_retained_representatives=invalid)


def test_metadata_module_imports_without_jax() -> None:
    root = Path(__file__).resolve().parents[2] / "jaxgam"
    script = f"""
import sys
import types
from pathlib import Path
root = Path({str(root)!r})
for name, path in (
    ("jaxgam", root),
    ("jaxgam.data", root / "data"),
    ("jaxgam.formula", root / "formula"),
):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
from jaxgam.formula.discrete.continuous_metadata import build_discretization_metadata
assert callable(build_discretization_metadata)
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
