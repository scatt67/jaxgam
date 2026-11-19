"""Whole-request ``predict.bamd`` compression against pinned ``discrete.mf``."""

from __future__ import annotations

import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.continuous_request import (
    FrozenRequestGroup,
    RequestCompressionSpec,
    RequestGroupSpec,
)
from jaxgam.data.discretize import DiscretizationBudget
from jaxgam.data.source import ArrayRowSource, DataFrameRowSource
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT

_LEVELS = ("low", "mid", "high")


def _spec(*groups: RequestGroupSpec, **budget: int) -> RequestCompressionSpec:
    return RequestCompressionSpec(groups, DiscretizationBudget(**budget))


def _r_rows(reference, name: str) -> np.ndarray:
    """Per-row compressed values from pinned ``discrete.mf`` output."""
    start = reference.index_spans[name][0]
    values = reference.frame[name].to_numpy()[: reference.n_unique[name]]
    return values[reference.row_indices[:, start]]


def _check_against_r(
    collector, label, frozen, request, reference, names, *, exact=False
):
    compressed = frozen.apply(request)
    for name in names:
        observed = np.asarray(compressed[name])
        expected = _r_rows(reference, name)
        if exact:
            collector.check(
                f"{label}/{name}",
                lambda o=observed, e=expected: np.testing.assert_array_equal(o, e),
            )
        elif observed.dtype.kind in "OUS" or expected.dtype.kind in "OUS":
            collector.check(
                f"{label}/{name}",
                lambda o=observed, e=expected: np.testing.assert_array_equal(
                    o.astype(str), e.astype(str)
                ),
            )
        else:
            collector.check(
                f"{label}/{name}",
                lambda o=observed, e=expected: np.testing.assert_allclose(
                    o, e, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )


def _uniform(seed: int, size: int, low: float, high: float) -> np.ndarray:
    return np.random.default_rng(seed).uniform(low, high, size)


def test_single_and_parametric_groups_match_pinned_discrete_mf(r_bridge) -> None:
    spec = _spec(RequestGroupSpec(("x",)), RequestGroupSpec(("z",)))
    collector = _AssertCollector()
    for size, rounded in ((800, False), (1001, True), (1300, True)):
        request = pd.DataFrame(
            {
                "x": _uniform(size, size, -0.2, 1.3),
                "z": _uniform(size + 1, size, -2.0, 2.0),
            }
        )
        frozen = spec.freeze(request)
        reference = r_bridge.discrete_model_frame(
            "y ~ z + s(x)", request.assign(y=0.0), m=None
        )
        for group in frozen.groups:
            assert group.rounded is rounded
            assert group.points_per_metric == 1000
            assert group.unique_threshold == 1000
            assert (group.n_unique is None) is rounded
            if not rounded:
                assert group.n_unique == reference.n_unique[group.names[0]]
            else:
                assert reference.n_unique[group.names[0]] <= 1000
        _check_against_r(collector, f"n{size}", frozen, request, reference, "xz")
        if not rounded:
            np.testing.assert_array_equal(frozen.apply(request)["x"], request.x)
    collector.raise_if_any("single-variable request compression")


def test_joint_group_uses_uniquecombs_token_equivalence(r_bridge) -> None:
    """Rows equal to 15 significant digits are one ``uniquecombs`` row."""
    base = pd.DataFrame(
        {
            "x": np.round(_uniform(11, 10_000, 0.0, 1.0), 6),
            "z": np.round(_uniform(12, 10_000, -1.0, 1.0), 6),
        }
    )
    nudged = base.iloc[[0]].assign(x=np.nextafter(base.x.iloc[0], 2.0))
    extra = pd.DataFrame({"x": [0.5000005], "z": [0.25]})
    spec = _spec(RequestGroupSpec(("x", "z")))
    collector = _AssertCollector()
    for label, request, rounded in (
        ("token-collision", pd.concat([base, nudged], ignore_index=True), False),
        ("nudged-first", pd.concat([nudged, base], ignore_index=True), False),
        ("one-extra", pd.concat([base, nudged, extra], ignore_index=True), True),
    ):
        assert request.drop_duplicates().shape[0] == len(request)
        frozen = spec.freeze(request)
        group = frozen.groups[0]
        reference = r_bridge.discrete_model_frame(
            "y ~ s(x, z)", request.assign(y=0.0), m=None
        )
        assert group.rounded is rounded, label
        assert group.points_per_metric == 100
        assert group.unique_threshold == 10_000
        if not rounded:
            assert group.n_unique == reference.n_unique["x"] == 10_000
            # uniquecombs keeps the first occurrence of a token-equal row.
            (merged,) = group.merged_rows
            rows = (
                tuple(nudged.iloc[0][["x", "z"]]),
                tuple(base.iloc[0][["x", "z"]]),
            )
            # (later exact row, first-occurrence representative)
            assert merged == (rows if label == "token-collision" else rows[::-1])
        else:
            assert group.merged_rows == ()
        _check_against_r(collector, label, frozen, request, reference, "xz", exact=True)
    collector.raise_if_any("joint request compression")


def test_mixed_factor_metric_group_counts_present_levels(r_bridge) -> None:
    group = RequestGroupSpec(("x", "g"), ("g",), (_LEVELS,))
    spec = _spec(group)
    collector = _AssertCollector()
    x = np.repeat(np.linspace(0.0, 1.0, 150), 2)
    for label, levels, rounded in (
        ("two-present", ("low", "high"), True),
        ("three-present", ("low", "mid", "high"), False),
    ):
        codes = np.resize(np.arange(len(levels)), len(x))
        values = np.asarray(levels, dtype=object)[codes]
        if len(levels) == 3:
            values = np.asarray(_LEVELS, dtype=object)[np.arange(len(x)) // 100]
        request = pd.DataFrame(
            {
                "x": x,
                "g": pd.Categorical(values, categories=_LEVELS),
            }
        )
        frozen = spec.freeze(request)
        assert frozen.groups[0].unique_threshold == len(levels) * 100, label
        assert frozen.groups[0].rounded is rounded, label
        reference = r_bridge.discrete_model_frame(
            "y ~ s(x, g, bs='re')",
            request.assign(y=0.0),
            factor_names=("g",),
            m=None,
        )
        _check_against_r(collector, label, frozen, request, reference, ("x", "g"))
    collector.raise_if_any("mixed request compression")


def test_factor_only_group_never_rounds_and_passes_values_through() -> None:
    spec = _spec(RequestGroupSpec(("g",), ("g",), (_LEVELS,)), RequestGroupSpec(("x",)))
    request = {
        "g": pd.Series(pd.Categorical(np.resize(np.array(_LEVELS), 2000))),
        "x": np.linspace(0.0, 1.0, 2000),
    }
    frozen = spec.freeze(request)
    factor_group, metric_group = frozen.groups
    assert not factor_group.rounded
    assert factor_group.unique_threshold is None
    assert metric_group.rounded
    compressed = frozen.apply(request)
    assert compressed["g"] is request["g"]
    assert frozen.rounded_names == ("x",)
    assert set(compressed["x"]) <= set(metric_group.grid("x"))
    with pytest.raises(KeyError, match="not a metric variable"):
        metric_group.grid("g")


def test_source_prepass_is_batch_invariant_and_freezes_before_apply() -> None:
    spec = _spec(RequestGroupSpec(("x",)), RequestGroupSpec(("w", "z")))
    size = 12_500
    request = pd.DataFrame(
        {
            "x": _uniform(21, size, -0.5, 1.5),
            "w": _uniform(22, size, 0.0, 1.0),
            "z": _uniform(23, size, 0.0, 1.0),
        }
    )
    whole = spec.freeze(request)
    expected = whole.apply(request)
    for batch_rows in (1, 997, size):
        source = DataFrameRowSource(request)
        frozen = spec.freeze_source(source, batch_rows)
        assert frozen.groups == whole.groups
        assert frozen.source_fingerprint == source.fingerprint()
        pieces = {name: [] for name in request}
        for batch in source.scan(batch_rows):
            compressed = frozen.apply(dict(batch.columns))
            for name in request:
                pieces[name].append(np.asarray(compressed[name]))
        for name in request:
            np.testing.assert_array_equal(np.concatenate(pieces[name]), expected[name])
    # Freezing each batch independently changes its ranges and decisions.
    first = spec.freeze(request.iloc[:997])
    assert first.groups != whole.groups


def test_prepass_retained_state_does_not_scale_with_rows() -> None:
    spec = _spec(RequestGroupSpec(("x",)), RequestGroupSpec(("w", "z")))
    peaks = []
    reserved = []
    for size in (50_000, 1_000_000):
        rng = np.random.default_rng(size)
        source = ArrayRowSource(
            {
                "x": rng.uniform(size=size),
                "w": rng.uniform(size=size),
                "z": rng.uniform(size=size),
            }
        )
        tracemalloc.start()
        frozen = spec.freeze_source(source, 4096)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        assert all(group.rounded for group in frozen.groups)
        reserved.append(frozen.construction_peak_bytes)
    # The reserved key state is fixed by mgcv's thresholds, not by n, and the
    # measured peak stays below one 1M-row float64 copy (8 MB).
    assert reserved[0] == reserved[1] < 8 * 1024 * 1024
    assert peaks[1] < 4 * 1024 * 1024
    assert peaks[1] < 2 * peaks[0] + 1024 * 1024


@pytest.mark.parametrize(
    ("request_columns", "error", "match"),
    [
        ({"x": np.array([0.1, np.nan])}, ValueError, "non-finite"),
        ({"x": np.array(["a", "b"])}, TypeError, "real numeric"),
        (
            {"x": pd.Series(pd.Categorical([0.1, 0.2]))},
            TypeError,
            "not categorical",
        ),
        ({"x": np.ones((2, 2))}, ValueError, "one-dimensional"),
        ({"w": np.array([0.1, 0.2])}, ValueError, "absent from newdata"),
    ],
)
def test_invalid_metric_requests_are_rejected(request_columns, error, match) -> None:
    with pytest.raises(error, match=match):
        _spec(RequestGroupSpec(("x",))).freeze(request_columns)


def test_factor_only_groups_pass_missing_and_unseen_values_through() -> None:
    """Factor semantics stay with the frozen evaluators, never compression."""
    spec = _spec(RequestGroupSpec(("g",), ("g",)), RequestGroupSpec(("x",)))
    for values in (
        np.array(["low", "new", None], dtype=object),
        pd.Series(pd.Categorical(["low", np.nan, "mid"], categories=_LEVELS)),
    ):
        request = {"x": np.linspace(0.0, 1.0, 3), "g": values}
        frozen = spec.freeze(request)
        assert not any(group.rounded for group in frozen.groups)
        compressed = frozen.apply(request)
        assert compressed["g"] is values
        np.testing.assert_array_equal(compressed["x"], request["x"])


def test_mixed_group_excludes_missing_and_unseen_rows() -> None:
    """Rows a mixed random effect zeroes neither count nor set the range."""
    spec = _spec(RequestGroupSpec(("x", "g"), ("g",), (_LEVELS,)))
    valid_x = np.linspace(0.0, 1.0, 150)
    request = {
        # Two present levels allow 2 * 100 rows; 300 valid rows are rounded.
        "x": np.r_[valid_x, valid_x + 0.001, [5.0, -3.0, 7.0]],
        "g": np.concatenate(
            (
                np.repeat("low", 150),
                np.repeat("high", 150),
                np.array(["new", None, "mid-typo"], dtype=object),
            )
        ).astype(object),
    }
    frozen = spec.freeze(request)
    (group,) = frozen.groups
    assert group.rounded
    assert group.unique_threshold == 200
    assert group.metric_ranges == ((0.0, float(np.max(valid_x + 0.001))),)
    assert group.factor_levels == (_LEVELS,)
    compressed = frozen.apply(request)
    assert set(compressed["x"][:300]) <= set(group.grid("x"))
    np.testing.assert_array_equal(compressed["x"][300:], [5.0, -3.0, 7.0])
    np.testing.assert_array_equal(group.included(request), np.arange(303) < 300)
    small = spec.freeze(
        {"x": np.array([0.1, 0.2, 0.3]), "g": np.array(["low", None, "mid"])}
    )
    assert small.groups[0].n_unique == 2
    assert small.groups[0].unique_threshold == 200
    with pytest.raises(ValueError, match="one-dimensional"):
        spec.freeze({"x": np.array([0.1, 0.2]), "g": np.array([["low"], ["mid"]])})


def test_mixed_group_merges_token_equal_rows_within_each_level() -> None:
    spec = _spec(RequestGroupSpec(("x", "g"), ("g",), (_LEVELS,)))
    x = np.round(np.linspace(0.01, 0.99, 100), 6)
    nudged = np.nextafter(x[0], 2.0)
    request = {
        "x": np.r_[x, nudged, nudged],
        "g": np.concatenate((np.repeat("low", 101), ["high"])).astype(object),
    }
    frozen = spec.freeze(request)
    (group,) = frozen.groups
    assert not group.rounded
    assert group.unique_threshold == 200
    assert group.n_unique == 101
    assert group.merged_rows == (((float(nudged), "low"), (float(x[0]), "low")),)
    compressed = frozen.apply(request)
    np.testing.assert_array_equal(compressed["x"][:101], np.r_[x, x[0]])
    assert compressed["x"][101] == nudged
    assert compressed["g"] is request["g"]


def test_rounded_joint_grid_with_merged_tokens_is_refused() -> None:
    size = 10_001
    request = {
        "x": 1e15 + 0.125 * (np.arange(size) % 9),
        "z": np.linspace(0.0, 1.0, size),
    }
    with pytest.raises(NotImplementedError, match="15 significant digits"):
        _spec(RequestGroupSpec(("x", "z"))).freeze(request)


def test_constant_metric_requiring_rounding_is_undefined() -> None:
    size = 10_001
    request = {"x": np.full(size, 0.25), "z": np.linspace(0.0, 1.0, size)}
    with pytest.raises(ValueError, match="undefined range grid"):
        _spec(RequestGroupSpec(("x", "z"))).freeze(request)


def test_budgets_reject_before_unbounded_state() -> None:
    request = {"x": np.linspace(0.0, 1.0, 1500)}
    with pytest.raises(MemoryError, match="max_table_bytes"):
        _spec(RequestGroupSpec(("x",)), max_table_bytes=1024).freeze(request)
    with pytest.raises(MemoryError, match="max_batch_bytes"):
        _spec(RequestGroupSpec(("x",)), max_batch_bytes=64).freeze(request)
    with pytest.raises(MemoryError, match="before reaching"):
        _spec(RequestGroupSpec(("x",)), max_unique_rows=10).freeze(request)
    small = _spec(RequestGroupSpec(("x",)), max_unique_rows=10).freeze(
        {"x": np.linspace(0.0, 1.0, 10)}
    )
    assert not small.groups[0].rounded
    assert small.groups[0].n_unique == 10


def test_apply_rejects_rows_outside_the_frozen_request() -> None:
    spec = _spec(RequestGroupSpec(("x",)))
    frozen = spec.freeze({"x": np.linspace(0.0, 1.0, 1500)})
    with pytest.raises(ValueError, match="outside the frozen prediction"):
        frozen.apply({"x": np.array([1.5])})


def test_source_requests_validate_rows_padding_and_mutation() -> None:
    spec = _spec(RequestGroupSpec(("x",)))
    with pytest.raises(ValueError, match="no rows"):
        spec.freeze({"x": np.array([], dtype=float)})
    with pytest.raises(ValueError, match="no variables"):
        spec.freeze({})
    with pytest.raises(ValueError, match="share one length"):
        _spec(RequestGroupSpec(("x",)), RequestGroupSpec(("z",))).freeze(
            {"x": np.ones(3), "z": np.ones(2)}
        )

    class Empty:
        n_rows = 0

        def fingerprint(self):
            return "empty"

        def scan(self, _batch_rows):
            raise AssertionError("an empty request must not be scanned")

    with pytest.raises(ValueError, match="no rows"):
        spec.freeze_source(Empty(), 4)

    class Changing(ArrayRowSource):
        calls = 0

        def fingerprint(self):
            self.calls += 1
            return f"version-{self.calls}"

    with pytest.raises(RuntimeError, match="changed during"):
        spec.freeze_source(Changing({"x": np.linspace(0.0, 1.0, 9)}), 4)

    class Padded(ArrayRowSource):
        def scan(self, batch_rows):
            for batch in super().scan(batch_rows):
                object.__setattr__(batch, "valid", np.zeros_like(batch.valid))
                yield batch

    with pytest.raises(NotImplementedError, match="padded"):
        spec.freeze_source(Padded({"x": np.linspace(0.0, 1.0, 9)}), 4)

    class Short(ArrayRowSource):
        def scan(self, batch_rows):
            yield next(iter(super().scan(batch_rows)))

    with pytest.raises(RuntimeError, match="inconsistent row count"):
        spec.freeze_source(Short({"x": np.linspace(0.0, 1.0, 9)}), 4)


def test_spec_and_frozen_group_validation() -> None:
    with pytest.raises(ValueError, match="distinct"):
        RequestGroupSpec(("x", "x"))
    with pytest.raises(ValueError, match="belong to the group"):
        RequestGroupSpec(("x",), ("g",))
    with pytest.raises(ValueError, match="do not match"):
        RequestGroupSpec(("x", "g"), ("g",), (_LEVELS, _LEVELS))
    with pytest.raises(ValueError, match="needs fitted levels"):
        RequestGroupSpec(("x", "g"), ("g",))
    with pytest.raises(ValueError, match="need their fitted levels"):
        RequestGroupSpec(("g",), ("g",), ((),))
    assert RequestGroupSpec(("g",), ("g",)).factor_levels == ()
    with pytest.raises(TypeError, match="DiscretizationBudget"):
        RequestCompressionSpec((RequestGroupSpec(("x",)),), budget=None)
    with pytest.raises(ValueError, match="share variables"):
        _spec(RequestGroupSpec(("x",)), RequestGroupSpec(("x", "z")))
    with pytest.raises(ValueError, match="ranges do not match"):
        FrozenRequestGroup(("x",), ("x",), (), 1000, 1000, 3, False, ())
    with pytest.raises(ValueError, match="unique threshold"):
        FrozenRequestGroup(("x",), ("x",), (), 1000, None, None, True, ((0, 1),))
    with pytest.raises(ValueError, match="only unrounded joint groups"):
        FrozenRequestGroup(
            ("x",), ("x",), (), 1000, 1000, 2, False, ((0, 1),), (((0.1,), (0.2,)),)
        )
    with pytest.raises(ValueError, match="only mixed groups carry"):
        FrozenRequestGroup(
            ("g",), (), ("g",), 1000, None, None, False, (), (), (_LEVELS,)
        )
    assert RequestGroupSpec(("x", "z", "w")).points_per_metric == 25


def test_request_module_imports_without_jax() -> None:
    root = Path(__file__).resolve().parents[2] / "jaxgam"
    script = f"""
import sys
import types
from pathlib import Path
root = Path({str(root)!r})
for name, path in (("jaxgam", root), ("jaxgam.data", root / "data")):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
from jaxgam.data.continuous_request import RequestCompressionSpec
assert callable(RequestCompressionSpec)
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
