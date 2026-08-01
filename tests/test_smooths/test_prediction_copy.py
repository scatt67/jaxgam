"""Prediction-only smooth copies and Phase-1 ``PredictSpec`` contracts."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import numpy as np
import pandas as pd

from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.predict_matrix import build_predict_spec
from jaxgam.formula.terms import SmoothSpec
from jaxgam.smooths.base import Smooth
from jaxgam.smooths.by_variable import FactorBySmooth, NumericBySmooth
from jaxgam.smooths.cubic import (
    CubicRegressionSmooth,
    CubicShrinkageSmooth,
    CyclicCubicSmooth,
)
from jaxgam.smooths.random_effects import RandomEffectSmooth
from jaxgam.smooths.registry import smooth_registry
from jaxgam.smooths.tprs import TPRSShrinkageSmooth, TPRSSmooth
from tests.helpers import _AssertCollector, check_that
from tests.tolerances import STRICT

_BASE_DEFAULT_OK = {
    TPRSSmooth,
    TPRSShrinkageSmooth,
    CubicRegressionSmooth,
    CubicShrinkageSmooth,
    CyclicCubicSmooth,
    RandomEffectSmooth,
}


def _assert_same_attr(original: object, clone: object, attr: str) -> None:
    check_that(
        getattr(clone, attr) is getattr(original, attr),
        f"{attr} transform was copied or dropped",
    )


def _assert_attr_id(obj: object, attr: str, expected_id: int) -> None:
    check_that(
        id(getattr(obj, attr)) == expected_id,
        f"original {attr} was mutated",
    )


def _assert_cache_dropped(obj: object, attr: str) -> None:
    value = getattr(obj, attr, None)
    if attr == "_penalties":
        check_that(value == [], "_penalties was retained")
    else:
        check_that(value is None, f"{attr} was retained")


def _smooth_graph(smooth: object) -> Iterator[object]:
    yield smooth
    base = getattr(smooth, "base_smooth", None)
    if base is not None:
        yield from _smooth_graph(base)
    for marginal in getattr(smooth, "_marginals", ()):
        yield from _smooth_graph(marginal)


def _paired_smooth_graph(
    original: object, clone: object
) -> Iterator[tuple[object, object]]:
    yield original, clone
    original_base = getattr(original, "base_smooth", None)
    if original_base is not None:
        yield from _paired_smooth_graph(original_base, clone.base_smooth)
    for original_marginal, clone_marginal in zip(
        getattr(original, "_marginals", ()),
        getattr(clone, "_marginals", ()),
        strict=True,
    ):
        yield from _paired_smooth_graph(original_marginal, clone_marginal)


def _copy_cases() -> list[tuple[str, object, dict[str, object]]]:
    n = 48
    x = np.linspace(0.01, 0.99, n)
    z = np.linspace(0.98, 0.02, n) + 0.01 * np.sin(np.arange(n))
    g = pd.Series(pd.Categorical(np.resize(["a", "b", "c"], n)))
    train = {"x": x, "z": z, "g": g}
    newdata = {"x": x[::5], "z": z[::5], "g": g.iloc[::5].reset_index(drop=True)}
    cases: list[tuple[str, object, dict[str, object]]] = []

    for key in smooth_registry.available:
        smooth_cls = smooth_registry.get_class(key)
        if key in ("te", "ti"):
            spec = SmoothSpec(
                variables=["x", "z"],
                bs="cr",
                k=5,
                smooth_type=key,
            )
        elif key == "re":
            spec = SmoothSpec(variables=["g"], bs="re", k=5)
        else:
            spec = SmoothSpec(variables=["x"], bs=key, k=6)
        smooth = smooth_cls(spec)
        smooth.setup(train)
        if key in ("te", "ti"):
            # Tensor currently has no design cache. Seed one so the defensive
            # guard remains covered if that implementation changes.
            smooth._X = np.eye(2)
        if key == "gp":
            # setup clears this dead store; repopulate it to exercise the
            # prediction-copy defense independently of that source fix.
            smooth._E_knot = np.eye(2)
        cases.append((key, smooth, newdata))

    factor_base = CubicRegressionSmooth(SmoothSpec(variables=["x"], bs="cr", k=6))
    factor_base.setup(train)
    factor_spec = SmoothSpec(variables=["x"], bs="cr", k=6, by="g")
    cases.append(
        (
            "factor-by",
            FactorBySmooth(
                factor_base,
                factor_spec,
                levels=["a", "b", "c"],
                by_variable="g",
            ),
            newdata,
        )
    )

    numeric_base = TPRSSmooth(SmoothSpec(variables=["x"], bs="tp", k=6))
    numeric_base.setup(train)
    numeric_spec = SmoothSpec(variables=["x"], bs="tp", k=6, by="z")
    cases.append(
        (
            "numeric-by",
            NumericBySmooth(numeric_base, numeric_spec, by_variable="z"),
            newdata,
        )
    )
    return cases


def test_copy_for_prediction_all_smooth_types() -> None:
    """All smooths drop fit caches without changing prediction output."""
    collector = _AssertCollector()
    transform_attrs = (
        "_Xu",
        "_knt",
        "_UZ",
        "_F",
        "_XP_list",
        "_Z_list",
        "_shift",
        "_levels",
    )

    for name, original, newdata in _copy_cases():
        expected = original.predict_matrix(newdata)
        original_nodes = list(_smooth_graph(original))
        cache_ids = {
            (id(node), attr): id(value)
            for node in original_nodes
            for attr in ("_X", "_S", "_E_knot", "_penalties")
            if (value := getattr(node, attr, None)) is not None
        }
        clone = original.copy_for_prediction()

        collector.check(
            f"{name}:prediction",
            lambda clone=clone, expected=expected, newdata=newdata: (
                np.testing.assert_allclose(
                    clone.predict_matrix(newdata),
                    expected,
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                )
            ),
        )
        for original_node, clone_node in _paired_smooth_graph(original, clone):
            collector.check(
                f"{name}:{type(original_node).__name__}_distinct",
                lambda original_node=original_node, clone_node=clone_node: check_that(
                    original_node is not clone_node, "smooth object was not copied"
                ),
            )
            for attr in transform_attrs:
                value = getattr(original_node, attr, None)
                if value is not None:
                    collector.check(
                        f"{name}:{type(original_node).__name__}.{attr}_shared",
                        partial(_assert_same_attr, original_node, clone_node, attr),
                    )
            for attr in ("_X", "_S", "_E_knot", "_penalties"):
                if (id(original_node), attr) in cache_ids:
                    collector.check(
                        f"{name}:{type(original_node).__name__}.{attr}_dropped",
                        partial(_assert_cache_dropped, clone_node, attr),
                    )

        for node in original_nodes:
            for attr in ("_X", "_S", "_E_knot", "_penalties"):
                key = (id(node), attr)
                if key in cache_ids:
                    expected_id = cache_ids[key]
                    collector.check(
                        f"{name}:{type(node).__name__}.{attr}_original_unchanged",
                        partial(_assert_attr_id, node, attr, expected_id),
                    )

    collector.raise_if_any("prediction-only smooth copies")


def test_copy_for_prediction_registry_audit() -> None:
    """New registered smooth classes must explicitly choose a copy strategy."""
    collector = _AssertCollector()
    for key in smooth_registry.available:
        smooth_cls = smooth_registry.get_class(key)
        collector.check(
            key,
            lambda smooth_cls=smooth_cls: check_that(
                smooth_cls.copy_for_prediction is not Smooth.copy_for_prediction
                or smooth_cls in _BASE_DEFAULT_OK,
                f"{smooth_cls.__name__} has not been audited for prediction copies",
            ),
        )
    for wrapper in (FactorBySmooth, NumericBySmooth):
        collector.check(
            wrapper.__name__,
            lambda wrapper=wrapper: check_that(
                "copy_for_prediction" in wrapper.__dict__,
                f"{wrapper.__name__} must recurse into base_smooth",
            ),
        )
    collector.raise_if_any("smooth copy registry audit")


def test_predict_spec_equivalence_aliasing_and_lazy_cache() -> None:
    """PredictSpec is lazy, equivalent, and leaves training caches intact."""
    n = 80
    x = np.linspace(0.01, 0.99, n)
    z = np.linspace(0.99, 0.01, n) + 0.01 * np.sin(np.arange(n))
    w = np.linspace(0.02, 0.98, n)
    data = pd.DataFrame({"x": x, "z": z, "w": w, "y": np.sin(2 * np.pi * x)})
    setup = ModelSetup.build(
        parse_formula("y ~ te(x, z, bs='cr', k=5) + s(w, bs='gp', k=8)"),
        data,
    )
    tensor, gp = [term.smooth for term in setup.coef_map.terms if term.smooth]
    gp._E_knot = np.eye(2)
    live_cache_ids = {
        (id(node), attr): id(value)
        for smooth in (tensor, gp)
        for node in _smooth_graph(smooth)
        for attr in ("_X", "_S", "_E_knot", "_penalties")
        if (value := getattr(node, attr, None)) is not None
    }
    collector = _AssertCollector()

    collector.check(
        "cache_initially_empty",
        lambda: check_that(
            setup._predict_spec_cache is None, "PredictSpec was built eagerly"
        ),
    )
    spec = build_predict_spec(setup)
    collector.check(
        "direct_build_does_not_fill_setup_cache",
        lambda: check_that(
            setup._predict_spec_cache is None,
            "standalone build_predict_spec mutated ModelSetup",
        ),
    )
    spec_matrix = spec.build_predict_matrix(data)
    delegated_matrix = setup.build_predict_matrix(data)
    collector.check(
        "matrix_equivalence",
        lambda: np.testing.assert_array_equal(spec_matrix, delegated_matrix),
    )
    collector.check(
        "training_roundtrip",
        lambda: np.testing.assert_array_equal(delegated_matrix, setup.X),
    )
    collector.check(
        "cache_populated_on_first_prediction",
        lambda: check_that(
            setup._predict_spec_cache is not None,
            "first prediction did not cache PredictSpec",
        ),
    )
    collector.check(
        "cache_reused",
        lambda: check_that(
            setup._lazy_predict_spec() is setup._predict_spec_cache,
            "lazy PredictSpec cache was rebuilt",
        ),
    )

    for smooth in (tensor, gp):
        for node in _smooth_graph(smooth):
            for attr in ("_X", "_S", "_E_knot", "_penalties"):
                key = (id(node), attr)
                if key in live_cache_ids:
                    collector.check(
                        f"live:{type(node).__name__}.{attr}",
                        lambda node=node, attr=attr, key=key: check_that(
                            id(getattr(node, attr)) == live_cache_ids[key],
                            f"live {attr} cache was mutated",
                        ),
                    )

    for term in spec.coef_map.terms:
        if term.smooth is None:
            continue
        for node in _smooth_graph(term.smooth):
            for attr in ("_X", "_S", "_E_knot"):
                collector.check(
                    f"spec:{type(node).__name__}.{attr}",
                    lambda node=node, attr=attr: check_that(
                        getattr(node, attr, None) is None,
                        f"predict spec retained {attr}",
                    ),
                )
            if hasattr(node, "_penalties"):
                collector.check(
                    f"spec:{type(node).__name__}._penalties",
                    lambda node=node: check_that(
                        node._penalties == [], "predict spec retained penalties"
                    ),
                )

    collector.raise_if_any("PredictSpec equivalence and ownership")
