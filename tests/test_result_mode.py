"""End-to-end gates for full and inference result materialization."""

from __future__ import annotations

import dataclasses
import pickle
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_type

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, GAMInferenceResult, GAMPredictor, GAMResults
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.registry import get_family
from jaxgam.links.links import LogLink
from tests.helpers import (
    SEED,
    _AssertCollector,
    _generate_family_data,
    _inference_equivalence_cases,
    _inference_r_cases,
    r_available,
)
from tests.tolerances import MODERATE

if TYPE_CHECKING:
    _typing_data = pd.DataFrame()
    assert_type(GAM("y ~ x").fit(_typing_data), GAMResults)
    assert_type(GAM("y ~ x").fit(_typing_data, result="full"), GAMResults)
    assert_type(
        GAM("y ~ x").fit(_typing_data, result="inference"),
        GAMInferenceResult,
    )


def _walk_objects(root: Any) -> list[Any]:
    """Recursively walk an object graph once per identity."""
    seen: set[int] = set()
    objects: list[Any] = []

    def visit(obj: Any) -> None:
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, type)):
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        objects.append(obj)

        if isinstance(obj, np.ndarray):
            return
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for field in dataclasses.fields(obj):
                visit(getattr(obj, field.name))
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                visit(key)
                visit(value)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for value in obj:
                visit(value)
            return
        for value in getattr(obj, "__dict__", {}).values():
            visit(value)

    visit(root)
    return objects


def _retained_array_bytes(root: Any) -> int:
    """Sum distinct NumPy buffers reachable from a result object."""
    return sum(obj.nbytes for obj in _walk_objects(root) if isinstance(obj, np.ndarray))


def _multi_smooth_case(n: int = 120) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Exercise tensor, TPRS, and GP prediction transforms together."""
    rng = np.random.default_rng(SEED + 800)
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    x3 = rng.uniform(size=n)
    x4 = rng.uniform(size=n)
    y = (
        np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
        + 0.4 * np.sin(2 * np.pi * x3)
        + 0.2 * x4
        + rng.normal(scale=0.2, size=n)
    )
    data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
    newdata = pd.DataFrame(
        {
            "x1": rng.uniform(size=24),
            "x2": rng.uniform(size=24),
            "x3": rng.uniform(size=24),
            "x4": rng.uniform(size=24),
        }
    )
    formula = "y ~ ti(x1, x2, k=4) + s(x3, k=6, bs='tp') + s(x4, k=6, bs='gp')"
    return formula, data, newdata


def test_result_types_surface_diagnostics_and_exports() -> None:
    """The mode selects two frozen, deliberately narrow public surfaces."""
    data = _generate_family_data("gaussian", n=90)
    model = GAM("y ~ s(x, k=6, bs='cr')", sp=[1.0])
    implicit = model.fit(data)
    full = model.fit(data, result="full")
    inference = model.fit(data, result="inference")
    collector = _AssertCollector()

    collector.check(
        "implicit full type",
        lambda: np.testing.assert_equal(isinstance(implicit, GAMResults), True),
    )
    collector.check(
        "explicit full type",
        lambda: np.testing.assert_equal(isinstance(full, GAMResults), True),
    )
    collector.check(
        "inference type",
        lambda: np.testing.assert_equal(
            isinstance(inference, GAMInferenceResult), True
        ),
    )
    collector.check(
        "predictor type",
        lambda: np.testing.assert_equal(
            isinstance(inference.to_predictor(), GAMPredictor), True
        ),
    )
    collector.check(
        "composed predictor identity",
        lambda: np.testing.assert_equal(
            inference.to_predictor() is inference._predictor, True
        ),
    )
    for name in ("coefficients", "Vp", "family"):
        collector.check(
            f"{name} delegates to predictor",
            lambda name=name: np.testing.assert_equal(
                getattr(inference, name) is getattr(inference._predictor, name),
                True,
            ),
        )
    for name in ("summary", "plot"):
        collector.check(
            f"no {name}",
            lambda name=name: np.testing.assert_equal(hasattr(inference, name), False),
        )
    with pytest.raises(TypeError):
        inference.predict()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="result must be"):
        model.fit(data, result="compact")  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        full.scale = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        inference.scale = 0.0  # type: ignore[misc]

    array_fields = {"coefficients", "Vp", "edf", "edf1", "smoothing_params"}
    shared_fields = (
        "coefficients",
        "Vp",
        "edf",
        "edf1",
        "edf_total",
        "deviance",
        "null_deviance",
        "score",
        "scale",
        "theta",
        "smoothing_params",
        "converged",
        "n_iter",
        "convergence_info",
        "method",
        "lambda_strategy",
        "execution_path",
        "n",
        "formula",
        "smooth_info",
        "term_names",
    )
    for name in shared_fields:
        actual = getattr(inference, name)
        expected = getattr(full, name)
        if name in array_fields:
            collector.check(
                f"{name} equality",
                lambda a=actual, e=expected: np.testing.assert_array_equal(a, e),
            )
        else:
            collector.check(
                f"{name} equality",
                lambda a=actual, e=expected: np.testing.assert_equal(a, e),
            )
    collector.check(
        "edf labels",
        lambda: np.testing.assert_equal(len(inference.smooth_info), len(inference.edf)),
    )
    collector.check(
        "family equality",
        lambda: np.testing.assert_equal(
            (
                type(inference.family),
                type(inference.family.link),
                inference.family.family_name,
            ),
            (type(full.family), type(full.family.link), full.family.family_name),
        ),
    )
    collector.check(
        "repr full-result hint",
        lambda: np.testing.assert_equal("result='full'" in repr(inference), True),
    )
    collector.check(
        "predict matrix",
        lambda: np.testing.assert_array_equal(
            inference.predict_matrix(data), full.predict_matrix(data)
        ),
    )
    collector.raise_if_any("result mode surface")


def test_inference_result_retains_no_banned_training_or_penalty_state() -> None:
    """Lean results keep transforms while dropping every banned owner/cache."""
    formula, data, _ = _multi_smooth_case()
    result = GAM(formula, sp=[1.0, 1.0, 1.0, 1.0]).fit(data, result="inference")
    collector = _AssertCollector()

    for name in (
        "setup",
        "X",
        "y",
        "weights",
        "offset",
        "fitted_values",
        "linear_predictor",
        "training_data",
    ):
        collector.check(
            f"no {name}",
            lambda name=name: np.testing.assert_equal(hasattr(result, name), False),
        )

    graph = _walk_objects(result._predictor._predict_spec)
    for obj in graph:
        for attr in ("_X", "_S", "_E_knot"):
            if hasattr(obj, attr):
                collector.check(
                    f"{type(obj).__name__}.{attr} dropped",
                    lambda obj=obj, attr=attr: np.testing.assert_equal(
                        getattr(obj, attr), None
                    ),
                )
        if hasattr(obj, "_penalties"):
            collector.check(
                f"{type(obj).__name__}._penalties dropped",
                lambda obj=obj: np.testing.assert_equal(len(obj._penalties), 0),
            )

    for attr in ("_XP_list", "_Z_list", "_Xu", "_UZ", "_knt", "_shift"):
        values = [getattr(obj, attr) for obj in graph if hasattr(obj, attr)]
        collector.check(
            f"{attr} present",
            lambda values=values: np.testing.assert_equal(
                any(
                    value is not None and (not isinstance(value, list) or value)
                    for value in values
                ),
                True,
            ),
        )
    collector.raise_if_any("lean retained-state invariant")


def test_full_and_inference_predictions_are_byte_identical() -> None:
    """The actual fit-mode wiring agrees across the prediction zoo."""
    collector = _AssertCollector()
    for name, case in _inference_equivalence_cases().items():
        formula, data, newdata, family, sp, train_offset, predict_offset = case
        model = GAM(formula, family=family, sp=sp)
        full = model.fit(data, offset=train_offset, result="full")
        inference = model.fit(data, offset=train_offset, result="inference")
        collector.check(
            f"{name}: matrix",
            lambda f=full, i=inference, nd=newdata: np.testing.assert_array_equal(
                f.predict_matrix(nd), i.predict_matrix(nd)
            ),
        )
        for pred_type in ("link", "response"):
            for se_fit in (False, True):

                def assert_prediction(
                    full_result=full,
                    inference_result=inference,
                    prediction_data=newdata,
                    prediction_type=pred_type,
                    with_se=se_fit,
                    prediction_offset=predict_offset,
                ) -> None:
                    np.testing.assert_array_equal(
                        full_result.predict(
                            prediction_data,
                            pred_type=prediction_type,
                            se_fit=with_se,
                            offset=prediction_offset,
                        ),
                        inference_result.predict(
                            prediction_data,
                            pred_type=prediction_type,
                            se_fit=with_se,
                            offset=prediction_offset,
                        ),
                    )

                collector.check(
                    f"{name}: {pred_type}, se_fit={se_fit}",
                    assert_prediction,
                )

    full_offset, inference_offset, offset_newdata = _offset_results()
    for label, offset_result in (
        ("full", full_offset),
        ("inference", inference_offset),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            offset_result.predict(offset_newdata)
        collector.check(
            f"{label} external-offset warning",
            lambda caught=caught: np.testing.assert_equal(
                any("external offset" in str(item.message) for item in caught), True
            ),
        )
        collector.check(
            f"{label} external-offset warning caller",
            lambda caught=caught: np.testing.assert_equal(
                Path(caught[0].filename).parts[-2:] if caught else None,
                Path(__file__).parts[-2:],
            ),
        )
    collector.raise_if_any("full/inference prediction equivalence")


def _offset_results() -> tuple[GAMResults, GAMInferenceResult, pd.DataFrame]:
    data = _generate_family_data("poisson", n=70)
    newdata = pd.DataFrame({"x": np.linspace(0.1, 0.9, 12)})
    model = GAM("y ~ s(x, k=6, bs='cr')", family="poisson", sp=[1.0])
    offset = np.full(len(data), 0.2)
    return (
        model.fit(data, offset=offset, result="full"),
        model.fit(data, offset=offset, result="inference"),
        newdata,
    )


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
def test_actual_inference_result_and_pickle_match_r(r_bridge) -> None:
    """The real lean result wiring retains direct mgcv prediction parity."""
    if r_bridge.mode != "rpy2":
        pytest.skip("Inference direct-R parity requires rpy2")

    collector = _AssertCollector()
    for name, case in _inference_r_cases().items():
        py_formula, r_formula, data, newdata, py_family, r_family = case
        result = GAM(py_formula, family=py_family).fit(data, result="inference")
        restored = pickle.loads(pickle.dumps(result))
        expected = r_bridge.predict_gam(
            r_formula,
            data,
            newdata,
            family=r_family,
            pred_type="response",
            se_fit=True,
        )
        for label, candidate in (("live", result), ("pickled", restored)):
            pred, se = candidate.predict(newdata, pred_type="response", se_fit=True)
            collector.check(
                f"{name}: {label} prediction",
                lambda a=pred, e=expected["predictions"]: np.testing.assert_allclose(
                    a, e, rtol=MODERATE.rtol, atol=MODERATE.atol
                ),
            )
            collector.check(
                f"{name}: {label} SE",
                lambda a=se, e=expected["se"]: np.testing.assert_allclose(
                    a, e, rtol=MODERATE.rtol, atol=MODERATE.atol
                ),
            )
    collector.raise_if_any("actual inference direct-R parity")


def test_from_fit_owns_independent_family_snapshots_with_final_theta() -> None:
    """Both real result paths snapshot families after fitted theta is stored."""
    data = _generate_family_data("gaussian", n=80)
    model = GAM("y ~ s(x, k=6, bs='cr')", sp=[1.0])
    full_predictor = model.fit(data, result="full").to_predictor()
    inference_predictor = model.fit(data, result="inference").to_predictor()
    registry_family = get_family("gaussian")
    original_link = registry_family.link
    collector = _AssertCollector()
    try:
        registry_family.link = LogLink()
        for label, predictor in (
            ("full", full_predictor),
            ("inference", inference_predictor),
        ):
            collector.check(
                f"{label} snapshot identity",
                lambda p=predictor: np.testing.assert_equal(
                    p.family is registry_family, False
                ),
            )
            collector.check(
                f"{label} snapshot mutation isolation",
                lambda p=predictor: np.testing.assert_equal(
                    type(p.family.link), type(original_link)
                ),
            )
    finally:
        registry_family.link = original_link

    nb_data = _generate_family_data("nb", n=100)
    for mode in ("full", "inference"):
        nb_result = GAM(
            "y ~ s(x, k=6, bs='cr')",
            family=NegativeBinomial(),
            sp=[1.0],
        ).fit(nb_data, result=mode)
        predictor = nb_result.to_predictor()
        collector.check(
            f"{mode} NB snapshot identity",
            lambda p=predictor: np.testing.assert_equal(
                p.family is get_family("nb"), False
            ),
        )
        collector.check(
            f"{mode} NB final theta",
            lambda p=predictor, r=nb_result: np.testing.assert_array_equal(
                p.family.get_theta(transformed=True)[0], r.theta
            ),
        )
    collector.raise_if_any("_from_fit family snapshot")


def test_to_predictor_ownership_lazy_setup_and_memory_reduction() -> None:
    """Extraction is lazy/non-aliasing and inference state is materially leaner."""
    formula, data, newdata = _multi_smooth_case(n=150)
    model = GAM(formula, sp=[1.0, 1.0, 1.0, 1.0])
    full = model.fit(data, result="full")
    collector = _AssertCollector()
    collector.check(
        "lazy cache initially empty",
        lambda: np.testing.assert_equal(full.setup._predict_spec_cache, None),
    )

    setup_graph = _walk_objects(full.setup.coef_map)
    predictor = full.to_predictor()
    predictor_graph = _walk_objects(predictor._predict_spec)
    collector.check(
        "lazy cache populated",
        lambda: np.testing.assert_equal(
            full.setup._predict_spec_cache is predictor._predict_spec, True
        ),
    )
    collector.check(
        "prediction preserved",
        lambda: np.testing.assert_array_equal(
            predictor.predict(newdata, se_fit=True), full.predict(newdata, se_fit=True)
        ),
    )
    collector.check(
        "coefficients owned",
        lambda: np.testing.assert_equal(
            np.shares_memory(predictor.coefficients, full.coefficients), False
        ),
    )
    collector.check(
        "Vp owned",
        lambda: np.testing.assert_equal(np.shares_memory(predictor.Vp, full.Vp), False),
    )
    collector.check(
        "full coefficients still writeable",
        lambda: np.testing.assert_equal(full.coefficients.flags.writeable, True),
    )
    collector.check(
        "full Vp still writeable",
        lambda: np.testing.assert_equal(full.Vp.flags.writeable, True),
    )
    for attr in ("_X", "_S"):
        collector.check(
            f"setup retains {attr}",
            lambda attr=attr: np.testing.assert_equal(
                any(
                    hasattr(obj, attr) and getattr(obj, attr) is not None
                    for obj in setup_graph
                ),
                True,
            ),
        )
        collector.check(
            f"predictor drops {attr}",
            lambda attr=attr: np.testing.assert_equal(
                any(
                    hasattr(obj, attr) and getattr(obj, attr) is not None
                    for obj in predictor_graph
                ),
                False,
            ),
        )
    collector.check(
        "setup retains tensor penalties",
        lambda: np.testing.assert_equal(
            any(
                hasattr(obj, "_penalties") and len(obj._penalties) > 0
                for obj in setup_graph
            ),
            True,
        ),
    )
    collector.check(
        "predictor drops tensor penalties",
        lambda: np.testing.assert_equal(
            any(
                hasattr(obj, "_penalties") and len(obj._penalties) > 0
                for obj in predictor_graph
            ),
            False,
        ),
    )

    for label, memory_formula, memory_data, sp, max_ratio in _memory_cases():
        memory_model = GAM(memory_formula, sp=sp)
        full_memory = memory_model.fit(memory_data, result="full")
        lean_memory = memory_model.fit(memory_data, result="inference")
        full_bytes = _retained_array_bytes(full_memory)
        lean_bytes = _retained_array_bytes(lean_memory)
        collector.check(
            f"{label} retained bytes",
            lambda f=full_bytes, lean=lean_bytes, limit=max_ratio: (
                np.testing.assert_equal(lean < limit * f, True)
            ),
        )
    collector.raise_if_any("to_predictor ownership and retained memory")


def _memory_cases() -> list[tuple[str, str, pd.DataFrame, list[float], float]]:
    rng = np.random.default_rng(SEED + 900)
    n = 220
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    tensor_data = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "y": np.sin(2 * np.pi * x1) + x2 + rng.normal(scale=0.2, size=n),
        }
    )
    x = rng.uniform(size=n)
    gp_data = pd.DataFrame(
        {"x": x, "y": np.sin(2 * np.pi * x) + rng.normal(scale=0.2, size=n)}
    )
    return [
        ("tensor", "y ~ te(x1, x2, k=8)", tensor_data, [1.0, 1.0], 0.25),
        ("GP", "y ~ s(x, bs='gp', k=30)", gp_data, [1.0], 0.45),
    ]
