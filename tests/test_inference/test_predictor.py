"""Behavioral gates for the lean, directly-constructed GAMPredictor."""

from __future__ import annotations

import copy
import dataclasses
import pickle
import warnings
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
import pytest

import jaxgam
import jaxgam.inference as inference
from jaxgam import GAM
from jaxgam.families.registry import get_family
from jaxgam.families.standard import Gaussian
from jaxgam.formula.predict_matrix import build_predict_spec
from jaxgam.inference import GAMPredictor
from jaxgam.links.links import Link, LogLink
from jaxgam.results import _offset_was_nonzero
from tests.helpers import (
    _AssertCollector,
    _generate_family_data,
    _inference_equivalence_cases,
    _inference_r_cases,
    r_available,
)
from tests.tolerances import MODERATE


def _make_predictor(result, *, family=None) -> GAMPredictor:
    """Construct the Commit-D predictor without Commit-E result wiring."""
    return GAMPredictor(
        coefficients=result.coefficients,
        Vp=result.Vp,
        family=copy.deepcopy(result.family if family is None else family),
        formula=result.formula,
        offset_was_nonzero=_offset_was_nonzero(result.setup),
        _predict_spec=build_predict_spec(result.setup),
    )


def test_predict_and_matrix_are_byte_identical_to_full_results() -> None:
    """The isolated predictor matches today's full result across the zoo."""
    collector = _AssertCollector()
    for name, case in _inference_equivalence_cases().items():
        formula, data, newdata, family, sp, train_offset, predict_offset = case
        result = GAM(formula, family=family, sp=sp).fit(data, offset=train_offset)
        predictor = _make_predictor(result)

        collector.check(
            f"{name}: matrix",
            lambda p=predictor, r=result, nd=newdata: np.testing.assert_array_equal(
                p.predict_matrix(nd), r.predict_matrix(nd)
            ),
        )
        for pred_type in ("link", "response"):
            for se_fit in (False, True):

                def assert_prediction(
                    p=predictor,
                    r=result,
                    nd=newdata,
                    pt=pred_type,
                    sf=se_fit,
                    po=predict_offset,
                ) -> None:
                    np.testing.assert_array_equal(
                        p.predict(
                            nd,
                            pred_type=pt,
                            se_fit=sf,
                            offset=po,
                        ),
                        r.predict(
                            nd,
                            pred_type=pt,
                            se_fit=sf,
                            offset=po,
                        ),
                    )

                collector.check(
                    f"{name}: {pred_type}, se_fit={se_fit}",
                    assert_prediction,
                )
    collector.raise_if_any("GAMPredictor equivalence")


@pytest.mark.skipif(not r_available(), reason="R/mgcv not available")
def test_predictor_and_pickled_predictor_match_r(r_bridge) -> None:
    """Prediction and SE retain direct mgcv parity through stdlib pickle."""
    if r_bridge.mode != "rpy2":
        pytest.skip("Inference direct-R parity requires rpy2")

    collector = _AssertCollector()
    for name, case in _inference_r_cases().items():
        py_formula, r_formula, data, newdata, py_family, r_family = case
        result = GAM(py_formula, family=py_family).fit(data)
        predictor = _make_predictor(result)
        restored = pickle.loads(pickle.dumps(predictor))
        r_prediction = r_bridge.predict_gam(
            r_formula,
            data,
            newdata,
            family=r_family,
            pred_type="response",
            se_fit=True,
        )
        for label, candidate in (("live", predictor), ("pickled", restored)):
            pred, se = candidate.predict(newdata, pred_type="response", se_fit=True)
            collector.check(
                f"{name}: {label} prediction",
                lambda a=pred, e=r_prediction["predictions"]: (
                    np.testing.assert_allclose(
                        a, e, rtol=MODERATE.rtol, atol=MODERATE.atol
                    )
                ),
            )
            collector.check(
                f"{name}: {label} SE",
                lambda a=se, e=r_prediction["se"]: np.testing.assert_allclose(
                    a, e, rtol=MODERATE.rtol, atol=MODERATE.atol
                ),
            )
    collector.raise_if_any("direct R predictor parity")


def test_pickle_cloudpickle_and_version_stamp() -> None:
    """Built-ins use pickle; a local link uses cloudpickle; versions warn."""
    data = _generate_family_data("gaussian", n=90)
    newdata = pd.DataFrame({"x": np.linspace(0.05, 0.95, 20)})
    result = GAM("y ~ s(x, k=6, bs='cr')", sp=[1.0]).fit(data)
    predictor = _make_predictor(result)
    expected = predictor.predict(newdata, se_fit=True)
    collector = _AssertCollector()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = pickle.loads(pickle.dumps(predictor))
    collector.check(
        "same-version warning",
        lambda: np.testing.assert_equal(len(caught), 0),
    )
    collector.check(
        "stdlib prediction",
        lambda: np.testing.assert_array_equal(
            restored.predict(newdata, se_fit=True), expected
        ),
    )
    collector.check(
        "stdlib link",
        lambda: np.testing.assert_equal(
            type(restored.family.link), type(predictor.family.link)
        ),
    )

    class LocalIdentityLink(Link):
        def link(self, mu):
            return np.asarray(mu)

        def inverse(self, eta):
            return np.asarray(eta)

        def derivative(self, mu):
            return np.ones_like(mu)

    local_predictor = _make_predictor(result, family=Gaussian(link=LocalIdentityLink()))
    local_expected = local_predictor.predict(newdata, se_fit=True)
    with pytest.raises((AttributeError, pickle.PicklingError)):
        pickle.dumps(local_predictor)
    local_restored = cloudpickle.loads(cloudpickle.dumps(local_predictor))
    collector.check(
        "cloudpickle prediction",
        lambda: np.testing.assert_array_equal(
            local_restored.predict(newdata, se_fit=True), local_expected
        ),
    )

    mismatched = dataclasses.replace(predictor, _jaxgam_version="0.invalid")
    with pytest.warns(UserWarning, match="not a cross-version format"):
        pickle.loads(pickle.dumps(mismatched))
    collector.raise_if_any("predictor serialization")


def test_coefficients_and_covariance_are_owned_and_read_only() -> None:
    """The only two arrays with a freeze guarantee stay frozen on load."""
    data = _generate_family_data("gaussian", n=80)
    result = GAM("y ~ s(x, k=6, bs='cr')", sp=[1.0]).fit(data)
    raw_coefficients = result.coefficients.copy()
    raw_Vp = result.Vp.copy()
    predictor = GAMPredictor(
        raw_coefficients,
        raw_Vp,
        copy.deepcopy(result.family),
        result.formula,
        False,
        build_predict_spec(result.setup),
    )
    restored = pickle.loads(pickle.dumps(predictor))
    collector = _AssertCollector()

    def assert_read_only(array: np.ndarray, index: Any) -> None:
        try:
            array[index] = 0.0
        except ValueError as exc:
            if "read-only" not in str(exc):
                raise AssertionError(f"unexpected write error: {exc}") from exc
        else:
            raise AssertionError("array accepted an in-place write")

    collector.check(
        "coefficients owned",
        lambda: np.testing.assert_equal(
            np.shares_memory(predictor.coefficients, raw_coefficients), False
        ),
    )
    collector.check(
        "Vp owned",
        lambda: np.testing.assert_equal(np.shares_memory(predictor.Vp, raw_Vp), False),
    )
    for label, candidate in (("constructed", predictor), ("restored", restored)):
        collector.check(
            f"{label} coefficients read-only",
            lambda c=candidate: assert_read_only(c.coefficients, 0),
        )
        collector.check(
            f"{label} Vp read-only",
            lambda c=candidate: assert_read_only(c.Vp, (0, 0)),
        )
    collector.raise_if_any("predictor array ownership")


def test_family_snapshot_is_independent_and_keeps_final_nb_theta() -> None:
    """A caller-owned snapshot is independent and contains fitted theta."""
    collector = _AssertCollector()
    data = _generate_family_data("gaussian", n=80)
    result = GAM("y ~ s(x, k=6, bs='cr')", sp=[1.0]).fit(data)
    predictor = _make_predictor(result)
    registry_family = get_family("gaussian")
    original_link = registry_family.link
    try:
        registry_family.link = LogLink()
        collector.check(
            "gaussian snapshot identity",
            lambda: np.testing.assert_equal(predictor.family is registry_family, False),
        )
        collector.check(
            "gaussian snapshot link",
            lambda: np.testing.assert_equal(
                type(predictor.family.link), type(original_link)
            ),
        )
    finally:
        registry_family.link = original_link

    nb_data = _generate_family_data("nb", n=100)
    nb_result = GAM("y ~ s(x, k=6, bs='cr')", family="nb", sp=[1.0]).fit(nb_data)
    nb_predictor = _make_predictor(nb_result)
    collector.check(
        "NB snapshot identity",
        lambda: np.testing.assert_equal(nb_predictor.family is get_family("nb"), False),
    )
    collector.check(
        "NB final theta",
        lambda: np.testing.assert_array_equal(
            nb_predictor.family.get_theta(transformed=True)[0],
            nb_result.theta,
        ),
    )
    collector.raise_if_any("family snapshot")


def test_validation_offset_warning_and_private_surface() -> None:
    """Core validation and the external-offset guard survive extraction."""
    data = _generate_family_data("poisson", n=80)
    result = GAM("y ~ s(x, k=6, bs='cr')", family="poisson", sp=[1.0]).fit(
        data, offset=np.full(len(data), 0.2)
    )
    predictor = _make_predictor(result)
    newdata = pd.DataFrame({"x": np.linspace(0.1, 0.9, 12)})
    collector = _AssertCollector()

    def assert_invalid_pred_type() -> None:
        try:
            predictor.predict(newdata, pred_type="terms")
        except ValueError as exc:
            if "pred_type" not in str(exc):
                raise AssertionError(f"unexpected validation error: {exc}") from exc
        else:
            raise AssertionError("invalid pred_type was accepted")

    collector.check("invalid pred_type", assert_invalid_pred_type)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictor.predict(newdata)
    collector.check(
        "external-offset warning",
        lambda: np.testing.assert_equal(
            any(
                issubclass(w.category, UserWarning)
                and "external offset" in str(w.message)
                for w in caught
            ),
            True,
        ),
    )
    collector.check(
        "predict_core private",
        lambda: np.testing.assert_equal(hasattr(inference, "predict_core"), False),
    )
    collector.check(
        "finish_prediction private",
        lambda: np.testing.assert_equal(hasattr(inference, "finish_prediction"), False),
    )
    collector.check(
        "public inference surface",
        lambda: np.testing.assert_equal(inference.__all__, ["GAMPredictor"]),
    )
    collector.check(
        "version stamp",
        lambda: np.testing.assert_equal(predictor._jaxgam_version, jaxgam.__version__),
    )
    collector.raise_if_any("predictor validation and surface")
