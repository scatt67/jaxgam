"""Pinned mgcv thin-plate parity across covariate layouts and ranks.

Runtime-generated regular, irregular, clustered and shifted covariates cover
``s(x,bs='tp',k)`` and ``ti(x,z,bs='tp',k)`` for ``k`` in 4, 6 and 8. Dense
and bounded operator setup share tprs_setup's construction, so their raw
smooth bases must be bitwise identical; both are compared with pinned
``smoothCon`` (raw basis and summed penalty) and with a pinned fixed-sp fit.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.operator_fit import fit_operator_fixed_sp
from jaxgam.families.standard import Gaussian
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator_prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from jaxgam.smooths.tensor import TensorInteractionSmooth
from jaxgam.smooths.tprs import TPRSSmooth
from tests.helpers import _AssertCollector, check_that
from tests.tolerances import STRICT

_LAYOUTS = ("regular", "irregular", "clustered", "shifted")
_RANKS = (4, 6, 8)
_CASES = [
    (layout, k, kind) for layout in _LAYOUTS for k in _RANKS for kind in ("s", "ti")
]
# Rlanczos reduces through R's linked BLAS. These two ti bases amplify that
# rounding through the margin reparameterization (cond(Xp) ~ 1e3): they are
# bitwise equal to R linked to the pinned OpenBLAS 0.3.26, but not to R linked
# to another BLAS. Their raw basis is asserted only against the pinned build;
# the penalty and fixed-sp fit are asserted everywhere.
_PINNED_BLAS_BASES = {("irregular", 8, "ti"), ("clustered", 6, "ti")}


def _frame(layout: str) -> pd.DataFrame:
    rng = np.random.default_rng(92613)
    regular = np.linspace(-1.3, 1.7, 61)
    irregular = np.sort(rng.uniform(-1.7, 2.4, 61))
    clustered = np.sort(np.r_[rng.normal(-1, 0.05, 30), rng.normal(1.2, 0.08, 31)])
    x = {
        "regular": regular,
        "irregular": irregular,
        "clustered": clustered,
        "shifted": regular + 111.3,
    }[layout]
    z = np.sin(np.linspace(-0.8, 2.2, 61)) + np.linspace(0, 0.1, 61)
    return pd.DataFrame({"x": x, "z": z, "y": np.cos(1.7 * x) + 0.2 * z})


def _expressions(kind: str, k: int) -> tuple[str, str]:
    if kind == "s":
        expression = f"s(x,bs='tp',k={k})"
        return expression, expression
    return f"ti(x,z,bs='tp',k={k})", f"ti(x,z,bs='tp',k=c({k},{k}))"


def _links_pinned_openblas(r_bridge) -> bool:
    name = os.path.basename(r_bridge.linked_lapack())
    return "openblas" in name and "0.3.26" in name


def _operator_rows(formula: str, data: pd.DataFrame):
    source = DataFrameRowSource(data, response="y")
    model = prepare_operator_model(
        parse_formula(formula),
        source,
        control=OperatorPreparationControl(batch_rows=7),
    )
    rows = np.vstack(
        [model.evaluate_batch(batch).modest_values for batch in source.scan(7)]
    )
    return source, model, rows


def _dense_raw_basis(kind: str, python: str, data: pd.DataFrame) -> np.ndarray:
    spec = parse_formula("y~0+" + python).smooth_terms[0]
    smooth = TPRSSmooth(spec) if kind == "s" else TensorInteractionSmooth(spec)
    smooth.setup({name: data[name].to_numpy() for name in spec.variables})
    return smooth._X


@pytest.mark.parametrize(("layout", "k", "kind"), _CASES)
def test_tprs_basis_penalty_and_fit_match_pinned_source(
    r_bridge, layout: str, k: int, kind: str
) -> None:
    data = _frame(layout)
    python, r_expression = _expressions(kind, k)
    expected = r_bridge.smooth_construct(r_expression, data, absorb_cons=True)
    expected_penalty = sum(expected["S"])
    dense = ModelSetup.build(parse_formula("y~0+" + python), data)
    _, model, rows = _operator_rows("y~0+" + python, data)
    collector = _AssertCollector()

    operator_smooth = model.modest.training_spec.coef_map.terms[-1].smooth
    columns = {name: data[name].to_numpy() for name in ("x", "z")}
    collector.check(
        "dense and operator raw bases are identical",
        lambda: np.testing.assert_array_equal(
            operator_smooth.training_matrix(columns),
            _dense_raw_basis(kind, python, data),
        ),
    )
    if (layout, k, kind) not in _PINNED_BLAS_BASES or _links_pinned_openblas(r_bridge):
        for name, actual in (("dense", dense.X), ("operator", rows)):
            collector.check(
                f"{name} basis",
                lambda actual=actual: np.testing.assert_allclose(
                    actual, expected["X"], rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
    for name, actual in (
        ("dense", dense.penalties.materialize()),
        ("operator", model.penalties.materialize()),
    ):
        collector.check(
            f"{name} penalty",
            lambda actual=actual: np.testing.assert_allclose(
                actual, expected_penalty, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )

    sp = np.full(2 if kind == "ti" else 1, 0.7)
    reference = r_bridge.fit_fixed_sp("y~" + r_expression, data, "gaussian", sp)
    source, fit_model, fit_rows = _operator_rows("y~" + python, data)
    fitted = fit_operator_fixed_sp(
        fit_model, source, Gaussian(), sp, memory_budget_bytes=128 * 1024 * 1024
    )
    collector.check(
        "fixed-sp converged",
        lambda: check_that(
            bool(fitted.converged) and reference["converged"], "fit did not converge"
        ),
    )
    collector.check(
        "fixed-sp fitted means",
        lambda: np.testing.assert_allclose(
            fit_rows @ fitted.coefficients,
            reference["fitted_values"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "fixed-sp deviance",
        lambda: np.testing.assert_allclose(
            fitted.deviance, reference["deviance"], rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.raise_if_any(f"{layout} {python}")


@pytest.mark.parametrize(
    ("label", "k"),
    [("tied", 8), ("pure-knot", 6)],
)
def test_tprs_tied_and_pure_knot_bases_match_pinned_source(
    r_bridge, label: str, k: int
) -> None:
    """tprs_setup maps tied data to knot rows; k unique rows skip Lanczos."""
    rng = np.random.default_rng(4)
    base = np.round(rng.uniform(-1.0, 2.0, 25), 2)
    x = (
        np.r_[base, base[:20], base[:7]]
        if label == "tied"
        else np.repeat(np.linspace(0.1, 1.3, k), 5)
    )
    data = pd.DataFrame({"x": x, "z": np.zeros_like(x), "y": np.cos(x)})
    expression = f"s(x,bs='tp',k={k})"
    raw = r_bridge.smooth_construct(expression, data, absorb_cons=False)
    expected = r_bridge.smooth_construct(expression, data, absorb_cons=True)
    smooth = TPRSSmooth(parse_formula("y~0+" + expression).smooth_terms[0])
    smooth.setup({"x": x})
    dense = ModelSetup.build(parse_formula("y~0+" + expression), data)
    _, model, rows = _operator_rows("y~0+" + expression, data)
    collector = _AssertCollector()
    for name, actual, reference in (
        ("raw basis", smooth._X, raw["X"]),
        ("raw penalty", smooth._S, raw["S"][0]),
        ("UZ", smooth._UZ, raw["UZ"]),
        ("dense basis", dense.X, expected["X"]),
        ("operator basis", rows, expected["X"]),
        ("dense penalty", dense.penalties.materialize(), sum(expected["S"])),
        ("operator penalty", model.penalties.materialize(), sum(expected["S"])),
    ):
        collector.check(
            name,
            lambda actual=actual, reference=reference: np.testing.assert_allclose(
                actual, reference, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any(f"{label} {expression}")


def test_tprs_bivariate_knot_rows_and_training_guards(r_bridge) -> None:
    """Joint knot rows serve d=2 training replay; other rows are refused."""
    data = _frame("irregular")
    expression = "s(x,z,bs='tp',k=10)"
    expected = r_bridge.smooth_construct(expression, data, absorb_cons=True)
    dense = ModelSetup.build(parse_formula("y~0+" + expression), data)
    _, model, rows = _operator_rows("y~0+" + expression, data)
    operator_smooth = model.modest.training_spec.coef_map.terms[-1].smooth
    columns = {name: data[name].to_numpy() for name in ("x", "z")}
    collector = _AssertCollector()
    collector.check(
        "dense and operator raw bases are identical",
        lambda: np.testing.assert_array_equal(
            operator_smooth.training_matrix(columns),
            _dense_raw_basis("s", expression, data),
        ),
    )
    for name, actual in (("dense", dense.X), ("operator", rows)):
        collector.check(
            f"{name} basis",
            lambda actual=actual: np.testing.assert_allclose(
                actual, expected["X"], rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.check(
        "operator penalty",
        lambda: np.testing.assert_allclose(
            model.penalties.materialize(),
            sum(expected["S"]),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any(expression)
    shifted = {"x": columns["x"][:3] + 0.125, "z": columns["z"][:3]}
    with pytest.raises(ValueError, match="coincide with the source knots"):
        operator_smooth.training_matrix(shifted)
    univariate = _operator_rows("y~0+s(x,bs='tp',k=6)", data)[1]
    margin = univariate.modest.training_spec.coef_map.terms[-1].smooth
    with pytest.raises(ValueError, match="coincide with the source knots"):
        margin.training_matrix({"x": np.array([data.x.max() + 1.0])})
    prediction = model.modest.predict_spec.coef_map.terms[-1].smooth
    with pytest.raises(RuntimeError, match="copy_for_training"):
        prediction.training_matrix(columns)
