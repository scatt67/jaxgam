"""Thin-plate knot choice against pinned ``smooth.construct.tp.smooth.spec``.

With more data than ``xt$max.knots`` (default 2000), smooth.construct.tp takes
``uniquecombs`` of the centered covariates and, if more unique rows remain
than the cap, samples ``max.knots`` of them under ``temp.seed(xt$seed)``
(default 1); ``tprs_setup`` sorts the supplied knots (``Xd_strip``). All data
are generated at runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.smooths.tprs import TPRSSmooth
from jaxgam.smooths.utils import radial_knot_options
from tests.helpers import (
    _AssertCollector,
    make_smooth_spec,
    r_available,
    r_links_pinned_openblas,
)
from tests.tolerances import STRICT


def test_knot_options_read_xt_and_the_direct_thin_plate_spelling() -> None:
    assert radial_knot_options(make_smooth_spec(["x"])) == (2000, 1)
    assert radial_knot_options(make_smooth_spec(["x"], max_knots=20, seed=3)) == (
        20,
        3,
    )
    assert radial_knot_options(
        make_smooth_spec(["x"], xt={"max.knots": 30, "seed": 5})
    ) == (30, 5)
    assert radial_knot_options(
        make_smooth_spec(["x"], max_knots=30, xt={"max_knots": 30})
    ) == (30, 1)
    # Direct arguments are thin-plate only; GP reads xt like mgcv.
    assert radial_knot_options(
        make_smooth_spec(["x"], bs="gp", max_knots=30, xt={"seed": 4})
    ) == (2000, 4)
    with pytest.raises(ValueError, match="different values"):
        radial_knot_options(make_smooth_spec(["x"], seed=2, xt={"seed": 3}))
    with pytest.raises(ValueError, match=r"max\.knots"):
        radial_knot_options(
            make_smooth_spec(["x"], xt={"max_knots": 3, "max.knots": 4})
        )
    with pytest.raises(TypeError, match="xt"):
        radial_knot_options(make_smooth_spec(["x"], xt=[1]))
    assert radial_knot_options(make_smooth_spec(["x"], xt=None)) == (2000, 1)


def _cases() -> dict[str, tuple[pd.DataFrame, list[str], int, dict, str]]:
    """(data, variables, k, Python options, R ``xt``) for each knot case.

    Values of 10 or more and negative values make ``uniquecombs``'s label
    order differ from the numeric one; duplicated rows exercise its
    first-row deduplication; a cap equal to the unique-row count (with more
    data than the cap) does not sample, one below does.
    """
    rng = np.random.default_rng(20261117)
    base = pd.DataFrame({"x": rng.uniform(-15, 25, 45), "z": rng.normal(0, 8, 45)})
    duplicated = pd.concat([base, base, base, base.iloc[[6]]], ignore_index=True)
    sixty = pd.DataFrame({"x": rng.uniform(-15, 25, 60), "z": rng.normal(0, 8, 60)})
    capped = pd.concat([sixty, sixty.iloc[:20]], ignore_index=True)
    return {
        "one/direct seed 4": (
            pd.DataFrame({"x": rng.uniform(-3, 12, 150)}),
            ["x"],
            8,
            {"max_knots": 40, "seed": 4},
            "list(max.knots=40,seed=4)",
        ),
        "two/xt seed 9": (
            pd.DataFrame({"x": rng.uniform(-15, 25, 120), "z": rng.normal(0, 8, 120)}),
            ["x", "z"],
            12,
            {"xt": {"max_knots": 35, "seed": 9}},
            "list(max.knots=35,seed=9)",
        ),
        "two/duplicated": (
            duplicated,
            ["x", "z"],
            10,
            {"xt": {"max.knots": 30, "seed": 2}},
            "list(max.knots=30,seed=2)",
        ),
        "two/cap=nu": (capped, ["x", "z"], 10, {"max_knots": 60}, "list(max.knots=60)"),
        "two/cap=nu-1": (
            capped,
            ["x", "z"],
            10,
            {"max_knots": 59},
            "list(max.knots=59)",
        ),
        # More than the default cap of 2000 unique rows, with no xt.
        "one/default cap": (
            pd.DataFrame({"x": rng.uniform(-3, 12, 2050)}),
            ["x"],
            10,
            {},
            "",
        ),
    }


@pytest.mark.platform_sensitive
@pytest.mark.skipif(not r_available(), reason="R+mgcv not available")
def test_knots_basis_penalty_and_prediction_match_pinned_smoothcon(r_bridge) -> None:
    """Knots bitwise; X, S and PredictMat bitwise on pinned R, else STRICT."""
    if r_bridge.mode != "rpy2":
        pytest.skip("Raw thin-plate basis oracle requires rpy2.")
    pinned = r_links_pinned_openblas(r_bridge)
    rng = np.random.default_rng(41)
    collector = _AssertCollector()

    def compare(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
        if pinned:
            collector.check(
                f"{label} bitwise",
                lambda: np.testing.assert_array_equal(actual, expected),
            )
        else:
            collector.check(
                f"{label} STRICT",
                lambda: np.testing.assert_allclose(
                    actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )

    for label, (data, variables, k, options, xt) in _cases().items():
        smooth = TPRSSmooth(make_smooth_spec(variables, k=k, **options))
        smooth.setup({name: data[name].to_numpy() for name in variables})
        term = f"s({','.join(variables)},bs='tp',k={k}" + (f",xt={xt})" if xt else ")")
        construct = r_bridge.smooth_construct(term, data)
        collector.check(
            f"{label}/knots bitwise",
            lambda smooth=smooth, expected=construct["Xu"]: (
                np.testing.assert_array_equal(smooth._Xu, expected)
            ),
        )
        new = pd.DataFrame(
            {
                name: rng.uniform(data[name].min(), data[name].max(), 25)
                for name in variables
            }
        )
        expected = r_bridge.source_smooth_basis(term, data, new)
        compare(f"{label}/X", smooth.training_design_matrix(), expected["X"])
        compare(f"{label}/S", smooth.build_penalty_matrices()[0].S, expected["S"][0])
        compare(
            f"{label}/PredictMat",
            smooth.predict_matrix({name: new[name].to_numpy() for name in variables}),
            expected["prediction"],
        )
    collector.raise_if_any("thin-plate knots vs pinned smoothCon")


@pytest.mark.platform_sensitive
@pytest.mark.skipif(not r_available(), reason="R+mgcv not available")
def test_factor_by_smooth_samples_the_same_knots_as_pinned_smoothcon(r_bridge) -> None:
    """A factor-by thin-plate term constructs one base basis on all rows."""
    if r_bridge.mode != "rpy2":
        pytest.skip("Raw thin-plate basis oracle requires rpy2.")
    rng = np.random.default_rng(43)
    n = 150
    data = pd.DataFrame(
        {
            "x": rng.uniform(-15, 25, n),
            "z": rng.normal(0, 8, n),
            "f": pd.Categorical(np.array(["a", "b", "c"])[np.arange(n) % 3]),
            "y": rng.normal(size=n),
        }
    )
    setup = ModelSetup.build(
        parse_formula("y~0+s(x,z,bs='tp',k=10,by=f,xt={'max_knots':40,'seed':6})"),
        data,
    )
    by_smooth = next(
        term.smooth for term in setup.coef_map.terms if term.term_type == "smooth"
    )
    expected = r_bridge.smooth_construct(
        "s(x,z,bs='tp',k=10,by=f,xt=list(max.knots=40,seed=6))",
        data,
        absorb_cons=True,
    )
    np.testing.assert_array_equal(by_smooth.base_smooth._Xu, expected["Xu"])
    width = expected["X"].shape[1]
    np.testing.assert_allclose(
        setup.X[:, :width], expected["X"], rtol=STRICT.rtol, atol=STRICT.atol
    )
