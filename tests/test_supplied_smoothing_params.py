"""Caller-supplied smoothing parameters are reported exactly as given.

Every fitting route works with ``log(sp)``, so a result built from the fit's
own state reports ``exp(log(sp))``. That round trip is inexact for most
decimal inputs and differs across CPUs and math libraries (native x86_64 CI
showed a one-ulp difference that macOS arm64 did not). A supplied smoothing
parameter is an input, so every result builder reports the supplied float64
values through ``jaxgam.results._report_smoothing_params``.

The two values below are fixed as hex literals. For each, the exact
exponential of either double bracketing ``log(x)`` lies at least 1.15 units
in the last place from ``x`` (27.6: 1.59, 0.0862: 1.15), so ``exp(log(x))``
differs from ``x`` on any platform whose ``log`` is faithfully rounded and
whose ``exp`` errs by less than a quarter ulp. The precondition test checks
this for NumPy and JAX on the running host.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.results import _report_smoothing_params

SP_A = float.fromhex("0x1.b99999999999ap+4")  # 27.6
SP_B = float.fromhex("0x1.6113404ea4a8cp-4")  # 0.0862
SUPPLIED = [SP_A, SP_B]
FORMULA = "y ~ s(x, bs='cr', k=8) + s(z, bs='cr', k=6)"


def _frame(family: str, n: int = 160, seed: int = 20261116):
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    eta = 0.6 * np.sin(2 * np.pi * x) + (z - 0.5) ** 2
    weights = rng.uniform(0.5, 1.5, n)
    offset = 0.05 * x
    if family == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta + 0.5 + offset)).astype(float)
    elif family == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
        offset = np.zeros(n)
    elif family == "gamma":
        mu = np.exp(eta + offset)
        y = rng.gamma(2.0, mu / 2.0)
    else:
        mu = np.exp(eta + 1.0 + offset)
        y = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(float)
    return pd.DataFrame({"x": x, "z": z, "y": y}), weights, offset


_FAMILIES = {
    "gaussian": Gaussian,
    "poisson": Poisson,
    "binomial": Binomial,
    "gamma": lambda: Gamma(link="log"),
    "nb_fixed": lambda: NegativeBinomial(3.0, fixed=True),
    "nb_estimated": lambda: NegativeBinomial(3.0),
}


def _data_family(name: str) -> str:
    return "nb" if name.startswith("nb") else name


def _assert_verbatim(result, supplied) -> None:
    expected = np.asarray(supplied, dtype=np.float64)
    assert result.smoothing_params.dtype == np.float64
    np.testing.assert_array_equal(
        result.smoothing_params.view(np.int64), expected.view(np.int64)
    )


def test_supplied_values_do_not_round_trip_on_this_host():
    """Precondition: the chosen inputs exercise an inexact exp(log(sp))."""
    values = np.asarray(SUPPLIED)
    assert np.all(np.exp(np.log(values)) != values)
    assert np.all(np.asarray(jnp.exp(jnp.log(jnp.asarray(values)))) != values)


def test_report_helper_overlays_supplied_values_and_keeps_estimates():
    reported = np.exp(np.log(np.asarray(SUPPLIED)))
    overlaid = _report_smoothing_params(reported, SUPPLIED)
    np.testing.assert_array_equal(
        overlaid.view(np.int64), np.asarray(SUPPLIED).view(np.int64)
    )

    estimated = np.array([0.25, 4.0])
    kept = _report_smoothing_params(estimated, None)
    np.testing.assert_array_equal(kept, estimated)
    assert kept is not estimated

    supplied = np.asarray(SUPPLIED)
    copied = _report_smoothing_params(reported, supplied)
    assert copied is not supplied
    assert not np.shares_memory(copied, supplied)

    scalar = _report_smoothing_params(np.array([np.exp(np.log(SP_A))]), SP_A)
    assert scalar.shape == (1,)
    assert scalar[0] == SP_A

    np.testing.assert_array_equal(_report_smoothing_params(np.empty(0), []), [])
    with pytest.raises(ValueError, match="shape"):
        _report_smoothing_params(reported, [SP_A])


@pytest.mark.parametrize("result_mode", ["full", "inference", "prediction"])
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_dense_fixed_sp_reports_supplied_values(family, result_mode):
    """Known-scale PIRLS and pinned Newton (free scale or theta) routes."""
    frame, weights, offset = _frame(_data_family(family))
    result = GAM(FORMULA, family=_FAMILIES[family](), sp=list(SUPPLIED)).fit(
        frame, weights=weights, offset=offset, result=result_mode
    )
    _assert_verbatim(result, SUPPLIED)


def test_dense_gaussian_qr_newton_reports_supplied_values():
    frame, weights, offset = _frame("gaussian")
    result = GAM(
        FORMULA, sp=list(SUPPLIED), control=FitControl(gaussian_compression=True)
    ).fit(frame, weights=weights, offset=offset)
    assert result.execution_route == "gaussian_qr"
    _assert_verbatim(result, SUPPLIED)


def test_dense_gaussian_qr_lbfgs_reports_supplied_values():
    """More than 50 pinned penalties select the pinned L-BFGS-B route."""
    rng = np.random.default_rng(7)
    columns = {f"v{i}": rng.uniform(0, 1, 300) for i in range(51)}
    frame = pd.DataFrame(columns)
    frame["y"] = sum(0.1 * np.sin(2 * np.pi * frame[c]) for c in columns)
    frame["y"] += rng.normal(0, 0.3, 300)
    formula = "y ~ " + " + ".join(f"s({c}, bs='cr', k=3)" for c in columns)
    supplied = [SP_A] * 51
    result = GAM(
        formula, sp=supplied, control=FitControl(gaussian_compression=True)
    ).fit(frame)
    assert result.lambda_strategy == "fixed"
    _assert_verbatim(result, supplied)


@pytest.mark.parametrize(
    ("family", "solver", "compression"),
    [
        ("gaussian", "cholesky", False),
        ("poisson", "cholesky", False),
        ("binomial", "cholesky", False),
        ("gaussian", "qr", False),
        ("poisson", "qr", False),
        ("gamma", "qr", False),
        ("nb_fixed", "qr", False),
        ("nb_estimated", "qr", False),
        ("gaussian", "cholesky", True),
    ],
)
def test_streamed_fixed_sp_reports_supplied_values(family, solver, compression):
    frame, weights, offset = _frame(_data_family(family))
    source = DataFrameRowSource(frame, response="y", weights=weights, offset=offset)
    result = GAM(
        FORMULA,
        family=_FAMILIES[family](),
        sp=list(SUPPLIED),
        control=FitControl(
            execution="stream",
            linear_solver=solver,
            batch_rows=48,
            uncertainty="fisher",
            gaussian_compression=compression,
        ),
    ).fit(source, result="prediction")
    if compression:
        assert result.execution_route == "stream_gaussian_qr"
    _assert_verbatim(result, SUPPLIED)


@pytest.mark.parametrize("family", ["gaussian", "poisson", "gamma", "nb_estimated"])
def test_discrete_fixed_sp_reports_supplied_values(family):
    frame, weights, offset = _frame(_data_family(family))
    source = DataFrameRowSource(frame, response="y", weights=weights, offset=offset)
    result = GAM(
        FORMULA,
        family=_FAMILIES[family](),
        sp=list(SUPPLIED),
        control=FitControl(
            execution="discrete",
            batch_rows=48,
            discretization=ContinuousDiscretizationControl(),
        ),
    ).fit(source, result="prediction")
    _assert_verbatim(result, SUPPLIED)


def test_single_penalty_scalar_sp_is_reported_verbatim():
    """A scalar sp for a one-penalty model, as in the CI refit."""
    frame, weights, offset = _frame("gamma")
    result = GAM("y ~ s(x, bs='cr', k=8)", family=Gamma(link="log"), sp=SP_A).fit(
        frame, weights=weights, offset=offset
    )
    _assert_verbatim(result, [SP_A])
