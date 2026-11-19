"""Public RowSource route precedence for the opt-in Gaussian QR request.

An explicit ``gaussian_compression=True`` request moves only an eligible exact
Gaussian identity Newton/fixed-sp fit to the frozen Gaussian QR route. Every
other RowSource fit, and an opted-in fit found ineligible, must take the route
it takes without the request, with bitwise-identical numbers, and report the
unmet request in ``execution_fallback_reason``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from tests.helpers import _AssertCollector

_FORMULA = 'y ~ s(x, bs="cr", k=5)'
_FIXED_SP = [0.8]
_NON_GAUSSIAN_IDENTITY = (
    "Gaussian QR requires the exact Gaussian family with identity link"
)

# family label -> (family factory, response generator)
_FAMILIES = {
    "gaussian-identity": (lambda: Gaussian(), "gaussian"),
    "gaussian-log": (lambda: Gaussian("log"), "positive"),
    "poisson-log": (lambda: Poisson(), "count"),
    "binomial-logit": (lambda: Binomial(), "binary"),
    "gamma-log": (lambda: Gamma("log"), "gamma"),
    "nb-fixed-log": (
        lambda: NegativeBinomial(theta=2.7, fixed=True, link="log"),
        "count",
    ),
    "nb-estimated-log": (lambda: NegativeBinomial(theta=2.7, link="log"), "count"),
}

_COMPARED_FIELDS = (
    "coefficients",
    "deviance",
    "score",
    "scale",
    "score_scale",
    "theta",
    "smoothing_params",
    "edf_total",
    "converged",
    "n_iter",
    "convergence_info",
    "lambda_strategy",
    "execution_route",
    "source_scans",
    "batches_scanned",
    "known_workspace_bytes",
)


def _frame(kind: str, n: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(73519)
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    signal = 0.3 + 0.5 * np.sin(3.0 * x)
    mu = np.exp(signal)
    if kind == "gaussian":
        y = signal + rng.normal(scale=0.2, size=n)
    elif kind == "positive":
        y = mu + rng.normal(scale=0.05, size=n)
    elif kind == "count":
        y = rng.poisson(mu).astype(float)
    elif kind == "binary":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-2.0 * signal))).astype(float)
    elif kind == "gamma":
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
    else:  # pragma: no cover - guarded by the table above
        raise ValueError(kind)
    return pd.DataFrame({"x": x, "y": y})


def _fit(
    label, sp, *, compression, linear_solver="qr", optimizer="newton", weight=None
):
    factory, kind = _FAMILIES[label]
    data = _frame(kind)
    source = DataFrameRowSource(data, response="y", weights=weight)
    result = GAM(
        _FORMULA,
        family=factory(),
        sp=sp,
        optimizer=optimizer,
        control=FitControl(
            execution="stream",
            linear_solver=linear_solver,
            batch_rows=23,
            gaussian_compression=compression,
        ),
    ).fit(source, result="prediction")
    return result, data


def _assert_bitwise_same_fit(actual, expected, data, label: str) -> None:
    collector = _AssertCollector()
    for name in _COMPARED_FIELDS:
        collector.check(
            name,
            lambda name=name: np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name)
            ),
        )
    collector.check(
        "prediction",
        lambda: np.testing.assert_array_equal(
            actual.predict(data[["x"]]), expected.predict(data[["x"]])
        ),
    )
    collector.raise_if_any(label)


@pytest.mark.parametrize("sp", [_FIXED_SP, None], ids=["fixed", "estimated"])
@pytest.mark.parametrize("label", list(_FAMILIES))
def test_gaussian_qr_request_selects_route_for_every_rowsource_family(
    label, sp
) -> None:
    ordinary, data = _fit(label, sp, compression=False)
    requested, _ = _fit(label, sp, compression=True)

    published_route = "stream_qr" if sp is not None else "stream_exact_reml_qr"
    published_strategy = "fixed" if sp is not None else "exact_reml"
    assert ordinary.execution_route == published_route
    assert ordinary.lambda_strategy == published_strategy
    assert ordinary.execution_fallback_reason is None

    if label == "gaussian-identity":
        assert requested.execution_route == "stream_gaussian_qr"
        assert requested.lambda_strategy == (
            "fixed" if sp is not None else "newton_reml"
        )
        assert requested.execution_fallback_reason is None
        assert requested.source_scans == 1
        assert requested.batches_scanned == -(-len(data) // 23)
        assert requested.known_workspace_bytes > 0
        assert requested.known_workspace_bytes <= FitControl().memory_budget_bytes
        assert requested.converged
        assert requested.convergence_info
        return

    assert requested.execution_fallback_reason == _NON_GAUSSIAN_IDENTITY
    _assert_bitwise_same_fit(requested, ordinary, data, f"{label} {sp}")


def test_estimated_gaussian_qr_request_does_not_require_qr_coefficient_solver() -> None:
    """The frozen QR route owns its solves; exact REML alone requires QR."""
    requested, _ = _fit(
        "gaussian-identity", None, compression=True, linear_solver="cholesky"
    )
    assert requested.execution_route == "stream_gaussian_qr"
    assert requested.execution_fallback_reason is None
    with pytest.raises(NotImplementedError, match="requires linear_solver='qr'"):
        _fit("gaussian-identity", None, compression=False, linear_solver="cholesky")


@pytest.mark.parametrize(
    ("sp", "linear_solver", "route"),
    [
        (_FIXED_SP, "cholesky", "stream"),
        (_FIXED_SP, "qr", "stream_qr"),
        (None, "qr", "stream_exact_reml_qr"),
    ],
    ids=["fixed-cholesky", "fixed-qr", "estimated-qr"],
)
def test_ineligible_gaussian_qr_weights_take_the_published_route(
    sp, linear_solver, route
) -> None:
    weight = np.ones(64)
    weight[5] = 0.0
    ordinary, data = _fit(
        "gaussian-identity",
        sp,
        compression=False,
        linear_solver=linear_solver,
        weight=weight,
    )
    requested, _ = _fit(
        "gaussian-identity",
        sp,
        compression=True,
        linear_solver=linear_solver,
        weight=weight,
    )
    assert ordinary.execution_route == route
    assert ordinary.execution_fallback_reason is None
    assert "unclipped finite positive weights" in requested.execution_fallback_reason
    _assert_bitwise_same_fit(requested, ordinary, data, f"weights {route}")


def test_ineligible_estimated_request_without_qr_solver_names_both_routes() -> None:
    weight = np.ones(64)
    weight[5] = 0.0
    with pytest.raises(
        NotImplementedError,
        match=r"unclipped finite positive weights.*linear_solver='qr'",
    ):
        _fit(
            "gaussian-identity",
            None,
            compression=True,
            linear_solver="cholesky",
            weight=weight,
        )


@pytest.mark.parametrize("sp", [[], None], ids=["fixed", "estimated"])
def test_unpenalized_gaussian_qr_request_takes_the_published_route(sp) -> None:
    """Provider ineligibility after preflight still yields the published fit."""
    data = _frame("gaussian")
    source = DataFrameRowSource(data, response="y")

    def fit(compression):
        return GAM(
            "y ~ x",
            family=Gaussian(),
            sp=sp,
            control=FitControl(
                execution="stream",
                linear_solver="qr",
                batch_rows=23,
                gaussian_compression=compression,
            ),
        ).fit(source, result="prediction")

    ordinary = fit(False)
    requested = fit(True)
    assert ordinary.execution_route == (
        "stream_qr" if sp is not None else "stream_exact_reml_qr"
    )
    assert "at least one penalty" in requested.execution_fallback_reason
    _assert_bitwise_same_fit(requested, ordinary, data, f"unpenalized {sp}")


def test_estimated_efs_keeps_its_route_and_reports_the_unmet_request() -> None:
    ordinary, data = _fit("gaussian-identity", None, compression=False, optimizer="efs")
    requested, _ = _fit("gaussian-identity", None, compression=True, optimizer="efs")
    assert ordinary.execution_route == "stream_efs_qr"
    assert ordinary.execution_fallback_reason is None
    assert requested.execution_fallback_reason == (
        "Gaussian QR compression requires Newton smoothing selection."
    )
    _assert_bitwise_same_fit(requested, ordinary, data, "efs")
