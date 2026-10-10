"""Public fixed-sp family dispatch, ownership and derivative compilation."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM, FitControl, GAMPredictionResult
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution import reml as regular_reml
from jaxgam.families.base import ExponentialFamily
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.pirls import _observed_weights, pirls_loop
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.links.links import IdentityLink
from tests.helpers import nb_optimizer_case_data, r_available
from tests.r_bridge import RBridge
from tests.test_execution.test_regular_starts import _LINKS, _case
from tests.test_fitting.test_family_execution import _UnregisteredQuadraticFamily
from tests.tolerances import STRICT


def _nb_case_frame() -> pd.DataFrame:
    """Generate the original reviewed weighted NB rows without saved arrays."""
    data, weight, offset = nb_optimizer_case_data()
    return data.assign(weight=weight, offset=offset)


@pytest.mark.parametrize("family_class", [Gaussian, Gamma, Poisson, Binomial])
@pytest.mark.parametrize("link", _LINKS)
def test_public_fixed_sp_regular_link_inventory(family_class, link) -> None:
    family, data, weight, offset, _start = _case(family_class, link)
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    model = GAM(
        "y~x",
        family=family,
        sp=[],
        control=FitControl(execution="stream", linear_solver="qr", batch_rows=17),
    )
    if family_class is Poisson and link in {"logit", "probit", "cloglog"}:
        # Pinned gam.fit3 fails at its rowwise y+0.1 mustart for these
        # count/bounded-link inputs; see the R boundary gate below.
        with pytest.raises(ValueError, match="initial predictor"):
            model.fit(source, result="prediction")
        return
    result = model.fit(source, result="prediction")
    assert isinstance(result, GAMPredictionResult)
    assert result.execution_route == "stream_qr"
    assert result.smoothing_params.size == 0
    assert np.all(np.isfinite(result.coefficients))
    assert np.isfinite(result.score)
    assert np.isfinite(result.scale)
    assert result.scale > 0
    assert np.isfinite(result.score_scale)
    assert result.score_scale > 0
    assert result.source_scans > 0
    assert result.batches_scanned >= result.source_scans
    assert 0 < result.known_workspace_bytes <= model.control.memory_budget_bytes
    prediction = result.predict(data[["x"]], offset=offset)
    assert np.all(np.isfinite(prediction))
    assert not any(
        value.shape == (len(data),)
        for value in result.__dict__.values()
        if isinstance(value, np.ndarray)
    )


class _ConformantUnregisteredFamily(_UnregisteredQuadraticFamily):
    """Exercise the contract seam without adding a registry or driver branch."""

    canonical_link_cls = IdentityLink

    def __init__(self) -> None:
        ExponentialFamily.__init__(self, IdentityLink())


def test_public_custom_family_contract_retains_only_snapshot() -> None:
    x = np.linspace(-0.5, 0.5, 31)
    data = pd.DataFrame({"x": x, "y": 0.5 + 0.2 * x + 0.03 * np.sin(7 * x)})
    family = _ConformantUnregisteredFamily()
    result = GAM(
        "y~x",
        family=family,
        sp=[],
        control=FitControl(
            execution="stream",
            linear_solver="qr",
            batch_rows=7,
            uncertainty="fisher",
        ),
    ).fit(DataFrameRowSource(data, response="y"), result="prediction")
    restored = pickle.loads(pickle.dumps(result))
    np.testing.assert_allclose(
        restored.predict(data[["x"]]),
        result.predict(data[["x"]]),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    _, se = restored.predict(data[["x"]], se_fit=True)
    assert np.all(np.isfinite(se))
    assert restored.family is not family


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_fixed_nb_family_snapshot_and_prediction_uncertainty(link: str) -> None:
    saved = _nb_case_frame()
    data = saved[["x", "y"]]
    source = DataFrameRowSource(
        data,
        response="y",
        weights=saved.weight.to_numpy(),
        offset=saved.offset.to_numpy(),
    )
    family = NegativeBinomial(theta=2.7, fixed=True, link=link)
    result = GAM(
        "y~x",
        family=family,
        sp=[],
        control=FitControl(
            execution="stream",
            linear_solver="qr",
            batch_rows=17,
            uncertainty="fisher",
        ),
    ).fit(source, result="prediction")
    original = result.predict(data[["x"]], offset=saved.offset.to_numpy())
    family.put_theta(np.asarray([np.log(1.2)]))
    np.testing.assert_allclose(
        result.predict(data[["x"]], offset=saved.offset.to_numpy()),
        original,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    assert result.family.get_theta(transformed=True)[0] == pytest.approx(2.7)
    assert result.score_scale == 1.0
    assert result.source_scans > 0
    assert result.known_workspace_bytes > 0
    restored = pickle.loads(pickle.dumps(result))
    _, se = restored.predict(data[["x"]], se_fit=True, offset=saved.offset.to_numpy())
    assert np.all(np.isfinite(se))


def test_estimated_nb_zero_penalty_theta_snapshot_and_pickle() -> None:
    saved = _nb_case_frame()
    data = saved[["x", "y"]]
    offset = saved.offset.to_numpy()
    family = NegativeBinomial(theta=2.7, link="log")
    result = GAM(
        "y~x",
        family=family,
        sp=[],
        control=FitControl(
            execution="stream",
            linear_solver="qr",
            batch_rows=17,
            uncertainty="fisher",
        ),
    ).fit(
        DataFrameRowSource(
            data, response="y", weights=saved.weight.to_numpy(), offset=offset
        ),
        result="prediction",
    )
    assert result.theta is not None
    assert result.smoothing_params.size == 0
    assert result.family.get_theta(transformed=True)[0] == pytest.approx(result.theta)
    original = result.predict(data[["x"]], offset=offset)
    family.put_theta(np.asarray([np.log(0.9)]))
    restored = pickle.loads(pickle.dumps(result))
    np.testing.assert_allclose(
        restored.predict(data[["x"]], offset=offset),
        original,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    _, se = restored.predict(data[["x"]], pred_type="link", se_fit=True, offset=offset)
    assert np.all(np.isfinite(se))


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_fixed_nb_zero_weight_observed_derivatives_compile(link: str) -> None:
    x = jnp.linspace(-0.5, 0.5, 19)
    X = jnp.column_stack((jnp.ones_like(x), x))
    y = jnp.asarray(
        [0, 1, 4, 0, 3, 2, 1, 0, 5, 2, 0, 1, 3, 0, 2, 4, 1, 0, 2],
        dtype=jnp.float64,
    )
    weight = jnp.ones_like(y).at[::5].set(0.0)
    family = NegativeBinomial(theta=2.7, fixed=True, link=link)
    beta = jnp.asarray([1.2, 0.1]) if link != "log" else jnp.asarray([0.2, 0.1])
    curvature = jax.jit(
        lambda coefficients: _observed_weights(
            family, X, y, weight, coefficients, jnp.zeros_like(y)
        )
    )(beta)
    assert np.all(np.isfinite(curvature))
    np.testing.assert_allclose(np.asarray(curvature)[::5], 0.0, atol=STRICT.atol)
    compiled = jax.jit(
        lambda initial: pirls_loop(
            X,
            y,
            initial,
            jnp.zeros((2, 2)),
            family,
            wt=weight,
            offset=jnp.zeros_like(y),
        )
    )(beta)
    assert np.isfinite(compiled.deviance)
    assert np.all(np.isfinite(compiled.XtWX))


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_dense_fixed_nb_zero_weight_score_remains_finite(link: str) -> None:
    saved = _nb_case_frame()
    weight = saved.weight.to_numpy(copy=True)
    weight[::17] = 0.0
    result = GAM(
        "y~x", family=NegativeBinomial(theta=2.7, fixed=True, link=link), sp=[]
    ).fit(
        saved[["x", "y"]],
        weights=weight,
        offset=saved.offset.to_numpy(),
        result="prediction",
    )
    assert np.isfinite(result.score)
    assert np.all(np.isfinite(result.coefficients))


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
def test_dense_fixed_nb_zero_weight_deviance_matches_pinned_r(link: str) -> None:
    """Zero-weight AD repair permits a valid dense/source comparison."""
    saved = _nb_case_frame()
    data = saved[["x", "y"]]
    weight = saved.weight.to_numpy(copy=True)
    weight[::17] = 0.0
    offset = saved.offset.to_numpy()
    oracle = RBridge(mode="rpy2").public_fixed_sp_fit(
        data, weight, offset, "nb", link, theta=2.7
    )
    result = GAM(
        "y~x", family=NegativeBinomial(theta=2.7, fixed=True, link=link), sp=[]
    ).fit(data, weights=weight, offset=offset, result="prediction")
    assert np.isfinite(result.score)
    np.testing.assert_allclose(
        result.deviance, oracle["deviance"], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.scale, oracle["scale"], rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_dense_fixed_nb_zero_weight_outer_derivative_jits() -> None:
    """The compiled Newton score/JVP uses smooth direct NB deviance."""
    saved = _nb_case_frame()
    weight = saved.weight.to_numpy(copy=True)
    weight[::17] = 0.0
    result = GAM(
        'y~s(x,bs="cr",k=8)',
        family=NegativeBinomial(theta=2.7, fixed=True, link="log"),
    ).fit(
        saved[["x", "y"]],
        weights=weight,
        offset=saved.offset.to_numpy(),
        result="prediction",
    )
    assert result.converged
    assert np.isfinite(result.score)
    assert np.all(np.isfinite(result.smoothing_params))


def test_regular_zero_weight_observed_derivative_compiles() -> None:
    x = jnp.linspace(-0.4, 0.4, 11)
    X = jnp.column_stack((jnp.ones_like(x), x))
    y = jnp.asarray([0, 1, 2, 0, 1, 3, 0, 2, 1, 0, 1], dtype=jnp.float64)
    weight = jnp.ones_like(y).at[0].set(0.0)
    family = Poisson("identity")
    curvature = jax.jit(
        lambda beta: _observed_weights(family, X, y, weight, beta, jnp.zeros_like(y))
    )(jnp.asarray([1.3, 0.1]))
    assert np.all(np.isfinite(curvature))
    assert float(curvature[0]) == pytest.approx(0.0, abs=STRICT.atol)


def test_regular_fixed_sp_pins_rho_outside_estimated_bounds(monkeypatch) -> None:
    family, data, weight, offset, _ = _case(Gaussian, "log")
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    stream = StreamDesign(
        prepare_model(parse_formula('y~s(x,bs="cr",k=8)'), source, family=family),
        source,
    )
    observed = []

    def evaluate(_stream, _family, params, **_kwargs):
        return SimpleNamespace(
            params=np.asarray(params), source_scans=0, batches_scanned=0, score=0.0
        )

    class CapturedBounds(Exception):
        pass

    def capture(_objective, params, bounds, _control):
        observed.append((np.asarray(params).copy(), tuple(bounds)))
        raise CapturedBounds

    monkeypatch.setattr(regular_reml, "evaluate_regular_stream_reml", evaluate)
    monkeypatch.setattr(regular_reml, "_run_parameterized_stream_reml", capture)
    for pin_lambda in (True, False):
        with pytest.raises(CapturedBounds):
            regular_reml.optimize_regular_stream_reml(
                stream,
                family,
                np.asarray([40.01, 0.0]),
                maximum_bytes=64_000_000,
                pin_lambda=pin_lambda,
            )
    np.testing.assert_array_equal(observed[0][0], [40.01, 0.0])
    assert observed[0][1][0] == (40.01, 40.01)
    np.testing.assert_array_equal(observed[1][0], [40.0, 0.0])
    assert observed[1][1][0] == (-40.0, 40.0)


def test_public_fixed_sp_outside_estimated_bound_is_exact() -> None:
    x = np.linspace(0.0, 1.0, 35)
    data = pd.DataFrame({"x": x, "y": 1.0 + 0.1 * x + 0.02 * np.sin(2.0 * x)})
    pinned = float(np.exp(-40.01))
    result = GAM(
        'y~s(x,bs="cr",k=8)',
        family=Gaussian(),
        sp=[pinned],
        control=FitControl(execution="stream", linear_solver="qr", batch_rows=8),
    ).fit(DataFrameRowSource(data, response="y"), result="prediction")
    np.testing.assert_array_equal(result.smoothing_params, [pinned])
    assert np.isfinite(result.score)


def test_public_fixed_sp_stream_never_builds_training_matrix(monkeypatch) -> None:
    family, data, weight, offset, _ = _case(Gamma, "log")

    def forbidden_build(*_args, **_kwargs):
        raise AssertionError("explicit streamed fitting built full training X")

    monkeypatch.setattr(ModelSetup, "build", forbidden_build)
    result = GAM(
        "y~x",
        family=family,
        sp=[],
        control=FitControl(execution="stream", linear_solver="qr", batch_rows=13),
    ).fit(
        DataFrameRowSource(data, response="y", weights=weight, offset=offset),
        result="prediction",
    )
    assert result.source_scans > 0
    assert np.isfinite(result.score)
