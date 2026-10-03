"""Provider ownership and source accepted/trial sequences without dense rows."""

from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.control import EFSControl
from jaxgam.execution import efs
from jaxgam.execution.efs_provider import EFSControllerContext, EFSFitRequest
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.fitting.efs import EFSStatistics
from tests.tolerances import STRICT


def _state(request, number, score, phi, *, theta=False, valid=True):
    statistics = EFSStatistics(
        jnp.asarray([2.0]),
        jnp.asarray([2.0 * np.exp(-float(request.log_lambda[0])) - 1.0]),
        jnp.asarray([phi / np.exp(0.02)]),
        jnp.asarray(True),
        jnp.asarray(True),
    )
    return efs.EFSFitState(
        log_lambda=request.log_lambda,
        pirls_result=SimpleNamespace(
            coefficients=jnp.full((2,), float(number)),
            deviance=jnp.asarray(20.0 + number),
            n_iter=3,
            converged=valid,
        ),
        score=jnp.asarray(score),
        edf=jnp.asarray(1.5),
        statistics=statistics,
        raw_update=None,
        valid=valid,
        inner_converged=valid,
        score_phi=request.score_phi,
        update_phi=jnp.asarray(phi),
        reported_phi=jnp.asarray(phi),
        carried_phi=jnp.asarray(phi),
        log_theta=jnp.asarray([0.1 * number]) if theta else None,
        theta_status=jnp.asarray(0) if theta else None,
        theta_n_iter=jnp.asarray(1) if theta else None,
    )


@pytest.mark.parametrize("unknown_scale", [False, True])
@pytest.mark.parametrize("winning_extension", [False, True])
def test_provider_trial_sequences_keep_saved_origin_and_split_phi(
    unknown_scale, winning_extension
):
    """Extension/rejection/contraction use old beta and nuisance state."""
    requests = []
    scores = [10.0, 9.0, 8.0 if winning_extension else 9.1, 11.0, 10.5]
    null = jnp.zeros(2)
    initial = EFSFitRequest(
        jnp.zeros(1),
        null,
        jnp.asarray(0.05) if unknown_scale else None,
        log_theta_start=None if unknown_scale else jnp.asarray([0.9]),
        beta_old_init=null if not unknown_scale else None,
    )
    context = EFSControllerContext(
        estimated_theta=not unknown_scale,
        reference_profile="source-provider-sequence",
        trace_method="exact-provider-fisher",
        source_fingerprint="immutable-source",
        basis_fingerprint="immutable-coordinates",
    )

    def provider(request):
        requests.append(request)
        number = len(requests)
        return _state(
            request,
            number,
            scores[number - 1],
            phi=0.1 * (number + 1) if unknown_scale else 1.0,
            theta=not unknown_scale,
        )

    run = efs._run_efs_unknown_scale if unknown_scale else efs._run_efs_known_scale
    result = run(provider, context, initial, EFSControl(outer_limit=2))
    assert not result.converged
    assert result.convergence_info == "iteration limit reached"
    assert requests[0] is initial
    np.testing.assert_array_equal(requests[1].beta_start, [1.0, 1.0])
    np.testing.assert_array_equal(requests[2].beta_start, [1.0, 1.0])
    selected = 3 if winning_extension else 2
    np.testing.assert_array_equal(requests[3].beta_start, [selected, selected])
    if winning_extension:
        assert len(requests) == 5
        np.testing.assert_array_equal(requests[4].beta_start, [selected, selected])
        np.testing.assert_allclose(
            requests[4].log_lambda - requests[3].log_lambda,
            [-0.02],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    else:
        assert len(requests) == 4
    if unknown_scale:
        assert [float(r.score_phi) for r in requests] == pytest.approx(
            [0.05, 0.2, 0.2, 0.3] + ([0.3] if winning_extension else [])
        )
        assert result.score_phi_history == pytest.approx((0.2, 0.3))
    else:
        assert [float(r.log_theta_start[0]) for r in requests] == pytest.approx(
            [0.9, 0.1, 0.1, 0.1 * selected]
            + ([0.1 * selected] if winning_extension else [])
        )
        assert all(r.beta_old_init is null for r in requests)
    assert result.score_history == (
        scores[2] if winning_extension else 9.0,
        scores[-1] if winning_extension else 11.0,
    )
    assert result.optimizer_diagnostics.reference_profile == context.reference_profile
    assert result.optimizer_diagnostics.trace_method == context.trace_method
    assert result.optimizer_diagnostics.inner_iterations == 3 * len(requests)
    assert result.optimizer_diagnostics.theta_iterations == (
        0 if unknown_scale else len(requests)
    )


@pytest.mark.parametrize("unknown_scale", [False, True])
@pytest.mark.parametrize("invalid_initial", [False, True])
def test_provider_failures_retain_last_accepted_state(unknown_scale, invalid_initial):
    requests = []
    initial = EFSFitRequest(jnp.zeros(1), jnp.zeros(2), jnp.asarray(0.05))

    def provider(request):
        requests.append(request)
        return _state(
            request,
            len(requests),
            10.0,
            phi=1.0,
            valid=not invalid_initial and len(requests) == 1,
        )

    run = efs._run_efs_unknown_scale if unknown_scale else efs._run_efs_known_scale
    result = run(provider, EFSControllerContext(fixed_theta=2.7), initial, EFSControl())
    assert not result.converged
    assert result.convergence_info == "inner_failure"
    assert len(requests) == (1 if invalid_initial else 2)
    assert result.n_iter == (0 if invalid_initial else 1)
    np.testing.assert_array_equal(result.pirls_result.coefficients, [1.0, 1.0])
    if not unknown_scale:
        assert result.theta == 2.7
    assert result.optimizer_diagnostics.invalid_fit_seen


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"regular_start_present": False},
        {"score_phi": jnp.asarray(0.05)},
        {"log_theta_start": jnp.asarray([0.1]), "beta_old_init": jnp.zeros(2)},
        {
            "log_theta_start": jnp.asarray([0.1]),
            "beta_old_init": jnp.zeros(2),
            "start_is_absent": True,
        },
    ],
)
def test_dense_provider_preserves_positional_and_keyword_call_shapes(
    monkeypatch, extra
):
    calls = []
    sentinel = object()
    fd, plan, control = object(), object(), EFSControl()

    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(efs, "_fit_state", fit)
    request = EFSFitRequest(jnp.zeros(1), jnp.zeros(2), **extra)
    assert efs._dense_fit_provider(fd, plan, control)(request) is sentinel
    args, kwargs = calls[0]
    assert args[:2] == (fd, plan)
    assert args[2] is request.log_lambda
    assert args[3] is request.beta_start
    assert args[4] is control
    assert len(args) == (6 if "score_phi" in extra else 5)
    if len(args) == 6:
        assert args[5] is request.score_phi
    assert set(kwargs) == set(extra) - {"score_phi"}
    for name, value in kwargs.items():
        assert value is extra[name]


def test_controller_context_and_request_are_small_immutable_state():
    request = EFSFitRequest(jnp.zeros(1), jnp.zeros(2))
    context = EFSControllerContext(
        source_fingerprint="source", basis_fingerprint="basis"
    )
    with pytest.raises(FrozenInstanceError):
        request.score_phi = jnp.asarray(1.0)
    with pytest.raises(FrozenInstanceError):
        context.fixed_theta = 1.0
    assert all(
        not hasattr(getattr(context, field.name), "shape") for field in fields(context)
    )
    assert set(vars(request)) == {
        "log_lambda",
        "beta_start",
        "score_phi",
        "log_theta_start",
        "beta_old_init",
        "start_is_absent",
        "regular_start_present",
    }


@pytest.mark.parametrize("unknown_scale", [False, True])
def test_dense_adapter_and_row_independent_provider_sequences_are_byte_equal(
    monkeypatch, unknown_scale
):
    """The same five source trials produce identical requests and results."""
    null = jnp.zeros(2)
    control = EFSControl(outer_limit=2)
    initial = EFSFitRequest(
        jnp.zeros(1), null, jnp.asarray(0.05) if unknown_scale else None
    )
    calls = [[], []]

    def produce(request, path):
        calls[path].append(request)
        number = len(calls[path])
        return _state(
            request,
            number,
            [10.0, 9.0, 8.0, 11.0, 10.5][number - 1],
            phi=0.1 * (number + 1) if unknown_scale else 1.0,
        )

    run = efs._run_efs_unknown_scale if unknown_scale else efs._run_efs_known_scale
    direct = run(
        lambda request: produce(request, 0), EFSControllerContext(), initial, control
    )
    family = Gaussian() if unknown_scale else Poisson()
    fd = SimpleNamespace(
        family=family,
        n_penalties=1,
        n_obs=1,
        n_coef=2,
        rank_deficit=0,
        wt=jnp.ones(1),
        y=jnp.full(1, 2.0),
        X=jnp.ones((1, 2)),
        log_lambda_init=jnp.full(1, -2.5),
        beta_init=null,
    )
    monkeypatch.setattr(efs, "prepare_efs_statistics", lambda _fd: object())

    def dense_fit(_fd, _plan, rho, beta, _control, *phi, **kwargs):
        return produce(EFSFitRequest(rho, beta, *phi, **kwargs), 1)

    monkeypatch.setattr(efs, "_fit_state", dense_fit)
    if unknown_scale:
        dense = efs.dense_efs_unknown_scale(
            fd,
            beta_init=null,
            initial_log_scale=jnp.log(jnp.asarray(0.05)),
            control=control,
        )
    else:
        dense = efs.dense_efs_known_scale(fd, beta_init=null, control=control)
    assert len(calls[0]) == len(calls[1]) == 5
    for left, right in zip(calls[0], calls[1], strict=True):
        for field in fields(left):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, jnp.ndarray):
                assert np.asarray(a).dtype == np.asarray(b).dtype
                assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
            else:
                assert a == b
    for field in fields(direct):
        a, b = getattr(direct, field.name), getattr(dense, field.name)
        if field.name == "pirls_result":
            for name in vars(a):
                assert (
                    np.asarray(getattr(a, name)).tobytes()
                    == np.asarray(getattr(b, name)).tobytes()
                )
        elif isinstance(a, jnp.ndarray):
            assert np.asarray(a).dtype == np.asarray(b).dtype
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
        else:
            assert a == b
