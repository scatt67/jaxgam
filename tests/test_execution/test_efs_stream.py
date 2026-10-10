"""Composition, ownership and measured costs of the row-free EFS loop."""

import copy
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.control import EFSControl
from jaxgam.execution import efs_stream, efs_stream_provider
from jaxgam.execution.efs_stream import fit_streamed_efs
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from tests.test_execution.test_efs_stream_start import _regular
from tests.test_execution.test_nb_stream import _fixture


@pytest.fixture(
    scope="module",
    params=["gaussian", "gamma", "poisson", "binomial"]
    + [
        f"nb-{link}-{mode}"
        for link in ("log", "identity", "sqrt")
        for mode in ("fixed", "estimated")
    ],
)
def execution(request):
    name = request.param
    if name.startswith("nb-"):
        _, link, mode = name.split("-")
        stream, family = _fixture(link, estimated=mode == "estimated", smooth=True)
    else:
        family = {
            "gaussian": Gaussian,
            "gamma": Gamma,
            "poisson": Poisson,
            "binomial": Binomial,
        }[name]()
        stream = _regular(family)
    original = copy.deepcopy(family)
    before = stream.source.scans, stream.source.batches
    anchors = []
    actual_fit = efs_stream.NBStreamEFSProvider.__call__

    def capture(provider, request):
        anchors.append(np.array(request.beta_old_init, copy=True))
        return actual_fit(provider, request)

    with patch.object(efs_stream.NBStreamEFSProvider, "__call__", capture):
        result = fit_streamed_efs(
            stream, family, maximum_bytes=10_000_000, batch_rows=11
        )
    measured = stream.source.scans - before[0], stream.source.batches - before[1]
    return result, family, original, measured, name, anchors


def test_full_loop_has_finite_selected_state_and_separate_complete_counts(execution):
    result, _family, _original, measured, _name, _anchors = execution
    fit = result.fit
    assert fit.converged, fit.convergence_info
    assert fit.pirls_result.converged
    assert np.isfinite(
        np.r_[
            fit.pirls_result.coefficients, fit.log_lambda, fit.score, fit.edf, fit.scale
        ]
    ).all()
    assert fit.pirls_result.deviance >= 0
    diagnostics = fit.optimizer_diagnostics
    assert diagnostics.startup_source_scans == result.startup.source_scans == 2
    assert diagnostics.startup_batches_scanned == result.startup.batches_scanned == 16
    assert measured == (
        diagnostics.startup_source_scans + diagnostics.provider_source_scans,
        diagnostics.startup_batches_scanned + diagnostics.provider_batches_scanned,
    )
    assert diagnostics.provider_source_scans > fit.pirls_result.source_scans
    assert diagnostics.outer_iterations == fit.n_iter
    assert len(diagnostics.accepted_score_history) <= EFSControl().history_limit


def test_selected_family_is_isolated_from_caller_and_initial_metadata(execution):
    result, family, original, _measured, name, anchors = execution
    assert result.family is not family
    assert result.family is not result.metadata.family
    assert not hasattr(result.metadata, "X")
    assert not hasattr(result.metadata, "y")
    assert not hasattr(result, "stream")
    assert not hasattr(result, "source")
    if name.startswith("nb-"):
        assert anchors
        for anchor in anchors:
            np.testing.assert_array_equal(anchor, np.zeros(result.metadata.n_coef))
        # The get.null.coef projection is separate from efsudr's zero
        # default recovery anchor, including every fixed-theta refit.
        assert np.any(result.startup.null_coefficients != 0)
        np.testing.assert_array_equal(family.get_theta(), original.get_theta())
        np.testing.assert_array_equal(
            result.metadata.family.get_theta(), original.get_theta()
        )
        if name.endswith("estimated"):
            np.testing.assert_array_equal(
                result.family.get_theta(), np.log([result.fit.theta])
            )
        else:
            np.testing.assert_array_equal(
                result.fit.theta, np.exp(original.get_theta())[0]
            )
            np.testing.assert_array_equal(
                result.family.get_theta(), original.get_theta()
            )
    else:
        assert result.fit.theta is None
    assert not result.startup.log_lambda.flags.writeable
    assert not result.startup.null_coefficients.flags.writeable


def test_selected_factor_actions_remain_jit_compatible(execution):
    result, *_ = execution
    factor = result.fit.pirls_result.fisher_coefficient_factor
    rhs = jnp.arange(result.metadata.n_coef, dtype=jnp.float64)
    actual = jax.jit(lambda f, b: f.hessian_inverse(b))(factor, rhs)
    np.testing.assert_array_equal(actual, factor.hessian_inverse(rhs))


@pytest.mark.parametrize("budget", [0, -1, True, "10000", 1.5])
def test_invalid_budget_rejects_before_provider_or_source(budget):
    family = Poisson()
    stream = _regular(family)
    before = stream.source.scans
    with (
        patch.object(efs_stream.RegularStreamEFSProvider, "create") as create,
        pytest.raises(ValueError, match="positive integer"),
    ):
        fit_streamed_efs(stream, family, maximum_bytes=budget)
    create.assert_not_called()
    assert stream.source.scans == before


def test_startup_retention_preflight_and_control_contract():
    family = Poisson()
    stream = _regular(family)
    before = stream.source.scans
    with patch.object(efs_stream.RegularStreamEFSProvider, "create") as create:
        with pytest.raises(MemoryError, match="startup retention"):
            fit_streamed_efs(stream, family, maximum_bytes=1)
        with pytest.raises(TypeError, match="EFSControl"):
            fit_streamed_efs(stream, family, maximum_bytes=10_000_000, control=object())
    create.assert_not_called()
    assert stream.source.scans == before


def test_parametric_bypass_is_not_claimed_as_an_efs_fit():
    stream, family = _fixture(smooth=False)
    with pytest.raises(ValueError, match="Parametric bypass"):
        fit_streamed_efs(stream, family, maximum_bytes=10_000_000)


@pytest.mark.parametrize("B", [0, -1, True, "11", 1.5])
def test_invalid_batch_rows_reject_before_provider(B):
    stream = _regular(Poisson())
    with (
        patch.object(efs_stream.RegularStreamEFSProvider, "create") as create,
        pytest.raises(ValueError, match="batch_rows"),
    ):
        fit_streamed_efs(stream, Poisson(), maximum_bytes=10_000_000, batch_rows=B)
    create.assert_not_called()


def test_combined_startup_bound_rejects_before_metadata_or_roots():
    family = Poisson()
    stream = _regular(family)
    persistent, _preparation, prior, trace = efs_stream._provider_memory_bounds(stream)
    p, m = stream.prepared.n_coef, stream.prepared.penalties.n_penalties
    retained = 8 * (3 * p + 3 * m + 32)
    history = 128 * (EFSControl().history_limit + 4) + 1024
    workspace = efs_stream._startup_workspace_bytes(stream, 11)
    budget = retained + persistent + prior + trace + history + workspace - 1
    before = stream.source.scans
    with (
        patch.object(efs_stream.RegularStreamEFSProvider, "create") as create,
        pytest.raises(MemoryError, match="startup/provider workspace"),
    ):
        fit_streamed_efs(stream, family, maximum_bytes=budget, batch_rows=11)
    create.assert_not_called()
    assert stream.source.scans == before


def test_outer_iteration_limit_is_not_reported_as_convergence():
    family = Poisson()
    stream = _regular(family)
    before = stream.source.scans, stream.source.batches
    result = fit_streamed_efs(
        stream,
        family,
        maximum_bytes=10_000_000,
        batch_rows=11,
        control=EFSControl(outer_limit=1),
        device=jax.devices()[0],
    )
    assert not result.fit.converged
    assert result.fit.optimizer_diagnostics.stop_reason == "iteration_limit"
    assert result.fit.pirls_result.converged
    assert np.isfinite(result.fit.score)
    diagnostics = result.fit.optimizer_diagnostics
    assert stream.source.scans - before[0] == (
        diagnostics.startup_source_scans + diagnostics.provider_source_scans
    )
    assert stream.source.batches - before[1] == (
        diagnostics.startup_batches_scanned + diagnostics.provider_batches_scanned
    )


def test_regular_default_efs_dispatch_supplies_zero_recovery_anchor_every_refit():
    family = Gaussian()
    stream = _regular(family)
    captured = []
    actual = efs_stream_provider.fit_regular_streamed_pirls

    def capture(*args, **kwargs):
        captured.append(np.array(kwargs["null_coefficients"], copy=True))
        return actual(*args, **kwargs)

    with patch.object(efs_stream_provider, "fit_regular_streamed_pirls", capture):
        result = fit_streamed_efs(
            stream,
            family,
            maximum_bytes=10_000_000,
            batch_rows=11,
            control=EFSControl(outer_limit=4),
        )
    assert captured
    for anchor in captured:
        np.testing.assert_array_equal(anchor, np.zeros(stream.prepared.n_coef))
    assert np.any(result.startup.null_coefficients != 0)


def test_missing_preparation_rejects_without_source_scan():
    from dataclasses import replace

    family = Poisson()
    stream = _regular(family)
    stream = replace(stream, prepared=replace(stream.prepared, fitting=None))
    with pytest.raises(ValueError, match="fitting preparation"):
        fit_streamed_efs(stream, family, maximum_bytes=10_000_000)
