"""Dense known-scale EFS execution adapter tests."""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import pytest

from jaxgam.execution.efs import EFSControl, dense_efs_known_scale
from jaxgam.families.standard import Binomial, Gaussian, Poisson
from tests.helpers import _setup_fd


@pytest.mark.parametrize("family", [Poisson(), Binomial()])
def test_dense_known_scale_efs_runs_from_one_time_shift(family) -> None:
    rng = np.random.default_rng(320)
    x = np.linspace(-1.0, 1.0, 48)
    eta = 0.2 + 0.5 * np.sin(3.0 * x)
    if family.family_name == "poisson":
        y = rng.poisson(np.exp(eta))
    else:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    fd = _setup_fd("y ~ s(x, bs='cr', k=6)", pd.DataFrame({"x": x, "y": y}), family)
    result = dense_efs_known_scale(fd, control=EFSControl(outer_limit=2))
    assert result.n_iter == 2
    assert result.convergence_info == "iteration limit reached"
    assert result.scale == 1.0
    assert result.score_history
    assert np.all(np.isfinite(np.asarray(result.smoothing_params)))
    assert result.update_residual is not None


def test_dense_efs_rejects_out_of_scope_family_before_any_newton_dispatch() -> None:
    x = np.linspace(0.0, 1.0, 20)
    fd = _setup_fd(
        "y ~ s(x, bs='cr', k=5)",
        pd.DataFrame({"x": x, "y": x}),
        Gaussian(),
    )
    with pytest.raises(NotImplementedError, match="Poisson/log"):
        dense_efs_known_scale(fd)


def test_existing_efs_statistics_kernel_compiles_and_executes() -> None:
    # This regression belongs near the execution adapter because every outer
    # proposal consumes its JIT statistics kernel.
    from jaxgam.fitting.efs import EFSStatistics, efs_raw_update

    stats = EFSStatistics(
        jax.numpy.array([1.0]),
        jax.numpy.array([0.25]),
        jax.numpy.array([1.0]),
        jax.numpy.array(True),
        jax.numpy.array(True),
    )
    result = jax.jit(efs_raw_update)(
        jax.numpy.array([0.0]),
        stats,
        jax.numpy.array(1.0),
        jax.numpy.array(1.0),
        jax.numpy.array(15.0),
    )
    assert bool(result.finite_positive)
