"""Profile jaxgam GAM fitting pipeline by stage.

Usage::

    uv run python scripts/profile_fit.py

Instruments GAM.fit() to measure time spent in each stage:
- Phase 1: parse_formula() + ModelSetup.build()
- Phase 1→2: FittingData.from_setup() plus device-transfer synchronization
- Newton init: NewtonOptimizer.__init__() (includes _build_differentiable_fns)
- Newton run: NewtonOptimizer.run() with per-iteration breakdown
- Phase 2→3: coefficient transfer only (not complete result materialization)
"""

from __future__ import annotations

import time
from dataclasses import fields, is_dataclass

import numpy as np
import pandas as pd


def _block_result_until_ready(value: object) -> None:
    """Synchronize every device array in a returned optimizer result."""
    import jax

    if hasattr(value, "block_until_ready"):
        jax.block_until_ready(value)
    elif is_dataclass(value):
        for item_field in fields(value):
            _block_result_until_ready(getattr(value, item_field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _block_result_until_ready(item)


# ---------------------------------------------------------------------------
# Data generators (same as benchmark)
# ---------------------------------------------------------------------------


def make_single_data(n: int, family: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x)
    if family == "poisson":
        eta = eta + 0.5
    elif family == "gamma":
        eta = 0.5 * eta + 1.0
    elif family == "binomial":
        eta = 2 * eta

    if family == "gaussian":
        y = eta + rng.normal(0, 0.3, n)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family == "gamma":
        mu = np.exp(eta)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    else:
        raise ValueError(f"Unknown family: {family}")
    return pd.DataFrame({"x": x, "y": y})


# ---------------------------------------------------------------------------
# Profiled fit
# ---------------------------------------------------------------------------


def profile_fit(
    formula: str,
    data: pd.DataFrame,
    family_str: str,
    method: str = "REML",
) -> dict:
    """Run a profiled GAM fit, returning timing breakdown."""
    from jaxgam.families.registry import get_family
    from jaxgam.fitting.data import FittingData
    from jaxgam.fitting.newton import NewtonOptimizer
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    timings = {}

    # Phase 1: parse + build
    t0 = time.perf_counter()
    spec = parse_formula(formula)
    setup = ModelSetup.build(spec, data)
    timings["phase1_setup"] = time.perf_counter() - t0

    # Phase 1→2: FittingData.from_setup plus the actual device transfer.
    family_obj = get_family(family_str)
    t0 = time.perf_counter()
    fd = FittingData.from_setup(setup, family_obj)
    import jax

    jax.block_until_ready(fd.X)
    jax.block_until_ready(fd.y)
    jax.block_until_ready(fd.wt)
    jax.block_until_ready(fd.log_lambda_init)
    if fd.beta_init is not None:
        jax.block_until_ready(fd.beta_init)
    timings["phase1_to_2"] = time.perf_counter() - t0

    # Newton init (includes _build_differentiable_fns for non-Gaussian)
    t0 = time.perf_counter()
    optimizer = NewtonOptimizer(fd, method=method)
    timings["newton_init"] = time.perf_counter() - t0

    # Newton run
    t0 = time.perf_counter()
    result = optimizer.run()
    # Attribute all queued result work to Newton, rather than a later host
    # conversion of whichever field happens to be accessed first.
    _block_result_until_ready(result)
    timings["newton_run"] = time.perf_counter() - t0

    # Phase 2→3 coefficient transfer only. The current adapter owns local
    # transforms; do not reconstruct a removed global reparameterization.
    t0 = time.perf_counter()
    _ = np.asarray(result.pirls_result.coefficients)
    timings["phase2_to_3_coefficients"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    timings["n_iter"] = result.n_iter
    timings["converged"] = result.converged

    return timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    formula = "y ~ s(x, k=10, bs='cr')"
    configs = [
        ("gaussian", 500),
        ("poisson", 500),
        ("binomial", 500),
        ("gamma", 500),
        ("gaussian", 2000),
        ("poisson", 2000),
    ]

    # JIT warmup
    print("JIT warmup...")
    for family, _ in configs:
        data = make_single_data(200, family)
        try:
            profile_fit(formula, data, family)
            print(f"  {family}: OK")
        except Exception as e:
            print(f"  {family}: FAILED ({e})")
    print()

    # Profile
    print(
        f"{'family':<10} {'n':>5} | {'phase1':>8} {'p1→2':>8} {'nwt_init':>8}"
        f" {'nwt_run':>8} {'coef_xfer':>8} | {'total':>8} {'iters':>5}"
    )
    print("-" * 85)

    for family, n in configs:
        data = make_single_data(n, family)

        # Run 3 times, take the median
        all_timings = []
        for _ in range(3):
            t = profile_fit(formula, data, family)
            all_timings.append(t)

        # Pick the run with median total time
        all_timings.sort(key=lambda x: x["total"])
        t = all_timings[1]  # median of 3

        print(
            f"{family:<10} {n:>5} | "
            f"{t['phase1_setup'] * 1000:>7.1f}ms "
            f"{t['phase1_to_2'] * 1000:>7.1f}ms "
            f"{t['newton_init'] * 1000:>7.1f}ms "
            f"{t['newton_run'] * 1000:>7.1f}ms "
            f"{t['phase2_to_3_coefficients'] * 1000:>7.1f}ms | "
            f"{t['total'] * 1000:>7.1f}ms "
            f"{t['n_iter']:>5}"
        )


if __name__ == "__main__":
    main()
