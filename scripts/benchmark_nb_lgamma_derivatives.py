"""Reproducible CPU benchmark for NB lgamma theta derivatives.

Run ``make benchmark-nb-lgamma``. It reports cold compile and synchronized
warm gradient/Hessian timings, varying row count and the maximum count
independently; ``--outlier`` adds one high-count observation.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.families.negative_binomial import _lgamma_diff_planned


def _legacy_derivative(theta: jax.Array, y: jax.Array, max_y: int) -> jax.Array:
    def body(k, acc):
        return acc - jnp.where(k < y, 1.0 / (theta + k), 0.0)

    return jax.lax.fori_loop(0, max_y, body, jnp.zeros_like(y))


def _seconds(fn, *args) -> float:
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - start


def _measure(n: int, max_y: int, outlier: bool, repeats: int) -> dict[str, float | int]:
    rng = np.random.default_rng(20260907)
    y_np = rng.integers(0, max_y + 1, size=n, dtype=np.int32)
    if outlier:
        y_np[-1] = max_y
    y = jnp.asarray(y_np, dtype=jnp.float64)
    indices = jnp.asarray(y_np)
    theta = jnp.array(2.0)
    prefix = jax.jit(
        jax.hessian(lambda t: jnp.sum(_lgamma_diff_planned(t, y, indices, max_y, True)))
    )
    legacy = jax.jit(jax.jacfwd(lambda t: jnp.sum(_legacy_derivative(t, y, max_y))))
    cold_prefix = _seconds(prefix, theta)
    cold_legacy = _seconds(legacy, theta)
    warm_prefix = np.median([_seconds(prefix, theta) for _ in range(repeats)])
    warm_legacy = np.median([_seconds(legacy, theta) for _ in range(repeats)])
    return {
        "n": n,
        "max_y": max_y,
        "outlier": int(outlier),
        "prefix_table_bytes": (max_y + 1) * 8,
        "cold_prefix_seconds": cold_prefix,
        "cold_legacy_seconds": cold_legacy,
        "warm_prefix_seconds": float(warm_prefix),
        "warm_legacy_seconds": float(warm_legacy),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1_000, 100_000])
    parser.add_argument("--max-counts", type=int, nargs="+", default=[16, 256])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outlier", action="store_true")
    args = parser.parse_args()
    results = [
        _measure(n, max_y, args.outlier, args.repeats)
        for n in args.rows
        for max_y in args.max_counts
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
