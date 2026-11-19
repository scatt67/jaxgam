"""Stacked working-model and XWXd kernels equal the per-smooth loops they replace.

The proposal kernel and ``discrete_xwxd`` used to unroll Python loops over
smooths and over pairs of smooths into their graphs. They now group blocks by
shape and run one traced loop per group. The per-smooth loops are kept here,
verbatim, as references. Every block or pair keeps its operations and order;
where XLA nevertheless rounds a product differently inside a loop body, the
affected field is compared at STRICT and its measured difference recorded.
"""

from __future__ import annotations

import re
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import ArrayRowSource
from jaxgam.execution.working_model import factor_working_model
from jaxgam.families import Poisson
from jaxgam.fitting import discrete_working, penalty_ops
from jaxgam.fitting import working_model as wm
from jaxgam.fitting.discrete_ops import (
    DeviceDiscreteDesign,
    DeviceLookupBlock,
    DeviceTensorLookupBlock,
    _block_rows,
    _block_stop,
    _subblock_pair_crossproduct,
    _validate_device_batch,
    discrete_xwxd,
    discrete_xwyd,
    lookup_pair_histogram_crossproduct,
    pair_reduction_plan,
    to_device_discrete_design,
)
from jaxgam.formula.continuous_design import build_continuous_discrete_design
from jaxgam.formula.continuous_setup import build_continuous_discrete_setup
from jaxgam.formula.discrete_design import (
    DiscreteDesignControl,
    DiscreteDesignUnavailable,
)
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT

jax.config.update("jax_enable_x64", True)

_N = 160
_STRUCTURES = {
    "one": "y ~ s(x0, bs='cr', k=6)",
    "two_mixed_k": "y ~ s(x0, bs='cr', k=6) + s(x1, bs='cr', k=8)",
    "five_mixed_k": "y ~ "
    + " + ".join(f"s(x{j}, bs='cr', k={k})" for j, k in enumerate((5, 6, 5, 7, 6))),
    "twenty_five": "y ~ "
    + " + ".join(f"s(x{j}, bs='cr', k={5 + j % 2})" for j in range(25)),
    "tensor": "y ~ s(x0, bs='cr', k=5) + te(x1, x2, bs='cr', k=3) "
    "+ s(x3, bs='cr', k=6)",
    "random_effect": "y ~ s(x0, bs='cr', k=5) + s(g, bs='re') + s(x1, bs='cr', k=5)",
    "by_factor": "y ~ g + s(x0, bs='cr', k=5, by=g) + s(x1, bs='cr', k=6)",
    "shared_covariates": "y ~ s(x0, bs='cr', k=5) + te(x0, x1, bs='cr', k=3) "
    "+ s(x1, bs='cr', k=6)",
}


def _columns() -> tuple[dict[str, np.ndarray], np.ndarray]:
    generator = np.random.default_rng(20261117)
    columns: dict[str, np.ndarray] = {
        f"x{j}": generator.uniform(size=_N) for j in range(25)
    }
    columns["g"] = pd.Series(
        pd.Categorical(np.resize(np.array(("a", "b", "c", "d")), _N))
    )
    y = sum(np.sin((j + 2.0) * columns[f"x{j}"]) for j in range(4))
    return columns, y + 0.1 * generator.normal(size=_N)


_BUILT: dict[str, tuple] = {}


def _built(name: str):
    """Working-model metadata and the device design of one structure."""
    if name not in _BUILT:
        columns, y = _columns()
        source = ArrayRowSource(columns, y=y)
        formula = parse_formula(_STRUCTURES[name])
        setup = build_continuous_discrete_setup(
            formula,
            source,
            factor_names=("g",) if re.search(r"\bg\b", _STRUCTURES[name]) else (),
        )
        design = build_continuous_discrete_design(
            setup, source, control=DiscreteDesignControl(batch_rows=_N)
        )
        assert not isinstance(design, DiscreteDesignUnavailable)
        metadata = wm.WorkingModelCriterionMetadata.from_structure(
            setup.penalties,
            _N,
            determinant_structure=setup.source_determinant_penalties,
        )
        setup.cleanup()
        selectors = design.batch_indices(np.arange(_N))
        _BUILT[name] = (metadata, to_device_discrete_design(design), selectors, y)
    return _BUILT[name]


def _equal(collector: _AssertCollector, name: str, observed, expected) -> None:
    def check() -> None:
        for left, right in zip(
            jax.tree_util.tree_leaves(observed),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    collector.check(name, check)


def _close(collector: _AssertCollector, name: str, observed, expected) -> None:
    def check() -> None:
        for left, right in zip(
            jax.tree_util.tree_leaves(observed),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            np.testing.assert_allclose(
                np.asarray(left), np.asarray(right), rtol=STRICT.rtol, atol=STRICT.atol
            )

    collector.check(name, check)


# Reference implementations: the per-smooth loops as they stood before
# stacking (fa19559), kept verbatim apart from names and module prefixes.


def _reference_initial_working_gram(gram, penalties):
    """Sl.initial.repara's initial row/column block transformation of X'WX."""
    transformed = gram
    for block in penalties.blocks:
        C = block.transform.values
        transformed = transformed.at[block.start : block.stop, :].set(
            C.T @ transformed[block.start : block.stop, :]
        )
        transformed = transformed.at[:, block.start : block.stop].set(
            transformed[:, block.start : block.stop] @ C
        )
    return transformed


def _reference_initial_working_rhs(rhs, penalties):
    """Sl.initial.repara's initial block transformation of X'Wz."""
    adjusted = rhs
    for block in penalties.blocks:
        adjusted = adjusted.at[block.start : block.stop].set(
            block.transform.values.T @ adjusted[block.start : block.stop]
        )
    return adjusted


def _reference_hessian_determinant_derivatives(
    inverse: jax.Array,
    penalties: penalty_ops.JaxPenaltyStructure,
    rho: jax.Array,
    smoothing: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """d.detXXS trace contractions on sequential local coefficient blocks.

    Each SPP owns only local rows of the inverse. Pair products select the
    other term's columns; there is no global square array per sp pair.
    """
    first = jnp.zeros_like(rho)
    second = jnp.zeros((rho.size, rho.size), dtype=inverse.dtype)
    for left in penalties.blocks:
        left_rows = inverse[left.start : left.stop, :]
        for left_sp, left_component in zip(
            left.sp_indices, left.penalties, strict=True
        ):
            left_S = penalty_ops._smoothing(
                rho, smoothing, left_sp
            ) * penalty_ops._dense_local(left_component)
            left_product = left_S @ left_rows
            first = first.at[left_sp].add(
                jnp.trace(left_product[:, left.start : left.stop])
            )
            for right in penalties.blocks:
                right_rows = inverse[right.start : right.stop, left.start : left.stop]
                for right_sp, right_component in zip(
                    right.sp_indices, right.penalties, strict=True
                ):
                    right_S = penalty_ops._smoothing(
                        rho, smoothing, right_sp
                    ) * penalty_ops._dense_local(right_component)
                    right_product = right_S @ right_rows
                    value = -jnp.sum(
                        left_product[:, right.start : right.stop].T * right_product
                    )
                    second = second.at[left_sp, right_sp].add(value)
    second = second + jnp.diag(first)
    return first, second


def _reference_add_penalties(penalties, gram, rho, smoothing=None):
    """Sl.addS: add each rounded ``lambda_j S_j`` to the matrix in turn.

    mgcv_madi updates ``A[ind, ind]`` component by component; summing the
    components first rounds differently once one smoothing parameter is
    very large.
    """
    result = gram
    for block in penalties.blocks:
        for sp, component in zip(block.sp_indices, block.penalties, strict=True):
            local = result[block.start : block.stop, block.start : block.stop]
            product = wm._source_rounded(
                penalty_ops._smoothing(rho, smoothing, sp)
                * penalty_ops._dense_local(component)
            )
            result = result.at[block.start : block.stop, block.start : block.stop].set(
                wm._source_rounded(local + product)
            )
    return result


def _reference_xwxd(
    design: DeviceDiscreteDesign,
    selectors: tuple[tuple[jax.Array, ...], ...],
    weights: jax.Array,
) -> jax.Array:
    """Bounded direct indexed ``X.T @ diag(w) @ X`` cross-product."""
    n_rows = _validate_device_batch(design, selectors)
    if weights.shape != (n_rows,):
        raise ValueError("weights do not match selector batch length")
    result = jnp.zeros((design.n_coef, design.n_coef), dtype=weights.dtype)
    for left_number, (left, left_selectors) in enumerate(
        zip(design.blocks, selectors, strict=True)
    ):
        left_slice = slice(left.col_start, _block_stop(left))
        for right_number in range(left_number, len(design.blocks)):
            right = design.blocks[right_number]
            right_selectors = selectors[right_number]
            right_slice = slice(right.col_start, _block_stop(right))
            if left_number == right_number and isinstance(left, DeviceLookupBlock):
                reduced = (
                    jnp.zeros(left.table.shape[0], dtype=weights.dtype)
                    .at[left_selectors[0]]
                    .add(weights)
                )
                part = left.table.T @ (reduced[:, None] * left.table)
            elif isinstance(left, DeviceTensorLookupBlock) or isinstance(
                right, DeviceTensorLookupBlock
            ):
                part = _subblock_pair_crossproduct(
                    design,
                    left,
                    right,
                    left_selectors,
                    right_selectors,
                    weights,
                )
            elif (
                design.pair_policy == "histogram_or_direct"
                and isinstance(left, DeviceLookupBlock)
                and isinstance(right, DeviceLookupBlock)
                and pair_reduction_plan(
                    left.table.shape[0],
                    right.table.shape[0],
                    design.max_pair_histogram_bytes,
                    n_rows=n_rows,
                    k_left=left.table.shape[1],
                    k_right=right.table.shape[1],
                ).method
                == "histogram"
            ):
                part = lookup_pair_histogram_crossproduct(
                    left.table,
                    right.table,
                    left_selectors[0],
                    right_selectors[0],
                    weights,
                )
            else:
                left_rows = _block_rows(left, left_selectors)
                right_rows = _block_rows(right, right_selectors)
                part = left_rows.T @ (weights[:, None] * right_rows)
            result = result.at[left_slice, right_slice].set(part)
            if right_number != left_number:
                result = result.at[right_slice, left_slice].set(part.T)
    return result


_POLICIES = {
    "default": {},
    "direct_only": {"pair_policy": "direct_only"},
    "tight_histogram": {"max_pair_histogram_bytes": 2048},
}


@pytest.mark.parametrize("policy", sorted(_POLICIES))
@pytest.mark.parametrize("name", list(_STRUCTURES))
def test_discrete_xwxd_matches_per_pair_loop(name: str, policy: str) -> None:
    _, device, selectors, _ = _built(name)
    device = replace(device, **_POLICIES[policy])
    weights = jnp.linspace(0.3, 1.7, _N)
    observed = discrete_xwxd(device, selectors, weights)
    expected = jax.jit(_reference_xwxd)(device, selectors, weights)
    # Only the one-column direct pairs (intercept rows against a smooth)
    # differ: XLA rounds that vector-matrix product differently in the
    # unrolled graph than in a loop body (the stacked result is the one the
    # pair gets when formed alone). Measured on macOS arm64 (venv-ob2): at
    # most 3.1e-15 absolute, in 12 to 168 entries of those blocks; every
    # other entry, and every structure without such pairs, bitwise.
    np.testing.assert_allclose(
        np.asarray(observed), np.asarray(expected), rtol=STRICT.rtol, atol=STRICT.atol
    )


@pytest.mark.parametrize("name", list(_STRUCTURES))
def test_discrete_working_batch_gram_matches_per_pair_loop(
    name: str, monkeypatch
) -> None:
    _, device, selectors, y = _built(name)
    p = device.n_coef
    arguments = (
        device,
        selectors,
        jnp.linspace(-0.02, 0.02, p),
        jnp.asarray(np.round(np.abs(y) * 2.0)),
        jnp.linspace(0.5, 1.5, _N),
        jnp.linspace(-0.1, 0.1, _N),
    )
    observed = discrete_working.discrete_working_batch(*arguments, family=Poisson())
    monkeypatch.setattr(discrete_working, "discrete_xwxd", _reference_xwxd)
    batch = discrete_working.discrete_working_batch.__wrapped__
    expected = jax.jit(
        lambda *args, **kwargs: batch(*args, **kwargs),
        static_argnames=("family", "statistics"),
    )(*arguments, family=Poisson())
    # As standalone, only X'WX's one-column direct pairs round differently:
    # measured on macOS arm64 (venv-ob2) at most 4.4e-15 absolute in 14 to
    # 182 entries. Deviance rows, X'Wz and sum(w z^2) are bitwise.
    collector = _AssertCollector()
    _equal(collector, "deviance rows", observed.deviance_rows, expected.deviance_rows)
    _close(collector, "X'WX", observed.gram, expected.gram)
    _equal(collector, "X'Wz", observed.rhs, expected.rhs)
    _equal(
        collector,
        "sum(w z^2)",
        observed.response_square_sum,
        expected.response_square_sum,
    )
    collector.raise_if_any(f"working batch cross-products, {name}")


def _working_inputs(name: str):
    metadata, device, selectors, y = _built(name)
    weights = jnp.linspace(0.6, 1.4, _N)
    gram = discrete_xwxd(device, selectors, weights)
    rhs = discrete_xwyd(device, selectors, weights, jnp.asarray(y))
    yy = jnp.sum(weights * jnp.asarray(y) ** 2)
    m = metadata.penalties.n_penalties
    smoothing = np.geomspace(0.05, 40.0, m)
    log_phi = np.log(0.3)
    return metadata, gram, rhs, yy, smoothing, log_phi


@pytest.mark.parametrize("name", list(_STRUCTURES))
def test_working_model_helpers_match_per_block_loops(name: str) -> None:
    """Each stacked helper, jitted alone, equals its per-block loop bitwise."""
    metadata, gram, rhs, _, smoothing, _ = _working_inputs(name)
    rho = jnp.log(smoothing)
    stacked, blocks = metadata.stacked, metadata.penalties
    factors = factor_working_model(
        wm.working_model_factor_inputs(gram, metadata, jnp.asarray(smoothing))
    )
    H = factors.hessian
    inverse = wm._source_cholesky_solve(
        H.upper, H.pivots, H.diagonal, jnp.eye(len(H.diagonal)), H.rank
    )
    collector = _AssertCollector()
    _equal(
        collector,
        "gram transform",
        jax.jit(wm._initial_working_gram)(gram, stacked),
        jax.jit(_reference_initial_working_gram)(gram, blocks),
    )
    _equal(
        collector,
        "rhs transform",
        jax.jit(wm._initial_working_rhs)(rhs, stacked),
        jax.jit(_reference_initial_working_rhs)(rhs, blocks),
    )
    for label, smooth in (("exp(rho)", None), ("supplied smoothing", smoothing)):
        _equal(
            collector,
            f"Sl.addS ({label})",
            jax.jit(wm._source_add_penalties)(stacked, gram, rho, smooth),
            jax.jit(_reference_add_penalties)(blocks, gram, rho, smooth),
        )
        _equal(
            collector,
            f"d.detXXS ({label})",
            jax.jit(wm._source_hessian_determinant_derivatives)(
                inverse, stacked, rho, smooth
            ),
            jax.jit(_reference_hessian_determinant_derivatives)(
                inverse, blocks, rho, smooth
            ),
        )
    collector.raise_if_any(f"stacked working-model helpers, {name}")


def _reference_kernels(monkeypatch, metadata):
    """The public kernels re-traced with the per-block reference helpers.

    The returned metadata carries the per-block structure where the kernels
    read their stacks, so the references receive it as traced arguments, as
    the per-block kernels did (closing over it would make it constants).
    """
    replacements = {
        "_initial_working_gram": _reference_initial_working_gram,
        "_initial_working_rhs": _reference_initial_working_rhs,
        "_source_add_penalties": _reference_add_penalties,
        "_source_hessian_determinant_derivatives": (
            _reference_hessian_determinant_derivatives
        ),
    }
    for attribute, function in replacements.items():
        monkeypatch.setattr(wm, attribute, function)
    # Fresh wrappers: JAX caches traces per function object, and the public
    # kernels' own traces must not be reused for the reference.
    factor_inputs = wm.working_model_factor_inputs.__wrapped__
    proposal = wm.gaussian_working_model_proposal.__wrapped__
    return (
        replace(metadata, stacked=metadata.penalties),
        jax.jit(lambda *args: factor_inputs(*args)),
        jax.jit(
            lambda *args, **kwargs: proposal(*args, **kwargs),
            static_argnames=("scale_is_fixed", "fixed_smoothing", "debug"),
        ),
    )


@pytest.mark.parametrize("name", list(_STRUCTURES))
def test_proposal_kernels_match_per_block_loops(name: str, monkeypatch) -> None:
    metadata, gram, rhs, yy, smoothing, log_phi = _working_inputs(name)
    smoothing = jnp.asarray(smoothing)
    parameters = jnp.r_[jnp.log(smoothing), log_phi]
    inputs = wm.working_model_factor_inputs(gram, metadata, smoothing)
    factors = factor_working_model(inputs)
    arguments = (parameters, gram, rhs, yy, metadata, factors)
    observed = {
        "unknown scale": wm.gaussian_working_model_proposal(
            *arguments, smoothing=smoothing, tolerance=jnp.asarray(1e-7)
        ),
        "fixed scale": wm.gaussian_working_model_proposal(
            parameters[:-1],
            *arguments[1:],
            smoothing=smoothing,
            scale_is_fixed=True,
            log_scale=jnp.asarray(log_phi),
        ),
    }
    with monkeypatch.context() as patch:
        reference_metadata, reference_inputs, reference_proposal = _reference_kernels(
            patch, metadata
        )
        expected_inputs = reference_inputs(gram, reference_metadata, smoothing)
        reference_arguments = (parameters, gram, rhs, yy, reference_metadata, factors)
        expected = {
            "unknown scale": reference_proposal(
                *reference_arguments,
                smoothing=smoothing,
                tolerance=jnp.asarray(1e-7),
            ),
            "fixed scale": reference_proposal(
                parameters[:-1],
                *reference_arguments[1:],
                smoothing=smoothing,
                scale_is_fixed=True,
                log_scale=jnp.asarray(log_phi),
            ),
        }
    # Factor inputs, coefficients, criterion, gradient, factor and
    # determinants are bitwise equal. Inside the proposal, XLA rounds a few
    # d.detXXS pair contractions in the loop body by one ulp (measured on
    # macOS arm64, venv-ob2: Hessian at most 5.6e-17 absolute, 2.3e-16
    # relative, in 1 to 8 entries); the Newton step inherits it (at most
    # 1.1e-13 absolute, 8.5e-14 relative). Those two fields are STRICT.
    collector = _AssertCollector()
    _equal(collector, "factor inputs", inputs, expected_inputs)
    for label, proposal in observed.items():
        reference = expected[label]
        for field in proposal._fields:
            compare = _close if field in ("hessian", "step") else _equal
            compare(
                collector,
                f"proposal {field} ({label})",
                getattr(proposal, field),
                getattr(reference, field),
            )
    collector.raise_if_any(f"stacked proposal kernels, {name}")
