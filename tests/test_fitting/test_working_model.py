"""Owning Sl.fitChol proposal parity and row-free JIT gates."""

from __future__ import annotations

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.working_model import (
    evaluate_gaussian_working_model_proposal,
    factor_working_model,
    source_exponentials,
)
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.working_model import (
    WorkingModelCriterionMetadata,
    WorkingModelFactor,
    _double_double_sum,
    _sequential_double_sum,
    _sl_newton_step,
    _source_penalty_determinant,
    _source_tcrossprod,
    _x87_sum,
    gaussian_working_model_proposal,
    source_sum,
    working_model_factor_inputs,
)
from jaxgam.formula.continuous_setup import build_continuous_discrete_setup
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.working_penalties import (
    build_working_penalty_blocks,
    source_root_product,
)
from jaxgam.penalties.structure import (
    IdentityPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
from jaxgam.smooths.source_lapack import _OPENBLAS_PREFIX, scipy_uses_accelerate_lapack
from tests.helpers import _AssertCollector, check_that
from tests.tolerances import MODERATE, STRICT

# JaxGAM's own Sl.setup roots equal pinned R's physical penalty operators
# but not its rounding (see the physical-coordinate root-product test). At
# sp=[1e-10, 100] the near-singular penalty determinant amplifies that to
# ~1.4e-4 in log|S| and its derivatives. Four source-arithmetic passes (source
# block coordinates; source pivoted factor and IFT; ordered BLAS contractions
# and preconditioning; pinned DPTSV/DGEMM/DLANGE plus determinant-only
# coordinates), then source-order projection (8447840), frozen pcrossprod
# root products (64f14fa), R-exact exp(rho) input and the Sl.addS addition
# order, left those end-to-end values within MODERATE only.
# Second derivatives, Hessians and Newton steps there are not reproducible
# even R-vs-R (one-ULP rho: 44.6x, 22.4x and 180x MODERATE), so they are
# gated against R's exact state by
# test_extreme_proposal_matches_pinned_r_at_matched_state instead.


def _frame():
    position = np.arange(97, dtype=float)
    x = (position + 0.2 * np.sin(position * 0.7)) / len(position)
    z = np.cos(position * 0.23) + 0.07 * np.sin(position * 0.91)
    y = 0.4 - 0.3 * z + np.sin(4.5 * x) + 0.05 * np.cos(position)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def _compact_working_model(param):
    frame = _frame()
    formula, smoothing = {
        "single": ("y ~ z + s(x,bs='cr',k=6)", np.array([0.55])),
        "separate": (
            "y ~ s(x,bs='cr',k=6)+s(z,bs='cr',k=6)",
            np.array([0.55, 0.7]),
        ),
        "coupled": ("y ~ te(x,z,bs='cr',k=3)", np.array([0.35, 0.7])),
        "coupled_extreme": (
            "y ~ te(x,z,bs='cr',k=3)",
            np.array([1e-10, 100.0]),
        ),
        "coupled_extreme_zero": (
            "y ~ te(x,z,bs='cr',k=3)",
            np.array([1e-10, 100.0]),
        ),
    }[param]
    setup = build_continuous_discrete_setup(
        parse_formula(formula),
        DataFrameRowSource(frame, response="y"),
        m=30,
    )
    snapped = {
        name: values[group.index.selector]
        for group in setup.compact.groups
        for name, values in zip(group.names, group.index.values, strict=True)
    }
    X = setup.predict_spec.build_predict_matrix(pd.DataFrame(snapped))
    metadata = WorkingModelCriterionMetadata.from_structure(
        setup.penalties,
        len(frame),
        determinant_structure=setup.source_determinant_penalties,
    )
    setup.cleanup()
    weights = np.linspace(0.6, 1.4, len(frame))
    if param == "coupled_extreme_zero":
        weights[0] = 0.0
    offset = np.linspace(-0.1, 0.2, len(frame))
    return frame, formula, smoothing, X, metadata, weights, offset


@pytest.fixture(params=["single", "separate", "coupled"])
def compact_working_model(request):
    return _compact_working_model(request.param)


@pytest.fixture(params=["coupled_extreme", "coupled_extreme_zero"])
def extreme_working_model(request):
    return _compact_working_model(request.param)


def _statistics(X, y, weights, offset, batch_rows):
    gram = np.zeros((X.shape[1], X.shape[1]))
    rhs = np.zeros(X.shape[1])
    yy = 0.0
    for start in range(0, len(y), batch_rows):
        stop = start + batch_rows
        block = X[start:stop]
        response = y[start:stop] - offset[start:stop]
        weight = weights[start:stop]
        gram += block.T @ (weight[:, None] * block)
        rhs += block.T @ (weight * response)
        yy += float(np.sum(weight * response**2))
    return jnp.asarray(gram), jnp.asarray(rhs), jnp.asarray(yy)


def source_working_metadata(state) -> WorkingModelCriterionMetadata:
    """Working metadata that holds pinned ``Sl.setup``'s exact local arrays.

    ``state`` statistics are already in ``Sl.initial.repara`` coordinates, so
    every block transform is the identity. Penalties are R's ``Sl$S`` at its
    ``ind`` coordinates; determinant roots/products are R's ``rS`` and
    ``pcrossprod(rS)``. Coefficients are then reported in R's coordinates.
    """
    blocks = []
    singleton_indices, singleton_ranks = [], []
    multi_indices, multi_ranks, multi_products, multi_roots = [], [], [], []
    sp = 0
    for block in state.blocks:
        assert block.reparameterized
        size = block.stop - block.start
        count = len(block.penalties)
        indices = tuple(range(sp, sp + count))
        sp += count
        if count == 1:
            local = (np.diag(block.penalized.astype(float)),)
            singleton_indices.append(indices[0])
            singleton_ranks.append(block.rank)
        else:
            local = []
            for penalty in block.penalties:
                padded = np.zeros((size, size))
                padded[np.ix_(block.penalized, block.penalized)] = penalty
                local.append(padded)
            multi_indices.append(indices)
            multi_ranks.append(block.rank)
            multi_products.append(tuple(map(jnp.asarray, block.root_products)))
            multi_roots.append(tuple(map(jnp.asarray, block.roots)))
        blocks.append(
            penalty_ops.JaxPenaltyBlock(
                block.start,
                block.stop,
                indices,
                tuple(
                    penalty_ops.JaxLocalPenalty("dense", jnp.asarray(S), size)
                    for S in local
                ),
                penalty_ops.JaxTransform("dense", jnp.eye(size), size),
            )
        )
    return WorkingModelCriterionMetadata(
        penalty_ops.JaxPenaltyStructure(len(state.gram), tuple(blocks)),
        state.n_obs,
        state.null_dimension,
        tuple(singleton_indices),
        tuple(singleton_ranks),
        jnp.zeros(len(singleton_indices)),
        tuple(multi_indices),
        tuple(multi_ranks),
        tuple(multi_products),
        tuple(multi_roots),
    )


def penalty_determinant(rho, metadata, smoothing):
    """ldetS from host DPSTRF factors of the same exponentials' ``S``."""
    n_coef = metadata.penalties.n_coef
    factors = factor_working_model(
        working_model_factor_inputs(jnp.eye(n_coef), metadata, jnp.asarray(smoothing))
    )
    return jax.jit(
        lambda values, exponentials, blocks: _source_penalty_determinant(
            values, metadata, blocks, exponentials
        )
    )(jnp.asarray(rho), jnp.asarray(smoothing), factors.penalties)


def ldet_tolerance(r_bridge):
    """ldetS bound for pinned R's own LAPACK provider.

    ldetSblock needs only DPSTRF, DSYRK, DTRSM and sum(). When R's LAPACK is
    the pinned OpenBLAS 0.3.26 build, the host DPSTRF runs the same code and
    reproduces R's factor bit for bit, so ldetS and its derivatives agree to
    R's long-double sum() rounding. R's macOS build instead runs libRlapack
    DPSTF2 over Accelerate BLAS, which no pinned provider reproduces; its
    remaining factor rounding is amplified by near-singular penalty blocks.
    """
    if r_bridge.source_linear_algebra().lapack_is_openblas(_OPENBLAS_PREFIX):
        return STRICT
    return MODERATE


def _exact_residual_norm(matrix, step, rhs):
    """``||matrix @ step - rhs||_2`` with the residual formed exactly."""
    total = Fraction(0)
    for row, target in zip(matrix, rhs, strict=True):
        value = -Fraction(float(target))
        for entry, component in zip(row, step, strict=True):
            value += Fraction(float(entry)) * Fraction(float(component))
        total += value * value
    return float(np.sqrt(float(total)))


def _newton_system(gradient, hessian, step, tolerance):
    """The (active, |eigen|-floored, capped) system a Sl.fitChol step solves.

    On the active coordinates the source step is ``-alpha V diag(1/mu) V' g``
    with ``mu = max(|eigenvalue|, floor)`` and ``alpha`` its four-cap. It
    solves ``Htilde s = -alpha g`` with ``Htilde = V diag(mu) V'``, which is
    the lower-triangle Hessian itself when no eigenvalue is modified. A capped
    step's ``alpha`` is its least-squares scale in that system.
    """
    gradient = np.asarray(gradient, dtype=float)
    hessian = np.asarray(hessian, dtype=float)
    step = np.asarray(step, dtype=float)
    active = np.flatnonzero(
        (np.abs(gradient) > tolerance) | (np.abs(np.diag(hessian)) > tolerance)
    )
    lower = np.tril(hessian[np.ix_(active, active)])
    symmetric = lower + np.tril(lower, -1).T
    values, vectors = np.linalg.eigh(symmetric)
    floor = np.max(np.abs(values)) * np.sqrt(np.finfo(float).eps)
    scales = np.maximum(np.abs(values), floor)
    modified = bool(np.any(values < 0.0) or np.any(np.abs(values) < floor))
    system = vectors @ np.diag(scales) @ vectors.T if modified else symmetric
    g = gradient[active]
    uncapped = -(vectors @ ((vectors.T @ g) / scales))
    capped = bool(np.max(np.abs(uncapped)) > 4.0)
    s = step[active]
    alpha = -float(g @ (system @ s)) / float(g @ g) if capped else 1.0
    return active, system, -alpha * g, s, capped


def check_newton_step(collector, label, observed, expected, tolerance):
    """Conditioning-aware step gates for one matched trial.

    ``observed``/``expected`` are JAX/R ``(gradient, hessian, step)``. (a)
    JAX's step is a backward-stable solution of JAX's own system. (b) The two
    steps obey the exact first-order perturbation identity
    ``||s_J - s_R|| <= ||H_R^-1|| (||H_R - H_J|| ||s_J|| + ||g_R - g_J||)``
    plus each solve's rounding, ``STRICT.rtol cond(H_R) ||s||``. Returns the
    backward error and the bound's slack (difference over bound).
    """
    active_j, system_j, rhs_j, step_j, capped_j = _newton_system(*observed, tolerance)
    active_r, system_r, rhs_r, step_r, capped_r = _newton_system(*expected, tolerance)
    full_step = np.asarray(observed[2], dtype=float)
    collector.check(
        f"{label} Newton active set and cap",
        lambda: check_that(
            np.array_equal(active_j, active_r)
            and capped_j == capped_r
            and not np.any(np.delete(full_step, active_j)),
            f"active {active_j.tolist()} vs {active_r.tolist()}, "
            f"capped {capped_j} vs {capped_r}",
        ),
    )
    if not len(active_j) or not np.array_equal(active_j, active_r):
        return None
    backward = _exact_residual_norm(system_j, step_j, rhs_j) / (
        np.linalg.norm(system_j, 2) * np.linalg.norm(step_j) + np.linalg.norm(rhs_j)
    )
    collector.check(
        f"{label} Newton backward error",
        lambda: check_that(
            backward <= STRICT.rtol, f"JAX step backward error {backward:.3e}"
        ),
    )
    singular = np.linalg.svd(system_r, compute_uv=False)
    lhs = np.linalg.norm(step_j - step_r)
    rhs = (
        np.linalg.norm(system_r - system_j, 2) * np.linalg.norm(step_j)
        + np.linalg.norm(rhs_r - rhs_j)
    ) / singular[-1] + STRICT.rtol * (singular[0] / singular[-1]) * (
        np.linalg.norm(step_r) + np.linalg.norm(step_j)
    )
    collector.check(
        f"{label} Newton perturbation bound",
        lambda: check_that(lhs <= rhs, f"step difference {lhs:.3e} > bound {rhs:.3e}"),
    )
    return backward, lhs / rhs if rhs > 0 else 0.0


@pytest.mark.platform_sensitive
def test_proposal_matches_pinned_sl_fitchol(r_bridge, compact_working_model):
    _assert_proposal_matches_pinned_sl_fitchol(r_bridge, compact_working_model)


@pytest.mark.platform_sensitive
def test_extreme_proposal_matches_pinned_sl_fitchol(r_bridge, extreme_working_model):
    _assert_proposal_matches_pinned_sl_fitchol(
        r_bridge, extreme_working_model, extreme=True
    )


def _assert_proposal_matches_pinned_sl_fitchol(
    r_bridge, compact_working_model, *, extreme=False
):
    frame, formula, smoothing, X, metadata, weights, offset = compact_working_model
    # See the module note: extreme criterion/gradient are MODERATE after the
    # documented passes; their Hessian/step belong to the matched-state gate.
    derivative = MODERATE if extreme else STRICT
    collector = _AssertCollector()
    for fixed in (False, True):
        parameters = np.log(smoothing) if fixed else np.r_[np.log(smoothing), -3.0]
        statistics = _statistics(X, frame.y.to_numpy(), weights, offset, 13)
        result = evaluate_gaussian_working_model_proposal(
            parameters,
            *statistics,
            metadata,
            scale_is_fixed=fixed,
            log_scale=-3.0,
        )
        expected = r_bridge.source_bam_sl_fitchol_proposal(
            formula,
            frame,
            smoothing,
            weights,
            offset,
            fixed_scale=fixed,
        )
        fields = [
            ("identical design", X, expected["X"], STRICT),
            ("coefficients", result.coefficients, expected["beta"], STRICT),
            ("criterion", result.criterion, expected["criterion"][0], derivative),
            ("gradient", result.gradient, expected["grad"], derivative),
        ]
        if not extreme:
            fields += [
                ("hessian", result.hessian, expected["hess"], STRICT),
                ("Newton proposal", result.step, expected["step"], STRICT),
            ]
        for name, observed, target, tolerance in fields:
            collector.check(
                f"fixed_scale={fixed} {name}",
                lambda observed=observed, target=target, tolerance=tolerance: (
                    np.testing.assert_allclose(
                        observed,
                        np.asarray(target),
                        rtol=tolerance.rtol,
                        atol=tolerance.atol,
                    )
                ),
            )
    collector.raise_if_any(formula)


@pytest.mark.platform_sensitive
def test_extreme_proposal_matches_pinned_r_at_matched_state(
    r_bridge, extreme_working_model
):
    """Ill-conditioned second derivatives and steps from R's exact state.

    R's own roots, root products, reparameterized moments, rho, log(phi) and
    exp(rho) enter the kernels and the host DPSTRF, so this gates the
    determinant, implicit-derivative and Newton arithmetic without JaxGAM's
    setup rounding. Gradient and Hessian are MODERATE: their trace terms run
    through R's own BLAS (Accelerate or OpenBLAS GEMM, long-double sums) on a
    penalized Hessian with condition up to 2.7e7, the four documented source
    passes notwithstanding. Steps use the conditioning-aware gates.
    """
    frame, formula, smoothing, _, _, weights, offset = extreme_working_model
    ldet = ldet_tolerance(r_bridge)
    collector = _AssertCollector()
    for fixed in (False, True):
        evaluation = r_bridge.source_bam_sl_fitchol_state(
            formula, frame, np.log(smoothing), weights, offset, fixed_scale=fixed
        )
        state, proposal = evaluation.state, evaluation.proposal
        metadata = source_working_metadata(state)
        determinant = penalty_determinant(
            proposal.log_smoothing, metadata, proposal.smoothing
        )
        parameters = (
            proposal.log_smoothing
            if fixed
            else np.r_[proposal.log_smoothing, proposal.log_scale]
        )
        result = evaluate_gaussian_working_model_proposal(
            parameters,
            jnp.asarray(state.gram),
            jnp.asarray(state.rhs),
            jnp.asarray(state.response_square_sum),
            metadata,
            scale_is_fixed=fixed,
            log_scale=proposal.log_scale,
            tolerance=proposal.tolerance,
            smoothing=proposal.smoothing,
        )
        for name, observed, target, tolerance in (
            ("ldetS", determinant[0], proposal.logdet_penalty, ldet),
            ("ldetS first derivative", determinant[1], evaluation.penalty_first, ldet),
            (
                "ldetS second derivative",
                determinant[2],
                evaluation.penalty_second,
                ldet,
            ),
            ("coefficients", result.coefficients, proposal.coefficients, STRICT),
            ("gradient", result.gradient, proposal.gradient, MODERATE),
            ("hessian", result.hessian, proposal.hessian, MODERATE),
            ("ldetXXS", result.logdet_hessian, proposal.logdet_hessian, STRICT),
            ("kernel ldetS", result.logdet_penalty, proposal.logdet_penalty, ldet),
        ):
            collector.check(
                f"fixed_scale={fixed} {name}",
                lambda observed=observed, target=target, tolerance=tolerance: (
                    np.testing.assert_allclose(
                        observed, target, rtol=tolerance.rtol, atol=tolerance.atol
                    )
                ),
            )
        check_newton_step(
            collector,
            f"fixed_scale={fixed}",
            (result.gradient, result.hessian, result.step),
            (proposal.gradient, proposal.hessian, proposal.step),
            proposal.tolerance,
        )
    collector.raise_if_any(f"{formula} at pinned matched state")


def test_proposal_jit_batch_invariance_and_row_free_graph(compact_working_model):
    frame, _, smoothing, X, metadata, weights, offset = compact_working_model
    parameters = jnp.asarray(np.r_[np.log(smoothing), -3.0])
    collector = _AssertCollector()
    baseline = None
    lam = jnp.asarray(source_exponentials(np.log(smoothing)))
    for batch_rows in (1, 13, 97):
        statistics = _statistics(X, frame.y.to_numpy(), weights, offset, batch_rows)
        inputs = jax.jit(working_model_factor_inputs)(statistics[0], metadata, lam)
        factors = factor_working_model(inputs)
        observed = jax.jit(gaussian_working_model_proposal)(
            parameters, *statistics, metadata, factors, smoothing=lam
        )
        if baseline is None:
            baseline = observed
        for field in observed._fields:
            collector.check(
                f"B={batch_rows} {field}",
                lambda field=field, observed=observed, baseline=baseline: (
                    np.testing.assert_allclose(
                        getattr(observed, field),
                        getattr(baseline, field),
                        rtol=STRICT.rtol,
                        atol=STRICT.atol,
                    )
                ),
            )
    collector.raise_if_any("working statistics batch invariance")
    graphs = (
        jax.make_jaxpr(working_model_factor_inputs)(statistics[0], metadata, lam),
        jax.make_jaxpr(
            lambda *args: gaussian_working_model_proposal(*args, smoothing=lam)
        )(parameters, *statistics, metadata, factors),
    )

    def check_graph(closed):
        graph = closed.jaxpr if hasattr(closed, "jaxpr") else closed
        for variable in (*graph.constvars, *graph.invars, *graph.outvars):
            assert len(frame) not in getattr(variable.aval, "shape", ())
        for equation in graph.eqns:
            for variable in equation.outvars:
                assert len(frame) not in getattr(variable.aval, "shape", ())
            for value in equation.params.values():
                if hasattr(value, "jaxpr"):
                    check_graph(value)
                elif isinstance(value, (tuple, list)):
                    for child in value:
                        if hasattr(child, "jaxpr"):
                            check_graph(child)

    for graph in graphs:
        check_graph(graph)


def test_extreme_proposal_physical_fields_are_batch_invariant(extreme_working_model):
    frame, _, smoothing, X, metadata, weights, offset = extreme_working_model
    parameters = jnp.asarray(np.r_[np.log(smoothing), -3.0])
    baseline = None
    collector = _AssertCollector()
    for batch_rows in (1, 13, 97):
        result = evaluate_gaussian_working_model_proposal(
            parameters,
            *_statistics(X, frame.y.to_numpy(), weights, offset, batch_rows),
            metadata,
        )
        if baseline is None:
            baseline = result
        for field in (
            "coefficients",
            "criterion",
            "gradient",
            "hessian",
            "step",
            "factor_rank",
            "factor_diagonal",
            "logdet_hessian",
            "logdet_penalty",
        ):
            collector.check(
                f"B={batch_rows} {field}",
                lambda field=field, result=result, baseline=baseline: (
                    np.testing.assert_allclose(
                        getattr(result, field),
                        getattr(baseline, field),
                        rtol=STRICT.rtol,
                        atol=STRICT.atol,
                    )
                ),
            )
    collector.raise_if_any("extreme working statistics batch invariance")


@pytest.mark.platform_sensitive
def test_extreme_penalty_determinant_matches_pinned_r(r_bridge, extreme_working_model):
    frame, formula, smoothing, _, metadata, _, _ = extreme_working_model
    rho = np.log(smoothing)
    expected = r_bridge.source_bam_penalty_determinant(formula, frame, rho)
    actual = penalty_determinant(rho, metadata, source_exponentials(rho))
    collector = _AssertCollector()
    # See the module note: MODERATE value/first derivative from JaxGAM's own
    # roots; the second derivative is gated at R's exact matched state.
    for name, observed, target in zip(
        ("value", "first"),
        actual[:2],
        (expected["value"], expected["first"]),
        strict=True,
    ):
        collector.check(
            name,
            lambda observed=observed, target=target: np.testing.assert_allclose(
                np.asarray(observed),
                np.asarray(target),
                rtol=MODERATE.rtol,
                atol=MODERATE.atol,
            ),
        )
    collector.raise_if_any("extreme source penalty determinant")


@pytest.mark.platform_sensitive
def test_source_newton_active_filter_indefinite_floor_and_step_cap():
    gradient = jnp.array([3.0, -1.0, 1e-8])
    hessian = jnp.array([[2.0, 0.0, 10.0], [0.0, -0.5, 10.0], [10.0, 10.0, 1e-10]])
    result = jax.jit(_sl_newton_step)(gradient, hessian, jnp.asarray(1e-6))
    np.testing.assert_allclose(
        result, [-1.5, 2.0, 0.0], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_array_equal(
        _sl_newton_step(jnp.zeros(2), jnp.zeros((2, 2)), 1e-6),
        np.zeros(2),
    )
    result = _sl_newton_step(
        jnp.array([0.0, 1.0]), jnp.diag(jnp.array([1.0, 0.0])), 0.0
    )
    np.testing.assert_allclose(result, [0.0, -4.0], rtol=STRICT.rtol, atol=STRICT.atol)


@pytest.fixture
def simple_metadata():
    structure = PenaltyStructure(
        2,
        (
            PenaltyBlock(
                0,
                2,
                (0,),
                (IdentityPenalty(2),),
                IdentityTransform(2),
                (2,),
            ),
        ),
    )
    return WorkingModelCriterionMetadata.from_structure(structure, 12)


def test_fixed_scale_no_penalties_and_debug(capsys, simple_metadata):
    metadata = WorkingModelCriterionMetadata.from_structure(PenaltyStructure(2, ()), 12)
    result = evaluate_gaussian_working_model_proposal(
        np.empty(0),
        jnp.eye(2),
        jnp.array([0.2, -0.3]),
        jnp.asarray(3.0),
        metadata,
        scale_is_fixed=True,
        log_scale=-3.0,
        debug=True,
    )
    result.coefficients.block_until_ready()
    np.testing.assert_allclose(
        result.coefficients, [0.2, -0.3], rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert result.gradient.shape == result.step.shape == (0,)
    assert result.hessian.shape == (0, 0)
    assert "Gaussian working model criterion=" in capsys.readouterr().out
    leaves, definition = jax.tree_util.tree_flatten(simple_metadata)
    restored = jax.tree_util.tree_unflatten(definition, leaves)
    assert restored.n_obs == 12
    assert restored.null_dimension == 0


@pytest.mark.parametrize("n_obs", [0, True, 2.5])
def test_metadata_observation_domain_rejects(n_obs):
    with pytest.raises(ValueError, match="positive integer"):
        WorkingModelCriterionMetadata.from_structure(PenaltyStructure(2, ()), n_obs)
    with pytest.raises(ValueError, match="exceed"):
        WorkingModelCriterionMetadata.from_structure(PenaltyStructure(2, ()), 2)


@pytest.mark.parametrize("field", ["parameters", "gram", "rhs", "response_square_sum"])
def test_proposal_static_shapes_reject(simple_metadata, field):
    smoothing = jnp.ones(1)
    arguments = {
        "parameters": jnp.array([0.0, -3.0]),
        "gram": jnp.eye(2),
        "rhs": jnp.ones(2),
        "response_square_sum": jnp.asarray(3.0),
        "metadata": simple_metadata,
        "factors": factor_working_model(
            working_model_factor_inputs(jnp.eye(2), simple_metadata, smoothing)
        ),
        "smoothing": smoothing,
    }
    arguments[field] = jnp.ones(3)
    with pytest.raises(ValueError, match=r"parameters|gram and rhs|scalar"):
        gaussian_working_model_proposal(**arguments)


def test_trial_exponentials_and_factors_are_one_consistent_input(simple_metadata):
    statistics = (jnp.eye(2), jnp.ones(2), jnp.asarray(3.0))
    default = evaluate_gaussian_working_model_proposal(
        np.array([0.4, -3.0]), *statistics, simple_metadata
    )
    # The host exponential is the C library exp pinned R calls.
    supplied = evaluate_gaussian_working_model_proposal(
        np.array([0.4, -3.0]),
        *statistics,
        simple_metadata,
        smoothing=np.array([np.exp(0.4)]),
    )
    for field in default._fields:
        np.testing.assert_array_equal(getattr(supplied, field), getattr(default, field))
    np.testing.assert_array_equal(
        source_exponentials(np.array([0.4, -23.0])), [np.exp(0.4), np.exp(-23.0)]
    )
    factors = factor_working_model(
        working_model_factor_inputs(jnp.eye(2), simple_metadata, jnp.ones(1))
    )
    with pytest.raises(ValueError, match="one value per log"):
        working_model_factor_inputs(jnp.eye(2), simple_metadata, jnp.ones(2))
    with pytest.raises(ValueError, match="one value per log"):
        gaussian_working_model_proposal(
            jnp.array([0.4, -3.0]),
            *statistics,
            simple_metadata,
            factors,
            smoothing=jnp.ones(2),
        )
    with pytest.raises(ValueError, match="coefficient dimension"):
        working_model_factor_inputs(jnp.eye(3), simple_metadata, jnp.ones(1))
    with pytest.raises(ValueError, match="coupled penalty block"):
        gaussian_working_model_proposal(
            jnp.array([0.4, -3.0]),
            *statistics,
            simple_metadata,
            factors._replace(penalties=(factors.hessian,)),
            smoothing=jnp.ones(1),
        )
    nonfinite = working_model_factor_inputs(
        jnp.eye(2), simple_metadata, jnp.array([np.inf])
    )
    with pytest.raises(FloatingPointError, match="Non-finite"):
        factor_working_model(nonfinite)


@pytest.mark.platform_sensitive
def test_source_scaled_triangular_solve_and_ift_contractions_are_jittable():
    import scipy.linalg as sla

    from jaxgam.fitting.data import _to_jax_structure
    from jaxgam.fitting.working_model import (
        _source_cholesky_solve,
        _source_hessian_determinant_derivatives,
        _source_ift_chol,
    )
    from jaxgam.penalties.structure import DenseLocalPenalty

    generator = np.random.default_rng(41725)
    X = generator.normal(size=(31, 5))
    G = X.T @ X
    components = []
    for _ in range(2):
        root = generator.normal(size=(5, 3))
        components.append(root @ root.T)
    structure = PenaltyStructure(
        5,
        (
            PenaltyBlock(
                0,
                5,
                (0, 1),
                tuple(DenseLocalPenalty(S) for S in components),
                IdentityTransform(5),
                (3, 3),
            ),
        ),
    )
    rho = np.log(np.array([0.4, 0.7]))
    weighted = [np.exp(value) * S for value, S in zip(rho, components, strict=True)]
    H = G + sum(weighted)
    diagonal = np.sqrt(np.diag(H))
    scaled = (H / diagonal[:, None]).T / diagonal[:, None]
    upper, pivots, rank, _ = sla.lapack.dpstrf(scaled, lower=0)
    assert rank == len(H)
    upper = np.triu(upper)
    pivots = pivots - 1
    rhs = generator.normal(size=5)
    solve = jax.jit(_source_cholesky_solve)
    beta = solve(upper, pivots, diagonal, rhs)
    expected_beta = np.linalg.solve(H, rhs)
    expected_D = np.column_stack([S @ expected_beta for S in weighted])
    expected_db = -np.linalg.solve(H, expected_D)
    first, second, db = jax.jit(_source_ift_chol)(
        G,
        beta,
        _to_jax_structure(structure, None),
        rho,
        upper,
        pivots,
        diagonal,
    )
    inverse = solve(upper, pivots, diagonal, np.eye(len(H)))
    det_first, det_second = jax.jit(_source_hessian_determinant_derivatives)(
        inverse,
        _to_jax_structure(structure, None),
        rho,
    )
    products = [S @ np.linalg.inv(H) for S in weighted]
    expected_det_first = np.array([np.trace(product) for product in products])
    expected_det_second = np.diag(expected_det_first) - np.array(
        [[np.sum(a.T * b) for b in products] for a in products]
    )
    expected_first = np.sum(expected_beta[:, None] * expected_D, axis=0)
    expected_second = np.diag(expected_first) + 2.0 * expected_D.T @ expected_db
    collector = _AssertCollector()
    for name, observed, target in (
        ("scaled/pivoted beta", beta, expected_beta),
        ("H determinant gradient", det_first, expected_det_first),
        ("H determinant Hessian", det_second, expected_det_second),
        ("implicit coefficient derivatives", db, expected_db),
        ("penalized RSS gradient", first, expected_first),
        ("expanded source penalized RSS Hessian", second, expected_second),
    ):
        collector.check(
            name,
            lambda observed=observed, target=target: np.testing.assert_allclose(
                observed, target, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("source triangular solve and IFT")


@pytest.mark.platform_sensitive
def test_source_penalty_trace_contractions_match_range_algebra(compact_working_model):
    _, _, smoothing, _, metadata, _, _ = compact_working_model
    rho = np.log(smoothing)
    lam = np.exp(rho)
    inputs = working_model_factor_inputs(
        jnp.eye(metadata.penalties.n_coef), metadata, jnp.asarray(lam)
    )
    # An unpivoted Cholesky factor isolates the trace algebra from DPSTRF.
    factors = tuple(
        WorkingModelFactor(
            np.linalg.cholesky(np.asarray(matrix)).T,
            np.arange(len(matrix), dtype=np.int32),
            np.int32(len(matrix)),
            np.asarray(diagonal),
        )
        for matrix, diagonal in zip(
            inputs.penalties, inputs.penalty_diagonals, strict=True
        )
    )
    observed = jax.jit(
        lambda values, blocks: _source_penalty_determinant(
            values, metadata, blocks, jnp.asarray(lam)
        )
    )(rho, factors)
    expected_value = 0.0
    expected_first = np.zeros_like(rho)
    expected_second = np.zeros((rho.size, rho.size))
    for sp, rank in zip(
        metadata.singleton_sp_indices, metadata.singleton_ranks, strict=True
    ):
        expected_value += rank * rho[sp]
        expected_first[sp] += rank
    for indices, roots in zip(
        metadata.multi_block_sp_indices, metadata.multi_block_roots, strict=True
    ):
        components = [
            np.exp(rho[sp]) * (np.asarray(root) @ np.asarray(root).T)
            for sp, root in zip(indices, roots, strict=True)
        ]
        combined = sum(components)
        expected_value += np.linalg.slogdet(combined)[1]
        inverse = np.linalg.inv(combined)
        products = [inverse @ component for component in components]
        for i, sp in enumerate(indices):
            derivative = np.trace(products[i])
            expected_first[sp] += derivative
            expected_second[sp, sp] += derivative
            for j, other in enumerate(indices):
                expected_second[sp, other] -= np.sum(products[i].T * products[j])
    collector = _AssertCollector()
    for name, actual, target in zip(
        ("determinant", "gradient", "Hessian"),
        observed,
        (expected_value, expected_first, expected_second),
        strict=True,
    ):
        collector.check(
            name,
            lambda actual=actual, target=target: np.testing.assert_allclose(
                actual, target, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("source penalty trace algebra")


@pytest.mark.platform_sensitive
def test_source_scaled_solve_rank_padding_preserves_dropped_coordinates():
    from jaxgam.fitting.working_model import _source_cholesky_solve

    # Source pivots put the zero direction last. Its residual factor storage
    # must not be interpreted as another invertible coefficient direction.
    upper = jnp.diag(jnp.array([2.0, 1.0, 0.0]))
    pivots = jnp.array([0, 2, 1])
    result = jax.jit(_source_cholesky_solve)(
        upper, pivots, jnp.ones(3), jnp.array([4.0, 0.0, 1.0]), jnp.asarray(2)
    )
    np.testing.assert_array_equal(result, np.array([1.0, 0.0, 1.0]))


@pytest.mark.platform_sensitive
def test_source_penalty_addition_follows_sl_adds_order():
    from jaxgam.fitting.data import _to_jax_structure
    from jaxgam.fitting.working_model import _source_add_penalties
    from jaxgam.penalties.structure import DenseLocalPenalty

    generator = np.random.default_rng(1729)
    gram = generator.normal(size=(5, 5))
    gram = gram @ gram.T
    components = [generator.normal(size=(4, 4)) for _ in range(2)]
    components = [S @ S.T for S in components]
    structure = PenaltyStructure(
        5,
        (
            PenaltyBlock(
                1,
                5,
                (0, 1),
                tuple(DenseLocalPenalty(S) for S in components),
                IdentityTransform(4),
                (4, 4),
            ),
        ),
    )
    lam = np.array([3.7e-10, 2.4e6])
    expected = gram.copy()
    for value, S in zip(lam, components, strict=True):
        # mgcv_madi: A[ind, ind] <- A[ind, ind] + (lambda_j * S_j), in turn.
        expected[1:, 1:] = expected[1:, 1:] + value * S
    actual = jax.jit(_source_add_penalties)(
        _to_jax_structure(structure, None),
        jnp.asarray(gram),
        jnp.log(lam),
        jnp.asarray(lam),
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.platform_sensitive
def test_source_newton_symmetric_eigen_uses_lower_triangle():
    gradient = np.array([0.4, -0.2])
    hessian = np.array([[4.0, 9.0], [1.0, 2.0]])
    symmetric = np.array([[4.0, 1.0], [1.0, 2.0]])
    actual = jax.jit(_sl_newton_step)(gradient, hessian, jnp.asarray(0.0))
    np.testing.assert_allclose(
        actual,
        -np.linalg.solve(symmetric, gradient),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


@pytest.mark.platform_sensitive
def test_source_root_crossproducts_match_pinned_reference_dsyrk(r_bridge):
    assert r_bridge.check_versions()[0]
    frame = _frame()
    expected = r_bridge.source_bam_penalty_root_products(
        'y ~ te(x,z,bs="cr",k=3)', frame
    )
    for root, product in expected:
        root = np.array(root, copy=True)
        product = np.array(product, copy=True)
        phase_one = source_root_product(root)
        if scipy_uses_accelerate_lapack():
            np.testing.assert_array_equal(phase_one, product)
        else:
            np.testing.assert_allclose(
                phase_one, product, rtol=STRICT.rtol, atol=STRICT.atol
            )
        dynamic = jax.jit(_source_tcrossprod)(jnp.asarray(root))
        np.testing.assert_allclose(dynamic, product, rtol=STRICT.rtol, atol=STRICT.atol)


@pytest.mark.platform_sensitive
def test_coupled_source_setup_root_products_match_pinned_sl_setup(r_bridge):
    assert r_bridge.check_versions()[0]
    frame = _frame()
    formula = 'y ~ te(x,z,bs="cr",k=3)'
    setup = build_continuous_discrete_setup(
        parse_formula(formula),
        DataFrameRowSource(frame, response="y"),
        m=30,
    )
    try:
        (source_block,) = setup.source_determinant_penalties.blocks
        penalties = tuple(source_block.dense_penalties())
        (block,) = build_working_penalty_blocks(
            setup.source_determinant_penalties, source_eigensolver=True
        )
    finally:
        setup.cleanup()
    transform, rank, _, roots = r_bridge.source_working_penalty_setup(
        penalties, source_block.ranks
    )
    assert rank == block.rank
    # Equal eigenvalues permit a rotation within each range eigenspace.
    # Compare the public penalty operator after mapping both roots back.
    for root, target in zip(block.roots, roots, strict=True):
        physical = block.transform[:, :rank] @ root
        source_physical = transform[:, :rank] @ target
        np.testing.assert_allclose(
            physical @ physical.T,
            source_physical @ source_physical.T,
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )


@pytest.mark.parametrize("scale_is_fixed", [False, True], ids=["free", "known"])
def test_fixed_smoothing_proposal_keeps_only_the_scale_coordinate(
    compact_working_model, scale_is_fixed
):
    """Sl.fitChol with a zero-column L: t(L) reml1 and t(L) reml2 L.

    Every log(sp) still enters the system; the gradient, Hessian and step keep
    only log(phi), or nothing when the scale is fixed too. Compiled and run.
    """
    frame, _, smoothing, X, metadata, weights, offset = compact_working_model
    statistics = _statistics(X, frame.y.to_numpy(), weights, offset, 13)
    parameters = np.log(smoothing) if scale_is_fixed else np.r_[np.log(smoothing), -3.0]
    lam = jnp.asarray(source_exponentials(np.log(smoothing)))
    factors = factor_working_model(
        jax.jit(working_model_factor_inputs)(statistics[0], metadata, lam)
    )
    proposal = jax.jit(
        gaussian_working_model_proposal,
        static_argnames=("scale_is_fixed", "fixed_smoothing"),
    )
    full = proposal(
        jnp.asarray(parameters),
        *statistics,
        metadata,
        factors,
        smoothing=lam,
        scale_is_fixed=scale_is_fixed,
        log_scale=-3.0,
    )
    fixed = proposal(
        jnp.asarray(parameters),
        *statistics,
        metadata,
        factors,
        smoothing=lam,
        scale_is_fixed=scale_is_fixed,
        fixed_smoothing=True,
        log_scale=-3.0,
    )
    free = slice(len(smoothing), None)
    collector = _AssertCollector()
    for name in ("coefficients", "criterion", "logdet_hessian", "logdet_penalty"):
        collector.check(
            name,
            lambda name=name: np.testing.assert_array_equal(
                getattr(fixed, name), getattr(full, name)
            ),
        )
    collector.check(
        "restricted gradient and Hessian",
        lambda: (
            np.testing.assert_array_equal(fixed.gradient, full.gradient[free]),
            np.testing.assert_array_equal(fixed.hessian, full.hessian[free, free]),
        ),
    )
    collector.check(
        "restricted step",
        lambda: np.testing.assert_allclose(
            fixed.step,
            _sl_newton_step(full.gradient[free], full.hessian[free, free], 0.0),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "shapes",
        lambda: check_that(
            fixed.step.shape == ((0,) if scale_is_fixed else (1,)),
            f"fixed-smoothing step shape {fixed.step.shape}",
        ),
    )
    collector.raise_if_any(f"fixed-smoothing proposal {scale_is_fixed=}")


@pytest.mark.platform_sensitive
def test_source_sum_is_rs_long_double_sum_and_compiles():
    """R's sum: element order, C long double accumulator, one rounding.

    NumPy's longdouble is the same C type as R's LDOUBLE on this platform,
    so its sequential sum is the reference. The values cancel to about 1e-3
    of their magnitude, where XLA's reduction and a double sum differ.
    """
    rng = np.random.default_rng(3)
    compiled = jax.jit(source_sum)
    for size in (1, 2, 7, 23, 64):
        for _ in range(50):
            values = rng.normal(scale=1e3, size=size)
            values[-1] = -np.sum(values[:-1]) + rng.normal()
            expected = np.longdouble(0.0)
            for value in values:
                expected += np.longdouble(value)
            assert float(compiled(jnp.asarray(values))) == float(expected)
    assert float(compiled(jnp.zeros(0))) == 0.0


def _rounded(value: Fraction, bits: int) -> Fraction:
    """``value`` to ``bits`` significant bits, ties to even (no exponent limit)."""
    if value == 0:
        return value
    sign = -1 if value < 0 else 1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    quantum = Fraction(2) ** (exponent - bits + 1)
    scaled = value / quantum
    whole = scaled.numerator // scaled.denominator
    remainder = scaled - whole
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and whole % 2):
        whole += 1
    return sign * whole * quantum


def _accumulated(values, bits: int) -> float:
    """Exact model of a ``bits``-significant-bit sequential accumulator."""
    total = Fraction(0)
    for value in values:
        total = _rounded(total + Fraction(float(value)), bits)
    return float(total)


def test_every_long_double_accumulator_matches_its_exact_model():
    """``source_sum`` and each LDOUBLE accumulator against an exact model.

    ``source_sum`` is checked at the width ``np.finfo(np.longdouble)``
    reports here, so native x86_64 runs exercise its 80-bit path.

    Double (53 bits, macOS arm64), x87 extended (64 bits, x86_64 Linux) and
    quad (113 bits, aarch64 Linux) are modelled by rounding the exact running
    sum to that many bits after every addition. On all three, pinned R's
    ``sum`` and NumPy's sequential ``longdouble`` sum were checked equal to
    this model (1209 cases, x86_64 under emulation). The cases include x87
    half-ulp ties that a wider accumulator resolves differently.
    """
    rng = np.random.default_rng(7)
    cases = []
    for size in (1, 2, 3, 7, 23, 64):
        for _ in range(40):
            values = rng.normal(scale=10.0 ** rng.integers(-3, 4), size=size)
            if size > 1:
                values[-1] = -np.sum(values[:-1]) + rng.normal() * 1e-6
            cases.append(values)
    tie = 2.0**-64
    cases += [
        np.array([1.0, tie, tie, tie]),
        np.array([1.0, tie, -tie, tie, tie]),
        np.array([1.0, 3 * tie, tie]),
        np.array([2.0**53, 1.0, 2.0**-11, -(2.0**53)]),
        np.array([1.0, -(2.0**-65), -(2.0**-65)]),
        np.array([1.0 + 2.0**-52, 2.0**-64 + 2.0**-116]),
        np.array([-1.0, -tie, -tie]),
        np.array([0.0, 0.0]),
        np.array([1.0, 2.0**-53, tie]),
    ]
    collector = _AssertCollector()
    # source_sum itself, against the exact model of the width this platform's
    # C long double reports (np.finfo): the 80-bit path on x86_64 CI.
    platform_bits = int(np.finfo(np.longdouble).nmant) + 1
    compiled_source = jax.jit(source_sum)
    platform_mismatches = [
        index
        for index, values in enumerate(cases)
        if float(compiled_source(jnp.asarray(values)))
        != _accumulated(values, platform_bits)
    ]
    collector.check(
        f"source_sum at {platform_bits} bits",
        lambda: check_that(
            not platform_mismatches,
            f"source_sum differs from the {platform_bits}-bit model in cases "
            f"{platform_mismatches}",
        ),
    )
    for name, accumulate, bits in (
        ("double", _sequential_double_sum, 53),
        ("x87", _x87_sum, 64),
        ("quad", _double_double_sum, 113),
    ):
        compiled = jax.jit(accumulate)
        mismatches = [
            index
            for index, values in enumerate(cases)
            if float(compiled(jnp.asarray(values))) != _accumulated(values, bits)
        ]
        collector.check(
            name,
            lambda name=name, mismatches=mismatches: check_that(
                not mismatches, f"{name} accumulator differs in cases {mismatches}"
            ),
        )
    # x87 rounds 1 + 2^-53 + 2^-64 to even at 64 bits, then to 1.0; quad
    # keeps it above the double midpoint, so the final double differs.
    separating = cases[-1]
    collector.check(
        "x87 ties",
        lambda: check_that(
            float(jax.jit(_x87_sum)(jnp.asarray(separating))) == 1.0
            and float(jax.jit(_double_double_sum)(jnp.asarray(separating)))
            == 1.0 + 2.0**-52,
            "tie case must separate x87 from quad",
        ),
    )
    collector.raise_if_any("long double accumulators")
