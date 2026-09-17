"""Exact factor-action EFS contractions and explicit bounded coordinates."""

import json
import logging
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from jaxgam.data.source import DataFrameRowSource
from jaxgam.families.standard import Poisson
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.fitting.efs import (
    EFSRoot,
    EFSStatisticsPlan,
    efs_statistics,
    prepare_efs_statistics,
)
from jaxgam.fitting.efs_factor import efs_factor_statistics
from jaxgam.fitting.signed_qr import SignedQRCoefficientFactor
from jaxgam.fitting.state import CholeskyCoefficientFactor, PivotedQRCoefficientFactor
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from tests.helpers import _AssertCollector, r_available
from tests.r_bridge import RBridge
from tests.test_fitting.test_efs import _plan
from tests.tolerances import STRICT


def _qr(H, keep=None):
    p = len(H)
    keep = np.arange(p) if keep is None else np.asarray(keep)
    reduced = H[np.ix_(keep, keep)]
    _, R, pivots = linalg.qr(np.linalg.cholesky(reduced).T, pivoting=True)
    return PivotedQRCoefficientFactor(
        jnp.asarray(R), jnp.asarray(pivots), jnp.asarray(keep), p
    )


def _arguments():
    plan = _plan()
    H = np.array(
        [
            [5.0, 0.2, 0.1, 0.0],
            [0.2, 4.0, 0.0, 0.1],
            [0.1, 0.0, 3.5, 0.4],
            [0.0, 0.1, 0.4, 4.5],
        ]
    )
    return plan, jnp.array([0.5, -1.0, 0.2, 0.7]), H, jnp.array([-0.2, 0.4, -0.3])


@pytest.mark.parametrize("kind", ["cholesky", "qr", "signed"])
def test_exact_factor_statistics_use_reviewed_actions_under_jit(kind):
    plan, beta, H, rho = _arguments()
    lower = jnp.asarray(np.linalg.cholesky(H))
    factor = CholeskyCoefficientFactor(lower, 4) if kind == "cholesky" else _qr(H)
    if kind == "signed":
        angle = 0.3
        V = np.eye(4)
        V[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        correction = np.array([0.5, 0.8, 1.0, 1.0])
        factor = SignedQRCoefficientFactor(
            factor, jnp.asarray(V), jnp.asarray(correction)
        )
        invroot = np.asarray(factor.root_inverse(jnp.eye(4)))
        H = np.linalg.inv(invroot @ invroot.T)
        lower = jnp.asarray(np.linalg.cholesky(H))
    expected = jax.jit(efs_statistics)(plan, beta, lower, rho)
    actual = jax.jit(efs_factor_statistics)(plan, beta, factor, rho)
    collector = _AssertCollector()
    for name in ("determinant_derivative", "fisher_trace", "quadratic"):
        collector.check(
            name,
            lambda name=name: np.testing.assert_allclose(
                getattr(actual, name),
                getattr(expected, name),
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
        )
    collector.raise_if_any("factor EFS contractions")
    assert actual.input_valid
    assert actual.determinant_valid
    if kind == "cholesky":
        for a, b in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    else:
        assert not np.array_equal(
            np.asarray(
                factor.absolute_factor.pivots if kind == "signed" else factor.pivots
            ),
            np.arange(4),
        )


def test_prepared_metadata_builds_statistics_plan_without_training_rows():
    n = 74
    x = np.linspace(0, 1, n)
    family = Poisson()
    source = DataFrameRowSource(
        pd.DataFrame({"x": x, "y": np.arange(n) % 5}), response="y"
    )
    prepared = prepare_model(
        parse_formula('y ~ s(x,bs="cr",k=8)'), source, family=family
    )
    metadata = PreparedFittingMetadata.from_prepared(prepared, family)
    assert not any(hasattr(metadata, key) for key in ("X", "y", "wt", "offset"))
    plan = prepare_efs_statistics(metadata)
    factor = CholeskyCoefficientFactor(jnp.eye(plan.n_coef), plan.n_coef)
    actual = jax.jit(efs_factor_statistics)(
        plan, jnp.ones(plan.n_coef), factor, metadata.log_lambda_init
    )
    assert actual.input_valid
    assert actual.determinant_valid
    assert plan.n_coef == prepared.n_coef
    assert plan.n_penalties == 1
    assert (
        actual.determinant_derivative[0]
        == prepared.fitting.penalty_structure.blocks[0].ranks[0]
    )


def test_rank_projection_and_permutation_are_explicit():
    plan, beta, H, rho = _arguments()
    factor = _qr(H, [3, 1])
    actual = jax.jit(efs_factor_statistics)(plan, beta, factor, rho)
    matrices = []
    for root in plan.roots:
        padded = np.zeros((4, root.values.shape[1]))
        padded[root.start : root.stop] = root.values
        matrices.append(padded @ padded.T)
    inverse = np.zeros((4, 4))
    keep = np.asarray(factor.keep)
    inverse[np.ix_(keep, keep)] = np.linalg.inv(H[np.ix_(keep, keep)])
    expected = [np.trace(inverse @ S) for S in matrices]
    np.testing.assert_allclose(
        actual.fisher_trace, expected, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert actual.input_valid
    # A valid contraction in this supplied subspace is not a score-domain gate.
    empty = PivotedQRCoefficientFactor(
        jnp.empty((0, 0)), jnp.empty(0, dtype=int), jnp.empty(0, dtype=int), 4
    )
    assert not efs_factor_statistics(plan, beta, empty, rho).input_valid


@pytest.mark.parametrize(
    "defect", ["nan", "zero", "pivots", "keep", "negative-correction"]
)
def test_bad_factor_values_never_report_valid_statistics(defect):
    plan, beta, H, rho = _arguments()
    factor = _qr(H)
    if defect == "nan":
        factor = replace(factor, R=factor.R.at[0, 0].set(jnp.nan))
    if defect == "zero":
        factor = replace(factor, R=factor.R.at[0, 0].set(0.0))
    if defect == "pivots":
        factor = replace(factor, pivots=jnp.zeros(4, dtype=int))
    if defect == "keep":
        factor = replace(factor, keep=jnp.array([-1, 1, 2, 4]))
    if defect == "negative-correction":
        factor = SignedQRCoefficientFactor(
            factor, jnp.eye(4), jnp.array([-0.1, 1.0, 1.0, 1.0])
        )
    assert not jax.jit(efs_factor_statistics)(plan, beta, factor, rho).input_valid


@pytest.mark.parametrize(
    "defect",
    [
        "beta-shape",
        "rho-shape",
        "coordinates",
        "lower-shape",
        "qr-shape",
        "index-dtype",
        "signed-shape",
        "beta-dtype",
        "factor-dtype",
    ],
)
def test_static_factor_contract_rejects_mismatched_shapes_and_dtypes(defect):
    plan, beta, H, rho = _arguments()
    factor = _qr(H)
    if defect == "beta-shape":
        beta = beta[:-1]
    if defect == "rho-shape":
        rho = rho[:-1]
    if defect == "coordinates":
        factor = replace(factor, original_n_coef=3)
    if defect == "lower-shape":
        factor = CholeskyCoefficientFactor(jnp.ones((3, 4)), 4)
    if defect == "qr-shape":
        factor = replace(factor, R=factor.R[:3])
    if defect == "index-dtype":
        factor = replace(factor, keep=factor.keep.astype(float))
    if defect == "signed-shape":
        factor = SignedQRCoefficientFactor(factor, jnp.eye(3), jnp.ones(4))
    if defect == "beta-dtype":
        beta = beta.astype(jnp.float32)
    if defect == "factor-dtype":
        factor = replace(factor, R=factor.R.astype(jnp.float32))
    with pytest.raises(
        (ValueError, TypeError), match=r"matching|different|float64|integer"
    ):
        jax.jit(efs_factor_statistics)(plan, beta, factor, rho)


@pytest.mark.parametrize("kind", ["identity", "diagonal", "dense"])
def test_local_root_blocks_above_rhs_budget_and_tail_columns(kind):
    p = 79
    values = jnp.asarray(1.2) if kind == "identity" else jnp.linspace(0.5, 1.5, p)
    if kind == "dense":
        values = jnp.diag(values)
    root = EFSRoot(0, p, 0, kind, values, () if kind == "dense" else tuple(range(p)))
    plan = EFSStatisticsPlan(p, 1, (root,), (0,), (p,), (), (), ())
    factor = PivotedQRCoefficientFactor(
        jnp.diag(jnp.linspace(2.0, 3.0, p)), jnp.arange(p - 1, -1, -1), jnp.arange(p), p
    )
    beta = jnp.linspace(-1, 1, p)
    actual = jax.jit(efs_factor_statistics)(plan, beta, factor, jnp.zeros(1))
    expected = efs_statistics(
        plan,
        beta,
        jnp.asarray(
            np.linalg.cholesky(
                np.linalg.inv(np.asarray(factor.hessian_inverse(jnp.eye(p))))
            )
        ),
        jnp.zeros(1),
    )
    for name in ("fisher_trace", "quadratic", "determinant_derivative"):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
    assert actual.input_valid


def test_debug_flag_compiles_as_dynamic_boolean(capsys):
    plan, beta, H, rho = _arguments()
    compiled = jax.jit(efs_factor_statistics)
    actual = compiled(plan, beta, _qr(H), rho, debug=jnp.asarray(True))
    actual.fisher_trace.block_until_ready()
    assert "EFS factor d=" in capsys.readouterr().out


@pytest.mark.skipif(not r_available(), reason="requires pinned R/mgcv")
@pytest.mark.parametrize("kind", ["cholesky", "qr", "signed"])
@pytest.mark.parametrize("log_smoothing", [(0.3, -0.5), (-6.0, 6.0), (6.0, -6.0)])
def test_factor_trace_matches_actual_pinned_reparam_covariance_contractions(
    kind, log_smoothing
):
    s1 = np.array([[2.0, 0.4], [0.4, 1.0]])
    s2 = np.array([[1.0, -0.3], [-0.3, 3.0]])
    roots = [np.linalg.cholesky(s1), np.linalg.cholesky(s2)]
    plan = EFSStatisticsPlan(
        3,
        2,
        tuple(
            EFSRoot(1, 3, i, "dense", jnp.asarray(root), ())
            for i, root in enumerate(roots)
        ),
        (),
        (),
        ((0, 1),),
        (2,),
        ((jnp.asarray(s1), jnp.asarray(s2)),),
    )
    H = np.array([[4.0, 0.2, 0.1], [0.2, 3.0, 0.4], [0.1, 0.4, 2.0]])
    beta = jnp.array([0.2, -0.8, 1.1])
    rho = jnp.asarray(log_smoothing)
    factor = (
        CholeskyCoefficientFactor(jnp.asarray(np.linalg.cholesky(H)), 3)
        if kind == "cholesky"
        else _qr(H)
    )
    if kind == "signed":
        factor = SignedQRCoefficientFactor(
            factor, jnp.eye(3), jnp.array([0.6, 0.8, 1.0])
        )
        invroot = np.asarray(factor.root_inverse(jnp.eye(3)))
        H = np.linalg.inv(invroot @ invroot.T)
    actual = jax.jit(efs_factor_statistics)(plan, beta, factor, rho)
    covariance_roots = [np.vstack((np.zeros((1, 2)), root)) for root in roots]
    oracle = RBridge(mode="rpy2").efs_statistics_algebra(
        np.asarray(rho),
        roots,
        covariance_roots,
        np.asarray(beta),
        np.linalg.cholesky(H),
    )
    collector = _AssertCollector()
    for field, key in (
        ("determinant_derivative", "d"),
        ("fisher_trace", "t"),
        ("quadratic", "q"),
    ):
        collector.check(
            field,
            lambda field=field, key=key: np.testing.assert_allclose(
                getattr(actual, field), oracle[key], rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("factor EFS contractions")


@pytest.mark.parametrize(
    "defect",
    [
        "range",
        "parameter",
        "dtype",
        "dense-shape",
        "duplicate",
        "outside",
        "identity-shape",
        "diagonal-shape",
        "unknown",
    ],
)
def test_local_root_contract_rejects_misattributed_or_malformed_inputs(defect):
    root = EFSRoot(0, 4, 0, "dense", jnp.eye(4), ())
    if defect == "range":
        root = replace(root, stop=5)
    if defect == "parameter":
        root = replace(root, sp_index=1)
    if defect == "dtype":
        root = replace(root, values=root.values.astype(jnp.float32))
    if defect == "dense-shape":
        root = replace(root, values=jnp.ones((3, 4)))
    if defect == "duplicate":
        root = replace(root, kind="identity", values=jnp.asarray(1.0), indices=(0, 0))
    if defect == "outside":
        root = replace(root, kind="identity", values=jnp.asarray(1.0), indices=(4,))
    if defect == "identity-shape":
        root = replace(root, kind="identity", values=jnp.ones(1), indices=(0,))
    if defect == "diagonal-shape":
        root = replace(root, kind="diagonal", values=jnp.ones(2), indices=(0,))
    if defect == "unknown":
        root = replace(root, kind="unreviewed")
    plan = EFSStatisticsPlan(4, 1, (root,), (0,), (4,), (), (), ())
    with pytest.raises((ValueError, TypeError), match=r"root|float64"):
        jax.jit(efs_factor_statistics)(
            plan, jnp.ones(4), CholeskyCoefficientFactor(jnp.eye(4), 4), jnp.zeros(1)
        )


def test_dynamic_factor_parameter_trials_are_isolated_and_nonfinite_inputs_invalid():
    plan, beta, H, rho = _arguments()
    compiled = jax.jit(efs_factor_statistics)
    factor = _qr(H)
    original = compiled(plan, beta, factor, rho)
    trial = compiled(plan, beta * 2, replace(factor, R=factor.R * 3), rho + 0.2)
    repeated = compiled(plan, beta, factor, rho)
    np.testing.assert_allclose(
        trial.fisher_trace,
        original.fisher_trace / 9,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    np.testing.assert_allclose(
        trial.quadratic, original.quadratic * 4, rtol=STRICT.rtol, atol=STRICT.atol
    )
    for first, again in zip(
        jax.tree_util.tree_leaves(original),
        jax.tree_util.tree_leaves(repeated),
        strict=True,
    ):
        assert np.asarray(first).tobytes() == np.asarray(again).tobytes()
    assert not compiled(plan, beta.at[0].set(jnp.nan), factor, rho).input_valid
    assert not compiled(plan, beta, factor, rho.at[0].set(jnp.inf)).input_valid


@pytest.mark.parametrize("p", [96, 192])
@pytest.mark.parametrize("m", [1, 5])
def test_compiled_memory_records_bounded_rhs_and_scalar_output(p, m, caplog):
    caplog.set_level(logging.INFO)

    roots = tuple(
        EFSRoot(
            p * i // m,
            p * (i + 1) // m,
            i,
            "identity",
            jnp.asarray(1.0),
            tuple(range(p * (i + 1) // m - p * i // m)),
        )
        for i in range(m)
    )
    ranks = tuple(root.stop - root.start for root in roots)
    plan = EFSStatisticsPlan(p, m, roots, tuple(range(m)), ranks, (), (), ())
    factor = PivotedQRCoefficientFactor(
        jnp.eye(p), jnp.arange(p - 1, -1, -1), jnp.arange(p), p
    )
    compiled = (
        jax.jit(efs_factor_statistics)
        .lower(plan, jnp.ones(p), factor, jnp.zeros(m))
        .compile()
    )
    actual = compiled(plan, jnp.ones(p), factor, jnp.zeros(m))
    actual.fisher_trace.block_until_ready()
    np.testing.assert_allclose(
        actual.fisher_trace, ranks, rtol=STRICT.rtol, atol=STRICT.atol
    )
    memory = compiled.memory_analysis()
    record = {
        "p": p,
        "m": m,
        "rhs_columns": 32,
        "largest_local_root_rank": max(ranks),
        "visible_rhs_and_action_bytes": 2 * p * min(32, max(ranks)) * 8,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "argument_bytes": memory.argument_size_in_bytes,
        "output_bytes": memory.output_size_in_bytes,
        "temporary_bytes": memory.temp_size_in_bytes,
        "alias_bytes": memory.alias_size_in_bytes,
        "generated_code_bytes": memory.generated_code_size_in_bytes,
    }
    assert memory.output_size_in_bytes < 128 + 3 * m * 8
    logging.getLogger(__name__).info(
        "EFS_FACTOR_MEMORY %s", json.dumps(record, sort_keys=True)
    )
    assert '"rhs_columns": 32' in caplog.text
