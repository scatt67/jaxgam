"""Tests for the stable penalty reparameterization (``gam.reparam`` port).

Inputs are generated at runtime from JaxGAM's own tensor smooth constructors:
each block's penalties are reduced to square roots in the range space of the
block's total penalty, the form in which pinned ``gam.fit3`` passes ``UrS`` to
``gam.reparam``.  R comparisons feed R the identical roots through
``RBridge.source_gam_reparam``.
"""

from __future__ import annotations

import itertools
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache, partial

import numpy as np
import pytest

from jaxgam.penalties.stable_reparam import (
    D_TOL,
    R_TOL,
    StableReparam,
    stable_reparam,
)
from jaxgam.smooths.cubic import CubicRegressionSmooth
from jaxgam.smooths.source_arithmetic import libm_exp
from jaxgam.smooths.tensor import TensorInteractionSmooth, TensorProductSmooth
from tests.helpers import _AssertCollector, check_that, make_smooth_spec
from tests.tolerances import MODERATE, STRICT, ToleranceClass

# kind, margin basis, k, dimension
_BLOCKS = {
    "te_cr_k5": ("te", "cr", 5, 2),
    "te_tp_k5": ("te", "tp", 5, 2),
    "ti_cr_k5": ("ti", "cr", 5, 2),
    "ti_tp_k6": ("ti", "tp", 6, 2),
    "te_cr_k4_3d": ("te", "cr", 4, 3),
    "ti_cr_k4_3d": ("ti", "cr", 4, 3),
}
_GAPS = (5.0, 10.0, 20.0, 40.0, 60.0)


def _close(actual, expected, label: str) -> None:
    np.testing.assert_allclose(
        actual, expected, rtol=STRICT.rtol, atol=STRICT.atol, err_msg=label
    )


def _grid(n_penalties: int) -> list[np.ndarray]:
    """Equal values, gaps of 5..60 in every order, and very negative values."""
    points = [np.full(n_penalties, value) for value in (0.0, 3.0, -40.0, -60.0)]
    for gap in _GAPS:
        if n_penalties == 2:
            points += [
                np.array([gap, 0.0]),
                np.array([0.0, gap]),
                np.array([-30.0, -30.0 - gap]),
                np.array([-30.0 - gap, -30.0]),
            ]
        else:
            orders = sorted(set(itertools.permutations((-gap, 0.0, gap))))
            points += [np.array(order) for order in orders]
            points.append(np.array([-50.0, -50.0, -50.0 - gap]))
    return points


def _eigen_root(penalty: np.ndarray, rank: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (penalty + penalty.T))
    return vectors[:, -rank:] * np.sqrt(values[-rank:])


@cache
def _tensor_penalties(name: str) -> tuple[np.ndarray, ...]:
    """Constrained penalties of one tensor smooth, as gam.setup leaves them."""
    kind, basis, k, dim = _BLOCKS[name]
    rng = np.random.default_rng(7)
    data = {f"x{i}": rng.uniform(0, 1, 200) for i in range(dim)}
    spec = make_smooth_spec(list(data), bs=basis, k=k, smooth_type=kind)
    smooth = (TensorProductSmooth if kind == "te" else TensorInteractionSmooth)(spec)
    smooth.setup(data)
    penalties = [penalty.S for penalty in smooth.build_penalty_matrices()]
    if kind == "te":  # absorb the sum-to-zero constraint as gam.setup does
        design = smooth.build_design_matrix(data)
        basis_q, _ = np.linalg.qr(design.mean(axis=0)[:, None], mode="complete")
        penalties = [basis_q[:, 1:].T @ S @ basis_q[:, 1:] for S in penalties]
    return tuple(0.5 * (S + S.T) for S in penalties)


def _ranks(penalties) -> list[int]:
    ranks = []
    for penalty in penalties:
        values = np.linalg.eigvalsh(penalty)
        ranks.append(int(np.sum(values > values[-1] * R_TOL)))
    return ranks


@cache
def _raw_roots(name: str) -> tuple[np.ndarray, ...]:
    """Penalty roots in the constrained coordinates (shared null space kept)."""
    penalties = _tensor_penalties(name)
    return tuple(
        _eigen_root(S, rank)
        for S, rank in zip(penalties, _ranks(penalties), strict=True)
    )


@cache
def _range_roots(name: str) -> tuple[np.ndarray, ...]:
    """Roots projected onto the range of the total penalty (gam.fit3's UrS)."""
    penalties = _tensor_penalties(name)
    total = sum(S / np.linalg.norm(S) for S in penalties)
    values, vectors = np.linalg.eigh(total)
    range_basis = vectors[:, values > values[-1] * R_TOL]
    return tuple(range_basis.T @ root for root in _raw_roots(name))


@cache
def _cubic_range_root(k: int) -> np.ndarray:
    rng = np.random.default_rng(11 + k)
    data = {"x": rng.uniform(0, 1, 150)}
    smooth = CubicRegressionSmooth(make_smooth_spec(["x"], bs="cr", k=k))
    smooth.setup(data)
    penalty = smooth.build_penalty_matrices()[0].S
    values, vectors = np.linalg.eigh(0.5 * (penalty + penalty.T))
    keep = values > values[-1] * R_TOL
    return vectors[:, keep].T @ _eigen_root(penalty, int(np.sum(keep)))


def _disjoint_roots() -> tuple[np.ndarray, np.ndarray]:
    first, second = _cubic_range_root(8), _cubic_range_root(6)
    size = len(first) + len(second)
    left = np.zeros((size, first.shape[1]))
    right = np.zeros((size, second.shape[1]))
    left[: len(first)] = first
    right[len(first) :] = second
    return left, right


def _fixed_root(size: int) -> np.ndarray:
    return np.random.default_rng(5).standard_normal((size, 3))


# --------------------------------------------------------------------------- #
#  R parity                                                                    #
# --------------------------------------------------------------------------- #


# Tolerance policy (rules addendum 2026-11-17): det2, and det1 where penalties
# are grouped, sit at get_stableS's rounding floor. det2_ii = det1_i -
# sp_i^2 tr(...) inherits det1's absolute rounding, and penalties that d_tol
# groups with unequal sp are not separated by the transform. Measured: against
# exact rationals the port is up to 4.7e-9 and pinned R up to 2.0e-9 off in
# det2 (x_te_d2, sp ratio 2^14), and det1 is 4.7x (port) / 3.3x (R) STRICT;
# port vs host R det2 differs by up to 6.8e-11 absolute (macOS arm64,
# OpenBLAS SciPy, venv-ob2). So det2 is compared at MODERATE; det stays STRICT
# everywhere. This replaces the former digest-bound allowance table and the
# lists of points where det2 was not asserted: every point is now asserted.
_DET2 = MODERATE


@pytest.mark.parametrize("name", list(_BLOCKS))
def test_determinant_and_derivatives_match_pinned_r(name: str, r_bridge) -> None:
    """det and det1 at STRICT over the whole grid; det2 at the floor class."""
    roots = _range_roots(name)
    collector = _AssertCollector()
    for rho in _grid(len(roots)):
        ours = stable_reparam(roots, rho, deriv=2)
        theirs = r_bridge.source_gam_reparam(roots, rho, 2)
        label = f"{name} rho={rho.tolist()}"
        collector.check(f"{label} det", partial(_close, ours.det, theirs.det, "det"))
        collector.check(
            f"{label} det1", partial(_close, ours.det1, theirs.det1, "det1")
        )
        collector.check(
            f"{label} det2",
            partial(_close_at, ours.det2, theirs.det2, _DET2, "det2"),
        )
        collector.check(
            f"{label} fixed flag",
            partial(check_that, not theirs.fixed_penalty, "unexpected fixed"),
        )
    collector.raise_if_any(f"{name} gam.reparam parity")


def test_identity_transform_reproduces_r_gauge(r_bridge) -> None:
    """When every penalty falls in one d_tol group, Qs = I and S, rS match R."""
    collector = _AssertCollector()
    for name in _BLOCKS:
        roots = _range_roots(name)
        for rho in (np.zeros(len(roots)), np.full(len(roots), -40.0)):
            ours = stable_reparam(roots, rho, deriv=2)
            theirs = r_bridge.source_gam_reparam(roots, rho, 2)
            label = f"{name} rho={rho.tolist()}"
            collector.check(
                f"{label} stages",
                partial(check_that, ours.stage_ranks == (), "transform taken"),
            )
            collector.check(f"{label} Qs", partial(_close, ours.Qs, theirs.Qs, "Qs"))
            collector.check(f"{label} S", partial(_close, ours.S, theirs.S, "S"))
            for j, (mine, source) in enumerate(zip(ours.rS, theirs.rS, strict=True)):
                collector.check(
                    f"{label} rS[{j}]", partial(_close, mine, source, f"rS[{j}]")
                )
            scale = np.sqrt(np.outer(np.diag(theirs.S), np.diag(theirs.S)))
            collector.check(
                f"{label} E'E",
                partial(
                    _close,
                    (ours.E.T @ ours.E) / scale,
                    (theirs.E.T @ theirs.E) / scale,
                    "E",
                ),
            )
    collector.raise_if_any("identity-transform gauge")


def test_derivative_order_matches_r(r_bridge) -> None:
    roots = _range_roots("te_cr_k5")
    rho = np.array([40.0, 0.0])
    for deriv in (0, 1, 2):
        ours = stable_reparam(roots, rho, deriv=deriv)
        theirs = r_bridge.source_gam_reparam(roots, rho, deriv)
        _close(ours.det, theirs.det, "det")
        assert (ours.det1 is None) == (deriv < 1)
        assert (theirs.det1 is None) == (deriv < 1)
        assert (ours.det2 is None) == (deriv < 2)
        assert (theirs.det2 is None) == (deriv < 2)
        if deriv:
            _close(ours.det1, theirs.det1, "det1")


def test_single_penalty_block_matches_r(r_bridge) -> None:
    """One penalty: Qs = I, S = sp S_1, det1 = q and det2 = 0, as in R."""
    root = _cubic_range_root(10)
    size = len(root)
    collector = _AssertCollector()
    for value in (-60.0, -5.0, 0.0, 7.0, 60.0):
        rho = np.array([value])
        ours = stable_reparam([root], rho, deriv=2)
        theirs = r_bridge.source_gam_reparam([root], rho, 2)
        label = f"rho={value}"
        for field in ("det", "det1", "det2", "Qs", "S"):
            collector.check(
                f"{label} {field}",
                partial(_close, getattr(ours, field), getattr(theirs, field), field),
            )
        collector.check(f"{label} rS", partial(_close, ours.rS[0], theirs.rS[0], "rS"))
        collector.check(f"{label} det1=q", partial(_close, ours.det1, [size], "q"))
        collector.check(f"{label} det2=0", partial(_close, ours.det2, [[0.0]], "0"))
        collector.check(f"{label} Qs=I", partial(_close, ours.Qs, np.eye(size), "I"))
    collector.raise_if_any("single penalty")


def test_disjoint_supports_separate_exactly_and_match_r(r_bridge) -> None:
    """Block-diagonal penalties: det splits into the two blocks' determinants."""
    left, right = _disjoint_roots()
    sizes = (left.shape[1], right.shape[1])
    half_dets = (
        np.linalg.slogdet(left[: sizes[0]] @ left[: sizes[0]].T)[1],
        np.linalg.slogdet(right[sizes[0] :] @ right[sizes[0] :].T)[1],
    )
    collector = _AssertCollector()
    for rho in _grid(2):
        ours = stable_reparam([left, right], rho, deriv=2)
        theirs = r_bridge.source_gam_reparam([left, right], rho, 2)
        label = f"rho={rho.tolist()}"
        for field in ("det", "det1", "det2"):
            collector.check(
                f"{label} {field}",
                partial(_close, getattr(ours, field), getattr(theirs, field), field),
            )
        expected = sizes[0] * rho[0] + sizes[1] * rho[1] + sum(half_dets)
        collector.check(f"{label} split det", partial(_close, ours.det, expected, "d"))
        collector.check(f"{label} det1", partial(_close, ours.det1, sizes, "det1"))
        collector.check(
            f"{label} det2", partial(_close, ours.det2, np.zeros((2, 2)), "det2")
        )
    collector.raise_if_any("disjoint supports")


def test_fixed_penalty_matches_r_and_is_left_untransformed(r_bridge) -> None:
    roots = _range_roots("te_cr_k5")
    fixed = _fixed_root(len(roots[0]))
    collector = _AssertCollector()
    for rho in (np.array([0.0, 0.0]), np.array([40.0, 0.0]), np.array([-20.0, 30.0])):
        ours = stable_reparam([*roots, fixed], rho, deriv=2)
        theirs = r_bridge.source_gam_reparam([*roots, fixed], rho, 2)
        label = f"rho={rho.tolist()}"
        collector.check(
            f"{label} flags",
            partial(
                check_that, ours.fixed_penalty and theirs.fixed_penalty, "fixed penalty"
            ),
        )
        for field in ("det", "det1", "det2"):
            collector.check(
                f"{label} {field}",
                partial(_close, getattr(ours, field), getattr(theirs, field), field),
            )
        collector.check(
            f"{label} fixed root",
            partial(
                check_that,
                np.array_equal(ours.rS[-1], fixed)
                and np.array_equal(theirs.rS[-1], fixed),
                "fixed root transformed",
            ),
        )
    collector.raise_if_any("fixed penalty")


# Multi-smooth models: per smooth either ("single", cr k) or a tensor block
# name.  Log sp differ by up to 60 across blocks; within a multi-penalty block
# no two log sp are exactly 10 apart (the reviewed grouped regime above).
_MODELS = {
    "s+te": (
        (("single", 10), "te_cr_k5"),
        [
            (0.0, 0.0, 0.0),
            (60.0, 0.0, 0.0),
            (0.0, 60.0, 0.0),
            (-60.0, 0.0, 40.0),
            (30.0, -30.0, 30.0),
            (-40.0, 20.0, -40.0),
        ],
    ),
    "s+ti+te3": (
        (("single", 8), "ti_tp_k6", "te_cr_k4_3d"),
        [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (60.0, 0.0, 5.0, -20.0, 0.0, 20.0),
            (-60.0, 40.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, -30.0, 10.0, 60.0, 40.0, 20.0),
            (20.0, 20.0, 20.0, -40.0, -40.0, -40.0),
            (-40.0, 60.0, 0.0, 30.0, -10.0, 10.0),
        ],
    ),
    "s+s+te": (
        (("single", 10), ("single", 6), "te_tp_k5"),
        [
            (0.0, 0.0, 0.0, 0.0),
            (60.0, -60.0, 0.0, 20.0),
            (-30.0, 30.0, 40.0, 0.0),
            (0.0, 0.0, -60.0, 0.0),
        ],
    ),
}


# Per-block sum against R's single global call: det1 at MODERATE, measured
# 1.46x STRICT at (60,0,5,-20,0,20) on this host, where a 110-digit evaluation
# of the global problem puts R's global det1 5.3e-10 off and the per-block sum
# 3.4e-15 off (stables-logs/perblock_truth.log); det2 at the floor class.


def _model_block_roots(model: str) -> list[tuple[np.ndarray, ...]]:
    return [
        (_cubic_range_root(block[1]),)
        if isinstance(block, tuple)
        else _range_roots(block)
        for block in _MODELS[model][0]
    ]


def _global_roots(blocks: list[tuple[np.ndarray, ...]]) -> list[np.ndarray]:
    """Embed every block's range roots in one global coordinate vector."""
    size = sum(roots[0].shape[0] for roots in blocks)
    result, offset = [], 0
    for roots in blocks:
        rows = roots[0].shape[0]
        for root in roots:
            embedded = np.zeros((size, root.shape[1]))
            embedded[offset : offset + rows] = root
            result.append(embedded)
        offset += rows
    return result


def _per_block_determinant(blocks, rho: np.ndarray):
    """Sum of per-block det, det1, det2; singleton blocks use rank*rho + const."""
    n_sp = len(rho)
    det, det1, det2 = 0.0, np.zeros(n_sp), np.zeros((n_sp, n_sp))
    start = 0
    for roots in blocks:
        index = slice(start, start + len(roots))
        if len(roots) == 1:
            rank = roots[0].shape[1]
            constant = np.linalg.slogdet(roots[0] @ roots[0].T)[1]
            det += rank * rho[start] + constant
            det1[start] = rank
        else:
            local = stable_reparam(roots, rho[index], deriv=2)
            det += local.det
            det1[index] = local.det1
            det2[index, index] = local.det2
        start += len(roots)
    return det, det1, det2


@pytest.mark.parametrize("model", list(_MODELS))
def test_per_block_determinants_sum_to_pinned_global_reparam(
    model: str, r_bridge
) -> None:
    """Justify the per-block call: it reproduces R's single global call.

    Pinned gam.fit3 calls gam.reparam once on all UrS; the sum of per-block
    det/det1/det2 (singletons exact) equals it: det at STRICT, det1 and det2 at
    the classes stated above.
    """
    blocks = _model_block_roots(model)
    roots = _global_roots(blocks)
    collector = _AssertCollector()
    for values in _MODELS[model][1]:
        rho = np.asarray(values)
        det, det1, det2 = _per_block_determinant(blocks, rho)
        theirs = r_bridge.source_gam_reparam(roots, rho, 2)
        label = f"{model} rho={values}"
        collector.check(f"{label} det", partial(_close, det, theirs.det, "det"))
        collector.check(
            f"{label} det1", partial(_close_at, det1, theirs.det1, MODERATE, "det1")
        )
        collector.check(
            f"{label} det2", partial(_close_at, det2, theirs.det2, _DET2, "det2")
        )
    collector.raise_if_any(f"{model} per-block vs global")


def _null_dimension(S: np.ndarray) -> int:
    """Null dimension of S after gam.reparam's diagonal pre-conditioning."""
    diagonal = np.sqrt(np.abs(np.diag(S)))
    diagonal[diagonal == 0.0] = 1.0
    values = np.linalg.eigvalsh(S / np.outer(diagonal, diagonal))
    return int(np.sum(values <= values[-1] * R_TOL))


def _zero_rows(roots) -> list[tuple[int, ...]]:
    return [tuple(np.flatnonzero(~root.any(axis=1))) for root in roots]


def test_shared_null_space_keeps_r_rank_structure(r_bridge) -> None:
    """Unprojected te roots: same stage zero rows, rank and null dimension as R."""
    roots = _raw_roots("te_cr_k5")
    collector = _AssertCollector()
    for rho in (np.zeros(2), np.array([40.0, 0.0]), np.array([0.0, 60.0])):
        ours = stable_reparam(roots, rho, deriv=0)
        theirs = r_bridge.source_gam_reparam(roots, rho, 0)
        label = f"rho={rho.tolist()}"
        collector.check(
            f"{label} null dimension",
            partial(
                check_that,
                _null_dimension(ours.S) == _null_dimension(theirs.S) == 3,
                f"{_null_dimension(ours.S)} vs {_null_dimension(theirs.S)}",
            ),
        )
        collector.check(
            f"{label} stage zero rows",
            partial(
                check_that,
                _zero_rows(ours.rS) == _zero_rows(theirs.rS),
                "zero rows differ",
            ),
        )
    collector.raise_if_any("shared null space")


# --------------------------------------------------------------------------- #
#  Gauge-invariant structure (R-free)                                          #
# --------------------------------------------------------------------------- #


def _scaled_close(actual, expected, scale: float, label: str) -> None:
    _close(np.asarray(actual) / scale, np.asarray(expected) / scale, label)


def _group_scale(index: int, rho: np.ndarray, penalties) -> float:
    """Accuracy scale of penalty ``index`` in S_index units.

    get_stableS separates penalties only beyond the d_tol ratio, so a penalty
    inherits roundoff from the largest weighted penalty it can be grouped
    with: max over k of sp_k |S_k| with sp_k |S_k| <= sp_j |S_j| / d_tol.
    """
    weights = np.exp(rho) * np.array([np.linalg.norm(S) for S in penalties])
    groupable = weights[weights <= weights[index] / D_TOL]
    return float(np.max(groupable) / np.exp(rho[index]))


@pytest.mark.parametrize("name", list(_BLOCKS))
def test_transform_is_orthogonal_similarity_of_each_penalty(name: str) -> None:
    roots = _range_roots(name)
    penalties = [root @ root.T for root in roots]
    collector = _AssertCollector()
    for rho in _grid(len(roots)):
        result = stable_reparam(roots, rho, deriv=0)
        label = f"{name} rho={rho.tolist()}"
        Qs = result.Qs
        collector.check(
            f"{label} Qs'Qs", partial(_close, Qs.T @ Qs, np.eye(len(Qs)), "Qs")
        )
        for j, (S_j, root) in enumerate(zip(penalties, result.rS, strict=True)):
            collector.check(
                f"{label} S_{j}",
                partial(
                    _scaled_close,
                    Qs.T @ S_j @ Qs,
                    root @ root.T,
                    _group_scale(j, rho, penalties),
                    "S_j",
                ),
            )
        total = sum(np.exp(r) * S_j for r, S_j in zip(rho, penalties, strict=True))
        collector.check(
            f"{label} S",
            partial(
                _scaled_close, Qs.T @ total @ Qs, result.S, np.linalg.norm(total), "S"
            ),
        )
        scale = np.sqrt(np.outer(np.diag(result.S), np.diag(result.S)))
        collector.check(
            f"{label} E'E",
            partial(_close, (result.E.T @ result.E) / scale, result.S / scale, "E"),
        )
        collector.check(
            f"{label} symmetric",
            partial(check_that, np.array_equal(result.S, result.S.T), "S asym"),
        )
    collector.raise_if_any(f"{name} structure")


def test_stage_ranks_follow_dominance_order() -> None:
    """te: dominant penalty's rank is peeled first; equal sp groups together."""
    roots = _range_roots("te_cr_k5")
    rank = roots[0].shape[1]
    assert stable_reparam(roots, [0.0, 0.0], deriv=0).stage_ranks == ()
    assert stable_reparam(roots, [40.0, 0.0], deriv=0).stage_ranks == (rank,)
    assert stable_reparam(roots, [0.0, 40.0], deriv=0).stage_ranks == (rank,)
    three = _range_roots("te_cr_k4_3d")
    stages = stable_reparam(three, [40.0, 0.0, -40.0], deriv=0).stage_ranks
    assert len(stages) == 2
    assert stages[0] == three[0].shape[1]


# --------------------------------------------------------------------------- #
#  Derivative consistency (R-free)                                             #
# --------------------------------------------------------------------------- #


def _richardson(function, rho: np.ndarray, index: int, step: float) -> np.ndarray:
    """Central differences at h, h/2, h/4 with two Richardson levels: O(h^6)."""

    def central(h: float) -> np.ndarray:
        offset = np.zeros_like(rho)
        offset[index] = h
        return (np.asarray(function(rho + offset)) - function(rho - offset)) / (2 * h)

    first = [central(step / 2**level) for level in range(3)]
    second = [(4 * first[i + 1] - first[i]) / 3 for i in range(2)]
    return (16 * second[1] - second[0]) / 15


@pytest.mark.parametrize(
    ("name", "rho"),
    [
        ("te_cr_k5", (0.0, 0.0)),
        ("te_cr_k5", (2.0, -1.0)),
        ("te_cr_k5", (40.0, 0.0)),
        ("te_tp_k5", (0.0, 40.0)),
        ("ti_cr_k5", (-20.0, 20.0)),
        ("te_cr_k4_3d", (40.0, 0.0, -1.0)),
    ],
)
def test_finite_differences_agree_with_analytic_derivatives(
    name: str, rho: tuple[float, ...]
) -> None:
    roots = _range_roots(name)
    point = np.asarray(rho)
    result = stable_reparam(roots, point, deriv=2)
    assert result.det1 is not None
    assert result.det2 is not None
    step = 0.05
    gradient = np.array(
        [
            _richardson(lambda r: stable_reparam(roots, r, 0).det, point, i, step)
            for i in range(len(point))
        ]
    )
    hessian = np.column_stack(
        [
            _richardson(lambda r: stable_reparam(roots, r, 1).det1, point, i, step)
            for i in range(len(point))
        ]
    )
    _close(gradient, result.det1, "finite-difference det1")
    _close(hessian, result.det2, "finite-difference det2")


# --------------------------------------------------------------------------- #
#  Input validation                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("roots", "rho", "deriv"),
    [
        ([np.eye(3)], [0.0], 3),
        ([], [], 0),
        ([np.ones(3)], [0.0], 0),
        ([np.eye(3), np.eye(4)], [0.0, 0.0], 0),
        ([np.eye(3)], [0.0, 0.0], 0),
        ([np.full((3, 3), np.nan)], [0.0], 0),
        ([np.eye(3)], [800.0], 0),
    ],
)
def test_invalid_inputs_raise(roots, rho, deriv) -> None:
    with pytest.raises(ValueError, match=r"."):
        stable_reparam(roots, rho, deriv=deriv)


def test_result_is_frozen_and_inputs_are_not_modified() -> None:
    roots = _range_roots("te_cr_k5")
    before = [root.copy() for root in roots]
    result = stable_reparam(roots, [40.0, 0.0], deriv=2)
    assert isinstance(result, StableReparam)
    assert all(np.array_equal(a, b) for a, b in zip(roots, before, strict=True))
    with pytest.raises(AttributeError):
        result.det = 0.0  # type: ignore[misc]


def test_pinned_provider_branch_gives_same_determinants(monkeypatch) -> None:
    """Off Accelerate both DSYEVR calls use the pinned provider (as on Linux)."""
    import jaxgam.penalties.stable_reparam as module

    roots = _range_roots("te_cr_k4_3d")
    collector = _AssertCollector()
    for rho in (
        np.zeros(3),
        np.array([40.0, 0.0, -40.0]),
        np.array([-20.0, 20.0, 0.0]),
    ):
        monkeypatch.setattr(module, "scipy_uses_accelerate_lapack", lambda: True)
        scipy_result = stable_reparam(roots, rho, deriv=2)
        monkeypatch.setattr(module, "scipy_uses_accelerate_lapack", lambda: False)
        provider_result = stable_reparam(roots, rho, deriv=2)
        label = f"rho={rho.tolist()}"
        for field in ("det", "det1", "det2"):
            collector.check(
                f"{label} {field}",
                partial(
                    _close,
                    getattr(provider_result, field),
                    getattr(scipy_result, field),
                    field,
                ),
            )
        collector.check(
            f"{label} stages",
            partial(
                check_that,
                provider_result.stage_ranks == scipy_result.stage_ranks,
                "stage ranks differ",
            ),
        )
        Qs = provider_result.Qs
        collector.check(
            f"{label} Qs'Qs", partial(_close, Qs.T @ Qs, np.eye(len(Qs)), "Qs")
        )
    collector.raise_if_any("pinned provider branch")


def test_global_penalty_rotation_delegates_to_stable_reparam() -> None:
    """fitting_prepare's global rotation is this port's Qs and roots, bitwise."""
    from jaxgam.formula.fitting_prepare import _stable_range_rotation

    roots = _range_roots("te_cr_k5")
    for rho in (np.zeros(2), np.array([40.0, 0.0])):
        qs, transformed = _stable_range_rotation(roots, rho)
        expected = stable_reparam(roots, rho, deriv=0)
        assert np.array_equal(qs, expected.Qs)
        assert all(
            np.array_equal(a, b) for a, b in zip(transformed, expected.rS, strict=True)
        )


# --------------------------------------------------------------------------- #
#  Exactly representable family with an exact rational oracle                 #
# --------------------------------------------------------------------------- #
#
# Integer penalty roots (difference matrices and Kronecker products with
# identities) and log smoothing parameters k*log(2): the input bytes are the
# same on every platform.  For a block with integer range basis Y (exact
# rational null space of N', scaled to integers), the port receives the integer
# roots Y'rS_j, so its S is A = Y'SY and its det is log det(A); the oracle
# evaluates det = log det(A), det1_j = sp_j tr(A^-1 Y'S_jY) and
# det2_ij = -sp_i sp_j tr(A^-1 Y'S_iY A^-1 Y'S_jY) + delta_ij det1_i exactly in
# rationals at the port's own sp = exp(rho) (an exact binary fraction), and
# rounds once to float.  log pdet(S) differs from det only by the
# rho-independent constant log det(Y'Y).

_LN2 = float.fromhex("0x1.62e42fefa39efp-1")  # log(2) rounded to nearest


def _difference(k: int, order: int) -> np.ndarray:
    matrix = np.eye(k, dtype=np.int64)
    for _ in range(order):
        matrix = matrix[1:] - matrix[:-1]
    return matrix


def _difference_null(k: int, order: int) -> np.ndarray:
    """Integer basis of null(D): the monomials 1, x, ... of degree < order."""
    x = np.arange(k, dtype=np.int64)
    return np.column_stack([x**power for power in range(order)])


def _exact_nullspace(matrix: np.ndarray) -> np.ndarray:
    """Integer basis (columns) of {v : matrix v = 0} by exact rational RREF."""
    rows = [[Fraction(int(value)) for value in row] for row in matrix]
    n_cols = len(rows[0])
    pivots: list[int] = []
    rank = 0
    for col in range(n_cols):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][col] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        lead = rows[rank][col]
        rows[rank] = [value / lead for value in rows[rank]]
        for r in range(len(rows)):
            if r != rank and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [
                    a - factor * b for a, b in zip(rows[r], rows[rank], strict=True)
                ]
        pivots.append(col)
        rank += 1
    basis = []
    for free in (c for c in range(n_cols) if c not in pivots):
        vector = [Fraction(0)] * n_cols
        vector[free] = Fraction(1)
        for r, col in enumerate(pivots):
            vector[col] = -rows[r][free]
        scale = math.lcm(*(value.denominator for value in vector))
        basis.append([int(value * scale) for value in vector])
    return np.array(basis, dtype=np.int64).T


def _kron_all(*factors: np.ndarray) -> np.ndarray:
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


@cache
def _exact_block(name: str) -> tuple[np.ndarray, ...]:
    """Integer range-space roots Y'rS_j of one exactly representable block."""
    if name == "x_te_d2":  # 2 penalties, shared null space (dimension 4)
        d1, d2 = _difference(5, 2), _difference(4, 2)
        roots = (
            _kron_all(d1.T, np.eye(4, dtype=np.int64)),
            _kron_all(np.eye(5, dtype=np.int64), d2.T),
        )
        null = _kron_all(_difference_null(5, 2), _difference_null(4, 2))
    elif name == "x_te3_d1":  # 3 penalties, shared null space (constants)
        d, eye = _difference(3, 1), np.eye(3, dtype=np.int64)
        roots = (
            _kron_all(d.T, eye, eye),
            _kron_all(eye, d.T, eye),
            _kron_all(eye, eye, d.T),
        )
        null = _kron_all(*(_difference_null(3, 1),) * 3)
    elif name == "x_single":  # 1 penalty: first differences, null = constants
        roots = (_difference(5, 1).T,)
        null = _difference_null(5, 1)
    elif name == "x_sep2":  # 2 penalties, separate null spaces, full rank
        roots = (_difference(6, 1).T, np.ones((6, 1), dtype=np.int64))
        null = None
    elif name == "x_sep3":  # 3 penalties, separate null spaces, full rank
        roots = (
            _difference(5, 1).T,
            np.ones((5, 1), dtype=np.int64),
            np.arange(5, dtype=np.int64)[:, None],
        )
        null = None
    else:
        raise KeyError(name)
    if null is None:
        return tuple(np.asarray(root, dtype=np.float64) for root in roots)
    range_basis = _exact_nullspace(null.T)
    return tuple(np.asarray(range_basis.T @ root, dtype=np.float64) for root in roots)


_EXACT_BLOCKS = ("x_te_d2", "x_te3_d1", "x_sep2", "x_sep3")


def _exact_grid(n_penalties: int) -> list[tuple[int, ...]]:
    """Exponents e (log sp = e log 2): equal, gaps 7/14/29/58/87, very negative."""
    points = [(value,) * n_penalties for value in (0, 4, -58, -87)]
    for gap in (7, 14, 29, 58, 87):
        if n_penalties == 2:
            points += [(gap, 0), (0, gap), (-43, -43 - gap), (-43 - gap, -43)]
        else:
            points += sorted(set(itertools.permutations((-gap, 0, gap))))
            points.append((-72, -72, -72 - gap))
    return points


def _exact_rho(exponents) -> np.ndarray:
    return np.array([e * _LN2 for e in exponents])


def _bareiss_solve(matrix: list[list[int]], rhs: list[list[int]]):
    """Fraction-free Gaussian elimination (Bareiss) of an integer SPD system.

    Returns ``(d, X)`` with ``d = det(matrix)`` and the integer matrix
    ``X = d * matrix^-1 rhs``.  ``matrix`` is positive definite, so the
    Bareiss pivots (its leading principal minors) are positive and no
    pivoting is needed; every division below is exact.
    """
    size = len(matrix)
    rows = [list(a) + list(b) for a, b in zip(matrix, rhs, strict=True)]
    previous = 1
    for k in range(size):
        pivot = rows[k][k]
        if pivot <= 0:
            raise ZeroDivisionError("exact oracle matrix is not positive definite")
        row_k = rows[k]
        for i in range(k + 1, size):
            row_i = rows[i]
            factor = row_i[k]
            rows[i] = [
                (pivot * a - factor * b) // previous
                for a, b in zip(row_i, row_k, strict=True)
            ]
        previous = pivot
    determinant = rows[size - 1][size - 1]
    n_rhs = len(rhs[0])
    solution = [[0] * n_rhs for _ in range(size)]
    for i in range(size - 1, -1, -1):
        row = rows[i]
        for c in range(n_rhs):
            total = determinant * row[size + c] - sum(
                row[j] * solution[j][c] for j in range(i + 1, size)
            )
            value, remainder = divmod(total, row[i])
            if remainder:
                raise ArithmeticError("Bareiss back substitution is not exact")
            solution[i][c] = value
    return determinant, solution


@cache
def _exact_oracle(roots_key: tuple, exponents: tuple[int, ...]):
    """Exact (det rounded once, det1, det2) for integer roots at sp = exp(rho).

    Each sp is an exact binary fraction; scaling by the common power-of-two
    denominator ``D`` makes ``A_int = D * A`` an integer matrix, solved
    exactly by Bareiss: ``d = det(A_int)``, ``X_j = d A_int^-1 (D C_j)`` so
    ``A^-1 C_j = X_j / d``.  Only the final quotients are rationals.
    """
    roots = [np.array(root).astype(np.int64) for root in roots_key]
    sp = [Fraction(float(value)) for value in libm_exp(_exact_rho(exponents))]
    denominator = math.lcm(*(value.denominator for value in sp))
    weights = [int(value * denominator) for value in sp]
    components = [
        [[int(v) for v in row] for row in (root @ root.T).tolist()] for root in roots
    ]
    size, n_sp = len(components[0]), len(sp)
    matrix = [
        [
            sum(weights[j] * components[j][a][b] for j in range(n_sp))
            for b in range(size)
        ]
        for a in range(size)
    ]
    rhs = [
        [denominator * components[j][a][b] for j in range(n_sp) for b in range(size)]
        for a in range(size)
    ]
    determinant, solution = _bareiss_solve(matrix, rhs)
    blocks = [
        [row[j * size : (j + 1) * size] for row in solution] for j in range(n_sp)
    ]  # d * A^-1 Y'S_jY
    traces = [sum(blocks[j][a][a] for a in range(size)) for j in range(n_sp)]
    det1 = [sp[j] * Fraction(traces[j], determinant) for j in range(n_sp)]
    det2 = [[Fraction(0)] * n_sp for _ in range(n_sp)]
    for i in range(n_sp):
        for j in range(i, n_sp):
            trace = sum(
                blocks[i][a][b] * blocks[j][b][a]
                for a in range(size)
                for b in range(size)
            )
            value = -sp[i] * sp[j] * Fraction(trace, determinant * determinant)
            if i == j:
                value += det1[i]
            det2[i][j] = det2[j][i] = value
    with localcontext() as context:
        context.prec = 60
        # det(A) = det(A_int) / D^size with D a power of two.
        log_det = Decimal(determinant).ln() - size * Decimal(denominator).ln()
    return (
        float(log_det),
        np.array([float(v) for v in det1]),
        np.array([[float(v) for v in row] for row in det2]),
    )


def _roots_key(roots) -> tuple:
    return tuple(tuple(map(tuple, np.asarray(root).tolist())) for root in roots)


def _close_at(actual, expected, tolerance: ToleranceClass, label: str) -> None:
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance.rtol, atol=tolerance.atol, err_msg=label
    )


@pytest.mark.parametrize("name", _EXACT_BLOCKS)
def test_exact_family_matches_rational_oracle(name: str) -> None:
    """det at STRICT, det1/det2 at the floor class, against exact values.

    det1 is also at MODERATE here: at the grouped points (sp ratio 2^14) it is
    4.7x STRICT from the exact value for the port and 3.3x for pinned R.
    """
    roots = _exact_block(name)
    key = _roots_key(roots)
    collector = _AssertCollector()
    for exponents in _exact_grid(len(roots)):
        exact = _exact_oracle(key, exponents)
        ours = stable_reparam(roots, _exact_rho(exponents), deriv=2)
        for field, actual, expected in zip(
            ("det", "det1", "det2"),
            (ours.det, ours.det1, ours.det2),
            exact,
            strict=True,
        ):
            tolerance = STRICT if field == "det" else MODERATE
            collector.check(
                f"{name} e={exponents} {field} ({tolerance.label})",
                partial(_close_at, actual, expected, tolerance, field),
            )
    collector.raise_if_any(f"{name} exact oracle")


def test_exact_family_pinned_r_is_within_the_floor_class(r_bridge) -> None:
    """Pinned R itself sits within the floor class of the exact value."""
    collector = _AssertCollector()
    for name in _EXACT_BLOCKS:
        roots = _exact_block(name)
        key = _roots_key(roots)
        for exponents in _exact_grid(len(roots)):
            _, det1, det2 = _exact_oracle(key, exponents)
            theirs = r_bridge.source_gam_reparam(roots, _exact_rho(exponents), 2)
            label = f"{name} e={exponents}"
            collector.check(
                f"{label} det1", partial(_close_at, theirs.det1, det1, MODERATE, "det1")
            )
            collector.check(
                f"{label} det2", partial(_close_at, theirs.det2, det2, _DET2, "det2")
            )
    collector.raise_if_any("pinned R against exact values")


_EXACT_MULTI_BLOCK = ("x_single", "x_te_d2", "x_sep3")
_EXACT_MULTI_POINTS = (
    (0, 0, 0, 0, 0, 0),
    (87, 0, 29, -29, 0, 29),
    (-87, 58, 0, 0, 0, 0),
    (0, -43, -72, 87, 58, 29),
    (-58, 87, 0, 43, -14, 14),
    (29, -87, -43, 0, 58, -29),
)


def test_exact_multi_block_per_block_and_global_match_exact_values() -> None:
    """Per-block sum and the port's global call both equal exact rationals.

    Against a 110-digit reference the per-block sum was the more accurate side
    at three of four smooth-constructor points (comment above
    test_per_block_determinants_sum_to_pinned_global_reparam).
    """
    blocks = [_exact_block(name) for name in _EXACT_MULTI_BLOCK]
    roots = _global_roots(blocks)
    key = _roots_key(roots)
    collector = _AssertCollector()
    for exponents in _EXACT_MULTI_POINTS:
        rho = _exact_rho(exponents)
        exact = _exact_oracle(key, exponents)
        per_block = _per_block_determinant(blocks, rho)
        whole = stable_reparam(roots, rho, deriv=2)
        for label, values in (
            ("per-block", per_block),
            ("global", (whole.det, whole.det1, whole.det2)),
        ):
            for field, actual, expected in zip(
                ("det", "det1", "det2"), values, exact, strict=True
            ):
                tolerance = _DET2 if field == "det2" else STRICT
                collector.check(
                    f"e={exponents} {label} {field} ({tolerance.label})",
                    partial(_close_at, actual, expected, tolerance, field),
                )
    collector.raise_if_any("exact multi-block")
