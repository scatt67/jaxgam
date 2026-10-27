"""Dense fixed-sp numerics against pinned mgcv: structural null space, no jitter.

mgcv's ``gam.fit3`` keeps the penalty null space structurally unpenalized
(``totalPenaltySpace`` + ``Mp`` zero padding) and never adds a diagonal
jitter (``pls_fit1``/``gdiPK`` in gdi.c).  These tests own the dense fixed-sp
PIRLS numerics: JaxGAM's production dense route is fed R's exact model matrix
and penalties (typed ``RBridge.get_smooth_components``) or uses a basis that
equals R's, and must reproduce R's fixed-sp fits at STRICT, including
smoothing parameters up to ``exp(40)``.  Broad final-model parity stays in
``tests/test_validation_matrix.py``.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam import GAM
from jaxgam.api import _fit_fixed_sp
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting import penalty_ops
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.formula.fitting_prepare import reparameterize_structure
from jaxgam.jax_utils import _CHOLESKY_SMALL_RELATIVE_JITTER, cho_factor
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
    make_penalty_structure,
)
from tests.helpers import _AssertCollector, check_that, r_available
from tests.tolerances import STRICT

_LOG_SP_SWEEP = (0.0, 10.0, 20.0, 30.0, 40.0)


def _assert_strict(actual, expected) -> None:
    np.testing.assert_allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)


def _psd(rng: np.random.Generator, size: int, rank: int) -> np.ndarray:
    basis = rng.standard_normal((size, rank))
    return basis @ basis.T


# ---------------------------------------------------------------------------
# Phase 1: structural reparameterization (no R)
# ---------------------------------------------------------------------------


def test_reparameterized_penalties_have_exact_zero_null_blocks() -> None:
    """Null rows/columns are exact zeros; range blocks equal ``D' S D``."""
    rng = np.random.default_rng(20261027)
    single = _psd(rng, 7, 5)
    shared, _ = np.linalg.qr(rng.standard_normal((6, 4)))  # common 2-dim null
    overlap = (
        shared @ _psd(rng, 4, 3) @ shared.T,
        shared @ _psd(rng, 4, 4) @ shared.T * 1e3,
    )
    by_a, by_b = np.zeros((8, 8)), np.zeros((8, 8))
    by_a[:4, :4] = _psd(rng, 4, 3)
    by_b[4:, 4:] = _psd(rng, 4, 2)
    structure = PenaltyStructure(
        23,
        (
            PenaltyBlock(
                0, 7, (0,), (DenseLocalPenalty(single),), IdentityTransform(7), (5,)
            ),
            PenaltyBlock(
                7,
                13,
                (1, 2),
                tuple(DenseLocalPenalty(S) for S in overlap),
                IdentityTransform(6),
                (3, 4),
            ),
            PenaltyBlock(
                13,
                21,
                (3, 4),
                (DenseLocalPenalty(by_a), DenseLocalPenalty(by_b)),
                IdentityTransform(8),
                (3, 2),
            ),
        ),
    )
    transformed = reparameterize_structure(structure)
    collector = _AssertCollector()
    for block, original in zip(transformed.blocks, structure.blocks, strict=True):
        D = block.transform.dense()
        local = block.dense_penalties()
        total = np.add.reduce([np.abs(S) for S in local])
        null = np.flatnonzero(np.diag(total) == 0.0)

        def _null_exact(local=local, null=null, block=block) -> None:
            check_that(len(null) > 0, f"block {block.start}: no null directions")
            for S in local:
                np.testing.assert_array_equal(S[null, :], 0.0)
                np.testing.assert_array_equal(S[:, null], 0.0)

        def _range_consistent(D=D, local=local, original=original, null=null) -> None:
            for S_new, S_old in zip(local, original.dense_penalties(), strict=True):
                expected = D.T @ S_old @ D
                keep = np.setdiff1d(np.arange(len(D)), null)
                np.testing.assert_allclose(
                    S_new[np.ix_(keep, keep)],
                    expected[np.ix_(keep, keep)],
                    rtol=STRICT.rtol,
                    atol=STRICT.rtol * np.max(np.abs(expected)),
                )

        def _ranks(block=block, local=local) -> None:
            for S, rank in zip(local, block.ranks, strict=True):
                check_that(
                    int(np.linalg.matrix_rank(S)) == rank,
                    f"structural rank {rank} disagrees with the local penalty",
                )

        collector.check(f"block {block.start}: exact null", _null_exact)
        collector.check(f"block {block.start}: range = D'SD", _range_consistent)
        collector.check(f"block {block.start}: ranks", _ranks)
    overlap_D = transformed.blocks[1].transform.dense()
    collector.check(
        "overlapping transform is orthogonal",
        lambda: _assert_strict(overlap_D.T @ overlap_D, np.eye(6)),
    )
    collector.raise_if_any("structural reparameterization")


# ---------------------------------------------------------------------------
# Phase 2: unjittered factors, explicit fallback diagnostic, JIT
# ---------------------------------------------------------------------------


def test_cho_factor_and_pirls_compile_and_report_regularization_under_jit() -> None:
    """Touched kernels compile; a PD fit is unregularized, a singular one is not."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((40, 5))
    y = X @ rng.standard_normal(5) + 0.1 * rng.standard_normal(40)
    S = np.diag([0.0, 0.0, 1.0, 2.0, 3.0])

    factor = jax.jit(cho_factor)
    L, level = factor(jnp.asarray(X.T @ X + S))
    assert float(level) == 0.0
    assert np.all(np.isfinite(np.asarray(L)))

    fit = jax.jit(
        lambda X_, y_, S_: pirls_loop(X_, y_, jnp.zeros(X_.shape[1]), S_, Gaussian())
    )
    regular = fit(jnp.asarray(X), jnp.asarray(y), jnp.asarray(S))
    assert bool(regular.converged)
    assert float(regular.factor_regularization) == 0.0

    aliased = np.column_stack([X, X[:, 0]])
    singular = fit(
        jnp.asarray(aliased), jnp.asarray(y), jnp.asarray(np.pad(S, ((0, 1), (0, 1))))
    )
    assert float(singular.factor_regularization) == _CHOLESKY_SMALL_RELATIVE_JITTER
    _assert_strict(np.asarray(singular.mu), np.asarray(regular.mu))


def test_dense_fit_warns_only_when_the_fallback_regularizes() -> None:
    """The fallback jitter is an explicit RuntimeWarning, never silent."""
    rng = np.random.default_rng(11)
    n = 80
    x = np.sort(rng.uniform(0, 1, n))
    data = pd.DataFrame(
        {"x": x, "z": rng.uniform(0, 1, n), "y": np.sin(4 * x) + rng.normal(0, 0.2, n)}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GAM("y ~ s(x, bs='cr', k=8)", sp=[0.5]).fit(data)
    assert not [w for w in caught if "relative diagonal jitter" in str(w.message)]

    # A parametric z confounded with the unpenalized linear null space of
    # s(z) leaves H singular (FittingData.rank_deficit == 1).
    with pytest.warns(RuntimeWarning, match="relative diagonal jitter of 1e-12"):
        fit = GAM("y ~ z + s(z, bs='cr', k=6)", sp=[0.5]).fit(data)
    assert np.all(np.isfinite(fit.fitted_values))


# ---------------------------------------------------------------------------
# R parity at STRICT (typed RBridge)
# ---------------------------------------------------------------------------


def _parity_frame(layout: str) -> pd.DataFrame:
    """PR12 TPRS source-parity layouts (regular and irregular covariates)."""
    rng = np.random.default_rng(92613)
    regular = np.linspace(-1.3, 1.7, 61)
    irregular = np.sort(rng.uniform(-1.7, 2.4, 61))
    x = {"regular": regular, "irregular": irregular}[layout]
    z = np.sin(np.linspace(-0.8, 2.2, 61)) + np.linspace(0, 0.1, 61)
    return pd.DataFrame({"x": x, "z": z, "y": np.cos(1.7 * x) + 0.2 * z})


def _dense_fit_on_r_design(components: dict, y: np.ndarray, sp: np.ndarray):
    """Run JaxGAM's production dense fixed-sp route on R's exact X and S."""
    X = np.asarray(components["model_matrix"], dtype=np.float64)
    widths = [basis.shape[1] for basis in components["basis_matrices"]]
    start = X.shape[1] - sum(widths)
    terms, penalties, sp_index = [], [], 0
    for width, smooth_penalties in zip(
        widths, components["penalty_matrices"], strict=True
    ):
        count = len(smooth_penalties)
        terms.append(
            SimpleNamespace(
                col_start=start,
                n_coefs=width,
                penalty_indices=list(range(sp_index, sp_index + count)),
            )
        )
        penalties.append([0.5 * (S + S.T) for S in smooth_penalties])
        start += width
        sp_index += count
    n = len(y)
    setup = SimpleNamespace(
        X=X,
        y=np.asarray(y, dtype=np.float64),
        weights=np.ones(n),
        offset=None,
        n_obs=n,
        penalties=make_penalty_structure(X.shape[1], terms, penalties),
    )
    fd = FittingData.from_setup(setup, Gaussian())
    fit = _fit_fixed_sp(fd, np.asarray(sp, dtype=np.float64))
    pirls = fit.pirls_result
    coefficients = np.asarray(
        penalty_ops.transform_coefficients(fd.penalty_structure, pirls.coefficients)
    )
    return fit, coefficients, np.asarray(pirls.mu), float(pirls.deviance)


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize(("layout", "term"), [("regular", "s(x,bs='tp',k=4)")])
def test_r_exact_design_fixed_sp_matches_pinned_r(
    r_bridge, layout: str, term: str
) -> None:
    """R's exact X and S through the dense fixed-sp route reproduce R at sp 0.7."""
    data = _parity_frame(layout)
    with contextlib.redirect_stdout(io.StringIO()):
        components = r_bridge.get_smooth_components(
            "y~" + term, data, "gaussian", "REML"
        )
    sp = np.full(sum(len(p) for p in components["penalty_matrices"]), 0.7)
    with contextlib.redirect_stdout(io.StringIO()):
        reference = r_bridge.fit_fixed_sp("y~" + term, data, "gaussian", sp)
    fit, coefficients, fitted, deviance = _dense_fit_on_r_design(
        components, data.y.to_numpy(), sp
    )
    collector = _AssertCollector()
    collector.check(
        "converged",
        lambda: check_that(
            bool(fit.converged) and reference["converged"], "not converged"
        ),
    )
    collector.check(
        "fitted", lambda: _assert_strict(fitted, reference["fitted_values"])
    )
    collector.check(
        "coefficients", lambda: _assert_strict(coefficients, reference["coefficients"])
    )
    collector.check("deviance", lambda: _assert_strict(deviance, reference["deviance"]))
    collector.check(
        "unregularized",
        lambda: check_that(
            float(fit.pirls_result.factor_regularization) == 0.0, "fallback fired"
        ),
    )
    collector.raise_if_any(f"R-exact dense fixed sp {layout} {term}")


def _sweep_frame() -> pd.DataFrame:
    rng = np.random.default_rng(4101)
    x = np.sort(rng.uniform(-1.0, 1.0, 120))
    signal = 0.4 + 0.6 * np.sin(3.0 * x)
    return pd.DataFrame(
        {
            "x": x,
            "gaussian": signal + rng.normal(scale=0.2, size=x.size),
            "count": rng.poisson(np.exp(signal)).astype(float),
            "gamma": rng.gamma(4.0, np.exp(signal) / 4.0),
            "binary": (rng.uniform(size=x.size) < 1 / (1 + np.exp(-signal))).astype(
                float
            ),
        }
    )


def _assert_hard_gates(
    fit, label: str, collector: _AssertCollector, *, must_converge: bool = True
) -> None:
    """Hard-gate invariants (design.md §18.1) for one fitted model.

    ``must_converge`` is R's own inner-fit status where one exists: at
    smoothing parameters near exp(40) the gam.fit3 gradient test is limited by
    the roundoff of ``lambda S beta`` and pinned R reports non-convergence.
    """

    def _gates() -> None:
        check_that(fit.converged or not must_converge, "not converged")
        check_that(fit.deviance >= 0, "negative deviance")
        for name in ("coefficients", "fitted_values", "linear_predictor"):
            check_that(np.all(np.isfinite(getattr(fit, name))), f"non-finite {name}")
        p = len(fit.coefficients)
        check_that(
            0 <= fit.edf_total <= p * (1 + STRICT.rtol),
            f"EDF {fit.edf_total} outside [0, {p}]",
        )
        Vp = np.asarray(fit.Vp)
        np.testing.assert_allclose(Vp, Vp.T, rtol=STRICT.rtol, atol=STRICT.atol)
        smallest = np.min(np.linalg.eigvalsh(0.5 * (Vp + Vp.T)))
        check_that(smallest >= -STRICT.rtol * np.max(np.abs(Vp)), "Vp not PSD")

    collector.check(f"{label}: hard gates", _gates)


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize(
    ("family", "r_family", "link", "column"),
    [
        (Gaussian, "gaussian", "identity", "gaussian"),
        (lambda: Gamma(link="log"), "gamma", "log", "gamma"),
        (Poisson, "poisson", "log", "count"),
        (Binomial, "binomial", "logit", "binary"),
    ],
    ids=["gaussian", "gamma-log", "poisson", "binomial"],
)
def test_supplied_sp_sweep_to_exp40_matches_pinned_r(
    r_bridge, family, r_family: str, link: str, column: str
) -> None:
    """Supplied sp from exp(0) to exp(40): every public field STRICT against R.

    Before the structural null space and the unjittered factor, a trace-scaled
    jitter grew with sp and shrank the unpenalized coefficients (Gaussian
    coefficient gap 2e-5 at log sp 20, 0.65 at 40; score 117.4 vs 41.6).
    """
    frame = _sweep_frame()
    data = pd.DataFrame({"x": frame.x, "y": frame[column]})
    n = len(data)
    collector = _AssertCollector()
    for log_sp in _LOG_SP_SWEEP:
        sp = float(np.exp(log_sp))
        with contextlib.redirect_stdout(io.StringIO()):
            reference = r_bridge.public_fixed_sp_fit(
                data, np.ones(n), np.zeros(n), r_family, link, sp=sp
            )
        fit = GAM('y ~ s(x, bs="cr", k=8)', family=family(), sp=[sp]).fit(data)
        label = f"log sp {log_sp:+.0f}"
        for name, actual, expected in (
            ("coefficients", fit.coefficients, reference["coefficients"]),
            ("fitted", fit.fitted_values, reference["fitted_values"]),
            ("deviance", fit.deviance, reference["deviance"]),
            ("score", fit.score, reference["score"]),
            ("edf", fit.edf_total, reference["edf"]),
            ("scale", fit.scale, reference["scale"]),
        ):
            collector.check(
                f"{label}: {name}",
                lambda actual=actual, expected=expected: _assert_strict(
                    actual, expected
                ),
            )
        _assert_hard_gates(
            fit, label, collector, must_converge=reference["inner_converged"]
        )
    collector.raise_if_any(f"supplied-sp sweep {r_family}/{link}")


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_factor_by_fixed_sp_matches_pinned_r_at_every_scale(r_bridge) -> None:
    """Factor-by smooths no longer drift from R in proportion to sp.

    The PR11 study measured 1.65e-6 / 1.65e-8 / 1.9e-10 at sp 1.9e8 / 1.9e6 /
    1.9e4 on this model before the structural null space.
    """
    rng = np.random.default_rng(20261025)
    x0, x1, x2 = rng.uniform(size=(3, 600))
    levels = ("a", "b", "c")
    fac = pd.Categorical(rng.choice(levels, 600), categories=levels)
    code = np.asarray(fac.codes)
    eta = (
        0.4 * np.sin(2 * np.pi * x0)
        + 2.0 * (x1 - 0.5) ** 2
        + np.array([0.0, 0.3, -0.2])[code]
        + np.select([code == 0, code == 1], [np.sin(3 * x2), -x2], 0.5 * np.cos(4 * x2))
    )
    data = pd.DataFrame({"x2": x2, "fac": fac, "y": eta + rng.normal(0, 0.5, 600)})
    formula = "y ~ fac + s(x2, bs='cr', k=8, by=fac)"
    collector = _AssertCollector()
    for scale in (1.0, 1e-2, 1e-4):
        sp = np.array([84.0287, 1.85527e8 * scale, 684.738])
        with contextlib.redirect_stdout(io.StringIO()):
            reference = r_bridge.fit_fixed_sp(formula, data, "gaussian", sp)
        fit = GAM(formula, sp=list(sp)).fit(data)
        label = f"sp[1]={sp[1]:.3e}"
        collector.check(
            f"{label}: fitted",
            lambda fit=fit, reference=reference: _assert_strict(
                fit.fitted_values, reference["fitted_values"]
            ),
        )
        collector.check(
            f"{label}: coefficients",
            lambda fit=fit, reference=reference: _assert_strict(
                fit.coefficients, reference["coefficients"]
            ),
        )
        collector.check(
            f"{label}: deviance",
            lambda fit=fit, reference=reference: _assert_strict(
                fit.deviance, reference["deviance"]
            ),
        )
        _assert_hard_gates(fit, label, collector)
    collector.raise_if_any("factor-by fixed sp")
