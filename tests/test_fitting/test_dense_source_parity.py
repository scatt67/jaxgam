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
import itertools
import warnings
from fractions import Fraction
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
from tests.helpers import (
    _AssertCollector,
    check_flat_ridge_endpoint,
    check_that,
    r_available,
)
from tests.tolerances import MODERATE, STRICT

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


def _ti_tp_frame() -> pd.DataFrame:
    """The public ti(tp) case of root question Q7 (n=300, Gaussian)."""
    rng = np.random.default_rng(20261116)
    x = rng.uniform(0.0, 1.0, 300)
    z = rng.uniform(0.0, 1.0, 300)
    y = np.sin(2 * np.pi * x) + 0.5 * np.cos(3 * z) + 0.3 * x * z
    return pd.DataFrame({"x": x, "z": z, "y": y + rng.normal(scale=0.3, size=300)})


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_public_ti_tp_fixed_at_r_reml_sp_matches_pinned_r(r_bridge) -> None:
    """Public dense ti(x,z,bs='tp',k=6) at pinned R's REML sp is STRICT.

    Before the series the trace-scaled jitter (1.8e-3 at these sp) acted as a
    ridge on the unpenalized bilinear direction (curvature 5.42), costing
    3.4e-4 EDF and 5e-5 in fitted values.  Gauge-invariant fields are checked
    (host R may use another BLAS, which can flip the TPRS eigenvector gauge);
    coefficients and Vp are gauge-dependent and not asserted here.
    """
    data = _ti_tp_frame()
    r_formula = "y~ti(x,z,bs='tp',k=c(6,6))"
    with contextlib.redirect_stdout(io.StringIO()):
        reml = r_bridge.fit_gam(r_formula, data, "gaussian", "REML")
        sp = np.asarray(reml["smoothing_params"], dtype=np.float64)
        reference = r_bridge.fit_fixed_sp(
            r_formula, data, "gaussian", sp, with_uncertainty=True
        )
    fit = GAM("y ~ ti(x, z, bs='tp', k=6)", sp=list(sp)).fit(data)
    _, link_se = fit.predict(data, pred_type="link", se_fit=True)
    collector = _AssertCollector()
    for name, actual, expected in (
        ("fitted", fit.fitted_values, reference["fitted_values"]),
        ("deviance", fit.deviance, reference["deviance"]),
        ("score", fit.score, reference["score"]),
        ("scale", fit.scale, reference["scale"]),
        ("edf", fit.edf_total, reml["edf_total"]),
        ("link_se", link_se, reference["link_se"]),
    ):
        collector.check(
            name,
            lambda actual=actual, expected=expected: _assert_strict(actual, expected),
        )
    _assert_hard_gates(fit, "ti tp at R's sp", collector)
    collector.raise_if_any("public ti(tp) at R's REML sp")


def _ridge_fixed_fits(r_bridge, jax_formula: str, r_formula: str, data):
    """Fixed-sp fit callables for ``check_flat_ridge_endpoint`` (Gaussian)."""

    def jax_fixed(sp):
        fit = GAM(jax_formula, sp=list(np.asarray(sp))).fit(data)
        return {
            "score": fit.score,
            "fitted_values": fit.fitted_values,
            "deviance": fit.deviance,
        }

    def r_fixed(sp):
        with contextlib.redirect_stdout(io.StringIO()):
            return r_bridge.fit_fixed_sp(r_formula, data, "gaussian", np.asarray(sp))

    return jax_fixed, r_fixed


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize(
    ("case", "jax_formula", "r_formula"),
    [
        (
            "public ti(tp) (root Q7)",
            "y ~ ti(x, z, bs='tp', k=6)",
            "y~ti(x,z,bs='tp',k=c(6,6))",
        ),
        (
            "finding-3 low-noise linear (root Q3)",
            "y ~ s(x, bs='cr', k=8)",
            "y ~ s(x, bs='cr', k=8)",
        ),
    ],
)
def test_reml_on_a_flat_ridge_meets_the_flat_ridge_rule(
    r_bridge, case: str, jax_formula: str, r_formula: str
) -> None:
    """REML whose infimum lies along a ridge toward sp -> inf.

    Pinned R's default-control newton stops on the ridge ("full convergence";
    its test is relative to |score|), so R's endpoint is not stationary for
    R's own criterion and the Newton-completed-point rule does not apply
    (root decision 1).  Measured: Q7 JaxGAM at sp (8.6e5, 1.8e5), score
    1.16e-4 below R's at (2.0e3, 2.0e4); finding 3 JaxGAM at log sp 21.14,
    2.7e-5 below R's at 17.39; criteria agree to ~1e-12 at both endpoints.
    Before the series (bb76a71) the finding-3 fit failed ("step failed") and
    the criteria disagreed by 3e-7..3e-5 at matched sp.
    """
    if case.startswith("public"):
        data = _ti_tp_frame()
    else:
        rng = np.random.default_rng(2)
        x = np.sort(rng.uniform(-1.0, 1.0, 150))
        data = pd.DataFrame({"x": x, "y": 1.0 + 2.0 * x + rng.normal(0, 0.1, x.size)})
    with contextlib.redirect_stdout(io.StringIO()):
        reference = r_bridge.fit_gam(r_formula, data, "gaussian", "REML")
    fit = GAM(jax_formula).fit(data)
    jax_fixed, r_fixed = _ridge_fixed_fits(r_bridge, jax_formula, r_formula, data)
    collector = _AssertCollector()
    check_flat_ridge_endpoint(
        collector,
        case,
        converged=bool(fit.converged),
        jax_sp=np.asarray(fit.smoothing_params),
        jax_score=float(fit.score),
        r_sp=np.asarray(reference["smoothing_params"]),
        r_score=float(reference["reml_score"]),
        jax_fixed=jax_fixed,
        r_fixed=r_fixed,
    )
    _assert_hard_gates(fit, case, collector)
    collector.raise_if_any(f"flat-ridge REML endpoint: {case}")


def test_every_fallback_jitter_is_visible_in_the_result() -> None:
    """No silent jitter (root decision 7): every factorization's fallback level.

    PIRLS factors, the criterion determinant and the IFT factor each report
    the cho_factor fallback they needed; the largest level and the names of
    the factors reach NewtonResult.factor_regularization and the public
    convergence_info, in addition to the RuntimeWarning.
    """
    from dataclasses import replace

    from jaxgam.fitting.newton import (
        NewtonResult,
        factorization_fallbacks,
        newton_optimize,
    )
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    rng = np.random.default_rng(3)
    z = np.sort(rng.uniform(-1.0, 1.0, 90))
    data = pd.DataFrame({"z": z, "y": np.sin(2 * z) + rng.normal(0, 0.2, 90)})
    collector = _AssertCollector()

    # Aliased parametric term: the PIRLS factor needs the fallback.
    with pytest.warns(RuntimeWarning, match="relative diagonal jitter"):
        fixed = GAM("y ~ z + s(z, bs='cr', k=6)", sp=[0.5]).fit(data)
    collector.check(
        "fixed sp: info names the PIRLS fallback",
        lambda: check_that(
            "fallback jitter 1e-12 in PIRLS" in fixed.convergence_info,
            fixed.convergence_info,
        ),
    )
    aliased = FittingData.from_setup(
        ModelSetup.build(parse_formula("y ~ z + s(z, bs='cr', k=6)"), data),
        Gaussian(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimized = newton_optimize(aliased)
    collector.check(
        "REML: NewtonResult carries the level and the info names it",
        lambda: check_that(
            isinstance(optimized, NewtonResult)
            and optimized.factor_regularization > 0.0
            and "fallback jitter" in optimized.convergence_info,
            f"{optimized.factor_regularization} {optimized.convergence_info!r}",
        ),
    )

    # A well-posed fit: no factor regularized, nothing in the info.
    clean = GAM("y ~ s(z, bs='cr', k=6)").fit(data)
    collector.check(
        "well-posed fit has no fallback",
        lambda: check_that(
            "fallback" not in clean.convergence_info, clean.convergence_info
        ),
    )

    # The criterion determinant and the IFT factor are checked on their own:
    # a singular X'WX + S at the reported state trips both (Gamma/log keeps
    # the Cholesky determinant), with a clean PIRLS factor.
    gamma_fd = FittingData.from_setup(
        ModelSetup.build(
            parse_formula("y ~ s(z, bs='cr', k=6)"), data.assign(y=np.exp(data.y))
        ),
        Gamma(link="log"),
    )
    log_lambda = jnp.asarray([0.0])
    fit = _fit_fixed_sp(gamma_fd, np.array([1.0])).pirls_result
    singular = replace(
        fit,
        XtWX=-gamma_fd.S_lambda(log_lambda) + 1e-30 * jnp.eye(fit.XtWX.shape[0]),
    )
    levels = factorization_fallbacks(gamma_fd, singular, log_lambda, include_ift=True)
    collector.check(
        "criterion and IFT levels reported",
        lambda: check_that(
            levels["PIRLS"] == 0.0
            and levels["criterion log|H|"] > 0.0
            and levels["IFT"] > 0.0,
            str(levels),
        ),
    )
    collector.raise_if_any("fallback visibility")


def _pr11_study_factor_by_frame() -> pd.DataFrame:
    """The exact rows of the PR11 dense factor-by large-sp repro.

    ``benchmark_discrete_approximation.make_data`` for the Gaussian 'study'
    case at its default seed 20261025 and n=600.
    """
    rng = np.random.default_rng(20261025)
    x0, x1, x2 = rng.uniform(size=(3, 600))
    levels = ("a", "b", "c")
    fac = pd.Categorical(rng.choice(levels, 600), categories=levels)
    groups = tuple(f"g{index:02d}" for index in range(20))
    g = pd.Categorical(rng.choice(groups, 600), categories=groups)
    code = np.asarray(fac.codes)
    effect = np.random.default_rng(20261025 + 2).normal(0.0, 0.3, len(groups))
    eta = (
        0.4 * np.sin(2.0 * np.pi * x0)
        + 2.0 * (x1 - 0.5) ** 2
        + 0.6 * np.sin(np.pi * x0) * np.cos(2.0 * np.pi * x1)
        + np.array([0.0, 0.3, -0.2])[code]
        + np.select(
            [code == 0, code == 1], [np.sin(3.0 * x2), -x2], 0.5 * np.cos(4.0 * x2)
        )
        + effect[np.asarray(g.codes)]
    )
    y = eta + rng.normal(0.0, 0.5, 600)
    return pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "fac": fac, "g": g, "y": y})


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_pr11_factor_by_study_rows_match_pinned_r_from_1e_4_to_1e4(r_bridge) -> None:
    """The PR11 repro rows, at R's REML sp with the by-smooth sp scaled.

    ``dense_factor_by_large_sp_repro.py`` scaled the middle sp by 1, 1e-2 and
    1e-4; the same study's large side (1e2, 1e4) is included.  Every field is
    STRICT against pinned R's fixed-sp fit.
    """
    data = _pr11_study_factor_by_frame()
    formula = "y ~ fac + s(x2, bs='cr', k=8, by=fac)"
    with contextlib.redirect_stdout(io.StringIO()):
        reml = r_bridge.fit_gam(formula, data, "gaussian", "REML")
    base = np.asarray(reml["smoothing_params"], dtype=np.float64)
    collector = _AssertCollector()
    for scale in (1e-4, 1e-2, 1.0, 1e2, 1e4):
        sp = base * np.array([1.0, scale, 1.0])
        with contextlib.redirect_stdout(io.StringIO()):
            reference = r_bridge.fit_fixed_sp(
                formula, data, "gaussian", sp, with_uncertainty=True
            )
        fit = GAM(formula, sp=list(sp)).fit(data)
        _, link_se = fit.predict(data, pred_type="link", se_fit=True)
        label = f"scale {scale:g}"
        for name, actual, expected in (
            ("coefficients", fit.coefficients, reference["coefficients"]),
            ("fitted", fit.fitted_values, reference["fitted_values"]),
            ("deviance", fit.deviance, reference["deviance"]),
            ("score", fit.score, reference["score"]),
            ("scale", fit.scale, reference["scale"]),
            ("Vp", fit.Vp, reference["Vp"]),
            ("link_se", link_se, reference["link_se"]),
        ):
            collector.check(
                f"{label}: {name}",
                lambda actual=actual, expected=expected: _assert_strict(
                    actual, expected
                ),
            )
        _assert_hard_gates(fit, label, collector)
    collector.raise_if_any("PR11 factor-by study rows")


# ---------------------------------------------------------------------------
# Stage 2: pls_fit1 QR inner solve
# ---------------------------------------------------------------------------


def test_penalized_qr_solve_and_augmented_root_compile_and_are_exact() -> None:
    """The QR kernel solves the penalized system without normal equations."""
    from fractions import Fraction

    from jaxgam.fitting.qr import penalized_qr_solve

    rng = np.random.default_rng(3)
    n, p = 30, 5
    X = rng.standard_normal((n, p))
    w = rng.uniform(0.5, 2.0, n)
    z = rng.standard_normal(n)
    roots = penalty_ops.JaxPenaltyRoots(
        n_coef=p,
        starts=(1, 2),
        stops=(4, 5),
        sp_indices=(0, 1),
        roots=(jnp.asarray(rng.standard_normal((2, 3))), jnp.asarray(np.eye(3))),
    )
    rho = jnp.asarray([2.5, -1.0])
    E = jax.jit(penalty_ops.augmented_root)(roots, rho)
    S = np.zeros((p, p))
    S[1:4, 1:4] += (
        np.exp(2.5) * np.asarray(roots.roots[0]).T @ np.asarray(roots.roots[0])
    )
    S[2:5, 2:5] += np.exp(-1.0) * np.eye(3)
    _assert_strict(np.asarray(E).T @ np.asarray(E), S)

    wx, wz = np.sqrt(w)[:, None] * X, np.sqrt(w) * z
    beta, L = jax.jit(penalized_qr_solve)(jnp.asarray(wx), jnp.asarray(wz), E)
    H = wx.T @ wx + np.asarray(E).T @ np.asarray(E)
    _assert_strict(np.asarray(L) @ np.asarray(L).T, H)
    assert np.all(np.diag(np.asarray(L)) > 0)

    exact = [[Fraction(float(v)) for v in row] for row in H]
    rhs = [Fraction(float(v)) for v in wx.T @ wz]
    for c in range(p):  # exact Gauss-Jordan on the float inputs
        inv = 1 / exact[c][c]
        exact[c], rhs[c] = [v * inv for v in exact[c]], rhs[c] * inv
        for r in range(p):
            if r != c:
                f = exact[r][c]
                exact[r] = [a - f * b for a, b in zip(exact[r], exact[c], strict=True)]
                rhs[r] -= f * rhs[c]
    _assert_strict(np.asarray(beta), np.array([float(v) for v in rhs]))


def test_qr_roots_follow_the_lambda_free_structural_rank() -> None:
    """Full-rank models get QR roots; a structurally aliased model keeps Cholesky."""
    rng = np.random.default_rng(11)
    n = 80
    x = np.sort(rng.uniform(0, 1, n))
    data = pd.DataFrame(
        {"x": x, "z": rng.uniform(0, 1, n), "y": np.sin(4 * x) + rng.normal(0, 0.2, n)}
    )
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    full = FittingData.from_setup(
        ModelSetup.build(parse_formula("y ~ s(x, bs='cr', k=8)"), data), Gaussian()
    )
    assert full.penalty_roots is not None
    E = np.asarray(full.penalty_root(jnp.asarray([0.3])))
    _assert_strict(E.T @ E, np.asarray(full.S_lambda(jnp.asarray([0.3]))))
    aliased = FittingData.from_setup(
        ModelSetup.build(parse_formula("y ~ z + s(z, bs='cr', k=6)"), data), Gaussian()
    )
    assert aliased.rank_deficit == 1
    assert aliased.penalty_roots is None


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize(
    ("layout", "term", "compare_coefficients"),
    [
        ("regular", "s(x,bs='tp',k=4)", True),
        ("irregular", "ti(x,z,bs='tp',k=c(8,8))", False),
        ("regular", "ti(x,z,bs='tp',k=c(8,8))", False),
    ],
)
def test_r_exact_design_at_r_reml_sp_matches_pinned_r(
    r_bridge, layout: str, term: str, compare_coefficients: bool
) -> None:
    """At R's REML sp (cond(H) up to ~1e15) the QR inner fit reproduces R.

    The regular ti optimum is essentially unpenalized (sp ~1e-20) with
    cond(X) ~5e14: the normal-equation Hessian is numerically singular and the
    Cholesky route could only regularize it.  Coefficients of such models are
    not identifiable to STRICT by any method (R's included), so only fitted
    values and deviance are compared there.  For the regular ti case
    (sp ~1e-20, cond(H) 4e18) the remaining fields wait for pls_fit1's
    per-call rank test and column drop (root decision 10, after the rebase);
    no tolerance is applied to them meanwhile.  The irregular ti case's
    other fields are gated in
    test_ti_irregular_at_r_reml_sp_matches_pinned_r_where_r_is_accurate.
    """
    data = _parity_frame(layout)
    with contextlib.redirect_stdout(io.StringIO()):
        components = r_bridge.get_smooth_components(
            "y~" + term, data, "gaussian", "REML"
        )
    sp = np.asarray(components["smoothing_params"], dtype=np.float64)
    fit, coefficients, fitted, deviance = _dense_fit_on_r_design(
        components, data.y.to_numpy(), sp
    )
    collector = _AssertCollector()
    collector.check(
        "fitted", lambda: _assert_strict(fitted, components["fitted_values"])
    )
    collector.check(
        "deviance", lambda: _assert_strict(deviance, components["deviance"])
    )
    if compare_coefficients:
        collector.check(
            "coefficients",
            lambda: _assert_strict(coefficients, components["coefficients"]),
        )
    collector.check(
        "unregularized QR",
        lambda: check_that(
            float(fit.pirls_result.factor_regularization) == 0.0, "fallback fired"
        ),
    )
    collector.raise_if_any(f"R-exact dense at R's REML sp {layout} {term}")


def test_qr_log_det_and_hat_compile_and_match_the_normal_equations() -> None:
    """Stage 2 completion: log|H| and EDF from the pls_fit1/gdi QR, under jit.

    On a well-conditioned system the QR determinant and influence matrix equal
    the normal-equation ones at STRICT, and the custom JVP's first and second
    derivatives (through H only, as the inner fit's IFT supplies dH) equal
    automatic differentiation of the Cholesky determinant.
    """
    from jaxgam.fitting.qr import penalized_qr_hat, qr_log_det

    rng = np.random.default_rng(8)
    n, p = 80, 6
    X = jnp.asarray(rng.standard_normal((n, p)))
    w = jnp.asarray(rng.uniform(0.2, 2.0, n))
    S1 = np.diag([0.0, 0.0, 1.0, 2.0, 0.0, 0.0])
    S2 = np.diag([0.0, 0.0, 0.0, 0.0, 3.0, 1.0])
    roots = (np.sqrt(S1)[2:4], np.sqrt(S2)[4:])
    wX = jnp.sqrt(w)[:, None] * X
    XtWX = wX.T @ wX

    def parts(rho):
        E = jnp.concatenate(
            [jnp.exp(0.5 * rho[0]) * roots[0], jnp.exp(0.5 * rho[1]) * roots[1]]
        )
        S = jnp.exp(rho[0]) * S1 + jnp.exp(rho[1]) * S2
        return E, S

    def qr_score(rho):
        E, S = parts(rho)
        return qr_log_det(XtWX + S, jax.lax.stop_gradient(wX), jax.lax.stop_gradient(E))

    def chol_score(rho):
        _, S = parts(rho)
        return 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(XtWX + S))))

    rho = jnp.asarray([0.3, -0.7])
    collector = _AssertCollector()
    collector.check(
        "log|H| value",
        lambda: _assert_strict(jax.jit(qr_score)(rho), chol_score(rho)),
    )
    collector.check(
        "gradient",
        lambda: _assert_strict(
            jax.jit(jax.grad(qr_score))(rho), jax.grad(chol_score)(rho)
        ),
    )
    collector.check(
        "Hessian",
        lambda: _assert_strict(
            jax.jit(jax.hessian(qr_score))(rho), jax.hessian(chol_score)(rho)
        ),
    )
    E, S = parts(rho)
    F, edf = jax.jit(penalized_qr_hat)(wX, E)
    F_normal = np.linalg.solve(np.asarray(XtWX + S), np.asarray(XtWX))
    collector.check("hat matrix", lambda: _assert_strict(F, F_normal))
    collector.check("EDF", lambda: _assert_strict(edf, np.trace(F_normal)))
    collector.raise_if_any("QR log|H| and EDF")


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_dense_fit_records_the_qr_fisher_influence_matrix(r_bridge) -> None:
    """Full-rank dense fits carry F and tr(F) from the QR; EDF equals R's."""
    data = _parity_frame("regular")
    fit_fixed = GAM("y ~ s(x, bs='cr', k=8) + s(z, bs='cr', k=6)", sp=[0.7, 2.0])
    fit = fit_fixed.fit(data)
    with contextlib.redirect_stdout(io.StringIO()):
        reference = r_bridge.public_fixed_sp_fit(
            data[["x", "y"]],
            np.ones(len(data)),
            np.zeros(len(data)),
            "gaussian",
            "identity",
            sp=0.7,
        )
    single = GAM('y ~ s(x, bs="cr", k=8)', sp=[0.7]).fit(data[["x", "y"]])
    collector = _AssertCollector()
    collector.check(
        "EDF vs R", lambda: _assert_strict(single.edf_total, reference["edf"])
    )
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    for family in (Gaussian(), Gamma(link="log")):
        response = (
            data.assign(y=np.exp(0.3 * data.y)) if isinstance(family, Gamma) else data
        )
        fd = FittingData.from_setup(
            ModelSetup.build(parse_formula("y ~ s(x, bs='cr', k=8)"), response), family
        )
        pirls = _fit_fixed_sp(fd, np.array([0.7])).pirls_result

        def _recorded(pirls=pirls, fd=fd) -> None:
            check_that(fd.penalty_roots is not None, "not on the QR route")
            check_that(pirls.fisher_hat is not None, "no QR influence matrix")
            _assert_strict(np.trace(np.asarray(pirls.fisher_hat)), pirls.fisher_edf)

        collector.check(f"{family.family_name}: QR F recorded", _recorded)
    collector.check(
        "per-smooth EDF sums to total",
        lambda: _assert_strict(np.sum(fit.edf) + 1.0, fit.edf_total),
    )
    collector.raise_if_any("QR Fisher influence matrix")


# ---------------------------------------------------------------------------
# Ill-conditioned fits on R's exact X/S and on exactly representable inputs
# (root decisions 4-6 as corrected 2026-11-16)
# ---------------------------------------------------------------------------

_TI_TERM = "ti(x,z,bs='tp',k=c(8,8))"


def _r_exact_fit_fields(components: dict, y: np.ndarray, sp: np.ndarray) -> dict:
    """Every JaxGAM field of the dense fixed-sp route on R's exact X and S."""
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
    scale = float(fit.scale)
    factor = np.asarray(pirls.L_fisher)
    p = factor.shape[0]
    inverse = np.linalg.solve(factor, np.eye(p))
    Vp = scale * np.asarray(
        penalty_ops.transform_covariance(
            fd.penalty_structure, jnp.asarray(inverse.T @ inverse)
        )
    )
    starts = np.cumsum([X.shape[1] - sum(widths), *widths])
    hat_diag = np.diag(np.asarray(pirls.fisher_hat))
    beta = np.asarray(
        penalty_ops.transform_coefficients(fd.penalty_structure, pirls.coefficients)
    )
    S_lambda = np.asarray(fd.S_lambda(jnp.log(jnp.asarray(sp, dtype=jnp.float64))))
    Xf = np.asarray(fd.X)
    residual = Xf.T @ (np.asarray(y) - Xf @ np.asarray(pirls.coefficients)) - (
        S_lambda @ np.asarray(pirls.coefficients)
    )
    H = Xf.T @ Xf + S_lambda
    backward_error = np.linalg.norm(residual) / (
        np.linalg.norm(H, 2) * np.linalg.norm(pirls.coefficients)
        + np.linalg.norm(Xf.T @ np.asarray(y))
    )
    return {
        "fit": fit,
        "coefficients": beta,
        "fitted": np.asarray(pirls.mu),
        "deviance": float(pirls.deviance),
        "score": float(fit.score),
        "scale": scale,
        "edf_total": float(fit.edf),
        "edf_per_smooth": np.array(
            [np.sum(hat_diag[a:b]) for a, b in itertools.pairwise(starts)]
        ),
        "Vp": Vp,
        "link_se": np.sqrt(scale)
        * np.linalg.norm(np.linalg.solve(factor, Xf.T), axis=0),
        "backward_error": float(backward_error),
        "regularization": float(np.max(np.asarray(pirls.factor_regularization))),
    }


def _r_link_se(X: np.ndarray, Vp: np.ndarray) -> np.ndarray:
    """predict.gam's link SE: sqrt(rowSums((X %*% Vp) * X))."""
    return np.sqrt(np.maximum(np.sum((X @ Vp) * X, axis=1), 0.0))


def _strict_multiple(actual, expected) -> float:
    """max |a - e| / (atol + rtol |e|) for STRICT; <= 1 passes."""
    actual = np.atleast_1d(np.asarray(actual, dtype=np.float64))
    expected = np.atleast_1d(np.asarray(expected, dtype=np.float64))
    return float(
        np.max(
            np.abs(actual - expected) / (STRICT.atol + STRICT.rtol * np.abs(expected))
        )
    )


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_ti_irregular_at_r_reml_sp_matches_pinned_r_where_r_is_accurate(
    r_bridge,
) -> None:
    """ti irregular at R's REML sp (cond H 1.4e12) on R's exact X and S.

    STRICT against R, with their measured margins (x STRICT): score 0.36,
    fitted values 0.061, deviance 1.7e-3, scale 7e-5, total and per-smooth EDF
    0.028 (EDF from the QR influence matrix).  JaxGAM's normwise backward
    error is asserted at STRICT.

    Raw coefficients, raw Vp and link SE (the ill-conditioned direction) are
    at MODERATE, one class looser (Nov 17 tolerance addendum).  Measured on
    macOS/arm64 against pinned R (Accelerate BLAS): 276x, 30x and 57x STRICT
    (relative 2.8e-8, 3e-9, 5.7e-9); against a 60-digit solve of the same
    floating-point X and S, R itself is 274x, 3600x and 57x STRICT from the
    truth while JaxGAM is 1.6x, 11x and 1.7e-4x, so the gap is R's rounding
    in a direction of cond(H) = 1.4e12, not a modelling difference.
    """
    data = _parity_frame("irregular")
    with contextlib.redirect_stdout(io.StringIO()):
        components = r_bridge.get_smooth_components(
            "y~" + _TI_TERM, data, "gaussian", "REML"
        )
    sp = np.asarray(components["smoothing_params"], dtype=np.float64)
    fields = _r_exact_fit_fields(components, data.y.to_numpy(), sp)
    collector = _AssertCollector()
    for name, expected in (
        ("score", components["reml_score"]),
        ("fitted", components["fitted_values"]),
        ("deviance", components["deviance"]),
        ("scale", components["scale"]),
        ("edf_total", components["edf_total"]),
        ("edf_per_smooth", components["edf"]),
    ):
        collector.check(
            name,
            lambda name=name, expected=expected: _assert_strict(fields[name], expected),
        )
    X = np.asarray(components["model_matrix"], dtype=np.float64)
    for name, expected in (
        # MODERATE: ill-conditioned direction, see the docstring.
        ("coefficients", components["coefficients"]),
        ("Vp", components["Vp"]),
        ("link_se", _r_link_se(X, np.asarray(components["Vp"]))),
    ):
        collector.check(
            f"{name} (MODERATE)",
            lambda name=name, expected=expected: np.testing.assert_allclose(
                fields[name], expected, rtol=MODERATE.rtol, atol=MODERATE.atol
            ),
        )
    collector.check(
        "backward error",
        lambda: check_that(
            fields["backward_error"] <= STRICT.rtol,
            f"backward error {fields['backward_error']:.2e}",
        ),
    )
    collector.check(
        "no fallback",
        lambda: check_that(fields["regularization"] == 0.0, "fallback fired"),
    )
    collector.raise_if_any("ti irregular at R's REML sp")


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_ti_irregular_at_sp_0_7_matches_pinned_r_where_r_is_accurate(
    r_bridge,
) -> None:
    """ti irregular at sp 0.7 on R's exact X and S.

    JaxGAM's normwise backward error (1.3e-16; R 2.0e-16) at STRICT, and the
    fields that hold with margin at STRICT against R (margins in the packet).
    Fitted values are measured, not asserted: after four documented attempts
    (x STRICT vs R: Stage 1 2.23, Stage 2 2.74, column pivoting 2.74, one
    refinement step on the true residual 1.85) they miss STRICT, and against a
    60-digit solve R is 1.77x and JaxGAM 0.97x.  Fitted values are a
    well-determined field, so the Nov 17 addendum does not allow moving them a
    class here; this is reported to root as an open item.
    """
    data = _parity_frame("irregular")
    with contextlib.redirect_stdout(io.StringIO()):
        components = r_bridge.get_smooth_components(
            "y~" + _TI_TERM, data, "gaussian", "REML"
        )
        n_sp = sum(len(p) for p in components["penalty_matrices"])
        sp = np.full(n_sp, 0.7)
        reference = r_bridge.fit_fixed_sp(
            "y~" + _TI_TERM, data, "gaussian", sp, with_uncertainty=True
        )
    fields = _r_exact_fit_fields(components, data.y.to_numpy(), sp)
    collector = _AssertCollector()
    collector.check(
        "backward error",
        lambda: check_that(
            fields["backward_error"] <= STRICT.rtol,
            f"backward error {fields['backward_error']:.2e}",
        ),
    )
    for name, expected in _ti_sp07_asserted(reference, components):
        collector.check(
            name,
            lambda name=name, expected=expected: _assert_strict(fields[name], expected),
        )
    collector.raise_if_any("ti irregular at sp 0.7")


def _ti_sp07_asserted(reference: dict, components: dict):
    """Fields of the sp-0.7 case asserted at STRICT against R.

    Measured margins (x STRICT): deviance 0.099, score 0.11, scale 0.087,
    Vp 0.38, link SE 0.11.  Coefficients (0.59x: R itself is 0.45x from the
    60-digit truth, JaxGAM 0.21x) and fitted values (2.74x) are measured, not
    asserted, pending the PR11 perturbation bound.
    """
    X = np.asarray(components["model_matrix"], dtype=np.float64)
    return (
        ("deviance", reference["deviance"]),
        ("score", reference["score"]),
        ("scale", reference["scale"]),
        ("Vp", reference["Vp"]),
        ("link_se", _r_link_se(X, np.asarray(reference["Vp"]))),
    )


def _exact_ill_conditioned_case():
    """Integer X, S and y and a power-of-two sp: every input is exact.

    Columns 1, t, t^2, M t + e1, M t^2 + e2 (t = -20..19, M = 2^12, e1, e2
    small integers) are nearly collinear after scaling (equilibrated
    cond(X'X + sp S) = 1.9e10); the last three columns carry an identity
    penalty at sp = 2^-6.
    """
    t = np.arange(-20, 20, dtype=np.int64)
    M = 2**12
    e1 = (t * 7) % 3 - 1
    e2 = (t * 11) % 5 - 2
    X = np.column_stack([np.ones_like(t), t, t * t, M * t + e1, M * t * t + e2])
    y = 3 + (t // 2) + (t * t) % 7 - 3 * ((t * 5) % 3)
    return X, y, (2, 3, 4), Fraction(1, 2**6)


def _exact_reference(X, y, penalized, sp):
    """Exact rational coefficients, fitted values, EDF and (X'X + sp S)^-1."""
    n, p = X.shape
    XF = [[Fraction(int(v)) for v in row] for row in X]
    XtX = [
        [sum((XF[i][a] * XF[i][c] for i in range(n)), Fraction(0)) for c in range(p)]
        for a in range(p)
    ]
    H = [
        [XtX[a][c] + (sp if a == c and a in penalized else 0) for c in range(p)]
        for a in range(p)
    ]
    rhs = [sum((XF[i][a] * int(y[i]) for i in range(n)), Fraction(0)) for a in range(p)]
    A = [
        H[a][:] + [rhs[a]] + [Fraction(int(a == c)) for c in range(p)] for a in range(p)
    ]
    for col in range(p):
        pivot = next(r for r in range(col, p) if A[r][col] != 0)
        A[col], A[pivot] = A[pivot], A[col]
        scale = A[col][col]
        A[col] = [v / scale for v in A[col]]
        for r in range(p):
            if r != col and A[r][col] != 0:
                factor = A[r][col]
                A[r] = [A[r][c] - factor * A[col][c] for c in range(2 * p + 1)]
    beta = [A[a][p] for a in range(p)]
    inverse = [[A[a][p + 1 + c] for c in range(p)] for a in range(p)]
    edf = sum(sum(inverse[a][m] * XtX[m][a] for m in range(p)) for a in range(p))
    fitted = [sum(XF[i][c] * beta[c] for c in range(p)) for i in range(n)]
    return {
        "coefficients": np.array([float(v) for v in beta]),
        "fitted": np.array([float(v) for v in fitted]),
        "edf": float(edf),
        "covariance": np.array([[float(v) for v in row] for row in inverse]),
    }


def _exact_case_fit():
    X, y, penalized, sp = _exact_ill_conditioned_case()
    n, p = X.shape
    terms = [
        SimpleNamespace(
            col_start=penalized[0], n_coefs=len(penalized), penalty_indices=[0]
        )
    ]
    setup = SimpleNamespace(
        X=X.astype(np.float64),
        y=y.astype(np.float64),
        weights=np.ones(n),
        offset=None,
        n_obs=n,
        penalties=make_penalty_structure(p, terms, [[np.eye(len(penalized))]]),
    )
    fd = FittingData.from_setup(setup, Gaussian())
    fit = _fit_fixed_sp(fd, np.array([float(sp)]))
    pirls = fit.pirls_result
    factor = np.asarray(pirls.L_fisher)
    inverse = np.linalg.solve(factor, np.eye(p))
    covariance = np.asarray(
        penalty_ops.transform_covariance(
            fd.penalty_structure, jnp.asarray(inverse.T @ inverse)
        )
    )
    coefficients = np.asarray(
        penalty_ops.transform_coefficients(fd.penalty_structure, pirls.coefficients)
    )
    return (
        fd,
        fit,
        {
            "coefficients": coefficients,
            "fitted": np.asarray(pirls.mu),
            "edf": float(fit.edf),
            "covariance": covariance,
        },
    )


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_exact_input_ill_conditioned_fit_is_closer_to_the_truth_than_r(
    r_bridge,
) -> None:
    """Exact rational reference: JaxGAM within STRICT where R is not.

    Measured (x STRICT against the exact values): JaxGAM coefficients 0.18,
    fitted values 0.30, EDF 3e-5, unscaled covariance 39.6 (MODERATE by
    class, the only relaxed field); pinned C_pls_fit1 on the same exact inputs
    coefficients 3.45, fitted values 0.33.  The coefficient test asserts both
    JaxGAM within STRICT and JaxGAM no farther from the truth than R.
    """
    X, y, penalized, sp = _exact_ill_conditioned_case()
    exact = _exact_reference(X, y, penalized, sp)
    fd, fit, jax_fields = _exact_case_fit()
    E = np.zeros((len(penalized), X.shape[1]))
    E[np.arange(len(penalized)), list(penalized)] = float(sp) ** 0.5
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, r_beta = r_bridge.source_signed_pls_fit1(
            X.astype(np.float64),
            np.ones(len(y)),
            y.astype(np.float64),
            y.astype(np.float64),
            E,
            use_weighted_response=False,
        )
    collector = _AssertCollector()
    collector.check(
        "QR route without fallback",
        lambda: check_that(
            fd.penalty_roots is not None
            and float(np.max(np.asarray(fit.pirls_result.factor_regularization)))
            == 0.0,
            "not the plain QR route",
        ),
    )
    for name in ("coefficients", "fitted", "edf"):
        collector.check(
            f"{name} vs exact",
            lambda name=name: _assert_strict(jax_fields[name], exact[name]),
        )
    collector.check(
        # MODERATE (Nov 17 addendum): the unscaled covariance in the
        # ill-conditioned direction is 39.6x STRICT (relative 4e-9) from the
        # exact rational inverse on macOS/arm64; coefficients, fitted values and
        # EDF of the same fit stay STRICT.
        "unscaled covariance vs exact (MODERATE)",
        lambda: np.testing.assert_allclose(
            jax_fields["covariance"],
            exact["covariance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        "JaxGAM coefficients no farther from the truth than R's",
        lambda: check_that(
            _strict_multiple(jax_fields["coefficients"], exact["coefficients"])
            <= _strict_multiple(r_beta, exact["coefficients"]),
            "R closer to the exact coefficients",
        ),
    )
    collector.raise_if_any("exact-input ill-conditioned fit")


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_qr_route_additive_disparate_sp_matches_pinned_r(r_bridge) -> None:
    """Two smooths whose log sp differ by up to 50 stay STRICT on the QR route.

    Each penalty keeps its own exp(rho/2)-scaled root rows (never summed), so
    an exp(40) penalty cannot swamp an exp(0) or exp(-20) one.
    """
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    rng = np.random.default_rng(31)
    x, z = rng.uniform(0.0, 1.0, (2, 200))
    y = (
        np.sin(2 * np.pi * x)
        + 0.5 * np.cos(3 * z)
        + 0.4 * x * z
        + rng.normal(0.0, 0.3, 200)
    )
    data = pd.DataFrame({"x": x, "z": z, "y": y})
    formula = "y ~ s(x, bs='cr', k=8) + s(z, bs='cr', k=6)"
    fd = FittingData.from_setup(
        ModelSetup.build(parse_formula(formula), data), Gaussian()
    )
    assert fd.penalty_roots is not None
    collector = _AssertCollector()
    for log_sp in ((40.0, 0.0), (0.0, 40.0), (40.0, 40.0), (-20.0, 30.0)):
        sp = np.exp(np.asarray(log_sp))
        with contextlib.redirect_stdout(io.StringIO()):
            reference = r_bridge.fit_fixed_sp(
                formula, data, "gaussian", sp, with_uncertainty=True
            )
        fit = GAM(formula, sp=list(sp)).fit(data)
        _, link_se = fit.predict(data, pred_type="link", se_fit=True)
        label = f"log sp {log_sp}"
        for name, actual, expected in (
            ("coefficients", fit.coefficients, reference["coefficients"]),
            ("fitted", fit.fitted_values, reference["fitted_values"]),
            ("deviance", fit.deviance, reference["deviance"]),
            ("score", fit.score, reference["score"]),
            ("scale", fit.scale, reference["scale"]),
            ("Vp", fit.Vp, reference["Vp"]),
            ("link_se", link_se, reference["link_se"]),
        ):
            collector.check(
                f"{label}: {name}",
                lambda actual=actual, expected=expected: _assert_strict(
                    actual, expected
                ),
            )
        _assert_hard_gates(fit, label, collector, must_converge=reference["converged"])
    collector.raise_if_any("QR route, additive disparate sp")
