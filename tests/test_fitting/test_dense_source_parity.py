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
from decimal import Decimal, localcontext
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


def _fixed_nb():
    from jaxgam.families.negative_binomial import NegativeBinomial

    return NegativeBinomial(theta=2.0, fixed=True)


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
        (_fixed_nb, "nb", "log", "count"),
    ],
    ids=["gaussian", "gamma-log", "poisson", "binomial", "nb-fixed-theta"],
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
        theta = 2.0 if r_family == "nb" else None
        with contextlib.redirect_stdout(io.StringIO()):
            reference = r_bridge.public_fixed_sp_fit(
                data, np.ones(n), np.zeros(n), r_family, link, theta=theta, sp=sp
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

    Fitted values are at MODERATE (Nov 17 addendum) against both a 60-digit
    solve of the same floating-point X, S and sp and pinned R; the measured
    distances are at the checks.  R itself is 1.77-1.85x STRICT from the
    60-digit solve on this host, so the gap to R is R's rounding in this
    ill-conditioned fit (cond H 1.4e12), not a modelling difference.
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
    truth = _sixty_digit_fitted(components, data.y.to_numpy(), sp)
    collector.check(
        # MODERATE: macOS/arm64 0.97x STRICT with Accelerate, 0.49x with
        # OpenBLAS; a margin below 1 that does not hold across platforms.
        "fitted vs 60-digit solve (MODERATE)",
        lambda: np.testing.assert_allclose(
            fields["fitted"], truth, rtol=MODERATE.rtol, atol=MODERATE.atol
        ),
    )
    collector.check(
        # MODERATE: macOS/arm64 2.74x STRICT with Accelerate, 1.46x (relative
        # 1.7e-10) with OpenBLAS, against this host's pinned R.
        "fitted vs R (MODERATE)",
        lambda: np.testing.assert_allclose(
            fields["fitted"],
            reference["fitted_values"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.raise_if_any("ti irregular at sp 0.7")


def _sixty_digit_fitted(components: dict, y: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """Fitted values of (X'X + sum sp_j S_j) beta = X'y at 60 digits.

    X, S and sp are R's floating-point values, taken exactly; S is symmetrized
    as the dense route does.  Partial-pivot elimination at 60 digits leaves an
    error near cond(H) 1e-60, far below STRICT.
    """
    X = np.asarray(components["model_matrix"], dtype=np.float64)
    n, p = X.shape
    widths = [basis.shape[1] for basis in components["basis_matrices"]]
    with localcontext() as context:
        context.prec = 60
        XD = [[Decimal(float(v)) for v in row] for row in X]
        H = [
            [sum((XD[i][a] * XD[i][c] for i in range(n)), Decimal(0)) for c in range(p)]
            for a in range(p)
        ]
        start, index = p - sum(widths), 0
        for width, penalties in zip(
            widths, components["penalty_matrices"], strict=True
        ):
            for S in penalties:
                weight = Decimal(float(sp[index])) / 2
                for a in range(width):
                    for c in range(width):
                        H[start + a][start + c] += weight * (
                            Decimal(float(S[a, c])) + Decimal(float(S[c, a]))
                        )
                index += 1
            start += width
        A = [
            H[a]
            + [sum((XD[i][a] * Decimal(float(y[i])) for i in range(n)), Decimal(0))]
            for a in range(p)
        ]
        for col in range(p):
            pivot = max(range(col, p), key=lambda r: abs(A[r][col]))
            A[col], A[pivot] = A[pivot], A[col]
            for r in range(col + 1, p):
                factor = A[r][col] / A[col][col]
                A[r] = [A[r][c] - factor * A[col][c] for c in range(p + 1)]
        beta = [Decimal(0)] * p
        for r in range(p - 1, -1, -1):
            tail = sum((A[r][c] * beta[c] for c in range(r + 1, p)), Decimal(0))
            beta[r] = (A[r][p] - tail) / A[r][r]
        return np.array(
            [
                float(sum((XD[i][c] * beta[c] for c in range(p)), Decimal(0)))
                for i in range(n)
            ]
        )


def _ti_sp07_asserted(reference: dict, components: dict):
    """Fields of the sp-0.7 case asserted at STRICT against R.

    Measured margins (x STRICT): deviance 0.099, score 0.11, scale 0.087,
    Vp 0.38, link SE 0.11.  Coefficients (0.59x: R itself is 0.45x from the
    60-digit truth, JaxGAM 0.21x) are measured, not asserted, pending the PR11
    perturbation bound; fitted values are gated in the test itself.
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


def test_exact_input_ill_conditioned_fit_matches_the_exact_solution() -> None:
    """Exact rational reference for an ill-conditioned fit with exact inputs.

    Fitted values and EDF, the well-determined fields, are STRICT against the
    exact values.  Coefficients and the unscaled covariance, which carry the
    ill-conditioned direction, are MODERATE (Nov 17 addendum); the measured
    distances are at each check.  An earlier "no farther from the truth than
    pinned C_pls_fit1" comparison is not asserted: R's coefficients are 3.45x
    STRICT from the truth on macOS/arm64, JaxGAM's 9.87x there with OpenBLAS
    and 4.02x on native x86_64, so the ordering is platform rounding.
    """
    X, y, penalized, sp = _exact_ill_conditioned_case()
    exact = _exact_reference(X, y, penalized, sp)
    fd, fit, jax_fields = _exact_case_fit()
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
    for name in ("fitted", "edf"):
        collector.check(
            f"{name} vs exact",
            lambda name=name: _assert_strict(jax_fields[name], exact[name]),
        )
    collector.check(
        # MODERATE (Nov 17 addendum): coefficients along the nearly collinear
        # columns are 4.02x STRICT (relative 4.0e-10) from the exact rational
        # solution on native x86_64 and 9.87x (relative 1.15e-9) on
        # macOS/arm64 with OpenBLAS; fitted values and EDF stay STRICT.
        "coefficients vs exact (MODERATE)",
        lambda: np.testing.assert_allclose(
            jax_fields["coefficients"],
            exact["coefficients"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
        ),
    )
    collector.check(
        # MODERATE (Nov 17 addendum): the unscaled covariance in the
        # ill-conditioned direction is 39.6x STRICT (relative 4e-9) from the
        # exact rational inverse on macOS/arm64 with Accelerate and 18.1x with
        # OpenBLAS; fitted values and EDF of the same fit stay STRICT.
        "unscaled covariance vs exact (MODERATE)",
        lambda: np.testing.assert_allclose(
            jax_fields["covariance"],
            exact["covariance"],
            rtol=MODERATE.rtol,
            atol=MODERATE.atol,
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


# ---------------------------------------------------------------------------
# Finding 2, Option A: pinned gam.fit3 / gam.fit4 inner loops
# ---------------------------------------------------------------------------


def test_sqrt_link_owns_the_positive_eta_domain() -> None:
    """R make.link("sqrt")$valideta is eta > 0; NB and Poisson delegate to it."""
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.links.links import LogLink, SqrtLink

    eta = np.array([-0.5, 0.0, 1e-300, 2.0, np.inf])
    np.testing.assert_array_equal(
        SqrtLink().valid_eta(eta), [False, False, True, True, False]
    )
    np.testing.assert_array_equal(
        LogLink().valid_eta(eta), [True, True, True, True, False]
    )
    for family in (
        NegativeBinomial(theta=2.0, fixed=True, link="sqrt"),
        Poisson(link="sqrt"),
    ):
        np.testing.assert_array_equal(
            np.asarray(family.valid_eta(jnp.asarray(eta))),
            SqrtLink().valid_eta(eta),
        )
    np.testing.assert_array_equal(
        np.asarray(Poisson().valid_eta(jnp.asarray(eta))), LogLink().valid_eta(eta)
    )


def test_source_inner_loops_compile_and_converge_under_jit() -> None:
    """gam.fit3 and gam.fit4 inner loops run under jax.jit with dynamic theta."""
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.fitting.source_pirls import source_pirls

    rng = np.random.default_rng(5)
    n = 120
    x = np.sort(rng.uniform(0, 1, n))
    X = jnp.asarray(np.column_stack([np.ones(n), x, np.sin(4 * x), np.cos(4 * x)]))
    S = jnp.diag(jnp.asarray([0.0, 0.0, 1.0, 1.0]))
    counts = jnp.asarray(rng.poisson(np.exp(0.5 + np.sin(4 * x))).astype(float))
    wt = jnp.ones(n)
    null_beta = jnp.asarray([float(np.log(np.mean(counts))), 0.0, 0.0, 0.0])
    beta0 = jnp.zeros(4).at[0].set(null_beta[0])

    def regular(beta_start):
        return source_pirls(
            X, counts, beta_start, S, Gamma(link="log"), wt, None, null_beta
        ).pirls_result

    gamma_fit = jax.jit(regular)(beta0)
    assert bool(gamma_fit.converged)
    assert int(gamma_fit.inner_status) == 0

    estimated = NegativeBinomial(theta=2.0)
    indices = jnp.asarray(np.asarray(counts, dtype=np.int64))
    max_y = int(np.max(np.asarray(counts)))

    def nb(log_theta):
        return source_pirls(
            X,
            counts,
            beta0,
            S,
            estimated,
            wt,
            None,
            null_beta,
            log_theta=log_theta,
            count_indices=indices,
            max_y=max_y,
        ).pirls_result

    compiled = jax.jit(nb)
    for log_theta in (0.3, 1.2):
        result = compiled(jnp.asarray([log_theta]))
        assert bool(result.converged)
        assert int(result.inner_status) == 0


def _nb_sqrt_frame(seed: int, shift: float) -> pd.DataFrame:
    """The PR11 explicit-design NB case (n=160, cr+cr+re, weights, offset)."""
    n = 160
    position = np.arange(n, dtype=float)
    x = (position + 0.2 * np.sin(position * 0.7)) / n
    z = np.cos(position * 0.23) + 0.07 * np.sin(position * 0.91)
    levels = ("a", "b", "c", "d")
    g = pd.Categorical(np.resize(np.array(levels), n), categories=levels)
    eta = (
        shift
        + 1.2 * np.sin(5.0 * x)
        + 0.7 * np.cos(3.0 * z)
        + 0.6 * np.array([-1.0, 0.4, 1.0, -0.6])[np.asarray(g.codes)]
    )
    rng = np.random.default_rng(seed)
    mean = np.exp(eta)
    y = rng.negative_binomial(3.0, 3.0 / (3.0 + mean)).astype(float)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "g": g,
            "y": y,
            "w": np.linspace(0.6, 1.4, n),
            "off": np.linspace(-0.1, 0.2, n),
        }
    )


_NB_SQRT_FORMULA = "y ~ s(x,bs='cr',k=6) + s(z,bs='cr',k=5) + s(g,bs='re')"


_NB_SQRT_SP = np.array([0.6, 1.5, 0.8])


def _fit_nb_sqrt(data: pd.DataFrame):
    from jaxgam.families.negative_binomial import NegativeBinomial

    family = NegativeBinomial(theta=2.5, fixed=True, link="sqrt")
    return GAM(_NB_SQRT_FORMULA, family=family, sp=list(_NB_SQRT_SP)).fit(
        data, weights=data.w.to_numpy(), offset=data.off.to_numpy()
    )


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
@pytest.mark.parametrize(("seed", "shift"), [(11, 1.0), (3, 2.0)])
def test_nb_fixed_theta_sqrt_matches_pinned_r_where_r_converges(
    r_bridge, seed: int, shift: float
) -> None:
    """Where pinned gam.fit4 converges, the dense NB/sqrt fit is STRICT.

    Seed 11 stalled before: Fisher scoring contracted at about 0.93 per
    iteration near mu -> 0. Seed 3 started at eta < 0 and Newton without the
    sqrt domain reached a mirror stationary point.
    """
    data = _nb_sqrt_frame(seed, shift)
    with contextlib.redirect_stdout(io.StringIO()):
        reference = r_bridge.fit_fixed_sp(
            _NB_SQRT_FORMULA + " + offset(off)",
            data,
            "nb_sqrt",
            _NB_SQRT_SP,
            weights="w",
            theta=2.5,
        )
    fit = _fit_nb_sqrt(data)
    collector = _AssertCollector()
    collector.check(
        "converged",
        lambda: check_that(fit.converged and reference["converged"], "not converged"),
    )
    collector.check(
        "fitted", lambda: _assert_strict(fit.fitted_values, reference["fitted_values"])
    )
    collector.check(
        "coefficients",
        lambda: _assert_strict(fit.coefficients, reference["coefficients"]),
    )
    collector.check(
        "deviance", lambda: _assert_strict(fit.deviance, reference["deviance"])
    )
    collector.check(
        "sqrt domain",
        lambda: check_that(np.all(fit.linear_predictor > 0), "eta <= 0 returned"),
    )
    collector.raise_if_any(f"NB fixed-theta sqrt seed {seed}")


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_nb_and_poisson_sqrt_fail_truthfully_where_pinned_r_errors(r_bridge) -> None:
    """Where pinned R stops ("inner loop 2; can't correct step size") the dense
    fit reports non-convergence with the inner-loop reason; it never returns a
    converged fit at a boundary point or one with eta < 0 (the old drift)."""
    rpy2_error = pytest.importorskip("rpy2.rinterface_lib.embedded").RRuntimeError
    data = _nb_sqrt_frame(5, 0.3)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        pytest.raises(rpy2_error, match="can't correct step size"),
    ):
        r_bridge.fit_fixed_sp(
            _NB_SQRT_FORMULA + " + offset(off)",
            data,
            "nb_sqrt",
            _NB_SQRT_SP,
            weights="w",
            theta=2.5,
        )
    nb_fit = _fit_nb_sqrt(data)
    assert not nb_fit.converged
    assert "can't correct step size" in nb_fit.convergence_info

    rng = np.random.default_rng(1)
    n = 150
    x = np.sort(rng.uniform(0, 1, n))
    poisson_data = pd.DataFrame(
        {"x": x, "y": rng.poisson(np.exp(-1.5 + 2.5 * np.sin(6 * x))).astype(float)}
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        pytest.raises(rpy2_error, match="can't correct step size"),
    ):
        r_bridge.public_fixed_sp_fit(
            poisson_data,
            np.linspace(0.6, 1.4, n),
            np.zeros(n),
            "poisson",
            "sqrt",
            sp=0.1,
        )
    poisson_fit = GAM(
        'y ~ s(x, bs="cr", k=8)', family=Poisson(link="sqrt"), sp=[0.1]
    ).fit(poisson_data, weights=np.linspace(0.6, 1.4, n))
    assert not poisson_fit.converged
    assert "inner fit did not converge" in poisson_fit.convergence_info
    assert np.all(poisson_fit.linear_predictor > 0)


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_poisson_sqrt_near_the_boundary_matches_pinned_r(r_bridge) -> None:
    """Near mu -> 0 the Newton (alpha) inner fit reproduces R at STRICT.

    The former Fisher loop stopped about 9e-9 short on this case.
    """
    rng = np.random.default_rng(7)
    n = 150
    x = np.sort(rng.uniform(0, 1, n))
    data = pd.DataFrame(
        {"x": x, "y": rng.poisson(np.exp(-2.0 + 2.5 * np.sin(6 * x))).astype(float)}
    )
    weight = np.linspace(0.6, 1.4, n)
    collector = _AssertCollector()
    for sp in (0.01, 0.1, 1.0):
        with contextlib.redirect_stdout(io.StringIO()):
            reference = r_bridge.public_fixed_sp_fit(
                data, weight, np.zeros(n), "poisson", "sqrt", sp=sp
            )
        fit = GAM('y ~ s(x, bs="cr", k=8)', family=Poisson(link="sqrt"), sp=[sp]).fit(
            data, weights=weight
        )
        for name, actual, expected in (
            ("fitted", fit.fitted_values, reference["fitted_values"]),
            ("coefficients", fit.coefficients, reference["coefficients"]),
            ("deviance", fit.deviance, reference["deviance"]),
        ):
            collector.check(
                f"sp={sp}: {name}",
                lambda actual=actual, expected=expected: _assert_strict(
                    actual, expected
                ),
            )
        _assert_hard_gates(fit, f"sp={sp}", collector)
    collector.raise_if_any("Poisson sqrt near the boundary")


def test_source_status_messages_name_r_stopping_reasons() -> None:
    """Every non-converged inner fit reports a specific, R-worded reason."""
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.fitting import efs_regular_pirls, pirls
    from jaxgam.fitting.source_pirls import status_message, uses_source_loop

    nb = NegativeBinomial(theta=2.0, fixed=True)
    assert "iteration limit" in status_message(Gaussian(), 0)
    assert "inner loop 2" in status_message(
        Poisson(), efs_regular_pirls.EFS_REGULAR_STATUS_DOMAIN_RECOVERY_FAILED
    )
    assert "inner loop 3" in status_message(
        nb, pirls._EFS_STATUS_DIVERGENCE_RECOVERY_FAILED
    )
    assert status_message(Gamma(), 99) == "inner-loop status 99"
    assert all(uses_source_loop(f) for f in (Gaussian(), Binomial(), Poisson(), nb))


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_gam_fit3_fisher_retry_is_taken_at_pinned_r_states(r_bridge) -> None:
    """The dense gam.fit3 loop retries with Fisher weights exactly where R does.

    Matched state: at every iteration the JAX loop's own state is handed to
    pinned ``C_pls_fit1`` with the Newton (alpha) weights; R's indefinite flag
    (``oo$n < 0``) must equal JAX's.  The loop runs on the QR route (an empty
    penalty root) under ``jax.jit`` and reproduces pinned gam.fit3's recovery
    count, coefficients, deviance and Fisher covariance ``rV rV'``.
    """
    from jaxgam.fitting.efs_regular_pirls import (
        _checked_exact_solve,
        _form_system,
        _regular_factors,
        efs_regular_pirls_loop,
    )

    rng = np.random.default_rng(1)
    x = np.linspace(-0.9, 0.9, 41)
    y = np.exp(rng.normal(0.0, 2.0, len(x)))
    weight = 0.1 + rng.uniform(size=len(x))
    offset = 0.02 * np.sin(x)
    X = np.column_stack((np.ones(len(x)), x))
    with contextlib.redirect_stdout(io.StringIO()):
        oracle = r_bridge.gamma_fisher_recovery_reference(X, y, weight, offset)
    family = Gamma("identity")
    Xj, yj, wj, oj = (jnp.asarray(v) for v in (X, y, weight, offset))
    null_beta = jnp.asarray([np.mean(y), 0.0])

    def run(max_iter: int):
        # R: eta = linkfun(mustart) with mustart = y, scale = exp(log 0.7).
        return efs_regular_pirls_loop(
            Xj,
            yj,
            jnp.zeros(2),
            yj,
            null_beta,
            Xj @ null_beta + oj,
            jnp.zeros((2, 2)),
            family,
            wj,
            oj,
            jnp.asarray(0.7),
            start_present=False,
            max_iter=max_iter,
            tol=1e-10,
            penalty_root=jnp.zeros((0, 2)),
        )

    result = run(100)
    fit = result.pirls_result
    collector = _AssertCollector()
    collector.check("converged", lambda: check_that(bool(fit.converged), "failed"))
    collector.check(
        "recovery count",
        lambda: check_that(
            int(result.n_fisher_fallbacks) == oracle["recoveries"],
            f"{int(result.n_fisher_fallbacks)} != R {oracle['recoveries']}",
        ),
    )
    collector.check(
        "coefficients and deviance",
        lambda: _assert_strict(
            np.r_[fit.coefficients, fit.deviance], oracle["reference"][:3]
        ),
    )
    collector.check(
        "Fisher covariance",
        lambda: _assert_strict(
            np.asarray(jax.scipy.linalg.cho_solve((fit.L_fisher, True), jnp.eye(2))),
            oracle["covariance"],
        ),
    )
    retries = 0
    for completed in range(int(fit.n_iter)):
        state = run(completed).pirls_result if completed else None
        eta = yj if state is None else state.eta
        mu = family.link.inverse(eta)
        W, z, _ = _regular_factors(family, yj, wj, oj, mu, eta, fisher=False)
        XtWX, rhs = _form_system(Xj, W, z)
        _, _, _, indefinite = _checked_exact_solve(XtWX, jnp.zeros((2, 2)), rhs)
        with contextlib.redirect_stdout(io.StringIO()):
            r_rank, _, _ = r_bridge.source_signed_pls_fit1(
                X,
                np.asarray(W),
                np.asarray(z),
                np.asarray(W * z),
                np.zeros((1, 2)),
                use_weighted_response=False,
            )
        retries += int(bool(indefinite))
        collector.check(
            f"iteration {completed + 1}: indefinite flag",
            lambda indefinite=indefinite, r_rank=r_rank: check_that(
                bool(indefinite) == (r_rank < 0),
                f"JAX {bool(indefinite)} vs R n={r_rank}",
            ),
        )
    collector.check(
        "retry replay",
        lambda: check_that(retries == oracle["recoveries"] > 0, f"{retries} retries"),
    )
    collector.raise_if_any("gam.fit3 Fisher retry at matched states")


def test_source_loops_hold_the_hard_gates_under_jit() -> None:
    """Objective monotonicity, H symmetry/PSD, penalty PSD and rank, compiled.

    Each source loop (canonical Poisson and noncanonical Gamma/log on the QR
    route, fixed-theta NB/sqrt on gam.fit4) is replayed under ``jax.jit`` with
    an increasing iteration cap: the penalized deviance never rises by more
    than R's divergence threshold, and the converged penalized Hessian (the
    Newton one for noncanonical links) is symmetric positive definite.
    """
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.fitting.source_pirls import source_pirls

    rng = np.random.default_rng(17)
    n = 150
    x = np.sort(rng.uniform(0.0, 1.0, n))
    X = jnp.asarray(
        np.column_stack([np.ones(n), x, np.sin(4 * x), np.cos(4 * x), np.sin(7 * x)])
    )
    S = np.diag([0.0, 0.0, 1.0, 1.0, 2.0])
    root = jnp.asarray(np.sqrt(S)[2:])
    signal = 0.8 + 0.6 * np.sin(4 * x)
    counts = jnp.asarray(rng.poisson(np.exp(signal)).astype(float))
    positive = jnp.asarray(rng.gamma(3.0, np.exp(signal) / 3.0))
    wt = jnp.linspace(0.7, 1.3, n)
    nb = NegativeBinomial(theta=2.5, fixed=True, link="sqrt")
    cases = (
        ("poisson/log", Poisson(), counts, jnp.log(jnp.mean(counts)), root),
        ("gamma/log", Gamma(link="log"), positive, jnp.log(jnp.mean(positive)), root),
        ("nb/sqrt", nb, counts, jnp.sqrt(jnp.mean(counts)), None),
    )
    collector = _AssertCollector()
    for label, family, y, intercept, penalty_root in cases:
        null_beta = jnp.zeros(5).at[0].set(intercept)
        start = null_beta.at[1].set(0.1)

        def fit(
            max_iter,
            family=family,
            y=y,
            null_beta=null_beta,
            start=start,
            penalty_root=penalty_root,
        ):
            return source_pirls(
                X,
                y,
                start,
                jnp.asarray(S),
                family,
                wt,
                None,
                null_beta,
                penalty_root=penalty_root,
                max_iter=max_iter,
            ).pirls_result

        compiled = jax.jit(fit, static_argnums=0)
        final = compiled(200)
        collector.check(
            f"{label}: converged",
            lambda final=final: check_that(
                bool(final.converged) and int(final.inner_status) == 0, "failed"
            ),
        )
        history = []
        for cap in range(1, int(final.n_iter) + 1):
            step = compiled(cap)
            beta = np.asarray(step.coefficients)
            history.append(float(step.deviance) + beta @ S @ beta)

        def _monotone(history=history) -> None:
            for before, after in itertools.pairwise(history):
                threshold = 10.0 * (0.1 + abs(before)) * np.sqrt(np.finfo(float).eps)
                check_that(after - before <= threshold, f"{before} -> {after}")

        collector.check(f"{label}: penalized deviance monotone", _monotone)

        def _hessian(final=final) -> None:
            H = np.asarray(final.XtWX) + S
            _assert_strict(H, H.T)
            eig = np.linalg.eigvalsh(H)
            check_that(eig.min() > 0.0, f"H not PD: {eig.min()}")
            L = np.asarray(final.L_fisher)
            check_that(np.all(np.diag(L) > 0.0), "Fisher factor not PD")

        collector.check(f"{label}: H symmetric PD", _hessian)
    collector.check(
        "penalty PSD and rank",
        lambda: check_that(
            np.linalg.eigvalsh(S).min() >= 0.0
            and np.linalg.matrix_rank(np.asarray(X)) == X.shape[1],
            "penalty or design rank",
        ),
    )
    collector.raise_if_any("source loop hard gates")


@pytest.mark.skipif(not r_available(), reason="Pinned R/mgcv unavailable")
def test_gam_fit4_positive_weight_retry_is_taken_where_pinned_pls_fit1_is_indefinite(
    r_bridge,
) -> None:
    """gam.fit4's positive-weight branch at a chosen indefinite state (3e).

    NB/identity at a state where zero counts give negative observed weights
    and the penalty is too weak to make X'WX + S positive definite: JaxGAM's
    observed Cholesky must fail exactly where pinned C_pls_fit1 returns
    n < 0, and JaxGAM must take the positive-weight retry.

    STOP item (review packet, root decision 8): the retry's *coefficients*
    differ from pinned R's.  gam.fit4 (gam.fit4.r 391-416) keeps -Deta/2 in
    wz for the zero-weight rows, so pls_fit1's stability test switches to
    the X'wz right side (gdi.c 3121-3133); JaxGAM uses w*z, which drops those
    rows' gradient.  The fix (always use wz in the retry) changes the outcome
    of the existing test_efs_theta_pirls.py::
    test_efs_indefinite_observed_system_retries_positive_curvature_under_jit,
    whose hand-derived expectation encodes the w*z rule, so it is not
    applied here pending the root decision; the measured gap is recorded.
    """
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.fitting import pirls

    rng = np.random.default_rng(16)
    n = 60
    x = np.linspace(0.0, 1.0, n)
    X = np.column_stack([np.ones(n), x, np.sin(4 * x), np.cos(4 * x)])
    mu = 4.0 + 1.2 * np.sin(5 * x)
    y = np.where(rng.uniform(size=n) < 0.6, 0.0, rng.poisson(2.0 * mu)).astype(float)
    E = np.diag([0.0, 0.0, 1e-3, 1e-3])[2:]
    S = E.T @ E
    family = NegativeBinomial(theta=2.0, fixed=True, link="identity")
    log_theta = jnp.reshape(jnp.log(2.0), (1,))
    eta = jnp.asarray(mu)
    offset = jnp.zeros(n)
    wt = jnp.ones(n)
    factors = pirls._efs_nb_observed_working_factors(
        eta, log_theta, jnp.asarray(y), wt, offset, family
    )

    def compute_working_factors(_mu, eta_value):
        return pirls._efs_nb_observed_working_factors(
            eta_value, log_theta, jnp.asarray(y), wt, offset, family
        )

    def form_wls(f):
        Wc = jnp.clip(f.weight, -pirls._W_MAX, pirls._W_MAX)
        rhs = jnp.where(f.use_weighted_response, f.weighted_response, Wc * f.response)
        Xj = jnp.asarray(X)
        return (Wc[:, None] * Xj).T @ Xj, Xj.T @ rhs

    def compute_dev(_mu, eta_value):
        return pirls._efs_nb_deviance(eta_value, log_theta, jnp.asarray(y), wt, family)

    beta0 = jnp.asarray(np.linalg.lstsq(X, mu, rcond=None)[0])
    step = jax.jit(
        lambda b: pirls._efs_beta_step_with_recovery(
            X=jnp.asarray(X),
            S_lambda=jnp.asarray(S),
            offset=offset,
            family=family,
            beta=b,
            eta=eta,
            mu=family.link.inverse(eta),
            beta_old=b,
            eta_old=eta,
            null_beta=b,
            null_eta=eta,
            initial_start_retained=jnp.asarray(False),
            iteration=jnp.asarray(1),
            baseline=jnp.asarray(1e300),
            compute_working_factors=compute_working_factors,
            form_wls=form_wls,
            compute_dev=compute_dev,
            max_recovery_halvings=100,
        )
    )(beta0)
    # Private host copies: the pinned .C entry may write into its inputs.
    W = np.array(factors.weight, copy=True)
    z = np.array(factors.response, copy=True)
    wz = np.array(factors.weighted_response, copy=True)
    use_wy = bool(factors.use_weighted_response)
    with contextlib.redirect_stdout(io.StringIO()):
        r_rank, _, _ = r_bridge.source_signed_pls_fit1(
            X,
            W.copy(),
            z.copy(),
            wz.copy(),
            E,
            use_weighted_response=use_wy,
        )
    collector = _AssertCollector()
    collector.check(
        "state is indefinite with negative weights",
        lambda: check_that(np.any(W < 0) and r_rank < 0, f"R n={r_rank}"),
    )
    collector.check(
        "JaxGAM takes the positive-weight retry",
        lambda: check_that(bool(step.positive_curvature_retry), "no retry"),
    )
    collector.raise_if_any("gam.fit4 positive-weight retry")
