"""Supplied prior weights and offset in pinned ``discrete.mf``'s seeded replay.

Pinned ``bam`` builds ``pmf`` with ``model.frame(gp$pf, weights=, offset=)``,
so a supplied ``weights=`` or ``offset=`` argument is one more parametric
column, ``(weights)``/``(offset)``, whatever its values (all ones included).
``discrete.mf`` compresses it after the formula's parametric variables, which
draws a shuffle from its seeded stream, and counts its unique rows toward
``maxr``. Both change the draws that pad a short metric margin, and ``gam.setup``
takes each smooth's centering constraint over that padded frame. A source
declares what it supplies with ``has_explicit_weights``/``has_explicit_offset``;
an undeclared attribute counts as supplied.

Gates on exactly representable inputs, STRICT except the selected smoothing
parameters (MODERATE; see ``_SELECTED_SP_TOLERANCE``):

* the compact frame, its padding, index and design against pinned
  ``bam(fit=FALSE)`` called with the fit's own arguments, in five settings;
* full fixed and selected fits against pinned ``bam`` on padded inputs;
* a weights column that alone raises ``maxr`` and so pads a margin;
* an undeclared caller-written source against ``bam`` with explicit
  ``weights=rep(1, n)``/``offset=rep(0, n)``, and the same source declaring
  ``False`` against ``bam`` without them;
* host-only checks that a supplied column is replayed, in order, and released.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.continuous_frame import CompactSetupFrame, build_compact_setup_frame
from jaxgam.data.source import DataFrameRowSource, RowBatch, declared_row_vectors
from jaxgam.execution.discrete.regular import regular_working_components
from jaxgam.formula.discrete.continuous_setup import build_continuous_discrete_setup
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector, check_that
from tests.test_execution.test_discrete_regular import (
    _BOUNDARY_FORMULA,
    _BOUNDARY_SP,
    _LEVELS,
    _close,
    _control,
    _coslike,
    _explike,
    _family,
    _follow,
    _sinlike,
    _source_decisions,
)
from tests.tolerances import MODERATE, STRICT

_FORMULA = _BOUNDARY_FORMULA
_M = 30


def _counts(rng, mean, trials):
    """Count of ``trials`` Bernoulli(mean/trials) draws: PCG64 bits and ``<``."""
    uniforms = rng.random((len(mean), trials))
    return (uniforms < (mean / trials)[:, None]).sum(axis=1).astype(float)


def _padded_frame(seed: int = 5, trials: int = 16, n: int = 150):
    """Low-count Poisson inputs from exact operations; ``z`` fills 28 of 30 grid
    points, so ``discrete.mf`` pads it with two seeded draws."""
    position = np.arange(n, dtype=float)
    x = (position + 0.3 * _sinlike(position)) / n
    z = _coslike(position * 0.19)
    g = pd.Categorical(np.resize(np.array(_LEVELS), n), categories=_LEVELS)
    eta = (
        _sinlike(4.0 * x)
        + 0.6 * _coslike(3.0 * z)
        + np.array([-0.3, 0.0, 0.2, 0.1])[np.asarray(g.codes)]
    )
    y = _counts(np.random.default_rng(seed), 0.08 * _explike(eta), trials)
    frame = pd.DataFrame({"x": x, "z": z, "g": g, "y": y})
    return frame, np.linspace(0.7, 1.3, n), np.linspace(-0.05, 0.05, n)


def _short_margin_frame(n: int = 150):
    """``x`` and ``z`` take 25 values each (all kept, no rounding at m = 30), so
    without weights ``maxr`` is 25 and nothing metric is padded; the weights'
    30 rounded rows alone raise ``maxr`` to 30 and pad both margins by five."""
    position = np.arange(n, dtype=float)
    x = np.mod(7.0 * position, 25.0) / 24.0
    z = np.mod(11.0 * position, 25.0) / 12.0 - 1.0
    g = pd.Categorical(np.resize(np.array(_LEVELS), n), categories=_LEVELS)
    eta = (
        0.5
        + _sinlike(4.0 * x)
        + 0.6 * _coslike(3.0 * z)
        + np.array([-0.3, 0.0, 0.2, 0.1])[np.asarray(g.codes)]
    )
    y = _counts(np.random.default_rng(3), 0.4 * _explike(eta), 16)
    frame = pd.DataFrame({"x": x, "z": z, "g": g, "y": y})
    return frame, np.linspace(0.7, 1.3, n)


# Settings: (label, use weights, use offset, all-ones weights).
_SETTINGS = {
    "neither": (False, False, False),
    "weights": (True, False, False),
    "offset": (False, True, False),
    "both": (True, True, False),
    "ones": (True, False, True),
}


def _arguments(setting, weights, offset):
    use_weights, use_offset, ones = _SETTINGS[setting]
    return {
        "weights": (np.ones_like(weights) if ones else weights)
        if use_weights
        else None,
        "offset": offset if use_offset else None,
    }


def _source(frame, *, weights=None, offset=None) -> DataFrameRowSource:
    return DataFrameRowSource(frame, response="y", weights=weights, offset=offset)


def _compact(source) -> CompactSetupFrame:
    result = build_compact_setup_frame(
        parse_formula(_FORMULA), source, factor_names=("g",), m=_M, batch_rows=17
    )
    assert isinstance(result, CompactSetupFrame)
    return result


def _design(source):
    """The explicit design at the snapped covariates of ``source``'s setup."""
    setup = build_continuous_discrete_setup(
        parse_formula(_FORMULA), source, m=_M, factor_names=("g",)
    )
    try:
        snapped = {
            name: values[group.index.selector]
            for group in setup.compact.groups
            for name, values in zip(group.names, group.index.values, strict=True)
        }
        return setup.predict_spec.build_predict_matrix(pd.DataFrame(snapped))
    finally:
        setup.cleanup()


class _CallerSource:
    """A caller-written source: batches only, no weight/offset declarations."""

    def __init__(self, frame, weights, offset):
        self._inner = _source(frame, weights=weights, offset=offset)

    @property
    def n_rows(self) -> int:
        return self._inner.n_rows

    def fingerprint(self) -> str:
        return self._inner.fingerprint()

    def scan(self, batch_rows: int):
        for batch in self._inner.scan(batch_rows):
            yield RowBatch(
                batch.columns,
                batch.y,
                batch.weight,
                batch.offset,
                batch.valid,
                batch.row_positions,
            )


class _DeclaredDefaultsSource(_CallerSource):
    """The same caller-written source declaring that it supplies neither."""

    has_explicit_weights = False
    has_explicit_offset = False


# --------------------------------------------------------------------------
# Host-only replay contracts.
# --------------------------------------------------------------------------


def test_declared_row_vectors_follow_the_caller() -> None:
    frame, weights, offset = _padded_frame()
    assert declared_row_vectors(_source(frame)) == (False, False)
    assert declared_row_vectors(_source(frame, weights=weights)) == (True, False)
    assert declared_row_vectors(_source(frame, offset=offset)) == (False, True)
    ones = _source(frame, weights=np.ones(len(frame)), offset=np.zeros(len(frame)))
    assert declared_row_vectors(ones) == (True, True)
    caller = _CallerSource(frame, None, None)
    assert declared_row_vectors(caller) == (True, True)
    assert declared_row_vectors(_DeclaredDefaultsSource(frame, None, None)) == (
        False,
        False,
    )

    class Malformed(_CallerSource):
        has_explicit_weights = 1

    with pytest.raises(TypeError, match="must be booleans"):
        declared_row_vectors(Malformed(frame, None, None))


def test_supplied_columns_are_replayed_in_model_frame_order() -> None:
    """Dropping or reordering a supplied column changes these padded frames."""
    frame, weights, offset = _padded_frame()
    neither = _compact(_source(frame))
    both = _compact(_source(frame, weights=weights, offset=offset))
    weights_only = _compact(_source(frame, weights=weights))
    ones = _compact(_source(frame, weights=np.ones(len(frame))))
    assert neither.replay_only_rows == ()
    assert both.replay_only_rows == (("(weights)", 30), ("(offset)", 30))
    assert weights_only.replay_only_rows == (("(weights)", 30),)
    assert ones.replay_only_rows == (("(weights)", 1),)
    for compact in (neither, both, weights_only, ones):
        assert tuple(compact.frame) == ("y", "g", "x", "z")
        assert compact.n_unique["z"] == 28
        assert len(compact.frame) == 30
    # The two padded z rows are seeded draws after every compressed column.
    padded = slice(28, 30)
    for other in (both, weights_only, ones):
        assert not np.array_equal(
            neither.frame["z"].to_numpy()[padded], other.frame["z"].to_numpy()[padded]
        )
    assert not np.array_equal(
        both.frame["z"].to_numpy()[padded], weights_only.frame["z"].to_numpy()[padded]
    )
    # Representatives and the index itself are drawn before the extra columns.
    for other in (both, weights_only, ones):
        np.testing.assert_array_equal(
            neither.frame["z"].to_numpy()[:28], other.frame["z"].to_numpy()[:28]
        )
        for mine, theirs in zip(neither.groups, other.groups, strict=True):
            np.testing.assert_array_equal(mine.index.selector, theirs.index.selector)
    # An undeclared source supplies both; declaring False supplies neither.
    caller = _compact(_CallerSource(frame, None, None))
    declared = _compact(_DeclaredDefaultsSource(frame, None, None))
    assert caller.replay_only_rows == (("(weights)", 1), ("(offset)", 1))
    assert declared.replay_only_rows == ()
    pd.testing.assert_frame_equal(declared.frame, neither.frame)
    assert not np.array_equal(
        caller.frame["z"].to_numpy()[padded], neither.frame["z"].to_numpy()[padded]
    )


def test_supplied_column_alone_raises_the_padded_length() -> None:
    frame, weights = _short_margin_frame()
    neither = _compact(_source(frame))
    supplied = _compact(_source(frame, weights=weights))
    assert neither.n_unique["x"] == neither.n_unique["z"] == 25
    assert len(neither.frame) == 25
    assert supplied.replay_only_rows == (("(weights)", 30),)
    assert len(supplied.frame) == 30
    np.testing.assert_array_equal(
        supplied.frame["x"].to_numpy()[:25], neither.frame["x"].to_numpy()
    )


def test_replay_only_selectors_are_released(tmp_path) -> None:
    frame, weights, offset = _padded_frame()
    compact = build_compact_setup_frame(
        parse_formula(_FORMULA),
        _source(frame, weights=weights, offset=offset),
        factor_names=("g",),
        m=_M,
        batch_rows=17,
        selector_memmap_dir=tmp_path,
    )
    assert isinstance(compact, CompactSetupFrame)
    try:
        assert len(compact.groups) == 4
        assert len(compact.selector_paths) == 4
        assert sorted(tmp_path.iterdir()) == sorted(compact.selector_paths)
        in_memory = _compact(_source(frame, weights=weights, offset=offset))
        neither = _compact(_source(frame))
        assert in_memory.retained_group_bytes == neither.retained_group_bytes
        assert in_memory.retained_selector_bytes == neither.retained_selector_bytes
    finally:
        compact.cleanup()


# --------------------------------------------------------------------------
# Pinned bam(fit=FALSE) with the fit's own arguments.
# --------------------------------------------------------------------------


def _check_compact_against_prefit(collector, label, compact, design, prefit):
    collector.check(
        f"{label} design",
        _close(design, prefit.expanded_design, STRICT),
    )
    for name in compact.frame:
        observed = compact.frame[name]
        expected = prefit.frame[name]
        if isinstance(observed.dtype, pd.CategoricalDtype):
            collector.check(
                f"{label} padded {name}",
                lambda o=observed, e=expected: np.testing.assert_array_equal(
                    np.asarray(o.astype(str)), np.asarray(e, dtype=str)
                ),
            )
        else:
            collector.check(
                f"{label} padded {name}",
                _close(observed.to_numpy(), expected.to_numpy(dtype=float), STRICT),
            )
    for column, group in enumerate(compact.groups):
        collector.check(
            f"{label} index {group.names}",
            lambda c=column, g=group: np.testing.assert_array_equal(
                g.index.selector, prefit.row_indices[:, c]
            ),
        )
    collector.check(
        f"{label} unique counts",
        lambda: check_that(
            all(prefit.n_unique[k] == v for k, v in compact.n_unique.items())
            and all(prefit.n_unique[k] == v for k, v in compact.replay_only_rows)
            and set(prefit.n_unique) - {"(Intercept)"}
            == set(compact.frame) | {name for name, _ in compact.replay_only_rows},
            f"{compact.n_unique=} {compact.replay_only_rows=} vs {prefit.n_unique=}",
        ),
    )


@pytest.mark.platform_sensitive
@pytest.mark.parametrize("setting", tuple(_SETTINGS))
def test_compact_frame_index_and_design_match_bam_prefit(r_bridge, setting) -> None:
    frame, weights, offset = _padded_frame()
    arguments = _arguments(setting, weights, offset)
    source = _source(frame, **arguments)
    prefit = r_bridge.bam_discrete_prefit(
        _FORMULA,
        frame,
        family="poisson",
        link="log",
        factor_names=("g",),
        m=_M,
        **arguments,
    )
    expected_names = (
        "y",
        "g",
        *(
            name
            for name, key in (("(weights)", "weights"), ("(offset)", "offset"))
            if arguments[key] is not None
        ),
    )
    collector = _AssertCollector()
    collector.check(
        "pmf.names",
        lambda: check_that(
            prefit.parametric_names == expected_names,
            f"{prefit.parametric_names=}",
        ),
    )
    _check_compact_against_prefit(
        collector, setting, _compact(source), _design(source), prefit
    )
    collector.raise_if_any(f"{setting} compact frame vs pinned bam(fit=FALSE)")


@pytest.mark.platform_sensitive
def test_supplied_column_alone_raising_maxr_matches_bam_prefit(r_bridge) -> None:
    frame, weights = _short_margin_frame()
    collector = _AssertCollector()
    for label, supplied in (("neither", None), ("weights", weights)):
        source = _source(frame, weights=supplied)
        prefit = r_bridge.bam_discrete_prefit(
            _FORMULA,
            frame,
            family="poisson",
            link="log",
            factor_names=("g",),
            m=_M,
            weights=supplied,
        )
        compact = _compact(source)
        collector.check(
            f"{label} premise",
            lambda c=compact, s=supplied: check_that(
                len(c.frame) == (25 if s is None else 30),
                f"{len(c.frame)=}",
            ),
        )
        _check_compact_against_prefit(
            collector, label, compact, _design(source), prefit
        )
    collector.raise_if_any("maxr raised by a supplied weights column")


# --------------------------------------------------------------------------
# Full fits against pinned bam.
# --------------------------------------------------------------------------


# Selected smoothing parameters are the end point of two independent optimizer
# runs on a criterion that is flat where a smooth is nearly linear, so they are
# compared at MODERATE; every physical field of the same fit (coefficients,
# fitted values, deviance, EDF, scale, score, covariance) stays at STRICT, and
# fixed-sp fits have no such field. Measured against pinned bam on seed 6/16:
# one parameter of order 1.8e7 differs by 1.5e-10 relative (1.51 x STRICT) on
# x86_64 without AVX-512 (run 44085978878); 0.25-0.50 x STRICT on macOS arm64,
# and within STRICT on x86_64 with AVX-512.
_SELECTED_SP_FIELD = "smoothing parameters"
_SELECTED_SP_TOLERANCE = MODERATE


def _fit_fields(r_bridge, frame, source, *, sp, selected, r_weights, r_offset, offset):
    """Fit ``source`` following pinned R's decisions; return named field pairs."""
    trace = r_bridge.trace_bam_regular_working_model(
        _FORMULA,
        frame,
        family="poisson",
        link="log",
        sp=sp,
        factor_names=("g",),
        weights=r_weights,
        offset=r_offset,
    )
    family = _family("poisson", "log")
    grams = []

    def run(replay):
        with regular_working_components(
            parse_formula(_FORMULA),
            source,
            family,
            sp,
            control=_control(selected, uncertainty="covariance"),
        ) as parts:
            fit = parts.run(_oracle_replay_halvings=replay)
            grams.append(fit.gram)
            return parts.result(fit)

    followed = _follow(run, _source_decisions(trace))
    fit, expected = followed.fit, trace.fit
    X = _design(source)
    fitted = np.asarray(family.link.inverse(X @ fit.coefficients + offset))
    coefficient_edf = np.diag(fit.covariance / fit.scale @ grams[-1])
    blocks = {"parametric": slice(0, 4), "s(x)": slice(4, 9), "s(z)": slice(9, 13)}
    fields = [
        ("coefficients", fit.coefficients, expected.coefficients),
        ("fitted values", fitted, expected.fitted_values),
        ("deviance", fit.deviance, expected.deviance),
        ("edf", fit.edf_total, expected.edf_total),
        *(
            (
                f"edf {name}",
                coefficient_edf[block].sum(),
                expected.coefficient_edf[block].sum(),
            )
            for name, block in blocks.items()
        ),
        ("scale", fit.scale, expected.scale),
        ("score", fit.score, expected.score),
        ("covariance", fit.covariance, expected.covariance),
    ]
    if selected:
        fields.append(
            (_SELECTED_SP_FIELD, fit.smoothing_params, expected.smoothing_params)
        )
    return followed, fit, expected, fields


def _gate_fit(collector, label, followed, fit, expected, fields):
    collector.check(
        f"{label} decisions",
        lambda: check_that(
            not followed.problems and not followed.replayed,
            f"{followed.problems=} {followed.replayed=}",
        ),
    )
    collector.check(
        f"{label} iterations",
        lambda: check_that(
            fit.converged
            and expected.mgcv_converged
            and fit.scale_iterations == expected.n_iter,
            f"{fit.converged=} {fit.scale_iterations=} vs pinned {expected.n_iter}",
        ),
    )
    for name, observed, target in fields:
        tolerance = _SELECTED_SP_TOLERANCE if name == _SELECTED_SP_FIELD else STRICT
        collector.check(f"{label} {name}", _close(observed, target, tolerance))


# Seed 7 (16 trials) is deliberately absent: factor level "a" has no counts,
# so the fit separates (intercept near -18.5, penalized Hessian condition
# number 4.8e7 at the end). That is a separated fit, not a parity question.
_PADDED_SEEDS = ((5, 16), (6, 16), (5, 32))


@pytest.mark.platform_sensitive
@pytest.mark.parametrize("selected", [False, True], ids=["fixed", "selected"])
@pytest.mark.parametrize(
    ("seed", "trials"), _PADDED_SEEDS, ids=[f"s{s}-t{t}" for s, t in _PADDED_SEEDS]
)
def test_padded_poisson_fit_matches_pinned_bam(r_bridge, seed, trials, selected):
    frame, weights, offset = _padded_frame(seed, trials)
    sp = None if selected else _BOUNDARY_SP
    followed, fit, expected, fields = _fit_fields(
        r_bridge,
        frame,
        _source(frame, weights=weights, offset=offset),
        sp=sp,
        selected=selected,
        r_weights=weights,
        r_offset=offset,
        offset=offset,
    )
    collector = _AssertCollector()
    _gate_fit(collector, "fit", followed, fit, expected, fields)
    collector.raise_if_any(
        f"seed {seed}/{trials} {'selected' if selected else 'fixed'} vs pinned bam"
    )


@pytest.mark.platform_sensitive
def test_maxr_raising_weights_fit_matches_pinned_bam(r_bridge) -> None:
    frame, weights = _short_margin_frame()
    followed, fit, expected, fields = _fit_fields(
        r_bridge,
        frame,
        _source(frame, weights=weights),
        sp=_BOUNDARY_SP,
        selected=False,
        r_weights=weights,
        r_offset=None,
        offset=0.0,
    )
    collector = _AssertCollector()
    _gate_fit(collector, "fit", followed, fit, expected, fields)
    collector.raise_if_any("maxr-raising weights fixed fit vs pinned bam")


@pytest.mark.platform_sensitive
def test_undeclared_caller_source_fits_as_bam_with_supplied_vectors(r_bridge):
    frame, _, _ = _padded_frame()
    n = len(frame)
    ones, zeros = np.ones(n), np.zeros(n)
    collector = _AssertCollector()
    undeclared = _fit_fields(
        r_bridge,
        frame,
        _CallerSource(frame, ones, zeros),
        sp=_BOUNDARY_SP,
        selected=False,
        r_weights=ones,
        r_offset=zeros,
        offset=zeros,
    )
    _gate_fit(collector, "undeclared vs bam(weights=1, offset=0)", *undeclared)
    without = _fit_fields(
        r_bridge,
        frame,
        _DeclaredDefaultsSource(frame, ones, zeros),
        sp=_BOUNDARY_SP,
        selected=False,
        r_weights=None,
        r_offset=None,
        offset=zeros,
    )
    _gate_fit(collector, "declared False vs bam()", *without)
    # The distinction is live: pinned bam's own two fits differ in coefficients.
    supplied_r, omitted_r = undeclared[2], without[2]
    collector.check(
        "pinned bam with and without the vectors differs",
        lambda: check_that(
            not np.allclose(
                supplied_r.coefficients,
                omitted_r.coefficients,
                rtol=STRICT.rtol,
                atol=STRICT.atol,
            ),
            "pinned bam(weights=1, offset=0) equals bam() at STRICT",
        ),
    )
    collector.raise_if_any("caller-written source vs pinned bam")
