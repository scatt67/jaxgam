"""Pinned ``bam(discrete=)`` setup gates for continuous descriptors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaxgam.data.source import ArrayRowSource
from jaxgam.fitting.discrete.ops import (
    discrete_diag_xvxd,
    discrete_xbd,
    discrete_xwxd,
    discrete_xwyd,
    to_device_discrete_design,
)
from jaxgam.formula.discrete.continuous_design import build_continuous_discrete_design
from jaxgam.formula.discrete.continuous_setup import (
    ContinuousDiscreteSetup,
    build_continuous_discrete_setup,
)
from jaxgam.formula.discrete.design import (
    DiscreteDesignControl,
    DiscreteDesignUnavailable,
    TensorLookupBlock,
)
from jaxgam.formula.parser import parse_formula
from tests.helpers import _AssertCollector
from tests.r_bridge import RBridgeError
from tests.tolerances import STRICT


def _factor_tensor_frame(n_rows: int = 72) -> pd.DataFrame:
    positions = np.arange(1, n_rows + 1, dtype=float)
    return pd.DataFrame(
        {
            "y": np.linspace(0.2, 1.8, n_rows),
            "x": (positions - 0.5) / n_rows + 0.03 * np.sin(positions),
            "w": np.cos(positions * 0.71) + positions / n_rows,
            "g": pd.Series(
                pd.Categorical(
                    np.resize(np.array(("low", "mid", "high")), n_rows),
                    categories=("low", "mid", "high"),
                    ordered=True,
                )
            ),
        }
    )


def _source(frame: pd.DataFrame) -> ArrayRowSource:
    return ArrayRowSource({name: frame[name] for name in frame})


def _r_constraint(vector: np.ndarray) -> np.ndarray:
    return (np.eye(len(vector)) - np.outer(vector, vector))[:, 1:]


def test_factor_tensor_setup_penalties_and_actions_match_pinned_bam(r_bridge) -> None:
    """Translate factor-by tensor ``v/qc`` and per-level penalties exactly."""
    frame = _factor_tensor_frame()
    formula = "y ~ te(x, w, bs='cr', k=3, by=g)"
    source = _source(frame)
    setup = build_continuous_discrete_setup(
        parse_formula(formula),
        source,
        factor_names=("g",),
        m=30,
        batch_rows=11,
    )
    assert isinstance(setup, ContinuousDiscreteSetup)
    descriptor = build_continuous_discrete_design(
        setup, source, control=DiscreteDesignControl(batch_rows=len(frame))
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    r_setup = r_bridge.bam_discrete_setup(formula, frame, factor_names=("g",), m=30)

    explicit = descriptor.evaluate_batch(np.arange(len(frame)))
    tensor = next(
        block for block in descriptor.blocks if isinstance(block, TensorLookupBlock)
    )
    smooth_term = next(
        term for term in setup.predict_spec.coef_map.terms if term.term_type == "smooth"
    )
    assert smooth_term.Z_centering is not None
    python_z = smooth_term.Z_centering[:9, :8]
    r_z = _r_constraint(r_setup.layout.constraint_vectors[1])

    assert setup.penalties is not None
    python_penalties = setup.penalties.blocks[0].local_penalties
    embedded_r_penalties = []
    for number, penalty in enumerate(r_setup.penalties):
        embedded = np.zeros((16, 16))
        start = 0 if number < 2 else 8
        embedded[start : start + 8, start : start + 8] = penalty
        embedded_r_penalties.append(embedded)

    weights = np.linspace(0.3, 1.2, len(frame))
    response = frame.y.to_numpy()
    beta = np.linspace(-0.4, 0.7, descriptor.n_coef)
    covariance = np.eye(descriptor.n_coef) + 0.02
    r_actions = r_bridge.discrete_operators(
        r_setup.layout, weights, response, beta, covariance
    )
    device = to_device_discrete_design(descriptor)
    selectors = descriptor.batch_indices(np.arange(len(frame)))

    collector = _AssertCollector()
    collector.check(
        "term_metadata",
        lambda: (
            np.testing.assert_array_equal(r_setup.layout.constraint_codes, (0, 1, 1)),
            np.testing.assert_array_equal(r_setup.layout.term_dimensions, (1, 3, 3)),
            np.testing.assert_array_equal(r_setup.layout.term_starts, (0, 1, 4)),
            np.testing.assert_array_equal(tensor.keep, np.arange(16)),
        ),
    )
    collector.check(
        "constraint_transform",
        lambda: np.testing.assert_allclose(
            python_z, r_z, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "expanded_design",
        lambda: np.testing.assert_allclose(
            explicit, r_setup.expanded_design, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    for number, (python, expected) in enumerate(
        zip(python_penalties, embedded_r_penalties, strict=True)
    ):
        collector.check(
            f"penalty_{number}",
            lambda python=python, expected=expected: np.testing.assert_allclose(
                python.dense(), expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.check(
        "r_xbd",
        lambda: np.testing.assert_allclose(
            r_actions["xbd"], explicit @ beta, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "python_xbd",
        lambda: np.testing.assert_allclose(
            np.asarray(discrete_xbd(device, selectors, beta)),
            r_actions["xbd"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "xwyd",
        lambda: np.testing.assert_allclose(
            np.asarray(discrete_xwyd(device, selectors, weights, response)),
            r_actions["xwyd"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "xwxd",
        lambda: np.testing.assert_allclose(
            np.asarray(discrete_xwxd(device, selectors, weights)),
            r_actions["xwxd"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "diag_xvxd",
        lambda: np.testing.assert_allclose(
            np.asarray(discrete_diag_xvxd(device, selectors, covariance)),
            r_actions["diag_xvxd"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    try:
        collector.raise_if_any("factor tensor pinned bam setup")
    finally:
        setup.cleanup()


def test_pinned_bam_sparse_tensor_constraint_is_rejected_before_xbd(r_bridge) -> None:
    """Do not pass mgcv's shortened subset QR to a full-width C operator."""
    n_rows = 72
    frame = pd.DataFrame(
        {
            "y": np.linspace(0.2, 1.8, n_rows),
            "x": np.resize(np.array((0.0, 0.2, 0.5, 0.8, 1.0)), n_rows),
            "w": np.resize(np.array((-1.0, 0.0, 1.0)), n_rows),
            "g": pd.Series(
                pd.Categorical(
                    np.resize(np.array(("low", "mid", "high")), n_rows),
                    categories=("low", "mid", "high"),
                    ordered=True,
                )
            ),
        }
    )
    with pytest.raises(
        RBridgeError,
        match="rank-one constraint 2 has 7 Householder entries for raw width 9",
    ):
        r_bridge.bam_discrete_setup(
            "y ~ te(x, w, bs='cr', k=3, by=g)",
            frame,
            factor_names=("g",),
            m=6,
        )


def test_pinned_bam_setup_default_uses_discrete_grid(r_bridge) -> None:
    """An omitted helper grid requests ``bam(discrete=TRUE)``, never NULL."""
    x = np.linspace(0.0, 1.0, 50)
    frame = pd.DataFrame({"y": np.sin(3.0 * x), "x": x})
    setup = r_bridge.bam_discrete_setup("y ~ s(x, bs='cr', k=6)", frame)
    assert setup.layout.marginal_tables
    assert setup.layout.row_indices.shape[0] == len(frame)
    assert setup.expanded_design.shape[0] == len(frame)
    assert setup.expanded_design.shape[1] > 1
    assert np.all(np.isfinite(setup.expanded_design))


@pytest.mark.platform_sensitive
@pytest.mark.parametrize(
    ("python_formula", "r_formula"),
    [
        (
            "y ~ s(x, bs='tp', k=8, max_knots=40, seed=3)",
            "y ~ s(x,bs='tp',k=8,xt=list(max.knots=40,seed=3))",
        ),
        (
            "y ~ s(x, z, bs='tp', k=12, xt={'max_knots': 40, 'seed': 3})",
            "y ~ s(x,z,bs='tp',k=12,xt=list(max.knots=40,seed=3))",
        ),
        (
            "y ~ s(x, bs='gp', k=8, xt={'max_knots': 40, 'seed': 5})",
            "y ~ s(x,bs='gp',k=8,xt=list(max.knots=40,seed=5))",
        ),
        (
            "y ~ s(x, z, bs='gp', k=10, xt={'max_knots': 40, 'seed': 5})",
            "y ~ s(x,z,bs='gp',k=10,xt=list(max.knots=40,seed=5))",
        ),
    ],
)
def test_sampled_radial_knots_match_pinned_bam_compact_frame(
    r_bridge, python_formula: str, r_formula: str
) -> None:
    """bam(discrete=TRUE) samples tp/gp knots from its compact frame."""
    rng = np.random.default_rng(20261118)
    n_rows = 2500
    frame = pd.DataFrame(
        {"x": rng.uniform(-3, 12, n_rows), "z": rng.normal(0, 8, n_rows)}
    )
    frame["y"] = np.sin(frame.x / 2) + rng.normal(scale=0.3, size=n_rows)
    source = _source(frame)
    setup = build_continuous_discrete_setup(
        parse_formula(python_formula), source, m=120, batch_rows=97
    )
    assert isinstance(setup, ContinuousDiscreteSetup)
    # The cap applies to the compact frame's discretized covariate rows:
    # fewer distinct rows than data rows, but more than the cap.
    variables = list(parse_formula(python_formula).smooth_terms[0].variables)
    compact_rows = np.unique(setup.compact.frame[variables].to_numpy(), axis=0)
    assert 40 < len(compact_rows) < n_rows
    descriptor = build_continuous_discrete_design(
        setup, source, control=DiscreteDesignControl(batch_rows=n_rows)
    )
    assert not isinstance(descriptor, DiscreteDesignUnavailable)
    expected = r_bridge.bam_discrete_setup(r_formula, frame, m=120).expanded_design
    np.testing.assert_allclose(
        descriptor.evaluate_batch(np.arange(n_rows)),
        expected,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
