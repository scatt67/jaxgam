"""Tests for the PR11 discretization approximation-study harness.

Small and fast: one real toy Gaussian case and one toy negative binomial
case (with a typed fake pinned-R bridge) exercise the fitting and metric
paths; provenance, child isolation and the command line use fakes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import benchmark_discrete_approximation as harness
from scripts.benchmark_discrete_approximation import (
    ApproximationCase,
    ProvenanceError,
    level_m,
)
from tests.helpers import _AssertCollector, check_that
from tests.tolerances import STRICT

_FAKE_IDENTITY = {"commit": "0" * 40, "dirty_sha256": "1" * 64}


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=harness-test",
            "-c",
            "user.email=harness-test@example.com",
            *arguments,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def committed_checkout(tmp_path: Path) -> Path:
    """A one-file committed git checkout (the Docker image has no ``.git``)."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "--quiet")
    (root / "tracked.py").write_text("value = 1\n")
    _git(root, "add", "tracked.py")
    _git(root, "commit", "--quiet", "-m", "initial")
    return root


def _toy_case(**overrides: object) -> ApproximationCase:
    fields: dict[str, object] = {
        "family": "gaussian",
        "n_rows": 80,
        "n_test": 20,
        "levels": ("4", "12", "exact"),
        "modes": ("fixed_sp", "fixed_sp_matched", "working_model_reml", "exact_reml"),
        "model": "toy",
    }
    fields.update(overrides)
    return ApproximationCase(**fields)  # type: ignore[arg-type]


class _FakeBridge:
    """Typed stand-in for the pinned-R ``RBridge`` methods the harness calls."""

    def __init__(self) -> None:
        self.bam_calls: list[dict[str, object]] = []

    def fit_gam(self, formula: str, data: pd.DataFrame, family: str) -> dict:
        self.gam_call = (formula, family)
        return {
            "fitted_values": np.ones(len(data)),
            "deviance": 1.0,
            "smoothing_params": np.ones(1),
            "edf": np.ones(1),
            "theta": 2.0,
        }

    def fit_bam_discrete(self, formula: str, data: pd.DataFrame, **kwargs: object):
        self.bam_calls.append({"formula": formula, **kwargs})
        newdata = kwargs["newdata"]
        assert isinstance(newdata, pd.DataFrame)
        return SimpleNamespace(
            fitted_values=np.ones(len(data)),
            prediction=np.zeros(len(newdata)),
            mgcv_converged=True,
            edf_total=3.0,
        )


@pytest.fixture(scope="module")
def toy_record() -> dict:
    """One real in-process toy Gaussian case (no pinned R)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(harness, "source_identity", lambda _root: dict(_FAKE_IDENTITY))
        return harness.run_case(_toy_case(), Path.cwd())


@pytest.fixture(scope="module")
def nb_record_and_bridge() -> tuple[dict, _FakeBridge]:
    """A real toy negative binomial case with a fake pinned-R bridge."""
    bridge = _FakeBridge()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(harness, "source_identity", lambda _root: dict(_FAKE_IDENTITY))
        patch.setattr(harness, "_r_bridge", lambda: (bridge, "fake R"))
        record = harness.run_case(
            _toy_case(
                family="nb",
                n_rows=150,
                levels=("12", "exact"),
                modes=("fixed_sp", "working_model_reml"),
                r_reference=True,
            ),
            Path.cwd(),
        )
    return record, bridge


def _level(record: dict, name: str) -> dict:
    return next(level for level in record["levels"] if level["level"] == name)


# --------------------------------------------------------------------------
# Case validation and levels.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"family": "tweedie"}, "family"),
        ({"model": "soap"}, "model"),
        ({"n_rows": 10}, "n_rows"),
        ({"n_test": 5}, "n_rows"),
        ({"batch_rows": 0}, "batch_rows"),
        ({"levels": ()}, "levels"),
        ({"levels": ("10", "10")}, "levels"),
        ({"levels": ("ten",)}, "not an integer"),
        ({"levels": ("1",)}, "at least 2"),
        ({"levels": ("010",)}, "at least 2"),
        ({"modes": ()}, "modes"),
        ({"modes": ("fixed_sp", "fixed_sp")}, "modes"),
        ({"modes": ("gcv",)}, "modes"),
    ],
)
def test_case_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _toy_case(**overrides)


def test_levels_map_to_grid_sizes() -> None:
    assert level_m("default", 500) is None
    assert level_m("exact", 500) == 500
    assert level_m("30", 500) == 30


def test_case_factor_names_follow_the_model() -> None:
    assert _toy_case(model="study").factor_names == ("fac", "g")
    assert _toy_case(model="tensor").factor_names == ("fac",)
    assert _toy_case().factor_names == ()


def test_ordered_levels_compute_exact_first() -> None:
    assert harness._ordered_levels(("10", "exact", "default")) == [
        "exact",
        "10",
        "default",
    ]


# --------------------------------------------------------------------------
# Provenance.
# --------------------------------------------------------------------------


def test_clean_checkout_identity_and_expected_commit(committed_checkout: Path) -> None:
    identity = harness.require_clean_checkout(committed_checkout)
    assert len(identity["commit"]) == 40
    prefix = identity["commit"][:8]
    assert harness.require_clean_checkout(committed_checkout, prefix) == identity
    with pytest.raises(ProvenanceError, match="not the expected"):
        harness.require_clean_checkout(committed_checkout, "f" * 40)


def test_dirty_checkout_is_refused(committed_checkout: Path) -> None:
    (committed_checkout / "untracked.py").write_text("value = 2\n")
    with pytest.raises(ProvenanceError, match="uncommitted or untracked"):
        harness.require_clean_checkout(committed_checkout)


def test_non_checkout_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="not a readable git checkout"):
        harness.require_clean_checkout(tmp_path / "missing")


def test_harness_identity_hashes_this_script_and_its_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = harness.harness_identity()
    assert identity["harness_path"].endswith("benchmark_discrete_approximation.py")
    assert len(identity["harness_sha256"]) == 64
    monkeypatch.setattr(
        harness.benchmark_scaling, "__file__", str(tmp_path / "benchmark_scaling.py")
    )
    with pytest.raises(ProvenanceError, match="PYTHONPATH"):
        harness.harness_identity()


def test_output_must_be_outside_the_checkout(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    outside = tmp_path / "results.json"
    assert harness.require_output_outside(outside, root) == outside.resolve()
    with pytest.raises(ProvenanceError, match="inside the checkout"):
        harness.require_output_outside(root / "results.json", root)


def test_child_provenance_must_match_the_parent() -> None:
    parent = {**_FAKE_IDENTITY, "harness_sha256": "2" * 64}
    harness.require_matching_child(parent, dict(parent))
    with pytest.raises(ProvenanceError, match="dirty_sha256"):
        harness.require_matching_child(parent, {**parent, "dirty_sha256": "3" * 64})


# --------------------------------------------------------------------------
# Data, families and small helpers.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", harness.FAMILIES)
@pytest.mark.parametrize("model", tuple(harness.MODELS))
def test_generated_rows_are_reproducible_and_in_domain(family: str, model: str) -> None:
    case = _toy_case(family=family, model=model, n_rows=60)
    train, eta = harness.make_data(case, test=False)
    again, _ = harness.make_data(case, test=False)
    test, _ = harness.make_data(case, test=True)
    collector = _AssertCollector()
    collector.check(
        "reproducible",
        lambda: check_that(
            harness._frame_sha256(train) == harness._frame_sha256(again),
            "same seed changed the rows",
        ),
    )
    collector.check(
        "held-out rows differ",
        lambda: check_that(
            harness._frame_sha256(train) != harness._frame_sha256(test),
            "held-out rows repeat the training rows",
        ),
    )
    collector.check(
        "shapes",
        lambda: check_that(
            len(train) == 60 and len(test) == 20 and eta.shape == (60,),
            "wrong row counts",
        ),
    )
    y = train.y.to_numpy()
    domain = {
        "gaussian": np.all(np.isfinite(y)),
        "poisson": np.all((y >= 0) & (y == np.floor(y))),
        "binomial": np.all((y == 0) | (y == 1)),
        "gamma": np.all(y > 0),
        "nb": np.all((y >= 0) & (y == np.floor(y))),
        "nb_fixed": np.all((y >= 0) & (y == np.floor(y))),
    }[family]
    collector.check(
        "response domain", lambda: check_that(bool(domain), "y out of domain")
    )
    collector.raise_if_any(f"generated rows {model}/{family}")


def test_families_and_theta_policy() -> None:
    from jaxgam.families.negative_binomial import NegativeBinomial

    assert type(harness._family("gamma")).__name__ == "Gamma"
    estimated = harness._family("nb")
    fixed = harness._family("nb", theta=2.5)
    known = harness._family("nb_fixed", theta=9.0)
    assert isinstance(estimated, NegativeBinomial)
    assert estimated.n_theta == 1
    assert fixed.n_theta == 0
    np.testing.assert_allclose(
        [fixed.get_theta(transformed=True)[0], known.get_theta(transformed=True)[0]],
        [2.5, harness._NB_THETA],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_summary_relative_and_ratio_helpers() -> None:
    assert harness._summary(np.array([3.0, -4.0])) == {
        "max_abs": 4.0,
        "rms": float(np.sqrt(12.5)),
    }
    assert harness._relative(3.0, 2.0) == 0.5
    assert harness._ratio_summary(np.ones(2), np.zeros(2)) == {
        "min": None,
        "median": None,
        "max": None,
    }
    assert harness._ratio_summary(np.array([1.0, 3.0]), np.array([1.0, 0.0])) == {
        "min": 1.0,
        "median": 1.0,
        "max": 1.0,
    }


def _info(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "label": "s(x2)",
        "variables": ("x2",),
        "by_variable": None,
        "is_random": False,
        "first_coef": 0,
        "last_coef": 2,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_grid_frames_cover_random_tensor_and_factor_by_terms() -> None:
    train, _ = harness.make_data(_toy_case(model="study", n_rows=60), test=False)
    random = harness._grid_frames(train, _info(variables=("g",), is_random=True))
    tensor = harness._grid_frames(train, _info(variables=("x0", "x1")))
    by = harness._grid_frames(train, _info(by_variable="fac"))
    assert [len(frame) for frame in random] == [len(harness._G_LEVELS)]
    assert list(random[0].g) == list(harness._G_LEVELS)
    assert [len(frame) for frame in tensor] == [900]
    assert [frame.fac.iloc[0] for frame in by] == list(harness._FAC_LEVELS)
    assert all(list(frame.columns) == ["x0", "x1", "x2", "fac", "g"] for frame in by)


class _LevelFit:
    """A two-column term that only rows of factor level ``a`` select."""

    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        selected = (frame.fac == "a").to_numpy(dtype=float)
        return np.column_stack([selected * frame.x2, selected * frame.x2**2])


def test_partial_effects_skip_unselected_levels_and_center_curves() -> None:
    train, _ = harness.make_data(_toy_case(model="study", n_rows=60), test=False)

    def view(coefficients: list[float]) -> harness.FitView:
        return harness.FitView(
            fit=_LevelFit(np.asarray(coefficients)),
            edf=np.ones(1),
            edf_total=1.0,
            covariance=np.eye(2),
            smoothing_params=np.ones(1),
            scale=1.0,
            theta=None,
        )

    records = harness.partial_effect_metrics(
        view([1.0, 0.5]), view([1.0, 0.0]), [_info(by_variable="fac")], train
    )
    assert [record["by_level"] for record in records] == ["a"]
    assert records[0]["centered"]
    assert records[0]["l2"] > 0.0
    assert records[0]["se_ratio"]["median"] == pytest.approx(1.0)


def test_term_knots_walk_by_and_tensor_smooths() -> None:
    plain = SimpleNamespace(_knots=np.arange(3.0))
    by = SimpleNamespace(base_smooth=plain)
    tensor = SimpleNamespace(_marginals=[plain, SimpleNamespace(_knots=None)])
    coef_map = SimpleNamespace(
        terms=[
            SimpleNamespace(label="s(x):a", smooth=by),
            SimpleNamespace(label="te(x,z)", smooth=tensor),
            SimpleNamespace(label="s(g)", smooth=SimpleNamespace()),
            SimpleNamespace(label="fac", smooth=None),
        ]
    )
    knots = harness.term_knots(coef_map)
    assert list(knots) == ["s(x):a", "te(x,z)"]
    np.testing.assert_array_equal(knots["te(x,z)"][0], np.arange(3.0))


def test_knot_displacement_handles_missing_and_reshaped_terms() -> None:
    reference = {"s(x)": [np.array([0.0, 0.5, 1.0])], "te(x,z)": [np.zeros(3)]}
    knots = {"s(x)": [np.array([0.0, 0.6, 1.0])], "te(x,z)": [np.zeros(4)]}
    displacement = harness.knot_displacement(knots, reference)
    assert displacement["s(x)"] == pytest.approx(0.1)
    assert displacement["te(x,z)"] is None
    assert harness.knot_displacement({}, reference)["s(x)"] is None


def test_r_bridge_reports_why_it_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.r_bridge as r_bridge

    class FakeRBridge:
        ready = False
        versions = (False, "mgcv 1.9-4")

        def __init__(self, mode: str) -> None:
            self.mode = mode

        @classmethod
        def available(cls) -> bool:
            return cls.ready

        @classmethod
        def check_versions(cls) -> tuple[bool, str]:
            return cls.versions

    monkeypatch.setattr(r_bridge, "RBridge", FakeRBridge)
    assert harness._r_bridge() == (None, "rpy2 or R not available")
    FakeRBridge.ready = True
    assert harness._r_bridge() == (None, "mgcv 1.9-4")
    FakeRBridge.versions = (True, "")
    bridge, status = harness._r_bridge()
    assert isinstance(bridge, FakeRBridge)
    assert bridge.mode == "rpy2"
    assert "mgcv 1.9-3" in status
    monkeypatch.setitem(sys.modules, "tests.r_bridge", None)
    bridge, status = harness._r_bridge()
    assert bridge is None
    assert status


def test_r_gam_anchor_has_no_fixed_theta_negative_binomial() -> None:
    record = harness.r_gam_anchor(
        _FakeBridge(), _toy_case(family="nb_fixed"), "y ~ x", pd.DataFrame(), None
    )
    assert record["status"] == "unsupported"


# --------------------------------------------------------------------------
# Real toy cases.
# --------------------------------------------------------------------------


def test_toy_gaussian_case_records_levels_modes_and_checks(toy_record: dict) -> None:
    record = toy_record
    coarse, refined, exact = (_level(record, name) for name in ("4", "12", "exact"))
    fixed = exact["modes"]["fixed_sp"]
    collector = _AssertCollector()
    collector.check(
        "level order preserved",
        lambda: check_that(
            [level["level"] for level in record["levels"]] == ["4", "12", "exact"],
            "levels reordered in the record",
        ),
    )
    collector.check(
        "grid coarser than k is an error, not a crash",
        lambda: check_that(
            all(mode["status"] == "error" for mode in coarse["modes"].values()),
            str(coarse["modes"]),
        ),
    )
    collector.check(
        "exact REML on the discrete design is unsupported",
        lambda: check_that(
            refined["modes"]["exact_reml"]["status"] == "unsupported"
            and exact["modes"]["exact_reml"]["status"] == "unsupported",
            "exact_reml unexpectedly ran",
        ),
    )
    collector.check(
        "exact level rounds nothing",
        lambda: check_that(
            not exact["covariates"]["x0"]["rounded"]
            and exact["covariates"]["x0"]["max_abs_error"] == 0.0,
            str(exact["covariates"]),
        ),
    )
    collector.check(
        "refined level rounds",
        lambda: check_that(
            refined["covariates"]["x0"]["rounded"]
            and refined["covariates"]["x0"]["grid_points"] == 12,
            str(refined["covariates"]),
        ),
    )
    collector.check(
        "same-strategy exact reference",
        lambda: check_that(
            "vs_exact_level" in refined["modes"]["working_model_reml"]
            and "vs_exact_level" in refined["modes"]["fixed_sp"]
            and "vs_exact_level" not in fixed,
            "vs_exact_level placement",
        ),
    )
    collector.check(
        "deviance at the discretized design reproduces the fit",
        lambda: np.testing.assert_allclose(
            fixed["self_consistency"]["harness_deviance"],
            fixed["self_consistency"]["fit_deviance"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "Gaussian reconstructed EDF reproduces the fit",
        lambda: np.testing.assert_allclose(
            fixed["self_consistency"]["harness_edf_total"],
            fixed["self_consistency"]["fit_edf_total"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    matched = refined["modes"]["fixed_sp_matched"]
    collector.check(
        "norm-matched sp divides by the penalty-norm ratio",
        lambda: np.testing.assert_allclose(
            matched["smoothing_params"],
            np.asarray(record["reference"]["reml"]["smoothing_params"])
            / np.asarray(refined["penalty_frobenius_ratio"]),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.check(
        "basis change: knots move only when the covariate is rounded",
        lambda: check_that(
            exact["knot_displacement_over_range"] == {"s(x0)": 0.0}
            and refined["knot_displacement_over_range"]["s(x0)"] > 0.0
            and exact["penalty_frobenius_ratio"] == [1.0],
            f"{exact['knot_displacement_over_range']} "
            f"{refined['knot_displacement_over_range']}",
        ),
    )
    held = fixed["vs_dense"]["held_out"]
    collector.check(
        "metric record",
        lambda: check_that(
            {"link_difference", "deviance", "intervals"} <= set(held)
            and len(fixed["vs_dense"]["partial_effects"]) == 1
            and record["runtime"]["commit"] == _FAKE_IDENTITY["commit"],
            "missing metric fields",
        ),
    )
    collector.check(
        "no R anchors when not requested",
        lambda: check_that(
            record["r_anchors"]["status"] == "not_requested",
            str(record["r_anchors"]),
        ),
    )
    collector.raise_if_any("toy Gaussian approximation case")


def test_exact_level_fixed_sp_matches_the_dense_fit() -> None:
    """The implementation check behind the exact level, at STRICT."""
    case = _toy_case()
    formula = harness.MODELS[case.model]
    train, _ = harness.make_data(case, test=False)
    test, _ = harness.make_data(case, test=True)
    dense = harness._fit_dense(case, formula, train)
    sp = np.asarray(dense.smoothing_params)
    discrete = harness._fit_discrete(
        case, formula, train, level_m("exact", case.n_rows), "fixed_sp", sp=sp
    )
    np.testing.assert_allclose(
        np.asarray(discrete.predict(test, pred_type="link")),
        np.asarray(dense.predict(test, pred_type="link")),
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_toy_negative_binomial_case_with_fake_r_anchors(
    nb_record_and_bridge: tuple[dict, _FakeBridge],
) -> None:
    record, bridge = nb_record_and_bridge
    exact = _level(record, "exact")
    fixed = exact["modes"]["fixed_sp"]
    working = _level(record, "12")["modes"]["working_model_reml"]
    theta = record["reference"]["reml"]["theta"]
    collector = _AssertCollector()
    collector.check(
        "estimated theta reference",
        lambda: check_that(theta is not None and theta > 0, f"theta {theta}"),
    )
    collector.check(
        "fixed-sp discrete fit keeps the dense theta",
        lambda: np.testing.assert_allclose(
            fixed["theta"], theta, rtol=STRICT.rtol, atol=STRICT.atol
        ),
    )
    collector.check(
        "working model estimates theta",
        lambda: check_that(
            working["status"] == "ok"
            and working["vs_dense"]["theta_relative_difference"] is not None,
            str(working.get("error")),
        ),
    )
    collector.check(
        "gam anchor",
        lambda: check_that(
            record["r_anchors"]["status"] == "ok"
            and record["r_anchors"]["gam_reml"]["status"] == "ok"
            and bridge.gam_call[1] == "nb",
            str(record["r_anchors"]),
        ),
    )
    call = bridge.bam_calls[0]
    collector.check(
        "bam anchor runs once, at the exact grid, sp and fixed theta",
        lambda: check_that(
            len(bridge.bam_calls) == 1
            and call["family"] == "nb"
            and call["m"] == 150
            and call["theta_fixed"] is True
            and call["theta"] == theta
            and fixed["r_bam_anchor"]["r_edf_total"] == 3.0,
            str(call),
        ),
    )
    collector.check(
        "negative binomial reconstructed deviance",
        lambda: np.testing.assert_allclose(
            fixed["self_consistency"]["harness_deviance"],
            fixed["self_consistency"]["fit_deviance"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        ),
    )
    collector.raise_if_any("toy negative binomial approximation case")


def test_bam_anchor_fixes_the_known_theta() -> None:
    case = _toy_case(family="nb_fixed", levels=("exact",), modes=("fixed_sp",))
    formula = harness.MODELS[case.model]
    train, _ = harness.make_data(case, test=False)
    test, _ = harness.make_data(case, test=True)
    dense = harness.dense_view(harness._fit_dense(case, formula, train))
    bridge = _FakeBridge()
    record = harness.r_bam_anchor(
        bridge,
        case,
        formula,
        train,
        test,
        case.n_rows,
        dense.smoothing_params,
        None,
        dense=dense,
        discrete=dense,
    )
    assert bridge.bam_calls[0]["theta"] == harness._NB_THETA
    assert record["discrete"] == record["dense"]


# --------------------------------------------------------------------------
# Summaries, child isolation and the command line.
# --------------------------------------------------------------------------


def test_summary_tables_cover_both_references_and_implementation_checks(
    toy_record: dict,
    nb_record_and_bridge: tuple[dict, _FakeBridge],
) -> None:
    payload = {"cases": [toy_record, nb_record_and_bridge[0]]}
    against_dense = harness.summary_rows(payload)
    against_exact = harness.summary_rows(payload, "vs_exact_level")
    implementation = harness.implementation_rows(payload)
    text = harness.format_summary(payload)
    assert {row["status"] for row in against_dense} == {"ok", "error", "unsupported"}
    assert all(row["status"] == "ok" for row in against_exact)
    assert {row["level"] for row in against_exact} == {"12"}
    assert [row["family"] for row in implementation] == ["gaussian", "nb"]
    assert implementation[0]["discrete_vs_bam"] is None
    anchor = _level(nb_record_and_bridge[0], "exact")["modes"]["fixed_sp"][
        "r_bam_anchor"
    ]
    assert (
        implementation[1]["discrete_vs_bam"]
        == anchor["discrete"]["held_out_link_max_abs_difference"]
    )
    assert "### Exact-level implementation checks (fixed sp)" in text
    assert "| n/a |" in text
    assert "| unsupported |" in text


def test_run_isolated_uses_a_task_owned_cache_and_checks_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = {**_FAKE_IDENTITY, "harness_sha256": "2" * 64}
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"runtime": parent}), stderr=""
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    record = harness.run_isolated(_toy_case(), Path.cwd(), parent)
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert record == {"runtime": parent}
    assert environment["PYTHONPATH"] == str(Path.cwd())
    assert "jaxgam-benchmark-jax-" in environment["JAX_COMPILATION_CACHE_DIR"]
    assert "--child" in captured["command"]  # type: ignore[operator]

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"runtime": {**parent, "commit": "9" * 40}}),
            stderr="",
        ),
    )
    with pytest.raises(ProvenanceError, match="commit"):
        harness.run_isolated(_toy_case(), Path.cwd(), parent)

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        harness.run_isolated(_toy_case(), Path.cwd(), parent)


def test_main_summarize_child_and_parent_modes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    toy_record: dict,
) -> None:
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps({"cases": [toy_record]}))
    harness.main(["--summarize", str(saved)])
    assert "Against the undiscretized dense fit" in capsys.readouterr().out

    monkeypatch.setattr(
        harness, "run_case", lambda case, _root: {"family": case.family}
    )
    harness.main(["--child", json.dumps(asdict(_toy_case()))])
    assert json.loads(capsys.readouterr().out) == {"family": "gaussian"}

    with pytest.raises(SystemExit):
        harness.main([])
    capsys.readouterr()

    with pytest.raises(ProvenanceError, match="inside the checkout"):
        harness.main(["--output", str(Path.cwd() / "results.json")])

    cases: list[ApproximationCase] = []

    def fake_isolated(case: ApproximationCase, _root: Path, parent: dict) -> dict:
        cases.append(case)
        assert parent["commit"] == _FAKE_IDENTITY["commit"]
        return toy_record

    monkeypatch.setattr(
        harness, "require_clean_checkout", lambda _root, _expect: dict(_FAKE_IDENTITY)
    )
    monkeypatch.setattr(harness, "run_isolated", fake_isolated)
    output = tmp_path / "results.json"
    harness.main(
        [
            "--output",
            str(output),
            "--models",
            "toy",
            "--families",
            "gaussian,poisson",
            "--n",
            "80,30000",
            "--levels",
            "12,exact",
            "--n-test",
            "20",
        ]
    )
    payload = json.loads(output.read_text())
    assert len(payload["cases"]) == 4
    assert payload["provenance"]["commit"] == _FAKE_IDENTITY["commit"]
    assert [case.r_reference for case in cases] == [True, False, True, False]
    assert "results:" in capsys.readouterr().err
