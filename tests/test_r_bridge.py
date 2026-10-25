"""Tests for the R bridge (tests.r_bridge).

These tests require R with mgcv installed.
They are automatically skipped if R is not available.
"""

import numpy as np
import pandas as pd
import pytest

from tests.efs_oracle import make_efs_fixture
from tests.helpers import _AssertCollector, check_that, make_smooth_spec
from tests.r_bridge import RBridge, RBridgeError, _r_array, gp_config_to_mgcv_m
from tests.tolerances import MODERATE, STRICT


def _r_available() -> bool:
    if not RBridge.available():
        return False
    ok, _ = RBridge.check_versions()
    return ok


@pytest.fixture
def gaussian_data() -> pd.DataFrame:
    """Small Gaussian dataset for R bridge testing."""
    rng = np.random.default_rng(123)
    n = 100
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x": x, "y": y})


class TestRBridgeAvailability:
    @pytest.mark.skipif(not RBridge.available(), reason="R not available")
    def test_check_versions_accepts_pinned_environment(self) -> None:
        ok, reason = RBridge.check_versions()
        assert ok, reason

    @pytest.mark.skipif(not RBridge.available(), reason="rpy2 with mgcv unavailable")
    def test_default_reference_never_launches_rscript(self, monkeypatch) -> None:
        """Default selection and version checks stay in the embedded R process."""
        import subprocess

        def unexpected_process(*_args, **_kwargs):
            raise AssertionError("RBridge attempted to launch an external process")

        monkeypatch.setattr(subprocess, "run", unexpected_process)
        monkeypatch.setattr(subprocess, "check_output", unexpected_process)
        assert RBridge.available()
        bridge = RBridge()
        bridge._require_rpy2()
        assert bridge.mode == "rpy2"

    @pytest.mark.skipif(not RBridge.available(), reason="rpy2 with mgcv unavailable")
    def test_object_transport_preserves_values_and_dimensions(self) -> None:
        """Subnormals, signed zero, and adjacent floats survive direct transfer."""
        bridge = RBridge()
        source = np.array(
            [[0.0, -0.0], [np.nextafter(1.0, 2.0), np.nextafter(0.0, 1.0)]]
        ).T
        result = _r_array(bridge._to_r_matrix(source), dtype=np.float64)
        np.testing.assert_array_equal(result.view(np.uint64), source.view(np.uint64))
        integers = np.array([0, 1, np.iinfo(np.int32).max], dtype=np.int64)
        np.testing.assert_array_equal(bridge._to_r_vector(integers), integers)
        with pytest.raises(ValueError, match="non-missing range"):
            bridge._to_r_vector(np.array([np.iinfo(np.int32).min]))
        with pytest.raises(ValueError, match="two-dimensional"):
            bridge._to_r_matrix(integers)


def _owns_memory(array: np.ndarray) -> bool:
    """Whether ``array`` (or the array it views) owns NumPy-allocated memory."""
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    return root.flags.owndata


def _churn_r_memory(bridge: RBridge) -> None:
    """Force R garbage collection between fresh R allocations (no RNG use)."""
    for size in (10_000, 100_000, 1_000_000, 100_000):
        bridge._base.gc()
        bridge._base.numeric(size)
        bridge._base.as_double(bridge._base.seq_len(size))
    bridge._base.gc()


@pytest.mark.skipif(not _r_available(), reason="pinned R with mgcv not available")
class TestRArrayOwnership:
    """R-to-NumPy conversions own their memory instead of viewing R's."""

    def test_r_array_copies_with_numpy_dtype_shape_and_layout(self) -> None:
        bridge = RBridge()
        values = np.arange(24.0) - 7.5
        cases = {
            "double": bridge._to_r_vector(values),
            "integer": bridge._ro.IntVector(np.arange(5, dtype=np.int32)),
            "logical": bridge._ro.BoolVector([True, False, True]),
            "matrix": bridge._to_r_matrix(values.reshape(4, 6)),
            "array": bridge._base.array(
                bridge._to_r_vector(values), dim=bridge._ro.IntVector([2, 3, 4])
            ),
        }
        collector = _AssertCollector()
        for name, r_value in cases.items():
            view = np.asarray(r_value)  # the zero-copy pattern being replaced
            owned = _r_array(r_value)

            def check(view=view, owned=owned) -> None:
                assert not view.flags.owndata
                assert owned.flags.owndata
                assert not np.shares_memory(owned, view)
                assert owned.dtype == view.dtype
                assert owned.shape == view.shape
                assert owned.flags.f_contiguous == view.flags.f_contiguous
                np.testing.assert_array_equal(owned, view)

            collector.check(name, check)
        converted = _r_array(cases["integer"], dtype=np.float64)
        collector.check(
            "requested dtype",
            lambda: np.testing.assert_array_equal(converted, np.arange(5.0)),
        )
        collector.raise_if_any("_r_array ownership")

    def test_converted_values_survive_forced_r_garbage_collection(self) -> None:
        """A copy keeps R's values after the R object is released and collected."""
        bridge = RBridge()
        source = np.linspace(-2.0, 3.0, 4096)
        expected = source * 2.0 + 1.0
        r_result = bridge._ro.r["+"](
            bridge._ro.r["*"](bridge._to_r_vector(source), 2), 1
        )
        owned = _r_array(r_result)
        del r_result
        _churn_r_memory(bridge)
        np.testing.assert_array_equal(owned, expected)

    def test_bridge_oracles_return_owned_arrays_that_survive_gc(self) -> None:
        bridge = RBridge()
        rng = np.random.default_rng(20261025)
        x = np.sort(rng.uniform(0.0, 1.0, 80))
        z = rng.uniform(0.0, 1.0, 80)
        data = pd.DataFrame({"x": x, "z": z, "y": np.sin(6 * x) + 0.3 * z})
        results = {
            "smoothCon": bridge.smooth_construct(
                "s(x,bs='tp',k=8)", data, absorb_cons=False
            ),
            "smoothCon list": bridge.smooth_construct_list(
                "ti(x,z,bs='tp',k=c(4,4))", data, absorb_cons=True
            )[0],
            "fit": bridge.fit_gam("y ~ s(x, k=8) + s(z, k=6)", data),
            "prediction": bridge.predict_gam(
                "y ~ s(x, k=8)", data, data.iloc[:7], se_fit=True
            ),
            "summary": bridge.summary_gam("y ~ s(x, k=8)", data),
            "components": bridge.get_smooth_components("y ~ s(x, k=8)", data),
        }

        def arrays(value, prefix=""):
            if isinstance(value, np.ndarray):
                yield prefix, value
            elif isinstance(value, dict):
                for key, item in value.items():
                    yield from arrays(item, f"{prefix}/{key}")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    yield from arrays(item, f"{prefix}[{index}]")

        found = list(arrays(results))
        assert len(found) > 20
        snapshot = {name: array.copy() for name, array in found}
        _churn_r_memory(bridge)
        for _ in range(3):
            bridge.smooth_construct("ti(x,z,bs='tp',k=c(6,6))", data, absorb_cons=True)
        _churn_r_memory(bridge)
        collector = _AssertCollector()
        for name, array in found:
            collector.check(
                f"{name} owned",
                lambda array=array, name=name: check_that(
                    _owns_memory(array), f"{name} views memory it does not own"
                ),
            )
            collector.check(
                f"{name} unchanged",
                lambda array=array, name=name: np.testing.assert_array_equal(
                    array, snapshot[name]
                ),
            )
        collector.raise_if_any("bridge results after R garbage collection")


@pytest.mark.skipif(not _r_available(), reason="pinned R with mgcv not available")
class TestRBridgeEFS:
    """Pinned EFS oracle calls use real mgcv fits and private instrumentation."""

    def test_fit_efs_has_stable_full_fit_and_control_metadata(self) -> None:
        data, fixture = make_efs_fixture(seed=91, n=48)
        bridge = RBridge(mode="rpy2")
        first = bridge.fit_efs(
            fixture.formula,
            data,
            family=fixture.family,
            weights="w",
            offset="offset",
            controls={"efs_lspmax": 12.0, "efs_tol": 0.05},
        )
        second = bridge.fit_efs(
            fixture.formula,
            data,
            family=fixture.family,
            weights="w",
            offset="offset",
            controls={"efs_lspmax": 12.0, "efs_tol": 0.05},
        )

        assert first["optimizer"] == "efs"
        assert first["controls"] == {"efs_lspmax": 12.0, "efs_tol": 0.05}
        assert first["provenance"]["data_hash"] == fixture.data_hash
        assert first["provenance"]["source_commit"] == fixture.source_commit
        assert first["outer_iterations"] > 0
        assert first["convergence"] in {"full convergence", "iteration limit reached"}
        np.testing.assert_allclose(
            first["coefficients"],
            second["coefficients"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        np.testing.assert_allclose(
            first["score_history"],
            second["score_history"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_efs_diagnostics_uses_real_penalty_and_pirls_statistics(self) -> None:
        data, fixture = make_efs_fixture(seed=92, n=48)
        bridge = RBridge(mode="rpy2")
        fit = bridge.fit_efs(
            fixture.formula, data, fixture.family, weights="w", offset="offset"
        )
        diagnostics = bridge.efs_diagnostics(
            fixture.formula, data, fixture.family, weights="w", offset="offset"
        )
        statistics = diagnostics["statistics"]

        assert {
            "ldetS1",
            "bSb",
            "trVS",
            "score_phi",
            "update_phi",
            "reported_phi",
        } <= set(statistics)
        assert len(statistics) >= len(fit["smoothing_params"])
        assert np.all(
            np.isfinite(statistics[["ldetS1", "bSb", "trVS", "score"]].to_numpy())
        )
        assert diagnostics["initial_shift"] == 2.5
        assert (
            diagnostics["source_commit"] == "fb7e8e718377513e78ba6c6bf7e60757fc6a32a9"
        )
        assert diagnostics["provenance"]["data_hash"] == fixture.data_hash
        # For Poisson the score uses phi=1 even though gam.fit3 reports a
        # Fletcher-like scale estimate for diagnostics.
        np.testing.assert_allclose(
            statistics["score_phi"], 1.0, rtol=STRICT.rtol, atol=STRICT.atol
        )
        np.testing.assert_allclose(
            statistics["update_phi"],
            statistics["score_phi"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        assert np.any(
            np.abs(statistics["reported_phi"] - statistics["score_phi"]) > MODERATE.atol
        )
        np.testing.assert_allclose(
            diagnostics["final_score"],
            fit["reml_score"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )

    def test_fit_efs_validates_structured_inputs(self) -> None:
        data, fixture = make_efs_fixture(n=24)
        bridge = RBridge(mode="rpy2")
        with pytest.raises(ValueError, match="weights column"):
            bridge.fit_efs(fixture.formula, data, fixture.family, weights="missing")
        with pytest.raises(ValueError, match="Unsupported EFS controls"):
            bridge.fit_efs(
                fixture.formula, data, fixture.family, controls={"outer_limit": 3}
            )

    def test_selected_efs_oracle_isolates_offset_null_deviance_failure(self) -> None:
        """The bounded-link source clone neither mutates nor masks public R."""
        x = np.linspace(-0.6, 0.7, 79)
        rng = np.random.default_rng(17019)
        y = rng.binomial(1, 0.5 + 0.1 * x).astype(float)
        weights = rng.uniform(0.5, 1.5, len(x))
        weights[::9] = 0.0
        data = pd.DataFrame({"x": x, "y": y, "w": weights, "off": 2.0})
        bridge = RBridge(mode="rpy2")
        arguments = {
            "weights": "w",
            "offset": "off",
            "initial_smoothing": np.array([13.427864978849476]),
            "scale": 1.0,
        }
        for _ in range(2):
            with pytest.raises(RBridgeError, match="no valid set of coefficients"):
                bridge.fit_efs(
                    'y ~ s(x, bs="cr", k=6)',
                    data,
                    "binomial_inverse",
                    **arguments,
                )
            selected = bridge.fit_efs_selected_before_offset_null_deviance(
                'y ~ s(x, bs="cr", k=6)',
                data,
                "binomial_inverse",
                **arguments,
            )
            assert selected["oracle_stage"] == "selected_before_offset_null_deviance"
            assert selected["outer_iterations"] == 54
            for field in (
                "coefficients",
                "fitted_values",
                "smoothing_params",
                "Vp",
                "deviance",
                "reml_score",
                "scale",
                "edf_total",
            ):
                assert np.all(np.isfinite(selected[field])), field


@pytest.mark.skipif(not _r_available(), reason="R with mgcv not available")
def test_public_fixed_sp_scale_derivatives_reproduce_newton_state() -> None:
    """Matched-state gam.fit3 reproduces newton()'s state where it belongs.

    newton()'s outer gradient/Hessian describe the returned ``reml.scale``
    only after ``full convergence``; there the matched-state evaluation at
    the returned scale must reproduce them. Which generated cells converge
    fully is roundoff- and BLAS-dependent, so at least one must.
    """
    from jaxgam.families.standard import Gamma, Gaussian
    from tests.test_execution.test_regular_starts import _case

    bridge = RBridge(mode="rpy2")
    checks = _AssertCollector()
    full = 0
    for family_class, link, sp in (
        (Gaussian, "log", 0.5),
        (Gaussian, "sqrt", 0.5),
        (Gamma, "inverse", 0.5),
        (Gamma, "sqrt", 0.5),
        (Gamma, "probit", 0.5),
        # Parametric y ~ x: newton's only parameter is log(scale).
        (Gaussian, "log", None),
        (Gamma, "inverse", None),
        (Gamma, "sqrt", None),
    ):
        family, data, weight, offset, _ = _case(family_class, link)
        name = family.family_name.lower()
        public = bridge.public_fixed_sp_fit(data, weight, offset, name, link, sp=sp)
        derivatives = bridge.public_fixed_sp_scale_derivatives(
            data, weight, offset, name, link, sp=sp
        )
        assert derivatives["returned_scale"] == public["score_scale"]
        assert derivatives["scale"] == public["score_scale"]
        np.testing.assert_allclose(
            derivatives["reml_scale"],
            public["score_scale"],
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        assert derivatives["outer_status"] == public["outer_status"]
        assert derivatives["inner_converged"]
        if derivatives["outer_status"] != "full convergence":
            continue
        full += 1
        for field, outer in (
            ("gradient", np.ravel(public["outer_gradient"])[-1]),
            ("hessian", np.ravel(public["outer_hessian"])[-1]),
        ):
            checks.check(
                f"{name}/{link} sp={sp} {field}",
                lambda field=field, outer=outer, derivatives=derivatives: (
                    np.testing.assert_allclose(
                        derivatives[field],
                        outer,
                        rtol=STRICT.rtol,
                        atol=STRICT.atol,
                    )
                ),
            )
    assert full > 0
    checks.raise_if_any("Matched-state pinned newton() derivatives")
    # A supplied scale is honored and the pinned sp coordinate is retained.
    family, data, weight, offset, _ = _case(Gaussian, "log")
    shifted = bridge.public_fixed_sp_scale_derivatives(
        data, weight, offset, "gaussian", "log", sp=0.5, scale=1e-3
    )
    assert shifted["scale"] == 1e-3
    np.testing.assert_allclose(
        shifted["reml_scale"], 1e-3, rtol=STRICT.rtol, atol=STRICT.atol
    )
    assert shifted["gradient"] > 0.0
    with pytest.raises(ValueError, match="unknown-scale"):
        bridge.public_fixed_sp_scale_derivatives(
            data, weight, offset, "poisson", "log", sp=0.5
        )


@pytest.mark.skipif(not _r_available(), reason="R with mgcv not available")
def test_public_exact_reml_derivatives_reproduce_estimated_endpoint_state() -> None:
    """Matched-state gam.fit3 reproduces R's estimated-smoothing endpoint.

    At R's returned (log sp, log scale) the fresh deriv=2 evaluation must
    return the public fit's own sp, scale and score, a converged inner fit and
    a symmetric positive definite outer Hessian. newton()'s outer gradient and
    Hessian describe the returned point only after ``full convergence``;
    there the evaluation must reproduce them. Which cells converge fully is
    platform-dependent, so at least one must.
    """
    from jaxgam.families.standard import Gamma, Gaussian
    from tests.test_validation_matrix import _public_exact_regular_case

    bridge = RBridge(mode="rpy2")
    with pytest.raises(ValueError, match="unknown-scale"):
        bridge.public_exact_reml_derivatives(
            *_public_exact_regular_case(Gamma, "log"), "poisson", "log"
        )
    checks = _AssertCollector()
    full = 0
    for family_class, link in (
        (Gamma, "logit"),
        (Gamma, "inverse_squared"),
        (Gaussian, "identity"),
    ):
        name = family_class.__name__.lower()
        data, weight, offset = _public_exact_regular_case(family_class, link)
        public = bridge.public_exact_reml_fit(data, weight, offset, name, link)
        derivatives = bridge.public_exact_reml_derivatives(
            data, weight, offset, name, link
        )
        assert derivatives["outer_status"] == public["outer_status"]
        assert derivatives["inner_converged"]
        assert derivatives["smoothing"] == public["smoothing"]
        assert derivatives["returned_scale"] == public["score_scale"]
        assert derivatives["returned_score"] == public["score"]
        np.testing.assert_array_equal(
            derivatives["log_parameters"],
            np.log([public["smoothing"], public["score_scale"]]),
        )
        hessian = derivatives["hessian"]
        np.testing.assert_array_equal(hessian, hessian.T)
        assert np.all(np.linalg.eigvalsh(hessian) > 0.0)
        for field, actual, reference in (
            ("REML", derivatives["score"], public["score"]),
            ("reml.scale", derivatives["reml_scale"], public["score_scale"]),
        ):
            checks.check(
                f"{name}/{link} {field}",
                lambda actual=actual, reference=reference: np.testing.assert_allclose(
                    actual, reference, rtol=STRICT.rtol, atol=STRICT.atol
                ),
            )
        if derivatives["outer_status"] != "full convergence":
            continue
        full += 1
        for field, actual, reference in (
            ("gradient", derivatives["gradient"], derivatives["outer_gradient"]),
            ("hessian", hessian, derivatives["outer_hessian"]),
        ):
            checks.check(
                f"{name}/{link} newton {field}",
                lambda actual=actual, reference=reference: np.testing.assert_allclose(
                    np.ravel(actual),
                    np.ravel(reference),
                    rtol=STRICT.rtol,
                    atol=STRICT.atol,
                ),
            )
    assert full > 0
    checks.raise_if_any("Matched-state pinned estimated-endpoint derivatives")


@pytest.mark.skipif(not _r_available(), reason="R with mgcv not available")
class TestRBridgeFitGam:
    def test_fit_gaussian_returns_expected_keys(
        self, gaussian_data: pd.DataFrame
    ) -> None:
        bridge = RBridge()
        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        expected_keys = {
            "coefficients",
            "fitted_values",
            "smoothing_params",
            "edf",
            "edf_total",
            "deviance",
            "null_deviance",
            "scale",
            "reml_scale",
            "Vp",
            "reml_score",
            "theta",
        }
        assert expected_keys <= set(result)
        if bridge.mode == "rpy2":
            assert isinstance(result["inner_converged"], bool)
            assert isinstance(result["outer_converged"], bool)
            assert isinstance(result["outer_status"], str | None)

    def test_fit_gaussian_types(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        assert isinstance(result["coefficients"], np.ndarray)
        assert result["coefficients"].dtype == np.float64
        assert isinstance(result["fitted_values"], np.ndarray)
        assert result["fitted_values"].dtype == np.float64
        assert isinstance(result["smoothing_params"], np.ndarray)
        assert isinstance(result["edf"], np.ndarray)
        assert isinstance(result["edf_total"], float)
        assert isinstance(result["deviance"], float)
        assert isinstance(result["scale"], float)
        assert isinstance(result["Vp"], np.ndarray)
        assert isinstance(result["reml_score"], float)

    def test_fit_gaussian_dimensions(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        n = len(gaussian_data)
        p = len(result["coefficients"])

        assert result["fitted_values"].shape == (n,)
        assert result["Vp"].shape == (p, p)
        assert result["deviance"] >= 0
        assert result["scale"] > 0

    def test_fit_gaussian_no_nan(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        assert np.all(np.isfinite(result["coefficients"]))
        assert np.all(np.isfinite(result["fitted_values"]))
        assert np.isfinite(result["deviance"])

    def test_fit_all_families(self) -> None:
        """All four v1.0 families should work."""
        bridge = RBridge()
        rng = np.random.default_rng(456)
        n = 100
        x = rng.uniform(0, 1, n)

        # Gaussian
        result = bridge.fit_gam(
            "y ~ s(x)",
            pd.DataFrame({"x": x, "y": np.sin(x) + rng.normal(0, 0.2, n)}),
            family="gaussian",
        )
        assert len(result["coefficients"]) > 0

        # Binomial
        eta = 2 * np.sin(2 * np.pi * x)
        p = 1 / (1 + np.exp(-eta))
        result = bridge.fit_gam(
            "y ~ s(x)",
            pd.DataFrame({"x": x, "y": rng.binomial(1, p, n).astype(float)}),
            family="binomial",
        )
        assert len(result["coefficients"]) > 0

        # Poisson
        result = bridge.fit_gam(
            "y ~ s(x)",
            pd.DataFrame(
                {"x": x, "y": rng.poisson(np.exp(np.sin(x)), n).astype(float)}
            ),
            family="poisson",
        )
        assert len(result["coefficients"]) > 0

        # Gamma
        mu = np.exp(0.5 * np.sin(2 * np.pi * x) + 1)
        result = bridge.fit_gam(
            "y ~ s(x)",
            pd.DataFrame({"x": x, "y": rng.gamma(5, scale=mu / 5, size=n)}),
            family="gamma",
        )
        assert len(result["coefficients"]) > 0

    def test_invalid_family_raises(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        with pytest.raises(ValueError, match="Unknown family"):
            bridge.fit_gam("y ~ s(x)", gaussian_data, family="tweedie")

    def test_vp_is_symmetric(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")
        vp = result["Vp"]
        np.testing.assert_allclose(
            vp, vp.T, atol=STRICT.atol, err_msg="Vp must be symmetric"
        )


@pytest.mark.skipif(not _r_available(), reason="R with mgcv not available")
class TestRBridgeSmoothComponents:
    def test_get_smooth_components_keys(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.get_smooth_components(
            "y ~ s(x)", gaussian_data, family="gaussian"
        )

        # Should have all fit_gam keys plus basis/penalty
        assert "basis_matrices" in result
        assert "penalty_matrices" in result
        assert "coefficients" in result

    def test_penalty_matrix_psd(self, gaussian_data: pd.DataFrame) -> None:
        bridge = RBridge()
        result = bridge.get_smooth_components(
            "y ~ s(x, k=10)", gaussian_data, family="gaussian"
        )

        for penalties in result["penalty_matrices"]:
            for S in penalties:
                assert S.shape[0] == S.shape[1], "Penalty must be square"
                eigenvalues = np.linalg.eigvalsh(S)
                assert np.all(eigenvalues >= -STRICT.rtol), "Penalty must be PSD"

    def test_multi_smooth_components(self) -> None:
        bridge = RBridge()
        rng = np.random.default_rng(789)
        n = 100
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + 0.5 * x2 + rng.normal(0, 0.2, n)
        data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

        result = bridge.get_smooth_components(
            "y ~ s(x1) + s(x2)", data, family="gaussian"
        )

        assert len(result["basis_matrices"]) == 2
        assert len(result["penalty_matrices"]) == 2


class TestGPConfigToMgcvM:
    def test_gp_config_to_mgcv_m_table(self) -> None:
        """Round-trip the GP kwargs to mgcv's numeric ``m=`` vector."""
        collector = _AssertCollector()

        def spherical_rho() -> None:
            assert gp_config_to_mgcv_m(
                make_smooth_spec(["x"], bs="gp", k=-1, kernel="spherical", rho=0.5)
            ) == [1.0, 0.5]

        def stationary_spherical() -> None:
            assert gp_config_to_mgcv_m(
                make_smooth_spec(
                    ["x"],
                    bs="gp",
                    k=-1,
                    kernel="spherical",
                    rho=0.5,
                    stationary=True,
                )
            ) == [-1.0, 0.5]

        def squared_exp() -> None:
            assert gp_config_to_mgcv_m(
                make_smooth_spec(
                    ["x"],
                    bs="gp",
                    k=-1,
                    kernel="power_exponential",
                    rho=0.5,
                    power=2.0,
                )
            ) == [2.0, 0.5, 2.0]

        def matern_5_2_default_rho_omitted() -> None:
            assert gp_config_to_mgcv_m(
                make_smooth_spec(["x"], bs="gp", k=-1, kernel="matern_5_2")
            ) == [4.0]

        def rho_override_beats_spec_rho() -> None:
            assert gp_config_to_mgcv_m(
                make_smooth_spec(
                    ["x"],
                    bs="gp",
                    k=-1,
                    kernel="matern_3_2",
                    rho=0.5,
                ),
                rho=0.7,
            ) == [3.0, 0.7]

        collector.check("spherical+rho", spherical_rho)
        collector.check("stationary spherical", stationary_spherical)
        collector.check("squared-exp", squared_exp)
        collector.check(
            "matern_5_2 default rho omitted",
            matern_5_2_default_rho_omitted,
        )
        collector.check("rho override beats spec.rho", rho_override_beats_spec_rho)
        collector.raise_if_any("gp_config_to_mgcv_m mapping")


@pytest.mark.skipif(not _r_available(), reason="R with mgcv not available")
class TestRBridgeSmoothConstructGP:
    def test_smooth_construct_gp_extracts_knt_and_defn(self) -> None:
        """Bridge round-trips ``knt`` and ``gp.defn`` for a tiny GP smooth."""
        try:
            bridge = RBridge(mode="rpy2")
        except Exception:
            pytest.skip("rpy2 not available")

        data = pd.DataFrame({"x": np.linspace(0.0, 1.0, 30)})
        knots = {"x": np.linspace(0.05, 0.95, 8)}
        # With non-stationary GP, k=9 gives rank 7, so 8 explicit knots
        # exercise mgcv's truncated-eigen branch and produce diagonal S.
        result = bridge.smooth_construct(
            "s(x, bs='gp', k=9)",
            data,
            absorb_cons=False,
            knots=knots,
        )
        collector = _AssertCollector()

        def knt_shape() -> None:
            assert result["knt"].shape == (8, 1)

        def gp_defn_length() -> None:
            assert result["gp_defn"].shape == (3,)

        def gp_defn_default_kernel() -> None:
            np.testing.assert_allclose(result["gp_defn"][0], 3.0)

        def E_shape() -> None:
            assert result["E"].shape == (8, 8)

        def E_symmetric() -> None:
            np.testing.assert_allclose(result["E"], result["E"].T)

        def S_diagonal() -> None:
            np.testing.assert_allclose(
                result["S"][0],
                np.diag(np.diag(result["S"][0])),
            )

        collector.check("knt shape", knt_shape)
        collector.check("gp_defn length 3", gp_defn_length)
        collector.check("gp_defn[0] == 3 (default Matérn 3/2)", gp_defn_default_kernel)
        collector.check("E shape (nk, nk)", E_shape)
        collector.check("E symmetric", E_symmetric)
        collector.check("S diagonal", S_diagonal)
        collector.raise_if_any("GP bridge round-trip")


@pytest.mark.skipif(not _r_available(), reason="R with mgcv not available")
class TestRBridgeSubprocessFallback:
    def test_subprocess_smooth_components(self, gaussian_data: pd.DataFrame) -> None:
        """Subprocess get_smooth_components should return basis/penalty."""
        bridge = RBridge(mode="subprocess")
        result = bridge.get_smooth_components(
            "y ~ s(x, k=10)", gaussian_data, family="gaussian"
        )

        assert "basis_matrices" in result
        assert "penalty_matrices" in result
        assert len(result["basis_matrices"]) == 1
        X_s = result["basis_matrices"][0]
        assert X_s.shape[0] == len(gaussian_data)
        assert X_s.shape[1] == 9  # k-1

    def test_subprocess_smooth_components_matches_rpy2(
        self, gaussian_data: pd.DataFrame
    ) -> None:
        """Subprocess and rpy2 smooth components should match."""
        try:
            bridge_rpy2 = RBridge(mode="rpy2")
        except Exception:
            pytest.skip("rpy2 not available for comparison")

        bridge_sub = RBridge(mode="subprocess")

        r1 = bridge_rpy2.get_smooth_components(
            "y ~ s(x, k=10)", gaussian_data, family="gaussian"
        )
        r2 = bridge_sub.get_smooth_components(
            "y ~ s(x, k=10)", gaussian_data, family="gaussian"
        )

        np.testing.assert_allclose(
            r1["coefficients"], r2["coefficients"], rtol=STRICT.rtol
        )
        np.testing.assert_allclose(
            r1["basis_matrices"][0], r2["basis_matrices"][0], rtol=STRICT.rtol
        )
        for s1, s2 in zip(
            r1["penalty_matrices"][0], r2["penalty_matrices"][0], strict=True
        ):
            np.testing.assert_allclose(s1, s2, rtol=STRICT.rtol, atol=STRICT.atol)

    def test_subprocess_matches_rpy2(self, gaussian_data: pd.DataFrame) -> None:
        """Both modes should produce identical results."""
        try:
            bridge_rpy2 = RBridge(mode="rpy2")
        except Exception:
            pytest.skip("rpy2 not available for comparison")

        bridge_sub = RBridge(mode="subprocess")

        r1 = bridge_rpy2.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")
        r2 = bridge_sub.fit_gam("y ~ s(x)", gaussian_data, family="gaussian")

        np.testing.assert_allclose(
            r1["coefficients"], r2["coefficients"], rtol=STRICT.rtol
        )
        np.testing.assert_allclose(
            r1["fitted_values"], r2["fitted_values"], rtol=STRICT.rtol
        )
        np.testing.assert_allclose(r1["deviance"], r2["deviance"], rtol=STRICT.rtol)
        np.testing.assert_allclose(r1["edf_total"], r2["edf_total"], rtol=STRICT.rtol)
