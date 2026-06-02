"""Structural tests for the Gaussian process smooth class.

Validates ``GaussianProcessSmooth`` per design §5.1-§5.10 and §8.3.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from jaxgam.formula.parser import parse_formula
from jaxgam.formula.terms import SmoothSpec
from jaxgam.smooths import gaussian_process as gp_module
from jaxgam.smooths.gaussian_process import GaussianProcessSmooth
from jaxgam.smooths.gp_kernels import GPKernel
from jaxgam.smooths.registry import get_smooth_class
from jaxgam.smooths.tensor import TensorInteractionSmooth, TensorProductSmooth
from tests.helpers import _AssertCollector, r_available
from tests.r_bridge import RBridge, gp_config_to_mgcv_m
from tests.tolerances import STRICT, ToleranceClass

_GP_R_PARITY_K = 9


def _make_spec(
    variables: list[str],
    k: int = -1,
    **extra_args: object,
) -> SmoothSpec:
    return SmoothSpec(
        variables=variables,
        bs="gp",
        k=k,
        by=None,
        smooth_type="s",
        extra_args=dict(extra_args),
    )


def _make_gp_smooth(
    data: dict,
    stationary: bool,
    k: int = -1,
    **extra_args: object,
) -> GaussianProcessSmooth:
    spec = _make_spec(
        list(data),
        k=k,
        stationary=stationary,
        **extra_args,
    )
    smooth = GaussianProcessSmooth(spec)
    smooth.setup(data)
    return smooth


def _assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: ToleranceClass,
) -> None:
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
    )


def _data_dict(data: pd.DataFrame | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(data, pd.DataFrame):
        return {col: data[col].to_numpy() for col in data.columns}
    return data


def _build_gp(
    spec: SmoothSpec,
    data_payload: dict[str, object],
) -> GaussianProcessSmooth:
    smooth = GaussianProcessSmooth(spec)
    smooth.setup(_data_dict(data_payload["data"]))
    return smooth


def _setup_r_and_py_gp(
    gp_kwargs: dict[str, object],
    data_payload: dict[str, object],
    r_bridge: RBridge,
) -> tuple[GaussianProcessSmooth, dict[str, object]]:
    if r_bridge.mode != "rpy2":
        pytest.skip("GP smooth_construct parity requires rpy2")

    # The fixture has 10 unique knot locations. k=9 avoids the default
    # k=12 unique-row floor and keeps mgcv in its truncated-eigen branch.
    spec = SmoothSpec(
        variables=["x"],
        bs="gp",
        k=_GP_R_PARITY_K,
        by=None,
        smooth_type="s",
        extra_args=gp_kwargs,
    )
    m_args = gp_config_to_mgcv_m(spec)
    r_formula = f"s(x, bs='gp', k={_GP_R_PARITY_K}, m=c({','.join(map(str, m_args))}))"
    r_result = r_bridge.smooth_construct(
        r_formula,
        data_payload["data"],
        knots=data_payload["knots"],
    )
    return _build_gp(spec, data_payload), r_result


def _build_tensor_gp(
    formula: str,
    data: pd.DataFrame,
) -> TensorProductSmooth | TensorInteractionSmooth:
    spec = parse_formula(f"y ~ {formula}").smooth_terms[0]
    smooth_cls = {
        "te": TensorProductSmooth,
        "ti": TensorInteractionSmooth,
    }[spec.smooth_type]
    smooth = smooth_cls(spec)
    smooth.setup(_data_dict(data))
    return smooth


def _setup_r_and_py_tensor_gp(
    py_formula: str,
    r_formula: str,
    data: pd.DataFrame,
    r_bridge: RBridge,
) -> tuple[TensorProductSmooth | TensorInteractionSmooth, dict[str, object]]:
    if r_bridge.mode != "rpy2":
        pytest.skip("tensor GP smooth_construct parity requires rpy2")

    r_result = r_bridge.smooth_construct(r_formula, data)
    return _build_tensor_gp(py_formula, data), r_result


class TestSetupInvariants:
    """Structural invariants of ``GaussianProcessSmooth.setup``."""

    @pytest.mark.parametrize("stationary", [False, True])
    def test_setup_state(self, gp_1d_data: dict, stationary: bool) -> None:
        """All post-setup invariants for one stationarity mode via a collector."""
        smooth = _make_gp_smooth(gp_1d_data, stationary=stationary)
        d = len(smooth.spec.variables)
        expected_nsd = 1 if stationary else d + 1
        collector = _AssertCollector()

        def null_space_dim() -> None:
            assert smooth.null_space_dim == expected_nsd

        def rank_matches() -> None:
            assert smooth.rank == smooth.n_coefs - smooth.null_space_dim

        def side_constrain_default() -> None:
            assert smooth.side_constrain is True

        def no_re_flags() -> None:
            assert not getattr(smooth, "_random", False)
            assert getattr(smooth, "_has_centering_constraint", True) is not False

        def shift_is_column_mean() -> None:
            x = gp_1d_data["x"]
            _assert_close(smooth._shift, np.array([np.mean(x)]), STRICT)

        def kernel_is_gp_kernel() -> None:
            assert isinstance(smooth._kernel, GPKernel)

        def resolved_rho_positive() -> None:
            assert smooth._resolved_rho is not None
            assert smooth._resolved_rho > 0

        def penalty_diagonal() -> None:
            S = smooth.build_penalty_matrices()[0].S
            _assert_close(S, np.diag(np.diag(S)), STRICT)

        def predict_equals_design() -> None:
            X = smooth.build_design_matrix(gp_1d_data)
            X_pred = smooth.predict_matrix(gp_1d_data)
            _assert_close(X_pred, X, STRICT)

        collector.check("null_space_dim", null_space_dim)
        collector.check("rank == n_coefs - null_space_dim", rank_matches)
        collector.check("side_constrain == True", side_constrain_default)
        collector.check("no _random / _has_centering_constraint", no_re_flags)
        collector.check("_shift == colMeans", shift_is_column_mean)
        collector.check("self._kernel is a GPKernel", kernel_is_gp_kernel)
        collector.check("_resolved_rho > 0", resolved_rho_positive)
        collector.check("penalty diagonal", penalty_diagonal)
        collector.check("predict_matrix == build_design_matrix", predict_equals_design)
        collector.raise_if_any(f"setup invariants (stationary={stationary})")

    def test_dimension_defaults(self, gp_1d_data: dict, gp_2d_data: dict) -> None:
        """Default bs.dim per dimension and the d>3 explicit-k requirement."""
        rng = np.random.default_rng(0)
        gp_3d_data = {
            "x1": rng.uniform(0, 1, 200),
            "x2": rng.uniform(0, 1, 200),
            "x3": rng.uniform(0, 1, 200),
        }
        gp_4d_data = {**gp_3d_data, "x4": rng.uniform(0, 1, 200)}
        collector = _AssertCollector()

        def check_1d() -> None:
            spec = _make_spec(["x"])
            smooth = GaussianProcessSmooth(spec)
            smooth.setup(gp_1d_data)
            assert smooth.n_coefs == 12

        def check_2d() -> None:
            spec = _make_spec(["x", "z"])
            smooth = GaussianProcessSmooth(spec)
            smooth.setup(gp_2d_data)
            assert smooth.n_coefs == 33

        def check_3d() -> None:
            spec = _make_spec(["x1", "x2", "x3"])
            smooth = GaussianProcessSmooth(spec)
            smooth.setup(gp_3d_data)
            assert smooth.n_coefs == 104

        def check_4d_raises() -> None:
            spec = _make_spec(["x1", "x2", "x3", "x4"])
            with pytest.raises(ValueError, match="d > 3"):
                GaussianProcessSmooth(spec).setup(gp_4d_data)

        collector.check("1D default bs_dim == 12", check_1d)
        collector.check("2D default bs_dim == 33", check_2d)
        collector.check("3D default bs_dim == 104", check_3d)
        collector.check("d > 3 raises", check_4d_raises)
        collector.raise_if_any("dimension defaults")

    def test_init_rejects_m_argument(self) -> None:
        """``m=`` is rejected at construction time."""
        spec = _make_spec(["x"], m=[3, 0.5])
        with pytest.raises(ValueError, match="kernel="):
            GaussianProcessSmooth(spec)


class TestTensorMarginInvariant:
    """Contract required by tensor GP margins."""

    def test_works_as_tensor_margin(self, gp_1d_data: dict) -> None:
        """Single-variable GP smooth satisfies tensor margin assumptions."""
        spec = SmoothSpec(variables=["x"], bs="gp", k=5, smooth_type="s")
        margin_cls = get_smooth_class(spec.bs)
        margin = margin_cls(spec)
        margin.setup(gp_1d_data)
        X = margin.build_design_matrix(gp_1d_data)
        penalties = margin.build_penalty_matrices()
        collector = _AssertCollector()

        def registry_dispatches_gp() -> None:
            assert margin_cls is GaussianProcessSmooth

        def setup_populated_s_scale() -> None:
            assert margin._s_scale > 0

        def design_matrix_has_bs_dim_cols() -> None:
            assert X.shape[1] == margin.n_coefs

        def one_penalty_per_margin() -> None:
            assert len(penalties) == 1

        def penalty_diagonal() -> None:
            _assert_close(
                penalties[0].S,
                np.diag(np.diag(penalties[0].S)),
                STRICT,
            )

        def noterp_is_false() -> None:
            assert margin._noterp is False

        def predict_matrix_roundtrip() -> None:
            _assert_close(
                margin.predict_matrix(gp_1d_data),
                X,
                STRICT,
            )

        collector.check(
            "registry dispatches gp to GaussianProcessSmooth",
            registry_dispatches_gp,
        )
        collector.check(
            "setup populated _s_scale",
            setup_populated_s_scale,
        )
        collector.check(
            "design matrix has bs_dim cols",
            design_matrix_has_bs_dim_cols,
        )
        collector.check(
            "one penalty per margin",
            one_penalty_per_margin,
        )
        collector.check(
            "penalty diagonal",
            penalty_diagonal,
        )
        collector.check(
            "_noterp is False (so tensor SVD reparam runs)",
            noterp_is_false,
        )
        collector.check(
            "predict_matrix roundtrip",
            predict_matrix_roundtrip,
        )
        collector.raise_if_any("univariate-margin invariants")


class TestKnotSubsampling:
    """Knot subsampling reproducibility and global-RNG isolation."""

    def test_knot_subsampling(self, gp_1d_data: dict, large_gp_data: dict) -> None:
        collector = _AssertCollector()

        def small_data_uses_all_unique_rows() -> None:
            spec = _make_spec(["x"])
            smooth = GaussianProcessSmooth(spec)
            smooth.setup(gp_1d_data)
            unique = np.unique(gp_1d_data["x"])
            assert smooth._knt.shape[0] == unique.shape[0]

        def large_data_caps_at_max_knots() -> None:
            spec = _make_spec(["x"], xt={"max_knots": 500, "seed": 7})
            smooth = GaussianProcessSmooth(spec)
            smooth.setup(large_gp_data)
            assert smooth._knt.shape[0] == 500

        def same_seed_gives_identical_knots() -> None:
            spec1 = _make_spec(["x"], xt={"max_knots": 500, "seed": 7})
            spec2 = _make_spec(["x"], xt={"max_knots": 500, "seed": 7})
            s1 = GaussianProcessSmooth(spec1)
            s2 = GaussianProcessSmooth(spec2)
            s1.setup(large_gp_data)
            s2.setup(large_gp_data)
            np.testing.assert_array_equal(s1._knt, s2._knt)

        def global_rng_untouched() -> None:
            # _subsample_knots uses np.random.RandomState(seed) (legacy
            # API for TPRS bit-exactness, see utils.py). Ensure it does
            # not leak into the global legacy RNG state.
            rng = np.random.RandomState(123)
            before_state = rng.get_state()
            np.random.set_state(before_state)  # noqa: NPY002
            spec = _make_spec(["x"], xt={"max_knots": 500, "seed": 99})
            GaussianProcessSmooth(spec).setup(large_gp_data)
            after_state = np.random.get_state()  # noqa: NPY002
            assert before_state[1].tobytes() == after_state[1].tobytes()
            assert before_state[2] == after_state[2]

        collector.check(
            "n <= max_knots → all unique rows",
            small_data_uses_all_unique_rows,
        )
        collector.check(
            "n > max_knots → exactly max_knots",
            large_data_caps_at_max_knots,
        )
        collector.check(
            "same seed → identical knots",
            same_seed_gives_identical_knots,
        )
        collector.check(
            "global RNG untouched after setup",
            global_rng_untouched,
        )
        collector.raise_if_any("knot subsampling")


class TestIndefiniteClip:
    """Negative truncated eigenvalues are clipped (§8.3).

    The supported kernels are PSD or numerically near-PSD at d ≤ 3, so an
    organic "indefinite" spectrum is hard to reproduce reliably. To exercise
    the clip code path deterministically, we monkey-patch ``_slanczos`` to
    return a spectrum with one injected negative eigenvalue. The full
    R-parity behavior on naturally indefinite spectra is exercised in
    Commit G's smooth-construct tests.
    """

    def test_indefinite_eigenvalues_are_clipped(
        self,
        gp_1d_data: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_slanczos = gp_module._slanczos

        def fake_slanczos(A, k, tol):
            eigvals, eigvecs = original_slanczos(A, k, tol=tol)
            eigvals = eigvals.copy()
            # Force the smallest-magnitude eigenvalue to be negative.
            idx = int(np.argmin(np.abs(eigvals)))
            eigvals[idx] = -abs(eigvals[idx]) if eigvals[idx] != 0 else -1e-6
            return eigvals, eigvecs

        monkeypatch.setattr(gp_module, "_slanczos", fake_slanczos)

        spec = _make_spec(["x"])
        smooth = GaussianProcessSmooth(spec)
        with pytest.warns(UserWarning, match="negative"):
            smooth.setup(gp_1d_data)

        S = smooth.build_penalty_matrices()[0].S
        assert (np.diag(S) >= 0).all()

    def test_well_conditioned_setup_does_not_warn(self, gp_1d_data: dict) -> None:
        spec = _make_spec(["x"])  # default Matern 3/2, non-stationary
        smooth = GaussianProcessSmooth(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            smooth.setup(gp_1d_data)


@pytest.mark.skipif(not r_available(), reason="R+mgcv not available")
class TestGPVsR:
    """Direct GP smooth construction parity against mgcv.

    Python clips negative truncated eigenvalues to keep penalties PSD while
    mgcv leaves them unchanged. The penalty-diagonal STRICT check is skipped
    for any R spectrum that exposes this documented parity gap. Raw eigenvector
    columns may also sign-flip, so the design comparison sign-aligns the
    UZ-derived columns before comparing raw ``X``.
    """

    @pytest.mark.parametrize(
        ("gp_kwargs", "label"),
        [
            ({"kernel": "spherical", "rho": 0.5}, "spherical"),
            (
                {"kernel": "power_exponential", "rho": 0.5, "power": 1.0},
                "power_exp_k1",
            ),
            (
                {"kernel": "power_exponential", "rho": 0.5, "power": 2.0},
                "squared_exp",
            ),
            ({"kernel": "matern_3_2", "rho": 0.5}, "matern_3_2"),
            ({"kernel": "matern_3_2"}, "matern_3_2_auto_rho"),
            ({"kernel": "matern_5_2", "rho": 0.5}, "matern_5_2"),
            ({"kernel": "matern_7_2", "rho": 0.5}, "matern_7_2"),
        ],
        ids=[
            "spherical",
            "power_exp_k1",
            "squared_exp",
            "matern_3_2",
            "matern_3_2_auto_rho",
            "matern_5_2",
            "matern_7_2",
        ],
    )
    def test_smooth_construct_matches_r(
        self,
        gp_kwargs: dict[str, object],
        label: str,
        gp_explicit_knots_data: dict[str, object],
        r_bridge,
    ) -> None:
        """Compare E, S, and sign-aligned X against ``smoothCon``.

        The fixture repeats the explicit knot locations as the only unique
        data rows, so Python's normal knot harvesting and R's test-only
        ``knots=`` path operate on identical centered knots.
        """
        py_smooth, r_result = _setup_r_and_py_gp(
            gp_kwargs,
            gp_explicit_knots_data,
            r_bridge,
        )
        r_S_diag = np.diag(r_result["S"][0])
        clipped = (r_S_diag < 0).any()

        collector = _AssertCollector()

        def shift_matches() -> None:
            _assert_close(py_smooth._shift, r_result["shift"], STRICT)

        def centered_knots_match() -> None:
            _assert_close(py_smooth._knt, r_result["knt"], STRICT)

        def resolved_rho_matches() -> None:
            _assert_close(
                np.array([py_smooth._resolved_rho]),
                np.array([r_result["gp_defn"][1]]),
                STRICT,
            )

        def E_matrix_matches() -> None:
            _assert_close(py_smooth._E_knot, r_result["E"], STRICT)

        def UZ_columns_align_without_rotation() -> None:
            column_dots = np.sum(py_smooth._UZ * r_result["UZ"], axis=0)
            _assert_close(
                np.abs(column_dots),
                np.ones_like(column_dots),
                STRICT,
            )

        def penalty_eigenvalues_match() -> None:
            _assert_close(np.diag(py_smooth._S), r_S_diag, STRICT)

        def design_matches_after_UZ_sign_alignment() -> None:
            column_dots = np.sum(py_smooth._UZ * r_result["UZ"], axis=0)
            signs = np.sign(column_dots)
            signs[signs == 0.0] = 1.0
            k = py_smooth._UZ.shape[1]
            py_X_aligned = py_smooth._X.copy()
            py_X_aligned[:, :k] *= signs
            _assert_close(py_X_aligned, r_result["X"], STRICT)

        collector.check("shift STRICT", shift_matches)
        collector.check("centered knots STRICT", centered_knots_match)
        collector.check("resolved rho STRICT", resolved_rho_matches)
        collector.check("E matrix STRICT", E_matrix_matches)
        collector.check(
            "UZ columns STRICT without rotation",
            UZ_columns_align_without_rotation,
        )
        if not clipped:
            collector.check("penalty eigenvalues STRICT", penalty_eigenvalues_match)
        else:
            # Current Commit-G fixtures are PSD; keep this visible for future
            # R-parity configs that expose mgcv's indefinite-spectrum behavior.
            warnings.warn(
                f"Skipping penalty diagonal STRICT check for GP vs R [{label}]: "
                "R returned negative truncated eigenvalues and Python clips them.",
                stacklevel=2,
            )
        collector.check(
            "X STRICT after UZ sign alignment",
            design_matches_after_UZ_sign_alignment,
        )
        collector.raise_if_any(f"GP vs R [{label}]")

    def test_null_space_matches_r(
        self,
        gp_explicit_knots_data: dict[str, object],
        r_bridge,
    ) -> None:
        """Stationary and non-stationary null-space columns match R."""
        collector = _AssertCollector()

        def check_case(gp_kwargs: dict[str, object], expected_nsd: int) -> None:
            py_smooth, r_result = _setup_r_and_py_gp(
                gp_kwargs,
                gp_explicit_knots_data,
                r_bridge,
            )
            assert (
                py_smooth.null_space_dim == r_result["null_space_dim"] == expected_nsd
            )
            _assert_close(
                py_smooth._X[:, -expected_nsd:],
                r_result["X"][:, -expected_nsd:],
                STRICT,
            )

            if expected_nsd == 1:
                _assert_close(
                    py_smooth._X[:, -1],
                    np.ones(py_smooth._X.shape[0]),
                    STRICT,
                )
                _assert_close(
                    r_result["X"][:, -1],
                    np.ones(r_result["X"].shape[0]),
                    STRICT,
                )

        def stationary_case() -> None:
            check_case(
                {"kernel": "matern_3_2", "rho": 0.5, "stationary": True},
                expected_nsd=1,
            )

        def nonstationary_case() -> None:
            check_case(
                {"kernel": "matern_3_2", "rho": 0.5},
                expected_nsd=2,
            )

        collector.check("stationary null.space.dim == 1 and gpT cols", stationary_case)
        collector.check(
            "non-stationary null.space.dim == d+1 and gpT cols",
            nonstationary_case,
        )
        collector.raise_if_any("null space vs R")


@pytest.mark.skipif(not r_available(), reason="R+mgcv not available")
class TestGPTensorMarginVsR:
    """Construction parity for GP margins through existing tensor wrappers.

    ``RBridge.smooth_construct`` exposes the top-level tensor smooth only, so
    this test compares the top-level sign-invariant basis, penalty ranks, and
    null-space dimension. Direct 1-D GP parity above covers the marginal GP
    construction itself.
    """

    @pytest.mark.parametrize(
        ("py_formula", "r_formula", "wrapper"),
        [
            pytest.param(
                "te(x1, x2, bs='gp', k=5)",
                "te(x1, x2, bs='gp', k=c(5, 5))",
                "te",
            ),
            pytest.param(
                "ti(x1, x2, bs='gp', k=5)",
                "ti(x1, x2, bs='gp', k=c(5, 5))",
                "ti",
            ),
        ],
    )
    def test_tensor_construct_matches_r(
        self,
        py_formula: str,
        r_formula: str,
        wrapper: str,
        gp_te_2d_data: pd.DataFrame,
        r_bridge,
    ) -> None:
        py_smooth, r_result = _setup_r_and_py_tensor_gp(
            py_formula,
            r_formula,
            gp_te_2d_data,
            r_bridge,
        )
        py_X = py_smooth.build_design_matrix(_data_dict(gp_te_2d_data))
        py_penalties = py_smooth.build_penalty_matrices()
        collector = _AssertCollector()

        def column_count_matches() -> None:
            expected_cols = 25 if wrapper == "te" else 16
            assert py_X.shape[1] == r_result["X"].shape[1] == expected_cols

        def sign_invariant_design_matches() -> None:
            _assert_close(
                py_X @ py_X.T,
                r_result["X"] @ r_result["X"].T,
                STRICT,
            )

        def all_penalty_ranks_match() -> None:
            py_ranks = np.array([penalty.rank for penalty in py_penalties])
            np.testing.assert_array_equal(py_ranks, r_result["rank_vector"])

        def penalty_frobenius_traces_match() -> None:
            py_traces = np.array(
                [np.trace(penalty.S @ penalty.S.T) for penalty in py_penalties]
            )
            r_traces = np.array([np.trace(S @ S.T) for S in r_result["S"]])
            _assert_close(py_traces, r_traces, STRICT)

        def null_space_dim_matches() -> None:
            assert py_smooth.null_space_dim == r_result["null_space_dim"]

        collector.check("X column count matches", column_count_matches)
        collector.check(
            "X @ X.T STRICT (sign / SVD-reparam-invariant)",
            sign_invariant_design_matches,
        )
        collector.check("all penalty ranks match", all_penalty_ranks_match)
        collector.check(
            "penalty Frobenius traces STRICT",
            penalty_frobenius_traces_match,
        )
        collector.check("null_space_dim matches", null_space_dim_matches)
        collector.raise_if_any(f"tensor GP construct [{wrapper}]")
