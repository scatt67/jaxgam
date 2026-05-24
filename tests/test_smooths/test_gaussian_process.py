"""Structural tests for the Gaussian process smooth class.

Validates ``GaussianProcessSmooth`` per design §5.1-§5.10 and §8.3.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from jaxgam.formula.terms import SmoothSpec
from jaxgam.smooths import gaussian_process as gp_module
from jaxgam.smooths.gaussian_process import GaussianProcessSmooth
from jaxgam.smooths.gp_kernels import GPKernel
from jaxgam.smooths.registry import get_smooth_class
from tests.helpers import _AssertCollector


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
            np.testing.assert_allclose(smooth._shift, [np.mean(x)])

        def kernel_is_gp_kernel() -> None:
            assert isinstance(smooth._kernel, GPKernel)

        def resolved_rho_positive() -> None:
            assert smooth._resolved_rho is not None
            assert smooth._resolved_rho > 0

        def penalty_diagonal() -> None:
            S = smooth.build_penalty_matrices()[0].S
            assert np.allclose(S, np.diag(np.diag(S)))

        def predict_equals_design() -> None:
            X = smooth.build_design_matrix(gp_1d_data)
            X_pred = smooth.predict_matrix(gp_1d_data)
            np.testing.assert_allclose(X_pred, X)

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
            np.testing.assert_allclose(
                penalties[0].S,
                np.diag(np.diag(penalties[0].S)),
            )

        def noterp_is_false() -> None:
            assert margin._noterp is False

        def predict_matrix_roundtrip() -> None:
            np.testing.assert_allclose(
                margin.predict_matrix(gp_1d_data),
                X,
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
