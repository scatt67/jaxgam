"""Gaussian process smooth (``bs="gp"``).

Low-rank kriging basis via truncated eigendecomposition of a kernel
correlation matrix on harvested knots. Phase 1 (NumPy only).

Design reference: ``docs/gaussian_process/design.md`` Sections 5 and 8.3.
R source reference: ``$MGCV_SOURCE/R/smooth.r`` lines 3441-3552.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

from jaxgam.formula.terms import SmoothSpec
from jaxgam.penalties.penalty import Penalty
from jaxgam.smooths.base import Smooth
from jaxgam.smooths.gp_kernels import GPKernel, gp_kernel_registry
from jaxgam.smooths.utils import (
    _compute_distance_matrix,
    _get_unique_rows,
    _slanczos,
    _subsample_knots,
)

_DEF_K = (10, 30, 100)


class GaussianProcessSmooth(Smooth):
    """Low-rank Gaussian process smooth (``bs="gp"``).

    Reduced-rank kriging basis via truncated eigendecomposition of a
    correlation matrix evaluated at harvested knots. Supports five kernels
    (spherical, power-exponential, Matern 3/2, 5/2, 7/2) with a single
    scalar range parameter rho.

    User-facing arguments live on ``spec.extra_args``:

    - ``kernel``: str, default ``"matern_3_2"``. Case-insensitive; one of
      ``{"spherical", "power_exponential", "matern_3_2", "matern_5_2",
      "matern_7_2"}``.
    - ``rho``: float > 0, optional. Kernel range; defaults to the
      Kammann-Wand maximum pairwise knot distance.
    - ``power``: float in (0, 2], default ``1.0``. Only consulted by
      ``kernel="power_exponential"``.
    - ``stationary``: bool, default ``False``. If True, drop the linear
      trend from the null space.
    - ``xt``: dict, optional. ``xt["max_knots"]`` (int, default 2000) and
      ``xt["seed"]`` (int, default 1).

    Notes
    -----
    mgcv encodes these knobs as a signed numeric vector ``m=c(sign*type,
    rho, power)``. Passing ``m=...`` here raises ``ValueError`` — the
    mgcv mapping happens only at the R-bridge boundary (design §6.4).

    The truncated eigendecomposition can produce negative eigenvalues
    on some kernel/data combinations. ``setup`` clips them to ``|lambda|``
    and warns; the fitting layer assumes PSD penalties (design §8.3).
    """

    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)

        if "m" in spec.extra_args:
            raise ValueError(
                "mgcv-style `m=` is not supported for JaxGAM GP smooths. "
                "Use `kernel=`, `rho=`, `power=`, and `stationary=` instead. "
                "See docs/gaussian_process/design.md §6.4 for the mapping."
            )

        kernel_name = spec.extra_args.get("kernel", "matern_3_2")
        self._kernel: GPKernel = gp_kernel_registry.get_instance(kernel_name)

        self._rho: float | None = spec.extra_args.get("rho")
        if self._rho is not None and self._rho <= 0.0:
            raise ValueError(
                "GP `rho` must be positive (or omitted for auto-selection); "
                f"got {self._rho!r}."
            )
        self._power: float = spec.extra_args.get("power", 1.0)
        self._stationary: bool = spec.extra_args.get("stationary", False)
        self._kernel.validate(self._power)

        self._resolved_rho: float | None = None
        self._shift: npt.NDArray[np.floating] | None = None
        self._knt: npt.NDArray[np.floating] | None = None
        self._E_knot: npt.NDArray[np.floating] | None = None
        self._UZ: npt.NDArray[np.floating] | None = None
        self._X: npt.NDArray[np.floating] | None = None
        self._S: npt.NDArray[np.floating] | None = None

    def copy_for_prediction(self) -> GaussianProcessSmooth:
        """Return predict-only GP state, defensively dropping the knot kernel."""
        clone = super().copy_for_prediction()
        clone._E_knot = None
        return clone

    @staticmethod
    def _default_bs_dim(k_spec: int, d: int) -> int:
        if k_spec is not None and k_spec > 0:
            bs_dim = k_spec
        else:
            if d > 3:
                raise ValueError(
                    "Default basis dim for GP smooth with d > 3 is undefined. "
                    "Please specify k explicitly."
                )
            bs_dim = d + 1 + _DEF_K[d - 1]
        min_bs = d + 2
        if bs_dim < min_bs:
            warnings.warn(
                f"GP basis dimension {bs_dim} below minimum {min_bs}; "
                f"reset to {min_bs}.",
                stacklevel=2,
            )
            bs_dim = min_bs
        return bs_dim

    def _harvest_knots(
        self,
        x: npt.NDArray[np.floating],
        max_knots: int,
        seed: int,
    ) -> npt.NDArray[np.floating]:
        xu, _ = _get_unique_rows(x)
        return _subsample_knots(xu, max_knots, seed=seed)

    def _gp_T(self, x_centered: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        n = x_centered.shape[0]
        if self._stationary:
            return np.ones((n, 1))
        return np.column_stack([np.ones(n), x_centered])

    def _gp_E(
        self,
        x: npt.NDArray[np.floating],
        xk: npt.NDArray[np.floating],
        *,
        resolved_rho: float | None = None,
    ) -> tuple[npt.NDArray[np.floating], float]:
        distances = _compute_distance_matrix(x, xk)

        if resolved_rho is not None:
            rho = float(resolved_rho)
        elif self._rho is not None:
            rho = float(self._rho)
        else:
            rho = float(distances.max())

        if rho <= 0.0:
            raise ValueError(
                f"GP kernel range `rho` must be positive; got {rho!r}. "
                "This usually means the knot set is degenerate (all rows "
                "identical) — check the data and the `max_knots` setting."
            )

        E = self._kernel.evaluate(distances / rho, power=self._power)
        return E, rho

    def _build_design(
        self, x_centered: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        E_xn, _ = self._gp_E(x_centered, self._knt, resolved_rho=self._resolved_rho)
        pen_block = E_xn @ self._UZ
        null_block = self._gp_T(x_centered)
        return np.hstack([pen_block, null_block])

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        for v in self.spec.variables:
            if v not in data:
                raise KeyError(
                    f"Variable '{v}' not found in data. Available: {list(data.keys())}"
                )

        xt = self.spec.extra_args.get("xt", {})
        max_knots = xt.get("max_knots", xt.get("max.knots", 2000))
        seed = xt.get("seed", 1)

        variables = self.spec.variables
        d = len(variables)
        x = np.column_stack([np.asarray(data[v], dtype=float) for v in variables])

        knt = self._harvest_knots(x, max_knots, seed)
        nk = knt.shape[0]

        null_space_dim = 1 if self._stationary else d + 1
        bs_dim = self._default_bs_dim(self.spec.k, d)
        k = bs_dim - null_space_dim
        if nk < bs_dim:
            raise ValueError(
                "A term has fewer unique covariate combinations than "
                "specified maximum degrees of freedom "
                f"(nk={nk}, bs_dim={bs_dim})."
            )

        self._shift = x.mean(axis=0)
        x_c = x - self._shift
        knt_c = knt - self._shift
        self._knt = knt_c

        E, rho_resolved = self._gp_E(knt_c, knt_c)
        self._resolved_rho = rho_resolved
        self._E_knot = E

        eigvals, eigvecs = _slanczos(E, k, tol=np.finfo(float).eps ** 0.5)
        self._E_knot = None

        if (eigvals < 0).any():
            warnings.warn(
                f"GP smooth on terms {self.spec.variables}: "
                f"{int((eigvals < 0).sum())} of {len(eigvals)} truncated "
                "eigenvalues are negative (indefinite kernel for these data). "
                "Replacing with |lambda|; this deviates from mgcv for these "
                "spectra. See docs/gaussian_process/design.md §8.3.",
                stacklevel=2,
            )
            eigvals = np.abs(eigvals)

        D = np.zeros((bs_dim, bs_dim))
        np.fill_diagonal(D[:k, :k], eigvals)
        self._UZ = eigvecs

        self.null_space_dim = null_space_dim
        self.rank = k
        self.n_coefs = bs_dim

        X = self._build_design(x_c)
        self._X = X

        [self._S], self._s_scale = self._smoothcon_normalize(X, [D])

        self._is_setup = True

    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        self._require_setup()
        return self.predict_matrix(data)

    def build_penalty_matrices(self) -> list[Penalty]:
        self._require_setup()
        return [
            Penalty(
                self._S,
                rank=self.rank,
                null_space_dim=self.null_space_dim,
            )
        ]

    def predict_matrix(
        self, new_data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        self._require_setup()
        x = np.column_stack(
            [np.asarray(new_data[v], dtype=float) for v in self.spec.variables]
        )
        x_c = x - self._shift
        return self._build_design(x_c)
