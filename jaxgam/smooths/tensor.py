"""Tensor product smooths (te, ti basis types).

Implements tensor product (te) and tensor interaction (ti) smooth
construction via row-wise Kronecker products of marginal bases.

- te(): Full tensor product — captures marginal + interaction effects
- ti(): Tensor interaction — captures ONLY interaction (for ANOVA decomposition)

This module is Phase 1 (NumPy only, no JAX imports).

R source reference: R/smooth.r smooth.construct.tensor.smooth.spec()
"""

from __future__ import annotations

import copy
from functools import reduce

import numpy as np
import numpy.typing as npt
from scipy import linalg

from jaxgam.formula.terms import SmoothSpec
from jaxgam.penalties.penalty import Penalty
from jaxgam.smooths.base import Smooth
from jaxgam.smooths.source_lapack import (
    SourceConstraintQR,
    source_matprod,
    source_svd,
)
from jaxgam.smooths.source_penalty_coordinates import (
    scipy_uses_accelerate_lapack,
    source_symmetric_max_eigenvalue,
)
from jaxgam.smooths.utils import row_tensor as _row_tensor

#: smooth.construct.tensor.smooth.spec skips a margin reparameterization when
#: ``sv$d[np]/sv$d[1] < .Machine$double.eps^.66`` ("unstable ... not done").
_REPARAMETERIZATION_FLOOR = np.finfo(float).eps ** 0.66


class SourceColumnMeans:
    """R's ``colMeans``: long-double column sums in source row order.

    do_colsum accumulates each column in ``LDOUBLE`` and divides there, so the
    accumulator's precision follows the platform's ``long double`` as R's does.
    """

    def __init__(self, width: int) -> None:
        self._sum = np.zeros(width, dtype=np.longdouble)
        self.count = 0

    def add(self, rows: npt.NDArray[np.floating]) -> None:
        rows = np.asarray(rows, dtype=np.longdouble)
        if len(rows):
            # cumsum's running sums fix the sequential source-row order.
            stacked = np.vstack((self._sum[np.newaxis], rows))
            self._sum = np.cumsum(stacked, axis=0)[-1].copy()
            self.count += len(rows)

    def result(self) -> npt.NDArray[np.float64]:
        if self.count == 0:
            raise ValueError("Cannot average columns without rows.")
        return np.asarray(self._sum / np.longdouble(self.count), dtype=np.float64)


def source_margin_constraint(
    rows: npt.NDArray[np.floating],
) -> SourceConstraintQR:
    """smoothCon's ti-margin centering constraint ``qr(t(colMeans(X)))``."""
    means = SourceColumnMeans(rows.shape[1])
    means.add(rows)
    return SourceConstraintQR.from_column(means.result())


def source_margin_reparameterization(
    Xp: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64] | None:
    """Return ``sv$v %*% (t(sv$u)/sv$d)`` for ``sv <- svd(Xp)``, or ``None``.

    ``None`` is smooth.r's "reparameterization unstable for margin" branch.
    """
    u, d, vt = source_svd(Xp)
    if d[-1] / d[0] < _REPARAMETERIZATION_FLOOR:
        return None
    return source_matprod(np.ascontiguousarray(vt.T), u.T / d[:, np.newaxis])


def source_reparameterized_penalty(
    S: npt.NDArray[np.floating], XP: npt.NDArray[np.floating]
) -> npt.NDArray[np.float64]:
    """smooth.r's ``t(XP) %*% Sm %*% XP``, evaluated left to right."""
    return source_matprod(source_matprod(np.ascontiguousarray(XP.T), S), XP)


# ---------------------------------------------------------------------------
# TensorProductSmooth (te)
# ---------------------------------------------------------------------------


class TensorProductSmooth(Smooth):
    """Tensor product smooth (te()).

    Constructs a tensor product basis from marginal smooths via
    row-wise Kronecker products. Each marginal contributes one
    penalty matrix to the tensor product.

    Parameters
    ----------
    spec : SmoothSpec
        Smooth term specification with multiple variables.
    """

    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)
        self._marginals: list[Smooth] = []
        self._penalties: list[Penalty] = []
        self._XP_list: list[npt.NDArray[np.floating] | None] = []
        self._X: npt.NDArray[np.floating] | None = None

    def copy_for_prediction(self) -> TensorProductSmooth:
        """Drop tensor penalties and fitting caches on nested marginals."""
        clone = copy.copy(self)
        if getattr(clone, "_X", None) is not None:
            clone._X = None
        clone._penalties = []
        clone._marginals = [
            marginal.copy_for_prediction() for marginal in self._marginals
        ]
        return clone

    def copy_for_training(self) -> TensorProductSmooth:
        """Return a predict-only copy whose margins evaluate training rows."""
        clone = self.copy_for_prediction()
        clone._marginals = [
            marginal.copy_for_training() for marginal in self._marginals
        ]
        return clone

    def _margin_constraint(self, j: int) -> SourceConstraintQR | None:
        """Return margin ``j``'s absorbed centering constraint (ti only)."""
        constraints = getattr(self, "_constraints", None)
        return None if not constraints else constraints[j]

    def _reparameterize_margin(
        self,
        j: int,
        marginal: Smooth,
        lower: float,
        upper: float,
        width: int,
    ) -> npt.NDArray[np.float64] | None:
        """Compute smooth.r's margin reparameterization ``XP`` or ``None``.

        The evaluation points are ``seq(min(x), max(x), length=np)`` and the
        margin is predicted (with any absorbed constraint) at them. Classes
        that are already optimal (``noterp``) and multi-covariate margins keep
        their basis.
        """
        if marginal._noterp or len(marginal.spec.variables) != 1:
            return None
        points = np.linspace(lower, upper, width)
        Xp = marginal.predict_matrix({marginal.spec.variables[0]: points})
        constraint = self._margin_constraint(j)
        if constraint is not None:
            Xp = constraint.absorb_rows(Xp)
        return source_margin_reparameterization(Xp)

    def _transform_margin(
        self, j: int, X: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Apply margin ``j``'s constraint (PredictMat) and ``%*% XP``."""
        constraint = self._margin_constraint(j)
        if constraint is not None:
            X = constraint.absorb_rows(X)
        XP = self._XP_list[j]
        return X if XP is None else source_matprod(X, XP)

    def _reparameterize_margins(
        self,
        X_list: list[npt.NDArray[np.floating]],
        S_list: list[npt.NDArray[np.floating]],
        data: dict[str, npt.NDArray[np.floating]],
    ) -> None:
        """Set ``_XP_list`` and transform the margin bases and penalties."""
        self._XP_list = []
        for j, marginal in enumerate(self._marginals):
            values = np.asarray(data[marginal.spec.variables[0]], dtype=float)
            XP = self._reparameterize_margin(
                j, marginal, np.min(values), np.max(values), X_list[j].shape[1]
            )
            self._XP_list.append(XP)
            if XP is not None:
                X_list[j] = source_matprod(X_list[j], XP)
                S_list[j] = source_reparameterized_penalty(S_list[j], XP)

    def _resolve_marginal_m(self, n_margins: int) -> list[int | None]:
        """Resolve the per-marginal penalty order ``m`` from ``spec.extra_args``.

        Mirrors R's ``te()``/``ti()`` (R/smooth.r:443,448): a scalar ``m`` is
        replicated to every margin; a per-margin list/tuple (one int per margin)
        is used as-is; negative orders are clamped to 0. ``None`` (``m`` absent)
        leaves each marginal at its own default order, preserving the prior
        default behaviour exactly. Only ``tp``/``ts`` margins consult ``m``;
        ``cr``/``cs``/``cc`` ignore it, matching R.
        """
        raw_m = self.spec.extra_args.get("m")
        if raw_m is None:
            return [None] * n_margins
        if isinstance(raw_m, (int, np.integer)):
            return [max(int(raw_m), 0)] * n_margins
        if isinstance(raw_m, (list, tuple)):
            if len(raw_m) != n_margins:
                raise ValueError(
                    f"te()/ti() marginal order m has length {len(raw_m)} but the "
                    f"smooth has {n_margins} marginal(s); supply a single integer "
                    f"or one integer per marginal."
                )
            out: list[int | None] = []
            for v in raw_m:
                if v is None:
                    out.append(None)
                elif isinstance(v, (int, np.integer)):
                    out.append(max(int(v), 0))
                else:
                    raise ValueError(
                        "te()/ti() per-margin m entries must be integers; nested "
                        "per-margin vectors (Duchon splines) are not supported."
                    )
            return out
        raise ValueError(
            "te()/ti() argument m must be an integer or a list/tuple of integers "
            "(one per marginal)."
        )

    def _create_marginals(
        self, data: dict[str, npt.NDArray[np.floating]], *, scaled: bool = False
    ) -> list[tuple[Smooth, npt.NDArray[np.floating], npt.NDArray[np.floating]]]:
        """Create and setup marginal smooths, returning (smooth, X, S).

        ``te`` margins use ``smooth.construct`` penalties (``S_raw``); ``ti``
        margins pass ``scaled=True`` for smoothCon's scaled penalties, since R
        constructs constrained margins with ``smoothCon``.

        Returns
        -------
        list of (Smooth, np.ndarray, np.ndarray)
            Each tuple is (marginal_smooth, design_matrix, penalty_matrix).
        """
        # R te()/ti() default marginal k is 5^d per margin (R/smooth.r te():
        # ``if (sum(is.na(k))||is.null(k)) k <- 5^d``). All jaxgam margins are
        # 1-D (d=1) -> default k=5, NOT the univariate s() default of 10. An
        # explicit user k (!= -1) is passed through unchanged to every margin,
        # matching R's ``k <- rep(k, n.bases)``.
        marginal_k = 5 if self.spec.k == -1 else self.spec.k
        # Thread the marginal penalty order m= into each marginal. R's te()/ti()
        # replicate a scalar m to every margin and use a per-margin list as-is
        # (R/smooth.r:443,448). m only changes tp/ts margins; cr/cs/cc ignore it,
        # matching R, and m absent leaves each marginal at its own default order.
        margin_m = self._resolve_marginal_m(len(self.spec.variables))
        marginals_info = []
        for i, var in enumerate(self.spec.variables):
            extra: dict = {} if margin_m[i] is None else {"m": margin_m[i]}
            marginal_spec = SmoothSpec(
                variables=[var],
                bs=self.spec.bs,
                k=marginal_k,
                smooth_type="s",
                extra_args=extra,
            )
            # Lazy import to break circular dependency: registry → tensor → registry
            from jaxgam.smooths.cubic import CubicRegressionSmooth
            from jaxgam.smooths.registry import get_smooth_class

            smooth_cls = get_smooth_class(self.spec.bs)
            marginal = smooth_cls(marginal_spec)
            # R's te() uses smooth.construct() for unconstrained cubic
            # marginals. ti() uses smoothCon() and retains its existing
            # normalization policy.
            raw_cubic = (
                self.spec.smooth_type == "te"
                and self.spec.bs in ("cr", "cs")
                and isinstance(marginal, CubicRegressionSmooth)
            )
            if raw_cubic:
                marginal.setup(data, _normalize_penalty=False)
            else:
                marginal.setup(data)
            X_j = marginal.consume_training_design_matrix()
            # Get the normalized penalty and undo smoothCon normalization
            S_normalized = marginal.build_penalty_matrices()[0].S
            S_raw = S_normalized if raw_cubic else S_normalized * marginal._s_scale
            marginals_info.append((marginal, X_j, S_normalized if scaled else S_raw))
        return marginals_info

    def _get_marginal_matrices(
        self,
        marginals_info: list[
            tuple[Smooth, npt.NDArray[np.floating], npt.NDArray[np.floating]]
        ],
    ) -> tuple[list[npt.NDArray[np.floating]], list[npt.NDArray[np.floating]]]:
        """Extract design matrices and penalty matrices from marginals info.

        Returns
        -------
        X_list : list[np.ndarray]
            Marginal design matrices.
        S_list : list[np.ndarray]
            Raw (un-normalized) marginal penalty matrices.
        """
        X_list = [info[1] for info in marginals_info]
        S_list = [info[2] for info in marginals_info]
        return X_list, S_list

    def _build_tensor_design(
        self, matrices: list[npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build tensor product design matrix via chained row-wise Kronecker."""
        return reduce(_row_tensor, matrices)

    def _build_tensor_penalties(
        self,
        S_list: list[npt.NDArray[np.floating]],
        dims: list[int],
        ranks: list[int],
    ) -> list[Penalty]:
        """Build tensor product penalty matrices.

        For each marginal j, constructs:
            P_j = I_{d_0} ⊗ ... ⊗ S_j_norm ⊗ ... ⊗ I_{d_{m-1}}

        where S_j_norm is normalized by its largest eigenvalue.

        Parameters
        ----------
        S_list : list[np.ndarray]
            Raw penalty matrices for each marginal.
        dims : list[int]
            Number of coefficients for each marginal.
        ranks : list[int]
            Penalty rank for each marginal.

        Returns
        -------
        list[Penalty]
            One penalty per marginal.
        """
        n_marginals = len(S_list)
        total_dim = int(np.prod(dims))
        penalties = []

        for j in range(n_marginals):
            # smooth.r: Sm/eigen(Sm, symmetric=TRUE, only.values=TRUE)$values[1],
            # La_rs's value-only DSYEVR on the pinned provider. Cubic margins
            # keep SciPy's EVR where SciPy links Accelerate (PR9 provider rule).
            matrix = np.asarray(S_list[j], dtype=np.float64)
            largest = (
                linalg.eigvalsh(matrix, lower=True, driver="evr")[-1]
                if self.spec.bs in ("cr", "cs") and scipy_uses_accelerate_lapack()
                else source_symmetric_max_eigenvalue(matrix)
            )
            S_norm = S_list[j] / largest if largest > 0 else S_list[j].copy()

            # Build Kronecker product: I ⊗ ... ⊗ S_norm ⊗ ... ⊗ I
            P = np.array([[1.0]])
            for i in range(n_marginals):
                P = np.kron(P, S_norm) if i == j else np.kron(P, np.eye(dims[i]))

            # Symmetrize
            P = 0.5 * (P + P.T)

            # Compute rank: rank(S_j) * product(d_i for i != j)
            other_dims = int(np.prod([dims[i] for i in range(n_marginals) if i != j]))
            rank_j = ranks[j] * other_dims

            penalties.append(Penalty(P, rank=rank_j, null_space_dim=total_dim - rank_j))

        return penalties

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct tensor product basis from data.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Must contain keys matching all variables in ``self.spec.variables``.
        """
        # Create and setup marginal smooths
        marginals_info = self._create_marginals(data)
        self._marginals = [info[0] for info in marginals_info]
        X_list, S_list = self._get_marginal_matrices(marginals_info)

        # SVD reparameterize 1D marginals without noterp
        self._reparameterize_margins(X_list, S_list, data)

        dims = [X.shape[1] for X in X_list]
        self.n_coefs = int(np.prod(dims))
        self.null_space_dim = int(np.prod([m.null_space_dim for m in self._marginals]))

        # Build tensor design matrix
        X_tensor = self._build_tensor_design(X_list)
        self._X = X_tensor

        # Build tensor penalty matrices
        self._penalties = self._build_tensor_penalties(
            S_list, dims, [m.rank for m in self._marginals]
        )

        # Apply smoothCon normalization — R normalizes each penalty
        # by its own 1-norm (not all by S[0]'s norm)
        # R's smoothCon uses base::norm(), which dispatches to LAPACK DLANGE.
        # Preserve that reduction order: NumPy's row/column sums can differ by
        # one ULP before the penalty scaling division.
        matrix_norm = linalg.get_lapack_funcs("lange", (X_tensor,))
        # R's norm(X, "I")^2 is R_pow with y == 2, i.e. x * x.
        norm_x = matrix_norm(b"I", X_tensor)
        max_x_sq = norm_x * norm_x
        if max_x_sq > 0:
            normalized = []
            for p in self._penalties:
                maS = matrix_norm(b"O", p.S) / max_x_sq
                normalized.append(
                    Penalty(p.S / maS, rank=p.rank, null_space_dim=p.null_space_dim)
                )
            self._penalties = normalized
        self._s_scale = 1.0

        self.rank = self.n_coefs - self.null_space_dim
        self._is_setup = True

    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build tensor product design matrix for the given data."""
        self._require_setup()
        return self.predict_matrix(data)

    def build_penalty_matrices(self) -> list[Penalty]:
        """Return tensor product penalty matrices (one per marginal)."""
        self._require_setup()
        return list(self._penalties)

    def predict_matrix(
        self, new_data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build prediction matrix for new data."""
        self._require_setup()
        return self._compose_margins(
            [m.predict_matrix(new_data) for m in self._marginals]
        )

    def training_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Row-tensor the margins' source training rows (smoothCon's ``X``)."""
        self._require_setup()
        return self._compose_margins([m.training_matrix(data) for m in self._marginals])

    def _compose_margins(
        self, matrices: list[npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Transform each margin as Predict.matrix.tensor.smooth and row-tensor."""
        return self._build_tensor_design(
            [self._transform_margin(j, X) for j, X in enumerate(matrices)]
        )


# ---------------------------------------------------------------------------
# TensorInteractionSmooth (ti)
# ---------------------------------------------------------------------------


class TensorInteractionSmooth(TensorProductSmooth):
    """Tensor interaction smooth (ti()).

    Like te(), but absorbs sum-to-zero constraints into each marginal
    BEFORE forming the tensor product. This removes marginal main
    effects, capturing ONLY the interaction. Used for ANOVA-style
    decomposition.

    Parameters
    ----------
    spec : SmoothSpec
        Smooth term specification with multiple variables.
    """

    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)
        self._Z_list: list[npt.NDArray[np.floating]] = []
        self._constraints: list[SourceConstraintQR] = []

    @staticmethod
    def _absorb_constraint(
        X: npt.NDArray[np.floating], S: npt.NDArray[np.floating]
    ) -> tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
    ]:
        """Absorb sum-to-zero constraint via QR decomposition.

        For ti() marginals: removes the sum-to-zero direction from the
        basis, reducing each marginal from k to k-1 columns.

        Parameters
        ----------
        X : np.ndarray
            Design matrix, shape ``(n, k)``.
        S : np.ndarray
            Penalty matrix, shape ``(k, k)``.

        Returns
        -------
        X_c : np.ndarray
            Constrained design matrix, shape ``(n, k-1)``.
        S_c : np.ndarray
            Constrained penalty matrix, shape ``(k-1, k-1)``, symmetrized.
            Setup keeps smoothCon's unsymmetrized ``Z'SZ`` instead, which
            tensor.prod.penalties symmetrizes after the Kronecker product.
        Z : np.ndarray
            Constraint absorption matrix, shape ``(k, k-1)``.
        """
        constraint = source_margin_constraint(X)
        S_c = constraint.absorb_penalty(S)
        return constraint.absorb_rows(X), (S_c + S_c.T) * 0.5, constraint.basis()

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct tensor interaction basis from data.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Must contain keys matching all variables in ``self.spec.variables``.
        """
        # Create and setup marginal smooths (smoothCon-scaled penalties)
        marginals_info = self._create_marginals(data, scaled=True)
        self._marginals = [info[0] for info in marginals_info]
        X_list_raw, S_list_raw = self._get_marginal_matrices(marginals_info)

        # Absorb smoothCon's centering constraint into each marginal
        self._constraints = [source_margin_constraint(X_j) for X_j in X_list_raw]
        self._Z_list = [constraint.basis() for constraint in self._constraints]
        X_list = [
            constraint.absorb_rows(X_j)
            for constraint, X_j in zip(self._constraints, X_list_raw, strict=True)
        ]
        S_list = [
            constraint.absorb_penalty(S_j)
            for constraint, S_j in zip(self._constraints, S_list_raw, strict=True)
        ]

        # SVD reparameterize constrained 1D marginals without noterp
        self._reparameterize_margins(X_list, S_list, data)

        constrained_dims = [X.shape[1] for X in X_list]
        self.n_coefs = int(np.prod(constrained_dims))

        # null_space_dim: each marginal's null space is reduced by 1
        constrained_nsds = [max(m.null_space_dim - 1, 0) for m in self._marginals]
        self.null_space_dim = int(np.prod(constrained_nsds)) if constrained_nsds else 0

        # Build tensor design matrix from constrained marginals
        X_tensor = self._build_tensor_design(X_list)
        self._X = X_tensor

        # Build tensor penalties from constrained marginals
        constrained_ranks = [
            d - max(m.null_space_dim - 1, 0)
            for d, m in zip(constrained_dims, self._marginals, strict=True)
        ]

        self._penalties = self._build_tensor_penalties(
            S_list,
            constrained_dims,
            constrained_ranks,
        )

        # Apply smoothCon normalization — per-penalty (matching R)
        max_x_sq = np.linalg.norm(X_tensor, ord=np.inf) ** 2
        if max_x_sq > 0:
            normalized = []
            for p in self._penalties:
                maS = np.linalg.norm(p.S, ord=1) / max_x_sq
                normalized.append(
                    Penalty(p.S / maS, rank=p.rank, null_space_dim=p.null_space_dim)
                )
            self._penalties = normalized
        self._s_scale = 1.0

        self.rank = self.n_coefs - self.null_space_dim
        self._is_setup = True

    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build ti design matrix for the given data."""
        self._require_setup()
        return self.predict_matrix(data)

    def predict_matrix(
        self, new_data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build prediction matrix for new data.

        Uses stored Z and XP matrices from setup to apply the same
        constraint absorption and SVD reparameterization to new data.
        """
        self._require_setup()
        return self._compose_margins(
            [m.predict_matrix(new_data) for m in self._marginals]
        )
