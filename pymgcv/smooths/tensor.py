"""Tensor product smooths (te, ti basis types).

Implements tensor product (te) and tensor interaction (ti) smooth
construction via row-wise Kronecker products of marginal bases.

- te(): Full tensor product — captures marginal + interaction effects
- ti(): Tensor interaction — captures ONLY interaction (for ANOVA decomposition)

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 5.5
R source reference: R/smooth.r smooth.construct.tensor.smooth.spec()
"""

from __future__ import annotations

from functools import reduce

import numba
import numpy as np
import numpy.typing as npt
from scipy import linalg

from pymgcv.formula.terms import SmoothSpec
from pymgcv.penalties.penalty import Penalty
from pymgcv.smooths.base import Smooth

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@numba.njit(numba.float64[:, :](numba.float64[:, :], numba.float64[:, :]))
def _row_tensor(
    A: npt.NDArray[np.floating], B: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Row-wise Kronecker product of two matrices.

    For each row i, computes ``A[i, :] ⊗ B[i, :]``.

    Parameters
    ----------
    A : np.ndarray
        Shape ``(n, ka)``.
    B : np.ndarray
        Shape ``(n, kb)``.

    Returns
    -------
    np.ndarray
        Shape ``(n, ka * kb)``.
    """
    n = A.shape[0]
    ka = A.shape[1]
    kb = B.shape[1]
    result = np.empty((n, ka * kb))
    for i in range(ka):
        for j in range(n):
            for k in range(kb):
                result[j, i * kb + k] = A[j, i] * B[j, k]
    return result


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

    @staticmethod
    def _svd_reparameterize(
        marginal: Smooth,
        X: npt.NDArray[np.floating],
        S: npt.NDArray[np.floating],
        data: dict[str, npt.NDArray[np.floating]],
        Z: npt.NDArray[np.floating] | None = None,
    ) -> (
        tuple[
            npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
        ]
        | None
    ):
        """SVD reparameterize a marginal smooth for tensor product construction.

        Replicates R's SVD reparameterization from
        ``smooth.construct.tensor.smooth.spec``. Creates evenly-spaced
        evaluation points, computes the prediction matrix, takes its SVD,
        and transforms the design and penalty matrices.

        Parameters
        ----------
        marginal : Smooth
            The marginal smooth object (must be 1D).
        X : np.ndarray
            Design matrix (possibly constrained for ti).
        S : np.ndarray
            Penalty matrix (possibly constrained for ti).
        data : dict[str, np.ndarray]
            Training data dictionary.
        Z : np.ndarray or None
            Constraint absorption matrix for ti(), or None for te().

        Returns
        -------
        tuple or None
            ``(X_new, S_new, XP)`` if successful, ``None`` if condition
            number is too poor to reparameterize safely.
        """
        var = marginal.spec.variables[0]
        x = data[var]
        n_cols = X.shape[1]

        # Evenly spaced evaluation points across covariate range
        eval_x = np.linspace(np.min(x), np.max(x), n_cols)
        eval_data = {var: eval_x}

        # Prediction matrix at eval points
        Xp = marginal.predict_matrix(eval_data)

        # Apply constraint absorption if ti()
        if Z is not None:
            Xp = Xp @ Z

        # SVD
        U, d, Vt = np.linalg.svd(Xp, full_matrices=False)

        # Condition check: skip if too ill-conditioned
        eps = np.finfo(float).eps
        if d[-1] / d[0] < eps ** (2.0 / 3.0):
            return None

        # Reparameterization matrix: V @ diag(1/d) @ U.T
        V = Vt.T
        XP = V @ (U.T / d[:, np.newaxis])

        # Transform design and penalty
        X_new = X @ XP
        S_new = XP.T @ S @ XP
        S_new = 0.5 * (S_new + S_new.T)

        return X_new, S_new, XP

    def _create_marginals(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> list[tuple[Smooth, npt.NDArray[np.floating], npt.NDArray[np.floating]]]:
        """Create and setup marginal smooths, returning (smooth, X, S_raw).

        Returns
        -------
        list of (Smooth, np.ndarray, np.ndarray)
            Each tuple is (marginal_smooth, design_matrix, raw_penalty_matrix).
        """
        marginals_info = []
        for var in self.spec.variables:
            marginal_spec = SmoothSpec(
                variables=[var],
                bs=self.spec.bs,
                k=self.spec.k,
                smooth_type="s",
            )
            # Lazy import to break circular dependency: registry → tensor → registry
            from pymgcv.smooths.registry import get_smooth_class

            smooth_cls = get_smooth_class(self.spec.bs)
            marginal = smooth_cls(marginal_spec)
            marginal.setup(data)
            X_j = marginal.build_design_matrix(data)
            # Get the normalized penalty and undo smoothCon normalization
            S_normalized = marginal.build_penalty_matrices()[0].S
            S_raw = S_normalized * marginal._s_scale
            marginals_info.append((marginal, X_j, S_raw))
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
            # Normalize by largest eigenvalue (matching R)
            eigvals_j = linalg.eigvalsh(S_list[j])
            max_eigval = np.max(np.abs(eigvals_j))
            S_norm = S_list[j] / max_eigval if max_eigval > 0 else S_list[j].copy()

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
        self._XP_list = []
        for j, marginal in enumerate(self._marginals):
            if not marginal._noterp and len(marginal.spec.variables) == 1:
                result = self._svd_reparameterize(marginal, X_list[j], S_list[j], data)
                if result is not None:
                    X_list[j], S_list[j], XP = result
                    self._XP_list.append(XP)
                else:
                    self._XP_list.append(None)
            else:
                self._XP_list.append(None)

        dims = [X.shape[1] for X in X_list]
        self.n_coefs = int(np.prod(dims))
        self.null_space_dim = int(np.prod([m.null_space_dim for m in self._marginals]))

        # Build tensor design matrix
        X_tensor = self._build_tensor_design(X_list)

        # Build tensor penalty matrices
        self._penalties = self._build_tensor_penalties(
            S_list, dims, [m.rank for m in self._marginals]
        )

        # Apply smoothCon normalization — R normalizes each penalty
        # by its own 1-norm (not all by S[0]'s norm)
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
        matrices = [m.predict_matrix(new_data) for m in self._marginals]
        for j, XP in enumerate(self._XP_list):
            if XP is not None:
                matrices[j] = matrices[j] @ XP
        return self._build_tensor_design(matrices)


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
            Constrained penalty matrix, shape ``(k-1, k-1)``.
        Z : np.ndarray
            Constraint absorption matrix, shape ``(k, k-1)``.
        """
        c = X.sum(axis=0)
        Q, _ = np.linalg.qr(c[:, np.newaxis], mode="complete")
        Z = Q[:, 1:]
        X_c = X @ Z
        S_c = Z.T @ S @ Z
        S_c = 0.5 * (S_c + S_c.T)
        return X_c, S_c, Z

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct tensor interaction basis from data.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Must contain keys matching all variables in ``self.spec.variables``.
        """
        # Create and setup marginal smooths
        marginals_info = self._create_marginals(data)
        self._marginals = [info[0] for info in marginals_info]
        X_list_raw, S_list_raw = self._get_marginal_matrices(marginals_info)

        # Absorb sum-to-zero constraints into each marginal
        X_list = []
        S_list = []
        self._Z_list = []
        for X_j, S_j in zip(X_list_raw, S_list_raw, strict=True):
            X_c, S_c, Z = self._absorb_constraint(X_j, S_j)
            X_list.append(X_c)
            S_list.append(S_c)
            self._Z_list.append(Z)

        # SVD reparameterize constrained 1D marginals without noterp
        self._XP_list = []
        for j, marginal in enumerate(self._marginals):
            if not marginal._noterp and len(marginal.spec.variables) == 1:
                result = self._svd_reparameterize(
                    marginal, X_list[j], S_list[j], data, Z=self._Z_list[j]
                )
                if result is not None:
                    X_list[j], S_list[j], XP = result
                    self._XP_list.append(XP)
                else:
                    self._XP_list.append(None)
            else:
                self._XP_list.append(None)

        constrained_dims = [X.shape[1] for X in X_list]
        self.n_coefs = int(np.prod(constrained_dims))

        # null_space_dim: each marginal's null space is reduced by 1
        constrained_nsds = [max(m.null_space_dim - 1, 0) for m in self._marginals]
        self.null_space_dim = int(np.prod(constrained_nsds)) if constrained_nsds else 0

        # Build tensor design matrix from constrained marginals
        X_tensor = self._build_tensor_design(X_list)

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
        X_list_raw = [m.predict_matrix(new_data) for m in self._marginals]
        matrices = []
        for j, (X, Z) in enumerate(zip(X_list_raw, self._Z_list, strict=True)):
            X_j = X @ Z
            XP = self._XP_list[j]
            if XP is not None:
                X_j = X_j @ XP
            matrices.append(X_j)
        return self._build_tensor_design(matrices)
