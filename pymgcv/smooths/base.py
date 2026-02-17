"""Abstract Smooth base class for all smooth term types.

Defines the interface that every smooth (tp, ts, cr, cs, cc, tensor)
must implement. This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 5.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy import linalg

from pymgcv.formula.terms import SmoothSpec
from pymgcv.penalties.penalty import Penalty


class Smooth(ABC):
    """Abstract base class for smooth terms.

    A smooth encapsulates the basis construction, penalty construction,
    and prediction matrix for one smooth term in a GAM formula.

    Parameters
    ----------
    spec : SmoothSpec
        Smooth term specification from the formula parser.
    """

    def __init__(self, spec: SmoothSpec) -> None:
        self.spec = spec
        self.n_coefs: int = 0
        self.null_space_dim: int = 0
        self.rank: int = 0
        self._is_setup: bool = False
        self._s_scale: float = 1.0

    def _require_setup(self) -> None:
        """Raise RuntimeError if setup() hasn't been called."""
        if not self._is_setup:
            raise RuntimeError(f"Call setup() before using {type(self).__name__}.")

    @staticmethod
    def _smoothcon_normalize(
        X: npt.NDArray[np.floating],
        S_list: list[npt.NDArray[np.floating]],
    ) -> tuple[list[npt.NDArray[np.floating]], float]:
        """Replicate R's smoothCon() penalty normalization.

        Computes ``s_scale = norm(S[0], ord=1) / norm(X, ord=inf)**2``
        and divides all S matrices by the same scale.

        Parameters
        ----------
        X : np.ndarray
            Design matrix.
        S_list : list[np.ndarray]
            Penalty matrices to normalize.

        Returns
        -------
        normalized : list[np.ndarray]
            Normalized penalty matrices.
        s_scale : float
            The normalization scale factor.
        """
        norm_X_inf = np.linalg.norm(X, ord=np.inf)
        norm_S_1 = np.linalg.norm(S_list[0], ord=1)
        maXX = norm_X_inf**2
        if maXX > 0:
            s_scale = norm_S_1 / maXX
            return [S / s_scale for S in S_list], s_scale
        return list(S_list), 1.0

    def _apply_shrinkage(self, S: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Hook for shrinkage penalty modification. No-op in base."""
        return S

    # ------------------------------------------------------------------
    # Shrinkage eigendecomposition utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _chained_geometric_replacement(
        null_rank: int, smallest_nonzero: float, shrink: float
    ) -> npt.NDArray[np.floating]:
        """cs style: each successive null eigenvalue is shrink * previous.

        Produces ascending values:
        ``[smallest * shrink^null_rank, ..., smallest * shrink]``.
        """
        factors = shrink ** np.arange(null_rank, 0, -1)
        return smallest_nonzero * factors

    @staticmethod
    def _uniform_replacement(
        null_rank: int, smallest_nonzero: float, shrink: float
    ) -> npt.NDArray[np.floating]:
        """ts style: all null eigenvalues get the same replacement value."""
        return np.full(null_rank, smallest_nonzero * shrink)

    @staticmethod
    def _decompose_and_replace(
        S: npt.NDArray[np.floating],
        k: int,
        shrink: float,
        replacement_fn: Callable[[int, float, float], npt.NDArray[np.floating]],
    ) -> npt.NDArray[np.floating]:
        """Eigendecompose S, replace near-zero eigenvalues via replacement_fn.

        Uses reconstruction from modified eigenvalues rather than direct
        modification, preserving range-space eigenvalues exactly.

        Parameters
        ----------
        S : np.ndarray
            Symmetric penalty matrix, shape ``(k, k)``.
        k : int
            Basis dimension (used for tolerance computation).
        shrink : float
            Shrinkage factor passed to ``replacement_fn``.
        replacement_fn : callable
            ``(null_rank, smallest_nonzero, shrink) -> np.ndarray`` of
            replacement eigenvalues for the null space.

        Returns
        -------
        np.ndarray
            Modified penalty matrix with near-zero eigenvalues replaced.
        """
        # Use driver='evr' (dsyevr) to match R's eigen(symmetric=TRUE)
        eigvals, eigvecs = linalg.eigh(S, driver="evr")

        tol = np.max(np.abs(eigvals)) * k * np.finfo(float).eps
        nonzero_mask = np.abs(eigvals) > tol
        null_rank = int(np.sum(~nonzero_mask))

        if null_rank > 0:
            if np.any(nonzero_mask):
                smallest_nonzero = np.min(np.abs(eigvals[nonzero_mask]))
            else:
                smallest_nonzero = 1.0
            eigvals[:null_rank] = replacement_fn(null_rank, smallest_nonzero, shrink)

        S_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return 0.5 * (S_new + S_new.T)

    @abstractmethod
    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct the smooth basis from data.

        This is the main construction method. After calling setup(),
        the smooth is ready to produce design and penalty matrices.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Mapping from variable names to data arrays. Must contain
            all variables referenced by ``self.spec.variables``.
        """

    @abstractmethod
    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build the design matrix for the given data.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Mapping from variable names to data arrays.

        Returns
        -------
        np.ndarray
            Design matrix, shape ``(n, n_coefs)``.
        """

    @abstractmethod
    def build_penalty_matrices(self) -> list[Penalty]:
        """Build the penalty matrices for this smooth.

        Returns
        -------
        list[Penalty]
            One or more penalty matrices. Most smooths have one;
            tensor products have one per marginal.
        """

    @abstractmethod
    def predict_matrix(
        self, new_data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build the prediction matrix for new data.

        Parameters
        ----------
        new_data : dict[str, np.ndarray]
            Mapping from variable names to new data arrays.

        Returns
        -------
        np.ndarray
            Prediction matrix, shape ``(n_new, n_coefs)``.
        """
