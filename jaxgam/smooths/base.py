"""Abstract Smooth base class for all smooth term types.

Defines the interface that every smooth (tp, ts, cr, cs, cc, tensor)
must implement. This module is Phase 1 (NumPy only, no JAX imports).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import linalg

from jaxgam.formula.terms import SmoothSpec
from jaxgam.penalties.penalty import Penalty


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
        # Controls whether SVD reparameterization is skipped in tensor
        # product construction (tensor.py).  False → reparameterize (default
        # for TPRS); True → skip (cubic splines, where the basis is already
        # well-conditioned).  Name mirrors R's mgcv internal flag.
        self._noterp: bool = False
        self.side_constrain: bool = True

    def _require_setup(self) -> None:
        """Raise RuntimeError if setup() hasn't been called."""
        if not self._is_setup:
            raise RuntimeError(f"Call setup() before using {type(self).__name__}.")

    def copy_for_prediction(self) -> Smooth:
        """Return a shallow predict-only copy with fitting caches removed.

        The prediction transforms (knots, constraint matrices, levels, and
        reparameterizations) remain shared by reference.  This default is valid
        only when a smooth's non-predict caches are exactly ``_X`` and ``_S``
        and it owns no nested smooths.  Smooths with additional caches or nested
        smooths must override this method.
        """
        clone = copy.copy(self)
        for attr in ("_X", "_S"):
            if getattr(clone, attr, None) is not None:
                setattr(clone, attr, None)
        return clone

    def copy_for_training(self) -> Smooth:
        """Return a predict-only copy that can still evaluate training rows.

        Bounded row-replay setup evaluates the training source batch by batch.
        Smooths whose training rows equal their prediction rows need nothing
        beyond :meth:`copy_for_prediction`.
        """
        return self.copy_for_prediction()

    def training_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Evaluate rows of the setup source in the training parameterization.

        mgcv's model matrix for most bases is their prediction matrix at the
        data. Bases whose constructor builds training rows differently
        (thin-plate knot rows) override this; it is never a prediction route.
        """
        return self.predict_matrix(data)

    def training_design_matrix(self) -> npt.NDArray[np.floating]:
        """Return the basis constructed during :meth:`setup`.

        Setup owns construction of the training basis, matching mgcv's
        ``smoothCon`` contract. Model assembly consumes this explicit handoff
        instead of asking ``build_design_matrix`` to infer whether supplied
        data happen to be the training rows. Prediction callers must continue
        to use ``build_design_matrix`` or ``predict_matrix``.
        """
        self._require_setup()
        X = getattr(self, "_X", None)
        if X is None:
            raise RuntimeError(
                f"{type(self).__name__} did not retain its training basis."
            )
        return X

    def consume_training_design_matrix(self) -> npt.NDArray[np.floating]:
        """Transfer ownership of the setup-built matrix to model assembly.

        The dense training matrix belongs to ``ModelSetup`` after Phase 1;
        prediction retains only basis transforms. Clearing the fit-only cache
        prevents tensor and wrapper smooths from retaining a second n-by-k
        matrix after assembly.
        """
        X = self.training_design_matrix()
        self._X = None
        return X

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
        return Smooth._smoothcon_normalize_from_norm(
            np.linalg.norm(X, ord=np.inf), S_list
        )

    @staticmethod
    def _smoothcon_normalize_from_norm(
        norm_X_inf: float,
        S_list: list[npt.NDArray[np.floating]],
    ) -> tuple[list[npt.NDArray[np.floating]], float]:
        """Apply smoothCon's normalization given ``norm(X, "I")``.

        Streamed setup supplies the maximum absolute row sum of ``X`` so that
        it shares this normalization with dense setup.
        """
        norm_S_1 = np.linalg.norm(S_list[0], ord=1)
        # R variable: maXX (mgcv smoothCon)
        max_x_sq = norm_X_inf**2
        if max_x_sq > 0:
            s_scale = norm_S_1 / max_x_sq
            return [S / s_scale for S in S_list], s_scale
        return list(S_list), 1.0

    @staticmethod
    def _source_smoothcon_normalize(
        X: npt.NDArray[np.floating],
        S_list: list[npt.NDArray[np.floating]],
    ) -> tuple[list[npt.NDArray[np.floating]], float]:
        """smoothCon's penalty scaling with R's ``norm`` arithmetic.

        ``maXX <- norm(X, "I")^2`` and ``maS <- norm(S[[i]])/maXX`` call
        LAPACK ``dlange``, whose reference loops sum absolute values in a
        fixed order: across columns for each row (``"I"``) and down rows for
        each column (``"O"``).  Each penalty is divided by its own ``maS``;
        the returned scale is the first penalty's, as in
        :meth:`_smoothcon_normalize`.
        """
        return Smooth._source_smoothcon_normalize_from_norm(
            Smooth._source_max_abs_row_sum(X), S_list
        )

    @staticmethod
    def _source_max_abs_row_sum(X: npt.NDArray[np.floating]) -> float:
        """``dlange("I")``: each row's absolute sum taken column by column.

        Rows are independent, so the maximum over streamed row blocks equals
        the whole matrix's value.
        """
        X = np.asarray(X, dtype=np.float64)
        row_sums = np.zeros(X.shape[0], dtype=np.float64)
        for column in range(X.shape[1]):
            row_sums += np.abs(X[:, column])
        return float(row_sums.max(initial=0.0))

    @staticmethod
    def _source_smoothcon_normalize_from_norm(
        norm_x: float,
        S_list: list[npt.NDArray[np.floating]],
    ) -> tuple[list[npt.NDArray[np.floating]], float]:
        """Scale penalties given ``dlange("I")`` of X, in dlange's order.

        ``maXX`` is ``R_pow(norm, 2)``, i.e. ``x * x``, and each ``maS`` takes
        ``dlange("O")``: every column's absolute sum down its rows.
        """
        max_x_sq = norm_x * norm_x
        if not max_x_sq > 0.0:
            return list(S_list), 1.0
        normalized = []
        scales = []
        for S in S_list:
            S = np.asarray(S, dtype=np.float64)
            column_sums = np.zeros(S.shape[1], dtype=np.float64)
            for row in range(S.shape[0]):
                column_sums += np.abs(S[row])
            scale = float(column_sums.max(initial=0.0)) / max_x_sq
            normalized.append(S / scale)
            scales.append(scale)
        return normalized, scales[0]

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
