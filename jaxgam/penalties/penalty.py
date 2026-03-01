"""Penalty and CompositePenalty classes for smoothing.

Penalty matrices encode smoothness priors on basis coefficients. Each smooth
term has one or more penalty matrices S_j (symmetric positive semi-definite),
and the combined weighted penalty is S_lambda = sum_j exp(log_lambda_j) * S_j.

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 10.2 (structured penalties)
R source reference: R/smooth.r smooth.construct.*.smooth.spec()$S
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Tolerances for symmetry validation, matching the project's STRICT tier.
_SYMMETRY_RTOL = 1e-10
_SYMMETRY_ATOL = 1e-12


class Penalty:
    """A single penalty matrix with rank and null space information.

    Stores a dense symmetric positive semi-definite penalty matrix S,
    its rank, and the dimension of its null space. This corresponds to
    one penalty term from a smooth (most smooths have one penalty, but
    tensor products have one per marginal).

    Parameters
    ----------
    S : np.ndarray
        Penalty matrix, shape (k, k). Should be symmetric and positive
        semi-definite. Symmetry is validated; PSD is assumed but not
        checked (the eigendecomposition for rank would catch gross
        violations).
    rank : int or None
        Rank of the penalty matrix. If None, computed from eigenvalues.
    null_space_dim : int or None
        Dimension of the null space (k - rank). If None, computed from rank.

    Raises
    ------
    ValueError
        If S is not 2D, not square, or not symmetric.
    """

    def __init__(
        self,
        S: npt.NDArray[np.floating],
        rank: int | None = None,
        null_space_dim: int | None = None,
    ) -> None:
        S = np.asarray(S, dtype=float)

        # Validate shape
        if S.ndim != 2:
            raise ValueError(
                f"Penalty matrix must be 2D, got {S.ndim}D array with shape {S.shape}."
            )
        if S.shape[0] != S.shape[1]:
            raise ValueError(f"Penalty matrix must be square, got shape {S.shape}.")

        # Validate symmetry
        if not np.allclose(S, S.T, rtol=_SYMMETRY_RTOL, atol=_SYMMETRY_ATOL):
            raise ValueError(
                "Penalty matrix must be symmetric. "
                f"Max asymmetry: {np.max(np.abs(S - S.T)):.2e}."
            )

        # Force exact symmetry (average out floating-point asymmetry)
        self.S: npt.NDArray[np.floating] = 0.5 * (S + S.T)
        k = self.S.shape[0]

        # Compute rank from eigenvalues if not provided
        if rank is None:
            eigvals = np.linalg.eigvalsh(self.S)
            max_eigval = np.max(np.abs(eigvals))
            if max_eigval > 0:
                # Standard numerical rank threshold (cf. np.linalg.matrix_rank)
                tol = max_eigval * max(k, 1) * np.finfo(float).eps
                rank = int(np.sum(eigvals > tol))
            else:
                rank = 0

        self.rank: int = rank

        # Compute null space dimension
        if null_space_dim is None:
            null_space_dim = k - self.rank

        self.null_space_dim: int = null_space_dim

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the penalty matrix."""
        return self.S.shape

    @property
    def size(self) -> int:
        """Number of coefficients (k) the penalty applies to."""
        return self.S.shape[0]

    def __repr__(self) -> str:
        return (
            f"Penalty(shape={self.shape}, rank={self.rank}, "
            f"null_space_dim={self.null_space_dim})"
        )


class CompositePenalty:
    """Collection of penalty matrices with smoothing parameters.

    Manages multiple Penalty objects, each corresponding to one penalty
    term in the model. Provides methods to compute the weighted sum
    S_lambda = sum_j exp(log_lambda_j) * S_j and to embed per-smooth
    penalty matrices into the global coefficient space.

    Parameters
    ----------
    penalties : list[Penalty]
        List of Penalty objects.
    log_smoothing_params : np.ndarray or None
        Initial log smoothing parameters, shape (n_penalties,).
        If None, defaults to zeros (lambda_j = 1 for all j).
    """

    def __init__(
        self,
        penalties: list[Penalty],
        log_smoothing_params: npt.NDArray[np.floating] | None = None,
    ) -> None:
        if not penalties:
            raise ValueError("CompositePenalty requires at least one Penalty.")

        expected_shape = penalties[0].shape
        for i, p in enumerate(penalties[1:], 1):
            if p.shape != expected_shape:
                raise ValueError(
                    f"All penalties must have the same shape. "
                    f"Penalty 0 has shape {expected_shape}, "
                    f"but penalty {i} has shape {p.shape}."
                )

        self.penalties: list[Penalty] = list(penalties)

        if log_smoothing_params is None:
            self.log_smoothing_params: npt.NDArray[np.floating] = np.zeros(
                len(penalties)
            )
        else:
            log_smoothing_params = np.asarray(log_smoothing_params, dtype=float)
            if log_smoothing_params.shape != (len(penalties),):
                raise ValueError(
                    f"log_smoothing_params must have shape ({len(penalties)},), "
                    f"got {log_smoothing_params.shape}."
                )
            self.log_smoothing_params = log_smoothing_params

    @property
    def n_penalties(self) -> int:
        """Number of penalty matrices."""
        return len(self.penalties)

    def weighted_penalty(
        self,
        log_lambda: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute the combined weighted penalty matrix.

        Computes S_lambda = sum_j exp(log_lambda_j) * S_j.

        Parameters
        ----------
        log_lambda : np.ndarray or None
            Log smoothing parameters, shape (n_penalties,). If None,
            uses the stored log_smoothing_params.

        Returns
        -------
        np.ndarray
            Combined weighted penalty matrix, same shape as each S_j.

        Raises
        ------
        ValueError
            If log_lambda has wrong shape or penalty matrices have
            incompatible shapes.
        """
        if log_lambda is None:
            log_lambda = self.log_smoothing_params
        else:
            log_lambda = np.asarray(log_lambda, dtype=float)

        if log_lambda.shape != (self.n_penalties,):
            raise ValueError(
                f"log_lambda must have shape ({self.n_penalties},), "
                f"got {log_lambda.shape}."
            )

        # Compute weighted sum
        result = np.zeros_like(self.penalties[0].S)
        for j, penalty in enumerate(self.penalties):
            lambda_j = np.exp(log_lambda[j])
            result = result + lambda_j * penalty.S
        return result

    @staticmethod
    def embed(
        S_j: npt.NDArray[np.floating],
        col_start: int,
        total_p: int,
    ) -> npt.NDArray[np.floating]:
        """Embed a per-smooth penalty into the global coefficient space.

        Takes a small (k x k) penalty matrix and places it into the
        correct block of a (total_p x total_p) zero matrix.

        Parameters
        ----------
        S_j : np.ndarray
            Per-smooth penalty matrix, shape (k, k).
        col_start : int
            Starting column index for this smooth's coefficients in the
            global model matrix.
        total_p : int
            Total number of parameters in the model.

        Returns
        -------
        np.ndarray
            Embedded penalty matrix, shape (total_p, total_p), with
            S_j placed in the block [col_start:col_start+k, col_start:col_start+k]
            and zeros elsewhere.

        Raises
        ------
        ValueError
            If S_j does not fit within the global matrix or col_start is negative.
        """
        S_j = np.asarray(S_j, dtype=float)
        k = S_j.shape[0]

        if S_j.ndim != 2 or S_j.shape[0] != S_j.shape[1]:
            raise ValueError(f"S_j must be a square 2D matrix, got shape {S_j.shape}.")
        if col_start < 0:
            raise ValueError(f"col_start must be non-negative, got {col_start}.")
        if col_start + k > total_p:
            raise ValueError(
                f"Penalty block (col_start={col_start}, k={k}) extends beyond "
                f"total_p={total_p}."
            )

        S_global = np.zeros((total_p, total_p), dtype=float)
        S_global[col_start : col_start + k, col_start : col_start + k] = S_j
        return S_global

    def __repr__(self) -> str:
        shapes = [p.shape for p in self.penalties]
        return f"CompositePenalty(n_penalties={self.n_penalties}, shapes={shapes})"
