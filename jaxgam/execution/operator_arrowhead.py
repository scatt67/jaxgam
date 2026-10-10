"""Exact modest-plus-indexed normal algebra with bounded source replay.

For H = [[A, B.T], [B, D]], D is positive diagonal and the modest Schur
complement is A - B.T D^-1 B. This module never expands the indexed block.
Its floating arithmetic is conditional on the assembled frozen H; it is not
by itself a calibrated EFS score error bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve

from jaxgam.execution.operator import FrozenScanSystem


@dataclass(frozen=True)
class ExactArrowheadFactorization:
    """One checked coupling replay and O(pk + k²) Schur factorization."""

    system: FrozenScanSystem
    coupling: np.ndarray
    indexed_diagonal: np.ndarray
    schur_lower: np.ndarray
    spectral_lower_bound: float
    spectral_condition_bound: float
    peak_bound_bytes: int
    source_scans: int

    @classmethod
    def from_system(
        cls, system: FrozenScanSystem, *, memory_budget_bytes: int
    ) -> ExactArrowheadFactorization:
        if not isinstance(system, FrozenScanSystem) or not system.has_exact_arrowhead:
            raise ValueError("An exact modest-indexed arrowhead is required.")
        p = system.model.n_coef
        k = len(system.model.modest_columns)
        required = system.memory_budget_bytes + 64 * p * k + 48 * p + 128 * k**2 + 8192
        if required > memory_budget_bytes:
            raise MemoryError("Arrowhead factorization exceeds its memory budget.")
        effect = system.model.random_effects[0]
        indexed = slice(effect.col_start, effect.col_start + effect.n_coef)
        diagonal = np.asarray(system.preconditioner.inverse.diagonal, dtype=np.float64)
        d = diagonal[indexed].copy()
        if not np.all(np.isfinite(d) & (d > 0)):
            raise FloatingPointError("Arrowhead indexed diagonal is invalid.")
        block = system.preconditioner.inverse.blocks[0]
        modest_lower = np.asarray(block.lower, dtype=np.float64)
        scans_before = system.diagnostics.completed_scans
        coupling = system.exact_arrowhead_coupling(
            memory_budget_bytes=memory_budget_bytes
        )
        weighted = coupling / np.sqrt(d)[:, None]
        schur = modest_lower @ modest_lower.T - weighted.T @ weighted
        if not np.all(np.isfinite(schur)):
            raise FloatingPointError("Arrowhead Schur complement is nonfinite.")
        lower = np.linalg.cholesky((schur + schur.T) * 0.5)
        if not np.all(np.isfinite(lower)):
            raise FloatingPointError("Arrowhead Schur factor is nonfinite.")
        # For 0 < t < min(D), H-tI is SPD exactly when its modest Schur
        # complement is SPD. Bracket the smallest eigenvalue using only k-by-k
        # Cholesky tests; no p-square matrix or additional source pass exists.
        modest_matrix = modest_lower @ modest_lower.T
        low, high = 0.0, float(np.min(d))
        identity = np.eye(k, dtype=np.float64)
        for _ in range(48):
            midpoint = (low + high) * 0.5
            if midpoint in (low, high):
                break
            scaled = coupling / np.sqrt(d - midpoint)[:, None]
            shifted = modest_matrix - midpoint * identity - scaled.T @ scaled
            if not np.all(np.isfinite(shifted)):
                high = midpoint
                continue
            try:
                np.linalg.cholesky((shifted + shifted.T) * 0.5)
            except np.linalg.LinAlgError:
                high = midpoint
            else:
                low = midpoint
        spectral_lower = float(low * (1.0 - 64 * np.finfo(float).eps))
        # Weyl's inequality bounds the coupling perturbation by ||B||_F.
        spectral_upper = float(
            max(np.linalg.eigvalsh(modest_matrix)[-1], np.max(d))
            + np.linalg.norm(coupling, ord="fro")
        )
        condition = spectral_upper / spectral_lower
        if not (
            math.isfinite(spectral_lower)
            and spectral_lower > 0
            and math.isfinite(condition)
            and condition >= 1
        ):
            raise FloatingPointError("Arrowhead spectral bound is invalid.")
        d.setflags(write=False)
        lower.setflags(write=False)
        return cls(
            system,
            coupling,
            d,
            lower,
            spectral_lower,
            condition,
            required,
            system.diagnostics.completed_scans - scans_before,
        )

    @property
    def log_determinant(self) -> float:
        """Log determinant of the assembled frozen normal matrix."""
        terms = (
            *(math.log(float(value)) for value in self.indexed_diagonal),
            *(2.0 * math.log(float(value)) for value in np.diag(self.schur_lower)),
        )
        value = math.fsum(terms)
        if not math.isfinite(value):
            raise FloatingPointError("Arrowhead determinant is nonfinite.")
        return value

    def inverse_diagonal(self) -> tuple[np.ndarray, np.ndarray]:
        """Return modest and indexed inverse diagonals without p² storage."""
        k = self.schur_lower.shape[0]
        modest_inverse = cho_solve(
            (self.schur_lower, True), np.eye(k, dtype=np.float64)
        )
        weighted = self.coupling / self.indexed_diagonal[:, None]
        correction = np.einsum(
            "ij,jk,ik->i", weighted, modest_inverse, weighted, optimize=True
        )
        indexed_inverse = 1.0 / self.indexed_diagonal + correction
        if not np.all(np.isfinite(indexed_inverse) & (indexed_inverse > 0)):
            raise FloatingPointError("Arrowhead indexed inverse is invalid.")
        modest_diagonal = np.diag(modest_inverse).copy()
        if not np.all(np.isfinite(modest_diagonal) & (modest_diagonal > 0)):
            raise FloatingPointError("Arrowhead modest inverse is invalid.")
        return modest_diagonal, indexed_inverse

    def modest_inverse(self) -> np.ndarray:
        """Return the small Schur inverse for local modest penalty contractions."""
        k = self.schur_lower.shape[0]
        return cho_solve((self.schur_lower, True), np.eye(k, dtype=np.float64))
