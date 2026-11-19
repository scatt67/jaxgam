"""Bounded local penalty pseudo-determinant for operator score components.

The active range is selected from unscaled local penalties, as in the dense
EFS metadata. Large indexed identity or diagonal terms remain symbolic. This
is only the ``log|S|_+`` part of a matched-state REML score.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    DiagonalPenalty,
    IdentityPenalty,
    LocalPenalty,
    PenaltyStructure,
)

_RANK_TOLERANCE = np.finfo(np.float64).eps ** (2.0 / 3.0)
_LOG_FLOOR = 1e-30


@dataclass(frozen=True)
class OperatorPenaltyLogDet:
    """Penalty pseudo-determinant and log-sp gradient in fitted coordinates."""

    value: float
    rank: int
    null_dimension: int
    block_values: np.ndarray
    peak_bound_bytes: int
    derivative: np.ndarray | None = None

    def __post_init__(self) -> None:
        owned = np.array(self.block_values, dtype=np.float64, copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "block_values", owned)
        if self.derivative is not None:
            derivative = np.array(self.derivative, dtype=np.float64, copy=True)
            derivative.setflags(write=False)
            object.__setattr__(self, "derivative", derivative)


def _penalty_diagonal(penalty: LocalPenalty) -> np.ndarray:
    if isinstance(penalty, IdentityPenalty):
        return np.full(penalty.size, penalty.scale, dtype=np.float64)
    if isinstance(penalty, DiagonalPenalty):
        return penalty.diagonal
    raise TypeError("A dense penalty cannot use diagonal-only storage.")


def _singleton(penalty: LocalPenalty, rho: float) -> tuple[float, int, float]:
    if isinstance(penalty, IdentityPenalty):
        if not np.isfinite(penalty.scale) or penalty.scale < 0:
            raise ValueError("Local penalty must be positive semidefinite.")
        if penalty.scale == 0:
            return 0.0, 0, 0.0
        value = penalty.size * (rho + float(np.log(max(penalty.scale, _LOG_FLOOR))))
        return value, penalty.size, float(penalty.size)
    if isinstance(penalty, DiagonalPenalty):
        eigenvalues = penalty.diagonal
    else:
        eigenvalues = np.linalg.eigvalsh(penalty.matrix)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Local penalty has nonfinite eigenvalues.")
    maximum = float(np.max(np.abs(eigenvalues), initial=0.0))
    if np.min(eigenvalues, initial=0.0) < -maximum * _RANK_TOLERANCE:
        raise ValueError("Local penalty must be positive semidefinite.")
    active = eigenvalues > maximum * _RANK_TOLERANCE
    rank = int(np.count_nonzero(active))
    return (
        rank * rho + float(np.sum(np.log(np.maximum(eigenvalues[active], _LOG_FLOOR)))),
        rank,
        float(rank),
    )


def _multidiagonal(
    penalties: tuple[LocalPenalty, ...], rho: np.ndarray
) -> tuple[float, int, np.ndarray]:
    size = penalties[0].size
    range_weight = np.zeros(size, dtype=np.float64)
    log_scales = []
    maxima = []
    for penalty in penalties:
        diagonal = _penalty_diagonal(penalty)
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal < 0):
            raise ValueError("Local diagonal penalty must be nonnegative finite.")
        maximum = float(np.max(diagonal, initial=0.0))
        scaled = diagonal / maximum if maximum else np.zeros(size)
        norm = float(np.linalg.norm(scaled))
        if norm:
            range_weight += scaled / norm
        log_scales.append(np.log(maximum) if maximum else -np.inf)
        maxima.append(maximum)
    maximum = float(np.max(range_weight, initial=0.0))
    active = range_weight > maximum * _RANK_TOLERANCE
    rank = int(np.count_nonzero(active))
    if not rank:
        return 0.0, 0, np.zeros(len(penalties), dtype=np.float64)
    offsets = rho + np.asarray(log_scales)
    offset = float(np.max(offsets))
    combined = np.zeros(rank, dtype=np.float64)
    for penalty, weight, maximum in zip(penalties, offsets, maxima, strict=True):
        if np.isfinite(weight):
            diagonal = _penalty_diagonal(penalty)
            combined += np.exp(weight - offset) * (diagonal[active] / maximum)
    if np.any(combined <= 0) or not np.all(np.isfinite(combined)):
        raise np.linalg.LinAlgError("Combined penalty lost its fixed range.")
    derivative = np.zeros(len(penalties), dtype=np.float64)
    for index, (penalty, weight, maximum) in enumerate(
        zip(penalties, offsets, maxima, strict=True)
    ):
        if np.isfinite(weight):
            diagonal = _penalty_diagonal(penalty)
            derivative[index] = float(
                np.sum(
                    np.exp(weight - offset) * (diagonal[active] / maximum) / combined
                )
            )
    return rank * offset + float(np.sum(np.log(combined))), rank, derivative


def _multidense(
    penalties: tuple[LocalPenalty, ...], rho: np.ndarray
) -> tuple[float, int, np.ndarray]:
    size = penalties[0].size
    normalized = np.zeros((size, size), dtype=np.float64)
    matrices = []
    log_scales = []
    for penalty in penalties:
        matrix = penalty.dense()
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Local penalty must be finite.")
        scale = float(np.max(np.abs(matrix), initial=0.0))
        scaled = matrix / scale if scale else np.zeros_like(matrix)
        eigenvalues = np.linalg.eigvalsh(scaled)
        maximum = float(np.max(np.abs(eigenvalues), initial=0.0))
        if np.min(eigenvalues, initial=0.0) < -maximum * _RANK_TOLERANCE:
            raise ValueError("Local penalty must be positive semidefinite.")
        norm = float(np.linalg.norm(scaled, "fro"))
        if norm:
            normalized += scaled / norm
        matrices.append(scaled)
        log_scales.append(np.log(scale) if scale else -np.inf)
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    maximum = float(np.max(eigenvalues, initial=0.0))
    active = eigenvalues > maximum * _RANK_TOLERANCE
    rank = int(np.count_nonzero(active))
    if not rank:
        return 0.0, 0, np.zeros(len(penalties), dtype=np.float64)
    range_basis = eigenvectors[:, active]
    offsets = rho + np.asarray(log_scales)
    offset = float(np.max(offsets))
    combined = np.zeros((rank, rank), dtype=np.float64)
    contributions = []
    for matrix, weight in zip(matrices, offsets, strict=True):
        if np.isfinite(weight):
            contribution = np.exp(weight - offset) * (
                range_basis.T @ matrix @ range_basis
            )
            combined += contribution
            contributions.append(contribution)
        else:
            contributions.append(np.zeros((rank, rank), dtype=np.float64))
    sign, logabs = np.linalg.slogdet(combined)
    if sign <= 0 or not np.isfinite(logabs):
        raise np.linalg.LinAlgError("Combined penalty lost its fixed range.")
    derivative = np.array(
        [float(np.trace(np.linalg.solve(combined, term))) for term in contributions]
    )
    return rank * offset + float(logabs), rank, derivative


class LocalOperatorPenaltyDeterminant:
    """Compute ``log|S|_+`` without a global coefficient-square allocation."""

    def __init__(
        self,
        structure: PenaltyStructure,
        *,
        memory_budget_bytes: int,
        max_dense_block_size: int = 64,
    ) -> None:
        if not isinstance(structure, PenaltyStructure):
            raise TypeError("A local penalty structure is required.")
        for name, value in (
            ("memory_budget_bytes", memory_budget_bytes),
            ("max_dense_block_size", max_dense_block_size),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        self.structure = structure
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.max_dense_block_size = int(max_dense_block_size)

    def _peak_bound(self) -> int:
        largest = 0
        for block in self.structure.blocks:
            size = block.size
            if len(block.local_penalties) == 1 and isinstance(
                block.local_penalties[0], IdentityPenalty
            ):
                largest = max(largest, 512)
                continue
            has_dense = any(
                isinstance(penalty, DenseLocalPenalty)
                for penalty in block.local_penalties
            )
            if has_dense and size > self.max_dense_block_size:
                raise MemoryError("Dense local penalty exceeds its hard size cap.")
            entries = (
                (2 * len(block.local_penalties) + 24) * size**2 + 64 * size
                if has_dense
                else 48 * size + 32 * len(block.local_penalties)
            )
            largest = max(largest, 8 * entries)
        return (
            largest
            + 32 * len(self.structure.blocks)
            + 16 * self.structure.n_penalties
            + 8192
        )

    def evaluate(self, log_lambda: np.ndarray) -> OperatorPenaltyLogDet:
        """Evaluate fixed-range local pseudo-determinants at supplied rho."""
        m = self.structure.n_penalties
        if (
            not isinstance(log_lambda, np.ndarray)
            or log_lambda.dtype != np.float64
            or log_lambda.shape != (m,)
            or not np.all(np.isfinite(log_lambda))
        ):
            raise ValueError("Penalty rho must be finite float64 in sp order.")
        required = self._peak_bound()
        if required > self.memory_budget_bytes:
            raise MemoryError(
                f"Local penalty determinant requires {required} bytes; "
                f"budget is {self.memory_budget_bytes}."
            )
        values = np.empty(len(self.structure.blocks), dtype=np.float64)
        derivative = np.zeros(m, dtype=np.float64)
        rank = 0
        for index, block in enumerate(self.structure.blocks):
            penalties = block.local_penalties
            weights = log_lambda[list(block.sp_indices)]
            if len(penalties) == 1:
                value, local_rank, local_derivative = _singleton(
                    penalties[0], float(weights[0])
                )
            elif all(
                isinstance(penalty, (IdentityPenalty, DiagonalPenalty))
                for penalty in penalties
            ):
                value, local_rank, local_derivative = _multidiagonal(penalties, weights)
            else:
                value, local_rank, local_derivative = _multidense(penalties, weights)
            if not np.isfinite(value):
                raise FloatingPointError("Local penalty determinant is nonfinite.")
            values[index] = value
            derivative[list(block.sp_indices)] += local_derivative
            rank += local_rank
        with np.errstate(over="ignore"):
            total = float(np.sum(values))
        if not np.isfinite(total):
            raise FloatingPointError("Total local penalty determinant is nonfinite.")
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError(
                "Local penalty determinant derivative is nonfinite."
            )
        return OperatorPenaltyLogDet(
            value=total,
            rank=rank,
            null_dimension=self.structure.n_coef - rank,
            block_values=values,
            peak_bound_bytes=required,
            derivative=derivative,
        )
