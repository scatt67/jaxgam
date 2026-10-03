"""Exact, bounded Fisher penalty contractions for moderate-p operator EFS.

This is the deterministic reference for an eventual shared-probe trace
provider. Every supported penalty coordinate requires one frozen-system solve;
the explicit cap prevents an accidental large-p claim or unbounded source work.
The caller supplies the Fisher system in the same coefficient coordinates as
the local penalties. Score evaluation and penalty log-determinant derivatives
remain separate dependencies of the EFS controller.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from jaxgam.execution.operator import FrozenScanSystem, PCGControl
from jaxgam.penalties.structure import (
    DenseLocalPenalty,
    DiagonalPenalty,
    IdentityPenalty,
    LocalPenalty,
)

_DEFAULT_TRACE_CONTROL = PCGControl()


@dataclass(frozen=True)
class ExactPenaltyDiagnostics:
    """Actual source work and prospective owned-memory bound."""

    completed_coordinates: int
    krylov_iterations: int
    normal_actions: int
    source_scans: int
    maximum_relative_residual: float
    peak_bound_bytes: int


@dataclass(frozen=True)
class ExactPenaltyStatistics:
    """Unscaled Fisher traces and coefficient quadratics in sp order."""

    traces: np.ndarray
    quadratics: np.ndarray
    diagnostics: ExactPenaltyDiagnostics

    def __post_init__(self) -> None:
        for name in ("traces", "quadratics"):
            owned = np.array(getattr(self, name), dtype=np.float64, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)


class ExactPenaltyTraceSolveError(RuntimeError):
    """A coordinate solve failed, so no partial statistic is returned."""

    def __init__(self, coordinate: int, status: int, iterations: int) -> None:
        self.coordinate = coordinate
        self.status = status
        self.iterations = iterations
        super().__init__(
            f"Fisher trace solve failed at coordinate {coordinate}: "
            f"status {status} after {iterations} iterations."
        )


def _local_action(penalty: LocalPenalty, vector: np.ndarray) -> np.ndarray:
    if isinstance(penalty, DenseLocalPenalty):
        return penalty.matrix @ vector
    if isinstance(penalty, DiagonalPenalty):
        return penalty.diagonal * vector
    if isinstance(penalty, IdentityPenalty):
        return penalty.scale * vector
    raise TypeError("Unknown local penalty representation.")


def _local_row_contract(
    penalty: LocalPenalty, local_row: int, inverse_column: np.ndarray
) -> float:
    if isinstance(penalty, DenseLocalPenalty):
        return float(penalty.matrix[local_row] @ inverse_column)
    if isinstance(penalty, DiagonalPenalty):
        return float(penalty.diagonal[local_row] * inverse_column[local_row])
    if isinstance(penalty, IdentityPenalty):
        return float(penalty.scale * inverse_column[local_row])
    raise TypeError("Unknown local penalty representation.")


class ExactOperatorPenaltyStatistics:
    """Compose a frozen Fisher inverse with local penalty actions.

    The cap is on actual coordinate solves (including repeated block support),
    and the memory bound reserves the full operator budget plus one extra RHS,
    one returned solution, and O(m) outputs. This reference is deliberately
    restricted to moderate p; stochastic traces require their own calibrated
    accuracy and score contracts.
    """

    def __init__(
        self,
        fisher_system: FrozenScanSystem,
        *,
        memory_budget_bytes: int,
        max_trace_coordinates: int = 256,
        control: PCGControl = _DEFAULT_TRACE_CONTROL,
    ) -> None:
        if not isinstance(fisher_system, FrozenScanSystem):
            raise TypeError("A frozen Fisher operator system is required.")
        for name, value in (
            ("memory_budget_bytes", memory_budget_bytes),
            ("max_trace_coordinates", max_trace_coordinates),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(control, PCGControl):
            raise TypeError("Exact trace requires PCGControl.")
        self.system = fisher_system
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.max_trace_coordinates = int(max_trace_coordinates)
        self.control = control

    def statistics(self, coefficients: np.ndarray) -> ExactPenaltyStatistics:
        """Compute q_j and tr(H_F^-1 S_j), exposing all source passes."""
        structure = self.system.model.penalties
        p, m = structure.n_coef, structure.n_penalties
        if (
            not isinstance(coefficients, np.ndarray)
            or coefficients.shape != (p,)
            or coefficients.dtype != np.float64
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("Fisher trace coefficients must be finite float64.")
        coordinates = sum(block.size for block in structure.blocks)
        if coordinates > self.max_trace_coordinates:
            raise ValueError(
                "Exact Fisher trace coordinate cap exceeded; "
                "a calibrated large-p trace provider is required."
            )
        required = self.system.memory_budget_bytes + 24 * p + 32 * m + 4096
        if required > self.memory_budget_bytes:
            raise MemoryError(
                f"Exact Fisher traces require {required} bytes; "
                f"budget is {self.memory_budget_bytes}."
            )
        traces = np.zeros(m, dtype=np.float64)
        quadratics = np.zeros(m, dtype=np.float64)
        rhs = np.zeros(p, dtype=np.float64)
        iterations = actions = scans = completed = 0
        maximum_relative_residual = 0.0
        for block in structure.blocks:
            local_coefficients = coefficients[block.start : block.stop]
            for sp, penalty in zip(
                block.sp_indices, block.local_penalties, strict=True
            ):
                quadratics[sp] += float(
                    local_coefficients @ _local_action(penalty, local_coefficients)
                )
            for coordinate in range(block.start, block.stop):
                rhs[coordinate] = 1.0
                before_actions = self.system.diagnostics.normal_actions
                before_scans = self.system.diagnostics.completed_scans
                try:
                    solved = self.system.solve(rhs, control=self.control)
                finally:
                    rhs[coordinate] = 0.0
                action_count = self.system.diagnostics.normal_actions - before_actions
                scan_count = self.system.diagnostics.completed_scans - before_scans
                if action_count != solved.matvec_count or scan_count != action_count:
                    raise RuntimeError(
                        "Fisher trace source-pass accounting is inconsistent."
                    )
                actions += action_count
                scans += scan_count
                iterations += solved.iterations
                maximum_relative_residual = max(
                    maximum_relative_residual, solved.relative_residual
                )
                if not solved.converged:
                    raise ExactPenaltyTraceSolveError(
                        coordinate, solved.status, solved.iterations
                    )
                local_column = solved.solution[block.start : block.stop]
                local_row = coordinate - block.start
                for sp, penalty in zip(
                    block.sp_indices, block.local_penalties, strict=True
                ):
                    traces[sp] += _local_row_contract(penalty, local_row, local_column)
                completed += 1
                del solved
        if not np.all(np.isfinite(traces)) or not np.all(np.isfinite(quadratics)):
            raise FloatingPointError("Exact Fisher penalty statistics are nonfinite.")
        return ExactPenaltyStatistics(
            traces,
            quadratics,
            ExactPenaltyDiagnostics(
                completed,
                iterations,
                actions,
                scans,
                maximum_relative_residual,
                required,
            ),
        )
