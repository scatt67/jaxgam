"""Exact, bounded Fisher penalty contractions for moderate-p operator EFS.

This is the deterministic reference for an eventual shared-probe trace
provider. Every supported penalty coordinate requires one frozen-system solve;
the explicit cap prevents an accidental large-p claim or unbounded source work.
The caller supplies the Fisher system in the same coefficient coordinates as
the local penalties. Score evaluation and penalty log-determinant derivatives
remain separate dependencies of the EFS controller.
"""

from __future__ import annotations

import hashlib
import math
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
    strategy: str = "coordinate_solve"


@dataclass(frozen=True)
class ExactPenaltyStatistics:
    """Unscaled Fisher traces and coefficient quadratics in sp order."""

    traces: np.ndarray
    quadratics: np.ndarray
    diagnostics: ExactPenaltyDiagnostics
    trace_rounding_bound: np.ndarray | None = None
    quadratic_rounding_bound: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("traces", "quadratics"):
            owned = np.array(getattr(self, name), dtype=np.float64, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
        for name in ("trace_rounding_bound", "quadratic_rounding_bound"):
            values = getattr(self, name)
            if values is not None:
                owned = np.array(values, dtype=np.float64, copy=True)
                if owned.shape != self.traces.shape or not np.all(
                    np.isfinite(owned) & (owned >= 0)
                ):
                    raise ValueError("Structured statistic bounds are invalid.")
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
        if self.system.has_exact_diagonal:
            return self._diagonal_statistics(coefficients)
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

    def _diagonal_statistics(self, coefficients: np.ndarray) -> ExactPenaltyStatistics:
        """Contract a single indexed block without coefficient solves or p² work."""
        structure = self.system.model.penalties
        p, m = structure.n_coef, structure.n_penalties
        required = self.system.memory_budget_bytes + 32 * p + 32 * m + 4096
        if required > self.memory_budget_bytes:
            raise MemoryError("Structured diagonal Fisher trace exceeds its budget.")
        diagonal = self.system.exact_diagonal()
        traces = np.zeros(m, dtype=np.float64)
        quadratics = np.zeros(m, dtype=np.float64)
        trace_bounds = np.zeros(m, dtype=np.float64)
        quadratic_bounds = np.zeros(m, dtype=np.float64)
        for block in structure.blocks:
            local_coefficients = coefficients[block.start : block.stop]
            for sp, penalty in zip(
                block.sp_indices, block.local_penalties, strict=True
            ):
                if not isinstance(penalty, IdentityPenalty):
                    raise ValueError("Structured indexed penalty must be identity.")
                trace_terms = penalty.scale / diagonal[block.start : block.stop]
                quadratic_terms = penalty.scale * local_coefficients**2
                trace = math.fsum(float(value) for value in trace_terms)
                quadratic = math.fsum(float(value) for value in quadratic_terms)
                traces[sp] += trace
                quadratics[sp] += quadratic
                trace_bounds[sp] += math.fsum(
                    math.ulp(value) for value in trace_terms
                ) + math.ulp(trace)
                quadratic_bounds[sp] += math.fsum(
                    4 * math.ulp(value) for value in quadratic_terms
                ) + math.ulp(quadratic)
        if not np.all(np.isfinite(traces)) or not np.all(np.isfinite(quadratics)):
            raise FloatingPointError("Structured Fisher statistics are nonfinite.")
        return ExactPenaltyStatistics(
            traces,
            quadratics,
            ExactPenaltyDiagnostics(0, 0, 0, 0, 0.0, required, "structured_diagonal"),
            trace_bounds,
            quadratic_bounds,
        )


@dataclass(frozen=True)
class StochasticPenaltyDiagnostics:
    """Separate sampling error from PCG residuals and actual source work."""

    probes: int
    seed: int
    probe_fingerprint: str
    krylov_iterations: int
    normal_actions: int
    source_scans: int
    maximum_relative_residual: float
    peak_bound_bytes: int


@dataclass(frozen=True)
class StochasticPenaltyStatistics:
    """Shared-probe Fisher traces, exact local q, and per-penalty sampling SE."""

    traces: np.ndarray
    quadratics: np.ndarray
    per_probe: np.ndarray
    monte_carlo_standard_error: np.ndarray
    diagnostics: StochasticPenaltyDiagnostics

    def __post_init__(self) -> None:
        for name in ("traces", "quadratics", "per_probe", "monte_carlo_standard_error"):
            owned = np.array(getattr(self, name), dtype=np.float64, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)


class StochasticPenaltyTraceSolveError(RuntimeError):
    """One probe solve failed; no partial Fisher trace is returned."""

    def __init__(self, probe: int, status: int, iterations: int) -> None:
        self.probe = probe
        self.status = status
        self.iterations = iterations
        super().__init__(
            f"Fisher trace probe {probe} failed: status {status} "
            f"after {iterations} iterations."
        )


class StochasticOperatorPenaltyStatistics:
    """Solve one shared Rademacher RHS per probe, regardless of penalty count.

    The supplied frozen system must contain Fisher row curvature at the same
    beta/theta/rho as the local penalties. Its generic type alone cannot prove
    that semantic condition. Empirical SE never certifies PCG bias, so a later
    controller must calibrate both before using this estimate in an EFS ratio.
    """

    def __init__(
        self,
        fisher_system: FrozenScanSystem,
        *,
        memory_budget_bytes: int,
        probes: int = 32,
        seed: int = 0,
        control: PCGControl = _DEFAULT_TRACE_CONTROL,
    ) -> None:
        if not isinstance(fisher_system, FrozenScanSystem):
            raise TypeError("A frozen Fisher operator system is required.")
        for name, value in (
            ("memory_budget_bytes", memory_budget_bytes),
            ("probes", probes),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if probes > 256:
            raise ValueError("Fisher trace probes may not exceed 256.")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
            raise ValueError("Fisher trace seed must be a nonnegative integer.")
        if not isinstance(control, PCGControl):
            raise TypeError("Stochastic trace requires PCGControl.")
        self.system = fisher_system
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.probes = int(probes)
        self.seed = int(seed)
        self.control = control

    def statistics(self, coefficients: np.ndarray) -> StochasticPenaltyStatistics:
        """Return unclamped t_j samples and q_j with bounded retained arrays."""
        structure = self.system.model.penalties
        p, m = structure.n_coef, structure.n_penalties
        if (
            not isinstance(coefficients, np.ndarray)
            or coefficients.shape != (p,)
            or coefficients.dtype != np.float64
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("Fisher trace coefficients must be finite float64.")
        required = (
            self.system.memory_budget_bytes
            + 32 * p
            + 32 * m
            + 32 * self.probes * m
            + 8192
        )
        if required > self.memory_budget_bytes:
            raise MemoryError(
                f"Stochastic Fisher traces require {required} bytes; "
                f"budget is {self.memory_budget_bytes}."
            )
        quadratics = np.zeros(m, dtype=np.float64)
        for block in structure.blocks:
            local_beta = coefficients[block.start : block.stop]
            for sp, penalty in zip(
                block.sp_indices, block.local_penalties, strict=True
            ):
                quadratics[sp] += float(local_beta @ _local_action(penalty, local_beta))
        samples = np.empty((self.probes, m), dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        digest = hashlib.sha256()
        digest.update(np.asarray([p, self.probes], dtype="<i8").tobytes())
        iterations = actions = scans = 0
        maximum_relative_residual = 0.0
        for probe in range(self.probes):
            signs = rng.integers(0, 2, size=p, dtype=np.int8)
            digest.update(signs.tobytes())
            rhs = 2 * signs.astype(np.float64) - 1
            before_actions = self.system.diagnostics.normal_actions
            before_scans = self.system.diagnostics.completed_scans
            solved = self.system.solve(rhs, control=self.control)
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
                raise StochasticPenaltyTraceSolveError(
                    probe, solved.status, solved.iterations
                )
            samples[probe] = 0.0
            for block in structure.blocks:
                local_rhs = rhs[block.start : block.stop]
                local_solution = solved.solution[block.start : block.stop]
                for sp, penalty in zip(
                    block.sp_indices, block.local_penalties, strict=True
                ):
                    samples[probe, sp] += float(
                        local_rhs @ _local_action(penalty, local_solution)
                    )
            del signs, rhs, solved
        traces = np.mean(samples, axis=0)
        standard_error = (
            np.std(samples, axis=0, ddof=1) / math.sqrt(self.probes)
            if self.probes > 1
            else np.full(m, np.inf)
        )
        if not np.all(np.isfinite(traces)) or not np.all(np.isfinite(quadratics)):
            raise FloatingPointError(
                "Stochastic Fisher penalty statistics are nonfinite."
            )
        return StochasticPenaltyStatistics(
            traces,
            quadratics,
            samples,
            standard_error,
            StochasticPenaltyDiagnostics(
                self.probes,
                self.seed,
                digest.hexdigest(),
                iterations,
                actions,
                scans,
                maximum_relative_residual,
                required,
            ),
        )
