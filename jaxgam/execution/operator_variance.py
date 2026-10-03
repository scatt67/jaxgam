"""Bounded selected-row Fisher variance from a frozen operator system.

The caller supplies a certified Fisher ``FrozenScanSystem`` at the selected
fit's coefficients, family parameters and smoothing parameters. A generic
``FrozenScanSystem`` does not encode whether its row curvature came from the
Fisher or observed information; this internal provider cannot establish that
semantic fact from positive definiteness alone. Public uncertainty must use
a family-aware constructor that certifies the Fisher weights. This host
component solves one prediction design row at a time; it retains no covariance
matrix, expanded prediction design or collection of coefficient solves.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from jaxgam.execution.operator import FrozenScanSystem, PCGControl
from jaxgam.formula.predict_matrix import Data
from jaxgam.inference.operator_geometry import OperatorPredictionGeometry, _rows

_DEFAULT_VARIANCE_CONTROL = PCGControl()


@dataclass(frozen=True)
class OperatorVarianceDiagnostics:
    """Completed selected-row solves and their charged source work."""

    requested_rows: int
    completed_rows: int
    krylov_iterations: int
    normal_actions: int
    source_scans: int
    maximum_relative_residual: float
    peak_bound_bytes: int


@dataclass(frozen=True)
class OperatorVarianceResult:
    """Link-scale coefficient variances for explicitly selected rows."""

    variance: np.ndarray
    diagnostics: OperatorVarianceDiagnostics

    def __post_init__(self) -> None:
        owned = np.array(self.variance, dtype=np.float64, copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "variance", owned)


class OperatorVarianceSolveError(RuntimeError):
    """A selected Fisher solve failed without returning partial variances."""

    def __init__(self, row: int, status: int, iterations: int) -> None:
        self.row = row
        self.status = status
        self.iterations = iterations
        super().__init__(
            f"Fisher variance solve failed at row {row}: "
            f"status {status} after {iterations} iterations."
        )


class SelectedOperatorVariance:
    """Compose generic prediction geometry with a frozen Fisher PCG system.

    The caller certifies that ``fisher_system`` uses Fisher row curvature;
    passing an observed-information system computes a different quadratic and
    is not a covariance estimate. ``system.memory_budget_bytes`` is reserved
    in full. The additional budget
    covers the caller-owned selected output, one coefficient RHS and bounded
    prediction geometry. The intentionally conservative split lets the
    existing PCG preflight account for compiler scratch without a hidden
    coefficient-square allocation. Only selected rows are supported; bulk
    uncertainty needs a separately validated approximation.
    """

    def __init__(
        self,
        fisher_system: FrozenScanSystem,
        *,
        scale: float,
        memory_budget_bytes: int,
        max_selected_rows: int = 256,
        control: PCGControl = _DEFAULT_VARIANCE_CONTROL,
    ) -> None:
        if not isinstance(fisher_system, FrozenScanSystem):
            raise TypeError("A frozen Fisher operator system is required.")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, numbers.Real)
            or not np.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("Fisher variance scale must be finite and positive.")
        for name, value in (
            ("memory_budget_bytes", memory_budget_bytes),
            ("max_selected_rows", max_selected_rows),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(control, PCGControl):
            raise TypeError("Selected variance requires PCGControl.")
        model = fisher_system.model
        self.geometry = OperatorPredictionGeometry(
            model.modest.predict_spec,
            model.modest_columns,
            model.random_effects,
            model.n_coef,
            model.batch_rows,
            model.modest_raw_width,
            model.metadata_bound_bytes,
        )
        self.system = fisher_system
        self.scale = float(scale)
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.max_selected_rows = int(max_selected_rows)
        self.control = control

    def variance(self, data: Data) -> OperatorVarianceResult:
        """Return selected link variances without retaining coefficient solves."""
        n_rows = _rows(data)
        if n_rows > self.max_selected_rows:
            raise ValueError(
                "Selected variance row cap exceeded; bulk uncertainty is unavailable."
            )
        required = self.system.memory_budget_bytes + self.geometry.workspace_bound(
            n_rows
        )
        if required > self.memory_budget_bytes:
            raise MemoryError(
                f"Selected variance requires {required} bytes; "
                f"budget is {self.memory_budget_bytes}."
            )
        variance = np.empty(n_rows, dtype=np.float64)
        iterations = actions = scans = completed = 0
        maximum_relative_residual = 0.0
        p = self.geometry.n_coef
        modest_columns = np.asarray(self.geometry.modest_columns, dtype=np.int64)
        for start, stop, modest, columns in self.geometry._batches(data):
            indexed = tuple(
                term.evaluate(columns) for term in self.geometry.random_effects
            )
            for row in range(stop - start):
                rhs = np.zeros(p, dtype=np.float64)
                rhs[modest_columns] = modest[row]
                for term, (ids, amplitude) in zip(
                    self.geometry.random_effects, indexed, strict=True
                ):
                    rhs[term.col_start + ids[row]] += amplitude[row]
                if not np.all(np.isfinite(rhs)):
                    raise ValueError(
                        f"Prediction design is nonfinite at row {start + row}."
                    )
                before_actions = self.system.diagnostics.normal_actions
                before_scans = self.system.diagnostics.completed_scans
                solved = self.system.solve(rhs, control=self.control)
                action_count = self.system.diagnostics.normal_actions - before_actions
                scan_count = self.system.diagnostics.completed_scans - before_scans
                if action_count != solved.matvec_count or scan_count != action_count:
                    raise RuntimeError(
                        "Fisher solve source-pass accounting is inconsistent."
                    )
                actions += action_count
                scans += scan_count
                iterations += solved.iterations
                maximum_relative_residual = max(
                    maximum_relative_residual, solved.relative_residual
                )
                if not solved.converged:
                    raise OperatorVarianceSolveError(
                        start + row, solved.status, solved.iterations
                    )
                quadratic = float(rhs @ solved.solution)
                roundoff = (
                    64
                    * np.finfo(np.float64).eps
                    * np.linalg.norm(rhs)
                    * np.linalg.norm(solved.solution)
                )
                if not np.isfinite(quadratic) or quadratic < -roundoff:
                    raise FloatingPointError(
                        f"Fisher variance is invalid at row {start + row}."
                    )
                value = self.scale * max(quadratic, 0.0)
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"Fisher variance overflowed at row {start + row}."
                    )
                variance[start + row] = value
                completed += 1
                del rhs, solved
            del indexed, modest, columns
        return OperatorVarianceResult(
            variance,
            OperatorVarianceDiagnostics(
                n_rows,
                completed,
                iterations,
                actions,
                scans,
                maximum_relative_residual,
                required,
            ),
        )
