"""Matched-state operator EFS components from one frozen family/source state.

This host provider composes observed score, Fisher trace, and local penalty
statistics. It does not refit coefficients or certify Monte Carlo/PCG errors
for an accepted-versus-trial controller decision. The caller must supply a
converged coefficient state and preserve the incoming EFS score scale.
"""

from __future__ import annotations

import hashlib
import numbers
from dataclasses import dataclass
from typing import Literal

import numpy as np

from jaxgam.data.source import RowSource
from jaxgam.execution.operator import FrozenScanSystem, PCGControl
from jaxgam.execution.operator_efs_trace import (
    ExactOperatorPenaltyStatistics,
    ExactPenaltyStatistics,
    StochasticOperatorPenaltyStatistics,
    StochasticPenaltyStatistics,
)
from jaxgam.execution.operator_fit import (
    FrozenOperatorWorkingState,
    OperatorFitResult,
    _operator_family_row_bound,
)
from jaxgam.execution.operator_logdet import (
    FrozenOperatorLogDet,
    OperatorLogDetControl,
)
from jaxgam.execution.operator_penalty_determinant import (
    LocalOperatorPenaltyDeterminant,
    OperatorPenaltyLogDet,
)
from jaxgam.execution.operator_score import (
    OperatorScoreEstimate,
    assemble_operator_score,
)
from jaxgam.execution.operator_score_rows import (
    MatchedOperatorScoreReducer,
    OperatorScoreReductions,
)
from jaxgam.fitting.family_execution import FamilyExecutionContext


@dataclass(frozen=True)
class MatchedOperatorEFSComponents:
    """Labeled score/d/t/q outputs; stochastic accuracy remains uncalibrated."""

    score: OperatorScoreEstimate
    score_rows: OperatorScoreReductions
    penalty: OperatorPenaltyLogDet
    fisher: ExactPenaltyStatistics | StochasticPenaltyStatistics
    edf_total: float
    trace_method: Literal["exact", "stochastic"]
    source_scans: int
    peak_bound_bytes: int
    fit_fingerprint: str
    source_batches: int = 0


def operator_efs_fit_fingerprint(
    model, family, beta: np.ndarray, rho: np.ndarray
) -> str:
    """Bind component outputs to one source, coordinate, family and fit state."""
    digest = hashlib.sha256()
    for value in (
        id(model),
        model.source_fingerprint,
        model.source_replay_digest,
        repr(FamilyExecutionContext.from_family(family)),
        repr(family.execution_parameter_snapshot()),
    ):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.asarray(beta, dtype="<f8").tobytes())
    digest.update(np.asarray(rho, dtype="<f8").tobytes())
    return digest.hexdigest()


class MatchedOperatorEFSProvider:
    """Own the observed/Fisher distinction and a bounded component sequence."""

    def __init__(
        self,
        state: FrozenOperatorWorkingState,
        source: RowSource,
        log_lambda: np.ndarray,
        *,
        score_phi: float,
        memory_budget_bytes: int,
        trace_method: Literal["exact", "stochastic"] = "exact",
        trace_probes: int = 32,
        trace_seed: int = 0,
        logdet_control: OperatorLogDetControl | None = None,
        solve_control: PCGControl | None = None,
        max_exact_coordinates: int = 256,
        max_dense_block_size: int = 64,
        max_y: int = 0,
    ) -> None:
        if not isinstance(state, FrozenOperatorWorkingState) or state.beta is None:
            raise TypeError("Matched EFS components require frozen coefficients.")
        if trace_method not in ("exact", "stochastic"):
            raise ValueError("trace_method must be exact or stochastic.")
        for name, value in (
            ("memory_budget_bytes", memory_budget_bytes),
            ("trace_probes", trace_probes),
            ("max_exact_coordinates", max_exact_coordinates),
            ("max_dense_block_size", max_dense_block_size),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if logdet_control is None:
            logdet_control = OperatorLogDetControl()
        if solve_control is None:
            solve_control = PCGControl()
        if not isinstance(logdet_control, OperatorLogDetControl):
            raise TypeError("logdet_control must be OperatorLogDetControl.")
        if not isinstance(solve_control, PCGControl):
            raise TypeError("solve_control must be PCGControl.")
        if trace_probes > 256:
            raise ValueError("Fisher trace probes may not exceed 256.")
        if (
            isinstance(trace_seed, bool)
            or not isinstance(trace_seed, numbers.Integral)
            or trace_seed < 0
        ):
            raise ValueError("Fisher trace seed must be a nonnegative integer.")
        if (
            not isinstance(log_lambda, np.ndarray)
            or log_lambda.dtype != np.float64
            or log_lambda.shape != (state.model.penalties.n_penalties,)
        ):
            raise ValueError("Matched EFS rho must be float64 in penalty order.")
        self.state = state
        self.source = source
        self.log_lambda = log_lambda.copy()
        self.log_lambda.setflags(write=False)
        self.score_phi = score_phi
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.trace_method = trace_method
        self.trace_probes = int(trace_probes)
        self.trace_seed = trace_seed
        self.logdet_control = logdet_control
        self.solve_control = solve_control
        self.max_exact_coordinates = int(max_exact_coordinates)
        self.max_dense_block_size = int(max_dense_block_size)
        self.max_y = max_y
        component_budget, _, external_bound = self._component_budgets()
        self.external_bound_bytes = external_bound
        # The reducer validates score scale, rho, theta and source identity.
        self.reducer = MatchedOperatorScoreReducer(
            state,
            source,
            self.log_lambda,
            score_phi=score_phi,
            memory_budget_bytes=component_budget,
            max_y=max_y,
        )

    @classmethod
    def from_fit(
        cls,
        fitted: OperatorFitResult,
        source: RowSource,
        *,
        score_phi: float,
        memory_budget_bytes: int,
        trace_method: Literal["exact", "stochastic"] = "exact",
        trace_probes: int = 32,
        trace_seed: int = 0,
        logdet_control: OperatorLogDetControl | None = None,
        solve_control: PCGControl | None = None,
        max_exact_coordinates: int = 256,
        max_dense_block_size: int = 64,
        max_y: int = 0,
    ) -> MatchedOperatorEFSProvider:
        """Bind all components to one converged fixed-sp operator fit."""
        if not isinstance(fitted, OperatorFitResult):
            raise TypeError("Matched EFS components require OperatorFitResult.")
        if not fitted.converged or not np.isfinite(fitted.deviance):
            raise ValueError("Matched EFS components require a converged finite fit.")
        state = FrozenOperatorWorkingState(
            fitted.model,
            fitted.family,
            fitted.coefficients,
            _operator_family_row_bound(fitted.model),
            17 * fitted.model.n_coef + 16_384,
        )
        return cls(
            state,
            source,
            fitted.log_lambda,
            score_phi=score_phi,
            memory_budget_bytes=memory_budget_bytes,
            trace_method=trace_method,
            trace_probes=trace_probes,
            trace_seed=trace_seed,
            logdet_control=logdet_control,
            solve_control=solve_control,
            max_exact_coordinates=max_exact_coordinates,
            max_dense_block_size=max_dense_block_size,
            max_y=max_y,
        )

    def _component_budgets(self) -> tuple[int, int, int]:
        p = self.state.model.n_coef
        m = self.state.model.penalties.n_penalties
        outputs = 8 * (
            4 * self.logdet_control.probes
            + 4 * self.trace_probes * m
            + 16 * m
            + 8 * p
            + 256
        )
        external = (
            self.state.model.memory_budget_bytes
            + self.state.row_memory_budget_bytes
            + self.state.retained_bound_bytes
            + outputs
            + 16_384
        )
        component = self.memory_budget_bytes - external
        system = component // 2
        if system < 1:
            raise MemoryError("Matched EFS components exceed their total budget.")
        return component, system, external

    def evaluate(self) -> MatchedOperatorEFSComponents:
        """Replay all components with shared beta/theta/rho and counted scans."""
        self.state.validate()
        component_budget, system_budget, external = self._component_budgets()
        rows = self.reducer.reduce()
        penalty = LocalOperatorPenaltyDeterminant(
            self.state.model.penalties,
            memory_budget_bytes=component_budget,
            max_dense_block_size=self.max_dense_block_size,
        ).evaluate(self.log_lambda)
        if penalty.derivative is None:
            raise RuntimeError("Matched EFS penalty derivative is unavailable.")
        observed_system = FrozenScanSystem(
            self.state.model,
            self.source,
            self.log_lambda,
            curvature_from_rows=self.state.curvature,
            memory_budget_bytes=system_budget,
        )
        logdet_provider = FrozenOperatorLogDet(
            observed_system,
            memory_budget_bytes=component_budget,
            control=self.logdet_control,
        )
        if observed_system.has_exact_diagonal:
            observed = logdet_provider.exact_structured_diagonal()
        elif observed_system.has_exact_arrowhead:
            observed = logdet_provider.exact_structured_arrowhead()
        else:
            observed = logdet_provider.estimate()
        observed_scans = observed_system.diagnostics.completed_scans
        observed_batches = observed_system.diagnostics.yielded_batches
        observed_peak = max(
            observed_system.construction_peak_bound_bytes,
            observed.diagnostics.peak_bound_bytes,
        )
        del logdet_provider
        del observed_system
        score = assemble_operator_score(rows.terms, observed, penalty)
        del observed
        fisher_system = FrozenScanSystem(
            self.state.model,
            self.source,
            self.log_lambda,
            curvature_from_rows=self.state.fisher_curvature,
            memory_budget_bytes=system_budget,
        )
        if self.trace_method == "exact":
            fisher = ExactOperatorPenaltyStatistics(
                fisher_system,
                memory_budget_bytes=component_budget,
                max_trace_coordinates=self.max_exact_coordinates,
                control=self.solve_control,
            ).statistics(self.state.beta)
        else:
            fisher = StochasticOperatorPenaltyStatistics(
                fisher_system,
                memory_budget_bytes=component_budget,
                probes=self.trace_probes,
                seed=self.trace_seed,
                control=self.solve_control,
            ).statistics(self.state.beta)
        fisher_scans = fisher_system.diagnostics.completed_scans
        fisher_batches = fisher_system.diagnostics.yielded_batches
        fisher_peak = max(
            fisher_system.construction_peak_bound_bytes,
            fisher.diagnostics.peak_bound_bytes,
        )
        edf = float(
            self.state.model.n_coef - np.dot(np.exp(self.log_lambda), fisher.traces)
        )
        if not np.isfinite(edf):
            raise FloatingPointError("Matched operator Fisher EDF is nonfinite.")
        self.state.validate()
        return MatchedOperatorEFSComponents(
            score,
            rows,
            penalty,
            fisher,
            edf,
            self.trace_method,
            rows.source_scans + observed_scans + fisher_scans,
            external
            + max(
                rows.peak_bound_bytes,
                penalty.peak_bound_bytes,
                observed_peak,
                fisher_peak,
            ),
            operator_efs_fit_fingerprint(
                self.state.model,
                self.state.family,
                self.state.beta,
                self.log_lambda,
            ),
            rows.source_batches + observed_batches + fisher_batches,
        )
