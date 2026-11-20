"""Calibrated bulk Fisher variance from bounded covariance samples.

For a certified Fisher system H = X'WX + S, draw b = X'W^(1/2) z + S^(1/2) q
and solve H u = b. Then E[(x'u)^2] = x'H^-1 x. Only the local penalty
blocks are factored; no global coefficient-square array is formed. Prediction
rows are evaluated in bounded batches against the retained p-by-probe sketch.
The reported Monte Carlo error excludes iterative-solve bias, which is
reported separately and must be controlled independently.

Calibration. Each probe term (x'u)^2 is x'H^-1 x times an independent
chi-square(1) draw, so with exact solves probes * estimate / exact is
chi-square with ``probes`` degrees of freedom. A selected row is judged by that
pivot, mapped to the standard normal deviate with the same tail probability;
the sample Monte Carlo standard error is not used for the decision, since it
is small exactly when the estimate is low.
"""

from __future__ import annotations

import hashlib
import numbers
from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg.lapack import dsyevr, dsyevr_lwork
from scipy.special import ndtri_exp
from scipy.stats import chi2

from jaxgam.data.source import RowSource
from jaxgam.execution.operator.system import FrozenScanSystem, PCGControl
from jaxgam.execution.operator.variance import SelectedOperatorVariance
from jaxgam.formula.predict_matrix import Data
from jaxgam.inference.operator_geometry import OperatorPredictionGeometry, _rows
from jaxgam.penalties.structure import (
    DiagonalPenalty,
    IdentityPenalty,
    PenaltyBlock,
)

_DEFAULT_BULK_CONTROL = PCGControl()


@dataclass(frozen=True)
class BulkVarianceDiagnostics:
    probes: int
    seed: int
    replay_preparation_scans: int
    fisher_construction_scans: int
    completed_solves: int
    krylov_iterations: int
    normal_actions: int
    training_source_scans: int
    calibration_source_scans: int
    calibration_rows: int
    prediction_source_scans: int
    maximum_relative_residual: float
    peak_bound_bytes: int
    approximation: str = "fisher_covariance_monte_carlo"


@dataclass(frozen=True)
class BulkVarianceBatch:
    """Ordered link-scale variance estimates and Monte Carlo standard errors."""

    positions: np.ndarray
    variance: np.ndarray
    monte_carlo_se: np.ndarray

    def __post_init__(self) -> None:
        for name in ("positions", "variance", "monte_carlo_se"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def link_se(self) -> np.ndarray:
        """Approximate link-scale prediction standard error."""
        value = np.sqrt(self.variance)
        value.setflags(write=False)
        return value


def equivalent_normal_deviates(
    estimated: np.ndarray, exact: np.ndarray, probes: int
) -> np.ndarray:
    """Standard normal deviates of the chi-square calibration pivot.

    ``probes * estimated / exact`` is chi-square with ``probes`` degrees of
    freedom when the estimate averages ``probes`` independent squared Gaussian
    projections with variance ``exact``. Each pivot is mapped to the normal
    deviate with the same tail probability: through the log distribution
    function below the median and the log survival function above it, so
    neither tail underflows before the double-precision limit: a tail
    probability below about 1e-308 (a deviate beyond about 37.5) is reported
    as an infinite deviate. A zero exact variance gives 0 for a zero estimate
    and infinity otherwise.
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if estimated.shape != exact.shape:
        raise ValueError("Estimated and exact variances must have one shape.")
    if (
        isinstance(probes, bool)
        or not isinstance(probes, numbers.Integral)
        or probes < 1
    ):
        raise ValueError("The calibration pivot needs a positive probe count.")
    if (
        not np.all(np.isfinite(estimated))
        or not np.all(np.isfinite(exact))
        or np.any(estimated < 0)
        or np.any(exact < 0)
    ):
        raise FloatingPointError("Calibration variances must be finite, nonnegative.")
    deviates = np.where(estimated == 0, 0.0, np.inf)
    positive = exact > 0
    pivot = probes * estimated[positive] / exact[positive]
    lower = chi2.logcdf(pivot, probes)
    upper = chi2.logsf(pivot, probes)
    deviates[positive] = np.where(lower < upper, ndtri_exp(lower), -ndtri_exp(upper))
    return deviates


@dataclass(frozen=True)
class BulkVarianceCalibration:
    """Selected exact Fisher checks; these do not bound unselected rows.

    ``equivalent_deviates`` are each row's chi-square calibration pivot as a
    standard normal deviate (:func:`equivalent_normal_deviates`), and
    ``maximum_absolute_deviate`` their largest absolute value; the sample
    ``monte_carlo_se`` is reported as a diagnostic only.
    """

    exact: np.ndarray
    estimated: np.ndarray
    monte_carlo_se: np.ndarray
    equivalent_deviates: np.ndarray
    maximum_absolute_deviate: float
    selected_source_scans: int

    def __post_init__(self) -> None:
        for name in ("exact", "estimated", "monte_carlo_se", "equivalent_deviates"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class BulkVarianceSolveError(RuntimeError):
    """A probe solve failed, so no partial sketch is returned."""

    def __init__(self, probe: int, status: int, iterations: int) -> None:
        self.probe = probe
        self.status = status
        self.iterations = iterations
        super().__init__(
            f"Fisher covariance probe {probe} failed: "
            f"status {status} after {iterations} iterations."
        )


def _dense_penalty_eigh_workspace(size: int) -> int:
    """Bound live local arrays using the selected LAPACK driver's work query."""
    if size == 0:
        return 0
    work, iwork, info = dsyevr_lwork(size)
    if info != 0:
        raise np.linalg.LinAlgError("Local penalty eigensolver work query failed.")
    # During assembly: local, penalty.dense() and its scaled copy (3 n²).
    # During decomposition: local, Fortran symmetric input and returned
    # eigenvectors (3 n²). Two extra n² slots cover input/wrapper copies.
    # dsyevr allocates the queried double/integer work arrays, n eigenvalues,
    # 2n support indices, and O(n) draw/scaling/output vectors.
    return int(
        8 * (5 * size * size + int(work) + 8 * size)
        + max(np.dtype(np.intp).itemsize, np.dtype(np.intc).itemsize)
        * (int(iwork) + 2 * size)
        + 4096
    )


def _penalty_noise(
    block: PenaltyBlock, log_sp: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw from the local summed penalty, preserving its null space."""
    size = block.size
    if size == 0:
        return np.empty(0)
    if len(block.local_penalties) == 1 and isinstance(
        block.local_penalties[0], IdentityPenalty
    ):
        variance = np.exp(log_sp[block.sp_indices[0]]) * block.local_penalties[0].scale
        return np.sqrt(variance) * rng.standard_normal(size)
    if all(
        isinstance(penalty, (IdentityPenalty, DiagonalPenalty))
        for penalty in block.local_penalties
    ):
        diagonal = np.zeros(size)
        for index, penalty in zip(block.sp_indices, block.local_penalties, strict=True):
            values = (
                penalty.scale
                if isinstance(penalty, IdentityPenalty)
                else penalty.diagonal
            )
            diagonal += np.exp(log_sp[index]) * values
        if np.any(diagonal < 0) or not np.all(np.isfinite(diagonal)):
            raise ValueError("Local Fisher penalty is not positive semidefinite.")
        return np.sqrt(diagonal) * rng.standard_normal(size)
    local = np.zeros((size, size))
    for index, penalty in zip(block.sp_indices, block.local_penalties, strict=True):
        local += np.exp(log_sp[index]) * penalty.dense()
    symmetric = np.array(local, order="F")
    symmetric += local.T
    symmetric *= 0.5
    work, iwork, info = dsyevr_lwork(size)
    if info != 0:
        raise np.linalg.LinAlgError("Local penalty eigensolver work query failed.")
    eigenvalues, vectors, _, _, info = dsyevr(
        symmetric,
        lwork=int(work),
        liwork=int(iwork),
        overwrite_a=1,
    )
    if info != 0:
        raise np.linalg.LinAlgError("Local penalty eigensolver failed.")
    tolerance = 64 * np.finfo(float).eps * max(1.0, np.max(np.abs(eigenvalues)))
    if np.any(eigenvalues < -tolerance):
        raise ValueError("Local Fisher penalty is not positive semidefinite.")
    return vectors @ (np.sqrt(np.maximum(eigenvalues, 0.0)) * rng.standard_normal(size))


class BulkOperatorVariance:
    """Build a bounded p-by-probe sketch from an externally certified Fisher system.

    ``fisher_system`` must use expected, family-owned curvature at the accepted
    beta/theta and smoothing parameters. Positive definiteness of an arbitrary
    frozen system is not enough to certify covariance semantics. The full
    system budget is reserved before probe or prediction allocations.
    """

    def __init__(
        self,
        fisher_system: FrozenScanSystem,
        *,
        scale: float,
        memory_budget_bytes: int,
        probes: int,
        seed: int,
        replay_preparation_scans: int = 0,
        control: PCGControl = _DEFAULT_BULK_CONTROL,
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
        for name, value, minimum in (
            ("memory_budget_bytes", memory_budget_bytes, 1),
            ("probes", probes, 2),
            ("seed", seed, 0),
            ("replay_preparation_scans", replay_preparation_scans, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer at least {minimum}.")
        if not isinstance(control, PCGControl):
            raise TypeError("Bulk variance requires PCGControl.")
        model = fisher_system.model
        self.system = fisher_system
        self.geometry = OperatorPredictionGeometry(
            model.modest.predict_spec,
            model.modest_columns,
            model.random_effects,
            model.n_coef,
            model.batch_rows,
            model.modest_raw_width,
            model.metadata_bound_bytes,
        )
        self.scale = float(scale)
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.probes = int(probes)
        self.seed = int(seed)
        self.replay_preparation_scans = int(replay_preparation_scans)
        self.control = control
        p, B, K = model.n_coef, model.batch_rows, len(model.modest_columns)
        largest_dense_penalty = max(
            (
                block.size
                for block in model.penalties.blocks
                if any(
                    not isinstance(penalty, (IdentityPenalty, DiagonalPenalty))
                    for penalty in block.local_penalties
                )
            ),
            default=0,
        )
        if largest_dense_penalty > K:
            raise MemoryError("Dense local penalty exceeds the modest column cap.")
        # Retained sketch, current RHS/solve, local penalty eigensystem,
        # bulk and exact-calibration geometries, and four simultaneous
        # B-by-probe prediction products and the probes-by-K modest gather.
        # In particular the p-by-probe sketch remains charged while an exact
        # selected calibration solve or a prediction batch is evaluated.
        self.peak_bound_bytes = (
            fisher_system.memory_budget_bytes
            + 8 * p * (self.probes + 4)
            + 2 * self.geometry.workspace_bound(B)
            + _dense_penalty_eigh_workspace(largest_dense_penalty)
            + 8 * self.probes * K
            + 32 * B * self.probes
            + 32 * B
            + 4096
        )
        if self.peak_bound_bytes > self.memory_budget_bytes:
            raise MemoryError(
                f"Bulk variance requires {self.peak_bound_bytes} bytes; "
                f"budget is {self.memory_budget_bytes}."
            )
        self._samples: np.ndarray | None = None
        self._diagnostics: BulkVarianceDiagnostics | None = None

    @property
    def diagnostics(self) -> BulkVarianceDiagnostics:
        if self._diagnostics is None:
            raise RuntimeError("Bulk Fisher sketch has not been completed.")
        return self._diagnostics

    def _sample_rhs(self, rng: np.random.Generator) -> np.ndarray:
        system = self.system
        if system.source.fingerprint() != system.model.source_fingerprint:
            raise RuntimeError("Frozen source changed before covariance sampling.")
        rhs = np.zeros(system.model.n_coef)
        columns = np.asarray(system.model.modest_columns, dtype=np.int64)
        digest = hashlib.sha256()
        for batch in system._checked.scan(system.batch_rows):
            rows = system._evaluate_rows(batch)
            weight = system._weight(batch, rows)
            if np.any(weight < 0):
                raise ValueError(
                    "Covariance sampling requires nonnegative Fisher rows."
                )
            digest.update(weight.tobytes())
            noise = np.sqrt(weight) * rng.standard_normal(len(weight))
            rhs[columns] += rows.modest_values.T @ noise
            for ids, amplitude, start, _ in rows.indexed_rows:
                np.add.at(rhs, start + ids, amplitude * noise)
            del rows, weight, noise, batch
        if digest.hexdigest() != system._curvature_digest:
            raise RuntimeError("Frozen Fisher curvature changed during sampling.")
        for block in system.model.penalties.blocks:
            rhs[block.start : block.stop] += _penalty_noise(
                block, np.asarray(system._rho), rng
            )
        if not np.all(np.isfinite(rhs)):
            raise FloatingPointError("Fisher covariance probe RHS is nonfinite.")
        return rhs

    def build(self) -> BulkOperatorVariance:
        """Draw and solve every fixed-seed probe; return no partial sketch."""
        if self._samples is not None:
            return self
        rng = np.random.default_rng(self.seed)
        p = self.geometry.n_coef
        samples = np.empty((self.probes, p), dtype=np.float64)
        before_scans = self.system.diagnostics.completed_scans
        before_actions = self.system.diagnostics.normal_actions
        iterations = 0
        maximum_relative_residual = 0.0
        for index in range(self.probes):
            rhs = self._sample_rhs(rng)
            solved = self.system.solve(rhs, control=self.control)
            if not solved.converged:
                raise BulkVarianceSolveError(index, solved.status, solved.iterations)
            samples[index] = solved.solution
            iterations += solved.iterations
            maximum_relative_residual = max(
                maximum_relative_residual, solved.relative_residual
            )
            del rhs, solved
        samples.setflags(write=False)
        actions = self.system.diagnostics.normal_actions - before_actions
        scans = self.system.diagnostics.completed_scans - before_scans
        if scans != self.probes + actions:
            raise RuntimeError("Fisher probe source-pass accounting is inconsistent.")
        self._samples = samples
        self._diagnostics = BulkVarianceDiagnostics(
            self.probes,
            self.seed,
            self.replay_preparation_scans,
            self.system.construction_source_scans,
            self.probes,
            iterations,
            actions,
            scans,
            0,
            0,
            0,
            maximum_relative_residual,
            self.peak_bound_bytes,
        )
        return self

    def _batch_estimate(self, data: Data) -> tuple[np.ndarray, np.ndarray]:
        if self._samples is None:
            raise RuntimeError("Build the bulk Fisher sketch first.")
        n = _rows(data)
        if n > self.geometry.batch_rows:
            raise ValueError("Bulk variance prediction batch exceeds its row bound.")
        projection = np.zeros((n, self.probes))
        modest_columns = np.asarray(self.geometry.modest_columns, dtype=np.int64)
        for start, stop, modest, columns in self.geometry._batches(data):
            projection[start:stop] = modest @ self._samples[:, modest_columns].T
            for term in self.geometry.random_effects:
                ids, amplitude = term.evaluate(columns)
                projection[start:stop] += (
                    amplitude[:, None] * self._samples[:, term.col_start + ids].T
                )
        squares = projection**2
        variance = self.scale * np.mean(squares, axis=1)
        error = self.scale * np.std(squares, axis=1, ddof=1) / np.sqrt(self.probes)
        if not np.all(np.isfinite(variance)) or not np.all(np.isfinite(error)):
            raise FloatingPointError("Bulk Fisher variance estimate is nonfinite.")
        return variance, error

    def estimate_iter(self, source: RowSource) -> Iterator[BulkVarianceBatch]:
        """Yield positional prediction batches after one replay-checked scan."""
        if self._samples is None:
            raise RuntimeError("Build the bulk Fisher sketch first.")
        if not isinstance(source, RowSource):
            raise TypeError("Bulk prediction requires a replayable RowSource.")
        fingerprint = source.fingerprint()
        expected = source.n_rows
        seen = 0
        for batch in source.scan(self.geometry.batch_rows):
            positions = np.asarray(batch.row_positions)
            n = len(positions)
            if (
                n > self.geometry.batch_rows
                or seen + n > expected
                or not np.all(batch.valid)
            ):
                raise RuntimeError("Bulk prediction source exceeded its row contract.")
            if (
                positions.ndim != 1
                or positions.dtype.kind not in "iu"
                or np.any(positions < 0)
            ):
                raise ValueError(
                    "Bulk prediction positions must be nonnegative integers."
                )
            seen += n
            columns = self.geometry.slice_columns(batch.columns, 0, n)
            variance, error = self._batch_estimate(columns)
            yield BulkVarianceBatch(positions, variance, error)
        if (
            seen != expected
            or source.n_rows != expected
            or source.fingerprint() != fingerprint
        ):
            raise RuntimeError("Bulk prediction source changed during replay.")
        self._diagnostics = replace(
            self.diagnostics,
            prediction_source_scans=self.diagnostics.prediction_source_scans + 1,
        )

    def calibrate(self, selected: Data) -> BulkVarianceCalibration:
        """Compare selected rows to exact Fisher solves without changing probes."""
        if self._samples is None:
            raise RuntimeError("Build the bulk Fisher sketch first.")
        selected_rows = _rows(selected)
        if selected_rows < 1:
            raise ValueError("Selected exact calibration requires at least one row.")
        if selected_rows > min(256, self.geometry.batch_rows):
            raise ValueError("Calibration exceeds the bounded selected-row batch.")
        exact = SelectedOperatorVariance(
            self.system,
            scale=self.scale,
            memory_budget_bytes=(
                self.memory_budget_bytes - 8 * self.probes * self.geometry.n_coef
            ),
            control=self.control,
        ).variance(selected)
        estimated, error = self._batch_estimate(selected)
        deviates = equivalent_normal_deviates(estimated, exact.variance, self.probes)
        calibration = BulkVarianceCalibration(
            exact.variance,
            estimated,
            error,
            deviates,
            float(np.max(np.abs(deviates), initial=0.0)),
            exact.diagnostics.source_scans,
        )
        self._diagnostics = replace(
            self.diagnostics,
            calibration_source_scans=(
                self.diagnostics.calibration_source_scans
                + exact.diagnostics.source_scans
            ),
            calibration_rows=self.diagnostics.calibration_rows + selected_rows,
        )
        return calibration
