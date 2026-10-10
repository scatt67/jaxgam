"""Bounded stochastic log determinant of one certified frozen SPD operator.

This component estimates only ``log det(H)``. A REML score additionally needs
the family likelihood, penalty pseudo-determinant, scale, and null-space terms.
The returned Monte Carlo error does not bound Lanczos truncation error, so a
controller must calibrate both before comparing trial scores.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from dataclasses import dataclass
from decimal import Decimal, Inexact, localcontext
from typing import Literal

import jax
import numpy as np

from jaxgam.execution.operator import FrozenScanSystem
from jaxgam.execution.operator_arrowhead import ExactArrowheadFactorization


@dataclass(frozen=True)
class OperatorLogDetControl:
    """Fixed common-probe policy for repeatable comparisons of equal-size H."""

    probes: int = 16
    lanczos_depth: int = 32
    seed: int = 0
    breakdown_rtol: float = 64 * np.finfo(np.float64).eps

    def __post_init__(self) -> None:
        for name, maximum in (("probes", 256), ("lanczos_depth", 128)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"{name} must be an integer in [1, {maximum}].")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, numbers.Integral)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer.")
        if (
            isinstance(self.breakdown_rtol, bool)
            or not isinstance(self.breakdown_rtol, numbers.Real)
            or not np.isfinite(self.breakdown_rtol)
            or self.breakdown_rtol <= 0
        ):
            raise ValueError("breakdown_rtol must be finite and positive.")


_DEFAULT_LOGDET_CONTROL = OperatorLogDetControl()


@dataclass(frozen=True)
class OperatorLogDetDiagnostics:
    """Actual scan work and separate sampling/truncation diagnostics.

    ``exact_krylov_probes`` counts numerical closure or full-dimensional
    recurrences; it is not a certified finite-precision logdet error bound.
    A depth-one truncation has no comparison depth and reports an infinite
    depth difference rather than suggesting zero truncation error.
    """

    probes: int
    lanczos_depth: int
    seed: int
    completed_actions: int
    source_scans: int
    exact_krylov_probes: int
    monte_carlo_standard_error: float
    maximum_depth_difference: float
    peak_bound_bytes: int
    n_coef: int
    probe_fingerprint: str
    strategy: Literal[
        "stochastic_lanczos", "structured_diagonal", "structured_arrowhead"
    ] = "stochastic_lanczos"
    rounding_error_bound: float | None = None
    diagonal_condition_ratio: float | None = None
    minimum_diagonal: float | None = None


@dataclass(frozen=True)
class OperatorLogDetEstimate:
    """Stochastic determinant component, not a complete or certified score."""

    value: float
    per_probe: np.ndarray
    diagnostics: OperatorLogDetDiagnostics

    def __post_init__(self) -> None:
        values = np.array(self.per_probe, dtype=np.float64, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "per_probe", values)


class OperatorLogDetError(RuntimeError):
    """A probe failed; no partial determinant estimate is returned."""

    def __init__(
        self, probe: int, step: int, completed_actions: int, source_scans: int
    ) -> None:
        self.probe = probe
        self.step = step
        self.completed_actions = completed_actions
        self.source_scans = source_scans
        super().__init__(
            f"Frozen logdet probe {probe} failed at Lanczos step {step} "
            f"after {completed_actions} actions and {source_scans} scans."
        )


def _log_quadrature(diagonal: np.ndarray, off_diagonal: np.ndarray) -> float:
    """Return e1' log(T) e1 for a small Lanczos tridiagonal."""
    tridiagonal = np.diag(diagonal)
    if len(off_diagonal):
        tridiagonal += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(tridiagonal)
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0):
        raise np.linalg.LinAlgError("Lanczos Ritz values are not positive finite.")
    return float(np.dot(eigenvectors[0] ** 2, np.log(eigenvalues)))


def _stable_norm(vector: np.ndarray) -> float:
    """Preserve the action scale when its squared norm would underflow."""
    maximum = float(np.max(np.abs(vector), initial=0.0))
    if maximum == 0.0:
        return 0.0
    return maximum * float(np.sqrt(np.sum((vector / maximum) ** 2)))


class FrozenOperatorLogDet:
    """Compose frozen actions and fixed probes with capped Lanczos workspace.

    ``system`` owns a sufficient SPD certificate. The caller supplies a total
    budget that includes the system's entire budget, one capped Krylov basis,
    vectors, small eigensolver workspace, and returned probe values. No global
    coefficient-square matrix is formed for large p.
    """

    def __init__(
        self,
        system: FrozenScanSystem,
        *,
        memory_budget_bytes: int,
        control: OperatorLogDetControl = _DEFAULT_LOGDET_CONTROL,
    ) -> None:
        if not isinstance(system, FrozenScanSystem):
            raise TypeError("A certified frozen SPD operator is required.")
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, numbers.Integral)
            or memory_budget_bytes < 1
        ):
            raise ValueError("memory_budget_bytes must be a positive integer.")
        if not isinstance(control, OperatorLogDetControl):
            raise TypeError("control must be OperatorLogDetControl.")
        self.system = system
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.control = control

    def _peak_bound(self) -> int:
        p = self.system.model.n_coef
        depth = min(p, self.control.lanczos_depth)
        # Basis, probe, current/previous/action vectors, JAX transfer copy,
        # reorthogonalization scratch, two capped eigensolver workspaces and
        # per-probe output. The frozen system reserves its full declared budget.
        return (
            self.system.memory_budget_bytes
            + 8 * p * (depth + 7)
            + 96 * depth**2
            + 16 * self.control.probes
            + 8192
        )

    def estimate(self) -> OperatorLogDetEstimate:
        """Estimate log det using reproducible Rademacher bilinear probes."""
        required = self._peak_bound()
        if required > self.memory_budget_bytes:
            raise MemoryError(
                f"Frozen logdet requires {required} bytes; "
                f"budget is {self.memory_budget_bytes}."
            )
        p = self.system.model.n_coef
        if p < 1:
            raise ValueError("Frozen logdet needs at least one coefficient.")
        depth = min(p, self.control.lanczos_depth)
        rng = np.random.default_rng(self.control.seed)
        probe_digest = hashlib.sha256()
        probe_digest.update(np.asarray([p, self.control.probes], dtype="<i8").tobytes())
        values = np.empty(self.control.probes, dtype=np.float64)
        basis = np.empty((depth, p), dtype=np.float64)
        diagonal = np.empty(depth, dtype=np.float64)
        off_diagonal = np.empty(max(depth - 1, 0), dtype=np.float64)
        maximum_depth_difference = 0.0
        exact_krylov_probes = 0
        actions_before = self.system.diagnostics.normal_actions
        scans_before = self.system.diagnostics.completed_scans
        for probe in range(self.control.probes):
            signs = rng.integers(0, 2, size=p, dtype=np.int8)
            probe_digest.update(signs.tobytes())
            initial = (2 * signs.astype(np.float64) - 1) / math.sqrt(p)
            basis[0] = initial
            previous_beta = 0.0
            used = 0
            exact = False
            for step in range(depth):
                try:
                    action = np.asarray(
                        self.system.matvec(
                            jax.device_put(basis[step], jax.devices("cpu")[0])
                        ),
                        dtype=np.float64,
                    )
                    if action.shape != (p,) or not np.all(np.isfinite(action)):
                        raise FloatingPointError("Frozen logdet action is nonfinite.")
                    alpha = float(np.dot(basis[step], action))
                    if not np.isfinite(alpha):
                        raise FloatingPointError(
                            "Frozen logdet projection is nonfinite."
                        )
                    diagonal[step] = alpha
                    remainder = action - alpha * basis[step]
                    if step:
                        remainder -= previous_beta * basis[step - 1]
                    # Full double reorthogonalization avoids ghost Ritz values.
                    for _ in range(2):
                        for prior in range(step + 1):
                            remainder -= np.dot(basis[prior], remainder) * basis[prior]
                    beta = _stable_norm(remainder)
                    used = step + 1
                    action_norm = _stable_norm(action)
                    if beta <= self.control.breakdown_rtol * max(
                        action_norm, np.finfo(np.float64).tiny
                    ):
                        exact = True
                        break
                    if step + 1 < depth:
                        off_diagonal[step] = beta
                        basis[step + 1] = remainder / beta
                        previous_beta = beta
                except Exception as error:
                    raise OperatorLogDetError(
                        probe,
                        step,
                        self.system.diagnostics.normal_actions - actions_before,
                        self.system.diagnostics.completed_scans - scans_before,
                    ) from error
            try:
                quadrature = _log_quadrature(
                    diagonal[:used], off_diagonal[: max(used - 1, 0)]
                )
                if used == 1 and not exact:
                    maximum_depth_difference = math.inf
                elif used > 1 and not exact:
                    prior = _log_quadrature(
                        diagonal[: used - 1], off_diagonal[: used - 2]
                    )
                    maximum_depth_difference = max(
                        maximum_depth_difference, p * abs(quadrature - prior)
                    )
            except np.linalg.LinAlgError as error:
                raise OperatorLogDetError(
                    probe,
                    used - 1,
                    self.system.diagnostics.normal_actions - actions_before,
                    self.system.diagnostics.completed_scans - scans_before,
                ) from error
            values[probe] = p * quadrature
            exact_krylov_probes += exact or used == p
        actions = self.system.diagnostics.normal_actions - actions_before
        scans = self.system.diagnostics.completed_scans - scans_before
        if actions != scans:
            raise RuntimeError("Frozen logdet action and source-scan counts differ.")
        standard_error = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else math.inf
        )
        return OperatorLogDetEstimate(
            value=float(np.mean(values)),
            per_probe=values,
            diagnostics=OperatorLogDetDiagnostics(
                probes=self.control.probes,
                lanczos_depth=depth,
                seed=self.control.seed,
                completed_actions=actions,
                source_scans=scans,
                exact_krylov_probes=exact_krylov_probes,
                monte_carlo_standard_error=standard_error,
                maximum_depth_difference=maximum_depth_difference,
                peak_bound_bytes=required,
                n_coef=p,
                probe_fingerprint=probe_digest.hexdigest(),
            ),
        )

    def exact_structured_diagonal(self) -> OperatorLogDetEstimate:
        """Sum log of the certified single-index diagonal with a rounding budget.

        Decimal logarithms are correctly rounded at 80 decimal digits. The
        bound covers their rounding and final float conversion for the
        assembled float64 diagonal; it does not certify source/JAX arithmetic.
        """
        if not self.system.has_exact_diagonal:
            raise ValueError("Structured logdet requires a diagonal operator.")
        p = self.system.model.n_coef
        required = self.system.memory_budget_bytes + 24 * p + 8192
        if required > self.memory_budget_bytes:
            raise MemoryError("Structured diagonal logdet exceeds its budget.")
        diagonal = self.system.exact_diagonal()
        precision = 80
        with localcontext() as context:
            context.prec = 160
            context.clear_flags()
            total = Decimal(0)
            rounding = Decimal(0)
            for value in diagonal:
                with localcontext() as log_context:
                    log_context.prec = precision
                    logarithm = Decimal.from_float(float(value)).ln()
                total += logarithm
                rounding += Decimal(5).scaleb(logarithm.adjusted() - precision)
            if context.flags[Inexact]:
                raise ArithmeticError("Structured decimal log sum lost precision.")
        result = float(total)
        bound = 2 * float(rounding) + math.ulp(result)
        if not math.isfinite(result) or not math.isfinite(bound):
            raise FloatingPointError("Structured diagonal logdet is nonfinite.")
        digest = hashlib.sha256()
        digest.update(b"structured-diagonal-v1")
        digest.update(np.asarray([p], dtype="<i8").tobytes())
        return OperatorLogDetEstimate(
            value=result,
            per_probe=np.array([result]),
            diagnostics=OperatorLogDetDiagnostics(
                probes=1,
                lanczos_depth=1,
                seed=0,
                completed_actions=0,
                source_scans=0,
                exact_krylov_probes=0,
                monte_carlo_standard_error=0.0,
                maximum_depth_difference=0.0,
                peak_bound_bytes=required,
                n_coef=p,
                probe_fingerprint=digest.hexdigest(),
                strategy="structured_diagonal",
                rounding_error_bound=bound,
                diagonal_condition_ratio=float(np.max(diagonal) / np.min(diagonal)),
                minimum_diagonal=float(np.min(diagonal)),
            ),
        )

    def exact_structured_arrowhead(self) -> OperatorLogDetEstimate:
        """Schur logdet for one modest block and one indexed identity block.

        Its conditional floating result has no whole-score error calibration;
        controllers must leave large-p decisions indeterminate until supplied
        an independent score bound for the supported family and conditioning.
        """
        factor = ExactArrowheadFactorization.from_system(
            self.system, memory_budget_bytes=self.memory_budget_bytes
        )
        result = factor.log_determinant
        digest = hashlib.sha256()
        digest.update(b"structured-arrowhead-v1")
        digest.update(np.asarray([self.system.model.n_coef], dtype="<i8").tobytes())
        diagonal = factor.indexed_diagonal
        return OperatorLogDetEstimate(
            value=result,
            per_probe=np.array([result]),
            diagnostics=OperatorLogDetDiagnostics(
                probes=1,
                lanczos_depth=1,
                seed=0,
                completed_actions=0,
                source_scans=factor.source_scans,
                exact_krylov_probes=0,
                monte_carlo_standard_error=0.0,
                maximum_depth_difference=0.0,
                peak_bound_bytes=factor.peak_bound_bytes,
                n_coef=self.system.model.n_coef,
                probe_fingerprint=digest.hexdigest(),
                strategy="structured_arrowhead",
                minimum_diagonal=float(np.min(diagonal)),
                diagonal_condition_ratio=float(np.max(diagonal) / np.min(diagonal)),
            ),
        )
