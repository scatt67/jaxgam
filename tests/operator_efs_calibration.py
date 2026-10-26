"""Generated rows and an independent decimal REML reference for operator EFS.

The calibration harness and the operator EFS calibration tests share these
runtime-generated models. The reference reads only the generated rows and the
model's identity-penalty scale. It evaluates the REML score, Fisher trace,
penalty quadratic and EDF with 50-digit decimal arithmetic and finds exact
penalized optima by decimal Newton iteration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pandas as pd

REGIMES = ("indexed_poisson", "indexed_gaussian", "arrowhead_poisson")
_DIGITS = 50
_PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628"
)
_FORMULAS = {
    ("indexed_poisson", "none"): "y~0+s(g,bs='re')",
    ("indexed_gaussian", "none"): "y~0+s(g,bs='re')",
    ("arrowhead_poisson", "x"): "y~x+s(g,bs='re')",
    ("arrowhead_poisson", "intercept"): "y~s(g,bs='re')",
}
# Defaults reproduce the generators of the released p<=512 calibration tests.
_DEFAULT_SHAPE = {
    ("indexed_poisson", "balanced"): (0.2, 0.3),
    ("indexed_poisson", "unbalanced"): (1.5, 0.7),
    ("indexed_gaussian", "balanced"): (1.2, 0.4),
    ("indexed_gaussian", "unbalanced"): (1.5, 0.7),
    ("arrowhead_poisson", "balanced"): (0.2, 0.3),
    ("arrowhead_poisson", "unbalanced"): (0.2, 0.3),
}
_GAUSSIAN_NOISE = 0.2


@dataclass(frozen=True)
class CalibrationCase:
    """One generated model; the role records whether it may set an envelope."""

    regime: str
    levels: int
    multiplicity: float
    seed: int
    mode: str = "balanced"
    batch_rows: int = 64
    base: float | None = None
    amplitude: float | None = None
    role: str = "calibration"
    modest: str | None = None

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"Unknown regime {self.regime!r}.")
        if self.modest is None:
            object.__setattr__(
                self, "modest", "x" if self.regime == "arrowhead_poisson" else "none"
            )
        if (self.regime, self.modest) not in _FORMULAS:
            raise ValueError(f"Unknown modest columns {self.modest!r}.")
        if self.mode not in ("balanced", "unbalanced"):
            raise ValueError(f"Unknown mode {self.mode!r}.")
        if self.levels < 2 or self.batch_rows < 1 or self.seed < 0:
            raise ValueError("Levels, batch rows and seed are invalid.")
        if not math.isfinite(self.multiplicity) or self.multiplicity < 1:
            raise ValueError("Every level needs at least one row.")
        if self.role not in ("calibration", "held_out", "boundary"):
            raise ValueError(f"Unknown role {self.role!r}.")

    @property
    def n_rows(self) -> int:
        return round(self.multiplicity * self.levels)

    @property
    def formula(self) -> str:
        return _FORMULAS[(self.regime, self.modest)]

    @property
    def shape(self) -> tuple[float, float]:
        base, amplitude = _DEFAULT_SHAPE[(self.regime, self.mode)]
        return (
            base if self.base is None else self.base,
            amplitude if self.amplitude is None else self.amplitude,
        )


def generate_frame(case: CalibrationCase) -> pd.DataFrame:
    """Deterministic rows; every level is occupied at least once."""
    p, n = case.levels, case.n_rows
    rng = np.random.default_rng(case.seed)
    if case.mode == "balanced":
        group = np.arange(n) % p
    else:
        probabilities = 1.0 / (1.0 + np.arange(p)) ** 1.1
        probabilities /= np.sum(probabilities)
        group = np.concatenate(
            (np.arange(p), rng.choice(p, size=n - p, p=probabilities))
        )
    base, amplitude = case.shape
    effect = amplitude * np.sin(group / 31.0)
    columns: dict[str, Any] = {}
    if case.regime == "indexed_poisson":
        response = rng.poisson(np.exp(base + effect))
    elif case.regime == "indexed_gaussian":
        response = base + effect + rng.normal(0.0, _GAUSSIAN_NOISE, n)
    elif case.modest == "x":
        x = np.linspace(-1.0, 1.0, n)
        response = rng.poisson(np.exp(base + 0.5 * x + effect))
        columns["x"] = x
    else:
        response = rng.poisson(np.exp(base + effect))
    if case.mode == "unbalanced" and case.regime != "indexed_gaussian":
        response[0] = 31
    labels = [f"g{index}" for index in range(p)]
    return pd.DataFrame(
        {
            "y": response,
            **columns,
            "g": pd.Categorical.from_codes(group, categories=labels),
        }
    )


def _d(value: float) -> Decimal:
    """Exact decimal image of one binary64 value."""
    return Decimal(float(value))


def _log_factorial(y: int, cache: dict[int, Decimal]) -> Decimal:
    if y not in cache:
        total = Decimal(0)
        for k in range(2, y + 1):
            total += Decimal(k).ln()
        cache[y] = total
    return cache[y]


def _cholesky_solve(matrix: list[list[Decimal]], rhs: list[Decimal]) -> list[Decimal]:
    k = len(rhs)
    lower = [[Decimal(0)] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][m] * lower[j][m] for m in range(j))
            lower[i][j] = value.sqrt() if i == j else value / lower[j][j]
    forward = [Decimal(0)] * k
    for i in range(k):
        forward[i] = (rhs[i] - sum(lower[i][m] * forward[m] for m in range(i))) / (
            lower[i][i]
        )
    solution = [Decimal(0)] * k
    for i in reversed(range(k)):
        solution[i] = (
            forward[i] - sum(lower[m][i] * solution[m] for m in range(i + 1, k))
        ) / lower[i][i]
    return solution


def _log_det_spd(matrix: list[list[Decimal]]) -> Decimal:
    k = len(matrix)
    lower = [[Decimal(0)] * k for _ in range(k)]
    total = Decimal(0)
    for i in range(k):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][m] * lower[j][m] for m in range(j))
            if i == j:
                if value <= 0:
                    raise ArithmeticError("Reference Schur complement is not SPD.")
                lower[i][j] = value.sqrt()
                total += 2 * lower[i][j].ln()
            else:
                lower[i][j] = value / lower[j][j]
    return total


@dataclass(frozen=True)
class ReferenceState:
    """Decimal score components at one coefficient vector."""

    score: Decimal
    deviance: Decimal
    trace: Decimal
    quadratic: Decimal
    edf: Decimal


class DecimalReference:
    """Independent decimal REML evaluation for the three structured regimes.

    It reads only the generated rows and the model's penalty scale. Penalized
    optima are found by decimal Newton iteration started from a supplied
    float64 coefficient vector.
    """

    def __init__(
        self,
        case: CalibrationCase,
        frame: pd.DataFrame,
        penalty_scale: float,
        modest_count: int,
    ) -> None:
        self.regime = case.regime
        self.levels = case.levels
        self.k = modest_count
        self.scale = _d(penalty_scale)
        self.group = frame["g"].cat.codes.to_numpy(dtype=np.int64)
        y = frame["y"].to_numpy(dtype=np.float64)
        self.y = y
        self.x = frame["x"].to_numpy(dtype=np.float64) if "x" in frame else None
        n = len(y)
        self.n = n
        with localcontext() as context:
            context.prec = _DIGITS
            self.count = np.bincount(self.group, minlength=self.levels)
            self.y_sum = [Decimal(0)] * self.levels
            self.y_log_y = [Decimal(0)] * self.levels
            self.y_square = [Decimal(0)] * self.levels
            self.row_y_log_y: list[Decimal] = []
            saturated = Decimal(0)
            cache: dict[int, Decimal] = {}
            for index in range(n):
                g = int(self.group[index])
                value = _d(y[index])
                self.y_sum[g] += value
                if self.regime == "indexed_gaussian":
                    self.y_square[g] += value * value
                    continue
                count = int(y[index])
                if count != y[index] or count < 0:
                    raise ValueError("Poisson reference requires counts.")
                ylogy = value * value.ln() if count > 0 else Decimal(0)
                self.y_log_y[g] += ylogy
                self.row_y_log_y.append(ylogy)
                saturated += ylogy - value - _log_factorial(count, cache)
            self.poisson_saturated = saturated

    def _lambda(self, rho: float) -> Decimal:
        return _d(rho).exp() * self.scale

    def saturated(self, phi: float) -> Decimal:
        with localcontext() as context:
            context.prec = _DIGITS
            if self.regime == "indexed_gaussian":
                return -Decimal(self.n) * (2 * _PI * _d(phi)).ln() / 2
            return self.poisson_saturated

    def evaluate(
        self, beta: Sequence[Decimal], rho: float, phi: float = 1.0
    ) -> ReferenceState:
        with localcontext() as context:
            context.prec = _DIGITS
            if self.regime == "arrowhead_poisson":
                return self._arrowhead(beta, rho)
            lam = self._lambda(rho)
            deviance = penalty = log_det = inverse_sum = Decimal(0)
            for g in range(self.levels):
                b = beta[g]
                c = Decimal(int(self.count[g]))
                if self.regime == "indexed_poisson":
                    mu = b.exp()
                    deviance += 2 * (
                        self.y_log_y[g] - self.y_sum[g] * b - self.y_sum[g] + c * mu
                    )
                    diagonal = c * mu + lam
                else:
                    deviance += self.y_square[g] - 2 * b * self.y_sum[g] + c * b * b
                    diagonal = c + lam
                penalty += b * b
                log_det += diagonal.ln()
                inverse_sum += 1 / diagonal
            dispersion = _d(phi)
            score = (
                (deviance + lam * penalty) / (2 * dispersion)
                - self.saturated(phi)
                + (log_det - self.levels * lam.ln()) / 2
            )
            return ReferenceState(
                score,
                deviance,
                self.scale * inverse_sum,
                self.scale * penalty,
                self.levels - lam * inverse_sum,
            )

    def _arrowhead_rows(
        self, beta: Sequence[Decimal]
    ) -> tuple[list[Decimal], list[Decimal]]:
        modest = beta[: self.k]
        mu, eta = [], []
        for index in range(self.n):
            z = self._modest_row(index)
            value = (
                sum(modest[j] * z[j] for j in range(self.k))
                + beta[self.k + int(self.group[index])]
            )
            eta.append(value)
            mu.append(value.exp())
        return mu, eta

    def _modest_row(self, index: int) -> tuple[Decimal, ...]:
        if self.x is None:
            return (Decimal(1),)
        return (Decimal(1), _d(self.x[index]))

    def _arrowhead_system(self, beta: Sequence[Decimal], rho: float) -> dict[str, Any]:
        lam = self._lambda(rho)
        mu, eta = self._arrowhead_rows(beta)
        k = self.k
        A = [[Decimal(0)] * k for _ in range(k)]
        B = [[Decimal(0)] * k for _ in range(self.levels)]
        D = [Decimal(0)] * self.levels
        gradient_modest = [Decimal(0)] * k
        gradient_indexed = [Decimal(0)] * self.levels
        deviance = Decimal(0)
        for index in range(self.n):
            g = int(self.group[index])
            z = self._modest_row(index)
            m = mu[index]
            y = _d(self.y[index])
            deviance += 2 * (self.row_y_log_y[index] - y * eta[index] - y + m)
            for i in range(k):
                gradient_modest[i] += (m - y) * z[i]
                B[g][i] += m * z[i]
                for j in range(k):
                    A[i][j] += m * z[i] * z[j]
            D[g] += m
            gradient_indexed[g] += m - y
        for g in range(self.levels):
            D[g] += lam
            gradient_indexed[g] += lam * beta[k + g]
        schur = [
            [
                A[i][j] - sum(B[g][i] * B[g][j] / D[g] for g in range(self.levels))
                for j in range(k)
            ]
            for i in range(k)
        ]
        return {
            "lam": lam,
            "B": B,
            "D": D,
            "schur": schur,
            "gradient_modest": gradient_modest,
            "gradient_indexed": gradient_indexed,
            "deviance": deviance,
        }

    def _arrowhead(self, beta: Sequence[Decimal], rho: float) -> ReferenceState:
        system = self._arrowhead_system(beta, rho)
        lam, B, D, schur = system["lam"], system["B"], system["D"], system["schur"]
        k = self.k
        penalty = sum(b * b for b in beta[k:])
        log_det = sum(value.ln() for value in D) + _log_det_spd(schur)
        inverse_sum = Decimal(0)
        for g in range(self.levels):
            weighted = [B[g][i] / D[g] for i in range(k)]
            solved = _cholesky_solve(schur, weighted)
            inverse_sum += 1 / D[g] + sum(weighted[i] * solved[i] for i in range(k))
        score = (
            (system["deviance"] + lam * penalty) / 2
            - self.poisson_saturated
            + (log_det - self.levels * lam.ln()) / 2
            - Decimal(k) * (2 * _PI).ln() / 2
        )
        return ReferenceState(
            score,
            system["deviance"],
            self.scale * inverse_sum,
            self.scale * penalty,
            (self.levels + k) - lam * inverse_sum,
        )

    def optimum(self, beta_start: np.ndarray, rho: float) -> list[Decimal]:
        """Exact penalized optimum at rho by decimal Newton iteration."""
        tolerance = Decimal(10) ** (-(_DIGITS - 8))
        with localcontext() as context:
            context.prec = _DIGITS
            beta = [_d(value) for value in beta_start]
            lam = self._lambda(rho)
            if self.regime == "indexed_gaussian":
                return [
                    self.y_sum[g] / (Decimal(int(self.count[g])) + lam)
                    for g in range(self.levels)
                ]
            if self.regime == "indexed_poisson":
                for g in range(self.levels):
                    c = Decimal(int(self.count[g]))
                    b = beta[g]
                    for _ in range(200):
                        mu = b.exp()
                        step = (c * mu - self.y_sum[g] + lam * b) / (c * mu + lam)
                        b -= step
                        if abs(step) <= tolerance:
                            break
                    else:
                        raise ArithmeticError("Indexed reference Newton failed.")
                    beta[g] = b
                return beta
            k = self.k
            for _ in range(50):
                system = self._arrowhead_system(beta, rho)
                B, D = system["B"], system["D"]
                rhs_modest = [-value for value in system["gradient_modest"]]
                rhs_indexed = [-value for value in system["gradient_indexed"]]
                reduced = [
                    rhs_modest[i]
                    - sum(B[g][i] * rhs_indexed[g] / D[g] for g in range(self.levels))
                    for i in range(k)
                ]
                step_modest = _cholesky_solve(system["schur"], reduced)
                step = [*step_modest]
                for g in range(self.levels):
                    step.append(
                        (
                            rhs_indexed[g]
                            - sum(B[g][i] * step_modest[i] for i in range(k))
                        )
                        / D[g]
                    )
                beta = [value + delta for value, delta in zip(beta, step, strict=True)]
                if max(abs(delta) for delta in step) <= tolerance:
                    return beta
            raise ArithmeticError("Arrowhead reference Newton failed.")


def score_errors(
    reference: DecimalReference,
    beta: np.ndarray,
    rho: float,
    score: float,
    score_phi: float = 1.0,
) -> tuple[float, float, ReferenceState]:
    """Return operator score error at its own and at the exact coefficients."""
    with localcontext() as context:
        context.prec = _DIGITS
        own = reference.evaluate([_d(value) for value in beta], rho, score_phi)
        exact = reference.evaluate(reference.optimum(beta, rho), rho, score_phi)
        return (
            float(abs(_d(score) - own.score)),
            float(abs(_d(score) - exact.score)),
            exact,
        )
