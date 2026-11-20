"""Calibrate structured operator EFS score envelopes beyond their release.

Each case generates deterministic rows for one structured operator EFS regime
and measures the operator implementation against an independent 50-digit
decimal evaluation of the same model:

* matched states: operator score, Fisher trace, penalty quadratic, EDF and
  coefficients at fixed smoothing parameters. The score error is split into
  an arithmetic part (decimal score at the operator's own coefficients) and a
  total part (decimal score at the exact penalized optimum), and each is
  compared with the envelope term that is meant to cover it;
* dense EFS at the same matched states, which isolates the dense route's
  jittered log determinant from operator implementation error;
* complete fits: operator EFS (admitted bound, or the unadmitted formula
  outside the release) against dense EFS. Score, smoothing parameters,
  deviance, coefficients, EDF, iterations and stop reasons are recorded.
  Differences from Newton REML are a method difference and are not measured
  here.

The envelope constants are not fitted here: each report states the measured
error and the bound it is compared with, so held-out sweeps can be read
without re-deriving the formula. Run one regime/size grid per process, e.g.::

    make benchmark-operator-efs-calibration \\
        ARGS="--regime indexed_poisson --levels 1024 2048 --seeds 41 42" \\
        OUTPUT=/tmp/jaxgam-operator-efs-calibration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import jaxgam
from jaxgam.control import EFSControl, OperatorControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.efs import (
    _fit_state,
    dense_efs_known_scale,
    dense_efs_unknown_scale,
)
from jaxgam.execution.operator.efs_controller import (
    OperatorEFSState,
    SourceOperatorEFSFitProvider,
    admitted_operator_efs_score_bound,
    operator_efs_score_envelope,
    run_operator_efs_gaussian_scale,
    run_operator_efs_known_scale,
)
from jaxgam.execution.operator.logdet import OperatorLogDetControl
from jaxgam.families.standard import Gaussian, Poisson
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.efs import prepare_efs_statistics
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.operator.prepare import (
    OperatorPreparationControl,
    prepare_operator_model,
)
from jaxgam.formula.parser import parse_formula
from tests.operator_efs_calibration import (
    _DIGITS,
    _GAUSSIAN_NOISE,
    REGIMES,
    CalibrationCase,
    DecimalReference,
    _d,
    generate_frame,
)
from tests.tolerances import STRICT

SCHEMA = "jaxgam-efs6-operator-calibration-v1"


def _family(case: CalibrationCase):
    return Gaussian() if case.regime == "indexed_gaussian" else Poisson()


@dataclass(frozen=True)
class HarnessControls:
    """Operator/EFS controls shared by matched and complete measurements."""

    operator: OperatorControl
    efs: EFSControl
    # The public operator EFS route's total-budget split.
    preparation_budget_bytes: int = 64 << 20
    fit_budget_bytes: int = 128 << 20
    components_budget_bytes: int = 256 << 20
    aggregate_budget_bytes: int = 512 << 20


def default_controls(case: CalibrationCase) -> HarnessControls:
    tolerance = 1e-7 if case.regime == "indexed_gaussian" else 1e-11
    return HarnessControls(
        OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
        EFSControl(outer_limit=50, pirls_tolerance=tolerance),
    )


class CalibrationModel:
    """Operator model, decimal reference and optional dense comparator."""

    def __init__(
        self,
        case: CalibrationCase,
        controls: HarnessControls,
        *,
        dense: bool,
    ) -> None:
        self.case = case
        self.controls = controls
        self.frame = generate_frame(case)
        self.family = _family(case)
        self.spec = parse_formula(case.formula)
        self.source = DataFrameRowSource(self.frame, response="y")
        self.model = prepare_operator_model(
            self.spec,
            self.source,
            control=OperatorPreparationControl(
                batch_rows=case.batch_rows,
                memory_budget_bytes=controls.preparation_budget_bytes,
            ),
        )
        effect = self.model.random_effects[0]
        modest = tuple(self.model.modest_columns)
        if modest != tuple(range(len(modest))) or effect.col_start != len(modest):
            raise ValueError("Calibration expects modest columns before levels.")
        block = self.model.penalties.blocks[0]
        self.penalty_scale = float(block.local_penalties[0].scale)
        self.reference = DecimalReference(
            case, self.frame, self.penalty_scale, len(modest)
        )
        self.dense_data = (
            FittingData.from_setup(ModelSetup.build(self.spec, self.frame), self.family)
            if dense
            else None
        )

    def provider(
        self,
        score_error_bound: Callable | None = None,
        provider_class: type[SourceOperatorEFSFitProvider] = (
            SourceOperatorEFSFitProvider
        ),
    ) -> SourceOperatorEFSFitProvider:
        return provider_class(
            self.model,
            self.source,
            self.family,
            fit_control=self.controls.operator,
            fit_budget_bytes=self.controls.fit_budget_bytes,
            components_budget_bytes=self.controls.components_budget_bytes,
            aggregate_budget_bytes=self.controls.aggregate_budget_bytes,
            score_error_bound=score_error_bound,
        )

    def accuracy(self) -> OperatorLogDetControl:
        return OperatorLogDetControl(probes=4, lanczos_depth=4, seed=self.case.seed)


def _abs_error(value: float, reference: Decimal) -> float:
    with localcontext() as context:
        context.prec = _DIGITS
        return float(abs(_d(value) - reference))


def _relative_error(value: float, reference: Decimal) -> float:
    with localcontext() as context:
        context.prec = _DIGITS
        return float(abs(_d(value) - reference) / max(abs(reference), Decimal(1)))


def _ratio(error: float, bound: float | None) -> float | None:
    if bound is None or bound <= 0:
        return None
    return error / bound


def audit_state(model: CalibrationModel, state: OperatorEFSState) -> dict[str, Any]:
    """Measure one operator state against the decimal reference."""
    fit, components = state.fit, state.components
    rho = float(fit.log_lambda[0])
    reference = model.reference
    with localcontext() as context:
        context.prec = _DIGITS
        beta_operator = [_d(value) for value in fit.coefficients]
    at_operator = reference.evaluate(beta_operator, rho, state.score_phi)
    optimum = reference.optimum(fit.coefficients, rho)
    at_optimum = reference.evaluate(optimum, rho, state.score_phi)
    envelope = operator_efs_score_envelope(fit, components)
    admitted, method = admitted_operator_efs_score_bound(
        fit, components, model.controls.operator
    )
    score = components.score.value
    arithmetic_error = _abs_error(score, at_operator.score)
    total_error = _abs_error(score, at_optimum.score)
    with localcontext() as context:
        context.prec = _DIGITS
        coefficient_score_error = float(abs(at_operator.score - at_optimum.score))
        beta_error = float(
            max(
                abs(_d(value) - exact)
                for value, exact in zip(fit.coefficients, optimum, strict=True)
            )
        )
    terms = asdict(envelope) if envelope is not None else None
    formula_bound = envelope.total if envelope is not None else None
    gradient = fit.diagnostics.final_gradient_norm
    lower = components.score.spectral_lower_estimate
    row_l1 = components.score_rows.maximum_row_l1_norm
    candidates = (
        None
        if lower is None
        else {
            "first_order_p": fit.model.n_coef * gradient / lower,
            "first_order_edf": components.edf_total
            * row_l1
            * math.sqrt(fit.model.n_coef)
            * gradient
            / lower,
        }
    )
    return {
        "log_lambda": rho,
        "score": score,
        "score_phi": state.score_phi,
        "admitted_bound": admitted,
        "admitted_method": method,
        "controller_bound": state.score_error_bound,
        "controller_method": state.score_bound_method,
        "envelope": terms,
        "envelope_total": formula_bound,
        "errors": {
            "score_arithmetic": arithmetic_error,
            "score_coefficient": coefficient_score_error,
            "score_total": total_error,
            "coefficient_max_abs": beta_error,
            "trace_relative": _relative_error(
                float(components.fisher.traces[0]), at_optimum.trace
            ),
            "quadratic_relative": _relative_error(
                float(components.fisher.quadratics[0]), at_optimum.quadratic
            ),
            "edf_abs": _abs_error(components.edf_total, at_optimum.edf),
            "deviance_relative": _relative_error(fit.deviance, at_optimum.deviance),
        },
        "ratios": {
            "total_to_bound": _ratio(total_error, formula_bound),
            "arithmetic_to_term": _ratio(
                arithmetic_error, None if terms is None else terms["arithmetic"]
            ),
            "coefficient_to_term": _ratio(
                coefficient_score_error,
                None if terms is None else terms["coefficient"],
            ),
        },
        "candidate_coefficient_terms": candidates,
        "edf_total": components.edf_total,
        "conditioning": {
            "gradient_norm": fit.diagnostics.final_gradient_norm,
            "diagonal_condition_ratio": components.score.diagonal_condition_ratio,
            "minimum_diagonal": components.score.minimum_diagonal,
            "spectral_lower_estimate": components.score.spectral_lower_estimate,
            "spectral_condition_estimate": (
                components.score.spectral_condition_estimate
            ),
            "max_abs_coefficient": float(np.max(np.abs(fit.coefficients))),
            "maximum_response": components.score_rows.maximum_response,
            "maximum_absolute_response": (
                components.score_rows.maximum_absolute_response
            ),
        },
    }


def _dense_log_determinants(hessian: np.ndarray) -> tuple[float, float]:
    """Plain and shared-route (unit-diagonal, 1e-12 jittered) log|H|."""
    plain = 2.0 * float(np.sum(np.log(np.diag(np.linalg.cholesky(hessian)))))
    diagonal = np.sqrt(np.maximum(np.diag(hessian), np.finfo(np.float64).tiny))
    scaled = hessian / np.outer(diagonal, diagonal)
    jitter = max(1e-12 * float(np.trace(scaled)) / len(scaled), 1e-14)
    lower = np.linalg.cholesky(scaled + jitter * np.eye(len(scaled)))
    shared = 2.0 * float(np.sum(np.log(np.diag(lower)))) + 2.0 * float(
        np.sum(np.log(diagonal))
    )
    return plain, shared


def _dense_matched(
    model: CalibrationModel, rho: float, phi: float, operator: OperatorEFSState
) -> dict[str, Any]:
    """Dense EFS at the same state; separates its jittered log determinant."""
    from jaxgam.fitting import penalty_ops

    data = model.dense_data
    assert data is not None
    phi = 1.0 if model.family.scale_known else phi
    state = _fit_state(
        data,
        prepare_efs_statistics(data),
        np.array([rho]),
        np.zeros(data.n_coef),
        replace(model.controls.efs, pirls_tolerance=1e-11),
        None if model.family.scale_known else np.array(phi),
        regular_start_present=False,
    )
    beta = np.asarray(state.pirls_result.coefficients)
    with localcontext() as context:
        context.prec = _DIGITS
        exact = model.reference.evaluate([_d(value) for value in beta], rho, phi)
    hessian = np.asarray(state.pirls_result.XtWX) + np.asarray(
        penalty_ops.materialize(data.penalty_structure, np.array([rho]))
    )
    plain, shared = _dense_log_determinants(hessian)
    dense_score = float(np.asarray(state.score))
    jitter_shift = 0.5 * (shared - plain)
    return {
        "valid": bool(state.valid),
        "score_error": _abs_error(dense_score, exact.score),
        "jitter_score_shift": jitter_shift,
        "score_error_without_jitter": _abs_error(
            dense_score - jitter_shift, exact.score
        ),
        "operator_minus_dense_score": operator.components.score.value - dense_score,
        "operator_minus_dense_beta_max_abs": float(
            np.max(np.abs(operator.fit.coefficients - beta))
        ),
        "operator_minus_dense_trace_relative": float(
            abs(
                operator.components.fisher.traces[0]
                - float(np.asarray(state.statistics.fisher_trace)[0])
            )
            / max(1.0, abs(float(operator.components.fisher.traces[0])))
        ),
    }


def matched_states(
    model: CalibrationModel, smoothing: Sequence[float], phi: float
) -> list[dict[str, Any]]:
    provider = model.provider()
    rows = []
    beta = None
    for sp in smoothing:
        rho = math.log(sp)
        fit = provider.fit(np.array([rho]), beta)
        if not fit.converged:
            rows.append({"log_lambda": rho, "operator_converged": False})
            continue
        if model.family.scale_known:
            state = provider.evaluate(fit, model.accuracy())
        else:
            state = provider.evaluate(fit, model.accuracy(), score_phi=phi)
        row = audit_state(model, state)
        row["operator_converged"] = True
        if model.dense_data is not None:
            row["dense"] = _dense_matched(model, rho, phi, state)
        rows.append(row)
        beta = fit.coefficients
    return rows


def _strict_close(actual: float | np.ndarray, expected: float | np.ndarray) -> bool:
    return bool(
        np.allclose(actual, expected, rtol=STRICT.rtol, atol=STRICT.atol)
        and np.shape(actual) == np.shape(expected)
    )


def _relative_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))


def complete_fit(
    model: CalibrationModel, *, bound_source: str, initial_sp: float = 0.8
) -> dict[str, Any]:
    """Complete operator EFS with every evaluated state audited."""
    audits: list[dict[str, Any]] = []
    harness = model

    class _AuditedProvider(SourceOperatorEFSFitProvider):
        def evaluate(self, fit, control, *, score_phi=1.0):
            state = super().evaluate(fit, control, score_phi=score_phi)
            audits.append(audit_state(harness, state))
            return state

    def formula_bound(fit, components):
        envelope = operator_efs_score_envelope(fit, components)
        return None if envelope is None else envelope.total

    provider = model.provider(
        formula_bound if bound_source == "formula" else None, _AuditedProvider
    )
    initial = np.log(np.array([initial_sp]))
    started = time.perf_counter()
    if model.family.scale_known:
        operator = run_operator_efs_known_scale(
            provider,
            initial,
            control=model.controls.efs,
            initial_accuracy=model.accuracy(),
        )
    else:
        operator = run_operator_efs_gaussian_scale(
            provider,
            initial,
            control=model.controls.efs,
            initial_accuracy=model.accuracy(),
        )
    operator_seconds = time.perf_counter() - started
    selected = operator.accepted
    errors = [row["errors"]["score_total"] for row in audits]
    formula = [row["envelope_total"] for row in audits]
    admitted = [row["admitted_bound"] for row in audits]
    result: dict[str, Any] = {
        "bound_source": bound_source,
        "operator": {
            "converged": operator.converged,
            "stop_reason": operator.stop_reason,
            "outer_iterations": operator.outer_iterations,
            "attempted_fits": operator.attempted_fits,
            "indeterminate_comparisons": operator.indeterminate_comparisons,
            "score": selected.components.score.value,
            "smoothing": float(np.exp(selected.fit.log_lambda[0])),
            "deviance": selected.fit.deviance,
            "edf_total": selected.components.edf_total,
            "reported_phi": selected.reported_phi,
            "final_bound": selected.score_error_bound,
            "final_bound_method": selected.score_bound_method,
            "seconds": operator_seconds,
            "source_scans": provider.completed_source_scans,
            "aggregate_peak_bound_bytes": provider.aggregate_peak_bound_bytes,
        },
        "states": len(audits),
        "accepted_score_steps": [
            abs(after - before)
            for before, after in zip(
                operator.score_history[:-1], operator.score_history[1:], strict=True
            )
        ],
        "accepted_bounds": list(operator.score_bound_history),
        "max_total_error": max(errors) if errors else None,
        "max_total_to_formula": max(
            (row["ratios"]["total_to_bound"] or 0.0) for row in audits
        )
        if audits
        else None,
        "all_states_admitted": all(bound is not None for bound in admitted),
        "admitted_violations": sum(
            bound is not None and error > bound
            for error, bound in zip(errors, admitted, strict=True)
        ),
        "formula_violations": sum(
            bound is not None and error > bound
            for error, bound in zip(errors, formula, strict=True)
        ),
        "audits": audits,
    }
    if model.dense_data is None:
        result["dense"] = None
        return result
    started = time.perf_counter()
    if model.family.scale_known:
        dense = dense_efs_known_scale(
            model.dense_data, initial_log_lambda=initial, control=model.controls.efs
        )
    else:
        dense = dense_efs_unknown_scale(
            model.dense_data, initial_log_lambda=initial, control=model.controls.efs
        )
    dense_seconds = time.perf_counter() - started
    dense_beta = np.asarray(dense.pirls_result.coefficients)
    dense_rho = float(np.asarray(dense.log_lambda)[0])
    dense_phi = (
        1.0
        if model.family.scale_known
        else float(np.asarray(dense.score_phi if dense.score_phi is not None else 1.0))
    )
    with localcontext() as context:
        context.prec = _DIGITS
        dense_exact = model.reference.evaluate(
            [_d(value) for value in dense_beta], dense_rho, dense_phi
        )
    dense_score = float(np.asarray(dense.score))
    dense_edf = float(np.sum(np.asarray(dense.edf)))
    fields = {
        "score": (selected.components.score.value, dense_score),
        "smoothing": (
            np.exp(selected.fit.log_lambda),
            np.asarray(dense.smoothing_params),
        ),
        "deviance": (selected.fit.deviance, float(dense.pirls_result.deviance)),
        "coefficients": (selected.fit.coefficients, dense_beta),
        "edf_total": (selected.components.edf_total, dense_edf),
    }
    if not model.family.scale_known:
        fields["reported_phi"] = (selected.reported_phi, float(np.asarray(dense.scale)))
    history = (tuple(operator.score_history), tuple(dense.score_history))
    result["dense"] = {
        "converged": bool(dense.converged),
        "convergence_info": dense.convergence_info,
        "outer_iterations": int(dense.n_iter),
        "seconds": dense_seconds,
        "score_error_vs_decimal": _abs_error(dense_score, dense_exact.score),
        "same_outer_iterations": operator.outer_iterations == int(dense.n_iter),
        "strict": {name: _strict_close(*pair) for name, pair in fields.items()},
        "relative_difference": {
            name: _relative_difference(*pair) for name, pair in fields.items()
        },
        "score_history_max_abs_difference": (
            float(np.max(np.abs(np.subtract(*history))))
            if len(history[0]) == len(history[1]) and history[0]
            else None
        ),
    }
    return result


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024 * value)


def _runtime() -> dict[str, Any]:
    import jax

    def git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    explicit = os.environ.get("JAXGAM_SOURCE_COMMIT")
    status = None if explicit else git("status", "--porcelain")
    return {
        "commit": explicit or git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "loaded_package": str(Path(jaxgam.__file__).resolve()),
        "thread_env": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def run_case(
    case: CalibrationCase,
    *,
    smoothing: Sequence[float],
    score_phi: float,
    complete: bool,
    dense_max_coefficients: int,
    bound_source: str,
    controls: HarnessControls | None = None,
) -> dict[str, Any]:
    """Measure one generated case; the report carries no raw rows."""
    started = time.perf_counter()
    controls = default_controls(case) if controls is None else controls
    model = CalibrationModel(
        case, controls, dense=case.levels <= dense_max_coefficients
    )
    matched = matched_states(model, smoothing, score_phi)
    matched_seconds = time.perf_counter() - started
    report = {
        "schema": SCHEMA,
        "case": {
            **asdict(case),
            "n_rows": case.n_rows,
            "n_coef": model.model.n_coef,
            "formula": case.formula,
            "base_amplitude": list(case.shape),
            "frame_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(model.frame, index=True).to_numpy().tobytes()
            ).hexdigest(),
            "penalty_scale": model.penalty_scale,
        },
        "controls": {
            "operator": asdict(controls.operator),
            "efs": asdict(controls.efs),
        },
        "matched": matched,
        "matched_seconds": matched_seconds,
        "complete": (
            complete_fit(model, bound_source=bound_source) if complete else None
        ),
        "seconds": time.perf_counter() - started,
        "rss_high_water_bytes": _peak_rss_bytes(),
    }
    return report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    case = report["case"]
    matched = [row for row in report["matched"] if row.get("operator_converged")]
    summary = {
        "regime": case["regime"],
        "levels": case["levels"],
        "n_rows": case["n_rows"],
        "seed": case["seed"],
        "batch_rows": case["batch_rows"],
        "role": case["role"],
        "matched_max_total_to_bound": max(
            (row["ratios"]["total_to_bound"] or 0.0) for row in matched
        )
        if matched
        else None,
        "matched_max_arithmetic_to_term": max(
            (row["ratios"]["arithmetic_to_term"] or 0.0) for row in matched
        )
        if matched
        else None,
        "matched_admitted": [row["admitted_bound"] is not None for row in matched],
        "seconds": report["seconds"],
    }
    complete = report["complete"]
    if complete is not None:
        summary["complete"] = {
            "converged": complete["operator"]["converged"],
            "stop_reason": complete["operator"]["stop_reason"],
            "outer": complete["operator"]["outer_iterations"],
            "max_total_to_formula": complete["max_total_to_formula"],
            "formula_violations": complete["formula_violations"],
            "admitted_violations": complete["admitted_violations"],
            "dense_strict": None
            if complete["dense"] is None
            else complete["dense"]["strict"],
            "dense_outer": None
            if complete["dense"] is None
            else complete["dense"]["outer_iterations"],
        }
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    parser.add_argument("--multiplicity", type=float, nargs="+", default=[2.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[41])
    parser.add_argument("--batch-rows", type=int, nargs="+", default=[64])
    parser.add_argument(
        "--mode", choices=("balanced", "unbalanced"), default="balanced"
    )
    parser.add_argument("--base", type=float, default=None)
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--modest", choices=("none", "x", "intercept"), default=None)
    parser.add_argument(
        "--role", choices=("calibration", "held_out", "boundary"), default="calibration"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        nargs="+",
        default=[0.8 * math.exp(2.5), 0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--score-phi", type=float, default=_GAUSSIAN_NOISE**2)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--dense-max-coefficients", type=int, default=1536)
    parser.add_argument(
        "--bound-source", choices=("admitted", "formula"), default="admitted"
    )
    parser.add_argument("--operator-tolerance", type=float, default=1e-11)
    parser.add_argument("--cg-rtol", type=float, default=1e-13)
    parser.add_argument("--cg-atol", type=float, default=1e-15)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/jaxgam-operator-efs-calibration"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    options = _parser().parse_args(argv)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    label = options.label or f"{options.regime}-{options.mode}-{options.role}"
    output = options.output_dir / f"{label}.jsonl"
    runtime = _runtime()
    for levels in options.levels:
        for multiplicity in options.multiplicity:
            for batch_rows in options.batch_rows:
                for seed in options.seeds:
                    case = CalibrationCase(
                        options.regime,
                        levels,
                        multiplicity,
                        seed,
                        options.mode,
                        batch_rows,
                        options.base,
                        options.amplitude,
                        options.role,
                        options.modest,
                    )
                    base = default_controls(case)
                    controls = replace(
                        base,
                        operator=OperatorControl(
                            tolerance=options.operator_tolerance,
                            cg_rtol=options.cg_rtol,
                            cg_atol=options.cg_atol,
                        ),
                    )
                    report = run_case(
                        case,
                        smoothing=options.smoothing,
                        score_phi=options.score_phi,
                        complete=options.complete,
                        dense_max_coefficients=options.dense_max_coefficients,
                        bound_source=options.bound_source,
                        controls=controls,
                    )
                    report["runtime"] = runtime
                    with output.open("a") as handle:
                        handle.write(json.dumps(report, allow_nan=False) + "\n")
                    print(json.dumps(_summary(report)), flush=True)


if __name__ == "__main__":
    main()
