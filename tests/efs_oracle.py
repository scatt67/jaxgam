"""Reproducible, pinned-source EFS oracle fixtures and controller traces.

This module is deliberately test infrastructure.  It models the outer-controller
bookkeeping in mgcv 1.9-3 ``efsudr`` so later JAX code can be compared against
branch-forced traces without making a Python EFS optimizer public.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

PINNED_MGCV_COMMIT = "fb7e8e718377513e78ba6c6bf7e60757fc6a32a9"
PINNED_MGCV_VERSION = "1.9-3"
PINNED_R_VERSION = "4.5.2"


@dataclass(frozen=True)
class EFSFixture:
    """Portable input/provenance record for one pinned mgcv EFS fixture."""

    formula: str
    family: str
    seed: int
    data_hash: str
    controls: dict[str, float | int]
    source_commit: str = PINNED_MGCV_COMMIT
    mgcv_version: str = PINNED_MGCV_VERSION
    r_version: str = PINNED_R_VERSION
    weights_column: str = "w"
    offset_column: str = "offset"
    fixed_parameters: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata without losing provenance fields."""
        return asdict(self)


def canonical_data_hash(data: pd.DataFrame) -> str:
    """Hash a fixture frame with stable rows, column order, and float formatting."""
    payload = data.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_efs_fixture(
    *, seed: int = 20260907, n: int = 96, family: str = "poisson"
) -> tuple[pd.DataFrame, EFSFixture]:
    """Create deterministic smooth-model data and its complete fixture metadata.

    The data includes prior weights and an offset so bridge callers can exercise
    both inputs without creating different ad-hoc generators in every test.
    """
    if n < 8:
        raise ValueError("n must be at least 8")
    if family not in {"poisson", "gaussian", "binomial", "gamma"}:
        raise ValueError("family must be poisson, gaussian, binomial, or gamma")

    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    offset = rng.normal(scale=0.08, size=n)
    weights = 0.5 + rng.uniform(size=n)
    eta = 0.25 + 0.75 * np.sin(np.pi * x) - 0.3 * z + offset
    if family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family == "binomial":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    elif family == "gamma":
        y = rng.gamma(shape=3.0, scale=np.exp(eta) / 3.0)
    else:
        y = eta + rng.normal(scale=0.35, size=n)
    data = pd.DataFrame({"y": y, "x": x, "z": z, "w": weights, "offset": offset})
    controls: dict[str, float | int] = {
        "efs_lspmax": 15.0,
        "efs_tol": 0.1,
        "outer_limit": 200,
    }
    fixture = EFSFixture(
        formula="y ~ s(x, k=8) + s(z, k=7)",
        family=family,
        seed=seed,
        data_hash=canonical_data_hash(data),
        controls=controls,
    )
    return data, fixture


def write_efs_fixture(output: Path, data: pd.DataFrame, fixture: EFSFixture) -> None:
    """Write CSV data and adjacent JSON provenance for reproducible R fixtures."""
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(fixture.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@dataclass(frozen=True)
class EFSControl:
    """Pinned mgcv 1.9-3 EFS outer-loop constants for branch forcing."""

    lspmax: float = 15.0
    score_tolerance: float = 0.1
    outer_limit: int = 200
    # ``efsudr`` uses control$eps (gam.control's epsilon default), not epsmach.
    deviance_epsilon: float = 1e-7


DEFAULT_EFS_CONTROL = EFSControl()


@dataclass(frozen=True)
class ScriptedFit:
    """Only the state ``efsudr`` reads from one recomputed coefficient fit."""

    score: float
    deviance: float
    scale: float = 1.0


@dataclass(frozen=True)
class EFSControllerEvent:
    """A proposal/refit event recorded with mgcv branch timing."""

    phase: Literal["initial", "candidate", "extension", "contraction"]
    iteration: int
    multiplier: float
    log_smoothing: tuple[float, ...]
    score: float
    deviance: float


@dataclass(frozen=True)
class EFSControllerTrace:
    """Result of a branch-forced ``efsudr`` controller trace."""

    events: tuple[EFSControllerEvent, ...]
    accepted_log_smoothing: tuple[float, ...]
    accepted_score: float
    multiplier: float
    stop_reason: Literal["score_window", "deviance", "iteration_limit"]


FitProvider = Callable[[str, np.ndarray, int, float], ScriptedFit]


def run_scripted_efs_controller(
    initial_log_smoothing: np.ndarray,
    log_ratio: np.ndarray,
    provider: FitProvider,
    *,
    control: EFSControl = DEFAULT_EFS_CONTROL,
) -> EFSControllerTrace:
    """Execute the pinned ``efsudr`` outer policy against a scripted refit provider.

    ``provider`` receives a phase, proposed log smoothing parameters, iteration,
    and multiplier.  It allows tests to force every controller branch while the
    real R diagnostic separately supplies the penalty statistics and PIRLS fits.
    The returned trace intentionally preserves finite score increases at a
    multiplier of one, matching pinned mgcv rather than Newton's policy.
    """
    lsp = np.asarray(initial_log_smoothing, dtype=float).copy()
    ratio_log = np.asarray(log_ratio, dtype=float)
    if lsp.ndim != 1 or ratio_log.shape != lsp.shape:
        raise ValueError(
            "initial_log_smoothing and log_ratio must be equal-length vectors"
        )
    if not np.all(np.isfinite(lsp)) or not np.all(np.isfinite(ratio_log)):
        raise ValueError("scripted EFS parameters must be finite")
    if control.outer_limit < 1:
        raise ValueError("outer_limit must be positive")

    events: list[EFSControllerEvent] = []

    def refit(
        phase: str, values: np.ndarray, iteration: int, multiplier: float
    ) -> ScriptedFit:
        fit = provider(phase, values.copy(), iteration, multiplier)
        if not np.isfinite(fit.score) or not np.isfinite(fit.deviance):
            raise ValueError(
                "scripted fit provider must return finite score and deviance"
            )
        events.append(
            EFSControllerEvent(
                phase=phase,  # type: ignore[arg-type]
                iteration=iteration,
                multiplier=multiplier,
                log_smoothing=tuple(values),
                score=fit.score,
                deviance=fit.deviance,
            )
        )
        return fit

    # efsudr shifts smoothing parameters once before its first gam.fit3 call.
    lsp = lsp + 2.5
    fit = refit("initial", lsp, 0, 1.0)
    multiplier = 1.0
    score_history: list[float] = []
    old_deviance: float | None = None

    for iteration in range(1, control.outer_limit + 1):
        old_score = fit.score
        candidate = np.minimum(lsp + ratio_log * multiplier, control.lspmax)
        # Pinned code saves max.step before extension/contraction selection.
        original_max_step = float(np.max(np.abs(candidate - lsp)))
        fit = refit("candidate", candidate, iteration, multiplier)

        if fit.score <= old_score:
            if original_max_step < 0.05:
                extension = np.minimum(
                    lsp + ratio_log * multiplier * 2.0, control.lspmax
                )
                extended_fit = refit(
                    "extension", extension, iteration, multiplier * 2.0
                )
                if extended_fit.score < fit.score:
                    fit = extended_fit
                    lsp = extension
                    multiplier *= 2.0
                else:
                    lsp = candidate
            else:
                lsp = candidate
        else:
            # EFS never contracts below one. A remaining finite worsening is accepted.
            while fit.score > old_score and multiplier > 1.0:
                multiplier /= 2.0
                candidate = np.minimum(lsp + ratio_log * multiplier, control.lspmax)
                fit = refit("contraction", candidate, iteration, multiplier)
            lsp = candidate
            multiplier = max(multiplier, 1.0)

        score_history.append(fit.score)
        if (
            iteration > 3
            and original_max_step < 0.05
            and max(abs(np.diff(score_history[-4:]))) < control.score_tolerance
        ):
            return EFSControllerTrace(
                tuple(events), tuple(lsp), fit.score, multiplier, "score_window"
            )
        if old_deviance is not None and abs(old_deviance - fit.deviance) < (
            100.0 * control.deviance_epsilon * abs(fit.deviance)
        ):
            return EFSControllerTrace(
                tuple(events), tuple(lsp), fit.score, multiplier, "deviance"
            )
        old_deviance = fit.deviance

    return EFSControllerTrace(
        tuple(events), tuple(lsp), fit.score, multiplier, "iteration_limit"
    )


def run_pinned_r_scripted_efs(
    initial_rho: np.ndarray, log_ratio: np.ndarray, fits: list[ScriptedFit]
) -> dict[str, object]:
    """Run installed pinned R ``efsudr`` with a private scripted ``gam.fit3``.

    The fake fitter supplies exactly the values read by ``efsudr`` while its
    penalty roots make each smoothing parameter's update ratio ``exp(log_ratio)``.
    This keeps the controller itself inside the installed, version-gated R code.
    """
    from tests.r_bridge import RBridge

    RBridge._require_pinned_efs_versions()
    if initial_rho.ndim != 1 or log_ratio.shape != initial_rho.shape:
        raise ValueError("initial_rho and log_ratio must be equal-length vectors")
    if not fits:
        raise ValueError("fits must contain at least one scripted fit")

    n_parameters = len(initial_rho)
    rho = ",".join(repr(float(x)) for x in initial_rho)
    ratio = ",".join(repr(float(x)) for x in log_ratio)
    scores = ",".join(repr(f.score) for f in fits)
    devs = ",".join(repr(f.deviance) for f in fits)
    source = RBridge._pinned_efsudr_source()
    script = "\n".join(
        [
            source,
            "e <- new.env(parent=environment(efsudr))",
            "e$i <- 0L",
            "e$rho <- list()",
            "e$gam.fit3 <- function(...) {",
            "  a <- list(...)",
            "  e$i <- e$i + 1L",
            "  e$rho[[e$i]] <- a$sp",
            f"  j <- min(e$i, {len(fits)}L)",
            "  list(",
            "    coefficients=rep(1, length(a$sp)),",
            "    rV=matrix(0, length(a$sp), length(a$sp)),",
            f"    ldetS1=exp(a$sp + c({ratio})),",
            "    scale=1,",
            f"    REML=c({scores})[j],",
            f"    dev=c({devs})[j]",
            "  )",
            "}",
            "environment(efsudr) <- e",
            f"m <- {n_parameters}L",
            "o <- efsudr(",
            "  x=matrix(1, 3, m),",
            "  y=rep(1, 3),",
            f"  lsp=c({rho}),",
            "  Eb=diag(m),",
            "  UrS=lapply(seq_len(m), function(i) diag(m)[, i, drop=FALSE]),",
            "  weights=rep(1, 3),",
            "  family=poisson(),",
            "  U1=diag(m),",
            "  Mp=0,",
            "  control=mgcv::gam.control()",
            ")",
            "write.csv(do.call(rbind, e$rho), 'rho.csv', row.names=FALSE)",
            "write.csv(",
            "  data.frame(sp=as.numeric(o$sp)),",
            "  'accepted_sp.csv',",
            "  row.names=FALSE",
            ")",
            "write.csv(",
            "  data.frame(score=as.numeric(o$outer.info$score.hist)),",
            "  'score_history.csv',",
            "  row.names=FALSE",
            ")",
            "writeLines(paste(o$iter, o$outer.info$conv, o$REML, sep='|'), 'out.txt')",
            "",
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "x.R"
        path.write_text(script)
        subprocess.run(
            ["Rscript", str(path)],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        proposals = pd.read_csv(Path(directory) / "rho.csv").to_numpy()
        accepted_sp = pd.read_csv(Path(directory) / "accepted_sp.csv")["sp"].to_numpy(
            dtype=np.float64
        )
        score_history = pd.read_csv(Path(directory) / "score_history.csv")[
            "score"
        ].to_numpy(dtype=np.float64)
        iteration, convergence, score = (
            (Path(directory) / "out.txt").read_text().split("|")
        )
    return {
        "proposals": proposals,
        "accepted_sp": accepted_sp,
        "final_score": float(score),
        "score_history": score_history,
        "iter": int(iteration),
        "convergence": convergence,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible EFS oracle fixture"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="fixture CSV output path"
    )
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--family", default="poisson")
    args = parser.parse_args()
    data, fixture = make_efs_fixture(seed=args.seed, n=args.n, family=args.family)
    write_efs_fixture(args.output, data, fixture)


if __name__ == "__main__":
    _main()
