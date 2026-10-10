"""Host-only bounded selection of an R-style family starting predictor.

This module deliberately does not route a public fit or modify the existing
stream scanner.  It is a replayable-source primitive for a later controller:
the selected result is only a shrink count and scan metadata, never a retained
``n``-row eta/mu/mustart array.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from jaxgam.data.source import RowBatch
from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting.family_execution import FamilyExecutionLineage
from jaxgam.formula.design_provider import StreamDesign

_MAX_INITIAL_SHRINKS = 20


@dataclass(frozen=True)
class InitialWorkingSelection:
    """A replayable first-working-state choice without row retention."""

    shrink_count: int
    input_ok: bool
    domain_ok: bool
    valid_rows: int
    source_scans: int


NullEtaFactory = Callable[[RowBatch], np.ndarray]


def _validate_before_scan(
    stream: StreamDesign,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
) -> None:
    lineage.validate(stream.prepared, family)
    if stream.source.fingerprint() != lineage.source_fingerprint:
        raise RuntimeError("RowSource changed after preparation; prepare again.")


def _scan_initial_state(
    stream: StreamDesign,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    batch_rows: int,
    shrink_count: int,
    null_eta_for_batch: NullEtaFactory,
) -> tuple[bool, bool, int]:
    """Replay one bounded candidate without retaining per-row state."""
    _validate_before_scan(stream, family, lineage)
    input_ok = True
    domain_ok = True
    valid_rows = 0
    for batch in stream.source.scan(batch_rows):
        lineage.validate(stream.prepared, family)
        if batch.y is None:
            raise ValueError("Initial working-state replay requires response values.")
        valid = np.asarray(batch.valid, dtype=bool)
        state = family.initial_working_state_cpu(batch.y, batch.weight, valid)
        input_ok = input_ok and state.input_ok
        valid_rows += int(np.sum(valid))
        if shrink_count:
            null_eta = np.asarray(null_eta_for_batch(batch), dtype=float)
            if null_eta.shape != valid.shape:
                raise ValueError(
                    "null_eta_for_batch must return one value per source row."
                )
            null_eta_safe = np.where(valid, null_eta, 0.0)
            # Reapply the literal recurrence, rather than a closed form, to
            # retain R's operation order while keeping only one batch alive.
            eta = state.eta
            for _ in range(shrink_count):
                eta = 0.9 * eta + 0.1 * null_eta_safe
            mu = np.asarray(family.link.inverse(eta), dtype=float)
            candidate_ok = (
                np.all(np.isfinite(null_eta[valid]))
                and np.all(np.isfinite(eta[valid]))
                and np.all(np.asarray(family.valid_eta(eta[valid]), dtype=bool))
                and np.all(np.isfinite(mu[valid]))
                and np.all(np.asarray(family.valid_mu(mu[valid]), dtype=bool))
            )
        else:
            candidate_ok = state.domain_ok
        domain_ok = domain_ok and bool(candidate_ok)
    _validate_before_scan(stream, family, lineage)
    if valid_rows != stream.prepared.n_obs:
        raise RuntimeError(
            "RowSource scan changed its valid row count after preparation; "
            "prepare again."
        )
    return input_ok, domain_ok, valid_rows


def select_initial_working_state_cpu(
    stream: StreamDesign,
    family: ExponentialFamily,
    lineage: FamilyExecutionLineage,
    *,
    batch_rows: int,
    null_eta_for_batch: NullEtaFactory,
) -> InitialWorkingSelection:
    """Find a globally valid R-style start by at most 20 .9/.1 shrinks.

    The supplied null predictor is intentionally a per-batch callback: this
    routine neither derives a null coefficient nor projects ``mustart`` onto
    a design matrix.  A family input error (for example Gamma ``y <= 0``) is
    returned fail-closed immediately.  A finite but invalid predictor is
    replayed at shrink counts 0 through 20, matching ``gam.fit3``'s bound.
    """
    if (
        not isinstance(batch_rows, int)
        or isinstance(batch_rows, bool)
        or batch_rows <= 0
    ):
        raise ValueError("batch_rows must be a positive integer.")
    for shrink_count in range(_MAX_INITIAL_SHRINKS + 1):
        input_ok, domain_ok, valid_rows = _scan_initial_state(
            stream,
            family,
            lineage,
            batch_rows,
            shrink_count,
            null_eta_for_batch,
        )
        scans = shrink_count + 1
        if not input_ok or domain_ok:
            return InitialWorkingSelection(
                shrink_count=shrink_count,
                input_ok=input_ok,
                domain_ok=domain_ok,
                valid_rows=valid_rows,
                source_scans=scans,
            )
    return InitialWorkingSelection(
        shrink_count=_MAX_INITIAL_SHRINKS,
        input_ok=input_ok,
        domain_ok=False,
        valid_rows=valid_rows,
        source_scans=_MAX_INITIAL_SHRINKS + 1,
    )
