"""Observation-independent accepted/trial interface for source-faithful EFS.

The controller owns parameter state and trial policy. Providers own fitting
inputs, frozen coordinate/family lineage, reconvergence and scalar reductions.
This seam does not enable a new public backend or change dense initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import jax

if TYPE_CHECKING:
    from jaxgam.execution.efs import EFSFitState


@dataclass(frozen=True)
class EFSFitRequest:
    """Immutable controller-owned values supplied to one reconverged fit.

    Optional start flags use None to retain the dense adapter's established
    call shape/defaults. All parameter/coordinate arrays are immutable JAX
    arrays; nuisance state always belongs to the saved accepted trial origin.
    """

    log_lambda: jax.Array
    beta_start: jax.Array
    score_phi: jax.Array | None = None
    log_theta_start: jax.Array | None = None
    beta_old_init: jax.Array | None = None
    start_is_absent: bool | None = None
    regular_start_present: bool | None = None


@dataclass(frozen=True)
class EFSControllerContext:
    """Small immutable controller metadata, without fitting rows or family.

    Providers validate their own coordinate/parameter lineage. Prepared
    providers record source/basis fingerprints here for trace provenance;
    dense callers retain their existing family/coordinate checks externally.
    """

    estimated_theta: bool = False
    fixed_theta: float | None = None
    reference_profile: str = "mgcv-1.9-3-efsudr-dense"
    trace_method: str = "exact-dense-fisher"
    source_fingerprint: str | None = None
    basis_fingerprint: str | None = None


class EFSFitProvider(Protocol):
    """Return one validly attributed fit, or a truthful failed fit state.

    Providers must reconverge coefficients at the requested parameters.
    An estimated NB provider owns conditional theta updates inside PIRLS,
    rather than alternating theta after a completed coefficient fit. It must
    retain bounded state and use the same explicit coordinate/rank convention
    for beta, information, Fisher traces and scalar score.
    """

    def __call__(self, request: EFSFitRequest) -> EFSFitState: ...
