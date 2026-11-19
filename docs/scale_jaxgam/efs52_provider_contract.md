# EFS5.2 provider extraction

This component separates the accepted/trial controller from dense fitting
inputs. It is the first review point of the full EFS5.2 streamed all-family
implementation, and does not enable streamed EFS publicly. Dense entry points
retain their initialization, capability, prior-weight and rank checks;
`FittingData` and the statistics plan remain inside the dense provider closure.
Neither initial smoothing nor initial scale helpers change here.

Ground truth is mgcv 1.9-3, commit
`fb7e8e718377513e78ba6c6bf7e60757fc6a32a9`:

| Contract | Pinned source |
| --- | --- |
| Accepted/trial policy, +2.5 initial shift, saved starts, extension/contraction, score stopping | `R/gam.fit4.r::efsudr`, lines821–954 |
| Regular coefficient recovery and final gdi1 score provenance | `R/gam.fit3.r::gam.fit3`; `src/gdi.c::gdi1` |
| Conditional theta inside NB PIRLS and baseline refresh | `R/gam.fit4.r::gam.fit4`; `R/efam.r::estimate.theta` |
| NB final observed score and rebuilt Fisher reporting factor | `src/gdi.c::gdi2`, especially2262–2299 |

`EFSFitRequest` carries rho, coefficient start, incoming score phi, explicit
log theta, the recovery anchor and start-presence flags. None for an optional
flag preserves the dense adapter's existing default/call shape. Arrays are
immutable JAX parameters, not response/design storage. `EFSControllerContext`
holds fixed/estimated theta mode, reference/trace provenance and optional
immutable source/basis fingerprints. A provider owns validation of its frozen
coordinate/family lineage and returns a coherent `EFSFitState`.

Both shared controllers fit each trial from the saved old accepted state.
Losing extensions and rejected trials cannot seed alternatives with their
beta, theta or phi. Incoming score phi remains separate from update/reported
phi. A winning unknown-scale extension retains its own update/reported phi
and carries the first candidate's phi into the next score, as the pinned
source does. Finite score increases at multiplier one retain the accepted
EFS policy; Newton and coefficient safeguards are unchanged. Bounded scalar
histories and compact diagnostics are the existing implementation, with
reference/trace fields supplied explicitly by the controller context.

The dense adapter preserves positional and keyword fit-call shapes, including
late binding of the existing fit function used by scripted regression tests.
The new owning tests run accepted, losing, halved and failed trial sequences
without constructing dense fitting data. Existing dense family, scale,
estimated-theta, source branch and public dispatch gates remain compatibility
evidence for this extraction. No new numerical tolerance class is introduced.

Subsequent review points retain the full EFS5.2 target:

1. Add retained-start/default-start streamed provider semantics, including
   the shared source natural-column null projection. Existing full-rank NB
   no-intercept gates do not close aliased null-anchor parity.
2. Adapt exact Fisher traces through the reviewed coefficient factor's
   `root_transpose_inverse` action and bounded root RHS blocks. Preserve its
   coordinate/rank convention; do not reconstruct an unrelated dense factor.
3. Add globally reduced conditional theta objective/gradient/Hessian and
   safeguarded scalar steps inside streamed NB coefficient iterations, with
   objective refresh after theta updates. Fully converging beta before
   alternating theta is outside this contract. Count-prefix budgets, source
   theta status and explicit trial ownership remain mandatory.
4. Validate all existing regular-family/link and fixed/estimated NB regimes
   against dense EFS and pinned EFS on identical input/model/control states,
   including unknown-scale score/update/carried phi timing. Complete public
   compact family snapshots, prediction/SE, ownership/pickle, bounded retained
   state and measured source-scan gates. Negative attribution gates must
   reject stale or mismatched rho, incoming score phi, theta and source/basis
   lineage before an actual streamed provider can return valid=True. The
   protocol's delegation of validation is not evidence of those checks.
   No whole X/y may be materialized.

Parent PR8.2's joint exact-REML theta optimizer is a separate algorithm and
cannot substitute for EFS's conditional in-PIRLS theta policy. This extraction
also makes no operator, stochastic trace, large-p or performance claim.
