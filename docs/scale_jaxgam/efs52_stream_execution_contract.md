# Complete internal streamed EFS execution

`execution.efs_stream.fit_streamed_efs` composes the reviewed source startup,
immutable provider/context and existing accepted/trial EFS outer loops. The
public adapter selects it only for explicit QR `RowSource` execution and
materializes a compact prediction result. Neither layer returns a source or
training rows. Streamed full/inference result modes remain outside this
contract because they retain row-aligned state.

The source policy is pinned mgcv 1.9-3 `R/gam.fit4.r::efsudr`: the first
smoothing start receives +2.5 once, accepted coefficients start subsequent
PIRLS fits, conditional theta remains inside NB PIRLS, and extension and
contraction preserve the existing score/update/carried-scale timing. This
composition makes no change to either outer decision function. Unknown-scale
regular families use the existing unknown-scale loop; NB and known-scale
regular families use its known-scale counterpart.

The pinned top-level EFS call (`R/mgcv.r:1663–1666`) omits `G$null.coef`.
For NB this adapter therefore supplies gam.fit4's zero default recovery anchor
on every first/refit call. The startup response-mean projection remains a
distinct recorded get.null.coef reduction and is not passed as this anchor.
Regular streamed EFS uses the same zero source default through an additive
`null_coefficients` seam. Its `None` default retains the existing projected
get.null.coef behavior byte-for-byte for non-EFS regular-controller callers.
The explicit vector is shape/finiteness validated, copied only after an extra
`8*p` prospective host-vector charge, and remains distinct from retained
initial coefficients and their predictor. A pinned trace captures every
gam.fit3 null.coefficient argument in a top-level EFS fit and verifies the same
zero p-vector on every refit. Existing dense and fixed/trial-controller
defaults are unchanged.

The preserved seed-1201, n=79 identity-link fixture exposed this distinction.
Before the correction, fixed-theta smoothing differed by 2.815e-4 (relative
1.184e-9), while estimated-theta coefficients, means, smoothing, EDF and theta
differed by up to 1.946e-9, 2.361e-9, 3.718e-3, 5.700e-9 and 5.015e-10
respectively. Deviance, score, scale and outer counts already passed STRICT.
After supplying the source zero anchor on every NB refit, every selected field
passes STRICT in both modes, as do log/sqrt and all four regular fixtures.
The owning tests capture every dispatched anchor. No tolerance was relaxed.

Metadata and providers retain their initial family snapshot for lineage
validation. Only an isolated reporting family receives the selected estimated
theta after fitting. Caller families and immutable initial metadata stay
unchanged. This distinction is required when consuming the final fit in
Phase 3; the final score is already attributed to its explicit source timing.

The numeric budget charges an additional `8*(2*p + 3*m + 32)` bytes for NB
and `8*(3*p + 3*m + 32)` bytes for regular EFS before provider construction.
These cover CPU null/source-anchor and device first-request vectors, including
the temporary shifted smoothing start. Startup workspace
and provider metadata/three prior fits/configured histories are separately
charged by their reviewed bounds. Their combined startup lifetime is checked
from existing descriptors before provider metadata transfer or root construction.
The outer loop returns its selected fit,
and provider roots/context are released when this function returns. This is a
prospective numeric-array bound, excluding source storage, Phase-1 basis
preparation and opaque native/XLA workspace; it is not an RSS guarantee.

Diagnostics record `startup_source_scans` and `startup_batches_scanned`
separately from complete executed-trial `provider_source_scans` and
`provider_batches_scanned`. They exclude caller-owned Phase-1 basis preparation
and source fingerprint reads, and do not claim total end-to-end source cost.
Missing provider measurements remain unavailable under the existing generic
accumulator contract. Dense fits leave the two new optional fields as `None`.

The owning tests exercise all four regular families and fixed/estimated NB
log, identity and sqrt modes, finite selected states, actual source counters,
family/cache isolation and compiled final factor actions. Broader final-model
pinned-R parity belongs in the validation matrix and public release gates.
