# Streamed NB working-system kernel contract

This PR7.5 layer implements pure, explicit-theta coefficient factors for
negative-binomial log, identity and square-root links. It does not enable a
public streamed estimated-theta route or implement either conditional theta
optimization or the joint REML objective. Ordinary dense PIRLS is unchanged.

The source reference is mgcv 1.9-3 (commit
`fb7e8e718377513e78ba6c6bf7e60757fc6a32a9`): `R/efam.r`'s `nb`/`Dd`,
`R/gam.fit4.r`'s `dDeta`, and coefficient/recovery lines 345–405. The local
clone may contain a later checkout; use `git show 1.9-3:path` to reproduce.

`nb_working_batch` accepts the frozen working predictor, response, prior
weights, offset, padding mask, immutable `FamilyExecutionParameters`, and a
static link name. Theta is always a dynamic argument, including fixed-theta
calls. No mutable family is captured, no coefficient system cache is created,
and the result records the exact log-theta coordinate that generated it.
An owner retaining small coefficient systems must key their provenance by
that coordinate and the accepted/trial lineage; a trial must not mutate an
accepted family's state. The starting predictor is separate from a null-beta
anchor and includes offsets.

Observed weights are half the deviance's second eta derivative. Expected
Fisher weights are returned separately. For square-root links, the derivative
transform follows `mu.eta(linkfun(mu))`, as in `dDeta`. Source arithmetic
temporaries and its derivative-ratio operation order are preserved using the
existing rounded-operation helper. The direct weighted response is
`W * (eta - offset) - Deta / 2`. Fractional responses use the source deviance's
`pmax(1,y)` convention; this is not an assertion that another optimizer's
saturated-likelihood or fractional-domain policy is interchangeable.

The controller OR-reduces `requires_direct_response` across the scan before
choosing normal or direct rows. Raw pseudo-responses remain available for
diagnostics, while `nb_selected_working_rows` supplies finite QR inputs.
The direct route does not require finite pseudo-responses and retains a finite
derivative RHS even when curvature is zero. Source good-row counts count the selected finite W/z or W/Wz rows, including
zero curvature with finite derivative RHS. The gam.fit4 data gate checks this
good-row count. Separate informative counts count usable nonzero curvature;
they must not reject a zero-curvature direct-RHS penalized solve. Factor and
penalty rank determine solve/score admissibility independently.
Empty, all-zero-weight and padded batches contribute neutral systems and
curvature counts. Real zero-prior rows remain source good rows on the direct
path; they contribute zero RHS/deviance. Global prior-weight support and family
domain checks are separate from source good-row counts.
Invalid real input fails the domain flag; controllers must check global
domain/admissibility and source good-row metadata before accepting a solve.

An indefinite observed solve requests a separate scan with
`nb_positive_observed_retry`. This zeros nonpositive/nonfinite observed
curvature and rebuilds the direct weighted response from the unchanged
derivative RHS. It is the `gam.fit4` positive-observed recovery, distinct from
regular-family Fisher recovery. A step-local recovery factor must not be
reused as the final observed score factor.

The kernels retain only batch vectors and scalar metadata; signed G/RHS are
parameter-sized. They allocate no count-prefix table and retain no whole
response. Global maximum-count/integer-count metadata remains owned by the
existing NB family summary contract. Saturated likelihood and its bounded
count-prefix workspace belong to their existing explicit-theta family layer,
not these coefficient kernels.

The owning tests compare binary, identical realized eta/mu inputs to live
pinned R `dDeta`/deviance at STRICT: three links, theta 0.1/2.7/1e6, tiny
means, large finite means, count tails, fractional rows, weights and offsets.
They also cover JIT, padded one-row/full reductions, zero curvature/direct
RHS, positive-observed retry, neutral batches, constant-size count metadata,
repeated theta changes and a fresh-process dynamic-theta cache check.
