# Bounded streamed EFS source initialization

This component implements the default fitting-boundary starts from pinned
mgcv 1.9-3 `initial.spg` and `get.null.coef` in `R/mgcv.r:4720–4819`
and `1852–1867`; `estimate.gam:1996–2034` supplies the incoming unknown
scale. It composes with the accepted regular/NB provider and does not change
ordinary dense initialization or public dispatch.

The first replay normalizes initialized responses with the family-owned hook,
combines global family metadata and the unweighted response mean, and reduces
unweighted PUBLIC-coordinate X against ones. Real zero-prior rows remain in
that QR. The accepted natural-column null helper projects `link(mean(y))`,
replaces source aliased NA coordinates by zero, and converts the public
coefficients to existing local fitting coordinates. Offset belongs to the
later predictor and is excluded from this projection. The public QR is
released before the second replay.

The second replay preserves family mustart separately from linkinv(eta).
Regular initial weights are `prior * mu_eta(link(mustart))² / variance(mustart)`.
NB initial observed weights are half Dmu2 times the same squared derivative,
with explicit immutable initial theta. NB globally selects expected Dmu2
weights if *any* observed initial weight is negative. This is distinct from
full Deta2 PIRLS weights and from its positive-observed recovery policy.
Both candidate diagonals accumulate the literal squared `sqrt(w) * PUBLIC_X`
operation. The original prepared penalty structure then enters the shared
CPU `initial_log_sp_from_diagonal` balancing routine; transformed fitting
penalties and Newton's prior-weight starts are not substituted.

The same replay reduces prior-weighted family deviance at the initialized
response's unweighted mean. Unknown incoming phi is that sum divided by the
original row count and ten. Known-scale families use one. Nonfinite or
nonpositive incoming scale is rejected without clipping. Startup retains
unshifted rho; `initial_request` adds efsudr's 2.5 once to the immutable
first-fit request. Estimated NB also supplies its initial theta and null
recovery anchor. No selected theta is fabricated by initialization. NB raw null deviance uses
pinned pmax(1,y) and the literal finite ratio (y+theta)/(mu+theta), with
initial theta explicitly attributed; ordinary defensive family defaults are
unchanged. Fractional/tail value-layer gates do not enable an excluded
fractional-below-one coefficient route.

Before any source evaluation/QR allocation, startup charges
`8 * (16p² + 32p + 8Bp + 64B + 8m + L)` bytes in addition to retained provider
ownership. Compact QR state, immutable/packed/stack copies, natural-column
helper and local-D conversion fit the coefficient term. Design and weighted
copies fit the Bp term; source/family and prior batch vectors fit the B term;
initial-sp vectors fit the m term. `L` is the largest local
`(b+4)k²` expansion plus the largest previous `k²` matrix: the shared CPU
balancing routine owns the complete dense local-penalty tuple while a prior
S and identity/absolute/activity temporaries can overlap it. The many-local
compressed-diagonal test exercises actual tuple expansion and balancing,
not just descriptor shapes. Startup does not overlap a PIRLS fit.
Source-owned storage, CPU basis preparation and opaque native/XLA scratch are
excluded; this is a known numeric workspace contract, not an RSS cap.
Source/basis/family lineage is checked around both replays. Startup scan and
batch counts are measured separately from the accepted provider totals.

Layer-specific tests compare live pinned initial.spg/get.null.coef outputs
with identical supplied public X, y, priors, local penalties and link objects.
The all-constructor matrix keeps integer Poisson responses and records the
specific bounded-link default-initialization failure against R. Such a
fixture boundary does not prove a constructor/link capability is unavailable
at a valid supplied start. Public dispatch, broad final-model parity and
result/serialization/resource release gates remain separate EFS5.2 work.
