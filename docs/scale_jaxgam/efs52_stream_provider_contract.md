# Exact streamed EFS accepted/trial providers

This internal component binds replayable coefficient fits to the reviewed EFS
accepted/trial controller. It requires explicit incoming smoothing parameters,
coefficient starts and, where applicable, score scale and theta/recovery anchor.
It does not expose public streamed EFS or supply default EFS starts yet.

Ground truth is pinned mgcv 1.9-3: `R/gam.fit3.r::efsudr/gam.fit3`,
`R/gam.fit4.r::gam.fit4`, `R/efam.r::estimate.theta`, `src/gdi.c::gdi1/gdi2`,
and `R/mgcv.r::initial.spg/get.null.coef`. The existing outer extension,
contraction, finite-increase and scale-carry policy is unchanged. Inner
safeguards remain owned by the reviewed regular and NB coefficient controllers.

`NBStreamEFSProvider` consumes fixed or estimated NB with log, identity and sqrt
links. Fixed theta belongs to the frozen family snapshot. Estimated theta is
explicitly supplied for every trial and updated conditionally **inside** PIRLS
using global bounded objective/gradient/Hessian reductions. Source stopping
deviance retains its entering theta, while outgoing theta determines final
curvature and saturated likelihood. The bounded theta transition history
verifies this intentional distinction; it cannot justify arbitrary stale theta.
See the [conditional-theta contract](efs52_nb_conditional_theta_contract.md).

`RegularStreamEFSProvider` consumes regular family primitives and the accepted
retained-start source-score payload. Pre-gdi1 factors remain distinct from
returned reporting coefficients. Scalar REML uses raw pre-gdi1 deviance plus
the final solve's candidate penalty, including a reporting fallback. Fisher
trace contractions use those pre-derived factors; quadratic contractions use
returned coefficients. Incoming `score_phi` remains distinct from Fletcher
`update_phi/reported_phi`; the existing controller owns subsequent carried phi.

Both providers retain replayable rows and compact local penalty metadata, not
training design/response arrays. Family snapshots are copied and shared by the
provider's fitting metadata. Prepared/source/basis/family lineage is validated
before dispatch and again after fitting. Returned rho, score scale and theta
transition attribution are checked before a valid fit state is constructed.
Failures remain truthful nonconverged states when a feasible last fit exists;
the coefficient controllers retain their existing explicit-domain exceptions.

Exact d/t/q uses the reviewed factor actions and local penalty roots from
[the factor contract](efs52_factor_contract.md). No new Hessian or
covariance reconstruction supplies the statistics. Scalar regular REML consumes
the observed factor's determinant and explicit source penalized deviance.

## Known numeric lifetime budget

The provider preflights descriptor-only upper bounds before copying/transferring
metadata or constructing numerical roots. Compressed values, projected local
matrices and roots are bounded by full local squares without an eigendecomposition.
It charges separate preparation scratch for expanded CPU penalties, temporary
unused device matrices and largest-block explicit eigen/root workspace; opaque
native scratch is excluded. After construction, actual retained array ownership
is counted by identity and verified against that prospective bound.

It additionally reserves three prior compact EFS fit states and live outer
proposals: `24p² + 192p + 32m + 192` float64 entries in total. Each prior state
has two information matrices, signed observed/positive Fisher factor matrices,
coefficient vectors and compact scalar diagnostics. Per prior fit, up to 8m
entries cover rho, the PIRLS rho copy, d/t/q and possible raw-update vectors;
another 8m covers live controller proposals. A losing extension remains alive
into the following outer iteration; the weakref regression demonstrates three
prior states during the next extension call. Bounded score/phi deques and their
returned tuple copies are separately charged as
`128 * (min(outer_limit, max(4, history_limit)) + 4) + 1024` bytes,
including conservative Python scalar/container overhead. No conditional-theta subhistory
survives in the returned EFSFitState. One current coefficient fit
uses its separately reviewed prospective workspace ledger, with this provider
reserve deducted from the caller's total budget before any fit scan.

After coefficient fitting returns, factor statistics retain only one root RHS
block and its solved action, each at most `p * min(32,p)` float64 entries. Those
buffers add `16p * min(32,p)` bytes to the provider reserve. No m-by-p-by-p
penalty/factor stack or n-row trial history is allocated. Arbitrarily many
caller-retained results are outside this controller-owned lifetime contract.
Source-owned data, CPU basis preparation and opaque native/XLA scratch remain
explicitly excluded; this is a numeric workspace budget, not an RSS cap.

## Count semantics and remaining scope

Per-fit counts include its source summary/null initialization, coefficient and
conditional-theta replays, trial/recovery scans and final reporting scans.
`provider_source_scans/provider_batches_scanned` in frozen optimizer diagnostics
sum every actually executed fit, including rejected/extended trials. A missing
measurement in any fit makes both aggregates `None`; ordinary dense fits keep
`None`. These counts exclude Phase-1 preparation and the future separate outer
initial-sp/scale preparation, and must not be described as total source cost.

Default startup still requires bounded PUBLIC-coordinate working diagonals
matching `initial.spg` (global NB observed-to-expected selection), the shared
CPU initial-sp balancing routine, and unweighted response mean plus weighted
null deviance/n/10 for incoming unknown scale. All-family startup, complete
dense/streamed/pinned EFS final-model parity, public routing/family snapshots,
result-mode/pickle gates and end-to-end resource evidence remain EFS5.2 work.
Matched starts in this component do not close those gates. Parent PR8.2's joint
exact-REML theta route remains a separate algorithm.
