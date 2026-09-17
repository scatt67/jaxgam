# Fixed and trial-theta streamed NB controller

The internal `fit_nb_streamed_pirls` controller consumes the accepted PR7.5
batch kernels and PR7.4 signed QR reducer. It fits coefficients at an explicit
theta for NB/log, identity and sqrt, with smoothing held fixed. An estimated
family requires an explicit trial `FamilyExecutionParameters`; this call
does not estimate theta. The caller-owned family and its prepared lineage
remain unchanged. Public all-family streaming belongs to PR7.6; joint exact
REML theta/smoothing optimization belongs to PR8.2. Conditional EFS theta
Newton is a distinct existing policy and is not used here.

Ground truth is mgcv 1.9-3, commit
`fb7e8e718377513e78ba6c6bf7e60757fc6a32a9`: `R/efam.r::nb`,
`R/gam.fit4.r::dDeta/gam.fit4`, `R/mgcv.r::get.null.coef`, and
`src/gdi.c::gdi2`. Read these with `git show 1.9-3:path` if the local checkout
has advanced.

Initialization retains the per-row `link(y + (y == 0)/6)` first predictor.
The recovery anchor separately projects the constant `link(mean(y))` onto
the unweighted design, excluding offsets. A literal intercept supplies that
constant directly; otherwise a bounded positive QR scan supplies the
projection. The initial source summary carries global maximum-count and
integer-response metadata without retaining a whole response or count table.
The initial family-domain gate is not regular `gam.fit3`'s .9/.1 shrink policy.

Each observed scan retains two compressed signed QR roots, a direct RHS,
small matrices/vectors and mergeable scalar metadata. A direct-response
request in any batch selects the accumulated `X.T Wz` globally. Source
good-row counts permit zero curvature with a finite nonzero derivative RHS;
curvature counts are separate and do not reject a penalized solve. An
indefinite proposal triggers a replay of the same predictor at positive
observed weights. That replay keeps the original derivative RHS and does
not substitute Fisher curvature. Trial domain/deviance checks and global
penalized-deviance backtracking add the penalty once per candidate.

Convergence uses the source deviance-change rule and a freshly recomputed
candidate gradient. `gam.fit4` uses its normal finite-pseudo-response good
mask for this gradient, separately from direct-RHS solve admissibility.
An iteration whose original observed solve was indefinite cannot declare
convergence merely because its positive-observed retry succeeded. A failed
line search or iteration limit remains nonconverged.

Final reported coefficients are the accepted PIRLS coefficients; there is
no regular-family `gdi1` polish. The observed factor is the likelihood factor.
`gdi2` then rebuilds `rV` and `K` with expected weights for posterior covariance
and EDF (`src/gdi.c:2262–2299`), so reporting uses the separate Fisher factor.
Its score penalty comes from the final observed solve's `PKtz`
(`get_bSb`), while the deviance remains the accepted PIRLS deviance. The
result records `source_deviance`, `gdi_penalty`, `score_penalized_deviance` and `reml_score`
explicitly; `StreamFitState.penalized_deviance` continues to describe the
returned coefficient state. A future public adapter must consume the
source score payload instead of silently recomputing it from returned beta.
Reported state deviance is bounded below by zero to satisfy the reporting
hard gate at exact residuals, where floating subtraction can produce a tiny
negative value. Only values within `64 * eps * (1 + n + sum(y))`
are bounded; a materially negative or nonfinite raw result raises instead
of returning a valid fit. This is an explicit accumulated-roundoff bound,
not a change to the source deviance formula. Source objective/history and score provenance keep the raw
value; this reporting bound does not change backtracking or convergence.

Fractional responses use pinned NB's literal `pmax(1,y)` deviance constant.
The controller compares that deviance and its score directly to `gam.fit4`,
including responses below one. This does not declare the ordinary dense
family's reporting convention or a different likelihood objective equivalent.
Theta, domain, coefficient-rank and observed score admissibility are separate
checks. Prepared host designs currently reject padded source batches;
the pure kernels support padding. Unsupported source/basis eligibility is
independent of NB/link capability.

The owning tests use live pinned `gam.fit4` at identical supplied fitting
matrices and penalty roots. They expose the final `oo$P` in a local clone's
return expression without changing fitting statements. Fields include beta,
deviance, observed/Fisher information, Fisher covariance/EDF, score penalty,
score and exact iteration/recovery counts, across batches and theta values.
Binary saved-input subprocess checks isolate trial theta and caches.

One original default-start input remains explicit boundary evidence:
seed1201, n79, cr k6, response generated at theta0.1, NB/identity theta0.1,
weights `0.5 + uniform` with every thirteenth zero, and offset
`0.5 + 0.1*sin(2*x)`. Pinned R rejects it with
`inner loop 1; can't correct step size` under epsilon1e-11/maxit100.
The owning boundary test retains its generator and requires rejection or
nonconvergence in the bounded controller. It does not classify the whole
link as unsupported. Successful comparisons use one identical response
input generated at theta2.7 across theta0.1/2.7/1e6. Sparse seed883/n96,
identity theta0.8/cr k5/offset1 separately exercises three positive-observed
recoveries and global backtracking. All new numerical assertions start at
repository STRICT; no tolerance exception is introduced.

The full-rank no-intercept oracle gates do not establish aliased null-anchor
parity. The current projection reuses the QR solver's rank convention;
pinned `get.null.coef` instead uses natural-column `qr`/`dqrdc2` and replaces
aliased NA coordinates by zero. Integration of the separately reviewed
shared source null-projection helper remains open. This controller does not
claim completion of that boundary or of the public PR7.6 release matrix.

## NB workspace accounting

The reused prospective ledger is a bound on known numeric buffers, excluding
source-owned data/preparation and opaque XLA/BLAS/LAPACK scratch. It is not an
RSS limit. Per batch, explicit host allowances are `8 B p + 64 B` float64
entries; coefficient allowances are `40 p² + 64 p`; visible device allowances
are `B p + 24 B + 8 p² + 16 p`. Local penalties/transforms/roots and bounded
history are charged separately by the existing ledger. These allowances do
not assert that every listed buffer coexists.

One completed NB scan holds at most six coefficient matrices: absolute and
negative observed QR roots, Fisher QR root, observed G, Fisher G and gradient
G. Its direct RHS and separate gradient RHS are counted among coefficient
vectors. At loop entry `current` and the previous `candidate_scan` alias the
same scan; positive recovery can replace current with a second scan. The
next accepted candidate replaces the old candidate reference. After fitting,
current, candidate and final can coexist, giving at most eighteen matrices,
rather than four independent retained scans. A scan's read-only construction
can temporarily retain three extra statistic copies; this occurs after its
QR updates, not during packed-QR scratch allocation.

A conservative simultaneous host matrix accounting is:

| Phase | Existing scans | New scan/statistic copies | Saved solve factors and null projection | QR or signed-solve visible scratch | Total p² units |
| --- | ---: | ---: | ---: | ---: | ---: |
| Working/recovery/candidate scan | 12 | 6 | 3 | 16 | 37 |
| Scan return with immutable statistic copies | 12 | 9 | 3 | 0 | 24 |
| Final observed or Fisher coefficient solve | 18 | 0 | 5 | 16 | 39 |

The sixteen-matrix scratch allowance covers bounded embedded local-root QR,
unpivoted input, stacked/packed copies and returned-root copy during QR;
these are released before signed correction's N/Z and SVD output/copies.
The latter fits within the same allowance. Roots are embedded individually,
never as an m-by-p-by-p global stack. The remaining matrix unit in `40 p²` covers a transient statistic product;
that product is not live during the larger final solve. The sixteen units
include separate data/root unpivoting, old/new balanced root, one embedded
root and normalized input, QR update U, 2p-by-p stacked/packed arrays,
returned R and its immutable copy, and spare factor/indexing copies.
Sixty-four host coefficient vectors cover three scans' QR responses, direct
and gradient RHS/pivots, beta/candidate/residual, solver correction/coefficient
vectors, and packed-Q/augmentation RHS copies.

The NB batch kernel has eleven row-array outputs plus scalar metadata.
Twenty-four device row-vector units cover outputs/selected rows or the old
and replacement working outputs during positive retry. Device inputs passed
to a completed call are released unless shared with an output. The host
`64 B` allowance separately covers source vectors, selected W/z/Wz, normal
mask/gradient rows, signed-weight response/masks and packed QR responses.
The eight host batch designs cover source X, absolute/negative weighted rows,
QR stack/packed copies, statistic products and a transfer. The device matrix
allowance covers retained observed/Fisher factors, information transfers and
visible covariance/EDF/score operands; compiled internal scratch is explicitly
excluded. Execution waits on host conversion each batch, so there is no
unbounded outstanding batch queue. QR roots/statistics are accumulated on
the CPU in this reviewed CPU QR route; a GPU transfer-cost claim is not made.

The owning memory gate holds B=256 and p=2 while n grows from 256 to 65,536.
It records both process high-water increments and Linux current RSS sampled
every 5ms after preparation, plus all retained numeric device-array sizes and
the prospective ledger. Earlier high-water peaks may hide allocations and
sub-5ms peaks may evade sampling. These observations complement the explicit
buffer accounting; they do not prove a universal process-memory ceiling or
a speedup.
