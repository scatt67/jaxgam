# Streamed NB conditional theta inside PIRLS

This internal EFS5.2 component extends the accepted fixed/trial-theta NB
controller with retained starts and an explicit `estimate_theta=True` route
for NB log, identity and sqrt. The default coefficient subproblem remains
fixed at its supplied theta. This component enables no public RowSource EFS
route and does not replace joint exact REML theta optimization with
alternating completed beta/theta fits. Existing dense bodies are unchanged.

Ground truth is pinned mgcv1.9-3 (`fb7e8e718377513e78ba6c6bf7e60757fc6a32a9`):
`R/gam.fit4.r` initialization/recovery at255–505, conditional update and
convergence at509–548, final `gdi2` score, and `R/efam.r::estimate.theta`.
The accepted NB batch kernels retain direct Wz and the derivative RHS when
positive-observed curvature is required; this retry is distinct from regular
families' Fisher recovery.

## Starts and source timing

A bounded source summary supplies mean response, maximum count and whether
responses are integer. The null anchor uses `link(mean(y))` without offsets
or prior weights. With no intercept, the accepted natural-column CPU helper
projects the constant onto an unweighted PUBLIC-coordinate compact QR,
including real zero-prior rows and excluding padding. Source aliased NA
coordinates become zero before converting to local fitting coordinates.
A supplied immutable copy of `beta_old_init` can replace this anchor.

A valid supplied coefficient start is retained only when its initial
penalized deviance does not exceed the null baseline. The first working
predictor remains per-row mustart when no start is retained. Finite/domain
recovery uses the retained old beta when present; immediate first-iteration
divergence separately uses the null anchor. The owning tests compare each
retained/default route against R at an identical start, rather than requiring
two different starts to select identical stopping coordinates.

For each accepted coefficient proposal the controller performs this source
sequence:

1. Evaluate candidate raw deviance and the convergence pdev at incoming theta.
2. Run conditional theta Newton at that candidate beta, replaying bounded
   batches for every global objective/gradient/Hessian evaluation.
3. Recompute the normal finite-pseudo-response gradient at new theta and
   compare convergence against the pre-update candidate pdev.
4. On a continuing iteration only, refresh deviance/pdev at new theta.
5. Form final observed and Fisher factors at final theta. The final score
   uses the source stopping deviance plus the final observed solve penalty;
   reporting separately uses final-theta deviance.

The result therefore records raw `source_deviance` with its explicit
`source_deviance_log_theta`, `stopping_penalized_deviance`, final-theta
reporting deviance, solve penalty, final theta and counted theta scans/steps.
Its raw score payload must be consumed as such; recomputing it from returned
beta and final theta silently changes source timing. A deliberately
appreciable last update at PIRLS tolerance0.5 is checked against live R:
both stop after two coefficient iterations, raw deviance differs from final
reporting deviance by more than2, and the nonzero solve penalty is retained.
The former guessed one-iteration test failure is preserved externally.

The pure JIT batch objective holds mu fixed while differentiating log(theta).
For nonlog links it maps that fixed mean into log space and reuses reviewed
eta-space deviance arithmetic; it does not differentiate beta/link inputs
through theta. The source scalar step uses absolute-curvature repair, a
maximum step4, at most25 halvings, at most100 iterations, objective-increase
roundoff allowance and the source gradient stopping inequality. Failure
returns the last valid theta and an explicit nonconverged status. Fractional
responses below one preserve the existing conditional EFS status8 boundary;
fixed/trial-theta source deviance remains independently supported.

Family parameters are explicit immutable trial inputs. Cached batch dispatch
validates source/basis/family lineage each time; neither accepted family nor
caller start arrays are updated. Histories retain at most one scalar theta
per coefficient iteration and only the last bounded conditional subhistory,
never row predictors, response arrays or trial designs. The full streamed
provider still needs request/state attribution gates before public integration.

## Prospective workspace and lifetimes

The [accepted NB ledger](pr75_nb_controller_contract.md#nb-workspace-accounting)
reserves host `40p²+64p`, `8Bp+64B` and visible device
`8p²+16p+Bp+24B` float64 entries, plus local penalties and histories.
Source/preparation storage, Python nonnumeric metadata and opaque compiled
XLA/native library scratch are excluded; this is not an RSS ceiling.

Null projection runs before all old/current/candidate working scans. Its
PUBLIC and fitting designs coexist within the eight host batch designs.
Compact old/new QR, packed/stacked QR copies and the helper's unpivoted
natural-column copy are covered by the existing coefficient scratch bound.
During helper projection only a compact QR plus one p-by-p copy and vectors
coexist with penalty/null-coordinate conversion operands. No working scan
exists then. The initial last `public_X`, fitting X, source row vectors, CPU
initial working arrays and null QR are explicitly released before derivative
scans. A weak-reference owning gate verifies the null QR root is gone at the
first working scan; the source gate includes real zero-prior rows.

| Phase | Simultaneous coefficient storage | Additional theta storage |
| --- | --- | --- |
| Initial unweighted QR/helper | At most two compact QR roots, bounded packed QR scratch, one natural-column helper copy and local transform/roots; no working scans | Initial theta/scalars |
| Working/retry/candidate | Existing NB maximum37p² plus a transient statistic product within40p² | No theta AD graph live |
| Conditional theta replay | At most current/old candidate scans and saved proposal/null vectors; no QR/solve scratch | Bounded batch AD, bounded count plan and scalar histories |
| Final observed/Fisher solve | Existing NB maximum39p² within40p² | Last bounded theta result only; no theta AD graph live |

Stage one charges an additional `64B` device float64 entries for combined
value/gradient/Hessian batch work and `128*(theta_max_iter+pirls_max_iter+2)`
bytes for scalar histories, before any summary scan. The original pdev
history remains separately charged. Previous conditional subhistories are
released before the next conditional solve; counts and carried theta have
already been accumulated. At return, the new two scalar lists/tuples and
one coefficient-iteration theta list/tuple fit that allowance, including
float objects, pointers, list over-allocation and bounded headers.

Stage two follows the one bounded summary pass, before any theta derivative
or oversized count-prefix dispatch. The inherited integer-prefix selection
policy uses `(capacity+1)*8*4 <= 8MiB`. If selected, the combined value/g/H
route conservatively charges sixteen float64 arrays of that capacity. If
not selected, inherited vector/recurrence work retains only bounded batch
vectors and creates no count-length table. Its computation can be
O(B*capacity); this component makes no runtime bound or speedup claim.
Fractional counts do not allocate an integer prefix. A prospective owning
gate fails immediately before AD when the discovered count plan exceeds
`maximum_bytes`; known batch/history excess fails before a source scan.
The initial summary also rejects a source exceeding the prospective batch
cap before evaluating a design or null QR.

Compiled CPU memory records at global capacity64: B16 has408 argument,
57 output and9160 temporary bytes; B64 has1608 argument,57 output and23432
temporary bytes (float64, JAX0.9.0.1, no aliases). These actual compiled
measurements distinguish opaque temporaries from the prospective visible
buffer accounting. The output consists of three scalars and validity.
A fixed B/p/count-capacity n-growth gate separately checks retained numeric
shapes, bounded histories, prospective ledger and actual scan counters.
These checks complement the fixed-theta controller's sampled RSS evidence;
no conditional-theta RSS or timing claim is inferred from it.

## Numerical review and remaining scope

All ordinary/default/retained controller fields and same-coordinate end
contractions remain STRICT. The narrowly reviewed six-row integer and
fractional inputs use `mu=[1.2,2.4,3.1,0.8,2.0,4.0]`, prior weights
`[0.8,1.2,0,1,0.9,1.1]`, responses `[0,2,3,4,0,7]` or
`[0,2.25,3.5,4.75,0,7.25]`, and a log(1e6) theta start. Four executed
corrections (matched JIT execution, per-row differentiation, literal source
ratio and stable log-mean arithmetic) still missed STRICT initial R
gradient/Hessian parity; the stable arithmetic matched an independent
65-digit derivative reference more closely. On only these exact inputs,
initial R objective/gradient/Hessian, nonlog dense initial derivatives,
fractional nonlog dense initial objective and integer selected final theta
have named MODERATE owning comparisons. Unchanged dense log-link and integer
initial objectives, final objective and identical-coordinate final
contractions, and the high-precision derivative reference remain STRICT.
Early theta trajectories exceed even MODERATE and remain explicit limitations.
Counts, stopping inequalities, validity and safeguards are independent
mandatory gates; this does not relax other inputs or controller paths.

The null helper's owning tests establish source alias rank/coordinates;
full aliased coefficient/score parity is not implied by the new full-rank
no-intercept controller gates. Initial sp/scale streamed reductions, regular
source-score payload binding, stale provider rho/phi/theta attribution,
all-family streamed outer EFS, public/result-mode gates and complete EFS5.2
acceptance remain follow-on work. This component does not close those items.
