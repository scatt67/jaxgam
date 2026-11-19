# PR11 Gaussian working-model smoothing component

Scope: port pinned mgcv 1.9-3 (`fb7e8e7`) additive-Gaussian `bgam.fitd`
performance iteration over the accepted compact design. Execution, grid
approximation and smoothing strategy remain separate controls. Explicit `sp`
owns fixed mode. This is separate from nested exact REML and EFS, even when
the Gaussian reduced criterion happens to coincide mathematically.

## Requirement and source map

| Requirement | Pinned source | Owning gate |
| --- | --- | --- |
| Frozen working cross-products, weighted adjusted response moments | `bam.r:635–659` | Same selected explicit X/G/h/yy; weighted/offset and batch boundaries |
| Initial block/range transform | `fast-REML.r::Sl.setup`, `Sl.initial.repara` | Source-invariant criterion, public coefficients and derivatives |
| Initial log smoothing and scale | `bam.r:681–696`, `initial.sp`; unweighted `var(y)*.05` | Same initialization with compact penalty/public order and stable moments |
| Trial coefficients and derivative contractions | `fast-REML.r:1585–1656` (`Sl.fitChol`) | STRICT beta, criterion, gradient/Hessian in fixed/free scale modes |
| Active-coordinate filter, absolute-Hessian spectrum, eigenvalue floor and step cap | `fast-REML.r:1657–1676` | Dropped inactive directions, indefinite/weak curvature, max step four, JIT |
| Accepted and proposed log parameter states | `bam.r:699–749` | Accepted/trial traces; a proposal is not accepted implicitly |
| Directional-gradient step halving | `bam.r:738–748` | Finite shortened trial path and source state timing |
| Gaussian frozen statistic reuse | `bam.r:586`, `663–665` | No training scans during smoothing/scale proposals |
| Deviance and scale convergence | `bam.r:669–674` | Same epsilon and finite scale-step checks; iteration limit reported truthfully |
| Final full-row criterion/null dimension | `bam.r:779–786` | Final pinned BAM fields, separate initial versus reported Mp |
| Stable final selected deviance | Accepted fixed-sp execution residual scan | Near-perfect fits, truthful total scan/batch count |
| Route/results ownership and memory | Scaling plan PR11, accepted discrete/fixed-sp ledgers | Explicit working-model strategy, frozen predictor/Fisher/covariance, p/m/B reserve |

The Gaussian proposal primitive reuses `_criterion_core` and local
`penalty_ops`. Public-coordinate statistics avoid reproducing an initial
block coordinate system unnecessarily; the pinned oracle applies
`Sl.initial.repara` then maps coefficients back, proving invariance. Static
zero inactive directions let JAX preserve the source active subproblem
without dynamic shapes. The full `n-Mp` Gaussian log-scale normalizer is
included. No rows appear in the kernel graph or metadata.

## Minimal same-input diagnostic before production edits

A local same-input diagnostic (not tracked) exited zero on accepted
row-memory image `120dc3f97292`. On 97 weighted rows with an offset,
ordinary cubic, separate cubics and a moderate coupled cubic tensor match
pinned `Sl.fitChol` at STRICT for design, G/h, criterion, gradient, Hessian
and proposal. Maximum criterion error is 7.049e-12; derivative error
2.108e-13, Hessian error 1.421e-13, step error 1.019e-12. Both scale modes
are subsequently covered in owning checked-in tests.

## Controller state and scan contract

The additive-Gaussian controller reduces one frozen G/h/yy and response
variance. It keeps accepted log smoothing/scale, the proposed next step,
trial coefficients and previous/current penalized deviance separately.
`bgam.fitd` takes one limited smoothing proposal per outer iteration;
directional-gradient control halves that trial step on the same statistics.
Convergence is checked before a new proposal, after iteration two, using
penalized-deviance change and estimated-scale step. The source's finite
iteration and final criterion timing must be preserved, rather than silently
substituting exact minimization or a different gradient convergence test.

The final selected residual scan computes deviance stably and freezes
prediction/uncertainty ownership. Setup scans remain separate from fitting
scans. Gaussian smoothing/scale iterations perform no source scans.

## Boundaries remaining after the proposal primitive

- The Gaussian performance controller, public opt-in strategy and final BAM
  fit gates are not yet implemented by the proposal primitive.
- Rank-dropping, linked/shared smoothing matrices and nuisance updates are
  not validated here; the regular (Poisson/Binomial/Gamma) families are
  covered by the section below. AR1 and NCV remain unsupported compatible
  features, not new mandatory milestones.
- Raw `yy-beta'h` has the pinned cancellation boundary; stable final
  deviance does not establish large-response-location oracle parity.
- The previous narrow fixed-sp extreme-coupled score tolerance exception
  does not authorize relaxing proposal derivative or smoothing-loop gates.
  Extreme-ratio proposal derivatives remain an explicit unvalidated domain.
- Request-wise prediction recompression is the opt-in
  `prediction_policy="mgcv_predict_bamd"` (pinned `predict.bamd`: the whole
  request is compressed by `discrete.mf` defaults in a bounded pre-pass
  before any batch is evaluated); `continuous_frozen_basis` stays the
  default exact policy. The policy changes only metric discretization:
  factor values keep the default path's semantics, so a missing or unseen
  random-effect level contributes zero (an intended divergence from
  mgcv's `NA`). Fitted results record per-group/per-term grids in
  `GAMPredictionResult.discretization`. The held-out approximation/interval
  study and quantified runtime gains remain full-PR11 gates.

## Proposal candidate provenance and owning validation

Base `3684a06dce4a4374f4df84ac4ba7fdcbe3d02dc2`, following accepted
fixed-sp score `4075305`, on branch `codex/pr11-working-model-kernel-20260917`.

Final immutable image `jaxgam-test:pr11-working-model-kernel-freeze2-20260917`,
arm64, `sha256:a580feb010c495fe10b9fe03834fafd9ae8cef323983ba406836c76fba453a19`.

- `make lint` passes.
- Owning pinned/JIT run: 15 passed, six rpy2 deprecation warnings, no skips,
  9.34 seconds, exit zero.
- JAX-initialized branch coverage: 15 passed in 12.04 seconds, 100% of new
  module (91 statements, 14 branches, no misses/partial branches), exit zero.
- All 168 `jaxgam`/`tests` Python files match the image exactly (nine scripts
  excluded; the host and image Python manifests have an empty diff).

Tests defer rpy2 import until the pinned-R availability fixture has passed,
so owning NumPy/JAX gates remain collectable without local R. No public
smoothing execution or whole-PR11 acceptance is claimed by this candidate.

## Regular-family performance iteration (Milestone A)

Scope: pinned mgcv 1.9-3 `bam(discrete=TRUE)` for Poisson (log, identity,
sqrt), Binomial (logit, probit, cloglog, log) and Gamma (inverse, identity,
log), fixed `sp` and `smoothing_strategy="working_model_reml"`. The negative
binomial is Milestone B (below). Code: `fitting/discrete_working.py` (per-batch kernels),
`execution/regular_working_model.py` (controller),
`execution/discrete_regular.py` (route), `families/*::source_dev_resids`.

### Discrete-method semantics (differ from the dense route)

| Element | Pinned source | JaxGAM |
| --- | --- | --- |
| Start: `eta = linkfun(mustart)`, `crit <- dev`, `validmu`/`valideta` only here | `bam.r:512-528`, `611` | `initial_working_state_cpu`, iteration one of `advance_regular_working_model` |
| Fisher `w = G$w*mu.eta^2/variance`, `z = (eta-offset)+(y-mu)/mu.eta`, `mu.eta == 0 -> .1`, `y.norm2 = sum(w*z^2)`, `XWXd`/`XWyd` | `bam.r:637-651` | `discrete_working._fisher_working_quantities`; no observed-curvature split, no clipping, no `good` filter |
| Raw `sum(dev.resids)` (no domain mask), row order, long double | `bam.r:577-588`; R `rsum` | `source_dev_resids`, `source_sum` |
| Step control from iteration three: halve towards `coef0` while `dev0+bSb0 < dev+bSb` or non-finite, `kk < 30`; none at iteration two | `bam.r:575-603` | step loop of `advance_regular_working_model` |
| Penalized deviance `dev + sum(rSb^2)`, `stop("Non-finite deviance")` | `bam.r:570-576`, `605-611`, `667` | `source_penalty_quadratic`; `FloatingPointError` |
| Convergence before the proposal: penalized change `< epsilon` and, for an estimated scale, `|Nstep[n.sp+1]| < epsilon(|log.phi|+1)` | `bam.r:669-674` | same test |
| `initial.sp` on untransformed iteration-one `X'WX`; `log(var(G$y)*.05)` | `bam.r:676-696` | `start(statistics, variance)` |
| Fixed `sp`: zero-column `L`, so `t(L) reml1`, step on `log.phi` only | `fast-REML.r:1585-1676` | `fixed_smoothing` proposal mode |
| Scale: 1 for Poisson/Binomial (`scale>0`), `log.phi` for Gamma | `bam.r:2185`, `688-696` | `scale_is_fixed` |
| Final `coef` (step-controlled at convergence, last proposal at `maxit`), `edf = diag(PP qrx$R)`, `Vp = PP*scale`, `crit` with reset `Mp` | `bam.r:751-759`, `776-823` | `RegularWorkingComponents.result` |

Rows are only reached by complete bounded scans (one per coefficient trial;
an accepted trial's scan also forms the next working model). The memory plan
charges per-batch family vectors, four host statistic sets and the measured
compiled proposal workspace.

### Ties and the `_oracle_replay_halvings` oracle port

Near convergence, and for Gamma while `log.phi` still moves after the
coefficients have converged, the step-control comparison is between two
penalized deviances equal to rounding. Pinned R's own outcome there is
input-noise driven: a one-ULP offset perturbation moves R's own
Gamma/inverse fixed-sp coefficients 67.9x STRICT (measured R vs R), and
JaxGAM's outcome can differ at a margin of 1e-16. Exact replication would
need bit-identical `Xbd`, link and `dev.resids` arithmetic.

Gates therefore require identical decisions, except that a differing
coefficient halving is admitted only when its relative margin is a STRICT
tie (`|margin| <= STRICT.rtol`); the fit is then repeated with the
reference's outcomes replayed (`fit_discrete_regular(...,
_oracle_replay_halvings=...)`), every replayed outcome JaxGAM's rule would not have
taken must again be a STRICT tie, and every field is gated at STRICT. Ties
replayed in the checked-in gates on the macOS host (overridden margins at
most 1.3e-15): Gamma/inverse fixed (both formulas; iterations 7-8),
Gamma/inverse selected (iteration 14), Gamma/log near-zero fixed
(iterations 11-25, 67 trials) and the explicit-design Gamma/inverse runs;
on Linux: Gamma/inverse tensor fixed (iteration 8), Gamma/log near-zero
fixed (38 trials, at most 8.2e-16) and the explicit-design Gamma/inverse
runs (one at a margin of exactly 0). Without replay those cases differ from
pinned R by up to 137x STRICT in coefficients; with it the worst field is
5.6e-3x STRICT.

### Validation (pinned R 4.5.2/mgcv 1.9-3)

Worst ratio to STRICT per gate and field, macOS host / Linux image
`jaxgam-pr11-nongaussian:35fc2cd`:

| Gate | coef | fitted | deviance | EDF | scale | score | sp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Explicit design, same controller (13 cases) | 0.41 / 0.44 | 4.9e-4 / 1.1e-3 | 6.4e-5 / 3.7e-5 | 3.5e-4 / 3.8e-4 | 0.47 / 0.47 | 1.4e-2 / 1.4e-2 | - |
| Pinned bam, fixed sp (13) | 0.42 / 0.45 | 2.6e-4 / 1.8e-4 | 1.2e-5 / 1.1e-5 | 3.6e-4 / 3.5e-4 | 1.2e-4 / 5.4e-5 | 1.1e-5 / 8.8e-6 | - |
| Pinned bam, working-model REML (12) | 8.6e-3 / 9.0e-3 | 2.2e-4 / 1.9e-4 | 2.4e-5 / 1.3e-5 | 9.9e-5 / 3.5e-5 | 1.4e-4 / 9.6e-5 | 1.8e-5 / 7.9e-6 | 0.80 / 5.90 (Binomial/log; MODERATE, below) |
| Boundary, fixed (3) | 3.9e-4 / 1.6e-3 | 1.4e-4 / 7.5e-5 | 3.6e-6 / 2.5e-6 | 6.0e-5 / 1.5e-5 | 5.6e-3 / 2.2e-3 | 1.2e-4 / 6.8e-5 | - |
| Boundary, selected (3) | 6.1e-4 / 2.6e-3 | 2.1e-4 / 6.6e-5 | 1.3e-5 / 2.1e-5 | 7.5e-5 / 2.4e-4 | 1.1e-3 / 3.5e-3 | 3.3e-6 / 3.8e-5 | 4.62 / 0.88 (Poisson low counts; MODERATE, below) |

Explicit-design fixed point: `pirls_loop` and the discrete fit iterated to
its rounding floor agree in penalized deviance to 9.0e-6x / 6.6e-6x STRICT.
(Final coefficients of the two loops are not compared: `bgam.fitd` stops on
a penalized deviance change, second order in the remaining step; on
Binomial/log the floor coefficients sit 1.4e-8 from the `pirls_loop` fixed
point with an equal penalized deviance.)

Matched state (8 cases), host / Linux: statistics 7.6e-3x / 6.5e-3x,
proposal coefficients 1.6e-3x / 0 (bit-identical), gradient 0.34x / 7.9e-3x,
Hessian 1.2e-3x / 4.4e-4x, ldetXXS 1.9e-6x / 5.1e-6x, ldetS 1.7e-6x /
1.8e-6x STRICT; raw Newton steps up to 3.1x / 3.2x STRICT on the drifting
Poisson low-count sp, which passes the authorized backward-error and
perturbation-bound gate.

`Sl.fitChol`'s `rss.bSb <- yy - sum(beta*f)` (`fast-REML.r:1646`) cancels to
the size of the log(phi) gradient. With XLA's double dot, the Gamma/log
near-zero matched-state gradient differed from pinned Linux R by 8.8e-12
absolute (8.8x STRICT) although beta matched bit for bit; `source_sum`
(R's `sum`: element order, C long double, one rounding) reproduces R's
gradient exactly and is shared by the Gaussian and regular proposals.

### Proposed MODERATE: two flat smoothing-parameter vectors

Two selected fits drive one log sp past 16, where the REML gradient and
curvature along it are cancellations of O(1) terms down to about 1e-6. Each
vector exceeds STRICT on one platform only; every physical field is STRICT
on both, and pinned R itself is not reproducible at STRICT there:

| Vector | JaxGAM vs R, host / Linux | R vs R, one-ULP offset, host / Linux |
| --- | --- | --- |
| Binomial/log, random-effect log sp 16.9 | 0.80x / 5.90x STRICT | 1.5-8.6x / 3.1-7.2x |
| Poisson low counts (boundary), log sp 17.8 | 4.62x / 0.88x STRICT | 2.6-4.6x / 0.2-4.9x |

Correction attempts, on each vector on the platform where it fails:

1. Single-batch statistics (R's single-pass accumulation order instead of
   17-row batches): Poisson host 4.62e-10 -> 2.61e-10; Binomial Linux
   5.90e-10 -> 9.25e-10. Not resolved; the vector follows statistic
   rounding order.
2. Long-double cross-batch accumulation of `X'WX`, `X'Wz`, `y.norm2`:
   Poisson host unchanged (4.62e-10); Binomial Linux 6.79e-10.
3. JaxGAM's Newton-step policy (active set, absolute eigenvalues, floor,
   cap) on pinned R's exact gradient/Hessian: R's step to 1.1e-16 (host) and
   2.2e-15 (Linux). The step policy is not the source.
4. Matched-state localization: from pinned R's exact state the last drift
   steps alone account for the whole gap (Poisson host iterations 12-13:
   7.4e-11, 3.2e-10, gradient relative error 3.9e-10 at |g| = 1.1e-6;
   Binomial Linux iterations 11-14: up to 6.3e-10 with the gradient
   matching and the curvature cancellation-limited). All pass the
   perturbation bound.

Root approved both vectors at MODERATE (2026-10-24). Each allowance is bound
to its exact case (`_REVIEWED_FLAT_SP`): a SHA-256 of the generated input
bytes (columns in order, factor codes and levels, weights, offset), the
formula, family/link, `sp=None` and the full `FitControl`. Trigonometric
libm differences change the generated low bits across platforms, so each
digest is the bytes of the platform where the vector exceeded STRICT
(Binomial/log: Linux arm64; Poisson low counts: macOS arm64); other bytes
stay STRICT, and reviewed bytes with a changed model or control fail. A
guard test perturbs one row, weight, offset or factor code by one ULP or
swap (allowance lost) and changes the formula, link, `sp`, batch size,
epsilon, uncertainty, grid size or smoothing strategy (assertion).

### Tie replay policy (root decision 2026-10-24)

- Replay is a matched-state oracle policy: it applies only when the
  differing halving margin is a STRICT tie, and every replayed field is
  STRICT.
- For a fit that needed a replay, the unreplayed public fit's
  coefficients, fitted values, deviance, EDF, scale and score are also
  gated against pinned R (or the explicit design) at MODERATE, since R's
  own result is not reproducible at STRICT through such ties (R vs R,
  67.9x STRICT).
- Replayed decisions are counted per fit and bounded by reviewed per-case
  maxima (`_REVIEWED_TIE_REPLAYS`, `_REVIEWED_MATCHED_REPLAYS`: the larger of
  the macOS host and Linux arm64 counts, e.g. 67 and 50 for Gamma/log near
  zero); every other case must need none.
- The hook is internal: `_oracle_replay_halvings`, unreachable from `GAM`,
  `FitControl` and every result (tested).

### Long double on x86_64

R's `sum` accumulates in C `long double`: 53 bits on macOS arm64, the 80-bit
x87 format (64 bits) on x86_64 Linux, IEEE quad (113 bits) on aarch64 Linux.
`source_sum` follows the platform's type exactly: a sequential double sum,
an exact x87 emulation (each addition rounded to 64 bits, ties to even, on a
double-double state), or TwoSum beyond quad. Each accumulator equals an
exact rational model of its width in every tested case, including x87
half-ulp ties; pinned R's `sum` and NumPy's sequential `longdouble` sum equal
the same model in 1209 cases on macOS arm64, Linux arm64 and x86_64 under
emulation (real x87), and `sum(1, 2^-53, 2^-64) - 1` is 0 on x87 and
2.2e-16 in quad, in R and in JaxGAM.

JaxGAM itself cannot run under the local amd64 emulation: Colima's QEMU
7.0.0 has no AVX in TCG, and jaxlib's CPU guard rejects it (also with
`QEMU_CPU=max`), as recorded for the earlier amd64 diagnostic image. The
regular discrete, Gaussian working-model and bridge files therefore have no
local amd64 run; native x86_64 CI remains their amd64 gate, where the
`source_sum` platform test compares against the native 80-bit `longdouble`.

### Recorded divergences and indeterminate cases

- Out-of-domain means follow pinned `dev.resids` (`source_dev_resids`):
  a negative Poisson identity mean at a zero count is finite (`2*mu*wt`),
  as in R. The inherited WIP masked every invalid row to `NaN`, which
  stopped fits that R continues. Gated against R row by row.
- Links keep JaxGAM's shared arithmetic: R's `make.link` clamps
  (`pmax(exp(eta), eps)` for log, probit `eta` clamped to `+-8.13`,
  cloglog `mu` clamped to `[eps, 1-eps]` via `expm1`) are not reproduced;
  they only matter at extreme predictors outside the gated domain.
- R's halved predictor recursion `eta <- eta0/2 + eta/2` is replaced by
  `Xbd(coef) + offset` at the halved coefficients (equal up to rounding;
  replicating the recursion needs per-row history).
- Poisson/identity low counts: pinned R continues past negative means and
  ends after 29-halving iterations; its own result moves to JaxGAM's under
  a one-ULP offset perturbation (R vs R coefficients 9.8e9x STRICT). Chaotic,
  not an end-to-end gate.
- Non-finite working statistics with a finite deviance raise
  `FloatingPointError`; R would continue to a `non-finite coefficients`
  warning.
- Gamma/inverse tensor/factor-by selected: both stop with a non-finite
  deviance at the top of iteration two (gated).
- Pre-existing at accepted `40b7615`: `test_completed_fitting_scan_releases_
  batch_arrays_before_residual_scan` (Gaussian) is GC-timing dependent (4 of
  7 macOS host runs of its file failed at `40b7615`, passing on Linux; an
  argument list inside JAX holds the last weight array until collection).

## Negative binomial on the discrete route (Milestone B)

Scope: pinned `bam(discrete=TRUE, family=nb())` for the log, identity and
sqrt links, theta fixed (`nb(theta=)`, `NegativeBinomial(fixed=True)`) or
estimated (`nb(theta=-start)`), fixed `sp` and `working_model_reml`. Code:
`families/negative_binomial.py` (pinned `nb()` arithmetic at explicit theta),
`links/links.py::source_g2g`, `fitting/discrete_working.py` (extended and
theta kernels), `execution/working_theta.py` (`estimate.theta`),
`execution/regular_working_model.py` (nuisance order) and
`execution/discrete_regular.py` (`_DiscreteExtendedRows`, `_ThetaNuisance`).

| Element | Pinned source | JaxGAM |
| --- | --- | --- |
| Extended family: `scale <- 1` (`nb` has no `family$scale`), `theta <- getTheta()` | `bam.r:464-475`, `545` | known scale; rows hold log theta, `exp` by libm |
| Step control and `dev0` with the current theta: `sum(dev.resids(G$y, mu0, G$w, theta))` | `bam.r:575-588` | `dev0` re-evaluated by a deviance scan when theta is estimated |
| From iteration two, `estimate.theta(theta, family, y, mu, scale=1, wt=G$w, tol=1e-7)` after step control, before the working model | `bam.r:613-621`, `efam.r:5-90` | `_ThetaNuisance.update`: one scan per `nlogl` call |
| `nlogl`: `dev/2 - ls`, `g = colSums(Dth)/2 - lsth1`, `H = colSums(Dth2)/2 - lsth2`; Newton with `|H|`, `1e-5` floor, step cap four, `eps^0.75` halving (25), `|g| <= tol(|nll|+1)` | `efam.r:14-89` | `source_estimate_theta`, R's long-double sums |
| `dDeta` Newton weights (`rho == 0`): `w = Deta2/2`, `z = (eta-offset) - Deta.Deta2`, non-finite rows zeroed | `bam.r:624-634`, `gam.fit4.r:4-45` | `source_newton_working`; `g2g` from `fix.family.link` |
| `nb()` `dev.resids` (`mu<=0 -> NA`), `Dd`, `ls` | `efam.r:199-275` | `source_dev_resids_at`, `source_dd`, `source_ls_rows`: R's operation order, no stability rewrites |
| The converging iteration updates theta before `break`; residuals use it | `bam.r:613-674`, `2747-2753` | final deviance re-evaluated at the final theta |

The penalized deviance of an iteration keeps the theta it entered with; the
working model, the next `dev0` and the final deviance use the updated one.
A non-finite mean at iteration two fails inside `estimate.theta` when theta
is estimated (R: `if (sum(uconv))` on `NA`) and at `stop("Non-finite
deviance")` when it is fixed; both routes stop at the same point (gated).

Validation (worst ratio to STRICT, macOS host / Linux arm64 image at
`c24e5c6`; every field STRICT, no MODERATE, no LOOSE):

| Gate | Field | Host | Linux |
| --- | --- | --- | --- |
| pinned `bam`, fixed sp | coefficients | 5.9e-3 | 1.4e-2 |
| | fitted values | 1.2e-4 | 2.0e-4 |
| | deviance | 9.8e-5 | 2.1e-4 |
| | EDF | 4.6e-5 | 4.2e-5 |
| | score | 3.9e-5 | 8.6e-5 |
| | theta | 7.2e-4 | 1.6e-3 |
| pinned `bam`, working-model REML | coefficients | 2.3e-3 | 3.1e-3 |
| | fitted values | 2.5e-4 | 3.0e-4 |
| | deviance | 8.0e-5 | 7.3e-5 |
| | EDF | 3.1e-5 | 3.2e-5 |
| | score | 3.1e-5 | 2.9e-5 |
| | smoothing parameters | 0.67 | 0.27 |
| | theta | 4.6e-4 | 3.5e-4 |
| matched state | `estimate.theta` update | 7.1e-4 | 8.0e-4 |
| | `X'WX` / `X'Wz` | 5.6e-3 / 5.6e-4 | 3.7e-3 / 1.5e-3 |
| | `y.norm2` | 8.2e-2 | 7.5e-2 |
| | proposal coefficients | 5.6e-4 | 0 |
| | gradient / Hessian | 1.6e-3 / 5.4e-4 | 6.7e-4 / 4.4e-4 |
| | `ldetXXS` / `ldetS` | 2.3e-6 / 2.2e-6 | 3.1e-6 / 2.7e-6 |
| | raw `Sl.fitChol` step | 2.1 | 1.05 |

The raw steps above one are the coupled log/tensor_by case; they pass the
step gate of this PR (STRICT backward error plus the perturbation bound),
as for the regular families. Pinned `nb()` row terms (`dev.resids`, `Dd`,
`ls`, `dDeta` `w`/`z`) and `estimate.theta` from three starts match R at
STRICT for every link. Tie replays (host): one decision for
log/estimated fixed sp (explicit design and `bam`) and two for
identity/estimated fixed sp; the unreplayed public fits stay within
MODERATE (coefficients 7.0e-2 of STRICT, theta 6.0e-4). Linux needs no
replay, and no matched-state NB iteration needs one on either platform.
The data are a runtime NB generator whose selected sp stay finite (every
log sp below four); weaker signals reproduce the drifting-sp phenomenon
above for NB too, and none was admitted. The identity link uses fixed sp
`(5, 10, 1)`: the shared fixed-sp vector leaves the NB domain at iteration
two in both implementations (gated as the domain exit).

JaxGAM's dense `pirls_loop` does not converge for NB/sqrt at this fixed
theta (its penalized deviance still moves after 2000 iterations), so the
explicit-design fixed-point check covers log and identity only; this is a
dense-route finding outside the discrete route.
