# Scaling acceptance cases (plan §8)

This document fixes the representative cases for the scaling plan's final
acceptance step (`jaxgam_scaling_implementation_plan.md` §8). It ships beside
the results document and says, for each case, what is run, which plan
question it answers, what is recorded and the exact command. It contains no
measurements: the dry runs listed below only prove that each command runs and
writes a valid record at a tiny size, and no timing taken on the preparation
host is evidence. No speedup claim is made, and different estimators are never
compared as one quantity.

## Conventions

- **Host.** The cases are sized for one 10-core, 16 GB host (the preparation
  host is an Apple M4 laptop). Where the plan suggests a size that host cannot
  run, the row says what replaces it and why.
- **Outputs.** Every command writes outside the repository, under one run
  directory `RUN=/path/outside/checkout/s8-<date>`. Only this document and
  the results document are tracked.
- **Command prefix.** Every command runs from the checkout as
  `caffeinate -s -i env -u JAX_COMPILATION_CACHE_DIR OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <command>`
  (`caffeinate` only on macOS; see the run plan).
- **Harness.** `make benchmark-scaling ARGS=...` runs
  `scripts/benchmark_scaling.py`. Each record holds a setup child and a fit
  child, each a fresh process with its own temporary JAX compilation cache. The
  fit child runs one cold fit and `--repeats` warm fits, each synchronized
  through coefficient completion. `--memory-pair-n SMALL LARGE` runs two such
  cases at fixed batch and p and reports the RSS above each child's measured
  baseline.
- **Common harness options** (abbreviated `P` below):
  `--result-mode prediction --repeats 1 --output $RUN/<ID>.json`.
- **Formulas.** The harness generates uniform covariates, seed 42.
  - `additive(m, k)`: `y ~ s(x0, k=k, bs='cr') + ... + s(x{m-1}, k=k, bs='cr')`;
    p = m(k-1)+1 with m penalties.
  - `tp(k)`: `y ~ s(x0, x1, k=k, bs='tp')`.
  - `gp(k)`: `y ~ s(x0, k=k, bs='gp')`.
  - `shared(k)`: `y ~ s(x0, k=k, bs='cr') + te(x0, x1, k=k, bs='cr')`.
  - `ti(k)`: `y ~ s(x0, k=k, bs='cr') + s(x1, k=k, bs='cr') + ti(x0, x1, k=k, bs='cr')`.
  - `tensor(k)`: `y ~ te(x0, x1, k=k, bs='cr')`.
  - Gaussian noise s.d. 0.3; Poisson from the harness's EFS generator; NB
    drawn at theta 3 and fitted with theta estimated from 1.0.
- **Route names.** dense = public dense PIRLS at fixed sp; dense-reml = dense
  Newton REML; stream-qr = streamed pivoted-QR PIRLS at fixed sp (all sp 1.0);
  stream-exact = exact streamed REML (L-BFGS-B, adjoint derivatives);
  stream-gaussian-qr = frozen Gaussian QR REML (one preparation scan);
  discrete-wm = continuous-discrete working-model REML (`bam(discrete=TRUE)`
  performance iteration); operator = matrix-free PCG.
- **Route limits that shape the matrix.** Streamed preparation supports only
  univariate `cr`/`cs` smooths and parametric terms, so tp, gp, tensor, `ti`
  and `re` cases run on the dense, discrete or operator routes. The discrete
  route needs `factor_names` in `ContinuousDiscretizationControl` for a factor
  `re` term, which the harness cannot pass (gap G8). Operator EFS is public
  only for canonical Poisson with one indexed random effect, 257 to 4096
  levels.

## Case matrix

Columns: **ID**; **model**; **family/link**; **n**; **p / penalties**;
**route, smoothing**; **question** (from plan §8); **record** (beyond the
fields every harness record carries: setup s, cold s, warm s, compile events,
iterations, scans and bytes, prediction and SE s, RSS, workspace, suspension
flag); **command**.

### 1. Large n, modest p

Plan question: does streamed memory plateau, and does Gaussian optimization
stop reading rows?

Replacement for 10m rows: only the one-scan Gaussian QR route (L6) is run at
10m. The exact streamed routes read the source on every trial (488 scans per
fit at n=1m in the PR11 runtime study), so 10m rows would take hours per fit,
and dense routes would hold a 10m x 101 design (8 GB). The in-memory source
alone is 0.9 GB at 10m rows and 11 columns. The plateau question is answered
by memory pairs at 1m and 3m.

| ID | model | family/link | n | p / pen. | route, smoothing | record | command |
|---|---|---|---|---|---|---|---|
| L1 | additive(10, 11) | Gaussian/identity | 1m and 3m pair | 101 / 10 | stream-gaussian-qr, REML | fit scans equal at both n and independent of iterations; known workspace plateau; RSS above baseline | `make benchmark-scaling ARGS="--execution stream-gaussian-qr --smooth-count 10 --k 11 --batch-rows 65536 --memory-pair-n 1000000 3000000 P"` |
| L2 | additive(2, 10) | Gaussian/identity | 100k and 1m pair | 19 / 2 | stream-exact, REML | scans per trial; derivative s; workspace plateau | `make benchmark-scaling ARGS="--execution stream-exact --smooth-count 2 --k 10 --batch-rows 65536 --memory-pair-n 100000 1000000 P"` |
| L3 | additive(10, 11) | Poisson/log | 100k and 1m pair | 101 / 10 | stream-qr, fixed sp | fit scans; workspace plateau | `make benchmark-scaling ARGS="--execution stream-qr --family poisson --smooth-count 10 --k 11 --batch-rows 65536 --memory-pair-n 100000 1000000 P"` |
| L4 | additive(10, 11) | Gaussian/identity | 1m, then 3m | 101 / 10 | discrete-wm | fit scans (expect 2); fit peak; compact rows | `make benchmark-scaling ARGS="--execution discrete-wm --smooth-count 10 --k 11 --batch-rows 65536 --n 1000000 P"`, then `--n 3000000` |
| L5 | additive(10, 11) | Gaussian/identity | 100k | 101 / 10 | dense-reml (reference: holds n x p by design) | RSS; compile | `make benchmark-scaling ARGS="--execution dense-reml --smooth-count 10 --k 11 --n 100000 P"` |
| L6 | additive(10, 11) | Gaussian/identity | 10m | 101 / 10 | stream-gaussian-qr, REML | as L1 at the plan's largest size | `make benchmark-scaling ARGS="--execution stream-gaussian-qr --smooth-count 10 --k 11 --batch-rows 65536 --n 10000000 P"` |
| L7 | additive(100, 11) | Gaussian/identity | 100k | 1001 / 100 | stream-gaussian-qr, REML, 1 GiB budget | finding F1: workspace preflight at B=16384; stop status at B=4096 | `make benchmark-scaling ARGS="--execution stream-gaussian-qr --smooth-count 100 --k 11 --batch-rows 16384 --memory-budget-mib 1024 --n 100000 P"`; repeat with `--batch-rows 4096` and no budget |

### 2. Many additive terms

Plan question: are penalty storage, compilation and smoothing derivatives
controlled?

Replacement for m=500: the formula parser overflows Python's recursion limit
between 450 and 500 additive terms (finding F9). M2 is written at m=450 and is
blocked until F9 and F11 (discrete compilation time) are decided.

| ID | model | family/link | n | p / pen. | route, smoothing | record | command |
|---|---|---|---|---|---|---|---|
| M1 | additive(100, 10) | Gaussian/identity | 100k | 901 / 100 | discrete-wm, 512 MiB table, 1 GiB fit budget | backend compile s and count (F11); setup construction peak; proposal evaluations. If setup is still rejected, record the required bytes and rerun with a larger table budget | `make benchmark-scaling ARGS="--execution discrete-wm --smooth-count 100 --k 10 --batch-rows 65536 --discrete-table-mib 512 --memory-budget-mib 1024 --n 100000 P"` |
| M2 | additive(450, 10) | Gaussian/identity | 100k | 4051 / 450 | discrete-wm, 2 GiB table and budget | as M1 (blocked: F9, F11) | `make benchmark-scaling ARGS="--execution discrete-wm --smooth-count 450 --k 10 --batch-rows 65536 --discrete-table-mib 2048 --memory-budget-mib 2048 --n 100000 P"` |
| M3 | additive(10, 10) | Poisson/log | 100k | 91 / 10 | stream-exact, REML | derivative s; outer iterations; cumulative PIRLS passes; stop status (F10) | `make benchmark-scaling ARGS="--execution stream-exact --family poisson --smooth-count 10 --k 10 --batch-rows 65536 --n 100000 P"` |
| M4 | additive(100, 10) | Gaussian/identity | 20k | 901 / 100 | dense-reml | Newton iterations; compile; RSS (dense m-penalty reference) | `make benchmark-scaling ARGS="--execution dense-reml --smooth-count 100 --k 10 --n 20000 P"` |
| M5 | additive(100, 20) | Gaussian/identity | 100k | 1901 / 100 | discrete-wm, 512 MiB table, 1 GiB budget | as M1 at k=20 | `make benchmark-scaling ARGS="--execution discrete-wm --smooth-count 100 --k 20 --batch-rows 65536 --discrete-table-mib 512 --memory-budget-mib 1024 --n 100000 P"` |

### 3. Expensive kernels (tp, gp)

Plan question: are all distance and kernel intermediates batch bounded?
The default knot cap (mgcv `max.knots` 2000) applies above 2000 unique rows.
Input dimension is d=1 (gp) and d=2 (tp); d>=3 is not in the harness (gap G9).

| ID | model | family/link | n | p / pen. | route, smoothing | record | command |
|---|---|---|---|---|---|---|---|
| K1 | tp(30) | Gaussian/identity | 100k | 30 / 1 | dense-reml | setup s and RSS (knot kernel, eigen basis); prediction s | `make benchmark-scaling ARGS="--execution dense-reml --model tp --k 30 --n 100000 P"` |
| K2 | gp(30) | Gaussian/identity | 100k | 30 / 1 | dense-reml | as K1 | `make benchmark-scaling ARGS="--execution dense-reml --model gp --k 30 --n 100000 P"` |
| K3 | tp(30) | Gaussian/identity | 1m | 30 / 1 | discrete-wm, 512 MiB table | setup construction peak against compact rows (F8) | `make benchmark-scaling ARGS="--execution discrete-wm --model tp --k 30 --batch-rows 65536 --discrete-table-mib 512 --n 1000000 P"` |
| K4 | gp(30) | Gaussian/identity | 1m | 30 / 1 | discrete-wm, 128 MiB table | as K3 | `make benchmark-scaling ARGS="--execution discrete-wm --model gp --k 30 --batch-rows 65536 --discrete-table-mib 128 --n 1000000 P"` |
| K5 | tp(k), gp(k), k = 10 and 30 | Gaussian/identity | 1k to 20k distinct rows | k / 1 | operator, fixed sp | preparation peak bound against distinct rows (F3) | Appendix A, F3 |

### 4. Shared variables and tensors

Plan question: do dependence detection and constraint preparation avoid
n-squared and n-by-p allocations?

| ID | model | family/link | n | p / pen. | route, smoothing | record | command |
|---|---|---|---|---|---|---|---|
| S1 | shared(6) | Gaussian/identity | 100k | 40 / 3 | dense-reml | setup s and RSS (gam.side dependence on a shared covariate) | `make benchmark-scaling ARGS="--execution dense-reml --model shared --k 6 --n 100000 P"` |
| S2 | ti(6) | Gaussian/identity | 100k | 36 / 4 | dense-reml | as S1 | `make benchmark-scaling ARGS="--execution dense-reml --model ti --k 6 --n 100000 P"` |
| S3 | shared(6) | Gaussian/identity | 1m | 40 / 3 | discrete-wm | setup construction peak; compact rows; fit scans | `make benchmark-scaling ARGS="--execution discrete-wm --model shared --k 6 --batch-rows 65536 --n 1000000 P"` |
| S4 | ti(6) | Gaussian/identity | 1m | 36 / 4 | discrete-wm | as S3 | `make benchmark-scaling ARGS="--execution discrete-wm --model ti --k 6 --batch-rows 65536 --n 1000000 P"` |
| S5 | tensor(6) | Gaussian/identity | 1m | 36 / 2 | dense, fixed sp, `result="inference"` | RSS of the n x p reference | `make benchmark-scaling ARGS="--execution dense --model tensor --k 6 --n 1000000 --result-mode inference --output $RUN/S5.json"` |

### 5. High-cardinality effects

Plan question: is storage proportional to indices and nonzeros rather than
one-hot dense arrays? Dense and discrete routes are not used: a dense 1m-level
system is 8 TB, and the harness has no `re` model (gap G8).

| ID | model | family/link | n | p / pen. | route, smoothing | record | command |
|---|---|---|---|---|---|---|---|
| H1 | `y~s(g,bs='re')`, 2 rows per level | Gaussian/identity | 20k, 200k, 2m | 10k, 100k, 1m / 1 | operator, fixed sp, PCG | peak bound against budget; scans reconciled; Krylov actions; RSS high water | `make benchmark-operator LEVELS=10000 OUTPUT=$RUN/H1`; repeat with `LEVELS=100000` and `LEVELS=1000000` |
| H2 | `y~0+s(g,bs='re')` | Poisson/log | 8192 (2 per level) | 4096 / 1 | operator EFS | score bound, stop reason, iterations, scans, fit against dense EFS | `make benchmark-operator-efs-calibration ARGS="--regime indexed_poisson --levels 4096 --complete" OUTPUT=$RUN/H2` |
| H3 | `y~s(x,bs='cr',k=6)+s(g,bs='re')` | Gaussian/identity | 4 rows per level | 1k and 10k levels | operator, fixed sp, paired PCG | actions and verified residuals with and without the coarse preconditioner | `uv run python scripts/benchmark_operator_preconditioner.py --levels 1000 10000 --output-dir $RUN/H3` |
| H4 | H1 fit at 10k levels | Gaussian/identity | 20k | 10k / 1 | `bulk_link_variance`, 48 probes | probe scans; calibration maximum; false-alarm rate (F6) | no script (gap G5); Appendix A, F6 for the procedure |

### 6. NB count tails

Plan question: is derivative work bounded without losing first and second
derivative accuracy? The harness's NB generator draws at theta 3 with no
outliers; outliers and theta near the Poisson limit are covered only by the
lgamma derivative benchmark (gap G10).

| ID | model | family/link | n | p / pen. | route, smoothing | record | command |
|---|---|---|---|---|---|---|---|
| N1 | additive(2, 10) | NB/log, theta estimated | 100k | 19 / 2 | dense-reml | theta; iterations | `make benchmark-scaling ARGS="--execution dense-reml --family nb --smooth-count 2 --k 10 --n 100000 P"` |
| N2 | additive(2, 10) | NB/log, theta estimated | 1m | 19 / 2 | discrete-wm | theta; fit scans per performance iteration | `make benchmark-scaling ARGS="--execution discrete-wm --family nb --smooth-count 2 --k 10 --batch-rows 65536 --n 1000000 P"` |
| N3 | additive(2, 10) | NB/log, theta estimated | 100k | 19 / 2 | stream-qr, fixed sp | fit scans; workspace | `make benchmark-scaling ARGS="--execution stream-qr --family nb --smooth-count 2 --k 10 --batch-rows 65536 --n 100000 P"` |
| N4 | additive(2, 10) | NB/log, theta estimated | 100k, then 1m | 19 / 2 | stream-exact, REML | stop status and projected gradient (F2); derivative s | `make benchmark-scaling ARGS="--execution stream-exact --family nb --smooth-count 2 --k 10 --batch-rows 65536 --n 100000 P"`, then `--n 1000000` |
| N5 | NB lgamma theta derivative kernel | NB | 1k and 1m rows; max count 16 and 4096; one outlier | – | prefix kernel and legacy recurrence | cold and warm s; compiler temp bytes; prefix table bytes | `env -u JAX_COMPILATION_CACHE_DIR uv run python scripts/benchmark_nb_lgamma_derivatives.py --rows 1000 1000000 --max-counts 16 4096 --repeats 5 --outlier` |

### 7. Numerical stress

Plan question: do solvers, derivatives, rank decisions and uncertainty remain
valid? Concurvity has no case or generator yet (gap G11). The open findings
are rows here; their classification is in the findings table.

| ID | case | route | record | command |
|---|---|---|---|---|
| T0 | existing zero/extreme-weight, offset, near-perfect and zero-residual gates | all | pass/fail; pinned R parity only counts in the pinned environment | `make test-local PYTEST_ADDOPTS="tests -k 'zero_weight or offset or near_perfect or nearperfect or exact_fit or zero_residual or nearly_exact' -q"` (acceptance: `make test`) |
| T1 | F1: Gaussian QR p=1001, 100 smooths | stream-gaussian-qr | preflight bytes; stop status | L7 and Appendix A |
| T2 | F2: exact streamed NB with estimated theta | stream-exact | stop status; projected gradient | N4 |
| T3 | F3: operator setup accounting against distinct rows | operator | preparation bound | K5, Appendix A |
| T4 | F4: dense `te` REML with log-sp spread >= 20 | dense-reml vs pinned R | score and fitted differences; stop status | Appendix A |
| T5 | F5: `y ~ x + s(x, bs='gp')` converged flag | dense-reml | converged, inner PIRLS status | Appendix A |
| T6 | F6: bulk calibration false-alarm rate | operator bulk variance | standardized-error distribution over seeds | Appendix A |
| T7 | F10: exact streamed Poisson, 10 smooths | stream-exact | stop status | M3 |

## Harness gaps

Filled in the harness (with tests; `scripts/benchmark_scaling.py` at 93%
statement and branch coverage, every new line covered):

- **Compilation** (`4e94406`): a `jax.monitoring` listener records the count
  and seconds of tracing, lowering, backend compilation and persistent-cache
  events for every cold and warm fit (`cold_fit_compilation`,
  `cold_fit_backend_compile_s`, `warm_fit_*`). Before, compilation was folded
  into `cold_compile_and_fit_s`.
- **Models** (`4e94406`): `--model tp | gp | shared | ti` for the dense and
  discrete routes.
- **G1 host suspension** (`94f0e61`): each fit is also timed by the wall clock;
  `host_suspension_suspected` flags a record whose wall duration exceeds its
  `perf_counter` duration by more than max(1 s, 5%). On the preparation host,
  two dry runs took 962 s and 915 s of wall time against 5.4 s and 7.2 s of
  `perf_counter` time because the laptop slept between maintenance wakes; JAX's
  compile events use `time.time()` and counted the sleep as compilation.
- **Budgets** (`94f0e61`, `50081b7`): `--discrete-table-mib` (discrete
  `DiscretizationBudget` table and batch limits) and `--memory-budget-mib`
  (`FitControl.memory_budget_bytes` on streamed and discrete fits and setup).
  The defaults rejected M1, M5, K3, K4, M2 and F1 before any fit.

Gap status (G7 onward filled in the second round, one commit each):

| Gap | What is missing | Proposal |
|---|---|---|
| G2 | compiler memory estimates (`compiler_memory_bytes` is always `None`) | per-route hook that lowers the route's main jitted kernel at the case shape and records `memory_analysis()`; ten kernel tests already do this and are the current evidence |
| G3 | temporary disk other than the compilation cache: discrete selector memmaps, the SQLite knot scope | run each child with `TMPDIR` and `selector_memmap_directory` in task-owned directories and sample their peak size from a thread |
| G4 | **filled.** `--execution operator` (fixed sp, PCG; every model, Gaussian or Poisson) and `--execution operator-efs` (calibrated Poisson EFS; model `re-only` = `y ~ s(g, bs='re')`) run in the isolated harness: a setup child that runs `prepare_operator_model` with the fit's own preparation budget (one eighth of it for EFS) and records scans, bytes, preparation peak bound and coefficient layout without expanding penalties to p x p; a fit child with a task-owned cache, compile events, `operator_diagnostics` (scans by kind, Krylov actions and iterations, peak bound, executable temporaries, inner status, gradient norm) and Krylov iterations as the inner count. Fixed-sp results have no score, EDF or Gaussian scale by design, so those fields are recorded as not provided; `predict(se_fit=True)` is not offered on these routes. NB is rejected because the operator fits NB only at fixed theta | – |
| G5 | **filled.** `scripts/benchmark_bulk_variance.py` (`make benchmark-bulk-variance`) fits one operator model (the bulk-variance test model, or `s(x,bs='cr',k=6)+s(g,bs='re')` with L levels) and calls the public `bulk_link_variance` once per seed with its raise limit out of reach. Per seed and calibration row it records exact variance, estimate, Monte Carlo SE, the current standardized error and the ratio estimate/exact, plus whether the default 5-SE rule would raise, and the provider diagnostics (probe and normal solves, Krylov iterations, every scan count, maximum PCG relative residual, peak bound). The summary gives the old rule's exceedance rate with a Clopper-Pearson interval and Kolmogorov-Smirnov tests of probes x ratio against chi-square(probes), per row over seeds. One bulk pass over the training rows is timed. Task-owned compilation cache | – |
| G6 | `--memory-pair-n` accepts only streamed routes | allow discrete pairs (the comparison already reads `known_workspace_bytes`) |
| G7 | **filled.** An unconverged, non-finite or rejected fit, a rejected setup and a crashed child each write a `valid=false` record: outcome (`unconverged`, `nonfinite`, `fit_raised`, `setup_raised`, `child_failed`), status, stop reason and its source, projected gradient (exact streamed REML), the operator inner status and final gradient norm, smoothing parameters, theta, score, iterations, the error type and message, and no timing fields. Comparisons over an invalid record are marked unavailable. The record is written first; then the command exits 3 and prints each invalid child on stderr | – |
| G8 | **filled.** `--model re --levels L`: `y ~ s(x0, k=k, bs='cr') + s(g, bs='re')` with L declared levels (unobserved levels stay declared) and a normal group effect; the factor reaches every source as a categorical Series, the discrete routes receive `factor_names=("g",)`, and the input hash covers codes and level labels. Dense, discrete and (G4) operator routes | – |
| G9 | kernels only in d=1 (gp) and d=2 (tp) | `--model tp3` (`s(x0,x1,x2)`) |
| G10 | NB generator fixed at theta 3 without outliers | `--nb-theta` and `--outlier-fraction` |
| G11 | no concurvity, zero or extreme weight, offset or near-perfect generators | models with `x1 = x0 + small noise`, weights and offsets through `ArrayRowSource` |
| G12 | device peak memory: the CPU backend exposes no allocator statistics | none on CPU; record as unavailable |
| G13 | derivative time only on stream-exact | route diagnostics for dense Newton, Gaussian QR and discrete working-model derivative stages |
| G14 | dense routes expose no inner PIRLS count (see F5) | public inner-iteration diagnostics |
| G15 | **filled.** `scripts/benchmark_manifest.py` (`make benchmark-manifest`) writes one JSON manifest per run directory: `--phase start` before the first case records OS, CPU model and feature flags, logical CPUs, RAM, Python and every installed package version, dtype and JAX x64/backend/devices, NumPy and SciPy BLAS/LAPACK builds, the source-LAPACK provider (scipy-openblas32 version and OpenBLAS config), the thread and cache environment, the checkout commit and whether its tree is clean, load averages and (macOS) sleep assertions; `--phase end` adds the end checkout state, end load and `same_checkout` | – |
| P4 | structural guards missing on some routes (allocation table) | forbidden-call or jaxpr guards as listed there |
| P5 | the coarse preconditioner's bytes are not split out in operator records | report it in `operator_diagnostics` |

Coverage by quantity and route:

| Quantity | dense, dense-reml | stream, stream-qr | stream-exact | stream-gaussian-qr | discrete, discrete-wm | operator fixed sp | operator EFS | bulk variance |
|---|---|---|---|---|---|---|---|---|
| Setup time, separate process | H | H | H | H | H | H | H | fit time B |
| Cold compilation | H | H | H | H | H | H | H | – |
| Warm fit | H | H | H | H | H | H | H | per-seed sketch and calibration B |
| Source scans and bytes | n/a | H | H | H | H | H | H | scans by kind B |
| Inner / outer iterations | outer H; inner – (G14) | public count | H both | H outer | H both | H (PIRLS / Krylov) | H (EFS / Krylov) | – |
| Derivative time | – (G13) | n/a | H | – (G13) | – (G13) | n/a | – (G13) | n/a |
| Prediction / SE time | H | H | H | H | H | prediction H; SE not offered | prediction H; SE not offered | bulk pass B |
| Host peak RSS | H | H | H | H | H | H | H | B |
| RSS minus baseline | – | H pair | H pair | H pair | – (G6) | – (G6) | – (G6) | – |
| Device peak | – (G12) | – | – | – | – | – | – | – |
| Compiler memory | – (G2) | – | – | – | – | – | – | – |
| Temporary disk | cache only (G3) | cache only | cache only | cache only | cache only; memmaps unmeasured | cache only | cache only | – |
| Synchronized timing | H | H | H | H | H | H (host PCG returns NumPy) | H | – |
| Isolated child processes | H | H | H | H | H | H | H | – |
| Isolated compilation cache | H | H | H | H | H | H | H | B |
| Host suspension flag | H | H | H | H | H | H | H | – |
| Invalid or rejected fit recorded (G7) | H | H | H | H | H | H | H | – |

H = `scripts/benchmark_scaling.py`; B = `scripts/benchmark_bulk_variance.py`; O = the named other script; – = gap.

## Allocation-shape claims: method per route

Plan §8 forbids four shapes on exact direct paths, and two more on operator
paths. Peak RSS alone cannot prove an absence, so each claim needs a structural
instrument (a forbidden call, a jaxpr or compiled-memory inspection, or a
symbolic bound checked before allocation), with the RSS memory pair as
corroboration only.

| Claim | Route | Existing instrument (test) | Method | Missing |
|---|---|---|---|---|
| No n x n Q | dense (gam.side dependence) | `tests/test_constraints.py::TestFixDependence::test_tall_input_uses_raw_qr_without_full_q`, `test_packed_reflectors_match_full_qt` | SciPy QR is wrapped to fail on a full-Q request at n=20,000 | nothing for the dense path |
| No n x n Q | stream, discrete, operator | none needed by construction (no row-dimension QR; dependence on prepared or compact blocks) | – | a guard like the dense one on the streamed prepare dependence step (proposal P4) |
| No n x p X in streaming | stream / stream-qr | `test_stream_fixed_public_family.py::test_public_fixed_sp_stream_never_builds_training_matrix`; `test_validation_matrix.py::test_public_exact_large_row_fit_stays_within_stream_budget` | `ModelSetup.build` patched to raise; budgeted run | – |
| No n x p X in streaming | stream-exact | `test_nb_stream_reml.py::test_nb_batch_compiled_workspace_is_bounded_by_batch_and_coefficient_size`; `test_nb_theta_stream.py::test_conditional_n_growth_keeps_numeric_buffers_histories_and_ledger_bounded` | compiled `memory_analysis` and growth in n | the forbidden-`ModelSetup.build` guard is applied only to fixed-sp; extend to exact REML (P4) |
| No n x p X in streaming | stream-gaussian-qr | `test_gaussian_stream.py::test_stream_provider_jaxprs_and_retained_state_have_no_row_dimension`; `test_gaussian_optimizer.py::test_provider_score_and_derivative_jaxprs_have_no_observation_arrays` | jaxpr shapes contain no n | – |
| No n x p X | discrete | `test_discrete_ops.py::test_factor_tensor_jaxpr_has_no_expanded_batch_level_tensor` | jaxpr inspection; harness `discrete_setup` ledger (compact rows, construction peak) | no forbidden-`ModelSetup.build` guard on a successful discrete fit (P4) |
| No n x p X | operator | `test_operators.py::test_high_cardinality_jaxpr_has_no_expanded_design_or_square_system` | jaxpr inspection | a forbidden-`ModelSetup.build` guard on a public operator fit (P4) |
| No m x p x p penalty stack | stream, stream-exact, gaussian-qr | `test_efs_stream_start.py::test_startup_bound_covers_shared_balancing_dense_tuple_for_many_local_penalties`; `test_rank_coordinates.py::test_global_penalty_scale_bounds_scratch_for_wide_penalty`; harness `_canonical_penalty_contract` reads local blocks | local-block storage, bounded scratch | a forbidden `PenaltyStructure.materialize` guard on these routes, as the operator tests have (P4) |
| No m x p x p penalty stack | dense | none: dense FittingData holds per-penalty matrices by design | – | out of scope (dense reference route) |
| No m x p x p penalty stack | operator | `PenaltyStructure.materialize` patched to raise in `test_operator.py`, `test_operator_fit.py`, `test_operator_logdet.py`, `test_operator_penalty_determinant.py`, `test_operator_efs_trace.py`, `test_operator_variance.py`, `test_operator_bulk_variance.py` | forbidden call | – |
| No n x r x d distance differences | dense, discrete (tp, gp prediction and design rows) | `test_tprs.py::TestCoveragePaths::test_prediction_distance_work_is_row_batched`; `test_gaussian_process.py::TestSetupInvariants::test_prediction_distance_work_is_row_batched` | patched kernel records batch sizes (`DISTANCE_BATCH_ROWS`) | the training-time knot kernel and the discrete compact-row kernel have no row-batch test; K3/F8 show the discrete construction bound growing with compact rows |
| No n x r x d distance differences | operator | `formula/operator_workspace.py` chunks by `DISTANCE_BATCH_ROWS`; covered indirectly by `test_operator_radial.py` | – | a patched-batch test as for tp/gp (P4) |
| No p x p G/H | operator | `test_operator.py::test_symbolic_product_storage_and_runtime_without_global_square`; `test_operator_fit.py::test_crossed_numeric_large_levels_fit_without_global_coefficient_square`, `test_mixed_numeric_large_levels_fit_without_global_coefficient_square`; `test_operator_logdet.py::test_large_symbolic_logdet_stays_below_global_square` | symbolic sizes far above RAM run inside a small budget | – |
| No full covariance | operator | `test_operator_variance.py::test_selected_large_symbolic_variance_has_no_coefficient_square`; `test_operator_bulk_variance.py::test_bulk_symbolic_large_p_preflights_and_never_materializes_covariance`; `test_prediction_result.py::test_point_only_result_retains_no_rows_or_dense_covariance` | symbolic large p; forbidden materialization | – |
| Coarse preconditioner accounted separately | operator | `test_preconditioners.py::test_large_diagonal_budget_is_checked_before_transfer_or_factor` | budget preflight | the harness/benchmark records do not split the preconditioner bytes out (P5) |

Across all routes the harness records no compiler memory estimate
(`compiler_memory_bytes` is always `None`; gap G2). Ten kernel tests already
read `jax.jit(...).lower(...).compile().memory_analysis()`; the acceptance
step should cite those, not the harness.

## Dry runs (preparation host, not evidence)

Every command above was run once at a tiny size on 2026-11-17 to prove that
it runs and writes a valid record (logs outside the repository). Harness
records were checked for JSON validity, within-run consistency (setup and
fit children on the same commit, input, formula and route) and convergence.
Sizes: n = 2,000 to 8,192 (memory pairs 2,000/4,000 and 4,096/8,192); the
operator scripts at 300 levels; the lgamma benchmark at 1,000 rows.

| ID | Dry-run size | Result |
|---|---|---|
| L1 | pair 4,096 / 8,192, B=1024 | valid; 17 fit scans at both n with 11 and 13 iterations; known workspace equal (plateau). At n < B (pair 2,000/4,000, B=4096) the workspace differs, so real pairs need n much larger than B |
| L2 | pair 2,000 / 4,000 | valid; plateau; 199 and 217 fit scans |
| L3 | pair 2,000 / 4,000 | valid; plateau; 34 fit scans at both n |
| L4, L5 | 4,000; 2,000 | valid |
| L6 | (same command as L1 at 10m) | covered by L1 |
| L7 / F1 | 20,000, 100 smooths | B=16384: preflight rejects (F1 reproduces). With 1 GiB: runs, `iteration limit`. B=4096: `iteration limit` |
| M1 | 5,000 | default table budget rejects (147 MB > 64 MiB); with 256 MiB valid, cold 723 s of which 664 s backend compilation, warm 32 s (F11) |
| M2 | 2,000 at 450 smooths | m=500 does not parse (F9). m=450 needs 1.37 GB of setup at n=2,000 (over 1 GiB); blocked |
| M3 | 2,000 | not converged: `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT; 4 source step window(s)` (F10); no record (G7) |
| M4 | 2,000 | valid; p=901, 17 Newton iterations; cold 263 s, warm 232 s |
| M5 | 5,000 | default table budget rejects (390 MB); 512 MiB table then fit budget rejects (561 MB > 512 MiB); with 1 GiB valid, cold 845 s of which 757 s compilation, warm 60 s, workspace 1.0 GB (F11) |
| K1, K2 | 2,000 | valid (p=30) |
| K3 | 2,000 | default budget rejects (262 MB); 512 MiB valid (F8) |
| K4 | 2,000 | default predictor-workspace budget rejects (69.4 MB > 64 MiB); 128 MiB valid |
| K5 / F3 | 1,000 to 20,000 rows | runs; see F3 |
| S1-S5 | 2,000 | valid (p=40, 36, 40, 36, 36) |
| H1 | 300 levels | valid; scans reconciled |
| H2 | 300 levels, `--complete` | valid JSONL record |
| H3 | default levels 20, 80 | valid |
| H4 / F6 | test model, 240 seeds | runs; see F6 |
| N1-N4 | 2,000 | valid; N4 converged (projected gradient 4.7e-11 at n=20,000 too) |
| N5 | 1,000 rows | valid; records compiler temporary bytes |
| T0 | the selection as written | 92 passed, 0 failed, 4,683 deselected (local pinned R 4.5.2 / mgcv 1.9-3 via rpy2; acceptance still runs in the pinned image) |
| T4, T5, T6 | Appendix A | run; see findings |

Two dry runs (L2, L5) spanned host sleeps: 962 s and 915 s of wall time
against 5.4 s and 7.2 s of `perf_counter` time. The harness now flags this
(G1).

## Run plan for the real measurements

**Order.**
1. Validity sweep, short cases: K1-K4, S1-S5, N1-N3, N5, K5, T4, T5.
2. Large n with the machine to itself: L1, L2, L3, L4, L5.
3. Many terms with the machine to itself: M1, M5, M4, M3, L7.
4. High cardinality: H1 (10k, 100k, then 1m levels), H2, H3.
5. NB exact: N4 at 100k, then 1m.
6. L6 (10m rows) last, with the machine to itself.
7. F6 false-alarm measurement (Appendix A).
8. T0 and every R-parity gate in the pinned environment (`make test`, root).
M2 and H4 are not scheduled until F9/F11 and G5 are decided.

**Expected durations on a quiet machine.** These are estimates, not
measurements. They are extrapolated from the PR11 runtime study (loaded host)
and PR9's n=1m Gaussian QR runs, and from the cost structure the dry runs
exposed (for example the 11 to 13 minutes of compilation for 100 discrete
smooths). Each harness case is one setup child plus one cold and one warm fit.

| Group | Cases | Estimate |
|---|---|---|
| 1 | K1-K4, S1-S5, N1-N3, N5, K5, T4, T5 | 35 min |
| 2 | L1 14, L2 12, L3 6, L4 7, L5 2 | 41 min |
| 3 | M1 25, M5 30, M4 30, M3 20, L7 30 | 135 min |
| 4 | H1 40, H2 10, H3 5 | 55 min |
| 5 | N4 | 25 min |
| 6 | L6 | 30 min |
| 7 | F6, 1,000 seeds per shape, 3 shapes | 60 min |
| Total | | about 6.5 hours, excluding M2, H4 and the pinned suite |

**Machine to itself.** Every memory pair (L1-L3), L4, L6, L7, M1, M3-M5, H1 at
1m levels and N4 at 1m: their RSS and wall times are recorded as results. The
short cases in group 1 may share the machine only if their timings are not
reported.

**Host settings.**
- On macOS hold `caffeinate -s -i` (on AC power) for the whole session. Reject
  any record with `host_suspension_suspected=true`.
- Unset `JAX_COMPILATION_CACHE_DIR`: the harness gives each child a task-owned
  cache, and the lgamma benchmark creates its own only when the variable is
  unset.
- `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`; leave XLA's thread
  pool at its default and record `XLA_FLAGS`.
- No other agents, Docker VM or test runs during groups 2-6.

**Environment each run records.**
- Already in every harness record: commit, dirty-tree hash, loaded-source
  hash, harness hash; Python, JAX, jaxlib, NumPy, SciPy and jaxgam versions;
  platform, machine, CPU count, NumPy BLAS/LAPACK names, thread environment,
  dtype float64, cold-cache policy; seed, n, k, smooth count, batch rows,
  memory and discretization budgets, family, theta policy, strategy; the
  input generator and frame hash.
- The run manifest (G15) holds what the records lack. Write it before the
  first case with `make benchmark-manifest ARGS="--phase start --output
  $RUN/manifest.json"` and finish it after the last with `--phase end`. A
  run whose manifest reports `same_checkout=false` (commit changed or tree
  dirty) is not evidence. R and mgcv versions for parity runs come from
  `RBridge.check_versions()` in the parity records.
- The other scripts record less: `benchmark_operator.py` records commit,
  dirty flag and versions but uses the caller's cache. Run them with
  `JAX_COMPILATION_CACHE_DIR` pointing at a new empty directory per run.

**Where results go.** Raw records go to `$RUN/` outside the checkout. The
results document `docs/scale_jaxgam/acceptance_results.md` reports each row
of this matrix: its measured values, the commit, the manifest summary, and
"not converged" or "rejected" with the status where that happened. It makes
no speedup claim and never sets different estimators side by side as a speed
comparison.

## Findings (diagnosis only)

Classification: **reproduces** at the stated size; **does not reproduce at
this size**; **needs the full size**. Sizes stayed at n <= 20,000.

| ID | Finding | Smallest reproduction | Classification and observation |
|---|---|---|---|
| F1 | Gaussian QR, p = 1001, 100 smooths | `make benchmark-scaling ARGS="--execution stream-gaussian-qr --smooth-count 100 --k 11 --batch-rows 16384 --n 20000 P"`; then `--batch-rows 4096`; then `--batch-rows 16384 --memory-budget-mib 1024` | **Reproduces at n = 20,000.** B=16384: "Known streamed Gaussian QR workspace requires 558529632 bytes, exceeding FitControl.memory_budget_bytes=536870912". B=4096: `converged=False`, status `iteration limit`. With 1 GiB at B=16384 the preflight passes and the stop is the same `iteration limit`, so the non-convergence does not depend on the batch size. The status is L-BFGS-B's iteration limit, not the relative-reduction stop that the Gaussian QR route showed elsewhere; the harness keeps no projected gradient for an invalid fit (G7) |
| F2 | Exact streamed REML, NB with estimated theta, `additive(2, 10)` | N4 at `--n 20000` | **Does not reproduce at n = 20,000** (or 2,000): converged, `CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL; 1 source step window(s)`, projected gradient 4.7e-11, theta 2.947. **Needs the full size** (100k, 1m). The relative-reduction stop message was not seen at this size |
| F3 | Operator setup accounting against distinct covariate rows | Appendix A, F3 | **Does not reproduce at <= 20,000 distinct uniform rows**: the preparation bound did not change with n (2-D tp and 1-D gp: 3.78 MiB at k=10 for n = 1,000-16,000; about 537 MB at k=30 for n = 2,500-20,000). **Needs the full size** (about 30,000 distinct 2-D rows). Related observation: at k=30 the bound (536,968,xxx bytes tp; 537,136,760 gp) just exceeds the default 512 MiB budget, so tp/gp k=30 operator fits are rejected at every n tried |
| F4 | Dense `te()` REML with log smoothing parameters >= 20 apart | Appendix A, F4 | **Reproduces at n = 1,000** (`te(x0,x1,bs='cr',k=6)`, y = sin(4 pi x0) + N(0, 0.02^2)). R: log sp (-6.02, 14.55), spread 20.6, full convergence. JaxGAM: (-6.02, 23.78), stop `step failed`; score differs by 8.2e-3 and fitted values by up to 1.0e-3. At spread 15.6 (noise 0.05, n = 600) both converge, with score difference 2.2e-5 and fitted 6.2e-5 |
| F5 | `y ~ x + s(x, bs='gp')` reports converged while its inner loop hits the limit | Appendix A, F5 | **Does not reproduce through public results at n = 200**: `converged=True`, `full convergence`, 5 outer iterations; a fixed-sp refit at the selected sp converges in 2 PIRLS iterations. The inner PIRLS state inside Newton trials is not exposed (G14), so the case needs internal instrumentation (being addressed elsewhere) |
| F6 | `bulk_link_variance` raises above 5 Monte Carlo standard errors | Appendix A, F6 | **Reproduces** on the test model (Gaussian `s(x,bs='cr',k=6)+s(g,bs='re')`, 47 rows, 5 calibration rows, 48 probes). Over 240 fresh seeds the per-call maximum standardized error exceeded 5 twice (0.83%; Clopper-Pearson 95% interval 0.10%-2.98%), with maxima 7.60 and 4.48 in the two batches, median 1.5, 95th percentile 3.6-3.7, and 19 calls above 3. The earlier 3.76-3.78 came from four seeds |
| F7 | Discrete route with a factor `re` term | Appendix A, F7 | **Reproduces; usage, not numerics.** Without `factor_names` it fails with "Metric column 'g' must be real numeric data"; with `ContinuousDiscretizationControl(factor_names=("g",))` it fits and converges. The message does not name the missing declaration |
| F8 | Discrete 2-D tp construction bytes against compact rows | `--execution discrete-wm --model tp --k 30` at n = 500-2,000 with a 1 MiB table budget | **Reproduces**: required 16.9, 64.1 and 250.0 MiB at 500, 1,000 and 2,000 rows (x4 per doubling, up to the 2000-knot cap). The default 64 MiB budget rejects 2-D tp k=30 above about 1,000 compact rows. 1-D gp stays near 64 MiB because the grid caps its unique values |
| F9 | Formula parser recursion | `parse_formula` with 500 additive `s()` terms | **Reproduces**: `RecursionError` (`formula/parser.py` `_visit_binop`, two frames per term); 450 terms parse. The plan's m=500 case cannot be expressed |
| F10 | Exact streamed Poisson REML with 10 smooths | M3 at n = 2,000 | **Reproduces at n = 2,000**: `converged=False`, `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT; 4 source step window(s)`. Whether it persists at 100k is unknown |
| F11 | Compilation time of discrete working-model REML with many smooths | M1, M5 at n = 5,000 | **Reproduces at n = 5,000**: 664 s (100 smooths, k=10) and 757 s (k=20) of XLA backend compilation in the cold fit; warm fits 32 s and 60 s; no host suspension |

**F6: proposed false-alarm measurement.**
- Shapes: the test model (Gaussian), its Poisson variant, and an operator
  random effect with 10k levels plus one cubic smooth.
- Settings: probes 16, 48 and 128; calibration rows 1, 5 and 64.
- Per setting: 1,000 seeds from a stream disjoint from the tests' seeds.
- Record per seed: the maximum standardized error and the per-row
  standardized errors.
- Report: the exceedance rate of 5 with a Clopper-Pearson interval, and
  quantiles of the per-row errors against Student t with probes-1 degrees of
  freedom (the Monte Carlo SE is estimated from the same probes, which makes
  the ratio heavier-tailed than normal).
- Cost on the test model: about 0.7 s per seed in the dry runs.
- Choosing the limit (for example the measured 1-alpha quantile of the
  maximum over rows, or a t/Bonferroni limit) is a root decision.

## Appendix A: reproductions without a harness mode

These snippets use only the public API and runtime-generated data; run them
from the checkout with `PYTHONPATH=$PWD uv run python`.

```python
# F3: operator preparation bound against distinct rows (budget 1 MiB so every
# fit reports its bound instead of running).
import re, numpy as np, pandas as pd
from jaxgam import GAM, FitControl
from jaxgam.data.source import DataFrameRowSource
for formula in ("y~s(x0,x1,bs='tp',k=10)", "y~s(x0,bs='gp',k=10)"):
    for n in (1000, 2000, 4000, 8000, 16000):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"x0": rng.uniform(0, 1, n), "x1": rng.uniform(0, 1, n)})
        df["y"] = np.sin(2 * np.pi * df.x0) + rng.normal(0, 0.3, n)
        control = FitControl(execution="operator", linear_solver="pcg",
                             batch_rows=4096, memory_budget_bytes=1 << 20)
        try:
            GAM(formula, sp=[1.0], control=control).fit(
                DataFrameRowSource(df, response="y"), result="prediction")
        except MemoryError as error:
            print(formula, n, re.search(r"needs at most (\d+)", str(error)).group(1))
```

```python
# F4: dense te() REML against pinned R with a large log-sp spread.
import numpy as np, pandas as pd
from jaxgam import GAM
from tests.r_bridge import RBridge
rng = np.random.default_rng(4); n = 1000
df = pd.DataFrame({"x0": rng.uniform(0, 1, n), "x1": rng.uniform(0, 1, n)})
df["y"] = np.sin(4 * np.pi * df.x0) + rng.normal(0, 0.02, n)
formula = "y ~ te(x0, x1, bs='cr', k=6)"
r = RBridge(mode="rpy2").fit_gam(formula, df, "gaussian", "REML")
j = GAM(formula, family="gaussian").fit(df)
print(np.log(r["smoothing_params"]), np.log(j.smoothing_params), j.convergence_info,
      abs(j.score - r["reml_score"]),
      np.max(np.abs(np.asarray(j.fitted_values) - r["fitted_values"])))
```

```python
# F5: rank-deficient dense GP fit; refit at the selected sp to see PIRLS.
import numpy as np, pandas as pd
from jaxgam import GAM
rng = np.random.default_rng(5); x = np.sort(rng.uniform(0, 1, 200))
df = pd.DataFrame({"x": x, "y": np.sin(2 * np.pi * x) + rng.normal(0, 0.3, 200)})
fit = GAM("y ~ x + s(x, bs='gp')", family="gaussian").fit(df)
refit = GAM("y ~ x + s(x, bs='gp')", family="gaussian", sp=fit.smoothing_params).fit(df)
print(fit.converged, fit.convergence_info, fit.n_iter, refit.convergence_info, refit.n_iter)
```

```python
# F6: distribution of the bulk calibration maximum over seeds (test model).
import numpy as np, pandas as pd
from jaxgam.execution.operator import PCGControl
from jaxgam.execution.operator_bulk_variance import BulkOperatorVariance
from jaxgam.families.standard import Gaussian
from tests.test_execution.test_operator_bulk_variance import _fisher_provider
row = np.arange(47); x = np.linspace(0.1, 1.4, 47)
frame = pd.DataFrame({"y": 1.3 + np.sin(2 * x) + 0.1 * np.cos(row), "x": x,
                      "g": pd.Categorical(np.resize(["a", "b", "c", "d"], 47))})
_, _, base = _fisher_provider(frame, "y~s(x,bs='cr',k=6)+s(g,bs='re')", Gaussian(),
                              np.array([0.7, 1.1]), scale=1.0, probes=48)
selected = frame.iloc[[0, 2, 13, 32, 46]]
maxima = [BulkOperatorVariance(base.system, scale=1.0, memory_budget_bytes=16 << 20,
                               probes=48, seed=seed,
                               control=PCGControl(rtol=1e-11, max_iterations=200))
          .build().calibrate(selected).maximum_standardized_error
          for seed in range(2000, 3000)]
print(np.mean(np.asarray(maxima) > 5), np.quantile(maxima, [0.5, 0.95, 0.99]))
```

```python
# F7: discrete route with a factor random effect.
import numpy as np, pandas as pd
from jaxgam import GAM, FitControl
from jaxgam.control import ContinuousDiscretizationControl
from jaxgam.data.source import DataFrameRowSource
rng = np.random.default_rng(7); n = 400
df = pd.DataFrame({"x0": rng.uniform(0, 1, n),
                   "g": pd.Categorical([f"g{i}" for i in rng.integers(0, 40, n)])})
df["y"] = np.sin(2 * np.pi * df.x0) + rng.normal(0, 0.3, n)
for factors in ((), ("g",)):
    control = FitControl(execution="discrete", batch_rows=128,
                         smoothing_strategy="working_model_reml",
                         discretization=ContinuousDiscretizationControl(factor_names=factors))
    GAM("y~s(x0,bs='cr',k=8)+s(g,bs='re')", control=control).fit(
        DataFrameRowSource(df, response="y"), result="prediction")
```
