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
  `re` term, which the harness passes for model `re` (G8). Operator EFS is
  public only for canonical Poisson with one indexed random effect, 257 to
  4096 levels. Operator tp and gp smooths are capped near 2,500 distinct rows
  at the default budget by F3.
- **Other scripts.** `make benchmark-bulk-variance` runs
  `scripts/benchmark_bulk_variance.py` (H6, T6) and `make benchmark-manifest`
  writes the run manifest (G15); both use the checkout they live in.

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

m=500 parses since the parser fix (F9). On the discrete working-model route
M2 runs only once the F11 fix lands: 100 smooths already spent 664 s (k=10)
and 757 s (k=20) in XLA compilation at n=5000, so 500 smooths would compile
for hours; those times are its recorded reason until then.

| ID | model | family/link | n | p / pen. | route, smoothing | record | command |
|---|---|---|---|---|---|---|---|
| M1 | additive(100, 10) | Gaussian/identity | 100k | 901 / 100 | discrete-wm, 512 MiB table, 1 GiB fit budget | backend compile s and count (F11); setup construction peak; proposal evaluations. If setup is still rejected, record the required bytes and rerun with a larger table budget | `make benchmark-scaling ARGS="--execution discrete-wm --smooth-count 100 --k 10 --batch-rows 65536 --discrete-table-mib 512 --memory-budget-mib 1024 --n 100000 P"` |
| M2 | additive(500, 10) | Gaussian/identity | 100k | 4501 / 500 | discrete-wm, 2 GiB table and budget | as M1. **Runs only once the F11 fix lands** (compile time) | `make benchmark-scaling ARGS="--execution discrete-wm --smooth-count 500 --k 10 --batch-rows 65536 --discrete-table-mib 2048 --memory-budget-mib 2048 --n 100000 P"` |
| M3 | additive(10, 10) | Poisson/log | 100k | 91 / 10 | stream-exact, REML | derivative s; outer iterations; cumulative PIRLS passes; stop status (F10) | `make benchmark-scaling ARGS="--execution stream-exact --family poisson --smooth-count 10 --k 10 --batch-rows 65536 --n 100000 P"` |
| M4 | additive(100, 10) | Gaussian/identity | 20k | 901 / 100 | dense-reml | Newton iterations; compile; RSS (dense m-penalty reference) | `make benchmark-scaling ARGS="--execution dense-reml --smooth-count 100 --k 10 --n 20000 P"` |
| M5 | additive(100, 20) | Gaussian/identity | 100k | 1901 / 100 | discrete-wm, 512 MiB table, 1 GiB budget | as M1 at k=20 | `make benchmark-scaling ARGS="--execution discrete-wm --smooth-count 100 --k 20 --batch-rows 65536 --discrete-table-mib 512 --memory-budget-mib 1024 --n 100000 P"` |

### 3. Expensive kernels (tp, gp)

Plan question: are all distance and kernel intermediates batch bounded?
The default knot cap (mgcv `max.knots` 2000) applies above 2000 unique rows.
Input dimension d = 1 (gp), 2 (tp) and 3 (tp3). The training-time row
batching is also a structural test (P4).

| ID | model | family/link | n | p / pen. | route, smoothing | record | command |
|---|---|---|---|---|---|---|---|
| K1 | tp(30) | Gaussian/identity | 100k | 30 / 1 | dense-reml | setup s and RSS (knot kernel, eigen basis); prediction s | `make benchmark-scaling ARGS="--execution dense-reml --model tp --k 30 --n 100000 P"` |
| K2 | gp(30) | Gaussian/identity | 100k | 30 / 1 | dense-reml | as K1 | `make benchmark-scaling ARGS="--execution dense-reml --model gp --k 30 --n 100000 P"` |
| K3 | tp(30) | Gaussian/identity | 1m | 30 / 1 | discrete-wm, 512 MiB table | setup construction peak against compact rows (F8) | `make benchmark-scaling ARGS="--execution discrete-wm --model tp --k 30 --batch-rows 65536 --discrete-table-mib 512 --n 1000000 P"` |
| K4 | gp(30) | Gaussian/identity | 1m | 30 / 1 | discrete-wm, 128 MiB table | as K3 | `make benchmark-scaling ARGS="--execution discrete-wm --model gp --k 30 --batch-rows 65536 --discrete-table-mib 128 --n 1000000 P"` |
| K5 | tp(10) and gp(10) | Gaussian/identity | 1,024 and 2,048 distinct rows (pair) | 10 / 1 | operator, fixed sp | preparation peak bound against distinct rows (F3: about 82 bytes x rows squared); Krylov iterations | `make benchmark-scaling ARGS="--execution operator --model tp --k 10 --batch-rows 1024 --memory-pair-n 1024 2048 P"`; repeat with `--model gp`. At 100k or 1m rows the same command is rejected by F3 |
| K6 | tp(30) | Gaussian/identity | 2,500 | 30 / 1 | operator, fixed sp, 1 GiB budget | preparation bound at k=30 (F3 caps the default 512 MiB budget near 2,500 distinct rows) | `make benchmark-scaling ARGS="--execution operator --model tp --k 30 --batch-rows 1024 --memory-budget-mib 1024 --n 2500 P"` |
| K7 | tp3(20) | Gaussian/identity | 100k | 20 / 1 | dense-reml | as K1 in d=3 | `make benchmark-scaling ARGS="--execution dense-reml --model tp3 --k 20 --n 100000 P"` |
| K8 | tp3(20) | Gaussian/identity | 1m | 20 / 1 | discrete-wm, 512 MiB table | as K3 in d=3 | `make benchmark-scaling ARGS="--execution discrete-wm --model tp3 --k 20 --batch-rows 65536 --discrete-table-mib 512 --n 1000000 P"` |

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
one-hot dense arrays? The operator routes carry the large cases; dense and
discrete `re` fits are references at 1,000 levels only (a dense 1m-level
system is 8 TB). Model `re` is `s(x0, bs='cr', k=6) + s(g, bs='re')`; model
`re-only` is `s(g, bs='re')`, the calibrated operator-EFS shape.

| ID | model | family/link | n | p / pen. | route, smoothing | record | command |
|---|---|---|---|---|---|---|---|
| H1 | re, 2 rows per level | Gaussian/identity | 20k, 200k, 2m | 10k, 100k, 1m / 2 | operator, fixed sp, PCG | peak bound against budget; scans by kind; Krylov iterations; preconditioner bytes; RSS | `make benchmark-scaling ARGS="--execution operator --model re --k 6 --levels 10000 --n 20000 --batch-rows 4096 P"`; repeat with `--levels 100000 --n 200000` and `--levels 1000000 --n 2000000` |
| H2 | re, 10k levels | Gaussian/identity | 20k and 200k pair | 10k / 2 | operator, fixed sp | workspace-bound plateau and RSS above baseline at fixed p | `make benchmark-scaling ARGS="--execution operator --model re --k 6 --levels 10000 --batch-rows 4096 --memory-pair-n 20000 200000 P"` |
| H3 | re-only | Poisson/log | 8192 (2 per level) | 4097 / 1 | operator EFS (public default controls) | score bound and method, stop reason, iterations, scans; an `indeterminate_score` stop is recorded as `valid=false` | `make benchmark-scaling ARGS="--execution operator-efs --model re-only --family poisson --levels 4096 --n 8192 --batch-rows 4096 P"` |
| H4 | `y~0+s(g,bs='re')` | Poisson/log | 8192 (2 per level) | 4096 / 1 | operator EFS at the calibration's tight controls | score bound, stop reason, iterations, scans, fit against dense EFS | `make benchmark-operator-efs-calibration ARGS="--regime indexed_poisson --levels 4096 --complete" OUTPUT=$RUN/H4` |
| H5 | `y~s(x,bs='cr',k=6)+s(g,bs='re')` | Gaussian/identity | 4 rows per level | 1k and 10k levels | operator, fixed sp, paired PCG | actions and verified residuals with and without the coarse preconditioner | `uv run python scripts/benchmark_operator_preconditioner.py --levels 1000 10000 --output-dir $RUN/H5` |
| H6 | re (bulk-variance script), 10k levels | Gaussian/identity | 20k | 10k / 2 | `bulk_link_variance`, 48 probes, 100 seeds | per seed: probe solves, scans, PCG residual; per row: ratio, current equivalent deviate, superseded standardized error; both rules' exceedances at the public limit (F6) | `make benchmark-bulk-variance ARGS="--model re --levels 10000 --rows-per-level 2 --seeds 100 --probes 48 --output $RUN/H6.json"` |
| H7 | re, 1000 levels | Gaussian/identity | 100k | 1005 / 2 | discrete-wm (`factor_names=("g",)`) | setup construction peak; compact rows; fit scans | `make benchmark-scaling ARGS="--execution discrete-wm --model re --k 6 --levels 1000 --n 100000 --batch-rows 65536 P"` |
| H8 | re, 1000 levels | Gaussian/identity | 100k | 1005 / 2 | dense-reml (reference: holds n x p) | RSS; Newton iterations | `make benchmark-scaling ARGS="--execution dense-reml --model re --k 6 --levels 1000 --n 100000 P"` |

### 6. NB count tails

Plan question: is derivative work bounded without losing first and second
derivative accuracy? `--outlier-fraction` replaces a fraction of rows by
counts 1000 + Poisson(1000); `--nb-theta` draws the data near the Poisson
limit while every fit still estimates theta from 1.0.

| ID | model | family/link | n | p / pen. | route, smoothing | record | command |
|---|---|---|---|---|---|---|---|
| N1 | additive(2, 10) | NB/log, theta estimated | 100k | 19 / 2 | dense-reml | theta; iterations | `make benchmark-scaling ARGS="--execution dense-reml --family nb --smooth-count 2 --k 10 --n 100000 P"` |
| N2 | additive(2, 10) | NB/log, theta estimated | 1m | 19 / 2 | discrete-wm | theta; fit scans per performance iteration | `make benchmark-scaling ARGS="--execution discrete-wm --family nb --smooth-count 2 --k 10 --batch-rows 65536 --n 1000000 P"` |
| N3 | additive(2, 10) | NB/log, theta estimated | 100k | 19 / 2 | stream-qr, fixed sp | fit scans; workspace | `make benchmark-scaling ARGS="--execution stream-qr --family nb --smooth-count 2 --k 10 --batch-rows 65536 --n 100000 P"` |
| N4 | additive(2, 10) | NB/log, theta estimated | 100k, then 1m | 19 / 2 | stream-exact, REML | stop status and projected gradient (F2); derivative s | `make benchmark-scaling ARGS="--execution stream-exact --family nb --smooth-count 2 --k 10 --batch-rows 65536 --n 100000 P"`, then `--n 1000000` |
| N5 | NB lgamma theta derivative kernel | NB | 1k and 1m rows; max count 16 and 4096; one outlier | – | prefix kernel and legacy recurrence | cold and warm s; compiler temp bytes; prefix table bytes | `env -u JAX_COMPILATION_CACHE_DIR uv run python scripts/benchmark_nb_lgamma_derivatives.py --rows 1000 1000000 --max-counts 16 4096 --repeats 5 --outlier` |
| N6 | additive(2, 10), 1% outliers | NB/log, theta estimated | 100k | 19 / 2 | dense-reml | theta; iterations; derivative accuracy is the dense gates' | `make benchmark-scaling ARGS="--execution dense-reml --family nb --smooth-count 2 --k 10 --outlier-fraction 0.01 --n 100000 P"` |
| N7 | additive(2, 10), 0.1% outliers | NB/log, theta estimated | 100k | 19 / 2 | stream-exact, REML | derivative s; stop status; theta | `make benchmark-scaling ARGS="--execution stream-exact --family nb --smooth-count 2 --k 10 --outlier-fraction 0.001 --batch-rows 65536 --n 100000 P"` |
| N8 | additive(2, 10), data theta 1e4 | NB/log, theta estimated | 100k | 19 / 2 | dense-reml | theta estimate near the Poisson limit | `make benchmark-scaling ARGS="--execution dense-reml --family nb --nb-theta 10000 --smooth-count 2 --k 10 --n 100000 P"` |
| N9 | additive(2, 10), data theta 1e4 | NB/log, theta estimated | 100k | 19 / 2 | stream-exact, REML | stop status and projected gradient; theta | `make benchmark-scaling ARGS="--execution stream-exact --family nb --nb-theta 10000 --smooth-count 2 --k 10 --batch-rows 65536 --n 100000 P"` |

### 7. Numerical stress

Plan question: do solvers, derivatives, rank decisions and uncertainty remain
valid? `--stress` (G11) generates concurvity, zero and extreme weights,
offsets and near-perfect Gaussian fits; weights and offsets on the discrete
route meet a known parity gap being fixed in PR11, which does not stop the
case. The open findings are rows here; their classification is in the
findings table.

| ID | case | route | record | command |
|---|---|---|---|---|
| T0 | existing zero/extreme-weight, offset, near-perfect and zero-residual gates | all | pass/fail; pinned R parity only counts in the pinned environment | `make test-local PYTEST_ADDOPTS="tests -k 'zero_weight or offset or near_perfect or nearperfect or exact_fit or zero_residual or nearly_exact' -q"` (acceptance: `make test`) |
| T1 | F1: Gaussian QR p=1001, 100 smooths | stream-gaussian-qr | preflight bytes; stop status | L7 |
| T2 | F2: exact streamed NB with estimated theta | stream-exact | stop status; projected gradient | N4, N9 |
| T3 | F3: operator setup accounting against distinct rows | operator | preparation bound | K5, K6; Appendix A |
| T4 | F4: dense `te` REML with log-sp spread >= 20 | dense-reml vs pinned R | score and fitted differences; stop status | Appendix A |
| T5 | F5: `y ~ x + s(x, bs='gp')` converged flag | dense-reml | converged, inner PIRLS status | Appendix A |
| T6 | F6: the replacement calibration rule, and how the superseded one would have done | operator bulk variance | per seed and row: ratio, `equivalent_deviates`, `superseded_standardized_error`; current-rule and superseded-rule exceedances with Clopper-Pearson intervals; KS test of the ratio | `make benchmark-bulk-variance ARGS="--model test --seeds 1000 --probes 48 --output $RUN/T6.json"`; Appendix A |
| T7 | F10: exact streamed Poisson, 10 smooths | stream-exact | stop status | M3 |
| T8 | concurvity, additive(10, 10) | dense-reml; stream-exact | EDF, smoothing parameters, stop status | `make benchmark-scaling ARGS="--execution dense-reml --stress concurvity --smooth-count 10 --k 10 --n 100000 P"`; repeat with `--execution stream-exact --batch-rows 65536` |
| T9 | zero weights, additive(10, 11) | stream-qr; discrete-wm | scans, workspace, EDF | `make benchmark-scaling ARGS="--execution stream-qr --stress zero-weights --smooth-count 10 --k 11 --batch-rows 65536 --n 100000 P"`; repeat with `--execution discrete-wm` |
| T10 | extreme weights 1e-4 to 1e4, additive(2, 10) | dense-reml; stream-exact | stop status, EDF, score | `make benchmark-scaling ARGS="--execution dense-reml --stress extreme-weights --smooth-count 2 --k 10 --n 100000 P"`; repeat with `--execution stream-exact --batch-rows 65536` |
| T11 | offset, additive(2, 10) | Poisson: dense-reml, stream-exact, operator | stop status, EDF | `make benchmark-scaling ARGS="--execution dense-reml --family poisson --stress offset --smooth-count 2 --k 10 --n 100000 P"`; repeat with `--execution stream-exact --batch-rows 65536` and `--execution operator --batch-rows 65536` |
| T12 | near-perfect Gaussian (noise 1e-6), additive(10, 11) | dense-reml; stream-gaussian-qr | scale, score, stop status | `make benchmark-scaling ARGS="--execution dense-reml --stress near-perfect --smooth-count 10 --k 11 --n 100000 P"`; repeat with `--execution stream-gaussian-qr --batch-rows 65536` |
| T13 | F12: Poisson with 2% count outliers at n = 400 | dense-reml | stop status (`step failed` at n = 400) | `make benchmark-scaling ARGS="--execution dense-reml --family poisson --outlier-fraction 0.02 --smooth-count 2 --k 5 --n 400 P"` |

## Harness gaps

Filled in the first round (with tests; `scripts/benchmark_scaling.py` is at
96% statement and branch coverage after both rounds, no new line missed):

- **Compilation** (`375a64b`): a `jax.monitoring` listener records the count
  and seconds of tracing, lowering, backend compilation and persistent-cache
  events for every cold and warm fit (`cold_fit_compilation`,
  `cold_fit_backend_compile_s`, `warm_fit_*`). Before, compilation was folded
  into `cold_compile_and_fit_s`.
- **Models** (`375a64b`): `--model tp | gp | shared | ti` for the dense and
  discrete routes.
- **G1 host suspension** (`11394a2`): each fit is also timed by the wall clock;
  `host_suspension_suspected` flags a record whose wall duration exceeds its
  `perf_counter` duration by more than max(1 s, 5%). On the preparation host,
  two dry runs took 962 s and 915 s of wall time against 5.4 s and 7.2 s of
  `perf_counter` time because the laptop slept between maintenance wakes; JAX's
  compile events use `time.time()` and counted the sleep as compilation.
- **Budgets** (`11394a2`, `258d958`): `--discrete-table-mib` (discrete
  `DiscretizationBudget` table and batch limits) and `--memory-budget-mib`
  (`FitControl.memory_budget_bytes` on streamed and discrete fits and setup).
  The defaults rejected M1, M5, K3, K4, M2 and F1 before any fit.

Gap status. The second round filled G3-G11, G15, P4 and P5, one commit each; G2, G12, G13 and G14 are deliberately not filled:

| Gap | What is missing | Proposal |
|---|---|---|
| G2 | **not filled (decision).** The harness records no compiler memory estimate (`compiler_memory_bytes` is `None`). The evidence remains the kernel tests that compile at bounded shapes and read `memory_analysis()`, in eight files: `tests/test_nb_lgamma_derivatives.py`, `tests/test_execution/test_working_model.py` and `tests/test_fitting/` `test_efs_factor.py`, `test_iterative.py`, `test_nb_stream_reml.py`, `test_nb_theta_stream.py`, `test_operator_family.py`, `test_preconditioners.py` | – |
| G3 | **filled.** Each child runs with `TMPDIR`, `TMP`, `TEMP` and `SQLITE_TMPDIR` pointing at its own task-owned directory, and a thread polls that directory every 50 ms during setup or fitting; every record (valid or not) carries `temporary_disk` with the peak and final bytes and the sample count. Polling makes the peak a lower bound for very short-lived files. Discrete selector memmaps are covered when they use the default temporary location; an explicit `selector_memmap_directory` is not set by the harness | – |
| G4 | **filled.** `--execution operator` (fixed sp, PCG; every model, Gaussian or Poisson) and `--execution operator-efs` (calibrated Poisson EFS; model `re-only` = `y ~ s(g, bs='re')`) run in the isolated harness: a setup child that runs `prepare_operator_model` with the fit's own preparation budget (one eighth of it for EFS) and records scans, bytes, preparation peak bound and coefficient layout without expanding penalties to p x p; a fit child with a task-owned cache, compile events, `operator_diagnostics` (scans by kind, Krylov actions and iterations, peak bound, executable temporaries, inner status, gradient norm) and Krylov iterations as the inner count. Fixed-sp results have no score, EDF or Gaussian scale by design, so those fields are recorded as not provided; `predict(se_fit=True)` is not offered on these routes. NB is rejected because the operator fits NB only at fixed theta | – |
| G5 | **filled.** `scripts/benchmark_bulk_variance.py` (`make benchmark-bulk-variance`) fits one operator model (the bulk-variance test model, or `s(x,bs='cr',k=6)+s(g,bs='re')` with L levels) and calls the public `bulk_link_variance` once per seed with its raise limit out of reach. Per seed and calibration row it records exact variance, estimate, Monte Carlo SE and the ratio estimate/exact; the product's current statistic, `equivalent_deviates` (the chi-square pivot probes x estimate/exact as a standard normal deviate) and `maximum_absolute_deviate`, with `current_rule_would_raise`; and the superseded statistic recomputed in the script, `superseded_standardized_error` (|estimate - exact| / se), `superseded_maximum_standardized_error` and `superseded_rule_would_raise`. Also the provider diagnostics (probe and normal solves, Krylov iterations, every scan count, maximum PCG relative residual, peak bound). The summary gives each rule's exceedance rate at the public limit with a Clopper-Pearson interval and Kolmogorov-Smirnov tests of probes x ratio against chi-square(probes), per row over seeds. One bulk pass over the training rows is timed. Task-owned compilation cache | – |
| G6 | **filled.** `--memory-pair-n` accepts every source-reading route (streamed, discrete, operator, operator EFS) and refuses dense routes before any child runs. Every case control except `n_rows` must match (family, model, levels, stress, budgets, discretization, batch). Besides `known_workspace_plateau` the comparison reports `workspace_bound_plateau` over the route's own bound: `known_workspace_bytes`, or `operator_diagnostics.peak_bound_bytes` where fixed-sp operator fits leave the former empty, with its source named | – |
| G7 | **filled.** An unconverged, non-finite or rejected fit, a rejected setup and a crashed child each write a `valid=false` record: outcome (`unconverged`, `nonfinite`, `fit_raised`, `setup_raised`, `child_failed`), status, stop reason and its source, projected gradient (exact streamed REML), the operator inner status and final gradient norm, smoothing parameters, theta, score, iterations, the error type and message, and no timing fields. Comparisons over an invalid record are marked unavailable. The record is written first; then the command exits 3 and prints each invalid child on stderr | – |
| G8 | **filled.** `--model re --levels L`: `y ~ s(x0, k=k, bs='cr') + s(g, bs='re')` with L declared levels (unobserved levels stay declared) and a normal group effect; the factor reaches every source as a categorical Series, the discrete routes receive `factor_names=("g",)`, and the input hash covers codes and level labels. Dense, discrete and (G4) operator routes | – |
| G9 | **filled.** `--model tp3`: `y ~ s(x0, x1, x2, k=k, bs='tp')` on three uniform covariates (dense, discrete and operator routes), so kernel cases cover d = 1 (gp), 2 (tp) and 3 (tp3) | – |
| G10 | **filled.** `--nb-theta` sets the theta NB data are drawn at (default 3; fits still estimate theta from 1.0), so theta near the Poisson limit (for example 1e4) is a case; `--outlier-fraction f` (Poisson or NB) replaces a fraction f of rows, on their own random stream, by counts 1000 + Poisson(1000). Generator name and route contract record both | – |
| G11 | **filled.** `--stress concurvity` (x1 = x0 + N(0, 0.01^2); additive, two or more smooths), `zero-weights` (10% of rows at weight 0), `extreme-weights` (weights 10^U(-4,4); Gaussian noise scaled by 1/sqrt(w)), `offset` (log exposure, U(0.5, 2)) and `near-perfect` (Gaussian noise s.d. 1e-6). Stress draws use a separate random stream, so unstressed inputs are unchanged; weights and offsets reach dense fits as `fit` arguments and every other route through the row source. On the discrete route weights and offsets meet a known parity gap being fixed in PR11 (the discretization's random stream ignores them); the cases run regardless | – |
| G12 | **unavailable on CPU.** The CPU backend exposes no allocator statistics; records say so in `device_peak_memory_status` | – |
| G13 | **not filled (decision).** Derivative time is recorded only on stream-exact; dense Newton, Gaussian QR and discrete working-model derivative stages need product diagnostics that belong to the dense series | – |
| G14 | **not filled (decision).** Dense routes expose no inner PIRLS count (see F5); public inner-iteration diagnostics belong to the dense series | – |
| G15 | **filled.** `scripts/benchmark_manifest.py` (`make benchmark-manifest`) writes one JSON manifest per run directory: `--phase start` before the first case records OS, CPU model and feature flags, logical CPUs, RAM, Python and every installed package version, dtype and JAX x64/backend/devices, NumPy and SciPy BLAS/LAPACK builds, the source-LAPACK provider (scipy-openblas32 version and OpenBLAS config), the thread and cache environment, the checkout commit and whether its tree is clean, load averages and (macOS) sleep assertions; `--phase end` adds the end checkout state, end load and `same_checkout` | – |
| P4 | **filled for three of the four shapes** (`tests/test_scaling_allocation_guards.py`, 14 cases). n x n Q: the full-Q request is forbidden on dense, discrete working-model, exact streamed REML, frozen Gaussian QR and operator fits at n = 3000. n x p X: the design builder never sees the training rows on exact streamed REML, Gaussian QR and operator fits (n = 5000); the discrete route builds only on its compact frame (about 1000 rows at n = 5000 and 20,000). n x r x d: training-time kernel distances stay at a patched 64-row limit on dense tp and gp, discrete tp, and operator tp and gp (n = 2500), the only larger call being gpE on the 2000-knot table (r x r x d, capped by max.knots, as in mgcv). **m x p x p is not guarded**: host `tracemalloc` peaks at m = 40, p = 361, n = 4000 were 16.9 MB (streamed QR), 15.7 MB (Gaussian QR) and 33.5 MB (exact REML) against 41.7 MB for a stack and 11.6 MB for n x p, so a host-peak bound cannot separate a stack from ordinary working memory, and device buffers are invisible to it. It needs a jaxpr or compiled-memory inspection of the exact-REML adjoint (deferred with G2) | – |
| P5 | **filled.** `PreparedPreconditioner` reports the coarse correction's retained device bytes and construction bound; `operator_diagnostics` (and so every operator record) carries `preconditioner_retained_bytes` (largest over the fit's frozen systems), `coarse_preconditioner_retained_bytes` and `coarse_preconditioner_construction_bound_bytes`, zero without `use_coarse`. They are reported apart from `peak_bound_bytes`, which already includes them | – |

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
| RSS minus baseline | – (dense holds n x p) | H pair | H pair | H pair | H pair | H pair | H pair | – |
| Device peak | – (G12) | – | – | – | – | – | – | – |
| Compiler memory | – (G2) | – | – | – | – | – | – | – |
| Temporary disk | H (task TMPDIR peak + cache) | H | H | H | H | H | H | – |
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
| No n x n Q | discrete, operator, stream-exact, stream-gaussian-qr | `test_scaling_allocation_guards.py::test_routes_never_request_an_n_by_n_q` | `scipy.linalg.qr` (`mode='full'`) and `numpy.linalg.qr` (`mode='complete'`) patched to fail on tall input; public fits at n = 3000 (shared-covariate model on the discrete route) | – |
| No n x p X in streaming | stream / stream-qr | `test_stream_fixed_public_family.py::test_public_fixed_sp_stream_never_builds_training_matrix`; `test_validation_matrix.py::test_public_exact_large_row_fit_stays_within_stream_budget` | `ModelSetup.build` patched to raise; budgeted run | – |
| No n x p X in streaming | stream-exact | `test_nb_stream_reml.py::test_nb_batch_compiled_workspace_is_bounded_by_batch_and_coefficient_size`; `test_nb_theta_stream.py::test_conditional_n_growth_keeps_numeric_buffers_histories_and_ledger_bounded`; `test_scaling_allocation_guards.py::test_source_routes_never_build_the_training_design[stream-exact]` | compiled `memory_analysis`, growth in n, forbidden design builder | – |
| No n x p X in streaming | stream-gaussian-qr | `test_gaussian_stream.py::test_stream_provider_jaxprs_and_retained_state_have_no_row_dimension`; `test_gaussian_optimizer.py::test_provider_score_and_derivative_jaxprs_have_no_observation_arrays`; `test_scaling_allocation_guards.py::test_source_routes_never_build_the_training_design[stream-gaussian-qr]` | jaxpr shapes contain no n; forbidden design builder | – |
| No n x p X | discrete | `test_discrete_ops.py::test_factor_tensor_jaxpr_has_no_expanded_batch_level_tensor` | jaxpr inspection; harness `discrete_setup` ledger; `test_scaling_allocation_guards.py::test_source_routes_never_build_the_training_design[discrete-wm]` (design built only on the compact frame, about 1000 rows at n = 5000 and 20,000) | – |
| No n x p X | operator | `test_operators.py::test_high_cardinality_jaxpr_has_no_expanded_design_or_square_system`; `test_scaling_allocation_guards.py::test_source_routes_never_build_the_training_design[operator]` | jaxpr inspection; forbidden design builder | – |
| No m x p x p penalty stack | stream, stream-exact, gaussian-qr | `test_efs_stream_start.py::test_startup_bound_covers_shared_balancing_dense_tuple_for_many_local_penalties`; `test_rank_coordinates.py::test_global_penalty_scale_bounds_scratch_for_wide_penalty`; penalties are stored per block (`PenaltyStructure` never pads globally) | local-block storage, bounded scratch | **open**: `materialize` builds one combined p x p (not a stack), so forbidding it is the wrong guard; host `tracemalloc` peaks cannot separate a stack at test sizes (P4 row); needs jaxpr or compiled-memory inspection of the exact-REML adjoint |
| No m x p x p penalty stack | dense | none: dense FittingData holds per-penalty matrices by design | – | out of scope (dense reference route) |
| No m x p x p penalty stack | operator | `PenaltyStructure.materialize` patched to raise in `test_operator.py`, `test_operator_fit.py`, `test_operator_logdet.py`, `test_operator_penalty_determinant.py`, `test_operator_efs_trace.py`, `test_operator_variance.py`, `test_operator_bulk_variance.py` | forbidden call | – |
| No n x r x d distance differences | dense, discrete (tp, gp prediction and design rows) | `test_tprs.py::TestCoveragePaths::test_prediction_distance_work_is_row_batched`; `test_gaussian_process.py::TestSetupInvariants::test_prediction_distance_work_is_row_batched`; `test_scaling_allocation_guards.py::test_training_kernel_distances_are_row_batched` (dense tp and gp, discrete tp at n = 2500) | patched kernel records batch sizes (`DISTANCE_BATCH_ROWS`) | – (the r x r knot kernel is capped by max.knots; F8 is the discrete construction bound, a separate finding) |
| No n x r x d distance differences | operator | `test_scaling_allocation_guards.py::test_training_kernel_distances_are_row_batched` (operator tp and gp at n = 2500) | patched kernel records batch sizes | – |
| No p x p G/H | operator | `test_operator.py::test_symbolic_product_storage_and_runtime_without_global_square`; `test_operator_fit.py::test_crossed_numeric_large_levels_fit_without_global_coefficient_square`, `test_mixed_numeric_large_levels_fit_without_global_coefficient_square`; `test_operator_logdet.py::test_large_symbolic_logdet_stays_below_global_square` | symbolic sizes far above RAM run inside a small budget | – |
| No full covariance | operator | `test_operator_variance.py::test_selected_large_symbolic_variance_has_no_coefficient_square`; `test_operator_bulk_variance.py::test_bulk_symbolic_large_p_preflights_and_never_materializes_covariance`; `test_prediction_result.py::test_point_only_result_retains_no_rows_or_dense_covariance` | symbolic large p; forbidden materialization | – |
| Coarse preconditioner accounted separately | operator | `test_preconditioners.py::test_large_diagonal_budget_is_checked_before_transfer_or_factor`, `test_coarse_share_of_storage_is_reported_separately`; `test_operator_fit.py::test_fit_diagnostics_report_the_coarse_preconditioner_separately` | budget preflight; exact byte count of the coarse share; fit diagnostics with and without `use_coarse` | – (P5 filled) |

Across all routes the harness records no compiler memory estimate
(`compiler_memory_bytes` is always `None`; gap G2, not filled by decision).
Kernel tests in eight files read
`jax.jit(...).lower(...).compile().memory_analysis()` (listed under G2); the
acceptance step cites those, not the harness.

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
| M2 | 2,000 at 450 smooths | first round: m=500 did not parse (F9, since fixed); m=450 needed 1.37 GB of setup at n=2,000 (over 1 GiB). Now blocked by F11 |
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

In that first round the high-cardinality rows were run with
`benchmark_operator.py` (300 levels, valid), the EFS calibration script (300
levels, valid) and the preconditioner script (valid); they are now H1-H3 on the
harness and H4-H5 on the scripts.

**Second round (after the harness gaps), every new or changed row.** All at
n <= 2,500, under one run manifest (start and end written, `same_checkout`
true on a clean tree); no record was flagged for host suspension.

| ID | Dry-run size | Result |
|---|---|---|
| K5 | pair 1,024 / 2,048, tp and gp | valid; operator peak bound 173 then 333 MB (no plateau: F3). At 2,048 / 4,096 the larger child is rejected by F3 and recorded `valid=false` (exit 3) |
| K6 | 2,500, 1 GiB | valid; peak bound 342 MB |
| K7, K8 | 2,000 | valid (tp3, p = 20) |
| H1 | 300 levels, 600 rows | valid; p = 306; 13 Krylov iterations; preconditioner 2,784 bytes |
| H2 | 300 levels, pair 600 / 1,200 | valid; `workspace_bound_plateau` true (1.44 MB) |
| H3 | 300 levels, 600 rows | `valid=false`, `indeterminate_score` (public default operator controls), written and exit 3 |
| H6 | 50 levels, 3 seeds, 8 probes | valid after `7c45fe8` (the first attempt imported another editable jaxgam); max PCG relative residual 3.8e-16 |
| H7, H8 | 100 levels, 2,000 rows | valid (discrete with `factor_names`, and dense) |
| N6-N9 | 2,000 | valid (outliers and theta 1e4, dense and exact streamed) |
| T6 | test model, 3 seeds, 8 probes | valid after `7c45fe8` |
| T8 | 2,000, 10 smooths | dense valid; exact streamed `valid=false`, `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT; 2 source step window(s)` |
| T9-T12 | 2,000 | valid; T9 on streamed QR recorded 16,912 bytes of temporary disk |
| T13 | 400 | `valid=false`, `step failed` (F12) |
| M2 | not run | blocked by F11 |

## Run plan for the real measurements

**Order.**
0. `make benchmark-manifest ARGS="--phase start --output $RUN/manifest.json"`.
1. Validity sweep, short cases: K1-K4, K7, S1-S5, N1-N3, N5, N6, N8, H7, T4,
   T5, T9, T12, T13.
2. Large n with the machine to itself: L1-L5.
3. Many terms with the machine to itself: M1, M5, M4, M3, L7.
4. Kernels and high cardinality: K5, K6, K8, H1 (10k, 100k, then 1m levels),
   H2, H3, H4, H5, H8.
5. Exact streamed REML under stress: N4 (100k, then 1m), N7, N9, T8, T10, T11.
6. Bulk variance: H6, T6.
7. L6 (10m rows) last, with the machine to itself.
8. `make benchmark-manifest ARGS="--phase end --output $RUN/manifest.json"`.
9. T0 and every R-parity gate in the pinned environment (`make test`, root).

M2 is not scheduled until the F11 fix lands (compile time). A command that ends
with exit status 3 has still written its record (`valid=false`, G7); the run
continues with the next case.

**Expected durations on a quiet machine.** These are estimates, not
measurements. They are extrapolated from the PR11 runtime study (loaded host)
and PR9's n=1m Gaussian QR runs, and from the cost structure the dry runs
exposed (for example 11 to 13 minutes of compilation for 100 discrete
smooths, and about 0.7 s per bulk-variance seed on the 47-row test model).
Each harness case is one setup child plus one cold and one warm fit.

| Group | Cases | Estimate |
|---|---|---|
| 1 | K1-K4, K7, S1-S5, N1-N3, N5, N6, N8, H7, T4, T5, T9, T12, T13 | 50 min |
| 2 | L1 14, L2 12, L3 6, L4 7, L5 2 | 41 min |
| 3 | M1 25, M5 30, M4 30, M3 20, L7 30 | 135 min |
| 4 | K5 5, K6 3, K8 3, H1 40, H2 5, H3 5, H4 10, H5 5, H8 10 | 86 min |
| 5 | N4 25, N7 10, N9 10, T8 18, T10 12, T11 15 | 90 min |
| 6 | H6 45, T6 25 (Gaussian and Poisson, 1,000 seeds each) | 70 min |
| 7 | L6 | 30 min |
| Total | | about 8.5 hours, excluding M2 and the pinned suite |

**Machine to itself.** Every memory pair (L1-L3, K5, H2), L4, L6, L7, M1,
M3-M5, H1 at 1m levels, H6, and N4 at 1m: their RSS and wall times are
recorded as results. The short cases in group 1 may share the machine only if
their timings are not reported.

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
- The other scripts record less: `benchmark_operator_efs_calibration.py` and
  `benchmark_operator_preconditioner.py` record commit and versions but use
  the caller's cache; run them with `JAX_COMPILATION_CACHE_DIR` pointing at a
  new empty directory. `benchmark_bulk_variance.py` uses a task-owned cache
  and records the package it loaded.

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
| F1 | Gaussian QR, p = 1001, 100 smooths | `make benchmark-scaling ARGS="--execution stream-gaussian-qr --smooth-count 100 --k 11 --batch-rows 16384 --n 20000 P"`; then `--batch-rows 4096`; then `--batch-rows 16384 --memory-budget-mib 1024` | **Documented limit (user decision, 2026-11-17): the streamed Gaussian QR route does not converge above about 50 penalties; not fixed in this effort.** Reproduces at n = 20,000. B=16384: "Known streamed Gaussian QR workspace requires 558529632 bytes, exceeding FitControl.memory_budget_bytes=536870912". B=4096: `converged=False`, status `iteration limit`. With 1 GiB at B=16384 the preflight passes and the stop is the same `iteration limit`, so the non-convergence does not depend on the batch size. The status is L-BFGS-B's iteration limit, not the relative-reduction stop that the Gaussian QR route showed elsewhere; the harness keeps no projected gradient for an invalid fit (G7) |
| F2 | Exact streamed REML, NB with estimated theta, `additive(2, 10)` | N4 at `--n 20000` | **Does not reproduce at n = 20,000** (or 2,000): converged, `CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL; 1 source step window(s)`, projected gradient 4.7e-11, theta 2.947. **Needs the full size** (100k, 1m). The relative-reduction stop was not seen for NB at this size; see F13 for the same stop on Gaussian REML |
| F3 | Operator setup accounting against distinct covariate rows | K5, K6; Appendix A, F3 | **Reproduces on the knot-series head (`299b6d6`), unchanged by it.** The preparation peak bound (`operator_diagnostics.preparation_peak_bound_bytes`, budget 16 GiB so the full requirement is reported) is quadratic in distinct rows, not bounded by the 2000-knot cap: 2-D tp k=10 96.1, 358.7, 1,386.1 and 5,454.4 MB at 1,024, 2,048, 4,096 and 8,192 distinct rows (81-92 bytes per row squared); 1-D gp 96.3, 358.1, 1,385.1 and 5,452.4 MB. Cause, by reading `formula/operator_radial.py`: `_setup_bytes(q, ...)` charges about 80 q^2 bytes for q = every distinct row discovered, before the rows are sampled to `max_knots`. The fit's own peak bound plateaus (333 MB from 2,048 rows on) and the 8,192-row fits completed in under 7 s, so the actual work is knot-bounded and only the preparation accounting grows. Rejection message at the default 512 MiB: "Operator preparation needs at most 537089966 bytes; budget is 536870912." (tp; 537,089,880 at 8,192 rows; gp 536,881,464 at both). The check runs once per newly discovered row and raises at the first row count whose requirement crosses the budget, so the message reports that crossing point, about the budget, not the requirement (1.39 GB at 4,096 rows, 5.45 GB at 8,192). The first round's "does not reproduce" was wrong: its 1 MiB budget tripped an earlier, n-independent check |
| F4 | Dense `te()` REML with log smoothing parameters >= 20 apart | Appendix A, F4 | **Reproduces at n = 1,000** (`te(x0,x1,bs='cr',k=6)`, y = sin(4 pi x0) + N(0, 0.02^2)). R: log sp (-6.02, 14.55), spread 20.6, full convergence. JaxGAM: (-6.02, 23.78), stop `step failed`; score differs by 8.2e-3 and fitted values by up to 1.0e-3. At spread 15.6 (noise 0.05, n = 600) both converge, with score difference 2.2e-5 and fitted 6.2e-5 |
| F5 | `y ~ x + s(x, bs='gp')` reports converged while its inner loop hits the limit | Appendix A, F5 | **Does not reproduce through public results at n = 200**: `converged=True`, `full convergence`, 5 outer iterations; a fixed-sp refit at the selected sp converges in 2 PIRLS iterations. The inner PIRLS state inside Newton trials is not exposed (G14), so the case needs internal instrumentation (being addressed elsewhere) |
| F6 | `bulk_link_variance` calibration rule | T6, H6 (`scripts/benchmark_bulk_variance.py`) | **The rule has been replaced** (PR13): calibration is now judged by the exact chi-square pivot, mapped to an equivalent normal deviate, against the same 5-SE limit. The finding concerned the superseded rule (`|estimate - exact| / Monte Carlo SE`): on the test model (Gaussian `s(x,bs='cr',k=6)+s(g,bs='re')`, 47 rows, 5 calibration rows, 48 probes) it exceeded 5 on 2 of 240 fresh seeds (0.83%; Clopper-Pearson 95% interval 0.10%-2.98%; maxima 7.60 and 4.48 in the two batches). The measurement now checks the replacement (current-rule exceedance rate at the public limit) and records how the superseded rule would have done on the same seeds |
| F7 | Discrete route with a factor `re` term | Appendix A, F7 | **Reproduces; usage, not numerics.** Without `factor_names` it fails with "Metric column 'g' must be real numeric data"; with `ContinuousDiscretizationControl(factor_names=("g",))` it fits and converges. The message does not name the missing declaration |
| F8 | Discrete 2-D tp construction bytes against compact rows | `--execution discrete-wm --model tp --k 30` at n = 500-2,000 with a 1 MiB table budget | **Reproduces**: required 16.9, 64.1 and 250.0 MiB at 500, 1,000 and 2,000 rows (x4 per doubling, up to the 2000-knot cap). The default 64 MiB budget rejects 2-D tp k=30 above about 1,000 compact rows. 1-D gp stays near 64 MiB because the grid caps its unique values |
| F9 | Formula parser recursion | `parse_formula` with 500 additive `s()` terms | **Fixed in `a892c63`.** The +/- chain walk and its error rendering are iterative; 500 and 5,000 terms parse, and the old and new parsers agree on 141 formulas. CPython's own `ast.parse` still fails somewhere between 5,000 and 20,000 terms |
| F10 | Exact streamed Poisson REML with 10 smooths | M3 at n = 2,000 | **Reproduces at n = 2,000**: `converged=False`, `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT; 4 source step window(s)`. Whether it persists at 100k is unknown  **Being fixed (streamed iteration budget) by another implementer.** |
| F11 | Compilation time of discrete working-model REML with many smooths | M1, M5 at n = 5,000 | **Reproduces at n = 5,000**: 664 s (100 smooths, k=10) and 757 s (k=20) of XLA backend compilation in the cold fit; warm fits 32 s and 60 s; no host suspension. **Being fixed by another implementer.** M2 (500 smooths) runs only once that fix lands |
| F12 | Dense REML Poisson with large count outliers at small n | T13 | **Reproduces at n = 400**: additive(2, 5) with 2% of counts replaced by 1000 + Poisson(1000) stops `converged=False`, `step failed`; the same data at n = 2,000 converge, as does NB with the same outliers (theta 0.27 and 0.13) |
| F13 | Exact streamed Gaussian REML with 40 smooths | `GAM` with `FitControl(execution="stream", linear_solver="qr", batch_rows=256)` on additive(40, 10), n = 4,000, seed 5 | **Reproduces, deterministically (three repeats)**: `converged=False`, `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH; 2 source step window(s).; projected gradient 1.623e-05 exceeds gtol 1.000e-06`, score 4428.4033376774205 each time. The stop is L-BFGS-B's relative-reduction test, the mechanism found on the Gaussian QR route |

**Unexplained events (for the heap-corruption investigation on this host).**
Both happened on 2026-11-17 in the same scratch script, run three times in a
row on the worktree at `3ea2d1d` (clean; the P5 commit before the rebase onto the knot series, now `4748465`), Apple M4, macOS
15.6, Python 3.13.10, JAX 0.9.0.1, under `caffeinate -s -i` with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, `PYTHONPATH` set to
the worktree and the shared persistent JAX cache `$OUT/jax-cache-s8`.

- What was running: one process that imported only `tracemalloc`,
  `warnings`, `numpy` and `jaxgam` (from `jaxgam` it used `GAM`, `FitControl`
  and `ArrayRowSource`). It built additive(40, 10) Gaussian data (n = 4,000,
  seed 5) and, for three routes in turn (streamed pivoted-QR at fixed sp,
  frozen Gaussian QR REML, exact streamed REML; `batch_rows=256`), ran one
  untraced fit and then one fit between `tracemalloc.start()` and
  `tracemalloc.stop()`.
- R was not loaded: nothing in the process imported `rpy2` or
  `tests.r_bridge`, and `jaxgam` itself imports neither.
- **E1, first run:** the process aborted with `libc++abi: terminating due to
  uncaught exception of type std::__1::system_error: mutex lock failed:
  Invalid argument`, printed twice and interleaved (two threads aborting at
  once). Standard output was block-buffered and lost, so the stage (which
  route, traced or untraced fit, or interpreter exit) is unknown; whether
  `tracemalloc` was active at that moment is therefore unknown, though it was
  started in that process at least once if the first route had begun its
  traced fit.
- **E2, second run (`python -u`):** a `ValueError: array must not contain
  infs or NaNs` from `numpy.asarray_chkfinite` inside `scipy.linalg.qr`
  (`scipy/linalg/_decomp_qr.py`, line 142, through
  `scipy/_lib/_util.py` line 1181). Only the last eight output lines were
  kept, so the calling `jaxgam` frame and route are unknown, and so is
  whether the fit was the traced one.
- The third run of the same script completed all six fits; three further
  untraced repeats of the exact streamed fit were bitwise-consistent (F13);
  the P4 guards, which wrap `scipy.linalg.qr`, never saw a non-finite input.

**F6 measurement.** `scripts/benchmark_bulk_variance.py` (T6, H6) records,
per seed and calibration row, the ratio estimate/exact, the product's current
`equivalent_deviates` and the superseded standardized error recomputed by the
script. One run gives the current rule's exceedance rate at the public limit,
the superseded rule's rate on the same seeds (each with a Clopper-Pearson
interval) and the Kolmogorov-Smirnov check of probes x ratio against
chi-square(probes).

## Appendix A: reproductions without a harness mode

These snippets use only the public API and runtime-generated data; run them
from the checkout with `PYTHONPATH=$PWD uv run python`.

```python
# F3: operator preparation bound against distinct rows, at two budgets.
import numpy as np, pandas as pd
from jaxgam import GAM, FitControl
from jaxgam.data.source import DataFrameRowSource
for formula in ("y~s(x0,x1,bs='tp',k=10)", "y~s(x0,bs='gp',k=10)"):
    for budget_mib in (512, 4096):
        for n in (1024, 2048, 3072, 4096, 6144, 8192):
            rng = np.random.default_rng(3)
            df = pd.DataFrame({"x0": rng.uniform(0, 1, n), "x1": rng.uniform(0, 1, n)})
            df["y"] = np.sin(2 * np.pi * df.x0) + rng.normal(0, 0.3, n)
            control = FitControl(execution="operator", linear_solver="pcg",
                                 batch_rows=1024, memory_budget_bytes=budget_mib << 20)
            try:
                fit = GAM(formula, sp=[1.0], control=control).fit(
                    DataFrameRowSource(df, response="y"), result="prediction")
                print(formula, budget_mib, n,
                      fit.operator_diagnostics.preparation_peak_bound_bytes)
            except MemoryError as error:
                print(formula, budget_mib, n, "rejected:", error)
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

F6 now runs as `make benchmark-bulk-variance` (rows T6 and H6).

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
