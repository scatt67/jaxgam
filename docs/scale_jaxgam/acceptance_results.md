# Scaling acceptance results (plan §8)

This document reports the measurements for the cases fixed in
`acceptance_cases.md` (plan §8 of `jaxgam_scaling_implementation_plan.md`).
Every row of that matrix appears below with its measured values, or with
"rejected" or "not converged" and the recorded status. The tables are
generated from the harness records, the run's status lines and its manifest
by a script kept with the raw records; no number in a table was typed by
hand.

Three limits on what the numbers mean:

- **No speedup claim.** Routes that compute different estimators (for example
  discretized working-model REML and exact REML) are never compared as one
  quantity. Where two routes compute the same estimator on the same data, only
  their statistical agreement is compared.
- **One laptop.** Every timing is from one Apple M4 laptop (10 cores, 16 GB)
  with BLAS held to one thread. Timings describe this run, not the routes in
  general.
- **Not a parity host.** mgcv parity is decided by the test suite on native
  Linux CI, green on this commit, and in the pinned Linux arm64 image: the
  full suite ran there on the parent commit, and the two test modules it
  failed were fixed and rerun green on this one. The two cases here that call
  R (T4 and part of T0) are reported as measured on this host.

## The run

| Item | Value |
|---|---|
| Commit (manifest, start and end) | `a47515593175e10810e402cf1ad6344aa2b4ecfa`; clean tree at both; `same_checkout=true` |
| Commit in the harness records | 83 setup/fit records; commits: `a47515593175e10810e402cf1ad6344aa2b4ecfa` |
| First case started, last case ended (UTC) | 2026-11-19T01:40:43Z to 2026-11-19T11:50:50Z |
| Host | Apple M4, 10 logical CPUs, 17,179,869,184 bytes RAM, macOS-15.6-arm64-arm-64bit-Mach-O |
| Power before each case | Now drawing from 'AC Power' |
| Sleep assertions at manifest start and end | {"PreventSystemSleep": "1", "PreventUserIdleSystemSleep": "1"}; {"PreventSystemSleep": "1", "PreventUserIdleSystemSleep": "1"} |
| Records flagged `host_suspension_suspected` | none |
| Python | 3.13.10 |
| JAX / jaxlib | 0.9.0.1 / 0.9.0.1 (cpu, float64) |
| NumPy / SciPy | 2.3.5 / 1.17.0 |
| BLAS | scipy-openblas 0.3.30 (OpenBLAS 0.3.30  USE64BITINT DYNAMIC_ARCH NO_AFFINITY neoversen1 MAX_THREADS=64) |
| Threads | `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; `XLA_FLAGS` unset |

- **One case ran on a different commit.** H5 ran on `d4ba9bd`, which differs
  from `a475155` only in `scripts/benchmark_operator_preconditioner.py` and
  its new test (see "Deviations from the run plan").
- **Process layout.** Each harness record is one setup child and one fit
  child, each a fresh process with its own empty JAX compilation cache. The
  cold fit is the first fit in its process and includes compilation; the
  backend-compile seconds inside it are in parentheses. The warm fit is one
  repeat in the same process.

### Reading the tables

- **Outcome.** `valid`: the fit converged and the record passed the harness's
  checks. `rejected`: a memory preflight refused the case before fitting; the
  record column carries the message and its byte counts. `not converged`: the
  fit ran and reported `converged=False`; the harness marks the record invalid
  and keeps no timings for it.
- **Fit RSS above baseline.** The fit child's peak resident memory after the
  warm fit, minus its resident memory before the first fit. It includes the
  JAX runtime and compiled code, so it is corroboration only. The structural
  numbers are the bounds in the record column and the known workspace.
- **Known workspace.** The bytes the route itself accounts for before it
  allocates.
- **Fit times and fit scans cover the whole fit call.** Each timed fit is one
  public `GAM(...).fit(...)` call, which includes model setup. The Setup
  column is that same setup measured alone in its own process, so the
  optimization itself is roughly the difference. Where setup dominates
  (L1, L6, M2c) the text says so.
- **Units.** MB is 10^6 bytes; MiB is 2^20 bytes. Budgets are stated in MiB as
  the harness options take them.

## Answers to the plan's questions

1. **Large n, modest p: does streamed memory plateau, and does Gaussian
   optimization stop reading rows?** Yes for every streamed route measured.
   A Gaussian QR fit call made 17 source scans and used a 218.5 MB workspace
   at 1, 3 and 10 million rows. Setup alone makes 16 of those scans, so the
   optimization reads the source once, whatever its iteration count (13, 13
   and 14). The exact streamed route kept a 212.6 MB workspace
   from 100,000 to 1 million rows; it reads the source on every score
   evaluation by design (316 and 604 scans). Fixed-sp streamed PIRLS used 34
   scans and 747.9 MB at both sizes. Resident memory above baseline did not
   grow with n in these pairs by more than 2% (734 to 748 MB for ten times
   the rows in L3b).
2. **Many additive terms: are penalty storage, compilation and smoothing
   derivatives controlled?** At 100 smooths, yes on the discrete route:
   3.3 s of backend compilation (k = 10) and 3.8 s (k = 20), where the defect
   F11 had cost 664 s. At 500 smooths the fit ran inside 6 GiB: a 1,001 s cold fit call, of which about 524 s is setup and 16.5 s backend compilation, with a fit peak bound of 5.85 GB. Two limits remain: the streamed Gaussian
   QR route does not converge at 100 smooths (F1, a documented limit), and the
   exact streamed route did not converge at 40 Gaussian smooths (F13).
3. **Expensive kernels: are distance and kernel intermediates batch bounded?**
   Yes on the dense and operator routes. The operator preparation bound is now
   flat above the 2,000-knot cap: 340.7 MB at 2,048 through 8,192 distinct
   rows (F3, fixed). On the discrete route the construction is bounded by the
   number of compact rows, not by n, but that number grows with the smooth's
   dimension, so 2-D and 3-D thin-plate smooths at 1 million rows need table
   budgets of 1.2 to 1.5 GB (F8).
4. **Shared variables and tensors: do dependence detection and constraint
   preparation avoid n-squared and n-by-p allocations?** Yes. The discrete
   route's setup construction peaked at 4.5 and 5.2 MB at 1 million rows. The
   dense setups at 100,000 rows took 0.27 s and 0.69 s on a host where an
   n-by-n factor would need 80 GB.
5. **High-cardinality effects: is storage proportional to indices and
   nonzeros?** Yes on the operator route: the peak bound was 29 MB at 10,000
   levels and 156 MB at 100,000 levels, and at fixed levels it was the same
   29.0 MB at 20,000 and 200,000 rows. At 1 million levels and 2 million rows the fit ran inside a 2 GiB budget with a peak bound of 1,431 MB, about 1.4 KB per level. The dense reference at
   only 1,000 levels needed 6.7 GB.
6. **NB count tails: is derivative work bounded without losing derivative
   accuracy?** The exact streamed NB fit agrees with the dense fit to 1e-12 in
   the score and 6e-9 in theta at 100,000 rows, with 2.1 s of derivative time
   in a 17.7 s fit. One case does not converge: data drawn near the Poisson
   limit (F14).
7. **Numerical stress: do solvers, derivatives, rank decisions and uncertainty
   remain valid?** The stress cases converged on every route they ran on, and
   dense and streamed REML agree to 2e-11 or better in the score on each
   shared data set. The bulk variance rule had no exceedance in 1,000 seeds
   for either family. The open items are F13, F14 and the rank test with
   column drop, which stays deferred.

## Results by section

### 1. Large n, modest p

| ID | Model | Family | n | p / pen. | Route | Budget | Outcome | Setup s | Cold fit s (compile s) | Warm fit s | Fit RSS above baseline MB | Known workspace MB | Route record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 (smaller) | additive(10, 11) | gaussian | 1,000,000 | 101 / 10 | stream-gaussian-qr | default | valid | 38.2 | 45.2 (2.55) | 42.0 | 1,093 | 219 | fit scans 17; 13 iterations; EDF 100.86 |
| L1 (larger) | additive(10, 11) | gaussian | 3,000,000 | 101 / 10 | stream-gaussian-qr | default | valid | 139.8 | 152.5 (2.36) | 150.7 | 947 | 219 | fit scans 17; 13 iterations; EDF 100.95 |
| L2 (smaller) | additive(2, 10) | gaussian | 100,000 | 19 / 2 | stream-exact | default | valid | 0.47 | 11.3 (1.38) | 9.26 | 476 | 213 | fit scans 316; 34 score evaluations, 28 accepted; derivative time 1.86 s; projected gradient 1.51e-07; EDF 18.93; stop: projected gradient <= pgtol, 2 step window(s) |
| L2 (larger) | additive(2, 10) | gaussian | 1,000,000 | 19 / 2 | stream-exact | default | valid | 6.86 | 176.5 (1.34) | 175.5 | 451 | 213 | fit scans 604; 66 score evaluations, 38 accepted; derivative time 13.9 s; projected gradient 3.49e-10; EDF 18.99; stop: relative reduction of F, 3 step window(s); curvature Newton completion converged |
| L3 (smaller) | additive(10, 11) | poisson | 100,000 | 101 / 10 | stream-qr | default | rejected | 2.35 | – | – | – | – | regular signed QR known workspace requires 747,940,832 bytes, exceeding maximum_bytes=536,870,912 |
| L3 (larger) | additive(10, 11) | poisson | 1,000,000 | 101 / 10 | stream-qr | default | rejected | 38.1 | – | – | – | – | regular signed QR known workspace requires 747,940,832 bytes, exceeding maximum_bytes=536,870,912 |
| L3b (smaller) | additive(10, 11) | poisson | 100,000 | 101 / 10 | stream-qr | 1,024 MiB fit | valid | 2.29 | 8.31 (0.80) | 7.24 | 749 | 748 | fit scans 34; 6 iterations; EDF 100.99 |
| L3b (larger) | additive(10, 11) | poisson | 1,000,000 | 101 / 10 | stream-qr | 1,024 MiB fit | valid | 37.5 | 89.2 (0.77) | 88.3 | 760 | 748 | fit scans 34; 6 iterations; EDF 101.00 |
| L4a | additive(10, 11) | gaussian | 1,000,000 | 101 / 10 | discrete-wm | default | valid | 11.7 | 13.3 (0.65) | 12.3 | 262 | 76 | fit scans 2; 17 proposal evaluations; fit peak bound 77 MB; setup construction peak 11.6 MB; compact rows per group up to 1,000; setup scans 37; EDF 100.86; converged after 18 iterations |
| L4b | additive(10, 11) | gaussian | 3,000,000 | 101 / 10 | discrete-wm | default | valid | 34.8 | 36.8 (0.65) | 36.0 | 219 | 164 | fit scans 2; 17 proposal evaluations; fit peak bound 165 MB; setup construction peak 11.6 MB; compact rows per group up to 1,000; setup scans 37; EDF 100.95; converged after 18 iterations |
| L5 | additive(10, 11) | gaussian | 100,000 | 101 / 10 | dense-reml | default | valid | 0.24 | 16.5 (3.23) | 12.9 | 1,390 | – | 11 outer iteration(s); EDF 99.87 |
| L6 | additive(10, 11) | gaussian | 10,000,000 | 101 / 10 | stream-gaussian-qr | default | valid | 555.7 | 597.5 (2.57) | 593.3 | 642 | 219 | fit scans 17; 14 iterations; EDF 100.99 |
| L7a | additive(100, 11) | gaussian | 100,000 | 1,001 / 100 | stream-gaussian-qr | 1,024 MiB fit | not converged | 31.0 | – | – | – | – | 100 iterations; status: iteration limit |
| L7b | additive(100, 11) | gaussian | 100,000 | 1,001 / 100 | stream-gaussian-qr | default | not converged | 31.3 | – | – | – | – | 100 iterations; status: iteration limit |

Memory pairs (all routes; the K5 and H2 rows belong to sections 3 and 5):

| ID | Model | Family | Route | n (small / large) | Fit scans | Workspace bound MB (source) | Bound equal at both n | Input MB | Fit RSS above baseline MB |
|---|---|---|---|---|---|---|---|---|---|
| L1 | additive(10, 11) | gaussian | stream-gaussian-qr | 1,000,000 / 3,000,000 | 17 / 17 | 218.5 / 218.5 (`known_workspace_bytes`) | yes | 88 / 264 | 1,074 / 924 |
| L2 | additive(2, 10) | gaussian | stream-exact | 100,000 / 1,000,000 | 316 / 604 | 212.6 / 212.6 (`known_workspace_bytes`) | yes | 2 / 24 | 463 / 413 |
| K5tp | tp(10) | gaussian | operator | 1,024 / 2,048 | 24 / 24 | 173.5 / 333.4 (`operator_diagnostics.peak_bound_bytes`) | no | 0 / 0 | 140 / 312 |
| K5gp | gp(10) | gaussian | operator | 1,024 / 2,048 | 22 / 22 | 173.2 / 333.2 (`operator_diagnostics.peak_bound_bytes`) | no | 0 / 0 | 252 / 271 |
| H2 | re, 10,000 levels | gaussian | operator | 20,000 / 200,000 | 37 / 37 | 29.0 / 29.0 (`operator_diagnostics.peak_bound_bytes`) | yes | 1 / 4 | 108 / 108 |
| L3b | additive(10, 11) | poisson | stream-qr | 100,000 / 1,000,000 | 34 / 34 | 747.9 / 747.9 (`known_workspace_bytes`) | yes | 9 / 88 | 734 / 748 |

- **L1, L6.** Scans, workspace and iteration counts are in answer 1. Setup
  dominates the fit call and grows in proportion to n: 38 of 45 s at 1
  million rows, 140 of 152 s at 3 million and 556 of 598 s at 10 million.
  The scan count follows the number of smooths, not the iterations: the
  two-smooth fit T12g2 made 9 scans (8 in setup) in 14 iterations.
- **L2.** The larger fit stopped on L-BFGS-B's relative-reduction test and
  was completed by one curvature Newton step; its final projected gradient is
  3.5e-10.
- **L3.** The default 512 MiB fit budget rejected both sizes: the workspace at
  65,536-row batches and p = 101 is 747,940,832 bytes, the same at both n.
  L3b is the same pair inside 1 GiB.
- **L4.** Two fit scans at both sizes. The fit peak bound grows from 77 MB to
  165 MB between 1 and 3 million rows, about 44 bytes per row. That is
  consistent with the per-row index arrays this route keeps, as
  `bam(discrete=TRUE)` does; it is linear in n with a small constant, not
  n-by-p. Setup construction stayed at 11.6 MB.
- **L5.** The dense reference holds the n-by-p design by design.
- **L7.** Both batch sizes stop at the iteration limit after 100 iterations
  (F1).

### 2. Many additive terms

| ID | Model | Family | n | p / pen. | Route | Budget | Outcome | Setup s | Cold fit s (compile s) | Warm fit s | Fit RSS above baseline MB | Known workspace MB | Route record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | additive(100, 10) | gaussian | 100,000 | 901 / 100 | discrete-wm | 1,024 MiB fit, 512 MiB table | valid | 28.9 | 37.5 (3.29) | 33.5 | 1,021 | 326 | fit scans 2; 34 proposal evaluations; fit peak bound 328 MB; setup construction peak 147.4 MB; compact rows per group up to 1,000; setup scans 307; EDF 218.09; converged after 35 iterations |
| M5 | additive(100, 20) | gaussian | 100,000 | 1,901 / 100 | discrete-wm | 1,024 MiB fit, 512 MiB table | rejected | 28.5 | – | – | – | – | Working-model compiled workspace requires 443,730,034 additional bytes; available workspace is 423,890,156. |
| M5b | additive(100, 20) | gaussian | 100,000 | 1,901 / 100 | discrete-wm | 2,048 MiB fit, 512 MiB table | valid | 28.2 | 56.6 (3.75) | 51.9 | 1,340 | 1,090 | fit scans 2; 44 proposal evaluations; fit peak bound 1,094 MB; setup construction peak 390.6 MB; compact rows per group up to 1,000; setup scans 307; EDF 344.08; converged after 45 iterations |
| M2 | additive(500, 10) | gaussian | 100,000 | 4,501 / 500 | discrete-wm | 2,048 MiB fit, 2,048 MiB table | rejected | 523.5 | – | – | – | – | Discrete fit workspace requires 3,369,443,492 bytes, exceeding FitControl.memory_budget_bytes=2,147,483,648. |
| M2b | additive(500, 10) | gaussian | 100,000 | 4,501 / 500 | discrete-wm | 4,096 MiB fit, 2,048 MiB table | rejected | 516.4 | – | – | – | – | Working-model compiled workspace requires 2,480,725,666 additional bytes; available workspace is 925,523,804. |
| M2c | additive(500, 10) | gaussian | 100,000 | 4,501 / 500 | discrete-wm | 6,144 MiB fit, 2,048 MiB table | valid | 523.8 | 1,000.9 (16.5) | 983.4 | 3,083 | 5,839 | fit scans 2; 74 proposal evaluations; fit peak bound 5,850 MB; setup construction peak 1,696.5 MB; compact rows per group up to 1,000; setup scans 1,507; EDF 685.90; converged after 75 iterations |
| M3 | additive(10, 10) | poisson | 100,000 | 91 / 10 | stream-exact | default | rejected | 2.20 | – | – | – | – | regular signed QR known workspace requires 678,845,952 bytes, exceeding maximum_bytes=536,022,128 |
| M3b | additive(10, 10) | poisson | 100,000 | 91 / 10 | stream-exact | 1,024 MiB fit | valid | 2.20 | 77.3 (1.72) | 75.1 | 886 | 784 | fit scans 330; 27 score evaluations, 20 accepted; derivative time 5.09 s; projected gradient 7.52e-07; EDF 90.16; stop: projected gradient <= pgtol, 2 step window(s) |
| M4 | additive(100, 10) | gaussian | 20,000 | 901 / 100 | dense-reml | default | valid | 0.36 | 202.8 (24.0) | 178.4 | 6,397 | – | 19 outer iteration(s); EDF 173.04 |
| F13 | additive(40, 10) | gaussian | 4,000 | 361 / 40 | stream-exact | default | not converged | 0.52 | – | – | – | – | 100 iterations; stop: L-BFGS-B iteration limit, 1 step window(s) |

- **M1, M5.** The 100-smooth fits converged with 34 and 44 proposal
  evaluations. M5 at the 1 GiB fit budget the matrix names was rejected (the
  compiled workspace needed 443,730,034 more bytes than the 423,890,156
  available); M5b is the same case at 2 GiB.
- **M2.** The 500-smooth formula parses, and setup takes 524 s (28.9 s at 100 smooths) with 1,507 source scans. The 2 GiB fit budget the matrix names was rejected on the fit workspace (3,369,443,492 bytes). 4 GiB (M2b) passed that check and was rejected on the compiled workspace, which needs 2,480,725,666 bytes more than the 925,523,804 then left. Together that is 5,850,169,158 bytes. Inside 6 GiB (M2c) the fit converged after 74 proposal evaluations. The cold fit call took 1,001 s: about 524 s of setup, 16.5 s of backend compilation and the rest optimization, with 3.1 GB resident above baseline.
- **M3.** Rejected at the default 512 MiB (workspace 678,845,952 bytes). At
  1 GiB (M3b) the fit converged on the projected gradient in two step windows
  with 27 score evaluations. The iteration-budget defect F10 no longer
  appears at full size.
- **M4.** The dense reference holds one p-by-p matrix per penalty by design:
  6.4 GB resident at 100 smooths.
- **F13.** Did not converge: L-BFGS-B reached its 100-iteration limit in one
  step window. See the findings.
- **Reading EDF in these rows.** The harness gives smooth i a signal with i
  cycles on [0, 1]. A basis of 10 or 11 functions cannot follow more than the
  first few, so in the 10-smooth and 100-smooth cases the fitted model is
  misspecified by construction: EDF sits near its ceiling and the residual
  scale is far above the noise variance of 0.09 (0.52 at 10 smooths, 7.7 at
  100, 39 at 500). These cases measure storage, compilation and derivative
  work at the stated sizes. They do not test smoothing-parameter selection at an interior
  optimum; the 2-smooth cases and the test suite do that.

### 3. Expensive kernels (tp, gp)

| ID | Model | Family | n | p / pen. | Route | Budget | Outcome | Setup s | Cold fit s (compile s) | Warm fit s | Fit RSS above baseline MB | Known workspace MB | Route record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K1 | tp(30) | gaussian | 100,000 | 30 / 1 | dense-reml | default | valid | 2.96 | 5.87 (1.40) | 4.18 | 1,421 | – | 9 outer iteration(s); EDF 29.98 |
| K2 | gp(30) | gaussian | 100,000 | 30 / 1 | dense-reml | default | valid | 2.00 | 4.71 (1.39) | 3.14 | 728 | – | 7 outer iteration(s); EDF 24.38 |
| K7 | tp3(20) | gaussian | 100,000 | 20 / 1 | dense-reml | default | valid | 2.26 | 4.41 (1.32) | 2.74 | 1,380 | – | 6 outer iteration(s); EDF 19.97 |
| K3 | tp(30) | gaussian | 1,000,000 | – | discrete-wm | 512 MiB table | rejected | – | – | – | – | – | compact basis setup exceeds combined construction budget: requires 798,058,996 bytes, budget is 536,870,912. |
| K3b | tp(30) | gaussian | 1,000,000 | – | discrete-wm | 1,024 MiB table | rejected | – | – | – | – | – | compact smooth predictor workspace exceeds budget: requires 1,465,387,157 bytes, budget is 1,073,741,824. |
| K3c | tp(30) | gaussian | 1,000,000 | 30 / 1 | discrete-wm | 2,048 MiB table | valid | 20.1 | 20.6 (0.34) | 20.3 | 1,271 | 70 | fit scans 2; 15 proposal evaluations; fit peak bound 71 MB; setup construction peak 798.1 MB; compact rows per group up to 10,000; setup scans 10; EDF 30.00; converged after 16 iterations |
| K4 | gp(30) | gaussian | 1,000,000 | 30 / 1 | discrete-wm | 128 MiB table | valid | 2.89 | 3.48 (0.35) | 2.95 | 286 | 61 | fit scans 2; 12 proposal evaluations; fit peak bound 62 MB; setup construction peak 67.2 MB; compact rows per group up to 1,000; setup scans 10; EDF 27.37; converged after 13 iterations |
| K8 | tp3(20) | gaussian | 1,000,000 | – | discrete-wm | 512 MiB table | rejected | – | – | – | – | – | compact basis setup exceeds combined construction budget: requires 1,160,017,239 bytes, budget is 536,870,912. |
| K8b | tp3(20) | gaussian | 1,000,000 | 20 / 1 | discrete-wm | 1,536 MiB table | valid | 27.7 | 28.5 (0.33) | 28.0 | 1,143 | 54 | fit scans 2; 14 proposal evaluations; fit peak bound 56 MB; setup construction peak 1,160.0 MB; compact rows per group up to 15,625; setup scans 10; EDF 20.00; converged after 15 iterations |
| K5tp (smaller) | tp(10) | gaussian | 1,024 | 10 / 1 | operator | default | valid | 0.086 | 0.59 (0.34) | 0.13 | 188 | – | 1 Krylov iterations; 13 source scans; peak bound 174 MB; preparation bound 96 MB; preconditioner 960 bytes |
| K5tp (larger) | tp(10) | gaussian | 2,048 | 10 / 1 | operator | default | valid | 0.28 | 1.31 (0.35) | 0.79 | 334 | – | 1 Krylov iterations; 13 source scans; peak bound 333 MB; preparation bound 341 MB; preconditioner 960 bytes |
| K5gp (smaller) | gp(10) | gaussian | 1,024 | 10 / 1 | operator | default | valid | 0.075 | 0.66 (0.33) | 0.22 | 303 | – | 1 Krylov iterations; 13 source scans; peak bound 173 MB; preparation bound 96 MB; preconditioner 960 bytes |
| K5gp (larger) | gp(10) | gaussian | 2,048 | 10 / 1 | operator | default | valid | 0.17 | 1.09 (0.36) | 0.63 | 280 | – | 1 Krylov iterations; 13 source scans; peak bound 333 MB; preparation bound 341 MB; preconditioner 960 bytes |
| K6 | tp(30) | gaussian | 2,500 | 30 / 1 | operator | 1,024 MiB fit | valid | 0.38 | 1.74 (0.42) | 1.15 | 508 | – | 1 Krylov iterations; 13 source scans; peak bound 342 MB; preparation bound 347 MB; preconditioner 7,680 bytes |

- **K1, K2, K7.** Dense setup (knot kernel and eigen basis) took 2.0 to 3.0 s
  at 100,000 rows.
- **K3, K8.** The discrete route's construction for a multi-dimensional
  thin-plate smooth is sized by its compact rows: 1,000 per group for a 1-D
  smooth, 10,000 for the 2-D smooth in K3c and 15,625 for the 3-D smooth in
  K8b. K8 needed 1,160,017,239 bytes
  against its 512 MiB table and ran inside 1,536 MiB (K8b); its fit peak
  bound is then only 56 MB. K3 took three budgets: 512 MiB was rejected on the construction (798,058,996 bytes), 1,024 MiB on the smooth predictor workspace (1,465,387,157 bytes), and 2,048 MiB ran (K3c).
- **K5, K6 and the F3 reproduction.** The operator preparation bound is 96 MB
  at 1,024 distinct rows and 341 MB at 2,048. The appendix reproduction of F3
  on this commit reports the same bound, 340,727,616 bytes (tp) and
  340,558,336 bytes (gp), at 2,048, 3,072, 4,096, 6,144 and 8,192 distinct
  rows, under both a 512 MiB and a 4 GiB budget. Before commit `dda536d` the
  bound at 8,192 rows was 5,454 MB.

### 4. Shared variables and tensors

| ID | Model | Family | n | p / pen. | Route | Budget | Outcome | Setup s | Cold fit s (compile s) | Warm fit s | Fit RSS above baseline MB | Known workspace MB | Route record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | shared(6) | gaussian | 100,000 | 40 / 3 | dense-reml | default | valid | 0.27 | 5.05 (2.13) | 2.42 | 1,081 | – | 9 outer iteration(s); EDF 34.83 |
| S2 | ti(6) | gaussian | 100,000 | 36 / 4 | dense-reml | default | valid | 0.69 | 5.37 (2.09) | 2.64 | 969 | – | 10 outer iteration(s); EDF 31.10 |
| S3 | shared(6) | gaussian | 1,000,000 | 40 / 3 | discrete-wm | default | valid | 3.87 | 4.93 (0.64) | 3.83 | 193 | 77 | fit scans 2; 14 proposal evaluations; fit peak bound 77 MB; setup construction peak 4.5 MB; compact rows per group up to 1,000; setup scans 13; EDF 35.85; converged after 15 iterations |
| S4 | ti(6) | gaussian | 1,000,000 | 36 / 4 | discrete-wm | default | valid | 3.72 | 4.97 (0.71) | 3.80 | 213 | 59 | fit scans 2; 14 proposal evaluations; fit peak bound 60 MB; setup construction peak 5.2 MB; compact rows per group up to 1,000; setup scans 13; EDF 31.91; converged after 15 iterations |
| S5 | tensor(6) | gaussian | 1,000,000 | 36 / 2 | dense | default | valid | 0.85 | 9.61 (2.03) | 7.34 | 3,447 | – | 1 outer iteration(s); EDF 35.99 |

- **S1, S2.** Dense setup with a shared covariate and with `ti` terms.
- **S3, S4.** Discrete setup read the source 13 times and the fit twice.
- **S5.** The dense fixed-sp reference at 1 million rows holds the n-by-p
  design (288 MB per copy) and reached 3.4 GB resident.

### 5. High-cardinality effects

| ID | Model | Family | n | p / pen. | Route | Budget | Outcome | Setup s | Cold fit s (compile s) | Warm fit s | Fit RSS above baseline MB | Known workspace MB | Route record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1a | re, 10,000 levels | gaussian | 20,000 | 10,006 / 2 | operator | default | valid | 0.35 | 3.40 (0.42) | 2.71 | 111 | – | 17 Krylov iterations; 32 source scans; peak bound 29 MB; preparation bound 8 MB; preconditioner 80,384 bytes |
| H1b | re, 100,000 levels | gaussian | 200,000 | 100,006 / 2 | operator | default | valid | 3.10 | 27.1 (0.45) | 26.3 | 154 | – | 16 Krylov iterations; 31 source scans; peak bound 156 MB; preparation bound 56 MB; preconditioner 800,384 bytes |
| H1c | re, 1,000,000 levels | gaussian | 2,000,000 | – | operator | default | rejected | – | – | – | – | – | Operator preparation needs at most 542,452,660 bytes; budget is 536,870,912. |
| H1cb | re, 1,000,000 levels | gaussian | 2,000,000 | 1,000,006 / 2 | operator | 1,024 MiB fit | rejected | 71.6 | – | – | – | – | Frozen operator requires 803,422,474 bytes; budget is 446,032,116. |
| H1cc | re, 1,000,000 levels | gaussian | 2,000,000 | 1,000,006 / 2 | operator | 2,048 MiB fit | valid | 72.5 | 363.4 (0.47) | 361.5 | 667 | – | 20 Krylov iterations; 35 source scans; peak bound 1,431 MB; preparation bound 542 MB; preconditioner 8,000,384 bytes |
| H2 (smaller) | re, 10,000 levels | gaussian | 20,000 | 10,006 / 2 | operator | default | valid | 0.35 | 3.40 (0.41) | 2.72 | 108 | – | 17 Krylov iterations; 32 source scans; peak bound 29 MB; preparation bound 8 MB; preconditioner 80,384 bytes |
| H2 (larger) | re, 10,000 levels | gaussian | 200,000 | 10,006 / 2 | operator | default | valid | 2.51 | 25.4 (0.41) | 24.9 | 111 | – | 17 Krylov iterations; 32 source scans; peak bound 29 MB; preparation bound 8 MB; preconditioner 80,384 bytes |
| H3 | re-only, 4,096 levels | poisson | 8,192 | 4,097 / 1 | operator-efs | default | not converged | 0.060 | – | – | – | – | status: indeterminate_score |
| H7 | re, 1,000 levels | gaussian | 100,000 | – | discrete-wm | default | rejected | – | – | – | – | – | compact basis setup exceeds combined construction budget: requires 249,620,241 bytes, budget is 67,108,864. |
| H7b | re, 1,000 levels | gaussian | 100,000 | – | discrete-wm | 256 MiB table | rejected | – | – | – | – | – | continuous indexed operator batch workspace exceeds budget: requires 1,582,271,008 bytes, budget is 268,435,456. |
| H7c | re, 1,000 levels | gaussian | 100,000 | 1,006 / 2 | discrete-wm | 2,048 MiB table | rejected | 0.54 | – | – | – | – | Discrete fit workspace requires 1,743,171,846 bytes, exceeding FitControl.memory_budget_bytes=536,870,912. |
| H7d | re, 1,000 levels | gaussian | 100,000 | 1,006 / 2 | discrete-wm | 256 MiB table | rejected | 0.54 | – | – | – | – | Working-model compiled workspace requires 250,740,033 additional bytes; available workspace is 175,746,810. |
| H7e | re, 1,000 levels | gaussian | 100,000 | 1,006 / 2 | discrete-wm | 1,024 MiB fit, 256 MiB table | valid | 0.54 | 3.23 (0.46) | 2.40 | 504 | 612 | fit scans 2; 16 proposal evaluations; fit peak bound 612 MB; setup construction peak 249.6 MB; compact rows per group up to 1,000; setup scans 12; EDF 1001.46; converged after 16 iterations |
| H7f | re, 1,000 levels | gaussian | 100,000 | 1,006 / 2 | discrete-wm | 3,072 MiB fit, 2,048 MiB table | valid | 0.54 | 3.15 (0.46) | 2.27 | 481 | 1,994 | fit scans 2; 16 proposal evaluations; fit peak bound 1,994 MB; setup construction peak 249.6 MB; compact rows per group up to 1,000; setup scans 12; EDF 1001.46; converged after 16 iterations |
| H8 | re, 1,000 levels | gaussian | 100,000 | 1,006 / 2 | dense-reml | default | valid | 1.29 | 336.3 (1.94) | 334.8 | 6,736 | – | 12 outer iteration(s); EDF 1001.46 |

- **H1, H2.** Model `re` is `s(x0, bs='cr', k=6) + s(g, bs='re')`. The
  retained preconditioner is 8 bytes per level plus 384 bytes. H2 holds the
  levels at 10,000 and grows the rows tenfold: same bound, same scan counts,
  same resident memory (108 MB above baseline).
- **H1 at 1 million levels.** Rejected at the default 512 MiB (preparation
  needs 542,452,660 bytes) and again at 1 GiB, where the frozen operator
  needs 803,422,474 bytes and 446,032,116 were left. At 2 GiB (H1cc) the fit converged in 20 Krylov iterations with a peak bound of 1,431 MB and 667 MB resident above baseline. A dense system at this size would need 8 TB.
- **H3.** Operator EFS at the public default controls stopped with
  `indeterminate_score` and reported `converged=False`. This is the documented
  behaviour of the default controls on this shape.
- **H4.** At the calibration's tight controls the same shape converged:
  stop reason `score_window` after 12 outer iterations. Over the audited
  states the largest ratio of actual score error to its bound was 3.6e-5,
  with no state above the formula bound and none above the admitted bound.
  The script records no dense EFS comparison at 4,096 levels
  (`dense_strict` is null), so the matrix's "fit against dense EFS" is not
  measured here; the test suite compares the two at smaller sizes.
- **H5.** Paired PCG solves of one fixed-sp system with the coarse preconditioner off and on, run on `d4ba9bd` (clean tree); the table follows this list. Every solve's residual, checked against the dense matrix, is at most 2.5e-11 of the right-hand side, and the coefficients are within 7.3e-11 of a direct solve. On this model the coarse preconditioner did not reduce the work: it took 16 iterations against 10 at 1,000 levels and 12 against 8 at 10,000.
- **H6.** See the bulk variance table in section 7.
- **H7.** Rejected at the default 64 MiB table (construction needs
  249,620,241 bytes) and again at 256 MiB, where the indexed-operator batch
  workspace needs 1,582,271,008 bytes: about 24 bytes per batch row and
  coefficient at 65,536-row batches and p = 1,006. Setup then passes inside a 2 GiB table (H7c) or with 8,192-row batches (H7d), and the default 512 MiB fit budget rejects the fit. With the fit budget raised the case runs: the fit peak bound is 612 MB at 8,192-row batches (H7e) and 1,994 MB at 65,536-row batches (H7f), and the two scores agree to 3e-16. On this route the batch size, not n, sets the memory of a wide random effect.
- **H8.** The dense reference at 1,000 levels holds the n-by-p design.

H5, paired PCG solves with the coarse preconditioner off and on:

| Levels | n | p | Condition number of H | Coarse preconditioner | PCG iterations | Operator actions | Source scans | Relative residual (PCG) | Relative residual (dense H) | Largest coefficient error against the direct solve | Peak bound bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 4,000 | 1,006 | 6.04e+05 | off | 10 | 12 | 13 | 2.6e-12 | 1.5e-11 | 7.3e-11 | 890,990 |
| 1,000 | 4,000 | 1,006 | 6.04e+05 | on | 16 | 19 | 20 | 4.5e-13 | 1.3e-11 | 4.5e-11 | 890,990 |
| 10,000 | 40,000 | 10,006 | 5.95e+06 | off | 8 | 10 | 11 | 3.6e-12 | 2.5e-11 | 8.2e-12 | 7,992,006 |
| 10,000 | 40,000 | 10,006 | 5.95e+06 | on | 12 | 14 | 15 | 4.9e-12 | 7.8e-12 | 3.0e-11 | 7,992,006 |

### 6. NB count tails

| ID | Model | Family | n | p / pen. | Route | Budget | Outcome | Setup s | Cold fit s (compile s) | Warm fit s | Fit RSS above baseline MB | Known workspace MB | Route record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N1 | additive(2, 10) | nb | 100,000 | 19 / 2 | dense-reml | default | valid | 0.12 | 6.16 (3.26) | 0.95 | 754 | – | 5 outer iteration(s); theta 3.075; EDF 18.33 |
| N2 | additive(2, 10) | nb | 1,000,000 | 19 / 2 | discrete-wm | default | valid | 3.70 | 8.33 (0.98) | 6.90 | 227 | 66 | fit scans 70; 10 proposal evaluations; fit peak bound 66 MB; setup construction peak 2.1 MB; compact rows per group up to 1,000; setup scans 12; theta 2.999; EDF 18.90; converged after 11 performance iterations |
| N3 | additive(2, 10) | nb | 100,000 | 19 / 2 | stream-qr | default | valid | 0.47 | 8.92 (1.85) | 6.29 | 512 | 217 | fit scans 179; 7 iterations; theta 3.075; EDF 19.00; stop: L-BFGS-B relative reduction of F; stationary within one score ULP |
| N4a | additive(2, 10) | nb | 100,000 | 19 / 2 | stream-exact | default | valid | 0.47 | 17.7 (1.80) | 15.1 | 503 | 217 | fit scans 431; 32 score evaluations, 21 accepted; derivative time 2.14 s; projected gradient 8.65e-11; theta 3.075; EDF 18.33; stop: projected gradient <= pgtol, 1 step window(s) |
| N4b | additive(2, 10) | nb | 1,000,000 | 19 / 2 | stream-exact | default | valid | 6.80 | 188.8 (1.75) | 188.5 | 637 | 217 | fit scans 535; 43 score evaluations, 19 accepted; derivative time 11.5 s; projected gradient 1.20e-03; theta 2.998; EDF 18.90; stop: relative reduction of F, 2 step window(s); accepted by mgcv newton()'s relative test |
| N6 | additive(2, 10), 1.0% outliers | nb | 100,000 | 19 / 2 | dense-reml | default | valid | 0.12 | 6.81 (3.44) | 1.37 | 747 | – | 7 outer iteration(s); theta 0.1408; EDF 14.70 |
| N7 | additive(2, 10), 0.1% outliers | nb | 100,000 | 19 / 2 | stream-exact | default | valid | 0.47 | 21.8 (1.83) | 19.3 | 511 | 217 | fit scans 555; 43 score evaluations, 21 accepted; derivative time 2.45 s; projected gradient 2.09e-06; theta 0.2856; EDF 18.88; stop: relative reduction of F, 2 step window(s); accepted by mgcv newton()'s relative test |
| N8 | additive(2, 10), data theta 10000 | nb | 100,000 | 19 / 2 | dense-reml | default | valid | 0.12 | 7.29 (3.39) | 1.90 | 776 | – | 12 outer iteration(s); theta 4668; EDF 18.44 |
| N9 | additive(2, 10), data theta 10000 | nb | 100,000 | 19 / 2 | stream-exact | default | not converged | 0.47 | – | – | – | – | 38 iterations; stop: L-BFGS-B abnormal line-search exit, 5 step window(s) |

- **Theta.** Data are drawn at theta 3 and every fit estimates theta from
  1.0. Dense REML and exact streamed REML give 3.0752 at 100,000 rows; the
  discrete route and the exact streamed route give 2.9987 and 2.9984 at 1
  million rows (different estimators, shown for orientation only).
- **N4 at 1 million rows.** L-BFGS-B stopped on its relative-reduction test
  with a projected gradient of 1.2e-3. The route then applies the test
  mgcv's `newton()` uses (gradient at most 5 x `conv.tol` x score scale, here
  7.5) and reports convergence, with the whole status string kept (see "Stop
  statuses").
- **N6, N7.** With 1% and 0.1% of rows replaced by counts near 2,000, theta
  falls to 0.14 and 0.29: the outliers make the counts far more
  overdispersed.
- **N8, N9.** Data drawn at theta 10,000 are Poisson for practical purposes
  and the likelihood is nearly flat in theta. The dense route converged at
  theta 4,668 with score 142160.9043. The exact streamed route stopped with
  an abnormal L-BFGS-B line-search exit after five step windows and reported
  `converged=False`; its last score, 142160.8203, is lower than the dense
  one by 5.9e-7 relative. See F14.

N5, the NB log-gamma theta-derivative kernel (five repeats; one outlier row
carries the largest count):

| Rows | Largest count | Prefix table bytes | Compiler temp bytes (prefix / legacy) | Cold s (prefix / legacy) | Warm s (prefix / legacy) |
|---|---|---|---|---|---|
| 1,000 | 16 | 136 | 8,256 / 8,288 | 0.052 / 0.037 | 2.34e-05 / 1.44e-05 |
| 1,000 | 4,096 | 32,776 | 67,720 / 8,288 | 0.049 / 0.034 | 3.33e-05 / 5.11e-04 |
| 1,000,000 | 16 | 136 | 8,250,000 / 8,250,080 | 0.051 / 0.040 | 4.66e-04 / 1.45e-03 |
| 1,000,000 | 4,096 | 32,776 | 8,250,000 / 8,250,080 | 0.052 / 0.329 | 4.85e-04 / 2.79e-01 |

The prefix table costs 8 bytes per count value up to the largest count. The
kernel is the same derivative computed two ways, so the two columns are one
quantity; they are recorded as the case asks, with no claim beyond the
numbers.

### 7. Numerical stress

| ID | Model | Family | n | p / pen. | Route | Budget | Outcome | Setup s | Cold fit s (compile s) | Warm fit s | Fit RSS above baseline MB | Known workspace MB | Route record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T8d | additive(10, 10), concurvity | gaussian | 100,000 | 91 / 10 | dense-reml | default | valid | 0.23 | 16.2 (3.00) | 12.8 | 1,373 | – | 13 outer iteration(s); EDF 87.88 |
| T8s | additive(10, 10), concurvity | gaussian | 100,000 | 91 / 10 | stream-exact | default | rejected | 2.19 | – | – | – | – | regular signed QR known workspace requires 678,845,952 bytes, exceeding maximum_bytes=536,021,904 |
| T8sb | additive(10, 10), concurvity | gaussian | 100,000 | 91 / 10 | stream-exact | 1,024 MiB fit | valid | 2.21 | 293.8 (1.60) | 291.8 | 833 | 784 | fit scans 1,314; 144 score evaluations, 117 accepted; derivative time 17.4 s; projected gradient 6.62e-05; EDF 87.88; stop: relative reduction of F, 2 step window(s); accepted by mgcv newton()'s relative test |
| T9s | additive(10, 11), zero-weights | gaussian | 100,000 | 101 / 10 | stream-qr | default | valid | 2.34 | 6.09 (1.02) | 4.76 | 860 | 219 | fit scans 27; 4 iterations; EDF 100.99 |
| T9d | additive(10, 11), zero-weights | gaussian | 100,000 | 101 / 10 | discrete-wm | default | valid | 1.39 | 2.38 (0.65) | 1.35 | 190 | 37 | fit scans 2; 14 proposal evaluations; fit peak bound 37 MB; setup construction peak 11.6 MB; compact rows per group up to 1,000; setup scans 37; EDF 99.84; converged after 15 iterations |
| T10d | additive(2, 10), extreme-weights | gaussian | 100,000 | 19 / 2 | dense-reml | default | valid | 0.12 | 2.75 (1.39) | 0.84 | 482 | – | 14 outer iteration(s); EDF 19.00 |
| T10s | additive(2, 10), extreme-weights | gaussian | 100,000 | 19 / 2 | stream-exact | default | valid | 0.47 | 14.5 (1.35) | 12.6 | 452 | 213 | fit scans 424; 46 score evaluations, 32 accepted; derivative time 2.14 s; projected gradient 2.18e-11; EDF 19.00; stop: relative reduction of F, 3 step window(s); curvature Newton completion converged |
| T11d | additive(2, 10), offset | poisson | 100,000 | 19 / 2 | dense-reml | default | valid | 0.12 | 3.03 (1.40) | 1.18 | 524 | – | 7 outer iteration(s); EDF 18.56 |
| T11s | additive(2, 10), offset | poisson | 100,000 | 19 / 2 | stream-exact | default | valid | 0.48 | 8.85 (1.44) | 6.81 | 468 | 213 | fit scans 216; 18 score evaluations, 13 accepted; derivative time 1.58 s; projected gradient 1.53e-07; EDF 18.56; stop: projected gradient <= pgtol, 2 step window(s) |
| T11o | additive(2, 10), offset | poisson | 100,000 | 19 / 2 | operator | default | rejected | 1.01 | – | – | – | – | Frozen operator requires 257,676,683 bytes; budget is 229,567,660. |
| T11ob | additive(2, 10), offset | poisson | 100,000 | 19 / 2 | operator | 1,024 MiB fit | valid | 1.01 | 8.40 (0.44) | 7.66 | 261 | – | 5 Krylov iterations; 37 source scans; peak bound 575 MB; preparation bound 105 MB; preconditioner 3,192 bytes |
| T12d | additive(10, 11), near-perfect | gaussian | 100,000 | 101 / 10 | dense-reml | default | valid | 0.24 | 16.1 (2.89) | 12.9 | 1,338 | – | 11 outer iteration(s); EDF 100.04 |
| T12g | additive(10, 11), near-perfect | gaussian | 100,000 | 101 / 10 | stream-gaussian-qr | default | valid | 2.33 | 5.91 (2.33) | 2.75 | 801 | 219 | fit scans 17; 11 iterations; EDF 100.04 |
| T12d2 | additive(2, 10), near-perfect | gaussian | 100,000 | 19 / 2 | dense-reml | default | valid | 0.12 | 2.85 (1.42) | 0.89 | 448 | – | 14 outer iteration(s); EDF 19.00 |
| T12g2 | additive(2, 10), near-perfect | gaussian | 100,000 | 19 / 2 | stream-gaussian-qr | default | valid | 0.48 | 2.21 (1.18) | 0.43 | 343 | 46 | fit scans 9; 14 iterations; EDF 19.00 |
| T13 | additive(2, 5), 2.0% outliers | poisson | 400 | 9 / 2 | dense-reml | default | valid | 0.089 | 1.75 (1.15) | 0.012 | 191 | – | 8 outer iteration(s); EDF 9.00 |

- **T0.** The existing zero-weight, extreme-weight, offset, near-perfect and
  zero-residual gates: 96 passed, 5,075 deselected, on this host. R-backed
  parity counts only in the pinned environment, where native Linux CI passed
  the full suite on this commit.
- **T4 (F4).** Dense `te(x0, x1, bs='cr', k=6)` REML against pinned R, n =
  1,000, noise 0.02: R selects log sp (-6.019, 14.554), JaxGAM (-6.019,
  18.447), both with full convergence. The score differs by 1.8e-4 and the
  fitted values by at most 4.5e-6. At noise 0.05 and n = 600 the differences
  are 4.1e-5 and 2.8e-6. The second smoothing parameter is on the flat part
  of the criterion in both fits. Before the `get_stableS` port this case
  stopped with `step failed`, 8.2e-3 from R's score.
- **T5 (F5).** `y ~ x + s(x, bs='gp')` at n = 200 reports
  `full convergence; fallback jitter 1e-12 in PIRLS` and raises a
  `RuntimeWarning` that the penalized Hessian of the final fit was not
  numerically positive definite. The rank deficiency is now disclosed. The
  rank test with column drop that would remove the aliased column is the
  deferred gap.
- **T8.** Concurvity (`x1 = x0 + N(0, 0.01^2)`). The streamed fit was
  rejected at the default budget and ran inside 1 GiB (T8sb): 144 score
  evaluations and 1,314 scans, accepted by `newton()`'s relative test with a
  projected gradient of 6.6e-5.
- **T9.** Ten percent of rows have weight zero. The two rows use different
  smoothing (fixed sp against REML), so they are not compared.
- **T11.** The operator fit was rejected at the default budget (frozen
  operator 257,676,683 bytes; 229,567,660 left) and ran inside 1 GiB
  (T11ob).
- **T12.** With ten smooths the model cannot follow the signal (F16): the fitted scale is 0.434, so that pair does not reach the regime the case names. The two-smooth pair (T12d2, T12g2) does: the fitted scale is 3.4e-6 against a signal variance of 0.16, EDF is at its ceiling of 19, both routes reach full convergence in 14 iterations, and they agree to 2.5e-15 in the score.
- **T13 (F12).** Poisson with 2% count outliers at n = 400 now reaches full
  convergence in 8 outer iterations.

Bulk variance calibration (T6 Gaussian, T6p Poisson, H6b at 10,000 levels):

| ID | Model | Family | Rows | p | Seeds | Current rule: exceedances (95% interval for the rate) | Largest deviate | Superseded rule: exceedances (95% interval) | Mean ratio by calibration row | Largest |z| of a row mean | Ratio variance (reference 2/48 = 0.0417) | Per-row KS p-values | Largest PCG relative residual | Seconds per seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T6 | `y~s(x,bs='cr',k=6)+s(g,bs='re')` | gaussian | 47 | 10 | 1,000 | 0 of 1000 (0.00% to 0.37%) | 3.46 | 9 of 1000 (0.41% to 1.70%) | 1.0018, 0.9991, 1.0095, 1.0139, 1.0084 | 2.15 | 0.0420 | 0.93, 0.95, 0.47, 0.09, 0.08 | 1.0e-10 | 1.9 |
| T6p | `y~s(x,bs='cr',k=6)+s(g,bs='re')` | poisson | 47 | 10 | 1,000 | 0 of 1000 (0.00% to 0.37%) | 3.63 | 12 of 1000 (0.62% to 2.09%) | 1.0023, 0.9991, 1.0125, 1.0133, 1.0064 | 2.06 | 0.0424 | 0.90, 0.89, 0.19, 0.14, 0.34 | 1.0e-10 | 1.9 |
| H6b | `y~s(x,bs='cr',k=6)+s(g,bs='re'), 10,000 levels` | gaussian | 20,000 | 10,006 | 10 | 0 of 10 (0.00% to 30.85%) | 3.04 | 0 of 10 (0.00% to 30.85%) | 1.0580, 0.8937, 0.9492, 0.8866, 0.9472 | 1.76 | 0.0562 | 0.35, 0.07, 0.57, 0.06, 0.55 | 1.0e-10 | 79.0 |

- **The current rule** judges each calibration row by the exact chi-square
  pivot, mapped to a normal deviate, against a limit of 5. It never fired.
- **The superseded rule** (`|estimate - exact| / Monte Carlo SE`) would have
  fired on about 1% of seeds, which is the finding F6 recorded.
- **Calibration of the estimator.** With exact solves the ratio
  `probes x estimate / exact` is chi-square(48)/48: mean 1, variance 0.0417.
  The measured variances in T6 and T6p (0.0420 and 0.0424) match, and no
  single row's KS test rejects at 5%. The row means in T6 and T6p are up to
  2.2 standard errors above 1, in the same two rows of both runs. The two
  runs share their probe seeds, so that is one draw, not two, and 2.2 is
  unremarkable for the largest of five rows.
- **H6b has 10 seeds**, too few to estimate a rate (the interval for zero of
  ten reaches 31%). It shows the rule's cost at p = 10,006: 79 s per seed for
  48 probe solves.

## Budget rejections and their reruns

Every case below was refused by a preflight before any fit. The message is
the route's own; the rerun changes only the budget, except where a batch size
is named.

| ID | Budget in the case | Rejection (verbatim) | Rerun | Rerun outcome |
|---|---|---|---|---|
| K3 | 512 MiB table | compact basis setup exceeds combined construction budget: requires 798,058,996 bytes, budget is 536,870,912. | K3b | K3b (1,024 MiB table): rejected |
| H7 | default | compact basis setup exceeds combined construction budget: requires 249,620,241 bytes, budget is 67,108,864. | H7b | H7b (256 MiB table): rejected |
| L3 | default | regular signed QR known workspace requires 747,940,832 bytes, exceeding maximum_bytes=536,870,912 | L3b | L3b (1,024 MiB fit): valid |
| M5 | 1,024 MiB fit, 512 MiB table | Working-model compiled workspace requires 443,730,034 additional bytes; available workspace is 423,890,156. | M5b | M5b (2,048 MiB fit, 512 MiB table): valid |
| M3 | default | regular signed QR known workspace requires 678,845,952 bytes, exceeding maximum_bytes=536,022,128 | M3b | M3b (1,024 MiB fit): valid |
| K8 | 512 MiB table | compact basis setup exceeds combined construction budget: requires 1,160,017,239 bytes, budget is 536,870,912. | K8b | K8b (1,536 MiB table): valid |
| H1c | default | Operator preparation needs at most 542,452,660 bytes; budget is 536,870,912. | H1cb | H1cb (1,024 MiB fit): rejected |
| T8s | default | regular signed QR known workspace requires 678,845,952 bytes, exceeding maximum_bytes=536,021,904 | T8sb | T8sb (1,024 MiB fit): valid |
| T11o | default | Frozen operator requires 257,676,683 bytes; budget is 229,567,660. | T11ob | T11ob (1,024 MiB fit): valid |
| M2 | 2,048 MiB fit, 2,048 MiB table | Discrete fit workspace requires 3,369,443,492 bytes, exceeding FitControl.memory_budget_bytes=2,147,483,648. | M2b | M2b (4,096 MiB fit, 2,048 MiB table): rejected |
| K3b | 1,024 MiB table | compact smooth predictor workspace exceeds budget: requires 1,465,387,157 bytes, budget is 1,073,741,824. | K3c | K3c (2,048 MiB table): valid |
| H7b | 256 MiB table | continuous indexed operator batch workspace exceeds budget: requires 1,582,271,008 bytes, budget is 268,435,456. | H7c, H7d | H7c (2,048 MiB table): rejected; H7d (256 MiB table, 8,192-row batches): rejected |
| H1cb | 1,024 MiB fit | Frozen operator requires 803,422,474 bytes; budget is 446,032,116. | H1cc | H1cc (2,048 MiB fit): valid |
| H7c | 2,048 MiB table | Discrete fit workspace requires 1,743,171,846 bytes, exceeding FitControl.memory_budget_bytes=536,870,912. | H7f | H7f (3,072 MiB fit, 2,048 MiB table): valid |
| H7d | 256 MiB table | Working-model compiled workspace requires 250,740,033 additional bytes; available workspace is 175,746,810. | H7e | H7e (1,024 MiB fit, 256 MiB table): valid |
| M2b | 4,096 MiB fit, 2,048 MiB table | Working-model compiled workspace requires 2,480,725,666 additional bytes; available workspace is 925,523,804. | M2c | M2c (6,144 MiB fit, 2,048 MiB table): valid |

The discrete and operator routes check their budgets in stages, so a rerun at
the size one message asks for can meet a further requirement (K3, H7, M2 and
H1 at 1 million levels). See F15.

## Agreement between routes that compute the same estimator

Dense Newton REML and the exact streamed route (or, for T12, the frozen
Gaussian QR route) optimize the same criterion on the same generated data.
Differences are relative to the dense value.

| Dense | Streamed | Data | Score, relative difference | EDF, relative difference | Deviance, relative difference | Scale or theta, relative difference |
|---|---|---|---|---|---|---|
| N1 | N4a | NB, theta estimated, additive(2, 10), n = 100,000 | 1.1e-12 | 8.2e-06 | 2.6e-09 | theta 5.9e-09 |
| T8d | T8sb | Gaussian, concurvity, additive(10, 10), n = 100,000 | 1.3e-16 | 8.9e-08 | 8.1e-11 | scale 3.0e-12 |
| T10d | T10s | Gaussian, extreme weights, additive(2, 10), n = 100,000 | 2.5e-13 | 2.3e-11 | 2.8e-15 | scale 1.7e-15 |
| T11d | T11s | Poisson with offset, additive(2, 10), n = 100,000 | 1.7e-11 | 2.0e-06 | 2.1e-10 | – (scale fixed at 1) |
| T12d | T12g | Gaussian, noise 1e-6, additive(10, 11), n = 100,000 | 4.3e-16 | 4.3e-16 | 3.4e-16 | scale 2.6e-16 |
| T12d2 | T12g2 | Gaussian, noise 1e-6, additive(2, 10), n = 100,000 | 2.5e-15 | 3.7e-16 | 2.4e-14 | scale 2.4e-14 |

The largest EDF difference, 8e-6, is the NB case. With the score equal to
1e-12, that is consistent with two optimizers stopping a small distance apart
on a flat criterion. N8 and N9 are not in the table because N9 did not
converge.

## Stop statuses

Verbatim status strings of the streamed and operator-EFS fits that did not
end with `full convergence`.

| ID | n | converged | Status (verbatim) |
|---|---|---|---|
| N3 | 100,000 | True | L-BFGS-B stage: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH; retained exact stationary candidate within one score ULP after the line-search roundoff stop.; pinned log-theta secant Newton completion: not run: L-BFGS-B stage converged, 0 accepted step(s), final proposed step none; last accepted log-theta step +1.644100e-10; projected gradient 1.390069e-11 (gtol 1.000e-09); secant curvature 3771.358134 |
| L2 (smaller) | 100,000 | True | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL; 2 source step window(s). |
| L2 (larger) | 1,000,000 | True | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH; 3 source step window(s).; free-parameter curvature Newton completion: 1 accepted step(s), stopped by converged; last proposed step max-norm 1.240e-06 (accepted); finite-difference curvature eigenvalues [3.9996, 3.99983, 499999] |
| L7a | 100,000 | False | iteration limit |
| L7b | 100,000 | False | iteration limit |
| H3 | 8,192 | False | indeterminate_score |
| N4a | 100,000 | True | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL; 1 source step window(s). |
| N4b | 1,000,000 | True | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH; 2 source step window(s).; projected gradient 1.200e-03 exceeds gtol 1.000e-09.; converged by pinned mgcv newton()'s relative test after an L-BFGS-B function-reduction exit: projected gradient 1.200e-03 <= 5*conv.tol*score.scale 7.525e+00, score change 0.000e+00 <= conv.tol*score.scale 1.505e+00 (conv.tol 1e-06, score.scale 1.50495e+06). |
| N7 | 100,000 | True | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH; 2 source step window(s).; projected gradient 2.092e-06 exceeds gtol 1.000e-09.; converged by pinned mgcv newton()'s relative test after an L-BFGS-B function-reduction exit: projected gradient 2.092e-06 <= 5*conv.tol*score.scale 1.030e+00, score change 2.590e-09 <= conv.tol*score.scale 2.059e-01 (conv.tol 1e-06, score.scale 205922). |
| N9 | 100,000 | False | ABNORMAL: ; 5 source step window(s). |
| T10s | 100,000 | True | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH; 3 source step window(s).; free-parameter curvature Newton completion: 1 accepted step(s), stopped by converged; last proposed step max-norm 1.303e-07 (accepted); finite-difference curvature eigenvalues [3.99936, 4, 49998.5] |
| T11s | 100,000 | True | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL; 2 source step window(s). |
| F13 | 4,000 | False | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT; 1 source step window(s). |
| M3b | 100,000 | True | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL; 2 source step window(s). |
| T8sb | 100,000 | True | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH; 2 source step window(s).; projected gradient 6.625e-05 exceeds gtol 1.000e-06.; converged by pinned mgcv newton()'s relative test after an L-BFGS-B function-reduction exit: projected gradient 6.625e-05 <= 5*conv.tol*score.scale 5.472e-01, score change 0.000e+00 <= conv.tol*score.scale 1.094e-01 (conv.tol 1e-06, score.scale 109436). |

## Findings

Status of the findings listed in `acceptance_cases.md`, and three new ones.

| ID | Finding | Status after this run | Evidence |
|---|---|---|---|
| F1 | Streamed Gaussian QR route above about 50 penalties | **Documented limit, reproduced at full size.** Not fixed in this effort by decision (2026-11-17); the analytic Newton port is deferred to the unify-fit work | L7a, L7b: `iteration limit` after 100 iterations at both batch sizes |
| F2 | Exact streamed NB with estimated theta stopping on relative reduction | **Resolved by classification.** The stop occurs at 1 million rows and is accepted by mgcv `newton()`'s relative test; at 100,000 rows the projected gradient reaches 8.7e-11 | N4a, N4b |
| F3 | Operator preparation bound quadratic in distinct rows | **Fixed** (`dda536d`): flat at 340.7 MB from 2,048 to 8,192 distinct rows | K5, K6, appendix F3 |
| F4 | Dense `te()` REML with log sp 20 apart | **Fixed** (`get_stableS`): full convergence, score within 1.8e-4 of R | T4 |
| F5 | Rank-deficient `x + s(x, bs='gp')` reporting plain convergence | **Disclosed, not removed.** Status and a warning now report the jitter; column drop is deferred | T5 |
| F6 | Bulk variance calibration rule | **Replaced rule confirmed**: 0 of 1,000 exceedances for each family; the superseded rule 9 and 12 of 1,000 | T6, T6p |
| F7 | Discrete route with a factor `re` term: unhelpful message | **Fixed**: the message names `ContinuousDiscretizationControl(factor_names=('g',))` | appendix F7 |
| F8 | Discrete multi-dimensional tp construction bytes | **Reproduced at full size; a budget matter, not a defect.** 2-D tp(30) at 1 million rows needs a 1.5 GB table and 3-D tp3(20) a 1.2 GB one; both then fit with a small fit bound | K3, K8 |
| F9 | Formula parser recursion at 500 terms | **Fixed earlier** (`a892c63`): the 500-smooth case parses | M2 |
| F10 | Exact streamed Poisson with 10 smooths hitting the iteration limit | **Fixed**: converges on the projected gradient at 100,000 rows | M3b |
| F11 | Compile time of discrete working-model REML with many smooths | **Fixed**: 3.3 s and 3.8 s at 100 smooths. At 500 smooths, 16.5 s of a 1,001 s cold fit call | M1, M5b, M2c |
| F12 | Dense Poisson REML with count outliers at n = 400 | **Fixed**: full convergence | T13 |
| F13 | Exact streamed Gaussian REML with 40 smooths | **Open.** Does not converge and says so: L-BFGS-B iteration limit after 100 iterations. The harness data (seed 42) differ from the data the finding was first recorded on (seed 5, relative-reduction stop), so the two stops are not the same event | F13 |
| F14 | **New.** Exact streamed NB REML on data near the Poisson limit | **Open.** Stops with an abnormal line-search exit and reports `converged=False`, at a score 5.9e-7 (relative) below the dense route's converged score. The route classifies function-reduction exits by `newton()`'s relative test but not abnormal line-search exits; whether it should is a decision, not a measurement | N8, N9 |
| F15 | **New.** Budget preflights run in stages | **Open, usability.** A rejection reports the first requirement that fails, not the total: K3 reported 798 MB and then 1,465 MB; H7 250 MB, then 1,582 MB, then the fit budget; M2 3.37 GB and then 2.48 GB more; H1 at 1 million levels 542 MB and then 803 MB. Finding the budget a model needs takes one run per stage | K3, H7, M2, H1c and their reruns |
| F16 | **New.** The harness's additive signal outruns its basis | **Limit of the cases, not of the library.** From about the fifth smooth on the signal cannot be represented, so the many-smooth rows and the ten-smooth T12 pair fit a misspecified model (see section 2) | M1, M4, M5b, L5, T12d |

## Deviations from the run plan

- **Budget reruns.** Ten matrix rows were rejected at the budget the matrix
  names. Each rejection is kept under its ID and each rerun has its own ID
  (suffix b to f) and row.
- **H6 cut from 100 seeds to 10.** At 10,000 levels one seed takes 79 s, so
  100 seeds need about 130 minutes, not the 45 the plan estimated. The
  100-seed run was stopped by hand after 84 minutes (the script writes its
  record only at the end, so nothing was kept) and rerun with 10 seeds as
  H6b.
- **H5 ran on `d4ba9bd`.** On `a475155` the preconditioner benchmark stopped
  on its own check: it compared `H @ solution` with the right-hand side
  element by element at STRICT, and one of 1,006 elements was 4.9e-10 off
  while the residual norm was 1.5e-11 of the right-hand side's. PCG stops on
  the residual norm, which does not bound single elements, so `d4ba9bd`
  checks the norm at STRICT and records it. Only the script and its test
  changed.
- **F13 uses the harness seed.** The harness has no seed option, so F13 ran
  on seed 42 rather than the seed 5 of the first report.
- **T12 added a two-smooth pair** (T12d2, T12g2) because the ten-smooth case
  cannot reach a near-zero scale (F16).
- **Four short reruns were repeated.** H7e, H7f, T12d2 and T12g2 were first
  started twice at once by a queueing mistake and overlapped each other.
  Those records were set aside and the four cases rerun alone; the tables
  show the reruns.
- **M2 ran.** The plan held it back until F11 was fixed; it was. It needed
  three budgets (2, 4 and 6 GiB).
- **Durations.** The plan estimated about 8.5 hours for groups 1 to 7.
  Measured wall time by group is below; the lettered groups are the reruns.

| Group | Cases | Plan estimate min | Measured wall min |
|---|---|---|---|
| 1 | K1, K2, K3, K4, K7, S1, S2, S3, S4, S5, N1, N2, N3, N5, N6, N8, H7, T4, T5, T9s, T9d, T12d, T12g, T13 | 50 | 5.5 |
| 2 | L1, L2, L3, L4a, L4b, L5 | 41 | 20.5 |
| 3 | M1, M5, M4, M3, L7a, L7b | 135 | 15.0 |
| 4 | K5tp, K5gp, K6, K8, H1a, H1b, H1c, H2, H3, H4, H5, H8 | 86 | 16.5 |
| 5 | N4a, N4b, N7, N9, T8d, T8s, T10d, T10s, T11d, T11s, T11o | 90 | 10.1 |
| 6 | H6, T6, T6p | 70 | 146.2 |
| 7 | L6 | 30 | 29.2 |
| 8 | M2, F13, AF3, AF7, T0 | – | 24.3 |
| f | K3b, H7b, K8b, L3b, M3b, T8sb, M5b, H1cb, T11ob, H6b | – | 37.2 |
| g | H5b, K3c, H7c, H7d, H1cc, M2b | – | 40.3 |
| h | H7e, H7f, T12d2, T12g2 | – | 0.5 |
| i | M2c | – | 42.6 |
