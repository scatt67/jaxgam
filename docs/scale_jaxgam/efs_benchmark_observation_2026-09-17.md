# Dense EFS bounded benchmark observation, 2026-09-17

This is one sequential measurement on frozen synthetic saved data, taken after
all cooperating agents held Docker work. It is not a general speedup claim or
large-p/operator benchmark. All cold and warm fits are finite and converged.
The selected STRICT EFS/R gate fails and the process exits 2; the original
failure is preserved. Subsequent main-agent numerical review found the exact
representative input and default controls admissible under the user's four-pass
MODERATE policy. Applying that decision to the persistent runtime selector
remains pending explicit approval after automatic-review rejection. The
retrospective numerical decision is separate from the original run history.

The [saved artifact](efs_benchmark_artifacts/2026-09-17/measurement.json)
records the exact command, terminal status, file hashes, source, and quiet
window. [Input CSV](efs_benchmark_artifacts/2026-09-17/input.csv),
[configuration](efs_benchmark_artifacts/2026-09-17/config.json),
[complete report](efs_benchmark_artifacts/2026-09-17/report.json), and
[host provenance](efs_benchmark_artifacts/2026-09-17/host-provenance.json)
are retained unchanged. The absolute data path in the configuration records
the original container worker location. See the
[validation record](efs_benchmark_validation_2026-09-17.md) for harness gates,
control mapping, and the separately reviewed smoke case.

## Input and runtime

Poisson/log REML uses ten cubic smooths, each k=6: n=1200, p=51, m=10, rank X=51,
seed=20260910. The CSV SHA256 is
`22291a4ce4d90220758ae9488d6a880643e75d49d47c809e9d5cf9fe7a1135d4`.
EFS and R use identical saved-input starts, scale 1, PIRLS tolerance 1e-7 and
limit 200, EFS score tolerance 0.1, outer limit 200, and log smoothing cap 15.
Newton keeps its public initial smoothing policy and unchanged defaults:
outer limit 200, convergence tolerance 1e-6, inner tolerance 1e-8 and limit 100,
maximum step 5, and log smoothing cap 40. The exact formula, vectors, composed
FitControl, and mapped R controls are in the configuration/report.

The measured source is `80a25dca1e97714dadc80e4766d9fabf6fb53487`, script SHA256
`a4b2b794bf17ea9d7cb93f601786ecabf67a580207b64b5a3a55e93b530e43da`, image
`sha256:f4eae281f047815852f2819db0760054a22e93112c7a5dca4031beeee834d6ba`.
Host: Apple M4, Mac16,12, 10 logical CPUs, 16 GiB. The aarch64 Colima VM uses
macOS Virtualization.Framework and exposes four CPUs, 16,734,150,656 bytes of
RAM, and Linux 6.8.0. Each method runs in its own subprocess; the Python methods
have separate empty temporary JAX caches. JAX uses the CPU backend with x64
enabled, Python 3.13.10, JAX 0.9.0.1, NumPy 2.3.5, pandas 3.0.0, and JaxGAM
1.0.0a1. R 4.5.2 uses mgcv 1.9-3, OpenBLAS 0.3.26, and LAPACK 3.12.0. R control and
OMP/OpenBLAS/MKL/VECLIB/NUMEXPR thread environment are all configured to one.
JAX's internal thread count is not independently measured. Host background
workloads and thermal/frequency state were not measured.

## Complete-fit observations

| Method | First fit (s) | Warm fits (s) | Warm median (s) | Outer iterations | Sampled process-tree peak RSS (bytes) |
| --- | ---: | --- | ---: | ---: | ---: |
| Public Newton | 2.587033 | 0.065131, 0.058529, 0.058886 | 0.058886 | 8 | 728195072 |
| Public dense EFS | 1.218478 | 0.111795, 0.129952, 0.110344 | 0.111795 | 27 | 538382336 |
| Pinned R EFS | 0.190000 | 0.174000, 0.241000, 0.174000 | 0.174000 | 27 | 344285184 |

EFS's warm complete fit is slower than Newton on this case. Newton and EFS
use different stopping policies, caps, and starts, and their fitted quantities
also differ (maximum coefficient residual 6.15e-4 and score residual 6.31e-4);
this is a runtime comparison, not optimizer-equivalence evidence. The Python
first-fit times include setup, compilation, fitting, synchronization, and
Phase 3 materialization; import/process startup is excluded. Compilation alone
is unavailable. R uses elapsed `system.time`, with no JIT compile measurement.
Three warm observations give a bounded sample, not a distribution estimate.
This single m=10 observation does not establish scaling with penalty count.

RSS is sampled every 10 ms over the whole worker process tree and includes
runtime/import overhead; shared pages may be counted more than once. It is
not an exact allocator peak or comparable device allocation metric. Device
memory is unavailable on this JAX CPU backend; R exposes no accelerator metric.

EFS reports 118 inner iterations, 27 outer iterations, zero theta iterations,
27 accepted-score entries, multiplier 2, 20 cap events, no numerator clamps or
ratio replacements, and no invalid-fit or stabilization indicators. Cumulative
inner iterations are unavailable for Newton and R in the current complete-fit
result APIs. Accepted-score count excludes rejected trials and is not a score
evaluation count. Coefficient factorization counts, RHS counts, compilation-only
time, penalty-derivative time, score-stage time, and complete trial-score counts
are unavailable with reasons in the report. Dense RowSource scans do not apply
to the in-memory route. No unobserved cost reduction is inferred.

## EFS versus pinned R residuals

| Field | Maximum absolute residual | STRICT |
| --- | ---: | --- |
| Coefficients | 1.0071430e-9 | Fail |
| Means | 7.3624289e-9 | Fail |
| Smoothing parameters | 1.1494629e-6 | Fail |
| Smooth EDF | 7.9588949e-9 | Fail |
| Deviance | 1.1029465e-7 | Pass |
| REML score | 4.7998583e-10 | Pass |
| Scale | 0 | Pass |
| Outer iterations | 27 / 27 | Pass |

Both implementations' fit-validity checks pass for every first/warm fit. The
MODERATE diagnostic passes all fields. Representative case-specific oracle,
controller, control-tightening, and conditioning review is separate from these
frozen measurements; its result does not retroactively change the recorded
STRICT selection or terminal exit.

## Case-specific numerical diagnostic procedure

The [numerical review](efs_benchmark_artifacts/2026-09-17/numerical-review.json)
preserves exact controls, named field residuals, fit validity, and selected
states; [conditioning](efs_benchmark_artifacts/2026-09-17/conditioning.json)
and [diagnostic provenance](efs_benchmark_artifacts/2026-09-17/numerical-review-provenance.json)
are separate from the quiet-window timing artifact. Thread settings match the
original one-thread workers. Four distinct attempts to explain/correct the
STRICT difference failed, after verifying roundtrip CSV and initializer
provenance on this case:

1. Call subprocess `RBridge.fit_efs` on the saved roundtrip data, same formula,
   Poisson, scale 1, and exact `efs_initial_sp`. Its output matches the benchmark
   R worker at STRICT: coefficient residual 6.94e-17, other fields exact.
   Python-versus-R fitted residuals remain those in the table above.
2. Build `ModelSetup` and `FittingData` from the saved input, call
   `dense_efs_known_scale` with `efs_initial_log_lambda`, and map coefficients
   using `_transform_coefficients_cpu`. Direct/public coefficients, means,
   smoothing, score, deviance, and scale match exactly; both converge in 27
   outer iterations. No public-adapter correction explains the residual.
3. Copy the saved configuration, set `score_tolerance=0.001`, retain PIRLS
   tolerance 1e-7, then run `_python_worker(..., 'efs')` and `_r_worker` on that
   configuration. Both converge in 29 outer iterations; STRICT still fails.
4. Copy the original configuration, set `pirls_tolerance=1e-10`, retain score
   tolerance 0.1, and run the same two workers. Both converge in 27 iterations;
   STRICT still fails. These two profiles do not replace the original default
   controls or the recorded performance run.

Use temporary per-worker JAX caches when reproducing these worker calls; no
user-global cache should be altered. Comparing each worker payload through
`_numerical_comparisons` produces separate STRICT and MODERATE field reports.

| Field | Default absolute residual | Score tolerance 0.001 | PIRLS tolerance 1e-10 |
| --- | ---: | ---: | ---: |
| Coefficients | 1.0071430e-9 | 9.6884450e-10 | 1.0227049e-9 |
| Means | 7.3624289e-9 | 7.2452870e-9 | 7.4062765e-9 |
| Smoothing parameters | 1.1494629e-6 | 1.1355176e-6 | 1.1614876e-6 |
| Smooth EDF | 7.9588949e-9 | 7.5157569e-9 | 8.1277145e-9 |
| Deviance | 1.1029465e-7 | 1.0764143e-7 | 1.1123302e-7 |
| REML score | 4.7998583e-10 | 5.4205884e-10 | 4.5906745e-10 |

All profiles pass MODERATE diagnostics with finite converged fits and matching
outer counts. X is full rank 51, with condition number 5.39173 in setup and
13.09053 in fitting coordinates; total penalty rank is 40. A reconstructed
reporting-state Fisher plus combined penalty is full rank 51, condition number
37751.88 for each implementation, minimum eigenvalue 86.8367, maximum
3278248.70, smoothing ratio about 213876, and symmetry residual at most
2.28e-13. These are reporting-state diagnostics, not captured pre-gdi factors
or a causal error bound. The evidence isolates serialization/oracle extraction
and public conversion; it does not identify a complete algorithmic cause.

The user's active-goal policy is: "If something doesn't pass STRICT tolerance
and you can't solve the issue after 4 passes you may lower the tolerance to
MEDIUM. You may never move STRICT to LOOSE." MEDIUM means repository MODERATE.
The September 17 restart handoff records the same authority; main-agent review
in `resumed_review_2026-09-17.md` records numerical review of these two frozen
profiles after five case-specific attempts. The representative runtime selector
has not been changed.

After inspecting those four failed attempts plus input provenance, finite
convergence, matching counts, and measured residuals, main-agent review on
September 17 accepted MODERATE for this exact CSV hash/formula/seed/n/p/m and
default-control combination. Only coefficients, means, smoothing, and EDF
need MODERATE; deviance, score, and scale retain STRICT, and outer counts remain
exact. The proposed field scope keeps the default CLI STRICT and admits no
other input or controller profile. The original report/exit 2 are
unchanged. This is numerical acceptance of a bounded benchmark observation,
not a method-wide tolerance or speedup claim.

The [separate reviewed decision](efs_benchmark_artifacts/2026-09-17/reviewed-acceptance.json)
evaluates those field-specific classes on the preserved payload; it records
original STRICT selection/exit 2 and claims no rerun. The current r4 program
still restricts explicit MODERATE to the smoke case. Implementing the second
reviewed selector scope encountered two automatic-review authorization
rejections, and those rejected changes were not applied. Representative CLI
selection therefore remains STRICT until that integration is resolved.
