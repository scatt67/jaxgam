# Dense EFS benchmark validation, 2026-09-17

This harness extends published #61 (`951ca4b`). It compares complete public
Newton and dense EFS fits with pinned R 4.5.2 / mgcv 1.9-3 EFS on one saved
Poisson/log input. The preserved smoke fails STRICT and passes MODERATE.
Following five unsuccessful correction/diagnostic passes, main-agent review
accepted MODERATE only for this exact saved smoke input, model, and default
controls under the user's four-pass policy. The default CLI selection remains
STRICT. No production optimizer or global tolerance changes are included.

## Reproduction and measurement scope

Run `make benchmark-efs-smoke` in the pinned test environment: that target
explicitly selects `--comparison-tolerance MODERATE` for the reviewed smoke.
`uv run python scripts/benchmark_efs.py --smoke --warm-repeats 1 --output-dir PATH`
retains the default STRICT selection and expected exit 2.
The representative target is `make benchmark-efs`: n=1200, ten cubic smooths,
k=6, seed=20260910, three warm repetitions, one configured thread. Each run
saves `input.csv`, `config.json`, and `report.json`. Set
`JAXGAM_BENCHMARK_SOURCE_COMMIT` and `JAXGAM_BENCHMARK_IMAGE_ID` when the image
does not contain git metadata. The report also records the script hash.

The driver writes 17-significant-digit CSV, reloads with roundtrip parsing,
and derives setup and smoothing starts from those saved values. Every Python
worker checks the CSV hash and uses the same parser; R reads the same saved
decimal input. EFS and R share the exact natural-scale initial smoothing
parameters through R's `in.out`; Newton retains its own public initializer.
Both policies and their vectors are recorded separately. EFS controller values
map to R's `epsilon`, `maxit`, `efs.lspmax`, and `efs.tol`; R's `efsudr` has a
source-fixed 200-iteration outer limit. Public Newton uses its unchanged
Poisson defaults, which are recorded in its worker payload.

Python workers run in separate subprocesses and temporary compilation-cache
directories. The first complete fit includes setup, compilation, fitting, and
Phase 3; warm samples repeat that same complete API call. Synchronization
precedes stopping each timer. R records first and repeated complete fits within
its separate process. Import/process-start time is outside those fit timers.
Every timed fit must converge and have finite fitted quantities; matching
infinities or unconverged fits cannot yield success. RSS is a 10 ms sampled sum
over the worker process tree, including interpreter/runtime overhead and
potentially shared pages; it is not exact allocator peak or isolated device
memory. Unavailable factorization, RHS, trial-score, device-memory, and stage
timing measurements remain null with reasons. Accepted-score history length
is not reported as a trial-evaluation count.

The bounded stdout/stderr regression retains root's TemporaryFile correction
for the unread-pipe deadlock and exercises 1 MiB on each stream with a timeout.
Other tests execute saved-data workers and an actual Python subprocess run,
check the schema and hash, and reject invalid fits and STRICT disagreement.

## Preserved smoke input and numerical review

The smoke uses n=128, p=13, m=3, full rank X, cubic k=5, seed=20260910.
Its CSV SHA256 is
`8a49df584d593df11ed6ffc44e66ec72ab37f2352ab37b353e1c42c4232a26b9`.
Default EFS/R controls are outer limit 200, log smoothing cap 15, score
tolerance 0.1, PIRLS tolerance 1e-7, and PIRLS limit 200. Both implementations
converge in 34 outer iterations with identical scale 1. The following are
maximum absolute residuals from the pinned aarch64 numerical review.

| Field | Default | Outer score tolerance 0.001 | PIRLS tolerance 1e-10 |
| --- | ---: | ---: | ---: |
| Coefficients | 8.5088006e-8 | 6.0857899e-8 | 8.5208618e-8 |
| Means | 3.4200173e-7 | 2.4365423e-7 | 3.4238113e-7 |
| Smoothing parameters | 3.5371006e-5 | 1.8159314e-5 | 3.5426895e-5 |
| Smooth EDF | 4.0398165e-7 | 2.4805806e-7 | 4.0452432e-7 |
| Deviance | 2.1089482e-6 | 1.3322856e-6 | 2.1117307e-6 |
| REML score | 2.2682769e-8 | 5.6113834e-9 | 2.2958289e-8 |
| Outer iterations, Python / R | 34 / 34 | 38 / 38 | 34 / 34 |

Five distinct correction/diagnostic passes failed to establish STRICT parity:

1. Corrected initialization and all Python input parsing to roundtrip the
   actual saved CSV. Default residuals above remain; this resolves the previous
   mismatched-input defect without resolving the numerical discrepancy.
2. Refit through the independent subprocess `RBridge.fit_efs`, on saved data
   with the exact EFS starting smoothing vector. Its coefficients, means,
   smoothing, EDF, deviance, score, scale, and outer count exactly match the
   benchmark R worker. Python-versus-R residuals remain unchanged.
3. Ran `dense_efs_known_scale` directly with `efs_initial_log_lambda` and mapped
   its coefficients through the existing Phase 3 coefficient transform.
   Direct/public coefficients, means, smoothing, score, deviance, and scale
   match exactly. The public adapter does not explain the remaining residual.
4. Tightened both implementations' score tolerance from 0.1 to 0.001 while
   retaining PIRLS 1e-7. The score now passes STRICT; the other fitted fields
   above still fail. This different controller profile does not alter the
   default-case acceptance result.
5. Tightened both implementations' PIRLS tolerance to 1e-10 with score
   tolerance 0.1. Residuals stay near the default values and still fail STRICT.

All three control profiles pass the MODERATE diagnostic, using the existing
rtol=1e-4 / atol=1e-6 class. STRICT uses rtol=1e-10 / atol=1e-12 and remains the
default CLI gate. These checks isolate input serialization, R extraction, and public
result conversion; they do not identify a complete algorithmic cause or
justify a broader parity claim.

Main-agent review on September 17 accepted an explicit MODERATE selection for
the default-profile smoke after inspecting all five passes, finite converged
fits, matching outer counts, and field residuals. The harness restricts that
selection to the saved CSV hash, formula, family, seed, n/p/m, and exact default
EFS controls recorded above. Both STRICT and MODERATE residual reports remain
available, and the selected class and review scope are recorded separately.
Invalid fits and mismatched outer counts still fail. Other input/model/control
combinations cannot select MODERATE through this harness. The representative
n=1200, m=10 run starts STRICT and needs independent numerical review if it
fails; no general method tolerance relaxation or LOOSE selection is included.

An additional source-coordinate conditioning diagnostic reconstructs the
reporting-state Poisson Fisher plus combined penalty. X has full rank 13,
condition number 5.01194 in setup coordinates and 7.10500 in fitting
coordinates; total penalty rank is 9. The selected penalized matrix has full
rank 13, condition number 222095.47 (Python) / 222095.50 (R), minimum eigenvalue
14.72025 and maximum eigenvalue 3269300.28; the smoothing ratio is about
196752. The symmetry residual is at most 1.07e-14. These are reconstructed
reporting-state diagnostics, not captured pre-gdi solver factors. They describe
the finite selected system but do not establish the numerical error's cause or
a tolerance bound.

The earlier numerical review passed 20 owning tests with 89% statement coverage
of `scripts/benchmark_efs.py`; full lint passed. That review used image
`sha256:8da633b0fea321ab9f1ecc955d60560239ad242729343339098cdbad20439408`
with the tested script hash recorded in the saved report. Subsequent metadata
additions explicitly record Newton controls and the reason for an unavailable
container CPU model. Final source/image and validation provenance accompanies
the handoff. Smoke and numerical-review timings were collected during other
validation and are not performance evidence.

Final selected-gate validation passed 24 owning tests in 17.22 seconds, pytest
and process exit 0, with 84.91% branch-inclusive script coverage (319 statements
and 92 branches). Full `make lint` passed. All 163 production/test/script Python
files match image
`sha256:f4eae281f047815852f2819db0760054a22e93112c7a5dca4031beeee834d6ba`;
the final benchmark script SHA256 is
`a4b2b794bf17ea9d7cb93f601786ecabf67a580207b64b5a3a55e93b530e43da`.
Tests retain the original STRICT failure, exercise explicit reviewed selection,
reject changed input/model/controls, and preserve convergence/iteration gates.
The real-R owning test now uses the standard pinned availability/version skip;
the mocked R-failure test remains available locally.

Representative measurements must run sequentially during a coordinated quiet
validation window. Any timing observation applies only to its frozen input,
code, container, and host; the harness makes no general speedup claim.
