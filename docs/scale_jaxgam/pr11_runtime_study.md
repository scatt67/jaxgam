# PR11 runtime study: discrete routes beside the undiscretized routes

This study records what the continuous-discrete routes cost next to the
undiscretized routes on the same models, at n = 100,000 and 1,000,000. It
reports measured numbers from one host run. It makes no general speedup
claim, and it does not treat different estimators as one quantity.
Approximation accuracy is the subject of
[pr11_approximation_study.md](pr11_approximation_study.md).

## Method

**Harness.** All records come from `scripts/benchmark_scaling.py`, using the
discrete, working-model, family and tensor modes added for this study:

```sh
make benchmark-scaling ARGS="--n 1000000 --k 10 --smooth-count 2 \
  --batch-rows 65536 --repeats 1 --result-mode prediction \
  --family poisson --execution discrete-wm --output /path/outside/checkout.json"
```

- **Isolation.** Each case runs as two fresh child processes, each with its
  own JAX compilation cache. The setup child measures Phase-1 preparation
  alone. The fit child runs one cold fit (compilation plus fit) and one warm
  fit. Every fit is timed through device completion of the coefficients.
- **Provenance.** Both children must report the same commit, loaded source,
  formula, input digest and route.
- **Runs.** One case at a time; each was run once.

**Routes compared.** The two discrete routes are measured against
undiscretized routes that make equivalent choices on the same data and
formula:

| Route | Smoothing | Compared with |
|---|---|---|
| `discrete` | fixed sp, all 1.0 | `stream-qr`: streamed pivoted-QR PIRLS at the same sp; `dense` for the tensor |
| `discrete-wm` | `working_model_reml` (`bgam.fitd` performance iteration) | `stream-exact`: exact streamed REML; `dense-reml`: dense Newton REML |

These are different estimators:

- The working-model strategy is a performance iteration; the exact routes
  optimize the nested REML criterion.
- The discrete route evaluates bases on the compact frame, where
  `bam(discrete=TRUE)` also normalizes the penalties. A fixed raw sp
  therefore does not give an identical effective penalty on the two routes.

The EDF column shows how close each pair's fitted models were.

**Models.**
- Additive: `y ~ s(x0, k=10, bs='cr') + s(x1, k=10, bs='cr')`, 19
  coefficients.
- Tensor: `y ~ te(x0, x1, k=6, bs='cr')`, 36 coefficients.
- Data: uniform covariates, seed 42, signals from the harness generators.
  - Gaussian has noise s.d. 0.3.
  - Poisson uses the harness's EFS Poisson generator.
  - Negative binomial is drawn at theta 3. Every NB route estimates theta
    from a start of 1.0; the theta column is the estimate.
- The streamed routes cannot prepare tensor smooths, so the tensor model is
  compared against the dense routes. The dense fixed-sp tensor fit uses
  `result="inference"`; every other route uses `result="prediction"` with
  `uncertainty="fisher"`.

**Controls.**
- `batch_rows=65536` on the streamed and discrete routes.
- Discretization: mgcv's default `bam(discrete=TRUE)` grid
  (`ContinuousDiscretizationControl(m=None)`).
- Prediction policy: `continuous_frozen_basis`.
- Default memory budget (512 MiB).

**Host.**
- Apple M4, 10 CPUs, 16 GB, macOS 15.6 (arm64).
- Python 3.13.10, JAX/jaxlib 0.9.0.1, NumPy 2.3.5 and SciPy 1.17.0
  (Accelerate), float64.
- `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. XLA's CPU thread
  pool was left at its default.
- The host was shared with other agents and a Docker VM and was swapping, so
  wall times carry load noise. For example, the two Gaussian additive
  discrete setups at n = 1m did identical work and took 8.63 s and 4.57 s.

**Provenance.** Commit `2e5026d` on the PR11 port, clean tree. Raw records
are kept outside the repository, and none is tracked.

## What the columns mean

| Column | Definition |
|---|---|
| Setup s, setup scans | Phase-1 preparation alone: the streamed `prepare_model` or the discrete route's own compact setup and indexed design (dense: `ModelSetup.build`), with its source scans |
| Cold s | First public fit in a fresh process: compilation plus fit, including the route's own preparation |
| Warm s | Second public fit, same process |
| Iter. | The route's public iteration count: outer REML or optimizer iterations, or performance-iteration steps (Gaussian fixed-sp discrete: pinned scale iterations) |
| Fit scans | The public `source_scans`: source scans made by the fit after Phase-1 setup |
| E2E scans, E2E MB read | Scans, and the `RowBatch` array bytes they yielded, over the whole cold public call, including preparation |
| Known WS MB | The public `known_workspace_bytes`: conservative known fit workspace, excluding retained Phase-1 setup |
| Fit peak MB | Discrete only: known workspace plus retained compact setup tables (the bound checked against the memory budget) |
| Setup RSS, Fit RSS | Process high-water RSS after setup (setup child) or after the warm fit (fit child). Includes the interpreter, imports and generated input. Before fitting, the fit children were already at 0.32–0.43 GB |

## Results

**gaussian additive, n = 100,000**

| Route | Setup s | Setup scans | Cold s | Warm s | Iter. | Fit scans | E2E scans | E2E MB read | Known WS MB | Fit peak MB | Setup RSS MB | Fit RSS MB | EDF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| discrete, fixed sp | 1.33 | 9 | 2.67 | 1.28 | 7 | 2 | 11 | 48.8 | 27.2 | 27.3 | 343 | 408 | 18.998 |
| discrete, working-model REML | 1.46 | 9 | 3.21 | 0.98 | 15 | 2 | 11 | 48.8 | 27.3 | 27.4 | 327 | 425 | 18.930 |
| streamed QR, fixed sp | 1.64 | 8 | 6.17 | 2.50 | 4 | 11 | 19 | 83.3 | 46.2 | – | 422 | 491 | 18.998 |
| streamed exact REML | 1.52 | 8 | 38.16 | 19.56 | 28 | 308 | 316 | 1538.6 | 212.6 | – | 424 | 644 | 18.930 |
| dense Newton REML | 0.80 | – | 8.70 | 1.24 | 10 | – | – | – | – | – | 344 | 556 | 18.930 |

**gaussian additive, n = 1,000,000**

| Route | Setup s | Setup scans | Cold s | Warm s | Iter. | Fit scans | E2E scans | E2E MB read | Known WS MB | Fit peak MB | Setup RSS MB | Fit RSS MB | EDF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| discrete, fixed sp | 8.63 | 9 | 5.93 | 5.07 | 7 | 2 | 11 | 401.6 | 38.0 | 38.1 | 408 | 457 | 19.000 |
| discrete, working-model REML | 4.57 | 9 | 5.70 | 4.48 | 17 | 2 | 11 | 401.6 | 38.1 | 38.2 | 415 | 423 | 18.992 |
| streamed QR, fixed sp | 21.32 | 8 | 30.25 | 26.18 | 4 | 11 | 19 | 833.0 | 46.2 | – | 416 | 551 | 19.000 |
| streamed exact REML | 21.76 | 8 | 278.12 | 268.65 | 38 | 488 | 496 | 24206.0 | 212.6 | – | 436 | 666 | 18.993 |
| dense Newton REML | 0.77 | – | 12.22 | 8.71 | 12 | – | – | – | – | – | 864 | 1339 | 18.993 |

**poisson additive, n = 100,000**

| Route | Setup s | Setup scans | Cold s | Warm s | Iter. | Fit scans | E2E scans | E2E MB read | Known WS MB | Fit peak MB | Setup RSS MB | Fit RSS MB | EDF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| discrete, fixed sp | 0.58 | 8 | 1.80 | 0.48 | 6 | 6 | 14 | 65.2 | 36.2 | 36.3 | 365 | 516 | 18.999 |
| discrete, working-model REML | 0.61 | 8 | 2.40 | 0.56 | 10 | 10 | 18 | 84.8 | 36.2 | 36.3 | 361 | 526 | 18.452 |
| streamed QR, fixed sp | 0.78 | 8 | 4.35 | 1.76 | 6 | 18 | 26 | 117.6 | 184.2 | – | 420 | 595 | 18.999 |
| streamed exact REML | 0.98 | 8 | 17.48 | 15.53 | 13 | 209 | 217 | 1053.5 | 212.6 | – | 438 | 702 | 18.451 |
| dense Newton REML | 0.65 | – | 7.63 | 1.41 | 7 | – | – | – | – | – | 386 | 522 | 18.451 |

**poisson additive, n = 1,000,000**

| Route | Setup s | Setup scans | Cold s | Warm s | Iter. | Fit scans | E2E scans | E2E MB read | Known WS MB | Fit peak MB | Setup RSS MB | Fit RSS MB | EDF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| discrete, fixed sp | 5.17 | 8 | 6.34 | 5.36 | 6 | 6 | 14 | 594.4 | 47.0 | 47.1 | 410 | 531 | 19.000 |
| discrete, working-model REML | 4.67 | 8 | 7.24 | 5.79 | 12 | 12 | 20 | 888.4 | 47.0 | 47.1 | 411 | 473 | 18.935 |
| streamed QR, fixed sp | 19.58 | 8 | 42.00 | 34.42 | 6 | 18 | 26 | 1176.0 | 184.2 | – | 410 | 551 | 19.000 |
| streamed exact REML | 19.69 | 8 | 322.73 | 322.39 | 11 | 517 | 525 | 25627.0 | 212.6 | – | 414 | 696 | 18.935 |
| dense Newton REML | 0.86 | – | 13.28 | 11.48 | 8 | – | – | – | – | – | 852 | 1176 | 18.934 |

**nb additive, n = 100,000**

| Route | Setup s | Setup scans | Cold s | Warm s | Iter. | Fit scans | E2E scans | E2E MB read | Known WS MB | Fit peak MB | Setup RSS MB | Fit RSS MB | EDF | theta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| discrete, fixed sp | 1.05 | 8 | 4.49 | 1.31 | 5 | 24 | 32 | 153.4 | 55.1 | 55.2 | 345 | 583 | 18.998 | 3.0775 |
| discrete, working-model REML | 1.09 | 8 | 5.16 | 1.67 | 9 | 43 | 51 | 246.5 | 55.1 | 55.2 | 350 | 542 | 18.334 | 3.0775 |
| streamed QR, fixed sp | 1.24 | 8 | 10.67 | 5.82 | 7 | 99 | 107 | 514.5 | 216.9 | – | 448 | 851 | 18.998 | 3.0752 |
| streamed exact REML | did not converge (no timing) |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dense Newton REML | 0.43 | – | 12.74 | 3.05 | 5 | – | – | – | – | – | 400 | 646 | 18.333 | 3.0752 |

**nb additive, n = 1,000,000**

| Route | Setup s | Setup scans | Cold s | Warm s | Iter. | Fit scans | E2E scans | E2E MB read | Known WS MB | Fit peak MB | Setup RSS MB | Fit RSS MB | EDF | theta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| discrete, fixed sp | 4.54 | 8 | 10.09 | 7.12 | 5 | 27 | 35 | 1623.4 | 65.9 | 66.0 | 423 | 492 | 19.000 | 2.9987 |
| discrete, working-model REML | 4.72 | 8 | 14.76 | 12.43 | 11 | 70 | 78 | 3730.4 | 65.9 | 66.0 | 352 | 475 | 18.905 | 2.9987 |
| streamed QR, fixed sp | 19.41 | 8 | 147.07 | 138.48 | 8 | 183 | 191 | 9261.0 | 216.9 | – | 444 | 647 | 19.000 | 2.9984 |
| streamed exact REML | did not converge (no timing) |  |  |  |  |  |  |  |  |  |  |  |  |  |
| dense Newton REML | 0.90 | – | 37.53 | 24.72 | 6 | – | – | – | – | – | 782 | 1936 | 18.899 | 2.9984 |

**gaussian tensor, n = 100,000**

| Route | Setup s | Setup scans | Cold s | Warm s | Iter. | Fit scans | E2E scans | E2E MB read | Known WS MB | Fit peak MB | Setup RSS MB | Fit RSS MB | EDF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| discrete, fixed sp | 1.12 | 9 | 2.35 | 1.06 | 7 | 2 | 11 | 48.8 | 65.0 | 65.1 | 345 | 436 | 35.929 |
| discrete, working-model REML | 1.17 | 9 | 3.96 | 0.94 | 14 | 2 | 11 | 48.8 | 65.6 | 65.7 | 353 | 447 | 34.837 |
| dense, fixed sp | 0.85 | – | 8.55 | 0.98 | 2 | – | – | – | – | – | 430 | 695 | 35.929 |
| dense Newton REML | 0.27 | – | 4.74 | 1.18 | 9 | – | – | – | – | – | 457 | 845 | 34.839 |

**gaussian tensor, n = 1,000,000**

| Route | Setup s | Setup scans | Cold s | Warm s | Iter. | Fit scans | E2E scans | E2E MB read | Known WS MB | Fit peak MB | Setup RSS MB | Fit RSS MB | EDF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| discrete, fixed sp | 4.74 | 9 | 5.48 | 4.18 | 7 | 2 | 11 | 401.6 | 75.8 | 75.9 | 399 | 424 | 35.993 |
| discrete, working-model REML | 4.48 | 9 | 6.61 | 4.58 | 15 | 2 | 11 | 401.6 | 76.4 | 76.5 | 398 | 432 | 35.850 |
| dense, fixed sp | 1.97 | – | 10.27 | 6.38 | 1 | – | – | – | – | – | 1148 | 1564 | 35.993 |
| dense Newton REML | 1.84 | – | 23.49 | 24.77 | 11 | – | – | – | – | – | 1170 | 1606 | 35.850 |

At both sizes, the NB exact streamed REML fit
(`NegativeBinomial(fixed=False)`, `linear_solver="qr"`) stopped at L-BFGS-B's
relative-reduction criterion without satisfying its gradient test. The
projected gradient was 1.13e-8 at n = 100k and 1.56e-4 at n = 1m, against
`gtol` 1e-9. The route reported `converged=False`, and the harness records no
timing for an unconverged fit. Dense Newton REML is the undiscretized
estimated-smoothing reference for NB.

## Observations from these runs

All of the following are from single runs on the loaded host described
above.

**Fit scans.**
- The Gaussian discrete routes made 2 fit scans, statistics then residuals,
  with fixed sp and with working-model REML, at both sizes.
- The exact streamed Gaussian REML route made 308 (100k) and 488 (1m).
  Streamed fixed-sp QR made 11.
- The non-Gaussian discrete routes make one or two scans per performance
  iteration, plus the theta updates for NB:
  - Poisson: 6–12 fit scans.
  - NB at 1m: 27 fit scans with fixed sp and 70 with working-model REML,
    against 183 for streamed fixed-sp QR.

**Bytes read at n = 1m, whole public call.**

| Model | Discrete (fixed / working-model) | Exact streamed REML | Streamed fixed-sp QR |
|---|---|---|---|
| Gaussian | 0.40 GB / 0.40 GB | 24.2 GB | 0.83 GB |
| Poisson | 0.59 GB / 0.89 GB | 25.6 GB | 1.18 GB |
| NB | 1.62 GB / 3.73 GB | did not converge | 9.26 GB |

**Warm fits at n = 1m.**

| Model | Discrete, working-model REML | Dense Newton REML | Exact streamed REML | Discrete, fixed sp | Streamed fixed-sp QR | Dense fixed sp |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian additive | 4.5 s | 8.7 s | 268.7 s | 5.1 s | 26.2 s | – |
| Poisson additive | 5.8 s | 11.5 s | 322.4 s | 5.4 s | 34.4 s | – |
| NB additive | 12.4 s | 24.7 s | did not converge | 7.1 s | 138.5 s | – |
| Gaussian tensor | 4.6 s | 24.8 s | – | 4.2 s | – | 6.4 s |

These are wall times of different estimators on a shared host, not a
controlled speed comparison.

At n = 100k the discrete and dense warm times were between 0.5 and 1.7 s,
except NB dense REML at 3.1 s. Cold times, which include compilation, were
between 1.8 and 12.7 s.

**Setup.**
- Discrete Phase-1 setup took 4.5–8.6 s at n = 1m.
- The streamed `prepare_model` took 19.4–21.8 s for the same models.
- Dense `ModelSetup.build` took under 2 s but materializes the n-by-p
  design.

**Memory at n = 1m.**
- Fit-child high-water RSS:
  - discrete routes: 0.42–0.53 GB;
  - streamed routes: 0.55–0.70 GB;
  - dense routes: 1.18–1.94 GB.
- Known fit workspace:
  - discrete routes: 38–76 MB;
  - streamed exact REML: 213 MB;
  - streamed QR: 46–217 MB.
- The discrete retained compact setup tables added about 0.1 MB, the
  difference between the discrete fit peak and known workspace.

**Agreement of the paired fits.**
- Fixed sp: total EDF of the discrete and undiscretized fits agree to the
  three decimals shown at both sizes. Examples: 18.998 / 18.998 at 100k and
  19.000 / 19.000 at 1m; tensor 35.929 / 35.929 and 35.993 / 35.993.
- Working-model REML against the exact routes: total EDF differs by at most
  0.002 for Gaussian, Poisson and the tensor (for example 18.992 against
  18.993, Gaussian at 1m). For NB against dense REML it differs by 0.001 at
  100k and 0.006 at 1m.
- The NB theta estimates agree to 2.3e-3 at 100k (3.0775 against 3.0752) and
  3e-4 at 1m (2.9987 against 2.9984).
- Prediction and interval accuracy are not measured here; see the
  approximation study.

## Limitations

- One run per case on a loaded laptop-class host with Accelerate. There are
  no repeated measurements or confidence intervals, and no pinned Linux or
  x86_64 runs.
- Only mgcv's default discretization grid, and two model families of
  modest p. The plan's 10m-row, many-term, high-cardinality and
  expensive-kernel cases are not covered.
- Bytes read are `RowBatch` array bytes from an in-memory `ArrayRowSource`,
  not disk I/O.
- RSS is process high-water and includes the runtime. Device allocator peaks
  were not exposed by the CPU backend.
- The NB exact streamed REML route did not converge on this model at either
  size, so no exact streamed NB timing exists here.
