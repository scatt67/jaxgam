# AGENTS.md - JaxGAM

## What This Project Is

JaxGAM is a Python port of Simon Wood's R package `mgcv` (Mixed GAM Computation Vehicle) for fitting Generalized Additive Models. The package version is `1.0.0`.

What the code implements today:

- **Default dense fit.** `gam(method="REML")`: PIRLS inside a Newton optimizer for the smoothing parameters, in memory.
- **Extended Fellner-Schall optimizer** (`optimizer="efs"`), on the dense, streamed and operator routes.
- **Row-source routes**, selected with `FitControl(execution=...)`:
  - `"stream"`: fits from a replayable row source in batches (fixed `sp`, exact REML, or Fellner-Schall; optional Gaussian QR compression).
  - `"discrete"`: the analogue of `bam(discrete=TRUE)` (compact discretized design, working-model REML).
  - `"operator"`: matrix-free conjugate gradients for random effects with very many levels.
- **Families:** Gaussian, Binomial, Poisson, Gamma, Negative Binomial (theta estimated or fixed), each with non-default links through the family class.
- **Smooths:** `tp`, `ts`, `cr`, `cs`, `cc`, `gp`, `re`, `te`, `ti`, and `by=` variables.
- **Results:** `result="full"` (`GAMResults`), `"inference"` (`GAMInferenceResult`), `"prediction"` (`GAMPredictionResult`); the row-source routes return prediction results only.

What is not implemented is listed under [What Is Not Implemented](#what-is-not-implemented).

## Where To Read

| You need | Read |
|---|---|
| How the code is organized today, with flow and class diagrams | `docs/architecture/` (start at `index.md`) |
| How a user calls each feature | `docs/guide/`, `docs/api.md`, `docs/quickstart.md` |
| The plans the row-source routes and Fellner-Schall were built from | `docs/scale_jaxgam/` |
| The planned refactor of `fitting/` and `execution/` | `docs/unify_fit/` |
| Which mgcv source to read for a task | The table under [R Source Reference](#r-source-reference); each architecture page names the routines its area ports |
| Design records of earlier features (random effects, Gaussian process, negative binomial, the result modes) | `docs/<feature>/design.md` |

The design records and plans describe what was intended when they were written. Where one disagrees with the code about what is implemented, the code and `docs/architecture/` are right.

## Project Setup

We use uv for dependency and package management.

- `make install` runs `uv sync --extra dev`.
- R comparison tests also need rpy2: `uv sync --extra dev --extra r`.
- On Apple silicon, install the OpenBLAS builds of NumPy and SciPy (see README, "Apple silicon"). With the default Accelerate builds `import jaxgam` warns, and large fits can return wrong results.

## Code Quality

Pre-commit runs `ruff --fix`, `ruff-format` and `vulture` on every commit (`make pre-commit-install` once).

- `make lint`: `ruff check`, `ruff format --check`, `vulture jaxgam --min-confidence 80`. CI runs this.
- `make format`: `ruff format`.

## Repo Developer Tasks Orchestration

We use `make` to orchestrate common tasks. Whenever we add new repo or CI capabilities, update the `Makefile` and call the tool through `make`. `make help` lists the targets.

## Test Environments

R comparison tests need the pinned **R 4.5.2 + mgcv 1.9-3**. `RBridge.check_versions()` checks both, and tests skip when R is missing or the versions differ.

- `make test-local`: the suite on this machine (`uv run pytest`).
- `make test`: the suite in the Docker image, which has the pinned R (needs colima; stops at the first failure).
- `make test-file FILE=tests/...`: one file in Docker.
- `make test-cov`: Docker, with coverage.

The Docker image pins BLAS to one thread (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`) so R's reference arithmetic is the same on every machine. `make test-local` does not; set those variables yourself when comparing numbers.

CI (`.github/workflows/test.yml`) runs the suite as **eight shards on parallel runners**, a `coverage` job that combines them and enforces 80%, eight replicas of the `platform_sensitive` tests, and four focused jobs, all behind one `gate` job. A test's shard is a hash of its node id (`tests/sharding.py`); `JAXGAM_TEST_SHARD=3/8` runs one shard locally.

## Architecture (Three Phases)

Every fit flows through three phases. This boundary is load-bearing - do not mix phases.

**Phase 1 - Setup (NumPy only).** Parse the formula, construct bases and penalties, absorb identifiability constraints (`CoefficientMap`), and for row sources scan the data to place knots or build the compact discretized design. Packages: `formula/`, `smooths/`, `penalties/`, `data/`. No module in these packages imports JAX.

**Phase 2 - Fit (JAX).** Two packages:

- `fitting/`: JAX kernels (PIRLS, the REML criterion, batch reductions, indexed design operators) and the dense Newton optimizer. Kernels that are JIT-compiled must be pure and use `jax.lax` control flow. The Newton outer loop in `fitting/newton.py` and `fitting/stable_log_det.py` run on the host between jitted evaluations.
- `execution/`: host-side drivers that own iteration, step control and data scans for the Fellner-Schall, Gaussian QR, streamed, discretized and operator routes. They call the kernels in `fitting/`.

`fitting/` does not import `execution/`, with one lazy exception in `fitting/data.py` (`structural_rank`).

Both packages, and `formula/`, are laid out by route. A module that only one route uses lives in that route's sub-package: `stream/`, `discrete/`, `operator/` and, in `fitting/` and `execution/`, `gaussian_qr/`. A module that the dense fit or more than one route uses stays at the top level of its package. The same route has the same sub-package name in each package, so `execution/operator/` drives the kernels in `fitting/operator/` on the model that `formula/operator/` prepares. A sub-package's `__init__.py` holds a description and imports nothing.

**Phase 3 - Post-estimation (NumPy).** `results.py`, `inference/`, `summary/`, `plot/`. Prediction uses the `CoefficientMap` and frozen smooth state from Phase 1. No module in `inference/`, `summary/` or `plot/` imports JAX.

```
Phase 1 (NumPy)  ──jax.device_put──▶  Phase 2 (JAX)  ──np.asarray──▶  Phase 3 (NumPy)
```

`families/` and `links/` are used by Phase 2 and import JAX. Phase 1 code receives family objects as arguments and never imports those packages.

## R Source Reference

The mgcv R source is cloned locally. **Always read the corresponding R implementation before writing Python.** The R code is ground truth for numerical edge cases, special handling, and algorithmic choices that no document here captures.

The clone location is in the `MGCV_SOURCE` environment variable (`echo $MGCV_SOURCE`). If it is unset, clone the CRAN mirror (`git clone https://github.com/cran/mgcv.git`) and export the variable. The tests are pinned to **mgcv 1.9-3**; if the clone's HEAD is newer, read the 1.9-3 commit (`fb7e8e7`).

Key R files:

| R file | Contains |
|---|---|
| `$MGCV_SOURCE/R/mgcv.r` | Top-level `gam()`, `gam.setup()`, `gam.side()` (identifiability), `gam.outer()`, `magic()`, `predict.gam()`, `summary.gam()`. |
| `$MGCV_SOURCE/R/smooth.r` | All smooth constructors (basis + penalty), `smoothCon()`. About 4,400 lines. Search for `smooth.construct.XX.smooth.spec`. |
| `$MGCV_SOURCE/R/gam.fit3.r` | PIRLS (`gam.fit3`), `newton()` for the smoothing parameters, `gam.reparam()`, `negbin()`, family fix-ups. |
| `$MGCV_SOURCE/R/gam.fit4.r` | `gam.fit4`, `gam.fit5` (extended families) and `efsudr` (extended Fellner-Schall). |
| `$MGCV_SOURCE/R/efam.r` | Extended families, including `nb()`, and `estimate.theta`. |
| `$MGCV_SOURCE/R/fast-REML.r` | The `Sl.*` routines, including `Sl.fitChol` (working-model REML step used by `bam`). |
| `$MGCV_SOURCE/R/bam.r` | `bam()`, `bgam.fitd` (performance iteration), `discrete.mf`, `compress.df`, `predict.bamd`. |
| `$MGCV_SOURCE/R/plots.r` | `plot.gam()`. |
| `$MGCV_SOURCE/src/tprs.c` | Thin-plate eigendecomposition and knot selection. |
| `$MGCV_SOURCE/src/gdi.c` | REML derivatives, `get_stableS`, `MLpenalty1`. |
| `$MGCV_SOURCE/src/discrete.c` | Discretized cross-products (`XWXd` and friends). |

**R reading tips:**
- `<-` is assignment. `.C()` / `.Call()` invoke C code in `src/`.
- R uses 1-based indexing - adjust when porting.
- `mgcv:::function_name` accesses unexported internals. These are often the functions we need most.
- Run `debug(mgcv:::gam.fit3)` in R to step through PIRLS and compare intermediate values.

## Code Conventions

### File Organization

```
jaxgam/
├── __init__.py          # Public API; enables float64, compile cache, Accelerate warning
├── api.py               # GAM class: argument checks, routing, fit orchestration
├── control.py           # FitControl, EFSControl, OperatorControl, ContinuousDiscretizationControl
├── results.py           # GAMResults, GAMInferenceResult, GAMPredictionResult
├── registry.py          # Generic string-key registry
├── jax_utils.py         # Array dispatch, device transfer, linear-algebra helpers
├── platform_guard.py    # Apple Accelerate LAPACK warning
├── data/                # Phase 1: row sources, discretization
├── formula/             # Phase 1: parsing, terms, design, prepared model
│   ├── discrete/        #   setup for the discretized route
│   └── operator/        #   setup for the operator route
├── smooths/             # Phase 1: bases, constraints, pinned LAPACK provider
├── penalties/           # Phase 1: penalty construction
├── families/            # Families (Phase 2; import JAX)
├── links/               # Link functions
├── fitting/             # Phase 2: JAX kernels (PIRLS, REML, EFS algebra) + dense Newton optimizer
│   ├── gaussian_qr/     #   kernels of one route each
│   ├── stream/
│   ├── discrete/
│   └── operator/
├── execution/           # Phase 2: host drivers; top level holds the EFS loop and shared reductions
│   ├── gaussian_qr/     #   drivers of one route each
│   ├── stream/
│   ├── discrete/
│   └── operator/
├── inference/           # Phase 3: prediction-only core (GAMPredictor)
├── summary/             # Phase 3: summary, EDF, p-values
└── plot/                # Phase 3: matplotlib plotting
tests/
├── conftest.py          # Shared fixtures, R bridge fixture, shard selection
├── helpers.py           # SEED, N, r_available, _AssertCollector, per-platform replay bounds
├── tolerances.py        # STRICT / MODERATE / LOOSE
├── r_bridge.py          # Typed rpy2 bridge to pinned R 4.5.2 + mgcv 1.9-3
├── sharding.py          # CI shard assignment
├── test_validation_matrix.py   # Broad final-model R parity and hard-gate invariants
└── test_<package>/      # One directory per package area (not divided by route)
scripts/                 # Benchmarks and demos
```

### Naming and Style

- Python 3.11+. Type hints on all public functions.
- `snake_case` everywhere. No abbreviations except established ones: `n` (observations), `p` (parameters), `k` (basis dimension), `edf` (effective degrees of freedom).
- Penalty matrices: `S` or `S_j` (per-smooth), `S_lambda` (combined weighted penalty).
- Model matrix: `X` (full), `X_s` (smooth block), `X_p` (parametric block).
- Greek letters in variable names: `eta` (linear predictor), `mu` (mean), `beta` (coefficients), `lambda_` (smoothing parameters, trailing underscore to avoid keyword).

### JAX Rules (Phase 2 code)

- Functions that are JIT-compiled must be pure - no side effects, no Python `if` on array values. Use `jax.lax.while_loop` and `jax.lax.cond`.
- Host loops (the Newton optimizer, every driver in `execution/`) may branch in Python on values pulled back from the device. Keep them out of jitted functions.
- Float64 everywhere: `jaxgam/__init__.py` sets `jax_enable_x64`. GAMs are numerically sensitive - float32 is not acceptable.
- On the row-source routes, every array a driver holds must be covered by a memory budget check (`FitControl.memory_budget_bytes`, `DiscretizationBudget`). Do not add an unaccounted array that grows with the number of rows.

### NumPy Rules (Phase 1 and 3 code)

- No JAX imports in `formula/`, `smooths/`, `penalties/`, `data/`, `inference/`, `summary/`, `plot/`. Use `numpy` and `scipy` only.

### Testing Rules

- Every new module gets a corresponding test file.
- Use the tolerance classes from `tests/tolerances.py`, never a hand-written `rtol`/`atol`: `STRICT` (rtol 1e-10, atol 1e-12), `MODERATE` (1e-4, 1e-6), `LOOSE` (1e-2, 1e-4).
- **R comparisons must match R.** Use `STRICT` wherever JaxGAM reproduces mgcv's arithmetic. Use `MODERATE` or `LOOSE` only for a quantity that is genuinely ill-determined or depends on the BLAS/LAPACK build (for example smoothing parameters on a flat criterion), and say why in a comment with the measured difference. Never loosen a tolerance to hide an unexplained difference.
- R comparison tests go through the typed rpy2 bridge in `tests/r_bridge.py`. Do not add subprocess, `Rscript`, CSV or saved-fixture paths: generate inputs in code and get reference values from the bridge.
- Broad final-model R parity and hard-gate invariants live in `tests/test_validation_matrix.py`. Other R-bridge tests should own layer-specific behavior (optimizer internals, fixed-`sp` PIRLS numerics, prediction/SE, summary tables, smooth construction, the bridge itself) rather than duplicate final-result field checks.
- When several related assertions share one expensive fixture or R fit, consolidate them with `tests.helpers._AssertCollector` so failures remain named without multiplying collected tests.
- Hard-gate invariants run in every CI run and block the build: convergence, deviance non-negativity, finite `eta`/`mu`, EDF bounds, covariance symmetric PSD, penalty PSD, `theta > 0`, rank conditions (`TestHardGateInvariants`), and objective monotonicity and Hessian symmetry in `tests/test_fitting/`.
- Mark a test `platform_sensitive` when its outcome can depend on the CPU's SIMD maths or the BLAS kernel; CI runs those on eight runners.
- Coverage: CI enforces 80% on the whole suite, with branch coverage, over the shards combined.
- Tests must not depend on other tests having run first; CI runs each test in one of eight shards.

### Commit and PR Conventions

- One logical change per commit. "Add TPRS basis construction" not "Add smooths".
- Every PR must include tests. No code without tests.
- PR title format: `[phase] component: description` - e.g., `[phase1] smooths/tprs: implement thin plate basis construction`.
- If a change touches Phase 2 code, the PR must include a JIT compilation test (the function compiles and runs without error under `jax.jit`).

## What Is Not Implemented

Do not implement any of the following without an explicit decision. They are out of scope today:

- Sparse-CPU execution (CHOLMOD, scikit-sparse)
- `method="ML"`, GCV and UBRE smoothness selection (REML only)
- Extended families beyond Negative Binomial (Tweedie, Beta, SHASH, Cox PH, ordered categorical)
- P-splines (`ps`, `cp`), B-splines (`bs`), `t2()` tensor products
- Factor-smooth interactions (`bs="fs"`)
- `select=TRUE`, `fx=TRUE`, per-term `sp=`, linked smoothing parameters (`id=`), user-specified `knots=`, `gamma` other than 1
- `offset()` and interaction terms (`x:z`, `x*z`) inside the formula (pass an offset to `fit()` or through the row source)
- Multi-GPU, Ray, distributed fitting
- `gamm()`, PQL
- Exotic smooths (soap film, MRF, adaptive, Duchon)
- `summary()` and `plot()` on the row-source routes (they return prediction results)

If you add a guard for an unsupported feature, raise `NotImplementedError` with a message that names the feature and, where there is one, the supported alternative.

## Common Pitfalls

1. **TPRS eigendecomposition must match R exactly.** The eigenvalue truncation and null space handling determine the basis. Small differences here cascade into large coefficient differences. Validate against R's `smooth.construct.tp.smooth.spec` output, not just final fit results.

2. **Knot placement matters.** mgcv uses a specific algorithm for selecting knots from the data (max-min distance for TPRS, quantile-based for cubic). If your knots don't match R's, nothing else will match either.

3. **CoefficientMap is the contract between Phase 1 and Phase 3.** It maps constrained coefficients (from the fit) back to the original basis for prediction. If this is wrong, `predict()` produces garbage even when the fit is correct. Test the roundtrip: `predict(model, original_data)` must reproduce `model.fitted_values`.

4. **The REML criterion is flat near the optimum.** Smoothing parameters can differ from R and the fit is still correct. Validate deviance and coefficients before chasing smoothing-parameter precision. With many smooths the discretized route's stopping point is not determined beyond its stopping rule, in JaxGAM or in `bam`.

5. **Step-halving is essential, not optional.** PIRLS without step-halving diverges on Binomial and Gamma models.

6. **Sum-to-zero constraints interact with factor-by smooths.** When `s(x, by=fac)` coexists with `s(x)`, the constraint absorption must be applied correctly or identifiability fails. Test with multi-smooth models from the start.

7. **A row source must replay identically.** A fit scans its source several times. Every scan must return the same rows in the same order, and the source owns the prior weights and offset. A custom source's `fingerprint()` is called many times per fit, so it must be cheap.

8. **Last-bit changes move ill-determined results.** A change that alters rounding (operation order, a different LAPACK) can move a result that sits on a flat criterion. Check whether the old code reaches the new value under a one-ulp perturbation before treating it as a bug, and never change a stopping rule to make a test pass.
