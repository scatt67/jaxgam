# AGENTS.md - JaxGAM

## What This Project Is

JaxGAM is a Python port of Simon Wood's R package `mgcv` (Mixed GAM Computation Vehicle) for fitting Generalized Additive Models. The v1.0 release is deliberately scoped: dense-only execution, four exponential families, three smooth types + tensor products, REML/ML optimization. See `docs/design.md` Section 1.2 for the full scope boundary.

The design document at `docs/design.md` is the authoritative reference for all architectural decisions. It is ~8000 lines. **Do not attempt to read it entirely.** Instead, read the specific sections referenced in each task. The table of contents is in the first 30 lines.

## Project Setup
We use uv for dependency and package management; init using uv and use `uv sync` when needed.

## Code Quality
We use pre-commit for code quality checks, and ruff for python specific linting, and formatting. 


## Repo Developer Tasks Orchestration
We use `make` and `Makefile` to orchestrate common tasks such as running linters, and static analysis. Whenever we add new repo, CICD capabilities please update the `Makefile` and use make to run common tools.

## Docker Test Environment
R comparison tests require pinned R 4.5.2 + mgcv 1.9-3 (enforced by `RBridge.check_versions()`). Without the correct versions, R tests auto-skip.

- `make test` - full suite in Docker (includes R comparison tests)
- `make test-local` - local tests only (R tests skip if R unavailable/wrong version)

## Architecture (Three Phases)

Every `gam()` call flows through three phases. This boundary is load-bearing - do not mix phases.

**Phase 1 - Setup (CPU, NumPy only).** Parse formula → construct basis matrices → build penalty matrices → assemble full model matrix X → apply identifiability constraints via `CoefficientMap`. No JAX imports permitted in Phase 1 code.

**Phase 2 - Fit (JAX, JIT-compiled).** Dense X and S_λ are transferred to device via `jax.device_put`. PIRLS inner loop (penalized iteratively reweighted least squares) nested inside a REML outer Newton loop. All code in `fitting/` must be JIT-compatible - no Python-level control flow that depends on array values (use `jax.lax.while_loop`, `jax.lax.cond`).

**Phase 3 - Post-estimation (CPU, NumPy).** Coefficients come back via `np.asarray()`. `predict()`, `summary()`, `plot()` operate on CPU NumPy arrays using the `CoefficientMap` from Phase 1.

```
Phase 1 (NumPy)  ──jax.device_put──▶  Phase 2 (JAX JIT)  ──np.asarray──▶  Phase 3 (NumPy)
```

## R Source Reference

The mgcv R source is cloned locally from the CRAN GitHub mirror. **Always read the corresponding R implementation before writing Python.** The R code is ground truth for numerical edge cases, special handling, and algorithmic choices that the design doc doesn't capture.

The clone location is in the `MGCV_SOURCE` environment variable. Run `echo $MGCV_SOURCE` to find it. See `docs/R_SOURCE_MAP.md` for setup instructions if the variable is not set.

`docs/R_SOURCE_MAP.md` maps every implementation task to the specific R files and functions to read. It also has a quick-lookup table for debugging ("my coefficients don't match R → look here").

Key R files you'll reference most often:

| R file | Contains |
|---|---|
| `$MGCV_SOURCE/R/smooth.r` | All smooth constructors (basis + penalty). ~8000 lines. Search for `smooth.construct.XX.smooth.spec` where XX is the basis type. |
| `$MGCV_SOURCE/R/gam.fit3.r` | PIRLS inner loop (`gam.fit3`), working weights, step-halving, convergence. |
| `$MGCV_SOURCE/R/fast-REML.r` | REML/ML criterion computation, Newton optimizer for λ. |
| `$MGCV_SOURCE/R/gam.r` | Top-level `gam()`, `predict.gam()`, `summary.gam()`, `gam.setup()`, `gam.side()` (identifiability). |
| `$MGCV_SOURCE/R/families.r` | Family definitions: `variance`, `dev.resids`, `initialize`, `validmu`. |
| `$MGCV_SOURCE/src/tprs.c` | C implementation of TPRS eigendecomposition and knot selection. The R wrapper just calls this. |
| `$MGCV_SOURCE/src/gdi.c` | C implementation of REML derivatives. Reading this helps validate our `jax.grad` output. |

**R reading tips:**
- `<-` is assignment. `.C()` / `.Call()` invoke C code in `src/`.
- R uses 1-based indexing - adjust when porting.
- `mgcv:::function_name` accesses unexported internals. These are often the functions we need most.
- Run `debug(mgcv:::gam.fit3)` in R to step through PIRLS and compare intermediate values.

## Design Document Reference

Location: `docs/design.md`

Key sections by topic area:

| If you're working on... | Read these sections |
|---|---|
| Smooth basis construction | §5.1 (base class), §5.2 (TPRS), §5.3 (cubic), §5.5 (tensor) |
| Factor-by smooths | §5.7 (full mechanism), §5.7.3 (identifiability) |
| Distribution families | §6.1 (base class), §6.2 (standard families) |
| Link functions | §7 |
| Penalty matrices | §8 |
| Autodiff strategy | §9.1 (where AD helps/hurts), §9.3 (extended family strategy - v1.18 rewrite) |
| PIRLS inner loop | §4.1, §4.2 |
| REML/ML outer loop | §4.3, §4.4 |
| Convergence & step-halving | §4.5 |
| Identifiability constraints | §5.10 (CoefficientMap) |
| Formula parsing | §13.1 (AST-based parser) |
| Design matrix assembly | §13.2 |
| Execution path routing | §10.1, §10.2 |
| Testing strategy | §18.1 (philosophy + tolerances), §18.2 (R bridge) |
| Numerical stability | §4.8 (jitter), §9.3 (AD stability) |

## Code Conventions

### File Organization

```
jaxgam/
├── __init__.py          # Public API only: gam, predict, summary, plot
├── api.py               # Top-level fitting orchestration
├── formula/             # Phase 1: parsing and term representation
├── smooths/             # Phase 1: basis and penalty construction
├── families/            # Families and link functions (used in Phase 1 + 2)
├── links/               # Link function implementations
├── penalties/           # Penalty matrix construction
├── fitting/             # Phase 2: PIRLS, REML, convergence (JAX)
├── linalg/              # Linear algebra (backend-aware)
├── autodiff/            # JAX AD wrappers
├── predict/             # Phase 3: prediction
├── summary/             # Phase 3: summary, EDF, p-values
├── plot/                # Phase 3: matplotlib plotting
└── tests/
    ├── r_bridge.py      # R bridge for testing (pinned R 4.5.2 + mgcv 1.9-3)
    ├── tolerances.py    # STRICT / MODERATE / LOOSE definitions
    └── conftest.py      # Shared fixtures, R bridge setup
```

### Naming and Style

- Python 3.11+. Type hints on all public functions.
- `snake_case` everywhere. No abbreviations except established ones: `n` (observations), `p` (parameters), `k` (basis dimension), `edf` (effective degrees of freedom).
- Penalty matrices: `S` or `S_j` (per-smooth), `S_lambda` (combined weighted penalty).
- Model matrix: `X` (full), `X_s` (smooth block), `X_p` (parametric block).
- Greek letters in variable names match the design doc: `eta` (linear predictor), `mu` (mean), `beta` (coefficients), `lambda_` (smoothing parameters, trailing underscore to avoid keyword).

### JAX Rules (Phase 2 code)

- All functions that will be JIT-compiled must be pure - no side effects, no Python `if` on array values.
- Use `jax.lax.while_loop` for PIRLS iteration, not Python `while`.
- Use `jax.lax.cond` for conditional logic on array values, not Python `if`.
- Debugging: functions should accept a `debug=False` parameter. When True, use `jax.debug.print()` (not Python print) and `jax.debug.callback()` to log PIRLS traces.
- Float64 everywhere: set `jax.config.update("jax_enable_x64", True)` at module import. GAMs are numerically sensitive - float32 is not acceptable.

### NumPy Rules (Phase 1 and 3 code)

- No JAX imports in Phase 1 modules (`formula/`, `smooths/`, `penalties/`). Use `numpy` and `scipy` only.

### Testing Rules

- Every new module gets a corresponding test file.
- Use the tolerance classes from `tests/tolerances.py`: `STRICT`, `MODERATE`, `LOOSE`.
- R comparison tests use `tests/r_bridge.py` to run the same model in R and compare results.
- **important** R comparisons tests must be identical to R results with `STRICT`, or `MODERATE` tolerance.
- jaxgam results (smooths, bases, coefficients, etc...) **must** be identical to the canonical R mgcv results.
- Hard-gate invariants (§18.1) are tested in every CI run and block the build on failure. These include: objective monotonicity, H symmetry/PSD, penalty PSD, rank conditions, EDF bounds, deviance non-negativity, no NaN in converged model.
- All new modules must have > 80% test coverage.

### Commit and PR Conventions

- One logical change per commit. "Add TPRS basis construction" not "Add smooths".
- Every PR must include tests. No code without tests.
- PR title format: `[phase] component: description` - e.g., `[phase1] smooths/tprs: implement thin plate basis construction`.
- If a change touches Phase 2 code, the PR must include a JIT compilation test (the function compiles and runs without error under `jax.jit`).

## What Is NOT in v1.0

Do not implement any of the following. They are designed for but deferred:

- Sparse-CPU execution path (CHOLMOD, scikit-sparse)
- `bam()`, chunked processing, fREML, Fellner-Schall
- Extended families (NB, Tweedie, Beta, SHASH, Cox PH, ordered categorical)
- P-splines (`ps`, `cp`), B-splines (`bs`)
- Random effects (`bs="re"`), factor-smooth interactions (`bs="fs"`)
- Multi-GPU SPMD, Ray, distributed anything
- `gamm()`, PQL
- Exotic smooths (soap film, MRF, adaptive, Duchon, GP)
- GCV / UBRE smoothness selection (REML and ML only)

If you encounter a code path that would require one of these, stub it with:

```python
raise NotImplementedError(
    "Feature X is planned for v1.1. See docs/design.md Section Y."
)
```

## Common Pitfalls

1. **TPRS eigendecomposition must match R exactly.** The eigenvalue truncation and null space handling determine the basis. Small differences here cascade into large coefficient differences. Validate against R's `smooth.construct.tp.smooth.spec` output, not just final fit results.

2. **Knot placement matters.** mgcv uses a specific algorithm for selecting knots from the data (max-min distance for TPRS, quantile-based for cubic). If your knots don't match R's, nothing else will match either.

3. **CoefficientMap is the contract between Phase 1 and Phase 3.** It maps constrained coefficients (from the fit) back to the original basis for prediction. If this is wrong, `predict()` produces garbage even when the fit is correct. Test the roundtrip: `predict(model, original_data)` must reproduce `model.fitted_values`.

4. **The REML criterion is flat near the optimum.** Smoothing parameters (λ) can differ by 1e-3 from R and the fit is still correct. Don't chase λ precision - validate deviance and coefficients instead.

5. **Step-halving is essential, not optional.** PIRLS without step-halving diverges on Binomial and Gamma models. The step-halving logic (halve step until penalized deviance decreases) must be in the first PIRLS implementation, not added later.

6. **Sum-to-zero constraints interact with factor-by smooths.** When `s(x, by=fac)` coexists with `s(x)`, the constraint absorption (§5.7.3) must be applied correctly or identifiability fails. Test with multi-smooth models from the start.
