# R Source Reference Map

This document maps each JaxGAM implementation task to the specific R mgcv source files and functions that implement the equivalent logic.

## Setup: Local mgcv Clone

Clone the CRAN GitHub mirror and set the `MGCV_SOURCE` environment variable:

```bash
git clone https://github.com/cran/mgcv.git <your-path>/mgcv
export MGCV_SOURCE=<your-path>/mgcv
```

Add the export to your shell profile (`.zshrc`, `.bashrc`, etc.) so it persists. Agents can run `echo $MGCV_SOURCE` to discover the clone location.

## How to Use This File

**When implementing a task**, find it in the table below. Read the listed R functions *before* writing Python using the local clone. The R code is the ground truth for numerical behavior - edge cases, special handling, and algorithmic choices that aren't captured in the design doc.

**Source layout:**
```
$MGCV_SOURCE/R/       # R source files
$MGCV_SOURCE/src/     # C source files
```

For example, to read the smooth constructors:
```
$MGCV_SOURCE/R/smooth.r
```

**When debugging a mismatch with R**, the R function listed here is where to look. Run the R code in an R session with `debug(function_name)` to step through and compare intermediate values.

**Reading R code tips for agents:**
- R uses 1-based indexing. Adjust when porting.
- `<-` is assignment. `<<-` is assignment to parent scope.
- `.C()` and `.Call()` invoke compiled C code in `src/`. The C implementations are the actual hot paths - the R wrappers often just do argument validation.
- `@` accesses S4 slots. `$` accesses list elements or S3 components.
- `mgcv:::` accesses unexported (internal) functions. These are often the ones we need most.

---

## Source File Inventory

```
$MGCV_SOURCE/
├── R/                    # R source files
│   ├── gam.fit.r         # PIRLS, working weights, convergence
│   ├── smooth.r          # All smooth constructors (basis + penalty)
│   ├── fast-REML.r       # REML/ML criterion, Newton optimizer
│   ├── gam.r             # Top-level gam(), predict.gam, summary.gam
│   ├── bam.r             # bam() - NOT needed for v1.0
│   ├── gamm.r            # gamm() - NOT needed for v1.0
│   ├── plots.r           # plot.gam
│   ├── families.r        # Family definitions (not extended)
│   ├── efam.r            # Extended families - NOT needed for v1.0
│   ├── sparse.r          # Sparse matrix routines - NOT needed for v1.0
│   └── mgcv.r            # Package-level utilities
├── src/                  # C source files
│   ├── tprs.c            # TPRS eigendecomposition (the real implementation)
│   ├── mat.c             # Matrix operations (Cholesky, QR, etc.)
│   ├── gdi.c             # Generalized derivative information (REML)
│   ├── discrete.c        # Discretization for bam - NOT needed for v1.0
│   └── mgcv.c            # Package init, misc C utilities
└── man/                  # Documentation (useful for parameter descriptions)
    ├── gam.Rd
    ├── s.Rd
    └── ...
```

---

## Task-to-Source Mapping

### Phase 1: Foundation

#### Task 1.1 - Link Functions

| R file | Function | What it does |
|---|---|---|
| `R/families.r` | `make.link()` | Constructs link with `linkfun`, `linkinv`, `mu.eta`, `valideta` |
| `R/families.r` | Individual family functions (e.g., `binomial()$linkfun`) | Per-family link implementations |

**Key detail:** R's `mu.eta` is dμ/dη (derivative of inverse link), not dη/dμ. Make sure our naming matches.

#### Task 1.2 - Families

| R file | Function | What it does |
|---|---|---|
| `R/families.r` | `gaussian()` | Returns family object with `variance`, `dev.resids`, `aic`, `initialize`, `validmu` |
| `R/families.r` | `binomial()` | Same structure. Note `initialize` handles proportion vs count input |
| `R/families.r` | `poisson()` | Note `dev.resids` uses `2 * wt * (y * log(ifelse(y==0, 1, y/mu)) - (y - mu))` - the `ifelse` for y=0 is critical |
| `R/families.r` | `Gamma()` | Note capital G. `dev.resids` uses `-2 * wt * (log(ifelse(y==0, 1, y/mu)) - (y - mu)/mu)` |

**Key detail:** R's `dev.resids` returns *per-observation* deviance contributions (not squared residuals and not summed). The total deviance is `sum(dev.resids(...))`. Watch for the sign - deviance residuals are `sign(y - mu) * sqrt(abs(dev.resids))`.

**Key detail:** `initialize` expressions in R families use the `eval(family$initialize)` pattern with `mustart` being set in the calling environment. This is R-specific - in Python, just return a starting μ vector.

#### Task 1.3 - Penalty Matrices

| R file | Function | What it does |
|---|---|---|
| `R/smooth.r` | `smooth.construct.*.smooth.spec()` (each smooth type) | Returns `$S` - list of penalty matrices |
| `R/smooth.r` | `smooth2penalty()` | Converts smooth object to penalty representation |

**Key detail:** Each smooth's `$S` is a *list* of penalty matrices (usually length 1, but tensor products have one per marginal). The `$sp` vector in the fitted model gives the corresponding smoothing parameters.

#### Task 1.4 - TPRS Basis and Penalty

| R file | Function | What it does |
|---|---|---|
| `R/smooth.r` | `smooth.construct.tp.smooth.spec()` | **START HERE.** R-level TPRS constructor. Calls C code for the heavy lifting. |
| `R/smooth.r` | `Predict.matrix.tprs.smooth()` | Prediction matrix construction for new data |
| `R/smooth.r` | `null.space.dimension()` | Computes null space dimension for given d and m |
| `src/tprs.c` | `construct_tprs()` | **THE REAL IMPLEMENTATION.** Eigendecomposition, knot selection, basis truncation. Read this carefully - the R wrapper just packages arguments. |
| `src/tprs.c` | `gen_tps_grad()` | TPS semi-kernel evaluation η(r) |
| `src/tprs.c` | `tprs_setup()` | Knot selection via max-min distance |

**Key details:**
- The eigendecomposition is of the **augmented** matrix `[E T; T' 0]` where E is the TPS kernel matrix evaluated at knots and T is the polynomial null space basis. This is not a simple eigendecomposition of E alone.
- `ts` (shrinkage) adds an extra penalty: the eigenvalues corresponding to the null space get a small penalty (1e-4 × max eigenvalue of the wiggly penalty). The exact scaling matters for matching R.
- Knot selection: when n > k, mgcv selects k knots from the data using a greedy max-min distance algorithm (`tprs_setup` in `src/tprs.c`). The knot order affects the basis, so it must match.

#### Task 1.5 - Cubic Regression Splines

| R file | Function | What it does |
|---|---|---|
| `R/smooth.r` | `smooth.construct.cr.smooth.spec()` | Natural cubic regression spline constructor |
| `R/smooth.r` | `smooth.construct.cs.smooth.spec()` | Shrinkage cubic (same as cr + null space penalty) |
| `R/smooth.r` | `smooth.construct.cc.smooth.spec()` | Cyclic cubic |
| `R/smooth.r` | `Predict.matrix.cr.smooth()` | Prediction matrix for cr |
| `R/smooth.r` | `place.knots()` | Quantile-based knot placement |
| `src/mat.c` | `CRpenalty()` | C implementation of cubic spline penalty |

**Key detail:** mgcv's cubic regression splines are NOT B-splines. They use a specific natural spline basis evaluated at the knots directly. The penalty is the integrated squared second derivative, computed analytically from the spline coefficients. `place.knots()` uses quantiles of unique data values, not raw quantiles - this matters when data has ties.

#### Task 1.6 - Tensor Products

| R file | Function | What it does |
|---|---|---|
| `R/smooth.r` | `smooth.construct.tensor.smooth.spec()` | `te()` constructor |
| `R/smooth.r` | `tensor.prod.model.matrix()` | Row-wise Kronecker product of marginal bases |
| `R/smooth.r` | `tensor.prod.penalties()` | Penalty construction from marginal penalties |
| `R/smooth.r` | `smooth.construct.t2.smooth.spec()` | `t2()` - different penalty structure than `te()` |

**Key detail:** `te()` penalties are `I ⊗ S_1`, `S_2 ⊗ I`, etc. - one penalty per marginal, each Kronecker-producted with identity matrices for the other marginals. `ti()` is `te()` with main effects removed via constraints. `t2()` uses a different penalty construction (sum of Kronecker products) - defer `t2` to later if needed.

#### Task 1.7 - Formula Parser

| R file | Function | What it does |
|---|---|---|
| `R/gam.r` | `interpret.gam()` | **THE KEY FUNCTION.** Parses formula, identifies smooth terms, separates parametric from smooth. |
| `R/gam.r` | `interpret.gam0()` | Helper that walks the formula tree |
| `R/smooth.r` | `s()` | Smooth term specification (captures arguments, doesn't build basis) |
| `R/smooth.r` | `te()`, `ti()`, `t2()` | Tensor product term specifications |

**Key detail:** `interpret.gam()` returns a list with `$pf` (parametric formula), `$smooth.spec` (list of smooth specs), and `$response` (response name). Each smooth spec is essentially our `SmoothSpec` - it holds the variable names, basis type, dimension, and options, but doesn't construct the basis yet. That happens later in `gam.setup()`.

#### Task 1.8 - Factor-By Smooths

| R file | Function | What it does |
|---|---|---|
| `R/smooth.r` | `smooth.construct()` (search for `by.var`) | The `by` variable handling in smooth construction |
| `R/gam.r` | `gam.setup()` (search for `by`) | How `by` variables are resolved during model setup |

**Key detail:** When `by` is a factor, mgcv creates one smooth per level and sets `$by.level` on each. The design matrix is block-diagonal - row i has nonzeros only in the block for its factor level. The penalty is duplicated per level (each gets its own λ). When `by` is numeric, the basis matrix is simply multiplied column-wise by the numeric variable.

**Key detail:** mgcv detects factor vs numeric `by` using `is.factor()`. Integers are NOT treated as factors. Character vectors are NOT automatically converted. Only explicit R factors trigger the factor-by path.

#### Task 1.9 - Identifiability Constraints

| R file | Function | What it does |
|---|---|---|
| `R/gam.r` | `gam.side()` | **THE KEY FUNCTION.** Identifies which smooths need sum-to-zero constraints to resolve identifiability. |
| `R/gam.r` | `gam.setup()` (search for `C` matrix) | How constraints are absorbed into the basis |
| `R/smooth.r` | `smooth.construct()` (search for `$C`) | Constraint matrix on each smooth |

**Key detail:** `gam.side()` works by checking whether any smooth's column space overlaps with the intercept or another smooth's column space. When overlap is detected, it imposes a sum-to-zero constraint: `1' X_s β_s = 0`. This is absorbed into the basis via QR decomposition, reducing the column count by 1. The constraint matrix `C` is stored on the smooth object and used to undo the absorption for prediction.

**Key detail:** The order of constraint application matters. `gam.side()` processes smooths in a specific order. If constraints are applied in a different order, the basis differs and results won't match R.

#### Task 1.10 - Design Matrix Assembly

| R file | Function | What it does |
|---|---|---|
| `R/gam.r` | `gam.setup()` | **THE MASTER ASSEMBLY FUNCTION.** Constructs all smooth bases, applies constraints, assembles full X, builds penalty list. |
| `R/gam.r` | `gam()` (first ~100 lines) | Orchestration: formula parsing → gam.setup → fitting |

**Key detail:** `gam.setup()` returns a list with `$X` (full model matrix), `$S` (list of penalty matrices), `$off` (column offsets per smooth in X), `$sp` (initial smoothing parameters), and constraint info. This is the Phase 1 output in our architecture.

---

### Phase 2: Fitting

#### Task 2.3 - PIRLS Inner Loop

| R file | Function | What it does |
|---|---|---|
| `R/gam.fit.r` | `gam.fit3()` | **THE INNER LOOP.** PIRLS for standard families. ~500 lines. |
| `R/gam.fit.r` | `gam.fit5()` | PIRLS for extended families (not needed for v1.0) |
| `src/gdi.c` | `gdi1()`, `gdi2()` | C implementations of the PIRLS numerics (XtWX, etc.) |

**Key details:**
- Search for `step.half` in `gam.fit3()` to find the step-halving logic. It's ~30 lines. The halving limit is `max.half = 15`.
- Convergence criterion: `abs(old.dev - dev) / (0.1 + abs(dev)) < control$epsilon` where `control$epsilon` defaults to `1e-7`.
- Working weights: computed as `w * mu.eta.val^2 / variance` where `w` is prior weights, `mu.eta.val` is dμ/dη, and `variance` is V(μ). This is `W` in our notation.
- Working response: `z <- (eta - offset) + (y - mu) / mu.eta.val`. Note the offset handling.
- Initialization: `eval(family$initialize)` sets `mustart`. Then `eta <- family$linkfun(mustart)`.

#### Task 2.4 - REML Criterion

| R file | Function | What it does |
|---|---|---|
| `R/fast-REML.r` | `Sl.initial.repara()` | Reparameterizes penalties for stable REML computation |
| `R/fast-REML.r` | `Sl.addS()` | Adds penalties together with smoothing parameter weighting |
| `R/fast-REML.r` | `fast.REML.fit()` | Newton optimizer for REML (ALSO contains the criterion computation) |
| `src/gdi.c` | `gdi()` | C implementation of REML derivatives (gradient and Hessian of V w.r.t. log λ) |
| `R/gam.fit.r` | `gam.fit3()` (search for `reml` or `ldetS`) | Where REML is evaluated during fitting |

**Key details:**
- The REML criterion in mgcv is: `V = deviance + log|XtWX + S_λ| - log|S_λ*|` where `S_λ*` is the penalty projected onto its range. The `ldetS` (log determinant of S_λ) computation is subtle - it only includes the non-zero eigenvalues.
- `gdi()` in C computes the derivatives of V w.r.t. `log(sp)` analytically. This is what we replace with `jax.grad`. But reading the analytical derivatives helps validate our AD output.
- `Sl.initial.repara()` reparameterizes the problem so each penalty has a nice form. This is an optimization for the Newton step - we may or may not need it.

#### Task 2.5 - Newton Outer Optimizer

| R file | Function | What it does |
|---|---|---|
| `R/fast-REML.r` | `fast.REML.fit()` | The outer Newton loop. Search for `while` to find the iteration. |
| `R/gam.fit.r` | `gam.fit3()` (search for `while`) | The outer-outer loop that alternates between PIRLS and λ updates |

**Key detail:** mgcv's outer iteration in `gam.fit3` is NOT a clean Newton loop on REML. It interleaves PIRLS convergence checks with smoothing parameter updates. The exact interleaving matters for convergence on difficult models. Read the `while` loop structure in `gam.fit3` carefully - it has PIRLS inner iterations nested inside λ update outer iterations, with a specific convergence protocol.

---

### Phase 3: Post-Estimation

#### Task 3.1 - Prediction

| R file | Function | What it does |
|---|---|---|
| `R/gam.r` | `predict.gam()` | Prediction from fitted GAM |
| `R/smooth.r` | `Predict.matrix.*()` | Per-smooth-type prediction matrix construction |
| `R/gam.r` | `PredictMat()` | Dispatches to the right `Predict.matrix` method |

**Key detail:** `predict.gam()` with `type="lpmatrix"` returns the full prediction matrix (our `X_p`). This is invaluable for debugging: if `X_p` matches, any prediction difference is in the coefficients, not the basis construction.

#### Task 3.2 - Summary and EDF

| R file | Function | What it does |
|---|---|---|
| `R/gam.r` | `summary.gam()` | Summary output including smooth significance tests |
| `R/gam.r` | `pen.edf()` | EDF computation from hat matrix trace |
| `R/smooth.r` | `testStat()` | Wood's (2013) test statistic for smooth significance |

**Key detail:** EDF is computed as `trace(F)` where `F = X (XtWX + S_λ)^{-1} XtW`. Per-smooth EDF is the trace of the corresponding diagonal block. mgcv computes this from the Bayesian covariance: `edf = rowSums(Vp * crossprod(X))` (which is `diag(Vp @ X.T @ X)` but computed without forming the full product).

#### Task 3.3 - Plotting

| R file | Function | What it does |
|---|---|---|
| `R/plots.r` | `plot.gam()` | The main plotting function |
| `R/plots.r` | `plot.mgcv.smooth()` | Per-smooth-type plotting |

**Key detail:** mgcv generates plot data by constructing a prediction grid, getting predictions ± SE, and plotting. For 1D smooths: 200 evenly-spaced points in the covariate range. For 2D: 30×30 grid. The SE bands are ±2*SE (not ±1.96).

---

## Accessing the Source

The mgcv source is cloned locally (see Setup above). Agents should read files directly from `$MGCV_SOURCE`:

```bash
# Discover the clone location
echo $MGCV_SOURCE

# Read a specific R file
$MGCV_SOURCE/R/smooth.r

# Read a C source file
$MGCV_SOURCE/src/tprs.c
```

For large files like `smooth.r` (~8000 lines), search for the specific function name rather than reading the entire file.

**Pinning a version:** The `master` branch tracks the latest CRAN release. To pin to a specific version for reproducibility, checkout a tagged release:
```bash
cd $MGCV_SOURCE && git checkout 1.9-1
```

The CRAN mirror is GPL-2 (Simon Wood). Attribution is required if redistributing; reading for reference is fine.

---

## Quick Lookup: "I'm Debugging X, Where Do I Look?"

| Problem | R file | Function | Search for |
|---|---|---|---|
| Basis matrix doesn't match R | `R/smooth.r` | `smooth.construct.*.smooth.spec()` | The specific basis type |
| Penalty matrix doesn't match | `R/smooth.r` | Same constructor | `$S` |
| Knots don't match | `R/smooth.r` or `src/tprs.c` | `place.knots()` or `tprs_setup()` | Knot selection algorithm |
| Coefficients don't match | `R/gam.fit.r` | `gam.fit3()` | `coef` |
| Deviance doesn't match | `R/families.r` | `*$dev.resids` | Per-family deviance |
| REML score doesn't match | `R/fast-REML.r` | `fast.REML.fit()` | `reml` |
| EDF doesn't match | `R/gam.r` | `pen.edf()` | EDF trace computation |
| Smoothing params don't match | `R/fast-REML.r` | `fast.REML.fit()` | `sp` |
| Prediction doesn't match | `R/gam.r` | `predict.gam()` | Use `type="lpmatrix"` to isolate |
| Step-halving behavior differs | `R/gam.fit.r` | `gam.fit3()` | `step.half` |
| Constraint/identifiability issue | `R/gam.r` | `gam.side()` | Constraint detection logic |
| Factor-by doesn't match | `R/smooth.r` | `smooth.construct()` | `by.var`, `by.level` |
| Summary p-values don't match | `R/gam.r` | `summary.gam()` | `testStat` |
