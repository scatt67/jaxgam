# Dense Random Effects (`bs="re"`) Implementation Design

- **Status:** Implemented
- **Design Date:** 2026-04-08
- **Completion Date:** 2026-05-23
- **Branch:** `add-dense-random-effects`

---

## Table of Contents

1. [Overview](#1-overview)
2. [R Reference Implementation Analysis](#2-r-reference-implementation-analysis)
3. [Mathematical Specification](#3-mathematical-specification)
4. [Cardinality Limits](#4-cardinality-limits)
5. [Smooth Class Design](#5-smooth-class-design)
6. [Formula Parsing](#6-formula-parsing)
7. [Constraint Pipeline Integration](#7-constraint-pipeline-integration)
8. [Penalty Construction](#8-penalty-construction)
9. [Prediction](#9-prediction)
10. [Summary / p-value Changes](#10-summary--p-value-changes)
11. [File Plan](#11-file-plan)
12. [Testing Strategy](#12-testing-strategy)
13. [Implementation Order](#13-implementation-order)

---

## 1. Overview

### 1.1 What Are Dense Random Effects?

In mgcv, `s(x, bs="re")` treats random effects as penalized smooth terms.
The smoothing-penalty duality gives random effects the same mathematical
treatment as splines: coefficients are penalized by an identity matrix (ridge
penalty), and the smoothing parameter λ estimates σ²_ε / σ²_b (the ratio of
residual to random-effect variance).

This is **not** a new estimation method — it exploits the existing PIRLS + 
REML/ML infrastructure. The smooth's penalty matrix is I_k (identity), making
it full rank with `null_space_dim = 0`. No centering or identifiability
constraints are needed because the full-rank penalty makes the term inherently
identifiable.

### 1.2 Use Cases

Random effects with `bs="re"` support any combination of variables:

| Formula | Model matrix | Interpretation |
|---|---|---|
| `s(g, bs="re")` | `model.matrix(~g - 1)` | Random intercepts by group `g` |
| `s(x, g, bs="re")` | `model.matrix(~x:g - 1)` | Random slopes of `x` by group `g` |
| `s(g1, g2, bs="re")` | `model.matrix(~g1:g2 - 1)` | Random interaction of `g1` and `g2` |
| `s(x, bs="re")` | `model.matrix(~x - 1)` | Ridge-penalized numeric effect |

Variables can be any mixture of factors and numerics. The model matrix is
always the interaction `~v1:v2:...:vN - 1`.

### 1.3 Scope

**In scope:**
- `RandomEffectSmooth` class with dense basis and identity penalty
- Registration in smooth registry as `"re"`
- Correct interaction with constraint pipeline (skip centering + gam_side)
- Prediction with unseen factor levels (zero contribution)
- Summary p-value type adjustment (`type_=1` for RE terms)
- Full R comparison tests (basis, penalty, fitted model)

**Out of scope (deferred):**
- Sparse random effects (for high-cardinality factors, deferred to sparse-CPU path)
- `bs="fs"` (factor-smooth interactions — different mechanism, §5.6 design doc)
- Custom precision matrices via `xt$S` argument (future enhancement)
- `gam.vcomp()` variance component extraction (future enhancement)
- Tensor-like RE setup (`re` as `tensor.smooth.spec` — only used in `bam(discrete=TRUE)`)

### 1.4 Dense-Only Constraint

Per CLAUDE.md, v1.0 is dense-only. The basis matrix X for a factor with L
levels is an n × L dense indicator matrix (mostly zeros). This is
memory-inefficient for large L but correct.

The practical limit is the `p ≤ n` constraint (Section 4). Dense RE works well
for L up to a few hundred levels.

---

## 2. R Reference Implementation Analysis

### 2.1 Source Location

The canonical R implementation is in:
- **Constructor:** `$MGCV_SOURCE/R/smooth.r` lines 2571-2646 (`smooth.construct.re.smooth.spec`)
- **Predict:** `$MGCV_SOURCE/R/smooth.r` lines 2650-2663 (`Predict.matrix.random.effect`)
- **Documentation:** `$MGCV_SOURCE/man/smooth.construct.re.smooth.spec.Rd`
- **Examples:** `$MGCV_SOURCE/man/random.effects.Rd`

### 2.2 Constructor Algorithm

```r
smooth.construct.re.smooth.spec <- function(object, data, knots) {
  # 1. Build interaction formula
  form <- as.formula(paste("~", paste(object$term, collapse=":"), "-1"))
  
  # 2. Build model matrix
  object$X <- model.matrix(form, data)
  object$bs.dim <- ncol(object$X)
  
  # 3. Construct penalty (identity by default)
  object$S <- list(diag(object$bs.dim))
  object$rank <- object$bs.dim
  
  # 4. Set key flags
  object$null.space.dim <- 0
  object$C <- matrix(0, 0, ncol(object$X))   # no centering constraint
  object$side.constrain <- FALSE              # skip gam.side()
  object$random <- TRUE                       # p-value test type
  object$noterp <- TRUE                       # skip SVD reparameterization
  object$plot.me <- TRUE
  object$te.ok <- 2
  
  class(object) <- "random.effect"
  object
}
```

Key observations:
1. The `k` argument is **ignored** — `bs.dim` is always `ncol(X)`
2. The penalty is always the identity matrix (full rank, no null space)
3. No centering constraint (`C` is empty)
4. Explicitly opted out of `gam.side()` identifiability checks
5. The `$random` flag signals different p-value computation in summary

### 2.3 Predict Method

```r
Predict.matrix.random.effect <- function(object, data) {
  data <- data[names needed for form]
  X <- model.matrix(object$form, model.frame(object$form, data, na.action=na.pass))
  X[!is.finite(X)] <- 0   # unseen levels → NA → 0
  X
}
```

Key: unseen factor levels produce NAs in the model matrix, which are then
zeroed out. This means predictions for new factor levels contribute zero.
R's `predict.gam()` sets factor observations to NA if they have levels not
present in the training data.

### 2.4 smoothCon() Integration

When `smoothCon()` processes an RE smooth:
- The empty `C` matrix means no centering absorption occurs
- `scale.penalty=TRUE` still applies penalty normalization (divide by `maXX`)
- The smooth is returned as-is except for normalization

---

## 3. Mathematical Specification

### 3.1 Model Matrix Construction

For variables `v1, v2, ..., vp`, the model matrix X is equivalent to
`model.matrix(~v1:v2:...:vp - 1)`:

**Single factor** `g` with L levels:
```
X[i, j] = 1  if g[i] == level_j
X[i, j] = 0  otherwise
```
Shape: (n, L). This is a one-hot indicator matrix.

**Factor × factor** `g1` (L1 levels) × `g2` (L2 levels):
```
X[i, (j1-1)*L2 + j2] = 1  if g1[i] == level_j1 AND g2[i] == level_j2
```
Shape: (n, L1 × L2). This is a one-hot indicator for each combination.

**Numeric × factor** `x` (numeric) × `g` (L levels):
```
X[i, j] = x[i]  if g[i] == level_j
X[i, j] = 0     otherwise
```
Shape: (n, L). Each column holds the numeric value for observations at that level.

**Numeric × numeric** `x1` × `x2`:
```
X[i, 1] = x1[i] * x2[i]
```
Shape: (n, 1). This is just the elementwise product (rarely used in practice).

### 3.2 Penalty Matrix

```
S = I_k    (k × k identity matrix)
```

This is the ridge penalty. It penalizes all coefficients equally with no
unpenalized null space (null_space_dim = 0, rank = k).

The smoothing parameter λ acts as:
```
σ²_b = σ²_ε / λ
```

where σ²_b is the random effect variance and σ²_ε is the residual variance.
Large λ → small random effect variance → coefficients shrunk toward zero.

### 3.3 Normalization

Like all smooths, the penalty is normalized by `smoothCon()`:
```
s_scale = ||S||_1 / ||X||_∞²
S_normalized = S / s_scale
```

For the identity penalty, `||S||_1 = 1`, so:
```
s_scale = 1.0 / ||X||_∞²
```

---

## 4. Cardinality Limits

### 4.1 mgcv Enforcement

mgcv enforces `p ≤ n` for penalized models (`mgcv.r` line 236):

```r
if (ncol(M$X) > nrow(M$X)) {
    if (m > 0) stop("Penalized model matrix must have no more columns than rows")
}
```

This is a **full-model** check, not per-smooth. The total number of columns
across intercept + parametric + all smooth bases must not exceed n.

### 4.2 Practical Limits

For a model `y ~ s(g, bs="re") + s(x)`:
- Intercept: 1 column
- `s(x)`: ~10 columns (default k)
- `s(g, bs="re")`: L columns (one per factor level)
- Constraint: L + 11 ≤ n, so max L ≈ n - 11

For factor interactions `s(g1, g2, bs="re")`:
- Columns = L1 × L2
- Max product of levels ≈ n - (other model columns)

### 4.3 No Explicit Per-Smooth Cardinality Check

There is no per-smooth cardinality check in mgcv. The constraint emerges
naturally from the full model matrix dimension check at fit time.

**Our implementation:** We should validate `p ≤ n` in the fitting layer
(where it's already checked) rather than adding a smooth-level limit.
However, we should add a helpful error message when an RE smooth creates
a model matrix wider than the data.

---

## 5. Smooth Class Design

### 5.1 Class: `RandomEffectSmooth`

```python
# jaxgam/smooths/random_effects.py

class RandomEffectSmooth(Smooth):
    """Dense random effects smooth (bs="re").
    
    For s(v1, v2, ..., bs="re"), constructs the model matrix equivalent
    to model.matrix(~v1:v2:...:vN - 1) with an identity penalty.
    """
    
    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)
        self.side_constrain = False   # skip gam.side()
        self._noterp = True           # skip SVD reparameterization
        self._random = True           # RE flag for summary p-values
        self._has_centering_constraint = False  # skip sum-to-zero
        
        # Stored at setup time
        self._levels: dict[str, list] | None = None  # var_name → ordered levels
        self._is_factor: dict[str, bool] | None = None
        self._X: np.ndarray | None = None
        self._S: np.ndarray | None = None
```

### 5.2 `setup()` Method

```python
def setup(self, data: dict[str, np.ndarray]) -> None:
    """Construct RE basis from data.
    
    1. Determine which variables are factors, store levels
    2. Build interaction model matrix (~v1:v2:...:vN - 1)
    3. Set penalty = normalized identity
    """
    variables = self.spec.variables
    
    # Detect factors and store levels
    self._is_factor = {}
    self._levels = {}
    for var in variables:
        col = data[var]
        if is_factor(col):
            self._is_factor[var] = True
            self._levels[var] = get_factor_levels(col)
        else:
            self._is_factor[var] = False
    
    # Build interaction model matrix
    X = self._build_interaction_matrix(data)
    
    self._X = X
    k = X.shape[1]
    self.n_coefs = k
    self.null_space_dim = 0
    self.rank = k
    
    # Penalty = identity, then normalize
    S = np.eye(k)
    [S], self._s_scale = self._smoothcon_normalize(X, [S])
    self._S = S
    
    self._is_setup = True
```

### 5.3 Model Matrix Construction

The interaction model matrix must replicate R's `model.matrix(~v1:v2:...:vN - 1)`:

**Single variable:**
- Factor: one-hot indicator matrix (n × L)
- Numeric: column vector (n × 1)

**Multiple variables (interaction):**
- Kronecker-style row-wise product of individual variable encodings
- For each observation: the row of the interaction matrix is the outer product
  of all individual variable rows, flattened

Implementation approach:
```python
def _build_interaction_matrix(self, data):
    """Build ~v1:v2:...:vN - 1 interaction matrix."""
    n = len(data[self.spec.variables[0]])
    
    # Start with column of ones
    result = np.ones((n, 1))
    
    for var in self.spec.variables:
        col = data[var]
        if self._is_factor[var]:
            # One-hot encode
            levels = self._levels[var]
            indicator = np.zeros((n, len(levels)))
            col_arr = np.asarray(col)
            for j, lev in enumerate(levels):
                indicator[:, j] = (col_arr == lev).astype(float)
            term = indicator
        else:
            term = np.asarray(col, dtype=float).reshape(n, 1)
        
        # Interaction: row-wise Kronecker product
        new_cols = result.shape[1] * term.shape[1]
        new_result = np.zeros((n, new_cols))
        for i in range(result.shape[1]):
            for j in range(term.shape[1]):
                new_result[:, i * term.shape[1] + j] = result[:, i] * term[:, j]
        result = new_result
    
    return result
```

### 5.4 Column Ordering

The column ordering must match R's `model.matrix()` for R comparison tests.
R's interaction ordering is: first variable varies fastest (leftmost term in
the interaction formula changes first).

For `s(g1, g2, bs="re")` with g1 levels [a,b] and g2 levels [x,y,z]:
- R columns: a:x, b:x, a:y, b:y, a:z, b:z
- First variable (g1) varies fastest, last (g2) varies slowest

Verified empirically against R 4.5.2 / mgcv 1.9-3 via `smoothCon(s(g1, g2,
bs="re"), data)`. The row-wise Kronecker implementation uses the new term's
index as the outer (slow) index to match this convention.

### 5.5 `build_design_matrix()` and `predict_matrix()`

```python
def build_design_matrix(self, data):
    self._require_setup()
    return self.predict_matrix(data)

def predict_matrix(self, new_data):
    self._require_setup()
    X = self._build_interaction_matrix(new_data)
    # Zero out rows with unseen factor levels (NaN → 0)
    X[~np.isfinite(X)] = 0.0
    return X
```

For prediction with new data:
- Factor levels not seen during training produce NaN in the indicator encoding
- These NaNs are zeroed out, matching R's behavior
- This means new factor levels contribute zero to predictions

### 5.6 `build_penalty_matrices()`

```python
def build_penalty_matrices(self):
    self._require_setup()
    return [Penalty(self._S, rank=self.rank, null_space_dim=0)]
```

---

## 6. Formula Parsing

### 6.1 No Parser Changes Required

The existing formula parser already handles `s(fac, bs="re")` correctly:
- Positional args become `SmoothSpec.variables = ["fac"]`
- `bs="re"` is captured in `SmoothSpec.bs = "re"`
- Multiple variables: `s(g1, g2, bs="re")` → `variables = ["g1", "g2"]`

### 6.2 k Argument

In R, the `k` argument is ignored for `bs="re"` — the basis dimension is
always `ncol(X)`. Our implementation should do the same: ignore `spec.k`
and set `n_coefs = ncol(X)`.

---

## 7. Constraint Pipeline Integration

### 7.1 Skip Centering Constraint

RE smooths must **not** have the sum-to-zero centering constraint applied.
In R, this is signaled by `C = matrix(0, 0, ncol(X))` (empty constraint).

In jaxgam, the centering pipeline in `CoefficientMap.build()` (constraints.py
lines 369-396) applies `apply_sum_to_zero()` to all smooths unless explicitly
skipped.

**Required change:** Add a check for `_has_centering_constraint`:

```python
# In CoefficientMap.build(), centering loop:
for i, sm in enumerate(smooths):
    # Skip RE smooths (no centering needed)
    if not getattr(sm, '_has_centering_constraint', True):
        continue
    
    # ... existing centering logic ...
```

The `RandomEffectSmooth` sets `_has_centering_constraint = False` in `__init__`.

### 7.2 Skip gam.side()

RE smooths set `side_constrain = False`. The existing `gam_side()` 
implementation already checks this flag (constraints.py lines 729, 753):

```python
if dim_i == d and CoefficientMap._smooth_side_constrain(sm):
```

And `_smooth_side_constrain()` (line 991-1008) correctly reads the 
`side_constrain` attribute. No changes needed here.

### 7.3 CoefficientMap Term Type

The `TermBlock` for an RE smooth should still have `term_type = "smooth"`.
The `_random` flag on the smooth object (not on TermBlock) is used for
p-value dispatch.

---

## 8. Penalty Construction

### 8.1 Identity Penalty

The penalty for `bs="re"` is always the identity matrix of dimension k
(where k = `ncol(X)`). After smoothCon normalization:

```
S_normalized = I_k / s_scale
```

where `s_scale = ||I_k||_1 / ||X||_∞² = 1 / ||X||_∞²`.

### 8.2 Embedding in Global Penalty Space

The existing `CompositePenalty.embed()` handles this correctly — it places
the k×k penalty into the correct block of the global (total_p × total_p) 
penalty matrix. No changes needed.

### 8.3 Properties

- **Rank:** k (full rank)
- **Null space dimension:** 0
- **Number of penalties:** 1
- **Smoothing parameter:** 1 (one λ for the term)

---

## 9. Prediction

### 9.1 Training Data

`predict_matrix(training_data)` should reproduce `build_design_matrix()`.
Since the construction is deterministic (no eigendecomposition), this is
straightforward.

### 9.2 New Data with Known Levels

For factor levels present in training: prediction works normally via the
indicator matrix.

### 9.3 New Data with Unseen Levels

For factor levels NOT present in training:
1. The indicator encoding produces NaN for unseen levels
2. NaNs are zeroed out in the prediction matrix
3. The unseen level contributes zero to the linear predictor
4. This matches R's behavior exactly

Implementation: in `predict_matrix()`, after building the interaction matrix,
zero out all non-finite values:
```python
X[~np.isfinite(X)] = 0.0
```

### 9.4 Level Storage for Prediction

Factor levels must be stored at `setup()` time and reused in
`predict_matrix()`. The stored levels define which columns exist in the basis.
New data is encoded against these stored levels.

---

## 10. Summary / p-value Changes

### 10.1 R Behavior

In R's `summary.gam()`, RE terms (identified by `$random == TRUE`) receive
a different p-value test:
- `type_ = 1` in `testStat()` (integer rank rounding instead of fractional)
- The test statistic interpretation changes for random effects

### 10.2 Required Change

In `jaxgam/summary/summary.py`, the smooth p-value loop (line 213-220)
currently hardcodes `type_=0`:

```python
res = _test_stat(p_i, X_i, V_i, rank=..., type_=0, res_df=rdf)
```

This needs to check whether the smooth is a random effect:

```python
# Determine test type: RE terms use type_=1 (integer rank)
is_re = getattr(si_smooth, '_random', False)
test_type = 1 if is_re else 0

res = _test_stat(p_i, X_i, V_i, rank=..., type_=test_type, res_df=rdf)
```

The `SmoothInfo` dataclass may need an additional field, or we access the
smooth object through the `coef_map`. The cleanest approach is to add an
`is_random` field to `SmoothInfo`.

### 10.3 SmoothInfo Extension

Add `is_random: bool = False` to `SmoothInfo` (design.py):

```python
@dataclass(frozen=True)
class SmoothInfo:
    # ... existing fields ...
    is_random: bool = False  # True for bs="re" terms
```

Populate from the smooth object in `_build_smooth_info()`:

```python
is_random = getattr(getattr(sm, 'base_smooth', sm), '_random', False)
```

---

## 11. File Plan

### 11.1 New Files

| File | Description |
|---|---|
| `jaxgam/smooths/random_effects.py` | `RandomEffectSmooth` class |
| `tests/test_smooths/test_random_effects.py` | Unit + R comparison tests |

### 11.2 Modified Files

| File | Change |
|---|---|
| `jaxgam/smooths/registry.py` | Add `"re": RandomEffectSmooth` |
| `jaxgam/smooths/constraints.py` | Skip centering for `_has_centering_constraint=False` |
| `jaxgam/formula/design.py` | Add `is_random` to `SmoothInfo`; populate in `_build_smooth_info()` |
| `jaxgam/summary/summary.py` | Pass `type_=1` for RE terms |
| `tests/conftest.py` | Add factor data fixtures |

### 11.3 No Changes Needed

| File | Reason |
|---|---|
| `jaxgam/formula/parser.py` | Already handles `bs="re"` correctly |
| `jaxgam/penalties/penalty.py` | Identity penalty is just `Penalty(np.eye(k))` |
| `jaxgam/fitting/*` | RE smooths are transparent to PIRLS/REML |
| `jaxgam/smooths/base.py` | Base class interface is sufficient |

---

## 12. Testing Strategy

Testing is split across two locations following the existing project pattern:

- **`tests/test_smooths/test_random_effects.py`** — basis-level unit tests
  and smooth-construct R comparisons (analogous to `test_cubic.py`,
  `test_tprs.py`). These test the `RandomEffectSmooth` class in isolation.

- **`tests/test_validation_matrix.py`** — full-model R comparisons across
  all smooth × family combinations. RE smooth configs are added as new cells
  in the existing parametrized matrix, inheriting all the existing test
  methods (deviance, fitted values, EDF, scale, coefficients, self-prediction
  roundtrip, hard-gate invariants).

### 12.1 Validation Matrix Integration

The validation matrix (`test_validation_matrix.py`) systematically tests every
smooth config × family cell. RE adds **3 new smooth configs** and a
**new data generator**, expanding the matrix from 35 cells to 50 cells.

#### 12.1.1 New Smooth Configs

```python
# Added to SMOOTH_CONFIGS in test_validation_matrix.py

"re": SmoothConfig(
    py_formula="y ~ s(g, bs='re')",
    r_formula="y ~ s(g, bs='re')",
    data_type="re",
),
"re_slope": SmoothConfig(
    py_formula="y ~ s(x, g, bs='re')",
    r_formula="y ~ s(x, g, bs='re')",
    data_type="re",
),
"re_mixed": SmoothConfig(
    py_formula="y ~ s(x, k=10, bs='tp') + s(g, bs='re')",
    r_formula="y ~ s(x, k=10, bs='tp') + s(g, bs='re')",
    data_type="re",
),
```

These cover the three most important RE patterns:
- `re` — random intercepts (RE-only model)
- `re_slope` — random slopes (numeric × factor interaction)
- `re_mixed` — RE alongside a standard smooth (most common real-world usage)

#### 12.1.2 New Data Generator

```python
def _make_re_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Random effects data: continuous x + factor g with group-level effects."""
    rng = np.random.default_rng(seed)
    n = 300
    n_groups = 20
    x = rng.uniform(0, 1, n)
    g = rng.choice([f"g{i}" for i in range(n_groups)], size=n)

    # True group effects
    b_intercept = rng.normal(0, 1.0, n_groups)
    group_idx = {f"g{i}": i for i in range(n_groups)}
    group_effect = np.array([b_intercept[group_idx[gi]] for gi in g])

    # Smooth + RE truth
    eta = np.sin(2 * np.pi * x) + group_effect

    if family_name == "gaussian":
        y = eta + rng.normal(0, 0.5, n)
    elif family_name == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        y = rng.poisson(np.exp(eta * 0.5 + 0.5)).astype(float)
    elif family_name == "gamma":
        mu = np.exp(eta * 0.3 + 1.0)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        mu = np.exp(eta * 0.5 + 0.5)
        theta = 2.0
        y = rng.negative_binomial(
            n=theta, p=theta / (mu + theta), size=n
        ).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({
        "x": x,
        "g": pd.Categorical(g),
        "y": y,
    })
```

Wire into `_get_data()`:
```python
if config.data_type == "re":
    return _make_re_data(family)
```

#### 12.1.3 Tolerance Rules

```python
# In _r_tol(): RE models use MODERATE for Gaussian, LOOSE otherwise
# (same rule as single-smooth models — RE has one sp, deterministic basis)
if family_name == "gaussian" and smooth_key in ("tp", "cr", "re"):
    return MODERATE
return LOOSE
```

```python
# In _compare_fitted_not_coefs(): RE coefficients are directly comparable
# (no sign ambiguity, no flat REML surface with multiple sp)
# re and re_slope: compare coefficients directly
# re_mixed: compare fitted values (contains TPRS with sign ambiguity)
return smooth_key in ("tp", "tp_by", "te", "ti", "te_by", "cr_by", "re_mixed")
```

#### 12.1.4 Inherited Tests

Adding RE configs to `SMOOTH_CONFIGS` automatically inherits all existing
test methods in both `TestValidationMatrix` and `TestHardGateInvariants`:

**TestValidationMatrix (R comparison, 15 new cells):**
- `test_deviance_vs_r` — deviance matches R
- `test_fitted_values_vs_r` — fitted values match R
- `test_edf_vs_r` — total EDF matches R
- `test_scale_vs_r` — scale parameter matches R
- `test_coefficients_vs_r` — coefficients or fitted values match R
- `test_self_prediction_roundtrip` — `predict()` reproduces `fitted_values`
- `test_theta_vs_r` — theta matches R (NB family only)

**TestHardGateInvariants (no R required, 15 new cells):**
- `test_convergence` — model converges
- `test_deviance_non_negative` — deviance ≥ 0
- `test_no_nan_in_converged` — no NaN/Inf in coefficients, fitted values
- `test_edf_bounds` — EDF in valid range
- `test_vp_symmetric_psd` — Bayesian covariance is symmetric PSD
- `test_penalty_psd` — penalty matrices are symmetric PSD
- `test_theta_positive` — theta > 0 (NB only)
- `test_model_matrix_rank` — model matrix has sufficient rank

### 12.2 Unit Tests (`test_random_effects.py`)

These test `RandomEffectSmooth` in isolation, without the full GAM fitting
pipeline. They live in `tests/test_smooths/test_random_effects.py`, following
the pattern of `test_cubic.py` and `test_tprs.py`.

**Structural tests (no R):**
- Identity penalty is symmetric PSD with correct rank
- Basis matrix shape matches number of factor levels
- `null_space_dim == 0` and `rank == n_coefs`
- `side_constrain == False` and `_has_centering_constraint == False`
- `_random == True`
- `predict_matrix` reproduces `build_design_matrix` on training data

**Factor handling (no R):**
- Single factor: correct indicator matrix
- Multiple factors: correct interaction columns and ordering
- Numeric × factor: correct weighted indicator
- Numeric only: correct single-column matrix
- Factor with many levels (stress test)

**Prediction edge cases (no R):**
- Unseen factor levels → zero rows in prediction matrix
- Subset of training levels → correct subset of columns

**R basis comparison (requires R + mgcv):**
- `s(g, bs="re")` — single factor basis X and penalty S vs R
- `s(g1, g2, bs="re")` — factor × factor interaction vs R
- `s(x, g, bs="re")` — numeric × factor vs R
- Compare X, S, rank, null_space_dim at STRICT tolerance (deterministic)

### 12.3 Tolerance Summary

| What | Location | Tolerance | Rationale |
|---|---|---|---|
| Basis matrix X | `test_random_effects.py` | STRICT | Deterministic, no eigendecomposition |
| Penalty matrix S | `test_random_effects.py` | STRICT | Identity matrix — exact after normalization |
| Deviance | validation matrix | MODERATE/LOOSE | Standard fitting tolerance |
| Fitted values | validation matrix | MODERATE/LOOSE | Standard fitting tolerance |
| Coefficients (re, re_slope) | validation matrix | MODERATE | Single sp, direct comparison |
| Fitted values (re_mixed) | validation matrix | MODERATE/LOOSE | Contains TPRS sign ambiguity |
| EDF | validation matrix | MODERATE/LOOSE | Derived from fitting |
| Hard-gate invariants | validation matrix | per-invariant | Structural, must always hold |

### 12.4 Test Data Fixtures

For `test_random_effects.py`, add fixtures to `tests/conftest.py`:

```python
@pytest.fixture
def re_factor_data():
    """Single-factor data for RE smooth tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    n_groups = 20
    g = rng.choice([f"g{i}" for i in range(n_groups)], size=n)
    return {
        "g": pd.Categorical(g),
    }

@pytest.fixture
def re_two_factor_data():
    """Two-factor data for RE interaction tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    g1 = rng.choice(["a", "b", "c"], size=n)
    g2 = rng.choice(["x", "y"], size=n)
    return {
        "g1": pd.Categorical(g1),
        "g2": pd.Categorical(g2),
    }

@pytest.fixture
def re_numeric_factor_data():
    """Numeric × factor data for random slope tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    x = rng.uniform(0, 1, n)
    g = rng.choice([f"g{i}" for i in range(10)], size=n)
    return {
        "x": x,
        "g": pd.Categorical(g),
    }
```

---

## 13. Implementation Plan

See [implementation_plan.md](implementation_plan.md) for the detailed
PR-by-PR breakdown.
