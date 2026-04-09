# Dense Random Effects - Research Notes

These are raw research notes gathered from the mgcv source and jaxgam codebase.
The synthesized design document is in `design.md`.

## mgcv R Implementation (smooth.r lines 2571-2663)

### smooth.construct.re.smooth.spec

Key behavior:
- Creates model matrix via `model.matrix(~ term1:term2:...:termN - 1)`
  - This produces the interaction of all variables with no intercept
  - For a single factor with L levels: L-column indicator matrix
  - For factor × factor: L1*L2 columns (full interaction)
  - For numeric × factor: L columns of numeric values
- `bs.dim` set to `ncol(X)` (NOT user-specified k)
- Penalty: `S = list(diag(bs.dim))` — identity matrix (ridge penalty)
- Rank = bs.dim (full rank penalty)
- `null.space.dim = 0` (no unpenalized null space)
- `C = matrix(0, 0, ncol(X))` — empty constraint (no centering needed)
- `side.constrain = FALSE` — excluded from gam.side()
- `random = TRUE` — used in summary.gam for p-value test type
- `noterp = TRUE` — skip SVD reparameterization in tensor products
- `te.ok = 2` — can be used as te marginal (but not plotted when so used)
- `plot.me = TRUE` — can be plotted by plot.gam
- `id` not supported (explicitly errors)
- Custom S matrices can be supplied via `xt$S` and `xt$rank`

### Predict.matrix.random.effect

- Uses `model.matrix(object$form, model.frame(..., na.action=na.pass))`
- Sets non-finite values to 0 (handles unseen factor levels gracefully)
- Unseen factor levels → NAs → zeroed out → zero prediction contribution

### Tensor-like RE setup

When `re` inherits from `tensor.smooth.spec`:
- Sets up margins for each variable
- Reorders so largest margin is last
- Class becomes `c("random.effect", "tensor.smooth")` 
- This is used in `bam(..., discrete=TRUE)` — NOT needed for dense v1.0

## Cardinality Limits

mgcv enforces `p <= n` for penalized models (mgcv.r line 236):
```r
if (ncol(M$X) > nrow(M$X)) {
    if (m > 0) stop("Penalized model matrix must have no more columns than rows")
}
```

The soft limit is effectively: sum of all model matrix columns (intercept + 
parametric + all smooth bases) must not exceed n_obs.

For a single-factor RE: max levels = n - (other model columns).
For factor interactions: max product of levels = n - (other model columns).

No explicit per-smooth cardinality check exists — the error comes from the 
full model matrix dimension check at fit time.

## Summary / p-value for RE Terms

In R's summary.gam (mgcv.r lines 3728-3744):
- RE terms have `$random = TRUE`
- They get `type_ = 1` in the testStat call (integer rank rounding)
- Different p-value computation path using `recov()` for Bayesian covariance
- The existing jaxgam `_test_stat()` already supports `type_=1`
- Current code hardcodes `type_=0` — need to pass `type_=1` for RE terms

## jaxgam Codebase Patterns for New Smooth Types

### Registration
1. Create class in `jaxgam/smooths/new_smooth.py`
2. Add entry to `jaxgam/smooths/registry.py` smooth_registry dict
3. Class must inherit from `Smooth` base class

### Required Methods (from Smooth ABC)
- `setup(data)` — construct basis from data
- `build_design_matrix(data)` — return design matrix
- `build_penalty_matrices()` — return list[Penalty]
- `predict_matrix(new_data)` — return prediction matrix for new data

### Key Properties to Set
- `n_coefs` — number of columns in basis
- `null_space_dim` — dimension of unpenalized null space
- `rank` — rank of penalty matrix
- `_is_setup` — flag (set True after setup)
- `side_constrain` — whether gam_side should apply (False for RE)
- `_noterp` — whether to skip SVD reparameterization in tensors
- `_s_scale` — penalty normalization scale factor

### Constraint Pipeline Interaction
- `CoefficientMap.build()` calls `apply_sum_to_zero()` per smooth
- RE terms should skip centering (null_space_dim=0 → no constraint needed,
  but the code path checks for constraint matrix C, not null_space_dim)
- Actually: looking at constraints.py line 371-396, centering is applied
  per-smooth. For RE with null_space_dim=0, the centering would still run
  through apply_sum_to_zero — need to verify this path doesn't break when
  null space is 0.
- Better: set side_constrain=False to skip gam_side, and also ensure no
  centering is applied. The `C = matrix(0,0,k)` in R means "no constraint."

### Formula Parser
- Already handles `bs="re"` (passes through as SmoothSpec.bs="re")
- No special parsing needed — `s(fac, bs="re")` just works
- Variables (positional args) become `SmoothSpec.variables`

### Design Matrix Assembly (design.py)
- `_build_smooth_components()` dispatches via registry
- Registry key selection: `spec.smooth_type if smooth_type in ("te","ti") else spec.bs`
- For `bs="re"`, key = "re" — correct dispatch

### Factor Variable Handling
- RE smooth needs to handle factor variables in its setup()
- Must use pandas Categorical or object dtype detection (like by_variable.py)
- `is_factor()` and `get_factor_levels()` from by_variable.py are available
- Need to store factor levels at setup time for prediction consistency

### Penalty Normalization
- Standard smooths use `_smoothcon_normalize()` to scale penalties
- For RE: R's smoothCon also normalizes the identity penalty
- Need to apply same normalization for R-matching results

## Test Patterns for Smooth Types

### Unit Tests (no R needed)
- Penalty symmetry and PSD
- Basis matrix shape and rank
- predict_matrix reproduces build_design_matrix on same data
- Edge cases (min data, single level, etc.)

### R Comparison Tests
- Decorated with `@pytest.mark.skipif(not r_available(), ...)`
- Use `RBridge().smooth_construct(expr, data)` for basis comparison
- Use `RBridge().fit_gam(formula, data, ...)` for full model comparison
- Compare X, S, rank, null_space_dim
- Tolerances: STRICT for deterministic constructions, MODERATE for fits

### Test Structure
```python
@pytest.mark.skipif(not r_available(), reason="R with mgcv not available")
class TestRComparison:
    def _setup_re(self):
        bridge = RBridge()
        r_result = bridge.smooth_construct("s(fac, bs='re')", data)
        # ... build Python smooth ...
        return smooth, r_result
    
    def test_re_X_values_vs_r(self):
        smooth, r_result = self._setup_re()
        # compare X matrices
    
    def test_re_S_values_vs_r(self):
        # compare penalty matrices
```

### conftest.py Fixtures
- `smooth_1d_data` — numeric data dict
- Need new fixtures for factor data

## What Needs to Change for Dense RE

### New Files
- `jaxgam/smooths/random_effects.py` — RandomEffectSmooth class
- `tests/test_smooths/test_random_effects.py` — tests

### Modified Files
- `jaxgam/smooths/registry.py` — add "re" → RandomEffectSmooth
- `jaxgam/summary/summary.py` — pass type_=1 for RE smooths
- `jaxgam/smooths/constraints.py` — may need to handle skip-centering for RE
- `tests/conftest.py` — add factor data fixtures

### Key Design Decision: Dense vs Sparse
- R's design doc §5.6 shows sparse implementation
- v1.0 is dense-only per CLAUDE.md
- Dense identity penalty = np.eye(n_levels)
- Dense indicator matrix = np.eye-like with row selection
- For large n_levels (>500), dense becomes expensive but still correct
- p <= n constraint naturally limits this
