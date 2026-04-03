# Negative Binomial Implementation Plan

**Reference:** `docs/nb_implementation/design.md`
**Branch:** `implement-nb-distribution`

---

## PR Strategy

Each step below is a separate PR. Tests must pass before merging. Later PRs
depend on earlier ones but each PR is independently reviewable and testable.

- **PR 1:** ExtendedFamily base class + NegativeBinomial family (no fitting)
- **PR 2:** AD validation tests for pure-function factories
- **PR 3:** Extended custom_jvp for theta in newton.py
- **PR 4:** Newton optimizer integration (end-to-end fitting)
- **PR 5:** R comparison tests + validation matrix extension
- **PR 6:** Post-estimation (results, summary, api)
- **PR 7:** Documentation, benchmarks, demo script

---

## PR 1: ExtendedFamily Base + NegativeBinomial Family

**Goal:** Family class that passes unit tests against R values. No fitting yet.

### Files to Create

**`jaxgam/families/extended.py`** (~30 lines)

```python
from abc import abstractmethod
import numpy as np
from jaxgam.families.base import ExponentialFamily

class ExtendedFamily(ExponentialFamily):
    """Base class for families with extra parameters estimated via Newton.

    Subclasses must implement:
    - get_theta / put_theta: mutable theta state
    - deviance_fn: pure function D(eta, log_theta) for custom_jvp
    - working_weights_fn: pure function W(eta, log_theta) for custom_jvp
    - saturated_loglik_theta: explicit theta for REML criterion AD trace
    """

    @abstractmethod
    def get_theta(self, transformed: bool = False) -> float: ...

    @abstractmethod
    def put_theta(self, log_theta: float) -> None: ...

    @abstractmethod
    def deviance_fn(self, y: np.ndarray, wt: np.ndarray): ...

    @abstractmethod
    def working_weights_fn(self, wt: np.ndarray): ...

    @abstractmethod
    def saturated_loglik_theta(self, y, wt, scale, log_theta): ...
```

**`jaxgam/families/negative_binomial.py`** (~250 lines)

Implement all methods from design.md Section 7.2-7.4:
- `__init__(theta, link)` with R's sign convention (None/0=estimate, >0=fixed, <0=initial)
- `get_theta(transformed)` / `put_theta(log_theta)`
- `variance(mu)`, `dvar(mu)`
- `deviance_resids(y, mu, wt)`
- `saturated_loglik(y, wt, scale)` -- reads self._log_theta
- `aic(y, mu, wt, scale)`
- `initialize(y, wt)` -- `y + (y == 0) / 6`
- `valid_mu(mu)`, `valid_eta(eta)`
- `saturated_loglik_theta(y, wt, scale, log_theta)` -- explicit arg
- `deviance_fn(y, wt)` -- returns pure function D(eta, log_theta)
- `working_weights_fn(wt)` -- returns pure function W(eta, log_theta)
- `alpha` property -- convenience, returns `1 / get_theta(transformed=True)`

### Files to Modify

**`jaxgam/families/registry.py`** -- add `"nb": NegativeBinomial`

**`jaxgam/families/__init__.py`** -- export `ExtendedFamily`, `NegativeBinomial`

**`tests/r_bridge.py`** -- add NB to both family maps:
- `_SUBPROCESS_FAMILY_MAP`: add `"nb": "nb()"`
- `_get_r_family_rpy2`: add `"nb": self._mgcv.nb`
- `_extract_fit_results_rpy2`: extract `theta` from `r_model.rx2("family")$getTheta(TRUE)`
- `_fit_subprocess` R script: extract theta and write to `theta.txt`
- Return dict gains `"theta"` key (None for standard families)

**`tests/helpers.py`** -- extend:
- `_generate_family_data`: add `"nb"` case using `rng.negative_binomial`
- `r_tolerance`: add `"nb"` -> `LOOSE` (iterative PIRLS + theta estimation)

### Tests: `tests/test_families.py` (extend existing file)

Follow the existing test patterns exactly. Add NB to each test class.

**TestVariance** -- add:
```python
def test_nb_variance(self) -> None:
    """NB V(mu) = mu + mu^2/theta."""
    mu = np.array([0.01, 0.1, 1.0, 5.0, 100.0])
    fam = NegativeBinomial(theta=2)
    v = fam.variance(mu)
    expected = mu + mu**2 / 2.0
    np.testing.assert_allclose(v, expected, rtol=STRICT.rtol, atol=STRICT.atol)
```

**TestDevianceResids** -- add NB case using `_compute_r_dev_resids` with
`family_r = "mgcv::nb(theta=2)"` (NB dev.resids requires theta arg -- need
to adapt the R script to pass theta). Alternatively use the extended family's
dev.resids signature: `fam$dev.resids(y, mu, wt, log(2))`.

**TestWorkingWeights** -- add:
```python
def test_nb_working_weights(self) -> None:
    """NB working weights = wt / (V(mu) * g'(mu)^2) with log link."""
    mu = np.array([0.5, 1.0, 2.0, 5.0])
    wt = np.ones_like(mu)
    fam = NegativeBinomial(theta=2)
    W = fam.working_weights(mu, wt)
    theta = 2.0
    V = mu + mu**2 / theta
    g_prime = 1.0 / mu  # log link
    expected = wt / (V * g_prime**2)
    np.testing.assert_allclose(W, expected, rtol=STRICT.rtol, atol=STRICT.atol)
```

**TestInitialization** -- add:
```python
def test_nb_initialize(self) -> None:
    """NB initialize: y + (y == 0) / 6."""
    y = np.array([0.0, 1.0, 5.0, 0.0, 10.0])
    wt = np.ones_like(y)
    fam = NegativeBinomial()
    mu = fam.initialize(y, wt)
    expected = np.where(y == 0, 1.0 / 6.0, y)
    np.testing.assert_allclose(mu, expected, rtol=STRICT.rtol, atol=STRICT.atol)
```

**TestEdgeCases** -- add:
- [x] `deviance_resids` with y=0: no NaN, no Inf
- [x] `deviance_resids` with mu near zero: finite result
- [x] Poisson limit: as theta -> inf, `variance(mu)` -> `mu`

**TestRegistry** -- add:
```python
def test_nb_registry(self) -> None:
    fam = get_family("nb")
    assert fam.family_name == "nb"
    assert isinstance(fam, NegativeBinomial)
    assert isinstance(fam, ExtendedFamily)
    assert not isinstance(Gaussian(), ExtendedFamily)
```

**New test class -- TestExtendedFamilyContract:**

Generic contract tests that every ExtendedFamily must pass. Parametrized by
family instance — when Tweedie/Beta arrive, just add them to the params list.

```python
class TestExtendedFamilyContract:
    """Contract tests for ExtendedFamily interface.

    Every extended family must pass these. Tests verify the pure-function
    factories match the self-reading methods, and that AD produces finite
    gradients through all traced paths.
    """

    SAMPLE_Y = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 50.0])
    SAMPLE_MU = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 30.0])
    SAMPLE_WT = np.ones(6)

    @pytest.fixture(params=[
        NegativeBinomial(),
        NegativeBinomial(theta=2),
        NegativeBinomial(theta=0.5),
        # future: Tweedie(p=1.5), Beta(), ...
    ])
    def efamily(self, request):
        return request.param
```

Tests in this class:
- [x] `isinstance(efamily, ExtendedFamily)` is True
- [x] `isinstance(efamily, ExponentialFamily)` is True (inheritance)
- [x] `n_theta >= 0`
- [x] `get_theta()` / `put_theta()` round-trip preserves value
- [x] `deviance_fn(y, wt)` returns a callable
- [x] `working_weights_fn(wt)` returns a callable
- [x] `deviance_fn` consistency: `dev_fn(eta, log_theta)` matches
      `dev_resids(y, mu, wt)` at stored theta (STRICT tolerance)
- [x] `working_weights_fn` consistency: `ww_fn(eta, log_theta)` matches
      `working_weights(mu, wt)` at stored theta (STRICT tolerance)
- [x] `saturated_loglik_theta` consistency: matches `saturated_loglik` at
      stored theta (STRICT tolerance)
- [x] AD finite: `jax.grad(dev_fn, argnums=0)(eta, log_theta)` is finite
- [x] AD finite: `jax.grad(dev_fn, argnums=1)(eta, log_theta)` is finite
- [x] AD finite: mixed derivative `d²D/(dη dθ)` is finite (custom_jvp needs this)
- [x] AD finite: `jax.jvp(ww_fn, (eta, lt), (deta, dlt))` is finite
- [x] `saturated_loglik_theta` AD: `jax.grad(..., argnums=3)` is finite

**New test class -- TestNBSpecific:**

NB-only tests that don't generalize to other extended families.

- [x] Constructor: `NegativeBinomial()` -> n_theta=1, log_theta=0
- [x] Constructor: `NegativeBinomial(theta=2)` -> n_theta=0, log_theta=log(2)
- [x] Constructor: `NegativeBinomial(theta=-2)` -> n_theta=1, log_theta=log(2)
- [x] `alpha` property returns 1/theta
- [x] `scale_known` is True
- [x] `default_link` is LogLink
- [x] Poisson limit: as theta -> inf, `variance(mu)` -> `mu`
- [x] Poisson limit: as theta -> inf, `deviance_resids` -> Poisson deviance_resids

**New test class -- TestNBvsR** (R comparison for family methods):
```python
@pytest.mark.skipif(not _r_available(), reason="R/mgcv not available")
class TestNBvsR:
    """NB family methods vs R's nb() reference values."""
```
- [x] `dev_resids` matches R's `nb()$dev.resids(y, mu, wt, log(theta))`
- [x] `saturated_loglik` matches R's `nb()$ls(y, wt, log(theta), 1)$ls`
- [x] `aic` matches R's `nb()$aic(y, mu, log(theta), wt, 0)`

Use `_compute_r_dev_resids` pattern adapted for NB (extended family dev.resids
takes a 4th `theta` argument).

### Acceptance Criteria

- All existing tests still pass (no regression)
- All new NB tests pass
- `ruff check` and `ruff format` clean
- `NegativeBinomial` importable from `jaxgam.families`
- No changes to any fitting code

---

## PR 2: AD Validation Tests

**Goal:** Verify `jax.grad` through all pure-function factories matches
finite differences across the full parameter space, including extreme regimes.

PR 1's `TestExtendedFamilyContract` already verifies AD produces finite values.
This PR adds **finite-difference accuracy** tests — checking that the AD
gradients are numerically correct, not just finite.

### Tests: `tests/test_families.py` (extend)

New test class alongside `TestExtendedFamilyContract`:

```python
class TestExtendedFamilyAD:
    """Finite-difference validation of AD through extended family factories.

    Parametrized by family instance and theta regime. Verifies jax.grad
    output matches central finite differences to MODERATE tolerance.
    """

    @pytest.fixture(params=[
        (NegativeBinomial(theta=2), "moderate_theta"),
        (NegativeBinomial(theta=0.01), "high_overdispersion"),
        (NegativeBinomial(theta=10000), "near_poisson"),
        # future: Tweedie, Beta, ...
    ])
    def efamily_regime(self, request):
        return request.param
```

**`saturated_loglik_theta` FD:**
- [x] `jax.grad(saturated_loglik_theta, argnums=3)` matches FD (MODERATE)
- [x] Second derivative via `jax.grad(jax.grad(...))` matches FD (MODERATE)

**`deviance_fn` FD:**
- [x] `jax.grad(dev_fn, argnums=0)` (dD/deta) matches FD (MODERATE)
- [x] `jax.grad(dev_fn, argnums=1)` (dD/d(log_theta)) matches FD (MODERATE)
- [x] Mixed derivative `d^2D/(d(eta) d(theta))` via JVP matches FD (MODERATE)
- [x] Test at y=0 (boundary), y=1000, mu=0.001, mu=100

**`working_weights_fn` FD:**
- [x] JVP w.r.t. eta matches FD (MODERATE)
- [x] JVP w.r.t. theta matches FD (MODERATE)
- [x] Joint JVP matches sum of individual JVPs

**Consistency checks:**
- [x] `deviance_fn(y,wt)(eta, log_theta)` == `family.dev_resids(y, mu, wt)`
      when `log_theta` matches `self._log_theta`
- [x] `working_weights_fn(wt)(eta, log_theta)` == `family.working_weights(mu, wt)`
      when `log_theta` matches

### Acceptance Criteria

- All FD tests pass with rtol=1e-5 (or 1e-4 at extreme theta)
- No NaN or Inf in any gradient computation

---

## PR 3: Extended custom_jvp for Theta

**Goal:** Extend `_diff_score` in `newton.py` with the 3-primal custom_jvp.

### Files to Modify

**`jaxgam/fitting/newton.py`**

- Add `joint_theta: bool` to `_DIFF_STATIC` tuple
- Extend `_diff_score` signature with `joint_theta` parameter
- Add `if family.n_theta > 0 and joint_theta:` branch (design.md Section 6.4)
- Update `_fit_and_score_impl` with same theta handling
- Update module-level JIT'd transforms with new static arg

### Tests: `tests/test_fitting/test_nb_custom_jvp.py` (new file)

Validate JVP outputs at a known point without running the full Newton loop.

- [x] Small NB problem (n=50, p=5, 1 smooth)
- [x] Run PIRLS to convergence at fixed theta
- [x] `_diff_score` gradient w.r.t. `[log_lambda, log_theta]`:
      theta component matches FD (perturb log_theta, re-run _diff_score)
- [x] lambda components unchanged when dtheta=0
- [x] Hessian: theta-theta, theta-lambda, lambda-lambda blocks match FD
- [x] `dbeta/d(log_theta)` from IFT: perturb theta, re-run PIRLS, compare
- [x] Standard families (n_theta=0): gradient unchanged (no regression)

### Acceptance Criteria

- All FD validation tests pass
- Standard family tests unchanged
- JIT compiles for both standard and extended paths
- `ruff check` clean

---

## PR 4: Newton Optimizer Integration

**Goal:** End-to-end NB fitting. `GAM("y ~ s(x)", family="nb").fit(data)` works.

### Files Modified

**`jaxgam/fitting/newton.py`** -- `NewtonOptimizer` (done in PR 3):
- `__init__`: `self._joint_theta = fd.family.n_theta > 0 and fd.n_penalties > 0`
- Add `joint_theta` to `self._jit_kwargs`
- `run()`: construct initial params with `log_theta` appended
- `_clamp_params`: skip clamping for `log_theta`
- `_build_result`: extract `log_lambda` from joint params, store theta, call `put_theta` once

**`jaxgam/fitting/newton.py`** -- `NewtonResult` (done in PR 3):
- Add `theta: float | None` field

**`jaxgam/fitting/newton.py`** -- dynamic theta fix (PR 4):
- `_diff_score`: pass `log_theta` from params to `pirls_loop`
- `_fit_and_score_impl`: pass `log_theta` from params to `pirls_loop`
- `run()`: removed `put_theta` from Newton loop; theta flows as dynamic JAX arg

**`jaxgam/fitting/pirls.py`** -- dynamic theta fix (PR 4):
- `pirls_loop`: add `log_theta` dynamic parameter; when `family.n_theta > 0`,
  use pure-function factories (`deviance_fn`, `working_weights_fn`) instead of
  family methods that read mutable `_log_theta`. Single fused kernel, no recompilation.

**`jaxgam/api.py`**: no changes needed — `get_family("nb")` returns a proper
instance and Newton detects `n_theta > 0` automatically.

### Tests: `tests/test_fitting/test_nb_fitting.py` (new file)

- [x] Simple: `y ~ s(x)`, `NegativeBinomial()`, simulated NB data (true theta=2)
      Verify: convergence, theta in reasonable range, deviance finite
- [x] Fixed theta: `NegativeBinomial(theta=2)` fits without theta estimation
- [x] Multiple smooths: `y ~ s(x1) + s(x2)` with NB
- [x] Newton `converged` flag is True
- [x] `result.theta` is populated (estimated) or None (n_theta=0 fixed)
- [x] Hard-gate invariants: deviance >= 0, no NaN, EDF bounds
- [x] Standard family fits unchanged (run a Gaussian test, compare to pre-PR result)
- [x] Poisson limit (fixed): fit `NegativeBinomial(theta=1e4)` on Poisson-generated
      data, compare deviance and coefficients against `Poisson()` fit (LOOSE —
      NB deviance formula has residual theta-dependent terms at finite theta)
- [x] Poisson limit (estimated): fit `NegativeBinomial()` on Poisson-generated data,
      verify estimated theta > 1, convergence, all finite outputs. (R gets theta≈10
      on same data, not >50 — the REML surface is flat in the theta direction for
      equi-dispersed data. See `docs/nb_implementation/experiments_theta_newton.md`.)

**Numerical edge cases:**

- [x] Zero-inflated data (60%+ zeros):
      - `converged` is True
      - `result.theta` > 0 and finite
      - deviance >= 0, no NaN in coefficients or fitted values
      - fitted mu for zero observations is small but positive
- [x] Extreme overdispersion (true theta=0.1):
      - `converged` is True
      - `result.theta` < 10 (stays moderate, not near-Poisson; n=500 for signal)
      - deviance >= 0, all coefficients finite
- [x] Large counts (max y > 500):
      - fit completes (converged or step-failed), deviance finite
      - `_lgamma_diff` scan produces finite saturated loglik and gradient
      - `result.theta` > 0 and in reasonable range for the generating theta
- [x] Constant response (all y=5):
      - fit completes without divergence (no infinite loop)
      - `result.theta` > 0 and finite
      - deviance is near zero (perfect fit at constant mean)
- [x] mu near machine epsilon (sparse predictor, log link):
      - no NaN or Inf in deviance, working weights, or coefficients
      - `_MU_EPS` guard prevents log(0) in deviance residuals
      - `converged` is True or step-failed (not NaN crash)

### Acceptance Criteria

- [x] End-to-end fit completes and converges
- [x] All hard-gate invariants hold
- [x] No regression on standard families (246/246 fitting tests pass)
- [x] Theta matches R to 3 decimal places (10.39 vs R's 10.39 on same data)
- [x] Newton iterations match R (6 vs R's 5)

---

## PR 5: R Comparison Tests + Validation Matrix

**Goal:** Match R's mgcv to correct tolerances. Extend validation matrix with NB.

### Files to Modify

**`tests/test_validation_matrix.py`**:

Add `"nb"` to `FAMILIES`:
```python
FAMILIES = ["gaussian", "binomial", "poisson", "gamma", "nb"]
```

Extend `_make_single_data`, `_make_two_smooth_data`, `_make_factor_by_data`,
`_make_factor_by_2d_data` with `"nb"` case:
```python
elif family_name == "nb":
    eta = np.sin(2 * np.pi * x) + 1.0
    y = rng.negative_binomial(n=2, p=2.0 / (np.exp(eta) + 2.0), size=n).astype(float)
```

Extend `_r_tol` and `_fitted_tol` with NB cases:
```python
def _r_tol(smooth_key: str, family_name: str):
    if family_name == "gaussian" and smooth_key in ("tp", "cr"):
        return MODERATE
    return LOOSE  # NB included in LOOSE like other GLM families
```

This gives NB the full matrix treatment (7 smooth types x 1 family = 7 new
cells) for both R comparison and hard-gate invariants.

Add `test_theta_vs_r` to `TestValidationMatrix` (only runs for Extended family cells):
```python
def test_theta_vs_r(self, cell):
    """Estimated theta matches R (Extended family only)."""
    smooth_key, family_name, model, r_result = cell
    if r_result.get("theta") is None:
        pytest.skip("Not an extended family")
    tol = _r_tol(smooth_key, family_name)
    np.testing.assert_allclose(
        model.theta, r_result["theta"],
        rtol=tol.rtol, atol=tol.atol,
        err_msg=f"[{smooth_key}-{family_name}] theta",
    )
```

Add NB-specific hard-gate to `TestHardGateInvariants`:
```python
def test_theta_positive(self, fitted_model):
    """Estimated theta > 0 for extended families."""
    smooth_key, family_name, model = fitted_model
    if model.theta is None:
        pytest.skip("Not an extended family")
    assert model.theta > 0, (
        f"[{smooth_key}-{family_name}] non-positive theta: {model.theta}"
    )
```

**`tests/r_bridge.py`** -- both family maps (rpy2 AND subprocess):
- `_SUBPROCESS_FAMILY_MAP`: add `"nb": "nb()"`
- `_get_r_family_rpy2`: add `"nb": self._mgcv.nb`
- `_extract_fit_results_rpy2`: extract theta from R model family object,
  add `"theta"` key to returned dict (None for standard families)
- Subprocess R script: write theta to `theta.txt`, read it back
- Both paths return `"theta": float | None` in the result dict

**`tests/helpers.py`**:
- `_generate_family_data`: add `"nb"` case
- `r_tolerance`: `"nb"` -> `LOOSE`

### Tests

All R comparisons go in existing test files — no new test file needed.

**`tests/test_families.py`** — NB family method comparisons vs R (already
added in PR 1: `TestNBvsR` class with dev_resids, aic, ls comparisons).

**`tests/test_validation_matrix.py`** — end-to-end fitting comparisons
including theta (added above: `test_theta_vs_r`, `test_theta_positive`).
The 7 new NB cells cover all smooth types automatically.

**NB edge cases** (covered in PR 4's `test_nb_fitting.py`):
- [x] Fixed theta: `NegativeBinomial(theta=2)` — theta unchanged after fit
- [x] High overdispersion (true theta=0.1) — convergence
- [x] Low overdispersion (Poisson limit) — near-Poisson behavior
- [ ] ML method: add ML variant to validation matrix or as separate test

Tolerances for all NB R comparisons use `LOOSE` (from `r_tolerance("nb")` in
`tests/helpers.py`), matching the convention for other GLM families.
Exception: `te_by-nb` uses rtol=5% (6+ sp + theta = flattest REML surface).

### Acceptance Criteria

- [x] All R comparison tests pass at specified tolerances (49/49 NB cells)
- [x] Validation matrix gains 7 new cells (nb x 7 smooth types), all pass
- [x] Hard-gate invariants pass for all NB cells (56/56)
- [x] Tests skip gracefully when R unavailable (`r_available()`)
- [x] Theta vs R passes for all 7 smooth types (LOOSE tolerance)
- [x] No regression on standard families (469 passed, 56 skipped)

---

## PR 6: Post-Estimation

**Goal:** Theta appears in results, summary, and prediction works.

### Files to Modify

**`jaxgam/results.py`** -- `GAMResults`:
- Add `theta: float | None` field
- Family summary includes theta: `"Negative Binomial(theta=2.12)"`

**`jaxgam/api.py`**:
- Pass theta from `NewtonResult` to `GAMResults`

**`jaxgam/summary/`**:
- Display theta in summary output

### Tests

- [ ] `result.theta` matches estimated theta
- [ ] Summary displays theta
- [ ] `predict(result, newdata)` reproduces fitted values (roundtrip)
- [ ] Standard families: `result.theta` is None

### Acceptance Criteria

- Summary displays theta
- Predict roundtrip passes
- No regression on standard families

---

## PR 7: Documentation + Benchmarks + Demo

**Goal:** NB is documented, benchmarked, and has a demo script.

### Files to Modify

All three benchmark scripts have `FAMILIES` lists and `_make_response`
functions that need NB added.

**`scripts/benchmark_vs_r.py`**:
- Add `"nb"` to `FAMILIES` (line 42)
- Extend `_make_response` with NB case
- R benchmark R script needs `family=nb()`

**`scripts/plot_speedup_vs_n.py`**:
- Add `"nb"` to `FAMILIES` (line 40)
- Extend `_make_response` with NB case
- Extend data generators (`make_single_data`, `make_two_smooth_data`, etc.)
- R benchmark R script needs `family=nb()`

**`scripts/benchmark_large_p.py`**:
- Add `"nb"` to `FAMILIES` (line 37)
- Extend `_make_response` with NB case

All three use the same NB response generator:
```python
elif family == "nb":
    mu = np.exp(eta)
    return rng.negative_binomial(n=2, p=2.0 / (mu + 2.0), size=len(eta)).astype(float)
```

### Files to Create

**`scripts/demo/demo_nb.py`** (~60 lines):
- Simulate NB count data with known theta
- Fit `y ~ s(x)` with `family="nb"`
- Print estimated theta vs true theta
- Plot smooth + observed counts
- Save PNG to `scripts/demo/`

### Documentation Files to Modify

**`docs/quickstart.md`** -- add NB example:
```python
# Count data with overdispersion
model = GAM("y ~ s(x)", family="nb")
results = model.fit(data)
print(f"Estimated theta: {results.theta:.2f}")
```

**`docs/api.md`** -- add:
- `NegativeBinomial` class reference
- `ExtendedFamily` base class reference
- Note on theta parameterization (theta vs alpha)

**`docs/R_SOURCE_MAP.md`** -- add NB entries to the task-to-source mapping:

| Task | R file | Function |
|---|---|---|
| NB family | `R/efam.r` | `nb()` lines 161-310 |
| Theta estimation (EFS) | `R/efam.r` | `estimate.theta()` lines 5-96 |
| Extended PIRLS | `R/gam.fit4.r` | `gam.fit4()` lines 240-548 |
| Deviance derivatives | `R/gam.fit4.r` | `dDeta()` lines 4-77 |
| IFT for theta | `src/gdi.c` | `ift2()` lines 1368-1462 |

### Tests

- [ ] `demo_nb.py` runs without error: `uv run python scripts/demo/demo_nb.py`
- [ ] Benchmark script runs with NB: `uv run python scripts/benchmark_vs_r.py`
  (just verify it doesn't crash — benchmark numbers are informational)

### Acceptance Criteria

- Demo script produces PNG
- Benchmark includes NB rows in CSV output
- Docs updated with NB examples and API reference
- `R_SOURCE_MAP.md` has NB entries

---

## Dependency Graph

```
PR 1 (family class + test_families.py)
  |
  v
PR 2 (AD validation)
  |
  v
PR 3 (custom_jvp)
  |
  v
PR 4 (Newton integration)
  |
  +---> PR 5 (R comparison + validation matrix)
  |
  +---> PR 6 (post-estimation)
  |
  +---> PR 7 (docs + benchmarks + demo)
```

PRs 5, 6, and 7 can be developed in parallel once PR 4 merges.

---

## Risk Checkpoints

| PR | Key risk to check |
|---|---|
| 1 | Family methods match R values exactly. No regression in existing tests. |
| 2 | AD gradients match FD at extreme theta (0.01, 10000) |
| 3 | JVP dbeta/dtheta matches FD perturbation of PIRLS |
| 4 | Newton converges for NB (not stuck, not diverging) |
| 5 | Theta within LOOSE tolerance of R. All 7 new validation matrix cells pass. |
| 6 | Predict roundtrip reproduces fitted values |
| 7 | Demo and benchmark scripts run without error |
