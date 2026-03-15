# GAM API Refactor: Implementation Plan

## Overview

Split `jaxgam/api.py` into a specification class (`GAM`) and an immutable results class (`GAMResults`), extract post-estimation computation into its own module, and provide backward-compatible deprecation forwarding.

**Branch:** `refactor-gam-api-composition`
**Design reference:** `docs/refactor_gam_api/design.md`

---

## Project Tooling

Per AGENTS.md, this project uses `uv` for dependency and package management and `make` for task orchestration. All Python execution must go through `uv run` — never invoke `python` or `pytest` directly.

| Task | Command |
|------|---------|
| Install dependencies | `make install` (`uv sync --extra dev`) |
| Run tests locally | `make test-local` (`uv run pytest`) |
| Run a single test file | `uv run pytest tests/test_post_estimation.py -x --tb=short -v` |
| Lint (ruff check + format + vulture) | `make lint` |
| Format | `make format` |
| Pre-commit hooks | `make pre-commit` |
| Full suite in Docker (with R tests) | `make test` |

When this plan says "run the test suite," use `make test-local`. When running a specific test file during development, use `uv run pytest <path> -x --tb=short -v`.

---

## Phase 1: Extract `PostEstimation` Module (DONE)

**Goal:** Pull the derived-quantity computation out of `_store_results()` into a testable, standalone module.

**Design rationale:** `_store_results()` does 9 distinct things in 90 lines (design.md §4.2). Extracting post-estimation computation achieves design goal #3 — making derived-quantity computation cohesive and independently testable (design.md §2). This phase is a prerequisite for Phase 2: `GAMResults._from_fit()` delegates to `compute_post_estimation()` rather than reimplementing the logic (design.md §4.2, "Benefits").

### Step 1.1: Create `jaxgam/post_estimation.py`

**What to do:**
- Define `PostEstimationResults` frozen dataclass holding: `coefficients`, `Vp`, `edf`, `edf1`, `edf_total`, `null_deviance`, `hat_matrix`, `scale`
- Move these functions from `api.py` into the new module:
  - `_compute_per_smooth_edf()` → `compute_per_smooth_edf()`
  - `_compute_per_smooth_edf1()` → `compute_per_smooth_edf1()`
  - `_compute_null_deviance()` → `compute_null_deviance()`
- Create `compute_post_estimation(result, setup, family, fd, smooth_info)` that:
  - Converts JAX arrays to NumPy
  - Computes `H_inv` via Cholesky solve
  - Computes hat matrix `F = H_inv @ XtWX`
  - Calls the EDF functions
  - Back-transforms reparameterized coefficients
  - Computes `Vp = phi * H_inv`
  - Computes null deviance
  - Returns `PostEstimationResults`

**Design reference:** design.md §4.2 defines the `PostEstimationResults` dataclass and `compute_post_estimation()` function signature. The module sits at the Phase 2→3 boundary (design.md §6 architecture diagram).

**Files touched:**
- Create: `jaxgam/post_estimation.py`
- Read (not yet modify): `jaxgam/api.py` (extract logic from `_store_results`)

**Tests:**
- Create: `tests/test_post_estimation.py`
  - Unit test EDF computation with a known hat matrix
  - Unit test null deviance with known family/data
  - Integration test: fit a model, pass raw results to `compute_post_estimation()`, verify outputs match current `_store_results()` outputs

### Step 1.2: Wire `post_estimation` into existing `api.py`

**What to do:**
- Replace inline computation in `_store_results()` with a call to `compute_post_estimation()`
- Keep the attribute-setting logic in `_store_results()` for now (Phase 2 replaces it)
- Run full test suite to verify no regressions

**Files touched:**
- Modify: `jaxgam/api.py`

**Validation:**
- `make test-local` passes
- All existing API tests pass unchanged

---

## Phase 2: Create `GAMResults` Class (DONE)

**Goal:** Define the frozen results dataclass with prediction, summary, and plot methods.

**Design rationale:** The Model/Results separation (design.md §3) is the core of this refactor. It follows the statsmodels pattern (design.md §3.1) because both libraries are frequentist regression frameworks with formula-based specification and rich result objects. The frozen dataclass eliminates the `_fitted` guard pattern by making impossible states unrepresentable (design.md §2, goals #1 and #2). The class is named `GAMResults` over alternatives like `FittedGAM` or `GAMFit` (design.md §3.6) to match the dominant Python stats convention.

### Step 2.1: Create `jaxgam/results.py`

**What to do:**
- Define `GAMResults` as a frozen dataclass with all attributes listed in design.md §3.4
  - No trailing underscores on attribute names — structural separation replaces the sklearn naming convention (design.md §3.4, decision #2)
  - Omit `Ve_` (frequentist covariance) entirely — no placeholders (design.md §9, decision #4)
  - Store `formula` and `family` as metadata fields on `GAMResults` — no back-reference to the `GAM` specification object to avoid circular dependencies (design.md §9, decision #3)
- Implement `_from_fit()` classmethod (design.md §3.4, decision #4) that:
  - Calls `compute_post_estimation()` from Phase 1
  - Calls `_extract_training_data()` (moved here or imported)
  - Assembles all fields into the frozen dataclass
- Implement `predict()` method — move logic from `GAM.predict()`, keeping the four-case structure (design.md §7)
- Implement `predict_matrix()` method — move logic from `GAM.predict_matrix()`
  - Prediction lives on `GAMResults` rather than a separate `GAMPredict` class because the logic is ~30 LOC and needs the same state `GAMResults` already holds (design.md §4.1, "GAMPredict" verdict)
- Implement `summary()` method — delegate to `jaxgam.summary.summary._summary()`
  - Update `_summary()` to accept `GAMResults` directly (design.md §9, decision #2)
- Implement `plot()` method — delegate to `jaxgam.plot.plot_gam()`
  - Update `plot_gam()` to accept `GAMResults` directly (design.md §9, decision #2)
- Implement `__repr__()` showing formula, family, convergence status, deviance explained

**Files touched:**
- Create: `jaxgam/results.py`
- Modify: `jaxgam/summary/summary.py` (accept `GAMResults`)
- Modify: `jaxgam/plot/plot_gam.py` (accept `GAMResults`)

**Tests:**
- Create: `tests/test_results.py`
  - Test `GAMResults` construction from mock data
  - Test `predict()` self-prediction matches `fitted_values`
  - Test `predict()` with new data
  - Test `predict()` with `se_fit=True`
  - Test `predict_matrix()` returns correct shape
  - Test `summary()` returns `GAMSummary`
  - Test immutability (assigning to a frozen field raises)

### Step 2.2: Update `summary/` and `plot/` to Use a Protocol

**What to do:**
- Define a `FittedModel` protocol in `jaxgam/results.py` (or a shared `_protocols.py`) that captures what `summary()` and `plot()` need:

```python
class FittedModel(Protocol):
    coefficients: np.ndarray
    Vp: np.ndarray
    edf: np.ndarray
    family: ExponentialFamily
    smooth_info: tuple[SmoothInfo, ...]
    # ... etc
```

- Update `_summary()` and `plot_gam()` to type-hint against this protocol
- Both `GAM` (with forwarding) and `GAMResults` satisfy the protocol

**Design rationale:** Protocols over ABCs follows design goal #6 — "Pythonic, not Java-ic" (design.md §2). The protocol enables summary/plot to work with either the old or new API during the migration period (design.md §3.5, migration path) without importing concrete classes.

**Files touched:**
- Create or modify: `jaxgam/results.py` (add protocol)
- Modify: `jaxgam/summary/summary.py` (type hint)
- Modify: `jaxgam/plot/plot_gam.py` (type hint)

---

## Phase 3: Refactor `GAM` Class (DONE)

**Goal:** Slim down `GAM` to specification + fit orchestration, with backward-compat forwarding.

**Design rationale:** With `GAMResults` and `post_estimation` in place, `GAM` can be narrowed to its true responsibility: holding specification parameters and orchestrating the fit pipeline (design.md §3.3). No separate `GAMFit` class is needed because fitting is already delegated to `fitting/` modules — what remains in GAM is ~40 lines of orchestration (design.md §4.1, "GAMFit" verdict).

**Deviation from plan:** The planned rename `api.py` → `gam.py` (Step 3.1) was skipped to reduce churn. The GAM class remains in `api.py`, with `results.py` and `post_estimation.py` as sibling modules. All other Phase 3 objectives were completed as planned.

### Step 3.1: ~~Rename `api.py` → `gam.py`~~ (SKIPPED)

Skipped — keeping GAM in `api.py` avoids import churn for no functional benefit. `results.py` and `post_estimation.py` live alongside `api.py` as top-level modules.

### Step 3.2: Slim Down `GAM` Class (DONE)

**What was done:**
- Removed `_store_results()` — replaced by `GAMResults._from_fit()` (design.md §3.3, "What moves out")
- Removed `_check_fitted()` — no longer needed; impossible states are unrepresentable with separate types (design.md §2, goal #2)
- Removed all fitted attribute setting (30+ attributes)
- Removed `predict()`, `predict_matrix()`, `summary()`, `plot()` method bodies — these now live on `GAMResults` (design.md §3.3, §3.4)
- `fit()` now:
  1. Orchestrates Phase 1 + Phase 2 (unchanged)
  2. Calls `GAMResults._from_fit(...)` to construct results
  3. Stores `self.results_ = results` — enables both `results = model.fit(data)` and `model.fit(data); model.results_.predict(...)` usage (design.md §9, decision #1)
  4. Returns `results`
- Added `__getattr__` forwarding with deprecation warnings (design.md §3.5) — maps `model.coefficients_` → `model.results_.coefficients` during migration
- Added deprecated forwarding methods for `predict()`, `summary()`, `plot()`, `predict_matrix()` (design.md §3.5, migration path: v1.0 emits warnings, v1.1+ removes forwarding)
- Removed dead helper functions (`_compute_per_smooth_edf`, `_compute_per_smooth_edf1`, `_compute_null_deviance`, `_extract_training_data`) — already moved to `post_estimation.py` and `results.py`

**Files touched:**
- Modify: `jaxgam/api.py`

**Tests:**
- Modified existing tests in `tests/test_api/` to use the new API (`results = model.fit(data)`)
- Added deprecation warning tests in `TestDeprecationForwarding` (verify old API emits warnings)
- Updated all test files accessing GAM fitted attributes to use non-underscore names:
  - `tests/test_api/test_gam.py`
  - `tests/test_edge_cases.py`
  - `tests/test_predict/test_predict.py`
  - `tests/test_summary/test_summary.py`
  - `tests/test_plot/test_plot.py`
  - `tests/test_validation_matrix.py`
  - `tests/test_post_estimation.py`

### Step 3.3: Update `__init__.py` Exports (DONE)

**What was done:**
- Exported both `GAM` and `GAMResults` from `jaxgam.__init__`
- `from jaxgam import GAM, GAMResults` works

**Files touched:**
- Modify: `jaxgam/__init__.py`

---

## Phase 4: Clean Up and Documentation (DONE)

**Goal:** Remove dead code, update tests to prefer the new API, and update project documentation.

### Step 4.1: Remove Helper Functions from `api.py` (DONE)

**What was done:**
- Verified `_compute_per_smooth_edf`, `_compute_per_smooth_edf1`, `_compute_null_deviance`, `_extract_training_data` moved to `post_estimation.py` / `results.py`
- All removed from `api.py` (done as part of Phase 3 rewrite)
- `_fit_fixed_sp` kept in `api.py` — still called by `GAM.fit()` for fixed smoothing parameter support

### Step 4.2: Update Existing Tests (DONE)

**What was done:**
- Audited all test files that import from `jaxgam.api` or access GAM fitted attributes
- Updated to use `GAMResults` non-underscore attribute names where `model = GAM(...).fit(data)` pattern is used
- Backward-compat paths tested via `TestDeprecationForwarding` in `tests/test_api/test_gam.py` and `TestMatchesLegacyGAM` in `tests/test_results.py`

### Step 4.3: Update `AGENTS.md` File Organization Section (DONE)

**What was done:**
- Updated file organization table to include `results.py` and `post_estimation.py`
- Updated `__init__.py` description to show `GAM, GAMResults` exports

---

## Success Criteria

1. ✅ `GAM` class core (specification + fit orchestration) is ~100 LOC (down from 693). Backward-compat forwarding methods add ~140 LOC that will be removed in v1.1.
2. ✅ `GAMResults` is a frozen dataclass with all fitted state (design.md §3.4)
3. ✅ `_store_results()` no longer exists — replaced by `GAMResults._from_fit()` + `compute_post_estimation()` (design.md §4.2)
4. ✅ All 1530 tests pass (with backward-compat forwarding per design.md §3.5)
5. ✅ New tests cover: `GAMResults` construction, prediction, immutability, deprecation warnings
6. ✅ Full test suite green at every phase boundary

---

## Estimated Scope (Actual)

| Phase | New Files | Modified Files | New Test Files |
|-------|-----------|----------------|----------------|
| 1 | 1 (`post_estimation.py`) | 1 (`api.py`) | 1 |
| 2 | 1 (`results.py`) | 2 (`summary.py`, `plot_gam.py`) | 1 |
| 3 | 0 (skipped rename) | 8 (`api.py`, `__init__.py`, 6 test files) | modify existing |
| 4 | 0 | 1 (`AGENTS.md`) | 0 |
| **Total** | **2** | **~12** | **2** |
