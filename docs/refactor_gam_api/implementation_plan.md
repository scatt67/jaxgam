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

## Phase 1: Extract `PostEstimation` Module

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

## Phase 2: Create `GAMResults` Class

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

## Phase 3: Refactor `GAM` Class

**Goal:** Slim down `GAM` to specification + fit orchestration, with backward-compat forwarding.

**Design rationale:** With `GAMResults` and `post_estimation` in place, `GAM` can be narrowed to its true responsibility: holding specification parameters and orchestrating the fit pipeline (design.md §3.3). No separate `GAMFit` class is needed because fitting is already delegated to `fitting/` modules — what remains in GAM is ~40 lines of orchestration (design.md §4.1, "GAMFit" verdict). The rename from `api.py` to `gam.py` gives each file a single clear responsibility (design.md §5, §8).

### Step 3.1: Rename `api.py` → `gam.py`

**What to do:**
- Rename `jaxgam/api.py` to `jaxgam/gam.py`
- Update all internal imports (`from jaxgam.api import ...` → `from jaxgam.gam import ...`)
- Keep `api.py` as a re-export shim for any external consumers:
  ```python
  # jaxgam/api.py (shim)
  from jaxgam.gam import GAM, _check_scope_guards, _resolve_device  # noqa: F401
  ```

**Design reference:** design.md §5 (module layout: `api.py` → split into `gam.py` + `results.py`).

**Files touched:**
- Create: `jaxgam/gam.py` (renamed from `api.py`)
- Modify: `jaxgam/api.py` (becomes re-export shim)
- Modify: `jaxgam/__init__.py` (import from `gam` instead of `api`)
- Modify: any internal imports referencing `api`

**Validation:**
- `make test-local` passes

### Step 3.2: Slim Down `GAM` Class

**What to do:**
- Remove `_store_results()` — replaced by `GAMResults._from_fit()` (design.md §3.3, "What moves out")
- Remove `_check_fitted()` — no longer needed; impossible states are unrepresentable with separate types (design.md §2, goal #2)
- Remove all fitted attribute setting (30+ attributes)
- Remove `predict()`, `predict_matrix()`, `summary()`, `plot()` method bodies — these now live on `GAMResults` (design.md §3.3, §3.4)
- `fit()` now:
  1. Orchestrates Phase 1 + Phase 2 (unchanged)
  2. Calls `GAMResults._from_fit(...)` to construct results
  3. Stores `self.results_ = results` — enables both `results = model.fit(data)` and `model.fit(data); model.results_.predict(...)` usage (design.md §9, decision #1)
  4. Returns `results`
- Add `__getattr__` forwarding with deprecation warnings (design.md §3.5) — maps `model.coefficients_` → `model.results_.coefficients` during migration
- Add deprecated forwarding methods for `predict()`, `summary()`, `plot()` (design.md §3.5, migration path: v1.0 emits warnings, v1.1+ removes forwarding)

**Files touched:**
- Modify: `jaxgam/gam.py`

**Tests:**
- Modify existing tests in `tests/test_api/` to use the new API (`results = model.fit(data)`)
- Add deprecation warning tests (verify old API emits warnings)
- Verify old-style access still works: `model.fit(data); model.coefficients_`

### Step 3.3: Update `__init__.py` Exports

**What to do:**
- Export both `GAM` and `GAMResults` from `jaxgam.__init__`
- Ensure `from jaxgam import GAM, GAMResults` works

**Design reference:** design.md §5 (`__init__.py` exports both `GAM` and `GAMResults`).

**Files touched:**
- Modify: `jaxgam/__init__.py`

---

## Phase 4: Clean Up and Documentation

**Goal:** Remove dead code, update tests to prefer the new API, and update project documentation.

**Design rationale:** With the refactor complete, the helper functions that were extracted in Phase 1 may still have stubs in `gam.py`. `_fit_fixed_sp` can be moved closer to the fitting code since it's an implementation detail of Phase 2 orchestration, not a GAM responsibility. The `GAMSummary` module requires no structural changes — it was already well-factored (design.md §4.1, "GAMSummary" verdict).

### Step 4.1: Remove Helper Functions from `gam.py`

**What to do:**
- Verify that `_compute_per_smooth_edf`, `_compute_per_smooth_edf1`, `_compute_null_deviance`, `_extract_training_data` have been moved to `post_estimation.py` or `results.py`
- Remove from `gam.py` if still present
- Remove `_fit_fixed_sp` if it can be inlined into `gam.py:fit()` or moved to `fitting/`

**Files touched:**
- Modify: `jaxgam/gam.py`

### Step 4.2: Update Existing Tests

**What to do:**
- Audit all test files that import from `jaxgam.api` or access GAM fitted attributes
- Update to use `GAMResults` where appropriate
- Ensure backward-compat paths are also tested (design.md §3.5, both APIs work during migration)

**Files touched:**
- Modify: `tests/test_api/test_gam.py`
- Modify: any other test files accessing GAM fitted attributes
- Modify: `tests/test_summary/`, `tests/test_plot/` if they construct GAM objects

### Step 4.3: Update `AGENTS.md` File Organization Section

**What to do:**
- Update the file organization table to reflect new module layout (design.md §5)
- Add `results.py` and `post_estimation.py` descriptions

**Files touched:**
- Modify: `AGENTS.md`

---

## Dependency Graph

```
Phase 1 ─────► Phase 2 ─────► Phase 3 ─────► Phase 4
                  │                │
                  │                ├── Step 3.2 depends on Step 2.1
                  │                │
                  ├── Step 2.1 depends on Step 1.1
                  │
                  └── Step 2.2 can run in parallel with Step 2.1
```

- **Phase 1** is fully independent — extract post_estimation, wire it in, verify no regressions
- **Phase 2** depends on Phase 1 (needs `compute_post_estimation()`)
- **Phase 3** depends on Phase 2 (needs `GAMResults` to exist)
- **Phase 4** depends on Phase 3 (cleanup after main refactor)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing tests | Each phase ends with full test suite run; backward-compat forwarding preserves old API (design.md §3.5) |
| `summary/` and `plot/` break when receiving `GAMResults` instead of `GAM` | Protocol-based typing (design.md §2, goal #6); both old and new types satisfy the protocol |
| Frozen dataclass limits future extensibility | Use `__post_init__` for derived fields; can relax to non-frozen if needed |
| Performance regression from dataclass construction | Negligible — construction is O(1) pointer assignment; actual computation is unchanged |
| Import cycles (`gam.py` ↔ `results.py`) | `gam.py` imports `GAMResults` for return type; `results.py` does not import `GAM` (design.md §9, decision #3 — no back-references) |

---

## Success Criteria

1. `GAM` class is under 150 LOC (down from 693) — narrowed to specification + orchestration (design.md §3.3)
2. `GAMResults` is a frozen dataclass with all fitted state (design.md §3.4)
3. `_store_results()` no longer exists — replaced by `GAMResults._from_fit()` + `compute_post_estimation()` (design.md §4.2)
4. All existing tests pass (with backward-compat forwarding per design.md §3.5)
5. New tests cover: `GAMResults` construction, prediction, immutability, deprecation warnings
6. `make test-local` green at every phase boundary

---

## Estimated Scope

| Phase | New Files | Modified Files | New Test Files |
|-------|-----------|----------------|----------------|
| 1 | 1 (`post_estimation.py`) | 1 (`api.py`) | 1 |
| 2 | 1 (`results.py`) | 2 (`summary.py`, `plot_gam.py`) | 1 |
| 3 | 1 (`gam.py`, rename) | 3 (`api.py`, `__init__.py`, imports) | modify existing |
| 4 | 0 | 3 (`gam.py`, tests, `AGENTS.md`) | 0 |
| **Total** | **3** | **~8** | **2** |
