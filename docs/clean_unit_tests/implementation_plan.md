# Test Suite Cleanup: Implementation Plan

## Overview

Reduce the jaxgam test suite from **2,151 collected tests / 30,020 lines** to roughly **1,250 collected tests / ~27,000 lines** while preserving every R-bridge comparison, hard-gate invariant, finite-difference AD validation, and documented regression test.

**Branch:** single working branch off `main` for the whole cleanup; one PR at the end.
**Audit reference:** `scratch/audit_unit_tests_v3.md` (synthesis of v1 collection-level audit and v2 per-file agent audit)
**Per-file cut lists:** `scratch/audit_unit_tests_v2.md`
**Mandatory-keep regression tests:** `scratch/audit_unit_tests_v2.md` §"Regression Test Inventory"

**Scope decision (resolved):** Random effects (`bs="re"`) are in v1.0. `CLAUDE.md` has been updated to remove RE from the "What Is NOT in v1.0" list. All RE tests remain in the default suite.

---

## Working Model — How This Plan Is Executed

**One PR total.** The entire cleanup ships as a single pull request from one working branch.

**Work is split into discrete commits, not PRs.** Each numbered "Commit" below is a self-contained unit of work that an agent executes end-to-end. Multiple commits land on the same branch in sequence.

**The agent must NOT run `git commit` or `git push`.** The user commits manually after reviewing each unit. The agent's job per commit is:

1. Read the unit's "What to do" and "Files touched" sections.
2. Make the code/test changes.
3. Run the validation commands listed for that unit (`make test-local` and, where R is needed, `make test`).
4. Stop and surface the results to the user (file list, test count delta, coverage delta, validation output). The user reviews, commits, then triggers the next unit.

If a unit's validation fails, fix the issue **in the same unit** before handing off — do not start the next unit on a broken tree. If the fix changes the scope of the unit, note that in the handoff so the user can adjust the commit message accordingly.

The user manages: branching off `main`, commit messages, force-pushes (if any), and the final PR. The agent stays inside the working tree.

---

## Project Tooling

Per CLAUDE.md, this project uses `uv` for dependency management and `make` for task orchestration. All Python execution must go through `uv run`.

| Task | Command |
|---|---|
| Run tests locally (no R) | `make test-local` |
| Full suite in Docker (with R 4.5.2 + mgcv 1.9-3) | `make test` |
| Count collected tests | `uv run pytest --collect-only -q tests \| tail -1` |
| Count source-level tests | `grep -rc "def test_" tests \| awk -F: '{s+=$2} END {print s}'` |
| Coverage | `uv run pytest --cov=jaxgam tests/` |
| Lint | `make lint` |
| Pre-commit | `make pre-commit` |

---

## Success Metrics

Track all three after every commit:

1. **Collected-test count** (`pytest --collect-only`) — primary indicator of CI speed impact.
2. **Source-level test count** (`grep -c "def test_"`) — indicator of maintenance surface.
3. **Coverage percentage** (must stay ≥80% per CLAUDE.md §Testing Rules).

**Baseline** (verified 2026-05-11 on branch `clean-unit-tests`):

- Collected: **2,151**
- Source-level: **~1,082**
- Coverage: TBD — capture in Phase 0.

**Targets after full execution:**

- Collected: ≤ **1,250** (~42% reduction)
- Source-level: ≤ **800** (~26% reduction)
- Coverage: ≥ 80% per module

---

## Phase 0 — Pre-flight: Baseline Capture (**DONE**)

**Goal:** Record current state so each subsequent commit can show measurable progress.

Run on a clean checkout of the working branch (before any cleanup commits):

```sh
uv run pytest --collect-only -q tests | tail -3
grep -rc "def test_" tests | grep -v ":0$" | awk -F: '{s+=$2} END {print s}'
uv run pytest --cov=jaxgam --cov-report=term tests/ 2>&1 | tail -20
time make test-local 2>&1 | tail -5
```

Record the four numbers (collected, source-level, coverage %, wall-clock) in this file (or a `BASELINE.md` next to it). Every subsequent unit references them.

**Exit criteria:** Baseline metrics captured in the repo so every commit's handoff can show measurable progress against them.

---

## Phase 1 — Validation Matrix Consolidation (Commit A) (**DONE**)

**Goal:** Reduce `tests/test_validation_matrix.py` from 15 methods/cell to 2 methods/cell. Saves ~650 collected tests — the single largest reduction in the suite.

**Audit reference:** v3 §1.

### Current structure

`tests/test_validation_matrix.py` has:

- 50 cells (10 smooth configs × 5 families: gaussian, poisson, binomial, gamma, nb), parameterized via fixtures.
- `TestValidationMatrix`: 7 R-comparison methods (`test_deviance_vs_r`, `test_fitted_values_vs_r`, `test_edf_vs_r`, `test_scale_vs_r`, `test_coefficients_vs_r`, `test_self_prediction_roundtrip`, `test_theta_vs_r`).
- `TestHardGateInvariants`: 8 invariant methods (`test_convergence`, `test_deviance_non_negative`, `test_no_nan_in_converged`, `test_edf_bounds`, `test_vp_symmetric_psd`, `test_penalty_psd`, `test_theta_positive`, `test_model_matrix_rank`).
- Total: 50 × 15 = **750 collected tests**.

### Target structure

```python
class TestValidationMatrix:
    def test_matches_r(self, cell):
        # One R fit + one Python fit; all 7 quantities asserted via collector.
        ...

class TestHardGateInvariants:
    def test_all_invariants(self, fitted_model):
        # One model; all 8 invariants asserted via collector.
        ...
```

Total: 50 × 2 = **100 collected tests**.

### Implementation steps

1. **Add a failure-accumulator helper** in `tests/helpers.py`:

   ```python
   class _AssertCollector:
       def __init__(self): self.failures: list[str] = []
       def check(self, name, fn):
           try: fn()
           except AssertionError as e: self.failures.append(f"{name}: {e}")
       def raise_if_any(self, label):
           if self.failures:
               raise AssertionError(
                   f"{label} failed:\n  - " + "\n  - ".join(self.failures)
               )
   ```

   This keeps failure messages debuggable — when consolidation breaks one quantity, the test report still names every broken quantity, not just the first.

2. **Rewrite `TestValidationMatrix`** to one `test_matches_r` method. Body computes the R reference once, then calls `collector.check(...)` for each of the 7 quantities, finally `collector.raise_if_any(cell_id)`.

3. **Rewrite `TestHardGateInvariants`** to one `test_all_invariants` method following the same pattern over the 8 invariants.

4. **Do NOT cut** `test_self_prediction_roundtrip` (v2 flagged it, but it's the CoefficientMap roundtrip check — CLAUDE.md §Common Pitfalls #3 calls this out as load-bearing). Fold it into the consolidated `test_matches_r`.

5. **Per-cell speedup:** since each consolidated test fits once instead of 15 times, also verify the R fit is memoized per cell using a session- or class-scoped fixture. If not, add the memoization in this commit.

### Files touched

- Modify: `tests/test_validation_matrix.py`
- Modify: `tests/helpers.py` (add `_AssertCollector`)

### Validation

- `make test-local` passes.
- `make test` (Docker, with R) passes.
- `pytest --collect-only -q tests/test_validation_matrix.py` reports **100** tests (down from 750).
- Coverage of `jaxgam/` does not drop.
- Wall-clock for the file shorter (fewer R fits per cell).

### Exit criteria

- Collected count for the file: 100.
- All 50 cells still exercise all 15 underlying assertions through the collector.
- **Agent stops and hands off to user for commit.** Do not proceed to Commit B.

---

## Phase 2 — R-Parity Ownership Sweep (Commit B)

**Goal:** Assign each R-parity behavior to exactly one canonical file; delete the duplicates.

**Audit reference:** v3 §2.

### Ownership rules

| File | Owns | Does not own |
|---|---|---|
| `tests/test_validation_matrix.py` | Broad family × smooth R parity (consolidated `test_matches_r`) | Layer internals |
| `tests/test_fitting/test_newton.py` | Optimizer internals: `newton_optimize`, objective traces, gradient/Hessian behavior, convergence diagnostics | Final-model R parity |
| `tests/test_fitting/test_pirls.py` | PIRLS internals: step-halving, offset equivalence, working-response behavior, curvature regressions | Family × R parity |
| `tests/test_predict/test_predict.py` | New-data prediction & SE vs R | Self-prediction roundtrip (matrix owns) |
| `tests/test_summary/test_summary.py` | Summary tables, EDF attribution, p-values, Davies | Final-model parity |
| `tests/test_api/test_gam.py` | Public API orchestration, routing, input validation, fixed-`sp` | Family × R parity, hard-gate copies |

### Implementation steps

1. For each non-matrix file, walk every test that calls `r_bridge`. For each:
   - Asserting on a layer-internal quantity (intermediate `newton_optimize` output, PIRLS working response, etc.)? → **keep**.
   - Asserting only on `GAMResults` fields the matrix now owns (deviance, fitted_values, edf, scale, coefficients, theta) or any hard-gate invariant? → **delete**.

2. Apply specific cuts validated by this sweep:
   - `test_api/test_gam.py::TestEndToEnd::test_gaussian_basic`, `test_all_fields_finite`, `test_vp_symmetric_psd`.
   - `test_fitting/test_newton.py::TestInvariants::test_deviance_non_negative`, `test_no_nan_in_converged`, `test_edf_bounds`.
   - `test_results.py` self-prediction roundtrips that duplicate the matrix.

3. Run the audit grep:

   ```sh
   grep -rl "r_bridge\.fit_gam\|RBridge(" tests/ | \
     xargs grep -l "deviance\|fitted_values\|edf\|coefficients"
   ```

   Each file in the result must own at least one behavior from the table. If not, all R comparisons in it are duplicates.

### Files touched

- `tests/test_api/test_gam.py`
- `tests/test_fitting/test_newton.py`
- `tests/test_fitting/test_pirls.py`
- `tests/test_results.py`
- `tests/test_post_estimation.py`

### Validation

- `make test-local` and `make test` pass.
- For each behavior in the ownership table, exactly one test file is responsible.

### Exit criteria

- ~30–50 source-level tests cut.
- Grep audit confirms no R-parity behavior is silently lost.
- **Agent stops and hands off to user for commit.** Do not proceed to Commit C.

---

## Phase 3 — File-Level Cleanup Commits (Commits C through H)

**Goal:** Apply v2's per-file cut lists. One commit per logical group. Order is biggest mechanical wins first; most numerically sensitive last.

**Audit reference:** v2 (each section below maps to v2's per-file findings).

For every commit in this phase:

- Continue on the same working branch (no new branches).
- Use the cut lists in `scratch/audit_unit_tests_v2.md` as the work list.
- **Do NOT cut anything in v2's "Regression Test Inventory — MUST PRESERVE" table.** Treat that table as a hard allow-list.
- Run `make test-local` after the changes; run `make test` (Docker + R) if any cut touches R-bridge tests or numerical comparisons.
- When handing off to the user, surface: (1) files touched, (2) test count before/after, (3) coverage delta, (4) validation output. The user uses this for the commit message.
- **Do not run `git commit` or `git push`.**

Commits in this phase touch disjoint file sets, so the order does not matter for correctness — but executing in the listed order keeps the most sensitive cleanup (fitting) last, after the others have validated the workflow.

### Commit C: Families & Links cleanup

**Files:** `tests/test_families.py`, `tests/test_links.py`
**Estimated cuts:** ~53–58 source-level tests, ~680 lines.
**Reference:** v2 §Group 3.

**Critical preservation (override v2 if seen):**
- `TestFamilyStaticCacheKey` (lines 1500–1600) — regression for commit `0512673` (NB deepcopy → JIT cache miss). The docstring explicitly names the failure mode.
- `TestNBJITCacheReuse` (lines 1603–end) — same regression.
- `TestNoJaxImports` — Phase-1/Phase-2 architectural boundary guard.

**Watch list:** after cuts, `tests/test_families.py` should still have at least one consolidated test exercising JIT compilation for each Phase 2 method per CLAUDE.md commit conventions.

### Commit D: Smooths cleanup

**Files:** `tests/test_smooths/test_tprs.py`, `test_cubic.py`, `test_tensor.py`, `test_by_variable.py`, `test_random_effects.py`
**Estimated cuts:** ~68 source-level tests, ~485 lines.
**Reference:** v2 §Group 1.

**Critical preservation:**
- Unseen-level RE tests in `test_random_effects.py` (commit `5014785`).
- Factor × factor and numeric × factor column-ordering checks (memory note: design doc had wrong column ordering; always verify against R).

**Pattern:** most cuts are per-basis (cr/cs/cc) duplicates that collapse to `@pytest.mark.parametrize('bs', ['cr','cs','cc'])`. Apply the parameterization rather than deleting the underlying checks.

### Commit E: Penalties / Constraints / Linalg / Edge / Validation cleanup

**Files:** `tests/test_penalties.py`, `tests/test_constraints.py`, `tests/test_linalg.py`, `tests/test_edge_cases.py`, `tests/test_validation_matrix.py` (small sweep beyond Phase 1)
**Estimated cuts:** ~44 source-level tests, ~400 lines.
**Reference:** v2 §Group 4.

**Critical preservation:**
- `CoefficientMap.constrained_to_full` roundtrip tests (CLAUDE.md §Common Pitfalls #3).
- `apply_sum_to_zero` factor-by centering tests (CLAUDE.md §Common Pitfalls #6).
- `test_cholesky_stability.py` — issue #6 regression. Keep as standalone file (do not merge into `test_linalg.py`); the file boundary aids incident traceability.

**Watch list:** input-validation tests in `test_edge_cases.py` (k>n raises, NaN-in-response raises, etc.) should be dropped only if equivalent input validation exists at the GAM construction layer. Verify with a quick read of `jaxgam/api.py` before deleting.

### Commit F: API / Results / Predict / Post-estimation cleanup

**Files:** `tests/test_api/test_gam.py`, `tests/test_results.py`, `tests/test_post_estimation.py`, `tests/test_predict/test_predict.py`
**Estimated cuts:** ~30 source-level tests, ~460 lines.
**Reference:** v2 §Group 5.

**Critical preservation:**
- `TestNBPostEstimation` (newer NB theta post-estimation feature).
- At least one canonical self-prediction roundtrip per family (CoefficientMap load-bearing).
- All `TestSEVsR` tests in predict.

**Note:** Phase 2 (Commit B) already removed the hard-gate duplicates from `test_api/test_gam.py`. This commit handles the remaining metadata / repr / immutability boilerplate (`test_construction_succeeds`, `test_no_trailing_underscores`, `TestRepr::*`, etc.).

### Commit G: Summary / Plot / Formula / Bridge / Registry / Infrastructure cleanup

**Files:** `tests/test_summary/test_summary.py`, `tests/test_plot/test_plot.py`, `tests/test_formula/test_parser.py`, `tests/test_formula/test_design.py`, `tests/test_r_bridge.py`, `tests/test_registry.py`, `tests/test_infrastructure.py`
**Estimated cuts:** ~39–51 source-level tests, ~435 lines.
**Reference:** v2 §Group 6.

**Critical preservation:**
- All R-bridge summary parity (`TestSummaryVsR`, `TestMultiSmoothSummaryVsR`, `TestDaviesVsR`, `TestRandomEffectSummaryVsR`).
- All 10 `TestErrorCases` parser tests.
- `r_bridge.check_versions()` test (CLAUDE.md §Docker Test Environment — protects against silently running tests against the wrong mgcv version).

### Commit H: Fitting cleanup

**Files:** `tests/test_fitting/test_newton.py`, `test_fitting_data.py`, `test_reml.py`, `test_pirls.py`, `test_nb_custom_jvp.py`, `test_nb_fitting.py`, `test_cholesky_stability.py`
**Estimated cuts:** ~32–40 source-level tests, ~500 lines.
**Reference:** v2 §Group 2.

**Most numerically sensitive — do last. Critical preservation:**
- `TestCurvatureConsistency::test_halved_step_curvature_matches_accepted_mu` (commit `7e0b3a5` regression).
- `test_cholesky_stability.py` (all 3 tests) — issue #6 regression.
- All finite-difference validation tests (REML gradient/Hessian FD, NB custom JVP FD, custom_jvp FD).
- Step-halving tests (CLAUDE.md §Common Pitfalls #5: "Step-halving is essential, not optional").

---

## Phase 4 — Final Sweep (Commit I)

**Goal:** After phases 1–3, walk the suite once more and confirm ownership and coverage. This is the last commit before the user opens the PR.

### Steps

1. **Re-run baseline measurement commands** from Phase 0. Confirm targets hit.

2. **Grep for remaining R-bridge duplication:**

   ```sh
   grep -rl "r_bridge\.fit_gam\|RBridge(" tests/ | \
     xargs grep -l "deviance\|fitted_values\|edf\|coefficients"
   ```

   For each file in the result, confirm it owns at least one unique behavior per Phase 2's ownership table. Delete any remaining duplicates.

3. **Run coverage:**

   ```sh
   uv run pytest --cov=jaxgam --cov-report=term-missing tests/
   ```

   Verify coverage ≥ 80% across modules. If any module dropped below, investigate which test was carrying it and either restore it or add a targeted replacement.

4. **Verify regression-test preservation.** Grep the test tree for each entry in v2's "MUST PRESERVE" table:

   ```sh
   grep -rn "TestFamilyStaticCacheKey\|TestNBJITCacheReuse\|test_halved_step_curvature_matches_accepted_mu\|test_cholesky_stability\|TestNoJaxImports" tests/
   ```

   Each must still exist.

5. **Update `CLAUDE.md` §Testing Rules** if any conventions changed during cleanup (e.g., a new helper or parameterization pattern that should be documented).

### Exit criteria

- Collected count ≤ 1,250.
- Source-level count ≤ 800.
- Coverage ≥ 80%.
- No R-bridge behavior tested in more than one file (per ownership table).
- All "MUST PRESERVE" regression tests still exist.
- **Agent stops and hands off to user.** The user reviews, commits, and opens the PR against `main`.

---

## Risk Management

### Risks specific to this cleanup

1. **Consolidated validation matrix loses failure granularity.** Mitigation: the `_AssertCollector` helper in Phase 1 accumulates failures so the test message names every broken quantity, not just the first.

2. **Deleting an R-parity test before another file owns it.** Mitigation: Phase 2 explicitly enforces the ownership table before file-level cleanup begins. The grep-based audit in Phase 4 catches anything missed.

3. **Cutting a test that uniquely covers a code path under JIT.** Mitigation: coverage check after every commit; investigate any per-module drop below 80%.

4. **Regression tests look like toothless tests.** Mitigation: v2's "Regression Test Inventory" is referenced from every commit's handoff notes. The user cross-references before committing.

### Rollback plan

Because each commit lands one logical change on a single working branch, individual commits can be reverted with `git revert <sha>` without disturbing others. Commits C–H touch disjoint files; only Commit B (ownership sweep) and Commit I (final sweep) depend on earlier work.

Commit A (validation matrix consolidation) is the most disruptive single change. If it causes problems, rollback is `git revert <commit-A-sha>` — only one file is affected.

---

## Sequencing Summary

All commits land on a single working branch off `main`. The agent executes one unit at a time, validates, hands off; the user commits manually before triggering the next unit.

```
Phase 0 (baseline capture — no commit)
        │
Commit A  Phase 1 — validation matrix consolidation   ── ~650 collected tests
Commit B  Phase 2 — R-parity ownership sweep          ── ~30-50 source tests
        │
Phase 3 (file-level cleanups; sequential commits on the same branch)
        ├─ Commit C  Families & Links                       ── ~53-58 source tests
        ├─ Commit D  Smooths                                 ── ~68 source tests
        ├─ Commit E  Penalties/Constraints/Linalg/Edge/Validation ── ~44 source tests
        ├─ Commit F  API/Results/Predict/Post-est            ── ~30 source tests
        ├─ Commit G  Summary/Plot/Formula/Bridge/Registry/Infra ── ~39-51 source tests
        └─ Commit H  Fitting (most sensitive — do last)      ── ~32-40 source tests
        │
Commit I  Phase 4 — final sweep + ownership verification + coverage check
        │
User opens single PR against `main`.
```

Commits A and B must complete before Phase 3 starts (Commit B depends on the matrix being the canonical owner, established in A). Phase 3 commits touch disjoint files, so the order within Phase 3 is for safety (most sensitive last), not correctness.

---

## Definition of Done

Met when the user opens the single PR for this cleanup:

- Default collected test count: ≤ 1,250.
- Source-level test count: ≤ 800.
- Coverage: ≥ 80% per module.
- All regression tests in v2's "MUST PRESERVE" table still exist and pass.
- All R-parity behaviors have exactly one canonical owning file (verified by grep).
- `make test` (Docker, with R 4.5.2 + mgcv 1.9-3) passes on the final commit.
- `CLAUDE.md` accurately reflects scope (no stale "RE not in v1.0" statement — fixed pre-Phase-0).
- All nine units (Phase 0 baseline + Commits A–I) are present in the branch history as individual commits authored by the user.
