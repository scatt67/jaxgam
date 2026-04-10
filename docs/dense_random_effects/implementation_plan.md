# Dense Random Effects — Implementation Plan

**Design doc:** [design.md](design.md)  
**Date:** 2026-04-08  
**Branch:** `add-dense-random-effects`

Six sequential PRs. Each is a small, independently reviewable unit that
leaves the codebase green. Later PRs depend on earlier ones.

---

## PR 1: `RandomEffectSmooth` class + unit tests (DONE)

**Title:** `[phase1] smooths/re: add RandomEffectSmooth basis and penalty construction`

**Goal:** Implement the core smooth class in isolation. After this PR, the
class can construct bases and penalties, but is not wired into the GAM
pipeline.

**Files created:**
- `jaxgam/smooths/random_effects.py` — `RandomEffectSmooth` class
- `tests/test_smooths/test_random_effects.py` — unit tests

**Work:**
1. Implement `RandomEffectSmooth(Smooth)` with:
   - `__init__` — set `side_constrain=False`, `_noterp=True`,
     `_random=True`, `_has_centering_constraint=False`
   - `setup(data)` — detect factors via `is_factor()`/`get_factor_levels()`
     from `by_variable.py`, store levels, build interaction model matrix
     (`~v1:v2:...:vN - 1`), set identity penalty with `_smoothcon_normalize`
   - `_build_interaction_matrix(data)` — row-wise Kronecker product of
     per-variable encodings (one-hot for factors, column for numerics)
   - `build_design_matrix(data)` — delegates to `predict_matrix`
   - `predict_matrix(new_data)` — rebuild interaction matrix, zero out
     non-finite entries (unseen factor levels)
   - `build_penalty_matrices()` — return `[Penalty(self._S, rank=k, null_space_dim=0)]`
2. Add test fixtures to `tests/conftest.py`:
   - `re_factor_data` — single factor, 20 levels
   - `re_two_factor_data` — two factors (3 × 2)
   - `re_numeric_factor_data` — numeric × factor (10 levels)
3. Write unit tests:
   - Structural: shape, rank, `null_space_dim==0`, flags
   - Penalty: symmetric PSD, identity (pre-normalization), full rank
   - Single factor: one-hot indicator matrix is correct
   - Factor × factor: interaction columns and ordering
   - Numeric × factor: weighted indicator
   - Numeric only: single column
   - Prediction: unseen levels → zero rows
   - Prediction: `predict_matrix` reproduces `build_design_matrix` on
     same data
4. Add R basis comparison tests (skip if R unavailable):
   - `s(g, bs='re')` — X, S, rank, null_space_dim vs R (STRICT)
   - `s(g1, g2, bs='re')` — factor interaction vs R (STRICT)
   - `s(x, g, bs='re')` — numeric × factor vs R (STRICT)

**Does NOT touch:** registry, constraints, design.py, summary, validation
matrix. The class exists but is not yet callable from `GAM()`.

**Verify:** `pytest tests/test_smooths/test_random_effects.py` passes.
Existing tests unaffected.

---

## PR 2: Registry + constraint pipeline integration (DONE)

**Title:** `[phase1] smooths/re: register bs="re" and skip centering constraints`

**Goal:** Wire `RandomEffectSmooth` into the model setup pipeline so that
`GAM("y ~ s(g, bs='re')")` can construct the model matrix. After this PR,
RE models can be assembled (Phase 1) but not yet fitted through the full
validation matrix.

**Files modified:**
- `jaxgam/smooths/registry.py` — add `"re": RandomEffectSmooth`
- `jaxgam/smooths/constraints.py` — skip centering for
  `_has_centering_constraint=False`

**Files modified (tests):**
- `tests/test_smooths/test_random_effects.py` — add integration tests

**Work:**
1. Add `"re": RandomEffectSmooth` to `smooth_registry` in `registry.py`.
   Import `RandomEffectSmooth` from `jaxgam.smooths.random_effects`.
2. In `CoefficientMap.build()` (constraints.py), add a skip condition to
   the centering loop:
   ```python
   # Before existing NumericBySmooth check:
   if not getattr(sm, '_has_centering_constraint', True):
       continue
   ```
3. Add integration tests:
   - `ModelSetup.build()` with `"y ~ s(g, bs='re')"` succeeds
   - RE smooth passes through constraint pipeline with no centering
     applied (verify `Z_centering is None` for the RE term)
   - RE smooth has no `del_index` from `gam_side`
   - RE + standard smooth together: `"y ~ s(x) + s(g, bs='re')"` —
     verify both terms are in the model matrix with correct dimensions
   - Verify the number of columns in X equals intercept + smooth cols +
     RE cols (no column lost to centering)

**Verify:** `pytest tests/test_smooths/ tests/test_constraints.py` passes.
`GAM("y ~ s(g, bs='re')").fit(data)` produces a fitted model (manual
smoke test).

---

## PR 3: `SmoothInfo.is_random` + summary p-value type

**Title:** `[phase3] summary: use integer-rank test for random effect terms`

**Goal:** RE terms get the correct p-value test type (`type_=1`) in
`summary()`. After this PR, `model.summary()` produces correct output
for RE models.

**Files modified:**
- `jaxgam/formula/design.py` — add `is_random: bool = False` to
  `SmoothInfo`; populate in `_build_smooth_info()`
- `jaxgam/summary/summary.py` — dispatch `type_=1` for RE terms

**Files modified (tests):**
- `tests/test_smooths/test_random_effects.py` — add summary test

**Work:**
1. Add `is_random: bool = False` field to `SmoothInfo` dataclass
   (design.py). Place it after the existing fields with a default so
   existing construction is unaffected.
2. In `_build_smooth_info()`, populate from the smooth object:
   ```python
   base = getattr(sm, 'base_smooth', sm)
   is_random = getattr(base, '_random', False)
   ```
3. In `summary.py`, replace the hardcoded `type_=0` with:
   ```python
   test_type = 1 if si.is_random else 0
   res = _test_stat(p_i, X_i, V_i, rank=..., type_=test_type, res_df=rdf)
   ```
   where `si` is the `SmoothInfo` for the current smooth. (Need to
   access `smooth_info` by index in the loop — already available as
   `si` from `for i, si in enumerate(smooth_info):`.)
4. Add test: fit `y ~ s(x) + s(g, bs='re')`, call `summary()`, verify
   the RE term's smooth table row uses integer ref.df (whole number)
   while the standard smooth's ref.df is fractional.

**Verify:** `pytest tests/test_smooths/test_random_effects.py tests/test_summary/` passes. Existing summary tests unaffected (no existing smooth
has `_random=True`).

---

## PR 4: Validation matrix — RE cells

**Title:** `[testing] validation-matrix: add re, re_slope, re_mixed cells`

**Goal:** Add RE smooth configs to the validation matrix, providing
systematic R comparison across all 5 families (15 new R-comparison cells +
15 new hard-gate invariant cells).

**Files modified:**
- `tests/test_validation_matrix.py`

**Work:**
1. Add `_make_re_data(family_name, seed)` data generator:
   - n=300, 20 groups, continuous x, true group effects
   - Same family-dispatch pattern as existing generators
2. Add 3 entries to `SMOOTH_CONFIGS`:
   ```python
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
3. Wire `_get_data()`:
   ```python
   if config.data_type == "re":
       return _make_re_data(family)
   ```
4. Update `_r_tol()`: add `"re"` to the Gaussian-MODERATE set:
   ```python
   if family_name == "gaussian" and smooth_key in ("tp", "cr", "re"):
   ```
5. Update `_compare_fitted_not_coefs()`: add `"re_mixed"` to the
   fitted-values set (TPRS sign ambiguity), leave `"re"` and
   `"re_slope"` out (direct coefficient comparison).

**Verify:** `pytest tests/test_validation_matrix.py` — all 50 cells pass
(35 existing + 15 new). Hard-gate invariants pass without R.

---

## PR 5: Prediction with unseen factor levels

**Title:** `[phase3] predict: handle unseen factor levels in RE terms`

**Goal:** Ensure `predict(newdata)` with factor levels not seen during
training produces zero contribution from the RE term, matching R's
behavior. This is already handled at the smooth level (PR 1), but needs
end-to-end validation through the full `predict()` pipeline including
`ModelSetup.build_predict_matrix()`.

**Files modified:**
- `tests/test_smooths/test_random_effects.py` — add end-to-end prediction
  tests

**Work:**
1. Test: fit `y ~ s(x) + s(g, bs='re')`, then predict with newdata
   containing a mix of seen and unseen factor levels. Verify:
   - Predictions for seen levels are finite and reasonable
   - Predictions for unseen levels equal the smooth-only prediction
     (RE contribution is zero)
   - Results match R's `predict.gam()` with the same unseen levels
2. Test: predict with newdata that omits the factor column entirely
   (using `exclude="s(g)"` pattern if supported, or verify the error
   message is clear).
3. Test: predict with a single unseen level — entire RE contribution
   is zero for that observation.

**Verify:** All existing + new tests pass.

---

## PR 6: Documentation + cleanup

**Title:** `[docs] smooths/re: add dense random effects to documentation`

**Goal:** Update user-facing docs and clean up any loose ends.

**Files modified:**
- `docs/dense_random_effects/design.md` — mark status as "Implemented"
- `docs/R_SOURCE_MAP.md` — add RE source mapping
- `docs/quickstart.md` — add RE example (if appropriate)

**Work:**
1. Update `design.md` status to "Implemented" with completion date.
2. Add RE entries to `R_SOURCE_MAP.md`:
   - `smooth.construct.re.smooth.spec` → `jaxgam/smooths/random_effects.py`
   - `Predict.matrix.random.effect` → same file
3. Add a brief RE usage example to quickstart if it includes examples
   of different smooth types.
4. Verify no TODO/FIXME comments remain in the RE code.

**Verify:** Full test suite passes. Documentation builds cleanly.

---

## Dependency Graph

```
PR 1  RandomEffectSmooth class + unit tests
  │
  ▼
PR 2  Registry + constraint pipeline ── requires PR 1
  │
  ├──────────────┬──────────────┐
  ▼              ▼              ▼
PR 3  Summary  PR 4  ValMatrix  PR 5  Prediction
  │              │              │
  └──────────────┴──────────────┘
                 │
                 ▼
        PR 6  Documentation
```

PRs 3, 4, and 5 only depend on PR 2, so they can be developed in parallel
after PR 2 merges. PR 6 is a final cleanup after everything else lands.
