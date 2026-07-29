# Robust API Redesign (`result` Mode + Lean Inference Core): Implementation Plan

## Overview

Implement the `result="full"|"inference"` fit mode, the two result types
(`GAMResults` / `GAMInferenceResult`), and the lean picklable
`GAMPredictor` inference core in jaxgam. The design is fully specified in
`docs/production_api/design.md` — this document is the **execution plan**
that converts that design into a sequence of self-contained commits.

**Branch:** single working branch off `main` (`production-api`) for the
entire feature; one PR at the end.
**Design reference:** `docs/production_api/design.md` (stable over ten
review rounds; round 10 was a **subtraction** pass — see below).
**Builds on:** `docs/refactor_gam_api/design.md` (the prior spec/results
split and the removed `_fitted` guard).

**Scope:** the **GP `_E_knot` dead-store fix** (Commit B0 — a prerequisite,
not a product of this design); a keyword-only `result` flag on `GAM.fit()`
returning two narrow types; a `GAMInferenceResult` that retains **no dense
training arrays and no penalty caches**; a `GAMPredictor` core (narrow,
`coefficients`/`Vp` read-only, picklable, version-stamped) + a Phase-1
`PredictSpec` (lazy) + a single Phase-1 predict-matrix builder + a single
Phase-3 predict finish; **export** the two new public types; and the
zero-numerical-risk DRY cleanups tied to the refactor (collapse the eight
`setup.*` duplicate fields — seven → `@property` reads, `n` → `_FitDiagnostics`
scalar; drop the dead `hasattr` guard).

**What round 10 removed (read before implementing D or E):** the design's
first nine rounds were purely additive; round 10 deleted rather than
patched. Three consequences for this plan: **(1)** there is **no
`_protocol.py` and no `PredictMatrixBuilder`** — `predict_core` is typed
directly on `PredictSpec`, and `GAMResults.predict()` passes
`self.setup._lazy_predict_spec()` (design §5.1). **(2)** `_E_knot` is a
**dead store fixed at the source in Commit B0**, not a `result`-mode win —
do not let it inflate the Commit-E memory numbers (design §1.1, §2.1).
**(3)** the lean type gains three **zero-byte** property reads —
`smooth_info`, `term_names`, `formula` — because `edf` is uninterpretable
without labels and `formula` must not become a ninth stored duplicate
(design §5.4, §5.5).
Out of scope: serialization *format* / `save()`/`load()` / versioning; a
JAX-free import path; distributional sampling / intervals; a point-only
(drop-`Vp`) mode; the **null-deviance DRY rewrite** and the **`summary()`
CQS change** (both **deferred** to their own PRs — design §10.3). See
design §1.3 / §3.2 for the full scope boundary.

---

**No numerics touched (load-bearing — read before any code):** this
design changes **no statistical result**. Every prediction, standard
error, EDF, scale, and deviance (including `null_deviance`) must be
**byte-identical** to today (design §10). The only behavioral additions
are the new `result` kwarg and the new lean type; the only refactors are
**mechanical** (move a builder, collapse duplicate fields, add a
predict-only copy hook). **The regression gate at every commit is the
existing `tests/test_predict/`, `tests/test_validation_matrix.py`,
`tests/test_summary/`, `tests/test_plot/`, and `tests/test_post_estimation.py`
suites passing unchanged.** If any existing test changes output, a
refactor is wrong — investigate before handoff.

**The no-retention invariant is about banned owners/attributes, NOT array
shape (load-bearing — read before any test code):** the
`result="inference"` memory test asserts that the result holds **none of**
`setup`, `X`, `y`, `weights`, `offset`, `fitted_values`,
`linear_predictor`, `training_data`, and that a **recursive walk** of the
predictor's smooth graph finds no non-`None` `_X`, no non-`None`
`_S`/`_E_knot`, and empty `_penalties` — while the predict transforms
(`_Xu`/`_knt`/`_UZ`/`_F`/`_XP_list`/`_Z_list`/`_shift`/levels) **ARE
present**. **Do not** test "no retained array has a dimension equal to
`n`": knots are legitimately `≤ n` rows (`_Xu` ≤ `max_knots`), so a
shape-based test gives false failures. Design §3.1, §7.1, §12.2.2.

**Droppable set vs kept transforms (load-bearing):** the only arrays a
`copy_for_prediction()` may null are **`_X`** (training design),
**`_S`** (per-smooth penalty), **tensor `_penalties`** (`O(n_coefs²)`),
and **GP `_E_knot`** (`O(n_knots²)` knot–knot kernel — already `None`
after Commit B0; the override is defense-in-depth). Each is dropped
**only because that type's `predict_matrix` does not read it** —
verified per type in design §2.1 with line citations. The predict
transforms (`_Xu`, `_knt`, `_UZ`, `_F`, `_XP_list`, `_Z_list`, `_shift`,
stored levels) **must be kept and shared by reference** — dropping or
deep-copying them silently corrupts predictions or defeats the memory
win. The **registry audit test** (§12.2.8) is the backstop: a new smooth
that neither overrides `copy_for_prediction` nor is allow-listed cannot
ship unaudited.

Two attribute facts verified against the code, so overrides are not
written against a doc-only mental model: **`TensorProductSmooth` has no
`_X` and no `_S`** (it owns `_penalties` + `_XP_list`; `tensor.py:49-50`),
and **`FactorBySmooth`/`NumericBySmooth` are not `Smooth` subclasses and
hold no `_X`** (`by_variable.py:42,244`) — their entire job in
`copy_for_prediction()` is recursing into `base_smooth`. Use
`getattr(clone, attr, None)` rather than assuming an attribute exists.

**Commit C changes the FULL path too (load-bearing):** once
`ModelSetup.build_predict_matrix()` delegates to the lazy `PredictSpec`,
**every** `result="full"` prediction also runs through
`copy_for_prediction()` copies. This is intended and is why
"`tests/test_predict/` passes byte-identically" is a real gate rather than
a formality — the whole existing predict suite becomes a regression test
on the cache-dropping. It is safe only because design §2.1 verified no
`predict_matrix` reads a dropped cache; if any predict test changes
output, that verification was wrong. Design §6.

**Consolidation discipline (load-bearing — read before any test code):**
this plan inherits the test-suite cleanup rules from
`docs/clean_unit_tests/` and `CLAUDE.md` §Testing Rules:

1. **`tests/test_validation_matrix.py` is the canonical owner** of broad
   final-model R parity and the hard-gate invariants. It is **unchanged**
   here (it fits `result="full"`). New files own only the new behavior.
2. **`tests.helpers._AssertCollector`** is required when multiple
   assertions share an expensive fixture or R fit. N assertions against
   one fit produce **one** collected test, not N.
3. **Parameterize**, do not enumerate. A predict-equivalence zoo
   (gaussian/binomial-by/poisson+offset/NB/non-default-link/te/ti ×
   `pred_type`) becomes one `@pytest.mark.parametrize`, not a method per
   cell.

Design §12.3 gives the target footprint: roughly **~14–16 new collected
tests total** (test_result_mode.py ~4→**~6**, test_predictor.py ~7,
test_smooths ~2, test_results ~1). The extra ~2 in `test_result_mode.py` are
the **actual-path wiring tests** (fit-mode `full`==`inference` equivalence +
direct-R on the real `GAMInferenceResult`, and the `_from_fit` snapshot —
§12.2.1/1b/.6): the design rounds to ~14 by folding mode-equivalence into
`test_predictor.py`, but that leg needs `fit(result="inference")` to exist, so
it actually lands in Commit E. Still **consolidated** (`@parametrize` +
`_AssertCollector`), not per-assertion enumeration — if a draft pushes well
past ~16, consolidate more before handing off.

**Direct-R parity for the inference path is rpy2-only (load-bearing):**
the lean/pickled path is compared to R `predict.gam` for tensor,
factor-by, NB, and a **non-default built-in link** — use
`Gamma(link="log")` (the existing `gamma_log` bridge key,
`r_bridge.py:213`). This case **must skip-guard the bridge mode**:
`RBridge()` defaults to `mode="auto"` and silently falls back to the
subprocess path when rpy2 is absent (`r_bridge.py:98`), where `gamma_log`
is not in the family map and would error. Guard it exactly like the
existing GP parity test (`tests/test_smooths/test_gaussian_process.py:93`):
take the shared auto `RBridge` and `pytest.skip(...)` when
`bridge.mode != "rpy2"`. **Offset and locally-defined custom links are
NOT in the direct-R gate** — `RBridge` has no offset argument
(`r_bridge.py:241,1068`) and local links have no R analogue; both are
covered by internal lean==full equivalence + the pickle round-trip
instead. Design §12.2.1b. **This gate runs on the actual
`GAMInferenceResult.predict` in Commit E** (the `_from_fit` wiring), not only
on Commit D's directly-constructed predictor — a constructed predictor can be
correct while the fit-mode wiring is wrong. Do **not** extend the subprocess
family map or add an offset argument to the bridge (RBridge is rpy2-only).

---

## Working Model — How This Plan Is Executed

**One PR total.** The entire redesign ships as a single pull request from
one working branch.

**Work is split into discrete commits, not PRs.** Each numbered "Commit"
below is a self-contained unit of work that an agent executes end-to-end.
Multiple commits land on the same branch in sequence.

**The agent must NOT run `git commit` or `git push`.** The user commits
manually after reviewing each unit. The agent's job per commit is:

1. Read the unit's "What to do" and "Files touched" sections.
2. Make the code/test changes.
3. Run `make test-cov` for validation. This single command runs the full
   suite in Docker (with R 4.5.2 + mgcv 1.9-3) and enforces the ≥80%
   coverage gate — do **not** also run `make test` or `make test-local`,
   they are redundant and slower.
4. Stop and surface the results to the user (file list, test count delta,
   coverage delta, retained-bytes delta where relevant, validation
   output). The user reviews, commits, then triggers the next unit.

If a unit's validation fails, fix the issue **in the same unit** before
handing off — do not start the next unit on a broken tree. If the fix
changes the scope of the unit, note that in the handoff so the user can
adjust the commit message accordingly.

The user manages: branching off `main`, commit messages, force-pushes
(if any), and the final PR. The agent stays inside the working tree.

---

## Project Tooling

Per CLAUDE.md, this project uses `uv` for dependency management and
`make` for task orchestration. All Python execution must go through
`uv run`.

| Task | Command |
|---|---|
| **Validation (use this — only this)** | `make test-cov` |
| Run one file in Docker | `make test-file FILE=tests/test_inference/test_predictor.py` |
| Count collected tests | `uv run pytest --collect-only -q tests \| tail -1` |
| Count source-level tests | `grep -rc "def test_" tests \| grep -v ":0$" \| awk -F: '{s+=$2} END {print s}'` |
| Lint | `make lint` |
| Pre-commit | `make pre-commit` |

`make test-cov` runs the full suite in Docker (R 4.5.2 + mgcv 1.9-3) with
coverage and the `--cov-fail-under=80` gate. It is a strict superset of
`make test` / `make test-local` plus the coverage check.

---

## Hard Allow-List — Must Not Regress

These behaviors must continue to work at **every** commit. Because this
design touches no numerics, the allow-list is the primary correctness
gate. If any breaks, the unit's validation has failed and the fix lands
in the same unit before handoff:

- **All `tests/test_predict/` tests** — Commit C moves the predict-matrix
  builder out of `ModelSetup`; new-data prediction and SE must be
  byte-identical pre/post.
- **CoefficientMap roundtrip** (CLAUDE.md §Common Pitfalls #3) —
  `predict(model, original_data) == model.fitted_values` for every fit,
  on `result="full"`.
- **All `tests/test_validation_matrix.py` cells** — final-model R parity
  and the eight hard-gate invariants (objective monotonicity, H
  symmetry/PSD, penalty PSD, rank, EDF bounds, deviance non-negativity,
  no NaN). Fitting is untouched, so these must pass unchanged.
- **`tests/test_summary/`, `tests/test_plot/`** — the full `GAMResults`
  diagnostic surface is unchanged.
- **`tests/test_post_estimation.py`** — `null_deviance` (deferred,
  untouched) and offset behavior stay byte-identical.
- **SE byte-parity** — `finish_prediction` reproduces
  `sqrt(rowSums((X_p @ Vp) * X_p)) * |mu_eta|` from `results.py:299–307`
  exactly (design §10.2).
- **R bridge version check** — `RBridge.check_versions()` continues to
  pass for R 4.5.2 + mgcv 1.9-3, and the existing non-GP/non-`gamma_log`
  bridge tests are untouched.

If any of these fail during a commit, **stop and surface the failure**
before continuing. Do not work around — fix the root cause.

---

## Success Metrics

Track after every commit:

1. **Collected-test count** (`pytest --collect-only`) — grows
   monotonically toward **~14–16** new tests across Commits B1–E — B0 adds
   **zero** (it restructures one existing assertion) (the
   actual-path wiring tests in Commit E account for the upper end — see the
   §12.3 footprint note). Should not shrink except where a refactor
   legitimately consolidates; **do not delete the Commit-E wiring tests** to
   chase a lower count.
2. **Coverage percentage** (must stay ≥80% per CLAUDE.md) — overall and
   for every new file (`jaxgam/inference/*`, `jaxgam/formula/predict_matrix.py`)
   by the time Commit E lands.
3. **Retained-bytes delta** — the raison d'être of this design. Measure the
   **true** footprint with a **recursive, `id()`-deduplicated `np.ndarray`
   walker** over the result object graph — **not** a sum of named fields,
   which both misses the **retained** predict transforms
   (`_Xu`/`_UZ`/`_F`/`_XP_list`/`_Z_list`/`_shift` and the lean predictor's
   `PredictSpec` arrays — the design counts these toward `O(p² + Σ
   predict-transform size)`, design §3.1/§7.1) and can double-count
   shared buffers. The walker sums every **distinct** reachable
   `ndarray.nbytes` and **line-items the dropped arrays** (`setup.X`,
   per-smooth `_X`/`_S`/`_penalties`/`_E_knot`, **`setup.penalties`** — the
   `CompositePenalty` in embedded `(total_p, total_p)` space, `design.py:133`,
   which is dropped with `setup` and is otherwise an unexplained residual on
   multi-penalty models — `training_data`, `fitted_values`,
   `linear_predictor`) so the `full` − `inference` delta is
   attributable. Run it for a `result="full"` vs `result="inference"` fit of
   the **same** model — a **tensor** model (so `_penalties` at `O(n_coefs²)`
   shows) and a **GP** model (so `_E_knot` at `O(n_knots²)` shows), at `n=300`
   and `n=5000`.

   **Three measurement points, not two — B0 must be bracketed.** Commit A
   records the `"full"` footprint of **today's** code (including the 32 MB
   `_E_knot` leak). Commit **B0** re-records `"full"` after the dead-store fix.
   Commit E records `"inference"`. The number this design is entitled to claim
   is **B0-full − E-inference**; the A → B0 drop belongs to the one-line bug
   fix. Reporting A-full − E-inference as the design's win would credit it with
   ~32 MB it did not earn (design §1.1).
4. **`make test-cov` wall-clock** — sanity check that new R-parity cells
   don't blow up runtime.

**Baseline (captured in Commit A):** record current collected count,
source-level count, coverage %, wall-clock, and the `result="full"`
retained-bytes figures. Reference these in every subsequent handoff.

---

## Commit A — Pre-flight: Baseline Capture + `cloudpickle` Dev Dep

**Goal:** Record current state so each subsequent commit can show
measurable progress and gate against regressions, and add the one
dev-only dependency the later pickle tests need.

**Design reference:** §8, §13 (commit A).

### What to do

1. Run on a clean checkout of the working branch (before any code
   change):

   ```sh
   uv run pytest --collect-only -q tests | tail -3
   grep -rc "def test_" tests | grep -v ":0$" | awk -F: '{s+=$2} END {print s}'
   time make test-cov 2>&1 | tail -25
   ```

2. **Capture the `result="full"` retained-bytes baseline.** Write a
   small throwaway script (do not commit it) that fits a tensor model and a
   GP model at `n ∈ {300, 5000}` and measures retained bytes with the
   **recursive, `id()`-deduplicated `np.ndarray` walker** of Metric #3 over
   the result object graph: sum every distinct array's `.nbytes`, and
   **line-item** both the soon-to-be-dropped arrays (tensor `_penalties`, GP
   `_E_knot`, `setup.X`, per-smooth `_X`/`_S`, `training_data`,
   `fitted_values`, `linear_predictor`) **and** the retained transforms
   (`_Xu`/`_UZ`/`_F`/`_XP_list`/`_Z_list`/`_shift`) so Commit E's reduction is
   attributable. Record the totals + line items. This is the "before" the
   design exists to improve.

3. Record all metrics in `docs/production_api/BASELINE.md` (mirroring
   `docs/gaussian_process/BASELINE.md`):
   - Collected test count
   - Source-level test count
   - Coverage %
   - `make test-cov` wall-clock
   - `result="full"` retained bytes: tensor & GP, `n=300` & `n=5000`,
     with the `_penalties` / `_E_knot` line items called out.

4. **Add `cloudpickle` to `[project.optional-dependencies].dev`** in
   `pyproject.toml` — the existing `dev` **extra** (`pyproject.toml:17–18`),
   **not** a uv `[dependency-groups]` section. Docker builds the test env with
   `uv sync --extra dev --extra r` (`Dockerfile:36,45`), so cloudpickle must
   live in that **extra** or Commit D's local-custom-link pickle test won't
   see it in Docker. It is **not** a runtime dependency — stdlib `pickle` is
   the public default (design §8). Run `uv sync --extra dev` to update the
   lock.

### Files touched

- Add: `docs/production_api/BASELINE.md`
- Modify: `pyproject.toml` (`cloudpickle` in the
  `[project.optional-dependencies].dev` extra) + lockfile

### Validation

- `make test-cov` runs cleanly (no behavior change yet).
- `cloudpickle` resolves in the dev environment (`uv run python -c
  "import cloudpickle"`).

### Exit criteria

- Baseline file committed with all five metric groups.
- `cloudpickle` present in the `dev` **extra** only (so `uv sync --extra dev`
  installs it) — **not** in `[project.dependencies]` runtime deps.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit B0.

---

## Commit B0 — GP `_E_knot` Dead-Store Fix (prerequisite, not part of the mode)

**Goal:** Stop retaining the `O(n_knots²)` GP knot–knot kernel past
`setup()`. This is a **standalone bug fix** that needs nothing from this
design, and it must land before the memory claims are measured so the
Commit-E delta reflects what `result="inference"` actually buys.

**Design reference:** §1.1, §2.1, §7.2, §13 (commit B0).

### Why this is separate

`_E_knot` is assigned at `gaussian_process.py:200` and **read nowhere in
`jaxgam/`** — verify before editing:

```sh
grep -rn "_E_knot" jaxgam/       # expect exactly 2 hits: :90 decl, :200 assign
```

It is **89% of the GP baseline footprint** (32.0 MB of 36.08 MB at
`n=5000`), and `result="full"` leaks all of it too. Fixing it inside
`copy_for_prediction()` would have hidden a plain dead store behind a new
API and credited this design with a win it did not earn.

### What to do

1. **`jaxgam/smooths/gaussian_process.py`** — in `setup()`, after `E` has
   been consumed by the eigendecomposition, drop the reference:
   `self._E_knot = None`. Keep the attribute declared at `:90` (the
   `copy_for_prediction()` override in Commit C and the §12.2.2 banned-state
   walk both expect the name to exist and be `None`).

2. **`tests/test_smooths/test_gaussian_process.py:497`** — the one existing
   reader (`_assert_close(py_smooth._E_knot, r_result["E"], STRICT)`) is an
   R-parity assertion on the knot–knot kernel and **must not be deleted** —
   it is real mgcv parity coverage. Restructure it to compare the matrix
   **where it is still live**: assert against the value computed inside
   setup (expose it via the existing `_gp_E` helper on the same inputs, or
   capture it before setup nulls the attribute). The assertion must still run
   and still compare to R's `E` at `STRICT`.

3. **Re-measure GP retained bytes** with the Commit-A walker and append the
   post-B0 `"full"` figures to `BASELINE.md` as the **second** measurement
   point (Metric #3). This is the true "before" for the `result` mode.

### Files touched

- Modify: `jaxgam/smooths/gaussian_process.py` (one line in `setup()`)
- Modify: `tests/test_smooths/test_gaussian_process.py` (restructure the
  `_E_knot` parity assertion so it still checks R's `E`)
- Modify: `docs/production_api/BASELINE.md` (post-B0 `"full"` figures)

### Validation

- `make test-cov` passes. **`tests/test_smooths/test_gaussian_process.py`
  passes in full**, including the restructured `E`-vs-R assertion — if that
  assertion was weakened or dropped, the commit is wrong.
- All other GP tests (construction, prediction, parity) unchanged.
- Collected test count **unchanged** (restructure, not addition).

### Exit criteria

- `grep -rn "_E_knot" jaxgam/` shows the declaration, the assignment, and
  the new `None` reset — and nothing reads it.
- GP `"full"` retained bytes drop by ~`n_knots² × 8` vs Commit A, recorded
  in `BASELINE.md`.
- R parity for the GP `E` matrix still asserted at `STRICT`.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit B1.

---

## Commit B1 — DRY: `setup.*` Duplicates → Properties + Drop Dead Guard

**Goal:** Collapse the **seven** genuinely `setup`-backed alias fields that
`GAMResults` stores verbatim into guarded `@property` reads, and remove the dead
`hasattr` guard in `__repr__`. (`n` is the eighth historical `setup.*` duplicate
but is **left alone here** — Commit E re-homes it into `_FitDiagnostics` as
scalar metadata, since the lean type has no `setup`; §5.5.) **Mechanical, zero
behavior change** — every field still reads the same value.

**Design reference:** §2.3, §2.4, §5.5.

### What to do

1. **`jaxgam/results.py`** — **seven** of the eight fields set verbatim from
   `setup.*` in `_from_fit` (`results.py:220–233`): `X`, `y`, `weights`,
   `offset`, `coef_map`, `smooth_info`, `term_names`. Remove them as stored
   fields and expose each as a guarded `@property` delegating to `self.setup`
   (design §5.5). The guard handles the (current) invariant that `setup` is
   always present on `GAMResults`; the property simply returns
   `self.setup.<field>`. **Leave `n` as the existing stored scalar** — it is
   the eighth historical `setup.*` alias (`= setup.n_obs`), but Commit E
   re-homes it into the `_FitDiagnostics` shared base (the lean type has no
   `setup`), so converting it to a property here only to undo that in E is
   pointless churn.

2. **Drop the dead `hasattr(self.family, "family_name")` guard** in
   `__repr__` (`results.py:406`) — `family_name` is always present on the
   family object (design §2.4).

3. Confirm nothing else in `results.py`, `summary/`, or `plot/` writes
   these as attributes (they should only ever be read).

### Files touched

- Modify: `jaxgam/results.py` (7 fields → properties; `n` left as a scalar
  for Commit E to re-home; drop dead guard)
- Modify: `tests/test_results.py` (one consolidated test that the
  duplicates now read through `setup` and equal the prior values)

### Validation

- `make test-cov` passes with **no output change** — this is the merge
  gate. Existing `tests/test_results.py`, `tests/test_summary/`,
  `tests/test_plot/` must pass unchanged.
- New tests collected count up by **~1** (a `_AssertCollector`-backed
  "duplicates-as-properties" test).
- Coverage of `jaxgam/results.py` does not drop.

### Exit criteria

- The **seven** fields read through `setup` via `@property`; `n` remains a
  stored scalar (Commit E re-homes it to `_FitDiagnostics`); no other stored
  aliases remain.
- Dead `hasattr` guard removed.
- All existing result/summary/plot tests pass identically.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit C.

---

## Commit C — Phase-1 Prediction State: `PredictSpec`, Builder Move, `copy_for_prediction`

**Goal:** Establish all **Phase-1** prediction plumbing the lean core
will consume: move `ModelSetup.build_predict_matrix` + its helpers to a
new `formula/predict_matrix.py` free function over a concrete
`PredictSpec`; build the spec **lazily** on the frozen `ModelSetup`; add
the polymorphic `Smooth.copy_for_prediction()` hook (+ tensor / GP /
by-variable overrides); and `build_predict_spec(setup)`. This is the
structural heart of the design. **No Phase-3 code, no new result type
yet** — but `ModelSetup.build_predict_matrix(newdata)` must keep working
identically (it now delegates), so the existing predict suite is the
regression gate.

**Design reference:** §5.1, §5.2, §6 (Phase-1 half), §7.

### What to do

1. **Create `jaxgam/formula/predict_matrix.py` (Phase 1, NumPy only — no
   JAX, no links).** Move the body of `ModelSetup.build_predict_matrix`
   (`design.py:415–491`) here as a free function
   `build_predict_matrix(spec: PredictSpec, newdata) -> np.ndarray`, and
   move its instance helpers to **module-level** functions — **the full
   transitive closure**, not just the three the builder calls directly:
   `_to_dict` (already a `@staticmethod`), `_validate_equal_lengths`,
   `_build_parametric_matrix`, **plus the two they pull in** —
   `_encode_factor` (`design.py:696`, called by `_build_parametric_matrix` at
   `:852`) and `_contr_poly` (`design.py:674`, called by `_encode_factor` at
   `:731`) (design §6). Moving the **whole** closure is what actually keeps
   the import one-directional (next paragraph): leaving `_encode_factor` /
   `_contr_poly` on `ModelSetup` would force `predict_matrix.py` to import
   `design.py`, the cycle the plan is avoiding.

   **These helpers are shared, not predict-only — do not orphan their other
   callers.** `ModelSetup.build()` calls three of them
   (`_to_dict`/`_validate_equal_lengths`/`_build_parametric_matrix`,
   `design.py:184/227/257`); and the formula suite calls **two** directly —
   `_build_parametric_matrix` (`test_design.py:858/880/892`) and `_contr_poly`
   (`test_design.py:947/960/961/967`). **Recommended (lowest-risk):** put the
   single implementation of all five in `predict_matrix.py` and keep **thin
   `@staticmethod` delegators** on `ModelSetup` for every helper with an
   external caller (`_to_dict`, `_validate_equal_lengths`,
   `_build_parametric_matrix`, `_contr_poly`; e.g.
   `return predict_matrix._build_parametric_matrix(...)`). Then `build()` and
   every existing formula test stay **byte-unchanged** (no edit to
   `test_design.py`), yet the design's goal is still met — the helpers are
   module-level functions and `build_predict_matrix` is a free function that
   no longer needs a `ModelSetup`. The import is one-directional: `design.py`
   imports from `predict_matrix.py`, which imports **no** `design.py` symbol
   at module scope (`build_predict_spec(setup)` takes `setup` as a duck-typed
   parameter), so there is no cycle. **Alternative (cleaner, more churn):**
   drop the delegators and update `build()`'s three call sites **and**
   `test_design.py`'s call sites (three `_build_parametric_matrix` + four
   `_contr_poly`) to the module-level functions — in which case **add
   `tests/test_formula/test_design.py` to Files-touched**.

2. **Define `PredictSpec`** in the same module — a **concrete frozen
   dataclass with explicit fields**, exactly the `n`-independent set the
   builder reads (design §2.2, §5.2): `coef_map`, `smooth_info`,
   `parametric_terms`, `factor_info`, `ordered_factors`, `has_intercept`,
   `parametric_keep_cols`, `dropped_param_names`, `total_coefs`. It does
   **not** carry `offset_was_nonzero` (the builder never reads it; that
   flag is a `GAMPredictor`/`setup` field only — single source of truth,
   §5.2). `PredictSpec.build_predict_matrix(self, newdata)` is a one-line
   delegation to the free function.

3. **Add `build_predict_spec(setup) -> PredictSpec`** (Phase 1): rebuild
   `coef_map` with **predict-only smooths** via `dataclasses.replace`
   (design §7.2 code) — **no in-place mutation of `setup`, no deepcopy of
   `_X`**. Shares `Z_centering` etc. by reference.

4. **`ModelSetup` lazy spec** (`jaxgam/formula/design.py`): add a private
   `field(init=False, default=None, compare=False, repr=False)` cache and
   a `_lazy_predict_spec()` that builds + caches the spec on first use via
   `object.__setattr__` (because `ModelSetup` is **frozen**,
   `design.py:79` — `functools.cached_property` does not work cleanly;
   design §5.2). `ModelSetup.build_predict_matrix(self, newdata)` becomes
   `return build_predict_matrix(self._lazy_predict_spec(), newdata)`.
   **A full fit that never predicts builds no spec** (goal #3).

5. **`Smooth.copy_for_prediction()` hook** — implement exactly per design
   §7.2:
   - **`jaxgam/smooths/base.py`**: base default — shallow `copy.copy`,
     null `_X` and `_S`, share transforms by reference. Document the
     **base-default precondition** (correct only when non-predict caches
     are exactly `{_X, _S}` and there are no nested smooths).
   - **`jaxgam/smooths/tensor.py`**: override — set `_penalties = []` and
     recurse `copy_for_prediction()` over `_marginals`; keep `_XP_list`
     (+ `ti`'s `_Z_list`) by reference. (Tensor has **no `_X`/`_S`** of its
     own — `tensor.py:49-50`; the `O(n·k)` training designs live on the
     marginals, which is what the recursion frees. `TensorInteractionSmooth`
     inherits this override.)
   - **`jaxgam/smooths/gaussian_process.py`**: override — `super()` (drops
     `_X`/`_S`) then null `_E_knot`. **Defense-in-depth only:** Commit B0
     already nulls it at `setup()`, so this is a no-op on a live fit. Keep it
     so the invariant survives a future change and so §12.2.2's banned-state
     walk has one consistent shape across types.
   - **`jaxgam/smooths/by_variable.py`** (`FactorBySmooth` /
     `NumericBySmooth`): override — recurse over `base_smooth`. These are
     **not `Smooth` subclasses** (`by_variable.py:42,244`) and hold no `_X`,
     so the recursion is the whole job; use `getattr(clone, "_X", None)`
     rather than assuming the attribute.
   - cubic / TPRS / random-effect use the **base default** (verified:
     their `predict_matrix` reads only `_knots`/`_F`, `_Xu`/`_UZ`, or
     stored levels — never `_S`; design §2.1).

   **Each override nulls a cache only because that type's `predict_matrix`
   does not read it** — re-verify against the cited lines while
   implementing (design §2.1).

6. **Tests** — Phase-1 only, in `tests/test_smooths/` (the canonical
   owner of `copy_for_prediction` per design §12.1), consolidated:
   - **Per-type `copy_for_prediction`** (`@parametrize` over registered
     smooth types + `_AssertCollector`, §12.2.8): after the copy,
     `_X`/`_S`/`_penalties`/`_E_knot` are dropped (recursively) **but**
     `predict_matrix(newdata)` equals the pre-copy result (transforms
     intact) **and** the original smooth is untouched (no aliasing of the
     drop into the live smooth).
   - **Registry audit** (§12.2.8): enumerate every type in
     `smooth_registry` (`registry.py:26`) + the by-wrappers; assert each
     either overrides `copy_for_prediction` or appears in an explicit
     `_BASE_DEFAULT_OK` allowlist. **Fails for any unaudited registered
     type** — this is how a future smooth that forgot the hook is caught.
   - **Spec equivalence + aliasing** (Phase 1): `build_predict_spec(setup)
     .build_predict_matrix(newdata)` equals
     `setup.build_predict_matrix(newdata)`; and `build_predict_spec` does
     **not** mutate `setup` — for a model with a **tensor** and a **GP**
     smooth, assert the live `setup`'s smooths still hold **all** their
     non-predict caches (`_X`, `_S`, tensor `_penalties` non-empty, GP
     `_E_knot` non-`None`) — the **full** no-mutation contract (design §7.2),
     not just `_X`/`_S`.
   - **Lazy spec stays unbuilt until needed** (goal #3, design §5.2/§3.1) —
     fold into the spec-equivalence collector above (no new collected test):
     assert the private spec-cache field is still `None` right after
     `ModelSetup.build(...)`, then non-`None` only after the **first**
     `build_predict_matrix(newdata)`. (The `GAM.fit(result="full")`
     never-predicts-no-spec check rides Commit E's aliasing collector, where a
     full fit exists.)

### Files touched

- Add: `jaxgam/formula/predict_matrix.py` (`PredictSpec`,
  `build_predict_matrix`, helpers, `build_predict_spec`)
- Modify: `jaxgam/formula/design.py` (`ModelSetup` lazy spec; `build_predict_matrix`
  + the **five** shared helpers —
  `_to_dict`/`_validate_equal_lengths`/`_build_parametric_matrix`/`_encode_factor`/`_contr_poly`
  — delegate to `predict_matrix.py`; keep `@staticmethod` shims so `build()`
  and the formula tests stay unchanged)
- (Only if you took the no-delegator alternative) Modify:
  `tests/test_formula/test_design.py` (**7** direct call sites → module-level
  helpers: 3× `_build_parametric_matrix` at `:858/880/892` + 4× `_contr_poly`
  at `:947/960/961/967`)
- Modify: `jaxgam/smooths/base.py`, `tensor.py`, `gaussian_process.py`,
  `by_variable.py` (add `copy_for_prediction()`)
- Modify: `tests/test_smooths/` (per-type copy + registry audit + spec
  equivalence/aliasing, consolidated)

### Validation

- `make test-cov` passes. **All `tests/test_predict/` tests pass
  byte-identically** — the builder move is the merge gate. **`tests/test_formula/test_design.py`
  also passes unchanged** (the `@staticmethod` delegators keep
  `ModelSetup.build()` and the direct `_build_parametric_matrix` **and
  `_contr_poly`** test calls working); if you took the no-delegator
  alternative, its seven call sites (3× `_build_parametric_matrix` + 4×
  `_contr_poly`) were updated instead.
- New tests collected count up by **~2–3** (per-type copy + registry
  audit + spec equivalence incl. the lazy-spec check, consolidated via
  parametrize/collector).
- Coverage of `jaxgam/formula/predict_matrix.py` ≥ 80%.

### Exit criteria

- `ModelSetup.build_predict_matrix(newdata)` delegates and produces
  identical output; a never-predicting full fit builds no spec.
- Every smooth type drops its non-predict caches and preserves
  `predict_matrix` output; the registry audit passes for all registered
  types.
- `build_predict_spec` does not mutate `setup`.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit D.

---

## Commit D — `inference/` Core: `predict_core` + `finish_prediction` + `GAMPredictor`

**Goal:** Build the lean Phase-3 inference core: the single
`finish_prediction` + `predict_core` finishing path over the concrete
`PredictSpec` (no Protocol — design §5.1), and the frozen, picklable
`GAMPredictor`.
Test predictor behavior directly (predict-equivalence, direct-R parity,
pickle/cloudpickle/version-stamp, read-only, snapshot independence) **before** the
result-mode wiring lands in E.

**Design reference:** §5.1, §5.3, §6 (Phase-3 half), §8, §12.2.

> **Ordering note.** `GAMResults.to_predictor()` does not exist until
> Commit E. D therefore constructs `GAMPredictor` **directly** in tests
> from a `result="full"` fit: `coefficients`/`Vp` from the result, a
> `copy.deepcopy` family snapshot, and `_predict_spec =
> build_predict_spec(res.setup)` (the Commit-C Phase-1 builder). Every test
> that needs the **real fit-mode wiring** lands in **E**, where
> `fit(result="inference")`/`to_predictor()` exist: the `to_predictor()`
> aliasing/ownership test (§12.2.5), the **actual-path `full`==`inference`
> predict-equivalence + direct-R gate on `GAMInferenceResult` (§12.2.1/1b)**,
> and the **authoritative `_from_fit` family-snapshot test (§12.2.6)**. D
> proves the predictor is correct *in isolation*; E proves `_from_fit` wires
> it correctly.

### What to do

1. **No `_protocol.py`.** Round 10 cut the `PredictMatrixBuilder` Protocol
   (design §5.1): after Commit C's delegation, `ModelSetup` and
   `PredictSpec` are one implementation and a forwarder, not two
   implementations, so the seam abstracted nothing. Do **not** create this
   file. `predict_core` is typed directly on `PredictSpec`.

2. **`jaxgam/inference/_core.py`** — `finish_prediction(eta, X_p, link,
   Vp, *, pred_type, se_fit)` and `predict_core(spec: PredictSpec,
   coefficients, Vp, link, newdata, *, pred_type="response",
   se_fit=False, offset=None, offset_was_nonzero=False)`, verbatim per
   design §6, calling `spec.build_predict_matrix(newdata)`. `Vp` is required
   (no `None` branch). SE is the **exact**
   `sqrt(rowSums((X_p @ Vp) * X_p)) * |mu_eta|` from `results.py:299–307`.
   The external-offset warning fires when `offset_was_nonzero` and no
   offset is passed (matches `predict.gam`). **Both `predict_core` and
   `finish_prediction` stay private** (not exported — design §11.1).

3. **`jaxgam/inference/predictor.py`** — the frozen `GAMPredictor`
   dataclass per design §5.3: fields `coefficients`, `Vp` (required),
   `family` (snapshot), `formula` (**the single owner of the formula string
   — Commit E's lean type reads it as a property, not a stored field**),
   `offset_was_nonzero` (explicit bool), `_predict_spec: PredictSpec`, and
   `_jaxgam_version: str` defaulting to `jaxgam.__version__`.
   `__post_init__` **defensively copies** `coefficients`/`Vp` (`np.array`)
   then `setflags(write=False)`; `__setstate__` re-applies `write=False`
   after unpickle (NumPy does not always preserve the flag) **and warns when
   `_jaxgam_version` does not match the running version** — the pickle
   contract is same-version (design §8), and a warning is the difference
   between a loud failure and silently wrong production predictions. It is a
   guardrail, **not** the versioned artifact format that stays out of scope:
   no schema, no migration, no integrity, no `save()`/`load()`.
   `predict(newdata,
   ...)` delegates to `predict_core` (passing `self._predict_spec` and
   `self.offset_was_nonzero`). **`predict_matrix(newdata)` does NOT go through
   `predict_core`** — `predict_core` returns *finished predictions* via
   `finish_prediction` (link-inverse applied), **not** the design matrix
   (design §6). `predict_matrix` returns the constrained `X_p` by delegating
   straight to `self._predict_spec.build_predict_matrix(newdata)`, exactly as
   `GAMResults.predict_matrix` does today (`results.py:327`).

4. **`jaxgam/inference/__init__.py`** — export **`GAMPredictor`** only.
   `predict_core` / `finish_prediction` stay private to `_core.py`
   (design §11.1).

5. **Tests** — `tests/test_inference/test_predictor.py` (new package;
   `@parametrize` + `_AssertCollector`), targeting design §12.2 invariants
   1, 1b, 4, 6, 7 (~7 collected):
   - **Predict-equivalence (`STRICT`)** — a directly-constructed
     `GAMPredictor.predict(newdata, se_fit=True)` equals the source
     `GAMResults.predict(newdata, se_fit=True)`, byte-identical, over the
     zoo: gaussian `s(x)`, binomial factor-by, poisson + offset, NB,
     non-default-link, `te()`, `ti()` × `pred_type`. In the **same
     collector**, also assert **`GAMPredictor.predict_matrix(newdata)` ==
     `res.predict_matrix(newdata)`** (≡ `setup.build_predict_matrix`) — the
     matrix seam, not just the finished prediction (the `predict_matrix`
     delegates to `build_predict_matrix`, not `predict_core`).
   - **Direct-R parity (`STRICT`/`MODERATE`, §12.2.1b)** — predictor
     **and** a pickle→unpickle→predict round-trip vs R `predict.gam` for
     `te()`, factor-by, NB, and **`Gamma(link="log")`** (the `gamma_log`
     bridge key). **Skip-guard `bridge.mode != "rpy2"`** per the
     load-bearing note above. Offset & locally-defined custom links are
     excluded (covered by equivalence + pickle).
   - **Pickle round-trip (`STRICT`)** — (a) stdlib `pickle` a
     built-in-family/link core → predict byte-identical, link survives,
     `coefficients`/`Vp` still read-only; (b) `cloudpickle` a
     local-custom-link core, same — and assert stdlib `pickle` **fails**
     that local-link case; (c) **version stamp** — a same-version round-trip
     emits **no** warning, and a blob whose `_jaxgam_version` is rewritten to
     a fake version **warns** on load (design §5.3, §8). Fold all three into
     one `_AssertCollector`.
   - **Read-only arrays** — `coefficients`/`Vp` raise on in-place write,
     both after construction and after an unpickle round-trip.
   - **Family snapshot independence + final theta** — the snapshot is not
     the registry singleton (`predictor.family is not get_family(name)`);
     mutating the registry instance does not affect it; for NB,
     `predictor.family.theta` is the **fitted** theta (post-`put_theta`).

### Files touched

- Add: `jaxgam/inference/__init__.py`, `_core.py`, `predictor.py`
  (**no `_protocol.py`** — design §5.1)
- Add: `tests/test_inference/__init__.py`,
  `tests/test_inference/test_predictor.py`

### Validation

- `make test-cov` passes; the JAX-importing family snapshot loads fine on
  the CPU/NumPy predict path (design §8 tradeoff).
- New tests collected count up by **~7** (per the §12.3 footprint).
- Coverage of every `jaxgam/inference/*` file ≥ 80%.
- The direct-R parity tests **skip cleanly** when `bridge.mode != "rpy2"`
  and **run** in Docker (rpy2 present).

### Exit criteria

- `GAMPredictor` predicts byte-identically to `GAMResults` across the zoo
  and matches R `predict.gam` for the four direct-R cases.
- Pickle (stdlib, built-in) and cloudpickle (local link) round-trips
  preserve predictions and re-freeze the two arrays; stdlib-pickle fails
  the local-link case as asserted; a version-mismatched blob warns.
- `coefficients`/`Vp` are read-only pre- and post-unpickle; the family
  snapshot is independent with final theta.
- **Neither** `predict_core` **nor** `finish_prediction` is importable from
  `jaxgam.inference`'s public surface (both stay private to `_core.py`), and
  **`jaxgam/inference/_protocol.py` does not exist**.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit E.

---

## Commit E — `result` Mode + Two Result Types + Family Snapshot + Exports

**Goal:** Wire the user-facing surface: the keyword-only
`@overload`ed `fit(..., result=...)`, the two result types
(`GAMInferenceResult` composing a `GAMPredictor`; reshaped `GAMResults`
with `to_predictor()`), the single post-`put_theta` family snapshot in
`_from_fit`, the internal renames that free the public `result` kwarg, and
the public exports.

**Design reference:** §4, §5.3, §5.4, §5.5, §11.2, §12.2.

### What to do

1. **`jaxgam/results.py`** — add a small shared `_FitDiagnostics` base
   carrying **exactly** the scalar diagnostics + metadata enumerated in design
   §5.4/§5.5 — `edf`, `edf1`, `edf_total`, `deviance`, `null_deviance`,
   `score`, `scale`, `theta`, `smoothing_params`, `converged`, `n_iter`,
   `convergence_info`, `method`, `lambda_strategy`,
   `execution_path`, `n` (all `O(1)`/`O(p)`, retained in **both** modes;
   **`formula` is deliberately NOT here** — `GAMPredictor` owns it (design
   §5.3) and the lean type reads it as a property; putting it in the shared
   base as well would make it a **ninth stored duplicate** in the design whose
   DRY headline is deleting eight. `GAMResults` keeps its own stored
   `formula` — it has no predictor until `to_predictor()`;
   `edf`/`edf1`/`smoothing_params` are `np.ndarray`, the rest scalars/`str`/
   `None` — `results.py:61–102`; the design's `edf*` shorthand expands to
   `edf`/`edf1`/`edf_total`; **`n` lands here from B1's deferred `setup.n_obs`
   alias** — the one `setup.*` duplicate that becomes `_FitDiagnostics` scalar
   metadata, not a `setup`-property, §5.5). **All
   three — the `_FitDiagnostics` base and both concrete result types — are
   `@dataclass(frozen=True)`**, preserving today's frozen `GAMResults`
   (`results.py:40`; design §5.4). The base must be frozen too: a frozen
   dataclass cannot inherit from a non-frozen one (Python raises `TypeError`),
   so once `_FitDiagnostics` carries the shared scalar fields it is frozen as
   well. Then make
   **`GAMInferenceResult`** (composes `_predictor: GAMPredictor`;
   `coefficients`/`Vp`/`family`/**`formula`** as properties → `_predictor`;
   **`smooth_info`/`term_names` as properties → `_predictor._predict_spec`**
   (design §5.4 — `SmoothInfo` is all `str`/`int`/`bool`, `design.py:68-77`,
   and the spec already carries `smooth_info`, so this costs **zero bytes**;
   without it `edf` is a bare unlabeled array on the type built for production
   logging);
   **`predict(newdata)` and `predict_matrix(newdata)` delegate to
   `_predictor`; `to_predictor()` returns `self._predictor`** — both required
   on the surface per design §4.3/§5.4; **no** `summary`/`plot`; `predict`
   requires `newdata`; `__repr__` hints to refit with `result="full"`) and the
   **reshaped `GAMResults`**
   (`to_predictor()` builds a `GAMPredictor` on demand from `setup`'s lazy
   `PredictSpec` + the family snapshot + `offset_was_nonzero`; the
   predictor's defensive copy means this never freezes the result's own
   arrays). **`offset_was_nonzero` is not a stored `ModelSetup` field** —
   `ModelSetup` carries only `offset` (`design.py:97`), so **compute** it from
   the live `setup` as `setup.offset is not None and not
   np.allclose(setup.offset, 0.0)` (the exact condition inlined today at
   `results.py:281`). The **lean path** computes the same flag in `_from_fit`
   **before discarding `setup`** and stores it on the predictor, so the
   external-offset warning survives the setup-drop; `to_predictor()` on a full
   result derives it from the still-live `setup`.
   - **Route `GAMResults.predict()` through the shared finish path** (design
     §6, §9 DRY — what makes the "one `finish_prediction`" claim true; this is
     the *DRY* half of the PR, not just the memory half): the **new-data**
     branch becomes `predict_core(setup._lazy_predict_spec(),
     self.coefficients, self.Vp,
     self.family.link, newdata, pred_type=…, se_fit=…, offset=offset,
     offset_was_nonzero=(setup.offset is not None and not
     np.allclose(setup.offset, 0.0)))` — **pass the spec, not `setup`**
     (round 10 cut the Protocol, design §5.1; `setup.build_predict_matrix`
     reaches the same cached spec, so output is byte-identical) — the same
     derived condition as
     `results.py:281` (no precomputed flag exists on `ModelSetup`,
     `design.py:97`); the **self-prediction**
     branch (`newdata=None`) keeps the cached `eta = linear_predictor.copy()`
     and `X_p = setup.X if se_fit else None`, then ends in
     `finish_prediction(eta, X_p, family.link, Vp, pred_type=…, se_fit=…)`. This
     **replaces the inline `linkinv` + `sqrt(rowSums((X_p @ Vp) * X_p)) *
     |mu_eta|` block in `GAMResults.predict()` (`results.py:271–311`)** — output
     must stay **byte-identical** (existing `tests/test_predict/` + the
     CoefficientMap roundtrip are the gate); the external-offset warning moves
     into `predict_core` (design §6) and must still fire for the offset case.
   - **`_from_fit` takes the family snapshot** (`copy.deepcopy`, **after**
     `put_theta` finalizes theta) — the **single owner** of the
     predictor/result family snapshot for **every** fit (design §5.3).
     Distinct from `api.py:122`'s pre-fit NB-only deepcopy, which is
     **unchanged**.
   - **Rename `_from_fit`'s `result: NewtonResult` param
     (`results.py:111`) → `fit_result`**, and thread the new mode as a
     separate `result_mode` argument (design §11.2, §5.5 — frees the
     public `result` name from the internal optimizer-output name).
     **This rename breaks existing direct callers:** `tests/test_results.py`
     constructs results via `GAMResults._from_fit(result=…)` at **`:64`** and
     **`:190`** (keyword arg). Update both to `fit_result=…` (the design
     favors the clean rename over a temporary keyword alias) — add the file to
     Files-touched below.

2. **`jaxgam/api.py`** — keyword-only `result="full"` on `GAM.fit()` with
   the two `@overload`s (design §4.1); validate the value (`ValueError`
   for anything but `"full"`/`"inference"`); thread `result_mode` into
   `_from_fit`. **Rename the local optimizer output `result`
   (`api.py:136`) → `fit_result`.** The NB pre-fit `deepcopy`
   (`api.py:122`) is **unchanged** (it is not the predictor snapshot).

3. **`jaxgam/__init__.py`** — add `GAMInferenceResult` and `GAMPredictor`
   to imports + `__all__` (today only `GAM`/`GAMResults` are exported).

4. **Tests** — `tests/test_result_mode.py` (new; `_AssertCollector`,
   **~6 collected** — §12.2.1/1b, .2, .3, .5, .6; the +2 over the design's
   rounded ~4 are the actual-path wiring tests, which live here because they
   need `fit(result="inference")`):
   - **Banned state absent** under `result="inference"` (hard gate): no
     `setup`/`fitted_values`/`linear_predictor`/`training_data`, and **no
     `X`/`y`/`weights`/`offset`** (on `GAMResults` these are `@property` reads
     of `setup`, so the lean type — which holds no `setup` — must not expose
     them; assert each is absent via `hasattr`); a recursive walk finds no
     non-`None` `_X`/`_S`/`_E_knot` and empty `_penalties`; transforms
     `_XP_list`/`_Z_list`/`_Xu`/`_UZ` **present**. (Banned owners/attributes —
     **not** "no dim == n", per the load-bearing note; the full banned set
     matches design §3.1/§7.1 — `X`/`y`/`weights`/`offset` included.)
   - **Type / surface**: `fit(result="inference")` returns a
     `GAMInferenceResult` instance (runtime `isinstance`) with no
     `summary`/`plot` and a `newdata`-required `predict`; **`predict_matrix(
     newdata)` and `to_predictor()` ARE present on the lean type** —
     `predict_matrix` returns the same matrix as a `result="full"` fit, and
     `to_predictor()` returns the composed predictor
     (`inf_result.to_predictor() is inf_result._predictor`, design §5.4);
     `fit()`/`fit(result="full")` returns `GAMResults`; invalid `result`
     raises `ValueError`; **every `_FitDiagnostics` field reads identically on
     both types** — parametrize over the concrete fields (`edf`, `edf1`,
     `edf_total`, `deviance`, `null_deviance`, `score`, `scale`, `theta`,
     `smoothing_params`, `converged`, `n_iter`, `convergence_info`,
     `method`, `lambda_strategy`, `execution_path`, `n` — `edf*` is the
     design's shorthand for `edf`/`edf1`/`edf_total`, not an attribute),
     comparing the **array** fields (`edf`/`edf1`/`smoothing_params`) with
     `np.testing.assert_array_equal` and the scalars/`str` with `==` (guard
     `theta is None` for non-NB). **Extend the same parametrization to the
     three property-backed reads — `formula`, `smooth_info`, `term_names`** —
     which are *not* `_FitDiagnostics` fields on the lean type but must still
     compare equal across modes (design §5.4/§5.5); `smooth_info`/`term_names`
     are what make `edf` interpretable, so assert
     `len(smooth_info) == len(edf)` on the lean result as well. **Both result
     types stay `@dataclass(frozen=True)`** — assigning to a field on each
     raises `FrozenInstanceError` (assert on both). Add a
     `typing.assert_type` block for the `@overload` (teeth only under a
     checker — §12.5).
   - **Actual-path predict-equivalence + direct-R (the fit-mode wiring gate —
     design §12.2.1, §12.2.1b):** fit the **same** model both ways and assert
     `full.predict(newdata, se_fit=True) == inference.predict(newdata,
     se_fit=True)` byte-identical over the zoo. This exercises
     `_from_fit(result_mode="inference")` end-to-end — Commit D's
     directly-constructed predictor **bypasses** it, so a wiring bug there
     would pass undetected. Then re-run the **direct-R `predict.gam` gate on the
     actual `GAMInferenceResult.predict`** (and its pickle round-trip) for
     `te()`, factor-by, NB, and `Gamma(link="log")`, **skip-guarding
     `bridge.mode != "rpy2"`** (load-bearing note). This is the *wiring* gate
     (does `_from_fit` compose the right predictor?), distinct from Commit D's
     predictor-*correctness* gate; both live in the new-behavior files, so
     `tests/test_validation_matrix.py` stays the sole owner of final-model R
     parity (no duplication). **Also assert the offset warning survives the
     setup-drop:** `fit(result="inference")` an offset model, then
     `inference_result.predict(newdata)` with **no** `offset=` warns (the
     `offset_was_nonzero` flag was computed in `_from_fit` before `setup` was
     discarded), matching the full-result behavior.
   - **Family snapshot via the real `_from_fit`** (design §5.3, §12.2.6): for
     **both** `fit(result="full").to_predictor()` and
     `fit(result="inference")`, assert the predictor's `family` is **not** the
     registry singleton (`is not get_family(name)`), mutating the registry
     instance does not leak in, and for **NB** `predictor.family.theta` equals
     the **fitted** theta (proving the snapshot is taken **after**
     `put_theta`). Commit D's snapshot test deep-copies the family by hand on a
     constructed predictor, so it does **not** exercise `_from_fit` — this is
     the authoritative owner-of-snapshot test.
   - **Aliasing / ownership** (`to_predictor()` on `GAMResults`, §12.2.5):
     afterwards `setup`'s smooths still hold
     `_X`/`_S`/`_penalties`/`_E_knot`; the predictor's spec smooths have
     them dropped but retain `_XP_list`/`_Z_list`;
     `predictor.coefficients`/`Vp` are **distinct objects** from the
     result's, and the result's arrays are **not** frozen by the call. Also
     assert the **lazy spec was unbuilt** on the result until this call —
     `setup`'s spec-cache is `None` after `fit(result="full")` and non-`None`
     only after `to_predictor()`/predict (goal #3, design §5.2).
   - **Retained-bytes assertion**: the `result="inference"` footprint is
     materially smaller than `result="full"` for tensor (drops `_penalties`)
     and GP (drops `setup.X`, `_X`, `training_data`, `_S`) models — record the
     figures in `BASELINE.md` against the **post-B0** "before", not Commit A's
     (Metric #3). GP's `_E_knot` is already gone by B0; crediting it here would
     overstate the design's win by ~32 MB at `n=5000` (design §1.1).

### Files touched

- Modify: `jaxgam/results.py` (`_FitDiagnostics`, `GAMInferenceResult`,
  reshaped `GAMResults` + `to_predictor()`, `_from_fit` snapshot + param
  rename + `result_mode`)
- Modify: `jaxgam/api.py` (`@overload`ed keyword-only `result` + value
  validation + `fit_result` rename)
- Modify: `jaxgam/__init__.py` (export the two new types)
- Modify: `tests/test_results.py` (update the two direct
  `GAMResults._from_fit(result=…)` calls at `:64`/`:190` → `fit_result=…`
  for the clean rename)
- Add: `tests/test_result_mode.py`
- Modify: `docs/production_api/BASELINE.md` (inference footprint deltas)

### Validation

- `make test-cov` passes. The full Hard Allow-List is intact — every
  `result="full"` **statistical output and existing post-estimation result is
  byte-identical** to today. (The one *intentional* difference: `_from_fit`
  now hands every result a `copy.deepcopy` family **snapshot**, so
  `results.family` is no longer the registry singleton's object identity —
  design §5.3, §12.2.6. No existing test asserts family object identity
  (verified), so this breaks nothing; it is an object-identity change, not a
  statistical one.)
- New tests collected count up by **~6** (the +2 over the design's rounded
  ~4 are the actual-path wiring tests — §12.2.1/1b/.6 — that can only run once
  `fit(result="inference")` exists).
- Coverage of the new/modified `results.py` paths ≥ 80%.
- `from jaxgam import GAMInferenceResult, GAMPredictor` works.

### Exit criteria

- `fit(result="inference")` returns a lean `GAMInferenceResult` holding
  none of the banned state; `fit(result="full")` is unchanged.
- `to_predictor()` round-trips and does not freeze/alias the result's
  arrays; the family snapshot is the single post-`put_theta` owner.
- Invalid `result` raises `ValueError`; both new types are exported.
- Retained-bytes reduction recorded vs the Commit-A baseline.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit F.

---

## Commit F — Documentation + Final Sweep

**Goal:** Document the new surface, mark the design implemented, and run
the closing verification before the user opens the PR.

**Design reference:** §8, §10.3, §12.5, §13 (commit F).

### What to do

1. **`docs/api.md`** — document `result="full"|"inference"` on `fit()`,
   the two return types, and `GAMInferenceResult`'s narrower surface (no
   `summary`/`plot`; `predict` requires `newdata`; scalar diagnostics plus
   `formula`/`smooth_info`/`term_names` retained). Document `GAMPredictor` +
   `to_predictor()`. State the pickle
   contract: **"stdlib `pickle` is the default for same-version /
   transient handoff; `cloudpickle` is required only for locally-defined
   custom links/families; neither is a durable cross-version format"** —
   and note that loading a predictor pickled by a different jaxgam version
   **warns** (design §8). Do **not** document a `save()`/`load()` or any
   serialization format (out of scope). State plainly that `result="inference"`
   reduces **retained** memory, not **peak** fit-time memory (design §1.1) —
   users will otherwise reach for it to survive a large-`n` fit.

2. **`docs/quickstart.md`** — add a short "lean inference result" example:
   `model.fit(df, result="inference")` → `predict(newdata)` →
   `pickle.dumps(res.to_predictor())`.

3. **`docs/index.md`** — if it carries an API/result-type table, add
   `GAMInferenceResult` / `GAMPredictor` one-liners.

4. **`docs/production_api/design.md`** — **reconcile the spec with the
   as-built code, then** change the status header from "Proposed (design only)"
   to "Implemented" with the completion date. Sections already corrected during
   planning, to be confirmed in sync: **§6** names the **five-helper**
   transitive closure (not three); **§8** calls `cloudpickle` the `dev`
   **optional extra** (not a "group"); **§5.1** documents the cut Protocol and
   must match reality (no `_protocol.py` in the tree); **§1.1/§2.1** describe
   `_E_knot` as a B0 dead-store fix rather than a mode win; **§5.4/§5.5** list
   `formula`/`smooth_info`/`term_names` as lean-type properties. Sweep for any
   other section that drifted from what shipped (helper names, file paths,
   field lists, line citations) and fix it — design.md is the authoritative
   spec.

5. **(Optional) `make typecheck`** — the repo configures no type checker
   (ruff + vulture only), so the `@overload`/`assert_type` is editor-aid
   only. Optionally add a `make typecheck` target running a checker over
   the new types (design §12.5); note it in `docs/api.md` if added.

6. **Final sweep** (the closing verification — no design commit follows):
   - Re-run the Commit-A baseline measurements; record final deltas
     (test count, coverage, wall-clock, retained bytes) in `BASELINE.md`.
   - **Ownership grep** — confirm new-behavior R parity lives only in
     `tests/test_inference/` and `tests/test_result_mode.py`, and that
     `tests/test_validation_matrix.py` is unchanged (final-model R parity
     is still its sole responsibility — no duplication).
   - **Hard Allow-List spot check**:
     ```sh
     uv run pytest tests/test_predict tests/test_summary tests/test_plot \
                    tests/test_post_estimation.py tests/test_validation_matrix.py -q
     ```
   - Confirm `jaxgam/__init__.py` exports exactly
     `GAM`, `GAMResults`, `GAMInferenceResult`, `GAMPredictor`.

### Files touched

- Modify: `docs/api.md`, `docs/quickstart.md`, possibly `docs/index.md`
- Modify: `docs/production_api/design.md` (status → Implemented)
- Modify: `docs/production_api/BASELINE.md` (final deltas)
- Possibly: `Makefile` + `pyproject.toml` (optional `make typecheck`)

### Validation

- `make test-cov` passes (docs-only code-wise; run anyway to confirm
  nothing was accidentally touched).
- Manually scan rendered docs for typos / broken links; confirm no `m=`-
  style or `save()`/`load()` example slipped in.
- Hard Allow-List spot check passes individually.

### Exit criteria

- Public docs describe the `result` mode, two result types,
  `GAMPredictor`, and the precise pickle contract.
- Design status is "Implemented"; baseline deltas recorded.
- No R-parity duplication; Hard Allow-List intact.
- **Agent stops and hands off to user.** The user reviews, commits, and
  opens the single PR against `main`.

---

## Risk Management

### Risks specific to this feature

1. **Builder move changes prediction output (Commit C).** Mitigation: it
   is a pure code move + a delegating lazy spec — no algorithm change. The
   gate is "`tests/test_predict/` passes byte-identically." If any predict
   test changes output, the move is wrong — investigate before handoff.

2. **A smooth nulls a cache its `predict_matrix` actually reads.**
   Mitigation: every `copy_for_prediction()` override is justified
   line-by-line against that type's `predict_matrix` (design §2.1); the
   per-type copy test asserts `predict_matrix(newdata)` is unchanged after
   the copy; the **registry audit** blocks any unaudited new smooth
   (§12.2.8). Since Commit C also routes the **full** path through the
   copies, the entire existing `tests/test_predict/` suite becomes a
   regression gate on this — a mistake here fails loudly and immediately,
   not only in the new lean-path tests.

2b. **Crediting the design with B0's memory win.** Mitigation: three
   measurement points (A-full, B0-full, E-inference; Metric #3). The number
   this PR may claim is **B0-full − E-inference**. `_E_knot` is a dead store
   that `result="full"` leaked too, and it is 89% of the GP baseline — quoting
   A-full − E-inference would overstate the design by ~32 MB at `n=5000`.
   Design §1.1, §2.1.

3. **No-retention test written against array shape, not owners.**
   Mitigation: the load-bearing note + §12.2.2 mandate banned
   owners/attributes; knots are legitimately `≤ n` rows, so a `dim == n`
   test gives false failures. The test asserts the **full banned owner set**
   absent —
   `setup`/`X`/`y`/`weights`/`offset`/`fitted_values`/`linear_predictor`/`training_data`
   — plus `_S`/`_E_knot is None` and empty `_penalties` (recursively), never a
   shape.

4. **`to_predictor()` freezing or aliasing the live result's arrays.**
   Mitigation: `GAMPredictor.__post_init__` **defensively copies**
   `coefficients`/`Vp` before `write=False`; the aliasing/ownership test
   (§12.2.5) asserts the predictor's arrays are distinct objects and the
   result's arrays remain writable. Without the copy, `to_predictor()`
   would freeze `GAMResults`' own arrays — the bug this guards.

5. **Family snapshot has the wrong theta or aliases the registry.**
   Mitigation: the snapshot is taken in `_from_fit` **after**
   `put_theta` (single owner, §5.3); the snapshot-independence test
   asserts it is not the registry singleton and, for NB, carries the
   fitted theta. Keep it distinct from `api.py:122`'s pre-fit NB-only
   deepcopy (do **not** move the snapshot into `api.py`).

6. **Direct-R parity case unrunnable / hard-erroring outside rpy2.**
   Mitigation: use the existing `gamma_log` bridge key (not an unwired
   `poisson_identity`) and **skip-guard `bridge.mode != "rpy2"`** like
   `test_gaussian_process.py:93`. Do not extend the subprocess family map
   or add a bridge offset argument (RBridge is rpy2-only).

7. **Internal equivalence mistaken for R parity.** Mitigation: the lean
   and full paths share one builder, so they can agree with each other yet
   drift from mgcv. §12.2.1b adds the **direct R `predict.gam`** comparison
   (and a pickled-path comparison) so a shared-but-wrong predict path is
   caught.

8. **Public `result` kwarg colliding with internal `result` names.**
   Mitigation: rename the optimizer-output local (`api.py:136`) and the
   `_from_fit` param (`results.py:111`) to `fit_result`, and thread the
   mode as `result_mode` (design §11.2). Done in Commit E.

9. **Migrating `GAMResults.predict()` to the shared finish path changes
   output (Commit E).** Mitigation: the new-data branch calls the same
   `build_predict_matrix` and the byte-identical `finish_prediction` SE
   formula; the self-prediction branch keeps the cached `eta` / `X_p =
   setup.X` path verbatim; the external-offset warning moves into
   `predict_core` unchanged. Gate: `tests/test_predict/` + the CoefficientMap
   roundtrip pass byte-identical and the offset warning still fires. If any
   predict test changes output, the migration is wrong — fix before handoff.

### Out-of-scope guardrails (do not implement here)

`save()`/`load()` / a serialization format / versioning / integrity; a
JAX-free import path; distributional sampling / intervals; a point-only
(drop-`Vp`) mode; the **null-deviance DRY rewrite** (`results.py:549–577`
— leave **exactly** as-is so `null_deviance` stays byte-identical) and the
**`summary()` CQS change** (`results.py:348–350`) — both **deferred** to
their own PRs (design §3.2, §10.3). If a code path seems to need one of
these, stop and confirm with the user before opening any new commit.

### Rollback plan

Each commit lands one logical change on a single working branch and can be
reverted with `git revert <sha>`. Most touch disjoint file sets. Commit C
(the builder move + `copy_for_prediction`) is the most structural; D and E
depend on it, so prefer fixing C in place over reverting. B1, A, and F are
cleanly revertable in isolation.

---

## Sequencing Summary

All commits land on a single working branch off `main`. The agent
executes one unit at a time, validates with `make test-cov`, hands off;
the user commits manually before triggering the next unit.

```
Phase 0 — Pre-flight
Commit A    Baseline capture + cloudpickle dev dep            ── docs/production_api/BASELINE.md, pyproject.toml
        │   (test counts, coverage, retained-bytes "before";
        │    cloudpickle in dev extra only)
        │
Phase 1 — Zero-risk fixes
Commit B0   GP _E_knot dead-store fix                         ── smooths/gaussian_process.py,
        │   (one line in setup(); restructure the R-parity       tests/test_smooths/test_gaussian_process.py,
        │    assertion; re-measure — this is the TRUE            BASELINE.md
        │    "before" for the result mode, not Commit A)
        │
Commit B1   setup.* duplicates → @property; drop dead guard   ── results.py, test_results.py
        │   (mechanical; output byte-identical)
        │
Phase 2 — Phase-1 prediction state (structural core)
Commit C    PredictSpec + builder move + copy_for_prediction  ── formula/predict_matrix.py, formula/design.py,
        │   (lazy spec on frozen ModelSetup; per-type cache       smooths/{base,tensor,gaussian_process,by_variable}.py,
        │    drop + registry audit; test_predict/ unchanged)      tests/test_smooths/
        │
Phase 3 — Lean core + user-facing surface
Commit D    inference/: predict_core + finish + GAMPredictor  ── jaxgam/inference/*, tests/test_inference/
        │   (NO _protocol.py — predict_core typed on
        │    PredictSpec; Vp required; defensive-copy
        │    read-only + version stamp; predict-equivalence +
        │    direct-R + pickle/cloudpickle tests; PRIVATE core)
Commit E    result mode + two types + snapshot + exports      ── results.py, api.py, __init__.py, test_result_mode.py
        │   (@overloaded keyword-only fit(); _from_fit
        │    snapshot post-put_theta; GAMResults.predict→shared
        │    predict_core/finish; actual-path full==inference +
        │    direct-R wiring tests; fit_result/result_mode
        │    renames; export the two new types)
        │
Phase 4 — Docs + finalize
Commit F    Docs + final sweep                                ── docs/api.md, docs/quickstart.md, design.md (status),
        │   (pickle contract; status→Implemented; ownership      BASELINE.md
        │    grep; Hard Allow-List spot check)
        │
User opens single PR against main.
```

Commit A precedes everything (cloudpickle must be available for D's
tests, and A is the only record of today's pre-B0 footprint). **B0 must
precede C** — C's GP `copy_for_prediction()` override is written as
defense-in-depth *given* that setup already nulls `_E_knot`, and B0's
re-measurement is the honest "before" every later memory claim is quoted
against. B1 is independent and could move, but lands early as low-risk
warm-up. C must precede D and E (both consume `PredictSpec` /
`build_predict_spec`). D precedes E (E composes `GAMPredictor` and wires
`to_predictor()`; every test needing the real fit-mode wiring —
`to_predictor()` aliasing, `full`==`inference` equivalence, direct-R on the
actual `GAMInferenceResult`, and the `_from_fit` snapshot — lands in E).

---

## Definition of Done

Met when the user opens the single PR for this feature:

- **`GAM.fit(..., result=...)`** is keyword-only, `@overload`ed, and
  value-validated: `result="full"` (default) returns today's
  `GAMResults`; `result="inference"` returns a lean `GAMInferenceResult`.
  An invalid value raises `ValueError`.
- **`GAMInferenceResult`** retains **none** of `setup`, `X`, `y`,
  `weights`, `offset`, `fitted_values`, `linear_predictor`,
  `training_data`, or any smooth's `_X`/`_S`/`_penalties`/`_E_knot`
  (recursively) — verified by the banned-owners test (§12.2.2), **not** a
  shape test. It **keeps** the predict transforms and the cheap scalar
  diagnostics, exposes `predict_matrix(newdata)`, `to_predictor()`
  (→ its `_predictor`), and the zero-byte metadata reads
  `formula`/`smooth_info`/`term_names` (so `edf` is interpretable), and has
  no `summary`/`plot` (its `predict` requires `newdata`).
- **`GAMPredictor`** is a frozen, picklable core owning
  `coefficients`/`Vp` (defensively copied, `write=False`, re-frozen on
  unpickle), a post-`put_theta` family snapshot, `offset_was_nonzero`,
  `_jaxgam_version` (warns on cross-version unpickle), and a concrete
  `PredictSpec`. `Vp` is required.
- **`_E_knot` fixed at the source (B0)**: `grep -rn "_E_knot" jaxgam/` shows
  it declared, assigned, and reset to `None` in `setup()` — read by nothing.
  The GP `E`-vs-R `STRICT` parity assertion still runs.
- **Memory win demonstrated, honestly attributed**: `result="inference"`
  retained bytes are materially below `result="full"` for tensor (drops
  `_penalties`) and GP models, quoted against the **post-B0** baseline. The
  A → B0 drop is reported separately as the dead-store fix, not as part of
  this design's win.
- **No numerics changed**: every prediction / SE / EDF / scale / deviance
  (incl. `null_deviance`) is byte-identical to today; the entire Hard
  Allow-List passes unchanged.
- **One Phase-1 builder, one Phase-3 finish, no seam**: `build_predict_matrix`
  over a single concrete `PredictSpec`; `finish_prediction`/`predict_core`
  shared by both result types by being **typed directly on `PredictSpec`** —
  **`GAMResults.predict()` routed through it** (new-data →
  `predict_core(setup._lazy_predict_spec(), …)`; self-prediction → the
  cached-`eta` `finish_prediction` branch). Because `predict_core` returns
  *finished predictions*, **`predict_matrix()` delegates to
  `build_predict_matrix`, not `predict_core`**. Both `predict_core` and
  `finish_prediction` stay private, and **`_protocol.py` was never created**
  (design §5.1).
- **`copy_for_prediction()`** implemented per type (base / tensor / GP /
  by-variable), each justified against its `predict_matrix`, with the
  **registry audit test** blocking any unaudited new smooth.
- **Direct-R parity** for the inference and pickled paths passes at
  STRICT/MODERATE for `te()` / factor-by / NB / `Gamma(link="log")` — on
  **both** the Commit-D constructed predictor and the **actual Commit-E
  `GAMInferenceResult.predict`** (the `_from_fit` wiring) — with the
  `bridge.mode != "rpy2"` skip-guard; offset and locally-defined custom
  links are covered by internal equivalence + pickle round-trip, not
  direct R.
- **Pickle contract**: stdlib `pickle` works for built-in families/links;
  `cloudpickle` (dev-only) covers locally-defined custom links; neither is
  a durable cross-version format.
- **DRY**: **seven** `setup.*` duplicate fields
  (`X`/`y`/`weights`/`offset`/`coef_map`/`smooth_info`/`term_names`) are
  `@property` reads; the eighth, `n`, is retained scalar metadata in
  `_FitDiagnostics` (shared by both result types); the dead `hasattr` guard is
  gone; and **no new duplicate was introduced** — `formula` lives on
  `GAMPredictor` only, read as a property by the lean type. (Null-deviance DRY
  and `summary()` CQS remain **deferred** — untouched.)
- **Exports**: `jaxgam/__init__.py` exports `GAM`, `GAMResults`,
  `GAMInferenceResult`, `GAMPredictor`.
- **Consolidation held**: ~14–16 new collected tests (not 50+ from
  per-assertion enumeration); every shared-fixture test uses
  `_AssertCollector`; the predict zoo is `@parametrize`d.
- **No R-parity duplication**: new-behavior R parity lives only in
  `tests/test_inference/` + `tests/test_result_mode.py`;
  `tests/test_validation_matrix.py` is unchanged.
- `make test-cov` passes on the final commit with ≥ 80% coverage overall
  and on every new file.
- `docs/production_api/design.md` status is "Implemented";
  `docs/api.md` + `docs/quickstart.md` document the new surface.
- All seven commit slots (A, B0, B1, C, D, E, F) are present in the branch
  history as individual commits authored by the user.
