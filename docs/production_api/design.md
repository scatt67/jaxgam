# Robust API Redesign: `result` Mode + Lean Inference Core

- **Status:** Proposed (design only)
- **Design Date:** 2026-06-05 (revised 2026-07-28 over ten review rounds)
- **Target Branch:** `production-api`
- **Builds on:** [`docs/refactor_gam_api/design.md`](../refactor_gam_api/design.md) (spec/results split)
- **Release context:** jaxgam is **pre-release** (`1.0.0a1`, unreleased). No
  downstream users to keep compatible, so this makes the **clean change** — no
  deprecation cycle, no `__getattr__` forwarding.

> **Scope (round 1).** An earlier draft bundled an MLOps serving layer
> (versioned/integrity artifact, `save()`/`load()`, a JAX-free import guarantee,
> family rehydration from a key, posterior sampling). **Removed.** The design
> keeps two goals: **(1) a robust, SOLID/DRY API** and **(2) a result that does
> not crush memory**, with the inference core a plain **picklable** object.
> Durable/cross-version persistence is out of scope (§8).
>
> **Lifecycle/layering (rounds 2–3).** The mode is a `fit()` option, not model
> spec; **Phase 1 owns prediction-state construction and cache-dropping**,
> **Phase 3 only consumes**; the predictor is **mode-specific** (never eagerly
> cloned); the no-retention invariant is framed by **banned owners/attributes**,
> not array shapes; pickle is **same-version/transient**.
>
> **Interface & correctness (round 4).** `result="inference"` returns a **distinct
> narrower type** (`GAMInferenceResult`) via `@overload`, not a partially-disabled
> `GAMResults`. Stripping uses a **`copy_for_prediction()`** hook; Phase 3 predict
> is **CPU/NumPy**; the mode keyword is keyword-only.
>
> **Tightening (round 5).** Two result types; `Vp` required; defensive-copy
> read-only on `coefficients`/`Vp`; null-deviance DRY rewrite **deferred** (left
> byte-identical by not touching it).
>
> **Honesty pass (round 6).** Nine fixes after a parity-focused review:
> **(1)** `copy_for_prediction()` also drops **penalty caches** (`_S`/`_penalties`)
> — *verified*: no `predict_matrix` reads them (`tprs.py:599`, `tensor.py:358–362`,
> `cubic.py:419`, `random_effects.py:222`), and tensor `_penalties` are
> `O(n_coefs²)` (often the dominant cost). The round-5 "drop `_X`, keep everything
> else" was wrong (§7). **(2)** `offset_was_nonzero` is an **explicit
> `GAMPredictor` field**, not pulled off the one-method Protocol (§5.3, §6).
> **(3)** the public read-only guarantee is **downgraded to exactly
> "`coefficients` and `Vp` are read-only"** (§5.3, §9). **(4)** `PredictSpec` is
> **owned as the explicit Phase-1→Phase-3 boundary value object** (`_predict_spec`,
> reversing round-5's Protocol-typed field — pickle exposes the concrete type
> anyway); the DIP claim is softened and the Protocol survives only as
> `predict_core`'s parameter seam (§5.1, §9). **(5)** `ModelSetup`'s `PredictSpec`
> is **lazy** (§5.2), and tests add **direct R parity** for the inference/pickled
> path (§12.2). **(6)** the family **snapshot has one owner** — `_from_fit`,
> *after* `put_theta` finalizes theta — distinct from api.py's pre-fit NB copy
> (§5.3, §11.2). **(7)** the user-facing flag is `result="full"|"inference"`
> (names the outcome). **(8)** the `summary()` CQS change is **deferred** out of
> scope (§9, §10.3). **(9)** the `@overload` typing claim is editor-aid-only
> unless `make typecheck` is added (§12.5).
>
> **Contract precision (round 7).** Eight fixes after an implementability review:
> **(1)** offset is **removed from the direct-R hard gate** — `RBridge` has no
> offset argument at fit or predict (`r_bridge.py:241,1068`), and offset parity is
> covered transitively (§12.2.1b). **(2)** GP overrides `copy_for_prediction()` to
> drop **`_E_knot`** — an `O(n_knots²)` setup-only cache not read by predict
> (`gaussian_process.py:200` vs `_build_design` `:157,246`); a real memory miss
> (§7.2). **(3)** `offset_was_nonzero` is **removed from `PredictSpec`** (the
> builder never reads it) — it lives only on `GAMPredictor`/`setup` (§5.2). **(4)**
> the public `result=` kwarg collides with the internal optimizer-output `result`
> (`api.py:136`, `results.py:111`); internal renames (`fit_result`, `result_mode`)
> are required (§11.2). **(5)** the lazy `PredictSpec` on a **frozen** `ModelSetup`
> (`design.py:79`) uses a `field(init=False)` cache + `object.__setattr__`, not
> `cached_property` (§5.2, §6). **(6)** stdlib **`pickle` is the public default**;
> `cloudpickle` (only for locally-defined custom links) stays dev-only —
> persistence is downstream's concern, not a runtime dep (§8). **(7)** `family` is
> described as a **snapshot**, not "read-only family" (§5.3). **(8)** `predict_core`
> stays **private**; `inference/` exports `GAMPredictor` (+ the
> `PredictMatrixBuilder` Protocol for typing) (§11.1).
>
> **Consistency & implementability polish (rounds 8–9).** No design changes —
> doc-internal fixes after two verification passes against the repo: the direct-R
> link example is now a **bridge-expressible** key — `Gamma(link="log")` (the
> existing `gamma_log`, `r_bridge.py:213`, rpy2-only with a `bridge.mode != "rpy2"`
> skip-guard per `test_gaussian_process.py:93`) — rather than the unwired
> `poisson(link="identity")` (§12.2.1b); **`_E_knot`** is listed in every
> banned-cache enumeration and `gaussian_process.py` added to the modified-file
> list (§11.2); `fitting/*` is correctly labeled **Phase 2** (§11.3); stdlib
> **`pickle`** is stated as the public default over `cloudpickle` (§8); and `_X` is
> restored to every drop/absence summary to match the hard gate (§4.2).
>
> **Subtraction pass (round 10).** Seven changes after an independent
> verification pass re-derived §2.1 from the code rather than from the doc. The
> central claim held (no `predict_matrix` reads `_S`/`_penalties`/`_E_knot`), but
> the rounds had been purely **additive** — round 10 removes rather than patches.
> **(1)** GP's **`_E_knot` is a dead store**, not a retention problem: it is
> assigned at `gaussian_process.py:200` and **read nowhere in `jaxgam/`**. It is
> **89% of the GP baseline footprint** (32.0 MB of 36.08 MB at `n=5000`), so
> `result="full"` leaks it too. Freed unconditionally at the end of GP `setup()`
> by a **separate one-line prerequisite commit (B0)**; the
> `copy_for_prediction()` override is retained as **defense-in-depth**, no longer
> load-bearing (§2.1, §7.2, §13, §14). **(2)** the **`PredictMatrixBuilder`
> Protocol is cut** — after §6's delegation, `ModelSetup` and `PredictSpec` are
> not two implementations but one implementation and a forwarder; `predict_core`
> is typed directly on `PredictSpec` (§5.1, §6, §9, §11.1). **(3)** the lean type
> **exposes `smooth_info`/`term_names`** (all-scalar metadata already carried by
> `PredictSpec`) so `edf` is interpretable off a `GAMInferenceResult` — zero
> bytes (§5.4, §5.5). **(4)** `formula` was about to become a **ninth stored
> duplicate** (on both `GAMPredictor` and `_FitDiagnostics`); the lean type reads
> it as a property (§5.4, §5.5). **(5)** the design reduces **retention, not
> peak** — stated explicitly rather than implied (§1.1, §3.2). **(6)** a
> `_jaxgam_version` stamp + `__setstate__` mismatch **warning** makes the
> same-version pickle contract *detectable* instead of silent (§8) — a guardrail,
> not the versioned format that stays out of scope. **(7)** `setup.penalties`
> (`CompositePenalty`, embedded `(total_p, total_p)`) added to the §2.1 inventory
> so the Metric-#3 delta is fully attributable. Citation drift fixed:
> `registry.py:25`→`:26`, `test_results.py:64`→`:63`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Current-State Analysis](#2-current-state-analysis)
3. [Design Goals and Non-Goals](#3-design-goals-and-non-goals)
4. [The `result` Mode and Two Result Types](#4-the-result-mode-and-two-result-types)
5. [Object Model](#5-object-model)
6. [The Shared Predict Path (Phase 1 builder + Phase 3 finish)](#6-the-shared-predict-path)
7. [Memory Reduction: `copy_for_prediction` and the Honest Invariant](#7-memory-reduction)
8. [Picklability](#8-picklability)
9. [SOLID / DRY Scorecard](#9-solid--dry-scorecard)
10. [Numerical and Correctness Considerations](#10-numerical-and-correctness-considerations)
11. [File Plan](#11-file-plan)
12. [Testing Strategy](#12-testing-strategy)
13. [Implementation Sequence](#13-implementation-sequence)
14. [Risks and Tradeoffs](#14-risks-and-tradeoffs)

---

## 1. Overview

### 1.1 The Problem: a training job pays for state it never uses

`GAM.fit(data)` returns a `GAMResults` (`jaxgam/results.py`) that retains
everything scaling with the **training** row count `n`: the design matrix
`setup.X` at `O(n·p)`, raw covariate columns in `training_data`, response/
weights/offset, in-sample `fitted_values`/`linear_predictor`, and per-smooth
training design caches `_X`. It *also* keeps per-smooth **penalty matrices**
(`_S`, tensor `_penalties`) that fitting needed but prediction does not. For
interactive analysis the full result is right — you want `.summary()`/`.plot()`.
For a job whose only output is a model to predict from, it is dead weight:
`setup.X` is typically larger than the input, `training_data` pins the input
frame alive, and a tensor's `_penalties` are `O(n_coefs²)`. None of it is needed
for `predict(newdata)`.

> **Not counted as motivation: GP's `_E_knot`.** The `O(n_knots²)` knot–knot
> kernel is 89% of the GP baseline footprint (32.0 MB of 36.08 MB at `n=5000`),
> but it is a **dead store**, not a retention problem — assigned at
> `gaussian_process.py:200` and **read nowhere in `jaxgam/`**, so `result="full"`
> leaks it just as badly. It is freed unconditionally at the end of GP `setup()`
> by a **one-line prerequisite commit (B0, §13)** that needs none of this design.
> Attributing it to `result="inference"` would inflate the headline with a bug
> fix. What this design actually buys on a GP fit is the remaining ~4 MB
> (`setup.X`, `_X`, `training_data`, `_S`).

**This reduces retention, not peak.** The fit still materializes `setup.X`, every
per-smooth `_X`, and every penalty; `result="inference"` drops them in `_from_fit`
*after* Phase 2 completes. That is exactly right for a long-lived serving object,
and does **nothing** for a large-`n` fit that OOMs while fitting. Peak-memory work
(chunking, sparse) is a different project (§3.2).

### 1.2 The Fix: a `result` mode (two result types) + a lean core

1. **A keyword-only `result: Literal["full", "inference"] = "full"` on
   `GAM.fit()`.** `"full"` (default) returns a **`GAMResults`** — today's surface.
   `"inference"` returns a distinct, narrower **`GAMInferenceResult`** that retains
   **none of the dense training arrays and no penalty caches** (§3.1, §7) and
   drops the **training-data-backed diagnostic surface**
   (`summary()`/`plot()`/self-prediction). It **keeps** the cheap scalar
   diagnostics (`edf*`, `deviance`, `null_deviance`, `score`, …) — they are
   already computed and cost `O(1)`/`O(p)` (§4.2). `fit()` is `@overload`ed so the
   static type is exact (§12.5).

   ```python
   model = GAM("y ~ s(x)", family="poisson")     # pure spec — reusable
   res:  GAMResults         = model.fit(df)                      # full
   lean: GAMInferenceResult = model.fit(df, result="inference")  # lean
   ```

   `result` lives on `fit()` (beside `weights`/`offset`) because it controls
   **result materialization**, not the statistical model. Naming the *outcome*
   (`"inference"`) rather than a behavior (`diagnostics=False`) keeps the call
   site self-documenting and is extensible (an eager-diagnostics mode could be a
   third value).

2. **A lean, picklable inference core, `GAMPredictor`** (`jaxgam/inference/`):
   the irreducible state for point + standard-error prediction — coefficients
   (β̂), the Bayesian posterior covariance `Vp` (**required**), a snapshot of the
   fitted family (exact link, final NB `theta`), an external-offset flag, and an
   `n`-independent predict-matrix spec. No training data, no dense `n`-row array,
   no penalty matrices, with **`coefficients`/`Vp` read-only** (§5.3), picklable
   for same-version/transient handoff (§8). `GAMInferenceResult` *composes* one;
   `GAMResults` *builds* one on demand via `to_predictor()`.

"Robust for downstream production" = the **in-memory objects** are lean, narrowly
typed, and read-only on the two arrays that matter — *not* that jaxgam ships a
durable serialization format (it does not; §8).

### 1.3 Scope

**In scope:** keyword-only `result` on `GAM.fit()` returning two types; a
`GAMInferenceResult` retaining no dense training arrays and no penalty caches; a
`GAMPredictor` core (narrow, `coefficients`/`Vp` read-only, picklable) + a Phase-1
`PredictSpec` (lazy) + a single Phase-1 predict-matrix builder + a single Phase-3
predict finish; **export** the new public types; the zero-numerical-risk DRY
cleanups tied to the result-type refactor (collapse the eight `setup.*` duplicate
fields; drop the dead `hasattr` guard); and the **GP `_E_knot` dead-store fix**
(§13 commit B0) — a prerequisite, not a product of the mode.

**Out of scope:** serialization *format* / `save()`/`load()` / versioning /
integrity / migration; a JAX-free import guarantee; distributional sampling /
intervals / `family.sample()`; a point-only (drop-`Vp`) mode; the **null-deviance
DRY rewrite** and the **`summary()` CQS change** (both **deferred** — orthogonal,
user-visible/math-adjacent; §10.3); back-compat machinery; **any change to a
numerical result** (predictions/SE/EDF/deviance incl. `null_deviance`
byte-identical to today — §10).

### 1.4 Relation to `refactor_gam_api`

Builds on the prior spec-vs-results split and the removed `_fitted` guard; does
not implement its deprecated `__getattr__` forwarding (pre-release).

---

## 2. Current-State Analysis

### 2.1 What `fit(df)` retains in memory

| Held array | Size | Copy or alias? | Notes |
|---|---|---|---|
| `setup.X` | **(n × p)** | own buffer | **dominant `O(n·p)` cost** |
| per-smooth `_X` (and nested) | (n × k) each | own buffer | training design caches inside each smooth (and `_marginals`/`base_smooth`) — §7 |
| per-smooth `_S` / tensor `_penalties` | **(k × k)** / **(n_coefs × n_coefs)** | own buffer | **penalty matrices — fitting-only, NOT read by predict** (§2.1 note) |
| `setup.penalties` (`CompositePenalty`) | n_penalties × **(total_p × total_p)** | own buffer | penalties embedded in constrained space (`design.py:133`); read post-fit only by `summary()` (`summary.py:534,634`). Dropped with `setup`; line-item it in the Metric-#3 walk or the delta has an unexplained residual on multi-penalty models. |
| GP `_E_knot` | (n_knots × n_knots) | own buffer | **dead store** — assigned `gaussian_process.py:200`, read nowhere. Freed at setup by commit B0 (§13); not a `result`-mode win. |
| `training_data` (dict) | Σ smooth-covariate cols (n each) | view-or-copy | **pins the input frame alive** |
| `setup.y`, `setup.weights`, `setup.offset` | (n,) each | derived | response / weights / offset |
| `fitted_values`, `linear_predictor` | (n,) each | fresh from device | in-sample predictions |
| `results.X / y / weights / offset` | — | **aliases of `setup.*`** | same buffers, not extra copies |

> **What `predict_matrix` actually reads (verified this round) — penalties are
> NOT predict-critical.** Every read of `_S`/`_penalties` (and GP's `_E_knot`) is
> in `build_penalty_matrices()`/setup/eigendecomp (Phase-1 fitting), **never** in a
> `predict_matrix`:
> - **TPRS** reads `_shift`, `_Xu` (knots), `_UZ` (reparam) — `tprs.py:588–599`; `_S` (`tprs.py:515`) unused in predict.
> - **Cubic** reads `_knots`, `_F` — `cubic.py:419`; `_S` (`cubic.py:375`) unused in predict.
> - **Random effect** reads stored levels via `_build_interaction_matrix` — `random_effects.py:222`; `_S` (`:98`) only in `build_penalty_matrices` (`:238`).
> - **GP** reads `_knt`/`_resolved_rho`/`_shift`/`_UZ` via `_build_design` (`gaussian_process.py:157,246`); `_S` (`:226`) **and `_E_knot`** (`:200`) are NOT read by predict.
> - **Tensor** reads `_marginals`, `_XP_list` (+ `_Z_list` for `ti`) — `tensor.py:358–362`; `_penalties` (`:321,477`) only in setup/`build_penalty_matrices`.
>
> **Stronger for `_E_knot`: it is read by *nothing*.** `grep -rn _E_knot jaxgam/`
> returns exactly two hits — the declaration (`:90`) and the assignment (`:200`).
> Not `predict_matrix`, not `build_penalty_matrices`, not `setup` after the line
> that writes it. It is a dead store in **both** result modes, which is why it is
> fixed at the source (commit B0, §13) rather than papered over by
> `copy_for_prediction()`. The GP override survives as **defense-in-depth** (§7.2).
>
> So the droppable set is `_X`, the penalty caches `_S`/`_penalties`, **and GP's `_E_knot`** (already `None` post-B0). The
> kept set is exactly the predict transforms: knots (`_Xu`), reparam/constraint
> (`_UZ`, `_F`, `_XP_list`, `_Z_list`), `_shift`, stored levels — all `O(knots·k)`
> or `O(k²)`, none `O(n)`. Tensor `_penalties` at `O(n_coefs²)` are frequently the
> single largest retained array, so dropping them is a material win, not cosmetic.

### 2.2 Field Inventory and Consumer Map

Categories: **S** serving-critical, **D** diagnostic, **T** training-data, **M**
metadata, **dup** duplicate.

| Field | Cat | Scales `n`? | Consumers | Notes |
|---|---|---|---|---|
| `coefficients` (p,) | **S** | no | predict, summary, plot | most important serving array |
| `Vp` (p,p) | **S** | no | predict (SE), summary, plot | **always populated by a fit** — §5.3 makes it required |
| `linear_predictor` (n,) | D | **yes** | predict (self only) | cache of `X@coef` |
| `fitted_values` (n,) | D | **yes** | summary, repr | cache of `linkinv(eta)` |
| `scale`, `edf`, `edf1`, `edf_total`, `deviance`, `null_deviance`, `smoothing_params`, `score` | D | no/`O(p)` | summary, plot, repr | cheap scalar/`O(p)` — **kept in both result types** (§4.2) |
| `converged`, `n_iter`, `convergence_info`, `theta`, `n`, `formula`, `method`, `lambda_strategy`, `execution_path` | M | no | repr, summary | provenance/metadata |
| `family` (object) | **S**/D | no | predict (link), summary, repr | fit copies it **only** for estimated-theta NB (`api.py:122`); standard families are shared registry singletons (`registry.py:16`). Snapshot owner = `_from_fit`, post-`put_theta` (§5.3). |
| `setup` (`ModelSetup`) | **S**/D | **yes** | predict, summary | predict needs `build_predict_matrix`; re-holds `X/y/weights/penalties` + per-smooth `_X`/penalties |
| `coef_map`, `smooth_info`, `term_names` | **dup**/M | no | summary, plot | `= setup.*`. **`SmoothInfo` is all `str`/`int`/`bool`** (`design.py:68–77`) — **zero arrays**, and `PredictSpec` already carries `smooth_info` (§5.2). So `smooth_info`/`term_names` are free to expose on the lean type, and must be: without them `edf` is an unlabeled array (§5.4). |
| `X`, `y`, `weights`, `offset` | **dup/T** | **yes** | predict (self SE), summary | `= setup.*` (aliases) |
| `training_data` (dict) | **T** | **yes** | plot | raw covariate columns, plotting only |

**Minimal inference set** (`predict(newdata, se_fit=True)`): `coefficients`,
`Vp`, the fitted `family` (for `link`), the external-offset flag, and the
predict-matrix metadata (with predict-only smooths — `_X`/`_S`/`_penalties`/`_E_knot`
dropped).

> **Verified (this session):** `ModelSetup.build_predict_matrix()`
> (`design.py:415–491`) reads only `newdata` + the `n`-independent metadata
> (`parametric_terms`, `has_intercept`, `factor_info`, `ordered_factors`,
> `dropped_param_names`, `parametric_keep_cols`, `coef_map.{terms, transform_X,
> total_coefs}`, plus `smooth_info` for length-validation) and calls
> `term.smooth.predict_matrix`. It also uses three **instance helpers**
> directly — `self._to_dict` (`:434`, a `@staticmethod`),
> `self._validate_equal_lengths` (`:450`), `self._build_parametric_matrix`
> (`:457`) — the last of which transitively calls `_encode_factor` →
> `_contr_poly`, so the §6 move relocates **five** helpers in all. Never reads
> `self.X`/`y`/
> `weights`. That **exact attribute list is the `PredictSpec` field set** (§5.2).

### 2.3 Duplication (DRY)

Eight fields are set verbatim from `setup.*` in `_from_fit` (`results.py:220–233`):
`X`, `y`, `weights`, `offset`, `coef_map`, `smooth_info`, `term_names`, `n` —
reference aliases that obscure ownership. **Seven** become `@property` reads of
`setup` (§5.5); `n` (= `setup.n_obs`) is instead re-homed to the
`_FitDiagnostics` shared base as scalar metadata, since the lean
`GAMInferenceResult` retains no `setup`.

### 2.4 Smells in scope

| Smell | Severity | Evidence | Disposition |
|---|---|---|---|
| Result retains all `O(n)` state + penalties by default; no opt-out | critical | `setup.X`, per-smooth `_X`, `_S`/`_penalties`/`_E_knot`, `training_data` | core of this design |
| Raw covariates + response retained; pins input frame | major | `training_data`, `setup.y` | dropped under `result="inference"` |
| Eight `setup.*` fields duplicated as stored aliases | major | `results.py:220–233` | fixed (→ properties) |
| Dead `hasattr(family, "family_name")` guard in `__repr__` | trivial | `results.py:404` | fixed |
| `summary()` prints to stdout *and* returns | minor | `results.py:348–350` | **deferred (§10.3)** — user-visible, orthogonal |
| Null-deviance re-implements IRLS instead of reusing family helpers | minor | `results.py:549–577` | **deferred (§10.3)** — math-adjacent |

### 2.5 SOLID Audit

- **SRP** — `GAMResults` changes for four reasons (inference, summary, plot,
  data-holding).
- **ISP/LSP** — a predict-only client carries matplotlib/summary/training_data;
  a single result type whose diagnostic methods raise under `result="inference"`
  would be a refused-bequest (round 4 fixes this with two types, §4).
- **DIP** — predict reaches into concrete `ModelSetup`/`CoefficientMap`. Round 6
  introduces a one-method seam for `predict_core` and an explicit boundary value
  object (`PredictSpec`); it does **not** claim the predictor is decoupled from
  Phase 1 (§9).

---

## 3. Design Goals and Non-Goals

### 3.1 Goals

1. **`result="inference"` ⇒ no dense training arrays and no penalty caches
   retained**, framed by **banned owners/attributes**: `setup`, `fitted_values`,
   `linear_predictor`, `training_data` not held; `X`/`y`/`weights`/`offset` never
   stored; and **no smooth retains `_X`, `_S`, `_penalties`, or GP's `_E_knot`** (recursively,
   including nested `_marginals`/`base_smooth`). It **does** retain the predict
   transforms (knots `_Xu`, reparam `_UZ`/`_F`/`_XP_list`/`_Z_list`, `_shift`,
   stored levels) — bounded by `max_knots`/`k`, not `n` — and the cheap scalar
   diagnostics. Retained inference state is `O(p² + Σ_smooth predict-transform
   size)`; the penalty `O(Σ k²)` / `O(Σ n_coefs²)` is **gone**. (Test asserts
   banned owners/attributes incl. `_S`/`_E_knot is None` and empty `_penalties`,
   **not** "no array has dim == n" — knots are legitimately `≤ n` rows; §12.2.2.)
2. **Phase 1 owns prediction-state construction and cache-dropping;** Phase 3
   only consumes. No Phase 3 → Phase 1 private access; no upward import.
3. **No eager predictor materialization** (`result="full"`): nothing is
   cloned/stripped at fit; `ModelSetup`'s `PredictSpec` is built **lazily** on the
   first `predict(newdata)`/`to_predictor()`, so a full fit that never predicts
   pays nothing (§5.2).
4. **Two narrow result types** (no runtime-raising diagnostic methods); the lean
   type exposes only inference + cheap scalars; both are **exported** (§11).
5. **A lean inference core** (`GAMPredictor`) that **owns** `coefficients`/`Vp`
   (defensive copy, then `write=False`), snapshots the family, carries the
   external-offset flag explicitly, and is picklable.
6. **Honest layering & one source per path:** **one** Phase-1 predict-matrix
   builder over **one concrete `PredictSpec`**; **one** Phase-3 predict finish;
   `predict_core` is typed directly on `PredictSpec` (**no Protocol seam** — §5.1);
   the predictor carries `PredictSpec` as an explicit boundary value object (not a
   hidden dependency, §9). Eliminate the eight `setup.*` *stored* duplications —
   and introduce no new ones (`formula`, §5.5). (Deliberate
   by-reference sharing of read-only predict transforms is allowed; this is about
   removing duplicate **stored aliases**, §5.3, §7.)
7. **Preserve the three-phase boundary;** no fitting math touched; mgcv parity
   preserved by construction *and verified directly for the inference path*
   (§12.2); Phase 3 predict is CPU/NumPy (§6).

### 3.2 Non-Goals

ML serving infrastructure; a JAX-free import path; distributional sampling /
intervals / `family.sample()`; a point-only (drop-`Vp`) mode; the null-deviance
DRY rewrite and the `summary()` CQS change (**deferred**, §10.3); back-compat
machinery; **peak** fit-time memory (chunking/sparse — this design changes
retention only, §1.1); **changing any numerical result** (predictions/SE/EDF/
deviance incl. `null_deviance` byte-identical to today — §10).

---

## 4. The `result` Mode and Two Result Types

### 4.1 Surface

```python
from typing import Literal, overload

class GAM:
    def __init__(self, formula, family="gaussian", method="REML", sp=None, **kwargs): ...

    @overload
    def fit(self, data, weights=None, offset=None, *,
            result: Literal["full"] = "full") -> "GAMResults": ...
    @overload
    def fit(self, data, weights=None, offset=None, *,
            result: Literal["inference"]) -> "GAMInferenceResult": ...
    def fit(self, data, weights=None, offset=None, *, result="full"): ...
```

`result` is **keyword-only** (`*`), cannot be passed positionally, and the
overloads disambiguate the return type statically. It controls **result
materialization**, alongside the per-call `weights`/`offset`. Invalid values
raise `ValueError` at the top of `fit()`.

### 4.2 What the mode means — exact disposition

`result` toggles the **training-data-backed diagnostic surface** + the penalty
caches, not the diagnostic *values*. Precisely:

| State | `GAMResults` (`result="full"`) | `GAMInferenceResult` (`result="inference"`) |
|---|---|---|
| Dense training arrays (`setup.X`, per-smooth `_X`, `training_data`, `y`/`weights`/`offset`, `fitted_values`, `linear_predictor`) | retained | **dropped** |
| Penalty caches (`_S`, tensor `_penalties`, GP `_E_knot`) | retained | **dropped** (fitting-only; §2.1) |
| Predict-critical state (`coefficients`, `Vp`, family snapshot, knots/reparam transforms) | retained | retained |
| Cheap scalar diagnostics (`edf*`, `deviance`, `null_deviance`, `score`, `scale`, `theta`, `converged`, …) | retained | **retained** (`O(1)`/`O(p)`, already computed; useful for production logging) |
| Diagnostic *surface*: `summary()`, `plot()`, self-prediction cache | present | **absent** |

So the cheap scalars are kept in **both** modes — dropping them would save
nothing. `result="inference"` means *no training arrays, no penalty caches, no
summary/plot surface* — the scalar diagnostics remain accessible.

### 4.3 Behavioral semantics

| | `GAMResults` | `GAMInferenceResult` |
|---|---|---|
| `predict(newdata, se_fit=...)` | ✅ | ✅ |
| `predict()` self-prediction (`newdata=None`) | ✅ (cached `eta`) | **method requires `newdata`** (no in-sample cache) |
| `predict_matrix(newdata)`, `to_predictor()` | ✅ | ✅ |
| `coefficients`, `Vp`, `family`/`link`, `formula` | ✅ | ✅ |
| scalar diagnostics (§4.2 row 4) | ✅ | ✅ |
| `summary()`, `plot()` | ✅ | **not present** (`AttributeError`) |

There is **no `DiagnosticsNotRetainedError`**: the lean type simply does not have
`summary()`/`plot()`, and its `predict(newdata)` requires `newdata`. The type
system (and `@overload`) communicates the contract; `GAMInferenceResult.__repr__`
adds a one-line hint ("fit with `result='full'` for `summary()`/`plot()`"). This
is the ISP/LSP-clean alternative to a partially-disabled object.

---

## 5. Object Model

Three frozen types. Roles: `GAMPredictor` ⊂ `GAMInferenceResult`'s surface;
`GAMResults` is the full diagnostic result.

### 5.1 No `PredictMatrixBuilder` Protocol (considered and cut, round 10)

Rounds 4–9 carried a one-method `@runtime_checkable` Protocol in
`jaxgam/inference/_protocol.py` so that `predict_core` could accept "either a
`PredictSpec` (lean path) or a `ModelSetup` (full-result self-prediction)."
**It is cut.** Once §6's delegation lands —

```python
# ModelSetup, after the §6 move
def build_predict_matrix(self, newdata):
    return build_predict_matrix(self._lazy_predict_spec(), newdata)
```

— `ModelSetup` and `PredictSpec` are **not two implementations of a seam**. They
are one implementation and a forwarder to it. Passing `setup` and passing
`setup._lazy_predict_spec()` reach the same cached `PredictSpec` and produce
byte-identical output, so the Protocol had exactly one shape and abstracted
nothing. `predict_core` is therefore typed directly:

```python
def predict_core(spec: "PredictSpec", coefficients, Vp, link, newdata, *, ...): ...
```

`GAMPredictor` passes `self._predict_spec`; `GAMResults` passes
`self.setup._lazy_predict_spec()` (already built by the time anything predicts —
`build_predict_matrix` builds it too, so goal #3 is untouched: a full fit that
never predicts still builds no spec).

**What this removes:** a file, a public export, a `runtime_checkable` decorator
Python does not enforce structurally anyway, and the §9 paragraph explaining why
the Protocol was not really achieving DIP. The design's honest DIP position was
already "bounded, no decoupling claimed" — the consistent conclusion is to delete
the seam rather than document its limits for a fourth round. This is the one
place the additive review process had visibly accumulated structure.

### 5.2 `PredictSpec` (NEW — Phase 1, the explicit boundary value object)

Lives in **`jaxgam/formula/predict_matrix.py` (Phase 1)**, with the builder and
helpers it feeds (§6). A **concrete dataclass with explicit fields** — the exact
`n`-independent set the builder reads (§2.2): `coef_map`, `smooth_info`,
`parametric_terms`, `factor_info`, `ordered_factors`, `has_intercept`,
`parametric_keep_cols`, `dropped_param_names`, `total_coefs`. (It does **not**
carry `offset_was_nonzero` — the matrix builder never reads it; that flag lives on
`GAMPredictor`/`setup` only, §5.3, to avoid two sources of truth.) Its
`coef_map`'s smooths are **predict-only copies** (training `_X` **and** penalty
caches `_S`/`_penalties`/`_E_knot` dropped, predict transforms shared by reference
— §7). `build_predict_matrix(self, newdata)` delegates to the Phase-1 free
function. It is the **deliberate, documented Phase-1→Phase-3 boundary value
object** (pure NumPy, `n`-independent, picklable by construction) — not hidden
behind an abstraction.

**Lazy on the full path.** `ModelSetup` does **not** build a `PredictSpec` at
setup. It builds one **lazily** when `ModelSetup.build_predict_matrix(newdata)` or
`to_predictor()` first needs it, so a `result="full"` fit that never predicts pays
nothing (goal #3). Because `ModelSetup` is a **frozen** dataclass
(`design.py:79`), the cache is a private `field(init=False, default=None,
compare=False, repr=False)` populated on first access via `object.__setattr__`
(plain `functools.cached_property` does not work cleanly on a frozen dataclass).
The lean path (`result="inference"`) builds the spec once in `_from_fit` and
discards `setup`.

### 5.3 `GAMPredictor` (NEW — Phase 3, the lean picklable core)

`jaxgam/inference/predictor.py`. Frozen; owns `coefficients`/`Vp`;
training-data-free; penalty-free.

```python
import numpy as np

@dataclass(frozen=True)
class GAMPredictor:                 # O(p² + Σ predict-transform size); no training/penalty arrays
    coefficients: np.ndarray        # (p,) constrained space (post Sl.setup back-transform)
    Vp: np.ndarray                  # (p,p) Bayesian posterior covariance — REQUIRED
    family: "ExponentialFamily"     # post-fit snapshot; exact fitted link (+ final NB theta)
    formula: str                    # signature / audit / repr (the ONE owner — §5.5)
    offset_was_nonzero: bool        # external-offset predict warning
    _predict_spec: "PredictSpec"    # explicit Phase-1 boundary value object (§5.2)
    _jaxgam_version: str = field(default_factory=lambda: jaxgam.__version__)

    def __post_init__(self):
        # Own coefficients/Vp outright: copy first (so we never alias — and then
        # freeze — a caller's buffer; this is what lets GAMResults.to_predictor()
        # be safe), then write-protect. O(p²) once; negligible beside a fit.
        object.__setattr__(self, "coefficients", np.array(self.coefficients))
        object.__setattr__(self, "Vp", np.array(self.Vp))
        self.coefficients.setflags(write=False)
        self.Vp.setflags(write=False)

    def __setstate__(self, state):  # re-apply the freeze after unpickling
        self.__dict__.update(state)
        self.coefficients.setflags(write=False)
        self.Vp.setflags(write=False)
        if self._jaxgam_version != jaxgam.__version__:
            warnings.warn(                              # detectable, not silent — §8
                f"GAMPredictor was pickled by jaxgam {self._jaxgam_version}, "
                f"loading under {jaxgam.__version__}. Pickles are not a "
                "cross-version format; predictions may be wrong or fail.",
                stacklevel=2,
            )
```

**`_jaxgam_version` makes the pickle contract detectable.** §8 says pickles are
same-version/transient because `PredictSpec`'s layout is not version-stable — but
saying so in prose does nothing at load time, where a stale pickle either raises
cryptically or silently mis-predicts. The stamp + warning costs one field and five
lines, turns the worst failure mode (silent wrong answers in production) into a
loud one, and is **not** the versioned artifact format that stays out of scope
(§3.2): no schema, no migration, no integrity, no `save()`/`load()`.

**Public contract (stable):** `predict(newdata, ...)` / `predict_matrix(newdata)`;
read-only `coefficients`/`Vp`; the `family` **snapshot** (an independent deep copy,
not a frozen object — see "Why a family snapshot" below) exposing `link` ≡
`family.link`; `formula`. `offset_was_nonzero`, `_predict_spec`, and
`_jaxgam_version` are internal (the spec is a concrete Phase-1 object; layout not
public, not version-stable — §8).

**`offset_was_nonzero` is an explicit field, not derived from the spec.**
`predict_core` needs it to reproduce the external-offset warning (§6). It is
deliberately **not** stored on `PredictSpec` (the matrix builder never reads it,
§5.2) and **not** recomputed from a retained offset array (the lean path holds
none). One `bool`, one owner: the predictor computes it in `_from_fit` before
`setup` is discarded and passes it down.

**`Vp` is required.** Every fit produces `Vp` (it backs summary SEs too); no path
drops it. Typing it `np.ndarray` removes a dead branch and guard. A future
point-only mode is **out of scope** (§3.2); it would be a separate type, not a
nullable field.

**Read-only guarantee — exactly two arrays.** The enforced, public guarantee is:
**`coefficients` and `Vp` are read-only** (defensively copied at construction and
re-frozen on unpickle, then `write=False`). That is all we promise. The family/
link object graph and the predict-transform arrays (`_Xu`/`_UZ`/`_XP_list`/
`_Z_list`/`_F`) are **shared by reference** and read-only **by contract** — the
predict path treats them as `const`, but they are **not** frozen (deep-freezing
would either mutate buffers a live `GAMResults` still owns or force an `O(Σk²)`
copy this design avoids). Callers must not mutate predictor internals. We do
**not** claim deep immutability. Tests check only the two enforced arrays
(§12.2.7) + the family snapshot's independence (§12.2.6).

**Why a family snapshot, and who owns it.** Two distinct copies exist; keep them
separate (round 6):

- **api.py pre-fit copy (unchanged):** `GAM.fit()` deep-copies the family **only**
  for estimated-theta NB (`api.py:122–125`), to protect the shared registry
  singleton from theta mutation *during* fitting. `theta` is then synced into
  this instance post-fit by `put_theta`.
- **`_from_fit` snapshot (the predictor's `family`):** taken **after** fitting,
  **after `put_theta` finalizes theta**, by `copy.deepcopy` of the post-fit
  family object — so the snapshot has **final** theta and the predictor never
  aliases the fitting instance or the registry singleton. This is the **single
  owner** of the predictor/result family snapshot, for **every** fit (standard
  families included — `registry.py:16` singletons, so the deepcopy also gives
  snapshot independence and preserves a custom link, `api.py:81`). `O(1)`,
  Phase 3, no JIT-cache implication.

`coefficients` are in the **constrained coefficient space after the Sl.setup
back-transform** (`results.py:178-182`), consumed against the **constrained**
predict matrix (`results.py:276-277`) — not raw basis (§10.1).

### 5.4 The two result types

Both are frozen dataclasses sharing the cheap scalar diagnostics + metadata (via
a small shared base, `_FitDiagnostics`); they differ in retained state and
surface. Neither exposes a method that raises for being in the "wrong mode."

**`GAMInferenceResult`** (`fit(result="inference")`):

| Holds | Notes |
|---|---|
| `_predictor: GAMPredictor` | composed at fit (via `copy_for_prediction`, §7) |
| scalar diagnostics + metadata | `edf*`, `deviance`, `null_deviance`, `score`, `scale`, `theta`, `smoothing_params`, `converged`, `n_iter`, `convergence_info`, `method`, `lambda_strategy`, `execution_path`, `n` (**not** `formula` — property → `_predictor`, §5.5) |

Surface: `predict(newdata)` / `predict_matrix(newdata)` (delegate to
`_predictor`), `to_predictor()` → `self._predictor`, `coefficients`/`Vp`/`family`/
`formula` (properties → `_predictor`), **`smooth_info`/`term_names`** (properties
→ `_predictor._predict_spec`), the scalars, `__repr__`. **No** `summary`/`plot`;
`predict` requires `newdata`.

> **Why `smooth_info`/`term_names` are on the lean type.** They were classified
> "dup" in §2.2 only because `GAMResults` reads them off `setup` — an accident of
> where they were stored, not a memory decision. `SmoothInfo` is all
> `str`/`int`/`bool` (`design.py:68–77`), and `PredictSpec` **already carries
> `smooth_info`** (§5.2), so the data is retained either way; §5.5 was simply not
> exposing it. Without it `GAMInferenceResult.edf` is a bare array with no way to
> say which smooth each entry belongs to — on the type whose stated purpose is
> production logging (§4.2). Exposing them costs **zero bytes**.

**`GAMResults`** (`fit(result="full")`):

| Holds | Notes |
|---|---|
| `coefficients`, `Vp`, `family` (snapshot) | core arrays, held directly |
| `setup: ModelSetup` | the predict-builder (lazy `PredictSpec`) + `X/y/weights/penalties` for summary |
| `fitted_values`, `linear_predictor`, `training_data` | in-sample caches + raw covariates |
| scalar diagnostics + metadata | same shared base as above |

Surface: everything in `GAMInferenceResult` **plus** `summary()`, `plot()`, and
self-prediction (`predict(newdata=None)` via `linear_predictor`/`setup`).
`to_predictor()` builds a `GAMPredictor` **on demand** from `setup`'s lazily-built
`PredictSpec` (§7) + the family snapshot + `offset_was_nonzero`; the predictor
**defensively copies** `coefficients`/`Vp` (§5.3), so this never freezes the
result's own arrays. **Seven** of the eight `setup.*` duplicate fields (`X`,
`y`, `weights`, `offset`, `coef_map`, `smooth_info`, `term_names`) become
guarded `@property` reads of `setup`; the eighth, `n`, is re-homed to the
`_FitDiagnostics` shared base as retained scalar metadata (§5.5) — the lean
`GAMInferenceResult` has no `setup` to back a property.

### 5.5 Field-Ownership Table (every current field → new home)

| Current field | `GAMResults` | `GAMInferenceResult` |
|---|---|---|
| `coefficients`, `Vp`, `family` | direct fields | via `_predictor` (property) |
| `formula` | direct field | **`@property` → `_predictor.formula`** (§5.3 owns it) |
| predict-matrix state | `setup` (lazy `PredictSpec`) | inside `_predictor` (`_predict_spec`) |
| external-offset flag | derived from `setup.offset` | `_predictor.offset_was_nonzero` |
| `linear_predictor`, `fitted_values`, `training_data` | held | **dropped** |
| `setup` | held | **dropped** |
| `smooth_info`, `term_names` | `@property` → `setup` | **`@property` → `_predictor._predict_spec`** (zero-byte metadata, §5.4) |
| `coef_map`, `X`, `y`, `weights`, `offset` | `@property` → `setup` | not exposed (no `setup`) |
| `edf*`, `deviance`, `null_deviance`, `smoothing_params`, `score`, `scale`, `theta`, `converged`, `n_iter`, `convergence_info`, `n`, `method`, `lambda_strategy`, `execution_path` | shared base | shared base |

**Eliminated:** all eight `setup.*` re-exposures as stored fields; all dense
training-array **and penalty-cache** retention under `result="inference"`.

> **`formula` has exactly one owner.** An earlier draft put `formula: str` on
> **both** `GAMPredictor` (§5.3, needed so a pickled predictor is self-describing
> when it travels alone) **and** `_FitDiagnostics` — which would have introduced a
> **ninth stored duplicate** in the same design whose DRY headline is deleting
> eight. The predictor owns it; `GAMInferenceResult.formula` is a property.
> `GAMResults` keeps its own stored `formula` (it has no predictor until
> `to_predictor()`).

---

## 6. The Shared Predict Path

The split follows the phase boundary, and **Phase 3 predict is CPU/NumPy**: it
accepts and returns NumPy arrays, calls the existing link objects (NumPy via
`array_module` — `links.py:179,187,…`), and performs **no `jax.device_put` and no
JIT**. (The current `GAMResults.predict` already does this; the refactor
preserves it.)

**Phase 1 — matrix construction** (`jaxgam/formula/predict_matrix.py`, numpy
only, no JAX, no links). The body in `ModelSetup.build_predict_matrix`
(`design.py:415–491`) moves here as a free function over a **single concrete
`PredictSpec`**; its instance helpers — **and their full transitive closure** —
become **module-level** Phase-1 functions. `build_predict_matrix` directly calls
`_to_dict`/`_validate_equal_lengths`/`_build_parametric_matrix`;
`_build_parametric_matrix` in turn calls `_encode_factor`, which calls
`_contr_poly`, so **all five** move (moving only the first three would force a
`predict_matrix.py` → `design.py` import). `ModelSetup.build()` and the formula
tests still call these, so keep thin `@staticmethod` shims on `ModelSetup`:

```python
# jaxgam/formula/predict_matrix.py   — Phase 1
def build_predict_matrix(spec: "PredictSpec", newdata) -> np.ndarray:
    """`spec` is a PredictSpec — a concrete dataclass carrying exactly the
    n-independent attribute set the build reads (§5.2). One type."""
    data_dict = _to_dict(newdata)
    _validate_equal_lengths(spec, data_dict, ...)
    X_parametric, _ = _build_parametric_matrix(spec.parametric_terms, newdata,
                                               spec.has_intercept, ...,
                                               spec.factor_info, spec.ordered_factors)
    # ... transform_X over spec.coef_map.terms; shape-check spec.total_coefs ...
    return X_p

def _to_dict(newdata): ...                 # moved from ModelSetup (already @staticmethod)
def _validate_equal_lengths(spec, ...): ...
def _build_parametric_matrix(parametric_terms, newdata, has_intercept, ...): ...
def _encode_factor(...): ...               # transitive: called by _build_parametric_matrix
def _contr_poly(n_levels): ...             # transitive: called by _encode_factor
```

`ModelSetup.build_predict_matrix(self, newdata)` delegates: `return
build_predict_matrix(self._lazy_predict_spec(), newdata)`, where
`_lazy_predict_spec()` builds and caches the spec on first use — on the frozen
`ModelSetup` via an `init=False` field set with `object.__setattr__` (§5.2).
`PredictSpec.build_predict_matrix` is the same one-line delegation. **One concrete
state, one builder.**

**Phase 3 — predict finish** (`jaxgam/inference/_core.py`, touches `link`). One
`finish_prediction`, called by both the new-data and cached self-prediction
paths. `Vp` is required, so there is no `None` branch; `offset_was_nonzero` is a
parameter (the predictor passes its field; `GAMResults` passes `setup`'s):

```python
# jaxgam/inference/_core.py
import warnings
import numpy as np

def finish_prediction(eta, X_p, link, Vp, *, pred_type, se_fit):
    pred = link.linkinv(eta) if pred_type == "response" else eta
    if not se_fit:
        return pred
    se = np.sqrt(np.sum((X_p @ Vp) * X_p, axis=1))           # identical to results.py:300–301
    if pred_type == "response":
        se = se * np.abs(np.asarray(link.mu_eta(eta)))
    return pred, se

def predict_core(spec: "PredictSpec", coefficients, Vp, link, newdata, *,
                 pred_type="response", se_fit=False, offset=None, offset_was_nonzero=False):
    if pred_type not in ("response", "link"):
        raise ValueError(f"pred_type must be 'response' or 'link', got {pred_type!r}")
    X_p = spec.build_predict_matrix(newdata)                 # one concrete type (§5.1)
    eta = X_p @ coefficients
    if offset is not None:
        eta = eta + np.asarray(offset, np.float64).ravel()
    elif offset_was_nonzero:
        warnings.warn("Model fit with an external offset; none passed to predict() "
                      "on new data — offset omitted (matches mgcv predict.gam).", stacklevel=3)
    return finish_prediction(eta, X_p, link, Vp, pred_type=pred_type, se_fit=se_fit)
```

`GAMPredictor.predict()` calls `predict_core(self._predict_spec, self.coefficients,
self.Vp, self.family.link, newdata, …, offset_was_nonzero=self.offset_was_nonzero)`.
`GAMResults.predict()` passes **`self.setup._lazy_predict_spec()`** (self-prediction
adds the cached-`eta` branch with `X_p = setup.X` only when `se_fit`) and its
derived offset flag — one finishing path over **one concrete type**, no Protocol
(§5.1). SE is the **exact** `sqrt(rowSums((X_p @ Vp) * X_p)) * |mu_eta|` from
`results.py:299–307` — byte-identical to today and to `predict.gam`.

> **Consequence worth stating: the full path predicts through predict-only copies
> too.** After this move, `setup.build_predict_matrix()` routes through the lazily
> built `PredictSpec`, whose smooths are `copy_for_prediction()` copies. So every
> existing test in `tests/test_predict/` exercises the cache-dropping path — the
> `_X`/`_S`/`_penalties` drop is regression-gated by the **whole** predict suite,
> not only by the new tests. This is safe precisely because §2.1 verified no
> `predict_matrix` reads a dropped cache, and it is why "`tests/test_predict/`
> passes byte-identically" is a strong gate rather than a formality.

---

## 7. Memory Reduction

### 7.1 The honest invariant

"Zero `O(n)` retention" is false for knot-based bases, and "no retained array has
a dimension equal to `n`" is inconsistent with §7's requirements: prediction
needs the knots. The invariant is framed by **banned owners/attributes** (§3.1):

> `result="inference"` retains none of: `setup`, `X`, `y`, `weights`, `offset`,
> `fitted_values`, `linear_predictor`, `training_data`, any smooth's `_X`, or any
> smooth's **penalty cache** (`_S`, tensor `_penalties`, GP's `_E_knot`) — recursively. It **does**
> retain the predict transforms (`_Xu`, `_knt`, `_UZ`, `_F`, `_XP_list`,
> `_Z_list`, `_shift`, stored levels) — bounded by `max_knots`/`k`, not `n`. The
> penalty `O(Σ k²)`/`O(Σ n_coefs²)` is dropped. Retained inference state is
> `O(p² + Σ_smooth predict-transform size)`.

### 7.2 `copy_for_prediction()` — drop `_X` + penalty caches, share the transforms

Each smooth owns the knowledge of which arrays are *not* read by its
`predict_matrix` (`_X` and the penalty caches) vs the predict transforms, via a
**public Phase-1 hook** returning a **predict-only copy**. A **shallow** copy that
nulls only the non-predict caches and **shares the predict transforms by
reference** — so it neither duplicates the `O(n·k)` `_X` (drops it) nor copies the
knot/reparam arrays (read-only during prediction):

```python
# jaxgam/smooths/base.py  (Phase 1)
import copy
class Smooth(ABC):
    def copy_for_prediction(self) -> "Smooth":
        """Predict-only shallow copy: drops the (n×k) training design `_X` AND the
        penalty cache `_S`; shares all predict transforms (_Xu, _knt, _UZ, _F, …)
        by reference.

        BASE-DEFAULT PRECONDITION: correct ONLY when the smooth's non-predict
        caches are exactly {_X, _S} and it has NO nested smooths. A smooth that
        violates this (e.g. tensor's `_penalties`, nested marginals) MUST
        override. The registry audit test (§12.2.8) fails for any registered
        smooth that neither overrides this nor is allow-listed as base-OK.
        """
        clone = copy.copy(self)
        for attr in ("_X", "_S"):
            if getattr(clone, attr, None) is not None:
                setattr(clone, attr, None)
        return clone

# jaxgam/smooths/tensor.py  — drop _penalties (O(n_coefs²)), KEEP _XP_list/_Z_list (predict reads them)
class TensorProductSmooth(Smooth):
    def copy_for_prediction(self):
        clone = copy.copy(self)
        if getattr(clone, "_X", None) is not None:
            clone._X = None
        clone._penalties = []                                # fitting-only (tensor.py:321,477)
        clone._marginals = [m.copy_for_prediction() for m in self._marginals]  # recurse: drop _X/_S
        return clone                                # _XP_list (+ ti's _Z_list) shared by reference

# jaxgam/smooths/by_variable.py  (FactorBySmooth / NumericBySmooth — not Smooth subclasses)
class FactorBySmooth:
    def copy_for_prediction(self):
        clone = copy.copy(self)
        if getattr(clone, "_X", None) is not None:
            clone._X = None
        clone.base_smooth = self.base_smooth.copy_for_prediction()
        return clone

# jaxgam/smooths/gaussian_process.py  — base drops _X/_S; _E_knot is already None post-B0
class GaussianProcessSmooth(Smooth):
    def copy_for_prediction(self):
        clone = super().copy_for_prediction()    # drops _X, _S
        clone._E_knot = None                      # defense-in-depth: B0 already nulls it at setup
        return clone
```

(cubic / TPRS / random-effect use the base default — verified: their
`predict_matrix` reads only `_knots`/`_F`, `_Xu`/`_UZ`, or stored levels, never
`_S`. The registry audit (§12.2.8) would flag a GP that forgot the override.)

> **The GP override is defense-in-depth, not the `_E_knot` fix.** Commit B0 (§13)
> nulls `_E_knot` at the end of `setup()`, so by the time any result exists it is
> already `None` in **both** modes — that is where the 32 MB actually comes back
> (§1.1, §2.1). The override is kept so the invariant survives a future change
> that starts retaining `_E_knot` past setup, and so the §12.2.2 banned-state
> assertion has a single consistent shape across smooth types. Do **not** cite it
> as the source of the GP memory win.

`build_predict_spec` rebuilds the `coef_map` with predict-only smooths — Phase 3
never names a private attribute, with **no in-place mutation** of the live `setup`
and **no deepcopy of `_X`**:

```python
# jaxgam/formula/predict_matrix.py  (Phase 1)
import dataclasses
def build_predict_spec(setup) -> PredictSpec:
    new_terms = tuple(
        dataclasses.replace(t, smooth=t.smooth.copy_for_prediction())
        if getattr(t, "smooth", None) is not None else t
        for t in setup.coef_map.terms
    )
    coef_map = dataclasses.replace(setup.coef_map, terms=new_terms)   # shares Z_centering etc. by ref
    return PredictSpec(coef_map=coef_map, smooth_info=setup.smooth_info, ...)
```

The predict-only smooths set `_X`/`_S`/`_penalties`/`_E_knot` to `None`/`[]` **on the
copies**, leaving `setup`'s own `coef_map.terms` intact for summary/
self-prediction.

- **`result="inference"`:** `_from_fit` builds the spec, composes the predictor,
  and **discards `setup`** — the original `_X`/`_S`/`_penalties`/`_E_knot` arrays are freed.
- **`result="full"` + `to_predictor()`:** `setup` stays live; the predictor reuses
  `setup`'s lazily-built `PredictSpec` (predict-only copies, transforms shared) and
  **defensively copies** `coefficients`/`Vp` (§5.3). No `O(n·k)`/`O(n_coefs²)`
  duplication, no mutation of `setup`'s smooths, no freezing of the result's
  arrays.

> **Each `copy_for_prediction()` is verified against that type's `predict_matrix`
> at implementation (Commit C)** — a smooth nulls a cache only because its
> `predict_matrix` does not read it (§2.1 line citations). The **registry audit
> test** (§12.2.8) is the backstop: it fails when a registered smooth type is
> neither overriding `copy_for_prediction` nor allow-listed as base-default-OK, so
> a *new* smooth cannot ship unaudited.

---

## 8. Picklability

`GAMPredictor` (and `GAMInferenceResult`, which composes one) is a frozen
dataclass of NumPy arrays, the family snapshot, a `str`, a `bool`, and the opaque
`_predict_spec` (a `PredictSpec`: pure-NumPy Phase-1 types; Phase-1 modules import
no JAX).

```python
import pickle
blob = pickle.dumps(results.to_predictor())        # public default — built-in families/links
pickle.loads(blob).predict(newdata, se_fit=True)
# locally-defined custom Link? install + use cloudpickle instead (see below).
```

**Precise contract:**

- **Purpose: same-version / transient handoff** (worker processes on the *same*
  jaxgam version; in-session caching). **Not** durable, cross-version
  persistence — a loaded predictor depends on `PredictSpec`'s (Phase-1) class
  layout, which can change across versions; pickles are **not** guaranteed
  portable between versions (pre-release). Durable persistence needs the versioned
  schema we cut (§3.2).
- **The contract is *detectable*, not just documented** (round 10): the predictor
  stamps `_jaxgam_version` at construction and `__setstate__` **warns** on
  mismatch (§5.3). Prose in a design doc cannot stop a stale pickle from silently
  mis-predicting in a serving worker; a five-line check can. This is a guardrail
  only — still no schema, no migration, no integrity, no `save()`/`load()`.
- **stdlib `pickle` is the public default** and works when every family/link/
  smooth is **built-in or module-top-level** — the common case. **`cloudpickle` is
  *required* only for *locally-defined* custom links/families** (a `Link` subclass
  defined in a function/test); stdlib `pickle` cannot serialize a locally-scoped
  class. We do **not** ship `cloudpickle` as a runtime dependency — persistence is
  downstream's concern (scope, §3.2); a user who deploys local custom links
  installs `cloudpickle` themselves (or we expose an optional `[serialization]`
  extra). `cloudpickle` is in the **`dev`** optional extra
  (`[project.optional-dependencies].dev`, installed in CI/Docker via `uv sync
  --extra dev`) because the tests exercise that path.
- **Read-only survives the round-trip:** `__setstate__` re-applies `write=False`
  to `coefficients`/`Vp` (NumPy does not always preserve the flag through pickle).
- **Not provided:** a file format, `save()`/`load()`, versioning, integrity/
  signing. **Pickle is trusted-input only.**
- **Tradeoff:** unpickling and using the core imports JAX transitively via the
  family module. No JAX-free path.

---

## 9. SOLID / DRY Scorecard

- **SRP** — Inference (`GAMPredictor`), the two results, and Phase-1 construction
  (`ModelSetup`) change for distinct reasons.
- **ISP/LSP** — **two narrow types instead of one with raising methods**:
  `GAMInferenceResult` exposes only inference + scalars; `GAMResults` adds the
  diagnostic surface. No refused-bequest, no `DiagnosticsNotRetainedError`.
- **DIP — not claimed, and the fake seam is gone.** Rounds 4–9 justified a
  one-method `PredictMatrixBuilder` Protocol as `predict_core`'s parameter type.
  Round 10 cut it: after §6's delegation there is only **one** implementation
  (`PredictSpec`) and a forwarder (`ModelSetup`), so the Protocol abstracted
  nothing while costing a file, an export, and three rounds of caveats (§5.1).
  `GAMPredictor` carries a **concrete `PredictSpec`** — an explicit,
  deliberately-designed Phase-1→Phase-3 value object (pure NumPy, `n`-independent,
  picklable). We **do not claim** the predictor is decoupled from Phase-1
  internals: pickle depends on `PredictSpec`'s layout (§8). The real SOLID wins
  are the ISP/LSP two-type split and the single builder/finish path — an
  unearned Protocol was never one of them.
- **OCP** — a new family adds a registry entry + `Link`; a new smooth implements
  `copy_for_prediction()` (or is audited as base-default-correct — enforced by the
  registry audit test §12.2.8) + `predict_matrix`. No new field per type.
- **DRY** — eight `setup.*` duplications removed; **one** Phase-1
  `build_predict_matrix` over one concrete state; **one** `finish_prediction`
  (cached + new-data + both result types). (The `summary()` CQS and null-deviance
  DRY cleanups are **deferred**, §10.3 — not part of this design's DRY claims.)

---

## 10. Numerical and Correctness Considerations

### 10.1 Coefficient space

`coefficients` are in the **constrained coefficient space after the Sl.setup
back-transform** (`results.py:178-182`), consumed against the **constrained**
predict matrix (`results.py:276-277`). Not raw/un-constrained basis.

### 10.2 SE byte-parity

`finish_prediction` uses the exact `sqrt(rowSums((X_p @ Vp) * X_p)) * |mu_eta|`
from `results.py:299–307`. New-data SE numerically identical to today; the
self-prediction fast path (`GAMResults` only) still reads `setup.X`.

### 10.3 Deferred cleanups (kept out of the critical path)

Two cosmetic improvements are **deferred to separate PRs** so this design touches
no numerics and no default-path I/O:

- **Null-deviance DRY** (`results.py:549–577`): math-adjacent and
  byte-parity-sensitive (the loop uses a `max(·, 1e-300)` weight floor that
  `family.working_weights` lacks). **Left exactly as-is**, so `null_deviance` is
  byte-identical *because nothing touches it*. If revisited: reuse
  `working_response` while keeping the floored weight; own byte-identity gate.
- **`summary()` CQS** (`results.py:348–350`): making `summary()` query-only
  (return instead of print) is a **user-visible default-path behavior change**
  unrelated to lean inference. Deferred to its own PR with its own test. (Two
  review rounds flagged bundling it as scope creep.)

Neither blocks the result-type/memory work; both are clearly separable.

---

## 11. File Plan

### 11.1 New Files

| File | Phase | Description |
|---|---|---|
| `jaxgam/formula/predict_matrix.py` | 1 | `PredictSpec` + `build_predict_matrix(spec, newdata)` + module-level `_to_dict`/`_validate_equal_lengths`/`_build_parametric_matrix`/`_encode_factor`/`_contr_poly` (the full transitive closure, moved from `ModelSetup`) + `build_predict_spec(setup)`. numpy only. §5.2, §6, §7 |
| `jaxgam/inference/__init__.py` | 3 | Public surface: **`GAMPredictor`** only. **`predict_core`/`finish_prediction` stay private** to `_core.py` — a low-level path over the Phase-1 spec, not public API. |
| `jaxgam/inference/_core.py` | 3 | `predict_core(spec, …)` + `finish_prediction`. §6 |
| `jaxgam/inference/predictor.py` | 3 | `GAMPredictor` (`Vp` required; `offset_was_nonzero` + `_jaxgam_version` fields; defensive-copy read-only on `coef`/`Vp`; carries `PredictSpec`). §5.3 |
| `tests/test_inference/test_predictor.py` | — | predict parity vs `GAMResults` (incl. offset) + **vs R** (tensor, factor-by, NB, non-default built-in link — offset & locally-defined custom links excluded, §12.2.1b); pickle + cloudpickle round-trip (read-only survives; **stale-version warning**); read-only `coef`/`Vp`; family-snapshot independence + final-theta; `to_predictor()` doesn't freeze/alias the result's arrays; `copy_for_prediction` drops `_X`/`_S`/`_penalties`/`_E_knot`, keeps `_XP_list`/`_Z_list`/`_UZ`. |
| `tests/test_result_mode.py` | — | `result="inference"` ⇒ `GAMInferenceResult` (runtime `isinstance`); banned dense arrays + penalty caches absent (recursive); no `summary`/`plot`; scalars + **`smooth_info`/`term_names`/`formula`** read on both; invalid `result` raises; `typing.assert_type` block (§12.5). |
| `docs/production_api/implementation_plan.md` | — | Commit-by-commit breakdown. |

> Package `jaxgam/inference/` (renamed from the first draft's `serving/`).
> `PredictSpec` + builder + strip live in **Phase 1** (`formula/`); only the
> predictor and the predict-finish are in Phase 3 (`inference/`).

### 11.2 Modified Files

| File | Change |
|---|---|
| `jaxgam/__init__.py` | **Export `GAMInferenceResult` and `GAMPredictor`** (add to imports + `__all__`); they are public return/handoff types. |
| `jaxgam/smooths/gaussian_process.py` (**commit B0, standalone**) | Null `_E_knot` at the end of `setup()` — a dead store (`:200`, read nowhere), 32 MB at `n=5000`. Independent of this design; lands first so the baseline delta is honest. §1.1, §2.1, §13 |
| `jaxgam/smooths/base.py` (+ `tensor.py`, `gaussian_process.py`, `by_variable.py`) | Add `copy_for_prediction()` — base drops `_X`+`_S`; tensor drops `_penalties` and recurses `_marginals` (it has no `_X`/`_S` of its own); **GP** additionally nulls `_E_knot` as defense-in-depth (already `None` post-B0); by-variable recurses `base_smooth` (`FactorBySmooth`/`NumericBySmooth` are **not** `Smooth` subclasses and hold no `_X`). Documented base-default precondition. §7.2 |
| `jaxgam/formula/design.py` | `ModelSetup.build_predict_matrix` + the five helpers (`_to_dict`/`_validate_equal_lengths`/`_build_parametric_matrix`/`_encode_factor`/`_contr_poly`) move to `formula/predict_matrix.py` (thin `@staticmethod` shims remain for `build()`/formula tests); `ModelSetup` builds a **lazy** cached `PredictSpec` and delegates. Phase 1 → Phase 1. |
| `jaxgam/api.py` | Keyword-only `result="full"` on **`GAM.fit()`** with `@overload`; validate value; thread the mode into `_from_fit`. **Rename the local optimizer output `result` (`api.py:136`) → `fit_result`** to free the public `result` kwarg. The **NB pre-fit `deepcopy` (`api.py:122`) is unchanged** — it is *not* the predictor snapshot (§5.3). |
| `jaxgam/results.py` | Add `_FitDiagnostics` base + `GAMInferenceResult` + reshaped `GAMResults`; both delegate predict to `predict_core`/`finish_prediction`; **`_from_fit` takes the family snapshot (`copy.deepcopy`, post-`put_theta`) — the single snapshot owner (§5.3)**; `to_predictor()` (compose / build-on-demand from lazy `setup` spec + snapshot + offset flag); seven `setup.*` → guarded `@property` (eighth, `n`, → `_FitDiagnostics` scalar, §5.5); drop dead `hasattr`. **Rename `_from_fit`'s `result: NewtonResult` param (`results.py:111`) → `fit_result`; the mode threads as a new `result_mode` arg** (no collision with the public `result`). **`null_deviance` + `summary()` untouched (§10.3).** |
| `pyproject.toml` | Add `cloudpickle` to the **`dev`** optional extra (`[project.optional-dependencies].dev`; tests exercise the local-custom-link path); **not** a runtime dep — stdlib `pickle` is the public default (§8). |

### 11.3 No Changes Needed

`fitting/*` (**Phase 2** JAX/JIT — unchanged; the predict *finish* lives in the
new Phase-3 `inference/`, not here); `families/*` (predictor snapshots the family;
no rehydration/`sample()`); `families/registry.py`; `penalties/*`;
`plot/plot_gam.py` (reads via `GAMResults`, present only when `result="full"`);
`smooths/constraints.py` (`CoefficientMap`/`TermBlock` rebuilt via
`dataclasses.replace`, §7.2).

---

## 12. Testing Strategy

Per `docs/clean_unit_tests/`: the validation matrix owns final-model R parity;
per-area files own layer-specific behavior; parametrize + `_AssertCollector`.

### 12.1 Ownership

`tests/test_validation_matrix.py` — unchanged (fits `result="full"`).
`tests/test_inference/` and `tests/test_result_mode.py` own the new
behavior, **including direct R parity for the inference path** (§12.2.1b). A
Phase-1 test owns `copy_for_prediction()` across smooth types + the registry
audit.

### 12.2 Core invariants

1. **Predict-equivalence (`STRICT`):** `predictor.predict(newdata, se_fit=True)
   == result.predict(newdata, se_fit=True)` byte-identical; `GAMResults` vs
   `GAMInferenceResult` give identical `predict(newdata)`. Zoo: gaussian `s(x)`,
   binomial factor-by, poisson + offset, NB, non-default-link, `te()`, `ti()` ×
   `pred_type`.
   **1b. Direct R parity for the lean & pickled path (`STRICT`/`MODERATE` per
   `tests/tolerances.py`):** because (1) only proves the two jaxgam paths agree
   with *each other*, also compare `GAMInferenceResult.predict` **and** a
   pickle→unpickle→`predict` round-trip **against R `predict.gam`** for at least
   `te()`, factor-by, NB, and a **non-default built-in link** — use one already
   expressible in `RBridge`: `Gamma(link="log")` (the existing `gamma_log` key,
   `r_bridge.py:213`, rpy2-path-only). The subprocess family map (`r_bridge.py:85`)
   carries default links only and is **not** extended — adding a `poisson_identity`
   key is out of scope (rpy2-only test infra; same rationale as offset below).
   Because the subprocess fallback can't express `gamma_log`, this case must run
   under rpy2 — guard it like the existing GP parity test (`test_gaussian_process.py:93`):
   take the shared auto `RBridge` and `pytest.skip("...requires rpy2")` when
   `bridge.mode != "rpy2"` (`RBridge()` silently falls back to subprocess when rpy2
   is unavailable, `r_bridge.py:98`; `RBridge(mode="rpy2")` would instead hard-error
   there, so the skip-guard is the safe form).
   This guards against a shared-but-wrong predict path (per the project rule:
   results must match canonical mgcv).
   **Locally-defined custom links/families have no R analogue**, so (like offset)
   they are covered by internal lean==full parity (§12.2.1) + the pickle/
   cloudpickle round-trip (§12.2.4), **not** direct R.
   **Offset is excluded from this direct-R gate:** `RBridge` has **no offset
   argument** at fit or predict (`r_bridge.py:241,1068`), and adding one is out of
   scope (rpy2-only test infra). Offset is instead covered by (1)'s internal
   lean==full equivalence (which *includes* the poisson+offset case, byte-
   identical) plus the fact that the offset enters as a post-matrix `eta + offset`
   term — identical in both jaxgam paths and to `predict.gam` — over a predict
   matrix whose R parity the non-offset cases above already establish.
2. **Banned state absent under `result="inference"`** (hard gate): the result
   holds no `setup`/`fitted_values`/`linear_predictor`/`training_data`; a
   **recursive walk** of the predictor's smooth graph finds no non-`None` `_X`,
   **no non-`None` `_S`/`_E_knot`, and empty `_penalties`**; and the predict
   transforms (`_XP_list`/`_Z_list`/`_Xu`/`_UZ`) **ARE present**. (Banned owners/attributes
   — not "no array has dim == n.")
3. **Type/surface:** `fit(result="inference")` returns a `GAMInferenceResult`
   **instance** (runtime `isinstance`) with **no** `summary`/`plot` and a
   `newdata`-required `predict`; `fit()`/`fit(result="full")` returns a
   `GAMResults` instance; an invalid `result` raises `ValueError`; scalars **and
   `smooth_info`/`term_names`/`formula`** read identically on both — the metadata
   leg is what makes `edf` interpretable on the lean type (§5.4). The **static**
   `@overload` is asserted with `typing.assert_type` (teeth only under a checker,
   §12.5); the `isinstance`/`ValueError` checks are the always-on runtime
   guarantee.
4. **Pickle round-trip (`STRICT`):** (a) `pickle` a built-in-family/link core →
   predict byte-identical + link survives + arrays still read-only; (b)
   `cloudpickle` a local-custom-link core, same (stdlib `pickle` expected to fail
   that case — asserted); (c) **version stamp**: a round-trip on the current
   version warns nothing, and a blob whose `_jaxgam_version` is mutated to a fake
   version **warns** on load (§5.3, §8).
5. **Aliasing / ownership (`to_predictor()` on `GAMResults`):** after the call,
   (a) `setup`'s smooths still hold `_X`/`_S`/`_penalties`/`_E_knot`; the predictor's spec
   smooths have them dropped but retain `_XP_list`/`_Z_list`; (b)
   `predictor.coefficients`/`Vp` are **distinct objects** from the result's, and
   the result's arrays are **not** frozen by the call (defensive copy, §5.3).
6. **Family snapshot independence + final theta:** `predictor.family is not
   get_family(name)`; mutating the registry instance does not affect the
   predictor; for NB, `predictor.family.theta` equals the **fitted** theta (taken
   post-`put_theta`, §5.3), not the initial value.
7. **Read-only arrays:** `coefficients`/`Vp` raise on in-place write (after
   construction *and* after an unpickle round-trip). (No claim is tested for
   transforms/family — they are read-only by contract only, §5.3.)
8. **`copy_for_prediction()` per type + registry audit** (Phase 1): after the
   copy, `_X`/`_S`/`_penalties`/`_E_knot` are dropped but `predict_matrix(newdata)` equals
   the pre-copy result (transforms intact); the original smooth is untouched. A
   **registry-level test enumerates every registered smooth/by-wrapper type** and
   asserts each either overrides `copy_for_prediction` or appears in an explicit
   `_BASE_DEFAULT_OK` allowlist — **failing for any unaudited registered type**.

### 12.3 Expected collected-test footprint

| Where | New collected tests | Pattern |
|---|---|---|
| `test_gaussian_process.py` (existing) | 0 (commit B0 **restructures** the existing `_E_knot` assertion at `:497` to check `E` before setup nulls it — no new test) | — |
| `test_result_mode.py` | ~4 (banned state absence; type/surface incl. `isinstance` + invalid-value + `smooth_info`/`term_names`/`formula`; scalars-read) | `_AssertCollector` |
| `test_predictor.py` | ~7 (predict equivalence × zoo; **R parity for lean + pickled**; pickle + cloudpickle + version stamp; aliasing/ownership; snapshot independence + final theta; read-only incl. post-unpickle) | `@parametrize` + `_AssertCollector` |
| `test_smooths` (existing) | ~2 (`copy_for_prediction()` per type — drops `_X`/`_S`/`_penalties`/`_E_knot`, keeps transforms; registry audit) | `@parametrize` |
| `test_results` (existing) | ~1 (duplicates-as-properties) | `_AssertCollector` |
| **Total** | **~14** | |

### 12.4 Hard gates

Predict-equivalence (§12.2.1), **direct R parity for the inference/pickled path
(§12.2.1b)**, and banned-state absence (§12.2.2) are the `STRICT` gates that block
the build. (Null-deviance and `summary()` are untouched, §10.3, so their existing
tests continue to pass unchanged.)

### 12.5 Typing

The `@overload` gives editors/IDEs the exact return type and documents the
contract. The repo configures **no type checker** (`pyproject.toml`: ruff +
vulture only). Therefore:

- The **runtime** contract (`fit(result="inference")` *is a* `GAMInferenceResult`;
  invalid `result` raises) is enforced by `isinstance`/`ValueError` tests
  (§12.2.3) — always on.
- The **static** overload is **editor/IDE aid only** unless a checker is added.
  Adding `pyright`/`mypy` (a `make typecheck` target running the `assert_type`
  block) is the way to give it teeth — flagged as optional maintainer tooling,
  not assumed by this design.

---

## 13. Implementation Sequence

Pre-release ⇒ an **implementation order**, not a migration. **One PR** at the end
against `main`, single working branch, discrete commits; the agent does **not**
`git commit`/`push`; each unit runs `make test-cov` and hands off.

| Commit | Scope | Risk | § |
|---|---|---|---|
| A | Pre-flight baseline (test counts, coverage, retained bytes at small/large `n`, incl. tensor `_penalties` and GP `_E_knot`). Add `cloudpickle` to dev deps. | none | §8 |
| **B0** | **GP `_E_knot` dead-store fix** — null it at the end of `setup()`; restructure the one existing assertion (`test_gaussian_process.py:497`) to check `E` before it is dropped; re-measure GP retained bytes. **Independent of this design**; lands first so A's "before" and E's "after" bracket only what the `result` mode actually buys. | low (one line + one test edit) | §1.1, §2.1, §7.2 |
| B1 | Duplicate `setup.*` fields → guarded properties; drop dead `hasattr` guard. | low (mechanical) | §2.3, §5.5 |
| C | **Phase-1 prediction state.** Move `build_predict_matrix` + helpers to `formula/predict_matrix.py`; `PredictSpec` (concrete, explicit fields); `ModelSetup` builds a **lazy** cached spec + delegates; `Smooth.copy_for_prediction()` (base drops `_X`+`_S`; tensor drops `_penalties` + recurses; GP nulls `_E_knot` defensively; by-variable recurses) + precondition; `build_predict_spec`; recursive-absence (incl. `_X`/`_S`/`_penalties`/`_E_knot`), aliasing, per-type copy, **and registry-audit** tests. | structural | §5.2, §6, §7 |
| D | **`inference/`: `predict_core` + `finish_prediction` + `GAMPredictor`.** `Vp` required; `offset_was_nonzero` + `_jaxgam_version` fields; defensive-copy read-only on `coef`/`Vp` (+ `__setstate__` re-freeze and version warning); `predict_core` typed on `PredictSpec` (**no Protocol** — §5.1); predict-equivalence `STRICT`, **R-parity for lean + pickled (§12.2.1b)**, pickle/cloudpickle/version-stamp, read-only, ownership tests. | core | §5.1, §5.3, §6, §8, §12 |
| E | **`result` mode + two result types + family snapshot + exports.** `_FitDiagnostics` base; `GAMInferenceResult` (incl. `smooth_info`/`term_names`/`formula` properties); reshaped `GAMResults` + `predict()` routed through `predict_core(setup._lazy_predict_spec(), …)`; `_from_fit` family snapshot (post-`put_theta`, single owner); **internal renames `result`→`fit_result` (api.py local + `_from_fit` param), mode threads as `result_mode`** to free the public kwarg; keyword-only `@overload`ed `fit()` with value validation; **export `GAMInferenceResult`/`GAMPredictor` from `jaxgam/__init__.py`** (`predict_core` stays private); `test_result_mode.py` (incl. `isinstance` + `assert_type` + invalid-value). | user-facing | §4, §5.3, §5.4, §11.2 |
| F | Docs (`docs/api.md`, `docs/quickstart.md`: `result` on `fit()`, two return types; "stdlib `pickle` is the default for same-version handoff; `cloudpickle` only for locally-defined custom links/families") + `implementation_plan.md`. Optionally note a `make typecheck` target (§12.5). | none | — |

> **Deferred (separate PRs):** the null-deviance DRY rewrite and the `summary()`
> CQS change (§10.3) — orthogonal, each with its own gate.

Predict-equivalence, R-parity for the inference path, and banned-state absence are
the correctness surface.

---

## 14. Risks and Tradeoffs

**Keep from the prior refactor:** the spec-vs-results split, removing `_fitted`,
the `_from_fit` factory. No `__getattr__` forwarding (pre-release).

- **Penalty caches are fitting-only, not predict-critical** (§2.1, §7) — *verified*
  that no `predict_matrix` reads `_S`/`_penalties` (`tprs.py:599`,
  `cubic.py:419`, `random_effects.py:222`, `gaussian_process.py:217`,
  `tensor.py:358–362`). `copy_for_prediction()` drops them; the recursive
  banned-state gate (§12.2.2) and per-type copy test (§12.2.8) confirm it.
  Tensor `_penalties` are `O(n_coefs²)` — the largest single win.
- **GP `_E_knot` is a dead store, and is NOT a win for this design** (§1.1, §2.1,
  §7.2) — assigned at `gaussian_process.py:200`, **read nowhere in `jaxgam/`**, so
  `result="full"` leaks all 32 MB (`n=5000`) too. Commit **B0** frees it at
  `setup()` for both modes; the `copy_for_prediction()` override stays as
  defense-in-depth. **Do not attribute the GP reduction to `result="inference"`** —
  after B0 the mode's GP win is the remaining ~4 MB. This was caught by a round-10
  verification pass that re-derived §2.1 from the code; the earlier rounds had it
  filed as a retention problem.
- **Transforms are predict-critical and kept** — `_Xu`/`_UZ`/`_F`/`_XP_list`/
  `_Z_list`; the predict-equivalence + R-parity tests (§12.2.1) gate this.
- **Internal equivalence ≠ R parity** — the lean and full paths sharing one
  builder could agree with each other yet drift from mgcv. §12.2.1b adds **direct
  R comparison** for the inference and pickled paths (tensor/factor-by/NB/
  non-default built-in link; locally-defined custom links have no R analogue →
  internal + pickle parity only). **Offset is not in the direct-R gate** — `RBridge` has no offset
  arg (`r_bridge.py:241,1068`); it is covered by internal lean==full equivalence
  (incl. poisson+offset) + the post-matrix `eta + offset` term (§12.2.1b).
- **`offset_was_nonzero` is an explicit predictor field** (§5.3) — computed in
  `_from_fit` before `setup` is discarded, so the external-offset warning survives
  the setup-drop without storing an offset array or a second copy on the spec.
- **`to_predictor()` ownership is safe by defensive copy** (§5.3) — the predictor
  copies `coefficients`/`Vp` before freezing, so building one from a `GAMResults`
  never freezes or aliases the result's own arrays. Tested §12.2.5.
- **Read-only is exactly two arrays** (§5.3) — `coefficients`/`Vp` only;
  family/link graph + transforms are shared by reference, const **by contract**,
  deliberately **not** deep-frozen (that would copy `O(Σk²)` or mutate a live
  `GAMResults`). The public guarantee is "coefficients and covariance are
  read-only," nothing more.
- **The `PredictMatrixBuilder` Protocol was cut** (§5.1, §9) — after §6's
  delegation, `ModelSetup` and `PredictSpec` were one implementation and a
  forwarder, not two implementations, so the seam abstracted nothing.
  `predict_core` is typed on `PredictSpec`; `GAMResults` passes
  `setup._lazy_predict_spec()`. Byte-identical output, one fewer file/export/
  concept. No decoupling was ever claimed, and pickle still depends on
  `PredictSpec`'s layout (§8).
- **The full path now predicts through predict-only copies** (§6) — a consequence
  of the lazy-spec delegation, not a separate decision. It is safe by §2.1 and it
  makes the entire existing `tests/test_predict/` suite a regression gate on the
  cache-dropping.
- **Retention, not peak** (§1.1, §3.2) — the fit still materializes everything;
  `_from_fit` drops it afterward. Correct for a serving object, useless for a fit
  that OOMs. Stated so nobody reaches for `result="inference"` to survive a large
  `n`.
- **Pickle staleness is detectable** (§5.3, §8) — `_jaxgam_version` + a
  `__setstate__` mismatch warning. Not a format; just the difference between a
  loud failure and silently wrong production predictions.
- **`smooth_info`/`term_names` stay on the lean type** (§5.4) — all-scalar
  metadata `PredictSpec` already carries. Without them `edf` is an unlabeled array
  on the type built for production logging. Zero bytes.
- **`formula` has one owner** (§5.5) — `GAMPredictor`; the lean result reads it as
  a property. Storing it on both would have been a ninth duplicate in the design
  that deletes eight.
- **`PredictSpec` is lazy on the full path** (§5.2) — a `result="full"` fit that
  never predicts builds nothing; goal #3 preserved.
- **Family snapshot has one owner** (§5.3) — `_from_fit`, **after** `put_theta`
  finalizes theta, distinct from api.py's NB pre-fit registry-protection copy.
  Tested for final theta + independence (§12.2.6).
- **`Vp` is required** (§5.3) — no dead nullable branch; a point-only mode would be
  a separate type.
- **Knot-based bases retain `O(knots·k + k²)` transforms** (§7.1) — inherent;
  documented. Still a large reduction from `O(n·p)` + penalties.
- **`result="full"|"inference"` names the outcome** (§4) — chosen over a
  `diagnostics` boolean (which read as "no diagnostics" while scalars are kept);
  self-documenting at the call site and extensible.
- **Public `result` collides with the internal optimizer-output `result`**
  (`api.py:136`, `results.py:111`) — freed by internal renames (`fit_result` local
  + `_from_fit` param; the mode threads as `result_mode`); §11.2, commit E.
- **Deferred cleanups** (§10.3) — null-deviance DRY and `summary()` CQS are out of
  this PR (math-adjacent / user-visible, orthogonal); `null_deviance` and
  `summary()` are byte/behavior-identical because untouched.
- **Pickle is same-version/transient, trusted-input only** (§8) — **stdlib
  `pickle` is the public default** (built-in families/links); `cloudpickle`
  (dev-only, **not** a runtime dep) is needed only for *locally-defined* custom
  links; no versioning/integrity; read-only re-applied on unpickle.
- **Typing is documented, not gate-enforced by default** (§12.5) — `@overload` +
  `assert_type` document the contract; runtime `isinstance`/`ValueError` enforce
  it; a static checker (`make typecheck`) is optional maintainer tooling.
- **Two result types instead of one** — slightly more surface (both exported), but
  ISP/LSP-clean; the shared scalar base + shared predict free-functions keep it
  DRY.

**Tradeoff accepted:** `result="inference"` retains no training arrays, no penalty
caches, and no summary/plot surface (it keeps the cheap scalars, §4.2). The binary
mode is simpler than precomputing partial diagnostics; an eager-diagnostics third
`result` value could be added later without disturbing this design.
