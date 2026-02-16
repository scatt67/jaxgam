# PyMGCV: Design Document for a Python Port of R's mgcv

**Version:** 1.18
**Date:** February 2026
**Status:** Design Phase — Post-Seventeenth Review: Scope Freeze

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background: What mgcv Does](#2-background-what-mgcv-does)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Core Computational Backend](#4-core-computational-backend)
5. [Smooth Function Specifications](#5-smooth-function-specifications)
6. [Distribution Families](#6-distribution-families)
7. [Penalized Iteratively Re-weighted Least Squares (PIRLS)](#7-penalized-iteratively-re-weighted-least-squares-pirls)
8. [Smoothness Selection: Smoothing Parameter Estimation](#8-smoothness-selection-smoothing-parameter-estimation)
9. [Automatic Differentiation Strategy](#9-automatic-differentiation-strategy)
10. [Execution Paths: Dense-GPU, Sparse-CPU, and Chunked](#10-execution-paths)
11. [GPU and Hardware Acceleration](#11-gpu-and-hardware-acceleration)
12. [Random Effects and Mixed Models](#12-random-effects-and-mixed-models)
13. [Formula Interface and Model Specification](#13-formula-interface-and-model-specification)
14. [Prediction, Summary, and Post-Estimation](#14-prediction-summary-and-post-estimation)
15. [Model Comparison, Concurvity, and Diagnostics](#15-model-comparison-concurvity-and-diagnostics)
16. [Distributed and Multi-Device Compute](#16-distributed-and-multi-device-compute)
17. [Public API Design](#17-public-api-design)
18. [Testing Strategy: Correctness Against R mgcv](#18-testing-strategy-correctness-against-r-mgcv)
19. [Implementation Phases and Agent Task Breakdown](#19-implementation-phases-and-agent-task-breakdown)
20. [Appendix A: Complete Smooth Class Catalog](#appendix-a-complete-smooth-class-catalog)
21. [Appendix B: Complete Family Catalog](#appendix-b-complete-family-catalog)
22. [Appendix C: Reference Test Cases](#appendix-c-reference-test-cases)

---

## Changelog

### v1.18 (February 2026) — Post-Seventeenth Review: Scope Freeze + Architecture Diagram + AD Strategy Reframing

External review identified execution risk as the primary concern: the gap between design and working library is enormous, and the original timeline didn't reflect that. This version freezes the v1.0 implementation scope, adds the missing architecture overview, and reframes the extended family AD strategy.

| Issue | Fix |
|---|---|
| **No explicit v1.0 implementation boundary — design implies everything ships together** | Section 1.2: v1.0 scope section with explicit "ships" table (dense-only, 4 families, 3+2 smooth types, REML/ML only), "does NOT ship" table with target versions, "what users cannot do" list for the README, and 32-cell validation surface analysis. |
| **Tier 1 table includes Sparse-CPU, Inverse Gaussian, P-splines, GCV/UBRE — too wide for v1.0** | Tier 1 (Section 1.1) updated to match v1.0 scope. Sparse-CPU, P-splines, Inverse Gaussian moved to Tier 2. GCV/UBRE deprioritized. Dependencies row added showing zero-C-compilation v1.0 install. |
| **19-week timeline is fantasy for numerical computing at this rigor level** | Section 1.2: realistic timeline of 25–34 weeks (~6–8 months) with phase breakdown. Explicitly 2–3× the original estimate. |
| **No single-page architecture overview — understanding requires reading 80 pages linearly** | Section 1.3: ASCII architecture diagram showing three phases (Setup/Fit/Post-estimation), phase boundaries (jax.device_put / np.asarray), data flow, v1.0 family/smooth scope, and future version hooks. Companion Mermaid diagram for interactive rendering. |
| **Tier 2/3 timelines and contents don't reflect what was deferred from Tier 1** | Tier 2 updated: absorbs Sparse-CPU, CHOLMOD, NB/Tweedie/Beta, fREML, bam(), P-splines, re/fs. Tier 3 updated: absorbs distributed SPMD, multi-host, out-of-core. Realistic timelines (months, not weeks). |
| **`custom_jvp` for all extended families is overly conservative and high-risk** | Section 9.1, 9.3, 9.4 rewritten. Per-family analysis shows 5/6 extended families (NB, Beta, Cox PH, SHASH, ordered categorical) work with standard `jax.grad` through stable forward passes. Only Tweedie's series evaluation genuinely needs `custom_jvp`. Reduces hand-derived gradient surface from 4+ families to 1. Propagated through exec summary, family docstrings, distributed section, testing, and implementation plan. |

---

### v1.17 (February 2026) — Post-Sixteenth Review: uv as Project Package Manager

| Issue | Fix |
|---|---|
| **CHOLMOD dependency hell — vendored wheel strategy requires building/maintaining platform-specific C library wheels** | Section 3.1: `uv` is the project package manager. `pyproject.toml` with optional extras (`sparse`, `gpu`, `distributed`, `full`). Pre-built scikit-sparse wheels hosted on `pymgcv-wheels` GitHub Pages index, referenced via `[tool.uv.sources]`. `uv sync --extra sparse` installs CHOLMOD with no C compiler. |
| **Multi-host version consistency required custom `_collect_version_pins()` with per-package iteration** | Section 16.8: `SetupManifest.version_pins` replaced with `uv_lock_hash` — SHA-256 of the `uv.lock` file. Single string comparison replaces per-package iteration. If all hosts ran `uv sync --frozen`, versions are identical by construction. Fallback hashes key package versions if `uv.lock` not found. |
| **Error messages reference `pip install scikit-sparse` — unhelpful for most users** | All error messages updated to reference `uv sync --extra sparse`. |
| **Dependency table doesn't show install extras or distinguish core from optional** | Section 3.1: dependency table updated with `Install extra` column. Clear separation of core (always installed) vs optional (sparse, gpu, distributed). |
| **No reproducible environment story for CI or multi-host clusters** | Section 3.1: `uv lock` generates cross-platform lockfile. `uv sync --frozen` on CI and cluster nodes guarantees identical environments. Docker image uses `uv sync --extra full --frozen`. |

---

### v1.16 (February 2026) — Post-Fifteenth Review: Dual Reviewer — Correctness, Packaging, Parser, Contracts

Two reviewers: R1 focused on performance cliffs and contract completeness; R2 focused on correctness failure modes and deployment risks.

| # | Source | Issue | Fix |
|---|---|---|---|
| 1 | R2#2 | **Formula parser uses regex — guaranteed bug factory for nested calls, interaction notation, operator precedence** | Section 13.1: complete rewrite using Python's `ast` module. `_SmoothExtractor` walks the AST to identify `Call` nodes for `s()`, `te()`, etc. Handles `s(x, k=int(log(n)))`, `y ~ a * s(x)`, and all nesting correctly. Regex eliminated. |
| 2 | R2#3 | **CHOLMOD dependency (`scikit-sparse`) is fragile — 80% of pip users will hit degraded mode** | Section 10.4: pre-built scikit-sparse wheels via uv package index (superseded vendored approach in v1.17). conda-forge path as alternative. Docker image with all dependencies. Degraded mode is edge case, not default. |
| 3 | R2#4 | **Step-halving in SPMD can diverge if decision variables differ across devices → collective deadlock** | Section 16.3: "Convergence decision broadcast" analysis. Current design is safe (all decision variables are post-all-reduce replicated scalars), but invariant explicitly stated and tested. Future refactoring must preserve "all decision variables are replicated" property. |
| 4 | R2#6 | **IFT backward pass for out-of-core REML can use "clean" H while β* came from jittered H → gradient explosion** | Section 16.5: `implicit_dbeta_dlambda` docstring now requires H_factor to be the exact same factorization (including jitter, pivoting, rank handling) used in forward solve. Violation analysis added. |
| 5 | R1#1,4 | **Execution path selection is opaque — users can't tell why they were routed or why multi-GPU is slow** | `GAMResult` gains `execution_path_reason`, `lambda_strategy`, `lambda_strategy_reason`, `routing_diagnostics` fields. `_routing_summary()` method for `summary()` output. Every automatic decision is explained and reversible. |
| 6 | R1#2 | **Routing cost model only counts X bytes — misses WX, XtWX, S_λ, Cholesky factor temporaries** | Section 16.1: `estimate_peak_memory()` computes full per-device budget: X_shard + WX_shard + 3×p²×8 replicated arrays + vectors. Host memory check before densification (psutil). `PeakMemoryEstimate` dataclass. |
| 7 | R1#3 | **SetupManifest checksum scope ambiguous — "what exactly is hashed?"** | Section 16.8: explicit field-by-field hash scope documented. Added `smooth_term_order`, `basis_types` fields. Per-package version pins checked for exact match across hosts (superseded by uv.lock hash in v1.17). |
| 8 | R2#1 | **"Missing middle" — no path for distributed sparse models (500k random effects)** | Section 16.6: gap explicitly acknowledged. Not supported in v1.0. Architectural hook identified: FactorBySmooth block independence enables future "block-parallel" mode. Listed as Tier 3 future work. |
| 9 | R2#5 | **fREML auto-switch at n_smooth=50 creates behavioral cliff vs R; switch points don't align with mgcv** | Section 16.6: divergence from R's switching explicitly documented. vs-R tests at fREML boundary use LOOSE tolerance. `lambda_strategy_reason` surfaces the switch. |
| 10 | R1-A | **Float64 mandatory = "GPU acceleration" only means data-center FP64 GPUs, not consumer cards** | Section 16.6: explicit caveat. Consumer GPUs have 1/32 FP64 throughput. "Reduced precision mode" is possible but not designed. |
| 11 | R1-B | **LOOSE tolerance can hide "wrong but plausible" outcomes — no hard-gate invariants** | Section 18.1: hard-gate invariants table (9 invariants). Objective monotonicity, H symmetry/PSD, rank conditions, EDF bounds, deviance non-negativity, no NaN in converged model, cross-path agreement always MODERATE. These block CI regardless of mgcv tolerance. |
| 12 | R1-C | **bam() O(p² + chunk_size × p) memory claim has no enforcement** | Section 10.5: explicit memory invariant. Three conditions that must hold. "No full X allocation in bam path" is a CI-enforced gate. |

---

### v1.15 (February 2026) — Post-Fourteenth Review: Performance Cliffs, Contracts, Operational Robustness

| Issue | Fix |
|---|---|
| **Densifying sparse X to test SPMD gates can OOM before routing decision is made** | Section 16.1: `route_execution_path()` estimates p, n_smooth, and dense bytes from Phase 1 metadata alone (no allocation). Dense X is only materialized after confirming the model stays on SPMD. Explicit performance expectation added: sparse-dominated models may be slower on multi-GPU SPMD than single-host Sparse-CPU. |
| **Coordinator broadcast is only half a contract — no verification that hosts assembled X identically** | Section 16.8: `SetupManifest` dataclass with SHA-256 checksum. Post-assembly `verify_local_assembly()` handshake on every host: checks column count, checksum integrity, and local-vs-global level consistency. Mismatch → immediate fail-fast error. SPMD invariant updated to reference verification. |
| **"Auto-switch to fREML" when n_smooth > 200 is a silent behavioral change** | Section 16.7: `auto_select_distributed_mode()` now returns `DistributedModeSelection` dataclass with explicit `lambda_strategy` and `lambda_strategy_reason` fields. Three-tier auto-selection: Newton REML (≤50), fREML (51–200), Fellner-Schall (>200), with cost anchoring. User-specified method is always respected (with warning if costly). `lambda_strategy_reason` is mandatory in `GAMResult`; `summary()` prints it. |
| **Scaling limits don't reflect dense S_λ commitment or dtype/determinism effects** | Section 16.7 scaling table updated: replicated memory per device computed as `3 * p² * 8` (XtWX + S_λ + Cholesky factor). Float64 mandatory noted. Determinism mode throughput caveat (10–30% reduction). Routing table adds λ strategy column. "Not a performance guarantee" caveat added. |
| **Empty-level, unseen-level, and dropped-level policies for factor-by in distributed unspecified** | Section 16.8: explicit policy table — zero-row levels kept (preserve column layout), novel prediction levels error with guidance to use `bs="fs"`. `SetupManifest` includes `empty_level_policy` field. |

---

### v1.14 (February 2026) — Post-Thirteenth Review: FactorBySmooth × Distributed Integration

| Issue | Fix |
|---|---|
| **SPMD path assumes dense X, but FactorBySmooth produces sparse block-diagonal X — no routing rule** | Section 16.1: explicit dense-only constraint. Sparse smooth types (FactorBySmooth, fs, re, mrf) are densified before `jax.device_put`. For factor-by models where densification pushes p above SPMD gates, route to Sparse-CPU or chunked. |
| **`auto_select_distributed_mode()` gates only on p, ignoring n_smooth blowup from per-level λ** | Section 16.7: `auto_select_distributed_mode()` now takes `n_smooth` parameter. n_smooth > 200 forces fREML (warns, does not error). Factor-by routing table added showing how p and n_smooth interact for realistic workloads. |
| **Distributed solver forms dense (p,p) H, making FactorBySmooth's "sparse throughout" story misleading** | Section 16.2: explicit note that block-diagonal penalty structure is NOT exploited on SPMD path. S_λ is densified for Phase 2. "Sparse throughout" is a Phase 1 (setup) property that avoids OOM during assembly; SPMD path is dense by design for moderate p. |
| **Setup determinism for factor-by not stated as distributed invariant** | SPMD invariants table (Section 16.3): new invariant "Identical setup outputs across hosts" — factor-level ordering, block-to-column mapping, and constraint absorption must be globally identical. Violation produces silent catastrophic error (incompatible column semantics across devices in all-reduce). |
| **Distributed knot placement broadcasts knots but not factor-level ordering** | Section 16.8: coordinator now also broadcasts canonical factor-level ordering. All processes use coordinator's ordering in `FactorBySmooth.setup()`. Same gather/broadcast pattern as knots, Phase 1 only. |
| **Scaling limits table missing n_smooth column** | Section 16.7 scaling limits table updated with n_smooth limit (200, warn + force fREML) alongside existing p limits. |

---

### v1.13 (February 2026) — Post-Twelfth Review: Factor `by` Smooth Mechanism

| Issue | Fix |
|---|---|
| **Factor `by` variable produces separate smooths per level, but the doc had no assembly or penalty specification** | New Section 5.7: full `FactorBySmooth` class — block-diagonal sparse design matrix (one block per level, no `toarray()`), one penalty per level embedded in global coefficient space, each with its own λ. Numeric `by` (varying-coefficient) also specified. |
| **Identifiability interaction between `s(x, by=fac)` and `s(x)` unspecified** | Section 5.7.3: three cases enumerated — factor-by alone (no constraint), factor-by alongside main-effect smooth (null-space absorption via QR), missing factor main effect (warning). Constraints recorded in `CoefficientMap`. |
| **REML outer loop dimension scales with factor levels but no guidance given** | Section 5.7.4: scaling table from 5 levels (trivial) to 500 parameters (O(125M) Newton cost). Explicit recommendation: switch to fREML/Fellner-Schall when n_smooth > 100. Runtime warning added. |
| **Formula parser had no dispatch between numeric and factor `by`** | Section 5.7.5: `resolve_by_variable()` routes factor `by` to `FactorBySmooth`, numeric `by` to pointwise multiplication. Factor detection uses dtype, never auto-promotes integer columns. |
| **Comparison with `bs="fs"` was implicit** | Explicit comparison table in 5.7.2: separate λ per level vs shared λ, independent estimation vs shrinkage, penalty count scaling, use-case guidance. |
| **Section numbering collision after insertion** | Downstream sections renumbered: Additional Smooth Classes → 5.8, Smooth Registry → 5.9, CoefficientMap → 5.10. Cross-reference on line 200 updated. |

---

### v1.12 (February 2026) — Post-Eleventh Review: SPMD Constraints, Setup Boundary, Lifecycle

| Issue | Fix |
|---|---|
| **Replicated solve is unstated inefficiency; no ceiling on p for SPMD** | Replicated solve acknowledged as conscious tradeoff. SPMD path gated: p ≤ 3000 single-host, p ≤ 2000 multi-host. Above threshold → single-device solve + broadcast β. |
| **Comms model is narrative, not a selector** | `auto_select_distributed_mode()` gate added: multi-host + p > threshold → error with guidance. Uses the existing cost model numbers. |
| **Determinism claim "same graph = same reduction tree" is overconfident** | Caveated: deterministic within single compilation, same device count, same topology. Cross-compilation determinism requires `set_deterministic(True)` + pinned versions (references Section 4.5). |
| **Distributed knot placement violates Phase 1/Phase 2 boundary** | Resolved: distributed setup gathers a subsample to coordinator (CPU/NumPy), broadcasts knots. Phase 1 stays CPU-only. Knot selection is NOT a JAX program. |
| **Ray bootstrap has no lifecycle invariants** | Added: no elastic membership, exactly-once `jax.distributed.initialize()`, straggler = collective hang → Ray health check kills job. Explicit "not supported" for worker restart mid-fit. |
| **Out-of-core REML "per chunk" differentiation is ambiguous** | Clarified: implicit fn thm differentiates the exact converged objective (not per-chunk), using accumulated H and β* post-convergence. Chunks only affect accumulation, not the differentiated function. |

---

### v1.11 (February 2026) — Post-Tenth Review: JAX-Native Distributed Architecture

**Section 16 completely rewritten.** The NumPy-based Dask/Ray provider architecture (v1.0–v1.10) is replaced with JAX-native SPMD parallelism.

| Old (killed) | New | Why |
|---|---|---|
| `DaskProvider` — NumPy workers, Python coordinator | `jax.sharding` SPMD — same `pirls_step_jax`, row-sharded X | NumPy workers broke JIT, autodiff, extended family AD |
| `RayProvider` — NumPy workers, Python coordinator | `JaxTrainer` bootstraps `jax.distributed`, then pure JAX SPMD | Ray orchestrates; JAX owns all computation |
| `StatisticsProvider` for all distributed paths | Only needed for out-of-core (data > device memory) | SPMD uses same function as single-GPU |
| `deterministic_reduce` with Kahan | XLA all-reduce (deterministic within single compilation; see v1.12 caveats) | Same compilation = same reduction tree |
| Python round-trip per PIRLS iteration | Eliminated — all devices run same XLA program | No serialization latency |

**Architecture tiers (revised):**

| Scale | Method | PIRLS | Autodiff | Extended family AD | Tier |
|---|---|---|---|---|---|
| Single GPU | `jax.jit` | Full JIT | Full | ✅ (jax.grad; Tweedie: custom_jvp) | 1 |
| Multi-GPU, one host | `jax.sharding` + Mesh | Full JIT (SPMD) | Full | ✅ per device | 2 |
| Multi-host cluster | `jax.distributed` + Ray | Full JIT (SPMD) | Full | ✅ per device | 2–3 |
| Out-of-core | `ChunkedJAXProvider` | JIT per chunk | Implicit fn thm | ✅ per chunk | 3 |

---

### v1.10 (February 2026) — Post-Ninth Review: Claim Calibration

| Issue | Fix |
|---|---|
| **Dense-GPU "≈ 5ms on A100" is a fantasy planning number** | Rewritten as "roofline best-case O(10ms), real cost higher due to kernel launch, HBM bandwidth, XLA graph boundaries." Framed as order-of-magnitude, not benchmark. |
| **"Correctness preserved" for degraded mode is overconfident** | Changed to "same objective within MODERATE tolerance vs CHOLMOD path" — dense cho_factor may differ in pivoting/fill-in ordering. |

No new mechanisms — claim calibration only.

---

### v1.9 (February 2026) — Post-Eighth Review: Footguns Closed

| Issue | Fix |
|---|---|
| **Degraded mode gates on p but not n×p — downstream X.toarray() can OOM** | Added second gate: `n * p * 8 > 500MB` also triggers hard error. `peak_bytes` now used in the actual branch condition, not just narrative. |
| **Stalled step-halving can livelock if max_iter is absent** | `max_iter` is now mandatory (no default=∞). Additionally: 3 consecutive stalled iterations (instability without progress) trigger early termination with `converged=False` and diagnostic message. |
| **PathTransferState validate() has no specified call frequency or test strategy** | validate() called at both creation and consumption (two calls per transfer). Test strategy added: property tests that randomize transfer timing, verify invariants, and check objective monotonicity post-transfer. |
| **Determinism testing story is a feature toggle, not a QA contract** | Section 18.1 now specifies: unit tests run default mode, cross-path and vs-R tests run default mode, CI determinism suite (separate job) runs `set_deterministic(True)` and checks STRICT reproducibility. No test depends on determinism it doesn't explicitly enable. |

---

### v1.8 (February 2026) — Post-Seventh Review: Wiring, Invariants, Budgets

| Issue | Fix |
|---|---|
| **Step-halving exhaustion not wired into instability counter** — spec says it's a detector but code doesn't increment | Fixed: exhaustion now increments `_instability_count` identically to Cholesky failure / NaN. Single code path for all three signals. |
| **PIRLS snippet uses `np.*` in what should be Dense-GPU (JAX) path** | Snippet explicitly labeled `⚠️ REFERENCE IMPLEMENTATION (NumPy)`. Added note that JAX path uses `jnp.*` equivalents with same logic. Production JAX PIRLS is in Section 4.2. |
| **Regularization jitter +1e-12/+1e-6 is scale-unaware** | Changed to `eps * trace(H) / p` (scale-relative). Jitter level recorded in `FitDiagnostics.regularization_applied` for surfacing in `summary()`. |
| **Step-halving fallback takes 1e-4 step unconditionally — can violate monotonicity** | Tiny step now validated: if `pen_dev_try > pen_dev_prev`, step is rejected, iteration marked as stalled (increments instability counter), beta unchanged. |
| **PathTransferState has no real invariants / state machine** | Full state machine added: 5 representation invariants (verified by `validate()`), explicit "what's recomputed vs carried", rollback rule if first sparse iteration diverges. |
| **Sparse-CPU degraded memory math is optimistic (32MB claim ignores temporaries)** | Replaced with actual budget: `3 * p² * 8` bytes (H + factor + temp) + `n * p * 8` for X. Threshold lowered to p ≤ 1500. Density gate removed — p threshold is sufficient. |
| **Kahan/deterministic reduce is a perf landmine if always-on** | Clarified: Kahan + sorted-key reduce only active under `set_deterministic(True)`. Default path uses standard tree-reduce. |

---

### v1.7 (February 2026) — Post-Sixth Review: Bailouts, Fallbacks, Invariants

| Issue | Fix |
|---|---|
| **Dense-GPU bailout uses weak diag-ratio estimator, checked only at iterations 0/3** | Replaced with Cholesky-failure + NaN + step-halving-exhaustion detection every iteration (zero extra cost). Diag-ratio kept as cheap supplementary warning. No Lanczos needed — the natural failure modes are the detector. |
| **Sparse-CPU degraded mode silently densifies, causing OOM in exactly the cases that need sparse** | Degraded mode now fails fast with clear error when `p > 2000` or `nnz(X) / (n*p) < 0.3`. Only small/dense problems get the fallback. |
| **Determinism contract claims bit-for-bit reproducibility that's impossible across driver/toolchain changes** | Reworded: "reproducible within tolerance on same stack+hardware+versions." CI determinism tests pin JAX/CUDA versions. |
| **Tweedie test tolerance atol=0.5 masks correctness bugs** | Replaced with stratified invariant tests: loglik monotonicity under step-halving (STRICT), AD gradient vs finite-diff (1e-5), deviance residual identity (MODERATE). Prediction tolerance stays LOOSE for hard families. |
| **Mid-fit path transfer (Dense-GPU → Sparse-CPU bailout) has no state spec** | `PathTransferState` dataclass: carries β, log_λ, iteration count. First iteration after transfer unconditionally accepted. CoefficientMap shared (path-independent). Penalty representations converted at boundary. |

---

### v1.6 (February 2026) — Post-Fifth Review: Contradictions, Allocations, Bailouts

**Contradictions resolved:**

| Issue | Fix |
|---|---|
| **Executive summary says "feature-complete" while Section 1.1 calls that a schedule trap** | Exec summary rewritten: "tiered Python port" with Tier 1 as initial release scope. No more "feature-complete" claim. |
| **Tier 1 "done" says Dense-GPU and Sparse-CPU "identical ±1e-10"** while tolerance section says MODERATE (1e-6) for cross-path | Tier 1 "done" criterion updated to MODERATE (1e-6) for cross-path agreement, matching tolerance strategy. |

**Allocation bombs removed:**

| Issue | Fix |
|---|---|
| **`_row_tensor(A, B)` allocates O(n × ka × kb)** — memory bomb for tensor products | Replaced with column-wise Kronecker: builds one column at a time, O(n) per column, O(n × ka × kb) total but O(n) peak. Chunked mode for n > 100k. |
| **Factor-smooth `toarray()` before LIL insert** defeats sparse path for large k/levels | Removed `toarray()`: uses `scipy.sparse.lil_matrix` direct assignment from sparse blocks via COO conversion. |
| **TPRS `np.diag(D_k ** -0.5)` allocates dense (k-M)×(k-M)** | Replaced with column scaling: `E_xk @ U_k * (D_k ** -0.5)[None, :]`. |

**Architecture hardening:**

| Issue | Fix |
|---|---|
| **Global `configure()` is a concurrency footgun** | Added `FitConfig` context manager for per-model config. Global `configure()` is default, `FitConfig` overrides per-call. |
| **Phase 2 boundary doesn't specify PyTree contract** | `StructuredPenalty` subclasses now declare `tree_flatten`/`tree_unflatten` for JAX PyTree registration. Static fields explicitly listed. |
| **Sparse-CPU depends on CHOLMOD (brittle packaging)** | Added degraded-mode fallback: `scipy.linalg.cho_factor` when CHOLMOD unavailable. Performance regresses ~5-10x; results agree within MODERATE tolerance (dense factorization may differ in pivoting). |
| **Dense-GPU bailout spec is aspirational** | Concrete spec: `cond_est = max(diag(H))/min(diag(H))` computed O(p), threshold 1e10, checked at iterations 0 and 3. Bailout re-routes to Sparse-CPU mid-fit. |

---

### v1.5 (February 2026) — Post-Fourth Review: Scope, Solvers, Determinism

**Structural risk reductions:**

| Issue | Fix |
|---|---|
| **"Feature-complete" scope is a schedule trap** — no parity tiers, no "done means X" | **Three-tier parity plan** (Section 1.1): Tier 1 (MVP) = tp/cr/ps + Gaussian/Binomial/Poisson/Gamma + REML/GCV, Dense-GPU + Sparse-CPU. Tier 2 = tensor/re/fs + NB/Tweedie/Beta + fREML/bam. Tier 3 = exotic (soap/Duchon/Cox/SHASH) + GAMM + distributed. Each tier has explicit "done" criteria. |
| **Dense-GPU solver is O(p³) normal equations without bailout** — conditioning issues, no stated strategy | **Solver strategy specified per path.** Dense-GPU: Cholesky on H = XtWX + S_λ (default), pivoted QR fallback on `LinAlgError`, condition-number check triggers Sparse-CPU re-route. Explicit bailout rules documented. |
| **Structured penalty log_det/trace not specified per type** — "we expose log_det" doesn't mean REML works | **Log-det/trace capability matrix** added to Section 10.2. Each `StructuredPenalty` subclass declares whether it supports exact `log_pseudo_det()`. Penalties without exact log-det route to Sparse-CPU for REML or use stochastic approximation. |
| **No determinism policy** — knot selection, distributed reduction, GPU non-associativity all produce flapping tests | **Global RNG policy** (Section 4.5): `pymgcv.set_seed(n)` seeds both NumPy and JAX PRNG. Setup-phase randomness uses `np.random.Generator` from global seed. Distributed reduction uses deterministic tree-reduce with Kahan compensation. |
| **Setup phase can OOM on large sparse terms** — identifiability SVD densifies term blocks | **Sparse-safe constraint discovery** added. `apply_joint_identifiability` uses randomized SVD (`scipy.sparse.linalg.svds`) when term blocks exceed 10k columns. Factor-smooth assembly stays sparse throughout. |
| **Basis implementations are Python-loop placeholders** — cubic spline and P-spline sketches are unvectorized | **Labeled as reference implementations** with explicit "must vectorize" requirements. Production path uses `scipy.interpolate.BSpline` (vectorized) or JAX `vmap` over knot intervals. |
| **Distributed accumulation has no precision/determinism contract** — nondeterministic reduce, no compensation | **Reduction protocol specified** (Section 16.1): deterministic sorted-key reduce with Kahan summation available via `set_deterministic(True)`. Default uses standard tree-reduce. Cost model added. |
| **Test tolerances not stratified** — GPU/BLAS differences cause either false failures or missed regressions | **Three tolerance classes** defined (Section 18.1): `STRICT` (1e-10, coefficient/deviance on CPU), `MODERATE` (1e-6, GPU vs CPU), `LOOSE` (1e-3, vs R mgcv). Per-quantity tolerance table added. |

---

### v1.4 (February 2026) — Post-Third Review: Numerical Rigor

**Fixes for high-risk correctness / performance holes:**

| Issue | Fix |
|---|---|
| **REML criterion mixes NumPy ops inside JAX autodiff** — `reml_criterion` used `np.linalg.cholesky`, `toarray()`, SciPy sparse sums, then was wrapped in `jax.grad`/`jax.hessian` which can't trace any of that | **Dual REML implementations.** `_reml_criterion_jax()` is pure `jax.numpy` end-to-end (receives dense `jax.Array` penalty, uses `jnp.linalg.slogdet` for log-dets). `_reml_criterion_numpy()` is the NumPy/SciPy reference. JAX path never touches `np.*` or `scipy.sparse` inside the traced function. |
| **PIRLS init uses `pen_dev_old = inf + beta @ S @ beta`** — inf arithmetic on iteration 0 poisons NaN into deviance comparisons | **Clean initialization.** First iteration is unconditionally accepted (no deviance comparison). `pen_dev_old` is set from the first accepted iterate, not from `inf`. State tuple simplified to `(beta, pen_dev, iteration, converged)` — one objective value, not two. |
| **`PenaltySet.to_dense_jax()` eagerly densifies to (p,p)** despite narrative promising structured representations | **`StructuredPenalty` protocol replaces eager densification.** Dense-GPU solver accepts `penalty.matvec(beta)` and `penalty.log_det()` instead of a materialized matrix. Diagonal penalties stay as vectors. Kronecker penalties apply via factor chain. Dense materialization only as explicit fallback (`penalty.to_dense()`). |
| **Auto path selector ignores penalty shape** — MRF with 50k levels hits `ValueError` in `to_dense_jax()` instead of routing to Sparse-CPU | **Selector now queries `penalty_set.has_large_penalty()`** before choosing Dense-GPU. Late failure replaced with early routing. |
| **EDF trace via `np.linalg.inv(H)` is unstable and wasteful** | **Replaced with Cholesky-based solve:** `tr(H⁻¹ XtWX) = tr(L⁻¹ XtWX L⁻ᵀ) = ‖L⁻¹ XtWX_chol‖²_F` computed without forming H⁻¹. |
| **"JAX purity boundary" definition fuzzy** — unclear which modules run at setup-time (CPU/NumPy OK) vs JIT-time (pure JAX required) | **Explicit two-phase architecture documented** in Section 4.4: "Setup phase" (knot selection, basis construction, constraint computation) runs on CPU with NumPy/SciPy. Only dense arrays + static metadata cross into "JIT phase". |
| **Outer Newton for λ has no actual trust region** despite claiming one | **Damped Newton with eigenvalue truncation specified.** Hessian eigenvalues floored at `max(eig)/1000` before inversion. Step norm capped. Acceptance test: REML must decrease or step is rejected and damping increases. |

---

### v1.3 (February 2026) — Post-Second Review: Interface Hardening

**Fixes for high-risk holes identified in second review:**

| Issue | Fix |
|---|---|
| **JAX/SciPy leakage** — NB log-lik used `scipy.special.gammaln` in what is supposed to be a JAX-differentiable path; Tweedie series body was `pass` | **JAX purity boundary enforced.** Hard rule: all `*_jax.py` and `autodiff/` modules import zero SciPy. NB now uses `jax.scipy.special.gammaln`. Tweedie series implemented with `jax.lax.while_loop`. CI lint guard added. All family code examples updated. |
| **ExtendedFamily API internally inconsistent** — `ll_derivatives_autodiff` called undefined `ad`, `working_weights` referenced unset `self._y`/`self._scale`, no clear contract | **Single canonical contract chosen: Option B.** Extended families provide `loglik_per_obs(eta_i, y_i, theta) → scalar` as a pure JAX function. The framework owns all differentiation, stabilization (damping, clipping), and conversion to working weights/response. No family ever computes its own derivatives. |
| **StatisticsProvider insufficient for smoothing selection** — `(XtWX, XtWz)` not enough for REML log-determinants, EDF traces, AIC/BIC | **Provider contract extended** to `IterationStatistics` dataclass returning `XtWX`, `XtWz`, `deviance`, `log_likelihood`, and `n_obs`. Trace/log-det computed from the p×p `H = XtWX + S_λ` (which the provider doesn't need to know about). Provider outputs explicitly documented per execution path. |
| **`gam_side` heuristic threshold + in-place mutation + dense blowup** | **Replaced with `CoefficientMap` layer.** All constraints/reparameterizations produce explicit linear operators recorded in a global `CoefficientMap`. Overlap detection uses SVD-based rank test (no arbitrary threshold). No in-place `term_info` mutation. Predict/summary always go through `CoefficientMap`. |
| **TPRS `data_n` undefined, O(k³) eigendecomp unscalable, `abs(D_k)` hides sign errors** | Fixed: `data_n` → `n`. Added explicit `k ≤ 2000` ceiling with error for larger. Negative eigenvalues now raise instead of being silently abs'd. |
| **FactorSmooth `X_level[~mask] = 0` invalid for sparse** | Replaced with per-level evaluation + sparse row-assembly via `scipy.sparse.lil_matrix`. |
| **`InverseSquaredLink` referenced but undefined** | Added concrete implementation. |
| **GPU dense path vs scipy.sparse penalties unresolved** | **Penalty representation is now per-path.** Dense-GPU path stores penalties as dense `jax.Array`. Sparse-CPU path stores as `scipy.sparse.csc_matrix`. `PenaltySet` object handles conversion at path boundaries. |

**New architectural elements:**

- **`CoefficientMap`** (Section 5.10): immutable record of all reparameterizations, used by predict/summary/diagnostics. Replaces in-place `term_info` mutation.
- **`JAX purity boundary`** (Section 4.4): CI-enforced import guard. JAX path modules cannot import scipy/numpy (except `numpy` for type annotations).
- **`PenaltySet`** (Section 10.1): path-aware penalty container with `.to_dense_jax()` / `.to_sparse_scipy()` / `.as_structured()` methods.
- **`IterationStatistics`** (Section 7.1): extended provider return type sufficient for REML/GCV/EDF computation.

---

### v1.2 (February 2026) — Post-Review Revision

**Breaking architectural changes from v1.0:**

| Change | Rationale |
|---|---|
| **JAX-first, single-backend design** replaces the multi-backend `ArrayBackend` Protocol | The v1.0 unified interface masked irreconcilable execution model differences (JAX tracing vs. PyTorch eager). Python control flow inside `jax.jit` silently produces wrong results. Maintaining 3 backends triples test surface with zero user benefit. NumPy+SciPy is retained as a reference/fallback only. |
| **Closed-form derivatives for standard families**; autodiff restricted to REML and new extended families | Standard families gain nothing from AD since V(μ) gives working weights in O(1). Extended families use `jax.grad` through numerically stable forward passes — `lgamma`, `logsumexp`, log-space arithmetic make derivatives stable by construction (see Section 9.3 for per-family analysis). Only Tweedie's series evaluation requires `custom_jvp` due to truncation-dependent terms. |
| **Three explicit execution paths** replace transparent sparse/dense switching | JAX sparse is experimental and cannot JIT. `scipy.sparse.linalg.lsqr` forces CPU round-trips from GPU. Transparent switching produced neither good GPU perf nor good sparse perf. Users/auto-selector now choose: Dense-GPU, Sparse-CPU, or Chunked-Hybrid. |
| **`StatisticsProvider` protocol** decouples PIRLS from data layout | v1.0 PIRLS took raw `X` arrays, preventing distributed/streaming use without rewriting the fitting loop. Now PIRLS operates on `(XtWX, XtWz)` sufficient statistics, making distributed execution a data-access swap. |
| **`formulaic` for parametric terms** replaces the custom parser | R formula semantics (contrasts, `*` expansion, `(a+b)^2`, `.`, `I()`) have decades of edge cases. `formulaic` handles this; we only write the smooth-term preprocessor. Budget increased from 1 week to 3 weeks. |
| **Joint identifiability constraints** (`gam_side`) added post-assembly | v1.0 applied sum-to-zero per-smooth, which produces rank-deficient models when `te(x1,x2)` overlaps `s(x1)+s(x2)`. Now mirrors mgcv's `gam.side` iterative absorption. |
| **PIRLS convergence hardened** following mgcv's battle-tested logic | v1.0 used simple deviance-change check. Now tracks penalized deviance, coefficient change, has special early-iteration handling, clamped weight floors for binomial boundary cases, and a trust-region fallback. |

**New sections:**

- **Section 15: Model Comparison, Concurvity, and Diagnostics** — AIC/BIC infrastructure, `anova.gam`, concurvity detection.
- **Section 16: Distributed and Multi-Device Compute** — JAX-native SPMD via `jax.sharding` (multi-GPU) and `jax.distributed` (multi-host), Ray bootstrap for clusters, `ChunkedJAXProvider` for out-of-core, implicit function theorem for out-of-core REML.
- Updated implementation phases (Section 19) add distributed compute phase and extend formula parser timeline.

---

## 1. Executive Summary

PyMGCV is a tiered Python port of Simon Wood's R package `mgcv` (Mixed GAM Computation Vehicle). The initial release (Tier 1) provides production-quality Generalized Additive Models with the most-used smooth classes (thin-plate, cubic, P-spline), standard exponential families (Gaussian, Binomial, Poisson, Gamma), and REML/GCV smoothing parameter estimation. Subsequent tiers add tensor products, extended families, `bam()` for large data, and exotic smooths — each independently shippable. See Section 1.1 for the full tier plan.

**Key design differentiators from a naive port:**

- **JAX as the sole first-class backend.** All performance-critical code is written in JAX, compiled via XLA, and targets CPU, CUDA, Metal, and ROCm. A pure NumPy+SciPy reference implementation exists for testing and for environments where JAX cannot be installed, but it is not optimized and does not support AD or JIT. PyTorch and PyTensor are **not** supported as compute backends — they add nothing JAX doesn't provide and triple the test surface. Interop utilities are provided at the boundary for users who need conversion.
- **Selective automatic differentiation** replaces hand-coded derivatives where it is both safe and beneficial. AD via `jax.grad` is used for REML/ML criterion derivatives w.r.t. smoothing parameters (small-dimensional, numerically benign, hard to hand-code). Standard exponential families retain closed-form variance functions `V(μ)` for working weights — AD adds overhead with no benefit here. Extended families (NB, Beta, Cox PH, SHASH, etc.) implement `log_likelihood` using numerically stable JAX primitives (`lgamma`, `logsumexp`, log-space arithmetic) and rely on standard `jax.grad` — if the forward pass is stable, the derivative is automatically stable. Only Tweedie requires `jax.custom_jvp` due to its series evaluation where truncation-dependent terms make naive AD unreliable (see Section 9.3).
- **Three explicit execution paths** instead of transparent sparse/dense switching: (1) Dense-GPU for n < ~200k with full JIT, (2) Sparse-CPU via SciPy+CHOLMOD for large n or high-dimensional smooths, (3) Chunked-Hybrid for n > ~1M combining GPU-accelerated per-chunk computation with CPU accumulation.
- **Compiled inner loops** via JAX's XLA compilation, with Cython fallbacks for the NumPy reference backend.
- **`StatisticsProvider` protocol** decouples PIRLS from data layout for in-memory and out-of-core execution. For multi-device/distributed compute, JAX-native SPMD (Section 16) is used instead — the same `pirls_step_jax` function works with sharded arrays.
- **Memory-efficient algorithms** mirroring Wood's discretization, marginal discretization, and chunk-based processing for datasets with millions of rows.

### 1.1 Parity Tiers: "Done" Means X

**v1.5:** "Feature-complete mgcv port" is a schedule risk disguised as scope. The combinatorial surface (smooth × family × method × path × constraints) makes "complete" a moving target. Instead, we define three parity tiers with explicit "done" criteria. Each tier is independently shippable.

**Tier 1 — MVP (v1.0, ~6–8 months).** Covers ~80% of real-world mgcv usage. Dense-only. See Section 1.2 for the full scoping rationale.

| Dimension | Included | "Done" criterion |
|---|---|---|
| Smooths | `tp`, `ts`, `cr`, `cs`, `cc`, `te`, `ti`, `s(x, by=fac)` | Coefficients match R ±1e-6 on 10 reference datasets |
| Families | Gaussian, Binomial, Poisson, Gamma | Deviance matches R ±1e-8 |
| Links | identity, log, logit, inverse, probit, cloglog, sqrt | Link/inverse/derivative match R ±1e-12 |
| Methods | REML, ML | λ matches R ±1e-4; REML score ±1e-6 |
| Paths | Dense-GPU (JAX), Dense-CPU (NumPy reference) | Cross-path agreement within MODERATE (1e-6) |
| Features | `gam()`, `predict()`, `summary()`, `plot()` basics | EDF, p-values, CI match R ±1e-4 |
| Constraints | Sum-to-zero, `gam_side` identifiability | CoefficientMap predict roundtrip exact |
| Dependencies | JAX, NumPy, SciPy, formulaic, matplotlib | `uv sync` — no C compilation, no optional extras |

**Tier 2 — Production (v1.1, ~4–6 months after v1.0).** Adds sparse path, big-data, extended families. Items deferred from Tier 1 land here.

| Dimension | Added | "Done" criterion |
|---|---|---|
| Smooths | `ps`, `cp`, `re`, `fs`, `by`-variable improvements | Tensor EDF matches R ±1e-3 |
| Families | NB, Tweedie, Beta, Inverse Gaussian, ordered categorical | AD gradients validated vs finite-diff ±1e-5; Tweedie custom_jvp validated separately |
| Methods | fREML, Fellner-Schall, GCV, UBRE | λ matches R ±1e-3 (relaxed for approximate methods) |
| Paths | Sparse-CPU (CHOLMOD via `uv sync --extra sparse`), Chunked-Hybrid | Matches Dense-GPU results ±1e-6 |
| Features | `bam()`, `anova.gam`, concurvity, path transfer (Dense→Sparse bailout) | bam on 10M rows completes in <5min |

**Tier 3 — Advanced (v1.2+, incremental after v1.1).** Distributed compute, exotic smooths, GAMM. May ship incrementally.

| Dimension | Added | "Done" criterion |
|---|---|---|
| Smooths | `gp`, `mrf`, `so`, `ad`, Duchon splines, `bs`, `t2` | Correctness vs R; some may be ±1e-2 |
| Families | Cox PH, SHASH, `gaulss`, `ziplss`, `mvn` | Each validated on 3+ reference datasets |
| Methods | GAMM via PQL | Matches R `gamm()` ±1e-2 (known approximation) |
| Distributed | SPMD multi-GPU, Ray multi-host, out-of-core | Matches in-memory results ±1e-5 |
| Features | Streaming, online updates, Stan/NumPyro export | Functional, not necessarily perf-optimized |

**Tier boundaries are hard:** no Tier 2 feature blocks a Tier 1 release. If Tier 3 slips indefinitely, Tier 1+2 is still a useful library.

### 1.2 v1.0 Implementation Scope (v1.17)

The tier plan above describes the *full library vision*. This section describes what actually ships as v1.0 — a deliberately narrow cut that is useful, testable, and honest about its limits.

**The scoping principle:** every dimension is cut to the minimum that produces a library people would actually use for real work, while keeping the architecture compatible with the full vision. Nothing in v1.0 forecloses a future feature — it's additive, not rearchitectural.

#### What ships in v1.0

| Dimension | v1.0 scope | Rationale |
|---|---|---|
| **Execution paths** | Dense-GPU (JAX) + Dense-CPU (NumPy reference) | Zero exotic dependencies. `uv sync` works everywhere. No CHOLMOD, no Ray, no Dask. |
| **Families** | Gaussian, Binomial, Poisson, Gamma | ~90% of applied GAM usage. All have closed-form working weights — no AD needed for inner loop. |
| **Smooths** | TPRS (`tp`/`ts`), cubic regression (`cr`/`cs`/`cc`), tensor products (`te`/`ti`), factor-by (`s(x, by=fac)`) | The workhorses. Factor-by is an assembly pattern, not a new basis type — low marginal cost. |
| **Links** | identity, log, logit, inverse, probit, cloglog, sqrt | All standard links for the four families. |
| **Methods** | REML, ML | Newton optimizer with exact Hessian. GCV/UBRE are trivial to add but lower priority. |
| **Features** | `gam()`, `predict()`, `summary()`, `plot()` | Core API. No `bam()`, no `gamm()`, no `anova.gam`. |
| **Constraints** | Sum-to-zero, `gam_side` identifiability via `CoefficientMap` | Required for correctness of multi-smooth models. |
| **p ceiling** | ~5000 (GPU), ~2000 (CPU dense) | Covers nearly all practical GAM models that don't involve high-cardinality random effects. |
| **n ceiling** | ~10M (dense X must fit in host memory + one GPU) | For larger n, users wait for `bam()` in v1.1. |
| **Dependencies** | JAX, NumPy, SciPy, formulaic, matplotlib | `uv sync` with no extras. No C compilation. |

#### What does NOT ship in v1.0

| Deferred | Why | Target |
|---|---|---|
| **Sparse-CPU path** (CHOLMOD) | CHOLMOD dependency hell. Pre-built wheel infrastructure needs time. Dense-only v1.0 has zero packaging risk. | v1.1 |
| **`bam()` / chunked path** | Requires discretization machinery (Section 10.9), chunk provider, fREML. Large implementation surface. | v1.1 |
| **Extended families** (NB, Tweedie, Beta, SHASH, Cox PH) | Each needs stable forward-pass implementation and thorough finite-difference validation across extreme parameter regions. Tweedie additionally requires `custom_jvp` for its series evaluation. Manageable risk, but best done with a working v1.0 baseline. | v1.1 |
| **fREML / Fellner-Schall** | Approximation-based optimizers. Need the exact Newton REML baseline to validate against. | v1.1 |
| **P-splines** (`ps`/`cp`) | Lower priority than TPRS and cubic. Easy to add but increases test surface. | v1.1 |
| **`bs="re"` / `bs="fs"`** | Random effects and factor-smooth interactions need Sparse-CPU for realistic cardinalities. | v1.1 |
| **Multi-GPU SPMD** (Section 16) | Entire distributed story — Ray bootstrap, SPMD sharding, SetupManifest, multi-host. | v1.2 |
| **Out-of-core** (ChunkedJAXProvider) | Data-larger-than-memory. Requires IFT for REML, chunk streaming. | v1.2 |
| **`gamm()` via PQL** | Notoriously tricky. Needs `lme4`-style mixed model machinery. | v1.2+ |
| **Exotic smooths** (soap film, MRF, adaptive, Duchon, GP) | Niche. Each is a standalone implementation effort. | v1.2+ |

#### What users cannot do with v1.0

This list must appear in the README, not buried in a design doc:

1. **Fit models with > ~5000 basis functions.** No sparse solver. Factor-by with many levels or large tensor products will hit the dense memory ceiling.
2. **Fit negative binomial, Tweedie, or other count/continuous mixture models.** Only Gaussian, Binomial, Poisson, Gamma.
3. **Fit models on datasets with > ~10M rows.** No chunked processing. Dense X must fit in memory.
4. **Use random effects (`bs="re"`) or factor-smooth interactions (`bs="fs"`).** These need sparse linear algebra for realistic cardinalities.
5. **Distribute fitting across multiple GPUs or hosts.** Single-device only.
6. **Fit GAMMs with correlated random effects.** No `gamm()`.

Each limitation has a clear path to resolution in a named future version. Users can make informed decisions about whether v1.0 meets their needs.

#### v1.0 validation surface

The scoped v1.0 has a manageable comparison surface against R:

```
3 smooth types × 4 families × 1 execution path × 1 optimizer = 12 cells
+ tensor products (te/ti): 2 × 4 × 1 × 1 = 8 cells
+ factor-by: 3 × 4 × 1 × 1 = 12 cells (same smooths, by-variable)
Total: ~32 cells to validate exhaustively against R
```

Compare with the full spec: ~(12 smooth types × 15 families × 3 paths × 3 optimizers) = ~1,620 cells. The scoped v1.0 is 2% of the full surface — each cell can be hand-checked.

#### Timeline reality check

The tier plan says "Weeks 1–13" for Tier 1. Seventeen review rounds have already surfaced fundamental redesigns (distributed architecture rewritten in v1.11, formula parser replaced in v1.16). Implementation will surface more. A realistic v1.0 timeline for experienced numerical computing engineers:

| Phase | Duration | Focus |
|---|---|---|
| Foundation (basis, penalty, link, family) | 8–10 weeks | Get the math right. TPRS eigendecomposition, cubic spline construction, penalty matrices. Every component validated against R individually. |
| Fitting (PIRLS, REML Newton, convergence) | 6–8 weeks | The hardest part. Step-halving, jitter, convergence detection, identifiability constraints. Edge cases in every family × basis combination. |
| Assembly (formula parser, design matrix, CoefficientMap) | 4–6 weeks | AST parser, factor-by expansion, constraint absorption. End-to-end `gam()` call. |
| API + diagnostics (predict, summary, plot) | 3–4 weeks | predict with SEs, EDF computation, p-values, basic plotting. |
| Testing + hardening | 4–6 weeks | 32-cell R comparison, edge cases, CI setup, documentation. |
| **Total** | **25–34 weeks** | **~6–8 months** |

This is 2–3× the original estimate. The original was optimistic; this reflects the actual complexity of numerical computing at this level of rigor.

### 1.3 Architecture Overview (One-Page Diagram)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER API                                   │
│  gam("y ~ s(x1) + s(x2, by=fac) + te(x3,x4)", data, family)      │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │    FORMULA PARSER     │
          │  (AST-based, §13.1)  │
          │                       │
          │  "y ~ s(x1) + ..."   │
          │    ↓                   │
          │  SmoothSpecs[]        │
          │  + parametric → formulaic
          └───────────┬───────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │        PHASE 1: SETUP (CPU)       │
    │         NumPy only — no JAX       │    ← CI guard: no jax imports
    │                                     │
    │  ┌─────────┐  ┌──────────────┐    │
    │  │ Smooth  │  │  Penalty     │    │
    │  │ Setup   │  │  Matrices    │    │
    │  │ (knots, │  │  (S_j per    │    │
    │  │  basis) │  │   smooth)    │    │
    │  └────┬────┘  └──────┬───────┘    │
    │       │              │             │
    │  ┌────▼──────────────▼──────────┐ │
    │  │   MODEL MATRIX ASSEMBLY      │ │
    │  │   X_parametric | X_smooth    │ │
    │  │   + FactorBySmooth expansion │ │
    │  │   + CoefficientMap (§5.10)   │ │
    │  └──────────────┬───────────────┘ │
    │                 │                  │
    │  ┌──────────────▼───────────────┐ │
    │  │    ROUTING DECISION          │ │
    │  │    estimate_peak_memory()    │ │
    │  │    (no dense allocation)     │ │
    │  │                              │ │
    │  │  v1.0: Dense-GPU or         │ │
    │  │        Dense-CPU only        │ │
    │  └──────────────┬───────────────┘ │
    └─────────────────┼─────────────────┘
                      │
    ══════════════════╪══════════ Phase boundary ═══════════
                      │
    ┌─────────────────▼─────────────────┐
    │      PHASE 2: FIT (JAX JIT)       │
    │                                     │
    │  ┌────────────────────────────┐    │
    │  │  REML OUTER LOOP (Newton)  │    │
    │  │  minimize V(λ) w.r.t. λ    │    │
    │  │  jax.grad(V) for gradient  │    │
    │  │                            │    │
    │  │  ┌──────────────────────┐  │    │
    │  │  │  PIRLS INNER LOOP   │  │    │
    │  │  │  jax.lax.while_loop │  │    │
    │  │  │                     │  │    │
    │  │  │  η = X @ β          │  │    │
    │  │  │  μ = g⁻¹(η)        │  │    │
    │  │  │  W = V(μ)⁻¹        │  │    │
    │  │  │  z = working resp   │  │    │
    │  │  │  H = XᵀWX + S_λ    │  │    │
    │  │  │  β = H⁻¹ Xᵀ Wz    │  │    │
    │  │  │                     │  │    │
    │  │  │  Step-halving:      │  │    │
    │  │  │  accept if          │  │    │
    │  │  │  dev↓ or stall→stop │  │    │
    │  │  └──────────────────────┘  │    │
    │  │                            │    │
    │  │  λ_new = λ - H_V⁻¹ ∇V    │    │
    │  └────────────────────────────┘    │
    │                                     │
    │  Instability detection:             │
    │  Cholesky fail / NaN / halving      │
    │  exhaust → instability_count++      │
    │  Scale-relative jitter: ε·tr(H)/p  │
    └─────────────────┬─────────────────┘
                      │
    ══════════════════╪══════════ Back to CPU ═══════════
                      │
    ┌─────────────────▼─────────────────┐
    │     PHASE 3: POST-ESTIMATION      │
    │     (CPU, NumPy, CoefficientMap)  │
    │                                     │
    │  predict() ← CoefficientMap       │
    │  summary() ← EDF, p-values, SEs  │
    │  plot()    ← matplotlib           │
    │                                     │
    │  GAMResult:                        │
    │    .coefficients    .edf           │
    │    .smoothing_params               │
    │    .execution_path_reason          │
    │    .lambda_strategy_reason         │
    └───────────────────────────────────┘


DATA FLOW (v1.0 only):

  User data (DataFrame)
       │
       ▼
  [Formula Parse] ──→ SmoothSpec[] + parametric terms
       │
       ▼
  [Basis Construction] ──→ X (dense, NumPy)  ← Phase 1
  [Penalty Construction] ──→ S_j[] (dense, NumPy)
       │
       ▼
  [jax.device_put] ──→ X, S_λ on GPU  ← Phase 1→2 boundary
       │
       ▼
  [PIRLS + REML] ──→ β*, λ*, converged  ← Phase 2 (JIT)
       │
       ▼
  [np.asarray] ──→ β, Vp on CPU  ← Phase 2→3 boundary
       │
       ▼
  [predict/summary/plot] ──→ GAMResult  ← Phase 3


FUTURE PATHS (not in v1.0, designed for):

  ┌─────────────────────────────┐
  │  v1.1: Sparse-CPU           │ X stays sparse throughout
  │  + CHOLMOD via uv            │ S_λ sparse, cho_factor sparse
  │  + NB/Tweedie/Beta           │ stable forward + jax.grad
  │  + bam() chunked             │ ChunkedProvider, fREML
  │  + P-splines, re, fs         │ New smooth types
  └─────────────────────────────┘
  ┌─────────────────────────────┐
  │  v1.2: Distributed           │ Same pirls_step_jax
  │  + SPMD multi-GPU            │ X row-sharded, XtWX all-reduce
  │  + Ray multi-host            │ jax.distributed.initialize()
  │  + Out-of-core               │ ChunkedJAXProvider + IFT
  └─────────────────────────────┘
  ┌─────────────────────────────┐
  │  v1.2+: Exotic               │
  │  + gamm() via PQL            │ lme4-style mixed model
  │  + Soap film, MRF, adaptive  │ Standalone smooth impls
  │  + Cox PH, SHASH, gaulss     │ Location-scale families
  └─────────────────────────────┘
```

---

## 2. Background: What mgcv Does

### 2.1 Core Problem

mgcv fits models of the form:

```
g(μ_i) = A_i θ = X_i β + f_1(x_{1i}) + f_2(x_{2i}, x_{3i}) + ... + Z_i b
```

where `g` is a link function, `f_j` are smooth functions represented as basis expansions `f_j(x) = Σ_k β_{jk} B_{jk}(x)`, each with an associated wiggliness penalty `λ_j β_j^T S_j β_j`, `X_i β` are parametric (fixed) effects, and `Z_i b` are random effects (which are also representable as penalized smooth terms).

### 2.2 What "Feature Complete" Means

We must support:

- **All 20+ smooth classes**: `tp` (thin plate), `ts` (thin plate with shrinkage), `cr` (cubic regression), `cs`, `cc` (cyclic), `ps` (P-splines), `cp` (cyclic P-splines), `ad` (adaptive), `bs` (B-splines), `gp` (Gaussian process), `mrf` (Markov random field), `re` (random effects), `fs` (factor-smooth interactions), `t2` (tensor product type 2), `te`/`ti` (tensor products and tensor interactions), `so` (soap film), `sz` (Duchon splines), linear functional terms, and others.
- **All 30+ distribution families**: Gaussian, Binomial, Poisson, Gamma, Inverse Gaussian, Negative Binomial (nb, negbin), Tweedie (tw), Beta, Ordered Categorical (ocat), Categorical (multinom), Zero-Inflated Poisson (zip), Cox PH (cox.ph), Scaled t, SHASH, GEVD, ZAGA, ZIPL, and all `extended.family` classes.
- **All fitting methods**: `gam()`, `bam()` (for big data), `gamm()` (via PQL/REML mixed model), `jagam()` (Bayesian via JAGS—we will provide Stan/NumPyro export instead).
- **Smoothness estimation**: GCV, UBRE, REML, ML, fREML (fast REML for bam).
- **All link functions per family**: identity, log, inverse, logit, probit, cloglog, sqrt, and family-specific links.
- **Model comparison and selection**: AIC, BIC, concurvity, `anova.gam`, hypothesis testing.
- **Visualization**: `plot.gam` equivalent functionality.

### 2.3 Key Algorithms in mgcv

| Algorithm | Purpose | Source Function |
|---|---|---|
| PIRLS | Inner loop: weighted penalized least squares | `gam.fit3`, `gam.fit5` |
| Newton/Outer iteration | Optimize smoothing parameters | `newton` in `gam.fit3` |
| Fellner-Schall | Fast smoothing parameter update | `fast.REML.fit` |
| Wood's stable QR | Numerically stable basis expansion | `qr.update` |
| Discretization | Memory-efficient large-data handling | `bam` internals |
| PQL | Penalized quasi-likelihood for GAMMs | `gamm` |
| EFS | Extended Fellner-Schall for extended families | `gam.fit5` |

---

## 3. High-Level Architecture

```
pymgcv/
├── __init__.py                    # Public API: gam(), bam(), gamm()
├── api.py                         # Top-level fitting functions
├── formula/
│   ├── __init__.py
│   ├── parser.py                  # Formula string parser (Wilkinson notation)
│   ├── terms.py                   # Term representation: smooth, parametric, random
│   └── design.py                  # Design matrix construction
├── smooths/
│   ├── __init__.py
│   ├── base.py                    # Abstract SmoothSpec / Smooth base classes
│   ├── tprs.py                    # Thin plate regression splines (tp, ts)
│   ├── cubic.py                   # Cubic regression splines (cr, cs, cc)
│   ├── pspline.py                 # P-splines (ps, cp)
│   ├── bspline.py                 # B-splines (bs)
│   ├── tensor.py                  # Tensor product smooths (te, ti, t2)
│   ├── adaptive.py                # Adaptive smooths (ad)
│   ├── gaussian_process.py        # GP smooths (gp)
│   ├── mrf.py                     # Markov random fields (mrf)
│   ├── soap_film.py               # Soap film smooths (so)
│   ├── duchon.py                  # Duchon splines (sz)
│   ├── random_effects.py          # re, fs smooth classes
│   ├── linear_functional.py       # Linear functional terms
│   └── registry.py                # Smooth class registry & dispatch
├── families/
│   ├── __init__.py
│   ├── base.py                    # Family / ExponentialFamily base class
│   ├── standard.py                # Gaussian, Binomial, Poisson, Gamma, InvGauss
│   ├── extended.py                # ExtendedFamily base with full log-lik interface
│   ├── negbin.py                  # Negative binomial (nb, negbin)
│   ├── tweedie.py                 # Tweedie (tw)
│   ├── beta_family.py             # Beta regression (betar)
│   ├── ordered_categorical.py     # Ordered categorical (ocat)
│   ├── multinomial.py             # Multinomial (multinom)
│   ├── zero_inflated.py           # ZIP, ZAGA, ZIPL
│   ├── survival.py                # Cox PH (cox.ph)
│   ├── location_scale.py          # gaulss, gammals, gevlss, shash, etc.
│   ├── scat.py                    # Scaled t (scat)
│   └── registry.py                # Family registry & dispatch
├── links/
│   ├── __init__.py
│   └── links.py                   # Link functions: logit, probit, cloglog, log, etc.
├── fitting/
│   ├── __init__.py
│   ├── pirls.py                   # PIRLS inner loop (gam.fit3 / gam.fit5 equivalent)
│   ├── newton.py                  # Outer Newton iteration for λ
│   ├── fellner_schall.py          # Fellner-Schall updates (fast REML)
│   ├── reml.py                    # REML / ML / GCV / UBRE criteria
│   ├── bam_fit.py                 # bam()-specific: discretize, fREML, chunk processing
│   ├── gamm_fit.py                # gamm() via PQL
│   ├── initialization.py          # Starting value computation
│   ├── convergence.py             # Convergence criteria and diagnostics
│   └── constraints.py             # Sum-to-zero and identifiability constraints
├── penalties/
│   ├── __init__.py
│   ├── penalty.py                 # Penalty matrix construction and manipulation
│   └── selection.py               # Extra shrinkage penalties, null space penalties
├── linalg/
│   ├── __init__.py
│   ├── backend.py                 # JAX-first backend; NumPy reference fallback
│   ├── qr.py                      # Pivoted QR, stable Householder
│   ├── cholesky.py                # Penalized Cholesky, sparse Cholesky
│   ├── sparse_ops.py              # Sparse matrix utilities
│   ├── woodbury.py                # Woodbury identity for efficient updates
│   └── eigen.py                   # Eigendecompositions for TPRS etc.
├── autodiff/
│   ├── __init__.py
│   ├── jax_ad.py                  # JAX grad/hessian/hvp wrappers
│   ├── custom_jvp_rules.py        # Tweedie series custom JVP (only family needing it)
│   └── interface.py               # Thin wrapper (JAX only; no multi-backend AD)
├── data/
│   ├── __init__.py
│   ├── discretize.py              # Covariate discretization (bam)
│   ├── chunk.py                   # Chunk-based processing
│   └── transforms.py              # Variable transformations, centering
├── distributed/
│   ├── __init__.py
│   ├── sharding.py                # Mesh creation + data sharding helpers
│   ├── chunked_jax_provider.py    # Out-of-core JAX chunked provider
│   ├── ray_launcher.py            # Ray JaxTrainer bootstrap
│   └── streaming.py               # Streaming/online GAM updates
├── predict/
│   ├── __init__.py
│   ├── predict.py                 # predict.gam equivalent
│   ├── lpmatrix.py                # Linear predictor matrix construction
│   └── posterior.py               # Posterior simulation, Bayesian CIs
├── summary/
│   ├── __init__.py
│   ├── summary.py                 # summary.gam equivalent
│   ├── anova.py                   # anova.gam equivalent
│   ├── diagnostics.py             # gam.check, influence
│   ├── concurvity.py              # Concurvity detection and reporting
│   ├── model_comparison.py        # AIC, BIC, model comparison infrastructure
│   └── information_criteria.py    # AIC, BIC, GCV score
├── plot/
│   ├── __init__.py
│   └── plot_gam.py                # plot.gam equivalent using matplotlib
├── compat/
│   ├── __init__.py
│   ├── r_bridge.py                # rpy2-based bridge for testing against R
│   └── export.py                  # Export to Stan, NumPyro (jagam equivalent)
└── tests/
    ├── __init__.py
    ├── conftest.py                # Shared fixtures, R bridge setup
    ├── reference_data/            # Pre-computed R results (JSON/pickle)
    ├── test_smooths/              # Per-smooth-class tests
    ├── test_families/             # Per-family tests
    ├── test_fitting/              # Fitting algorithm tests
    ├── test_api/                  # End-to-end API tests
    ├── test_sparse/               # Sparse correctness tests
    ├── test_gpu/                  # GPU parity tests
    └── benchmarks/                # Performance benchmarks
```

### 3.1 Dependency Stack and Package Management

**uv is the project package manager (v1.16).** All dependency resolution, lockfile generation, environment creation, and CI reproducibility use `uv`. This is not a soft recommendation — `uv.lock` is the single source of truth for the dependency graph, and the multi-host distributed story (Section 16) depends on it for version consistency.

```toml
# pyproject.toml

[project]
name = "pymgcv"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.11",
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "formulaic>=1.0",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
sparse = ["scikit-sparse>=0.4.8"]
gpu = ["jax[cuda12]>=0.4.20"]
distributed = ["ray[default]>=2.9"]
full = ["pymgcv[sparse,gpu,distributed]"]
dev = [
    "pymgcv[full]",
    "pytest>=8.0",
    "rpy2>=3.5",           # R bridge for correctness tests
    "hypothesis>=6.0",     # Property-based testing
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
# Platform-specific resolution: uv.lock captures the exact wheel
# for scikit-sparse on each target, eliminating ABI mismatch at runtime.
environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "sys_platform == 'darwin' and platform_machine == 'arm64'",
    "sys_platform == 'darwin' and platform_machine == 'x86_64'",
]

[[tool.uv.index]]
name = "pymgcv-wheels"
url = "https://pymgcv.github.io/wheels/"
explicit = true

[tool.uv.sources]
# Pre-built scikit-sparse wheels with statically-linked SuiteSparse.
# Built in CI (GitHub Actions), hosted on GitHub Pages.
# Users never need a C compiler.
scikit-sparse = { index = "pymgcv-wheels" }
```

**Install paths:**

```bash
# Basic (dense-only, no C dependencies):
uv sync

# With sparse solver (pre-built CHOLMOD wheel from pymgcv-wheels index):
uv sync --extra sparse

# With GPU:
uv sync --extra gpu

# Full (sparse + GPU + distributed):
uv sync --extra full

# Multi-host cluster: ship uv.lock, run on every node:
uv sync --extra full --frozen  # --frozen = use exact lockfile, no resolution
```

The `--frozen` flag is critical for distributed: it guarantees every host installs exactly the same versions, byte-for-byte. This replaces the custom `_collect_version_pins()` machinery in `SetupManifest` (Section 16.8) — if all hosts ran `uv sync --frozen` from the same `uv.lock`, version divergence is impossible by construction.

**Dependency table:**

| Layer | Primary | Fallback | Install extra | Purpose |
|---|---|---|---|---|
| Array computation | JAX | NumPy (reference only) | (core) | Array ops, compilation |
| Automatic differentiation | JAX (grad, jacfwd, custom_jvp for Tweedie only) | — | (core) | Derivatives of REML, extended family log-lik |
| Sparse matrices | scipy.sparse (CSC/CSR) | — | (core) | Penalty and basis matrices (CPU path only) |
| Sparse solvers | SuiteSparse CHOLMOD/SPQR via scikit-sparse | scipy.sparse.linalg | `sparse` | Sparse Cholesky, sparse QR |
| Dense linear algebra | JAX XLA (GPU path) / scipy.linalg (CPU path) | LAPACK via scipy | (core) | QR, Cholesky, eigendecomp |
| GPU compilation | JAX XLA (CUDA, Metal, ROCm) | — | `gpu` | Hardware acceleration |
| Formula parsing | formulaic (parametric) + AST-based (smooth terms) | — | (core) | R-style formula interface |
| Visualization | matplotlib | — | (core) | Plotting |
| Distributed | Ray (optional) | — | `distributed` | Multi-node cluster |
| R bridge (test only) | rpy2 | subprocess + Rscript | `dev` | Reference comparison |

**Pre-built scikit-sparse wheel infrastructure:**

The `pymgcv-wheels` index hosts scikit-sparse wheels with SuiteSparse 7.x statically linked against OpenBLAS. These are built in GitHub Actions CI:

| Platform | Wheel | Status |
|---|---|---|
| Linux x86_64 (manylinux2014) | ✅ | Primary target |
| macOS arm64 (Apple Silicon) | ✅ | Via cross-compilation |
| macOS x86_64 | ✅ | Legacy Intel Macs |
| Windows x86_64 | ❌ | Not built (MSVC ABI issues). Windows users: use WSL2 or conda. |

When `uv sync --extra sparse` runs, uv checks `pymgcv-wheels` first (per the `explicit = true` + `[tool.uv.sources]` config), finds the pre-built wheel, and installs it. No C compiler needed. If the platform doesn't have a pre-built wheel, uv falls back to PyPI's scikit-sparse (which may require compilation), and if that fails, the install fails at install time — not at runtime.

**Docker image:** `ghcr.io/pymgcv/pymgcv:latest` runs `uv sync --extra full --frozen` from the repo's `uv.lock`. This is the recommended deployment target for production and multi-host clusters.

---

## 4. Core Computational Backend

### 4.1 JAX-First Design Principle

**There is no multi-backend abstraction layer.** v1.0 proposed a `ArrayBackend` Protocol unifying JAX, NumPy, and PyTorch. This was removed because:

1. JAX traces computation graphs — Python `for`/`if`/`break` inside `jax.jit` captures only one execution path, silently producing wrong results. The PIRLS step-halving loop and convergence checks require `jax.lax.while_loop` and `jax.lax.cond`, which have fundamentally different signatures from NumPy equivalents.
2. Every iterative algorithm would need two genuinely different implementations hidden behind a "unified" interface, defeating the purpose of abstraction.
3. PyTorch and PyTensor add nothing that JAX doesn't provide for this use case, but triple the test surface.

Instead, the codebase has two distinct implementations:

| | **JAX path** (primary) | **NumPy path** (reference/fallback) |
|---|---|---|
| **Purpose** | Production use, performance, GPU | Testing, environments without JAX |
| **AD support** | Full (grad, hessian; custom_jvp for Tweedie only) | None — analytical derivatives only |
| **JIT** | Yes (XLA compilation) | No |
| **GPU** | Yes (CUDA, Metal, ROCm) | No |
| **Sparse** | No (dense only on GPU) | Yes (scipy.sparse + CHOLMOD) |
| **Iterative loops** | `jax.lax.while_loop` / `jax.lax.scan` | Python `for` / `while` |
| **Maintained as** | Primary, fully optimized | Minimal, correctness-only |

```python
# linalg/backend.py

"""
JAX-first backend with NumPy reference fallback.

Usage:
    import pymgcv
    pymgcv.configure(backend="jax", device="gpu")  # Production
    pymgcv.configure(backend="numpy")               # Fallback / testing

All performance-critical code has two implementations:
    fitting/_pirls_jax.py     — JAX path (jax.lax loops, JIT'd)
    fitting/_pirls_numpy.py   — NumPy path (Python loops, scipy solvers)

The top-level fitting functions dispatch based on the configured backend.
"""

import enum

class Backend(enum.Enum):
    JAX = "jax"
    NUMPY = "numpy"

class DeviceConfig:
    """Global configuration singleton."""
    def __init__(self):
        self.backend: Backend = Backend.JAX
        self.device: str = "cpu"  # "cpu" or "gpu"
        self._jax_initialized: bool = False

    def configure(self, backend: str = "jax", device: str = "cpu"):
        self.backend = Backend(backend)
        self.device = device
        if self.backend == Backend.JAX:
            self._init_jax()

    def _init_jax(self):
        import jax
        if self.device == "gpu":
            gpu_devices = jax.devices("gpu")
            if not gpu_devices:
                raise RuntimeError(
                    "No GPU found. Install jax[cuda12] (NVIDIA), "
                    "jax-metal (Apple), or jax[rocm] (AMD)."
                )
            # Set default device
            jax.config.update("jax_default_device", gpu_devices[0])
        # Enable 64-bit floats (critical for numerical stability)
        jax.config.update("jax_enable_x64", True)
        self._jax_initialized = True

    @property
    def use_jax(self) -> bool:
        return self.backend == Backend.JAX

_config = DeviceConfig()

def configure(backend: str = "jax", device: str = "cpu"):
    """Configure the global default backend. Call once at startup."""
    _config.configure(backend, device)

def get_config() -> DeviceConfig:
    return _config


class FitConfig:
    """
    Per-model configuration context manager.

    v1.6: The global configure() is a concurrency footgun — multi-model
    fits in parallel threads, or libraries embedding pymgcv, will get
    heisenbugs. FitConfig provides per-call overrides:

        with pymgcv.FitConfig(device="gpu", execution_path="dense_gpu"):
            model1 = pymgcv.gam("y ~ s(x1)", data=df1)

        with pymgcv.FitConfig(device="cpu", execution_path="sparse_cpu"):
            model2 = pymgcv.gam("y ~ s(x1) + s(x2, bs='mrf')", data=df2)

    When used, FitConfig overrides the global _config for all operations
    within the context block. Thread-local storage ensures no cross-thread
    contamination.
    """
    _thread_local = threading.local()

    def __init__(self, backend=None, device=None, execution_path=None,
                 seed=None, deterministic=False):
        self.backend = Backend(backend) if backend else None
        self.device = device
        self.execution_path = execution_path
        self.seed = seed
        self.deterministic = deterministic

    def __enter__(self):
        self._prev = getattr(FitConfig._thread_local, 'config', None)
        FitConfig._thread_local.config = self
        return self

    def __exit__(self, *args):
        FitConfig._thread_local.config = self._prev

    @staticmethod
    def active():
        """Return active FitConfig (thread-local) or None."""
        return getattr(FitConfig._thread_local, 'config', None)


import threading
```

### 4.2 JIT Compilation Strategy

All performance-critical JAX functions use `jax.lax` control flow primitives to remain JIT-compatible. **Python-level control flow (`for`, `if`, `break`) is never used inside JIT-compiled functions.**

```python
# fitting/_pirls_jax.py — JAX PIRLS inner step (JIT-safe)

import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(5,))
def pirls_step_jax(X, y, beta, S_lambda, family_params, family_type):
    """
    One PIRLS iteration, fully JIT-compiled.
    No Python control flow — all branching via jax.lax.
    """
    eta = X @ beta
    mu = _link_inverse(eta, family_type)
    W = _working_weights(mu, family_params, family_type)
    z = _working_response(y, mu, eta, family_params, family_type)

    # Clamp weights (JIT-safe: no Python if)
    W = jnp.clip(W, 1e-10, 1e10)
    W_sqrt = jnp.sqrt(W)

    # Normal equations: (X^T W X + S_λ) β = X^T W z
    WX = W_sqrt[:, None] * X
    XtWX = WX.T @ WX + S_lambda
    XtWz = WX.T @ (W_sqrt * z)
    beta_new = jnp.linalg.solve(XtWX, XtWz)

    dev = _deviance(y, mu, family_params, family_type)
    return beta_new, mu, eta, dev, W, XtWX


def pirls_loop_jax(X, y, family_params, family_type, S_lambda,
                   beta_init, max_iter=100, tol=1e-7):
    """
    Full PIRLS loop using jax.lax.while_loop (JIT-compatible).
    Step halving uses jax.lax.fori_loop + jax.lax.cond.
    """
    def cond_fn(state):
        i, _, _, _, dev, dev_old, converged = state
        still_iterating = jnp.logical_and(i < max_iter, ~converged)
        return still_iterating

    def body_fn(state):
        i, beta, mu, eta, dev_old, _, _ = state

        beta_new, mu_new, eta_new, dev_new, W, XtWX = pirls_step_jax(
            X, y, beta, S_lambda, family_params, family_type
        )

        # Step halving via jax.lax.while_loop
        step = beta_new - beta
        beta_accepted, mu_accepted, dev_accepted = _step_halving_jax(
            X, y, beta, step, family_params, family_type, dev_old
        )

        # Convergence: penalized deviance change + coefficient change
        pen_dev = dev_accepted + beta_accepted @ S_lambda @ beta_accepted
        pen_dev_old = dev_old + beta @ S_lambda @ beta
        dev_change = jnp.abs(pen_dev - pen_dev_old) / (0.1 + jnp.abs(pen_dev))
        coef_change = jnp.max(jnp.abs(beta_accepted - beta)) / (
            0.1 + jnp.max(jnp.abs(beta_accepted))
        )
        converged = jnp.logical_and(dev_change < tol, coef_change < tol)
        # Allow deviance increase for first 3 iterations
        converged = jnp.where(i < 3, False, converged)

        return (i + 1, beta_accepted, mu_accepted,
                eta_new, dev_accepted, dev_old, converged)

    # Use large finite sentinel for first iteration comparison.
    # i==0 check in body_fn ensures first step is accepted regardless.
    init_state = (0, beta_init, jnp.zeros_like(y), jnp.zeros_like(y),
                  1e30, 1e30, False)
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_state
```

**Compilation targets:**

| Target | Method | When Used |
|---|---|---|
| CPU (x86/ARM) | JAX XLA | Default |
| CUDA GPU | JAX XLA CUDA | `configure(device="gpu")` on NVIDIA |
| Apple Metal | jax-metal plugin | `configure(device="gpu")` on Apple Silicon |
| ROCm GPU | JAX ROCm | `configure(device="gpu")` on AMD |
| CPU (no JAX) | Python + scipy + Cython | `configure(backend="numpy")` |

### 4.3 Cython Fallback Kernels

For the NumPy reference backend, performance-critical inner loops are implemented in Cython:

```
pymgcv/
├── _cython/
│   ├── _pirls_core.pyx        # PIRLS inner loop
│   ├── _basis_eval.pyx        # Basis function evaluation (TPRS, B-splines)
│   ├── _sparse_ops.pyx        # Sparse penalty operations
│   └── _discretize.pyx        # Covariate discretization
```

These are used only by the NumPy backend. The JAX backend relies entirely on XLA compilation.

### 4.4 JAX Purity Boundary (CI-Enforced)

**Hard rule: JAX-path modules import zero SciPy and zero NumPy at runtime.**

All modules in the JAX execution path (`fitting/_pirls_jax.py`, `autodiff/`, `families/*_jax.py`) must use only `jax`, `jax.numpy`, `jax.scipy`, and `jax.lax`. This is enforced by a CI lint guard:

```python
# ci/check_jax_purity.py — runs in CI on every PR

import ast, sys, pathlib

FORBIDDEN_IN_JAX = {"scipy", "numpy", "np"}
JAX_PATH_GLOBS = [
    "pymgcv/fitting/*_jax.py",
    "pymgcv/autodiff/*.py",
    "pymgcv/families/*_jax.py",
    "pymgcv/linalg/*_jax.py",
]

def check_file(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text())
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IN_JAX:
                    violations.append(
                        f"{path}:{node.lineno} imports '{alias.name}' — "
                        f"JAX-path modules must not import {root}"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_IN_JAX:
                violations.append(
                    f"{path}:{node.lineno} imports from '{node.module}' — "
                    f"JAX-path modules must not import {root}"
                )
    return violations
```

**Practical implications:**

| Need | SciPy way (forbidden in JAX path) | JAX way (required) |
|---|---|---|
| `gammaln` | `scipy.special.gammaln` | `jax.scipy.special.gammaln` |
| `digamma` | `scipy.special.digamma` | `jax.scipy.special.digamma` |
| Normal CDF | `scipy.stats.norm.cdf` | `jax.scipy.stats.norm.cdf` |
| Sparse matrices | `scipy.sparse` | Dense `jax.Array` (GPU path) |
| Control flow | `for`/`while`/`if`/`break` | `jax.lax.fori_loop`/`while_loop`/`cond` |
| Random state | `numpy.random` | `jax.random` with explicit PRNG key |

**Two-Phase Architecture (v1.4):**

The JAX purity boundary is not just an import rule — it reflects a two-phase execution model that must be understood to avoid "setup leaked into JIT" bugs:

**Phase 1: Setup (CPU, NumPy/SciPy allowed).** All of the following run once, on CPU, using NumPy/SciPy freely:

| Module | What it does | Output type |
|---|---|---|
| `formula/parser.py` | Parse formula, extract smooth terms | `ParsedFormula` (Python objects) |
| `smooths/*.py` `.setup()` | Select knots, compute eigendecompositions | NumPy arrays stored in Smooth |
| `smooths/*.py` `.build_design_matrix()` | Evaluate basis functions at data | NumPy array or scipy.sparse |
| `smooths/*.py` `.build_penalty_matrices()` | Construct penalty matrices | NumPy/scipy.sparse |
| `fitting/constraints.py` | `apply_joint_identifiability()` | `CoefficientMap` (Python + NumPy) |
| `penalties/structured.py` | Wrap penalties as `StructuredPenalty` | Python objects with JAX arrays |
| `linalg/execution_path.py` | Choose execution path | Enum |

**Boundary transfer:** At the end of Phase 1, the following are converted to `jax.Array` and transferred to device:

```python
# This is the explicit boundary. It runs once per model fit.
# After this point, ONLY Phase 2 code touches these arrays.

X_jax = jax.device_put(X_numpy)           # Design matrix
y_jax = jax.device_put(y_numpy)           # Response
wt_jax = jax.device_put(weights_numpy)    # Prior weights
# Penalties are already StructuredPenalty objects holding jax.Array
# S_lambda_dense is already jax.Array (for REML)
```

**Phase 2: Fit (JAX-only, JIT-able).** All iterative computation — PIRLS, step-halving, REML evaluation, EDF computation — runs in Phase 2. This code:

- Imports only `jax`, `jax.numpy`, `jax.scipy`, `jax.lax`
- Receives only `jax.Array` and `StructuredPenalty` objects
- Uses only `jax.lax.while_loop`/`cond`/`fori_loop` for control flow
- Can be fully JIT-compiled into a single XLA program

**What CANNOT cross the boundary into Phase 2:**

- `SmoothSpec` objects (contain Python strings, lists)
- `scipy.sparse` matrices (not JAX-traceable)
- `CoefficientMap` (contains Python tuples, used only for post-estimation)
- Any `dict`, `list`, or variable-length Python object

**Post-estimation (Phase 1 again):** After Phase 2 converges, results are transferred back to CPU. `predict()`, `summary()`, `plot()` run in Phase 1 using `CoefficientMap` and NumPy.

### 4.5 Determinism and RNG Policy

**v1.5:** Without an explicit determinism contract, randomized knot selection, GPU non-associativity, and distributed reduction ordering all produce flapping tests and non-reproducible results.

**Global seed:**

```python
import pymgcv

pymgcv.set_seed(42)  # Seeds BOTH np.random and jax.random

# Internally:
# _rng = np.random.default_rng(seed)
# _jax_key = jax.random.PRNGKey(seed)
# All setup-phase randomness draws from _rng.
# All JIT-phase randomness uses _jax_key splits.
```

**Where randomness enters:**

| Operation | Phase | RNG source | Determinism guarantee |
|---|---|---|---|
| Knot selection (max-min subsample) | Setup | `_rng` (NumPy) | Deterministic given seed + data order |
| Hutchinson trace probes | Fit | `_rng` (NumPy) or `_jax_key` (JAX) | Deterministic given seed |
| Stochastic gradient (fREML) | Fit | `_jax_key` | Deterministic given seed |
| Distributed reduction order | Fit | Sorted worker keys | Deterministic regardless of arrival order |
| GPU floating-point | Fit | N/A (non-associativity) | NOT bit-for-bit. Tolerance: ±1e-10 vs CPU. |

**Test modes:**

- `pymgcv.set_deterministic(True)`: forces CPU-ordered reductions, enables `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, uses sorted key reduction in distributed mode. Slower, but reproducible within STRICT tolerance (1e-10) **on the same hardware, OS, JAX version, and CUDA driver**. NOT guaranteed across toolchain updates — floating-point codegen can change between JAX/XLA releases. CI determinism tests pin specific versions.
- Default: allows XLA reordering for speed. Results are correct within MODERATE tolerance (1e-6) across runs on same hardware, but not reproducible at STRICT level.

---

## 5. Smooth Function Specifications

### 5.1 Base Smooth Class

```python
# smooths/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
from scipy import sparse

@dataclass
class SmoothSpec:
    """Specification for a smooth term parsed from a formula."""
    term_label: str           # e.g., "s(x1)", "te(x1,x2)"
    variables: list[str]      # Covariate names
    by_variable: Optional[str] = None  # Factor-by variable
    bs: str = "tp"            # Basis type
    k: int = -1               # Basis dimension (-1 = default)
    m: Optional[list[int]] = None  # Penalty order
    sp: Optional[list[float]] = None  # Fixed smoothing parameters
    fx: bool = False          # Fixed df (unpenalized)?
    id: Optional[str] = None  # Shared smoothing parameter ID
    xt: Optional[dict] = None # Extra arguments (e.g., MRF penalty matrix)
    pc: Optional[np.ndarray] = None  # Point constraint
    knots: Optional[dict] = None  # User-supplied knots


class Smooth(ABC):
    """Abstract base class for all smooth term implementations."""

    def __init__(self, spec: SmoothSpec):
        self.spec = spec
        self.n_coefs: int = 0          # Number of basis functions
        self.n_penalties: int = 0      # Number of penalty matrices
        self.null_space_dim: int = 0   # Dimension of null space
        self.penalty_matrices: list[sparse.spmatrix] = []
        self.constraint_matrix: Optional[np.ndarray] = None  # C for Cβ=0

    @abstractmethod
    def setup(self, data: dict[str, np.ndarray], knots: Optional[dict] = None):
        """Compute knots, eigendecompositions, and any precomputation.
        Called once before fitting."""
        ...

    @abstractmethod
    def build_design_matrix(self, data: dict[str, np.ndarray]) -> np.ndarray | sparse.spmatrix:
        """Evaluate basis functions at data points. Returns (n, n_coefs) matrix."""
        ...

    @abstractmethod
    def build_penalty_matrices(self) -> list[sparse.spmatrix]:
        """Return penalty matrix/matrices S_j. Each is (n_coefs, n_coefs).
        Stored as sparse CSC matrices."""
        ...

    def apply_identifiability_constraint(self, X, C=None):
        """Apply sum-to-zero or other identifiability constraints.
        Default: absorb constraint via QR reparameterization."""
        if C is None:
            C = np.ones((1, self.n_coefs))  # Sum-to-zero
        Q, _ = np.linalg.qr(C.T, mode='complete')
        Z = Q[:, C.shape[0]:]  # Null space of C
        X_constrained = X @ Z
        S_constrained = [Z.T @ S @ Z for S in self.penalty_matrices]
        return X_constrained, S_constrained, Z

    def predict_matrix(self, new_data: dict[str, np.ndarray]) -> np.ndarray | sparse.spmatrix:
        """Build design matrix for prediction (may differ from fitting matrix)."""
        return self.build_design_matrix(new_data)
```

### 5.2 Thin Plate Regression Splines (TPRS) — `tp`, `ts`

This is the default and most complex smooth class. Implementation follows Wood (2003).

```python
# smooths/tprs.py

class ThinPlateSmooth(Smooth):
    """
    Thin plate regression splines following Wood (2003).

    Algorithm:
    1. Compute full TPS basis: E (n×n) matrix from pairwise distances
       η_{md}(r) for d dimensions, m penalty order
    2. Compute T: polynomial null space basis (n × M)
    3. Eigen-decompose E = U D U^T
    4. Truncate to k basis functions using largest eigenvalues
    5. Reparameterize to orthogonalize penalty from null space

    Penalty order m default: For d covariates, default m = floor(d/2) + 1
    which ensures the penalty is on the smallest integer-order derivative
    that yields a non-degenerate penalty.

    Null space dimension M = choose(m + d - 1, d)
    """

    def setup(self, data: dict[str, np.ndarray], knots=None):
        d = len(self.spec.variables)
        m = self.spec.m[0] if self.spec.m else max(d // 2 + 1, 2) if d > 1 else 2

        # Null space dimension
        from math import comb
        M = comb(m + d - 1, d)

        # 1. Select knot locations (subsample if n > max_knots)
        X_covariates = np.column_stack(
            [data[v] for v in self.spec.variables]
        )
        n = X_covariates.shape[0]

        # Default k, with hard ceiling to prevent O(k³) blowup.
        # For k > 2000, users should use bam() with discretization
        # or a different smooth class (gp, ps).
        MAX_K = 2000
        if self.spec.k > 0:
            k = self.spec.k
            if k > MAX_K:
                raise ValueError(
                    f"TPRS basis dimension k={k} exceeds maximum {MAX_K}. "
                    f"TPRS requires O(k³) eigendecomposition. Use bs='gp' or "
                    f"bs='ps' for large basis dimensions, or use bam()."
                )
        else:
            k = min(max(10 * d, 50), n, MAX_K)

        if n > k:
            # Use space-filling subsample for knots
            self._knots = self._select_knots(X_covariates, k)
        else:
            self._knots = X_covariates.copy()

        nk = self._knots.shape[0]

        # 2. Compute E matrix (nk × nk) from eta_md distances between knots
        E = self._compute_tps_matrix(self._knots, self._knots, m, d)

        # 3. Compute T matrix: polynomial terms at knots
        T = self._compute_polynomial_basis(self._knots, m, d)  # (nk × M)

        # 4. Eigendecompose E
        eigenvalues, eigenvectors = np.linalg.eigh(E)

        # 5. Truncate: keep k - M largest eigenvalues
        n_basis = k - M
        idx = np.argsort(np.abs(eigenvalues))[::-1][:n_basis]
        U_k = eigenvectors[:, idx]
        D_k = eigenvalues[idx]

        # Validate: TPS eigenvalues should be positive for the
        # selected eigenvectors. Small negative values can arise
        # from numerical error; large negative values indicate a bug.
        min_eig = np.min(D_k)
        if min_eig < -1e-6 * np.max(np.abs(D_k)):
            raise ValueError(
                f"TPRS eigendecomposition produced large negative eigenvalue "
                f"({min_eig:.4e}). This indicates a degenerate distance "
                f"matrix — check for duplicate or near-duplicate knots."
            )
        # Floor small negatives to a small positive value
        D_k = np.maximum(D_k, 1e-12 * np.max(D_k))

        # Store for basis evaluation
        self._U_k = U_k
        self._D_k = D_k
        self._T = T
        self._M = M
        self._m = m
        self._d = d
        self.n_coefs = k
        self.null_space_dim = M

    def build_design_matrix(self, data):
        X_covariates = np.column_stack([data[v] for v in self.spec.variables])
        # E matrix: data points to knots
        E_xk = self._compute_tps_matrix(X_covariates, self._knots, self._m, self._d)
        # Basis: [(E_xk @ U_k) * D_k^{-1/2} | T_x]
        # D_k is guaranteed positive after validation in setup()
        # v1.6: Column scaling instead of np.diag(D_k**-0.5) which
        # allocates a dense (k-M)×(k-M) matrix unnecessarily.
        X_smooth = (E_xk @ self._U_k) * (self._D_k ** -0.5)[None, :]
        T_x = self._compute_polynomial_basis(X_covariates, self._m, self._d)
        return np.column_stack([X_smooth, T_x])

    def build_penalty_matrices(self):
        # Penalty is diagonal on the truncated eigenbasis, zero on null space
        n_penalized = self.n_coefs - self._M
        diag_vals = np.ones(n_penalized)
        S = sparse.block_diag([
            sparse.diags(diag_vals),
            sparse.csc_matrix((self._M, self._M))
        ], format='csc')
        self.penalty_matrices = [S]
        self.n_penalties = 1
        return self.penalty_matrices

    @staticmethod
    def _compute_tps_matrix(X1, X2, m, d):
        """Compute η_{md}(||x1_i - x2_j||) matrix."""
        from scipy.spatial.distance import cdist
        r = cdist(X1, X2)
        # η_{md}(r) depends on whether 2m - d is even or odd
        if (2 * m - d) % 2 == 0:
            # η(r) = c * r^{2m-d} log(r) (with 0*log(0)=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                E = np.where(r > 0, r ** (2*m - d) * np.log(r), 0.0)
        else:
            E = r ** (2*m - d)
        return E

    @staticmethod
    def _compute_polynomial_basis(X, m, d):
        """Compute polynomial null space basis up to degree m-1."""
        # Generate all monomials of degree <= m-1 in d variables
        from itertools import combinations_with_replacement
        terms = []
        for deg in range(m):
            for combo in combinations_with_replacement(range(d), deg):
                col = np.ones(X.shape[0])
                for dim in combo:
                    col *= X[:, dim]
                terms.append(col)
        return np.column_stack(terms) if terms else np.ones((X.shape[0], 1))

    @staticmethod
    def _select_knots(X, k):
        """Space-filling subsample of k points from X.
        Uses the max-min distance algorithm (same as mgcv)."""
        n = X.shape[0]
        selected = [np.random.randint(n)]
        min_dists = np.full(n, np.inf)
        for _ in range(k - 1):
            dists = np.sum((X - X[selected[-1]]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, dists)
            selected.append(np.argmax(min_dists))
        return X[np.array(selected)]


class ThinPlateShrinkageSmooth(ThinPlateSmooth):
    """ts: Thin plate with extra shrinkage penalty on null space.
    Adds a second penalty matrix targeting the null space so that
    smoothing can shrink the term entirely to zero."""

    def build_penalty_matrices(self):
        super().build_penalty_matrices()
        # Add null space penalty
        n_penalized = self.n_coefs - self._M
        S_null = sparse.block_diag([
            sparse.csc_matrix((n_penalized, n_penalized)),
            sparse.eye(self._M, format='csc')
        ], format='csc')
        self.penalty_matrices.append(S_null)
        self.n_penalties = 2
        return self.penalty_matrices
```

### 5.3 Cubic Regression Splines — `cr`, `cs`, `cc`

```python
# smooths/cubic.py

class CubicRegressionSmooth(Smooth):
    """
    Cubic regression spline (cr).

    Uses natural cubic spline basis with equally spaced or
    quantile-based knots. The penalty is the integrated
    squared second derivative.

    Knot placement: Quantiles of the covariate by default.
    Basis: Standard B-spline basis of order 4, reparameterized.
    Penalty: ∫ f''(x)² dx = β^T S β where S is the second
    derivative inner product matrix.
    """

    def setup(self, data, knots=None):
        x = data[self.spec.variables[0]]
        k = self.spec.k if self.spec.k > 0 else 10

        # Place knots at quantiles
        probs = np.linspace(0, 1, k)
        self._knots_interior = np.quantile(x, probs)
        self.n_coefs = k
        self.null_space_dim = 2  # linear functions

        # Precompute penalty matrix (second derivative inner product)
        h = np.diff(self._knots_interior)
        # Tridiagonal band matrix for natural cubic spline penalty
        S = self._build_second_deriv_penalty(h, k)
        self.penalty_matrices = [sparse.csc_matrix(S)]
        self.n_penalties = 1

    def build_design_matrix(self, data):
        x = data[self.spec.variables[0]]
        return self._natural_cubic_basis(x, self._knots_interior)

    def build_penalty_matrices(self):
        return self.penalty_matrices

    @staticmethod
    def _natural_cubic_basis(x, knots):
        """
        Evaluate natural cubic spline basis at x given knots.

        ⚠️ REFERENCE IMPLEMENTATION ONLY — O(n*k) Python loops.
        Production path MUST use one of:
        - scipy.interpolate.BSpline (vectorized C, ~100x faster)
        - JAX vmap over knot intervals (GPU-compatible)
        - Cython inner loop (for NumPy fallback path)
        """
        # Using the "value + slope" parameterization
        k = len(knots)
        n = len(x)
        X = np.zeros((n, k))
        h = np.diff(knots)
        for i in range(n):
            for j in range(k):
                # Natural cubic spline basis functions (cardinal)
                X[i, j] = _cubic_basis_function(x[i], j, knots, h)
        return X

    @staticmethod
    def _build_second_deriv_penalty(h, k):
        """Build the integrated squared second derivative penalty matrix."""
        # Standard tridiagonal second-derivative penalty for cubic splines
        B = np.zeros((k - 2, k))
        for i in range(k - 2):
            B[i, i] = 1.0 / h[i]
            B[i, i+1] = -(1.0/h[i] + 1.0/h[i+1])
            B[i, i+2] = 1.0 / h[i+1]
        R = np.zeros((k-2, k-2))
        for i in range(k-2):
            R[i, i] = (h[i] + h[i+1]) / 3.0
            if i < k - 3:
                R[i, i+1] = h[i+1] / 6.0
                R[i+1, i] = h[i+1] / 6.0
        # S = B^T R^{-1} B (but compute via Cholesky for stability)
        L = np.linalg.cholesky(R)
        BL = np.linalg.solve(L, B)
        S = BL.T @ BL
        return S


class CyclicCubicSmooth(CubicRegressionSmooth):
    """cc: Cyclic cubic regression spline with wrap-around constraint."""

    def setup(self, data, knots=None):
        super().setup(data, knots)
        # Enforce f(lower) = f(upper) by removing last basis function
        # and wrapping the penalty
        self.n_coefs -= 1
        self.null_space_dim = 1  # constant functions only
```

### 5.4 P-Splines — `ps`, `cp`

```python
# smooths/pspline.py

class PSplineSmooth(Smooth):
    """
    P-splines (ps): B-spline basis with discrete difference penalty.

    Basis: B-spline of order m[1]+1 (default cubic, m[1]=2 → order 3 → degree 3)
    Penalty: Δ^{m[0]} β = D_{m[0]}^T D_{m[0]} (discrete difference penalty)
    Default m = [2, 2] → second-order difference penalty on cubic B-splines.

    Knots are equally spaced covering the data range with
    appropriate boundary extension.
    """

    def setup(self, data, knots=None):
        x = data[self.spec.variables[0]]
        k = self.spec.k if self.spec.k > 0 else 20
        m = self.spec.m if self.spec.m else [2, 2]
        self._penalty_order = m[0]
        self._spline_order = m[1] + 1  # B-spline order

        # Equally spaced knots with boundary padding
        x_min, x_max = x.min(), x.max()
        n_interior = k - self._spline_order
        knot_spacing = (x_max - x_min) / (n_interior - 1)
        pad = self._spline_order * knot_spacing
        self._knots_full = np.linspace(
            x_min - pad, x_max + pad,
            n_interior + 2 * self._spline_order
        )
        self.n_coefs = k
        self.null_space_dim = self._penalty_order  # polynomials up to degree m-1

    def build_design_matrix(self, data):
        """
        ⚠️ REFERENCE IMPLEMENTATION — evaluates each basis function
        individually via Python loop. O(n*k) with Python overhead.

        Production path: single vectorized BSpline call:
            tck = (self._knots_full, np.eye(self.n_coefs), self._spline_order - 1)
            X = scipy.interpolate.BSpline.design_matrix(
                x, self._knots_full, self._spline_order - 1
            ).toarray()
        This is ~50x faster and returns a sparse matrix.
        """
        from scipy.interpolate import BSpline
        x = data[self.spec.variables[0]]
        c_eye = np.eye(self.n_coefs)
        X = np.column_stack([
            BSpline(self._knots_full, c_eye[i], self._spline_order - 1)(x)
            for i in range(self.n_coefs)
        ])
        return X

    def build_penalty_matrices(self):
        # Difference matrix D of order m
        D = np.eye(self.n_coefs)
        for _ in range(self._penalty_order):
            D = np.diff(D, axis=0)
        S = sparse.csc_matrix(D.T @ D)
        self.penalty_matrices = [S]
        self.n_penalties = 1
        return self.penalty_matrices
```

### 5.5 Tensor Product Smooths — `te`, `ti`, `t2`

```python
# smooths/tensor.py

class TensorProductSmooth(Smooth):
    """
    te(): Tensor product smooth.

    Given marginal smooths s_1(x_1), ..., s_d(x_d) with bases
    B_1, ..., B_d and penalties S_1, ..., S_d:

    - Basis: B = B_1 ⊗ B_2 ⊗ ... ⊗ B_d (row-wise Kronecker product)
    - Penalties: S_j = I ⊗ ... ⊗ S_j ⊗ ... ⊗ I (one per marginal penalty)

    For te(), each marginal penalty produces one smoothing parameter.
    For ti(), the basis is constructed to be identifiable alongside
    lower-order terms (ANOVA decomposition).
    For t2(), an alternative parameterization after Wood, Scheipl & Faraway (2013).
    """

    def __init__(self, spec: SmoothSpec, marginal_smooths: list[Smooth]):
        super().__init__(spec)
        self.marginals = marginal_smooths

    def setup(self, data, knots=None):
        for s in self.marginals:
            s.setup(data, knots)
        self.n_coefs = 1
        for s in self.marginals:
            self.n_coefs *= s.n_coefs
        self.null_space_dim = 1
        for s in self.marginals:
            self.null_space_dim *= s.null_space_dim

    def build_design_matrix(self, data):
        """Row-wise Kronecker product of marginal basis matrices."""
        matrices = [s.build_design_matrix(data) for s in self.marginals]
        X = matrices[0]
        for M in matrices[1:]:
            X = self._row_tensor(X, M)
        return X

    def build_penalty_matrices(self):
        """Kronecker sum penalties: I ⊗ S_j and S_j ⊗ I."""
        penalties = []
        for j, s in enumerate(self.marginals):
            s_penalties = s.build_penalty_matrices()
            for S_j in s_penalties:
                # Build I ⊗ ... ⊗ S_j ⊗ ... ⊗ I
                P = S_j
                for i, s2 in enumerate(self.marginals):
                    if i != j:
                        if i < j:
                            P = sparse.kron(sparse.eye(s2.n_coefs), P, format='csc')
                        else:
                            P = sparse.kron(P, sparse.eye(s2.n_coefs), format='csc')
                penalties.append(P)
        self.penalty_matrices = penalties
        self.n_penalties = len(penalties)
        return self.penalty_matrices

    @staticmethod
    def _row_tensor(A, B):
        """
        Row-wise Kronecker product: result[i, :] = A[i,:] ⊗ B[i,:].

        v1.6: Avoids the O(n × ka × kb) 3D intermediate from
        (A[:,:,None] * B[:,None,:]).reshape(...). Instead, builds
        one "A-column × all-B-columns" block at a time.
        Peak memory: O(n × kb) per block instead of O(n × ka × kb) total.

        For production/JAX path: use jnp.einsum('ni,nj->nij', A, B)
        which XLA can fuse without materializing the 3D tensor.
        """
        n = A.shape[0]
        ka, kb = A.shape[1], B.shape[1]
        result = np.empty((n, ka * kb))
        for i in range(ka):
            # Broadcast A[:,i] against all columns of B: O(n × kb)
            result[:, i * kb:(i + 1) * kb] = A[:, i:i+1] * B  # (n,1) * (n,kb)
        return result


class TensorInteractionSmooth(TensorProductSmooth):
    """ti(): Tensor product interaction (removes main effects for ANOVA decomposition)."""

    def build_design_matrix(self, data):
        X_full = super().build_design_matrix(data)
        # Remove columns corresponding to marginal null spaces
        # (i.e., keep only interaction components)
        return self._remove_null_space_components(X_full)


class TensorProductType2Smooth(TensorProductSmooth):
    """t2(): Alternative tensor product with single penalty per marginal."""
    # Uses Wood, Scheipl & Faraway (2013) parameterization
    pass
```

### 5.6 Random Effects and Factor-Smooth Interactions

```python
# smooths/random_effects.py

class RandomEffectSmooth(Smooth):
    """
    re: Random effects as penalized smooth terms.

    For a factor variable with L levels:
    - Basis: L×L identity matrix (one column per level)
    - Penalty: I_L (ridge penalty → random intercepts)
    - The smoothing parameter λ estimates σ²/σ²_b
    """

    def setup(self, data, knots=None):
        self._levels = np.unique(data[self.spec.variables[0]])
        self.n_coefs = len(self._levels)
        self.null_space_dim = 0  # Fully penalized
        self._level_map = {lev: i for i, lev in enumerate(self._levels)}

    def build_design_matrix(self, data):
        x = data[self.spec.variables[0]]
        n = len(x)
        rows = np.arange(n)
        cols = np.array([self._level_map[v] for v in x])
        return sparse.csc_matrix(
            (np.ones(n), (rows, cols)),
            shape=(n, self.n_coefs)
        )

    def build_penalty_matrices(self):
        S = sparse.eye(self.n_coefs, format='csc')
        self.penalty_matrices = [S]
        self.n_penalties = 1
        return self.penalty_matrices


class FactorSmoothInteractionSmooth(Smooth):
    """
    fs: Factor-smooth interaction.

    Creates a separate smooth for each level of a factor,
    with a shared smoothing parameter. Implemented as a block-diagonal
    basis with block-diagonal penalty.
    """

    def __init__(self, spec: SmoothSpec, base_smooth_class: type):
        super().__init__(spec)
        self._base_class = base_smooth_class

    def setup(self, data, knots=None):
        factor_var = self.spec.by_variable
        self._levels = np.unique(data[factor_var])
        self._per_level_smooths = {}
        for level in self._levels:
            mask = data[factor_var] == level
            level_data = {k: v[mask] for k, v in data.items()}
            s = self._base_class(self.spec)
            s.setup(level_data, knots)
            self._per_level_smooths[level] = s
        k_per_level = self._per_level_smooths[self._levels[0]].n_coefs
        self.n_coefs = len(self._levels) * k_per_level
        self.null_space_dim = len(self._levels) * \
            self._per_level_smooths[self._levels[0]].null_space_dim

    def build_design_matrix(self, data):
        """
        Block-diagonal design matrix: each level's smooth occupies
        its own column block, with nonzero rows only where the
        factor matches that level.

        Uses sparse assembly (lil_matrix → csc) to avoid the
        invalid X_level[~mask] = 0 pattern, which fails on
        sparse matrices and is wasteful for dense.
        """
        factor_vals = data[self.spec.by_variable]
        n = len(factor_vals)
        k_per = self._per_level_smooths[self._levels[0]].n_coefs
        total_cols = len(self._levels) * k_per

        # Build via lil_matrix for efficient row-by-row insertion
        X = sparse.lil_matrix((n, total_cols))

        for level_idx, level in enumerate(self._levels):
            mask = factor_vals == level
            row_indices = np.where(mask)[0]
            if len(row_indices) == 0:
                continue

            s = self._per_level_smooths[level]
            # Evaluate basis ONLY at matching rows
            level_data = {k: v[mask] for k, v in data.items()}
            X_level = s.build_design_matrix(level_data)

            col_start = level_idx * k_per
            col_end = col_start + k_per

            # v1.6: Insert WITHOUT toarray(). Convert sparse X_level
            # to COO for efficient element-wise insertion into LIL.
            if sparse.issparse(X_level):
                X_coo = X_level.tocoo()
                for r, c, v in zip(X_coo.row, X_coo.col, X_coo.data):
                    X[row_indices[r], col_start + c] = v
            else:
                X[np.ix_(row_indices, range(col_start, col_end))] = X_level

        return X.tocsc()

    def build_penalty_matrices(self):
        # Shared penalty across all levels (block-diagonal)
        level_penalties = self._per_level_smooths[self._levels[0]].build_penalty_matrices()
        n_levels = len(self._levels)
        self.penalty_matrices = []
        for S in level_penalties:
            self.penalty_matrices.append(
                sparse.block_diag([S] * n_levels, format='csc')
            )
        self.n_penalties = len(self.penalty_matrices)
        return self.penalty_matrices
```

### 5.7 The `by` Variable Mechanism

The `by` argument in smooth terms (`s(x, by=z)`) interacts a smooth with another variable. The behavior depends on whether `by` is numeric or a factor:

#### 5.7.1 Numeric `by` (Varying-Coefficient Model)

`s(x, by=z)` where `z` is continuous creates a varying-coefficient model: the smooth `f(x)` is multiplied pointwise by `z`. The design matrix is the element-wise product of `z` with the smooth basis:

```python
# smooths/by_variable.py

def apply_numeric_by(X_smooth, z):
    """
    Numeric by: X_by[i, j] = z[i] * X_smooth[i, j]

    The smooth f(x) becomes z * f(x). Penalty is unchanged
    (still penalizes wiggliness of f, not z).
    """
    if sparse.issparse(X_smooth):
        # Sparse: scale rows without densifying
        return sparse.diags(z) @ X_smooth
    else:
        return z[:, None] * X_smooth
```

This is straightforward — same penalty count, same identifiability constraints. The only subtlety is that the smooth may need centering relative to the `by` variable.

#### 5.7.2 Factor `by` (Separate Smooth Per Level)

`s(x, by=fac)` where `fac` is a categorical factor creates a **separate smooth of x for each factor level**, each with its own smoothing parameter λ. This is fundamentally different from `bs="fs"` (Section 5.6):

| | `s(x, by=fac)` (factor by) | `s(x, fac, bs="fs")` (factor-smooth) |
|---|---|---|
| Smoothing parameters | **Separate λ per level** — each group's wiggliness is estimated independently | Single shared λ — all levels share one smoothing parameter |
| Shrinkage | None between levels — each smooth is fully independent | Levels shrink toward each other (random-effect-like) |
| Penalty count | `n_levels` penalties (one per level-smooth) | 1 penalty (block-diagonal, shared λ) |
| Identifiability | Each level-smooth needs its own constraint (or a main-effect `s(x)` absorbs the null space) | Global identifiability via the shared penalty |
| Use case | Genuinely different functional forms per group | Similar shapes across groups, borrowing strength |
| REML dimension | Adds `n_levels` smoothing parameters to optimize | Adds 1 smoothing parameter |

**Design matrix construction:**

```python
# smooths/by_variable.py

class FactorBySmooth:
    """
    s(x, by=fac): creates one smooth per factor level, each with its
    own basis evaluation, penalty, and smoothing parameter.

    The global design matrix has block structure: for K levels and a
    base smooth with p columns, the result has K*p columns. Row i has
    nonzeros only in the block corresponding to fac[i]'s level.

    This is NOT a subclass of Smooth. It's a model-assembly mechanism
    that expands one SmoothSpec into K independent Smooth objects.
    """

    def __init__(self, base_smooth_class: type, spec: SmoothSpec):
        self._base_class = base_smooth_class
        self._spec = spec
        self._levels = None  # Set during setup
        self._per_level_smooths = {}

    def setup(self, data, knots=None):
        """Create one Smooth per factor level."""
        fac = data[self._spec.by_variable]
        self._levels = np.unique(fac)

        for level in self._levels:
            mask = fac == level
            level_spec = SmoothSpec(
                variables=self._spec.variables,
                basis_type=self._spec.basis_type,
                n_knots=self._spec.n_knots,
                by_variable=None,  # Level-smooth has no further 'by'
            )
            smooth = self._base_class(level_spec)
            # Setup on the FULL data (knots should reflect global range)
            # but basis evaluation will be masked per level
            smooth.setup(data, knots=knots)
            self._per_level_smooths[level] = smooth

        base = self._per_level_smooths[self._levels[0]]
        self.n_coefs = len(self._levels) * base.n_coefs
        self.null_space_dim = len(self._levels) * base.null_space_dim

    def build_design_matrix(self, data):
        """
        Block-diagonal design matrix: K blocks of p columns each.

        For level k, rows where fac==k get the smooth basis; all other
        rows are zero in that block. Sparse assembly (no toarray()).
        """
        fac = data[self._spec.by_variable]
        n = len(fac)
        base = self._per_level_smooths[self._levels[0]]
        k_per = base.n_coefs
        total_cols = len(self._levels) * k_per

        X = sparse.lil_matrix((n, total_cols))

        for level_idx, level in enumerate(self._levels):
            mask = fac == level
            row_indices = np.where(mask)[0]
            if len(row_indices) == 0:
                continue

            s = self._per_level_smooths[level]
            level_data = {k: v[mask] for k, v in data.items()}
            X_level = s.build_design_matrix(level_data)

            col_start = level_idx * k_per
            # Sparse insertion without toarray() (v1.6 pattern)
            if sparse.issparse(X_level):
                X_coo = X_level.tocoo()
                for r, c, v in zip(X_coo.row, X_coo.col, X_coo.data):
                    X[row_indices[r], col_start + c] = v
            else:
                X[np.ix_(row_indices, range(col_start, col_start + k_per))] = X_level

        return X.tocsc()

    def build_penalty_matrices(self):
        """
        One penalty per level: K independent penalty matrices, each
        embedded in the global (total_p × total_p) space.

        Each penalty gets its OWN smoothing parameter λ_k in the
        REML outer loop. This is the key difference from bs="fs",
        which shares one λ across all levels.
        """
        base = self._per_level_smooths[self._levels[0]]
        k_per = base.n_coefs
        total_p = len(self._levels) * k_per
        penalties = []

        for level_idx, level in enumerate(self._levels):
            s = self._per_level_smooths[level]
            S_level_list = s.build_penalty_matrices()

            for S_level in S_level_list:
                # Embed level-penalty in global space
                S_global = sparse.lil_matrix((total_p, total_p))
                col_start = level_idx * k_per
                col_end = col_start + k_per

                if sparse.issparse(S_level):
                    S_coo = S_level.tocoo()
                    for r, c, v in zip(S_coo.row, S_coo.col, S_coo.data):
                        S_global[col_start + r, col_start + c] = v
                else:
                    S_global[col_start:col_end, col_start:col_end] = S_level

                penalties.append(S_global.tocsc())

        self.penalty_matrices = penalties
        self.n_penalties = len(penalties)
        return penalties
```

#### 5.7.3 Identifiability: `s(x, by=fac)` Alongside `s(x)`

When a model contains both `s(x)` and `s(x, by=fac)`, the smooth's null space (typically the constant and linear functions) is confounded between the main effect and the by-smooths. mgcv handles this by absorbing the null space of the by-smooths — each level-smooth is constrained so its null-space component is zero, leaving the main-effect `s(x)` to capture the shared constant/linear trend.

This interacts with the `CoefficientMap` (Section 5.10) via `apply_joint_identifiability`:

```python
# fitting/constraints.py — factor-by identifiability

def constrain_factor_by_smooths(factor_by_smooth, main_effect_present):
    """
    Apply identifiability constraints to factor-by smooths.

    Case 1: s(x, by=fac) WITHOUT s(x) in the model.
        No extra constraint needed. Each level-smooth is fully
        identified (the parametric part of the model handles the
        intercept per level via the factor main effect).

    Case 2: s(x, by=fac) WITH s(x) in the model.
        Each level-smooth must have its null space absorbed
        (sum-to-zero + zero-slope constraints). This ensures
        the main effect s(x) captures the "average" smooth and
        the by-smooths capture level-specific DEVIATIONS.

        Implemented via QR reparameterization of each level's
        basis: project out the null space columns. The
        CoefficientMap records this projection for predict().

    Case 3: s(x, by=fac) WITHOUT factor main effect in the model.
        The factor main effect (dummy variables) should be in the
        parametric part of the model for identifiability. If absent,
        warn — the level-smooths' intercepts are confounded.
    """
    if not main_effect_present:
        return  # Case 1: no constraint needed

    base = factor_by_smooth._per_level_smooths[factor_by_smooth._levels[0]]
    null_dim = base.null_space_dim

    if null_dim == 0:
        return  # Fully penalized smooth (e.g., random effect): nothing to absorb

    constraints = []
    for level_idx, level in enumerate(factor_by_smooth._levels):
        s = factor_by_smooth._per_level_smooths[level]
        # QR decomposition of the null space basis
        # Constraint: C^T β_level = 0 where C spans the null space
        C = s.get_null_space_basis()  # (p, null_dim) matrix
        constraints.append(TermConstraint(
            term_index=level_idx,
            constraint_matrix=C,
            method='absorb',  # QR reparameterization
        ))

    return constraints
```

#### 5.7.4 REML Dimension Scaling

Factor-by smooths directly affect the REML outer loop because each level adds its own smoothing parameter:

```
Model: y ~ s(x1) + s(x2, by=fac)   where fac has 5 levels

Smoothing parameters to optimize:
  λ_1          for s(x1)                         — 1 parameter
  λ_2 ... λ_6  for s(x2, by=fac), one per level  — 5 parameters
  Total: 6 smoothing parameters (n_smooth = 6)

REML gradient:  (6,) vector
REML Hessian:   (6, 6) matrix
Newton step:    O(6³) = trivial
```

This is still low-dimensional optimization, but it scales linearly with the number of factor levels. For factors with many levels (e.g., US states: 50 levels), the REML dimension can grow significantly:

| Scenario | n_smooth | REML Newton cost | Concern |
|---|---|---|---|
| `s(x, by=fac)`, 5 levels | 5 | O(125) | Negligible |
| `s(x, by=fac)`, 50 levels | 50 | O(125k) | Negligible |
| `s(x1, by=fac) + s(x2, by=fac)`, 50 levels | 100 | O(1M) | Noticeable but fine |
| `s(x1, by=fac) + s(x2, by=fac) + ... (10 smooths)`, 50 levels | 500 | O(125M) | REML Hessian becomes expensive; switch to fREML/Fellner-Schall |

For models where factor-by pushes n_smooth above ~100, the doc's fREML (Section 8.3) or Fellner-Schall update (Section 8.4) should be preferred over Newton-based REML, as they avoid forming the full Hessian.

```python
# fitting/reml.py — scaling check

def check_reml_dimension(n_smooth, method):
    """
    Warn if REML dimension is large enough to prefer fREML.
    Factor-by smooths are the main source of high n_smooth.
    """
    if method == "REML" and n_smooth > 100:
        import warnings
        warnings.warn(
            f"Model has {n_smooth} smoothing parameters. "
            f"Newton REML requires a ({n_smooth},{n_smooth}) Hessian. "
            f"Consider method='fREML' or method='fellner_schall' for "
            f"faster smoothing parameter estimation."
        )
```

#### 5.7.5 Formula Parser Integration

The formula parser must detect whether `by` is numeric or factor and route to the correct assembly path:

```python
# formula/smooth_parser.py — by-variable dispatch

def resolve_by_variable(spec: SmoothSpec, data) -> list:
    """
    Resolve by-variable into concrete smooth(s).

    Returns a list because factor-by expands one SmoothSpec into
    one FactorBySmooth (which internally holds K smooths).
    Numeric-by returns the original smooth with a flag.

    Called during model matrix assembly (Phase 1), not during parsing.
    """
    if spec.by_variable is None:
        return [spec]  # No by: standard smooth

    by_col = data[spec.by_variable]

    if _is_factor(by_col):
        # Factor by: expand into FactorBySmooth
        base_class = get_smooth_class(spec.basis_type)
        factor_smooth = FactorBySmooth(base_class, spec)
        return [factor_smooth]
    else:
        # Numeric by: flag for pointwise multiplication
        spec.by_numeric = True
        return [spec]


def _is_factor(col):
    """
    Detect whether a column is a factor.
    - pandas Categorical / object dtype → factor
    - numpy string/object dtype → factor
    - integer with few unique values → NOT automatically factor
      (user must explicitly cast; avoids silent misinterpretation)
    """
    if hasattr(col, 'cat'):  # pandas Categorical
        return True
    if col.dtype == object or col.dtype.kind in ('U', 'S'):
        return True
    return False
```

### 5.8 Additional Smooth Classes (Specifications)

Each of the following must be fully implemented following the same `Smooth` interface:

| Class | File | Key Details |
|---|---|---|
| `bs` (B-splines) | `bspline.py` | Standard B-spline basis, derivative-based penalty, variable order |
| `gp` (Gaussian Process) | `gaussian_process.py` | Matérn, exponential, power-exponential kernels; covariance as penalty |
| `mrf` (Markov Random Field) | `mrf.py` | User-supplied neighbourhood penalty matrix; sparse identity-like basis |
| `so` (Soap Film) | `soap_film.py` | Boundary-respecting 2D smooth; PDE-based; requires boundary polygon |
| `sz` (Duchon) | `duchon.py` | Generalization of TPRS with fractional derivatives |
| `ad` (Adaptive) | `adaptive.py` | Locally varying smoothness; multiple penalties with spatially varying weights |
| Linear functional | `linear_functional.py` | Integral/functional covariates; basis is integral of standard basis |

### 5.9 Smooth Registry

```python
# smooths/registry.py

_SMOOTH_REGISTRY: dict[str, type[Smooth]] = {}

def register_smooth(bs_name: str, smooth_class: type[Smooth]):
    _SMOOTH_REGISTRY[bs_name] = smooth_class

def get_smooth_class(bs_name: str) -> type[Smooth]:
    if bs_name not in _SMOOTH_REGISTRY:
        raise ValueError(f"Unknown smooth basis type: {bs_name}")
    return _SMOOTH_REGISTRY[bs_name]

# Auto-register all built-in smooths
register_smooth("tp", ThinPlateSmooth)
register_smooth("ts", ThinPlateShrinkageSmooth)
register_smooth("cr", CubicRegressionSmooth)
register_smooth("cs", CubicShrinkageSmooth)
register_smooth("cc", CyclicCubicSmooth)
register_smooth("ps", PSplineSmooth)
register_smooth("cp", CyclicPSplineSmooth)
register_smooth("bs", BSplineSmooth)
register_smooth("ad", AdaptiveSmooth)
register_smooth("gp", GaussianProcessSmooth)
register_smooth("mrf", MarkovRandomFieldSmooth)
register_smooth("re", RandomEffectSmooth)
register_smooth("fs", FactorSmoothInteractionSmooth)
register_smooth("so", SoapFilmSmooth)
register_smooth("sz", DuchonSplineSmooth)
```

### 5.10 CoefficientMap and Joint Identifiability (`gam_side`)

**v1.2 bug fix (refined in v1.3):** v1.2 introduced `gam_side` with a heuristic threshold (`overlap > 0.9`), in-place `term_info` mutation, and `_to_dense` calls that could blow memory on the sparse path. v1.3 replaces this with a rigorous `CoefficientMap` layer.

**The `CoefficientMap` is a first-class, immutable object** that records every constraint and reparameterization applied to the model matrix. It is stored in the `GAMResult` and used by all post-estimation code (predict, summary, concurvity, anova). No code ever mutates `term_info` column indices in-place.

```python
# fitting/coefficient_map.py

from dataclasses import dataclass, field
import numpy as np
from scipy import sparse

@dataclass(frozen=True)  # Immutable
class TermBlock:
    """One term's position and reparameterization in the model matrix."""
    label: str
    col_start: int           # Column offset in FINAL (constrained) X
    n_coefs: int             # Number of columns in FINAL X
    n_coefs_raw: int         # Number of columns BEFORE constraints
    type: str                # "parametric" | "smooth"
    smooth: object = None    # Reference to Smooth object (if smooth)
    penalty_indices: tuple = ()  # Indices into global penalty list

    # Reparameterization chain: raw_beta = Z_sum_to_zero @ Z_side @ coefs
    # Each Z is a (n_raw, n_constrained) matrix.
    # If no constraints, this is [I].
    reparam_matrices: tuple = ()  # Tuple of np.ndarray

    def raw_to_constrained(self, beta_raw):
        """Map raw coefficients to constrained space."""
        b = beta_raw
        for Z in reversed(self.reparam_matrices):
            b = Z @ b
        return b

    def constrained_to_raw(self, beta_constrained):
        """Map constrained coefficients back to raw space (for prediction)."""
        b = beta_constrained
        for Z in self.reparam_matrices:
            b = Z @ b
        return b


@dataclass(frozen=True)
class CoefficientMap:
    """
    Global, immutable mapping from model coefficients to term structure.

    Created once during model setup. Used by predict(), summary(),
    concurvity(), and all post-estimation code.
    """
    terms: tuple[TermBlock, ...]
    total_coefs: int
    penalty_matrices: tuple  # Global list of penalty matrices

    def get_term(self, label: str) -> TermBlock:
        for t in self.terms:
            if t.label == label:
                return t
        raise KeyError(f"No term '{label}'")

    def term_slice(self, label: str) -> slice:
        t = self.get_term(label)
        return slice(t.col_start, t.col_start + t.n_coefs)

    def build_prediction_matrix(self, new_data, terms=None, exclude=None):
        """
        Build Xp for prediction, applying all reparameterizations.
        Uses the same CoefficientMap that was used during fitting.
        """
        blocks = []
        for term in self.terms:
            if exclude and term.label in exclude:
                continue
            if terms and term.label not in terms:
                continue
            if term.type == "parametric":
                Xp_raw = _build_parametric_block(new_data, term)
            else:
                Xp_raw = term.smooth.predict_matrix(new_data)
            # Apply same reparameterizations used during fitting
            for Z in term.reparam_matrices:
                Xp_raw = Xp_raw @ Z
            blocks.append(Xp_raw)
        return np.column_stack(blocks) if blocks else np.empty((0, 0))


def apply_joint_identifiability(X_raw, term_blocks_raw, smooth_objects):
    """
    gam.side equivalent: resolve inter-term identifiability
    and return an immutable CoefficientMap.

    v1.3 improvements over v1.2:
    - SVD-based rank test replaces arbitrary threshold
    - No in-place mutation of term_info
    - No _to_dense: works on column subsets via efficient projections
    - Returns immutable CoefficientMap

    Algorithm:
    1. For each smooth j with a non-trivial null space N_j:
    2.   Compute X_j @ N_j (the "null space projection" of term j)
    3.   For each other smooth k:
    4.     Compute SVD of [X_k^+ @ X_j @ N_j] to get overlap
    5.     If any singular values > 1 - eps (near-perfect overlap):
    6.       Record a constraint: remove those directions from k's basis
    7. Build CoefficientMap with all recorded reparameterizations.
    """
    term_blocks = list(term_blocks_raw)  # Work on a copy
    reparam_chains = {i: [] for i in range(len(term_blocks))}

    for j, block_j in enumerate(term_blocks):
        if block_j['type'] != 'smooth':
            continue
        smooth_j = block_j['smooth']
        if smooth_j.null_space_dim == 0:
            continue

        cols_j = slice(block_j['col_start'], block_j['col_end'])
        N_j = _get_null_space_basis(smooth_j)
        if N_j is None:
            continue

        # X_j @ N_j: the part of term j that sits in the null space.
        # N_j is small (null_space_dim columns), so sparse @ dense is fine.
        X_j_block = X_raw[:, cols_j]
        if sparse.issparse(X_j_block):
            XjN = X_j_block.toarray() @ N_j  # (n, null_dim) — null_dim is tiny
        else:
            XjN = X_j_block @ N_j

        for k, block_k in enumerate(term_blocks):
            if k == j or block_k['type'] != 'smooth':
                continue
            cols_k = slice(block_k['col_start'], block_k['col_end'])
            X_k = X_raw[:, cols_k]
            n_cols_k = block_k['col_end'] - block_k['col_start']

            # v1.5: sparse-safe SVD for large term blocks.
            # For n_cols_k > 10,000 (e.g., MRF, large factor-smooth),
            # use randomized SVD on the sparse matrix directly.
            LARGE_BLOCK = 10_000
            if sparse.issparse(X_k) and n_cols_k > LARGE_BLOCK:
                # Randomized SVD via ARPACK — works on sparse, O(nnz * k)
                from scipy.sparse.linalg import svds
                # Only need top few singular vectors for overlap detection
                n_svd = min(smooth_j.null_space_dim + 5, n_cols_k - 1)
                U_k, s_k, _ = svds(X_k, k=n_svd)
                rank_k = np.sum(s_k > 1e-10 * s_k[0])
                U_k = U_k[:, :rank_k]
            else:
                if sparse.issparse(X_k):
                    X_k = X_k.toarray()
                U_k, s_k, _ = np.linalg.svd(X_k, full_matrices=False)
                rank_k = np.sum(s_k > 1e-10 * s_k[0])
                U_k = U_k[:, :rank_k]

            # Project XjN onto col(X_k)
            proj_coefs = U_k.T @ XjN  # (rank_k, null_dim_j)
            _, s_overlap, _ = np.linalg.svd(proj_coefs)

            # Overlap directions: singular values near ||XjN_col||
            xjn_norms = np.linalg.norm(XjN, axis=0)
            n_overlap = np.sum(s_overlap > 0.99 * np.max(xjn_norms))

            if n_overlap > 0:
                # Constraint: remove overlapping directions from term k
                # via QR of the overlap coefficients
                C = np.linalg.lstsq(X_k, XjN[:, :n_overlap], rcond=None)[0]
                Q, R = np.linalg.qr(C, mode='complete')
                Z = Q[:, n_overlap:]  # Columns orthogonal to overlap
                reparam_chains[k].append(Z)

    # Build immutable CoefficientMap
    final_terms = []
    col_offset = 0
    for i, block in enumerate(term_blocks):
        n_raw = block['n_coefs']
        reparam = tuple(reparam_chains.get(i, []))
        n_final = n_raw
        for Z in reparam:
            n_final = Z.shape[1]

        final_terms.append(TermBlock(
            label=block.get('label', 'parametric'),
            col_start=col_offset,
            n_coefs=n_final,
            n_coefs_raw=n_raw,
            type=block['type'],
            smooth=block.get('smooth'),
            penalty_indices=tuple(block.get('penalty_indices', [])),
            reparam_matrices=reparam,
        ))
        col_offset += n_final

    return CoefficientMap(
        terms=tuple(final_terms),
        total_coefs=col_offset,
        penalty_matrices=tuple(),  # Filled by caller
    )
```

---

## 6. Distribution Families

### 6.1 Family Base Classes

```python
# families/base.py

from abc import ABC, abstractmethod
from pymgcv.links.links import Link
import numpy as np

class Family(ABC):
    """
    Base class for exponential family distributions.

    Standard families provide:
    - variance(mu): V(μ)
    - deviance_residuals(y, mu, wt): deviance residuals
    - log_likelihood(y, mu, scale, wt): log-likelihood

    The PIRLS algorithm uses: working weights W = 1/(V(μ) * g'(μ)²)
    and working response z = η + (y - μ) * g'(μ)
    """

    def __init__(self, link: str | Link = None):
        self.link: Link = self._resolve_link(link)
        self.family_name: str = ""
        self.n_theta: int = 0  # Number of extra parameters to estimate
        self.scale_known: bool = False

    @abstractmethod
    def variance(self, mu: np.ndarray) -> np.ndarray:
        """Variance function V(μ)."""
        ...

    @abstractmethod
    def deviance(self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray) -> float:
        """Total deviance: Σ wt_i * d_i."""
        ...

    def log_likelihood(self, y, mu, scale, wt):
        """Log-likelihood. Default: computed from deviance for EDMs."""
        return -0.5 * self.deviance(y, mu, wt) / scale

    def working_weights(self, mu, wt):
        """PIRLS working weights: W = wt / (V(μ) * g'(μ)²)."""
        g_prime = self.link.derivative(mu)
        return wt / (self.variance(mu) * g_prime ** 2)

    def working_response(self, y, mu, eta):
        """PIRLS working response: z = η + (y - μ) * g'(μ)."""
        g_prime = self.link.derivative(mu)
        return eta + (y - mu) * g_prime

    def initialize(self, y, wt):
        """Initialize μ from y. Called before first PIRLS iteration."""
        return (y + y.mean()) / 2  # Safe default

    def scale_estimate(self, y, mu, wt, n, p):
        """Estimate dispersion/scale parameter φ."""
        if self.scale_known:
            return 1.0
        return self.deviance(y, mu, wt) / (n - p)


class ExtendedFamily(Family):
    """
    Extended family for distributions not in the exponential family.

    CONTRACT B (v1.3): Extended families provide ONLY a per-observation
    log-likelihood as a pure JAX function. The framework owns ALL
    differentiation, stabilization (damping, clipping), and conversion
    to working weights/response. No family ever computes its own derivatives.

    The canonical interface is:

        loglik_per_obs(eta_i: scalar, y_i: scalar, theta: array) → scalar

    This function must be:
    1. A pure function of its arguments (no side effects, no self)
    2. Written in pure JAX (jax.numpy, jax.scipy, jax.lax only)
    3. Differentiable by JAX — use stable primitives (lgamma, logsumexp, log-space arithmetic) so jax.grad produces stable gradients. Only Tweedie needs custom_jvp (see Section 9.3).
    4. Parameterized in eta (linear predictor), NOT mu — this avoids
       a chain-rule step and keeps derivatives in the space where
       the fitting algorithm operates.

    The framework then computes:
    - dl/deta_i and d²l/deta_i² via jax.vmap(jax.grad(...))
    - Working weights: W_i = max(-d²l/deta_i², epsilon)
    - Working response: z_i = eta_i + (dl/deta_i) / W_i
    - All stabilization: weight clamping, Hessian damping, step control
    """

    def __init__(self, link=None):
        super().__init__(link)
        self.is_extended = True
        self.n_theta: int = 0  # Number of extra parameters
        self.theta_init: np.ndarray = np.array([])

    @abstractmethod
    def loglik_per_obs_fn(self):
        """
        Return a pure JAX function with signature:

            f(eta_i: float, y_i: float, theta: jax.Array) -> float

        This is a STATIC function (no self), suitable for JIT and vmap.
        It must import ONLY from jax.*.

        Example for Negative Binomial:
            def nb_loglik(eta_i, y_i, theta):
                mu_i = jnp.exp(eta_i)  # log link
                th = jnp.exp(theta[0])
                return (jax.scipy.special.gammaln(y_i + th) - ...)
        """
        ...

    def theta_bounds(self) -> list[tuple[float, float]]:
        """Bounds for extra parameters (for constrained optimization)."""
        return [(-np.inf, np.inf)] * self.n_theta

    # ── Framework-provided methods (families do NOT override these) ──

    def _compute_working_quantities(self, eta, y, theta, wt):
        """
        Framework method: compute W and z from the family's loglik_per_obs_fn.

        This is the ONLY place derivatives are taken. Families never
        see gradients or Hessians.

        Called by the extended PIRLS loop (Section 7.3).
        """
        import jax
        ll_fn = self.loglik_per_obs_fn()

        # dl/deta per observation (vmapped reverse-mode)
        dll_deta = jax.vmap(jax.grad(ll_fn, argnums=0))(eta, y, theta)

        # d²l/deta² per observation (vmapped forward-over-reverse)
        def hess_single(eta_i, y_i, theta_):
            return jax.grad(jax.grad(ll_fn, argnums=0), argnums=0)(
                eta_i, y_i, theta_
            )
        d2ll_deta2 = jax.vmap(hess_single)(eta, y, theta)

        # Stabilization: ensure positive weights
        # Floor at 1e-7 * max(|H|) to handle near-boundary cases
        abs_hess = jnp.abs(d2ll_deta2)
        hess_floor = 1e-7 * jnp.max(abs_hess)
        W = jnp.maximum(-d2ll_deta2, hess_floor) * wt

        # Working response
        z = eta + dll_deta / jnp.maximum(-d2ll_deta2, hess_floor)

        return W, z

    # Variance/deviance are not meaningful for extended families
    def variance(self, mu):
        raise NotImplementedError("Extended families use loglik_per_obs, not variance")

    def deviance(self, y, mu, wt):
        """Deviance = -2 * sum(loglik). Computed from loglik_per_obs_fn."""
        import jax.numpy as jnp
        ll_fn = self.loglik_per_obs_fn()
        eta = self.link.link(mu)
        ll_total = jnp.sum(
            jax.vmap(ll_fn)(eta, y, self.theta_init) * wt
        )
        return -2.0 * ll_total
```

### 6.2 Standard Families

```python
# families/standard.py

class Gaussian(Family):
    family_name = "gaussian"
    scale_known = False

    def __init__(self, link="identity"):
        super().__init__(link or "identity")

    def variance(self, mu):
        return np.ones_like(mu)

    def deviance(self, y, mu, wt):
        return np.sum(wt * (y - mu) ** 2)

    def log_likelihood(self, y, mu, scale, wt):
        n = len(y)
        return -0.5 * (n * np.log(2 * np.pi * scale) +
                       self.deviance(y, mu, wt) / scale)


class Binomial(Family):
    family_name = "binomial"
    scale_known = True  # Scale = 1

    def __init__(self, link="logit"):
        super().__init__(link or "logit")

    def variance(self, mu):
        return mu * (1 - mu)

    def deviance(self, y, mu, wt):
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return 2 * np.sum(wt * (
            y * np.log(np.where(y > 0, y / mu, 1)) +
            (1 - y) * np.log(np.where(y < 1, (1 - y) / (1 - mu), 1))
        ))

    def initialize(self, y, wt):
        return (y + 0.5) / 2


class Poisson(Family):
    family_name = "poisson"
    scale_known = True

    def __init__(self, link="log"):
        super().__init__(link or "log")

    def variance(self, mu):
        return mu

    def deviance(self, y, mu, wt):
        mu = np.maximum(mu, 1e-10)
        return 2 * np.sum(wt * (
            y * np.log(np.where(y > 0, y / mu, 1)) - (y - mu)
        ))

    def initialize(self, y, wt):
        return y + 0.1


class Gamma(Family):
    family_name = "Gamma"
    scale_known = False

    def __init__(self, link="inverse"):
        super().__init__(link or "inverse")

    def variance(self, mu):
        return mu ** 2


class InverseGaussian(Family):
    family_name = "inverse.gaussian"
    scale_known = False

    def __init__(self, link="1/mu^2"):
        super().__init__(link or "inverse_squared")

    def variance(self, mu):
        return mu ** 3
```

### 6.3 Extended Families (Autograd-Powered)

```python
# families/negbin.py

class NegativeBinomial(ExtendedFamily):
    """
    Negative Binomial with theta (overdispersion) estimated.

    Implements Contract B: provides loglik_per_obs_fn returning a
    pure JAX function. The function uses jax.scipy (NOT scipy).

    v1.18: Standard jax.grad through this log-likelihood is numerically
    stable. JAX differentiates lgamma → digamma, which is a well-conditioned
    special function. At large θ where digamma(y+θ)-digamma(θ) is small,
    NB converges to Poisson and gradient imprecision doesn't affect the fit.
    No custom_jvp needed.
    """
    family_name = "nb"

    def __init__(self, link="log", theta=None):
        super().__init__(link or "log")
        self.n_theta = 1
        self.theta_init = np.array([np.log(theta)]) if theta else np.array([0.0])

    def loglik_per_obs_fn(self):
        """Return pure JAX log-likelihood function."""
        # This is a static function — no self, no scipy, no numpy
        def nb_loglik(eta_i, y_i, theta):
            import jax.numpy as jnp
            import jax.scipy.special as jsp
            mu_i = jnp.exp(eta_i)  # log link baked in
            th = jnp.exp(theta[0])  # theta > 0 parameterized as log(theta)
            mu_i = jnp.maximum(mu_i, 1e-10)
            return (jsp.gammaln(y_i + th) - jsp.gammaln(th) -
                    jsp.gammaln(y_i + 1) +
                    th * jnp.log(th) + y_i * jnp.log(mu_i) -
                    (y_i + th) * jnp.log(mu_i + th))
        return nb_loglik


# families/tweedie.py

class Tweedie(ExtendedFamily):
    """
    Tweedie distribution with power parameter p ∈ (1, 2).

    Implements Contract B with a custom_jvp-registered loglik because
    the series evaluation is numerically delicate under naive AD.
    This is the ONLY family in the library requiring custom_jvp —
    all others use standard jax.grad through stable forward passes
    (see Section 9.3 for the full analysis).
    """
    family_name = "tw"

    def __init__(self, link="log", p=1.5):
        super().__init__(link or "log")
        self.n_theta = 1
        self.theta_init = np.array([p])  # Power parameter

    def loglik_per_obs_fn(self):
        """
        Return pure JAX Tweedie log-density function.

        Uses the series evaluation approach from Dunn & Smyth (2005).
        The series is computed via jax.lax.while_loop for JIT
        compatibility. A custom_jvp rule is registered separately
        (see autodiff/tweedie_jvp.py) because naive AD through
        the while_loop produces unstable gradients — the truncation
        point depends on data, and differentiating through it
        amplifies truncation error. This is the only family in the
        library that needs this treatment.
        """
        from pymgcv.autodiff.tweedie_jvp import tweedie_loglik_single
        return tweedie_loglik_single  # Has custom_jvp registered
        pass


# families/beta_family.py

class BetaFamily(ExtendedFamily):
    """
    Beta regression for y ∈ (0, 1).
    Parameterized as Beta(μφ, (1-μ)φ) where φ is precision.

    Benign under naive AD — no custom_jvp needed.
    """
    family_name = "betar"

    def __init__(self, link="logit"):
        super().__init__(link or "logit")
        self.n_theta = 1  # log(phi)
        self.theta_init = np.array([0.0])

    def loglik_per_obs_fn(self):
        def beta_loglik(eta_i, y_i, theta):
            import jax.numpy as jnp
            import jax.scipy.special as jsp
            # logit link: mu = sigmoid(eta)
            mu_i = jax.nn.sigmoid(eta_i)
            phi = jnp.exp(theta[0])
            a = mu_i * phi
            b = (1 - mu_i) * phi
            return (jsp.gammaln(phi) - jsp.gammaln(a) - jsp.gammaln(b) +
                    (a - 1) * jnp.log(y_i) + (b - 1) * jnp.log(1 - y_i))
        return beta_loglik


# families/location_scale.py

class LocationScaleFamily(ExtendedFamily):
    """
    Base for multi-parameter location-scale families where
    each parameter gets its own linear predictor.

    Examples: gaulss (Gaussian location-scale), gevlss, shash, gammals.
    These require multi-linear-predictor GAMs.
    """

    def __init__(self, links: list[str], n_params: int):
        super().__init__(links[0])
        self.links = [Link.from_name(l) for l in links]
        self.n_linear_predictors = n_params


class GaussianLocationScale(LocationScaleFamily):
    """gaulss: Gaussian with both mean and variance modeled."""
    family_name = "gaulss"

    def __init__(self):
        super().__init__(links=["identity", "log"], n_params=2)

    def log_likelihood(self, y, params, theta, wt, scale):
        mu = params[0]  # location
        sigma = np.exp(params[1])  # scale (log link)
        ll = -0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma) +
                      ((y - mu) / sigma) ** 2)
        return np.sum(wt * ll) if wt is not None else np.sum(ll)


class SHASH(LocationScaleFamily):
    """
    Sinh-arcsinh (SHASH) distribution.
    Four parameters: location μ, scale σ, skewness ε, kurtosis δ.
    """
    family_name = "shash"

    def __init__(self):
        super().__init__(
            links=["identity", "log", "identity", "log"],
            n_params=4
        )

    def log_likelihood(self, y, params, theta, wt, scale):
        mu, log_sigma, eps, log_delta = params
        sigma = np.exp(log_sigma)
        delta = np.exp(log_delta)
        z = (y - mu) / sigma
        # sinh-arcsinh transform
        s = np.sinh(delta * np.arcsinh(z) - eps)
        C = np.cosh(delta * np.arcsinh(z) - eps)
        ll = (-0.5 * np.log(2 * np.pi) - np.log(sigma) +
              np.log(delta) + np.log(C) -
              0.5 * np.log(1 + z**2) - 0.5 * s**2)
        return np.sum(wt * ll) if wt is not None else np.sum(ll)
```

### 6.4 Link Functions

```python
# links/links.py

class Link(ABC):
    """Abstract link function: g(μ) = η, g^{-1}(η) = μ."""

    @abstractmethod
    def link(self, mu): ...

    @abstractmethod
    def inverse(self, eta): ...

    @abstractmethod
    def derivative(self, mu):
        """dη/dμ = g'(μ)."""
        ...

    @staticmethod
    def from_name(name: str) -> "Link":
        return _LINK_REGISTRY[name]()


class LogitLink(Link):
    def link(self, mu):
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.log(mu / (1 - mu))
    def inverse(self, eta):
        return 1 / (1 + np.exp(-eta))
    def derivative(self, mu):
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return 1 / (mu * (1 - mu))


class LogLink(Link):
    def link(self, mu): return np.log(np.maximum(mu, 1e-10))
    def inverse(self, eta): return np.exp(eta)
    def derivative(self, mu): return 1 / np.maximum(mu, 1e-10)


class IdentityLink(Link):
    def link(self, mu): return mu
    def inverse(self, eta): return eta
    def derivative(self, mu): return np.ones_like(mu)


class InverseLink(Link):
    def link(self, mu): return 1 / mu
    def inverse(self, eta): return 1 / eta
    def derivative(self, mu): return -1 / mu**2


class ProbitLink(Link):
    def link(self, mu):
        from scipy.special import ndtri
        return ndtri(np.clip(mu, 1e-10, 1 - 1e-10))
    def inverse(self, eta):
        from scipy.special import ndtr
        return ndtr(eta)
    def derivative(self, mu):
        from scipy.stats import norm
        return 1 / norm.pdf(self.link(mu))


class CloglogLink(Link):
    def link(self, mu):
        return np.log(-np.log(1 - np.clip(mu, 1e-10, 1 - 1e-10)))
    def inverse(self, eta):
        return 1 - np.exp(-np.exp(eta))
    def derivative(self, mu):
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return 1 / ((1 - mu) * (-np.log(1 - mu)))


class SqrtLink(Link):
    def link(self, mu): return np.sqrt(mu)
    def inverse(self, eta): return eta ** 2
    def derivative(self, mu): return 0.5 / np.sqrt(np.maximum(mu, 1e-10))


class InverseSquaredLink(Link):
    """g(μ) = 1/μ² — default link for Inverse Gaussian family."""
    def link(self, mu): return 1 / np.maximum(mu, 1e-10) ** 2
    def inverse(self, eta): return 1 / np.sqrt(np.maximum(eta, 1e-10))
    def derivative(self, mu): return -2 / np.maximum(mu, 1e-10) ** 3


_LINK_REGISTRY = {
    "logit": LogitLink, "log": LogLink, "identity": IdentityLink,
    "inverse": InverseLink, "probit": ProbitLink, "cloglog": CloglogLink,
    "sqrt": SqrtLink, "inverse_squared": InverseSquaredLink,
}
```

---

## 7. Penalized Iteratively Re-weighted Least Squares (PIRLS)

### 7.1 StatisticsProvider Protocol

**v1.11 note:** With JAX-native SPMD (Section 16), the distributed/multi-GPU path no longer needs `StatisticsProvider` — the same `pirls_step_jax` function works with sharded arrays. `StatisticsProvider` remains the abstraction for two cases: (1) the **out-of-core** path where data exceeds aggregate device memory (`ChunkedJAXProvider`, Section 16.5), and (2) the **NumPy reference** PIRLS path (Section 7.2) used for testing and Sparse-CPU execution.

The PIRLS loop only needs two p-dimensional objects per iteration — `XtWX` (p×p) and `XtWz` (p×1) — regardless of n. By abstracting data access behind a `StatisticsProvider`, the reference PIRLS loop works for in-memory and out-of-core data.

```python
# distributed/stats_provider.py

from typing import Protocol
import numpy as np

class StatisticsProvider(Protocol):
    """
    Protocol for computing PIRLS sufficient statistics from data.

    v1.3 EXTENDED: The provider returns an IterationStatistics object
    that contains everything needed for BOTH coefficient updates AND
    smoothing parameter estimation (REML/GCV/EDF). This closes the
    gap identified in review: (XtWX, XtWz) alone is insufficient
    for log-determinants and trace computations needed by REML.

    The key insight: all REML/GCV quantities derive from the p×p
    matrix H = XtWX + S_λ, which the PIRLS loop already computes.
    The provider only needs to supply XtWX, deviance, and log-lik —
    the smoothing parameter optimizer handles the rest using H.
    """

    def compute_iteration_stats(self, beta: np.ndarray, family,
                                wt: np.ndarray) -> "IterationStatistics":
        """Compute all statistics needed for one PIRLS + outer iteration."""
        ...

    def compute_deviance(self, beta: np.ndarray, family,
                         wt: np.ndarray) -> float:
        """Lightweight deviance-only computation for step halving.
        Does NOT compute working weights/response or cross-products."""
        ...

    @property
    def n_observations(self) -> int: ...

    @property
    def n_parameters(self) -> int: ...


@dataclass
class IterationStatistics:
    """
    Complete sufficient statistics for one PIRLS iteration
    plus smoothing parameter estimation.

    All quantities are p-dimensional (or scalar) regardless of n.
    """
    XtWX: np.ndarray          # p × p: weighted cross-product
    XtWz: np.ndarray          # p × 1: weighted cross-product with response
    deviance: float           # Scalar: family deviance at current beta
    log_likelihood: float     # Scalar: full log-likelihood (for AIC/BIC)
    n_obs: int                # Scalar: number of observations in this compute
    sum_log_weights: float    # Scalar: Σ log(W_i) — needed for REML constant

    # ── Derived quantities (computed by the fitting loop, not the provider) ──
    # These are filled in by the PIRLS/outer loop using the above:
    #   H = XtWX + S_λ                    (penalized cross-product)
    #   log_det_H = logdet(H)             (via Cholesky of H)
    #   edf_total = tr(H^{-1} XtWX)      (trace of p×p product)
    #   edf_per_term[j] = tr(H^{-1} XtWX restricted to cols of term j)
    #   reml = deviance + β^T S_λ β + log_det_H - log_det_S + const


class InMemoryProvider:
    """Standard in-memory provider. X and y are numpy/JAX arrays."""

    def __init__(self, X, y, offset=None):
        self.X = X
        self.y = y
        self.offset = offset or np.zeros(len(y))
        self._n, self._p = X.shape

    def compute_iteration_stats(self, beta, family, wt):
        eta = self.X @ beta + self.offset
        mu = family.link.inverse(eta)
        W = family.working_weights(mu, wt)
        z = family.working_response(self.y, mu, eta - self.offset)

        W = np.clip(W, 1e-10, 1e10)
        W_sqrt = np.sqrt(W)
        WX = W_sqrt[:, None] * self.X
        XtWX = WX.T @ WX
        XtWz = WX.T @ (W_sqrt * z)

        dev = family.deviance(self.y, mu, wt)
        ll = family.log_likelihood(self.y, mu, scale=1.0, wt=wt) \
            if hasattr(family, 'log_likelihood') else -0.5 * dev
        sum_log_w = np.sum(np.log(np.maximum(W, 1e-300)))

        return IterationStatistics(
            XtWX=XtWX, XtWz=XtWz, deviance=dev,
            log_likelihood=ll, n_obs=self._n,
            sum_log_weights=sum_log_w,
        )

    def compute_deviance(self, beta, family, wt):
        """Lightweight deviance for step halving (no cross-products)."""
        eta = self.X @ beta + self.offset
        mu = family.link.inverse(eta)
        return family.deviance(self.y, mu, wt)

    @property
    def n_observations(self): return self._n

    @property
    def n_parameters(self): return self._p
```

### 7.2 Standard PIRLS (gam.fit3 equivalent)

This is the inner loop that estimates coefficients β for fixed smoothing parameters λ.

```python
# fitting/pirls.py

from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass
class PIRLSResult:
    coefficients: np.ndarray       # β
    fitted_values: np.ndarray      # μ = g^{-1}(Xβ) [or None, computed lazily]
    linear_predictor: np.ndarray   # η = Xβ [or None]
    working_weights: np.ndarray    # W [or None]
    deviance: float
    penalized_deviance: float
    n_iter: int
    converged: bool
    hat_matrix_trace: float        # tr(A) for EDF
    Vp: np.ndarray                 # Bayesian covariance matrix
    final_stats: object = None     # IterationStatistics for REML/GCV


def pirls_fit(provider: "StatisticsProvider", family, smoothing_penalties,
              weights=None, beta_init=None, max_iter=100, tol=1e-7):
    """
    Penalized IRLS for GAM fitting.

    v1.9: max_iter is mandatory and always enforced (default 100, never
    None or inf). Additionally, 3 consecutive "stalled" iterations
    (instability event with no objective progress) trigger early
    termination to prevent livelock from expensive stats recomputation.

    This implements the core of mgcv's gam.fit3, now accepting a
    StatisticsProvider instead of raw arrays:
    1. Initialize μ from family.initialize()
    2. Iterate:
       a. Provider computes XtWX, XtWz (sufficient statistics)
       b. Solve (XtWX + S_λ) β = XtWz
       c. Step halving with penalized deviance monitoring
       d. Check convergence on BOTH penalized deviance AND coefficient change
    3. Return coefficients and diagnostics

    Convergence improvements over v1.0 (matching mgcv's battle-tested logic):
    - Tracks penalized deviance (not raw deviance) to prevent false convergence
    - Allows deviance increase for first 3 iterations (warm-up phase)
    - Dual convergence criterion: deviance change AND coefficient change
    - Weight floor at 1e-7 * max(W) to handle binomial boundary cases
    - Trust-region fallback when step halving exhausts

    Parameters
    ----------
    provider : StatisticsProvider
        Data access abstraction (in-memory, Dask, Ray, etc.)
    family : Family
        Distribution family with link function
    smoothing_penalties : array, shape (p, p)
        Combined penalty matrix: S_λ = Σ λ_j S_j
    weights : array, shape (n,), optional
        Prior weights
    """
    n = provider.n_observations
    p = provider.n_parameters
    wt = weights if weights is not None else np.ones(n)

    # 1. Initialize
    beta = beta_init if beta_init is not None else np.zeros(p)
    S_lambda = smoothing_penalties
    if sparse.issparse(S_lambda):
        S_lambda_dense = S_lambda.toarray()
    else:
        S_lambda_dense = S_lambda

    converged = False
    # State: one objective value (pen_dev), set after first accepted step.
    # No inf initialization — first iteration is unconditionally accepted.
    pen_dev_prev = None  # Sentinel: first iteration always accepted
    instability_count = 0  # Unified counter for all failure signals
    jitter_applied = 0.0   # Track regularization for diagnostics
    consecutive_stalls = 0 # v1.9: 3 consecutive → early termination

    # v1.9: max_iter is mandatory. Guard against misuse.
    assert max_iter is not None and max_iter > 0 and np.isfinite(max_iter), \
        f"max_iter must be a positive integer, got {max_iter}"

    for iteration in range(max_iter):
        # 2a. Provider computes full iteration statistics
        stats = provider.compute_iteration_stats(beta, family, wt)

        # 2b. Solve H β = g where H = XtWX + S_λ (SPD by construction)
        #
        # ⚠️ REFERENCE IMPLEMENTATION (NumPy). The production Dense-GPU
        # path uses jnp.* equivalents with identical logic (Section 4.2).
        # This snippet uses np.* for clarity and to serve as the
        # Sparse-CPU / NumPy-reference path.
        #
        # Solver strategy (Section 10.3):
        # 1. Cholesky (default, O(p³/3))
        # 2. Cholesky with scale-relative jitter (near-singular)
        # 3. SVD-based lstsq (last resort)
        #
        # Instability detection (v1.8): three signals, one counter:
        #   - Cholesky failure (H lost positive-definiteness)
        #   - NaN/Inf in beta_new
        #   - Step-halving exhaustion (all 25 halvings + tiny step failed)
        # After 2 events, warn user to switch to Sparse-CPU.
        H = stats.XtWX + S_lambda_dense
        cholesky_failed = False

        # v1.8: scale-relative jitter instead of fixed 1e-12.
        # eps * trace(H)/p scales with the problem, so we don't
        # under-regularize large-scale problems or over-regularize
        # small-scale ones. Jitter level is recorded for diagnostics.
        trace_H = np.trace(H)
        eps_small = 1e-12 * trace_H / p
        eps_large = 1e-6 * trace_H / p

        try:
            L = np.linalg.cholesky(H + eps_small * np.eye(p))
            beta_new = np.linalg.solve(
                L.T, np.linalg.solve(L, stats.XtWz)
            )
        except np.linalg.LinAlgError:
            cholesky_failed = True
            jitter_applied = max(jitter_applied, eps_large)
            try:
                L = np.linalg.cholesky(H + eps_large * np.eye(p))
                beta_new = np.linalg.solve(
                    L.T, np.linalg.solve(L, stats.XtWz)
                )
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(H, stats.XtWz, rcond=None)[0]

        # NaN/Inf check on proposed coefficients
        has_nan = not np.all(np.isfinite(beta_new))
        if has_nan:
            beta_new = beta  # Can't use this step

        # Unified instability tracking (v1.8)
        if cholesky_failed or has_nan:
            instability_count += 1

        # 2c. Step halving on PENALIZED deviance
        step = beta_new - beta
        step_factor = 1.0
        accepted = False
        for _half in range(25):
            beta_try = beta + step_factor * step
            try:
                dev_try = provider.compute_deviance(beta_try, family, wt)
                pen_dev_try = dev_try + float(beta_try @ S_lambda_dense @ beta_try)
                if not np.isfinite(pen_dev_try):
                    step_factor *= 0.5
                    continue
                # First iteration: unconditionally accept (no reference point)
                if pen_dev_prev is None:
                    accepted = True
                    break
                # Subsequent: accept if objective decreases (with tolerance)
                if pen_dev_try <= pen_dev_prev + 1e-7 * abs(pen_dev_prev):
                    accepted = True
                    break
            except (ValueError, FloatingPointError):
                pass
            step_factor *= 0.5

        if not accepted:
            # v1.8: tiny step with monotonicity validation.
            # If even the tiny step increases objective, reject it entirely
            # (beta unchanged) and count as instability event.
            step_factor = 1e-4
            beta_try = beta + step_factor * step
            try:
                dev_try = provider.compute_deviance(beta_try, family, wt)
                pen_dev_try = dev_try + float(beta_try @ S_lambda_dense @ beta_try)
                if pen_dev_prev is not None and (
                    not np.isfinite(pen_dev_try)
                    or pen_dev_try > pen_dev_prev + 1e-7 * abs(pen_dev_prev)
                ):
                    # Tiny step also violates monotonicity — reject entirely
                    beta_try = beta
                    pen_dev_try = pen_dev_prev
            except (ValueError, FloatingPointError):
                beta_try = beta
                pen_dev_try = pen_dev_prev

            # Step-halving exhaustion is an instability signal (v1.8)
            instability_count += 1

        # Warn after 2+ instability events from ANY source
        if instability_count >= 2 and (cholesky_failed or has_nan or not accepted):
            import warnings
            warnings.warn(
                f"Dense path: {instability_count} instability events "
                f"(Cholesky fail / NaN / step-halving exhaustion) by "
                f"iteration {iteration}. Consider "
                f"execution_path='sparse_cpu'."
            )

        beta_old = beta
        beta = beta_try

        # 2d. Convergence check (skipped for first 3 iterations)
        if pen_dev_prev is not None and iteration >= 3:
            dev_change = abs(pen_dev_try - pen_dev_prev) / (0.1 + abs(pen_dev_try))
            coef_change = np.max(np.abs(beta - beta_old)) / (
                0.1 + np.max(np.abs(beta))
            )
            if dev_change < tol and coef_change < tol:
                converged = True
                break

        # v1.9: Consecutive stall detection (anti-livelock).
        # A "stall" = instability event with no objective progress.
        # Without this, the loop can spin doing expensive stats
        # recomputation for max_iter iterations with beta unchanged.
        this_iter_stalled = (
            (cholesky_failed or has_nan or not accepted)
            and np.array_equal(beta, beta_old)
        )
        if this_iter_stalled:
            consecutive_stalls += 1
        else:
            consecutive_stalls = 0

        if consecutive_stalls >= 3:
            import warnings
            warnings.warn(
                f"PIRLS terminated early: {consecutive_stalls} consecutive "
                f"stalled iterations (no progress + instability) at "
                f"iteration {iteration}. Model may not have converged."
            )
            break

        pen_dev_prev = pen_dev_try

    # 3. Final statistics for diagnostics and smoothing parameter estimation
    final_stats = provider.compute_iteration_stats(beta, family, wt)
    H_final = final_stats.XtWX + S_lambda_dense

    # Cholesky factorization of H (more stable than inv)
    # v1.8: scale-relative jitter, consistent with PIRLS loop
    trace_H_final = np.trace(H_final)
    try:
        L_H = np.linalg.cholesky(H_final + (1e-12 * trace_H_final / p) * np.eye(p))
    except np.linalg.LinAlgError:
        eps_final = 1e-6 * trace_H_final / p
        jitter_applied = max(jitter_applied, eps_final)
        L_H = np.linalg.cholesky(H_final + eps_final * np.eye(p))

    # EDF trace: tr(H^{-1} XtWX) without forming H^{-1}
    # Using: tr(H^{-1} A) = tr(L^{-T} L^{-1} A) = ||L^{-1} A||_F^2 ... no
    # Correct: tr(H^{-1} A) = sum_{ij} (L^{-1})_{ij} (L^{-1} A)_{ij}
    # Compute Z = L^{-1} via forward substitution, then Q = Z @ XtWX
    from scipy.linalg import solve_triangular
    Z = solve_triangular(L_H, np.eye(p), lower=True)     # L^{-1}, (p,p)
    Q = Z @ final_stats.XtWX                              # L^{-1} XtWX
    hat_trace = np.sum(Z * Q)  # tr(Z^T Q) = tr(L^{-T} L^{-1} XtWX) = tr(H^{-1} XtWX)

    # Bayesian covariance: Vp = H^{-1} = (L^{-1})^T (L^{-1})
    # We need the full matrix for standard errors / p-values, but
    # compute it from the triangular factor, not from inv(H) directly.
    Vp = Z.T @ Z

    pen_dev_final = final_stats.deviance + float(beta @ S_lambda_dense @ beta)

    return PIRLSResult(
        coefficients=beta,
        fitted_values=None,  # Computed lazily; provider holds data
        linear_predictor=None,
        working_weights=None,
        deviance=final_stats.deviance,
        penalized_deviance=pen_dev_final,
        n_iter=iteration + 1,
        converged=converged,
        hat_matrix_trace=hat_trace,
        Vp=Vp,
        final_stats=final_stats,  # For REML/GCV smoothing param estimation
    )
```

### 7.2 Extended PIRLS (gam.fit5 equivalent)

For extended families, the fitting uses derivatives of the log-likelihood directly.

```python
# fitting/pirls.py (continued)

def extended_pirls_fit(X, y, family: ExtendedFamily, smoothing_penalties,
                       weights=None, offset=None, beta_init=None,
                       max_iter=200, tol=1e-7, backend=None):
    """
    Extended PIRLS for extended families (gam.fit5 equivalent).

    Key difference from standard PIRLS:
    - Working weights = -d²l/dμ² (from autodiff)
    - Working response = η - (dl/dμ) / (d²l/dμ²) (from autodiff)
    - Theta (extra parameters) updated in outer iteration via
      Fellner-Schall or Newton
    - Uses full log-likelihood, not deviance

    With autodiff, this is dramatically simpler than the original
    mgcv implementation because we don't need hand-coded derivatives
    for each family.
    """
    import jax
    # JAX AD used directly (no multi-backend)

    n, p = X.shape
    wt = weights if weights is not None else np.ones(n)
    off = offset if offset is not None else np.zeros(n)

    mu = family.initialize(y, wt)
    eta = family.link.link(mu)
    beta = beta_init if beta_init is not None else np.zeros(p)
    theta = family.theta.copy()

    for iteration in range(max_iter):
        # Compute log-lik derivatives w.r.t. mu via autodiff
        def ll_per_obs(mu_):
            return family.log_likelihood(y, mu_, theta, wt, scale=1.0)

        # Vectorized gradient and Hessian diagonal
        grad_mu = ad.elementwise_grad(ll_per_obs)(mu)
        hess_mu = ad.elementwise_hessian_diag(ll_per_obs)(mu)

        # Working quantities
        W = np.maximum(-hess_mu, 1e-10)  # Must be positive
        z = eta - off + grad_mu / W

        # Now same augmented QR solve as standard PIRLS
        # [√W X; √S_λ] β = [√W z; 0]
        W_sqrt = np.sqrt(W)
        WX = W_sqrt[:, None] * X if not sparse.issparse(X) \
            else sparse.diags(W_sqrt) @ X
        Wz = W_sqrt * z

        # ... (same augmented solve as Section 7.1)

        # Update theta via extended Fellner-Schall
        # (see Section 8.2)

    return PIRLSResult(...)
```

---

## 8. Smoothness Selection: Smoothing Parameter Estimation

### 8.1 REML / ML Criterion — Dual Implementations

**v1.4 fix:** v1.3 showed `reml_criterion` using `np.linalg.cholesky`, `toarray()`, and SciPy sparse operations, then wrapped it in `jax.grad`/`jax.hessian`. JAX cannot trace any of that; the code would silently return wrong gradients or fail at JIT time.

The fix: two completely separate implementations. The JAX path is pure `jax.numpy` and receives only dense `jax.Array` inputs — all SciPy/sparse conversion happens *before* entering the traced function.

```python
# fitting/reml_jax.py — Pure JAX, JIT-able, autodiff-able

import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(6,))
def reml_criterion_jax(log_lambda, XtWX, S_list_dense, beta,
                       pen_deviance, sum_log_w, n):
    """
    REML criterion as a pure JAX function.

    ALL inputs are jax.Array (dense). Conversion from SciPy sparse
    happens OUTSIDE this function, at the caller boundary.

    Inputs (all jax.Array):
      log_lambda:    (n_smooth,) — log smoothing parameters
      XtWX:          (p, p)     — from IterationStatistics
      S_list_dense:  list of (p, p) — penalty matrices, pre-densified
      beta:          (p,)       — current coefficients
      pen_deviance:  scalar     — deviance at current beta
      sum_log_w:     scalar     — Σ log(W_i) for REML constant
      n:             int        — number of observations (static)

    The key insight: REML as a function of log_lambda alone
    (with beta, XtWX, etc. held fixed from the inner PIRLS)
    is a smooth, low-dimensional function. JAX autodiff through
    slogdet and matrix ops is numerically stable here because
    the dimension is p (basis size), not n (data size).
    """
    lambdas = jnp.exp(log_lambda)
    p = XtWX.shape[0]

    # Combined penalty: S_λ = Σ λ_j S_j
    S_lambda = jnp.zeros_like(XtWX)
    for lam, S_j in zip(lambdas, S_list_dense):
        S_lambda = S_lambda + lam * S_j

    # Penalized cross-product
    H = XtWX + S_lambda

    # log|H| via slogdet (numerically stable, JAX-differentiable)
    sign, log_det_H = jnp.linalg.slogdet(H)
    # If sign < 0, H is not positive definite — return large value
    log_det_H = jnp.where(sign > 0, log_det_H, 1e10)

    # log|S_λ^+| — log pseudo-determinant of penalty
    # This is Σ_j (rank_j * log(λ_j)) for well-separated penalties
    # More precisely: log det of the non-zero eigenvalues of S_λ
    # We compute via eigendecomposition of S_λ
    eigs_S = jnp.linalg.eigvalsh(S_lambda)
    # Only count eigenvalues above numerical zero
    threshold = 1e-10 * jnp.max(eigs_S)
    log_det_S = jnp.sum(jnp.where(eigs_S > threshold, jnp.log(eigs_S), 0.0))

    # Penalized objective
    penalty_term = jnp.sum(
        jnp.array([lam * beta @ S_j @ beta
                    for lam, S_j in zip(lambdas, S_list_dense)])
    )
    V = pen_deviance + penalty_term

    # REML = V + log|H| - log|S^+| + const(log weights)
    reml = V + log_det_H - log_det_S
    return reml


# The caller prepares JAX inputs from IterationStatistics:

def _prepare_reml_inputs(pirls_result, penalty_set, provider):
    """
    Bridge from StatisticsProvider world to pure-JAX REML world.
    This is the JAX purity boundary for smoothing parameter estimation.
    """
    import jax.numpy as jnp
    stats = pirls_result.final_stats

    # Convert everything to jax.Array ONCE
    XtWX_jax = jnp.asarray(stats.XtWX)
    beta_jax = jnp.asarray(pirls_result.coefficients)
    S_list_jax = [jnp.asarray(
        S.toarray() if hasattr(S, 'toarray') else S
    ) for S in penalty_set.penalties]

    return XtWX_jax, S_list_jax, beta_jax, stats.deviance, stats.sum_log_weights
```

```python
# fitting/reml_numpy.py — NumPy/SciPy reference (no AD, analytical gradients)

import numpy as np
from scipy import sparse

def reml_criterion_numpy(log_lambda, XtWX, S_list, beta, deviance):
    """
    NumPy reference REML. No autodiff — returns value only.
    Used for testing and for the NumPy fallback backend.
    """
    lambdas = np.exp(log_lambda)
    p = XtWX.shape[0]

    S_lambda = sum(lam * (S.toarray() if sparse.issparse(S) else S)
                   for lam, S in zip(lambdas, S_list))
    H = XtWX + S_lambda

    # Log-det via Cholesky
    try:
        L = np.linalg.cholesky(H + 1e-12 * np.eye(p))
        log_det_H = 2 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        log_det_H = np.inf

    # Penalty pseudo log-det
    eigs = np.linalg.eigvalsh(S_lambda)
    log_det_S = np.sum(np.log(eigs[eigs > 1e-10 * np.max(eigs)]))

    penalty = sum(lam * float(beta @ (S.toarray() if sparse.issparse(S) else S) @ beta)
                  for lam, S in zip(lambdas, S_list))

    return deviance + penalty + log_det_H - log_det_S
```

### 8.2 Outer Newton with Damped Hessian (Smoothing Parameter Optimization)

**v1.4 fix:** v1.3 claimed "trust-region fallback" but showed a bare Newton step with gradient-descent fallback. That fails on badly scaled λ with near-nonidentifiable smooths. Now specified: damped Newton with eigenvalue truncation and explicit acceptance test.

```python
# fitting/smooth_optimize.py

def optimize_smoothing_parameters(provider, penalty_set, family, weights,
                                  method="REML", max_iter=100):
    """
    Outer iteration for smoothing parameters.

    Damped Newton on the REML (or GCV/ML) criterion:
    1. Inner PIRLS gives β̂(λ) and IterationStatistics
    2. Compute REML(log λ) and its gradient/Hessian via JAX AD
    3. Damped Newton step with eigenvalue truncation
    4. Acceptance test: REML must decrease, or increase damping
    5. Convergence: |step| < tol AND |REML change| < tol

    The Hessian eigenvalue truncation prevents catastrophic
    steps when the Hessian has near-zero eigenvalues (common
    when two smoothing parameters are nearly confounded).
    """
    import jax
    import jax.numpy as jnp

    n_smooth = len(penalty_set.penalties)
    log_lambda = jnp.zeros(n_smooth)

    # Build JAX autodiff functions (traced once, reused)
    reml_grad_fn = jax.grad(reml_criterion_jax, argnums=0)
    reml_hess_fn = jax.hessian(reml_criterion_jax, argnums=0)

    damping = 1.0  # Levenberg-Marquardt damping factor
    reml_old = None  # Sentinel: first iteration always accepted

    for outer_iter in range(max_iter):
        # 1. Inner PIRLS at current λ
        S_combined = penalty_set.to_combined(jnp.exp(log_lambda))
        pirls_result = pirls_fit(provider, family, S_combined, weights)

        # 2. Prepare pure-JAX inputs
        XtWX_j, S_list_j, beta_j, dev, slw = _prepare_reml_inputs(
            pirls_result, penalty_set, provider
        )

        # 3. REML value + gradient + Hessian
        reml_val = reml_criterion_jax(
            log_lambda, XtWX_j, S_list_j, beta_j, dev, slw,
            provider.n_observations
        )
        g = reml_grad_fn(
            log_lambda, XtWX_j, S_list_j, beta_j, dev, slw,
            provider.n_observations
        )
        H_reml = reml_hess_fn(
            log_lambda, XtWX_j, S_list_j, beta_j, dev, slw,
            provider.n_observations
        )

        # 4. Damped Newton step with eigenvalue truncation
        # Eigendecompose Hessian, floor small eigenvalues
        eig_vals, eig_vecs = jnp.linalg.eigh(H_reml)
        max_eig = jnp.max(jnp.abs(eig_vals))
        # Floor: no eigenvalue smaller than max/1000
        eig_vals_safe = jnp.maximum(eig_vals, max_eig / 1000.0)
        # Add Levenberg-Marquardt damping
        eig_vals_damped = eig_vals_safe + damping * max_eig

        # Step = -H_damped^{-1} g  (via eigendecomposition)
        step = -eig_vecs @ (
            (eig_vecs.T @ g) / eig_vals_damped
        )

        # Cap step norm to prevent wild jumps
        step_norm = jnp.linalg.norm(step)
        max_step = 5.0  # Max step in log-lambda space
        step = jnp.where(step_norm > max_step, step * max_step / step_norm, step)

        # 5. Acceptance test
        log_lambda_new = log_lambda + step
        # Re-fit PIRLS at proposed λ (cheap inner check)
        S_new = penalty_set.to_combined(jnp.exp(log_lambda_new))
        pirls_new = pirls_fit(provider, family, S_new, weights)
        XtWX_n, S_list_n, beta_n, dev_n, slw_n = _prepare_reml_inputs(
            pirls_new, penalty_set, provider
        )
        reml_new = reml_criterion_jax(
            log_lambda_new, XtWX_n, S_list_n, beta_n, dev_n, slw_n,
            provider.n_observations
        )

        # 5. Acceptance test (first iteration always accepted)
        if outer_iter == 0:
            accept = True
        else:
            accept = float(reml_new) < float(reml_old) - 1e-7 * abs(float(reml_old))

        if accept:
            # Accept step, decrease damping
            log_lambda = log_lambda_new
            damping = max(damping / 2.0, 1e-7)
            reml_old = reml_new
            pirls_result = pirls_new
        else:
            # Reject step, increase damping
            damping = min(damping * 4.0, 1e6)

        # 6. Convergence
        if reml_old is not None:
            reml_change = abs(float(reml_new) - float(reml_old))
            if jnp.max(jnp.abs(step)) < 1e-7 and reml_change < 1e-7:
                break

    return jnp.exp(log_lambda), pirls_result
```

### 8.2 Fellner-Schall Method (Fast REML for bam)

```python
# fitting/fellner_schall.py

def fellner_schall_update(lambda_j, S_j, beta, F_inv, n, p):
    """
    Fellner-Schall update for smoothing parameter λ_j.

    This is the fast update used in bam() and as an alternative
    in gam(). It's a one-step update that avoids computing
    the full Hessian of the REML criterion.

    λ_j^{new} = (p_j / (β^T S_j β)) * λ_j

    where p_j = rank(S_j) - λ_j * tr(F^{-1} S_j)
    is the effective degrees of freedom consumed by penalty j.

    F = X^T W X + S_λ is the penalized Fisher information.
    """
    # tr(F^{-1} S_j) — computed efficiently
    if sparse.issparse(S_j):
        trace_term = np.sum(F_inv * S_j.toarray())
    else:
        trace_term = np.trace(F_inv @ S_j)

    rank_Sj = np.linalg.matrix_rank(
        S_j.toarray() if sparse.issparse(S_j) else S_j
    )
    p_j = rank_Sj - lambda_j * trace_term
    beta_S_beta = beta @ (S_j @ beta)

    # Update
    lambda_new = max(p_j / max(beta_S_beta, 1e-10), 1e-10)
    return lambda_new


def extended_fellner_schall_update(lambda_j, S_j, beta, grad_ll,
                                   hess_ll, theta, family):
    """
    Extended Fellner-Schall for extended families.
    Uses log-likelihood derivatives (from autodiff) instead of
    deviance-based quantities.
    """
    # Similar structure but uses full log-likelihood Hessian
    pass
```

### 8.3 GCV and UBRE Criteria

```python
# fitting/reml.py (continued)

def gcv_criterion(X, y, beta, mu, W, S_lambda, family, n, scale=None):
    """
    Generalized Cross-Validation score.

    GCV = n * D(β̂) / (n - γ * tr(A))²

    where A = X(X^T W X + S_λ)^{-1} X^T W is the hat matrix
    and γ is the GCV inflation factor (default 1.4 in mgcv).
    """
    gamma = 1.4  # mgcv default
    dev = family.deviance(y, mu, np.ones_like(y))
    edf = _hat_matrix_trace(X, W, S_lambda)
    return n * dev / (n - gamma * edf) ** 2


def ubre_criterion(X, y, beta, mu, W, S_lambda, family, n, scale):
    """
    Un-Biased Risk Estimator (for known scale families like binomial).

    UBRE = D(β̂)/n + 2 * scale * tr(A) / n - scale
    """
    dev = family.deviance(y, mu, np.ones_like(y))
    edf = _hat_matrix_trace(X, W, S_lambda)
    return dev / n + 2 * scale * edf / n - scale
```

---

## 9. Automatic Differentiation Strategy

### 9.1 Where Autodiff Helps vs. Hurts

In mgcv, Simon Wood hand-codes derivatives for:
1. Log-likelihood w.r.t. μ for each extended family (dl/dμ, d²l/dμ², d³l/dμ³, d⁴l/dμ⁴)
2. REML/ML criterion w.r.t. log(λ) (first and second derivatives)
3. Derivatives of theta (extra family parameters) w.r.t. the REML criterion
4. Saturated likelihood derivatives for GCV

**Our nuanced approach (revised v1.18):**

| Component | Strategy | Rationale |
|---|---|---|
| Standard family working weights | **Closed-form V(μ)** | AD adds overhead with zero benefit. `W = wt / (V(μ) * g'(μ)²)` is trivially fast and numerically exact. |
| Standard family deviance | **Closed-form** | Same rationale. |
| REML criterion d/d(log λ) | **JAX autodiff** | Small-dimensional (n_smooth params, typically 2-10), numerically benign, extremely tedious to hand-code (~500 lines in mgcv). Best use case for AD. |
| REML criterion d²/d(log λ)² | **JAX autodiff** | Same — the Hessian is even more tedious. |
| Extended family ll derivatives | **JAX autodiff through stable forward pass** | If the forward computation uses numerically stable primitives (`lgamma`, `logsumexp`, clamped inputs, log-space arithmetic), JAX's AD produces stable gradients automatically. This covers NB, Beta, Cox PH, SHASH, ordered categorical, zero-inflated, and all location-scale families. See Section 9.3 for detailed analysis. |
| Tweedie series evaluation | **`jax.custom_jvp`** | The only family where standard AD provably fails. The series evaluation (Wright's generalized Bessel function) involves a truncated sum of terms that individually overflow while the sum converges. Differentiating through the truncation amplifies error. Requires hand-derived derivative of the series. |
| Theta estimation | **JAX autodiff** | Extra params (NB size, Tweedie power) are optimized w.r.t. REML — same small-dimensional, benign case. |
| Location-scale families | **JAX autodiff** | Multi-parameter families (gaulss, shash) have well-conditioned likelihoods in η-space. |

**v1.18: The key insight is that "numerically tricky forward computation" ≠ "numerically tricky derivative."** For most extended families, if the forward log-likelihood is written in a stable way, the derivative is automatically stable because JAX differentiates the *stable computation*, not the mathematical expression. The doc previously conflated these two problems, leading to an overly conservative `custom_jvp` strategy that would have required hand-deriving and maintaining gradients for 6+ families — exactly the error-prone manual work that autodiff is designed to eliminate.

### 9.2 JAX AD Interface (No Multi-Backend)

v1.0 proposed an `ADBackend` Protocol with JAX, PyTensor, and PyTorch implementations. **This is removed.** AD is JAX-only. The NumPy fallback backend uses analytical derivatives exclusively and does not support extended families that lack them.

```python
# autodiff/interface.py

"""
Thin JAX-only AD interface. No multi-backend abstraction.
"""
import jax
import jax.numpy as jnp
from functools import partial

def grad(fn, argnums=0):
    """Gradient of scalar-valued function."""
    return jax.grad(fn, argnums=argnums)

def hessian(fn, argnums=0):
    """Full Hessian of scalar-valued function."""
    return jax.hessian(fn, argnums=argnums)

def hvp(fn, primals, tangents):
    """Hessian-vector product via forward-over-reverse."""
    grad_fn = jax.grad(fn)
    _, hvp_result = jax.jvp(grad_fn, (primals,), (tangents,))
    return hvp_result

@jax.jit
def per_obs_ll_derivatives(ll_single, y, mu, theta):
    """
    Compute per-observation dl/dμ and d²l/dμ² for extended families.

    Uses vmap over a scalar log-likelihood function:
        ll_single(y_i, mu_i, theta) → scalar

    This is O(n) forward passes but each is trivially cheap.
    The vmap compiles to a single vectorized XLA kernel.
    """
    grad_fn = jax.grad(ll_single, argnums=1)
    hess_fn = jax.grad(grad_fn, argnums=1)
    dll = jax.vmap(lambda yi, mi: grad_fn(yi, mi, theta))(y, mu)
    d2ll = jax.vmap(lambda yi, mi: hess_fn(yi, mi, theta))(y, mu)
    return dll, d2ll
```

### 9.3 Extended Family AD Strategy: Stable Forward Pass + Autodiff

**v1.18: Replaces the previous "custom_jvp for all tricky families" strategy.**

The previous design (v1.0–v1.17) treated NB, Tweedie, Cox PH, and SHASH as requiring hand-derived `custom_jvp` rules because "naive AD through `gammaln` differences produces catastrophic cancellation." This was overly conservative. The claim conflates two distinct problems:

1. **Numerically unstable *forward computation*** — writing `gammaln(y+θ) - gammaln(θ)` in a way that loses precision for large θ.
2. **Numerically unstable *derivative*** — the AD system producing bad gradients even when the forward pass is fine.

Problem (1) is real but is a forward-pass concern, solved by writing the log-likelihood using stable primitives. Problem (2) is much rarer — JAX differentiates the *computation graph*, not the mathematical formula. If the computation is stable, the derivative inherits that stability.

**Family-by-family analysis:**

| Family | Forward stability concern | AD through stable forward? | custom_jvp needed? |
|---|---|---|---|
| **NB** | `lgamma(y+θ) - lgamma(θ)` cancels for large θ | ✅ JAX differentiates `lgamma` → `digamma`, which is a stable special function. At large θ where the difference is tiny, NB converges to Poisson anyway — gradient imprecision doesn't affect the fit. | **No** |
| **Beta** | `lgamma(μφ)`, `lgamma((1-μ)φ)` with μ near 0 or 1 | ✅ Edge cases handled by clamping μ in the forward pass. AD through the clamped version is correct (gradient is zero at the clamp, which is the right answer). | **No** |
| **Cox PH** | `log(Σ exp(η_j))` over risk sets can overflow | ✅ Use `jax.scipy.special.logsumexp` — numerically stable by construction. AD through stable `logsumexp` produces stable gradients. | **No** |
| **Ordered categorical** | `log(σ(a) - σ(b))` when a ≈ b | ✅ Write as `log_diff_exp(log_sigmoid(a), log_sigmoid(b))` in log-space. Stable forward → stable gradient. | **No** |
| **SHASH** | `sinh(τ·arcsinh(x) - ε)` overflow for large args | ✅ `jnp.sinh`/`jnp.arcsinh` handle this. Normalizing constant needs log-space computation (forward-pass concern, not AD). | **No** |
| **Tweedie** | Series evaluation (Wright's generalized Bessel function). Individual terms overflow while sum converges. Truncation is data-dependent. | ❌ Differentiating through a truncated `lax.while_loop` where truncation point depends on data amplifies truncation error. The derivative of the series needs its own convergence analysis. | **Yes** |

**The strategy (v1.18):**

1. Write each family's `log_likelihood` using numerically stable JAX primitives (`lgamma`, `logsumexp`, `log_sigmoid`, `jnp.clip`, log-space arithmetic).
2. Let `jax.grad` differentiate it. No `custom_jvp`.
3. Validate against finite differences across the full parameter space, including extreme regions (large θ, μ near boundaries, zero counts, high overdispersion).
4. **Only if step 3 reveals a genuine AD failure** — not a forward-pass issue — add `custom_jvp` for the specific failing function.
5. Currently, only Tweedie's series evaluation requires `custom_jvp`.

```python
# autodiff/tweedie_jvp.py
#
# Tweedie is the ONE family that genuinely needs custom_jvp.
# The series evaluation involves a truncated sum where:
#   - Individual terms can be exp(1000+) before the sum converges
#   - The truncation point depends on y, μ, and power p
#   - Differentiating through the while_loop that computes the
#     truncation amplifies the truncation error
#
# The derivative of the Tweedie density w.r.t. (μ, p) requires
# differentiating the series term-by-term with its own convergence
# analysis, per Dunn & Smyth (2005) Section 4.
#
# ALL OTHER families use standard jax.grad through stable forward passes.

import jax
import jax.numpy as jnp
from jax import custom_jvp


@custom_jvp
def tweedie_loglik_single(y, mu, log_p):
    """
    Tweedie log-density for a single observation.

    Uses the series evaluation approach from Dunn & Smyth (2005).
    The series is computed via jax.lax.while_loop for JIT compatibility.
    """
    p = jax.nn.sigmoid(log_p) + 1  # p ∈ (1, 2)
    # ... series evaluation (see Section 6.3 for full implementation)
    pass


@tweedie_loglik_single.defjvp
def tweedie_ll_jvp(primals, tangents):
    """
    Custom JVP for Tweedie log-density.

    Differentiates the series term-by-term using the recurrence
    relations from Dunn & Smyth (2005) Section 4. The derivative
    series has its own convergence criterion, independent of the
    forward series truncation.

    This is the only custom_jvp in the library. Every other family
    uses standard jax.grad.
    """
    y, mu, log_p = primals
    dy, dmu, dlog_p = tangents
    # ... term-by-term derivative series
    pass
```

**What this changes from v1.0–v1.17:**

| | Previous (v1.0–v1.17) | New (v1.18) |
|---|---|---|
| NB | `custom_jvp` with hand-coded digamma recurrence | `jax.grad` through stable `lgamma`-based forward pass |
| Cox PH | `custom_jvp` planned | `jax.grad` through `logsumexp`-based partial likelihood |
| SHASH | `custom_jvp` planned | `jax.grad` through stable sinh/arcsinh forward pass |
| Beta | Already marked "benign" | Unchanged — `jax.grad` |
| Tweedie | `custom_jvp` | `custom_jvp` (unchanged — genuinely needed) |
| Maintenance burden | Hand-derived gradients for 4+ families | Hand-derived gradient for 1 family |
| Correctness risk | High (each custom rule is a potential sign error) | Low (trust JAX's AD for 5/6 families) |

**Validation contract (unchanged):** Every extended family — whether using standard AD or custom_jvp — must pass the finite-difference validation in Section 9.5 across the full parameter space including extreme regions. The validation is family-agnostic; it doesn't care whether the gradient came from AD or a custom rule.

### 9.4 AD Integration Points (Revised v1.18)

| Component | Method | AD Mode | Notes |
|---|---|---|---|
| Standard family weights W | **Closed-form V(μ)** | N/A | No AD. Direct formula. |
| Extended family ll derivs (all except Tweedie) | **JAX autodiff** | vmap(grad) + vmap(grad(grad)) | NB, Beta, Cox PH, SHASH, ocat, zip, gaulss, gammals — all use stable forward passes |
| Tweedie ll derivs | **`custom_jvp`** | Registered custom rule | Only family needing it — series evaluation truncation |
| REML criterion | JAX autodiff | Reverse + reverse | Best use of AD: small dim, benign |
| Theta estimation | JAX autodiff | Reverse | Same as REML |
| GCV/UBRE | JAX autodiff | Reverse | Alternative to REML |

### 9.5 Validation: AD vs. Finite Differences

Every extended family — whether using standard `jax.grad` or the Tweedie `custom_jvp` — must be validated against finite differences across the full parameter space, including extreme regions. The validation is family-agnostic; it doesn't care how the gradient was computed.

```python
# Test pattern for ALL extended family implementations.
# This same test applies to NB (standard AD), Tweedie (custom_jvp),
# Cox PH (standard AD), etc. The gradient source doesn't matter.

def test_nb_autodiff_matches_finite_diff():
    """Verify jax.grad through NB loglik matches finite differences."""
    from pymgcv.families.negbin import NegativeBinomial
    nb_loglik = NegativeBinomial().loglik_per_obs_fn()

    y, mu, log_theta = 5.0, 3.0, jnp.log(10.0)

    # AD gradient (standard jax.grad — no custom_jvp)
    grad_fn = jax.grad(nb_loglik, argnums=(1, 2))
    ad_grad = grad_fn(y, mu, log_theta)

    # Finite difference gradient
    eps = 1e-7
    fd_grad_mu = (nb_loglik(y, mu + eps, log_theta) -
                  nb_loglik(y, mu - eps, log_theta)) / (2 * eps)
    fd_grad_lt = (nb_loglik(y, mu, log_theta + eps) -
                  nb_loglik(y, mu, log_theta - eps)) / (2 * eps)

    np.testing.assert_allclose(ad_grad[0], fd_grad_mu, rtol=1e-5)
    np.testing.assert_allclose(ad_grad[1], fd_grad_lt, rtol=1e-5)

    # Critical: test at extreme θ where digamma(y+θ)-digamma(θ) is small.
    # v1.18 claim: standard AD handles this fine because JAX's digamma
    # is a well-conditioned special function.
    for large_theta in [100.0, 1000.0, 10000.0]:
        log_theta_large = jnp.log(large_theta)
        ad_g = grad_fn(y, mu, log_theta_large)
        fd_g = _finite_diff(nb_loglik, y, mu, log_theta_large)
        np.testing.assert_allclose(ad_g, fd_g, rtol=1e-4,
            err_msg=f"AD gradient fails at theta={large_theta}")
```

---

## 10. Execution Paths: Dense-GPU, Sparse-CPU, and Chunked

### 10.1 The Sparse/Dense/GPU Triangle Problem (Resolved)

v1.0 proposed transparent sparse/dense switching. This was removed because JAX sparse is experimental and cannot JIT, scipy sparse forces CPU round-trips from GPU, and the "transparent" decision logic was neither good for GPU perf nor good for sparse perf.

**Instead, the library provides three explicit, well-optimized execution paths.** The user can choose explicitly, or an auto-selector picks based on n and p.

| Path | When Used | Inner Loop | Sparse | GPU | Memory |
|---|---|---|---|---|---|
| **Dense-GPU** | n < ~200k, p < ~5k | JAX JIT (fused PIRLS) | No — all dense | Yes | O(n × p) |
| **Sparse-CPU** | n < ~1M, large p or sparse bases | SciPy + CHOLMOD/SPQR | Yes — full sparse pipeline | No | O(nnz + p²) |
| **Chunked-Hybrid** | n > ~1M (bam) | Per-chunk: Dense-GPU; accumulate: CPU | Penalty only | Chunks on GPU | O(p² + chunk × p) |

```python
# linalg/execution_path.py

import enum

class ExecutionPath(enum.Enum):
    DENSE_GPU = "dense_gpu"
    SPARSE_CPU = "sparse_cpu"
    CHUNKED_HYBRID = "chunked_hybrid"

def auto_select_path(n: int, p: int, has_sparse_basis: bool,
                     gpu_available: bool,
                     penalty_set: "CompositePenalty | None" = None) -> ExecutionPath:
    """
    Auto-select execution path based on problem size, hardware,
    and penalty structure.

    Decision tree:
    1. If n > 1M → CHUNKED_HYBRID (memory bounded)
    2. If penalty_set.has_large_penalty() → SPARSE_CPU
       (GPU can't hold large MRF/spatial penalties; route early
       instead of hitting ValueError in to_dense later)
    3. If n > 200k OR has_sparse_basis → SPARSE_CPU
       (GPU memory can't hold n×p dense; sparse solvers more efficient)
    4. If gpu_available → DENSE_GPU (maximum throughput)
    5. Else → SPARSE_CPU (CPU dense is slower than sparse for any size)
    """
    if n > 1_000_000:
        return ExecutionPath.CHUNKED_HYBRID
    if penalty_set is not None and penalty_set.has_large_penalty():
        return ExecutionPath.SPARSE_CPU
    if n > 200_000 or (has_sparse_basis and p > 500):
        return ExecutionPath.SPARSE_CPU
    if gpu_available:
        return ExecutionPath.DENSE_GPU
    return ExecutionPath.SPARSE_CPU
```

### 10.2 Structured Penalty Representations

**v1.4 fix:** v1.3's `PenaltySet.to_dense_jax()` built a full `(p, p)` dense matrix via `jnp.zeros((p, p))` + `jnp.diag(...)`, making the "structured penalty" narrative fiction. The solver consumed only a materialized matrix, so diagonal/Kronecker structure provided zero benefit.

**The fix:** Penalties are represented as structured linear operators. The solver interface accepts `.matvec(beta)` and `.quadform(beta)` — never a materialized matrix. Dense fallback exists but is opt-in, not default.

```python
# penalties/structured.py

from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse


class StructuredPenalty(ABC):
    """
    Abstract penalty that exposes operator semantics, not a matrix.

    The PIRLS solver needs exactly two operations:
      1. S_λ @ beta  (for forming H @ beta = XtWX @ beta + S_λ @ beta)
      2. beta^T S_λ beta  (for penalized deviance)

    The REML criterion needs additionally:
      3. log|S_λ^+|  (log pseudo-determinant)

    None of these require materializing S as a dense (p, p) matrix
    if the structure is known.

    JAX PyTree contract (v1.6):

    All StructuredPenalty subclasses must be registered as JAX PyTrees
    so they can be passed into JIT-compiled functions. This means:
    - "Children" (dynamic, jax.Array): d, lam, S, A, B — anything
      that varies during optimization (e.g., lam changes with λ updates)
    - "Aux data" (static, hashable): col_start, col_end, total_p, shape info
    - All methods used inside JIT must be pure functions of children + aux

    Subclasses implement tree_flatten() and tree_unflatten() and register
    via jax.tree_util.register_pytree_class(). Example for DiagonalPenalty:

        def tree_flatten(self):
            children = (self.d, self.lam)  # jax.Array / float
            aux = (self._dim(),)            # static int
            return children, aux

        @classmethod
        def tree_unflatten(cls, aux, children):
            d, lam = children
            obj = cls.__new__(cls)
            obj.d = d; obj.lam = lam
            return obj

    Without this, passing a StructuredPenalty into a @jax.jit function
    will either fail ("not a valid JAX type") or silently treat the
    object as a static constant (recompiling on every call).
    """

    @abstractmethod
    def matvec(self, beta: jnp.ndarray) -> jnp.ndarray:
        """S_λ @ beta without forming S. O(p) for diagonal, O(p√p) for Kronecker."""
        ...

    @abstractmethod
    def quadform(self, beta: jnp.ndarray) -> float:
        """beta^T S_λ beta. O(p) for diagonal."""
        ...

    @abstractmethod
    def log_pseudo_det(self) -> float:
        """log|S_λ^+| = sum of log of nonzero eigenvalues."""
        ...

    @abstractmethod
    def add_to_dense(self, H: jnp.ndarray) -> jnp.ndarray:
        """H + S_λ → H. Used when XtWX is already dense (p×p).
        For diagonal: O(p). For dense: O(p²). Never allocates a new (p,p)."""
        ...

    @abstractmethod
    def rank(self) -> int:
        """Number of nonzero eigenvalues (penalty rank)."""
        ...

    def to_dense(self) -> jnp.ndarray:
        """Explicit dense materialization. Use only for debugging/testing."""
        p = self._dim()
        return jax.vmap(self.matvec)(jnp.eye(p))

    @abstractmethod
    def _dim(self) -> int: ...


class DiagonalPenalty(StructuredPenalty):
    """
    S_λ = λ * diag(d) where d is a 1D vector.

    This is the most common case after TPRS eigen-reparameterization.
    ALL operations are O(p), zero allocation beyond the vector.
    """

    def __init__(self, diag_values: jnp.ndarray, lam: float):
        self.d = diag_values  # (p,) or (n_penalized,) with zeros for null space
        self.lam = lam

    def matvec(self, beta):
        return self.lam * self.d * beta  # O(p), elementwise

    def quadform(self, beta):
        return self.lam * jnp.sum(self.d * beta ** 2)  # O(p)

    def log_pseudo_det(self):
        nonzero = self.d > 1e-10 * jnp.max(self.d)
        return jnp.sum(jnp.where(nonzero, jnp.log(self.lam * self.d), 0.0))

    def add_to_dense(self, H):
        # Add to diagonal in-place semantics: O(p)
        idx = jnp.arange(H.shape[0])
        return H.at[idx, idx].add(self.lam * self.d)

    def rank(self):
        return int(jnp.sum(self.d > 1e-10 * jnp.max(self.d)))

    def _dim(self):
        return len(self.d)


class DensePenalty(StructuredPenalty):
    """
    S_λ = λ * S where S is a dense (p_j, p_j) matrix.

    Used for GP smooths, small general penalties. p_j is typically
    small (< 200 per term), so the global penalty is block-diagonal
    with dense blocks embedded in a larger sparse structure.
    """

    def __init__(self, matrix: jnp.ndarray, lam: float,
                 col_start: int, total_p: int):
        self.S = matrix           # (p_j, p_j)
        self.lam = lam
        self.col_start = col_start
        self.col_end = col_start + matrix.shape[0]
        self.total_p = total_p

    def matvec(self, beta):
        result = jnp.zeros_like(beta)
        block = beta[self.col_start:self.col_end]
        result = result.at[self.col_start:self.col_end].set(
            self.lam * self.S @ block
        )
        return result

    def quadform(self, beta):
        block = beta[self.col_start:self.col_end]
        return self.lam * block @ self.S @ block

    def log_pseudo_det(self):
        eigs = jnp.linalg.eigvalsh(self.lam * self.S)
        nonzero = eigs > 1e-10 * jnp.max(eigs)
        return jnp.sum(jnp.where(nonzero, jnp.log(eigs), 0.0))

    def add_to_dense(self, H):
        slc = slice(self.col_start, self.col_end)
        return H.at[slc, slc].add(self.lam * self.S)

    def rank(self):
        eigs = jnp.linalg.eigvalsh(self.S)
        return int(jnp.sum(eigs > 1e-10 * jnp.max(eigs)))

    def _dim(self):
        return self.total_p


class KroneckerPenalty(StructuredPenalty):
    """
    S_λ = λ * (A ⊗ B) where A is (m, m) and B is (k, k).

    Arises from tensor product smooths: te(x1, x2).
    matvec: O(mk + mk) instead of O(m²k²).
    Never forms the (mk, mk) product.

    Uses the vec-permutation identity:
      (A ⊗ B) vec(X) = vec(B X A^T)
    where beta is reshaped as X of shape (k, m).
    """

    def __init__(self, A: jnp.ndarray, B: jnp.ndarray, lam: float,
                 col_start: int, total_p: int):
        self.A = A  # (m, m)
        self.B = B  # (k, k)
        self.lam = lam
        self.m = A.shape[0]
        self.k = B.shape[0]
        self.col_start = col_start
        self.col_end = col_start + self.m * self.k
        self.total_p = total_p

    def matvec(self, beta):
        result = jnp.zeros_like(beta)
        block = beta[self.col_start:self.col_end]
        X = block.reshape(self.k, self.m)      # Reshape to matrix
        Y = self.B @ X @ self.A.T              # (k,m) — two small matmuls
        result = result.at[self.col_start:self.col_end].set(
            self.lam * Y.ravel()
        )
        return result

    def quadform(self, beta):
        block = beta[self.col_start:self.col_end]
        return self.lam * block @ self.matvec_block(block)

    def matvec_block(self, block):
        """Apply (A ⊗ B) to a block without global zero-padding."""
        X = block.reshape(self.k, self.m)
        return (self.B @ X @ self.A.T).ravel()

    def log_pseudo_det(self):
        eigs_A = jnp.linalg.eigvalsh(self.A)
        eigs_B = jnp.linalg.eigvalsh(self.B)
        # Eigenvalues of A⊗B are all products eig_A[i] * eig_B[j]
        all_eigs = self.lam * jnp.outer(eigs_A, eigs_B).ravel()
        nonzero = all_eigs > 1e-10 * jnp.max(all_eigs)
        return jnp.sum(jnp.where(nonzero, jnp.log(all_eigs), 0.0))

    def add_to_dense(self, H):
        slc = slice(self.col_start, self.col_end)
        kron_dense = self.lam * jnp.kron(self.A, self.B)  # Only when forced
        return H.at[slc, slc].add(kron_dense)

    def rank(self):
        rank_A = int(jnp.sum(jnp.linalg.eigvalsh(self.A) > 1e-10))
        rank_B = int(jnp.sum(jnp.linalg.eigvalsh(self.B) > 1e-10))
        return rank_A * rank_B

    def _dim(self):
        return self.total_p


class CompositePenalty:
    """
    S_λ = Σ_j S_j  where each S_j is a StructuredPenalty.

    This is the top-level object the solver receives.
    All operations delegate to individual terms.
    """

    def __init__(self, terms: list[StructuredPenalty]):
        self.terms = terms

    def matvec(self, beta: jnp.ndarray) -> jnp.ndarray:
        """S_λ @ beta = Σ_j S_j @ beta. Each term is O(p_j) or O(p_j²)."""
        result = jnp.zeros_like(beta)
        for term in self.terms:
            result = result + term.matvec(beta)
        return result

    def quadform(self, beta: jnp.ndarray) -> float:
        """beta^T S_λ beta = Σ_j beta^T S_j beta."""
        return sum(term.quadform(beta) for term in self.terms)

    def log_pseudo_det(self) -> float:
        """
        log|S_λ^+| for the composite penalty.

        When penalties have non-overlapping column supports (typical):
        log|S^+| = Σ_j log|S_j^+|. When they overlap (rare, e.g.,
        double-penalty models): falls back to eigendecomposition
        of the materialized matrix.
        """
        # Check if column supports are disjoint
        if self._supports_disjoint():
            return sum(term.log_pseudo_det() for term in self.terms)
        else:
            # Fallback: materialize and eigendecompose
            S_dense = self.to_dense()
            eigs = jnp.linalg.eigvalsh(S_dense)
            nonzero = eigs > 1e-10 * jnp.max(eigs)
            return jnp.sum(jnp.where(nonzero, jnp.log(eigs), 0.0))

    def add_to_dense(self, H: jnp.ndarray) -> jnp.ndarray:
        """Add all penalties to an existing dense H, in-place semantics."""
        for term in self.terms:
            H = term.add_to_dense(H)
        return H

    def to_dense(self) -> jnp.ndarray:
        """Explicit fallback. Avoid in production."""
        p = self.terms[0]._dim()
        S = jnp.zeros((p, p))
        for term in self.terms:
            S = term.add_to_dense(S)
        return S

    def has_large_penalty(self, threshold: int = 10000) -> bool:
        """Check if any term would blow memory on GPU."""
        for term in self.terms:
            if isinstance(term, DensePenalty):
                if term.S.shape[0] > threshold:
                    return True
        return False

    def _supports_disjoint(self) -> bool:
        """Check if penalty column supports are non-overlapping."""
        ranges = []
        for t in self.terms:
            if hasattr(t, 'col_start'):
                ranges.append((t.col_start, t.col_end))
            else:
                return False  # Can't determine; assume overlap
        ranges.sort()
        for i in range(len(ranges) - 1):
            if ranges[i][1] > ranges[i + 1][0]:
                return False
        return True

    def to_sparse_scipy(self):
        """Convert to scipy.sparse for the Sparse-CPU path."""
        p = self.terms[0]._dim()
        S = sparse.csc_matrix((p, p))
        for term in self.terms:
            S_term = np.asarray(term.to_dense())  # Per-term is small
            S = S + sparse.csc_matrix(S_term)
        return S

    def update_lambdas(self, new_lambdas: jnp.ndarray) -> "CompositePenalty":
        """Return new CompositePenalty with updated smoothing parameters."""
        new_terms = []
        for term, lam in zip(self.terms, new_lambdas):
            # Create new term with updated lambda
            import copy
            t = copy.copy(term)
            t.lam = float(lam)
            new_terms.append(t)
        return CompositePenalty(new_terms)
```

**Log-det / trace capability matrix (v1.5):**

Not all penalty structures can provide exact `log_pseudo_det()`. The REML criterion requires this quantity, so the availability determines which execution path is used for smoothing parameter estimation:

| Penalty type | `matvec` | `quadform` | `log_pseudo_det` | `add_to_dense` | REML path |
|---|---|---|---|---|---|
| `DiagonalPenalty` | O(p) exact | O(p) exact | O(p) exact — sum of log(d_i) | O(p) | Dense-GPU or Sparse-CPU |
| `DensePenalty` (small block, p_j < 500) | O(p_j²) | O(p_j²) | O(p_j³) exact — eigvalsh of block | O(p_j²) | Dense-GPU or Sparse-CPU |
| `KroneckerPenalty` | O(mk) exact | O(mk) exact | O(m³+k³) exact — eigenvalues of factors | O(m²k²) reluctantly | Dense-GPU or Sparse-CPU |
| MRF Laplacian (large, sparse) | O(nnz) via sparse matvec | O(nnz) | **NOT exact** — routes to Sparse-CPU with `CHOLMOD.logdet()` | N/A (too large) | Sparse-CPU only |
| Soap film / Duchon | O(p²) dense | O(p²) dense | O(p³) exact (dense block) | O(p²) | Dense-GPU if p small; Sparse-CPU otherwise |
| Adaptive (spatially varying λ) | Depends on base | Depends on base | **Approximated** — stochastic trace est. for REML gradient | Depends | Sparse-CPU + Fellner-Schall |

**Routing rule:** If any penalty in the `CompositePenalty` cannot provide exact `log_pseudo_det()`, the REML optimizer either (a) routes to Sparse-CPU where `CHOLMOD` can compute log-det of the full `H` directly, or (b) uses the Fellner-Schall update (Section 8.3) which requires only traces, not log-dets. This is declared per penalty via:

```python
class StructuredPenalty(ABC):
    # ... existing methods ...

    @property
    def supports_exact_logdet(self) -> bool:
        """Whether log_pseudo_det() is exact (not approximated)."""
        return True  # Override to False for MRF, adaptive, etc.
```

**How the solver uses this (Dense-GPU path):**

The PIRLS loop never builds `S_lambda` as a `(p, p)` matrix. Instead:

```python
# In PIRLS inner loop (Dense-GPU path):

# Form H = XtWX + S_λ by adding structured penalties to dense XtWX
H = penalty.add_to_dense(stats.XtWX)   # O(Σ p_j²), NOT O(p²)

# Penalized deviance uses quadform
pen_dev = dev + penalty.quadform(beta)  # O(p) for all-diagonal

# REML log-det uses structured log_pseudo_det
log_det_S = penalty.log_pseudo_det()    # O(Σ p_j³) per-block, not O(p³)
```

For diagonal-only penalties (TPRS after reparameterization), the entire penalty contribution is `O(p)` per iteration — no `(p, p)` allocation at all. For Kronecker penalties from `te()`, `matvec` costs `O(mk + mk)` instead of `O(m²k²)`.

**When dense materialization is unavoidable:**

The `add_to_dense(XtWX)` call does modify a `(p, p)` matrix — but that matrix (`XtWX`) *already exists* as dense because the Dense-GPU path computes `X.T @ W @ X` densely. The penalty is being added *into an existing dense matrix*, not creating a new one. The savings are:

1. No separate `(p, p)` penalty allocation
2. Diagonal penalties add to diagonal only: `O(p)` not `O(p²)`
3. `quadform` and `log_pseudo_det` never touch `(p, p)` at all

### 10.3 Dense-GPU Path

Everything is a dense JAX array on the GPU device. The entire PIRLS iteration is a single JIT-compiled function — no CPU/GPU round-trips.

**Solver strategy (v1.5):**

The core solve at each PIRLS iteration is `H β = g` where `H = XtWX + S_λ` is `(p, p)` symmetric positive definite (by construction, since `S_λ` is positive semi-definite and `XtWX` is PSD for sufficient data).

| Step | Method | Cost | When |
|---|---|---|---|
| Primary | `jnp.linalg.cholesky(H + ε_s·I)` → two triangular solves, where `ε_s = 1e-12 · tr(H)/p` (scale-relative) | O(p³/3) | Default. H is SPD by construction. |
| Fallback 1 | Add `ε_L·I` where `ε_L = 1e-6 · tr(H)/p` (scale-relative) and re-try Cholesky. **Records `ε_L` in `FitDiagnostics.regularization_applied`** for surfacing in `summary()`. | O(p³/3) | `LinAlgError` from near-singular H |
| Fallback 2 | `jnp.linalg.lstsq(H, g)` (SVD-based) | O(p³) | Double `LinAlgError`. Logs warning. |
| Bailout | Warn to re-route to Sparse-CPU | — | After 2+ instability events (Cholesky failure, NaN in β, step-halving exhaustion) across any iterations. Zero-cost detection — these are the natural failure modes, not a separate estimator. |

**Operating envelope constraints:**

| Parameter | Limit | Rationale |
|---|---|---|
| p (basis dimension) | ≤ 5,000 | p³ ≈ 125G FLOPs; Cholesky takes ~0.5s on A100 |
| n × p (design matrix) | ≤ 1G elements | 8GB at float64; fits in 40GB GPU memory |
| Iteration count | Typically 5–15 | Gaussian: 1 iter; Binomial boundary: up to 50 |

The auto-selector enforces these limits. If exceeded, routes to Chunked-Hybrid or Sparse-CPU.

**What the GPU actually does per iteration:**

1. `eta = X @ beta + offset` — GEMV, O(np)
2. `mu, W, z` from family — elementwise, O(n) via `jax.vmap`
3. `WX = sqrt(W)[:,None] * X` — broadcasting, O(np) *memory bandwidth bound*
4. `XtWX = WX.T @ WX` — GEMM, O(np²) *compute bound, the expensive step*
5. `XtWz = WX.T @ (sqrt(W) * z)` — GEMV, O(np)
6. `H = XtWX + S_λ` — penalty.add_to_dense, O(p) to O(p²)
7. `L = cholesky(H); β = solve(L, solve(L, g))` — O(p³/3)
8. Step-halving: repeat steps 1–2 with trial β, O(n) each

Steps 3–4 dominate. Total per-iteration: O(np² + p³). For n=100k, p=1000: ~100G FLOPs. Roofline best-case on A100 is O(10ms) per iteration, but real end-to-end cost will be higher due to kernel launch overhead, HBM bandwidth limits on XtWX formation, XLA graph boundaries, and non-fused operations. Use as order-of-magnitude planning number, not a benchmark.

```python
# StatisticsProvider implementation:
class DenseGPUProvider(InMemoryProvider):
    """All arrays on GPU as dense jax.Array."""

    def __init__(self, X, y, offset=None):
        import jax
        super().__init__(
            jax.device_put(X),
            jax.device_put(y),
            jax.device_put(offset) if offset is not None else None,
        )
```

### 10.4 Sparse-CPU Path

Penalty matrices, random effects design matrices, and banded basis matrices stay sparse. Uses SciPy's sparse linear algebra throughout.

**Dependency chain and degraded mode (v1.6, revised v1.16):**

The full Sparse-CPU path uses SuiteSparse CHOLMOD/SPQR via `scikit-sparse`. This is not a trivial dependency — it requires compiled C libraries and has ABI compatibility issues across OS/architecture combinations. Since Sparse-CPU is a first-class execution path, we need a story for when CHOLMOD is unavailable:

**Install strategy (v1.16, uv-based):** The install hierarchy uses uv's index and lockfile infrastructure (Section 3.1):

1. **`uv sync --extra sparse` (preferred):** Installs scikit-sparse from the `pymgcv-wheels` index, which hosts pre-built wheels with statically-linked SuiteSparse. No C compiler needed. If no pre-built wheel exists for the user's platform, install fails at install time with a clear error.
2. **`conda install pymgcv` (conda-forge):** Links against conda-forge's SuiteSparse package, which is well-maintained and ABI-stable within conda environments.
3. **System CHOLMOD (fallback):** If installed via plain `pip` without uv, falls back to whatever `scikit-sparse` can find on the system. This is the fragile path and is not the recommended install.
4. **Degraded mode:** Dense fallback with hard gates (below). No silent OOM.

**Docker image:** `ghcr.io/pymgcv/pymgcv:latest` runs `uv sync --extra full --frozen` and ships with CHOLMOD, JAX with CUDA, and all dependencies. Recommended for production.

**"Degraded mode will become the default for 80% of users" risk:** The uv + pre-built wheel strategy prevents this. `uv sync --extra sparse` either succeeds (pre-built wheel) or fails clearly at install time. Runtime degraded mode only triggers if the user deliberately installed without the `sparse` extra. Error messages reference `uv sync --extra sparse` as the fix.

| Capability | With CHOLMOD (`scikit-sparse`) | Without CHOLMOD (degraded: p ≤ 1500 AND n×p×8 ≤ 500MB) |
|---|---|---|
| Sparse Cholesky | `sksparse.cholmod.cholesky(H)` — O(nnz) fill-in | `scipy.linalg.cho_factor(H.toarray())` — O(p³), dense |
| Sparse log-det | `factor.logdet()` — O(nnz) | `2 * sum(log(diag(L)))` from dense Cholesky — O(p³) |
| Sparse QR | SPQR via `scipy.sparse.linalg` | `scipy.linalg.qr(X.toarray())` — O(n×p²) |
| p > 1500 OR n×p×8 > 500MB | Full performance | **Hard error with install instructions** — no silent OOM |
| Peak memory (in envelope) | O(nnz) | ~3×p²×8 bytes (H) + n×p×8 bytes (X if QR fallback) |
| Correctness | Full | Same objective within MODERATE tolerance vs CHOLMOD path (dense cho_factor may differ in pivoting/fill-in ordering) |

```python
# linalg/sparse_solve.py

_CHOLMOD_AVAILABLE = None  # Lazy check

def _check_cholmod():
    global _CHOLMOD_AVAILABLE
    if _CHOLMOD_AVAILABLE is None:
        try:
            from sksparse.cholmod import cholesky  # noqa: F401
            _CHOLMOD_AVAILABLE = True
        except ImportError:
            _CHOLMOD_AVAILABLE = False
    return _CHOLMOD_AVAILABLE


def sparse_cholesky(H_sparse, n_obs=None):
    """
    Sparse Cholesky with CHOLMOD, or fail-fast degraded mode.

    v1.9: Two gates for degraded mode safety:
    1. p ≤ 1500 (H densification: ~54MB for H + factor + temps)
    2. n_obs × p × 8 ≤ 500MB (X densification budget, guards against
       downstream QR fallback calling X.toarray())

    If either gate fails → hard error with install instructions.
    If both pass → dense fallback with warning.
    """
    if _check_cholmod():
        from sksparse.cholmod import cholesky
        return cholesky(H_sparse)

    # No CHOLMOD — check if dense fallback is safe
    p = H_sparse.shape[0]
    SAFE_DENSE_P = 1500
    SAFE_X_BYTES = 500 * 1024 * 1024  # 500MB

    # Gate 1: p dimension (H + factor + temporaries)
    h_budget = 3 * p * p * 8
    if p > SAFE_DENSE_P:
        raise ImportError(
            f"scikit-sparse (CHOLMOD) required for Sparse-CPU with "
            f"p={p} (>{SAFE_DENSE_P}). Dense fallback: ~{h_budget/1e9:.2f}GB.\n"
            f"Install: uv sync --extra sparse\n"
            f"Or use execution_path='dense_gpu' if p ≤ 5000 and GPU available."
        )

    # Gate 2: n×p dimension (guards against X.toarray() downstream)
    if n_obs is not None:
        x_budget = n_obs * p * 8
        if x_budget > SAFE_X_BYTES:
            raise ImportError(
                f"scikit-sparse (CHOLMOD) required: X densification "
                f"~{x_budget/1e9:.1f}GB (n={n_obs}, p={p}, "
                f"budget={SAFE_X_BYTES//(1024**2)}MB).\n"
                f"Install: uv sync --extra sparse"
            )

    import warnings
    warnings.warn(
        f"scikit-sparse not installed; using dense fallback "
        f"(p={p}, H budget ~{h_budget/1e6:.0f}MB). "
        f"Install with: uv sync --extra sparse",
        stacklevel=2,
    )
    from scipy.linalg import cho_factor
    H_dense = H_sparse.toarray()
    L, lower = cho_factor(H_dense)
    return _DenseCholFallback(L, lower)
```

```python
# The workhorse for large problems.
# Uses CHOLMOD for sparse Cholesky (via scikit-sparse)
# and SPQR for sparse QR (via scipy).

class SparseCPUProvider(InMemoryProvider):
    """Sparse basis matrices, scipy sparse solvers on CPU."""

    def compute_iteration_stats(self, beta, family, wt):
        from scipy import sparse
        eta = self.X @ beta + self.offset
        mu = family.link.inverse(eta)
        W = family.working_weights(mu, wt)
        z = family.working_response(self.y, mu, eta - self.offset)

        W = np.clip(W, 1e-10, 1e10)
        # For sparse X, use sparse operations throughout
        if sparse.issparse(self.X):
            W_diag = sparse.diags(W)
            XtWX = (self.X.T @ W_diag @ self.X).toarray()  # p×p is small
            XtWz = self.X.T @ (W * z)
        else:
            W_sqrt = np.sqrt(W)
            WX = W_sqrt[:, None] * self.X
            XtWX = WX.T @ WX
            XtWz = WX.T @ (W_sqrt * z)

        dev = family.deviance(self.y, mu, wt)
        ll = -0.5 * dev  # Default; overridden for extended families
        return IterationStatistics(
            XtWX=XtWX, XtWz=XtWz, deviance=dev,
            log_likelihood=ll, n_obs=self._n,
            sum_log_weights=np.sum(np.log(np.maximum(W, 1e-300))),
        )
```

### 10.5 Chunked-Hybrid Path (bam)

For n > ~1M, data is processed in chunks. Each chunk can optionally use GPU for the `WX.T @ WX` multiply, with accumulation on CPU.

**Memory invariant (v1.15):** `bam()` must never allocate a dense `(n, p)` matrix for the full dataset. The O(p² + chunk_size × p) claim holds only if:

1. `X` is stored as a sparse matrix (CSC/CSR) or memory-mapped on disk — never densified for full n.
2. Per-chunk processing creates at most `WX_c` (chunk_size × p) and `chunk_XtWX` (p × p) temporaries, freed after accumulation.
3. No code path downstream of `ChunkedProvider` calls `X.toarray()` or equivalent.

The accumulator footprint is: `XtWX` (p × p, 8 bytes) + `XtWz` (p, 8 bytes) + one chunk of `WX_c` (chunk_size × p, 8 bytes) + scalar accumulators. For p=2000 and chunk_size=10000: 32MB + 160MB = ~192MB, independent of n. If any future code violates invariant #3, the bam path loses its memory guarantee and degrades to O(n × p), which defeats the purpose.

```python
class ChunkedProvider:
    """
    Process data in chunks for memory-bounded XtWX accumulation.

    Memory: O(p² + chunk_size × p) regardless of n.
    """

    def __init__(self, X, y, offset=None, chunk_size=10000, use_gpu_chunks=False):
        self.X = X
        self.y = y
        self.offset = offset or np.zeros(len(y))
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu_chunks
        self._n, self._p = X.shape

    def compute_iteration_stats(self, beta, family, wt):
        p = self._p
        XtWX = np.zeros((p, p))
        XtWz = np.zeros(p)
        total_dev = 0.0
        total_ll = 0.0
        total_sum_log_w = 0.0

        for start in range(0, self._n, self.chunk_size):
            end = min(start + self.chunk_size, self._n)
            X_c = self.X[start:end]
            y_c = self.y[start:end]
            off_c = self.offset[start:end]
            wt_c = wt[start:end]

            if self.use_gpu:
                import jax
                X_c = jax.device_put(X_c)

            eta_c = X_c @ beta + off_c
            mu_c = family.link.inverse(eta_c)
            W_c = family.working_weights(mu_c, wt_c)
            z_c = family.working_response(y_c, mu_c, eta_c - off_c)

            W_c = np.clip(W_c, 1e-10, 1e10)
            W_sqrt_c = np.sqrt(W_c)
            WX_c = W_sqrt_c[:, None] * X_c

            # Accumulate p×p statistics (back on CPU if GPU chunk)
            chunk_XtWX = np.asarray(WX_c.T @ WX_c)
            chunk_XtWz = np.asarray(WX_c.T @ (W_sqrt_c * z_c))
            XtWX += chunk_XtWX
            XtWz += chunk_XtWz

            total_dev += family.deviance(y_c, mu_c, wt_c)
            total_sum_log_w += np.sum(np.log(np.maximum(W_c, 1e-300)))

        return IterationStatistics(
            XtWX=XtWX, XtWz=XtWz, deviance=total_dev,
            log_likelihood=-0.5 * total_dev, n_obs=self._n,
            sum_log_weights=total_sum_log_w,
        )

    def compute_deviance(self, beta, family, wt):
        """Lightweight chunked deviance for step halving."""
        total_dev = 0.0
        for start in range(0, self._n, self.chunk_size):
            end = min(start + self.chunk_size, self._n)
            eta_c = self.X[start:end] @ beta + self.offset[start:end]
            mu_c = family.link.inverse(eta_c)
            total_dev += family.deviance(self.y[start:end], mu_c, wt[start:end])
        return total_dev

    @property
    def n_observations(self): return self._n
    @property
    def n_parameters(self): return self._p
```

### 10.6 Mid-Fit Path Transfer (v1.7, revised v1.8)

When the Dense-GPU path detects instability (Section 10.3 bailout), the user may warm-start Sparse-CPU from the current state. This is the highest-risk correctness seam in the library — it's where algebra implementations meet.

**Transfer state:**

```python
@dataclass(frozen=True)
class PathTransferState:
    """
    Minimal state for transferring a fit between execution paths.

    REPRESENTATION INVARIANTS (checked by validate()):

    INV-1: beta is float64 np.ndarray, shape (p,), finite, on CPU.
           No JAX DeviceArray, no float32, no NaN/Inf.

    INV-2: log_lambda is float64 np.ndarray, shape (n_smooth,), finite.
           Values are LOG smoothing params (not raw λ).
           Ordering matches CoefficientMap.smooth_term_order.

    INV-3: pen_deviance is finite float64. If source path's last
           iteration was rejected (beta unchanged), pen_deviance
           reflects the ACCEPTED state, not the rejected proposal.

    INV-4: iteration + outer_iteration are non-negative ints.
           iteration is PIRLS (inner) count; outer_iteration is
           smoothing-param (Newton/Fellner-Schall) count.

    INV-5: Penalty structure is compatible across paths. Specifically:
           - Number of penalty terms == n_smooth (same for all paths)
           - Penalty column ranges match CoefficientMap (path-independent)
           - Sign convention: S_j is positive semi-definite (all paths)
    """
    beta: np.ndarray           # (p,) — current coefficients
    log_lambda: np.ndarray     # (n_smooth,) — log smoothing params
    iteration: int             # PIRLS (inner) iteration count
    outer_iteration: int       # Smoothing param (outer) iteration count
    pen_deviance: float        # Last ACCEPTED penalized deviance

    def validate(self, coefficient_map):
        """
        Verify all representation invariants. Called at creation
        AND at consumption. Raises ValueError on any violation.
        """
        # INV-1
        assert isinstance(self.beta, np.ndarray), "beta must be np.ndarray"
        assert self.beta.dtype == np.float64, f"beta dtype {self.beta.dtype} != float64"
        assert self.beta.shape == (coefficient_map.total_p,), \
            f"beta shape {self.beta.shape} != ({coefficient_map.total_p},)"
        assert np.all(np.isfinite(self.beta)), "beta contains NaN/Inf"

        # INV-2
        n_smooth = len(coefficient_map.smooth_terms)
        assert self.log_lambda.shape == (n_smooth,), \
            f"log_lambda shape {self.log_lambda.shape} != ({n_smooth},)"
        assert np.all(np.isfinite(self.log_lambda)), "log_lambda contains NaN/Inf"

        # INV-3
        assert np.isfinite(self.pen_deviance), "pen_deviance is not finite"

        # INV-4
        assert self.iteration >= 0 and self.outer_iteration >= 0

    # NOT transferred (path-specific, rebuilt by target):
    # - Working weights W, pseudo-response z, μ, η
    #   → Recomputed from beta on first target iteration
    # - Cholesky factors / solver state
    #   → Target path builds its own
    # - XtWX / XtWz
    #   → Recomputed by target provider from beta
    # - JAX DeviceArrays
    #   → Target may be CPU-only
```

**What's recomputed vs. carried:**

| Quantity | Carried? | Why |
|---|---|---|
| β (coefficients) | ✅ Yes | Core state, path-independent |
| log(λ) (smoothing params) | ✅ Yes | Outer-loop state, path-independent |
| pen_deviance | ✅ Yes | Needed for convergence check continuity |
| iteration counts | ✅ Yes | Convergence check needs iteration history |
| W, z, μ, η (working quantities) | ❌ No | Recomputed from β — these are deterministic given β + family + data |
| XtWX, XtWz | ❌ No | Recomputed by target provider (different sparse/dense representation) |
| Cholesky factors | ❌ No | Path-specific solver state |
| Penalty matrices | ❌ No | Rebuilt in target format from shared `CoefficientMap` + λ |

**Transfer protocol:**

```python
def transfer_to_path(source_state: PathTransferState,
                     coefficient_map: CoefficientMap,
                     target_provider, target_penalty_set,
                     family, weights, max_iter_remaining=50):
    """
    Warm-start a new execution path from a partial fit.

    Protocol:
    1. Validate source state (INV-1 through INV-5)
    2. Convert penalties to target format at boundary
    3. First PIRLS iteration: unconditionally accepted
       (pen_dev_prev = None, same as fresh-fit iteration 0)
    4. If first iteration diverges (NaN / Cholesky fail),
       ROLLBACK to source beta and abort with error
    5. Convergence check resumes from source iteration count
    """
    # Gate: validate before touching target path
    source_state.validate(coefficient_map)

    # Penalty conversion at boundary:
    # Dense-GPU uses StructuredPenalty (jax.Array inside)
    # Sparse-CPU uses scipy.sparse.csc_matrix
    # Conversion is via CoefficientMap column ranges (path-independent)
    lambdas = np.exp(source_state.log_lambda)
    target_penalty = target_penalty_set.to_sparse_scipy()
    S_lambda = sum(
        lam * S_j for lam, S_j in zip(lambdas, target_penalty)
    )

    # PIRLS warm-start
    # pen_dev_prev = None → first iteration unconditionally accepted
    result = pirls_fit(
        target_provider, family, S_lambda, weights,
        beta_init=source_state.beta,
        start_iteration=source_state.iteration,
        max_iter=max_iter_remaining,
    )

    # Rollback check: if first target iteration produced NaN or
    # Cholesky failure, the target path can't handle this problem either
    if result.instability_count > 0 and result.n_iter == source_state.iteration + 1:
        raise RuntimeError(
            f"Path transfer failed: target path (Sparse-CPU) also "
            f"unstable at first iteration. Source beta preserved. "
            f"Consider reducing model complexity or checking data."
        )

    return result
```

**Rollback rules:**

If the first iteration on the target path produces any instability signal (Cholesky failure, NaN, step-halving exhaustion), the transfer is considered failed. Source β is preserved (it's immutable in the frozen dataclass). The user gets a clear error rather than a silently-diverged fit.

**Validation frequency (v1.9):**

`validate()` is called exactly twice per transfer — once at creation (source side) and once at consumption (target side, before first PIRLS iteration). This catches:
- Source-side bugs: β has NaN from an undetected instability, log_λ has wrong length from a penalty mismatch
- Boundary corruption: dtype downcast during device-to-host transfer, shape change from a stale CoefficientMap

The cost is O(p) per call (finiteness check + shape check). Negligible compared to a single PIRLS iteration.

**Transfer test strategy (v1.9):**

```python
# tests/test_path_transfer.py

import hypothesis
from hypothesis import given, strategies as st

@given(
    transfer_iteration=st.integers(min_value=1, max_value=50),
    n_smooth=st.integers(min_value=1, max_value=5),
)
def test_transfer_invariants_hold(transfer_iteration, n_smooth, gaussian_test_data):
    """
    Property test: transfer at any iteration should produce a
    PathTransferState that passes validate() on both sides.
    """
    # Fit on Dense path, interrupt at transfer_iteration
    partial_result = pymgcv.gam(
        "y ~ s(x1) + s(x2)", data=gaussian_test_data,
        execution_path="dense_gpu",
        _debug_max_inner_iter=transfer_iteration,
    )
    state = PathTransferState.from_result(partial_result)
    state.validate(partial_result._coefficient_map)  # Source side

    # Transfer to Sparse-CPU
    result = transfer_to_path(
        state, sparse_provider, penalty_set,
        partial_result.family, weights=None,
    )
    # Objective must not increase on first accepted iteration
    assert result.penalized_deviance <= state.pen_deviance + 1e-7 * abs(state.pen_deviance), \
        "Transfer violated objective monotonicity"


def test_transfer_rollback_on_pathological_input():
    """
    Transfer with a deliberately ill-conditioned problem should
    raise RuntimeError (rollback), not silently diverge.
    """
    # Create problem that Dense-GPU can barely handle
    ill_result = pymgcv.gam(
        "y ~ s(x, k=200)", data=ill_conditioned_data,
        execution_path="dense_gpu",
        _debug_max_inner_iter=5,
    )
    state = PathTransferState.from_result(ill_result)
    # If Sparse-CPU also can't handle it, we get a clear error
    with pytest.raises(RuntimeError, match="target path.*also unstable"):
        transfer_to_path(state, sparse_provider, penalty_set,
                         ill_result.family, weights=None)
```

### 10.7 Sparsity Rules (When Matrices Are Sparse)

Within the Sparse-CPU and Chunked-Hybrid paths:

| Matrix | Storage | Rationale |
|---|---|---|
| Penalty matrices S_j | **Always sparse** (CSC) | Banded/block structure, p×p |
| Random effect design Z | **Always sparse** (CSC) | Indicator matrix, density ~ 1/n_levels |
| B-spline basis X | **Sparse if density < 30%** | Banded, ~5% nonzero for cubic |
| TPRS basis X | **Always dense** | Full, no exploitable structure |
| Tensor product basis | **Sparse if product dims > 100** | Can be very large but structured |
| Combined S_λ = Σ λ_j S_j | **Always sparse** (block-diagonal) | `scipy.sparse.block_diag` |
| Cross-product X^T W X | **Always dense** | p×p, filled by multiplication |
| Hat matrix A | **Never formed** | Use trace estimators |

### 10.8 Efficient Trace Estimation

```python
def efficient_trace_hat_matrix(XtWX, S_lambda, method="cholesky", X=None, W=None,
                               n_probes=30):
    """
    Estimate tr(A) = tr((XtWX + S_λ)^{-1} XtWX).

    v1.4 fix: Removed np.linalg.inv(H) path. All methods now use
    Cholesky factorization for numerical stability.

    Methods:
    - "cholesky": tr(H^{-1} XtWX) via L^{-1} and elementwise product.
      O(p³) for factorization + O(p²) for trace. Fine for p < 5000.
      Does NOT form H^{-1} as a dense matrix.
    - "hutchinson": Stochastic trace estimator via Rademacher probes.
      O(p² × n_probes). For very large p (> 5000) or when only
      an approximation is needed.

    For the Chunked path:
    - "hutchinson_full": Requires random probes through the full data,
      meaning another chunk pass. Used only when exact is too expensive.
    """
    from scipy.linalg import solve_triangular

    H = XtWX + S_lambda
    p = H.shape[0]

    if method == "cholesky":
        # Cholesky factorize H = L L^T
        L = np.linalg.cholesky(H + 1e-12 * np.eye(p))

        # Z = L^{-1} via forward substitution (triangular solve)
        Z = solve_triangular(L, np.eye(p), lower=True)

        # tr(H^{-1} XtWX) = tr(Z^T Z XtWX)
        #                   = tr(Z XtWX Z^T)    [cyclic property]
        #                   = sum_{ij} Z_{ij} (Z @ XtWX)_{ij}
        # This is an elementwise product, no matrix inversion formed.
        Q = Z @ XtWX
        return np.sum(Z * Q)

    elif method == "hutchinson":
        # Stochastic trace estimator: tr(H^{-1} XtWX) ≈ E[z^T H^{-1} XtWX z]
        # Uses Cholesky solve (not inv) for each probe.
        L = np.linalg.cholesky(H + 1e-12 * np.eye(p))
        traces = []
        rng = np.random.default_rng(42)
        for _ in range(n_probes):
            z = rng.choice([-1.0, 1.0], size=p)
            # H^{-1} XtWX z = L^{-T} L^{-1} XtWX z
            rhs = XtWX @ z
            v = solve_triangular(L, rhs, lower=True)       # L^{-1} (XtWX z)
            w = solve_triangular(L.T, v, lower=False)       # L^{-T} v
            traces.append(z @ w)
        return np.mean(traces)


def edf_per_term(XtWX, S_lambda, coefficient_map):
    """
    Effective degrees of freedom per smooth term.

    edf_j = tr(H^{-1} XtWX restricted to columns of term j)

    Uses a single Cholesky factorization of H, then extracts
    per-term traces via column slicing — O(p³) total, not O(p³)
    per term.
    """
    from scipy.linalg import solve_triangular

    H = XtWX + S_lambda
    p = H.shape[0]
    L = np.linalg.cholesky(H + 1e-12 * np.eye(p))
    Z = solve_triangular(L, np.eye(p), lower=True)  # L^{-1}

    edfs = {}
    for term in coefficient_map.terms:
        if term.type != "smooth":
            continue
        cols = slice(term.col_start, term.col_start + term.n_coefs)
        # XtWX restricted to columns of this term
        XtWX_j = XtWX[:, cols]
        Q_j = Z @ XtWX_j
        Z_j = Z[:, cols]
        edfs[term.label] = np.sum(Z_j * Q_j)  # tr(H^{-1} XtWX_j) restricted

    return edfs
```

### 10.9 Discretization (bam-specific)

Covariate discretization for `bam()` lives in `data/discretize.py`. This is orthogonal to execution path selection — discretization reduces n effectively by mapping covariates to a small number of bins, after which any execution path can be used. See Phase 5 (Section 19) for implementation details.

---

## 11. GPU and Hardware Acceleration

### 11.1 GPU Strategy

The primary GPU path is through JAX's XLA compiler, which supports:
- NVIDIA GPUs via CUDA/cuDNN
- Apple Silicon GPUs via Metal (jax-metal plugin)
- AMD GPUs via ROCm
- Google TPUs

```python
# Example: GPU-accelerated fitting

import pymgcv
pymgcv.configure(backend="jax", device="gpu")

# All subsequent operations automatically use GPU
model = pymgcv.gam(
    "y ~ s(x1) + s(x2) + te(x3, x4)",
    data=df,
    family="gaussian"
)
```

### 11.2 What Runs on GPU

| Operation | GPU Benefit | Implementation |
|---|---|---|
| Design matrix construction | High (parallel basis eval) | JAX vmap over observations |
| Matrix multiply X^T W X | Very High | XLA GEMM kernel |
| QR decomposition | High | XLA QR (cuSOLVER backend) |
| Cholesky factorization | High | XLA Cholesky (cuSOLVER) |
| PIRLS inner loop | Very High (all operations fused) | Single JAX JIT function |
| Log-likelihood evaluation | High (element-wise) | JAX vmap |
| Autodiff through log-lik | High | JAX grad (compiled) |
| Sparse operations | Medium | JAX experimental sparse / cuSPARSE |

### 11.3 CPU/GPU Data Transfer Minimization

```python
# Key principle: move data to device ONCE, keep it there

@jax.jit
def full_pirls_iteration(X_device, y_device, beta, S_lambda, family_params):
    """
    Entire PIRLS iteration as a single JIT-compiled function.
    No CPU/GPU round-trips within the iteration.
    """
    eta = X_device @ beta
    mu = link_inverse(eta)
    W = compute_weights(mu, family_params)
    z = compute_response(y_device, mu, eta, family_params)
    W_sqrt = jnp.sqrt(W)
    WX = W_sqrt[:, None] * X_device
    XtWX = WX.T @ WX + S_lambda
    XtWz = WX.T @ (W_sqrt * z)
    beta_new = jnp.linalg.solve(XtWX, XtWz)
    dev = compute_deviance(y_device, mu, family_params)
    return beta_new, mu, dev
```

### 11.4 Metal (Apple Silicon) Considerations

```python
# Apple Metal via jax-metal plugin
# Limitations: some operations not supported on Metal
# Fallback strategy:

def get_device_capabilities(device):
    """Check what operations the current device supports."""
    caps = {
        "sparse_ops": False,    # Metal doesn't support sparse yet
        "64bit_float": True,    # Metal supports float64 since M1
        "complex": False,       # Limited complex number support
    }
    if "gpu" in str(device):
        backend = jax.default_backend()
        if backend == "metal":
            caps["sparse_ops"] = False
            caps["complex"] = False
        elif backend == "cuda":
            caps["sparse_ops"] = True
            caps["complex"] = True
    return caps
```

---

## 12. Random Effects and Mixed Models

### 12.1 Random Effects as Penalized Smooths

In mgcv, random effects are just another smooth term with an identity penalty:

```python
# s(group, bs="re") is equivalent to:
# Design matrix Z: indicator matrix (n × n_levels)
# Penalty matrix: I_{n_levels}
# Smoothing parameter λ estimates σ²_ε / σ²_b
```

This is already handled by `RandomEffectSmooth` in Section 5.6.

### 12.2 gamm() via PQL

```python
# fitting/gamm_fit.py

def gamm_fit(formula, data, family, correlation=None, random=None):
    """
    Fit GAMM using Penalized Quasi-Likelihood.

    gamm() is for models with:
    - Correlation structures (AR1, spatial, etc.)
    - Complex random effect structures beyond simple intercepts
    - Cases where the GAM smoothing parameter estimation
      should be embedded in a mixed model framework

    Algorithm:
    1. Set up GAM as a mixed model: smooth terms → random effects
       with precision matrices as penalty matrices
    2. For Gaussian: directly fit as LMM
    3. For non-Gaussian: iterate PQL:
       a. Compute working response and weights (PIRLS-like)
       b. Fit LMM on working response with working weights
       c. Update linear predictor
       d. Iterate to convergence

    We use the same LMM solver as lme4 (sparse Cholesky on the
    penalized system) for the inner mixed model fit.
    """
    pass
```

### 12.3 Correlation Structures

```python
# For gamm(), support these correlation structures:

class CorrelationStructure(ABC):
    @abstractmethod
    def get_correlation_matrix(self, params, groups, times) -> sparse.spmatrix:
        ...

class CorAR1(CorrelationStructure):
    """AR(1) correlation within groups."""
    def get_correlation_matrix(self, phi, groups, times):
        # Block-diagonal AR(1) correlation matrix
        pass

class CorCompSymm(CorrelationStructure):
    """Compound symmetry (exchangeable) correlation."""
    pass

class CorSpatial(CorrelationStructure):
    """Spatial correlation (exponential, Gaussian, spherical, Matérn)."""
    pass
```

---

## 13. Formula Interface and Model Specification

### 13.1 Formula Parser

**Architecture (v1.15: AST-based, not regex):** The formula parser is split into two layers. The parametric part uses `formulaic` (successor to `patsy`), which handles R-style formula semantics correctly: `*` expansion, `(a+b)^2` interactions, factor contrasts (treatment, sum, Helmert), `.` for "all other columns", `I()` for protecting arithmetic, and proper handling of categorical encoding. We write a **preprocessor** that extracts smooth terms before passing the remainder to `formulaic`.

**v1.15: Why AST, not regex.** Previous versions used regex with balanced parentheses to extract smooth calls. This is a guaranteed bug factory: nested calls like `s(x, k=int(log(n)))`, interaction notation `s(x):z`, and formulas like `y ~ a * s(x)` (stripping `s(x)` leaves `y ~ a *`, invalid syntax for formulaic) all break regex extraction. The parser now uses Python's `ast` module to identify `Call` nodes, which handles arbitrary nesting, operator precedence, and complex arguments correctly.

```python
# formula/parser.py

"""
Two-layer formula parser:

Layer 1 (custom, AST-based): Extract smooth terms s(), te(), ti(), t2(),
  offset() from the formula string using Python's ast module.
  Replaces them with unique placeholder column names.

Layer 2 (formulaic): Parse the remaining parametric formula using
  formulaic, which handles all R-style semantics correctly.

Example:
  "y ~ x1 * x2 + s(x3, bs='cr', k=20) + te(x4, x5) + offset(log_n)"

  → Smooth terms extracted:
    [SmoothSpec("s(x3, bs='cr', k=20)", vars=["x3"], bs="cr", k=20),
     SmoothSpec("te(x4, x5)", vars=["x4", "x5"], bs="tp")]

  → Parametric formula passed to formulaic:
    "y ~ x1 * x2"
    (which formulaic expands to: intercept + x1 + x2 + x1:x2)

  → Offset extracted: ["log_n"]

Supported smooth syntax:
  s(x1)                          # Default TPRS
  s(x1, bs="cr", k=20)          # Cubic regression, 20 knots
  s(x1, by=group)               # Factor-by smooth
  te(x1, x2)                    # Tensor product
  ti(x1, x2)                    # Tensor interaction
  t2(x1, x2, bs=["cr","ps"])    # Type 2 tensor
  s(group, bs="re")             # Random intercept
  s(x1, group, bs="fs")         # Factor-smooth interaction
  s(x1, x2, bs="so", xt=...)    # Soap film
  offset(log_exposure)           # Offset
"""

from dataclasses import dataclass
from typing import Optional
import ast

_SMOOTH_FUNCTIONS = {'s', 'te', 'ti', 't2'}
_OFFSET_FUNCTIONS = {'offset'}


@dataclass
class ParsedFormula:
    response: str
    parametric_formula: str          # Passed to formulaic
    parametric_terms: list[str]      # Resolved by formulaic
    smooth_terms: list["SmoothSpec"]
    offset_terms: list[str]
    random_terms: list["SmoothSpec"]  # bs="re" or bs="fs"


class _SmoothExtractor(ast.NodeVisitor):
    """
    Walk the AST of the formula RHS, identify Call nodes for smooth
    functions, record their positions, and extract arguments.

    Handles arbitrary nesting: s(x, k=int(log(n))) parses correctly
    because ast handles balanced parens at the language level.
    """

    def __init__(self):
        self.smooth_calls = []   # (col_offset, end_col_offset, SmoothSpec)
        self.offset_calls = []   # (col_offset, end_col_offset, variable_name)

    def visit_Call(self, node):
        func_name = self._get_func_name(node)
        if func_name in _SMOOTH_FUNCTIONS:
            spec = _parse_smooth_call_ast(func_name, node)
            self.smooth_calls.append((node.col_offset, node.end_col_offset, spec))
        elif func_name in _OFFSET_FUNCTIONS and node.args:
            var_name = ast.unparse(node.args[0])
            self.offset_calls.append((node.col_offset, node.end_col_offset, var_name))
        self.generic_visit(node)

    def _get_func_name(self, node):
        return node.func.id if isinstance(node.func, ast.Name) else None


def _parse_smooth_call_ast(func_name, call_node):
    """Extract SmoothSpec from an AST Call node."""
    variables = [ast.unparse(arg) for arg in call_node.args]
    kwargs = {}
    for kw in call_node.keywords:
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            kwargs[kw.arg] = ast.unparse(kw.value)

    return SmoothSpec(
        variables=variables,
        basis_type=kwargs.get('bs', _default_basis(func_name)),
        n_knots=kwargs.get('k', None),
        by_variable=kwargs.get('by', None),
    )


def parse_formula(formula_str: str, data=None) -> ParsedFormula:
    """
    Parse a formula string into components using AST extraction.

    1. Split on ~ to get LHS (response) and RHS
    2. Parse RHS as a Python expression via ast.parse
    3. Walk AST to find smooth/offset Call nodes
    4. Replace smooth calls with placeholders in source (right-to-left)
    5. Clean up and pass parametric remainder to formulaic
    """
    lhs, rhs = formula_str.split("~", 1)
    response = lhs.strip()
    rhs = rhs.strip()

    rhs_py = _r_formula_to_python_expr(rhs)
    tree = ast.parse(rhs_py, mode='eval')
    extractor = _SmoothExtractor()
    extractor.visit(tree.body)

    # Replace smooth calls with placeholders (right-to-left)
    smooth_specs = []
    rhs_clean = rhs_py
    for i, (start, end, spec) in enumerate(
        sorted(extractor.smooth_calls, key=lambda x: x[0], reverse=True)
    ):
        rhs_clean = rhs_clean[:start] + f"__smooth_{i}__" + rhs_clean[end:]
        if spec.basis_type in ("re", "fs"):
            smooth_specs.append(('random', spec))
        else:
            smooth_specs.append(('smooth', spec))

    for start, end, var in sorted(extractor.offset_calls, key=lambda x: x[0], reverse=True):
        rhs_clean = rhs_clean[:start] + rhs_clean[end:]

    # Strip placeholders and clean dangling operators
    for i in range(len(extractor.smooth_calls)):
        rhs_clean = rhs_clean.replace(f"__smooth_{i}__", "")
    rhs_clean = _clean_formula_rhs(rhs_clean)
    if not rhs_clean or rhs_clean.strip() in ('', '1'):
        rhs_clean = '1'

    parametric_formula = f"{response} ~ {rhs_clean}"
    smooths = [s for t, s in smooth_specs if t == 'smooth']
    randoms = [s for t, s in smooth_specs if t == 'random']
    offsets = [var for _, _, var in extractor.offset_calls]

    return ParsedFormula(
        response=response,
        parametric_formula=parametric_formula,
        parametric_terms=[],
        smooth_terms=smooths,
        offset_terms=offsets,
        random_terms=randoms,
    )


def build_parametric_matrix(parametric_formula: str, data):
    """
    Build the parametric design matrix using formulaic.

    Handles all R-style semantics:
    - x1 * x2 → x1 + x2 + x1:x2
    - (x1 + x2 + x3)^2 → all pairwise interactions
    - C(x, Treatment) → treatment-coded contrasts
    - I(x^2) → literal x-squared column
    - . → all other columns
    """
    import formulaic
    model_matrix = formulaic.model_matrix(parametric_formula, data)
    return model_matrix.values, list(model_matrix.columns)
```

### 13.2 Design Matrix Assembly

```python
# formula/design.py

def build_model_matrix(parsed_formula: ParsedFormula,
                       data: dict[str, np.ndarray]):
    """
    Assemble the full model matrix from parsed formula.

    Returns:
    - X: Full model matrix (n × p) — may be sparse
    - term_info: Metadata mapping column ranges to terms
    - penalty_matrices: List of (sparse) penalty matrices
    - constraint_matrices: Identifiability constraints

    Process:
    1. Build parametric design matrix (intercept + linear terms + interactions)
    2. For each smooth term:
       a. Instantiate correct Smooth subclass from registry
       b. Call setup() to determine knots, eigenvectors
       c. Call build_design_matrix() to get basis matrix
       d. Apply identifiability constraints
       e. Call build_penalty_matrices()
    3. Concatenate horizontally: X = [X_parametric | X_smooth1 | X_smooth2 | ...]
    4. Build block-diagonal penalty matrix S = blockdiag(0, S_1, S_2, ...)
    """
    from pymgcv.smooths.registry import get_smooth_class

    blocks = []
    penalties = []
    term_info = []
    col_offset = 0

    # 1. Parametric terms
    X_param = _build_parametric_matrix(parsed_formula.parametric_terms, data)
    n_param = X_param.shape[1]
    blocks.append(X_param)
    term_info.append({
        'type': 'parametric',
        'col_start': 0,
        'col_end': n_param,
        'n_coefs': n_param,
    })
    col_offset = n_param
    # Zero penalty for parametric terms
    penalties.append(sparse.csc_matrix((n_param, n_param)))

    # 2. Smooth terms
    all_smooths = parsed_formula.smooth_terms + parsed_formula.random_terms
    smooth_objects = []

    for spec in all_smooths:
        SmoothClass = get_smooth_class(spec.bs)
        smooth = SmoothClass(spec)
        smooth.setup(data)

        X_s = smooth.build_design_matrix(data)
        S_list = smooth.build_penalty_matrices()

        # Apply identifiability constraint (sum-to-zero)
        if smooth.null_space_dim > 0 and spec.bs not in ("re",):
            X_s, S_list, Z = smooth.apply_identifiability_constraint(X_s)
            smooth._constraint_transform = Z

        n_s = X_s.shape[1]
        blocks.append(X_s)
        term_info.append({
            'type': 'smooth',
            'label': spec.term_label,
            'col_start': col_offset,
            'col_end': col_offset + n_s,
            'n_coefs': n_s,
            'smooth': smooth,
            'penalty_indices': list(range(len(penalties),
                                         len(penalties) + len(S_list))),
        })
        penalties.extend(S_list)
        smooth_objects.append(smooth)
        col_offset += n_s

    # 3. Assemble
    if any(sparse.issparse(b) for b in blocks):
        X = sparse.hstack(blocks, format='csc')
    else:
        X = np.column_stack(blocks)

    return X, term_info, penalties, smooth_objects
```

---

## 14. Prediction, Summary, and Post-Estimation

### 14.1 Prediction

```python
# predict/predict.py

def predict_gam(model, newdata=None, type="link", se_fit=False,
                terms=None, exclude=None, n_samples=0):
    """
    Prediction from fitted GAM.

    type: "link" (linear predictor), "response" (μ scale), "terms" (per-term)
    se_fit: If True, return standard errors
    n_samples: If > 0, return posterior samples

    Standard errors are based on the Bayesian posterior covariance:
    Var(Xp β) = Xp Vβ Xp^T

    where Vβ is the posterior covariance from the fit.
    """
    if newdata is None:
        Xp = model.X
    else:
        Xp = _build_prediction_matrix(model, newdata, terms, exclude)

    # Point prediction
    eta = Xp @ model.coefficients
    if model.offset is not None and newdata is None:
        eta += model.offset

    if type == "response":
        mu = model.family.link.inverse(eta)
        prediction = mu
    elif type == "terms":
        # Per-term contributions
        prediction = {}
        for info in model.term_info:
            cols = slice(info['col_start'], info['col_end'])
            prediction[info.get('label', 'parametric')] = \
                Xp[:, cols] @ model.coefficients[cols]
    else:
        prediction = eta

    result = {'fit': prediction}

    if se_fit:
        # Bayesian standard errors
        if sparse.issparse(Xp):
            # Avoid forming full XVX^T
            V = model.Vp
            se = np.sqrt(np.array(
                (Xp @ V @ Xp.T).diagonal()
            ).flatten())
        else:
            se = np.sqrt(np.sum((Xp @ model.Vp) * Xp, axis=1))
        result['se'] = se

    if n_samples > 0:
        # Posterior simulation
        from numpy.random import multivariate_normal
        beta_samples = multivariate_normal(
            model.coefficients, model.Vp, size=n_samples
        )
        eta_samples = Xp @ beta_samples.T
        if type == "response":
            result['samples'] = model.family.link.inverse(eta_samples)
        else:
            result['samples'] = eta_samples

    return result
```

### 14.2 Summary

```python
# summary/summary.py

def summary_gam(model, dispersion=None, freq=False):
    """
    summary.gam equivalent.

    Returns:
    - Parametric coefficient table (estimates, se, z/t, p-values)
    - Smooth term table (EDF, Ref.df, F/Chi-sq, p-values)
    - R-sq, deviance explained, scale estimate
    - GCV/UBRE/REML score

    Smooth term p-values use Wood's (2013) method based on
    the rank of the penalty and the Bayesian covariance.
    """
    result = {}

    # Parametric terms
    n_param = model.term_info[0]['n_coefs']
    beta_param = model.coefficients[:n_param]
    se_param = np.sqrt(np.diag(model.Vp)[:n_param])
    t_vals = beta_param / se_param
    from scipy.stats import t as t_dist
    df_resid = model.n - model.edf_total
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_vals), df=df_resid))

    result['parametric'] = {
        'estimate': beta_param,
        'std_error': se_param,
        't_value': t_vals,
        'p_value': p_vals,
    }

    # Smooth terms
    smooth_table = []
    for info in model.term_info[1:]:  # Skip parametric
        s = info['smooth']
        cols = slice(info['col_start'], info['col_end'])
        beta_s = model.coefficients[cols]
        Vp_s = model.Vp[cols, cols]

        # EDF for this smooth
        edf_s = np.trace(
            model.hat_matrix[cols, cols]
        ) if hasattr(model, 'hat_matrix') else info.get('edf', 1)

        # Test statistic (Wood 2013)
        F_stat, p_val = _smooth_test(beta_s, Vp_s, s.penalty_matrices,
                                      model.scale, edf_s)

        smooth_table.append({
            'label': info['label'],
            'edf': edf_s,
            'ref_df': _reference_df(edf_s, s.n_coefs),
            'F': F_stat,
            'p_value': p_val,
        })

    result['smooth'] = smooth_table
    result['r_squared'] = _compute_r_squared(model)
    result['deviance_explained'] = _deviance_explained(model)
    result['scale'] = model.scale
    result['n'] = model.n

    return result
```

---

## 15. Model Comparison, Concurvity, and Diagnostics

### 15.1 Model Comparison Infrastructure

The `GAMResult` object must store enough information to support post-hoc model comparison without re-fitting. This was missing from v1.0.

```python
# summary/model_comparison.py

@dataclass
class ModelComparisonInfo:
    """Stored in GAMResult for post-hoc comparison."""
    log_likelihood: float          # Full (penalized) log-likelihood
    null_log_likelihood: float     # Log-lik of intercept-only model
    edf_per_term: np.ndarray       # Per-term effective degrees of freedom
    edf_total: float               # Total EDF (sum of per-term)
    n_obs: int
    scale: float
    reml_score: float              # Or GCV/ML depending on method
    aic: float                     # -2*ll + 2*edf
    bic: float                     # -2*ll + log(n)*edf


def compute_aic(model):
    """AIC = -2 * log_lik + 2 * edf_total."""
    return -2 * model.comparison_info.log_likelihood + 2 * model.comparison_info.edf_total

def compute_bic(model):
    """BIC = -2 * log_lik + log(n) * edf_total."""
    n = model.comparison_info.n_obs
    return (-2 * model.comparison_info.log_likelihood +
            np.log(n) * model.comparison_info.edf_total)


def anova_gam(*models, test="F"):
    """
    anova.gam equivalent: compare nested GAM models.

    For a single model: tests each smooth term against zero
    (using Wood 2013 p-values).

    For multiple models: sequential comparison using F-tests
    or Chi-squared tests for the change in deviance / EDF.

    Parameters
    ----------
    *models : GAMResult
        One or more fitted models (must be nested for multi-model).
    test : str
        "F" for F-test, "Chisq" for chi-squared.
    """
    if len(models) == 1:
        return _anova_single_model(models[0], test)
    else:
        # Sort by EDF (simplest first)
        sorted_models = sorted(models, key=lambda m: m.comparison_info.edf_total)
        return _anova_sequential(sorted_models, test)
```

### 15.2 Concurvity Detection

Concurvity is the smooth analogue of multicollinearity: it measures how much each smooth term can be approximated by the other terms. This was listed in v1.0 but had no implementation detail.

```python
# summary/concurvity.py

def concurvity(model, full=True):
    """
    Compute concurvity measures for a fitted GAM.

    Concurvity measures how much of each smooth's column space
    is "explained" by the other smooth terms.

    Three measures (following mgcv):

    1. worst: max concurvity over all smooth directions
       C_j = 1 - min eigenvalue of (I - P_{-j}) restricted to col(X_j)
       where P_{-j} is projection onto all terms except j.

    2. observed: concurvity in the direction of the fitted smooth
       C_j = ||P_{-j} f̂_j||² / ||f̂_j||²

    3. estimate: average concurvity over the smooth's column space
       C_j = 1 - (1/k_j) * tr((I - P_{-j}) restricted to col(X_j))

    Values near 1 indicate serious concurvity (the smooth is nearly
    redundant given the other terms). Values near 0 are fine.

    Parameters
    ----------
    model : GAMResult
        Fitted GAM model.
    full : bool
        If True, return pairwise concurvity between all term pairs.
        If False, return only the per-term summary.

    Returns
    -------
    dict with keys 'worst', 'observed', 'estimate',
    each mapping term labels to concurvity values.
    """
    term_info = [t for t in model.term_info if t['type'] == 'smooth']
    n_smooth = len(term_info)
    X = model.X

    results = {
        'worst': {},
        'observed': {},
        'estimate': {},
    }

    for j, info_j in enumerate(term_info):
        cols_j = slice(info_j['col_start'], info_j['col_end'])
        n_cols_j = info_j['col_end'] - info_j['col_start']

        # v1.5: skip concurvity for very large sparse terms (e.g., MRF).
        # Densifying these would OOM. Concurvity is a diagnostic, not
        # required for fitting — better to skip and warn than to crash.
        MAX_CONCURVITY_COLS = 10_000
        if n_cols_j > MAX_CONCURVITY_COLS:
            import warnings
            warnings.warn(
                f"Skipping concurvity for term '{info_j.get('label', j)}' "
                f"({n_cols_j} columns) — too large to densify. "
                f"Use individual term comparisons instead."
            )
            continue

        X_j = _to_dense(X[:, cols_j])

        # Build X_{-j}: all columns except term j
        other_cols = []
        for k, info_k in enumerate(term_info):
            if k != j:
                cols_k = slice(info_k['col_start'], info_k['col_end'])
                other_cols.append(_to_dense(X[:, cols_k]))
        if not other_cols:
            continue
        X_minus_j = np.column_stack(other_cols)

        # Projection matrix P_{-j} (hat matrix of X_{-j})
        # Efficient: only need P_{-j} @ X_j, not full P
        Q, _ = np.linalg.qr(X_minus_j, mode='reduced')
        PX_j = Q @ (Q.T @ X_j)  # P_{-j} X_j

        # Worst case: largest singular value of P_{-j} restricted to col(X_j)
        _, s, _ = np.linalg.svd(PX_j, full_matrices=False)
        norms_j = np.linalg.norm(X_j, axis=0)
        norms_j = np.maximum(norms_j, 1e-10)
        worst_j = np.max(s / np.linalg.norm(X_j, 'fro'))

        # Observed: concurvity for the fitted smooth
        beta_j = model.coefficients[cols_j]
        f_j = X_j @ beta_j
        Pf_j = Q @ (Q.T @ f_j)
        obs_j = np.sum(Pf_j**2) / max(np.sum(f_j**2), 1e-10)

        # Estimate: average
        est_j = np.sum(s**2) / max(np.sum(np.linalg.norm(X_j, axis=0)**2), 1e-10)

        label = info_j['label']
        results['worst'][label] = float(np.clip(worst_j, 0, 1))
        results['observed'][label] = float(np.clip(obs_j, 0, 1))
        results['estimate'][label] = float(np.clip(est_j, 0, 1))

    return results
```

---

## 16. Distributed and Multi-Device Compute

**v1.11: Complete rewrite.** Previous versions (v1.0–v1.10) used NumPy-based Dask/Ray providers where workers computed sufficient statistics in NumPy and shipped them to a Python coordinator. This broke JAX out of the loop: no JIT for the outer PIRLS loop, no autodiff through the full computation, no extended family AD on workers (breaking NB/Tweedie), and a Python round-trip per iteration. The new architecture uses JAX's native SPMD model at every scale.

### 16.1 Design Principle: Same Code, Different Shardings

The key insight is that JAX's SPMD model means the distributed PIRLS step is **identical code** to the single-GPU step. You don't write a distributed version — you shard the input arrays and JAX's compiler handles communication.

```python
# Single GPU:
X = jax.device_put(X_numpy, jax.devices()[0])

# Multi-GPU, one host:
mesh = jax.make_mesh((len(jax.devices()),), ('data',))
X = jax.device_put(X_numpy, NamedSharding(mesh, P('data', None)))

# Multi-host cluster (after jax.distributed.initialize()):
mesh = jax.make_mesh((total_devices,), ('data',))
X = jax.device_put(X_local, NamedSharding(mesh, P('data', None)))

# The PIRLS function is THE SAME in all three cases.
# JAX sees the sharding and compiles communication automatically.
```

This replaces the `StatisticsProvider` abstraction for the distributed case. `StatisticsProvider` remains relevant only for the out-of-core case (Section 16.5) where data doesn't fit in aggregate device memory.

**Dense-only constraint (v1.14):** The SPMD path operates on **dense** `jax.Array` objects. The row-sharding model (`P('data', None)`) assumes a dense `(n, p)` matrix where every device's local shard has the same column count and layout. Smooth types that produce structurally sparse design matrices — `FactorBySmooth` (Section 5.7), `bs="fs"` (Section 5.6), `bs="re"` (Section 5.6), `bs="mrf"` (Section 5.8) — must be densified before `jax.device_put` into the SPMD mesh.

**Critical: route BEFORE densifying (v1.14).** Densifying a large sparse X can be the dominant memory event — potentially OOM on the host before any GPU work begins. The routing decision must therefore be made from Phase 1 metadata alone, without materializing the dense matrix:

```python
# fitting/model_assembly.py — pre-materialization routing

def estimate_peak_memory(smooth_specs, data, n_devices=1):
    """
    v1.15: Estimate peak GPU memory from Phase 1 metadata only.
    No dense allocation occurs here.

    Components estimated (all on each device for SPMD):
      X_shard:     (n/d) * p * 8    — row-sharded design matrix
      WX_shard:    (n/d) * p * 8    — weighted X (same shape, temporary per iteration)
      XtWX:        p * p * 8        — replicated (post all-reduce)
      S_lambda:    p * p * 8        — replicated (dense, even if structurally block-diag)
      Cholesky:    p * p * 8        — replicated (factorization of XtWX + S_lambda)
      XtWz:        p * 8            — replicated (negligible)
      beta/mu/eta: n/d * 8 * 3      — per-device vectors (negligible vs above)

    Total per device ≈ 2*(n/d)*p*8 + 3*p²*8

    For FactorBySmooth: p_expanded = n_levels * k_per_level
    Note: constraint absorption (CoefficientMap) can reduce effective p.
    We estimate conservatively using pre-constraint p.
    """
    total_p = _count_parametric_cols(smooth_specs, data)
    for spec in smooth_specs:
        if isinstance(spec, FactorBySmooth):
            n_levels = len(np.unique(data[spec._spec.by_variable]))
            base_k = spec._spec.n_knots
            total_p += n_levels * base_k
        else:
            total_p += spec.n_coefs
    n_smooth = sum(s.n_penalties for s in smooth_specs)

    n_obs = data.shape[0]
    n_per_device = n_obs // max(n_devices, 1)

    # Peak memory per device
    x_shard_bytes = n_per_device * total_p * 8
    wx_shard_bytes = n_per_device * total_p * 8  # WX temporary
    replicated_bytes = 3 * total_p * total_p * 8  # XtWX + S_lambda + Cholesky
    vector_bytes = n_per_device * 8 * 5  # beta, mu, eta, W, z
    peak_bytes_per_device = x_shard_bytes + wx_shard_bytes + replicated_bytes + vector_bytes

    # Host memory for densification (one-time, before device_put)
    host_dense_bytes = n_obs * total_p * 8

    return PeakMemoryEstimate(
        total_p=total_p,
        n_smooth=n_smooth,
        n_obs=n_obs,
        peak_bytes_per_device=peak_bytes_per_device,
        host_dense_bytes=host_dense_bytes,
        replicated_bytes=replicated_bytes,
        x_shard_bytes=x_shard_bytes,
    )


@dataclass
class PeakMemoryEstimate:
    total_p: int
    n_smooth: int
    n_obs: int
    peak_bytes_per_device: int
    host_dense_bytes: int
    replicated_bytes: int
    x_shard_bytes: int


def route_execution_path(smooth_specs, data, mesh=None, interconnect="nvlink",
                          user_method=None):
    """
    v1.15: Decide execution path BEFORE materializing dense X.
    Returns (mode_selection, X_matrix, execution_path_reason).

    Order of operations:
      1. Estimate p, n_smooth, peak memory from metadata (no allocation)
      2. Check host memory for densification feasibility
      3. Call auto_select_distributed_mode() with estimates
      4. Only if result is 'spmd' or 'spmd_single_solve': densify X
      5. Otherwise: stay sparse, route to Sparse-CPU or chunked

    The execution_path_reason string goes directly into GAMResult
    for surfacing in summary(). Every routing decision is explained.
    """
    n_devices = len(mesh.devices) if mesh is not None else 1
    est = estimate_peak_memory(smooth_specs, data, n_devices)

    reason_parts = [
        f"p={est.total_p}, n={est.n_obs}, n_smooth={est.n_smooth}",
    ]

    if mesh is not None:
        # Check host memory BEFORE attempting densification
        import psutil
        available_host_bytes = psutil.virtual_memory().available
        if est.host_dense_bytes > 0.8 * available_host_bytes:
            reason = (
                f"Host OOM risk: dense X would require "
                f"{est.host_dense_bytes/1e9:.1f}GB but only "
                f"{available_host_bytes/1e9:.1f}GB available. "
                f"Routing to chunked path."
            )
            X_sparse = _assemble_sparse_model_matrix(smooth_specs, data)
            return 'chunked', X_sparse, reason

        mode_sel = auto_select_distributed_mode(
            est.total_p, est.n_obs, est.n_smooth, mesh, interconnect,
            user_method=user_method,
        )
        reason_parts.append(f"peak_mem/device={est.peak_bytes_per_device/1e6:.0f}MB")
        reason_parts.append(f"replicated={est.replicated_bytes/1e6:.0f}MB")

        if mode_sel.spmd_mode in ('spmd', 'spmd_single_solve'):
            reason_parts.append(f"mode={mode_sel.spmd_mode}")
            X_dense = _densify_model_matrix(smooth_specs, data)
            reason = "; ".join(reason_parts)
            return mode_sel, X_dense, reason
        else:
            reason_parts.append(f"mode={mode_sel.spmd_mode} (sparse path)")
            X_sparse = _assemble_sparse_model_matrix(smooth_specs, data)
            reason = "; ".join(reason_parts)
            return mode_sel, X_sparse, reason
    else:
        # Single-device
        if est.total_p <= SAFE_DENSE_P:
            reason_parts.append(f"p≤{SAFE_DENSE_P}: dense single-GPU")
            return 'single_gpu', _densify_model_matrix(smooth_specs, data), "; ".join(reason_parts)
        else:
            reason_parts.append(f"p>{SAFE_DENSE_P}: sparse-CPU")
            return 'sparse_cpu', _assemble_sparse_model_matrix(smooth_specs, data), "; ".join(reason_parts)
```

For moderate p (within the gates in Section 16.7), densification is acceptable: a 2000-column dense X costs 16KB/row regardless of sparsity pattern. For factor-by smooths with many levels where the estimated p exceeds SPMD gates, routing diverts to Sparse-CPU or out-of-core without ever allocating the dense matrix.

**Performance expectation (v1.14):** Users with naturally sparse models (FactorBySmooth with many levels, `bs="re"` with many groups) should expect that multi-GPU SPMD can be **slower** than single-host Sparse-CPU for their workload, because SPMD discards sparsity structure. This is inherent to the "same code, different shardings" design — JAX's SPMD model has no sparse-aware sharding. The doc does not treat this as a bug. If a user's model is sparse-dominated and p is moderate, Sparse-CPU is the correct path. Multi-GPU SPMD is for dense-dominated models with large n.

### 16.2 How XtWX Formation Parallelizes

The expensive PIRLS operation is forming the sufficient statistics `XtWX` (p×p) and `XtWz` (p×1) from the n×p design matrix. When X is row-sharded across devices:

```
Device 0 holds X_0 (n/d × p)  →  computes  X_0^T W_0 X_0   (local, no communication)
Device 1 holds X_1 (n/d × p)  →  computes  X_1^T W_1 X_1   (local, no communication)
...
Device d holds X_d (n/d × p)  →  computes  X_d^T W_d X_d   (local, no communication)

XtWX = Σ_i X_i^T W_i X_i  →  XLA inserts all-reduce (p×p, single communication)
XtWz = Σ_i X_i^T W_i z_i  →  XLA inserts all-reduce (p×1, single communication)
```

The solve `β = (XtWX + S_λ)⁻¹ XtWz` is replicated: every device computes it (p×p Cholesky, cheap) and gets the same β. No broadcast needed — the all-reduce already puts XtWX on all devices.

**v1.13 note on penalty structure:** On the SPMD path, S_λ is a dense `(p, p)` replicated array, even when the underlying penalty is structurally block-diagonal (as with `FactorBySmooth`, Section 5.7, or tensor products, Section 5.5). FactorBySmooth's "sparse throughout" assembly (Section 5.7.2) is a Phase 1 (setup) property — it avoids OOM during basis construction. Once the model enters Phase 2 (JIT fitting), the assembled S_λ is densified for `jax.device_put`. For p within the SPMD gates (≤ 3000), this is a `(3000)² × 8 = 72MB` replicated array per device — acceptable. The block-diagonal structure of FactorBySmooth penalties is not exploited on the SPMD path. Exploiting it would require a block-sparse solver, which JAX does not natively support and which would break the "same code, different shardings" principle.

```python
# distributed/sharding.py

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial


def create_gam_mesh(devices=None):
    """
    Create a 1D Mesh for data-parallel GAM fitting.

    GAMs are data-parallel problems: X is large in n, moderate in p.
    We shard X rows across devices. No model parallelism needed
    for typical GAM sizes.

    v1.12: Replicated solve is a conscious inefficiency. Every device
    independently solves the same (p,p) system — O(p³) wasted compute
    per extra device. This is acceptable because:
    - For p ≤ 2000, Cholesky takes <10ms — negligible vs XtWX formation
    - It avoids a broadcast of β (which would add latency)
    - It keeps the XLA program simple (no asymmetric device roles)

    For p > 3000 (single-host) or p > 2000 (multi-host), this waste
    becomes material. See auto_select_distributed_mode() for gates.
    """
    if devices is None:
        devices = jax.devices()
    return Mesh(devices, axis_names=('data',))


def shard_gam_data(X, y, weights, offset, mesh):
    """
    Shard GAM data for distributed fitting.

    X: (n, p) — row-sharded across 'data' axis
    y: (n,)   — sharded to match X rows
    β, S_λ:   — replicated (all devices get full copy)

    Returns sharded jax.Arrays. After this call, the same
    pirls_step_jax function works without modification.
    """
    data_spec = NamedSharding(mesh, P('data', None))
    vec_spec = NamedSharding(mesh, P('data',))

    X_sharded = jax.device_put(X, data_spec)
    y_sharded = jax.device_put(y, vec_spec)
    wt_sharded = jax.device_put(weights, vec_spec)
    off_sharded = jax.device_put(offset, vec_spec)

    return X_sharded, y_sharded, wt_sharded, off_sharded
```

### 16.3 PIRLS Under SPMD: Full JIT, Full Autodiff

The PIRLS step function from Section 4.2 works unchanged under sharding. JAX's compiler traces through the matmuls, sees that `WX.T @ WX` contracts over a sharded axis, and inserts an all-reduce:

```python
# fitting/_pirls_jax.py — UNCHANGED from single-GPU version

@partial(jax.jit, static_argnums=(5,))
def pirls_step_jax(X, y, beta, S_lambda, family_params, family_type):
    """
    One PIRLS iteration, fully JIT-compiled.

    When X is row-sharded across a Mesh:
      - eta = X @ beta             → local matmul per device (β is replicated)
      - mu, W, z                   → local elementwise (no communication)
      - WX.T @ WX                  → local matmul + XLA-inserted all-reduce
      - WX.T @ (W_sqrt * z)        → local matmul + XLA-inserted all-reduce
      - jnp.linalg.solve(H, g)    → replicated solve (all devices, same result)
      - jax.grad for extended families → works on each device's local data

    No distributed-specific code. Same function for 1 GPU or 100.
    """
    eta = X @ beta
    mu = _link_inverse(eta, family_type)         # AD works here (local per device)
    W = _working_weights(mu, family_params, family_type)
    z = _working_response(y, mu, eta, family_params, family_type)

    W = jnp.clip(W, 1e-10, 1e10)
    W_sqrt = jnp.sqrt(W)

    WX = W_sqrt[:, None] * X
    XtWX = WX.T @ WX + S_lambda   # ← XLA inserts all-reduce here
    XtWz = WX.T @ (W_sqrt * z)    # ← and here
    beta_new = jnp.linalg.solve(XtWX, XtWz)

    dev = _deviance(y, mu, family_params, family_type)
    return beta_new, mu, eta, dev, W, XtWX
```

**Why this works for autodiff:**

The `pirls_loop_jax` (Section 4.2) wraps `pirls_step_jax` in `jax.lax.while_loop`. The entire loop — including the all-reduces that XLA inserts — compiles to a single XLA program. `jax.grad` and `jax.hessian` can differentiate through this program, including through the collective operations. This means:

- **REML autodiff works end-to-end.** `jax.grad(reml_criterion_jax)` differentiates through the sharded XtWX formation. No implicit function theorem workaround needed (though it remains available as a fallback for the out-of-core path).
- **Extended family AD works on every device.** Each device evaluates the family's log-likelihood on its local data shard. Standard `jax.grad` (and Tweedie's `custom_jvp`) executes locally — no special distributed handling.
- **fREML works.** The fast REML update needs derivatives of β* w.r.t. λ. In the SPMD path, these flow through the JIT-compiled while_loop naturally.

**Convergence decision broadcast (v1.15, critical correctness fix):** In JAX SPMD, the deviance `pen_dev` is computed via an all-reduce. Within a single compiled XLA program, the all-reduce is deterministic — all devices get the same bit-for-bit result. The `while_loop`'s `cond_fn` evaluates identically on every device because it operates on replicated scalars (deviance, β norm, iteration count) that are all post-all-reduce.

However, step-halving adds a second decision point: `pen_dev_try <= pen_dev_prev`. If the step-halving check uses any locally-computed quantity that differs across devices (e.g., a deviance computed before the all-reduce), devices can diverge in their acceptance decision, causing some to accept the step and others to reject it. In `lax.while_loop`, divergent `cond_fn` results across SPMD devices cause a collective deadlock.

The invariant: **every scalar used in `cond_fn` and step-halving decisions must be replicated (post-all-reduce)**. In the current design, this holds because deviance is accumulated via the same XtWX all-reduce path and the solve is replicated. But this invariant must be explicitly tested:

```python
# tests/distributed/test_spmd_convergence.py

def test_step_halving_decision_replicated():
    """
    Verify that all devices reach the same accept/reject decision
    for step-halving. Failure here → collective deadlock in production.

    Runs identical data on 2+ devices, checks that the deviance
    comparison (pen_dev_try <= pen_dev_prev) produces the same
    boolean on every device at every iteration.
    """
    ...

def test_convergence_cond_replicated():
    """
    Verify that cond_fn in while_loop evaluates identically on
    all devices. Checks: deviance, beta_norm, iteration count
    are all replicated scalars.
    """
    ...
```

**If future code changes introduce locally-computed decision variables:** the fix is to broadcast the decision boolean from device 0 via `jax.lax.psum` or by computing it only on replicated quantities. The current design avoids this need, but any refactoring of the PIRLS loop must preserve the "all decision variables are replicated" invariant.

**SPMD path invariants (v1.12):**

The SPMD PIRLS path requires all of the following to hold. Violations produce hangs, incorrect results, or crashes:

| Invariant | Requirement | Consequence of violation |
|---|---|---|
| **Fixed device count** | Mesh shape set once at startup, never changes | XLA compiles for specific device count; mismatch → crash |
| **Fixed topology** | No device join/leave during fit | Collective hang (waiting for absent device) |
| **All processes same program** | Same `pirls_step_jax`, same control flow, same iteration count | Collective deadlock (processes at different barriers) |
| **dtype = float64** | All arrays in SPMD path must be float64 | XtWX all-reduce + Cholesky at float32 loses ~4 digits; insufficient for REML gradients. `jax.config.update("jax_enable_x64", True)` enforced at init. |
| **X row-sharded, β/S_λ replicated** | Sharding annotations as specified in Section 16.2 | Wrong sharding → silent wrong results (XtWX not summed correctly) |
| **No elastic membership** | Workers cannot be added or removed mid-fit | See Section 16.4 lifecycle invariants |
| **Identical setup outputs across hosts (v1.14)** | All processes must produce the same column layout, factor-level ordering, constraint absorption, and `CoefficientMap` after Phase 1. For `FactorBySmooth` (Section 5.7), this means: same factor levels in the same order, same block-to-column mapping, same null-space constraints. Enforced by: (a) coordinator broadcasts `SetupManifest` including knots, factor-level ordering, and constraint spec (Section 16.8), (b) each process verifies post-assembly column count and manifest checksum via `verify_local_assembly()` (Section 16.8), (c) mismatch → immediate fail-fast error, never silent. | Silent catastrophic error if verification is skipped: devices have "same shapes, different column semantics" — XtWX all-reduce sums incompatible matrices, producing garbage β with no detectable signal. |

**Determinism in SPMD (v1.12):**

XLA's all-reduce is deterministic within a single compiled program execution: same XLA graph + same device count + same topology → same reduction tree → same numerical result. However, this does NOT guarantee determinism across:

- Different device counts (different reduction tree)
- Different XLA/JAX versions (compiler may choose different reduction strategy)
- Different hardware topologies (ring vs tree vs recursive halving)

Cross-compilation determinism still requires `set_deterministic(True)` + pinned JAX/XLA/driver versions, as specified in Section 4.5. The SPMD path does not change the determinism contract — it inherits it.

### 16.4 Multi-Host Clusters: Ray Bootstraps, JAX Computes

For clusters spanning multiple hosts (each with one or more GPUs), the architecture separates concerns:

- **Ray** handles process orchestration: launching workers, fault tolerance, resource allocation, data loading.
- **JAX** handles computation: JIT compilation, SPMD parallelism, inter-device communication, autodiff.

Ray's `JaxTrainer` initializes `jax.distributed` on each worker, then each worker runs the same JAX program:

```python
# distributed/ray_launcher.py

import ray
from ray.train.v2.jax import JaxTrainer
from ray.train import ScalingConfig


def fit_distributed(formula, data_path, family, n_workers, gpus_per_worker=1):
    """
    Launch a distributed GAM fit across a Ray cluster.

    Ray handles:
    - Process placement and lifecycle
    - jax.distributed.initialize() on each worker
    - Data loading (each worker loads its shard)

    JAX handles:
    - SPMD compilation of the PIRLS loop
    - Inter-device all-reduce for XtWX/XtWz
    - Autodiff, JIT — all native (extended family AD works per-device)
    """

    def train_func():
        """Runs on each worker in SPMD mode."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        # Ray has already called jax.distributed.initialize()
        # All devices across all workers are visible as one global Mesh
        mesh = jax.make_mesh((jax.device_count(),), ('data',))

        # Each worker loads its LOCAL shard of the data
        local_X, local_y = _load_local_shard(data_path, jax.process_index())

        # Convert local arrays to globally-sharded jax.Arrays
        from jax.experimental.multihost_utils import host_local_array_to_global_array
        X_global = host_local_array_to_global_array(local_X, mesh, P('data', None))
        y_global = host_local_array_to_global_array(local_y, mesh, P('data',))

        # Setup phase (Phase 1): knots, penalties, constraints
        # Runs identically on each worker (deterministic given same formula)
        smooths, S_list, coef_map = _setup_model(formula, family, X_global, y_global)

        # Fit phase (Phase 2): PIRLS + REML, fully JIT-compiled SPMD
        # This is the SAME pirls_loop_jax from Section 4.2.
        # JAX compiles it as one XLA program spanning all devices.
        S_lambda = _build_penalty(S_list, log_lambda_init)
        result = pirls_loop_jax(
            X_global, y_global, family_params, family_type,
            S_lambda, beta_init, max_iter=100, tol=1e-7,
        )

        # REML outer loop — also SPMD, also JIT-able
        # jax.grad(reml_criterion_jax) works through the sharded computation
        for outer_iter in range(max_outer):
            reml_grad = jax.grad(reml_criterion_jax)(log_lambda, ...)
            log_lambda = _newton_step(log_lambda, reml_grad, reml_hess)
            result = pirls_loop_jax(X_global, y_global, ..., S_lambda)

        # Return result (replicated on all devices, same value)
        ray.train.report({"converged": bool(result.converged)})

    trainer = JaxTrainer(
        train_func,
        scaling_config=ScalingConfig(
            num_workers=n_workers,
            resources_per_worker={"GPU": gpus_per_worker},
        ),
    )
    return trainer.fit()
```

**Critical constraint:** All processes must run the same JAX operations in the same order. This is natural for PIRLS (every iteration is the same function applied to the same globally-sharded arrays), but means:

- Convergence decisions must be identical on all processes (they will be — the converged XtWX/β are replicated).
- No process can skip an iteration or take a different code path. `jax.lax.while_loop` guarantees this (same cond_fn evaluated on replicated state).
- If one process crashes, all will hang unless detected and killed.

**Multi-host lifecycle invariants (v1.12):**

Real distributed systems fail at process lifecycle boundaries, not in the math. These invariants are mandatory:

| Invariant | Requirement | Failure mode if violated |
|---|---|---|
| **Exactly-once init** | Each process calls `jax.distributed.initialize()` exactly once, before any JAX computation. Ray's `JaxTrainer` handles this. | Double init → crash. Missing init → process invisible to collective. |
| **No elastic membership** | Worker count is fixed for the entire fit. No workers join or leave mid-fit. This is a hard constraint of JAX's SPMD model, not a PyMGCV limitation. | Added worker → not part of compiled collective → hang. Lost worker → collective waits forever. |
| **No worker restart** | If a worker dies, the entire fit is aborted. There is no checkpoint/resume within a single PIRLS+REML optimization. | Restarted worker has stale XLA program / different compilation → silent corruption or hang. |
| **Straggler = hang** | If one worker is slow (GC, preemption, network), all others block at the next collective. Ray's health monitor should detect this and kill the job after a timeout. | Unbounded wait at all-reduce. |
| **Clean shutdown** | All processes must complete the `lax.while_loop` (converge or hit max_iter) before any process exits. | Early exit → other processes hang at next collective. |

```python
# distributed/ray_launcher.py — lifecycle management

def _validate_ray_cluster(scaling_config):
    """Pre-flight checks before launching distributed fit."""
    n_workers = scaling_config.num_workers
    if n_workers < 2:
        raise ValueError("Multi-host requires ≥ 2 workers")

    # Check all workers are available (no partial cluster)
    available = ray.available_resources().get("GPU", 0)
    needed = n_workers * scaling_config.resources_per_worker.get("GPU", 1)
    if available < needed:
        raise RuntimeError(
            f"Cluster has {available} GPUs, need {needed}. "
            f"Partial clusters are not supported (no elastic membership)."
        )
```

**What is NOT supported (and won't be):** Elastic training (adding/removing workers mid-fit), checkpoint-resume across different cluster sizes, heterogeneous device types within one fit (e.g., mixing A100 and V100).

### 16.5 Out-of-Core: Data Larger Than Aggregate Device Memory

When data exceeds total device memory (e.g., n=1B rows, 10 GPUs with 80GB each), the SPMD approach doesn't work — X can't be sharded because it won't fit. This is the **only** case that needs a Python outer loop and the `StatisticsProvider` abstraction.

```python
# distributed/chunked_jax_provider.py

import jax
import jax.numpy as jnp


class ChunkedJAXProvider:
    """
    Out-of-core provider: streams data chunks through JIT-compiled
    JAX computation, accumulates sufficient statistics on device.

    Unlike v1.0-v1.10 DaskProvider (NumPy workers), each chunk is
    processed by a JIT-compiled JAX function. Extended family AD works.
    Unlike SPMD mode, the outer accumulation loop is Python.
    This means: no autodiff through the full PIRLS loop.

    For REML: uses implicit function theorem (Section 8.x) rather
    than differentiating through PIRLS iterations. This only needs
    the converged H, β*, and penalty structure.
    """

    def __init__(self, data_path, smooths, chunk_size="256MB"):
        self._data_path = data_path
        self._smooths = smooths
        self._chunk_size = chunk_size
        # Pre-compile the chunk computation function
        self._chunk_fn = jax.jit(_compute_chunk_stats_jax, static_argnums=(4,))

    def compute_iteration_stats(self, beta, family_params, family_type, wt):
        """
        Stream chunks from disk, compute per-chunk stats on GPU,
        accumulate on device. Only p×p accumulator lives in memory
        at steady state (plus one chunk).
        """
        XtWX_acc = jnp.zeros((self._p, self._p))
        XtWz_acc = jnp.zeros((self._p,))
        dev_acc = jnp.array(0.0)
        n_total = 0

        for chunk_X, chunk_y, chunk_wt in self._stream_chunks():
            # JIT-compiled: jax.grad works, GPU-accelerated
            stats = self._chunk_fn(
                chunk_X, chunk_y, beta, family_params, family_type
            )
            XtWX_acc = XtWX_acc + stats.XtWX
            XtWz_acc = XtWz_acc + stats.XtWz
            dev_acc = dev_acc + stats.deviance
            n_total += stats.n_obs

        return IterationStatistics(
            XtWX=XtWX_acc, XtWz=XtWz_acc,
            deviance=dev_acc, n_obs=n_total, ...
        )


@jax.jit
def _compute_chunk_stats_jax(X_chunk, y_chunk, beta, family_params, family_type):
    """
    Per-chunk sufficient statistics. JIT-compiled, runs on GPU.
    Extended family AD (and Tweedie's custom_jvp) works here — this is pure JAX.
    """
    eta = X_chunk @ beta
    mu = _link_inverse(eta, family_type)
    W = _working_weights(mu, family_params, family_type)
    z = _working_response(y_chunk, mu, eta, family_params, family_type)
    W = jnp.clip(W, 1e-10, 1e10)
    W_sqrt = jnp.sqrt(W)
    WX = W_sqrt[:, None] * X_chunk
    return ChunkStats(
        XtWX=WX.T @ WX,
        XtWz=WX.T @ (W_sqrt * z),
        deviance=_deviance(y_chunk, mu, family_params, family_type),
        n_obs=X_chunk.shape[0],
    )
```

**REML in the out-of-core path (v1.12 clarification):**

Because the PIRLS outer loop is Python (not JIT), `jax.grad` can't trace through it. REML uses the **implicit function theorem** instead.

**What is differentiated:** The exact REML objective at the converged β*, not a per-chunk approximation. At convergence, PIRLS has produced:
- `H = XtWX + S_λ` (accumulated across all chunks — the exact full-data matrix)
- `β*` (the converged coefficients)
- `pen_deviance` (the exact full-data penalized deviance)

The implicit function theorem gives `dβ*/dλ` from these converged quantities alone. Chunks affect only how H was accumulated (summation order), not the mathematical function being differentiated. The REML criterion `V(λ) = pen_dev + log|H| - log|S⁺|` is the exact same objective as in the SPMD path — just evaluated on identically-accumulated statistics.

```python
def implicit_dbeta_dlambda(H_factor, S_list, beta, lambdas):
    """
    dβ*/d(log λ_j) = -H⁻¹ (λ_j S_j β*)

    From the fixed-point condition ∂L/∂β = 0 at convergence.
    Only needs the converged H, β*, and penalty structure —
    NOT tracing through PIRLS iterations.

    This is Wood (2004, 2011): the same approach mgcv uses.
    Works regardless of how H was formed (SPMD, chunked, etc.).

    CRITICAL (v1.15): The H_factor used here MUST be the exact same
    factorization used in the forward solve — including any jitter
    (Section 10.3 regularization_applied), pivoting strategy, and
    null-space handling. If the forward solve added ε·I to H for
    numerical stability, the IFT backward pass must use the
    regularized H (not the original), because β* was computed from
    the regularized system.

    Specifically:
      - If forward used jitter: H_factor = cholesky(XtWX + S_λ + ε·I)
        → IFT uses the SAME H_factor (already includes ε·I)
      - If forward dropped rank-deficient columns: H_factor is
        (p-r)×(p-r) → IFT computes dβ*/dλ in reduced space, then
        pads zeros for dropped columns
      - If forward used generalized inverse: do NOT naively invert H;
        use the same generalized inverse strategy

    Violation: using the "clean" H (without jitter) while β* came from
    the jittered system produces gradient error proportional to
    ε × cond(H), which can be catastrophically large for near-singular H.
    The regularization_applied field in FitDiagnostics records ε.
    """
    p = len(beta)
    n_smooth = len(lambdas)
    dbeta = jnp.zeros((p, n_smooth))
    for j in range(n_smooth):
        rhs = lambdas[j] * S_list[j] @ beta
        dbeta = dbeta.at[:, j].set(
            -jax.scipy.linalg.cho_solve(H_factor, rhs)
        )
    return dbeta
```

### 16.6 Architecture Summary

| Scale | Method | PIRLS loop | Solve | Autodiff | Extended family AD | p limit | Tier |
|---|---|---|---|---|---|---|---|
| Single GPU | `jax.jit` | Full JIT | Single device | Full | ✅ (jax.grad; Tweedie: custom_jvp) | 5000 | 1 |
| Multi-GPU, one host (p ≤ 3000) | `jax.sharding` + `Mesh` | Full JIT (SPMD) | Replicated | Full | ✅ per device | 3000 | 2 |
| Multi-GPU, one host (p > 3000) | `jax.sharding` + `Mesh` | Full JIT (SPMD) | Device 0 + broadcast | Full | ✅ per device | 5000 | 2 |
| Multi-host cluster | `jax.distributed` + Ray | Full JIT (SPMD) | Replicated | Full | ✅ per device | 2000 (IB) / 1500 (Eth) | 2–3 |
| Out-of-core (data > memory) | `ChunkedJAXProvider` | JIT per chunk | Single device | Implicit fn thm | ✅ per chunk | 5000 | 3 |

**What changed from v1.0–v1.10:**

| Old (v1.0–v1.10) | New (v1.11) | Why |
|---|---|---|
| `DaskProvider` (NumPy workers, Python coordinator) | `jax.sharding` SPMD or `ChunkedJAXProvider` (JAX workers) | NumPy workers broke JIT, autodiff, extended family AD |
| `RayProvider` (NumPy workers, Python coordinator) | `JaxTrainer` bootstraps `jax.distributed`, pure JAX SPMD | Ray orchestrates processes; JAX owns all computation |
| `StatisticsProvider` needed for all distributed paths | Only needed for out-of-core (data > device memory) | SPMD path uses same `pirls_step_jax` as single-GPU |
| `deterministic_reduce` with Kahan summation | XLA's all-reduce (deterministic within single compilation + fixed topology; see Section 16.3 caveats) | No manual reduction code needed for SPMD path |
| Python coordinator round-trip per PIRLS iteration | No coordinator — all devices run same XLA program | Eliminates serialization latency |
| Extended family AD unavailable on distributed workers | Works everywhere (all workers run JAX) | NB, Tweedie, Beta work in distributed mode |

**The "missing middle" — distributed sparse (v1.15):** The architecture has a deliberate gap between "dense SPMD" (multi-GPU, p ≤ 3000) and "sparse single-host" (Sparse-CPU, any p but one node). High-cardinality random effects (`s(user_id, bs='re')` with 500k users) or massive factor-smooth interactions are too sparse for SPMD (densification would OOM) and potentially too large for single-host RAM. This is the standard "big data GAM" use case, and PyMGCV v1.0 cannot fit it.

This is acceptable for Tier 1–2 but must be addressed for Tier 3. Potential future paths: (a) distributed conjugate gradient solver that keeps X sparse across workers, (b) block-diagonal exploitation where independent factor levels are solved on separate workers, (c) stochastic/minibatch approaches that avoid forming the full XtWX. The current architecture provides a hook for (b): `FactorBySmooth`'s block-diagonal structure (Section 5.7) means level-blocks are independent given λ, so a future "block-parallel" mode could solve each level's sub-problem on a separate device. This is not designed or specified; it's an architectural affordance.

**Float64 requirement is a product constraint (v1.15):** The mandatory float64 on GPU paths (SPMD invariant, Section 16.3) is correct for numerical stability — mgcv-style inference needs it. But it is a significant performance constraint: consumer GPUs (RTX 3090, 4090) have ~1/32 FP64 throughput vs FP32, and some accelerators (TPU v3, older AMD MI-series) have limited or no FP64 support. "GPU acceleration" means "fast FP64 GPUs" — data center cards (A100, H100, MI250X). The doc should not market this as general GPU support. A future "reduced precision mode" (FP32 PIRLS with FP64 REML gradients) is mathematically possible but not designed.

**fREML auto-switch alignment with R (v1.15):** The auto-switch from Newton REML to fREML at n_smooth > 50 (Section 16.7) introduces a behavioral cliff: adding one factor level can change results slightly (fREML is an approximation). R's mgcv also switches methods based on model size, but at different thresholds and with different approximations (`bam()` uses fREML by default; `gam()` uses Newton REML). PyMGCV's switch points do NOT align with R's, so the "correctness vs R" tests (Section 18.1) must account for this: when comparing fREML results against R's Newton REML, the tolerance class is LOOSE (not MODERATE). The `lambda_strategy_reason` field in `GAMResult` surfaces the switch so users understand the source of any difference.

### 16.7 Communication Cost Model

The SPMD approach's communication cost per PIRLS iteration:

| Operation | Size | Communication | Notes |
|---|---|---|---|
| XtWX all-reduce | p² × 8 bytes | One all-reduce across all devices | XLA uses ring all-reduce or tree all-reduce depending on topology |
| XtWz all-reduce | p × 8 bytes | One all-reduce (pipelined with XtWX) | Negligible vs XtWX |
| β broadcast | p × 8 bytes | None (replicated from all-reduce result) | Already on all devices after solve |
| Total per iteration | ~p² × 8 bytes | ~2 all-reduces | For p=2000: 32MB. NVLink: <1ms. Ethernet: ~10ms. |

**Scaling limits (v1.14: enforcement gates reflecting dense X and dense S_λ):**

These limits assume float64 (mandatory on SPMD, see invariant table in Section 16.3) and default (non-deterministic) mode. Deterministic mode (`set_deterministic(True)`) may reduce throughput by 10–30% due to `--xla_gpu_deterministic_ops`, effectively lowering the practical n limit.

The p limits account for **three** dense (p, p) arrays on every device: XtWX, S_λ (dense even when structurally block-diagonal, per Section 16.2), and the Cholesky factor. Total replicated memory per device: `3 * p² * 8` bytes.

| Topology | p limit | λ strategy (auto) | Replicated mem/device at p limit | n limit | Bottleneck |
|---|---|---|---|---|---|
| Single host, NVLink (4–8 GPUs) | 3000 | REML (≤50), fREML (51–200), F-S (>200) | 216MB | ~1B (aggregate HBM) | p³ replicated solve time |
| Multi-host, InfiniBand (16–64 GPUs) | 2000 | REML (≤50), fREML (51–200), F-S (>200) | 96MB | ~10B | All-reduce bandwidth for p² |
| Multi-host, Ethernet (16–64 GPUs) | 1500 | REML (≤50), fREML (51–200), F-S (>200) | 54MB | ~10B | Network bandwidth |

**Factor-by routing rules (v1.14):**

Factor-by smooths (Section 5.7) push both p and n_smooth. The interaction determines the path. The "Route" column now reflects the structured `DistributedModeSelection` return:

| Scenario | p | n_smooth | Route | λ strategy | Rationale |
|---|---|---|---|---|---|
| `s(x, by=fac)`, 5 levels, k=20 | +100 | +5 | SPMD | REML | Both dimensions small; no issue |
| `s(x, by=fac)`, 50 levels, k=20 | +1000 | +50 | SPMD | REML | p within gate; n_smooth at REML/fREML boundary |
| `s(x, by=state)`, 50 levels, k=50 | +2500 | +50 | SPMD single-solve (single-host) or **error** (multi-host) | REML | p exceeds multi-host gate |
| 3 factor-by smooths × 50 levels × k=20 | +3000 | +150 | SPMD single-solve | fREML (auto) | p at gate ceiling; n_smooth in fREML range |
| 5 factor-by smooths × 50 levels × k=20 | +5000 | +250 | **Sparse-CPU or chunked** | Fellner-Schall (auto) | p exceeds all SPMD gates |
| 10 factor-by smooths × 5 levels × k=10 | +500 | +50 | SPMD | REML | Small p, moderate n_smooth — the n_smooth gate catches this if levels grow |

The key insight: factor-by with many levels and moderate basis dimension will hit the **p gate** before the n_smooth gate in most practical cases (because p grows as `n_levels × k` while n_smooth grows as `n_levels`). The n_smooth gate catches the remaining case: many factor-by terms with small bases where p is modest but the REML outer loop is expensive. These are heuristic routing rules, not performance guarantees — actual throughput depends on hardware, data distribution, and model structure.

```python
# distributed/selector.py

@dataclass
class DistributedModeSelection:
    """
    v1.14: Structured return from path selection. Every behavioral
    decision is explicit and inspectable — no silent mode changes.
    """
    spmd_mode: str              # 'spmd', 'spmd_single_solve', 'chunked', 'sparse_cpu'
    lambda_strategy: str        # 'reml', 'freml', 'fellner_schall'
    lambda_strategy_reason: str # Why this strategy was chosen (empty if user-specified)
    p: int
    n_smooth: int
    dense_bytes: int


def auto_select_distributed_mode(p, n, n_smooth, mesh, interconnect="nvlink",
                                  user_method=None):
    """
    v1.14: Gate SPMD mode based on comms model AND outer-loop cost.
    Returns DistributedModeSelection (never silently changes behavior).

    This is NOT advisory — it enforces the scaling limits.

    Gates on TWO dimensions:
      - p: determines all-reduce cost (p² bytes) and solve cost (p³ FLOPS)
      - n_smooth: determines REML outer-loop cost (Hessian is n_smooth × n_smooth)

    Factor-by smooths (Section 5.7) can push both dimensions high
    simultaneously: K levels × p_base columns, K penalties.

    Lambda strategy selection:
      - If user explicitly set method='REML'/'fREML'/'fellner_schall',
        that choice is respected (with a warning if n_smooth is large).
      - If user left method=None (auto), the selector chooses:
        n_smooth ≤ 50:  Newton REML (exact, fast for small dimension)
        50 < n_smooth ≤ 200: fREML (avoids full Hessian)
        n_smooth > 200: Fellner-Schall (no Hessian at all, O(n_smooth) per step)

    These thresholds are anchored to cost:
      - Newton REML: O(n_smooth³) per outer iteration. At 50: 125K FLOPS (trivial).
        At 200: 8M FLOPS (noticeable on every device). At 500: 125M FLOPS.
      - fREML: O(n_smooth²) per outer iteration (diagonal + rank-1 updates).
      - Fellner-Schall: O(n_smooth) per outer iteration (no Hessian).
    """
    n_hosts = _count_hosts(mesh)
    is_multi_host = n_hosts > 1
    dense_bytes = n * p * 8

    # Gate 1: p limits for SPMD path (communication cost)
    if is_multi_host and interconnect == "ethernet" and p > 1500:
        raise ValueError(
            f"SPMD PIRLS not supported for p={p} on Ethernet multi-host "
            f"(p limit 1500). Reduce basis dimension (lower k in s(..., k=)) "
            f"or use single-host multi-GPU."
        )
    if is_multi_host and p > 2000:
        raise ValueError(
            f"SPMD PIRLS not supported for p={p} on multi-host "
            f"(p limit 2000). Reduce basis dimension or use single-host."
        )

    # Gate 2 (v1.14): lambda strategy selection
    if user_method is not None:
        # User explicitly chose — respect it, but warn if costly
        lambda_strategy = user_method
        lambda_reason = ""
        if user_method == "REML" and n_smooth > 50:
            import warnings
            warnings.warn(
                f"User-specified method='REML' with {n_smooth} smoothing "
                f"parameters. Newton REML outer loop is O({n_smooth}³) per "
                f"iteration. Consider method='fREML' or 'fellner_schall'."
            )
    else:
        # Auto-select based on n_smooth
        if n_smooth <= 50:
            lambda_strategy = "REML"
            lambda_reason = ""
        elif n_smooth <= 200:
            lambda_strategy = "fREML"
            lambda_reason = (
                f"n_smooth={n_smooth} > 50: auto-selected fREML to avoid "
                f"O({n_smooth}³) Newton Hessian. Override with method='REML'."
            )
        else:
            lambda_strategy = "fellner_schall"
            lambda_reason = (
                f"n_smooth={n_smooth} > 200: auto-selected Fellner-Schall "
                f"(no Hessian, O({n_smooth}) per step). Override with "
                f"method='fREML' or method='REML' (expensive)."
            )

    # Gate 3: for large p, replicated solve wastes compute.
    # Use single-device solve + broadcast instead.
    if p > 3000:
        spmd_mode = 'spmd_single_solve'
    else:
        # Gate 4: data too large for aggregate device memory → chunked
        total_device_bytes = _total_device_memory(mesh)
        data_bytes = n * p * 8 * 2  # X + working arrays
        if data_bytes > 0.7 * total_device_bytes:
            spmd_mode = 'chunked'
        else:
            spmd_mode = 'spmd'

    result = DistributedModeSelection(
        spmd_mode=spmd_mode,
        lambda_strategy=lambda_strategy,
        lambda_strategy_reason=lambda_reason,
        p=p, n_smooth=n_smooth, dense_bytes=dense_bytes,
    )

    # Surface the decision — always logged, never silent
    if lambda_reason:
        import warnings
        warnings.warn(lambda_reason)

    return result
```

**v1.14: The `lambda_strategy_reason` field is mandatory in `GAMResult`.** If the selector auto-switched from Newton REML to fREML or Fellner-Schall, `GAMResult.lambda_strategy_reason` contains the explanation and the override instruction. `summary()` prints it. This ensures the behavioral change is never silent — the user always sees what happened and how to revert it.

For `'spmd_single_solve'` mode (p > 3000 on single-host): XtWX formation uses all-reduce as usual, but `jnp.linalg.solve` runs on device 0 only and β is broadcast. This avoids N redundant O(p³) solves at the cost of one p-vector broadcast (~24KB for p=3000, negligible).

### 16.8 Distributed Knot Placement

Knot selection (max-min distance subsample) is a **Phase 1 (setup) operation**. Phase 1 is CPU/NumPy (Section 4.4). In distributed mode, data is sharded across devices, but we do NOT run knot selection as a JAX program — that would violate the Phase 1/Phase 2 boundary.

Instead, each process contributes a local subsample, and a coordinator runs the final selection on CPU:

```python
def distributed_knot_selection(data_path, variable_name, n_knots, process_index, n_processes):
    """
    Distributed knot selection that respects the Phase 1 (CPU/NumPy) boundary.

    v1.12: Does NOT use sharded jax.Arrays or JAX collective ops.
    Each process loads its local data shard (NumPy), subsamples,
    then process 0 gathers candidates and runs final selection.

    This runs BEFORE jax.device_put — no JAX arrays exist yet.
    """
    # Step 1: each process loads its local shard and subsamples (NumPy)
    local_data = _load_local_column(data_path, variable_name, process_index)
    local_candidates = _maxmin_subsample_numpy(local_data, n_knots * 4)

    # Step 2: gather candidates to process 0 via Ray/MPI (NOT JAX collectives)
    # Candidates are small: 4 * n_knots floats, typically < 1KB
    all_candidates = _gather_to_coordinator(local_candidates, process_index)

    # Step 3: process 0 runs final max-min selection (CPU, NumPy)
    if process_index == 0:
        knots = _maxmin_distance_subsample(np.concatenate(all_candidates), n_knots)
    else:
        knots = None

    # Step 4: broadcast knots to all processes (Ray/MPI, NOT JAX)
    knots = _broadcast_from_coordinator(knots, process_index)
    return knots  # NumPy array, same on all processes
```

This preserves the clean boundary: knots are computed in Phase 1 (CPU/NumPy), then used during basis construction (also Phase 1), and the resulting X matrix is `jax.device_put` into the SPMD mesh for Phase 2. No JAX arrays or collectives during setup.

**v1.14: Factor-level ordering in distributed setup.** When the model contains factor-by smooths (Section 5.7), the coordinator must also broadcast the canonical factor-level ordering alongside knots. Each process may see a different subset of factor levels in its local data shard. If processes independently compute `np.unique(fac)` on their local data, they can produce different level orderings (or miss levels entirely), which means their locally-assembled X matrices have different column semantics — catastrophic for the XtWX all-reduce (see SPMD invariant "Identical setup outputs across hosts"). The fix: process 0 computes the global level ordering from the gathered factor columns, broadcasts it, and all processes use that ordering in `FactorBySmooth.setup()`. This is the same gather/broadcast pattern as knot selection, adds negligible communication (a list of level labels), and runs entirely in Phase 1 (NumPy/Ray, no JAX).

**Coordinator broadcast contract (v1.14).** The coordinator broadcasts a `SetupManifest` — not just knots and level orderings, but the full specification needed to assemble X identically on every host:

```python
# distributed/setup_manifest.py

@dataclass(frozen=True)
class SetupManifest:
    """
    Everything a non-coordinator host needs to assemble X
    identically to the coordinator. Broadcast once during Phase 1.

    This is the single source of truth for column layout.

    v1.15: Explicit hash scope. The checksum covers ALL fields that
    affect column semantics. If two hosts have the same checksum,
    their X matrices have the same column layout. Fields and their
    inclusion rationale:

    HASHED (affect column layout):
      knots              — different knots → different basis → different columns
      factor_levels      — different level ordering → columns in wrong blocks
      level_to_index     — redundant with factor_levels but cheap to verify
      empty_level_policy — 'keep' vs 'drop' changes column count
      constraint_spec    — different constraints → different effective columns
      total_p            — summary check
      smooth_term_order  — order of smooth terms in the model matrix
      basis_types        — different basis type → different columns

    NOT HASHED (don't affect column layout):
      uv_lock_hash       — checked separately (exact match, not hash)
      checksum itself    — obviously
    """
    knots: dict[str, np.ndarray]          # variable -> knot array
    factor_levels: dict[str, list[str]]   # by-variable -> ordered level list
    level_to_index: dict[str, dict]       # by-variable -> {level: block_index}
    empty_level_policy: str               # 'keep' or 'drop' (see below)
    constraint_spec: list[tuple]          # (term_index, constraint_type, ...)
    smooth_term_order: list[str]          # ordered list of smooth term labels
    basis_types: list[str]               # basis type per smooth term
    total_p: int                          # expected column count after assembly
    checksum: str                         # SHA-256 of hashed fields above
    # v1.16: uv.lock hash replaces custom version_pins dict.
    # If all hosts ran `uv sync --frozen` from the same uv.lock,
    # version divergence is impossible by construction.
    # This hash is a runtime check for the case where someone
    # didn't use --frozen or manually modified their environment.
    uv_lock_hash: str                     # SHA-256 of uv.lock file contents


def _compute_manifest_checksum(manifest):
    """
    Deterministic hash of all fields that affect column layout.
    Fields are serialized with sorted keys. Knot arrays are
    rounded to 15 significant digits to avoid platform-specific
    float formatting differences.
    """
    import hashlib, json
    payload = json.dumps({
        'knots': {k: [round(x, 15) for x in v.tolist()]
                  for k, v in sorted(manifest.knots.items())},
        'factor_levels': manifest.factor_levels,
        'level_to_index': manifest.level_to_index,
        'empty_level_policy': manifest.empty_level_policy,
        'constraint_spec': [str(c) for c in manifest.constraint_spec],
        'smooth_term_order': manifest.smooth_term_order,
        'basis_types': manifest.basis_types,
        'total_p': manifest.total_p,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _compute_uv_lock_hash():
    """
    v1.16: Hash the uv.lock file for cross-host verification.

    If all hosts used `uv sync --frozen` from the same uv.lock,
    this hash is identical everywhere and version verification is
    a single string comparison — no per-package iteration needed.

    Falls back to per-package version collection if uv.lock is
    not found (e.g., user installed via pip directly).
    """
    import hashlib
    from pathlib import Path

    # Look for uv.lock in standard locations
    for candidate in [
        Path.cwd() / "uv.lock",
        Path(__file__).parent.parent / "uv.lock",
    ]:
        if candidate.exists():
            content = candidate.read_bytes()
            return hashlib.sha256(content).hexdigest()

    # Fallback: hash key package versions individually
    # This is less reliable (doesn't catch transitive deps)
    # but works when uv.lock isn't available
    import jax, numpy, scipy
    version_str = f"jax={jax.__version__},numpy={numpy.__version__},scipy={scipy.__version__}"
    return hashlib.sha256(version_str.encode()).hexdigest()
```

**Verification handshake (v1.16).** Broadcasting the manifest is necessary but not sufficient. Each host must verify that its local assembly produced the expected result, AND that its environment matches:

```python
# distributed/setup_verify.py

def verify_local_assembly(X_local_shape, manifest, process_index):
    """
    Post-assembly verification on each host. Called after
    FactorBySmooth.setup() and model matrix assembly, before
    jax.device_put.

    Checks:
      1. Column count matches manifest.total_p
      2. Manifest checksum matches recomputed checksum
      3. Local factor levels are a subset of manifest levels
      4. (v1.16) uv.lock hash matches coordinator's hash

    Fails fast with a clear error. This catches:
      - Data filtering/dtype parsing differences across hosts
      - Stale data shards with different factor levels
      - Bugs in constraint absorption that produce different
        column counts
      - Environment drift between hosts
    """
    # Check 1: column count
    if X_local_shape[1] != manifest.total_p:
        raise RuntimeError(
            f"Process {process_index}: assembled X has "
            f"{X_local_shape[1]} columns, expected {manifest.total_p}. "
            f"Column layout divergence detected — aborting."
        )

    # Check 2: checksum integrity
    expected = _compute_manifest_checksum(manifest)
    if manifest.checksum != expected:
        raise RuntimeError(
            f"Process {process_index}: manifest checksum mismatch. "
            f"Setup metadata may have been corrupted in transit."
        )

    # Check 3 is performed during assembly (see FactorBySmooth.setup)

    # Check 4 (v1.16): uv.lock hash — single comparison replaces
    # per-package version iteration. If this passes, ALL package
    # versions (including transitive deps) are identical.
    local_lock_hash = _compute_uv_lock_hash()
    if local_lock_hash != manifest.uv_lock_hash:
        raise RuntimeError(
            f"Process {process_index}: environment mismatch. "
            f"uv.lock hash differs from coordinator "
            f"(local={local_lock_hash[:12]}..., "
            f"expected={manifest.uv_lock_hash[:12]}...). "
            f"Run `uv sync --frozen` on all hosts from the same uv.lock."
        )
```

**Empty-level and unseen-level policies (v1.14):**

| Situation | Policy | Rationale |
|---|---|---|
| **Level present globally but zero rows on this host** | `keep`: allocate the block columns, fill with zeros. The host's local X has the correct column count; zero rows contribute nothing to XtWX. | Dropping the block would change the column layout, breaking the all-reduce invariant. Zero-row blocks cost columns but no FLOPS. |
| **Level present in training but absent in new prediction data** | `keep`: the block columns exist in the coefficient vector. Prediction for that level returns zero contribution (no rows activate those columns). | Standard GAM prediction behavior — β for unused levels exists but isn't evaluated. |
| **Novel level in prediction data not seen during training** | `error`: raise with guidance to refit or use a factor-smooth (`bs="fs"`) model that shrinks toward a population smooth. | Factor-by has no sharing between levels. A novel level has no estimated β — prediction would be meaningless. This differs from `bs="fs"`, which has a population-level smooth to fall back on. |

### 16.9 API Integration

```python
# Multi-GPU fitting is exposed via the same gam()/bam() API.
# The only difference is the mesh argument.

# Single GPU (default):
model = pymgcv.gam("y ~ s(x1) + s(x2)", data=df, family="gaussian")

# Multi-GPU, one host (all visible GPUs):
model = pymgcv.gam(
    "y ~ s(x1) + s(x2) + te(x3, x4)",
    data=df,
    family="gaussian",
    mesh=jax.make_mesh((len(jax.devices()),), ('data',)),
)

# Multi-host via Ray (call from Ray train_func):
# jax.distributed.initialize() already called by JaxTrainer
mesh = jax.make_mesh((jax.device_count(),), ('data',))
model = pymgcv.gam("y ~ s(x1) + s(x2)", data=df_local, family="gaussian",
                    mesh=mesh)

# Out-of-core (data on disk, too large for device memory):
model = pymgcv.bam(
    "y ~ s(x1) + s(x2) + te(x3, x4)",
    data="/path/to/data.parquet",
    family="gaussian",
    method="fREML",
    chunk_size="512MB",
)
```

---

## 17. Public API Design

### 15.1 Main Entry Points

```python
# api.py

import pandas as pd
import numpy as np
from pymgcv.formula.parser import parse_formula
from pymgcv.formula.design import build_model_matrix
from pymgcv.fitting.pirls import pirls_fit
from pymgcv.fitting.newton import optimize_smoothing_parameters


class GAMResult:
    """Container for fitted GAM results."""
    def __init__(self):
        self.coefficients: np.ndarray = None
        self.fitted_values: np.ndarray = None
        self.linear_predictor: np.ndarray = None
        self.family: Family = None
        self.smooth_terms: list = []
        self.term_info: list = []
        self.Vp: np.ndarray = None      # Bayesian covariance
        self.Ve: np.ndarray = None      # Frequentist covariance
        self.scale: float = 1.0
        self.edf: np.ndarray = None     # Per-term EDF
        self.edf_total: float = 0
        self.smoothing_params: np.ndarray = None
        self.deviance: float = 0
        self.null_deviance: float = 0
        self.n: int = 0
        self.converged: bool = False
        self.method: str = "REML"
        self.formula: str = ""
        self.X: np.ndarray = None       # Model matrix
        self.offset: np.ndarray = None
        # v1.8: Fit diagnostics — surfaced in summary()
        self.n_iter: int = 0            # PIRLS iterations used
        self.instability_count: int = 0 # Cholesky fail + NaN + halving exhaust
        self.regularization_applied: float = 0.0  # Max jitter added to H
        self.execution_path: str = ""   # Which path actually ran
        # v1.15: Routing diagnostics — surfaced in summary()
        # These make every automatic decision explicit and reversible.
        self.execution_path_reason: str = ""   # WHY this path was selected
        self.lambda_strategy: str = ""         # REML / fREML / fellner_schall
        self.lambda_strategy_reason: str = ""  # WHY this strategy (empty if user chose)
        self.routing_diagnostics: dict = None  # Full routing metadata (p, n_smooth, dense_bytes, gates hit)

    def _routing_summary(self):
        """
        Human-readable routing explanation for summary() output.

        Example output:
          Execution path: spmd (multi-GPU, 4 devices)
            Reason: p=1200, n=5M, dense_bytes=48MB (within SPMD gates)
            Sparse-dominated model: multi-GPU may be slower than sparse-CPU for this workload.
          Lambda strategy: fREML (auto-selected)
            Reason: n_smooth=120 > 50: auto-selected fREML to avoid O(120³) Newton Hessian.
                    Override with method='REML'.
        """
        lines = [f"  Execution path: {self.execution_path}"]
        if self.execution_path_reason:
            lines.append(f"    Reason: {self.execution_path_reason}")
        lines.append(f"  Lambda strategy: {self.lambda_strategy}")
        if self.lambda_strategy_reason:
            lines.append(f"    Reason: {self.lambda_strategy_reason}")
        return "\n".join(lines)

    def predict(self, newdata=None, **kwargs):
        from pymgcv.predict.predict import predict_gam
        return predict_gam(self, newdata, **kwargs)

    def summary(self, **kwargs):
        from pymgcv.summary.summary import summary_gam
        return summary_gam(self, **kwargs)

    def plot(self, **kwargs):
        from pymgcv.plot.plot_gam import plot_gam
        return plot_gam(self, **kwargs)

    def check(self, **kwargs):
        from pymgcv.summary.diagnostics import gam_check
        return gam_check(self, **kwargs)


def gam(formula: str, data: pd.DataFrame | dict,
        family: str | Family = "gaussian",
        method: str = "REML",
        weights: np.ndarray = None,
        offset: np.ndarray = None,
        optimizer: str = "newton",
        scale: float = 0,
        select: bool = False,
        gamma: float = 1.0,
        knots: dict = None,
        sp: list = None,
        backend: str = "jax",
        device: str = "cpu",
        **kwargs) -> GAMResult:
    """
    Fit a Generalized Additive Model.

    Parameters
    ----------
    formula : str
        Model formula in R-style notation.
        Example: "y ~ s(x1) + s(x2) + x3 + te(x4, x5)"
    data : DataFrame or dict
        Data containing variables referenced in formula.
    family : str or Family
        Distribution family. One of: "gaussian", "binomial", "poisson",
        "Gamma", "inverse.gaussian", "nb", "tw", "betar", "ocat",
        "multinom", "zip", "cox.ph", "scat", "gaulss", "shash", etc.
    method : str
        Smoothness selection: "REML" (default), "ML", "GCV.Cp", "UBRE".
    optimizer : str
        Outer optimizer: "newton" (default), "efs" (extended Fellner-Schall),
        "bfgs".
    select : bool
        If True, add extra shrinkage penalties for variable selection.
    gamma : float
        Multiplier for effective degrees of freedom in GCV/UBRE.
    sp : list, optional
        Fixed smoothing parameters (one per penalty).
    backend : str
        Computation backend: "jax" (default) or "numpy".
    device : str
        Device: "cpu" (default) or "gpu".

    Returns
    -------
    GAMResult
        Fitted model object with predict(), summary(), plot() methods.

    Examples
    --------
    >>> import pymgcv
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> n = 1000
    >>> df = pd.DataFrame({
    ...     'x1': np.random.uniform(0, 1, n),
    ...     'x2': np.random.uniform(0, 1, n),
    ... })
    >>> df['y'] = np.sin(2 * np.pi * df['x1']) + np.random.normal(0, 0.2, n)
    >>>
    >>> model = pymgcv.gam("y ~ s(x1) + s(x2)", data=df)
    >>> model.summary()
    >>> model.plot()
    """
    from pymgcv.linalg.backend import configure
    configure(backend, device)

    # Parse formula
    parsed = parse_formula(formula)

    # Resolve family
    if isinstance(family, str):
        from pymgcv.families.registry import get_family
        family = get_family(family)

    # Convert data
    if isinstance(data, pd.DataFrame):
        data_dict = {col: data[col].values for col in data.columns}
    else:
        data_dict = data

    y = data_dict[parsed.response]
    n = len(y)

    # Build model matrix
    X, term_info, penalty_list, smooth_objects = build_model_matrix(parsed, data_dict)

    # Add extra shrinkage penalties if select=True
    if select:
        from pymgcv.penalties.selection import add_shrinkage_penalties
        penalty_list, smooth_objects = add_shrinkage_penalties(
            penalty_list, smooth_objects
        )

    # Smoothing parameter optimization
    if sp is not None:
        # Fixed smoothing parameters
        lambdas = np.array(sp)
        S_lambda = sum(lam * S for lam, S in zip(lambdas, penalty_list)
                       if S.nnz > 0)
        result = pirls_fit(X, y, family, S_lambda, weights, offset)
    else:
        lambdas, result = optimize_smoothing_parameters(
            X, penalty_list, y, family,
            weights=weights,
            method=method,
            optimizer=optimizer,
            gamma=gamma,
        )

    # Package results
    gam_result = GAMResult()
    gam_result.coefficients = result.coefficients
    gam_result.fitted_values = result.fitted_values
    gam_result.linear_predictor = result.linear_predictor
    gam_result.family = family
    gam_result.smooth_terms = smooth_objects
    gam_result.term_info = term_info
    gam_result.Vp = result.Vp
    gam_result.scale = family.scale_estimate(
        y, result.fitted_values, weights or np.ones(n),
        n, result.hat_matrix_trace
    )
    gam_result.smoothing_params = lambdas
    gam_result.deviance = result.deviance
    gam_result.n = n
    gam_result.converged = result.converged
    gam_result.method = method
    gam_result.formula = formula
    gam_result.X = X
    gam_result.offset = offset
    gam_result.edf_total = result.hat_matrix_trace

    return gam_result


def bam(formula: str, data: pd.DataFrame | dict,
        family: str | Family = "gaussian",
        method: str = "fREML",
        chunk_size: int = 10000,
        discrete: bool = True,
        n_threads: int = 1,
        **kwargs) -> GAMResult:
    """
    Fit a GAM to large datasets.

    Uses discretization, chunked processing, and fREML
    (fast REML via Fellner-Schall updates) for scalability.

    Handles millions of observations with O(p² + chunk_size × p) memory.
    """
    from pymgcv.fitting.bam_fit import bam_fit
    return bam_fit(formula, data, family, method, chunk_size, discrete, **kwargs)


def gamm(formula: str, data: pd.DataFrame | dict,
         family: str | Family = "gaussian",
         random: dict = None,
         correlation: CorrelationStructure = None,
         **kwargs) -> tuple:
    """
    Fit a GAMM via PQL.

    Returns (gam_result, lme_result) tuple mirroring R's gamm().
    """
    from pymgcv.fitting.gamm_fit import gamm_fit
    return gamm_fit(formula, data, family, random=random,
                    correlation=correlation, **kwargs)
```

---

## 18. Testing Strategy: Correctness Against R mgcv

### 18.1 Testing Philosophy

Every numerical result must be validated against R's mgcv to within specified tolerances. Testing proceeds in layers:

1. **Unit tests**: Individual components (basis functions, penalty matrices, link functions)
2. **Integration tests**: Full model fits on known datasets
3. **Regression tests**: Pre-computed R reference results stored as fixtures
4. **Property tests**: Mathematical invariants (positive definiteness, symmetry, rank)
5. **Fuzz tests**: Random model specifications tested for no-crash, no-NaN
6. **Performance benchmarks**: Runtime and memory comparisons

**Tolerance classes (v1.5):**

Without stratified tolerances, tests either fail from GPU/BLAS noise (too strict) or miss real regressions (too loose). Three tolerance classes, applied per-quantity and per-execution-path:

```python
# tests/tolerances.py

from dataclasses import dataclass

@dataclass
class ToleranceClass:
    rtol: float  # Relative tolerance
    atol: float  # Absolute tolerance
    label: str

STRICT   = ToleranceClass(rtol=1e-10, atol=1e-12, label="strict")
MODERATE = ToleranceClass(rtol=1e-6,  atol=1e-8,  label="moderate")
LOOSE    = ToleranceClass(rtol=1e-3,  atol=1e-5,  label="loose")
```

**Tolerance assignments per quantity:**

| Quantity | STRICT (CPU self-consistency) | MODERATE (GPU vs CPU) | LOOSE (vs R mgcv) |
|---|---|---|---|
| Link function g(μ), g⁻¹(η) | ✓ (1e-12) | ✓ (1e-10) | ✓ (1e-12) |
| Basis matrix X entries | ✓ (1e-10) | ✓ (1e-8) | ✓ (1e-6) — knot placement may differ |
| Penalty matrix S entries | ✓ (1e-10) | ✓ (1e-8) | ✓ (1e-6) |
| Deviance at convergence | ✓ (1e-10) | ✓ (1e-8) | — (1e-6) |
| Coefficients β | — | — (1e-6) | — (1e-4) — ill-conditioned models vary more |
| Smoothing parameters λ | — | — (1e-4) | — (1e-3) — REML is flat near optimum |
| EDF per term | — | — (1e-4) | — (1e-2) — sensitive to λ differences |
| p-values | — | — | — (1e-2) — notoriously unstable |
| AD gradient vs finite-diff | — (1e-5) | ✓ (1e-5) | N/A |

**Path-specific tolerance rules:**

- **Dense-GPU vs Sparse-CPU**: MODERATE. Different linear algebra paths (XLA vs CHOLMOD) produce different rounding. This is expected and correct.
- **Chunked vs Dense-GPU**: MODERATE. Chunked accumulation introduces summation-order differences.
- **Multi-device SPMD vs single GPU**: MODERATE. XLA all-reduce is deterministic within a single compilation + fixed device count + fixed topology (Section 16.3). Cross-compilation or topology changes may shift results within MODERATE tolerance.
- **Out-of-core (ChunkedJAXProvider) vs In-Memory**: MODERATE with `set_deterministic(True)` (Python accumulation order fixed); LOOSE with default chunk ordering.
- **PyMGCV vs R mgcv**: LOOSE. Different implementations, different BLAS, sometimes different algorithms (especially for λ selection).

**Determinism testing contract (v1.9):**

The `set_deterministic(True)` flag is a feature toggle, not a universal CI mode. Tests must not accidentally depend on determinism they don't explicitly enable. Concrete rules:

| Test suite | `deterministic=` | Why |
|---|---|---|
| Unit tests (basis, link, penalty) | `False` (default) | These are deterministic by construction — no GPU reduce, no chunking. Testing with the flag off ensures they don't accidentally rely on it. |
| Cross-path tests (Dense-GPU vs Sparse-CPU) | `False` (default) | MODERATE tolerance absorbs non-determinism. These test the same code paths users run. |
| vs-R tests (PyMGCV vs mgcv) | `False` (default) | LOOSE tolerance. No point enabling determinism for cross-implementation comparison. |
| **CI determinism suite** (separate job) | **`True`** | Dedicated job, pinned JAX + CUDA + driver versions. Runs a subset of cross-path tests at STRICT tolerance. Checks that two identical runs produce identical results. Fails if STRICT tolerance is violated — this catches XLA codegen regressions. |
| Multi-device SPMD tests | `False` (default) | XLA all-reduce is deterministic within a compiled program. MODERATE tolerance. |
| Out-of-core tests (ChunkedJAXProvider) | `False` for default; `True` for reproducibility check | Default chunk ordering may vary. The `True` suite fixes chunk order and checks MODERATE. |

**Key invariant:** no test outside the dedicated determinism suite sets `set_deterministic(True)`. If a test only passes with determinism enabled, that's a bug in the test (tolerance too tight) or the code (non-determinism where none should exist).

```python
# Example test using stratified tolerances:

def test_gaussian_gam_coefficients():
    result = pymgcv.gam("y ~ s(x1) + s(x2)", data=test_df)
    r_result = r_bridge.fit("y ~ s(x1) + s(x2)", data=test_df)

    # Coefficients: LOOSE vs R (different BLAS, optimizer path)
    np.testing.assert_allclose(
        result.coefficients, r_result['coefficients'],
        **LOOSE.__dict__
    )

    # Deviance: MODERATE (same algorithm, different implementation)
    np.testing.assert_allclose(
        result.deviance, r_result['deviance'],
        rtol=1e-6, atol=1e-8
    )


# ── Hard-family invariant tests (v1.7) ──
# These catch correctness bugs that LOOSE prediction tolerances would miss.
# Each tests an internal mathematical invariant, not a comparison to R.

def test_extended_family_loglik_monotonicity(family_class, test_data):
    """
    Invariant: penalized log-likelihood must not increase during
    step-halving. If it does, the gradient or Hessian is wrong.

    Tested at STRICT tolerance. Applies to NB, Tweedie, Beta, SHASH.
    """
    model = pymgcv.gam("y ~ s(x)", data=test_data, family=family_class)

    # Access PIRLS trace (logged during fit when debug=True)
    for i in range(1, len(model._debug_trace)):
        prev = model._debug_trace[i-1]['pen_loglik']
        curr = model._debug_trace[i]['pen_loglik']
        # After step-halving, objective must not have increased
        # (within floating-point tolerance)
        assert curr <= prev + 1e-10 * abs(prev), (
            f"Penalized log-likelihood increased at iteration {i}: "
            f"{prev:.10e} → {curr:.10e}"
        )


def test_extended_family_gradient_accuracy(family_class):
    """
    Invariant: AD gradients (whether standard jax.grad or Tweedie's
    custom_jvp) must match finite differences at MODERATE tolerance
    across the parameter space, including extreme regions.

    This test is family-agnostic — it validates the gradient regardless
    of whether it came from standard AD or a custom rule. The v1.18
    claim is that stable forward passes make standard AD sufficient for
    all families except Tweedie.

    Test points include:
    - theta near 0 and near +inf (NB overdispersion extremes)
    - mu near 0 and near 1 (binomial boundary)
    - y = 0 (zero-inflation edge)
    """
    ll_fn = family_class().loglik_per_obs_fn()
    import jax

    test_points = _generate_extreme_test_points(family_class)
    for eta, y, theta in test_points:
        # AD gradient (standard jax.grad for most; custom_jvp for Tweedie)
        ad_grad = jax.grad(ll_fn, argnums=0)(eta, y, theta)
        # Finite difference gradient
        eps = 1e-5
        fd_grad = (ll_fn(eta + eps, y, theta) - ll_fn(eta - eps, y, theta)) / (2 * eps)

        np.testing.assert_allclose(
            float(ad_grad), float(fd_grad),
            rtol=1e-4, atol=1e-6,
            err_msg=f"AD gradient mismatch at eta={eta}, y={y}, theta={theta}"
        )


def test_deviance_residual_identity(family_class, test_data):
    """
    Invariant: for standard families, sum of squared deviance residuals
    equals the model deviance. For extended families, deviance = -2 * loglik.

    Tested at MODERATE tolerance.
    """
    model = pymgcv.gam("y ~ s(x)", data=test_data, family=family_class)
    if hasattr(model.family, 'loglik_per_obs_fn'):
        # Extended family: deviance = -2 * sum(loglik)
        ll_fn = model.family.loglik_per_obs_fn()
        ll_total = sum(
            ll_fn(eta_i, y_i, model.family.theta_init)
            for eta_i, y_i in zip(model.linear_predictor, test_data['y'])
        )
        np.testing.assert_allclose(
            model.deviance, -2.0 * ll_total,
            rtol=1e-6, atol=1e-8,
        )
```

**Hard-gate invariants — never LOOSE (v1.15):**

The tolerance strategy allows LOOSE comparisons vs R for quantities like p-values and EDF. But some mathematical invariants must hold regardless of mgcv comparison tolerance. These are correctness gates, not comparison tests — if they fail, the implementation is wrong, not merely imprecise:

| Invariant | Tolerance | Rationale |
|---|---|---|
| **Penalized objective monotonicity under step-halving** | STRICT (1e-10) | If pen_dev increases after accepted step, gradient or Hessian is wrong. Catches sign errors, wrong working weights, broken AD or Tweedie custom_jvp. |
| **H = XtWX + S_λ is symmetric positive semi-definite** | STRICT (1e-12 asymmetry) | Asymmetric H → wrong Cholesky → wrong β. Checks `max(abs(H - H.T))`. |
| **Penalty matrix S_j is symmetric positive semi-definite** | STRICT (1e-12) | Broken penalty → wrong smoothing → model makes no statistical sense. |
| **Rank(X) ≥ p - null_space_dim** | Exact | Rank-deficient X beyond expected null space → identifiability constraint bug. |
| **EDF ∈ [0, p] per term, total EDF ∈ [0, n]** | Exact bounds | Negative or impossible EDF → wrong hat matrix trace. |
| **Deviance ≥ 0** | Exact | Negative deviance → log-likelihood computation error. |
| **Converged β produces finite η, μ** | Exact (no NaN/Inf) | NaN in converged model → family link implementation bug. |
| **Cross-path β agreement** | MODERATE (1e-6) | Dense-GPU and Sparse-CPU must agree. If they don't, one path has a bug. Never LOOSE — these are the same algorithm, different arithmetic. |
| **bam() never allocates dense (n, p)** | Exact (assert) | Memory invariant from Section 10.5. If violated, bam loses its purpose. |

These invariants are checked in every CI run, not just the determinism suite. A failure in any of them blocks the build regardless of tolerance class.

### 18.2 R Bridge for Live Comparison

```python
# compat/r_bridge.py

import subprocess
import json
import numpy as np
import tempfile
import os


class RBridge:
    """
    Interface to R's mgcv for reference comparison.

    Two modes:
    1. rpy2 (preferred): Direct R execution in-process
    2. subprocess: Run Rscript and parse output (fallback)
    """

    def __init__(self, mode="auto"):
        if mode == "auto":
            try:
                import rpy2.robjects as ro
                self.mode = "rpy2"
                self._setup_rpy2()
            except ImportError:
                self.mode = "subprocess"
        else:
            self.mode = mode

    def _setup_rpy2(self):
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        self.mgcv = importr("mgcv")
        self.base = importr("base")
        self.stats = importr("stats")

    def fit_gam(self, formula: str, data: dict, family: str = "gaussian",
                method: str = "REML", **kwargs) -> dict:
        """
        Fit a GAM in R and return all results as Python objects.

        Returns dict with:
        - coefficients: ndarray
        - fitted_values: ndarray
        - smoothing_params: ndarray (sp)
        - edf: ndarray (effective degrees of freedom per term)
        - deviance: float
        - scale: float
        - Vp: ndarray (Bayesian covariance)
        - gcv_ubre: float
        - reml: float
        """
        if self.mode == "rpy2":
            return self._fit_rpy2(formula, data, family, method, **kwargs)
        else:
            return self._fit_subprocess(formula, data, family, method, **kwargs)

    def _fit_rpy2(self, formula, data, family, method, **kwargs):
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        import pandas as pd

        pandas2ri.activate()
        df = pd.DataFrame(data)
        r_df = pandas2ri.py2rpy(df)

        # Construct family
        r_family = self._get_r_family(family)

        # Fit
        r_model = self.mgcv.gam(
            ro.Formula(formula),
            data=r_df,
            family=r_family,
            method=method,
        )

        # Extract results
        return {
            'coefficients': np.array(r_model.rx2('coefficients')),
            'fitted_values': np.array(r_model.rx2('fitted.values')),
            'smoothing_params': np.array(r_model.rx2('sp')),
            'edf': np.array(self.base.summary(r_model).rx2('edf')),
            'deviance': float(r_model.rx2('deviance')[0]),
            'scale': float(r_model.rx2('scale')[0]),
            'Vp': np.array(r_model.rx2('Vp')),
            'reml': float(r_model.rx2('gcv.ubre')[0]),
        }

    def _fit_subprocess(self, formula, data, family, method, **kwargs):
        """Fallback: write data to CSV, run Rscript, parse JSON output."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write data
            df = pd.DataFrame(data)
            data_path = os.path.join(tmpdir, "data.csv")
            df.to_csv(data_path, index=False)

            # Write R script
            script = f"""
            library(mgcv)
            library(jsonlite)
            data <- read.csv("{data_path}")
            model <- gam({formula}, data=data, family={family}, method="{method}")
            results <- list(
                coefficients = as.numeric(coef(model)),
                fitted_values = as.numeric(fitted(model)),
                sp = as.numeric(model$sp),
                edf = as.numeric(summary(model)$edf),
                deviance = as.numeric(deviance(model)),
                scale = as.numeric(model$scale),
                Vp = as.matrix(model$Vp),
                reml = as.numeric(model$gcv.ubre)
            )
            writeLines(toJSON(results, digits=15), "{tmpdir}/results.json")
            """
            script_path = os.path.join(tmpdir, "fit.R")
            with open(script_path, "w") as f:
                f.write(script)

            # Run
            subprocess.run(["Rscript", script_path], check=True,
                           capture_output=True)

            # Parse
            with open(os.path.join(tmpdir, "results.json")) as f:
                results = json.load(f)

            return {k: np.array(v) for k, v in results.items()}

    def get_basis_matrix(self, smooth_spec: str, data: dict) -> np.ndarray:
        """Get the design matrix for a smooth term from R."""
        # Uses model.matrix or smoothCon to get basis
        pass

    def get_penalty_matrix(self, smooth_spec: str, data: dict) -> np.ndarray:
        """Get the penalty matrix for a smooth term from R."""
        pass
```

### 18.3 Test Suite Structure

```python
# tests/conftest.py

import pytest
import numpy as np

@pytest.fixture
def r_bridge():
    """Fixture providing R bridge for reference comparison."""
    from pymgcv.compat.r_bridge import RBridge
    try:
        bridge = RBridge(mode="auto")
        return bridge
    except Exception:
        pytest.skip("R not available for reference testing")


@pytest.fixture
def simple_gaussian_data():
    """Standard test dataset for Gaussian GAM."""
    np.random.seed(42)
    n = 500
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    x3 = np.random.uniform(0, 1, n)
    f1 = np.sin(2 * np.pi * x1)
    f2 = 0.5 * x2 ** 2
    f3 = np.exp(-3 * x3)
    y = f1 + f2 + f3 + np.random.normal(0, 0.2, n)
    return {'x1': x1, 'x2': x2, 'x3': x3, 'y': y}


@pytest.fixture
def binary_data():
    """Standard test dataset for logistic GAM."""
    np.random.seed(123)
    n = 1000
    x1 = np.random.uniform(0, 1, n)
    eta = 2 * np.sin(4 * x1) - 1
    p = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p, n).astype(float)
    return {'x1': x1, 'y': y}


@pytest.fixture
def count_data():
    """Standard test dataset for Poisson/NB GAM."""
    np.random.seed(456)
    n = 500
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    eta = 1 + np.sin(2 * np.pi * x1) + 0.5 * x2
    y = np.random.poisson(np.exp(eta))
    return {'x1': x1, 'x2': x2, 'y': y}


@pytest.fixture
def random_effects_data():
    """Dataset with random effects."""
    np.random.seed(789)
    n_groups = 20
    n_per_group = 50
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.uniform(0, 1, n)
    group_effects = np.random.normal(0, 0.5, n_groups)
    y = np.sin(2 * np.pi * x) + group_effects[group] + np.random.normal(0, 0.2, n)
    return {'x': x, 'group': group.astype(str), 'y': y}
```

### 18.4 Component-Level Tests

```python
# tests/test_smooths/test_tprs.py

import numpy as np
import pytest
from pymgcv.smooths.tprs import ThinPlateSmooth, ThinPlateShrinkageSmooth
from pymgcv.smooths.base import SmoothSpec


class TestTPRS:
    """Test thin plate regression spline basis and penalty."""

    def test_basis_dimensions(self, simple_gaussian_data):
        spec = SmoothSpec(
            term_label="s(x1)", variables=["x1"], bs="tp", k=10
        )
        smooth = ThinPlateSmooth(spec)
        smooth.setup(simple_gaussian_data)
        X = smooth.build_design_matrix(simple_gaussian_data)

        assert X.shape == (len(simple_gaussian_data['x1']), 10)
        assert smooth.n_coefs == 10
        assert smooth.null_space_dim == 2  # linear + constant

    def test_penalty_positive_semidefinite(self, simple_gaussian_data):
        spec = SmoothSpec(
            term_label="s(x1)", variables=["x1"], bs="tp", k=10
        )
        smooth = ThinPlateSmooth(spec)
        smooth.setup(simple_gaussian_data)
        S = smooth.build_penalty_matrices()[0].toarray()

        eigenvalues = np.linalg.eigvalsh(S)
        assert np.all(eigenvalues >= -1e-10), "Penalty must be PSD"

    def test_penalty_null_space_rank(self, simple_gaussian_data):
        spec = SmoothSpec(
            term_label="s(x1)", variables=["x1"], bs="tp", k=10
        )
        smooth = ThinPlateSmooth(spec)
        smooth.setup(simple_gaussian_data)
        S = smooth.build_penalty_matrices()[0].toarray()

        rank = np.linalg.matrix_rank(S, tol=1e-8)
        assert rank == 10 - 2, "Null space dim should be 2 for 1D TPRS"

    def test_2d_tprs(self, simple_gaussian_data):
        spec = SmoothSpec(
            term_label="s(x1,x2)", variables=["x1", "x2"], bs="tp", k=30
        )
        smooth = ThinPlateSmooth(spec)
        smooth.setup(simple_gaussian_data)
        X = smooth.build_design_matrix(simple_gaussian_data)

        assert X.shape[1] == 30
        assert smooth.null_space_dim == 3  # constant + x1 + x2

    @pytest.mark.r_comparison
    def test_basis_matches_r(self, simple_gaussian_data, r_bridge):
        """Compare basis matrix against R's mgcv."""
        spec = SmoothSpec(
            term_label="s(x1)", variables=["x1"], bs="tp", k=10
        )
        smooth = ThinPlateSmooth(spec)
        smooth.setup(simple_gaussian_data)
        X_py = smooth.build_design_matrix(simple_gaussian_data)

        # Get R's basis matrix
        X_r = r_bridge.get_basis_matrix("s(x1, bs='tp', k=10)",
                                         simple_gaussian_data)

        # Bases may differ by rotation, so compare column spaces
        # via: ||X_py X_py^+ - X_r X_r^+|| ≈ 0
        P_py = X_py @ np.linalg.pinv(X_py)
        P_r = X_r @ np.linalg.pinv(X_r)
        np.testing.assert_allclose(P_py, P_r, atol=1e-6,
            err_msg="Column spaces of Python and R bases must match")

    @pytest.mark.r_comparison
    def test_penalty_matches_r(self, simple_gaussian_data, r_bridge):
        """Compare penalty eigenvalues against R's mgcv."""
        spec = SmoothSpec(
            term_label="s(x1)", variables=["x1"], bs="tp", k=10
        )
        smooth = ThinPlateSmooth(spec)
        smooth.setup(simple_gaussian_data)
        S_py = smooth.build_penalty_matrices()[0].toarray()

        S_r = r_bridge.get_penalty_matrix("s(x1, bs='tp', k=10)",
                                           simple_gaussian_data)

        # Compare eigenvalues (rotation-invariant)
        eig_py = np.sort(np.linalg.eigvalsh(S_py))
        eig_r = np.sort(np.linalg.eigvalsh(S_r))
        np.testing.assert_allclose(eig_py, eig_r, rtol=1e-5,
            err_msg="Penalty eigenvalues must match R")


# tests/test_smooths/test_all_bases.py

@pytest.mark.parametrize("bs,k,d,expected_null_dim", [
    ("tp", 10, 1, 2),
    ("tp", 30, 2, 3),
    ("ts", 10, 1, 2),
    ("cr", 10, 1, 2),
    ("cs", 10, 1, 2),
    ("cc", 9, 1, 1),    # Cyclic: constant only
    ("ps", 20, 1, 2),   # Default m=2 P-spline
    ("bs", 10, 1, 0),   # B-splines with default penalty
])
def test_smooth_null_space_dimension(bs, k, d, expected_null_dim,
                                     simple_gaussian_data):
    vars = ["x1"] if d == 1 else ["x1", "x2"]
    spec = SmoothSpec(
        term_label=f"s({','.join(vars)})", variables=vars, bs=bs, k=k
    )
    SmoothClass = get_smooth_class(bs)
    smooth = SmoothClass(spec)
    smooth.setup(simple_gaussian_data)
    assert smooth.null_space_dim == expected_null_dim
```

### 18.5 End-to-End Comparison Tests

```python
# tests/test_api/test_full_models.py

class TestGaussianGAM:
    """End-to-end tests for Gaussian GAM against R."""

    TOLERANCES = {
        'coefficients': {'atol': 1e-4, 'rtol': 1e-3},
        'smoothing_params': {'atol': 0, 'rtol': 0.1},  # λ can vary more
        'fitted_values': {'atol': 1e-4, 'rtol': 1e-3},
        'edf': {'atol': 0.5, 'rtol': 0.05},
        'deviance': {'atol': 1e-2, 'rtol': 1e-3},
        'scale': {'atol': 1e-3, 'rtol': 1e-2},
    }

    @pytest.mark.r_comparison
    def test_simple_gam(self, simple_gaussian_data, r_bridge):
        formula = "y ~ s(x1) + s(x2) + s(x3)"

        # Fit in Python
        py_result = gam(formula, simple_gaussian_data)

        # Fit in R
        r_result = r_bridge.fit_gam(formula, simple_gaussian_data)

        # Compare
        np.testing.assert_allclose(
            py_result.fitted_values, r_result['fitted_values'],
            **self.TOLERANCES['fitted_values'],
            err_msg="Fitted values must match R"
        )
        np.testing.assert_allclose(
            py_result.deviance, r_result['deviance'],
            **self.TOLERANCES['deviance'],
            err_msg="Deviance must match R"
        )

    @pytest.mark.r_comparison
    def test_tensor_product(self, simple_gaussian_data, r_bridge):
        formula = "y ~ te(x1, x2) + s(x3)"
        py_result = gam(formula, simple_gaussian_data)
        r_result = r_bridge.fit_gam(formula, simple_gaussian_data)

        np.testing.assert_allclose(
            py_result.fitted_values, r_result['fitted_values'],
            atol=1e-3, rtol=1e-2,
        )

    @pytest.mark.r_comparison
    def test_with_fixed_effects(self, simple_gaussian_data, r_bridge):
        simple_gaussian_data['z'] = np.random.normal(0, 1,
            len(simple_gaussian_data['y']))
        formula = "y ~ z + s(x1) + s(x2)"
        py_result = gam(formula, simple_gaussian_data)
        r_result = r_bridge.fit_gam(formula, simple_gaussian_data)

        np.testing.assert_allclose(
            py_result.coefficients[:2], r_result['coefficients'][:2],
            atol=1e-3,
        )


class TestBinomialGAM:
    @pytest.mark.r_comparison
    def test_logistic_gam(self, binary_data, r_bridge):
        formula = "y ~ s(x1)"
        py_result = gam(formula, binary_data, family="binomial")
        r_result = r_bridge.fit_gam(formula, binary_data, family="binomial()")

        np.testing.assert_allclose(
            py_result.fitted_values, r_result['fitted_values'],
            atol=1e-3, rtol=1e-2,
        )


class TestExtendedFamilies:
    @pytest.mark.r_comparison
    def test_negative_binomial(self, count_data, r_bridge):
        formula = "y ~ s(x1) + s(x2)"
        py_result = gam(formula, count_data, family="nb")
        r_result = r_bridge.fit_gam(formula, count_data, family="nb()")

        np.testing.assert_allclose(
            py_result.fitted_values, r_result['fitted_values'],
            atol=0.1, rtol=0.05,
        )

    @pytest.mark.r_comparison
    def test_tweedie(self, r_bridge):
        np.random.seed(99)
        n = 500
        x = np.random.uniform(0, 1, n)
        mu = np.exp(1 + np.sin(2 * np.pi * x))
        # Simulate from Tweedie (approximate)
        y = np.random.gamma(shape=mu/0.5, scale=0.5, size=n)

        data = {'x': x, 'y': y}
        formula = "y ~ s(x)"

        py_result = gam(formula, data, family="tw")
        r_result = r_bridge.fit_gam(formula, data, family="tw()")

        np.testing.assert_allclose(
            py_result.fitted_values, r_result['fitted_values'],
            atol=0.5, rtol=0.1,
        )


class TestRandomEffects:
    @pytest.mark.r_comparison
    def test_random_intercept(self, random_effects_data, r_bridge):
        formula = "y ~ s(x) + s(group, bs='re')"
        py_result = gam(formula, random_effects_data)
        r_result = r_bridge.fit_gam(formula, random_effects_data)

        np.testing.assert_allclose(
            py_result.fitted_values, r_result['fitted_values'],
            atol=1e-2, rtol=1e-2,
        )


class TestPrediction:
    @pytest.mark.r_comparison
    def test_predict_se(self, simple_gaussian_data, r_bridge):
        formula = "y ~ s(x1) + s(x2)"
        py_result = gam(formula, simple_gaussian_data)

        # Predict at new points
        new_data = {'x1': np.linspace(0, 1, 50),
                    'x2': np.full(50, 0.5)}
        py_pred = py_result.predict(new_data, se_fit=True)

        # Compare against R
        r_pred = r_bridge.predict_gam(formula, simple_gaussian_data,
                                       new_data, se_fit=True)

        np.testing.assert_allclose(
            py_pred['fit'], r_pred['fit'], atol=1e-3
        )
        np.testing.assert_allclose(
            py_pred['se'], r_pred['se'], atol=1e-2, rtol=0.05
        )
```

### 18.6 Test Matrix: Systematic Coverage

The following test matrix ensures comprehensive coverage. Each cell must pass:

| | Gaussian | Binomial | Poisson | Gamma | NB | Tweedie | Beta | SHASH |
|---|---|---|---|---|---|---|---|---|
| s(x, bs="tp") | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s(x, bs="cr") | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s(x, bs="ps") | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s(x, bs="cc") | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| te(x1, x2) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| ti(x1, x2) | ✓ | ✓ | ✓ | — | — | — | — | — |
| s(g, bs="re") | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — |
| s(x, g, bs="fs") | ✓ | ✓ | — | — | — | — | — | — |
| s(x, by=g) | ✓ | ✓ | ✓ | — | — | — | — | — |
| Fixed effects | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| select=TRUE | ✓ | ✓ | ✓ | — | — | — | — | — |
| method=GCV | ✓ | — | — | — | — | — | — | — |
| method=REML | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| bam() | ✓ | ✓ | ✓ | — | ✓ | — | — | — |

### 18.7 Reference Data Generation Script

```r
# tests/generate_reference_data.R
# Run this script to generate reference results from R's mgcv

library(mgcv)
library(jsonlite)

set.seed(42)
n <- 500
x1 <- runif(n)
x2 <- runif(n)
x3 <- runif(n)
f1 <- sin(2*pi*x1)
f2 <- 0.5 * x2^2
f3 <- exp(-3*x3)
y <- f1 + f2 + f3 + rnorm(n, 0, 0.2)
data <- data.frame(x1=x1, x2=x2, x3=x3, y=y)

# Test case 1: Simple Gaussian GAM
m1 <- gam(y ~ s(x1) + s(x2) + s(x3), data=data, method="REML")
ref1 <- list(
    coefficients = as.numeric(coef(m1)),
    fitted = as.numeric(fitted(m1)),
    sp = as.numeric(m1$sp),
    edf = as.numeric(m1$edf),
    deviance = as.numeric(deviance(m1)),
    scale = m1$sig2,
    reml = as.numeric(m1$gcv.ubre),
    Vp = as.matrix(m1$Vp)
)
write(toJSON(ref1, digits=15), "tests/reference_data/gaussian_simple.json")

# Test case 2: Different basis types
m2 <- gam(y ~ s(x1, bs="cr") + s(x2, bs="ps") + s(x3, bs="tp"),
           data=data, method="REML")
# ... save similarly

# Test case 3: Tensor products
m3 <- gam(y ~ te(x1, x2, k=c(5,5)) + s(x3), data=data, method="REML")
# ... save

# Test case 4: Binomial
yb <- rbinom(n, 1, plogis(f1))
m4 <- gam(yb ~ s(x1), data=data.frame(x1=x1, yb=yb),
           family=binomial(), method="REML")
# ... save

# Test case 5: Negative binomial
yc <- rnbinom(n, mu=exp(1+f1), size=2)
m5 <- gam(yc ~ s(x1), data=data.frame(x1=x1, yc=yc),
           family=nb(), method="REML")
# ... save

# Continue for all family × smooth × method combinations...
```

### 18.8 GPU Parity Tests

```python
# tests/test_gpu/test_gpu_parity.py

@pytest.mark.gpu
class TestGPUParity:
    """Verify GPU results match CPU results."""

    def test_gaussian_cpu_gpu_match(self, simple_gaussian_data):
        formula = "y ~ s(x1) + s(x2)"

        # Fit on CPU
        cpu_result = gam(formula, simple_gaussian_data,
                         backend="jax", device="cpu")

        # Fit on GPU
        gpu_result = gam(formula, simple_gaussian_data,
                         backend="jax", device="gpu")

        np.testing.assert_allclose(
            cpu_result.coefficients, gpu_result.coefficients,
            atol=1e-5, rtol=1e-4,
            err_msg="GPU results must match CPU"
        )
        np.testing.assert_allclose(
            cpu_result.smoothing_params, gpu_result.smoothing_params,
            rtol=1e-3,
        )
```

### 18.9 Performance Benchmarks

```python
# tests/benchmarks/benchmark_pirls.py

import time
import numpy as np

def benchmark_pirls_scaling():
    """Benchmark PIRLS performance across data sizes."""
    results = []
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        data = generate_test_data(n)
        for backend in ["jax_cpu", "jax_gpu", "numpy"]:
            t0 = time.perf_counter()
            gam("y ~ s(x1) + s(x2)", data, backend=backend.split("_")[0],
                device=backend.split("_")[1] if "_" in backend else "cpu")
            elapsed = time.perf_counter() - t0
            results.append({
                'n': n, 'backend': backend, 'time_sec': elapsed
            })
    return results
```

---

## 19. Implementation Phases and Agent Task Breakdown

### Phase 1: Foundation (Weeks 1-4)

| Task | Agent Assignment | Dependencies | Deliverables |
|---|---|---|---|
| 1.1 Project scaffolding | Infra Agent | None | Package structure, CI/CD, dependencies |
| 1.2 JAX-first backend + NumPy fallback | Core Agent | 1.1 | `linalg/backend.py`, device config, JIT patterns |
| 1.3 JAX AD interface | Core Agent | 1.2 | `autodiff/interface.py`, grad/hessian/hvp wrappers |
| 1.4 Link functions | Family Agent | 1.1 | All 7+ link functions, tests |
| 1.5 Standard families (closed-form only) | Family Agent | 1.4 | Gaussian, Binomial, Poisson, Gamma, InvGauss |
| 1.6 Formula parser (formulaic + smooth extractor) | API Agent | 1.1 | Two-layer parser, 3 weeks (not 1) |
| 1.7 R bridge | Test Agent | 1.1 | rpy2 + subprocess bridge, fixture generation |
| 1.8 StatisticsProvider protocol | Core Agent | 1.2 | `distributed/stats_provider.py`, InMemoryProvider |

### Phase 2: Core Smooths (Weeks 3-6)

| Task | Agent Assignment | Dependencies | Deliverables |
|---|---|---|---|
| 2.1 Smooth base class | Smooth Agent | 1.2 | `smooths/base.py` with constraint handling |
| 2.2 TPRS (tp, ts) | Smooth Agent | 2.1 | Full TPRS with eigendecomp, tests vs R |
| 2.3 Cubic splines (cr, cs, cc) | Smooth Agent | 2.1 | Cubic + cyclic + shrinkage |
| 2.4 P-splines (ps, cp) | Smooth Agent | 2.1 | P-splines + cyclic P-splines |
| 2.5 B-splines (bs) | Smooth Agent | 2.1 | Standard B-spline basis |
| 2.6 Tensor products (te, ti, t2) | Smooth Agent | 2.2-2.4 | All tensor product types |
| 2.7 Random effects (re, fs) | Smooth Agent | 2.1 | Random effects + factor-smooth |
| 2.8 Design matrix assembly | API Agent | 2.1-2.7, 1.6 | `formula/design.py` |

### Phase 3: Fitting Engine (Weeks 5-9)

| Task | Agent Assignment | Dependencies | Deliverables |
|---|---|---|---|
| 3.1 PIRLS (via StatisticsProvider) | Fitting Agent | 2.8, 1.5, 1.8 | `fitting/pirls.py`, JAX + NumPy paths |
| 3.2 Newton outer iteration (AD-powered) | Fitting Agent | 3.1, 1.3 | REML optimization with JAX autodiff |
| 3.3 Fellner-Schall updates | Fitting Agent | 3.1 | Fast REML alternative |
| 3.4 REML/ML/GCV/UBRE criteria | Fitting Agent | 3.1, 1.3 | All smoothness criteria |
| 3.5 Convergence hardening | Fitting Agent | 3.1 | Penalized deviance tracking, trust-region, weight floors |
| 3.6 Joint identifiability (gam_side) | Fitting Agent | 2.8 | `fitting/constraints.py`, cross-term overlap detection |
| 3.7 Three execution paths | Performance Agent | 3.1 | Dense-GPU, Sparse-CPU, Chunked providers |
| 3.8 gam() top-level API | API Agent | 3.1-3.7, 2.8 | Full `gam()` function |
| 3.9 End-to-end R comparison | Test Agent | 3.8, 1.7 | All Gaussian/Binomial/Poisson tests pass |

### Phase 4: Extended Families (Weeks 8-11)

| Task | Agent Assignment | Dependencies | Deliverables |
|---|---|---|---|
| 4.1 Extended PIRLS (gam.fit5) | Fitting Agent | 3.1, 1.3 | Extended family fitting |
| 4.2 Negative Binomial | Family Agent | 4.1 | nb() with stable lgamma forward pass, validated via jax.grad |
| 4.3 Tweedie + custom_jvp | Family Agent | 4.1 | tw() with series derivative rules — only family needing custom_jvp |
| 4.4 Beta regression | Family Agent | 4.1 | betar() — stable forward, plain AD |
| 4.5 Ordered categorical | Family Agent | 4.1 | ocat() — log_diff_exp forward, plain AD |
| 4.6 Zero-inflated (ZIP, ZAGA) | Family Agent | 4.1 | All zero-inflated families |
| 4.7 Location-scale families | Family Agent | 4.1 | gaulss, gammals, shash, gevlss |
| 4.8 Cox PH | Family Agent | 4.1 | Survival GAM with logsumexp-based partial ll, plain AD |
| 4.9 Scaled t | Family Agent | 4.1 | scat() |
| 4.10 Multinomial | Family Agent | 4.1 | multinom() |
| 4.11 AD validation suite | Test Agent | 4.2-4.10 | Finite-diff vs AD gradient tests for all families |

### Phase 5: Large Data + Performance (Weeks 10-13)

| Task | Agent Assignment | Dependencies | Deliverables |
|---|---|---|---|
| 5.1 Discretization | Performance Agent | 2.8 | Covariate discretization |
| 5.2 ChunkedProvider | Performance Agent | 1.8 | Memory-bounded XtWX via StatisticsProvider |
| 5.3 bam() implementation | Performance Agent | 5.1, 5.2, 3.3 | Full bam() with fREML |
| 5.4 Dense-GPU path optimization | Performance Agent | 3.7 | Fused JIT PIRLS, GPU benchmarks |
| 5.5 Sparse-CPU path optimization | Performance Agent | 3.7 | CHOLMOD integration, sparse benchmarks |
| 5.6 Cython fallbacks (NumPy path) | Performance Agent | 3.1 | PIRLS + basis eval Cython |
| 5.7 Performance benchmarks | Performance Agent | 5.1-5.6 | Timing suite, R comparison |

### Phase 5b: Distributed Compute (Weeks 12-14)

| Task | Agent Assignment | Dependencies | Deliverables |
|---|---|---|---|
| 5b.1 ChunkedJAXProvider | Distributed Agent | 1.8, 5.2 | Out-of-core chunked JAX fitting |
| 5b.2 Ray JaxTrainer launcher | Distributed Agent | 1.8, 5.2 | Multi-host cluster bootstrap |
| 5b.3 Distributed knot placement | Distributed Agent | 5b.1 | Sketch-based knot selection |
| 5b.4 bam() distributed API | Distributed Agent | 5b.1, 5b.2, 5.3 | `bam(data="/path", distributed="dask")` |
| 5b.5 Distributed integration tests | Test Agent | 5b.1-5b.4 | Correctness vs single-node |

### Phase 6: Advanced Smooths + Remaining Features (Weeks 13-17)

| Task | Agent Assignment | Dependencies | Deliverables |
|---|---|---|---|
| 6.1 Gaussian process (gp) | Smooth Agent | 2.1 | GP smooth with multiple kernels |
| 6.2 MRF (mrf) | Smooth Agent | 2.1 | Markov random field |
| 6.3 Soap film (so) | Smooth Agent | 2.1 | Boundary-respecting 2D smooth |
| 6.4 Duchon splines (sz) | Smooth Agent | 2.1 | Generalized TPRS |
| 6.5 Adaptive smooth (ad) | Smooth Agent | 2.1, 3.1 | Locally adaptive penalties |
| 6.6 Linear functionals | Smooth Agent | 2.1 | Functional covariate terms |
| 6.7 gamm() via PQL | Fitting Agent | 3.1 | Mixed model fitting |
| 6.8 Prediction + SE | API Agent | 3.8 | predict() with all options |
| 6.9 Summary + anova | API Agent | 3.8 | summary(), anova_gam(), diagnostics |
| 6.10 Concurvity detection | API Agent | 3.8 | concurvity(), pairwise measures |
| 6.11 Model comparison | API Agent | 3.8 | AIC, BIC, anova_gam (multi-model) |
| 6.12 Plotting | API Agent | 3.8, 6.8 | plot() for all smooth types |

### Phase 7: Polish + Release (Weeks 16-19)

| Task | Agent Assignment | Dependencies | Deliverables |
|---|---|---|---|
| 7.1 Full test matrix | Test Agent | All | All cells in Section 18.6 pass |
| 7.2 Documentation | Doc Agent | All | API docs, tutorials, examples |
| 7.3 Stan/NumPyro export | API Agent | 3.8 | jagam() equivalent |
| 7.4 PyPI packaging | Infra Agent | All | pip install pymgcv |
| 7.5 Performance optimization | Performance Agent | 5.7 | Final tuning, profiling |
| 7.6 Distributed smoke tests | Test Agent | 5b.5 | Dask + Ray end-to-end on sample clusters |

---

## Appendix A: Complete Smooth Class Catalog

| bs code | Name | Dim | Penalty | Null Space | Key Implementation Detail |
|---|---|---|---|---|---|
| tp | Thin plate | any | ∫(∂^m f)² | polynomials deg < m | Eigendecomp of distance matrix |
| ts | Thin plate + shrinkage | any | Same + null space penalty | Same | Extra penalty on null space |
| cr | Cubic regression | 1 | ∫ f''² | linear | Natural cubic spline basis |
| cs | Cubic + shrinkage | 1 | Same + null space | linear | Extra null space penalty |
| cc | Cyclic cubic | 1 | ∫ f''² (periodic) | constant | Wrap-around boundary |
| ps | P-spline | 1 | D^m (difference) | poly deg < m | B-spline + difference penalty |
| cp | Cyclic P-spline | 1 | Cyclic difference | constant | Circular B-splines |
| bs | B-spline | 1 | Derivative penalty | varies | Standard B-spline, variable order |
| ad | Adaptive | 1-2 | Locally varying | varies | Multiple penalties, adaptive λ |
| gp | Gaussian process | any | K^{-1} (precision) | constant | Matérn/exponential covariance |
| mrf | Markov random field | discrete | Adjacency Laplacian | constant | User-supplied graph |
| re | Random effects | factor | I | none | Identity basis, identity penalty |
| fs | Factor-smooth | 1+factor | Block-diagonal | per-level | Separate smooth per level |
| te | Tensor product | any | Kronecker sum | product | Kronecker of marginals |
| ti | Tensor interaction | any | Same, constrained | interaction only | ANOVA decomposition |
| t2 | Tensor type 2 | any | Single per marginal | product | Wood et al. 2013 |
| so | Soap film | 2 (spatial) | PDE-based | none | Boundary polygon required |
| sz | Duchon | any | Fractional derivative | varies | Generalized TPRS |

## Appendix B: Complete Family Catalog

| Family | Type | Extra Params | Default Link | Log-lik Implementation |
|---|---|---|---|---|
| gaussian | Standard | — | identity | Closed form |
| binomial | Standard | — | logit | Closed form |
| poisson | Standard | — | log | Closed form |
| Gamma | Standard | — | inverse | Closed form |
| inverse.gaussian | Standard | — | 1/μ² | Closed form |
| nb | Extended | θ (size) | log | gammaln series |
| negbin | Extended | θ (fixed) | log | gammaln series |
| tw | Extended | p (power) | log | Tweedie series/FFT |
| betar | Extended | φ (precision) | logit | Beta density |
| ocat | Extended | θ (cut points) | — | Ordered probit/logit |
| multinom | Extended | — | — | Softmax categorical |
| zip | Extended | π (zero prob) | log | Mixture |
| cox.ph | Extended | — | log | Partial likelihood |
| scat | Extended | ν (df) | identity | Scaled t |
| gaulss | Location-scale | — | identity, log | Normal with σ(x) |
| gammals | Location-scale | — | log, log | Gamma with shape(x) |
| gevlss | Location-scale | — | identity, log, logit | GEV with all params |
| shash | Location-scale | — | identity, log, identity, log | Sinh-arcsinh |
| ziplss | Location-scale | — | log, logit | ZIP with both params |

## Appendix C: Reference Test Cases

### C.1 Analytical Test Functions

```python
# Known functions for verifying smooth recovery

def test_function_1d():
    """f(x) = sin(2πx), x ∈ [0,1]"""
    return lambda x: np.sin(2 * np.pi * x)

def test_function_2d():
    """f(x1,x2) = 0.2*x1^11*(10*(1-x2))^6 + 10*(10*x2)^3 / (1+(10*x2)^3)"""
    # Wood's test function
    return lambda x1, x2: (
        0.2 * x1**11 * (10*(1-x2))**6 +
        10 * (10*x2)**3 / (1 + (10*x2)**3)
    )

def test_function_additive():
    """f1 + f2 + f3 with known components for additivity testing."""
    f1 = lambda x: np.sin(2 * np.pi * x)
    f2 = lambda x: np.exp(2 * x) / (1 + np.exp(2 * x))
    f3 = lambda x: 0.2 * x**11 * (10*(1-x))**6 + 10*(10*x)**3/(1+(10*x)**3)
    return f1, f2, f3
```

### C.2 Tolerance Guidelines

| Quantity | Absolute Tolerance | Relative Tolerance | Notes |
|---|---|---|---|
| Coefficients β | 1e-4 | 1e-3 | After matching parameterization |
| Fitted values μ | 1e-4 | 1e-3 | Most reliable comparison |
| Smoothing params λ | — | 0.1 (10%) | λ is on log scale; 10% is tight |
| EDF per term | 0.5 | 0.05 | Can vary due to λ differences |
| Deviance | 1e-2 | 1e-3 | Should match closely |
| Scale estimate | 1e-3 | 1e-2 | Depends on EDF |
| REML score | 1e-1 | 1e-2 | Sensitive to parameterization |
| Standard errors | 1e-2 | 0.05 | Depends on Vp |
| p-values (smooth) | 1e-1 | 0.1 | Approximate test |

---

*End of Design Document*
