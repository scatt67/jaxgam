# Unified Fitting Engine: Implementation Plan

**Design reference:** [design.md](design.md)

**When:** start only after the current body of work is complete (see Entry criteria). This series must be the only active work on the stack, so that every step composes once.

**Shape:** a sequence of PRs, one logical unification each, starting from `main` (the stacked PRs this plan was written against were merged into `main` as one squash commit, #87). Each PR proves bit-identical numerics. Nothing is merged without the user.

---

## Entry criteria

All of the following must hold before Phase 0 starts:

1. PR9, PR11 (incl. its NB `estimate.theta`, approximation and runtime studies), PR12, EFS6 and PR13 are composed, published as stacked PRs and native-CI green.
2. The scaling plan's §8 performance and numerical acceptance gates are met and recorded.
3. `verify_active_stack.py` (or its successor) passes: every head, base edge and native-stack position is verified.
4. No other implementation branch is in flight.

**Status (2026-11-20):** criteria 1 and 2 are met: that work was consolidated and merged into `main` as pull request #87 (commit `e919938`) with CI green, after the acceptance measurements were taken. Criterion 3 belonged to the stacked-PR workflow and no longer applies; the equivalent check is that Phase 0 starts from `main`. Criterion 4 must be checked again when Phase 0 starts.

---

## Project tooling

Per AGENTS.md, use `uv` and `make`; never call `python` or `pytest` directly.

| Task | Command |
|---|---|
| Install | `make install` |
| Lint (ruff, format, vulture; later the duplicate check) | `make lint` |
| Focused local tests | `uv run pytest <path> -x --tb=short` |
| Full pinned suite (R 4.5.2 + mgcv 1.9-3, one BLAS thread) | `make test` / `make docker-build` then `docker run --rm …` |
| Focused pinned file | `make test-file FILE=…` |

Further rules:
- R comparisons use typed rpy2 `RBridge` only. No saved fixtures, raw evidence or local diagnostic scripts are tracked.
- The fingerprint harness below is tooling and is tracked; its outputs are not.

---

## Global rules for every PR in this series

1. **One logical unification per PR.** PR title format: `[phase2] execution: …` or `[refactor] fitting: …`.
2. **Bit-identity gate.** Run `make fit-fingerprint` (Phase 0) on the PR base and on the PR head, on the host and on the pinned Linux image. Every fingerprinted field must be byte-identical.
   - Any intended difference must be listed in the PR body, per case, with its pinned-R comparison still at the original tolerance class. The only expected case is the §2.3 drift in the design.
3. **No tolerance, oracle or digest edits.** If a gate fails, the refactor is wrong.
4. **Full pinned Linux suite and native CI green.** Branch coverage must not fall below the base. New modules need >80%.
5. **Delete as you go.** A PR that introduces the unified path removes the replaced implementation in the same PR. No dead parallel copies survive a merge point.
6. **Record the reduction.** Report line counts and duplicate-scan results for the touched modules, before and after.
7. **JIT tests.** Any changed Phase-2 kernel keeps or gains a compile-and-run test. Record compile counts on the fingerprint cases; a regression needs justification.

---

## Phase 0: Baseline, harness and guardrails (no production change)

**Goal:** make every later step measurable and protected.

### 0.1 Fingerprint harness

Add `scripts/fit_fingerprint.py` and `make fit-fingerprint`. The harness:
- runs a fixed, runtime-generated case list over every route, family/link and mode that the validation matrix covers (dense, stream fixed-sp, exact REML, Gaussian QR, EFS dense/stream, discrete, operator, bulk uncertainty);
- writes a SHA-256 per result field (coefficients, smoothing parameters, scale, theta, score, EDF, fitted values, SEs, convergence fields, work counters) to a local JSON outside the repo;
- in `--compare A B` mode, diffs two checkouts, as `benchmark_scaling.py --checkout` does today: commit and dirty-tree provenance per child, refusing mixed provenance.

Tests cover the harness itself: deterministic output, provenance refusal and diff reporting.

### 0.2 Architecture test, report-only

Add `tests/test_architecture.py`, with an AST import scan and a scan for `isinstance` checks against concrete family classes.
- **Enforced now:** `fitting/` never imports `execution/`, host `scipy`, `sqlite3` or file I/O; Phase-1 packages have no JAX. These hold on `main` with one exception to allowlist and then remove in Phase 2: `fitting/data.py::_structural_penalty_roots` imports `execution/qr.py` inside the function.
- **Recorded as an explicit allowlist:** the current host-loop exceptions (`fitting/newton.py::NewtonOptimizer`, `fitting/data.py` preparation) and the current family `isinstance` sites (36–43 per line in October 2026; 46 on `main` after the merge). Later phases shrink the allowlist to empty.

### 0.3 Duplicate-code check, report-only

Choose pylint `duplicate-code` or `jscpd` (design §9 Q1) and add it to `make lint` in report mode, with a committed baseline. Also commit the structural scan used in design §2.2, normalised-AST similarity over functions with at least 15 body lines, as `scripts/duplicate_scan.py`. The textual tools miss renamed copies.

### 0.4 Re-measure

Re-run the design §2 inventory on the final composed stack and update `design.md` §2 tables with the real starting point.

**Acceptance:**
- the harness is deterministic across two runs on the same commit, on host and on Linux;
- the architecture test and duplicate report pass in report-only mode;
- the inventory is updated.

---

## Phase 1: Exact and near-exact duplicates (small, low risk)

Each item is its own commit, and the phase ships as one PR.

1. **`_factor_inverse_residual`:** keep one copy, move it to a shared execution utility, and use it from both `reml.py` and `nb_reml.py`.
2. **`_link_name`:** delete both copies, `execution/nb_stream.py` and `fitting/nb_stream_reml.py`. Use a family/link capability (e.g. `link.name`, or a capability enumerating supported streamed NB links).
3. **Saturated log-likelihood:** keep one kernel and make the other call site use it. The two are `fitting/family_execution.py::batch_saturated_loglikelihood` and `fitting/stream_kernels.py::saturated_loglik_reduction`. Prove reduction-order identity by fingerprint.
4. **NB theta objective:** keep one jitted NB negative log-likelihood kernel, used by `fitting/efs_theta.py` and `fitting/nb_theta_stream.py` (structural similarity 0.98).
5. **`efsudr` update:** the operator EFS controller uses `fitting/efs.py::efs_raw_update` (or a NumPy wrapper over the same kernel), and `_raw_ratio`/`_trial_rho` are deleted.
   - First resolve the `phi*n/q` vs `(n/q)*phi` order against pinned `R/gam.fit4.r::efsudr` (1.9-3).
   - The non-matching route's change is the one documented exception. Its R gates must stay STRICT.
6. **Score-bound helpers:** make `_indexed_poisson_score_bound`, `_indexed_gaussian_score_bound` and `_arrowhead_poisson_score_bound` one family-generic bound, parameterised by family capability terms.

**Acceptance:** global rules. Fingerprint identical except item 5's documented case.

---

## Phase 2: Names and boundary moves (mechanical)

1. **Renames:** `execution/reml.py` → `execution/stream_reml.py` and `execution/nb_reml.py` → `execution/nb_stream_reml.py`. Do the same for any other pair whose name implies a kernel/driver relationship that does not exist; check every pair.
2. **Move host code out of `fitting/`:**
   - `NewtonOptimizer`'s host loop moves to `execution/optimizers/newton.py`. Its jitted derivative pieces stay in `fitting/newton.py`, renamed to reflect that they are kernels.
   - `FittingData.from_setup` NumPy preparation moves to `execution/providers/in_memory.py`, or stays as a transfer type, per design §9 Q2.
3. **Re-export shims:** add them only where something outside the package imports the old path. Check `jaxgam/__init__.py`, docs and benchmark scripts. Internal modules get no shims.
4. **Architecture test:** remove the host-loop exceptions from its allowlist.

**Acceptance:** global rules (pure moves, so the fingerprint must be identical). `git log --follow` history is preserved by using `git mv`.

---

## Phase 3: Provider and solver interfaces (adapters, no behaviour change)

1. **Introduce the interfaces:** `FitProvider`, `ProviderCapabilities`, `FitState`, `CoefficientSolver`, `WorkingSystem` and `OptimizationResult` (design §5.1). Generalise the existing `EFSFitProvider`/`EFSFitRequest`/`EFSControllerContext` rather than adding a parallel set.
2. **Wrap existing data access as providers,** delegating to today's functions:
   - `in_memory` (dense);
   - `stream`, regular and NB (`stream.py`, `regular_stream.py`, `nb_stream.py`);
   - `gaussian_qr` (PR9);
   - `discrete` (PR11);
   - `operator` (PR12/EFS6).
3. **Wrap the solvers:** cholesky, qr, signed_qr, pcg, and `working_model` (`Sl.fitChol` with the pinned DPSTRF host factor).
4. **Contract tests for each provider:**
   - exact gradient vs finite differences of the score at STRICT where exact;
   - replay determinism;
   - work-counter monotonicity;
   - the capability flags truthfully reflect implemented methods.

**Acceptance:** global rules. Routes are not yet switched, so the fingerprint is identical.

---

## Phase 4: One EFS optimizer

1. **Merge the two dense loops.** `_run_efs_known_scale` and `_run_efs_unknown_scale` become `execution/optimizers/efs.py`, with scale handling as an option. Preserve each loop's exact step and scale-update order, as named branches where they differ.
2. **Fold in the operator controller.** `_run_operator_efs` becomes the same optimizer plus a provider whose `ScoreTerms` carries an error bound. The optimizer's three-way comparison (better / worse / uncertain) reproduces the operator controller's "retain accepted state on uncertain comparisons" policy and its calibration boundaries.
3. **Delete** the old loops and `OperatorEFSFitProvider` (now a `FitProvider`).

**Acceptance:**
- global rules;
- every EFS gate passes (dense, stream, operator, p257–512 calibration, `test_validation_matrix.py -k efs`), with an identical fingerprint;
- EFS controller code is roughly halved; record the actual figure.

---

## Phase 5: One exact streamed REML optimizer

1. **Merge the evaluators.** `evaluate_regular_stream_reml` and `evaluate_nb_stream_reml` become one evaluator, with theta as an optional coordinate supplied by the provider.
2. **Merge the drivers.** `optimize_regular_stream_reml` and `optimize_nb_stream_reml` become one `exact_lbfgs` optimizer. It keeps the shared pinned-first `newton()` completions (pinned scale/theta, free m≤4) and their routing precedence. The completion helpers move to `optimizers/completion.py`.
3. **Merge preflight.** `_preflight_regular_reml` and `_preflight_nb_reml` become one capability-driven preflight.
4. **Canonical driver:** retire `optimize_stream_reml` (canonical known-scale), or keep it as a mode, per design §9 Q3. Either way it shares the evaluator.

**Acceptance:**
- global rules;
- every streamed REML, NB theta-mode, completion-routing and forced-stop test passes with an identical fingerprint;
- #67/#68 public parity cells unchanged.

---

## Phase 6: One PIRLS driver

1. **Extract the step policy.** Create `fitting/pirls_step.py` for the pure kernels (step, midpoint, divergence/domain recovery) and `execution/inner/pirls.py` for the host driver. The policy covers `gam.fit3`/`gam.fit4` step halving, divergence and domain recovery, and convergence.
2. **Make the system a strategy.** `WorkingSystem` selects Fisher, observed, signed, or the EFS system with `gam.fit3` scoring. Choose it from family capabilities.
3. **Port the inner loops one at a time,** each as its own commit with a fingerprint check:
   - dense `fitting/pirls.py`;
   - `fitting/efs_regular_pirls.py`;
   - streamed canonical (`execution/stream.py`);
   - streamed signed (`execution/regular_stream.py`);
   - streamed NB (`execution/nb_stream.py`);
   - operator (`execution/operator_fit.py`);
   - discrete working model (PR11).

   Fused `lax.while_loop` kernels stay fused where they are today. The unification is in the step-policy functions they call, not in turning jitted loops into host loops.
4. **Theta strategies.** Move the nuisance theta strategies to `execution/inner/theta.py`: joint REML, conditional `estimate.theta`, and EFS in-PIRLS theta.

**Acceptance:**
- global rules, with an identical fingerprint for every route;
- JIT compile counts unchanged or justified;
- the PIRLS-loop count goes from about 7 to 1 driver plus strategies; record it.

---

## Phase 7: Newton over providers; dense becomes the in-memory provider

1. **Generalise the newton optimizer.** `execution/optimizers/newton.py` takes curvature from the provider: exact AD Hessian (in-memory), HVP, or central differences of exact gradients.
2. **Route dense through the engine.** Dense `gam()` becomes in-memory provider + cholesky + PIRLS + newton, and the dense-specific orchestration is deleted. This is the highest-risk step: it touches the validated v1.0 path, so it runs last of the numerical phases.
3. **PR9.** `optimize_gaussian_qr_lbfgs` becomes the `gaussian_qr` provider plus `newton`/`exact_lbfgs`.
4. **PR11.** The performance-iteration controller (`advance_*_working_model`) becomes `optimizers/performance_iteration.py` over the discrete provider and `working_model` solver.

**Acceptance:**
- global rules;
- the complete validation matrix and the v1.0 dense R gates pass with an identical fingerprint on host and Linux;
- §8 benchmark timings within noise of the baseline, recorded.

---

## Phase 8: Routing and results

1. **Router.** Add `execution/routing.py`, which builds a `FitPlan` from `FitControl`, family capabilities and provider/solver capabilities. The `api.py` route functions (`_fit_stream_*`, `_fit_operator*`, `_fit_discrete`, `_fit_fixed_sp`, …) collapse into validate → plan → execute → build result.
   - `execution_route` and `lambda_strategy` labels are derived from the plan.
   - Rejection messages are preserved; add a test per message.
2. **Results.** One result builder replaces the per-route constructors (`_from_stream_state` and siblings).
3. **Routing tests.** A table-driven test covers every `FitControl` × family × sp-mode cell, asserting the selected plan and the label.

**Acceptance:**
- global rules;
- the public API is unchanged: same signatures, same result fields;
- every route-selection, rejection and validation test passes.

---

## Phase 9: Lock in

1. **Blocking guardrails.** Switch the duplicate-code check to blocking. Shrink the architecture-test allowlists to empty, or to a reviewed and justified minimum.
2. **Docs.** Update the architecture pages under `docs/architecture/` and `docs/api.md` where internals are described. Add the review checklist from design §7.
3. **AGENTS.md.** Propose wording to the user covering the layers, the dependency rule and "extend via `FitProvider`/solver/strategy, never by copying a controller or PIRLS loop". The user applies it.
4. **Final report.** Before/after line counts per package, duplicate-scan results, compile counts and benchmark timings.

**Acceptance:**
- full pinned Linux suite and native CI green;
- fingerprint identical to the Phase-0 baseline, apart from the documented Phase-1 item 5 case;
- guardrails blocking.

---

## Sequencing and rough size

| Phase | PRs | Risk | Depends on |
|---|---|---|---|
| 0 Harness and guardrails | 1 | low | entry criteria |
| 1 Exact duplicates | 1 | low | 0 |
| 2 Renames and moves | 1 | low | 1 |
| 3 Interfaces and adapters | 1–2 | medium | 2 |
| 4 EFS optimizer | 1 | medium | 3 |
| 5 Exact REML optimizer | 1 | medium | 3 |
| 6 PIRLS driver | 2–3 (one per inner-loop group) | high | 3 (4 and 5 recommended first) |
| 7 Newton and dense provider | 1–2 | high | 6 |
| 8 Routing and results | 1 | medium | 7 |
| 9 Lock in | 1 | low | 8 |

Phases 4 and 5 are independent and may be developed in parallel worktrees, but they publish in order.

**Rollback:** each PR is self-contained and deletes only what it replaces, so reverting the PR restores the previous path. Do not start a phase until the previous one is published and native-green.
