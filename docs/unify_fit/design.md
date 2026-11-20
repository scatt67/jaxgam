# Unified Fitting Engine: Design Document

**Status:** proposed, not started. Its precondition is met: the scaling and EFS work was merged into `main` on 2026-11-20 as pull request #87 (commit `e919938`). The PR labels used below (PR9, PR11, PR12, EFS6, PR13) are those of the scaling plan in [`docs/scale_jaxgam/`](../scale_jaxgam/jaxgam_scaling_implementation_plan.md); all of them are part of that merge. One preparatory step is done: on 2026-11-20 the modules of `fitting/`, `execution/` and `formula/` were grouped into sub-packages by route (§9, question 5), with no change to their contents. Paths in this document use that layout; the file and line counts in §2 were taken before it and are unchanged by it.

For a description of the code as it is today, which this design starts from, see the [architecture pages](../architecture/index.md).

**Plan:** [implementation_plan.md](implementation_plan.md).

**Scope:** internal architecture of `jaxgam/fitting/`, `jaxgam/execution/` and the routing in `jaxgam/api.py`. The public `GAM(...).fit(...)` API, the numerical results and every R-parity tolerance stay unchanged.

---

## 1. Problem statement

JaxGAM now fits the same mgcv model in several ways:
- dense `gam()`;
- streamed exact REML;
- dense and streamed extended Fellner–Schall (EFS);
- the Gaussian frozen-QR route;
- the discrete working model (`bam`/`bgam.fitd`);
- operator/Krylov fits with operator EFS and bulk uncertainty.

Each mode was added as a **vertical slice**, with its own inner PIRLS loop, its own outer controller, its own score evaluator and its own `api.py` route. The new slice usually started as a copy of the previous one and was then specialised. The same mgcv procedures are therefore implemented several times, in both `execution/` and `fitting/`.

That is a correctness risk as well as a maintenance cost. Every R comparison is STRICT, so two ports of one R routine must agree with R *and with each other*, down to operation order. They already drift: see §2.3.

### 1.1 What is *not* the problem

- **The `fitting/` vs `execution/` split itself.** It is prescribed by the scaling plan: pure JIT kernels go in `fitting/`, host control flow and I/O in `execution/`. It is the only structural marker of AGENTS.md's load-bearing Phase-2 rule, and it should stay.
- **The family capability contract** (`FamilyExecutionLineage`, `fisher_equals_observed_for_score`, `row_separable`, …). It is the right abstraction, though it is not yet used everywhere.
- **The EFS provider interface.** `execution/efs_provider.py` (`EFSFitProvider`, `EFSFitRequest`, `EFSControllerContext`) already lets dense and streamed EFS share one controller. That is the pattern this design generalises.

---

## 2. Current state (measured)

Measurements were taken on 2026-10-24 from frozen exports of each line:

| Line | Commit | `execution/` | `fitting/` |
|---|---|---|---|
| Published stack top, #68 | `4352d87` | 16 files / 9,994 lines | 22 files / 10,518 lines |
| PR9 composed | `f3256a8` | 18 / 11,198 | 24 / 11,039 |
| PR11 | `40b7615` (old base) | 8 / 4,289 | 17 / 8,980 |
| PR12+EFS6 | `03bbc7e` | 28 / 14,499 | 27 / 11,403 |
| PR13 | `dcd88d5` | 21 / 11,769 | 26 / 11,343 |

Measured again on `main` after the merge (commit `e919938`, 2026-11-20): `execution/` has 36 files / 21,653 lines and `fitting/` has 33 files / 15,446 lines. The duplicate scans of §2.2 have not been re-run on that commit; do that in Phase 0.

### 2.1 Parallel implementations of the same procedure

| mgcv procedure | JaxGAM implementations |
|---|---|
| PIRLS step control (`gam.fit3`/`gam.fit4` step halving, divergence and domain recovery) | `fitting/pirls.py` (dense; 1,765 lines incl. an EFS-specific beta step); `fitting/efs_regular_pirls.py` (EFS regular, 635); `execution/stream/pirls.py` (streamed canonical); `execution/stream/regular.py` (streamed signed); `execution/stream/nb.py` (streamed NB); `execution/operator/fit.py` (operator, PR12); `execution/discrete/gaussian.py`/`working_model.py` (discrete, PR11) |
| Outer smoothing-parameter controller | `fitting/newton.py::NewtonOptimizer` (dense, host loop inside `fitting/`); `execution/stream/reml.py::optimize_stream_reml` and `optimize_regular_stream_reml`; `execution/stream/nb_reml.py::optimize_nb_stream_reml`; `execution/gaussian_qr/optimizer.py::optimize_gaussian_qr_lbfgs` (PR9); `execution/efs.py::_run_efs_known_scale` and `_run_efs_unknown_scale`; `execution/operator/efs_controller.py::_run_operator_efs` (EFS6); `execution/discrete/working_model.py::advance_gaussian_working_model` (PR11) |
| REML/score assembly | `fitting/reml.py` (dense); `fitting/stream/reml.py`; `fitting/stream/nb_reml.py`; `fitting/gaussian_qr/kernel.py` (PR9); `fitting/discrete/working_model.py` (PR11, `Sl.fitChol`); `fitting/operator/efs_score.py` + `execution/operator/score.py` (EFS6) |
| `efsudr` raw update | `fitting/efs.py::efs_raw_update` (JAX); `execution/operator/efs_controller.py::_raw_ratio` + `_trial_rho` (NumPy) |
| NB theta | `fitting/efs_theta.py` (conditional Newton); `fitting/stream/nb_theta.py` + `execution/stream/nb_theta.py` (streamed); joint-REML theta in `execution/stream/nb_reml.py`; PR11 milestone B will add `estimate.theta` |
| Saturated log-likelihood reduction | `fitting/family_execution.py::batch_saturated_loglikelihood`; `fitting/stream/kernels.py::saturated_loglik_reduction` |
| Route preflight | `execution/stream/reml.py::_preflight_regular_reml`; `execution/stream/nb_reml.py::_preflight_nb_reml`; `execution/stream/pirls.py::_preflight`; plus `api.py` `_check_stream_*` helpers |
| Public routing | `api.py`: `_fit_stream`, `_fit_stream_gaussian_qr`, `_fit_stream_fixed_regular`, `_fit_stream_fixed_nb`, `_fit_stream_exact_reml`, `_fit_stream_parametric_bypass`, `_fit_operator`, `_fit_operator_efs`, `_fit_discrete`, `_fit_fixed_sp`; about 1,400 lines across lines; six `execution_route` labels |

### 2.2 Near-duplicate scans

Two scans were run, both over functions with at least 15 body lines.
- **Token scan:** token 5-gram Jaccard, which finds copy-paste.
- **Structural scan:** normalised-AST `SequenceMatcher` ratio, which finds "same algorithm, renamed".

Representative results:

| Pair | Structural | Token |
|---|---|---|
| `fitting/efs_theta.py::_conditional_theta_nll_jit` ↔ `fitting/stream/nb_theta.py::objective` | 0.98 | — |
| `fitting/efs_regular_pirls.py::midpoint` ↔ `fitting/pirls.py::midpoint` | 0.98 | — |
| `execution/stream/nb_reml.py::evaluate` ↔ `execution/stream/reml.py::evaluate` | 0.97 | — |
| `execution/stream/nb.py::scan_at` ↔ `execution/stream/regular.py::scan_at` | 0.96 | — |
| `execution/efs.py::_run_efs_known_scale` (175 L) ↔ `_run_efs_unknown_scale` (192 L) | 0.86 | — |
| `execution/stream/reml.py::optimize_regular_stream_reml` ↔ `execution/stream/nb_reml.py::optimize_nb_stream_reml` | 0.72 | — |
| `execution/stream/reml.py::evaluate_regular_stream_reml` (195 L) ↔ `execution/stream/nb_reml.py::evaluate_nb_stream_reml` (193 L) | 0.80 | 0.36 |
| `fitting/family_execution.py::batch_saturated_loglikelihood` ↔ `fitting/stream/kernels.py::saturated_loglik_reduction` | 0.80 | 0.42 |
| `execution/operator/efs_controller.py::_indexed_poisson_score_bound` ↔ `_indexed_gaussian_score_bound` | — | 0.68 |
| `execution/operator/efs_controller.py::_run_operator_efs` (284 L) ↔ `execution/efs.py::_run_efs_known_scale` | 0.42 | — |
| `fitting/efs_regular_pirls.py::_efs_regular_pirls_loop_jit` ↔ `fitting/pirls.py::_pirls_loop_jit` | 0.12 | — |

Exact helper copies: `_factor_inverse_residual` is identical in `execution/stream/reml.py` and `execution/stream/nb_reml.py`; `_link_name` is near-identical in `execution/stream/nb.py` and `fitting/stream/nb_reml.py`.

The low PIRLS and operator-EFS ratios matter. Most of the duplication is **algorithmic**, not textual: the same R procedure is re-derived with different names and data access. A copy-paste linter alone will not find it.

### 2.3 Observed drift between ports

- **`efsudr` ratio order.** `fitting/efs.py::efs_raw_update` computes `phi * numerator / max(0, q)`. `execution/operator/efs_controller.py::_raw_ratio` computes `numerator / max(0, q) * phi`. Both claim pinned `efsudr` order, and they can differ in the last bit.
- **Family allowlists.** Between 36 and 43 `isinstance(family, (Gaussian|Poisson|…))` checks in `execution/` and `fitting/` per line, instead of capability queries; on `main` after the merge there are 46. The family capability contract exists so that such allowlists are not needed.
- **Host control flow in `fitting/`.** `fitting/newton.py::NewtonOptimizer` (`while` loops with `bool(jnp.isfinite(...))` host pulls) and `fitting/data.py::FittingData.from_setup` (NumPy preparation) break the "`fitting/` is JIT-only" rule. The scaling plan's audit already noted this. At the time of the October measurement nothing in `fitting/` imported `execution/` or host SciPy. On `main` after the merge there is one such import: `fitting/data.py::_structural_penalty_roots` imports `structural_rank` from `execution/qr.py` inside the function, to avoid a circular import. Phase 2 should remove it.
- **Misleading pairs.** `fitting/reml.py` is the dense criterion. `execution/stream/reml.py` is the *streamed* driver and does not use it; its kernels are in `fitting/stream/reml.py`. The same holds for `execution/stream/nb_reml.py` ↔ `fitting/stream/nb_reml.py`.

### 2.4 Root cause

There is no shared interface between "how data is accessed" and "which mgcv algorithm runs on it". EFS has one (`EFSFitProvider`), and that is why dense and streamed EFS share a controller. Everything else couples:
- data access (in-memory X, RowSource batches, discrete tables, operators);
- coefficient solving (Cholesky, positive QR, signed QR, PCG, `Sl.fitChol`);
- the inner loop;
- the outer optimizer.

All four are coupled in each route.

---

## 3. Decision: keep `GAM`; BAM features are strategies, not a replacement

Deprecating the dense `gam()` path in favour of a BAM-only design was considered and **rejected**:

1. **Different estimator.** `bam`'s performance iteration re-estimates smoothing parameters on a working model at each IRLS step. For non-Gaussian families its fixed point differs from `gam`'s Laplace-REML optimum, and it is not guaranteed to converge. Only Gaussian/identity is equivalent.
2. **Discretization is an approximation.** `bam(discrete=TRUE)` changes the fitted model.
3. **The correctness contract is `gam` parity.** AGENTS.md, the validation matrix and all published PRs validate against pinned `gam` at STRICT.
4. **mgcv keeps both deliberately:** `gam` is exact and usually faster at moderate n; `bam` is for data that does not fit.

BAM's distinctive pieces remain opt-in strategies on the shared engine: performance iteration (`smoothing_strategy="working_model_reml"`), fREML working-model criterion, and discretization (`execution="discrete"`).

---

## 4. Goals and non-goals

**Goals**

1. **One implementation per mgcv procedure:**
   - PIRLS step control;
   - each outer optimizer (`newton`, exact L-BFGS-B plus `newton()` completion, `efsudr`, `bgam.fitd`);
   - REML score assembly from sufficient statistics;
   - the `efsudr` update;
   - each NB theta strategy;
   - the saturated log-likelihood.
2. **Data access behind one provider interface.** Dense becomes the in-memory provider, not a separate stack.
3. **Capability-driven routing.** Families, providers and solvers declare capabilities, and the router derives a `FitPlan`. No family-name allowlists.
4. **An honest boundary.** `fitting/` holds only pure JIT kernels; host loops live in `execution/`. Both are enforced by tests.
5. **Bit-identical numerics.** Every unification is proven by before/after equality of fit outputs, not by tolerance.
6. **Guardrails** (architecture tests plus duplicate-code lint) so the duplication does not come back.

**Non-goals**

- No public API change. `GAM`, `FitControl`, result types and `execution_route` labels are preserved. The labels become derived from the `FitPlan`.
- No numerical change. Tolerances, oracle tests and digest-bound allowances are untouched. Where two current ports disagree (§2.3), the plan picks the one that matches pinned R and proves every affected STRICT gate still passes. That is the only permitted output change, and it must be documented per case.
- No new features, families, solvers or performance work.
- No change to Phase 1 (setup) or Phase 3 (post-estimation), beyond consuming a unified result builder.

---

## 5. Target architecture

```
jaxgam/api.py                  GAM facade: validate → plan → execute → build result
jaxgam/execution/routing.py    capability matching → FitPlan(provider, solver, inner, optimizer, uncertainty)
jaxgam/execution/optimizers/   host controllers, one per mgcv outer algorithm
    newton.py                    gam.outer/newton(); exact or finite-difference curvature from the provider
    exact_lbfgs.py               bounded L-BFGS-B + shared newton() completion (pinned and free)
    efs.py                       efsudr, incl. scale and theta options and uncertain score comparisons
    performance_iteration.py     bgam.fitd (working-model Sl.fitChol steps, estimate.theta nuisance)
    completion.py                shared newton() acceptance (today's _newton_model_rejection and friends)
jaxgam/execution/inner/        one PIRLS host driver; nuisance (theta/scale) strategies
    pirls.py                     the step-control policy, parameterised by provider + solver + system
    theta.py                     joint-REML, conditional (estimate.theta) and EFS in-PIRLS theta
jaxgam/execution/providers/    data access; each implements FitProvider
    in_memory.py                 today's dense X/FittingData path
    stream.py                    RowSource batches (regular and NB), replay/lineage checks
    gaussian_qr.py               frozen Gaussian QR statistics (PR9)
    discrete.py                  compressed tables / discrete operators (PR11)
    operator.py                  matvec/Krylov providers, arrowhead/indexed (PR12/EFS6)
jaxgam/execution/solvers/      coefficient-system solvers behind CoefficientSolver
    cholesky.py  qr.py  signed_qr.py  pcg.py  working_model.py (Sl.fitChol, pinned DPSTRF)
jaxgam/execution/uncertainty/  selected and bulk variance (PR13), Fisher/observed factors
jaxgam/fitting/                pure JAX kernels only (no host loops, no I/O, no SciPy)
    family/ (link/variance/deviance/derivative kernels), pirls_step.py, penalty_ops.py,
    reml_terms.py (score from sufficient statistics), efs_update.py, linalg actions (qr/signed_qr),
    the route kernels: stream/, discrete/, operator/, gaussian_qr/
```

Today `execution/` and `fitting/` are grouped by route, not by role: `execution/stream/`, `execution/discrete/`, `execution/operator/` and `execution/gaussian_qr/` each hold one route's drivers, and `fitting/` has the same four folders for its kernels. Each provider above is carved out of the matching route folder; the optimizers, the inner driver and the solvers are what the route folders have in common.

**Dependency rule:** `api → routing → optimizers → inner → providers/solvers → fitting`. Nothing in `fitting/` imports `execution/`. Providers and solvers never import optimizers. Phase-1 modules never import JAX (already enforced).

### 5.1 Core interfaces

These are typed `Protocol`s and frozen dataclasses, following existing JaxGAM style. The names are indicative; the existing `EFSFitProvider` is the seed.

```python
class FitProvider(Protocol):
    capabilities: ProviderCapabilities  # exact_gradient, exact_hessian, hessian_vector,
                                        # approximate_score (with error bound), replayable,
                                        # memory model, supported solver kinds
    def start(self, request: StartRequest) -> FitState: ...
    def fit(self, request: FitRequest, warm: FitState | None) -> FitState:
        """Reconverged coefficients at (rho, theta, phi) via the plan's inner loop and solver."""
    def score(self, state: FitState) -> ScoreTerms:
        """Exact score terms, or an estimate plus an error bound."""
    def gradient(self, state: FitState) -> np.ndarray: ...  # adjoint / exact derivative
    def curvature(self, state: FitState) -> Curvature | None: ...  # exact Hessian, HVP, or None
    def efs_statistics(self, state: FitState) -> EFSStatistics: ...  # d, t, q (+ bounds)
    def work(self) -> WorkCounters: ...  # scans, batches, PIRLS passes, refits, timings

class CoefficientSolver(Protocol):
    kind: Literal["cholesky", "qr", "signed_qr", "pcg", "working_model"]
    def factor(self, system: WorkingSystem) -> Factor: ...
    def solve(self, factor: Factor, rhs) -> ...: ...
    def log_determinant(self, factor: Factor) -> ...: ...
    def traces(self, factor: Factor, penalties) -> ...: ...

class Optimizer(Protocol):
    def run(self, provider: FitProvider, start: FitState,
            control: OptimizerControl) -> OptimizationResult: ...
```

- **`FitState`** is observation-independent: β, ρ, θ, φ, factors, the fit fingerprint and lineage. Providers own row access.
- **`OptimizationResult`** carries truthful convergence, the stop reason, completion diagnostics and the work counters.
- **One result builder** maps `(FitPlan, OptimizationResult, provider diagnostics)` to `GAMResults`, `GAMInferenceResult` or `GAMPredictionResult`.

### 5.2 How each existing route maps

| Today | `FitPlan` |
|---|---|
| dense `gam()` REML | in-memory provider + cholesky solver + PIRLS (Fisher/observed system per family capability) + newton optimizer (exact AD Hessian) |
| streamed fixed sp | stream provider + qr/signed_qr solver + PIRLS + no optimizer (or the pinned-scale/theta completion only) |
| streamed exact REML (#68) | stream provider + qr/signed_qr + PIRLS + exact_lbfgs (+ completion) |
| Gaussian frozen QR (PR9) | gaussian_qr provider + qr + no PIRLS (closed form) + newton/exact_lbfgs |
| dense or streamed EFS | in-memory or stream provider + PIRLS (EFS system) + efs optimizer |
| discrete working model (PR11) | discrete provider + working_model solver + performance_iteration optimizer |
| operator fixed sp / operator EFS (PR12/EFS6) | operator provider + pcg solver + PIRLS + efs optimizer (score comparison may return "uncertain") |
| bulk uncertainty (PR13) | uncertainty module consuming the final `FitState` + provider factor/trace actions |

### 5.3 Where the variation lives

Each difference between routes moves to exactly one place:

- **Coefficient system** (Fisher vs observed vs signed): a `WorkingSystem` strategy chosen from family capabilities. This removes the parallel PIRLS loops; `efs_regular_pirls` becomes PIRLS plus the EFS system and `gam.fit3` scoring option.
- **Scale and theta handling:** optimizer options (EFS known/unknown scale become one loop) and `inner/theta.py` strategies (joint REML, conditional `estimate.theta`, EFS in-PIRLS theta).
- **Exact vs approximate scores:** `ScoreTerms` carries an optional error bound. Optimizers use a three-way comparison (better / worse / uncertain). This is what the operator EFS controller does today in its own copy.
- **Curvature availability:** the provider capability. The newton optimizer uses the exact Hessian (in-memory AD), HVPs, or central differences of exact gradients (as the free completion in #68 already does).
- **Family/link support:** capabilities only. Unsupported combinations are rejected by the router with the same messages as today.

---

## 6. Numerical-fidelity rules

1. **Operation order is part of the contract.** A unified kernel must reproduce the exact floating-point sequence of the implementation it replaces. Where routes legitimately differ in order (e.g. `gam.fit3` vs `gam.fit4` scoring), the difference becomes an explicit, named strategy, not an accident.
2. **Bit-identity first.** Every unification PR runs the fingerprint harness (plan Phase 0) before and after. All fitted fields of the validation matrix and the execution tests must be **bitwise identical**, on host and on pinned Linux. Any intended exception (§2.3 drift) is listed per case with its R comparison, and must keep every STRICT gate.
3. **No tolerance edits.** `tests/tolerances.py`, digest-bound allowances and oracle tests are not modified. A failing gate means the refactor is wrong.
4. **R references stay typed rpy2.** No new oracles are needed; existing gates are the regression net.

---

## 7. Guardrails

- **Architecture test** (`tests/test_architecture.py`). It checks:
  - `fitting/` never imports `execution/`, host `scipy`, `sqlite3` or file I/O;
  - Phase-1 packages never import JAX;
  - providers and solvers never import optimizers;
  - no `isinstance(<family>, <concrete family>)` in `execution/` or `fitting/` outside a short reviewed allowlist (target: empty).
- **Duplicate-code lint.** A `make lint` step runs a duplicate detector, pylint `duplicate-code` or `jscpd`, chosen in Phase 0. It starts in report-only mode against a committed baseline and becomes blocking in the final phase.
- **Review checklist** (in `docs/unify_fit/` and the PR template): a new execution mode implements `FitProvider`, or a solver or strategy, and must not add a controller or PIRLS loop.
- **AGENTS.md:** propose wording for the user to apply, covering the layers, the dependency rule and "extend via providers". Agents do not edit AGENTS.md themselves.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Behaviour change in the validated v1.0 dense path | Do the dense → in-memory-provider step last, behind bitwise fingerprints and the full validation matrix |
| Hidden operation-order dependence breaks STRICT parity | Fingerprint harness at every step; one unification per PR; keep the old path behind an internal switch until the fingerprint matches, then delete it in the same PR series |
| JIT recompilation or performance regressions from extra indirection | Keep kernels monomorphic and jitted at the same boundaries; record compile counts and §8 benchmark timings before and after; abstractions live on the host side |
| Large churn conflicts with other work | Start only after all current lines are published; the unification series is the only active work |
| Over-abstraction | Interfaces are extracted from the existing EFS provider and real routes, not designed speculatively; each protocol method has at least two implementations |

---

## 9. Open questions (resolve in Phase 0)

1. **Duplicate-detector tool:** pylint `duplicate-code` (already Python tooling) or `jscpd` (token-based, faster). Confirm the threshold and baseline format.
2. **`FittingData`:** should it become the in-memory provider's internal state, or stay as a shared Phase-1→2 transfer type? Decide from its current call sites.
3. **Retiring `optimize_stream_reml` (canonical known-scale):** retire it outright, or keep it as a mode of `exact_lbfgs`. Its regime is a strict subset of `optimize_regular_stream_reml`.
4. **The two §2.3 port disagreements:** which order matches pinned `efsudr`? Verify against `R/gam.fit4.r` 1.9-3 before unifying.
5. **Sub-packages by route first?** (raised 2026-11-20; **decided and done the same day**) `fitting/`, `execution/` and `formula/` were grouped into sub-packages by route (`stream/`, `discrete/`, `operator/` and, in the first two, `gaussian_qr/`) in one move-only change; dense and shared modules stay at the top of each package. It replaced the renames that plan Phase 2 item 1 listed, removed the module names that existed in both packages without being a pair, and makes each route folder the starting point of the matching provider in §5.
