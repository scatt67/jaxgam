**JaxGAM: implementation plan for large datasets and large coefficient counts**

> **Status (2026-11-20): delivered.** The work planned here was built and merged
> into `main` as pull request #87: the streamed route, the Gaussian QR
> compression, the discretized route (the analogue of `bam(discrete=TRUE)`),
> the operator route with bulk uncertainty, and the acceptance harness
> (`scripts/benchmark_scaling.py`). The text below is the plan as written and
> amended during the work. It is kept as a record and is not maintained: its
> PR numbers are the plan's own, and it refers to working documents that are
> no longer in the repository. For what was built see the
> [architecture pages](../architecture/index.md); for how to use it see the
> [user guide](../guide/choosing_a_route.md).
>
> Known limits at the time of the merge: the Gaussian QR route can stop short
> of the REML optimum above 50 penalties (it reports `converged=False`); exact
> streamed REML did not converge on some many-smooth and near-Poisson
> negative binomial cases and reports that; mgcv's rank test with column drop
> for structurally rank-deficient dense models is not ported.

Target: [scatt67/jaxgam at c1a31346188186bbd2fe67eea8d0d51ba6c95d79](https://github.com/scatt67/jaxgam/tree/c1a31346188186bbd2fe67eea8d0d51ba6c95d79). Prepared 2026-09-05; audited against the working repository and local mgcv source on 2026-09-06.

This is an implementation roadmap, not a claim that every described backend is implemented. The original audit below describes the September 6 baseline; the delivery log that accompanied it is no longer in the repository. API sketches describe intended interfaces unless explicitly identified as shipped. The earlier draft quoted CPU probes without checked-in scripts, inputs, or results; those measurements were removed as acceptance evidence. Performance and numerical improvements must be measured by the implementing PRs.

**September 8 scope correction:** published plan PR7 ([GitHub #40](https://github.com/scatt67/jaxgam/pull/40)) is the initial canonical-family, cross-product/Cholesky backend only. It does not complete general QR/TSQR solving or all-family streaming. Those are mandatory new PRs **7.1–7.6**, with exact-REML extensions **8.1–8.3**, specified below. Preserve original PR1–13 numbering; decimal IDs identify additional, separately reviewed PRs, not changes folded into published PR7. Internal development labels such as PR8a/provider and PR8b/L-BFGS are not these new decimal-numbered PRs.

**0. Audited baseline and source provenance**

`echo $MGCV_SOURCE` resolves to `/Users/shanecatts/Documents/GitHub/mgcv`. Its checked-out source is **mgcv 1.9-4**, commit `1b6a4c8374612da27e36420b4459e93acb183f2d`. The repository's test oracle remains **R 4.5.2 + mgcv 1.9-3**, enforced by [RBridge.check_versions](../../tests/r_bridge.py); the local clone contains the `1.9-3` tag at `fb7e8e718377513e78ba6c6bf7e60757fc6a32a9`. All mgcv links in this document point to the inspected 1.9-4 commit. Read the files locally and compare with the pinned tag before porting behavior:

~~~sh
echo $MGCV_SOURCE
git -C "$MGCV_SOURCE" rev-parse HEAD
rg -n '^(qr_update|bgam.fitd|bam.fit) ' "$MGCV_SOURCE/R/bam.r"
git -C "$MGCV_SOURCE" show 1.9-3:R/bam.r
git -C "$MGCV_SOURCE" diff 1.9-3 HEAD -- R/bam.r R/fast-REML.r
~~~

This comparison matters: 1.9-4 changes `bgam.fit`/`bgam.fitd` nuisance-scale updates and reported criteria, and adds `Sl.fit`'s `reml.pen` return. These are not interchangeable with the 1.9-3 oracle. Keep the clone at its current checkout; `git show` is sufficient for reading the pinned implementation.

| Area | Actual implementation at the audited JaxGAM commit | Consequence for this plan |
|---|---|---|
| Public fitting API | [api.py](../../jaxgam/api.py): `GAM(...).fit(data, weights=None, offset=None, *, result="full")`; data is a DataFrame or dict; `result="inference"` already exists | Preserve both result modes. `FitControl`, `RowSource`, and `result="prediction"` below are additions. Unknown constructor options, including `control`, currently raise `TypeError`. |
| Execution and methods | `api._check_scope_guards`: only the JAX backend and REML are supported; `method="ML"` raises `NotImplementedError` | CPU fitting also uses JAX. Do not claim an existing NumPy fitting backend, ML parity, or an execution router. Fixed `sp` uses `_fit_fixed_sp`; estimated-theta NB can still require outer optimization with lambda pinned. |
| Families and smooths | [families/registry.py](../../jaxgam/families/registry.py) includes `nb`; [smooths/registry.py](../../jaxgam/smooths/registry.py) includes `tp/ts`, `cr/cs/cc`, `gp`, `re`, `te/ti` | NB, GP, and modest dense random effects are already implemented despite older scope prose. `ps/cp/bs`, `fs`, sparse fitting, fREML, and Fellner-Schall remain deferred. |
| Penalties and fitting coordinates | `ModelSetup.build` embeds constrained penalties globally; `FittingData.from_setup` constructs dense `D_global` and transforms X and every S; `_build_block_metadata` and `jax_utils.block_log_det_S` already handle block determinants | PR 2 changes storage and applications of existing algebra. Preserve `TermBlock` ordering and existing rank thresholds rather than creating a second determinant implementation. |
| Prediction state | [formula/predict_matrix.py](../../jaxgam/formula/predict_matrix.py): `PredictSpec`, `build_predict_spec`; each smooth has `copy_for_prediction`; [inference/_core.py](../../jaxgam/inference/_core.py) shares prediction finishing | Observation-independent prediction metadata already exists. Extend or compose it in `PreparedModel`; keep one factor/constraint/prediction contract. |
| Result costs | `GAMResults._from_fit` computes dense inverse/covariance, EDF/EDF1, and transfers mu/eta before selecting the inference branch | Lean retention does not make fitting or result construction bounded in n or p. PR 6 must move optional work behind the result choice. |
| Fitting boundary | `pirls._pirls_loop_jit` is a fused JAX loop; `NewtonOptimizer` already uses host control flow; `FittingData.from_setup` and `initialize_beta` perform NumPy preparation | Preserve pure compiled kernels and NumPy setup/prediction. Put new file scanning and optimizer orchestration in `execution/`; do not describe all existing `fitting/` functions as JIT kernels. |

`docs/design.md` (since removed from the repository) was the architecture reference at the time, but its examples are not proof of implemented APIs. Read §1.2 for the scope boundary, §5.10/§13.2 for setup contracts, §7/§8 for PIRLS/REML, §10.2/§10.5 for proposed structured/chunked designs, and §18.1/§18.2 for validation. In this version, §4.3 is **Cython Fallback Kernels** and §4.4 is **JAX Purity Boundary**, not the REML/ML sections suggested by older source-map guidance. The local mgcv top-level file is `R/mgcv.r`, and NB is in `R/efam.r`. Use the function-level source map in §10 below.

**1. Delivery strategy and compatibility contract**

The plan contains 13 top-level delivery items, explicit follow-up PRs 7.1–7.6 and 8.1–8.3, and the smaller PR13 implementation pieces in the [extended Fellner–Schall sub-plan](fellner_schall_implementation_plan.md). The number of implementation PRs is therefore greater than 13. PRs 1–4 optimize the current dense implementation. The streamed/discrete/operator backends and new smoothing strategies are post-v1.0 work: their delivery must include corresponding scope/API/design updates, while existing unsupported-feature guards remain effective until each path is implemented. The negative-binomial derivative optimization is a standalone PR 4 and can ship independently of the backend refactors. Dense exact-trace EFS can also start independently of the operator backend; it is a new optimizer, not an exact-objective storage refactor.

Delivery milestones (the dense Gaussian specialization in PR 9 can land earlier):

1. Remove avoidable dense work while retaining the current numerical method.
2. Separate reusable basis metadata from observations, then ship streamed prediction, fixed-smoothing PIRLS, and exact-objective REML gradients.
3. Specialize Gaussian identity-link fitting so smoothing optimization no longer scans observations.
4. Add two distinct large-model options: discretized designs with direct solves, and operator designs with iterative solves.

Stage 4 is a branch: discretization and operator execution can be combined, but neither should be a prerequisite for the other. Discretization targets expensive basis evaluation and repeated covariate values. Operators target coefficient counts for which dense cross-products and factorizations are unacceptable.

Define n as rows, d as input columns, p as constrained coefficients, m as smoothing parameters, t as coefficient term blocks, B as batch size, k_j as a term's coefficient count, r_j as its kernel-knot count, and q_j as its number of discrete covariate values. A tensor or factor-by term can have several smoothing parameters, so t and m are not interchangeable. Route on p, m, representation, measured workspace, and data-pass cost. An n-only threshold is inadequate.

Preserve these contracts:

- Existing calls to GAM.fit(DataFrame) retain their current defaults and result semantics during migration.
- The existing dense solver remains the numerical reference.
- Phase 1 basis and penalty construction stays NumPy/SciPy. JAX adapters and kernels live in the fitting layer.
- Keep CoefficientMap as the mapping from public constrained coefficients to raw basis coefficients. Retain the separate fitting reparameterization already in FittingData, storing its transforms block by block.
- Exact means the same frozen basis, penalty conventions, and statistical objective to declared numerical tolerances. It does not mean bitwise equality, identical line-search histories, or guaranteed selection of the same local optimum.
- Preserve the current knot-selection policy for an exact storage refactor. Changing that policy, discretizing continuous covariates, changing to working-model REML, and estimating traces stochastically are explicit choices recorded in results. Inherited TPRS/GP knot subsampling must also be recorded; it is already part of the dense basis, and currently differs from R above the knot cap (PR 5).
- Broadening beyond the current dense-only v1 scope requires updating the repository scope and architecture text as part of the change. The current planning pseudocode is not executable specification.

Useful first production target: replayable input, additive cubic smooths, canonical Gaussian/Poisson/binomial, fixed or exactly optimized smoothing, and p small enough for dense coefficient-space algebra. This is an initial milestone, not the final family-support boundary. Completion of direct streaming requires all five currently registered families (Gaussian, Poisson, binomial, Gamma, and NB), their supported non-default links, and both fixed and estimated NB theta. PR7.1 owns an explicit inventory of advertised family/link combinations from the actual registry, public documentation, dense implementation and pinned R; constructor acceptance alone is not proof that a combination is supported. Every supported combination must have an assigned implementation/parity gate. A missing combination remains unfinished work, not an implicit scope reduction.

Streaming is an execution policy, independent of the family and smoothing optimizer. Extend the existing family interfaces with batch-safe mathematics, bounded reductions and explicit capabilities; do not grow family-name allowlists throughout the stream engine. Future row-separable families should integrate through that contract without changing host iteration. Non-row-separable likelihoods require a specialized reduction adapter and separate validation; this does not authorize implementing additional distribution families in this task. Family coverage and prepared-basis coverage are separate axes: these follow-ups do not imply that PR5 already prepares every smooth/factor interaction.

Adapt mgcv's established algorithms before designing alternatives: `fixDependence`, the `Sl.*` penalty operations, `qr_update`, `bam.fit`, and `XWXd`/`XWyd`/`Xbd` are concrete starting points. Preserve JaxGAM's public coefficients and frozen basis when changing representation. The host-streamed adjoint, L-BFGS storage choice, and stochastic operator methods are additional JaxGAM work; the cited mgcv routines do not already implement those backends.

**2. Proposed interfaces**

Separate these concerns instead of adding an optional X field throughout the current code:

| Component | Ownership and responsibility |
|---|---|
| PreparedModel | Compose existing PredictSpec/TermBlock metadata with fitting transforms, local penalties, normalization constants, schema, and rank metadata; do not duplicate factor or constraint logic |
| RowSource | Repeatable scans over raw covariates, response, prior weights, offsets, and stable row positions |
| DenseDesign | PreparedModel plus the materialized dense design; adapter for the existing backend |
| StreamDesign | PreparedModel plus RowSource; creates only bounded design batches |
| StatisticsProvider | Batch-reduced fitting statistics, trial objectives, and derivative contractions |
| SolveState | Coefficients, factors/operators, objective, convergence diagnostics, and parameter values; no required training-row arrays |
| CovarianceProvider | Optional factor- or operator-based prediction variance; explicit dense covariance materialization |

Introduce these roles only as needed by the relevant PR, starting with adapters around `ModelSetup`, `FittingData`, and `PredictSpec`. The CPU-side schema can begin as:

~~~python
from dataclasses import dataclass
from typing import Iterator, Protocol
import numpy as np

@dataclass(frozen=True)
class PenaltyBlock:
    term_id: int
    start: int
    stop: int
    sp_indices: tuple[int, ...]
    local_penalties: tuple[object, ...]  # DenseLocal, Diagonal, Identity
    transform: object                  # local D, diagonal D, or Identity
    range_metadata: object

@dataclass(frozen=True)
class RowBatch:
    columns: dict[str, object]         # preserve categorical dtype and ordered levels
    y: np.ndarray | None              # required for fitting, absent for new-data prediction
    weight: np.ndarray
    offset: np.ndarray
    valid: np.ndarray                  # distinguishes padding from real rows
    row_positions: np.ndarray         # stable source positions, including filtered scans

class RowSource(Protocol):
    def scan(self, batch_rows: int) -> Iterator[RowBatch]: ...
    def fingerprint(self) -> str: ...

class StatisticsProvider(Protocol):
    def working_stats(self, beta, theta): ...
    def trial_objectives(self, beta, direction, alphas, theta): ...
    def final_stats(self, beta, theta, need_fisher: bool): ...
    def statistic_vjp(self, beta, theta, cotangents): ...
~~~

Use stable term IDs or positional TermBlocks, never labels as identifiers: the existing code deliberately supports label-colliding smooths.

Add explicit control options rather than overloading the currently accepted backend="jax":

~~~python
# Proposed API, not supported by the audited commit.
control = FitControl(
    execution="stream",              # dense | stream | discrete | operator
    memory_budget_bytes=8 << 30,
    batch_rows=8192,
    basis_policy="exact",
    linear_solver="cholesky",
    smoothing_strategy="exact_reml",
    sp_optimizer="lbfgs",
)

fit = GAM(formula, family="poisson", control=control).fit(
    source,
    result="prediction",             # new mode: no mandatory Vp/EDF reporting
)

for row_positions, predictions in fit.predict_iter(new_source):
    prediction_sink.write(row_positions, predictions)
~~~

Batch size here is an example; the allocator must shrink it to fit the declared host and device budgets. Initially require explicit execution selection. Later, automatic routing may choose equivalent exact implementations, but must not enable an approximation without an explicit approximation policy. Add the public option, exports, constructor validation, and result-mode overloads together in `api.py`/`__init__.py`, with tests in `tests/test_api/test_gam.py` and `tests/test_result_mode.py`. The sketch does not authorize accepting and silently ignoring new options.

**3. Stage 1: block structure and duplicate-work removal**

**PR 1 — remove full-Q allocation in dependence detection.**

Affected: [smooths/constraints.py, fix_dependence](https://github.com/scatt67/jaxgam/blob/c1a31346188186bbd2fe67eea8d0d51ba6c95d79/jaxgam/smooths/constraints.py#L602).

Read first: [`R/mgcv.r::fixDependence`][r-dependence], especially `qr(X1, LAPACK=TRUE)` followed by `qr.qty(qr1, X2)`. R already applies the packed reflectors without constructing full Q. The Python change should reproduce that implementation, including its trailing-R-block mean threshold; it is not a new dependence algorithm. `augment.smX` and `gam.side` in the same file explain the penalty augmentation and column deletion bookkeeping.

The first QR currently defaults to full mode. For a tall X1 it creates an n-by-n Q; overlapping smooths can reach this path. At one million rows, an unaugmented float64 Q alone would require 8 TB.

Use pivoted raw Householder QR of X1, then apply Q-transpose to X2 with LAPACK ormqr. Keep the existing residual QR, pivot mapping, rank threshold, and column-drop logic. Simply changing the call to economic QR is incorrect: the current algorithm needs the orthogonal-complement rows of Q-transpose times X2.

~~~python
# Core operation; handle LAPACK status/workspace and edge cases in production.
(packed, tau), R1, piv1 = scipy.linalg.qr(
    X1, pivoting=True, mode="raw"
)
QtX2 = apply_householder_qt(packed, tau, X2)  # ormqr; never form full Q
residual_coordinates = QtX2[X1.shape[1]:, :]
~~~

Gate: extend `tests/test_constraints.py` for identical selected dependent columns on current exact, nested, repeated-smooth, and rank-deficient fixtures, plus direct `fixDependence` comparisons through RBridge; no n-by-n allocation. Exercise tall inputs and the currently supported dimensional edge cases. Record reflector-residual error in the test rather than relying on the earlier untracked probe.

**PR 2 — migrate penalty storage and fitting transforms.**

Affected: formula/design.py, penalties/penalty.py, fitting/data.py, jax_utils.py, fitting/newton.py, fitting/reml.py, and results.py. Add penalties/structure.py and fitting/penalty_ops.py.

Read first: [`R/fast-REML.r::Sl.setup`][r-sl-setup] for local `start/stop`, `S`, `D/Di`, penalized indices, rank, and splitting disjoint multiple penalties into singletons; [`Sl.initial.repara`][r-sl-repara] for coefficient/model-matrix/covariance transformations; [`Sl.Sb`/`Sl.rSb`][r-sl-action], [`Sl.addS`][r-sl-add], and [`ldetS`][r-ldets] for the operations below. R's diagonal singleton branch retains D as a vector. Its dense `Sl.setup` also creates global E/S rank-check matrices: port the block representation and preserve rank semantics, but do not copy those allocations into the eventual operator path.

Migrate `_compute_repara_D`, `_build_block_metadata`, and `block_log_det_S`; they already implement much of this logic. `Penalty`/`CompositePenalty` currently require dense arrays, so keep compatibility materialization explicit at their consumers during migration. A local transform belongs to a coefficient block, which may own multiple S_j; it is not one independently chosen D per overlapping penalty.

- Stop embedding each S_j into global p-by-p coordinates in ModelSetup.build.
- Group overlapping penalties by coefficient block. Factor-by levels may have disjoint subblocks; tensor penalties overlap within one term.
- Compute each local D_j and apply X_j @ D_j. Retain identity or diagonal transforms without dense matrices.
- Transform each local penalty as D_j.T @ S_j @ D_j. For q-level random intercepts, store a scaled identity of size q symbolically, not a q-by-q identity.
- Derive total penalty range/nullity from block decompositions. Preserve the existing global cutoff where the current implementation applies a cutoff relative to the largest eigenvalue of the assembled normalized total penalty. Independent per-block cutoffs are not automatically equivalent.
- Check unpenalized confounding on X times the combined unpenalized directions. A block-diagonal penalty does not imply that the unpenalized design is identifiable.
- Reuse the existing block log-pseudodeterminant formulas: singleton terms have rank*rho plus a constant; multi-penalty terms need a local range-space determinant. Do not sum individual pseudodeterminants for overlapping penalties.
- Add operations apply(beta,rho), quadratic(beta,rho), add_to_dense(G,rho), log_pdet(rho), and parameter_vjp(beta,adjoint,rho).
- Keep NumPy descriptors in penalties/. Transfer arrays and register JAX pytrees in fitting/. Smoothing parameters are dynamic leaves; slices and representations are static metadata. Preserve the family-parameter cache safeguards discussed in PR 4.

The direct solver may still allocate one dense H and its factor. Eliminate m separate global penalties first. Functional JAX updates do not guarantee in-place execution: measure compiled workspace and apply buffer donation where ownership is clear.

For covariance transformation, preserve off-diagonal blocks: V_public[j,l] = D_j @ V_fit[j,l] @ D_l.T. Transforming only diagonal covariance blocks changes uncertainty.

Expected storage: O(sum local penalty sizes), with diagonal/identity special cases O(p) or smaller. Design reparameterization changes from O(n*p²) to O(n*sum(k_j²)); local dense penalty transformations cost the sum of their local cubic costs.

Gate: penalty action/quadratic form/log-pseudodeterminant and derivatives match the old representation and corresponding R `Sl.*` operations; prediction and covariance roundtrips pass; lambda ordering stays unchanged; tensors, factor-by, shrinkage, duplicate labels, and parametric-only models are included. Extend `tests/test_penalties.py`, `tests/test_fitting/test_fitting_data.py`, `tests/test_linalg.py`, and the prediction/result tests, with dedicated files for the new modules.

Storage regression to measure: compare `sum(S.nbytes)` before embedding, after embedding, and after device transfer for many additive cubic smooths. Report peak live memory as well as retained penalty bytes, including `D_global`, total-penalty eigenspaces, and factor-by wrapper penalties.

**PR 3 — eliminate repeated basis work and initialization.**

Affected: smooths/base.py, cubic.py, tprs.py, gaussian_process.py, tensor.py, by_variable.py, **smooths/utils.py** (`_compute_distance_matrix`), formula/design.py, fitting/data.py, fitting/initialization.py, fitting/newton.py, api.py.

Read first: [`R/smooth.r::smoothCon`][r-smoothcon] and the smooth constructors, which return the training matrix in `sm$X`; [`PredictMat`][r-predictmat] is the separate evaluation path for new data. For initial smoothing statistics use [`R/mgcv.r::initial.sp`][r-initial-sp]: its `XX=TRUE` branch already consumes a cross-product diagonal, and penalties stay local with `off` indices. Preserve JaxGAM's current activity-threshold calculation during the storage migration: row/column means over a globally padded penalty have a different divisor from means over a local block.

- Give setup orchestration an explicit way to consume the just-built training basis. Do not use equal row count or mutable object identity as a public data-cache key. build_design_matrix(newdata) must continue to evaluate the supplied data correctly.
- Reuse marginal bases in tensor setup; compose wrappers from known training blocks.
- Remove the nearest-knot index calculation after TPRS subsampling when its result cannot be used by the subsequent branch.
- Replace broadcast distances with bounded cdist/kernel/projection batches. Never retain an n-by-r-by-d difference tensor. In the dense backend write projected rows directly into the final n-by-k buffer.
- Reduce unnecessary copies in constraint application using explicit ownership; preserve prediction-state immutability.
- Compute initial-sp diagonal statistics by bounded reductions instead of constructing wX and wX*wX over the full design.
- Compute and cache beta initialization once before device transfer. Reuse it for initial scale estimation and the first score call.
- Keep family-specific validity checks and offset handling. A sampled warm start is an optional optimization, not a replacement for validity checking.

The concrete initialization duplication is in `NewtonOptimizer.run`: the unknown-scale branch and first score call both call `_initial_beta`, which converts device X/y/weights back to NumPy. `initialize_beta` currently returns a JAX array. Split CPU starting-value computation from transfer at the existing Phase 1→2 boundary, after the fitting transform is known, then pass that one result through. Keep the family-specific null-start fallback and the distinct initial PIRLS fit used to estimate scale.

TPRS `build_design_matrix` currently checks equal row count but then rebuilds the kernel/polynomial projection; despite its cache-related comments, it does not return cached `_X` or compare data values in that branch. Cubic/GP/tensor paths also evaluate again through prediction methods. Preserve their behavior while introducing an explicit handoff of the just-built training matrix.

Gate: training basis construction is called once per required normalization stage; changing same-length data produces new predictions; data/offset/theta changes invalidate the appropriate caches; beta initialization runs once; existing numerical tolerances remain unchanged. Extend the smooth tests, `tests/test_smooths/test_prediction_copy.py`, and the initialization/Newton tests.

**PR 4 — replace the negative-binomial derivative scan with prefix sums.**

Affected: [families/negative_binomial.py, _lgamma_diff_jvp](https://github.com/scatt67/jaxgam/blob/c1a31346188186bbd2fe67eea8d0d51ba6c95d79/jaxgam/families/negative_binomial.py#L365), fitting/data.py, and the max_y/static-argument plumbing in fitting/newton.py and fitting/reml.py. Add tests/test_nb_lgamma_derivatives.py. Extend tests/test_fitting/test_nb_custom_jvp.py for fit-level integration and the scaling benchmark for this kernel.

Read first: [`R/efam.r::nb`][r-nb], specifically `Dd`, `ls`, `initialize`, `getTheta`, and `putTheta`. `nb()$ls` already supplies vectorized digamma/trigamma formulas for the saturated likelihood and its first/second derivatives in **log(theta)**, with per-observation weights. R does not implement a max-count scan or a count-prefix planner. Reuse those formulas as the reference on well-conditioned cases; the prefix optimization below is specific to JaxGAM's stable recurrence JVP near the Poisson limit. Benchmark the R-style vectorized formulas too, without replacing the stable recurrence with cancellation-prone subtraction at large theta.

Dependency: none of the new backends. Merge as an independent Stage 1 optimization, before estimated-theta NB is enabled in the streamed REML path. Its scope is the gamma-difference derivative and count metadata; retain the current likelihood formula, dispersion parameterization, and PIRLS algorithm.

The existing JVP scans k from zero to max(y)-1, updating an n-element accumulator on every step. For scalar theta and integer counts, all observations share the same reciprocal sequence:

~~~text
f(theta, y) = lgamma(theta) - lgamma(theta + y)
partial_theta f = -sum(k=0..y-1, 1/(theta+k))
partial_theta_theta f = sum(k=0..y-1, 1/(theta+k)²)
~~~

Build that sequence once per current theta and gather its prefixes:

~~~python
# JVP sketch for validated integer counts and a bounded static capacity.
# capacity must cover every lookup, including any y_safe sentinel.
reciprocal = 1.0 / (theta + jnp.arange(capacity, dtype=jnp.float64))
prefix = jnp.concatenate((
    jnp.zeros(1, dtype=jnp.float64),
    jnp.cumsum(reciprocal),
))
dtheta_value = -prefix[count_index]
tangent = dtheta_value * theta_dot - jsp.digamma(theta + y) * y_dot
~~~

Keep the primal value path unchanged in this PR. Preserve the existing y-tangent term where supported; fitting normally holds integer responses constant. Do not silently truncate fractional responses to indices. Validate the integer fast-path domain and explicitly preserve or define behavior outside it.

Differentiate the prefix calculation through theta so the Hessian receives the positive reciprocal-square sum. Test the actual optimization coordinate too: for t=log(theta), dF/dt=theta*dF/dtheta and d²F/dt²=theta*dF/dtheta+theta²*d²F/dtheta². Passing only the first-derivative test is insufficient for the current outer optimizer.

Implementation steps:

1. Introduce a reusable count plan with integer indices, maximum required lookup, optional unique-count mapping, and bounded table capacity. Prepare indices once per fit; recompute theta-dependent table values whenever theta changes.
2. Normalize max_y to a neutral value for families that do not use it; `FittingData.from_setup` currently computes it for all responses, and `_DIFF_STATIC` includes it. Capacity buckets and unique-count plans are optional follow-ups if profiling justifies them. Estimated log_theta is already dynamic. Fixed theta currently enters the family's static cache key in `NegativeBinomial._static_cache_key`; preserve that safeguard unless all fixed-theta reads are explicitly migrated to dynamic arguments and tested across different theta values.
3. Handle empty helper inputs, zero counts, all-zero responses, and _saturated_loglik_jax's y_safe=1 convention without an out-of-range gather. JAX's indexing behavior must not silently hide an undersized table.
4. Enforce a prefix-workspace budget before allocating. Above it, reuse R-style vectorized derivatives only in a validated conditioning regime, otherwise use a recurrence fallback whose compiled first/second-derivative workspace fits the budget. The current scan passes `jnp.arange(max_y)` as its input: do not assume it is a memory-safe fallback without inspecting the compiled allocation. A counted recurrence retaining the existing n-element updates is acceptable initially if its workspace gate passes; a faster tiled requested-prefix implementation can follow if profiling warrants it. Avoid requiring a new sorting/count-planner subsystem for the ordinary fast path.
5. State the large-count limit: an n-accumulator recurrence still performs O(n*max(y)) work; a tiled prefix recurrence still performs work proportional to the maximum count. A validated asymptotic tail can be subsequent work; raw subtraction of nearly equal digamma/trigamma values is not an acceptable untested replacement near the Poisson limit.
6. In the streamed backend, share a theta-specific table or requested-count values across row batches. Rebuilding the full table for each batch multiplies its cost by the number of batches. Only count metadata is reusable across different theta values.

For the ordinary bounded-table path, work changes from O(n*max(y)) to O(n+capacity), with O(n+capacity) total indices/output/table storage and O(capacity) table workspace. Set capacity to the required maximum, or a bounded bucket if implemented, allowing for the all-zero sentinel case. Any unique-count sorting is one-time preprocessing and must be timed separately.

Validation gates:

- Compare first and second theta derivatives against recurrence sums on tractable counts, including y=0, repeated counts, capacity boundaries, theta well below one, and theta up to 1e12. Use a high-accuracy reference for extreme-scale cases rather than finite differences of cancellation-prone gamma values alone.
- Verify derivatives in log(theta), JVP linearity, weighted saturated-likelihood sums, mixed REML derivatives with smoothing parameters, and the bounded-memory fallback.
- Preserve fixed- and estimated-theta behavior, fitted theta, smoothing parameters, predictions, deviance, and convergence within existing tolerances on supported fixtures. Noninteger handling must be explicit and tested.
- Benchmark cold compilation and synchronized warm JVP/Hessian execution, varying n and max(y) separately and including one large-count outlier. Measure temporary memory and compile-cache reuse. Run end-to-end estimated-theta NB fits separately.

Neither current JaxGAM initialization nor R's `nb()$initialize` rejects fractional nonnegative y. The integer recurrence identity does not apply there, so rounding/casting is not a compatible fix. Add a validated noninteger derivative path (with the same gamma-function meaning as `nb()$ls`) or explicitly scope and document any input-policy change separately. Extreme-theta checks need a high-accuracy oracle because the R polygamma difference can also lose precision. No kernel or end-to-end speedup is established by this document audit.

**4. Stage 2: streamed setup, prediction, PIRLS, and REML**

**PR 5 — introduce replayable sources and observation-independent setup.**

Add data/source.py, formula/prepare.py, and formula/design_provider.py. Extend smooths with metadata discovery, reduction, finalize, and evaluate_batch operations. Keep ModelSetup as the dense compatibility wrapper.

Read first: [`R/bam.r::mini.mf`][r-mini-mf] and [`bam`'s setup dispatch][r-bam-setup], then [`R/smooth.r::smoothCon`][r-smoothcon], [`PredictMat`][r-predictmat], and each supported constructor. mgcv already separates basis setup from repeated block evaluation. However, `mini.mf` constructs a sampled frame with covariate extremes and factor-level coverage; normal `bam` setup uses that frame. Copying it would change knots, centering, and normalization relative to JaxGAM's full-data `ModelSetup.build`. Use it only for an explicitly sampled-basis policy. Exact full-data reduction passes below are additional work, not an existing `bam` guarantee.

Build the preparation result around the existing `PredictSpec`, `CoefficientMap`, smooth prediction-copy hooks, and parametric alias metadata. Do not build dense `ModelSetup` and discard X afterward: that only reduces retention. `RowSource` is new I/O infrastructure; mgcv's `bam` still constructs a model frame and keeps observation vectors, so its memory behavior is not an out-of-core raw-input contract.

Implement sources in this order: arrays/DataFrame, NumPy memmap, then an optional columnar file adapter. A source must be restartable because PIRLS and line searches revisit data. Reject a one-shot iterator before fitting unless the user explicitly selected a spool-to-disk policy. A source fingerprint covers schema, data version, row selection, weight/offset definitions, and basis configuration.

Preparation needs multiple passes, not a universal two-pass promise:

| Pass | Work and retained state |
|---|---|
| Discovery | Row counts; response/domain checks; means/ranges; factor dictionaries; unique-value/knot information; conservative p and memory estimate |
| Basis construction | Freeze knots and local kernel eigenspaces; construct raw local penalties |
| Normalization reductions | Column sums and sum of squares; finalize RMS scales where needed |
| Dependent reductions | Re-evaluate normalized rows to compute infinity norms and any downstream centering/tensor statistics |
| Constraints | Freeze centering, side-constraint deletions, and unpenalized rank decisions |
| Fitting preparation | Freeze local D transforms; compute initial-sp/initial-beta information; expose the batch evaluator |

Preserve the existing order of shrinkage, basis scaling, penalty normalization, constraints, and fitting reparameterization. In particular:

- Cubic knots are interpolated through sorted unique covariate values, not quantiles of the full empirical row distribution. An ordinary quantile sketch changes the basis when values repeat.
- Exact unique-value or multi-column unique-row processing can require external sorting. It cannot, for arbitrary inputs, be replaced by a fixed-size in-memory sample while claiming exact equivalence.
- For TPRS, RMS scaling changes row absolute sums. The smoothCon infinity norm after scaling generally needs another pass.
- Sum-to-zero constraints are global column sums, not independent per-batch centering.
- Factor-by constraints use the base smooth across all observations, before level indicators, as the existing implementation specifies.
- te/ti preparation follows dependencies among marginal normalization, marginal constraints, tensor construction, and final term constraints.

Concrete basis references: [`smooth.construct.cr.smooth.spec`][r-cr] calls `quantile(unique(x), ...)`; [`place.knots`][r-knots] expresses the equivalent rank interpolation used by JaxGAM's `_place_knots`. [`smooth.construct.tp.smooth.spec`][r-tp] caps unique locations at 2000 by default using R's seeded `sample`, then calls [`src/tprs.c::construct_tprs`/`tprs_setup`][r-tprs-c]. The C implementation truncates E's eigensystem and handles the polynomial null space; do not replace it with an assumed augmented-matrix eigensolve or max-min knot algorithm from stale prose. [`smooth.construct.gp.smooth.spec`][r-gp] has its own knot/kernel setup. Keep the existing Python evaluators and normalization order as the dense reference.

Known parity limitation: `smooths/utils.py::_subsample_knots` uses NumPy `RandomState`, and its docstring explicitly states that selected knots differ from R's sampler. Preserve that selection for dense-versus-streamed tests. Strict/MODERATE R basis parity above the cap requires matching the actual selected knots/R sampling first; do not claim that this plan has fixed it. `GAM(knots=...)` is currently guarded as unsupported. Initial exact-streaming delivery excludes those sampled TPRS/GP cases.

Preserve schema semantics from `formula/predict_matrix.py::_to_dict` and the shared factor helpers: integer-valued categoricals must remain factors, ordered-factor reference levels must remain stable across batches, and row positions must be positional rather than pandas-label aligned. Preserve finite-input validation, nonnegative prior weights, positive total weight mass, and parametric alias decisions from `ModelSetup.build`.

Start exact streaming with univariate cubic and modest parametric blocks. Add TPRS/GP and complex overlapping terms only when their preparation passes meet the same contract. Unsupported combinations must fail at preflight; they must not silently materialize a full design.

For streamed side constraints, adapt [`qr_update`][r-qr-update] to compress each overlapping term group, then apply `fixDependence`/`gam.side` to the small system. Compute `augment.smX`'s scaling statistics on the original rows before compression: mean absolute entries are not preserved by QR. Append the correctly scaled penalty roots in their original block coordinates. Preserve unweighted centering, ordering, pivots, thresholds, and deletion decisions. This compressed use of `gam.side` is a new adaptation requiring tests, especially at rank boundaries; mgcv's `bam` does not provide exact full-data side constraints through `mini.mf`.

Gate: prepared prediction state retains no observation-aligned training arrays; knots/scales/constraints agree with dense setup; prediction is invariant to batch boundaries; no full design allocation during preparation. Basis metadata can grow with knots or factor levels and must be budgeted separately; an accidental equality between k/q and n is not proof of retained training data. Reuse the object-graph checks in `tests/test_result_mode.py` and `tests/test_smooths/test_prediction_copy.py`, with new source/preparation tests and extensions to `tests/test_formula/test_design.py`.

**PR 6 — stream prediction and separate optional uncertainty.**

Affected: formula/predict_matrix.py, inference/_core.py, inference/predictor.py, results.py. Add a result mode for prediction plus basic convergence diagnostics that does not require retaining Vp, EDF/EDF1 diagnostics, or training rows.

Read first: [`R/mgcv.r::predict.gam`][r-predict-gam] already divides ordinary predictions into blocks; [`R/bam.r::predict.bam`][r-predict-bam] delegates to that implementation, and [`Sl.postproc`][r-sl-postproc] separates optional covariance postprocessing. Reuse JaxGAM's `PredictSpec.build_predict_matrix`, `predict_core`, and `finish_prediction` for the initial batch adapter. Per-term matvec and factor-based covariance can follow without reimplementing categorical encoding, coefficient transforms, or response-link SE conversion.

- predict_iter yields ordered batches; a convenience predict may concatenate only when explicitly requested and within its output budget.
- For point predictions, evaluate and accumulate terms directly into eta. No full B-by-p matrix is required if the term evaluators expose matvec operations.
- For exact standard errors at manageable p, retain the Fisher factor L and fitting transforms instead of forming H inverse.

~~~python
X_fit_batch = X_public_batch @ D       # implemented through local transforms
Z = solve_triangular(L, X_fit_batch.T, lower=True)
variance = phi * np.sum(Z * Z, axis=0)
~~~

The solve is still O(B*p²) for dense factors. This avoids explicit inverse/covariance construction; it does not make exact arbitrary prediction variances linear in p.

Preserve existing full and inference result APIs through adapters, including their dense Vp contract and shared operation order; full-versus-inference predictions are currently tested for byte equality in `tests/test_result_mode.py`. For the new mode, branch before `_from_fit` constructs H inverse, F, Vp, EDF1, or transfers row outputs. `PIRLSResult` currently always holds mu/eta/weights and both information matrices/factors, so fit-state changes are also required to avoid retaining those rows. An explicit Vp request on the new mode can materialize covariance subject to a memory check. A prediction-only result without a covariance provider must raise a clear unavailable-uncertainty error for se_fit=True. Factor-based SEs must use `L_fisher` and the reported scale convention currently used by `_from_fit`.

Per-term EDF/EDF1 and optional diagnostic reporting can be deferred, but the total Fisher EDF required by `fitting/reml.py::fletcher_scale`/`estimate_scale` cannot be skipped when preserving unknown-scale families' reported scale. Separate computing that scalar from retaining diagnostics; a different or approximate scale convention is not an exact storage refactor. Use selected/block right-hand-side solves and correct cross-block terms for requested diagnostics; do not approximate full EDF1 with diagonal term blocks. A full result's training X cannot be supplied for free by an out-of-core fit.

Gate: batch predictions and standard errors match dense outputs, including external-offset omission warnings, supplied offsets, categorical levels, transformed coefficients, empty/tail batches, and input row ordering. Keep `tests/test_predict/test_predict.py`, `tests/test_inference/test_predictor.py`, and `tests/test_result_mode.py` as compatibility gates. New modes may compare numerically to old modes; do not weaken the existing full-versus-inference byte-equality assertion.

**PR 7 — initial streamed PIRLS for fixed smoothing parameters (published foundation).**

Published scope: Gaussian/identity, Poisson/log and binomial/logit; replayable prepared cubic/numeric designs; fixed sp and compact prediction results; accumulated cross-products with an SPD Cholesky solve. The general QR/TSQR, non-canonical and NB requirements in the original milestone are carried into PRs 7.1–7.6 below. Publishing PR7 does not mark those requirements complete.

Add execution/stream.py for host iteration and I/O, fitting/stream_kernels.py for pure JIT batch reductions, and fitting/state.py for observation-independent fit state.

Read first: [`R/bam.r::qr_up`/`qr_update`][r-qr-update] and [`bgam.fit`][r-bgam-fit] for chunkwise working-response/weight reduction and QR reuse; [`R/gam.fit3.r::gam.fit3`][r-gam-fit3] for penalized-deviance backtracking, validity, and the stationarity check before accepting convergence. For later NB/non-canonical support read [`R/gam.fit4.r::dDeta`/`gam.fit4`][r-gam-fit4] and the observed/Fisher split in `src/gdi.c`. `bgam.fit` updates smoothing on each working model; reuse its reductions without importing that smoothing strategy into this fixed-sp path or PR 8.

Do not put a Python file iterator inside a JAX while_loop. The dense backend retains its fused loop; the streamed backend uses host orchestration around compiled kernels. Share the working-quantity and accepted-step calculations in `fitting/pirls.py` rather than maintaining separate family mathematics. Its NB observed-weight branch, signed cross-products, and Fisher postprocessing must remain distinct.

Per coefficient iteration:

1. Scan batches at the current beta. Evaluate eta, mu, W, and z using the existing family/link conventions; accumulate G, b, deviance, and required scalar statistics.
2. Add the structured penalty to G and solve the small coefficient system.
3. Evaluate the proposed coefficient step against global penalized deviance and domain validity. Add the penalty once per candidate, not once per batch.
4. If needed, backtrack. Optionally evaluate a small group of candidate step sizes in a single source scan using eta_current and delta_eta, retaining only batch-sized candidate arrays.
5. Check the accepted coefficient/deviance criteria plus a scaled stationarity residual. A failed line search with an unchanged beta must not be mistaken for convergence.
6. Recompute final statistics at the accepted beta, including distinct observed and Fisher curvature where required.

~~~python
# Schematic canonical positive-weight reduction.
# Invalid padding inputs must be replaced by finite, family-valid values first.
@jax.jit
def accumulate(state, X, y, prior_w, offset, valid, beta, family_params):
    eta = X @ beta + offset
    w, z = working_quantities(y, eta, prior_w, offset, family_params)
    w = jnp.where(valid, clip_working_weights(w), 0.0)
    z = jnp.where(valid, z, 0.0)
    WX = jnp.sqrt(w)[:, None] * X
    return state.add(
        G=WX.T @ WX,
        b=WX.T @ (jnp.sqrt(w) * z),
        deviance=masked_deviance(y, eta, prior_w, valid, family_params),
        valid_domain=masked_domain_check(eta, valid, family_params),
    )
~~~

Mask after weight clipping: otherwise padded zero-weight rows get promoted to the current positive floor. Avoid 0*NaN by sanitizing padding before link/deviance evaluation. Preserve real-row weight behavior separately.

Memory and transfer policy:

- Keep p-by-p accumulators on the selected compute device across batches. Copying one G back to the CPU per batch adds O(number_of_batches*p²) transfer traffic.
- Bound host prefetch and outstanding device work, initially to two batches. A long asynchronous launch queue can retain every input buffer and defeat streaming.
- Fit and score reductions use float64. Use stable/pairwise accumulation or compensation for long reductions as justified by the error budget.
- Estimate memory using all live design, weighted design, kernel, factor, adjoint, and allocator workspaces. O(B*p+p²) is an asymptotic invariant, not a claim that only two arrays exist.
- PR7.2 supplies QR/TSQR coefficient solves for ill-conditioned positive-weight designs. PR7.4 supplies the separate sign-aware path for signed observed weights; sqrt(W), ordinary Cholesky, and PCG are not universal fallbacks.
- PR7 is the initial canonical Poisson/binomial/Gaussian fixed-sp release. PR7.3 adds regular-family/link coverage, PR7.5 adds NB coefficient kernels and parameter state, and PR7.6 completes the public all-family gate, including estimated theta through PR8.2.

Gate: fixed-sp results agree with dense fits within STRICT/MODERATE tolerances across B; offsets, zero weights, rare factor levels, extreme means, rank deficiencies, and backtracking are tested; measured additional RAM is bounded as n increases with B and p fixed. Extend `tests/test_fitting/test_pirls.py` for the shared numerics and add a streamed-kernel JIT test plus host-source integration tests. RBridge comparisons for this PR hold sp and the basis fixed.

**Required new PR7 follow-ups — shared engine, robust solvers, all existing families.**

These are new implementation PRs after the published foundation, not optional polish. They may be developed alongside the existing PR8/PR9 work, but their dependency and acceptance gates remain explicit.

**PR 7.1 — shared family execution contract and conformance suite.**

Extend/combine `families/base.py`, `families/extended.py` and shared PIRLS kernels rather than create a second streaming family hierarchy. Read the corresponding `gam.fit3`, `gam.fit4`/`dDeta`, family and link implementations before extracting numerical behavior. Keep CPU preparation descriptors free of JAX imports; device kernels consume explicit dynamic family parameters.

- Specify batch working quantities, direct deviance/likelihood contributions, domain-valid padding/initialization, observed and Fisher information, and bounded mergeable summaries for initialization, scale, likelihood and count-prefix metadata. Use a finalization step for global decisions; per-batch initialization/fallbacks must not change the statistical algorithm.
- Separate static family/link configuration and mathematical capabilities from dynamic theta/phi and from validation/release status. Capabilities describe the required coefficient system, derivative availability and reduction semantics, not a hard-coded family name.
- Preserve static-cache and source/basis/family lineage contracts, with immutable accepted versus trial parameter state. Dense defaults and their operation order stay unchanged during extraction. Do not put Python source iteration inside JAX transformations.
- Create one checked-in family/link/parameter-mode inventory. Required first rows include Gaussian/identity plus log/inverse, Poisson/log plus identity/sqrt, binomial/logit plus probit/cloglog, Gamma/inverse plus log/identity, and NB/log/identity/sqrt in both theta modes. Inventory all other currently supported combinations as well; this minimum list is not a way to drop them. Distinguish unsupported input domains from unfinished family implementations.

Gate: existing dense and PR7 numerical/cache tests are unchanged; family contract tests cover batch-size invariance, padding/domain behavior and global reductions. An unregistered test-only row-separable family or independently implemented family adapter must work without editing the stream driver or spoofing a built-in `family_name`. Missing capabilities reject explicitly. No new production distribution is required for this extensibility test.

**PR 7.2 — general positive-weight QR/TSQR coefficient solver.**

Depend on PR7.1 and reuse reviewed QR primitives from PR9 where applicable, without requiring its Gaussian-only optimizer. Read `bam.r::qr_up`/`qr_update`, `bgam.fit`, and mgcv's penalized least-squares/rank handling. Update a bounded R/transformed-response state from weighted design batches, then incorporate local penalty roots and solve the reduced penalized problem. Preserve pivots, original row counts, the residual tail, fitting transforms and the fixed identifiable coefficient subspace.

Expose an explicit solver policy and truthful route/failure diagnostics; retain the reviewed Cholesky fast path for suitable systems. QR must not first form normal equations and merely factor those: its purpose is to avoid their squared conditioning. Do not construct n-by-n Q, retain n-by-p rows, or stack m global penalties. Rank deficiency needs an explicit identifiable-subspace decision consistent with CoefficientMap, not jitter disguised as rank handling.

Gate: fixed-sp dense/pinned-R parity for all currently enabled positive-weight families; ill-conditioned and aliased designs; B=1, B<p, empty/tail batches, long reductions, weights/offsets and backtracking. Include cases where cross-product Cholesky loses accuracy but QR remains accurate. Verify JIT-compatible solve kernels, bounded CPU QR helpers, source-scan counts and increasing-n memory. Signed weights are owned by PR7.4, not silently passed through sqrt(W).

**PR 7.3 — all regular families and their supported links.**

Depend on PR7.1–7.2; use PR7.4 wherever a signed observed system is needed. Enable Gaussian, Poisson, binomial and Gamma through the shared contract, including the non-default links in the inventory. Match pinned `gam.fit3`'s working-system policy: canonical links use Fisher scoring, while noncanonical links use alpha-adjusted Newton working weights/responses, including its explicit recovery behavior. Do not extend the existing dense Fisher implementation to every streamed regular link and call that the same mgcv algorithm. Keep observed information for REML separate from Fisher EDF/covariance. Add bounded family-specific initialization and reported-scale reductions, especially Gamma/Fletcher scale; Gaussian's closed-form score-scale shortcut is not a generic scale estimator. Default `start=NULL` initialization must preserve per-row `eta=link(mustart)` and its distinct null-coefficient backoff anchor; a QR projection of that eta is not an interchangeable first working state.

Gate: every inventoried regular-family/link cell passes fixed-sp coefficient, fitted-mean, deviance, reported-scale, total-EDF and requested-uncertainty parity at STRICT/MODERATE on the same prepared basis. Exercise non-canonical links, valid null starts, near-domain boundaries, zero/extreme weights, offsets, exact-zero residuals and failed-step status. Do not substitute a canonical link or silently change observed to Fisher information. Joint smoothing/phi optimization is a separate PR8.1 gate.

September 8 initialization review adds explicit boundary gates: preserve R's literal `alpha == 0` replacement without a near-zero deadband that can reverse signed curvature. The internal PR7.3b prototype exposes a separate Binomial/log cancellation-resolution diagnostic for exact and near-success responses; resolving and validating that discrepancy is still required before the complete family/link release gate. It is not a permanent exclusion or evidence that all initial working systems already match. Normalize zero-weight Binomial responses consistently before CPU/JIT input validation; distinguish invalid positive-weight working inputs from legitimate zero-information rows. Batch admissibility must compose across empty/zero-weight batches, with an informative-count requirement applied globally. Raw Fisher/observed diagnostics must not silently replace the selected coefficient system or cause an unrelated selected-system failure.

**PR 7.4 — sign-aware observed-curvature fitting and scoring.**

Depend on PR7.1 and the coefficient solver interface from PR7.2. Read `gam.fit4`, `dDeta` and the corresponding signed least-squares C routines. Add bounded signed cross-products and a validated solve/safeguard policy for indefinite intermediate working systems; preserve the pinned algorithm's recovery and global deviance backtracking. Keep separate observed and Fisher statistics/factors and make their consumers explicit. A positive Fisher matrix is not evidence that an observed matrix can use Cholesky/PCG.

Gate: matched-state signed-system tests plus real non-canonical/extended-family cases cover mixed-sign working weights, indefinite intermediates, recovery, genuine failure and consistent final observed/Fisher statistics. No absolute-weight substitution or new Fisher-scoring algorithm may be introduced without a separately identified numerical-policy change. Coefficient solves and score/log-determinant admissibility are separate checks.

**PR 7.5 — NB fixed-theta fitting and dynamic family-parameter state.**

Depend on PR7.1, PR7.4 and PR4's count metadata. Enable NB's log, identity and sqrt links at explicitly supplied theta; expose the same pure coefficient kernels at a trial theta for estimated-family callers. Preserve the appropriate existing default versus optimizer-specific observed/Fisher conventions, count-prefix budget, stable likelihood primitives and source-owned weights/offsets. No full-response/count copy may be hidden in preparation. Fixed-theta cache isolation and dynamic-theta updates must both be tested.

Gate: fixed-theta NB parity across links, small/large theta, count tails, multiple batches and signed-curvature cases; identical results across repeated and fresh-process fits with different theta values. Any fractional-response discrepancy in the pinned oracle must be resolved or recorded as an explicit, reviewed compatibility issue, not hidden with relative-score comparisons. This PR does not freeze theta when the user requested estimation: public estimated-theta fitting awaits PR8.2. EFS's conditional in-PIRLS theta policy remains owned by EFS-5, not substituted for exact REML.

**PR 7.6 — public all-family fixed-sp integration and release matrix.**

Depend on PR7.1–7.5 and PR8.2 for estimated NB with lambda pinned. Wire family capabilities and solver selection into the public streamed route, compact result/family snapshots, prediction/SE and diagnostics. Separate basis eligibility, family/link capability, parameter-estimation mode, solver admissibility and derivative eligibility; do not collapse them into a family-name allowlist. Explicit streamed requests must never fall back to materializing full training X without separate user selection.

Gate: all five families and every supported inventoried link, both fixed and estimated NB theta, pass the public fixed-sp matrix with pinned R and dense comparisons at STRICT/MODERATE. Test reported/score scale conventions, ownership/pickle, family isolation, bounded retained state and near-boundary failures. Demonstrate the custom-family extension seam from PR7.1 at the public driver. New families added later must provide capabilities and conformance tests; they must not require rewriting host scanning. Mark complete only when missing family cells are implemented or the user explicitly approves a changed scope. Unsupported prepared smooths remain a separately documented limitation.

**PR 8 — exact-objective streamed REML foundation using an adjoint gradient.**

The current internal provider/L-BFGS implementation is a canonical Poisson/log and binomial/logit, known-scale, differentiable-interior foundation. The full algorithm below is the target; PR8 alone does not deliver the remaining regular-family scale/non-canonical or NB theta paths. PRs 8.1–8.3 make those follow-ups explicit.

Affected: `fitting/newton.py::_diff_score`, `_jit_diff_grad_hess`, `fitting/reml.py::_criterion_core`, and the new stream state/reductions; add a pure coefficient/statistic derivative kernel and an `execution/reml.py` host driver. The current module-level score accepts full X/y/weights as explicit dynamic arguments; the nested custom_jvp closes over those arguments, and Newton evaluates a full outer Hessian. The existing cache design already avoids a separate compiled score closure per fit. Passing only G and b to REML and treating them as constants loses how beta and W vary with smoothing parameters.

Read first: [`gam.fit3`][r-gam-fit3]/[`gam.fit4`][r-gam-fit4] and [`src/gdi.c::gdi1`/`gdi2`][r-gdi] for score and implicit derivatives; [`R/gam.fit3.r::newton`][r-newton] for inner tolerance tightening and accepted-fit warm starts; [`bfgs`][r-bfgs] for an existing first-derivative optimizer on the same fitted-model criterion. mgcv's BFGS uses a full inverse-Hessian approximation. Use an existing bounded L-BFGS implementation, such as SciPy's L-BFGS-B in the host driver, to bound m-space storage rather than write a new quasi-Newton algorithm. The streamed VJP/adjoint interface itself is additional work and must be validated against the current custom_jvp.

Use the same scalar REML criterion with a new explicit derivative provider. Let:

~~~text
rho = log smoothing parameters
g(beta, rho, theta) = 0.5 * gradient_beta(deviance) + S(rho) beta
H_beta = derivative_beta(g)
F(beta, rho, theta, phi) = the existing REML scalar criterion
~~~

At a converged coefficient solution:

~~~text
solve H_beta.T a = partial_beta F
total_gradient_rho F =
    partial_rho F - (partial_rho g).T a

For rho_j: partial_rho_j g = exp(rho_j) S_j beta
~~~

Implementation:

1. Converge streamed PIRLS at the trial rho/theta. Cache accepted beta, final G_observed, G_fisher if needed, deviance, family statistics, factors, and the exact parameter values.
2. Differentiate the p-dimensional REML core with respect to its explicit inputs. This produces cotangents for G_observed, deviance, beta, rho, and relevant family scalars.
3. Scan rows again. For each batch, apply a JAX VJP of the batch statistics with these cotangents and sum beta/theta contributions. No reverse-mode tape spans the source iterator.
4. Solve one coefficient-space adjoint system.
5. Contract the adjoint with local penalty derivative actions. This is local block work for all smoothing parameters, rather than materializing m global penalty derivatives.
6. Add theta and scale derivatives with their actual dependence. For NB, include both the saturated-likelihood theta derivative and partial_theta g. Do not freeze estimated dispersion.
7. Use L-BFGS-B on rho and joint parameters. Preserve the current lambda bounds from `NewtonOptimizer._clamp_params` (default rho in [-40, 40]); log_theta/log_phi are currently unbounded. Each trial score must re-converge beta and use the global objective; warm-start from the last accepted fit. Retain the actual parameter ordering `[log_lambda, <log_theta>, <log_phi>]`, `Mp`/rank corrections, and the distinction between optimized phi and reported Fletcher scale.

L-BFGS changes the optimization trajectory, not the intended exact REML objective. Keep dense Newton as the reference and optionally for small m. Do not produce a full m-by-m Hessian by default. Finite-difference refitting for each rho_j would add roughly m full fits per gradient and is only appropriate as a small validation oracle.

Preserve the existing solve/score regularization conventions. The current log determinant uses a scaled system and jitter, whereas the IFT solve uses observed curvature with its own stabilization. Sharing factors is valid only when those systems coincide. A genuinely indefinite or unidentified system needs a stated algorithmic decision, not an unconditional SPD solve.

The adjoint equations assume a converged, differentiable stationarity equation with a fixed identifiable subspace. Check active weight clipping separately: current PIRLS clips weights, while the canonical `_diff_score` JVP differentiates the underlying working-weight function. Agreement among that custom JVP, finite differences, and a new stationarity derivative is not automatic at clipping/rank boundaries. Start derivative parity in the smooth interior and resolve discrepancies explicitly before expanding the streamed path to those cases.

Complexity remains dominated by batch G formation and its VJP, generally O(n*p²), plus coefficient factorizations. The advantage is bounded row workspace and removal of full Hessian/m-direction intermediates. Exact streaming is a capacity improvement; repeated passes can still be expensive.

Gate: gradients agree with the existing custom_jvp and finite differences on manageable data; scalar objective, fitted predictions, EDF when requested, and convergence residuals meet reference tolerances. Exercise multiple penalties, coupled tensor penalties, theta, scale, non-canonical links, and rank issues separately. Use `tests/test_fitting/test_reml.py`, `test_newton.py`, and `test_nb_custom_jvp.py` as layer references; consolidate broad final fits in `tests/test_validation_matrix.py`.

First proof required: a reproducible, well-conditioned Poisson case with multiple penalties, comparing the batch VJP/adjoint to `_diff_score` at several trial rho values and batch sizes. Current captured proof evidence belongs in the implementation checkpoint; the earlier draft's unverified quoted error is not acceptance evidence.

**PR 8.1 — regular-family adjoints, non-canonical links and unknown scale.**

Depend on the PR8 foundation and PR7.1–7.4. Generalize derivative contributions through the family contract rather than reusing the canonical identity `dD/deta = 2*wt*(mu-y)` for arbitrary links. Preserve observed stationarity/score curvature, Fisher reporting, actual saturated likelihood and joint log_phi derivatives for Gaussian/Gamma. Keep accepted/trial states bound to exact rho/phi and the prepared model. QR-stabilized coefficient solves require a consistent adjoint and log-determinant policy, not reuse of an unrelated factor.

Gate: all regular-family/link inventory cells have scalar, gradient, refit finite-difference, fitted-model and reported-scale parity. Clipping/rank/conditioning boundaries have explicit validated behavior; unsupported derivatives must not produce a plausible gradient. Include the high-rho penalty-null-space roundoff issue found during PR9 review. No full outer Hessian or reverse-mode tape over source scans.

**PR 8.2 — NB theta derivatives and exact parameter optimization.**

Depend on PR8.1, PR7.5 and PR4. Add dynamic-theta batch cotangents, saturated-likelihood derivatives and partial_theta stationarity terms for the supported NB links, preserving parameter order `[log_lambda, log_theta]` and phi=1. Optimize smoothing and theta jointly for exact REML; support fixed-sp calls by pinning lambda while leaving theta free. Reuse the common driver, not a host alternation that fully fits beta and then changes the statistical objective. EFS conditional theta Newton is a different policy and is not an exact-REML replacement.

Gate: fixed/estimated-theta NB derivatives, final fits and pin-lambda semantics meet dense/pinned-R comparisons, including non-default links, parameter isolation and post-theta consistency. Every accepted coefficient/score/gradient state belongs to the returned theta. Failure and boundary behavior stay explicit.

**PR 8.3 — public all-family exact streamed REML release.**

Depend on PR7.6 and PR8.1–8.2. Expose the reviewed limited-memory optimizer through the public streaming controls with truthful convergence, solver, objective, scale/theta and source-scan diagnostics. Preserve dense Newton defaults. The release matrix covers every inventoried family/link with fixed and estimated smoothing, both NB theta modes, prediction/uncertainty and memory behavior. Family-specific derivative limitations cannot be hidden behind a generic public "all families" claim. This is the all-family exact large-row milestone; PR9 remains its optional Gaussian acceleration, not a substitute for general-family support.

**5. Stage 3: Gaussian identity-link specialization**

**PR 9 — compress the data once and optimize in coefficient space.**

Add fitting/gaussian.py and execution/gaussian.py. Route only Gaussian with identity link, a frozen design and prior weights, and supported offset/penalty conventions. Other Gaussian links remain on general PIRLS.

Read first: [`R/bam.r::bam.fit`][r-bam-fit] already compresses weighted X and y-offset using `qr_update`, carries `R`, `f`, `y.norm2`, `rss.extra`, and `n.true`, then fits the reduced problem. [`R/fast-REML.r::fast.REML.fit`/`Sl.fit`][r-sl-fit] already optimize a Gaussian penalized least-squares REML criterion in coefficient space, with scale derivatives and penalty reparameterization. Adapt that compression flow and the existing JaxGAM `_criterion_core`; do not design a second Gaussian likelihood or copy the non-Gaussian working-model iteration. The Gaussian compression is exact for the frozen model even though the R routine is named fast REML.

A dense-input version can ship after PRs 2–3 using `FittingData` adapters; the RowSource version needs PR 5. Neither needs the new non-Gaussian adjoint in PR 8. Keep the current Newton optimizer initially, adding the bounded-memory optimizer only when m warrants it.

After basis preparation, one reduction computes:

~~~python
r = y_batch - offset_batch
G += X_batch.T @ (weight_batch[:, None] * X_batch)
h += X_batch.T @ (weight_batch * r)
c += np.dot(weight_batch * r, r)
~~~

Retain row/positive-weight counts, sum of log positive weights, and statistics required by null deviance and scale reporting.

For every smoothing trial:

~~~python
H = penalty_pack.add_to_dense(G, rho)
L = factor(H)
beta = solve_factor(L, h)
rss = c - 2 * beta @ h + beta @ G @ beta
score = gaussian_reml_core(beta, rss, G, L, rho, log_phi, metadata)
~~~

This expression is algebraic specification, not the preferred residual evaluation for a near-perfect fit. Compute RSS from a QR compression when cancellation is significant:

~~~text
sqrt(W) X = Q R
f = Q.T sqrt(W) (y-offset)
RSS(beta) = residual_norm_squared + ||f - R beta||²
~~~

Accumulate the orthogonal residual norm stably during QR updates, retaining the discarded `Q.T @ y` tail's squared norm at each merge and preserving column pivots. This is a deliberate stability improvement to `bam.fit`'s `y.norm2 - sum(f^2)` residual calculation; the initial compression inside `fast.REML.fit` and augmented QR in `Sl.fit` already demonstrate tail-norm accumulation. Deriving it only as c-||f||² recreates the cancellation problem.

Keep these details explicit:

- The audited PIRLS clips working weights but uses original prior weights in deviance. To reproduce it across clipping edges, keep separate solve and deviance statistics, or change zero/extreme-weight handling in a separately validated numerical change. A single G does not represent both if weights differ.
- Initial delivery can gate on weights for which these conventions coincide; expand after the edge cases are covered.
- Use the current joint scale objective and public reported-scale behavior first. The code distinguishes optimized log_phi from the Fletcher scale stored for reporting. Do not replace either blindly with RSS/(n-p).
- Preserve `Gaussian.saturated_loglik`'s positive-weight count and log-weight term, and `fletcher_scale`/`estimate_scale`'s current row-count conventions. A reduced matrix with p rows must not replace the original n in likelihood or reporting statistics.
- Derive beta sensitivity from the coefficient solve, so cross-penalty derivatives are retained. Automatic differentiation of this coefficient-space score is feasible without any training-row arrays.
- Preserve penalty null-space dimension and generalized determinant treatment in rank-deficient cases.
- Retain Newton for small m and reuse the exact-gradient/L-BFGS strategy for large m. Gaussian compression removes n from optimization, not p³ factorizations or large-m costs.
- Only scan training data again when row-level fitted values, residuals, or a requested validation pass is needed.

Gate: current Gaussian fixed-sp and estimated-sp cases, weights, offsets, multiple/tensor penalties, unknown-scale behavior, near-perfect fits, and rank deficiency agree; source scan counters do not increase during smoothing optimization after statistics are frozen.

Expected cost: one O(n*p²) data reduction, then coefficient-space work per smoothing evaluation. Prediction is a separate pass. Setup itself may have required several earlier scans. Add `tests/test_fitting/test_gaussian.py`; compare the compressed objective/derivatives to current dense REML and the corresponding R reduced fit before adding final-model cases to the validation matrix.

**6. Stage 4A: discretized design and working-model smoothing**

**PR 10 — exact discrete representation before continuous binning.**

Add data/discretize.py, formula/discrete_design.py, and fitting/discrete_ops.py.

Read first: [`R/bam.r::compress.df`/`discrete.mf`][r-discrete-mf] for unique-row indexing, sharing covariate tables, factors, and tensor-margin bookkeeping; [`R/misc.r::XWXd`][r-xwxd], [`XWyd`/`Xbd`/`diagXVXd`][r-discrete-actions] for operation contracts; and [`src/discrete.c::XWXd0`/`XWXijs`/`indReduce`][r-discrete-c] for the actual algorithms. The R wrappers call C; a generic dense pair-histogram implementation would miss the main reuse opportunity. Translate the needed single-predictor, independent-row cases into NumPy descriptors and JAX kernels, with tests against these R routines. AR correlation, multiple linear predictors, and smooth summation conventions in the R implementation are outside this plan's initial scope.

Preserve the frozen basis, normalization, and constraints from PR 5 before creating lookup tables. Running `smoothCon` on an unweighted unique-only frame changes normalization/centering when observations repeat. mgcv's discrete setup uses a compressed/padded model frame and its own constraint representation, so default `bam(discrete=TRUE)` is not the reference for this exact-representation milestone.

For a univariate term, store basis table B_j of shape (q_j,k_j) and row indices g_j. If using observed unique values, this changes representation without approximating covariate values.

~~~text
X_j = L_j B_j                     # L_j is implicit indexed selection
X_j v = (B_j v)[g_j]
X_j.T u = B_j.T segment_sum(u, g_j)
~~~

For dense coefficient cross-products:

~~~text
G_jl = B_j.T C_jl B_l
C_jl[a,b] = sum of current row weights with g_j=a and g_l=b
~~~

Include products of numeric-by multipliers and the appropriate factor indicators in cross-term weights. The C `XWXijs` implementation already specializes the diagonal same-term case, partitions tensor terms into subblocks, and uses `indReduce` to accumulate a sparse weighted pair representation when the dense pair table is too large. Reuse those choices, adapting all row scratch space to B for streamed execution; the original C code uses n-sized work buffers. Build pair tables sequentially or in bounded groups; never retain all q_j-by-q_l tables.

This representation does not eliminate off-diagonal term products or per-row IRLS weights. Across t terms, naive pair aggregation can still cost O(n*t²), before tensor subblocks. If a pair table is too large or sparse with too many unique pairs, use the indexed/direct-product alternative or reject that kernel choice. Compare histogram accumulation, direct block GEMM, and operator execution.

Discretize tensor margins separately; evaluate tensor contractions using marginal tables and coefficient ordering metadata. Port the meaning of `Xd`, `kd`, `ks`, `ts`, `dt`, `v`, `qc`, and dropped-column indices from R, with explicit zero-based indices and translation to JaxGAM's `TermBlock` transforms. mgcv may reorder tensor margins; retain a permutation back to public coefficient and smoothing-parameter order. Do not construct a global grid across d input features. A high-order tensor still has a coefficient product: compression does not remove that model-size explosion. Keep row index arrays in the source/memmap when needed; storing all n-by-margin indices on device is not a bounded-row guarantee.

Gate: at exact unique-value indices, matvec, transpose action, diagonal, and all cross-term blocks agree with the ordinary design. Include correlated covariates, numeric-by terms, factors, and tensors so tests cannot pass by ignoring cross terms.

**PR 11 — optional continuous discretization and working-model REML.**

Read first: `compress.df` rounds metric variables to an evenly spaced range grid only when there are too many unique combinations, while preserving factor values; `discrete.mf` uses a fixed temporary RNG state to shuffle/pad the setup frame. Adapt this existing policy before inventing quantile bins or representative selection. For smoothing, port the sequence in [`R/bam.r::bgam.fitd`][r-bgam-fitd] and [`R/fast-REML.r::Sl.fitChol`][r-sl-fit-chol]: working cross-products, initial block transform, proposed Newton update, coefficient step control, and convergence checks. Use [`R/efam.r::estimate.theta`][r-estimate-theta] only for the corresponding working-model nuisance update. Check the 1.9-3/1.9-4 differences in §0 before choosing the parity target.

Expose the chosen grid policy per term or margin, with representative values and prediction policy. Evaluate full-data deviance on the selected discretized design; retain per-row responses and offsets through the source. Discretization and smoothing strategy are separate controls: the frozen discrete design can use fixed-sp PIRLS or the exact-objective solver; `bgam.fitd`-style performance iteration is an additional opt-in strategy.

First validate fixed-sp discrete fitting against the same explicit discretized design. Then add the large-data smoothing loop:

1. Compute current working model statistics from the discrete design.
2. Take a limited smoothing update on that working model.
3. Update coefficients and, where supported, nuisance parameters.
4. Repeat with damping and convergence checks; record the actual criterion and strategy.

This is a new fitting method for JaxGAM with an existing mgcv implementation to port. It differs from current nested REML even when every covariate already lies on the discrete grid. In particular, the nuisance update in `estimate.theta` conditions on current mu, whereas JaxGAM's estimated-theta NB path uses joint outer REML. Record both execution and smoothing strategy; do not label performance iteration as PR 8's exact-objective optimization.

Refine q per margin and compare predictions, held-out deviance, smooth shapes, EDF, and interval behavior against the undiscretized reference. Do not demand strict coefficient equality across different bases. Repeated exact discrete values require no covariate approximation; continuous bins do.

Prediction must state its semantics. [`predict.bamd`][r-predict-bamd] calls `discrete.mf` on **newdata**, then evaluates the fitted smooth metadata on that newly compressed frame; it does not simply snap all future values to stored training bins. Match that behavior when exposing an mgcv-compatible discrete prediction mode. Freeze the prediction grid over the whole prediction request before yielding batches; independently rediscretizing each batch changes its ranges/representatives. A policy that always uses training-grid representatives, or evaluates every new value continuously, is a separate documented choice. Reuse the fitted knots/transforms in every mode and test held-out values, extrapolation, factors, and changes in prediction batch boundaries.

Gate: first compare discrete operators and fixed-sp fits against the identical explicit design at STRICT/MODERATE tolerance; then compare the chosen working-model implementation to pinned mgcv `bam` with matching setup, grid, ordering, and method. Accuracy relative to an undiscretized `gam` is a separate approximation study. Extend RBridge with explicit `bam`/discrete-operation helpers when needed; the current `fit_gam` helper does not establish this parity. No unbounded joint-grid or pair-table allocation.

**7. Stage 4B: operator design and iterative coefficient solving**

**PR 12 — implement operators and preconditioners with fixed sp first.**

Add fitting/operators.py, fitting/preconditioners.py, fitting/iterative.py, and an execution/operator.py host orchestrator.

Read first: R's `Xbd`/`XWyd` indexed actions and `Sl.Sb` from PRs 2/10; adapt those for matvec/rmatvec rather than designing another smooth evaluator. [`smooth.construct.re.smooth.spec`/`Predict.matrix.random.effect`][r-re] define the random-effect interaction columns, full-rank identity penalty, no-centering rule, and zero contribution for unseen levels. JaxGAM's `RandomEffectSmooth` already implements those semantics densely. The preconditioner and host-streamed Krylov solver are new work: mgcv's dense `bam` cross-products/Cholesky are not an existing no-p² solver.

Make the no-p² contract apply to setup as well as fitting. PR 5 initially supports cubic/modest parametric blocks; PR 12 must add indexed random-effect preparation before claiming high-cardinality support. Avoid `RandomEffectSmooth.setup`'s dense interaction encodings/identity, `CoefficientMap.build`'s requirement for dense training X blocks, dense penalty-root augmentation, dense range bases, and global D. Budget each local block too: a single q-level term or tensor can itself have size p, so a dense "local" block is not safe automatically.

The fused batch operation must not allocate the n-vector Xv outside a batch:

~~~python
def normal_matvec(v):
    out = penalty.apply(v, rho)
    for batch in source.scan(batch_rows):
        # beta and theta are fixed throughout this linear solve
        w = working_weights_at_current_fit(batch, beta, theta)
        xv = design.matvec_batch(batch, v)
        out += design.rmatvec_batch(batch, w * xv)
    return out
~~~

Freeze beta, theta, weights, transforms, and the preconditioner during each Krylov solve. If weight caching is too large, recompute from the same frozen state. Changing them within ordinary PCG changes the linear system.

Start with SPD canonical systems. Add fixed-theta NB only for cases where the chosen observed system is also SPD: fixing theta alone is not an SPD guarantee for every link. Preserve the current observed/Fisher split; switching coefficient updates to Fisher scoring is a separate tested algorithm choice. Use block-Jacobi term Hessians plus penalties for small blocks and diagonal/indexed actions for large blocks, augmented with a budgeted coarse solve for intercept/unpenalized directions. Strong concurvity often makes diagonal preconditioning inadequate; measure convergence before adding a sketch/coarse correction. Multi-device execution is outside this plan.

For indexed random intercepts, use level IDs and weighted counts directly. For a single grouping factor plus a modest fixed/smooth block, an exact Schur complement is an additional option because the random-effect diagonal is cheap to eliminate. Crossed random effects lose this simple structure.

B-splines and P-splines (`bs/ps/cp`) are deferred by the repository's scope and are not a prerequisite or deliverable of PR 12. Use the existing supported smooths. A later spline extension must preserve sparse application through centering and reparameterization; dense Z/D can erase local support. Treat any changed spline space or penalty as a model change.

Use implicit differentiation of converged solves where differentiation is needed. For host-streamed operators, call the solver and its adjoint explicitly. JAX custom_linear_solve is appropriate when the matvec itself is a pure JAX operation, such as resident discrete tables; it cannot directly trace a Python I/O iterator.

Gate: adjoint identity dot(Xv,u)=dot(v,X.Tu), symmetry, scaled residuals, convergence and preconditioner behavior; fixed-sp predictions/deviance match the direct solver. Include correlated terms and sparse high-cardinality factors.

An operator solve costs approximately n_cg full data passes, where n_cg is the Krylov iteration count. With dense bases, arithmetic is O(n_cg*n*p); with indexed support it is O(n_cg*nnz(X)), excluding basis recomputation. Its advantage depends on preconditioning and data bandwidth. Re-reading remote data hundreds of times can be slower than direct statistics accumulation.

**PR 13 — add large-p smoothing selection and uncertainty.**

The [extended Fellner–Schall implementation sub-plan](fellner_schall_implementation_plan.md) owns the theory-to-code mapping, pinned mgcv 1.9-3 reference, update algebra, controller, scale/theta behavior, API integration, and EFS acceptance tests. Read its Wood–Fasiolo paper references alongside `R/gam.fit4.r::efsudr`; `efsud` targets the more general `gam.fit5` path and is not the initial port.

Split delivery rather than requiring all of PR 13 to land together:

- Fixed externally supplied sp is the first useful operator release in PR 12.
- EFS-1 through EFS-5 deliver the dense, exact-trace reference on current `FittingData`, with parent PR 2's local penalties optional for initial correctness. This can ship before PR 12 and avoids the full smoothing-parameter Hessian, not dense coefficient algebra. Exact traces do not remove the general-family EFS approximation or guarantee the Newton REML optimum.
- EFS-5.1 completes the remaining existing-family/link gates; EFS-5.2 adds a streamed direct fit/statistic/score adapter using the PR7.x family contract. Dense EFS support is not evidence that its streamed counterpart exists. These remain manageable-p exact-trace deliverables with EFS-specific scale and in-PIRLS theta policies.
- EFS-6 adapts that reference to validated operator fit/trace/score providers after PR 12. Stochastic traces and log determinants are separate approximation changes; bulk uncertainty remains a separate deliverable under this parent plan.

The EFS update needs traces and quadratic forms, but mgcv's step selection and score-window stopping still require REML scores. A no-p² backend needs a validated exact or approximate score provider; omitting score checks is a different algorithm. The sub-plan also makes the non-monotone mgcv step policy an explicit architecture/release decision, without weakening existing Newton gates.

For stochastic methods, use fixed probes across trial comparisons and expose probe count, Lanczos depth, solver tolerances, seed, residuals, and estimated uncertainty. Increase accuracy when stochastic or solve error is comparable to the proposed improvement or stopping threshold. Do not claim exact REML or treat a noisy decrease as certain. These capabilities are new work, not supplied by mgcv's basis-truncation Lanczos routines.

Project out unidentifiable directions and handle penalty null spaces explicitly before log-determinant estimation. An SPD CG/SLQ implementation is not a solution for arbitrary signed observed Hessians.

For selected prediction variances, solve against requested design rows in bounded groups. For bulk uncertainty, require a separately validated approximation. Saving a dense covariance would defeat the large-p representation.

Gate: dense EFS meets the sub-plan's pinned EFS parity and compatibility tests first. On moderate p, operator/randomized statistics and fits agree with the dense EFS reference within calibrated error; record differences from Newton separately. Results identify approximation and optimization strategy. Public EFS and operator approximations remain post-v1.0 capabilities with their own scope gates.

**8. Performance and numerical acceptance**

Extend the existing [scripts/benchmark_vs_r.py](../../scripts/benchmark_vs_r.py), [scripts/benchmark_large_p.py](../../scripts/benchmark_large_p.py), and [scripts/profile_fit.py](../../scripts/profile_fit.py) fixtures/timing approach; add `scripts/benchmark_scaling.py` only for the new source-scan and memory measurements. The large-p script already compares `gam(REML)` and `bam(fREML)`, but those timings do not establish equal bases/objectives. Both benchmark scripts currently delete a user-global JAX cache; new cold-compilation runs must use an isolated temporary cache rather than copying that behavior. Enforce RBridge's version check for parity measurements.

Add Makefile targets such as `benchmark-scaling` and `test-scaling-memory` when the harness lands, using `uv run`. Existing commands are `make lint`, `make test-local`, `make test-file FILE=...`, `make test` (Docker/R), and `make test-cov`; there is no scaling target yet. Keep large benchmarks outside ordinary unit-test runtime.

Use representative cases rather than a full infeasible Cartesian product:

| Case | Suggested dimensions | Concrete question |
|---|---|---|
| Large n, modest p | n=100k/1m/10m; p around 100/1k | Does streamed memory plateau and does Gaussian optimization stop reading rows? |
| Many additive terms | n=100k/1m; m=10/100/500; k=10/20 | Are penalty storage, compilation, and smoothing derivatives controlled? |
| Expensive kernels | tp/gp with default knot cap; varying input dimension | Are all distance/kernel intermediates batch bounded? |
| Shared variables/tensors | s(x)+te(x,z), repeated smooths, ti terms | Do dependence detection and constraint preparation avoid n²/n*p allocations? |
| High-cardinality effects | 10k/100k/1m groups where supported | Is storage proportional to indices/nonzeros rather than one-hot dense arrays? |
| NB count tails | Modest counts plus large outliers; theta near Poisson limit | Is derivative work bounded without losing first/second derivative accuracy? |
| Numerical stress | Concurvity, zero/extreme weights, offsets, near-perfect Gaussian fits | Do solvers, derivatives, rank decisions, and uncertainty remain valid? |

Measure setup time, cold compilation, warm fitting, source scan count and bytes, inner/outer iterations, derivative time, prediction/SE time, host peak RSS, device peak memory, and temporary disk use. Explicitly synchronize device results before timing. CPU profiler time attributed to array-to-scalar conversion often includes waiting for prior device computation; do not interpret it all as Python conversion overhead.

Define each run's hardware, package versions, dtype, BLAS/thread settings, formula, basis/penalty counts, seed, input source, memory budget, and execution/approximation strategy.

Numerical gates:

- Import the actual constants from `tests/tolerances.py`: STRICT is rtol=1e-10/atol=1e-12 and MODERATE is rtol=1e-4/atol=1e-6. The design document quotes different values. Do not duplicate those stale constants or loosen existing tolerances to make a refactor pass.
- The active repository instructions require R parity at STRICT or MODERATE. Existing tolerance prose and parts of `tests/test_validation_matrix.py` still use LOOSE; preserve those regression assertions, but passing them alone is insufficient acceptance for these PRs. Add explicit STRICT/MODERATE gates for each newly supported equivalent-model path. Record unresolved legacy parity gaps rather than calling them passed or weakening the new gates.
- Keep broad final-model R comparisons and hard-gate invariants in `tests/test_validation_matrix.py`; layer-specific R tests own basis construction, penalty/derivative algebra, fixed-sp PIRLS, prediction/SE, and summary behavior. Use `tests.helpers._AssertCollector` when assertions share an expensive fit. Every new module gets a corresponding test file and >80% coverage; Phase 2 changes require a JIT compilation-and-execution test.
- Compare penalty action, local transformations, score, gradients, stationarity residuals, predictions, deviance, scale, EDF and uncertainty as relevant.
- Hold the prepared basis fixed when comparing solvers. Evaluate changed-basis/discretized methods using approximation and prediction metrics separately.
- Verify memory behavior with increasing n at fixed B/p in isolated processes, subtracting measured input/runtime baselines. Instrument allocation sites and inspect compiler memory estimates in addition to peak-memory sampling.
- Exact direct paths must not allocate n-by-n Q, n-by-p X in streaming, m-by-p-by-p penalty arrays, or n-by-r-by-d distance differences.
- Operator paths additionally must not allocate p-by-p G/H or full covariance. A permitted small coarse preconditioner is accounted for separately.
- Use `make test`/`make test-cov` in the pinned R 4.5.2/mgcv 1.9-3 environment for reference gates. Local R auto-skips are not evidence of parity. Reading 1.9-4 source is useful but does not change the oracle; version-specific behavior needs a documented target and matching tests.

No speedup target is justified by this source audit. The measurable early guarantees are removal of the identified array shapes, fewer data passes, equivalent numerical outputs, and recorded throughput on the target workload.

**9. Merge sequence and exit criteria**

| PR | Deliverable | Dependency | Exit criterion |
|---|---|---|---|
| 1 | Raw-reflector dependence QR | None | No n² Q; same dependence decisions |
| 2 | Local penalties and D transforms | None; merge after 1 | No m global penalties; algebra/derivative parity |
| 3 | Reused bases, bounded kernels, one initialization | 2 where interfaces change | Duplicate-work counters and prediction correctness |
| 4 | NB prefix-sum derivative with bounded workspace and fallback | None; independent Stage 1 fix | First/second derivative parity against recurrence and nb()$ls; explicit fallback cost; synchronized benchmarks |
| 5 | PreparedModel and replayable exact preparation | 2–3 | No full training design during supported setup |
| 6 | Batched prediction and optional covariance | Existing PredictSpec for array batches; 5 for RowSource | Existing full/inference contracts pass; bounded workspace for new mode |
| 7 | Published canonical-family fixed-sp streaming foundation | 5–6 | Gaussian/identity, Poisson/log, binomial/logit; bounded cross-product/Cholesky path only |
| 7.1 | Shared family execution contract and support inventory | 7; existing dense family/PIRLS interfaces | Generic batch/reduction capabilities; unchanged defaults; custom-family conformance |
| 7.2 | Positive-weight QR/TSQR coefficient solver | 7.1; reusable PR9 QR primitives when available | No normal-equation accuracy loss or retained training design; stable rank/pivot handling |
| 7.3 | All regular families and supported links | 7.1–7.2; 7.4 for signed observed information | Every inventoried Gaussian/Poisson/binomial/Gamma fixed-sp cell passes |
| 7.4 | Sign-aware observed-curvature solver and safeguards | 7.1–7.2 | Signed/indefinite intermediate recovery and distinct observed/Fisher contracts |
| 7.5 | NB coefficient fitting and explicit fixed/dynamic theta state | 7.1, 7.4, 4 | NB/log/identity/sqrt fixed-theta parity and safe trial-theta kernels |
| 7.6 | Public all-family fixed-sp streaming integration | 7.1–7.5; 8.2 for free theta with pinned lambda | Five families, all supported links, both NB theta modes; public/memory matrix |
| 8 | Canonical known-scale streamed adjoint/L-BFGS foundation | 7 | Initial Poisson/binomial score/gradient parity without full outer Hessian |
| 8.1 | Regular-family adjoints and unknown-scale optimization | 8, 7.1–7.4 | Non-canonical and joint-phi derivative/final-fit parity |
| 8.2 | NB exact theta/smoothing optimization and pin-lambda mode | 8.1, 7.5, 4 | Theta cotangents, parameter isolation and exact fixed/free-sp NB parity |
| 8.3 | Public all-family exact streamed REML | 7.6, 8.1–8.2 | Full inventoried family/link/parameter-mode release matrix |
| 9 | Gaussian QR/statistics specialization adapted from bam.fit/Sl.fit | 2–3 for dense adapter; 5 for RowSource; 8's optimizer optional | No source scans during smoothing optimization; same REML/scale conventions |
| 10 | Exact discrete-index operators/cross-products | 5, 7; shared family contract from 7.1 | Algebraic parity on repeated values; family mathematics not duplicated |
| 11 | Continuous discretization and working-model smoothing | 10; applicable 7.x family capabilities and validated nuisance-update policy | Quantified approximation/runtime gains; explicit per-family method gates |
| 12 | Fixed-sp operator solves/preconditioning | 2, 5, 7.1; applicable 7.x family contracts | No p² system; controlled solve residuals; solver admissibility remains explicit |
| 13 | [Dense and streamed EFS, then operator smoothing](fellner_schall_implementation_plan.md); separate uncertainty approximations | Dense EFS: current FittingData; streamed EFS: 7.x contracts; operator EFS: 12 plus validated score provider | Pinned EFS parity first; calibrated operator traces/scores and fit error later |

IDs preserve the original roadmap, not a strict numeric merge order. For example, 7.4 can land before 7.3's final signed-curvature gates, and 8.2 supplies estimated-theta fitting required by 7.6. There is no dependency cycle: 8.2 uses the internal family kernels from 7.5, not 7.6's public release adapter. The already-developed PR8 foundation and PR9 can finish while the new follow-ups are implemented in isolated branches; their smaller scopes must stay explicit.

PRs 1–4 are useful dense-backend improvements. PR7 is the initial three-family streamed fixed-sp milestone, **PR7.6 is the all-family fixed-sp milestone**, and **PR8.3 is the all-family exact-REML milestone**. PR9 is the optional Gaussian QR acceleration. Both blockwise/QR and discretized bam-style algorithms are in this plan; a complete drop-in `bam()` API with every R option is a different compatibility claim. PRs 10–13 remain separate post-v1.0 algorithmic deliverables, not conditions for shipping the early optimizations. PR13 remains the final top-level milestone; its EFS sub-plan now explicitly includes wider-family and streamed-provider follow-ups before operator EFS, followed by separately reviewed uncertainty approximations.

Implementation risk is highest in streamed constraints/normalization, exact non-Gaussian derivatives, and operator smoothing. Estimate schedule after PRs 1–5 establish the actual abstractions and target hardware benchmarks; do not assign a speedup or delivery date solely from n.

**10. Source anchors and method references**

This is the implementation reading map, complementing `docs/R_SOURCE_MAP.md` (since removed from the repository). Paths in the mgcv column are relative to `$MGCV_SOURCE`; links resolve to the exact locally inspected commit identified in §0. Function names are the durable anchors; line numbers change between 1.9-3 and 1.9-4.

| PR | Existing JaxGAM anchor | mgcv implementation to read/adapt | Additional work or compatibility limit |
|---|---|---|---|
| 1 | [CoefficientMap.fix_dependence/gam_side](../../jaxgam/smooths/constraints.py) | [R/mgcv.r: fixDependence, augment.smX, gam.side][r-dependence] | Map R's packed QR application to SciPy/LAPACK; preserve dependent-column choices. |
| 2 | [ModelSetup.build](../../jaxgam/formula/design.py), [FittingData helpers](../../jaxgam/fitting/data.py), [block_log_det_S](../../jaxgam/jax_utils.py) | [Sl.setup][r-sl-setup], [Sl.initial.repara][r-sl-repara], [Sl.Sb/Sl.rSb][r-sl-action], [Sl.addS][r-sl-add], [ldetS][r-ldets]; [totalPenaltySpace][r-total-penalty] | Change storage, not penalty order, fitting coordinates, or current rank cutoffs. mgcv's totalPenaltySpace uses eps^.66; JaxGAM uses eps^(2/3), so copying thresholds is a separate numerical change. |
| 3 | [Smooth and prediction-copy hooks](../../jaxgam/smooths/base.py), [distance helper](../../jaxgam/smooths/utils.py), [initialize_beta](../../jaxgam/fitting/initialization.py) | [smoothCon][r-smoothcon], [tensor constructor's Xm reuse][r-tensor], [initial.sp][r-initial-sp] | Explicit training-basis handoff, bounded evaluation, and initialization at the transfer boundary. |
| 4 | [_lgamma_diff_jvp/_saturated_loglik_jax](../../jaxgam/families/negative_binomial.py) | [R/efam.r: nb()$Dd, nb()$ls, nb()$initialize][r-nb] | Prefixes optimize an existing JaxGAM stability workaround; they are not an mgcv algorithm. Preserve derivatives in log(theta). |
| 5 | [PredictSpec/build_predict_spec](../../jaxgam/formula/predict_matrix.py), [ModelSetup.build](../../jaxgam/formula/design.py) | [mini.mf][r-mini-mf], [bam setup][r-bam-setup], [smoothCon][r-smoothcon], [PredictMat][r-predictmat], [cr][r-cr]/[place.knots][r-knots], [tp][r-tp]/[tprs_setup][r-tprs-c], [gp][r-gp], [tensor][r-tensor] | Reuse metadata/evaluators; exact full-data reductions and replayable raw-input I/O are new. Sampled/unique-only setup is not dense setup parity. |
| 6 | [predict_core/finish_prediction](../../jaxgam/inference/_core.py), [GAMPredictor](../../jaxgam/inference/predictor.py), [GAMResults._from_fit](../../jaxgam/results.py) | [predict.gam][r-predict-gam], [predict.bam][r-predict-bam], [Sl.postproc][r-sl-postproc] | Batch existing prediction; optional factor covariance is a new result capability. |
| 7, 7.1–7.6 | [_pirls_loop_jit/pirls_loop](../../jaxgam/fitting/pirls.py), existing family and link interfaces | [qr_update/qr_up][r-qr-update], [bgam.fit][r-bgam-fit], [gam.fit3][r-gam-fit3], [gam.fit4][r-gam-fit4], [gdi1/gdi2][r-gdi], [nb()][r-nb] | Share family mathematics; add QR and signed-system safeguards, explicit parameter state and separately gated all-family public streaming. |
| 8, 8.1–8.3 | [_diff_score/NewtonOptimizer](../../jaxgam/fitting/newton.py), [_criterion_core](../../jaxgam/fitting/reml.py) | [gam.fit3][r-gam-fit3], [gam.fit4][r-gam-fit4], [gdi1/gdi2][r-gdi], [newton][r-newton], [bfgs][r-bfgs] | Preserve the full fitted-model REML objective through regular-family/phi and NB/theta extensions; EFS conditional updates are not interchangeable. |
| 9 | [REML/scale functions](../../jaxgam/fitting/reml.py), [Gaussian.saturated_loglik](../../jaxgam/families/standard.py) | [bam.fit][r-bam-fit], [qr_update][r-qr-update], [Sl.fit/fast.REML.fit][r-sl-fit] | Reuse Gaussian QR compression; preserve original observation counts, weights, and reported scale. |
| 10 | [PredictSpec](../../jaxgam/formula/predict_matrix.py), [tensor evaluation](../../jaxgam/smooths/tensor.py), local penalties from PR 2 | [compress.df/discrete.mf][r-discrete-mf], [XWXd][r-xwxd], [XWyd/Xbd/diagXVXd][r-discrete-actions], [discrete.c indexed reductions][r-discrete-c] | Use R's indexed/block algorithms on the already-frozen JaxGAM basis; bound C-style n-sized workspace by batching. |
| 11 | New discrete operators plus existing family functions | [bgam.fitd][r-bgam-fitd], [Sl.fitChol][r-sl-fit-chol], [estimate.theta][r-estimate-theta], [predict.bamd][r-predict-bamd] | Explicit performance iteration and prediction policy, with pinned-version bam parity and separate approximation evaluation. |
| 12 | [RandomEffectSmooth](../../jaxgam/smooths/random_effects.py), existing smooth evaluators and PR 2 actions | [random-effect constructor/prediction][r-re], [Xbd/XWyd][r-discrete-actions], [Sl.Sb][r-sl-action] | Indexed high-cardinality setup and operator solver/preconditioner are new. No implied sparse bam/PCG backend exists in JaxGAM. |
| 13 | Existing FittingData/PIRLS/block determinant machinery first; new operators later | [efsudr][r-efsudr] (gam.fit3/4); [dedicated EFS plan](fellner_schall_implementation_plan.md) pins the 1.9-3 oracle and paper mapping | Port dense EFS first; conditional NB theta Newton is distinct from outer EFS. Operator traces/scores and bulk uncertainty need separately validated approximations. |

For each implementation PR, record the exact R function/branch ported and any deliberate departure (for example QR residual-tail accumulation, preserved JaxGAM clipping, or a bounded-memory derivative). Run the layer test against the corresponding R operation, then the existing dense regression gates. Do not use a `bam` runtime comparison, a default discretized fit, or a skipped R test as evidence of identical `gam(REML)` behavior.

[r-dependence]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/mgcv.r#L502
[r-sl-setup]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/fast-REML.r#L68
[r-sl-repara]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/fast-REML.r#L517
[r-sl-action]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/fast-REML.r#L431
[r-sl-add]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/fast-REML.r#L1016
[r-ldets]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/fast-REML.r#L762
[r-initial-sp]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/mgcv.r#L4626
[r-total-penalty]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/gam.fit3.r#L2661
[r-smoothcon]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/smooth.r#L3770
[r-predictmat]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/smooth.r#L4302
[r-tp]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/smooth.r#L1248
[r-tprs-c]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/src/tprs.c#L307
[r-cr]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/smooth.r#L1444
[r-knots]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/smooth.r#L1562
[r-gp]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/smooth.r#L3441
[r-tensor]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/smooth.r#L741
[r-re]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/smooth.r#L2571
[r-nb]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/efam.r#L161
[r-qr-update]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/bam.r#L46
[r-mini-mf]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/bam.r#L384
[r-bam-setup]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/bam.r#L2376
[r-predict-gam]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/mgcv.r#L2692
[r-predict-bam]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/bam.r#L1415
[r-sl-postproc]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/fast-REML.r#L1976
[r-bgam-fit]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/bam.r#L909
[r-gam-fit3]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/gam.fit3.r#L67
[r-gam-fit4]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/gam.fit4.r#L240
[r-gdi]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/src/gdi.c#L1953
[r-newton]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/gam.fit3.r#L1290
[r-bfgs]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/gam.fit3.r#L1722
[r-bam-fit]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/bam.r#L1503
[r-sl-fit]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/fast-REML.r#L1682
[r-discrete-mf]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/bam.r#L122
[r-xwxd]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/misc.r#L191
[r-discrete-actions]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/misc.r#L333
[r-discrete-c]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/src/discrete.c#L57
[r-bgam-fitd]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/bam.r#L430
[r-sl-fit-chol]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/fast-REML.r#L1585
[r-estimate-theta]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/efam.r#L5
[r-predict-bamd]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/bam.r#L1773
[r-efsudr]: https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/gam.fit4.r#L822
