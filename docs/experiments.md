# Newton Optimizer Experiments Log

## Baseline

**Committed code (HEAD = b44d3c0)**: 26 validation matrix failures, 72/72 Newton tests pass.

Failures breakdown:
- **ti-* (all 4 families)**: 3 fitted + 4 edf + 4 coef = 11 failures - caused by incorrect centering constraint on `ti()` smooths (constraints.py applies sum-to-zero after ti already absorbed marginal constraints)
- **tp_by-binomial, cr_by-binomial, te_by-binomial, te_by-poisson**: 4 fitted + 4 edf + 4 coef = 12 failures - factor-by / tensor GLM models with multiple sp on flat REML surfaces
- **te-gamma**: 1 convergence failure - flat REML surface, optimizer never converges
- **ti-gaussian**: 1 edf + 1 coef = 2 failures (part of ti centering bug)

Committed code settings:
- `lsp_max = 15.0`
- Norm-based step cap (`step * max_step / step_norm`)
- Quadratic error check + steepest descent fallback + STUCK outcome
- NO uconv.ind subsetting
- NO adaptive reparameterization
- AD gradient for all families (no FD)
- `_check_convergence()` combines grad + reml checks
- `max_iter = 200`

## Non-Newton Changes (applied in all experiments below)

These changes are independent of the Newton optimizer and should be kept:

1. **`jaxgam/fitting/reml.py`**: Diagonal preconditioning for `log|H|` - replaces `slogdet(H)` with `cholesky(D^{-1} H D^{-1})` + `2*sum(log(diag(L))) + 2*sum(log(d))`. Improves conditioning for binomial (W varies 0-0.25) and reduces AD noise through jax.hessian.

2. **`jaxgam/smooths/constraints.py`**: Skip sum-to-zero centering for `TensorInteractionSmooth` (ti). ti() already absorbs marginal constraints during construction; applying centering again incorrectly removes a column. **This fixed all 11 ti-* failures.**

3. **`jaxgam/fitting/data.py`**: Added `multi_block_S_local` field to FittingData for adaptive reparameterization support.

4. **`jaxgam/jax_utils.py`**: Added `recompute_multi_block_proj()` for adaptive reparameterization.

## Experiment Results

All experiments below include the 4 non-Newton changes listed above.

### Experiment 1: lsp_max=40, no uconv, no adaptive reparam, no practical convergence

Settings: `lsp_max=40`, norm-based step cap, quadratic error check + STUCK, no uconv.ind, no adaptive reparameterization.

**Result: 14 failures, 72/72 Newton tests pass**

Failures:
- tp_by-binomial, cr_by-binomial, te_by-binomial, te_by-poisson: fitted + edf + coef (12)
- te-gamma, ti-gamma: convergence (2)

### Experiment 2: lsp_max=40, component-wise step clamp, no uconv, no adaptive reparam

Same as Exp 1 but with component-wise clamping (`step * max_step / max(|step|)`) instead of norm-based.

**Result: 14 failures, 72/72 Newton tests pass**

Same failures as Exp 1. Step capping approach makes no difference.

### Experiment 3: lsp_max=40, no uconv, WITH adaptive reparam

Same as Exp 1 but with adaptive reparameterization (recompute multi-block projections at each Newton iteration).

**Result: 14 failures, 72/72 Newton tests pass**

Same failures as Exp 1. Adaptive reparameterization makes no difference.

### Experiment 4: lsp_max=40, WITH uconv.ind, no adaptive reparam

Same as Exp 1 but with uconv.ind subsetting of converged dimensions.

**Result: 14 failures, 72/72 Newton tests pass**

Same failures as Exp 1. uconv.ind makes no difference.

### Experiment 5: lsp_max=15, no uconv, no adaptive reparam

Same as Exp 1 but with `lsp_max=15` (matching committed code).

**Result: 16 failures, 72/72 Newton tests pass**

Extra failures vs Exp 1: cr_by-gaussian and te_by-gaussian convergence failures. lsp_max=15 is too restrictive for these models.

### Experiment 6: lsp_max=40, practical convergence (5 consecutive reml_converged)

Same as Exp 1 plus: if `reml_converged` is True for 5 consecutive accepted steps, declare convergence.

**Result: 13 validation failures, 1 Newton test failure**

Fixed: te-gamma and ti-gamma convergence. New failure: ti-gamma EDF. **Regressed**: `TestMultiSmooth::test_two_smooths_vs_r` Newton test - practical convergence exits too early for two-smooth Gaussian (deviance 15.919 vs R's 15.917, rtol 1.3e-4 > 1e-4 threshold).

### Experiment 7: lsp_max=40, practical convergence (10 consecutive reml_converged)

Same as Exp 6 but with threshold increased to 10 consecutive.

**Result: 13 validation failures, 1 Newton test failure**

Same results as Exp 6 - the two-smooth model's reml_converged stays True from iter ~78 onward, so threshold 10 triggers at iter ~88, still too early.

### Experiment 8: lsp_max=15, practical convergence (10 consecutive)

Combining lsp_max=15 with practical convergence.

**Result: 13 validation failures, 1 Newton test failure**

Same as Exp 7 - practical convergence dominates over lsp_max.

### Experiment 9: lsp_max=15, expected change convergence (`|g^T step| < tol * reml_scale`)

Instead of counting consecutive reml_converged, check if the expected first-order score change from the Newton step is below threshold.

**Result: 16 validation failures, 1 Newton test failure**

Worse - the expected change check is too aggressive for single-smooth Gaussian models (tp-gaussian, cr-gaussian now fail because sp hits lsp_max=15 and expected change is trivially small).

### Experiment 10: lsp_max=40, expected change convergence

Same as Exp 9 but with lsp_max=40.

**Result: Not fully tested** (interrupted), but Newton test `test_fitted_values_vs_r[gaussian]` failed - expected change convergence too aggressive even with lsp_max=40.

### PIRLS-outer / Newton-inner experiments (from previous session)

**PIRLS-outer dispatch enabled** (non-Gaussian families use `_run_glm()` with inner Newton at fixed XtWX):

**Result: 24 validation failures**

The inner Newton over-optimizes sp in flat REML directions because it runs to full convergence at each PIRLS iteration. On flat surfaces, inner Newton pushes sp to extreme values (10^9-10^10 vs R's 10^4-10^7).

## Root Cause Analysis

### The 12 persistent GLM failures (tp_by-binomial, cr_by-binomial, te_by-binomial, te_by-poisson)

These 4 cells fail fitted values, EDF, and coefficients in ALL experiments. They are factor-by or tensor models with binomial/poisson families (non-constant working weights) and multiple smoothing parameters.

The core issue: AD gradient treats PIRLS output as fixed. By PIRLS stationarity (IFT), this is asymptotically exact at the converged solution. But for factor-by models with binomial, the PIRLS convergence is approximate (not exact), and the residual gradient error is enough to push the optimizer to slightly different sp values than R.

The plan's hypothesis was that PIRLS-outer/Newton-inner (fixing XtWX per outer iteration) would make AD exact. This was correct in theory but failed in practice because the inner Newton over-optimized on flat surfaces.

### The 2 convergence failures (te-gamma, ti-gamma)

These are tensor/interaction models with Gamma family (unknown scale, joint optimization of log_lambda + log_phi = 3 params). One smoothing parameter is on a flat REML surface where the Hessian eigenvalue is ~0.004 vs ~104 for the well-conditioned direction (condition number 26000:1).

The gradient decreases linearly (not quadratically): from 9.58 to 0.06 in 200 iterations. Linear convergence is characteristic of inaccurate Hessian (AD through PIRLS). The gradient threshold is `tol * reml_scale ≈ 5e-6`, and at iter 200 the gradient is still 12x above threshold.

R handles this via lsp_max=15: the flat parameter hits the cap, projected gradient zeros it, and convergence triggers. Our lsp_max=40 is too permissive for this mechanism to work, but lsp_max=15 causes other convergence failures.

Practical convergence (consecutive reml_converged) fixes these but regresses the two-smooth Gaussian Newton test. Expected change convergence (`|g^T step|`) is too aggressive.

## Key Findings

1. **Step capping approach (norm vs component-wise)**: No effect.
2. **uconv.ind subsetting**: No effect.
3. **Adaptive reparameterization**: No effect.
4. **lsp_max**: 15 causes extra convergence failures; 40 causes flat-surface drift but allows more models to converge.
5. **Practical convergence**: Fixes te-gamma/ti-gamma convergence but regresses two-smooth Gaussian Newton test.
6. **PIRLS-outer/Newton-inner**: Makes things worse (24 failures) due to over-optimization in flat directions.
7. **The 12 core GLM failures are robust to ALL optimizer tuning** - they appear in every experiment. The issue is fundamental to AD-through-PIRLS gradient accuracy for these specific model configurations.
8. **The reml.py preconditioning and constraints.py ti fix together improved from 26 → 14 failures** (12 failure reduction), which is the main improvement.

### Experiment 11: R-matching Newton changes (family-dependent tol, R's convergence logic)

Comprehensive set of changes to match R's `newton()` (gam.fit3.r lines 1290-1719), identified by reading R source:

**New changes** (not previously tried):
- Family-dependent tolerance: `sqrt(eps)` for Gaussian (matching `fast.REML.fit`), `1e-6` for non-Gaussian (matching `gam.control()$newton$conv.tol`)
- PIRLS tolerance tightened to `min(tol/100, 1e-8)` (R line 1308)
- R's `score_scale = abs(log(scale)) + abs(score)` for REML convergence (R line 1648)
- Gradient convergence uses `5 * conv_tol` factor (R line 1652)
- Eigenvalue floor `eps^0.7` instead of `sqrt(eps)` (R line 1450)
- `is_pdef` gating: only accept Newton step immediately if Hessian is positive definite (R line 1499)

**Already tried** (included for completeness, no independent effect per Exp 2/4):
- Component-wise step cap (same as Exp 2)
- uconv.ind subsetting (same as Exp 4)

Settings: `lsp_max=40`, component-wise step cap, R's convergence logic, uconv.ind, projected gradient at bounds.

**Result: 15 validation failures (+1 vs Exp 1), 5 Newton test signature failures + 1 Newton test regression**

Changes vs Exp 1 (14 failures):
- **Fixed**: te-gamma, ti-gamma convergence (2 fixed) - the `1e-6` tolerance is loose enough to declare convergence before 200 iterations
- **New failures**: te-gamma EDF, ti-poisson EDF, ti-gamma EDF (3 new) - models now converge but to slightly different sp values than R, causing EDF mismatch
- **Newton test regression**: `test_tensor_product_vs_r` - sp drifts to 3.24e6 on flat surface (R: 4.80e5), outside the `well_determined` threshold

The 5 `TestSafeNewtonStep` unit test failures are purely signature changes (function now returns tuple `(step, is_pdef)` instead of just `step`) - not behavioral regressions.

**Analysis**: The family-dependent tolerance is the key change. For non-Gaussian, `1e-6` (67x looser than `sqrt(eps)`) allows convergence in flat-surface models, but the looser stopping point lands at different sp values. The other R-matching changes (score_scale, 5*tol factor, eps^0.7 floor, is_pdef gating) have marginal or no independent effect - the tolerance dominates.

**Net assessment**: Experiment 11 trades 2 convergence failures for 3 EDF failures - a net regression. The family-dependent tolerance concept is sound but needs refinement to avoid the EDF regressions.

### Experiment 12: IFT gradient correction for non-Gaussian families

The AD gradient through the REML criterion treats XtWX in `H = XtWX + S_λ` as constant w.r.t. ρ = log(λ). For non-Gaussian families, the working weights W depend on μ(β*(ρ)) through the implicit function theorem. The missing gradient term is:

```
Δ(dREML/dρ_j) = (1/2) tr(H⁻¹ dXtWX_j)
```

where `dXtWX_j = X^T diag(dW_j) X`, `dW_j = (dW/dη) * (X @ dβ*/dρ_j)`, and `dβ*/dρ_j = -H⁻¹(λ_j S_j β*)` by IFT at PIRLS stationarity.

Implementation: `_ift_gradient_correction()` method on `NewtonOptimizer`. Uses `jax.jvp` through `η → μ → W` for `dW/dη`, batched `cho_solve` for `dβ*/dρ_j`, and the efficient formula `correction_j = 0.5 * dW_j · diag(X H⁻¹ X^T)`. For Gaussian, W is constant (dW/dη = 0), so the correction is exactly zero and computation is skipped.

Settings: All Exp 11 changes + IFT gradient correction added to `_check_convergence()`.

**Result: 3 validation failures, 80/80 Newton tests pass (8 new IFT tests)**

Failures:
- te-poisson: convergence
- ti-poisson: convergence
- te_by-binomial: convergence

Changes vs Exp 11 (15 failures):
- **Fixed**: All 12 persistent GLM fitted/edf/coef failures (tp_by-binomial, cr_by-binomial, te_by-binomial, te_by-poisson fitted + edf + coef) - the corrected gradient steers the optimizer to the same sp as R
- **Fixed**: te-gamma, ti-gamma convergence - corrected gradient has faster convergence on flat surfaces
- **Fixed**: te-gamma EDF, ti-poisson EDF, ti-gamma EDF - optimizer reaches correct sp with accurate gradient
- **Remaining**: 3 convergence failures for multi-penalty non-Gaussian tensor/interaction models

New tests added:
- `test_gaussian_correction_is_zero`: IFT correction is exactly 0 for Gaussian
- `test_non_gaussian_correction_nonzero`: correction is nonzero for Poisson/Binomial/Gamma
- `test_corrected_grad_closer_to_fd`: IFT-corrected gradient is closer to central FD gradient than uncorrected
- `test_convergence_speed`: single-smooth non-Gaussian converges in <30 iterations

**Analysis**: The IFT correction is the single most impactful change across all experiments. It fixes the fundamental issue: AD-through-PIRLS misses the `dXtWX/dρ` contribution to `d(log|H|)/dρ`. Without it, the gradient has a systematic bias for non-Gaussian families that compounds through the optimization. The 3 remaining convergence failures are multi-penalty tensor/interaction models where flat REML surfaces and the `sqrt(eps)` convergence tolerance make convergence difficult within 200 iterations.

### Experiment 13: Family-dependent tolerance (1e-6 for non-Gaussian)

Same as Exp 12 plus: `tol = 1e-6` for non-Gaussian families (matching R's `gam.control()$newton$conv.tol`), `sqrt(eps)` for Gaussian (matching `fast.REML.fit`).

**Result: 2 validation failures, 80/80 Newton tests pass**

Failures:
- ti-poisson: EDF
- ti-gamma: EDF (2.368 vs R's 2.335, 1.4%)

Changes vs Exp 12 (3 failures):
- **Fixed**: te-poisson, ti-poisson, te_by-binomial convergence - 1e-6 tolerance allows convergence on flat surfaces
- **New**: ti-poisson EDF, ti-gamma EDF - optimizer converges to different sp on flat REML surfaces (scores match R to 4 sig figs)

### Experiment 14: IFT Hessian correction (terms [1]+[2]+[4]+[5]) - reverted

Tried adding the missing d²(log|H|)/dρ² terms to the Hessian: term [1] (second-order IFT with d²β/dρ² and d²W/dη²), term [2] (product of first weight derivs), terms [4]+[5] (cross weight-penalty coupling). This matches R's analytical Hessian from `get_ddetXWXpS` in gdi.c.

**Result: 2 validation failures, 80/80 Newton tests pass - identical to Exp 13**

Ablation study comparing no Hessian correction / partial / full showed **zero effect** on iteration counts:

| Model | No Hess | [2]+[4]+[5] | Full [1-5] | R |
|---|---|---|---|---|
| tp-poisson | 7 | 7 | 7 | 3 |
| tp-binomial | 15 | 15 | 15 | 3 |
| te-poisson | 72 | 72 | 71 | 7 |
| ti-poisson | 89 | 89 | 88 | 7 |
| te_by-binomial | 153 | 152 | 148 | 10 |

The IFT Hessian correction terms are numerically small relative to the AD Hessian (`jax.hessian` captures terms [3]+[6] which dominate). The 2-15x iteration gap vs R is driven by AD noise in `jax.hessian` through PIRLS/slogdet, not systematic Hessian bias.

**Decision**: Hessian correction reverted - no benefit for the added complexity.

### Experiment 15: FD Hessian from IFT-corrected gradient - reverted

Hypothesis: `jax.hessian` has noise from differentiating twice through PIRLS/slogdet, causing erratic Newton steps. Replace with central FD of the (accurate) IFT-corrected gradient for non-Gaussian families.

**Result: 2 validation failures, 80/80 Newton tests pass - identical to Exp 13**

Iteration counts unchanged for ALL models. Diagnostic at convergence (ti-poisson) revealed:
- `|AD_hess - FD_of_AD_grad| = 1.3e-10` - the AD Hessian is NOT noisy
- `|AD_hess - FD_of_corrected_grad| = 8e-4` - the IFT Hessian correction is negligible
- AD eigenvalues [0.110, 0.161] vs FD eigenvalues [0.111, 0.161] - both well-conditioned
- Gradient at convergence: |grad_max| = 1.8e-3, right at the 5*tol*score_scale threshold

**Key insight**: The Hessian was never the bottleneck - it's already accurate via `jax.hessian`. The 2-15x iteration gap vs R is NOT from Hessian noise. The AD second derivative through PIRLS/slogdet is clean. The slow convergence must stem from a different cause (step acceptance logic, quadratic error checks, or trajectory differences from the non-analytical gradient).

**Decision**: FD Hessian reverted - identical results with added complexity.

**CRITICAL NOTE**: Exp 15's FD Hessian was computed incorrectly. It used a FIXED PIRLS result for all FD perturbations (same β*, W, XtWX at ρ±h). This is equivalent to the AD Hessian. The TRUE FD Hessian requires re-running PIRLS at each perturbed ρ to capture how β* and W change with ρ. See Experiment 16 for the correct approach.

### Experiment 16: FD Hessian with PIRLS reconvergence - **the fix**

**Root cause discovery**: Instrumented comparison of Python vs R trajectories on ti-poisson revealed:
1. Python converges linearly (~5% gradient reduction/iter, 89 iterations vs R's 7)
2. All 89 Newton steps accepted immediately (PD Hessian, qerror < 0.8) - no halving
3. qerror ~0.45 throughout early optimization (quadratic model predicts only 55% of actual change)
4. R's REML criterion and gradient match Python's exactly (at R's optimum: score diff 2e-12, gradient diff 0)

Key diagnostic: computed the TRUE Hessian at multiple points along the optimization path by FD of the IFT-corrected gradient with PIRLS reconvergence at each perturbation (unlike Exp 15 which used a fixed PIRLS):

| Point | AD Hessian eigs | True Hessian eigs | Relative error | Step error |
|---|---|---|---|---|
| iter 1 (sp=[0,0]) | [0.87, 1.16] | [0.26, 0.47] | 71% | 369% |
| iter 5 (sp=[3.4,0.9]) | [0.87, 1.16] | [0.26, 0.47] | 71% | 378% |
| iter 20 (sp=[4.4,2.6]) | [0.52, 0.74] | [0.03, 0.20] | 93% | 2606% |
| converged (sp=[5.5,5.8]) | [0.11, 0.16] | [0.005, 0.013] | 92% | 2261% |

**The AD Hessian is 71-93% wrong** throughout the optimization because it treats β*, W, and XtWX from PIRLS as constants. The true Hessian (accounting for how PIRLS reconverges at each ρ) has 10-30x smaller eigenvalues. The AD Hessian massively overestimates curvature, producing Newton steps that are 10-30x too small, causing linear convergence.

Exp 15's FD Hessian didn't help because it used a FIXED PIRLS result: `criterion.gradient(ρ±h)` evaluates the gradient at perturbed ρ but with the SAME baked-in PIRLS result. The correct approach re-runs PIRLS at each ρ±h.

**Implementation**: `_fd_hessian()` method on `NewtonOptimizer`. For each of the 2m perturbations (central FD), runs PIRLS to convergence at the perturbed ρ, computes the IFT-corrected gradient, and assembles the Hessian. Used for non-Gaussian families only; Gaussian uses the exact AD Hessian (W is constant).

**Cost**: 2m extra PIRLS calls per Newton iteration (m = number of smoothing parameters). But 5-12x fewer Newton iterations, so total PIRLS calls decrease:
- ti-poisson (m=2): 9 iters × 5 PIRLS = 45 total (was 89 × 1 = 89)
- te-poisson (m=4): 7 iters × 9 PIRLS = 63 total (was 72 × 1 = 72)
- tp-poisson (m=1): 4 iters × 3 PIRLS = 12 total (was 9 × 1 = 9)

**Result: 0 validation failures, 80/80 Newton tests, 1448/1448 total tests pass**

Full iteration comparison vs R:

| Model | FD Hess (Exp 16) | AD Hess (Exp 13) | R | Ratio |
|---|---|---|---|---|
| tp-poisson | 4 | 9 | 3 | 1.3x |
| tp-binomial | 4 | 15 | 3 | 1.3x |
| tp-gamma | 2 | 6 | 6 | 0.3x |
| te-poisson | 7 | 72 | 7 | 1.0x |
| te-gamma | 9 | 46 | 10 | 0.9x |
| ti-poisson | 9 | 89 | 7 | 1.3x |
| ti-gamma | 10 | 33 | 6 | 1.7x |
| tp_by-poisson | 8 | 22 | 8 | 1.0x |
| tp_by-binomial | 10 | 20 | 10 | 1.0x |
| te_by-binomial | 10 | 153 | 10 | 1.0x |
| te_by-poisson | 11 | 61 | 11 | 1.0x |

ti-poisson sp convergence:
- FD Hessian: [5.652, 6.279] - matches R's [5.645, 6.276] within 0.01
- AD Hessian: [5.464, 5.807] - differs by [0.18, 0.47]
- REML score: 368.442559 vs R's 368.442560 (diff 1e-6)

### Experiment 17: FD Hessian threshold (m ≤ 4)

**Goal**: Reduce total PIRLS cost by only using FD Hessian when m is small enough to be cost-effective.

**Analysis**: Pure FD Hessian (Exp 16) costs 2m extra PIRLS per Newton iteration. For small m (1-4), the 5-12x iteration reduction more than compensates. For large m (12+), the per-iteration overhead dominates: te_by models cost 25x more PIRLS with FD vs AD.

**Implementation**: In `_check_convergence()`, use `len(params) ≤ 4` as the threshold:
- m ≤ 4: FD Hessian with PIRLS reconvergence (near-quadratic convergence)
- m > 4: AD Hessian (linear convergence, but cheaper per iteration)

**Cost comparison** (total PIRLS calls across all 11 test models):
- Pure AD (Exp 13): 537 PIRLS, 2 validation failures
- Pure FD (Exp 16): 5894 PIRLS (11x), 0 failures
- Threshold m≤4 (Exp 17): 647 PIRLS (1.2x), 0 failures

The threshold gives 0 failures at only ~20% overhead vs pure AD, avoiding the 11x blowup from pure FD on high-m models.

**Result: 0 validation failures, 80/80 Newton tests, 1448/1448 total tests pass**

### Experiment 18: custom_jvp on PIRLS - analytical Hessian via implicit differentiation

**Goal**: Replace both `_ift_gradient_correction()` and `_fd_hessian()` with a single `jax.custom_jvp` mechanism that gives the correct analytical gradient and Hessian at O(p²n) cost per iteration, with no extra PIRLS calls.

**Approach**: Define `jax.custom_jvp` on the PIRLS output function `(S_lambda, beta_warm) -> (beta*, XtWX, deviance)`. The JVP rule computes tangents via the IFT:
- `dβ = -H⁻¹(dS @ β)` (Cholesky solve)
- `dW = jax.jvp` through `η → μ → W` chain
- `dXtWX = X^T diag(dW) X`
- `ddeviance = jax.jvp` through `η → μ → deviance` chain

Then compose with `_criterion_core` to form an end-to-end `score(params, beta_warm)` function. `jax.grad(score)` gives the correct gradient (no separate IFT correction needed). `jax.hessian(score)` gives the correct Hessian by differentiating through the JVP rule itself - since the rule uses standard JAX ops (cho_solve, jvp, matmul), higher-order AD works automatically.

**Implementation**:
- `_build_differentiable_fns()` method on `NewtonOptimizer`, called once in `__init__` for non-Gaussian families
- Returns pre-JIT'd `(grad_fn, hess_fn)` closures
- `_check_convergence()` uses these instead of `criterion.gradient() + _ift_gradient_correction()` + `_fd_hessian()` / `criterion.hessian()`
- Handles all 4 variants: REML/ML × fixed/joint scale
- `_ift_gradient_correction()` and `_fd_hessian()` removed (dead code)

**Result: 0 validation failures, 86/86 Newton tests, 1448/1448 total tests pass**

Iteration comparison vs Exp 17 and R:

| Model | Exp 18 (custom_jvp) | Exp 17 (FD m≤4) | Exp 13 (AD) | R |
|---|---|---|---|---|
| cr-poisson (m=1) | 3 | 4 | 9 | 3 |
| cr-binomial (m=1) | 4 | 4 | 15 | 3 |
| cr-gamma (m=1) | 4 | 2 | 6 | 6 |
| ti-poisson (m=2) | 10 | 9 | 89 | 7 |
| te-poisson (m=2) | 9 | 7 | 72 | 7 |
| te-gamma (m=2+1) | 10 | 9 | 46 | 10 |
| ti-gamma (m=2+1) | 11 | 10 | 33 | 6 |
| cr_by-binomial (m=3) | 10 | 10 | 20 | 10 |
| cr_by-poisson (m=3) | 10 | 8 | 22 | 8 |
| te_by-binomial (m=6) | 10 | **153** | 153 | 10 |

Key improvements over Exp 17:
- **te_by-binomial (m=6)**: 10 iters (was 153 with AD Hessian since m>4 threshold)
- **No m≤4 threshold needed**: all models get the correct Hessian regardless of m
- **1 PIRLS per Newton iter** (was 1 + 2m for m≤4 models)
- **Total PIRLS calls across all models**: ~100 (was 647 for Exp 17, 537 for Exp 13)

Cost comparison per Newton iteration:

| Approach | PIRLS per iter | Hessian cost | Newton iters (te_by, m=6) |
|---|---|---|---|
| Pure AD (Exp 13) | 1 | O(m² · JIT trace) | 153 |
| FD m≤4 (Exp 17) | 1 + 2m (m≤4) / 1 (m>4) | O(m · PIRLS) | 153 (m>4, AD) |
| custom_jvp (Exp 18) | 1 | O(p²n) analytical | 10 |
| R (gdi.c) | 1 | O(p²n) analytical | 10 |

## Best Configuration Found

**Experiment 18** (0 failures, 0 Newton test regressions): `custom_jvp` on PIRLS for non-Gaussian families gives the correct analytical Hessian via implicit differentiation at O(p²n) cost, matching R's `gdi.c` approach. No m≤4 threshold, no extra PIRLS calls, near-quadratic convergence for all models. Down from 26 failures at baseline to 0.
