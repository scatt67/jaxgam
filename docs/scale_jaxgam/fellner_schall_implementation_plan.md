# JaxGAM: extended Fellner–Schall implementation plan

Status: proposed post-v1.0 work; no EFS implementation or public API is enabled by this document. Audited 2026-09-06 against JaxGAM commit `c1a31346188186bbd2fe67eea8d0d51ba6c95d79`.

Parent: [JaxGAM scaling implementation plan](jaxgam_scaling_implementation_plan.md), PR 13. This sub-plan owns the EFS algorithm, dense reference implementation, and later operator adapter. The parent owns execution representations, preparation, memory budgets, and bulk uncertainty. EFS means the generalized/extended Fellner–Schall smoothing update used by mgcv's `efsudr`, not `bam(method="fREML")` and not the existing Newton REML optimizer.

## 1. Outcome and delivery boundary

Deliver an explicitly selected EFS optimizer that reuses JaxGAM's basis, penalties, PIRLS, and result construction, with update-level and final-fit parity against the corresponding pinned mgcv EFS path. Start with exact dense trace evaluation at manageable coefficient count p. Its initial benefit is avoiding the full smoothing-parameter Hessian when the number of penalties m is large; it does not remove dense X, coefficient factorizations, or every O(p²) allocation.

Deliver dense reference, streamed adaptation, and operator adaptation separately:

1. **Dense reference, EFS-1 through EFS-5:** deterministic penalty statistics, mgcv-style outer control, regular-family scale handling, then NB theta integration. No operator solver, stochastic trace, streaming input, or new smooth is required.
2. **Wider family coverage and streamed direct providers, EFS-5.1–EFS-5.2:** make the remaining existing-family/link gates explicit, then reuse the parent plan's family contract and streamed coefficient kernels without changing the EFS controller. This remains a manageable-p, exact-trace backend, not a no-p² operator claim.
3. **Operator adapter, EFS-6:** substitute validated iterative solves and, subsequently, stochastic statistics while retaining the dense EFS reference. This requires parent PR 12. It is a separate approximation contract, not a storage-only port of `efsudr`.

The dense work can start on today's `FittingData`. Parent PR 2's local penalties reduce its memory cost but are not a correctness prerequisite. Parent PRs 5, 7, 8, 9, 10, and 12 are not prerequisites for dense EFS. Gaussian QR compression from PR 9 is an optional fit-provider adapter; the NB derivative optimization in PR 4 is an optional performance improvement, not a replacement for the theta algorithm below.

**September 8 clarification:** parent PR7 ([#40](https://github.com/scatt67/jaxgam/pull/40)) only delivers canonical Gaussian/Poisson/binomial cross-product/Cholesky streaming. Its explicit follow-ups PR7.1–7.6 own the generic family contract, QR/TSQR, signed curvature, remaining regular links, NB parameter plumbing and public all-family release. Dense EFS support does not automatically enable streamed EFS, nor does a streamed coefficient fit automatically supply an exact REML derivative. Parent PR8.1–8.3 own exact-REML derivatives/optimization; EFS continues to own its distinct theta-update and scale-carry policies. Use capabilities and shared family kernels rather than add another family-name-based streaming engine.

### Initial support and expansion

| Delivery | Families / models | Gate before expansion |
|---|---|---|
| First dense controller | Poisson/log and binomial/logit; fixed known scale; finite positive weights; identifiable penalized system; cubic additive and coupled `te`/`ti` penalties | Exact trace/update parity and real fitted-model EFS parity, not only singleton-penalty examples |
| Regular-family scale | Gaussian/identity, then Gamma with individually validated links | Incoming score scale versus updated/reported scale is reproduced; final covariance/EDF remain compatible |
| Extended family | NB/log with fixed theta, then estimated theta | Fisher trace versus observed-score split; conditional theta Newton and joint coefficient/theta convergence |
| Wider existing basis/link coverage | Existing `tp/ts`, `cr/cs/cc`, `gp`, dense `re`, factor-by, other supported links | Frozen basis parity; rank, clipping, zero/extreme weights, and family-specific validity tests |
| Operator EFS | Initially fixed-scale, supported positive-definite systems | Parent PR 12 and the scored-iteration/trace accuracy contract in §9 |

No automatic switch from Newton based on n, p, or m. Unsupported EFS combinations must fail clearly at preflight, not change the basis, optimizer, or likelihood silently. ML, GCV/UBRE, `gam.fit5` general families, Tweedie, adaptive smooths, `ps/cp/bs`, `fs`, sparse direct libraries, and distributed execution are not deliverables here. The paper's examples do not expand repository scope.

### Required architecture reconciliation

[docs/design.md](../design.md) remains authoritative. Read §1.2, §4.4, §5.10, §7, §8.1, the subsection titled **8.2 Fellner-Schall Method (Fast REML for bam)**, and §18.1–§18.2. There are two §8.2 headings; use the title as well as the number.

Before public EFS ships, its implementation PR must reconcile the post-v1 scope, optimizer/API contract, and objective-monotonicity policy with the architecture and repository instructions. This plan does not silently override those rules. The old §8.2 sketch is not an executable specification: it conflates EFS with fREML, uses a singleton-rank shortcut that is not general for coupled penalties, omits scale handling, and disagrees between its docstring and code about a lambda factor. Replace that sketch with an accurate summary and a link here when the architectural change is approved. Do not implement it verbatim.

## 2. References and source provenance

### 2.1 Theory reference

Wood, S. N. and Fasiolo, M. (2017), *A generalized Fellner–Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models*, Biometrics 73(4), 1071–1081, [doi:10.1111/biom.12666][paper]. Use the [published article][paper-full] for equation numbering: §2 / equation (3) for Gaussian updates, §3 for convergence assumptions, and §4 / equation (8) for general likelihoods. The [arXiv preprint][preprint] is supplementary; its numbering differs. The publisher records a December 2017 correction, so prefer the corrected published text for theoretical claims.

The general-likelihood development neglects a likelihood-curvature dependence; exact trace evaluation does not remove that approximation. Gaussian ascent requires suitable step-length control, and the general update does not promise exact finite-sample REML maximization. These distinctions govern the validation contract below. [Wood and Fasiolo, §§2–4][paper-full].

Historical provenance: [Fellner (1986), Robust Estimation of Variance Components][fellner], and [Schall (1991), Estimation in generalized linear models with random effects][schall]. The generalized paper is the main implementation reference because JaxGAM supports several penalties acting on the same coefficient block.

### 2.2 Executable reference

`echo $MGCV_SOURCE` resolves to `/Users/shanecatts/Documents/GitHub/mgcv`. Its checkout is mgcv 1.9-4 at `1b6a4c8374612da27e36420b4459e93acb183f2d`; the test oracle is R 4.5.2 + mgcv 1.9-3, enforced by [RBridge.check_versions](../../tests/r_bridge.py). Unlike the parent document's 1.9-4 links, **all mgcv implementation links in this sub-plan use the pinned 1.9-3 commit** `fb7e8e718377513e78ba6c6bf7e60757fc6a32a9`.

Read without changing the clone checkout:

```sh
echo $MGCV_SOURCE
git -C "$MGCV_SOURCE" rev-parse HEAD 1.9-3
git -C "$MGCV_SOURCE" show 1.9-3:R/gam.fit4.r
git -C "$MGCV_SOURCE" show 1.9-3:R/gam.fit3.r
git -C "$MGCV_SOURCE" show 1.9-3:R/efam.r
git -C "$MGCV_SOURCE" diff 1.9-3 HEAD -- R/gam.fit4.r R/gam.fit3.r R/efam.r
```

The inspected `efsudr` body is unchanged between these revisions, but that does not establish that every callee is unchanged. For example, the 1.9-4 `gam.fit4` adds an `ml.pen` score attribute. Record source version and relevant branch in each implementing PR and generated fixture.

| Responsibility | Pinned mgcv code to read | What to reuse / inspect |
|---|---|---|
| Public dispatch | [`R/mgcv.r::gam.outer`][r-outer] | `optimizer="efs"` dispatch, `gam.fit3/4` versus general-family `gam.fit5`, no-free-sp handling |
| Initial state | [`initial.spg`][r-initial-spg], [`get.null.coef`][r-null], [`estimate.gam`][r-estimate] | Family-dependent initial working weights, initial-sp calculation, initial scale, and packing before `efsudr` |
| Outer algorithm | [`R/gam.fit4.r::efsudr`][r-efs] | Range-space products, trace/quadratic update, scale, caps, extension/contraction, stopping |
| Penalty determinant derivative | [`R/gam.fit3.r::gam.reparam`][r-reparam] | `det1` is derivative with respect to **log** smoothing parameters; `deriv=1` suffices |
| Penalty space | [`totalPenaltySpace`][r-penalty-space] | Common range/null space and penalty roots; compare existing JaxGAM ranks before changing thresholds |
| Regular-family fit | [`gam.fit3`][r-fit3] | `scoreType="EFS"` requests penalty-det derivatives but suppresses full REML derivatives; Fletcher scale and score scale are separate |
| Extended-family fit | [`gam.fit4`][r-fit4], [`dDeta`][r-ddeta] | Theta update inside PIRLS, deviance refresh, validity/convergence, observed information |
| Covariance root | [`src/gdi.c::gdi1`][r-gdi1], [`gdi2`][r-gdi2] | Returned `rV` uses expected/Fisher information after coefficient/score calculations; rank dropping and pivots |
| Conditional theta update | [`R/efam.r::estimate.theta`][r-theta] | Conditional likelihood Newton, safeguarded Hessian, bounded step, line search, stopping |
| NB numerics | [`R/efam.r::nb`][r-nb] | `Dd`, `ls`, `getTheta`, `putTheta`, initialization; log-theta derivative conventions |
| Defaults | [`R/mgcv.r::gam.control`][r-control] | `epsilon=1e-7`, `maxit=200`, `efs.lspmax=15`, `efs.tol=0.1`, Fletcher scale |
| Explicit non-target | [`R/gam.fit4.r::efsud`][r-efs-general] | This is the `gam.fit5` route, not the first port for JaxGAM's existing families |

Theory-to-code traceability belongs in tests, not just a bibliography:

| Paper anchor | mgcv realization | Proposed test responsibility |
|---|---|---|
| §2, equation (3) | `efsudr`'s `ldetS1`, `trVS`, `bSb`, and `phi` | Gaussian update algebra, singleton reduction, coupled penalties |
| §2, Theorem 1 / Remark 1 | Penalty range handling and nonnegative numerator | Null-space/rank fixtures and information-matrix assumptions |
| §3 | Outer multiplier and stopping policy | Branch fixtures; no unconditional Newton-speed or convergence guarantee |
| §4, equation (8) | `gam.fit3/4(scoreType="EFS")` and returned covariance root | Family/curvature conventions; EFS-versus-Newton differences reported separately |

## 3. Current repository integration points

| Current implementation | Required use / change |
|---|---|
| [`api.GAM.fit`, `_check_scope_guards`](../../jaxgam/api.py) | `optimizer` is an accepted keyword but only `None`/`"newton"` is allowed; the value is not currently stored for dispatch. Add validation, storage, dispatch, and tests together. `method` remains `"REML"`. |
| [`api._fit_fixed_sp`](../../jaxgam/api.py) | Natural-scale strictly positive fixed sp. Unknown-scale **or** estimated-theta families currently route to pinned-lambda Newton. Do not call this as the EFS trial-fit provider: it can run another outer optimization. |
| [`FittingData.from_setup`](../../jaxgam/fitting/data.py) | Reuse `X`, `S_list`, `max_y`, `repara_D`, rank information, and existing singleton/multi-penalty metadata. Its current `log_lambda_init` uses prior weights, not `initial.spg`'s family-dependent working weights; add an EFS initialization adapter without a second basis/constraint setup. |
| [`initialize_beta`](../../jaxgam/fitting/initialization.py) | Initialize once, then warm-start from the accepted state. R-parity controller fixtures can supply a matched initial state independently of public initialization. |
| [`pirls_loop`, `PIRLSResult`](../../jaxgam/fitting/pirls.py) | Existing pure JAX inner solve; output already includes observed/score and Fisher matrices/factors. Preserve backtracking and stationarity safeguards. |
| [`block_log_det_S`](../../jaxgam/jax_utils.py) | Compute only its first log-lambda derivative for EFS, or equivalent block solves. Do not build a second pseudo-determinant engine. |
| [`reml_criterion`, `reml_criterion_joint`, `_criterion_core`](../../jaxgam/fitting/reml.py) | Reuse scalar score evaluation, saturated likelihood, `Mp` correction, and rank convention. `_criterion_core` alone omits the outer REML null-space correction. |
| [`estimate_edf`, `fletcher_scale`, `estimate_scale`](../../jaxgam/fitting/reml.py) | Retain Fisher EDF and reported-scale behavior. Do not confuse optimized/incoming score phi with reported phi. |
| [`NewtonOptimizer`, `NewtonResult`](../../jaxgam/fitting/newton.py) | Regression reference and common result fields, not the EFS controller. Do not call `_diff_score`'s gradient/Hessian on the production EFS path. |
| [`NegativeBinomial`](../../jaxgam/families/negative_binomial.py) | Reuse `deviance_fn`, `saturated_loglik_theta`, explicit log-theta arguments, stable count-tail derivatives, and fixed-theta cache safeguards. |
| [`GAMResults._from_fit`](../../jaxgam/results.py) | Preserve coefficient back-transform, Fisher covariance, EDF, full/inference materialization, and family snapshot. This still constructs dense covariance in stage one. |
| [`RBridge`](../../tests/r_bridge.py) | `fit_gam` has no optimizer/control/initial-state arguments today. Add a tested EFS bridge entry point; do not pretend current calls select EFS. |

CPU fitting still uses JAX. Proposed kernels belong in `fitting/`; new host control belongs in `execution/`, which does not exist at the audited commit. Keep NumPy-only setup/prediction and pure JIT kernels separate even though the current Newton driver already contains Python orchestration.

## 4. Numerical contract and exact dense statistics

### 4.1 Coordinates, scaling, and matrix roles

Use one frozen fitting coordinate system throughout an iteration. In the definitions below, beta is `PIRLSResult.coefficients` and S_j is the matching **unweighted** penalty in `FittingData.S_list`, after the existing fitting reparameterization. Do not pair public/back-transformed beta with fitting-space S_j. Term labels are not identifiers; use positional penalty indices and `TermBlock` metadata.

Define:

```text
rho_j = log(lambda_j)
S_lambda = sum_j exp(rho_j) S_j

G_F = Fisher-weighted cross-product, with the existing unit-deviance scale convention
H_F = G_F + S_lambda
C_F = H_F^{-1}                    # inverse action, not a required stored inverse

d_j = partial_{rho_j} log|S_lambda|_+
    = lambda_j trace(S_lambda^+ S_j), on the fixed penalty range
t_j = trace(C_F S_j)
q_j = beta.T S_j beta
```

For the mgcv target, t_j corresponds to `trVS`, q_j to `bSb`, and d_j to `fit$ldetS1`. The comment above `efsudr` abbreviates the derivative convention; its actual `exp(-lsp)` conversion and `gam.reparam` establish that `ldetS1` is a **log-lambda derivative**, not directly `trace(S_lambda^+ S_j)`.

The Fisher factor supplies the EFS covariance trace because that is what mgcv returns in `rV`: inspect the [gdi1 Fisher postprocessing][r-fisher1] and [gdi2 Fisher postprocessing][r-fisher2]. The score's log determinant retains the repository's observed-information convention. Do not replace every factor with `L_fisher` or with `L` indiscriminately. Canonical first-stage fixtures avoid many differences, but non-canonical and NB tests must expose them explicitly.

Neither these definitions nor the inverse notation authorize a global inverse allocation. They assume an identifiable positive-definite system and consistent rank treatment. Start there; add rank-deficient support only after projection/pivot parity tests. Existing JaxGAM jitter and clipping are not R's rank-revealing QR and must remain visible in the comparison.

### 4.2 Penalty derivative and coupled blocks

For a singleton block, the current log determinant is `rank * rho + constant`, hence d_j equals that stored rank. For a multi-penalty block, use the existing common-range projected penalties A_j and form the scaled block sum B = sum_j lambda_j A_j. Compute d_j through the current `block_log_det_S` gradient or the equivalent `lambda_j * trace(solve(B, A_j))`, preserving its scaling and static range metadata.

Do not substitute `rank(S_j)` for d_j on a tensor block. Several penalties can occupy the same coefficients; their contributions depend on every lambda in the block. Do not re-detect each S_j's rank at every smoothing iteration, silently change the fixed range, or differentiate through data-dependent rank selection. In particular, an invalid/singular block must not acquire a plausible zero derivative from `block_log_det_S`'s existing negative-sign sentinel.

First implementation: JIT the gradient of the **penalty-only** log determinant, not the full fitted-model REML score. Check singleton outputs analytically and coupled outputs against `gam.reparam(..., deriv=1)$det1`. A local-solve optimization can follow with the same tests. No m-by-m Hessian is needed.

### 4.3 Quadratic forms and covariance traces

For a penalty root B_j with S_j = B_j B_j.T in the same coordinates:

```text
q_j = ||B_j.T beta||²
t_j = ||solve_triangular(L_F, B_j, lower=True)||_F²
```

This directly adapts `efsudr`'s root products, without constructing covariance. Reuse prepared penalty/range decompositions; if new roots are needed, compute them once on CPU using the existing rank tolerances, then transfer them. Do not eigendecompose every global penalty per EFS iteration.

With parent PR 2, store roots locally and scatter bounded groups of right-hand sides into fitting coordinates. Dense factor solves still mix coefficient blocks: using only a diagonal block of H_F in place of the full solve is incorrect. For identity/diagonal penalties, use equivalent selected-column or diagonal contractions, with the same full-system semantics. Bound RHS workspace; do not stack roots or covariance derivatives into an m-by-p-by-p array.

`PIRLSResult.L_fisher` factors a stabilized system. Verify that its inverse action agrees with the R covariance-root contraction over the declared support. Do not silently reuse `_criterion_core`'s differently scaled/jittered score factor. Record failures at clipping/rank boundaries rather than claim that equal formulas imply equal finite-precision behavior.

R also forms `edf = p - sum_j(lambda_j * t_j)`. This identity is useful on a full-rank, unregularized system. JaxGAM's jitter and structural rank deficit can change its relation to `estimate_edf(G_F, L_fisher)`. Keep the existing EDF calculation for public reporting until the alternative is validated; do not replace p with an invented effective rank inside the R algorithm. Initial support can reject structural rank-deficit cases.

### 4.4 Update and degenerate branches

The pinned `efsudr` arithmetic is:

```text
a_j = max(0, exp(-rho_j) * d_j - t_j)
ratio_j = phi_update * a_j / max(0, q_j)

if a_j == 0 and q_j == 0: ratio_j = 1
if ratio_j is not finite: ratio_j = 1e6

rho_trial_j = min(rho_j + multiplier * log(ratio_j), log_lambda_max)
```

Here phi_update is the scale chosen by §6. A fixed known scale is used once; t_j is the **unscaled** covariance trace. Multiplying t_j by reported phi and then multiplying the whole ratio by phi again would double-count scale.

Test the branch order. Positive numerator with zero denominator takes the finite replacement ratio; zero over zero means no change. A zero numerator with positive denominator instead gives ratio zero and a log-lambda proposal of negative infinity: R's outer code has no Newton-style lower cap. Do not add `1e-10` denominators or copy Newton's `[-40, 40]` clipping and call it the same update.

The finite-positive-ratio domain is the first production contract. Outside it, the initial public adapter must return a named non-convergence status with the last valid fit, or raise when no valid fit exists, rather than feed non-finite parameters into the current log-determinant kernels. The raw update unit test still records R's zero-ratio behavior. Supporting an actual zero-lambda boundary or imposing a finite lower bound needs a separately documented range/limit policy and matched R fixtures. A non-finite coefficient fit is not covered by R's non-finite **ratio** replacement.

An algebraically equivalent scaled numerator, `max(0, d_j - lambda_j*t_j)`, divided by `lambda_j*q_j`, can improve some extreme-scale calculations. Treat any reordering as a tested numerical change, especially around cancellation and the clamp. Compare raw d_j, t_j, q_j and numerator before clamping so cancellation is not mistaken for evidence that a term is absent.

## 5. Dense outer controller

### 5.1 State and initialization

Use an immutable accepted/trial state containing rho, explicit log_theta if applicable, beta/PIRLS output, d/t/q statistics, score, score_phi, update_phi, reported_phi, inner convergence status, and stabilization metadata. Every cached quantity must identify the same fitting coordinates and the exact parameter state that produced it. Unknown-scale R fits require the multiple phi fields described in §6; do not force them into one ambiguous `scale` slot.

For default initialization, adapt [`initial.spg`][r-initial-spg]'s conventional-regression branch around the existing family initialization and `FittingData._initial_sp` machinery. R initializes mu, obtains family working weights, then calls `initial.sp` on the weighted design. Current `fd.log_lambda_init` instead uses prior weights directly; those starts are not interchangeable for non-Gaussian families.

Preserve original, pre-`repara_D` coordinates for this initialization. For regular families the initial weights are `wt * mu_eta(eta_start)**2 / variance(mu_start)`; for NB read the observed `Dd$Dmu2` branch and its expected-curvature fallback if any initial weight is negative. Pass working weights, not their square roots, into the current `_initial_sp` helper, which applies the square root itself. Reuse its existing penalty normalization conventions and validate against the local-block R operation; any padding/threshold mismatch belongs in the initialization parity tests. Add an EFS-specific fitting-boundary initialization policy without changing Newton's defaults or copying basis construction. Keep NumPy setup and any JAX family-derivative kernels on their respective sides of the phase boundary.

mgcv's `efsudr` then adds **2.5** to initial log smoothing parameters once, before the first coefficient fit. Do not add it on every restart or accepted iteration. Initialize `multiplier=1`. A layer test must also permit a matched initial state so initialization differences cannot hide controller bugs; use R's `in.out` or an internal fit helper when appropriate. Current public `sp` means fixed values, not starting values. Match unknown-scale initialization through §6.2 rather than borrow Newton's Fletcher-initialized log_phi blindly.

The R initial fit does not pass exactly the same start arguments as all later trials: initial `efsudr` omits its `mustart`/`etastart` parameters in the `gam.fit3` call, whereas subsequent trials pass `mustart`. Preserve the pinned behavior in reference fixtures; do not invent public start semantics during this port.

### 5.2 Accepted and trial steps

Adapt the actual `efsudr` loop, using a fit provider that reconverges coefficients at each proposed rho. Do not run only one working-model PIRLS step, call `bam`'s performance iteration, or invoke `newton_optimize` inside each trial.

1. From the accepted fit, compute the update ratio and candidate rho. Save the old score, multiplier, and coefficient start.
2. Fit the candidate from that saved coefficient start, following the scale/theta state rules below. A trial must produce finite valid coefficients and usable factors; otherwise it is a numerical failure, not a successful EFS step.
3. If candidate score is no larger than old score, accept it. When the original `max.step < 0.05`, also try a step with **twice** the multiplier from the same old state/start; accept the extension only if it strictly improves on the first candidate. Double the persistent multiplier only when that extension wins.
4. If the candidate worsens the score, halve the multiplier and refit while the score is worse **and multiplier > 1**. The mgcv policy accepts the resulting step at multiplier 1 even if it still worsens the score. It does not contract below 1 to enforce REML monotonicity.
5. Append the selected fit's score to the history and evaluate stopping. Start the next outer iteration from that selected fit, not the best score seen anywhere in the run.

Preserve the R timing of `max.step`: it is computed for the original capped proposal before trial nuisance-parameter updates and is not recomputed after extension/contraction. Likewise, alternative trials are formed from the old full parameter vector; do not seed them with a rejected trial's theta or phi. A cleaner-looking state update can otherwise change control flow. Record both R's stopping statistic and the actual accepted displacement for diagnostics.

Default constants are distinct from Newton's:

| Item | Pinned EFS value / rule |
|---|---|
| Initial rho shift | +2.5 once |
| Outer iteration limit | 200, hard-coded in `efsudr` |
| Maximum log lambda | `gam.control()$efs.lspmax = 15` |
| Small proposal threshold | 0.05 in max absolute log-parameter displacement |
| Score stopping tolerance | `gam.control()$efs.tol = 0.1`, absolute |
| Score-history stopping | After more than 3 iterations: small proposal and all 3 successive differences among the last 4 scores below tolerance |
| Alternate stopping | After the first iteration: `abs(old_dev - new_dev) < 100 * epsilon * abs(new_dev)` |
| Inner tolerance / limit | mgcv defaults `epsilon=1e-7`, `maxit=200`; current `pirls_loop` defaults are `tol=1e-7`, `max_iter=100`, so pass the EFS profile explicitly; Newton uses a tighter inner tolerance |

R's `control$eps` in the deviance test partially matches `epsilon`; it is not machine epsilon. R's `fit$dev` and `fit$scale` likewise rely on partial field matching. Use explicit field names in Python and in the bridge extraction. At iteration 200 R labels the result as an iteration-limit outcome; record boundary behavior in tests.

### 5.3 Monotonicity, failure, and convergence policy

Recommended reference policy: **mgcv-style EFS**, including finite score-increasing outer steps. This is intentionally not the existing Newton acceptance policy. Keep every Newton regression assertion unchanged. The current `tests/test_fitting/test_newton.py::test_reml_monotonicity` compares final and initial scores; it does not expose a complete accepted-step history. Add an actual EFS trace for branch tests rather than re-label that test as an EFS trajectory guarantee.

Repository instructions currently state objective monotonicity as a hard gate. Before public activation, obtain the architectural/instruction update that scopes this correctly: coefficient backtracking remains safeguarded at fixed nuisance parameters; Newton retains its existing score contract; mgcv EFS validates its own step rules and convergence status. Do not merely delete or relax an assertion. If strict outer monotonicity remains mandatory, expose no mgcv-faithful public EFS until a separately named monotone variant is designed and validated; do not claim that variant is an exact port.

The stopping heuristic is not proof of a small full REML gradient. Record the reason (`score_window`, `deviance_change`, `iteration_limit`, `inner_failure`, `invalid_update`, etc.), the final update residual, and inner stationarity diagnostics. A public result must not claim convergence after a failed inner fit, invalid factor, non-finite state, or exhausted solver. Do not fabricate a zero REML gradient to satisfy `NewtonResult`'s shape. Full score derivatives are optional small-model diagnostics only, not required EFS work.

No automatic Newton fallback. If a later opt-in fallback is added, record both strategies and the reason; otherwise performance and numerical behavior are impossible to attribute.

## 6. Family, scale, and theta behavior

### 6.1 Known-scale regular families

Poisson and binomial use phi=1 under the existing public contract. Reuse `pirls_loop`, then evaluate the scalar REML criterion and Fisher statistics. Scale cannot accidentally become an additional EFS smoothing parameter. Preserve offsets, prior-weight semantics, binomial initialization, and coefficient-map transformations.

Initially exclude zero/extreme-weight clipping edges; add them through explicit parity tests. Clipped solve weights versus original deviance weights are an existing numerical convention, not a reason to redefine the likelihood in the EFS driver.

### 6.2 Gaussian and Gamma

The current Newton implementation jointly optimizes log_phi when scale is unknown and penalties are present, but reports a Fletcher scale. The EFS port must follow `gam.fit3(scoreType="EFS")` and `efsudr`, not retain a hidden joint Newton optimization for phi.

For the initial incoming phi, [`estimate.gam`][r-estimate] uses `get.null.coef`'s `null.scale / 10`, unless `in.out$scale` is supplied. That null scale is the family deviance at the initialized response's unweighted mean, divided by the original row count, with prior weights applied in the deviance. It is not the public null-deviance calculation or the existing Newton Fletcher initialization. Reuse family initialization/deviance primitives and retain the resulting scalar at the fitting boundary. Test offsets, weight conventions, and degenerate/nonpositive null scale explicitly; the finite-positive-scale contract must not silently turn log(0) into a valid optimizer state.

For a regular unknown-scale trial, the incoming full log-parameter vector supplies phi used in the REML score. The coefficient fit also produces a new `scale.est`; `efsudr` uses that estimate in subsequent updates and copies its log into the next parameter state. The score was already computed using the incoming phi. Keep `score_phi`, `update_phi`, and `reported_phi` explicit and capture this timing in R fixtures. Re-evaluating every cached score at newly reported phi changes the reference controller, even if the resulting bookkeeping seems more intuitive.

Use existing `fletcher_scale`/`estimate_scale` for the reported value where they match the pinned branch, preserving positive-weight likelihood counts, log-weight terms, total Fisher EDF, and observation-count conventions. Optional per-term EDF reporting does not remove total EDF needed for scale. Do not replace phi with `RSS/(n-p)` or reuse a Gaussian-only formula for Gamma. A Gaussian QR provider must retain the original n, not substitute its reduced row count.

`GAMResults.score` should remain the scalar criterion supplied by the chosen optimizer, with diagnostics recording its scale convention. Preserve the public reported-scale/covariance contract through `_from_fit`. The R bridge already extracts `reml_scale` separately from `scale`; extend this to intermediate EFS state, not just final output.

### 6.3 Fixed-theta NB

NB has known phi=1, independently of whether theta is fixed or estimated. Start with `NegativeBinomial(theta=..., fixed=True)` and the existing explicit-theta coefficient kernels. Keep the observed information used for fitting/scoring distinct from Fisher information used for `trVS`, EDF, and covariance. Fixed theta alone is not an SPD guarantee for arbitrary links.

Preserve fixed-theta family cache behavior; do not mutate a static family object between compiled calls and assume JAX will notice. Initial support is NB/log in the validated finite/SPD regime; other links require separate tests.

### 6.4 Estimated-theta NB

This is not a scalar application of the smoothing-parameter ratio. In the pinned source, `gam.fit4(scoreType="EFS")` calls `estimate.theta` **inside PIRLS**, at fixed current mu, then refreshes working quantities and the penalized-deviance baseline. JaxGAM currently estimates theta through joint outer Newton, so this is a real integration change.

Port the needed one-parameter NB branch of `R/efam.r::estimate.theta`. Despite stale source-map prose, that routine is a **conditional Newton algorithm**, not another Fellner–Schall trace update. Its objective is:

```text
NLL(log_theta | mu) = deviance(y, mu, log_theta, wt) / (2 * phi)
                     - saturated_loglik(y, wt, log_theta, phi)
```

Use `NegativeBinomial.deviance_fn` and `saturated_loglik_theta` with explicit dynamic log_theta, differentiating only this conditional scalar objective at fixed mu/eta. Include saturated-likelihood derivatives and the log-theta chain rule. Reuse the existing stable NB primitives and `max_y`; do not introduce a new gamma-function cancellation workaround here. The paper's removal of high-order smoothing derivatives does not eliminate the first/second derivatives needed by conditional theta Newton.

Match the R branch: tolerance `1e-7`, up to 100 conditional iterations, active-gradient threshold `tol * (abs(NLL)+1)`, absolute-eigenvalue correction for nonpositive Hessian with relative floor `1e-5`, max absolute log-theta step 4, and halving against `eps_machine^0.75 * abs(NLL)` with the existing 25-halving/unchanged-parameter failure rule. Test pathological zero curvature rather than rely on a zero relative floor to fix it. Propagate conditional failure explicitly.

Add an EFS-specific theta update hook/state to the compiled PIRLS path, or a shared pure inner-iteration kernel plus a dedicated compiled EFS loop. Do not copy all of PIRLS or replace R's in-loop theta updates with a host loop that fully converges beta and then updates theta; that is a different algorithm. Preserve default Newton/fixed-theta compilation and numerical behavior.

September 8 review checkpoint: the internal EFS5e stage at `16b29d5` now preserves absent-start provenance and R's per-row default/reset mustart eta, independently checked at the first iteration and selected outer fit. A retained first proposal requiring nonfinite/domain recovery still returns the explicit `retained_start_invalid_trial` outcome. Complete the corresponding pinned `gam.fit4` sequential recovery before claiming whole EFS5: earlier validity recovery uses the retained coefficient/eta anchor, while later first-divergence handling resets to the null anchor. A shared null-origin halving result must not bypass this distinction. Keep the compiled beta/theta-loop status separate from conditional-theta Newton status in the controller.

The carried state must include log_theta. After theta changes, recompute deviance/working quantities before the next convergence or backtracking decision so old and new objectives are not compared at different theta accidentally. R's exact update/convergence ordering, including the pre/post-theta deviance values, needs fixtures; any stronger consistency check is a named departure. Require a consistent finite final fit at the returned theta before declaring public convergence.

R packs family parameters **before** smoothing parameters, while current JaxGAM Newton uses `[log_lambda, <log_theta>, <log_phi>]`. Prefer named fields in EFS state; bridge packing is explicit. Do not return R's full `exp(lsp)` as JaxGAM `smoothing_params`: that public array contains only the m smoothing parameters.

Copy the family per fit and commit the selected final theta to that copy for Phase 3. Rejected trials must not mutate accepted family state or the registry singleton. Existing result-family snapshot and repeated-fit tests remain gates.

## 7. Proposed implementation and API

All names below are proposed, not existing interfaces. Introduce the smallest pieces needed by each PR; if a parent-plan module has landed, extend it instead of creating a parallel abstraction.

| Location | Responsibility | Tests |
|---|---|---|
| `jaxgam/fitting/efs.py` | Pure statistics/update/convergence kernels; EFS state pytrees as needed | `tests/test_fitting/test_efs.py` |
| `jaxgam/execution/efs.py` | Host accepted/trial controller and dense fit-provider adapter | `tests/test_execution/test_efs.py` |
| `jaxgam/fitting/efs_theta.py` | Later conditional NB theta kernel | `tests/test_fitting/test_efs_theta.py` |
| Existing `fitting/pirls.py` | Shared inner step and opt-in EFS theta integration | Extend `tests/test_fitting/test_pirls.py`; JIT default and EFS paths |
| Existing `jax_utils.py` / `fitting/data.py` | Reuse penalty determinant metadata; add a first-derivative helper or prepared local roots only if needed | Extend existing utility/data tests |
| Existing `api.py`, `results.py` | Dispatch and optimizer-neutral consumed-result contract | Existing API/result-mode tests plus EFS cases |
| Existing `tests/r_bridge.py` | Explicit EFS end-to-end and intermediate-state oracle | Extend `tests/test_r_bridge.py` |

Pure kernels take explicit arrays/static structure, run in float64, and compile under `jax.jit`. Array-dependent branches/loops use `jax.lax.cond`/`while_loop`; debug mode uses JAX debug facilities. Host code may select trials and call compiled kernels, following the existing phase boundary. No Python I/O iterator or family mutation inside a JIT trace. Measure compilation reuse; do not generate a new jitted closure per outer step.

### Public selection and controls

Proposed minimum call after the public release gate:

```python
# Proposed API; optimizer="efs" is rejected by the audited repository.
fit = GAM("y ~ s(x, bs='cr')", family="poisson",
          method="REML", optimizer="efs").fit(data, result="inference")
```

Keep `optimizer="newton"` and omitted optimizer byte-for-byte compatible where current tests require it. Keep backend `"jax"`, method `"REML"`, and result modes `"full"`/`"inference"` unchanged. Set an explicit strategy such as `lambda_strategy="efs_reml"`; this records optimizer selection, not a promise of exact Newton REML equivalence. Do not call the method `"fREML"`.

Use an internal immutable `EFSControl` with explicit fields for outer limit, maximum log lambda, score tolerance, PIRLS tolerance/limit, and debug/history policy. Its reference defaults are in §5.2. Initial shift, ratio conventions, and extension policy are algorithm constants, not arbitrary tuning options. A public control entry point must be added with validated types, accepted-keyword handling, exports, and tests in the same PR. If parent `FitControl` is present by then, compose this control there; otherwise a narrowly scoped `efs_control` keyword is sufficient. Do not introduce both competing routes or assume `control=` works today.

Fixed `sp` keeps precedence and its current validation/dispersion route; report `lambda_strategy="fixed"`, not EFS, when lambda was not estimated. No mixed positive/fixed and negative/estimated sp convention is added. With zero penalties, bypass EFS rather than take maxima over empty arrays. Exercise the current parametric/estimated-theta shortcut separately; do not assume its behavior is equivalent to R merely because it bypasses smoothing.

### Results and diagnostics

`GAMResults._from_fit` currently accepts `NewtonResult` but consumes common fit fields, not a full score Hessian. Add an optimizer-neutral internal protocol/result adapter containing the fields it actually uses. Keep existing `NewtonResult` behavior; an EFS update residual must not masquerade as its REML gradient. Compose with parent `fitting/state.py` if available.

Store compact diagnostics: selected optimizer, source/reference profile, exact-versus-stochastic trace method, step policy, stop reason, outer/inner/theta iteration counts, accepted scores and score_phi, multiplier, maximum proposed/accepted movement, numerator-clamp and cap events, and invalid/stabilized solve indicators. Keep history bounded/configurable; do not retain all n-row fits or p-by-p factors per trial. No `gradient=0` sentinel suggesting stationarity.

Dense EFS initially uses existing result postprocessing, including full Vp and per-term EDF/EDF1. Preserve full-versus-inference prediction byte equality from `tests/test_result_mode.py`. A prediction-only/operator result can later use parent PR 6; new covariance approximations do not belong in the dense EFS milestone.

## 8. Validation and acceptance

### 8.1 Three distinct comparisons

1. **Algebra / matched-state parity:** hold basis, coefficients, rho, theta, phi, weights, and coordinate transforms fixed. Compare d_j, t_j, q_j, ratio, and proposed step to the corresponding R operation and independent small NumPy calculations. This isolates EFS correctness from optimizer initialization and trajectory.
2. **Pinned EFS fit parity:** compare to `mgcv::gam(..., method="REML", optimizer="efs")` using matched formulas, inputs, controls, and family parameterization. Capture the EFS profile and score-scale timing. This is the acceptance oracle for the new algorithm.
3. **Existing Newton reference:** same frozen model, evaluate differences in criterion, coefficients/predictions, deviance, scale, theta, EDF, and uncertainty. Gaussian well-conditioned cases provide a strong agreement gate; general-family EFS is not required to land at the exact Newton stationary point. Report method differences separately from implementation error. Do not replace an EFS oracle with default R `gam` or `bam(fREML)`.

Use [tests/tolerances.py](../../tests/tolerances.py): STRICT `rtol=1e-10, atol=1e-12`; MODERATE `rtol=1e-4, atol=1e-6`. New equivalent-operation/model R gates must use STRICT or MODERATE under repository instructions. If mgcv's default stopping produces cross-language branch differences beyond those tolerances, first separate matched-state algebra, controlled/tightened EFS fixtures, and default-profile compatibility. Do not loosen tolerances, cherry-pick a different reference optimizer, or claim default-profile parity from tightened-control results. Unresolved supported-case discrepancies block that capability.

### 8.2 Bridge and reproducible fixtures

Add a dedicated `RBridge.fit_efs` and narrowly scoped EFS diagnostic helper rather than a generic arbitrary-R-evaluation API. Support both existing rpy2 and subprocess modes, or explicitly mark a layer helper rpy2-only with a clear skip reason; final EFS parity must work in the pinned CI environment. Existing `fit_gam` calls/defaults remain unchanged. Extend structured inputs for weights, offsets, link, fixed/estimated theta, EFS controls, and matched initial states as needed; use existing data conversion and extraction code.

Record full-fit outputs plus intermediate state: `ldetS1`, covariance-root contractions, `bSb`, input/output rho/theta/phi, initial shift, multiplier, score and score scale, deviance, selected branch, stop reason, and fit status. Use isolated R instrumentation or a locally scoped function environment for controller fixtures; do not edit the installed mgcv package or the source checkout. Restore temporary tracing/mocks on exit. Pin fixture-generation code as well as serialized results.

A saved fixture must include R/mgcv version, source commit, formula, seed/data hash, weights/offsets, basis dimensions, fixed versus estimated parameters, and control values. Use real `gam.reparam`/`gam.fit3/4` outputs for algebra; a Python re-expression of the EFS formula is not an independent R oracle. Scripted fit-provider sequences are additionally useful to force every controller branch without expensive models.

### 8.3 Test ownership and matrix

| Layer | Required cases / assertions |
|---|---|
| Statistics | Singleton penalty; two noncommuting PSD penalties on one block; actual `te`/`ti`; intercept/null space; identity penalty; penalty permutation; rescaling S_j with inverse lambda rescaling; fitting/public coordinate roundtrip |
| Range and conditioning | Common null space; structural rank deficit; large lambda ratios; near-zero penalty quadratic; near-singular Fisher system; jitter-sensitive cases; no recomputed rank discontinuities |
| Raw update | Positive ratio; zero/zero; positive/zero; zero/positive; non-finite ratio replacement; upper cap; initial +2.5 shift; finite arithmetic versus stable reordered variant |
| Controller / initialization | Family working-weight starts versus prior-weight starts; matched R `in.out`; initial null scale; ordinary improvement; winning/losing extension; contraction back to 1; accepted finite worsening; original small-step statistic timing; both stops; iteration 200; inner failure; no rejected-state leakage |
| Scale | Known phi; Gaussian and Gamma Fletcher reporting; different input versus output phi; final `score` versus `reml_scale`; original row/positive-weight counts; near-perfect Gaussian fits |
| NB | Fixed and estimated theta; conditional derivatives and Newton steps; small/large theta and count tails; theta near Poisson limit; valid-domain failures; family isolation; parameter ordering; post-theta objective consistency |
| Public compatibility | Omitted/newton optimizer unchanged; explicit EFS dispatch; unsupported method/link errors; fixed-sp precedence; zero-penalty bypass; full/inference outputs; pickle/frozen result behavior where already tested |
| Compilation / cost | Statistics/update kernels and NB EFS inner loop compile and run; warm reuse across rho values; no full REML Hessian/custom-JVP fit derivative invoked; bounded RHS/history workspace |

Layer-specific tests live with the implementations above. Broad final `GAMResults` comparisons and hard-gate invariants belong in `tests/test_validation_matrix.py`; add representative optimizer-aware cells rather than duplicate expensive fits in every EFS test file. Reuse `tests.helpers._AssertCollector` for named assertions sharing one fit. Preserve current Newton cases and existing LOOSE legacy assertions, but those alone do not satisfy new EFS parity gates.

Require finite converged state, valid family parameters, deviance non-negativity, penalty symmetry/PSD, identifiable supported systems, Fisher covariance symmetry/PSD, appropriate EDF bounds, and prediction-on-training-data roundtrip. Signed observed curvature needs its existing algorithm-specific handling; do not force a PSD assertion by silently swapping matrices. EFS outer monotonicity is governed by the approved policy in §5.3, not an unannounced exception.

Every new module gets a corresponding test file and >80% coverage. Phase 2 changes require explicit JIT compilation-and-execution tests. Relevant existing suites include `tests/test_fitting/test_pirls.py`, `test_reml.py`, `test_newton.py`, `test_nb_custom_jvp.py`, `tests/test_api/test_gam.py`, and `tests/test_result_mode.py`.

Run through existing Makefile tasks:

```sh
make lint
make test-file FILE=tests/test_fitting/test_efs.py
make test-file FILE=tests/test_execution/test_efs.py
make test
make test-cov
```

The new file commands apply after those tests exist. `make test`/`test-file`/`test-cov` use Docker with the pinned R oracle. `make test-local` is useful for development, but R auto-skips are not parity evidence. This documentation change does not require new Makefile targets; add targets only when new reusable benchmark/test capabilities land.

## 9. Later operator and stochastic EFS

First adapt the already-tested controller to the parent PR 12 solve interface and validate exact contractions on moderate p. Keep all quantities' matrix/scale conventions unchanged. Do not conflate a trace-provider change with a new theta algorithm, new basis, or new step policy.

The first scalable option should retain exact d_j from cheap local penalty blocks wherever possible and approximate only the expensive covariance traces. With shared random vectors z_k satisfying E[z_k z_k.T]=I, solve H_F u_k = z_k once per probe and contract `z_k.T @ S_j @ u_k` for each penalty. This shares solves across penalties. Individual samples can be negative; clamping noisy numerator estimates introduces bias. Require accuracy checks before applying the deterministic clamp, especially when d_j/lambda_j and t_j nearly cancel.

Reuse fixed probes across trial comparisons within a run, with explicit seed/probe count and controlled probe growth. Record empirical Monte Carlo uncertainty separately from iterative-solve error; standard error across probes does not bound CG bias. Tighten solves/increase probes when uncertainty is comparable to the update or stopping threshold. Validate calibration across independent seeds, not only reproducibility under one seed. These are new approximation policies, not behavior supplied by mgcv's dense `rV` products.

**Score availability is a dependency:** the EFS ratio itself needs traces/quadratics, but `efsudr`'s extension/contraction and score-window stop still use REML scores, including log determinants. An operator backend therefore needs either a validated structured exact score or a separately validated approximate score provider, such as the parent's stochastic log-determinant work. Removing score checks to avoid log determinants creates a different controller. Do not return a fictitious finite score or a converged result with NaN score to satisfy the existing result API.

If approximate scores are used, compare their uncertainty against trial-score differences and record indeterminate comparisons explicitly. A noisy decrease is not a demonstrated improvement. Projection of unidentifiable directions, the penalty pseudo-determinant, and the observed score matrix must be handled before claiming a usable score provider. A Fisher SPD solver does not make an indefinite observed log determinant suitable for CG/SLQ.

Parent PR 12 owns preconditioning and Krylov implementation; parent PR 6 owns factor/operator prediction interfaces; bulk approximate intervals remain a separate deliverable. No p-by-p inverse, covariance, or dense global penalty/range matrix may be hidden in setup or postprocessing. This no-p² rule is for the operator stage, not a retroactive claim about dense EFS.

Operator acceptance compares against the dense **EFS** reference at matched states and on complete fits, then records approximation error relative to the existing Newton reference separately. Parent-plan acceptance governs memory growth, source scans, and numerical uncertainty. An operator EFS implementation without a reliable score or trace provider remains an internal experiment, not a supported large-p optimizer.

## 10. Implementation PR sequence and exit criteria

| PR | Bounded deliverable | Dependencies | Exit criterion |
|---|---|---|---|
| EFS-1 | Pinned-source oracle, fixture schema, controller branch traces, architecture/policy decisions | Current repo; no parent backend work | Reproducible `efsudr` statistics/branches and explicit scope/monotonicity release decision |
| EFS-2 | Dense d/t/q kernels, roots/adapters, raw ratio update | EFS-1; parent PR 2 optional | STRICT/MODERATE R algebra parity, including coupled penalties; JIT and memory checks |
| EFS-3 | Dense known-scale controller and optimizer-neutral result adapter; opt-in API only after release gate | EFS-2 | Poisson/binomial EFS parity, all controller branches, no full outer Hessian, unchanged Newton defaults |
| EFS-4 | Regular-family unknown-scale behavior, Gaussian then Gamma | EFS-3 | Input/update/reported phi and score timing match pinned EFS; EDF/covariance/public contracts pass |
| EFS-5 | Fixed-theta NB, then conditional-theta kernel and EFS-specific PIRLS integration | EFS-3; EFS-4's state design reused; parent PR 4 optional | NB conditional and final EFS parity, no registry mutation or default-PIRLS regression |
| EFS-5.1 | Remaining existing-family/link dense EFS gates and public capability inventory | EFS-3–EFS-5; shared inventory/contract from parent 7.1 | All five existing families and supported links, both NB theta modes, have explicit pinned-EFS parity or a reviewed unresolved compatibility issue; no claim of completion while required cells remain missing |
| EFS-5.2 | Streamed direct coefficient/statistic/score provider for EFS | EFS-5.1; parent 5–6 and applicable 7.1–7.5 kernels | Same controller/scale/theta policy with bounded row scans, exact manageable-p traces/scores, public family snapshots and all-family release gates |
| EFS-6 | Operator fit/trace/score adapters, then optional stochastic statistics | Dense EFS reference for the selected family; parent PR 12; validated score provider | Calibrated trace/fit error and solver status, no p² hidden work; labeled approximation |

EFS-5 may be split into fixed-theta and estimated-theta PRs if the inner-loop change is large. EFS-6 is an adapter milestone with separate trace/score approximation changes, not permission to bundle CG, SLQ, theta updates, and uncertainty into one unreviewable PR. Public capability expands only as each declared family/link regime passes its gate.

EFS-5.1 is the explicit follow-up for the earlier table's "wider existing basis/link coverage" family/link requirement; it does not require inventing new distributions or expanding the prepared-basis domain. Use the parent PR7.1 inventory as the source of family/link scope, not only default links. Preserve named EFS-specific departures and pinned-R defects transparently; an expected failure is not a completed support cell. Shared-kernel refactoring must leave dense Newton defaults unchanged.

EFS-5.2 adapts the existing accepted/trial controller to replayable fit, exact Fisher-trace and scalar-score providers. Preserve score_phi/update_phi/carried_phi timing and NB conditional theta updates **inside** the streamed PIRLS iteration, with globally reduced conditional statistics and objective refreshes. Do not fully converge beta and then alternate theta as a host shortcut, and do not substitute the exact-REML joint theta optimizer from parent PR8.2. Compare streamed versus dense EFS on identical models and against pinned EFS, with STRICT/MODERATE gates, bounded state and explicit source-scan counts. No full training design or response vector may be materialized behind this adapter. This is a new reviewable PR, not assumed to be supplied by the original PR7 or dense EFS implementation.

### Performance evidence

Extend [scripts/benchmark_large_p.py](../../scripts/benchmark_large_p.py) and [scripts/benchmark_vs_r.py](../../scripts/benchmark_vs_r.py) with explicitly labeled EFS runs. Compare JaxGAM Newton, JaxGAM EFS, and pinned R `gam(optimizer="efs")` on the same model. `bam(fREML)` may be a separate runtime comparator, not a parity oracle. Reuse [scripts/profile_fit.py](../../scripts/profile_fit.py) for attribution.

Measure cold compilation, warm fit time, inner/outer/theta iterations, coefficient factorizations, RHS solves, penalty derivative time, score evaluations, peak host/device memory, and final numerical differences. Synchronize device results. Use isolated temporary JAX caches rather than copying the existing benchmark scripts' user-global cache deletion. Record hardware, versions, dtype, thread settings, formula, seed, n/p/m, rank, and all controls. Add an appropriate Makefile benchmark target when the harness is implemented.

Expected shape of the benefit: removal of m-by-m smoothing Hessian formation/storage and full fitted-model derivative propagation. Dense coefficient solves remain generally O(p³); exact root contractions can cost O(p² times the number of RHS columns); row scans and ordinary PIRLS remain. Parent PR 2 is needed before claiming local-only penalty storage instead of today's O(m*p²) representation. Avoid an O(m)-per-iteration or fixed-speedup claim without accounting for these costs and measurements.

### Completion checklist

- [ ] Approved post-v1 optimizer scope, API, and method-specific monotonicity policy; architecture sketch corrected without weakening Newton gates.
- [ ] Paper equation → pinned R routine → JaxGAM kernel/test mapping is implemented and reviewed.
- [ ] Exact dense penalty statistics cover overlapping penalties and matching coordinate/scale conventions.
- [ ] Controller reproduces finite-domain reference branches, initial shift, caps, state timing, and stopping; boundary departures are named.
- [ ] Supported regular-family scale and NB theta paths pass their own gates; unimplemented combinations reject explicitly.
- [ ] EFS-5.1 completes the shared existing-family/link inventory; default-link-only support is not labeled all-family completion.
- [ ] EFS-5.2 validates streamed EFS independently of dense EFS and parent PR8's exact-REML optimizer, including in-loop theta and carried-scale timing.
- [ ] Full/inference results preserve existing coefficient, covariance, family-snapshot, and prediction contracts.
- [ ] STRICT/MODERATE pinned EFS comparisons run in CI; no skipped-oracle or Newton-only substitute.
- [ ] New modules exceed 80% coverage; JIT paths compile and run; no production full-REML Hessian is evaluated.
- [ ] Reproducible benchmark artifacts distinguish many-m improvement from large-p/operator capability.
- [ ] Operator/stochastic support, if delivered, has separate calibrated trace/score error and truthful strategy diagnostics.

[paper]: https://doi.org/10.1111/biom.12666
[paper-full]: https://academic.oup.com/biometrics/article/73/4/1071/7537785
[preprint]: https://arxiv.org/abs/1606.04802
[fellner]: https://doi.org/10.1080/00401706.1986.10488097
[schall]: https://doi.org/10.1093/biomet/78.4.719
[r-outer]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/mgcv.r#L1632
[r-initial-spg]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/mgcv.r#L4720
[r-null]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/mgcv.r#L1852
[r-estimate]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/mgcv.r#L1870
[r-efs]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/gam.fit4.r#L821
[r-reparam]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/gam.fit3.r#L9
[r-penalty-space]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/gam.fit3.r#L2659
[r-fit3]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/gam.fit3.r#L67
[r-fit4]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/gam.fit4.r#L240
[r-ddeta]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/gam.fit4.r#L4
[r-gdi1]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/src/gdi.c#L2325
[r-gdi2]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/src/gdi.c#L1953
[r-fisher1]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/src/gdi.c#L2769
[r-fisher2]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/src/gdi.c#L2262
[r-theta]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/efam.r#L5
[r-nb]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/efam.r#L160
[r-control]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/mgcv.r#L2469
[r-efs-general]: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/gam.fit4.r#L1478
