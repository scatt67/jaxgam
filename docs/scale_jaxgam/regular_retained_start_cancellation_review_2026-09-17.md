# Regular retained-start cancellation fixtures: review proposal

The root reviewed and approved the exact numerical scope below after four executed correction candidates. The immutable r8f revision remains at STRICT with its original two failures preserved; the separate r8g child implements only the approved assertion changes. This decision applies only to the original seeded two-coefficient matrix fixtures; it does not alter working-weight admissibility, literal alpha replacement, convergence controls, default initialization guards, or other family/link assertions.

The unchanged fixtures are `_case(Poisson, "identity")` and `_case(Binomial, "log")`, seed73300 /73301, n83, one real zero-prior row, unequal positive priors and offset, supplied coefficient start. Pinned R4.5.2/mgcv1.9-3 and Python use epsilon1e-7/maxit200. All measured fits converge in four iterations.

Each correction candidate was executed in the actual full controller on B1/17/200. Candidate1 is implemented in the WIP shared kernel/controller; candidates2–4 are isolated bounded solver alternatives injected for measurement, with complete function sources archived. They were not retained because they did not restore STRICT. Candidates2–4 are cumulative and exercise distinct source operations; they are not separate oracle/control checks.

## Original poisson/identity fixture

| Correction attempt | Concrete change | Max beta absolute error | Max REML absolute error | Max Fisher covariance absolute error | STRICT restored |
|---|---|---:|---:|---:|---|
| source_good_rows | Subset source-good rows before working QR; leave unweighted public NULL scan unchanged. | 6.59898136e-10 | 5.51409585e-10 | 1.76862283e-11 | No |
| source_final_qr | Reproduce gdi.c final second QR even when the penalty has zero rows. | 6.59898136e-10 | 5.51409585e-10 | 1.76862283e-11 | No |
| preserve_literal_qtz | Use the materialized second-Q transformed pseudodata directly; avoid inverse/reconstruction of Qᵀz. | 6.59897914e-10 | 5.51409585e-10 | 1.76862283e-11 | No |
| source_triangular_order | Use literal source-order forward/back-substitution accumulations in the candidate coefficient solve. | 6.59897914e-10 | 5.51409585e-10 | 1.76862283e-11 | No |

Actual final fitted-mean and SE residuals for the retained production candidate:

| Batch cap | Mean max absolute / relative | Link-SE max absolute / relative | All fits converge |
|---|---:|---:|---|
| 1 | 4.5939641e-10 / 1.18577308e-09 | 8.37472591e-11 / 6.3871864e-10 | Yes |
| 17 | 4.86843454e-10 / 1.25615069e-09 | 7.28331839e-11 / 5.5547524e-10 | Yes |
| 200 | 4.72413109e-10 / 1.20347686e-09 | 4.70314898e-11 / 3.586864e-10 | Yes |
## Original binomial/log fixture

| Correction attempt | Concrete change | Max beta absolute error | Max REML absolute error | Max Fisher covariance absolute error | STRICT restored |
|---|---|---:|---:|---:|---|
| source_good_rows | Subset source-good rows before working QR; leave unweighted public NULL scan unchanged. | 8.31818703e-09 | 1.46127377e-08 | 6.79030346e-10 | No |
| source_final_qr | Reproduce gdi.c final second QR even when the penalty has zero rows. | 8.31818703e-09 | 1.46127377e-08 | 6.79030346e-10 | No |
| preserve_literal_qtz | Use the materialized second-Q transformed pseudodata directly; avoid inverse/reconstruction of Qᵀz. | 8.6651526e-09 | 1.45267904e-08 | 6.94856436e-10 | No |
| source_triangular_order | Use literal source-order forward/back-substitution accumulations in the candidate coefficient solve. | 8.6651526e-09 | 1.45267904e-08 | 6.94856436e-10 | No |

Actual final fitted-mean and SE residuals for the retained production candidate:

| Batch cap | Mean max absolute / relative | Link-SE max absolute / relative | All fits converge |
|---|---:|---:|---|
| 1 | 2.82013113e-09 / 7.04540132e-09 | 1.25286412e-09 / 7.72209158e-09 | Yes |
| 17 | 5.23298849e-10 / 1.29916292e-09 | 1.30044903e-09 / 8.05064704e-09 | Yes |
| 200 | 9.05786779e-10 / 1.21030665e-09 | 1.0561905e-09 / 6.59376305e-09 | Yes |

## Oracle-only diagnostics (not counted as correction attempts)

The original tightened epsilon1e-12 Poisson/identity fixture is preserved as paired genuine source/Python nonconvergence at the same maxit100. It cannot supply a converged reference. Default controls are an optimizer control decision; result comparison still starts STRICT.

Bit-preserved shared-state C_pls_fit1 comparisons distinguish the kernel from reduction arithmetic. Poisson/identity W/z are bit-for-bit identical at the same source eta; B200 then matches the default C solve to2.22e-16, while B1/B17 with the identical R arrays have slope differences1.31e-9/1.18e-9. Source use.wy=0 adaptive fallback remains false. The diagnostic forced use.wy=1 changes the source slope1.23e-9, demonstrating sensitivity but not authorizing a production route change. Binomial/log preserves raw cancellation/sign mismatches and small working-weight differences; no signs are repaired or clipped.

The earlier text-output frozen-system diagnostic is superseded by binary arrays; its serialization precision is not used in numerical conclusions. Reader/path-only harness failures are preserved but are not counted as correction attempts.

## Exact narrow decision, root reviewed

For these two existing matrix fixtures at B1/17/200 only: use repository MODERATE for beta, fitted means, Fisher covariance and link SE; also REML for Binomial/log only. Every deviance/known-scale/EDF/NULL-map assertion, Poisson REML, and all other30matrix cells remain STRICT. Require true source/Python convergence and finite complete final fields. Existing cancellation-only/exact-success failures and approved earlier boundary gates remain intact. This is not a family-wide guard or tolerance change.

All residuals are explicitly measured above. Link-SE relative residues are below8.06e-9 here; this is distinct from the earlier near-boundary fixture where small SEs passed MODERATE using its absolute tolerance.

## Exact revision evidence

Production base15a3 plus accepted null helper childf63c8c58ae8d001be1d5a63395eb4898fc0975ac. Worktree `/private/tmp/jaxgam-pr73-regular-matrix-r8-20260917`. Immutable image `jaxgam-test:pr73-regular-matrix-r8f-20260917`, ID sha256:2fa3ea487a9d1dd0ac4ef77d8f6634d7fad6365368700ac23d538a8206a6fe5c. All167Python source/test hashes match.

Collected unchanged32-cell/retained/no-intercept/resource/tight-boundary owning file:35passed2failed200.90s,terminal1. Affected shared initialization/controller/recovery/JIT regressions:109passed175.06s,terminal0,no skips/xfails,91.93% combined coverage (controller88%,CPUinitializer90%,shared kernel97%). Focused CPU/JIT mask gates17passed. Lint passes.

Files: `r8f-owning.log`, `r8f-regressions.log`, `r8f-frozen-system-r3.py/.log`, `r8f-correction-candidates-r2.py/.log/.json`, `r8f-candidate-{name}.py`, `r8f-final-fields.py/.log/.json`, `r8f-review-manifest.json`, all under `/private/tmp/jaxgam-pr73-regular-matrix-`. No r8f validation remains live.

This is a bounded internal controller/retained-start milestone. Whole PR7.3/7.4 public routing, full release resources/controls and final EFS integration remain open; PR9 rank/score requirements remain separate.

## Penalized source-score attribution gate

The r8g owner uses the unchanged seed737 nonlinear penalized Gamma/identity fit, fixed smoothing.35, incoming score_phi.7, and its separately computed Fletcher reported_phi. A clone of pinned gam.fit3 records the actual C_gdi1 boundary: raw pre-gdi deviance, stopping penalized deviance, candidate solve penalty, candidate validity and incoming scale. Every ordinary source computation is preserved.

A second, explicitly controlled source-clone branch replaces only the final gdi1 candidate with an invalid fitting-coordinate p-vector and a consistent candidate penalty. It preserves real PIRLS iterations, pre-gdi observed/Fisher systems, determinant and EDF; the actual source then returns its saved feasible coefficient through the literal invalid-refit recovery. Python substitutes the same frozen final candidate at the corresponding return boundary. This verifies that scalar score retains the candidate penalty, while coefficients/means/Fletcher use the feasible reported state; it does not claim the naturally converged fixture entered that branch. All source-score payload and final-model assertions in this gate start STRICT.
