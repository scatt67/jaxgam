# Regular/signed fitting publication composition

This candidate composes the reviewed PR7.3 regular-family and PR7.4 signed-system components onto published d67b799d0a4d498cb4a11ea8d80c65ef0d370b25. It supplies an internal fixed-sp/trial-score-scale engine for all32regular family/link cells, with the shared family contract, source row starts and natural-column null anchor, optional retained fitting p-vector, signed QR/SVD coefficient recovery, source deviance halving, observed score versus Fisher covariance and source final-refit attribution. Public all-family routing remains PR7.6; joint smoothing/phi optimization remains PR8.1.

## Reviewed implementation provenance

- 4a9aa1242600d69a96aed2d0be0f0ccfee6fedb0: source materialized initial-system operations and exact identity JVP. Literal alpha==0 replacement and raw cancellation diagnostics remain; only the reviewed controller opts into source signed recovery. Shared LogLink second derivative has affected dense EFS tests.
- e4d3f3ae5b7cb99e55e3f1b0971a43279c5334e3: bounded absolute/negative QR roots, compact negative SVD correction and source direct-RHS fallback; coefficient and score/logdet admissibility are separate.
- 1c30346cc07ce6d5e3428186d70185542355e867 and f1f236a04f2c1d33e9ad6a3a9f3b61d7b2157ab8: source regular controller, finite trial recovery, full source Fisher retry/exhaustion and boundary admission.
- 15a3fdf0db93de47a813119f870a0f28978c2d2f: streamed global Gaussian initialization hooks; its shared dense/EFS changes are already published and retained here.
- f63c8c58ae8d001be1d5a63395eb4898fc0975ac: source natural-column null projection.
- 1bfd61f90bca105dbf8e438635aeb417ef778d72: real no-intercept/alias and retained starts, source-good working row mask, pre-gdi source score/candidate penalty versus reported state. Actual penalized Gamma source and controlled invalid-final-refit gates prove score timing; supplied-Dp/logdet path is JIT-tested and preserves existing default/PivotedQR result call bytes.
- 5457abd67fbfa46b4d96f7fa20d95d37c8bed38e: actual rank1-penalty32-cell gates and four extreme-prior policies; exact reviewed MODERATE fields are digest-guarded and recorded separately.

Pinned R4.5.2/mgcv1.9-3 is the source oracle. Read gam.fit3 plus family/link materialization and signed pls_fit1/gdi1 source; the natural null helper follows pinned get.null.coef and R dqrdc2/dqrsl. No runtime R dependency is introduced.

## Scope and preserved published work

All reviewed source/test files are copied byte-for-byte, except standard.py retains the already published source-correct Gaussian.aic method verbatim. Its remaining AST equals the reviewed private file. Shared EFS production is already byte-identical to the published file. API/Makefile/workflow, Gaussian AIC tests/docs, EFS benchmark implementation/artifacts/tests and all other published files remain present. No PR9 Gaussian optimizer/compression orchestration or PR11 work is imported.

Known workspace is prospectively charged for batched design/weighted stacks/raw QR copies, compact factors/penalty roots, null-helper overlap/alias shift and bounded numeric history length. The ledger is a known Python-visible workspace bound; opaque native/compiled scratch is not claimed as an allocator hard cap. There are no retained n-row coefficient arrays or Python RowSource iteration inside JAX transformations.

## Prior exact component validation

Unchanged reviewed regular core:179pinned tests,20warnings,643.40s,original exit0,no skips/xfails;93.19%combined coverage (controller91%,CPUinitializer90%,shared kernel97%). Signed/null new modules separately exceed80%. Release matrix slice:36passed85.13s,original exit0,no skips/xfails/warnings. All32actual rank1-penalty cellsB17 and four extreme-prior policiesB1/17/200 are covered. The first2-failure release gate and complete executed correction functions/counters/raw errors are preserved; accurate history is one inherited source-good correction plus three executed alternatives. Final publication-image tests and full-suite gate are recorded separately after execution.
