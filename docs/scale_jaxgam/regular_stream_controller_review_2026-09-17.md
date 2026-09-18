# Bounded regular streamed controller review

The accepted component adds an internal fixed-sp, explicit trial-scale regular
PIRLS controller over prepared RowSource batches. It does not release the full
PR7.3/PR7.4 family matrix or change public execution routing.

The controller preserves the distinction between initial mustart predictors and
the null coefficient anchor, measured source and batch counts, source-style
penalized-deviance halving, step-local Fisher recovery of an indefinite observed
solve, and gam.fit3's final coefficient refit. Coefficient validity, observed
score determinant validity and Fisher reporting validity are separate checks.
Canonical links use their family-owned Fisher score contract; other regular
links retain signed observed score factors and positive Fisher covariance/EDF
factors. Phase 3 consumes the tagged determinant without rebuilding a normal
equation solve. Nonfinite trial quantities are rejected; nonfinite working
statistics and final quantities fail before convergence is reported.

The numerical-workspace preflight runs before source scans, penalty root
construction and device transfers. It includes conservative batch/coefficient
buffers and iteration-dependent history storage, including float objects,
list references and the final tuple. Source-owned storage and opaque native or
compiled scratch remain outside this numerical-workspace budget; it is not a
process RSS limit.

## Validation and provenance

The frozen six-file r3 candidate was tested in
`jaxgam-test:pr74-regular-r3-20260917`, image
`sha256:2621635b2ed6a68b1bf07944119efebd60b71312dad2c2a6b8c1d5e75941d9b0`.
All 161 production/test Python file hashes match that image. The owning pinned
run passed 87 tests without skips or xfails in 47.13 seconds. Coverage is 88%
for the execution controller and 100% for its JIT trial kernel, combined
88.59%. Repository lint passes.

Direct pinned R 4.5.2/mgcv 1.9-3 gam.fit3 gates compare all three Gamma links
(identity, log and inverse) with weights and offsets at batch sizes 1, 7 and
200. A nonlinear Gamma/identity cubic-smooth fixture uses seven coefficients,
nonempty UrS/Eb, fixed sp=.35 and trial scale=.7 at batch sizes 1, 17 and 200.
Coefficients, deviance, Fletcher reporting scale, EDF and REML pass STRICT; the
penalized fixture's full unscaled Fisher inverse also passes STRICT.

Additional gates cover genuinely negative observed working curvature and its
Fisher retry, finite-row statistics overflow, JIT invalid-trial reductions,
empty/zero/reordered/repeated source blocks with measured scan counts,
prospective memory rejection, and iteration-history accounting. A p20/n4096/
B2048 packed-QR diagnostic observes 1,333,120 bytes of distinct live design
buffers against a 4,562,320-byte ledger including 5,008 bytes of history. This
measurement is a lower bound on visible buffers, not a native allocation cap.
Affected initialization, signed solver and dense regular EFS tests are included
in the owning run.

Source references are pinned mgcv 1.9-3 R/gam.fit3.r, R/mgcv.r::get.null.coef,
and src/gdi.c::pls_fit1/gdi1. NB gam.fit4's positive-observed retry is distinct.

## Remaining release requirements

The internal controller currently requires a literal prepared intercept and
explicit trial scale for unknown-scale families. No-intercept/deficient-design
NULL anchors, the full regular family/link fixed-sp matrix, remaining recovery
and failure trajectories, broader factor/device memory measurements, complete
outer scale/smoothing control and public release gates remain outstanding.
The Binomial/log cancellation guard is preserved. Its first-system diagnostic
alone does not authorize guard removal; actual paired source recovery and final
quantity evidence are required. No tolerance relaxation was introduced.
