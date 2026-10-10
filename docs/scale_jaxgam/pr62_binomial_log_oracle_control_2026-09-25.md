# Binomial/log mixed-boundary R oracle control

The unchanged 328-row `mixed` fixture in
`test_binomial_log_admission_matches_pinned_default_final_fields_and_se` is a
platform-sensitive stopping boundary inside pinned `mgcv::gam.fit3`.  With
`epsilon=1e-10,maxit=100`, R 4.5.2/mgcv 1.9-3 converges in six iterations on
the accepted ARM64 image but reaches 100 iterations without convergence on
native AMD64 CI.  Increasing only `maxit` to 200 does not resolve the AMD64
oscillation.  It also remains nonconverged after changing epsilon to its next
representable value and to `1.1e-10`, `2e-10`, and `5e-10`.

This gate now uses `epsilon=1e-9,maxit=100` for that exact `mixed` R oracle.
This is still 100 times tighter than `gam.control()`'s source default
`epsilon=1e-7`.  It converges in three ARM64 iterations and four AMD64
iterations.  The fixture arrays, R family fixes, model, source fit, and Python
controller are unchanged; Python continues to use `tol=1e-10`.  The
`adjacent`, `near`, and `exact` cases retain the original R epsilon of
`1e-10`.  In particular, applying `1e-9` to `near` changes its endpoint and is
therefore rejected.

At the selected mixed control, the ARM64 and AMD64 R results differ by at
most `1.25e-16` in beta, `4.55e-13` in deviance, `5.55e-16` in EDF,
`1.93e-10` in REML, `5.13e-15` in Fisher covariance, `5.11e-13` in standard
error, and `1.11e-16` in fitted mean.  On the exact accepted ARM64 publication
image, all original B1/B17/B200/compiled/eager final-field assertions pass
against the selected oracle.  Comparing those same Python results with the
AMD64 R oracle also passes every existing field-specific STRICT or previously
reviewed MODERATE tolerance.  Native AMD64 Python execution remains the final
CI gate; local QEMU cannot import the pinned JAX runtime because its emulated
CPU lacks AVX.

Validation used accepted #62 image
`sha256:f1da0b2dbf382f36470bc7c8fa8bcf2b2b19bdb099858b2bd7bead1c08dc693b`.
The R-only AMD64 diagnostic image was
`sha256:f0f26f9c5c7c6c3227a2b52755f3f73e99d7b4d20886c788d3ffeaa28a736e65`;
it has the pinned R and mgcv versions but newer Matrix/nlme/lattice packages,
so it supplies cross-architecture source evidence rather than release
validation.  The original failed control and all sweep outputs remain in the
review evidence bundle.
