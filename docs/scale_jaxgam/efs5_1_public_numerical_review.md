# EFS 5.1 public numerical review

Date: 2026-09-10. Oracle: R 4.5.2 with mgcv 1.9-3. Unless stated
otherwise, both implementations use the public EFS profile: 200 outer
iterations, maximum log smoothing parameter 15, score tolerance 0.1, and an
inner PIRLS tolerance of 1e-7 with 200 iterations. Python supplies R with the
same initial smoothing parameter computed by `efs_initial_log_lambda`.

This record documents the four distinct review passes that support two narrow
STRICT-to-MODERATE choices. It does not change global tolerances or any other
family/link gate.

## Poisson/log seed 1033

The checked-in validation gate constructs 96 rows with
`_make_single_data("poisson", seed=1033)`, then fits
`y ~ s(x, bs='cr', k=6)`. The generated CSV had SHA-256
`5a6907c13fbbcba100d18d0a7b777b2ba7beb031503eb2868f5609f69af5c216`.

1. The same-input public fit and pinned R fit had maximum absolute residuals
   of 3.710e-10 for coefficients, 4.004e-10 for fitted means, 2.218e-8 for the
   smoothing parameter, 8.499e-10 for per-term EDF, 8.482e-10 for total EDF,
   and 3.141e-11 for covariance. Deviance, score, null deviance, and scale
   passed STRICT.
2. Public Phase 3 output was compared with the direct EFS controller.
   Coefficients differed by 5.551e-17; fitted means, deviance, score, and
   smoothing were equal. Both routes used the same outer-iteration count.
3. A repeated public fit was array-exact for coefficients, fitted means,
   smoothing, score, and diagnostics. Python and R both stopped after six
   outer iterations; Python reported `score_window`.
4. Reducing the score tolerance to 0.001 produced eight outer iterations in
   both implementations. Coefficient and smoothing residuals fell to
   5.797e-11 and 2.270e-9; fitted means, deviance, score, EDF, null deviance,
   and scale passed STRICT. The public and direct-controller results remained
   STRICT.

The selected-fit gate therefore uses MODERATE only for coefficients, fitted
means, smoothing parameters, EDF, total EDF, and covariance. Deviance, REML
score, null deviance, scale, outer count, dispatch strategy, and diagnostics
remain STRICT or exact.

## Estimated negative-binomial identity/square-root seed 901

The checked-in validation gate constructs 160 rows with seed 901, theta 2.7,
unit offset, and mean
`exp(1.2 + sin(3.2*x) + 0.35*cos(6*x))`, then fits
`y ~ s(x, bs='cr', k=7)` without an explicit coefficient or theta start.

1. With zero offset, both pinned R and Python reject the default start for
   identity and square-root. `gam.fit3.r`/`gam.fit4.r` use a zero-coefficient
   recovery anchor, which maps to invalid mean zero for these links.
2. Pinned R default-start fits succeeded for both links with each constant
   positive offset 0.25, 0.5, 1, 2, and 4. Before correction, Python
   square-root succeeded while identity stopped with `inner_failure` before
   the first outer update.
3. At the first identity divergence, the observed Hessian was indefinite.
   The implementation was corrected to follow `gam.fit4.r:391-416`: retry the
   WLS solve after zeroing nonpositive-curvature rows. A JIT regression checks
   that algebra at STRICT.
4. Four nonlinear seeds were compared after the source correction. For seed
   901 the outer counts match. Square-root passes STRICT for every selected
   field. Identity has maximum absolute residuals of 1.678e-9 for
   coefficients, 1.939e-9 for fitted means, and 7.343e-9 for smoothing;
   deviance (1.424e-10), score (5.946e-11), and theta (8.953e-11) pass STRICT.

The identity gate uses MODERATE only for coefficients, fitted means, and
smoothing. Identity deviance, score, theta, outer count, convergence, and
diagnostics remain STRICT or exact. Every square-root field remains STRICT.
LOOSE is not used for either fixture.
