# Shared Binomial source boundaries: reviewed component

This bounded correction is based on the accepted internal regular controller
`1c30346cc07ce6d5e3428186d70185542355e867`. It does not admit the guarded
Binomial/log cancellation profile or complete PR7.3/PR7.4.

## Pinned source contract

R 4.5.2 stats `binomial()$aic` uses `dbinom(round(m*y), round(m), mu,
log=TRUE)` with `m=weights` for single-column responses (`n=1`). mgcv 1.9-3
`fix.family.ls` obtains saturated likelihood through that AIC method at
`mu=y`. Both the binomial coefficient and the probability exponents must
therefore use rounded trial and success counts. The former unrounded
exponents caused a 2.525909 REML discrepancy on the preserved fractional-
weight boundary fixture.

The shared likelihood and CPU AIC now follow those rounded counts with stable
`xlogy`/`xlog1py` endpoint terms. CPU AIC preserves actual valid probabilities,
including near-zero/near-one means and compatible endpoints. Impossible
endpoint observations retain positive infinity. Zero-prior rows remain neutral.

Direct deviance, deviance residuals, and smooth derivative contributions retain
valid means within (0,1). The former clipping at 1-1e-10 flattened valid
near-success observed curvature and could create a false score-system alias.
Invalid-domain derivative operands are neutral; execution validity owns
rejection. The existing source literal alpha==0 policy and cancellation guard
remain unchanged.

## Exact validation and review

Frozen image `jaxgam-test:pr73-binomial-boundary-r2-20260917`, ID
`sha256:4c0196ee98f0a0b9acab995c2beba87dcc1208a281fd32e51c794809e3396b2d`.
All 162 production/test Python hashes match. Owning pinned-source, eager/JIT,
family/link, initialization, dense PIRLS/REML, regular EFS and regular streamed
controller regressions: **355 passed**, no skips or xfails; standard.py coverage
**99.60%**. Repository lint passes. All new numerical gates use STRICT.
The main agent independently matched all image/source hashes and reviewed the
pinned source and both changed Python files before accepting this component.

Detailed evidence is retained in
`/private/tmp/jaxgam-pr73-binomial-boundary-r2-review-handoff.md`, its manifest,
and the owning/lint logs of the same stem. Diagnostic-only guard bypasses and
nonconvergent tightened-control trajectories are research evidence, not
production eligibility or release gates.
