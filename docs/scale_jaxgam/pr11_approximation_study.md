# PR11 discretization approximation study

Status: measured study, not a parity gate. It answers the plan's PR11 item
"refine q per margin and compare predictions, held-out deviance, smooth
shapes, EDF, and interval behavior against the undiscretized reference"
(scaling plan §PR 11), which the plan keeps separate from the pinned-`bam`
parity gates in [pr11_working_model_plan.md](pr11_working_model_plan.md).
Nothing here changes a tolerance or a test.

## Summary

- **Implementation error is at rounding level.** At the `exact` level
  (no covariate rounded), with fixed smoothing parameters, the discrete
  route matches pinned `bam(discrete=TRUE)` at the same grid, `sp` and theta
  to at most 2.2e-14 in held-out link predictions for Poisson, Binomial,
  Gamma and both negative binomial modes, and to at most 8.3e-11 for
  Gaussian. Its EDF equals `bam`'s to at most 1.6e-10. The remaining
  discrete-versus-dense gap (1e-9 to 6e-8; up to 5e-7 for Gaussian) is the
  dense route's: in every case the dense fit is exactly as far from `bam`
  as it is from the discrete fit.
- **Discretization error is negligible from m = 100 on, and material at
  m = 10.** The comparison here is working-model REML against the same
  strategy with nothing rounded, across all 20 family/model/size cases.
  - m >= 100: held-out η stays within 0.22 reference standard errors (0.16
    for the study model at n = 10,000). Total EDF moves by at most 0.15 and
    relative held-out deviance by at most 6.9e-4. Pointwise SE ratios stay
    within 0.97–1.03, and the discrete intervals cover every undiscretized
    term curve.
  - mgcv default (1000 points per margin): 0.01–0.13 SE (max |Δη| at most
    9.2e-3), |ΔEDF| <= 0.06, deviance change <= 2.6e-4, median SE ratio
    1.000–1.001.
  - m = 10: 0.66–5.4 SE (the largest at n = 50,000) and EDF up to +3.1. The
    study model's `ti` interaction changes shape by 37–38% in relative L2,
    and the
    discrete intervals cover the undiscretized term curve at as few as 15% of
    grid points.
- **Two error components.** Covariate rounding shrinks at least in
  proportion to the grid spacing up to m of about 100–300. Beyond that the
  error flattens at about 1e-3 rms in η. This plateau comes from a basis
  change that does not vanish until nothing is rounded. Knots are placed on
  the compacted unique values, so they move by 1–1.5% of the covariate range
  at every rounded level (0.6% at n = 50,000). Penalty normalization
  is also computed on the compact frame.
- **Fixed raw `sp` is not a fixed penalty across levels.** The compact-frame
  normalization changes penalty norms by factors of 0.37–1.14. At the same
  `sp`, EDF therefore rises by 0.7–1.2 at m = 100 and 0.15–0.26 at the
  default (n = 10,000). Dividing each `sp` by its penalty-norm ratio removes
  almost all of this (|ΔEDF| <= 0.07 at m >= 100). Working-model REML re-selects `sp`
  and is unaffected.
- **Strategy difference is separate and small.** Without rounding,
  working-model REML (`bgam.fitd` performance iteration) differs from dense
  nested REML by at most 0.05 SE in held-out η. For the negative binomial
  with estimated theta, `estimate.theta` (conditional on the current mean)
  gives a theta 2% (n = 10,000) to 10% (n = 2000) above dense joint REML.

## Setup

**Harness.** `scripts/benchmark_discrete_approximation.py`, run as
`make benchmark-discrete-approximation OUTPUT=... ARGS=...`, with tests in
`tests/test_benchmark_discrete_approximation.py`. It fits each case in a
fresh child process with a task-owned JAX cache. It refuses a dirty or
unexpected checkout, a child whose source identity differs from the
parent's, and an output path inside the checkout. Raw JSON stays outside the
repository; the tables below were condensed from it.

**Provenance of the run.**
- Produced on the PR11 line before its port onto PR9: commit `d5c038f`, a
  clean tree; harness SHA-256 `d7a10428…` as recorded in the run (it covers
  the harness and that line's `scripts/benchmark_scaling.py`). On the port
  the harness file is byte-identical but `benchmark_scaling.py` is the
  PR8.3 version, so the harness digest there is `dd473f7d…`. The study was
  not rerun on the port.
- macOS 15.6 arm64 with 10 CPUs, shared with other jobs.
- Python 3.13.10, JAX/jaxlib 0.9.0.1, NumPy 2.3.5 (Accelerate), SciPy 1.17.0; float64.
- `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.
- The R anchors used host R 4.5.2 with mgcv 1.9-3 through the typed rpy2
  `RBridge` (version check enforced). This R links Accelerate, not the
  pinned Linux OpenBLAS.

Wall times: the study model took 8.5 min, the tensor model 3.6 min and
n = 50,000 2.5 min (14.5 min in all).

**Data.** Seed 20261025 for training rows and 20261026 for 2000 held-out
rows.
- Covariates: x0, x1, x2 ~ U(0, 1), so every training value is unique; `fac`
  has 3 levels and `g` has 20.
- True predictor: 0.4 sin(2πx0) + 2(x1 − 0.5)² + 0.6 sin(πx0) cos(2πx1), plus a
  `fac` main effect, a level-specific x2 curve, and (study model only) N(0, 0.3²)
  group effects.
- Responses:
  - Gaussian: σ = 0.5.
  - Poisson: log link.
  - Binomial: Bernoulli, logit link.
  - Gamma: log link, shape 2.
  - Negative binomial: θ = 3, log link.

**Models.**
- study: `y ~ fac + s(x0,bs='cr',k=10) + s(x1,bs='cr',k=10) + ti(x0,x1,bs='cr',k=5) + s(x2,bs='cr',k=8,by=fac) + s(g,bs='re')`;
- tensor: `y ~ fac + te(x0,x1,bs='cr',k=6) + s(x2,bs='cr',k=8,by=fac)`.

Tensor margins are discretized separately, as in `discrete.mf`, so every
metric group is one-dimensional. Factor values are never rounded.

**Sizes.**
- study: n = 2000 and 10,000 for all six family modes, plus n = 50,000 for
  Gaussian and Poisson;
- tensor: n = 10,000.

**Discretization ladder.** m = 10, 30, 100, 300, the default (`compress.df`
with `m=NULL`: 1000 points for a one-variable group), 3000 and `exact`
(m = n, nothing rounded). At n = 2000 the 3000 level rounds nothing either.

**Families.** Gaussian, Poisson, Binomial, Gamma (log link), and the
negative binomial with estimated theta ("NB, est. θ") or theta fixed at 3
("NB, θ=3").

**Smoothing strategies.** "Dense" is the undiscretized JaxGAM fit:
`GAM(...).fit(frame)`, REML.
- `fixed_sp`: discrete fit at the dense REML `sp`. For NB with estimated
  theta, theta is fixed at the dense estimate. Compared with the dense fit at
  those same values.
- `fixed_sp_matched`: as `fixed_sp`, but each `sp_j` is divided by the ratio
  ‖S_j‖_F(discrete) / ‖S_j‖_F(dense). This holds `sp_j ‖S_j‖` at the dense
  value.
- `working_model_reml`: discrete working-model selection. It is compared with
  dense nested REML ("vs dense") and with the same strategy at the `exact`
  level ("vs exact"). The second comparison is the discretization effect
  alone.
- Exact REML on the discrete design: `GAM._fit_discrete` raises
  `NotImplementedError` for `sp=None` without `working_model_reml`. The
  harness records it as `unsupported` (`--modes exact_reml`).

**Metrics.**
- Held-out (2000 rows):
  - max and rms |Δη| on the link scale; response-scale differences are also
    recorded;
  - max |Δη| divided by the reference fit's pointwise SE;
  - relative change in held-out deviance, each fit using its own family and
    theta;
  - SE ratio, discrete over reference (median, min, max);
  - the share of rows where the discrete 95% interval covers the reference
    estimate, and each fit's coverage of the true η.
- Terms: each smooth's partial effect is evaluated on a grid (200 points; 30
  × 30 for tensors; per level for factor-by and random effects). Smooth
  curves are centred on the grid, because `bam(discrete=TRUE)` takes its
  centring constraints from the compact frame. Recorded per curve:
  - relative L2 difference;
  - SE ratio;
  - "curve coverage": the share of grid points where the discrete 95%
    interval covers the reference curve.
- EDF: total and per term. For a discrete fit, per-term EDF is reconstructed
  as diag((X'WX+S)⁻¹X'WX) on its discretized design, with `bgam.fitd` working
  weights at the final coefficients. This reconstruction reproduces the fit's
  own deviance to 1e-14. It reproduces the fit's total EDF exactly for
  Gaussian and Gamma, and to about 4e-6 relative otherwise (see the EDF
  notes below).
- Basis change: covariate rounding error, maximum knot displacement relative
  to the range, and penalty Frobenius-norm ratios.

## Implementation checks (exact level, fixed sp)

All entries are max |difference| in held-out link predictions, except the
EDF columns (total EDF differences) and the last column (the relative
difference between the reconstructed and reported deviance). `bam` is
pinned `bam(discrete=TRUE)` at the same m (= n), `sp` and theta, predicted
exactly (`predict.gam`).

| model | family | n | discrete − dense | discrete − `bam` | dense − `bam` | EDF: discrete − `bam` | EDF: dense − `bam` | deviance self-cons. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| study | Gaussian | 2000 | 1.1e-10 | 5.4e-15 | 1.1e-10 | +1.4e-14 | -1.6e-09 | -1.1e-16 |
| study | Gaussian | 10000 | 3.7e-07 | 8.3e-11 | 3.7e-07 | -1.6e-10 | -7.6e-06 | 0.0e+00 |
| study | Poisson | 2000 | 1.8e-09 | 1.1e-14 | 1.8e-09 | +0.0e+00 | +1.6e-04 | -1.5e-15 |
| study | Poisson | 10000 | 3.0e-09 | 1.3e-14 | 3.0e-09 | -2.1e-14 | +2.5e-04 | -2.1e-15 |
| study | Binomial | 2000 | 3.6e-08 | 1.7e-14 | 3.6e-08 | +2.8e-14 | +3.3e-06 | 1.6e-15 |
| study | Binomial | 10000 | 1.4e-09 | 1.4e-14 | 1.4e-09 | -1.4e-14 | +5.7e-06 | -2.5e-15 |
| study | Gamma (log) | 2000 | 6.1e-08 | 8.4e-15 | 6.1e-08 | +7.1e-15 | -3.9e-07 | -2.1e-15 |
| study | Gamma (log) | 10000 | 3.0e-08 | 8.7e-15 | 3.0e-08 | -7.1e-15 | -2.3e-09 | 5.1e-16 |
| study | NB, est. θ | 2000 | 2.4e-08 | 9.2e-15 | 2.4e-08 | +3.6e-14 | -6.4e-03 | -7.8e-16 |
| study | NB, est. θ | 10000 | 3.8e-08 | 9.4e-15 | 3.8e-08 | +2.8e-14 | +7.9e-03 | 1.0e-15 |
| study | NB, θ=3 | 2000 | 1.2e-08 | 8.4e-15 | 1.2e-08 | -7.1e-15 | -6.9e-03 | -1.6e-15 |
| study | NB, θ=3 | 10000 | 2.9e-08 | 9.8e-15 | 2.9e-08 | +0.0e+00 | +7.9e-03 | 7.0e-16 |
| study | Gaussian | 50000 | 2.7e-08 | 7.1e-12 | 2.7e-08 | -1.4e-11 | -5.9e-07 | 0.0e+00 |
| study | Poisson | 50000 | 2.9e-09 | 2.2e-14 | 2.9e-09 | -1.6e-13 | +3.7e-04 | -1.2e-14 |
| tensor | Gaussian | 10000 | 5.0e-07 | 3.2e-12 | 5.0e-07 | +6.1e-12 | -1.4e-05 | 0.0e+00 |
| tensor | Poisson | 10000 | 1.8e-09 | 1.6e-14 | 1.8e-09 | +0.0e+00 | +1.8e-04 | -5.7e-15 |
| tensor | Binomial | 10000 | 8.7e-09 | 1.5e-14 | 8.7e-09 | +0.0e+00 | +5.0e-06 | 2.9e-15 |
| tensor | Gamma (log) | 10000 | 2.2e-08 | 9.5e-15 | 2.2e-08 | -1.4e-14 | -1.9e-09 | 6.7e-16 |
| tensor | NB, est. θ | 10000 | 2.7e-08 | 1.5e-14 | 2.7e-08 | -1.4e-14 | +4.8e-03 | 2.4e-15 |
| tensor | NB, θ=3 | 10000 | 2.4e-08 | 1.3e-14 | 2.4e-08 | -4.3e-14 | +4.8e-03 | 3.4e-15 |

Readings:

- The discrete route reproduces pinned `bam` at rounding level for every
  family. The Gaussian n = 10,000 value of 8.3e-11 has one smoothing
  parameter near 2e8 (an almost linear factor-by level) and is the largest.
- The dense route agrees with the same model only to its own convergence
  tolerance (1e-9 to 6e-8). For Gaussian it is worse: 3.7e-7 at n = 10,000 and
  5.0e-7 for the tensor model. This is consistent with the published-code
  finding that dense fixed-sp Gaussian fits are not STRICT against an exact
  basis (likely normal-equations conditioning). A focused repro (n = 600)
  gives dense-versus-`bam` errors of
  1.65e-6, 1.65e-8 and 1.9e-10 on `y ~ fac + s(x2,bs='cr',k=8,by=fac)` at a
  factor-by `sp` of 1.9e8, 1.9e6 and 1.9e4: the error is proportional to `sp`.
  Over the same range the discrete route stays at 8e-11 to 7e-14 from `bam`,
  and pinned `gam` and `bam(m=n)` agree to 5e-15 at a shared `sp`. The study
  does not fix this; it is reported for the dense-route owner.
- EDF definitions differ between the two mgcv paths. These are method
  differences, not discretization:
  - Poisson: dense minus `bam` is +1.6e-4 to +3.7e-4. `bgam.fitd` reports the
    EDF of its last working model, whose weights come from the previous
    iterate. The harness's reconstruction at the final coefficients differs
    from the discrete fit's own EDF by the same amount (3–4e-6 relative).
  - Negative binomial: the gap is about 7e-3 (−6.4e-3 to +7.9e-3 in the study
    model, +4.8e-3 for the tensor model). Pinned `gam.fit4` reports
    diag(F) with F = (X'WX+S)⁻¹X'W_F X: Newton weights W inside the inverse
    and Fisher weights W_F outside. `bgam.fitd` uses Newton weights
    throughout. Dense JaxGAM follows `gam` here: its EDF matches pinned
    `gam` to within the two fits' REML optima.

The pinned `gam(method="REML")` anchor for the dense reference
(`RBridge.fit_gam`; not available for the fixed-theta negative binomial)
agrees in fitted values to between 3e-8 (Gaussian, n = 2000) and 2.7e-3
(tensor NB). Its relative deviance differences are at most 3.7e-6, which
is within the flat-REML-optimum differences AGENTS.md anticipates.

## Discretization effect

### Working-model REML against the dense fit, study model, n = 10,000

This is the realistic use: discrete fitting with the opt-in working-model
REML, compared with the undiscretized dense REML fit. At the `exact` row only
the smoothing strategy differs.

| family | m | max abs Δη | rms Δη | max abs Δη / SE | rel. Δ test deviance | Δ EDF | worst term (rel. L2) | SE ratio median [min, max] | curve coverage (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| Gaussian | 10 | 1.3e-01 | 3.0e-02 | 2.25 | +2.0e-03 | +0.36 | `ti(x0,x1)` 3.7e-01 | 0.996 [0.85, 1.03] | 0.31 |
| Gaussian | 30 | 2.5e-02 | 6.1e-03 | 0.48 | -7.3e-05 | -0.01 | `ti(x0,x1)` 7.8e-02 | 0.999 [0.96, 1.01] | 1.00 |
| Gaussian | 100 | 8.0e-03 | 2.0e-03 | 0.16 | +3.2e-04 | +0.00 | `ti(x0,x1)` 1.8e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Gaussian | 300 | 4.9e-03 | 1.3e-03 | 0.11 | -9.2e-05 | +0.01 | `ti(x0,x1)` 1.3e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Gaussian | 1000 (default) | 4.4e-03 | 1.2e-03 | 0.10 | -6.0e-05 | +0.01 | `ti(x0,x1)` 1.6e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Gaussian | 3000 | 3.0e-03 | 9.1e-04 | 0.08 | -5.9e-05 | +0.00 | `ti(x0,x1)` 1.2e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Gaussian | exact | 1.2e-06 | 3.2e-07 | 0.00 | -6.2e-08 | +0.00 | `s(x2):facb` 2.0e-06 | 1.000 [1.00, 1.00] | 1.00 |
| Poisson | 10 | 1.7e-01 | 3.4e-02 | 2.07 | -6.0e-04 | +0.79 | `ti(x0,x1)` 3.8e-01 | 0.978 [0.86, 1.01] | 0.55 |
| Poisson | 30 | 3.8e-02 | 8.5e-03 | 0.46 | +1.4e-04 | -0.20 | `ti(x0,x1)` 8.1e-02 | 0.995 [0.97, 1.01] | 1.00 |
| Poisson | 100 | 1.2e-02 | 2.3e-03 | 0.13 | -3.5e-06 | +0.02 | `ti(x0,x1)` 2.0e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Poisson | 300 | 5.1e-03 | 1.4e-03 | 0.07 | -3.8e-05 | +0.01 | `ti(x0,x1)` 1.3e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Poisson | 1000 (default) | 4.9e-03 | 1.2e-03 | 0.07 | -3.5e-05 | -0.02 | `ti(x0,x1)` 1.3e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Poisson | 3000 | 3.1e-03 | 8.9e-04 | 0.05 | -3.1e-05 | -0.02 | `ti(x0,x1)` 1.3e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Poisson | exact | 1.9e-04 | 4.1e-05 | 0.00 | +1.6e-06 | -0.01 | `s(x2):facb` 1.7e-04 | 1.000 [1.00, 1.00] | 1.00 |
| Binomial | 10 | 1.5e-01 | 4.0e-02 | 0.77 | -5.2e-04 | +0.74 | `ti(x0,x1)` 3.7e-01 | 0.985 [0.88, 1.00] | 0.99 |
| Binomial | 30 | 5.9e-02 | 1.3e-02 | 0.28 | -2.0e-04 | -0.19 | `ti(x0,x1)` 8.8e-02 | 0.996 [0.97, 1.00] | 1.00 |
| Binomial | 100 | 2.1e-02 | 3.4e-03 | 0.09 | +3.9e-05 | -0.08 | `ti(x0,x1)` 1.8e-02 | 0.999 [0.99, 1.00] | 1.00 |
| Binomial | 300 | 8.1e-03 | 1.9e-03 | 0.04 | -2.9e-05 | -0.10 | `ti(x0,x1)` 1.1e-02 | 0.999 [0.99, 1.00] | 1.00 |
| Binomial | 1000 (default) | 6.5e-03 | 1.5e-03 | 0.04 | +1.2e-05 | -0.08 | `ti(x0,x1)` 1.4e-02 | 0.999 [0.99, 1.00] | 1.00 |
| Binomial | 3000 | 5.4e-03 | 1.4e-03 | 0.03 | +1.2e-05 | -0.08 | `ti(x0,x1)` 1.4e-02 | 0.999 [0.99, 1.00] | 1.00 |
| Binomial | exact | 2.5e-03 | 6.4e-04 | 0.02 | -5.3e-06 | -0.10 | `ti(x0,x1)` 2.8e-03 | 0.999 [1.00, 1.00] | 1.00 |
| Gamma (log) | 10 | 1.4e-01 | 3.6e-02 | 2.10 | +3.4e-03 | +0.14 | `ti(x0,x1)` 3.8e-01 | 0.995 [0.85, 1.02] | 0.42 |
| Gamma (log) | 30 | 2.8e-02 | 7.2e-03 | 0.42 | +2.0e-04 | +0.04 | `ti(x0,x1)` 7.7e-02 | 0.999 [0.96, 1.01] | 1.00 |
| Gamma (log) | 100 | 1.1e-02 | 2.3e-03 | 0.16 | +7.1e-05 | -0.03 | `ti(x0,x1)` 1.9e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Gamma (log) | 300 | 6.7e-03 | 1.6e-03 | 0.13 | +1.3e-05 | -0.01 | `ti(x0,x1)` 1.3e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Gamma (log) | 1000 (default) | 6.0e-03 | 1.5e-03 | 0.12 | -3.4e-05 | +0.00 | `ti(x0,x1)` 1.6e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Gamma (log) | 3000 | 4.1e-03 | 9.9e-04 | 0.08 | -1.3e-05 | +0.01 | `ti(x0,x1)` 1.3e-02 | 1.000 [0.99, 1.01] | 1.00 |
| Gamma (log) | exact | 1.6e-04 | 3.7e-05 | 0.00 | -1.6e-06 | +0.01 | `s(x2):facb` 1.7e-04 | 1.000 [1.00, 1.00] | 1.00 |
| NB, est. θ | 10 | 1.7e-01 | 3.7e-02 | 2.08 | -5.3e-03 | +2.99 | `ti(x0,x1)` 3.8e-01 | 0.994 [0.88, 1.20] | 0.56 |
| NB, est. θ | 30 | 3.6e-02 | 7.6e-03 | 0.31 | +4.9e-03 | +0.06 | `ti(x0,x1)` 8.3e-02 | 0.994 [0.96, 1.01] | 1.00 |
| NB, est. θ | 100 | 1.3e-02 | 2.8e-03 | 0.13 | +5.2e-03 | -0.01 | `ti(x0,x1)` 1.7e-02 | 0.997 [0.98, 1.01] | 1.00 |
| NB, est. θ | 300 | 7.0e-03 | 1.7e-03 | 0.10 | +5.4e-03 | +0.01 | `ti(x0,x1)` 1.0e-02 | 0.997 [0.98, 1.01] | 1.00 |
| NB, est. θ | 1000 (default) | 6.2e-03 | 1.4e-03 | 0.08 | +5.2e-03 | -0.00 | `ti(x0,x1)` 1.3e-02 | 0.997 [0.98, 1.01] | 1.00 |
| NB, est. θ | 3000 | 4.3e-03 | 9.9e-04 | 0.05 | +5.2e-03 | -0.00 | `ti(x0,x1)` 1.2e-02 | 0.997 [0.98, 1.01] | 1.00 |
| NB, est. θ | exact | 1.1e-03 | 2.4e-04 | 0.01 | +5.2e-03 | +0.01 | `ti(x0,x1)` 7.6e-04 | 0.997 [0.99, 1.01] | 1.00 |
| NB, θ=3 | 10 | 1.7e-01 | 3.7e-02 | 2.07 | +1.6e-03 | +3.06 | `ti(x0,x1)` 3.8e-01 | 0.990 [0.88, 1.20] | 0.56 |
| NB, θ=3 | 30 | 3.6e-02 | 7.5e-03 | 0.31 | -7.9e-06 | +0.02 | `ti(x0,x1)` 8.3e-02 | 0.998 [0.96, 1.01] | 1.00 |
| NB, θ=3 | 100 | 1.3e-02 | 2.8e-03 | 0.12 | -1.1e-04 | -0.05 | `ti(x0,x1)` 1.7e-02 | 1.000 [0.99, 1.01] | 1.00 |
| NB, θ=3 | 300 | 6.9e-03 | 1.7e-03 | 0.09 | +4.3e-05 | -0.04 | `ti(x0,x1)` 1.0e-02 | 1.000 [0.99, 1.01] | 1.00 |
| NB, θ=3 | 1000 (default) | 6.1e-03 | 1.4e-03 | 0.08 | -2.8e-05 | -0.05 | `ti(x0,x1)` 1.3e-02 | 1.000 [0.99, 1.01] | 1.00 |
| NB, θ=3 | 3000 | 4.2e-03 | 9.9e-04 | 0.05 | -1.7e-05 | -0.04 | `ti(x0,x1)` 1.2e-02 | 1.000 [0.99, 1.01] | 1.00 |
| NB, θ=3 | exact | 2.8e-04 | 5.9e-05 | 0.00 | -5.1e-06 | -0.03 | `ti(x0,x1)` 2.2e-04 | 1.000 [0.99, 1.01] | 1.00 |

The row-level coverage of the dense estimate by the discrete 95% interval
is 1.000 at every m >= 30 and 0.991–1.000 at m = 10. Here, coverage of the
true η by the discrete intervals is within 0.005 of the dense fit's at
m >= 30; across all cases it is within 0.016 at m >= 30 and 0.006 at
m >= 100. At m = 10 it falls from 0.971 to 0.916 (Gaussian) and from 0.927
to 0.872 (Gamma) at n = 10,000, and from 0.960 to 0.731 for Gaussian at
n = 50,000.

### Strategies side by side

Each cell gives max |Δη| / SE and the total-EDF difference. `fixed_sp` and
`fixed_sp_matched` are compared with the dense fit at the same `sp` (and
theta). `working_model_reml` is compared with itself at the exact level.

n = 10,000:

| family | m | fixed_sp: max abs Δη / SE, Δ EDF | fixed_sp_matched: max abs Δη / SE, Δ EDF | working_model_reml: max abs Δη / SE, Δ EDF |
| --- | --- | --- | --- | --- |
| Gaussian | 10 | 2.32, +2.59 | 2.26, +1.28 | 2.25, +0.36 |
| Gaussian | 100 | 0.21, +0.68 | 0.15, +0.02 | 0.16, +0.00 |
| Gaussian | 1000 (default) | 0.10, +0.15 | 0.10, -0.00 | 0.10, +0.01 |
| Gaussian | 3000 | 0.08, -0.07 | 0.07, +0.01 | 0.08, +0.00 |
| Poisson | 10 | 2.05, +3.73 | 2.03, +1.55 | 2.07, +0.81 |
| Poisson | 100 | 0.29, +0.99 | 0.13, +0.01 | 0.13, +0.03 |
| Poisson | 1000 (default) | 0.08, +0.20 | 0.07, -0.02 | 0.07, -0.01 |
| Poisson | 3000 | 0.08, -0.11 | 0.05, -0.01 | 0.05, -0.00 |
| Binomial | 10 | 0.78, +4.86 | 0.72, +1.53 | 0.77, +0.83 |
| Binomial | 100 | 0.25, +1.24 | 0.09, +0.05 | 0.09, +0.02 |
| Binomial | 1000 (default) | 0.07, +0.26 | 0.04, +0.02 | 0.04, +0.02 |
| Binomial | 3000 | 0.05, -0.11 | 0.03, +0.02 | 0.03, +0.02 |
| Gamma (log) | 10 | 2.05, +3.24 | 2.02, +1.58 | 2.10, +0.12 |
| Gamma (log) | 100 | 0.23, +0.81 | 0.15, +0.02 | 0.16, -0.04 |
| Gamma (log) | 1000 (default) | 0.11, +0.17 | 0.11, -0.01 | 0.12, -0.01 |
| Gamma (log) | 3000 | 0.08, -0.08 | 0.08, +0.00 | 0.08, -0.00 |
| NB, est. θ | 10 | 1.42, +4.14 | 1.39, +1.65 | 2.11, +2.98 |
| NB, est. θ | 100 | 0.22, +1.12 | 0.13, +0.01 | 0.12, -0.02 |
| NB, est. θ | 1000 (default) | 0.08, +0.23 | 0.07, -0.02 | 0.08, -0.01 |
| NB, est. θ | 3000 | 0.06, -0.13 | 0.05, -0.00 | 0.05, -0.01 |
| NB, θ=3 | 10 | 1.40, +4.16 | 1.38, +1.65 | 2.08, +3.10 |
| NB, θ=3 | 100 | 0.22, +1.13 | 0.13, +0.02 | 0.12, -0.02 |
| NB, θ=3 | 1000 (default) | 0.08, +0.23 | 0.07, -0.02 | 0.08, -0.01 |
| NB, θ=3 | 3000 | 0.06, -0.13 | 0.05, -0.00 | 0.05, -0.01 |

n = 2000 (3000 rounds nothing at this size):

| family | m | fixed_sp: max abs Δη / SE, Δ EDF | fixed_sp_matched: max abs Δη / SE, Δ EDF | working_model_reml: max abs Δη / SE, Δ EDF |
| --- | --- | --- | --- | --- |
| Gaussian | 10 | 1.37, +4.00 | 1.43, +1.72 | 1.52, +1.07 |
| Gaussian | 100 | 0.19, +0.82 | 0.11, +0.04 | 0.12, -0.01 |
| Gaussian | 1000 (default) | 0.11, -0.32 | 0.07, -0.00 | 0.07, +0.06 |
| Gaussian | 3000 | 0.00, +0.00 | 0.00, +0.00 | 0.00, +0.00 |
| Poisson | 10 | 1.25, +4.14 | 1.28, +1.50 | 1.26, +0.79 |
| Poisson | 100 | 0.25, +0.91 | 0.10, +0.04 | 0.11, +0.02 |
| Poisson | 1000 (default) | 0.08, -0.28 | 0.04, -0.01 | 0.03, +0.01 |
| Poisson | 3000 | 0.00, -0.00 | 0.00, -0.00 | 0.00, +0.00 |
| Binomial | 10 | 0.62, +2.31 | 0.53, +0.47 | 0.66, +0.89 |
| Binomial | 100 | 0.09, +0.33 | 0.05, +0.02 | 0.07, -0.15 |
| Binomial | 1000 (default) | 0.05, +0.02 | 0.02, +0.02 | 0.01, +0.02 |
| Binomial | 3000 | 0.00, -0.00 | 0.00, -0.00 | 0.00, +0.00 |
| Gamma (log) | 10 | 1.00, +4.18 | 1.03, +1.52 | 1.08, +0.33 |
| Gamma (log) | 100 | 0.19, +0.93 | 0.10, +0.07 | 0.10, +0.01 |
| Gamma (log) | 1000 (default) | 0.07, -0.22 | 0.05, +0.04 | 0.04, +0.01 |
| Gamma (log) | 3000 | 0.00, +0.00 | 0.00, +0.00 | 0.00, +0.00 |
| NB, est. θ | 10 | 0.73, +3.45 | 0.81, +1.49 | 0.93, +0.41 |
| NB, est. θ | 100 | 0.22, +0.92 | 0.08, +0.06 | 0.08, +0.00 |
| NB, est. θ | 1000 (default) | 0.08, -0.24 | 0.04, +0.01 | 0.04, -0.00 |
| NB, est. θ | 3000 | 0.00, +0.01 | 0.00, +0.01 | 0.00, +0.00 |
| NB, θ=3 | 10 | 0.71, +3.45 | 0.79, +1.48 | 0.88, +0.64 |
| NB, θ=3 | 100 | 0.21, +0.92 | 0.08, +0.06 | 0.07, +0.01 |
| NB, θ=3 | 1000 (default) | 0.08, -0.24 | 0.04, +0.01 | 0.04, +0.00 |
| NB, θ=3 | 3000 | 0.00, +0.01 | 0.00, +0.01 | 0.00, +0.00 |

n = 50,000:

| family | m | fixed_sp: max abs Δη / SE, Δ EDF | fixed_sp_matched: max abs Δη / SE, Δ EDF | working_model_reml: max abs Δη / SE, Δ EDF |
| --- | --- | --- | --- | --- |
| Gaussian | 10 | 5.34, +1.10 | 5.37, +0.56 | 5.36, +1.91 |
| Gaussian | 100 | 0.19, +0.36 | 0.18, +0.02 | 0.18, +0.00 |
| Gaussian | 1000 (default) | 0.06, +0.21 | 0.06, +0.01 | 0.06, +0.00 |
| Gaussian | 3000 | 0.05, +0.16 | 0.05, +0.01 | 0.05, +0.00 |
| Poisson | 10 | 3.32, +1.98 | 3.33, +0.95 | 3.32, +0.62 |
| Poisson | 100 | 0.17, +0.61 | 0.13, +0.02 | 0.14, -0.01 |
| Poisson | 1000 (default) | 0.10, +0.35 | 0.05, +0.00 | 0.04, -0.01 |
| Poisson | 3000 | 0.09, +0.26 | 0.05, +0.00 | 0.05, -0.01 |

### Tensor model (`te` + factor-by), n = 10,000

| family | m | max abs Δη | rms Δη | max abs Δη / SE | rel. Δ test deviance | Δ EDF | worst term (rel. L2) | SE ratio median [min, max] | curve coverage (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| Gaussian | 10 | 1.2e-01 | 3.4e-02 | 2.70 | +9.4e-04 | +0.19 | `te(x0,x1)` 6.6e-02 | 0.990 [0.81, 1.05] | 0.92 |
| Gaussian | 30 | 2.7e-02 | 6.3e-03 | 0.61 | -4.5e-04 | +0.06 | `s(x2):faca` 1.5e-02 | 0.998 [0.95, 1.03] | 1.00 |
| Gaussian | 100 | 6.9e-03 | 2.0e-03 | 0.18 | +1.9e-04 | -0.00 | `te(x0,x1)` 3.9e-03 | 1.000 [0.97, 1.03] | 1.00 |
| Gaussian | 300 | 5.5e-03 | 1.4e-03 | 0.13 | -8.7e-05 | -0.01 | `te(x0,x1)` 2.7e-03 | 1.000 [0.97, 1.03] | 1.00 |
| Gaussian | 1000 (default) | 5.4e-03 | 1.3e-03 | 0.13 | -8.9e-05 | -0.01 | `te(x0,x1)` 2.6e-03 | 1.000 [0.98, 1.03] | 1.00 |
| Gaussian | 3000 | 4.2e-03 | 9.8e-04 | 0.10 | -1.2e-04 | -0.01 | `te(x0,x1)` 1.9e-03 | 1.000 [0.98, 1.02] | 1.00 |
| Gaussian | exact | 8.0e-06 | 2.2e-06 | 0.00 | -4.4e-07 | +0.00 | `s(x2):facb` 1.4e-05 | 1.000 [1.00, 1.00] | 1.00 |
| Poisson | 10 | 1.4e-01 | 3.8e-02 | 2.48 | +2.3e-03 | +1.45 | `te(x0,x1)` 7.8e-02 | 0.977 [0.83, 1.09] | 0.96 |
| Poisson | 30 | 2.7e-02 | 7.0e-03 | 0.43 | +2.2e-04 | +0.08 | `te(x0,x1)` 1.4e-02 | 0.996 [0.96, 1.03] | 1.00 |
| Poisson | 100 | 7.7e-03 | 2.3e-03 | 0.14 | -9.4e-05 | -0.06 | `te(x0,x1)` 4.4e-03 | 0.999 [0.97, 1.02] | 1.00 |
| Poisson | 300 | 6.2e-03 | 1.6e-03 | 0.10 | +1.1e-04 | -0.06 | `te(x0,x1)` 2.9e-03 | 1.000 [0.98, 1.02] | 1.00 |
| Poisson | 1000 (default) | 5.2e-03 | 1.4e-03 | 0.09 | +8.4e-05 | -0.06 | `te(x0,x1)` 2.6e-03 | 1.000 [0.98, 1.02] | 1.00 |
| Poisson | 3000 | 4.3e-03 | 1.2e-03 | 0.08 | +7.4e-05 | -0.06 | `s(x2):facc` 2.5e-03 | 1.000 [0.98, 1.02] | 1.00 |
| Poisson | exact | 4.3e-04 | 1.0e-04 | 0.01 | +3.2e-06 | -0.03 | `s(x2):facb` 6.4e-04 | 1.000 [0.99, 1.00] | 1.00 |
| Binomial | 10 | 1.5e-01 | 4.5e-02 | 0.99 | -1.0e-03 | +1.07 | `te(x0,x1)` 9.1e-02 | 0.982 [0.86, 1.02] | 1.00 |
| Binomial | 30 | 4.8e-02 | 1.0e-02 | 0.26 | -1.7e-04 | -0.00 | `s(x2):faca` 2.1e-02 | 0.996 [0.97, 1.02] | 1.00 |
| Binomial | 100 | 1.7e-02 | 3.2e-03 | 0.09 | +4.3e-05 | -0.03 | `s(x2):faca` 8.3e-03 | 0.999 [0.98, 1.02] | 1.00 |
| Binomial | 300 | 5.8e-03 | 1.8e-03 | 0.05 | -2.6e-05 | -0.06 | `te(x0,x1)` 3.4e-03 | 0.999 [0.99, 1.02] | 1.00 |
| Binomial | 1000 (default) | 4.3e-03 | 1.4e-03 | 0.04 | -2.7e-06 | -0.04 | `te(x0,x1)` 2.9e-03 | 1.000 [0.99, 1.02] | 1.00 |
| Binomial | 3000 | 3.5e-03 | 1.1e-03 | 0.04 | -1.1e-05 | -0.03 | `te(x0,x1)` 2.3e-03 | 1.000 [0.99, 1.01] | 1.00 |
| Binomial | exact | 8.1e-04 | 1.7e-04 | 0.00 | -4.4e-06 | -0.04 | `s(x2):faca` 3.6e-04 | 0.999 [1.00, 1.00] | 1.00 |
| Gamma (log) | 10 | 1.5e-01 | 4.0e-02 | 2.34 | +3.8e-03 | +0.15 | `te(x0,x1)` 7.7e-02 | 0.989 [0.83, 1.04] | 0.96 |
| Gamma (log) | 30 | 2.0e-02 | 6.2e-03 | 0.48 | +8.7e-04 | +0.13 | `te(x0,x1)` 1.2e-02 | 0.998 [0.95, 1.03] | 1.00 |
| Gamma (log) | 100 | 9.3e-03 | 2.4e-03 | 0.22 | +5.2e-05 | -0.01 | `te(x0,x1)` 4.6e-03 | 1.000 [0.98, 1.02] | 1.00 |
| Gamma (log) | 300 | 5.5e-03 | 1.6e-03 | 0.13 | +7.7e-05 | +0.02 | `te(x0,x1)` 2.9e-03 | 1.001 [0.98, 1.03] | 1.00 |
| Gamma (log) | 1000 (default) | 5.3e-03 | 1.4e-03 | 0.13 | +9.0e-05 | +0.02 | `te(x0,x1)` 2.6e-03 | 1.001 [0.98, 1.03] | 1.00 |
| Gamma (log) | 3000 | 4.1e-03 | 1.1e-03 | 0.10 | +6.4e-05 | +0.02 | `te(x0,x1)` 2.0e-03 | 1.001 [0.98, 1.02] | 1.00 |
| Gamma (log) | exact | 2.1e-04 | 4.3e-05 | 0.00 | +2.5e-06 | +0.02 | `s(x2):facb` 2.3e-04 | 1.000 [1.00, 1.00] | 1.00 |
| NB, est. θ | 10 | 1.4e-01 | 4.7e-02 | 1.79 | -1.2e-02 | +0.65 | `te(x0,x1)` 9.8e-02 | 0.978 [0.85, 1.03] | 0.98 |
| NB, est. θ | 30 | 3.5e-02 | 8.2e-03 | 0.44 | +1.9e-03 | -0.18 | `s(x2):faca` 1.9e-02 | 0.992 [0.95, 1.02] | 1.00 |
| NB, est. θ | 100 | 1.1e-02 | 2.7e-03 | 0.15 | +3.8e-03 | -0.04 | `te(x0,x1)` 5.2e-03 | 0.996 [0.97, 1.02] | 1.00 |
| NB, est. θ | 300 | 5.9e-03 | 1.7e-03 | 0.13 | +3.8e-03 | -0.06 | `s(x2):facb` 4.9e-03 | 0.997 [0.97, 1.02] | 1.00 |
| NB, est. θ | 1000 (default) | 5.8e-03 | 1.7e-03 | 0.12 | +3.8e-03 | -0.07 | `s(x2):facb` 4.8e-03 | 0.997 [0.97, 1.02] | 1.00 |
| NB, est. θ | 3000 | 4.6e-03 | 1.3e-03 | 0.09 | +3.8e-03 | -0.07 | `s(x2):facb` 4.8e-03 | 0.997 [0.97, 1.02] | 1.00 |
| NB, est. θ | exact | 2.9e-03 | 7.7e-04 | 0.05 | +3.7e-03 | -0.06 | `s(x2):facb` 4.9e-03 | 0.997 [0.97, 1.01] | 1.00 |
| NB, θ=3 | 10 | 1.4e-01 | 4.7e-02 | 1.78 | +2.9e-03 | +0.77 | `te(x0,x1)` 9.7e-02 | 0.970 [0.84, 1.02] | 0.98 |
| NB, θ=3 | 30 | 3.5e-02 | 8.2e-03 | 0.43 | +2.4e-04 | -0.20 | `s(x2):faca` 1.9e-02 | 0.993 [0.95, 1.02] | 0.99 |
| NB, θ=3 | 100 | 1.1e-02 | 2.8e-03 | 0.15 | +2.2e-05 | -0.09 | `te(x0,x1)` 5.3e-03 | 0.998 [0.97, 1.02] | 1.00 |
| NB, θ=3 | 300 | 5.9e-03 | 1.7e-03 | 0.13 | +2.4e-05 | -0.10 | `s(x2):facb` 5.1e-03 | 0.999 [0.97, 1.02] | 1.00 |
| NB, θ=3 | 1000 (default) | 5.9e-03 | 1.7e-03 | 0.12 | +7.4e-05 | -0.11 | `s(x2):facb` 5.1e-03 | 0.999 [0.97, 1.02] | 1.00 |
| NB, θ=3 | 3000 | 4.6e-03 | 1.3e-03 | 0.09 | +3.3e-05 | -0.11 | `s(x2):facb` 5.1e-03 | 0.999 [0.97, 1.02] | 1.00 |
| NB, θ=3 | exact | 2.7e-03 | 7.9e-04 | 0.05 | -2.4e-06 | -0.10 | `s(x2):facb` 5.0e-03 | 0.999 [0.98, 1.01] | 1.00 |

Strategies side by side:

| family | m | fixed_sp: max abs Δη / SE, Δ EDF | fixed_sp_matched: max abs Δη / SE, Δ EDF | working_model_reml: max abs Δη / SE, Δ EDF |
| --- | --- | --- | --- | --- |
| Gaussian | 10 | 2.64, +1.44 | 2.66, +0.86 | 2.70, +0.19 |
| Gaussian | 100 | 0.19, +0.08 | 0.18, +0.01 | 0.18, -0.00 |
| Gaussian | 1000 (default) | 0.13, +0.01 | 0.13, -0.00 | 0.13, -0.02 |
| Gaussian | 3000 | 0.11, +0.03 | 0.10, +0.01 | 0.10, -0.01 |
| Poisson | 10 | 2.41, +2.18 | 2.38, +1.20 | 2.49, +1.48 |
| Poisson | 100 | 0.13, +0.14 | 0.13, -0.00 | 0.13, -0.04 |
| Poisson | 1000 (default) | 0.09, +0.02 | 0.09, -0.02 | 0.09, -0.03 |
| Poisson | 3000 | 0.08, +0.05 | 0.08, -0.00 | 0.08, -0.03 |
| Binomial | 10 | 1.02, +2.87 | 0.98, +1.33 | 0.99, +1.11 |
| Binomial | 100 | 0.10, +0.25 | 0.09, +0.03 | 0.09, +0.02 |
| Binomial | 1000 (default) | 0.05, +0.09 | 0.04, +0.02 | 0.04, +0.01 |
| Binomial | 3000 | 0.05, +0.12 | 0.04, +0.04 | 0.04, +0.01 |
| Gamma (log) | 10 | 2.33, +2.08 | 2.32, +1.18 | 2.34, +0.13 |
| Gamma (log) | 100 | 0.24, +0.12 | 0.22, +0.00 | 0.22, -0.03 |
| Gamma (log) | 1000 (default) | 0.14, +0.01 | 0.13, -0.01 | 0.13, +0.00 |
| Gamma (log) | 3000 | 0.11, +0.04 | 0.10, +0.01 | 0.10, +0.00 |
| NB, est. θ | 10 | 1.75, +2.50 | 1.75, +1.32 | 1.80, +0.71 |
| NB, est. θ | 100 | 0.14, +0.17 | 0.15, +0.00 | 0.16, +0.03 |
| NB, est. θ | 1000 (default) | 0.09, +0.03 | 0.09, -0.02 | 0.09, -0.01 |
| NB, est. θ | 3000 | 0.06, +0.07 | 0.06, +0.00 | 0.06, -0.00 |
| NB, θ=3 | 10 | 1.74, +2.50 | 1.75, +1.33 | 1.79, +0.87 |
| NB, θ=3 | 100 | 0.14, +0.17 | 0.15, +0.00 | 0.15, +0.01 |
| NB, θ=3 | 1000 (default) | 0.09, +0.03 | 0.09, -0.02 | 0.09, -0.01 |
| NB, θ=3 | 3000 | 0.06, +0.07 | 0.06, +0.00 | 0.06, -0.01 |

### Why the error flattens: the basis change

Study model, Gaussian, n = 10,000. The pattern is the same for every family,
because it comes from Phase-1 setup:

| m | max covariate error / range | max knot move / range | penalty-norm ratios (min, max) |
| --- | ---: | ---: | --- |
| 10 | 5.6e-02 | 1.1e-02 | 0.37, 1.00 |
| 30 | 1.7e-02 | 1.1e-02 | 0.69, 1.00 |
| 100 | 5.0e-03 | 1.1e-02 | 0.64, 1.02 |
| 300 | 1.7e-03 | 1.1e-02 | 0.72, 1.02 |
| 1000 (default) | 5.0e-04 | 1.1e-02 | 0.92, 1.02 |
| 3000 | 1.7e-04 | 8.5e-03 | 0.95, 1.09 |
| exact | 0.0e+00 | 0.0e+00 | 1.00, 1.00 |

- **Knots.** Cubic regression knots are placed at equally spaced ranks of the
  unique values (`place.knots`). After rounding, those unique values are the
  occupied grid points, so the knots become almost evenly spaced. Without
  rounding they sit at sample quantiles. The displacement is 1–1.5% of the
  range at every rounded level and does not shrink with m. It does shrink
  with n (1.4–1.5% at n = 2000, 0.85–1.1% at 10,000, 0.6% at 50,000),
  because the sample quantiles approach even spacing.
- **Penalty normalization.** mgcv rescales each penalty by
  ‖X‖∞²/‖S‖₁ on the frame used for setup. For `bam(discrete=TRUE)` that is
  the padded compact frame, so the ratios differ from 1 at every rounded
  level. The `ti` penalties change the most.

The rms error in η falls from 3.0e-2 (m = 10) through 6.1e-3 (m = 30) to
2.0e-3 (m = 100) for Gaussian working-model REML. That is a factor of 15
over a tenfold refinement, faster than linear in the grid spacing at first.
It then levels off at 0.9–1.3e-3 for m = 300–3000. At n = 2000, where the
3000 level rounds nothing, the plateau vanishes at once. Interval width is
almost unaffected: against the exact level, median SE ratios are within
0.994–1.001 for m >= 30 in every case (0.998–1.001 for m >= 100).

### Strategy difference at the exact level

Working-model REML against dense nested REML with nothing rounded, study
model, n = 10,000:

| family | max abs Δη | max abs Δη / SE | Δ EDF | θ: working / dense | rel. Δ test deviance |
| --- | ---: | ---: | ---: | --- | ---: |
| Gaussian | 1.2e-06 | 0.000 | +0.00 | – | -6.2e-08 |
| Poisson | 1.9e-04 | 0.002 | -0.01 | – | +1.6e-06 |
| Binomial | 2.5e-03 | 0.015 | -0.10 | – | -5.3e-06 |
| Gamma (log) | 1.6e-04 | 0.003 | +0.01 | – | -1.6e-06 |
| NB, est. θ | 1.1e-03 | 0.010 | +0.01 | 3.177 / 3.111 | +5.2e-03 |
| NB, θ=3 | 2.8e-04 | 0.003 | -0.03 | – | -5.1e-06 |

For Gaussian the two criteria coincide; the residual is the dense-route
error above. For the other families the performance iteration selects
slightly different `sp`. The effect is at most 0.05 SE in held-out η and
0.24 in total EDF (NB, est. θ, n = 2000). For NB with estimated theta the
held-out deviance difference (+5.2e-3 here, +2.3e-2 at n = 2000) mostly
reflects the different theta: deviance depends on theta, so these values
are not a comparison of means. With theta fixed the deviance difference is
5e-6 here and 1.2e-4 at n = 2000.

## Interpretation

- Discretization error in this study is a method error with two parts.
  - Covariate rounding dominates at coarse grids and shrinks quickly with m.
  - The compact-frame basis (knots and penalty normalization) leaves a floor
    of about 1e-3 in η, roughly 0.03–0.13 SE at n = 10,000. That floor stays
    until no covariate is rounded and shrinks slowly with n.
- Neither part is an implementation error. The exact level and the pinned
  `bam` anchors put the implementation at 1e-14 for the non-Gaussian
  families.
- **Negligible:** m >= 100 per margin in these designs, including mgcv's
  default.
  - Held-out η moves by at most 0.22 SE (default: 0.01–0.13 SE).
  - Held-out deviance moves by at most 6.9e-4 relative (default: 2.6e-4).
  - Total EDF moves by at most 0.15 under working-model REML (default:
    0.06).
  - Pointwise SE ratios stay within 0.97–1.03, and interval coverage of the
    undiscretized fit is complete.
- **Material:**
  - At m = 10 in every family: 0.8–2.7 SE at n = 10,000 and up to 5.4 SE
    at n = 50,000, rising with n because the bias does not shrink while the
    SEs do.
  - At m = 30, smaller but visible: 0.18–0.89 SE, with the study model's
    `ti` interaction off by about 8% in shape.
  - In the study model the `ti` interaction is the most sensitive term at
    every rounded level. In the tensor model the `te` surface and the
    factor-by curves share that role.
- **Fixed `sp` across discretization levels is not a like-for-like
  comparison in mgcv's parameterization.** Users who fix `sp` should expect
  the EDF shifts in the `fixed_sp` columns, and should match penalty norms
  if they want the same effective penalty. Working-model REML adapts
  automatically.
- **Strategy (performance iteration versus nested REML)** and the
  **negative binomial theta update** are separate, small method differences.
  They should not be attributed to discretization.

## Limitations

- **Design scope.** One synthetic design family with uniform continuous
  covariates and smooth truths. Strongly skewed covariates, heavy duplicates
  (which need no rounding), sharp features below the grid spacing, and
  high-dimensional joint groups (`s(x,z)` thin-plate, 25 points per variable
  by default) were not studied. Results are for `cr` bases with modest k; tp
  and gp margins were not run.
- **Prediction policy.** Prediction uses the default `continuous_frozen_basis`
  policy, which evaluates new values exactly. The mgcv-compatible
  `mgcv_predict_bamd` policy re-compresses each prediction request, which
  adds prediction-time rounding (1000 points per margin by default). That was
  not measured.
- **Sample size.** One seed per configuration, so there is no Monte Carlo
  spread. Truth coverage is reported for context, not as a calibration
  claim.
- **Platform.** Host-only: macOS arm64 with Accelerate NumPy, and host R with
  Accelerate. The implementation anchors are not the pinned Linux oracle.
  The discretization metrics themselves are far above platform rounding.
- **Not a runtime study.** Fit times include cold compilation in a fresh
  child on a loaded host. The runtime study belongs in
  `scripts/benchmark_scaling.py` after the PR9 port.
- **EDF reconstruction.** Per-term EDF for discrete fits is a harness
  reconstruction. It matches each fit's total to about 4e-6 relative for
  iterated families (see above); per-term values carry the same lag.

## Reproduce

From a clean checkout containing the harness (with a shared environment
whose editable install points elsewhere, also set `PYTHONPATH` to the
checkout):

```sh
make benchmark-discrete-approximation OUTPUT=/path/outside/study.json \
  ARGS='--models study --families gaussian,poisson,binomial,gamma,nb,nb_fixed --n 2000,10000'
make benchmark-discrete-approximation OUTPUT=/path/outside/tensor.json \
  ARGS='--models tensor --families gaussian,poisson,binomial,gamma,nb,nb_fixed --n 10000'
make benchmark-discrete-approximation OUTPUT=/path/outside/study-50k.json \
  ARGS='--models study --families gaussian,poisson --n 50000 --r-reference-max-n 50000'
uv run python scripts/benchmark_discrete_approximation.py --summarize /path/outside/study.json
make test-discrete-approximation-harness
```

R anchors run when pinned R 4.5.2 with mgcv 1.9-3 is available through rpy2
and n is at most `--r-reference-max-n` (default 20,000). Otherwise the
record says why they were skipped.
