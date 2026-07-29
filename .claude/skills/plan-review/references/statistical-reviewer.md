# Statistical pass — checklist

The second of the two inline passes in SKILL.md §5. Review as a statistician:
JaxGAM is a Python port of Simon Wood's R package `mgcv`, and the question here
is whether the code computes the right quantity, by a numerically sound route,
matching mgcv.

Scope this pass to the **mathematics and statistics** — the engineering pass
already covered style, structure, and test mechanics, so do not re-litigate
them. Read-only, as before: inspect and report, never edit, never commit.

## Ground truth, in order of authority

1. **The mgcv R/C source.** `echo $MGCV_SOURCE` for the clone; `docs/R_SOURCE_MAP.md`
   maps task → R file/function. **Read the R implementation of anything the
   diff touches before judging it.** The R code is ground truth for edge cases
   and algorithmic choices the design doc does not capture. Key files:
   `R/smooth.r` (basis + penalty constructors, `smooth.construct.XX.smooth.spec`),
   `R/gam.fit3.r` (PIRLS, working weights, step-halving, convergence),
   `R/fast-REML.r` (REML/ML criterion, Newton optimizer for λ),
   `R/gam.r` (`gam()`, `predict.gam()`, `summary.gam()`, `gam.side()`),
   `R/families.r` (variance, `dev.resids`, `initialize`, `validmu`),
   `src/tprs.c` (TPRS eigendecomposition + knot selection),
   `src/gdi.c` (REML derivatives).
2. `docs/design.md` — the authoritative architecture spec. Use the topic→section
   table in `CLAUDE.md`; read only the cited sections.
3. The feature `design.md` and the plan task in the shared context.

Where the diff disagrees with R, R wins — unless the design doc documents a
deliberate, justified departure. Quote the R function and line when you flag
a divergence.

## What to check

### Correctness of the quantity
- Is this the estimator the docs specify, or something that merely resembles
  it? Check the algebra term by term against R.
- Likelihood, deviance, and scale: correct family parameterization, correct
  saturated-model term, correct scale estimate (known vs estimated), correct
  dispersion handling. Deviance must be non-negative.
- Link functions and their inverses/derivatives: consistency of `eta` ↔ `mu`,
  `mu.eta`, and the working weights derived from them. Canonical vs
  non-canonical link handling.
- REML/ML criterion: the log|S_λ|₊ generalized-determinant term, the correct
  null-space dimension `Mp`, log|X'WX + S_λ|, the `+p/2·log(2π)` style constants
  — and whether a constant that cancels in optimization has been dropped
  somewhere it must not be (any place the criterion value itself is compared,
  reported, or differenced across models).
- EDF: `tr(F)` with `F = (X'WX + S_λ)⁻¹X'WX`, per-term partitioning, and the
  bound `0 ≤ edf ≤ k`. Check which EDF variant the code uses where mgcv uses a
  different one (edf vs edf1 vs edf2 in p-values).
- Covariance: `Vp` (Bayesian, penalized) vs `Vb`/frequentist — mixing them
  silently is a real and common bug. Check the scale multiplier.
- Smoothing parameter parameterization: is λ on the log scale where the
  optimizer expects it? Are gradients w.r.t. `log λ`?

### Basis and penalty construction
- Knot placement must match R's algorithm exactly (max-min distance for TPRS,
  quantile-based for cubic). Different knots ⇒ nothing downstream matches.
- TPRS eigendecomposition: eigenvalue truncation rule, ordering, null-space
  handling, sign conventions.
- Penalty matrices: symmetric, PSD, correct rank, correct null space.
  Reparameterization/rescaling of `S` must be undone consistently wherever the
  penalty is read back.
- Identifiability: sum-to-zero constraint absorption, and its interaction with
  factor-`by` smooths when `s(x, by=fac)` coexists with `s(x)` (design §5.7.3).
  Check the `CoefficientMap` roundtrip — constrained coefficients must map back
  to the original basis, or `predict()` silently returns garbage on a correct
  fit.
- Tensor products: Kronecker ordering of margins, and one penalty per margin
  with the right identity padding on the other margins.

### Numerical soundness
- Solving via a factorization (Cholesky/QR/SVD) rather than an explicit inverse.
- Behavior at rank deficiency and near-collinearity; the jitter/ridge strategy
  in design §4.8 applied where R applies it.
- Catastrophic cancellation: differences of large nearly-equal quantities,
  `log(1+x)` / `exp(x)-1` written without `log1p`/`expm1`, `log(sum(exp))`
  without a max-shift.
- Overflow/underflow at extreme `eta` — logit near 0/1, log link near 0, Gamma
  and NB at small `mu`.
- λ at both extremes: `λ → 0` (unpenalized, possibly rank-deficient) and
  `λ → ∞` (collapse to the null space). Both must be finite and correct.
- PIRLS: correct working response `z` and weights `W`, **step-halving present**
  (without it Binomial and Gamma diverge — this is not optional), a convergence
  criterion on penalized deviance matching R's, a finite iteration cap, and
  monotone descent of the objective.
- Under `jax.jit`: no Python branching on traced values; `jax.lax.while_loop` /
  `cond` used instead. A JIT-invisible branch silently freezes one path.
- `jax.grad` through any custom reparameterization — check the derivative is
  actually what `src/gdi.c` computes, not a plausible-looking substitute.

### Edge cases and failure modes to actively test in your head
n = 1; p > n; a smooth with k larger than the number of unique x values;
duplicated x; constant predictor; a `by` factor level with zero observations;
all-zero or all-one Binomial responses; zero counts under Poisson/NB with a log
link; weights of zero; missing values; a single-observation group. State the
input and the wrong result it produces.

### Statistical hard gates (`docs/design.md` §18.1)
These block CI and must hold: objective monotonicity, `H` symmetric and PSD,
penalty PSD, rank conditions, EDF bounds, deviance non-negative, no NaN in a
converged model. If the diff can violate one, that is your top finding.

### Tests, statistically
- Do the R-parity tests actually pin the statistics, or only shapes and
  finiteness? A parity test that does not compare numbers to mgcv is not a
  parity test.
- Are R-comparison tolerances at STRICT or MODERATE (`CLAUDE.md` requires it)?
  Flag any loosening — but note that a genuinely-needed loose tolerance must be
  justified by a *measured* gap, and a tolerance far looser than the true
  agreement hides regressions.
- Are golden values sourced from R/mgcv or an independent derivation, rather
  than regenerated from this implementation's own output?
- Is the model in the test statistically capable of catching the bug — enough
  data, a non-degenerate design, a family that exercises the changed path?
- **The REML criterion is flat near its optimum.** λ differing from R by ~1e-3
  is fine and is not a finding; validate deviance, coefficients, fitted values,
  and EDF instead. Do not report λ precision as a defect.

## Carrying findings forward

Add this pass's findings to the engineering pass's list, then go to SKILL.md §6
to verify the combined set before reporting. For each finding record:
`file:line`, the incorrect mathematics stated precisely (what it computes vs
what it should compute), the concrete input that exposes it, and the R function
or design section it contradicts.

Mark anything you have not confirmed against the file *and* the R source — a
suspicion you did not check is not a finding, and §6 is where it either gets
checked or gets dropped. Where this pass contradicts the engineering pass on
the same lines, flag the pair so §6 resolves it against the docs and R rather
than reporting both. An empty pass is a valid pass.
