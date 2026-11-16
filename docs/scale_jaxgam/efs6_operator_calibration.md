# Operator EFS score calibration v2 (beyond 512 levels)

Operator EFS accepts or rejects a trial only when both scores carry an
admitted absolute error bound (`_run_operator_efs`,
`compare_operator_scores`); otherwise the comparison is indeterminate and the
fit stops without claiming convergence. The structured operator paths compute
exact Fisher traces and exact structured log determinants, so the bound covers
floating arithmetic in the source reductions and the effect of the inner
coefficient fit stopping at a small nonzero gradient. Calibration v1 admitted
257-512 levels, 2-3 rows per level and `batch_rows=64`. This note records how
v2 was measured, what changed, what it admits and what it does not.

## Method error versus implementation error

* **Implementation error** is operator output minus the exact value of the
  same quantity for the same model, smoothing parameter and scale. The score
  bound must cover it, and this calibration measures it.
* **Method error** is EFS against Newton REML. EFS stops on its own score
  window, so its optimum differs from Newton's (for example score 870.04
  against 869.97 at 300 levels). That is not an operator parity failure and is
  not measured here.
* **Dense EFS is not the ground truth for implementation error.** Its score
  uses the shared jittered `_criterion_log_determinants` (see "Dense EFS log
  determinant" below). Operator states are compared with an independent
  50-digit decimal evaluation instead. Dense EFS remains the complete-fit
  comparator for iterations, decisions and final fields.

The decimal reference (`tests/operator_efs_calibration.py`) reads only the
generated rows and the identity-penalty scale. It evaluates deviance, the
saturated likelihood, the penalty, `log|H|` (diagonal, or diagonal plus the
modest Schur complement), the Fisher trace, the penalty quadratic and EDF, and
finds the exact penalized optimum by decimal Newton iteration. It is checked
against dense float64 algebra at STRICT in
`tests/test_operator_efs_calibration.py`. Each operator state is split into

* arithmetic error: operator score minus the decimal score at the operator's
  own coefficients;
* coefficient error: decimal score at the operator coefficients minus the
  decimal score at the exact optimum;
* total error: operator score minus the decimal score at the exact optimum,
  which the admitted bound must cover.

## Envelope (v2)

`M` sums the magnitudes of the penalized deviance (over phi for Gaussian), the
saturated likelihood, the penalty log determinant and the score; `g` is the
final infinity-norm gradient of the penalized deviance; `floor = lambda s` for
the identity penalty; `sigma` is the arrowhead's empirically buffered spectral
estimate; `r` is the largest public row l1 norm.

| Term | indexed Poisson | indexed Gaussian | arrowhead Poisson |
|---|---|---|---|
| arithmetic | `32 eps (n+p) M` | `32 eps (n+p) M` | `32 eps (n+p+k^2) M` |
| coefficient | `p (g + g^2) / floor` | `p g^2 / (floor phi)` | `(EDF r sqrt(p) g + p g^2) / sigma` |
| determinant | decimal log rounding | decimal log rounding | none |

Three terms changed from v1:

* **Arithmetic factor 512 -> 32.** The row deviance and saturated terms, the
  local penalty quadratic and the indexed diagonal sums each have one sign, so
  batched recursive summation has a first-order rounding bound of about
  `(n + p) eps M`. The largest measured arithmetic error over every
  calibration state was `6.7e-4 (n + p) eps M`, so 32 keeps a factor of about
  16 over the first-order estimate and about 5e4 over the data. With 512, the
  bound (about 5.7e-5 per state at 4096 levels and 8192 rows) was larger than
  genuine late contraction differences (4.0e-5 and 4.4e-5), and the
  4096-level arrowhead and Gaussian fits stopped as indeterminate although
  their measured errors were near 1e-12.
* **Arrowhead coefficient term.** v1 had only `p g^2 / sigma`. For a
  canonical family `d log|H| / d beta_j = sum_i a_i x_ij`, where the weighted
  leverages `a_i` lie in `[0, 1]` and sum to the EDF, so the first-order
  determinant change from a coefficient residual is at most
  `EDF r sqrt(p) g / (4 sigma)`. The measured coefficient-induced score error
  exceeded the v1 term by up to 3.6e9 times; v1 only held because its
  arithmetic allowance was large. The score-row reduction now reports `r`.
* **Gaussian coefficient term.** For Gaussian identity `log|H|` does not
  depend on beta and `H >= floor I` everywhere, so a coefficient residual
  changes the score by at most `p g^2 / (8 floor phi)`; v1's extra first-order
  part `p g / (floor phi)` was not needed. It was removed after the held-out
  complete fits showed it could dominate the comparison radius (1.4e-5 at
  3000 levels). Two facts support the change despite that timing: it has a
  theoretical basis (the beta-independence of `log|H|` above), and it was
  re-validated on a fresh Gaussian held-out set run afterwards (52 states,
  maximum error/bound 1.5e-6). Against every recorded Gaussian state the
  measured coefficient-induced error is at most 0.086 of the new term
  (theory: 1/8).

The indexed Poisson coefficient term is unchanged: each
`d log H_gg / d beta_g` is at most one, so the first-order change is a quarter
of `p g / floor`.

The method labels are `indexed_poisson_empirical_v2`,
`indexed_gaussian_empirical_v2` and `arrowhead_poisson_empirical_v2`. Public
operator EFS results report the label as `score_bound_method` beside
`score_error_bound`; a fit whose accepted state has no admitted bound reports
`unavailable` and never claims a validated bound.

## Measurements

Cases are generated at runtime by
`scripts/benchmark_operator_efs_calibration.py`
(`make benchmark-operator-efs-calibration`), which writes JSON lines outside
the repository. Calibration cases set the domain. Held-out cases (different
seeds, level counts, rows per level and batch sizes) were run after the domain
was fixed and were not used to change it. Matched states use smoothing
parameters 0.1, 1, 10, 100 and the controller's shifted start; the stress
cases use larger group effects and smoothing parameters from 1e-3 to 1e5.
Unless stated, operator controls are `tolerance=1e-11, cg_rtol=1e-13,
cg_atol=1e-15`; `1e-7` rows use the public defaults. Ratios use the v2
envelope; "admitted" applies the v2 domain and conditioning gates.

Matched states, calibration set (13 seeds; operator controls tight unless
the row says `1e-7`):

| Regime | Operator tolerance | Levels | Rows per level | Batch rows | States (admitted) | Max total error | Max error / bound | Max arithmetic error / term | Max coefficient error / term |
|---|---|---|---|---|---|---|---|---|---|
| indexed Poisson | 1e-11 | 384-4096 | 1-16 | 16-65536 | 435 (425) | 1.1e-07 | 6.7e-03 | 1.3e-05 | 1.7e-02 |
| indexed Poisson | 1e-7 | 1024-4096 | 2-3 | 64 | 20 (20) | 8.4e-06 | 1.5e-03 | 4.9e-06 | 1.5e-03 |
| indexed Gaussian | 1e-11 | 384-4096 | 1-16 | 16-65536 | 417 (412) | 2.9e-10 | 1.4e-05 | 1.4e-05 | 8.6e-02 |
| indexed Gaussian | 1e-7 | 1024-4096 | 2-3 | 64 | 20 (20) | 4.4e-11 | 7.3e-06 | 7.3e-06 | 2.3e-02 |
| arrowhead Poisson, `y~x+s(g)` | 1e-11 | 384-4096 | 1-16 | 16-65536 | 508 (473) | 2.1e-09 | 1.4e-04 | 2.1e-05 | 2.2e-04 |
| arrowhead Poisson, `y~x+s(g)` | 1e-7 | 1024-4096 | 2-3 | 64 | 20 (17) | 2.6e-05 | 4.1e-05 | 4.1e-06 | 5.0e-05 |
| arrowhead Poisson, `y~s(g)` | 1e-11 | 768-4096 | 2-3 | 64 | 30 (26) | 2.5e-10 | 5.6e-05 | 6.2e-06 | 1.2e-04 |

The ratio does not grow with size or rows per level. Its largest values per
level count, over all three regimes, are 1.4e-4 to 6.7e-3 at 384-1024
levels and 5.8e-5 to 5.9e-3 at 2048-4096 levels; per rows-per-level bucket
(1, 1.5-3, 4-8, 16) they are at most 2.7e-3, 6.7e-3, 3.8e-5 and 2.0e-5. The
largest indexed Poisson ratios come from the stress cases (group effects
five times larger, smoothing parameters down to 1e-3), where the coefficient
term dominates; measured coefficient-induced error stays below 1/58 of its
term. Batch sizes from 16 to 65536 rows give the same maxima. Non-admitted
states are those with fewer than 64 batch rows, those beyond the unchanged
conditioning gates (small smoothing parameters with many rows), or
public-default fits whose final gradient exceeded the `g <= 1e-5` gate; their
formula ratios are included in the "max error / bound" column.

Against the decimal values at the exact optimum (so including the coefficient
residual), the worst relative differences at tight controls are 1.8e-11 for
the Fisher trace, 1.5e-10 for the penalty quadratic, 1.1e-11 for deviance and
5e-12 for EDF; at the public-default tolerance they reach 2.6e-8, 1.7e-7,
1.5e-8 and 8.3e-9. Those residuals belong to the inner fit tolerance, not to
the score envelope; at the operator's own coefficients the boundary tests
check the same statistics at STRICT.

Complete fits (operator EFS against dense EFS from the same start; dense runs
up to 2048 levels):

| Regime | Levels | Rows per level | Batch rows | Role | Operator stop (outer) | Dense outer | Max relative difference vs dense: score, sp, deviance, coefficients, EDF |
|---|---|---|---|---|---|---|---|
| indexed Poisson | 768-1536 (4 fits) | 2-3 | 64 | calibration | score_window (5-9) | same | <= 2e-13, 9e-12, 2e-12, 2e-11, 7e-12 |
| indexed Poisson | 2048 | 1, 4, 8, 16 | 4096 | calibration | score_window (13, 8, 4, 6) | same | <= 3e-13, 7e-12, 3e-12, 5e-12, 5e-12 |
| indexed Poisson | 4096 | 2 | 64 | calibration | score_window (12) | - | - |
| indexed Poisson, unbalanced | 3072 | 3 | 65536 | calibration | indeterminate (6) | - | - |
| indexed Gaussian | 768-1536 (4 fits) | 2-3 | 64 | calibration | deviance_change (6-8) | same | <= 3e-13, 1e-13, 2e-12, 3e-12, 7e-14 |
| indexed Gaussian | 2048 | 4, 8, 16 | 4096 | calibration | deviance_change (5, 4, 3) | same | <= 3e-12, 7e-14, 7e-13, 1e-12, 2e-14 |
| indexed Gaussian | 2048 | 1 | 4096 | calibration | indeterminate (2) | 2 | (stopped before dense's final step) |
| indexed Gaussian | 4096 / 3072 unbalanced | 2 / 3 | 64 / 65536 | calibration | deviance_change (8, 6) | - | - |
| arrowhead Poisson | 768-1536 (4 fits) | 2-3 | 64 | calibration | score_window (7-19) | same | <= 2e-13, 6e-11, 2e-12, 6e-12, 4e-11 |
| arrowhead Poisson | 2048 | 1, 4, 8, 16 | 4096 | calibration | score_window (12, 11, 8, 7) | same | <= 3e-13, 3e-11, 3e-12, 9e-12, 3e-11 |
| arrowhead Poisson, unbalanced | 3072 | 3 | 65536 | calibration | score_window (20) | - | - |
| arrowhead Poisson | 4096 | 2 | 64 | calibration | indeterminate (10) | - | - |
| indexed Poisson | 1300 | 6 | 3000 | held-out | score_window (4) | same | <= 5e-14, 4e-12, 4e-13, 2e-12, 2e-12 |
| indexed Poisson, unbalanced | 3000 | 2.5 | 20000 | held-out | indeterminate (6) | - | - |
| indexed Gaussian | 800, 1700 | 10, 4 | 512, 2048 | held-out | deviance_change (4, 5) | same | <= 4e-13, 7e-14, 7e-13, 1e-12, 2e-14 |
| indexed Gaussian | 4096 / 3000 unbalanced | 3 / 2.5 | 8192 / 20000 | held-out | deviance_change (6, 7) | - | - |
| indexed Gaussian | 1300 / 3500 unbalanced | 6 / 2 | 3000 / 65536 | held-out | indeterminate (5, 8) | 5 / - | (1300: stopped before dense's final step) |
| arrowhead Poisson | 1300 | 6 | 3000 | held-out | score_window (9) | same | <= 6e-14, 1e-11, 7e-13, 3e-12, 8e-12 |
| arrowhead Poisson, unbalanced / `y~s(g)` | 3000 / 2500 | 2.5 / 3 | 20000 / 4096 | held-out | score_window (14, 11) | - | - |

The 768-1536 level rows ran with the v1 arithmetic factor (512) and made no
undecided comparison, so the smaller v2 bounds give the same decisions; all
other rows ran with the final v2 envelope and controller. Every converged fit
that has a dense comparator took the same number of outer iterations, and
every field agrees within STRICT. Across all complete-fit audits, which
include every evaluated trial, no state's total error exceeded its bound. The
largest error/bound ratio was 1e-4 (indexed Poisson, 1536 levels).

Six of the 41 measured complete fits stopped as `indeterminate_score`: 5 of
the 24 fits at 2048 levels or more, and one held-out Gaussian fit at 1300
levels (3 of 18 calibration fits at 2048-4096 levels, 3 of 11 held-out
fits). In each, the undecided comparison was
a genuine score difference smaller than the sum of two bounds; the decimal
reference confirms both scores to about 1e-12:

* arrowhead, 4096 levels: a contraction that stayed at multiplier 2
  (difference +7.07e-6, radius 7.47e-6);
* indexed Poisson, unbalanced, 3000-3072 levels: main comparisons at
  multiplier 1 while EFS creeps towards its fixed point (+3.6e-6 against
  7.9e-6; +2.5e-6 against 1.6e-5);
* Gaussian, 1300 and 3500 levels: final improvements of 6.8e-7 and 2.6e-6
  against radii of 2.6e-6 and 4.9e-6, where dense accepts the step and then
  stops on deviance change;
* Gaussian, one row per level: the EFS ratio is one (random-effect and
  residual variances are confounded), so each refit repeats rho up to one or
  two ulps and the scores differ by about 1e-13.

These are resolution limits of a per-state absolute bound, not bound
failures: near convergence the score changes quadratically in the step while
dense's deviance-change stop needs the step itself to be small.
`_run_operator_efs` resolves undecided comparisons by three explicit rules,
and counts every one in `indeterminate_comparisons` (public results:
`operator_diagnostics.efs_indeterminate_comparisons`):

* an undecided comparison never extends the step. efsudr tries the doubled
  step only near the optimum and keeps the plain step unless the extension
  is strictly better; declining is that other branch. The two candidates
  then differ by less than their certified bounds, and the score-window and
  step-size convergence test is unaffected, so the fit continues and may
  still converge;
* after the final contraction to multiplier 1 the step is accepted, because
  efsudr accepts it whatever the score;
* any other undecided comparison (the main trial, or a contraction still
  above multiplier 1) stops the fit as `indeterminate_score` with
  `converged=False`.

Held-out matched states (new seeds; 800-4096 levels; 1-16 rows per level;
batch rows 64, 512, 777, 2048-65536; balanced and unbalanced; plus
`y~s(g)`), evaluated after the domain was fixed:

| Regime | States (admitted) | Max total error | Max error / bound | Max arithmetic error / term | Max coefficient error / term |
|---|---|---|---|---|---|
| indexed Poisson | 56 (54) | 9.7e-09 | 2.4e-03 | 1.8e-05 | 1.8e-02 |
| indexed Gaussian | 129 (129) | 2.3e-10 | 6.6e-06 | 6.6e-06 | 2.1e-02 |
| arrowhead Poisson, `y~x+s(g)` | 77 (69) | 1.7e-08 | 1.2e-05 | 1.6e-05 | 6.0e-05 |
| arrowhead Poisson, `y~s(g)` | 19 (19) | 5.1e-10 | 2.8e-06 | 2.0e-06 | 3.8e-06 |

All refused held-out states fail an unchanged conditioning gate (arrowhead
spectral condition above 1e5, indexed diagonal condition ratio above 1e4) at
smoothing parameters 0.1 and 1 with many rows per level. The Gaussian
held-out set includes a fresh group run after the Gaussian coefficient-term
change (52 states, max error/bound 1.5e-6).

## Admitted regime (v2)

`CALIBRATED_OPERATOR_EFS_DOMAIN` (shared by the three regimes and the public
pre-check in `GAM._fit_operator_efs`):

| Limit | v1 | v2 |
|---|---|---|
| indexed levels | 257-512 | 257-4096 |
| rows per level | 2-3 | 1-16 |
| `batch_rows` | 64 | 64-65536 |
| modest coefficients (arrowhead) | 1-4 | 1-4 (only 1 and 2 measured) |

Everything else is unchanged and still required: canonical Poisson (indexed
or arrowhead) or Gaussian identity (indexed); one full-rank indexed random
effect with one scaled identity penalty; unit weights; zero offset; Poisson
responses at most 32, Gaussian responses at most 10 in magnitude and incoming
phi in `[1e-8, 1e4]`; converged inner fits with `tolerance <= 1e-7`,
`cg_rtol <= 1e-10`, `cg_atol <= 1e-12`; final gradient at most 1e-5;
coefficients at most 20 in magnitude; diagonal condition ratio at most 1e4
and minimum diagonal at least 1e-6 (indexed); spectral lower estimate at least
1e-3 and spectral condition estimate at most 1e5 (arrowhead). Outside these
limits the formula is still computed for diagnostics, but the state's bound
is `None` with method `unavailable`, so its comparisons are indeterminate and
the fit cannot claim convergence through a score decision. The public route
rejects model shapes outside the domain before any EFS replay. Only
canonical Poisson is public; Gaussian operator EFS remains internal.

The minimum of 64 batch rows is kept although 16-row batches measured the
same: smaller batches have no use and the released rejection gate for 32 rows
stays as it was.

## Dense EFS log determinant

Dense EFS scores call the shared `_criterion_log_determinants`, which scales
`H` to unit diagonal and factors `H_scaled + 1e-12 I`. The resulting score
shift is `0.5e-12 tr(H_scaled^-1)`: exactly `p * 5e-13` for the diagonal
systems and 1.0-1.5 times that for the arrowhead coupling at 384-1024 levels
(measured over 380 dense matched states). It accounts for almost all of dense
EFS's error against the decimal reference: the remaining dense error is at
most 3.4e-10. Operator EFS does not use this path; its structured log
determinants are exact.

Effect on operator-versus-dense comparisons: at matched states the operator
minus dense score is at most 9.9e-9 (arrowhead, 1024 levels, including dense
coefficient differences up to 2.8e-9), and on complete fits the final dense
score is off by `p * 5e-13` (1.0e-9 at 2048 levels). Relative to STRICT
(`rtol 1e-10` on scores above 1e3) this is at most a few percent of the
allowance, and because the shift varies slowly with rho it never changed an
accept/reject decision: every dense-compared complete fit had identical
iterations. The shift grows linearly with p, so it will remain small relative
to STRICT for scores that grow with n.

Recommendation: leave the shared dense/Newton code unchanged; it is not an
operator EFS error. Calibrate operator EFS against the decimal reference, as
done here. If an unjittered dense EFS score is wanted, dense EFS can pass its
own exact Fisher factor's log determinant through the existing
`log_det_hessian` argument of `reml_criterion_from_penalized_deviance` when the
family's observed information equals its Fisher information. That change is
EFS-owned, would need dense EFS pinned-R revalidation, and is low priority. A
planned dense source-parity series is expected to remove the default jitter
later; the operator calibration does not depend on it, because it uses the
decimal reference rather than dense EFS scores as ground truth.

## Limitations

* The envelope is empirical where noted: the arithmetic factor is supported
  by a first-order summation argument and by measurement, and the arrowhead
  spectral estimate is not a certified eigenvalue bound. The bounds are not
  probability statements.
* Measured on runtime-generated sinusoidal group effects (balanced and
  Zipf-unbalanced), one host (macOS arm64) and the stated controls. The
  conditioning gates are enforced at run time, but real data patterns were
  not part of the calibration. Pinned-Linux and x86_64 runs are pending.
* More than 4096 levels or 16 rows per level were not measured and are
  rejected. Arrowhead models with 3 or 4 modest coefficients remain admitted
  as in v1 but were not measured.
* Complete fits can still stop truthfully as `indeterminate_score` inside
  the domain: 6 of the 41 measured complete fits, 5 of the 24 at 2048 levels
  or more. Near convergence the per-state absolute bounds stay roughly
  constant while genuine late score differences shrink quadratically in the
  step, so a comparison can become undecidable before efsudr's
  deviance-change stop fires. These fits report `converged=False` and keep
  the last accepted state. A tighter decision needs a bound on score
  *differences* between nearby states, whose rounding errors are strongly
  correlated, rather than the sum of two absolute bounds. That is recorded as
  a deferred follow-up.
* Public-default operator controls (`tolerance=1e-7`) give larger final
  gradients; some states then fail the unchanged `g <= 1e-5` gate (3 of 20
  arrowhead states measured), which ends the fit as indeterminate.
* `_run_operator_efs` remains a third EFS controller copy; its decision
  rules follow pinned efsudr but its floating-point order differs from
  `fitting/efs.py::efs_raw_update` (unify-fit item).
