# Operator fitting

The operator route fits models whose random effects have very many levels
without ever forming the coefficient matrix. Select it with
`FitControl(execution="operator", linear_solver="pcg")`.

Each coefficient system is solved by preconditioned conjugate gradients
(PCG), an iterative solver that needs only products of the system with a
vector. Each product is computed by scanning the row source, so neither the
model matrix nor any matrix of size coefficients by coefficients is held in
memory. A random effect `s(g, bs="re")` is stored as one level index per row.

**Use it when**

- a random effect has so many levels that the default fit would need a
  coefficient matrix too large for memory;
- you know the smoothing parameters (`sp=`), or the model is a Poisson random
  intercept inside the shape where Fellner-Schall is admitted (see below).

**Do not use it when**

- you need the smoothing parameters estimated for any other model: use the
  [discretized route](discretized.md) or the default fit;
- you need `predict(se_fit=True)`, `summary()` or EDF per term. There is an
  approximate variance (see below), but no exact standard errors.

The setup block makes a random-intercept model with 1,000 levels.

```python
import numpy as np
import pandas as pd

from jaxgam import GAM, FitControl, OperatorControl
from jaxgam.data import DataFrameRowSource
from jaxgam.families import NegativeBinomial

rng = np.random.default_rng(4)
levels = 1000
n = 3000
g = rng.integers(0, levels, n)
x = rng.uniform(0, 1, n)
effect = rng.normal(scale=0.5, size=levels)
eta = 0.2 + 0.5 * x + effect[g]
data = pd.DataFrame(
    {
        "x": x,
        "g": pd.Categorical(g, categories=range(levels)),
        "y": eta + rng.normal(scale=0.3, size=n),
        "count": rng.poisson(np.exp(eta)),
    }
)
source = DataFrameRowSource(data, response="y")
counts = DataFrameRowSource(data, response="count")
newdata = pd.DataFrame({"x": [0.2, 0.4], "g": pd.Categorical([0, 5000])})
```

See [Row sources](row_sources.md) for the built-in sources, prior weights and
offsets.

## A minimal fit with fixed smoothing parameters

```python
control = FitControl(execution="operator", linear_solver="pcg")
fit = GAM("y ~ x + s(g, bs='re')", sp=[1.0], control=control).fit(
    source, result="prediction"
)
print(fit.converged, fit.convergence_info)
# True Operator observed Newton fit converged.
print(len(fit.coefficients), fit.source_scans)
# 1002 21
print(fit.predict(newdata))
# [0.12566849 0.4141132 ]
```

`sp` must be given in formula order, one value per penalty. A model with no
penalized term still needs `sp=[]`. Level 5000 was not in training: an unseen
level contributes zero, so its prediction is the fixed part only.

The levels of a random effect are a categorical column's declared categories
(unused ones included), otherwise the sorted observed values.

## What you get back

A `GAMPredictionResult` (see the [API reference](../api.md#gampredictionresult))
with `predict`, `predict_iter` and `predict_matrix`. `pred_type` is
`"response"` or `"link"`. With fixed smoothing parameters no REML score,
trace or covariance is computed, so several attributes are `None`:

```python
print(fit.score, fit.edf_total, fit.scale, fit.known_workspace_bytes)
# None None None None
try:
    fit.predict(newdata, se_fit=True)
except RuntimeError as error:
    print(error)
# Prediction standard errors are not available on the operator route. Use
# bulk_link_variance() on the fit result for approximate link-scale variances.
```

`scale` is `None` for families with an unknown scale and `1.0` for Poisson,
Binomial and the negative binomial. The advice in the `se_fit` message does
not apply here: this route refuses any `uncertainty` other than `"none"`.
`materialize_covariance` raises `RuntimeError: No covariance provider was
retained for this result.` For approximate variances, see
[Approximate variances](#approximate-variances) below.

| Attribute | Meaning on this route |
|---|---|
| `converged`, `convergence_info`, `n_iter` | Newton iterations for fixed `sp`; outer iterations for Fellner-Schall. |
| `execution_route`, `lambda_strategy` | `"operator"`; `"fixed"` or `"efs_reml"`. |
| `smoothing_params`, `deviance`, `theta` | The fitted values. |
| `score`, `edf_total`, `score_error_bound`, `score_bound_method` | Fellner-Schall only. |
| `source_scans`, `batches_scanned` | Scans made by the fit, not by the setup pass. |
| `known_workspace_bytes` | `None` for fixed `sp`; the planned workspace for Fellner-Schall. |
| `operator_diagnostics` | Solver and scan detail (below). |

### Telling whether the fit converged

The fit is a Newton iteration on the penalized deviance, using the observed
curvature of the family. It stops when the relative change of the objective
is below `OperatorControl.tolerance` and the gradient is small; a Gaussian
identity model stops after its first solve. `convergence_info` is one of:

- `"Operator observed Newton fit converged."`;
- `"Maximum operator iterations reached."` (raise `max_iterations`);
- `"Operator PCG failed with status N."` (raise `cg_max_iterations`, or try
  `use_coarse=True`).

The last two return `converged=False` without raising. Some failures raise
instead: when the system cannot be shown to be positive definite, a
`LinAlgError` such as `Actual weighted system has no sufficient Schur SPD
certificate.`, and after 100 step halvings `RuntimeError: Operator step
halving could not recover a valid step.`

`fit.operator_diagnostics` holds the detail: `inner_status` (status of the
last PCG solve, 1 means converged), `final_gradient_norm`,
`krylov_iterations`, `completed_solves`, `peak_bound_bytes`, and the scans
used for each purpose (`rhs_scans`, `objective_scans`, `certificate_scans`,
`preparation_scans`):

```python
diagnostics = fit.operator_diagnostics
print(diagnostics.inner_status, diagnostics.completed_solves, diagnostics.krylov_iterations)
# 1 1 6
```

Every byte figure on this route is an accounting bound for arrays jaxgam
knows it will hold; it excludes the source's own storage and allocator or
compiler memory.

## Models and families

**Random effects.** Any number of `s(..., bs="re")` terms. Factors in the term
form the levels (two factors give one level per combination), and a numeric
variable in the term gives a random slope, for example `s(g, x, bs="re")`.
Their width is not limited by `max_modest_coefficients`.

**The small block.** Everything else (the intercept, numeric parametric terms
and `s()`, `te()` or `ti()` smooths with `cr`, `cs`, `cc`, `tp`, `ts` or `gp`
bases, with or without a factor or numeric `by`) is held as an ordinary small
matrix. Its raw width is capped by `OperatorControl.max_modest_coefficients`
(64):

```python
try:
    GAM("y ~ s(x, bs='cr', k=70) + s(g, bs='re')", sp=[1.0, 1.0],
        control=control).fit(source, result="prediction")
except MemoryError as error:
    print(error)
# Modest operator width 71 exceeds hard cap 64.
```

**Families.** There is no fixed list: any family that provides a row-wise
deviance with its derivatives and observed information is accepted. The tests compare with mgcv for
Gaussian (identity, log, inverse), Poisson (log, identity, sqrt), Binomial
(logit, probit, cloglog, log), Gamma (inverse, log, identity) and the
negative binomial with fixed theta (log, identity, sqrt). Theta must be
fixed:

```python
try:
    GAM("count ~ x + s(g, bs='re')", family="nb", sp=[1.0], control=control).fit(
        counts, result="prediction")
except NotImplementedError as error:
    print(error)
# Operator fitting currently requires fixed theta.

fit_nb = GAM("count ~ x + s(g, bs='re')", family=NegativeBinomial(3.0, fixed=True),
             sp=[1.0], control=control).fit(counts, result="prediction")
print(fit_nb.converged)
# True
```

A thin plate, Gaussian process or tensor smooth in the small block needs
workspace for every row of a batch. The fit plans for `batch_rows` rows, or
for the number of rows in the source when that is smaller, so a source of a
few hundred rows fits at the defaults. On the 3,000 rows above the default
budget is not enough for `y ~ s(x, k=8) + s(g, bs='re')`:

```text
MemoryError: Frozen operator requires 487335048 bytes; budget is 48614104. Lower batch_rows or raise the memory budget.
```

The budget in the message is what is left of `memory_budget_bytes` after the
fit's other arrays. Lower `batch_rows` (or raise `memory_budget_bytes`):

```python
smaller_batches = FitControl(execution="operator", linear_solver="pcg",
                             batch_rows=1024)
fit_tp = GAM("y ~ s(x, k=8) + s(g, bs='re')", sp=[1.0, 1.0],
             control=smaller_batches).fit(source, result="prediction")
print(fit_tp.converged)
# True
```

## Options that matter

| Option | Default | What it does |
|---|---|---|
| `FitControl.linear_solver` | `"cholesky"` | Must be `"pcg"` on this route. |
| `FitControl.batch_rows` | `65_536` | Rows per source scan. The fit uses the source's row count when that is smaller. |
| `FitControl.memory_budget_bytes` | 512 MiB | Bound on the arrays the setup pass and the fit know they will hold. |
| `OperatorControl.max_modest_coefficients` | `64` | Cap on the width of the small block. |
| `OperatorControl.max_iterations` | `200` | Newton iterations. |
| `OperatorControl.tolerance` | `1e-7` | Newton stopping tolerance. |
| `OperatorControl.cg_max_iterations` | `1000` | PCG iterations per solve. |
| `OperatorControl.cg_rtol`, `cg_atol` | `1e-10`, `1e-12` | PCG stops when the residual is below `max(cg_atol, cg_rtol * norm of the right-hand side)`. |
| `OperatorControl.use_coarse` | `False` | Adds a coarse correction to the preconditioner. |

Pass them as `FitControl(operator=OperatorControl(...))`. Field-level detail
is in the [API reference](../api.md#fit-control).

## Fellner-Schall on this route

Without `sp`, `optimizer="efs"` selects the smoothing parameter by the
extended Fellner-Schall update (mgcv's `efsudr`), but only inside the model
shape over which its score error bound was measured:

- Poisson with the log link;
- exactly one random effect, made of factors only (no random slopes), with
  257 to 4,096 levels and between 1 and 16 rows per level on average;
- at most 4 other coefficients, all unpenalized (the intercept and numeric
  parametric terms);
- unit prior weights and no offset;
- `batch_rows` from 64 to 65,536.

```python
def efs_frame(levels, rows_per_level, seed):
    rows = levels * rows_per_level
    gen = np.random.default_rng(seed)
    group = np.arange(rows) % levels
    xs = np.linspace(-1, 1, rows)
    ys = gen.poisson(np.exp(0.2 + 0.5 * xs + 0.3 * np.sin(group / 31.0)))
    return pd.DataFrame({"y": ys, "x": xs, "g": pd.Categorical(group)})

efs_data = efs_frame(300, 2, 32031)
efs_source = DataFrameRowSource(efs_data, response="y")
fit_efs = GAM("y ~ x + s(g, bs='re')", family="poisson", optimizer="efs",
              control=control).fit(efs_source, result="prediction")
print(fit_efs.converged, fit_efs.convergence_info, fit_efs.n_iter)
# True score_window 4
print(fit_efs.smoothing_params, fit_efs.score_bound_method)
# [27.99472604] arrowhead_poisson_empirical_v2
print(fit_efs.edf_total, fit_efs.score_error_bound)
# 27.524338405228207 8.67825993407174e-08
```

Outside that shape the fit raises before it starts:

| Not admitted | Error |
|---|---|
| Another family or link | `NotImplementedError: Operator EFS currently supports calibrated Poisson/log indexed fits.` |
| `batch_rows` outside 64 to 65,536 | `NotImplementedError: Calibrated operator EFS requires batch_rows 64-65536.` |
| Levels, rows per level or other coefficients outside the shape | `NotImplementedError: Calibrated operator EFS requires 257-4096 indexed levels, 1-16 rows per level and batch_rows 64-65536, and at most 4 modest coefficients.` |
| A penalized smooth, a random slope, weights or an offset | `NotImplementedError` beginning `Operator EFS source start needs ...` |

The Fellner-Schall controls (`outer_limit`, `score_tolerance`,
`pirls_tolerance`, `log_lambda_max`, `history_limit`) come from
`FitControl(efs=EFSControl(...))`; the inner fits use `OperatorControl`.
On these results `operator_diagnostics` describes only the final accepted
fit, while `source_scans` counts the whole search.

### What an "indeterminate" stop means

The score on this route is computed with a known error bound. Fellner-Schall
compares each trial's score with the accepted one, and the comparison counts
only when the difference is larger than the two bounds added together. If it
is not, or if a bound could not be given, the comparison is indeterminate.
The fit then stops, keeps the last accepted smoothing parameter, and returns
`converged=False` with `convergence_info="indeterminate_score"`. No warning
is issued. `operator_diagnostics.efs_indeterminate_comparisons` counts these
comparisons.

A bound needs, among other conditions, all responses at most 32, a converged
PCG solve, and `OperatorControl` no looser than its defaults
(`tolerance <= 1e-7`, `cg_rtol <= 1e-10`, `cg_atol <= 1e-12`). Loosening any
of them removes the bound, and the fit stops after one iteration:

```python
loose = FitControl(execution="operator", linear_solver="pcg",
                   operator=OperatorControl(tolerance=1e-6))
fit_loose = GAM("y ~ x + s(g, bs='re')", family="poisson", optimizer="efs",
                control=loose).fit(efs_source, result="prediction")
print(fit_loose.converged, fit_loose.convergence_info, fit_loose.n_iter)
# False indeterminate_score 1
print(fit_loose.score_bound_method)
# unavailable
```

An indeterminate stop can also happen at the default controls, because the
fitted coefficients are then less exact than the score comparison needs.
Tighter controls make the comparisons decidable more often; the test suite
fits these models with
`OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15)`.

The other stops are `"score_window"` and `"deviance_change"` (converged), and
`"iteration_limit"`, `"inner_failure"` and `"invalid_update"` (not
converged).

## Approximate variances

`fit.bulk_link_variance` estimates the variance of the linear predictor (the
link scale) for any number of new rows, from random probes of the penalized
Fisher information. It needs a converged operator fit and the original
training source, which it scans again. `calibration_data` (at most 256 rows)
is checked against exact solves; the call raises `RuntimeError` when the
estimate disagrees with them by more than `max_calibration_standard_errors`
(default 5).

```python
variance, calibration = fit_nb.bulk_link_variance(
    counts, newdata, probes=16, seed=0, memory_budget_bytes=256 * 1024 * 1024
)
print(calibration.exact, calibration.estimated)
# [0.3246366  0.00154859] [0.11842111 0.00343223]
for batch in variance.estimate_iter(DataFrameRowSource(newdata)):
    print(batch.positions, batch.link_se, batch.monte_carlo_se)
# [0 1] [0.34412368 0.05858525] [0.02807059 0.00109346]
```

With 16 probes the estimate is rough (compare `exact` and `estimated`); the
second row, at an unseen level, has only the fixed part's variance.

`estimate_iter` takes any row source with the model's covariates and yields
batches with `positions`, `variance`, `monte_carlo_se` (the Monte Carlo error
of the estimate) and `link_se` (the square root of `variance`). More probes
give a smaller Monte Carlo error and more training scans; each probe is one
PCG solve. For a family with an unknown scale (Gaussian, Gamma), pass
`scale=` yourself: `ValueError: Unknown-scale family requires an explicit
Fisher scale.` `variance.diagnostics` reports the solves, scans and
`maximum_relative_residual`. The calibration checks only the selected rows;
it does not bound the other rows or the bias of the iterative solves.

## Limits

| Not accepted | Error |
|---|---|
| `linear_solver` other than `"pcg"` (the default included) | `ValueError: execution='operator' requires linear_solver='pcg'.` |
| `result="full"` or `"inference"` | `NotImplementedError: Operator full/inference results need planned score/EDF/variance providers; this route currently requires result='prediction'.` |
| `uncertainty="fisher"` or `"covariance"` | `NotImplementedError: Operator uncertainty needs a planned selected-variance provider.` |
| No `sp` (without `optimizer="efs"`) | `NotImplementedError: Operator fitting requires explicit fixed sp.` |
| Estimated theta | `NotImplementedError: Operator fitting currently requires fixed theta.` |
| `device="gpu"` | `NotImplementedError: Frozen-system operator PCG currently uses CPU.` |
| A DataFrame instead of a row source | `TypeError: Operator fitting requires a replayable RowSource.` |
| `fit(weights=...)` or `fit(offset=...)` | `ValueError: Operator weights/offset must be owned by the RowSource.` |
| `by=` on a random effect, `te()` with `re`, other bases | `NotImplementedError: Operator preparation supports numeric terms, cr/cs/cc, tp/ts/gp and re.` |
| Extra options (such as `xt`) on `cr`, `cs`, `cc` or `re` smooths | `NotImplementedError: Operator custom smooth/precision/id options are unsupported.` |
| A factor parametric term | `NotImplementedError: Prepared setup currently supports numeric parametric terms only.` |
| Wrong number of smoothing parameters | `ValueError: sp must have N elements for this model.` |

`gaussian_compression=True` is ignored on this route. Models without an
intercept are fitted, but some layouts are refused with a
`NotImplementedError` (for example `General no-intercept operator null
projection is not yet available.`). A `gp` smooth whose penalty has a negative
eigenvalue raises `ValueError: Each modest penalty must be positive
semidefinite.`

## Related pages

- [Choosing a route](choosing_a_route.md)
- [Row sources](row_sources.md)
- [Smoothing selection](smoothing_selection.md)
- [Prediction results](prediction_results.md)
- [Memory and performance](memory_and_performance.md)
- [Fitting routes in the API reference](../api.md#fitting-routes)
