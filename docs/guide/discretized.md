# Discretized fitting

The discretized route is jaxgam's counterpart of mgcv's
`bam(discrete=TRUE)`. Select it with `FitControl(execution="discrete")`. It
fits from a row source and stores each part of the model matrix once per
distinct covariate value, instead of once per row.

**What discretization does.** Each smooth's covariates are reduced to their
distinct values. When a covariate has more distinct values than a grid allows
(1,000 for a smooth of one variable, by default), its values are first
rounded to an evenly spaced grid over their observed range. The basis is then
evaluated once per distinct value, and each row keeps only an index into that
small table. Memory for the model matrix then grows with the number of
distinct values, not with the number of rows.

This is a different estimator from the default fit, as `bam` is from `gam` in
mgcv: the covariates may be rounded, and the smoothing parameters are chosen
by `bam`'s performance iteration rather than by the nested REML optimization
of the default fit. Expect close, not identical, results.

**Use it when**

- you have many rows, and want what `bam(discrete=TRUE)` would give you;
- the model needs smooth types the [streamed route](streamed.md) does not
  accept: thin plate, cyclic, Gaussian process, random-effect and tensor
  smooths, `by=` variables and factor terms all work here.

**Do not use it when**

- you need the exact default-fit estimate, `summary()` or `plot()`: use the
  default fit;
- your random effects have very many levels and dominate the model: see the
  [operator route](operator.md).

The setup block makes the data and the row sources used throughout the page.

```python
import os
import tempfile
import warnings

import numpy as np
import pandas as pd

from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data import DataFrameRowSource, DiscretizationBudget
from jaxgam.families import NegativeBinomial

rng = np.random.default_rng(2)
n = 20_000
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
g = rng.integers(0, 20, n)
eta = np.sin(2 * np.pi * x) + (z - 0.5) ** 2 + rng.normal(scale=0.5, size=20)[g]
data = pd.DataFrame(
    {
        "x": x,
        "z": z,
        "g": pd.Categorical(g),
        "y": eta + rng.normal(scale=0.3, size=n),
        "count": rng.poisson(np.exp(0.5 * eta)),
    }
)
source = DataFrameRowSource(data, response="y")
counts = DataFrameRowSource(data, response="count")
newdata = pd.DataFrame({"x": [0.1, 0.5], "z": [0.3, 0.3], "g": pd.Categorical([0, 1])})
```

See [Row sources](row_sources.md) for the built-in sources, prior weights and
offsets.

## A minimal fit

A discretized fit needs four things:

1. a row source as the data;
2. an explicit `ContinuousDiscretizationControl` (the discretization policy);
3. either `sp=` or `smoothing_strategy="working_model_reml"`;
4. `result="prediction"`.

```python
control = FitControl(
    execution="discrete",
    discretization=ContinuousDiscretizationControl(),
    smoothing_strategy="working_model_reml",
)
fit = GAM("y ~ s(x, bs='cr') + s(z, bs='cr')", control=control).fit(
    source, result="prediction"
)
print(fit.execution_route, fit.converged)
# discrete True
print(fit.convergence_info)
# Gaussian working-model REML converged after 12 iterations and 11 proposals
print(fit.predict(newdata))
# [0.63771438 0.05804595]
```

Leaving out any of the four raises an error:

```python
policy = ContinuousDiscretizationControl()
attempts = {
    "no policy": lambda: FitControl(execution="discrete"),
    "no strategy and no sp": lambda: GAM(
        "y ~ s(x, bs='cr')",
        control=FitControl(execution="discrete", discretization=policy),
    ).fit(source, result="prediction"),
    "default result": lambda: GAM("y ~ s(x, bs='cr')", control=control).fit(source),
    "DataFrame data": lambda: GAM("y ~ s(x, bs='cr')", control=control).fit(
        data, result="prediction"
    ),
}
for name, attempt in attempts.items():
    try:
        attempt()
    except (ValueError, TypeError, NotImplementedError) as error:
        print(f"{name}: {type(error).__name__}: {error}")
```

```text
no policy: ValueError: execution='discrete' requires an explicit discretization policy.
no strategy and no sp: NotImplementedError: Discrete fitting requires explicit fixed sp or opt-in smoothing_strategy='working_model_reml'.
default result: NotImplementedError: Discrete fitting currently supports result='prediction' only; full and inference results retain unsupported row diagnostics.
DataFrame data: TypeError: FitControl(execution='discrete') requires a replayable RowSource.
```

Passing `fit(weights=...)` or `fit(offset=...)` raises `ValueError: weights
and offset must be owned by the RowSource for execution='discrete'; do not
pass separate arrays.`

## The grid size `m`

`ContinuousDiscretizationControl(m=...)` sets the grid. The covariates of one
`s()` term are discretized together as a group; each `te()` or `ti()` margin
and each `by` variable is its own group.

- `m=None` (the default): 1,000 grid points per variable for a group of one
  variable, 100 for two, 25 for three or more. These are mgcv's defaults.
- an integer: used as given for a group of one variable; for a group of `d`
  variables, `round(m ** (1 / d)) + 1` points per variable.
- a tuple: one value per smooth term, in formula order.

A group is rounded only when it has more distinct rows than its grid can
hold. Otherwise its values are kept exactly. On small data nothing is
rounded, and a fixed-`sp` fit then matches the default fit:

```python
small = data.iloc[:800]
fixed = FitControl(execution="discrete", discretization=policy)
discrete_fit = GAM("y ~ s(x, bs='cr') + s(z, bs='cr')", sp=[1.0, 2.0],
                   control=fixed).fit(
    DataFrameRowSource(small, response="y"), result="prediction")
dense_fit = GAM("y ~ s(x, bs='cr') + s(z, bs='cr')", sp=[1.0, 2.0]).fit(small)
print(np.max(np.abs(discrete_fit.coefficients - dense_fit.coefficients)) < 1e-12)
# True
```

`fit.discretization.groups` records each group's decision:

```python
for group in fit.discretization.groups:
    print(group.variables, group.points_per_metric, group.discretized, group.n_unique)
# ('x',) 1000 True 1000
# ('z',) 1000 True 1000
# ('y',) 1000 True 808
```

Here both covariates had more than 1,000 distinct values, so each was rounded
to its 1,000-point grid. The list also has a group for the response and one
for each parametric variable, which the setup frame compresses as mgcv's does.

Because the basis is built on the distinct values, an `m` smaller than a
smooth's `k` fails, for example `ValueError: Basis dimension k=10 exceeds
number of unique data values (5). Reduce k or add more distinct data.` for
`m=5` and `k=10`.

## Families and links

| Family | Links |
|---|---|
| Gaussian | identity |
| Poisson | log, identity, sqrt |
| Binomial | logit, probit, cloglog, log |
| Gamma | inverse, identity, log |
| Negative binomial | log, identity, sqrt |

Gaussian identity models read the source twice; smoothing selection then
works on the stored statistics without further scans. The other families use
`bam`'s performance iteration (mgcv's `bgam.fitd`): each iteration reads the
source once, forms the working weights and working response, and takes one
smoothing-parameter step. The Gamma scale is estimated; Poisson and Binomial
use a scale of one. Anything else raises, for example `NotImplementedError:
Discrete fitting supports Gaussian identity, the Poisson (log, identity,
sqrt), Binomial (logit, probit, cloglog, log) and Gamma (inverse, identity,
log) families, and the negative binomial (log, identity, sqrt); got
Gaussian(link=LogLink).`

## Fixed or estimated smoothing

An explicit `sp=` always gives a fixed-smoothing fit (`lambda_strategy` is
`"fixed"`), whatever `smoothing_strategy` says. Without `sp`,
`smoothing_strategy="working_model_reml"` selects the smoothing parameters by
`bam`'s method: one Newton proposal on each iteration's working model
(mgcv's `Sl.fitChol`), until the penalized deviance stops changing.

`optimizer="efs"` does not apply here: without `sp` it raises
`NotImplementedError: optimizer='efs' does not apply to discrete fitting;
smoothing_strategy='working_model_reml' selects its smoothing parameters.`
With `sp` it has no effect.

!!! note "Compare fits by deviance and fitted values"
    With many smooths the REML criterion is often flat in some direction
    (for example when one smoothing parameter heads towards a very large
    value), and the point where the search stops is then not sharply
    determined. This is true of `bam` too: the tests record cases where
    mgcv's own selected smoothing parameters move under input changes at
    rounding level. Compare the deviance and the fitted values, not the
    smoothing parameters.

The discretized fit above against the default fit on the same rows:

```python
dense = GAM("y ~ s(x, bs='cr') + s(z, bs='cr')").fit(data)
print(fit.deviance, dense.deviance)
print(np.max(np.abs(fit.predict(data) - dense.predict(data))))
print(fit.smoothing_params, dense.smoothing_params)
# 7414.301445351851 7414.480596587477
# 0.0011857272687984428
# [  53.3540798 6202.072597 ] [  52.70551092 6139.62446449]
```

## Random effects

Factors are not inferred from the data type on this route. Every factor (a
parametric factor, a `by` factor, or the grouping variable of `bs="re"`) must
be listed in `factor_names`; any other column is treated as a numeric
covariate:

```python
with_factor = FitControl(
    execution="discrete",
    discretization=ContinuousDiscretizationControl(factor_names=("g",)),
    smoothing_strategy="working_model_reml",
)
fit_re = GAM("y ~ s(x, bs='cr') + s(g, bs='re')", control=with_factor).fit(
    source, result="prediction"
)
print(fit_re.converged, fit_re.smoothing_params.shape)
# True (2,)
```

Without the declaration a categorical column raises `TypeError: Metric column
'g' must be real numeric data, got dtype category. Variables not listed in
factor_names are discretized as metric. ...`. An integer-coded grouping
column that is not declared is silently treated as one numeric covariate.
List only columns the formula uses: a name it does not use raises
`ValueError: factor_names must be distinct referenced variables`.

A random-effect level that is missing from the prediction data or was not
seen in training contributes zero to the prediction (where mgcv's
`predict.bam` gives `NA` for a missing group). A multi-variable random effect
such as `s(x, g, bs="re")` is refused (see Limits).

## Negative binomial

With `family="nb"` theta is estimated by mgcv's `estimate.theta` at every
iteration from the second. Each estimate reads the source several times, so
the fit makes many more scans. `NegativeBinomial(theta, fixed=True)` holds
theta:

```python
estimated = GAM("count ~ s(x, bs='cr') + s(z, bs='cr')", family="nb",
                control=control).fit(counts, result="prediction")
held = GAM("count ~ s(x, bs='cr') + s(z, bs='cr')",
           family=NegativeBinomial(3.0, fixed=True),
           control=control).fit(counts, result="prediction")
print(round(estimated.theta, 2), estimated.source_scans)
print(round(held.theta, 2), held.source_scans)
# 13.45 45
# 3.0 10
```

## Budgets

Two kinds of limit apply. The `DiscretizationBudget` (passed as
`ContinuousDiscretizationControl(budget=...)`) bounds the setup pass that
builds the compact tables. `FitControl.memory_budget_bytes` bounds the fit.
Each is a check made before allocation, against arrays the code knows it will
need, and each raises `MemoryError` naming the step that did not fit.

| Field | Default | Message names |
|---|---|---|
| `DiscretizationBudget.max_table_bytes` | 64 MiB | "table", "lookup", "basis setup", "predictor workspace" or "metadata" |
| `DiscretizationBudget.max_selector_bytes` | 256 MiB | "selector" |
| `DiscretizationBudget.max_batch_bytes` | 64 MiB | "batch workspace" (grows with `FitControl.batch_rows`) |
| `DiscretizationBudget.max_unique_rows` | 1,000,000 | "uniquecombs rows exceed row cap" (the message reports bytes, not rows) |
| `FitControl.memory_budget_bytes` | 512 MiB | "Discrete fit workspace requires ..." |

The default table budget fits the cubic regression splines above but not, at
this data size, the default thin plate smooth `s(x)`:

```python
try:
    GAM("y ~ s(x)", control=control).fit(source, result="prediction")
except MemoryError as error:
    print(error)
# compact smooth predictor workspace exceeds budget: requires 89472425 bytes,
# budget is 67108864.
```

The byte count varies slightly between runs. Raise the table budget, or use a
coarser grid:

```python
larger_table = ContinuousDiscretizationControl(
    budget=DiscretizationBudget(max_table_bytes=256 * 1024 * 1024)
)
fit_tp = GAM("y ~ s(x)", control=FitControl(
    execution="discrete", discretization=larger_table,
    smoothing_strategy="working_model_reml",
)).fit(source, result="prediction")
coarse = ContinuousDiscretizationControl(m=200)
fit_tp_coarse = GAM("y ~ s(x)", control=FitControl(
    execution="discrete", discretization=coarse,
    smoothing_strategy="working_model_reml",
)).fit(source, result="prediction")
print(fit_tp.converged, fit_tp_coarse.converged)
# True True
```

When a "batch workspace" message appears, lower `FitControl.batch_rows` or
raise `max_batch_bytes`. When the fit itself does not fit:

```python
try:
    GAM("y ~ s(x, bs='cr') + s(z, bs='cr')", control=FitControl(
        execution="discrete", discretization=policy,
        smoothing_strategy="working_model_reml", memory_budget_bytes=10_000_000,
    )).fit(source, result="prediction")
except MemoryError as error:
    print(error)
# Discrete fit workspace requires 26306232 bytes, exceeding
# FitControl.memory_budget_bytes=10000000.
```

`fit.fit_peak_bytes` is the figure this check uses, so a successful fit tells
you how much it needed.

### Row selectors on disk

During setup each row's index into the compact tables is held in memory,
charged to `max_selector_bytes`. `selector_memmap_directory` writes these
selectors to files in an existing directory instead. The files
(`compact-selector-{group}.dat`) are removed when the fit ends, and the
directory must not already contain them:

```python
with tempfile.TemporaryDirectory() as directory:
    on_disk = ContinuousDiscretizationControl(selector_memmap_directory=directory)
    fit_disk = GAM("y ~ s(x, bs='cr') + s(z, bs='cr')", control=FitControl(
        execution="discrete", discretization=on_disk,
        smoothing_strategy="working_model_reml",
    )).fit(source, result="prediction")
    print(fit_disk.converged, os.listdir(directory))
# True []
```

## Prediction policy

`ContinuousDiscretizationControl(prediction_policy=...)` sets how new data are
predicted. Both reuse the fitted knots and constraints.

| Value | What it does |
|---|---|
| `"continuous_frozen_basis"` (default) | Evaluates each new value exactly, as `predict.gam` does. Rows are independent, so a prediction never depends on the rest of the request or on batching. |
| `"mgcv_predict_bamd"` | Follows mgcv's `predict.bamd`: the whole request is first discretized on the default (`m=None`) grid, then evaluated. Results then depend on what else is in the request, as in mgcv, but not on how it is batched. |

The difference shows when a request has more distinct values than the grid:

```python
bamd = FitControl(
    execution="discrete",
    discretization=ContinuousDiscretizationControl(prediction_policy="mgcv_predict_bamd"),
    smoothing_strategy="working_model_reml",
)
fit_bamd = GAM("y ~ s(x, bs='cr') + s(z, bs='cr')", control=bamd).fit(
    source, result="prediction"
)
print(fit.predict(data)[:2], fit.predict(data.iloc[:5])[:2])
# [1.13731702 1.16322397] [1.13731702 1.16322397]
print(fit_bamd.predict(data)[:2], fit_bamd.predict(data.iloc[:5])[:2])
# [1.13767284 1.16329154] [1.13731702 1.16322397]
```

With the whole training data as the request, `mgcv_predict_bamd` rounds the
20,000 values to the grid first; five rows are evaluated exactly.

A model without smooths has nothing to discretize; its result records
`"continuous_frozen_basis"`.

## Other options

| Option | Default | What it does |
|---|---|---|
| `ContinuousDiscretizationControl.factor_names` | `()` | Columns to treat as factors. |
| `ContinuousDiscretizationControl.seed` | `8547` | Seed of the shuffle that mgcv applies to the compact rows (a signed 32-bit integer). |
| `ContinuousDiscretizationControl.pair_policy` | `"histogram_or_direct"` | How a pair of compact blocks is reduced: by a weight histogram when that is no larger than gathering rows, or always by gathering rows (`"direct_only"`). |
| `FitControl.working_model_max_iterations` | `200` | Limit of the performance iteration. |
| `FitControl.working_model_epsilon` | `1e-7` | Relative change of the penalized deviance that stops it. |
| `FitControl.uncertainty` | `"none"` | `"fisher"` or `"covariance"` keeps what `predict(se_fit=True)` needs. |
| `FitControl.batch_rows` | `65_536` | Rows per source scan. |

Field-level detail is in the [API reference](../api.md#fit-control).

## What you get back

A `GAMPredictionResult` (see the [API reference](../api.md#gampredictionresult)),
with no `summary()` or `plot()`.

| Attribute | Meaning on this route |
|---|---|
| `converged`, `n_iter`, `convergence_info` | Whether the iteration met its stopping test, and how it stopped. |
| `execution_route` | `"discrete"`. |
| `smoothing_strategy`, `lambda_strategy` | `"fixed"` or `"working_model_reml"`. |
| `smoothing_params`, `deviance`, `scale`, `edf_total` | The fitted values. `edf_total` is the total effective degrees of freedom. |
| `theta` | The negative binomial theta (fixed or estimated), else `None`. |
| `score` | `bam`'s criterion at the stopping point. |
| `source_scans`, `batches_scanned` | Scans made by the fit, not by the setup pass. |
| `known_workspace_bytes` | The compact design and solver workspace. |
| `fit_peak_bytes`, `fit_memory_budget_bytes` | The figure checked against `memory_budget_bytes`, and that budget. |
| `working_model_proposal_evaluations` | Smoothing-parameter proposals tried (0 for a fixed-`sp` Gaussian fit). |
| `prediction_policy`, `discretization` | The prediction policy and the per-group grid record. |

### Telling whether the fit converged

A fixed-`sp` Gaussian fit is a direct solve and always reports
`converged=True`; its `n_iter` counts the scale iterations. Every other fit
stops when, from the third iteration, the relative change of the penalized
deviance is below `working_model_epsilon` (and, when a scale is estimated,
so is the change of the log scale). At `working_model_max_iterations` the fit
is still returned, with `converged=False` and a `RuntimeWarning`:

```python
short = FitControl(execution="discrete", discretization=policy,
                   smoothing_strategy="working_model_reml",
                   working_model_max_iterations=3)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fit_short = GAM("y ~ s(x, bs='cr') + s(z, bs='cr')", control=short).fit(
        source, result="prediction")
print(fit_short.converged, [str(w.message) for w in caught])
# False ['Gaussian working-model REML did not converge within its iteration limit.']
```

For the other families the warning reads `{family} discrete performance
iteration did not converge within its iteration limit.`

## Limits

| Not accepted | Error |
|---|---|
| `linear_solver="qr"` | `NotImplementedError: linear_solver='qr' currently requires execution='stream'.` |
| `linear_solver="pcg"` | `NotImplementedError: linear_solver='pcg' currently requires execution='operator'.` |
| A multi-variable random effect, for example `s(x, g, bs='re')` | `NotImplementedError: s(x,g, bs='re') is not supported by execution='discrete': mgcv's bam discretizes each variable of a multi-variable random effect as a separate tensor margin, which the compact setup does not reproduce.` |
| Smooths that group shared variables differently, for example `s(x, z) + s(x)` | `ValueError: continuous discretization has incompatible nesting` |
| A non-finite covariate value | `ValueError: Metric column 'x' contains non-finite values` |
| `method` other than `"REML"` (including `"fREML"`) | `ValueError` when the `GAM` is created |

Other known rough edges:

- `gaussian_compression=True` is not used here; the fit runs unchanged and
  `execution_fallback_reason` reads `Gaussian QR compression requires
  execution 'dense' or 'stream'.`
- As in mgcv, the performance iteration does not check that the mean stays in
  the family's range, and its second iteration has no step control. A first
  step that leaves the range raises `FloatingPointError: Non-finite
  working-model deviance at iteration 2.`, where `bam` also stops.
- A fixed-`sp` Gaussian fit whose coefficient system is not positive definite
  (an unidentifiable model) raises `LinAlgError`: rank-deficient fitting is
  not available for that fit.
- A row source you write yourself that does not declare
  `has_explicit_weights` and `has_explicit_offset` is treated as supplying
  both, as `bam` treats supplied `weights=` and `offset=` arguments. That
  changes mgcv's seeded padding of the compact rows, and so the fitted
  numbers slightly. See [Row sources](row_sources.md).

## Related pages

- [Choosing a route](choosing_a_route.md)
- [Row sources](row_sources.md)
- [Prediction results](prediction_results.md)
- [Memory and performance](memory_and_performance.md)
- [Fitting routes in the API reference](../api.md#fitting-routes)
