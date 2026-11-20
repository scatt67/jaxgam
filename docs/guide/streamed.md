# Streamed fitting

The streamed route fits a GAM from a row source, reading the data in batches,
so the full model matrix is never held in memory. Select it with
`FitControl(execution="stream")`.

**Use it when**

- the data do not fit in memory as a model matrix, but can be read again and
  again in batches (from arrays, a DataFrame or `.npy` files on disk);
- the model uses one-dimensional cubic regression splines (`bs="cr"` or
  `bs="cs"`) and numeric parametric terms.

**Do not use it when**

- the model needs thin plate (the default `s(x)`), cyclic, random-effect or
  tensor smooths, a `by=` variable or factor terms: use the
  [discretized route](discretized.md) or the default fit;
- you need `summary()`, `plot()` or per-term EDF: a streamed fit returns only a
  prediction result.

Every example on this page runs in order in one Python session. The setup
block makes the data and a row source used throughout.

```python
import numpy as np
import pandas as pd

from jaxgam import GAM, FitControl
from jaxgam.data import DataFrameRowSource
from jaxgam.families import Poisson

rng = np.random.default_rng(1)
n = 2000
x = rng.uniform(0, 1, n)
z = rng.normal(size=n)
eta = np.sin(2 * np.pi * x) + 0.3 * z
data = pd.DataFrame(
    {
        "x": x,
        "z": z,
        "y": eta + rng.normal(scale=0.3, size=n),
        "count": rng.poisson(np.exp(eta)),
    }
)
source = DataFrameRowSource(data, response="y")
counts = DataFrameRowSource(data, response="count")
newdata = pd.DataFrame({"x": [0.1, 0.5, 0.9], "z": [0.0, 0.0, 0.0]})
```

A row source is an object that can replay the same rows, batch by batch,
every time it is scanned. See [Row sources](row_sources.md) for the built-in
sources, prior weights and offsets, and how to write your own.

## Fixed smoothing parameters

Pass `sp=` to fix the smoothing parameters. A streamed fit always needs
`result="prediction"`.

```python
control = FitControl(execution="stream", batch_rows=500)
fit = GAM("y ~ z + s(x, bs='cr')", sp=[1.0], control=control).fit(
    source, result="prediction"
)
print(fit.execution_route, fit.converged, fit.source_scans, fit.batches_scanned)
# stream True 11 44
print(fit.predict(newdata))
# [ 0.57438901 -0.03205741 -0.57433759]
```

Each PIRLS iteration (penalized iteratively reweighted least squares, the
coefficient fit) reads the whole source once, so `source_scans` grows with the
number of iterations. The scans made by the setup pass (knots, bases and
constraints) are not counted.

### Choosing the linear solver

`FitControl.linear_solver` picks how each coefficient system is solved:
`"cholesky"` (the default) or `"qr"`.

| Family and link | `"cholesky"` | `"qr"` |
|---|---|---|
| Gaussian identity, Poisson log, Binomial logit | yes | yes |
| Other links of Gaussian, Poisson and Binomial | no | yes |
| Gamma (any supported link) | no | yes |
| Negative binomial (log, identity, sqrt) | no | yes |

The Cholesky solve covers only the families whose Fisher information equals
the observed information and whose scale is known or is the Gaussian
residual scale. Everything else asks for QR:

```python
try:
    GAM("count ~ z + s(x, bs='cr')", family=Poisson(link="sqrt"), sp=[1.0],
        control=control).fit(counts, result="prediction")
except NotImplementedError as error:
    print(error)
# Streamed observed-information fitting requires linear_solver='qr'.

qr = FitControl(execution="stream", batch_rows=500, linear_solver="qr")
fit = GAM("count ~ z + s(x, bs='cr')", family=Poisson(link="sqrt"), sp=[1.0],
          control=qr).fit(counts, result="prediction")
print(fit.execution_route, fit.converged)
# stream_qr True
```

For the negative binomial the message is `Streamed NB observed-information
fitting requires linear_solver='qr'.`

A fixed `sp` does not always mean nothing else is estimated. For Gamma and for
a non-identity Gaussian link the scale is still estimated, and for
`family="nb"` the theta parameter is still estimated, with the smoothing
parameters held. `lambda_strategy` is still `"fixed"`, and
`convergence_info` then holds the optimizer's message. To hold the negative
binomial theta too, pass `NegativeBinomial(theta, fixed=True)`.

## Estimated smoothing by exact REML

Omit `sp` to estimate the smoothing parameters by REML. On this route that
requires `linear_solver="qr"`:

```python
try:
    GAM("y ~ z + s(x, bs='cr')", control=control).fit(source, result="prediction")
except NotImplementedError as error:
    print(error)
# Exact streamed REML requires linear_solver='qr'.

fit = GAM("y ~ z + s(x, bs='cr')", control=qr).fit(source, result="prediction")
print(fit.execution_route, fit.lambda_strategy, fit.converged)
# stream_exact_reml_qr exact_reml True
print(fit.convergence_info)
# CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL; 1 source step window(s).
diagnostics = fit.exact_optimizer_diagnostics
print(diagnostics.status, diagnostics.n_evaluations, diagnostics.n_accepted)
# 0 16 14
```

The criterion is the same exact REML score as the default fit, but the
optimizer is different: SciPy's L-BFGS-B with exact gradients, not the Newton
iteration of the default fit. It works on the log smoothing parameters
(bounded to [-40, 40]), plus the log scale for families with an unknown scale
and log theta for a negative binomial with estimated theta. Every score
evaluation is a full coefficient fit over the source, so this sub-route makes
many scans (146 in the example above, including 2 start-up scans). Each step
is limited to 5 on the log scale per "window"; at most 20 windows are opened.

`exact_optimizer_diagnostics` holds the L-BFGS-B `status` (0 means it met its
stopping test), `n_evaluations`, `n_accepted`, `projected_gradient_inf`, the
accepted score history and the start-up scan counts. A model with no
penalty, a known scale and no theta skips the optimizer: it reports
`lambda_strategy="exact_reml_no_parameters"`.

## Streamed Fellner-Schall

`optimizer="efs"` selects the smoothing parameters by the extended
Fellner-Schall update (mgcv's `efsudr` method, which the dense
`optimizer="efs"` fit also follows) instead of L-BFGS-B. It also requires
`linear_solver="qr"`:

```python
fit = GAM("y ~ z + s(x, bs='cr')", optimizer="efs", control=qr).fit(
    source, result="prediction"
)
print(fit.execution_route, fit.lambda_strategy, fit.converged, fit.convergence_info)
# stream_efs_qr efs_reml True deviance_change
efs = fit.optimizer_diagnostics
print(fit.source_scans, efs.startup_source_scans + efs.provider_source_scans)
# 8 26
```

The top-level `source_scans` counts only the final accepted fit; the sum of
the two diagnostics counts is the whole search.

It accepts Gaussian, Binomial, Poisson, Gamma and the negative binomial with
a log, identity or sqrt link; estimated theta is updated inside the fit.
Other families raise `Streamed EFS supports registered Gaussian, Binomial,
Poisson, Gamma, and NegativeBinomial families only.` The limits and
tolerances come from `FitControl(efs=EFSControl(...))`; see
[Smoothing selection](smoothing_selection.md).

`convergence_info` is `"score_window"` or `"deviance_change"` when it
converged. With no penalties, Fellner-Schall has nothing to select: the fit
runs one coefficient fit (`execution_route="stream_parametric_qr"`) and, for
the negative binomial, leaves theta at its starting value and reports
`theta=None`.

## Streamed Gaussian QR

For a Gaussian identity-link model, `gaussian_compression=True` reads the
source once to build a weighted QR reduction, then selects the smoothing
parameters using only coefficient-sized arrays, with no further scans:

```python
gaussian_qr = FitControl(execution="stream", batch_rows=500,
                         gaussian_compression=True)
fit = GAM("y ~ z + s(x, bs='cr')", control=gaussian_qr).fit(
    source, result="prediction"
)
print(fit.execution_route, fit.lambda_strategy, fit.source_scans)
# stream_gaussian_qr newton_reml 1
```

It works with either `linear_solver`. It needs the exact `Gaussian` family
with the identity link, at least one penalty, a full-rank coefficient map, and
prior weights between `1e-10` and `1e10`. When a request does not qualify, the
fit takes the route it would take without the request and says why in
`execution_fallback_reason`:

```python
request = FitControl(execution="stream", batch_rows=500, linear_solver="qr",
                     gaussian_compression=True)
fit = GAM("count ~ z + s(x, bs='cr')", family="poisson", control=request).fit(
    counts, result="prediction"
)
print(fit.execution_route)
# stream_exact_reml_qr
print(fit.execution_fallback_reason)
# Gaussian QR requires the exact Gaussian family with identity link
```

With estimated smoothing and the default `"cholesky"` solver there is no
route to fall back to, so an ineligible request raises
`Estimated-sp RowSource fitting requires an eligible Gaussian QR route (...)
or linear_solver='qr' for exact streamed REML.` With `optimizer="efs"` the
request is not used and the reason is `Gaussian QR compression requires
Newton smoothing selection.`

Up to 50 penalties the smoothing parameters are selected by Newton
(`lambda_strategy="newton_reml"`). Above 50 penalties L-BFGS-B is used
instead (`"lbfgs_reml"`), limited to 100 iterations. If it stops at that
limit the fit returns with `converged=False` and `convergence_info="iteration
limit"`. With `sp=` the smoothing parameters are held, the scale is still
estimated, and `lambda_strategy` is `"fixed"`.

## Options that matter

| Option | Default | What it does |
|---|---|---|
| `execution` | `"dense"` | `"stream"` selects this route. |
| `linear_solver` | `"cholesky"` | `"qr"` is needed for estimated smoothing, Fellner-Schall and most families. `"pcg"` is refused. |
| `batch_rows` | `65_536` | Rows per batch in every fit scan, and the default batch size of `predict_iter`. |
| `memory_budget_bytes` | 512 MiB | Bound on the arrays the setup pass and the fit know they will hold. |
| `output_budget_bytes` | 512 MiB | Bound on what one `predict` call may return. |
| `uncertainty` | `"none"` | `"fisher"` or `"covariance"` keeps what standard errors need. |
| `gaussian_compression` | `False` | Requests the Gaussian QR sub-route. |
| `efs` | `EFSControl()` | Fellner-Schall limits, used with `optimizer="efs"`. |

There is no option for the PIRLS iteration limit or tolerance on this route.
Field-level detail is in the [API reference](../api.md#fit-control).

## Batch size and memory budget

`memory_budget_bytes` is a check made before arrays are allocated, against
the arrays the setup pass and the fit know they will need. It is not a cap on
the process: the row source's own storage and native LAPACK or XLA workspace
are outside it. A failed check raises `MemoryError` before the work starts.

The setup pass shrinks its own scan size until it fits the budget. If not even
one row fits, it stops before reading the data. A model with a cubic smooth
always needs a fixed 16 MiB allowance for exact knot placement, so budgets
below about 17 MB fail this way:

```python
try:
    GAM("y ~ s(x, bs='cr')", sp=[1.0],
        control=FitControl(execution="stream", memory_budget_bytes=1_000_000)
        ).fit(source, result="prediction")
except MemoryError as error:
    print(error)
# Streamed setup requires at least 17318011 bytes to scan one row, exceeding
# the setup memory budget of 1000000 bytes (FitControl.memory_budget_bytes).
```

The fit does not shrink its scans: its check charges a full `batch_rows`
batch, whatever the number of rows in the source. When it fails, lower
`batch_rows` (or raise the budget):

```python
tight = FitControl(execution="stream", memory_budget_bytes=18_000_000)
try:
    GAM("y ~ z + s(x, bs='cr')", sp=[1.0], control=tight).fit(
        source, result="prediction"
    )
except MemoryError as error:
    print(error)
# Known streamed PIRLS workspace requires 21507776 bytes, exceeding
# FitControl.memory_budget_bytes=18000000. Setup scans are budgeted
# separately.

smaller = FitControl(execution="stream", memory_budget_bytes=18_000_000,
                     batch_rows=4096)
fit = GAM("y ~ z + s(x, bs='cr')", sp=[1.0], control=smaller).fit(
    source, result="prediction"
)
print(fit.converged, fit.known_workspace_bytes)
# True 1355456
```

The QR solver's message reads `Known streamed QR PIRLS workspace requires
...`. The Gaussian QR sub-route raises `Known streamed Gaussian QR workspace
requires ...` and does not fall back. See
[Memory and performance](memory_and_performance.md) for every budget.

## Standard errors

By default a streamed fit keeps nothing for standard errors, and
`predict(se_fit=True)` raises `RuntimeError`. Fit with
`uncertainty="fisher"` to keep the factor of the penalized Fisher information:

```python
fisher = FitControl(execution="stream", linear_solver="qr", uncertainty="fisher")
fit = GAM("y ~ z + s(x, bs='cr')", control=fisher).fit(source, result="prediction")
mean, se = fit.predict(newdata, se_fit=True)
print(se)
# [0.02150264 0.01786659 0.02194168]
Vp = fit.materialize_covariance(memory_budget_bytes=10_000_000)
print(Vp.shape)
# (11, 11)
```

`materialize_covariance` builds the coefficient covariance on request, under
its own budget. `uncertainty="covariance"` builds it during the fit instead
(checked against `memory_budget_bytes`); read it from `fit.to_predictor().Vp`.
See [Prediction results](prediction_results.md).

## What you get back

A `GAMPredictionResult` (see the [API reference](../api.md#gampredictionresult)).
It has `predict`, `predict_iter`, `predict_matrix`, `to_predictor` and
`materialize_covariance`, but no `summary()`, `plot()` or per-term EDF.

| Attribute | Meaning on this route |
|---|---|
| `converged`, `convergence_info`, `n_iter` | Whether the fit, or the smoothing search, met its stopping test, and how it stopped. |
| `execution_route` | `"stream"`, `"stream_qr"`, `"stream_exact_reml_qr"`, `"stream_efs_qr"`, `"stream_parametric_qr"` or `"stream_gaussian_qr"`. |
| `lambda_strategy` | `"fixed"`, `"exact_reml"`, `"exact_reml_no_parameters"`, `"efs_reml"`, `"newton_reml"` or `"lbfgs_reml"`. |
| `smoothing_params`, `scale`, `theta`, `edf_total`, `deviance`, `score` | The fitted values of these quantities. |
| `source_scans`, `batches_scanned` | Fit scans, not setup scans. Exact REML adds its start-up scans. On Fellner-Schall they count only the final fit; the totals are `optimizer_diagnostics.startup_source_scans + provider_source_scans`. |
| `known_workspace_bytes` | The fit's known-array bound. `None` on Fellner-Schall and on the parametric bypass. |
| `exact_optimizer_diagnostics` | Exact REML only (see above). |
| `optimizer_diagnostics` | Fellner-Schall only: an `EFSOptimizerDiagnostics`. |
| `execution_fallback_reason` | Why a `gaussian_compression=True` request was not used, else `None`. |

### Telling whether the fit converged

Check `fit.converged` and read `fit.convergence_info`. The streamed route
issues no warning when it stops short; a fit that did not converge is
returned with `converged=False`. The messages you can see:

- fixed `sp`: `"fixed sp streamed PIRLS"` or `"fixed sp streamed PIRLS did not
  converge"`. The PIRLS limit (100 iterations) cannot be changed here.
- exact REML: the L-BFGS-B message plus the window count. If the window limit
  was reached it adds `Window limit 20 reached with descent beyond the last
  window.`
- Fellner-Schall: `"score_window"` or `"deviance_change"` when converged,
  otherwise `"iteration limit reached"` or a failure label. Raise
  `EFSControl(outer_limit=...)` for the first.
- Gaussian QR: `"iteration limit"` or `"step failed"` when not converged.

Some failures raise instead of returning. Under exact REML, a coefficient fit
that does not converge at any trial raises `RuntimeError` (for example
`Regular streamed REML requires a converged, valid final source state.`, or
`NB streamed REML requires a converged valid coefficient state` for the
negative binomial). An unpenalized
model matrix that is rank deficient raises `LinAlgError`. A source that
changes during the fit raises `RuntimeError`.

## Limits

| Not accepted | Error |
|---|---|
| `result="full"` or `"inference"` (the default is `"full"`) | `NotImplementedError: Streamed fitting supports result='prediction' only; full and inference results retain unsupported row diagnostics.` |
| A DataFrame or dict instead of a row source | `TypeError: FitControl(execution='stream') requires a replayable RowSource.` |
| `fit(weights=...)` or `fit(offset=...)` | `ValueError: weights and offset must be owned by the RowSource for execution='stream'; do not pass separate arrays.` |
| `linear_solver="pcg"` | `ValueError: linear_solver='pcg' requires execution='operator'.` |
| `s(x)` (thin plate), `bs="cc"`, `bs="re"`, `bs="gp"`, multi-variable `s()` | `NotImplementedError: Prepared setup currently supports only univariate cr/cs smooths.` |
| `te()`, `ti()`, or any `by=` | `NotImplementedError: Prepared setup does not yet support by-variable or tensor smooths.` |
| A factor parametric term | `NotImplementedError: Prepared setup currently supports numeric parametric terms only.` |
| Two cubic smooths of the same variable | `NotImplementedError: Prepared setup does not yet support overlapping cubic terms.` |

Accepted: one-dimensional `bs="cr"` and `bs="cs"` smooths (default `k=10`),
numeric parametric terms, with or without an intercept. The formula parser's
own limits (for example no `id=` and no `offset()` term) apply on every route;
see [Formula syntax](../api.md#formula-syntax).

## Related pages

- [Choosing a route](choosing_a_route.md)
- [Row sources](row_sources.md)
- [Smoothing selection](smoothing_selection.md)
- [Prediction results](prediction_results.md)
- [Memory and performance](memory_and_performance.md)
- [Fitting routes in the API reference](../api.md#fitting-routes)
