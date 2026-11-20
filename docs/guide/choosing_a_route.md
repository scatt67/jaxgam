# Choosing a route

jaxgam can fit the same kind of model in several ways. This page tells you
which one to use for your data and what each one gives back. A *route* is the
way the fit is carried out; you choose it with `FitControl(execution=...)`
and, for smoothing parameter selection, with a few `GAM` arguments.

**Use the default fit when** your data fit in memory as a pandas DataFrame.
It supports every family, smooth type and result mode.

**Pick another route when** the data do not fit in memory, when you want the
estimates of mgcv's `bam(discrete=TRUE)` rather than `gam()`, or when a random
effect has a very large number of levels.

## The six ways to fit at a glance

| Way to fit | How you ask for it | Data | How smoothing parameters are chosen | Details |
|---|---|---|---|---|
| Default fit (dense route) | `GAM(formula)` | DataFrame or dict | Exact REML by Newton's method, as mgcv's `gam(method="REML")`, or fixed with `sp=` | [Smoothing parameter selection](smoothing_selection.md) |
| Fellner-Schall | `GAM(formula, optimizer="efs")` | DataFrame, or a row source on the streamed and operator routes | Extended Fellner-Schall updates, as `gam(method="REML", optimizer="efs")` | [Smoothing parameter selection](smoothing_selection.md) |
| Gaussian QR compression | `FitControl(gaussian_compression=True)` | DataFrame, or a row source on the streamed route | Exact REML on one QR reduction of the weighted design (Gaussian, identity link) | [Smoothing parameter selection](smoothing_selection.md) |
| Streamed route | `FitControl(execution="stream", ...)` | Row source | Fixed, exact REML or Fellner-Schall, reading the rows in batches | [Streamed fitting](streamed.md) |
| Discretized route | `FitControl(execution="discrete", discretization=...)` | Row source | Fixed, or mgcv's `bam(discrete=TRUE)` performance iteration | [Discretized fitting](discretized.md) |
| Operator route | `FitControl(execution="operator", linear_solver="pcg")` | Row source | Fixed `sp=`; Fellner-Schall for one supported Poisson model shape | [Operator fitting](operator.md) |

A *row source* is an object that hands the fit your data in batches of rows,
and can do so again as many times as the fit asks. See
[Row sources](row_sources.md).

Fellner-Schall and Gaussian QR compression are not separate places to put
data: they change how smoothing parameters are found on the dense route and,
where the table says so, on a row-source route.

## Which route for your situation

| Your situation | Use |
|---|---|
| Data fit in memory, you want the same estimates as `gam(method="REML")` | Default fit |
| Data fit in memory, you want mgcv's `optimizer="efs"` behaviour | Fellner-Schall |
| Data fit in memory, Gaussian family with identity link | Default fit with `gaussian_compression=True` |
| Data do not fit in memory, you want `gam()` estimates, the model has only one-variable cubic regression splines (`bs="cr"` or `"cs"`) and numeric terms | Streamed route |
| Data do not fit in memory, you accept `bam(discrete=TRUE)` estimates, any smooth type | Discretized route |
| A random effect `s(g, bs="re")` with very many levels | Operator route |

The streamed route reads every row on every pass and gives the exact
estimates; the discretized route first rounds covariates onto grids, as
`bam(discrete=TRUE)` does, so its estimates are those of that model, not of
`gam()`.

## What each route takes and returns

| Route | `data` argument | Weights and offset | Result modes | `FitControl.uncertainty` |
|---|---|---|---|---|
| Dense (including Fellner-Schall and Gaussian QR) | DataFrame or dict | `fit(data, weights=..., offset=...)` | `"full"`, `"inference"`, `"prediction"` | `"none"`, `"fisher"`, `"covariance"` (prediction mode only) |
| Streamed | Row source | Held by the row source | `"prediction"` only | `"none"`, `"fisher"`, `"covariance"` |
| Discretized | Row source | Held by the row source | `"prediction"` only | `"none"`, `"fisher"`, `"covariance"` |
| Operator | Row source | Held by the row source | `"prediction"` only | `"none"` only |

`fit()` defaults to `result="full"`, so a row-source fit must pass
`result="prediction"`. Otherwise it raises, for example on the streamed
route:

```text
NotImplementedError: Streamed fitting supports result='prediction' only; full and inference results retain unsupported row diagnostics.
```

A row source given to the dense route raises:

```text
TypeError: A RowSource requires FitControl execution 'operator', 'stream' or 'discrete'; the dense route accepts only a DataFrame or dict.
```

## The three result modes

`GAM.fit(..., result=...)` chooses how much the result keeps.

| Mode | Returns | Keeps | Use it for |
|---|---|---|---|
| `"full"` (default) | `GAMResults` | Training data, fitted values, `Vp`, per-smooth EDF | `summary()`, `plot()`, in-sample diagnostics |
| `"inference"` | `GAMInferenceResult` | `Vp`, per-smooth EDF, scalar diagnostics; no training rows | New-data prediction with standard errors |
| `"prediction"` | `GAMPredictionResult` | Coefficients, scalar diagnostics, and covariance only if you ask for it | Prediction on new data; the only mode on row-source routes |

Only the dense route offers all three. See
[Prediction results](prediction_results.md) for the prediction result and
[Fit result modes](../api.md#fit-result-modes) in the API reference for the
field lists.

## What is the same on every route

- **Formula syntax.** Every route parses the same mgcv-style formula
  (`s()`, `te()`, `ti()`, `bs=`, `k=`, `by=`). Each route accepts a subset:
  the streamed route takes one-variable `bs="cr"` or `"cs"` smooths and
  numeric terms; the operator route takes numeric terms, `cr`/`cs`/`cc`,
  `tp`/`ts`/`gp` and `re` smooths; the discretized and dense routes take
  every smooth type. See [Formula syntax](../api.md#formula-syntax).
- **Families.** The same family names and objects (`"gaussian"`,
  `"binomial"`, `"poisson"`, `"gamma"`, `"nb"`, or an instance such as
  `Gamma(link="log")`). Which family and link combinations a route accepts is
  listed on its own page. See [Families](../api.md#families).
- **Prediction.** Every result has `predict(newdata, pred_type="response",
  se_fit=False, offset=None)` with the same meaning, and `to_predictor()`.
- **Convergence reporting.** Every result has `converged`, `n_iter`,
  `convergence_info`, `smoothing_params` and `execution_route`.

## Two short examples

The setup block makes a small data set used by both examples.

```python
import numpy as np
import pandas as pd
from jaxgam import GAM, FitControl
from jaxgam.data import DataFrameRowSource

rng = np.random.default_rng(0)
n = 500
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
y = np.sin(2 * np.pi * x) + 0.5 * z + rng.normal(0, 0.3, n)
data = pd.DataFrame({"x": x, "z": z, "y": y})
newdata = pd.DataFrame({"x": [0.25, 0.75], "z": [0.5, 0.5]})
```

The default fit takes the DataFrame and returns a full result:

```python
fit = GAM("y ~ s(x) + z").fit(data)
print(fit.execution_route, fit.converged, fit.convergence_info)
print(fit.predict(newdata))
```

```text
dense True full convergence
[ 1.30141681 -0.76120126]
```

The same kind of model on the streamed route takes a row source and returns
a prediction result. The smooth uses `bs="cr"` because the streamed route
accepts only cubic regression splines, and `linear_solver="qr"` because
estimating the smoothing parameters there requires it:

```python
source = DataFrameRowSource(data, response="y")
control = FitControl(execution="stream", linear_solver="qr")
fit = GAM("y ~ s(x, bs='cr') + z", control=control).fit(source, result="prediction")
print(type(fit).__name__, fit.execution_route, fit.converged)
print(fit.predict(newdata))
```

```text
GAMPredictionResult stream_exact_reml_qr True
[ 1.31145334 -0.75912431]
```

The two predictions differ slightly because the bases differ (thin plate
against cubic regression spline), not because of the route.

## Which route ran

`execution_route` on every result names the route that actually ran:

| Value | Meaning |
|---|---|
| `"dense"` | Default fit, including Fellner-Schall and fixed `sp` |
| `"gaussian_qr"` | Dense fit with Gaussian QR compression |
| `"stream"`, `"stream_qr"` | Streamed fit with fixed `sp` (Cholesky or QR solver) |
| `"stream_exact_reml_qr"` | Streamed exact REML |
| `"stream_efs_qr"` | Streamed Fellner-Schall |
| `"stream_parametric_qr"` | Streamed Fellner-Schall request on a model with no penalties |
| `"stream_gaussian_qr"` | Streamed fit with Gaussian QR compression |
| `"discrete"` | Discretized fit |
| `"operator"` | Operator fit |

When you asked for Gaussian QR compression and the model did not qualify,
`execution_fallback_reason` says why. `lambda_strategy` names how the
smoothing parameters were chosen (`"newton_reml"`, `"fixed"`, `"efs_reml"`,
`"exact_reml"`, `"lbfgs_reml"`, `"working_model_reml"` and so on).

## Related pages

- [Smoothing parameter selection](smoothing_selection.md)
- [Row sources](row_sources.md)
- [Streamed fitting](streamed.md), [Discretized fitting](discretized.md),
  [Operator fitting](operator.md)
- [Prediction results](prediction_results.md)
- [Memory and performance](memory_and_performance.md)
- API reference: [GAM](../api.md#gam), [FitControl](../api.md#fit-control),
  [Fitting routes](../api.md#fitting-routes)
