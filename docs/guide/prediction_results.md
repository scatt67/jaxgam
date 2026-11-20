# Prediction results

`GAMPredictionResult` is the small result that `fit(..., result="prediction")`
returns. It keeps the coefficients, a few scalar diagnostics and, only if you
ask for it, what is needed for standard errors. It keeps no training rows, so
its size does not grow with the data.

**Use it when** you fit on a row-source route (it is the only result those
routes return), or when you fit a DataFrame and only need to predict on new
data.

**Do not use it when** you need `summary()`, `plot()`, per-smooth EDF or
in-sample fitted values: fit the DataFrame with `result="full"` instead.

All examples on this page use this setup block. It fits on the dense route
and keeps what standard errors need:

```python
import pickle

import numpy as np
import pandas as pd
from jaxgam import GAM, FitControl
from jaxgam.data import DataFrameRowSource

rng = np.random.default_rng(1)
n = 400
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
y = np.sin(2 * np.pi * x) + 0.5 * z + rng.normal(0, 0.3, n)
data = pd.DataFrame({"x": x, "z": z, "y": y})
newdata = pd.DataFrame({"x": [0.1, 0.4, 0.7], "z": [0.5, 0.5, 0.5]})

control = FitControl(uncertainty="fisher")
fit = GAM("y ~ s(x) + z", control=control).fit(data, result="prediction")
```

## `predict()`

```python
print(fit.predict(newdata))
prediction, se = fit.predict(newdata, se_fit=True)
print(se)
```

```text
[ 0.85121636  0.84844347 -0.78349386]
[0.04486273 0.04101813 0.04336924]
```

| Option | Default | What it does |
|---|---|---|
| `newdata` | required | DataFrame or dict holding every variable the formula uses. |
| `pred_type` | `"response"` | `"response"` for the mean, `"link"` for the linear predictor. Anything else raises `ValueError`. |
| `se_fit` | `False` | Also return standard errors, as `(prediction, se)`. On the response scale the standard error is the link-scale one times the absolute derivative of the inverse link. |
| `offset` | `None` | One finite value per row of `newdata`. A model fitted with a nonzero offset warns when you leave it out, and predicts without it. |

`predict_matrix(newdata)` returns the matrix that maps the coefficients to
the linear predictor (mgcv's `type="lpmatrix"`).

## `predict_iter()`: new data from a row source

When the new data are themselves too large for one call, pass a row source.
`predict_iter(source, batch_rows=None, *, pred_type="response",
se_fit=False)` yields one `(row_positions, prediction)` pair per batch, or
`(row_positions, (prediction, se))` with `se_fit=True`. The source needs no
response. Offsets come from the source.

```python
new_source = DataFrameRowSource(newdata)
for positions, (batch_prediction, batch_se) in fit.predict_iter(
    new_source, batch_rows=2, se_fit=True
):
    print(positions, batch_prediction.round(4), batch_se.round(4))
```

```text
[0 1] [0.8512 0.8484] [0.0449 0.041 ]
[2] [-0.7835] [0.0434]
```

`batch_rows` defaults to the `FitControl.batch_rows` of the fit. Each batch
is one prediction call, so each must fit the output and memory budgets (see
[Memory and performance](memory_and_performance.md)). Writing your own
source is covered in [Row sources](row_sources.md).

## Standard errors and `FitControl(uncertainty=...)`

A prediction result keeps nothing for standard errors unless you ask for it
when you fit:

| `uncertainty` | Keeps | `predict(se_fit=True)` |
|---|---|---|
| `"none"` (default) | Nothing | Raises `RuntimeError` |
| `"fisher"` | A factor of the penalized Fisher information (size grows with the number of coefficients squared, not with rows) | Exact standard errors for the requested rows |
| `"covariance"` | The coefficient covariance matrix `Vp`, built at the end of the fit (the streamed QR routes also keep their factor) | Standard errors |

The standard errors match those of a full result of the same fit:

```python
full = GAM("y ~ s(x) + z").fit(data)
_, full_se = full.predict(newdata, se_fit=True)
print(np.allclose(se, full_se))
```

```text
True
```

Without uncertainty, `se_fit=True` raises:

```text
RuntimeError: Prediction uncertainty is unavailable for this prediction-only result. Fit with FitControl(uncertainty='fisher' or 'covariance').
```

With `"fisher"` you can still build `Vp` later, under a budget you give:

```python
Vp = fit.materialize_covariance(memory_budget_bytes=10_000_000)
print(Vp.shape)
```

```text
(11, 11)
```

The dense, streamed and discretized routes support all three values. The
operator route accepts only `"none"`; its approximate variances come from
`bulk_link_variance()`, described in [Operator fitting](operator.md), and
`se_fit=True` on an operator result raises an error that points there.
`"covariance"` raises `MemoryError` when `Vp` would exceed
`FitControl.memory_budget_bytes`. A prediction result has no `Vp`
attribute; with `"covariance"` read it as `fit.to_predictor().Vp`.

## What you get back

`GAMPredictionResult` attributes worth reading:

| Attribute | Meaning |
|---|---|
| `converged`, `n_iter`, `convergence_info` | Whether the fit converged, its outer iteration count, and the stop reason in words. Always check `converged`. |
| `smoothing_params` | Selected or supplied smoothing parameters. |
| `edf_total` | Total effective degrees of freedom (`None` on fixed-`sp` operator fits). |
| `score`, `deviance`, `scale` | REML score, deviance and scale (`score` is `None` on fixed-`sp` operator fits). |
| `theta` | Negative binomial theta. `None` for other families. |
| `coefficients`, `family`, `formula`, `n` | The fitted coefficients, family, formula and number of rows. |
| `execution_route`, `execution_fallback_reason` | Which route ran, and why Gaussian QR compression was not used, if you asked for it. |
| `lambda_strategy`, `method` | How the smoothing parameters were chosen, and `"REML"`. |
| `source_scans`, `batches_scanned` | Full scans and batches the fit read from a row source, excluding the setup pass (`None` on the dense route). |
| `known_workspace_bytes` | The route's estimate of its fit workspace, where it has one. |
| `optimizer_diagnostics` | `EFSOptimizerDiagnostics` for Fellner-Schall fits. |

```python
print(fit.converged, fit.convergence_info, fit.execution_route)
print(fit.edf_total, fit.scale, fit.theta, fit.source_scans)
```

```text
True full convergence dense
9.799818771624803 0.09034243612941105 None None
```

Route-specific fields (for example `prediction_policy` and `discretization`
on the discretized route, `operator_diagnostics` and `score_error_bound` on
the operator route) are described on each route's page and in
[GAMPredictionResult](../api.md#gampredictionresult) in the API reference.

For a negative binomial with fixed theta, `theta` is `None` from a Newton
fit on the dense or streamed route, and the fixed value from a
Fellner-Schall, discretized or operator fit.

## What is not available

A prediction result has no `summary()`, `plot()`, `fitted_values`,
`linear_predictor`, `Vp`, per-smooth `edf` or `null_deviance`, and
`predict()` always needs `newdata`. To get the in-sample fit, predict on the
training data (on a row-source route, with `predict_iter()` on the training
source).

## `to_predictor()` and pickling

`to_predictor()` returns the `GAMPredictor` the result holds, an object
with `predict()`, `predict_matrix()` and `predict_iter()` and nothing else.
Hand it to code that should only predict.

```python
predictor = fit.to_predictor()
restored = pickle.loads(pickle.dumps(predictor))
print(np.allclose(restored.predict(newdata), fit.predict(newdata)))

restored_fit = pickle.loads(pickle.dumps(fit))
print(np.allclose(restored_fit.predict(newdata), fit.predict(newdata)))
```

```text
True
True
```

Both the result and the predictor pickle with the standard library. Treat a
pickle as a short-lived handoff between processes running the same jaxgam
version: loading a predictor pickled by another version issues a
`UserWarning` that predictions may be wrong or fail. A family whose link
class is defined inside a function cannot be pickled with `pickle`; the
separate `cloudpickle` package can do it.

## How it compares with the other result modes

| | `"full"` (`GAMResults`) | `"inference"` (`GAMInferenceResult`) | `"prediction"` (`GAMPredictionResult`) |
|---|---|---|---|
| Routes | Dense | Dense | All |
| Training rows, fitted values | Yes | No | No |
| `summary()`, `plot()` | Yes | No | No |
| `Vp`, per-smooth EDF | Yes | Yes | No (covariance only on request) |
| Standard errors | Always | Always | With `uncertainty="fisher"` or `"covariance"` |
| `predict_iter()` | Through `to_predictor()` | Through `to_predictor()` | Yes |
| Prediction budgets | Not applied | Not applied | `output_budget_bytes`, `memory_budget_bytes` |

## Limits

- `predict_iter()` needs a row source:
  `TypeError: predict_iter requires a replayable RowSource.`
- A prediction whose output exceeds `FitControl.output_budget_bytes` raises,
  for example
  `MemoryError: Prediction output requires 4000 bytes, exceeding output budget 1000 bytes. Use predict_iter() with smaller batches.`
- A level of a parametric factor not seen in fitting raises
  `ValueError: Parametric factor 'f' has new level(s) ['w'] not seen during fitting. Predictions for unseen levels of a parametric factor are undefined.`
  An unseen or missing level of a random effect `s(g, bs="re")` predicts with
  a random-effect contribution of zero (mgcv returns `NA` for a missing one).

## Related pages

- [Choosing a route](choosing_a_route.md)
- [Row sources](row_sources.md)
- [Memory and performance](memory_and_performance.md)
- [Operator fitting](operator.md) for `bulk_link_variance()`
- API reference: [GAMPredictionResult](../api.md#gampredictionresult),
  [Fit result modes](../api.md#fit-result-modes),
  [FitControl](../api.md#fit-control)
