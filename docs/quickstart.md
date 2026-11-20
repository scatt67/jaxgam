# Quickstart

This tutorial walks through jaxgam's core features on data that fit in
memory: fitting GAMs with different families, using several smooth types,
and post-estimation (prediction, summary, plotting).

All examples assume:

```python
import numpy as np
import pandas as pd
from jaxgam import GAM, GAMResults
```

## Choose a result type

Every fit returns one of three result types, chosen by the `result`
argument of `fit()`:

| Fit call | Return type | Use it for |
|---|---|---|
| `model.fit(data, result="full")` | `GAMResults` | Prediction, `summary()`, `plot()`, and in-sample values |
| `model.fit(data, result="inference")` | `GAMInferenceResult` | New-data prediction with standard errors and fit diagnostics, without the training arrays |
| `model.fit(data, result="prediction")` | `GAMPredictionResult` | New-data prediction only, with optional memory budgets; the only result of the routes for data that do not fit in memory |

`result="full"` is the default, so `model.fit(data)` means exactly the same
thing. Sections 1-10 and 12-14 use the full result. Section 11 introduces
the inference result. Every result type predicts directly; `to_predictor()`
is never required after fitting.

## 1. Gaussian GAM

The simplest case: a continuous response with a smooth effect.

```python
rng = np.random.default_rng(42)
n = 200
x = rng.uniform(0, 1, n)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
data = pd.DataFrame({"x": x, "y": y})

results = GAM("y ~ s(x, k=10, bs='cr')").fit(data, result="full")
print(f"Converged: {results.converged}")
print(f"EDF: {results.edf}")
print(f"Scale: {results.scale:.4f}")
```

The formula `"y ~ s(x, k=10, bs='cr')"` specifies:

- `y` as the response variable
- `s(x, ...)` as a smooth term over `x`
- `k=10` sets the basis dimension (for a cubic regression spline, the
  number of knots)
- `bs='cr'` selects a cubic regression spline

If `bs` is omitted, a thin-plate regression spline (`tp`) is used.

Here `result="full"` is written explicitly to show the mode. Later examples
use the shorter equivalent `.fit(data)`.

Always check `results.converged` before using a fit. When it is `False`,
`results.convergence_info` says how the fit stopped.

## 2. Binomial GAM

Binary response (0/1) with a logit link.

```python
rng = np.random.default_rng(42)
n = 300
x = rng.uniform(0, 1, n)
eta = 2 * np.sin(2 * np.pi * x)
prob = 1 / (1 + np.exp(-eta))
y = rng.binomial(1, prob, n).astype(float)
data = pd.DataFrame({"x": x, "y": y})

results = GAM("y ~ s(x, k=10, bs='cr')", family="binomial").fit(data)
```

## 3. Poisson GAM

Count data with a log link.

```python
rng = np.random.default_rng(42)
n = 200
x = rng.uniform(0, 1, n)
eta = np.sin(2 * np.pi * x) + 0.5
y = rng.poisson(np.exp(eta)).astype(float)
data = pd.DataFrame({"x": x, "y": y})

results = GAM("y ~ s(x, k=10, bs='cr')", family="poisson").fit(data)
```

## 4. Gamma GAM

Positive continuous response. A family name always uses that family's
default link (inverse for Gamma). The data below are generated with
`mu = exp(eta)`, so we pass a family instance, `Gamma(link="log")`, to use
the log link instead.

```python
from jaxgam.families import Gamma

rng = np.random.default_rng(42)
n = 200
x = rng.uniform(0, 1, n)
eta = 0.5 * np.sin(2 * np.pi * x) + 1.0
mu = np.exp(eta)
y = rng.gamma(5.0, scale=mu / 5.0, size=n)
data = pd.DataFrame({"x": x, "y": y})

results = GAM("y ~ s(x, k=10, bs='cr')", family=Gamma(link="log")).fit(data)
```

## 5. Negative binomial GAM

Count data with overdispersion (variance larger than the mean). The
dispersion parameter theta is estimated along with the smoothing
parameters.

```python
rng = np.random.default_rng(42)
n = 300
x = rng.uniform(0, 1, n)
eta = np.sin(2 * np.pi * x) + 0.5
mu = np.exp(eta)
theta = 2.0
y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
data = pd.DataFrame({"x": x, "y": y})

results = GAM("y ~ s(x, k=10, bs='cr')", family="nb").fit(data)
print(f"Estimated theta: {results.theta:.2f}")
```

To fix theta instead of estimating it:

```python
from jaxgam.families import NegativeBinomial

results = GAM("y ~ s(x)", family=NegativeBinomial(theta=2, fixed=True)).fit(data)
```

With a fixed theta, `results.theta` is `None`; the fitted family still
holds the value, `results.family.get_theta(transformed=True)`.

## 6. Multiple smooths

Add several smooth terms with `+`.

```python
rng = np.random.default_rng(42)
n = 200
x1 = rng.uniform(0, 1, n)
x2 = rng.uniform(0, 1, n)
y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

results = GAM("y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr')").fit(data)
print(f"Per-smooth EDF: {results.edf}")
```

## 7. Tensor product smooths

Model interactions between covariates with `te()`.

```python
results = GAM("y ~ te(x1, x2, k=5)").fit(data)
```

The scalar `k=5` gives each margin 5 basis functions, a 5 x 5 = 25 basis
function tensor product. Each margin uses a cubic regression spline by
default.

Use `ti()` for tensor interaction terms (without main effects):

```python
results = GAM("y ~ s(x1, k=8) + s(x2, k=8) + ti(x1, x2, k=5)").fit(data)
```

## 8. Factor-by smooths

Fit a separate smooth curve for each level of a factor variable.

```python
rng = np.random.default_rng(42)
n = 300
x = rng.uniform(0, 1, n)
levels = ["a", "b", "c"]
fac = rng.choice(levels, n)

# Different functions per level
eta = np.where(
    fac == "a",
    np.sin(2 * np.pi * x),
    np.where(fac == "b", 0.5 * x, -0.3 * x),
)
y = eta + rng.normal(0, 0.3, n)

data = pd.DataFrame({
    "x": x,
    "fac": pd.Categorical(fac, categories=levels),
    "y": y,
})

results = GAM("y ~ s(x, by=fac, k=10, bs='cr') + fac").fit(data)
```

The `+ fac` adds a parametric intercept shift per level, as in R. The
`by=fac` argument creates a separate smooth per factor level, each with its
own smoothing parameter.

**Important:** the factor column must be a `pd.Categorical` or a string
column so jaxgam recognizes it as a factor. An integer-coded column is
treated as numeric.

## 9. Random effects

Use `bs='re'` for simple independent random effects, represented as
penalized smooths in the model matrix. This suits a modest number of
levels; for random effects with very many levels, see the operator route in
the [user guide](guide/operator.md).

```python
rng = np.random.default_rng(42)
n = 300
groups = [f"g{i}" for i in range(20)]
g = rng.choice(groups, size=n)
x = rng.uniform(0, 1, n)
group_effect = rng.normal(0, 0.7, len(groups))
effect_lookup = dict(zip(groups, group_effect, strict=True))
y = (
    np.sin(2 * np.pi * x)
    + np.array([effect_lookup[level] for level in g])
    + rng.normal(0, 0.25, n)
)

data = pd.DataFrame({
    "x": x,
    "g": pd.Categorical(g),
    "y": y,
})

results = GAM("y ~ s(x, k=10, bs='cr') + s(g, bs='re')").fit(data)
```

`s(g, bs='re')` adds one penalized coefficient per group level. A
random-slope term without a random intercept is:

```python
results = GAM("y ~ s(x, g, bs='re')").fit(data)
```

Use both terms when you want random intercepts and random slopes:

```python
results = GAM("y ~ s(g, bs='re') + s(x, g, bs='re')").fit(data)
```

As with `by=`, the grouping column must be a categorical or string column.

When you predict for a group level that was not in the training data, or a
missing group, the random-effect contribution is zero (the prior mean of
the random effect), so the prediction is finite.

## 10. Prediction

The examples below use the random-intercept model from section 9.

```python
results = GAM("y ~ s(x, k=10, bs='cr') + s(g, bs='re')").fit(data)
```

### Self-prediction

```python
# Predictions on the training data
mu_hat = results.predict()                          # response scale
eta_hat = results.predict(pred_type="link")         # link scale
mu_hat, se = results.predict(se_fit=True)           # with standard errors
```

### New data

New data must hold every variable the formula uses.

```python
newdata = pd.DataFrame({
    "x": np.linspace(0, 1, 100),
    "g": pd.Categorical(["g0"] * 100, categories=groups),
})
predictions = results.predict(newdata)
predictions, se = results.predict(newdata, se_fit=True)
```

### Prediction matrix

For manual inference, get the constrained prediction matrix (mgcv's
`predict.gam(type = "lpmatrix")`):

```python
X_new = results.predict_matrix(newdata)
eta_new = X_new @ results.coefficients  # equals predict(newdata, pred_type="link")
```

## 11. Lean inference result

When you keep a fitted object for new-data prediction rather than
interactive analysis, request the lean result mode:

```python
lean = GAM("y ~ s(x, k=10, bs='cr') + s(g, bs='re')").fit(data, result="inference")
predictions = lean.predict(newdata)
```

That is all you need for inference. The returned `GAMInferenceResult` is
already lean and directly supports `predict()` and `predict_matrix()`.
**Do not call `to_predictor()` as a routine second step:** it does not make
an inference result smaller.

The lean result keeps diagnostics such as `edf`, `deviance` and
`converged`, plus `formula`, `smooth_info` and `term_names`. It has no
`summary()` or `plot()`, and prediction requires `newdata`. This mode
reduces memory kept after the fit; it does not reduce peak memory during
the fit.

A third mode, `result="prediction"`, keeps even less (no per-smooth EDF,
and no standard errors unless you ask for them). It is described in the
[API reference](api.md#gampredictionresult).

### Optional prediction-only handoff

Use `to_predictor()` only when another component should receive the
narrower prediction-only interface, without the inference result's
diagnostics:

```python
import pickle

predictor = lean.to_predictor()  # returns lean's existing core; no copy
blob = pickle.dumps(predictor)
restored = pickle.loads(blob)
same_predictions = restored.predict(newdata)
```

If the receiving process needs diagnostics too, pickle `lean` directly. For
a full `GAMResults`, `to_predictor()` instead builds an independent lean
core; discard the full result afterward if you want its training-backed
memory to be released.

Pickles are for trusted, same-version, short-lived handoff. Loading a
predictor under another jaxgam version warns and is not guaranteed to work.
Locally defined custom links or families require `cloudpickle`; built-in
and module-level definitions work with the standard library `pickle`.

## 12. Summary

`summary()` prints and returns a summary object with parametric
coefficient tests, smooth term significance tests (Wood 2013), and
model-level statistics.

```python
s = results.summary()
```

Output includes:

- Parametric coefficients with z or t tests
- Smooth terms with estimated degrees of freedom, F or chi-squared
  statistics, and p-values
- Adjusted R-squared, deviance explained, scale estimate and REML score

## 13. Plotting

`plot()` produces one panel per smooth term and returns the figure and its
axes.

```python
fig, axes = results.plot()
```

### Options

```python
# Select specific smooth terms (0-indexed)
fig, axes = results.plot(select=[0])

# Customize appearance
fig, axes = results.plot(
    rug=True,       # show data rug marks (default: True)
    se=True,        # show SE bands (default: True)
    shade=True,     # shaded bands vs dashed lines (default: True)
)
```

Panels show partial effects on the link scale. A one-variable smooth is
drawn as a line with a shaded confidence band. A two-variable smooth or
tensor product is drawn as a filled contour plot. A factor-by smooth gets
one panel per level, and a `bs='re'` term a normal quantile plot of its
coefficients.

## 14. Fitting options

### Smoothing parameter method

```python
# REML (default, and the only supported method)
results = GAM("y ~ s(x)", method="REML").fit(data)

# Maximum likelihood (ML) is not available: method="ML" raises
# NotImplementedError.
```

### Fixed smoothing parameters

Hold the smoothing parameters at values you supply instead of estimating
them:

```python
results = GAM("y ~ s(x, k=10, bs='cr')", sp=[1.0]).fit(data)
```

The `sp` list must have one positive entry per penalty in the model.
`results.smoothing_params` reports the values you supplied. For Gaussian,
Gamma and estimated-theta negative binomial models, the scale or theta is
still estimated.

### Weights and offset

```python
weights = np.ones(len(data))
offset = np.zeros(len(data))
results = GAM("y ~ s(x)").fit(data, weights=weights, offset=offset)
```

When a model was fitted with a nonzero offset, pass `offset=` to
`predict()` for new data; without it, jaxgam warns and omits the offset.

### Fellner-Schall optimizer

`optimizer="efs"` selects smoothing parameters with mgcv's extended
Fellner-Schall update instead of Newton's method:

```python
results = GAM("y ~ s(x, k=10, bs='cr') + s(g, bs='re')", optimizer="efs").fit(data)
print(results.lambda_strategy, results.convergence_info)
```

See [Smoothing parameter selection](guide/smoothing_selection.md) for when
to use it and which families it supports.

## 15. GAMResults attributes

By default, `fit()` returns a `GAMResults` frozen dataclass. Its fields
cannot be reassigned, although the arrays inside are not themselves
frozen. The most used attributes:

| Attribute | Description |
|---|---|
| `coefficients` | Coefficient vector (p,) |
| `fitted_values` | Fitted values on the response scale (n,) |
| `linear_predictor` | Linear predictor eta (n,) |
| `Vp` | Bayesian covariance matrix (p, p) |
| `edf` | Per-smooth effective degrees of freedom |
| `edf1` | Alternative EDF used for significance tests |
| `edf_total` | Total effective degrees of freedom |
| `scale` | Scale (dispersion) parameter |
| `deviance` | Model deviance |
| `null_deviance` | Null model deviance |
| `smoothing_params` | Smoothing parameters, estimated or as supplied |
| `converged` | Whether the fit converged |
| `convergence_info` | How the fit stopped |
| `n_iter` | Number of outer iterations |
| `score` | REML criterion at the returned smoothing parameters |
| `X` | Model matrix (n, p) |
| `y` | Response vector (n,) |
| `weights` | Prior weights (n,) |
| `family` | Family object used for fitting |
| `formula` | Model formula string |
| `theta` | Estimated theta for a negative binomial with estimated theta; `None` otherwise |
| `method` | Smoothing parameter method (`"REML"`) |
| `n` | Number of observations |

The [API reference](api.md#gamresults-attributes) lists every attribute.

## Next steps

- For data that do not fit in memory, for the Fellner-Schall optimizer and
  Gaussian QR compression, and for the streamed, discretized and operator
  routes, start with [Choosing a route](guide/choosing_a_route.md) in the
  user guide.
- The [API reference](api.md) lists every class, argument and result
  attribute.
