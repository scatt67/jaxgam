# Quickstart

This tutorial walks through jaxgam's core features: fitting GAMs with
different families, using multiple smooth types, and post-estimation
(prediction, summary, plotting).

All examples assume:

```python
import numpy as np
import pandas as pd
from jaxgam import GAM, GAMResults
```

## 1. Gaussian GAM

The simplest case: a continuous response with a smooth effect.

```python
rng = np.random.default_rng(42)
n = 200
x = rng.uniform(0, 1, n)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
data = pd.DataFrame({"x": x, "y": y})

results = GAM("y ~ s(x, k=10, bs='cr')").fit(data)
print(f"Converged: {results.converged}")
print(f"EDF: {results.edf}")
print(f"Scale: {results.scale:.4f}")
```

The formula `"y ~ s(x, k=10, bs='cr')"` specifies:
- `y` as the response variable
- `s(x, ...)` as a smooth term over `x`
- `k=10` sets the basis dimension (number of knots)
- `bs='cr'` selects cubic regression splines

If `bs` is omitted, thin-plate regression splines (`tp`) are used by
default.

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

Positive continuous response with a log link. The data-generating process
uses `mu = exp(eta)`, so we pass `Gamma(link="log")` to match.

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

## 5. Multiple smooths

Add multiple smooth terms with `+`.

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

## 6. Tensor product smooths

Model interactions between covariates with `te()`.

```python
results = GAM("y ~ te(x1, x2, k=5)").fit(data)
```

The scalar `k=5` creates a 5x5 = 25 basis function tensor product.
Each marginal uses cubic regression splines by default.

Use `ti()` for tensor interaction terms (without main effects):

```python
results = GAM("y ~ s(x1, k=8) + s(x2, k=8) + ti(x1, x2, k=5)").fit(data)
```

## 7. Factor-by smooths

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

The `+ fac` adds a parametric intercept shift per level, analogous to R.
The `by=fac` argument creates a separate smooth per factor level, each
with its own smoothing parameter.

**Important:** The factor column must be `pd.Categorical` (or string
dtype) so jaxgam recognizes it as a factor.

## 8. Prediction

### Self-prediction

```python
# Predictions on the training data
mu_hat = results.predict()                          # response scale
eta_hat = results.predict(pred_type="link")          # link scale
mu_hat, se = results.predict(se_fit=True)           # with standard errors
```

### New data

```python
newdata = pd.DataFrame({"x": np.linspace(0, 1, 100)})
predictions = results.predict(newdata)
predictions, se = results.predict(newdata, se_fit=True)
```

### Prediction matrix

For manual inference, get the constrained design matrix:

```python
X_new = results.predict_matrix(newdata)
# Manual prediction: eta = X_new @ results.coefficients
```

## 9. Summary

`summary()` prints and returns a summary object with parametric
coefficient tests, smooth term significance tests (Wood 2013), and
model-level statistics.

```python
s = results.summary()
```

Output includes:
- Parametric coefficients with z/t-tests
- Smooth terms with estimated degrees of freedom, F/chi-squared
  statistics, and p-values
- R-squared, deviance explained, scale estimate

## 10. Plotting

`plot()` produces one panel per smooth term.

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

For 1D smooths, `plot()` shows the partial effect with shaded
confidence bands. For 2D tensor products, it shows filled contour
plots. Factor-by smooths produce one panel per level.

## 11. Fitting options

### Smoothing parameter method

```python
# REML (default, recommended)
results = GAM("y ~ s(x)", method="REML").fit(data)

# Maximum likelihood
results = GAM("y ~ s(x)", method="ML").fit(data)
```

### Fixed smoothing parameters

Skip the Newton optimization and fit at user-supplied smoothing
parameters:

```python
results = GAM("y ~ s(x, k=10, bs='cr')", sp=[1.0]).fit(data)
```

The `sp` list must have one entry per penalty term in the model.

### Weights and offset

```python
weights = np.ones(n)
offset = np.zeros(n)
results = GAM("y ~ s(x)").fit(data, weights=weights, offset=offset)
```

## 12. GAMResults attributes

`fit()` returns a `GAMResults` frozen dataclass. All attributes are
read-only:

| Attribute | Description |
|---|---|
| `coefficients` | Coefficient vector (p,) |
| `fitted_values` | Fitted values on response scale (n,) |
| `linear_predictor` | Linear predictor eta (n,) |
| `Vp` | Bayesian covariance matrix (p, p) |
| `edf` | Per-smooth effective degrees of freedom |
| `edf1` | Alternative EDF for significance testing |
| `edf_total` | Total effective degrees of freedom |
| `scale` | Estimated scale (dispersion) parameter |
| `deviance` | Model deviance |
| `null_deviance` | Null model deviance |
| `smoothing_params` | Estimated smoothing parameters |
| `converged` | Whether the optimizer converged |
| `n_iter` | Number of Newton iterations |
| `score` | REML/ML value at convergence |
| `X` | Design matrix (n, p) |
| `y` | Response vector (n,) |
| `weights` | Prior weights (n,) |
| `family` | Family object used for fitting |
| `formula` | Model formula string |
| `method` | Smoothing parameter method ("REML" or "ML") |
| `n` | Number of observations |

!!! warning "Deprecated: accessing fitted attributes on GAM"
    The old pattern `model.coefficients_` (with trailing underscore) on the
    `GAM` instance still works but emits `DeprecationWarning` and will be
    removed in v1.1. Use `results.coefficients` instead.

## Further reading

- [Design document](design.md) -- architecture, algorithms, and
  implementation decisions
- [R source map](R_SOURCE_MAP.md) -- correspondence between jaxgam
  modules and R mgcv source files
