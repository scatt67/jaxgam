# API Reference

## GAM

Model specification and fit orchestration. `fit()` returns a `GAMResults`
frozen dataclass containing all fitted state.

```python
from jaxgam import GAM, GAMResults

model = GAM("y ~ s(x)", family="gaussian")
results = model.fit(data)
results.predict(newdata)
results.summary()
```

::: jaxgam.api.GAM
    options:
      members:
        - __init__
        - fit

---

## GAMResults

Immutable results object returned by `GAM.fit()`. All post-estimation
methods (prediction, summary, plotting) live here.

```python
from jaxgam import GAMResults
```

::: jaxgam.results.GAMResults
    options:
      members:
        - predict
        - predict_matrix
        - summary
        - plot

---

## Families

Distribution families for the response variable. Normally specified as a
string (`family="gaussian"`) when constructing a `GAM`, but the classes
can be used directly for custom link functions.

```python
from jaxgam.families import Gaussian, Binomial, Poisson, Gamma
```

### Gaussian

::: jaxgam.families.standard.Gaussian
    options:
      members:
        - __init__
        - variance
        - deviance_resids
        - initialize

### Binomial

::: jaxgam.families.standard.Binomial
    options:
      members:
        - __init__
        - variance
        - deviance_resids
        - initialize

### Poisson

::: jaxgam.families.standard.Poisson
    options:
      members:
        - __init__
        - variance
        - deviance_resids
        - initialize

### Gamma

::: jaxgam.families.standard.Gamma
    options:
      members:
        - __init__
        - variance
        - deviance_resids
        - initialize

---

## Formula syntax

Models are specified with R-style formulas:

```python
# Single smooth
GAM("y ~ s(x)")

# Multiple smooths
GAM("y ~ s(x1) + s(x2)")

# Tensor product
GAM("y ~ te(x1, x2, k=5)")

# Factor-by smooth
GAM("y ~ s(x, by=fac, k=10) + fac")
```

### Smooth term arguments

| Argument | Description | Default |
|---|---|---|
| `k` | Basis dimension (number of knots). `-1` means auto-select: resolves to `10` for 1D TPRS/cubic, `30` for 2D TPRS. | -1 (auto) |
| `bs` | Basis type: `'tp'`, `'ts'`, `'cr'`, `'cs'`, `'cc'` | `'tp'` |
| `by` | Factor variable for factor-by smooths | None |

### Tensor product arguments

| Argument | Description | Default |
|---|---|---|
| `k` | Marginal basis dimension (scalar applied to all margins). `-1` means auto-select (resolves to `10` for the default `cr` marginals). | -1 (auto) |

Use `te()` for full tensor products and `ti()` for interaction-only terms
(excludes main effects).

---

## Custom registrations

JaxGAM ships with built-in smooths, families, and links, but you can
register your own at runtime. Custom entries extend the registry without
modifying or removing built-in entries.

### Registering a custom smooth

Your class must subclass `jaxgam.smooths.Smooth`.

```python
from jaxgam.smooths import smooth_registry, Smooth

class PSplineSmooth(Smooth):
    ...  # implement setup(), basis(), penalty(), etc.

smooth_registry.register("ps", PSplineSmooth)

# Now usable in formulas:
GAM("y ~ s(x, bs='ps')").fit(data)
```

### Registering a custom family

Your class must subclass `jaxgam.families.ExponentialFamily`.

```python
from jaxgam.families import family_registry, ExponentialFamily

class NegativeBinomial(ExponentialFamily):
    ...  # implement variance(), deviance_resids(), initialize(), etc.

family_registry.register("nb", NegativeBinomial)

# Now usable by name:
GAM("y ~ s(x)", family="nb").fit(data)
```

### Registering a custom link

Your class must subclass `jaxgam.links.Link`.

```python
from jaxgam.links import link_registry, Link

class CauchitLink(Link):
    ...  # implement link(), inverse(), derivative()

link_registry.register("cauchit", CauchitLink)
```

### Rules

- Keys are **case-insensitive** — `"PS"` and `"ps"` are the same key.
- You **cannot override** a built-in or previously registered key. Attempting
  to do so raises `ValueError`.
- Registrations are global and take effect immediately — any subsequent
  `GAM` call will see the new entry.

### Inspecting a registry

```python
from jaxgam.smooths import smooth_registry

smooth_registry.available      # ('cc', 'cr', 'cs', 'te', 'ti', 'tp', 'ts')
"tp" in smooth_registry        # True
len(smooth_registry)           # 7
```

---

## GAMResults attributes

The `GAMResults` object returned by `fit()` exposes all fitted state as
read-only attributes (frozen dataclass):

| Attribute | Type | Description |
|---|---|---|
| `coefficients` | `ndarray (p,)` | Coefficient vector |
| `fitted_values` | `ndarray (n,)` | Fitted values on response scale |
| `linear_predictor` | `ndarray (n,)` | Linear predictor |
| `Vp` | `ndarray (p, p)` | Bayesian covariance matrix |
| `edf` | `ndarray` | Per-smooth effective degrees of freedom |
| `edf1` | `ndarray` | Alternative EDF for significance testing |
| `edf_total` | `float` | Total effective degrees of freedom |
| `scale` | `float` | Estimated scale (dispersion) parameter |
| `deviance` | `float` | Model deviance |
| `null_deviance` | `float` | Null model deviance |
| `smoothing_params` | `ndarray` | Estimated smoothing parameters |
| `converged` | `bool` | Whether the optimizer converged |
| `n_iter` | `int` | Number of Newton iterations |
| `score` | `float` | REML/ML value at convergence |
| `X` | `ndarray (n, p)` | Design matrix |
| `y` | `ndarray (n,)` | Response vector |
| `weights` | `ndarray (n,)` | Prior weights |
| `family` | `ExponentialFamily` | Family object used for fitting |
| `formula` | `str` | Model formula |
| `method` | `str` | Smoothing parameter method ("REML" or "ML") |
| `n` | `int` | Number of observations |

