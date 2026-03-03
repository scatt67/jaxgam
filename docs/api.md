# API Reference

## GAM

The main entry point for fitting generalized additive models.

```python
from jaxgam import GAM
```

::: jaxgam.api.GAM
    options:
      members:
        - __init__
        - fit
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

## Fitted model attributes

After calling `fit()`, these attributes are available on the `GAM` instance:

| Attribute | Type | Description |
|---|---|---|
| `coefficients_` | `ndarray (p,)` | Coefficient vector |
| `fitted_values_` | `ndarray (n,)` | Fitted values on response scale |
| `linear_predictor_` | `ndarray (n,)` | Linear predictor |
| `Vp_` | `ndarray (p, p)` | Bayesian covariance matrix |
| `edf_` | `ndarray` | Per-smooth effective degrees of freedom |
| `edf_total_` | `float` | Total effective degrees of freedom |
| `scale_` | `float` | Estimated scale (dispersion) parameter |
| `deviance_` | `float` | Model deviance |
| `null_deviance_` | `float` | Null model deviance |
| `smoothing_params_` | `ndarray` | Estimated smoothing parameters |
| `converged_` | `bool` | Whether the optimizer converged |
| `n_iter_` | `int` | Number of Newton iterations |
| `X_` | `ndarray (n, p)` | Design matrix |
| `family_` | `ExponentialFamily` | Family object used for fitting |
