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
from jaxgam.families import Gaussian, Binomial, Poisson, Gamma, NegativeBinomial
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

### Negative Binomial (Extended Family)

The Negative Binomial family models overdispersed count data. It is an
**extended family** with an extra dispersion parameter theta that can be
estimated alongside smoothing parameters, or fixed.

- **Variance:** `mu + mu^2 / theta`
- **As theta -> infinity:** NB approaches Poisson
- **Theta parameterization:** `theta > 0` (R's "size" parameter); `alpha = 1/theta`

```python
from jaxgam.families import NegativeBinomial

# Estimate theta (default, starting from 1)
GAM("y ~ s(x)", family="nb").fit(data)
GAM("y ~ s(x)", family=NegativeBinomial()).fit(data)

# Estimate theta with a different starting value
GAM("y ~ s(x)", family=NegativeBinomial(theta=3)).fit(data)

# Fix theta at a known value
GAM("y ~ s(x)", family=NegativeBinomial(theta=2, fixed=True)).fit(data)
```

Constructor parameters:
- `theta` (float, default 1.0): dispersion parameter (must be positive)
- `fixed` (bool, default False): if True, theta is held constant during fitting

::: jaxgam.families.negative_binomial.NegativeBinomial
    options:
      members:
        - __init__
        - variance
        - deviance_resids
        - initialize
        - get_theta
        - put_theta

### ExtendedFamily Base Class

Base class for families with extra distributional parameters estimated
via Newton optimization. `NegativeBinomial` inherits from this.
Future extended families (Tweedie, Beta, etc.) will also subclass it.

::: jaxgam.families.extended.ExtendedFamily

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

# Gaussian process smooth
GAM("y ~ s(x, z, bs='gp', k=30)")

# Factor-by smooth
GAM("y ~ s(x, by=fac, k=10) + fac")
```

### Smooth term arguments

| Argument | Description | Default |
|---|---|---|
| `k` | Basis dimension (number of knots). `-1` means auto-select: resolves to `10` for 1D TPRS/cubic, `30` for 2D TPRS. GP uses a different rule (see the Gaussian process section). | -1 (auto) |
| `bs` | Basis type: `'tp'`, `'ts'`, `'cr'`, `'cs'`, `'cc'`, `'gp'`, `'re'` | `'tp'` |
| `by` | Factor variable for factor-by smooths | None |

### Smooth catalog

| Formula syntax | Basis type |
|---|---|
| `s(x, bs='tp')` | Thin-plate regression spline (default) |
| `s(x, bs='cr')` | Cubic regression spline |
| `s(x, bs='cs')` | Cubic spline with shrinkage |
| `s(x, bs='cc')` | Cyclic cubic spline |
| `s(x, z, bs='gp')` | Low-rank Gaussian process smooth |
| `s(g, bs='re')` | Dense random effect smooth |

### Gaussian process smooths

`bs='gp'` constructs a low-rank kriging smooth using the Kammann-Wand
Gaussian process construction: a reduced-rank basis from the leading
eigenvectors of a correlation matrix evaluated on predictor knots. It
supports 1D, 2D, and 3D continuous predictors, and can also be used as a
tensor-product margin via `te()` and `ti()`.

With the default `k=-1`, the GP basis dimension auto-resolves to `12`
(1D), `33` (2D), or `104` (3D); 4D and higher predictors require an
explicit `k`.

Supported `kernel` values are case-insensitive canonical names, with no
aliases:

| Kernel name | Description |
|---|---|
| `"spherical"` | Compactly supported spherical correlation |
| `"power_exponential"` | Power-exponential correlation |
| `"matern_3_2"` | Matérn 3/2 correlation (default) |
| `"matern_5_2"` | Matérn 5/2 correlation |
| `"matern_7_2"` | Matérn 7/2 correlation |

GP smooths accept these Python-native keyword arguments inside `s()`:

| Argument | Description | Default |
|---|---|---|
| `kernel` | One of the five kernel names above | `"matern_3_2"` |
| `rho` | Positive range parameter; omit for the Kammann-Wand automatic range | auto |
| `power` | Power for `kernel='power_exponential'`; must be in `(0, 2]` | `1.0` |
| `stationary` | If True, use a stationary GP with no linear trend in the null space | `False` |
| `xt` | Knot subsampling options: `{"max_knots": int, "seed": int}` | `{"max_knots": 2000, "seed": 1}` |

```python
from jaxgam import GAM

# Defaults: Matérn 3/2, automatic rho, non-stationary null space
results = GAM("y ~ s(x, z, bs='gp', k=30)").fit(df)

# Power-exponential kernel with explicit range and squared-exponential power
results = GAM(
    "y ~ s(x, bs='gp', kernel='power_exponential', rho=0.5, power=2.0)"
).fit(df)

# Stationary spherical GP
results = GAM(
    "y ~ s(x, bs='gp', kernel='spherical', stationary=True)"
).fit(df)
```

Implementation note: mgcv encodes these knobs as a single signed numeric
vector `m`; JaxGAM intentionally exposes them as named kwargs. Passing
`m=` raises `ValueError` - see §6.4 of
[the GP design doc](gaussian_process/design.md) for the mgcv to JaxGAM
mapping.

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
| `theta` | `float \| None` | Estimated theta for NB (None for standard families) |
| `method` | `str` | Smoothing parameter method ("REML" or "ML") |
| `n` | `int` | Number of observations |
