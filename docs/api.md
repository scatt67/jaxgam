# API Reference

## GAM

Model specification and fit orchestration. `fit()` returns either a full
`GAMResults` or a lean `GAMInferenceResult`, selected with the keyword-only
`result` argument.

```python
from jaxgam import GAM, GAMInferenceResult, GAMResults

model = GAM("y ~ s(x)", family="gaussian")
results: GAMResults = model.fit(data, result="full")  # explicit default mode
results.predict(newdata)
results.summary()

lean: GAMInferenceResult = model.fit(data, result="inference")
lean.predict(newdata)
```

::: jaxgam.api.GAM
    options:
      members:
        - __init__
        - fit

---

## Fit result modes

`GAM.fit()` accepts `result="full"` (the default) or
`result="inference"`:

| Mode | Return type | Intended use |
|---|---|---|
| `"full"` | `GAMResults` | Interactive analysis, self-prediction, summaries, and plots |
| `"inference"` | `GAMInferenceResult` | New-data prediction with substantially less retained training state |

If you request `result="inference"`, the returned object is already lean and
ready to predict. **You do not need to call `to_predictor()` afterward.**

The inference mode keeps coefficients, covariance, the fitted family snapshot,
prediction transforms, and cheap fit diagnostics. It drops dense training
arrays, in-sample caches, training data, and fitting-only penalty caches. It has
no `summary()` or `plot()`, and `predict()` requires `newdata`.

This mode reduces memory **retained after fitting**. It does not reduce peak
fit-time memory: the dense model matrix and penalties are still constructed and
used during the fit before the lean result is materialized.

### Which object should I keep?

| Object | Keep it when you need | What `to_predictor()` does |
|---|---|---|
| `GAMResults` | Prediction, `summary()`, `plot()`, or in-sample state | Builds an independent prediction-only core. The full result remains in memory unless you discard it. |
| `GAMInferenceResult` | New-data prediction plus fit diagnostics and labels | Returns the predictor it already contains. It performs no copy and provides no further memory reduction. |
| `GAMPredictor` | Only `predict()` and `predict_matrix()` | Not applicable; it is already the prediction-only core. |

`to_predictor()` is therefore an optional handoff boundary, not a required step
after an inference fit. It is most useful when downstream code should receive a
deliberately prediction-only interface, or when extracting a lean core from a
full result.

---

## GAMResults

Frozen full-results object returned by `GAM.fit()` by default. All
post-estimation methods (prediction, summary, plotting) live here, and
`to_predictor()` creates an independent prediction-only core.

```python
from jaxgam import GAMResults
```

::: jaxgam.results.GAMResults
    options:
      members:
        - predict
        - predict_matrix
        - to_predictor
        - summary
        - plot

---

## GAMInferenceResult

Lean result returned by `GAM.fit(..., result="inference")`. It supports
new-data prediction and prediction-matrix construction, but deliberately has no
training-data-backed `summary()` or `plot()` surface. Unlike
`GAMResults.predict()`, its `predict()` method requires `newdata`.

It retains `coefficients`, `Vp`, `family`, `formula`, `smooth_info`, and
`term_names`, plus the scalar and small-array diagnostics `edf`, `edf1`,
`edf_total`, `deviance`, `null_deviance`, `score`, `scale`, `theta`,
`smoothing_params`, `converged`, `n_iter`, `convergence_info`, `method`,
`lambda_strategy`, `execution_path`, and `n`. The smooth and term metadata make
the retained EDF arrays interpretable without retaining the training setup.

```python
from jaxgam import GAM, GAMInferenceResult

lean: GAMInferenceResult = GAM("y ~ s(x)").fit(
    data,
    result="inference",
)
predictions, se = lean.predict(newdata, se_fit=True)
```

That is the complete inference workflow. Call `lean.to_predictor()` only if a
downstream consumer should receive prediction methods without the retained fit
diagnostics. On an inference result, the method returns the already-composed
predictor and does not reduce memory further.

::: jaxgam.results.GAMInferenceResult
    options:
      members:
        - predict
        - predict_matrix
        - to_predictor

---

## GAMPredictor

Frozen, prediction-only core produced by either result type's
`to_predictor()`. Its `coefficients` and `Vp` arrays are defensively copied and
read-only. It supports `predict(newdata, ...)` and
`predict_matrix(newdata)` without retaining dense training arrays or the
summary/plot surface.

`GAMPredictor` is a boundary object for prediction-only consumers. It is not a
second inference mode:

```python
# Already lean and directly usable: keep diagnostics and labels.
lean = model.fit(data, result="inference")
predictions = lean.predict(newdata)

# Optional: expose only prediction state to another component.
predictor = lean.to_predictor()  # returns lean's existing core; no copy

# Or extract prediction state from a full analytical result.
full = model.fit(data, result="full")
predictor_from_full = full.to_predictor()  # constructs an independent core
```

Calling `full.to_predictor()` does not mutate or slim `full`; discard the full
result if its training-backed state is no longer needed.

stdlib `pickle` is the default for same-version / transient handoff;
`cloudpickle` is required only for locally-defined custom links/families;
neither is a durable cross-version format. Loading a predictor pickled by a
different jaxgam version emits a warning. As with all pickle data, only load
trusted input. JaxGAM does not provide a serialized model format or
`save()`/`load()` API.

```python
import pickle

full = model.fit(data, result="full")
predictor = full.to_predictor()  # optional prediction-only handoff
blob = pickle.dumps(predictor)
restored = pickle.loads(blob)
predictions = restored.predict(newdata)
```

You may also pickle a `GAMInferenceResult` directly when the receiving process
needs its diagnostics. Converting it to `GAMPredictor` narrows the receiving
interface; it is not an additional memory optimization.

::: jaxgam.inference.predictor.GAMPredictor
    options:
      members:
        - predict
        - predict_matrix

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
| `k` | Marginal basis dimension (scalar applied to all margins). `-1` means auto-select, which resolves to `5` per (1-D) margin, matching R mgcv's `te()`/`ti()` default of 5^d. | -1 (auto) |

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

Your class must subclass `jaxgam.families.ExponentialFamily`. Register it under
a key that is not already built in (the built-in keys are `gaussian`,
`binomial`, `poisson`, `gamma`, and `nb`; re-registering a built-in raises a
`ValueError`).

```python
from jaxgam.families import family_registry, ExponentialFamily

class Tweedie(ExponentialFamily):
    ...  # implement variance(), deviance_resids(), initialize(), etc.

family_registry.register("tweedie", Tweedie)

# Now usable by name:
GAM("y ~ s(x)", family="tweedie").fit(data)
```

Negative Binomial is built in — use `family="nb"` directly, no registration.

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

smooth_registry.available      # ('cc', 'cr', 'cs', 'gp', 're', 'te', 'ti', 'tp', 'ts')
"tp" in smooth_registry        # True
len(smooth_registry)           # 9
```

---

## GAMResults attributes

The `GAMResults` object returned by the default `fit()` mode is a frozen
dataclass: fields cannot be reassigned, although contained mutable objects are
not recursively frozen.

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
| `score` | `float` | REML value at convergence |
| `X` | `ndarray (n, p)` | Design matrix |
| `y` | `ndarray (n,)` | Response vector |
| `weights` | `ndarray (n,)` | Prior weights |
| `family` | `ExponentialFamily` | Family object used for fitting |
| `formula` | `str` | Model formula |
| `theta` | `float \| None` | Estimated theta for NB (None for standard families) |
| `method` | `str` | Smoothing parameter method (always `"REML"` in v1.0; `"ML"` raises `NotImplementedError`) |
| `n` | `int` | Number of observations |
