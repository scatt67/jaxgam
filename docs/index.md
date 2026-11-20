# jaxgam

Python port of R's [mgcv](https://cran.r-project.org/package=mgcv) package
for Generalized Additive Models.

jaxgam uses [JAX](https://github.com/google/jax) for JIT-compiled fitting
with automatic differentiation through the PIRLS inner loop and Newton
outer loop. No C compilation required.

Beyond the default in-memory fit, jaxgam can fit from data that does not fit
in memory, reproduce `bam(discrete=TRUE)`, and fit random effects with very
many levels. The [user guide](guide/choosing_a_route.md) explains each
option.

## Installation

```bash
# Clone and install with uv
git clone https://github.com/scatt67/jaxgam.git
cd jaxgam
uv sync
```

### Dependencies

- Python >= 3.11
- JAX and jaxlib >= 0.4.20
- NumPy >= 1.24
- SciPy >= 1.11
- pandas >= 2.0
- matplotlib >= 3.7
- numba >= 0.58
- scipy-openblas32 0.3.26.0.4 (a pinned OpenBLAS build, the one the reference
  R installation links, which jaxgam loads for the specific linear-algebra
  calls that must reproduce mgcv's results; it does not replace the BLAS that
  NumPy and SciPy use)

!!! warning "Apple silicon"
    The default NumPy and SciPy wheels on Apple silicon use Apple Accelerate,
    and large fits can then return wrong results or abort. `import jaxgam`
    warns when this applies. Install the OpenBLAS builds of NumPy and SciPy
    instead; see [Memory and performance](guide/memory_and_performance.md).

## Quick example

`GAM.fit()` returns one of three result types:

- `result="full"` (default) → `GAMResults` with prediction, summaries, plots,
  and in-sample state.
- `result="inference"` → lean `GAMInferenceResult` with new-data prediction and
  lightweight diagnostics, but no dense training state, `summary()`, or
  `plot()`.
- `result="prediction"` → `GAMPredictionResult`, which only predicts on new
  data and keeps no training rows. The streamed, discretized and operator
  routes return this type.

```python
import numpy as np
import pandas as pd
from jaxgam import GAM

# Generate data
rng = np.random.default_rng(42)
x = rng.uniform(0, 1, 200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, 200)
data = pd.DataFrame({"x": x, "y": y})

# Full analytical result. This is equivalent to the default .fit(data).
results = GAM("y ~ s(x, k=10, bs='cr')").fit(data, result="full")

# Inspect results
results.summary()
fig, axes = results.plot()

# Predict on new data
newdata = pd.DataFrame({"x": np.linspace(0, 1, 100)})
predictions = results.predict(newdata)
predictions, se = results.predict(newdata, se_fit=True)

# Lean inference result: already ready to predict; no conversion required.
lean = GAM("y ~ s(x, k=10, bs='cr')").fit(data, result="inference")
lean_predictions = lean.predict(newdata)
```

See the [Quickstart](quickstart.md) for a full tutorial covering
all families, smooth types, and post-estimation tools.

## Features

### Families

Gaussian, Binomial, Poisson, Gamma, and Negative Binomial -- each with REML
smoothing parameter selection. Each has a default link; other links are set
through the family class, for example `Gamma(link="log")`.

### Smooth types

| Formula syntax | Basis type |
|---|---|
| `s(x, bs='tp')` | Thin-plate regression spline (default) |
| `s(x, bs='ts')` | Thin-plate spline with shrinkage |
| `s(x, bs='cr')` | Cubic regression spline |
| `s(x, bs='cs')` | Cubic spline with shrinkage |
| `s(x, bs='cc')` | Cyclic cubic spline |
| `s(x, z, bs='gp')` | Low-rank Gaussian process smooth |
| `te(x1, x2)` | Tensor product smooth |
| `ti(x1, x2)` | Tensor interaction (no main effects) |
| `s(x, by=fac)` | Factor-by smooth (separate curve per level) |
| `s(g, bs='re')` | Random effect smooth |

### Fitting routes

The default fit holds the model matrix in memory and selects smoothing
parameters by exact REML, as `gam(method="REML")` does. The other routes are
opt-in:

| Route | Select with | Use it for |
|---|---|---|
| Dense (default) | nothing | Data that fits in memory |
| Fellner-Schall | `optimizer="efs"` | An alternative smoothing parameter optimizer, as in `gam(optimizer="efs")` |
| Gaussian QR | `FitControl(gaussian_compression=True)` | Gaussian identity-link models with many rows |
| Streamed | `FitControl(execution="stream")` | Data read in batches from a row source, for models built from cubic regression splines |
| Discretized | `FitControl(execution="discrete")` | The method of `bam(discrete=TRUE)` |
| Operator | `FitControl(execution="operator")` | Random effects with very many levels |

Start with [Choosing a route](guide/choosing_a_route.md).

### Post-estimation

- `predict()` -- response or link scale, with optional standard errors
- `summary()` -- parametric and smooth term significance tests
- `plot()` -- 1D smooth curves with SE bands, 2D contour plots, rug marks

### Result types

- `GAMResults` -- the default full result for prediction, summaries, plots, and
  in-sample diagnostics.
- `GAMInferenceResult` -- a lean `result="inference"` fit that keeps new-data
  prediction and cheap diagnostics without dense training state. It is already
  ready to predict; no conversion is required.
- `GAMPredictionResult` -- the `result="prediction"` fit: new-data prediction
  only, with no training rows. See
  [Prediction results](guide/prediction_results.md).
- `GAMPredictor` -- the prediction-only, same-version picklable core returned by
  a result's optional `to_predictor()` handoff. On an inference result,
  that method returns the existing core and does not reduce memory further.

## Performance

jaxgam uses JAX's XLA compiler. The first fit in a process pays a one-time
JIT compilation, and compiled code is cached on disk for later sessions.
Benchmarks against R's `gam` and `bam` are in the
[README](https://github.com/scatt67/jaxgam#performance).

## Limitations

These are deliberate scope boundaries, not bugs:

1. **No sparse solver.** Models with > ~5,000 basis functions will hit the
   dense memory ceiling. Random effects with very many levels can use the
   operator route instead.
2. **Five families.** Gaussian, Binomial, Poisson, Gamma, and negative
   binomial are supported. Other extended families (Tweedie, Beta, SHASH,
   Cox PH, ordered categorical) are not yet available.
3. **The default route holds the design matrix in memory.** For data that
   does not fit, use the streamed or discretized route. Those routes return
   prediction results only: no `summary()` or `plot()`.
4. **No factor-smooth interactions.** `bs="fs"` is not implemented.
5. **Single device only.** No multi-GPU or distributed fitting.
6. **No GAMM.** Correlated random effects (`gamm()`) are not supported.
7. **REML only.** `method="ML"` and GCV/UBRE smoothness selection are not
   available.

## Development

```bash
# Install dev dependencies (add --extra r for the R comparison tests)
uv sync --extra dev

# Run tests on this machine
make test-local

# Run the full suite in Docker, with the pinned R
make test

# Run linter
make lint
```

R comparison tests need R 4.5.2 with mgcv 1.9-3 and rpy2. They are skipped
when R is missing or the versions differ; the Docker image has the pinned
versions.

The [Architecture](architecture/index.md) section describes how the code is
organized, for maintainers and contributors.
