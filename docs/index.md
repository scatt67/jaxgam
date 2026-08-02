# jaxgam

Python port of R's [mgcv](https://cran.r-project.org/package=mgcv) package
for Generalized Additive Models.

jaxgam uses [JAX](https://github.com/google/jax) for JIT-compiled fitting
with automatic differentiation through the PIRLS inner loop and Newton
outer loop. No C compilation required.

## Installation

```bash
# Clone and install with uv
git clone https://github.com/scatt67/jaxgam.git
cd jaxgam
uv sync
```

### Dependencies

- Python >= 3.11
- JAX >= 0.4.20
- NumPy >= 1.24
- SciPy >= 1.11
- pandas >= 2.0
- matplotlib >= 3.7

## Quick example

`GAM.fit()` returns one of two result types:

- `result="full"` (default) → `GAMResults` with prediction, summaries, plots,
  and in-sample state.
- `result="inference"` → lean `GAMInferenceResult` with new-data prediction and
  lightweight diagnostics, but no dense training state, `summary()`, or
  `plot()`.

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

Gaussian, Binomial, Poisson, Gamma, and Negative Binomial -- each with its
default link and REML smoothing parameter selection.

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
| `s(g, bs='re')` | Dense random effect smooth |

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
- `GAMPredictor` -- the prediction-only, same-version picklable core returned by
  either result's optional `to_predictor()` handoff. On an inference result,
  that method returns the existing core and does not reduce memory further.

## Performance

jaxgam uses JAX's XLA compiler. After a one-time JIT compilation
(~275ms with persistent disk cache), fits are 1-16x faster than R's mgcv
at n=500 and competitive at n=10,000.

## v1.0 limitations

These are deliberate scope boundaries, not bugs:

1. **No sparse solver.** Models with > ~5,000 basis functions will hit the
   dense memory ceiling.
2. **Five families.** Gaussian, Binomial, Poisson, Gamma, and negative
   binomial are supported. Other extended families (Tweedie, Beta, SHASH,
   Cox PH, ordered categorical) are not yet available.
3. **Dense design matrix must fit in memory.** Datasets with > ~10M rows
   require chunked processing, which is not implemented.
4. **Dense random effects only.** `bs="re"` is supported for modest
   cardinality. High-cardinality random effects and `bs="fs"` require
   sparse linear algebra, which is not implemented.
5. **Single device only.** No multi-GPU or distributed fitting.
6. **No GAMM.** Correlated random effects (`gamm()`) are not supported.

See the [Design Doc](design.md) Section 1.2 for details on what
is planned for v1.1+.

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/

# Run linter
make lint
```

R comparison tests require R with mgcv and rpy2 installed. They are
skipped automatically if R is not available.
