# jaxgam

Python port of R's [mgcv](https://cran.r-project.org/package=mgcv) package
for Generalized Additive Models.

jaxgam uses [JAX](https://github.com/google/jax) for JIT-compiled fitting
with automatic differentiation through the PIRLS inner loop and Newton
outer loop. No C compilation required.

## Installation

```bash
# Clone and install with uv
git clone https://github.com/<org>/jaxgam.git
cd jaxgam
uv sync
```

## Quickstart

```python
import numpy as np
import pandas as pd
from jaxgam import GAM

# Generate data
rng = np.random.default_rng(42)
x = rng.uniform(0, 1, 200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, 200)
data = pd.DataFrame({"x": x, "y": y})

# Fit a GAM
model = GAM("y ~ s(x, k=10, bs='cr')").fit(data)

# Inspect results
model.summary()
fig, axes = model.plot()

# Predict on new data
newdata = pd.DataFrame({"x": np.linspace(0, 1, 100)})
predictions = model.predict(newdata)
predictions, se = model.predict(newdata, se_fit=True)
```

See [docs/quickstart.md](docs/quickstart.md) for a full tutorial covering
all families, smooth types, and post-estimation tools.

## What v1.0 supports

### Families

Gaussian, Binomial, Poisson, Gamma --- each with its default link and
REML/ML smoothing parameter selection.

### Smooth types

| Formula syntax | Basis type |
|---|---|
| `s(x, bs='tp')` | Thin-plate regression spline (default) |
| `s(x, bs='cr')` | Cubic regression spline |
| `s(x, bs='cs')` | Cubic spline with shrinkage |
| `s(x, bs='cc')` | Cyclic cubic spline |
| `te(x1, x2)` | Tensor product smooth |
| `ti(x1, x2)` | Tensor interaction (no main effects) |
| `s(x, by=fac)` | Factor-by smooth (separate curve per level) |

### Post-estimation

- `predict()` --- response or link scale, with optional standard errors
- `summary()` --- parametric and smooth term significance tests
- `plot()` --- 1D smooth curves with SE bands, 2D contour plots, rug marks

## v1.0 limitations

These are deliberate scope boundaries, not bugs:

1. **No sparse solver.** Models with > ~5,000 basis functions will hit the
   dense memory ceiling. Factor-by with many levels or large tensor products
   are most affected.
2. **Four families only.** Negative binomial, Tweedie, Beta, and other
   extended families are not yet available.
3. **Dense design matrix must fit in memory.** Datasets with > ~10M rows
   require chunked processing, which is not implemented.
4. **No random effects.** `bs="re"` (random effects) and `bs="fs"`
   (factor-smooth interactions) require sparse linear algebra.
5. **Single device only.** No multi-GPU or distributed fitting.
6. **No GAMM.** Correlated random effects (`gamm()`) are not supported.

See the [design document](docs/design.md) Section 1.2 for details on what
is planned for v1.1+.

## Performance

jaxgam uses JAX's XLA compiler for JIT-compiled fitting. Performance
depends on whether the JIT cache is warm (compiled code reused) or cold
(first fit triggers compilation).

### Warm fits (JIT cached)

After the first fit for a given model structure, subsequent fits reuse
compiled XLA code. jaxgam is **1.3--12x faster** than R across all
families and smooth types, with the advantage growing at larger n:

| n | Single smooth | Two smooths | Tensor product | Factor-by |
|---:|---:|---:|---:|---:|
| 500 | 1.3--1.5x | 1.1--2.3x | 0.4--1.4x | 0.9--1.4x |
| 10,000 | 1.9--3.6x | 1.7--6.0x | 1.5--8.5x | 2.5--3.7x |
| 100,000 | 3.2--6.6x | 3.0--4.7x | 3.1--5.7x | 2.0--4.1x |
| 500,000 | 2.9--4.6x | 2.5--3.5x | 2.4--12.0x | 1.8--3.1x |

Tensor products with Gamma show the largest gains (12x at n=500k)
because R's penalty iteration cost scales with complexity while jaxgam's
fused XLA kernels amortize overhead.

### Cold starts

The first fit includes JIT tracing + XLA compilation (~700--1200ms
overhead). This makes jaxgam slower than R for small datasets on first
use, but the compiled code is cached to disk and reused across sessions.

![Cold-start speedup vs dataset size](docs/img/speedup_vs_n.png)

The crossover where even a cold jaxgam fit beats R is around n=100,000.

### High-dimensional models

For models with many basis functions (k=100--500), jaxgam's XLA-compiled
dense linear algebra outperforms R even on the very first cold-start fit:

![jaxgam vs R at large p](docs/img/large_p_results.png)

### When to use jaxgam over R

**jaxgam is a good fit when:**
- You need GAMs in a Python workflow without switching to R
- You fit the same model structure repeatedly (bootstrap, CV,
  simulation) --- warm fits are 2--12x faster than R
- Your datasets are large (n > 10,000) --- the XLA advantage grows
  with n
- You use tensor products or Gamma family --- these see the largest
  speedups

**R's mgcv may be better when:**
- You need one-shot fits on small data (n < 2,000) and cold-start
  latency matters
- You need features beyond v1.0 scope (sparse solvers, extended
  families, random effects, bam)
- You need multi-GPU or distributed fitting --- jaxgam automatically
  parallelizes across CPU cores and runs on GPU, but multi-device
  SPMD is on the roadmap

A persistent compilation cache (`~/.cache/jaxgam/jax/`) is enabled by
default to minimize cold-start overhead across Python sessions.
Disable it with `PYMGCV_NO_COMPILATION_CACHE=1`.

## Correctness

jaxgam is validated against R's mgcv 1.9-3 across a 1,461-test suite.
Every model configuration (4 families x 6 smooth types) is fitted in
both jaxgam and R, then compared value-by-value:

- **Coefficients, fitted values, deviance** --- must match R at STRICT
  (rtol=1e-10) or MODERATE (rtol=1e-4) tolerance depending on the
  model type
- **Smoothing parameters** --- compared at MODERATE or LOOSE (rtol=1e-2)
  because the REML surface is flat near the optimum
- **Basis matrices and penalty matrices** --- compared element-wise
  against R's `smoothCon()` output with sign normalization to handle
  LAPACK eigenvector sign ambiguity
- **Summary statistics** --- EDF, p-values, and significance tests
  validated against R's `summary.gam()`
- **Predictions and standard errors** --- `predict()` output compared
  against R's `predict.gam()` on both training and new data

R comparison tests run inside a Docker container with pinned R 4.5.2 +
mgcv 1.9-3 to ensure reproducibility. Tests are skipped automatically
when running locally without the correct R version.

Hard-gate invariants are checked on every test run: REML objective
monotonicity, Hessian symmetry/PSD, penalty PSD, EDF bounds, deviance
non-negativity, and no NaN in converged models.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests locally (R tests auto-skip without pinned R version)
make test-local

# Run full test suite in Docker (includes R comparison tests)
make test

# Run linter
make lint
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide,
including Docker setup, testing rules, and PR conventions.

## License

See [LICENSE](LICENSE) for details.
