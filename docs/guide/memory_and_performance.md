# Memory and performance

This page lists the settings that bound how much memory a fit or a
prediction may use, the error each one raises when a fit would exceed it,
and what to change. It also covers compile time, BLAS threads and a problem
specific to Apple silicon Macs.

**Use this page when** a fit or prediction raises `MemoryError`, when you
want to bound the memory of a row-source fit, or when you set up a machine
to run jaxgam.

**Keep in mind** that every budget is jaxgam's own estimate of the arrays it
allocates, checked before it allocates them. None is a hard cap on the
process: the row source's own storage, allocator overhead and the workspace
of the underlying linear algebra libraries are outside it. The dense fit
itself is not budgeted at all.

All examples on this page use this setup block:

```python
import tempfile

import numpy as np
import pandas as pd
from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data import DataFrameRowSource, DiscretizationBudget

rng = np.random.default_rng(1)
n = 20_000
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
y = np.sin(2 * np.pi * x) + 0.5 * z + rng.normal(0, 0.3, n)
data = pd.DataFrame({"x": x, "z": z, "y": y})
source = DataFrameRowSource(data, response="y")
newdata = pd.DataFrame({"x": np.linspace(0, 1, 500), "z": 0.5})


def discrete_control(discretization=None, **options):
    """FitControl for the discretized route with smoothing parameters estimated."""
    return FitControl(
        execution="discrete",
        discretization=discretization or ContinuousDiscretizationControl(),
        smoothing_strategy="working_model_reml",
        **options,
    )
```

## Which setting bounds what

| Setting | Default | Bounds |
|---|---|---|
| `FitControl.batch_rows` | `65_536` | Rows per scan of a row source on the streamed, discretized and operator routes, and the default batch of `predict_iter()`. Streamed setup scans use smaller batches when needed to fit `memory_budget_bytes`. The dense route ignores it. |
| `FitControl.memory_budget_bytes` | 512 MiB | Streamed route: setup scans and the fit workspace. Discretized route: the whole fit workspace. Operator route: preparation and fit. Prediction results on every route: `Vp` and the workspace of each prediction call. |
| `FitControl.output_budget_bytes` | 512 MiB | Bytes one `predict()` or `predict_matrix()` call on a prediction result may return; each `predict_iter()` batch counts as one call. |
| `DiscretizationBudget.max_table_bytes` | 64 MiB | Discretized route: the compact tables of distinct rows and the basis evaluated on them, summed over all groups of variables. |
| `DiscretizationBudget.max_selector_bytes` | 256 MiB | Discretized route: the in-memory row selectors (one index per row and group) and their validation state. |
| `DiscretizationBudget.max_batch_bytes` | 64 MiB | Discretized route: the workspace of one batch, which grows with `batch_rows`. |
| `DiscretizationBudget.max_unique_rows` | `1_000_000` | Discretized route: distinct rows in any one group. |
| `ContinuousDiscretizationControl.selector_memmap_directory` | `None` | Discretized route: when set to an existing directory, row selectors are written there as files instead of held in memory, and are removed when the fit ends. |

`DiscretizationBudget` comes from `jaxgam.data` and goes into
`ContinuousDiscretizationControl(budget=...)`. `uncertainty`, `batch_rows`
and both `FitControl` budgets act on the dense route only with
`result="prediction"`; full and inference results are not budgeted.

## Errors and what to change

| Error message (numbers vary) | Raised by | Change |
|---|---|---|
| `Streamed setup requires at least N bytes to scan one row, exceeding the setup memory budget of M bytes (FitControl.memory_budget_bytes).` | Streamed setup pass | Raise `memory_budget_bytes`. Any model with a cubic smooth needs at least about 17 MB here. |
| `Known streamed QR PIRLS workspace requires N bytes, exceeding FitControl.memory_budget_bytes=M. Setup scans are budgeted separately.` (or `Known streamed PIRLS workspace ...`) | Streamed fit | Lower `batch_rows`, or raise `memory_budget_bytes`. The estimate is for a full batch of `batch_rows` rows even when the source has fewer. |
| `Discrete fit workspace requires N bytes, exceeding FitControl.memory_budget_bytes=M.` | Discretized fit | Raise `memory_budget_bytes`, or lower `batch_rows`. |
| `Operator preparation needs at most N bytes; budget is M. Lower batch_rows or raise the memory budget.`, `Frozen operator requires N bytes; budget is M. Lower batch_rows or raise the memory budget.` | Operator setup pass and fit | Lower `batch_rows`, or raise `memory_budget_bytes`. The operator fit already plans for no more rows per batch than the source holds. In the second message `M` is what is left of the budget after the fit's other arrays. |
| `uniquecombs rows exceed construction byte budget: requires N bytes, budget is M.`, `compact smooth predictor workspace exceeds budget: requires N bytes, budget is M.` | `max_table_bytes` | Raise `max_table_bytes`, or use a coarser grid (`m=`). |
| `row selector/validation state exceeds in-memory selector budget: requires N bytes, budget is M.` | `max_selector_bytes` | Raise `max_selector_bytes`, or set `selector_memmap_directory`. |
| `range-analysis batch workspace exceeds budget: requires N bytes, budget is M.` | `max_batch_bytes` | Raise `max_batch_bytes`, or lower `batch_rows`. |
| `uniquecombs rows exceed row cap: requires N bytes, budget is M.` | `max_unique_rows` | Raise `max_unique_rows`, or use a coarser grid. The numbers in this message are table bytes, not rows. |
| `Prediction output requires N bytes, exceeding output budget M bytes. Use predict_iter() with smaller batches.` | `predict()` on a prediction result | Use `predict_iter()` with a smaller `batch_rows`, or raise `output_budget_bytes`. |
| `Prediction workspace requires N bytes, exceeding memory budget M bytes. Use predict_iter() with smaller batches.` | `predict()` on a prediction result | As above, or raise `memory_budget_bytes`. |
| `Vp requires N bytes, exceeding FitControl.memory_budget_bytes=M.` | `uncertainty="covariance"` | Raise `memory_budget_bytes`, or use `uncertainty="fisher"`. |
| `Vp materialization requires N workspace bytes, exceeding budget M bytes.` | `materialize_covariance()` | Pass a larger budget. |

All of these are `MemoryError`. A budget that is not a positive integer
raises `ValueError` when you construct the control.

The streamed fit workspace check, for example, charges a full batch of
`batch_rows` rows, whatever the size of the source. With a 24 MiB budget, a
small model on 2,000 rows fails at the default batch size of 65,536 rows and
fits once `batch_rows` is lowered:

```python
small_source = DataFrameRowSource(data.iloc[:2000], response="y")
budget = 24 * 1024 * 1024
formula = "y ~ s(x, bs='cr') + z"
try:
    control = FitControl(execution="stream", linear_solver="qr", memory_budget_bytes=budget)
    GAM(formula, control=control).fit(small_source, result="prediction")
except MemoryError as error:
    print(str(error)[:44])

control = FitControl(
    execution="stream", linear_solver="qr", memory_budget_bytes=budget, batch_rows=4096
)
fit = GAM(formula, control=control).fit(small_source, result="prediction")
print(fit.converged)
```

```text
Known streamed QR PIRLS workspace requires 2
True
```

A prediction larger than `output_budget_bytes` goes through
`predict_iter()` instead:

```python
small = FitControl(output_budget_bytes=1_000)
fit = GAM("y ~ s(x) + z", control=small).fit(data, result="prediction")
try:
    fit.predict(newdata)
except MemoryError as error:
    print(error)
rows = sum(len(p) for _, p in fit.predict_iter(DataFrameRowSource(newdata), batch_rows=100))
print(rows)
```

```text
Prediction output requires 4000 bytes, exceeding output budget 1000 bytes. Use predict_iter() with smaller batches.
500
```

## Thin-plate smooths on the discretized route

The default `DiscretizationBudget` is too small for a default thin-plate
smooth, `s(x)`, on the discretized route, while a cubic regression spline
fits within it:

```python
try:
    GAM("y ~ s(x) + z", control=discrete_control()).fit(source, result="prediction")
except MemoryError as error:
    print(error)

cubic = GAM("y ~ s(x, bs='cr') + z", control=discrete_control())
print(cubic.fit(source, result="prediction").converged)
```

```text
compact smooth predictor workspace exceeds budget: requires 89497761 bytes, budget is 67108864.
True
```

To fit the thin-plate smooth, raise `max_table_bytes`, or use a coarser grid
with `m=` (which changes the discretization and therefore the estimates):

```python
larger = ContinuousDiscretizationControl(
    budget=DiscretizationBudget(max_table_bytes=128 * 1024 * 1024)
)
print(GAM("y ~ s(x) + z", control=discrete_control(larger)).fit(
    source, result="prediction"
).converged)

coarser = ContinuousDiscretizationControl(m=200)
print(GAM("y ~ s(x) + z", control=discrete_control(coarser)).fit(
    source, result="prediction"
).converged)
```

```text
True
True
```

Row selectors can go to disk when `max_selector_bytes` is the limit. The
directory must exist; its selector files are removed when the fit ends:

```python
directory = tempfile.mkdtemp()
on_disk = ContinuousDiscretizationControl(
    selector_memmap_directory=directory,
    budget=DiscretizationBudget(max_selector_bytes=100_000),
)
fit = GAM("y ~ s(x, bs='cr') + z", control=discrete_control(on_disk)).fit(
    source, result="prediction"
)
print(fit.converged)
```

```text
True
```

See [Discretized fitting](discretized.md) for the grid (`m=`) and the rest of
the discretization policy.

## First call and the compilation cache

The first fit of a model in a Python process takes longer than later fits of
the same model and data sizes, because JAX compiles the computations it
runs. Importing jaxgam also compiles a few helper functions.

To save that work across processes, importing jaxgam turns on JAX's
persistent compilation cache in `~/.cache/jaxgam/jax`, and caches every
compiled computation however quickly it compiled. Entries are written from
the first import on.

To put the cache elsewhere, set `JAX_COMPILATION_CACHE_DIR` before Python
imports jaxgam. A `jax.config.update("jax_compilation_cache_dir", ...)` made
before the import is overwritten by jaxgam's default.

```bash
JAX_COMPILATION_CACHE_DIR=/path/to/cache python fit_model.py
```

To stop jaxgam from configuring the cache, set `JAXGAM_NO_COMPILATION_CACHE`
to any non-empty value before the import (even `0` counts). If
`JAX_COMPILATION_CACHE_DIR` is also set, JAX itself still caches there.

```bash
JAXGAM_NO_COMPILATION_CACHE=1 python fit_model.py
```

## BLAS threads

jaxgam is tested with one BLAS thread, and we recommend the same:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python fit_model.py
```

Set them in the environment before Python starts. A multithreaded BLAS adds up sums in an order that depends on the number of
cores, so the last digits of results can differ from machine to machine;
with one thread they do not. The LAPACK library jaxgam ships for its own
basis computations (the `scipy-openblas32` package) always runs on one
thread, whatever these variables say.

There is no thread-count argument in jaxgam. JAX's CPU backend still uses
several cores by itself.

## Apple silicon

On Apple silicon Macs the default NumPy and SciPy wheels use Apple
Accelerate, and JAX's CPU linear algebra routines take their LAPACK from SciPy. When
several LAPACK calls run at the same time inside one compiled computation,
memory can be corrupted: the process may abort, and occasionally results are
wrong with no error. This has been seen fitting models with about 100 smooths
(about 1,000 coefficients), and not with 50 smooths or fewer. Linux is not
affected.

Importing jaxgam on an affected Mac issues a `RuntimeWarning` that names the
fix for the NumPy and SciPy versions you have installed. The fix is to
install the OpenBLAS builds of the same versions. For numpy 2.3.5 and
scipy 1.17.0 (the versions in jaxgam's lock file), with uv:

```bash
uv pip install --reinstall --no-deps --python-platform aarch64-apple-darwin \
    numpy==2.3.5 scipy==1.17.0
```

or with pip:

```bash
pip download --only-binary=:all: --no-deps --platform macosx_12_0_arm64 \
    numpy==2.3.5 scipy==1.17.0 -d wheels
pip install --force-reinstall --no-deps wheels/*.whl
```

Running `uv sync` afterwards can put the Accelerate builds back; repeat the
step after it. To silence the warning without changing the install, set
`JAXGAM_ALLOW_ACCELERATE=1` (exactly `1`).

## Related pages

- [Choosing a route](choosing_a_route.md)
- [Row sources](row_sources.md)
- [Streamed fitting](streamed.md), [Discretized fitting](discretized.md),
  [Operator fitting](operator.md)
- [Prediction results](prediction_results.md)
- API reference: [FitControl](../api.md#fit-control),
  [Row sources](../api.md#row-sources)
