# API reference

This page lists every public class and argument, what each one does, and
where it stops. For task-oriented explanations, start with the user guide:
[Choosing a route](guide/choosing_a_route.md).

The top-level package exports `GAM`, `FitControl`, `EFSControl`,
`OperatorControl`, `ContinuousDiscretizationControl`,
`EFSOptimizerDiagnostics`, `GAMResults`, `GAMInferenceResult`,
`GAMPredictionResult` and `GAMPredictor`. Row sources and
`DiscretizationBudget` come from `jaxgam.data`, families from
`jaxgam.families` and links from `jaxgam.links`.

!!! note "Setup for the examples on this page"
    The examples run in order in one Python process. The block below
    defines the data they share: a data frame `data` with 600 rows, a row
    source `source` over it with response `y`, a second row source
    `count_source` with the count response `count`, and a small `newdata`
    frame. The grouping factor `g` has 300 levels with two rows each; `f`
    is a factor with three levels.

```python
import numpy as np
import pandas as pd

from jaxgam import GAM, FitControl
from jaxgam.data import DataFrameRowSource

rng = np.random.default_rng(1)
n = 600
levels = [f"g{i:03d}" for i in range(300)]
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
g = pd.Categorical(np.repeat(levels, 2), categories=levels)
group_effect = rng.normal(0, 0.5, len(levels))
data = pd.DataFrame(
    {
        "x": x,
        "z": z,
        "g": g,
        "f": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        "y": np.sin(2 * np.pi * x) + 0.5 * z + rng.normal(0, 0.3, n),
        "count": rng.poisson(np.exp(0.3 + 0.5 * x + group_effect[g.codes])).astype(
            float
        ),
    }
)
source = DataFrameRowSource(data, response="y")
count_source = DataFrameRowSource(data, response="count")
newdata = pd.DataFrame(
    {
        "x": np.linspace(0.05, 0.95, 5),
        "z": 0.5,
        "g": pd.Categorical(["g000"] * 5, categories=levels),
        "f": pd.Categorical(["a"] * 5, categories=["a", "b", "c"]),
    }
)
```

---

## GAM

`GAM` holds a model specification. `GAM.fit()` fits it and returns a frozen
result object.

```python
model = GAM("y ~ s(x) + s(z)", family="gaussian")
fit = model.fit(data)  # result="full" is the default
print(fit.converged, fit.execution_route, fit.lambda_strategy)
```

```text
True dense newton_reml
```

### Constructor

`GAM(formula, family="gaussian", method="REML", sp=None, control=None, **kwargs)`

| Argument | Default | What it does |
|---|---|---|
| `formula` | required | Model formula, for example `"y ~ s(x) + z"`. See [Formula syntax](#formula-syntax). The formula is parsed in `fit()`, not in the constructor. |
| `family` | `"gaussian"` | A family name (`"gaussian"`, `"binomial"`, `"poisson"`, `"gamma"`, `"nb"`) or a family instance. A name always gives the default link; pass an instance such as `Gamma(link="log")` for another link. An unknown name raises `KeyError` in `fit()`. See [Families](#families). |
| `method` | `"REML"` | Smoothing parameter criterion. Only `"REML"` is accepted (case-insensitive). `"ML"` raises `NotImplementedError`; any other value raises `ValueError`. |
| `sp` | `None` | Fixed smoothing parameters: one positive, finite value per penalty, on the natural scale. When given, smoothing parameters are not estimated. A wrong length or a non-positive value raises `ValueError` in `fit()`. `fit.smoothing_params` reports the values you supplied. |
| `control` | `None` | A [`FitControl`](#fit-control). `None` means `FitControl()`, the dense route. Anything else raises `TypeError("control must be a FitControl instance or None.")`. |

`**kwargs` accepts only these keywords. Any other keyword raises
`TypeError`.

| Keyword | Accepted values | What it does |
|---|---|---|
| `optimizer` | `"newton"` (default), `"efs"` | `"efs"` selects the [Fellner-Schall optimizer](#fellner-schall-optimizer) when smoothing parameters are estimated. Another string raises `NotImplementedError`; a non-string raises `ValueError`. |
| `device` | `None` (default), `"cpu"`, `"gpu"` | Where the fit runs. `None` leaves placement to JAX's default device. `"gpu"` raises `RuntimeError` in `fit()` when JAX has no GPU backend. Another value raises `ValueError`. The operator route accepts only `None` or `"cpu"`. |
| `backend` | `None`, `"jax"` | Anything else raises `NotImplementedError`. |
| `select` | `False` | `True` (shrinkage selection) raises `NotImplementedError`. |
| `gamma` | `1.0` | Any other value raises `NotImplementedError`. |
| `knots` | `None` | User-supplied knots raise `NotImplementedError`. |

### `fit`

`fit(data, weights=None, offset=None, *, result="full")`

| Argument | What it does |
|---|---|
| `data` | A pandas DataFrame or a dict of columns on the dense route. A [row source](#row-sources) on the streamed, discretized and operator routes. |
| `weights` | Prior weights, shape `(n,)`. Dense route only. |
| `offset` | Offset added to the linear predictor, shape `(n,)`. Dense route only. |
| `result` | `"full"`, `"inference"` or `"prediction"`. Anything else raises `ValueError`. |

The three result modes:

| `result=` | Returns | Routes | Keeps |
|---|---|---|---|
| `"full"` (default) | [`GAMResults`](#gamresults) | dense | Everything: training design, fitted values, `Vp`, per-smooth EDF, `summary()`, `plot()` |
| `"inference"` | [`GAMInferenceResult`](#gaminferenceresult) | dense | Coefficients, `Vp`, per-smooth EDF and fit diagnostics; no training arrays |
| `"prediction"` | [`GAMPredictionResult`](#gampredictionresult) | all four | Coefficients and scalar diagnostics; standard errors only if `FitControl.uncertainty` asks for them |

What each route takes as data:

| Route | `FitControl(execution=...)` | `data` | Weights and offset | `result` |
|---|---|---|---|---|
| Dense | `"dense"` (default) | DataFrame or dict. A row source raises `TypeError`. | `fit(weights=, offset=)` | any of the three |
| Streamed | `"stream"` | Row source. A DataFrame raises `TypeError`. | Owned by the row source. Passing `weights=` or `offset=` to `fit()` raises `ValueError`. | `"prediction"` only. Any other value raises `NotImplementedError`. |
| Discretized | `"discrete"` | Row source | Owned by the row source | `"prediction"` only |
| Operator | `"operator"` | Row source | Owned by the row source | `"prediction"` only |

Because `fit()` defaults to `result="full"`, a fit on the streamed,
discretized or operator route must pass `result="prediction"`.

::: jaxgam.api.GAM
    options:
      heading_level: 3
      members:
        - __init__
        - fit

---

## Fit control

All five control classes are frozen dataclasses. Each checks its own fields
when constructed and raises `ValueError` (or `TypeError` where noted) with
the message quoted below. Integer fields must be Python `int` values: a
NumPy integer is rejected, except in `OperatorControl`. Combinations that
depend on the model or the data are checked in `GAM.fit()`; the route
sections under [Fitting routes](#fitting-routes) list them.

```python
from jaxgam import ContinuousDiscretizationControl, EFSControl, OperatorControl
from jaxgam.data import DiscretizationBudget

control = FitControl(
    execution="discrete",
    discretization=ContinuousDiscretizationControl(
        m=50, budget=DiscretizationBudget(max_table_bytes=128 * 1024 * 1024)
    ),
    smoothing_strategy="working_model_reml",
)
print(control.execution, control.discretization.m)
```

```text
discrete 50
```

### FitControl

`FitControl` chooses the route, the linear solver, the memory budgets and
what a prediction result keeps for standard errors.

| Field | Default | Read by | What it does |
|---|---|---|---|
| `execution` | `"dense"` | all | The route: `"dense"`, `"stream"`, `"discrete"` or `"operator"`. |
| `batch_rows` | `65_536` | streamed, discretized, operator; prediction results | Rows requested per row-source scan. Streamed setup scans use fewer rows when needed to fit `memory_budget_bytes`, and the operator fit uses the source's row count when that is smaller. Also the default batch of `GAMPredictionResult.predict_iter()` and the scan size of `bulk_link_variance()`. The dense fit does not read it. |
| `memory_budget_bytes` | `512 * 1024 * 1024` | streamed, discretized, operator; prediction results | Upper bound on the known arrays of the setup pass and the fit. A fit whose planned workspace exceeds it raises `MemoryError` before allocating. On every route it also bounds `Vp` and per-call prediction workspace of a prediction result. The dense fit itself is not budgeted. |
| `output_budget_bytes` | `512 * 1024 * 1024` | prediction results | Largest output one `predict()` or `predict_matrix()` call on a prediction result may return; each `predict_iter()` batch is one call. |
| `uncertainty` | `"none"` | prediction results | What a prediction result keeps for standard errors: `"none"` (nothing; `predict(se_fit=True)` raises `RuntimeError`), `"fisher"` (a factor of the penalized Fisher system), `"covariance"` (the dense `Vp` as well, checked against `memory_budget_bytes`). The operator route accepts only `"none"`. Full and inference results always keep `Vp`. |
| `linear_solver` | `"cholesky"` | all | `"cholesky"`, `"qr"` or `"pcg"`. Dense and discretized take `"cholesky"`; streamed takes `"cholesky"` or `"qr"`; operator requires `"pcg"`. |
| `gaussian_compression` | `False` | dense, streamed | Requests [Gaussian QR compression](#gaussian-qr-compression). Must be a Python `bool`. |
| `efs` | `EFSControl()` | dense, streamed, operator | Limits of the Fellner-Schall optimizer. |
| `operator` | `OperatorControl()` | operator | Limits of the operator fit. |
| `discretization` | `None` | discretized | The discretization policy. Required when `execution="discrete"`, refused otherwise. |
| `smoothing_strategy` | `"exact_reml"` | discretized | `"working_model_reml"` selects smoothing parameters on the discretized route. The discretized route cannot estimate them with `"exact_reml"`. |
| `working_model_max_iterations` | `200` | discretized | Iteration limit of the discretized fit, except a Gaussian identity fit with fixed `sp`. Reaching it gives a `RuntimeWarning` and `converged=False`. |
| `working_model_epsilon` | `1e-7` | discretized | Convergence threshold on the relative change in penalized deviance (and in the log scale, when the scale is estimated). |

Checks at construction, in order:

1. `execution` not one of the four: `"execution must be 'dense', 'stream', 'operator', or 'discrete'."`
2. `batch_rows`, `memory_budget_bytes` or `output_budget_bytes` not a positive `int`: `"{name} must be a positive integer."`
3. `"uncertainty must be 'none', 'fisher', or 'covariance'."`
4. `"linear_solver must be 'cholesky', 'qr', or 'pcg'."`
5. `"Unsupported smoothing_strategy."`
6. `smoothing_strategy="working_model_reml"` with another route: `"working_model_reml requires execution='discrete'."`
7. `"working_model_max_iterations must be a positive integer."`
8. `"working_model_epsilon must be positive and finite."`
9. `"gaussian_compression must be a boolean."`
10. `"operator must be an OperatorControl instance."` and `"efs must be an EFSControl instance."`
11. `TypeError`: `"discretization must be a ContinuousDiscretizationControl or None."`
12. `execution="discrete"` without a policy: `"execution='discrete' requires an explicit discretization policy."`
13. A policy with another route: `"A discretization policy requires execution='discrete'."`

Only rules 6, 12 and 13 combine fields. A solver that does not suit the
route, `uncertainty` on the operator route, or `gaussian_compression` on a
route that ignores it all construct without error and are handled in
`GAM.fit()`.

::: jaxgam.control.FitControl
    options:
      heading_level: 4

### EFSControl

`EFSControl` sets the limits of the Fellner-Schall optimizer. It is read only
when `optimizer="efs"` and `sp` is `None`, on the dense, streamed and
operator routes. The discretized route never reads it.

| Field | Default | Read by | What it does |
|---|---|---|---|
| `outer_limit` | `200` | dense, streamed, operator | Maximum outer iterations. A stop first met on the last allowed iteration is reported as an iteration limit, not as converged. |
| `log_lambda_max` | `15.0` | dense, streamed, operator | Cap on trial log smoothing parameters. |
| `score_tolerance` | `0.1` | dense, streamed, operator | Score-window stop: after iteration 3, the largest proposed log smoothing parameter step is below 0.05 and the last four scores change by less than this. |
| `pirls_tolerance` | `1e-7` | dense, streamed, operator | Inner fit tolerance on dense and streamed; on all three, the deviance-change stop `abs(change) < 100 * pirls_tolerance * abs(deviance)`. |
| `pirls_max_iter` | `200` | dense, streamed | Inner fit iteration limit. The operator route uses `OperatorControl` for its inner fit instead. |
| `history_limit` | `200` | dense, streamed, operator | Maximum length of the score history kept in `optimizer_diagnostics`. |

Checks at construction (messages have no trailing period):

- `outer_limit`, `pirls_max_iter`, `history_limit` not a positive `int`: `"EFS {name} must be a positive integer"`
- `log_lambda_max`, `score_tolerance`, `pirls_tolerance` not a real number: `"EFS {name} must be a real number"`; not finite: `"EFS {name} must be finite"`
- `"EFS score_tolerance must be nonnegative"`; `pirls_tolerance <= 0`: `"EFS pirls_tolerance must be positive"`

::: jaxgam.control.EFSControl
    options:
      heading_level: 4

### OperatorControl

`OperatorControl` sets the limits of the operator route's coefficient fit,
for fixed `sp` and for every refit inside the operator Fellner-Schall
optimizer.

| Field | Default | Read by | What it does |
|---|---|---|---|
| `max_modest_coefficients` | `64` | operator | Cap on the width of the small block: intercept, parametric terms and smooths other than `bs="re"`. A wider block raises `MemoryError("Modest operator width N exceeds hard cap M.")`. Random-effect levels are not counted. |
| `max_iterations` | `200` | operator | Maximum Newton iterations of the coefficient fit. |
| `tolerance` | `1e-7` | operator | Convergence: relative objective change below `tolerance` and a gradient norm within `tolerance` times the objective scale. |
| `cg_max_iterations` | `1000` | operator | Iteration limit of each conjugate-gradient solve. |
| `cg_rtol` | `1e-10` | operator | Relative residual tolerance of each conjugate-gradient solve. |
| `cg_atol` | `1e-12` | operator | Absolute residual tolerance of each conjugate-gradient solve. |
| `use_coarse` | `False` | operator | Adds a coarse correction to the preconditioner. |

Checks at construction:

- the three integer fields not a positive integer (any `numbers.Integral`): `"Operator {name} must be a positive integer."`
- `tolerance`, `cg_rtol`, `cg_atol` not finite and nonnegative: `"Operator {name} must be finite and nonnegative."`
- `"Operator tolerance must be strictly positive."`
- both CG tolerances zero: `"Operator CG requires a positive residual tolerance."`
- `use_coarse` not a Python `bool`: `"Operator use_coarse must be boolean."`

!!! warning
    `OperatorControl` accepts a NumPy integer for `max_modest_coefficients`,
    but the operator setup pass then raises
    `ValueError("max_modest_coefficients must be a positive integer.")`.
    Pass a Python `int`.

The operator Fellner-Schall optimizer attaches an error bound to a score only
when `tolerance <= 1e-7`, `cg_rtol <= 1e-10` and `cg_atol <= 1e-12`. The
defaults sit exactly on those limits, so loosening any of them leaves every
score comparison undecided. See [Operator route](#operator-route).

::: jaxgam.control.OperatorControl
    options:
      heading_level: 4

### ContinuousDiscretizationControl

`ContinuousDiscretizationControl` is the discretization policy of the
discretized route: how covariates are compressed to a grid, which columns
are factors, and how new data are predicted.

| Field | Default | Read by | What it does |
|---|---|---|---|
| `m` | `None` | discretized | Grid points per metric variable of a group. `None` gives 1000 for a group of one variable, 100 for two and 25 for three or more. An `int` is used as given for one variable and as `round(m ** (1 / d)) + 1` for `d` variables. A tuple gives one value per smooth term, in formula order; the response and parametric groups use its largest entry. |
| `factor_names` | `()` | discretized | Variables to treat as factors. Every factor (parametric factor, `by` factor, `bs="re"` grouping) must be listed: an unlisted pandas categorical raises `TypeError`, and an unlisted integer-coded grouping is treated as one numeric variable. |
| `budget` | `DiscretizationBudget()` | discretized | Storage limits of the setup pass. See [DiscretizationBudget](#discretizationbudget). |
| `selector_memmap_directory` | `None` | discretized | An existing directory. Row selectors are written there as `compact-selector-{k}.dat` instead of held in memory, and removed when the fit ends. A missing directory raises `FileNotFoundError`; an existing selector file raises `FileExistsError`. |
| `pair_policy` | `"histogram_or_direct"` | discretized | How two different lookup blocks are combined in the cross-product. `"histogram_or_direct"` uses a weight histogram when it is no larger than gathering rows and fits `max_pair_histogram_bytes`; `"direct_only"` always gathers rows. |
| `max_pair_histogram_bytes` | `32 * 1024 * 1024` | discretized | Largest pair histogram allowed. |
| `seed` | `8547` | discretized | Seed of the shuffle and padding draws that mgcv's discretization uses. Must fit a signed 32-bit integer, checked when the fit runs. |
| `prediction_policy` | `"continuous_frozen_basis"` | discretized prediction | `"continuous_frozen_basis"` evaluates each new value exactly. `"mgcv_predict_bamd"` first compresses the whole prediction request, as mgcv's `predict.bamd` does, so predictions depend on the request's composition but not on how it is batched. A model without smooths records `"continuous_frozen_basis"`. |
| `max_retained_representatives` | `10_000` | discretized | Groups with at most this many distinct rows keep their representative values in `fit.discretization`. |

Checks at construction:

- `"m tuple entries must be integers of at least 2."`; `"m must be an integer of at least 2, a tuple, or None."`
- `"factor_names must be a tuple of non-empty names."`; `"factor_names must not contain duplicates."`
- `TypeError`: `"budget must be a DiscretizationBudget."`
- `"pair_policy must be 'histogram_or_direct' or 'direct_only'."`
- `"max_pair_histogram_bytes must be a positive integer."`
- `"seed must be an integer."`
- `"prediction_policy must be 'continuous_frozen_basis' or 'mgcv_predict_bamd'."`
- `"max_retained_representatives must be a non-negative integer."`

Checked when the fit runs: `"seed must be a signed 32-bit integer"`, `"m
tuple must have one entry per smooth term"` and `"factor_names must be
distinct referenced variables"`.

::: jaxgam.control.ContinuousDiscretizationControl
    options:
      heading_level: 4

### DiscretizationBudget

`DiscretizationBudget` (from `jaxgam.data`, not the top-level package)
bounds what the discretized route's setup pass may store.

| Field | Default | Read by | What it does |
|---|---|---|---|
| `max_table_bytes` | `64 * 1024 * 1024` | discretized | Compact tables built and kept, summed over all groups. |
| `max_selector_bytes` | `256 * 1024 * 1024` | discretized | Row selectors held in memory, summed over all groups. A selector written to `selector_memmap_directory` is not charged, only its validation array. |
| `max_batch_bytes` | `64 * 1024 * 1024` | discretized | Workspace of one batch in one group. It grows with `FitControl.batch_rows`. |
| `max_unique_rows` | `1_000_000` | discretized | Distinct rows allowed per group. |

Each field must be a positive `int`, otherwise `ValueError("{name} must be a
positive integer")`. An exceeded budget raises `MemoryError("{reason}:
requires N bytes, budget is M.")` during the fit. Under
`prediction_policy="mgcv_predict_bamd"` the result keeps the budget, and it
also bounds the compression of each prediction request.

::: jaxgam.data.discretize.DiscretizationBudget
    options:
      heading_level: 4

---

## Row sources

A row source is an object that replays the training rows in batches, in the
same order every time. The streamed, discretized and operator routes read
their data only through one, so the full data set never has to be in one
array. The how-to, including writing your own source, is in
[Row sources](guide/row_sources.md).

All five names below come from `jaxgam.data`.

### RowSource

`RowSource` is a runtime-checkable protocol. An object satisfies it when it
has:

| Member | What it must do |
|---|---|
| `n_rows` | Property: the number of rows. |
| `scan(batch_rows)` | Return an iterator of `RowBatch` objects, each with at most `batch_rows` rows, in stable positional order. Every call replays the same rows. |
| `fingerprint()` | Return a string that changes when the data change. Fits compare fingerprints to detect data changed during the fit, and raise `RuntimeError`. |

A source may also declare `has_explicit_weights` and `has_explicit_offset`
(booleans). `False` says the batches hold default weights (ones) or a
default offset (zeros). An attribute the source does not declare counts as
supplied. On the discretized route this matters: like mgcv's `bam`, a
supplied weight or offset column changes the seeded padding of the
discretization, and so the fitted numbers.

`isinstance(obj, RowSource)` checks only that the three members exist.

::: jaxgam.data.source.RowSource
    options:
      heading_level: 4

### RowBatch

`RowBatch` is one batch yielded by `scan()`. It is a frozen dataclass.

| Field | What it holds |
|---|---|
| `columns` | Mapping from column name to the batch's values (NumPy array or pandas Series). |
| `y` | Response for the batch, or `None` when the source has no response. |
| `weight` | Prior weights, shape `(rows,)`. |
| `offset` | Offset, shape `(rows,)`. |
| `valid` | Boolean mask, shape `(rows,)`. Every fit and `predict_iter()` require all rows valid and raise `NotImplementedError` on a padded batch. |
| `row_positions` | Integer positions of the rows in the source. |

A column, `weight`, `offset`, `valid` or `y` whose length differs from
`row_positions` raises `ValueError` when the batch is built.

```python
for batch in source.scan(256):
    print(len(batch.row_positions), batch.y.shape, sorted(batch.columns))
```

```text
256 (256,) ['count', 'f', 'g', 'x', 'z']
256 (256,) ['count', 'f', 'g', 'x', 'z']
88 (88,) ['count', 'f', 'g', 'x', 'z']
```

::: jaxgam.data.source.RowBatch
    options:
      heading_level: 4

### ArrayRowSource

`ArrayRowSource(columns, *, y=None, weights=None, offset=None,
row_selection=None, version=None)` is a source over aligned
one-dimensional arrays, pandas Series or memmaps.

| Argument | What it does |
|---|---|
| `columns` | Mapping from column name to values. All columns must have the same length. |
| `y` | Response. Every fit requires one. |
| `weights` | Prior weights. Must be finite and non-negative, with a positive sum over the selected rows. `None` means ones, and `has_explicit_weights` is `False`. |
| `offset` | Offset. Must be finite. `None` means zeros, and `has_explicit_offset` is `False`. |
| `row_selection` | Integer positions to keep, in the order given. `None` keeps every row. |
| `version` | A string mixed into the fingerprint. The fingerprint always hashes the full content as well. |

`fingerprint()` hashes the schema and all values, so it costs one pass over
the data. The streamed and discretized routes compute it once when the fit
starts and once when it ends.

```python
from jaxgam.data import ArrayRowSource

array_source = ArrayRowSource(
    {"x": data["x"].to_numpy(), "z": data["z"].to_numpy()},
    y=data["y"].to_numpy(),
    weights=np.ones(n),
)
print(array_source.n_rows, array_source.has_explicit_weights, array_source.has_explicit_offset)
```

```text
600 True False
```

::: jaxgam.data.source.ArrayRowSource
    options:
      heading_level: 4
      members:
        - scan
        - fingerprint
        - has_explicit_weights
        - has_explicit_offset

### DataFrameRowSource

`DataFrameRowSource(data, *, response=None, **kwargs)` takes the columns of
a pandas DataFrame, keeps categorical and ordered dtypes, and uses the
column named by `response` as `y`. The remaining keyword arguments are those
of `ArrayRowSource`. A `response` that is not a column raises `ValueError`.
The setup block above builds `source` and `count_source` this way.

```python
weighted_source = DataFrameRowSource(data, response="y", weights=np.ones(n))
print(weighted_source.n_rows, weighted_source.has_explicit_weights)
```

```text
600 True
```

::: jaxgam.data.source.DataFrameRowSource
    options:
      heading_level: 4

### MemmapRowSource

`MemmapRowSource.from_npy_files(columns, *, y=None, weights=None,
offset=None, **kwargs)` opens one-dimensional `.npy` files with
`np.load(mmap_mode="r")`, so the data stay on disk and each scan reads only
its batch. `columns` maps column names to file paths; `y`, `weights` and
`offset` are paths. The remaining keyword arguments are those of
`ArrayRowSource`.

```python
import tempfile
from pathlib import Path

from jaxgam.data import MemmapRowSource

folder = Path(tempfile.mkdtemp())
for name in ("x", "z", "y"):
    np.save(folder / f"{name}.npy", data[name].to_numpy())
disk_source = MemmapRowSource.from_npy_files(
    {"x": folder / "x.npy", "z": folder / "z.npy"}, y=folder / "y.npy"
)
print(disk_source.n_rows)
```

```text
600
```

::: jaxgam.data.source.MemmapRowSource
    options:
      heading_level: 4
      members:
        - from_npy_files

---

## Fitting routes

A route is the way a fit reads its data and solves for the coefficients.
`FitControl.execution` selects it; `optimizer`, `sp`, `linear_solver` and
`gaussian_compression` select variants within a route. Every result reports
the variant that ran in `execution_route` and how smoothing parameters were
chosen in `lambda_strategy`.

| Route | `execution_route` values | `lambda_strategy` values |
|---|---|---|
| Dense | `"dense"`, `"gaussian_qr"` | `"fixed"`, `"newton_reml"`, `"efs_reml"`, `"lbfgs_reml"` |
| Streamed | `"stream"`, `"stream_qr"`, `"stream_exact_reml_qr"`, `"stream_efs_qr"`, `"stream_gaussian_qr"`, `"stream_parametric_qr"` | `"fixed"`, `"exact_reml"`, `"exact_reml_no_parameters"`, `"efs_reml"`, `"newton_reml"`, `"lbfgs_reml"` |
| Discretized | `"discrete"` | `"fixed"`, `"working_model_reml"` |
| Operator | `"operator"` | `"fixed"`, `"efs_reml"` |

On every route, check `converged` and `convergence_info` before using a fit.
A fit that stops early returns a result with `converged=False`; it does not
raise.

### Dense (default)

**Use it when** the data fit in memory as a data frame.
**Requires** a DataFrame or dict and `linear_solver="cholesky"` (the default).
**Returns** any of the three result types; `GAMResults` by default.

```python
fit = GAM("y ~ s(x) + s(z)").fit(data)
print(fit.converged, fit.n_iter, fit.convergence_info, fit.lambda_strategy)
```

```text
True 14 full convergence newton_reml
```

The setup pass builds the full model matrix and the penalty matrices in
memory. The fit then selects the smoothing parameters by Newton's method on
the REML criterion, refitting the coefficients by penalized iteratively
reweighted least squares (PIRLS) at every trial, as mgcv's `gam(method =
"REML")` does. For Gaussian and Gamma the log scale is also a Newton
coordinate, and for `NegativeBinomial(fixed=False)` the log theta. The
gradient and Hessian of the criterion come from automatic differentiation.

| Case | What happens |
|---|---|
| `sp=None` (default) | Smoothing parameters selected by REML. `lambda_strategy="newton_reml"`. |
| `sp=[...]` | Smoothing parameters held at the values given. `lambda_strategy="fixed"`. Poisson, Binomial and fixed-theta negative binomial run one PIRLS fit (`n_iter=0`, `convergence_info="fixed sp"`). Gaussian, Gamma and estimated-theta negative binomial still run Newton steps on the scale or theta. |
| No penalized terms | One PIRLS fit: `n_iter=0`, `converged=True`, `lambda_strategy="newton_reml"`. |
| `optimizer="efs"` | See [Fellner-Schall optimizer](#fellner-schall-optimizer). |
| `gaussian_compression=True` | See [Gaussian QR compression](#gaussian-qr-compression). |

Result fields to check: `converged`, `convergence_info` (`"full
convergence"`, `"step failed"`, `"iteration limit"` or `"non-finite REML
score or gradient"`, possibly followed by a note on the inner fit or a
jitter), `n_iter` (outer iterations), `lambda_strategy`, `execution_route`.

Limits:

- `linear_solver="qr"` raises `NotImplementedError("linear_solver='qr'
  currently requires execution='stream'.")`; `"pcg"` names
  `execution='operator'`.
- A row source raises `TypeError("A RowSource requires FitControl execution
  'operator', 'stream' or 'discrete'; the dense route accepts only a
  DataFrame or dict.")`.
- The dense fit is not memory-budgeted: the full model matrix is built
  whatever the result mode.
- When the final penalized Hessian is not numerically positive definite, the
  fit adds a small diagonal jitter, records it in `convergence_info` and
  issues a `RuntimeWarning`. This happens for models with unidentifiable
  terms; mgcv's dropping of unidentifiable columns is not implemented.
- A negative binomial model with estimated theta and no penalized terms
  keeps theta at its starting value, and `fit.theta` is `None`.

How the dense fit works internally is in
[The dense fit](architecture/fitting_dense.md).

### Fellner-Schall optimizer

**Use it when** you want mgcv's extended Fellner-Schall update
(`gam(optimizer = "efs")`) instead of Newton's method to select smoothing
parameters.
**Requires** `GAM(..., optimizer="efs")` with `sp=None`, a model with at
least one penalty, and a supported family. On the streamed route also
`linear_solver="qr"`.
**Returns** the route's usual result, with `lambda_strategy="efs_reml"` and
an `EFSOptimizerDiagnostics` record in `optimizer_diagnostics`.

```python
from jaxgam import EFSControl

fit = GAM(
    "count ~ s(x) + s(z)",
    family="poisson",
    optimizer="efs",
    control=FitControl(efs=EFSControl(history_limit=8)),
).fit(data)
print(fit.converged, fit.convergence_info, fit.lambda_strategy)
print(fit.optimizer_diagnostics.stop_reason, fit.optimizer_diagnostics.outer_iterations)
```

```text
True deviance_change efs_reml
deviance_change 8
```

The optimizer follows mgcv 1.9-3's `efsudr`. Each outer iteration refits the
coefficients at trial smoothing parameters and updates their logs by the
Fellner-Schall ratio. The streamed and operator routes run the same
optimizer over a row source; see [Streamed route](#streamed-route) and
[Operator route](#operator-route).

Supported families on the dense route:

| Scale | Families and links |
|---|---|
| Known | Poisson and Binomial with any link; negative binomial with log, identity or sqrt link, theta fixed or estimated |
| Estimated | Gaussian and Gamma with any link |

Anything else raises `NotImplementedError`. The scale is not an optimizer
coordinate: each fit is scored at the incoming scale and passes its
estimate to the next. For an estimated-scale family, `fit.score` is the
criterion at the incoming scale, while `fit.scale` is the final fit's
estimate.

Stop reasons in `convergence_info`: `"score_window"` and
`"deviance_change"` are converged; `"iteration limit reached"` (at
`EFSControl.outer_limit`) is not.

Limits:

- `optimizer="efs"` with a fixed `sp`, or a model with no penalties, runs
  the ordinary dense fit; `lambda_strategy` is then `"fixed"` or
  `"newton_reml"` and `optimizer_diagnostics` is `None`.
- Prior weights must lie in `[1e-10, 1e10]`, and the penalized system must
  be identifiable; otherwise `ValueError`.
- A fixed-theta negative binomial rejects responses strictly between 0 and
  1.
- An estimated-theta negative binomial with identity or sqrt link raises
  `ValueError` unless an offset makes every zero-coefficient mean positive
  (with no offset it always raises); mgcv rejects the same input.
- With `gaussian_compression=True`, Fellner-Schall runs on the ordinary
  dense fit and `execution_fallback_reason` is `"Gaussian QR compression
  requires Newton smoothing selection."`
- For a fixed-theta negative binomial, `fit.theta` is the fixed value under
  `optimizer="efs"` but `None` under the default optimizer.

`optimizer_diagnostics` holds, among others, `stop_reason`,
`outer_iterations`, `inner_iterations`, `theta_iterations`,
`accepted_score_history` (at most `history_limit` entries) and flags for
numerical events. On the streamed route it also counts the scans the
optimizer made (`startup_source_scans`, `provider_source_scans`). The
user guide page is [Smoothing parameter selection](guide/smoothing_selection.md).

::: jaxgam.fitting.state.EFSOptimizerDiagnostics
    options:
      heading_level: 4

### Gaussian QR compression

**Use it when** the model is Gaussian with identity link and you want each
smoothing trial to work on coefficient-sized arrays instead of the full
model matrix.
**Requires** `FitControl(gaussian_compression=True)` on the dense or
streamed route, the exact `Gaussian` family with identity link, Newton
selection or fixed `sp`, at least one penalty, an identifiable model, and
prior weights in `[1e-10, 1e10]`.
**Returns** the route's usual result with `execution_route="gaussian_qr"`
(dense) or `"stream_gaussian_qr"` (streamed).

```python
fit = GAM(
    "y ~ s(x) + s(z)",
    control=FitControl(gaussian_compression=True),
).fit(data)
print(fit.execution_route, fit.execution_fallback_reason, fit.lambda_strategy)
```

```text
gaussian_qr None newton_reml
```

The training rows are reduced once by a weighted QR factorization. The REML
criterion and its derivatives are then computed from the small triangular
factor, so the smoothing search never touches the rows again. On the dense
route the model matrix is still built first; on the streamed route the
reduction takes one pass over the source.

A request that is not eligible never raises on the dense route: the
ordinary fit runs and `execution_fallback_reason` says why.

```python
fit = GAM(
    "count ~ s(x)",
    family="poisson",
    control=FitControl(gaussian_compression=True),
).fit(data)
print(fit.execution_route, "|", fit.execution_fallback_reason)
```

```text
dense | Gaussian QR requires the exact Gaussian family with identity link
```

| Penalties | Optimizer | `lambda_strategy` |
|---|---|---|
| 50 or fewer | Newton's method, as on the dense route | `"newton_reml"` (`"fixed"` with `sp`) |
| more than 50 | SciPy's L-BFGS-B with exact gradients, log smoothing parameters bounded to `[-40, 40]`, at most 100 iterations and 500 evaluations | `"lbfgs_reml"` (`"fixed"` with `sp`) |

With fixed `sp`, the smoothing parameters stay fixed and the log scale is
still optimized.

Limits:

- Above 50 penalties L-BFGS-B can make slow progress when the smoothing
  parameters are badly scaled against each other. It may stop at its
  iteration limit with `converged=False` and `"iteration limit"` in
  `convergence_info`. Above 50 penalties it is tested only at 51 penalties.
  For models with many smooths, the [discretized route](#discretized-route)
  selects smoothing parameters by Newton proposals instead.
- On the streamed route a QR workspace above `memory_budget_bytes` raises
  `MemoryError("Known streamed Gaussian QR workspace requires ...")`; it
  does not fall back.
- On the streamed route an ineligible request with `sp=None` and
  `linear_solver="cholesky"` raises `NotImplementedError`, because the
  fallback (exact streamed REML) needs `linear_solver="qr"`.
- The discretized route ignores the request and sets
  `execution_fallback_reason` to `"Gaussian QR compression requires
  execution 'dense' or 'stream'."`. The operator route ignores it without
  recording a reason.

The user guide page is [Smoothing parameter selection](guide/smoothing_selection.md).

### Streamed route

**Use it when** the data are too large to hold as one model matrix, or are
on disk, and the model uses only univariate cubic regression smooths and
numeric parametric terms.
**Requires** `FitControl(execution="stream")`, a row source,
`result="prediction"`, and `linear_solver="qr"` for estimated smoothing
parameters, Gamma, the negative binomial or any non-default link.
**Returns** a `GAMPredictionResult`.

```python
fit = GAM(
    'y ~ z + s(x, bs="cr", k=10)',
    control=FitControl(execution="stream", linear_solver="qr", batch_rows=4096),
).fit(source, result="prediction")
print(fit.execution_route, fit.lambda_strategy, fit.converged)
print(fit.predict(newdata).round(3))
```

```text
stream_exact_reml_qr exact_reml True
[ 0.595  1.293  0.212 -0.743 -0.038]
```

The setup pass scans the source to find knots, build the bases and fix the
identifiability constraints, without building the model matrix. The fit then
reads the source batch by batch at each iteration and keeps only
coefficient-sized sums. `batch_rows` bounds the rows in memory at once;
`memory_budget_bytes` bounds the known arrays.

| `sp` | `optimizer` | `linear_solver` | Sub-route | `execution_route` | `lambda_strategy` |
|---|---|---|---|---|---|
| given | any | `"cholesky"` | Fixed smoothing, normal equations | `"stream"` | `"fixed"` |
| given | any | `"qr"` | Fixed smoothing, QR | `"stream_qr"` | `"fixed"` |
| `None` | `"newton"` | `"qr"` | Exact REML | `"stream_exact_reml_qr"` | `"exact_reml"`, or `"exact_reml_no_parameters"` when nothing is free |
| `None` | `"efs"` | `"qr"` | Fellner-Schall | `"stream_efs_qr"`, or `"stream_parametric_qr"` without penalties | `"efs_reml"`, or `"newton_reml"` without penalties |
| given or `None` (Newton) | `"newton"` with `gaussian_compression=True` | either | Gaussian QR | `"stream_gaussian_qr"` | `"newton_reml"`, `"lbfgs_reml"` or `"fixed"` |

`sp=[]` counts as given.

Families:

- With `linear_solver="cholesky"` and fixed `sp`: Gaussian identity,
  Poisson log and Binomial logit only. Any other family or link raises
  `NotImplementedError("Streamed observed-information fitting requires
  linear_solver='qr'.")` (for the negative binomial: `"Streamed NB
  observed-information fitting requires linear_solver='qr'."`).
- With `linear_solver="qr"`: Gaussian, Poisson, Binomial and Gamma with
  their links, and the negative binomial with log, identity or sqrt link,
  theta fixed or estimated. The data must lie in the family's response and
  starting-value domain; Poisson with logit, probit or cloglog link is
  rejected because its default start leaves the link's domain.
- Fellner-Schall: Gaussian, Binomial, Poisson, Gamma and the negative
  binomial (log, identity or sqrt).

Exact REML selects the log smoothing parameters, together with the log scale
for Gaussian and Gamma and the log theta for
`NegativeBinomial(fixed=False)`, by SciPy's L-BFGS-B with exact gradients.
It minimizes the same REML criterion as the dense route, but by a different
path, so the two agree at the optimum to numerical tolerance, not step by
step. Each L-BFGS-B run is limited to steps of 5 in every log parameter (a
window) and 100 iterations and 500 evaluations; a run that stops at a
window edge while the score still falls opens a new window, at most 20.
`exact_optimizer_diagnostics` reports the optimizer status, evaluation
counts, the final projected gradient and the accepted score history.

With fixed `sp`, an unknown-scale family (Gaussian with a non-identity link,
Gamma) still optimizes the log scale, and an estimated-theta negative
binomial still estimates theta; `lambda_strategy` stays `"fixed"`.

Formula limits (all `NotImplementedError`):

- Smooths must be univariate `s(x, bs="cr")` or `s(x, bs="cs")`. A default
  `s(x)` (thin plate), `bs="cc"`, `bs="re"`, `bs="gp"` and multivariate
  smooths raise `"Prepared setup currently supports only univariate cr/cs
  smooths."`
- `te()`, `ti()` and `by=` raise `"Prepared setup does not yet support
  by-variable or tensor smooths."`
- Parametric terms must be numeric: `"Prepared setup currently supports
  numeric parametric terms only."`
- Two smooths of the same variable: `"Prepared setup does not yet support
  overlapping cubic terms."`

Other limits:

- `linear_solver="pcg"` raises `ValueError("linear_solver='pcg' requires
  execution='operator'.")`.
- `sp=None` with `linear_solver="cholesky"` raises `NotImplementedError`
  (`"Exact streamed REML requires linear_solver='qr'."` or
  `"Streamed optimizer='efs' requires linear_solver='qr'."`), unless an
  eligible Gaussian QR request takes the fit.
- Memory checks are prospective estimates of known arrays, not an allocator
  cap. Too small a budget for the setup pass raises `MemoryError("Streamed
  setup requires at least N bytes to scan one row, ...")` before any full
  scan. The fit check charges the requested `batch_rows`, not the
  source's row count, and raises `MemoryError("Known streamed[ QR] PIRLS
  workspace requires ...")`; lower `batch_rows` for a model with a few
  hundred coefficients. A model with a cubic smooth needs a budget above
  about 16 MiB, the allowance for exact knot placement.
- Under exact REML, an inner coefficient fit that fails at any trial raises
  `RuntimeError` instead of returning `converged=False`.
- No warning is issued on non-convergence; check `converged`.

Result fields to check: `converged`, `convergence_info`,
`exact_optimizer_diagnostics` (exact REML), `optimizer_diagnostics`
(Fellner-Schall), `source_scans` and `batches_scanned` (scans made by the
fit, not by the setup pass), `known_workspace_bytes` (the planned fit
workspace; `None` on Fellner-Schall), and `score_scale` (the scale at which
`score` was evaluated). On the Fellner-Schall sub-route `source_scans`
counts the final accepted fit only; the optimizer's total is
`optimizer_diagnostics.startup_source_scans +
optimizer_diagnostics.provider_source_scans`.

The user guide page is [Streamed fitting](guide/streamed.md); internals are
in [The streamed fit](architecture/fitting_streamed.md).

### Discretized route

**Use it when** you want mgcv's `bam(discrete=TRUE)` fit: covariates
compressed to a grid, with any of the smooth types the dense route supports.
**Requires** `FitControl(execution="discrete",
discretization=ContinuousDiscretizationControl(...))`, a row source,
`result="prediction"`, and either a fixed `sp` or
`smoothing_strategy="working_model_reml"`.
**Returns** a `GAMPredictionResult` with `execution_route="discrete"`.

```python
from jaxgam import ContinuousDiscretizationControl

control = FitControl(
    execution="discrete",
    discretization=ContinuousDiscretizationControl(factor_names=("g",)),
    smoothing_strategy="working_model_reml",
    batch_rows=4096,
)
fit = GAM("count ~ x + s(g, bs='re')", family="poisson", control=control).fit(
    count_source, result="prediction"
)
print(fit.converged, fit.lambda_strategy, fit.source_scans)
print(fit.convergence_info)
```

```text
True working_model_reml 14
poisson discrete Fisher performance iteration (working-model REML) converged after 14 iterations, 28 proposals and 0 coefficient step halvings
```

The setup pass compresses each group of covariates (the variables of one
`s()` term, each `te()`/`ti()` margin, each `by` variable, the response and
each parametric variable) to its distinct values. A group is rounded to an
even grid of `m` points per variable only when it has more distinct rows
than that grid holds, so on small data nothing is rounded. Bases are built
on the compact rows, and the fit works on index tables instead of a full
model matrix.

Smoothing parameters are fixed with `sp=` or selected with
`smoothing_strategy="working_model_reml"`, mgcv's performance iteration
(`bgam.fitd`): each iteration forms a working model and takes one limited
Newton proposal (mgcv's `Sl.fitChol`) for the smoothing parameters. This is
a different method from the dense route's nested REML, so the selected
smoothing parameters can differ from a dense fit.

| Family | Links | Working model | Scale |
|---|---|---|---|
| Gaussian | identity | one frozen `X'WX` and `X'Wy` | estimated |
| Poisson | log, identity, sqrt | Fisher weights re-formed each iteration | 1 |
| Binomial | logit, probit, cloglog, log | Fisher weights re-formed each iteration | 1 |
| Gamma | inverse, identity, log | Fisher weights re-formed each iteration | estimated |
| Negative binomial | log, identity, sqrt | Newton weights re-formed each iteration; theta fixed or re-estimated each iteration by mgcv's `estimate.theta` | 1 |

Non-Gaussian fits halve a coefficient step while the penalized deviance
rises, from the third iteration. An estimated negative binomial theta is
updated once per iteration at the current mean; it is not optimized jointly
with the smoothing parameters, and every theta evaluation is a full scan.

Weights and offset come from the row source. As in `bam`, a supplied
weight or offset column changes the seeded padding of short grids and so
the centering constraints of the smooths. The built-in sources report
`has_explicit_weights=False` and `has_explicit_offset=False` when built
without them; a source that does not declare them counts as supplying both.

Limits:

- `linear_solver` other than `"cholesky"` raises `NotImplementedError`.
- `sp=None` with the default `smoothing_strategy` raises
  `NotImplementedError("Discrete fitting requires explicit fixed sp or
  opt-in smoothing_strategy='working_model_reml'.")`.
- `sp=None` with `optimizer="efs"` raises `NotImplementedError`; with a fixed
  `sp`, `optimizer` has no effect.
- Any family or link not in the table raises `NotImplementedError`.
- A multi-variable random effect such as `s(x, g, bs="re")` raises
  `NotImplementedError`.
- Smooths that share a variable in different groupings, such as
  `s(x, z) + s(x)`, raise `ValueError("continuous discretization has
  incompatible nesting")`.
- Factors are not inferred from the dtype; list them in `factor_names`.
- `m` smaller than a smooth's `k` fails with a `"Basis dimension k=...
  exceeds number of unique data values"` error, which counts compact rows.
- The default budgets do not admit every model. A factor with many levels
  can exceed `max_batch_bytes` at the default `batch_rows=65_536`; the
  example above lowers `batch_rows` to 4096 for that reason. On large data
  a single thin-plate (`tp`, `ts`) or Gaussian-process smooth can exceed
  `max_table_bytes`; lower `m`, set `max_knots`, or raise the budget.
- The fit's planned peak is checked against `memory_budget_bytes`:
  `MemoryError("Discrete fit workspace requires N bytes, exceeding
  FitControl.memory_budget_bytes=M.")`.
- A first proposal that leaves the family's domain raises
  `FloatingPointError("Non-finite working-model deviance at iteration 2.")`,
  where `bam` also stops.
- A rank-deficient model is not supported.

Result fields to check: `converged`, `convergence_info`, `n_iter`,
`smoothing_strategy`, `working_model_proposal_evaluations`, `source_scans`
and `batches_scanned` (fit scans only), `known_workspace_bytes`,
`fit_peak_bytes` and `fit_memory_budget_bytes`, `prediction_policy`,
`discretization` (the grids used) and, for the negative binomial, `theta`.
For a fixed-`sp` Gaussian fit, `n_iter` counts scale iterations and
`converged` is always `True`. Reaching `working_model_max_iterations` gives
a `RuntimeWarning` and `converged=False`.

The user guide page is [Discretized fitting](guide/discretized.md);
internals are in [The discretized fit](architecture/fitting_discrete.md).

### Operator route

**Use it when** a random effect has too many levels for a dense
coefficient system.
**Requires** `FitControl(execution="operator", linear_solver="pcg")`, a row
source, `result="prediction"`, fixed theta, CPU, and either a fixed `sp` or
`optimizer="efs"` inside the domain described below.
**Returns** a `GAMPredictionResult` with `execution_route="operator"`.

```python
fit = GAM(
    "count ~ x + s(g, bs='re')",
    family="poisson",
    sp=[1.0],
    control=FitControl(execution="operator", linear_solver="pcg"),
).fit(count_source, result="prediction")
print(fit.converged, fit.convergence_info)
print(fit.score, fit.edf_total, fit.scale)
```

```text
True Operator observed Newton fit converged.
None None 1.0
```

A `bs="re"` term is held as one level index and one amplitude per row,
never as dense columns, and its penalty as a scaled identity. The other
terms (the "modest" block) are small dense columns, capped by
`OperatorControl.max_modest_coefficients`. The coefficients are found by a
Newton iteration whose linear systems are solved by preconditioned
conjugate gradients (PCG), each matrix product being one scan of the
source. The fit uses the observed curvature of the family's deviance for
every family; a system that cannot be shown positive definite is refused,
not switched to Fisher scoring.

Formula scope: numeric parametric terms, `s()`, `te()` and `ti()` with
`cr`, `cs`, `cc`, `tp`, `ts` or `gp` bases (with or without `by`), and
`s(..., bs="re")`. Other bases, `by=` on a random effect and `te()`/`ti()`
with a random effect raise `NotImplementedError("Operator preparation
supports numeric terms, cr/cs/cc, tp/ts/gp and re.")`. Factor parametric
terms raise `NotImplementedError("Prepared setup currently supports numeric
parametric terms only.")`. A random effect's levels are a categorical's
declared categories (unused ones included), otherwise the sorted observed
values.

Fixed `sp` (as in the example):

- Supply positive finite `sp`, one value per penalty in formula order; a
  formula with no penalty needs `sp=[]`.
- `score`, `edf_total` and `Vp` are not computed (`None`), and `scale` is
  `None` for an unknown-scale family (1.0 for a known-scale family).
- Fixed-theta negative binomial is supported:
  `NegativeBinomial(theta, fixed=True)`. `family="nb"` (estimated theta)
  raises `NotImplementedError("Operator fitting currently requires fixed
  theta.")`.

Fellner-Schall (`optimizer="efs"`, `sp=None`) is limited to the domain in
which its score error bounds were measured:

- Poisson with log link, else `NotImplementedError("Operator EFS currently
  supports calibrated Poisson/log indexed fits.")`;
- exactly one factor-only random effect with 257 to 4096 levels, 1 to 16
  rows per level on average, `batch_rows` from 64 to 65536, and at most 4
  other coefficients, all unpenalized (intercept and numeric parametric
  terms);
- unit prior weights and a zero offset. Weights, an offset, a penalized
  smooth or a random slope raise `NotImplementedError` before the fit
  starts.

```python
from jaxgam import OperatorControl

fit = GAM(
    "count ~ x + s(g, bs='re')",
    family="poisson",
    optimizer="efs",
    control=FitControl(
        execution="operator",
        linear_solver="pcg",
        operator=OperatorControl(tolerance=1e-11, cg_rtol=1e-13, cg_atol=1e-15),
    ),
).fit(count_source, result="prediction")
print(fit.converged, fit.convergence_info, fit.score_bound_method)
```

```text
True score_window arrowhead_poisson_empirical_v2
```

Each Fellner-Schall decision compares two operator REML scores that carry
error bounds. A trial is accepted or rejected only when the score
difference exceeds the sum of the two bounds; otherwise the comparison is
undecided and the fit stops with `converged=False` and
`convergence_info="indeterminate_score"`, keeping the last accepted state.
A score gets a bound only when the responses are at most 32 and the
`OperatorControl` tolerances are at or below their defaults, among other
conditions. With the default `OperatorControl`, fits in this domain often
stop as `"indeterminate_score"`; the example above uses the tighter controls
that the comparison tests against mgcv use. Other stop reasons:
`"score_window"` and `"deviance_change"` (converged); `"iteration_limit"`,
`"inner_failure"` and `"invalid_update"` (not converged).
`operator_diagnostics.efs_indeterminate_comparisons` counts undecided
comparisons. The bounds are empirical, measured against a high-precision
reference over this domain; they are not guarantees.

Fellner-Schall results add `score`, `edf_total`, `score_error_bound` and
`score_bound_method` (`"indexed_poisson_empirical_v2"`,
`"arrowhead_poisson_empirical_v2"`, or `"unavailable"` with a `None` bound
when the accepted state had none).

Other limits:

- `linear_solver` other than `"pcg"` raises `ValueError("execution='operator'
  requires linear_solver='pcg'.")`.
- `uncertainty` other than `"none"` raises `NotImplementedError`, and
  `predict(se_fit=True)` raises `RuntimeError("Prediction standard errors are
  not available on the operator route. ...")`. Use `bulk_link_variance()`
  below.
- A memory check that fails raises `MemoryError` ending `"Lower batch_rows or
  raise the memory budget."`. The fit plans for `batch_rows` rows per batch,
  or for the source's row count when that is smaller.
- `sp=None` without `optimizer="efs"` raises `NotImplementedError("Operator
  fitting requires explicit fixed sp.")`.
- `device` other than `None` or `"cpu"` raises `NotImplementedError`.
- A system without a positive-definiteness certificate raises
  `LinAlgError` out of `fit()`; some positive-definite systems are refused.
  A PCG failure or the iteration limit returns `converged=False`.
- Every scan re-hashes every row of the source to prove the replay is
  identical.

Result fields to check: `converged`, `convergence_info`, `n_iter` (Newton
iterations for fixed `sp`, outer iterations for Fellner-Schall),
`source_scans` and `batches_scanned` (fit work; setup scans are in
`operator_diagnostics.preparation_started_scans`),
`known_workspace_bytes` (`None` for fixed `sp`), and `operator_diagnostics`
(Krylov iterations, solves, the final gradient norm). On Fellner-Schall
results `operator_diagnostics` describes the accepted refit only, while
`source_scans` covers the whole run.

#### Approximate variances with `bulk_link_variance`

A converged operator result can estimate link-scale prediction variances
without forming a coefficient covariance:

```python
fixed_fit = GAM(
    "count ~ x + s(g, bs='re')",
    family="poisson",
    sp=[1.0],
    control=FitControl(execution="operator", linear_solver="pcg"),
).fit(count_source, result="prediction")
estimator, calibration = fixed_fit.bulk_link_variance(
    count_source,
    data.iloc[:8],
    probes=32,
    seed=7,
    memory_budget_bytes=256 * 1024 * 1024,
)
print(round(calibration.maximum_absolute_deviate, 2))
for batch in estimator.estimate_iter(DataFrameRowSource(newdata)):
    print(batch.positions, batch.link_se.round(3))
```

```text
1.16
[0 1 2 3 4] [0.437 0.426 0.418 0.412 0.408]
```

`bulk_link_variance(training_source, calibration_data, *, probes, seed,
memory_budget_bytes, control=None, scale=None,
max_calibration_standard_errors=5.0)` rebuilds the Fisher system at the
fitted coefficients from the original training source (its fingerprint and
row count must match the fit), solves it for `probes` random right-hand
sides (at least 2), and keeps a probes-by-coefficients sketch.
`estimator.estimate_iter(source)` then yields batches with `positions`,
`variance`, `monte_carlo_se` and `link_se`, on the link scale only.

The rows of `calibration_data` (at least 1, at most `min(256, batch_rows)`)
are also solved exactly. Each estimate times `probes`, divided by the exact
variance, should follow a chi-square distribution with `probes` degrees of
freedom; the call converts that to an equivalent normal deviate
(`calibration.equivalent_deviates`) and raises `RuntimeError` when any
exceeds `max_calibration_standard_errors` (default 5, a two-sided level of
5.7e-7 per row). The Monte Carlo standard error describes sampling
variation only, not bias of the iterative solves, and calibration on the
selected rows is not a guarantee for other rows. An unknown-scale family
needs `scale=`. `control` is a `PCGControl` from `jaxgam.execution.operator`
(default relative tolerance 1e-10), independent of `FitControl.operator`.
The method raises `RuntimeError` on a result from another route or one that
did not converge.

The user guide page is [Operator fitting](guide/operator.md); internals are
in [The operator fit](architecture/fitting_operator.md).

---

## Fit result modes

`GAM.fit(..., result=...)` chooses what the returned object keeps.

| Mode | Return type | Routes | Keeps | Drops |
|---|---|---|---|---|
| `"full"` (default) | `GAMResults` | dense | Coefficients, `Vp`, fitted values, linear predictor, per-smooth EDF, the setup (model matrix, response, weights, offset) and training data | nothing |
| `"inference"` | `GAMInferenceResult` | dense | Coefficients, `Vp`, the fitted family, prediction transforms, per-smooth EDF, smooth labels and scalar diagnostics | model matrix, fitted values, training data, penalty caches; no `summary()` or `plot()` |
| `"prediction"` | `GAMPredictionResult` | dense, streamed, discretized, operator | Coefficients, the fitted family, prediction transforms, scalar diagnostics and, if requested by `FitControl.uncertainty`, a Fisher factor or `Vp` | everything above plus per-smooth EDF, `null_deviance` and, by default, any covariance |

The inference and prediction modes reduce memory kept after the fit. On the
dense route they do not reduce memory during the fit: the full model matrix
is built and used before the result is made.

### Which object should I keep?

| Object | Keep it when you need | What `to_predictor()` does |
|---|---|---|
| `GAMResults` | Prediction, `summary()`, `plot()`, per-smooth EDF, or in-sample values | Builds an independent `GAMPredictor`. The full result stays in memory until you discard it. |
| `GAMInferenceResult` | New-data prediction with standard errors, plus per-smooth EDF and fit diagnostics | Returns the predictor it already holds; no copy, no further memory reduction. |
| `GAMPredictionResult` | Prediction from any route, batched prediction with `predict_iter()`, budgeted output, or operator variances | Returns the predictor it already holds. |
| `GAMPredictor` | Only prediction, handed to another component | Not applicable: it is already the prediction core. |

You never need to call `to_predictor()` after an inference or prediction
fit; both predict directly. Call it when downstream code should receive a
prediction-only object.

---

## GAMResults

`GAMResults` is the full result of a dense fit, returned by default. It is
a frozen dataclass: its fields cannot be reassigned, although the arrays
inside are not themselves frozen. All attributes are listed under
[GAMResults attributes](#gamresults-attributes).

| Method | What it does |
|---|---|
| `predict(newdata=None, pred_type="response", se_fit=False, offset=None)` | Predictions on the response (`"response"`) or link (`"link"`) scale; any other `pred_type` raises `ValueError`. `newdata=None` predicts the training rows (and ignores `offset`). `se_fit=True` returns `(prediction, se)`; the response-scale standard error is the link-scale one times the absolute derivative of the inverse link. |
| `predict_matrix(newdata)` | The constrained prediction matrix (mgcv's `predict.gam(type = "lpmatrix")`): `predict_matrix(newdata) @ coefficients` is the linear predictor without offset. |
| `to_predictor()` | A new, independent `GAMPredictor`. |
| `summary()` | Prints and returns a summary: parametric coefficient tests, smooth term tests (Wood 2013), adjusted R-squared, deviance explained, scale and REML score. |
| `plot(select=None, pages=0, rug=True, se=True, shade=True, **kwargs)` | One panel per smooth term on the link scale; returns `(fig, axes)`. |

`newdata` is a DataFrame or dict holding every variable the formula uses,
of equal lengths. A model fitted with a nonzero offset warns when
`predict()` gets new data without `offset=`, and omits the offset.

At prediction time, a level not seen in training gives:

| Term | Unseen level | Missing value (`None` or `NaN`) |
|---|---|---|
| `s(g, bs="re")` | Zero random-effect contribution (the prior mean); no warning | Zero contribution as well (mgcv returns `NA`) |
| factor `by=` smooth | `NaN` and a `UserWarning` | `NaN` and a `UserWarning` |
| parametric factor | `ValueError` | `NaN` |

```python
fit = GAM("y ~ s(x) + s(z)").fit(data)
prediction, se = fit.predict(newdata, se_fit=True)
print(prediction.round(3), se.round(3))
```

```text
[ 0.591  1.293  0.216 -0.738 -0.035] [0.037 0.033 0.034 0.034 0.036]
```

::: jaxgam.results.GAMResults
    options:
      heading_level: 3
      members:
        - predict
        - predict_matrix
        - to_predictor
        - summary
        - plot

---

## GAMInferenceResult

`GAMInferenceResult` is returned by `fit(..., result="inference")` on the
dense route. It predicts on new data with standard errors and keeps the fit
diagnostics, but none of the training arrays. It has no `summary()` or
`plot()`, and `predict()` requires `newdata`.

It holds `coefficients`, `Vp`, `family`, `formula`, `smooth_info` and
`term_names`, plus `edf`, `edf1`, `edf_total`, `deviance`, `null_deviance`,
`score`, `scale`, `theta`, `smoothing_params`, `converged`, `n_iter`,
`convergence_info`, `method`, `lambda_strategy`, `execution_path`,
`execution_route`, `execution_fallback_reason`, `optimizer_diagnostics` and
`n`.

```python
lean = GAM("y ~ s(x) + s(z)").fit(data, result="inference")
prediction, se = lean.predict(newdata, se_fit=True)
print(type(lean).__name__, lean.converged, se.round(3))
```

```text
GAMInferenceResult True [0.037 0.033 0.034 0.034 0.036]
```

`lean.to_predictor()` returns the predictor `lean` already holds, without a
copy.

::: jaxgam.results.GAMInferenceResult
    options:
      heading_level: 3
      members:
        - predict
        - predict_matrix
        - to_predictor

---

## GAMPredictionResult

`GAMPredictionResult` is returned by `fit(..., result="prediction")`. It is
the only result of the streamed, discretized and operator routes, and is
also available on the dense route. It is a small, picklable object: it
keeps the coefficients, the fitted family, what prediction needs, and scalar
diagnostics, but no training rows, no per-smooth EDF and, by default, no
covariance. It has no `summary()` or `plot()`.

| Method | What it does |
|---|---|
| `predict(newdata, pred_type="response", se_fit=False, offset=None)` | Predicts one batch of new data. `se_fit=True` needs `FitControl(uncertainty="fisher")` or `"covariance"`, otherwise `RuntimeError`. An output larger than `output_budget_bytes`, or a workspace larger than `memory_budget_bytes`, raises `MemoryError` and suggests `predict_iter()`. |
| `predict_iter(source, batch_rows=None, *, pred_type="response", se_fit=False)` | Yields `(row_positions, prediction)` pairs, or `(row_positions, (prediction, se))` with `se_fit=True`, for a row source. `batch_rows=None` uses `FitControl.batch_rows`. The source's offsets are applied. It raises `RuntimeError` if the source changes during the replay. |
| `predict_matrix(newdata)` | The constrained prediction matrix for one batch, subject to the same budgets. |
| `to_predictor()` | Returns the `GAMPredictor` the result holds. |
| `bulk_link_variance(...)` | Approximate link-scale variances for an operator result. See [Approximate variances](#approximate-variances-with-bulk_link_variance). Raises `RuntimeError` on other routes. |
| `materialize_covariance(memory_budget_bytes)` | Builds `Vp` from a retained Fisher factor, within the given budget. Raises `RuntimeError` when no factor was kept (`uncertainty="none"`). |

Attributes worth reading:

| Attribute | What it holds |
|---|---|
| `coefficients`, `family`, `formula` | The fitted coefficients, the family with its fitted link and theta, and the formula. |
| `converged`, `n_iter`, `convergence_info` | Whether and how the fit stopped. |
| `execution_route`, `execution_path`, `lambda_strategy`, `execution_fallback_reason` | Which route and smoothing selection ran, and why a Gaussian QR request was not met. |
| `deviance`, `score`, `scale`, `score_scale`, `theta`, `smoothing_params`, `edf_total`, `n` | Fit summaries. `score`, `edf_total` and `scale` can be `None` on the operator route. |
| `source_scans`, `batches_scanned`, `known_workspace_bytes` | Work and planned memory of the fit, excluding the setup pass. `None` on the dense route. |
| `optimizer_diagnostics`, `exact_optimizer_diagnostics`, `operator_diagnostics` | Route-specific optimizer records, `None` where they do not apply. |
| `smoothing_strategy`, `discretization_strategy`, `prediction_policy`, `discretization`, `working_model_proposal_evaluations`, `fit_peak_bytes`, `fit_memory_budget_bytes` | Discretized route only. |
| `score_error_bound`, `score_bound_method` | Operator Fellner-Schall only. |

`Vp` is not an attribute of this class. With `uncertainty="covariance"` it
is on `fit.to_predictor().Vp`; with `"fisher"`, call
`materialize_covariance()`.

```python
fit = GAM(
    'y ~ z + s(x, bs="cr")',
    control=FitControl(execution="stream", linear_solver="qr", uncertainty="fisher"),
).fit(source, result="prediction")
for positions, (prediction, se) in fit.predict_iter(
    DataFrameRowSource(newdata), batch_rows=2, se_fit=True
):
    print(positions, prediction.round(3), se.round(3))
```

```text
[0 1] [0.595 1.293] [0.037 0.032]
[2 3] [ 0.212 -0.743] [0.032 0.032]
[4] [-0.038] [0.036]
```

Results round-trip through the standard library `pickle` and predict the
same afterwards.

::: jaxgam.results.GAMPredictionResult
    options:
      heading_level: 3
      members:
        - predict
        - predict_iter
        - predict_matrix
        - to_predictor
        - bulk_link_variance

---

## GAMPredictor

`GAMPredictor` is the prediction-only core. Inference and prediction
results hold one and return it from `to_predictor()`;
`GAMResults.to_predictor()` builds a new one. It is a frozen dataclass whose `coefficients`
and `Vp` are read-only copies; `Vp` is `None` when the fit kept no
covariance. It has `predict(newdata, pred_type="response", se_fit=False,
offset=None)`, `predict_matrix(newdata)`, `predict_iter(source, batch_rows,
*, pred_type="response", se_fit=False)` (here `batch_rows` is required),
`freeze_request(newdata)` and the `prediction_policy` property (the
discretized prediction policy, `None` on other routes). It holds no
training arrays.

```python
import pickle

full = GAM("y ~ s(x) + s(z)").fit(data)
predictor = full.to_predictor()  # a new, independent core
restored = pickle.loads(pickle.dumps(predictor))
print(np.allclose(restored.predict(newdata), full.predict(newdata)))
```

```text
True
```

`full.to_predictor()` does not change `full`; discard the full result if you
no longer need its training state.

Use the standard library `pickle` for a same-version, short-lived handoff.
A predictor that uses a locally defined custom link or family needs
`cloudpickle`. A predictor records the jaxgam version that pickled it and
warns (`UserWarning`) when loaded under another version; pickles are not a
cross-version format, and only trusted pickles should be loaded. jaxgam has
no `save()`/`load()` API or file format.

::: jaxgam.inference.predictor.GAMPredictor
    options:
      heading_level: 3
      members:
        - predict
        - predict_matrix
        - predict_iter

---

## Families

A family sets the response distribution and the link. Pass a name to `GAM`
for the default link, or an instance for any other link:

```python
from jaxgam.families import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson

log_gamma = Gamma(link="log")
print(type(log_gamma.link).__name__, type(Poisson().link).__name__)
```

```text
LogLink LogLink
```

| Family | Name for `GAM` | Default link | Links mgcv offers for it | Scale |
|---|---|---|---|---|
| `Gaussian` | `"gaussian"` | identity | identity, log, inverse | estimated |
| `Binomial` | `"binomial"` | logit | logit, probit, cloglog, log | 1 |
| `Poisson` | `"poisson"` | log | log, identity, sqrt | 1 |
| `Gamma` | `"gamma"` | inverse | inverse, identity, log | estimated |
| `NegativeBinomial` | `"nb"` | log | log, identity, sqrt | 1 |

Each constructor takes `link=None` (the default link), a link name or a
`Link` instance; another type raises `TypeError` and an unknown name raises
`ValueError("Unknown link function ...")`. The registered link names are
`identity`, `log`, `logit`, `inverse`, `probit`, `cloglog`, `sqrt` and
`inverse_squared`. The constructor does not check that the link suits the
family. Use the links in the table: those are the combinations the routes
are tested with, and each route section lists which of them it supports.
The response must lie in the family's support (for example, in `[0, 1]`
for Binomial), checked when the fit starts.

### Gaussian

::: jaxgam.families.standard.Gaussian
    options:
      heading_level: 4
      members:
        - __init__
        - variance
        - deviance_resids
        - initialize

### Binomial

::: jaxgam.families.standard.Binomial
    options:
      heading_level: 4
      members:
        - __init__
        - variance
        - deviance_resids
        - initialize

### Poisson

::: jaxgam.families.standard.Poisson
    options:
      heading_level: 4
      members:
        - __init__
        - variance
        - deviance_resids
        - initialize

### Gamma

::: jaxgam.families.standard.Gamma
    options:
      heading_level: 4
      members:
        - __init__
        - variance
        - deviance_resids
        - initialize

### Negative binomial

`NegativeBinomial(theta=1.0, *, fixed=False, link=None)` models
overdispersed counts with variance `mu + mu**2 / theta` (theta is mgcv's
`size`; it must be positive). With `fixed=False`, theta is estimated
starting from `theta`; with `fixed=True` it is held at `theta`.
`family="nb"` is `NegativeBinomial()`.

```python
estimated = GAM("count ~ s(x)", family="nb").fit(data)
held = GAM("count ~ s(x)", family=NegativeBinomial(theta=2, fixed=True)).fit(data)
print(round(estimated.theta, 2), held.theta)
```

```text
3.88 None
```

Where the estimate is reported:

- `fit.theta` is the estimated theta when theta is estimated.
- For a fixed theta it is `None` on the dense and streamed routes with the
  default optimizer, and the fixed value under `optimizer="efs"` and on the
  discretized and operator routes. The fitted family always holds it:
  `fit.family.get_theta(transformed=True)`.
- The family passed to `GAM` is copied before an estimated-theta fit, so
  your instance is not changed.

::: jaxgam.families.negative_binomial.NegativeBinomial
    options:
      heading_level: 4
      members:
        - __init__
        - variance
        - deviance_resids
        - initialize
        - get_theta
        - put_theta

### ExtendedFamily base class

`ExtendedFamily` is the base class for families with extra distribution
parameters estimated during the fit. `NegativeBinomial` is the only one
built in.

::: jaxgam.families.extended.ExtendedFamily
    options:
      heading_level: 4

---

## Formula syntax

Formulas follow R's notation for `mgcv::gam`:

```python
formulas = [
    "y ~ s(x)",                     # one smooth (thin plate)
    "y ~ s(x) + z",                 # a smooth plus a numeric parametric term
    "y ~ te(x, z, k=5)",            # tensor product
    "y ~ s(x) + s(z) + ti(x, z)",   # main effects plus tensor interaction
    "y ~ s(x, z, bs='gp', k=30)",   # Gaussian process smooth
    "y ~ s(x, by=f) + f",           # factor-by smooth
    "y ~ s(x) + s(g, bs='re')",     # random intercept
    "y ~ 0 + s(x)",                 # no intercept
]
for formula in formulas:
    print(formula, GAM(formula).fit(data).converged)
```

```text
y ~ s(x) True
y ~ s(x) + z True
y ~ te(x, z, k=5) True
y ~ s(x) + s(z) + ti(x, z) True
y ~ s(x, z, bs='gp', k=30) True
y ~ s(x, by=f) + f True
y ~ s(x) + s(g, bs='re') True
y ~ 0 + s(x) True
```

What the parser accepts:

- Calls `s()`, `te()` and `ti()` only. Terms join with `+`. `0 +`, `- 1`
  or `-1` drops the intercept. A bare name is a parametric term: a numeric
  column enters as is, a factor column (pandas categorical or string) with
  treatment contrasts, an ordered factor with polynomial contrasts.
- Positional arguments of a smooth must be variable names.
- `k` must be an integer literal and `bs` a string literal, so per-margin
  lists such as `te(x, z, k=[5, 6])` raise `ValueError`.

What it rejects:

| Input | Error |
|---|---|
| `x1*x2` | `ValueError("Unsupported operator in formula: Mult. Only '+' and '-' are supported. ...")` |
| `x1:x2` | `ValueError("Cannot parse formula RHS as an expression: ...")` |
| `I(x)`, `offset(o)`, `factor(g)`, `t2(x, z)` | `ValueError("Unknown function 'I()' in formula. Supported: ['s', 'te', 'ti']")` |
| `s(log(x))` | `ValueError("Positional arguments to s() must be variable names. ...")` |
| `s(x, fx=True)` | `NotImplementedError` (fixed-df unpenalized splines are not supported) |
| `s(x, sp=1)` | `NotImplementedError`; use `GAM(sp=[...])` instead |
| `s(x, id=1)` | `NotImplementedError` (linked smoothing parameters are not supported) |
| `s(x, bs="ps")` and any other unregistered basis | `KeyError("Unknown basis type: 'ps'. Available: ...")` when the fit runs |

Other keyword arguments inside a smooth are evaluated as literals and
passed to the smooth; keywords that no smooth reads (for example `pc=`) are
silently ignored. Use the offset argument of `fit()` (dense route) or of the
row source for an offset.

### Smooth term arguments

| Argument | Applies to | Default | What it does |
|---|---|---|---|
| `k` | all | `-1` (automatic) | Basis dimension. Automatic: 10 for one-variable `tp` and cubic smooths, 30 for two-variable `tp`; for `gp` see below. Ignored by `re`. |
| `bs` | `s()`, and the margins of `te()`/`ti()` | `"tp"` for `s()`, `"cr"` for tensor margins | Basis type. See the catalog below. |
| `by` | all | `None` | A factor (pandas categorical or string column) gives one smooth and one smoothing parameter per level; an ordered factor drops its first level. A numeric column multiplies the smooth (a varying-coefficient term). |
| `m` | `tp`, `ts` (also tensor margins) | basis default | Penalty order. |
| `xt` | `tp`, `ts`, `gp` | `{"max_knots": 2000, "seed": 1}` | Knot subsampling when there are more distinct covariate values than `max_knots`. `tp` and `ts` also accept `max_knots=` and `seed=` directly. |

### Smooth catalog

| Formula | Basis |
|---|---|
| `s(x)`, `s(x, bs='tp')` | Thin-plate regression spline (default); any number of variables |
| `s(x, bs='ts')` | Thin-plate spline with a shrinkage penalty on the null space |
| `s(x, bs='cr')` | Cubic regression spline; one variable |
| `s(x, bs='cs')` | Cubic regression spline with shrinkage |
| `s(x, bs='cc')` | Cyclic cubic regression spline |
| `s(x, z, bs='gp')` | Low-rank Gaussian process smooth; one to three variables, more with an explicit `k` |
| `s(g, bs='re')` | Random effect: one coefficient per level, identity penalty; `s(x, g, bs='re')` is a random slope |
| `te(x, z)` | Tensor product of one-variable margins, one penalty per margin |
| `ti(x, z)` | Tensor interaction, without the margins' main effects |

`te` and `ti` are chosen by the function name; `s(x, bs="te")` is not
valid. A `bs="re"` grouping variable must be a pandas categorical or string
column: an integer column gives a single numeric coefficient.

### Gaussian process smooths

`bs='gp'` builds a low-rank Gaussian process (kriging) smooth: the leading
eigenvectors of a correlation matrix evaluated at knots, as in mgcv's
`smooth.construct.gp.smooth.spec`. With `k=-1` the basis dimension is 12
for one variable, 33 for two and 104 for three; four or more variables need
an explicit `k`.

| Argument | Default | What it does |
|---|---|---|
| `kernel` | `"matern_3_2"` | `"spherical"`, `"power_exponential"`, `"matern_3_2"`, `"matern_5_2"` or `"matern_7_2"` (case-insensitive) |
| `rho` | automatic | Positive range parameter; by default the largest distance between knots |
| `power` | `1.0` | Power of `kernel='power_exponential'`, in `(0, 2]` |
| `stationary` | `False` | `True` drops the linear trend from the null space |
| `xt` | `{"max_knots": 2000, "seed": 1}` | Knot subsampling |

mgcv encodes these options in one signed numeric vector `m`; jaxgam uses
named arguments instead, and `m=` on a `gp` smooth raises `ValueError`.
Gaussian process options given to `te()` or `ti()` are not passed to the
margins.

```python
fit = GAM("y ~ s(x, bs='gp', kernel='power_exponential', rho=0.5, power=2.0)").fit(data)
print(fit.converged)
```

```text
True
```

### Tensor product arguments

| Argument | Default | What it does |
|---|---|---|
| `k` | `-1` (automatic) | One integer applied to every margin; automatic is 5 per margin, as in mgcv's `te()` |
| `bs` | `"cr"` | One basis type for every margin |

---

## Custom registrations

Smooth types, families and links are looked up in registries. You can add
entries at run time; built-in entries cannot be replaced or removed.

| Registry | Import | Used by |
|---|---|---|
| `smooth_registry` | `from jaxgam.smooths import smooth_registry` | `bs=` in formulas |
| `family_registry` | `from jaxgam.families import family_registry` | `GAM(family="name")` |
| `link_registry` | `from jaxgam.links import link_registry` | `Family(link="name")` |

```python
from jaxgam.families import family_registry
from jaxgam.links import LogLink, link_registry
from jaxgam.smooths import smooth_registry
from jaxgam.smooths.cubic import CubicRegressionSmooth

print(smooth_registry.available, len(smooth_registry), "tp" in smooth_registry)


class MyCubic(CubicRegressionSmooth):
    """A cubic regression spline registered under a new name."""


class MyPoisson(Poisson):
    family_name = "mypoisson"


class MyLog(LogLink):
    pass


smooth_registry.register("mycr", MyCubic)
family_registry.register("mypoisson", MyPoisson)
link_registry.register("mylog", MyLog)

fit = GAM("count ~ s(x, bs='mycr')", family="mypoisson").fit(data)
fit_link = GAM("count ~ s(x)", family=Poisson(link="mylog")).fit(data)
print(fit.converged, fit_link.converged)
```

```text
('cc', 'cr', 'cs', 'gp', 're', 'te', 'ti', 'tp', 'ts') 9 True
True True
```

What a new class must provide:

- A smooth subclasses `jaxgam.smooths.Smooth`, takes a `SmoothSpec` in its
  constructor, and implements `setup`, `build_design_matrix`,
  `build_penalty_matrices` and `predict_matrix`. `setup` must leave the
  object marked as set up and keep the training basis, or model assembly
  raises `RuntimeError`.
- A family subclasses `jaxgam.families.ExponentialFamily`, sets
  `family_name`, and implements `default_link`, `variance`, `dvar`,
  `saturated_loglik`, `deviance_resids`, `aic`, `_initialize_impl`,
  `valid_mu` and `valid_eta`. Looking a family up by name calls the class
  with no arguments and caches the instance.
- A link subclasses `jaxgam.links.Link` and implements `link`, `inverse`
  and `derivative`.

Rules:

- Keys are case-insensitive: `"MYCR"` and `"mycr"` are the same key.
- Registering an existing key, built in or custom, raises
  `ValueError("Key ... is already registered in the ... registry.")`. There
  is no way to unregister.
- `register()` does not check the class type.
- Registrations are global to the process and apply to every later `GAM`.
- Custom classes are exercised on the dense route only.

---

## GAMResults attributes

`GAMResults` is a frozen dataclass. Its fields cannot be reassigned,
although the arrays inside are not themselves frozen.

| Attribute | Type | What it holds |
|---|---|---|
| `coefficients` | `ndarray (p,)` | Fitted coefficients |
| `fitted_values` | `ndarray (n,)` | Fitted values on the response scale |
| `linear_predictor` | `ndarray (n,)` | Linear predictor, including any offset |
| `Vp` | `ndarray (p, p)` | Bayesian posterior covariance of the coefficients |
| `edf` | `ndarray` | Effective degrees of freedom per smooth (one entry per level of a factor-by smooth) |
| `edf1` | `ndarray` | Alternative per-smooth EDF used for significance tests |
| `edf_total` | `float` | Total effective degrees of freedom |
| `scale` | `float` | Scale (dispersion) parameter: estimated for Gaussian and Gamma, 1 otherwise |
| `deviance` | `float` | Model deviance |
| `null_deviance` | `float` | Deviance of the null model |
| `score` | `float` | REML criterion at the returned smoothing parameters |
| `smoothing_params` | `ndarray` | Smoothing parameters: selected, or the `sp` values you supplied |
| `theta` | `float \| None` | Negative binomial theta when estimated (see [Negative binomial](#negative-binomial)); `None` for other families |
| `converged` | `bool` | Whether the smoothing parameter search converged |
| `n_iter` | `int` | Outer iterations (Newton or Fellner-Schall) |
| `convergence_info` | `str` | How the fit stopped |
| `method` | `str` | `"REML"` |
| `lambda_strategy` | `str` | How smoothing parameters were chosen: `"newton_reml"`, `"fixed"`, `"efs_reml"` or `"lbfgs_reml"` |
| `execution_path` | `str` | `"jax"` |
| `execution_route` | `str` | `"dense"` or `"gaussian_qr"` |
| `execution_fallback_reason` | `str \| None` | Why a Gaussian QR request was not met, else `None` |
| `optimizer_diagnostics` | `EFSOptimizerDiagnostics \| None` | Fellner-Schall record, `None` for other fits |
| `n` | `int` | Number of observations |
| `family` | family instance | The family used, with its fitted link and theta |
| `formula` | `str` | The model formula |
| `setup` | `ModelSetup` | The setup pass output: model matrix, penalties and constraints |
| `training_data` | `dict` | Training columns, kept for plotting |
| `X` | `ndarray (n, p)` | Model matrix (from `setup`) |
| `y` | `ndarray (n,)` | Response (from `setup`) |
| `weights` | `ndarray (n,)` | Prior weights (from `setup`) |
| `offset` | `ndarray \| None` | Training offset (from `setup`) |
| `coef_map` | `CoefficientMap` | Map between constrained and original basis coefficients (from `setup`) |
| `smooth_info` | `tuple` | Per-smooth labels and coefficient ranges (from `setup`) |
| `term_names` | `tuple[str, ...]` | Model matrix column names (from `setup`) |
