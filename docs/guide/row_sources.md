# Row sources

A row source hands a fit your data in batches of rows, and can replay them
from the start as often as the fit asks. The streamed, discretized and
operator routes read their data only through a row source, so the data never
have to sit in memory as one table.

**Use a row source when** you fit with `FitControl(execution="stream")`,
`"discrete"` or `"operator"`, or when you predict on new data with
`predict_iter()` (see [Prediction results](prediction_results.md)).

**Do not use one for** the default dense fit: it takes a DataFrame or dict
and rejects a row source with `TypeError`.

Formally, a row source is any object that satisfies the protocol
`jaxgam.data.RowSource`:

| Member | What it must do |
|---|---|
| `n_rows` | Property: the number of rows the source yields in one scan. |
| `scan(batch_rows)` | Yield `RowBatch` objects of at most `batch_rows` rows each, covering every row once, in the same order on every call. |
| `fingerprint()` | Return a string that stays the same while the data stay the same, and changes when they change. |
| `has_explicit_weights`, `has_explicit_offset` | Optional booleans: `False` says the batches carry default weights (ones) or offset (zeros). |

jaxgam ships three row sources in `jaxgam.data`. They are not exported from
the top-level `jaxgam` package. All three accept `weights=`, `offset=` and
`row_selection=`, and a response (`response=` names a column of a
DataFrame, `y=` gives it directly to the other two); see [Row sources](../api.md#row-sources) in the API
reference for every argument.

All examples on this page use this setup block:

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data import (
    ArrayRowSource,
    DataFrameRowSource,
    MemmapRowSource,
    RowBatch,
    RowSource,
)

rng = np.random.default_rng(1)
n = 400
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
y = np.sin(2 * np.pi * x) + 0.5 * z + rng.normal(0, 0.3, n)
data = pd.DataFrame({"x": x, "z": z, "y": y})

control = FitControl(execution="stream", linear_solver="qr")
model = GAM("y ~ s(x, bs='cr') + z", control=control)
```

## `DataFrameRowSource`: a DataFrame you already have

`DataFrameRowSource(data, response="y")` wraps a DataFrame. The column named
by `response` becomes the response; every other column is a covariate.
Categorical columns keep their categories and ordering.

```python
source = DataFrameRowSource(data, response="y")
fit = model.fit(source, result="prediction")
print(source.n_rows, fit.converged, fit.execution_route)
```

```text
400 True stream_exact_reml_qr
```

A fit needs the response: a source made without `response=` raises
`ValueError: Prepared fitting requires a response in the RowSource.` on the
streamed route. For `predict_iter()` the response is not needed.

## `ArrayRowSource`: aligned arrays

`ArrayRowSource(columns, y=..., weights=..., offset=...)` takes a mapping of
column names to one-dimensional arrays (or pandas Series, or memory maps) of
equal length.

```python
weights = rng.uniform(0.5, 2.0, n)
source = ArrayRowSource({"x": x, "z": z}, y=y, weights=weights)
fit = model.fit(source, result="prediction")
print(source.has_explicit_weights, source.has_explicit_offset, fit.converged)
```

```text
True False True
```

## `MemmapRowSource.from_npy_files`: data on disk

`MemmapRowSource.from_npy_files` opens one `.npy` file per column as a
read-only NumPy memory map, so a scan reads only the rows of the current
batch. Write each column with `np.save` as a one-dimensional array.

```python
directory = Path(tempfile.mkdtemp())
for name, values in {"x": x, "z": z, "y": y}.items():
    np.save(directory / f"{name}.npy", values)

source = MemmapRowSource.from_npy_files(
    {"x": directory / "x.npy", "z": directory / "z.npy"},
    y=directory / "y.npy",
)
fit = model.fit(source, result="prediction")
print(source.n_rows, fit.converged)
```

```text
400 True
```

`weights=` and `offset=` take file paths here too.

## Weights and offset belong to the source

On a row-source route, prior weights and the offset come from the source.
Passing them to `fit()` raises, for example on the streamed route:

```text
ValueError: weights and offset must be owned by the RowSource for execution='stream'; do not pass separate arrays.
```

The built-in sources check them when you construct the source: weights must
be finite and non-negative with a positive sum over the selected rows, and
the offset must be finite.

## `row_selection`: fit a subset of rows

`row_selection=` takes integer positions into the columns. The source then
yields only those rows, in the order given, and `n_rows` is their count.

```python
even_rows = ArrayRowSource({"x": x, "z": z}, y=y, row_selection=np.arange(0, n, 2))
fit = model.fit(even_rows, result="prediction")
print(even_rows.n_rows, fit.n)
```

```text
200 200
```

An empty selection, or a position outside the columns, raises `ValueError`.

## What a fit requires of a source

A fit scans the source several times: the setup pass reads it to place knots
and build constraints, and the fit reads it again for every iteration. Every
scan must therefore:

- yield **the same rows in the same order**, whatever `batch_rows` is (the
  fit asks for different batch sizes; the streamed route, for example, first
  calls `scan(1)` twice to check that two scans start with the same rows);
- yield batches of **at most `batch_rows`** rows, with every row marked
  valid (padded batches are not supported);
- give each row the **same `row_positions`** on every scan;
- keep the **same columns** and, for categorical columns, the same
  categories in every batch.

The fit compares `fingerprint()` before and after its scans. If the source
changed, it raises `RuntimeError`, for example
`RuntimeError: RowSource changed during the fit` or
`RuntimeError: RowSource changed after preparation; prepare again.` The
built-in sources hash their whole content. On the streamed and discretized
routes that hash is computed once when the fit starts and once more when it
ends, so a change made during the fit raises at the end.

## Writing your own source

Write your own source when your data live somewhere the built-in sources do
not read: CSV or Parquet files, a database, or files spread over a directory.
The class below reads a CSV file in chunks with pandas, so only one batch is
in memory at a time. It treats the columns listed in `factor_levels` as
factors with fixed categories.

```python
class CsvRowSource:
    """Replay a CSV file in batches; categorical columns get fixed levels."""

    has_explicit_weights = False  # batches carry weight 1 for every row
    has_explicit_offset = False  # batches carry offset 0 for every row

    def __init__(self, path, response, factor_levels=None):
        self.path = Path(path)
        self.response = response
        self.factor_levels = factor_levels or {}
        self._n_rows = sum(len(chunk) for chunk in pd.read_csv(self.path, chunksize=65_536))

    @property
    def n_rows(self):
        return self._n_rows

    def fingerprint(self):
        # Called many times per fit: keep it cheap, but change it when the file changes.
        status = self.path.stat()
        return f"{self.path.resolve()}|{status.st_size}|{status.st_mtime_ns}"

    def scan(self, batch_rows):
        start = 0
        for chunk in pd.read_csv(self.path, chunksize=batch_rows):
            rows = len(chunk)
            columns = {}
            for name in chunk.columns:
                if name == self.response:
                    continue
                if name in self.factor_levels:
                    columns[name] = pd.Series(
                        pd.Categorical(chunk[name], categories=self.factor_levels[name])
                    )
                else:
                    columns[name] = chunk[name].to_numpy()
            yield RowBatch(
                columns=columns,
                y=chunk[self.response].to_numpy(dtype=float),
                weight=np.ones(rows),
                offset=np.zeros(rows),
                valid=np.ones(rows, dtype=bool),
                row_positions=np.arange(start, start + rows, dtype=np.intp),
            )
            start += rows
```

Write a CSV file, check the class against the protocol and fit with it:

```python
csv_path = directory / "data.csv"
data.to_csv(csv_path, index=False)

source = CsvRowSource(csv_path, response="y")
print(isinstance(source, RowSource))
fit = model.fit(source, result="prediction")
print(fit.converged, fit.execution_route)
```

```text
True
True stream_exact_reml_qr
```

The streamed route takes no factor terms. The discretized route does, so the
same class with a grouping column can fit a random effect there. That route
also needs the factor named in `factor_names` (see
[Discretized fitting](discretized.md)):

```python
group = rng.integers(0, 20, n)
grouped = data.assign(g=group, y=data["y"] + rng.normal(0, 0.5, 20)[group])
grouped_path = directory / "grouped.csv"
grouped.to_csv(grouped_path, index=False)

source = CsvRowSource(grouped_path, response="y", factor_levels={"g": list(range(20))})
discrete = FitControl(
    execution="discrete",
    discretization=ContinuousDiscretizationControl(factor_names=("g",)),
    smoothing_strategy="working_model_reml",
)
fit = GAM("y ~ s(x, bs='cr') + z + s(g, bs='re')", control=discrete).fit(
    source, result="prediction"
)
print(fit.converged, fit.execution_route, len(fit.smoothing_params))
```

```text
True discrete 2
```

What a custom source must get right:

- **A cheap `fingerprint()`.** The fit calls it many times (hundreds of times
  for a small streamed fit), and the reuse of one hash per fit
  described above applies only to the built-in sources. Derive it from metadata such as a
  path, size and modification time, or a version string you control, and
  make sure it changes when the data change.
- **Batches of at most `batch_rows` rows.** A larger batch raises, on the
  streamed route `ValueError: RowSource yielded N rows for a setup scan
  requesting at most M.` (with the actual counts).
- **The same rows in the same order on every scan**, with stable
  `row_positions` (here the row number in the file).
- **The same factor levels in every batch.** Give a factor column as a
  pandas Categorical Series with a fixed `categories=` list. Levels that
  depend on which rows happen to be in a batch raise, for example
  `RuntimeError: Factor column 'g' changed levels during scan` on the
  discretized route.
- **`has_explicit_weights` and `has_explicit_offset`.** Declare them `False`
  when your batches carry the defaults, as above. If you leave them out,
  jaxgam treats the source as supplying both, as mgcv's `bam()` treats
  `weights=` and `offset=` arguments; on the discretized route they are then
  carried through the compression as extra columns, and `predict_iter()`
  applies the batch offset without warning.
- **A response for fitting.** Set `y` on every batch; it may be `None` in a
  source used only for prediction.

Reading Parquet works the same way: `scan()` can loop over pyarrow's
`ParquetFile.iter_batches(batch_size=batch_rows)` (a third-party method, not
part of jaxgam) and convert each record batch to columns, with the
fingerprint taken from the file's path, size and modification time.

## Related pages

- [Choosing a route](choosing_a_route.md)
- [Streamed fitting](streamed.md), [Discretized fitting](discretized.md),
  [Operator fitting](operator.md)
- [Prediction results](prediction_results.md) for `predict_iter()`
- [Memory and performance](memory_and_performance.md) for `batch_rows`
- API reference: [Row sources](../api.md#row-sources)
