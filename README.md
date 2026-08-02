<p align="center">
  <img src="docs/img/jaxgam_icon.svg" alt="jaxgam" width="400"/>
</p>

<p align="center">
  <a href="https://codecov.io/gh/scatt67/jaxgam">
    <img src="https://codecov.io/gh/scatt67/jaxgam/branch/main/graph/badge.svg" alt="coverage"/>
  </a>
</p>

A Python port of R's
[mgcv](https://cran.r-project.org/package=mgcv) package by
[Simon N. Wood](https://webhomes.maths.ed.ac.uk/~swood34/), for fitting
Generalized Additive Models. mgcv is the gold-standard GAM library and
the algorithms in jaxgam - penalised iteratively re-weighted least
squares (PIRLS), Laplace-approximate REML (empirical bayes), and the full smooth
construction pipeline - follow Wood's `mgcv` package, published methods, and his
[*Generalized Additive Models: An Introduction with R*](https://www.routledge.com/Generalized-Additive-Models-An-Introduction-with-R-Second-Edition/Wood/p/book/9781498728331) textbook.

Full attribution is given to Simon Wood and the R `mgcv` package, `jaxgam` being a derivative work follows the same [license](LICENSE) as such.

jaxgam uses [jax](https://github.com/google/jax) for JIT-compiled fitting
with automatic differentiation through the PIRLS inner loop and Newton
outer loop, and [numba](https://numba.pydata.org/) eager compilation for TPRS/tensor basis construction and p-value computation. The reason for doing this is because `mgcv` has custom C code for performance critical portions of the code.

## AI / Agentic development transparency note

This project was built heavily with [Claude Code](https://docs.anthropic.com/en/docs/claude-code).
I wanted to learn the tool while porting my favorite R package, so this
was the excuse. It's a side project for fun and learning, but I tried to
test thoroughly against R's reference output so it might actually be useful
to someone.

My strategy was to first create an in depth [design document](docs/design.md) which I went back and forth with claude and third party AI reviewers on to flesh out the design and scope. I then used claude to create an [implementation plan](docs/IMPLEMENTATION_PLAN.md) from the design doc.

What I found helpful was including a local mgcv source and an [R reference map](docs/R_SOURCE_MAP.md) for agents to utilize when porting certain functionality. While others have used skills, I just used an [AGENTS.md](AGENTS.md) file with certain instructions. Though I did have to constantly remind the agent to check it...

Some issues I found with claude/agents, 1. I had to review the tests closely, they tried to cheat by changing the tolerances for `np.assert*`, 2. When they read a lot of R code they don't adhere to idiomatic python (which I thought was personally interesting). I used ruff in our pre-commit to try and catch some of this, but whatever ruff or I didn't catch I add an independent review agent go through later and check for PEP violations. 3. Even with AGENTS.md it still would forget to use tooling provided in the repo from time to time (e.g. `uv`), maybe I should have setup one of those skills ?  

Overall, this was fun I learned a lot about agents/claude code, but I also learned more about mgcv. While I thought I knew a decent amount, and I have perused the source code and docs many times in the past there was still implementations I never knew of (Some custom C implementations I didn't know of !). Is this a production ready package, most likely not... but maybe it can be useful, and a demonstration of want agentic development can do. 

### AI development knowledge sharing

An interesting workflow that I learned to use and worked well (which in hindsight is obvious) for very hard problems is to setup an experiment tracking document. What I found was that even though claude code had a `MEMORY.md` it still can get circular when it comes to solving hard problems. My presumption is that it loses context. 

Example: 

For the Newton optimzer used in the smoothing parameter outer loop had many convergence problems as we missed many conventions used by mgcv in our design document setup, e.g. there is a C implementation `gdi.c` which has many optimization helpers that I didn't know of, and we weren't fully differentiating through the PIRLs loop (treating it as a constant, linear convergence). When going back and forth with claude code it was just spinning in circles, even if I restarted the session to clear context and rely on it's "memory" the result was the same.

The break through came when we setup an experiment document to track past experiments in improving the convergence. On each prompt to claude I refered this document, and had it update it after each go at solving the convergence issue. Within a very few iterations claude was able to solve the problem for most families (later we fixed the Gaussian). For the benefit of others (and transparency) I included this [experiments](docs/experiments.md) document! The end solution was a `custom_jvp` for PIRLS inner loop which is obvious in hindsight. 


## Installation

```bash
# Clone and install with uv
git clone https://github.com/scatt67/jaxgam.git
cd jaxgam
uv sync
```

## Quickstart

`GAM.fit()` has two explicit result modes:

- `result="full"` (the default) returns `GAMResults` for prediction,
  `summary()`, `plot()`, and in-sample state.
- `result="inference"` returns a lean `GAMInferenceResult` for new-data
  prediction plus lightweight diagnostics, without dense training state.

```python
import numpy as np
import pandas as pd
from jaxgam import GAM

# Generate data
rng = np.random.default_rng(42)
x = rng.uniform(0, 1, 200)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, 200)
data = pd.DataFrame({"x": x, "y": y})

# Full analytical result. The explicit mode is equivalent to .fit(data).
results = GAM("y ~ s(x, k=10, bs='cr')").fit(data, result="full")

# Inspect results
results.summary()
fig, axes = results.plot()

# Predict on new data
newdata = pd.DataFrame({"x": np.linspace(0, 1, 100)})
predictions = results.predict(newdata)
predictions, se = results.predict(newdata, se_fit=True)
```

For new-data prediction plus lightweight fit diagnostics, choose the other mode
at fit time:

```python
lean = GAM("y ~ s(x, k=10, bs='cr')").fit(data, result="inference")
predictions = lean.predict(newdata)
```

`GAMInferenceResult` is already lean and ready to predict. You do **not** need
to call `to_predictor()` afterward. That method is only an optional handoff when
another component should receive the narrower prediction-only `GAMPredictor`;
on an inference result it returns the existing core without copying or reducing
memory further.

See [docs/quickstart.md](docs/quickstart.md) for a full tutorial covering
all families, smooth types, and post-estimation tools.

## What v1.0 supports

### Families

Gaussian, Binomial, Poisson, Gamma, and Negative Binomial - each with
its default link and REML smoothing parameter selection. The Negative
Binomial is an extended family with an estimated dispersion parameter
(theta).

### Smooth types

| Formula syntax | Basis type |
|---|---|
| `s(x, bs='tp')` | Thin-plate regression spline (default) |
| `s(x, bs='ts')` | Thin-plate spline with shrinkage |
| `s(x, bs='cr')` | Cubic regression spline |
| `s(x, bs='cs')` | Cubic spline with shrinkage |
| `s(x, bs='cc')` | Cyclic cubic spline |
| `s(x, z, bs='gp')` | Low-rank Gaussian process smooth |
| `s(g, bs='re')` | Dense random effect smooth |
| `te(x1, x2)` | Tensor product smooth |
| `ti(x1, x2)` | Tensor interaction (no main effects) |
| `s(x, by=fac)` | Factor-by smooth (separate curve per level) |

### Post-estimation

- `predict()` - response or link scale, with optional standard errors
- `summary()` - parametric and smooth term significance tests
- `plot()` - 1D smooth curves with SE bands, 2D contour plots, rug marks

## v1.0 limitations

These are deliberate scope boundaries, not bugs:

1. **No sparse solver.** Models with > ~5,000 basis functions will hit the
   dense memory ceiling. Factor-by with many levels or large tensor products
   are most affected.
2. **Five families only.** Tweedie, Beta, and other extended families
   beyond Negative Binomial are not yet available.
3. **Dense design matrix must fit in memory.** Datasets with > ~10M rows
   require chunked processing, which is not implemented.
4. **No factor-smooth interactions.** `bs="fs"` requires sparse linear
   algebra and is not implemented. Simple dense random effects
   (`bs="re"`, including random slopes) *are* supported — see the
   quickstart.
5. **No GAMM.** Correlated random effects (`gamm()`) are not supported.

See the [design document](docs/design.md) Section 1.2 for details on what
is planned for v1.1+.

## Performance

jaxgam uses JAX's XLA compiler for JIT-compiled fitting. Performance
depends on whether the JIT cache is warm (compiled code reused) or cold
(first fit triggers compilation). R is benchmarked with both
`gam(method="REML")` and `bam(method="fREML")`.

**Note on R's BLAS:** R is benchmarked using its default (reference) BLAS
and LAPACK, which are notoriously slow. Building R with OpenBLAS would
give R a significant speedup, but we avoided this because OpenBLAS must
be compiled from source with multi-threading disabled — Simon Wood notes
in the [mgcv changelog](https://github.com/cran/mgcv/blob/master/ChangeLog#L13-L16)
that multi-threaded BLAS can cause issues with mgcv's internal
parallelism. The benchmarks therefore reflect a common R installation
rather than an optimally configured one.

### Benchmark results

Full benchmark comparing jaxgam (cold, warm) against R
`gam(REML)`. Iteration counts are included to show that both
implementations converge in a similar number of outer Newton steps.
The Negative Binomial (nb) is an extended family with an outer loop
over the dispersion parameter theta, which adds overhead to both
cold and warm fits.

| smooth | family | n | cold (ms) | warm (ms) | R gam (ms) | cold/R | warm/R | py iter | R iter |
|--------|--------|---:|---:|---:|---:|---:|---:|---:|---:|
| cr | gaussian | 500 | 6 | 4 | 8 | 1.4x | 1.9x | 5 | 5 |
| cr | gaussian | 2,000 | 852 | 7 | 12 | 0.01x | 1.6x | 7 | 7 |
| cr | gaussian | 10,000 | 880 | 20 | 46 | 0.05x | 2.3x | 8 | 8 |
| cr | gaussian | 100,000 | 1046 | 163 | 824 | 0.79x | 5.1x | 10 | 10 |
| cr | gaussian | 500,000 | 1014 | 981 | 3620 | 3.6x | 3.7x | 11 | 11 |
| cr | poisson | 500 | 7 | 4 | 5 | 0.75x | 1.3x | 3 | 2 |
| cr | poisson | 2,000 | 875 | 7 | 13 | 0.01x | 1.9x | 3 | 4 |
| cr | poisson | 10,000 | 902 | 23 | 61 | 0.07x | 2.7x | 5 | 5 |
| cr | poisson | 100,000 | 1296 | 175 | 820 | 0.63x | 4.7x | 6 | 7 |
| cr | poisson | 500,000 | 1129 | 1138 | 4486 | 4.0x | 3.9x | 7 | 8 |
| cr | binomial | 500 | 8 | 5 | 6 | 0.79x | 1.2x | 3 | 3 |
| cr | binomial | 2,000 | 952 | 11 | 14 | 0.01x | 1.3x | 5 | 4 |
| cr | binomial | 10,000 | 968 | 30 | 62 | 0.06x | 2.1x | 6 | 5 |
| cr | binomial | 100,000 | 1168 | 228 | 902 | 0.77x | 4.0x | 7 | 6 |
| cr | binomial | 500,000 | 1597 | 1568 | 4588 | 2.9x | 2.9x | 9 | 8 |
| cr | gamma | 500 | 9 | 6 | 11 | 1.2x | 2.0x | 3 | 5 |
| cr | gamma | 2,000 | 1112 | 8 | 21 | 0.02x | 2.8x | 3 | 6 |
| cr | gamma | 10,000 | 1120 | 21 | 90 | 0.08x | 4.4x | 3 | 7 |
| cr | gamma | 100,000 | 1337 | 188 | 1345 | 1.0x | 7.1x | 5 | 9 |
| cr | gamma | 500,000 | 1264 | 1230 | 6059 | 4.8x | 4.9x | 6 | 10 |
| cr | nb | 500 | 11 | 8 | 8 | 0.72x | 0.97x | 4 | 4 |
| cr | nb | 2,000 | 1664 | 13 | 14 | 0.01x | 1.1x | 3 | 3 |
| cr | nb | 10,000 | 1946 | 46 | 78 | 0.04x | 1.7x | 5 | 4 |
| cr | nb | 100,000 | 2076 | 414 | 1048 | 0.50x | 2.5x | 6 | 6 |
| cr | nb | 500,000 | 2687 | 2641 | 6352 | 2.4x | 2.4x | 7 | 7 |
| two | gaussian | 500 | 11 | 7 | 9 | 0.84x | 1.2x | 6 | 6 |
| two | gaussian | 2,000 | 910 | 15 | 21 | 0.02x | 1.4x | 8 | 8 |
| two | gaussian | 10,000 | 915 | 41 | 75 | 0.08x | 1.8x | 12 | 9 |
| two | gaussian | 100,000 | 1171 | 288 | 994 | 0.85x | 3.4x | 11 | 11 |
| two | gaussian | 500,000 | 1697 | 1686 | 6401 | 3.8x | 3.8x | 12 | 12 |
| two | poisson | 500 | 12 | 9 | 14 | 1.1x | 1.5x | 8 | 8 |
| two | poisson | 2,000 | 838 | 13 | 22 | 0.03x | 1.7x | 6 | 5 |
| two | poisson | 10,000 | 860 | 36 | 131 | 0.15x | 3.6x | 6 | 7 |
| two | poisson | 100,000 | 1119 | 317 | 1302 | 1.2x | 4.1x | 7 | 8 |
| two | poisson | 500,000 | 2086 | 2088 | 6689 | 3.2x | 3.2x | 8 | 9 |
| two | binomial | 500 | 15 | 12 | 12 | 0.78x | 1.0x | 9 | 9 |
| two | binomial | 2,000 | 984 | 10 | 18 | 0.02x | 1.7x | 3 | 5 |
| two | binomial | 10,000 | 1019 | 40 | 90 | 0.09x | 2.3x | 5 | 6 |
| two | binomial | 100,000 | 1333 | 370 | 1297 | 0.97x | 3.5x | 7 | 7 |
| two | binomial | 500,000 | 2452 | 2419 | 6154 | 2.5x | 2.5x | 8 | 8 |
| two | gamma | 500 | 10 | 7 | 18 | 1.8x | 2.6x | 4 | 6 |
| two | gamma | 2,000 | 1222 | 15 | 39 | 0.03x | 2.6x | 6 | 8 |
| two | gamma | 10,000 | 1200 | 36 | 231 | 0.19x | 6.4x | 4 | 9 |
| two | gamma | 100,000 | 1530 | 352 | 1640 | 1.1x | 4.7x | 6 | 10 |
| two | gamma | 500,000 | 2323 | 2257 | 8982 | 3.9x | 4.0x | 7 | 11 |
| two | nb | 500 | 16 | 10 | 12 | 0.74x | 1.2x | 4 | 4 |
| two | nb | 2,000 | 1976 | 24 | 30 | 0.02x | 1.2x | 6 | 6 |
| two | nb | 10,000 | 1671 | 68 | 114 | 0.07x | 1.7x | 5 | 5 |
| two | nb | 100,000 | 2442 | 835 | 1714 | 0.70x | 2.1x | 7 | 7 |
| two | nb | 500,000 | 5037 | 4992 | 8879 | 1.8x | 1.8x | 8 | 8 |
| te | gaussian | 500 | 19 | 13 | 20 | 1.1x | 1.5x | 14 | 10 |
| te | gaussian | 2,000 | 1172 | 17 | 41 | 0.03x | 2.5x | 8 | 9 |
| te | gaussian | 10,000 | 1127 | 49 | 578 | 0.51x | 11.8x | 10 | 10 |
| te | gaussian | 100,000 | 1681 | 619 | 3660 | 2.2x | 5.9x | 14 | 11 |
| te | gaussian | 500,000 | 3317 | 3268 | 41665 | 12.6x | 12.8x | 12 | 13 |
| te | poisson | 500 | 18 | 12 | 15 | 0.85x | 1.2x | 8 | 6 |
| te | poisson | 2,000 | 1042 | 16 | 28 | 0.03x | 1.8x | 5 | 5 |
| te | poisson | 10,000 | 1057 | 56 | 151 | 0.14x | 2.7x | 6 | 7 |
| te | poisson | 100,000 | 1639 | 663 | 2306 | 1.4x | 3.5x | 8 | 8 |
| te | poisson | 500,000 | 3892 | 3923 | 16679 | 4.3x | 4.3x | 8 | 9 |
| te | binomial | 500 | 31 | 26 | 14 | 0.45x | 0.54x | 14 | 6 |
| te | binomial | 2,000 | 1195 | 18 | 27 | 0.02x | 1.5x | 5 | 5 |
| te | binomial | 10,000 | 1180 | 74 | 110 | 0.09x | 1.5x | 5 | 5 |
| te | binomial | 100,000 | 1792 | 710 | 2424 | 1.4x | 3.4x | 7 | 8 |
| te | binomial | 500,000 | 4338 | 4227 | 11895 | 2.7x | 2.8x | 8 | 9 |
| te | gamma | 500 | 17 | 12 | 14 | 0.83x | 1.2x | 6 | 6 |
| te | gamma | 2,000 | 1766 | 29 | 38 | 0.02x | 1.3x | 8 | 7 |
| te | gamma | 10,000 | 1385 | 64 | 664 | 0.48x | 10.4x | 5 | 9 |
| te | gamma | 100,000 | 2071 | 715 | 4393 | 2.1x | 6.1x | 7 | 10 |
| te | gamma | 500,000 | 4254 | 4247 | 55270 | 13.0x | 13.0x | 7 | 11 |
| te | nb | 500 | 18 | 14 | 17 | 0.92x | 1.2x | 4 | 5 |
| te | nb | 2,000 | 1869 | 26 | 30 | 0.02x | 1.2x | 4 | 4 |
| te | nb | 10,000 | 1872 | 95 | 159 | 0.08x | 1.7x | 5 | 5 |
| te | nb | 100,000 | 3130 | 1347 | 2226 | 0.71x | 1.7x | 7 | 7 |
| te | nb | 500,000 | 8228 | 8197 | 15609 | 1.9x | 1.9x | 8 | 8 |
| cr_by | gaussian | 500 | 30 | 21 | 18 | 0.60x | 0.86x | 14 | 10 |
| cr_by | gaussian | 2,000 | 1077 | 29 | 32 | 0.03x | 1.1x | 13 | 8 |
| cr_by | gaussian | 10,000 | 1050 | 41 | 103 | 0.10x | 2.5x | 7 | 7 |
| cr_by | gaussian | 100,000 | 1616 | 539 | 1579 | 0.98x | 2.9x | 9 | 9 |
| cr_by | gaussian | 500,000 | 3856 | 3633 | 9560 | 2.5x | 2.6x | 10 | 10 |
| cr_by | poisson | 500 | 27 | 18 | 18 | 0.67x | 0.99x | 8 | 8 |
| cr_by | poisson | 2,000 | 1030 | 28 | 41 | 0.04x | 1.5x | 7 | 7 |
| cr_by | poisson | 10,000 | 976 | 56 | 140 | 0.14x | 2.5x | 5 | 5 |
| cr_by | poisson | 100,000 | 1606 | 681 | 1880 | 1.2x | 2.8x | 6 | 6 |
| cr_by | poisson | 500,000 | 3999 | 3989 | 10528 | 2.6x | 2.6x | 6 | 7 |
| cr_by | binomial | 500 | 26 | 20 | 19 | 0.72x | 0.94x | 8 | 8 |
| cr_by | binomial | 2,000 | 1658 | 30 | 43 | 0.03x | 1.4x | 8 | 7 |
| cr_by | binomial | 10,000 | 1086 | 63 | 129 | 0.12x | 2.0x | 5 | 5 |
| cr_by | binomial | 100,000 | 2512 | 838 | 1662 | 0.66x | 2.0x | 7 | 5 |
| cr_by | binomial | 500,000 | 5512 | 5315 | 9225 | 1.7x | 1.7x | 8 | 6 |
| cr_by | gamma | 500 | 15 | 11 | 14 | 0.95x | 1.3x | 4 | 5 |
| cr_by | gamma | 2,000 | 1213 | 16 | 35 | 0.03x | 2.2x | 3 | 5 |
| cr_by | gamma | 10,000 | 1270 | 68 | 216 | 0.17x | 3.2x | 5 | 6 |
| cr_by | gamma | 100,000 | 1799 | 596 | 2448 | 1.4x | 4.1x | 4 | 8 |
| cr_by | gamma | 500,000 | 4199 | 4634 | 13300 | 3.2x | 2.9x | 5 | 9 |
| cr_by | nb | 500 | 20 | 17 | 16 | 0.81x | 0.94x | 4 | 4 |
| cr_by | nb | 2,000 | 1656 | 40 | 49 | 0.03x | 1.2x | 6 | 6 |
| cr_by | nb | 10,000 | 1721 | 90 | 154 | 0.09x | 1.7x | 4 | 4 |
| cr_by | nb | 100,000 | 3163 | 1558 | 2052 | 0.65x | 1.3x | 5 | 5 |
| cr_by | nb | 500,000 | 10605 | 10930 | 13000 | 1.2x | 1.2x | 7 | 6 |

### Cold starts

The first fit includes JIT tracing + XLA compilation (~900-1500ms
overhead). This makes jaxgam slower than R for small datasets on first
use, but the compiled code is cached to disk and reused across sessions.

![Cold-start speedup vs dataset size](docs/img/speedup_vs_n.png)

The crossover where even a cold jaxgam fit beats `gam(REML)` is around n=100,000. The `bam(fREML)` wins in all n as it was purpose built for fitting very large data! Many people don't associate `mgcv` for large scale training, but if this benchmark shows anything is that it is certainly up for the task!

### High-dimensional models

For models with many basis functions (k=100-500), jaxgam's XLA-compiled
dense linear algebra outperforms R `gam(REML)` even on the very first
cold-start fit. We also benchmark against R's `bam(fREML)` with 8
threads, which was purpose-built for large datasets and is very fast at
high k. A `bam(fREML)` port is on the roadmap.

![jaxgam vs R at large p](docs/img/large_p_results.png)

Note: jaxgam currently implements `gam(REML)` only. The benchmarks above
compare against both R's `gam(REML)` (apples-to-apples) and `bam(fREML)`
(to show what mgcv can do with its large data optimizer!).

### When to use jaxgam

**jaxgam is a good fit when:**
- You want REML based GAMs in Python. AFAIK other Python GAM
  implementations offer Generalized Cross Validation (GCV) or full
  Bayes, whereas REML (empirical Bayes) is generally more robust than
  GCV and faster than full Bayes
- You fit the same model structure repeatedly (bootstrap, CV,
  simulation) - warm fits are 2-13x faster than R `gam(REML)` for
  standard families (Gaussian, Poisson, Binomial, Gamma)
- Your datasets are large (n > 100,000) - the XLA advantage grows
  with n

**R's mgcv may be better when:**
- You don't care about using python or R you just care about the best tool for the job, or you can use R in your production environement (R is a great tool) !
- You need one-shot fits on small data (n < 100,000) and cold-start
  latency matters
- You can use `bam(fREML)` for very large datasets, or features beyond
  v1.0 scope (sparse solvers, Tweedie/Beta and other extended families,
  factor-smooth interactions)

In most cases you probably should just use the original `mgcv` in R it's very robust and efficient! If you are a pure python user, or your tech stack only supports python maybe jaxgam can be useful.

A persistent compilation cache (`~/.cache/jaxgam/jax/`) is enabled by
default to minimize cold-start overhead across Python sessions.
Disable it with `JAXGAM_NO_COMPILATION_CACHE=1`.

## Correctness

jaxgam is validated against R's mgcv 1.9-3 across an extensive R-parity test
suite. Every model configuration in the validation matrix (5 families x the
supported smooth configurations) is fitted in both jaxgam and R, then compared
value-by-value:

- **Coefficients, fitted values, deviance** - must match R at STRICT
  (rtol=1e-10) or MODERATE (rtol=1e-4) tolerance depending on the
  model type
- **Smoothing parameters** - compared at MODERATE or LOOSE (rtol=1e-2)
  because the REML surface is flat near the optimum
- **Basis matrices and penalty matrices** - compared element-wise
  against R's `smoothCon()` output with sign normalization to handle
  LAPACK eigenvector sign ambiguity
- **Summary statistics** - EDF, p-values, and significance tests
  validated against R's `summary.gam()`
- **Predictions and standard errors** - `predict()` output compared
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

## Citations

jaxgam is a Python port of Simon Wood's
[mgcv](https://cran.r-project.org/package=mgcv) R package. The
statistical methods are entirely his work. If you use jaxgam, please
cite the relevant mgcv papers:

- **GAM method (REML/ML estimation)** -- Wood SN (2011). "Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models." *Journal of the Royal Statistical Society (B)*, 73(1), 3--36. [doi:10.1111/j.1467-9868.2010.00749.x](https://doi.org/10.1111/j.1467-9868.2010.00749.x)

- **Beyond exponential family** -- Wood SN, Pya N, Säfken B (2016). "Smoothing parameter and model selection for general smooth models (with discussion)." *Journal of the American Statistical Association*, 111, 1548--1575. [doi:10.1080/01621459.2016.1180986](https://doi.org/10.1080/01621459.2016.1180986)

- **GCV-based model method and basics of GAMM** -- Wood SN (2004). "Stable and efficient multiple smoothing parameter estimation for generalized additive models." *Journal of the American Statistical Association*, 99(467), 673--686. [doi:10.1198/016214504000000980](https://doi.org/10.1198/016214504000000980)

- **Overview** -- Wood SN (2017). *Generalized Additive Models: An Introduction with R*, 2 edition. Chapman and Hall/CRC.

- **Thin plate regression splines** -- Wood SN (2003). "Thin-plate regression splines." *Journal of the Royal Statistical Society (B)*, 65(1), 95--114. [doi:10.1111/1467-9868.00374](https://doi.org/10.1111/1467-9868.00374)

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata
and BibTeX entries.

## License

Licensed under [GPL-2.0-or-later](LICENSE), matching mgcv's `GPL (>= 2)`
license. As a derivative work of mgcv, this ensures downstream users have
the same freedoms granted by the original package.
