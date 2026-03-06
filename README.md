<p align="center">
  <img src="docs/img/jaxgam_icon.svg" alt="jaxgam" width="400"/>
</p>

<p align="center">
  <a href="https://codecov.io/gh/scatt67/jaxgam">
    <img src="https://codecov.io/gh/scatt67/jaxgam/branch/main/graph/badge.svg" alt="coverage"/>
  </a>
</p>

A Python reimplementation of R's
[mgcv](https://cran.r-project.org/package=mgcv) package by
[Simon N. Wood](https://webhomes.maths.ed.ac.uk/~swood34/), for fitting
Generalized Additive Models. mgcv is the gold-standard GAM library and
the algorithms in jaxgam - penalised iteratively re-weighted least
squares (PIRLS), Laplace-approximate REML (empirical bayes), and the full smooth
construction pipeline - follow Wood's published methods and his
[*Generalized Additive Models: An Introduction with R*](https://www.routledge.com/Generalized-Additive-Models-An-Introduction-with-R-Second-Edition/Wood/p/book/9781498728331) textbook.

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

Gaussian, Binomial, Poisson, Gamma - each with its default link and
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

- `predict()` - response or link scale, with optional standard errors
- `summary()` - parametric and smooth term significance tests
- `plot()` - 1D smooth curves with SE bands, 2D contour plots, rug marks

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

Full benchmark comparing jaxgam (true cold, cold, warm) against R
`gam(REML)`. Iteration counts are included to show that both
implementations converge in a similar number of outer Newton steps.

| smooth | family | n | true cold (ms) | cold (ms) | warm (ms) | R gam (ms) | cold/R | warm/R | py iter | R iter |
|--------|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cr | gaussian | 500 | | 4 | 5 | 7 | 1.7x | 1.5x | 5 | 5 |
| cr | gaussian | 2,000 | | 716 | 8 | 11 | 0.02x | 1.3x | 7 | 7 |
| cr | gaussian | 10,000 | | 794 | 21 | 49 | 0.06x | 2.4x | 8 | 8 |
| cr | gaussian | 100,000 | | 1,010 | 172 | 854 | 0.85x | 5.0x | 10 | 10 |
| cr | gaussian | 500,000 | 1,766 | 1,109 | 1,036 | 3,558 | 3.2x | 3.4x | 11 | 11 |
| cr | poisson | 500 | | 6 | 7 | 4 | 0.68x | 0.61x | 3 | 2 |
| cr | poisson | 2,000 | | 822 | 7 | 12 | 0.01x | 1.8x | 3 | 4 |
| cr | poisson | 10,000 | | 713 | 23 | 68 | 0.10x | 3.0x | 5 | 5 |
| cr | poisson | 100,000 | | 1,018 | 187 | 941 | 0.92x | 5.0x | 6 | 7 |
| cr | poisson | 500,000 | 1,811 | 1,285 | 1,175 | 4,713 | 3.7x | 4.0x | 7 | 8 |
| cr | binomial | 500 | | 6 | 5 | 6 | 1.0x | 1.3x | 3 | 3 |
| cr | binomial | 2,000 | | 855 | 11 | 12 | 0.01x | 1.1x | 5 | 4 |
| cr | binomial | 10,000 | | 794 | 30 | 57 | 0.07x | 1.9x | 6 | 5 |
| cr | binomial | 100,000 | | 1,185 | 221 | 703 | 0.59x | 3.2x | 7 | 6 |
| cr | binomial | 500,000 | 1,940 | 1,762 | 1,580 | 4,579 | 2.6x | 2.9x | 9 | 8 |
| cr | gamma | 500 | | 8 | 6 | 7 | 0.91x | 1.3x | 3 | 5 |
| cr | gamma | 2,000 | | 1,161 | 9 | 16 | 0.01x | 1.9x | 3 | 6 |
| cr | gamma | 10,000 | | 942 | 25 | 91 | 0.10x | 3.6x | 3 | 7 |
| cr | gamma | 100,000 | | 1,271 | 202 | 1,324 | 1.0x | 6.6x | 5 | 9 |
| cr | gamma | 500,000 | 1,991 | 1,479 | 1,326 | 6,068 | 4.1x | 4.6x | 6 | 10 |
| two | gaussian | 500 | | 39 | 7 | 9 | 0.23x | 1.3x | 6 | 6 |
| two | gaussian | 2,000 | | 790 | 14 | 16 | 0.02x | 1.1x | 8 | 8 |
| two | gaussian | 10,000 | | 936 | 43 | 72 | 0.08x | 1.7x | 12 | 9 |
| two | gaussian | 100,000 | | 1,092 | 321 | 956 | 0.88x | 3.0x | 11 | 11 |
| two | gaussian | 500,000 | 2,587 | 2,061 | 1,857 | 5,056 | 2.5x | 2.7x | 12 | 12 |
| two | poisson | 500 | | 52 | 10 | 11 | 0.21x | 1.1x | 8 | 8 |
| two | poisson | 2,000 | | 944 | 14 | 18 | 0.02x | 1.3x | 6 | 5 |
| two | poisson | 10,000 | | 699 | 43 | 99 | 0.14x | 2.3x | 6 | 7 |
| two | poisson | 100,000 | | 1,151 | 350 | 1,251 | 1.1x | 3.6x | 7 | 8 |
| two | poisson | 500,000 | 2,593 | 2,403 | 2,286 | 6,703 | 2.8x | 2.9x | 8 | 9 |
| two | binomial | 500 | | 33 | 10 | 12 | 0.36x | 1.2x | 9 | 9 |
| two | binomial | 2,000 | | 866 | 10 | 18 | 0.02x | 1.8x | 3 | 5 |
| two | binomial | 10,000 | | 928 | 40 | 87 | 0.09x | 2.2x | 5 | 6 |
| two | binomial | 100,000 | | 1,106 | 366 | 1,130 | 1.0x | 3.1x | 7 | 7 |
| two | binomial | 500,000 | 2,675 | 2,520 | 2,450 | 6,069 | 2.4x | 2.5x | 8 | 8 |
| two | gamma | 500 | | 10 | 7 | 17 | 1.6x | 2.3x | 4 | 6 |
| two | gamma | 2,000 | | 1,144 | 18 | 38 | 0.03x | 2.1x | 6 | 8 |
| two | gamma | 10,000 | | 965 | 37 | 225 | 0.23x | 6.0x | 4 | 9 |
| two | gamma | 100,000 | | 1,391 | 376 | 1,765 | 1.3x | 4.7x | 6 | 10 |
| two | gamma | 500,000 | 3,028 | 2,652 | 2,494 | 8,724 | 3.3x | 3.5x | 7 | 11 |
| te | gaussian | 500 | | 18 | 14 | 14 | 0.78x | 0.98x | 14 | 10 |
| te | gaussian | 2,000 | | 1,166 | 18 | 36 | 0.03x | 2.0x | 8 | 9 |
| te | gaussian | 10,000 | | 954 | 54 | 591 | 0.62x | 11.0x | 10 | 10 |
| te | gaussian | 100,000 | | 1,963 | 682 | 3,909 | 2.0x | 5.7x | 14 | 11 |
| te | gaussian | 500,000 | 3,966 | 3,439 | 3,453 | 40,432 | 11.8x | 11.7x | 12 | 13 |
| te | poisson | 500 | | 16 | 11 | 16 | 1.0x | 1.4x | 8 | 6 |
| te | poisson | 2,000 | | 794 | 19 | 27 | 0.03x | 1.4x | 5 | 5 |
| te | poisson | 10,000 | | 1,018 | 60 | 145 | 0.14x | 2.4x | 6 | 7 |
| te | poisson | 100,000 | | 1,479 | 694 | 2,172 | 1.5x | 3.1x | 8 | 8 |
| te | poisson | 500,000 | 4,097 | 4,027 | 4,370 | 16,049 | 4.0x | 3.7x | 8 | 9 |
| te | binomial | 500 | | 30 | 32 | 13 | 0.43x | 0.41x | 15 | 6 |
| te | binomial | 2,000 | | 903 | 23 | 28 | 0.03x | 1.2x | 5 | 5 |
| te | binomial | 10,000 | | 904 | 69 | 107 | 0.12x | 1.5x | 5 | 5 |
| te | binomial | 100,000 | | 1,561 | 725 | 2,228 | 1.4x | 3.1x | 7 | 8 |
| te | binomial | 500,000 | 4,386 | 4,435 | 4,843 | 11,582 | 2.6x | 2.4x | 8 | 9 |
| te | gamma | 500 | | 17 | 15 | 14 | 0.81x | 0.94x | 6 | 6 |
| te | gamma | 2,000 | | 1,123 | 38 | 39 | 0.03x | 1.0x | 8 | 7 |
| te | gamma | 10,000 | | 1,159 | 79 | 671 | 0.58x | 8.5x | 5 | 9 |
| te | gamma | 100,000 | | 1,853 | 799 | 4,300 | 2.3x | 5.4x | 7 | 10 |
| te | gamma | 500,000 | 4,691 | 4,443 | 4,361 | 52,525 | 11.8x | 12.0x | 7 | 11 |
| cr_by | gaussian | 500 | | 27 | 18 | 17 | 0.62x | 0.95x | 14 | 10 |
| cr_by | gaussian | 2,000 | | 819 | 26 | 30 | 0.04x | 1.2x | 13 | 8 |
| cr_by | gaussian | 10,000 | | 820 | 40 | 99 | 0.12x | 2.5x | 7 | 7 |
| cr_by | gaussian | 100,000 | | 1,377 | 513 | 1,554 | 1.1x | 3.0x | 9 | 9 |
| cr_by | gaussian | 500,000 | 4,239 | 3,631 | 3,284 | 9,219 | 2.5x | 2.8x | 10 | 10 |
| cr_by | poisson | 500 | | 23 | 15 | 20 | 0.87x | 1.4x | 8 | 8 |
| cr_by | poisson | 2,000 | | 746 | 24 | 41 | 0.05x | 1.7x | 7 | 7 |
| cr_by | poisson | 10,000 | | 796 | 49 | 147 | 0.18x | 3.0x | 5 | 5 |
| cr_by | poisson | 100,000 | | 1,420 | 625 | 1,816 | 1.3x | 2.9x | 6 | 6 |
| cr_by | poisson | 500,000 | 4,341 | 4,342 | 3,940 | 10,061 | 2.3x | 2.6x | 6 | 7 |
| cr_by | binomial | 500 | | 24 | 17 | 19 | 0.81x | 1.1x | 8 | 8 |
| cr_by | binomial | 2,000 | | 1,097 | 27 | 41 | 0.04x | 1.5x | 8 | 7 |
| cr_by | binomial | 10,000 | | 894 | 51 | 129 | 0.14x | 2.5x | 5 | 5 |
| cr_by | binomial | 100,000 | | 1,640 | 740 | 1,491 | 0.91x | 2.0x | 7 | 5 |
| cr_by | binomial | 500,000 | 5,543 | 5,853 | 4,993 | 8,777 | 1.5x | 1.8x | 8 | 6 |
| cr_by | gamma | 500 | | 14 | 12 | 15 | 1.1x | 1.3x | 4 | 5 |
| cr_by | gamma | 2,000 | | 1,075 | 13 | 36 | 0.03x | 2.7x | 3 | 5 |
| cr_by | gamma | 10,000 | | 1,105 | 57 | 210 | 0.19x | 3.7x | 5 | 6 |
| cr_by | gamma | 100,000 | | 1,649 | 566 | 2,325 | 1.4x | 4.1x | 4 | 8 |
| cr_by | gamma | 500,000 | 4,756 | 4,553 | 4,039 | 12,502 | 2.7x | 3.1x | 5 | 9 |

### Cold starts

The first fit includes JIT tracing + XLA compilation (~700-1200ms
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
  simulation) - warm fits are 2-12x faster than R `gam(REML)`
- Your datasets are large (n > 100,000) - the XLA advantage grows
  with n

**R's mgcv may be better when:**
- You don't care about using python or R you just care about the best tool for the job, or you can use R in your production environement (R is a great tool) !
- You need one-shot fits on small data (n < 100,000) and cold-start
  latency matters
- You can use `bam(fREML)` for very large datasets, or features beyond
  v1.0 scope (sparse solvers, extended families, random effects)

In most cases you probably should just use the original `mgcv` in R it's very robust and efficient! If you are a pure python user, or your tech stack only supports python maybe jaxgam can be useful.

A persistent compilation cache (`~/.cache/jaxgam/jax/`) is enabled by
default to minimize cold-start overhead across Python sessions.
Disable it with `JAXGAM_NO_COMPILATION_CACHE=1`.

## Correctness

jaxgam is validated against R's mgcv 1.9-3 across a 1,450-test suite.
Every model configuration (4 families x 6 smooth types) is fitted in
both jaxgam and R, then compared value-by-value:

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

- **Beyond exponential family** -- Wood SN, Pya N, Safken B (2016). "Smoothing parameter and model selection for general smooth models (with discussion)." *Journal of the American Statistical Association*, 111, 1548--1575. [doi:10.1080/01621459.2016.1180986](https://doi.org/10.1080/01621459.2016.1180986)

- **GCV-based estimation and GAMM basics** -- Wood SN (2004). "Stable and efficient multiple smoothing parameter estimation for generalized additive models." *Journal of the American Statistical Association*, 99(467), 673--686. [doi:10.1198/016214504000000980](https://doi.org/10.1198/016214504000000980)

- **Overview** -- Wood SN (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.

- **Thin plate regression splines** -- Wood SN (2003). "Thin-plate regression splines." *Journal of the Royal Statistical Society (B)*, 65(1), 95--114. [doi:10.1111/1467-9868.00374](https://doi.org/10.1111/1467-9868.00374)

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata
and BibTeX entries.

## License

See [LICENSE](LICENSE) for details.
