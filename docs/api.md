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

### Gaussian QR fitting

Dense Gaussian identity-link models can opt into frozen QR compression. The
training rows are reduced once, and REML smoothing trials then use only
coefficient-sized statistics. Models with at most 50 penalties retain Newton;
larger penalty vectors use bounded L-BFGS-B with exact gradients and no full
outer Hessian:

```python
from jaxgam import FitControl, GAM

result = GAM(
    "y ~ s(x1) + s(x2)",
    family="gaussian",
    control=FitControl(gaussian_compression=True),
).fit(data)
```

The ordinary dense route remains the default. The QR route accepts finite
positive prior weights between `1e-10` and `1e10` and a full identifiable
coefficient map. Unsupported cases use dense fitting and report the reason in
`execution_fallback_reason`. Estimated EFS smoothing also uses dense fitting.
For a replayable `RowSource`, set `execution="stream"` and request
`result="prediction"`; eligible Gaussian identity models use the same frozen
QR statistics with a prospective memory check, with fixed or estimated
smoothing and either `linear_solver`. An opted-in RowSource fit that is not
eligible (another family or link, estimated EFS selection, weights outside the
QR boundary, or an incomplete coefficient map) takes the streamed route it
would take without the request and reports the reason in
`execution_fallback_reason`; estimated smoothing then uses exact streamed REML,
which requires `linear_solver="qr"`.

Known limit above 50 penalties: the QR route then optimizes with L-BFGS-B,
which can make slow progress when many smoothing parameters and the log
scale are badly scaled against each other. It may stop at its iteration
limit short of the REML optimum. It then reports `converged=False` and the
stop in `convergence_info` (for example `iteration limit`). For models with
many smooths, use the discrete route (`execution="discrete"`), whose
working-model smoothing selection runs Newton steps.

### Streamed QR coefficient solves

Fixed-sp streamed fits can request the positive-weight QR/TSQR solver
explicitly:

```python
from jaxgam import FitControl, GAM

result = GAM(
    'y ~ x + s(z, bs="cr", k=8)',
    family="poisson",
    sp=[0.4],
    control=FitControl(
        execution="stream", linear_solver="qr", batch_rows=4096
    ),
).fit(source, result="prediction")
```

Fixed smoothing parameters support Gaussian, Gamma, Poisson and Binomial
with their built-in link constructors, plus negative binomial with log,
identity and square-root links. Inputs must satisfy the family's response,
mean and initialization domain; invalid default starts still reject.
Noncanonical and NB fits require `linear_solver="qr"`. The existing canonical
Gaussian/identity, Poisson/log and Binomial/logit Cholesky route remains
available.

Canonical QR fits factor positive Fisher working rows and local penalty
roots. Other regular and NB links use signed observed-information fitting,
with separate Fisher factors for EDF and optional prediction uncertainty.
Unknown-scale regular families estimate the REML scale while keeping every
supplied smoothing parameter fixed. `NegativeBinomial(fixed=False)` estimates
theta with the exact REML parameter driver, including an unpenalized model
specified with `sp=[]`; `fixed=True` preserves the supplied theta. The fitted
result owns a family snapshot and does not mutate the caller's family.

With every smoothing parameter fixed, the scale is the only free REML
coordinate. After the limited-memory stage, the fit continues mgcv's exact
log-scale Newton step on fully refitted source trials until the next
correction is no longer representable, so `score_scale` is the stationary
REML scale. Supplied smoothing parameters are never changed.

Inspect `converged` and `convergence_info`: a finite fit with an exhausted
optimizer is reported as nonconverged. `score_scale` records the scale used
by the returned REML score separately from the reported `scale`.
`source_scans`, `batches_scanned` and `known_workspace_bytes` describe the
executed fit and its conservative fit-workspace bound. Request
`uncertainty="fisher"` for batched standard errors.

Preparation scans also honor `FitControl.batch_rows` and
`memory_budget_bytes`. Setup requests at most `batch_rows` rows per scan and
shrinks that size until its prospective workspace, including the bounded
SQLite scope used for exact cubic knots, fits the budget. A budget too small
for one row is rejected before any full data scan. When the budget also has
room for them, the cubic covariates are gathered in memory (8 bytes per row
per column, as many columns per scan as fit) and their exact knots come from
a sort instead of SQLite; the knots are bitwise identical, and a smaller
budget keeps setup memory independent of the number of rows. Knot, alias and
rank decisions do not depend on the batch size. The source's own storage is
outside these bounds.

Model preparation fixes the identifiable coefficient subspace before fitting.
Exactly aliased numeric parametric columns follow the same retained-column and
`CoefficientMap` convention as dense setup, independent of source batch
boundaries. Remaining coupled rank deficiency between parametric and smooth
null-space directions is rejected explicitly; it is not resolved by QR pivot
choice or numerical jitter. Successful fits report
`execution_route="stream_qr"`.

### Estimated smoothing with exact streamed REML

Omit `sp` with the default optimizer to select smoothing parameters by exact
REML over a replayable source:

```python
from jaxgam import FitControl, GAM

result = GAM(
    'y ~ x + s(z, bs="cr", k=10)',
    family="gamma",
    control=FitControl(
        execution="stream", linear_solver="qr", batch_rows=4096
    ),
).fit(source, result="prediction")
```

Each trial refits the coefficients by streamed PIRLS and evaluates the exact
REML score and gradient with a batched adjoint. A bounded-memory L-BFGS-B
optimizer then updates the log smoothing parameters, together with the log
scale for Gaussian and Gamma and the log theta for
`NegativeBinomial(fixed=False)`. No full smoothing-parameter Hessian or
training design is formed. The objective is the dense REML criterion, but the
optimization path differs from the dense Newton optimizer, so fits agree at
the optimum to numerical tolerance rather than step by step. The supported
families and links are those of the fixed-sp route, and `linear_solver="qr"`
is required. A model with no smoothing, scale or theta parameter to select
falls back to a plain coefficient fit. An eligible Gaussian identity fit that
sets `gaussian_compression=True` takes the frozen Gaussian QR route instead
and reports `execution_route="stream_gaussian_qr"`.

Near the optimum, the remaining REML decrease can fall below the score's
floating-point resolution. L-BFGS-B may then stop on its relative-reduction
test, or after an abnormal line search, before the gradient tolerance is met.
In that case the fit continues mgcv's `newton()` iteration:

- When log scale or log theta is the only free coordinate, it takes exact
  Newton steps on that coordinate (exact curvature for the scale, a secant
  for theta).
- Otherwise it steps on up to four free log parameters, using a central
  difference Hessian built from exact gradients.

Every step refits the source fully and must pass `newton()`'s step cap and
quadratic-model test. Descent is judged by the integrated exact gradient
within a float64 score envelope. At most one such continuation runs per fit.
Fixed smoothing parameters never move, and `converged` is restated on the
returned trial.

The public routes cap each L-BFGS-B proposal with source step windows of
plus or minus 5 in every log parameter (mgcv's `maxNstep`). A run that stops
at a window edge, with descent beyond it, reopens a fresh L-BFGS-B window from
the accepted trial. Each window is a fresh L-BFGS-B run with its own
iteration and evaluation budget (100 iterations, 500 evaluations), and at
most 20 windows are opened. Twenty windows of 5 cross the whole [-40, 40]
smoothing-parameter range. A fit that still descends past the last window
reports `converged=False` and "Window limit 20 reached" in
`convergence_info`. `n_iter` counts iterations over all windows.

Successful fits report `execution_route="stream_exact_reml_qr"` and
`lambda_strategy="exact_reml"`. `converged` requires the projected gradient
to meet the optimizer tolerance. `exact_optimizer_diagnostics` records the
optimizer status, trial evaluations, accepted steps, final projected gradient,
accepted score history, startup scans, cumulative PIRLS passes, final score
refits and derivative time. These counts include the `newton()` continuation
trials.

### Fixed-sp operator fitting

Use CPU operator fitting when an indexed random-effect design has too many
coefficients for a dense coefficient system:

```python
from jaxgam import FitControl, GAM, OperatorControl

result = GAM(
    'y ~ s(group, bs="re")',
    family="poisson",
    sp=[1.0],
    control=FitControl(
        execution="operator",
        linear_solver="pcg",
        batch_rows=4096,
        memory_budget_bytes=512 * 1024 * 1024,
        operator=OperatorControl(cg_rtol=1e-10),
    ),
).fit(source, result="prediction")
```

The source owns weights and offsets. Supply positive finite smoothing
parameters and, for negative binomial, fixed theta. Preparation represents
random-effect levels and interactions by indices. Supported cubic,
thin-plate, Gaussian-process and tensor terms use bounded modest blocks;
`max_modest_coefficients` and the memory budget limit those blocks.

The coefficient driver keeps the family state and working system fixed during
each PCG solve. A working system must satisfy the bounded positive-definiteness
checks. Signed systems outside that certificate reject explicitly, including
some mathematically positive-definite systems; fitting does not silently
switch to Fisher scoring. Diagnostics report actual Krylov iterations,
residuals, source scans and workspace bounds. Check `converged` before using a
fit.

The prediction result retains indexed geometry and coefficients, without the
training source or a dense covariance. This fixed-sp route has `score=None`
and `edf_total=None`; unknown-scale families also have `scale=None`. These
quantities require additional trace and score providers. Standard errors,
full/inference results and estimated theta are not available on this route;
automatic smoothing selection is limited to the calibrated EFS regime below.
Dense and streamed direct routes retain their existing numeric results and
uncertainty options.

### Calibrated operator EFS

Omit `sp` and select `optimizer="efs"` to estimate the smoothing parameter of
an operator fit:

```python
result = GAM(
    'y ~ 0 + s(group, bs="re")',
    family="poisson",
    optimizer="efs",
    control=FitControl(execution="operator", linear_solver="pcg", batch_rows=4096),
).fit(source, result="prediction")
```

The controller follows pinned mgcv's `efsudr`, but each trial decision
compares two operator REML scores that carry absolute error bounds. A trial
is accepted or rejected only when both bounds are admitted and the score
difference exceeds their sum. Otherwise the comparison is indeterminate and
the fit stops with `converged=False` and
`convergence_info="indeterminate_score"`, keeping the last accepted state.
The other stop reasons are `score_window` and `deviance_change` (converged),
and `iteration_limit`, `inner_failure` and `invalid_update` (not converged).
Two undecided comparisons do not stop the fit, because pinned efsudr's action
does not depend on them: an undecided comparison never extends the step (the
plain step is kept), and the step after the final contraction to multiplier 1
is accepted. `operator_diagnostics.efs_indeterminate_comparisons` counts every
undecided comparison, including these.

On every operator result, fixed-sp or EFS, `source_scans` and
`batches_scanned` count the fit's own source work (for EFS: the start
reduction, every refit and every score replay), as on the streamed routes.
Phase-1 preparation is reported separately as
`operator_diagnostics.preparation_started_scans`,
`preparation_yielded_batches` and `preparation_peak_bound_bytes`. On operator
EFS results `known_workspace_bytes` is the planned fit workspace, excluding
preparation; fixed-sp operator results leave it `None`.

The bounds are empirical: they were measured against an independent
50-digit decimal REML reference over a fixed domain
(`docs/scale_jaxgam/efs6_operator_calibration.md`). Only that domain is
public: canonical Poisson with one indexed random effect, either alone or
with at most 4 modest parametric coefficients (`y ~ x + s(group, bs="re")`),
257 to 4096 levels, 1 to 16 rows per level and `batch_rows` from 64 to
65536. Any other shape is rejected before EFS starts. Inside that domain a
state's bound also needs unit prior weights, a zero offset, responses of at
most 32, tight operator controls (`tolerance <= 1e-7`, `cg_rtol <= 1e-10`,
`cg_atol <= 1e-12`) and the conditioning gates listed in the calibration
note; a state that fails them has no admitted bound. The default
`OperatorControl` meets those control limits, but its final gradients are
larger than at the calibration's `tolerance=1e-11`, so more fits can end as
indeterminate. Results report `lambda_strategy="efs_reml"`, the selected
`smoothing_params`, `score`, `edf_total` and the score's error bound:
`score_error_bound` and the calibration that produced it in
`score_bound_method` (`indexed_poisson_empirical_v2` or
`arrowhead_poisson_empirical_v2`). A bound of `None` with method
`unavailable` means the accepted state was outside the admitted conditioning
or control gates and no validated bound is claimed. These labels describe the
score approximation; `converged` and `convergence_info` describe the
optimizer, separately.

### Operator bulk Fisher uncertainty

A converged operator prediction result can build an approximate link-scale
variance provider without forming a coefficient covariance:

```python
provider, calibration = result.bulk_link_variance(
    source, selected_rows, probes=48, seed=7, memory_budget_bytes=256 << 20
)
for batch in provider.estimate_iter(prediction_source):
    batch.variance, batch.monte_carlo_se, batch.link_se
```

Pass the original training source again; its fingerprint, prepared-row
digest and coefficient count, and the family snapshot, must match the fit.
Each probe solves the Fisher system once against streamed covariance noise.
`selected_rows` (at least one and at most `min(256, batch_rows)` rows) are
compared with exact Fisher solves. Each probe's squared projection is the
exact variance times an independent chi-square(1) draw, so with exact solves
`probes * estimate / exact` is chi-square with `probes` degrees of freedom.
The call converts that pivot to the standard normal deviate with the same tail
probability (`calibration.equivalent_deviates`) and raises when any absolute
deviate exceeds `max_calibration_standard_errors` (default 5, a two-sided
level of 5.7e-7 per row). The sample Monte Carlo standard error
(`calibration.monte_carlo_se`, and `batch.monte_carlo_se` for predictions) is
a diagnostic; it is not used for the decision, because it is small exactly when
the estimate is low. It describes sampling variation only. It does not bound iterative-solve bias, which is reported
separately through the PCG residuals, and calibration on selected rows is
not a guarantee for other rows. Unknown-scale families need an explicit
`scale`. `predict(se_fit=True)` never uses this approximation.

### Extended Fellner--Schall fitting

Select the dense extended Fellner--Schall controller explicitly with
`optimizer="efs"`. It follows the pinned mgcv 1.9-3 `efsudr` update and keeps
its controls inside the composed `FitControl` object:

```python
from jaxgam import EFSControl, FitControl, GAM

result = GAM(
    "y ~ s(x1) + s(x2)",
    family="poisson",
    optimizer="efs",
    control=FitControl(efs=EFSControl(history_limit=8)),
).fit(data)
```

For replayable input, request the compact prediction result and QR solver:

```python
prediction = GAM(
    "y ~ s(x1, bs='cr') + s(x2, bs='cr')",
    family="poisson",
    optimizer="efs",
    control=FitControl(execution="stream", linear_solver="qr"),
).fit(source, result="prediction")
```

EFS supports dense input and replayable `RowSource` input for the built-in
regular-family link constructors and negative binomial log, identity, and
square-root links within their tested response/start domains. The streamed
route is selected with `FitControl(execution="stream", linear_solver="qr")`
and currently returns `result="prediction"`. Its preparation path supports
cubic regression (`bs="cr"`) and cubic shrinkage (`bs="cs"`) smooths. Other
result modes use dense execution. Prediction results retain selected total EDF
and may retain Fisher uncertainty for batched prediction without retaining the
training source.

Estimated negative-binomial identity and square-root fits require an offset
whose mean at the zero-coefficient recovery anchor is positive. A zero offset
produces a zero mean for these links, so both execution routes reject that
input just as pinned mgcv does. Explicit fixed smoothing parameters and models
without penalties keep their existing fixed/Newton routes. Gaussian compression
applies to the Newton or fixed-smoothing Gaussian route.

Poisson logit, probit, and complementary-log-log are constructor extensions,
not advertised Poisson links. Their integer-response NULL initialization
leaves the bounded link domain when any response is positive, while an
all-zero response has a nonfinite null-link mean. The public streamed route
therefore preserves the pinned default-start rejection; the internal
matched-start evidence remains recorded separately and is not presented as a
public default-start capability.

Every EFS result mode retains `optimizer_diagnostics`, an immutable compact
record containing the pinned profile, exact trace and step policies, stop
reason, outer, inner, and theta iteration counts, bounded accepted score/scale
histories, final multiplier/update residual, movement maxima, and numerical
event flags. Streamed EFS diagnostics additionally separate startup scans from
scans made by every executed outer trial. These counts exclude Phase-1 model
preparation.
Ordinary Newton and fixed-smoothing results set this field to `None`.

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

## Discrete execution

`FitControl(execution="discrete")` fits on mgcv's compact discretized design
(`bam(discrete=TRUE)`) from a replayable `RowSource`, under an explicit
`ContinuousDiscretizationControl` grid policy and memory budget, and returns a
row-free prediction result (`result="prediction"`). Smoothing parameters are
either fixed with `sp=` or selected with
`smoothing_strategy="working_model_reml"`, pinned `bgam.fitd`'s performance
iteration (one limited `Sl.fitChol` Newton proposal per working model). This
is a different fitting method from the dense route's nested exact REML.

| Family | Links | Working model | Scale |
|---|---|---|---|
| Gaussian | identity | one frozen `X'WX`/`X'Wy` | estimated |
| Poisson | log, identity, sqrt | Fisher IRLS weights re-formed each iteration | 1 |
| Binomial | logit, probit, cloglog, log | Fisher IRLS weights re-formed each iteration | 1 |
| Gamma | inverse, identity, log | Fisher IRLS weights re-formed each iteration | estimated (`log.phi`) |
| Negative binomial | log, identity, sqrt | `dDeta` Newton weights re-formed each iteration; theta fixed or re-estimated by `estimate.theta` | 1 |

Poisson, Binomial and Gamma use Fisher weights for every link (there is no
observed-curvature split); all non-Gaussian families halve a coefficient step
while the raw penalized `sum(dev.resids)` rises and converge on the penalized
deviance change. An estimated negative-binomial theta is updated once per
iteration conditionally on the current mean (it is not jointly optimized with
the smoothing parameters); the result reports the final `theta`.

Prior weights and the offset are owned by the `RowSource`. Pinned `bam`'s
discretization treats a supplied `weights=` or `offset=` argument as one more
model-frame column, whatever its values (all-ones weights included), and that
column changes the seeded draws that pad short covariate grids, and so the
smooths' centering constraints. A source states what it supplies through
`has_explicit_weights` and `has_explicit_offset`. The built-in sources
(`ArrayRowSource`, `DataFrameRowSource`, `MemmapRowSource`) return `False`
when constructed with `weights=None` or `offset=None`. A source that does not
declare an attribute is treated as supplying it, exactly as `bam` treats a
supplied `weights=` or `offset=`; a caller-written source may declare `False`
for either.

As on the streamed routes, `source_scans` and `batches_scanned` count the
source scans and batches read by the fit itself (Phase-1 setup scans are
excluded) and `known_workspace_bytes` is the conservative known
fit-workspace bound: here the indexed design (host tables and device
actions) and the solver workspace. `fit_peak_bytes` adds the retained
compact setup tables; it is the bound the discrete route checks against
`memory_budget_bytes` (reported as `fit_memory_budget_bytes`).

```python
from jaxgam import GAM, ContinuousDiscretizationControl, FitControl
from jaxgam.data.source import DataFrameRowSource

control = FitControl(
    execution="discrete",
    smoothing_strategy="working_model_reml",
    discretization=ContinuousDiscretizationControl(m=30),
)
source = DataFrameRowSource(frame, response="y", weights=w, offset=off)
fit = GAM("y ~ s(x) + s(z)", family="poisson", control=control).fit(
    source, result="prediction"
)
fit.predict(newdata, pred_type="response")
```

::: jaxgam.control.FitControl

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
EFS fits additionally retain the compact `optimizer_diagnostics` record
described above; the field is `None` for ordinary and fixed-smoothing fits.

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
| `optimizer_diagnostics` | `EFSOptimizerDiagnostics \| None` | Compact bounded EFS controller trace, or `None` for other routes |
