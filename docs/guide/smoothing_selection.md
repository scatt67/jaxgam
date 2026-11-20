# Smoothing parameter selection

Each smooth term carries one or more smoothing parameters that set how wiggly
it may be. This page covers the ways jaxgam chooses them, or lets you fix
them, when your data are an in-memory DataFrame (the dense route).

**Use this page when** you fit from a DataFrame and want to control how
smoothing parameters are chosen.

**Do not use it when** your data come from a row source: the streamed,
discretized and operator routes have their own rules (see
[Streamed fitting](streamed.md), [Discretized fitting](discretized.md) and
[Operator fitting](operator.md)).

The four choices:

| Choice | How you ask for it | `lambda_strategy` |
|---|---|---|
| Exact REML by Newton's method (default) | `GAM(formula)` | `"newton_reml"` |
| Fixed smoothing parameters | `GAM(formula, sp=[...])` | `"fixed"` |
| Extended Fellner-Schall | `GAM(formula, optimizer="efs")` | `"efs_reml"` |
| Gaussian QR compression | `GAM(formula, control=FitControl(gaussian_compression=True))` | `"newton_reml"`, or `"lbfgs_reml"` above 50 penalties |

All examples on this page use this setup block:

```python
import numpy as np
import pandas as pd
from jaxgam import GAM, EFSControl, FitControl

rng = np.random.default_rng(1)
n = 400
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
y = np.sin(2 * np.pi * x) + 0.5 * z + rng.normal(0, 0.3, n)
counts = rng.poisson(np.exp(0.5 + np.sin(2 * np.pi * x)))
data = pd.DataFrame({"x": x, "z": z, "y": y, "counts": counts})
```

## Default: exact REML by Newton's method

With no `sp=` and the default `optimizer="newton"`, jaxgam minimizes the
restricted maximum likelihood (REML) criterion over the log smoothing
parameters by Newton's method, refitting the coefficients by penalized
iteratively reweighted least squares at each trial. This is the estimate of
mgcv's `gam(method="REML")`.

```python
fit = GAM("y ~ s(x) + z").fit(data)
print(fit.converged, fit.n_iter, fit.convergence_info)
print(fit.lambda_strategy, fit.smoothing_params)
```

```text
True 7 full convergence
newton_reml [0.01101987]
```

**What you get back.** `converged` is `True` when the Newton iteration met
its tolerance. `convergence_info` is `"full convergence"`, `"step failed"`,
`"iteration limit"` or `"non-finite REML score or gradient"`, sometimes with
a note appended (for example that an inner fit did not converge, or that a
small diagonal jitter was added to a factorization, which also issues a
`RuntimeWarning`). `n_iter` counts outer Newton iterations. `score` is the
REML criterion and `smoothing_params` the selected values. For Gaussian and
Gamma models `scale` is estimated; for `family="nb"` theta is estimated and
reported in `theta` (only when the model has at least one smooth; otherwise
theta stays at its starting value and `theta` is `None`).

The Newton iteration has no public settings: its iteration limit and
tolerances are fixed.

## Fixed smoothing parameters with `sp=`

Pass one positive value per penalty, on the natural (not log) scale. The
order is the order of the penalties in the formula: smooth terms left to
right, each term's penalties together. A `te()` term has one penalty per
margin, so `te(x, z)` takes two values.

```python
fixed = GAM("y ~ s(x) + z", sp=[0.01]).fit(data)
print(fixed.lambda_strategy, fixed.smoothing_params)

fixed_counts = GAM("counts ~ s(x)", family="poisson", sp=[0.01]).fit(data)
print(fixed_counts.n_iter, fixed_counts.convergence_info)

tensor = GAM("y ~ te(x, z)").fit(data)
print(len(tensor.smoothing_params))
```

```text
fixed [0.01]
0 fixed sp
2
```

`smoothing_params` reports the values exactly as you supplied them.
For Poisson, Binomial and fixed-theta negative binomial models a fixed `sp`
means one coefficient fit (`n_iter=0`, `convergence_info="fixed sp"`). For
Gaussian and Gamma models the scale is still estimated, and for
`family="nb"` theta is still estimated, so an outer iteration still runs over
those parameters with the smoothing parameters held, as in mgcv.

A supplied `sp` takes precedence over `optimizer`: `GAM(formula, sp=...,
optimizer="efs")` is a fixed-`sp` fit.

**Limits.** A wrong count or a non-positive value raises `ValueError`:

```text
ValueError: sp has 2 elements but model has 1 penalty terms.
ValueError: sp must contain strictly positive, finite values (smoothing parameters are on the natural scale and are log-transformed internally); got invalid value(s) [0.0].
```

Per-term `sp=` inside `s()`, `te()` or `ti()` is not supported and raises
`NotImplementedError`; use the model-level `GAM(formula, sp=[...])`.

## Extended Fellner-Schall (`optimizer="efs"`)

The extended Fellner-Schall method updates each smoothing parameter by a
closed-form step instead of a Newton step. It follows mgcv's
`gam(method="REML", optimizer="efs")` (the routine `efsudr`).

**Use it when** you want mgcv's `optimizer="efs"` results.

**Do not use it when** you need a fixed `sp` (it is then ignored), or when
the family or link is outside the list below.

```python
efs = GAM("y ~ s(x) + z", optimizer="efs").fit(data)
print(efs.converged, efs.n_iter, efs.convergence_info, efs.lambda_strategy)
print(efs.smoothing_params)
d = efs.optimizer_diagnostics
print(d.stop_reason, d.outer_iterations, len(d.accepted_score_history))
```

```text
True 4 deviance_change efs_reml
[0.01101975]
deviance_change 4 4
```

Settings go in `FitControl(efs=EFSControl(...))`:

| Option | Default | What it does |
|---|---|---|
| `outer_limit` | `200` | Maximum outer iterations. Stopping on the last one is reported as an iteration limit. |
| `log_lambda_max` | `15.0` | Cap on trial log smoothing parameters. |
| `score_tolerance` | `0.1` | Stop when the last four REML scores differ by less than this and the largest proposed log step is below 0.05 (after iteration 3). |
| `pirls_tolerance` | `1e-7` | Inner coefficient-fit tolerance; also stop when the deviance changes by less than `100 * pirls_tolerance` relative. |
| `pirls_max_iter` | `200` | Maximum inner coefficient-fit iterations. |
| `history_limit` | `200` | Longest score history kept in the diagnostics. |

```python
short = GAM(
    "counts ~ s(x)",
    family="poisson",
    optimizer="efs",
    control=FitControl(efs=EFSControl(outer_limit=2)),
).fit(data)
print(short.converged, short.convergence_info)
```

```text
False iteration limit reached
```

**Families and links on the dense route.** Gaussian and Gamma (scale
estimated), Poisson and Binomial with any link, and the negative binomial
with log, identity or sqrt link, theta fixed or estimated.

**What you get back.** `converged` is `True` when the fit stopped on
`"score_window"` or `"deviance_change"`; `convergence_info` holds that stop
reason, or `"iteration limit reached"`. `n_iter` is the number of outer
iterations. `optimizer_diagnostics` is an `EFSOptimizerDiagnostics` with,
among others, `stop_reason`, `outer_iterations`, `inner_iterations`,
`theta_iterations`, `accepted_score_history` and, for Gaussian and Gamma,
`accepted_score_phi_history`. For a negative binomial with fixed theta,
`theta` is the fixed value (a Newton fit reports `None` there).

**Limits.** These raise:

| Case | Error |
|---|---|
| Another family or link (for example `NegativeBinomial(link="probit")`) | `NotImplementedError: optimizer='efs' does not support this known-scale family/link` (or `unknown-scale`) |
| Estimated-theta negative binomial with identity or sqrt link and no offset that makes the zero-coefficient mean positive | `ValueError: Estimated NB identity/sqrt EFS has no valid default zero-coefficient recovery anchor for this offset; mgcv rejects the same input. Supply an offset whose zero-coefficient means are positive.` |
| Rank-deficient penalized model | `ValueError` (for example `EFS known-scale path requires an identifiable penalized system`) |
| Prior weights outside 1e-10 to 1e10 (zero weights are allowed only for Gaussian with log or inverse link) | `ValueError` (for example `EFS known-scale path requires finite positive prior weights inside PIRLS clipping bounds`) |
| Fixed-theta negative binomial with responses strictly between 0 and 1 | `ValueError: Fixed-theta NB EFS does not support fractional responses below one` |

A model without penalties skips Fellner-Schall and is fitted directly.

## Gaussian QR compression (`gaussian_compression=True`)

For a Gaussian model with identity link, jaxgam can reduce the weighted
design to one QR factorization and then run every REML trial on
coefficient-sized arrays instead of the rows.

**Use it when** the model is Gaussian with identity link and has at least
one smooth.

**Do not use it when** you want Fellner-Schall: the request is then not
applied.

```python
qr = GAM("y ~ s(x) + z", control=FitControl(gaussian_compression=True)).fit(data)
print(qr.execution_route, qr.execution_fallback_reason, qr.smoothing_params)

ineligible = GAM(
    "counts ~ s(x)",
    family="poisson",
    control=FitControl(gaussian_compression=True),
).fit(data)
print(ineligible.execution_route, "|", ineligible.execution_fallback_reason)
```

```text
gaussian_qr None [0.01101987]
dense | Gaussian QR requires the exact Gaussian family with identity link
```

**Eligibility.** The family is exactly `Gaussian` with identity link, the
model has at least one penalty, the coefficients are identifiable (no rank
deficiency), the data are finite and every prior weight is between 1e-10 and
1e10. It also works with a fixed `sp`.

**How to tell whether it was used.** An ineligible request never raises:
the ordinary dense fit runs, `execution_route` is `"dense"` and
`execution_fallback_reason` holds the reason. Possible reasons:

- `Gaussian QR requires the exact Gaussian family with identity link`
- `Gaussian QR smoothing optimization requires at least one penalty`
- `Gaussian QR currently requires a full identifiable coefficient map`
- `Gaussian QR requires finite dense fitting data`
- `Gaussian QR currently requires unclipped finite positive weights`
- `Gaussian QR compression requires Newton smoothing selection.` (with
  `optimizer="efs"` and no `sp`)

When it is used, `execution_route` is `"gaussian_qr"`.

**Many penalties.** Up to 50 penalties the same Newton iteration runs on the
compressed problem (`lambda_strategy="newton_reml"`). Above 50 it switches to
SciPy's L-BFGS-B (`lambda_strategy="lbfgs_reml"`). The tests cover this
switch only just above the threshold, at 51 penalties. There, where the REML
criterion is flat, L-BFGS-B can stop at smoothing parameters far from the
Newton fit's while the score and fitted values agree closely; compare
`score` and predictions, not `smoothing_params`, if you check one against
the other.

## Only REML is available

`method="REML"` is the default and the only accepted method. `method="ML"`
raises at construction:

```python
try:
    GAM("y ~ s(x)", method="ML")
except NotImplementedError as error:
    print(type(error).__name__, str(error).split(". ")[0])
```

```text
NotImplementedError method='ML' is not supported
```

The full message is `method='ML' is not supported. mgcv's ML criterion uses
the penalty range-space projection of log|X'WX+S| (the C routine MLpenalty1
in gdi.c), which differs from REML's full-space determinant. Only
method='REML' is available; ML is deferred until the range-space determinant
is implemented.` Any other method, such as GCV, raises `ValueError`.

Also rejected at construction with `NotImplementedError`: `select=True`,
`gamma` other than 1.0 and `knots=`. In the formula, `fx=TRUE`, per-term
`sp=` and `id=` raise `NotImplementedError`. `optimizer` accepts only
`"newton"` and `"efs"`.

## Related pages

- [Choosing a route](choosing_a_route.md)
- [Streamed fitting](streamed.md) for exact REML, Fellner-Schall and Gaussian
  QR compression on a row source
- [Prediction results](prediction_results.md)
- API reference: [GAM](../api.md#gam), [FitControl](../api.md#fit-control),
  [Families](../api.md#families)
