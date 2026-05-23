# Gaussian Process Smooth (`bs="gp"`) Implementation Design

- **Status:** Proposed
- **Design Date:** 2026-05-23
- **Target Branch:** `add-gaussian-process-smooth`

---

## Table of Contents

1. [Overview](#1-overview)
2. [R Reference Implementation Analysis](#2-r-reference-implementation-analysis)
3. [Mathematical Specification](#3-mathematical-specification)
4. [Knot Selection and Cardinality](#4-knot-selection-and-cardinality)
5. [Smooth Class Design](#5-smooth-class-design)
6. [Formula Parsing](#6-formula-parsing)
7. [Constraint Pipeline Integration](#7-constraint-pipeline-integration)
8. [Penalty Construction](#8-penalty-construction)
9. [Prediction](#9-prediction)
10. [Numerical Considerations](#10-numerical-considerations)
11. [File Plan](#11-file-plan)
12. [Testing Strategy](#12-testing-strategy)
13. [Implementation Plan](#13-implementation-plan)

---

## 1. Overview

### 1.1 What Is a Gaussian Process Smooth?

In mgcv, `s(x, bs="gp")` constructs a **low-rank Gaussian process** smooth: a
reduced-rank kriging model whose basis is built from the leading eigenvectors
of a correlation matrix evaluated on the predictor values. The penalty is the
diagonal of the corresponding eigenvalues, so the smooth is mathematically a
finite-rank approximation to a full GP regression with a chosen covariance
kernel.

This is the construction of Kammann & Wand (2003) using the eigen
approximation method of Wood (2003), the same low-rank trick that powers TPRS
(`bs="tp"`). The key differences from TPRS are:

| Aspect | TPRS (`bs="tp"`) | GP (`bs="gp"`) |
|---|---|---|
| Radial basis | `r^{2m-d} log(r)` (TPS semi-kernel) | Correlation kernel `K(r/ρ)` |
| Null space | All monomials of degree `< m` | Intercept (+ linear trend, optional) |
| User-facing parameterization | Single integer `m` (penalty order) | Explicit kwargs: `kernel=`, `rho=`, `power=`, `stationary=` — JaxGAM does **not** accept mgcv's `m=`. See §1.6. |
| Implementation | Substantial C code (`src/tprs.c`) | Pure R, ~200 lines |
| Stationarity flag | N/A | Yes (`stationary=True` ⇒ no linear trend; mgcv encodes this as `m[1] < 0`) |

The smoothing parameter λ plays its usual role: large λ shrinks toward the
unpenalized null space (mean / linear trend), small λ allows the GP component
to fit local structure.

### 1.2 Use Cases

GP smooths support **isotropic** smoothing of one or more continuous
predictors with a user-chosen correlation structure:

| Formula | Dim | Kernel | Use case |
|---|---|---|---|
| `s(x, bs="gp")` | 1 | Matérn ν=3/2, auto ρ | Univariate smooth with default kernel |
| `s(x, z, bs="gp", k=50)` | 2 | Matérn ν=3/2, auto ρ | 2D spatial smoothing |
| `s(x, z, bs="gp", kernel="spherical", rho=0.5)` | 2 | Spherical, ρ=0.5 | Geographic kriging with known range |
| `s(x, bs="gp", kernel="power_exponential", rho=1.0, power=2.0, stationary=True)` | 1 | Stationary squared-exponential, ρ=1 | True stationary GP regression |
| `s(x, z, bs="gp", kernel="matern_3_2", rho=0.3)` | 2 | Matérn 3/2 with explicit ρ | Reproducible kernel choice |
| `te(x, z, bs="gp", kernel="matern_3_2", k=5)` | 2 (tensor) | Per-margin Matérn 3/2 | Anisotropic tensor GP |

The five supported kernels (`kernel ∈ {"spherical",
"power_exponential", "matern_3_2", "matern_5_2", "matern_7_2"}`) give
users a substantive choice of smoothness behavior, from compactly
supported (spherical) to infinitely differentiable
(`kernel="power_exponential", power=2.0` — the squared-exponential /
Gaussian kernel). See §1.6 for the full kwarg surface and §6.4 for the
mgcv ↔ JaxGAM mapping.

### 1.3 Scope

**In scope — both mgcv GP pathways:**

1. **Direct GP smooth** via `s(..., bs="gp")` — a single
   `GaussianProcessSmooth` instance covering 1-D, 2-D, and 3-D joint
   covariate spaces:
   - One joint Euclidean distance over `(x_1, …, x_d)`.
   - One knot set in the joint covariate space.
   - One `nk × nk` covariance matrix `E`.
   - One truncated eigendecomposition.
   - One penalty `S` (diagonal of top-`k` eigenvalues), one
     smoothing parameter `λ`.
2. **GP tensor margins** via the existing tensor wrappers
   `te(..., bs="gp")` and `ti(..., bs="gp")` —
   `TensorProductSmooth._create_marginals()` at `tensor.py:136-148`
   already iterates `spec.variables` and instantiates the registered
   margin class per variable, so once `"gp"` lands in
   `smooth_registry` this works without a separate GP-tensor class:
   - One **univariate** `GaussianProcessSmooth` margin per variable.
   - Tensor-product basis from the row-tensor of marginal bases.
   - One marginal penalty per variable, **multiple smoothing
     parameters** (one per margin).

   No GP-specific code in `tensor.py`. No `bs == "gp"` guard.
   `GaussianProcessSmooth` must therefore work correctly as a
   univariate margin, which is the same code path as the 1-D direct
   smooth.

Other in-scope items (apply to both pathways):
- All five mgcv kernels: spherical, power exponential, Matérn ν=3/2,
  Matérn ν=5/2, Matérn ν=7/2 (selected via the `kernel="..."` kwarg —
  see §1.6 for the full name list and aliases).
- Stationary vs non-stationary modes (selected via the `stationary=`
  bool kwarg).
- Auto-selected `ρ` (Kammann–Wand default = max pairwise distance) and
  user-supplied `ρ` via the `rho=` kwarg.
- `power=` kwarg for `kernel="power_exponential"` (default `1.0`;
  `2.0` gives squared-exponential).
- `xt` argument with `max_knots` (default 2000) and `seed` (default 1) for
  reproducible knot subsampling.
- Random-sampled knot selection via a seeded `np.random.RandomState`
  (legacy API, single helper `_subsample_knots`). Knot sets do **not**
  match mgcv bit-for-bit when subsampling fires — see §10.6.
- Truncated symmetric eigendecomposition (top-`k` largest-magnitude
  eigenvalues, matching R's `slanczos(E, k, -1)`).
- Standard sum-to-zero centering constraint via `smoothCon` pipeline.
- Prediction at arbitrary new points using stored `shift`, `knt`, `UZ`,
  `gp.defn`.
- R comparison tests at smooth-construct level and in the validation
  matrix for **both** direct and tensor pathways.

**Direct vs tensor GP — not redundant.** A direct multivariate GP
`s(x, z, bs="gp")` and a tensor-margin GP `te(x, z, bs="gp")` are
different basis constructions producing different fits:

| Aspect | `s(x, z, bs="gp")` (direct) | `te(x, z, bs="gp")` (tensor) |
|---|---|---|
| Distance | Joint Euclidean over `(x, z)` | Per-margin (1-D) Euclidean per variable |
| Knots | One joint set in 2-D | One set per margin (1-D each) |
| Basis | Single low-rank kriging basis | Row-tensor product of two 1-D bases |
| Penalty | One diagonal `S` | Two marginal penalties, embedded as tensor sums |
| Smoothing params | One `λ` | Two `λ_j`, one per margin |
| Anisotropy | Isotropic (single `ρ`) | Naturally anisotropic (per-margin `ρ`) |

Validation-matrix coverage keeps **both** as separate configs
(`gp_2d` direct vs `gp_te` tensor) — they are not interchangeable.

**`k` argument convention for tensor margins.** JaxGAM's current
`TensorProductSmooth._create_marginals()` passes the *same scalar*
`spec.k` to each margin (`tensor.py:140`). That matches existing
tensor behavior for `te(x, z, bs="tp", k=5)` (each TPRS margin gets
`k=5`). R's `te()` accepts `k=c(k_1, k_2)`; for R-parity tests, when
Python writes `te(x1, x2, bs='gp', k=5)`, R writes
`te(x1, x2, bs='gp', k=c(5, 5))`. Per-margin distinct `k` is a
future tensor-wrapper enhancement, not a GP-specific issue.

**Out of scope (deferred):**
- Anisotropic kernels *within a single* GP smooth (per-dimension `ρ`
  inside one `_gp_E` call) — mgcv only supports scalar `ρ`. Use a
  tensor product if you need per-direction range tuning.
- Custom user-defined kernels — only the five mgcv kernels.
- `fx=TRUE` (unpenalized GP) — mgcv supports this generically; we inherit
  whatever the framework provides.
- Sparse covariance approximations (predictive process, inducing points
  beyond knot sampling) — deferred to v1.1 / sparse-CPU path.

### 1.4 Reuse of Existing Infrastructure

A close audit of `jaxgam/smooths/` reveals **substantial overlap** between
GP and the existing TPRS implementation. Both are reduced-rank kriging-
style constructions with knot harvesting, pairwise distance evaluation,
truncated eigendecomposition, and centered covariates. The bulk of that
shared code lives in `jaxgam/smooths/tprs.py` today, and porting GP
naively would either duplicate it or force a cross-smooth import.

**The plan is a small, focused refactor (PR 0)**: extract the shared
pieces to `jaxgam/smooths/utils.py`, which already houses cross-smooth
helpers (`row_tensor`, `interaction_matrix`, factor detection). Both TPRS
and GP then import from `utils.py`. This avoids `gp → tprs` imports and
matches the convention used by `random_effects.py` and `tensor.py`.

#### 1.4.1 Pieces to extract from `tprs.py` to `utils.py`

| Symbol | Location in `tprs.py` | Why it's shared |
|---|---|---|
| `_slanczos_jit` | lines 278-406 | Byte-faithful port of R's `Rlanczos`. GP calls `slanczos(E, k, -1)` — identical signature. |
| `_slanczos` | lines 409-442 | Public wrapper for the above. |
| `_compute_distance_matrix` | lines 232-252 | Pairwise Euclidean distance via broadcasting. GP's `gpE` opens with literally the same computation. |
| `_get_unique_rows` | lines 255-275 | `np.unique(X, axis=0, return_inverse=True)`. GP harvests unique covariate combinations the same way. |
| `_subsample_knots` (new) | inlined in TPRS `setup()` lines 627-639 | Knot-subsample-by-seed logic. TPRS hardcodes `seed=1`; GP needs a parameterized version (`xt$seed` defaults to 1 but is user-settable). Extract with a `seed` parameter so TPRS keeps `seed=1` semantics and GP threads in `xt["seed"]`. |

#### 1.4.2 Why these in particular (not the rest of `tprs.py`)

| Symbol | Stays in `tprs.py` | Reason |
|---|---|---|
| `_null_space_basis_r` / `_null_space_basis_r_jit` | ✓ | Right-Householder QR for `T.T @ U_k` null space. GP does *not* use this — its null space is the explicit polynomial basis (`gpT`: intercept + linear trend), not the QR complement of a projection. |
| `tps_semi_kernel`, `eta_const` | ✓ | TPS-specific radial basis `r^{2m-d} log(r)`. GP uses entirely different kernels (Matérn, spherical, etc.). |
| `compute_polynomial_basis` (+ helpers `_fill_polynomial_basis`, `_monomial_indices`, `_partitions`) | ✓ | Generates *all* monomials of total degree < `m`. GP's `gpT` only needs `[1]` (stationary) or `[1, x_1, …, x_d]` (non-stationary) — a 5-line direct construction. Importing this for two columns of output is overkill. |
| `_default_k` | ✓ | TPRS-specific table (`{1: 10, 2: 30}`). GP has its own table (`d + 1 + def.k[d]` where `def.k = (10, 30, 100)`). Different semantics. |

#### 1.4.3 Already-available shared infrastructure (no work needed)

GP also inherits from the broader smooth scaffolding without any changes:

| Symbol | Source | Why GP uses it as-is |
|---|---|---|
| `Smooth._smoothcon_normalize` | `base.py:54-84` | `s_scale = ||S||_1 / ||X||_∞²` — GP applies this identically to its diagonal penalty. |
| `Smooth._require_setup` | `base.py:48-51` | Standard setup-guard pattern. |
| `Penalty(S, rank, null_space_dim)` | `penalties/penalty.py` | Diagonal penalty is just a `Penalty` instance. |
| `apply_sum_to_zero()` in `constraints.py` | `constraints.py` | GP participates in normal centering (unlike RE which opts out). |
| `_svd_reparameterize` in `tensor.py` | `tensor.py:52-123` | **Exercised** when GP appears as a tensor margin (`te(..., bs="gp")` / `ti(..., bs="gp")`, in scope per §1.3). `TensorProductSmooth._create_marginals()` at `tensor.py:136-148` builds each margin via the registry, so once `"gp"` is registered the SVD reparameterization applies unchanged. `GaussianProcessSmooth` therefore keeps `_noterp = False` (the `Smooth` default) so this code path is not blocked. |
| Factor helpers in `utils.py` (`is_factor`, `get_factor_levels`, `interaction_matrix`) | `utils.py` | GP operates on continuous predictors only; these aren't called. |
| Shrinkage utilities in `base.py` (`_decompose_and_replace`, etc.) | `base.py:94-157` | Not needed for v1.0 GP. A future "shrinkage GP" (analog of `bs="cs"`) could reuse them. |

#### 1.4.4 Byte-faithfulness of `_slanczos`

The Lanczos solver matters most because it is the single most numerically
delicate piece of TPRS (multiple PRs went into matching R exactly):

- Uses **R's exact deterministic LCG starting vector** (`jran = (jran * 106
  + 1283) % 6075`, matching `mgcv/src/mat.c` line-for-line).
- Performs **double reorthogonalization** (matching R's CGS implementation).
- Implements **R's convergence-check schedule** (`f_check = max(k // 2, 10)`).
- Returns the **top-`k` largest-magnitude eigenvalues** — exactly what
  mgcv's GP constructor calls (`slanczos(E, k, -1)`).

Because the same deterministic starting vector is used, GP's eigenvectors
will match R's *up to sign* in the same way TPRS's already do — so the
same coefficient-vs-fitted-values comparison strategy used for TPRS
applies to GP unchanged.

#### 1.4.5 Bottom line

GP's setup() can be written as:

```python
from jaxgam.smooths.utils import (
    _slanczos,
    _compute_distance_matrix,
    _get_unique_rows,
    _subsample_knots,
)
```

…plus a ~40-line GP-specific `_gp_E` kernel evaluator and a ~10-line
`gpT` null-space builder. The remaining wiring (smoothcon normalization,
penalty embedding, constraint pipeline) is fully covered by `base.py`
and the existing constraint machinery. **PR 0 prep is the load-bearing
work**; PRs 1-2 are then assembly of pre-tested components.

### 1.5 Dense-Only Constraint

Per CLAUDE.md, v1.0 is dense-only. For GP smooths the dominant memory cost
is the `nk × nk` knot-knot covariance matrix `E` (where `nk ≤ max.knots =
2000`). At `nk = 2000` this is a `2000 × 2000` dense matrix (~32 MB for
float64). The truncated eigendecomposition only needs the top `k` eigenpairs
(`k` defaults to 12 for 1D, 33 for 2D), but a full Lanczos pass still works
on the dense `E`.

The prediction-time `n × nk` matrix `E(x_new, knt)` is also dense. mgcv
chunks by `nk` rows; **JaxGAM v1.0 does not chunk** — numpy handles a
single `n × nk` allocation comfortably up to ~10⁸ entries. Chunking is
deferred to v1.1; revisit only if memory profiling forces it (§9.5).

### 1.6 Python-Native API (no mgcv `m=`)

mgcv overloads a single signed numeric vector
`m = c(sign·type, rho, power)` to encode kernel type, range, power, and
stationarity. This is dense and order-dependent — `m=c(-2, 1, 2)` reads
"stationary squared-exponential with ρ=1" only if you remember the
table. JaxGAM intentionally exposes clearer, named keyword arguments
instead.

**Public surface.** GP smooths take these kwargs through the formula's
`extra_args`:

| Argument | Type | Default | Effect |
|---|---|---|---|
| `kernel` | str | `"matern_3_2"` | One of `"spherical"`, `"power_exponential"`, `"matern_3_2"`, `"matern_5_2"`, `"matern_7_2"`. Aliases also accepted: `"power"`, `"matern32"`, `"matern52"`, `"matern72"`. |
| `rho` | float > 0 | auto (max pairwise knot distance) | Kernel range. |
| `power` | float in (0, 2] | `1.0` | Only used by `kernel="power_exponential"`. `2.0` ⇒ squared-exponential / Gaussian. |
| `stationary` | bool | `False` | If True, drop the linear-trend null space; keep only the intercept. |
| `xt` | dict | `{}` | `max_knots` (int, default 2000), `seed` (int, default 1). |

Passing `m=...` to a GP smooth raises `ValueError` from
`parse_gp_config`:

> `mgcv-style \`m=\` is not supported for JaxGAM GP smooths. Use`
> `\`kernel=\`, \`rho=\`, \`power=\`, and \`stationary=\` instead.`

This is a deliberate divergence from mgcv. Internally we convert to
mgcv's `m` only at the R-bridge boundary (see §11.2's
`gp_config_to_mgcv_m` helper), so R-parity tests still drive mgcv
through its native API end-to-end.

**Internal config object.** `parse_gp_config(extra_args)` returns a
frozen `GPConfig` dataclass that the smooth class stores as
`self._config`. The resolved `ρ` (the actual value used at fit time,
auto-derived or user-supplied) is stored separately as
`self._resolved_rho` because it is frozen at setup and reused for all
predictions.

```python
# jaxgam/smooths/gp_kernels.py

from dataclasses import dataclass
from enum import StrEnum


class GPKernelName(StrEnum):
    SPHERICAL = "spherical"
    POWER_EXPONENTIAL = "power_exponential"
    MATERN_3_2 = "matern_3_2"
    MATERN_5_2 = "matern_5_2"
    MATERN_7_2 = "matern_7_2"


@dataclass(frozen=True)
class GPConfig:
    kernel: GPKernelName = GPKernelName.MATERN_3_2
    stationary: bool = False
    rho: float | None = None      # None ⇒ auto (max pairwise distance)
    power: float = 1.0            # only consulted by power_exponential
```

**Kernel evaluation is delegated to kernel classes.** Each kernel is a
small class with `evaluate(e, *, power)` and `validate(power)`. The
module-level `gp_kernel_registry` is an instance of the existing
generic `jaxgam.registry.Registry[T]` (same abstraction used by
`smooth_registry`, `family_registry`, `link_registry`), so we get
case-insensitive lookups, runtime `register()` for user extension,
and instance caching for free:

```python
# jaxgam/smooths/gp_kernels.py

from abc import ABC, abstractmethod

import numpy as np

from jaxgam.registry import Registry


class GPKernel(ABC):
    @abstractmethod
    def evaluate(self, e: np.ndarray, *, power: float = 1.0) -> np.ndarray: ...

    def validate(self, power: float) -> None:
        """Default: ignore power. Overridden by PowerExponentialKernel."""


class SphericalKernel(GPKernel):
    def evaluate(self, e, *, power=1.0):
        return (1 - 1.5 * e + 0.5 * e ** 3) * (e <= 1)


class PowerExponentialKernel(GPKernel):
    def evaluate(self, e, *, power=1.0):
        return np.exp(-(e ** power))

    def validate(self, power):
        if not (0.0 < power <= 2.0):
            raise ValueError(
                f"GP power-exponential `power` must be in (0, 2], got {power!r}."
            )


class Matern32Kernel(GPKernel):
    def evaluate(self, e, *, power=1.0):
        return (1.0 + e) * np.exp(-e)


class Matern52Kernel(GPKernel):
    def evaluate(self, e, *, power=1.0):
        eE = np.exp(-e)
        return eE + (e * eE) * (1.0 + e / 3.0)


class Matern72Kernel(GPKernel):
    def evaluate(self, e, *, power=1.0):
        eE = np.exp(-e)
        return eE + (e * eE) * (1.0 + 0.4 * e + e ** 2 / 15.0)


gp_kernel_registry: Registry[GPKernel] = Registry(
    {
        "spherical":         SphericalKernel,
        "power_exponential": PowerExponentialKernel,
        "power":             PowerExponentialKernel,
        "matern_3_2":        Matern32Kernel,
        "matern32":          Matern32Kernel,
        "matern_5_2":        Matern52Kernel,
        "matern52":          Matern52Kernel,
        "matern_7_2":        Matern72Kernel,
        "matern72":          Matern72Kernel,
    },
    name="GP kernel",
    cache_instances=True,
)
```

Lookups go through `gp_kernel_registry.get_instance(name)` — which
returns a cached `GPKernel` instance, or raises `KeyError` with an
"Available: ..." message if the name is unknown. No bespoke wrapper
function is needed.

**Why kernel *classes* not a switch.** Five branches would be borderline
acceptable, but `PowerExponentialKernel.validate` (which enforces
`power ∈ (0, 2]`) is genuinely different from the others (which ignore
`power`). Putting validation on each kernel keeps it co-located with
`evaluate`. Adding a future kernel becomes "new class + new registry
row" instead of "edit the switch and the validation block".

**Why this registry, not direct imports.** `parse_gp_config` receives a
*string* from the formula (e.g. `kernel="matern_3_2"`); it needs a
runtime name → instance lookup. Reusing the existing `Registry[T]`
generic also documents the public name surface (canonical names +
aliases) in one place, gives us case-insensitive matching, and lets
users register custom kernels at runtime via `register()` exactly the
same way they would for smooths or families.

**Stationarity is not part of the kernel registry.** It controls the
null-space basis only (see §3.5 and §5.6). The kernel class itself
does not need to know about it.

---

## 2. R Reference Implementation Analysis

### 2.1 Source Location

| Component | Path | Lines |
|---|---|---|
| Constructor | `$MGCV_SOURCE/R/smooth.r` | 3441–3552 |
| Predict method | `$MGCV_SOURCE/R/smooth.r` | 3556–3595 |
| Kernel evaluator `gpE` | `$MGCV_SOURCE/R/smooth.r` | 3410–3439 |
| Null-space basis `gpT` | `$MGCV_SOURCE/R/smooth.r` | 3404–3408 |
| Documentation | `$MGCV_SOURCE/man/smooth.construct.gp.smooth.spec.Rd` | (full) |
| Lanczos backend | `$MGCV_SOURCE/R/mgcv.r` line 19 → `src/mat.c:3637` | — |
| RNG state guard | `$MGCV_SOURCE/R/misc.r` | 840–861 (`temp.seed`) |

There is **no C code dedicated to GP**. The implementation is pure R,
~196 lines total. The only C dependency is the generic `slanczos()` /
`C_Rlanczos` truncated symmetric eigensolver — and **we already have a
byte-faithful Numba-JIT port** of this in `jaxgam/smooths/tprs.py`
(`_slanczos` at line 409, backed by `_slanczos_jit` at lines 278-406).
As part of GP integration this code is **relocated to
`jaxgam/smooths/utils.py`** along with the other shared kriging-style
helpers (`_compute_distance_matrix`, `_get_unique_rows`, and a new
`_subsample_knots`), then consumed from both `tprs.py` and the new
`gaussian_process.py`. See §1.4 for the full audit and §5.4 for the
step-by-step refactor.

### 2.2 Constructor Algorithm

The full algorithm (annotated, paraphrased from `smooth.r:3441-3552`):

```r
smooth.construct.gp.smooth.spec <- function(object, data, knots) {
  # 1. Decide stationary vs non-stationary from m[1]
  if (is.na(object$p.order) || length(object$p.order) < 1) {
    stationary <- FALSE          # default: non-stationary (linear trend)
  } else {
    stationary <- object$p.order[1] < 0
  }

  # 2. Read xt for knot subsampling
  xtra$max.knots <- object$xt$max.knots %||% 2000
  xtra$seed      <- object$xt$seed      %||% 1

  # 3. Stack covariates into n x d matrix
  x <- do.call(cbind, data[object$term])      # n x d

  # 4. Harvest knots
  if (!is.null(knots)) {
    knt <- do.call(cbind, knots[object$term])
    if (nrow(knt) > nrow(x)) stop("more knots than data in an ms term")
  } else {
    xu <- uniquecombs(x, TRUE)               # unique covariate combinations
    if (nrow(xu) < object$bs.dim)
      stop("A term has fewer unique covariate combinations than ...")
    if (nrow(x) > xtra$max.knots && nrow(xu) > xtra$max.knots) {
      temp.seed(xtra$seed)                   # save/restore global RNG
      knt <- xu[sample(seq_len(nrow(xu)), xtra$max.knots), , drop = FALSE]
    } else {
      knt <- xu
    }
  }

  # 5. Centre covariates and knots
  object$shift <- colMeans(x)
  x   <- sweep(x,   2, object$shift)
  knt <- sweep(knt, 2, object$shift)

  # 6. Build correlation matrix E on knots
  E <- gpE(knt, knt, object$p.order)         # nk x nk, attr "defn" attached
  object$gp.defn <- attr(E, "defn")          # c(sign*type, rho, k_power)

  # 7. Resolve dimensions
  def.k <- c(10, 30, 100)
  dd <- ncol(knt)
  if (object$bs.dim[1] < 0)
    object$bs.dim <- ncol(knt) + 1 + def.k[dd]
  if (object$bs.dim < ncol(knt) + 2) {
    object$bs.dim <- ncol(knt) + 2
    warning("basis dimension reset to minimum possible")
  }
  object$null.space.dim <- if (stationary) 1 else ncol(knt) + 1
  k <- object$bs.dim - object$null.space.dim

  # 8. Truncated eigendecomposition of E (or skip if k >= nk)
  if (k < nrow(knt)) {
    er <- slanczos(E, k, -1)                  # top-k largest-magnitude
    D  <- diag(c(er$values, rep(0, object$null.space.dim)))
    UZ <- er$vectors                          # nk x k
  } else {
    D  <- matrix(0, object$bs.dim, object$bs.dim)
    D[1:k, 1:k] <- E
    UZ <- diag(k)
  }

  # 9. Store state
  object$S    <- list(D)
  object$UZ   <- UZ
  object$knt  <- knt
  object$df   <- object$bs.dim
  object$rank <- k

  # 10. Build training X by reusing the predict method
  class(object) <- "gp.smooth"
  object$X <- Predict.matrix.gp.smooth(object, data)
  object
}
```

Key observations:
1. **No `noterp`, `side.constrain`, `random`, or empty `C` flags** — GP
   smooths participate in the *normal* sum-to-zero centering and gam.side
   pipeline, unlike RE smooths.
2. **`p.order` (`m`) is multi-purpose**: `m[1]` encodes both kernel type
   (`|m[1]|`) and stationarity (`sign(m[1])`); `m[2]` is `ρ`; `m[3]` is
   the power for `type=2` power-exponential.
3. **`xt` is narrow**: only `max.knots` and `seed`. Kernel/range live in `m`.
4. **Knot selection is *random* sampling**, not max-min. RNG state is
   saved/restored via `temp.seed()` so the user's global RNG is undisturbed.
5. **Penalty is diagonal** — eigenvalues of `E` on the first `k` entries,
   zeros on the last `null.space.dim`. Indefinite for spherical and small-ρ
   power-exponential kernels.
6. **Training `X` is built by calling the predict method**, eliminating
   duplicate code.

### 2.3 Kernel Evaluator: `gpE`

`$MGCV_SOURCE/R/smooth.r:3410-3439`:

```r
gpE <- function(x, xk, defn = NA) {
  # 1. Build Euclidean distance matrix between x (n) and xk (nk)
  ind <- expand.grid(x = 1:nrow(x), xk = 1:nrow(xk))
  E <- matrix(sqrt(rowSums((x[ind$x, , drop = FALSE]
                          - xk[ind$xk, , drop = FALSE])^2)),
              nrow(x), nrow(xk))

  # 2. Decode defn = c(sign*type, rho, k_power)
  rho <- -1; k_pow <- 1; sign.type <- 1
  if (length(defn) < 1 || is.na(defn[1])) {
    type <- 3                                       # default Matern 3/2
  } else {
    type <- abs(round(defn[1]))
    sign.type <- sign(defn[1])
  }
  if (length(defn) > 1) rho   <- defn[2]
  if (length(defn) > 2) k_pow <- defn[3]

  # 3. Kammann-Wand default for rho: max pairwise distance
  if (rho <= 0) rho <- max(E)
  E <- E / rho

  if (!type %in% 1:5 || k_pow > 2 || k_pow <= 0)
    stop("incorrect arguments to GP smoother")

  # 4. Evaluate kernel (see table in Section 3.3)
  if (type > 2) eE <- exp(-E)
  E <- switch(type,
    (1 - 1.5*E + 0.5*E^3) * (E <= 1),  # 1 Spherical
    exp(-E^k_pow),                      # 2 Power exponential
    (1 + E) * eE,                       # 3 Matern 3/2
    eE + (E*eE) * (1 + E/3),            # 4 Matern 5/2
    eE + (E*eE) * (1 + 0.4*E + E^2/15)  # 5 Matern 7/2
  )

  attr(E, "defn") <- c(sign.type * type, rho, k_pow)  # round-trip resolved defn
  E
}
```

Critical details for porting:

- **Distance is isotropic Euclidean** over all `d` columns. No per-dimension
  scaling (mgcv leaves this to the user).
- **`rho ≤ 0` triggers auto-selection** to `max(E)` (= max pairwise
  distance, after the `/ rho` step using the raw distances).
- **`type=2` validity guard**: `k_pow ∈ (0, 2]`. `k_pow=1` is exponential,
  `k_pow=2` is squared-exponential / Gaussian.
- **`defn` attribute is round-tripped**: `c(sign(m[1])*type, ρ_resolved, k_pow)`
  is attached to `E` and stored as `object$gp.defn`. Predictions reuse this,
  so the *training-time* `ρ` is what the predict method uses — never a
  recomputed `max(E)` over the new data.

### 2.4 Null-Space Basis: `gpT`

```r
gpT <- function(x, defn) {
  if (defn[1] < 0) {
    x[, 1] * 0 + 1            # stationary: only intercept (1 column)
  } else {
    cbind(x[, 1] * 0 + 1, x)  # non-stationary: intercept + linear trend
  }
}
```

The null space is the unpenalized parametric part:
- **Stationary** (`m[1] < 0`): just the intercept (`null.space.dim = 1`).
- **Non-stationary** (`m[1] > 0` or default): intercept + each covariate
  as a linear column (`null.space.dim = d + 1`).

Note that this is *centered* `x` (after subtracting `shift`), so the linear
columns have zero mean.

### 2.5 Predict Method

`$MGCV_SOURCE/R/smooth.r:3556-3595`:

```r
Predict.matrix.gp.smooth <- function(object, data) {
  nk <- nrow(object$knt)
  x  <- do.call(cbind, data[object$term])        # n x d
  x  <- sweep(x, 2, object$shift)                # apply training shift

  if (nrow(x) > nk) {
    # Chunked construction to limit RAM
    n.chunk <- nrow(x) %/% nk
    for (i in 1:n.chunk) {
      ind <- 1:nk + (i - 1) * nk
      Xc  <- gpE(x[ind, , drop = FALSE], object$knt, object$gp.defn)
      Xc  <- cbind(Xc %*% object$UZ, gpT(x[ind, , drop = FALSE], object$gp.defn))
      if (i == 1) X <- matrix(0, nrow(x), ncol(Xc))
      X[ind, ] <- Xc
    }
    # leftover rows
    if (nrow(x) > ind[nk]) {
      ind <- (ind[nk] + 1):nrow(x)
      Xc  <- gpE(x[ind, , drop = FALSE], object$knt, object$gp.defn)
      Xc  <- cbind(Xc %*% object$UZ, gpT(x[ind, , drop = FALSE], object$gp.defn))
      X[ind, ] <- Xc
    }
  } else {
    X <- gpE(x, object$knt, object$gp.defn)
    X <- cbind(X %*% object$UZ, gpT(x, object$gp.defn))
  }
  X
}
```

Column layout: `[E(x, knt) @ UZ | T(x)]` — `k` penalized columns followed
by `null.space.dim` unpenalized columns. Total `bs.dim = k + null.space.dim`.

### 2.6 smoothCon Integration

Unlike RE smooths, GP smooths follow the *standard* `smoothCon` path:
- A sum-to-zero centering constraint `C = colMeans(X)` is computed and
  absorbed (one column dropped, `n_coefs = bs.dim - 1`).
- `side.constrain` defaults to TRUE, so `gam.side()` will detect and
  resolve any linear-trend collinearity (relevant for non-stationary
  mode with parametric `x` terms in the formula).
- `scale.penalty=TRUE` normalizes `S` by `||S||_1 / ||X||_∞²`.

### 2.7 R-Bridge Extraction Gaps

Current `RBridge.smooth_construct()` (`tests/r_bridge.py:564`) is
TPRS-shaped: it accepts `(smooth_expr, data, absorb_cons)` only and
extracts `X`, `S`, `rank`, `null_space_dim`, `Xu`, `UZ`, `shift`. Two
gaps matter for GP smooth-construct parity tests:

1. **No `knots=` argument.** The production API
   (`jaxgam/api.py:225`) intentionally rejects user-supplied knots
   (`NotImplementedError`, planned for v1.1), and `ModelSetup` calls
   `smooth.setup(data_dict)` with no knots passthrough
   (`jaxgam/formula/design.py:657`). For STRICT R parity we need both
   sides operating on the same knot set, which means **the bridge
   needs a test-only `knots=` parameter** that the test harness can
   feed into R, while our Python-side `GaussianProcessSmooth` is
   constructed directly with matching knots in the test fixture (not
   via `gam()`).

2. **GP-specific fields not extracted.** mgcv's GP smooth stores
   `object$knt` (centered knot matrix) and `object$gp.defn`
   (`c(sign·type, ρ_resolved, k_power)`) — both used by the predict
   method. The bridge ignores them. `Xu` is TPRS-specific and is empty
   for GP. **Add to the R-side `list(...)` return**: `knt = if
   (!is.null(sm$knt)) sm$knt else matrix(0, 0, 0)`, `gp_defn = if
   (!is.null(sm$gp.defn)) sm$gp.defn else numeric(0)`, and surface
   them on the Python return dict.

Both gaps are addressed in the implementation-plan unit that lands
between Commits E and F (see §13).

---

## 3. Mathematical Specification

### 3.1 Model

For `s(x_1, ..., x_d, bs="gp")`, the smooth has the form

```
f(x) = ∑_{j=1}^k β_j φ_j(x)  +  ∑_{l=0}^{M-1} γ_l ψ_l(x)
       \---- penalized GP ----/   \---- null space ----/
```

where:
- `φ_j(x) = [K(‖x − knt_1‖/ρ), …, K(‖x − knt_{nk}‖/ρ)] · UZ[:, j]`
  is the `j`-th low-rank eigenfunction of the correlation operator,
- `ψ_l(x)` are the unpenalized polynomial basis functions (intercept and
  optionally linear trend),
- `M = null_space_dim` (`1` if stationary, `d+1` if not),
- `k = bs.dim − M`.

The full model matrix (after centering: `x ← x − shift`) is

```
X = [ K(x, knt) · UZ | T(x) ]
    \--- n × k ---/   \-- n × M --/
```

with total `bs.dim = k + M` columns *before* sum-to-zero centering, and
`n_coefs = bs.dim − 1` after.

### 3.2 Distance Metric

Pairwise distances are plain **isotropic Euclidean** in the centered
covariate space:

```
d(x_i, knt_j) = sqrt( ∑_{l=1}^d (x_{i,l} − knt_{j,l})² )
```

No per-dimension scaling. Users with anisotropic predictors must
pre-standardize.

### 3.3 Kernel Catalogue

Let `e = d / ρ` be the scaled distance. The five supported kernels are:

| `|m[1]|` | Name | `K(e)` | Notes |
|---|---|---|---|
| 1 | Spherical | `(1 − 1.5·e + 0.5·e³) · 𝟙{e ≤ 1}` | Compact support; not PD for d > 3 |
| 2 | Power exponential | `exp(−eᵏ)`, `k ∈ (0, 2]` from `m[3]` (default 1) | `k=2` ⇒ squared-exponential / Gaussian |
| 3 | Matérn ν=3/2 | `(1 + e) · exp(−e)` | **Default** when `m[1]` unspecified |
| 4 | Matérn ν=5/2 | `(1 + e + e²/3) · exp(−e)` | Twice differentiable |
| 5 | Matérn ν=7/2 | `(1 + e + 0.4·e² + e³/15) · exp(−e)` | Thrice differentiable |

The R code emits exactly these formulas; we must match bit-for-bit at
STRICT tolerance for the kernel evaluator unit tests.

### 3.4 Range Parameter ρ

- **User-supplied**: `rho > 0` (positive float kwarg) → use directly.
- **Auto (default — `rho=None` or omitted)**: `ρ = max_{i,j} d(knt_i, knt_j)`
  — the **maximum pairwise knot distance** (Kammann & Wand default).
  This is computed on the *centered* training knots and stored as
  `self._resolved_rho` on the smooth.

The resolved `ρ` is **frozen at fit time** and reused for all predictions.
Predictions on data with a different spatial extent do not recompute `ρ`.

(mgcv encodes this as `m[2]`: positive ⇒ user-supplied, non-positive or
absent ⇒ auto. JaxGAM users always go through the explicit `rho=` kwarg;
see §1.6 and §6.4.)

### 3.5 Stationarity

The `stationary` bool kwarg controls whether a linear trend is included
in the null space:

- `stationary=False` (default): **non-stationary**, `T(x) = [1, x_1, …, x_d]`,
  `null_space_dim = d + 1`.
- `stationary=True`: **stationary**, `T(x) = [1]`, `null_space_dim = 1`.

(mgcv encodes this as `sign(m[1])`: `m[1] < 0` ⇒ stationary. See §6.4
for the full mgcv ↔ JaxGAM mapping.)

"Stationary" here is a statistical-modelling term, not a kernel property —
the kernel itself is always stationary in the geostatistical sense (depends
only on distance). The flag controls whether the model deliberately includes
a parametric linear trend (which the *GP component* would then have to
detrend if absent).

### 3.6 Penalty

The penalty is constructed from the truncated eigendecomposition of the
knot–knot correlation matrix `E`:

```
E       = U · Λ · Uᵀ     (nk × nk symmetric)
UZ      = U[:, top k]    (nk × k)
S_raw   = diag([λ_1, …, λ_k, 0, …, 0])   (bs.dim × bs.dim)
S       = S_raw / s_scale                  (smoothCon normalization)
```

where `λ_1 ≥ … ≥ λ_k` are the **`k` largest-magnitude eigenvalues** of `E`
(R's `slanczos(E, k, -1)` returns top-`k` by magnitude, not value).

Because some kernels (spherical, power-exponential with small ρ) produce
indefinite `E`, some `λ_j` can be negative. **JaxGAM normalizes the GP
penalty to PSD at setup time** by replacing negative eigenvalues with
their absolute value before constructing `S` — see §8.3 for the policy
and rationale. The fitting layer (`jax_utils.py`, `fitting/data.py`)
therefore never sees an indefinite GP penalty. This deviates from mgcv
on indefinite spectra (uncommon at our supported `d ∈ {1, 2, 3}`); the
deviation is documented in §8.3 and the R-bridge test in Commit G skips
STRICT diagonal-eigenvalue comparison whenever R's spectrum is negative.

### 3.7 Normalization

Standard smoothCon normalization applies:

```
maXX    = ||S||_1 / ||X||_∞²
S       = S / maXX
```

For GP, `||S||_1 = ∑_j |λ_j|` (sum of |eigenvalues|). This is computed
*after* the diagonal padding with zeros.

---

## 4. Knot Selection and Cardinality

### 4.1 Knot Algorithm

Reproducing R's behavior verbatim:

1. Compute `xu = unique combinations of x` (the unique rows of the centered
   covariate matrix).
2. Guard: `nrow(xu) ≥ bs.dim`, else raise an error matching R's message.
3. If `n > max.knots` **AND** `nrow(xu) > max.knots`:
   - Snapshot the global RNG seed (`temp.seed` analog).
   - Sample `max.knots` indices from `1..nrow(xu)` without replacement,
     using a *standalone* RNG seeded with `xt$seed` (default 1).
   - Restore the global RNG to its prior state.
   - `knt = xu[sampled_indices, :]`.
4. Else: `knt = xu`.

**Critical**: this must use the same sampling distribution as R's
`sample(n, k, replace=FALSE)` to produce identical knot sets across the two
implementations. R's `sample()` uses a specific algorithm (rejection
sampling with a hashed map for `k < n/2`, else Fisher–Yates); for exact
parity we should mimic R's algorithm or accept a documented tolerance
deviation when `n > max.knots`.

**Practical approach**: at MODERATE/LOOSE tolerance in the validation
matrix, the *fitted model* is robust to which 2000 of N knots we choose, so
exact knot parity is not required. For the smooth-construct STRICT tests
we either use small data (`n < max.knots`, no sampling) or supply explicit
`knots`.

### 4.2 Default Basis Dimension

R uses (`smooth.r:3519-3528`):

```r
def.k <- c(10, 30, 100)         # one per dimension d = 1, 2, 3+
if (bs.dim < 0) bs.dim <- ncol(knt) + 1 + def.k[dd]
if (bs.dim < ncol(knt) + 2) bs.dim <- ncol(knt) + 2
```

**bs.dim does not depend on stationarity.** R computes `bs.dim = d + 1 +
def.k[d]` *before* deciding `null.space.dim`, so total basis size is
identical between stationary and non-stationary modes; only `rank =
bs.dim - null_space_dim` shifts.

| Dimension d | Default bs.dim (both modes) | rank (non-stationary) | rank (stationary) |
|---|---|---|---|
| 1 | 12 = 1 + 1 + 10 | 10 = 12 − 2 | 11 = 12 − 1 |
| 2 | 33 = 2 + 1 + 30 | 30 = 33 − 3 | 32 = 33 − 1 |
| 3 | 104 = 3 + 1 + 100 | 100 = 104 − 4 | 103 = 104 − 1 |

Note: R indexes `def.k[dd]` where `dd = ncol(knt) = d`. For `d ≥ 4` this
produces `NA` and the construction fails arithmetically — a latent mgcv
bug. **Our implementation will mirror this** (i.e. only support
`d ∈ {1, 2, 3}` without explicit `k`, and require explicit `k` for `d ≥ 4`).

### 4.3 Minimum Basis Dimension

`bs.dim ≥ d + 2` (one penalized column plus the `d + 1` null-space). If a
user requests less, R warns and bumps to the minimum; we do the same.

### 4.4 Cardinality Limit

Like all smooths, GP must satisfy the full-model `p ≤ n` constraint at fit
time. There is no per-smooth cardinality cap beyond:
- `nrow(xu) ≥ bs.dim` (more unique covariate combinations than basis
  columns) — error if violated.
- `nk ≥ d + 2` (so the eigendecomp has enough rank).

---

## 5. Smooth Class Design

### 5.1 Class: `GaussianProcessSmooth`

```python
# jaxgam/smooths/gaussian_process.py

class GaussianProcessSmooth(Smooth):
    """Low-rank Gaussian process smooth (bs="gp").

    For s(x_1, ..., x_d, bs="gp"), constructs a reduced-rank kriging basis
    via truncated eigendecomposition of a correlation matrix. Supports five
    kernels (spherical, power-exponential, Matern 3/2, 5/2, 7/2) with a
    single scalar range parameter rho.

    User-facing arguments (via the formula's keyword args, surfaced on
    `spec.extra_args`):
        kernel : str, default "matern_3_2"
            One of {"spherical", "power_exponential",
            "matern_3_2", "matern_5_2", "matern_7_2"}. Aliases
            {"power", "matern32", "matern52", "matern72"} also accepted.
        rho : float, optional
            Kernel range (> 0). If omitted, defaults to the
            Kammann–Wand maximum pairwise knot distance.
        power : float, default 1.0
            Only consulted by `kernel="power_exponential"`; must be in
            (0, 2]. `power=2.0` ⇒ squared-exponential / Gaussian.
        stationary : bool, default False
            If True, the null space is the intercept only. If False
            (default), the null space is intercept + linear trend in
            each covariate.
        xt : dict, optional
            xt["max_knots"]: int, default 2000.
            xt["seed"]: int, default 1.

    Notes
    -----
    mgcv exposes the same four knobs through a single signed numeric
    vector `m = c(sign·type, rho, power)`. JaxGAM intentionally does
    NOT accept `m=` — passing `m=` raises `ValueError` from
    `parse_gp_config`. See §1.6 for the rationale and §6.4 for the
    mgcv ↔ JaxGAM mapping table.
    """

    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)
        # Standard centering and gam.side participate
        # (no _has_centering_constraint override, no side_constrain override)

        # Cached at setup
        self._config: GPConfig | None = None       # parsed kwargs
        self._resolved_rho: float | None = None    # rho actually used (frozen at setup)
        self._shift: np.ndarray | None = None      # (d,)
        self._knt: np.ndarray | None = None        # (nk, d) — centered
        self._E_knot: np.ndarray | None = None     # (nk, nk) — knot-knot kernel matrix (un-truncated input to _slanczos); exposed for R-parity tests (Commit G)
        self._UZ: np.ndarray | None = None         # (nk, k)
        self._stationary: bool = False             # mirrors self._config.stationary
        self._X: np.ndarray | None = None
        self._S: np.ndarray | None = None
```

### 5.2 `setup()` Method

```python
def setup(self, data: dict[str, np.ndarray]) -> None:
    """Construct GP basis from training data."""
    # 1. Parse explicit kernel kwargs (rejects mgcv-style `m=`).
    config = parse_gp_config(self.spec.extra_args)
    self._config = config
    self._stationary = config.stationary

    # 2. Read xt (max_knots, seed)
    xt = self.spec.extra_args.get("xt", {})
    max_knots = xt.get("max_knots", xt.get("max.knots", 2000))
    seed = xt.get("seed", 1)

    # 3. Stack covariates into (n, d) matrix
    variables = self.spec.variables
    d = len(variables)
    x = np.column_stack([np.asarray(data[v], dtype=float) for v in variables])
    n = x.shape[0]

    # 4. Knot harvesting
    knt = self._harvest_knots(x, max_knots, seed)
    nk = knt.shape[0]

    # 5. Resolve dimensions and validate cardinality BEFORE building E.
    #    nk < bs_dim would make k = bs_dim - null_space_dim larger than
    #    available knots and the eigendecomp would be ill-posed; also,
    #    a degenerate nk == 1 case would feed rho = max(distances) = 0
    #    into _gp_E and trigger divide-by-zero. Catch both up front.
    null_space_dim = 1 if self._stationary else d + 1
    bs_dim = self._default_bs_dim(self.spec.k, d, nk, null_space_dim)
    k = bs_dim - null_space_dim
    if nk < bs_dim:
        raise ValueError(
            "A term has fewer unique covariate combinations than "
            "specified maximum degrees of freedom "
            f"(nk={nk}, bs_dim={bs_dim})."
        )

    # 6. Centre covariates and knots.
    self._shift = x.mean(axis=0)
    x_c = x - self._shift
    knt_c = knt - self._shift
    self._knt = knt_c                                # store centered knots

    # 7. Build E on centered knots; resolve rho if auto. _gp_E guards
    #    against rho <= 0 (degenerate spatial extent).
    E, rho_resolved = _gp_E(knt_c, knt_c, config)
    self._resolved_rho = rho_resolved
    self._E_knot = E                                  # cached for R-parity tests

    # 8. Truncated eigendecomposition (top-k by magnitude). The
    #    cardinality check above guarantees k < nk, so the "k >= nk"
    #    branch in mgcv's R source is unreachable here — single code
    #    path. Pass tol=eps**0.5 to match mgcv's slanczos(E, k, -1)
    #    default (TPRS uses the function default eps**0.7; see §5.4.1).
    eigvals, eigvecs = _slanczos(E, k, tol=np.finfo(float).eps ** 0.5)

    # Indefinite-penalty clip (§8.3): PSD normalization, always on.
    # Preserves rank; deviates from mgcv on indefinite spectra
    # (documented in §8.3).
    if (eigvals < 0).any():
        warnings.warn(
            f"GP smooth on {self.spec.variables}: "
            f"{int((eigvals < 0).sum())} of {len(eigvals)} truncated "
            "eigenvalues are negative; clipping to |λ| (see §8.3)."
        )
        eigvals = np.abs(eigvals)
    D = np.zeros((bs_dim, bs_dim))
    np.fill_diagonal(D[:k, :k], eigvals)
    # remaining null_space_dim diagonal entries stay zero
    UZ = eigvecs                                      # (nk, k)
    self._UZ = UZ

    # 9. Store base shape info.
    self.null_space_dim = null_space_dim
    self.rank = k
    self.n_coefs = bs_dim  # before centering; constraint absorption reduces by 1

    # 10. Build training X via the predict pathway.
    X = self._build_design(x_c)
    self._X = X

    # 11. smoothCon normalization.
    [self._S], self._s_scale = self._smoothcon_normalize(X, [D])

    self._is_setup = True
```

### 5.3 Kernel Evaluator and Config Parser

Both live as **module-level functions** in `jaxgam/smooths/gp_kernels.py`
(not methods on the smooth class), so kernel math can be unit-tested in
isolation and the kernel module has no dependency on the smooth class.

```python
# jaxgam/smooths/gp_kernels.py

def parse_gp_config(extra_args: dict[str, object]) -> GPConfig:
    """Build a GPConfig from a smooth spec's extra_args.

    Rejects mgcv-style `m=` with a clear ValueError pointing at the
    Python-native kwargs (see §1.6, §6.4).
    """
    if "m" in extra_args:
        raise ValueError(
            "mgcv-style `m=` is not supported for JaxGAM GP smooths. "
            "Use `kernel=`, `rho=`, `power=`, and `stationary=` instead. "
            "See docs/gaussian_process/design.md §6.4 for the mapping."
        )

    kernel_raw = str(extra_args.get("kernel", GPKernelName.MATERN_3_2.value))
    # Normalize alias → canonical via the registry's case-insensitive
    # lookup, then promote to the GPKernelName enum.
    if kernel_raw.lower() not in gp_kernel_registry:
        raise KeyError(  # registry-style message
            f"Unknown GP kernel: {kernel_raw!r}. "
            f"Available: {', '.join(gp_kernel_registry.available)}"
        )
    canonical = _CANONICAL_FOR.get(kernel_raw.lower(), kernel_raw.lower())
    kernel = GPKernelName(canonical)

    rho_raw = extra_args.get("rho", None)
    rho = None if rho_raw is None else float(rho_raw)
    if rho is not None and rho <= 0.0:
        raise ValueError(
            f"GP `rho` must be positive (or omitted for auto-selection); got {rho!r}."
        )

    power = float(extra_args.get("power", 1.0))
    stationary = bool(extra_args.get("stationary", False))

    return GPConfig(
        kernel=kernel,
        stationary=stationary,
        rho=rho,
        power=power,
    )


def _gp_E(
    x: np.ndarray,
    xk: np.ndarray,
    config: GPConfig,
    resolved_rho: float | None = None,
) -> tuple[np.ndarray, float]:
    """Evaluate correlation kernel between rows of x and xk.

    Parameters
    ----------
    x : (n, d) array
    xk : (nk, d) array
    config : GPConfig
        Parsed kernel configuration. `config.stationary` is *not* used
        here; it controls the null space (`_gp_T`, §5.6) only.
    resolved_rho : float, optional
        If provided, used directly — this is the path the smooth class
        takes at prediction time to re-apply the training-time rho. If
        None, falls back to `config.rho` (user-supplied), and finally
        to the Kammann–Wand max-pairwise-distance default.

    Returns
    -------
    E : (n, nk) array
        Kernel-evaluated correlation matrix.
    rho_resolved : float
        The rho actually used. Cached on the smooth at setup; passed
        back as `resolved_rho` for prediction calls.
    """
    distances = _compute_distance_matrix(x, xk)

    if resolved_rho is not None:
        rho = float(resolved_rho)
    elif config.rho is not None:
        rho = float(config.rho)
    else:
        rho = float(distances.max())

    # Defensive guard: rho <= 0 is only reachable on degenerate input
    # (e.g. all-identical knots producing distances.max() == 0). The
    # cardinality check in setup() should prevent this, but fail loud
    # rather than divide-by-zero if it slips through.
    if rho <= 0.0:
        raise ValueError(
            f"GP kernel range `rho` must be positive; got {rho!r}. "
            "This usually means the knot set is degenerate (all rows "
            "identical) — check the data and the `max_knots` setting."
        )

    kernel = gp_kernel_registry.get_instance(config.kernel.value)
    kernel.validate(config.power)

    E = kernel.evaluate(distances / rho, power=config.power)
    return E, rho
```

`_CANONICAL_FOR` is a small alias-resolution table sitting next to the
registry — it normalizes `"matern32"` → `"matern_3_2"` etc. so the
`GPConfig` always holds a canonical `GPKernelName` enum value, while
the `Registry` keeps accepting both spellings for `get_instance()`.

### 5.4 Shared Kriging Infrastructure in `utils.py`

This section walks through the **PR 0 refactor** (see §13 for sequencing).
Four pieces of code currently in `tprs.py` are also needed by GP. Moving
them to `utils.py` lets both smooths consume them without cross-smooth
imports. Each move is a code-relocation, not a rewrite — behaviorally
identical to the current TPRS.

#### 5.4.1 Pieces being moved

**1. `_slanczos_jit` + `_slanczos`** (`tprs.py:278-442` → `utils.py`)

The Numba-JIT Lanczos solver. Byte-faithful port of R's `Rlanczos`:
- Same LCG starting vector (`jran = (jran * 106 + 1283) % 6075`).
- Same double reorthogonalization.
- Same convergence-check schedule.
- Returns top-`k` largest-magnitude eigenpairs.
- `cache=True` Numba decorator — JIT cost paid once per session.

Signature (unchanged):
```python
_slanczos(A: np.ndarray, k: int, tol: float | None = None) -> tuple[D, U]
```

**Tolerance caveat (load-bearing for GP).** TPRS callers use the function's
default `tol = np.finfo(float).eps ** 0.7`, which was chosen for `r^{2m-d}
log(r)` TPS kernel eigendecomps. **mgcv's GP constructor calls
`slanczos(E, k, -1)`, which uses R's wrapper default `tol =
.Machine$double.eps^0.5`** (`mgcv/R/mgcv.r:19`). Reusing the same helper
is fine — but GP must **explicitly thread `tol=np.finfo(float).eps ** 0.5`**
at every call site, otherwise we silently diverge from mgcv by a few
orders of magnitude on the convergence floor. Do not change the
function's default (that would alter TPRS numerics); the GP-side change
is per-call.

**2. `_compute_distance_matrix`** (`tprs.py:232-252` → `utils.py`)

Pairwise Euclidean distance via broadcasting:
```python
def _compute_distance_matrix(X1, X2):
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))
```

GP's `_gp_E` (Section 5.3) opens with this exact computation. No
modification needed.

**3. `_get_unique_rows`** (`tprs.py:255-275` → `utils.py`)

Wraps `np.unique(X, axis=0, return_inverse=True)`. Both TPRS and GP need
this to find unique covariate combinations before knot subsampling.

**4. `_subsample_knots`** (new helper, extracted from TPRS `setup()` lines
627-639)

TPRS currently inlines:
```python
rng = np.random.RandomState(1)            # hardcoded seed
idx = rng.choice(n_unique, max_knots, replace=False)
idx.sort()
Xu = Xu[idx]
```

GP needs the same logic but with a configurable seed (`xt["seed"]`,
default `1`). Extract as:
```python
def _subsample_knots(
    Xu: np.ndarray, max_knots: int, seed: int = 1
) -> np.ndarray:
    """Reproducible knot subsampling. Matches R's mgcv pattern.

    Uses np.random.RandomState (legacy API) intentionally to preserve
    bit-exact reproducibility with TPRS's pre-refactor behavior.
    """
    if Xu.shape[0] <= max_knots:
        return Xu
    rng = np.random.RandomState(seed)
    idx = rng.choice(Xu.shape[0], max_knots, replace=False)
    idx.sort()
    return Xu[idx]
```

TPRS's call site becomes `Xu = _subsample_knots(Xu, max_knots, seed=1)` —
behavior unchanged. GP calls `_subsample_knots(Xu, max_knots, seed=xt_seed)`.

> **Note on TPRS knot-index reuse**: TPRS currently uses `inverse` from
> `np.unique` to map data → knots when knots equal the unique rows, and
> recomputes nearest-knot indices via `_nearest_knot_indices` when
> subsampling occurs. If we extract `_subsample_knots`, the caller still
> needs to handle this `inverse` rebuilding. The cleanest split: the
> helper returns just the subsampled rows; rebuilding `inverse` stays
> at the call site (TPRS already has `_nearest_knot_indices`).

#### 5.4.2 Refactor steps

1. **Add to `utils.py`** (after `interaction_matrix`): the four helpers
   above. `utils.py` already imports `numba` and `numpy`, so no new
   top-level imports.
2. **Delete from `tprs.py`**: lines 232-275 (`_compute_distance_matrix`,
   `_get_unique_rows`) and 278-442 (`_slanczos_jit`, `_slanczos`). Replace
   the inline subsample block at 627-639 with a call to `_subsample_knots`.
3. **Update `tprs.py` imports**:
   ```python
   from jaxgam.smooths.utils import (
       _compute_distance_matrix,
       _get_unique_rows,
       _slanczos,
       _subsample_knots,
   )
   ```
4. **Re-run TPRS tests** to confirm no behavioral change. This is a pure
   code-move; failures indicate a missed dependency, not a logic bug.
5. **Optionally relocate `_slanczos` direct tests** from `test_tprs.py` to
   a new `test_utils.py` block — cosmetic, defer if it adds noise.

#### 5.4.3 GP call sites after the refactor

```python
from jaxgam.smooths.utils import (
    _slanczos,
    _compute_distance_matrix,
    _get_unique_rows,
    _subsample_knots,
)

# In GaussianProcessSmooth.setup():
x = np.column_stack([np.asarray(data[v], dtype=float) for v in variables])
xu, _ = _get_unique_rows(x)
knt = _subsample_knots(xu, max_knots, seed=xt_seed)

# In _gp_E:
E_distances = _compute_distance_matrix(x, xk)   # raw Euclidean
# ... then E_distances / rho, kernel transform ...

# Eigendecomposition: pass mgcv-faithful tol explicitly (§5.4.1 caveat).
eigvals, eigvecs = _slanczos(E, k, tol=np.finfo(float).eps ** 0.5)
```

#### 5.4.4 What stays in `tprs.py` (and why)

| Symbol | Stays | Reason |
|---|---|---|
| `_null_space_basis_r` / `_null_space_basis_r_jit` (lines 445-530) | ✓ | Right-Householder QR for `T.T @ U_k` null-space construction. GP's null space is the explicit polynomial basis (`gpT`: `[1]` or `[1, x_1, …, x_d]`), not derived from QR. |
| `tps_semi_kernel`, `eta_const` | ✓ | TPS-specific radial function `r^{2m-d} log(r)`. GP uses Matérn / spherical / power-exponential — entirely different kernel family. |
| `compute_polynomial_basis` + helpers | ✓ | Generates all monomials of degree < `m`. GP only needs degree ≤ 1 (intercept + linear); a 5-line direct construction is simpler than importing this. |
| `_default_k` | ✓ | TPRS basis-dim table (`{1: 10, 2: 30}`). GP has a different table and uses `null.space.dim + def.k[d]` formula. |
| `_nearest_knot_indices` (lines 801-806) | ✓ | Only used after knot subsampling in TPRS to rebuild the inverse mapping. GP rebuilds its design matrix from scratch via `_gp_E` (no nearest-knot mapping needed). |

#### 5.4.5 Risk and validation

The refactor is **mechanical** — no algorithm changes. Risks:

1. **Import cycles**: `utils.py` must not import from `tprs.py` or any
   other smooth module. All four helpers are leaf functions (numpy +
   numba only), so this is fine.
2. **Numba cache key**: moving `_slanczos_jit` changes its module path,
   which may invalidate cached Numba binaries on first run after the
   move. Acceptable — recompilation is a one-time cost.
3. **TPRS test parity**: every TPRS test must pass after the move with
   identical numerical output (same seed, same algorithm, just a
   different file). Add a regression test that fixes seed and compares
   pre/post-refactor TPRS fits if uncertainty remains.

These risks are all addressable in the same PR; PR 0 should not merge
without a clean TPRS test run.

### 5.5 Knot Harvesting

Knot harvesting reuses the shared `_get_unique_rows` and
`_subsample_knots` helpers from `utils.py` (§5.4) — there is **only
one** subsampling RNG in the codebase (`np.random.RandomState`, legacy
API, used by both TPRS and GP).

```python
def _harvest_knots(
    self,
    x: np.ndarray,
    max_knots: int,
    seed: int,
) -> np.ndarray:
    """Reproducible knot selection.

    1. Unique rows of x via `_get_unique_rows`.
    2. If unique-row count > max_knots, subsample via
       `_subsample_knots(seed)`; otherwise use all unique rows.

    Note: R's `sample()` algorithm differs from numpy's
    `RandomState.choice`, so subsampled knot sets do not match mgcv
    bit-for-bit. STRICT R parity is achievable only when no
    subsampling occurs (`nrow(xu) <= max_knots`) or via the
    test-only explicit-knots bridge path (Commit F). See §10.6.
    """
    xu, _ = _get_unique_rows(x)
    return _subsample_knots(xu, max_knots, seed=seed)
```

**Knot ordering caveat**: `np.unique(x, axis=0)` returns sorted unique
rows. R's `uniquecombs` returns rows in first-occurrence order. This
ordering difference affects:
1. Which knots are sampled (when subsampling)
2. The eigendecomposition (eigenvectors of permuted `E` differ by row
   permutation)
3. The final basis up to a column-wise sign/orthogonal transform

**Mitigation**: for STRICT R-comparison tests, supply explicit `knots` (so
ordering is user-controlled). For validation-matrix tests, ordering
differences vanish in fitted values / deviance comparisons.

### 5.6 Design Matrix Construction

Both helpers live as **module-level functions** in `gp_kernels.py`
(parallel to `_gp_E`), so the smooth class just composes them with the
training-time `_resolved_rho`:

```python
# jaxgam/smooths/gp_kernels.py

def _gp_T(x_centered: np.ndarray, stationary: bool) -> np.ndarray:
    """Null-space basis: intercept (+ linear trend if non-stationary)."""
    n = x_centered.shape[0]
    if stationary:
        return np.ones((n, 1))
    return np.column_stack([np.ones(n), x_centered])


# jaxgam/smooths/gaussian_process.py

def _build_design(self, x_centered: np.ndarray) -> np.ndarray:
    """Build [E(x, knt) @ UZ | T(x)] for centered x."""
    E_xn, _ = _gp_E(
        x_centered, self._knt, self._config,
        resolved_rho=self._resolved_rho,
    )
    pen_block = E_xn @ self._UZ                          # (n, k)
    null_block = _gp_T(x_centered, self._stationary)     # (n, M)
    return np.hstack([pen_block, null_block])
```

For very large `n` we can chunk on `nk`-sized blocks to bound memory, but
unlike R this is not strictly required; numpy handles a `n × nk` allocation
fine up to `n × nk ≈ 10⁸` entries (~800 MB). Add chunking if memory
profiling indicates it's needed.

### 5.7 `build_design_matrix()` and `predict_matrix()`

```python
def build_design_matrix(self, data):
    self._require_setup()
    return self.predict_matrix(data)

def predict_matrix(self, new_data):
    self._require_setup()
    variables = self.spec.variables
    x = np.column_stack([np.asarray(new_data[v], dtype=float) for v in variables])
    x_c = x - self._shift                         # apply training shift
    return self._build_design(x_c)
```

### 5.8 `build_penalty_matrices()`

```python
def build_penalty_matrices(self):
    self._require_setup()
    return [Penalty(self._S, rank=self.rank, null_space_dim=self.null_space_dim)]
```

### 5.9 Stationarity Decoding

No bespoke helper is needed: `parse_gp_config` already pulls the
`stationary` bool straight out of `extra_args` (default `False`) and
the smooth class mirrors it onto `self._stationary` for the
null-space builder (`_gp_T`, §5.6).

```python
# inside parse_gp_config (excerpt — see §5.3 for the full body):
stationary = bool(extra_args.get("stationary", False))
return GPConfig(..., stationary=stationary, ...)
```

No `_decide_stationary` static method; no parsing of signed numbers.
The mgcv ↔ JaxGAM mapping (`sign(m[1])` → `stationary`) is documented
in §6.4 and only matters at the R-bridge boundary
(`gp_config_to_mgcv_m`, §11.2).

### 5.10 Default Basis Dimension

```python
@staticmethod
def _default_bs_dim(k_spec: int, d: int, nk: int, null_space_dim: int) -> int:
    DEF_K = (10, 30, 100)
    if k_spec is not None and k_spec > 0:
        bs_dim = k_spec
    else:
        if d > 3:
            raise ValueError(
                "Default basis dim for GP smooth with d > 3 is undefined. "
                "Please specify k explicitly."
            )
        bs_dim = d + 1 + DEF_K[d - 1]
    min_bs = d + 2
    if bs_dim < min_bs:
        warnings.warn(
            f"GP basis dimension {bs_dim} below minimum {min_bs}; reset to {min_bs}."
        )
        bs_dim = min_bs
    return bs_dim
```

---

## 6. Formula Parsing

### 6.1 Parser Capabilities and Limits

The current parser (`jaxgam/formula/parser.py:243-310`) evaluates each
keyword argument with `ast.literal_eval` — it accepts Python literals
(ints, floats, strings, bools, lists, tuples, dicts) but **does not
interpret R-style `c(...)` or `list(...)` function calls**. Adding
R-syntax shimming is out of scope for the GP commit (it would touch
the parser broadly and benefit every smooth, not just GP).

GP exposes its kernel and runtime knobs through **Python literals**
in the formula string. The supported keyword arguments are documented
in §1.6; the parser is generic and has no GP-specific code — it just
surfaces everything onto `SmoothSpec.extra_args`, and
`parse_gp_config` (§5.3) materializes a `GPConfig` from those kwargs
at smooth-setup time.

### 6.2 Supported Formula Patterns

```python
# Defaults: Matérn 3/2, auto ρ, non-stationary.
s(x, bs="gp")
s(x, z, bs="gp", k=50)

# Explicit kernel and range.
s(x, z, bs="gp", kernel="matern_3_2", rho=0.5)
s(x, bs="gp", kernel="power_exponential", rho=1.0, power=2.0)
s(x, bs="gp", kernel="spherical", rho=0.5)

# Stationary mode (no linear-trend null space).
s(x, bs="gp", kernel="spherical", stationary=True)

# Tensor margins (one univariate GP per variable, via the existing
# tensor wrappers + registry dispatch — see §1.3).
te(x, z, bs="gp", kernel="matern_3_2", k=5)
ti(x, z, bs="gp", k=5)

# Knot subsampling knobs.
s(x, bs="gp", xt={"max_knots": 500, "seed": 42})
```

Each translates to a `SmoothSpec(variables=[...], bs="gp",
extra_args={...})`. Aliases (`"power"`, `"matern32"`, `"matern52"`,
`"matern72"`) are accepted by the registry and normalized to the
canonical `GPKernelName` value inside `parse_gp_config`.

### 6.3 `m=` Is Rejected at Setup

`parse_gp_config` raises `ValueError` if `extra_args` contains an `m`
key:

> mgcv-style `m=` is not supported for JaxGAM GP smooths. Use
> `kernel=`, `rho=`, `power=`, and `stationary=` instead. See
> docs/gaussian_process/design.md §6.4 for the mapping.

This is a deliberate divergence from mgcv. The rejection lives in
`parse_gp_config` (Phase 1 smooth setup), not the parser — the parser
remains generic and has no GP knowledge. Pinning the rejection as a
**positive parser test** (`m=` parses fine into `extra_args`) **plus**
a **smooth-construct negative test** (`GaussianProcessSmooth(spec).setup(data)`
raises on `extra_args["m"]`) is the contract Commit D / E exercise
(§12.2).

R-style `c(...)` continues to fail at the parser layer via
`ast.literal_eval` (`s(x, bs='gp', m=c(3, 0.5))` raises before ever
reaching `parse_gp_config`). Commit E keeps this as a separate
negative parser test so a future R-syntax patch breaks it loudly.

### 6.4 mgcv ↔ JaxGAM Mapping (for reference)

For readers porting code from mgcv, or for the R-bridge converter
(§11.2's `gp_config_to_mgcv_m`):

| mgcv | JaxGAM |
|---|---|
| `m=c(1, 0.5)` | `kernel="spherical", rho=0.5` |
| `m=c(-1, 0.5)` | `kernel="spherical", rho=0.5, stationary=True` |
| `m=c(2, 0.5)` | `kernel="power_exponential", rho=0.5` (default `power=1.0`) |
| `m=c(2, 0.5, 2)` | `kernel="power_exponential", rho=0.5, power=2.0` (squared-exp) |
| `m=c(-2, 0.5, 2)` | `kernel="power_exponential", rho=0.5, power=2.0, stationary=True` |
| `m=c(3, 0.5)` | `kernel="matern_3_2", rho=0.5` |
| `m=c(4, 0.5)` | `kernel="matern_5_2", rho=0.5` |
| `m=c(5, 0.5)` | `kernel="matern_7_2", rho=0.5` |
| `m=NA` or omitted | (defaults: Matérn 3/2, auto ρ, `stationary=False`) |

### 6.5 `xt` Argument

`xt` is a plain Python dict. Recognized keys:

- `max_knots` (int, default 2000) — knot subsampling cap.
- `seed` (int, default 1) — RNG seed for reproducible subsampling.

R's `xt=list(max.knots=500, seed=42)` becomes
`xt={"max_knots": 500, "seed": 42}` on the Python side. The class
also accepts the dot-key form `{"max.knots": ...}` for users who
paste verbatim from R docs, but Python-literal `max_knots` is the
documented form.

---

## 7. Constraint Pipeline Integration

### 7.1 Standard Centering Applies

Unlike RE smooths, GP smooths **do not** opt out of sum-to-zero centering.
The `CoefficientMap.build()` path applies `apply_sum_to_zero()` to the GP
design matrix, dropping one column and reducing `n_coefs` from `bs.dim`
to `bs.dim - 1`.

This works correctly out of the box — no changes to `constraints.py`.

### 7.2 gam.side() Applies

`side_constrain = True` (the default). For non-stationary GP, the linear
trend in the null space (`T(x)`) can collide with:
- A parametric `x` term in the same formula: `gam(y ~ x + s(x, bs="gp"))`.
- The linear trend in another smooth.

`gam.side()` will detect this via the existing pivoted-QR mechanism and
drop the redundant columns. No GP-specific code needed.

For stationary GP (`m[1] < 0`), the null space is just the intercept,
which is already absorbed by centering, so `gam.side()` will typically be
a no-op.

### 7.3 No `_random`, `_noterp`, `_has_centering_constraint` Flags

GP smooths inherit the defaults from `Smooth.__init__`:
- `side_constrain = True`
- `_noterp = False` — **required for the tensor-margin pathway**
  (`te(..., bs="gp")` / `ti(..., bs="gp")`). When `_noterp` is False,
  `TensorProductSmooth._svd_reparameterize` (`tensor.py:52-123`) runs
  on each GP margin and produces a numerically conditioned tensor
  basis. Do not set `_noterp = True` without R evidence that mgcv's
  GP margins also disable reparameterization (none currently
  exists).
- No `_random` flag set
- No `_has_centering_constraint = False` override

### 7.4 No SmoothInfo Changes

GP smooths do not require any field additions to `SmoothInfo`. Summary
p-values use the standard `type_=0` path (smooth p-value, not random
effect p-value).

---

## 8. Penalty Construction

### 8.1 Penalty Structure

```
S = diag([λ_1, λ_2, …, λ_k, 0, 0, …, 0]) / s_scale
                              \-- null_space_dim zeros --/
```

`bs.dim × bs.dim` diagonal matrix. After smoothCon normalization, the
penalty has spectrum `{λ_j / s_scale}` on the penalized block and zeros on
the null-space block.

### 8.2 Rank and Null Space

- **Rank** = `k = bs.dim − null_space_dim`.
- **Null space dimension** = `1` (stationary) or `d + 1` (non-stationary).

These are computed at setup and stored on the smooth and `Penalty` object.

### 8.3 Indefinite Penalty Handling

Some kernels (`type=1` spherical for `d ≥ 4`, `type=2` with extreme ρ)
produce **negative eigenvalues** in the truncated eigendecomposition. The
penalty matrix `S` is then indefinite (diagonal with mixed signs).

**Why this is a hard constraint, not a soft preference.** Three places in
the current fitting pipeline assume PSD penalties:

1. `jaxgam/jax_utils.py:242` (and `:274`): the log-determinant path
   computes `sign, logdet = slogdet(S)` and then `jnp.where(sign > 0,
   logdet, -1e10)` — indefinite `S` gets clipped to `-1e10`, destroying
   REML scoring. This is not a real pseudo-det fallback.
2. `jaxgam/fitting/data.py:434`: per-smooth metadata counts only
   eigenvalues exceeding `max(|eigs|) * eps^(2/3)` as nonzero, so
   negative eigenvalues collapse into the null space silently —
   `singleton_ranks` and `singleton_eig_constants` are wrong.
3. `jaxgam/fitting/data.py:545`: the singleton reparameterization
   `D_diag = 1/sqrt(eigs[mask])` masks on positive eigenvalues only.
   Negatives get `D_diag = 1` (unscaled), corrupting the reparameterized
   penalty's identity-block structure.

Fixing all three for true indefinite support is a non-trivial fitting
overhaul, and indefinite GP penalties are uncommon at our supported
`d ∈ {1, 2, 3}`. Confine the workaround to GP setup instead.

**Resolution — clip at setup (was Option (a))**: in
`GaussianProcessSmooth.setup()`, after `_slanczos` returns
`(eigvals, eigvecs)`, replace any negative eigenvalue with its
**absolute value** before building `D`:

```python
if (eigvals < 0).any():
    warnings.warn(
        f"GP smooth on terms {self.spec.variables}: {int((eigvals < 0).sum())} "
        f"of {len(eigvals)} truncated eigenvalues are negative (indefinite "
        f"kernel for these data). Replacing with |λ|; this deviates from "
        f"mgcv for these spectra. See docs/gaussian_process/design.md §8.3."
    )
    eigvals = np.abs(eigvals)
```

This:
- Preserves rank (`k = bs.dim − null_space_dim`) and the diagonal
  pattern of the penalty.
- Touches **only** `gaussian_process.py`; no changes to `jax_utils.py`,
  `fitting/data.py`, or `Penalty`.
- Breaks bit-faithful R parity for those specific spectra. Tests must
  not assert STRICT penalty equality against R when any clipping
  occurred — the R-bridge test in Commit G compares `diag(S)` only when
  the R-side eigenvalues are all non-negative; otherwise it logs the
  divergence and skips.

Options (b) and (c) from earlier drafts are rejected: (b) requires the
three-place fitting fix above; (c) reduces rank and changes
`null_space_dim`, which then requires updating `Penalty.rank` /
`null_space_dim` accounting at setup and propagating through
`CoefficientMap`. Option (a) is the only change-local-to-GP fix that
keeps fitting untouched.

This change lives in Commit D (GP class), not a separate Commit H —
there is nothing conditional about it.

### 8.4 Embedding in Global Penalty Space

Existing `CompositePenalty.embed()` handles GP correctly — the diagonal
penalty is embedded as a single `bs.dim × bs.dim` block at the GP smooth's
column range.

### 8.5 Single Smoothing Parameter

GP has exactly one penalty (one λ), unlike tensor products with multiple
margin penalties. This makes it well-behaved for REML/ML — one extra
dimension in the optimizer's parameter space.

---

## 9. Prediction

### 9.1 Training Data Round-Trip

`predict_matrix(training_data)` reproduces `build_design_matrix()`.
Implementation: both delegate to `_build_design(x − shift)`, so the
roundtrip is exact.

### 9.2 New Data with Same Support

For new `x_new`:
1. Apply the stored shift: `x_c = x_new − shift`.
2. Evaluate `E_new = K(‖x_c − knt_c‖ / ρ)` using the stored `gp_defn`.
3. Project: `X_new = [E_new @ UZ | T(x_c)]`.

The stored `ρ` (in `gp_defn`) is the training-time value. This is correct
behavior: we want to extrapolate the same GP, not a re-tuned GP.

### 9.3 New Data Outside Training Support

For points `x_new` far from any training knot:
- The kernel value `K(d/ρ)` is small (decays per the kernel).
- The penalized contribution `E_new @ UZ` is small.
- The null-space contribution `T(x_new) = [1, x_new, …]` is unbounded
  (linear trend extrapolates).

This matches R's behavior: GP regression with linear-trend null space
extrapolates linearly far from data. Stationary GP (`m[1] < 0`)
extrapolates to a constant. Users should be aware of this when predicting
beyond the training support.

### 9.4 Out-of-Domain Robustness

For the spherical kernel (`type=1`), `K(e) = 0` for `e > 1`. New points
farther than `ρ` from every knot contribute zero through the penalized
block. This is correct and matches R.

For power-exponential with `k_pow=2` and small `ρ`, `K(e) ≈ 0` very
quickly, so far-from-knot predictions reduce to the null space alone.

No special handling needed beyond the standard pathway.

### 9.5 Chunked Construction (Deferred to v1.1)

R chunks on `nk`-sized blocks when `n > nk` for memory bounds. **JaxGAM
v1.0 does not chunk** — numpy handles a single `n × nk` allocation
comfortably up to ~10⁸ entries (~800 MB at float64). Should a future
benchmark surface memory pressure, the implementation is straightforward:

```python
def _build_design_chunked(self, x_centered, chunk_size):
    n = x_centered.shape[0]
    X = np.empty((n, self.n_coefs))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        X[start:end] = self._build_design(x_centered[start:end])
    return X
```

Out of scope for the v1.0 GP feature. Add in v1.1 only if memory
profiling proves it necessary.

---

## 10. Numerical Considerations

### 10.1 Indefinite Penalty

**Policy**: GP normalizes its penalty to PSD at setup time (§8.3 clip).
The fitting layer assumes PSD penalties and is *not* expected to
tolerate indefinite `S`. The Commit-D regression test asserts that the
clip fires on a constructed indefinite-spectrum case and that the
resulting `diag(S) >= 0` everywhere. No fitting-layer changes are made
and no validation-matrix invariant is relaxed.

### 10.2 Eigenvalue Numerical Floor

For well-behaved Matérn kernels, eigenvalues of `E` decay geometrically.
Past some `k`, eigenvalues fall below numerical precision and the
corresponding eigenvectors are numerically random.

Mitigation:
- Trust `_slanczos` to return reliable top-`k` eigenpairs (the Lanczos
  algorithm naturally handles this — eigenvalues below `eps * λ_max` are
  not extracted).
- The default `k = 12` (1D) or `33` (2D) is well within the reliably
  computable range for typical data.

### 10.3 Eigenvector Sign Ambiguity

Eigenvectors are determined up to sign. Different Lanczos implementations
may flip signs. The penalty (diagonal of eigenvalues) is invariant, but
the design matrix `X = E @ UZ` flips signs column-wise.

Impact: coefficient β_j is determined up to sign correspondingly. Fitted
values `X @ β` are sign-invariant.

**Test implication**: when comparing GP to R, compare *fitted values* and
*deviance* (sign-invariant), not coefficients column-by-column.

If we use the same Numba `_slanczos` as TPRS with the same deterministic
starting vector (LCG with R seed=1), eigenvector signs should match R for
the same eigenvalue. But we should validate this empirically rather than
assume.

### 10.4 ρ Auto-Selection Stability

`ρ = max(E_distances)` is determined by a single pair of points. Adding
or removing one extreme point can change ρ substantially, which shifts
the entire kernel evaluation.

This is mgcv's behavior. We mirror it. Users wanting reproducibility
across data resamples should supply explicit `m[1]`.

### 10.5 Centering Constraint Interaction

The sum-to-zero centering constraint `C = colMeans(X)` is computed from
the GP `X` matrix. Because the first column of `T(x_c)` is `1` (intercept),
this column has `colMean = 1` exactly, so centering effectively drops the
intercept column from the null space.

Result: after centering, the null space loses its constant component but
keeps the linear-trend columns (non-stationary). This is correct: the
global intercept lives in the model's parametric intercept, not in the
smooth.

For stationary GP, centering reduces null_space_dim from 1 to 0
effectively (only intercept removed). The smooth becomes "almost
full-rank" with no null space — same situation as `bs="cs"` shrinkage
cubic.

### 10.6 Knot Sampling and R Parity

R's `sample()` differs from `numpy.random.default_rng().choice()` in
algorithm. We cannot match knot selection bit-for-bit when subsampling.

**Test strategy**:
- For STRICT smooth-construct tests (kernel evaluation, penalty values),
  use small data where no subsampling occurs (`n ≤ max.knots`), OR pass
  explicit `knots`.
- For validation-matrix tests (fitted models), use MODERATE/LOOSE
  tolerance on coefficients (or fitted values) — the model is robust to
  which 2000 knots get sampled.
- Document this in design.md and test docstrings.

---

## 11. File Plan

### 11.1 New Files

| File | Description |
|---|---|
| `jaxgam/smooths/gp_kernels.py` | `GPKernelName` enum, `GPConfig` dataclass, `GPKernel` ABC + five kernel classes, `gp_kernel_registry` (instance of `jaxgam.registry.Registry`), `_CANONICAL_FOR` alias table, `parse_gp_config`, `_gp_E`, `_gp_T`. |
| `jaxgam/smooths/gaussian_process.py` | `GaussianProcessSmooth` class. Imports the kernel module above. |
| `tests/test_smooths/test_gp_kernels.py` | Unit tests for the kernel module: closed-form kernel values, `parse_gp_config` (defaults + `m=` rejection + alias resolution + invalid kernel/rho/power), `_gp_T` shapes. Lives separately from `test_gaussian_process.py` so the kernel module can be tested without the smooth class. (Optional: fold into `test_gaussian_process.py` if file count is a concern — both placements are fine.) |
| `tests/test_smooths/test_gaussian_process.py` | Smooth-class structural tests + R smooth-construct comparison. |
| `docs/gaussian_process/implementation_plan.md` | PR-by-PR breakdown |

### 11.2 Modified Files

| File | Change |
|---|---|
| `jaxgam/smooths/utils.py` | **Add** `_slanczos`, `_slanczos_jit`, `_compute_distance_matrix`, `_get_unique_rows`, `_subsample_knots` (see §5.4) |
| `jaxgam/smooths/tprs.py` | Delete the relocated function bodies; replace with `from jaxgam.smooths.utils import …`; swap the inline knot-subsample block for a `_subsample_knots()` call. No behavioral change. |
| `jaxgam/smooths/registry.py` | Add `"gp": GaussianProcessSmooth`. Registration alone enables both direct GP (`s(..., bs="gp")`) and GP tensor margins (`te(..., bs="gp")` / `ti(..., bs="gp")`) — the existing `TensorProductSmooth._create_marginals()` at `tensor.py:146` dispatches through `get_smooth_class(self.spec.bs)`, so no tensor-side code change is needed. |
| `jaxgam/smooths/tensor.py` | **No code change.** GP tensor margins are intentionally supported via the existing dispatch loop. Do **not** add a `bs == "gp"` guard. |
| `tests/r_bridge.py` | **Extend** `RBridge.smooth_construct()` to (a) accept `knots: dict[str, np.ndarray] \| None = None` and pass it into the R-side `smoothCon(..., knots=...)` call, and (b) extract GP-specific fields `knt` and `gp.defn` from the returned smooth object alongside the existing `Xu`/`UZ`/`shift` (which are TPRS-shaped — `Xu` is empty for GP). See §2.7. **Also add** a module-level helper `gp_config_to_mgcv_m(config: GPConfig, rho: float \| None = None) -> list[float]` that converts a `GPConfig` to mgcv's `m=c(...)` numeric vector (kernel→signed type, then rho, then power for `power_exponential`). Used by `test_validation_matrix.py` and `test_gaussian_process.py` to build the R-side formulas from the Python-side `GPConfig` so the two sides cannot drift. |
| `tests/test_validation_matrix.py` | Add GP smooth configs: direct (`gp`, `gp_2d`, `gp_mixed`) and tensor (`gp_te`, optionally `gp_ti`). Default-kernel cells write `s(...bs='gp')` on both sides verbatim; kernel-specific cells (if added) build the Python formula with `kernel=`/`rho=`/etc. and the R formula via `gp_config_to_mgcv_m`. |
| `tests/conftest.py` | Add `gp_*` data fixtures (1D, 2D direct, 2D tensor — the tensor fixture can reuse the 2D-direct data). |

### 11.3 Possibly Modified

| File | Change | Trigger |
|---|---|---|
| `jaxgam/jax_utils.py` | Indefinite-penalty logdet path | If §8.3 testing reveals issues |
| `tests/test_smooths/test_utils.py` (new or existing) | Move any direct `_slanczos` / distance-matrix tests from `test_tprs.py` | Optional follow-up |

Note: the PR 0 refactor (§5.4) is a mechanical code move — TPRS tests
must continue to pass unchanged. Treat that as a merge gate: if any TPRS
test changes numerical output, the move is wrong.

### 11.4 No Changes Needed

| File | Reason |
|---|---|
| `jaxgam/formula/parser.py` | `extra_args` already accepts arbitrary Python literals (`kernel="..."`, `rho=0.5`, `power=2.0`, `stationary=True`, `xt={...}`). GP-specific validation (including the `m=` rejection) lives in `parse_gp_config`, not the parser. |
| `jaxgam/registry.py` | Existing generic `Registry[T]` is reused for `gp_kernel_registry` — same abstraction as `smooth_registry`, `family_registry`, `link_registry`. |
| `jaxgam/smooths/constraints.py` | Standard centering/gam.side suffice |
| `jaxgam/smooths/base.py` | Base interface is sufficient |
| `jaxgam/penalties/penalty.py` | Standard `Penalty(S, rank, null_space_dim)` |
| `jaxgam/fitting/*` | GP is transparent to PIRLS/REML (one λ, one penalty) |
| `jaxgam/summary/summary.py` | Standard smooth p-value path (`type_=0`) |
| `jaxgam/formula/design.py` | No `SmoothInfo` changes |

---

## 12. Testing Strategy

Testing follows the project's two-tier pattern:

- **`tests/test_smooths/test_gaussian_process.py`** — basis-level unit
  tests + smooth-construct R comparisons.
- **`tests/test_validation_matrix.py`** — full-model R comparisons in the
  parametrized matrix, inheriting all existing test methods.

### 12.0 Consolidation Discipline (from `docs/clean_unit_tests/`)

The recent test-suite cleanup (`docs/clean_unit_tests/implementation_plan.md`)
established hard rules about test-count proliferation that GP must
respect:

1. **`tests/test_validation_matrix.py`** is the canonical owner of broad
   final-model R parity. Per-smooth files like
   `test_gaussian_process.py` must **not** duplicate `GAMResults` field
   checks (deviance, fitted_values, edf, coefficients, theta) — those
   live in the matrix's `test_matches_r`.

2. **`_AssertCollector`** (`tests/helpers.py`) is the required pattern
   when multiple assertions share an expensive fixture or R fit. Adding
   N assertions that all run against the same `setup()` or R fit must
   produce **one** collected test, not N.

3. **Parameterize**, do not enumerate. Five kernels become
   `@pytest.mark.parametrize("kernel", [...])` over a single test
   method, not five `def test_kernel_N(...)` methods. This is the same
   pattern Commit D of the cleanup applied to cubic splines (cr/cs/cc).

4. **Per-smooth files own layer-specific behavior**:
   - Smooth-construct R parity (basis/penalty/design-matrix shape vs R).
   - Setup invariants (null_space_dim, rank, stored shift/UZ/gp_defn).
   - Kernel-evaluator math (closed-form per kernel).
   - Parser / formula tests for `m=`, `xt=`.

   They do **not** own end-to-end fit parity — that's the validation
   matrix's job.

### 12.0.1 Expected collected-test footprint

| Where | New collected tests | Pattern |
|---|---|---|
| `test_gp_kernels.py` — kernel math | ~3 (parametrized over 5 kernels + ρ resolution + per-kernel validation) | `@parametrize` over `kernel` name + `_AssertCollector` |
| `test_gp_kernels.py` — `parse_gp_config` | ~2 (defaults + alias resolution; `m=` rejection + invalid kernel/rho/power) | `_AssertCollector` |
| `test_gp_kernels.py` — `_gp_T` | ~1 (stationary vs non-stationary shapes) | `_AssertCollector` |
| `test_gaussian_process.py` — setup invariants | ~2-3 (one consolidated per stationarity mode + one for dimensions) | `_AssertCollector` |
| `test_gaussian_process.py` — knot selection | ~2 (subsample reproducibility + global-RNG-untouched) | direct |
| `test_gaussian_process.py` — prediction edge cases | ~2 (training roundtrip + out-of-support) | direct |
| `test_gaussian_process.py` — R smooth-construct | ~3 (one per: kernel-matrix STRICT, penalty STRICT, X via `X@X.T` STRICT) — kernel parameter is a `@parametrize` axis | `@parametrize` + `_AssertCollector` |
| `test_validation_matrix.py` — GP cells | **30** (3 configs × 5 families × 2 methods/cell), or **40-50** if tensor `gp_te` / `gp_ti` included | matrix consolidation |
| `test_formula/test_parser.py` — GP parsing | ~3 (positive cases for `kernel=`/`rho=`/`power=`/`stationary=`/`xt`; `c(...)` raises at parser layer; `m=...` parses fine — rejection is at setup time, tested in `test_gp_kernels.py`) | `_AssertCollector` |
| **Total** | **~50-65 new collected tests** | |

If a draft pushes past ~70, that's a signal the consolidation
discipline slipped — go back and apply `_AssertCollector` /
`@parametrize` before moving on.

### 12.1 Validation Matrix Integration

GP adds **5 new smooth configs** (three direct: `gp`, `gp_2d`,
`gp_mixed`; two tensor: `gp_te`, optionally `gp_ti`) and **4 new
data generators** (`_make_gp_1d_data`, `_make_gp_2d_data`,
`_make_gp_1d_par_data`, `_make_gp_te_2d_data`), expanding the matrix
by 20–25 cells (5 configs × 5 families, or 20 if `gp_ti` is dropped).

#### 12.1.1 New Smooth Configs

```python
# Added to SMOOTH_CONFIGS in test_validation_matrix.py

"gp": SmoothConfig(
    py_formula="y ~ s(x, bs='gp')",
    r_formula="y ~ s(x, bs='gp')",
    data_type="gp_1d",
),
"gp_2d": SmoothConfig(
    py_formula="y ~ s(x, z, bs='gp', k=30)",
    r_formula="y ~ s(x, z, bs='gp', k=30)",
    data_type="gp_2d",
),
"gp_mixed": SmoothConfig(
    py_formula="y ~ x + s(x, bs='gp')",
    r_formula="y ~ x + s(x, bs='gp')",
    data_type="gp_1d_par",
),
"gp_te": SmoothConfig(
    # Tensor product of two univariate GP margins. JaxGAM's tensor
    # wrapper passes a single scalar k to every margin
    # (tensor.py:140), so Python writes k=5 and R must write
    # k=c(5, 5) to land on the same per-margin basis size.
    py_formula="y ~ te(x1, x2, bs='gp', k=5)",
    r_formula="y ~ te(x1, x2, bs='gp', k=c(5, 5))",
    data_type="gp_te_2d",
),
"gp_ti": SmoothConfig(
    # Optional: tensor *interaction* (pure interaction, no main
    # effects). Useful for ANOVA decomposition test coverage.
    py_formula="y ~ s(x1, bs='gp', k=5) + s(x2, bs='gp', k=5) "
                "+ ti(x1, x2, bs='gp', k=5)",
    r_formula="y ~ s(x1, bs='gp', k=5) + s(x2, bs='gp', k=5) "
              "+ ti(x1, x2, bs='gp', k=c(5, 5))",
    data_type="gp_te_2d",
),
```

These cover:
- `gp` — direct univariate GP with default Matérn 3/2.
- `gp_2d` — direct multivariate GP (one joint kernel, one knot set,
  one λ).
- `gp_mixed` — direct 1-D GP **on the same variable that also appears
  parametrically** (`x + s(x, bs='gp')`). This is the real gam.side
  collision case: the non-stationary GP's null space contains a linear
  column in `x`, which is collinear with the parametric `x` term;
  `gam.side()` must detect and drop the redundant column. Using a
  *different* parametric variable would not exercise this code path —
  the null-space linear trend lives in the smooth's variable, not in
  an unrelated parametric covariate.
- `gp_te` — **tensor product** of two univariate GP margins via the
  existing `TensorProductSmooth` wrapper. Distinct construction from
  `gp_2d` (per-margin kernels and knot sets, multiple λ). Exercises
  the registry → `_create_marginals()` → tensor-product path with
  `GaussianProcessSmooth` as the margin class.
- `gp_ti` (optional but recommended) — tensor **interaction** smooth.
  Pulls the main effects out as separate 1-D GP smooths and lets
  `ti()` carry only the pure interaction. Exercises
  `_svd_reparameterize` (`tensor.py:52-123`) on GP margins.

Additional kernel-specific configs may be added incrementally:
- `gp_spherical` — `m=c(1, 0.5)`.
- `gp_squared_exp` — `m=c(2, 0.3, 2)`.
- `gp_stationary` — `m=c(-3)` (stationary Matérn 3/2).

#### 12.1.2 New Data Generators

```python
def _make_gp_1d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """1D GP data: smooth function of a single continuous predictor."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, n)
    eta = np.sin(3 * np.pi * x) * 0.8 + np.cos(2 * np.pi * x) * 0.4
    return _eta_to_response(eta, family_name, rng, x_extra=None)


def _make_gp_2d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """2D GP data: smooth bivariate function (kriging-style)."""
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) + 0.3 * (x + z)
    return _eta_to_response(eta, family_name, rng, x_extra={"x": x, "z": z})


def _make_gp_1d_par_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """1D GP + parametric on the *same* x: exercises gam.side dropping
    the redundant linear-trend column from the GP null space."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(0, 1, n)
    # Linear trend in x absorbed by the parametric term; smooth carries
    # the curvature only. gam.side() must drop the GP's linear-trend
    # null-space column to avoid singularity.
    eta = 0.7 * x + np.sin(3 * np.pi * x) * 0.6
    return _eta_to_response(eta, family_name, rng, x_extra={"x": x})


def _make_gp_te_2d_data(family_name: str, seed: int = SEED) -> pd.DataFrame:
    """Bivariate data for *tensor* GP (te / ti) tests.

    Variable names `x1` / `x2` distinguish from the direct-GP
    `gp_2d_data` (which uses `x` / `z`) so the same matrix row can
    drive both pathways without column collisions in the validation-
    matrix loop. The signal mixes a separable component (te can fit
    perfectly) with a pure interaction component (only ti / te can
    fit) so the test data is informative for both smooths.
    """
    rng = np.random.default_rng(seed)
    n = 400
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    eta = (
        np.sin(2 * np.pi * x1)
        + np.cos(2 * np.pi * x2)
        + 0.5 * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
    )
    return _eta_to_response(eta, family_name, rng, x_extra={"x1": x1, "x2": x2})
```

Helper `_eta_to_response` reuses the family→response mapping from
existing RE data generators.

#### 12.1.3 Tolerance Rules

```python
# In _r_tol(): GP follows the same rule as TPRS (single-sp direct GPs
# get MODERATE on Gaussian). Tensor GPs inherit the existing
# tensor MODERATE rule via "te" / "ti" handling — listing them
# explicitly here is belt-and-braces.
if family_name == "gaussian" and smooth_key in (
    "tp", "cr", "re",
    "gp", "gp_2d",           # direct GP
    "gp_te", "gp_ti",        # tensor-margin GP
):
    return MODERATE
return LOOSE
```

```python
# In _compare_fitted_not_coefs(): both direct and tensor GP have
# eigenvector sign ambiguity (direct) and tensor SVD-reparam
# ambiguity (tensor), so compare fitted values not coefficients.
return smooth_key in (
    "tp", "tp_by", "te", "ti", "te_by", "cr_by", "re_mixed",
    "gp", "gp_2d", "gp_mixed",      # direct GP
    "gp_te", "gp_ti",               # tensor-margin GP
)
```

#### 12.1.4 Inherited Tests

Adding GP configs automatically inherits all `TestValidationMatrix` and
`TestHardGateInvariants` methods:

- Deviance, fitted values, EDF, scale, theta (NB) vs R.
- Self-prediction roundtrip.
- Convergence, no-NaN, deviance ≥ 0, EDF bounds, Vp PSD, penalty PSD,
  model matrix rank.

**Note on `penalty_psd`**: this test is unconditionally safe for GP.
The §8.3 clip in Commit D normalizes the GP penalty diagonal to
non-negative values before it ever reaches fitting, and tensor GP
penalties are built from the same clipped per-margin penalties.
**No relaxation, no `xfail`.** A `test_penalty_psd` failure on a GP
cell means the clip is broken — fix Commit D, do not relax the gate.

### 12.2 Unit Tests (`test_gaussian_process.py`)

Lives in `tests/test_smooths/test_gaussian_process.py` following the
pattern of `test_tprs.py`, `test_cubic.py`, `test_random_effects.py`.

**Structural tests (no R):**
- `null_space_dim` = 1 stationary / d+1 non-stationary.
- `rank` = `bs.dim − null_space_dim`.
- Penalty is diagonal with correct eigenvalues on first `k` entries.
- `side_constrain == True`, no `_random`, no `_has_centering_constraint`
  override.
- `predict_matrix` reproduces `build_design_matrix` on training data.
- `_shift` matches `colMeans(x)` of training data.
- `_config` is a populated `GPConfig` and `_resolved_rho` is positive.

**Kernel evaluator tests (no R; in `test_gp_kernels.py`):**
- All 5 kernels evaluated at known `(d, ρ)` match closed-form formulas
  at STRICT tolerance. Parametrize over the kernel **name string**
  (`"spherical"`, `"power_exponential"`, `"matern_3_2"`, `"matern_5_2"`,
  `"matern_7_2"`).
- Spherical kernel is exactly 0 for `e > 1`.
- Power-exponential with `power=2.0` matches `exp(-(d/ρ)²)`.
- `ρ` auto-defaults to `max(distance_matrix)` when `config.rho is None`.
- `ρ` user-supplied (positive float) is used verbatim.
- Alias resolution: `"matern32"` and `"matern_3_2"` produce the same
  kernel instance (registry caches by canonical key).
- Unknown kernel name → `KeyError` from
  `gp_kernel_registry.get_instance`.
- `PowerExponentialKernel.validate` rejects `power ≤ 0` and `power > 2`
  with a clear ValueError; other kernels ignore `power`.

**`parse_gp_config` tests (no R; in `test_gp_kernels.py`):**
- Defaults: empty `extra_args` → Matérn 3/2, `rho=None`, `power=1.0`,
  `stationary=False`.
- `m=` rejection: `parse_gp_config({"m": [3, 0.5]})` raises
  `ValueError` whose message contains both "kernel=" and "rho=".
- Alias normalization: `{"kernel": "matern32"}` →
  `GPKernelName.MATERN_3_2`.
- `rho <= 0` raises (auto requires omission/`None`, not 0).

**Stationarity tests (no R):**
- `stationary=True` → `null_space_dim == 1`, `_stationary == True`.
- `stationary=False` (default) → `null_space_dim == d + 1`.
- Default `extra_args` (no `stationary` key) → non-stationary.

**Knot selection tests (no R):**
- `n ≤ max_knots` → all unique rows used as knots.
- `n > max_knots` → exactly `max_knots` knots sampled.
- Same `seed` → identical knot set (reproducibility).
- Different `seed` → different knot set.
- Global RNG state untouched after `setup()`.

**Dimension tests (no R):**
- 1D: default `bs.dim = 12`.
- 2D: default `bs.dim = 33`.
- 3D: default `bs.dim = 104` (one structural test confirming d=3 works
  end-to-end — direct multivariate GP, optional but recommended).
- `d > 3` without explicit `k` raises clear error.
- `bs.dim < d + 2` warning + bump to minimum.
- `nrow(xu) < bs.dim` raises clear error.

**Univariate-margin invariant (no R, supports the tensor pathway):**
`GaussianProcessSmooth` instantiated with a single-variable
`SmoothSpec` (i.e. exactly what `TensorProductSmooth._create_marginals()`
hands it at `tensor.py:137-148`) must:
- `setup()` cleanly.
- Produce a design matrix with `bs.dim` columns and a diagonal penalty.
- Expose `_s_scale` (the `Smooth._smoothcon_normalize` output) so the
  tensor wrapper can undo the normalization via `S_raw = S_normalized
  * marginal._s_scale` (`tensor.py:152`).
- Keep `_noterp = False` (the `Smooth` default) so
  `_svd_reparameterize` runs.

One consolidated test method confirms all of the above with a
`_AssertCollector`; this is what makes the no-code-change tensor
integration safe to land.

**Prediction edge cases (no R):**
- Training-data roundtrip.
- New data within support: smooth predictions.
- New data outside support: GP component → 0 (kernel decay), null-space
  linear trend extrapolates.
- Spherical kernel: predictions beyond `ρ` from any knot are zero (in the
  penalized block).

**R smooth-construct comparison (requires R + mgcv):**

The Python side always uses `kernel=`/`rho=`/`power=`/`stationary=`
kwargs. The R-side formula is built via
`gp_config_to_mgcv_m(config)` (§11.2), so both sides agree on the
underlying kernel without manual translation.

*Direct GP:*
- `s(x, bs="gp")` 1D with explicit knots: compare X, S, rank,
  null_space_dim at STRICT tolerance (post-construction, no fitting).
- `s(x, z, bs="gp", k=30)` 2D with explicit knots: compare at MODERATE
  for raw X (eigenvector sign ambiguity may flip column signs) and at
  STRICT for `X @ X.T` (sign-invariant).
- All 5 kernels: compare `E` matrix at STRICT for fixed knots and the
  matching `GPConfig` ↔ `m=c(...)` pair.
- Stationary vs non-stationary: compare null-space columns.

*Tensor GP margins:* one consolidated parametrized test that
constructs the tensor smooth on both sides and compares the *fitted*
quantities (raw coefficients are tensor-reparameterization-sensitive,
so STRICT only on `X @ X.T` for the marginal blocks and on `S`
embedded into the joint coefficient space):
```python
@pytest.mark.parametrize("py_formula,r_formula,wrapper", [
    ("te(x1, x2, bs='gp', k=5)",
     "te(x1, x2, bs='gp', k=c(5, 5))", "te"),
    ("ti(x1, x2, bs='gp', k=5)",
     "ti(x1, x2, bs='gp', k=c(5, 5))", "ti"),
])
def test_tensor_gp_margin_construct(py_formula, r_formula, wrapper, gp_te_2d_data):
    ...  # one R fit per case, collector for marginal-X@X.T,
         # marginal penalty, embedded tensor penalty.
```

The strategy for STRICT R parity on direct GP is: **supply explicit
`knots=` in both R and Python** so the knot-sampling difference
(Section 10.6) is eliminated. Then the only remaining divergence is
eigenvector signs, which we either match (via deterministic Lanczos
starting vector) or work around (compare `X @ X.T` instead of `X` for
the penalized block).

For tensor GP we cannot supply explicit knots into the marginal
constructors (mgcv's `te` does not accept per-margin `knots=`), so
tensor R-parity is MODERATE on raw quantities; STRICT only on the
sign-invariant marginal `X @ X.T`.

### 12.3 Tolerance Summary

| What | Location | Tolerance | Rationale |
|---|---|---|---|
| Kernel evaluator (closed-form) | unit | STRICT | Deterministic arithmetic |
| Penalty diagonal values | unit + smooth-construct | STRICT | Eigenvalues are sign-determined |
| Design matrix X | smooth-construct | MODERATE | Eigenvector sign / row-order |
| X @ X.T (penalized block) | smooth-construct | STRICT | Sign-invariant |
| Deviance / fitted values | validation matrix | MODERATE/LOOSE | Standard fitting tolerance |
| Coefficients | validation matrix | n/a — compare fitted values | Sign ambiguity |
| EDF | validation matrix | MODERATE/LOOSE | Derived from fitting |
| Hard-gate invariants | validation matrix | per-invariant | Structural (some relaxed for indefinite S) |

### 12.4 Test Data Fixtures

Add to `tests/conftest.py`:

```python
@pytest.fixture
def gp_1d_data():
    """Univariate continuous data for GP smooth tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    x = rng.uniform(0, 1, n)
    return {"x": x}


@pytest.fixture
def gp_2d_data():
    """Bivariate continuous data for 2D GP tests."""
    rng = np.random.default_rng(SEED)
    n = 200
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    return {"x": x, "z": z}


@pytest.fixture
def gp_explicit_knots_data():
    """Small dataset with explicit knot positions for STRICT R comparison."""
    rng = np.random.default_rng(SEED)
    n = 100
    x = rng.uniform(0, 1, n)
    knots = {"x": np.linspace(0.05, 0.95, 10)}
    return {"data": {"x": x}, "knots": knots}
```

### 12.5 Known Test Gaps

The following are documented gaps requiring follow-up:

1. **Knot subsampling parity with R**: R's `sample()` algorithm differs
   from numpy's. We accept that subsampled-knot GPs cannot be compared
   STRICTLY to R; we test full-knot or explicit-knot configurations
   strictly, and fall back to MODERATE for the validation matrix.
2. **Eigenvector sign**: even with identical Lanczos starting vector, R
   and our `_slanczos` may differ in sign for individual eigenvectors.
   Tests compare `X @ X.T` or fitted values, not raw `X`.
3. **`d > 3` default `k`**: mgcv has a latent bug here; we raise instead.
   No R comparison is attempted.

---

## 13. Implementation Plan

See [implementation_plan.md](implementation_plan.md) for the detailed
commit-by-commit breakdown.

**Working model** (following the pattern established in
`docs/clean_unit_tests/implementation_plan.md`):

- **One PR total** at the end, against `main`.
- All work lands on a **single working branch** as a sequence of
  discrete commits (Commits A, B, C, …).
- **The agent does NOT run `git commit` or `git push`.** Each commit unit
  is executed end-to-end by the agent (read instructions, make changes,
  run `make test-cov`, stop and hand off). The user reviews, commits
  manually, then triggers the next unit.
- Validation is a single command per unit: `make test-cov` (full Docker
  suite + R 4.5.2 + mgcv 1.9-3 + ≥80% coverage gate). Do **not** also
  run `make test` or `make test-local`.

### High-level commit sequence

| Commit | Scope | Anchored to design §  |
|---|---|---|
| A | Pre-flight baseline capture (test counts, coverage, wall-clock) | — |
| B | Extract shared kriging helpers from `tprs.py` to `utils.py` (the four pieces in §1.4.1; TPRS tests must pass unchanged) | §1.4, §5.4 |
| C | GP-specific kernel evaluator `_gp_E` + null-space builder `gpT` + STRICT unit tests for all 5 kernels. Use `tol=np.finfo(float).eps ** 0.5` for the eigendecomp call site (§5.4.1). | §3.3, §5.3 |
| D | `GaussianProcessSmooth` class (setup, knot harvest, eigendecomp, design matrix, penalty) + structural unit tests. **Includes the indefinite-eigenvalue clip (§8.3) — not conditional, always on, with a regression test that builds an indefinite-spectrum case and asserts the clip + warning fire.** | §5.1–§5.10, §8.3 |
| E | Registry wire-up (registering `"gp"` enables **both** direct GP and tensor-margin GP automatically via `tensor.py:146`'s existing dispatch — no guard, no tensor-side code). Includes the univariate-margin invariant test (§12.2) that proves the tensor pathway is safe to land in Commit H. **Parser tests use Python literals only (`m=[3, 0.5]`, `xt={"max_knots": 500}`) — see §6.1 for why R-style `c()` / `list()` are out of scope.** | §6, §11.2, §12.2 |
| F | **(NEW) RBridge GP enhancements.** Extend `RBridge.smooth_construct()` with a `knots=` argument and add `knt` / `gp.defn` extraction (see §2.7). Add direct bridge-level tests that round-trip a tiny GP fit. **No GP-side tests yet** — this is pure test-infrastructure work that enables Commit G. | §2.7 |
| G | R smooth-construct comparison tests for **direct GP** (STRICT with explicit knots via the new bridge `knots=` arg; MODERATE for default-knot configurations) plus a tensor-margin smooth-construct test that confirms the registry + `_create_marginals()` path produces a usable basis end-to-end. Was Commit F. | §12.2 |
| H | Validation matrix integration: **5 configs** — direct (`gp`, `gp_2d`, `gp_mixed` with the same-variable `y ~ x + s(x, bs='gp')` formula) and tensor (`gp_te`, optionally `gp_ti`) via the existing `TensorProductSmooth` / `TensorInteractionSmooth` wrappers. + data generators (incl. `_make_gp_te_2d_data`). Was Commit G. | §12.1 |
| ~~(old H)~~ | ~~Indefinite-penalty robustness — **deleted**, handled inside Commit D's clip; current fitting cannot tolerate true indefinite penalties (§8.3).~~ | — |
| I | Documentation updates (`docs/api.md`, `docs/index.md` smooth catalog) | — |
| J | Final sweep: re-run baseline, confirm targets, verify ownership of new tests | — |

Commit B is the load-bearing refactor; Commits C, D, E, F are
infrastructure that enables R-parity testing in G; Commits G and H are
the parity surface. There is no longer a contingent commit — the
indefinite-penalty case is handled deterministically in D.

See `implementation_plan.md` for: per-commit "What to do", file lists,
validation commands, exit criteria, and the hard allow-list of behaviors
that must not regress.
