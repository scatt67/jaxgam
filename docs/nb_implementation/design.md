# Negative Binomial Family Implementation Design

**Status:** Design Phase
**Date:** 2026-03-21
**Branch:** `implement-nb-distribution`

---

## Table of Contents

1. [Overview](#1-overview)
2. [R Reference Implementation Analysis](#2-r-reference-implementation-analysis)
3. [Architectural Decision: Which R Family to Port](#3-architectural-decision-which-r-family-to-port)
4. [Mathematical Specification](#4-mathematical-specification)
5. [Theta Optimization Strategy](#5-theta-optimization-strategy)
6. [Extending the custom_jvp for Theta](#6-extending-the-custom_jvp-for-theta)
7. [Family Class Design](#7-family-class-design)
8. [PIRLS Integration](#8-pirls-integration)
9. [Newton Optimizer Changes](#9-newton-optimizer-changes)
10. [Phase-by-Phase Changes](#10-phase-by-phase-changes)
11. [File Plan](#11-file-plan)
12. [Testing Strategy](#12-testing-strategy)
13. [Risk Register](#13-risk-register)
14. [Implementation Order](#14-implementation-order)

---

## 1. Overview

### 1.1 What Is the Negative Binomial Family?

The Negative Binomial (NB) distribution models count data with overdispersion
(variance exceeds the mean). It generalizes Poisson by adding a dispersion
parameter theta (also called "size" or "shape"):

- **Mean:** mu
- **Variance:** mu + mu^2/theta
- As theta -> infinity, NB -> Poisson

In mgcv, NB is an **extended family** -- unlike standard families (Gaussian,
Poisson, Binomial, Gamma), it has an extra distributional parameter theta that
must be estimated alongside the smoothing parameters and coefficients.

### 1.2 Why This Is High-Risk

NB touches nearly every layer of the fitting pipeline:

1. **Family layer:** New `ExtendedFamily` base class between `ExponentialFamily`
   and `NegativeBinomial`, with pure-function factories (`deviance_fn`,
   `working_weights_fn`) for AD inside the custom_jvp.
2. **custom_jvp on PIRLS:** Must be extended to compute `d(beta*)/d(theta)` via
   the implicit function theorem, matching R's `ift2` in `gdi.c`.
3. **Newton outer loop:** `params` vector grows from `[log_lambda]` to
   `[log_lambda, log_theta]`. The REML criterion and its gradient/Hessian
   include theta dimensions.
4. **REML criterion:** The saturated log-likelihood `ls_sat` must take
   `log_theta` as an explicit JAX-traced argument so `jax.grad` captures it.

### 1.3 Scope

This design covers the `nb()` extended family with:
- Estimated theta (default) and fixed theta modes
- Log link (default), identity link, sqrt link
- Joint theta optimization in the existing Newton optimizer
- Full R-matching correctness

Out of scope (deferred):
- `negbin()` standard family (fixed theta with grid search)
- Truncated NB, zero-inflated NB
- Other extended families (Tweedie, Beta, SHASH, etc.) -- though NB
  establishes the infrastructure they will reuse
- Fellner-Schall optimizer -- the family interface is designed to not block
  a future EFS implementation (which would use `estimate_theta()` separately
  rather than joint optimization), but EFS itself is not built here

---

## 2. R Reference Implementation Analysis

mgcv provides **two** NB implementations.

### 2.1 `negbin()` -- Standard Family (Fixed theta)

**File:** `$MGCV_SOURCE/R/gam.fit3.r` lines 2564-2642

- theta is **fixed** (passed as argument, never estimated during fitting)
- Uses `gam.fit3` -- standard PIRLS with `V(mu) * g'(mu)^2` weights
- Class is `"family"` (standard), not `"extended.family"`

### 2.2 `nb()` -- Extended Family (Estimated theta)

**File:** `$MGCV_SOURCE/R/efam.r` lines 161-310

- theta can be **estimated** (default) or fixed
- Uses `gam.fit4` -- extended PIRLS with `Dd()` deviance derivatives
- theta stored internally as `log(theta)` for unconstrained optimization
- `n.theta = 1` when estimated, `0` when fixed
- `getTheta(trans=FALSE)` returns log(theta); `getTheta(trans=TRUE)` returns theta
- `Dd()` provides deviance derivatives up to 4th order w.r.t. mu plus
  mixed partials w.r.t. theta (efam.r lines 207-237)
- `ls()` provides saturated log-likelihood with first and second derivatives
  w.r.t. theta via digamma/trigamma (efam.r lines 248-275)

### 2.3 How R's Newton Jointly Optimizes Theta

This was a critical finding during design. R's Newton optimizer **does** jointly
optimize theta -- it is NOT an outer iteration.

The `sp` vector passed through the pipeline is `[theta, log_lambda, <log_scale>]`.
`gam.fit3` dispatches to `gam.fit4` for extended families (gam.fit3.r line 111).
`gam.fit4` returns `REML1` and `REML2` -- the gradient and Hessian of the REML
criterion w.r.t. the **full** parameter vector including theta (gam.fit4.r
lines 738-764). `newton()` then takes a joint Newton step over all dimensions.

The REML gradient has four components for the theta dimensions:

| Component | R variable | Source | What it captures |
|---|---|---|---|
| Deviance gradient | `D1[1:nth]` | `gdi2` C routine | `d(beta*)/d(theta) . dD/d(eta) + dD/d(theta)` -- both indirect (through beta) and direct |
| Penalty gradient | `P1[1:nth]` | `gdi2` C routine | `2 * beta' . S . d(beta*)/d(theta)` -- indirect through beta only |
| Log-det gradient | `ldet1[1:nth]` | `gdi2` C routine | `tr(H^{-1} . dH/d(theta))` -- through working weights changing with theta |
| Saturated loglik | `lsth1` | `family$ls()` | `d(ls_sat)/d(theta)` -- direct via digamma/trigamma |

The `d(beta*)/d(theta)` terms come from `ift2` (gdi.c lines 1368-1462) which
applies the implicit function theorem:

```
d(beta*)/d(theta) = -H^{-1} . 0.5 . X' . d^2D/(d(eta) d(theta))
```

where `d^2D/(d(eta) d(theta))` is the mixed derivative of deviance w.r.t.
linear predictor and theta. R computes this analytically via `Dd$Dmuth` ->
`dDeta$Detath` (gam.fit4.r lines 4-77).

The `estimate.theta()` function (efam.r lines 5-96) is used only by the
**EFS** score type (gam.fit4.r line 509: `if (scoreType=="EFS")`), not by
Newton REML/ML.

### 2.4 Key R Source File Map

| File | Lines | What |
|---|---|---|
| `$MGCV_SOURCE/R/efam.r` | 161-310 | `nb()` extended family definition |
| `$MGCV_SOURCE/R/efam.r` | 5-96 | `estimate.theta()` (used by EFS only) |
| `$MGCV_SOURCE/R/gam.fit3.r` | 2564-2642 | `negbin()` standard family |
| `$MGCV_SOURCE/R/gam.fit3.r` | 106-116 | `gam.fit3` dispatch to `gam.fit4` |
| `$MGCV_SOURCE/R/gam.fit4.r` | 4-77 | `dDeta()` -- mu-to-eta derivative conversion |
| `$MGCV_SOURCE/R/gam.fit4.r` | 240-548 | `gam.fit4()` -- PIRLS for extended families |
| `$MGCV_SOURCE/R/gam.fit4.r` | 725-764 | REML criterion with theta derivatives |
| `$MGCV_SOURCE/src/gdi.c` | 1368-1462 | `ift2()` -- IFT for d(beta*)/d(theta) |
| `$MGCV_SOURCE/src/gdi.c` | 1953-2250 | `gdi2()` -- full derivative computation |

---

## 3. Architectural Decision: Which R Family to Port

**Decision: Port `nb()` (extended family).**

1. Theta estimation is the key feature -- users rarely know theta a priori.
2. `negbin()` (fixed theta) is trivial to add later as `NegativeBinomial(theta=5.0, fixed=True)`.
3. NB establishes the extended family infrastructure for Tweedie, Beta, etc.

**Key design choice: AD replaces hand-coded derivatives.**

R's `gam.fit4` uses `Dd()` -- ~50 lines of hand-coded analytical derivatives
of the deviance up to 4th order, passed to the C routine `gdi2`. We replace
all of this with `jax.grad` through stable forward passes (per design.md
section 9.3). No `custom_jvp` needed for NB itself -- only for the PIRLS IFT.

---

## 4. Mathematical Specification

### 4.1 Log-Likelihood (Per-Observation)

```
l(y, mu, theta) = lgamma(y + theta) - lgamma(theta) - lgamma(y + 1)
                  + theta * log(theta) - theta * log(mu + theta)
                  + y * log(mu) - y * log(mu + theta)
```

### 4.2 Variance Function

```
V(mu) = mu + mu^2/theta
V'(mu) = 1 + 2*mu/theta
```

### 4.3 Deviance Residuals

Per-observation unit deviance (R's `dev.resids`, efam.r lines 199-205):

```
d_i = 2 * wt_i * [y_i * log(max(1, y_i) / mu_i)
                   - (y_i + theta) * log((y_i + theta) / (mu_i + theta))]
```

### 4.4 Saturated Log-Likelihood

R's `family$ls()` (efam.r lines 248-275):

```
ls = -sum(wt * [(y + theta)*log(y + theta) - ylogy + lgamma(y + 1)
                - theta*log(theta) + lgamma(theta) - lgamma(theta + y)])
```

where `ylogy = y*log(y)` for `y > 0`, else `0`.

### 4.5 Initialize

R's initialization (efam.r line 280): `mustart = y + (y == 0) / 6`

---

## 5. Theta Optimization Strategy

### 5.1 Joint Optimization Matching R's Newton

R's Newton jointly optimizes `[theta, log_lambda]` with the full REML gradient
and Hessian. We match this: `log_theta` is appended to the `params` vector
and `jax.grad`/`jax.hessian` of the REML criterion capture all theta
dependencies automatically.

The params vector for NB is:

```
params = [log_lambda_1, ..., log_lambda_m, log_theta]
```

This follows the identical pattern used for unknown-scale families (Gaussian,
Gamma) where `log_phi` is appended:

```
params = [log_lambda_1, ..., log_lambda_m, log_phi]
```

The static flag `joint_theta` (analogous to `joint_scale`) controls whether
`log_theta` is present in `params`.

### 5.2 How Theta Enters the AD Trace

Theta affects the REML criterion through two paths:

**Path A: Direct (outside PIRLS).** The saturated log-likelihood `ls_sat(theta)`
depends on theta directly. This is handled by calling
`family.saturated_loglik_theta(y, wt, phi, log_theta)` with `log_theta` as a
traced JAX value inside `_diff_score`. `jax.grad` differentiates through it
automatically. This is structurally identical to how `phi` enters for
Gaussian/Gamma via `family.saturated_loglik(y, wt, phi)`.

**Path B: Indirect (through PIRLS).** The PIRLS solution `(beta*, XtWX, deviance)`
depends on theta because the deviance formula and variance function both contain
theta. R captures this via `ift2` (the implicit function theorem in `gdi.c`).
We capture it by extending the `custom_jvp` on PIRLS to include `log_theta` as
a primal -- see Section 6.

Both paths are needed for the full gradient. Path A alone (the "partial joint"
approach considered earlier) would miss 3 of 4 gradient terms and give poor
Newton convergence in the theta dimension.

### 5.3 Design Constraint: Future Optimizer Compatibility

The family interface provides both:
- **Mutable state** (`get_theta`/`put_theta` + methods reading `self._log_theta`)
  -- used by PIRLS at runtime and available for future optimizers (e.g. EFS)
  that estimate theta separately
- **Explicit-argument pure functions** (`deviance_fn`, `working_weights_fn`,
  `saturated_loglik_theta`) -- used by Newton's custom_jvp and criterion

This separation means a future optimizer can use the same family object with a
different theta-update strategy without any family changes.

---

## 6. Extending the custom_jvp for Theta

### 6.1 Current custom_jvp (Standard Families)

The existing custom_jvp (newton.py lines 179-222) has 2 primals:

```python
_pirls_out(S_lambda, beta_warm) -> (beta, XtWX, dev)
```

The JVP computes how `(beta, XtWX, dev)` change when `S_lambda` is perturbed,
using the IFT:

```
dbeta = -H^{-1} . (dS @ beta)        # IFT for lambda
deta = X @ dbeta
dW = JVP of W(eta) w.r.t. eta        # chain through working weights
ddev = JVP of D(eta) w.r.t. eta      # chain through deviance
dXtWX = X' . diag(dW) . X
```

### 6.2 Extended custom_jvp (Extended Families)

For `n_theta > 0`, the custom_jvp gains a 3rd primal:

```python
_pirls_out(S_lambda, log_theta, beta_warm) -> (beta, XtWX, dev)
```

The JVP now handles tangents `(dS, dtheta, _)`:

**Step 1: IFT with both lambda and theta contributions.**

The IFT at the PIRLS stationary point `d(0.5*PenDev)/d(beta) = 0`:

```
For lambda: d^2(0.5*PenDev)/(d(beta) d(S)) in direction dS = dS . beta
For theta:  d^2(0.5*PenDev)/(d(beta) d(theta)) . dtheta = 0.5 * X' . d^2D/(d(eta) d(theta)) . dtheta
```

R computes `d^2D/(d(eta) d(theta))` analytically via `Dd$Dmuth` -> `dDeta$Detath`.
We compute it with JAX AD:

```python
dev_fn = family.deviance_fn(y, wt)  # pure function D(eta, log_theta) -> scalar
grad_D_eta = jax.grad(dev_fn, argnums=0)  # dD/d(eta)
_, d_grad_D = jax.jvp(
    lambda lt: grad_D_eta(eta, lt),
    (log_theta,), (dtheta,)
)  # d^2D/(d(eta) d(theta)) . dtheta, shape (n,)
```

Combined IFT solve:

```python
rhs = dS @ beta + 0.5 * (X.T @ d_grad_D)
dbeta = cho_solve((L, True), -rhs)
```

This matches R's `ift2` (gdi.c lines 1396-1408) where the first `n_theta`
columns of `b1` use `Db_th = -0.5 * X' @ Det_th` and the remaining columns
use `Db_th = -sp[i] * S_i @ beta`.

**Step 2: ddev and dW via joint (eta, theta) JVPs.**

Unlike standard families where only eta changes, here both eta and theta change.
We use joint JVPs to capture both effects in one call:

```python
deta = X @ dbeta
_, ddev = jax.jvp(dev_fn, (eta, log_theta), (deta, dtheta))
_, dW = jax.jvp(ww_fn, (eta, log_theta), (deta, dtheta))
dXtWX = (X.T * dW) @ X
```

This captures:
- `ddev = dD/d(eta) . deta + dD/d(theta) . dtheta` (indirect + direct)
- `dW = dW/d(eta) . deta + dW/d(theta) . dtheta` (indirect + direct)

Matching all four R gradient terms from Section 2.3.

**Step 3: Standard families unchanged.**

When `family.n_theta == 0` (compile-time constant via static `family` arg),
the standard 2-primal custom_jvp is used. Zero cost for existing families.

### 6.3 Performance Impact

Adding `log_theta` as a 3rd primal adds one tangent direction. For
`jax.hessian`, cost goes from `O(m^2)` to `O((m+1)^2)` JVP passes where
`m` = number of smoothing parameters. Since the theta JVP involves a scalar
tangent (vs the `(p,p)` matrix tangent for `S_lambda`), each theta-related
pass is cheap. The increase is marginal.

### 6.4 Concrete Code Structure in `_diff_score`

```python
def _diff_score(params, beta_warm, X, y, wt, offset, S_list, ...,
                family, ..., joint_theta, joint_scale, n_lambda, ...):

    # Parse params
    idx = n_lambda
    log_lambda = params[:idx]
    if joint_theta:
        log_theta = params[idx]; idx += 1
    if joint_scale:
        phi = jnp.exp(params[idx])
    else:
        phi = jnp.array(1.0)

    # ls_sat -- direct theta dependence (outside custom_jvp)
    if joint_theta:
        ls_sat = family.saturated_loglik_theta(y, wt, phi, log_theta)
    else:
        ls_sat = family.saturated_loglik(y, wt, phi)

    S_lambda = build_S_lambda(log_lambda, S_list, p)

    if family.n_theta > 0 and joint_theta:
        # --- Extended family path: 3 primals ---
        dev_fn = family.deviance_fn(y, wt)
        ww_fn = family.working_weights_fn(wt)

        @jax.custom_jvp
        def _pirls_out(S_lam, lt, bw):
            result = pirls_loop(X, y, bw, S_lam, family, wt, offset, ...)
            return result.coefficients, result.XtWX, result.deviance

        @_pirls_out.defjvp
        def _pirls_jvp(primals, tangents):
            S_lam, lt, bw = primals
            dS, dlt, _ = tangents
            beta, XtWX, dev = _pirls_out(S_lam, lt, bw)

            H = XtWX + S_lam
            L, _ = cho_factor(H)
            eta = X @ beta + offset

            # IFT: lambda + theta
            rhs = dS @ beta
            grad_D_eta = jax.grad(dev_fn, argnums=0)
            _, d_grad_D = jax.jvp(
                lambda lt_: grad_D_eta(eta, lt_), (lt,), (dlt,)
            )
            rhs = rhs + 0.5 * (X.T @ d_grad_D)
            dbeta = cho_solve((L, True), -rhs)
            deta = X @ dbeta

            # ddev and dW: joint JVPs over (eta, theta)
            _, ddev = jax.jvp(dev_fn, (eta, lt), (deta, dlt))
            _, dW = jax.jvp(ww_fn, (eta, lt), (deta, dlt))
            dXtWX = (X.T * dW) @ X

            return (beta, XtWX, dev), (dbeta, dXtWX, ddev)

        beta, XtWX, dev = _pirls_out(S_lambda, log_theta, beta_warm)

    else:
        # --- Standard family path: 2 primals (existing, unchanged) ---
        @jax.custom_jvp
        def _pirls_out(S_lam, bw):
            ...  # existing code

        beta, XtWX, dev = _pirls_out(S_lambda, beta_warm)

    # Criterion (unchanged)
    core = _criterion_core(log_lambda, XtWX, beta, dev, ls_sat, ...)
    ...
```

---

## 7. Family Class Design

### 7.1 ExtendedFamily Base Class

Extended families have extra distributional parameters (theta) that are jointly
estimated with smoothing parameters via Newton. The `ExtendedFamily` base class
sits between `ExponentialFamily` and concrete families like `NegativeBinomial`:

```
ExponentialFamily          (variance, deviance_resids, aic, initialize, ...)
  |
  +-- Gaussian, Binomial, Poisson, Gamma    (n_theta = 0, standard families)
  |
  +-- ExtendedFamily       (get_theta, put_theta, deviance_fn, working_weights_fn, ...)
        |
        +-- NegativeBinomial                 (n_theta = 1)
        +-- (future: Tweedie, Beta, SHASH)
```

```python
# families/extended.py

class ExtendedFamily(ExponentialFamily):
    """Base class for families with extra parameters estimated via Newton.

    Extended families provide:
    - Mutable theta state (get_theta/put_theta) for PIRLS runtime
    - Pure-function factories (deviance_fn, working_weights_fn) for the
      custom_jvp on PIRLS, where theta must be a traced JAX value
    - Explicit-theta saturated loglik for the REML criterion AD trace

    The theta interface uses arrays of shape (n_theta,) throughout,
    so families with multiple extra parameters (e.g. scat with df + scale)
    work without interface changes.

    The fitting code branches on `family.n_theta > 0` (compile-time check)
    to select the extended custom_jvp path. The abstractmethods here
    ensure any new extended family implements the required interface.
    """

    @abstractmethod
    def get_theta(self, transformed: bool = False) -> np.ndarray:
        """Extra parameter vector, shape (n_theta,).

        Log-scale by default. transformed=True returns natural scale
        (e.g. exp(log_theta) for positive parameters).
        """
        ...

    @abstractmethod
    def put_theta(self, log_theta: np.ndarray) -> None:
        """Set extra parameter vector (log-scale), shape (n_theta,).

        Called by Newton after each accepted step.
        """
        ...

    @abstractmethod
    def deviance_fn(self, y: np.ndarray, wt: np.ndarray):
        """Return pure JAX function D(eta, log_theta_vec) -> scalar.

        log_theta_vec has shape (n_theta,).
        Used by the custom_jvp for IFT theta terms and joint JVPs.
        Must capture (y, wt, link) in closure; theta is an explicit arg.
        """
        ...

    @abstractmethod
    def working_weights_fn(self, wt: np.ndarray):
        """Return pure JAX function W(eta, log_theta_vec) -> (n,) array.

        log_theta_vec has shape (n_theta,).
        Used by the custom_jvp for joint dW JVPs.
        Must capture (wt, link) in closure; theta is an explicit arg.
        """
        ...

    @abstractmethod
    def saturated_loglik_theta(
        self, y: np.ndarray, wt: np.ndarray, scale: float, log_theta: np.ndarray
    ):
        """Saturated log-likelihood with explicit theta for AD trace.

        log_theta has shape (n_theta,).
        Called inside _diff_score where log_theta is a traced JAX array.
        jax.grad differentiates through this w.r.t. log_theta.
        """
        ...
```

This is ~30 lines with no logic -- just abstract method declarations that
enforce the contract. `ExponentialFamily` is unchanged; `n_theta: int = 0`
stays on it so the fitting code can check `family.n_theta > 0` without
caring about the type hierarchy.

### 7.2 NegativeBinomial Class

```python
# families/negative_binomial.py

class NegativeBinomial(ExtendedFamily):

    family_name: str = "nb"
    scale_known: bool = True   # phi = 1
    n_theta: int = 1           # overridden in __init__ based on theta arg

    def __init__(self, theta=1.0, *, fixed=False, link=None):
        super().__init__(link)
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        self._log_theta = np.array([np.log(theta)])
        self.n_theta = 0 if fixed else 1

    @property
    def default_link(self) -> Link:
        return LogLink()

    # -- ExtendedFamily interface --

    def get_theta(self, transformed=False):
        if transformed:
            return np.exp(self._log_theta)
        return self._log_theta

    def put_theta(self, log_theta):
        self._log_theta = np.asarray(log_theta, dtype=np.float64).reshape(self._log_theta.shape)
```

### 7.3 Standard Family Methods (Read Theta from self)

Used by PIRLS (Phase 2 runtime) and post-estimation (Phase 3). These
read `self._log_theta` which is fixed during a single PIRLS run.

All standard family methods read `self._log_theta[0]` (scalar) from the
stored vector. The `[0]` index extracts the single NB dispersion parameter.
Future multi-theta families (e.g. scat) would index multiple elements.

```python
    def variance(self, mu):
        xp = array_module(mu)
        theta = xp.exp(self._log_theta[0])
        return mu + mu**2 / theta

    def dvar(self, mu):
        theta = jnp.exp(self._log_theta[0])
        return 1.0 + 2.0 * mu / theta

    def deviance_resids(self, y, mu, wt):
        # R: efam.r lines 199-205
        xp = array_module(y)
        theta = xp.exp(self._log_theta[0])
        mu_safe = xp.maximum(mu, 1e-10)
        y_safe = xp.where(y > 0, y, 1.0)
        d = 2.0 * wt * (
            y * xp.log(y_safe / mu_safe)
            - (y + theta) * xp.log((y + theta) / (mu_safe + theta))
        )
        d = xp.maximum(d, 0.0)
        return xp.sign(y - mu_safe) * xp.sqrt(d)

    def saturated_loglik(self, y, wt, scale):
        # R: efam.r lines 248-275 (forward pass only, no theta derivs)
        theta = jnp.exp(self._log_theta[0])
        ylogy = jnp.where(y > 0, y * jnp.log(y), 0.0)
        term = (
            (y + theta) * jnp.log(y + theta) - ylogy
            + jsp.gammaln(y + 1.0) - theta * jnp.log(theta)
            + jsp.gammaln(theta) - jsp.gammaln(theta + y)
        )
        return -jnp.sum(term * wt)

    def aic(self, y, mu, wt, scale):
        # R: efam.r lines 239-246. Phase 3 only (NumPy).
        theta = np.exp(self._log_theta[0])
        mu_safe = np.maximum(mu, 1e-10)
        term = (
            (y + theta) * np.log(mu_safe + theta) - y * np.log(mu_safe)
            + gammaln(y + 1.0) - theta * np.log(theta)
            + gammaln(theta) - gammaln(theta + y)
        )
        return float(2.0 * np.sum(term * wt))

    def initialize(self, y, wt):
        # R: mustart <- y + (y == 0)/6
        y_arr = np.asarray(y, dtype=float)
        return np.where(y_arr == 0, y_arr + 1.0/6.0, y_arr)

    def valid_mu(self, mu):
        return mu > 0

    def valid_eta(self, eta):
        return np.isfinite(eta)
```

### 7.4 Pure-Function Factories (Explicit Theta for AD)

Used by the custom_jvp (Section 6) and by the criterion's `ls_sat` term.
These return pure JAX functions that take `log_theta` as an explicit array
argument of shape `(n_theta,)` so it participates in the AD trace.
NB indexes `log_theta[0]` for its single parameter.

```python
    def saturated_loglik_theta(self, y, wt, scale, log_theta):
        """Saturated log-likelihood with explicit theta for AD trace.
        log_theta has shape (n_theta,) = (1,) for NB."""
        theta = jnp.exp(log_theta[0])
        ylogy = jnp.where(y > 0, y * jnp.log(y), 0.0)
        term = (
            (y + theta) * jnp.log(y + theta) - ylogy
            + jsp.gammaln(y + 1.0) - theta * jnp.log(theta)
            + jsp.gammaln(theta) - jsp.gammaln(theta + y)
        )
        return -jnp.sum(term * wt)

    def deviance_fn(self, y, wt):
        """Return pure JAX function D(eta, log_theta_vec) -> scalar.
        log_theta_vec has shape (n_theta,) = (1,) for NB.

        Used by the custom_jvp for IFT theta terms and joint JVPs.
        Captures (y, wt, link) in closure; theta is an explicit arg.
        """
        link_inv = self.link.inverse
        def _dev(eta, log_theta):
            theta = jnp.exp(log_theta[0])
            mu = link_inv(eta)
            mu_safe = jnp.maximum(mu, 1e-10)
            y_safe = jnp.where(y > 0, y, 1.0)
            return jnp.sum(2.0 * wt * (
                y * jnp.log(y_safe / mu_safe)
                - (y + theta) * jnp.log((y + theta) / (mu_safe + theta))
            ))
        return _dev

    def working_weights_fn(self, wt):
        """Return pure JAX function W(eta, log_theta_vec) -> (n,) array.
        log_theta_vec has shape (n_theta,) = (1,) for NB.

        Used by the custom_jvp for joint dW JVPs.
        Captures (wt, link) in closure; theta is an explicit arg.
        """
        link_inv = self.link.inverse
        link_deriv = self.link.derivative
        def _ww(eta, log_theta):
            theta = jnp.exp(log_theta[0])
            mu = link_inv(eta)
            V = mu + mu**2 / theta
            g_prime = link_deriv(mu)
            return wt / (V * g_prime**2)
        return _ww
```

### 7.5 Why This Interface

| Method | Who calls it | Why theta handling differs |
|---|---|---|
| `variance(mu)` | PIRLS hot loop (JIT) | Reads `self._log_theta` -- theta is fixed during PIRLS |
| `deviance_resids(y, mu, wt)` | PIRLS, post-estimation | Same -- fixed theta |
| `saturated_loglik(y, wt, scale)` | Criterion (fixed theta) | Standard families, or NB with fixed theta |
| `saturated_loglik_theta(y, wt, scale, log_theta)` | Criterion (joint theta) | `log_theta` is a traced JAX value for `jax.grad` |
| `deviance_fn(y, wt)` | custom_jvp only | Returns `D(eta, log_theta)` -- both args traced for IFT + JVP |
| `working_weights_fn(wt)` | custom_jvp only | Returns `W(eta, log_theta)` -- both args traced for dW JVP |

Standard families don't implement the last three methods (they're `@abstractmethod`
on `ExtendedFamily`, not on `ExponentialFamily`). The `if family.n_theta > 0`
branch in `_diff_score` (compile-time constant) ensures they're never called
for standard families.

---

## 8. PIRLS Integration

### 8.1 No PIRLS Changes Required

For the **log link** (canonical for NB), Fisher scoring weights from
`wt / (V(mu) * g'(mu)^2)` give correct results. The existing `pirls_loop`
calls `family.working_weights(mu, wt)` and `family.working_response(y, mu, eta)`,
which NB inherits from `ExponentialFamily` (using `variance` and `link`).

NB overrides `variance(mu)` to return `mu + mu^2/theta` and `dev_resids` for
the NB deviance formula. Everything else works unchanged.

### 8.2 Non-Canonical Links (Deferred)

For identity and sqrt links, R's `gam.fit4` uses observed-Hessian weights
via `Dd$Deta2` instead of Fisher weights. This would require overriding
`working_weights` with AD-based observed weights. Deferred -- log link covers
>95% of NB use cases.

---

## 9. Newton Optimizer Changes

### 9.1 NewtonOptimizer.__init__

```python
self._joint_theta = fd.family.n_theta > 0 and fd.n_penalties > 0
```

Add `joint_theta` to `_jit_kwargs` as a static arg.

### 9.2 Initial Params Construction

```python
params = log_lambda
if self._joint_theta:
    log_theta_init = jnp.array(fd.family.get_theta())
    params = jnp.concatenate([params, log_theta_init[None]])
if self._joint_scale:
    params = jnp.concatenate([params, log_phi_init[None]])
```

### 9.3 After Each Accepted Step

Update the family's stored theta so the next PIRLS run uses the correct value:

```python
if self._joint_theta:
    new_log_theta = float(params_new[self._fd.n_penalties])
    self._fd.family.put_theta(new_log_theta)
```

This triggers JIT recompilation (family is a static arg with changed internal
state). Acceptable cost: theta changes once per Newton iteration (~5-10 times
total), recompilation is ~50ms vs ~500ms PIRLS work.

### 9.4 _build_result

Extract `log_lambda` from the joint params vector (skip theta/phi components)
for the returned `NewtonResult`. Store estimated theta on the result.

### 9.5 Clamping

`_clamp_params` should only clamp `log_lambda`, not `log_theta`. Theta has no
natural bounds (though very large `|log_theta|` values are unlikely).

---

## 10. Phase-by-Phase Changes

### Phase 1 (NumPy, Setup)

| File | Change |
|---|---|
| `families/extended.py` | **New.** `ExtendedFamily` base class (Section 7.1) |
| `families/negative_binomial.py` | **New.** `NegativeBinomial` class (Section 7.2-7.4) |
| `families/registry.py` | Register `"nb"` key |
| `families/__init__.py` | Export `ExtendedFamily`, `NegativeBinomial` |

### Phase 2 (JAX, Fitting)

| File | Change |
|---|---|
| `fitting/newton.py` | Extend `_diff_score` with 3-primal custom_jvp (Section 6.4). Add `joint_theta` flag to `NewtonOptimizer`. Update params construction, step acceptance, result building. |
| `fitting/pirls.py` | **No changes.** |
| `fitting/reml.py` | **No changes.** Theta is handled in `_diff_score`, not in the criterion classes. |

### Phase 3 (NumPy, Post-estimation)

| File | Change |
|---|---|
| `results.py` | Store estimated theta on `GAMResults` |
| `api.py` | Detect `n_theta > 0`, pass theta to results |

---

## 11. File Plan

### New Files

```
jaxgam/
  families/
    extended.py              # ExtendedFamily base class (~30 lines)
    negative_binomial.py     # NegativeBinomial class (~250 lines)

tests/
  test_negative_binomial.py  # Unit + AD validation tests (~300 lines)
  test_fitting/
    test_nb_fitting.py       # End-to-end R comparison tests (~250 lines)
```

### Modified Files

```
jaxgam/
  families/registry.py      # Add "nb" entry (~2 lines)
  families/__init__.py       # Export ExtendedFamily, NegativeBinomial (~3 lines)
  fitting/newton.py          # Extended custom_jvp + joint_theta (~80 lines)
  api.py                     # Detect extended family (~10 lines)
  results.py                 # Store theta (~5 lines)
```

---

## 12. Testing Strategy

### 12.1 Unit Tests (test_negative_binomial.py)

**Family methods against R values:**
- `variance(mu)` matches `mu + mu^2/theta` for various theta
- `deviance_resids(y, mu, wt)` matches R's `nb()$dev.resids()`
- `saturated_loglik(y, wt, scale)` matches R's `nb()$ls()$ls`
- `aic(y, mu, wt, scale)` matches R's `nb()$aic()`
- `initialize(y, wt)` matches `y + (y == 0)/6`
- Edge cases: y=0, large y, large theta, theta -> infinity (Poisson limit)

**Theta management:**
- `get_theta()` / `put_theta()` round-trip
- Constructor: positive theta -> fixed (n_theta=0)
- Constructor: negative theta -> estimated, starting at -theta
- Constructor: None -> estimated, starting at 1

**AD validation (per design.md section 9.5):**
- `jax.grad` through `saturated_loglik_theta` matches finite differences
- `jax.grad` through `deviance_fn(y, wt)` matches finite differences
- Mixed derivative `d^2D/(d(eta) d(theta))` matches finite differences
- Test at extreme theta: 0.01, 0.1, 1, 10, 100, 1000, 10000
- Test at extreme y/mu combinations

**custom_jvp validation:**
- `dβ*/d(log_theta)` from IFT matches finite-difference perturbation of theta
- `d(deviance)/d(log_theta)` matches finite differences
- `d(XtWX)/d(log_theta)` matches finite differences
- Full REML gradient w.r.t. `log_theta` matches finite differences

### 12.2 R Comparison Tests (test_nb_fitting.py)

Using `r_bridge.py`:

1. Simple: `y ~ s(x)` with `nb()`, estimated theta
2. Fixed theta: `y ~ s(x)` with `nb(theta=2)`
3. Multiple smooths: `y ~ s(x1) + s(x2)` with `nb()`
4. With offset: `y ~ s(x) + offset(log(exposure))`
5. Factor-by smooth: `y ~ s(x, by=group)` with `nb()`
6. High overdispersion (theta small): test convergence
7. Low overdispersion (theta large, near Poisson): test theta -> infinity

**Tolerances:**
- Coefficients: STRICT (rtol=1e-4)
- Deviance: STRICT
- Theta: MODERATE (rtol=1e-3) -- REML surface is flat near theta optimum
- Smoothing parameters: MODERATE
- EDF: MODERATE
- Fitted values: STRICT

### 12.3 Hard-Gate Invariants

- Deviance >= 0
- No NaN in converged model
- EDF bounds: 1 <= edf_j <= k_j
- Penalty PSD, H symmetry/PSD
- Objective monotonicity (penalized deviance decreases each PIRLS iteration)
- Theta > 0 after estimation

---

## 13. Risk Register

### R1: custom_jvp Correctness

**Risk:** The extended JVP with theta IFT terms could have sign errors or
missing factors (the 0.5 factor from the PenDev convention is subtle).

**Mitigation:** Validate every JVP output against finite differences. Test
`dbeta`, `ddev`, `dXtWX` individually, not just the final REML gradient.
R's `ift2` in `gdi.c` is the reference for the exact formula.

### R2: JIT Recompilation on Theta Change

**Risk:** `put_theta()` changes `self._log_theta` on the family object (static
JIT arg), triggering recompilation.

**Mitigation:** Acceptable cost -- theta changes ~5-10 times per fit,
recompilation is ~50ms vs ~500ms PIRLS. Future optimization: pass theta as a
dynamic arg (requires broader refactor).

### R3: Numerical Stability of AD Through lgamma

**Risk:** `lgamma(y+theta) - lgamma(theta)` cancellation at large theta.

**Mitigation:** Per design.md section 9.3, JAX differentiates `lgamma` to
`digamma` (stable special function). At large theta, NB -> Poisson and
gradient imprecision is harmless. Validated by extreme-theta AD tests.

### R4: Non-Canonical Link Support

**Risk:** Identity/sqrt links need observed-Hessian weights, not Fisher.

**Mitigation:** Deferred. Log link covers >95% of NB use. Document limitation.

---

## 14. Implementation Order

### Step 1: ExtendedFamily Base + NegativeBinomial Family Class
- `families/extended.py` -- `ExtendedFamily` base class (Section 7.1)
- `families/negative_binomial.py` -- all methods from Section 7.2-7.4
- Registry registration
- Unit tests against R values (no fitting yet)

### Step 2: AD Validation
- Finite-difference tests for all pure-function factories
- Mixed derivative `d^2D/(d(eta) d(theta))` validation
- Extreme parameter regime tests

### Step 3: Extended custom_jvp
- Extend `_diff_score` in `newton.py` (Section 6.4)
- `joint_theta` flag and params construction
- Validate JVP outputs against finite differences

### Step 4: Newton Integration
- `NewtonOptimizer` changes (Section 9)
- `put_theta` after step acceptance
- End-to-end test: simple `y ~ s(x)` with NB

### Step 5: Full R Comparison Tests
- All test models from Section 12.2
- Verify STRICT/MODERATE tolerances
- Hard-gate invariants

### Step 6: Post-Estimation
- Store theta on `GAMResults`
- Display in summary
