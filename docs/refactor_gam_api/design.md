# GAM API Refactor: Design Document

## 1. Problem Statement

The `GAM` class in `jaxgam/api.py` (693 LOC) acts as a monolithic orchestrator that:

- Holds model **specification** (formula, family, method) *and* fitted **results** (30+ attributes)
- Uses a mutable `_fitted` flag with `_check_fitted()` guards to prevent unfitted access
- Contains a 90-line `_store_results()` that computes derived quantities (EDF, covariance, null deviance) and sets all fitted attributes atomically
- Mixes responsibilities: validation, device resolution, fitting orchestration, result storage, prediction, summary delegation, plotting delegation

While the underlying architecture is well-separated (Phase 1/2/3 with clear module boundaries), the GAM class itself conflates "what model to fit" with "what came out of the fit."

### What Is *Not* a Problem

The GAM class is already a **thin facade** — the actual algorithm work is delegated to `formula/`, `fitting/`, `summary/`, and `plot/` modules. The issue is not algorithm coupling; it's **state management and API shape**.

---

## 2. Design Goals

1. **Separate specification from results** — a fitted model should be a distinct, immutable object
2. **Eliminate the `_fitted` guard pattern** — impossible states should be unrepresentable
3. **Reduce `_store_results()` complexity** — derived-quantity computation should be cohesive and testable
4. **Maintain the three-phase boundary** — the Phase 1 (NumPy) → Phase 2 (JAX) → Phase 3 (NumPy) contract is load-bearing and must not be violated
5. **Keep the user-facing API ergonomic** — sklearn-style `model.fit(data).predict(newdata)` chaining must still work
6. **Pythonic, not Java-ic** — prefer composition over inheritance, protocols over ABCs, dataclasses over boilerplate

---

## 3. Design: Model/Results Separation

### 3.1 Prior Art

This pattern is well-established in the Python stats ecosystem:

| Library | Specification | Fitted Result |
|---------|--------------|---------------|
| **statsmodels** | `GLM(endog, exog, family)` | `GLMResults` (from `.fit()`) |
| **scikit-learn** | `LinearRegression()` | mutates self (`.coef_`, `.intercept_`) |
| **PyMC** | `Model()` context | `InferenceData` (from `sample()`) |
| **brms (R)** | `brm(formula, data, family)` | `brmsfit` object |
| **mgcv (R)** | `gam(formula, family, data)` | returns a `gam` object (spec + results combined) |

The **statsmodels pattern** is the strongest fit for jaxgam:
- Both are frequentist regression frameworks
- Both have formula-based model specification
- Both produce rich result objects (coefficients, covariance, diagnostics)
- statsmodels' separation is considered one of its best design decisions

### 3.2 Proposed Architecture

```
GAM (specification — lightweight, immutable-ish)
│
├── formula: str
├── family: str | ExponentialFamily
├── method: "REML" | "ML"
├── sp: array | None
├── device: "cpu" | "gpu" | None
│
└── fit(data, weights?, offset?) ─────────►  GAMResults (frozen dataclass)
                                                  │
                                                  ├── coefficients_: ndarray
                                                  ├── fitted_values_: ndarray
                                                  ├── linear_predictor_: ndarray
                                                  ├── Vp_: ndarray
                                                  ├── edf_: ndarray
                                                  ├── ... (all fitted state)
                                                  │
                                                  ├── predict(newdata?, ...)
                                                  ├── summary()
                                                  ├── plot(...)
                                                  └── predict_matrix(newdata)
```

### 3.3 `GAM` Class (Specification)

```python
class GAM:
    """Generalized Additive Model specification.

    Parameters
    ----------
    formula : str
        Model formula, e.g. "y ~ s(x1) + s(x2, bs='cr')".
    family : str or ExponentialFamily
        Response distribution. One of "gaussian", "poisson", "binomial", "gamma".
    method : str
        Smoothing parameter selection method. "REML" or "ML".
    sp : array-like or None
        Fixed smoothing parameters. If None, estimated via `method`.
    """

    def __init__(
        self,
        formula: str,
        family: str | ExponentialFamily = "gaussian",
        method: str = "REML",
        sp: np.ndarray | list | None = None,
        **kwargs,
    ) -> None: ...

    def fit(
        self,
        data: pd.DataFrame | dict,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ) -> "GAMResults": ...
```

**Responsibilities (narrowed):**
- Store specification parameters
- Validate scope guards (`_check_scope_guards`)
- Orchestrate the fit pipeline: parse → build → transfer → optimize → wrap results
- Return a `GAMResults` instance

**What moves out:**
- All fitted attributes (`coefficients_`, `Vp_`, `edf_`, etc.)
- `predict()`, `summary()`, `plot()`, `predict_matrix()`
- `_check_fitted()` (no longer needed — results are a separate object)
- `_store_results()` (becomes `GAMResults.from_fit()` or similar)

### 3.4 `GAMResults` Class (Fitted Model)

```python
@dataclass(frozen=True)
class GAMResults:
    """Results from a fitted GAM.

    All attributes are read-only (frozen dataclass). This object is the
    primary interface for post-estimation: prediction, inference, and
    visualization.
    """

    # ── Core estimates ──────────────────────────────────────
    coefficients: np.ndarray        # (p,) fitted coefficients
    fitted_values: np.ndarray       # (n,) response-scale fitted values
    linear_predictor: np.ndarray    # (n,) link-scale linear predictor

    # ── Covariance & scale ──────────────────────────────────
    Vp: np.ndarray                  # (p, p) Bayesian posterior covariance
    scale: float                    # dispersion parameter (phi)

    # ── Degrees of freedom ──────────────────────────────────
    edf: np.ndarray                 # (n_smooths,) per-smooth EDF
    edf1: np.ndarray                # (n_smooths,) alternative EDF
    edf_total: float                # total model EDF

    # ── Deviance ────────────────────────────────────────────
    deviance: float
    null_deviance: float

    # ── Smoothing parameters ────────────────────────────────
    smoothing_params: np.ndarray    # (n_penalties,) estimated lambda

    # ── Convergence ─────────────────────────────────────────
    converged: bool
    n_iter: int
    score: float                    # REML/ML value at convergence

    # ── Model structure (Phase 1 artifacts) ─────────────────
    family: ExponentialFamily
    setup: ModelSetup               # frozen Phase 1 output
    coef_map: CoefficientMap        # Phase 1→3 coefficient mapping
    smooth_info: tuple[SmoothInfo, ...]
    term_names: tuple[str, ...]

    # ── Data references ─────────────────────────────────────
    X: np.ndarray                   # (n, p) design matrix
    y: np.ndarray                   # (n,) response
    weights: np.ndarray             # (n,) prior weights
    offset: np.ndarray | None

    # ── Metadata ────────────────────────────────────────────
    n: int
    execution_path: str
    lambda_strategy: str
    formula: str                    # echoed from specification
    training_data: dict[str, np.ndarray]  # for plotting

    # ── Methods ─────────────────────────────────────────────
    def predict(
        self,
        newdata: pd.DataFrame | dict | None = None,
        pred_type: str = "response",
        se_fit: bool = False,
        offset: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...

    def predict_matrix(
        self, newdata: pd.DataFrame | dict,
    ) -> np.ndarray: ...

    def summary(self) -> "GAMSummary": ...

    def plot(
        self,
        select: int | list | None = None,
        pages: int = 0,
        rug: bool = True,
        se: bool = True,
        shade: bool = True,
        **kwargs,
    ) -> tuple: ...
```

**Key design decisions:**

1. **Frozen dataclass** — all attributes are set at construction and immutable. This eliminates the entire class of bugs where someone accidentally mutates fitted state.

2. **No trailing underscores** — the sklearn convention (`coef_`) distinguishes fitted from unfitted attributes on the *same* object. With separate types, this distinction is structural, not naming-based. The underscore suffix is unnecessary.

3. **Methods on the dataclass** — `predict()`, `summary()`, `plot()` live on `GAMResults` because they need fitted state. This is the same as statsmodels' `Results.predict()`.

4. **Factory classmethod for construction** — a `_from_fit()` classmethod handles the derived-quantity computation that currently lives in `_store_results()`:

```python
@classmethod
def _from_fit(
    cls,
    result: NewtonResult,
    setup: ModelSetup,
    spec: FormulaSpec,
    data: pd.DataFrame | dict,
    family: ExponentialFamily,
    fd: FittingData,
    lambda_strategy: str,
) -> "GAMResults":
    """Construct GAMResults from raw fit output.

    Computes derived quantities: covariance, EDF, null deviance.
    """
    # ... (current _store_results logic, cleaned up)
```

### 3.5 Backward Compatibility via `__getattr__`

To avoid a hard API break, `GAM` can forward attribute access to an internal `results_` attribute:

```python
class GAM:
    def fit(self, data, ...) -> GAMResults:
        self.results_ = GAMResults._from_fit(...)
        return self.results_

    def __getattr__(self, name):
        # Forward fitted attribute access: model.coefficients_ → model.results_.coefficients
        if name.endswith("_") and name != "results_":
            stripped = name.rstrip("_")
            if hasattr(self, "results_") and hasattr(self.results_, stripped):
                import warnings
                warnings.warn(
                    f"Accessing '{name}' on GAM is deprecated. "
                    f"Use 'results.{stripped}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return getattr(self.results_, stripped)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # Thin forwarding methods (deprecated)
    def predict(self, *args, **kwargs):
        warnings.warn("Use results.predict() instead", DeprecationWarning)
        return self.results_.predict(*args, **kwargs)
```

**Migration path:**
1. v1.0: Both APIs work; old API emits `DeprecationWarning`
2. v1.1+: Remove `__getattr__` forwarding

### 3.6 Naming: Why Not `FittedGAM`?

Considered alternatives:

| Name | Pros | Cons |
|------|------|------|
| `GAMResults` | Matches statsmodels convention; clearly "results of a fit" | Slightly verbose |
| `FittedGAM` | Reads naturally ("a fitted GAM") | Implies it *is* a GAM, which conflates spec and results |
| `GAMFit` | Short | Ambiguous: is it a noun (the fit) or a verb (to fit)? |
| `GAMEstimate` | Precise | Uncommon in Python stats |

**Decision: `GAMResults`** — follows the dominant Python convention and is unambiguous.

---

## 4. Internal Composition within `GAMResults`

### 4.1 Do We Need `GAMFit`, `GAMPredict`, `GAMSummary` as Separate Classes?

Let's evaluate the original brainstorm against what we now know:

**GAMFit as a class:**
- The fitting logic is already delegated to `fitting/newton.py` and `fitting/pirls.py`
- What remains in GAM is orchestration (~40 lines) and result construction (~90 lines)
- A separate `GAMFit` class would either duplicate this delegation or just be a thin wrapper around `newton_optimize()`
- **Verdict: Not needed as a class.** The `GAM.fit()` method + `GAMResults._from_fit()` classmethod cover this cleanly.

**GAMPredict as a class:**
- Prediction logic is ~30 lines (build matrix, matmul, link inverse, optional SE)
- It needs: `setup`, `family`, `coefficients`, `Vp`, `linear_predictor`
- Extracting this into a class would require passing all that state, which is exactly what `GAMResults` already holds
- **Verdict: Not needed as a separate class.** `GAMResults.predict()` is the right home. If prediction grows more complex (e.g., term-wise prediction, newdata validation), it can be extracted into a `_predict` module that `GAMResults` delegates to — same pattern as `summary/`.

**GAMSummary as a class:**
- Already exists: `jaxgam/summary/summary.py` computes summary, `GAMSummary` is a frozen dataclass
- `GAM.summary()` is a 3-line delegation: check fitted → call `_summary(self)` → return
- **Verdict: Already well-factored.** No change needed.

### 4.2 What *Should* Be Extracted

The real complexity is in `_store_results()`, which does too many things:

1. Convert JAX arrays to NumPy
2. Compute covariance via Cholesky solve
3. Compute hat matrix F
4. Compute per-smooth EDF and EDF1
5. Back-transform reparameterized coefficients
6. Compute Bayesian covariance Vp
7. Compute null deviance
8. Extract training data for plotting
9. Set 30+ attributes

**Proposal: Extract a `PostEstimation` module** (`jaxgam/post_estimation.py` or expand `fitting/reml.py`):

```python
# jaxgam/post_estimation.py

@dataclass(frozen=True)
class PostEstimationResults:
    """Derived quantities computed from raw fit output."""
    coefficients: np.ndarray
    Vp: np.ndarray
    edf: np.ndarray
    edf1: np.ndarray
    edf_total: float
    null_deviance: float
    hat_matrix: np.ndarray  # F, kept for debugging

def compute_post_estimation(
    result: NewtonResult,
    setup: ModelSetup,
    family: ExponentialFamily,
    fd: FittingData,
) -> PostEstimationResults:
    """Compute all derived quantities from raw fit output.

    This function encapsulates the covariance computation, EDF
    calculation, coefficient back-transformation, and null deviance
    — everything that currently lives in GAM._store_results().
    """
    ...
```

**Benefits:**
- Independently testable (unit test covariance computation without fitting a model)
- Clear inputs/outputs (NewtonResult → PostEstimationResults)
- Reduces `GAMResults._from_fit()` to: call `compute_post_estimation()` + assemble the dataclass

---

## 5. Full Proposed Module Layout

```
jaxgam/
├── __init__.py              # Public API: GAM, GAMResults
├── api.py                   # GAM class (specification + fit orchestration)
├── results.py               # GAMResults frozen dataclass + methods
├── post_estimation.py       # Derived-quantity computation (from _store_results)
├── jax_utils.py             # (unchanged)
├── formula/                 # (unchanged)
├── smooths/                 # (unchanged)
├── families/                # (unchanged)
├── links/                   # (unchanged)
├── penalties/               # (unchanged)
├── fitting/                 # (unchanged)
├── summary/                 # (unchanged)
├── plot/                    # (unchanged)
```

**Changes from original layout:**
- `api.py` slimmed to specification + orchestration (results/post-estimation extracted)
- New `results.py` (GAMResults frozen dataclass)
- New `post_estimation.py` (extracted from `_store_results`)
- `__init__.py` exports both `GAM` and `GAMResults`

---

## 6. Interaction with Three-Phase Architecture

The refactor **strengthens** the three-phase boundary:

```
Phase 1 (NumPy)          Phase 2 (JAX)           Phase 3 (NumPy)
─────────────────        ──────────────           ─────────────────
GAM.fit() calls:         newton_optimize()        GAMResults holds:
  parse_formula()          pirls_loop()             .predict()
  ModelSetup.build()       ↓                        .summary()
  FittingData.from()     NewtonResult               .plot()
       │                      │
       │    jax.device_put    │     np.asarray
       ├─────────────────────►├──────────────────►
       │                      │
       │                      │     compute_post_estimation()
       │                      │            │
       │                      │     GAMResults._from_fit()
```

- `GAM` owns Phase 1 orchestration and the Phase 1→2 handoff
- `GAMResults` owns Phase 3 (post-estimation methods)
- `compute_post_estimation()` sits at the Phase 2→3 boundary
- The Phase 2 code (`fitting/`) is completely untouched

---

## 7. `predict()` Method Design

Prediction currently handles four cases. This logic stays on `GAMResults`:

```python
def predict(
    self,
    newdata: pd.DataFrame | dict | None = None,
    pred_type: str = "response",
    se_fit: bool = False,
    offset: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if newdata is None:
        eta = self.linear_predictor
        if offset is not None:
            eta = eta + offset
    else:
        X_p = self.setup.build_predict_matrix(newdata)
        eta = X_p @ self.coefficients
        if offset is not None:
            eta = eta + offset

    if pred_type == "response":
        mu = self.family.link.linkinv(eta)
    else:
        mu = eta

    if not se_fit:
        return mu

    # SE computation
    if newdata is None:
        X_p = self.X
    se = np.sqrt(np.sum((X_p @ self.Vp) * X_p, axis=1))
    return mu, se
```

**If prediction grows complex** (v1.1+: term-wise prediction, exclude terms, confidence intervals), extract to `jaxgam/prediction.py` with `GAMResults.predict()` as a thin delegation.

---

## 8. Summary of Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Model/Results separation** over mixins | Eliminates impossible states structurally; follows statsmodels convention |
| **Frozen dataclass** for results | Immutability prevents accidental mutation; all state set at construction |
| **No `GAMFit` class** | Fitting is already delegated to `fitting/`; GAM.fit() is just orchestration |
| **No `GAMPredict` class** | Prediction is simple (~30 LOC); lives naturally on GAMResults |
| **Extract `post_estimation` module** | Makes derived-quantity computation testable and reduces results construction complexity |
| **`GAMResults`** naming | Matches statsmodels; unambiguous |
| **No trailing underscores on results** | Structural separation replaces naming convention |
| **Backward-compat `__getattr__`** | Gradual migration; old code emits deprecation warnings |
| **`api.py` + `results.py` + `post_estimation.py`** | Each file has a single clear responsibility; rename to `gam.py` skipped to reduce churn |

---

## 9. Resolved Design Decisions

1. **`GAM.fit()` stores `self.results_`** in addition to returning `GAMResults`. This enables both `results = model.fit(data)` and `model.fit(data); model.results_.predict(...)` usage patterns, and makes deprecation forwarding work naturally.

2. **`summary()` and `plot()` accept `GAMResults`** directly instead of the full GAM. Internal modules are updated to work with the results object.

3. **`GAMResults` stores `formula` and `family` as metadata** — no back-reference to the `GAM` specification object, avoiding circular dependencies.

4. **`Ve_` (frequentist covariance) is omitted** from `GAMResults`. No placeholders — add when implemented.
