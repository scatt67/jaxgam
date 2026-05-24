# Gaussian Process Smooth (`bs="gp"`): Implementation Plan

## Overview

Implement the Gaussian process smooth (`bs="gp"`) in jaxgam, porting
mgcv's `smooth.construct.gp.smooth.spec` (R/smooth.r:3441-3552). The
design is fully specified in `docs/gaussian_process/design.md` — this
document is the **execution plan** that converts that design into a
sequence of self-contained commits.

**Branch:** single working branch off `main` for the entire feature; one
PR at the end.
**Design reference:** `docs/gaussian_process/design.md`
**R source reference:** `$MGCV_SOURCE/R/smooth.r` lines 3399-3595 and
`$MGCV_SOURCE/man/smooth.construct.gp.smooth.spec.Rd`

**Scope:** Five kernels (spherical, power-exponential, Matérn 3/2, 5/2,
7/2), 1- to 3-D continuous predictors, stationary + non-stationary
modes, standard sum-to-zero centering and gam.side participation, and
GP as a tensor margin via the existing `te()` / `ti()` wrappers. Out
of scope: anisotropic kernels within a single GP smooth, custom
user-defined kernel functions. See design doc §1.3 for the full scope
boundary.

**Scope status:** As of 2026-05-23, `CLAUDE.md` was updated to remove GP
from §"What Is NOT in v1.0" (the entry previously listed "Exotic smooths
(…, GP)"). GP is now an in-scope v1.0 smooth.

**User-facing API (load-bearing — read before any code):** JaxGAM does
**not** expose mgcv's `m=` parameter. GP kwargs are explicit:
`kernel=` (string, default `"matern_3_2"`), `rho=` (positive float or
omitted), `power=` (float, only used by `kernel="power_exponential"`),
`stationary=` (bool). The kernel module reuses the generic
`jaxgam.registry.Registry[T]` for `gp_kernel_registry`, the same
abstraction that powers `smooth_registry`, `family_registry`, and
`link_registry`. Passing `m=` to a GP smooth raises `ValueError` from
`GaussianProcessSmooth.__init__`. mgcv-side conversion happens **only**
at the R-bridge boundary via `gp_config_to_mgcv_m(spec)` (Commit F).
See design §1.6, §5.1, §5.3, §6.3, §6.4.

**Consolidation discipline (load-bearing — read before any test code):**
This plan inherits the test-suite cleanup rules from
`docs/clean_unit_tests/implementation_plan.md` and the testing rules in
`CLAUDE.md` §Testing Rules. The relevant rules for GP:

1. **`tests/test_validation_matrix.py` is the canonical owner** of broad
   final-model R parity. Per-smooth files must not duplicate
   `GAMResults` field checks.
2. **`tests.helpers._AssertCollector`** is required when multiple
   assertions share an expensive fixture or R fit. N assertions against
   one fixture produce **one** collected test, not N.
3. **Parameterize**, do not enumerate. Five kernels become
   `@pytest.mark.parametrize("kernel", [...])` over a single test
   method, not five separate methods.

Design doc §12.0 / §12.0.1 give the target collected-test footprint:
roughly **~55-80 new collected tests total** across all GP-related
files (wider than a naive enumeration would suggest because of the
Commit-F RBridge tests, the Commit-D indefinite-clip regression, and
the two tensor-margin configs in Commit H — `gp_te` always, `gp_ti`
optionally). The 80 upper bound assumes `gp_ti` is included; drop to
~70 if it's not. If a draft commit pushes past ~85, apply more
consolidation before handing off.

**Commit-letter renumber (from the two design-review patches, 2026-05-23):**
The original plan was A-J with H being conditional. After review the
sequence is:

- A, B, C unchanged.
- D now ships the indefinite-eigenvalue clip (was Commit H).
- E does **not** add a tensor-margin guard (an earlier patch was
  incorrect — mgcv supports tensor-margin GP via `te()` / `ti()`,
  and JaxGAM's existing tensor wrappers handle this through the
  registry once `"gp"` is registered). E adds the registry entry,
  parser tests (Python literals only), and a **univariate-margin
  invariant test** proving `GaussianProcessSmooth` works correctly
  as the per-variable margin that `TensorProductSmooth._create_marginals()`
  builds.
- **F is new**: RBridge GP enhancements (`knots=` argument + extract
  `knt` / `gp.defn`). Pure test-infrastructure work.
- G is what was F (R smooth-construct comparison) **plus** a
  tensor-margin smooth-construct test exercising `te(..., bs='gp')`
  and `ti(..., bs='gp')` through the existing wrappers.
- H is what was G (validation matrix), expanded from 3 to **5**
  configs: direct (`gp`, `gp_2d`, `gp_mixed`) and tensor (`gp_te`,
  optionally `gp_ti`). `gp_mixed` formula corrected to
  `x + s(x, bs='gp')`.
- ~~Old H~~ removed entirely; an empty marker section remains so
  cross-references resolve.
- I, J unchanged.

**No GP-tensor class.** GP appears as a tensor margin via the
existing `TensorProductSmooth` / `TensorInteractionSmooth`
dispatch at `tensor.py:136-148`. Do not create
`gaussian_process_tensor.py` or any analog.

---

## Working Model — How This Plan Is Executed

**One PR total.** The entire GP implementation ships as a single pull
request from one working branch.

**Work is split into discrete commits, not PRs.** Each numbered "Commit"
below is a self-contained unit of work that an agent executes end-to-end.
Multiple commits land on the same branch in sequence.

**The agent must NOT run `git commit` or `git push`.** The user commits
manually after reviewing each unit. The agent's job per commit is:

1. Read the unit's "What to do" and "Files touched" sections.
2. Make the code/test changes.
3. Run `make test-cov` for validation. This single command runs the full
   suite in Docker (with R 4.5.2 + mgcv 1.9-3) and enforces the ≥80%
   coverage gate — do **not** also run `make test` or `make test-local`,
   they are redundant and slow.
4. Stop and surface the results to the user (file list, test count delta,
   coverage delta, validation output). The user reviews, commits, then
   triggers the next unit.

If a unit's validation fails, fix the issue **in the same unit** before
handing off — do not start the next unit on a broken tree. If the fix
changes the scope of the unit, note that in the handoff so the user can
adjust the commit message accordingly.

The user manages: branching off `main`, commit messages, force-pushes
(if any), and the final PR. The agent stays inside the working tree.

---

## Project Tooling

Per CLAUDE.md, this project uses `uv` for dependency management and
`make` for task orchestration. All Python execution must go through `uv
run`.

| Task | Command |
|---|---|
| **Validation (use this — only this)** | `make test-cov` |
| Count collected tests | `uv run pytest --collect-only -q tests \| tail -1` |
| Count source-level tests | `grep -rc "def test_" tests \| awk -F: '{s+=$2} END {print s}'` |
| Lint | `make lint` |
| Pre-commit | `make pre-commit` |

`make test-cov` runs the full suite in Docker (R 4.5.2 + mgcv 1.9-3)
with coverage and the `--cov-fail-under=80` gate. It is a strict
superset of `make test` and `make test-local` plus the coverage check,
so the agent should run it instead of all three.

---

## Hard Allow-List — Must Not Regress

These behaviors must continue to work at every commit. If any of them
breaks, the unit's validation has failed and the fix lands in the same
unit before handoff:

- **All TPRS tests** (`tests/test_smooths/test_tprs.py` and any TPRS
  cells in `tests/test_validation_matrix.py`) — Commit B is a pure code
  move, so TPRS numerics must be byte-identical pre/post.
- **Random effects unseen-level test** (`tests/test_smooths/test_random_effects.py`,
  commit `5014785`) — already validated in v1.0, GP work touches nearby
  files (`utils.py`, `registry.py`).
- **CoefficientMap roundtrip** (CLAUDE.md §Common Pitfalls #3) —
  `predict(model, original_data) == model.fitted_values` for every fit.
- **Step-halving** (CLAUDE.md §Common Pitfalls #5) — every PIRLS fit
  with Binomial/Gamma family must converge.
- **Cholesky stability tests** (`tests/test_cholesky_stability.py`, issue
  #6 regression).
- **R bridge version check** — `RBridge.check_versions()` continues to
  pass for R 4.5.2 + mgcv 1.9-3.

If any of these fail during a commit, **stop and surface the failure**
before continuing. Do not work around — fix the root cause.

---

## Success Metrics

Track after every commit:

1. **Collected-test count** (`pytest --collect-only`) — should grow
   monotonically as GP tests are added (Commits C–H). Should not shrink
   except in cleanup-related units.
2. **Coverage percentage** (must stay ≥80% per CLAUDE.md §Testing
   Rules) — should not drop below 80% in `jaxgam/` overall, and any new
   files (`jaxgam/smooths/gaussian_process.py`) must reach ≥80% by the
   time Commit H lands (the GP class file is fully written by D; F adds
   bridge code; G and H add tests that exercise the remaining paths).
3. **`make test-cov` wall-clock** — sanity check that GP cells don't
   blow up the validation-matrix runtime.

**Baseline (captured in Commit A):** record current collected count,
source-level count, coverage %, wall-clock. Reference these in every
subsequent handoff.

---

## Commit A — Pre-flight: Baseline Capture (**DONE**)

**Goal:** Record current state so each subsequent commit can show
measurable progress and gate against regressions.

### What to do

Run on a clean checkout of the working branch (before any GP code):

```sh
uv run pytest --collect-only -q tests | tail -3
grep -rc "def test_" tests | grep -v ":0$" | awk -F: '{s+=$2} END {print s}'
time make test-cov 2>&1 | tail -25
```

Record four numbers in `docs/gaussian_process/BASELINE.md`:

- Collected test count
- Source-level test count
- Coverage %
- `make test-cov` wall-clock

Also snapshot the current TPRS test count specifically:

```sh
uv run pytest --collect-only -q tests/test_smooths/test_tprs.py | tail -1
```

This number must not change in Commit B (the refactor).

### Files touched

- Add: `docs/gaussian_process/BASELINE.md`

### Validation

- `make test-cov` runs cleanly (no GP-related failures since GP doesn't
  exist yet).

### Exit criteria

- Baseline file committed with four metrics + TPRS test count.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit B.

---

## Commit B — Extract Shared Kriging Helpers to `utils.py` (**DONE**)

**Goal:** Move four reusable pieces from `tprs.py` to `utils.py` so GP
can consume them without cross-smooth imports. **Mechanical refactor —
zero numerical change to TPRS.**

**Design reference:** §1.4, §5.4 of the design doc.

### What to do

1. **Add to `jaxgam/smooths/utils.py`** (after the existing
   `interaction_matrix` definition):

   - `_slanczos_jit` and `_slanczos` (verbatim from `tprs.py:278-442`)
   - `_compute_distance_matrix` (verbatim from `tprs.py:232-252`)
   - `_get_unique_rows` (verbatim from `tprs.py:255-275`)
   - **New** `_subsample_knots(Xu, max_knots, seed=1) -> np.ndarray`
     extracted from TPRS's inline block at `tprs.py:627-639`. Signature
     and body specified in design §5.4.1.

2. **Delete from `jaxgam/smooths/tprs.py`**:
   - The function bodies of `_slanczos_jit`, `_slanczos`,
     `_compute_distance_matrix`, `_get_unique_rows`.
   - The inline subsample block at lines 627-639; replace with:
     ```python
     if n_unique > max_knots:
         Xu = _subsample_knots(Xu, max_knots, seed=1)
         n_unique = max_knots
         inverse = _nearest_knot_indices(X_centered, Xu)
     ```
     Note: `seed=1` is the TPRS-specific value (preserves bit-exact
     pre-refactor behavior).

3. **Add imports to `jaxgam/smooths/tprs.py`** at the top of the file:
   ```python
   from jaxgam.smooths.utils import (
       _compute_distance_matrix,
       _get_unique_rows,
       _slanczos,
       _subsample_knots,
   )
   ```

4. **Confirm no import cycle**: `utils.py` must not import from any
   smooth module (it should already be a leaf — verify by reading
   `utils.py` imports).

### Files touched

- Modify: `jaxgam/smooths/utils.py` (add four symbols)
- Modify: `jaxgam/smooths/tprs.py` (delete four function bodies, replace
  inline block, add imports)

### Validation

- `make test-cov` passes. **No TPRS test should change numerical
  output** — this is the merge gate.
- TPRS test count matches the snapshot from Commit A's baseline.
- Coverage of `jaxgam/smooths/tprs.py` does not drop (the lines moved
  out are now covered by tests reaching them via `utils.py`).
- Coverage of `jaxgam/smooths/utils.py` increases by the lines added.

### Exit criteria

- All TPRS tests pass with identical output.
- Coverage ≥ 80% maintained.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit C.

---

## Commit C — GP Kernel Module: Classes + Registry (**DONE**)

**Goal:** Implement the GP correlation-kernel surface in a dedicated
`gp_kernels.py` module: `GPKernel` ABC, five concrete kernel classes,
and `gp_kernel_registry` keyed by canonical name. **Nothing else** —
no config dataclass, no parser, no evaluation/null-space helpers.
`GaussianProcessSmooth.__init__` (Commit D) owns resolution of
`spec.extra_args` and the `_gp_E` / `_gp_T` methods.

**Design reference:** §1.6, §3.3, §5.3, §3.5, §6.3 of the design doc.

### What to do

1. **Create `jaxgam/smooths/gp_kernels.py`** with (in this order):
   - Module docstring (Phase 1, NumPy only; design refs §1.6, §5.3).
   - `class GPKernel(ABC)` with `evaluate(e, *, power)` (abstract) and
     `validate(power)` (no-op default).
   - Five concrete kernels: `SphericalKernel`,
     `PowerExponentialKernel` (overrides `validate`),
     `Matern32Kernel`, `Matern52Kernel`, `Matern72Kernel`. Closed-form
     bodies per design §3.3 / §5.3.
   - `gp_kernel_registry: Registry[GPKernel] = Registry({...},
     name="GP kernel")` — five entries (one per kernel), keyed by the
     canonical names from design §1.6. **No aliases.** The registry's
     keys *are* the canonical spellings; case-insensitivity is the only
     normalization (provided by `Registry` itself).

   The module is **purely Phase 1** (NumPy only — do not import JAX)
   and has no dependency on the smooth class. The smooth-class wiring
   (including the kernel/rho/power/stationary resolution from
   `spec.extra_args`, the `m=` rejection, and the `_gp_E` / `_gp_T`
   methods) lives in Commit D.

2. **Create `jaxgam/smooths/gaussian_process.py`** as an empty stub
   plus module docstring referencing §5 of the design doc. The
   `GaussianProcessSmooth` class lands in Commit D — this file exists
   in Commit C only so import paths line up; if you prefer not to
   land an empty file, defer creation entirely to Commit D.

3. **Create `tests/test_smooths/test_gp_kernels.py`** following the
   consolidation discipline from `docs/clean_unit_tests/`. The kernel
   module has no smooth-instance dependencies, so tests exercise the
   kernel classes and registry directly:

   ```python
   class TestKernelMath:
       @pytest.mark.parametrize("kernel_cls,power", [
           (SphericalKernel,         1.0),
           (PowerExponentialKernel,  1.0),
           (PowerExponentialKernel,  2.0),
           (Matern32Kernel,          1.0),
           (Matern52Kernel,          1.0),
           (Matern72Kernel,          1.0),
       ])
       def test_kernel_matches_closed_form(self, kernel_cls, power):
           """STRICT closed-form match for one kernel/power per case."""
           kernel = kernel_cls()
           e = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
           result = kernel.evaluate(e, power=power)
           ...  # collector + closed-form comparison

   class TestPowerValidation:
       def test_power_validation(self):
           """PowerExponentialKernel.validate bounds; matern ignores power."""
           ...

   class TestRegistry:
       def test_registry_contents(self):
           """Five canonical-name entries, each resolving to the right class."""
           ...
   ```

   `__init__`-time resolution (defaults / `m=` rejection / invalid rho
   / unknown kernel) and `_gp_E` / `_gp_T` behavior are exercised in
   `test_gaussian_process.py` (Commit D) — they require a smooth
   instance, which this commit does not produce.

### Files touched

- Add: `jaxgam/smooths/gp_kernels.py` (kernels + registry only).
- Add: `jaxgam/smooths/gaussian_process.py` (empty stub — class lands
  in Commit D; or defer creation to Commit D entirely).
- Add: `tests/test_smooths/test_gp_kernels.py` (kernel-class + registry
  tests).

### Validation

- `make test-cov` passes.
- New tests collected count up by **~8** (one parametrize axis with 6
  cases, plus the power-validation and registry-contents methods).
- Coverage of `jaxgam/smooths/gp_kernels.py` ≥ 80% (kernel classes and
  registry exercised by the new tests).

### Exit criteria

- All 5 kernels match closed-form at STRICT (covered via the
  `kernel_cls` parametrize axis on `test_kernel_matches_closed_form`).
- `PowerExponentialKernel.validate` rejects `power ≤ 0` and `power > 2`;
  other kernels ignore `power`.
- `gp_kernel_registry` has exactly the five canonical-name entries and
  raises `KeyError` on unknown names.
- Test count footprint ≤14 (consolidation discipline holds).
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit D.

---

## Commit D — `GaussianProcessSmooth` Class + Structural Tests

**Goal:** Wire the `Smooth` subclass around the kernel evaluator from
Commit C. Implement `setup`, `build_design_matrix`, `build_penalty_matrices`,
`predict_matrix` per design §5.1-§5.10. **Also implement the always-on
indefinite-eigenvalue clip per design §8.3** — current fitting code
(`jax_utils.py:242,274`, `fitting/data.py:434,545`) cannot tolerate
indefinite penalties, so we replace negative truncated eigenvalues with
`|λ|` at setup time and emit a warning. This subsumes what used to be
Commit H.

**Design reference:** §5, §8.3 of the design doc.

### What to do

1. **Add `GaussianProcessSmooth` class** to
   `jaxgam/smooths/gaussian_process.py`. Imports at the top of the file:

   ```python
   from jaxgam.smooths.base import Smooth
   from jaxgam.smooths.utils import (
       _compute_distance_matrix,
       _slanczos,
       _get_unique_rows,
       _subsample_knots,
   )
   from jaxgam.smooths.gp_kernels import GPKernel, gp_kernel_registry
   from jaxgam.formula.terms import SmoothSpec
   from jaxgam.penalties.penalty import Penalty
   ```

   Implement per design §5.1 (`__init__`), §5.2 (`setup`), §5.3
   (`_gp_E` method), §5.5 (knot harvesting), §5.6 (`_gp_T` method +
   `_build_design`), §5.7 (`build_design_matrix` / `predict_matrix`),
   §5.8 (`build_penalty_matrices`), §5.10 (default `bs.dim`). No
   `_decide_stationary`, no `GPConfig`, no `parse_gp_config` — kwarg
   resolution is inline in `__init__` per design §5.9.

   - **`__init__(self, spec)`** (design §5.1):
     - Call `super().__init__(spec)`.
     - Raise `ValueError` if `"m" in spec.extra_args` (mgcv-compat
       rejection, the only GP-specific guard in `__init__`).
     - `self._kernel = gp_kernel_registry.get_instance(
       spec.extra_args.get("kernel", "matern_3_2"))` — raises
       `KeyError` on unknown name via the registry.
     - `self._rho = spec.extra_args.get("rho")`; raise if not None and
       `<= 0.0`.
     - `self._power = spec.extra_args.get("power", 1.0)`.
     - `self._stationary = spec.extra_args.get("stationary", False)`.
     - `self._kernel.validate(self._power)` — surfaces
       `PowerExponentialKernel` bounds errors at construction time.
   - **`_gp_E(self, x, xk, *, resolved_rho=None)`** (design §5.3): owns
     the rho-resolution policy (`resolved_rho` > `self._rho` > auto via
     `distances.max()`), guards against `rho <= 0`, and returns
     `(E, rho_used)` from `self._kernel.evaluate(distances / rho,
     power=self._power)`.
   - **`_gp_T(self, x_centered)`** (design §5.6): returns
     `np.ones((n, 1))` if `self._stationary`, else
     `np.column_stack([np.ones(n), x_centered])`.
   - **`setup()`** (design §5.2): data-only work — knot harvesting,
     centering, `E, rho = self._gp_E(knt_c, knt_c)`, eigendecomp,
     indefinite clip, design + penalty matrices.
   - `null_space_dim = 1` if `self._stationary`, else `d + 1`.
   - `rank = bs_dim - null_space_dim`.
   - `n_coefs = bs_dim` (pre-centering; constraint absorption reduces
     by 1 downstream).
   - Standard `_smoothcon_normalize` from `Smooth` base.
   - **Do not** override `side_constrain` (default `True` — GP
     participates in gam.side).
   - **Do not** override `_random` or `_has_centering_constraint` (GP
     uses standard centering).
   - **Eigendecomp call**: `_slanczos(E, k, tol=np.finfo(float).eps ** 0.5)`
     — explicit `tol` matching mgcv's R-side `slanczos(E, k, -1)` default.
     Do **not** rely on the function default (`eps**0.7`, TPRS-shaped).
     See design §5.4.1.
   - **Indefinite clip (design §8.3, always on)**: after `_slanczos`
     returns `(eigvals, eigvecs)`, if `(eigvals < 0).any()` emit a
     `warnings.warn(...)` naming the smooth's variables and the count
     of clipped entries, then set `eigvals = np.abs(eigvals)` before
     building `D`. **No fallback path, no flag** — this runs every
     time. Document in the class docstring.
   - `_build_design(x_c)` calls
     `self._gp_E(x_c, self._knt, resolved_rho=self._resolved_rho)`
     (training-time rho is frozen) and `self._gp_T(x_c)`.

2. **Add structural tests** to
   `tests/test_smooths/test_gaussian_process.py`. **Consolidate via
   `_AssertCollector`** — these all share the same `setup()` call, so
   they collapse to **2-3 collected tests**, not 15. `stationary` is
   passed in via the new `extra_args["stationary"]=True/False` kwarg:

   ```python
   class TestSetupInvariants:
       @pytest.mark.parametrize("stationary", [False, True])
       def test_setup_state(self, gp_1d_data, stationary):
           """One R-free fixture; collector accumulates all invariants."""
           smooth = _make_gp_smooth(gp_1d_data, stationary=stationary)
           # _make_gp_smooth builds a SmoothSpec with
           # extra_args={"stationary": stationary} — NO m= anywhere.
           smooth.setup(gp_1d_data)
           collector = _AssertCollector()
           collector.check("null_space_dim", lambda: ...)
           collector.check("rank == bs_dim - null_space_dim", lambda: ...)
           collector.check("side_constrain == True", lambda: ...)
           collector.check("no _random / no _has_centering_constraint", lambda: ...)
           collector.check("_shift == colMeans", lambda: ...)
           collector.check("self._kernel is a GPKernel", lambda: ...)
           collector.check("_resolved_rho > 0", lambda: ...)
           collector.check("penalty diagonal", lambda: ...)
           collector.check("predict_matrix == build_design_matrix", lambda: ...)
           collector.raise_if_any(f"setup invariants (stationary={stationary})")

       def test_dimension_defaults(self):
           """One test for the {1D: 12, 2D: 33, 3D: 104, d>3: raise} table."""
           collector = _AssertCollector()
           collector.check("1D default bs_dim == 12", lambda: ...)
           collector.check("2D default bs_dim == 33", lambda: ...)
           collector.check("3D default bs_dim == 104", lambda: ...)
           collector.check("d > 3 raises", lambda: ...)
           collector.raise_if_any("dimension defaults")

       def test_init_rejects_m_argument(self):
           """`m=` is rejected at construction time — before any data
           touches setup()."""
           spec = SmoothSpec(variables=["x"], bs="gp",
                             extra_args={"m": [3, 0.5]})
           with pytest.raises(ValueError, match="kernel="):
               GaussianProcessSmooth(spec)
   ```

3. **Knot-selection tests** — one consolidated method:

   ```python
   def test_knot_subsampling(self, large_gp_data):
       """All knot-selection invariants via one collector."""
       collector = _AssertCollector()
       collector.check("n <= max_knots → all unique rows", lambda: ...)
       collector.check("n > max_knots → exactly max_knots", lambda: ...)
       collector.check("same seed → identical knots", lambda: ...)
       collector.check("global RNG untouched after setup", lambda: ...)
       collector.raise_if_any("knot subsampling")
   ```

4. **Indefinite-clip regression test** (one method, ~5 lines):

   ```python
   def test_indefinite_eigenvalues_are_clipped(self):
       """Spherical kernel on d=3 with tight rho can yield negative
       truncated eigenvalues. Verify we warn and clip rather than
       passing indefinite S into Penalty / fitting (which can't
       tolerate it — see design §8.3)."""
       # Construct a small GP smooth in a regime where _slanczos
       # returns at least one negative eigenvalue (e.g. spherical
       # m=[1, 0.3] on d=3 random data).
       with pytest.warns(UserWarning, match="negative"):
           smooth = _build_gp_indefinite_case()
       # Penalty diagonal must be non-negative after the clip.
       S = smooth.build_penalty_matrices()[0].S
       assert (np.diag(S) >= 0).all()
   ```

5. **Conftest fixtures** (`tests/conftest.py`): add `gp_1d_data`,
   `gp_2d_data`, `gp_explicit_knots_data` per design §12.4. Also add a
   `large_gp_data` fixture (n > max_knots) for the subsampling test.

### Files touched

- Modify: `jaxgam/smooths/gaussian_process.py` (add the class)
- Modify: `tests/test_smooths/test_gaussian_process.py` (add structural
  tests)
- Modify: `tests/conftest.py` (add fixtures)

### Validation

- `make test-cov` passes.
- New tests collected count up by **~6-8** (consolidated via
  `_AssertCollector` and parametrize, not enumerated as ~15 separate
  methods; +1 for the indefinite-clip regression test; +1 for the
  `m=` integration test). If the count exceeds ~10, you missed
  consolidation.
- Coverage of `jaxgam/smooths/gaussian_process.py` ≥ 80%.

### Exit criteria

- `GaussianProcessSmooth` instantiates, sets up, builds X / S / predict
  matrix without errors for fixture data.
- All structural assertions hold (verified through collectors —
  failures still name the broken invariant).
- Indefinite-clip path is exercised at least once and produces a
  non-negative diagonal `S` plus the documented warning.
- Passing `extra_args={"m": [...]}` raises `ValueError` at `setup()`
  time with the documented message (integration-level pin on the
  kernel-module rejection in Commit C).
- Test count footprint ≤10 (consolidation discipline holds).
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit E.

---

## Commit E — Registry Wire-Up + Univariate-Margin Invariant + Parser Tests

**Goal:** Register `"gp"` in the smooth registry, prove that
`GaussianProcessSmooth` works correctly as the **per-variable margin**
that `TensorProductSmooth._create_marginals()` (`tensor.py:136-148`)
builds, and verify the parser handles the GP-specific arguments
(`kernel=`, `rho=`, `power=`, `stationary=`, `xt=`, `k`) **using Python
literals only** — R-style `c(...)` / `list(...)` is not supported by
`ast.literal_eval` in `parser.py:282-310`, so the documented surface
is Python syntax. R-syntax shimming would be a parser-wide change
orthogonal to GP.

**No tensor-side code change.** Registering `"gp"` is *sufficient* to
enable both direct GP (`s(x, z, bs="gp")`) and tensor-margin GP
(`te(x1, x2, bs="gp")` / `ti(x1, x2, bs="gp")`) because the existing
tensor wrappers already dispatch through `get_smooth_class(self.spec.bs)`
at `tensor.py:146`. Do not add a `bs == "gp"` guard. Do not edit
`tensor.py`. Commit G's tensor smooth-construct R-parity test is the
end-to-end gate; Commit H's `gp_te` / `gp_ti` validation-matrix cells
are the fitting gate.

**Design reference:** §6, §11.2, §12.2 of the design doc.

### What to do

1. **Modify `jaxgam/smooths/registry.py`**: add `"gp":
   GaussianProcessSmooth` to `smooth_registry`. Import at the top:
   ```python
   from jaxgam.smooths.gaussian_process import GaussianProcessSmooth
   ```

2. **Add a univariate-margin invariant test** to
   `tests/test_smooths/test_gaussian_process.py`. This is the test
   that proves the tensor-margin pathway is safe to land in Commit H
   *without* any tensor-side code change. Build a single-variable
   `SmoothSpec` exactly as `TensorProductSmooth._create_marginals()`
   would, run `setup()`, and assert what the tensor wrapper relies on:

   ```python
   def test_works_as_tensor_margin(self, gp_1d_data):
       """The contract TensorProductSmooth._create_marginals() relies
       on (tensor.py:137-152). One test, collector — proves tensor GP
       is enabled by registration alone."""
       spec = SmoothSpec(variables=["x"], bs="gp", k=5,
                         smooth_type="s")  # exactly what tensor builds
       margin = GaussianProcessSmooth(spec)
       margin.setup(gp_1d_data)
       X = margin.build_design_matrix(gp_1d_data)
       pen = margin.build_penalty_matrices()
       collector = _AssertCollector()
       collector.check("setup populated _s_scale",
                       lambda: margin._s_scale > 0)
       collector.check("design matrix has bs_dim cols",
                       lambda: X.shape[1] == margin.n_coefs)
       collector.check("one penalty per margin",
                       lambda: len(pen) == 1)
       collector.check("penalty diagonal",
                       lambda: np.allclose(pen[0].S,
                                            np.diag(np.diag(pen[0].S))))
       collector.check("_noterp is False (so tensor SVD reparam runs)",
                       lambda: margin._noterp is False)
       collector.check("predict_matrix roundtrip",
                       lambda: np.allclose(
                           margin.predict_matrix(gp_1d_data), X))
       collector.raise_if_any("univariate-margin invariants")
   ```

3. **Add parsing tests** to `tests/test_formula/test_parser.py`
   (or wherever `bs=` parsing lives — locate the existing TPRS / RE
   parser tests and add GP analogs nearby). **Use the Python-native
   GP API** (`kernel=`/`rho=`/`power=`/`stationary=`); `m=` is *not*
   the documented surface and parser tests must not advertise it.
   Consolidate the positive cases into a single
   `test_gp_kwargs_parse` method that walks several formulas through
   one `_AssertCollector`:

   - `s(x, bs="gp")` → `SmoothSpec(variables=["x"], bs="gp")`.
   - `s(x, z, bs="gp", k=50)` → captures `k=50` and two variables
     (direct multivariate GP).
   - `te(x1, x2, bs="gp", k=5)` → `SmoothSpec(variables=["x1","x2"],
     bs="gp", k=5, smooth_type="te")` (tensor GP, scalar `k` shared
     across margins per current tensor convention — see design §1.3).
   - `ti(x1, x2, bs="gp", k=5)` → analogous with `smooth_type="ti"`.
   - `s(x, bs="gp", kernel="matern_3_2")` →
     `extra_args["kernel"] == "matern_3_2"`.
   - `s(x, bs="gp", kernel="power_exponential", rho=0.5, power=2.0)` →
     `extra_args` carries all three kwargs as the expected Python types
     (str / float / float).
   - `s(x, bs="gp", stationary=True)` →
     `extra_args["stationary"] is True` (parser preserves Python bool).
   - `s(x, bs="gp", xt={"max_knots": 500, "seed": 42})` →
     `extra_args["xt"] == {"max_knots": 500, "seed": 42}`.

   **Negative parser test (keep)**: `s(x, bs="gp", kernel=c("matern_3_2"))`
   → `ValueError` from `_eval_kwarg_value` ("Cannot evaluate argument
   'kernel'…"). One negative test that pins the current parser
   behavior so a future R-syntax patch breaks this test loudly.

   **Pinning `m=` parser behavior** (not a rejection at parser level):
   `s(x, bs="gp", m=[3, 0.5])` should parse fine — `extra_args["m"]`
   is set, no error at parse time. Add **one** consolidated positive
   case for this in the collector so a future change that adds parser-
   level GP knowledge (and starts rejecting `m=` early) breaks it
   loudly and forces the reviewer to think about layering. The setup-
   time rejection of `m=` is tested in Commit C (`test_gp_kernels.py`)
   and Commit D (integration smoke test) — **not** here.

4. **Do NOT add an end-to-end fit test here.** This was previously
   listed in the plan and removed: per CLAUDE.md §Testing Rules and the
   Phase 2 ownership sweep in `docs/clean_unit_tests/`, end-to-end fit
   and `predict()` parity for GP live exclusively in
   `tests/test_validation_matrix.py` (Commit H — was G before
   renumbering). Adding it here duplicates the matrix and violates the
   canonical-owner rule.

   The "does GP fit at all?" gate is implicit: Commit H's matrix
   integration *must* run a real fit per GP cell, so if registry
   wire-up is broken those cells fail.

### Files touched

- Modify: `jaxgam/smooths/registry.py`
- Modify: `tests/test_smooths/test_gaussian_process.py` (add the
  univariate-margin invariant test)
- Modify: `tests/test_formula/test_parser.py` (add GP parsing tests
  consolidated via `_AssertCollector`, including `te` / `ti` parsing
  for `bs="gp"`)
- **Do NOT modify** `jaxgam/smooths/tensor.py`. GP tensor margins are
  supported via the existing dispatch and an earlier patch was wrong
  to add a guard. If the working tree contains such a guard from a
  prior attempt, remove it as part of this commit.

### Validation

- `make test-cov` passes.
- New tests collected count up by **~3-5** (one parser collector +
  one univariate-margin invariant test + tensor-formula parser cases
  collapse into the parser collector).
- Parser correctly extracts `kernel`, `rho`, `power`, `stationary`,
  `xt`, `k`, and (parses but does not interpret) `m` for both `s()`
  and `te()` / `ti()`; R-style `c(...)` raises `ValueError` at parser
  level.
- `te(x, z, bs="gp")` **constructs** (no longer raises) — full fit
  parity is exercised in Commit H.
- `tensor.py` diff is empty.

### Exit criteria

- All GP-specific parser variants resolve to the right `SmoothSpec`,
  including `te` / `ti` wrappers and the Python-native kwargs
  (`kernel=`/`rho=`/`power=`/`stationary=`); R-style `c(...)` is
  pinned as a negative test; `m=` parses cleanly (rejection is at
  setup, not parse).
- Univariate-margin invariant test passes — `GaussianProcessSmooth`
  obeys the contract `TensorProductSmooth._create_marginals()` relies
  on.
- No tensor-side code change.
- Test count footprint ≤5.
- Note: end-to-end tensor GP fit verification is deferred to Commit
  H's `gp_te` / `gp_ti` validation-matrix cells (canonical owner per
  CLAUDE.md §Testing Rules).
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit F (RBridge GP enhancements).

---

## Commit F — RBridge GP Enhancements + `gp_config_to_mgcv_m` Helper

**Goal:** Extend `RBridge.smooth_construct()` so Commit G's R-parity
tests can (a) pass explicit knots into mgcv's `smoothCon()`, (b) read
back the GP-specific fields `knt` and `gp.defn`, and (c) translate a
Python-side GP `SmoothSpec` into mgcv's `m=c(...)` numeric vector via
a new module-level helper `gp_config_to_mgcv_m`. This commit touches
**only `tests/r_bridge.py`** and a small direct unit test — it adds
no GP-side production code and no GP-side tests yet.

**Design reference:** §2.7, §11.2 of the design doc.

### Why this is its own commit

Without these changes the bridge cannot exercise GP smooth-construct
parity at all: current `RBridge.smooth_construct(smooth_expr, data,
absorb_cons)` has no `knots=` parameter (`tests/r_bridge.py:564`),
and it extracts only TPRS-shaped fields (`Xu`, `UZ`, `shift`) — for
GP, `Xu` is empty and `knt` / `gp.defn` are silently dropped. The
production `GAM.fit()` API rejects user knots (`jaxgam/api.py:225`)
and `ModelSetup` does not thread them through
(`jaxgam/formula/design.py:657`), so the bridge knot-injection path
is **test-only infrastructure** that does not affect production
behavior.

Keeping this separate from Commit G (which consumes it) makes the
diff readable and lets us add a focused bridge-only unit test
without entangling it with the GP-side `TestGPVsR` class.

### What to do

1. **Modify `RBridge.smooth_construct()` signature** at
   `tests/r_bridge.py:564`:
   ```python
   def smooth_construct(
       self,
       smooth_expr: str,
       data: pd.DataFrame,
       absorb_cons: bool = False,
       knots: dict[str, np.ndarray] | None = None,
   ) -> dict[str, Any]:
   ```
   Thread `knots` through to `_smooth_construct_rpy2` only — see
   step 2 for why the subprocess path is out of scope.

2. **rpy2 path only** (`_smooth_construct_rpy2`, around
   `tests/r_bridge.py:591`). Tests rely on the rpy2 path; the
   subprocess fallback (`_smooth_construct_subprocess`) is legacy
   code that GP does not exercise. Do not extend it.
   - If `knots is not None`, marshal it into an R named list and
     bind to `ro.globalenv["knots_input"]` alongside the existing
     `dat_input`; render the embedded R code to call
     `smoothCon(..., knots = knots_input)`. Remember to
     `del ro.globalenv["knots_input"]` in the `finally` block (same
     pattern as the existing `dat_input` cleanup at
     `r_bridge.py:621`).
   - Extend the returned R `list(...)` with three fields:
     ```r
     knt     = if (!is.null(sm$knt)) sm$knt else matrix(0, 0, 0),
     gp_defn = if (!is.null(sm$gp.defn)) sm$gp.defn else numeric(0),
     E       = if (!is.null(sm$knt))
                 mgcv:::gpE(sm$knt, sm$knt, sm$gp.defn)
               else matrix(0, 0, 0)
     ```
     `E` is the knot–knot kernel matrix re-computed from the stored
     centered knots and `gp.defn` — the un-truncated input to
     `slanczos`. Commit G's R-parity test compares this against
     `py_smooth._E_knot`.
   - In the Python return dict, surface `"knt": np.array(result.rx2("knt"), dtype=np.float64)`,
     `"gp_defn": np.array(result.rx2("gp_defn"), dtype=np.float64)`,
     and `"E": np.array(result.rx2("E"), dtype=np.float64)`
     (matching the dtype convention used by the existing X / S
     extraction at `r_bridge.py:623-634`).

3. **Add one direct unit test** to `tests/test_r_bridge.py` (or
   wherever the bridge is currently tested):
   ```python
   @pytest.mark.skipif(not r_available(), reason="R+mgcv not available")
   def test_smooth_construct_gp_extracts_knt_and_defn():
       """Bridge round-trips knt and gp.defn for a tiny GP fit."""
       rb = RBridge()
       data = pd.DataFrame({"x": np.linspace(0, 1, 30)})
       knots = {"x": np.linspace(0.05, 0.95, 8)}
       result = rb.smooth_construct(
           "s(x, bs='gp', k=10)", data, absorb_cons=False, knots=knots
       )
       collector = _AssertCollector()
       collector.check("knt shape", lambda: result["knt"].shape == (8, 1))
       collector.check("gp_defn length 3",
                       lambda: result["gp_defn"].shape == (3,))
       collector.check("gp_defn[0] == 3 (default Matérn 3/2)",
                       lambda: int(round(result["gp_defn"][0])) == 3)
       collector.check("E shape (nk, nk)",
                       lambda: result["E"].shape == (8, 8))
       collector.check("E symmetric",
                       lambda: np.allclose(result["E"], result["E"].T))
       collector.check("S diagonal",
                       lambda: np.allclose(result["S"][0],
                                            np.diag(np.diag(result["S"][0]))))
       collector.raise_if_any("GP bridge round-trip")
   ```

4. **Add the `gp_config_to_mgcv_m` helper** at module level in
   `tests/r_bridge.py` (or a sibling test-helper module — keep it in
   `r_bridge.py` so test files can `from tests.r_bridge import
   gp_config_to_mgcv_m`):

   ```python
   from jaxgam.formula.terms import SmoothSpec

   _KERNEL_TO_MGCV_TYPE = {
       "spherical":         1,
       "power_exponential": 2,
       "matern_3_2":        3,
       "matern_5_2":        4,
       "matern_7_2":        5,
   }


   def gp_config_to_mgcv_m(
       spec: SmoothSpec, rho: float | None = None
   ) -> list[float]:
       """Translate a GP ``SmoothSpec`` to mgcv's signed `m` vector.

       Reads the same ``spec.extra_args`` that ``GaussianProcessSmooth.__init__``
       consumes; used by Commit G's smooth-construct R-parity tests and
       Commit H's validation-matrix R formulas so the two sides cannot
       drift.
       """
       kernel = spec.extra_args.get("kernel", "matern_3_2")
       stationary = spec.extra_args.get("stationary", False)
       power = spec.extra_args.get("power", 1.0)
       spec_rho = spec.extra_args.get("rho")

       type_id = _KERNEL_TO_MGCV_TYPE[kernel]
       if stationary:
           type_id = -type_id

       out: list[float] = [float(type_id)]
       resolved_rho = rho if rho is not None else spec_rho
       if resolved_rho is not None:
           out.append(float(resolved_rho))

       if kernel == "power_exponential":
           # Pad rho if absent so power lands at m[2].
           if len(out) == 1:
               out.append(-1.0)
           out.append(float(power))

       return out
   ```

6. **Add a direct unit test for `gp_config_to_mgcv_m`** to
   `tests/test_r_bridge.py` (consolidated via `_AssertCollector` —
   no R needed):
   ```python
   def _gp_spec(**extra_args) -> SmoothSpec:
       """Helper: minimal GP SmoothSpec carrying just extra_args."""
       return SmoothSpec(
           variables=["x"], bs="gp", k=-1, by=None,
           smooth_type="s", extra_args=extra_args,
       )


   def test_gp_config_to_mgcv_m_table():
       """Round-trip every row of design §6.4's mgcv ↔ JaxGAM table."""
       collector = _AssertCollector()
       collector.check(
           "spherical+rho",
           lambda: gp_config_to_mgcv_m(
               _gp_spec(kernel="spherical", rho=0.5)
           ) == [1.0, 0.5]
       )
       collector.check(
           "stationary spherical",
           lambda: gp_config_to_mgcv_m(
               _gp_spec(kernel="spherical", rho=0.5, stationary=True)
           ) == [-1.0, 0.5]
       )
       collector.check(
           "squared-exp",
           lambda: gp_config_to_mgcv_m(
               _gp_spec(kernel="power_exponential", rho=0.5, power=2.0)
           ) == [2.0, 0.5, 2.0]
       )
       collector.check(
           "matern_5_2 default rho omitted",
           lambda: gp_config_to_mgcv_m(
               _gp_spec(kernel="matern_5_2")
           ) == [4.0]
       )
       collector.check(
           "rho override beats spec.rho",
           lambda: gp_config_to_mgcv_m(
               _gp_spec(kernel="matern_3_2", rho=0.5),
               rho=0.7,
           ) == [3.0, 0.7]
       )
       collector.raise_if_any("gp_config_to_mgcv_m mapping")
   ```

5. **Backwards-compatibility**: `knots=None` (the default) must
   reproduce the previous behavior bit-for-bit. Add no regression
   test for non-GP smooths — existing bridge tests cover that.

### Files touched

- Modify: `tests/r_bridge.py` (signature + rpy2 path only +
  return-dict additions + `gp_config_to_mgcv_m` helper; do not touch
  the subprocess fallback)
- Modify: `tests/test_r_bridge.py` (two new tests: GP bridge
  round-trip + `gp_config_to_mgcv_m` table-driven, both consolidated
  via `_AssertCollector`)

### Validation

- `make test-cov` passes. The GP bridge round-trip test skips when R
  is unavailable; the `gp_config_to_mgcv_m` test is R-free and always
  runs.
- Existing bridge tests (TPRS, cubic, RE smooth_construct cases)
  must still pass — `knots=None` default path is untouched.
- New tests collected count up by **+2** (bridge round-trip +
  helper table).

### Exit criteria

- Bridge accepts `knots=` and returns `knt` / `gp_defn` keys.
- `gp_config_to_mgcv_m` round-trips every row of design §6.4's
  mapping table.
- One direct round-trip test passes in Docker.
- No regression in existing `r_bridge` tests.
- **Agent stops and hands off to user for commit.** Do not proceed
  to Commit G.

---

## Commit G — R Smooth-Construct Comparison Tests

**Goal:** Validate GP basis, penalty, and design matrix against mgcv's
output at STRICT tolerance (with explicit knots to control for
subsampling differences) and MODERATE (with auto knots). Was Commit F
before the RBridge insertion.

**Design reference:** §12.2 (smooth-construct comparisons) and §10.6
(knot sampling parity).

### What to do

1. **Add R-bridge smooth-construct tests** to
   `tests/test_smooths/test_gaussian_process.py`. **Apply consolidation
   + parametrize discipline** — one R fit per parametrize case, then
   `_AssertCollector` for all three quantities (E / S / X·Xᵀ). The
   parametrize axis carries GP-kwargs dicts (the same shape that lands
   in `spec.extra_args`); the R-side formula is built via the Commit-F
   `gp_config_to_mgcv_m` helper so the two sides cannot drift:

   ```python
   from jaxgam.formula.terms import SmoothSpec
   from tests.r_bridge import gp_config_to_mgcv_m


   @pytest.mark.skipif(not r_available(), reason="R+mgcv not available")
   class TestGPVsR:
       @pytest.mark.parametrize("gp_kwargs,label", [
           ({"kernel": "spherical",         "rho": 0.5},               "spherical"),
           ({"kernel": "power_exponential", "rho": 0.5, "power": 1.0}, "power_exp_k1"),
           ({"kernel": "power_exponential", "rho": 0.5, "power": 2.0}, "squared_exp"),
           ({"kernel": "matern_3_2",        "rho": 0.5},               "matern_3_2"),
           ({"kernel": "matern_5_2",        "rho": 0.5},               "matern_5_2"),
           ({"kernel": "matern_7_2",        "rho": 0.5},               "matern_7_2"),
       ])
       def test_smooth_construct_matches_r(self, gp_kwargs, label, gp_explicit_knots_data):
           """One R fit per kernel; all 3 quantities via collector.

           Uses the Commit F bridge extensions (knots= argument; knt
           / gp_defn extraction) so R and Python operate on identical
           knot sets; uses gp_config_to_mgcv_m so the R-side `m=` is
           generated from the same SmoothSpec that builds the Python
           smooth.
           """
           spec = SmoothSpec(
               variables=["x"], bs="gp", k=-1, by=None,
               smooth_type="s", extra_args=gp_kwargs,
           )
           m_args = gp_config_to_mgcv_m(spec)
           r_formula = f"s(x, bs='gp', m=c({','.join(map(str, m_args))}))"
           r_result = r_bridge.smooth_construct(
               r_formula,
               gp_explicit_knots_data["data"],
               knots=gp_explicit_knots_data["knots"],
           )
           py_smooth = _build_gp(spec, gp_explicit_knots_data)
           # py_smooth diagonal is normalized to PSD via the §8.3 clip
           # at setup (Commit D); STRICT equality on the diagonal only
           # holds when R's spectrum is also non-negative.
           r_S_diag = np.diag(r_result["S"][0])
           clipped = (r_S_diag < 0).any()

           collector = _AssertCollector()
           # Knot-knot kernel matrix is the un-truncated input to
           # _slanczos. Both sides expose it: Python via
           # `py_smooth._E_knot` (stored at setup, see design §5.1),
           # R via the new `mgcv:::gpE(...)` recomputation surfaced in
           # the Commit-F bridge extension (`r_result["E"]`).
           collector.check("E matrix STRICT",
                          lambda: assert_close(py_smooth._E_knot,
                                                r_result["E"], STRICT))
           if not clipped:
               collector.check("penalty eigenvalues STRICT",
                              lambda: assert_close(np.diag(py_smooth._S),
                                                    r_S_diag, STRICT))
           collector.check("X @ X.T STRICT (sign-invariant)",
                          lambda: assert_close(py_smooth._X @ py_smooth._X.T,
                                                r_result["X"] @ r_result["X"].T,
                                                STRICT))
           collector.raise_if_any(f"GP vs R [{label}]")

       def test_null_space_matches_r(self):
           """Both stationarity modes via one collector — null.space.dim + gpT cols."""
           collector = _AssertCollector()
           collector.check("stationary null.space.dim == 1", lambda: ...)
           collector.check("non-stationary null.space.dim == d+1", lambda: ...)
           collector.check("gpT columns match R", lambda: ...)
           collector.raise_if_any("null space vs R")
   ```

   `_build_gp(spec, …)` instantiates `GaussianProcessSmooth(spec)` and
   runs `setup(data)` — kwargs flow straight from `spec.extra_args`
   into `__init__`. No mention of `m=` on the Python side.

2. **Document the eigenvector sign caveat** in the test docstring: raw
   `X` comparison would require MODERATE; we compare `X @ X.T` (sign-
   invariant), so STRICT is achievable.

3. **Knot parity**: use `gp_explicit_knots_data` (small dataset with
   explicit knot positions) so R and Python operate on identical knot
   sets via the Commit-F `knots=` argument. Avoids the `sample()`
   algorithm divergence (design §10.6).

4. **Indefinite-clip parity gap**: the test deliberately skips the
   STRICT diagonal-eigenvalue check when R's S has any negative
   diagonal entry, because Python clips to `|λ|` per design §8.3 and
   R does not. This is the documented deviation — record it in the
   class-level docstring.

5. **Add a tensor-margin smooth-construct test** in the same file.
   This is the *construction* gate proving GP works through
   `TensorProductSmooth` / `TensorInteractionSmooth` without any
   tensor-side code change. **Compare only top-level smoothCon
   artifacts** — the existing `RBridge.smooth_construct` returns
   `sm$X`, `sm$S`, `sm$rank`, `sm$null.space.dim` for the constructed
   tensor smooth and does **not** descend into `sm$margin` for
   per-margin extraction. Adding nested-margin extraction is out of
   scope for this PR; if a future commit needs it, extend the bridge
   then. Marginal-level construction parity is exercised indirectly
   by the direct-GP test above (one univariate margin = one direct
   1-D GP) and by Commit H's `gp_te` / `gp_ti` fits.

   ```python
   @pytest.mark.skipif(not r_available(), reason="R+mgcv not available")
   class TestGPTensorMarginVsR:
       """Confirms te(..., bs='gp') and ti(..., bs='gp') construct via
       the existing tensor wrapper + registry dispatch (no tensor.py
       change). Compares the top-level tensor X @ X.T at STRICT
       (sign-invariant under SVD reparameterization). Per-margin
       parity is not extracted from the R side (see comment above)."""

       @pytest.mark.parametrize("py_formula,r_formula,wrapper", [
           ("te(x1, x2, bs='gp', k=5)",
            "te(x1, x2, bs='gp', k=c(5, 5))", "te"),
           ("ti(x1, x2, bs='gp', k=5)",
            "ti(x1, x2, bs='gp', k=c(5, 5))", "ti"),
       ])
       def test_tensor_construct_matches_r(self, py_formula, r_formula,
                                            wrapper, gp_te_2d_data):
           r_result = r_bridge.smooth_construct(r_formula, gp_te_2d_data)
           py_smooth = _build_tensor_gp(py_formula, gp_te_2d_data)
           collector = _AssertCollector()
           collector.check("X column count matches",
                          lambda: py_smooth._X.shape[1] == r_result["X"].shape[1])
           collector.check("X @ X.T STRICT (sign / SVD-reparam-invariant)",
                          lambda: assert_close(py_smooth._X @ py_smooth._X.T,
                                                r_result["X"] @ r_result["X"].T,
                                                STRICT))
           collector.check("rank matches",
                          lambda: int(py_smooth.rank) == int(r_result["rank"]))
           collector.check("null_space_dim matches",
                          lambda: int(py_smooth.null_space_dim) == int(r_result["null_space_dim"]))
           collector.raise_if_any(f"tensor GP construct [{wrapper}]")
   ```

   The validation-matrix `gp_te` / `gp_ti` cells in Commit H are the
   *fitting* gate; this commit's tensor test is the *construction*
   gate that flags wiring breakage before fits get involved.

### Files touched

- Modify: `tests/test_smooths/test_gaussian_process.py` (add R-bridge
  direct-GP test class + tensor-margin test class)
- Modify: `tests/conftest.py` (add `gp_te_2d_data` fixture; can be a
  trimmed variant of the validation-matrix `_make_gp_te_2d_data` from
  Commit H)

### Validation

- `make test-cov` passes. R-bridge tests skip cleanly if R is unavailable
  (covered by `r_available()` guard).
- New tests collected count up by **~3-9** (one parametrized
  `test_smooth_construct_matches_r` over 6 kernel cases, one
  `test_null_space_matches_r`, plus one parametrized
  `test_tensor_margin_matches_r` over 2 wrapper cases). Each
  parametrize case runs one R fit and consolidates assertions via
  `_AssertCollector`.
- In Docker (where R is present), all R-bridge tests pass.

### Exit criteria

- Direct GP: E matrix matches R at STRICT for all 6 kernel configurations.
- Direct GP: penalty eigenvalues match R at STRICT for the explicit-
  knots case **when R's spectrum is non-negative** (clipped cases
  skip this assertion with a logged note).
- Direct GP: design matrix matches R at STRICT (`X @ X.T` is sign-
  invariant).
- Tensor GP: `te(..., bs='gp')` and `ti(..., bs='gp')` construct via
  the existing `tensor.py` dispatch (no code change) and match R on
  marginal sign-invariant quantities.
- Test count footprint ≤10.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit H.

---

## Commit H — Validation Matrix Integration

**Goal:** Add GP cells to `tests/test_validation_matrix.py` so GP is
exercised against all 5 families (gaussian, binomial, poisson, gamma, nb)
through the existing parametrized matrix. Inherits all 7 R-comparison
quantities and all 8 hard-gate invariants per cell. Was Commit G before
the renumber.

**Design reference:** §12.1.

### What to do

1. **Add 5 GP smooth configs** to `SMOOTH_CONFIGS` in
   `tests/test_validation_matrix.py` — **three direct GP** (`gp`,
   `gp_2d`, `gp_mixed`) and **two tensor GP** (`gp_te`, `gp_ti`).
   Direct and tensor are mathematically distinct constructions
   (design §1.3 table); neither is redundant.

   `gp_mixed` uses `x + s(x, bs='gp')` — same variable on both sides
   so the parametric linear term collides with the GP's non-stationary
   null-space linear column. `gp_te` / `gp_ti` use the existing
   tensor wrappers and pass scalar `k=5` (JaxGAM tensor convention,
   `tensor.py:140`); the R formula spells out `k=c(5, 5)` for
   per-margin parity.

   ```python
   # Direct GP
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
   # Tensor GP (via the existing tensor wrappers + registry dispatch)
   "gp_te": SmoothConfig(
       py_formula="y ~ te(x1, x2, bs='gp', k=5)",
       r_formula="y ~ te(x1, x2, bs='gp', k=c(5, 5))",
       data_type="gp_te_2d",
   ),
   "gp_ti": SmoothConfig(
       py_formula=("y ~ s(x1, bs='gp', k=5) + s(x2, bs='gp', k=5) "
                   "+ ti(x1, x2, bs='gp', k=5)"),
       r_formula=("y ~ s(x1, bs='gp', k=5) + s(x2, bs='gp', k=5) "
                  "+ ti(x1, x2, bs='gp', k=c(5, 5))"),
       data_type="gp_te_2d",
   ),
   ```

   `gp_ti` is optional but recommended — it exercises
   `_svd_reparameterize` (`tensor.py:52-123`) on GP margins, which
   is otherwise only covered indirectly through `te`. If runtime is
   a concern, drop `gp_ti` and revisit in a follow-up.

   **All five configs above use the default kernel** (Matérn 3/2,
   auto ρ, non-stationary) — no `kernel=`/`rho=`/`power=`/
   `stationary=` kwargs needed, and the R-side formula is identical
   to the Python-side formula verbatim. If a future follow-up adds
   kernel-specific cells (e.g. `gp_spherical`, `gp_squared_exp`,
   `gp_stationary`), the Python `py_formula` uses the new kwargs and
   the `r_formula` is built once at module load via
   `gp_config_to_mgcv_m(config)` (Commit F). Do **not** hand-write
   the R-side `m=c(...)` literal — let the helper do it so a future
   refactor of the mgcv-side encoding only has to touch one place.

2. **Add 4 data generators** per design §12.1.2:
   - `_make_gp_1d_data(family_name)`
   - `_make_gp_2d_data(family_name)` — uses `x` / `z`
   - `_make_gp_1d_par_data(family_name)` — generates a single `x`
     column (no separate `x_par`), with `eta = 0.7*x + sin(3πx)*0.6`
     so the linear trend is real and `gam.side()` has something to
     drop.
   - `_make_gp_te_2d_data(family_name)` — uses `x1` / `x2` (distinct
     column names from `gp_2d_data` so direct-vs-tensor cells don't
     collide in the matrix), with a separable + interaction signal so
     both `te` and `ti` have something to fit.

   Wire into `_get_data()`:
   ```python
   if config.data_type in ("gp_1d", "gp_2d", "gp_1d_par", "gp_te_2d"):
       return {
           "gp_1d":     _make_gp_1d_data,
           "gp_2d":     _make_gp_2d_data,
           "gp_1d_par": _make_gp_1d_par_data,
           "gp_te_2d":  _make_gp_te_2d_data,
       }[config.data_type](family)
   ```

3. **Update tolerance rules** per design §12.1.3:
   - In `_r_tol()`: add `"gp", "gp_2d", "gp_te", "gp_ti"` to the
     Gaussian-MODERATE list (same as TPRS / `te`).
   - In `_compare_fitted_not_coefs()`: add `"gp", "gp_2d",
     "gp_mixed", "gp_te", "gp_ti"` to the list. Direct GP has
     eigenvector sign ambiguity; tensor GP has the same plus tensor
     SVD-reparameterization ambiguity (`_svd_reparameterize` is
     orthogonal-equivalent across implementations). Both pathways
     must compare fitted values, never raw coefficients.

4. **`penalty_psd` is unconditionally safe** for direct GP: Commit
   D's always-on indefinite-eigenvalue clip (design §8.3) guarantees
   each marginal penalty diagonal is non-negative before it reaches
   fitting. Since tensor GP penalties are built from the same
   per-margin penalties (each margin runs through Commit D's clip),
   tensor cells inherit the guarantee. No `xfail` or relaxation is
   required. If a `test_penalty_psd` failure does occur for GP,
   **the clip is broken** — fix Commit D, do not relax the gate.

### Files touched

- Modify: `tests/test_validation_matrix.py` (add 5 configs + 4 data
  generators + tolerance rules)

### Validation

- `make test-cov` passes.
- New cells: 5 configs × 5 families = **25 new cells** (or 20 if
  `gp_ti` is dropped per the optional note above). With Commit A's
  validation-matrix consolidation (Phase 1) at 2 methods/cell, this
  is **50 new collected tests** (40 if `gp_ti` is dropped).
- All cells pass `test_matches_r` and `test_all_invariants`.
- `gp_te` and `gp_ti` confirm the registry → `tensor.py:146`
  dispatch produces fittable tensor GP smooths without any
  `tensor.py` code change.

### Exit criteria

- 25 (or 20) new validation-matrix cells exist and pass across
  direct (`gp`, `gp_2d`, `gp_mixed`) and tensor (`gp_te`,
  optionally `gp_ti`) configs.
- `gp_mixed` exercises the `x + s(x, bs='gp')` collision and
  `gam.side()` resolves it without singularity.
- `gp_te` / `gp_ti` exercise tensor GP through the existing
  wrappers (no tensor-side code change in this commit or any
  earlier one).
- Coverage ≥ 80%.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit I (the conditional Commit H from the original plan has been
  folded into D).

---

## Commit ~~H~~ — Indefinite Penalty Robustness (REMOVED)

> **This commit was removed during the design-doc review.** Current
> fitting code (`jax_utils.py:242,274`, `fitting/data.py:434,545`)
> cannot tolerate indefinite penalties at all — `slogdet` is clipped
> to `-1e10` for negative-determinant `S`, rank counting drops
> negative eigenvalues silently, and singleton reparameterization
> only scales positive eigenvalues. A real fix would be a non-local
> fitting overhaul.
>
> Instead, **Commit D ships an always-on absolute-value clip** on the
> truncated GP eigenvalues (design §8.3). The clip preserves rank
> and the diagonal pattern of `S`, deviates from mgcv only on
> specific indefinite spectra (uncommon at d ∈ {1, 2, 3}), and
> requires zero changes to `jax_utils.py` or `fitting/data.py`.
>
> The original Commit H said "skip and go straight to I" if no
> regression appeared. Since the clip is unconditional, that
> instruction now reads: **skip directly from H (validation matrix)
> to I (documentation).** No work happens at this slot.

### Exit criteria

- N/A — this slot is intentionally empty. **Proceed directly from
  Commit H (validation matrix) to Commit I (documentation).** If you
  found yourself wanting to add real work here, re-read design §8.3:
  any indefinite-penalty fix that touches `jax_utils.py` or
  `fitting/data.py` is out of scope and must be discussed with the
  user before opening any new commit.

---

## Commit I — Documentation Updates

**Goal:** Update user-facing documentation so GP appears in the smooth
catalog and API examples.

### What to do

1. **`docs/api.md`** — add a GP section under the smooth catalog,
   using the **Python-native kwargs only** (`kernel=`, `rho=`,
   `power=`, `stationary=`, `xt=`):
   - Brief description (low-rank kriging, Kammann-Wand reference).
   - Supported kernels by name string (`"spherical"`,
     `"power_exponential"`, `"matern_3_2"`, `"matern_5_2"`,
     `"matern_7_2"`). Case-insensitive; no aliases.
   - `xt` argument options.
   - Examples:
     - `gam("y ~ s(x, z, bs='gp', k=30)", data=df)` (defaults).
     - `gam("y ~ s(x, bs='gp', kernel='power_exponential', rho=0.5, power=2.0)", data=df)`.
     - `gam("y ~ s(x, bs='gp', kernel='spherical', stationary=True)", data=df)`.
   - One short implementation note: "mgcv encodes these knobs as a
     single signed numeric vector `m`; JaxGAM intentionally exposes
     them as named kwargs. Passing `m=` raises `ValueError` — see
     §6.4 of the GP design doc for the mgcv ↔ JaxGAM mapping."
   - Do **not** document `m=` in any example.

2. **`docs/index.md`** — add `"gp"` to the smooth-catalog table if one
   exists. Brief one-line description.

3. **`docs/design.md`** — add a one-line reference to
   `docs/gaussian_process/design.md` if the table of section refs
   doesn't already point there.

4. **Update `docs/gaussian_process/design.md` status**: change
   "Status: Proposed" to "Status: Implemented" with completion date.

### Files touched

- Modify: `docs/api.md`
- Modify: `docs/index.md`
- Possibly: `docs/design.md` (cross-reference)
- Modify: `docs/gaussian_process/design.md` (status header)

### Validation

- `make test-cov` passes (no code changes; docs only — but run anyway
  to confirm nothing was accidentally touched).
- Manually scan the rendered docs for typos and broken links.

### Exit criteria

- GP appears in the public smooth catalog.
- Design doc status reflects completion.
- **Agent stops and hands off to user for commit.** Do not proceed to
  Commit J.

---

## Commit J — Final Sweep

**Goal:** Re-run baseline measurements, confirm GP cells own their
behaviors uniquely (no R-parity duplication), verify the hard allow-list
is intact. This is the last commit before the user opens the PR.

### What to do

1. **Re-run baseline measurements** from Commit A:
   ```sh
   uv run pytest --collect-only -q tests | tail -3
   grep -rc "def test_" tests | grep -v ":0$" | awk -F: '{s+=$2} END {print s}'
   time make test-cov 2>&1 | tail -25
   ```
   Record the deltas vs the Commit-A baseline in
   `docs/gaussian_process/BASELINE.md`.

2. **Grep for R-bridge duplication** (per the test-cleanup pattern from
   `docs/clean_unit_tests/implementation_plan.md`):
   ```sh
   grep -rl "r_bridge\.fit_gam\|RBridge(" tests/ | \
     xargs grep -l "gp\|gaussian_process"
   ```
   Three files are expected to match, each with a distinct
   responsibility — no other file should hit:
   - `tests/test_validation_matrix.py` — owns **broad final-model R
     parity** (deviance / fitted-values / EDF / scale / theta across
     all GP cells × families).
   - `tests/test_smooths/test_gaussian_process.py` — owns
     **smooth-construct R parity** (E, S, X@X.T, null-space columns)
     plus structural / kernel-math tests.
   - `tests/test_r_bridge.py` — owns **bridge plumbing** for GP
     (`knots=` argument round-trip, `knt` / `gp.defn` / `E`
     extraction, `gp_config_to_mgcv_m`). This is test infrastructure,
     not R parity — the round-trip test asserts shapes and basic
     symmetry, never compares against a Python smooth.

   A fourth file matching the grep means R-parity coverage has
   drifted; investigate.

3. **Verify hard allow-list** still passes (TPRS, RE unseen-level,
   CoefficientMap roundtrip, step-halving, Cholesky stability, R bridge
   version check). Spot-check by:
   ```sh
   uv run pytest tests/test_smooths/test_tprs.py \
                  tests/test_smooths/test_random_effects.py \
                  tests/test_cholesky_stability.py \
                  tests/test_r_bridge.py -v
   ```

4. **Verify `CLAUDE.md` scope.** Confirm:
   - Line 164 (the "Exotic smooths" exclusion) does **not** list "GP".
     (This was fixed in a separate edit on 2026-05-23; if a future
     rebase restores the old text, fix it again here.)
   - Optionally add a brief mention of GP under the smooth types
     overview (§What This Project Is / §Code Conventions).

5. **Final `make test-cov` run.** Confirm:
   - All tests pass.
   - Coverage ≥ 80% per module (especially the new
     `jaxgam/smooths/gaussian_process.py`).
   - No silent regressions vs Commit-A baseline.

### Files touched

- Modify: `docs/gaussian_process/BASELINE.md` (final deltas)
- Possibly: `CLAUDE.md` (scope update)

### Validation

- `make test-cov` passes.
- All hard allow-list tests pass individually.
- Test count grew by approximately (post-consolidation, post-renumber,
  post-tensor-margin patch):
  - Commit C: +3-8  (kernel math + null-space, parametrized + collector)
  - Commit D: +5-7  (setup invariants + knot subsampling + indefinite-
                     clip regression, collector)
  - Commit E: +3-5  (parser collector for both `s()` and `te()`/`ti()`
                     + univariate-margin invariant test)
  - Commit F: +1    (RBridge GP round-trip, collector)
  - Commit G: +3-9  (direct-GP R-bridge smooth-construct, parametrized;
                     plus tensor-margin smooth-construct, parametrized
                     over `te` / `ti`)
  - Commit H: +40-50 (validation matrix 20-25 cells × 2 methods/cell —
                     5 configs if `gp_ti` included, 4 if dropped)
  - **Total**: roughly **+55-80 new collected tests**, with the spread
    driven mostly by whether `gp_ti` is included.

If the actual count came in above ~85, audit Commits C/D/G for missed
consolidation opportunities — re-read `_AssertCollector` usage in
existing tests (e.g. `tests/test_validation_matrix.py`,
`tests/test_smooths/test_random_effects.py`) for examples. The tensor
GP cells are non-negotiable; the consolidation must come from the
per-smooth files.

### Exit criteria

- Baseline deltas recorded.
- No R-parity duplication.
- Hard allow-list intact.
- Coverage ≥ 80% per module.
- `make test-cov` passes.
- **Agent stops and hands off to user.** The user reviews, commits, and
  opens the PR against `main`.

---

## Risk Management

### Risks specific to this feature

1. **Commit B refactor changes TPRS numerics.** Mitigation: the refactor
   is a pure code move (no algorithm change). The validation gate is
   "TPRS test count and outputs identical to Commit-A baseline". If any
   TPRS test changes output, the move is wrong — investigate before
   handoff.

2. **Knot sampling parity with R.** R's `sample()` algorithm differs
   from numpy's `Generator.choice()`. We accept that GP fits with
   subsampled knots cannot match R bit-for-bit; design §10.6 documents
   this and routes R-bridge tests through explicit-knot or no-subsample
   configurations.

3. **Eigenvector sign ambiguity.** Even with identical Lanczos starting
   vector, individual eigenvectors may sign-flip vs R. Tests in Commit F
   compare `X @ X.T` (sign-invariant) or fitted values, not raw `X` /
   coefficients. Design §10.3.

4. **Indefinite penalties.** Spherical and small-ρ power-exponential
   kernels produce negative eigenvalues. Current fitting code
   (`jax_utils.py:242,274`, `fitting/data.py:434,545`) does **not**
   tolerate indefinite penalties — it clips, drops, or scales them
   incorrectly. Commit D's always-on absolute-value clip on the
   truncated eigenvalues is the resolution; the original Commit H
   contingency was deleted. The clip deviates from mgcv only on
   spectra that produce negative eigenvalues (uncommon at our
   supported d ≤ 3). Commit G's R-parity tests skip the STRICT
   diagonal-equality check when R's spectrum is negative.

5. **`d > 3` default `k` bug in mgcv.** mgcv has a latent bug for
   higher dimensions (NA indexing). We raise an explicit error instead.
   No R comparison is attempted for `d > 3`. Documented in design §4.2.

6. **Parser drift.** Docs and tests assume Python-literal `extra_args`
   only — `c(...)` and `list(...)` are not parsed. Commit E pins this
   with a negative test so any future R-syntax patch must touch the GP
   parser tests deliberately.

   Note: the **`m=` rejection lives at the GP smooth-construction
   layer** (`GaussianProcessSmooth.__init__` in
   `gaussian_process.py`), not at the parser. The parser accepts
   `m=...` as an arbitrary `extra_args` entry; the smooth class
   raises `ValueError` from `__init__` when it reads that entry.
   Commit E pins this layering with one parser positive test
   (`m=[3, 0.5]` parses fine) plus an integration test in Commit D
   (instantiating `GaussianProcessSmooth(spec_with_m)` raises). Do
   not collapse the rejection into the parser — that would couple
   parser to GP semantics.

   The **Python-native API** (`kernel=`, `rho=`, `power=`,
   `stationary=`) is documented in design §1.6 and §6.4. Any future
   change that adds `m=` to public examples breaks the API contract;
   the existence of `gp_config_to_mgcv_m` (Commit F) is the only
   internal-only `m=` consumer.

7. **Tensor-margin pathway.** Registering `"gp"` makes
   `te(..., bs="gp")` and `ti(..., bs="gp")` work *automatically*
   through the existing tensor wrappers (`tensor.py:136-148`). This
   is **intentional** and mgcv supports the same. The risk is the
   inverse of what an earlier patch assumed: not that tensor GP
   leaks in, but that someone re-adds a guard rejecting it. Commit E
   includes a univariate-margin invariant test and Commit H's
   `gp_te` / `gp_ti` cells as the regression gate against that.

8. **Tensor `k` is scalar in JaxGAM today.** `TensorProductSmooth._create_marginals()`
   at `tensor.py:140` passes `self.spec.k` (a single integer) to every
   marginal `SmoothSpec`. R's `te()` accepts `k=c(k_1, …, k_d)`. For
   R-parity tests, when Python writes `te(x1, x2, bs='gp', k=5)`, R
   must write `te(x1, x2, bs='gp', k=c(5, 5))`. Per-margin distinct
   `k` is a tensor-wrapper enhancement orthogonal to GP and out of
   scope for this PR.

### Rollback plan

Each commit lands one logical change on a single working branch.
Individual commits can be reverted with `git revert <sha>` without
disturbing others. Most commits touch disjoint file sets:

- Commit B (refactor) is the most invasive. If it causes problems,
  revert removes the move; GP commits C-J on top would also need to be
  rebased to import from `tprs.py` instead of `utils.py`. Prefer fixing
  Commit B in-place over reverting.
- Commits C-I are additive (new files / new tests / new docs) and can
  be reverted cleanly.

---

## Sequencing Summary

All commits land on a single working branch off `main`. The agent
executes one unit at a time, validates, hands off; the user commits
manually before triggering the next unit.

```
Phase 0
Commit A    Baseline capture                                  ── docs/gaussian_process/BASELINE.md
        │
Phase 1 — Shared infrastructure
Commit B    Extract _slanczos + 3 helpers to utils.py         ── tprs.py, utils.py
        │   (TPRS tests must pass with identical output)
        │
Phase 2 — GP implementation
Commit C    GP kernel module (classes + registry)             ── gp_kernels.py, test_gp_kernels.py
            (GPKernel ABC + five kernel classes;
             gp_kernel_registry via jaxgam.registry.Registry
             keyed by canonical names only (no aliases).
             No config dataclass, no parser, no helpers —
             all that lives on GaussianProcessSmooth.)
Commit D    GaussianProcessSmooth class + structural tests    ── gaussian_process.py, test_gaussian_process.py, conftest.py
            (__init__ resolves spec.extra_args via the
             registry and stores self._kernel/_rho/_power/
             _stationary; _gp_E and _gp_T are methods;
             m= rejection in __init__; indefinite-eigenvalue
             clip per design §8.3.)
Commit E    Registry + univariate-margin invariant + parser   ── registry.py, test_gaussian_process.py, test_parser.py
            (NO tensor.py change — tensor GP enabled by
             registration alone via tensor.py:146 dispatch.
             Parser tests use Python-native kwargs only —
             kernel=/rho=/power=/stationary=; c()/list() not
             supported.)
        │
Phase 3 — R parity infrastructure + tests
Commit F    RBridge GP enhancements + gp_config_to_mgcv_m     ── r_bridge.py, test_r_bridge.py
            (knots= argument, knt/gp.defn extraction,
             SmoothSpec → mgcv-m helper; no GP-side code)
Commit G    Direct GP + tensor-margin R smooth-construct      ── test_gaussian_process.py
            (parametrize over GP kwargs dicts; R formula built
             via gp_config_to_mgcv_m so the two sides cannot drift)
Commit H    Validation matrix integration (20-25 cells)       ── test_validation_matrix.py
            (direct: gp, gp_2d, gp_mixed; tensor: gp_te,
             optionally gp_ti — all using default kernel,
             no kwargs needed)
        │
Phase 4 — Docs + finalize
~~Commit H (old) — indefinite penalty robustness — REMOVED, handled in D~~
Commit I    Documentation updates (Python-native API only)    ── docs/api.md, docs/index.md
        │
Commit J    Final sweep + baseline deltas + ownership check   ── BASELINE.md, possibly CLAUDE.md
        │
User opens single PR against main.
```

Commits A and B must complete before Phase 2 starts (Commits C-D depend
on `utils.py` having the shared kriging helpers). Commit F is pure
test-infrastructure work that enables Commit G's R-parity tests. The
old conditional Commit H is gone — the indefinite-eigenvalue handling
is baked into Commit D's setup and runs unconditionally.

---

## Definition of Done

Met when the user opens the single PR for this feature:

- **Direct GP** via `s(..., bs="gp")` works in 1D, 2D, and 3D, with
  the **Python-native kwargs** documented in design §1.6 / §6:
  `s(x, bs="gp")` (defaults), `s(x, z, bs="gp", k=30)`,
  `s(x, bs="gp", kernel="matern_3_2", rho=0.5)`,
  `s(x, bs="gp", kernel="power_exponential", rho=0.5, power=2.0)`,
  and `s(x, bs="gp", kernel="spherical", stationary=True)` all
  parse, construct, fit, and predict via the public `gam()` API.
- **`m=` is rejected.** Passing `m=...` to a GP smooth raises a
  clear `ValueError` from `GaussianProcessSmooth.__init__` at
  construction time, naming the four replacement kwargs. R-style
  `c(...)` is rejected one layer earlier, at the parser, by
  `ast.literal_eval`.
- **Tensor-margin GP** via the existing wrappers works:
  `te(x1, x2, bs="gp", k=5)` and `ti(x1, x2, bs="gp", k=5)` parse,
  construct, fit, and predict — through `TensorProductSmooth` /
  `TensorInteractionSmooth` instantiating `GaussianProcessSmooth`
  per margin via the registry. **No code change in `tensor.py`.**
- **Kernel registry** reuses the existing
  `jaxgam.registry.Registry[T]` generic (same abstraction used by
  smooth, family, and link registries) — no bespoke wrapper. The
  registry's keys are the canonical kernel names; lookups are
  case-insensitive but there are no aliases.
- **mgcv-side conversion** happens only at the R-bridge boundary
  via `tests/r_bridge.py:gp_config_to_mgcv_m`. No production code
  consumes mgcv's `m=` numeric encoding.
- All 5 kernels (spherical, power-exponential, Matérn 3/2, 5/2, 7/2)
  are supported and tested at STRICT against closed-form.
- Stationary and non-stationary modes both work.
- Indefinite-eigenvalue clip in Commit D fires on at least one
  regression test and produces a non-negative penalty diagonal; no GP
  cell ever sees an indefinite `S` reach fitting (direct or tensor).
- 20-25 new validation-matrix cells (4 or 5 configs × 5 families) all
  pass `test_matches_r` and `test_all_invariants` (2 methods/cell —
  matches the consolidated structure from `docs/clean_unit_tests/`).
  Direct cells (`gp`, `gp_2d`, `gp_mixed`) and tensor cells (`gp_te`,
  optionally `gp_ti`) are both present; `gp_2d` and `gp_te` are not
  redundant — they exercise mathematically distinct constructions.
  `gp_mixed` uses `x + s(x, bs='gp')` (same variable) and exercises
  gam.side identifiability.
- R smooth-construct comparison tests pass at STRICT (with explicit
  knots threaded through the Commit-F bridge `knots=` argument, via
  `X @ X.T` for sign-invariance). The STRICT eigenvalue check is
  skipped on R-side indefinite spectra (documented deviation, §8.3).
- **Consolidation discipline held**: total new collected tests in the
  ~55-80 range (widened from the original ~45 budget because of
  tensor-margin coverage), not the 120+ a naive per-assertion
  approach would yield. Every multi-assertion test that shares a
  fixture/R fit uses `_AssertCollector`; every kernel-family
  enumeration uses `@pytest.mark.parametrize`.
- **No R-parity duplication**: smooth-construct R parity lives only in
  `test_gaussian_process.py`; final-model R parity lives only in
  `test_validation_matrix.py`; bridge-level GP round-trip lives only in
  `test_r_bridge.py`. Verified by the Commit J grep.
- `make test-cov` passes on the final commit with ≥ 80% coverage.
- Hard allow-list intact: all TPRS, RE, CoefficientMap roundtrip, step-
  halving, Cholesky stability, and R-bridge version tests still pass.
  Existing non-GP bridge tests pass with the new `knots=None` default
  unchanged.
- `docs/gaussian_process/design.md` status is "Implemented".
- `docs/api.md` documents GP usage with Python-literal `extra_args`.
- `CLAUDE.md` does not list GP under §What Is NOT in v1.0 (fixed
  2026-05-23; verify still correct at PR time).
- All nine active commit slots (A-G, H, I, J — the old H is an empty
  marker) are present in the branch history as individual commits
  authored by the user.
