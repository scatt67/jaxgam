# Registry Refactor — Implementation Plan

LLM-optimized step-by-step plan.

**Before starting:** Read `docs/refactor_class_system/design.md` lines 1-19 for the problem statement and goals driving this refactor. Read lines 121-132 for the inconsistency summary table that motivates unifying the three registries. Read lines 134-146 for alternatives considered and why `Registry[T]` was chosen.

## Step 1: Create `jaxgam/registry.py` ✅

Create generic `Registry[T]` class. No dependencies on smooths/families/links.

**Design context:** Read `design.md` lines 148-274 for the full `Registry[T]` API specification, constructor signature, all method signatures with docstrings, and the key design decisions (immutability via `MappingProxyType`, case-insensitive keys, `cache_instances` flag, `get_class()` vs `get_instance()` separation, and error type rationale).

- `MappingProxyType` for immutability
- `get_class(key) -> type[T]` — case-insensitive, raises `KeyError`
- `get_instance(key) -> T` — optional caching via `cache_instances` flag
- `.available` property, `__contains__`, `__len__`, `__repr__`
- Error format: `f"Unknown {self._name}: {key!r}. Available: {available}"`

**Verify:** `python -c "from jaxgam.registry import Registry"`

## Step 2: Create `tests/test_registry.py` ✅

Test `Registry` in isolation with a trivial `DummyClass`.

**Design context:** Read `design.md` lines 514-537 for the full test list and the specific existing test assertions that must remain green.

Cover: `get_class` valid/invalid, case-insensitive lookup, `get_instance` cached/uncached, `available`, `__contains__`, `__len__`, `__repr__`, immutability (`TypeError` on `_entries` mutation).

**Verify:** `make test-local` (or `python -m pytest tests/test_registry.py -v`)

## Step 3: Migrate smooth registry ✅

**Design context:** Read `design.md` lines 21-48 for the current smooth registry implementation and its call sites. Read lines 278-321 for the exact before/after code and error message compatibility analysis.

**File:** `jaxgam/smooths/registry.py`

1. Add `from jaxgam.registry import Registry`
2. Replace `_SMOOTH_REGISTRY` dict with `smooth_registry = Registry({...}, name="basis type")`
3. Simplify `get_smooth_class()` to delegate: `return smooth_registry.get_class(bs_name)`

**File:** `jaxgam/smooths/__init__.py`

4. Add `from jaxgam.smooths.registry import smooth_registry` import
5. Add `"smooth_registry"` to `__all__`

**Verify:** `make test-local` — all smooth tests pass unchanged.

## Step 4: Migrate family registry ✅

**Design context:** Read `design.md` lines 50-85 for the current family registry implementation, its call sites, and the special caching behavior for JAX JIT object identity. Read lines 323-369 for the exact before/after code, error message compatibility analysis (the `"Available families"` → `"Available"` change), and how `cache_instances=True` replaces the manual `_FAMILY_CACHE`.

**File:** `jaxgam/families/registry.py`

1. Add `from jaxgam.registry import Registry`
2. Replace `_FAMILY_REGISTRY` + `_FAMILY_CACHE` with `family_registry = Registry({...}, name="family", cache_instances=True)`
3. Simplify `get_family()`:
   - Pass-through for `ExponentialFamily` instances (unchanged)
   - String path: `return family_registry.get_instance(name_or_instance)`
   - `TypeError` path unchanged

**File:** `jaxgam/families/__init__.py`

4. Add `from jaxgam.families.registry import family_registry` import
5. Add `"family_registry"` to `__all__`

**Critical check:** `tests/test_families.py:683` matches `"Unknown family"` — the `name="family"` parameter produces `"Unknown family: ..."` which matches.

**Verify:** `make test-local` — all family tests pass unchanged.

## Step 5: Migrate link registry ✅

**Design context:** Read `design.md` lines 87-119 for the current link registry (embedded in the ABC) and its `ValueError` (not `KeyError`) error behavior. Read lines 371-428 for the full migration plan: creating `jaxgam/links/registry.py`, removing `_LINK_REGISTRY` from `links.py`, and the `from_name()` wrapper that catches `KeyError` and re-raises as `ValueError` for backward compatibility. Read lines 430-440 for the error handling summary table.

**File:** `jaxgam/links/registry.py` (CREATE)

1. Import all link classes from `jaxgam.links.links`
2. Import `Registry` from `jaxgam.registry`
3. Create `link_registry = Registry({...}, name="link function")`
4. Create `get_link(name: str) -> Link` wrapping `link_registry.get_instance(name)`

**File:** `jaxgam/links/links.py` (MODIFY)

5. Remove `_LINK_REGISTRY` dict (lines 324-333)
6. Update `Link.from_name()`:
   ```python
   @staticmethod
   def from_name(name: str) -> Link:
       from jaxgam.links.registry import get_link
       try:
           return get_link(name)
       except KeyError:
           from jaxgam.links.registry import link_registry
           valid = ", ".join(link_registry.available)
           raise ValueError(
               f"Unknown link function {name!r}. Valid options: {valid}"
           ) from None
   ```

**File:** `jaxgam/links/__init__.py` (MODIFY)

7. Add imports: `from jaxgam.links.registry import link_registry, get_link`
8. Add `"link_registry"`, `"get_link"` to `__all__`

**Critical check:** `tests/test_links.py:318` matches `ValueError` + `"Unknown link function"` — preserved by the `from_name()` wrapper.

**Verify:** `make test-local` — all link tests pass unchanged.

## Step 6: Final verification ✅

**Design context:** Read `design.md` lines 442-470 for the full compatibility guarantees — zero-impact public API signatures, zero-impact call sites, and the list of new additions.

```bash
make test-local          # all tests green
python -c "
from jaxgam.smooths import smooth_registry, get_smooth_class
from jaxgam.families import family_registry, get_family
from jaxgam.links import link_registry, get_link
print(smooth_registry)
print(family_registry)
print(link_registry)
print(smooth_registry.available)
print(family_registry.available)
print(link_registry.available)
"
```

## File checklist

| # | File | Action | Design doc reference |
|---|------|--------|---------------------|
| 1 | `jaxgam/registry.py` | CREATE | Lines 148-262 (full API) |
| 2 | `tests/test_registry.py` | CREATE | Lines 514-537 (test plan) |
| 3 | `jaxgam/smooths/registry.py` | MODIFY | Lines 278-321 (before/after + compat) |
| 4 | `jaxgam/smooths/__init__.py` | MODIFY | Line 321 (export note) |
| 5 | `jaxgam/families/registry.py` | MODIFY | Lines 323-369 (before/after + compat) |
| 6 | `jaxgam/families/__init__.py` | MODIFY | Line 369 (export note) |
| 7 | `jaxgam/links/registry.py` | CREATE | Lines 371-428 (new file + migration) |
| 8 | `jaxgam/links/links.py` | MODIFY | Lines 405-427 (remove dict, update from_name) |
| 9 | `jaxgam/links/__init__.py` | MODIFY | Line 428 (export note) |
