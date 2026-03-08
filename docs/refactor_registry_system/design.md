# Registry Refactor — Design Document

## 1. Problem Statement

JaxGAM has three separate registries mapping string keys to classes/instances:

| Registry | Location | Pattern | Caching |
|----------|----------|---------|---------|
| Smooth | `smooths/registry.py` | Module-level `dict` | None |
| Family | `families/registry.py` | Module-level `dict` + separate `_FAMILY_CACHE` dict | Manual per-key cache |
| Link | `links/links.py` | Module-level `dict` embedded alongside the ABC | None |

These registries do the same thing — map `str → type[T]` — but each uses slightly different patterns, error types, error messages, and caching strategies. This makes the codebase harder to maintain and extend.

**Goals:**
1. Unify all three under a single generic `Registry[T]` class.
2. Make registry contents immutable (no accidental mutation of the global dict).
3. Provide consistent, introspectable API (`.available`, `__contains__`, `__len__`).
4. Preserve all existing public signatures — zero breaking changes.

## 2. Current State Analysis

### 2.1 Smooth Registry (`jaxgam/smooths/registry.py`)

```python
_SMOOTH_REGISTRY: dict[str, type[Smooth]] = {
    "tp": TPRSSmooth,
    "ts": TPRSShrinkageSmooth,
    "cr": CubicRegressionSmooth,
    "cs": CubicShrinkageSmooth,
    "cc": CyclicCubicSmooth,
    "te": TensorProductSmooth,
    "ti": TensorInteractionSmooth,
}

def get_smooth_class(bs_name: str) -> type[Smooth]:
    key = bs_name.lower()
    if key not in _SMOOTH_REGISTRY:
        available = ", ".join(sorted(_SMOOTH_REGISTRY.keys()))
        raise KeyError(f"Unknown basis type: {bs_name!r}. Available: {available}")
    return _SMOOTH_REGISTRY[key]
```

**Call sites:**
- `jaxgam/formula/design.py:654` — `get_smooth_class(key)` to construct smooths during formula setup.
- `jaxgam/smooths/tensor.py:180` — lazy import to break circular dependency when building marginal smooths.

**Error behavior:** Raises `KeyError` with message `"Unknown basis type: {name!r}. Available: {available}"`.

### 2.2 Family Registry (`jaxgam/families/registry.py`)

```python
_FAMILY_REGISTRY: dict[str, type[ExponentialFamily]] = {
    "gaussian": Gaussian,
    "binomial": Binomial,
    "poisson": Poisson,
    "gamma": Gamma,
}

_FAMILY_CACHE: dict[str, ExponentialFamily] = {}

def get_family(name_or_instance: str | ExponentialFamily) -> ExponentialFamily:
    if isinstance(name_or_instance, ExponentialFamily):
        return name_or_instance
    if isinstance(name_or_instance, str):
        key = name_or_instance.lower()
        if key not in _FAMILY_REGISTRY:
            available = ", ".join(sorted(_FAMILY_REGISTRY.keys()))
            raise KeyError(
                f"Unknown family: {name_or_instance!r}. Available families: {available}"
            )
        if key not in _FAMILY_CACHE:
            _FAMILY_CACHE[key] = _FAMILY_REGISTRY[key]()
        return _FAMILY_CACHE[key]
    raise TypeError(...)
```

**Call sites:**
- `jaxgam/api.py:139` — `get_family(self.family)` during fit orchestration.

**Special behavior:**
- Accepts either a string or an existing `ExponentialFamily` instance (pass-through).
- Instance caching: default-link families are cached so JAX JIT sees the same object identity across fits, avoiding recompilation.

**Error behavior:** Raises `KeyError` with message `"Unknown family: {name!r}. Available families: {available}"`.

### 2.3 Link Registry (`jaxgam/links/links.py`)

```python
_LINK_REGISTRY: dict[str, type[Link]] = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "inverse": InverseLink,
    "probit": ProbitLink,
    "cloglog": CloglogLink,
    "sqrt": SqrtLink,
    "inverse_squared": InverseSquaredLink,
}
```

Accessed via `Link.from_name()` static method:

```python
@staticmethod
def from_name(name: str) -> Link:
    try:
        return _LINK_REGISTRY[name]()
    except KeyError:
        valid = ", ".join(sorted(_LINK_REGISTRY))
        raise ValueError(
            f"Unknown link function {name!r}. Valid options: {valid}"
        ) from None
```

**Call sites:**
- `jaxgam/families/base.py:52` — `Link.from_name(link)` during family construction.

**Error behavior:** Raises `ValueError` (not `KeyError`) with message `"Unknown link function {name!r}. Valid options: {valid}"`.

### 2.4 Summary of Inconsistencies

| Aspect | Smooth | Family | Link |
|--------|--------|--------|------|
| Dict location | Own module | Own module | Inside ABC module |
| Dict visibility | Module-level `_` | Module-level `_` | Module-level `_` |
| Error type | `KeyError` | `KeyError` | `ValueError` |
| Error message pattern | `"Unknown basis type: ..."` | `"Unknown family: ..."` | `"Unknown link function ..."` |
| Available key label | `"Available"` | `"Available families"` | `"Valid options"` |
| Caching | None | Separate `_FAMILY_CACHE` | None (instantiates each call) |
| Case-insensitive | Yes (`.lower()`) | Yes (`.lower()`) | No |
| Immutability | None | None | None |

## 3. Design Alternatives Considered

### 3.1 Frozen Dataclass
Each registry as a `@dataclass(frozen=True)` with fields for each entry. Rejected: inflexible for varying numbers of entries, adds boilerplate.

### 3.2 Enum
`class SmoothType(Enum): TP = TPRSSmooth`. Rejected: awkward API (`SmoothType["TP"].value`), doesn't naturally support case-insensitive lookup.

### 3.3 Plain Dict + Helper Functions (status quo)
Keep dicts, just standardize the helper functions. Rejected: doesn't solve mutability or provide introspection.

### 3.4 Generic `Registry[T]` Class (chosen)
A single generic class wrapping a `MappingProxyType` (read-only view of dict). Provides consistent lookup, caching, introspection, and error handling. Minimal code, maximum consistency.

## 4. Chosen Approach: `Registry[T]`

### 4.1 Location

`jaxgam/registry.py` — top-level module, no dependencies on smooths/families/links.

### 4.2 Full API

```python
from __future__ import annotations

from types import MappingProxyType
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Immutable, case-insensitive registry mapping string keys to classes.

    Parameters
    ----------
    entries : dict[str, type[T]]
        Mapping of canonical names to classes. Keys are normalized to
        lowercase on insertion.
    name : str
        Human-readable registry name for error messages (e.g. "smooth",
        "family", "link").
    cache_instances : bool
        If True, ``get_instance()`` caches created instances by key.
        Useful for families where JAX JIT cache benefits from object
        identity stability.
    """

    def __init__(
        self,
        entries: dict[str, type[T]],
        name: str,
        cache_instances: bool = False,
    ) -> None:
        self._name = name
        self._entries: MappingProxyType[str, type[T]] = MappingProxyType(
            {k.lower(): v for k, v in entries.items()}
        )
        self._cache_instances = cache_instances
        self._instance_cache: dict[str, T] = {}

    def get_class(self, key: str) -> type[T]:
        """Look up a class by name (case-insensitive).

        Parameters
        ----------
        key : str
            Registry key (e.g. "tp", "gaussian", "logit").

        Returns
        -------
        type[T]
            The registered class.

        Raises
        ------
        KeyError
            If the key is not found.
        """
        normalized = key.lower()
        try:
            return self._entries[normalized]
        except KeyError:
            available = ", ".join(sorted(self._entries))
            raise KeyError(
                f"Unknown {self._name}: {key!r}. Available: {available}"
            ) from None

    def get_instance(self, key: str) -> T:
        """Look up and return an instance (optionally cached).

        Instantiates the class with no arguments. If ``cache_instances``
        was set to True, subsequent calls with the same key return the
        same object.

        Parameters
        ----------
        key : str
            Registry key.

        Returns
        -------
        T
            An instance of the registered class.
        """
        normalized = key.lower()
        if self._cache_instances and normalized in self._instance_cache:
            return self._instance_cache[normalized]
        cls = self.get_class(key)
        instance = cls()
        if self._cache_instances:
            self._instance_cache[normalized] = instance
        return instance

    @property
    def available(self) -> tuple[str, ...]:
        """Sorted tuple of all registered keys."""
        return tuple(sorted(self._entries))

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._entries))
        return f"Registry({self._name!r}, [{keys}])"
```

### 4.3 Key Design Decisions

**Immutability via `MappingProxyType`:** The internal `_entries` dict is wrapped in a read-only proxy. No `register()` method — all entries are provided at construction time. This prevents accidental mutation of global state.

**Case-insensitive keys:** All keys are lowercased on insertion and lookup. This matches the existing behavior of smooth and family registries and adds it to the link registry (which currently lacks it).

**`cache_instances` flag:** Only the family registry needs instance caching (for JAX JIT object identity). Rather than building a separate mechanism, this is a constructor flag. Smooth and link registries set it to `False`.

**`get_class()` vs `get_instance()`:** Two distinct methods. `get_class()` returns `type[T]` (needed by smooth registry). `get_instance()` returns `T` (needed by link and family registries). This avoids overloading a single method with ambiguous return types.

**Error type is always `KeyError`:** All three registries will raise `KeyError` for unknown keys. The link registry currently raises `ValueError`; `Link.from_name()` will catch the `KeyError` and re-raise as `ValueError` to preserve backward compatibility for callers matching on `ValueError`.

## 5. Per-Registry Migration

### 5.1 Smooth Registry

**File:** `jaxgam/smooths/registry.py`

**Before:**
```python
_SMOOTH_REGISTRY: dict[str, type[Smooth]] = { ... }

def get_smooth_class(bs_name: str) -> type[Smooth]:
    key = bs_name.lower()
    if key not in _SMOOTH_REGISTRY:
        available = ", ".join(sorted(_SMOOTH_REGISTRY.keys()))
        raise KeyError(f"Unknown basis type: {bs_name!r}. Available: {available}")
    return _SMOOTH_REGISTRY[key]
```

**After:**
```python
from jaxgam.registry import Registry

smooth_registry: Registry[Smooth] = Registry(
    {
        "tp": TPRSSmooth,
        "ts": TPRSShrinkageSmooth,
        "cr": CubicRegressionSmooth,
        "cs": CubicShrinkageSmooth,
        "cc": CyclicCubicSmooth,
        "te": TensorProductSmooth,
        "ti": TensorInteractionSmooth,
    },
    name="basis type",
)

def get_smooth_class(bs_name: str) -> type[Smooth]:
    """Look up a Smooth class by basis type name.

    Thin wrapper around ``smooth_registry.get_class()`` for backward compatibility.
    """
    return smooth_registry.get_class(bs_name)
```

**Error message compatibility:** The `Registry.get_class()` error message is `"Unknown basis type: {key!r}. Available: {available}"` which matches the existing pattern exactly (the `name` parameter is set to `"basis type"`).

**Exports:** `smooth_registry` is added to `jaxgam/smooths/__init__.py` `__all__`.

### 5.2 Family Registry

**File:** `jaxgam/families/registry.py`

**Before:**
```python
_FAMILY_REGISTRY: dict[str, type[ExponentialFamily]] = { ... }
_FAMILY_CACHE: dict[str, ExponentialFamily] = {}

def get_family(name_or_instance: str | ExponentialFamily) -> ExponentialFamily:
    ...
```

**After:**
```python
from jaxgam.registry import Registry

family_registry: Registry[ExponentialFamily] = Registry(
    {
        "gaussian": Gaussian,
        "binomial": Binomial,
        "poisson": Poisson,
        "gamma": Gamma,
    },
    name="family",
    cache_instances=True,
)

def get_family(name_or_instance: str | ExponentialFamily) -> ExponentialFamily:
    """Look up and return a family instance.
    [docstring preserved]
    """
    if isinstance(name_or_instance, ExponentialFamily):
        return name_or_instance
    if isinstance(name_or_instance, str):
        return family_registry.get_instance(name_or_instance)
    raise TypeError(
        f"Expected a string or ExponentialFamily instance, "
        f"got {type(name_or_instance)!r}"
    )
```

**Error message compatibility:** The current error is `"Unknown family: {name!r}. Available families: {available}"`. The new `Registry` with `name="family"` produces `"Unknown family: {key!r}. Available: {available}"`. The test at `tests/test_families.py:683` matches on `"Unknown family"` which still matches. The phrase `"Available families"` changes to `"Available"` — this is acceptable since no test matches on that suffix.

**Caching:** The `cache_instances=True` flag replaces the manual `_FAMILY_CACHE` dict. Semantics are identical: first call creates an instance, subsequent calls return the cached one.

**Exports:** `family_registry` is added to `jaxgam/families/__init__.py` `__all__`.

### 5.3 Link Registry

**Files:** `jaxgam/links/registry.py` (new), `jaxgam/links/links.py` (modified)

**New file — `jaxgam/links/registry.py`:**
```python
from jaxgam.links.links import (
    CloglogLink, IdentityLink, InverseLink, InverseSquaredLink,
    Link, LogLink, LogitLink, ProbitLink, SqrtLink,
)
from jaxgam.registry import Registry

link_registry: Registry[Link] = Registry(
    {
        "identity": IdentityLink,
        "log": LogLink,
        "logit": LogitLink,
        "inverse": InverseLink,
        "probit": ProbitLink,
        "cloglog": CloglogLink,
        "sqrt": SqrtLink,
        "inverse_squared": InverseSquaredLink,
    },
    name="link function",
)

def get_link(name: str) -> Link:
    """Look up a link function by name.

    Thin wrapper around ``link_registry.get_instance()``.
    """
    return link_registry.get_instance(name)
```

**Modified — `jaxgam/links/links.py`:**

Remove `_LINK_REGISTRY` dict. Update `Link.from_name()` to delegate:

```python
@staticmethod
def from_name(name: str) -> Link:
    """Look up a link function by name.
    [docstring preserved]
    """
    from jaxgam.links.registry import get_link

    try:
        return get_link(name)
    except KeyError:
        valid = ", ".join(sorted(link_registry.available))
        raise ValueError(
            f"Unknown link function {name!r}. Valid options: {valid}"
        ) from None
```

**Error compatibility:** `Link.from_name()` continues to raise `ValueError` (not `KeyError`) so `tests/test_links.py:318` (`pytest.raises(ValueError, match="Unknown link function")`) continues to pass. The new `get_link()` function raises `KeyError` (via `Registry.get_class()`), but `from_name()` catches and converts it.

**Exports:** `get_link` and `link_registry` are added to `jaxgam/links/__init__.py` `__all__`.

## 6. Error Handling Summary

| Caller | Exception | Message pattern | Test match string |
|--------|-----------|-----------------|-------------------|
| `smooth_registry.get_class()` | `KeyError` | `"Unknown basis type: {key!r}. Available: ..."` | (no test matches on this directly) |
| `get_smooth_class()` | `KeyError` | same (delegates) | same |
| `family_registry.get_instance()` | `KeyError` | `"Unknown family: {key!r}. Available: ..."` | `"Unknown family"` |
| `get_family()` | `KeyError` / `TypeError` | same (delegates) | `"Unknown family"` |
| `link_registry.get_instance()` | `KeyError` | `"Unknown link function: {key!r}. Available: ..."` | (not tested directly) |
| `get_link()` | `KeyError` | same (delegates) | (not tested directly) |
| `Link.from_name()` | `ValueError` | `"Unknown link function {name!r}. Valid options: ..."` | `"Unknown link function"` |

## 7. Compatibility Guarantees

### 7.1 Zero-Impact Public API

These functions retain their exact signatures and behavior:

- `get_smooth_class(bs_name: str) -> type[Smooth]` — same `KeyError`.
- `get_family(name_or_instance: str | ExponentialFamily) -> ExponentialFamily` — same `KeyError`/`TypeError`, same caching.
- `Link.from_name(name: str) -> Link` — same `ValueError`.

### 7.2 Zero-Impact Call Sites

These files call the wrapper functions, not the registries directly, and require no changes:

- `jaxgam/api.py` — calls `get_family()`
- `jaxgam/formula/design.py` — calls `get_smooth_class()`
- `jaxgam/smooths/tensor.py` — calls `get_smooth_class()`
- `jaxgam/families/base.py` — calls `Link.from_name()`
- All existing test files

### 7.3 New Public API

These are **additions only** — no existing names are removed or changed:

- `jaxgam.registry.Registry` — generic class
- `jaxgam.smooths.registry.smooth_registry` — `Registry[Smooth]` instance
- `jaxgam.families.registry.family_registry` — `Registry[ExponentialFamily]` instance
- `jaxgam.links.registry.link_registry` — `Registry[Link]` instance
- `jaxgam.links.registry.get_link()` — convenience function

## 8. Adding New Entries

To add a new smooth type (e.g., P-spline `"ps"`):

```python
# In jaxgam/smooths/registry.py, add to the smooth_registry constructor dict:
smooth_registry: Registry[Smooth] = Registry(
    {
        ...
        "ps": PSplineSmooth,  # new entry
    },
    name="basis type",
)
```

To add a new family (e.g., Negative Binomial):

```python
# In jaxgam/families/registry.py, add to the family_registry constructor dict:
family_registry: Registry[ExponentialFamily] = Registry(
    {
        ...
        "nb": NegativeBinomial,  # new entry
    },
    name="family",
    cache_instances=True,
)
```

To add a new link function:

```python
# In jaxgam/links/registry.py, add to the link_registry constructor dict:
link_registry: Registry[Link] = Registry(
    {
        ...
        "cauchit": CauchitLink,  # new entry
    },
    name="link function",
)
```

## 9. Testing Strategy

### 9.1 New Tests (`tests/test_registry.py`)

Test the generic `Registry[T]` class in isolation:

- **`test_get_class_valid`** — known key returns correct class.
- **`test_get_class_unknown`** — unknown key raises `KeyError` with correct message.
- **`test_get_class_case_insensitive`** — `"TP"`, `"Tp"`, `"tp"` all resolve.
- **`test_get_instance`** — returns an instance of the registered class.
- **`test_get_instance_cached`** — with `cache_instances=True`, same object returned.
- **`test_get_instance_uncached`** — with `cache_instances=False`, different objects.
- **`test_available`** — returns sorted tuple of keys.
- **`test_contains`** — `"tp" in registry` works, case-insensitive.
- **`test_len`** — returns correct count.
- **`test_repr`** — includes name and keys.
- **`test_immutable`** — cannot add/remove entries via `_entries`.

### 9.2 Existing Tests

All existing tests must pass unchanged. Key tests to watch:

- `tests/test_families.py:683` — `pytest.raises(KeyError, match="Unknown family")`
- `tests/test_links.py:318` — `pytest.raises(ValueError, match="Unknown link function")`

## 10. File Change Summary

| File | Action | Lines changed (est.) |
|------|--------|---------------------|
| `jaxgam/registry.py` | CREATE | ~80 |
| `jaxgam/links/registry.py` | CREATE | ~30 |
| `jaxgam/smooths/registry.py` | MODIFY | ~20 (replace dict with `smooth_registry`, simplify function) |
| `jaxgam/families/registry.py` | MODIFY | ~20 (replace dict+cache with `family_registry`, simplify function) |
| `jaxgam/links/links.py` | MODIFY | ~15 (remove `_LINK_REGISTRY`, update `from_name()`) |
| `jaxgam/links/__init__.py` | MODIFY | ~3 (add exports) |
| `jaxgam/smooths/__init__.py` | MODIFY | ~2 (add export) |
| `jaxgam/families/__init__.py` | MODIFY | ~2 (add export) |
| `tests/test_registry.py` | CREATE | ~80 |
