"""Generic registry mapping string keys to classes.

Built-in entries are immutable. Users can register additional entries
via ``register()`` without overriding built-ins.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Case-insensitive registry mapping string keys to classes.

    Built-in entries (provided at construction) are immutable. Custom
    entries can be added at runtime via :meth:`register` but cannot
    override existing keys.

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
        self._custom_entries: dict[str, type[T]] = {}
        self._cache_instances = cache_instances
        self._instance_cache: dict[str, T] = {}

    def register(self, key: str, cls: type[T]) -> None:
        """Register a custom entry.

        Parameters
        ----------
        key : str
            Registry key (case-insensitive).
        cls : type[T]
            The class to register.

        Raises
        ------
        ValueError
            If the key is already registered (built-in or custom).
        """
        normalized = key.lower()
        if normalized in self._entries or normalized in self._custom_entries:
            raise ValueError(
                f"Key {key!r} is already registered in the "
                f"{self._name} registry."
            )
        self._custom_entries[normalized] = cls

    def _lookup(self, normalized: str) -> type[T] | None:
        """Return the class for a normalized key, or None."""
        if normalized in self._entries:
            return self._entries[normalized]
        return self._custom_entries.get(normalized)

    @property
    def _all_keys(self) -> set[str]:
        """Union of built-in and custom keys."""
        return set(self._entries) | set(self._custom_entries)

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
        cls = self._lookup(normalized)
        if cls is not None:
            return cls
        available = ", ".join(sorted(self._all_keys))
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
        return tuple(sorted(self._all_keys))

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._all_keys

    def __len__(self) -> int:
        return len(self._all_keys)

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._all_keys))
        return f"Registry({self._name!r}, [{keys}])"
