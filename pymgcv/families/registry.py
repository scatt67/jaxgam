"""Family registry and dispatch.

Provides ``get_family()`` which accepts either a string name or an
existing family instance and returns a ready-to-use family object.

Design doc reference: Section 6.1
"""

from __future__ import annotations

from pymgcv.families.base import ExponentialFamily
from pymgcv.families.standard import Binomial, Gamma, Gaussian, Poisson

# Canonical name -> family class mapping
_FAMILY_REGISTRY: dict[str, type[ExponentialFamily]] = {
    "gaussian": Gaussian,
    "binomial": Binomial,
    "poisson": Poisson,
    "gamma": Gamma,
}

# Cached instances for default-link families. Reusing the same object
# ensures JAX's JIT cache (keyed by object identity) hits across fits
# with the same family, avoiding costly recompilation of jax.hessian.
_FAMILY_CACHE: dict[str, ExponentialFamily] = {}


def get_family(name_or_instance: str | ExponentialFamily) -> ExponentialFamily:
    """Look up and return a family instance.

    Parameters
    ----------
    name_or_instance : str or ExponentialFamily
        If a string, looks up the family by name (case-insensitive)
        and returns a cached instance with the default link.  Cached
        instances are shared across calls to ensure JAX JIT cache hits.
        If already an ``ExponentialFamily`` instance, returns it as-is.

    Returns
    -------
    ExponentialFamily
        A family instance ready for use.

    Raises
    ------
    KeyError
        If the name is not in the registry.
    TypeError
        If the argument is neither a string nor an ExponentialFamily.

    Examples
    --------
    >>> fam = get_family("gaussian")
    >>> fam.family_name
    'gaussian'

    >>> from pymgcv.families.standard import Poisson
    >>> fam = get_family(Poisson(link="log"))
    >>> fam.family_name
    'poisson'
    """
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

    raise TypeError(
        f"Expected a string or ExponentialFamily instance, "
        f"got {type(name_or_instance)!r}"
    )
