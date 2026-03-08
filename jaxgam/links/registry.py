"""Link function registry.

Provides ``link_registry`` instance and ``get_link()`` convenience
function for looking up link functions by name.

Design doc reference: §7
"""

from __future__ import annotations

from jaxgam.links.links import (
    CloglogLink,
    IdentityLink,
    InverseLink,
    InverseSquaredLink,
    Link,
    LogitLink,
    LogLink,
    ProbitLink,
    SqrtLink,
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

    Parameters
    ----------
    name : str
        Link name (e.g. "logit", "log", "identity").

    Returns
    -------
    Link
        An instance of the corresponding link class.
    """
    return link_registry.get_instance(name)
