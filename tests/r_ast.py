"""Validated, private R-language instrumentation for pinned mgcv oracles.

The adapter constructs R language objects through rpy2.  It never parses
generated source or changes a function in the installed mgcv namespace.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from weakref import ref

import rpy2.robjects as ro
from rpy2 import rinterface


@dataclass
class _CallbackOwner:
    callback: Callable[..., Any]


def make_r_callback(callback: Callable[..., Any]) -> Any:
    """Keep a Python callback alive for exactly its returned R wrapper's lifetime.

    rpy2 retains its external-pointer callable in an interpreter-wide registry.
    A weak trampoline prevents that registry from retaining captured R objects
    until after the embedded interpreter's Python bookkeeping is torn down.
    The caller must retain the returned wrapper while R can invoke it.
    """
    owner = _CallbackOwner(callback)
    reference = ref(owner)

    def invoke(*args: Any, **kwargs: Any) -> Any:
        active = reference()
        if active is None:
            raise RuntimeError("R callback invoked after its owner was released")
        return active.callback(*args, **kwargs)

    function = rinterface.rternalize(invoke)
    function._jaxgam_callback_owner = owner
    return function


def call(head: str | Any, *arguments: Any) -> Any:
    """Construct an unevaluated R call from object-level components."""
    if isinstance(head, str):
        head = ro.r["as.name"](head)
    return ro.r["as.call"](rinterface.ListSexpVector([head, *arguments]))


def symbol(name: str) -> Any:
    """Construct an R symbol without evaluating it."""
    return ro.r["as.name"](name)


def _is_call(value: Any) -> bool:
    return value.typeof == rinterface.RTYPES.LANGSXP


def _matches_symbol(value: Any, name: str) -> bool:
    return value.typeof == rinterface.RTYPES.SYMSXP and value.rsame(symbol(name))


def find_call_paths(
    function: Any,
    head: str,
    *,
    required_symbols: Sequence[str] = (),
) -> tuple[tuple[int, ...], ...]:
    """Find zero-based body paths with the exact call head and symbol anchors."""
    found: list[tuple[int, ...]] = []

    def has_symbol(node: Any, name: str) -> bool:
        if _matches_symbol(node, name):
            return True
        return _is_call(node) and any(has_symbol(item, name) for item in node)

    def visit(node: Any, path: tuple[int, ...]) -> None:
        if not _is_call(node):
            return
        if _matches_symbol(node[0], head) and all(
            has_symbol(node, name) for name in required_symbols
        ):
            found.append(path)
        for index, child in enumerate(node):
            if index:
                visit(child, (*path, index))

    visit(ro.r["body"](function), ())
    return tuple(found)


def _validated_node(body: Any, path: tuple[int, ...], expected_head: str) -> Any:
    node = body
    for index in path:
        if not _is_call(node) or index < 1 or index >= len(node):
            raise ValueError(f"R AST path {path!r} is absent")
        node = node[index]
    if not _is_call(node) or not _matches_symbol(node[0], expected_head):
        raise ValueError(f"R AST path {path!r} has no {expected_head!r} call")
    return node


def _replace_node(body: Any, path: tuple[int, ...], replacement: Any) -> Any:
    if not path:
        return replacement
    index = path[0]
    children = list(body)
    children[index] = _replace_node(children[index], path[1:], replacement)
    values = rinterface.ListSexpVector(children)
    environment = ro.r["new.env"](parent=ro.r["baseenv"]())
    environment[".jaxgam_ast_node"] = body
    original = ro.r["eval"](
        call("quote", call("as.list", symbol(".jaxgam_ast_node"))),
        envir=environment,
    )
    if original.names is not rinterface.NULL:
        values.names = original.names
    return ro.r["as.call"](values)


def clone_function(function: Any, *, environment: Any, body: Any | None = None) -> Any:
    """Clone a closure into a private environment without touching its source."""
    expression = call(
        "function",
        ro.r["formals"](function),
        ro.r["body"](function) if body is None else body,
    )
    private = ro.r["eval"](call("quote", expression), envir=environment)
    private._jaxgam_callback_handles = getattr(function, "_jaxgam_callback_handles", ())
    return private


def replace_call(
    function: Any,
    *,
    path: tuple[int, ...],
    expected_head: str,
    replacement: Any,
) -> Any:
    """Return a private closure with one validated body expression replaced."""
    body = ro.r["body"](function)
    _validated_node(body, path, expected_head)
    edited = _replace_node(body, path, replacement)
    return clone_function(
        function, environment=ro.r["environment"](function), body=edited
    )


def instrument_function(
    function: Any,
    *,
    path: tuple[int, ...],
    expected_head: str,
    capture_symbols: Sequence[str],
    callback: Callable[..., Any],
    when: Literal["before", "after"] = "before",
) -> Any:
    """Call Python with live R locals around an anchored expression.

    The original expression runs exactly once and retains its return value.
    The resulting closure is private; the installed namespace is untouched.
    """
    body = ro.r["body"](function)
    target = _validated_node(body, path, expected_head)
    ordinal = len(getattr(function, "_jaxgam_callback_handles", ())) // 2
    status = f".jaxgam_oracle_callback_status_{ordinal}"
    if find_call_paths(function, "<-", required_symbols=(status,)):
        raise ValueError("R AST already uses the callback status binding")

    def capture(*values: Any) -> Any:
        try:
            callback(*values)
        except Exception as exc:
            return rinterface.StrSexpVector([str(exc)])
        return rinterface.StrSexpVector([""])

    r_callback = make_r_callback(capture)
    hook = call(
        "{",
        call(
            "<-",
            symbol(status),
            call(r_callback, *(symbol(x) for x in capture_symbols)),
        ),
        call("if", call("nzchar", symbol(status)), call("stop", symbol(status))),
    )
    if when == "before":
        replacement = call("{", hook, target)
    elif when == "after":
        temp = f".jaxgam_oracle_capture_result_{ordinal}"
        if find_call_paths(function, "<-", required_symbols=(temp,)):
            raise ValueError("R AST already uses the capture result binding")
        replacement = call("{", call("<-", symbol(temp), target), hook, symbol(temp))
    else:
        raise ValueError("when must be 'before' or 'after'")
    instrumented = replace_call(
        function, path=path, expected_head=expected_head, replacement=replacement
    )
    instrumented._jaxgam_callback_handles = (
        *instrumented._jaxgam_callback_handles,
        capture,
        r_callback,
    )
    return instrumented
