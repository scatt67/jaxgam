"""Object-level R AST instrumentation contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("rpy2.robjects")
from rpy2 import rinterface

from tests.r_ast import (
    call,
    clone_function,
    find_call_paths,
    instrument_function,
    make_r_callback,
    replace_call,
    symbol,
)


def _function():
    import rpy2.robjects as ro

    body = call(
        "{",
        call("<-", symbol("answer"), call("+", symbol("x"), ro.IntVector([2]))),
        symbol("answer"),
    )
    expression = call("function", ro.r["formals"](ro.r["identity"]), body)
    return ro.r["eval"](expression)


def test_private_capture_preserves_return_and_installed_function() -> None:
    original = _function()
    paths = find_call_paths(original, "<-", required_symbols=("answer",))
    assert len(paths) == 1
    seen: list[int] = []
    traced = instrument_function(
        original,
        path=paths[0],
        expected_head="<-",
        capture_symbols=("x",),
        callback=lambda x: seen.append(int(x[0])),
    )
    assert int(traced(5)[0]) == 7
    assert seen == [5]
    assert int(original(8)[0]) == 10
    assert seen == [5]


def test_validated_path_and_anchor_fail_closed() -> None:
    original = _function()
    path = find_call_paths(original, "<-", required_symbols=("answer",))[0]
    with pytest.raises(ValueError, match="has no"):
        replace_call(original, path=path, expected_head="if", replacement=symbol("x"))
    with pytest.raises(ValueError, match="absent"):
        replace_call(original, path=(999,), expected_head="<-", replacement=symbol("x"))


def test_private_callback_exception_leaves_original_callable() -> None:
    original = _function()
    path = find_call_paths(original, "<-", required_symbols=("answer",))[0]

    def fail(_x):
        raise RuntimeError("capture failed")

    traced = instrument_function(
        original,
        path=path,
        expected_head="<-",
        capture_symbols=("x",),
        callback=fail,
    )
    with pytest.raises(Exception, match="capture failed"):
        traced(4)
    assert int(original(4)[0]) == 6


def test_named_arguments_and_multiple_callback_lifetimes() -> None:
    import rpy2.robjects as ro

    arguments = rinterface.ListSexpVector(
        [symbol("paste"), symbol("x"), ro.StrVector(["!"]), ro.StrVector([""])]
    )
    arguments.names = ro.StrVector(["", "", "", "sep"])
    body = call(
        "{", call("<-", symbol("answer"), ro.r["as.call"](arguments)), symbol("answer")
    )
    original = ro.r["eval"](call("function", ro.r["formals"](ro.r["identity"]), body))
    path = find_call_paths(original, "<-", required_symbols=("answer",))[0]
    seen: list[str] = []
    first = instrument_function(
        original,
        path=path,
        expected_head="<-",
        capture_symbols=("x",),
        callback=lambda x: seen.append(str(x[0])),
        when="before",
    )
    second_path = find_call_paths(first, "paste")[0]
    second = instrument_function(
        first,
        path=second_path,
        expected_head="paste",
        capture_symbols=("x",),
        callback=lambda x: seen.append(f"after:{x[0]}"),
        when="after",
    )
    del first
    assert str(second("hi")[0]) == "hi!"
    assert seen == ["hi", "after:hi"]


def test_private_environment_resolves_only_its_own_override() -> None:
    import rpy2.robjects as ro

    private = ro.r["new.env"](parent=ro.r["baseenv"]())
    private["marker"] = ro.r["identity"]
    original = ro.r["identity"]
    cloned = clone_function(
        original, environment=private, body=call("marker", symbol("x"))
    )
    assert int(cloned(9)[0]) == 9
    assert int(original(4)[0]) == 4


def test_callback_releases_captured_objects_with_its_wrapper() -> None:
    """R's callback registry must not keep each completed oracle alive."""
    import gc
    import weakref

    import rpy2.robjects as ro

    class Captured:
        def __init__(self) -> None:
            self.value = ro.IntVector([7])

    captured = Captured()
    reference = weakref.ref(captured)

    def callback(x, payload=captured):
        return ro.r["+"](x, payload.value)

    function = make_r_callback(callback)
    del callback, captured
    gc.collect()
    assert reference() is not None
    assert int(function(ro.IntVector([3]))[0]) == 10
    del function
    gc.collect()
    assert reference() is None
