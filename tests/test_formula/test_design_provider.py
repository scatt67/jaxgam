"""Prepared design-provider adapters."""

from __future__ import annotations

import numpy as np

from jaxgam.data.source import ArrayRowSource
from jaxgam.formula.design_provider import DenseDesign, StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model


def test_stream_design_checks_replay_before_yielding_batches() -> None:
    x = np.linspace(0.0, 1.0, 19)
    source = ArrayRowSource({"x": x}, y=x)
    prepared = prepare_model(parse_formula('y ~ s(x, bs="cr", k=6)'), source)
    stream = StreamDesign(prepared, source)
    batches = list(stream.batches(7))
    assert [X.shape for X, _ in batches] == [(7, 6), (7, 6), (5, 6)]
    dense = DenseDesign.materialize(stream, 8)
    assert dense.X.shape == (19, 6)
