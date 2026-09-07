"""Adapters from prepared metadata to bounded or dense design consumers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jaxgam.data.source import RowBatch, RowSource
from jaxgam.formula.prepare import PreparedModel


@dataclass(frozen=True)
class StreamDesign:
    """Prepared metadata plus replayable rows; evaluates only one batch."""

    prepared: PreparedModel
    source: RowSource

    def batches(
        self, batch_rows: int
    ) -> Iterator[tuple[npt.NDArray[np.floating], RowBatch]]:
        if self.source.fingerprint() != self.prepared.source_fingerprint:
            raise RuntimeError("RowSource changed after preparation; prepare again.")
        for batch in self.source.scan(batch_rows):
            yield self.prepared.evaluate_batch(batch), batch


@dataclass(frozen=True)
class DenseDesign:
    """Explicit compatibility materialization for an existing dense backend."""

    prepared: PreparedModel
    X: npt.NDArray[np.floating]

    @classmethod
    def materialize(cls, stream: StreamDesign, batch_rows: int) -> DenseDesign:
        return cls(
            stream.prepared, np.vstack([X for X, _ in stream.batches(batch_rows)])
        )
