"""CPU prediction geometry for modest bases and symbolic random effects."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from jaxgam.formula.operator_prepare import IndexedRandomEffect
from jaxgam.formula.predict_matrix import Data, PredictSpec


def _freeze(value, seen: set[int]) -> None:
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        for child in value.values():
            _freeze(child, seen)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _freeze(child, seen)
    elif hasattr(value, "__dict__"):
        for child in vars(value).values():
            _freeze(child, seen)


def _rows(data: Data) -> int:
    return len(data) if hasattr(data, "columns") else len(next(iter(data.values()), ()))


@dataclass(frozen=True)
class OperatorPredictionGeometry:
    """Frozen Phase-3 encoding without training rows, penalties or factors.

    Indexed dimensions retain declared level-product order symbolically.
    Point predictions allocate modest rows and IDs/amplitudes for one batch.
    A dense matrix is constructed only through the explicit matrix request.
    The caller preflights the returned output and this retained/row workspace.
    """

    modest: PredictSpec
    modest_columns: tuple[int, ...]
    random_effects: tuple[IndexedRandomEffect, ...]
    n_coef: int
    batch_rows: int
    raw_width: int
    metadata_bound_bytes: int

    def __post_init__(self) -> None:
        modest = copy.deepcopy(self.modest)
        _freeze(modest, set())
        object.__setattr__(self, "modest", modest)

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        _freeze(self.modest, set())

    def workspace_bound(self, n_rows: int, *, matrix_output: bool = False) -> int:
        from jaxgam.formula.operator_workspace import (
            predict_spec_evaluator_workspace_bound,
        )

        B = min(n_rows, self.batch_rows)
        variables = max((len(t.variables) for t in self.random_effects), default=0)
        row_workspace = 8 * B * (
            8 * self.raw_width + 2 * len(self.random_effects) + 16 + 2 * variables
        ) + 16 * len(self.modest_columns)
        row_workspace += predict_spec_evaluator_workspace_bound(self.modest, B)
        # Include the frozen coefficient vector and metadata. Four linear
        # output vectors cover eta/response/offset conversion and finishing.
        output_workspace = 8 * n_rows * self.n_coef if matrix_output else 32 * n_rows
        return (
            self.metadata_bound_bytes
            + 8 * self.n_coef
            + row_workspace
            + output_workspace
        )

    def _batches(self, data: Data):
        n = _rows(data)
        for start in range(0, n, self.batch_rows):
            stop = min(start + self.batch_rows, n)
            columns = self.slice_columns(data, start, stop)
            if self.modest.total_coefs:
                X = self.modest.build_predict_matrix(columns)
            else:
                X = np.empty((stop - start, 0))
            yield start, stop, X, columns

    def slice_columns(self, data: Data, start: int, stop: int) -> dict:
        """Reference only required covariates before any column conversion."""
        variables = tuple(
            dict.fromkeys(
                [
                    *self.modest.required_covariates,
                    *(name for term in self.random_effects for name in term.variables),
                ]
            )
        )
        columns = {}
        for name in variables:
            value = data[name]
            columns[name] = (
                value.iloc[start:stop] if hasattr(value, "iloc") else value[start:stop]
            )
        if not variables and self.modest.total_coefs:
            columns["_operator_row_marker"] = np.empty(stop - start)
        return columns

    def forward(self, data: Data, coefficients: np.ndarray) -> np.ndarray:
        eta = np.empty(_rows(data))
        for start, stop, X, columns in self._batches(data):
            eta[start:stop] = X @ coefficients[list(self.modest_columns)]
            for term in self.random_effects:
                ids, amplitude = term.evaluate(columns)
                eta[start:stop] += amplitude * coefficients[term.col_start + ids]
                del ids, amplitude
        return eta

    def build_predict_matrix(self, data: Data) -> np.ndarray:
        matrix = np.zeros((_rows(data), self.n_coef))
        for start, stop, X, columns in self._batches(data):
            matrix[start:stop, list(self.modest_columns)] = X
            rows = np.arange(start, stop)
            for term in self.random_effects:
                ids, amplitude = term.evaluate(columns)
                matrix[rows, term.col_start + ids] = amplitude
                del ids, amplitude
        return matrix
