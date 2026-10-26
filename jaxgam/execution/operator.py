"""CPU host orchestration for frozen operator systems and PCG.

Krylov vector arithmetic is pure JIT code. This module owns source I/O,
memory preflight, finite status handling and true-residual verification.
It does not select a family, replace observed curvature, or provide score,
EDF or uncertainty metadata.
"""

from __future__ import annotations

import copy
import hashlib
import numbers
from collections.abc import Callable
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from jaxgam.data.source import RowBatch, RowSource
from jaxgam.fitting.data import _to_jax_structure
from jaxgam.fitting.iterative import (
    CONVERGED,
    MAX_ITERATIONS,
    NONFINITE,
    READY,
    VERIFY_RESIDUAL,
    PCGState,
    advance_pcg,
    initialize_pcg,
    verify_pcg,
)
from jaxgam.fitting.operators import (
    DeviceIndexedRows,
    DeviceOperatorBatch,
    accumulate_normal_matvec_batch,
    penalty_matvec,
    to_device_operator_batch,
)
from jaxgam.fitting.preconditioners import (
    PreparedPreconditioner,
    apply_preconditioner,
    prepare_preconditioner,
)
from jaxgam.formula.operator_prepare import (
    OperatorBatch,
    OperatorPreparationControl,
    PreparedOperatorModel,
    _CheckedSource,
)
from jaxgam.penalties.structure import IdentityPenalty

_MAX_SIGNED_DOWNDATE_RANK = 8


def _signed_row_dot(
    first: tuple[np.ndarray, tuple[tuple[int, float], ...]],
    second: tuple[np.ndarray, tuple[tuple[int, float], ...]],
) -> float:
    """Inner product of two bounded sparse negative design rows."""
    first_modest, first_indexed = first
    second_modest, second_indexed = second
    result = float(np.dot(first_modest, second_modest))
    second_lookup = dict(second_indexed)
    return result + sum(
        amplitude * second_lookup.get(column, 0.0)
        for column, amplitude in first_indexed
    )


def _signed_exact_span_coordinates(
    row: tuple[np.ndarray, tuple[tuple[int, float], ...]],
    basis: list[tuple[np.ndarray, tuple[tuple[int, float], ...]]],
    gram: np.ndarray,
) -> np.ndarray | None:
    """Compress only an algebraically exact small-integer row dependency.

    The integer coefficients cover indicator crossed effects without treating
    a nearly dependent numeric design as rank deficient. Reconstruction is
    checked in the original source coordinates before dropping a direction.
    """
    if not basis:
        return None
    cross = np.array([_signed_row_dot(prior, row) for prior in basis])
    try:
        candidate = np.linalg.solve(gram, cross)
    except np.linalg.LinAlgError:
        return None
    rounded = np.rint(candidate)
    if np.any(np.abs(candidate - rounded) > 1e-10):
        return None
    coefficients = rounded.astype(np.float64)
    modest, indexed = row
    reconstructed_modest = np.zeros_like(modest)
    reconstructed_indexed: dict[int, float] = {}
    for coefficient, (prior_modest, prior_indexed) in zip(
        coefficients, basis, strict=True
    ):
        reconstructed_modest += coefficient * prior_modest
        for column, amplitude in prior_indexed:
            reconstructed_indexed[column] = (
                reconstructed_indexed.get(column, 0.0) + coefficient * amplitude
            )
    if not np.array_equal(reconstructed_modest, modest):
        return None
    if any(
        reconstructed_indexed.get(column, 0.0) != amplitude
        for column, amplitude in indexed
    ):
        return None
    if any(
        value != 0.0
        for column, value in reconstructed_indexed.items()
        if column not in dict(indexed)
    ):
        return None
    return coefficients


@dataclass(frozen=True)
class PCGControl:
    rtol: float = 1e-10
    atol: float = 0.0
    max_iterations: int = 1000

    def __post_init__(self) -> None:
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.rtol == 0 and self.atol == 0:
            raise ValueError("PCG needs a positive residual tolerance.")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, numbers.Integral)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer.")


@dataclass(frozen=True)
class HostPCGResult:
    solution: np.ndarray
    status: int
    iterations: int
    matvec_count: int
    absolute_residual: float
    relative_residual: float
    vector_peak_bound_bytes: int
    executable_temporary_bytes: int
    additional_peak_bound_bytes: int

    def __post_init__(self) -> None:
        owned = self.solution.copy()
        owned.setflags(write=False)
        object.__setattr__(self, "solution", owned)

    def __reduce__(self):
        return type(self), tuple(vars(self).values())

    @property
    def converged(self) -> bool:
        return self.status == CONVERGED


_DEFAULT_PCG_CONTROL = PCGControl()


@dataclass
class ScanDiagnostics:
    started_scans: int = 0
    completed_scans: int = 0
    yielded_batches: int = 0
    yielded_rows: int = 0
    fingerprint_checks: int = 0
    normal_actions: int = 0


class _CountedSource:
    def __init__(self, source: RowSource, diagnostics: ScanDiagnostics):
        self.source = source
        self.diagnostics = diagnostics

    @property
    def n_rows(self):
        return self.source.n_rows

    def fingerprint(self):
        self.diagnostics.fingerprint_checks += 1
        return self.source.fingerprint()

    def scan(self, rows):
        self.diagnostics.started_scans += 1
        for batch in self.source.scan(rows):
            self.diagnostics.yielded_batches += 1
            self.diagnostics.yielded_rows += len(batch.row_positions)
            yield batch
        self.diagnostics.completed_scans += 1


def _padded_geometry(rows: OperatorBatch, size: int) -> OperatorBatch:
    n, width = rows.modest_values.shape
    if n == size:
        return rows
    matrix = np.zeros((size, width))
    matrix[:n] = rows.modest_values
    terms = []
    for ids, amplitude, start, count in rows.indexed_rows:
        padded_ids = np.zeros(size, dtype=np.int64)
        padded_amplitude = np.zeros(size)
        padded_ids[:n], padded_amplitude[:n] = ids, amplitude
        terms.append((padded_ids, padded_amplitude, start, count))
    return OperatorBatch(matrix, rows.modest_columns, tuple(terms), rows.n_coef)


def _freeze_host_arrays(value, visited: set[int]) -> None:
    if id(value) in visited:
        return
    visited.add(id(value))
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        for child in value.values():
            _freeze_host_arrays(child, visited)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _freeze_host_arrays(child, visited)
    elif hasattr(value, "__dict__"):
        for child in vars(value).values():
            _freeze_host_arrays(child, visited)


class FrozenScanSystem:
    """A source-replayed X'WX+S system with a sufficient SPD certificate.

    The certificate uses the actual signed row curvature. Positive rows and
    full-rank indexed penalties establish a block lower bound; negative rows
    are admitted only when their leverage against that bound is below one.
    A failed sufficient bound does not establish that the total H is indefinite.

    A supplied curvature policy must be pure and own its frozen working state.
    Its output is replay-hashed on every complete action, independently of
    source content. No family-name dispatch or Fisher replacement occurs here.
    Caller-owned source backing, batches, policy storage and native compiler
    RSS are excluded; the caller must charge policy-owned state separately.
    """

    def __init__(
        self,
        model: PreparedOperatorModel,
        source: RowSource,
        log_sp: np.ndarray,
        *,
        curvature: Callable[[RowBatch], np.ndarray] | None = None,
        curvature_from_rows: Callable[[RowBatch, OperatorBatch], np.ndarray]
        | None = None,
        memory_budget_bytes: int,
        use_coarse: bool = False,
    ) -> None:
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, numbers.Integral)
            or memory_budget_bytes < 1
        ):
            raise ValueError("memory_budget_bytes must be a positive integer.")
        if (
            not isinstance(log_sp, np.ndarray)
            or log_sp.dtype != np.float64
            or log_sp.shape != (model.penalties.n_penalties,)
        ):
            raise ValueError("Frozen log-sp must match the float64 model penalties.")
        if not isinstance(source, RowSource) or source.n_rows != model.n_obs:
            raise ValueError(
                "Frozen source must match the prepared selected row count."
            )
        if type(use_coarse) is not bool:
            raise TypeError("use_coarse must be boolean.")
        if curvature is not None and curvature_from_rows is not None:
            raise ValueError("Choose one actual row curvature policy.")
        if any(
            policy is not None and not callable(policy)
            for policy in (curvature, curvature_from_rows)
        ):
            raise TypeError("Actual row curvature policy must be callable.")
        self.curvature_from_rows = curvature_from_rows
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.diagnostics = ScanDiagnostics()
        self.source = _CountedSource(source, self.diagnostics)
        self.curvature = curvature
        self.batch_rows = model.batch_rows
        self._solver_reserve = 0
        K, p, B = len(model.modest_columns), model.n_coef, model.batch_rows
        self.row_peak_bound_bytes = (
            model.batch_workspace_bound(B)
            + 32 * B * (K + 2 * len(model.random_effects) + 8)
            + 16 * K
        )
        # Caller statistics diagonal (8p) plus preconditioner ownership and
        # validation (17p), plus one optional indexed Schur coupling vector.
        # Signed positive/negative Gram statistics and their factorization
        # coexist with the preconditioner's capped small-block construction.
        construction = (
            model.metadata_bound_bytes
            + self.row_peak_bound_bytes
            + 65 * p
            + 288 * K**2
            + (32 * p * K if model.random_effects else 0)
            + 512 * K
            + 256 * _MAX_SIGNED_DOWNDATE_RANK * (K + len(model.random_effects) + 2)
            + 4096
            + 2 * 64 * model.modest_raw_width**2
        )
        self._check_memory(construction)
        # Snapshot capped prediction transforms and local penalty arrays.
        # Immutable factor lookup/level metadata remains shared; no p-sized
        # identity, constraint or range arrays are introduced by this copy.
        modest = copy.deepcopy(model.modest)
        penalties = copy.deepcopy(model.penalties)
        _freeze_host_arrays(modest, set())
        _freeze_host_arrays(penalties, set())
        model = replace(model, modest=modest, penalties=penalties)
        self.model = model
        self._required_variables = tuple(
            dict.fromkeys(
                [
                    *modest.predict_spec.required_covariates,
                    *(name for term in model.random_effects for name in term.variables),
                ]
            )
        )
        if not np.all(np.isfinite(log_sp)) or np.any(
            log_sp > np.log(np.finfo(float).max)
        ):
            raise ValueError("Frozen penalties must have finite positive weights.")
        sp = np.exp(log_sp)
        if not np.all(np.isfinite(sp)) or np.any(sp <= 0):
            raise ValueError("Frozen penalties must have finite positive weights.")
        if self.source.fingerprint() != model.source_fingerprint:
            raise RuntimeError("Frozen source fingerprint differs from preparation.")
        control = OperatorPreparationControl(
            B, self.memory_budget_bytes, max(1, model.modest_raw_width)
        )
        self._checked = _CheckedSource(self.source, control, construction)
        self._checked.digest = model.source_replay_digest
        locations = {
            global_col: local for local, global_col in enumerate(model.modest_columns)
        }
        modest_penalty = np.zeros((K, K))
        structural_penalty = np.zeros((K, K))
        diagonal = np.zeros(p)
        indexed_penalty_weights = {}
        for block in model.penalties.blocks:
            if block.start in locations:
                local = [locations[col] for col in range(block.start, block.stop)]
                if len(local) > model.modest_raw_width:
                    raise MemoryError(
                        "Penalty block exceeds the independent modest cap."
                    )
                for penalty, index in zip(
                    block.local_penalties, block.sp_indices, strict=True
                ):
                    matrix = penalty.dense()
                    local_eigenvalues = np.linalg.eigvalsh(matrix)
                    tolerance = (
                        max(1.0, np.max(np.abs(local_eigenvalues), initial=0.0)) * 1e-12
                    )
                    if np.any(local_eigenvalues < -tolerance):
                        raise ValueError(
                            "Each modest penalty must be positive semidefinite."
                        )
                    structural_penalty[np.ix_(local, local)] += matrix
                    modest_penalty[np.ix_(local, local)] += sp[index] * matrix
            else:
                if (
                    len(block.local_penalties) != 1
                    or not isinstance(block.local_penalties[0], IdentityPenalty)
                    or block.ranks != (block.size,)
                ):
                    raise NotImplementedError(
                        "Large blocks require symbolic full-rank identity penalties."
                    )
                if block.local_penalties[0].scale <= 0:
                    raise ValueError("Indexed penalties must be strictly positive.")
                diagonal[block.start : block.stop] = (
                    sp[block.sp_indices[0]] * block.local_penalties[0].scale
                )
                indexed_penalty_weights[block.start] = diagonal[block.start]
        if not np.all(np.isfinite(modest_penalty)) or not np.all(np.isfinite(diagonal)):
            raise ValueError("Frozen weighted penalty overflowed.")
        eigenvalues, eigenvectors = np.linalg.eigh(structural_penalty)
        cutoff = max(1.0, np.max(np.abs(eigenvalues), initial=0.0)) * 1e-12
        if np.any(eigenvalues < -cutoff):
            raise ValueError("The modest penalty is not positive semidefinite.")
        null = eigenvectors[:, eigenvalues <= cutoff]
        gram = np.zeros((K, K))
        positive_gram = np.zeros((K, K))
        negative_gram = np.zeros((K, K))
        digest = hashlib.sha256()
        re_relative_trace = 0.0
        negative_re_relative_trace = 0.0
        has_negative_rows = False
        negative_direction_overflow = False
        negative_basis: list[tuple[np.ndarray, tuple[tuple[int, float], ...]]] = []
        negative_basis_gram = np.empty((0, 0))
        negative_covariance = np.empty((0, 0))
        self.certification_actions = 0
        self.certification_peak_bound_bytes = construction
        coupling = np.zeros((p, K)) if model.random_effects and K else None
        indexed_offdiagonal_bound = np.zeros(p)
        positive_indexed_row_sum = np.zeros(p)
        for batch in self._checked.scan(B):
            rows = self._evaluate_rows(batch)
            weight = self._weight(batch, rows)
            digest.update(weight.tobytes())
            X = rows.modest_values
            gram += X.T @ (weight[:, None] * X)
            positive = np.maximum(weight, 0.0)
            negative = np.maximum(-weight, 0.0)
            if np.any(negative):
                has_negative_rows = True
            positive_gram += X.T @ (positive[:, None] * X)
            negative_gram += X.T @ (negative[:, None] * X)
            diagonal[list(model.modest_columns)] += np.sum(
                weight[:, None] * X**2, axis=0
            )
            total_indexed_amplitude = sum(
                (np.abs(amplitude) for _, amplitude, _, _ in rows.indexed_rows),
                np.zeros(len(weight)),
            )
            total_normalized_amplitude = sum(
                (
                    np.abs(amplitude) / np.sqrt(indexed_penalty_weights[start])
                    for _, amplitude, start, _ in rows.indexed_rows
                ),
                np.zeros(len(weight)),
            )
            for ids, amplitude, start, count in rows.indexed_rows:
                contributions = weight * amplitude**2
                penalty_weight = indexed_penalty_weights[start]
                if not np.isfinite(penalty_weight) or penalty_weight <= 0:
                    raise ValueError("Indexed weighted penalties must be positive.")
                re_relative_trace += np.sum(positive * amplitude**2) / penalty_weight
                negative_re_relative_trace += (
                    np.sum(negative * amplitude**2) / penalty_weight
                )
                np.add.at(diagonal[start : start + count], ids, contributions)
                np.add.at(
                    indexed_offdiagonal_bound[start : start + count],
                    ids,
                    np.abs(weight * amplitude)
                    * (total_indexed_amplitude - np.abs(amplitude)),
                )
                np.add.at(
                    positive_indexed_row_sum[start : start + count],
                    ids,
                    positive
                    * np.abs(amplitude)
                    / np.sqrt(penalty_weight)
                    * total_normalized_amplitude,
                )
                if coupling is not None:
                    np.add.at(
                        coupling[start : start + count],
                        ids,
                        (weight * amplitude)[:, None] * X,
                    )
                del contributions
            for row_index in np.flatnonzero(negative):
                indexed = tuple(
                    (
                        start + int(ids[row_index]),
                        float(amplitude[row_index]),
                    )
                    for ids, amplitude, start, _ in rows.indexed_rows
                )
                if negative_direction_overflow:
                    continue
                row = (X[row_index].copy(), indexed)
                coordinates = _signed_exact_span_coordinates(
                    row, negative_basis, negative_basis_gram
                )
                if coordinates is not None:
                    negative_covariance += negative[row_index] * np.outer(
                        coordinates, coordinates
                    )
                elif len(negative_basis) < _MAX_SIGNED_DOWNDATE_RANK:
                    cross = np.array(
                        [_signed_row_dot(prior, row) for prior in negative_basis]
                    )
                    new_gram = np.empty((len(negative_basis) + 1,) * 2)
                    new_gram[:-1, :-1] = negative_basis_gram
                    new_gram[-1, :-1] = cross
                    new_gram[:-1, -1] = cross
                    new_gram[-1, -1] = _signed_row_dot(row, row)
                    new_covariance = np.zeros_like(new_gram)
                    new_covariance[:-1, :-1] = negative_covariance
                    new_covariance[-1, -1] = negative[row_index]
                    negative_basis.append(row)
                    negative_basis_gram = new_gram
                    negative_covariance = new_covariance
                else:
                    negative_direction_overflow = True
            del rows, X, weight, batch
        self._curvature_digest = digest.hexdigest()
        self.certification_rank = len(negative_basis)
        # Eliminate the indexed coordinates analytically for a conservative
        # Schur lower bound. If C is the weighted indexed design and Lambda
        # its positive diagonal penalty, C Lambda^-1 C.T has spectral radius
        # no larger than its trace. Thus the modest Schur complement is at
        # least S_modest + Gram / (1 + trace). This tests the actual weighted
        # penalty, including amplified numerical negative eigenvalues, without
        # clipping or assuming a tolerance-based PSD test proves SPD.
        if not np.isfinite(re_relative_trace):
            raise np.linalg.LinAlgError("Indexed Schur certificate overflowed.")
        indexed_columns = (
            np.concatenate(
                [
                    np.arange(term.col_start, term.col_start + term.n_coef)
                    for term in model.random_effects
                ]
            )
            if model.random_effects
            else np.empty(0, dtype=np.int64)
        )
        indexed_lower = (
            diagonal[indexed_columns] - indexed_offdiagonal_bound[indexed_columns]
        )
        diagonally_dominant_indexed = bool(
            np.all(np.isfinite(indexed_lower) & (indexed_lower > 0))
        )
        certificate = None
        if has_negative_rows and model.random_effects and diagonally_dominant_indexed:
            if K:
                schur = (
                    modest_penalty
                    + gram
                    - coupling[indexed_columns].T
                    @ (coupling[indexed_columns] / indexed_lower[:, None])
                )
                if np.all(np.isfinite(schur)):
                    try:
                        np.linalg.cholesky((schur + schur.T) * 0.5)
                    except np.linalg.LinAlgError:
                        pass
                    else:
                        certificate = modest_penalty + gram
            else:
                certificate = modest_penalty + gram
        if has_negative_rows and not model.random_effects:
            certificate = modest_penalty + gram
        elif has_negative_rows and certificate is None:
            positive_re_trace = min(
                re_relative_trace,
                np.max(positive_indexed_row_sum, initial=0.0),
            )
            if not np.isfinite(positive_re_trace):
                raise np.linalg.LinAlgError("Signed Schur certificate overflowed.")
            # Positive data are bounded below by diag(S+G/(2+t), Lambda/(1+t)),
            # where t bounds the indexed positive Gram relative to Lambda.
            # A rank-one negative row consumes at most its leverage in this
            # block metric; total leverage below one proves H positive.
            lower_modest = modest_penalty + positive_gram / (2.0 + positive_re_trace)
            conservative_certified = False
            try:
                lower_factor = (
                    np.linalg.cholesky((lower_modest + lower_modest.T) * 0.5)
                    if K
                    else None
                )
                modest_leverage = (
                    np.trace(
                        np.linalg.solve(
                            lower_factor.T,
                            np.linalg.solve(lower_factor, negative_gram),
                        )
                    )
                    if K
                    else 0.0
                )
                negative_leverage = (
                    modest_leverage
                    + (1.0 + positive_re_trace) * negative_re_relative_trace
                )
                conservative_certified = bool(
                    np.isfinite(negative_leverage) and negative_leverage < 1.0
                )
            except np.linalg.LinAlgError:
                pass
            if conservative_certified:
                certificate = lower_modest
            elif not negative_direction_overflow and self._certify_sparse_negative_rows(
                model=model,
                log_sp=log_sp,
                construction=construction,
                modest_penalty=modest_penalty,
                positive_gram=positive_gram,
                positive_re_bound=positive_re_trace,
                indexed_penalty_weights=indexed_penalty_weights,
                basis=negative_basis,
                basis_gram=negative_basis_gram,
                covariance=negative_covariance,
            ):
                certificate = modest_penalty + gram
            else:
                raise np.linalg.LinAlgError(
                    "Actual signed system has no sufficient Schur SPD certificate."
                )
        elif certificate is None:
            certificate = modest_penalty + gram / (1.0 + re_relative_trace)
        if K:
            try:
                np.linalg.cholesky((certificate + certificate.T) * 0.5)
            except np.linalg.LinAlgError as error:
                raise np.linalg.LinAlgError(
                    "Actual weighted system has no sufficient Schur SPD certificate."
                ) from error
        del certificate
        coarse_gram = null.T @ gram @ null
        if null.shape[1]:
            try:
                np.linalg.cholesky((coarse_gram + coarse_gram.T) * 0.5)
            except np.linalg.LinAlgError as error:
                raise np.linalg.LinAlgError(
                    "Data curvature is not SPD on modest penalty-null directions."
                ) from error
        diagonal[list(model.modest_columns)] += np.diag(modest_penalty)
        modest_hessian = gram + modest_penalty
        columns = np.asarray(model.modest_columns, dtype=np.int64)
        blocks = ((columns, modest_hessian),) if K else ()
        coarse = (columns, null, coarse_gram) if use_coarse and null.shape[1] else None
        # The signed construction statistics have no role in the retained
        # preconditioner. Release their p-vectors before its separately
        # budgeted diagonal and validation arrays are allocated.
        del indexed_offdiagonal_bound, positive_indexed_row_sum
        del indexed_columns, indexed_lower, negative_basis
        del negative_basis_gram, negative_covariance
        del coupling, positive_gram, negative_gram
        self.preconditioner = prepare_preconditioner(
            diagonal,
            blocks=blocks,
            coarse=coarse,
            max_modest_coefficients=max(1, model.modest_raw_width),
            memory_budget_bytes=self.memory_budget_bytes
            - model.metadata_bound_bytes
            - self.row_peak_bound_bytes
            - 8 * p,
            device=jax.devices("cpu")[0],
        )
        self._penalties = _to_jax_structure(model.penalties, jax.devices("cpu")[0])
        rho_copy = log_sp.copy()
        rho_copy.setflags(write=False)
        self._rho = jax.device_put(rho_copy, jax.devices("cpu")[0])
        penalty_bytes = sum(
            leaf.nbytes for leaf in jax.tree_util.tree_leaves(self._penalties)
        )
        self.retained_bound_bytes = (
            model.metadata_bound_bytes
            + 2 * self.preconditioner.retained_device_bytes
            + 2 * penalty_bytes
            + 16 * len(log_sp)
            + 4096
            + 512 * len(self._checked.schemas)
            + 2 * 64 * model.modest_raw_width**2
        )
        vector = jax.ShapeDtypeStruct((p,), np.float64)
        abstract_rows = DeviceOperatorBatch(
            jax.ShapeDtypeStruct((B, K), np.float64),
            jax.ShapeDtypeStruct((K,), np.int64),
            tuple(
                DeviceIndexedRows(
                    jax.ShapeDtypeStruct((B,), np.int64),
                    jax.ShapeDtypeStruct((B,), np.float64),
                    term.col_start,
                    term.n_coef,
                )
                for term in model.random_effects
            ),
            p,
        )
        self._normal = accumulate_normal_matvec_batch.lower(
            vector, abstract_rows, vector, jax.ShapeDtypeStruct((B,), np.float64)
        ).compile()
        self._penalty = penalty_matvec.lower(
            self._penalties, vector, self._rho
        ).compile()
        self.executable_temporary_bytes = max(
            self._normal.memory_analysis().temp_size_in_bytes,
            self._penalty.memory_analysis().temp_size_in_bytes,
        )
        self.donated_accumulator_bytes = (
            self._normal.memory_analysis().alias_size_in_bytes
        )
        self.action_peak_bound_bytes = (
            self.retained_bound_bytes
            + self.row_peak_bound_bytes
            + 8 * p
            + self.executable_temporary_bytes
        )
        self._check_memory(self.action_peak_bound_bytes)
        self.construction_peak_bound_bytes = max(
            construction,
            self.certification_peak_bound_bytes,
            self.action_peak_bound_bytes,
            self._checked.peak_bound_bytes,
        )
        self._checked.reserved_bytes = self.action_peak_bound_bytes

    def _certify_sparse_negative_rows(
        self,
        *,
        model: PreparedOperatorModel,
        log_sp: np.ndarray,
        construction: int,
        modest_penalty: np.ndarray,
        positive_gram: np.ndarray,
        positive_re_bound: float,
        indexed_penalty_weights: dict[int, float],
        basis: list[tuple[np.ndarray, tuple[tuple[int, float], ...]]],
        basis_gram: np.ndarray,
        covariance: np.ndarray,
    ) -> bool:
        """Certify a capped signed downdate using positive solves and errors.

        For H = H_pos - U U', H is SPD exactly when I-U' H_pos^-1 U is
        SPD. Each positive PCG solve is checked by a fresh true residual;
        a proved lower eigenvalue for H_pos bounds the resulting Gram error.
        The signed operator itself is never replaced by H_pos.
        """
        p, K = model.n_coef, len(model.modest_columns)
        rank = len(basis)
        if rank == 0 or rank > _MAX_SIGNED_DOWNDATE_RANK:
            return False
        lower_modest = modest_penalty + positive_gram / (2.0 + positive_re_bound)
        modest_lower = (
            float(np.linalg.eigvalsh((lower_modest + lower_modest.T) * 0.5)[0])
            if K
            else np.inf
        )
        indexed_lower = min(indexed_penalty_weights.values()) / (
            1.0 + positive_re_bound
        )
        eigen_lower = 0.5 * min(modest_lower, indexed_lower)
        if not np.isfinite(eigen_lower) or eigen_lower <= 0:
            return False
        # Reserve the outer setup, one rhs/solution/residual, compact capped
        # negative rows, and rank-square Gram/error arrays before replay.
        negative_workspace = 64 * p + 128 * rank * (K + len(model.random_effects))
        negative_workspace += 64 * rank * rank + 4096
        positive_budget = self.memory_budget_bytes - construction - negative_workspace
        if positive_budget <= 0:
            raise MemoryError("Signed downdate certification exceeds its budget.")

        def positive_curvature(batch: RowBatch, rows: OperatorBatch) -> np.ndarray:
            return np.maximum(self._weight(batch, rows), 0.0)

        positive_system = FrozenScanSystem(
            model,
            self.source,
            log_sp,
            curvature_from_rows=positive_curvature,
            memory_budget_bytes=positive_budget,
        )
        self.certification_peak_bound_bytes = max(
            self.certification_peak_bound_bytes,
            construction
            + negative_workspace
            + positive_system.construction_peak_bound_bytes,
        )
        try:
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            return False
        columns = np.asarray(model.modest_columns, dtype=np.int64)
        norms = np.sqrt(np.maximum(np.diag(factor.T @ basis_gram @ factor), 0.0))
        gram = np.zeros((rank, rank))
        error = np.zeros((rank, rank))
        cpu = jax.devices("cpu")[0]
        control = PCGControl(
            rtol=1e-13,
            atol=1e-14,
            max_iterations=max(64, min(4 * p, 1000)),
        )
        for column in range(rank):
            rhs = np.zeros(p)
            for scale, (modest, indexed) in zip(factor[:, column], basis, strict=True):
                rhs[columns] += scale * modest
                for coefficient, amplitude in indexed:
                    rhs[coefficient] += scale * amplitude
            solved = positive_system.solve(rhs, control=control)
            self.certification_peak_bound_bytes = max(
                self.certification_peak_bound_bytes,
                construction
                + negative_workspace
                + positive_system.action_peak_bound_bytes
                + solved.additional_peak_bound_bytes,
            )
            self.certification_actions += solved.matvec_count
            self.diagnostics.normal_actions += solved.matvec_count
            if not solved.converged:
                return False
            solution = solved.solution
            residual = rhs - np.asarray(
                positive_system.matvec(jax.device_put(solution, cpu))
            )
            self.certification_actions += 1
            self.diagnostics.normal_actions += 1
            residual_norm = float(np.linalg.norm(residual))
            if not np.isfinite(residual_norm):
                return False
            basis_action = np.array(
                [
                    np.dot(modest, solution[columns])
                    + sum(
                        amplitude * solution[coefficient]
                        for coefficient, amplitude in indexed
                    )
                    for modest, indexed in basis
                ]
            )
            gram[:, column] = factor.T @ basis_action
            for row in range(rank):
                error[row, column] = norms[row] * residual_norm / eigen_lower
        signed_digest = hashlib.sha256()
        for batch in self._checked.scan(self.batch_rows):
            rows = self._evaluate_rows(batch)
            signed_digest.update(self._weight(batch, rows).tobytes())
            del rows, batch
        if signed_digest.hexdigest() != self._curvature_digest:
            raise RuntimeError("Frozen row curvature changed during certification.")
        upper = float(np.linalg.eigvalsh((gram + gram.T) * 0.5)[-1])
        upper += float(np.linalg.norm(error, ord="fro"))
        upper += 64 * np.finfo(float).eps * max(1.0, abs(upper))
        return np.isfinite(upper) and upper < 1.0

    def _check_memory(self, required: int) -> None:
        if required > self.memory_budget_bytes:
            raise MemoryError(
                f"Frozen operator requires {required} bytes; "
                f"budget is {self.memory_budget_bytes}."
            )

    def _weight(self, batch: RowBatch, rows: OperatorBatch) -> np.ndarray:
        if self.curvature_from_rows is not None:
            values = self.curvature_from_rows(batch, rows)
        else:
            values = batch.weight if self.curvature is None else self.curvature(batch)
        weight = np.asarray(values, dtype=np.float64)
        if weight.shape != (len(batch.row_positions),) or not np.all(
            np.isfinite(weight)
        ):
            raise ValueError("Actual row curvature must be a finite row vector.")
        return weight

    def _evaluate_rows(self, batch: RowBatch) -> OperatorBatch:
        subset = replace(
            batch,
            columns={name: batch.columns[name] for name in self._required_variables},
        )
        if not self._required_variables and self.model.modest.n_coef == 0:
            return OperatorBatch(
                np.empty((len(batch.row_positions), 0)), (), (), self.model.n_coef
            )
        return self.model.evaluate_batch(
            subset, available_workspace_bytes=self.row_peak_bound_bytes
        )

    def reserve_solver_workspace(self, required: int) -> None:
        total = self.action_peak_bound_bytes + required
        self._check_memory(total)
        self._solver_reserve = required
        self._checked.reserved_bytes = total

    def matvec(self, vector: jax.Array) -> jax.Array:
        """One complete frozen normal action with a donated p-accumulator."""
        if (
            vector.shape != (self.model.n_coef,)
            or vector.dtype != jnp.float64
            or vector.device.platform != "cpu"
        ):
            raise ValueError("Frozen vector must match the CPU float64 coefficients.")
        self._check_memory(self.action_peak_bound_bytes + self._solver_reserve)
        if self.source.fingerprint() != self.model.source_fingerprint:
            raise RuntimeError("Frozen source changed after preparation.")
        accumulator = self._penalty(self._penalties, vector, self._rho)
        digest = hashlib.sha256()
        for batch in self._checked.scan(self.batch_rows):
            rows = self._evaluate_rows(batch)
            weight = self._weight(batch, rows)
            digest.update(weight.tobytes())
            padded = _padded_geometry(rows, self.batch_rows)
            device_rows = to_device_operator_batch(
                padded,
                memory_budget_bytes=self.row_peak_bound_bytes,
                device=jax.devices("cpu")[0],
            )
            padded_weight = np.zeros(self.batch_rows)
            padded_weight[: len(weight)] = weight
            accumulator = self._normal(
                accumulator,
                device_rows,
                vector,
                jax.device_put(padded_weight, jax.devices("cpu")[0]),
            )
            accumulator.block_until_ready()
            del rows, padded, device_rows, padded_weight, weight, batch
        if digest.hexdigest() != self._curvature_digest:
            raise RuntimeError("Frozen row curvature changed during linear solve.")
        self.diagnostics.normal_actions += 1
        return accumulator

    def solve(
        self,
        rhs: np.ndarray,
        *,
        initial: np.ndarray | None = None,
        control: PCGControl = _DEFAULT_PCG_CONTROL,
    ) -> HostPCGResult:
        try:
            return solve_host_pcg(
                self.matvec,
                self.preconditioner,
                rhs,
                initial=initial,
                control=control,
                available_memory_bytes=self.memory_budget_bytes
                - self.action_peak_bound_bytes,
                reserve_workspace=self.reserve_solver_workspace,
            )
        finally:
            self._solver_reserve = 0
            self._checked.reserved_bytes = self.action_peak_bound_bytes

    def solve_adjoint(
        self, rhs: np.ndarray, *, control: PCGControl = _DEFAULT_PCG_CONTROL
    ) -> HostPCGResult:
        """Explicit host adjoint solve using the same frozen symmetric system."""
        return self.solve(rhs, control=control)


def solve_host_pcg(
    matvec: Callable[[jax.Array], jax.Array],
    preconditioner: PreparedPreconditioner,
    rhs: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    control: PCGControl = _DEFAULT_PCG_CONTROL,
    available_memory_bytes: int,
    reserve_workspace: Callable[[int], None] | None = None,
) -> HostPCGResult:
    """Solve a proven SPD frozen system without differentiating host I/O.

    The available budget excludes existing operator/preconditioner ownership.
    ``reserve_workspace`` lets a scanning system combine this solve's bound
    with its retained and row-action storage before any coefficient allocation.
    The caller supplies the positive-system certificate; sampled positive CG
    curvature alone does not certify the total matrix. Every invocation of
    ``matvec``, including verification and failure diagnostics, is counted.
    """
    if not isinstance(control, PCGControl):
        raise TypeError("control must be PCGControl.")
    if (
        isinstance(available_memory_bytes, bool)
        or not isinstance(available_memory_bytes, numbers.Integral)
        or available_memory_bytes < 1
    ):
        raise ValueError("available_memory_bytes must be a positive integer.")
    if (
        not isinstance(rhs, np.ndarray)
        or rhs.ndim != 1
        or rhs.dtype != np.float64
        or not len(rhs)
    ):
        raise ValueError("PCG rhs must be a nonempty float64 vector.")
    if initial is not None and (
        not isinstance(initial, np.ndarray)
        or initial.shape != rhs.shape
        or initial.dtype != np.float64
    ):
        raise ValueError("PCG initial coefficients must match the float64 rhs.")
    if preconditioner.inverse.diagonal.shape != rhs.shape:
        raise ValueError("PCG preconditioner must match the rhs dimension.")
    if any(
        leaf.device.platform != "cpu"
        for leaf in jax.tree_util.tree_leaves(preconditioner.inverse)
    ):
        raise ValueError("Host PCG requires a CPU-owned preconditioner.")
    p = len(rhs)
    # Four rhs/initial host-device ownership buffers, four old and four new
    # state vectors, one operator action and one true residual: 14 float64
    # vectors. Initial copies are released after initialization; final host
    # solution ownership fits within the same conservative overlap bound.
    vector_bound = 14 * 8 * p
    if vector_bound > available_memory_bytes:
        raise MemoryError(f"PCG vector ownership requires {vector_bound} bytes.")
    abstract_vector = jax.ShapeDtypeStruct(rhs.shape, np.float64)
    abstract_scalar = jax.ShapeDtypeStruct((), np.float64)
    abstract_state = PCGState(
        *([abstract_vector] * 4),
        abstract_scalar,
        abstract_scalar,
        jax.ShapeDtypeStruct((), np.int64),
        jax.ShapeDtypeStruct((), np.int32),
    )
    inverse = preconditioner.inverse
    step = jax.jit(
        lambda state, action, inv, tolerance: advance_pcg(
            state, action, lambda v: apply_preconditioner(inv, v), tolerance
        )
    )
    verify = jax.jit(
        lambda state, residual, inv, tolerance: verify_pcg(
            state, residual, lambda v: apply_preconditioner(inv, v), tolerance
        )
    )
    initialize = jax.jit(initialize_pcg)
    inverse_action = jax.jit(apply_preconditioner)
    compiled = (
        step.lower(abstract_state, abstract_vector, inverse, abstract_scalar).compile(),
        verify.lower(
            abstract_state, abstract_vector, inverse, abstract_scalar
        ).compile(),
        initialize.lower(
            abstract_vector, abstract_vector, abstract_vector, abstract_scalar
        ).compile(),
        inverse_action.lower(inverse, abstract_vector).compile(),
    )
    scratch = max(item.memory_analysis().temp_size_in_bytes for item in compiled)
    additional = vector_bound + scratch
    if additional > available_memory_bytes:
        raise MemoryError(
            f"PCG vectors and measured executable scratch require {additional} bytes."
        )
    if reserve_workspace is not None:
        reserve_workspace(additional)
    if not np.all(np.isfinite(rhs)) or (
        initial is not None and not np.all(np.isfinite(initial))
    ):
        raise ValueError("PCG starting vectors must be finite.")
    cpu = jax.devices("cpu")[0]
    rhs_copy = rhs.copy()
    rhs_copy.setflags(write=False)
    rhs_device = jax.device_put(rhs_copy, cpu)
    initial_copy = np.zeros(p) if initial is None else initial.copy()
    initial_copy.setflags(write=False)
    solution = jax.device_put(initial_copy, cpu)
    action_count = 0

    def action(vector):
        nonlocal action_count
        result = matvec(vector)
        action_count += 1
        if (
            result.shape != rhs.shape
            or result.dtype != jnp.float64
            or result.device.platform != "cpu"
        ):
            raise ValueError(
                "Frozen matvec must return a CPU float64 coefficient vector."
            )
        return result

    tolerance = jnp.asarray(
        max(control.atol, control.rtol * np.linalg.norm(rhs)), dtype=jnp.float64
    )
    residual = rhs_device - action(solution)
    preconditioned = compiled[3](inverse, residual)
    state = compiled[2](solution, residual, preconditioned, tolerance)
    del initial_copy, solution, residual, preconditioned
    while (
        int(state.status) in (READY, VERIFY_RESIDUAL)
        and int(state.iteration) < control.max_iterations
    ):
        if int(state.status) == VERIFY_RESIDUAL:
            actual = rhs_device - action(state.solution)
            state = compiled[1](state, actual, inverse, tolerance)
            del actual
        else:
            product = action(state.direction)
            state = compiled[0](state, product, inverse, tolerance)
            del product
    if int(state.status) != CONVERGED:
        actual = rhs_device - action(state.solution)
        if int(state.status) in (READY, VERIFY_RESIDUAL):
            state = compiled[1](state, actual, inverse, tolerance)
            if int(state.status) == READY:
                state = state._replace(
                    status=jnp.asarray(MAX_ITERATIONS, dtype=jnp.int32)
                )
        else:
            norm = jnp.linalg.norm(actual)
            state = state._replace(
                residual_norm=norm,
                status=jnp.where(jnp.isfinite(norm), state.status, NONFINITE),
            )
        del actual
    owned_solution = np.asarray(state.solution).copy()
    owned_solution.setflags(write=False)
    norm = float(state.residual_norm)
    rhs_norm = np.linalg.norm(rhs)
    return HostPCGResult(
        owned_solution,
        int(state.status),
        int(state.iteration),
        action_count,
        norm,
        norm / (rhs_norm if rhs_norm > 0 else 1.0),
        vector_bound,
        scratch,
        additional,
    )
