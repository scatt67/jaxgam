"""Bounded rowless setup for thin-plate and Gaussian-process smooths.

The knot harvesting policy follows mgcv 1.9-3 ``smooth.r``: joint rows use
``uniquecombs(..., ordered=TRUE)`` equivalence and large knot sets use R 4.5's
local Mersenne-Twister/rejection sampler without changing process RNG state.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

import numpy as np

from jaxgam.data.source_tokens import r_row_token, r_sample_knot_rows
from jaxgam.formula.terms import SmoothSpec
from jaxgam.smooths.gaussian_process import GaussianProcessSmooth
from jaxgam.smooths.source_lapack import (
    source_lanczos,
    source_lanczos_workspace_bound,
)
from jaxgam.smooths.tprs import (
    TPRSShrinkageSmooth,
    TPRSSmooth,
    _default_k,
    _source_tprs_knot_basis,
    _SourceColumnSquares,
    default_penalty_order,
    null_space_dimension,
)
from jaxgam.smooths.utils import _SourceMeanAccumulator

if TYPE_CHECKING:
    from jaxgam.formula.operator_prepare import (
        OperatorPreparationControl,
        _CheckedSource,
    )


def radial_width(spec: SmoothSpec) -> int:
    """Return the source basis width without constructing source rows."""
    d = len(spec.variables)
    if spec.bs in {"tp", "ts"}:
        order = int(spec.extra_args.get("m", default_penalty_order(d)))
        null = null_space_dimension(d, order)
        return max(spec.k if spec.k > 0 else _default_k(d, null), null + 1)
    if spec.bs == "gp":
        return GaussianProcessSmooth._default_bs_dim(spec.k, d)
    raise ValueError(f"{spec.bs!r} is not a radial smooth basis")


def _radial_options(spec: SmoothSpec) -> tuple[int, int]:
    if spec.bs in {"tp", "ts"}:
        allowed = {"m", "max_knots", "seed"}
        unknown = set(spec.extra_args) - allowed
        max_knots = spec.extra_args.get("max_knots", 2000)
        seed = spec.extra_args.get("seed", 1)
    else:
        allowed = {"kernel", "rho", "power", "stationary", "xt"}
        unknown = set(spec.extra_args) - allowed
        xt = spec.extra_args.get("xt", {})
        if not isinstance(xt, dict) or set(xt) - {"max_knots", "max.knots", "seed"}:
            raise NotImplementedError(
                "Operator GP xt supports only max_knots and seed."
            )
        max_knots = xt.get("max_knots", xt.get("max.knots", 2000))
        seed = xt.get("seed", 1)
    if unknown:
        raise NotImplementedError(
            f"Operator {spec.bs} options are unsupported: {sorted(unknown)}"
        )
    for name, value in (("max_knots", max_knots), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if max_knots < 1:
        raise ValueError("max_knots must be positive")
    if not -(2**31) <= seed < 2**31:
        raise ValueError("seed must be a signed 32-bit integer")
    return max_knots, seed


def _row_key(row: tuple[float, ...], mode: Literal["exact", "r_token"]):
    if mode == "exact":
        # Numeric equality intentionally identifies +/-0, matching Xd_strip.
        return row
    if len(row) == 1:
        # mgcv uniquecombs uses its numeric single-column path.
        return row
    return r_row_token(row)


def _setup_bytes(q: int, d: int, k: int, batch_rows: int) -> int:
    """Conservative simultaneous discovery/eigen/predictor array bound."""
    # E, distance, Lanczos Q/work, roots/products and ownership copies.
    square = 10 * q * q
    rectangular = 8 * q * (k + d + 4)
    batch = batch_rows * (q + k + 3 * d + 8)
    return max(
        8 * (square + rectangular + batch + 8 * k * k) + 4096,
        # Retained knots/shift, centered knots, E and the largest possible T
        # remain live while the source Lanczos workspace is allocated.
        8 * q * q
        + 8 * q * (k + 2 * d)
        + 8 * d
        + source_lanczos_workspace_bound(max(q, k), k),
    )


def _collect_rows(
    spec: SmoothSpec,
    source: _CheckedSource,
    control: OperatorPreparationControl,
    *,
    max_knots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Collect bounded source-equivalent knots and the full-source shift.

    The final flag reports whether the knots were sampled, i.e. supplied to
    tprs_setup as ``knt`` rather than taken as the data's unique rows.
    """
    from jaxgam.formula.operator_prepare import _check_budget

    d = len(spec.variables)
    k = radial_width(spec)
    base_reserved = source.reserved_bytes
    exact: dict[tuple[float, ...], tuple[float, ...]] = {}
    token: dict[object, tuple[float, ...]] | None = (
        {} if spec.bs == "gp" or source.n_rows > max_knots else None
    )
    retained = sys.getsizeof(exact) + (sys.getsizeof(token) if token is not None else 0)
    source_mean = _SourceMeanAccumulator(d)
    peak = base_reserved + retained
    for batch in source.scan(control.batch_rows):
        columns = []
        for name in spec.variables:
            if name not in batch.columns:
                raise ValueError(f"Smooth variable {name!r} is missing.")
            values = np.asarray(batch.columns[name], dtype=np.float64)
            if values.shape != (len(batch.row_positions),) or not np.all(
                np.isfinite(values)
            ):
                raise ValueError("Radial smooth covariates must be finite vectors.")
            columns.append(values)
        for local in range(len(batch.row_positions)):
            row = tuple(float(values[local]) for values in columns)
            source_mean.add_first(row)
        del batch, columns
    if source_mean.count != source.n_rows:
        raise RuntimeError("Radial smooth source count changed during setup.")
    if spec.bs in {"tp", "ts"}:
        # R's mean.default corrects the extended-precision first-pass mean
        # with a second pass over x - mean(x). A one-ULP shift can change
        # Lanczos Ritz signs and therefore the tensor penalty scale.
        # Reserve the O(d) accumulator, row conversion, and column references
        # so the replay scan's per-row preflight includes this live workspace.
        correction_bytes = 128 * d + 256
        _check_budget(source.reserved_bytes + correction_bytes, control)
        source.reserved_bytes += correction_bytes
        try:
            source_mean.begin_correction()
            for batch in source.scan(control.batch_rows):
                columns = [
                    np.asarray(batch.columns[name], dtype=np.float64)
                    for name in spec.variables
                ]
                for local in range(len(batch.row_positions)):
                    source_mean.add_correction(tuple(col[local] for col in columns))
        finally:
            source.reserved_bytes -= correction_bytes
    shift = source_mean.result()
    # smooth.construct.tp centers before uniquecombs; GP harvests unique rows
    # before centering. The distinction changes R's formatted joint-row order.
    for batch in source.scan(control.batch_rows):
        columns = []
        for name in spec.variables:
            values = np.asarray(batch.columns[name], dtype=np.float64)
            if values.shape != (len(batch.row_positions),) or not np.all(
                np.isfinite(values)
            ):
                raise ValueError("Radial smooth covariates must be finite vectors.")
            columns.append(values)
        for local in range(len(batch.row_positions)):
            row = tuple(float(values[local]) for values in columns)
            key_row = (
                tuple(float(row[j] - shift[j]) for j in range(d))
                if spec.bs in {"tp", "ts"}
                else row
            )
            for mapping, mode in ((exact, "exact"), (token, "r_token")):
                if mapping is None:
                    continue
                key = _row_key(key_row, mode)
                if key in mapping:
                    continue
                q = len(mapping) + 1
                old_size = sys.getsizeof(mapping)
                payload = sys.getsizeof(key) + sys.getsizeof(row) + 64 * (d + 2)
                resize_overlap = 4 * old_size + 1024
                required = (
                    base_reserved
                    + retained
                    + payload
                    + resize_overlap
                    + _setup_bytes(q, d, k, control.batch_rows)
                )
                _check_budget(required, control)
                mapping[key] = row
                retained += payload + max(0, sys.getsizeof(mapping) - old_size)
                source.reserved_bytes = base_reserved + retained
                peak = max(peak, required)
        del batch, columns
    if spec.bs == "gp" or (
        source.n_rows > max_knots and token is not None and len(token) > max_knots
    ):
        selected = token
    else:
        selected = exact
    assert selected is not None
    if len(selected) < k:
        raise ValueError(
            f"A term has fewer unique covariate combinations ({len(selected)}) "
            f"than its basis dimension ({k})."
        )
    if selected is token and d > 1:
        ordered = sorted(selected.items(), key=lambda item: str(item[0]).encode())
        rows = [value for _, value in ordered]
    else:
        rows = sorted(selected.values())
    materialize = retained + 2 * len(rows) * d * 8 + len(rows) * 16
    required = (
        base_reserved + materialize + _setup_bytes(len(rows), d, k, control.batch_rows)
    )
    _check_budget(required, control)
    peak = max(
        peak,
        base_reserved + materialize + _setup_bytes(len(rows), d, k, control.batch_rows),
    )
    unique = np.asarray(rows, dtype=np.float64)
    sampled = source.n_rows > max_knots and len(unique) > max_knots
    if sampled:
        unique = r_sample_knot_rows(unique, max_knots, seed)
    if spec.bs in {"tp", "ts"}:
        # tprs.c passes supplied knots through Xd_strip, which sorts the
        # sampled locations before E/T and exposes that order as object$Xu.
        order = np.lexsort(tuple(unique[:, j] for j in reversed(range(d))))
        unique = unique[order].copy()
    source.reserved_bytes = base_reserved + int(unique.nbytes) + shift.nbytes
    return unique, shift, peak, sampled


def _stream_design_stats(
    smooth: TPRSSmooth | GaussianProcessSmooth,
    source: _CheckedSource,
    control: OperatorPreparationControl,
    *,
    rms: bool,
) -> tuple[np.ndarray, float]:
    from jaxgam.formula.operator_prepare import _check_budget
    from jaxgam.formula.operator_workspace import smooth_evaluator_workspace_bound

    workspace = smooth_evaluator_workspace_bound(smooth, control.batch_rows) + 8 * (
        2 * smooth.n_coefs + control.batch_rows * (2 * smooth.n_coefs + 2)
    )
    _check_budget(source.reserved_bytes + workspace, control)
    source.peak_bound_bytes = max(
        source.peak_bound_bytes, source.reserved_bytes + workspace
    )
    squares = _SourceColumnSquares(smooth.n_coefs)
    infinity = 0.0
    for batch in source.scan(control.batch_rows):
        rows = smooth.training_matrix(batch.columns)
        squares.add(rows)
        infinity = max(infinity, float(np.max(np.sum(np.abs(rows), axis=1))))
        del rows, batch
    if squares.count != source.n_rows:
        raise RuntimeError("Radial smooth source count changed during setup.")
    scales = squares.scales() if rms else np.ones(smooth.n_coefs)
    return scales, infinity


def prepare_radial_smooth(
    spec: SmoothSpec,
    source: _CheckedSource,
    control: OperatorPreparationControl,
) -> tuple[TPRSSmooth | GaussianProcessSmooth, int]:
    """Prepare one frozen TP/TS/GP evaluator without retaining training rows."""
    max_knots, seed = _radial_options(spec)
    knots, shift, peak, sampled = _collect_rows(
        spec, source, control, max_knots=max_knots, seed=seed
    )
    d = len(spec.variables)
    if spec.bs in {"tp", "ts"}:
        smooth: TPRSSmooth = (
            TPRSShrinkageSmooth(spec) if spec.bs == "ts" else TPRSSmooth(spec)
        )
        order = int(spec.extra_args.get("m", default_penalty_order(d)))
        null = null_space_dimension(d, order)
        k = radial_width(spec)
        centered = knots - shift
        available = (
            control.memory_budget_bytes - source.reserved_bytes - centered.nbytes
        )
        # The dense constructor and this bounded replay share tprs_setup.
        basis = _source_tprs_knot_basis(
            centered, order, k, available_workspace_bytes=max(available, 0)
        )
        rank = k - null
        smooth._d = d
        smooth._m = order
        smooth._M = null
        smooth._k = k
        smooth._shift = shift
        smooth._Xu = centered
        # Unscaled training rows and UZ give tprs_setup's column statistics.
        smooth._UZ = basis.UZ
        smooth._knots_from_data = not sampled and basis.knot_rows is not None
        smooth._knot_rows = basis.knot_rows if smooth._knots_from_data else None
        smooth.n_coefs = k
        smooth.rank = rank
        smooth.null_space_dim = null
        smooth._is_setup = True
        del knots
        retained = sum(
            int(value.nbytes)
            for value in (smooth._Xu, basis.UZ, smooth._shift, basis.S)
            + ((basis.knot_rows,) if basis.knot_rows is not None else ())
        )
        source.reserved_bytes += retained
        source.peak_bound_bytes = max(source.peak_bound_bytes, peak)
        scales, _ = _stream_design_stats(smooth, source, control, rms=True)
        smooth._apply_source_scaling(basis, scales, smooth._knots_from_data)
        smooth._S = smooth._apply_shrinkage(smooth._S)
        if spec.bs == "ts":
            smooth.rank = k
            smooth.null_space_dim = 0
        smooth._X = None
        return smooth, peak

    gp = GaussianProcessSmooth(spec)
    null = 1 if gp._stationary else d + 1
    width = radial_width(spec)
    rank = width - null
    centered = knots - shift
    gp._shift = shift
    gp._knt = centered
    E, rho = gp._gp_E(centered, centered)
    rho_bytes = int(np.asarray(rho).nbytes)
    lanczos_live = centered.nbytes + E.nbytes + rho_bytes
    available = control.memory_budget_bytes - source.reserved_bytes - lanczos_live
    values, vectors = source_lanczos(
        E,
        rank,
        tol=np.finfo(float).eps ** 0.5,
        available_workspace_bytes=max(available, 0),
    )
    # Pinned smooth.r stores signed retained values. The eventual SPD
    # certificate may reject an indefinite fitted system; setup never repairs
    # the model by taking absolute values.
    D = np.zeros((width, width))
    np.fill_diagonal(D[:rank, :rank], values)
    gp._resolved_rho = rho
    gp._UZ = vectors
    gp.n_coefs = width
    gp.rank = rank
    gp.null_space_dim = null
    gp._S = D
    gp._E_knot = None
    gp._X = None
    gp._is_setup = True
    del E, values, vectors, knots
    retained = sum(int(value.nbytes) for value in (gp._knt, gp._UZ, gp._shift, gp._S))
    source.reserved_bytes += retained
    source.peak_bound_bytes = max(source.peak_bound_bytes, peak)
    return gp, peak
