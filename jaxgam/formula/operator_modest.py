"""Bounded, rowless cubic/by preparation for operator geometry (NumPy)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import linalg

from jaxgam.formula.design import ModelSetup
from jaxgam.formula.predict_matrix import PredictSpec
from jaxgam.formula.prepare import (
    PreparedModel,
    _cubic_from_knots,
    _exact_cubic_knots,
    prepare_model,
)
from jaxgam.formula.terms import FormulaSpec, SmoothSpec
from jaxgam.penalties.penalty import Penalty
from jaxgam.penalties.structure import make_penalty_structure
from jaxgam.smooths.base import Smooth
from jaxgam.smooths.by_variable import FactorBySmooth, NumericBySmooth
from jaxgam.smooths.constraints import CoefficientMap, TermBlock
from jaxgam.smooths.source_lapack import (
    SourceConstraintQR,
    cubic_shrinkage_workspace_bound,
)
from jaxgam.smooths.tensor import (
    SourceColumnMeans,
    TensorInteractionSmooth,
    TensorProductSmooth,
    source_reparameterized_penalty,
)
from jaxgam.smooths.utils import is_factor, is_ordered_factor

if TYPE_CHECKING:
    from jaxgam.formula.operator_prepare import (
        OperatorPreparationControl,
        _CheckedSource,
    )


def _label_bound(value) -> int:
    if isinstance(value, str):
        return 512 + 8 * len(value)
    if isinstance(value, bytes):
        return 512 + 32 * len(value)
    if isinstance(value, int):
        return 512 + 8 * (2 + math.ceil(value.bit_length() * math.log10(2)))
    return 1024


def _width(term) -> int:
    if term.bs in {"tp", "ts", "gp"}:
        from jaxgam.formula.operator_radial import radial_width

        if term.smooth_type == "s":
            return radial_width(term)
        margin = replace(
            term,
            variables=[term.variables[0]],
            k=5 if term.k == -1 else term.k,
            smooth_type="s",
            by=None,
        )
        width = radial_width(margin) - int(term.smooth_type == "ti")
        return width ** len(term.variables)
    default = 10 if term.smooth_type == "s" else 5
    k = default if term.k == -1 else max(term.k, 4 if term.bs == "cc" else 3)
    if term.smooth_type == "s":
        # Include the cyclic endpoint in the conservative evaluator width.
        return k
    width = k - int(term.bs == "cc") - int(term.smooth_type == "ti")
    return width ** len(term.variables)


def _cubic_with_budget(spec, knots, source, control):
    """Construct one cubic basis with prospective CS provider accounting."""
    from jaxgam.formula.operator_prepare import _check_budget

    workspace = cubic_shrinkage_workspace_bound(len(knots)) if spec.bs == "cs" else 0
    _check_budget(source.reserved_bytes + workspace, control)
    source.peak_bound_bytes = max(
        source.peak_bound_bytes, source.reserved_bytes + workspace
    )
    return _cubic_from_knots(
        spec,
        knots,
        shrinkage_workspace_bytes=(
            control.memory_budget_bytes - source.reserved_bytes
            if spec.bs == "cs"
            else None
        ),
    )


def _make_tensor(term, source, control):
    """Construct capped margin metadata and tensor transforms without rows."""
    tensor_class = (
        TensorInteractionSmooth if term.smooth_type == "ti" else TensorProductSmooth
    )
    tensor = tensor_class(replace(term, by=None))
    margins = []
    for variable in term.variables:
        if term.bs in {"tp", "ts", "gp"}:
            from jaxgam.formula.operator_radial import prepare_radial_smooth

            margin_spec = replace(
                term,
                variables=[variable],
                k=5 if term.k == -1 else term.k,
                smooth_type="s",
                by=None,
            )
            margin, _ = prepare_radial_smooth(margin_spec, source, control)
        else:
            k = 5 if term.k == -1 else max(term.k, 4 if term.bs == "cc" else 3)
            margin_spec = SmoothSpec([variable], bs=term.bs, k=k)
            knots = _exact_cubic_knots(source, variable, k)
            margin = _cubic_with_budget(margin_spec, knots, source, control)
        margins.append(margin)
    tensor._marginals = margins
    from jaxgam.formula.operator_prepare import _check_budget
    from jaxgam.formula.operator_workspace import smooth_evaluator_workspace_bound

    margin_workspace = sum(
        smooth_evaluator_workspace_bound(margin, control.batch_rows)
        for margin in margins
    ) + 8 * control.batch_rows * (4 * sum(margin.n_coefs for margin in margins) + 8)
    _check_budget(source.reserved_bytes + margin_workspace, control)
    source.peak_bound_bytes = max(
        source.peak_bound_bytes, source.reserved_bytes + margin_workspace
    )
    S_list = [margin._S for margin in margins]
    dims = [margin.n_coefs for margin in margins]
    ranks = [margin.rank for margin in margins]
    nulls = [margin.null_space_dim for margin in margins]
    if term.smooth_type == "ti":
        # ti margins are built by smoothCon: colMeans centering constraint and
        # scale.penalty, exactly as the dense TensorInteractionSmooth does.
        means = [SourceColumnMeans(margin.n_coefs) for margin in margins]
        row_norms = [0.0 for _ in margins]
        for batch in source.scan(control.batch_rows):
            for i, margin in enumerate(margins):
                rows = margin.training_matrix(batch.columns)
                means[i].add(rows)
                row_norms[i] = max(
                    row_norms[i], float(np.max(np.sum(np.abs(rows), axis=1)))
                )
                del rows
            del batch
        tensor._constraints = [
            SourceConstraintQR.from_column(mean.result()) for mean in means
        ]
        tensor._Z_list = [constraint.basis() for constraint in tensor._constraints]
        scaled = []
        for S, norm in zip(S_list, row_norms, strict=True):
            [S], _ = Smooth._smoothcon_normalize_from_norm(norm, [S])
            scaled.append(S)
        S_list = [
            constraint.absorb_penalty(S)
            for constraint, S in zip(tensor._constraints, scaled, strict=True)
        ]
        dims = [dim - 1 for dim in dims]
        nulls = [max(null - 1, 0) for null in nulls]
        ranks = [dim - null for dim, null in zip(dims, nulls, strict=True)]
    ranges = [[math.inf, -math.inf] for _ in margins]
    for batch in source.scan(control.batch_rows):
        for i, margin in enumerate(margins):
            values = np.asarray(batch.columns[margin.spec.variables[0]], dtype=float)
            ranges[i][0] = min(ranges[i][0], float(np.min(values)))
            ranges[i][1] = max(ranges[i][1], float(np.max(values)))
        del batch
    tensor._XP_list = []
    for i, margin in enumerate(margins):
        if margin._noterp or len(margin.spec.variables) != 1:
            tensor._XP_list.append(None)
            continue
        predictor_workspace = smooth_evaluator_workspace_bound(margin, dims[i])
        _check_budget(source.reserved_bytes + predictor_workspace, control)
        source.peak_bound_bytes = max(
            source.peak_bound_bytes, source.reserved_bytes + predictor_workspace
        )
        XP = tensor._reparameterize_margin(
            i, margin, ranges[i][0], ranges[i][1], dims[i]
        )
        tensor._XP_list.append(XP)
        if XP is not None:
            S_list[i] = source_reparameterized_penalty(S_list[i], XP)
    tensor.n_coefs = math.prod(dims)
    tensor.null_space_dim = math.prod(nulls)
    tensor.rank = tensor.n_coefs - tensor.null_space_dim
    tensor._penalties = tensor._build_tensor_penalties(S_list, dims, ranks)
    tensor._s_scale = 1.0
    tensor._is_setup = True
    return tensor


def prepare_modest_cubic_by(
    spec: FormulaSpec,
    source: _CheckedSource,
    control: OperatorPreparationControl,
) -> tuple[PreparedModel, int, int, int]:
    """Prepare capped cubic/by terms and source-aligned overlap constraints.

    ``source`` is the operator's checked replay wrapper. Every owned level,
    expanded local matrix and joint row reduction is budgeted before creation.
    Returns model, raw width, additional retained bound and construction bound.
    """
    # Import lazily to retain the existing operator preparation entry point.
    from jaxgam.formula.operator_prepare import _check_budget

    fingerprint = source.fingerprint()
    first = next(source.scan(1))
    terms = spec.smooth_terms
    sizes = [_width(term) for term in terms]
    parametric_raw = int(spec.has_intercept) + len(spec.parametric_terms)
    raw_width = parametric_raw + sum(sizes)
    levels = {}
    ordered = {}
    object_levels = {}
    extra_metadata = 1024 * (len(terms) + 1)
    initial_reserved = source.reserved_bytes
    replay_workspace = max(0, source.peak_bound_bytes - initial_reserved)
    label_costs = [
        8
        * (256 + sum(len(v) for v in term.variables) + len(term.by or ""))
        * (sizes[i] + 1)
        for i, term in enumerate(terms)
    ]
    extra_metadata += sum(label_costs)
    _check_budget(initial_reserved + extra_metadata + replay_workspace, control)
    source.reserved_bytes = initial_reserved + extra_metadata

    def check_width(width):
        if width > control.max_modest_coefficients:
            raise MemoryError(
                f"Modest operator width {width} exceeds hard cap "
                f"{control.max_modest_coefficients}."
            )

    check_width(raw_width)
    for i, term in enumerate(terms):
        if term.smooth_type != "s":
            k = 5 if term.k == -1 else max(term.k, 4 if term.bs == "cc" else 3)
            check_width(k * len(term.variables))
        if term.by is None:
            continue
        column = first.columns[term.by]
        if not is_factor(column):
            continue
        ordered[i] = is_ordered_factor(column)
        if isinstance(getattr(column, "dtype", None), pd.CategoricalDtype):
            categories = column.dtype.categories
            if not len(categories):
                raise ValueError("Factor by covariates require declared levels.")
            count = len(categories) - int(ordered[i] and len(categories) > 1)
            check_width(raw_width + (count - 1) * sizes[i])
            cost = sum(
                (_label_bound(value) * (sizes[i] + 1)) + label_costs[i]
                for value in categories
            )
            _check_budget(
                initial_reserved + extra_metadata + cost + replay_workspace, control
            )
            extra_metadata += cost
            source.reserved_bytes = initial_reserved + extra_metadata
            levels[i] = tuple(categories)
            raw_width += (count - 1) * sizes[i]
        else:
            object_levels[i] = {}
    del first
    if object_levels:
        for batch in source.scan(control.batch_rows):
            for i, discovered in object_levels.items():
                for value in batch.columns[terms[i].by]:
                    if value in discovered:
                        continue
                    future = len(discovered) + 1
                    future_width = raw_width + (sizes[i] if future > 1 else 0)
                    check_width(future_width)
                    cost = _label_bound(value) * (sizes[i] + 1) + label_costs[i]
                    replay_workspace = max(
                        replay_workspace,
                        source.peak_bound_bytes - source.reserved_bytes,
                    )
                    _check_budget(
                        initial_reserved + extra_metadata + cost + replay_workspace,
                        control,
                    )
                    extra_metadata += cost
                    source.reserved_bytes = initial_reserved + extra_metadata
                    discovered[value] = None
                    raw_width = future_width
        for i, discovered in object_levels.items():
            levels[i] = tuple(sorted(discovered))
            check_width(raw_width)
    n_penalties = sum(
        (len(levels[i]) - int(ordered[i] and len(levels[i]) > 1) if i in levels else 1)
        * (len(term.variables) if term.smooth_type != "s" else 1)
        for i, term in enumerate(terms)
    )
    # Simultaneous raw/constrained per-level penalties, joint QR, temporary
    # products, retained transforms and rows; independent of global RE width.
    construction = (
        initial_reserved
        + extra_metadata
        + 8
        * (
            (2 * n_penalties + 32) * (raw_width + 1) ** 2
            + control.batch_rows * (8 * raw_width + 32)
        )
    )
    _check_budget(construction + replay_workspace, control)
    source.reserved_bytes = construction
    parametric = prepare_model(replace(spec, smooth_terms=[]), source)
    parametric_spec = parametric.predict_spec
    n_parametric = parametric.n_coef
    bases = []
    for term in terms:
        if term.smooth_type != "s":
            bases.append(_make_tensor(term, source, control))
            continue
        if term.bs in {"tp", "ts", "gp"}:
            from jaxgam.formula.operator_radial import prepare_radial_smooth

            base, _ = prepare_radial_smooth(replace(term, by=None), source, control)
            bases.append(base)
        else:
            k = 10 if term.k == -1 else max(term.k, 4 if term.bs == "cc" else 3)
            knots = _exact_cubic_knots(source, term.variables[0], k)
            bases.append(
                _cubic_with_budget(replace(term, by=None), knots, source, control)
            )
    sums = [np.zeros(base.n_coefs) for base in bases]
    norms = [0.0] * len(bases)
    moments = [[0, 0.0, 0.0] for _ in bases]
    from jaxgam.formula.operator_workspace import smooth_evaluator_workspace_bound

    evaluator_workspace = sum(
        smooth_evaluator_workspace_bound(base, control.batch_rows) for base in bases
    ) + 8 * control.batch_rows * (4 * sum(base.n_coefs for base in bases) + 8)
    _check_budget(source.reserved_bytes + evaluator_workspace, control)
    source.peak_bound_bytes = max(
        source.peak_bound_bytes, source.reserved_bytes + evaluator_workspace
    )
    for batch in source.scan(control.batch_rows):
        for i, base in enumerate(bases):
            X = base.training_matrix(batch.columns)
            sums[i] += np.sum(X, axis=0)
            norms[i] = max(norms[i], float(np.max(np.sum(np.abs(X), axis=1))))
            if terms[i].by is not None and i not in levels:
                by = np.asarray(batch.columns[terms[i].by], dtype=np.float64)
                if not np.all(np.isfinite(by)):
                    raise ValueError("Numeric by covariates must be finite.")
                count, mean, m2 = moments[i]
                n = len(by)
                leaf_mean = float(np.mean(by))
                centered = by - leaf_mean
                leaf_m2 = float(np.dot(centered, centered))
                total = count + n
                delta = leaf_mean - mean
                moments[i] = [
                    total,
                    mean + delta * n / total,
                    m2 + leaf_m2 + delta**2 * count * n / total,
                ]
            del X
        del batch
    smooths = []
    transforms = []
    penalties = []
    widths = []
    for i, (base, term) in enumerate(zip(bases, terms, strict=True)):
        raw_penalties = [penalty.S for penalty in base.build_penalty_matrices()]
        if norms[i] > 0:
            scales = [np.linalg.norm(S, ord=1) / norms[i] ** 2 for S in raw_penalties]
            raw_penalties = [
                S / scale for S, scale in zip(raw_penalties, scales, strict=True)
            ]
        else:
            scales = [1.0] * len(raw_penalties)
        if isinstance(base, TensorProductSmooth):
            base._penalties = [
                Penalty(S, rank=penalty.rank, null_space_dim=penalty.null_space_dim)
                for penalty, S in zip(base._penalties, raw_penalties, strict=True)
            ]
        else:
            base._S = raw_penalties[0]
            base._s_scale = scales[0]
        Z = np.linalg.qr(sums[i][:, None], mode="complete")[0][:, 1:]
        if i in levels:
            all_levels = levels[i]
            active = (
                all_levels[1:] if ordered[i] and len(all_levels) > 1 else all_levels
            )
            smooth = FactorBySmooth(base, term, list(active), term.by, list(all_levels))
            smooth._base_constraint = sums[i]
            Z = np.kron(np.eye(len(active)), Z)
            S = [penalty.S for penalty in smooth.build_penalty_matrices()]
        elif term.by is not None:
            count, mean, m2 = moments[i]
            sd = math.sqrt(m2 / (count - 1)) if count > 1 else math.nan
            nonconstant = sd > mean * np.finfo(float).eps * 1000
            smooth = NumericBySmooth(base, term, term.by)
            smooth.has_centering_constraint = not nonconstant
            if nonconstant:
                Z = None
            S = raw_penalties
        else:
            smooth = base
            S = raw_penalties
        if isinstance(base, TensorInteractionSmooth):
            Z = None
        if Z is not None:
            S = [Z.T @ penalty @ Z for penalty in S]
        S = [(penalty + penalty.T) * 0.5 for penalty in S]
        smooths.append(smooth)
        transforms.append(Z)
        penalties.append(S)
        widths.append(smooth.n_coefs if Z is None else Z.shape[1])

    # The leading constant column is retained explicitly in the row reduction
    # so gam.side references use Q.T @ 1 rather than fabricated reduced ones.
    R = np.empty((0, 1 + n_parametric + sum(widths)))
    # PreparedModel's replay contract: the scan-order row positions digest.
    row_positions = hashlib.sha256(b"jaxgam-row-positions-v1\0")
    for batch in source.scan(control.batch_rows):
        row_positions.update(
            np.ascontiguousarray(batch.row_positions, dtype=np.int64).view(np.uint8)
        )
        components = [np.ones((len(batch.row_positions), 1))]
        if n_parametric:
            components.append(parametric.evaluate_batch(batch))
        for smooth, Z in zip(smooths, transforms, strict=True):
            X = smooth.training_matrix(batch.columns)
            components.append(X if Z is None else X @ Z)
        rows = np.column_stack(components)
        if not np.all(np.isfinite(rows)):
            raise ValueError("Modest operator covariates must be finite.")
        R = linalg.qr(np.vstack((R, rows)), mode="economic")[1]
        del batch, components, rows, X
    offset = 1 + n_parametric
    blocks = []
    for width in widths:
        blocks.append(R[:, offset : offset + width].copy())
        offset += width
    deletions = CoefficientMap.gam_side(
        smooths,
        blocks,
        penalties,
        R[:, 1 : 1 + n_parametric],
        intercept_reference=R[:, 0] if spec.has_intercept else None,
    )
    term_blocks = list(parametric_spec.coef_map.terms)
    training_blocks = list(parametric_spec.coef_map.terms)
    offset = n_parametric
    penalty_offset = 0
    for smooth, Z, X, S, deleted in zip(
        smooths, transforms, blocks, penalties, deletions, strict=True
    ):
        block = TermBlock(
            CoefficientMap.smooth_label(smooth),
            offset,
            X.shape[1],
            smooth.n_coefs,
            "smooth",
            smooth.copy_for_prediction(),
            tuple(range(penalty_offset, penalty_offset + len(S))),
            Z,
            deleted or (),
        )
        term_blocks.append(block)
        training_blocks.append(replace(block, smooth=smooth.copy_for_training()))
        offset += X.shape[1]
        penalty_offset += len(S)
    coef_map = CoefficientMap(
        tuple(term_blocks),
        offset,
        n_parametric + sum(smooth.n_coefs for smooth in smooths),
        spec.has_intercept,
    )
    structure = make_penalty_structure(
        offset, [term for term in term_blocks if term.term_type == "smooth"], penalties
    )
    prediction = PredictSpec(
        coef_map,
        tuple(ModelSetup._build_smooth_info(smooths, coef_map)),
        ModelSetup._build_term_names(
            list(parametric_spec.term_names), smooths, coef_map
        ),
        tuple(spec.parametric_terms),
        {},
        frozenset(),
        spec.has_intercept,
        parametric_spec.parametric_keep_cols,
        parametric_spec.dropped_param_names,
        offset,
    )
    if source.fingerprint() != fingerprint:
        raise RuntimeError("RowSource changed during modest operator preparation.")
    basis = hashlib.sha256()
    for term in term_blocks:
        if term.Z_centering is not None:
            basis.update(term.Z_centering.tobytes())
        basis.update(repr((term.label, term.n_coefs, term.del_index)).encode())
    for base in bases:
        for margin in (
            base._marginals if isinstance(base, TensorProductSmooth) else [base]
        ):
            locations = getattr(margin, "_knots", None)
            if locations is None:
                locations = getattr(margin, "_Xu", None)
            if locations is None:
                locations = getattr(margin, "_knt", None)
            if locations is not None:
                basis.update(locations.tobytes())
    # Coupled tensor penalties within each factor level can prevent the
    # disjoint-interval optimization: retain every actual modest matrix.
    radial_retained = 0
    for base in bases:
        for margin in (
            base._marginals if isinstance(base, TensorProductSmooth) else [base]
        ):
            if margin.spec.bs not in {"tp", "ts", "gp"}:
                continue
            for name in (
                "_Xu",
                "_UZ",
                "_shift",
                "_col_norms",
                "_knt",
                "_S",
                "_knot_rows",
            ):
                value = getattr(margin, name, None)
                if isinstance(value, np.ndarray):
                    radial_retained += int(value.nbytes)
    retained = (
        extra_metadata
        + radial_retained
        + 8 * (2 * n_penalties + 16) * (raw_width + 1) ** 2
    )
    construction = max(construction, source.peak_bound_bytes)
    training = replace(
        prediction,
        coef_map=replace(coef_map, terms=tuple(training_blocks)),
    )
    model = PreparedModel(
        prediction,
        structure,
        fingerprint,
        row_positions.hexdigest(),
        basis.hexdigest(),
        source.n_rows,
        spec.response,
        training_spec=training,
    )
    return model, raw_width, retained, construction
