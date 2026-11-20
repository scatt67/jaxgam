"""Build continuous-discrete descriptors from already-owned compact indices."""

from __future__ import annotations

from typing import Any

import numpy as np

from jaxgam.data.discretize import ExactIndexTable
from jaxgam.data.source import RowSource
from jaxgam.formula import predict_matrix
from jaxgam.formula.discrete.continuous_setup import ContinuousDiscreteSetup
from jaxgam.formula.discrete.design import (
    DiscreteBlock,
    DiscreteDesignControl,
    DiscreteDesignLineage,
    DiscreteDesignUnavailable,
    FrozenDiscreteDesign,
    LookupBlock,
    TensorLookupBlock,
    _Ledger,
    _memory_plan,
    _prospective_matrix_bytes,
    _smooth_evaluator_bytes,
)
from jaxgam.smooths.by_variable import FactorBySmooth, NumericBySmooth
from jaxgam.smooths.constraints import TermBlock
from jaxgam.smooths.tensor import TensorInteractionSmooth, TensorProductSmooth


def _index_for_names(
    setup: ContinuousDiscreteSetup, names: tuple[str, ...]
) -> ExactIndexTable:
    mapping = dict(setup.compact.name_group_indices)
    try:
        group_numbers = {mapping[name] for name in names}
    except KeyError as error:
        raise KeyError(f"No compact index for variable {error.args[0]!r}") from error
    if len(group_numbers) != 1:
        raise ValueError("compact descriptor requires one shared index per joint term")
    group = setup.compact.groups[next(iter(group_numbers))]
    if set(group.names) != set(names):
        raise ValueError("compact joint-index variables do not match the frozen term")
    return group.index.index


def _single_index(setup: ContinuousDiscreteSetup, name: str) -> ExactIndexTable:
    mapping = dict(setup.compact.name_group_indices)
    try:
        return setup.compact.groups[mapping[name]].index.index
    except KeyError as error:
        raise KeyError(f"No compact index for variable {name!r}") from error


def _values(index: ExactIndexTable) -> dict[str, np.ndarray]:
    return dict(zip(index.names, index.values, strict=True))


def _value_for_name(index: ExactIndexTable, name: str) -> np.ndarray:
    """Select one named column from a possibly shared joint index."""
    try:
        column = index.names.index(name)
    except ValueError as error:
        raise KeyError(f"Compact index does not contain variable {name!r}") from error
    return index.values[column]


def _keep(term: TermBlock, transformed_width: int) -> np.ndarray:
    if not term.del_index:
        return np.arange(transformed_width, dtype=np.int32)
    deleted = set(term.del_index)
    return np.asarray(
        [column for column in range(transformed_width) if column not in deleted],
        dtype=np.int32,
    )


def _reserve_product_metadata(
    term: TermBlock,
    raw_width: int,
    n_levels: int,
    ledger: _Ledger,
    lineage: DiscreteDesignLineage,
) -> tuple[np.ndarray, DiscreteDesignUnavailable | None]:
    transformed = (
        raw_width * n_levels if term.Z_centering is None else term.Z_centering.shape[1]
    )
    failed = ledger.check_temporary(
        transformed * np.dtype(np.int32).itemsize,
        lineage,
        "compact retained-column selector exceeds budget",
    )
    if failed is not None:
        return np.empty(0, dtype=np.int32), failed
    keep = _keep(term, transformed)
    for metadata in (term.Z_centering, keep):
        if metadata is None:
            continue
        failed = ledger.add_metadata(int(metadata.nbytes), lineage)
        if failed is not None:
            return keep, failed
    return keep, None


def _by_metadata(
    wrapper: NumericBySmooth | FactorBySmooth | None,
    setup: ContinuousDiscreteSetup,
    ledger: _Ledger,
    lineage: DiscreteDesignLineage,
) -> tuple[
    ExactIndexTable | None,
    np.ndarray | None,
    ExactIndexTable | None,
    np.ndarray | None,
    int,
    DiscreteDesignUnavailable | None,
]:
    if wrapper is None:
        return None, None, None, None, 1, None
    by_index = _single_index(setup, wrapper.by_variable)
    by_values = _value_for_name(by_index, wrapper.by_variable)
    if isinstance(wrapper, NumericBySmooth):
        failed = ledger.check_temporary(
            by_index.n_unique * np.dtype(np.float64).itemsize,
            lineage,
            "compact numeric-by metadata exceeds budget",
        )
        if failed is not None:
            return None, None, None, None, 1, failed
        multiplier = np.asarray(by_values, dtype=np.float64)
        failed = ledger.add_metadata(int(multiplier.nbytes), lineage)
        return by_index, multiplier, None, None, 1, failed
    failed = ledger.check_temporary(
        by_index.n_unique * np.dtype(np.int32).itemsize,
        lineage,
        "compact factor-by metadata exceeds budget",
    )
    if failed is not None:
        return None, None, None, None, wrapper.n_levels, failed
    mapping = {value: number for number, value in enumerate(wrapper.levels)}
    unknown = [value for value in by_values if value not in wrapper.all_levels]
    if unknown:
        raise ValueError(
            "factor replay contains levels absent from frozen prediction state"
        )
    codes = np.asarray([mapping.get(value, -1) for value in by_values], dtype=np.int32)
    failed = ledger.add_metadata(int(codes.nbytes), lineage)
    return None, None, by_index, codes, wrapper.n_levels, failed


def _product_block(
    term: TermBlock,
    setup: ContinuousDiscreteSetup,
    ledger: _Ledger,
    lineage: DiscreteDesignLineage,
) -> TensorLookupBlock | DiscreteDesignUnavailable:
    wrapper = (
        term.smooth
        if isinstance(term.smooth, (NumericBySmooth, FactorBySmooth))
        else None
    )
    smooth = wrapper.base_smooth if wrapper is not None else term.smooth
    assert smooth is not None
    tables: list[np.ndarray] = []
    indices: list[ExactIndexTable] = []
    if isinstance(smooth, TensorProductSmooth):
        items: list[tuple[str, Any, np.ndarray | None, np.ndarray | None]] = []
        for number, (name, marginal) in enumerate(
            zip(smooth.spec.variables, smooth._marginals, strict=True)
        ):
            z = (
                smooth._Z_list[number]
                if isinstance(smooth, TensorInteractionSmooth)
                else None
            )
            items.append((name, marginal, z, smooth._XP_list[number]))
    else:
        items = [("", smooth, None, None)]

    for name, marginal, z, xp in items:
        index = (
            _index_for_names(setup, tuple(smooth.spec.variables))
            if not name
            else _single_index(setup, name)
        )
        input_values = _values(index)
        transform_widths = tuple(
            transform.shape[1] for transform in (z, xp) if transform is not None
        )
        failed = ledger.check_temporary(
            _prospective_matrix_bytes(
                index.n_unique,
                marginal.n_coefs,
                transform_widths,
                evaluator_bytes=_smooth_evaluator_bytes(
                    marginal, index.n_unique, marginal.n_coefs
                ),
            ),
            lineage,
            "compact product predictor workspace exceeds budget",
        )
        if failed is not None:
            return failed
        table = marginal.predict_matrix(input_values)
        if z is not None:
            table = table @ z
        if xp is not None:
            table = table @ xp
        failed = ledger.add_table(
            int(table.nbytes), lineage, live_temporary_bytes=int(table.nbytes)
        )
        if failed is not None:
            return failed
        tables.append(table)
        indices.append(index)

    by = _by_metadata(wrapper, setup, ledger, lineage)
    multiplier_index, multiplier, level_index, codes, n_levels, failed = by
    if failed is not None:
        return failed
    raw_width = int(np.prod([table.shape[1] for table in tables]))
    keep, failed = _reserve_product_metadata(term, raw_width, n_levels, ledger, lineage)
    if failed is not None:
        return failed
    source_copy_bytes = sum(int(table.nbytes) for table in tables) + int(keep.nbytes)
    source_copy_bytes += sum(
        int(value.nbytes)
        for value in (term.Z_centering, multiplier, codes)
        if value is not None
    )
    failed = ledger.check_temporary(
        source_copy_bytes,
        lineage,
        "compact product ownership copy exceeds budget",
    )
    if failed is not None:
        return failed
    return TensorLookupBlock(
        term.col_start,
        tuple(tables),
        tuple(indices),
        term.Z_centering,
        keep,
        multiplier_index,
        multiplier,
        level_index,
        codes,
        n_levels,
    )


def _parametric_blocks(
    setup: ContinuousDiscreteSetup,
    term: TermBlock,
    ledger: _Ledger,
    lineage: DiscreteDesignLineage,
) -> tuple[list[LookupBlock], DiscreteDesignUnavailable | None]:
    spec = setup.predict_spec
    keep = (
        set(spec.parametric_keep_cols)
        if spec.dropped_param_names
        else set(range(term.n_coefs_raw))
    )
    blocks: list[LookupBlock] = []
    raw_column = 0
    output_column = term.col_start
    if spec.has_intercept:
        index = setup.compact.groups[0].index.index
        if raw_column in keep:
            failed = ledger.check_temporary(
                2 * index.n_unique * np.dtype(np.float64).itemsize,
                lineage,
                "compact intercept table exceeds budget",
            )
            if failed is not None:
                return [], failed
            table = np.ones((index.n_unique, 1), dtype=np.float64)
            failed = ledger.add_table(int(table.nbytes), lineage)
            if failed is not None:
                return [], failed
            blocks.append(LookupBlock(output_column, table, index))
            output_column += 1
        raw_column += 1
    seen_factor = False
    for param in spec.parametric_terms:
        index = _single_index(setup, param.name)
        values = _value_for_name(index, param.name)
        if param.name in spec.factor_info:
            drop_reference = spec.has_intercept or seen_factor
            encoded_width = len(spec.factor_info[param.name]) - int(drop_reference)
            failed = ledger.check_temporary(
                _prospective_matrix_bytes(index.n_unique, encoded_width, ()),
                lineage,
                "compact parametric predictor workspace exceeds budget",
            )
            if failed is not None:
                return [], failed
            encoded, _ = predict_matrix._encode_factor(
                values,
                spec.factor_info[param.name],
                drop_reference,
                param.name in spec.ordered_factors,
            )
            seen_factor = True
        else:
            failed = ledger.check_temporary(
                _prospective_matrix_bytes(index.n_unique, 1, ()),
                lineage,
                "compact parametric predictor workspace exceeds budget",
            )
            if failed is not None:
                return [], failed
            encoded = np.asarray(values, dtype=np.float64)[:, None]
        selected = [
            local for local in range(encoded.shape[1]) if raw_column + local in keep
        ]
        if selected:
            table = encoded[:, selected]
            failed = ledger.add_table(
                int(table.nbytes),
                lineage,
                live_temporary_bytes=int(encoded.nbytes + table.nbytes),
            )
            if failed is not None:
                return [], failed
            blocks.append(LookupBlock(output_column, table, index))
            output_column += len(selected)
        raw_column += encoded.shape[1]
    if raw_column != term.n_coefs_raw or output_column != term.col_start + term.n_coefs:
        raise RuntimeError("compact parametric block does not match frozen coordinates")
    return blocks, None


def build_continuous_discrete_design(
    setup: ContinuousDiscreteSetup,
    source: RowSource,
    *,
    control: DiscreteDesignControl | None = None,
) -> FrozenDiscreteDesign | DiscreteDesignUnavailable:
    """Evaluate frozen bases on compact representatives without source scans."""
    if control is None:
        control = DiscreteDesignControl()
    lineage = DiscreteDesignLineage(
        setup.source_fingerprint, setup.basis_fingerprint, "continuous"
    )
    if source.n_rows != setup.n_obs:
        raise ValueError("continuous setup and RowSource have different row counts")
    if source.fingerprint() != setup.source_fingerprint:
        raise RuntimeError("RowSource changed before continuous descriptor setup")
    ledger = _Ledger(control)
    ledger.table_bytes = setup.retained_table_bytes
    ledger.selector_bytes = setup.compact.retained_selector_bytes
    ledger.selector_storage_bytes = sum(
        group.index.index.selector_bytes for group in setup.compact.groups
    )
    ledger.construction_peak_bytes = setup.construction_peak_bytes
    if ledger.table_bytes > control.budget.max_table_bytes:
        return DiscreteDesignUnavailable(
            "retained compact setup exceeds descriptor table budget",
            ledger.table_bytes,
            control.budget.max_table_bytes,
            lineage,
        )
    if ledger.selector_bytes > control.budget.max_selector_bytes:
        return DiscreteDesignUnavailable(
            "retained compact selectors exceed descriptor selector budget",
            ledger.selector_bytes,
            control.budget.max_selector_bytes,
            lineage,
        )

    blocks: list[DiscreteBlock] = []
    for term in setup.predict_spec.coef_map.terms:
        if term.term_type == "parametric":
            parametric, failed = _parametric_blocks(setup, term, ledger, lineage)
            if failed is not None:
                return failed
            blocks.extend(parametric)
            continue
        if term.smooth is None:
            return DiscreteDesignUnavailable(
                "term has no frozen smooth evaluator", 0, 0, lineage
            )
        if isinstance(
            term.smooth, (TensorProductSmooth, FactorBySmooth, NumericBySmooth)
        ):
            product = _product_block(term, setup, ledger, lineage)
            if isinstance(product, DiscreteDesignUnavailable):
                return product
            blocks.append(product)
            continue
        index = _index_for_names(setup, tuple(term.smooth.spec.variables))
        transforms = []
        if term.Z_centering is not None:
            transforms.append(term.Z_centering.shape[1])
        if term.del_index:
            transforms.append(term.n_coefs)
        failed = ledger.check_temporary(
            _prospective_matrix_bytes(
                index.n_unique,
                term.n_coefs_raw,
                tuple(transforms),
                evaluator_bytes=_smooth_evaluator_bytes(
                    term.smooth, index.n_unique, term.n_coefs_raw
                ),
            ),
            lineage,
            "compact smooth predictor workspace exceeds budget",
        )
        if failed is not None:
            return failed
        raw = term.smooth.predict_matrix(_values(index))
        table = setup.predict_spec.coef_map.transform_X(raw, term)
        failed = ledger.add_table(
            int(table.nbytes),
            lineage,
            live_temporary_bytes=int(raw.nbytes + table.nbytes),
        )
        if failed is not None:
            return failed
        blocks.append(LookupBlock(term.col_start, table, index))

    block_tuple = tuple(blocks)
    memory = _memory_plan(block_tuple, setup.predict_spec.total_coefs, ledger, control)
    batch_bytes = max(memory.host_batch_bytes, memory.device_action_bytes)
    if batch_bytes > control.budget.max_batch_bytes:
        return DiscreteDesignUnavailable(
            "continuous indexed operator batch workspace exceeds budget",
            batch_bytes,
            control.budget.max_batch_bytes,
            lineage,
        )
    result = FrozenDiscreteDesign(
        block_tuple,
        setup.n_obs,
        setup.predict_spec.total_coefs,
        lineage,
        control,
        memory,
    )
    if source.fingerprint() != setup.source_fingerprint:
        raise RuntimeError("RowSource changed during continuous descriptor setup")
    return result
