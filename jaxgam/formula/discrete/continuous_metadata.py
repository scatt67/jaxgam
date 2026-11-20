"""Row-free record of a fitted continuous discretization.

The record mirrors what pinned mgcv 1.9-3 ``bam(discrete=TRUE)`` builds in
``discrete.mf``/``compress.df`` (``R/bam.r``): one *group* per jointly
discretized smooth, tensor margin, by variable or parametric covariate (the
response is compressed too, though no term reads it), and for every model
term the ordered list of groups whose compact tables form its ``Xd``
margins (``ts``/``dt``/``ks``).  Each group records the policy, whether
``compress.df`` actually rounded it, its grid resolution and unique-row
threshold, its metric ranges and its compact row count ``q``.

Retained storage never scales with the number of observations.  The compact
representative values (in fitted ``discrete.mf`` order) are copied only when
``q`` is at most ``max_retained_representatives``; larger groups keep their
bounded descriptor instead.  For a rounded group that descriptor is exact:
every representative of metric variable ``v`` lies on
``seq(low, high, length=points_per_metric)``.  For an unrounded group the
representatives are the ``q`` distinct observed rows inside the recorded
ranges.

This module is Phase 1: NumPy only, no JAX imports.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from jaxgam.data.continuous_discretize import RangeGridIndex, range_grid
from jaxgam.data.continuous_request import RequestCompressionSpec, RequestGroupSpec
from jaxgam.data.discretize import DiscretizationBudget
from jaxgam.formula.discrete.continuous_setup import ContinuousDiscreteSetup

GroupRole = Literal["by", "tensor_margin", "smooth", "parametric", "response"]
TermKind = Literal["intercept", "parametric", "smooth"]

#: The only training grid policy: pinned ``compress.df``.
TRAINING_GRID_POLICY = "mgcv_compress_df"


@dataclass(frozen=True, eq=False)
class DiscretizationGroupMetadata:
    """One jointly discretized group of the fitted compact setup.

    Equality is identity: the record owns NumPy representative arrays.
    """

    variables: tuple[str, ...]
    factor_variables: tuple[str, ...]
    role: GroupRole
    policy: str
    requested_m: int | None
    points_per_metric: int
    unique_threshold: int
    discretized: bool
    n_unique: int
    metric_ranges: tuple[tuple[str, float, float], ...]
    factor_level_counts: tuple[tuple[str, int], ...]
    representatives: tuple[npt.NDArray[Any], ...] | None

    def __post_init__(self) -> None:
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise ValueError("discretization group needs distinct variables")
        metric = {name for name, _, _ in self.metric_ranges}
        if metric | set(self.factor_variables) != set(self.variables) or (
            metric & set(self.factor_variables)
        ):
            raise ValueError("group ranges and factor variables must cover variables")
        if self.n_unique <= 0 or self.unique_threshold <= 0:
            raise ValueError("group row counts must be positive")
        if not self.discretized and self.n_unique > self.unique_threshold:
            raise ValueError("an unrounded group cannot exceed its unique threshold")
        if self.representatives is not None:
            if len(self.representatives) != len(self.variables):
                raise ValueError("representatives need one column per variable")
            owned = []
            for values in self.representatives:
                array = np.array(values, copy=True)
                if array.shape != (self.n_unique,):
                    raise ValueError("representative columns must have q rows")
                array.setflags(write=False)
                owned.append(array)
            object.__setattr__(self, "representatives", tuple(owned))

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        for values in self.representatives or ():
            values.setflags(write=False)

    @property
    def metric_variables(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.metric_ranges)

    @property
    def representatives_retained(self) -> bool:
        return self.representatives is not None

    def range_of(self, variable: str) -> tuple[float, float]:
        for name, low, high in self.metric_ranges:
            if name == variable:
                return low, high
        raise KeyError(f"'{variable}' is not a metric variable of this group")

    def grid(self, variable: str) -> npt.NDArray[np.float64] | None:
        """The rounding grid of a metric variable, or ``None`` if exact."""
        if not self.discretized:
            return None
        low, high = self.range_of(variable)
        return range_grid(low, high, self.points_per_metric)

    def representatives_of(self, variable: str) -> npt.NDArray[Any] | None:
        """Representative values of one variable in fitted compact order."""
        if self.representatives is None:
            return None
        return self.representatives[self.variables.index(variable)]


@dataclass(frozen=True)
class DiscreteTermMetadata:
    """The ordered ``Xd`` margins (``ts``/``dt``/``ks``) of one model term.

    Smooth labels match ``SmoothInfo.label`` and mgcv's per-level labels
    (for example ``te(x,w):gmid``).  A by variable is the first margin, as in
    mgcv's discrete design.  The intercept has no data margin.
    ``margin_roles`` give each margin's role in this term, and
    ``margin_groups`` index the (possibly shared) group in
    ``ContinuousDiscretizationMetadata.groups``.
    """

    label: str
    kind: TermKind
    margins: tuple[tuple[str, ...], ...]
    margin_roles: tuple[GroupRole, ...]
    margin_groups: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.margins) == len(self.margin_roles) == len(self.margin_groups)):
            raise ValueError("term margins, roles and groups must align")


@dataclass(frozen=True, eq=False)
class ContinuousDiscretizationMetadata:
    """Bounded per-group and per-term record of a discrete fit.

    ``prediction_policy`` is the policy actually used for prediction and
    ``requested_prediction_policy`` the one the control asked for.  They
    differ only when ``mgcv_predict_bamd`` was requested for a model without
    smooth terms: pinned ``predict.bam`` then has no ``dinfo`` and predicts
    exactly, which is ``continuous_frozen_basis``; ``prediction_policy_note``
    says why.  Equality is identity, because groups own NumPy arrays.
    """

    strategy: str
    prediction_policy: str
    groups: tuple[DiscretizationGroupMetadata, ...]
    terms: tuple[DiscreteTermMetadata, ...]
    max_retained_representatives: int
    shuffle_seed: int
    requested_prediction_policy: str | None = None
    prediction_policy_note: str | None = None

    def __post_init__(self) -> None:
        if self.requested_prediction_policy is None:
            object.__setattr__(
                self, "requested_prediction_policy", self.prediction_policy
            )

    def group_of(self, variable: str) -> DiscretizationGroupMetadata:
        for group in self.groups:
            if variable in group.variables:
                return group
        raise KeyError(f"No discretization group contains '{variable}'")

    def term(self, label: str) -> DiscreteTermMetadata:
        for term in self.terms:
            if term.label == label:
                return term
        raise KeyError(f"No discrete term labelled '{label}'")

    @property
    def retained_bytes(self) -> int:
        """Resident representative bytes, bounded independently of ``n``."""
        total = 0
        for group in self.groups:
            for values in group.representatives or ():
                total += int(values.nbytes)
                if values.dtype == object:
                    total += sum(sys.getsizeof(item) for item in values)
        return total


def _group_roles(setup: ContinuousDiscreteSetup) -> list[GroupRole]:
    roles: list[GroupRole] = []
    for group in setup.compact.groups:
        if group.role == "parametric" and group.names == (setup.response,):
            roles.append("response")
        else:
            roles.append(group.role)
    return roles


def _group_metadata(
    setup: ContinuousDiscreteSetup, max_retained: int
) -> tuple[DiscretizationGroupMetadata, ...]:
    result = []
    for group, role in zip(setup.compact.groups, _group_roles(setup), strict=True):
        index = group.index
        q = index.n_unique
        representatives = (
            tuple(np.array(values, copy=True) for values in index.values)
            if q <= max_retained
            else None
        )
        counts = index.factor_counts or tuple(
            len(levels) for levels in index.factor_levels
        )
        result.append(
            DiscretizationGroupMetadata(
                variables=index.names,
                factor_variables=index.factor_names,
                role=role,
                policy=TRAINING_GRID_POLICY,
                requested_m=group.m,
                points_per_metric=index.points_per_metric,
                unique_threshold=index.unique_threshold,
                discretized=index.rounded,
                n_unique=q,
                metric_ranges=tuple(
                    (name, float(low), float(high))
                    for name, (low, high) in zip(
                        index.metric_names, index.metric_ranges, strict=True
                    )
                ),
                factor_level_counts=tuple(
                    zip(
                        index.factor_names,
                        (int(count) for count in counts),
                        strict=True,
                    )
                ),
                representatives=representatives,
            )
        )
    return tuple(result)


def _term_metadata(setup: ContinuousDiscreteSetup) -> tuple[DiscreteTermMetadata, ...]:
    group_of = dict(setup.compact.name_group_indices)

    def margin(
        names: tuple[str, ...], role: GroupRole
    ) -> tuple[tuple[str, ...], GroupRole, int]:
        number = group_of[names[0]]
        group_names = setup.compact.groups[number].names
        if set(group_names) != set(names):
            raise RuntimeError("term margin does not match its compact group")
        return group_names, role, number

    spec = setup.predict_spec
    terms: list[DiscreteTermMetadata] = []
    if spec.has_intercept:
        terms.append(DiscreteTermMetadata("(Intercept)", "intercept", (), (), ()))
    for parametric in spec.parametric_terms:
        parts = [margin((parametric.name,), "parametric")]
        terms.append(_term(parametric.name, "parametric", parts))
    for info in spec.smooth_info:
        parts = []
        if info.by_variable is not None:
            parts.append(margin((info.by_variable,), "by"))
        if info.term_type in ("te", "ti"):
            parts.extend(margin((name,), "tensor_margin") for name in info.variables)
        else:
            parts.append(margin(tuple(info.variables), "smooth"))
        terms.append(_term(info.label, "smooth", parts))
    return tuple(terms)


def _term(
    label: str,
    kind: TermKind,
    parts: list[tuple[tuple[str, ...], GroupRole, int]],
) -> DiscreteTermMetadata:
    return DiscreteTermMetadata(
        label,
        kind,
        tuple(part[0] for part in parts),
        tuple(part[1] for part in parts),
        tuple(part[2] for part in parts),
    )


_NO_SMOOTH_NOTE = (
    "mgcv_predict_bamd was requested, but the model has no smooth terms: "
    "pinned bam keeps no dinfo for it and predict.bam evaluates it exactly."
)


def effective_prediction_policy(
    setup: ContinuousDiscreteSetup, requested: str
) -> tuple[str, str | None]:
    """The policy prediction actually follows, and why it differs if it does."""
    if requested == "mgcv_predict_bamd" and not setup.predict_spec.smooth_info:
        return "continuous_frozen_basis", _NO_SMOOTH_NOTE
    return requested, None


def build_discretization_metadata(
    setup: ContinuousDiscreteSetup,
    *,
    prediction_policy: str,
    max_retained_representatives: int,
    shuffle_seed: int,
) -> ContinuousDiscretizationMetadata:
    """Copy the bounded fitted record before compact selectors are released.

    ``prediction_policy`` is the requested policy; the record stores both it
    and the effective policy.
    """
    if max_retained_representatives < 0:
        raise ValueError("max_retained_representatives must be non-negative")
    effective, note = effective_prediction_policy(setup, prediction_policy)
    return ContinuousDiscretizationMetadata(
        strategy="mgcv_range_grid",
        prediction_policy=effective,
        groups=_group_metadata(setup, max_retained_representatives),
        terms=_term_metadata(setup),
        max_retained_representatives=max_retained_representatives,
        shuffle_seed=shuffle_seed,
        requested_prediction_policy=prediction_policy,
        prediction_policy_note=note,
    )


def _observed_levels(index: RangeGridIndex, name: str) -> tuple[Any, ...]:
    """Fitted levels observed in the data, in fitted level order.

    Every observed level occurs among a group's compact representatives,
    and mgcv's model frame drops unused levels before prediction.
    """
    observed = set(index.values[index.names.index(name)].tolist())
    levels = index.factor_levels[index.factor_names.index(name)]
    return tuple(level for level in levels if level in observed)


def build_request_compression(
    setup: ContinuousDiscreteSetup, budget: DiscretizationBudget
) -> RequestCompressionSpec | None:
    """Fitted grouping for ``predict.bamd``; ``None`` without smooth terms.

    ``predict.bam`` dispatches to ``predict.bamd`` only when the fit carries
    ``dinfo``; pinned ``bam`` drops discretization for a model without smooth
    terms, so such a model predicts exactly.  The response group is not part
    of a prediction request (``predict.gam(type="newdata")`` builds the frame
    from the covariate-only ``pred.formula``).  Factor-only groups pass their
    values through unchanged; a group mixing factor and metric variables
    carries its observed fitted levels, to exclude rows with missing or
    unseen factor values from compression.
    """
    if not setup.predict_spec.smooth_info:
        return None
    groups = []
    for group, role in zip(setup.compact.groups, _group_roles(setup), strict=True):
        if role == "response":
            continue
        index = group.index
        mixed = bool(index.factor_names) and bool(index.metric_names)
        groups.append(
            RequestGroupSpec(
                index.names,
                index.factor_names,
                tuple(_observed_levels(index, name) for name in index.factor_names)
                if mixed
                else (),
            )
        )
    return RequestCompressionSpec(tuple(groups), budget)
