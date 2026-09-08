"""Conformance tests for family-owned bounded execution primitives."""

from __future__ import annotations

import ast
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxgam.families.base import REAL, ExponentialFamily
from jaxgam.families.execution_inventory import FAMILY_EXECUTION_INVENTORY
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.registry import family_registry
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionLineage,
    FamilyExecutionParameters,
    batch_execution_summary,
    batch_saturated_loglikelihood,
    batch_working_quantities,
    finalize_execution_summary,
    merge_execution_summaries,
)
from jaxgam.fitting.pirls import canonical_working_quantities
from jaxgam.links.links import IdentityLink, Link, LogLink
from jaxgam.links.registry import link_registry
from tests.helpers import r_available
from tests.tolerances import STRICT


class _ScaledIdentityLink(Link):
    """Stateful link used to prove config is not keyed only by identity."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def link(self, mu):
        return mu / self.scale

    def inverse(self, eta):
        return eta * self.scale

    def derivative(self, mu):
        xp = jnp if isinstance(mu, jax.Array) else np
        return xp.ones_like(mu) / self.scale


class _UnregisteredQuadraticFamily(ExponentialFamily):
    """A family never added to the registry or a driver allowlist."""

    family_name = "unregistered_quadratic"
    response_support = REAL
    scale_known = True
    canonical_link_cls = _ScaledIdentityLink

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__(_ScaledIdentityLink(scale))

    @property
    def default_link(self) -> Link:
        return IdentityLink()

    def variance(self, mu):
        xp = jnp if isinstance(mu, jax.Array) else np
        return xp.ones_like(mu)

    def dvar(self, mu):
        xp = jnp if isinstance(mu, jax.Array) else np
        return xp.zeros_like(mu)

    def saturated_loglik(self, y, wt, scale, *, max_y: int = 0):  # noqa: ARG002
        return jnp.sum(jnp.where(wt > 0, -0.5 * jnp.log(scale / wt), 0.0))

    def deviance_resids(self, y, mu, wt):
        xp = jnp if isinstance(y, jax.Array) else np
        return xp.sign(y - mu) * xp.sqrt(wt * (y - mu) ** 2)

    def deviance_contributions(self, y, mu, wt):
        return wt * (y - mu) ** 2

    def deviance_derivative_contributions(self, y, mu, wt):
        return wt * (y - mu) ** 2

    def aic(self, y, mu, wt, scale):  # noqa: ARG002
        return float(np.sum(wt * (y - mu) ** 2))

    def _initialize_impl(self, y, wt):  # noqa: ARG002
        return y.copy()

    def valid_mu(self, mu):
        xp = jnp if isinstance(mu, jax.Array) else np
        return xp.isfinite(mu)

    def valid_eta(self, eta):
        xp = jnp if isinstance(eta, jax.Array) else np
        return xp.isfinite(eta)


class _VectorSummaryFamily(_UnregisteredQuadraticFamily):
    """Proves summary pytrees are family-owned rather than fixed scalars."""

    def execution_summary_from_batch(self, y, wt, valid):
        base = super().execution_summary_from_batch(y, wt, valid)
        safe_y = jnp.where(valid, y, 0.0)
        return (*base, jnp.array([jnp.sum(safe_y), jnp.max(safe_y)]))

    def merge_execution_summaries(self, left, right):
        return (
            left[0] + right[0],
            left[1] + right[1],
            left[2] + right[2],
            jnp.logical_and(left[3], right[3]),
            jnp.array([left[4][0] + right[4][0], jnp.maximum(left[4][1], right[4][1])]),
        )

    def finalize_execution_summary(self, summary):
        base = super().finalize_execution_summary(summary[:4])
        base["response_sum"] = float(summary[4][0])
        base["response_max"] = float(summary[4][1])
        return base


class _NoObservedInformationFamily(_UnregisteredQuadraticFamily):
    def execution_capabilities(self):
        return replace(super().execution_capabilities(), observed_information=False)


@pytest.mark.parametrize(
    ("family", "y", "mu"),
    [
        (Gaussian(), [0.0, 2.0], [0.5, 1.0]),
        (Binomial(), [0.0, 1.0], [0.25, 0.75]),
        (Poisson(), [0.0, 2.0], [0.8, 1.5]),
        (Gamma(), [1.0, 2.0], [0.8, 1.5]),
        (NegativeBinomial(theta=2.0, fixed=True), [0.0, 2.0], [0.8, 1.5]),
    ],
)
def test_direct_deviance_matches_legacy_residual_square(family, y, mu) -> None:
    y_array = jnp.asarray(y)
    mu_array = jnp.asarray(mu)
    weight = jnp.asarray([1.0, 2.0])
    direct = family.deviance_contributions(y_array, mu_array, weight)
    legacy = family.deviance_resids(y_array, mu_array, weight) ** 2
    np.testing.assert_allclose(direct, legacy, rtol=STRICT.rtol, atol=STRICT.atol)


def test_generic_working_primitive_reuses_dense_fisher_arithmetic() -> None:
    family = Poisson()
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    X = jnp.array([[1.0], [1.0]])
    y = jnp.array([1.0, 3.0])
    weight = jnp.array([1.0, 0.0])
    offset = jnp.array([0.2, -0.1])
    beta = jnp.array([0.4])
    result = batch_working_quantities(
        X, y, weight, offset, jnp.array([True, True]), beta, parameters, family, context
    )
    eta = X @ beta + offset
    mu = family.link.inverse(eta)
    dense_weight, dense_z = canonical_working_quantities(
        family, y, mu, eta, weight, offset
    )
    np.testing.assert_allclose(
        result.fisher_weight, dense_weight, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.working_response, dense_z, rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_dense_default_working_helper_is_bitwise_unchanged() -> None:
    """The optional explicit-theta hook must not perturb existing dense calls."""
    family = Poisson()
    y = jnp.array([1.0, 3.0])
    mu = jnp.array([1.2, 0.9])
    eta = jnp.log(mu) + jnp.array([0.2, -0.1])
    weight = jnp.array([1.0, 0.0])
    offset = jnp.array([0.2, -0.1])
    actual = jax.jit(
        lambda y, mu, eta, weight, offset: canonical_working_quantities(
            family, y, mu, eta, weight, offset
        )
    )(y, mu, eta, weight, offset)
    expected = jax.jit(
        lambda y, mu, eta, weight, offset: (
            family.working_weights(mu, weight),
            family.working_response(y, mu, eta - offset),
        )
    )(y, mu, eta, weight, offset)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_padding_uses_safe_eta_before_inverse_link_evaluation() -> None:
    family = Gamma()
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    result = batch_working_quantities(
        jnp.array([[1.0], [jnp.nan]]),
        jnp.array([1.0, jnp.nan]),
        jnp.array([1.0, jnp.nan]),
        jnp.array([0.0, jnp.nan]),
        jnp.array([True, False]),
        jnp.array([1.0]),
        parameters,
        family,
        context,
    )
    assert bool(result.domain_ok)
    assert bool(jnp.all(jnp.isfinite(result.mu)))
    assert float(result.working_weight[1]) == 0.0


def test_negative_real_weight_and_missing_capability_fail_closed() -> None:
    family = Poisson()
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    invalid = batch_working_quantities(
        jnp.ones((1, 1)),
        jnp.ones(1),
        jnp.array([-1.0]),
        jnp.zeros(1),
        jnp.array([True]),
        jnp.zeros(1),
        parameters,
        family,
        context,
    )
    assert not bool(invalid.domain_ok)

    unsupported = _NoObservedInformationFamily()
    unsupported_context = FamilyExecutionContext.from_family(unsupported)
    unsupported_parameters = FamilyExecutionParameters.from_snapshot(
        unsupported.execution_parameter_snapshot()
    )
    with pytest.raises(NotImplementedError, match="observed_information"):
        batch_working_quantities(
            jnp.ones((1, 1)),
            jnp.ones(1),
            jnp.ones(1),
            jnp.zeros(1),
            jnp.array([True]),
            jnp.zeros(1),
            unsupported_parameters,
            unsupported,
            unsupported_context,
        )


@pytest.mark.parametrize(
    ("family", "y", "beta", "expected"),
    [
        (Poisson(), 1.0, 0.0, 1.0),
        (Binomial(), 0.5, 0.0, 0.25),
        (Gamma(link=LogLink()), 1.0, 0.0, 1.0),
        (NegativeBinomial(theta=2.0, fixed=True), 1.0, 0.0, 2.0 / 3.0),
    ],
)
def test_exact_fit_observed_information_uses_smooth_deviance(
    family, y: float, beta: float, expected: float
) -> None:
    """No reporting clamp may halve curvature at an exact fit."""
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    result = batch_working_quantities(
        jnp.ones((1, 1)),
        jnp.array([y]),
        jnp.ones(1),
        jnp.zeros(1),
        jnp.array([True]),
        jnp.array([beta]),
        parameters,
        family,
        context,
    )
    np.testing.assert_allclose(
        result.observed_weight, [expected], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        result.fisher_weight, [expected], rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_noncanonical_gamma_observed_hessian_matches_nearby_finite_difference() -> None:
    family = Gamma(link=LogLink())
    y = jnp.array([1.0])
    wt = jnp.ones(1)

    def deviance(eta):
        mu = family.link.inverse(eta)
        return jnp.sum(family.deviance_derivative_contributions(y, mu, wt))

    eta = jnp.array([0.03])
    observed = 0.5 * jax.hessian(deviance)(eta)[0, 0]
    step = 1e-4
    finite_difference = (
        0.5
        * (deviance(eta + step) - 2.0 * deviance(eta) + deviance(eta - step))
        / step**2
    )
    np.testing.assert_allclose(observed, finite_difference, rtol=2e-7, atol=2e-7)


def test_unregistered_family_executes_and_mutation_invalidates_lineage() -> None:
    family = _UnregisteredQuadraticFamily(scale=1.0)

    def prepared_for(current_family):
        return SimpleNamespace(
            source_fingerprint="source",
            basis_fingerprint="basis",
            fitting=SimpleNamespace(
                family_name=current_family.family_name,
                link_name=type(current_family.link).__qualname__,
                family_execution_static_config=current_family.execution_static_config(),
                family_parameter_snapshot=current_family.execution_parameter_snapshot(),
            ),
        )

    prepared = prepared_for(family)
    lineage = FamilyExecutionLineage.from_prepared(prepared, family)
    context = lineage.context
    parameters = FamilyExecutionParameters.from_snapshot(lineage.parameters)
    args = (
        jnp.array([[1.0]]),
        jnp.array([3.0]),
        jnp.array([1.0]),
        jnp.array([0.0]),
        jnp.array([True]),
        jnp.array([1.0]),
        parameters,
        family,
    )
    before = batch_working_quantities(*args, context).deviance
    family.link.scale = 2.0
    with pytest.raises(RuntimeError, match="static configuration changed"):
        lineage.validate(prepared, family)
    refreshed = FamilyExecutionLineage.from_prepared(prepared_for(family), family)
    after = batch_working_quantities(*args, refreshed.context).deviance
    assert float(before) != float(after)


def test_summary_rejects_nonfinite_real_row_and_merges_nb_metadata() -> None:
    family = NegativeBinomial(theta=2.0, fixed=True)
    context = FamilyExecutionContext.from_family(family)
    bad = batch_execution_summary(
        jnp.array([1.0, jnp.nan]),
        jnp.ones(2),
        jnp.array([True, True]),
        family,
        context,
    )
    with pytest.raises(ValueError, match="non-finite real rows"):
        finalize_execution_summary(bad, family)

    left = batch_execution_summary(
        jnp.array([0.0, 3.0]), jnp.ones(2), jnp.array([True, True]), family, context
    )
    right = batch_execution_summary(
        jnp.array([2.0]), jnp.ones(1), jnp.array([True]), family, context
    )
    summary = merge_execution_summaries(left, right, family, context)
    final = finalize_execution_summary(summary, family)
    assert final["max_count"] == 3.0
    assert final["integer_counts"]
    assert final["n_valid_rows"] == 3.0


def test_family_owned_vector_summary_pytree_survives_merge_and_finalize() -> None:
    family = _VectorSummaryFamily()
    context = FamilyExecutionContext.from_family(family)
    left = batch_execution_summary(
        jnp.array([1.0, 2.0]), jnp.ones(2), jnp.array([True, True]), family, context
    )
    right = batch_execution_summary(
        jnp.array([3.0]), jnp.ones(1), jnp.array([True]), family, context
    )
    final = finalize_execution_summary(
        merge_execution_summaries(left, right, family, context), family
    )
    assert final["response_sum"] == 6.0
    assert final["response_max"] == 3.0


def test_saturated_likelihood_and_inventory_are_jittable_complete() -> None:
    family = Poisson()
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    value, valid = batch_saturated_loglikelihood(
        jnp.array([0.0, 2.0]),
        jnp.ones(2),
        jnp.array([True, True]),
        jnp.array(1.0),
        parameters,
        family,
        context,
    )
    assert bool(valid)
    assert bool(jnp.isfinite(value))
    assert len(FAMILY_EXECUTION_INVENTORY) == 48
    assert all(entry.dense_status for entry in FAMILY_EXECUTION_INVENTORY)
    regular = [entry for entry in FAMILY_EXECUTION_INVENTORY if entry.family != "nb"]
    assert len(regular) == 32
    assert all(entry.r_constructor_status == "accepted" for entry in regular)
    inverse_squared = next(
        entry
        for entry in regular
        if entry.family == "gaussian" and entry.link == "inverse_squared"
    )
    assert inverse_squared.r_constructor_link == "1/mu^2"
    assert not inverse_squared.r_advertised
    assert inverse_squared.efs_status == "implementation_missing"

    nb = [entry for entry in FAMILY_EXECUTION_INVENTORY if entry.family == "nb"]
    assert len(nb) == 16
    assert sum(entry.r_constructor_status == "accepted" for entry in nb) == 6
    assert sum(entry.efs_status == "r_rejected" for entry in nb) == 10
    assert all(
        entry.r_advertised for entry in nb if entry.r_constructor_status == "accepted"
    )
    nb_log = next(
        entry
        for entry in FAMILY_EXECUTION_INVENTORY
        if entry.family == "nb"
        and entry.link == "log"
        and entry.parameter_mode == "estimated_theta"
    )
    assert nb_log.default_link
    assert not nb_log.mathematical_canonical
    assert not nb_log.legacy_route_is_canonical
    assert nb_log.efs_status == "implementation_missing"
    assert not nb_log.numerical_boundaries


@pytest.mark.skipif(not r_available(), reason="pinned R/mgcv oracle unavailable")
def test_execution_inventory_matches_pinned_r_constructors_and_evidence() -> None:
    """Tie registered cells to pinned R construction and checked-in tests."""
    links = link_registry.available
    families = family_registry.available
    expected_cells: set[tuple[str, str, str]] = set()
    for family in families:
        modes = ("fixed_theta", "estimated_theta") if family == "nb" else ("none",)
        expected_cells.update((family, link, mode) for link in links for mode in modes)
    entries = {
        (entry.family, entry.link, entry.parameter_mode): entry
        for entry in FAMILY_EXECUTION_INVENTORY
    }
    assert set(entries) == expected_cells

    for entry in FAMILY_EXECUTION_INVENTORY:
        family_cls = family_registry.get_class(entry.family)
        if entry.family == "nb":
            instance = family_cls(
                theta=1.2,
                fixed=entry.parameter_mode == "fixed_theta",
                link=entry.link,
            )
        else:
            instance = family_cls(link=entry.link)
        assert instance.family_name.lower() == entry.family

    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        output = root / "constructors.csv"
        script = root / "constructors.R"
        script.write_text(
            f"""suppressPackageStartupMessages(library(mgcv))
if (as.character(getRversion()) != "4.5.2" ||
    packageVersion("mgcv") != package_version("1.9.3")) stop("pinned R/mgcv required")
links <- c("identity", "log", "logit", "inverse", "probit", "cloglog", "sqrt", "1/mu^2")
regular <- c(gaussian="gaussian", binomial="binomial", poisson="poisson", gamma="Gamma")
rows <- list(); i <- 0L
for (family_name in names(regular)) for (link_name in links) {{
  i <- i + 1L
  accepted <- !inherits(
    try(do.call(regular[[family_name]], list(link=link_name)), silent=TRUE),
    "try-error"
  )
  rows[[i]] <- data.frame(family=family_name, link=link_name, accepted=accepted)
}}
for (link_name in links) {{
  i <- i + 1L
  accepted <- !inherits(
    try(
      eval(parse(text=sprintf("mgcv::nb(theta=1.2, link=%s)", shQuote(link_name)))),
      silent=TRUE
    ),
    "try-error"
  )
  rows[[i]] <- data.frame(family="nb", link=link_name, accepted=accepted)
}}
write.csv(do.call(rbind, rows), {str(output)!r}, row.names=FALSE)
"""
        )
        subprocess.run(
            ["Rscript", str(script)], check=True, capture_output=True, text=True
        )
        r_rows = pd.read_csv(output)

    r_accepts = {
        (str(row.family), str(row.link)): bool(row.accepted)
        for row in r_rows.itertuples(index=False)
    }
    assert len(r_accepts) == 40
    for entry in FAMILY_EXECUTION_INVENTORY:
        assert r_accepts[(entry.family, entry.r_constructor_link)] == (
            entry.r_constructor_status == "accepted"
        )

    repository = Path(__file__).resolve().parents[2]
    for entry in FAMILY_EXECUTION_INVENTORY:
        for evidence in entry.evidence:
            if not evidence.startswith("tests/"):
                continue
            test_path, qualified_name = evidence.split("::", maxsplit=1)
            parts = qualified_name.split("::")
            tree = ast.parse((repository / test_path).read_text())
            if len(parts) == 1:
                found = any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == parts[0]
                    for node in tree.body
                )
            elif len(parts) == 2:
                found = any(
                    isinstance(node, ast.ClassDef)
                    and node.name == parts[0]
                    and any(
                        isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and method.name == parts[1]
                        for method in node.body
                    )
                    for node in tree.body
                )
            else:
                found = False
            assert found, evidence
