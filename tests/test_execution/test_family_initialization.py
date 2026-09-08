"""Tests for opt-in bounded R-style family initialization primitives."""

from __future__ import annotations

import subprocess
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.data.source import ArrayRowSource
from jaxgam.execution.family_initialization import select_initial_working_state_cpu
from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.family_execution import (
    FamilyExecutionContext,
    FamilyExecutionLineage,
    FamilyExecutionParameters,
    batch_initial_working_quantities,
)
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.jax_utils import array_module
from jaxgam.links.links import (
    CloglogLink,
    IdentityLink,
    InverseLink,
    LogitLink,
    LogLink,
    ProbitLink,
    SqrtLink,
)
from tests.helpers import _AssertCollector, r_available
from tests.tolerances import MODERATE, STRICT


@pytest.mark.parametrize(
    ("family", "y", "weight"),
    [
        (Gaussian(IdentityLink()), [-2.0, 0.25], [1.0, 1.0]),
        (Gaussian(LogLink()), [1e-300, 2.0], [1.0, 1.0]),
        (Gaussian(InverseLink()), [-2.0, 0.5], [1.0, 1.0]),
        (Binomial(LogitLink()), [0.0, 1.0], [1.0, 0.0]),
        (Binomial(ProbitLink()), [0.0, 1.0], [1.0, 0.0]),
        (Binomial(CloglogLink()), [0.0, 1.0], [1.0, 0.0]),
        (Binomial(LogLink()), [0.0, 1.0], [1.0, 0.0]),
        (Poisson(LogLink()), [0.0, 2.0], [1.0, 0.0]),
        (Poisson(IdentityLink()), [0.0, 2.0], [1.0, 0.0]),
        (Poisson(SqrtLink()), [0.0, 2.0], [1.0, 0.0]),
        (Gamma(InverseLink()), [0.25, 2.0], [1.0, 0.0]),
        (Gamma(LogLink()), [0.25, 2.0], [1.0, 0.0]),
        (Gamma(IdentityLink()), [0.25, 2.0], [1.0, 0.0]),
    ],
)
def test_initial_working_state_matches_family_owned_mustart_and_strict_link(
    family, y, weight
) -> None:
    """All pinned-R advertised regular cells start from per-row mustart."""
    y_array = np.asarray(y)
    weight_array = np.asarray(weight)
    state = family.initial_working_state_cpu(
        y_array, weight_array, np.ones(len(y_array), dtype=bool)
    )
    normalized = family.execution_initial_response_cpu(y_array, weight_array)
    expected_mustart = family.execution_initial_mustart_cpu(normalized, weight_array)
    expected_eta = family.link.initial_link_cpu(expected_mustart)
    expected_mu = family.link.inverse(expected_eta)
    assert state.input_ok
    assert state.domain_ok
    np.testing.assert_allclose(
        state.mustart, expected_mustart, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        state.eta, expected_eta, rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        state.mu, expected_mu, rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_initial_working_state_masks_nan_padding_but_not_real_input_errors() -> None:
    family = Gamma(InverseLink())
    padded = family.initial_working_state_cpu(
        np.array([0.5, np.nan]),
        np.array([1.0, np.nan]),
        np.array([True, False]),
    )
    assert padded.input_ok
    assert padded.domain_ok
    assert np.all(np.isfinite(padded.eta))
    assert np.all(np.isfinite(padded.mu))

    assert (
        not Gaussian(LogLink())
        .initial_working_state_cpu(np.array([-1.0]), np.ones(1), np.ones(1, dtype=bool))
        .input_ok
    )
    assert (
        not Gaussian(InverseLink())
        .initial_working_state_cpu(np.array([0.0]), np.ones(1), np.ones(1, dtype=bool))
        .input_ok
    )
    assert (
        not Gamma()
        .initial_working_state_cpu(np.array([0.0]), np.ones(1), np.ones(1, dtype=bool))
        .input_ok
    )


def test_binomial_zero_weight_normalization_matches_r_initialize() -> None:
    state = Binomial().initial_working_state_cpu(
        np.array([np.nan, 1.0]),
        np.array([0.0, 1.0]),
        np.array([True, True]),
    )
    assert state.input_ok
    assert state.domain_ok
    np.testing.assert_allclose(
        state.mustart, [0.5, 0.75], rtol=STRICT.rtol, atol=STRICT.atol
    )
    with pytest.raises(ValueError, match="read-only"):
        state.eta[0] = 0.0


def test_binomial_zero_weight_nan_is_normalized_before_jit_support_checks() -> None:
    """stats::binomial permits an otherwise invalid response at weight zero."""
    family = Binomial(LogitLink())
    y = jnp.asarray([jnp.nan, 1.0])
    weight = jnp.asarray([0.0, 1.0])
    state = family.initial_working_state_cpu(
        np.asarray(y), np.asarray(weight), np.ones(2, bool)
    )
    result = batch_initial_working_quantities(
        y,
        weight,
        jnp.zeros(2),
        jnp.ones(2, dtype=bool),
        jnp.asarray(state.eta),
        FamilyExecutionParameters.from_snapshot(family.execution_parameter_snapshot()),
        family,
        FamilyExecutionContext.from_family(family),
    )
    assert bool(result.input_ok)
    assert bool(result.domain_ok)
    assert bool(result.working_system_admissible)
    np.testing.assert_array_equal(np.asarray(result.fisher_weight[:1]), [0.0])
    np.testing.assert_array_equal(np.asarray(result.newton_response[:1]), [0.0])


class _ZeroMuEtaIdentity(IdentityLink):
    """An opt-in diagnostic link whose positive rows are legitimately not good."""

    def mu_eta(self, eta):
        return array_module(eta).zeros_like(eta)


class _InfiniteMuEtaIdentity(IdentityLink):
    """An invalid working link used to prove positive rows fail closed."""

    def mu_eta(self, eta):
        return array_module(eta).full_like(eta, np.inf)


class _NonfiniteD2Identity(IdentityLink):
    """Makes only the unselected alpha/Newton diagnostic non-finite."""

    def second_derivative(self, mu):
        return array_module(mu).full_like(mu, np.inf)


class _ZeroVarianceGaussian(Gaussian):
    """An invalid variance contract used to prove positive rows fail closed."""

    family_name = "zero_variance_gaussian"

    def variance(self, mu):
        return array_module(mu).zeros_like(mu)


@pytest.mark.parametrize(
    "family",
    [Gaussian(_InfiniteMuEtaIdentity()), _ZeroVarianceGaussian(IdentityLink())],
)
def test_positive_rows_with_invalid_working_inputs_fail_closed(family) -> None:
    result = batch_initial_working_quantities(
        jnp.asarray([1.0]),
        jnp.ones(1),
        jnp.zeros(1),
        jnp.ones(1, dtype=bool),
        jnp.asarray([1.0]),
        FamilyExecutionParameters.from_snapshot(family.execution_parameter_snapshot()),
        family,
        FamilyExecutionContext.from_family(family),
    )
    assert bool(result.input_ok)
    assert bool(result.domain_ok)
    assert not bool(result.working_inputs_ok)
    assert not bool(result.working_system_admissible)


def test_zero_mu_eta_and_empty_batches_are_neutral_not_invalid() -> None:
    """mgcv excludes finite zero mu.eta rows; a global driver owns total count."""
    family = Gaussian(_ZeroMuEtaIdentity())
    result = batch_initial_working_quantities(
        jnp.asarray([1.0]),
        jnp.ones(1),
        jnp.zeros(1),
        jnp.ones(1, dtype=bool),
        jnp.asarray([1.0]),
        FamilyExecutionParameters.from_snapshot(family.execution_parameter_snapshot()),
        family,
        FamilyExecutionContext.from_family(family),
    )
    assert bool(result.working_inputs_ok)
    assert bool(result.working_system_admissible)
    assert int(result.informative_count) == 0
    np.testing.assert_array_equal(np.asarray(result.fisher_weight), [0.0])


def test_selected_fisher_system_ignores_unselected_nonfinite_newton_diagnostic() -> (
    None
):
    """Retain raw diagnostics without making the selected Fisher W/z unusable."""
    family = Gaussian(_NonfiniteD2Identity())
    result = batch_initial_working_quantities(
        jnp.asarray([2.0]),
        jnp.ones(1),
        jnp.zeros(1),
        jnp.ones(1, dtype=bool),
        jnp.asarray([1.0]),
        FamilyExecutionParameters.from_snapshot(family.execution_parameter_snapshot()),
        family,
        FamilyExecutionContext.from_family(family),
    )
    assert bool(result.fisher_system_ok)
    assert not bool(result.newton_system_ok)
    assert bool(result.observed_information_ok)
    assert bool(result.working_system_admissible)
    assert np.isinf(np.asarray(result.newton_alpha_raw)[0])
    np.testing.assert_allclose(result.fisher_weight, [1.0])


@pytest.mark.parametrize("fixed", [True, False])
def test_nb_parameter_mutation_invalidates_initial_working_lineage(fixed) -> None:
    """Both fixed and estimated theta snapshots reject a later theta mutation."""
    family = NegativeBinomial(theta=2.0, fixed=fixed)
    snapshot = family.execution_parameter_snapshot()
    prepared = SimpleNamespace(
        source_fingerprint="source",
        basis_fingerprint="basis",
        fitting=SimpleNamespace(
            family_name=family.family_name,
            link_name=type(family.link).__qualname__,
            family_execution_static_config=family.execution_static_config(),
            family_parameter_snapshot=snapshot,
        ),
    )
    lineage = FamilyExecutionLineage.from_prepared(prepared, family)
    family.put_theta(np.log(np.asarray([3.0])))
    with pytest.raises(
        RuntimeError, match=r"(static configuration|parameter state) changed"
    ):
        lineage.validate(prepared, family)


def test_dynamic_theta_nb_initial_working_system_is_deferred_to_pr75() -> None:
    """Do not combine static NB variance with an explicit trial-theta leaf."""
    family = NegativeBinomial(theta=2.0, fixed=False)
    context = FamilyExecutionContext.from_family(family)
    with pytest.raises(NotImplementedError, match=r"PR7\.5"):
        batch_initial_working_quantities(
            jnp.asarray([1.0]),
            jnp.ones(1),
            jnp.zeros(1),
            jnp.ones(1, dtype=bool),
            jnp.zeros(1),
            FamilyExecutionParameters.from_snapshot(
                family.execution_parameter_snapshot()
            ),
            family,
            context,
        )


def test_unclipped_second_derivative_preserves_valid_inverse_and_gamma_domains() -> (
    None
):
    """The opt-in raw hooks must not inherit legacy positive-domain clipping."""
    inverse = InverseLink()
    np.testing.assert_allclose(inverse.second_derivative(np.array([-2.0])), [-0.25])
    np.testing.assert_allclose(
        inverse.second_derivative(np.array([-1e-12])), [-2e36], rtol=MODERATE.rtol
    )
    log = LogLink()
    np.testing.assert_allclose(log.second_derivative(np.array([1e-12])), [-1e24])


@pytest.mark.parametrize(
    ("family", "y", "eta", "expected_alpha"),
    [
        (Gaussian(InverseLink()), -1.0, -0.5, 2.0),
        (Gamma(LogLink()), 2e-12, np.log(1e-12), 2.0),
    ],
)
def test_initial_raw_alpha_uses_unclipped_second_link_derivative(
    family, y, eta, expected_alpha
) -> None:
    """Valid negative/tiny-positive domains retain the pinned-R arithmetic."""
    result = batch_initial_working_quantities(
        jnp.asarray([y]),
        jnp.ones(1),
        jnp.zeros(1),
        jnp.ones(1, dtype=bool),
        jnp.asarray([eta]),
        FamilyExecutionParameters.from_snapshot(family.execution_parameter_snapshot()),
        family,
        FamilyExecutionContext.from_family(family),
    )
    assert bool(result.working_system_admissible)
    np.testing.assert_allclose(
        result.newton_alpha_raw, [expected_alpha], rtol=MODERATE.rtol, atol=0.0
    )


def test_binomial_log_reports_unresolved_near_zero_alpha_without_rewriting() -> None:
    """Keep raw alpha/sign and fail the operational parity gate near cancellation."""
    family = Binomial(LogLink())
    y = jnp.asarray([0.0, 1.0, 0.0])
    weight = jnp.asarray([1.0, 0.8, 1.3])
    state = family.initial_working_state_cpu(
        np.asarray(y), np.asarray(weight), np.ones(3, bool)
    )
    result = batch_initial_working_quantities(
        y,
        weight,
        jnp.asarray([0.1, -0.1, 0.05]),
        jnp.ones(3, dtype=bool),
        jnp.asarray(state.eta),
        FamilyExecutionParameters.from_snapshot(family.execution_parameter_snapshot()),
        family,
        FamilyExecutionContext.from_family(family),
    )
    assert bool(result.input_ok)
    assert bool(result.domain_ok)
    assert bool(result.alpha_resolution_unresolved[1])
    assert not bool(result.working_system_admissible)
    assert np.asarray(result.newton_alpha_raw)[1] != np.finfo(float).eps
    np.testing.assert_allclose(
        result.newton_alpha, result.newton_alpha_raw, rtol=STRICT.rtol, atol=STRICT.atol
    )

    nextafter = batch_initial_working_quantities(
        jnp.asarray([0.0, np.nextafter(1.0, 0.0), 0.0]),
        weight,
        jnp.asarray([0.1, -0.1, 0.05]),
        jnp.ones(3, dtype=bool),
        jnp.asarray(state.eta),
        FamilyExecutionParameters.from_snapshot(family.execution_parameter_snapshot()),
        family,
        FamilyExecutionContext.from_family(family),
    )
    assert bool(nextafter.alpha_resolution_unresolved[1])
    assert not bool(nextafter.working_system_admissible)
    assert np.asarray(nextafter.newton_alpha_raw)[1] > 0.0

    far = batch_initial_working_quantities(
        jnp.asarray([0.0, 1.0 - 1e-12, 0.0]),
        weight,
        jnp.asarray([0.1, -0.1, 0.05]),
        jnp.ones(3, dtype=bool),
        jnp.asarray(state.eta),
        FamilyExecutionParameters.from_snapshot(family.execution_parameter_snapshot()),
        family,
        FamilyExecutionContext.from_family(family),
    )
    assert not bool(far.alpha_resolution_unresolved[1])
    assert bool(far.working_system_admissible)


@pytest.mark.parametrize(
    "family",
    [
        Gaussian(IdentityLink()),
        Gaussian(LogLink()),
        Gaussian(InverseLink()),
        Binomial(LogitLink()),
        Binomial(ProbitLink()),
        Binomial(CloglogLink()),
        Binomial(LogLink()),
        Poisson(LogLink()),
        Poisson(IdentityLink()),
        Poisson(SqrtLink()),
        Gamma(InverseLink()),
        Gamma(LogLink()),
        Gamma(IdentityLink()),
    ],
)
def test_initial_working_kernel_jits_and_preserves_raw_systems(family) -> None:
    state = family.initial_working_state_cpu(
        np.array([0.3, 0.8]) if isinstance(family, Binomial) else np.array([0.5, 1.4]),
        np.array([1.0, 0.0]),
        np.array([True, True]),
    )
    context = FamilyExecutionContext.from_family(family)
    parameters = FamilyExecutionParameters.from_snapshot(
        family.execution_parameter_snapshot()
    )
    result = jax.jit(
        lambda eta: batch_initial_working_quantities(
            jnp.asarray([0.3, 0.8])
            if isinstance(family, Binomial)
            else jnp.asarray([0.5, 1.4]),
            jnp.asarray([1.0, 0.0]),
            jnp.asarray([0.1, -0.2]),
            jnp.asarray([True, True]),
            eta,
            parameters,
            family,
            context,
        )
    )(jnp.asarray(state.eta))
    assert bool(result.domain_ok)
    assert bool(result.working_system_admissible)
    for value in (
        result.fisher_weight,
        result.fisher_response,
        result.observed_weight,
        result.newton_alpha,
        result.newton_weight,
        result.newton_response,
    ):
        assert np.all(np.isfinite(np.asarray(value)))
    # At an exact first residual, the operational alpha system and smooth
    # observed curvature agree without an abs/positive-weight policy.
    exact = batch_initial_working_quantities(
        jnp.asarray(result.mu),
        jnp.asarray([1.0, 1.0]),
        jnp.zeros(2),
        jnp.asarray([True, True]),
        result.eta,
        parameters,
        family,
        context,
    )
    np.testing.assert_allclose(
        exact.observed_weight,
        exact.newton_weight,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


class _PositiveEtaStartGaussian(Gaussian):
    """Makes a predictor-domain failure shrinkable without an input error."""

    family_name = "positive_eta_start_gaussian"

    def execution_initial_mustart_cpu(self, y, prior_weight):  # noqa: ARG002
        return -np.ones_like(y)

    def valid_eta(self, eta):
        return np.asarray(eta) > 0.0


def _selection_fixture() -> tuple[StreamDesign, _PositiveEtaStartGaussian, object]:
    source = ArrayRowSource(
        {"x": np.linspace(-0.3, 0.4, 7)}, y=np.linspace(0.4, 1.1, 7)
    )
    family = _PositiveEtaStartGaussian()
    prepared = prepare_model(parse_formula("y ~ x"), source, family=family)
    stream = StreamDesign(prepared, source)
    return stream, family, FamilyExecutionLineage.from_prepared(prepared, family)


def test_host_selection_replays_global_r_shrink_without_retaining_rows() -> None:
    stream, family, lineage = _selection_fixture()
    selected = select_initial_working_state_cpu(
        stream,
        family,
        lineage,
        batch_rows=3,
        null_eta_for_batch=lambda batch: np.ones_like(batch.weight),
    )
    assert selected.input_ok
    assert selected.domain_ok
    assert selected.shrink_count == 7
    assert selected.source_scans == 8
    assert selected.valid_rows == stream.prepared.n_obs


def test_host_selection_rejects_source_mutation_during_replay() -> None:
    stream, family, lineage = _selection_fixture()

    def mutate_source(batch):
        stream.source._y[0] += 0.1  # type: ignore[attr-defined]
        return np.ones_like(batch.weight)

    with pytest.raises(RuntimeError, match="RowSource changed"):
        select_initial_working_state_cpu(
            stream,
            family,
            lineage,
            batch_rows=3,
            null_eta_for_batch=mutate_source,
        )


def test_host_selection_rejects_family_mutation_before_replay() -> None:
    stream, family, lineage = _selection_fixture()
    family.link.runtime_marker = "mutated"  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="static configuration changed"):
        select_initial_working_state_cpu(
            stream,
            family,
            lineage,
            batch_rows=3,
            null_eta_for_batch=lambda batch: np.ones_like(batch.weight),
        )


def test_host_selection_returns_fail_closed_after_r_shrink_bound() -> None:
    stream, family, lineage = _selection_fixture()
    selected = select_initial_working_state_cpu(
        stream,
        family,
        lineage,
        batch_rows=4,
        null_eta_for_batch=lambda batch: -np.ones_like(batch.weight),
    )
    assert selected.input_ok
    assert not selected.domain_ok
    assert selected.shrink_count == 20
    assert selected.source_scans == 21


_R_FIRST_ITERATION_CASES = (
    ("gaussian_identity", Gaussian(IdentityLink()), "regular"),
    ("gaussian_log", Gaussian(LogLink()), "regular"),
    ("gaussian_inverse", Gaussian(InverseLink()), "regular"),
    ("binomial_logit", Binomial(LogitLink()), "binomial"),
    ("binomial_probit", Binomial(ProbitLink()), "binomial"),
    ("binomial_cloglog", Binomial(CloglogLink()), "binomial"),
    ("binomial_log", Binomial(LogLink()), "binomial"),
    ("poisson_log", Poisson(LogLink()), "regular"),
    ("poisson_identity", Poisson(IdentityLink()), "regular"),
    ("poisson_sqrt", Poisson(SqrtLink()), "regular"),
    ("gamma_inverse", Gamma(InverseLink()), "regular"),
    ("gamma_log", Gamma(LogLink()), "regular"),
    ("gamma_identity", Gamma(IdentityLink()), "regular"),
)


def _pinned_first_iteration_oracle() -> dict[str, dict[str, np.ndarray]]:
    """Capture raw first W/z at gam.fit3's pre-PLS trace point once."""
    script = r"""
        suppressPackageStartupMessages(library(mgcv))
        run_one <- function(tag, fam, y) {
          f <- mgcv:::fix.family.var(mgcv:::fix.family.link(fam))
          capture <- new.env(parent=emptyenv())
          assign(".jaxgam_capture", capture, envir=.GlobalEnv)
          suppressMessages(trace(
            mgcv:::gam.fit3, at=list(c(40, 4, 12, 4, 20)),
            tracer=quote({
              .jaxgam_capture$eta <- eta
              .jaxgam_capture$mu <- mu
              .jaxgam_capture$w <- w
              .jaxgam_capture$z <- z
            }), print=FALSE
          ))
          control <- gam.control(); control$maxit <- 1
          ignored <- try(mgcv:::gam.fit3(
            diag(3), y, numeric(), matrix(0, 3, 3), UrS=list(),
            weights=c(1, .8, 1.3), offset=c(.1, -.1, .05), U1=diag(3),
            Mp=0, family=f, control=control, deriv=0, scale=1,
            scoreType="GCV.Cp", null.coef=rep(1, 3)
          ), silent=TRUE)
          suppressMessages(untrace(mgcv:::gam.fit3))
          if (!all(c("eta", "mu", "w", "z") %in% ls(capture)))
            stop("gam.fit3 did not reach initial C_pls_fit1 for ", tag)
          fields <- lapply(c("eta", "mu", "w", "z"), function(key)
            paste(sprintf("%.17g", capture[[key]]), collapse=","))
          cat(paste(c("CELL", tag, unlist(fields)), collapse="|"), "\n")
          rm(.jaxgam_capture, envir=.GlobalEnv)
        }
        run_one("gaussian_identity", gaussian("identity"), c(1.2, 1.7, 2.1))
        run_one("gaussian_log", gaussian("log"), c(1.2, 1.7, 2.1))
        run_one("gaussian_inverse", gaussian("inverse"), c(1.2, 1.7, 2.1))
        run_one("binomial_logit", binomial("logit"), c(0, 1, 0))
        run_one("binomial_probit", binomial("probit"), c(0, 1, 0))
        run_one("binomial_cloglog", binomial("cloglog"), c(0, 1, 0))
        run_one("binomial_log", binomial("log"), c(0, 1, 0))
        run_one("poisson_log", poisson("log"), c(1.2, 1.7, 2.1))
        run_one("poisson_identity", poisson("identity"), c(1.2, 1.7, 2.1))
        run_one("poisson_sqrt", poisson("sqrt"), c(1.2, 1.7, 2.1))
        run_one("gamma_inverse", Gamma("inverse"), c(1.2, 1.7, 2.1))
        run_one("gamma_log", Gamma("log"), c(1.2, 1.7, 2.1))
        run_one("gamma_identity", Gamma("identity"), c(1.2, 1.7, 2.1))
    """
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, dict[str, np.ndarray]] = {}
    for line in completed.stdout.splitlines():
        if not line.startswith("CELL|"):
            continue
        _, name, eta, mu, weight, z = line.split("|")
        result[name] = {
            "eta": np.fromstring(eta, sep=","),
            "mu": np.fromstring(mu, sep=","),
            "weight": np.fromstring(weight, sep=","),
            "z": np.fromstring(z, sep=","),
        }
    if len(result) != len(_R_FIRST_ITERATION_CASES):
        raise AssertionError(
            f"Missing pinned first-iteration cells: {completed.stderr}"
        )
    return result


def _assert_r_vector(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=MODERATE.rtol, atol=MODERATE.atol
    )


@pytest.mark.skipif(not r_available(), reason="requires pinned R 4.5.2 + mgcv 1.9-3")
def test_first_working_kernel_matches_pinned_gam_fit3_all_regular_links() -> None:
    """Compare all 13 initial eta/mu and every admissible raw W/z to pinned R."""
    oracle = _pinned_first_iteration_oracle()
    weight = jnp.asarray([1.0, 0.8, 1.3])
    offset = jnp.asarray([0.1, -0.1, 0.05])
    valid = jnp.ones(3, dtype=bool)
    collector = _AssertCollector()
    for name, family, response_kind in _R_FIRST_ITERATION_CASES:
        response = [0.0, 1.0, 0.0] if response_kind == "binomial" else [1.2, 1.7, 2.1]
        y = jnp.asarray(response)
        state = family.initial_working_state_cpu(
            np.asarray(y), np.asarray(weight), np.ones(3, bool)
        )
        result = batch_initial_working_quantities(
            y,
            weight,
            offset,
            valid,
            jnp.asarray(state.eta),
            FamilyExecutionParameters.from_snapshot(
                family.execution_parameter_snapshot()
            ),
            family,
            FamilyExecutionContext.from_family(family),
        )
        expected = oracle[name]
        selected_weight = (
            result.fisher_weight if family.is_canonical else result.newton_weight
        )
        selected_z = (
            result.fisher_response if family.is_canonical else result.newton_response
        )
        collector.check(
            f"{name}: eta/mu",
            lambda result=result, expected=expected: _assert_r_vector(
                result.eta, expected["eta"]
            ),
        )
        collector.check(
            f"{name}: mu",
            lambda result=result, expected=expected: _assert_r_vector(
                result.mu, expected["mu"]
            ),
        )
        if name == "binomial_log":
            # This cell is an explicit future 7.4 compatibility gate: pinned
            # R and compiled JAX differ by ulps around alpha=0.  Eta/mu and
            # raw systems remain available for diagnosis, but cannot be sent
            # to an observed Newton solver as R-validated values.
            assert bool(result.alpha_resolution_unresolved[1])
            assert not bool(result.working_system_admissible)
            admissible_rows = ~np.asarray(result.alpha_resolution_unresolved)
            collector.check(
                f"{name}: admissible raw weight",
                partial(
                    _assert_r_vector,
                    np.asarray(selected_weight)[admissible_rows],
                    expected["weight"][admissible_rows],
                ),
            )
            collector.check(
                f"{name}: admissible raw z",
                partial(
                    _assert_r_vector,
                    np.asarray(selected_z)[admissible_rows],
                    expected["z"][admissible_rows],
                ),
            )
        else:
            collector.check(
                f"{name}: raw weight",
                lambda selected_weight=selected_weight, expected=expected: (
                    _assert_r_vector(selected_weight, expected["weight"])
                ),
            )
            collector.check(
                f"{name}: raw z",
                lambda selected_z=selected_z, expected=expected: _assert_r_vector(
                    selected_z, expected["z"]
                ),
            )
    collector.raise_if_any("pinned gam.fit3 first-working oracle")
